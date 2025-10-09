# /bot/main.py
import os
import time
import math
import json
import queue
import threading
import http.server
import socketserver
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd

# --- Telegram (v13) ---
from telegram import Bot, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext
from telegram.utils.request import Request
import telegram.error as tgerr


# =========================
#        CONFIG
# =========================
ENV = os.getenv
TELEGRAM_TOKEN = ENV("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = int(ENV("TELEGRAM_CHAT_ID", "0"))

# режим
MODE = ENV("MODE", "DEMO").upper()
DEMO_MODE = ENV("DEMO_MODE", "true").lower() == "true"
DRY_RUN = ENV("DRY_RUN", "false").lower() == "true"  # если захочешь не торговать даже в демо

# пары и таймфрейм
PAIRS = [s.strip() for s in ENV("PAIRS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
TIMEFRAME = ENV("TIMEFRAME", "5m")  # 1m,3m,5m,15m,1h,4h
TICK_SEC = int(ENV("TICK", "10"))

# риски и комиссия
FEE_PCT = float(ENV("FEE_PCT", "0.0006"))        # 0.06%
LEVERAGE = float(ENV("LEVERAGE", "5"))
RISK_PCT = float(ENV("RISK_PCT", "1.0"))         # риск на сделку от equity (в %)
TP_PCT = float(ENV("TP_PCT", "0.35"))/100.0      # TP в %
ATR_LEN = int(ENV("ATR_LEN", "14"))
ATR_MULT_SL = float(ENV("ATR_MULT_SL", "1.0"))
EMA_LEN = int(ENV("EMA_LEN", "100"))
EMA_SLOPE_BARS = int(ENV("EMA_SLOPE_BARS", "8"))
COOLDOWN_SEC = int(ENV("COOLDOWN_SEC", "180"))
DAILY_SUMMARY = ENV("DAILY_SUMMARY", "1") == "1"

DEMO_START_BALANCE = float(ENV("DEMO_START_BALANCE", "5000"))
TZ = ENV("TX", "UTC").upper()
TZINFO = timezone.utc if TZ == "UTC" else timezone.utc

PORT = int(os.getenv("PORT", "8080"))

# =========================
#       UTIL / LOG
# =========================
def utcnow():
    return datetime.now(timezone.utc)

def log(msg, level="INFO"):
    t = utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} | {level} | {msg}", flush=True)

# =========================
#   DATA: Binance KLN
# =========================
# Для демо берём котировки с Binance — без ключа, стабильно.
# https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=500
BINANCE = "https://api.binance.com"

def get_klines_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        raise ValueError("Empty klines")
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ("open","high","low","close","volume"):
        df[c] = df[c].astype(float)
    return df[["open_time","open","high","low","close","volume"]]

# =========================
#   INDICATORS & SIGNALS
# =========================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def slope(series: pd.Series, bars: int) -> float:
    if len(series) < bars+1: return 0.0
    y = series.iloc[-bars:].values
    x = np.arange(bars)
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    b = np.polyfit(x, y, 1)[0]
    return float(b)

def is_hammer(row) -> bool:
    o,h,l,c = row.open,row.high,row.low,row.close
    body = abs(c-o)
    rng = h-l
    low_tail = o-l if c>=o else c-l
    return rng>0 and body/rng<0.3 and low_tail/rng>0.5 and c>o

def is_bull_engulf(prev, cur) -> bool:
    return (cur.close>cur.open) and (prev.close<prev.open) and (cur.close>=prev.open) and (cur.open<=prev.close)

def is_bear_engulf(prev, cur) -> bool:
    return (cur.close<cur.open) and (prev.close>prev.open) and (cur.open>=prev.close) and (cur.close<=prev.open)

def breakout_up(df: pd.DataFrame, lookback=20) -> bool:
    if len(df)<lookback+1: return False
    hh = df["high"].iloc[-(lookback+1):-1].max()
    return df["close"].iloc[-1] > hh

def breakout_down(df: pd.DataFrame, lookback=20) -> bool:
    if len(df)<lookback+1: return False
    ll = df["low"].iloc[-(lookback+1):-1].min()
    return df["close"].iloc[-1] < ll

def build_signal(df: pd.DataFrame) -> Tuple[str, Dict]:
    """Возвращает ('LONG'|'SHORT'|'' , meta)"""
    if len(df) < max(EMA_LEN, ATR_LEN) + 5:
        return "", {}
    c = df["close"]
    df = df.copy()
    df["ema"] = ema(c, EMA_LEN)
    df["atr"] = atr(df, ATR_LEN)
    slp = slope(df["ema"], EMA_SLOPE_BARS)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    meta = {"slope": round(slp,5), "ATR": round(last["atr"],5)}

    # Паттерны
    hammer_ok = is_hammer(last)
    bull_eng = is_bull_engulf(prev, last)
    bear_eng = is_bear_engulf(prev, last)
    brk_up = breakout_up(df)
    brk_dn = breakout_down(df)

    # Простая логика (консервативнее)
    # long: up-slope + breakout_up OR bull/hammer с up-slope
    if slp>0 and (brk_up or bull_eng or hammer_ok):
        meta["signal"] = ("breakout_up" if brk_up else "bull_engulf" if bull_eng else "hammer")
        return "LONG", meta

    # short: down-slope + breakout_down OR bear_engulf c down-slope
    if slp<0 and (brk_dn or bear_eng):
        meta["signal"] = ("breakout_down" if brk_dn else "bear_engulf")
        return "SHORT", meta

    meta["signal"] = "—"
    return "", meta

# =========================
#   DEMO BROKER
# =========================
@dataclass
class Position:
    side: str        # LONG | SHORT
    entry: float
    qty: float
    tp: float
    sl: float
    open_time: datetime

@dataclass
class PairState:
    last_signal_time: float = 0.0
    pos: Optional[Position] = None
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0  # реализованный PnL

@dataclass
class GlobalState:
    equity: float = DEMO_START_BALANCE
    start_equity: float = DEMO_START_BALANCE
    per_pair: Dict[str, PairState] = field(default_factory=dict)
    day_anchor: datetime.date = field(default_factory=lambda: utcnow().date())

STATE = GlobalState()
for p in PAIRS:
    STATE.per_pair[p] = PairState()

def notional(symbol: str, price: float) -> float:
    # риск в $ на сделку
    risk_amount = STATE.equity * (RISK_PCT/100.0)
    # рассчитываем qty через стоп (ATR SL)
    # здесь берём SL от ATR_MULT_SL * ATR (подставим извне)
    # для простоты вернём qty позже, когда знаем ATR
    return risk_amount

def format_money(x: float) -> str:
    return f"{x:.5f}".rstrip("0").rstrip(".")

def calc_upnl(sym: str, price: float) -> float:
    st = STATE.per_pair[sym]
    if not st.pos: return 0.0
    pos = st.pos
    if pos.side == "LONG":
        pl = (price - pos.entry) * pos.qty
    else:
        pl = (pos.entry - price) * pos.qty
    fee = price * pos.qty * FEE_PCT  # приблизительно
    return pl - fee

# =========================
#  TELEGRAM NOTIFICATIONS
# =========================
def send(bot: Bot, text: str):
    if TELEGRAM_CHAT_ID:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            log(f"Telegram send err: {e}", "WARN")

def box(title: str, lines: List[str]) -> str:
    return "📊 *"+title+"*\n" + "\n".join(lines)

def compose_status() -> str:
    lines = []
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    upnl_total = 0.0

    for s in PAIRS:
        st = STATE.per_pair[s]
        total_trades += st.trades
        total_wins += st.wins
        total_pnl += st.pnl

        # текущая цена
        try:
            df = get_klines_binance(s, TIMEFRAME, limit=2)
            price = float(df["close"].iloc[-1])
        except Exception:
            price = 0.0

        upnl = calc_upnl(s, price)
        upnl_total += upnl

        pos_line = "—"
        if st.pos:
            p = st.pos
            pos_line = f"{p.side} @ {format_money(p.entry)} (TP {format_money(p.tp)} / SL {format_money(p.sl)})"

        lines += [
            f"*{s}* • trades: {st.trades}  WR: {0 if st.trades==0 else round(100*st.wins/st.trades,2)}%  "
            f"PnL: {format_money(st.pnl)}",
            f"{format_money(price)}  pos: {pos_line}",
            "—"
        ]

    wr_total = 0 if total_trades==0 else round(100*total_wins/total_trades,2)
    equity = STATE.equity + upnl_total
    lines += [
        f"*TOTAL* • trades: {total_trades}  WR: {wr_total}%  PnL: {format_money(total_pnl)}",
        f"equity: {format_money(equity)}  ({round(100*(equity-STATE.start_equity)/STATE.start_equity,2)}% с начала)",
        f"leverage: {LEVERAGE:.1f}x  fee: {round(FEE_PCT*100,3)}%"
    ]
    return box(f"STATUS {utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", lines)

# =========================
#        ENGINE
# =========================
def place_order(sym: str, side: str, price: float, atr_val: float, meta_sig: Dict, bot: Bot):
    st = STATE.per_pair[sym]
    if st.pos: 
        return  # одна позиция на пару

    # anti-spam cooldown
    now = time.time()
    if now - st.last_signal_time < COOLDOWN_SEC:
        return
    st.last_signal_time = now

    # риск и размер
    risk_usd = notional(sym, price)
    sl_dist = max(0.0001, ATR_MULT_SL * atr_val)
    if side == "LONG":
        sl = price - sl_dist
        tp = price * (1 + TP_PCT)
    else:
        sl = price + sl_dist
        tp = price * (1 - TP_PCT)
    # риск на доллар: движение до SL
    per_unit_loss = abs(price - sl)
    qty = max(0.0, risk_usd / per_unit_loss)
    qty = qty / price * LEVERAGE  # контрактов по USDT
    qty = max(qty, 0.0)

    if qty <= 0:
        return

    if DRY_RUN:
        pass

    st.pos = Position(side=side, entry=price, qty=qty, tp=tp, sl=sl, open_time=utcnow())

    text = (
        f"🔴 *OPEN {sym} {side}*\n"
        f"• time: {utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"• entry: {format_money(price)}\n"
        f"• qty: {format_money(qty)}\n"
        f"• TP: {format_money(tp)}   SL: {format_money(sl)}\n"
        f"• signal: {meta_sig.get('signal','')}, slope {meta_sig.get('slope',0)}, ATR {meta_sig.get('ATR',0)}\n"
        f"• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
    )
    send(bot, text)

def maybe_close(sym: str, price: float, bot: Bot):
    st = STATE.per_pair[sym]
    if not st.pos: 
        return
    p = st.pos
    hit_tp = (price >= p.tp) if p.side=="LONG" else (price <= p.tp)
    hit_sl = (price <= p.sl) if p.side=="LONG" else (price >= p.sl)

    if hit_tp or hit_sl:
        # реализация pnl
        if p.side=="LONG":
            gross = (price - p.entry) * p.qty
        else:
            gross = (p.entry - price) * p.qty
        fee = (p.entry + price) * p.qty * FEE_PCT
        pnl = gross - fee
        STATE.equity += pnl
        st.pnl += pnl
        st.trades += 1
        if pnl > 0:
            st.wins += 1

        icon = "✅" if pnl>0 else "❌"
        reason = "TP" if hit_tp else "SL"
        text = (
            f"{icon} *CLOSE {sym} ({reason})*\n"
            f"• time: {utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"• exit: {format_money(price)}\n"
            f"• PnL: {format_money(pnl)}\n"
            f"• pair stats: trades {st.trades}, WR {0 if st.trades==0 else round(100*st.wins/st.trades,2)}%, PnL {format_money(st.pnl)}\n"
            f"• total: trades {sum(ps.trades for ps in STATE.per_pair.values())}, "
            f"WR {calc_wr_total()}%, PnL {format_money(sum(ps.pnl for ps in STATE.per_pair.values()))}\n"
            f"• balance: {format_money(STATE.equity)}  (Δ {format_money(STATE.equity-STATE.start_equity)} | "
            f"{round(100*(STATE.equity-STATE.start_equity)/STATE.start_equity,2)}%)\n"
            f"• since start: {round(100*(STATE.equity-STATE.start_equity)/STATE.start_equity,2)}%   "
            f"(lev {LEVERAGE:.1f}x, fee {round(FEE_PCT*100,3)}%)"
        )
        send(bot, text)
        st.pos = None

def calc_wr_total() -> float:
    t = sum(ps.trades for ps in STATE.per_pair.values())
    w = sum(ps.wins for ps in STATE.per_pair.values())
    return 0.0 if t==0 else round(100*w/t,2)

def pair_loop(sym: str, bot: Bot, stop_ev: threading.Event):
    log(f"Loop started for {sym}")
    while not stop_ev.is_set():
        try:
            df = get_klines_binance(sym, TIMEFRAME, limit=max(200, EMA_LEN+ATR_LEN+5))
            sig, meta = build_signal(df)
            price = float(df["close"].iloc[-1])
            atr_val = float(atr(df, ATR_LEN).iloc[-1])
            maybe_close(sym, price, bot)
            if sig:
                place_order(sym, sig, price, atr_val, meta, bot)

            # ежедневная инициализация/дайджест
            if DAILY_SUMMARY and STATE.day_anchor != utcnow().date():
                STATE.day_anchor = utcnow().date()
                send(bot, compose_status())

        except requests.RequestException as e:
            log(f"Data error {sym}: {e}", "WARN")
        except Exception as e:
            log(f"{sym} loop error: {e}", "WARN")
        time.sleep(TICK_SEC)

# =========================
#     TELEGRAM HANDLERS
# =========================
def cmd_start(update, ctx: CallbackContext):
    update.message.reply_text("Бот запущен. Команды: /status /pause /resume /panic_close")

def cmd_status(update, ctx: CallbackContext):
    update.message.reply_text(compose_status(), parse_mode=ParseMode.MARKDOWN)

PAUSED = threading.Event()  # не используется в этой версии для торговли, оставлено на будущее

def cmd_pause(update, ctx):
    PAUSED.set()
    update.message.reply_text("⏸️ Пауза.")

def cmd_resume(update, ctx):
    PAUSED.clear()
    update.message.reply_text("▶️ Продолжаю.")

def cmd_panic(update, ctx):
    # мгновенно закрыть все
    for s, st in STATE.per_pair.items():
        if st.pos:
            price = st.pos.entry  # закроем по entry как по рынку для демо (или можно подтянуть последнюю цену)
            maybe_close(s, price, ctx.bot)
            st.pos = None
    update.message.reply_text("🛑 Все позиции закрыты (panic).")

# =========================
#       HEALTH SERVER
# =========================
def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args): pass
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), Quiet) as httpd:
            httpd.serve_forever()
    except OSError as e:
        log(f"Health server bind skipped: {e}", "WARN")

# =========================
#          MAIN
# =========================
def start_bot() -> Tuple[Updater, List[threading.Thread], threading.Event]:
    req = Request(con_pool_size=8, connect_timeout=5, read_timeout=35)
    bot = Bot(token=TELEGRAM_TOKEN, request=req)

    # гарантированно гасим webhook перед polling
    try:
        bot.delete_webhook(drop_pending_updates=True)
        time.sleep(0.4)
    except Exception as e:
        log(f"Webhook delete warn: {e}", "WARN")

    updater = Updater(bot=bot, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("pause", cmd_pause))
    dp.add_handler(CommandHandler("resume", cmd_resume))
    dp.add_handler(CommandHandler("panic_close", cmd_panic))

    # запускаем polling (без устаревшего clean=)
    updater.start_polling(drop_pending_updates=True, timeout=35)

    # воркеры по парам
    stop_ev = threading.Event()
    threads = []
    for p in PAIRS:
        th = threading.Thread(target=pair_loop, args=(p, bot, stop_ev), daemon=True)
        th.start()
        threads.append(th)
        send(bot, f"✅ Loop started for *{p}*")

    # привет
    send(bot,
         f"🤖 *mybot9* started successfully!\n"
         f"Mode: {'DEMO' if DEMO_MODE else MODE} | Leverage {LEVERAGE:.1f}x | Fee {round(FEE_PCT*100,3)}% |\n"
         f"Risk {RISK_PCT:.1f}%\n"
         f"Pairs: {', '.join(PAIRS)} | TF *{TIMEFRAME}* | Tick *{TICK_SEC}s*\n"
         f"Balance: {format_money(STATE.equity)}  USDT")

    return updater, threads, stop_ev

def calc_env_preview():
    return (f"Mode {MODE} demo={DEMO_MODE} dry={DRY_RUN}  pairs={PAIRS} tf={TIMEFRAME}  "
            f"risk={RISK_PCT}% tp={TP_PCT*100:.3f}% atr_len={ATR_LEN} atr_mult={ATR_MULT_SL} ema={EMA_LEN}/{EMA_SLOPE_BARS}")

def main():
    log(calc_env_preview())
    # health server
    threading.Thread(target=serve, daemon=True).start()

    try:
        updater, threads, stop_ev = start_bot()
        # держим процесс
        while True:
            time.sleep(5)
    except tgerr.Conflict as e:
        log(f"Telegram Conflict: {e}. Tip: ensure deleteWebhook and single process.", "WARN")
        time.sleep(3)
    except Exception as e:
        log(f"Fatal: {e}", "ERROR")
        time.sleep(2)

if __name__ == "__main__":
    main()
