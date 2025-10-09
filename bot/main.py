# bot/main.py
# ==========================================
# Strategy #9 (test preset) – WR-focused
# PTB 13.15 (без job-queue), requests/pandas/numpy
# Сообщения в телеграме: ТОЛЬКО в моменты открытия/закрытия + /status
# Демо-исполнение на живых котировках (Binance futures k-lines)
# ==========================================

import os
import time
import json
import math
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
from telegram import ParseMode, Bot
from telegram.ext import Updater, CommandHandler

# ---------- ENV & Defaults (Strategy #9 test preset) ----------
ENV = os.getenv

TOKEN = ENV("TELEGRAM_TOKEN", "")  # ОБЯЗАТЕЛЬНО
CHAT_ID = ENV("TELEGRAM_CHAT_ID", "")  # ОБЯЗАТЕЛЬНО

PAIRS = [s.strip().upper() for s in ENV("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",") if s.strip()]
TF = ENV("TIMEFRAME", "1m")
TICK_SECONDS = int(float(ENV("TICK_SECONDS", "10")))  # частота опроса
LIMIT_KLINES = int(ENV("LIMIT_KLINES", "500"))

MODE = ENV("MODE", "DEMO").upper()  # только демо сообщения, без реальных ордеров
FEE_PCT = float(ENV("FEE_PCT", "0.06")) / 100.0  # комиссия в процентах
LEVERAGE = float(ENV("LEVERAGE", "3.0"))

# ===== Strategy #9 — ключевые параметры «как в тесте» =====
# 6 паттернов, небольшой TP, SL шире, брейк-ивен и мягкий трейлинг, без тренд-фильтра
USE_TREND_FILTER = int(ENV("USE_TREND_FILTER", "0"))   # 0 — как в тесте
EMA_PERIOD = int(ENV("EMA_PERIOD", "48"))
MIN_SLOPE = float(ENV("MIN_SLOPE", "0.0"))             # без отсечки по уклону

ATR_PERIOD = int(ENV("ATR_PERIOD", "14"))
ATR_MULT_TP = float(ENV("ATR_MULT_TP", "0.60"))        # близкая цель
ATR_MULT_SL = float(ENV("ATR_MULT_SL", "1.20"))        # SL шире TP

BREAKEVEN_ATR = float(ENV("BREAKEVEN_ATR", "0.40"))    # перенос SL в 0
BREAKEVEN_OFFSET_ATR = float(ENV("BREAKEVEN_OFFSET_ATR", "0.10"))
TRAIL_ATR = float(ENV("TRAIL_ATR", "0.50"))            # мягкий трейлинг
ENTRY_BUFFER_ATR = float(ENV("ENTRY_BUFFER_ATR", "0.10"))

RISK_PCT = float(ENV("RISK_PCT", "0.35")) / 100.0      # риск на сделку от капитала
MAX_OPEN_POS = int(ENV("MAX_OPEN_POS", "1"))
COOLDOWN_SEC = int(ENV("COOLDOWN_SEC", "180"))
MAX_CONSEC_LOSSES = int(ENV("MAX_CONSEC_LOSSES", "4"))
DAILY_MAX_DD_PCT = float(ENV("DAILY_MAX_DD_PCT", "3.0")) / 100.0

# стартовый капитал демо
START_EQUITY = float(ENV("START_EQUITY", "1000"))

# шесть паттернов из теста
PATTERNS = [s.strip() for s in ENV(
    "PATTERNS",
    "bull_engulf,bear_engulf,hammer,shooting,breakout_up,breakout_down"
).split(",") if s.strip()]

BINANCE_URL = "https://api.binance.com/api/v3/klines"

# ---------- State ----------
lock = threading.Lock()

state = {
    "equity": START_EQUITY,
    "equity_high": START_EQUITY,
    "start_equity": START_EQUITY,
    "opened": {},  # pair -> position dict
    "stats": defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0}),
    "total_trades": 0,
    "total_wins": 0,
    "total_pnl": 0.0,
    "last_open_time": defaultdict(lambda: 0.0),
    "consec_losses": 0,
    "day_anchor": datetime.utcnow().date(),
}

# ---------- Utils ----------
def now_utc():
    return datetime.now(timezone.utc)

def fmt(x, dec=5):
    return f"{x:.{dec}f}"

def pct(x):
    return f"{x*100:.2f}%"

def send(bot: Bot, text: str):
    if not TOKEN or not CHAT_ID:
        print(text)
        return
    bot.send_message(chat_id=CHAT_ID, text=text[:4096], parse_mode=ParseMode.MARKDOWN)

def get_klines(pair: str, interval: str = TF, limit: int = LIMIT_KLINES) -> pd.DataFrame:
    params = {"symbol": pair, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_URL, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    cols = ["t", "open", "high", "low", "close", "vol", "_1", "_2", "_3", "_4", "_5", "_6"]
    df = pd.DataFrame(raw, columns=cols)[["t", "open", "high", "low", "close", "vol"]]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA для тренд-уклона
    df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    # простой уклон ema за 5 баров
    df["ema_slope"] = df["ema"].diff(5)
    # ATR
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df

def pattern_signals(df: pd.DataFrame) -> dict:
    # свечные сигналы
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)

    bull = (c.shift(1) < o.shift(1)) & (c > o) & (c >= o.shift(1)) & (o <= c.shift(1))
    bear = (c.shift(1) > o.shift(1)) & (c < o) & (c <= o.shift(1)) & (o >= c.shift(1))

    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    hammer = (lower / rng > 0.6) & (upper / rng < 0.2)
    shooting = (upper / rng > 0.6) & (lower / rng < 0.2)

    # Breakout: закрытие выше/ниже последних N high/low
    N = 20
    breakout_up = c > df["high"].shift(1).rolling(N).max()
    breakout_dn = c < df["low"].shift(1).rolling(N).min()

    sig = dict(
        bull_engulf=bool(bull.iloc[-1]),
        bear_engulf=bool(bear.iloc[-1]),
        hammer=bool(hammer.iloc[-1]),
        shooting=bool(shooting.iloc[-1]),
        breakout_up=bool(breakout_up.iloc[-1]),
        breakout_down=bool(breakout_dn.iloc[-1]),
    )
    return sig

def allow_trend(df: pd.DataFrame, side: str) -> bool:
    if not USE_TREND_FILTER:
        return True
    slope = df["ema_slope"].iloc[-1]
    if side == "LONG":
        return slope >= MIN_SLOPE
    else:
        return slope <= -MIN_SLOPE

def pick_signal(sig: dict) -> str | None:
    # Strategy #9: любой из 6 сигналов подходит
    prio = [
        "breakout_up", "breakout_down",
        "bull_engulf", "bear_engulf",
        "hammer", "shooting"
    ]
    for k in prio:
        if k in PATTERNS and sig.get(k, False):
            return k
    return None

def side_from_signal(s: str) -> str:
    if s in ("breakout_up", "bull_engulf", "hammer"):
        return "LONG"
    if s in ("breakout_down", "bear_engulf", "shooting"):
        return "SHORT"
    return "LONG"

def lot_size(equity: float, atr: float, price: float) -> float:
    # риск от капитала, переводим в количество
    # риск в деньгах ~ (SL расстояние) * qty; SL ~ ATR_MULT_SL*ATR
    if atr <= 0 or price <= 0:
        return 0.0
    cash_risk = equity * RISK_PCT
    px_risk = ATR_MULT_SL * atr
    qty = cash_risk / px_risk
    # учтём плечо (капитал * левередж / цена – но мы уже по риску считаем,
    # чтобы не лезть в излишне большой размер)
    return max(qty, 0.0)

def be_and_trail(pos: dict, last: float, atr: float):
    # перенос в безубыток
    if not pos.get("be_done", False):
        if pos["side"] == "LONG":
            if last >= pos["entry"] + BREAKEVEN_ATR * atr:
                pos["sl"] = pos["entry"] + BREAKEVEN_OFFSET_ATR * atr
                pos["be_done"] = True
        else:
            if last <= pos["entry"] - BREAKEVEN_ATR * atr:
                pos["sl"] = pos["entry"] - BREAKEVEN_OFFSET_ATR * atr
                pos["be_done"] = True
    # мягкий трейлинг после BE
    if pos.get("be_done", False) and TRAIL_ATR > 0 and atr > 0:
        if pos["side"] == "LONG":
            trail = last - TRAIL_ATR * atr
            pos["sl"] = max(pos["sl"], trail)
        else:
            trail = last + TRAIL_ATR * atr
            pos["sl"] = min(pos["sl"], trail)

def fees_for(qty: float, entry: float, exit_px: float) -> float:
    gross = qty * (entry + exit_px)
    return gross * FEE_PCT

def open_text(pair, side, entry, qty, tp, sl, signal, slope, atr) -> str:
    t = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"🔴 *OPEN {pair} {side}*\n"
        f"• time: {t}\n"
        f"• entry: {fmt(entry, 5)}\n"
        f"• qty: {fmt(qty, 6)}\n"
        f"• TP: {fmt(tp, 5)}   SL: {fmt(sl, 5)}\n"
        f"• signal: {signal}, slope {fmt(slope, 5)}, ATR {fmt(atr, 5)}\n"
        f"• mode: {MODE}"
    )

def close_text(pair, side, exit_px, pnl, pstats, tstats, equity, delta_equity) -> str:
    t = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    sign = "✅" if pnl >= 0 else "❌"
    wr_pair = 0.0 if pstats["trades"] == 0 else pstats["wins"]/pstats["trades"]*100
    wr_tot = 0.0 if tstats["trades"] == 0 else tstats["wins"]/tstats["trades"]*100
    since = (equity/state["start_equity"] - 1.0) * 100.0
    return (
        f"{sign} *CLOSE {pair} ({side})*\n"
        f"• time: {t}\n"
        f"• exit: {fmt(exit_px, 5)}\n"
        f"• PnL: {('+' if pnl>=0 else '')}{fmt(pnl, 5)}\n"
        f"• pair stats: trades {pstats['trades']}, WR {wr_pair:.2f}%, PnL {fmt(pstats['pnl'],5)}\n"
        f"• total: trades {tstats['trades']}, WR {wr_tot:.2f}%, PnL {fmt(tstats['pnl'],5)}\n"
        f"• balance: {fmt(equity,5)}  (Δ {('+' if delta_equity>=0 else '')}{fmt(delta_equity,5)} | {since:.2f}%)\n"
        f"• since start: {since:.2f}%   (lev {LEVERAGE:.1f}x, fee {FEE_PCT*100:.3f}%)"
    )

# ---------- Core Loop ----------
def process_pair(bot: Bot, pair: str, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            df = get_klines(pair)
            df = calc_indicators(df)
            if len(df) < max(ATR_PERIOD + 5, 50):
                time.sleep(TICK_SECONDS)
                continue

            last = df.iloc[-1]
            price = float(last["close"])
            atr = float(last["atr"])
            slope = float(last["ema_slope"])

            with lock:
                # суточный стоп-аут
                if state["day_anchor"] != datetime.utcnow().date():
                    state["day_anchor"] = datetime.utcnow().date()
                    state["equity_high"] = state["equity"]
                dd = (state["equity_high"] - state["equity"]) / max(state["equity_high"], 1e-9)
                if dd >= DAILY_MAX_DD_PCT:
                    # ждём до следующего дня
                    time.sleep(TICK_SECONDS)
                    continue

                pos = state["opened"].get(pair)

            # управление открытой позицией
            if pos:
                # обновляем TP/SL через BE+Trailing
                be_and_trail(pos, price, atr)

                # TP/SL проверки
                exit_px = None
                reason = None
                if pos["side"] == "LONG":
                    if price >= pos["tp"]:
                        exit_px, reason = price, "TP"
                    elif price <= pos["sl"]:
                        exit_px, reason = price, "SL"
                else:
                    if price <= pos["tp"]:
                        exit_px, reason = price, "TP"
                    elif price >= pos["sl"]:
                        exit_px, reason = price, "SL"

                if exit_px is not None:
                    with lock:
                        # PnL
                        gross = (exit_px - pos["entry"]) * pos["qty"]
                        if pos["side"] == "SHORT":
                            gross = -gross
                        fee = fees_for(pos["qty"], pos["entry"], exit_px)
                        pnl = gross - fee

                        # Обновление стат
                        st_p = state["stats"][pair]
                        st_t = {"trades": state["total_trades"], "wins": state["total_wins"], "pnl": state["total_pnl"]}

                        st_p["trades"] += 1
                        st_p["pnl"] += pnl
                        state["total_trades"] += 1
                        state["total_pnl"] += pnl
                        if pnl >= 0:
                            st_p["wins"] += 1
                            state["total_wins"] += 1
                            state["consec_losses"] = 0
                        else:
                            state["consec_losses"] += 1

                        old_eq = state["equity"]
                        state["equity"] += pnl
                        delta_eq = state["equity"] - old_eq
                        state["opened"].pop(pair, None)

                    txt = close_text(pair, pos["side"], exit_px, pnl, st_p, 
                                     {"trades": state["total_trades"], "wins": state["total_wins"], "pnl": state["total_pnl"]},
                                     state["equity"], delta_eq)
                    send(bot, txt)

                    time.sleep(TICK_SECONDS)
                    continue

            # нет позиции — ищем вход
            if not pos:
                # кулдаун и серии лосей
                with lock:
                    last_open_ts = state["last_open_time"][pair]
                    if time.time() - last_open_ts < COOLDOWN_SEC:
                        time.sleep(TICK_SECONDS); continue
                    if state["consec_losses"] >= MAX_CONSEC_LOSSES:
                        time.sleep(TICK_SECONDS); continue
                    if len(state["opened"]) >= MAX_OPEN_POS:
                        time.sleep(TICK_SECONDS); continue

                sig = pattern_signals(df)
                sname = pick_signal(sig)
                if not sname:
                    time.sleep(TICK_SECONDS); continue

                side = side_from_signal(sname)
                if not allow_trend(df, side):
                    time.sleep(TICK_SECONDS); continue

                # уровни TP/SL от ATR
                if atr <= 0:
                    time.sleep(TICK_SECONDS); continue

                entry = price
                buf = ENTRY_BUFFER_ATR * atr
                if side == "LONG":
                    entry += buf
                    tp = entry + ATR_MULT_TP * atr
                    sl = entry - ATR_MULT_SL * atr
                else:
                    entry -= buf
                    tp = entry - ATR_MULT_TP * atr
                    sl = entry + ATR_MULT_SL * atr

                # размер
                with lock:
                    qty = lot_size(state["equity"], atr, entry)
                    if qty <= 0:
                        time.sleep(TICK_SECONDS); continue

                pos = {
                    "pair": pair,
                    "side": side,
                    "entry": entry,
                    "qty": qty,
                    "tp": tp,
                    "sl": sl,
                    "be_done": False,
                    "signal": sname,
                    "opened_at": time.time(),
                }
                with lock:
                    state["opened"][pair] = pos
                    state["last_open_time"][pair] = time.time()

                txt = open_text(pair, side, entry, qty, tp, sl, sname, slope, atr)
                send(Bot(TOKEN), txt)

        except Exception as e:
            print(f"[{pair}] loop error: {e}")

        time.sleep(TICK_SECONDS)

# ---------- Commands ----------
def cmd_status(update, _):
    with lock:
        lines = []
        lines.append(f"📊 *STATUS {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        tot_tr = state["total_trades"]
        tot_wr = 0.0 if tot_tr == 0 else state["total_wins"]/tot_tr*100
        for p in PAIRS:
            st = state["stats"][p]
            wr = 0.0 if st["trades"] == 0 else st["wins"]/st["trades"]*100
            lines.append(f"{p}• trades: {st['trades']}  WR: {wr:.2f}%  PnL: {fmt(st['pnl'],5)}")
            pos = state["opened"].get(p)
            if pos:
                lines.append(f"{fmt(pos['qty'],6)}  pos: {pos['side']} @ {fmt(pos['entry'],5)} "
                             f"(TP {fmt(pos['tp'],5)} / SL {fmt(pos['sl'],5)})")
            else:
                lines.append(f"{fmt(0,5)}  pos: —")
            lines.append("—")
        lines.append(f"TOTAL • trades: {tot_tr}  WR: {tot_wr:.2f}%  PnL: {fmt(state['total_pnl'],5)}")
        delta = state["equity"] - state["start_equity"]
        since = (state["equity"]/state["start_equity"] - 1.0) * 100.0
        lines.append(f"equity: {fmt(state['equity'],5)}  ({since:.2f}% с начала)")
        lines.append(f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT*100:.3f}%")
        update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

def cmd_start(update, _):
    update.message.reply_text(
        "Бот запущен в режиме DEMO.\n"
        f"Пары: {', '.join(PAIRS)} | TF {TF} | Tick {TICK_SECONDS}s\n"
        "Команды: /status",
        parse_mode=ParseMode.MARKDOWN
    )

# ---------- Main ----------
def main():
    if not TOKEN or not CHAT_ID:
        print("ERROR: TELEGRAM_TOKEN/TELEGRAM_CHAT_ID не заданы")
        return

    print("🤖 mybot9 started successfully!")
    print(f"Mode: {MODE} | Leverage {LEVERAGE}x | Fee {FEE_PCT*100:.3f}% | Risk {RISK_PCT*100:.2f}%")
    print(f"Pairs: {', '.join(PAIRS)} | TF {TF} | Tick {TICK_SECONDS}s")

    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))

    # background loops
    stop_event = threading.Event()
    for p in PAIRS:
        th = threading.Thread(target=process_pair, args=(updater.bot, p, stop_event), daemon=True)
        th.start()
        print(f"Loop started for {p}")

    # polling
    updater.start_polling(clean=True)
    updater.idle()
    stop_event.set()

if __name__ == "__main__":
    main()
