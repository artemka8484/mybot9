# bot/main.py
# DrGrand X Edition — compact & robust
# python-telegram-bot==13.15, requests, pandas, numpy

import os
import time
import json
import math
import queue
import signal
import random
import logging
import threading
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd

from telegram import Bot, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext

# ────────────────────────── CONFIG ──────────────────────────

ENV = lambda k, d=None: os.getenv(k, d)

TELEGRAM_TOKEN = ENV("TELEGRAM_TOKEN", "")
CHAT_ID        = int(ENV("TELEGRAM_CHAT_ID", "0"))

DEMO_MODE      = ENV("DEMO_MODE", "true").lower() == "true"
DRY_RUN        = ENV("DRY_RUN", "true").lower() == "true"  # алиас
MODE           = ENV("MODE", "DEMO")

PAIRS          = [s.strip().upper() for s in ENV("PAIRS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
TIMEFRAME      = ENV("TIMEFRAME", "5m").lower()  # 1m/5m/15m

EMA_LEN        = int(ENV("EMA_LEN", "100"))
EMA_SLOPE_BARS = int(ENV("EMA_SLOPE_BARS", "8"))

ATR_LEN        = int(ENV("ATR_LEN", "14"))
ATR_MULT_SL    = float(ENV("ATR_MULT_SL", "1.0"))

TP_PCT         = float(ENV("TP_PCT", "0.35")) / 100.0   # из % в долю
RISK_PCT       = float(ENV("RISK_PCT", "1.0")) / 100.0  # доля капитала на сделку
LEVERAGE       = float(ENV("LEVERAGE", "5"))
FEE_PCT        = float(ENV("FEE_PCT", "0.0006"))        # 0.060%

COOLDOWN_SEC   = int(ENV("COOLDOWN_SEC", "180"))
DAILY_SUMMARY  = ENV("DAILY_SUMMARY", "1") == "1"

MEXC_BASE_URL  = ENV("MEXC_BASE_URL", "https://contract.mexc.com")
MEXC_API_KEY   = ENV("MEXC_API_KEY", "")
MEXC_API_SECRET= ENV("MEXC_API_SECRET", "")

TZ             = timezone.utc if ENV("TX","UTC").upper() == "UTC" else None

# ───────────────────── GLOBAL STATE ────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("DrGrandX")

state_lock = threading.Lock()

state = dict(
    equity      = float(ENV("DEMO_START_BALANCE", "5000")) if DEMO_MODE else 0.0,
    start_equity= float(ENV("DEMO_START_BALANCE", "5000")) if DEMO_MODE else 0.0,
    day_anchor  = datetime.now(TZ).date(),
    pairs       = {},
    # {pair: {
    #   'cooldown_until': ts,
    #   'pos': {'side','entry','qty','sl','tp','opened_at','partial_done'}
    #   'stats': {'trades','wins','pnl'}
    # }}
)

for p in PAIRS:
    state["pairs"][p] = dict(
        cooldown_until=0.0,
        pos=None,
        stats=dict(trades=0, wins=0, pnl=0.0)
    )

# ────────────────────── UTILITIES ──────────────────────────

def utcnow():
    return datetime.now(TZ)

def fmt_money(x):
    return f"{x:,.5f}".replace(",", " ")

def send(bot: Bot, text: str):
    if CHAT_ID == 0 or not TELEGRAM_TOKEN:
        log.info("[TG] %s", text)
        return
    try:
        bot.send_message(CHAT_ID, text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        log.warning("Telegram error: %s", e)

def tf_to_mexc(tf: str) -> str:
    # MEXC contract uses "Min1/Min5/Min15/Min60/Day"
    m = {"1m":"Min1","3m":"Min3","5m":"Min5","15m":"Min15","30m":"Min30","1h":"Min60","4h":"Hour4","1d":"Day"}
    return m.get(tf, "Min5")

def tf_to_binance(tf: str) -> str:
    return tf  # binance uses '1m','5m','15m','1h','1d'

# ───────────────────── DATA FETCHERS ───────────────────────

def fetch_klines_mexc(pair: str, tf: str, limit=300) -> pd.DataFrame:
    try:
        interval = tf_to_mexc(tf)
        url = f"{MEXC_BASE_URL}/api/v1/contract/kline/{pair}"
        params = {"interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        js = r.json()
        if not js or "data" not in js or not js["data"]:
            return pd.DataFrame()
        # MEXC contract format: [t, open, high, low, close, vol]
        rows = js["data"]
        df = pd.DataFrame(rows, columns=["t","open","high","low","close","vol"])
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        for c in ["open","high","low","close","vol"]:
            df[c] = pd.to_numeric(df[c])
        df = df.sort_values("t").reset_index(drop=True)
        return df
    except Exception as e:
        log.warning("MEXC fetch fail %s %s: %s", pair, tf, e)
        return pd.DataFrame()

def fetch_klines_binance(pair: str, tf: str, limit=300) -> pd.DataFrame:
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": pair, "interval": tf_to_binance(tf), "limit": limit}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        arr = r.json()
        if not arr:
            return pd.DataFrame()
        # [Open time, O,H,L,C,Volume, Close time, ...]
        rows = []
        for x in arr:
            rows.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        df = pd.DataFrame(rows, columns=["t","open","high","low","close","vol"])
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        for c in ["open","high","low","close","vol"]:
            df[c] = pd.to_numeric(df[c])
        df = df.sort_values("t").reset_index(drop=True)
        return df
    except Exception as e:
        log.warning("Binance fetch fail %s %s: %s", pair, tf, e)
        return pd.DataFrame()

def get_klines(pair: str, tf: str, limit=300) -> pd.DataFrame:
    df = fetch_klines_mexc(pair, tf, limit)
    if df.empty:
        df = fetch_klines_binance(pair, tf, limit)
    return df.tail(limit)

# ─────────────────── INDICATORS & SIGNALS ──────────────────

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema"] = out["close"].ewm(span=EMA_LEN, adjust=False).mean()
    # slope of EMA over last N bars (simple regression over index)
    sl_window = EMA_SLOPE_BARS
    if len(out) >= sl_window + 1:
        y = out["ema"].tail(sl_window).values
        x = np.arange(sl_window)
        # slope per bar
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0.0
    out["ema_slope"] = 0.0
    out.loc[out.index[-1], "ema_slope"] = slope

    # ATR
    h, l, c = out["high"], out["low"], out["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_LEN).mean()
    # Bollinger squeeze (John Carter)
    ma20 = out["close"].rolling(20).mean()
    std20 = out["close"].rolling(20).std()
    out["bb_width"] = (std20 * 2) / ma20  # нормализованный width
    out["squeeze"] = out["bb_width"] < out["bb_width"].rolling(120).quantile(0.2)
    return out

def candle_patterns(df: pd.DataFrame) -> dict:
    """Возвращает паттерн на последней свече (bool)."""
    if len(df) < 3:
        return {k: False for k in ["bull_engulf","bear_engulf","hammer","shooting_star","breakout_up","breakout_down"]}

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # последняя
    O, H, L, C = o[-1], h[-1], l[-1], c[-1]
    O1, C1 = o[-2], c[-2]

    body = abs(C - O)
    rng  = H - L + 1e-9
    upper = H - max(C,O)
    lower = min(C,O) - L

    bull_engulf = (C > O) and (O <= min(O1,C1)) and (C >= max(O1,C1))
    bear_engulf = (C < O) and (O >= max(O1,C1)) and (C <= min(O1,C1))

    hammer = (C > O) and (lower > 2*body) and (upper < body*0.6)
    shooting = (C < O) and (upper > 2*body) and (lower < body*0.6)

    # простейший breakout: закрытие выше/ниже max/min N
    N = 40
    breakout_up = C > df["high"].rolling(N).max().iloc[-2] and C > O
    breakout_dn = C < df["low"].rolling(N).min().iloc[-2] and C < O

    return dict(
        bull_engulf=bull_engulf,
        bear_engulf=bear_engulf,
        hammer=hammer,
        shooting_star=shooting,
        breakout_up=breakout_up,
        breakout_down=breakout_dn,
    )

def build_signal(df: pd.DataFrame) -> tuple:
    """Возвращает ('LONG'/'SHORT'/None, reason, ema_slope, atr)."""
    ind = indicators(df)
    pat = candle_patterns(ind)

    row = ind.iloc[-1]
    slope = float(row["ema_slope"])
    atr   = float(row["atr"]) if not math.isnan(row["atr"]) else 0.0
    ema   = float(row["ema"])
    price = float(row["close"])
    in_squeeze = bool(row["squeeze"])

    # поведенческий фильтр (Kahneman bias control):
    # торгуем только в сторону EMA-bias (цена над EMA -> long bias; под -> short bias)
    long_bias  = price >= ema
    short_bias = price <= ema

    long_triggers = [
        ("bull_engulf", pat["bull_engulf"]),
        ("hammer", pat["hammer"]),
        ("breakout_up", pat["breakout_up"])
    ]
    short_triggers = [
        ("bear_engulf", pat["bear_engulf"]),
        ("shooting_star", pat["shooting_star"]),
        ("breakout_down", pat["breakout_down"])
    ]

    # требуется выход из сжатия (или сжатие False) чтобы не стрелять в тишине
    if in_squeeze:
        return (None, "squeeze", slope, atr)

    # небольшой фильтр на наклон EMA (усиливает трендовые отборы)
    slope_ok_long  = slope >= 0 or pat["breakout_up"]
    slope_ok_short = slope <= 0 or pat["breakout_down"]

    for name, ok in long_triggers:
        if ok and long_bias and slope_ok_long:
            return ("LONG", name, slope, atr)
    for name, ok in short_triggers:
        if ok and short_bias and slope_ok_short:
            return ("SHORT", name, slope, atr)

    return (None, "", slope, atr)

# ───────────────────── TRADING ENGINE ──────────────────────

def now_ts() -> float:
    return time.time()

def can_trade(pair: str) -> bool:
    with state_lock:
        cd = state["pairs"][pair]["cooldown_until"]
    return now_ts() >= cd

def set_cooldown(pair: str):
    with state_lock:
        state["pairs"][pair]["cooldown_until"] = now_ts() + COOLDOWN_SEC

def open_demo(pair: str, side: str, price: float, atr: float, reason: str, bot: Bot):
    with state_lock:
        eq = state["equity"]
    risk_amount = eq * RISK_PCT
    # позиция в USDT, размер в монетах:
    qty = max(risk_amount * LEVERAGE / price, 0.00001)

    # ATR SL
    sl = price - ATR_MULT_SL * atr if side == "LONG" else price + ATR_MULT_SL * atr
    # первичный TP (частичный)
    tp = price * (1 + TP_PCT) if side == "LONG" else price * (1 - TP_PCT)

    pos = dict(side=side, entry=price, qty=qty, sl=sl, tp=tp, opened_at=now_ts(), partial_done=False, atr=atr)
    with state_lock:
        state["pairs"][pair]["pos"] = pos
        state["pairs"][pair]["stats"]["trades"] += 1

    text = (
        f"🔴 *OPEN {pair} {side}*\n"
        f"• time: {utcnow():%Y-%m-%d %H:%M:%S} UTC\n"
        f"• entry: {price:.5f}\n"
        f"• qty: {qty:.6f}\n"
        f"• TP: {tp:.5f}   SL: {sl:.5f}\n"
        f"• signal: {reason}\n"
        f"• mode: {'DEMO' if DEMO_MODE else 'REAL'}"
    )
    send(bot, text)

def close_demo(pair: str, price: float, reason: str, bot: Bot):
    with state_lock:
        pos = state["pairs"][pair]["pos"]
    if not pos:
        return
    side = pos["side"]
    qty  = pos["qty"]
    entry= pos["entry"]

    pnl = (price - entry) * qty if side == "LONG" else (entry - price) * qty
    fee = abs(entry*qty)*FEE_PCT + abs(price*qty)*FEE_PCT
    pnl -= fee

    with state_lock:
        state["pairs"][pair]["pos"] = None
        st = state["pairs"][pair]["stats"]
        st["pnl"] += pnl
        if pnl > 0: st["wins"] += 1
        state["equity"] += pnl

    text = (
        f"{'✅' if pnl>0 else '❌'} *CLOSE {pair} ({reason})*\n"
        f"• time: {utcnow():%Y-%m-%d %H:%M:%S} UTC\n"
        f"• exit: {price:.5f}\n"
        f"• PnL: {pnl:+.5f}\n"
        f"• pair stats: trades {st['trades']}, WR {st['wins']*100.0/max(1,st['trades']):.2f}%, PnL {st['pnl']:.5f}\n"
        f"• total: {summary_line()}"
    )
    send(bot, text)
    set_cooldown(pair)

def trail_manage(pair: str, price: float, bot: Bot):
    """частичный TP 50%, затем трейлинг по 1*ATR от экстремума прибыли."""
    with state_lock:
        pos = state["pairs"][pair]["pos"]
    if not pos: return

    side = pos["side"]; entry = pos["entry"]; qty = pos["qty"]
    atr = max(pos.get("atr", 0.0), 1e-9)

    # частичный TP
    if not pos["partial_done"]:
        if (side=="LONG" and price >= pos["tp"]) or (side=="SHORT" and price <= pos["tp"]):
            new_qty = qty * 0.5
            pnl = (pos["tp"]-entry)* (qty - new_qty) if side=="LONG" else (entry-pos["tp"])*(qty - new_qty)
            fee = abs(entry*(qty - new_qty))*FEE_PCT + abs(pos["tp"]*(qty - new_qty))*FEE_PCT
            pnl -= fee
            with state_lock:
                state["equity"] += pnl
                pos["qty"] = new_qty
                pos["partial_done"] = True
                # сдвигаем SL в безубыток +/- 0.2*ATR
                pos["sl"] = entry + (0.2*atr if side=="LONG" else -0.2*atr)
            send(bot, f"🟢 *PARTIAL TP {pair}* @ {pos['tp']:.5f} | qty→{new_qty:.6f}")
            return

    # трейлинг остатка: SL = лучшая цена -/+ 1*ATR
    if pos["partial_done"]:
        if side=="LONG":
            best = max(entry, price)
            new_sl = best - 1.0*atr
            with state_lock:
                pos["sl"] = max(pos["sl"], new_sl)
        else:
            best = min(entry, price)
            new_sl = best + 1.0*atr
            with state_lock:
                pos["sl"] = min(pos["sl"], new_sl)

def summary_line() -> str:
    with state_lock:
        eq = state["equity"]; start = state["start_equity"]
        delta = eq - start; pct = (delta/start*100.0) if start>0 else 0.0
    return f"trades {total_trades()} WR {total_wr():.2f}% PnL {delta:+.5f}\n• balance: {eq:.5f}  (Δ {delta:+.5f} | {pct:.2f}%)\n• since start: {pct:.2f}%   (lev {LEVERAGE:.1f}x, fee {FEE_PCT*100:.3f}%)"

def total_trades() -> int:
    with state_lock:
        return sum(s["stats"]["trades"] for s in state["pairs"].values())

def total_wr() -> float:
    with state_lock:
        wins = sum(s["stats"]["wins"] for s in state["pairs"].values())
        tr   = sum(s["stats"]["trades"] for s in state["pairs"].values())
    return 100.0 * wins / max(1, tr)

# ───────────────────── LOOPS PER PAIR ──────────────────────

def pair_loop(pair: str, bot: Bot, stop_event: threading.Event):
    log.info("Loop started for %s", pair)
    last_bar_time = None

    while not stop_event.is_set():
        try:
            df = get_klines(pair, TIMEFRAME, limit=300)
            if df.empty or len(df) < max(EMA_LEN, ATR_LEN)+5:
                send(bot, f"⚠️ {pair} loop error: no klines for {pair}")
                time.sleep(10); continue

            # работаем только на закрытии свежего бара: сравниваем последнюю t
            cur_bar_time = df["t"].iloc[-1]
            price = float(df["close"].iloc[-1])

            # активная позиция → сопровождение + выходы SL/TP
            with state_lock:
                pos = state["pairs"][pair]["pos"]

            if pos:
                trail_manage(pair, price, bot)
                # выходы по SL
                if (pos["side"]=="LONG" and price <= pos["sl"]) or (pos["side"]=="SHORT" and price >= pos["sl"]):
                    close_demo(pair, price, "SL", bot)
                # если остаток крошечный — закрыть
                with state_lock:
                    pos = state["pairs"][pair]["pos"]
                if pos and pos["qty"] < 1e-8:
                    close_demo(pair, price, "dust", bot)

            # сигналы только при новом баре
            if last_bar_time is not None and cur_bar_time == last_bar_time:
                time.sleep(2); continue

            last_bar_time = cur_bar_time

            if not pos and can_trade(pair):
                side, reason, slope, atr = build_signal(df)
                if side in ("LONG","SHORT") and atr>0:
                    open_demo(pair, side, price, atr, f"{reason}, slope {slope:+.5f}, ATR {atr:.5f}", bot)

        except requests.exceptions.RequestException as e:
            log.warning("Network error %s: %s", pair, e)
            time.sleep(3)
        except Exception as e:
            log.exception("Loop error %s: %s", pair, e)
            time.sleep(3)

# ──────────────────── TELEGRAM COMMANDS ────────────────────

def cmd_start(update, context: CallbackContext):
    update.message.reply_text("🤖 DrGrand X запущен. Команды: /status /reset /help")

def compose_status() -> str:
    lines = [f"📊 *STATUS {utcnow():%Y-%m-%d %H:%M:%S} UTC*"]
    with state_lock:
        for p, s in state["pairs"].items():
            st = s["stats"]; pos = s["pos"]
            line = f"{p} • trades: {st['trades']}  WR: { (st['wins']*100.0/max(1,st['trades'])):.2f}%  PnL:\n{st['pnl']:.5f}"
            lines.append(line)
            if pos:
                lines.append(f"{pos['qty']:.6f}  pos: {pos['side']} @ {pos['entry']:.5f} (TP {pos['tp']:.5f} / SL {pos['sl']:.5f})")
            else:
                lines.append("0.00000  pos: —")
            lines.append("—")
        eq = state["equity"]; start = state["start_equity"]
        delta = eq - start; pct = (delta/start*100.0) if start>0 else 0.0
        lines.append(f"TOTAL • trades: {total_trades()}  WR: {total_wr():.2f}%  PnL:\n{delta:+.5f}")
        lines.append(f"equity: {eq:.5f}  ({pct:+.2f}% с начала)")
        lines.append(f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT*100:.3f}%")
    return "\n".join(lines)

def cmd_status(update, context: CallbackContext):
    update.message.reply_text(compose_status(), parse_mode=ParseMode.MARKDOWN)

def cmd_reset(update, context: CallbackContext):
    with state_lock:
        for p in PAIRS:
            state["pairs"][p]["pos"] = None
            state["pairs"][p]["cooldown_until"] = 0.0
            state["pairs"][p]["stats"] = dict(trades=0, wins=0, pnl=0.0)
        if DEMO_MODE:
            state["equity"] = state["start_equity"]
    update.message.reply_text("♻️ Сброс выполнен.")

def cmd_help(update, context: CallbackContext):
    txt = (
        "*DrGrand X*\n"
        "• Вход: 6 свечных паттернов + EMA bias + Squeeze фильтр\n"
        "• Риск: RISK_PCT на сделку, ATR_SL, частичный TP 50% + трейлинг\n"
        "Команды: /status /reset /help"
    )
    update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

# ───────────────────────── MAIN ────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN is empty")
        return

    bot = Bot(TELEGRAM_TOKEN)

    # приветствие при старте
    hdr = (
        f"🤖 mybot9 started successfully!\n"
        f"Mode: {'DEMO' if DEMO_MODE else 'REAL'} | Leverage {LEVERAGE:.1f}x | Fee {FEE_PCT*100:.3f}% | Risk {RISK_PCT*100:.1f}%\n"
        f"Pairs: {', '.join(PAIRS)} | TF {TIMEFRAME} | Tick {10}s\n"
        f"Balance: {state['equity']:.2f}  USDT"
    )
    send(bot, hdr)

    # Telegram updater
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("reset", cmd_reset))
    dp.add_handler(CommandHandler("help", cmd_help))

    # НЕ используем clean=True (чтобы не падал на 13.15),
    # а конфликты решаем тем, что игнорируем исключение и пробуем снова.
    def safe_polling():
        while True:
            try:
                updater.start_polling(timeout=30, read_latency=5.0)
                updater.idle()  # блокирует пока не stop()
                break
            except Exception as e:
                # типичный случай: Conflict getUpdates — значит, параллельный инстанс. Ждём и пробуем снова.
                log.warning("Telegram polling error: %s", e)
                time.sleep(5)

    stop_event = threading.Event()
    # стартуем лупы по парам
    for p in PAIRS:
        t = threading.Thread(target=pair_loop, args=(p, bot, stop_event), daemon=True)
        t.start()

    # polling в основном потоке
    try:
        safe_polling()
    finally:
        stop_event.set()

if __name__ == "__main__":
    main()
