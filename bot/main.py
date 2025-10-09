import os
import time
import math
import json
import queue
import atexit
import signal
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

import requests
import numpy as np
import pandas as pd

from telegram import Bot, ParseMode, Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# ========= ENV =========
ENV = lambda k, d=None: os.getenv(k, d)

TELEGRAM_TOKEN      = ENV("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID    = int(ENV("TELEGRAM_CHAT_ID", "0"))

MODE                = ENV("MODE", ENV("DEMO_MODE", "true")).upper()
DEMO_MODE           = MODE == "DEMO" or ENV("DEMO_MODE", "true").lower() == "true"
DRY_RUN             = ENV("DRY_RUN", "false").lower() == "true"     # если true — не «отправляем» реальные ордера

PAIRS               = [p.strip().upper() for p in ENV("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",")]

TIMEFRAME           = ENV("TIMEFRAME", "5m").lower()   # 1m/3m/5m/15m/1h

LEVERAGE            = float(ENV("LEVERAGE", "5"))
FEE_PCT             = float(ENV("FEE_PCT", "0.0006"))   # 0.06%
RISK_PCT            = float(ENV("RISK_PCT", "3.0"))     # риск на сделку от equity (без учёта плеча)

TP_PCT              = float(ENV("TP_PCT", "1"))         # % от цены (например 1 = 1% )
ATR_LEN             = int(ENV("ATR_LEN", "14"))
ATR_MULT_SL         = float(ENV("ATR_MULT_SL", "0.5"))  # SL = ATR * множитель
EMA_LEN             = int(ENV("EMA_LEN", "48"))
EMA_SLOPE_BARS      = int(ENV("EMA_SLOPE_BARS", "5"))

COOLDOWN_SEC        = int(ENV("COOLDOWN_SEC", "60"))
TICK_SEC            = 10  # внутренний тик

DEMO_START_BALANCE  = float(ENV("DEMO_START_BALANCE", "5000"))

MEXC_BASE_URL       = ENV("MEXC_BASE_URL", "https://contract.mexc.com")

DAILY_SUMMARY       = ENV("DAILY_SUMMARY", "1") == "1"

# ========= GLOBAL STATE =========
state_lock = threading.Lock()

pair_state: Dict[str, Dict[str, Any]] = {
    p: dict(
        last_kline_ts=0,
        open_pos=None,         # dict(side, entry, qty, tp, sl, opened_at)
        last_signal_time=0,
        stats=dict(trades=0, wins=0, pnl=0.0)
    ) for p in PAIRS
}

account = dict(
    start_equity = DEMO_START_BALANCE if DEMO_MODE else 0.0,
    equity       = DEMO_START_BALANCE if DEMO_MODE else 0.0,
    since_start  = 0.0,    # %
)

# ========= UTILS =========
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def safe_get(url, params=None, headers=None, timeout=10):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, f"{r.status_code} {r.text[:160]}"
    except Exception as e:
        return None, str(e)

# ========= DATA: MEXC + fallback BINANCE =========
MEXC_TF = {
    "1m": "Min1", "3m": "Min3", "5m": "Min5", "15m": "Min15",
    "1h": "Hour1", "4h": "Hour4"
}
BIN_TF = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","1h":"1h","4h":"4h"}

def klines_mexc(symbol: str, tf: str, limit=200) -> Optional[pd.DataFrame]:
    if tf not in MEXC_TF: return None
    url = f"{MEXC_BASE_URL}/api/v1/contract/kline/{symbol}"
    params = {"interval": MEXC_TF[tf], "limit": limit}
    js, err = safe_get(url, params=params)
    if js is None: 
        return None
    # ожидаемый формат: list[ { t, o, h, l, c, v }, ... ]
    data = js if isinstance(js, list) else js.get("data") or js.get("success") or None
    if not data: 
        return None
    rows = []
    for it in data:
        try:
            t = int(it["t"])
            rows.append([t, float(it["o"]), float(it["h"]), float(it["l"]), float(it["c"]), float(it["v"])])
        except Exception:
            return None
    df = pd.DataFrame(rows, columns=["t","open","high","low","close","vol"]).sort_values("t")
    return df

def klines_binance(symbol: str, tf: str, limit=200) -> Optional[pd.DataFrame]:
    if tf not in BIN_TF: return None
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": BIN_TF[tf], "limit": limit}
    js, err = safe_get(url, params=params)
    if js is None: 
        return None
    # формат: [ [openTime, o,h,l,c, v, closeTime, ...], ... ]
    rows = []
    try:
        for k in js:
            rows.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
    except Exception:
        return None
    df = pd.DataFrame(rows, columns=["t","open","high","low","close","vol"]).sort_values("t")
    return df

def get_klines(symbol: str, tf: str, limit=200) -> Optional[pd.DataFrame]:
    df = klines_mexc(symbol, tf, limit)
    if df is not None and len(df) >= 50: 
        return df
    # fallback
    return klines_binance(symbol, tf, limit)

# ========= INDICATORS / PATTERNS =========
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length=14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def last_candle(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1]

def body(c): 
    return abs(c["close"] - c["open"])

def is_bull_engulf(df: pd.DataFrame) -> bool:
    a, b = df.iloc[-2], df.iloc[-1]
    return (b["close"] > b["open"]) and (a["close"] < a["open"]) and (b["close"] >= a["open"]) and (b["open"] <= a["close"])

def is_bear_engulf(df: pd.DataFrame) -> bool:
    a, b = df.iloc[-2], df.iloc[-1]
    return (b["close"] < b["open"]) and (a["close"] > a["open"]) and (b["close"] <= a["open"]) and (b["open"] >= a["close"])

def is_hammer(df: pd.DataFrame) -> bool:
    c = df.iloc[-1]
    rng = c["high"] - c["low"]
    if rng <= 0: return False
    lower = (min(c["open"], c["close"]) - c["low"]) / rng
    upper = (c["high"] - max(c["open"], c["close"])) / rng
    return lower >= 0.6 and upper <= 0.2

def is_shooting_star(df: pd.DataFrame) -> bool:
    c = df.iloc[-1]
    rng = c["high"] - c["low"]
    if rng <= 0: return False
    upper = (c["high"] - max(c["open"], c["close"])) / rng
    lower = (min(c["open"], c["close"]) - c["low"]) / rng
    return upper >= 0.6 and lower <= 0.2

def breakouts(df: pd.DataFrame) -> (bool, bool):
    # простая логика: пробой максимума/минимума послед. N-баров
    N = 10
    highs = df["high"].rolling(N).max()
    lows  = df["low"].rolling(N).min()
    c = df.iloc[-1]
    up   = c["close"] >= highs.iloc[-2] and c["close"] > c["open"]
    down = c["close"] <= lows.iloc[-2] and c["close"] < c["open"]
    return up, down

def ema_slope(df: pd.DataFrame) -> float:
    e = ema(df["close"], EMA_LEN)
    if len(e) < EMA_SLOPE_BARS+1: 
        return 0.0
    return float(e.iloc[-1] - e.iloc[-1-EMA_SLOPE_BARS])

# ========= POSITION / SIZING =========
def calc_qty(price: float) -> float:
    # Риск от equity; плечо учитываем в доступности
    with state_lock:
        eq = account["equity"]
    risk_cash = eq * (RISK_PCT / 100.0)
    # берём стоп как ATR*mult — оценим средн. шагом 0.5% цены для worst-case
    # Чтобы не занижать, приблизим SL% = max( ATR_mult*0.5%, 0.3% ) — но реальный SL посчитается из ATR ниже
    est_sl_pct = max(0.003, ATR_MULT_SL * 0.005)
    notional = risk_cash / est_sl_pct
    notional = min(notional, eq * LEVERAGE)  # не превышать кредитное плечо
    qty = notional / price
    return round(qty, 6)

def open_position(pair: str, side: str, entry: float, atr_val: float, bot: Bot):
    qty = calc_qty(entry)
    tp  = entry * (1 + TP_PCT/100.0) if side == "LONG" else entry * (1 - TP_PCT/100.0)
    # SL по ATR
    sl  = entry - atr_val*ATR_MULT_SL if side == "LONG" else entry + atr_val*ATR_MULT_SL

    with state_lock:
        pair_state[pair]["open_pos"] = dict(
            side=side, entry=entry, qty=qty, tp=tp, sl=sl, opened_at=now_utc().isoformat()
        )

    sigs = []
    # только для логов (что именно сработало)
    bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=(
            f"🔴 OPEN {pair} {side}\n"
            f"• time: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"• entry: {entry:.6f}\n"
            f"• qty: {qty}\n"
            f"• TP: {tp:.6f}   SL: {sl:.6f}\n"
            f"• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
        )
    )

def close_position(pair: str, price: float, hit: str, bot: Bot):
    with state_lock:
        pos = pair_state[pair]["open_pos"]
        pair_state[pair]["open_pos"] = None
    if not pos:
        return
    side = pos["side"]
    qty  = pos["qty"]
    entry= pos["entry"]

    # PnL (в «условных» USDT); комиссия с двух сторон
    notional = entry*qty + price*qty
    fee = notional * FEE_PCT
    raw = (price - entry) * qty if side == "LONG" else (entry - price) * qty
    pnl = raw - fee

    win = pnl > 0
    emoji = "✅" if win else "❌"

    with state_lock:
        st = pair_state[pair]["stats"]
        st["trades"] += 1
        if win: st["wins"] += 1
        st["pnl"] += pnl

        account["equity"] += pnl
        account["since_start"] = (account["equity"] / account["start_equity"] - 1) * 100.0

    pair_wr = (st["wins"]/st["trades"]*100.0) if st["trades"]>0 else 0.0

    # Тотал
    with state_lock:
        tot_trades = sum(pair_state[p]["stats"]["trades"] for p in PAIRS)
        tot_wins   = sum(pair_state[p]["stats"]["wins"]   for p in PAIRS)
        tot_pnl    = sum(pair_state[p]["stats"]["pnl"]    for p in PAIRS)
        tot_wr     = (tot_wins/tot_trades*100.0) if tot_trades>0 else 0.0
        bal        = account["equity"]
        since      = account["since_start"]

    bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=(
            f"{emoji} CLOSE {pair} ({hit})\n"
            f"• time: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"• exit: {price:.6f}\n"
            f"• PnL: {pnl:+.5f}\n"
            f"• pair stats: trades {st['trades']}, WR {pair_wr:.2f}%, PnL {st['pnl']:.5f}\n"
            f"• total: trades {tot_trades}, WR {tot_wr:.2f}%, PnL {tot_pnl:.5f}\n"
            f"• balance: {bal:.5f}  (Δ {tot_pnl:+.5f} | {since:.2f}%)\n"
            f"• since start: {since:.2f}%   (lev {LEVERAGE:.1f}x, fee {FEE_PCT*100:.3f}%)"
        )
    )

# ========= STRATEGY =========
def decide_and_trade(pair: str, df: pd.DataFrame, bot: Bot):
    if len(df) < max(EMA_LEN, ATR_LEN) + 20:
        return

    with state_lock:
        pos = pair_state[pair]["open_pos"]
        last_sig = pair_state[pair]["last_signal_time"]

    # TP/SL триггеры
    if pos:
        price = float(df.iloc[-1]["close"])
        if pos["side"] == "LONG":
            if price >= pos["tp"]:
                close_position(pair, price, "TP", bot); return
            if price <= pos["sl"]:
                close_position(pair, price, "SL", bot); return
        else:
            if price <= pos["tp"]:
                close_position(pair, price, "TP", bot); return
            if price >= pos["sl"]:
                close_position(pair, price, "SL", bot); return
        return  # пока позиция открыта — новых не открываем

    # Cooldown
    if time.time() - last_sig < COOLDOWN_SEC:
        return

    # Индикаторы
    atr_val = float(atr(df, ATR_LEN).iloc[-1])
    slope   = ema_slope(df)
    up_bo, down_bo = breakouts(df)

    long_sig  = (is_bull_engulf(df) or is_hammer(df) or up_bo) and slope > 0
    short_sig = (is_bear_engulf(df) or is_shooting_star(df) or down_bo) and slope < 0

    price = float(df.iloc[-1]["close"])

    if long_sig:
        with state_lock: pair_state[pair]["last_signal_time"] = time.time()
        open_position(pair, "LONG", price, atr_val, bot)

    elif short_sig:
        with state_lock: pair_state[pair]["last_signal_time"] = time.time()
        open_position(pair, "SHORT", price, atr_val, bot)

# ========= LOOP PER PAIR =========
def pair_loop(pair: str, bot: Bot):
    bot.send_message(TELEGRAM_CHAT_ID, f"✅ Loop started for <b>{pair}</b>", parse_mode=ParseMode.HTML)
    while True:
        try:
            df = get_klines(pair, TIMEFRAME, limit=300)
            if df is None or df.empty:
                bot.send_message(TELEGRAM_CHAT_ID, f"⚠️ {pair} loop error: <i>no klines for {pair}</i>", parse_mode=ParseMode.HTML)
                time.sleep(TICK_SEC)
                continue
            decide_and_trade(pair, df, bot)
        except Exception as e:
            bot.send_message(TELEGRAM_CHAT_ID, f"⚠️ {pair} loop error: {e}")
        time.sleep(TICK_SEC)

# ========= TELEGRAM CMDS =========
def cmd_start(update: Update, ctx: CallbackContext):
    update.message.reply_text(
        "Привет! Я готов. Используй /status чтобы посмотреть статистику."
    )

def cmd_status(update: Update, ctx: CallbackContext):
    lines = [f"📊 <b>STATUS {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}</b>"]
    tot_tr, tot_wi, tot_pnl = 0, 0, 0.0
    for p in PAIRS:
        with state_lock:
            st = pair_state[p]["stats"]
            pos = pair_state[p]["open_pos"]
        wr = (st["wins"]/st["trades"]*100.0) if st["trades"]>0 else 0.0
        lines.append(f"{p} • trades: {st['trades']:d}  WR: {wr:.2f}%  PnL: {st['pnl']:.5f}")
        if pos:
            side = pos['side']; e=pos['entry']; tp=pos['tp']; sl=pos['sl']
            lines.append(f"{pos['qty']:.6f}  pos: {side} @ {e:.5f} (TP {tp:.5f} / SL {sl:.5f})")
        else:
            lines.append("0.00000  pos: —")
        tot_tr += st["trades"]; tot_wi += st["wins"]; tot_pnl += st["pnl"]
    tot_wr = (tot_wi/tot_tr*100.0) if tot_tr>0 else 0.0
    with state_lock:
        eq = account["equity"]; since = account["since_start"]
    lines += [
        "—",
        f"TOTAL • trades: {tot_tr}  WR: {tot_wr:.2f}%  PnL: {tot_pnl:.5f}",
        f"equity: {eq:.5f}  ({since:.2f}% с начала)",
        f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT*100:.3f}%"
    ]
    update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

# ========= BOOT =========
def start_health_http_server():
    # простой «заглушка» сервер уже поднимается платформой, второй порт не нужен.
    # оставляем функцию для совместимости — ничего не делаем.
    return

def on_exit(bot: Bot):
    try:
        bot.send_message(TELEGRAM_CHAT_ID, "🛑 mybot9 is stopping…")
    except Exception:
        pass

def main():
    assert TELEGRAM_TOKEN, "No TELEGRAM_TOKEN provided"
    bot = Bot(token=TELEGRAM_TOKEN)

    # приветствие
    bot.send_message(
        TELEGRAM_CHAT_ID,
        (
            "🤖 mybot9 started successfully!\n"
            f"Mode: {'DEMO' if DEMO_MODE else 'LIVE'} | Leverage {LEVERAGE:.1f}x | Fee {FEE_PCT*100:.3f}% | Risk {RISK_PCT:.1f}%\n"
            f"Pairs: {', '.join(PAIRS)} | TF {TIMEFRAME} | Tick {TICK_SEC}s\n"
            f"Balance: {account['equity']:.2f}  USDT"
        )
    )

    # фоновая торговля по парам
    for p in PAIRS:
        th = threading.Thread(target=pair_loop, args=(p, bot), daemon=True)
        th.start()

    # Telegram polling
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))

    # очищаем подвисшие апдейты — уменьшаем риск «Conflict: terminated by other getUpdates request»
    updater.start_polling(clean=True)
    atexit.register(on_exit, bot)

    # держим процесс живым
    updater.idle()

if __name__ == "__main__":
    start_health_http_server()
    main()
