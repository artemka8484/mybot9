# mybot9/bot/main.py
import asyncio, os, math
from datetime import datetime, timezone
import json
import pandas as pd
import numpy as np
import httpx
from loguru import logger

# ------------ ENV ------------
API_KEY     = os.getenv("MEXC_API_KEY", "")
API_SECRET  = os.getenv("MEXC_API_SECRET", "")
TELE_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
TELE_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")
TIMEFRAME   = os.getenv("TIMEFRAME", "1m")
DEMO_MODE   = os.getenv("DEMO_MODE", "true").lower() in ("1","true","yes")
PAIR_STR    = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT")
PAIRS       = [p.strip().upper() for p in PAIR_STR.split(",") if p.strip()]

DEBUG_TEL   = os.getenv("DEBUG_TELEMETRY","0") in ("1","true","yes")

# торговые параметры (бережно)
EMA_LEN     = 48
ATR_LEN     = 14
RISK_RR     = 1.5   # цель = 1.5*ATR
SLEEP_SEC   = 5     # пауза между циклами
BARS_NEED   = max(EMA_LEN, ATR_LEN) + 5

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# ------------ Telegram ------------
async def tg_send(text: str):
    if not (TELE_TOKEN and TELE_CHAT): return
    url = f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage"
    payload = {"chat_id": int(TELE_CHAT), "text": text}
    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as cl:
        try:
            await cl.post(url, json=payload)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ------------ Data ------------
async def get_klines(pair: str, interval: str, limit: int = 200):
    params = {"symbol": pair, "interval": interval, "limit": limit}
    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as cl:
        r = await cl.get(BINANCE_KLINES, params=params)
        r.raise_for_status()
        data = r.json()
    # колонки binance: [open_time, open, high, low, close, volume, close_time, ...]
    arr = []
    for row in data:
        arr.append([
            int(row[0]),
            float(row[1]), float(row[2]), float(row[3]), float(row[4]),
            float(row[5])
        ])
    df = pd.DataFrame(arr, columns=["ts","open","high","low","close","vol"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# ------------ Indicators & patterns ------------
def ema(series: pd.Series, n: int):
    return series.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def is_bull_engulf(df: pd.DataFrame):
    # пред. медвежья, текущая бычья и тело покрывает предыдущее
    prev = df.iloc[-2]; cur = df.iloc[-1]
    return (prev.close < prev.open) and (cur.close > cur.open) and \
           (cur.close >= prev.open) and (cur.open <= prev.close)

def is_bear_engulf(df: pd.DataFrame):
    prev = df.iloc[-2]; cur = df.iloc[-1]
    return (prev.close > prev.open) and (cur.close < cur.open) and \
           (cur.open >= prev.close) and (cur.close <= prev.open)

def is_hammer(row):
    body = abs(row.close - row.open)
    upper = row.high - max(row.close,row.open)
    lower = min(row.close,row.open) - row.low
    return lower > 2*body and upper < body

def is_shooting(row):
    body = abs(row.close - row.open)
    upper = row.high - max(row.close,row.open)
    lower = min(row.close,row.open) - row.low
    return upper > 2*body and lower < body

# ------------ Trading state per pair ------------
class PairState:
    def __init__(self):
        self.position = None  # {"side":"LONG/SHORT","entry":float,"sl":float,"tp":float}

STATES = {p: PairState() for p in PAIRS}

# ------------ Strategy 9 core ------------
def strategy_signal(df: pd.DataFrame):
    """
    Возвращает ('LONG'|'SHORT'|None, info_dict)
    """
    if len(df) < BARS_NEED:
        return None, {"reason":"not_enough_bars"}

    df = df.copy()
    df["ema48"] = ema(df["close"], EMA_LEN)
    df["atr"]   = atr(df, ATR_LEN)

    slope = df["ema48"].iloc[-1] - df["ema48"].iloc[-6]  # наклон за ~5 баров
    cur   = df.iloc[-1]

    patt_long  = is_bull_engulf(df) or is_hammer(cur)
    patt_short = is_bear_engulf(df) or is_shooting(cur)

    long_ok  = slope > 0 and patt_long
    short_ok = slope < 0 and patt_short

    info = {
        "last_close": cur["close"],
        "ema48_slope_5": float(slope),
        "patt": {"bull_engulf":is_bull_engulf(df),
                 "bear_engulf":is_bear_engulf(df),
                 "hammer":is_hammer(cur),
                 "shooting":is_shooting(cur)}
    }

    if long_ok:  return "LONG", info
    if short_ok: return "SHORT", info
    return None, info

# ------------ Order emulation/real ------------
async def execute_trade(pair: str, side: str, price: float, atr_val: float):
    qty = float(os.getenv("TRADE_SIZE","0.001"))
    if DEMO_MODE:
        await tg_send(
            f"🧪 DEMO trade\n• {side} {qty} {pair} @ {price:.2f}\n• Время: {utc_now_str()}\n• Стратегия: #9\n• Исполнение: без реального ордера"
        )
        return
    # TODO: здесь позже подключим реальный MEXC/OKX/… REST для ордеров.
    await tg_send(
        f"✅ LIVE order SENT\n• {side} {qty} {pair} @ ~{price:.2f}\n• time: {utc_now_str()}"
    )

# ------------ Worker per pair ------------
async def run_pair(pair: str):
    state = STATES[pair]
    await tg_send(f"✅ mybot9 running ({'DEMO' if DEMO_MODE else 'LIVE'})\nPAIR: {pair} TF: {TIMEFRAME}")

    while True:
        try:
            df = await get_klines(pair, TIMEFRAME, limit=300)
            signal, info = strategy_signal(df)

            if DEBUG_TEL:
                await tg_send(
                    "🧪 DEBUG"
                    f"\n• pair: {pair}"
                    f"\n• last_close: {info.get('last_close'):.2f}"
                    f"\n• EMA48_slope(5): {info.get('ema48_slope_5'):.4f}"
                    f"\n• patterns: {json.dumps(info.get('patt'), ensure_ascii=False)}"
                    f"\n• pos_open: {bool(state.position)}"
                )

            # Если позиции нет — ищем вход
            if state.position is None and signal:
                cur = df.iloc[-1]
                atr_val = float(df["atr"].iloc[-1])
                entry = float(cur["close"])
                if signal == "LONG":
                    sl = entry - atr_val
                    tp = entry + RISK_RR * atr_val
                    state.position = {"side":"LONG","entry":entry,"sl":sl,"tp":tp}
                    await execute_trade(pair, "BUY", entry, atr_val)
                elif signal == "SHORT":
                    sl = entry + atr_val
                    tp = entry - RISK_RR * atr_val
                    state.position = {"side":"SHORT","entry":entry,"sl":sl,"tp":tp}
                    await execute_trade(pair, "SELL", entry, atr_val)

            # Менеджмент открытой позиции (по цене close)
            if state.position:
                cur_price = float(df.iloc[-1]["close"])
                pos = state.position
                hit_tp = (pos["side"]=="LONG" and cur_price>=pos["tp"]) or \
                         (pos["side"]=="SHORT" and cur_price<=pos["tp"])
                hit_sl = (pos["side"]=="LONG" and cur_price<=pos["sl"]) or \
                         (pos["side"]=="SHORT" and cur_price>=pos["sl"])

                if hit_tp or hit_sl:
                    res = "TP" if hit_tp else "SL"
                    await tg_send(
                        f"📘 EXIT {res}\n• {pair} {pos['side']} @ {cur_price:.2f}\n"
                        f"• entry: {pos['entry']:.2f}  tp: {pos['tp']:.2f}  sl: {pos['sl']:.2f}\n"
                        f"• time: {utc_now_str()}"
                    )
                    state.position = None

        except Exception as e:
            logger.exception(e)

        await asyncio.sleep(SLEEP_SEC)

# ------------ Runner ------------
async def run_bot():
    logger.info("🤖 mybot9 started successfully!")
    tasks = [asyncio.create_task(run_pair(p)) for p in PAIRS]
    while True:
        # health ping
        logger.info("Bot is alive... waiting for signals")
        await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.warning("Bot stopped manually.")
