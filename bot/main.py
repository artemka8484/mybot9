# bot/main.py
import os
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from loguru import logger
from telegram import Bot

# ------------ ENV ------------
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
TX = os.getenv("TX", "UTC")
START_BALANCE = float(os.getenv("START_BALANCE_USDT", "1000"))

# ------------ TELEGRAM ------------
tg = Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

async def send(msg: str):
    if tg:
        try:
            await tg.send_message(CHAT_ID, msg, disable_web_page_preview=True)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

# ------------ DATA ------------
BASE = "https://api.mexc.com"

async def fetch_klines(pair: str, limit: int = 1000) -> pd.DataFrame:
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": pair, "interval": TIMEFRAME, "limit": limit}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    # MEXC вернёт 12 полей, но нам нужны 6
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "_1","_2","_3","_4","_5","close_time"
    ])
    df = df[["open_time","open","high","low","close","volume","close_time"]].copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df.dropna().reset_index(drop=True)

# ------------ STRATEGY #9 ------------
def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    if len(df) < 3:  # нужно минимум 3 свечи
        return {"bull_engulf": False, "bear_engulf": False, "hammer": False, "shooting": False}

    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()

    # индексы последних трёх свечей
    i2, i1, i0 = -3, -2, -1

    # Bullish engulfing: пред медв. (c<o), текущ быч. (c>o) и тело перекрывает предыдущее тело
    bull = (c[i1] < o[i1]) and (c[i0] > o[i0]) and (c[i0] - o[i0] > o[i1] - c[i1] > 0)

    # Bearish engulfing
    bear = (c[i1] > o[i1]) and (c[i0] < o[i0]) and (o[i0] - c[i0] > c[i1] - o[i1] > 0)

    # Простой hammer / shooting star по пропорциям теней
    body = abs(c[i0] - o[i0])
    upper = h[i0] - max(c[i0], o[i0])
    lower = min(c[i0], o[i0]) - l[i0]
    hammer = (lower > body * 2) and (upper < body)
    shooting = (upper > body * 2) and (lower < body)

    return {"bull_engulf": bool(bull), "bear_engulf": bool(bear), "hammer": bool(hammer), "shooting": bool(shooting)}

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def strategy9_signal(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Возвращает ('BUY'|'SELL'|'HOLD', info)
    Условия (упрощённо):
      - BUY: EMA48 наклон > 0 и bull_engulf/hammer
      - SELL: EMA48 наклон < 0 и bear_engulf/shooting
    """
    if len(df) < 60:
        return "HOLD", {}

    df = df.copy()
    df["ema48"] = ema(df["close"], 48)
    # наклон как разность последней и 5 свечей назад
    slope = float(df["ema48"].iloc[-1] - df["ema48"].iloc[-6])

    patt = detect_patterns(df)
    last_close = float(df["close"].iloc[-1])

    if slope > 0 and (patt["bull_engulf"] or patt["hammer"]):
        reason = "trend_up + pattern(bull_engulf/hammer)"
        return "BUY", {"last_close": last_close, "slope": slope, "pattern": next(k for k,v in patt.items() if v)}
    if slope < 0 and (patt["bear_engulf"] or patt["shooting"]):
        reason = "trend_down + pattern(bear_engulf/shooting)"
        return "SELL", {"last_close": last_close, "slope": slope, "pattern": next(k for k,v in patt.items() if v)}

    return "HOLD", {"last_close": last_close, "slope": slope, "pattern": None}

# ------------ EXECUTION & STATE (DEMO) ------------
class Portfolio:
    def __init__(self, start_balance: float):
        self.balance = start_balance  # USDT
        self.pos: Dict[str, Dict[str, Any]] = {}  # pair -> {side, qty, entry, entry_ts, reason}

    def position_open(self, pair: str) -> bool:
        return pair in self.pos

    def get_pos(self, pair: str) -> Dict[str, Any] | None:
        return self.pos.get(pair)

    def open_demo(self, pair: str, side: str, price: float, qty: float, reason: str):
        self.pos[pair] = {
            "side": side,
            "qty": qty,
            "entry": price,
            "entry_ts": datetime.now(timezone.utc),
            "reason_open": reason
        }

    def close_demo(self, pair: str, price: float) -> Dict[str, Any]:
        p = self.pos.pop(pair)
        pnl = (price - p["entry"]) * p["qty"] * (1 if p["side"] == "BUY" else -1)
        self.balance += pnl
        return {
            "side": p["side"],
            "qty": p["qty"],
            "entry": p["entry"],
            "entry_ts": p["entry_ts"],
            "exit": price,
            "exit_ts": datetime.now(timezone.utc),
            "pnl": pnl,
            "balance": self.balance,
            "reason_open": p.get("reason_open", "")
        }

portfolio = Portfolio(START_BALANCE)

def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

async def report_open(pair: str, side: str, qty: float, price: float, pattern: str, reason: str):
    msg = (
        f"🟩 OPEN {side}\n"
        f"• Pair: {pair}  TF: {TIMEFRAME}\n"
        f"• Qty: {qty}\n"
        f"• Entry: {price:.5f}\n"
        f"• Time: {fmt_ts(datetime.now(timezone.utc))}\n"
        f"• Pattern: {pattern or '—'}\n"
        f"• Reason: {reason}\n"
        f"• Mode: {'DEMO' if DEMO_MODE else 'REAL'}"
    )
    await send(msg)

async def report_close(pair: str, result: Dict[str, Any], pattern: str):
    side = result["side"]
    msg = (
        f"🟥 CLOSE {side}\n"
        f"• Pair: {pair}  TF: {TIMEFRAME}\n"
        f"• Qty: {result['qty']}\n"
        f"• Entry: {result['entry']:.5f}  →  Exit: {result['exit']:.5f}\n"
        f"• Open: {fmt_ts(result['entry_ts'])}\n"
        f"• Close: {fmt_ts(result['exit_ts'])}\n"
        f"• Pattern (exit): {pattern or '—'}\n"
        f"• PnL: {'+' if result['pnl']>=0 else ''}{result['pnl']:.4f} USDT\n"
        f"• Balance (DEMO): {result['balance']:.2f} USDT\n"
        f"• Open reason: {result.get('reason_open','')}"
    )
    await send(msg)

# ------------ WORKER PER PAIR ------------
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))

async def run_pair(pair: str):
    logger.info(f"worker started: {pair}")
    while True:
        try:
            df = await fetch_klines(pair, limit=400)
            signal, info = strategy9_signal(df)

            if signal == "BUY":
                if not portfolio.position_open(pair):
                    price = float(info["last_close"])
                    portfolio.open_demo(pair, "BUY", price, TRADE_SIZE, f"{info['pattern']} / slope>{info['slope']:.5f}")
                    await report_open(pair, "BUY", TRADE_SIZE, price, info["pattern"], "trend_up + pattern")
            elif signal == "SELL":
                if portfolio.position_open(pair):
                    # закрываем BUY (или игнор если вдруг позиция SELL — в демо мы открываем только BUY)
                    price = float(info["last_close"])
                    result = portfolio.close_demo(pair, price)
                    await report_close(pair, result, info["pattern"])

            # никаких heartbeat и отладочных сообщений
        except Exception as e:
            logger.exception(e)
        await asyncio.sleep(10)  # шаг цикла по паре

# ------------ BOOT ------------
async def run_bot():
    logger.info("🤖 mybot9 started successfully!")
    tasks = [asyncio.create_task(run_pair(p.strip())) for p in PAIRS if p.strip()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run_bot())
