# bot/main.py
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, Literal

import httpx
import pandas as pd
from loguru import logger

from bot.strategy9 import decide  # наша логика сигналов (EMA48 + свечные + ATR)

# ====== ENV ======
PAIR = os.getenv("PAIR", "BTCUSDT")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")           # 1m/5m/15m/1h...
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"     # 1 = демо
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))
DEMO_START_BALANCE = float(os.getenv("DEMO_START_BALANCE", "10000"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ====== Telegram ======
async def tg_send(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(url, json=payload)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ====== MEXC market data (public) ======
async def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    url = "https://api.mexc.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        raw = r.json()  # список списков
    rows = []
    for k in raw:
        try:
            # [0] openTime, [1] open, [2] high, [3] low, [4] close, [5] volume, [6] closeTime, [7] quoteVolume, ...
            open_, high, low, close, volume = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
            rows.append([int(k[0]), open_, high, low, close, volume])
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    return df

# ====== DEMO portfolio / position ======
Side = Literal["LONG", "SHORT"]

class Position:
    def __init__(self) -> None:
        self.side: Optional[Side] = None
        self.qty: float = 0.0
        self.entry: Optional[float] = None
        self.tp: Optional[float] = None
        self.sl: Optional[float] = None
        self.reason: str = ""

    def is_open(self) -> bool:
        return self.side is not None and self.qty > 0 and self.entry is not None

    def reset(self) -> None:
        self.side = None
        self.qty = 0.0
        self.entry = None
        self.tp = None
        self.sl = None
        self.reason = ""

class DemoAccount:
    def __init__(self, start_balance: float) -> None:
        self.usdt = start_balance
        self.realized_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pos = Position()

    def summary(self) -> str:
        wr = (self.wins * 100 / self.trades) if self.trades > 0 else 0.0
        return (
            f"💼 <b>DEMO summary</b>\n"
            f"• Balance: <b>{self.usdt:.2f} USDT</b>\n"
            f"• Realized PnL: <b>{self.realized_pnl:.2f} USDT</b>\n"
            f"• Trades: <b>{self.trades}</b> | Win-rate: <b>{wr:.1f}%</b>"
        )

account = DemoAccount(DEMO_START_BALANCE)

# ====== DEMO execution (симуляция TP/SL) ======
async def open_position(price: float, side: Side, atr_value: float, reason: str) -> None:
    if account.pos.is_open():
        return  # игнор, пока позиция открыта
    qty = TRADE_SIZE
    account.pos.side = side
    account.pos.qty = qty
    account.pos.entry = price
    # SL=0.5*ATR, TP=1.0*ATR от цены входа
    if side == "LONG":
        account.pos.tp = price + 1.0 * atr_value
        account.pos.sl = price - 0.5 * atr_value
    else:
        account.pos.tp = price - 1.0 * atr_value
        account.pos.sl = price + 0.5 * atr_value
    account.pos.reason = reason

    await tg_send(
        "🚀 <b>OPEN</b>\n"
        f"• {side} {qty} {PAIR}\n"
        f"• Entry: <b>{price:.2f}</b> | TP: <b>{account.pos.tp:.2f}</b> | SL: <b>{account.pos.sl:.2f}</b>\n"
        f"• Reason: <code>{reason}</code>\n"
        f"• Time: <code>{now_utc()}</code>\n"
        "• Mode: DEMO"
    )

async def try_close_position(last_close: float) -> None:
    if not account.pos.is_open():
        return
    side = account.pos.side
    entry = account.pos.entry or last_close
    tp = account.pos.tp or last_close
    sl = account.pos.sl or last_close
    qty = account.pos.qty

    hit_tp = (last_close >= tp) if side == "LONG" else (last_close <= tp)
    hit_sl = (last_close <= sl) if side == "LONG" else (last_close >= sl)
    if not (hit_tp or hit_sl):
        return

    # Исполнение по цене срабатывания (берём last_close)
    exit_price = last_close
    pnl = (exit_price - entry) * qty if side == "LONG" else (entry - exit_price) * qty
    account.usdt += pnl
    account.realized_pnl += pnl
    account.trades += 1
    if pnl > 0:
        account.wins += 1
    elif pnl < 0:
        account.losses += 1

    tag = "✅ TP" if hit_tp else "⛔ SL"
    await tg_send(
        f"{tag} <b>CLOSE</b>\n"
        f"• {side} {qty} {PAIR}\n"
        f"• Exit: <b>{exit_price:.2f}</b>\n"
        f"• PnL: <b>{pnl:+.2f} USDT</b>\n"
        f"• Balance: <b>{account.usdt:.2f} USDT</b>\n"
        f"• Time: <code>{now_utc()}</code>"
    )
    account.pos.reset()

# ====== Main loop ======
async def run_bot():
    logger.info("🤖 mybot9 started successfully!")
    await tg_send(f"✅ mybot9 running (DEMO)\nPAIR: <b>{PAIR}</b> TF: <b>{TIMEFRAME}</b>")

    while True:
        try:
            df = await get_klines(PAIR, TIMEFRAME, limit=120)
        except Exception as e:
            logger.warning(f"Klines fetch failed: {e}")
            await asyncio.sleep(5)
            continue

        if len(df) < 60:
            await asyncio.sleep(5)
            continue

        # 1) если позиция открыта — проверяем TP/SL по последней цене
        last_close = float(df["close"].iloc[-1])
        await try_close_position(last_close)

        # 2) если позиции нет — ищем вход по стратегии #9
        if not account.pos.is_open():
            sig = decide(df)  # {'side': 'LONG'|'SHORT', 'reason': str, 'atr': float} или None
            if sig and DEMO_MODE:
                await open_position(price=last_close, side=sig["side"], atr_value=float(sig["atr"]), reason=sig["reason"])

        # каждые 20 итераций — сводка
        if account.trades > 0 and account.trades % 20 == 0 and not account.pos.is_open():
            await tg_send(account.summary())

        logger.info("Bot is alive... waiting for signals")
        await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.warning("Bot stopped manually.")
