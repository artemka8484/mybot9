# bot/main.py
import os
import asyncio
from datetime import datetime, timezone
from loguru import logger
import httpx

# ---------- ENV ----------
PAIR = os.getenv("PAIR", "BTCUSDT")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")  # пока не используем, просто для вида
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))  # в базовой валюте (BTC при BTCUSDT)
START_BALANCE = float(os.getenv("DEMO_START_BALANCE", "10000"))  # USD(T)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------- Telegram helpers ----------
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

# ---------- Market data ----------
async def fetch_price(symbol: str) -> float:
    url = f"https://api.mexc.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return float(data["price"])

# ---------- Simple demo strategy (#9 placeholder) ----------
# тут просто чередуем BUY/SELL каждые ~12 сек, берём реальную последнюю цену
async def strategy_signal(counter: int) -> str:
    return "BUY" if counter % 2 == 1 else "SELL"

# ---------- DEMO portfolio state ----------
class DemoState:
    def __init__(self) -> None:
        self.usdt = START_BALANCE
        self.base = 0.0  # BTC количество
        self.entry_price = None  # цена входа для позиции (long)
        self.realized_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.losses = 0

    def summary_text(self) -> str:
        equity = self.usdt + (self.base * last_price if (last_price := 0) else self.usdt)
        # Для краткости, equity без mark-to-market, чтобы не дёргать цену тут
        return (
            f"💼 <b>DEMO summary</b>\n"
            f"• Balance: <b>{self.usdt:.2f} USDT</b>\n"
            f"• Realized PnL: <b>{self.realized_pnl:.2f} USDT</b>\n"
            f"• Trades: <b>{self.trades}</b> | Win-rate: <b>{(self.wins*100/max(1,self.trades)):.1f}%</b>"
        )

state = DemoState()

# ---------- Trade executor ----------
async def execute_trade(side: str, price: float) -> None:
    """
    В DEMO:
      BUY  -> покупаем TRADE_SIZE BTC по price (тратим USDT)
      SELL -> если есть позиция, продаём весь объём (или TRADE_SIZE, но ниже закрываем всю позицию)
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if DEMO_MODE:
        note = "без реального ордера"
        if side == "BUY":
            # Покупаем TRADE_SIZE BTC
            cost = TRADE_SIZE * price
            state.usdt -= cost
            state.base += TRADE_SIZE
            state.entry_price = price if state.entry_price is None else state.entry_price
            state.trades += 1
            await tg_send(
                "🧪 <b>DEMO trade</b>\n"
                f"• <b>BUY</b> {TRADE_SIZE:.3f} {PAIR.replace('USDT','')}/USDT @ <b>{price:.2f}</b>\n"
                f"• Время: <code>{ts}</code>\n"
                "• Стратегия: #9\n"
                f"• Исполнение: {note}"
            )
        else:  # SELL
            # Закрываем весь объём, если он есть; если нет — делаем короткую имитацию на TRADE_SIZE
            qty = state.base if state.base > 0 else TRADE_SIZE
            pnl = 0.0
            if state.base > 0 and state.entry_price is not None:
                pnl = qty * (price - state.entry_price)
            # Обновляем балансы
            state.usdt += qty * price
            state.base -= qty
            # PnL учитываем только при закрытии лонга
            if pnl != 0.0:
                state.realized_pnl += pnl
                if pnl > 0:
                    state.wins += 1
                else:
                    state.losses += 1
            state.entry_price = None if state.base <= 0 else state.entry_price
            state.trades += 1
            sign = "▲" if pnl > 0 else ("▼" if pnl < 0 else "•")
            await tg_send(
                "🧪 <b>DEMO trade</b>\n"
                f"• <b>SELL</b> {qty:.3f} {PAIR.replace('USDT','')}/USDT @ <b>{price:.2f}</b>\n"
                f"• Время: <code>{ts}</code>\n"
                "• Стратегия: #9\n"
                f"• PnL: <b>{sign} {pnl:.2f} USDT</b>\n"
                f"• Баланс: <b>{state.usdt:.2f} USDT</b>\n"
                f"• Исполнение: {note}"
            )
    else:
        # прод-вариант (реальные ордера) — не используется сейчас
        pass

# ---------- Main loop ----------
async def run_bot():
    logger.info("🤖 mybot9 started successfully!")
    if DEMO_MODE:
        await tg_send("✅ mybot9 is running with strategy #9\n(DEMO mode active)")

    counter = 0
    while True:
        try:
            price = await fetch_price(PAIR)
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}")
            await asyncio.sleep(5)
            continue

        counter += 1
        side = await strategy_signal(counter)
        await execute_trade(side, price)

        # Каждые 10 операций — краткий отчёт
        if DEMO_MODE and state.trades % 10 == 0:
            await tg_send(state.summary_text())

        logger.info("Bot is alive... waiting for signals")
        await asyncio.sleep(12)  # частота "сигналов" в демо

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.warning("Bot stopped manually.")
