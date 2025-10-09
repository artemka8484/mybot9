import os
import asyncio
import datetime
import random
import pandas as pd
import httpx
from loguru import logger
from telegram import Bot

# === Переменные окружения ===
API_KEY = os.getenv("MEXC_API_KEY", "")
API_SECRET = os.getenv("MEXC_API_SECRET", "")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT").split(",")
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"

bot = Bot(token=TG_TOKEN)


# === Telegram ===
async def send_message(text):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


# === Получение свечей ===
async def get_klines(pair):
    url = f"https://api.mexc.com/api/v3/klines?symbol={pair}&interval={TIMEFRAME}&limit=100"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        r.raise_for_status()
        raw = r.json()

        # Формируем DataFrame с безопасной структурой
        rows = []
        for k in raw:
            try:
                open_, high, low, close, volume = (
                    float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
                )
                rows.append([int(k[0]), open_, high, low, close, volume])
            except Exception:
                continue

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        return df


# === Простая стратегия SMA5/SMA20 ===
async def strategy(pair):
    df = await get_klines(pair)
    if len(df) < 25:
        return None

    df["sma5"] = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()

    if df["sma5"].iloc[-2] < df["sma20"].iloc[-2] and df["sma5"].iloc[-1] > df["sma20"].iloc[-1]:
        return "BUY"
    if df["sma5"].iloc[-2] > df["sma20"].iloc[-2] and df["sma5"].iloc[-1] < df["sma20"].iloc[-1]:
        return "SELL"
    return None


# === Главный цикл ===
async def run_bot():
    logger.info("🤖 mybot9 started successfully!")
    await send_message("✅ mybot9 is running with strategy #9 (DEMO mode active)")
    trade_id = 0

    while True:
        for pair in PAIRS:
            signal = await strategy(pair)
            if signal:
                trade_id += 1
                price = round(random.uniform(60000, 60500), 2)
                ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                msg = (
                    f"📈 <b>DEMO trade #{trade_id}</b>\n"
                    f"• {signal} {TRADE_SIZE} {pair} @ {price}\n"
                    f"• Время: {ts}\n• Стратегия: #9\n"
                    f"• Исполнение: без реального ордера"
                )
                logger.info(f"[DEMO] {signal} {pair} @ {price}")
                await send_message(msg)
            await asyncio.sleep(3)

        await asyncio.sleep(20)


if __name__ == "__main__":
    asyncio.run(run_bot())
