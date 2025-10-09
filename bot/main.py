import os, asyncio, json, datetime, random
import pandas as pd
import httpx
from loguru import logger
from telegram import Bot

API_KEY = os.getenv("MEXC_API_KEY", "")
API_SECRET = os.getenv("MEXC_API_SECRET", "")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT").split(",")
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"

bot = Bot(token=TG_TOKEN)

async def send_message(text):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

async def get_klines(pair):
    url = f"https://api.mexc.com/api/v3/klines?symbol={pair}&interval={TIMEFRAME}&limit=100"
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close",
            "volume", "_1", "_2", "_3", "_4", "_5", "_6"])
        df["close"] = df["close"].astype(float)
        return df

async def strategy(pair):
    df = await get_klines(pair)
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    if df["sma5"].iloc[-2] < df["sma20"].iloc[-2] and df["sma5"].iloc[-1] > df["sma20"].iloc[-1]:
        return "BUY"
    if df["sma5"].iloc[-2] > df["sma20"].iloc[-2] and df["sma5"].iloc[-1] < df["sma20"].iloc[-1]:
        return "SELL"
    return None

async def run_bot():
    logger.info("🤖 mybot9 started successfully!")
    await send_message("✅ mybot9 is running with strategy #9 (DEMO)")
    trade_id = 0
    while True:
        for pair in PAIRS:
            signal = await strategy(pair)
            if signal:
                trade_id += 1
                price = round(random.uniform(60000, 60500), 2)
                ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                msg = f"📈 <b>DEMO trade #{trade_id}</b>\n" \
                      f"• {signal} {TRADE_SIZE} {pair} @ {price}\n" \
                      f"• Время: {ts}\n• Стратегия: #9\n• Исполнение: без реального ордера"
                logger.info(f"[DEMO] {signal} {pair} @ {price}")
                await send_message(msg)
            await asyncio.sleep(3)
        await asyncio.sleep(20)

if __name__ == "__main__":
    asyncio.run(run_bot())
