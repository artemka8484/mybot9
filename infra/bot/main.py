import asyncio
import os
from loguru import logger
import websockets
import json

API_KEY = os.getenv("MEXC_API_KEY", "")
API_SECRET = os.getenv("MEXC_API_SECRET", "")

async def run_bot():
    logger.info("🤖 mybot9 started successfully!")
    while True:
        await asyncio.sleep(5)
        logger.info("Bot is alive... waiting for signals")

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.warning("Bot stopped manually.")
