# bot/main.py
import os
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import httpx
from loguru import logger
from telegram import Bot

# -----------------------
# Конфигурация из ENV
# -----------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in {"1", "true", "yes", "on"}
DEBUG_TELEMETRY = os.getenv("DEBUG_TELEMETRY", "0") in {"1", "true", "yes", "on"}
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))

# Пары: через запятую. Пример: "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
PAIRS = [
    p.strip().upper()
    for p in os.getenv(
        "PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
    ).split(",")
    if p.strip()
]

# Таймфрейм: "1m", "5m", "15m" ... (для MEXC)
TIMEFRAME = os.getenv("TIMEFRAME", "1m").lower()

# Пауза между циклами по каждой паре (сек). Делаем короче таймфрейма.
TF_TO_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}
CYCLE_SLEEP = max(5, min(20, TF_TO_SECONDS.get(TIMEFRAME, 60) // 3))

# Глобальный Telegram-бот
tg_bot: Bot | None = None


# -----------------------
# HTTP-сервер для healthcheck (порт 8080)
# -----------------------
class _Ping(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def log_message(self, *_args, **_kwargs):
        # глушим стандартный спам http.server
        return


def _run_http():
    srv = HTTPServer(("0.0.0.0", 8080), _Ping)
    logger.info("HTTP health server started on :8080")
    srv.serve_forever()


# -----------------------
# Хелперы
# -----------------------
async def send_telegram(text: str):
    """Безопасно отправляем сообщение в Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        await tg_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


async def fetch_klines(symbol: str, interval: str, limit: int = 120) -> pd.DataFrame:
    """
    Берём свечи с MEXC (v3), формат массива: 8 полей на свечу.
    https://api.mexc.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=120
    """
    url = "https://api.mexc.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": str(limit)}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()  # list[list]
    # MEXC kline -> 8 колонок
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
    ]
    df = pd.DataFrame(data, columns=cols)
    # приведение типов
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def candle_patterns(df: pd.DataFrame) -> dict:
    """
    Очень простые (игрушечные) правила паттернов:
    - бычье поглощение
    - медвежье поглощение
    - молот
    - падающая звезда
    """
    if len(df) < 3:
        return {"bull_engulf": False, "bear_engulf": False, "hammer": False, "shooting": False}

    c1_open, c1_close, c1_high, c1_low = (
        float(df["open"].iloc[-2]),
        float(df["close"].iloc[-2]),
        float(df["high"].iloc[-2]),
        float(df["low"].iloc[-2]),
    )
    c2_open, c2_close, c2_high, c2_low = (
        float(df["open"].iloc[-1]),
        float(df["close"].iloc[-1]),
        float(df["high"].iloc[-1]),
        float(df["low"].iloc[-1]),
    )

    # engulfing: тело текущей свечи длиннее и покрывает пред.тело
    bull_engulf = (c2_close > c2_open) and (c1_close < c1_open) and (c2_close >= c1_open) and (c2_open <= c1_close)
    bear_engulf = (c2_close < c2_open) and (c1_close > c1_open) and (c2_open >= c1_close) and (c2_close <= c1_open)

    # hammer / shooting star: соотношения теней (очень грубо)
    body = abs(c2_close - c2_open)
    upper = c2_high - max(c2_close, c2_open)
    lower = min(c2_close, c2_open) - c2_low

    hammer = (lower > body * 2) and (upper < body * 0.5)
    shooting = (upper > body * 2) and (lower < body * 0.5)

    return {
        "bull_engulf": bool(bull_engulf),
        "bear_engulf": bool(bear_engulf),
        "hammer": bool(hammer),
        "shooting": bool(shooting),
    }


def strategy9(df: pd.DataFrame) -> tuple[str | None, dict]:
    """
    Правила #9:
    - EMA48, EMA14
    - Угол (наклон) EMA48 по последним 5 барам
    - Сигнал BUY: наклон > 0 и (bull_engulf или hammer)
    - Сигнал SELL: наклон < 0 и (bear_engulf или shooting)
    Возвращает (signal, info)
    """
    if len(df) < 60:
        return None, {}

    df = df.copy()
    df["ema14"] = ema(df["close"], 14)
    df["ema48"] = ema(df["close"], 48)

    # Наклон EMA48: линейная регрессия по 5 последним точкам
    tail = df["ema48"].tail(5).to_numpy(dtype=float)
    x = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])  # коэффициент при x

    patt = candle_patterns(df)

    signal = None
    if slope > 0 and (patt["bull_engulf"] or patt["hammer"]):
        signal = "BUY"
    elif slope < 0 and (patt["bear_engulf"] or patt["shooting"]):
        signal = "SELL"

    info = {
        "last_close": float(df["close"].iloc[-1]),
        "ema48_slope_5": float(slope),
        "patt": {k: bool(v) for k, v in patt.items()},  # гарантируем обычные bool
    }
    return signal, info


async def execute_demo(symbol: str, side: str, price: float):
    """Фейковое исполнение сделки (демо)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = (
        f"🧪 DEMO trade • {symbol}"
        f"\n• {side} {TRADE_SIZE:g} @ {price:.4f}"
        f"\n• Время: {ts}"
        f"\n• Стратегия: #9"
        f"\n• Исполнение: без реального ордера"
    )
    await send_telegram(text)


async def run_pair(symbol: str):
    """Цикл по конкретной паре."""
    pos_open = False  # простейшее состояние позиции
    await send_telegram(f"✅ mybot9 running (DEMO) PAIR: {symbol} TF: {TIMEFRAME}")

    while True:
        try:
            df = await fetch_klines(symbol, TIMEFRAME, limit=150)
            signal, info = strategy9(df)

            # DEBUG-вывод (только если включен)
            if DEBUG_TELEMETRY:
                patt = info.get("patt", {})
                # json.dumps падал на numpy.bool_ -> приводим заранее
                patt = {k: bool(v) for k, v in patt.items()}
                debug_text = (
                    "🧪 DEBUG"
                    f"\n• pair: {symbol}"
                    f"\n• last_close: {float(info.get('last_close', 0)):.5f}"
                    f"\n• EMA48_slope(5 bars): {float(info.get('ema48_slope_5', 0)):.6f}"
                    f"\n• patterns: {json.dumps(patt, ensure_ascii=False)}"
                    f"\n• pos_open: {bool(pos_open)}"
                )
                await send_telegram(debug_text)

            # Простейшая логика позиций (для демо)
            if signal == "BUY" and not pos_open:
                await execute_demo(symbol, "BUY", float(info.get("last_close", 0.0)))
                pos_open = True
            elif signal == "SELL" and pos_open:
                await execute_demo(symbol, "SELL", float(info.get("last_close", 0.0)))
                pos_open = False

        except Exception as e:
            # Лог + уведомление, но не падаем
            logger.exception(e)
            await send_telegram(f"⚠️ Ошибка по паре {symbol}: {e}")

        await asyncio.sleep(CYCLE_SLEEP)


async def run_bot():
    global tg_bot
    logger.info("🤖 mybot9 started successfully!")
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        tg_bot = Bot(TELEGRAM_TOKEN)
        mode_note = "(DEMO mode active)" if DEMO_MODE else "(LIVE)"
        await send_telegram(f"✅ mybot9 is running with strategy #9 {mode_note}")
    else:
        logger.warning("TELEGRAM_* не заданы — оповещения отключены.")

    # запускаем циклы по всем парам
    tasks = []
    delay = 0
    for p in PAIRS:
        # Небольшой сдвиг старта, чтобы не фетчить всё одновременно
        async def _delayed(pair=p, d=delay):
            await asyncio.sleep(d)
            await run_pair(pair)
        tasks.append(asyncio.create_task(_delayed()))
        delay += 2  # по 2 секунды сдвига

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # HTTP health server в отдельном потоке
    threading.Thread(target=_run_http, daemon=True).start()
    asyncio.run(run_bot())
