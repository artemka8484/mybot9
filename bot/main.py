# /workspace/bot/main.py
import os
import asyncio
import json
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import httpx
import numpy as np
import pandas as pd
from dateutil import tz
from loguru import logger

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------
BOT_NAME = os.getenv("BOT_NAME", "mybot9")
TZ_NAME = os.getenv("TX", "UTC")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")  # 1m/5m/15m/1h
PAIRS = [p.strip().upper() for p in os.getenv(
    "PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
).split(",") if p.strip()]

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in {"1", "true", "yes"}
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

DEBUG_TELEMETRY = os.getenv("DEBUG_TELEMETRY", "0") in {"1", "true", "yes"}

# стратегия #9 — параметры входа/выхода (можешь менять)
EMA_PERIOD = 48
EMA_SLOPE_BARS = 5
SLOPE_LONG = 0.05     # насколько позитивный наклон для лонга
SLOPE_SHORT = -0.05   # негативный наклон для шорта
TP_PCT = 0.30 / 100.0   # 0.30%
SL_PCT = 0.25 / 100.0   # 0.25%

# -----------------------------------------------------------------------------
# Хранилище состояния (в RAM). Если нужно переживать рестарт — подключим Redis.
# -----------------------------------------------------------------------------
class PairStats:
    def __init__(self) -> None:
        self.trades_closed: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.realized_pnl_usdt: float = 0.0
        self.last_result: Optional[str] = None  # "WIN" | "LOSS"
        self.last_pnl_usdt: Optional[float] = None
        self.last_pnl_pct: Optional[float] = None
        self.last_closed_at: Optional[str] = None

    @property
    def winrate(self) -> float:
        if self.trades_closed == 0:
            return 0.0
        return (self.wins / self.trades_closed) * 100.0


class Position:
    def __init__(self, side: str, entry: float, size: float, tp: float, sl: float) -> None:
        self.side = side  # "LONG" | "SHORT"
        self.entry = entry
        self.size = size
        self.tp = tp
        self.sl = sl
        self.opened_at = now_utc_str()

    def as_dict(self) -> Dict[str, Any]:
        return dict(side=self.side, entry=self.entry, size=self.size, tp=self.tp, sl=self.sl, opened_at=self.opened_at)


class State:
    def __init__(self, pairs: List[str]) -> None:
        self.positions: Dict[str, Optional[Position]] = {p: None for p in pairs}
        self.stats: Dict[str, PairStats] = {p: PairStats() for p in pairs}
        self.total_realized_pnl: float = 0.0

    def total_trades(self) -> int:
        return int(sum(self.stats[p].trades_closed for p in self.stats))

    def total_wins(self) -> int:
        return int(sum(self.stats[p].wins for p in self.stats))

    def total_winrate(self) -> float:
        t = self.total_trades()
        return (self.total_wins() / t) * 100.0 if t else 0.0


state = State(PAIRS)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a / b) * 100.0


def fmt_usd(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.5f}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}%"


def result_emoji(win: Optional[str]) -> str:
    if win == "WIN":
        return "✅"
    if win == "LOSS":
        return "❌"
    return "➖"


# -----------------------------------------------------------------------------
# Market data: MEXC kline public
# -----------------------------------------------------------------------------
MEXC_BASE = "https://api.mexc.com"

async def get_klines(symbol: str, interval: str, limit: int = 120) -> pd.DataFrame:
    url = f"{MEXC_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    # MEXC возвращает 12 полей. Нам нужны: open time, open, high, low, close, volume
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "_c","_d","_e","_f","_g","close_time"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open_time","open","high","low","close","volume"]]


# -----------------------------------------------------------------------------
# Strategy #9
# -----------------------------------------------------------------------------
def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    # simple candlestick patterns for the last bar
    if len(df) < 3:
        return {"bull_engulf": False, "bear_engulf": False, "hammer": False, "shooting": False}

    o1, c1 = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
    o2, c2 = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    h2, l2 = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])

    bull_engulf = (c1 < o1) and (c2 > o2) and (c2 >= o1) and (o2 <= c1)
    bear_engulf = (c1 > o1) and (c2 < o2) and (o2 >= c1) and (c2 <= o1)

    body = abs(c2 - o2)
    upper = h2 - max(c2, o2)
    lower = min(c2, o2) - l2
    small = (body > 0) and (upper / body > 2.5 or lower / body > 2.5)

    hammer = small and lower > (2 * body) and upper < body
    shooting = small and upper > (2 * body) and lower < body

    return {
        "bull_engulf": bool(bull_engulf),
        "bear_engulf": bool(bear_engulf),
        "hammer": bool(hammer),
        "shooting": bool(shooting),
    }


def decision_from_indicators(df: pd.DataFrame) -> str:
    # returns "LONG" | "SHORT" | "FLAT"
    if len(df) < max(EMA_PERIOD, EMA_SLOPE_BARS + 1):
        return "FLAT"

    ema = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    last = float(df["close"].iloc[-1])

    # slope of ema over last EMA_SLOPE_BARS
    y2 = float(ema.iloc[-1])
    y1 = float(ema.iloc[-1 - EMA_SLOPE_BARS])
    slope = (y2 - y1) / EMA_SLOPE_BARS  # абсолютный шаг
    slope_pct = pct(slope, y1)

    patt = detect_patterns(df)

    go_long = (slope_pct >= SLOPE_LONG) and (patt["bull_engulf"] or patt["hammer"])
    go_short = (slope_pct <= SLOPE_SHORT) and (patt["bear_engulf"] or patt["shooting"])

    if go_long:
        return "LONG"
    if go_short:
        return "SHORT"
    return "FLAT"


def calc_tp_sl(entry: float, side: str) -> Tuple[float, float]:
    if side == "LONG":
        tp = entry * (1 + TP_PCT)
        sl = entry * (1 - SL_PCT)
    else:
        tp = entry * (1 - TP_PCT)
        sl = entry * (1 + SL_PCT)
    return (tp, sl)


# -----------------------------------------------------------------------------
# Telegram
# -----------------------------------------------------------------------------
async def tg_send(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    if not TELEGRAM_TOKEN or TELEGRAM_CHAT_ID == 0:
        return
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)


def status_block_for_pair(pair: str) -> str:
    st = state.stats[pair]
    pos = state.positions[pair]
    wr = st.winrate
    pnl = st.realized_pnl_usdt

    last_line = ""
    if st.last_result is not None:
        last_line = (f"\n  last: {result_emoji(st.last_result)} "
                     f"{fmt_usd(st.last_pnl_usdt)} USDT ({fmt_pct(st.last_pnl_pct)}) "
                     f"at {st.last_closed_at or '—'}")

    pos_line = "—"
    if pos:
        pos_line = (f"{pos.side} @ {pos.entry:.5f} "
                    f"(TP {pos.tp:.5f} / SL {pos.sl:.5f})")

    return (
        f"{pair} • trades: {st.trades_closed}  WR: {wr:.1f}%  PnL: {fmt_usd(pnl)}"
        f"\n  pos: {pos_line}{last_line}"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    header = f"📊 STATUS {now_utc_str()}"
    lines = [header]
    for p in PAIRS:
        lines.append(status_block_for_pair(p))
    lines.append("\n—"*20)
    lines.append(f"TOTAL • trades: {state.total_trades()}  WR: {state.total_winrate():.1f}%"
                 f"  PnL: {fmt_usd(state.total_realized_pnl)}")
    await tg_send(context, "\n".join(lines))


# -----------------------------------------------------------------------------
# Execution (DEMO): эмулируем исполнение по TP/SL
# -----------------------------------------------------------------------------
def unrealized_pnl(entry: float, side: str, last: float, size: float) -> float:
    if side == "LONG":
        return (last - entry) * size
    else:
        return (entry - last) * size


def realized_pnl(entry: float, exit: float, side: str, size: float) -> float:
    if side == "LONG":
        return (exit - entry) * size
    else:
        return (entry - exit) * size


async def process_pair(pair: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    df = await get_klines(pair, TIMEFRAME, limit=120)
    last_close = float(df["close"].iloc[-1])

    pos = state.positions[pair]

    # 1) Если позиция открыта — проверим TP/SL (для DEMO симуляции)
    if pos is not None:
        closed = False
        exit_price = None
        result = None  # "WIN" | "LOSS"

        if pos.side == "LONG":
            if last_close >= pos.tp:
                closed = True
                exit_price = pos.tp
                result = "WIN"
            elif last_close <= pos.sl:
                closed = True
                exit_price = pos.sl
                result = "LOSS"
        else:  # SHORT
            if last_close <= pos.tp:
                closed = True
                exit_price = pos.tp
                result = "WIN"
            elif last_close >= pos.sl:
                closed = True
                exit_price = pos.sl
                result = "LOSS"

        if closed and exit_price is not None:
            pnl = realized_pnl(pos.entry, exit_price, pos.side, pos.size)
            pnl_pct = pct(abs(exit_price - pos.entry), pos.entry) * (1 if result == "WIN" else 1)  # просто информативно

            st = state.stats[pair]
            st.trades_closed += 1
            if result == "WIN":
                st.wins += 1
            else:
                st.losses += 1
            st.realized_pnl_usdt += pnl
            st.last_result = result
            st.last_pnl_usdt = pnl
            st.last_pnl_pct = pnl_pct if pos.side == "LONG" else pnl_pct  # симметрично
            st.last_closed_at = now_utc_str()
            state.total_realized_pnl += pnl

            emoji = "✅" if result == "WIN" else "❌"
            msg = (
                f"{emoji} {pair} CLOSED {pos.side} {pos.size:g} @ {exit_price:.5f}\n"
                f"• entry: {pos.entry:.5f}\n"
                f"• PnL: {pnl:+.5f} USDT ({fmt_pct(st.last_pnl_pct)})\n"
                f"• balance: {fmt_usd(state.total_realized_pnl)} USDT\n"
                f"• trades: {st.trades_closed}  WR(pair): {st.winrate:.1f}%  WR(total): {state.total_winrate():.1f}%"
            )
            await tg_send(context, msg)

            # позицию закрыли
            state.positions[pair] = None
            pos = None

    # 2) Если позиции нет — ищем сигнал
    if pos is None:
        side = decision_from_indicators(df)
        if side in {"LONG", "SHORT"}:
            entry = last_close
            tp, sl = calc_tp_sl(entry, side)
            size = TRADE_SIZE  # DEMO размер

            state.positions[pair] = Position(side, entry, size, tp, sl)
            await tg_send(
                context,
                f"🟢 OPEN {pair} {side} {size:g} @ {entry:.5f} "
                f"(TP {tp:.5f} / SL {sl:.5f})"
            )

    # 3) Отладка (по желанию)
    if DEBUG_TELEMETRY:
        patt = detect_patterns(df)
        ema = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        y2 = float(ema.iloc[-1])
        y1 = float(ema.iloc[-1 - EMA_SLOPE_BARS]) if len(ema) > EMA_SLOPE_BARS else y2
        slope_pct = pct((y2 - y1) / EMA_SLOPE_BARS, y1)

        pos_str = "—" if state.positions[pair] is None else f"{state.positions[pair].side} @ {state.positions[pair].entry:.5f}"
        dbg = (
            f"🧪 DEBUG {pair}\n"
            f"• last_close: {last_close:.5f}\n"
            f"• EMA{safestr(EMA_PERIOD)}_slope({EMA_SLOPE_BARS} bars): {slope_pct:.6f}\n"
            f"• patterns: {json.dumps({k: bool(v) for k, v in patt.items()}, ensure_ascii=False)}\n"
            f"• pos: {pos_str}"
        )
        await tg_send(context, dbg)


def safestr(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "?"


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------
async def run_loop(app: Application) -> None:
    # единый тикер: раз в 10 секунд обрабатываем все пары (последовательно)
    while True:
        try:
            for p in PAIRS:
                await process_pair(p, app.bot._application_context)  # type: ignore
        except Exception as e:
            logger.exception(e)
        await asyncio.sleep(10)


# -----------------------------------------------------------------------------
# Telegram wiring
# -----------------------------------------------------------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await tg_send(context, f"✅ {BOT_NAME} is running with strategy #9 "
                           f"({'DEMO' if DEMO_MODE else 'LIVE'})\nPAIRS: {', '.join(PAIRS)} TF: {TIMEFRAME}")

def build_application() -> Application:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", cmd_status))
    return app


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
async def run_bot() -> None:
    logger.info(f"🤖 {BOT_NAME} started successfully!")

    if not TELEGRAM_TOKEN or TELEGRAM_CHAT_ID == 0:
        logger.warning("Telegram credentials missing; the bot will run headless.")

    app = build_application()

    # ВАЖНО: initialize → start → post_init tasks → run loop
    await app.initialize()
    await app.start()

    # Пихаем bot контекст для удобства в process_pair
    app.bot._application_context = ContextTypes.DEFAULT_TYPE(application=app)  # type: ignore

    # Пингуем старт
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                   text=f"✅ {BOT_NAME} booted. Mode: {'DEMO' if DEMO_MODE else 'LIVE'}. TF: {TIMEFRAME}.")
    except Exception:
        pass

    # Поднимаем основной цикл
    loop_task = asyncio.create_task(run_loop(app))

    # Полноценный idle
    try:
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)  # типовой idle в PTB v21
    except Exception:
        # Если updater не сконфигурирован (webhook/healthcheck) — просто держим фоновые задачи
        await loop_task

if __name__ == "__main__":
    asyncio.run(run_bot())
