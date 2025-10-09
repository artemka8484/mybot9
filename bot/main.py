# /workspace/bot/main.py
import os
import math
import json
import asyncio
from datetime import datetime, timezone
from collections import defaultdict

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from telegram import Update, constants
from telegram.ext import (
    Application, CommandHandler, ContextTypes, JobQueue
)

# ----------------------- ENV -----------------------
TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
PAIR_STR = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1m").strip().lower()  # 1m/5m/15m/1h
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))
TP_PCT = float(os.getenv("TP_PCT", "0.25"))  # %
SL_PCT = float(os.getenv("SL_PCT", "0.25"))  # %
DEBUG_TELEMETRY = os.getenv("DEBUG_TELEMETRY", "0") in ("1", "true", "yes")

# период проверки сигналов (сек)
TICK_SECONDS = int(os.getenv("TICK_SECONDS", "10"))

PAIRS = [p.strip().upper() for p in PAIR_STR.split(",") if p.strip()]
MEXC_KLINES = "https://api.mexc.com/api/v3/klines"

if not TOKEN or not CHAT_ID:
    raise RuntimeError("TELEGRAM_TOKEN и TELEGRAM_CHAT_ID должны быть заданы")

# ----------------------- STATE -----------------------
class PairState:
    def __init__(self, pair: str):
        self.pair = pair
        self.pos_open = False
        self.side = None         # "LONG"|"SHORT"
        self.entry = 0.0
        self.qty = 0.0
        self.tp = 0.0
        self.sl = 0.0
        # статистика
        self.closed_trades = 0
        self.wins = 0
        self.pnl_total = 0.0

    @property
    def winrate(self) -> float:
        return (self.wins / self.closed_trades * 100.0) if self.closed_trades else 0.0

# Глобальная статистика
GLOBAL = {
    "closed": 0,
    "wins": 0,
    "pnl": 0.0,
}

# Состояние по всем парам
STATES: dict[str, PairState] = {p: PairState(p) for p in PAIRS}

# ----------------------- UTILS -----------------------
def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fmt_money(x: float) -> str:
    return f"{x:.5f}"

def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def html_escape(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

# ----------------------- DATA -----------------------
async def fetch_klines(pair: str, interval: str = TIMEFRAME, limit: int = 100) -> pd.DataFrame:
    params = {"symbol": pair, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(MEXC_KLINES, params=params)
        r.raise_for_status()
        data = r.json()
    # MEXC может вернуть 8..12 колонок — берём только нужные индексы
    # [0]=open_time, [1]=open, [2]=high, [3]=low, [4]=close, [5]=volume
    rows = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in data]
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df

# ----------------------- STRATEGY #9 -----------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def detect_patterns(df: pd.DataFrame) -> dict:
    # Две последних свечи
    if len(df) < 3:
        return {"bull_engulf": False, "bear_engulf": False, "hammer": False, "shooting": False}

    o1, c1 = df["open"].iloc[-2], df["close"].iloc[-2]
    o2, c2, h2, l2 = df["open"].iloc[-1], df["close"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1]
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    rng2 = h2 - l2

    # Поглощения
    bull_engulf = (c1 < o1) and (c2 > o2) and (o2 <= c1) and (c2 >= o1) and (body2 > body1*1.1)
    bear_engulf = (c1 > o1) and (c2 < o2) and (o2 >= c1) and (c2 <= o1) and (body2 > body1*1.1)

    # “молот” / “падающая звезда”
    lower_wick = min(o2, c2) - l2
    upper_wick = h2 - max(o2, c2)
    hammer = (body2 > 0) and (lower_wick > body2*2.5) and (upper_wick < body2*0.5)
    shooting = (body2 > 0) and (upper_wick > body2*2.5) and (lower_wick < body2*0.5)

    return {
        "bull_engulf": bool(bull_engulf),
        "bear_engulf": bool(bear_engulf),
        "hammer": bool(hammer),
        "shooting": bool(shooting),
    }

def strategy9(df: pd.DataFrame) -> dict:
    """
    Возвращает сигнал:
    {"signal": "LONG"|"SHORT"|None, "reason": "...", "ema_slope": float, "patterns": {...}}
    """
    if len(df) < 60:
        return {"signal": None, "reason": "not_enough_bars", "ema_slope": 0.0, "patterns": {}}

    df = df.copy()
    df["ema48"] = ema(df["close"], 48)
    # угловая за последние 5 баров
    slope = float(df["ema48"].iloc[-1] - df["ema48"].iloc[-6])
    patt = detect_patterns(df)

    long_ok = slope > 0 and (patt["bull_engulf"] or patt["hammer"])
    short_ok = slope < 0 and (patt["bear_engulf"] or patt["shooting"])

    if long_ok and not short_ok:
        return {"signal": "LONG", "reason": "patt+ema_up", "ema_slope": slope, "patterns": patt}
    if short_ok and not long_ok:
        return {"signal": "SHORT", "reason": "patt+ema_down", "ema_slope": slope, "patterns": patt}
    return {"signal": None, "reason": "no_setup", "ema_slope": slope, "patterns": patt}

# ----------------------- TRADING (demo) -----------------------
async def open_position(ctx: ContextTypes.DEFAULT_TYPE, st: PairState, price: float, side: str):
    st.pos_open = True
    st.side = side
    st.entry = price
    st.qty = TRADE_SIZE
    if side == "LONG":
        st.tp = price * (1 + TP_PCT/100)
        st.sl = price * (1 - SL_PCT/100)
    else:
        st.tp = price * (1 - TP_PCT/100)
        st.sl = price * (1 + SL_PCT/100)

    icon = "🟢" if side == "LONG" else "🔴"
    await ctx.bot.send_message(
        CHAT_ID,
        (
            f"{icon} <b>OPEN</b> {st.pair} {side}\n"
            f"• time: {now_utc_str()}\n"
            f"• entry: <code>{fmt_money(st.entry)}</code>\n"
            f"• qty: <code>{st.qty}</code>\n"
            f"• TP: <code>{fmt_money(st.tp)}</code>  SL: <code>{fmt_money(st.sl)}</code>\n"
            f"• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
        ),
        parse_mode=constants.ParseMode.HTML
    )

async def close_position(ctx: ContextTypes.DEFAULT_TYPE, st: PairState, price: float, reason: str):
    if not st.pos_open:
        return
    side = st.side
    pnl = (price - st.entry) * st.qty if side == "LONG" else (st.entry - price) * st.qty
    st.pos_open = False
    st.side = None
    st.entry = 0.0
    st.qty = 0.0

    st.closed_trades += 1
    GLOBAL["closed"] += 1
    GLOBAL["pnl"] += pnl
    st.pnl_total += pnl
    win = pnl >= 0
    if win:
        st.wins += 1
        GLOBAL["wins"] += 1

    icon = "✅" if win else "❌"
    sign = "+" if pnl >= 0 else "−"
    await ctx.bot.send_message(
        CHAT_ID,
        (
            f"{icon} <b>CLOSE</b> {st.pair} ({reason})\n"
            f"• time: {now_utc_str()}\n"
            f"• exit: <code>{fmt_money(price)}</code>\n"
            f"• PnL: <b>{sign}{fmt_money(abs(pnl))}</b>\n"
            f"• pair stats: trades {st.closed_trades}, WR {fmt_pct(st.winrate)}, PnL {fmt_money(st.pnl_total)}\n"
            f"• total: trades {GLOBAL['closed']}, WR {fmt_pct((GLOBAL['wins']/GLOBAL['closed']*100) if GLOBAL['closed'] else 0)}, PnL {fmt_money(GLOBAL['pnl'])}"
        ),
        parse_mode=constants.ParseMode.HTML
    )

# ----------------------- JOB: per pair -----------------------
async def pair_job(ctx: ContextTypes.DEFAULT_TYPE):
    pair = ctx.job.data["pair"]
    st = STATES[pair]
    try:
        df = await fetch_klines(pair, TIMEFRAME, 150)
        if df.empty:
            return
        last = float(df["close"].iloc[-1])

        # если позиция открыта — проверяем TP/SL
        if st.pos_open:
            if st.side == "LONG":
                if last >= st.tp:
                    await close_position(ctx, st, last, "TP")
                    return
                if last <= st.sl:
                    await close_position(ctx, st, last, "SL")
                    return
            else:  # SHORT
                if last <= st.tp:
                    await close_position(ctx, st, last, "TP")
                    return
                if last >= st.sl:
                    await close_position(ctx, st, last, "SL")
                    return

        # иначе ищем новый сигнал
        sig = strategy9(df)
        if DEBUG_TELEMETRY and (sig["signal"] or (ctx.job.data["dbg_tick"] := (ctx.job.data.get("dbg_tick", 0)+1)) % 12 == 0):
            # раз в 12 тиков (≈2 минуты при 10s) или при сигнале
            await ctx.bot.send_message(
                CHAT_ID,
                (
                    f"🧪 <b>DEBUG</b>\n"
                    f"• pair: <b>{pair}</b>\n"
                    f"• last_close: <code>{fmt_money(last)}</code>\n"
                    f"• EMA48_slope(5 bars): <code>{sig['ema_slope']:.6f}</code>\n"
                    f"• patterns: <code>{json.dumps(sig['patterns'])}</code>\n"
                    f"• pos_open: <b>{st.pos_open}</b>"
                ),
                parse_mode=constants.ParseMode.HTML
            )

        if not st.pos_open and sig["signal"] in ("LONG", "SHORT"):
            await open_position(ctx, st, last, sig["signal"])

    except Exception as e:
        logger.exception(e)

# ----------------------- COMMANDS -----------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(
        f"✅ mybot9 running ({'DEMO' if DEMO_MODE else 'LIVE'})\n"
        f"PAIRS: <code>{', '.join(PAIRS)}</code>  TF: <b>{TIMEFRAME}</b>"
    )

def pair_status_line(st: PairState) -> str:
    pos = "—"
    if st.pos_open:
        pos = f"{st.side} @ {fmt_money(st.entry)} (TP {fmt_money(st.tp)}/ SL {fmt_money(st.sl)})"
    return (f"{st.pair}• trades: {st.closed_trades}  WR: {fmt_pct(st.winrate)}  "
            f"PnL: {fmt_money(st.pnl_total)}\n0.00000  pos: {pos}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["📊 <b>STATUS</b> " + now_utc_str()]
    for p in PAIRS:
        st = STATES[p]
        lines.append(pair_status_line(st))
    total_wr = (GLOBAL["wins"] / GLOBAL["closed"] * 100) if GLOBAL["closed"] else 0.0
    lines.append(f"—\nTOTAL • trades: {GLOBAL['closed']}  WR: {fmt_pct(total_wr)}  PnL: {fmt_money(GLOBAL['pnl'])}")
    await update.message.reply_html("\n".join(lines))

# ----------------------- MAIN -----------------------
def build_app() -> Application:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))

    # Планируем по джобам одну задачу на каждую пару
    for p in PAIRS:
        app.job_queue.run_repeating(
            pair_job,
            interval=TICK_SECONDS,
            first=2,  # старт через 2 сек после запуска
            data={"pair": p}
        )
    return app

def main():
    logger.info("🤖 mybot9 started successfully!")
    app = build_app()
    # Стартуем polling (инициализация и запуск делаются внутри)
    app.run_polling(allowed_updates=constants.Update.ALL_TYPES)

if __name__ == "__main__":
    main()
