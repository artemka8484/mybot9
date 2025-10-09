# /workspace/bot/main.py
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import httpx
from loguru import logger

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ========= ENV =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

DEMO_MODE = str(os.getenv("DEMO_MODE", "true")).lower() in ("1", "true", "yes")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
PAIRS = [p.strip().upper() for p in os.getenv(
    "PAIRS",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
).split(",") if p.strip()]

TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))       # размер базового актива
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.3"))  # % от цены (пример: 0.3%)
STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT", "0.3"))    # % от цены
DEBUG_TELEMETRY = str(os.getenv("DEBUG_TELEMETRY", "0")) in ("1", "true", "yes")

# окно EMA и сколько баров берём для наклона
EMA_LEN = int(os.getenv("EMA_LEN", "48"))
SLOPE_BARS = int(os.getenv("SLOPE_BARS", "5"))

# период опроса свечей (сек)
POLL_SEC = int(os.getenv("POLL_SEC", "10"))

# ========= STATE =========
class Position:
    def __init__(self):
        self.is_open: bool = False
        self.side: Optional[str] = None  # "LONG" / "SHORT"
        self.entry_price: float = 0.0
        self.size: float = 0.0
        self.tp: float = 0.0
        self.sl: float = 0.0
        self.open_ts: Optional[datetime] = None
        self.reason: str = ""  # какой паттерн/сигнал открыл позицию

class Stats:
    def __init__(self):
        self.trades:int = 0
        self.wins:int = 0
        self.losses:int = 0
        self.realized_pnl_usdt: float = 0.0

    @property
    def winrate(self)->float:
        if self.trades == 0:
            return 0.0
        return (self.wins / self.trades) * 100.0

pair_pos: Dict[str, Position] = {p: Position() for p in PAIRS}
pair_stats: Dict[str, Stats] = {p: Stats() for p in PAIRS}

# ========= HELPERS =========
def utc_now_str()->str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

async def tg_send(app: Application, text: str):
    if TELEGRAM_CHAT_ID:
        try:
            await app.bot.send_message(
                TELEGRAM_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

def pct(val: float)->str:
    return f"{val:.2f}%"

def fmt(n: float, digits: int = 5)->str:
    return f"{n:.{digits}f}"

def patterns(df: pd.DataFrame)->Dict[str, bool]:
    """простые свечные паттерны"""
    # свечи
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    def bull_engulf(i):
        if i < 1: return False
        prev_red = c[i-1] < o[i-1]
        cur_green = c[i] > o[i]
        engulf = (o[i] <= c[i-1]) and (c[i] >= o[i-1])
        return bool(prev_red and cur_green and engulf)

    def bear_engulf(i):
        if i < 1: return False
        prev_green = c[i-1] > o[i-1]
        cur_red = c[i] < o[i]
        engulf = (o[i] >= c[i-1]) and (c[i] <= o[i-1])
        return bool(prev_green and cur_red and engulf)

    def hammer(i):
        body = abs(c[i]-o[i])
        lower = o[i]-l[i] if c[i]>=o[i] else c[i]-l[i]
        upper = h[i]-c[i] if c[i]>=o[i] else h[i]-o[i]
        return bool(lower > body*2 and upper < body)

    def shooting(i):
        body = abs(c[i]-o[i])
        upper = h[i]-c[i] if c[i]>=o[i] else h[i]-o[i]
        lower = o[i]-l[i] if c[i]>=o[i] else c[i]-l[i]
        return bool(upper > body*2 and lower < body)

    i = len(df)-1
    return {
        "bull_engulf": bull_engulf(i),
        "bear_engulf": bear_engulf(i),
        "hammer": hammer(i),
        "shooting": shooting(i)
    }

def decide_signal(df: pd.DataFrame)->Dict[str, Any]:
    """стратегия #9: EMA48 + наклон + паттерны
       Возвращает: {'action': 'BUY'/'SELL'/'HOLD', 'reason': '...'}"""
    if len(df) < EMA_LEN + SLOPE_BARS + 1:
        return {"action":"HOLD", "reason":"warmup"}

    close = df["close"]
    ema = close.ewm(span=EMA_LEN, adjust=False).mean()
    slope = ema.diff().tail(SLOPE_BARS).sum()  # суммарный наклон последних SLOPE_BARS

    patt = patterns(df)
    last = close.iloc[-1]

    # условные правила:
    # Buy: цена > EMA и наклон положительный И (бычье поглощение или молот)
    if last > ema.iloc[-1] and slope > 0 and (patt["bull_engulf"] or patt["hammer"]):
        return {"action":"BUY", "reason":"EMA_up + pattern(bull/hammer)"}
    # Sell (short): цена < EMA и наклон отрицательный И (медвежье поглощение или падающая звезда)
    if last < ema.iloc[-1] and slope < 0 and (patt["bear_engulf"] or patt["shooting"]):
        return {"action":"SELL", "reason":"EMA_down + pattern(bear/shooting)"}

    return {"action":"HOLD", "reason":"no edge"}

async def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """MEXC public klines (без ключей)"""
    url = "https://api.mexc.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": str(limit)}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        raw: List[List[Any]] = r.json()  # [open_time,open,high,low,close,volume, ...]
    if not raw:
        raise RuntimeError("empty klines")

    arr = []
    for row in raw:
        arr.append({
            "time":   int(row[0])//1000,
            "open":   float(row[1]),
            "high":   float(row[2]),
            "low":    float(row[3]),
            "close":  float(row[4]),
            "volume": float(row[5]),
        })
    df = pd.DataFrame(arr)
    return df

def setup_tp_sl(entry: float, side: str)->(float, float):
    tp = entry * (1 + TAKE_PROFIT_PCT/100) if side == "LONG" else entry * (1 - TAKE_PROFIT_PCT/100)
    sl = entry * (1 - STOP_LOSS_PCT/100)   if side == "LONG" else entry * (1 + STOP_LOSS_PCT/100)
    return tp, sl

async def open_position(app: Application, pair: str, price: float, side: str, reason: str):
    pos = pair_pos[pair]
    if pos.is_open:
        return  # уже открыта

    pos.is_open = True
    pos.side = "LONG" if side == "BUY" else "SHORT"
    pos.entry_price = price
    pos.size = TRADE_SIZE
    pos.tp, pos.sl = setup_tp_sl(price, pos.side)
    pos.open_ts = datetime.now(timezone.utc)
    pos.reason = reason

    # реальный ордер: здесь бы отправляли на биржу, но пока DEMO
    mode = "DEMO" if DEMO_MODE else "LIVE"
    txt = (
        f"🟢 *OPEN {pos.side}* `{pair}`\n"
        f"• Price: *{fmt(price,5)}*\n"
        f"• Size: *{pos.size}*\n"
        f"• TP: *{fmt(pos.tp,5)}*  SL: *{fmt(pos.sl,5)}*\n"
        f"• Reason: _{reason}_\n"
        f"• Time: {utc_now_str()}\n"
        f"• Mode: *{mode}*"
    )
    await tg_send(app, txt)
    logger.info(f"[{pair}] OPEN {pos.side} @ {price} ({reason})")

async def close_position(app: Application, pair: str, price: float, cause: str):
    pos = pair_pos[pair]
    if not pos.is_open:
        return

    # PnL в USDT (для шорта инверсия)
    if pos.side == "LONG":
        pnl = (price - pos.entry_price) * pos.size
    else:
        pnl = (pos.entry_price - price) * pos.size

    st = pair_stats[pair]
    st.trades += 1
    if pnl >= 0:
        st.wins += 1
    else:
        st.losses += 1
    st.realized_pnl_usdt += pnl

    dur = ""
    if pos.open_ts:
        sec = (datetime.now(timezone.utc) - pos.open_ts).total_seconds()
        m = int(sec//60)
        s = int(sec%60)
        dur = f"{m}m {s}s"

    txt = (
        f"🔴 *CLOSE {pos.side}* `{pair}`\n"
        f"• Exit: *{fmt(price,5)}*  (entry {fmt(pos.entry_price,5)})\n"
        f"• PnL: *{fmt(pnl,5)} USDT*\n"
        f"• Trades: *{st.trades}*  Winrate: *{st.winrate:.1f}%*\n"
        f"• Cum PnL: *{fmt(st.realized_pnl_usdt,5)} USDT*\n"
        f"• Held: {dur}\n"
        f"• Cause: _{cause}_\n"
        f"• Time: {utc_now_str()}"
    )
    await tg_send(app, txt)
    logger.info(f"[{pair}] CLOSE {pos.side} @ {price} cause={cause} pnl={pnl}")

    # сброс позиции
    pair_pos[pair] = Position()

# ========= WORKER PER PAIR =========
async def run_pair(app: Application, pair: str):
    logger.info(f"Started worker for {pair}")
    while True:
        try:
            df = await fetch_klines(pair, TIMEFRAME, limit=max(EMA_LEN+SLOPE_BARS+5, 120))
            last = df["close"].iloc[-1]
            sig = decide_signal(df)

            # закрытия по TP/SL, если позиция уже открыта
            pos = pair_pos[pair]
            if pos.is_open:
                if pos.side == "LONG":
                    if last >= pos.tp:
                        await close_position(app, pair, last, "TP")
                    elif last <= pos.sl:
                        await close_position(app, pair, last, "SL")
                else:
                    if last <= pos.tp:
                        await close_position(app, pair, last, "TP")
                    elif last >= pos.sl:
                        await close_position(app, pair, last, "SL")

            # если позиции нет — рассмотреть открытие
            pos = pair_pos[pair]  # мог закрыться выше
            if not pos.is_open and sig["action"] in ("BUY", "SELL"):
                await open_position(app, pair, last, sig["action"], sig["reason"])

            # debug-заметки (по запросу)
            if DEBUG_TELEMETRY:
                patt = patterns(df)
                t = (
                    f"🧪 *DEBUG*\n"
                    f"• pair: {pair}\n"
                    f"• last_close: {fmt(last,5)}\n"
                    f"• EMA{EMA_LEN}_slope({SLOPE_BARS} bars): {fmt((df['close'].ewm(span=EMA_LEN, adjust=False).mean().diff().tail(SLOPE_BARS).sum()),6)}\n"
                    f"• patterns: {json.dumps({k: bool(v) for k,v in patt.items()}, ensure_ascii=False)}\n"
                    f"• pos_open: {pos.is_open}"
                )
                await tg_send(app, t)

        except Exception as e:
            logger.error(f"[{pair}] worker error: {e}")

        await asyncio.sleep(POLL_SEC)

# ========= TELEGRAM COMMANDS =========
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    mode = "DEMO" if DEMO_MODE else "LIVE"
    await update.message.reply_text(f"✅ mybot9 running ({mode})\nPAIRs: {', '.join(PAIRS)}  TF: {TIMEFRAME}")

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lines = [f"📊 *STATUS* {utc_now_str()}"]
    for p in PAIRS:
        st = pair_stats[p]
        pos = pair_pos[p]
        pos_line = "—"
        if pos.is_open:
            pos_line = f"{pos.side} @ {fmt(pos.entry_price,5)} (TP {fmt(pos.tp,5)} / SL {fmt(pos.sl,5)})"
        lines.append(
            f"`{p}` • trades: {st.trades}  WR: {st.winrate:.1f}%  PnL: {fmt(st.realized_pnl_usdt,5)}  pos: {pos_line}"
        )
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lines = [f"📌 *OPEN POSITIONS*"]
    has = False
    for p in PAIRS:
        pos = pair_pos[p]
        if pos.is_open:
            has = True
            lines.append(
                f"`{p}` {pos.side} size {pos.size} entry {fmt(pos.entry_price,5)} | TP {fmt(pos.tp,5)} SL {fmt(pos.sl,5)}"
            )
    if not has:
        lines.append("_no open positions_")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# ========= MAIN =========
async def run_bot():
    logger.info("🤖 mybot9 started successfully!")

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("positions", cmd_positions))

    # Правильная последовательность для PTB v21
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Telegram polling started")

    # запустим воркеры по парам
    tasks = [asyncio.create_task(run_pair(application, p)) for p in PAIRS]

    try:
        while True:
            logger.info("Bot is alive... waiting for signals")
            await asyncio.sleep(60)
    finally:
        for t in tasks:
            t.cancel()
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Telegram polling stopped")

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass
