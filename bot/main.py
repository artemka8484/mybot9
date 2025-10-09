# bot/main.py
import os
import asyncio
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import httpx
from loguru import logger

# Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= ENV & GLOBALS =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

TIMEFRAME = os.getenv("TIMEFRAME", "1m").lower()  # '1m' | '5m' | '15m'...
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))
DEMO_MODE = str(os.getenv("DEMO_MODE", "true")).lower() in ("1", "true", "yes")

TP_PCT = float(os.getenv("TP_PCT", "0.004"))     # 0.4%
SL_PCT = float(os.getenv("SL_PCT", "0.004"))     # 0.4%
TRAILING = int(os.getenv("TRAILING", "0"))       # 0/1
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.002"))
DEBUG_TELEMETRY = int(os.getenv("DEBUG_TELEMETRY", "0"))

pairs_env = os.getenv(
    "PAIRS",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
)
PAIRS = [p.strip().upper() for p in pairs_env.split(",") if p.strip()]

# per-pair runtime state
RUNTIME_POSITIONS: dict[str, dict] = {p: {
    "pos_open": False, "side": None, "qty": 0.0, "entry": None,
    "take": None, "stop": None, "trail_on": False, "trail_max": None,
    "last_close_ts": 0
} for p in PAIRS}

# last prices cache for /status
LAST_PRICES: dict[str, float] = {p: 0.0 for p in PAIRS}

# day stats
DAILY_STATS = {
    "day": datetime.now(timezone.utc).date(),
    "trades_today": 0,
    "wins_today": 0,
    "realized_pnl_today": 0.0,
    "balance": 0.0,
}

SETTINGS = {
    "TIMEFRAME": TIMEFRAME,
    "TP_PCT": TP_PCT,
    "SL_PCT": SL_PCT,
    "TRAILING": TRAILING,
    "TRAIL_PCT": TRAIL_PCT,
    "TRADE_SIZE": TRADE_SIZE,
    "DEMO_MODE": DEMO_MODE,
}

BINANCE_BASE = "https://api.binance.com"


# ========= UTIL =========
def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "-"


def reset_if_new_day():
    utc_today = datetime.now(timezone.utc).date()
    if DAILY_STATS["day"] != utc_today:
        DAILY_STATS["day"] = utc_today
        DAILY_STATS["trades_today"] = 0
        DAILY_STATS["wins_today"] = 0
        DAILY_STATS["realized_pnl_today"] = 0.0


async def tg_send(context: Application | ContextTypes.DEFAULT_TYPE, text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        if isinstance(context, Application):
            bot = context.bot
        else:
            bot = context.bot  # ContextTypes has .bot
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


def norm_bool_dict(d: dict) -> dict:
    """Приводим numpy.bool_ к обычным bool (для json и отправки в TG)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.bool_, np.bool8)):
            out[k] = bool(v)
        else:
            out[k] = bool(v)
    return out


# ========= DATA =========
async def fetch_klines(symbol: str, interval: str = TIMEFRAME, limit: int = 200) -> pd.DataFrame:
    """
    Binance klines: 12 columns:
    0 Open time, 1 Open, 2 High, 3 Low, 4 Close, 5 Volume,
    6 Close time, 7 Quote asset volume, 8 Number of trades,
    9 Taker buy base volume, 10 Taker buy quote volume, 11 Ignore.
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbbv","tbqv","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df


# ========= STRATEGY #9 =========
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA48 + наклон (разница за N баров) + простые паттерны."""
    if df.empty:
        return df
    df = df.copy()

    # EMA48 по close
    df["ema48"] = df["close"].ewm(span=48, adjust=False).mean()

    # наклон ema48 за последние 5 баров
    df["ema48_slope5"] = df["ema48"] - df["ema48"].shift(5)

    # паттерны
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Bullish Engulfing
    prev_bear = c.shift(1) < o.shift(1)
    curr_bull = c > o
    engulf = (o <= c.shift(1)) & (c >= o.shift(1))
    df["bull_engulf"] = (prev_bear & curr_bull & engulf).astype(bool)

    # Bearish Engulfing
    prev_bull = c.shift(1) > o.shift(1)
    curr_bear = c < o
    engulf_b = (o >= c.shift(1)) & (c <= o.shift(1))
    df["bear_engulf"] = (prev_bull & curr_bear & engulf_b).astype(bool)

    # Hammer (простая эвристика)
    body = (c - o).abs()
    rng = h - l
    lower_sh = ((np.minimum(c, o) - l) > body * 2)
    upper_sh_small = ((h - np.maximum(c, o)) < body * 0.5)
    df["hammer"] = (lower_sh & upper_sh_small & (rng > 0)).astype(bool)

    # Shooting Star
    upper_sh = ((h - np.maximum(c, o)) > body * 2)
    lower_small = ((np.minimum(c, o) - l) < body * 0.5)
    df["shooting"] = (upper_sh & lower_small & (rng > 0)).astype(bool)

    return df


def strategy_nine_signal(df: pd.DataFrame) -> tuple[str | None, dict]:
    """
    Возвращает ('BUY'|'SELL'|None, info_dict)
    Условия (примерные):
      - тренд по ema48_slope5
      - свечный паттерн в сторону входа
    """
    out = {"info": "no_signal"}
    if len(df) < 60:
        return None, out

    last = df.iloc[-1]
    last_close = float(last["close"])
    slope5 = float(last["ema48_slope5"])

    patt = {
        "bull_engulf": bool(last["bull_engulf"]),
        "bear_engulf": bool(last["bear_engulf"]),
        "hammer": bool(last["hammer"]),
        "shooting": bool(last["shooting"]),
    }

    # тренд вверх + бычий паттерн
    if slope5 > 0 and (patt["bull_engulf"] or patt["hammer"]):
        out = {"reason": "trend_up+pattern_long", "last_close": last_close,
               "ema48_slope5": slope5, "patt": patt}
        return "BUY", out

    # тренд вниз + медвежий паттерн
    if slope5 < 0 and (patt["bear_engulf"] or patt["shooting"]):
        out = {"reason": "trend_down+pattern_short", "last_close": last_close,
               "ema48_slope5": slope5, "patt": patt}
        return "SELL", out

    out = {"last_close": last_close, "ema48_slope5": slope5, "patt": patt}
    return None, out


# ========= REPORTS =========
def build_trade_open_report(pair: str, side: str, qty: float, entry: float, info: dict) -> str:
    patt = norm_bool_dict(info.get("patt", {}))
    lines = [
        "🟢 <b>OPEN</b>",
        f"• <b>{pair}</b> {side}  qty <code>{qty}</code>",
        f"• entry: <code>{entry:.6f}</code>",
        f"• reason: <code>{info.get('reason','')}</code>",
        f"• patterns: <code>{json.dumps(patt, ensure_ascii=False)}</code>",
        f"• time: <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>",
        f"• mode: <code>{'DEMO' if DEMO_MODE else 'LIVE'}</code>",
    ]
    return "\n".join(lines)


def build_trade_close_report(pair: str, side: str, qty: float, entry: float, exit_price: float,
                             pnl: float, reason: str) -> str:
    win_text = "✅ WIN" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ BREAKEVEN"
    lines = [
        "🔴 <b>CLOSE</b>",
        f"• <b>{pair}</b> {side}  qty <code>{qty}</code>",
        f"• entry: <code>{entry:.6f}</code>  exit: <code>{exit_price:.6f}</code>",
        f"• PnL: <b><code>{pnl:.6f}</code></b>  {win_text}",
        f"• reason: <code>{reason}</code>",
        "— — —",
        f"• trades today: <code>{DAILY_STATS['trades_today']}</code>, "
        f"wins: <code>{DAILY_STATS['wins_today']}</code>, "
        f"win-rate: <code>{(DAILY_STATS['wins_today']/DAILY_STATS['trades_today']*100):.1f}%</code>"
        if DAILY_STATS["trades_today"] else "• win-rate: <code>0.0%</code>",
        f"• realized PnL today: <code>{DAILY_STATS['realized_pnl_today']:.6f}</code>",
        f"• balance: <b><code>{DAILY_STATS['balance']:.6f}</code></b>",
        f"• time: <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>",
    ]
    return "\n".join(lines)


# ========= STATUS (/status) =========
def build_status_message(
    pairs,
    positions,
    stats,
    settings,
    last_prices=None
) -> str:
    last_prices = last_prices or {}
    lines = []
    lines.append("✅ <b>Status</b>")
    lines.append(f"• Time: <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>")

    tf = settings.get("TIMEFRAME", "1m")
    tp = settings.get("TP_PCT", 0.005)
    sl = settings.get("SL_PCT", 0.005)
    trailing = settings.get("TRAILING", 0)
    trail_pct = settings.get("TRAIL_PCT", 0.0)
    size = settings.get("TRADE_SIZE", 0.001)

    lines.append(
        "• Settings: "
        f"TF <code>{tf}</code>, "
        f"TP {_fmt_pct(tp)}, SL {_fmt_pct(sl)}, "
        f"Trailing <code>{'on' if trailing else 'off'}</code>{' '+_fmt_pct(trail_pct) if trailing else ''}, "
        f"Size <code>{size}</code>, Mode <code>{'DEMO' if DEMO_MODE else 'LIVE'}</code>"
    )

    lines.append("— — —")
    open_cnt = 0
    for p in pairs:
        st = positions.get(p, {})
        if not st.get("pos_open"):
            continue
        open_cnt += 1
        side = "LONG" if st.get("side") in ("LONG", "BUY") else "SHORT"
        qty = st.get("qty", 0.0)
        entry = st.get("entry")
        last = last_prices.get(p, entry)
        pnl = (last - entry) * qty if side == "LONG" else (entry - last) * qty
        lines.append(
            f"• <b>{p}</b> {side} qty <code>{qty}</code>"
            f"\n  entry <code>{entry:.6f}</code>  last <code>{last:.6f}</code>"
            f"\n  uPnL <code>{pnl:.6f}</code>"
        )
    if open_cnt == 0:
        lines.append("• Open positions: <i>none</i>")

    lines.append("— — —")
    n_tr = stats.get("trades_today", 0)
    wins = stats.get("wins_today", 0)
    wr = (wins / n_tr * 100.0) if n_tr else 0.0
    lines.append(
        f"• trades today: <code>{n_tr}</code>, wins: <code>{wins}</code>, "
        f"win-rate: <code>{wr:.1f}%</code>"
    )
    lines.append(f"• realized PnL today: <code>{stats.get('realized_pnl_today',0.0):.6f}</code>")
    lines.append(f"• balance: <b><code>{stats.get('balance',0.0):.6f}</code></b>")

    return "\n".join(lines)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = build_status_message(PAIRS, RUNTIME_POSITIONS, DAILY_STATS, SETTINGS, LAST_PRICES)
    await tg_send(context, msg)


# ========= EXECUTION (DEMO) =========
def enter_position(pair: str, side: str, entry: float, qty: float):
    st = RUNTIME_POSITIONS[pair]
    st["pos_open"] = True
    st["side"] = "LONG" if side == "BUY" else "SHORT"
    st["qty"] = qty
    st["entry"] = entry
    st["take"] = entry * (1 + TP_PCT) if st["side"] == "LONG" else entry * (1 - TP_PCT)
    st["stop"] = entry * (1 - SL_PCT) if st["side"] == "LONG" else entry * (1 + SL_PCT)
    st["trail_on"] = False
    st["trail_max"] = entry


def should_close(pair: str, price: float) -> tuple[bool, str]:
    st = RUNTIME_POSITIONS[pair]
    if not st["pos_open"]:
        return False, ""
    side = st["side"]
    take = st["take"]
    stop = st["stop"]

    # trailing (простой)
    if TRAILING:
        if side == "LONG":
            st["trail_max"] = max(st["trail_max"], price)
            dyn_stop = st["trail_max"] * (1 - TRAIL_PCT)
            if dyn_stop > stop:
                stop = dyn_stop
        else:
            st["trail_max"] = min(st["trail_max"], price) if st["trail_max"] else price
            dyn_stop = st["trail_max"] * (1 + TRAIL_PCT)
            if dyn_stop < stop:
                stop = dyn_stop

    if side == "LONG":
        if price >= take:
            return True, "take-profit"
        if price <= stop:
            return True, "stop-loss"
    else:
        if price <= take:
            return True, "take-profit"
        if price >= stop:
            return True, "stop-loss"

    return False, ""


def close_position(pair: str, price: float, reason: str) -> float:
    st = RUNTIME_POSITIONS[pair]
    side = st["side"]
    qty = st["qty"]
    entry = st["entry"]

    pnl = (price - entry) * qty if side == "LONG" else (entry - price) * qty

    # update stats
    reset_if_new_day()
    DAILY_STATS["trades_today"] += 1
    if pnl > 0:
        DAILY_STATS["wins_today"] += 1
    DAILY_STATS["realized_pnl_today"] += pnl
    DAILY_STATS["balance"] += pnl

    # reset state
    st["pos_open"] = False
    st["side"] = None
    st["qty"] = 0.0
    st["entry"] = None
    st["take"] = None
    st["stop"] = None
    st["trail_on"] = False
    st["trail_max"] = None

    return pnl


# ========= WORKER PER PAIR =========
async def run_pair(pair: str, app: Application):
    """Основной цикл по инструменту: ждём новой свечи, решаем, торгуем."""
    tf_to_sec = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400
    }
    step = tf_to_sec.get(TIMEFRAME, 60)
    last_close_time = None

    while True:
        try:
            df = await fetch_klines(pair, TIMEFRAME, limit=200)
            if df.empty:
                await asyncio.sleep(5)
                continue

            df = compute_indicators(df)
            LAST_PRICES[pair] = float(df.iloc[-1]["close"])

            # работаем только на НОВОЙ закрытой свече
            close_time = int(df.iloc[-1]["close_time"])
            if last_close_time is None:
                last_close_time = close_time
            elif close_time == last_close_time:
                await asyncio.sleep(3)
                continue
            last_close_time = close_time

            # проверяем выход из позиции
            st = RUNTIME_POSITIONS[pair]
            price = float(df.iloc[-1]["close"])
            if st["pos_open"]:
                do_close, why = should_close(pair, price)
                if do_close:
                    pnl = close_position(pair, price, why)
                    report = build_trade_close_report(pair, st["side"], st["qty"], st["entry"], price, pnl, why)
                    await tg_send(app, report)

            # если позиции нет — ищем сигнал
            if not RUNTIME_POSITIONS[pair]["pos_open"]:
                signal, info = strategy_nine_signal(df)
                if signal in ("BUY", "SELL"):
                    enter_position(pair, signal, price, TRADE_SIZE)
                    open_msg = build_trade_open_report(pair, signal, TRADE_SIZE, price, info)
                    await tg_send(app, open_msg)

            # DEBUG
            if DEBUG_TELEMETRY:
                info_line = {
                    "last_close": float(df.iloc[-1]["close"]),
                    "EMA48_slope(5)": float(df.iloc[-1]["ema48_slope5"]),
                    "patterns": norm_bool_dict({
                        "bull_engulf": df.iloc[-1]["bull_engulf"],
                        "bear_engulf": df.iloc[-1]["bear_engulf"],
                        "hammer": df.iloc[-1]["hammer"],
                        "shooting": df.iloc[-1]["shooting"],
                    }),
                    "pos_open": bool(RUNTIME_POSITIONS[pair]["pos_open"])
                }
                text = (
                    "🧪 <b>DEBUG</b>"
                    f"\n• pair: {pair}"
                    f"\n• last_close: <code>{info_line['last_close']:.5f}</code>"
                    f"\n• EMA48_slope(5): <code>{info_line['EMA48_slope(5)']:.6f}</code>"
                    f"\n• patterns: <code>{json.dumps(info_line['patterns'], ensure_ascii=False)}</code>"
                    f"\n• pos_open: <code>{info_line['pos_open']}</code>"
                )
                await tg_send(app, text)

            await asyncio.sleep(max(1, step // 2))
        except Exception as e:
            logger.exception(e)
            await asyncio.sleep(5)


# ========= TELEGRAM CMDS =========
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        f"✅ mybot9 is running with strategy #9 "
        f"(mode: {'DEMO' if DEMO_MODE else 'LIVE'})\n"
        f"PAIRs: {', '.join(PAIRS)}  TF: {TIMEFRAME}"
    )
    await tg_send(context, txt)


# ========= MAIN =========
async def run_bot():
    logger.info("🤖 mybot9 started successfully!")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))

    # запускаем телеграм polling фоном
    asyncio.create_task(app.initialize())
    asyncio.create_task(app.start())

    # воркеры по каждому инструменту
    tasks = [asyncio.create_task(run_pair(p, app)) for p in PAIRS]

    # heartbeat
    while True:
        logger.info("Bot is alive... waiting for signals")
        await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass
