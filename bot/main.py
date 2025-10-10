# bot/main.py
# Универсальный запуск: поддержка python-telegram-bot v13.x и v20+
import os
import logging
from datetime import datetime
from threading import RLock
import time

import numpy as np
import pandas as pd
import requests

# ────────────────────────── LOGGING ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mybot9")

# ─────────────────────── ENV / CONFIG ────────────────────────
TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

MEXC_BASE_URL = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com")
PAIRS = [x.strip() for x in os.getenv("PAIRS", "BTCUSDT,ETHUSDT").split(",") if x.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "5m")

RISK_PCT = float(os.getenv("RISK_PCT", "1"))
TP_PCT = float(os.getenv("TP_PCT", "0.35"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1"))
EMA_LEN = int(os.getenv("EMA_LEN", "100"))
EMA_SLOPE_BARS = int(os.getenv("EMA_SLOPE_BARS", "8"))
LEVERAGE = float(os.getenv("LEVERAGE", "5"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.0006"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "240"))

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
START_BALANCE = float(os.getenv("DEMO_START_BALANCE", "5000"))

if not TOKEN or not CHAT_ID:
    raise SystemExit("TELEGRAM_TOKEN и TELEGRAM_CHAT_ID обязательны")

# ──────────────── STATE (в памяти процесса) ──────────────────
positions = {}  # pair -> {dir, entry, tp, sl, qty}
balance = START_BALANCE
stats = {p: {"trades": 0, "wins": 0, "pnl": 0.0} for p in PAIRS}
_started = set()
lock = RLock()

# ────────────────────── MARKET HELPERS ───────────────────────
def get_candles(symbol: str, limit: int = 200) -> pd.DataFrame | None:
    url = f"{MEXC_BASE_URL}/api/v1/contract/kline/{symbol}"
    params = {"interval": TIMEFRAME, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        log.warning(f"{symbol} get_candles error: {e}")
        return None

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(n).mean()

def ema(df: pd.DataFrame, n: int) -> pd.Series:
    return df["close"].ewm(span=n, adjust=False).mean()

def analyze_signal(df: pd.DataFrame):
    if len(df) < EMA_LEN + EMA_SLOPE_BARS + 2:
        return None
    df = df.copy()
    df["ema"] = ema(df, EMA_LEN)
    slope = df["ema"].iloc[-1] - df["ema"].iloc[-EMA_SLOPE_BARS]

    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    rng = last["high"] - last["low"]
    atr_val = atr(df, ATR_LEN).iloc[-1]
    if rng <= 0 or np.isnan(atr_val):
        return None

    shadow_ratio = rng / max(body, 1e-9)
    if body < atr_val * 0.25 and shadow_ratio > 2:
        direction = "LONG" if last["close"] > last["open"] else "SHORT"
        return ("hammer", direction, float(slope), float(atr_val))
    return None

def _qty_from_risk(entry: float, sl: float) -> float:
    risk_usd = balance * (RISK_PCT / 100.0)
    per_unit = abs(entry - sl)
    if per_unit <= 0:
        return 0.0
    return risk_usd / per_unit

# ─────────────────────── CORE ACTIONS ────────────────────────
def _open_trade(pair: str, bot_send):
    global positions
    df = get_candles(pair, 200)
    if df is None:
        return
    sig = analyze_signal(df)
    if not sig:
        return

    name, direction, slope, atr_val = sig
    entry = float(df["close"].iloc[-1])

    if direction == "LONG":
        tp = entry * (1 + TP_PCT / 100.0)
        sl = entry - ATR_MULT_SL * atr_val
    else:
        tp = entry * (1 - TP_PCT / 100.0)
        sl = entry + ATR_MULT_SL * atr_val

    qty = _qty_from_risk(entry, sl)
    if qty <= 0:
        return

    positions[pair] = {"dir": direction, "entry": entry, "tp": float(tp), "sl": float(sl), "qty": float(qty)}

    msg = (
        f"🔴 OPEN {pair} {direction}\n"
        f"• time: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC\n"
        f"• entry: {entry:.5f}\n"
        f"• qty: {qty:.5f}\n"
        f"• TP: {tp:.5f}   SL: {sl:.5f}\n"
        f"• signal: {name}, slope {slope:.5f}, ATR {atr_val:.5f}\n"
        f"• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
    )
    bot_send(msg)

def _check_positions(bot_send_price, bot_send_msg):
    """bot_send_price(pair)->last_price, bot_send_msg(text) -> send."""
    global balance, positions, stats
    with lock:
        for pair, pos in list(positions.items()):
            price = bot_send_price(pair)
            if price is None:
                continue
            closed = False
            pnl = 0.0
            if pos["dir"] == "LONG":
                if price >= pos["tp"]:
                    pnl = (pos["tp"] - pos["entry"]) * pos["qty"]
                    closed = True
                elif price <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
                    closed = True
            else:
                if price <= pos["tp"]:
                    pnl = (pos["entry"] - pos["tp"]) * pos["qty"]
                    closed = True
                elif price >= pos["sl"]:
                    pnl = (pos["entry"] - pos["sl"]) * pos["qty"]
                    closed = True

            if closed:
                balance += pnl
                s = stats[pair]
                s["trades"] += 1
                s["pnl"] += pnl
                if pnl > 0:
                    s["wins"] += 1
                label = "TP" if pnl > 0 else "SL"
                result = "✅ CLOSE" if pnl > 0 else "❌ CLOSE"
                bot_send_msg(
                    f"{result} {pair} ({label})\n"
                    f"• time: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC\n"
                    f"• exit: {price:.5f}\n"
                    f"• PnL: {pnl:.5f}\n"
                    f"• equity: {balance:.2f}  (lev {LEVERAGE:.1f}x, fee {FEE_PCT*100:.3f}%)"
                )
                del positions[pair]

def _status_text() -> str:
    total_tr = sum(s["trades"] for s in stats.values())
    total_win = sum(s["wins"] for s in stats.values())
    wr = (total_win / total_tr * 100) if total_tr else 0.0
    total_pnl = sum(s["pnl"] for s in stats.values())
    lines = [f"📊 STATUS {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC"]
    for p in PAIRS:
        s = stats[p]
        pair_wr = (s["wins"]/s["trades"]*100) if s["trades"] else 0.0
        lines.append(f"{p} • trades: {s['trades']}  WR: {pair_wr:.2f}%  PnL: {s['pnl']:.5f}")
    if positions:
        for p, pos in positions.items():
            lines.append(f"{p} • pos: {pos['dir']} @ {pos['entry']:.5f} (TP {pos['tp']:.5f} / SL {pos['sl']:.5f})")
    else:
        lines.append("—")
    lines.append("—")
    lines.append(f"TOTAL • trades: {total_tr}  WR: {wr:.2f}%  PnL: {total_pnl:.5f}")
    lines.append(f"equity: {balance:.2f}")
    lines.append(f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT*100:.3f}%")
    return "\n".join(lines)

# ────────────────────── COMPAT LAYER ─────────────────────────
def _price_now(pair: str) -> float | None:
    df = get_candles(pair, 3)
    if df is None:
        return None
    return float(df["close"].iloc[-1])

# Попытка импортировать API v20+
try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

    USE_V20 = True
except Exception:
    USE_V20 = False

# ───────────────────────── v20+ path ─────────────────────────
if USE_V20:
    import asyncio

    async def _send(ctx: ContextTypes.DEFAULT_TYPE, text: str):
        await ctx.bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="HTML")

    async def loop_job(ctx: ContextTypes.DEFAULT_TYPE):
        pair = ctx.job.data
        if pair not in _started:
            _started.add(pair)
            await _send(ctx, f"✅ Loop started for <b>{pair}</b>")
        # check
        _check_positions(_price_now, lambda t: asyncio.create_task(_send(ctx, t)))
        # open if flat
        if pair not in positions:
            _open_trade(pair, lambda t: asyncio.create_task(_send(ctx, t)))

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await _send(context, _status_text())

    async def on_start(app):
        try:
            await app.bot.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass
        for p in PAIRS:
            app.job_queue.run_repeating(loop_job, interval=COOLDOWN_SEC, first=1, data=p, name=f"loop-{p}")
        await app.bot.send_message(
            chat_id=CHAT_ID,
            text=(f"🤖 mybot9 started (v20)\nMode: {'DEMO' if DEMO_MODE else 'LIVE'} | "
                  f"Risk {RISK_PCT:.1f}% | TF {TIMEFRAME} | Tick {COOLDOWN_SEC}s\n"
                  f"Pairs: {', '.join(PAIRS)} | Balance: {START_BALANCE:.0f} USDT")
        )

    def main():
        app = ApplicationBuilder().token(TOKEN).build()
        app.add_handler(CommandHandler("status", cmd_status))
        app.post_init = on_start
        app.run_polling(close_loop=False)

# ───────────────────────── v13.x path ────────────────────────
else:
    # Старый синхронный API
    from telegram.ext import Updater, CommandHandler, CallbackContext

    def _send_sync(context: CallbackContext, text: str):
        context.bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="HTML")

    def loop_job_sync(context: CallbackContext):
        pair = context.job.context
        if pair not in _started:
            _started.add(pair)
            _send_sync(context, f"✅ Loop started for <b>{pair}</b>")
        _check_positions(_price_now, lambda t: _send_sync(context, t))
        if pair not in positions:
            _open_trade(pair, lambda t: _send_sync(context, t))

    def cmd_status_sync(update, context: CallbackContext):
        _send_sync(context, _status_text())

    def main():
        # В v13 НЕЛЬЗЯ передавать параметр 'request' → используем по умолчанию
        updater = Updater(token=TOKEN, use_context=True)
        dp = updater.dispatcher
        dp.add_handler(CommandHandler("status", cmd_status_sync))

        # убрать вебхук на случай прошлых запусков
        try:
            updater.bot.delete_webhook()
        except Exception:
            pass

        # Планируем задачи
        for p in PAIRS:
            updater.job_queue.run_repeating(
                loop_job_sync, interval=COOLDOWN_SEC, first=1, context=p, name=f"loop-{p}"
            )

        # старт
        updater.bot.send_message(
            chat_id=CHAT_ID,
            text=(f"🤖 mybot9 started (v13)\nMode: {'DEMO' if DEMO_MODE else 'LIVE'} | "
                  f"Risk {RISK_PCT:.1f}% | TF {TIMEFRAME} | Tick {COOLDOWN_SEC}s\n"
                  f"Pairs: {', '.join(PAIRS)} | Balance: {START_BALANCE:.0f} USDT")
        )
        updater.start_polling(clean=True)
        updater.idle()

# ──────────────────────────── RUN ────────────────────────────
if __name__ == "__main__":
    main()
