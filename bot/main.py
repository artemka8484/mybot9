# bot/main.py
import os
import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

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

PAIRS = [x.strip() for x in os.getenv("PAIRS", "BTCUSDT,ETHUSDT").split(",")]
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

# ──────────────── RUNTIME (в памяти процесса) ────────────────
positions: dict[str, dict] = {}      # pair -> {dir, entry, tp, sl, qty}
balance: float = START_BALANCE
stats: dict[str, dict] = {p: {"trades": 0, "wins": 0, "pnl": 0.0} for p in PAIRS}
_started_pairs: set[str] = set()

alock = asyncio.Lock()               # для конкурентной защиты

# ────────────────────── MARKET HELPERS ───────────────────────
def get_candles(symbol: str, limit: int = 200) -> pd.DataFrame | None:
    """Берём свечи с MEXC. Возвращаем DataFrame или None."""
    url = f"{MEXC_BASE_URL}/api/v1/contract/kline/{symbol}"
    params = {"interval": TIMEFRAME, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        # приводим типы (в ответе MEXC бывают строки)
        num_cols = ["open", "high", "low", "close"]
        for c in num_cols:
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
    """Ищем простую «hammer» + фильтр по наклону EMA."""
    if len(df) < EMA_LEN + EMA_SLOPE_BARS + 2:
        return None
    df = df.copy()
    df["ema"] = ema(df, EMA_LEN)
    slope = df["ema"].iloc[-1] - df["ema"].iloc[-EMA_SLOPE_BARS]

    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    rng = last["high"] - last["low"]
    atr_val = atr(df, ATR_LEN).iloc[-1]

    # «молот»: маленькое тело и большой диапазон/тени
    if rng <= 0 or np.isnan(atr_val):
        return None
    shadow_ratio = rng / max(body, 1e-9)

    if body < atr_val * 0.25 and shadow_ratio > 2:
        direction = "LONG" if last["close"] > last["open"] else "SHORT"
        return ("hammer", direction, float(slope), float(atr_val))
    return None

# ─────────────────────── TRADING LOGIC ───────────────────────
def _qty_from_risk(entry: float, sl: float) -> float:
    risk_usd = balance * (RISK_PCT / 100.0)
    per_unit_loss = abs(entry - sl)
    if per_unit_loss <= 0:
        return 0.0
    return risk_usd / per_unit_loss

async def open_trade(pair: str, ctx: ContextTypes.DEFAULT_TYPE):
    global balance
    df = get_candles(pair, 200)
    if df is None:
        return

    sig = analyze_signal(df)
    if not sig:
        return

    name, direction, slope, atr_val = sig
    entry = df["close"].iloc[-1]

    if direction == "LONG":
        tp = entry * (1 + TP_PCT / 100.0)
        sl = entry - ATR_MULT_SL * atr_val
    else:
        tp = entry * (1 - TP_PCT / 100.0)
        sl = entry + ATR_MULT_SL * atr_val

    qty = _qty_from_risk(entry, sl)
    if qty <= 0:
        return

    positions[pair] = {
        "dir": direction, "entry": float(entry),
        "tp": float(tp), "sl": float(sl), "qty": float(qty)
    }

    msg = (
        f"🔴 OPEN {pair} {direction}\n"
        f"• time: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC\n"
        f"• entry: {entry:.5f}\n"
        f"• qty: {qty:.5f}\n"
        f"• TP: {tp:.5f}   SL: {sl:.5f}\n"
        f"• signal: {name}, slope {slope:.5f}, ATR {atr_val:.5f}\n"
        f"• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
    )
    await ctx.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")

async def check_positions(ctx: ContextTypes.DEFAULT_TYPE):
    """Проверяем TP/SL, закрываем при срабатывании, ведём статистику."""
    global balance
    async with alock:
        for pair, pos in list(positions.items()):
            df = get_candles(pair, 3)
            if df is None:
                continue
            price = float(df["close"].iloc[-1])
            closed = False
            pnl = 0.0

            if pos["dir"] == "LONG":
                if price >= pos["tp"]:
                    pnl = (pos["tp"] - pos["entry"]) * pos["qty"]
                    closed = True
                elif price <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
                    closed = True
            else:  # SHORT
                if price <= pos["tp"]:
                    pnl = (pos["entry"] - pos["tp"]) * pos["qty"]
                    closed = True
                elif price >= pos["sl"]:
                    pnl = (pos["entry"] - pos["sl"]) * pos["qty"]
                    closed = True

            if closed:
                balance += pnl
                stats[pair]["trades"] += 1
                stats[pair]["pnl"] += pnl
                if pnl > 0:
                    stats[pair]["wins"] += 1

                result = "✅ CLOSE" if pnl > 0 else "❌ CLOSE"
                label = "TP" if pnl > 0 else "SL"
                msg = (
                    f"{result} {pair} ({label})\n"
                    f"• time: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC\n"
                    f"• exit: {price:.5f}\n"
                    f"• PnL: {pnl:.5f}\n"
                    f"• equity: {balance:.2f}  (lev {LEVERAGE:.1f}x, fee {FEE_PCT*100:.3f}%)"
                )
                await ctx.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")
                del positions[pair]

# ──────────────────── JOB QUEUE CALLBACKS ────────────────────
async def loop_pair(ctx: ContextTypes.DEFAULT_TYPE):
    """Повторяющаяся задача: на каждую пару — проверка позиций + возможное открытие."""
    pair = ctx.job.data
    if pair not in _started_pairs:
        _started_pairs.add(pair)
        await ctx.bot.send_message(chat_id=CHAT_ID, text=f"✅ Loop started for <b>{pair}</b>", parse_mode="HTML")
    await check_positions(ctx)
    if pair not in positions:
        await open_trade(pair, ctx)

# ──────────────────────── /status CMD ────────────────────────
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    eq = balance
    text = [f"📊 STATUS {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC"]

    total_trades = sum(s["trades"] for s in stats.values())
    wr = (sum(s["wins"] for s in stats.values()) / total_trades * 100) if total_trades else 0.0
    total_pnl = sum(s["pnl"] for s in stats.values())

    for pair in PAIRS:
        s = stats[pair]
        text.append(
            f"{pair} • trades: {s['trades']}  WR: { (s['wins']/s['trades']*100 if s['trades'] else 0):.2f}%  "
            f"PnL: {s['pnl']:.5f}"
        )

    if positions:
        for p, pos in positions.items():
            text.append(f"{p} • pos: {pos['dir']} @ {pos['entry']:.5f} (TP {pos['tp']:.5f} / SL {pos['sl']:.5f})")
    else:
        text.append("—")

    text.append("—")
    text.append(f"TOTAL • trades: {total_trades}  WR: {wr:.2f}%  PnL: {total_pnl:.5f}")
    text.append(f"equity: {eq:.2f}  (0.0% с начала)")  # для простоты
    text.append(f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT*100:.3f}%")

    await context.bot.send_message(chat_id=CHAT_ID, text="\n".join(text), parse_mode="HTML")

# ────────────────────────── MAIN ─────────────────────────────
async def on_start(app):
    # polling-режим: убедимся, что вебхука нет (важно после прошлых экспериментов)
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass

    # Планируем по задаче на каждую пару
    for p in PAIRS:
        # «первый запуск» через 1 сек, затем каждые COOLDOWN_SEC
        app.job_queue.run_repeating(loop_pair, interval=COOLDOWN_SEC, first=1, data=p, name=f"loop-{p}")

    await app.bot.send_message(
        chat_id=CHAT_ID,
        text=(
            "🤖 mybot9 started successfully!\n"
            f"Mode: {'DEMO' if DEMO_MODE else 'LIVE'} | Leverage {LEVERAGE:.1f}x | Fee {FEE_PCT*100:.3f}% | "
            f"Risk {RISK_PCT:.1f}%\nPairs: {', '.join(PAIRS)} | TF {TIMEFRAME} | Tick {COOLDOWN_SEC}s\n"
            f"Balance: {START_BALANCE:.0f}  USDT"
        ),
    )

def build_app():
    if not TOKEN or not CHAT_ID:
        raise SystemExit("TELEGRAM_TOKEN и TELEGRAM_CHAT_ID обязательны")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("status", cmd_status))
    app.post_init = on_start
    return app

def main():
    app = build_app()
    # run_polling — асинхронный, сам поднимает loop и SIGINT handler
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
