import os
import sys
import time
import threading
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from telegram.utils.request import Request

# ========== CONFIG ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# --- env variables ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
MEXC_API_KEY = os.getenv("MEXC_API_KEY")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET")
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
TX = os.getenv("TX", "UTC")

# --- trading state ---
positions = {}
balance = START_BALANCE
stats = {pair: {"trades": 0, "wr": 0, "pnl": 0} for pair in PAIRS}
SCHED_STARTED = False
lock = threading.Lock()

# ========== TELEGRAM SETUP ==========

# увеличенный пул и таймауты
req = Request(con_pool_size=8, connect_timeout=20, read_timeout=20)
updater = Updater(token=TELEGRAM_TOKEN, request=req, use_context=True)
dispatcher = updater.dispatcher
bot = Bot(token=TELEGRAM_TOKEN)

# ========== FUNCTIONS ==========

def get_candles(symbol, limit=100):
    url = f"{MEXC_BASE_URL}/api/v1/contract/kline/{symbol}"
    params = {"interval": TIMEFRAME, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        log.warning(f"{symbol} get_candles error: {e}")
        return None

def atr(df, n=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(n).mean()

def ema(df, n=48):
    return df["close"].ewm(span=n, adjust=False).mean()

def analyze_signal(df):
    if len(df) < EMA_LEN + 2:
        return None
    df["ema"] = ema(df, EMA_LEN)
    slope = df["ema"].iloc[-1] - df["ema"].iloc[-EMA_SLOPE_BARS]
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last["close"] - last["open"])
    shadow = (last["high"] - last["low"]) / max(body, 0.00001)
    atr_val = atr(df, ATR_LEN).iloc[-1]
    if body < (atr_val * 0.2) and shadow > 2:
        if last["close"] > last["open"]:
            return ("hammer", "LONG", slope, atr_val)
        else:
            return ("hammer", "SHORT", slope, atr_val)
    return None

def open_trade(pair):
    global balance
    df = get_candles(pair, 100)
    if df is None: return
    sig = analyze_signal(df)
    if not sig: return
    name, direction, slope, atr_val = sig
    entry = df["close"].iloc[-1]
    tp = entry * (1 + TP_PCT / 100 * (1 if direction == "LONG" else -1))
    sl = entry - ATR_MULT_SL * atr_val if direction == "LONG" else entry + ATR_MULT_SL * atr_val
    qty = (balance * (RISK_PCT / 100)) / abs(entry - sl)
    positions[pair] = {"dir": direction, "entry": entry, "tp": tp, "sl": sl, "qty": qty}
    msg = (
        f"🔴 OPEN {pair} {direction}\n"
        f"• time: {datetime.utcnow()} UTC\n"
        f"• entry: {entry:.5f}\n• qty: {qty:.5f}\n"
        f"• TP: {tp:.5f}  SL: {sl:.5f}\n"
        f"• signal: {name}, slope {slope:.5f}, ATR {atr_val:.5f}\n• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
    )
    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")

def check_positions():
    global balance
    with lock:
        for pair, pos in list(positions.items()):
            df = get_candles(pair, 3)
            if df is None: continue
            price = df["close"].iloc[-1]
            pnl = 0
            closed = False
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
                result = "✅ CLOSE" if pnl > 0 else "❌ CLOSE"
                bot.send_message(
                    chat_id=CHAT_ID,
                    text=(
                        f"{result} {pair} ({'TP' if pnl > 0 else 'SL'})\n"
                        f"• time: {datetime.utcnow()} UTC\n"
                        f"• exit: {price:.5f}\n"
                        f"• PnL: {pnl:.5f}\n"
                        f"• equity: {balance:.2f} USDT"
                    ),
                    parse_mode="HTML"
                )
                del positions[pair]

def loop_symbol(pair):
    bot.send_message(chat_id=CHAT_ID, text=f"✅ Loop started for <b>{pair}</b>", parse_mode="HTML")
    while True:
        try:
            check_positions()
            if pair not in positions:
                open_trade(pair)
            time.sleep(COOLDOWN_SEC)
        except Exception as e:
            log.warning(f"{pair} loop error: {e}")
            time.sleep(10)

def start_loops():
    for p in PAIRS:
        threading.Thread(target=loop_symbol, args=(p,), daemon=True).start()

# ========== TELEGRAM COMMANDS ==========
def status(update: Update, ctx: CallbackContext):
    eq = f"{balance:.2f}"
    text = f"📊 STATUS {datetime.utcnow()} UTC\n"
    total_trades = sum(v['trades'] for v in stats.values())
    text += f"TOTAL • trades: {total_trades}  equity: {eq} USDT\n\n"
    for pair, pos in positions.items():
        text += f"{pair} • {pos['dir']} @ {pos['entry']:.5f} (TP {pos['tp']:.5f} / SL {pos['sl']:.5f})\n"
    bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="HTML")

dispatcher.add_handler(CommandHandler("status", status))

# ========== START ==========
def main():
    log.info(f"Mode DEMO={DEMO_MODE}  pairs={PAIRS} tf={TIMEFRAME}  risk={RISK_PCT}% tp={TP_PCT}% atr_len={ATR_LEN} atr_mult={ATR_MULT_SL} ema={EMA_LEN}/{EMA_SLOPE_BARS}")
    try:
        Bot(TELEGRAM_TOKEN).delete_webhook(drop_pending_updates=True)
    except Exception:
        pass
    log.info("Starting polling…")
    updater.start_polling(drop_pending_updates=True)
    start_loops()
    updater.idle()

if __name__ == "__main__":
    main()
