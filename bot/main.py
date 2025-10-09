# bot/main.py
import os
import time
import math
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List

import requests
import pandas as pd
import numpy as np

from telegram import ParseMode
from telegram.ext import Updater, CommandHandler

# =========================
# -------- CONFIG ---------
# =========================

ENV = os.getenv

TELEGRAM_TOKEN = ENV("TELEGRAM_TOKEN", "")
CHAT_ID = int(ENV("TELEGRAM_CHAT_ID", "0") or "0")

MODE = ENV("MODE", "DEMO").upper()          # DEMO / LIVE (но трейды всё равно dry-run без реальных ордеров)
DEMO_MODE = ENV("DEMO_MODE", "true").lower() == "true"
DEMO_START_BALANCE = float(ENV("DEMO_START_BALANCE", "5000"))

PAIRS = [s.strip() for s in ENV("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",") if s.strip()]

TIMEFRAME = ENV("TIMEFRAME", "5m").lower()  # "1m"|"5m"|"15m"
TICK_SECONDS = int(ENV("COOLDOWN_SEC", "60"))

EMA_LEN = int(ENV("EMA_LEN", "48"))
EMA_SLOPE_BARS = int(ENV("EMA_SLOPE_BARS", "5"))
ATR_LEN = int(ENV("ATR_LEN", "14"))
ATR_MULT_SL = float(ENV("ATR_MULT_SL", "0.5"))

TP_PCT = float(ENV("TP_PCT", "1"))           # ВНИМАНИЕ: это % (а не доля). 1 = 1%
RISK_PCT = float(ENV("RISK_PCT", "3.0"))     # % риска на сделку от equity
LEVERAGE = float(ENV("LEVERAGE", "5"))
FEE_PCT = float(ENV("FEE_PCT", "0.0006"))    # комиссия за одну сторону (0.06%)

DAILY_SUMMARY = ENV("DAILY_SUMMARY", "1") == "1"

TX = ENV("TX", "UTC").upper()
TZ = timezone.utc if TX == "UTC" else timezone(timedelta(hours=0))  # оставим UTC

MEXC_BASE_URL = ENV("MEXC_BASE_URL", "https://contract.mexc.com").rstrip("/")

# ---- вспомогательные маппинги таймфреймов ----
BINANCE_TF = {"1m": "1m", "5m": "5m", "15m": "15m"}
MEXC_TF = {"1m": "Min1", "5m": "Min5", "15m": "Min15"}

# =========================
# -------- STATE ----------
# =========================
state: Dict = {
    "equity": DEMO_START_BALANCE,
    "start_equity": DEMO_START_BALANCE,
    "open_positions": {},               # pair -> dict(position)
    "stats": {},                        # pair -> {"wins": int, "losses": int, "pnl": float, "trades": int}
    "total": {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0},
    "day_anchor": datetime.now(timezone.utc).date()
}
for p in PAIRS:
    state["stats"][p] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}

# =========================
# ---- TELEGRAM HELPERS ---
# =========================

def tg_send(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=15)
    except Exception:
        pass


# =========================
# ------- UTILITIES -------
# =========================

def pct(a: float) -> str:
    return f"{a*100:.2f}%"

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_get(url: str, params=None, headers=None, timeout=15) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        try:
            return r.json(), None
        except Exception as e:
            return None, f"json error: {e}"
    except Exception as e:
        return None, str(e)

# =========================
# ------ DATA FEEDS -------
# =========================

def klines_binance(symbol: str, tf: str, limit=200) -> Optional[pd.DataFrame]:
    if tf not in BINANCE_TF:
        return None
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": BINANCE_TF[tf], "limit": min(limit, 1000)}
    js, err = safe_get(url, params=params)
    if not js or not isinstance(js, list):
        return None
    rows = []
    for it in js:
        # [ openTime, open, high, low, close, volume, closeTime, ... ]
        rows.append([int(it[0]), float(it[1]), float(it[2]), float(it[3]), float(it[4]), float(it[5])])
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vol"]).sort_values("t")
    return df


def get_klines(symbol: str, tf: str, limit=200) -> Optional[pd.DataFrame]:
    """
    Исправленный MEXC эндпоинт + корректный fallback на Binance.
    """
    df = None
    # --- MEXC ---
    if tf in MEXC_TF:
        sym_mexc = symbol.replace("USDT", "_USDT")
        url = f"{MEXC_BASE_URL}/api/v1/contract/kline"
        params = {"symbol": sym_mexc, "interval": MEXC_TF[tf], "limit": min(limit, 200)}
        js, err = safe_get(url, params=params)
        if js and isinstance(js, dict) and js.get("success"):
            data = js.get("data") or []
            rows = []
            for it in data:
                # it = { "t":167xxx, "o":"", "h":"", "l":"", "c":"", "v":"" }
                try:
                    rows.append([int(it["t"]), float(it["o"]), float(it["h"]),
                                 float(it["l"]), float(it["c"]), float(it["v"])])
                except Exception:
                    continue
            if rows:
                df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vol"]).sort_values("t")

    # --- Fallback Binance ---
    if df is None or df.empty:
        df = klines_binance(symbol, tf, limit)

    return df

# =========================
# ---- INDICATORS/SIGNS ---
# =========================

def ema(arr: pd.Series, n: int) -> pd.Series:
    return arr.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def ema_slope(series: pd.Series, bars: int) -> float:
    if len(series) < bars + 1:
        return 0.0
    x = np.arange(bars)
    y = series.iloc[-bars:].values
    # простой наклон
    denom = (bars - 1)
    if denom <= 0:
        return 0.0
    return (y[-1] - y[0]) / denom

def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    6 сигналов: bull_engulf, bear_engulf, hammer, shooting, breakout_up, breakout_dn
    """
    out = {k: False for k in ["bull_engulf", "bear_engulf", "hammer", "shooting", "breakout_up", "breakout_dn"]}
    if len(df) < 3:
        return out

    o1, c1, h1, l1 = df.iloc[-2][["open", "close", "high", "low"]]
    o2, c2, h2, l2 = df.iloc[-1][["open", "close", "high", "low"]]

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    rng2 = h2 - l2 + 1e-12

    # engulfings
    if (c1 < o1) and (c2 > o2) and (o2 < c1) and (c2 > o1):
        out["bull_engulf"] = True
    if (c1 > o1) and (c2 < o2) and (o2 > c1) and (c2 < o1):
        out["bear_engulf"] = True

    # hammer / shooting-star: хвост >= 2 * тело
    lower = min(o2, c2) - l2
    upper = h2 - max(o2, c2)
    if lower >= 2 * body2 and body2 / rng2 < 0.6:
        out["hammer"] = True
    if upper >= 2 * body2 and body2 / rng2 < 0.6:
        out["shooting"] = True

    # breakout
    hh = df["high"].iloc[-20:-1].max()
    ll = df["low"].iloc[-20:-1].min()
    if h2 > hh and c2 > o2:
        out["breakout_up"] = True
    if l2 < ll and c2 < o2:
        out["breakout_dn"] = True

    return out

# =========================
# ------- SIZING ----------
# =========================

def position_size(pair: str, price: float, equity: float, risk_pct: float) -> float:
    """
    Размер позиции (квота USDT) = equity * risk_pct * leverage
    Возвращаем кол-во базовой валюты (qty).
    """
    capital = equity * (risk_pct / 100.0) * LEVERAGE
    if capital <= 0:
        return 0.0
    qty = capital / max(price, 1e-9)
    return qty

# =========================
# ------- TRADING ---------
# =========================

def open_signal(df: pd.DataFrame) -> Tuple[Optional[str], Dict[str, float], str]:
    """
    Возвращает ("LONG"/"SHORT"/None, targets, reason)
    """
    pats = detect_patterns(df)
    close = df["close"].iloc[-1]
    ema48 = ema(df["close"], EMA_LEN)
    slope = ema_slope(ema48, EMA_SLOPE_BARS)
    a = atr(df, ATR_LEN).iloc[-1]

    # Приоритет: breakout > engulf > hammer/shooting
    reason = []
    if pats["breakout_up"]:
        reason.append("breakout_up")
        direction = "LONG"
    elif pats["breakout_dn"]:
        reason.append("breakout_dn")
        direction = "SHORT"
    elif pats["bull_engulf"] or pats["hammer"]:
        reason.append("bull_engulf" if pats["bull_engulf"] else "hammer")
        direction = "LONG"
    elif pats["bear_engulf"] or pats["shooting"]:
        reason.append("bear_engulf" if pats["bear_engulf"] else "shooting")
        direction = "SHORT"
    else:
        return None, {}, ""

    # ТП/SL: TP_PCT (%) относительно цены, SL по ATR
    tp = close * (1.0 + (TP_PCT / 100.0)) if direction == "LONG" else close * (1.0 - (TP_PCT / 100.0))
    sl = (close - ATR_MULT_SL * a) if direction == "LONG" else (close + ATR_MULT_SL * a)

    reason.append(f"slope {slope:.5f}")
    reason.append(f"ATR {a:.5f}")
    return direction, {"tp": tp, "sl": sl}, ", ".join(reason)

def try_open(pair: str, df: pd.DataFrame):
    if pair in state["open_positions"]:
        return
    sig, trg, why = open_signal(df)
    if not sig:
        return

    price = df["close"].iloc[-1]
    qty = position_size(pair, price, state["equity"], RISK_PCT)
    if qty <= 0:
        return

    state["open_positions"][pair] = {
        "dir": sig,
        "entry": price,
        "qty": qty,
        "tp": trg["tp"],
        "sl": trg["sl"],
        "time": datetime.now(timezone.utc),
        "why": why,
    }

    emoji = "🔴" if sig == "SHORT" else "🔴"
    msg = []
    msg.append(f"{emoji} <b>OPEN {pair} {sig}</b>")
    msg.append(f"• time: {now_utc_str()}")
    msg.append(f"• entry: {price:.5f}")
    msg.append(f"• qty: {qty:.6f}")
    msg.append(f"• TP: {trg['tp']:.5f}   SL: {trg['sl']:.5f}")
    msg.append(f"• signal: {why}")
    msg.append(f"• mode: {'DEMO' if DEMO_MODE else MODE}")
    tg_send("\n".join(msg))

def check_close(pair: str, df: pd.DataFrame):
    pos = state["open_positions"].get(pair)
    if not pos:
        return

    high = df["high"].iloc[-1]
    low = df["low"].iloc[-1]
    exit_price = None
    reason = None

    if pos["dir"] == "LONG":
        if high >= pos["tp"]:
            exit_price = pos["tp"]; reason = "TP"
        elif low <= pos["sl"]:
            exit_price = pos["sl"]; reason = "SL"
    else:  # SHORT
        if low <= pos["tp"]:
            exit_price = pos["tp"]; reason = "TP"
        elif high >= pos["sl"]:
            exit_price = pos["sl"]; reason = "SL"

    if exit_price is None:
        return

    # --- закрываем ---
    entry = pos["entry"]
    qty = pos["qty"]
    side = pos["dir"]

    # Комиссии за вход и выход
    fee_in = entry * qty * FEE_PCT
    fee_out = exit_price * qty * FEE_PCT

    if side == "LONG":
        pnl = (exit_price - entry) * qty - fee_in - fee_out
    else:
        pnl = (entry - exit_price) * qty - fee_in - fee_out

    state["equity"] += pnl
    state["total"]["pnl"] += pnl
    state["total"]["trades"] += 1

    st = state["stats"][pair]
    st["pnl"] += pnl
    st["trades"] += 1
    if pnl >= 0:
        st["wins"] += 1
        state["total"]["wins"] += 1
        emoji = "✅"
    else:
        st["losses"] += 1
        state["total"]["losses"] += 1
        emoji = "❌"

    del state["open_positions"][pair]

    # Сообщение о закрытии
    wr_pair = (st["wins"] / st["trades"] * 100.0) if st["trades"] else 0.0
    wr_total = (state["total"]["wins"] / state["total"]["trades"] * 100.0) if state["total"]["trades"] else 0.0

    msg = []
    msg.append(f"{emoji} <b>CLOSE {pair} ({reason})</b>")
    msg.append(f"• time: {now_utc_str()}")
    msg.append(f"• exit: {exit_price:.5f}")
    msg.append(f"• PnL: {pnl:+.5f}")
    msg.append(f"• pair stats: trades {st['trades']}, WR {wr_pair:.2f}%, PnL {st['pnl']:.5f}")
    msg.append(f"• total: trades {state['total']['trades']}, WR {wr_total:.2f}%, PnL {state['total']['pnl']:.5f}")
    msg.append(f"• balance: {state['equity']:.5f}   (Δ {state['equity']-state['start_equity']:+.5f} | {((state['equity']/state['start_equity'])-1)*100:.2f}%)")
    msg.append(f"• since start: {((state['equity']/state['start_equity'])-1)*100:.2f}%   (lev {LEVERAGE:.1f}x, fee {FEE_PCT:.3%})")
    tg_send("\n".join(msg))


# =========================
# ------- LOOP ------------
# =========================

def pair_loop(pair: str):
    tg_send(f"✅ Loop started for <b>{pair}</b>")
    while True:
        try:
            df = get_klines(pair, TIMEFRAME, limit=300)
            if df is None or df.empty:
                tg_send(f"⚠️ {pair} loop error: <code>no klines</code> for {pair}")
                time.sleep(TICK_SECONDS)
                continue

            # индикаторы для сигналов
            df["ema"] = ema(df["close"], EMA_LEN)
            df["atr"] = atr(df, ATR_LEN)

            # последовательность: сначала проверяем закрытие, потом новые открытия
            check_close(pair, df)
            try_open(pair, df)

        except Exception as e:
            tg_send(f"⚠️ {pair} loop error: {e}")

        time.sleep(TICK_SECONDS)

# =========================
# ----- TELEGRAM BOT ------
# =========================

def cmd_status(update, context):
    lines = [f"📊 <b>STATUS {now_utc_str()}</b>"]
    total_trades = state["total"]["trades"]
    total_wr = (state["total"]["wins"] / total_trades * 100.0) if total_trades else 0.0
    for p in PAIRS:
        st = state["stats"][p]
        wr = (st["wins"] / st["trades"] * 100.0) if st["trades"] else 0.0
        lines.append(f"<b>{p}</b> • trades: {st['trades']}  WR: {wr:.2f}%  PnL: {st['pnl']:.5f}")
        pos = state["open_positions"].get(p)
        if pos:
            lines.append(f"{'%.5f' % pos['entry']}  pos: {pos['dir']} @ {pos['entry']:.5f} (TP {pos['tp']:.5f} / SL {pos['sl']:.5f})")
        else:
            lines.append("0.00000  pos: —")
    lines.append("—")
    lines.append(f"<b>TOTAL</b> • trades: {total_trades}  WR: {total_wr:.2f}%  PnL: {state['total']['pnl']:.5f}")
    lines.append(f"equity: {state['equity']:.5f}  ({((state['equity']/state['start_equity'])-1)*100:.2f}% с начала)")
    lines.append(f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT:.3%}")
    update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

def start_bot():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("status", cmd_status))

    # очистим возможные «висячие» апдейты (аналог drop_pending_updates)
    try:
        requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                     params={"offset": -1}, timeout=10)
    except Exception:
        pass

    # запуск
    updater.start_polling(clean=True)
    tg_send(f"🤖 <b>mybot9</b> started successfully!\n"
            f"Mode: <b>{'DEMO' if DEMO_MODE else MODE}</b> | Leverage <b>{LEVERAGE:.1f}x</b> | Fee <b>{FEE_PCT:.3%}</b> | Risk <b>{RISK_PCT:.1f}%</b>\n"
            f"Pairs: {', '.join(PAIRS)} | TF <b>{TIMEFRAME}</b> | Tick <b>{TICK_SECONDS}s</b>\n"
            f"Balance: <b>{state['equity']:.2f}</b>  USDT")

    # фоновые потоки по каждому инструменту
    for p in PAIRS:
        threading.Thread(target=pair_loop, args=(p,), daemon=True).start()

    updater.idle()

# =========================
# ---- HEALTH SERVER  -----
# =========================

class Quiet(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        return
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def start_health_http_server():
    port = int(os.getenv("PORT", "8080"))
    try:
        httpd = HTTPServer(("0.0.0.0", port), Quiet)
    except OSError:
        # порт занят — не мешаем основному боту
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

# =========================
# --------- MAIN ----------
# =========================

if __name__ == "__main__":
    start_health_http_server()
    start_bot()
