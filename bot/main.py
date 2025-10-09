# bot/main.py
# -*- coding: utf-8 -*-
"""
Телеграм-бот демо-трейдинга с реалистичным учётом плеча и комиссий.
— без шума: сообщения только при ОТКРЫТИИ/ЗАКРЫТИИ позиции + /status
— qty = equity * RISK_PCT * LEVERAGE / price  (исправлено)
— PnL учитывает комиссию на вход и выход
— winrate и pnl по каждому активу и по счёту
"""

import os
import json
import time
import math
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
import numpy as np
import pandas as pd

from telegram import ParseMode
from telegram.utils.request import Request
from telegram.ext import Updater, CommandHandler, CallbackContext

# -------------------- ENV --------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
PAIRS = [p.strip().upper() for p in os.getenv(
    "PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
).split(",") if p.strip()]

INTERVAL = os.getenv("TF", "1m")
TICK_SECONDS = int(os.getenv("TICK_SECONDS", "10"))  # частота тика на пару
HISTORY = int(os.getenv("HISTORY", "200"))          # кол-во свечей

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in {"1", "true", "yes"}
LEVERAGE = float(os.getenv("LEVERAGE", "5"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.03"))     # риск на сделку от equity
FEE_PCT = float(os.getenv("FEE_PCT", "0.0006"))     # 0.06% на каждую сторону
EQUITY_START = float(os.getenv("EQUITY_START", "1000"))

# -------------------- STATE --------------------
equity = EQUITY_START
start_equity = EQUITY_START

positions = {p: None for p in PAIRS}
stats = {
    p: {"trades": 0, "wins": 0, "pnl": 0.0}
    for p in PAIRS
}
stats_total = {"trades": 0, "wins": 0, "pnl": 0.0}

# чтобы не дублировать открытие позиции в один и тот же тик
last_signal_ts = {p: 0 for p in PAIRS}


# -------------------- HEALTH --------------------
def start_health_http_server():
    class Quiet(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # тихий лог
            return

        def do_GET(self):
            if self.path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

    # иногда порт уже занят — просто пропустим
    try:
        port = int(os.getenv("PORT", "8080"))
        httpd = HTTPServer(("0.0.0.0", port), Quiet)
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
    except Exception as e:
        print(f"[WARN] Health server bind failed: {e}")


# -------------------- MARKET DATA --------------------
def _mexc_klines_url(symbol, interval, limit):
    # MEXC spot kline endpoint
    return f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

def fetch_klines(pair: str) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками: t, open, high, low, close, vol
    Делает код устойчивым к 8/12-элементным массивам свечи.
    """
    url = _mexc_klines_url(pair, INTERVAL, HISTORY)
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError(f"Bad klines response: {data}")

    rows = []
    for row in data:
        # ожидаем минимум: [openTime, open, high, low, close, volume, ...]
        # MEXC часто отдаёт 12 полей, иногда агрегаторы — 8. Берём только первые 6.
        if len(row) < 6:
            continue
        t = int(row[0])
        o = float(row[1])
        h = float(row[2])
        l = float(row[3])
        c = float(row[4])
        v = float(row[5])
        rows.append([t, o, h, l, c, v])

    if not rows:
        raise ValueError("Empty klines")

    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vol"])
    return df


# -------------------- INDICATORS & STRATEGY --------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def detect_patterns(df: pd.DataFrame):
    """
    4 простых свечных сигнала: булл/бер энга́льфинг, молот, падающая звезда
    """
    o = df["open"].iloc[-2]
    c = df["close"].iloc[-2]
    h = df["high"].iloc[-2]
    l = df["low"].iloc[-2]
    body = abs(c - o)
    range_ = h - l + 1e-9

    bull_engulf = (c > o) and (c - o > body * 1.2) and (o < df["close"].iloc[-3] < c)
    bear_engulf = (o > c) and (o - c > body * 1.2) and (c < df["close"].iloc[-3] < o)

    # молот/падающая звезда по простым правилам
    lower_tail = (min(o, c) - l)
    upper_tail = (h - max(o, c))
    hammer = (lower_tail > body * 2) and (upper_tail < body * 0.5)
    shooting = (upper_tail > body * 2) and (lower_tail < body * 0.5)

    return {
        "bull_engulf": bool(bull_engulf),
        "bear_engulf": bool(bear_engulf),
        "hammer": bool(hammer),
        "shooting": bool(shooting),
    }

def strategy_signal(df: pd.DataFrame):
    """
    Возвращает ('LONG'|'SHORT'|None, reason, tp, sl) по последней закрытой свече.
    Логика: наклон EMA48 + свечные паттерны + простой breakout.
    TP/SL считаем через ATR.
    """
    if len(df) < 60:
        return None, None, None, None

    ema48 = ema(df["close"], 48)
    slope = float(ema48.diff().tail(5).mean())  # средний наклон по 5 баров
    atr14 = float(atr(df, 14).iloc[-2])

    patt = detect_patterns(df)
    last_close = float(df["close"].iloc[-2])
    prev_high = float(df["high"].iloc[-2])
    prev_low = float(df["low"].iloc[-2])

    # сигналы
    long_trig = (slope > 0 and (patt["bull_engulf"] or patt["hammer"])) \
                or (last_close > prev_high and slope >= 0)  # небольшой breakout

    short_trig = (slope < 0 and (patt["bear_engulf"] or patt["shooting"])) \
                 or (last_close < prev_low and slope <= 0)  # breakout вниз

    # TP/SL через ATR (чуть шире, чтобы сделки были реалистичнее)
    tp_mul = 1.5
    sl_mul = 1.2

    if long_trig:
        tp = last_close + tp_mul * atr14
        sl = last_close - sl_mul * atr14
        return "LONG", _reason("long", patt, slope, atr14), tp, sl

    if short_trig:
        tp = last_close - tp_mul * atr14
        sl = last_close + sl_mul * atr14
        return "SHORT", _reason("short", patt, slope, atr14), tp, sl

    return None, None, None, None

def _reason(side, patt, slope, atr14):
    used = []
    for k, v in patt.items():
        if v:
            used.append(k)
    if abs(slope) > 0:
        used.append(f"slope {round(slope, 5)}")
    used.append(f"ATR {round(atr14, 5)}")
    return ", ".join(used) if used else side


# -------------------- RISK, PNL --------------------
def calc_qty(price: float) -> float:
    """
    Исправленная формула:
    qty = (equity * RISK_PCT) * LEVERAGE / price
    """
    position_value = max(0.0, equity) * RISK_PCT * LEVERAGE
    qty = position_value / max(1e-9, price)
    return qty

def calc_fees(entry_price: float, exit_price: float, qty: float) -> float:
    """
    Комиссия на вход и на выход: (entry*qty + exit*qty) * FEE_PCT
    """
    return (entry_price * qty + exit_price * qty) * FEE_PCT

def close_position(pair: str, exit_price: float, now_ts: int):
    global equity, stats_total

    pos = positions[pair]
    if not pos:
        return

    side = pos["side"]
    entry = pos["entry"]
    qty = pos["qty"]

    if side == "LONG":
        gross = (exit_price - entry) * qty
    else:
        gross = (entry - exit_price) * qty

    fees = calc_fees(entry, exit_price, qty)
    pnl = gross - fees

    # обновляем equity и статистику
    old_equity = equity
    equity = round(equity + pnl, 8)

    stats[pair]["trades"] += 1
    stats[pair]["pnl"] = round(stats[pair]["pnl"] + pnl, 8)
    stats[pair]["wins"] += 1 if pnl >= 0 else 0

    stats_total["trades"] += 1
    stats_total["pnl"] = round(stats_total["pnl"] + pnl, 8)
    stats_total["wins"] += 1 if pnl >= 0 else 0

    # сообщение
    wr_pair = (stats[pair]["wins"] / stats[pair]["trades"] * 100.0) if stats[pair]["trades"] else 0.0
    wr_total = (stats_total["wins"] / stats_total["trades"] * 100.0) if stats_total["trades"] else 0.0

    status_emoji = "✅" if pnl >= 0 else "❌"
    sign = "+" if pnl >= 0 else "−"

    msg = (
        f"{status_emoji} CLOSE {pair} ({'TP' if pos.get('reason')=='TP' else 'SL' if pos.get('reason')=='SL' else 'EXIT'})\n"
        f"• time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now_ts/1000))} UTC\n"
        f"• exit: {exit_price:.5f}\n"
        f"• PnL: {sign}{abs(pnl):.5f}\n"
        f"• pair stats: trades {stats[pair]['trades']}, WR {wr_pair:.2f}%, PnL {stats[pair]['pnl']:.5f}\n"
        f"• total: trades {stats_total['trades']}, WR {wr_total:.2f}%, PnL {stats_total['pnl']:.5f}\n"
        f"• balance: {equity:.5f}  (Δ {sign}{abs(pnl):.5f} | {((equity-start_equity)/max(1e-9,start_equity))*100:.2f}%)\n"
        f"• since start: {((equity-start_equity)/max(1e-9,start_equity))*100:.2f}%   (lev {LEVERAGE:.1f}x, fee {FEE_PCT*100:.3f}%)"
    )
    tg_send(msg)

    positions[pair] = None


# -------------------- LOOP --------------------
def pair_loop(pair: str):
    global equity

    while True:
        try:
            df = fetch_klines(pair)
            now_ts = int(df["t"].iloc[-1])  # открытая текущая свеча
            last_closed_idx = -2            # последняя закрытая свеча
            last_close = float(df["close"].iloc[last_closed_idx])

            # если есть позиция — проверяем TP/SL
            pos = positions[pair]
            if pos:
                side = pos["side"]
                tp = pos["tp"]
                sl = pos["sl"]

                exit_reason = None
                if side == "LONG":
                    if last_close >= tp:
                        exit_reason = "TP"
                    elif last_close <= sl:
                        exit_reason = "SL"
                else:  # SHORT
                    if last_close <= tp:
                        exit_reason = "TP"
                    elif last_close >= sl:
                        exit_reason = "SL"

                if exit_reason:
                    positions[pair]["reason"] = exit_reason
                    close_position(pair, last_close, now_ts)

            # если позиции нет — ищем сигнал
            if not positions[pair]:
                side, reason, tp, sl = strategy_signal(df)
                if side and (now_ts != last_signal_ts[pair]):  # простая защита от повтора
                    qty = calc_qty(last_close)
                    if qty <= 0:
                        time.sleep(TICK_SECONDS)
                        continue

                    positions[pair] = {
                        "side": side,
                        "entry": last_close,
                        "qty": qty,
                        "tp": tp,
                        "sl": sl,
                        "open_ts": now_ts,
                    }
                    last_signal_ts[pair] = now_ts

                    msg = (
                        f"🔴 OPEN {pair} {side}\n"
                        f"• time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now_ts/1000))} UTC\n"
                        f"• entry: {last_close:.5f}\n"
                        f"• qty: {qty:.6f}\n"
                        f"• TP: {tp:.5f}    SL: {sl:.5f}\n"
                        f"• signal: {reason if reason else 'strategy#9'}\n"
                        f"• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
                    )
                    tg_send(msg)

        except Exception as e:
            print(f"[ERROR] loop {pair}: {e}")

        time.sleep(TICK_SECONDS)


# -------------------- TELEGRAM --------------------
def tg_send(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[TG]", text)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[WARN] tg_send failed: {e}")

def cmd_status(update, context: CallbackContext):
    lines = [f"📊 STATUS {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC"]
    for p in PAIRS:
        s = stats[p]
        wr = (s["wins"] / s["trades"] * 100.0) if s["trades"] else 0.0
        pos = positions[p]
        pos_line = "—"
        if pos:
            pos_line = f"{pos['side']} @ {pos['entry']:.5f} (TP {pos['tp']:.5f} / SL {pos['sl']:.5f})"
        lines.append(
            f"{p} • trades: {s['trades']}  WR: {wr:.2f}%  PnL: {s['pnl']:.5f}\n0.00000  pos: {pos_line}"
        )
    wr_t = (stats_total["wins"] / stats_total["trades"] * 100.0) if stats_total["trades"] else 0.0
    lines.append("—")
    lines.append(f"TOTAL • trades: {stats_total['trades']}  WR: {wr_t:.2f}%  PnL: {stats_total['pnl']:.5f}")
    lines.append(f"equity: {equity:.5f}  ({((equity-start_equity)/max(1e-9,start_equity))*100:.2f}% с начала)")
    lines.append(f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT*100:.3f}%")
    update.message.reply_text("\n".join(lines))

def build_bot():
    if not TELEGRAM_TOKEN:
        print("[WARN] TELEGRAM_TOKEN not set – буду логировать в stdout")
        return None
    req = Request(con_pool_size=4, read_timeout=20, connect_timeout=20)
    updater = Updater(token=TELEGRAM_TOKEN, request_kwargs={"con_pool_size": 4}, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("status", cmd_status))
    return updater

def run_bot_threads():
    # health
    start_health_http_server()

    # telegram
    updater = build_bot()
    if updater:
        th = threading.Thread(target=updater.start_polling, daemon=True)
        th.start()
        print("Telegram polling started")

    # pairs
    for p in PAIRS:
        t = threading.Thread(target=pair_loop, args=(p,), daemon=True)
        t.start()
        print(f"Loop started for {p}")

    # keep alive
    while True:
        time.sleep(60)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("🤖 mybot9 started successfully!")
    print(f"Mode: {'DEMO' if DEMO_MODE else 'LIVE'} | Leverage {LEVERAGE}x | Fee {FEE_PCT*100:.3f}% | Risk {RISK_PCT*100:.1f}%")
    print(f"Pairs: {', '.join(PAIRS)} | TF {INTERVAL} | Tick {TICK_SECONDS}s")
    run_bot_threads()
