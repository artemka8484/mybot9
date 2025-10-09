# bot/main.py
# -*- coding: utf-8 -*-
"""
mybot9 — простая демо-стратегия (тест #9):
- Пары из ENV (по-умолчанию BTCUSDT, ETHUSDT)
- Таймфрейм: из ENV (по-умолчанию 5m)
- TP фиксированный в процентах (TP_PCT, напр. 0.0035 = 0.35%)
- SL = ATR * ATR_MULT_SL
- Вход по 6 паттернам + фильтр тренда EMA100 slope (EMA_SLOPE_BARS)
- Жёсткий «trend guard»: LONG разрешён только при slope>0, SHORT — только при slope<0
- Риск от баланса (RISK_PCT), плечо (LEVERAGE), демо-режим с учётом комиссии FEE_PCT
- Кулдаун COOLDOWN_SEC между сделками по инструменту
- Телеграм: /start, /status
"""

import os, time, threading, math, json, traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import requests
import numpy as np
import pandas as pd

from telegram import ParseMode
from telegram.ext import Updater, CommandHandler

# ---------- ENV ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0") or "0")

PAIRS = [s.strip().upper() for s in os.getenv("PAIRS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "5m").lower()  # 1m/3m/5m/15m/1h...
TICK_SEC = int(os.getenv("TICK_SEC", "10"))       # частота цикла
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "240"))

EMA_LEN = int(os.getenv("EMA_LEN", "100"))
EMA_SLOPE_BARS = int(os.getenv("EMA_SLOPE_BARS", "8"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1"))   # SL = ATR*mult

TP_PCT = float(os.getenv("TP_PCT", "0.0035"))        # 0.35% по-умолчанию
RISK_PCT = float(os.getenv("RISK_PCT", "1"))         # % риска от equity на сделку
LEVERAGE = float(os.getenv("LEVERAGE", "5"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.0006"))      # 0.06% (2 стороны ~0.12%)

MODE = os.getenv("MODE", "DEMO").upper()
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
DEMO_START_BALANCE = float(os.getenv("DEMO_START_BALANCE", "5000"))

DAILY_SUMMARY = os.getenv("DAILY_SUMMARY", "1") == "1"

BINANCE_BASE = "https://api.binance.com"  # публичные свечи берём у Binance

# ---------- Глобальное состояние ----------
state_lock = threading.Lock()
state: Dict[str, Any] = {
    "equity": DEMO_START_BALANCE,
    "start_equity": DEMO_START_BALANCE,
    "pairs": {},
    "day_anchor": datetime.now(timezone.utc).date(),
}
for p in PAIRS:
    state["pairs"][p] = {
        "pos": None,               # dict|None
        "last_entry_ts": 0,
        "stats": {"trades": 0, "wins": 0, "pnl": 0.0},
    }

# ---------- Утилиты ----------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def tf_to_binance_interval(tf: str) -> str:
    return tf

def send(msg: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[TG disabled]", msg)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as e:
        print("Telegram error:", e)

def fmt_pct(x):
    return f"{x*100:.2f}%"

def fmt_num(x):
    return f"{x:.5f}"

# ---------- Маркет данные ----------
def fetch_klines(pair: str, limit: int = 500) -> Optional[pd.DataFrame]:
    """
    Свечи с Binance: openTime, open, high, low, close, volume, ...
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": pair, "interval": tf_to_binance_interval(TIMEFRAME), "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        cols = ["t","open","high","low","close","vol","ct","qv","n","tbav","tbqv","i"]
        df = pd.DataFrame(data, columns=cols)[["t","open","high","low","close","vol"]]
        for c in ["open","high","low","close","vol"]:
            df[c] = df[c].astype(float)
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        return df
    except Exception:
        return None

def talib_basic(df: pd.DataFrame) -> pd.DataFrame:
    # EMA
    df["ema"] = df["close"].ewm(span=EMA_LEN, adjust=False).mean()
    # ATR (True Range)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()
    # slope ema за N баров
    df["ema_slope"] = df["ema"] - df["ema"].shift(EMA_SLOPE_BARS)
    return df

# ---------- Паттерны (6 штук) ----------
def pattern_signals(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Берём последний бар (= -2, т.к. -1 может быть незакрытым у некоторых бирж),
    но для надёжности используем -2.
    """
    if len(df) < 10:
        return {"bull_engulf": False, "bear_engulf": False, "hammer": False, "shooting_star": False, "breakout_up": False, "breakout_down": False}
    i = -2
    o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
    op, hp, lp, cp = df["open"].iloc[i-1], df["high"].iloc[i-1], df["low"].iloc[i-1], df["close"].iloc[i-1]

    body = abs(c-o)
    rng = h-l + 1e-12
    upper = h - max(c, o)
    lower = min(c, o) - l

    # 1) Bullish Engulfing
    bull_engulf = (cp > op) is False and (c > o) and (c >= op) and (o <= cp)
    # 2) Bearish Engulfing
    bear_engulf = (cp < op) is False and (c < o) and (c <= op) and (o >= cp)
    # 3) Hammer (малое тело сверху, длинная нижняя тень)
    hammer = (c > o) and (lower / rng > 0.55) and (upper / rng < 0.2)
    # 4) Shooting star (малое тело снизу, длинная верхняя тень)
    shooting_star = (c < o) and (upper / rng > 0.55) and (lower / rng < 0.2)
    # 5) Breakout up (пробой максимума N баров)
    N = 20
    breakout_up = c > df["high"].iloc[-N-2:-2].max()
    # 6) Breakout down
    breakout_down = c < df["low"].iloc[-N-2:-2].min()

    return {
        "bull_engulf": bool(bull_engulf),
        "bear_engulf": bool(bear_engulf),
        "hammer": bool(hammer),
        "shooting_star": bool(shooting_star),
        "breakout_up": bool(breakout_up),
        "breakout_down": bool(breakout_down),
    }

# ---------- Логика входа/выхода ----------
def decide_entry(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Возвращает (side, reason) или None.
    """
    sig = pattern_signals(df)
    slope = df["ema_slope"].iloc[-2]

    # первичное направление по паттернам
    long_trigs = [sig["bull_engulf"], sig["hammer"], sig["breakout_up"]]
    short_trigs = [sig["bear_engulf"], sig["shooting_star"], sig["breakout_down"]]

    if any(long_trigs):
        side = "LONG"
        reason = ", ".join([k for k, v in sig.items() if v and k in ["bull_engulf","hammer","breakout_up"]])
    elif any(short_trigs):
        side = "SHORT"
        reason = ", ".join([k for k, v in sig.items() if v and k in ["bear_engulf","shooting_star","breakout_down"]])
    else:
        return None

    # === TREND GUARD (жёстко как в тесте #9) ===
    if side == "LONG" and slope <= 0:
        return None
    if side == "SHORT" and slope >= 0:
        return None

    return side, reason

def position_size(pair: str, price: float) -> float:
    with state_lock:
        eq = state["equity"]
    risk_cash = eq * (RISK_PCT / 100.0)
    # на фьючерсах с плечом можно контролировать через риск -> приблизим qty так,
    # чтобы потенциальный убыток по SL (в %) не превышал risk_cash.
    # Если SL считаем как ATR * mult -> дельта цены sl_dist
    sl_dist = max(1e-8, state["pairs"][pair].get("last_atr", 0.0) * ATR_MULT_SL)
    # риск ~ qty * sl_dist
    qty = risk_cash / sl_dist
    # учитываем плечо (на фьючах маржа ~ price*qty/leverage)
    # Для демо достаточно ограничить размер относительно эквити:
    max_qty_by_leverage = (eq * LEVERAGE) / max(price, 1e-8)
    qty = min(qty, max_qty_by_leverage)
    return max(qty, 0.0)

def open_position(pair: str, side: str, price: float, df: pd.DataFrame, reason: str):
    atr = float(df["atr"].iloc[-2])
    with state_lock:
        st = state["pairs"][pair]
        if st["pos"] is not None:
            return
        # cooldown
        if time.time() - st["last_entry_ts"] < COOLDOWN_SEC:
            return
        st["last_atr"] = atr
    qty = position_size(pair, price)
    if qty <= 0:
        return
    # цели
    if side == "LONG":
        tp = price * (1 + TP_PCT)
        sl = price - ATR_MULT_SL * atr
    else:
        tp = price * (1 - TP_PCT)
        sl = price + ATR_MULT_SL * atr

    pos = {
        "side": side, "entry": price, "qty": qty,
        "tp": tp, "sl": sl, "ts": time.time(), "reason": reason
    }
    with state_lock:
        state["pairs"][pair]["pos"] = pos
        state["pairs"][pair]["last_entry_ts"] = time.time()

    send(
        "🔴 <b>OPEN {pair} {side}</b>\n"
        "• time: {t}\n"
        "• entry: {entry:.5f}\n"
        "• qty: {qty:.6f}\n"
        "• TP: {tp:.5f}   SL: {sl:.5f}\n"
        "• signal: {reason}, slope {slope:.5f}, ATR {atr:.5f}\n"
        "• mode: {mode}".format(
            pair=pair, side=side, t=now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
            entry=price, qty=qty, tp=tp, sl=sl, reason=reason,
            slope=df["ema_slope"].iloc[-2], atr=atr, mode="DEMO" if DEMO_MODE else "LIVE"
        )
    )

def close_position(pair: str, exit_price: float, tag: str):
    with state_lock:
        pos = state["pairs"][pair]["pos"]
        if pos is None:
            return
        side = pos["side"]
        entry = pos["entry"]
        qty = pos["qty"]
        # PnL (без учёта фи и плечо уже в размере позиции)
        pnl = (exit_price - entry) * qty if side == "LONG" else (entry - exit_price) * qty
        # комиссии (2 стороны)
        fee = (abs(entry) + abs(exit_price)) * qty * FEE_PCT
        pnl -= fee
        # обновляем equity
        state["equity"] += pnl
        # статистика
        st = state["pairs"][pair]["stats"]
        st["trades"] += 1
        st["pnl"] += pnl
        if pnl > 0:
            st["wins"] += 1
        state["pairs"][pair]["pos"] = None

        total_trades = sum(state["pairs"][p]["stats"]["trades"] for p in state["pairs"])
        total_wins = sum(state["pairs"][p]["stats"]["wins"] for p in state["pairs"])
        total_pnl = sum(state["pairs"][p]["stats"]["pnl"] for p in state["pairs"])
        wr_pair = (st["wins"] / st["trades"] * 100.0) if st["trades"] else 0.0
        wr_total = (total_wins / total_trades * 100.0) if total_trades else 0.0
        delta = state["equity"] - state["start_equity"]
        delta_pct = 100.0 * (delta / state["start_equity"])

    mark = "✅" if pnl > 0 else "❌"
    send(
        "{mark} <b>CLOSE {pair} ({tag})</b>\n"
        "• time: {t}\n"
        "• exit: {exit:.5f}\n"
        "• PnL: {pnl:+.5f}\n"
        "• pair stats: trades {pt}, WR {pwr:.2f}%, PnL {ppnl:.5f}\n"
        "• total: trades {tt}, WR {twr:.2f}%, PnL {tpnl:.5f}\n"
        "• balance: {eq:.5f}  (Δ {d:+.5f} | {dp:.2f}%)\n"
        "• since start: {dp:.2f}%   (lev {lev:.1f}x, fee {fee:.3f}%)".format(
            mark=mark, pair=pair, tag=tag, t=now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
            exit=exit_price, pnl=pnl,
            pt=st["trades"], pwr=wr_pair, ppnl=st["pnl"],
            tt=total_trades, twr=wr_total, tpnl=total_pnl,
            eq=state["equity"], d=delta, dp=delta_pct, lev=LEVERAGE, fee=FEE_PCT*100
        )
    )

def check_exit(pair: str, last_price: float):
    with state_lock:
        pos = state["pairs"][pair]["pos"]
    if not pos:
        return
    if pos["side"] == "LONG":
        if last_price >= pos["tp"]:
            close_position(pair, pos["tp"], "TP")
        elif last_price <= pos["sl"]:
            close_position(pair, pos["sl"], "SL")
    else:
        if last_price <= pos["tp"]:
            close_position(pair, pos["tp"], "TP")
        elif last_price >= pos["sl"]:
            close_position(pair, pos["sl"], "SL")

# ---------- Цикл по паре ----------
def pair_loop(pair: str):
    send(f"✅ Loop started for <b>{pair}</b>")
    while True:
        try:
            df = fetch_klines(pair, limit=500)
            if df is None or len(df) < max(ATR_LEN, EMA_LEN)+EMA_SLOPE_BARS+5:
                send(f"⚠️ {pair} loop error: no  klines  for {pair}")
                time.sleep(TICK_SEC)
                continue

            df = talib_basic(df)
            last_price = float(df["close"].iloc[-1])

            # 1) Выход по TP/SL
            check_exit(pair, last_price)

            # 2) Вход (если позиции нет)
            with state_lock:
                has_pos = state["pairs"][pair]["pos"] is not None
            if not has_pos:
                dec = decide_entry(df)
                if dec:
                    side, reason = dec
                    open_position(pair, side, float(df["close"].iloc[-2]), df, reason)

            # 3) Ежедневная сводка (UTC)
            if DAILY_SUMMARY:
                with state_lock:
                    if state["day_anchor"] != now_utc().date():
                        state["day_anchor"] = now_utc().date()
                        send(status_text())

        except Exception as e:
            print(f"{pair} loop error:", e)
            traceback.print_exc()
        time.sleep(TICK_SEC)

# ---------- Команды TG ----------
def status_text() -> str:
    with state_lock:
        lines = [f"📊 <b>STATUS {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}</b>"]
        total_tr, total_wins, total_pnl = 0, 0, 0.0
        for pair, ps in state["pairs"].items():
            st = ps["stats"]
            pos = ps["pos"]
            total_tr += st["trades"]; total_wins += st["wins"]; total_pnl += st["pnl"]
            wr = (st["wins"]/st["trades"]*100.0) if st["trades"] else 0.0
            lines.append(f"<b>{pair}</b> • trades: {st['trades']}  WR: {wr:.2f}%  PnL: {fmt_num(st['pnl'])}")
            if pos:
                lines.append(f"{pos['qty']:.6f}  pos: <b>{pos['side']}</b> @ {pos['entry']:.5f} (TP {pos['tp']:.5f} / SL {pos['sl']:.5f})")
            else:
                lines.append("0.00000  pos: —")
        twr = (total_wins/total_tr*100.0) if total_tr else 0.0
        lines.append("—")
        lines.append(f"<b>TOTAL</b> • trades: {total_tr}  WR: {twr:.2f}%  PnL: {fmt_num(total_pnl)}")
        delta = state["equity"] - state["start_equity"]
        dp = 100.0 * delta / state["start_equity"]
        lines.append(f"equity: {state['equity']:.5f}  ({dp:.2f}% с начала)")
        lines.append(f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT*100:.3f}%")
    return "\n".join(lines)

def cmd_start(update, context):
    msg = (
        "🤖 <b>mybot9 started successfully!</b>\n"
        f"Mode: <b>{'DEMO' if DEMO_MODE else 'LIVE'}</b> | Leverage <b>{LEVERAGE:.1f}x</b> | Fee <b>{FEE_PCT*100:.3f}%</b> | "
        f"Risk <b>{RISK_PCT:.1f}%</b>\n"
        f"Pairs: {', '.join(PAIRS)} | TF <b>{TIMEFRAME}</b> | Tick <b>{TICK_SEC}s</b>\n"
        f"Balance: {state['equity']:.2f}  USDT"
    )
    send(msg)
    update.message.reply_text("ok")

def cmd_status(update, context):
    update.message.reply_text(status_text(), parse_mode=ParseMode.HTML, disable_web_page_preview=True)

# ---------- Main ----------
def main():
    # Telegram
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))

    # Запуск потоков по парам
    for p in PAIRS:
        th = threading.Thread(target=pair_loop, args=(p,), daemon=True)
        th.start()

    # Поллинг. В Koyeb/Heroku иногда параллельные инстансы дают конфликт.
    # drop_pending_updates=True аналог clean=True, чтобы не зависать на древних апдейтах.
    try:
        updater.start_polling(drop_pending_updates=True)
        # ping-сервер здоровья (best-effort), игнорируем занятый порт
        try:
            import http.server, socketserver
            class Quiet(http.server.SimpleHTTPRequestHandler):
                def log_message(self, *a): pass
            PORT = 8080
            def serve():
                with socketserver.TCPServer(("0.0.0.0", PORT), Quiet) as httpd:
                    httpd.serve_forever()
            threading.Thread(target=serve, daemon=True).start()
        except Exception:
            pass
        # Сообщение о запуске
        send(
            "🤖 <b>mybot9 started successfully!</b>\n"
            f"Mode: <b>{'DEMO' if DEMO_MODE else 'LIVE'}</b> | Leverage <b>{LEVERAGE:.1f}x</b> | Fee <b>{FEE_PCT*100:.3f}%</b> | Risk <b>{RISK_PCT:.1f}%</b>\n"
            f"Pairs: {', '.join(PAIRS)} | TF <b>{TIMEFRAME}</b> | Tick <b>{TICK_SEC}s</b>\n"
            f"Balance: {state['equity']:.2f}  USDT"
        )
        updater.idle()
    except Exception as e:
        # Если где-то крутится второй инстанс — Telegram вернёт Conflict.
        # Просто логируем и продолжаем фоны (бот всё равно торгует/шлёт send() через API).
        send(f"⚠️ Telegram polling error: {e}")

if __name__ == "__main__":
    main()
