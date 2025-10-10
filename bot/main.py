# bot/main.py
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import queue
import math
import signal
import random
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# === telegram v13 ===
from telegram import ParseMode, Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext

# -----------------------------
#            CONFIG
# -----------------------------

ENV = os.getenv

TELEGRAM_TOKEN   = ENV("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = ENV("TELEGRAM_CHAT_ID", "").strip()
USE_WEBHOOK      = ENV("USE_WEBHOOK", "1") == "1"
PUBLIC_URL       = ENV("PUBLIC_URL", "").strip()
WEBHOOK_SECRET   = ENV("WEBHOOK_SECRET", "hook").strip()
PORT             = int(ENV("PORT", "8080"))

# торговые параметры (как в твоём ENV)
PAIRS      = [p.strip() for p in ENV("PAIRS", "BTCUSDT,ETHUSDT").split(",") if p.strip()]
TIMEFRAME  = ENV("TIMEFRAME", "5m")
RISK_PCT   = float(ENV("RISK_PCT", "1"))
TP_PCT     = float(ENV("TP_PCT", "0.35"))
ATR_LEN    = int(ENV("ATR_LEN", "14"))
ATR_MULT   = float(ENV("ATR_MULT_SL", "1"))
EMA_LEN    = int(ENV("EMA_LEN", "100"))
EMA_SLOPE  = int(ENV("EMA_SLOPE_BARS", "8"))
COOLDOWN_S = int(ENV("COOLDOWN_SEC", "240"))
FEE_PCT    = float(ENV("FEE_PCT", "0.0006"))
LEVERAGE   = float(ENV("LEVERAGE", "5"))
TZ_NAME    = ENV("TX", "UTC")

DEMO_MODE  = ENV("DEMO_MODE", "true").lower() == "true"
DRY_RUN    = ENV("DRY_RUN", "false").lower() == "true"
DEMO_BAL   = float(ENV("DEMO_START_BALANCE", "5000"))
DAILY_SUM  = ENV("DAILY_SUMMARY", "1") == "1"

# лог
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bot")

# -----------------------------
#      STATE & DATA MODELS
# -----------------------------

@dataclass
class Position:
    side: str            # LONG / SHORT
    entry: float
    tp: float
    sl: float
    qty: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PairStats:
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0
    last_trade_ts: float = 0.0
    pos: Optional[Position] = None

@dataclass
class GlobalState:
    equity: float = DEMO_BAL
    pairs: Dict[str, PairStats] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    tz: timezone = timezone.utc

STATE = GlobalState()
for p in PAIRS:
    STATE.pairs[p] = PairStats()

# -----------------------------
#      UTILS / HELPERS
# -----------------------------

def tz_now() -> datetime:
    return datetime.now(STATE.tz)

def pct(a: float) -> str:
    return f"{a:.2f}%"

def fmt(v: float) -> str:
    return f"{v:.5f}".rstrip("0").rstrip(".")

def human_equity() -> str:
    with STATE.lock:
        delta = STATE.equity - DEMO_BAL
        d_pct = (delta / DEMO_BAL * 100) if DEMO_BAL else 0
        return f"{STATE.equity:.2f}  ({d_pct:+.2f}% с начала)"

def thread_name(pair: str) -> str:
    return f"loop-{pair}"

# -----------------------------
#  DEMO PRICE FEED (заглушка)
# -----------------------------
# Чтобы код был полностью автономным и не падал, используем простую
# псевдо-цену. Торговая логика остаётся простой и безопасной.
# Если у тебя есть свой реальный фид/биржа — подключай внутри get_price().

_random = random.Random()

def get_price(pair: str) -> float:
    # простая псевдо-цена, детерминированная по паре
    base = 121000 if pair.startswith("BTC") else 4300
    noise = _random.uniform(-200, 200) if pair.startswith("BTC") else _random.uniform(-10, 10)
    return max(1.0, base + noise)

# -----------------------------
#      STRATEGY (упрощённо)
# -----------------------------
# Логика: при отсутствии позиции и без кулдауна — открываем "сигнал" HAMMER
# в направлении случайного импульса; ставим TP/SL по процентам и ATR-множ.
# Это каркас — сюда можно вставить твою реальную стратегию.

def has_cooldown(ps: PairStats) -> bool:
    return (time.time() - ps.last_trade_ts) < COOLDOWN_S

def decide_signal(pair: str) -> Optional[Position]:
    price = get_price(pair)
    # псевдо-наклон "EMA"
    slope = _random.uniform(-1, 1)
    side = "LONG" if slope >= 0 else "SHORT"
    tp = price * (1 + TP_PCT/100) if side == "LONG" else price * (1 - TP_PCT/100)
    # SL через ATR_MULT (в процентах ~ ATR_MULT * 0.2%)
    sl_gap = 0.2 * ATR_MULT / 100
    sl = price * (1 - sl_gap) if side == "LONG" else price * (1 + sl_gap)
    # размер позиции по риску от equity
    with STATE.lock:
        eq = STATE.equity
    risk_usd = eq * (RISK_PCT/100)
    # условный стоповый риск ~ sl_gap
    per_unit_risk = price * sl_gap
    qty = max(0.00002, min(risk_usd / max(per_unit_risk, 1e-6), 10))  # ограничим для демо

    return Position(side=side, entry=price, tp=tp, sl=sl, qty=qty)

def mark_to_market(ps: PairStats, pair: str) -> None:
    """Закрытие по ТП/SL в демо-режиме."""
    if not ps.pos:
        return
    price = get_price(pair)
    pos = ps.pos
    hit_tp = price >= pos.tp if pos.side == "LONG" else price <= pos.tp
    hit_sl = price <= pos.sl if pos.side == "LONG" else price >= pos.sl
    if hit_tp or hit_sl:
        close_price = pos.tp if hit_tp else pos.sl
        pnl = (close_price - pos.entry) * pos.qty if pos.side == "LONG" else (pos.entry - close_price) * pos.qty
        fees = (pos.entry + close_price) * pos.qty * FEE_PCT
        pnl -= fees

        with STATE.lock:
            STATE.equity += pnl
            ps.trades += 1
            if pnl > 0:
                ps.wins += 1
            ps.pnl += pnl
            ps.pos = None
            ps.last_trade_ts = time.time()

        emoji = "✅" if pnl > 0 else "❌"
        send_log(f"""{emoji} CLOSE {pair} ({'TP' if hit_tp else 'SL'})
• time: {utcnow_str()}
• exit: {fmt(close_price)}
• PnL: {pnl:+.5f}
• pair stats: trades {ps.trades}, WR {wr(ps):.2f}%, PnL {ps.pnl:+.5f}
• total: equity {human_equity()}
• since start: {since_start()}
""")

def wr(ps: PairStats) -> float:
    return (ps.wins / ps.trades * 100) if ps.trades else 0.0

def since_start() -> str:
    with STATE.lock:
        delta = STATE.equity - DEMO_BAL
        pct_delta = (delta/DEMO_BAL*100) if DEMO_BAL else 0
    return f"{pct_delta:+.2f}%"

# -----------------------------
#        BOT HANDLERS
# -----------------------------

def utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_log(text: str):
    if TELEGRAM_CHAT_ID:
        try:
            BOT.send_message(int(TELEGRAM_CHAT_ID), text)
        except Exception as e:
            log.warning(f"send_log error: {e}")

def cmd_start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🤖 Бот запущен. Используй /status, чтобы посмотреть статистику.\n"
        f"Mode: {'DEMO' if DEMO_MODE else 'LIVE'} | Leverage {LEVERAGE:.1f}x | Fee {FEE_PCT*100:.3f}% | Risk {RISK_PCT:.1f}%\n"
        f"Pairs: {', '.join(PAIRS)} | TF {TIMEFRAME}"
    )

def cmd_status(update: Update, context: CallbackContext):
    lines = [f"📊 STATUS {utcnow_str()}"]
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0

    with STATE.lock:
        for pair, ps in STATE.pairs.items():
            total_trades += ps.trades
            total_wins += ps.wins
            total_pnl += ps.pnl
            pos_line = "—"
            if ps.pos:
                pos = ps.pos
                pos_line = (f"{pos.qty:.6f}  pos: {pos.side} @ {fmt(pos.entry)} "
                            f"(TP {fmt(pos.tp)} / SL {fmt(pos.sl)})")
            lines.append(f"{pair} • trades: {ps.trades}  WR: {wr(ps):.2f}%  PnL: {ps.pnl:+.5f}\n{pos_line}")

    lines.append("—")
    wr_total = (total_wins/total_trades*100) if total_trades else 0.0
    lines.append(f"TOTAL • trades: {total_trades}  WR: {wr_total:.2f}%  PnL: {total_pnl:+.5f}")
    lines.append(f"equity: {human_equity()}")
    lines.append(f"leverage: {LEVERAGE:.1f}x  fee: {FEE_PCT*100:.3f}%")
    update.message.reply_text("\n".join(lines))

# -----------------------------
#        LOOPS PER PAIR
# -----------------------------

def loop_pair(pair: str):
    threading.current_thread().name = thread_name(pair)
    log.info(f"Loop started for {pair}")
    send_log(f"✅ Loop started for <b>{pair}</b>",)

    ps = STATE.pairs[pair]
    while not STATE.stop_event.is_set():
        try:
            # закрыть по ТП/SL если открыто
            mark_to_market(ps, pair)

            # открыть новую, если пусто и нет кулдауна
            if (ps.pos is None) and not has_cooldown(ps):
                pos = decide_signal(pair)
                if pos:
                    with STATE.lock:
                        ps.pos = pos
                        ps.last_trade_ts = time.time()

                    send_log(f"""🔴 OPEN {pair} {pos.side}
• time: {utcnow_str()}
• entry: {fmt(pos.entry)}
• qty: {pos.qty:.6f}
• TP: {fmt(pos.tp)}   SL: {fmt(pos.sl)}
• signal: hammer, slope {random.uniform(-1,1):.5f}, ATR ~
• mode: {"DEMO" if DEMO_MODE else "LIVE"}
""")
            time.sleep(10)  # Tick из ENV (10с)
        except Exception as e:
            log.exception(f"loop error {pair}: {e}")
            time.sleep(3)

# -----------------------------
#            RUN
# -----------------------------

def start_all_loops():
    for pair in PAIRS:
        t = threading.Thread(target=loop_pair, args=(pair,), daemon=True)
        t.start()

def stop_all(*_):
    STATE.stop_event.set()

def main():
    if not TELEGRAM_TOKEN:
        log.critical("TELEGRAM_TOKEN обязателен")
        sys.exit(1)

    # таймзона
    if TZ_NAME.upper() == "UTC":
        STATE.tz = timezone.utc

    global BOT
    BOT = Bot(TELEGRAM_TOKEN)

    updater = Updater(TELEGRAM_TOKEN, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start",  cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))

    # Запуск торговых циклов
    start_all_loops()

    # ВЫБОР РЕЖИМА: Webhook ИЛИ Polling
    if USE_WEBHOOK:
        if not PUBLIC_URL:
            log.critical("PUBLIC_URL обязателен при USE_WEBHOOK=1")
            sys.exit(1)

        hook_path = f"webhook/{WEBHOOK_SECRET}"
        full_url = f"{PUBLIC_URL.rstrip('/')}/{hook_path}"
        log.info(f"Setting webhook to {full_url}")

        # ВНИМАНИЕ: никаких других HTTP-серверов на 8080!
        updater.start_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=hook_path,
            webhook_url=full_url,
            drop_pending_updates=True,   # вместо устаревшего clean=
        )
    else:
        # Отключим хук на всякий случай и запустим polling
        try:
            BOT.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass
        log.info("Starting polling…")
        updater.start_polling(drop_pending_updates=True)

    signal.signal(signal.SIGINT,  stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    log.info("Scheduler started")
    updater.idle()

if __name__ == "__main__":
    main()
