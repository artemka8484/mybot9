# -*- coding: utf-8 -*-
"""
Единый стабильный бот-торговец (v13 ptb).
- По умолчанию POLLING. Webhook можно включить через USE_WEBHOOK=1.
- Health server (8080) стартует только в POLLING и только если HEALTH_SERVER=1.
- Источник kline: Binance public API (без ключей).
- Стратегия: EMA + slope, ATR SL/TP, свечи: hammer, engulf, breakout.
- Комиссия учитывается, риск на сделку, cooldown между входами.
"""
import os
import time
import json
import threading
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta, timezone
from math import copysign

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from telegram import Bot, ParseMode
from telegram.ext import (
    Updater, CommandHandler, MessageHandler, Filters, CallbackContext
)
from telegram.error import Conflict

# ------------- LOGGING -------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("mybot9")

# ------------- ENV -------------
def env_str(name, default=None):
    v = os.getenv(name)
    return v if v is not None else default

def env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

# Telegram / run
TELEGRAM_TOKEN  = env_str("TELEGRAM_TOKEN")
CHAT_ID         = env_str("TELEGRAM_CHAT_ID")
USE_WEBHOOK     = env_int("USE_WEBHOOK", 0) == 1
PUBLIC_URL      = env_str("PUBLIC_URL")  # обязательно для webhook
WEBHOOK_SECRET  = env_str("WEBHOOK_SECRET", "hook")
PORT            = env_int("PORT", 8080)
HEALTH_SERVER   = env_int("HEALTH_SERVER", 1)  # только для polling
TZ_NAME         = env_str("TX", "UTC")

# Trade / strategy
PAIRS           = [s.strip().upper() for s in env_str("PAIRS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
TIMEFRAME       = env_str("TIMEFRAME", "5m")   # поддержка 1m/3m/5m/15m/30m/1h
EMA_LEN         = env_int("EMA_LEN", 100)
EMA_SLOPE_BARS  = env_int("EMA_SLOPE_BARS", 8)
ATR_LEN         = env_int("ATR_LEN", 14)
ATR_MULT_SL     = env_float("ATR_MULT_SL", 1.0)
TP_PCT          = env_float("TP_PCT", 0.35) / 100.0  # 0.35% -> 0.0035
RISK_PCT        = env_float("RISK_PCT", 1.0) / 100.0
LEVERAGE        = env_float("LEVERAGE", 5.0)
FEE_PCT         = env_float("FEE_PCT", 0.0006)  # 0.06%
COOLDOWN_SEC    = env_int("COOLDOWN_SEC", 240)
DEMO_MODE       = env_str("MODE", "DEMO").upper() == "DEMO" or env_str("DEMO_MODE", "true").lower() == "true"
DEMO_START_BAL  = env_float("DEMO_START_BALANCE", 5000.0)
TICK_SEC        = env_int("TICK_SEC", COOLDOWN_SEC)  # как часто работать циклу

# ------------- TIMEZONE -------------
try:
    tz = timezone.utc if TZ_NAME.upper() == "UTC" else timezone(timedelta(0))
except Exception:
    tz = timezone.utc

# ------------- GLOBAL STATE -------------
session = requests.Session()
session.headers.update({"User-Agent": "mybot9/1.0"})
# binance endpoints
BINANCE_KLINES = "https://api.binance.com/api/v3/klines?symbol={sym}&interval={itv}&limit={lim}"

state_lock = threading.Lock()

state = {
    "equity": DEMO_START_BAL,
    "pairs": {},      # symbol -> dict with pos/stats/last_ts/...
    "started": datetime.now(tz=tz),
    "version": "v13"
}
for p in PAIRS:
    state["pairs"][p] = {
        "pos": None,          # dict | None
        "trades": 0,
        "wins": 0,
        "pnl": 0.0,
        "last_signal_ts": 0.0
    }

# ------------- UTILS -------------
def fmt(n, prec=5):
    try:
        return f"{float(n):.{prec}f}"
    except Exception:
        return str(n)

def now_utc():
    return datetime.now(timezone.utc)

def send(msg: str, parse=ParseMode.HTML):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        log.warning("TELEGRAM_TOKEN/CHAT_ID not set; skip send.")
        return
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=parse, disable_web_page_preview=True)
    except Exception as e:
        log.error(f"send(): {e}")

# ------------- INDICATORS -------------
def ema(values, length):
    if not values or len(values) < length:
        return []
    k = 2 / (length + 1)
    out = [sum(values[:length]) / length]
    for v in values[length:]:
        out.append(out[-1] + k * (v - out[-1]))
    # подгон по длине входа
    pad = [None] * (len(values) - len(out))
    return pad + out

def atr(h, l, c, length):
    if len(c) < length + 1:
        return []
    trs = []
    for i in range(1, len(c)):
        tr = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        trs.append(tr)
    # простое RMA
    rma = []
    alpha = 1.0 / length
    rma.append(sum(trs[:length]) / length)
    for v in trs[length:]:
        rma.append((1 - alpha) * rma[-1] + alpha * v)
    # выравниваем
    pad = [None] * (len(c) - len(rma) - 1)
    return pad + rma

def detect_hammer(o, h, l, c):
    """Простой hammer: длинная тень снизу/сверху и маленькое тело."""
    if len(c) < 2:
        return 0
    i = -2  # предыдущая свеча (закрытая)
    body = abs(c[i] - o[i])
    rng = h[i] - l[i]
    if rng == 0:
        return 0
    lower = min(c[i], o[i]) - l[i]
    upper = h[i] - max(c[i], o[i])
    # тело маленькое, одна тень длинная
    if body <= 0.3 * rng:
        if lower >= 0.5 * rng:
            return +1  # bullish hammer
        if upper >= 0.5 * rng:
            return -1  # bearish hammer (inverted)
    return 0

def detect_engulf(o, c):
    """Bull/Bear engulf по двум последним закрытым свечам."""
    if len(c) < 3:
        return 0
    i2, i1 = -3, -2
    # бычье поглощение
    if c[i2] < o[i2] and c[i1] > o[i1] and c[i1] >= o[i2] and o[i1] <= c[i2]:
        return +1
    # медвежье
    if c[i2] > o[i2] and c[i1] < o[i1] and c[i1] <= o[i2] and o[i1] >= c[i2]:
        return -1
    return 0

def detect_breakout(h, l, c, bars=20):
    """Прорыв high/low за N баров (закрытая свеча)."""
    if len(c) < bars + 2:
        return 0
    i = -2
    hi = max(h[-bars-1:-1])
    lo = min(l[-bars-1:-1])
    if c[i] > hi:
        return +1
    if c[i] < lo:
        return -1
    return 0

# ------------- DATA -------------
def binance_interval(tf: str) -> str:
    tf = tf.lower()
    mapping = {"1m":"1m", "3m":"3m", "5m":"5m", "15m":"15m", "30m":"30m", "1h":"1h"}
    return mapping.get(tf, "5m")

def fetch_klines(symbol: str, limit=300):
    url = BINANCE_KLINES.format(sym=symbol, itv=binance_interval(TIMEFRAME), lim=limit)
    r = session.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    o, h, l, c = [], [], [], []
    for k in data:
        o.append(float(k[1])); h.append(float(k[2])); l.append(float(k[3])); c.append(float(k[4]))
    return o, h, l, c

# ------------- TRADING CORE -------------
def compute_signal(o, h, l, c):
    if len(c) < max(EMA_LEN + EMA_SLOPE_BARS, ATR_LEN) + 5:
        return None
    e = ema(c, EMA_LEN)
    slope = None
    if e[-EMA_SLOPE_BARS-1] is not None and e[-1] is not None:
        slope = (e[-1] - e[-EMA_SLOPE_BARS-1]) / EMA_SLOPE_BARS

    hammer = detect_hammer(o, h, l, c)
    engulf = detect_engulf(o, c)
    brk = detect_breakout(h, l, c, bars=20)

    direction = 0
    label = None
    for sig, name in ((hammer, "hammer"), (engulf, "engulf"), (brk, "breakout")):
        if sig != 0:
            direction = sig
            label = name
            break

    if direction == 0 or slope is None:
        return None

    # фильтр трендом EMA: берем сигнал только в сторону наклона EMA
    if direction > 0 and slope <= 0:
        return None
    if direction < 0 and slope >= 0:
        return None

    atr_vals = atr(h, l, c, ATR_LEN)
    if not atr_vals or atr_vals[-1] is None:
        return None
    return {
        "dir": "LONG" if direction > 0 else "SHORT",
        "entry": c[-1],
        "atr": atr_vals[-1],
        "slope": slope,
        "label": label
    }

def open_position(sym: str, sig: dict):
    with state_lock:
        p = state["pairs"][sym]
        if p["pos"] is not None:
            return None  # уже в позиции
        # риск на сделку в USDT
        risk_cash = state["equity"] * RISK_PCT
        # стоп размером ATR*mult → допустимая ден. потеря = risk_cash
        # size в монетах (примерно): risk_cash / (ATR*mult) / price (на 1x), потом умножаем плечо
        qty = max(0.0, (risk_cash / max(sig["atr"] * ATR_MULT_SL, 1e-8)) / sig["entry"]) * LEVERAGE
        if qty <= 0:
            return None

        if sig["dir"] == "LONG":
            sl = sig["entry"] - ATR_MULT_SL * sig["atr"]
            tp = sig["entry"] * (1 + TP_PCT)
        else:
            sl = sig["entry"] + ATR_MULT_SL * sig["atr"]
            tp = sig["entry"] * (1 - TP_PCT)

        p["pos"] = {
            "side": sig["dir"],
            "entry": sig["entry"],
            "qty": qty,
            "sl": sl,
            "tp": tp,
            "atr": sig["atr"],
            "slope": sig["slope"],
            "label": sig["label"],
            "opened": now_utc()
        }
        p["last_signal_ts"] = time.time()

        msg = (
            f"🔴 <b>OPEN {sym} {sig['dir']}</b>\n"
            f"• time: {now_utc():%Y-%m-%d %H:%M:%S} UTC\n"
            f"• entry: {fmt(sig['entry'], 5)}\n"
            f"• qty: {fmt(qty, 6)}\n"
            f"• TP: {fmt(tp,5)}   SL: {fmt(sl,5)}\n"
            f"• signal: {sig['label']}, slope {fmt(sig['slope'],5)}, ATR {fmt(sig['atr'],5)}\n"
            f"• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
        )
    send(msg)

def close_position(sym: str, price: float, reason: str):
    with state_lock:
        p = state["pairs"][sym]
        pos = p["pos"]
        if not pos:
            return
        side = pos["side"]
        entry = pos["entry"]
        qty = pos["qty"]

        if side == "LONG":
            gross = (price - entry) * qty * LEVERAGE
        else:
            gross = (entry - price) * qty * LEVERAGE

        fees = (entry + price) * qty * FEE_PCT
        pnl = gross - fees

        state["equity"] += pnl
        p["pnl"] += pnl
        p["trades"] += 1
        if pnl > 0:
            p["wins"] += 1
        p["pos"] = None

        msg = (
            f"{'✅' if pnl>0 else '❌'} <b>CLOSE {sym} ({reason})</b>\n"
            f"• time: {now_utc():%Y-%m-%d %H:%M:%S} UTC\n"
            f"• exit: {fmt(price,5)}\n"
            f"• PnL: {fmt(pnl,5)}\n"
            f"• pair stats: trades {p['trades']}, WR {fmt((p['wins']/max(p['trades'],1))*100,2)}%, PnL {fmt(p['pnl'],5)}\n"
            f"• total: equity {fmt(state['equity'],2)}  ({fmt(((state['equity']/DEMO_START_BAL)-1)*100,2)}% с начала)"
        )
    send(msg)

def check_manage(sym: str, last_price: float):
    with state_lock:
        pos = state["pairs"][sym]["pos"]
    if not pos:
        return
    if pos["side"] == "LONG":
        if last_price >= pos["tp"]:
            close_position(sym, pos["tp"], "TP")
        elif last_price <= pos["sl"]:
            close_position(sym, pos["sl"], "SL")
    else:
        if last_price <= pos["tp"]:
            close_position(sym, pos["tp"], "TP")
        elif last_price >= pos["sl"]:
            close_position(sym, pos["sl"], "SL")

def loop_symbol(sym: str):
    try:
        o,h,l,c = fetch_klines(sym, limit=max(EMA_LEN+ATR_LEN+30, 120))
    except Exception as e:
        log.warning(f"{sym} klines error: {e}")
        return

    if len(c) < 10:
        return

    last = c[-1]
    # сопровождение
    check_manage(sym, last)

    # входы
    with state_lock:
        last_sig_ts = state["pairs"][sym]["last_signal_ts"]
        in_pos = state["pairs"][sym]["pos"] is not None

    if in_pos or (time.time() - last_sig_ts) < COOLDOWN_SEC:
        return

    sig = compute_signal(o,h,l,c)
    if not sig:
        return
    open_position(sym, sig)

# ------------- TELEGRAM HANDLERS -------------
def cmd_start(update, context: CallbackContext):
    update.message.reply_text("Привет! Команда: /status — покажу состояние.")

def cmd_status(update, context: CallbackContext):
    with state_lock:
        lines = [f"📊 STATUS {now_utc():%Y-%m-%d %H:%M:%S} UTC"]
        total_trades = 0
        total_wr_num = 0
        total_pnl = 0.0
        for sym in PAIRS:
            ps = state["pairs"][sym]
            pos_line = "—"
            if ps["pos"]:
                pp = ps["pos"]
                pos_line = f"{fmt(pp['qty'],6)} pos: {pp['side']} @ {fmt(pp['entry'],5)} (TP {fmt(pp['tp'],5)} / SL {fmt(pp['sl'],5)})"
            lines.append(
                f"{sym} • trades: {ps['trades']}  WR: {fmt((ps['wins']/max(ps['trades'],1))*100,2)}%  PnL: {fmt(ps['pnl'],5)}\n{pos_line}"
            )
            total_trades += ps["trades"]
            total_wr_num += ps["wins"]
            total_pnl += ps["pnl"]
        lines.append("—")
        wr_total = (total_wr_num / max(total_trades,1)) * 100
        lines.append(
            f"TOTAL • trades: {total_trades}  WR: {fmt(wr_total,2)}%  PnL: {fmt(total_pnl,5)}\n"
            f"equity: {fmt(state['equity'],2)}  ({fmt(((state['equity']/DEMO_START_BAL)-1)*100,2)}% с начала)\n"
            f"leverage: {fmt(LEVERAGE,1)}x  fee: {fmt(FEE_PCT*100,3)}%"
        )
    update.message.reply_text("\n".join(lines))

def unknown(update, context):
    update.message.reply_text("Команда не распознана. Доступно: /status")

def on_error(update, context: CallbackContext):
    err = context.error
    if isinstance(err, Conflict):
        log.critical("Conflict 409: другой процесс читает getUpdates. Останавливаю polling.")
        try:
            context.bot.stop()
        except Exception:
            pass
        return
    log.error("Telegram error", exc_info=err)

# ------------- HEALTH (для polling) -------------
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(b"ok\n")

def start_health_server():
    try:
        httpd = HTTPServer(("0.0.0.0", 8080), Health)
        log.info("Health server on :8080")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    except OSError as e:
        log.warning(f"Health server bind skipped: {e}")

# ------------- BOOT -------------
def start_scheduler():
    sch = BackgroundScheduler(timezone="UTC")
    for sym in PAIRS:
        sch.add_job(
            func=lambda s=sym: loop_symbol(s),
            trigger=IntervalTrigger(seconds=TICK_SEC),
            id=f"loop-{sym}",
            replace_existing=True
        )
        log.info(f"Loop started for {sym}")
    sch.start()
    return sch

def start_bot():
    global bot
    if not TELEGRAM_TOKEN:
        log.critical("TELEGRAM_TOKEN не задан")
        raise SystemExit(1)

    # увеличиваем пул, чтобы не было "Connection pool is full"
    updater = Updater(
        token=TELEGRAM_TOKEN,
        use_context=True,
        request_kwargs={"con_pool_size": 8, "read_timeout": 30, "connect_timeout": 30},
    )
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(MessageHandler(Filters.command, unknown))
    dp.add_error_handler(on_error)

    global bot
    bot = updater.bot

    # режим запуска
    if USE_WEBHOOK:
        if not PUBLIC_URL:
            log.critical("Для webhook нужен PUBLIC_URL")
            raise SystemExit(1)
        wh_url = f"{PUBLIC_URL.rstrip('/')}/webhook/{WEBHOOK_SECRET}"
        log.info(f"Setting webhook to {wh_url}")
        updater.start_webhook(
            listen="0.0.0.0", port=PORT, url_path=f"webhook/{WEBHOOK_SECRET}",
            webhook_url=wh_url,
            drop_pending_updates=True,
        )
    else:
        # вырубаем вебхук и чистим хвосты апдейтов, чтобы не было 409
        try:
            updater.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            log.warning(f"delete_webhook: {e}")
        updater.start_polling(drop_pending_updates=True)
        if HEALTH_SERVER:
            start_health_server()

    # приветствие
    mode = "DEMO" if DEMO_MODE else "LIVE"
    pairs_line = ", ".join(PAIRS)
    msg = (
        f"🤖 <b>mybot9 started ({state['version']})</b>\n"
        f"Mode: {mode} | Risk {fmt(RISK_PCT*100,1)}% | TF {TIMEFRAME} | Tick {TICK_SEC}s\n"
        f"Pairs: {pairs_line} | Balance: {int(DEMO_START_BAL)} USDT"
    )
    send(msg)

    return updater

def main():
    # старт
    scheduler = start_scheduler()
    updater = start_bot()
    # держим процесс
    updater.idle()
    scheduler.shutdown(wait=False)

if __name__ == "__main__":
    main()
