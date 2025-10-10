# -*- coding: utf-8 -*-
import os
import json
import time
import threading
import logging
import socketserver
import http.server
from datetime import datetime, timezone

import pytz
import requests

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from telegram import Bot, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram.utils.request import Request

# ---------------------------------------
# ЛОГГИРОВАНИЕ
# ---------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------
# ENV
# ---------------------------------------
ENV = os.getenv

TELEGRAM_TOKEN = ENV("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = ENV("TELEGRAM_CHAT_ID", "").strip()

USE_WEBHOOK = ENV("USE_WEBHOOK", "0").strip() == "1"
PUBLIC_URL = ENV("PUBLIC_URL", "").strip()
WEBHOOK_SECRET = ENV("WEBHOOK_SECRET", "hook").strip()

PORT = int(ENV("PORT", "8080"))

PAIRS = [s.strip() for s in ENV("PAIRS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
TIMEFRAME = ENV("TIMEFRAME", "5m").strip()
RISK_PCT = float(ENV("RISK_PCT", "1"))
TP_PCT = float(ENV("TP_PCT", "0.35"))
ATR_LEN = int(ENV("ATR_LEN", "14"))
ATR_MULT_SL = float(ENV("ATR_MULT_SL", "1.0"))
EMA_LEN = int(ENV("EMA_LEN", "100"))
EMA_SLOPE_BARS = int(ENV("EMA_SLOPE_BARS", "8"))
COOLDOWN_SEC = int(ENV("COOLDOWN_SEC", "240"))
FEE_PCT = float(ENV("FEE_PCT", "0.0006"))
LEVERAGE = float(ENV("LEVERAGE", "5"))
DEMO_MODE = ENV("DEMO_MODE", "true").lower() == "true"
DRY_RUN = ENV("DRY_RUN", "false").lower() == "true"
DAILY_SUMMARY = ENV("DAILY_SUMMARY", "1") == "1"

TX_NAME = ENV("TX", "UTC")
try:
    TZ = pytz.timezone(TX_NAME)
except Exception:
    logger.warning("Bad TX timezone '%s', fallback to UTC", TX_NAME)
    TZ = pytz.UTC

# ---------------------------------------
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ (упрощённо)
# ---------------------------------------
state = {
    "balances": {"demo": float(ENV("DEMO_START_BALANCE", "5000"))},
    "pnl": 0.0,
    "wins": 0,
    "losses": 0,
    "trades": 0,
    "last_trade_ts": None,
    "pair_stats": {p: {"trades": 0, "wins": 0, "losses": 0} for p in PAIRS},
}

bot: Bot = None
updater: Updater = None
scheduler: BackgroundScheduler = None

# ---------------------------------------
# HEALTH-СЕРВЕР (ТОЛЬКО для POLLING)
# ---------------------------------------
class Quiet(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # тише в логах
        return
    def do_GET(self):
        if self.path.startswith("/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"OK\n")

def start_health_server():
    def _serve():
        try:
            with socketserver.TCPServer(("0.0.0.0", PORT), Quiet) as httpd:
                logger.info("Health server started on :%s", PORT)
                httpd.serve_forever()
        except OSError as e:
            logger.warning("Health server bind skipped: %s", e)
    t = threading.Thread(target=_serve, name="health", daemon=True)
    t.start()

# ---------------------------------------
# ОСНОВНОЙ ЛУП ПО ПАРЕ (заглушка)
# ---------------------------------------
def loop_pair(pair: str):
    try:
        logger.info("Tick %s @ %s", pair, datetime.now(timezone.utc).strftime("%H:%M:%S"))
        # TODO: здесь реальная торговая логика (сигналы/TP/SL/DEMO-LIVE)
    except Exception as e:
        logger.exception("Loop error %s: %s", pair, e)

# ---------------------------------------
# APSCHEDULER
# ---------------------------------------
def start_scheduler():
    global scheduler
    if scheduler:
        return scheduler

    scheduler = BackgroundScheduler(timezone=TZ)
    for p in PAIRS:
        scheduler.add_job(
            loop_pair,
            trigger=IntervalTrigger(seconds=COOLDOWN_SEC, timezone=TZ),
            args=[p],
            id=f"loop-{p}",
            replace_existing=True,
        )
    scheduler.start()
    logger.info("Scheduler started")
    return scheduler

# ---------------------------------------
# TELEGRAM
# ---------------------------------------
def build_updater() -> Updater:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не задан")

    # В PTB 13.15 нельзя передавать request=... в Updater.
    # Нужно создать Request -> Bot(..., request=req) -> Updater(bot=bot)
    req = Request(con_pool_size=8, connect_timeout=30, read_timeout=30)
    tg_bot = Bot(token=TELEGRAM_TOKEN, request=req)
    _updater = Updater(bot=tg_bot, use_context=True)
    return _updater

# ---------------------------------------
# HANDLERS
# ---------------------------------------
def cmd_start(update, context):
    update.message.reply_text(
        "Привет! Бот запущен.\n"
        "Команды:\n"
        "  /status — статус и параметры\n"
        "  /config — текущие ENV параметры\n"
    )

def fmt_money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")

def cmd_status(update, context):
    now_utc = datetime.now(timezone.utc)
    lines = [
        f"📊 STATUS {now_utc:%Y-%m-%d %H:%M:%S} UTC",
        f"Mode: {'DEMO' if DEMO_MODE else 'LIVE'}  |  DRY_RUN: {DRY_RUN}",
        f"Pairs: {', '.join(PAIRS)}  |  TF: {TIMEFRAME}  |  Interval: {COOLDOWN_SEC}s",
        f"Risk: {RISK_PCT:.2f}%  |  TP: {TP_PCT:.3f}%  |  ATR: len={ATR_LEN} x{ATR_MULT_SL}",
        f"EMA: len={EMA_LEN} slopeBars={EMA_SLOPE_BARS}  | Fee: {FEE_PCT*100:.3f}%  | Lev: {LEVERAGE}x",
        "",
        f"Баланс (demo): {fmt_money(state['balances']['demo'])}",
        f"PnL: {fmt_money(state['pnl'])}  |  Trades: {state['trades']}  |  W/L: {state['wins']}/{state['losses']}",
        "",
    ]
    for p in PAIRS:
        ps = state["pair_stats"][p]
        lines.append(f"{p}: trades={ps['trades']}  w/l={ps['wins']}/{ps['losses']}")
    update.message.reply_text("\n".join(lines))

def cmd_config(update, context):
    cfg = {
        "USE_WEBHOOK": USE_WEBHOOK,
        "PUBLIC_URL": PUBLIC_URL,
        "WEBHOOK_SECRET": WEBHOOK_SECRET,
        "PORT": PORT,
        "PAIRS": PAIRS,
        "TIMEFRAME": TIMEFRAME,
        "COOLDOWN_SEC": COOLDOWN_SEC,
        "RISK_PCT": RISK_PCT,
        "TP_PCT": TP_PCT,
        "ATR_LEN": ATR_LEN,
        "ATR_MULT_SL": ATR_MULT_SL,
        "EMA_LEN": EMA_LEN,
        "EMA_SLOPE_BARS": EMA_SLOPE_BARS,
        "FEE_PCT": FEE_PCT,
        "LEVERAGE": LEVERAGE,
        "DEMO_MODE": DEMO_MODE,
        "DRY_RUN": DRY_RUN,
        "DAILY_SUMMARY": DAILY_SUMMARY,
        "TX": TX_NAME,
    }
    update.message.reply_text(
        "<b>CONFIG</b>\n<pre>" + json.dumps(cfg, indent=2, ensure_ascii=False) + "</pre>",
        parse_mode=ParseMode.HTML,
    )

def unknown(update, context):
    update.message.reply_text("Не знаю такую команду. Есть /status и /config.")

def init_handlers(_updater: Updater):
    dp = _updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("config", cmd_config))
    dp.add_handler(MessageHandler(Filters.command, unknown))

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    global bot, updater

    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN обязателен")

    # Планировщик один раз
    start_scheduler()

    # Telegram
    updater = build_updater()
    bot = updater.bot
    init_handlers(updater)

    if USE_WEBHOOK:
        if not PUBLIC_URL:
            raise RuntimeError("PUBLIC_URL обязателен при USE_WEBHOOK=1")

        path = f"webhook/{WEBHOOK_SECRET}"
        webhook_url = f"{PUBLIC_URL.rstrip('/')}/{path}"
        logger.info("Webhook mode on port %s", PORT)
        logger.info("Setting webhook to %s", webhook_url)

        # сбросить возможный старый webhook и поставить новый
        bot.delete_webhook()
        bot.set_webhook(webhook_url)

        updater.start_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=path,
            drop_pending_updates=True,
        )
        logger.info("Webhook server started")
        updater.idle()
    else:
        # Polling + health endpoint (для Koyeb)
        start_health_server()
        logger.info("Starting polling…")
        updater.start_polling(drop_pending_updates=True)
        updater.idle()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal: %s", e)
        raise
