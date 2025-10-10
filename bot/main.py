# /bot/main.py
import os
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from telegram import Bot, ParseMode, Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from time import sleep, time

# ---------- logging ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------- env ----------
TOKEN       = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID     = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
PUBLIC_URL  = os.getenv("PUBLIC_URL", "").rstrip("/")  # https://<your-koyeb-app>.koyeb.app
WEBHOOK_SEC = os.getenv("WEBHOOK_SECRET", "hook")      # любой секретный хвост
PORT        = int(os.getenv("PORT", "8080"))
DROP_UPD    = os.getenv("DROP_PENDING", "1") == "1"

if not TOKEN or not PUBLIC_URL:
    raise SystemExit("TELEGRAM_TOKEN и PUBLIC_URL обязательны")

# ---------- простейший /status и заглушки торговли ----------
state = {
    "equity": 5000.0,
    "pairs": ["BTCUSDT", "ETHUSDT"],
    "lev": 5.0,
    "fee": 0.0006,
    "trades": 0,
    "pnl": 0.0,
    "positions": {},
    "started": int(time()),
}

def fmt_money(x): return f"{x:.5f}".rstrip("0").rstrip("."

                                                    ) if isinstance(x, float) else str(x)

def cmd_start(update: Update, _: CallbackContext):
    update.message.reply_text("Бот запущен. Используй /status")

def cmd_status(update: Update, _: CallbackContext):
    lines = ["📊 STATUS"]
    for p in state["pairs"]:
        pos = state["positions"].get(p, "—")
        lines.append(f"{p} • trades: 0  WR: 0%  PnL: 0\n{pos}")
        lines.append("—")
    lines.append(
        f"TOTAL • trades: {state['trades']}  WR: 0%  PnL: {state['pnl']:.5f}\n"
        f"equity: {fmt_money(state['equity'])}  (0.0% с начала)\n"
        f"leverage: {state['lev']:.1f}x  fee: {state['fee']*100:.03f}%"
    )
    update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

# ---------- health endpoint (по умолчанию выключен) ----------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        elif self.path == f"/webhook/{WEBHOOK_SEC}":
            # Telegram стучится сюда POST'ом. GET — просто 200
            self.send_response(200); self.end_headers(); self.wfile.write(b"tg")
        else:
            self.send_response(404); self.end_headers()

def run_health_server():
    if os.getenv("HEALTH_SERVER", "0") == "0":
        log.info("Health server disabled"); return
    try:
        httpd = HTTPServer(("0.0.0.0", PORT), HealthHandler)
        log.info("Health server on :%d", PORT)
        httpd.serve_forever()
    except OSError as e:
        log.warning("Health server bind skipped: %s", e)

# ---------- main ----------
def main():
    bot = Bot(TOKEN)
    updater = Updater(bot=bot, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))

    # Запускаем вебхук (без polling — значит, конфликтов не будет)
    path = f"/webhook/{WEBHOOK_SEC}"
    webhook_url = f"{PUBLIC_URL}{path}"
    log.info("Setting webhook to %s", webhook_url)

    # Сначала снимаем старый хук (на всякий случай)
    try:
        bot.delete_webhook(drop_pending_updates=DROP_UPD)
    except Exception as e:
        log.warning("delete_webhook: %s", e)

    updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=path,              # путь, который слушает сервер
        webhook_url=webhook_url,    # публичный URL, сообщаем Telegram
        clean=DROP_UPD,             # v13: удалить «хвост» апдейтов при старте
    )
    log.info("Webhook started")

    # опциональный health-сервер
    Thread(target=run_health_server, daemon=True).start()

    # эмуляция рабочих лупов торговли
    def trading_loop(pair):
        log.info("Loop started for %s", pair)
        while True:
            sleep(5)

    for p in state["pairs"]:
        Thread(target=trading_loop, args=(p,), daemon=True).start()

    updater.idle()

if __name__ == "__main__":
    main()
