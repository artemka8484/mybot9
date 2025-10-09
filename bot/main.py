import os
import asyncio
from loguru import logger
from datetime import datetime
from typing import Optional

# Telegram
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# === ENV ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # можно задать в Koyeb
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# Заглушка параметров API биржи (на будущее)
API_KEY = os.getenv("MEXC_API_KEY", "")
API_SECRET = os.getenv("MEXC_API_SECRET", "")

# Глобально держим ссылку на Telegram app
tg_app: Optional[Application] = None


async def tg_send(text: str) -> None:
    """Отправка сообщения в Telegram (если задан CHAT_ID)."""
    if not tg_app:
        logger.warning("Telegram app is not ready yet")
        return
    chat_id = TELEGRAM_CHAT_ID
    if not chat_id:
        logger.warning("TELEGRAM_CHAT_ID не задан. Сообщение не отправлено.")
        return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception(f"Failed to send Telegram message: {e}")


# ===== Handlers =====
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Если TELEGRAM_CHAT_ID не задан — подсказываем как его поставить."""
    user_chat_id = update.effective_chat.id if update.effective_chat else None
    msg = [
        "🤖 *mybot9* готов.",
        f"DRY_RUN: *{DRY_RUN}*",
    ]
    if TELEGRAM_CHAT_ID:
        msg.append(f"Отчёты шлём в: `{TELEGRAM_CHAT_ID}`")
    else:
        msg.append(
            "Похоже, переменная `TELEGRAM_CHAT_ID` не задана.\n"
            f"Твой chat_id: `{user_chat_id}` — добавь его в Koyeb → Settings → Environment variables."
        )
    await update.message.reply_text("\n".join(msg), parse_mode=ParseMode.MARKDOWN)


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🏓 pong")


# ===== Простая «демо-торговля» =====
async def demo_strategy_loop():
    """
    Заглушка «Стратегии #9»: раз в N секунд генерируем фиктивные сделки.
    Здесь позже подключим реальную логику сигналов и исполнение ордеров.
    """
    trade_id = 1
    while True:
        # TODO: сюда вставим реальную стратегию #9 (получение цены, сигналов и т.д.)
        await asyncio.sleep(12)  # Период «проверки рынка»

        symbol = "BTC/USDT"
        side = "BUY" if trade_id % 2 else "SELL"
        qty = 0.001
        price = 60000 + trade_id * 10  # для демо: «цена»
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        if DRY_RUN:
            # Имитация сделки
            logger.info(f"[DEMO] Executed {side} {qty} {symbol} @ {price}")
            await tg_send(
                f"🧪 *DEMO trade* #{trade_id}\n"
                f"• *{side}* `{qty}` `{symbol}` @ `{price}`\n"
                f"• Время: `{ts}`\n"
                f"• Стратегия: *#9*\n"
                f"• Исполнение: *без реального ордера*"
            )
        else:
            # Здесь будет реальный вызов биржи (например MEXC) и последующий отчёт
            pass

        trade_id += 1


async def run_bot():
    logger.info("🤖 mybot9 started successfully!")

    # Инициализируем Telegram приложение (polling)
    global tg_app
    if TELEGRAM_TOKEN:
        tg_app = Application.builder().token(TELEGRAM_TOKEN).build()
        tg_app.add_handler(CommandHandler("start", cmd_start))
        tg_app.add_handler(CommandHandler("ping", cmd_ping))

        # Запускаем polling без блокировки основного цикла:
        await tg_app.initialize()
        await tg_app.start()
        logger.info("Telegram polling started")
    else:
        logger.warning("TELEGRAM_TOKEN не задан — Telegram-бот отключён.")

    # Параллельно запускаем луп стратегии
    strategy_task = asyncio.create_task(demo_strategy_loop())

    # Держим процесс
    try:
        while True:
            await asyncio.sleep(5)
            logger.info("Bot is alive... waiting for signals")
    finally:
        strategy_task.cancel()
        if tg_app:
            await tg_app.stop()
            await tg_app.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.warning("Bot stopped manually.")
