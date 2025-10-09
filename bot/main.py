# bot/main.py
import os
import fcntl  # для single-instance lock
import asyncio
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List, Tuple

import httpx
import numpy as np
import pandas as pd
from loguru import logger
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ================== ENV ==================
TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

PAIRS = [p.strip().upper() for p in os.getenv(
    "PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
).split(",") if p.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "1m")         # 1m/5m/15m...
LIMIT = int(os.getenv("KL_LIMIT", "300"))        # свечей для расчётов

# Демо-торговля с фьючерсной моделью
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")
START_BALANCE = float(os.getenv("START_BALANCE", "1000"))
LEVERAGE = min(5.0, float(os.getenv("LEVERAGE", "3")))      # максимум 5х
FEE_PCT = float(os.getenv("FEE_PCT", "0.0006"))             # 0.06% за сторону
RISK_PCT = float(os.getenv("RISK_PCT", "0.01"))             # риск на сделку от эквити, 1%

ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))

EMA_LEN = int(os.getenv("EMA_LEN", "48"))
EMA_SLOPE_BARS = int(os.getenv("EMA_SLOPE_BARS", "5"))

COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "120"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "40"))

DEBUG_TELEMETRY = os.getenv("DEBUG_TELEMETRY", "0").lower() in ("1", "true", "yes")

# ================== STATE ==================
state: Dict[str, Any] = {
    "equity": START_BALANCE,
    "high_water": START_BALANCE,
    "pairs": {
        p: {
            "pos": None,                 # открытая позиция или None
            "last_entry_ts": 0.0,        # время последнего входа (UTC ts)
            "day": None,                 # контроль суточных лимитов
            "trades_today": 0,
            "stats": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        } for p in PAIRS
    }
}

# ================== UTILS ==================
def ensure_single_instance(lock_path: str = "/tmp/mybot9.lock"):
    """Не допускаем одновременный запуск двух экземпляров (409 в Telegram)."""
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd.write(str(os.getpid()))
        fd.flush()
        return fd  # держим открытым, иначе лок снимется GC
    except BlockingIOError:
        logger.error("Another bot instance is already running. Exiting.")
        raise SystemExit(0)

def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fmt(x: float, digits=5) -> str:
    return f"{x:.{digits}f}"

async def tg_send(app: Application, text: str):
    if not TOKEN or not CHAT_ID:
        logger.warning("No TELEGRAM creds")
        return
    try:
        await app.bot.send_message(
            chat_id=CHAT_ID, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"TG send error: {e}")

def start_health_http_server():
    """Поднимаем health-сервер; если порт занят — не валимся, пробуем другой/пропускаем."""
    ports_to_try = [int(os.getenv("PORT", "8080")), 8081]

    class Quiet(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # приглушаем логи
            pass
        def do_GET(self):
            self.send_response(200 if self.path == "/" else 404)
            self.end_headers()
            if self.path == "/":
                self.wfile.write(b"OK")

    for port in ports_to_try:
        try:
            httpd = HTTPServer(("0.0.0.0", port), Quiet)
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(httpd.serve_forever))
            logger.info(f"Health server on :{port}")
            return
        except OSError as e:
            logger.warning(f"Health server bind failed on :{port} ({e}); try next/skip")

# ================== DATA (MEXC) ==================
MEXC_BASE = "https://api.mexc.com"
TF_MAP = {"1m": "1m", "5m": "5m", "15m": "15m"}

async def fetch_klines(pair: str) -> pd.DataFrame:
    """
    Резильентный парсер klines MEXC: иногда возвращают 8 полей, иногда 12.
    Берём первые 6: [openTime, open, high, low, close, volume].
    """
    tf = TF_MAP.get(TIMEFRAME, "1m")
    url = f"{MEXC_BASE}/api/v3/klines"
    params = {"symbol": pair, "interval": tf, "limit": LIMIT}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        raw = r.json()

    rows: List[List[Any]] = []
    for k in raw:
        if isinstance(k, list) and len(k) >= 6:
            rows.append([k[0], k[1], k[2], k[3], k[4], k[5]])

    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vol"])
    # типы
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df.dropna(inplace=True)
    return df

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def patterns(df: pd.DataFrame) -> Dict[str, bool]:
    o, h, l, c = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    o1, c1 = df["open"].iloc[-2], df["close"].iloc[-2]

    # engulfing
    bull_engulf = (c > o) and (c1 < o1) and (c >= o1) and (o <= c1)
    bear_engulf = (c < o) and (c1 > o1) and (c <= o1) and (o >= c1)

    # hammer & shooting-star
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    lower_tail = (min(c, o) - l) / rng
    upper_tail = (h - max(c, o)) / rng
    hammer = (lower_tail >= 0.55) and (body / rng <= 0.2)
    shooting = (upper_tail >= 0.55) and (body / rng <= 0.2)

    # breakout последних 20 свечей
    win = df["close"].iloc[-20:]
    breakout_up = c >= win.max()
    breakout_dn = c <= win.min()

    return {
        "bull_engulf": bool(bull_engulf),
        "bear_engulf": bool(bear_engulf),
        "hammer": bool(hammer),
        "shooting": bool(shooting),
        "breakout_up": bool(breakout_up),
        "breakout_dn": bool(breakout_dn),
    }

def ema_slope(df: pd.DataFrame, length=EMA_LEN, bars=EMA_SLOPE_BARS) -> float:
    e = ema(df["close"], length)
    return float(e.iloc[-1] - e.iloc[-bars])

# ================== TRADING MODEL ==================
def position_size_from_risk(entry: float, sl: float, equity: float) -> float:
    risk_usd = equity * RISK_PCT
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.0
    qty = risk_usd / (sl_dist * LEVERAGE)
    return max(0.0, qty)

def fees_cost(notional: float) -> float:
    # комиссия на вход и выход (taker)
    return notional * FEE_PCT

def simulate_close(side: str, entry: float, exit_: float, qty: float) -> float:
    direction = -1 if side == "SHORT" else 1
    gross = (exit_ - entry) * qty * LEVERAGE * direction
    fee = fees_cost(entry * qty) + fees_cost(exit_ * qty)
    return gross - fee

def decide(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    patt = patterns(df)
    slope = ema_slope(df, EMA_LEN, EMA_SLOPE_BARS)
    atr_val = float(atr(df, ATR_LEN).iloc[-1])
    last = float(df["close"].iloc[-1])

    long_sig = ((patt["bull_engulf"] or patt["hammer"] or patt["breakout_up"]) and slope > 0)
    short_sig = ((patt["bear_engulf"] or patt["shooting"] or patt["breakout_dn"]) and slope < 0)

    return ("LONG" if long_sig else "SHORT" if short_sig else "FLAT"), {
        "patt": patt, "slope": slope, "atr": atr_val, "last": last
    }

# ================== PAIR LOOP ==================
async def pair_loop(app: Application, pair: str):
    logger.info(f"Loop started for {pair}")
    while True:
        try:
            df = await fetch_klines(pair)
            if len(df) < max(EMA_LEN, ATR_LEN, EMA_SLOPE_BARS) + 2:
                await asyncio.sleep(5 if TIMEFRAME == '1m' else 15)
                continue

            sig, info = decide(df)
            pr = state["pairs"][pair]

            # daily counters
            now = datetime.now(timezone.utc)
            if pr["day"] != now.date():
                pr["day"] = now.date()
                pr["trades_today"] = 0

            last = info["last"]
            atr_val = max(1e-9, info["atr"])

            # ---- manage open position
            if pr["pos"]:
                pos = pr["pos"]
                side = pos["side"]; entry = pos["entry"]
                tp = pos["tp"]; sl = pos["sl"]; qty = pos["qty"]

                hit_tp = (last >= tp) if side == "LONG" else (last <= tp)
                hit_sl = (last <= sl) if side == "LONG" else (last >= sl)

                if hit_tp or hit_sl:
                    pnl = simulate_close(side, entry, last, qty)
                    pr["pos"] = None

                    # stats
                    st = pr["stats"]
                    st["trades"] += 1
                    st["pnl"] += pnl
                    if pnl >= 0:
                        st["wins"] += 1
                    else:
                        st["losses"] += 1

                    state["equity"] += pnl
                    state["high_water"] = max(state["high_water"], state["equity"])

                    wr_pair = 100.0 * st["wins"] / st["trades"] if st["trades"] else 0.0
                    tot_trades = sum(state["pairs"][p]["stats"]["trades"] for p in PAIRS)
                    tot_wins = sum(state["pairs"][p]["stats"]["wins"] for p in PAIRS)
                    tot_pnl = sum(state["pairs"][p]["stats"]["pnl"] for p in PAIRS)
                    wr_tot = 100.0 * tot_wins / tot_trades if tot_trades else 0.0

                    badge = "✅" if pnl >= 0 else "❌"
                    txt = (
                        f"{badge} <b>CLOSE {pair}</b> ({'TP' if hit_tp else 'SL'})"
                        f"\n• time: {utcnow()}"
                        f"\n• exit: {fmt(last,5)}"
                        f"\n• PnL: {'+' if pnl>=0 else ''}{fmt(pnl,5)}"
                        f"\n• pair stats: trades {st['trades']}, WR {fmt(wr_pair,2)}%, PnL {fmt(st['pnl'],5)}"
                        f"\n• total: trades {tot_trades}, WR {fmt(wr_tot,2)}%, PnL {fmt(tot_pnl,5)}"
                        f"\n• equity: {fmt(state['equity'],5)} (lev {LEVERAGE}×, fee {FEE_PCT*100:.3f}%)"
                    )
                    await tg_send(app, txt)

            # ---- open new position
            can_open = (sig != "FLAT") and (pr["pos"] is None)
            if can_open:
                cooldown_ok = (datetime.now(timezone.utc).timestamp() - pr["last_entry_ts"]) >= COOLDOWN_SEC
                limit_ok = pr["trades_today"] < MAX_TRADES_PER_DAY

                if cooldown_ok and limit_ok:
                    side = "LONG" if sig == "LONG" else "SHORT"
                    entry = last
                    sl = entry - ATR_MULT_SL * atr_val if side == "LONG" else entry + ATR_MULT_SL * atr_val
                    tp = entry + ATR_MULT_TP * atr_val if side == "LONG" else entry - ATR_MULT_TP * atr_val
                    qty = position_size_from_risk(entry, sl, state["equity"])

                    if qty > 0:
                        pr["pos"] = {
                            "side": side, "entry": entry, "sl": sl, "tp": tp,
                            "qty": qty, "ts": datetime.now(timezone.utc).timestamp()
                        }
                        pr["trades_today"] += 1
                        pr["last_entry_ts"] = datetime.now(timezone.utc).timestamp()

                        patt_name = ", ".join([k for k, v in info["patt"].items() if v]) or "—"
                        txt = (
                            f"🔴 <b>OPEN {pair} {side}</b>"
                            f"\n• time: {utcnow()}"
                            f"\n• entry: {fmt(entry,5)}"
                            f"\n• qty: {fmt(qty,6)}"
                            f"\n• TP: {fmt(tp,5)}   SL: {fmt(sl,5)}"
                            f"\n• signal: {patt_name}, slope {fmt(info['slope'],5)}, ATR {fmt(atr_val,5)}"
                            f"\n• mode: {'DEMO' if DEMO_MODE else 'LIVE'}"
                        )
                        await tg_send(app, txt)

            # отладка по запросу
            if DEBUG_TELEMETRY:
                pr_open = pr["pos"]
                patt_name = ", ".join([k for k, v in patterns(df).items() if v]) or "—"
                await tg_send(app,
                    f"🧪 <b>DEBUG {pair}</b>"
                    f"\nlast: {fmt(df['close'].iloc[-1],5)}  ema48_slope({EMA_SLOPE_BARS}): {fmt(ema_slope(df),5)}"
                    f"\nATR: {fmt(atr(df, ATR_LEN).iloc[-1],5)}  patt: {patt_name}"
                    f"\npos: {'—' if not pr_open else pr_open}"
                )

        except Exception as e:
            logger.exception(f"Loop error {pair}: {e}")

        await asyncio.sleep(5 if TIMEFRAME == '1m' else 15)

# ================== COMMANDS ==================
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tot_tr = tot_w = 0
    tot_pnl = 0.0
    lines = [f"📊 <b>STATUS {utcnow()}</b>"]
    for p in PAIRS:
        st = state["pairs"][p]["stats"]
        tr, w, pnl = st["trades"], st["wins"], st["pnl"]
        wr = 100.0 * w / tr if tr else 0.0
        pos = state["pairs"][p]["pos"]
        pos_str = "—" if not pos else f"{pos['side']} @ {fmt(pos['entry'],5)} (TP {fmt(pos['tp'],5)} / SL {fmt(pos['sl'],5)})"
        lines.append(f"{p}• trades: {tr}  WR: {fmt(wr,2)}%  PnL: {fmt(pnl,5)}\npos: {pos_str}")
        tot_tr += tr; tot_w += w; tot_pnl += pnl
    wr_tot = 100.0 * tot_w / tot_tr if tot_tr else 0.0
    lines.append("—")
    lines.append(f"TOTAL • trades: {tot_tr}  WR: {fmt(wr_tot,2)}%  PnL: {fmt(tot_pnl,5)}")
    lines.append(f"equity: {fmt(state['equity'],5)}  leverage: {LEVERAGE}×  fee: {FEE_PCT*100:.3f}%")
    await update.effective_chat.send_message("\n".join(lines), parse_mode=ParseMode.HTML)

# ================== APP ==================
def build_app() -> Application:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("status", cmd_status))

    async def post_init(a: Application):
        # Сбрасываем вебхук, чтобы не конфликтовать с polling
        try:
            await a.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            logger.warning(f"delete_webhook warning: {e}")

        # Запуск фоновых лупов по всем парам
        for p in PAIRS:
            asyncio.create_task(pair_loop(app, p))
        logger.info("Background loops started")

    app.post_init = post_init  # type: ignore
    return app

def main():
    logger.info("🤖 mybot9 started successfully!")

    # Health server (не валим процесс, если порт занят)
    try:
        start_health_http_server()
    except Exception as e:
        logger.warning(f"Health server init skipped: {e}")

    # Единственный экземпляр
    _lock_fd = ensure_single_instance()  # держим ссылку, чтобы лок не снялся

    app = build_app()
    app.run_polling()

if __name__ == "__main__":
    main()
