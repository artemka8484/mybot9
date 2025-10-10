# -*- coding: utf-8 -*-
import os, json, time, math, threading, socketserver, http.server
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
import requests
import logging
from collections import deque

from telegram import ParseMode, Bot
from telegram.ext import Updater, CommandHandler

# --------------------------- logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mybot9")

# --------------------------- env -------------------------------------
def envf(name, default=None, cast=str):
    v = os.getenv(name, default)
    if v is None: return None
    try:
        return cast(v) if cast else v
    except Exception:
        return v

TELEGRAM_TOKEN = envf("TELEGRAM_TOKEN")
CHAT_ID        = envf("TELEGRAM_CHAT_ID")
assert TELEGRAM_TOKEN and CHAT_ID, "TELEGRAM_TOKEN и TELEGRAM_CHAT_ID обязательны"

MODE        = envf("MODE", "DEMO")
DEMO        = str(envf("DEMO_MODE", "true")).lower() == "true"
START_BAL   = float(envf("DEMO_START_BALANCE", 5000, float))
DRY_RUN     = str(envf("DRY_RUN", "false")).lower() == "true"

PAIRS = [p.strip().upper() for p in envf("PAIRS", "BTCUSDT,ETHUSDT").split(",") if p.strip()]
TF    = envf("TIMEFRAME", "5m")
RISK  = float(envf("RISK_PCT", 1.0, float)) / 100.0
LEV   = float(envf("LEVERAGE", 5, float))
FEE   = float(envf("FEE_PCT", 0.0006, float))
TP_P  = float(envf("TP_PCT", 0.35, float)) / 100.0

ATR_LEN  = int(envf("ATR_LEN", 14, int))
ATR_K    = float(envf("ATR_MULT_SL", 1.0, float))
EMA_LEN  = int(envf("EMA_LEN", 100, int))
SLOPE_N  = int(envf("EMA_SLOPE_BARS", 8, int))
COOLDOWN = int(envf("COOLDOWN_SEC", 240, int))

USE_WEBHOOK = str(envf("USE_WEBHOOK", "0")).lower() in ("1","true","yes")

MEXC_URL = envf("MEXC_BASE_URL", "https://contract.mexc.com")

# --------------------------- time ------------------------------------
TZ = timezone.utc
def now_utc(): return datetime.now(TZ)

# --------------------------- state & persistence ----------------------
STATE_FILE = "state.json"
state_lock = threading.Lock()

balance = START_BAL
positions: Dict[str, Dict] = {}       # pair -> dict(entry, qty, tp, sl, side, opened_at, reason)
stats = {
    "pairs": {p: {"trades": 0, "wins": 0, "pnl": 0.0} for p in PAIRS},
    "total": {"trades": 0, "wins": 0, "pnl": 0.0}
}

def load_state():
    global balance, positions, stats
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        balance = float(data.get("balance", START_BAL))
        positions.update(data.get("positions", {}))
        saved = data.get("stats", {})
        if saved:
            stats["pairs"].update(saved.get("pairs", {}))
            stats["total"].update(saved.get("total", {}))
        log.info(f"State loaded: balance={balance:.2f}, open={list(positions.keys())}")
    except FileNotFoundError:
        log.info("No previous state file found.")
    except Exception as e:
        log.warning(f"Failed to load state: {e}")

def save_state():
    try:
        payload = {"balance": balance, "positions": positions, "stats": stats}
        with open(STATE_FILE, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        log.warning(f"Failed to save state: {e}")

# --------------------------- health HTTP (koyeb) ----------------------
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs): pass

def serve_health():
    PORT = 8080
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), Quiet) as httpd:
            httpd.serve_forever()
    except OSError as e:
        log.warning(f"Health server bind skipped: {e}")

threading.Thread(target=serve_health, daemon=True).start()

# --------------------------- market data ------------------------------
session = requests.Session()
session.headers.update({"User-Agent": "mybot9/1.0"})

INTERVAL_MAP = {
    "1m":"Min1", "3m":"Min3", "5m":"Min5", "15m":"Min15", "30m":"Min30",
    "1h":"Hour1", "4h":"Hour4", "1d":"Day1"
}
def kline_interval(tf: str) -> str:
    return INTERVAL_MAP.get(tf, "Min5")

def fetch_klines(pair: str, limit=EMA_LEN+ATR_LEN+SLOPE_N+5) -> List[Tuple[float,float,float,float]]:
    """
    returns list of tuples (ts, open, high, low, close)
    """
    url = f"{MEXC_URL}/api/v1/contract/kline/{pair}"
    params = {"interval": kline_interval(TF), "limit": limit}
    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    kl = []
    for it in data.get("data", []):
        ts = int(it["t"]) / 1000
        o,h,l,c = map(float, (it["o"], it["h"], it["l"], it["c"]))
        kl.append((ts,o,h,l,c))
    if not kl:
        raise RuntimeError(f"no klines for {pair}")
    return kl

# --------------------------- indicators --------------------------------
def ema(values: List[float], length: int) -> float:
    k = 2/(length+1)
    e = values[0]
    for v in values[1:]:
        e = v*k + e*(1-k)
    return e

def atr(ohlc: List[Tuple[float,float,float,float,float]], length: int) -> float:
    trs = []
    prev_close = ohlc[0][4]
    for (_,o,h,l,c) in ohlc[1:]:
        tr = max(h-l, abs(h-prev_close), abs(l-prev_close))
        trs.append(tr)
        prev_close = c
    if len(trs) < length: length = len(trs)
    return sum(trs[-length:])/length if length>0 else 0.0

def ema_slope(closes: List[float], length:int, bars:int) -> float:
    if len(closes) < length+bars: return 0.0
    e2 = ema(closes[-length:], length)
    e1 = ema(closes[-length-bars:-bars], length)
    return (e2 - e1) / max(1e-9, e1)

def detect_signal(ohlc) -> Tuple[str, str]:
    """
    простая логика: молот/перевёрнутый молот + направление EMA-наклона
    возвращает (side, reason) — side in {"LONG","SHORT",None}
    """
    *_, (t,o,h,l,c) = ohlc
    body = abs(c - o)
    upper = h - max(o,c)
    lower = min(o,c) - l
    is_hammer = lower > body*2 and upper < body
    is_shoot  = upper > body*2 and lower < body

    closes = [x[4] for x in ohlc]
    slope = ema_slope(closes, EMA_LEN, SLOPE_N)

    if is_hammer and slope >= 0:
        return "LONG", f"hammer, slope {slope:.5f}"
    if is_shoot and slope <= 0:
        return "SHORT", f"hammer, slope {slope:.5f}"  # «hammer» для коротких — перевёрнутый
    return None, f"slope {slope:.5f}"

# --------------------------- trading engine ----------------------------
last_trade_time: Dict[str, float] = {p: 0.0 for p in PAIRS}

def position_size(balance_usdt: float, price: float) -> float:
    risk_capital = balance_usdt * RISK
    notional = risk_capital * LEV
    qty = max(1e-8, notional / price)
    return qty

def open_position(pair: str, side: str, price: float, atr_val: float, reason: str):
    global balance
    if positions.get(pair):
        return

    # cooldown
    if time.time() - last_trade_time.get(pair, 0) < COOLDOWN:
        return
    last_trade_time[pair] = time.time()

    qty = position_size(balance, price)
    tp = price * (1 + TP_P) if side == "LONG" else price * (1 - TP_P)
    sl_off = ATR_K * atr_val
    sl = price - sl_off if side == "LONG" else price + sl_off

    positions[pair] = {
        "side": side, "entry": price, "qty": qty,
        "tp": tp, "sl": sl, "opened_at": now_utc().isoformat(),
        "reason": reason
    }
    save_state()
    send_open(pair, side, price, qty, tp, sl, reason)

def check_positions(pair: str, last_price: float):
    global balance, stats
    pos = positions.get(pair)
    if not pos: return
    side = pos["side"]
    entry, qty, tp, sl = pos["entry"], pos["qty"], pos["tp"], pos["sl"]

    hit_tp = last_price >= tp if side == "LONG" else last_price <= tp
    hit_sl = last_price <= sl if side == "LONG" else last_price >= sl
    if not (hit_tp or hit_sl):
        return

    # close
    exit_price = tp if hit_tp else sl
    pnl_notional = (exit_price - entry) * qty * (1 if side=="LONG" else -1)
    fee_cost = (entry + exit_price) * qty * FEE
    pnl = pnl_notional - fee_cost
    balance += pnl

    # stats
    s = stats["pairs"][pair]
    s["trades"] += 1
    s["pnl"] += pnl
    if pnl > 0: s["wins"] += 1
    stats["total"]["trades"] += 1
    stats["total"]["pnl"] += pnl
    if pnl > 0: stats["total"]["wins"] += 1

    # message
    send_close(pair, exit_price, pnl, hit_tp)

    # remove
    del positions[pair]
    save_state()

# --------------------------- telegram ---------------------------------
bot = Bot(token=TELEGRAM_TOKEN)
updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
dp = updater.dispatcher

def human_pair_stat(pair: str) -> str:
    s = stats["pairs"][pair]
    wr = (s["wins"]/s["trades"]*100) if s["trades"] else 0.0
    pnl = s["pnl"]
    pos = positions.get(pair)
    pos_line = "—"
    if pos:
        pos_line = f"{pos['qty']:.6f}  pos: <b>{pos['side']}</b> @ {pos['entry']:.5f} (TP {pos['tp']:.5f} / SL {pos['sl']:.5f})"
    return (f"{pair} • trades: {s['trades']}  WR: {wr:.2f}%  PnL: {pnl:.5f}\n"
            f"{pos_line}")

def cmd_start(update, ctx):
    msg = (f"🤖 mybot9 started (v13)\n"
           f"Mode: {MODE} | Risk {RISK*100:.1f}% | TF {TF} | Tick {COOLDOWN}s\n"
           f"Pairs: {', '.join(PAIRS)} | Balance: {balance:.2f} USDT")
    ctx.bot.send_message(CHAT_ID, msg)

def cmd_status(update, ctx):
    lines = [f"📊 STATUS {now_utc():%Y-%m-%d %H:%M:%S} UTC"]
    for p in PAIRS:
        lines.append(human_pair_stat(p))
        lines.append("—")
    tot = stats["total"]
    wr = (tot["wins"]/tot["trades"]*100) if tot["trades"] else 0.0
    lines.append(f"TOTAL • trades: {tot['trades']}  WR: {wr:.2f}%  PnL: {tot['pnl']:.5f}")
    lines.append(f"equity: {balance:.2f}")
    lines.append(f"leverage: {LEV:.1f}x  fee: {FEE*100:.3f}%")
    ctx.bot.send_message(CHAT_ID, "\n".join(lines), parse_mode=ParseMode.HTML)

dp.add_handler(CommandHandler("start", cmd_start))
dp.add_handler(CommandHandler("status", cmd_status))

def send_open(pair, side, entry, qty, tp, sl, reason):
    text = (f"🔴 OPEN {pair} {side}\n"
            f"• time: {now_utc():%Y-%m-%d %H:%M:%S} UTC\n"
            f"• entry: {entry:.5f}\n"
            f"• qty: {qty:.6f}\n"
            f"• TP: {tp:.5f}   SL: {sl:.5f}\n"
            f"• signal: {reason}\n"
            f"• mode: {MODE}")
    bot.send_message(chat_id=CHAT_ID, text=text)

def send_close(pair, exit_price, pnl, is_tp):
    s = stats["pairs"][pair]
    tot = stats["total"]
    text = (f"{'✅' if is_tp else '❌'} CLOSE {pair} ({'TP' if is_tp else 'SL'})\n"
            f"• time: {now_utc():%Y-%m-%d %H:%M:%S} UTC\n"
            f"• exit: {exit_price:.5f}\n"
            f"• PnL: {pnl:+.5f}\n"
            f"• pair stats: trades {s['trades']}, WR {(s['wins']/s['trades']*100 if s['trades'] else 0):.2f}%, PnL {s['pnl']:+.5f}\n"
            f"• total: trades {tot['trades']}, WR {(tot['wins']/tot['trades']*100 if tot['trades'] else 0):.2f}%, PnL {tot['pnl']:+.5f}\n"
            f"• balance: {balance:.5f}")
    bot.send_message(chat_id=CHAT_ID, text=text)

# --------------------------- main loop --------------------------------
def loop_pair(pair: str):
    try:
        ohlc = fetch_klines(pair)
        atr_v = atr(ohlc[-(ATR_LEN+1):], ATR_LEN)
        side, reason = detect_signal(ohlc)
        last_close = ohlc[-1][4]

        # manage open
        check_positions(pair, last_close)

        # open if no position
        if side and pair not in positions:
            open_position(pair, side, last_close, atr_v, reason)

    except Exception as e:
        log.warning(f"{pair} loop error: {e}")

# --------------------------- scheduler --------------------------------
from apscheduler.schedulers.background import BackgroundScheduler

def main():
    load_state()
    log.info(f"Mode {MODE} demo={DEMO} dry={DRY_RUN}  pairs={PAIRS} tf={TF}  "
             f"risk={RISK*100:.1f}% tp={TP_P*100:.3f}% atr_len={ATR_LEN} atr_mult={ATR_K} ema={EMA_LEN}/{SLOPE_N}")

    # старт polling (без вебхука, чтобы не ловить конфликт портов)
    updater.start_polling(clean=True)

    # работающий jobs
    sched = BackgroundScheduler(timezone=TZ)
    for p in PAIRS:
        # интервал чуть чаще TF, но сдвинут чтобы не ловить сырую свечу
        sched.add_job(lambda pair=p: loop_pair(pair), "interval", seconds=COOLDOWN, id=f"loop-{p}")
        bot.send_message(CHAT_ID, f"✅ Loop started for <b>{p}</b>", parse_mode=ParseMode.HTML)
    sched.start()
    log.info("Scheduler started")

    updater.idle()
    # при стопе сохраним
    save_state()

if __name__ == "__main__":
    main()
