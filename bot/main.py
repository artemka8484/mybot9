# bot/main.py
import os, hmac, hashlib, asyncio, json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import httpx
from loguru import logger
from telegram import Bot

# ===================== ENV =====================
PAIRS = [p.strip().upper() for p in os.getenv(
    "PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",") if p.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "1m")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0") or "0")
TX = os.getenv("TX", "UTC")
START_BALANCE = float(os.getenv("START_BALANCE_USDT", "1000"))
TRADE_SIZE = float(os.getenv("TRADE_SIZE", "0.001"))

# risk management
TP_PCT = float(os.getenv("TP_PCT", "0.005"))          # 0.5% по умолчанию
SL_PCT = float(os.getenv("SL_PCT", "0.004"))          # 0.4%
TRAILING = os.getenv("TRAILING", "0").lower() in ("1","true","yes")
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.003"))    # 0.3%

# daily summary
DAILY_SUMMARY = os.getenv("DAILY_SUMMARY", "1").lower() in ("1","true","yes")
SUMMARY_HOUR = int(os.getenv("SUMMARY_HOUR", "21"))   # час отправки по UTC

# MEXC keys (для REAL режима)
MEXC_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET = os.getenv("MEXC_API_SECRET", "")

BASE_SPOT = "https://api.mexc.com"
tg = Bot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

# ===================== UTILS =====================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

async def send(msg: str):
    if tg:
        try:
            await tg.send_message(CHAT_ID, msg, disable_web_page_preview=True)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

# ===================== DATA =====================
async def fetch_klines(pair: str, limit: int = 600) -> pd.DataFrame:
    """
    /api/v3/klines -> 12 полей; берём нужные.
    """
    url = f"{BASE_SPOT}/api/v3/klines"
    params = {"symbol": pair, "interval": TIMEFRAME, "limit": limit}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "_1","_2","_3","_4","_5","close_time"
    ])
    df = df[["open_time","open","high","low","close","volume","close_time"]].copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df.dropna().reset_index(drop=True)

# ===================== STRATEGY #9 =====================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    if len(df) < 3:
        return {"bull_engulf": False, "bear_engulf": False, "hammer": False, "shooting": False}
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    i1, i0 = -2, -1
    # engulf
    bull = (c[i1] < o[i1]) and (c[i0] > o[i0]) and ((c[i0]-o[i0]) > (o[i1]-c[i1]) > 0)
    bear = (c[i1] > o[i1]) and (c[i0] < o[i0]) and ((o[i0]-c[i0]) > (c[i1]-o[i1]) > 0)
    # hammer / shooting
    body = abs(c[i0]-o[i0])
    up = h[i0]-max(c[i0], o[i0])
    lo = min(c[i0], o[i0])-l[i0]
    hammer = (lo > body*2) and (up < body)
    shooting = (up > body*2) and (lo < body)
    return {k: bool(v) for k,v in {
        "bull_engulf": bull, "bear_engulf": bear, "hammer": hammer, "shooting": shooting
    }.items()}

def strategy9_signal(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    if len(df) < 60:
        return "HOLD", {}
    df = df.copy()
    df["ema48"] = ema(df["close"], 48)
    slope = float(df["ema48"].iloc[-1] - df["ema48"].iloc[-6])
    patt = detect_patterns(df)
    last = float(df["close"].iloc[-1])

    if slope > 0 and (patt["bull_engulf"] or patt["hammer"]):
        pat = "bull_engulf" if patt["bull_engulf"] else "hammer"
        return "BUY", {"last": last, "slope": slope, "pattern": pat, "reason": "trend_up + pattern"}
    if slope < 0 and (patt["bear_engulf"] or patt["shooting"]):
        pat = "bear_engulf" if patt["bear_engulf"] else "shooting"
        return "SELL", {"last": last, "slope": slope, "pattern": pat, "reason": "trend_down + pattern"}
    return "HOLD", {"last": last, "slope": slope, "pattern": None}

# ===================== EXECUTION LAYER =====================
class Portfolio:
    def __init__(self, start_balance: float):
        self.balance = start_balance  # USDT
        self.pos: Dict[str, Dict[str, Any]] = {}  # pair -> state
        self.closed: List[Dict[str, Any]] = []    # сделки за день

    def is_open(self, pair: str) -> bool:
        return pair in self.pos

    def open(self, pair: str, side: str, price: float, qty: float, reason: str,
             tp_pct: float, sl_pct: float, trailing: bool, trail_pct: float):
        # Только BUY для упрощения (лонг). SELL сигнал закрывает лонг/не открывает шорт.
        state = {
            "side": side,
            "qty": qty,
            "entry": price,
            "entry_ts": now_utc(),
            "reason_open": reason,
            "tp": price*(1+tp_pct) if side=="BUY" else price*(1-tp_pct),
            "sl": price*(1-sl_pct) if side=="BUY" else price*(1+sl_pct),
            "trailing": trailing,
            "trail_pct": trail_pct,
            "trail_anchor": price  # максимум после входа (для BUY)
        }
        self.pos[pair] = state
        return state

    def update_trailing(self, pair: str, last: float):
        p = self.pos.get(pair)
        if not p or not p.get("trailing"): return
        if p["side"] == "BUY":
            if last > p["trail_anchor"]:
                p["trail_anchor"] = last
                p["sl"] = p["trail_anchor"]*(1 - p["trail_pct"])  # подтягиваем стоп

    def close(self, pair: str, price: float, reason_close: str) -> Dict[str, Any]:
        p = self.pos.pop(pair)
        pnl = (price - p["entry"]) * p["qty"] * (1 if p["side"]=="BUY" else -1)
        self.balance += pnl
        res = {
            "pair": pair, "side": p["side"], "qty": p["qty"],
            "entry": p["entry"], "entry_ts": p["entry_ts"],
            "exit": price, "exit_ts": now_utc(), "pnl": pnl,
            "balance": self.balance,
            "reason_open": p.get("reason_open",""),
            "reason_close": reason_close
        }
        self.closed.append(res)
        return res

portfolio = Portfolio(START_BALANCE)

# ----------- MEXC REAL (MARKET) ----------
def _sign(query: str) -> str:
    return hmac.new(MEXC_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

async def mexc_order_market(symbol: str, side: str, quantity: float) -> Dict[str, Any]:
    """
    Простой маркет-ордер (base quantity). Используется ТОЛЬКО когда DEMO_MODE=False.
    """
    ts = int(datetime.utcnow().timestamp()*1000)
    params = f"symbol={symbol}&side={side}&type=MARKET&quantity={quantity}&timestamp={ts}&recvWindow=5000"
    headers = {"X-MEXC-APIKEY": MEXC_KEY}
    url = f"{BASE_SPOT}/api/v3/order?{params}&signature={_sign(params)}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers)
        r.raise_for_status()
        return r.json()

# ===================== REPORTS =====================
async def report_open(pair: str, side: str, qty: float, price: float, pattern: Optional[str], reason: str):
    msg = (
        f"🟩 OPEN {side}\n"
        f"• Pair: {pair}  TF: {TIMEFRAME}\n"
        f"• Qty: {qty}\n"
        f"• Entry: {price:.6f}\n"
        f"• Time: {fmt_ts(now_utc())}\n"
        f"• Pattern: {pattern or '—'}\n"
        f"• Reason: {reason}\n"
        f"• Mode: {'DEMO' if DEMO_MODE else 'REAL'}"
    )
    await send(msg)

async def report_close(result: Dict[str, Any], pattern_exit: Optional[str]):
    msg = (
        f"🟥 CLOSE {result['side']}\n"
        f"• Pair: {result['pair']}  TF: {TIMEFRAME}\n"
        f"• Qty: {result['qty']}\n"
        f"• Entry: {result['entry']:.6f} → Exit: {result['exit']:.6f}\n"
        f"• Open: {fmt_ts(result['entry_ts'])}\n"
        f"• Close: {fmt_ts(result['exit_ts'])}\n"
        f"• Exit pattern: {pattern_exit or '—'}\n"
        f"• Reason: {result['reason_close']}\n"
        f"• PnL: {'+' if result['pnl']>=0 else ''}{result['pnl']:.4f} USDT\n"
        f"• Balance (DEMO): {result['balance']:.2f} USDT"
    )
    await send(msg)

async def send_daily_summary():
    if not portfolio.closed:
        await send("📊 Daily summary: сделок не было.")
        return
    total = sum(t["pnl"] for t in portfolio.closed)
    wins = sum(1 for t in portfolio.closed if t["pnl"]>0)
    losses = len(portfolio.closed)-wins
    by_pair: Dict[str, float] = {}
    for t in portfolio.closed:
        by_pair[t["pair"]] = by_pair.get(t["pair"], 0.0) + t["pnl"]
    lines = [f"📊 Daily summary ({datetime.utcnow().date()} UTC)"]
    lines.append(f"• Deals: {len(portfolio.closed)} | Win: {wins} | Loss: {losses}")
    lines.append(f"• PnL total: {'+' if total>=0 else ''}{total:.4f} USDT")
    lines.append("• By pair:")
    for k,v in by_pair.items():
        lines.append(f"  - {k}: {'+' if v>=0 else ''}{v:.4f}")
    lines.append(f"• Balance (DEMO): {portfolio.balance:.2f} USDT")
    await send("\n".join(lines))
    # очистим список сделок на новый день
    portfolio.closed.clear()

# ===================== PAIR WORKER =====================
async def run_pair(pair: str):
    logger.info(f"worker started: {pair}")
    while True:
        try:
            df = await fetch_klines(pair, limit=400)
            signal, info = strategy9_signal(df)
            last = info.get("last", float(df["close"].iloc[-1]))

            # trailing обновление
            if portfolio.is_open(pair):
                portfolio.update_trailing(pair, last)

            # срабатывание TP/SL (для BUY)
            if portfolio.is_open(pair):
                st = portfolio.pos[pair]
                if st["side"] == "BUY":
                    if last >= st["tp"]:
                        res = portfolio.close(pair, last, "Take Profit")
                        await report_close(res, detect_patterns(df) and
                                           ( "bullish_cont" if last>=st["tp"] else None))
                    elif last <= st["sl"]:
                        res = portfolio.close(pair, last, "Stop Loss")
                        await report_close(res, detect_patterns(df) and
                                           ( "bearish_break" if last<=st["sl"] else None))

            # торговые сигналы
            if signal == "BUY" and not portfolio.is_open(pair):
                # DEMO/REAL open
                if DEMO_MODE:
                    portfolio.open(pair, "BUY", last, TRADE_SIZE, info["reason"],
                                   TP_PCT, SL_PCT, TRAILING, TRAIL_PCT)
                else:
                    try:
                        await mexc_order_market(pair, "BUY", TRADE_SIZE)
                        portfolio.open(pair, "BUY", last, TRADE_SIZE, info["reason"],
                                       TP_PCT, SL_PCT, TRAILING, TRAIL_PCT)
                    except Exception as e:
                        logger.error(f"REAL BUY failed {pair}: {e}")
                        await send(f"⚠️ REAL BUY failed {pair}: {e}")
                await report_open(pair, "BUY", TRADE_SIZE, last, info.get("pattern"), info["reason"])

            elif signal == "SELL" and portfolio.is_open(pair):
                if DEMO_MODE:
                    res = portfolio.close(pair, last, "Opposite signal")
                else:
                    try:
                        await mexc_order_market(pair, "SELL", TRADE_SIZE)
                        res = portfolio.close(pair, last, "Opposite signal")
                    except Exception as e:
                        logger.error(f"REAL SELL failed {pair}: {e}")
                        await send(f"⚠️ REAL SELL failed {pair}: {e}")
                        res = None
                if res:
                    await report_close(res, info.get("pattern"))

        except Exception as e:
            logger.exception(e)
        await asyncio.sleep(10)  # частота обработки по паре

# ===================== DAILY SUMMARY SCHEDULER =====================
async def summary_scheduler():
    """
    Каждую минуту проверяем — если наступил SUMMARY_HOUR (UTC) и ещё не отправляли сегодня — шлём.
    """
    if not DAILY_SUMMARY:  # можно полностью отключить
        return
    sent_for_day: Optional[datetime.date] = None
    while True:
        try:
            utcnow = datetime.utcnow()
            if utcnow.hour == SUMMARY_HOUR and (sent_for_day != utcnow.date()):
                await send_daily_summary()
                sent_for_day = utcnow.date()
        except Exception as e:
            logger.exception(e)
        await asyncio.sleep(60)

# ===================== BOOT =====================
async def run_bot():
    logger.info("🤖 mybot9 started successfully!")
    tasks = [asyncio.create_task(run_pair(p)) for p in PAIRS]
    if DAILY_SUMMARY:
        tasks.append(asyncio.create_task(summary_scheduler()))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run_bot())
