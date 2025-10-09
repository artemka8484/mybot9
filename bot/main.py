# /workspace/bot/main.py

import os
import time
import math
import json
import threading
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional

import requests
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# ENV & CONFIG
# ────────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MEXC_BASE_URL    = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com").rstrip("/")
TIMEFRAME        = os.getenv("TIMEFRAME", "5m")   # 1m/5m/15m etc.
PAIRS            = [p.strip().upper() for p in os.getenv("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",") if p.strip()]

EMA_LEN          = int(os.getenv("EMA_LEN", "48"))
EMA_SLOPE_BARS   = int(os.getenv("EMA_SLOPE_BARS", "5"))

ATR_LEN          = int(os.getenv("ATR_LEN", "14"))
ATR_MULT_SL      = float(os.getenv("ATR_MULT_SL", "0.5"))  # SL = 0.5 × ATR (по тесту #9)
TP_PCT_RAW       = float(os.getenv("TP_PCT", "1"))         # если > 0, используем TP = X × ATR (по умолчанию 1)
FEE_PCT          = float(os.getenv("FEE_PCT", "0.0006"))

RISK_PCT         = float(os.getenv("RISK_PCT", "3.0"))
LEVERAGE         = float(os.getenv("LEVERAGE", "5"))

DEMO_MODE        = os.getenv("DEMO_MODE", "true").lower() == "true"
DEMO_START_BAL   = float(os.getenv("DEMO_START_BALANCE", "5000"))
DRY_RUN          = os.getenv("DRY_RUN", "false").lower() == "true"  # если true — не исполняем даже демо, только логика (для отладки)

COOLDOWN_SEC     = int(os.getenv("COOLDOWN_SEC", "60"))
DAILY_SUMMARY    = os.getenv("DAILY_SUMMARY", "1") == "1"

TICK_SEC         = 10  # период опроса рынка
HEALTH_PORTS     = [8080, 8081, 8090]  # на случай занятого порта

# ────────────────────────────────────────────────────────────────────────────────
# STATE (DEMO)
# ────────────────────────────────────────────────────────────────────────────────
lock = threading.Lock()

state: Dict[str, Any] = {
    "balance": DEMO_START_BAL,
    "open_positions": {},    # {pair: {side, qty, entry, sl, tp, time}}
    "cooldown": {},          # {pair: ts}
    "stats": {               # winrate & pnl
        "total": {"wins": 0, "losses": 0, "trades": 0, "pnl": 0.0},
        "by_pair": {}        # {pair: {wins, losses, trades, pnl}}
    },
    "day_anchor": datetime.now(timezone.utc).date(),
}

for p in PAIRS:
    state["stats"]["by_pair"][p] = {"wins": 0, "losses": 0, "trades": 0, "pnl": 0.0}

# ────────────────────────────────────────────────────────────────────────────────
# TELEGRAM helpers
# ────────────────────────────────────────────────────────────────────────────────
TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def tg_delete_webhook():
    if not TELEGRAM_TOKEN:
        return
    try:
        requests.get(f"{TG_API}/deleteWebhook", params={"drop_pending_updates": True}, timeout=10)
    except Exception:
        pass

def tg_send(text: str, parse_mode: Optional[str] = "HTML", disable_notification: bool = False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"{TG_API}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
                "disable_web_page_preview": True
            },
            timeout=10
        )
    except Exception:
        pass

# ────────────────────────────────────────────────────────────────────────────────
# Health server (не мешает, если порт занят)
# ────────────────────────────────────────────────────────────────────────────────
class Quiet(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, fmt, *args): return

def start_health_http_server():
    def serve():
        for port in HEALTH_PORTS:
            try:
                httpd = HTTPServer(("0.0.0.0", port), Quiet)
                httpd.serve_forever()
                break
            except OSError:
                continue
    th = threading.Thread(target=serve, daemon=True)
    th.start()

# ────────────────────────────────────────────────────────────────────────────────
# MARKET DATA (MEXC)
# ────────────────────────────────────────────────────────────────────────────────
def _mexc_tf(tf: str) -> str:
    # MEXC контрактный API любит интервалы формата Min1/Min5/Min15 ...
    if tf.endswith("m"):
        return f"Min{tf[:-1]}"
    if tf.endswith("h"):
        return f"Hour{tf[:-1]}"
    if tf.endswith("d"):
        return f"Day{tf[:-1]}"
    return "Min5"

def fetch_klines(pair: str, limit: int = 200) -> pd.DataFrame:
    """
    Возвращает DataFrame со столбцами: t, open, high, low, close, vol
    Робастно парсим разные ответы MEXC (списки/объекты).
    """
    sym = pair.replace("/", "").upper()
    interval = _mexc_tf(TIMEFRAME)
    url_variants = [
        f"{MEXC_BASE_URL}/api/v1/contract/kline/{sym}?interval={interval}&limit={limit}",
        f"{MEXC_BASE_URL}/api/v1/contract/kline?symbol={sym}&interval={interval}&limit={limit}",
    ]
    data = None
    for url in url_variants:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            js = r.json()
            # Возможны варианты: {"data":[[t,o,h,l,c,vol,...],...]} либо просто [[...],...]
            if isinstance(js, dict) and "data" in js:
                data = js["data"]
            elif isinstance(js, list):
                data = js
            if data:
                break
        except Exception:
            continue

    if not data or len(data) < 50:
        raise ValueError(f"no klines for {pair}")

    rows = []
    for row in data:
        # list: [t, open, high, low, close, vol, ...] или dict
        if isinstance(row, list):
            t = int(row[0])
            o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4]); v = float(row[5])
        elif isinstance(row, dict):
            t = int(row.get("time") or row.get("t") or row.get("timestamp"))
            o = float(row.get("open") or row.get("o")); h = float(row.get("high") or row.get("h"))
            l = float(row.get("low") or row.get("l"));  c = float(row.get("close") or row.get("c"))
            v = float(row.get("vol") or row.get("volume") or row.get("v") or 0)
        else:
            continue
        rows.append([t, o, h, l, c, v])

    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vol"])
    df = df.sort_values("t").reset_index(drop=True)
    return df

# ────────────────────────────────────────────────────────────────────────────────
# INDICATORS & PATTERNS
# ────────────────────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA
    df["ema"] = df["close"].ewm(span=EMA_LEN, adjust=False).mean()

    # EMA slope (разница за EMA_SLOPE_BARS)
    df["ema_slope"] = df["ema"] - df["ema"].shift(EMA_SLOPE_BARS)

    # ATR
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()

    # Свечные характеристики
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    df["body"] = body; df["upper"] = upper; df["lower"] = lower

    # Паттерны (булевы, обычные Python bool)
    def bull_engulf(i):
        if i < 1: return False
        o1,c1 = df.loc[i-1,["open","close"]]
        o2,c2 = df.loc[i,["open","close"]]
        return (c2 > o2) and (c1 < o1) and (c2 >= o1) and (o2 <= c1) and ((c2-o2) > (o1-c1))

    def bear_engulf(i):
        if i < 1: return False
        o1,c1 = df.loc[i-1,["open","close"]]
        o2,c2 = df.loc[i,["open","close"]]
        return (c2 < o2) and (c1 > o1) and (o2 >= c1) and (c2 <= o1) and ((o2-c2) > (c1-o1))

    def hammer(i):
        b = df.loc[i,"body"]; up = df.loc[i,"upper"]; lo = df.loc[i,"lower"]
        return (lo >= 2*b) and (up <= 0.3*b)

    def inv_hammer(i):
        b = df.loc[i,"body"]; up = df.loc[i,"upper"]; lo = df.loc[i,"lower"]
        return (up >= 2*b) and (lo <= 0.3*b)

    # Сохраняем булевы как Python bool
    bulls, bears, hams, invs = [], [], [], []
    for i in range(len(df)):
        bulls.append(bool(bull_engulf(i)))
        bears.append(bool(bear_engulf(i)))
        hams.append(bool(hammer(i)))
        invs.append(bool(inv_hammer(i)))
    df["bull_engulf"] = bulls
    df["bear_engulf"] = bears
    df["hammer"]      = hams
    df["shooting"]    = invs
    return df

# ────────────────────────────────────────────────────────────────────────────────
# STRATEGY #9
# ────────────────────────────────────────────────────────────────────────────────
def can_open(pair: str) -> bool:
    now = time.time()
    last = state["cooldown"].get(pair, 0)
    if now - last < COOLDOWN_SEC:
        return False
    if pair in state["open_positions"]:
        return False
    return True

def size_from_risk(price: float, atr: float) -> float:
    """
    Объём в «монетах» от риска (3% баланса) и стоп-расстояния = ATR_MULT_SL * atr.
    Для USDT-фьючерсов риск на 1 монету = stop_distance (в $),
    значит qty = risk_amount / stop_distance.
    """
    risk_amount = state["balance"] * (RISK_PCT / 100.0)
    stop_dist = max(1e-8, ATR_MULT_SL * atr)
    qty = risk_amount / stop_dist
    # ограничим по плечу: не больше value <= balance * leverage
    max_qty_by_leverage = (state["balance"] * LEVERAGE) / max(1e-8, price)
    qty = min(qty, max_qty_by_leverage)
    return max(0.0, qty)

def open_signal(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Правила входа:
    - лонг: ema_slope > 0 и один из паттернов: bull_engulf или hammer
    - шорт: ema_slope < 0 и один из паттернов: bear_engulf или shooting
    """
    i = len(df) - 1
    row = df.iloc[i]
    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return None

    long_ok = (row["ema_slope"] > 0) and (row["bull_engulf"] or row["hammer"])
    short_ok= (row["ema_slope"] < 0) and (row["bear_engulf"] or row["shooting"])
    if not (long_ok or short_ok):
        return None

    price = float(row["close"])
    atr = float(row["atr"])
    qty = size_from_risk(price, atr)
    if qty <= 0:
        return None

    # цели
    tp = price + (TP_PCT_RAW * atr) if long_ok else price - (TP_PCT_RAW * atr)
    sl = price - (ATR_MULT_SL * atr) if long_ok else price + (ATR_MULT_SL * atr)
    side = "LONG" if long_ok else "SHORT"
    pattern = "bull_engulf" if row["bull_engulf"] else "hammer" if row["hammer"] else "bear_engulf" if row["bear_engulf"] else "shooting"
    return {"side": side, "entry": price, "tp": tp, "sl": sl, "qty": qty, "pattern": pattern}

def check_exit(pos: Dict[str, Any], price: float) -> Optional[Dict[str, Any]]:
    """
    Простая проверка выхода: если цена достигла TP или SL.
    """
    side = pos["side"]
    entry= pos["entry"]
    tp   = pos["tp"]
    sl   = pos["sl"]
    qty  = pos["qty"]

    reached_tp = price >= tp if side == "LONG" else price <= tp
    reached_sl = price <= sl if side == "LONG" else price >= sl

    if not (reached_tp or reached_sl):
        return None

    # PnL (без финанс. учета маржи, демо упрощение)
    pnl_per_unit = (tp - entry) if reached_tp else (sl - entry)
    if side == "SHORT":
        pnl_per_unit = -pnl_per_unit
    gross = pnl_per_unit * qty
    fee_open = entry * qty * FEE_PCT
    fee_close= (tp if reached_tp else sl) * qty * FEE_PCT
    pnl = gross - fee_open - fee_close

    return {
        "exit_price": (tp if reached_tp else sl),
        "pnl": pnl,
        "win": pnl > 0.0
    }

# ────────────────────────────────────────────────────────────────────────────────
# LOOPS
# ────────────────────────────────────────────────────────────────────────────────
def pair_loop(pair: str):
    tg_send(f"✅ Loop started for <b>{pair}</b>", disable_notification=True)
    last_candle_ts = 0

    while True:
        try:
            df = fetch_klines(pair, limit=300)
            df = add_indicators(df)
            # обрабатываем только новую свечу (по времени)
            latest_ts = int(df["t"].iloc[-1])
            price = float(df["close"].iloc[-1])

            # 1) Закрытие позиции если есть
            with lock:
                pos = state["open_positions"].get(pair)

            if pos:
                ex = check_exit(pos, price)
                if ex:
                    with lock:
                        # обновить баланс, статы
                        state["balance"] += ex["pnl"]
                        st = state["stats"]
                        st["total"]["trades"] += 1
                        st["total"]["pnl"]    += ex["pnl"]
                        if ex["win"]:
                            st["total"]["wins"]  += 1
                        else:
                            st["total"]["losses"]+= 1

                        pst = st["by_pair"][pair]
                        pst["trades"] += 1
                        pst["pnl"]    += ex["pnl"]
                        if ex["win"]: pst["wins"] += 1
                        else:         pst["losses"] += 1

                        # собрать сообщение
                        total_trades = st["total"]["trades"]
                        total_wr = (st["total"]["wins"] / total_trades * 100.0) if total_trades else 0.0
                        pair_wr = (pst["wins"] / pst["trades"] * 100.0) if pst["trades"] else 0.0

                        emoji = "✅" if ex["win"] else "❌"
                        msg = (
                            f"{emoji} <b>{pair}</b> {pos['side']} <b>CLOSED</b>\n"
                            f"• entry: <code>{pos['entry']:.4f}</code>\n"
                            f"• exit: <code>{ex['exit_price']:.4f}</code>\n"
                            f"• qty: <code>{pos['qty']:.6f}</code>\n"
                            f"• pnl: <b><code>{ex['pnl']:.2f} USDT</code></b>\n"
                            f"• balance: <b><code>{state['balance']:.2f} USDT</code></b>\n"
                            f"• pair WR: <b>{pair_wr:.1f}%</b> ({pst['wins']}/{pst['trades']})\n"
                            f"• total WR: <b>{total_wr:.1f}%</b> ({st['total']['wins']}/{st['total']['trades']})"
                        )
                        tg_send(msg)
                        # позицию закрыть
                        state["open_positions"].pop(pair, None)
                        state["cooldown"][pair] = time.time()

            # 2) Открытие (только на новой свече, чтобы не дергаться)
            if latest_ts != last_candle_ts and can_open(pair):
                sig = open_signal(df)
                if sig and not DRY_RUN:
                    with lock:
                        # открыть позицию
                        state["open_positions"][pair] = {
                            "side": sig["side"],
                            "qty": sig["qty"],
                            "entry": sig["entry"],
                            "tp": sig["tp"],
                            "sl": sig["sl"],
                            "pattern": sig["pattern"],
                            "time": datetime.now(timezone.utc).isoformat()
                        }
                        # уведомление
                        msg = (
                            f"🟢 <b>{pair}</b> {sig['side']} <b>OPEN</b>\n"
                            f"• entry: <code>{sig['entry']:.4f}</code>\n"
                            f"• tp/sl: <code>{sig['tp']:.4f}</code> / <code>{sig['sl']:.4f}</code>\n"
                            f"• qty: <code>{sig['qty']:.6f}</code>\n"
                            f"• pattern: <code>{sig['pattern']}</code>\n"
                            f"• ema_slope/atr(len={ATR_LEN}): <code>{df['ema_slope'].iloc[-1]:.6f}</code> / <code>{df['atr'].iloc[-1]:.6f}</code>\n"
                            f"• risk: <code>{RISK_PCT:.1f}%</code> | lev: <code>{LEVERAGE:.1f}x</code>"
                        )
                        tg_send(msg, disable_notification=True)

            last_candle_ts = latest_ts
        except Exception as e:
            tg_send(f"⚠️ <b>{pair}</b> loop error: <code>{str(e)}</code>", disable_notification=True)
            time.sleep(3)
        time.sleep(TICK_SEC)

def daily_summary_loop():
    """
    Раз в минуту проверяем смену UTC-даты и шлём сводку за день.
    """
    while True:
        try:
            today = datetime.now(timezone.utc).date()
            with lock:
                if state["day_anchor"] != today and DAILY_SUMMARY:
                    state["day_anchor"] = today
                    st = state["stats"]
                    total = st["total"]
                    wr = (total["wins"] / total["trades"] * 100.0) if total["trades"] else 0.0

                    lines = [
                        "🧾 <b>Daily summary</b> (UTC)",
                        f"• balance: <b><code>{state['balance']:.2f} USDT</code></b>",
                        f"• total pnl: <b><code>{total['pnl']:.2f} USDT</code></b>",
                        f"• total trades: <b>{total['trades']}</b> | WR: <b>{wr:.1f}%</b>",
                        "• by pair:"
                    ]
                    for p, s in state["stats"]["by_pair"].items():
                        pwr = (s["wins"] / s["trades"] * 100.0) if s["trades"] else 0.0
                        lines.append(f"   - {p}: pnl <code>{s['pnl']:.2f}</code> | {s['wins']}/{s['trades']} ({pwr:.1f}%)")
                    tg_send("\n".join(lines))
        except Exception:
            pass
        time.sleep(60)

# ────────────────────────────────────────────────────────────────────────────────
# START
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # убрать конфликты Telegram
    tg_delete_webhook()

    # health
    start_health_http_server()

    # приветствие
    mode_str = "DEMO" if DEMO_MODE else "LIVE"
    tg_send(
        f"🤖 <b>mybot9</b> started successfully!\n"
        f"Mode: <b>{mode_str}</b> | Leverage <b>{LEVERAGE:.1f}x</b> | Fee <b>{FEE_PCT*100:.3f}%</b> | Risk <b>{RISK_PCT:.1f}%</b>\n"
        f"Pairs: <code>{', '.join(PAIRS)}</code> | TF <code>{TIMEFRAME}</code> | Tick <code>{TICK_SEC}s</code>\n"
        f"Balance: <b><code>{state['balance']:.2f} USDT</code></b>"
    )

    # запустить лупы по инструментам
    for p in PAIRS:
        th = threading.Thread(target=pair_loop, args=(p,), daemon=True)
        th.start()

    # суточная сводка
    th2 = threading.Thread(target=daily_summary_loop, daemon=True)
    th2.start()

    # держать процесс живым
    while True:
        time.sleep(5)

if __name__ == "__main__":
    main()
