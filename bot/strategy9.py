# bot/strategy9.py
import pandas as pd

# === индикаторы ===
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = (h - l).abs().to_frame("a")
    tr["b"] = (h - c.shift()).abs()
    tr["c"] = (l - c.shift()).abs()
    return tr.max(axis=1).rolling(n).mean()

# === свечные паттерны ===
def candle_patterns(df: pd.DataFrame) -> dict:
    out = {"bull_engulf": False, "bear_engulf": False, "hammer": False, "shooting": False}
    if len(df) < 2:
        return out
    a, b = df.iloc[-2], df.iloc[-1]
    # бычье поглощение
    if (b.close > b.open) and (a.close < a.open) and (b.close >= a.open) and (b.open <= a.close):
        out["bull_engulf"] = True
    # медвежье поглощение
    if (b.close < b.open) and (a.close > a.open) and (b.open >= a.close) and (b.close <= a.open):
        out["bear_engulf"] = True
    # молот
    body = abs(b.close - b.open)
    rng = b.high - b.low
    low_tail = min(b.close, b.open) - b.low
    up_tail = b.high - max(b.close, b.open)
    if rng > 0 and low_tail > 2 * body and up_tail < body:
        out["hammer"] = True
    # падающая звезда
    if rng > 0 and up_tail > 2 * body and low_tail < body:
        out["shooting"] = True
    return out

# === решение входа по стратегии #9 ===
def decide(df: pd.DataFrame) -> dict | None:
    """
    Правило:
      - тренд EMA48 растёт → ищем бычьи свечные сигналы (bull_engulf, hammer) → LONG
      - тренд EMA48 падает → ищем медвежьи (bear_engulf, shooting) → SHORT
      - стоп/тейк из ATR(14): SL = 0.5*ATR, TP = 1.0*ATR от цены входа
    Возврат: dict(side, reason, atr)
    """
    if len(df) < 60:
        return None
    e48 = ema(df["close"], 48)
    slope48 = e48.iloc[-1] - e48.iloc[-6]  # ~5 последних свечей
    patt = candle_patterns(df)
    direction = None
    if slope48 > 0 and (patt["bull_engulf"] or patt["hammer"]):
        direction = "LONG"
    elif slope48 < 0 and (patt["bear_engulf"] or patt["shooting"]):
        direction = "SHORT"
    if not direction:
        return None
    a = atr(df).iloc[-1]
    return {"side": direction, "reason": f"EMA48_slope={'up' if slope48>0 else 'down'} patt={patt}", "atr": float(a)}
