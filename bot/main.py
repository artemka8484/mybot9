# bot/strategy_fusion_grand_v1.py
# -*- coding: utf-8 -*-
"""
Fusion Grand v1 (DrGrand #9 + Candlestick Bible + Nison + A-Z + Carter)
— таймфрейм M5, контекст H1, плечо x5, риск от баланса.
— входы: Engulfing, Morning/Evening Star c подтверждением (close > F1 / < F2)
— фильтры: H1 EMA50/200, M5 EMA20/50, ATR-медиана, импульс >= 1.2*ATR,
            свежесть уровня <= 10 дней, ретест <= 0.4*ATR
— менеджмент: TP1 = 0.5*(F1-F2), TP2 = 1.0*(F1-F2); после TP1 -> BE + трейл.
Автор: DrGrand
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ========================= utils / indicators =========================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr

def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    return true_range(h, l, c).rolling(n).mean()

# ========================= patterns (по книгам) =======================

@dataclass
class PatternFlags:
    bull_eng: pd.Series
    bear_eng: pd.Series
    morning:  pd.Series
    evening:  pd.Series

def detect_patterns(df: pd.DataFrame) -> PatternFlags:
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)

    # Строгое поглощение телом
    bull_eng_1 = (c1 > o1) & (c2 < o2) & ((c1 - o1) >= (o2 - c2)) & (c1 >= o2) & (o1 <= c2)
    bear_eng_1 = (o1 > c1) & (o2 < c2) & ((o1 - c1) >= (c2 - o2)) & (o1 >= c2) & (c1 <= o2)

    # Morning/Evening Star (трехсвечные, завершаются на i-1)
    body1 = (c1 - o1).abs()
    rng1  = (h1 - l1).abs()
    small_body1 = (body1 <= 0.3 * rng1)
    bear2 = (o2 > c2)
    bull1 = (c1 > o1)
    bear1 = (o1 > c1)

    morning_1 = bear2 & small_body1 & bull1 & (c1 > (o2 + c2) / 2)
    evening_1 = (~bear2 & (o2 < c2)) & small_body1 & bear1 & (c1 < (o2 + c2) / 2)

    return PatternFlags(bull_eng_1.fillna(False),
                        bear_eng_1.fillna(False),
                        morning_1.fillna(False),
                        evening_1.fillna(False))

# ========================= core params ================================

@dataclass
class FGParams:
    risk_pct: float = 0.03         # риск от баланса на сделку (3% как в #9)
    leverage: float = 5.0
    atr_quantile: float = 0.50     # торгуем верхнюю половину ATR
    impulse_mult: float = 1.2      # диапазон паттерна >= 1.2*ATR
    fresh_days: int = 10           # свежесть уровня (дней)
    retest_tol: float = 0.4        # допуск ретеста в долях ATR
    time_exit_bars: int = 25       # max удержание позиции (M5 ~ 2 часа)
    trail_lookback: int = 3        # трейл по локальным экстремумам
    tp1_frac: float = 0.5          # TP1 = 0.5*(F1-F2)
    tp2_frac: float = 1.0          # TP2 = 1.0*(F1-F2)

# ========================= signal structure ===========================

@dataclass
class Signal:
    ts: pd.Timestamp
    side: str                      # 'long' | 'short'
    entry: float
    sl: float
    tp1: float
    tp2: float
    size: float                    # объём в "монетах" (без учёта плеча)
    info: Dict

# ========================= main strategy ==============================

class FusionGrandV1:
    def __init__(self, params: Optional[FGParams] = None):
        self.p = params or FGParams()

    def _prepare(self, df5: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        df5: индекс — datetime UTC, колонки open,high,low,close,volume
        Возвращает (df5_enriched, h1_enriched)
        """
        df5 = df5.copy().sort_index()
        # индикаторы M5
        df5['ema20'] = ema(df5['close'], 20)
        df5['ema50'] = ema(df5['close'], 50)
        df5['atr']   = atr(df5['high'], df5['low'], df5['close'], 14)

        # контекст H1
        agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        h1 = df5.resample('1H').apply(agg).dropna()
        h1['ema50'] = ema(h1['close'], 50)
        h1['ema200'] = ema(h1['close'], 200)
        h1['trend'] = np.where(h1['ema50'] > h1['ema200'], 1, np.where(h1['ema50'] < h1['ema200'], -1, 0))
        df5['h1_trend'] = h1['trend'].reindex(df5.index, method='ffill').fillna(0)

        # фильтр ATR по перцентилю
        thr = df5['atr'].quantile(self.p.atr_quantile)
        df5['atr_ok'] = df5['atr'] >= thr

        # паттерны (на баре i-1)
        pats = detect_patterns(df5)
        o, h, l, c = df5['open'], df5['high'], df5['low'], df5['close']
        h1b, l1b = h.shift(1), l.shift(1)

        # импульс паттерна, свежесть, ретест
        impulse_ok = ( (h1b - l1b).abs() >= self.p.impulse_mult * df5['atr'].shift(1) )

        look = int((self.p.fresh_days * 24 * 60) / 5)  # N баров M5
        fresh_high = (h1b >= df5['high'].rolling(look).max().shift(1))
        fresh_low  = (l1b <= df5['low'].rolling(look).min().shift(1))

        tol = self.p.retest_tol * df5['atr']
        retest_long  = ( (h1b - l).abs() <= tol )
        retest_short = ( (h - l1b).abs() <= tol )

        # подтверждение A-Z (вход на текущем баре i)
        long_mask = (df5['h1_trend'] == 1) & df5['atr_ok'] & impulse_ok & retest_long & \
                    ((pats.bull_eng | pats.morning)) & (c > h1b) & fresh_high
        short_mask = (df5['h1_trend'] == -1) & df5['atr_ok'] & impulse_ok & retest_short & \
                     ((pats.bear_eng | pats.evening)) & (c < l1b) & fresh_low

        df5['long_sig'] = long_mask.fillna(False)
        df5['short_sig'] = short_mask.fillna(False)
        df5['F1'] = np.where(df5['long_sig'], h1b, np.where(df5['short_sig'], l1b, np.nan))
        df5['F2'] = np.where(df5['long_sig'], l1b, np.where(df5['short_sig'], h1b, np.nan))

        return df5, h1

    # --- позиционирование из риска ---
    def _position_size(self, equity: float, entry: float, sl: float) -> float:
        risk_usdt = max(1e-9, equity * self.p.risk_pct)
        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            return 0.0
        units = risk_usdt / risk_per_unit
        return float(units)

    # --- генератор сигналов на истории ---
    def generate_signals(self, df5: pd.DataFrame, equity: float) -> List[Signal]:
        df5, _ = self._prepare(df5)
        sigs: List[Signal] = []
        for ts, row in df5.iloc[250:].iterrows():  # пропуск «разогрева»
            if not (row.long_sig or row.short_sig):
                continue
            F1, F2, atr_now = float(row.F1), float(row.F2), float(row.atr)
            rng = max(1e-9, abs(F1 - F2))
            entry = float(row.close)
            if row.long_sig:
                sl = F2 - 0.1 * atr_now
                tp1 = entry + self.p.tp1_frac * rng
                tp2 = entry + self.p.tp2_frac * rng
                side = 'long'
            else:
                sl = F2 + 0.1 * atr_now
                tp1 = entry - self.p.tp1_frac * rng
                tp2 = entry - self.p.tp2_frac * rng
                side = 'short'

            size = self._position_size(equity, entry, sl)
            if size <= 0:
                continue

            sigs.append(Signal(
                ts=ts, side=side, entry=entry, sl=float(sl), tp1=float(tp1), tp2=float(tp2),
                size=size,
                info=dict(F1=float(F1), F2=float(F2), atr=float(atr_now),
                          h1_trend=int(row.h1_trend), impulse=self.p.impulse_mult,
                          fresh_days=self.p.fresh_days, retest_tol=self.p.retest_tol)
            ))
        return sigs

    # --- простая «эмуляция» менеджмента (TP1/TP2/BE/трейл) на истории ---
    def simulate(self, df5: pd.DataFrame, equity_start: float = 10_000.0) -> pd.DataFrame:
        df5 = df5.copy().sort_index()
        sigs = self.generate_signals(df5, equity_start)
        if not sigs:
            return pd.DataFrame(columns=[
                'open_time','close_time','side','entry','sl','tp1','tp2',
                'pnl','equity','result','half_taken'
            ])

        equity = equity_start
        recs = []
        i_map = {ts: i for i, ts in enumerate(df5.index)}
        for s in sigs:
            if s.ts not in i_map:  # защита
                continue
            opened_i = i_map[s.ts]
            in_pos = True
            half = False
            sl = s.sl
            result = ''
            pnl = 0.0
            close_ts = s.ts

            for j in range(opened_i + 1, min(len(df5), opened_i + self.p.time_exit_bars + 1)):
                row = df5.iloc[j]
                high, low, close = float(row.high), float(row.low), float(row.close)
                close_ts = df5.index[j]

                if s.side == 'long':
                    if low <= sl:
                        pnl = (sl - s.entry) * s.size; result = 'SL'; in_pos = False
                    else:
                        if (not half) and (high >= s.tp1):
                            pnl += (s.tp1 - s.entry) * (s.size * 0.5)
                            sl = s.entry; half = True
                        if high >= s.tp2:
                            pnl += (s.tp2 - s.entry) * (s.size * (0.5 if half else 1.0))
                            result = 'TP2' if half else 'TP_full'; in_pos = False
                        if half and in_pos:
                            sl = max(sl, float(df5['low'].iloc[max(j - self.p.trail_lookback, opened_i):j].min()))
                else:
                    if high >= sl:
                        pnl = (s.entry - sl) * s.size; result = 'SL'; in_pos = False
                    else:
                        if (not half) and (low <= s.tp1):
                            pnl += (s.entry - s.tp1) * (s.size * 0.5)
                            sl = s.entry; half = True
                        if low <= s.tp2:
                            pnl += (s.entry - s.tp2) * (s.size * (0.5 if half else 1.0))
                            result = 'TP2' if half else 'TP_full'; in_pos = False
                        if half and in_pos:
                            sl = min(sl, float(df5['high'].iloc[max(j - self.p.trail_lookback, opened_i):j].max()))

                if not in_pos:
                    break

            if in_pos:  # тайм-аут
                close = float(df5['close'].iloc[min(opened_i + self.p.time_exit_bars, len(df5) - 1)])
                rem = s.size * (0.5 if half else 1.0)
                pnl += ((close - s.entry) if s.side == 'long' else (s.entry - close)) * rem
                result = 'HX'

            equity += pnl
            recs.append(dict(
                open_time=s.ts, close_time=close_ts, side=s.side, entry=s.entry,
                sl=s.sl, tp1=s.tp1, tp2=s.tp2, pnl=pnl, equity=equity,
                result=result, half_taken=half
            ))

        return pd.DataFrame(recs)

# ========================= пример использования =======================
if __name__ == "__main__":
    # пример: загрузка CSV с M5 (колонки: time/open_time, open, high, low, close, volume)
    import os
    path = os.getenv("M5_CSV", "/mnt/data/BTCUSDT_5m_3y.csv")
    df = pd.read_csv(path)
    # нормализация колонок
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    ts_col = 'time' if 'time' in cols else ('open_time' if 'open_time' in cols else cols[0])
    df['time'] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['time','open','high','low','close']).set_index('time').sort_index()

    strat = FusionGrandV1(FGParams())
    report = strat.simulate(df, equity_start=10_000.0)
    out = "fusion_grand_report.csv"
    report.to_csv(out, index=False)
    print("Saved:", out, "Trades:", len(report), "End equity:", float(report['equity'].iloc[-1]) if len(report) else 0)
