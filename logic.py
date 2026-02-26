# logic.py
# -*- coding: utf-8 -*-
"""
日本株スイング自動スキャン ロジック（Stooq）
- 取得: Stooq（終値/高値/安値/出来高）
- 指標: SMA/RSI/ATR/出来高平均/出来高倍率 等
- セクター事前絞り込み（quant / ai_overlay）
- 0件時 auto-relax（pullback→breakout + 緩和 + sector OFF 再スキャン）
- Stage1: 事前スコア（トレンド/流動性/モメンタム/押し目/ブレイクアウト）
- Stage2: topKだけ簡易バックテストし、勝率×利確幅×頻度のバランスで順位付け
"""

from __future__ import annotations

import datetime
import math
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple
import dataclasses

import numpy as np
import pandas as pd
import requests

LOGIC_BUILD = "2026-02-26T00:00:00Z / stable"


# -------------------------
# 1) Params / Result types
# -------------------------
@dataclass(frozen=True)
class SwingParams:
    # Filters (structure)
    require_sma25_over_sma75: bool = True
    entry_mode: str = "pullback"  # pullback / breakout

    # RSI band
    rsi_low: float = 40.0
    rsi_high: float = 70.0

    # Pullback: % diff from SMA25 (negative values)
    pullback_low: float = -8.0
    pullback_high: float = -3.0

    # ATR% bounds
    atr_pct_min: float = 1.5
    atr_pct_max: float = 10.0

    # Liquidity
    vol_avg20_min: float = 50_000.0
    turnover_avg20_min_yen: float = 0.0  # optional (0 = disable)

    # Breakout
    breakout_lookback: int = 20
    breakout_vol_ratio: float = 1.6

    # Risk / exits
    atr_mult_stop: float = 2.0
    tp1_r: float = 1.0
    tp2_r: float = 3.0
    time_stop_days: int = 10
    risk_pct: float = 0.01  # capital risk per trade (for plan sizing)


@dataclass
class BacktestResult:
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    avg_win_r: float
    avg_loss_r: float
    max_drawdown: float
    equity_curve_r: Optional[pd.Series]
    trades: pd.DataFrame


# -------------------------
# 2) Data access (Stooq)
# -------------------------
def _stooq_symbol(ticker: str) -> str:
    """
    Convert JP ticker like '8306.T' into Stooq symbol if needed.
    Stooq uses formats like '8306.jp' for Japan in many cases.
    """
    t = ticker.strip()
    if t.endswith(".T"):
        code = t[:-2]
        if code.isdigit():
            return f"{code}.jp"
    # accept already stooq-like
    return t


def get_market_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV from Stooq as daily data.
    Note: Stooq offers daily; interval is kept for compatibility.
    """
    sym = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        df = pd.read_csv(url)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.strip().capitalize() for c in df.columns]
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        # Standardize
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                # Sometimes volume missing
                if c == "Volume":
                    df["Volume"] = np.nan
                else:
                    return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        # Period slicing (approx)
        if period:
            df = _slice_period(df, period)
        return df
    except Exception:
        return pd.DataFrame()


def _slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    p = str(period).strip().lower()
    if p.endswith("y"):
        years = float(p[:-1])
        days = int(365 * years)
        return df.tail(days)
    if p.endswith("mo"):
        months = float(p[:-2])
        days = int(30 * months)
        return df.tail(days)
    if p.endswith("m"):
        months = float(p[:-1])
        days = int(30 * months)
        return df.tail(days)
    if p.endswith("d"):
        days = int(float(p[:-1]))
        return df.tail(days)
    return df


def get_company_name(ticker: str) -> str:
    # Minimal placeholder (can be replaced by JPX name master)
    return ""


# -------------------------
# 3) Indicators
# -------------------------
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def calculate_indicators(df: pd.DataFrame, include_sma200: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["SMA_5"] = sma(out["Close"], 5)
    out["SMA_25"] = sma(out["Close"], 25)
    out["SMA_75"] = sma(out["Close"], 75)
    if include_sma200:
        out["SMA_200"] = sma(out["Close"], 200)

    out["RSI"] = rsi(out["Close"], 14)
    out["ATR"] = atr(out, 14)
    out["ATR_PCT"] = (out["ATR"] / out["Close"]) * 100.0

    out["VOL_AVG_20"] = out["Volume"].rolling(20).mean()
    out["VOL_RATIO"] = out["Volume"] / (out["VOL_AVG_20"] + 1e-12)

    out["SMA_DIFF"] = (out["Close"] / (out["SMA_25"] + 1e-12) - 1.0) * 100.0

    # Turnover: approximate (Close * Volume)
    out["TURNOVER"] = out["Close"] * out["Volume"]
    out["TURNOVER_AVG_20"] = out["TURNOVER"].rolling(20).mean()

    out = out.dropna()
    return out


# -------------------------
# 4) Universe / Master (JPX)
# -------------------------
def load_jpx_master() -> pd.DataFrame:
    """
    Minimal master for this project: you should replace with a real JPX master
    (ticker, name, sector). For now, we build a lightweight list from Stooq availability.
    """
    # Placeholder: user likely has their own master list in repo.
    # If not, return empty and scan won't run.
    return pd.DataFrame(columns=["ticker", "name", "sector"])


def get_jpx_tickers() -> List[str]:
    master = load_jpx_master()
    if master is not None and not master.empty and "ticker" in master.columns:
        return master["ticker"].astype(str).tolist()
    return []


# -------------------------
# 5) Regime filter
# -------------------------
def check_regime_ok() -> Optional[bool]:
    """
    Simple regime: N225 Close > SMA200
    Uses Stooq symbol for Nikkei 225: 'nkx.jp' often works; fallback returns None.
    """
    try:
        df = get_market_data("nkx.jp", period="2y", interval="1d")
        ind = calculate_indicators(df, include_sma200=True)
        if ind.empty or "SMA_200" not in ind.columns:
            return None
        last = ind.iloc[-1]
        return bool(float(last["Close"]) > float(last["SMA_200"]))
    except Exception:
        return None


# -------------------------
# 6) Prelim scoring & filters
# -------------------------
def _score_prelim(latest: pd.Series, params: SwingParams) -> float:
    """
    Stage1 score: trend+liquidity+momentum+setup quality.
    ここは「候補を拾う」ための粗いスコアで、最終はStage2のバックテスト評価へ。
    """
    # trend score: SMA25 vs SMA75
    sma25 = float(latest.get("SMA_25", np.nan))
    sma75 = float(latest.get("SMA_75", np.nan))
    close = float(latest.get("Close", np.nan))
    if not (np.isfinite(sma25) and np.isfinite(sma75) and np.isfinite(close)):
        return 0.0
    trend = 1.0 if sma25 > sma75 else 0.0

    # liquidity score: volume and turnover
    vol_avg20 = float(latest.get("VOL_AVG_20", 0.0))
    turn_avg20 = float(latest.get("TURNOVER_AVG_20", 0.0))
    liq_score = min(1.0, (vol_avg20 / (params.vol_avg20_min + 1e-9)))
    if params.turnover_avg20_min_yen and params.turnover_avg20_min_yen > 0:
        liq_score = (liq_score + min(1.0, turn_avg20 / (params.turnover_avg20_min_yen + 1e-9))) / 2.0

    # RSI score: within band -> closer to mid is better
    r = float(latest.get("RSI", 50.0))
    r_mid = (params.rsi_low + params.rsi_high) / 2.0
    r_span = max(5.0, (params.rsi_high - params.rsi_low))
    rsi_score = max(0.0, 1.0 - abs(r - r_mid) / (r_span / 2.0))

    # setup score (pullback vs breakout)
    if params.entry_mode == "pullback":
        sma_diff = float(latest.get("SMA_DIFF", 0.0))
        # more negative (within range) is better, but avoid too deep
        pb_center = (params.pullback_low + params.pullback_high) / 2.0
        pb_span = max(1.0, params.pullback_high - params.pullback_low)
        setup = max(0.0, 1.0 - abs(sma_diff - pb_center) / (pb_span / 2.0 + 0.5))
    else:
        vr = float(latest.get("VOL_RATIO", 1.0))
        setup = min(1.0, max(0.0, (vr - 1.0) / (params.breakout_vol_ratio - 1.0 + 1e-9)))

    # ATR% score: prefer mid in [min,max]
    atr_pct = float(latest.get("ATR_PCT", np.nan))
    atr_center = (float(params.atr_pct_min) + float(params.atr_pct_max)) / 2.0
    atr_span = max(0.5, float(params.atr_pct_max) - float(params.atr_pct_min))
    atr_score_mid = 1.0 - min(1.0, abs(atr_pct - atr_center) / (atr_span / 2.0 + 0.5))

    score = 35.0 * trend + 25.0 * liq_score + 15.0 * setup + 15.0 * rsi_score + 10.0 * atr_score_mid
    return float(score)


def _passes_filters(ind: pd.DataFrame, params: SwingParams, stats: Dict[str, int]) -> Tuple[bool, Optional[str]]:
    if ind is None or ind.empty:
        stats["fail_data"] += 1
        return False, "no_data"

    latest = ind.iloc[-1]
    price = float(latest["Close"])
    if not np.isfinite(price) or price <= 0:
        stats["fail_price"] += 1
        return False, "bad_price"

    sma25 = float(latest["SMA_25"])
    sma75 = float(latest["SMA_75"])
    trend_ok = True
    if params.require_sma25_over_sma75:
        trend_ok = bool(sma25 > sma75)

    rsi_v = float(latest["RSI"])
    rsi_ok = bool(params.rsi_low <= rsi_v <= params.rsi_high)

    atr_pct = float(latest["ATR_PCT"])
    atr_ok = bool(params.atr_pct_min <= atr_pct <= params.atr_pct_max)

    vol_avg20 = float(latest["VOL_AVG_20"])
    vol_ok = bool(vol_avg20 >= params.vol_avg20_min)

    turnover_ok = True
    if params.turnover_avg20_min_yen and params.turnover_avg20_min_yen > 0:
        turnover_ok = bool(float(latest["TURNOVER_AVG_20"]) >= params.turnover_avg20_min_yen)

    if not trend_ok:
        stats["fail_trend"] += 1
        return False, "trend"
    stats["trend_ok"] += 1

    if not rsi_ok:
        stats["fail_rsi"] += 1
        return False, "rsi"
    stats["rsi_ok"] += 1

    if not atr_ok:
        stats["fail_atr"] += 1
        return False, "atr"
    stats["atr_ok"] += 1

    if not vol_ok:
        stats["fail_vol"] += 1
        return False, "vol"
    stats["vol_ok"] += 1

    if not turnover_ok:
        stats["fail_turnover"] += 1
        return False, "turnover"
    stats["turnover_ok"] += 1

    # Setup condition
    if params.entry_mode == "pullback":
        sma_diff = float(latest["SMA_DIFF"])
        pb_ok = bool(params.pullback_low <= sma_diff <= params.pullback_high)
        trigger = bool(ind["Close"].iloc[-1] > ind["High"].iloc[-2]) if len(ind) >= 2 else False
        if not (pb_ok and trigger):
            stats["fail_setup"] += 1
            return False, "pullback_setup"
        stats["setup_ok"] += 1
    else:
        lb = int(params.breakout_lookback)
        prev_high = float(ind["High"].iloc[-lb - 1 : -1].max()) if len(ind) > lb + 1 else float("nan")
        breakout = bool(price > prev_high) if np.isfinite(prev_high) else False
        vr = float(latest["VOL_RATIO"])
        if not (breakout and vr >= params.breakout_vol_ratio):
            stats["fail_setup"] += 1
            return False, "breakout_setup"
        stats["setup_ok"] += 1

    return True, None


# -------------------------
# 7) Backtest (simple R-based)
# -------------------------
def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / (peak.abs() + 1e-12)
    return float(dd.min())  # negative


def backtest_swing(df_ind: pd.DataFrame, params: SwingParams) -> BacktestResult:
    """
    Simple rules:
    - Entry signal is based on mode (pullback trigger / breakout)
    - Stop = entry - ATR*mult
    - TP1 at +tp1_r R (partial exit assumed in expectancy), TP2 at +tp2_r R
    - Time stop if TP1未達
    R-based accounting: 1R = (entry - stop)
    """
    if df_ind is None or df_ind.empty or len(df_ind) < 60:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, pd.Series(dtype=float), pd.DataFrame())

    df = df_ind.copy()
    trades = []
    equity = []
    eq = 0.0

    in_pos = False
    entry = stop = tp1 = tp2 = 0.0
    entry_i = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if not in_pos:
            # Build entry signal
            if params.entry_mode == "pullback":
                sma_diff = float(row["SMA_DIFF"])
                pb_ok = params.pullback_low <= sma_diff <= params.pullback_high
                trigger = float(row["Close"]) > float(prev["High"])
                if pb_ok and trigger:
                    entry = float(row["Close"])
                else:
                    equity.append(eq)
                    continue
            else:
                lb = int(params.breakout_lookback)
                if i <= lb + 1:
                    equity.append(eq)
                    continue
                prev_high = float(df["High"].iloc[i - lb : i].max())
                breakout = float(row["Close"]) > prev_high
                vr = float(row["VOL_RATIO"])
                if breakout and vr >= params.breakout_vol_ratio:
                    entry = float(row["Close"])
                else:
                    equity.append(eq)
                    continue

            atr_v = float(row["ATR"])
            stop = entry - atr_v * float(params.atr_mult_stop)
            if stop <= 0 or entry <= stop:
                equity.append(eq)
                continue
            r_unit = entry - stop
            tp1 = entry + float(params.tp1_r) * r_unit
            tp2 = entry + float(params.tp2_r) * r_unit
            in_pos = True
            entry_i = i
            equity.append(eq)
            continue

        # If in position, check exits
        low = float(row["Low"])
        high = float(row["High"])
        close = float(row["Close"])

        r_unit = entry - stop
        if r_unit <= 0:
            in_pos = False
            equity.append(eq)
            continue

        # Stop hit
        if low <= stop:
            r = -1.0
            eq += r
            trades.append({"entry_i": entry_i, "exit_i": i, "r": r, "reason": "stop"})
            in_pos = False
            equity.append(eq)
            continue

        # TP2 hit
        if high >= tp2:
            r = float(params.tp2_r)
            eq += r
            trades.append({"entry_i": entry_i, "exit_i": i, "r": r, "reason": "tp2"})
            in_pos = False
            equity.append(eq)
            continue

        # TP1 partial: we model as half exit at tp1, then either tp2/stop/breakeven
        # For simplicity, approximate: if TP1 hit then set stop to entry (breakeven) and keep going.
        if high >= tp1:
            # realize half at TP1
            r_tp1 = float(params.tp1_r) * 0.5
            eq += r_tp1
            # move stop to breakeven for remaining half
            stop = entry
            # then we keep position; if later TP2, we take remaining half at TP2
            # (approx: remaining half contributes 0.5*tp2_r)
            # but to keep it simple without state explosion, handle as:
            # if later hit tp2 => add 0.5*tp2_r; if stop => add 0.0 for remaining.
            # We'll convert by tracking a flag in_pos_half
            in_pos_half = True
        else:
            in_pos_half = False

        # time stop (if TP1未達)
        held_days = i - (entry_i if entry_i is not None else i)
        if (not in_pos_half) and held_days >= int(params.time_stop_days):
            # exit at close with R
            r = (close - entry) / r_unit
            eq += r
            trades.append({"entry_i": entry_i, "exit_i": i, "r": r, "reason": "time"})
            in_pos = False
            equity.append(eq)
            continue

        # If TP1 was hit earlier (in_pos_half), and now tp2 hit => take remaining half
        if in_pos_half and high >= tp2:
            eq += float(params.tp2_r) * 0.5
            trades.append({"entry_i": entry_i, "exit_i": i, "r": (r_tp1 + float(params.tp2_r) * 0.5), "reason": "tp1_tp2"})
            in_pos = False
            equity.append(eq)
            continue

        # If TP1 was hit earlier and stop moved to entry, then if low <= stop => remaining half exits at 0R
        if in_pos_half and low <= stop:
            trades.append({"entry_i": entry_i, "exit_i": i, "r": r_tp1, "reason": "tp1_be"})
            in_pos = False
            equity.append(eq)
            continue

        equity.append(eq)

    equity_curve = pd.Series(equity, index=df.index[: len(equity)])
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return BacktestResult(0, 0, 0, 0, 0, 0, _max_drawdown(equity_curve), equity_curve, trades_df)

    wins = trades_df[trades_df["r"] > 0]
    losses = trades_df[trades_df["r"] < 0]
    win_rate = float(len(wins) / len(trades_df)) if len(trades_df) else 0.0
    gross_profit = float(wins["r"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["r"].sum()) if not losses.empty else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy = float(trades_df["r"].mean())
    avg_win = float(wins["r"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["r"].mean()) if not losses.empty else 0.0  # negative
    max_dd = _max_drawdown(equity_curve)

    return BacktestResult(
        n_trades=int(len(trades_df)),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_r=expectancy,
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
        max_drawdown=max_dd,
        equity_curve_r=equity_curve,
        trades=trades_df,
    )


# -------------------------
# 8) Grid search (single ticker)
# -------------------------
def grid_search_params(df_ind: pd.DataFrame, base: SwingParams) -> pd.DataFrame:
    """
    1銘柄に対して、軽量なグリッドでパラメータ感度を見る。
    PFとExpectancyを見て、"勝率＋利確"が両立するレンジを見つける用途。
    """
    if df_ind is None or df_ind.empty:
        return pd.DataFrame()

    rsi_lows = [35, 40, 45]
    rsi_highs = [60, 65, 70]
    pb_lows = [-8, -6, -4]
    pb_highs = [-3, -2, -1]
    modes = ["pullback", "breakout"]

    rows = []
    for mode in modes:
        for rl in rsi_lows:
            for rh in rsi_highs:
                if rl >= rh:
                    continue
                for pl in pb_lows:
                    for ph in pb_highs:
                        if pl >= ph:
                            continue
                        params = SwingParams(
                            rsi_low=float(rl),
                            rsi_high=float(rh),
                            pullback_low=float(pl),
                            pullback_high=float(ph),
                            atr_pct_min=base.atr_pct_min,
                            atr_pct_max=base.atr_pct_max,
                            vol_avg20_min=base.vol_avg20_min,
                            turnover_avg20_min_yen=base.turnover_avg20_min_yen,
                            require_sma25_over_sma75=base.require_sma25_over_sma75,
                            entry_mode=mode,
                            atr_mult_stop=base.atr_mult_stop,
                            tp1_r=base.tp1_r,
                            tp2_r=base.tp2_r,
                            time_stop_days=base.time_stop_days,
                            breakout_lookback=base.breakout_lookback,
                            breakout_vol_ratio=base.breakout_vol_ratio,
                            risk_pct=base.risk_pct,
                        )
                        bt = backtest_swing(df_ind, params)
                        # Score: expectancy primary, PF secondary, trades penalize too-low sample
                        sample = bt.n_trades
                        sample_pen = min(1.0, sample / 15.0)  # 15 trades to reach 1.0
                        score = (bt.expectancy_r * 100.0) * sample_pen + (min(5.0, bt.profit_factor) * 5.0) + (bt.win_rate * 10.0)
                        rows.append(
                            {
                                "mode": mode,
                                "rsi": f"{rl}-{rh}",
                                "pullback": f"{pl}-{ph}",
                                "trades": bt.n_trades,
                                "win_rate": bt.win_rate,
                                "pf": bt.profit_factor,
                                "avg_r": bt.expectancy_r,
                                "max_dd": bt.max_drawdown,
                                "score": score,
                            }
                        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["score", "avg_r", "pf", "trades"], ascending=False).reset_index(drop=True)
    return out


# -------------------------
# 9) Sector prefilter (placeholder)
# -------------------------
def prefilter_sectors(master: pd.DataFrame, top_n: int = 6, method: str = "quant", api_key: Optional[str] = None) -> Tuple[List[str], List[Dict[str, object]]]:
    """
    Quant-only fallback: pick most frequent sectors in master.
    Replace here with your own sector scoring.
    """
    if master is None or master.empty or "sector" not in master.columns:
        return [], []

    counts = master["sector"].astype(str).value_counts()
    selected = counts.head(int(top_n)).index.tolist()
    ranking = [{"sector": s, "score": float(counts.loc[s])} for s in selected]
    return selected, ranking


# -------------------------
# 10) Candidate scan (auto-relax included)
# -------------------------
def scan_swing_candidates(
    budget_yen: int,
    top_n: int,
    params: SwingParams,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    backtest_period: str = "2y",
    backtest_topk: int = 20,
    sector_prefilter: bool = True,
    sector_top_n: int = 6,
    sector_method: str = "quant",
    api_key: Optional[str] = None,
    relax_level: int = 0,
) -> Dict[str, object]:
    stats = {
        "universe": 0,
        "fail_data": 0,
        "fail_price": 0,
        "fail_trend": 0,
        "fail_rsi": 0,
        "fail_atr": 0,
        "fail_vol": 0,
        "fail_turnover": 0,
        "fail_setup": 0,
        "trend_ok": 0,
        "rsi_ok": 0,
        "atr_ok": 0,
        "vol_ok": 0,
        "turnover_ok": 0,
        "setup_ok": 0,
        "prelim_pass": 0,
    }

    auto_relax_trace = []
    mode = params.entry_mode

    regime_ok = check_regime_ok()

    master = load_jpx_master()
    if master is None or master.empty:
        return {
            "mode": mode,
            "regime_ok": regime_ok,
            "candidates": [],
            "prelim_count": 0,
            "bt_count": 0,
            "selected_sectors": [],
            "sector_ranking": [],
            "universe": 0,
            "filter_stats": stats,
            "relax_level": relax_level,
            "params_effective": dataclasses.asdict(params),
            "auto_relax_trace": auto_relax_trace,
            "error": "JPX master が空です（ticker/name/sector の一覧が必要）",
        }

    # Sector prefilter
    selected_sectors = []
    sector_rank_records = []
    filtered_master = master
    if sector_prefilter:
        selected_sectors, sector_rank_records = prefilter_sectors(master, top_n=sector_top_n, method=sector_method, api_key=api_key)
        if selected_sectors:
            filtered_master = master[master["sector"].astype(str).isin(selected_sectors)].copy()

    tickers = filtered_master["ticker"].astype(str).tolist()
    stats["universe"] = len(tickers)

    prelim: List[Dict[str, object]] = []
    total = len(tickers)

    for i, t in enumerate(tickers, start=1):
        if progress_callback and i % 10 == 0:
            progress_callback(i, total, f"指標計算 {t}")

        df = get_market_data(t, period=backtest_period, interval="1d")
        ind = calculate_indicators(df, include_sma200=False)
        ok, reason = _passes_filters(ind, params, stats)
        if not ok:
            continue

        latest = ind.iloc[-1]
        score_pre = _score_prelim(latest, params)

        row_m = filtered_master[filtered_master["ticker"] == t].iloc[0]
        prelim.append(
            {
                "ticker": t,
                "name": str(row_m.get("name", "")),
                "sector": str(row_m.get("sector", "")),
                "price": float(latest["Close"]),
                "rsi": float(latest["RSI"]),
                "atr": float(latest["ATR"]),
                "atr_pct": float(latest["ATR_PCT"]),
                "vol_avg20": float(latest["VOL_AVG_20"]),
                "turnover_avg20": float(latest.get("TURNOVER_AVG_20", 0.0)),
                "score_pre": float(score_pre),
            }
        )

    prelim_count = len(prelim)
    stats["prelim_pass"] = prelim_count

    # Auto-relax if 0 candidates
    if prelim_count == 0:
        if relax_level == 0:
            relaxed = replace(
                params,
                require_sma25_over_sma75=False,
                rsi_low=max(20.0, float(params.rsi_low) - 5.0),
                rsi_high=min(85.0, float(params.rsi_high) + 5.0),
                pullback_low=float(params.pullback_low) - 4.0,
                pullback_high=float(params.pullback_high) + 2.0,
                atr_pct_min=max(0.5, float(params.atr_pct_min) - 0.5),
                atr_pct_max=min(25.0, float(params.atr_pct_max) + 4.0),
                vol_avg20_min=max(20_000.0, float(params.vol_avg20_min) * 0.5),
                turnover_avg20_min_yen=0.0,
            )
            if str(params.entry_mode) == "pullback":
                relaxed = replace(
                    relaxed,
                    entry_mode="breakout",
                    breakout_vol_ratio=max(1.2, float(params.breakout_vol_ratio) - 0.3),
                )

            auto_relax_trace.append(
                {
                    "from": dataclasses.asdict(params),
                    "to": dataclasses.asdict(relaxed),
                    "sector_prefilter": bool(sector_prefilter),
                    "sector_prefilter_relaxed": False,
                }
            )

            return scan_swing_candidates(
                budget_yen=budget_yen,
                top_n=top_n,
                params=relaxed,
                progress_callback=progress_callback,
                backtest_period=backtest_period,
                backtest_topk=backtest_topk,
                sector_prefilter=False,  # 緩和時は全体で拾う
                sector_top_n=sector_top_n,
                sector_method=sector_method,
                api_key=api_key,
                relax_level=1,
            )

        return {
            "mode": mode,
            "regime_ok": regime_ok,
            "candidates": [],
            "prelim_count": 0,
            "bt_count": 0,
            "selected_sectors": selected_sectors,
            "sector_ranking": sector_rank_records,
            "universe": len(tickers),
            "filter_stats": stats,
            "relax_level": relax_level,
            "params_effective": dataclasses.asdict(params),
            "auto_relax_trace": auto_relax_trace,
            "error": "候補が0件でした（条件が厳しい/データ取得失敗の可能性）",
        }

    prelim_sorted = sorted(prelim, key=lambda x: float(x["score_pre"]), reverse=True)[: max(int(backtest_topk), int(top_n))]
    bt_count = len(prelim_sorted)

    # Stage 2: backtest topK and re-rank by balanced score
    ranked: List[Dict[str, object]] = []
    years = _period_to_years(backtest_period)

    for i, item in enumerate(prelim_sorted, start=1):
        if progress_callback:
            progress_callback(i, bt_count, f"バックテスト {item['ticker']}")

        df = get_market_data(item["ticker"], period=backtest_period, interval="1d")
        ind = calculate_indicators(df, include_sma200=False)
        bt = backtest_swing(ind, params)

        ranked.append(
            {
                **item,
                "bt_trades": bt.n_trades,
                "bt_win_rate": bt.win_rate,
                "bt_pf": bt.profit_factor,
                "bt_avg_r": bt.expectancy_r,
                "bt_max_dd": bt.max_drawdown,
                "bt_score": _rank_score(bt, years, params),
            }
        )

    ranked = sorted(
        ranked,
        key=lambda x: float(x.get("bt_score", 0.0)),
        reverse=True,
    )

    return {
        "mode": mode,
        "regime_ok": regime_ok,
        "selected_sectors": selected_sectors,
        "sector_ranking": sector_rank_records,
        "universe": len(tickers),
        "candidates": ranked[: int(top_n)],
        "prelim_count": prelim_count,
        "bt_count": bt_count,
        "filter_stats": stats,
        "relax_level": relax_level,
        "params_effective": dataclasses.asdict(params),
        "auto_relax_trace": auto_relax_trace,
    }


def _period_to_years(period: str) -> float:
    p = str(period).strip().lower()
    if p.endswith("y"):
        return float(p[:-1])
    if p.endswith("mo"):
        return float(p[:-2]) / 12.0
    if p.endswith("m"):
        return float(p[:-1]) / 12.0
    if p.endswith("d"):
        return float(p[:-1]) / 365.0
    return 2.0


def _rank_score(bt: BacktestResult, years: float, params: SwingParams) -> float:
    """
    Stage2ランキング用の安定スコア（勝率×利確幅×頻度 を同時に設計）。
    - expectancy(R) を主軸にしつつ、少数トレードを縮小し、頻度・PF・DDで整える。
    """
    n = float(bt.n_trades)
    trades_per_year = n / max(0.25, float(years))

    # Sample penalty: few trades => lower confidence
    sample_pen = min(1.0, n / 20.0)  # 20 trades to reach 1.0

    # Frequency term: too low => bad; too high => likely noise (cap)
    freq = min(2.0, math.sqrt(max(0.0, trades_per_year) / 6.0))  # 6/year => 1.0

    # PF cap to avoid inf dominating
    pf = bt.profit_factor
    if not np.isfinite(pf):
        pf = 5.0
    pf = float(max(0.0, min(5.0, pf)))

    # DD penalty (negative), cap
    dd = float(bt.max_drawdown)
    dd_pen = max(-0.5, dd)  # dd is negative

    # Score
    score = (
        (bt.expectancy_r * 120.0) * sample_pen
        + (bt.win_rate * 20.0)
        + (pf * 8.0)
        + (freq * 10.0)
        + (dd_pen * 40.0)
    )
    return float(score)


# -------------------------
# 11) Trade plan (order sheet)
# -------------------------
def build_trade_plan(
    df_ind: pd.DataFrame,
    params: SwingParams,
    capital_yen: int,
    risk_pct: float,
    allow_partial_tp: bool = True,
) -> Dict[str, object]:
    """
    現在の指標から、次のエントリーを想定した注文書を数値で生成。
    """
    if df_ind is None or df_ind.empty:
        return {}

    latest = df_ind.iloc[-1]
    price = float(latest["Close"])
    atr_v = float(latest["ATR"])
    entry_price = price
    stop_price = entry_price - atr_v * float(params.atr_mult_stop)
    if stop_price <= 0 or entry_price <= stop_price:
        return {}

    r_unit = entry_price - stop_price
    tp1_price = entry_price + float(params.tp1_r) * r_unit
    tp2_price = entry_price + float(params.tp2_r) * r_unit

    risk_budget = float(capital_yen) * float(risk_pct)
    shares = int(max(0, math.floor(risk_budget / max(1e-9, r_unit))))
    shares = int((shares // 100) * 100)  # round down to 100-share lots

    warning = None
    if shares <= 0:
        warning = "資金/損切幅に対して推奨株数が0になりました（単元制約）。損切幅を縮める/資金を増やす/低位株を選ぶ等を検討。"

    return {
        "entry_price": entry_price,
        "stop_price": stop_price,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "risk_per_share": r_unit,
        "shares": shares,
        "warning": warning,
    }


# -------------------------
# 12) Optional OpenAI calls (kept minimal)
# -------------------------
def _call_openai(api_key: str, system: str, user: str) -> str:
    # Placeholder: user already has their own implementation; keep safe.
    return "（OpenAI呼び出しはこのテンプレでは未実装です。プロジェクト側の実装に置き換えてください）"


def get_ai_market_analysis(api_key: str, ctx: dict) -> str:
    return _call_openai(
        api_key,
        "あなたは実戦派の株式ストラテジストです。曖昧な表現を避け、箇条書き中心で要点とリスクを短く提示してください。",
        f"""対象:{ctx.get('pair_label')}
現在値:{ctx.get('price'):.2f}円
RSI:{ctx.get('rsi'):.1f}
ATR:{ctx.get('atr'):.2f}
SMA5:{ctx.get('sma5')}
SMA25:{ctx.get('sma25')}
バックテスト: PF={ctx.get('pf')}, AvgR={ctx.get('avg_r')}, WinRate={ctx.get('win_rate')}
この銘柄を「1週間〜1ヶ月のスイング」で取引する前提で、優位性が出る局面・避けたい局面・注意点を提案してください。
""",
    )


def get_ai_order_strategy(api_key: str, ctx: dict) -> str:
    return _call_openai(
        api_key,
        "あなたは冷徹な執行責任者です。必ず数値を含め、注文実行の手順（何をいつ置くか）を明確に書いてください。",
        f"""対象:{ctx.get('pair_label')}
現在値:{ctx.get('price'):.2f}円
想定エントリー:{ctx.get('entry_price'):.2f}円
損切:{ctx.get('stop_price'):.2f}円
利確1:{ctx.get('tp1_price'):.2f}円
利確2:{ctx.get('tp2_price'):.2f}円
時間切れ:{ctx.get('time_stop_days')}営業日
推奨株数:{ctx.get('shares')}
PF={ctx.get('pf')}, AvgR={ctx.get('avg_r')}, WinRate={ctx.get('win_rate')}
この条件で、単元株（100株単位）を前提に、OCO/逆指値が使える一般的な国内株注文として命令書を作成してください。
""",
    )


def get_ai_portfolio(api_key: str, ctx: dict) -> str:
    return _call_openai(
        api_key,
        "あなたはポートフォリオマネージャーです。短期スイングにおける『保有継続/撤退/追加』判断をロジック優先で提案してください。",
        f"""対象:{ctx.get('pair_label')}
PF={ctx.get('pf')}, AvgR={ctx.get('avg_r')}, WinRate={ctx.get('win_rate')}, MaxDD={ctx.get('max_dd')}
この銘柄を保有している前提で、優先すべきリスク管理と、追加/撤退判断を提案してください。
""",
    )

