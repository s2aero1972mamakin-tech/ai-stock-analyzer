# logic.py
# -*- coding: utf-8 -*-
"""
JPX Swing Auto Scanner logic — STABLE5c-2026-02-28 (FULL)
- JPXユニバース: JPX公式の「東証上場銘柄一覧（33業種）」Excelから生成（CSV不要）
- データ: Stooq（日足CSV）
- 指標: SMA/RSI/ATR/出来高/売買代金
- フィルタ→Prelimスコア→TopK簡易バックテスト→ランキング
- 0件時 auto-relax（pullback→breakout + 条件緩和 + sector OFF 再スキャン）
"""

from __future__ import annotations

import dataclasses
import math
import time
import traceback
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

LOGIC_BUILD = "STABLE5c-2026-02-28"
_STOOQ_DOMAINS = ["stooq.pl", "stooq.com"]


@dataclass(frozen=True)
class SwingParams:
    require_sma25_over_sma75: bool = True
    entry_mode: str = "pullback"  # pullback / breakout

    rsi_low: float = 40.0
    rsi_high: float = 70.0

    pullback_low: float = -8.0
    pullback_high: float = -3.0

    atr_pct_min: float = 1.5
    atr_pct_max: float = 10.0

    vol_avg20_min: float = 50_000.0
    turnover_avg20_min_yen: float = 0.0

    breakout_lookback: int = 20
    breakout_vol_ratio: float = 1.6

    atr_mult_stop: float = 2.0
    tp1_r: float = 1.0
    tp2_r: float = 3.0
    time_stop_days: int = 10
    risk_pct: float = 0.01


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
# JPX master (no CSV)
# -------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_jpx_master() -> pd.DataFrame:
    """
    JPX公式の「東証上場銘柄一覧（33業種）」Excelを取得して ticker,name,sector を返す。
    取得失敗時は空DataFrameを返す（呼び出し側でエラー処理）。
    """
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        df = pd.read_excel(url)
    except Exception:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            # xlrd は requirements に入っている前提（あなたの環境ログにあり）
            df = pd.read_excel(BytesIO(r.content), engine="xlrd")
        except Exception:
            return pd.DataFrame()

    needed = {"コード", "銘柄名", "33業種区分"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.copy()
    df = df[df["33業種区分"].notna()]
    df = df[df["33業種区分"] != "-"]
    df["ticker"] = df["コード"].astype(str).str.zfill(4) + ".T"
    df = df.rename(columns={"銘柄名": "name", "33業種区分": "sector"})
    return df[["ticker", "name", "sector"]].drop_duplicates().reset_index(drop=True)


def get_company_name(ticker: str) -> str:
    m = get_jpx_master()
    if m.empty:
        return ticker
    row = m[m["ticker"] == ticker]
    if row.empty:
        return ticker
    return str(row.iloc[0]["name"])


# -------------------------
# Data (Stooq)
# -------------------------
def _stooq_symbol(ticker: str) -> str:
    t = str(ticker).strip()
    if t.endswith(".T"):
        code = t[:-2]
        if code.isdigit():
            return f"{code}.jp"
    if len(t) == 4 and t.isdigit():
        return f"{t}.jp"
    return t


def _http_get_csv(url: str, timeout: int = 25, retries: int = 3) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-stock-analyzer/1.0)"}
    backoff = 1.0
    for _ in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (429, 503, 502):
                time.sleep(backoff)
                backoff = min(8.0, backoff * 2)
                continue
            if r.status_code != 200:
                return None
            text = r.text
            if "Date,Open,High,Low,Close" not in text[:200]:
                return None
            return text
        except Exception:
            time.sleep(backoff)
            backoff = min(8.0, backoff * 2)
    return None


def get_market_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    sym = _stooq_symbol(ticker)
    for dom in _STOOQ_DOMAINS:
        url = f"https://{dom}/q/d/l/?s={sym}&i=d"
        text = _http_get_csv(url)
        if not text:
            continue
        try:
            from io import StringIO

            df = pd.read_csv(StringIO(text))
            if df.empty:
                continue
            df.columns = [c.strip().capitalize() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            if "Volume" not in df.columns:
                df["Volume"] = np.nan
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df = _slice_period(df, period)
            return df
        except Exception:
            continue
    return pd.DataFrame()


def _slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    p = str(period).strip().lower()
    if p.endswith("y"):
        years = float(p[:-1])
        return df.tail(int(365 * years))
    if p.endswith("d"):
        return df.tail(int(float(p[:-1])))
    return df.tail(730)


# -------------------------
# Indicators
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


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["SMA_25"] = sma(out["Close"], 25)
    out["SMA_75"] = sma(out["Close"], 75)
    out["RSI"] = rsi(out["Close"], 14)
    out["ATR"] = atr(out, 14)
    out["ATR_PCT"] = (out["ATR"] / out["Close"]) * 100.0
    out["VOL_AVG_20"] = out["Volume"].rolling(20).mean()
    out["VOL_RATIO"] = out["Volume"] / (out["VOL_AVG_20"] + 1e-12)
    out["SMA_DIFF"] = (out["Close"] / (out["SMA_25"] + 1e-12) - 1.0) * 100.0
    out["TURNOVER"] = out["Close"] * out["Volume"]
    out["TURNOVER_AVG_20"] = out["TURNOVER"].rolling(20).mean()
    out = out.dropna()
    return out


# -------------------------
# Filters / scoring
# -------------------------
def _score_prelim(latest: pd.Series, params: SwingParams) -> float:
    sma25 = float(latest.get("SMA_25", np.nan))
    sma75 = float(latest.get("SMA_75", np.nan))
    close = float(latest.get("Close", np.nan))
    if not (np.isfinite(sma25) and np.isfinite(sma75) and np.isfinite(close)):
        return 0.0
    trend = 1.0 if sma25 > sma75 else 0.0

    vol_avg20 = float(latest.get("VOL_AVG_20", 0.0))
    turn_avg20 = float(latest.get("TURNOVER_AVG_20", 0.0))
    liq = min(1.0, vol_avg20 / (params.vol_avg20_min + 1e-9))
    if params.turnover_avg20_min_yen > 0:
        liq = (liq + min(1.0, turn_avg20 / (params.turnover_avg20_min_yen + 1e-9))) / 2.0

    r = float(latest.get("RSI", 50.0))
    r_mid = (params.rsi_low + params.rsi_high) / 2.0
    r_span = max(5.0, (params.rsi_high - params.rsi_low))
    rsi_s = max(0.0, 1.0 - abs(r - r_mid) / (r_span / 2.0))

    atr_pct = float(latest.get("ATR_PCT", np.nan))
    atr_center = (params.atr_pct_min + params.atr_pct_max) / 2.0
    atr_span = max(0.5, params.atr_pct_max - params.atr_pct_min)
    atr_s = 1.0 - min(1.0, abs(atr_pct - atr_center) / (atr_span / 2.0 + 0.5))

    if params.entry_mode == "pullback":
        sma_diff = float(latest.get("SMA_DIFF", 0.0))
        pb_center = (params.pullback_low + params.pullback_high) / 2.0
        pb_span = max(1.0, params.pullback_high - params.pullback_low)
        setup = max(0.0, 1.0 - abs(sma_diff - pb_center) / (pb_span / 2.0 + 0.5))
    else:
        vr = float(latest.get("VOL_RATIO", 1.0))
        setup = min(1.0, max(0.0, (vr - 1.0) / (params.breakout_vol_ratio - 1.0 + 1e-9)))

    return float(35 * trend + 25 * liq + 15 * setup + 15 * rsi_s + 10 * atr_s)


def _passes_filters(ind: pd.DataFrame, params: SwingParams, stats: Dict[str, int]) -> bool:
    if ind is None or ind.empty:
        stats["fail_data"] += 1
        return False

    latest = ind.iloc[-1]
    sma25 = float(latest["SMA_25"])
    sma75 = float(latest["SMA_75"])
    if params.require_sma25_over_sma75 and not (sma25 > sma75):
        stats["fail_trend"] += 1
        return False

    rsi_v = float(latest["RSI"])
    if not (params.rsi_low <= rsi_v <= params.rsi_high):
        stats["fail_rsi"] += 1
        return False

    atr_pct = float(latest["ATR_PCT"])
    if not (params.atr_pct_min <= atr_pct <= params.atr_pct_max):
        stats["fail_atr"] += 1
        return False

    vol_avg20 = float(latest["VOL_AVG_20"])
    if vol_avg20 < params.vol_avg20_min:
        stats["fail_vol"] += 1
        return False

    if params.turnover_avg20_min_yen > 0:
        turn = float(latest.get("TURNOVER_AVG_20", 0.0))
        if turn < params.turnover_avg20_min_yen:
            stats["fail_turnover"] += 1
            return False

    if params.entry_mode == "pullback":
        sma_diff = float(latest["SMA_DIFF"])
        pb_ok = params.pullback_low <= sma_diff <= params.pullback_high
        trigger = bool(ind["Close"].iloc[-1] > ind["High"].iloc[-2]) if len(ind) >= 2 else False
        if not (pb_ok and trigger):
            stats["fail_setup"] += 1
            return False
    else:
        lb = int(params.breakout_lookback)
        if len(ind) < lb + 2:
            stats["fail_setup"] += 1
            return False
        prev_high = float(ind["High"].iloc[-lb - 1 : -1].max())
        breakout = float(latest["Close"]) > prev_high
        vr = float(latest["VOL_RATIO"])
        if not (breakout and vr >= params.breakout_vol_ratio):
            stats["fail_setup"] += 1
            return False

    stats["pass"] += 1
    return True


# -------------------------
# Backtest (same as previous)
# -------------------------
def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / (peak.abs() + 1e-12)
    return float(dd.min())


def backtest_swing(df_ind: pd.DataFrame, params: SwingParams) -> BacktestResult:
    if df_ind is None or df_ind.empty or len(df_ind) < 80:
        return BacktestResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pd.Series(dtype=float), pd.DataFrame())

    df = df_ind.copy()
    trades = []
    eq = 0.0
    equity = []

    in_pos = False
    entry = stop = tp1 = tp2 = 0.0
    entry_i = -1
    tp1_hit = False

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if not in_pos:
            if params.entry_mode == "pullback":
                sma_diff = float(row["SMA_DIFF"])
                pb_ok = params.pullback_low <= sma_diff <= params.pullback_high
                trigger = float(row["Close"]) > float(prev["High"])
                if not (pb_ok and trigger):
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
                if not (breakout and vr >= params.breakout_vol_ratio):
                    equity.append(eq)
                    continue

            entry = float(row["Close"])
            atr_v = float(row["ATR"])
            stop = entry - atr_v * float(params.atr_mult_stop)
            if stop <= 0 or entry <= stop:
                equity.append(eq)
                continue
            r_unit = entry - stop
            tp1 = entry + float(params.tp1_r) * r_unit
            tp2 = entry + float(params.tp2_r) * r_unit
            in_pos = True
            tp1_hit = False
            entry_i = i
            equity.append(eq)
            continue

        low = float(row["Low"])
        high = float(row["High"])
        close = float(row["Close"])
        r_unit = entry - stop
        held = i - entry_i

        if low <= stop:
            r = -1.0 if not tp1_hit else 0.0
            eq += r
            trades.append({"entry_i": entry_i, "exit_i": i, "r": r, "reason": "stop"})
            in_pos = False
            equity.append(eq)
            continue

        if high >= tp2:
            r = float(params.tp2_r) if not tp1_hit else (float(params.tp1_r) * 0.5 + float(params.tp2_r) * 0.5)
            eq += r
            trades.append({"entry_i": entry_i, "exit_i": i, "r": r, "reason": "tp2"})
            in_pos = False
            equity.append(eq)
            continue

        if (not tp1_hit) and high >= tp1:
            eq += float(params.tp1_r) * 0.5
            stop = entry
            tp1_hit = True
            equity.append(eq)
            continue

        if (not tp1_hit) and held >= int(params.time_stop_days):
            r = (close - entry) / (r_unit + 1e-12)
            eq += float(r)
            trades.append({"entry_i": entry_i, "exit_i": i, "r": float(r), "reason": "time"})
            in_pos = False
            equity.append(eq)
            continue

        equity.append(eq)

    equity_curve = pd.Series(equity, index=df.index[: len(equity)])
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return BacktestResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, _max_drawdown(equity_curve), equity_curve, trades_df)

    wins = trades_df[trades_df["r"] > 0]
    losses = trades_df[trades_df["r"] < 0]
    win_rate = float(len(wins) / len(trades_df))
    gross_profit = float(wins["r"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["r"].sum()) if not losses.empty else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy = float(trades_df["r"].mean())
    avg_win = float(wins["r"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["r"].mean()) if not losses.empty else 0.0
    max_dd = _max_drawdown(equity_curve)

    return BacktestResult(int(len(trades_df)), win_rate, profit_factor, expectancy, avg_win, avg_loss, max_dd, equity_curve, trades_df)


def _period_years(period: str) -> float:
    p = str(period).lower().strip()
    if p.endswith("y"):
        return float(p[:-1])
    return 2.0


def _rank_score(bt: BacktestResult, years: float) -> float:
    n = float(bt.n_trades)
    if n <= 0:
        return -1e9
    trades_per_year = n / max(0.25, float(years))
    sample_pen = min(1.0, n / 20.0)
    freq = min(2.0, math.log1p(trades_per_year))
    pf = bt.profit_factor
    if not np.isfinite(pf):
        pf = 5.0
    pf = float(min(5.0, max(0.0, pf)))
    dd = float(bt.max_drawdown)
    dd_pen = math.exp(-abs(dd) / 0.25)
    return float((bt.expectancy_r * 120.0) * sample_pen * (1.0 + freq) * (0.5 + 0.1 * pf) * dd_pen)


# -------------------------
# Scan with auto-relax
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
    relax_level: int = 0,
) -> Dict[str, object]:
    stats = {
        "universe": 0,
        "pass": 0,
        "fail_data": 0,
        "fail_trend": 0,
        "fail_rsi": 0,
        "fail_atr": 0,
        "fail_vol": 0,
        "fail_turnover": 0,
        "fail_setup": 0,
        "budget_ok": 0,
        "budget_ok": 0,
    }
    auto_relax_trace: List[dict] = []

    master = get_jpx_master()
    if master.empty:
        return {
            "mode": params.entry_mode,
            "candidates": [],
            "selected_sectors": [],
            "filter_stats": stats,
            "relax_level": relax_level,
            "params_effective": dataclasses.asdict(params),
            "auto_relax_trace": auto_relax_trace,
            "error": "JPXマスター（JPX公式Excel）の取得に失敗しました。ネットワーク/JPX側変更を確認してください。",
        }

    selected_sectors: List[str] = []
    if sector_prefilter:
        vc = master["sector"].astype(str).value_counts()
        selected_sectors = vc.head(int(sector_top_n)).index.tolist()
        master = master[master["sector"].astype(str).isin(selected_sectors)].copy()

    tickers = master["ticker"].astype(str).tolist()
    stats["universe"] = len(tickers)

    prelim: List[dict] = []
    total = len(tickers)

    partial_top: List[dict] = []
    partial_limit = max(8, int(top_n) * 3)

    for i, t in enumerate(tickers, start=1):
        if progress_callback and (i % 10 == 0 or i == 1):
            progress_callback(i, total, f"fetch+indicators {t}", partial=partial_top, stats=stats)

        df = get_market_data(t, period=backtest_period)
        ind = calculate_indicators(df)
        if not _passes_filters(ind, params, stats):
            continue

        latest = ind.iloc[-1]
        price = float(latest["Close"])
        if price * 100 > float(budget_yen):
            continue
        stats["budget_ok"] += 1
        stats["budget_ok"] += 1

        row = master[master["ticker"] == t].iloc[0]
        item = {
                "ticker": t,
                "name": str(row.get("name", "")),
                "sector": str(row.get("sector", "")),
                "price": price,
                "rsi": float(latest["RSI"]),
                "atr": float(latest["ATR"]),
                "atr_pct": float(latest["ATR_PCT"]),
                "vol_avg20": float(latest["VOL_AVG_20"]),
                "turnover_avg20": float(latest.get("TURNOVER_AVG_20", 0.0)),
                "score_pre": float(_score_prelim(latest, params)),
        }
        prelim.append(item)
        partial_top = sorted(prelim, key=lambda x: float(x.get("score_pre",0.0)), reverse=True)[:partial_limit]

    if not prelim:
        if relax_level == 0:
            relaxed = replace(
                params,
                require_sma25_over_sma75=False,
                rsi_low=max(20.0, params.rsi_low - 7.0),
                rsi_high=min(85.0, params.rsi_high + 7.0),
                atr_pct_min=max(0.5, params.atr_pct_min - 0.5),
                atr_pct_max=min(25.0, params.atr_pct_max + 5.0),
                vol_avg20_min=max(20_000.0, params.vol_avg20_min * 0.5),
                turnover_avg20_min_yen=0.0,
            )
            if params.entry_mode == "pullback":
                relaxed = replace(relaxed, entry_mode="breakout", breakout_vol_ratio=max(1.15, params.breakout_vol_ratio - 0.35))

            auto_relax_trace.append({"step": "auto_relax", "reason": "prelim_zero", "from": dataclasses.asdict(params), "to": dataclasses.asdict(relaxed)})

            return scan_swing_candidates(
                budget_yen=budget_yen,
                top_n=top_n,
                params=relaxed,
                progress_callback=progress_callback,
                backtest_period=backtest_period,
                backtest_topk=backtest_topk,
                sector_prefilter=False,
                sector_top_n=sector_top_n,
                relax_level=1,
            )

        return {
            "mode": params.entry_mode,
            "candidates": [],
            "selected_sectors": selected_sectors,
            "filter_stats": stats,
            "relax_level": relax_level,
            "params_effective": dataclasses.asdict(params),
            "auto_relax_trace": auto_relax_trace,
            "error": "候補0件（データ取得失敗 or 条件厳しすぎ）。filter_stats参照。",
        }

    prelim = sorted(prelim, key=lambda x: float(x["score_pre"]), reverse=True)[: max(int(backtest_topk), int(top_n))]
    years = _period_years(backtest_period)

    ranked: List[dict] = []
    for j, item in enumerate(prelim, start=1):
        if progress_callback:
            progress_callback(j, len(prelim), f"backtest {item['ticker']}", partial=prelim[: min(len(prelim), 12)], stats=stats)
        df = get_market_data(item["ticker"], period=backtest_period)
        ind = calculate_indicators(df)
        bt = backtest_swing(ind, params)
        ranked.append(
            {
                **item,
                "bt_trades": bt.n_trades,
                "bt_win_rate": bt.win_rate,
                "bt_pf": bt.profit_factor,
                "bt_avg_r": bt.expectancy_r,
                "bt_max_dd": bt.max_drawdown,
                "bt_score": _rank_score(bt, years),
            }
        )

    ranked = sorted(ranked, key=lambda x: float(x.get("bt_score", -1e9)), reverse=True)
    return {
        "mode": params.entry_mode,
        "candidates": ranked[: int(top_n)],
        "selected_sectors": selected_sectors,
        "filter_stats": stats,
        "relax_level": relax_level,
        "params_effective": dataclasses.asdict(params),
        "auto_relax_trace": auto_relax_trace,
    }


def build_trade_plan(df_ind: pd.DataFrame, params: SwingParams, capital_yen: int, risk_pct: float) -> Dict[str, object]:
    if df_ind is None or df_ind.empty:
        return {}
    latest = df_ind.iloc[-1]
    price = float(latest["Close"])
    atr_v = float(latest["ATR"])
    entry = price
    stop = entry - atr_v * float(params.atr_mult_stop)
    if stop <= 0 or entry <= stop:
        return {}
    r_unit = entry - stop
    tp1 = entry + float(params.tp1_r) * r_unit
    tp2 = entry + float(params.tp2_r) * r_unit
    risk_budget = float(capital_yen) * float(risk_pct)
    shares = int(max(0, math.floor(risk_budget / max(1e-9, r_unit))))
    shares = int((shares // 100) * 100)
    warning = None
    if shares <= 0:
        warning = "推奨株数が0（単元制約）。損切幅/資金/銘柄単価を調整してください。"
    return {
        "entry_price": entry,
        "stop_price": stop,
        "tp1_price": tp1,
        "tp2_price": tp2,
        "risk_per_share": r_unit,
        "shares": shares,
        "warning": warning,
    }
