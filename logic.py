# logic.py
# -*- coding: utf-8 -*-
"""
JPX Swing Auto Scanner logic — FIXED3
- JPX公式 data_j.xls を read_excel 失敗時に read_html でフォールバック
- Stooq CSV取得はタイムアウト/リトライ/バックオフ付き
- filter_stats 後方互換維持 + fail_budget 追加
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

LOGIC_BUILD = "STABLE5d-2026-02-28-FIXED3"

JPX_XLS_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
STOOQ_URLS = [
    "https://stooq.pl/q/d/l/?s={sym}&i=d",
    "https://stooq.com/q/d/l/?s={sym}&i=d",
]


@dataclass(frozen=True)
class SwingParams:
    entry_mode: str = "pullback"
    require_sma25_over_sma75: bool = True
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


@st.cache_data(ttl=86400, show_spinner=False)
def get_jpx_master() -> pd.DataFrame:
    try:
        r = requests.get(JPX_XLS_URL, timeout=30)
        r.raise_for_status()
        content = r.content
    except Exception:
        return pd.DataFrame()

    try:
        df = pd.read_excel(BytesIO(content))
        return _normalize_jpx(df)
    except Exception:
        pass

    try:
        html = content.decode("cp932", errors="ignore")
        tables = pd.read_html(StringIO(html))
        if not tables:
            return pd.DataFrame()
        df = tables[0]
        return _normalize_jpx(df)
    except Exception:
        return pd.DataFrame()


def _normalize_jpx(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = set(df.columns.astype(str).tolist())
    need = {"コード", "銘柄名", "33業種区分"}
    if not need.issubset(cols):
        return pd.DataFrame()

    out = df.copy()
    out = out[out["33業種区分"].notna()]
    out = out[out["33業種区分"] != "-"]
    out["ticker"] = out["コード"].astype(str).str.zfill(4) + ".T"
    out = out.rename(columns={"銘柄名": "name", "33業種区分": "sector"})
    return out[["ticker", "name", "sector"]].drop_duplicates().reset_index(drop=True)


def _stooq_symbol(ticker: str) -> str:
    t = str(ticker).strip()
    if t.endswith(".T"):
        code = t[:-2]
        if code.isdigit():
            return f"{code}.jp"
    if len(t) == 4 and t.isdigit():
        return f"{t}.jp"
    return t


def _http_get_csv(url: str, timeout: int = 20, retries: int = 4) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-stock-analyzer/1.0)"}
    backoff = 1.0
    for _ in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (429, 502, 503):
                time.sleep(backoff)
                backoff = min(10.0, backoff * 2)
                continue
            if r.status_code != 200:
                return None
            text = r.text
            if "Date,Open,High,Low,Close" not in text[:300]:
                return None
            return text
        except Exception:
            time.sleep(backoff)
            backoff = min(10.0, backoff * 2)
    return None


def get_market_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    sym = _stooq_symbol(ticker)
    for templ in STOOQ_URLS:
        url = templ.format(sym=sym)
        text = _http_get_csv(url)
        if not text:
            continue
        try:
            df = pd.read_csv(StringIO(text))
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
            df.columns = [c.strip().capitalize() for c in df.columns]
            if "Volume" not in df.columns:
                df["Volume"] = np.nan
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open","High","Low","Close"])
            return _slice_period(df, period)
        except Exception:
            continue
    return pd.DataFrame()


def _slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    p = str(period).lower().strip()
    if p.endswith("y"):
        years = float(p[:-1])
        return df.tail(int(365 * years))
    if p.endswith("d"):
        return df.tail(int(float(p[:-1])))
    return df.tail(730)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


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
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
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
    out["SMA_DIFF"] = (out["Close"] / (out["SMA_25"] + 1e-12) - 1.0) * 100.0
    return out.dropna()


def _passes_filters(ind: pd.DataFrame, params: SwingParams, stats: Dict[str, int]) -> bool:
    if ind is None or ind.empty:
        stats["fail_data"] += 1
        return False
    latest = ind.iloc[-1]

    if params.require_sma25_over_sma75 and not (float(latest["SMA_25"]) > float(latest["SMA_75"])):
        stats["fail_trend"] += 1
        return False

    r = float(latest["RSI"])
    if not (params.rsi_low <= r <= params.rsi_high):
        stats["fail_rsi"] += 1
        return False

    atrp = float(latest["ATR_PCT"])
    if not (params.atr_pct_min <= atrp <= params.atr_pct_max):
        stats["fail_atr"] += 1
        return False

    vol20 = float(latest["VOL_AVG_20"])
    if vol20 < params.vol_avg20_min:
        stats["fail_vol"] += 1
        return False

    sma_diff = float(latest["SMA_DIFF"])
    if params.entry_mode == "pullback":
        if not (params.pullback_low <= sma_diff <= params.pullback_high):
            stats["fail_setup"] += 1
            return False
    else:
        lb = int(params.breakout_lookback)
        if len(ind) < lb + 2:
            stats["fail_setup"] += 1
            return False
        prev_high = float(ind["High"].iloc[-lb - 1 : -1].max())
        if not (float(latest["Close"]) > prev_high):
            stats["fail_setup"] += 1
            return False

    stats["pass"] += 1
    return True


def scan_swing_candidates(
    budget_yen: int,
    top_n: int,
    params: SwingParams,
    progress_callback: Optional[Callable] = None,
    period: str = "2y",
    sector_prefilter: bool = True,
    sector_top_n: int = 6,
) -> Dict[str, object]:
    stats = {
        "universe": 0,
        "pass": 0,
        "fail_data": 0,
        "fail_trend": 0,
        "fail_rsi": 0,
        "fail_atr": 0,
        "fail_vol": 0,
        "fail_setup": 0,
        "budget_ok": 0,
        "fail_budget": 0,
    }
    auto_relax_trace: List[dict] = []

    master = get_jpx_master()
    if master.empty:
        return {
            "mode": params.entry_mode,
            "candidates": [],
            "selected_sectors": [],
            "filter_stats": stats,
            "auto_relax_trace": auto_relax_trace,
            "params_effective": dataclasses.asdict(params),
            "error": "JPXマスター取得に失敗（ネットワーク/JPX側仕様/HTML解釈）",
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
    partial_limit = max(10, int(top_n) * 5)

    for i, t in enumerate(tickers, start=1):
        if progress_callback and (i % 10 == 0 or i == 1):
            progress_callback(i, total, f"fetch+indicators {t}", partial=partial_top, stats=stats)

        df = get_market_data(t, period=period)
        ind = calculate_indicators(df)
        if not _passes_filters(ind, params, stats):
            continue

        price = float(ind.iloc[-1]["Close"])
        if price * 100 > float(budget_yen):
            stats["fail_budget"] += 1
            continue
        stats["budget_ok"] += 1

        row = master[master["ticker"] == t].iloc[0]
        item = {
            "ticker": t,
            "name": str(row.get("name", "")),
            "sector": str(row.get("sector", "")),
            "price": float(price),
            "rsi": float(ind.iloc[-1]["RSI"]),
            "atr_pct": float(ind.iloc[-1]["ATR_PCT"]),
        }
        prelim.append(item)
        partial_top = prelim[:partial_limit]

    prelim = sorted(prelim, key=lambda x: x["atr_pct"], reverse=True)
    return {
        "mode": params.entry_mode,
        "candidates": prelim[: int(top_n)],
        "selected_sectors": selected_sectors,
        "filter_stats": stats,
        "auto_relax_trace": auto_relax_trace,
        "params_effective": dataclasses.asdict(params),
    }
