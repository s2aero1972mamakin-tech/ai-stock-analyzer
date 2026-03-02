# -*- coding: utf-8 -*-
# ================================================================
# logic.py — JPX Sector-First Swing Trader (STOOQ)
# 目的:
#   - 4000銘柄を毎回スキャンしない
#   - まずセクター強弱を少数サンプルで推定 → 上位セクターだけを深掘り
#   - 並列 + 自動スロットリング + ディスクキャッシュで安定運用
#   - ランキング + 推奨ポートフォリオ（シャープ最大化近似 / 相関抑制）
#
# 注意:
#   - データは Stooq 日足（遅延/欠損あり）
#   - 売買執行は行わない（銘柄候補と配分案を出す）
# ================================================================

from __future__ import annotations

import os
import time
import json
import math
import hashlib
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================
# Config
# =============================

STOOQ_URL = "https://stooq.pl/q/d/l/?s={}&i=d"

MAX_WORKERS = 8
BASE_DELAY = 0.12                 # 初期スロットリング
MAX_DELAY = 0.8                   # エラー多発時に増える
CACHE_DIR = ".cache_stooq"
CACHE_TTL_SEC = 20 * 60 * 60      # 20時間
TAIL_DAYS = 420                   # 1.6年程度

SECTOR_REP_PER_SECTOR = 6         # セクター代表サンプル数
TOP_SECTORS = 3                   # 上位セクター数
MAX_TICKERS_PER_SECTOR = 180      # 深掘り上限（全銘柄スキャン回避）

DEFAULT_LOT = 100
ALLOW_SINGLE_SHARE_FALLBACK = True


# =============================
# Throttle / Cache
# =============================

_lock = threading.Lock()
_last_request = 0.0
_dynamic_delay = BASE_DELAY


def _throttle():
    global _last_request, _dynamic_delay
    with _lock:
        now = time.time()
        wait = _dynamic_delay - (now - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()


def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(ticker.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{key}.parquet")


def _cache_is_fresh(path: str) -> bool:
    try:
        st = os.stat(path)
        return (time.time() - st.st_mtime) <= CACHE_TTL_SEC
    except Exception:
        return False


def _to_stooq_symbol(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith(".T"):
        t = t[:-2]
    return f"{t}.jp"


def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    # Stooq日足を取得（ディスクキャッシュ有）
    global _dynamic_delay
    cpath = _cache_path(ticker)
    if _cache_is_fresh(cpath):
        try:
            return pd.read_parquet(cpath)
        except Exception:
            pass

    sym = _to_stooq_symbol(ticker)
    url = STOOQ_URL.format(sym)

    try:
        _throttle()
        r = requests.get(url, timeout=20)
        if r.status_code in (429, 403, 502, 503, 504):
            with _lock:
                _dynamic_delay = min(MAX_DELAY, _dynamic_delay * 1.35 + 0.02)
            return pd.DataFrame()
        if r.status_code != 200:
            return pd.DataFrame()

        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df.tail(TAIL_DAYS)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan

        try:
            df.to_parquet(cpath, index=True)
        except Exception:
            pass

        with _lock:
            _dynamic_delay = max(BASE_DELAY, _dynamic_delay * 0.97)

        return df

    except Exception:
        with _lock:
            _dynamic_delay = min(MAX_DELAY, _dynamic_delay * 1.25 + 0.03)
        return pd.DataFrame()


# =============================
# Indicators
# =============================

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / (dn.ewm(alpha=1/n, adjust=False).mean() + 1e-12)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def _safe_last(x: pd.Series) -> float:
    if x is None or len(x) == 0:
        return float("nan")
    return float(x.iloc[-1])


# =============================
# Sector-first pipeline
# =============================

def pick_sector_representatives(universe: pd.DataFrame, per_sector: int = SECTOR_REP_PER_SECTOR) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for sec, g in universe.groupby("sector"):
        out[sec] = g["ticker"].head(per_sector).tolist()
    return out


def sector_strength_score(rep_dfs: List[pd.DataFrame]) -> float:
    vals = []
    for df in rep_dfs:
        if df is None or df.empty or df["Close"].dropna().shape[0] < 60:
            continue
        close = df["Close"].astype(float)

        ret20 = close.pct_change(20).iloc[-1]
        ret5 = close.pct_change(5).iloc[-1]
        vol20 = close.pct_change().rolling(20).std().iloc[-1]

        vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
        vr = (vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1e-12)) if vol.dropna().shape[0] >= 25 else 1.0
        vscore = float(np.log(max(0.2, min(6.0, vr))))

        s = 0.55*ret20 + 0.30*ret5 - 0.35*vol20 + 0.10*vscore
        if np.isfinite(s):
            vals.append(float(s))

    if not vals:
        return -1e9
    return float(np.mean(vals))


def rank_sectors(universe: pd.DataFrame, top_k: int = TOP_SECTORS) -> Tuple[List[str], Dict[str, float]]:
    reps = pick_sector_representatives(universe, SECTOR_REP_PER_SECTOR)

    all_rep = sorted({t for ts in reps.values() for t in ts})
    rep_map: Dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 10)) as ex:
        futs = {ex.submit(fetch_ohlcv, t): t for t in all_rep}
        for f in as_completed(futs):
            t = futs[f]
            rep_map[t] = f.result()

    sec_scores: Dict[str, float] = {}
    for sec, ts in reps.items():
        rep_dfs = [rep_map.get(t, pd.DataFrame()) for t in ts]
        sec_scores[sec] = sector_strength_score(rep_dfs)

    ranked = sorted(sec_scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [sec for sec, _ in ranked[:top_k]]
    return top, sec_scores


# =============================
# Candidate evaluation (profit precision focus)
# =============================

@dataclass
class EvalResult:
    ticker: str
    sector: str
    price: float
    atr: float
    rsi: float
    vol_ratio: float
    mom5: float
    mom20: float
    win_rate_est: float
    expectancy_r: float
    annual_r_est: float
    score: float


def _estimate_high_tp_model(df: pd.DataFrame) -> Tuple[float, float, float]:
    if df.empty or df["Close"].dropna().shape[0] < 160:
        return 0.0, -9.0, -9.0

    close = df["Close"].astype(float)
    df = df.copy()
    df["RSI"] = _rsi(close, 14)
    df["ATR"] = _atr(df, 14)

    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
    vr = vol / (vol.rolling(20).mean() + 1e-12)

    trades: List[float] = []
    for i in range(60, len(df) - 5):
        rsi_v = df["RSI"].iloc[i]
        if not (30.0 < rsi_v < 42.0):
            continue

        if np.isfinite(vr.iloc[i]) and vr.iloc[i] < 1.15:
            continue

        vol20 = close.pct_change().rolling(20).std().iloc[i]
        vol100m = close.pct_change().rolling(100).std().mean()
        if np.isfinite(vol20) and np.isfinite(vol100m) and vol20 > 1.35 * vol100m:
            continue

        entry = close.iloc[i]
        atr_v = df["ATR"].iloc[i]
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue

        stop = entry - atr_v
        risk = entry - stop
        mom = close.pct_change(3).iloc[i]
        tpR = 0.30 if (not np.isfinite(mom) or mom < 0) else 0.50
        target = entry + tpR * risk

        hit = None
        for j in range(i + 1, i + 4):
            if df["High"].iloc[j] >= target:
                hit = tpR
                break
            if df["Low"].iloc[j] <= stop:
                hit = -1.0
                break
        if hit is not None:
            trades.append(float(hit))

    if not trades:
        return 0.0, -1.0, -1.0

    tr = np.array(trades, dtype=float)
    win_rate = float((tr > 0).mean())
    expectancy = float(tr.mean())
    trades_per_year = min(260.0, max(20.0, len(tr) * 2.0))
    annual_r = expectancy * trades_per_year
    return win_rate, expectancy, float(annual_r)


def evaluate_ticker(ticker: str, sector: str, capital_yen: float, lot: int = DEFAULT_LOT) -> Optional[EvalResult]:
    df = fetch_ohlcv(ticker)
    if df.empty or df["Close"].dropna().shape[0] < 200:
        return None

    price = float(df["Close"].iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return None

    can_lot = (price * lot) <= capital_yen
    can_1 = price <= capital_yen

    if not can_lot and not (ALLOW_SINGLE_SHARE_FALLBACK and can_1):
        return None

    close = df["Close"].astype(float)
    atr = _safe_last(_atr(df, 14))
    rsi = _safe_last(_rsi(close, 14))

    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
    vol_ratio = float((vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1e-12))) if vol.dropna().shape[0] >= 25 else 1.0

    mom5 = float(close.pct_change(5).iloc[-1])
    mom20 = float(close.pct_change(20).iloc[-1])

    win_rate, expectancy_r, annual_r = _estimate_high_tp_model(df)

    score = (1.25 * annual_r) + (0.40 * win_rate) + (0.20 * mom20) - (0.10 * (atr / price))
    return EvalResult(
        ticker=ticker,
        sector=sector,
        price=price,
        atr=atr,
        rsi=rsi,
        vol_ratio=vol_ratio,
        mom5=mom5,
        mom20=mom20,
        win_rate_est=win_rate,
        expectancy_r=expectancy_r,
        annual_r_est=annual_r,
        score=float(score),
    )


# =============================
# Portfolio construction
# =============================

def _max_sharpe_weights(returns_df: pd.DataFrame, ridge: float = 1e-6) -> np.ndarray:
    mu = returns_df.mean().values
    cov = returns_df.cov().values + np.eye(returns_df.shape[1]) * ridge
    inv = np.linalg.pinv(cov)
    w = inv.dot(mu)
    w = np.maximum(w, 0.0)
    if w.sum() < 1e-12:
        w = np.ones_like(w)
    return w / w.sum()


def build_portfolio(
    ranked: List[EvalResult],
    capital_yen: float,
    max_positions: int = 4,
    lot: int = DEFAULT_LOT,
    corr_max: float = 0.65,
) -> List[dict]:
    picks: List[EvalResult] = []
    returns_map: Dict[str, pd.Series] = {}

    # まず上位からリターン列を集める（最大25本）
    for r in ranked[: min(len(ranked), 40)]:
        df = fetch_ohlcv(r.ticker)
        if df.empty:
            continue
        rets = df["Close"].astype(float).pct_change().dropna()
        if rets.shape[0] < 180:
            continue
        returns_map[r.ticker] = rets
        if len(returns_map) >= 25:
            break

    if not returns_map:
        return []

    ret_df = pd.DataFrame(returns_map).dropna()
    if ret_df.empty:
        return []

    corr = ret_df.corr()

    # 相関を見ながら採用
    for r in ranked:
        if r.ticker not in ret_df.columns:
            continue
        ok = True
        for p in picks:
            if corr.loc[r.ticker, p.ticker] > corr_max:
                ok = False
                break
        if ok:
            picks.append(r)
        if len(picks) >= max_positions:
            break

    if not picks:
        return []

    sel_df = ret_df[[p.ticker for p in picks]].dropna()
    if sel_df.empty:
        return []

    w = _max_sharpe_weights(sel_df)
    weights = {picks[i].ticker: float(w[i]) for i in range(len(picks))}

    portfolio: List[dict] = []
    cash = float(capital_yen)

    ordered = sorted(picks, key=lambda x: weights.get(x.ticker, 0.0), reverse=True)
    for p in ordered:
        wgt = weights.get(p.ticker, 0.0)
        if wgt <= 0:
            continue
        budget = cash * min(0.98, max(0.05, wgt))
        price = p.price
        if price <= 0:
            continue

        shares = int(budget // (price * lot)) * lot
        if shares <= 0 and ALLOW_SINGLE_SHARE_FALLBACK:
            shares = int(budget // price)

        if shares <= 0:
            continue

        cost = shares * price
        if cost > cash:
            continue

        stop = price - max(1e-9, p.atr)
        tpR = 0.50 if (p.mom5 > 0) else 0.30
        target = price + tpR * (price - stop)

        portfolio.append({
            "ticker": p.ticker,
            "sector": p.sector,
            "shares": int(shares),
            "entry": float(price),
            "stop": float(stop),
            "target": float(target),
            "weight_hint": float(wgt),
            "win_rate_est": float(p.win_rate_est),
            "annual_r_est": float(p.annual_r_est),
        })
        cash -= cost
        if len(portfolio) >= max_positions:
            break

    return portfolio


# =============================
# Public API
# =============================

def scan_sector_first(
    universe: pd.DataFrame,
    *,
    capital_yen: float = 300_000.0,
    top_sectors: int = TOP_SECTORS,
    max_per_sector: int = MAX_TICKERS_PER_SECTOR,
    lot: int = DEFAULT_LOT,
    max_positions: int = 4,
) -> dict:
    t0 = time.time()

    chosen_sectors, sector_scores = rank_sectors(universe, top_k=top_sectors)

    cand_df = universe[universe["sector"].isin(chosen_sectors)].copy()
    cand_list: List[Tuple[str, str]] = []
    for sec, g in cand_df.groupby("sector"):
        for t in g["ticker"].head(max_per_sector).tolist():
            cand_list.append((t, sec))

    results: List[EvalResult] = []

    def _proc(item):
        t, sec = item
        return evaluate_ticker(t, sec, float(capital_yen), lot=int(lot))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_proc, it) for it in cand_list]
        for f in as_completed(futs):
            r = f.result()
            if r is not None and np.isfinite(r.score):
                results.append(r)

    results.sort(key=lambda x: x.score, reverse=True)
    portfolio = build_portfolio(results, float(capital_yen), max_positions=int(max_positions), lot=int(lot))

    return {
        "meta": {
            "engine": "JPX Sector-First Swing Trader (Stooq)",
            "capital_yen": float(capital_yen),
            "top_sectors": int(top_sectors),
            "max_per_sector": int(max_per_sector),
            "lot": int(lot),
            "max_positions": int(max_positions),
            "dynamic_delay_sec": float(_dynamic_delay),
            "elapsed_sec": float(time.time() - t0),
        },
        "sector_scores": {k: float(v) for k, v in sector_scores.items()},
        "chosen_sectors": chosen_sectors,
        "ranked": [
            {
                "ticker": r.ticker,
                "sector": r.sector,
                "price": r.price,
                "atr": r.atr,
                "rsi": r.rsi,
                "vol_ratio": r.vol_ratio,
                "mom5": r.mom5,
                "mom20": r.mom20,
                "win_rate_est": r.win_rate_est,
                "expectancy_r": r.expectancy_r,
                "annual_r_est": r.annual_r_est,
                "score": r.score,
            }
            for r in results[:200]
        ],
        "portfolio": portfolio,
    }
