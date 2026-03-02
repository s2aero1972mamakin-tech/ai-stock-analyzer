# -*- coding: utf-8 -*-
# logic.py — JPX Sector-First Swing Trader (Robust Fetch + Diagnostics)

from __future__ import annotations

import os
import time
import hashlib
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

STOOQ_URLS = [
    "https://stooq.pl/q/d/l/?s={}&i=d",
    "https://stooq.com/q/d/l/?s={}&i=d",
]

MAX_WORKERS = 8
BASE_DELAY = 0.18
MAX_DELAY = 1.2
CACHE_DIR = ".cache_stooq"
CACHE_TTL_SEC = 20 * 60 * 60
TAIL_DAYS = 520

SECTOR_REP_PER_SECTOR = 6
TOP_SECTORS = 3
MAX_TICKERS_PER_SECTOR = 180

DEFAULT_LOT = 100
ALLOW_SINGLE_SHARE_FALLBACK = True

HTTP_TIMEOUT = 20
HTTP_RETRIES = 3
UA = "Mozilla/5.0 (compatible; SectorFirstBot/1.0)"

_lock = threading.Lock()
_last_request = 0.0
_dynamic_delay = BASE_DELAY

_diag = {
    "fetch_calls": 0,
    "fetch_ok": 0,
    "fetch_empty": 0,
    "http_status_counts": {},
    "empty_tickers_sample": [],
    "blocked_suspected": 0,
}

def _diag_inc_status(code: int):
    k = str(code)
    _diag["http_status_counts"][k] = _diag["http_status_counts"].get(k, 0) + 1

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
        return (time.time() - os.stat(path).st_mtime) <= CACHE_TTL_SEC
    except Exception:
        return False

def _to_stooq_symbol(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith(".T"):
        t = t[:-2]
    return f"{t}.jp"

def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    global _dynamic_delay
    _diag["fetch_calls"] += 1

    cpath = _cache_path(ticker)
    if _cache_is_fresh(cpath):
        try:
            df = pd.read_parquet(cpath)
            if df.empty:
                _diag["fetch_empty"] += 1
            else:
                _diag["fetch_ok"] += 1
            return df
        except Exception:
            pass

    sym = _to_stooq_symbol(ticker)
    headers = {"User-Agent": UA, "Accept": "text/csv,*/*;q=0.9"}

    for base in STOOQ_URLS:
        url = base.format(sym)
        for attempt in range(1, HTTP_RETRIES + 1):
            try:
                _throttle()
                r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
                _diag_inc_status(r.status_code)

                if r.status_code in (403, 429):
                    _diag["blocked_suspected"] += 1
                    with _lock:
                        _dynamic_delay = min(MAX_DELAY, _dynamic_delay * 1.45 + 0.04)
                    time.sleep(min(2.0, 0.6 * attempt))
                    continue

                if r.status_code in (502, 503, 504):
                    with _lock:
                        _dynamic_delay = min(MAX_DELAY, _dynamic_delay * 1.25 + 0.03)
                    time.sleep(min(1.8, 0.5 * attempt))
                    continue

                if r.status_code != 200:
                    time.sleep(min(1.0, 0.4 * attempt))
                    continue

                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                if df.empty:
                    time.sleep(min(0.8, 0.3 * attempt))
                    continue

                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index().tail(TAIL_DAYS)

                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in df.columns:
                        df[col] = np.nan

                try:
                    df.to_parquet(cpath, index=True)
                except Exception:
                    pass

                with _lock:
                    _dynamic_delay = max(BASE_DELAY, _dynamic_delay * 0.95)

                _diag["fetch_ok"] += 1
                return df

            except Exception:
                with _lock:
                    _dynamic_delay = min(MAX_DELAY, _dynamic_delay * 1.25 + 0.03)
                time.sleep(min(1.8, 0.5 * attempt))
                continue

    _diag["fetch_empty"] += 1
    if len(_diag["empty_tickers_sample"]) < 30:
        _diag["empty_tickers_sample"].append(ticker)
    return pd.DataFrame()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / (dn.ewm(alpha=1/n, adjust=False).mean() + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(high-low).abs(), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def pick_sector_representatives(universe: pd.DataFrame, per_sector: int) -> Dict[str, List[str]]:
    return {sec: g["ticker"].head(per_sector).tolist() for sec, g in universe.groupby("sector")}

def sector_strength_score(rep_dfs: List[pd.DataFrame]) -> float:
    vals = []
    for df in rep_dfs:
        if df is None or df.empty or df["Close"].dropna().shape[0] < 80:
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
    return float(np.mean(vals)) if vals else -1e9

def rank_sectors(universe: pd.DataFrame, *, top_k: int, reps_per_sector: int) -> Tuple[List[str], Dict[str, float]]:
    reps = pick_sector_representatives(universe, reps_per_sector)
    all_rep = sorted({t for ts in reps.values() for t in ts})
    rep_map: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 12)) as ex:
        futs = {ex.submit(fetch_ohlcv, t): t for t in all_rep}
        for f in as_completed(futs):
            t = futs[f]
            rep_map[t] = f.result()
    sec_scores = {sec: sector_strength_score([rep_map.get(t, pd.DataFrame()) for t in ts]) for sec, ts in reps.items()}
    ranked = sorted(sec_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [sec for sec, _ in ranked[:top_k]], sec_scores

@dataclass
class EvalResult:
    ticker: str
    sector: str
    price: float
    atr: float
    rsi: float
    mom20: float
    win_rate_est: float
    expectancy_r: float
    annual_r_est: float
    score: float

def _estimate_high_tp_model(df: pd.DataFrame) -> Tuple[float, float, float]:
    if df.empty or df["Close"].dropna().shape[0] < 200:
        return 0.0, -9.0, -9.0
    close = df["Close"].astype(float)
    df = df.copy()
    df["RSI"] = _rsi(close, 14)
    df["ATR"] = _atr(df, 14)
    trades = []
    for i in range(80, len(df)-5):
        rsi_v = df["RSI"].iloc[i]
        if not (30 < rsi_v < 42):
            continue
        entry = close.iloc[i]
        atr_v = df["ATR"].iloc[i]
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue
        stop = entry - atr_v
        risk = entry - stop
        tpR = 0.30
        target = entry + tpR * risk
        hit = None
        for j in range(i+1, i+4):
            if df["High"].iloc[j] >= target:
                hit = tpR
                break
            if df["Low"].iloc[j] <= stop:
                hit = -1.0
                break
        if hit is not None:
            trades.append(hit)
    if not trades:
        return 0.0, -1.0, -1.0
    tr = np.array(trades, dtype=float)
    win = float((tr > 0).mean())
    exp = float(tr.mean())
    annual = exp * min(260.0, max(20.0, len(tr) * 2.0))
    return win, exp, float(annual)

def evaluate_ticker(ticker: str, sector: str, capital_yen: float, lot: int) -> Optional[EvalResult]:
    df = fetch_ohlcv(ticker)
    if df.empty or df["Close"].dropna().shape[0] < 220:
        return None
    price = float(df["Close"].iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return None
    if (price * lot) > capital_yen and price > capital_yen:
        return None
    close = df["Close"].astype(float)
    atr = float(_atr(df, 14).iloc[-1])
    rsi = float(_rsi(close, 14).iloc[-1])
    mom20 = float(close.pct_change(20).iloc[-1])
    win, exp, annual = _estimate_high_tp_model(df)
    score = (1.25 * annual) + (0.40 * win) + (0.20 * mom20) - (0.10 * (atr / price))
    return EvalResult(ticker, sector, price, atr, rsi, mom20, win, exp, annual, float(score))

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
    _diag["fetch_calls"] = _diag["fetch_ok"] = _diag["fetch_empty"] = _diag["blocked_suspected"] = 0
    _diag["http_status_counts"] = {}
    _diag["empty_tickers_sample"] = []

    chosen_sectors, sector_scores = rank_sectors(universe, top_k=int(top_sectors), reps_per_sector=int(SECTOR_REP_PER_SECTOR))
    all_failed = all(v <= -1e8 for v in sector_scores.values()) if sector_scores else True

    results: List[EvalResult] = []
    if not all_failed:
        cand_df = universe[universe["sector"].isin(chosen_sectors)].copy()
        cand_list: List[Tuple[str,str]] = []
        for sec, g in cand_df.groupby("sector"):
            for t in g["ticker"].head(int(max_per_sector)).tolist():
                cand_list.append((t, sec))

        def _proc(item):
            t, sec = item
            return evaluate_ticker(t, sec, float(capital_yen), int(lot))

        with ThreadPoolExecutor(max_workers=int(MAX_WORKERS)) as ex:
            futs = [ex.submit(_proc, it) for it in cand_list]
            for f in as_completed(futs):
                r = f.result()
                if r is not None and np.isfinite(r.score):
                    results.append(r)

        results.sort(key=lambda x: x.score, reverse=True)

    meta = {
        "engine": "JPX Sector-First Swing Trader (Robust Fetch)",
        "capital_yen": float(capital_yen),
        "top_sectors": int(top_sectors),
        "max_per_sector": int(max_per_sector),
        "lot": int(lot),
        "max_positions": int(max_positions),
        "dynamic_delay_sec": float(_dynamic_delay),
        "elapsed_sec": float(time.time() - t0),
        "diagnostics": {
            "fetch_calls": int(_diag["fetch_calls"]),
            "fetch_ok": int(_diag["fetch_ok"]),
            "fetch_empty": int(_diag["fetch_empty"]),
            "fetch_success_rate": float((_diag["fetch_ok"] / max(1, _diag["fetch_calls"]))),
            "http_status_counts": dict(_diag["http_status_counts"]),
            "blocked_suspected": int(_diag["blocked_suspected"]),
            "empty_tickers_sample": list(_diag["empty_tickers_sample"]),
            "global_fetch_failed": bool(all_failed),
        }
    }

    return {
        "meta": meta,
        "sector_scores": {k: float(v) for k, v in sector_scores.items()},
        "chosen_sectors": chosen_sectors,
        "ranked": [{
            "ticker": r.ticker,
            "sector": r.sector,
            "price": r.price,
            "atr": r.atr,
            "rsi": r.rsi,
            "mom20": r.mom20,
            "win_rate_est": r.win_rate_est,
            "expectancy_r": r.expectancy_r,
            "annual_r_est": r.annual_r_est,
            "score": r.score,
        } for r in results[:200]],
        "portfolio": [],
    }
