# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ULTIMATE14
- 根因: Stooq の daily hits limit (Exceeded the daily hits limit / Przekroczony dzienny limit wywolan) で fetch が壊れ、heavy_sims が全滅する
- 対策:
  1) データ取得は stooq -> yfinance のフォールバック（同じ OHLC を作る）
  2) "daily hits limit" を検出したら即座に「stooqブロック中」フラグを立て、残りは yfinance を優先
  3) 取得失敗が多い場合も部分結果で返す（止まらない）
- UI側の診断JSONダウンロード不具合:
  - サイドバーに加えてメインにも download_button を常設
  - diag を「文字列」でも session_state に保存し、ボタンに直接渡す（pickle/大型dict問題回避）
"""

import math
import time
import os
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

_STOOQ_DOMAINS = ["stooq.pl", "stooq.com"]
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-stock-analyzer/ULT14)"}


# --- OHLC disk cache (safe: pickle) ---
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache_ohlc")
os.makedirs(_CACHE_DIR, exist_ok=True)

def _cache_path(ticker: str) -> str:
    safe = str(ticker).replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.pkl")

def _load_cache(ticker: str, max_age_days: int = 2) -> Optional[pd.DataFrame]:
    p = _cache_path(ticker)
    try:
        if not os.path.exists(p):
            return None
        if max_age_days <= 0:
            return None
        age_days = (time.time() - os.path.getmtime(p)) / (24 * 3600)
        if age_days > float(max_age_days):
            return None
        df = pd.read_pickle(p)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    try:
        if df is None or df.empty:
            return
        p = _cache_path(ticker)
        df.to_pickle(p)
    except Exception:
        pass



def get_jpx_master() -> pd.DataFrame:
    """
    JPXが提供するマスター（Excel）を取得。
    できるだけ多くの列を保持し、後段で「市場区分（プライム等）」や「規模区分」があればフィルタできるようにする。
    """
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        df = pd.read_excel(url)
    except Exception:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=20)
            r.raise_for_status()
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

    market_cols = ["市場・商品区分", "市場区分", "市場・商品区分（詳細）", "市場商品区分"]
    found_market = next((c for c in market_cols if c in df.columns), None)
    df["market"] = df[found_market].astype(str) if found_market else np.nan

    size_cols = ["規模区分", "規模", "Size", "規模区分（TOPIX）"]
    found_size = next((c for c in size_cols if c in df.columns), None)
    df["size"] = df[found_size].astype(str) if found_size else np.nan

    keep = ["ticker", "name", "sector", "market", "size"]
    return df[keep].drop_duplicates().reset_index(drop=True)


def _normalize_market(m: str) -> str:
    s = (m or "").strip().lower().replace("　", " ")
    return s

def filter_master(master: pd.DataFrame, market_filter: str = "ALL", size_filter: str = "ALL") -> pd.DataFrame:
    df = master.copy()
    mf = (market_filter or "ALL").upper()
    sf = (size_filter or "ALL").upper()

    if "market" in df.columns and mf != "ALL":
        mk = df["market"].astype(str).apply(_normalize_market)
        if mf == "PRIME":
            df = df[mk.str.contains("プライム|prime", regex=True, na=False)]
        elif mf == "STANDARD":
            df = df[mk.str.contains("スタンダード|standard", regex=True, na=False)]
        elif mf == "GROWTH":
            df = df[mk.str.contains("グロース|growth", regex=True, na=False)]

    if "size" in df.columns and sf != "ALL":
        sz = df["size"].astype(str).apply(_normalize_market)
        if sf == "LARGE":
            df = df[sz.str.contains("大型|large", regex=True, na=False)]
        elif sf == "MID":
            df = df[sz.str.contains("中型|mid", regex=True, na=False)]
        elif sf == "SMALL":
            df = df[sz.str.contains("小型|small", regex=True, na=False)]

    return df.reset_index(drop=True)

def pick_universe_tickers(master_filtered: pd.DataFrame, universe_limit: int, mode: str = "RANDOM_STRATIFIED", seed: int = 0):
    df = master_filtered.copy()
    ticks = df["ticker"].astype(str).tolist()
    if universe_limit <= 0 or len(ticks) <= universe_limit:
        return ticks

    mode = (mode or "RANDOM_STRATIFIED").upper()
    if mode == "HEAD":
        return ticks[: int(universe_limit)]

    rng = np.random.default_rng(int(seed))
    out = []
    total = len(df)
    for sec, g in df.groupby("sector", dropna=False):
        k = max(1, int(round(len(g) / max(1, total) * universe_limit)))
        cand = g["ticker"].astype(str).tolist()
        if len(cand) <= k:
            out.extend(cand)
        else:
            out.extend(list(rng.choice(cand, size=k, replace=False)))
        if len(out) >= universe_limit:
            break

    if len(out) < universe_limit:
        remain = list(set(ticks) - set(out))
        if remain:
            out.extend(list(rng.choice(remain, size=min(len(remain), universe_limit - len(out)), replace=False)))

    if len(out) > universe_limit:
        out = list(rng.choice(out, size=universe_limit, replace=False))

    return out


def _stooq_symbol(ticker: str) -> str:
    t = str(ticker).strip()
    if t.endswith(".T"):
        code = t[:-2]
        if code.isdigit():
            return f"{code}.jp"
    if len(t) == 4 and t.isdigit():
        return f"{t}.jp"
    return t

def _looks_like_rate_limit(text: str) -> bool:
    if not text:
        return False
    s = text.strip().lower()
    return ("exceeded the daily hits limit" in s) or ("przekroczony dzienny limit" in s) or ("limit wywolan" in s)

def _http_get_csv(url: str, timeout: int = 10, retries: int = 2) -> Tuple[Optional[str], dict]:
    backoff = 0.8
    meta: dict = {"url": url}
    for _ in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            text = r.text or ""
            meta.update({"status_code": int(r.status_code), "len": int(len(text))})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff = min(3.0, backoff * 1.7); continue
            if r.status_code != 200:
                return None, meta
            if _looks_like_rate_limit(text):
                meta.update({"note": "stooq_rate_limited", "head": text[:120]})
                return None, meta
            if "Date,Open,High,Low,Close" not in text[:400]:
                meta.update({"note": "non_csv_or_blocked", "head": text[:120]})
                return None, meta
            return text, meta
        except Exception as e:
            meta.update({"exc": f"{type(e).__name__}: {e}"})
            time.sleep(backoff); backoff = min(3.0, backoff * 1.7)
    return None, meta

def fetch_daily_stooq(ticker: str, *, min_rows: int = 260) -> Tuple[Optional[pd.DataFrame], dict]:
    sym = _stooq_symbol(ticker)
    last_meta: dict = {"provider": "stooq", "ticker": ticker, "sym": sym}
    for dom in _STOOQ_DOMAINS:
        url = f"https://{dom}/q/d/l/?s={sym}&i=d"
        text, meta = _http_get_csv(url)
        last_meta.update(meta or {})
        if not text:
            continue
        try:
            df = pd.read_csv(StringIO(text))
            if df is None or df.empty:
                last_meta["note"] = "empty_csv"
                continue
            df.columns = [c.strip().capitalize() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
            if "Volume" not in df.columns:
                df["Volume"] = np.nan
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if len(df) < int(min_rows):
                last_meta.update({"note": "too_short", "rows": int(len(df))})
                return None, last_meta
            df = df.set_index("Date")
            out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            _save_cache(ticker, out)
            return out, last_meta
        except Exception as e:
            last_meta.update({"note": "csv_parse_error", "exc": f"{type(e).__name__}: {e}"})
            return None, last_meta
    return None, last_meta

def fetch_daily_yf(ticker: str, *, min_rows: int = 260) -> Tuple[Optional[pd.DataFrame], dict]:
    meta = {"provider": "yfinance", "ticker": ticker}
    try:
        df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            meta["note"] = "empty"
            return None, meta
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        if "Adj close" in df.columns:
            df = df.rename(columns={"Adj close": "AdjClose"})
        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        df = df.dropna(subset=["Open","High","Low","Close"])
        if len(df) < int(min_rows):
            meta.update({"note": "too_short", "rows": int(len(df))})
            return None, meta
        df.index = pd.to_datetime(df.index)
        out = df[["Open","High","Low","Close","Volume"]].copy()
        _save_cache(ticker, out)
        return out, meta
    except Exception as e:
        meta["exc"] = f"{type(e).__name__}: {e}"
        return None, meta

def fetch_daily(ticker: str, *, min_rows: int = 260, prefer_yf: bool = False, cache_days: int = 2) -> Tuple[Optional[pd.DataFrame], dict, bool]:
    cached = _load_cache(ticker, max_age_days=int(cache_days))
    if cached is not None and not cached.empty:
        return cached, {"provider": "cache", "ticker": ticker, "note": "cache_hit"}, False

    if prefer_yf:
        df, meta = fetch_daily_yf(ticker, min_rows=min_rows)
        if df is not None and not df.empty:
            return df, meta, False

    df, meta = fetch_daily_stooq(ticker, min_rows=min_rows)
    if df is not None and not df.empty:
        return df, meta, False

    rate_limited = (meta or {}).get("note") == "stooq_rate_limited"
    df2, meta2 = fetch_daily_yf(ticker, min_rows=min_rows)
    if df2 is not None and not df2.empty:
        return df2, meta2, rate_limited
    return None, (meta2 or meta), rate_limited

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA25"] = out["Close"].rolling(25).mean()
    out["SMA75"] = out["Close"].rolling(75).mean()
    out["RSI"] = _rsi(out["Close"], 14)
    out["ATR"] = _atr(out, 14)
    out["ATR_PCT"] = (out["ATR"] / (out["Close"] + 1e-12)) * 100.0
    out["RET_3M"] = out["Close"].pct_change(60)
    out["VOL_AVG20"] = out["Volume"].rolling(20).mean()
    return out.dropna()

def calc_sector_strength(latest_df: pd.DataFrame) -> pd.DataFrame:
    g = latest_df.groupby("sector", dropna=False).agg(
        ret_3m=("RET_3M", "mean"),
        rsi=("RSI", "mean"),
        atr_pct=("ATR_PCT", "mean"),
        vol=("VOL_AVG20", "mean"),
        n=("Symbol", "count"),
    ).reset_index()
    g["score"] = (
        (g["ret_3m"].fillna(0) * 100.0) * 0.55
        + (g["rsi"].fillna(50) / 100.0) * 0.20
        + (np.log1p(g["vol"].fillna(0)) / 20.0) * 0.15
        - (g["atr_pct"].fillna(0) / 30.0) * 0.10
    )
    return g.sort_values("score", ascending=False)

def pre_score(latest: pd.Series) -> float:
    trend = 1.0 if float(latest.get("SMA25", 0)) > float(latest.get("SMA75", 0)) else 0.0
    r3m = float(latest.get("RET_3M", 0.0) or 0.0)
    rsi = float(latest.get("RSI", 50.0) or 50.0)
    vol = float(latest.get("VOL_AVG20", 0.0) or 0.0)
    atrp = float(latest.get("ATR_PCT", 0.0) or 0.0)
    return float(
        0.35 * trend
        + 0.35 * r3m
        + 0.15 * (1.0 - abs(rsi - 55.0) / 55.0)
        + 0.10 * (np.log1p(vol) / 20.0)
        - 0.05 * (atrp / 30.0)
    )

@dataclass(frozen=True)
class WFParams:
    atr_mult_stop: float
    tp2_r: float
    time_stop_days: int

def _backtest_r(df_ind: pd.DataFrame, p: WFParams) -> Tuple[float, float, float, int]:
    if df_ind is None or df_ind.empty or len(df_ind) < 120:
        return 0.0, 0.0, 1.0, 0
    df = df_ind.copy()
    trades_r: List[float] = []
    in_pos = False
    entry = stop = tp2 = 0.0
    entry_i = -1
    tp1_hit = False
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if not in_pos:
            if not (float(row["SMA25"]) > float(row["SMA75"])):
                continue
            rsi = float(row["RSI"])
            if not (40.0 <= rsi <= 75.0):
                continue
            if float(row["Close"]) > float(row["SMA25"]) * 1.01:
                continue
            if float(row["Close"]) <= float(prev["High"]):
                continue
            entry = float(row["Close"])
            atr = float(row["ATR"])
            stop = entry - atr * float(p.atr_mult_stop)
            if stop <= 0 or entry <= stop:
                continue
            r_unit = entry - stop
            tp2 = entry + r_unit * float(p.tp2_r)
            in_pos = True
            tp1_hit = False
            entry_i = i
            continue
        low, high, close = float(row["Low"]), float(row["High"]), float(row["Close"])
        r_unit = entry - stop
        held = i - entry_i
        if low <= stop:
            trades_r.append(-1.0 if not tp1_hit else 0.0)
            in_pos = False
            continue
        if high >= tp2:
            r = float(p.tp2_r) if not tp1_hit else (0.5 * 1.0 + 0.5 * float(p.tp2_r))
            trades_r.append(r)
            in_pos = False
            continue
        if (not tp1_hit) and high >= entry + r_unit * 1.0:
            tp1_hit = True
            stop = entry
            continue
        if held >= int(p.time_stop_days):
            r = (close - entry) / (r_unit + 1e-12)
            trades_r.append(float(r))
            in_pos = False
            continue
    if not trades_r:
        return 0.0, 0.0, 1.0, 0
    arr = np.array(trades_r, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float(len(wins) / len(arr))
    exp = float(arr.mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else -1.0
    rr = float(avg_win / (abs(avg_loss) + 1e-12)) if avg_loss < 0 else 2.0
    return exp, win_rate, rr, int(len(arr))

def walk_forward_optimize(df_ind: pd.DataFrame) -> Tuple[WFParams, dict]:
    grid = [WFParams(a, t, d) for a in (1.6, 2.0, 2.4) for t in (2.0, 3.0, 4.0) for d in (7, 10, 14)]
    train, test, step = 500, 120, 120
    if df_ind is None or df_ind.empty or len(df_ind) < (train + test + 20):
        p = WFParams(2.0, 3.0, 10)
        exp, wr, rr, n = _backtest_r(df_ind, p)
        return p, {"wf_oos_mean_exp": exp, "wf_oos_wr": wr, "wf_oos_rr": rr, "wf_oos_trades": n, "note": "short_history_fallback"}
    oos = {p: [] for p in grid}
    for start in range(0, len(df_ind) - (train + test), step):
        train_df = df_ind.iloc[start : start + train]
        test_df = df_ind.iloc[start + train : start + train + test]
        best_p, best_train = None, -1e9
        for p in grid:
            exp, _, _, n = _backtest_r(train_df, p)
            score = exp * min(1.0, n / 10.0)
            if score > best_train:
                best_train, best_p = score, p
        if best_p is None:
            continue
        exp_oos, wr_oos, rr_oos, n_oos = _backtest_r(test_df, best_p)
        oos[best_p].append((exp_oos, wr_oos, rr_oos, n_oos))
    best_p = WFParams(2.0, 3.0, 10)
    best_score, best_sum = -1e9, {}
    for p, vals in oos.items():
        if not vals:
            continue
        exps = np.array([v[0] for v in vals], float)
        wrs = np.array([v[1] for v in vals], float)
        rrs = np.array([v[2] for v in vals], float)
        ns = np.array([v[3] for v in vals], float)
        mean_exp = float(exps.mean())
        mean_wr = float(wrs.mean())
        mean_rr = float(rrs.mean())
        mean_n = float(ns.mean())
        score = mean_exp * min(1.0, mean_n / 8.0)
        if score > best_score:
            best_score = score
            best_p = p
            best_sum = {"wf_oos_mean_exp": mean_exp, "wf_oos_wr": mean_wr, "wf_oos_rr": mean_rr, "wf_oos_trades": mean_n}
    return best_p, best_sum

def monte_carlo_maxdd_from_daily_returns(daily_returns: np.ndarray, n_sim: int = 200) -> float:
    r = np.array(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 80:
        return 0.0
    max_dds = []
    for _ in range(int(n_sim)):
        shuffled = np.random.permutation(r)
        equity = np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dds.append(float(dd.min()))
    return float(np.percentile(max_dds, 5))

def portfolio_ror(win_rate: float, rr: float, capital_units: float) -> float:
    p = float(win_rate); rr = max(1e-6, float(rr))
    edge = p - (1.0 - p) / rr
    if not np.isfinite(edge) or edge <= 0:
        return 1.0
    base = (1.0 - edge) / (1.0 + edge)
    base = min(0.999999, max(1e-9, base))
    return float(base ** float(capital_units))

def dd_factor(current_dd: float) -> float:
    dd = float(current_dd)
    if dd < 0.05: return 1.0
    if dd < 0.10: return 0.7
    if dd < 0.15: return 0.4
    return 0.0

def market_vol_factor(vol_ratio: float) -> float:
    v = float(vol_ratio)
    if v >= 1.6: return 0.55
    if v >= 1.3: return 0.75
    if v <= 0.8: return 1.05
    return 1.0

def compute_market_vol_ratio(progress_cb: Optional[Callable[[str, dict], None]] = None) -> Tuple[float, dict]:
    meta = {"proxy": "1306.T", "ok": False}
    if progress_cb:
        progress_cb("market_vol_fetch", {"proxy": meta["proxy"]})
    df, m, _ = fetch_daily("1306.T", min_rows=260, prefer_yf=False, cache_days=2)
    meta["fetch_meta"] = m
    if df is None or df.empty:
        meta["note"] = "fetch_failed"
        return 1.0, meta
    ind = add_indicators(df)
    if ind.empty:
        meta["note"] = "ind_empty"
        return 1.0, meta
    atrp = ind["ATR_PCT"]
    cur = float(atrp.iloc[-1]); med = float(atrp.tail(260).median())
    if not (np.isfinite(cur) and np.isfinite(med) and med > 0):
        meta["note"] = "atr_nan"
        return 1.0, meta
    meta.update({"ok": True, "current_atr_pct": cur, "median_atr_pct_1y": med})
    return float(cur / med), meta

def correlation_filter(price_dict: Dict[str, pd.Series], symbols: List[str], threshold: float = 0.7) -> List[str]:
    final: List[str] = []
    for sym in symbols:
        if sym not in price_dict:
            continue
        keep = True
        for f in final:
            if f not in price_dict:
                continue
            c = price_dict[sym].pct_change().corr(price_dict[f].pct_change())
            if np.isfinite(c) and abs(float(c)) >= float(threshold):
                keep = False
                break
        if keep:
            final.append(sym)
    return final

def scan_engine(
    *,
    universe_limit: int = 700,
    market_filter: str = "ALL",
    size_filter: str = "ALL",
    universe_mode: str = "RANDOM_STRATIFIED",
    min_price: float = 200.0,
    min_avg_volume: float = 50000.0,
    cache_days: int = 2,
    sector_top_n: int = 6,
    pre_top_m: int = 35,
    top_n: int = 6,
    corr_threshold: float = 0.7,
    max_workers: int = 8,
    time_budget_sec: int = 55,
    progress_cb: Optional[Callable[[str, dict], None]] = None,
) -> Dict[str, object]:
    t0 = time.time()
    diag: dict = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "stage": "init",
        "stats": {},
        "errors": [],
        "sample_failures": [],
        "timings_sec": {},
        "provider_stats": {"stooq_rate_limited": False, "prefer_yfinance": False},
    }

    def tick(stage: str, extra: Optional[dict] = None):
        diag["stage"] = stage
        diag["timings_sec"][stage] = float(time.time() - t0)
        if progress_cb:
            progress_cb(stage, extra or {})

    try:
        master = get_jpx_master()
        if master.empty:
            tick("error", {"reason": "jpx_master_empty"})
            diag["errors"].append("JPXマスター取得失敗")
            return {"ok": False, "error": "JPXマスター取得失敗", "diag": diag, "sector_ranking": pd.DataFrame(), "candidates": pd.DataFrame()}

        tick("universe")
        master_f = filter_master(master, market_filter=market_filter, size_filter=size_filter)
        diag["stats"]["master_total"] = int(len(master))
        diag["stats"]["master_after_filter"] = int(len(master_f))
        seed = int(time.time() // (24*3600))
        tickers = pick_universe_tickers(master_f, int(universe_limit), mode=universe_mode, seed=seed)
        diag["stats"]["universe"] = int(len(tickers))
        diag["stats"]["market_filter"] = str(market_filter)
        diag["stats"]["size_filter"] = str(size_filter)
        diag["stats"]["universe_mode"] = str(universe_mode)
        diag["stats"]["min_price"] = float(min_price)
        diag["stats"]["min_avg_volume"] = float(min_avg_volume)
        diag["stats"]["cache_days"] = int(cache_days)

        tick("fetch")
        rows_latest: List[dict] = []
        price_dict: Dict[str, pd.Series] = {}
        fail_count = 0
        prefer_yf = False

        def _one(t: str, prefer_yf_local: bool):
            df, meta, rl = fetch_daily(t, min_rows=260, prefer_yf=prefer_yf_local, cache_days=int(cache_days))
            return t, df, meta, rl

        futures = []
        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            for t in tickers:
                futures.append(ex.submit(_one, t, prefer_yf))

            done = 0
            for fut in as_completed(futures):
                done += 1
                if (time.time() - t0) > float(time_budget_sec):
                    diag["errors"].append("time_budget_exceeded_fetch_partial")
                    break

                t, df, meta, rl = fut.result()
                if rl:
                    prefer_yf = True
                    diag["provider_stats"]["stooq_rate_limited"] = True
                    diag["provider_stats"]["prefer_yfinance"] = True

                if df is None or df.empty:
                    fail_count += 1
                    if len(diag["sample_failures"]) < 25:
                        diag["sample_failures"].append({"ticker": t, "meta": meta})
                else:
                    ind = add_indicators(df)
                    if ind.empty:
                        fail_count += 1
                        if len(diag["sample_failures"]) < 25:
                            diag["sample_failures"].append({"ticker": t, "meta": {**(meta or {}), "note": "ind_empty"}})
                    else:
                        last = ind.iloc[-1]
                        # Liquidity/price filters: ノイズ銘柄を先に落としてWF/MC精度を守る
                        try:
                            px = float(last.get("Close", np.nan))
                            v20 = float(last.get("VOL_AVG20", np.nan))
                        except Exception:
                            px, v20 = np.nan, np.nan
                        if (np.isfinite(px) and px < float(min_price)) or (np.isfinite(v20) and v20 < float(min_avg_volume)):
                            fail_count += 1
                            if len(diag["sample_failures"]) < 25:
                                diag["sample_failures"].append({"ticker": t, "meta": {**(meta or {}), "note": "filtered_low_liquidity_or_price", "close": px, "vol_avg20": v20}})
                        else:
                            latest = last.to_dict()
                            latest["Symbol"] = t
                            latest["pre_score"] = pre_score(last)
                            rows_latest.append(latest)
                            price_dict[t] = ind["Close"].tail(260)

                if progress_cb and (done % 20 == 0 or done == 1):
                    progress_cb("fetch_progress", {"done": done, "total": len(tickers), "ok": len(rows_latest), "fail": fail_count, "prefer_yf": prefer_yf})

        diag["stats"]["fetched_ok"] = int(len(rows_latest))
        diag["stats"]["fetch_failed"] = int(fail_count)

        if not rows_latest:
            tick("error", {"reason": "all_fetch_failed"})
            diag["errors"].append("データ取得が全滅（Stooq/外部制限）")
            return {"ok": False, "error": "データ取得が全滅", "diag": diag, "sector_ranking": pd.DataFrame(), "candidates": pd.DataFrame()}

        tick("merge")
        latest_df = pd.DataFrame(rows_latest).merge(master, left_on="Symbol", right_on="ticker", how="left")

        tick("sector_strength")
        sec_rank = calc_sector_strength(latest_df[["Symbol", "sector", "RET_3M", "RSI", "ATR_PCT", "VOL_AVG20"]].copy())
        top_sectors = sec_rank.head(int(sector_top_n))["sector"].astype(str).tolist()
        diag["stats"]["top_sectors"] = top_sectors

        tick("preselect")
        cand0 = latest_df[latest_df["sector"].astype(str).isin(top_sectors)].copy()
        cand0 = cand0.sort_values("pre_score", ascending=False).head(int(pre_top_m)).copy()
        diag["stats"]["pre_top_m_actual"] = int(len(cand0))

        tick("heavy_sims")
        heavy_rows = []
        heavy_fail = 0
        syms = cand0["Symbol"].astype(str).tolist()

        for idx, sym in enumerate(syms, start=1):
            if (time.time() - t0) > float(time_budget_sec):
                diag["errors"].append("time_budget_exceeded_heavy_partial")
                break

            df, meta, _ = fetch_daily(sym, min_rows=260, prefer_yf=diag["provider_stats"]["prefer_yfinance"], cache_days=int(cache_days))
            if df is None or df.empty:
                heavy_fail += 1
                continue
            ind = add_indicators(df)
            if ind.empty:
                heavy_fail += 1
                continue

            p_best, wf_sum = walk_forward_optimize(ind)
            daily_ret = ind["Close"].pct_change().dropna().values
            mc_dd = monte_carlo_maxdd_from_daily_returns(daily_ret, n_sim=180)
            exp_full, wr_full, rr_full, n_full = _backtest_r(ind, p_best)

            last = ind.iloc[-1]
            rname = str(cand0[cand0["Symbol"] == sym].iloc[0].get("name", ""))
            rsec = str(cand0[cand0["Symbol"] == sym].iloc[0].get("sector", ""))

            heavy_rows.append({
                "Symbol": sym,
                "name": rname,
                "sector": rsec,
                "Close": float(last["Close"]),
                "ATR": float(last["ATR"]),
                "RSI": float(last["RSI"]),
                "RET_3M": float(last["RET_3M"]),
                "pre_score": float(cand0[cand0["Symbol"] == sym].iloc[0]["pre_score"]),
                "wf_best": {"atr_mult_stop": p_best.atr_mult_stop, "tp2_r": p_best.tp2_r, "time_stop_days": p_best.time_stop_days},
                **wf_sum,
                "mc_dd_p05": float(mc_dd),
                "exp_full": float(exp_full),
                "wr_full": float(wr_full),
                "rr_full": float(rr_full),
                "n_full": int(n_full),
            })

            if progress_cb and (idx % 4 == 0 or idx == 1):
                progress_cb("heavy_progress", {"done": idx, "total": len(syms), "heavy_ok": len(heavy_rows), "heavy_fail": heavy_fail})

        diag["stats"]["heavy_ok"] = int(len(heavy_rows))
        diag["stats"]["heavy_fail"] = int(heavy_fail)

        if not heavy_rows:
            tick("error", {"reason": "heavy_failed"})
            diag["errors"].append("heavy_failed: 最適化対象のデータ取得が不足（Stooq制限/ネット）")
            return {"ok": False, "error": "heavy_sims失敗", "diag": diag, "sector_ranking": sec_rank, "candidates": pd.DataFrame()}

        tick("final_rank")
        heavy_df = pd.DataFrame(heavy_rows)
        sample_pen = np.clip(heavy_df.get("wf_oos_trades", 0).fillna(0).astype(float) / 8.0, 0.2, 1.0)
        heavy_df["final_score"] = (
            heavy_df.get("wf_oos_mean_exp", 0).fillna(0).astype(float) * 150.0 * sample_pen
            + heavy_df.get("wf_oos_wr", 0).fillna(0).astype(float) * 50.0
            + np.clip(heavy_df.get("wf_oos_rr", 1.0).fillna(1.0).astype(float), 0.5, 4.0) * 20.0
            + heavy_df["RET_3M"].fillna(0).astype(float) * 15.0
            + heavy_df["pre_score"].fillna(0).astype(float) * 10.0
            - np.abs(heavy_df["mc_dd_p05"].fillna(0).astype(float)) * 15.0
        )

        ranked = heavy_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        symbols_ranked = ranked["Symbol"].astype(str).tolist()
        symbols_div = correlation_filter(price_dict, symbols_ranked, threshold=float(corr_threshold))
        final_syms = symbols_div[: int(top_n)]
        final_df = ranked[ranked["Symbol"].astype(str).isin(final_syms)].copy()
        final_df = final_df.sort_values("final_score", ascending=False).reset_index(drop=True)

        diag["stats"]["selected"] = final_syms
        tick("done")
        return {"ok": True, "diag": diag, "sector_ranking": sec_rank, "candidates": final_df}

    except Exception as e:
        tick("error", {"reason": "exception"})
        diag["errors"].append(f"exception: {type(e).__name__}: {e}")
        return {"ok": False, "error": f"例外: {type(e).__name__}", "diag": diag, "sector_ranking": pd.DataFrame(), "candidates": pd.DataFrame()}

def build_orders(
    candidates: pd.DataFrame,
    *,
    capital_yen: int,
    risk_pct_per_trade: float,
    current_dd: float,
    vol_ratio: float,
) -> Dict[str, object]:
    if candidates is None or candidates.empty:
        return {"orders": pd.DataFrame(), "portfolio": {}, "ror": 1.0}

    df = candidates.copy()
    w = df.get("n_full", 0).fillna(0).astype(float).values
    w = np.maximum(1.0, w)
    wr = float(np.average(df.get("wr_full", 0.0).fillna(0.0).astype(float).values, weights=w))
    rr = float(np.average(df.get("rr_full", 1.0).fillna(1.0).astype(float).values, weights=w))
    rr = max(0.5, rr)

    units = float(1.0 / max(1e-6, risk_pct_per_trade))
    ror = portfolio_ror(wr, rr, units)

    ror_factor = 1.0 if ror <= 0.02 else (0.8 if ror <= 0.05 else (0.6 if ror <= 0.10 else 0.3))
    ddf = dd_factor(current_dd)
    mvf = market_vol_factor(vol_ratio)
    base_risk_budget = float(capital_yen) * float(risk_pct_per_trade) * float(ddf) * float(mvf) * float(ror_factor)

    orders = []
    for _, row in df.iterrows():
        price = float(row["Close"])
        atr = float(row["ATR"])
        p = row.get("wf_best", {}) or {}
        atr_mult = float(p.get("atr_mult_stop", 2.0))
        tp2_r = float(p.get("tp2_r", 3.0))

        stop = price - atr * atr_mult
        r_unit = price - stop

        if r_unit <= 0:
            shares = 0
        else:
            risk_budget = base_risk_budget / max(1.0, len(df))
            shares = int(math.floor(risk_budget / r_unit))
            shares = int((shares // 100) * 100)
            shares = max(0, shares)

        orders.append({
            "Symbol": str(row["Symbol"]),
            "name": str(row.get("name", "")),
            "sector": str(row.get("sector", "")),
            "entry": price,
            "stop": stop,
            "tp1": price + r_unit * 1.0,
            "tp2": price + r_unit * tp2_r,
            "shares": shares,
            "atr_mult": atr_mult,
            "tp2_r": tp2_r,
            "note": "" if shares > 0 else "shares=0（単元/リスク不足）",
        })

    portfolio = {
        "win_rate_est": wr,
        "rr_est": rr,
        "capital_units": units,
        "ror": ror,
        "dd_factor": ddf,
        "market_vol_ratio": float(vol_ratio),
        "market_vol_factor": mvf,
        "ror_factor": ror_factor,
        "risk_budget_total_yen": base_risk_budget,
    }
    return {"orders": pd.DataFrame(orders), "portfolio": portfolio, "ror": ror}
