
# -*- coding: utf-8 -*-
"""
SBI 半自動プロ仕様 — 機関レベルコア（Resilient + Diagnostics）

主目的:
- Streamlit Cloud等でも「データ取得失敗で止まらない」
- 失敗要因を診断JSONで可視化してダウンロード可能
- B運用: 上位候補だけに Walk-Forward / MonteCarlo を適用
- ポートフォリオ単位 Risk of Ruin（RoR）
- 市場ボラ連動ロット、DD制御、相関除外、セクター分散

データ:
- JPXマスター: JPX公式Excel
- 株価: Stooq 日足CSV（stooq.pl / stooq.com）
"""

from __future__ import annotations

import math
import time
import json
import traceback
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor


# -----------------------------
# Settings
# -----------------------------
_STOOQ_DOMAINS = ["stooq.pl", "stooq.com"]
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-stock-analyzer/2.0)"}


# -----------------------------
# JPX master
# -----------------------------
def get_jpx_master() -> pd.DataFrame:
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        df = pd.read_excel(url)
    except Exception:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30)
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
    return df[["ticker", "name", "sector"]].drop_duplicates().reset_index(drop=True)


# -----------------------------
# Stooq fetch (single)
# -----------------------------
def _stooq_symbol(ticker: str) -> str:
    t = str(ticker).strip()
    if t.endswith(".T"):
        code = t[:-2]
        if code.isdigit():
            return f"{code}.jp"
    if len(t) == 4 and t.isdigit():
        return f"{t}.jp"
    return t


def _http_get_csv(url: str, timeout: int = 20, retries: int = 3) -> Tuple[Optional[str], Optional[dict]]:
    backoff = 1.0
    last_meta = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            last_meta = {"status_code": r.status_code, "len": len(r.text or ""), "url": url}
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(8.0, backoff * 2)
                continue
            if r.status_code != 200:
                return None, last_meta
            text = r.text or ""
            # validate it's a CSV with OHLC header
            if "Date,Open,High,Low,Close" not in text[:300]:
                return None, {**(last_meta or {}), "note": "non_csv_or_blocked", "head": text[:120]}
            return text, last_meta
        except Exception as e:
            last_meta = {**(last_meta or {}), "exc": f"{type(e).__name__}: {e}"}
            time.sleep(backoff)
            backoff = min(8.0, backoff * 2)
    return None, last_meta


def fetch_daily_stooq(ticker: str, *, min_rows: int = 260) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    sym = _stooq_symbol(ticker)
    last_meta = None
    for dom in _STOOQ_DOMAINS:
        url = f"https://{dom}/q/d/l/?s={sym}&i=d"
        text, meta = _http_get_csv(url)
        last_meta = meta
        if not text:
            continue
        try:
            df = pd.read_csv(StringIO(text))
            if df is None or df.empty:
                continue
            df.columns = [c.strip().capitalize() for c in df.columns]
            if "Date" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            # ensure required cols exist
            for c in ["Open", "High", "Low", "Close"]:
                if c not in df.columns:
                    return None, {**(last_meta or {}), "note": f"missing_col_{c}"}
            if "Volume" not in df.columns:
                df["Volume"] = np.nan
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if len(df) < int(min_rows):
                return None, {**(last_meta or {}), "note": "too_short", "rows": int(len(df))}
            df = df.set_index("Date")
            return df[["Open", "High", "Low", "Close", "Volume"]].copy(), last_meta
        except Exception as e:
            return None, {**(last_meta or {}), "note": "csv_parse_error", "exc": f"{type(e).__name__}: {e}"}
    return None, last_meta


# -----------------------------
# Indicators
# -----------------------------
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
    out["TURNOVER"] = out["Close"] * out["Volume"]
    out["TURNOVER_AVG20"] = out["TURNOVER"].rolling(20).mean()
    return out.dropna()


# -----------------------------
# Sector strength
# -----------------------------
def calc_sector_strength(latest_df: pd.DataFrame) -> pd.DataFrame:
    # latest_df must include: sector, RET_3M, RSI, ATR_PCT, VOL_AVG20
    g = latest_df.groupby("sector", dropna=False).agg(
        ret_3m=("RET_3M", "mean"),
        rsi=("RSI", "mean"),
        atr_pct=("ATR_PCT", "mean"),
        vol=("VOL_AVG20", "mean"),
        n=("Symbol", "count"),
    ).reset_index()
    # score: return and trendiness, penalize excessive vol (atr_pct too high)
    g["score"] = (
        (g["ret_3m"].fillna(0) * 100.0) * 0.55
        + (g["rsi"].fillna(50) / 100.0) * 0.20
        + (np.log1p(g["vol"].fillna(0)) / 20.0) * 0.15
        - (g["atr_pct"].fillna(0) / 30.0) * 0.10
    )
    return g.sort_values("score", ascending=False)


# -----------------------------
# Pre-score (cheap ranking before heavy sims)
# -----------------------------
def pre_score(latest: pd.Series) -> float:
    # trend: SMA25>SMA75
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


# -----------------------------
# Walk-forward optimization (lightweight)
# -----------------------------
@dataclass(frozen=True)
class WFParams:
    atr_mult_stop: float
    tp2_r: float
    time_stop_days: int


def _backtest_r(df_ind: pd.DataFrame, p: WFParams) -> Tuple[float, float, float, int]:
    """
    Very compact swing backtest to produce:
    - expectancy (mean R)
    - win_rate
    - rr (avg_win / abs(avg_loss)) approximate
    - n_trades
    Rules:
    - enter on close when SMA25>SMA75 and RSI in [40,75] and pullback-ish (close below SMA25 slightly) then close breaks prev high
    - stop = entry - ATR*atr_mult
    - tp2 = entry + R*tp2_r, partial at +1R (optional), time stop
    """
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
            # mild pullback condition: close <= SMA25 * 1.01
            if float(row["Close"]) > float(row["SMA25"]) * 1.01:
                continue
            # trigger: close breaks prev high
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
            # if tp1 already hit we assume breakeven on rest
            r = -1.0 if not tp1_hit else 0.0
            trades_r.append(r)
            in_pos = False
            continue

        if high >= tp2:
            r = float(p.tp2_r) if not tp1_hit else (0.5 * 1.0 + 0.5 * float(p.tp2_r))
            trades_r.append(r)
            in_pos = False
            continue

        if (not tp1_hit) and high >= entry + r_unit * 1.0:
            tp1_hit = True
            stop = entry  # move to breakeven
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
    """
    Walk-forward optimize on small grid and return best param and out-of-sample score summary.
    B運用: 上位候補だけに適用する前提。
    """
    # small grid (fast)
    grid = [
        WFParams(atr_mult_stop=a, tp2_r=t, time_stop_days=d)
        for a in (1.6, 2.0, 2.4)
        for t in (2.0, 3.0, 4.0)
        for d in (7, 10, 14)
    ]

    # rolling scheme: train 2y (~500d), test 6m (~120d)
    train = 500
    test = 120
    step = 120

    if df_ind is None or df_ind.empty or len(df_ind) < (train + test + 20):
        # fallback: choose a sane default
        p = WFParams(2.0, 3.0, 10)
        exp, wr, rr, n = _backtest_r(df_ind, p)
        return p, {"wf_oos_mean_exp": exp, "wf_oos_wr": wr, "wf_oos_rr": rr, "wf_oos_trades": n, "note": "short_history_fallback"}

    oos_scores = {p: [] for p in grid}

    for start in range(0, len(df_ind) - (train + test), step):
        train_df = df_ind.iloc[start : start + train]
        test_df = df_ind.iloc[start + train : start + train + test]

        # choose best on train by expectancy
        best_p = None
        best_train = -1e9
        for p in grid:
            exp, _, _, n = _backtest_r(train_df, p)
            # penalize tiny samples
            score = exp * min(1.0, n / 10.0)
            if score > best_train:
                best_train = score
                best_p = p

        if best_p is None:
            continue

        exp_oos, wr_oos, rr_oos, n_oos = _backtest_r(test_df, best_p)
        oos_scores[best_p].append((exp_oos, wr_oos, rr_oos, n_oos))

    # aggregate by mean oos expectancy with sample penalty
    best_p = WFParams(2.0, 3.0, 10)
    best_score = -1e9
    best_summary = {}

    for p, vals in oos_scores.items():
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
        # penalty if low avg trades
        score = mean_exp * min(1.0, mean_n / 8.0)
        if score > best_score:
            best_score = score
            best_p = p
            best_summary = {
                "wf_oos_mean_exp": mean_exp,
                "wf_oos_wr": mean_wr,
                "wf_oos_rr": mean_rr,
                "wf_oos_trades": mean_n,
            }

    return best_p, best_summary


# -----------------------------
# Monte Carlo max DD (trade-return resampling)
# -----------------------------
def monte_carlo_maxdd_from_daily_returns(daily_returns: np.ndarray, n_sim: int = 1500) -> float:
    r = np.array(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 60:
        return 0.0
    max_dds = []
    for _ in range(int(n_sim)):
        shuffled = np.random.permutation(r)
        equity = np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dds.append(float(dd.min()))
    # 95% worst-case (more conservative)
    return float(np.percentile(max_dds, 5))


# -----------------------------
# Risk of Ruin (portfolio level)
# -----------------------------
def portfolio_ror(win_rate: float, rr: float, capital_units: float) -> float:
    """
    Edge = p - (1-p)/rr
    RoR = ((1-Edge)/(1+Edge))^(units)
    """
    p = float(win_rate)
    rr = max(1e-6, float(rr))
    edge = p - (1.0 - p) / rr
    if not np.isfinite(edge) or edge <= 0:
        return 1.0
    base = (1.0 - edge) / (1.0 + edge)
    base = min(0.999999, max(1e-9, base))
    return float(base ** float(capital_units))


# -----------------------------
# Correlation filter
# -----------------------------
def correlation_filter(price_dict: Dict[str, pd.Series], symbols: List[str], threshold: float = 0.7) -> List[str]:
    final: List[str] = []
    for sym in symbols:
        if sym not in price_dict:
            continue
        keep = True
        for f in final:
            if f not in price_dict:
                continue
            a = price_dict[sym].pct_change()
            b = price_dict[f].pct_change()
            c = a.corr(b)
            if np.isfinite(c) and abs(float(c)) >= float(threshold):
                keep = False
                break
        if keep:
            final.append(sym)
    return final


# -----------------------------
# DD control + Market vol control
# -----------------------------
def dd_factor(current_dd: float) -> float:
    dd = float(current_dd)
    if dd < 0.05:
        return 1.0
    if dd < 0.10:
        return 0.7
    if dd < 0.15:
        return 0.4
    return 0.0


def market_vol_factor(vol_ratio: float) -> float:
    v = float(vol_ratio)
    if v >= 1.6:
        return 0.55
    if v >= 1.3:
        return 0.75
    if v <= 0.8:
        return 1.05
    return 1.0


def compute_market_vol_ratio() -> Tuple[float, dict]:
    """
    Market vol proxy using 1306.T (TOPIX ETF) daily ATR%.
    Returns vol_ratio = current_atr_pct / median_atr_pct(1y).
    """
    meta = {"proxy": "1306.T", "ok": False}
    df, m = fetch_daily_stooq("1306.T", min_rows=260)
    meta["fetch_meta"] = m
    if df is None or df.empty:
        return 1.0, meta

    ind = add_indicators(df)
    if ind.empty:
        return 1.0, {**meta, "note": "ind_empty"}

    atrp = ind["ATR_PCT"]
    cur = float(atrp.iloc[-1])
    med = float(atrp.tail(260).median())
    if not (np.isfinite(cur) and np.isfinite(med) and med > 0):
        return 1.0, {**meta, "note": "atr_nan"}

    meta.update({"ok": True, "current_atr_pct": cur, "median_atr_pct_1y": med})
    return float(cur / med), meta


# -----------------------------
# Main scan engine (B-mode heavy sims on top candidates only)
# -----------------------------
def scan_engine(
    *,
    universe_limit: int = 1200,
    sector_top_n: int = 6,
    pre_top_m: int = 60,
    top_n: int = 6,
    corr_threshold: float = 0.7,
) -> Dict[str, object]:
    diag: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "init",
        "stats": {},
        "errors": [],
        "sample_failures": [],
    }

    master = get_jpx_master()
    if master.empty:
        diag["stage"] = "error"
        diag["errors"].append("JPXマスター取得失敗（JPX公式Excel）")
        return {"ok": False, "error": "JPXマスター取得失敗", "diag": diag, "sector_ranking": pd.DataFrame(), "candidates": pd.DataFrame(), "orders": pd.DataFrame()}

    # Limit universe for cloud stability (still large enough)
    tickers = master["ticker"].astype(str).tolist()[: int(universe_limit)]
    diag["stats"]["universe"] = int(len(tickers))

    # fetch concurrently
    rows_latest: List[dict] = []
    price_dict: Dict[str, pd.Series] = {}
    fail_count = 0

    def _one(t: str):
        df, meta = fetch_daily_stooq(t, min_rows=260)
        return t, df, meta

    diag["stage"] = "fetch"
    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(_one, tickers))

    for t, df, meta in results:
        if df is None or df.empty:
            fail_count += 1
            if len(diag["sample_failures"]) < 25:
                diag["sample_failures"].append({"ticker": t, "meta": meta})
            continue
        ind = add_indicators(df)
        if ind.empty:
            fail_count += 1
            if len(diag["sample_failures"]) < 25:
                diag["sample_failures"].append({"ticker": t, "meta": {**(meta or {}), "note": "ind_empty"}})
            continue

        latest = ind.iloc[-1].to_dict()
        latest["Symbol"] = t
        latest["pre_score"] = pre_score(ind.iloc[-1])
        rows_latest.append(latest)
        # keep prices for correlation
        price_dict[t] = ind["Close"].tail(260)

    diag["stats"]["fetched_ok"] = int(len(rows_latest))
    diag["stats"]["fetch_failed"] = int(fail_count)

    if not rows_latest:
        diag["stage"] = "error"
        diag["errors"].append("Stooq取得が全滅（CloudのIPブロック/レート制限/ネットワーク）")
        return {"ok": False, "error": "Stooq取得が全滅", "diag": diag, "sector_ranking": pd.DataFrame(), "candidates": pd.DataFrame(), "orders": pd.DataFrame()}

    latest_df = pd.DataFrame(rows_latest)
    latest_df = latest_df.merge(master, left_on="Symbol", right_on="ticker", how="left")

    # sector strength
    diag["stage"] = "sector_strength"
    sec_rank = calc_sector_strength(latest_df[["Symbol", "sector", "RET_3M", "RSI", "ATR_PCT", "VOL_AVG20"]].copy())
    top_sectors = sec_rank.head(int(sector_top_n))["sector"].astype(str).tolist()
    diag["stats"]["top_sectors"] = top_sectors

    # filter to top sectors and pre-score top M
    cand0 = latest_df[latest_df["sector"].astype(str).isin(top_sectors)].copy()
    cand0 = cand0.sort_values("pre_score", ascending=False).head(int(pre_top_m)).copy()
    diag["stats"]["pre_top_m_actual"] = int(len(cand0))

    # Heavy sims only on these symbols
    diag["stage"] = "heavy_sims"
    heavy_rows = []
    heavy_fail = 0
    for sym in cand0["Symbol"].astype(str).tolist():
        # reconstruct df to run WF/MC: refetch only for top list to save memory
        df, meta = fetch_daily_stooq(sym, min_rows=260)
        if df is None or df.empty:
            heavy_fail += 1
            continue
        ind = add_indicators(df)
        if ind.empty:
            heavy_fail += 1
            continue

        # walk-forward optimize
        p_best, wf_sum = walk_forward_optimize(ind)

        # MonteCarlo DD from daily returns (Close pct change)
        daily_ret = ind["Close"].pct_change().dropna().values
        mc_dd = monte_carlo_maxdd_from_daily_returns(daily_ret, n_sim=1200)

        # Backtest on full (for portfolio RoR estimation)
        exp_full, wr_full, rr_full, n_full = _backtest_r(ind, p_best)

        last = ind.iloc[-1]
        heavy_rows.append({
            "Symbol": sym,
            "Close": float(last["Close"]),
            "ATR": float(last["ATR"]),
            "RSI": float(last["RSI"]),
            "RET_3M": float(last["RET_3M"]),
            "sector": str(cand0[cand0["Symbol"] == sym].iloc[0]["sector"]),
            "name": str(cand0[cand0["Symbol"] == sym].iloc[0]["name"]),
            "pre_score": float(cand0[cand0["Symbol"] == sym].iloc[0]["pre_score"]),
            "wf_best": {"atr_mult_stop": p_best.atr_mult_stop, "tp2_r": p_best.tp2_r, "time_stop_days": p_best.time_stop_days},
            **wf_sum,
            "mc_dd_p05": float(mc_dd),  # negative number (worse)
            "exp_full": float(exp_full),
            "wr_full": float(wr_full),
            "rr_full": float(rr_full),
            "n_full": int(n_full),
        })

    diag["stats"]["heavy_ok"] = int(len(heavy_rows))
    diag["stats"]["heavy_fail"] = int(heavy_fail)

    if not heavy_rows:
        diag["stage"] = "error"
        diag["errors"].append("上位候補の重い計算（WF/MC）が全滅。Stooqの断続ブロックの可能性。")
        return {"ok": False, "error": "heavy_sims失敗", "diag": diag, "sector_ranking": sec_rank, "candidates": pd.DataFrame(), "orders": pd.DataFrame()}

    heavy_df = pd.DataFrame(heavy_rows)

    # Final score: favor oos expectancy + 3m return, penalize MonteCarlo DD and tiny samples
    sample_pen = np.clip(heavy_df["wf_oos_trades"].fillna(0).astype(float) / 8.0, 0.2, 1.0)
    heavy_df["final_score"] = (
        heavy_df["wf_oos_mean_exp"].fillna(0).astype(float) * 120.0 * sample_pen
        + heavy_df["RET_3M"].fillna(0).astype(float) * 35.0
        + heavy_df["pre_score"].fillna(0).astype(float) * 10.0
        - np.abs(heavy_df["mc_dd_p05"].fillna(0).astype(float)) * 10.0
    )

    # sort and correlation filter
    ranked = heavy_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    symbols_ranked = ranked["Symbol"].astype(str).tolist()
    symbols_div = correlation_filter(price_dict, symbols_ranked, threshold=float(corr_threshold))
    final_syms = symbols_div[: int(top_n)]
    final_df = ranked[ranked["Symbol"].astype(str).isin(final_syms)].copy()
    final_df = final_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    diag["stage"] = "done"
    diag["stats"]["selected"] = final_syms
    return {"ok": True, "diag": diag, "sector_ranking": sec_rank, "candidates": final_df}


# -----------------------------
# Portfolio RoR + Position sizing + Order sheet
# -----------------------------
def build_orders(
    candidates: pd.DataFrame,
    *,
    capital_yen: int,
    risk_pct_per_trade: float,
    current_dd: float,
    vol_ratio: float,
) -> Dict[str, object]:
    """
    - portfolio RoR computed from candidates' (wr_full, rr_full) aggregated (weighted by n_full)
    - size each position via:
      base_risk = capital * risk_pct_per_trade
      * dd_factor
      * market_vol_factor
      * ror_factor
      then risk per share = ATR*atr_mult_stop
    """
    if candidates is None or candidates.empty:
        return {"orders": pd.DataFrame(), "portfolio": {}, "ror": 1.0}

    df = candidates.copy()

    # aggregate portfolio win rate / rr by trade counts
    w = df["n_full"].fillna(0).astype(float).values
    w = np.maximum(1.0, w)
    wr = float(np.average(df["wr_full"].fillna(0.0).astype(float).values, weights=w))
    rr = float(np.average(df["rr_full"].fillna(1.0).astype(float).values, weights=w))
    rr = max(0.5, rr)

    # capital units: how many "R-units" the account can withstand (rough proxy)
    # Here: unit = risk_pct_per_trade of capital. If risk=1%, then 100 units.
    units = float(1.0 / max(1e-6, risk_pct_per_trade))

    ror = portfolio_ror(wr, rr, units)

    # ror factor: if ror high, scale down
    # ror<=2%:1.0, <=5%:0.8, <=10%:0.6 else 0.3
    if ror <= 0.02:
        ror_factor = 1.0
    elif ror <= 0.05:
        ror_factor = 0.8
    elif ror <= 0.10:
        ror_factor = 0.6
    else:
        ror_factor = 0.3

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
            risk_budget = base_risk_budget / max(1.0, len(df))  # risk parity-ish by equal risk allocation
            shares = int(math.floor(risk_budget / r_unit))
            shares = int((shares // 100) * 100)  # JP lot
            shares = max(0, shares)

        tp1 = price + r_unit * 1.0
        tp2 = price + r_unit * tp2_r

        orders.append({
            "Symbol": str(row["Symbol"]),
            "name": str(row.get("name", "")),
            "sector": str(row.get("sector", "")),
            "entry": price,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
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
