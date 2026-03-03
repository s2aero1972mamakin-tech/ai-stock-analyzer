# -*- coding: utf-8 -*-
from __future__ import annotations

"""
JPX × SBI 半自動トレード（2ファイル完全版）

この logic.py は、以下を 1ファイルで完結させています:
- JPXマスター取得（銘柄/業種/市場区分/規模区分）
- OHLC取得（DB優先 → キャッシュ → yfinance/stooq）
- 日次バッチ（全銘柄をSQLiteへ保存：外部制限対策）
- セクター強度 → 候補抽出 → WF（簡易） → MC-DD → スコア → 相関除外
- 資金/市場ボラ/DD を加味して SBI注文書CSV 用 DataFrame を生成

重要:
- 「止まらない」を最優先。取得失敗が発生しても部分結果で diag を返す設計。
- Streamlit 側は network I/O を最小化するため、DB(=ローカル)を優先。
"""

import math
import os
import time
import sqlite3
import datetime as _dt
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 外部依存（ユーザー環境にある前提）
import yfinance as yf  # type: ignore

_STOOQ_DOMAINS = ["stooq.pl", "stooq.com"]
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-stock-analyzer/2file-complete)"}

# -----------------------------
# Paths
# -----------------------------
_BASE_DIR = os.path.dirname(__file__)
_CACHE_DIR = os.path.join(_BASE_DIR, ".cache_ohlc_pickle")
_DATA_DIR = os.path.join(_BASE_DIR, ".data")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.makedirs(_DATA_DIR, exist_ok=True)

DEFAULT_DB_PATH = os.path.join(_DATA_DIR, "jpx_ohlc.sqlite")

# -----------------------------
# SQLite OHLC Store
# -----------------------------
def _db_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    con = _db_connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlc (
              ticker TEXT NOT NULL,
              d TEXT NOT NULL,
              Open REAL, High REAL, Low REAL, Close REAL,
              AdjClose REAL,
              Volume REAL,
              PRIMARY KEY (ticker, d)
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_ticker_d ON ohlc(ticker, d);")
        con.commit()
        # meta table for batch status
        con.execute("""
            CREATE TABLE IF NOT EXISTS meta (
              k TEXT PRIMARY KEY,
              v TEXT NOT NULL
            )
        """)
        con.commit()
    finally:
        con.close()


def db_set_meta(db_path: str, k: str, v: str) -> None:
    init_db(db_path)
    con = _db_connect(db_path)
    try:
        con.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v;", (k, str(v)))
        con.commit()
    finally:
        con.close()

def db_get_meta(db_path: str) -> dict:
    init_db(db_path)
    con = _db_connect(db_path)
    try:
        cur = con.execute("SELECT k, v FROM meta;")
        rows = cur.fetchall()
        return {str(k): str(v) for (k, v) in rows}
    except Exception:
        return {}
    finally:
        con.close()

def db_get_last_batch_jst_date(db_path: str) -> Optional[_dt.date]:
    meta = db_get_meta(db_path)
    v = meta.get("last_batch_jst")
    if not v:
        return None
    try:
        dt = _dt.datetime.fromisoformat(v)
        return dt.date()
    except Exception:
        return None

def db_get_last_date(db_path: str, ticker: str) -> Optional[_dt.date]:
    init_db(db_path)
    con = _db_connect(db_path)
    try:
        cur = con.execute("SELECT MAX(d) FROM ohlc WHERE ticker=?", (ticker,))
        row = cur.fetchone()
        if not row or not row[0]:
            return None
        return _dt.date.fromisoformat(str(row[0]))
    finally:
        con.close()

def db_load_ohlc(db_path: str, ticker: str, max_age_days: int = 2) -> Optional[pd.DataFrame]:
    """
    DB優先でOHLCを返す。max_age_days<=0 は常にmiss。
    """
    if max_age_days <= 0:
        return None
    last = db_get_last_date(db_path, ticker)
    if last is None:
        return None
    age = (_dt.date.today() - last).days
    if age > int(max_age_days):
        return None

    con = _db_connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT d, Open, High, Low, Close, AdjClose, Volume FROM ohlc WHERE ticker=? ORDER BY d ASC",
            con,
            params=(ticker,),
        )
    finally:
        con.close()

    if df is None or df.empty:
        return None
    df["d"] = pd.to_datetime(df["d"])
    df = df.set_index("d")
    df.index.name = "Date"
    # yfinance互換列名（Adj Close）
    if "AdjClose" in df.columns and "Adj Close" not in df.columns:
        df = df.rename(columns={"AdjClose": "Adj Close"})
    return df

def db_upsert_ohlc(db_path: str, ticker: str, df: pd.DataFrame) -> int:
    """
    df index=Date, columns: Open High Low Close Volume (optional Adj Close)
    戻り値: 挿入/更新行数（概算）
    """
    if df is None or df.empty:
        return 0
    init_db(db_path)

    use = df.copy()
    use = use.reset_index()
    if "Date" not in use.columns:
        # index名違い救済
        use = use.rename(columns={use.columns[0]: "Date"})
    use["d"] = pd.to_datetime(use["Date"]).dt.date.astype(str)

    # normalize column names
    if "Adj Close" in use.columns and "AdjClose" not in use.columns:
        use["AdjClose"] = use["Adj Close"]
    cols = ["Open","High","Low","Close","AdjClose","Volume"]
    for c in cols:
        if c not in use.columns:
            use[c] = np.nan

    rows = list(
        zip(
            [str(ticker)] * len(use),
            use["d"].astype(str).tolist(),
            use["Open"].astype(float).tolist(),
            use["High"].astype(float).tolist(),
            use["Low"].astype(float).tolist(),
            use["Close"].astype(float).tolist(),
            use["AdjClose"].astype(float).tolist(),
            use["Volume"].astype(float).tolist(),
        )
    )
    con = _db_connect(db_path)
    try:
        con.executemany(
            """
            INSERT INTO ohlc (ticker, d, Open, High, Low, Close, AdjClose, Volume)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker, d) DO UPDATE SET
              Open=excluded.Open,
              High=excluded.High,
              Low=excluded.Low,
              Close=excluded.Close,
              AdjClose=excluded.AdjClose,
              Volume=excluded.Volume
            """,
            rows,
        )
        con.commit()
        return len(rows)
    finally:
        con.close()

# -----------------------------
# Pickle cache (optional, tiny)
# -----------------------------
def _cache_path(ticker: str) -> str:
    safe = str(ticker).replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.pkl")

def _load_pickle_cache(ticker: str, max_age_days: int = 2) -> Optional[pd.DataFrame]:
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

def _save_pickle_cache(ticker: str, df: pd.DataFrame) -> None:
    try:
        if df is None or df.empty:
            return
        p = _cache_path(ticker)
        df.to_pickle(p)
    except Exception:
        pass

# -----------------------------
# JPX Master
# -----------------------------
def get_jpx_master() -> pd.DataFrame:
    """
    JPXが提供するマスター（Excel）を取得。
    """
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        df = pd.read_excel(url)
    except Exception:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=20)
            r.raise_for_status()
            df = pd.read_excel(BytesIO(r.content))
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
    return (m or "").strip().lower().replace("　", " ")

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
    for _, g in df.groupby("sector", dropna=False):
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

# -----------------------------
# Data fetchers
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
            df = pd.read_csv(BytesIO(text.encode("utf-8")))
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
            # stooq may not provide Adj Close
            if "Adj Close" not in df.columns and "AdjClose" not in df.columns:
                df["Adj Close"] = df["Close"]
            df = df[["Open","High","Low","Close","Adj Close","Volume"]].dropna()
            if len(df) < min_rows:
                last_meta.update({"note": "too_few_rows", "rows": int(len(df))})
                return None, last_meta
            return df, last_meta
        except Exception as e:
            last_meta.update({"exc": f"{type(e).__name__}: {e}"})
            continue
    return None, last_meta

def fetch_daily_yf(ticker: str, *, min_rows: int = 260, period: str = "max") -> Tuple[Optional[pd.DataFrame], dict]:
    meta = {"provider": "yfinance", "ticker": ticker, "period": period}
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            meta.update({"note": "empty"})
            return None, meta
        df = df.dropna()
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        df = df[["Open","High","Low","Close","Adj Close","Volume"]]
        if len(df) < min_rows:
            meta.update({"note": "too_few_rows", "rows": int(len(df))})
            return None, meta
        return df, meta
    except Exception as e:
        meta.update({"exc": f"{type(e).__name__}: {e}"})
        return None, meta

def fetch_daily(
    ticker: str,
    *,
    min_rows: int = 260,
    prefer_yf: bool = True,
    cache_days: int = 2,
    db_path: str = DEFAULT_DB_PATH,
) -> Tuple[Optional[pd.DataFrame], dict, bool, bool]:
    """
    戻り:
      df, meta, stooq_rate_limited, db_hit
    """
    # 1) DB hit
    df_db = db_load_ohlc(db_path, ticker, max_age_days=int(cache_days))
    if df_db is not None and not df_db.empty:
        return df_db, {"provider": "db", "ticker": ticker, "note": "db_hit"}, False, True

    # 2) pickle cache hit
    cached = _load_pickle_cache(ticker, max_age_days=int(cache_days))
    if cached is not None and not cached.empty:
        return cached, {"provider": "cache", "ticker": ticker, "note": "pickle_hit"}, False, False

    # 3) network
    if prefer_yf:
        df, meta = fetch_daily_yf(ticker, min_rows=min_rows)
        if df is not None and not df.empty:
            # persist
            _save_pickle_cache(ticker, df)
            db_upsert_ohlc(db_path, ticker, df)
            return df, meta, False, False

    df, meta = fetch_daily_stooq(ticker, min_rows=min_rows)
    if df is not None and not df.empty:
        _save_pickle_cache(ticker, df)
        db_upsert_ohlc(db_path, ticker, df)
        return df, meta, False, False

    rate_limited = (meta or {}).get("note") == "stooq_rate_limited"
    df2, meta2 = fetch_daily_yf(ticker, min_rows=min_rows)
    if df2 is not None and not df2.empty:
        _save_pickle_cache(ticker, df2)
        db_upsert_ohlc(db_path, ticker, df2)
        return df2, meta2, rate_limited, False

    return None, (meta2 or meta), rate_limited, False

# -----------------------------
# Indicators & scoring
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
    # Avoid duplicate column names (pandas may return DataFrame for out['ATR'] if duplicated)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    out["SMA25"] = out["Close"].rolling(25).mean()
    out["SMA75"] = out["Close"].rolling(75).mean()
    out["RSI"] = _rsi(out["Close"], 14)
    out["ATR"] = pd.Series(_atr(out, 14), index=out.index).astype(float)
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
        (g["ret_3m"].fillna(0) * 100.0) * 0.45
        + (g["rsi"].fillna(50) / 100.0) * 0.15
        + (np.log1p(g["vol"].fillna(0)) / 20.0) * 0.25
        - (g["atr_pct"].fillna(0) / 30.0) * 0.15
    )
    return g.sort_values("score", ascending=False)

def pre_score(latest: pd.Series) -> float:
    trend = 1.0 if float(latest.get("SMA25", 0)) > float(latest.get("SMA75", 0)) else 0.0
    r3m = float(latest.get("RET_3M", 0.0) or 0.0)
    rsi = float(latest.get("RSI", 50.0) or 50.0)
    vol = float(latest.get("VOL_AVG20", 0.0) or 0.0)
    atrp = float(latest.get("ATR_PCT", 0.0) or 0.0)
    # RET偏重を弱め、流動性/ATR/DD側で安定させる方向
    return float(
        0.30 * trend
        + 0.25 * r3m
        + 0.20 * (1.0 - abs(rsi - 55.0) / 55.0)
        + 0.20 * (np.log1p(vol) / 20.0)
        - 0.05 * (atrp / 30.0)
    )


def add_trade_plan_columns(df: pd.DataFrame) -> pd.DataFrame:
    """UI表示用に、戦略タイプと Entry/SL/TP の目安を付与。

    - heavy_sims が間に合わない場合でも「銘柄＋方式＋価格目安」を返せるようにする。
    - ここでの価格は “目安” であり、最終執行はユーザー側アプリで判断。
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    def _strategy_row(r) -> str:
        r3m = float(r.get("RET_3M", 0.0) or 0.0)
        rsi = float(r.get("RSI", 50.0) or 50.0)
        atrp = float(r.get("ATR_PCT", 0.0) or 0.0)
        if r3m > 0.12 and rsi >= 55:
            return "Trend/Breakout"
        if rsi <= 35:
            return "MeanRevert/Rebound"
        if atrp <= 2.0:
            return "Range/LowVol"
        return "Hybrid"

    out["strategy"] = out.apply(_strategy_row, axis=1)

    price = out.get("Close")
    if price is None:
        out["entry"] = np.nan
        out["stop_loss"] = np.nan
        out["take_profit"] = np.nan
        return out

    # ATR%があれば ATR ベース、無ければ固定幅
    atrp = out.get("ATR_PCT", pd.Series([np.nan]*len(out)))
    base_stop = 0.03
    stop_pct = (atrp.fillna(3.0) / 100.0 * 1.2).clip(0.02, 0.12)
    stop_pct = np.maximum(stop_pct, base_stop)

    rr = out.get("wf_oos_rr", pd.Series([np.nan]*len(out))).fillna(2.0).clip(1.2, 3.5)
    tp_pct = (stop_pct * rr).clip(0.04, 0.30)

    out["entry"] = price.astype(float)
    out["stop_loss"] = (price.astype(float) * (1.0 - stop_pct)).astype(float)
    out["take_profit"] = (price.astype(float) * (1.0 + tp_pct)).astype(float)
    return out

# -----------------------------
# Walk-forward (simplified)
# -----------------------------
def _backtest_r(returns: pd.Series, stop_r: float = 1.0, tp2_r: float = 2.0, time_stop: int = 20) -> Tuple[float, float, int]:
    """
    超軽量バックテスト（擬似）
    - returns: daily returns
    戻り: (expected_R, win_rate, trades)
    """
    if returns is None or len(returns) < 60:
        return 0.0, 0.0, 0
    r = returns.dropna().values
    # 擬似：累積が -stop_r 相当になったら負け、 +tp2_r 相当になったら勝ち、time_stopでクローズ
    # ここでは daily return を R にスケールするため ATR 等は使わず簡略化
    wins = 0
    total = 0
    sum_r = 0.0
    i = 0
    while i < len(r) - 1:
        total += 1
        acc = 0.0
        held = 0
        while i < len(r) and held < time_stop:
            acc += float(r[i])
            held += 1
            i += 1
            if acc <= -abs(stop_r) * 0.03:  # rough
                sum_r += -1.0
                break
            if acc >= abs(tp2_r) * 0.03:
                wins += 1
                sum_r += float(tp2_r)
                break
        else:
            # time stop
            sum_r += max(-1.0, min(tp2_r, acc / 0.03))
    if total <= 0:
        return 0.0, 0.0, 0
    exp_r = sum_r / total
    wr = wins / total
    return float(exp_r), float(wr), int(total)

def walk_forward_optimize(df: pd.DataFrame) -> dict:
    """
    簡易WF: パラメータ候補を走査しOOS期待値が最大のものを採用
    """
    close = df["Close"].astype(float)
    rets = close.pct_change().dropna()

    candidates = [
        (1.0, 1.8, 15),
        (1.0, 2.0, 20),
        (1.2, 2.0, 20),
        (1.0, 2.3, 25),
        (1.3, 2.4, 20),
    ]

    split = int(len(rets) * 0.7)
    ins = rets.iloc[:split]
    oos = rets.iloc[split:]

    best = None
    best_score = -1e9
    for stop_r, tp2_r, tstop in candidates:
        exp_in, wr_in, tr_in = _backtest_r(ins, stop_r=stop_r, tp2_r=tp2_r, time_stop=tstop)
        exp_o, wr_o, tr_o = _backtest_r(oos, stop_r=stop_r, tp2_r=tp2_r, time_stop=tstop)
        # OOS重視、trades少なすぎは罰
        score = exp_o + 0.2 * wr_o - (0.3 if tr_o < 6 else 0.0)
        if score > best_score:
            best_score = score
            best = {
                "atr_mult_stop": float(stop_r),
                "tp2_r": float(tp2_r),
                "time_stop_days": int(tstop),
                "wf_oos_mean_exp": float(exp_o),
                "wf_oos_wr": float(wr_o),
                "wf_oos_rr": float(tp2_r / max(1e-9, stop_r)),
                "wf_oos_trades": int(tr_o),
            }
    return best or {
        "atr_mult_stop": 1.0,
        "tp2_r": 2.0,
        "time_stop_days": 20,
        "wf_oos_mean_exp": 0.0,
        "wf_oos_wr": 0.0,
        "wf_oos_rr": 2.0,
        "wf_oos_trades": 0,
    }

def monte_carlo_maxdd_from_daily_returns(r: pd.Series, n_paths: int = 400) -> dict:
    """
    非常に軽いMC: リターンのブートストラップで最大DD分布のp05を推定
    """
    r = r.dropna()
    if len(r) < 120:
        return {"mc_dd_p05": 0.20, "paths": 0}

    arr = r.values.astype(float)
    L = len(arr)
    rng = np.random.default_rng(0)
    dds = []
    for _ in range(int(n_paths)):
        samp = rng.choice(arr, size=L, replace=True)
        eq = np.cumprod(1.0 + samp)
        peak = np.maximum.accumulate(eq)
        dd = 1.0 - (eq / (peak + 1e-12))
        dds.append(float(np.max(dd)))
    dds = np.array(dds)
    return {"mc_dd_p05": float(np.quantile(dds, 0.05)), "paths": int(n_paths)}

def portfolio_ror(p_win: float, rr: float, risk_frac: float, n: int = 200) -> float:
    """
    Risk of Ruin 近似（簡易）
    """
    p = float(np.clip(p_win, 1e-6, 1 - 1e-6))
    rr = max(1e-6, float(rr))
    q = 1.0 - p
    # 期待成長がマイナスなら高い
    edge = p * rr - q
    if edge <= 0:
        return 0.85
    # ざっくり: 破産確率 ~ exp(-k)
    k = edge / (risk_frac + 1e-12)
    return float(np.clip(math.exp(-k), 0.0, 1.0))

def dd_factor(current_dd: float) -> float:
    # DDが大きいほど縮小（0.0..1.0）
    return float(np.clip(1.0 - (current_dd / 0.25), 0.15, 1.0))

def market_vol_factor(vol_ratio: float) -> float:
    # 市場ボラが高いほど縮小
    return float(np.clip(1.0 / (1.0 + max(0.0, vol_ratio - 1.0)), 0.30, 1.0))

def compute_market_vol_ratio(progress_cb: Optional[Callable] = None) -> Tuple[float, dict]:
    """
    1306.T を proxy として直近ATR%を算出し、平常時=1.0として比率化
    """
    if progress_cb:
        progress_cb("market_vol_fetch", {})
    df, meta = fetch_daily_yf("1306.T", min_rows=260)
    if df is None or df.empty:
        return 1.0, {"ok": False, "meta": meta}
    ind = add_indicators(df)
    atrp = float(ind["ATR_PCT"].iloc[-1])
    # 平常ATR%を 1.2% と仮置き（必要なら後で調整）
    vol_ratio = float(np.clip(atrp / 1.2, 0.5, 3.0))
    return vol_ratio, {"ok": True, "atr_pct": atrp, "meta": meta}

def correlation_filter(df: pd.DataFrame, corr_threshold: float = 0.70) -> pd.DataFrame:
    """
    Closeの相関で似た銘柄を除外（上位スコア優先）
    """
    if df is None or df.empty or "Symbol" not in df.columns:
        return df
    syms = df["Symbol"].astype(str).tolist()
    closes = []
    used = []
    for s in syms:
        ser = df.loc[df["Symbol"] == s, "_close_series"].values
        if len(ser) == 0:
            continue
        closes.append(pd.Series(ser[0], name=s))
        used.append(s)
    if not closes:
        return df
    mat = pd.concat(closes, axis=1).pct_change().dropna()
    if mat.empty:
        return df
    corr = mat.corr().fillna(0.0)
    keep = []
    for s in used:
        ok = True
        for k in keep:
            if abs(float(corr.loc[s, k])) >= float(corr_threshold):
                ok = False
                break
        if ok:
            keep.append(s)
    return df[df["Symbol"].astype(str).isin(keep)].reset_index(drop=True)

# -----------------------------
# Scan engine
# -----------------------------
def scan_engine(
    *,
    universe_limit: int = 800,
    market_filter: str = "PRIME",
    size_filter: str = "ALL",
    universe_mode: str = "RANDOM_STRATIFIED",
    min_price: float = 200.0,
    min_avg_volume: float = 50000.0,
    cache_days: int = 2,
    sector_top_n: int = 6,
    pre_top_m: int = 35,
    top_n: int = 6,
    corr_threshold: float = 0.70,
    max_workers: int = 8,
    time_budget_sec: int = 70,
    progress_cb: Optional[Callable] = None,
    db_path: str = DEFAULT_DB_PATH,
    yf_bulk_chunk: int = 180,
) -> dict:
    t0 = time.time()
    diag = {
        "ok": False,
        "stage": "init",
        "errors": [],
        "sample_failures": [],
        "meta": {
            "universe_limit": int(universe_limit),
            "market_filter": str(market_filter),
            "size_filter": str(size_filter),
            "universe_mode": str(universe_mode),
            "min_price": float(min_price),
            "min_avg_volume": float(min_avg_volume),
            "cache_days": int(cache_days),
            "pre_top_m": int(pre_top_m),
            "top_n": int(top_n),
            "time_budget_sec": int(time_budget_sec),
            "db_path": str(db_path),
        },
        "stats": {},
    }

    try:
        init_db(db_path)
    except Exception as e:
        diag["errors"].append(f"db_init_failed: {type(e).__name__}: {e}")

    if progress_cb:
        progress_cb("universe", {"msg": "loading jpx master"})
    master = get_jpx_master()
    if master is None or master.empty:
        diag["errors"].append("jpx_master_empty")
        diag["stage"] = "error"
        return {"ok": False, "error": "jpx_master_empty", "diag": diag, "sector_ranking": pd.DataFrame(), "candidates": pd.DataFrame()}

    master_f = filter_master(master, market_filter=market_filter, size_filter=size_filter)
    seed = int(_dt.date.today().strftime("%Y%m%d"))
    ticks = pick_universe_tickers(master_f, int(universe_limit), mode=universe_mode, seed=seed)
    diag["stats"]["universe_tickers"] = int(len(ticks))

    # --- Fetch OHLC (DB first, then yfinance bulk to reduce rate-limit hits) ---
    if progress_cb:
        progress_cb("fetch", {"total": len(ticks)})

    MIN_ROWS = 260

    ok = 0
    fail = 0
    db_hit = 0
    rows: List[dict] = []
    failures: List[dict] = []

    # 1) DB hit check (fast, no external calls)
    missing: List[str] = []
    for i, tk in enumerate(ticks, start=1):
        if time.time() - t0 > float(time_budget_sec):
            diag["errors"].append("time_budget_exceeded_during_db_check")
            break
        df = db_load_ohlc(db_path=db_path, ticker=tk, max_age_days=int(cache_days))
        meta = {"provider": "db"} if df is not None and not df.empty else {"provider": "db", "note": "miss_or_stale"}
        if df is None or df.empty:
            missing.append(tk)
            continue
        db_hit += 1
        ind = add_indicators(df)
        if ind is None or ind.empty:
            fail += 1
            if len(failures) < 25:
                failures.append({"ticker": tk, "meta": {"note": "indicator_empty_db", **(meta or {})}})
            continue
        last = ind.iloc[-1].copy()
        if float(last.get("Close", 0.0)) < float(min_price) or float(last.get("VOL_AVG20", 0.0)) < float(min_avg_volume):
            continue
        ok += 1
        rows.append({
            "Symbol": tk,
            "Close": float(last.get("Close", 0.0)),
            "RET_3M": float(last.get("RET_3M", 0.0)),
            "RSI": float(last.get("RSI", 50.0)),
            "ATR_PCT": float(last.get("ATR_PCT", 0.0)),
            "VOL_AVG20": float(last.get("VOL_AVG20", 0.0)),
            "_close_series": df["Close"].astype(float).values,
        })
        if progress_cb and i % 50 == 0:
            progress_cb("fetch_progress", {"done": i, "total": len(ticks), "ok": ok, "fail": fail, "db_hit": db_hit, "missing": len(missing), "mode": "db_check"})

    # 2) Bulk yfinance download for missing (dramatically reduces external hit count)
    if missing and (time.time() - t0) <= float(time_budget_sec):
        done_n = 0
        for chunk in [missing[i:i+int(yf_bulk_chunk)] for i in range(0, len(missing), int(yf_bulk_chunk))]:
            if time.time() - t0 > float(time_budget_sec):
                diag["errors"].append("time_budget_exceeded_during_yf_bulk")
                break
            try:
                got = _yf_multi_download(chunk, days=730)
            except Exception as e:
                got = {}
                diag["errors"].append(f"yf_bulk_exc: {type(e).__name__}: {e}")

            for tk in chunk:
                done_n += 1
                df = got.get(tk)
                if df is None or df.empty or len(df) < int(MIN_ROWS):
                    fail += 1
                    if len(failures) < 25:
                        failures.append({"ticker": tk, "meta": {"provider": "yfinance_bulk", "note": "empty_or_short"}})
                    continue
                try:
                    df2 = df.copy()
                    df2.index = pd.to_datetime(df2.index)
                    df2 = df2.sort_index()
                    cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df2.columns]
                    df2 = df2[cols]
                    if "Adj Close" not in df2.columns and "Close" in df2.columns:
                        df2["Adj Close"] = df2["Close"]

                    db_upsert_ohlc(db_path=db_path, ticker=tk, df=df2)

                    ind = add_indicators(df2)
                    if ind is None or ind.empty:
                        fail += 1
                        if len(failures) < 25:
                            failures.append({"ticker": tk, "meta": {"provider": "yfinance_bulk", "note": "indicator_empty"}})
                        continue
                    last = ind.iloc[-1].copy()
                    if float(last.get("Close", 0.0)) < float(min_price) or float(last.get("VOL_AVG20", 0.0)) < float(min_avg_volume):
                        continue
                    ok += 1
                    rows.append({
                        "Symbol": tk,
                        "Close": float(last.get("Close", 0.0)),
                        "RET_3M": float(last.get("RET_3M", 0.0)),
                        "RSI": float(last.get("RSI", 50.0)),
                        "ATR_PCT": float(last.get("ATR_PCT", 0.0)),
                        "VOL_AVG20": float(last.get("VOL_AVG20", 0.0)),
                        "_close_series": df2["Close"].astype(float).values,
                    })
                except Exception as e:
                    fail += 1
                    if len(failures) < 25:
                        failures.append({"ticker": tk, "meta": {"provider": "yfinance_bulk", "exc": f"{type(e).__name__}: {e}"}})

            if progress_cb:
                progress_cb("fetch_progress", {"done": min(len(ticks), done_n), "total": len(ticks), "ok": ok, "fail": fail, "db_hit": db_hit, "mode": "yfinance_bulk"})

    diag["sample_failures"] = failures
    diag["stats"]["fetch_ok"] = int(ok)
    diag["stats"]["fetch_fail"] = int(fail)
    diag["stats"]["db_hit"] = int(db_hit)

    if not rows:
        diag["errors"].append("no_rows_after_filters")
        diag["stage"] = "error"
        return {"ok": False, "error": "no_rows_after_filters", "diag": diag, "sector_ranking": pd.DataFrame(), "candidates": pd.DataFrame()}

    latest_df = pd.DataFrame(rows)
    latest_df = latest_df.merge(master_f[["ticker","name","sector"]], how="left", left_on="Symbol", right_on="ticker")
    latest_df = latest_df.drop(columns=["ticker"], errors="ignore")
    latest_df["name"] = latest_df["name"].fillna("")
    latest_df["sector"] = latest_df["sector"].fillna("")

    # Sector strength
    if progress_cb:
        progress_cb("sector_strength", {})
    sec_rank = calc_sector_strength(latest_df)
    top_secs = sec_rank.head(int(sector_top_n))["sector"].astype(str).tolist()
    diag["stats"]["top_sectors"] = top_secs

    # preselect
    if progress_cb:
        progress_cb("preselect", {})
    latest_df["_pre"] = latest_df.apply(lambda r: pre_score(r), axis=1)
    pool = latest_df[latest_df["sector"].astype(str).isin(top_secs)].copy()
    pool = pool.sort_values("_pre", ascending=False).head(int(pre_top_m)).reset_index(drop=True)
    diag["stats"]["preselect_n"] = int(len(pool))

    # heavy sims (WF + MC)
    if progress_cb:
        progress_cb("heavy_sims", {"total": len(pool)})

    heavy_ok = 0
    heavy_fail = 0

    def heavy_one(sym: str):
        # fetch full (DB) again for returns
        df, meta, _, _ = fetch_daily(sym, min_rows=260, prefer_yf=True, cache_days=cache_days, db_path=db_path)
        if df is None or df.empty:
            return sym, None, {"note": "fetch_fail", **(meta or {})}
        df = df.dropna()
        wf = walk_forward_optimize(df)
        mc = monte_carlo_maxdd_from_daily_returns(df["Close"].pct_change().dropna(), n_paths=450)
        out = {**wf, **mc}
        return sym, out, {"provider": (meta or {}).get("provider")}

    heavy_rows = []
    with ThreadPoolExecutor(max_workers=max(2, int(max_workers)//2)) as ex:
        futs = {ex.submit(heavy_one, s): s for s in pool["Symbol"].astype(str).tolist()}
        done_n = 0
        for fut in as_completed(futs):
            done_n += 1
            if time.time() - t0 > float(time_budget_sec):
                diag["errors"].append("time_budget_exceeded_during_heavy")
                break
            s = futs[fut]
            try:
                sym, out, meta = fut.result()
            except Exception as e:
                sym, out, meta = s, None, {"exc": f"{type(e).__name__}: {e}"}
            if out is None:
                heavy_fail += 1
            else:
                heavy_ok += 1
                heavy_rows.append({"Symbol": sym, **out})
            if progress_cb and done_n % 5 == 0:
                progress_cb("heavy_progress", {
                    "done": done_n, "total": len(pool),
                    "heavy_ok": heavy_ok, "heavy_fail": heavy_fail
                })

    diag["stats"]["heavy_ok"] = int(heavy_ok)
    diag["stats"]["heavy_fail"] = int(heavy_fail)

    heavy_df = pd.DataFrame(heavy_rows)
    if heavy_df.empty:
        # Degraded mode: still return a ranked pool with simple risk bands and plan.
        diag["errors"].append("heavy_df_empty_degraded")
        merged = pool.copy()
        merged["wf_oos_wr"] = np.nan
        merged["wf_oos_rr"] = np.nan
        merged["wf_oos_mean_exp"] = np.nan
        merged["wf_oos_trades"] = np.nan
        merged["mc_dd_p05"] = np.nan
        merged["final_score"] = merged["_pre"].astype(float)
        merged = merged.sort_values("final_score", ascending=False).head(int(top_n)).reset_index(drop=True)
        merged = add_trade_plan_columns(merged)
        keep_cols = [
            "Symbol","name","sector","Close","RET_3M","RSI","ATR_PCT","VOL_AVG20",
            "strategy","entry","stop_loss","take_profit",
            "final_score",
        ]
        for c in keep_cols:
            if c not in merged.columns:
                merged[c] = np.nan
        merged = merged[keep_cols]

        diag["ok"] = True
        diag["stage"] = "done"
        diag["elapsed_sec"] = float(time.time() - t0)
        if progress_cb:
            progress_cb("done", {"elapsed_sec": diag["elapsed_sec"], "mode": "degraded"})
        return {"ok": True, "diag": diag, "sector_ranking": sec_rank, "candidates": merged}

    merged = pool.merge(heavy_df, on="Symbol", how="left")
    merged = merged.dropna(subset=["wf_oos_wr","wf_oos_rr","mc_dd_p05"], how="any")

    if merged.empty:
        diag["errors"].append("merged_empty_after_heavy")
        diag["stage"] = "error"
        return {"ok": False, "error": "merged_empty_after_heavy", "diag": diag, "sector_ranking": sec_rank, "candidates": pd.DataFrame()}

    # final score: RET偏重を下げ、WF/MC/DD側を強める
    merged["final_score"] = (
        0.22 * merged["RET_3M"].clip(-0.4, 0.8).fillna(0.0)
        + 0.30 * merged["wf_oos_wr"].clip(0.0, 1.0).fillna(0.0)
        + 0.28 * merged["wf_oos_rr"].clip(0.5, 4.0).fillna(0.0) / 4.0
        + 0.20 * merged["wf_oos_mean_exp"].clip(-2.0, 3.0).fillna(0.0) / 3.0
        - 0.45 * merged["mc_dd_p05"].clip(0.05, 0.80).fillna(0.25)
    )

    merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    merged = correlation_filter(merged, corr_threshold=float(corr_threshold))
    merged = merged.head(int(top_n)).reset_index(drop=True)

    merged = add_trade_plan_columns(merged)

    # keep columns
    keep_cols = [
        "Symbol","name","sector","Close","RET_3M","RSI","ATR_PCT","VOL_AVG20",
        "wf_oos_mean_exp","wf_oos_wr","wf_oos_rr","wf_oos_trades",
        "mc_dd_p05","strategy","entry","stop_loss","take_profit","final_score",
    ]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[keep_cols]

    diag["ok"] = True
    diag["stage"] = "done"
    diag["elapsed_sec"] = float(time.time() - t0)

    if progress_cb:
        progress_cb("done", {"elapsed_sec": diag["elapsed_sec"]})

    return {"ok": True, "diag": diag, "sector_ranking": sec_rank, "candidates": merged}

# -----------------------------
# Orders
# -----------------------------
def build_orders(
    cands: pd.DataFrame,
    *,
    capital_yen: int = 1_000_000,
    risk_pct_per_trade: float = 0.01,
    current_dd: float = 0.0,
    vol_ratio: float = 1.0,
) -> dict:
    if cands is None or cands.empty:
        return {"portfolio": {"ok": False, "error": "no_candidates"}, "orders": pd.DataFrame()}

    cap = float(capital_yen)
    risk_frac = float(risk_pct_per_trade)
    f_dd = dd_factor(float(current_dd))
    f_vol = market_vol_factor(float(vol_ratio))
    adj_risk = risk_frac * f_dd * f_vol

    orders = []
    p_wins = []
    rrs = []

    for _, r in cands.iterrows():
        sym = str(r["Symbol"])
        price = float(r.get("Close", 0.0) or 0.0)
        if price <= 0:
            continue

        p_win = float(np.clip(r.get("wf_oos_wr", 0.5) or 0.5, 0.05, 0.95))
        rr = float(np.clip(r.get("wf_oos_rr", 2.0) or 2.0, 0.5, 4.0))
        mcdd = float(np.clip(r.get("mc_dd_p05", 0.25) or 0.25, 0.05, 0.80))

        # SL/TP: ATRはここでは簡略、DDが大きいほどポジ縮小＆SL広め
        stop_pct = float(np.clip(0.03 + 0.10 * mcdd, 0.03, 0.18))
        tp_pct = float(np.clip(stop_pct * rr, 0.05, 0.40))

        risk_yen = cap * adj_risk
        qty = int(max(0, math.floor(risk_yen / max(1.0, price * stop_pct))))
        if qty <= 0:
            continue

        sl = price * (1.0 - stop_pct)
        tp = price * (1.0 + tp_pct)

        orders.append({
            "symbol": sym,
            "side": "BUY",
            "qty": int(qty),
            "entry": float(price),
            "stop_loss": float(sl),
            "take_profit": float(tp),
        })
        p_wins.append(p_win)
        rrs.append(rr)

    orders_df = pd.DataFrame(orders)
    if orders_df.empty:
        return {"portfolio": {"ok": False, "error": "qty_zero_or_price_missing"}, "orders": orders_df}

    # Portfolio-level rough risk stats
    p_win_port = float(np.mean(p_wins)) if p_wins else 0.5
    rr_port = float(np.mean(rrs)) if rrs else 2.0
    ror = portfolio_ror(p_win_port, rr_port, adj_risk)

    portfolio = {
        "ok": True,
        "capital_yen": int(capital_yen),
        "risk_pct_per_trade": float(risk_pct_per_trade),
        "dd_factor": float(f_dd),
        "market_vol_factor": float(f_vol),
        "adjusted_risk_frac": float(adj_risk),
        "p_win_port": float(p_win_port),
        "rr_port": float(rr_port),
        "risk_of_ruin_est": float(ror),
        "n_orders": int(len(orders_df)),
    }
    return {"portfolio": portfolio, "orders": orders_df}

# -----------------------------
# Daily batch
# -----------------------------
def _yf_multi_download(tickers: List[str], days: int = 730) -> Dict[str, pd.DataFrame]:
    """
    yfinance は複数ティッカーまとめDLが可能（ヒット数を削減）
    """
    if not tickers:
        return {}
    period = "max"  # maxで取り、あとで末尾 days に切る
    df = yf.download(" ".join(tickers), period=period, interval="1d", auto_adjust=False, progress=False, threads=True, group_by="ticker")
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    # 1銘柄のときは列階層が違うことがある
    if isinstance(df.columns, pd.MultiIndex):
        for tk in tickers:
            if tk in df.columns.get_level_values(0):
                sub = df[tk].dropna()
                if not sub.empty:
                    if "Adj Close" not in sub.columns:
                        sub["Adj Close"] = sub["Close"]
                    out[tk] = sub[["Open","High","Low","Close","Adj Close","Volume"]].tail(int(days))
    else:
        sub = df.dropna()
        if not sub.empty:
            if "Adj Close" not in sub.columns:
                sub["Adj Close"] = sub["Close"]
            out[tickers[0]] = sub[["Open","High","Low","Close","Adj Close","Volume"]].tail(int(days))
    return out

def daily_batch_update(
    *,
    db_path: str = DEFAULT_DB_PATH,
    days: int = 730,
    max_workers: int = 6,
    chunk_size: int = 120,
    prefer_provider: str = "yfinance",
    min_rows: int = 260,
    sleep_sec: float = 0.0,
) -> dict:
    """
    毎朝1回実行する想定の「全銘柄OHLCのDB更新」。
    - yfinance multi-download を優先（ヒット数削減）
    - 失敗してもレポートを返す（止まらない）
    """
    t0 = time.time()
    init_db(db_path)

    master = get_jpx_master()
    if master is None or master.empty:
        return {"ok": False, "error": "jpx_master_empty"}

    tickers = master["ticker"].astype(str).drop_duplicates().tolist()

    ok = 0
    fail = 0
    inserted = 0
    failures: List[dict] = []

    # chunking
    chunks = [tickers[i:i+int(chunk_size)] for i in range(0, len(tickers), int(chunk_size))]

    for ci, ch in enumerate(chunks, start=1):
        # already-fresh skip (optional): if many tickers already updated today, still ok to skip
        try:
            if prefer_provider in ("yfinance","auto"):
                pack = _yf_multi_download(ch, days=int(days))
                for tk in ch:
                    df = pack.get(tk)
                    if df is None or df.empty or len(df) < int(min_rows):
                        fail += 1
                        if len(failures) < 25:
                            failures.append({"ticker": tk, "note": "yf_empty_or_few_rows"})
                        continue
                    inserted += db_upsert_ohlc(db_path, tk, df)
                    ok += 1
            else:
                # stooq only (not recommended)
                with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
                    futs = {ex.submit(fetch_daily_stooq, tk, min_rows=min_rows): tk for tk in ch}
                    for fut in as_completed(futs):
                        tk = futs[fut]
                        df, meta = fut.result()
                        if df is None or df.empty:
                            fail += 1
                            if len(failures) < 25:
                                failures.append({"ticker": tk, "meta": meta})
                            continue
                        inserted += db_upsert_ohlc(db_path, tk, df.tail(int(days)))
                        ok += 1
        except Exception as e:
            # chunk failure
            fail += len(ch)
            if len(failures) < 25:
                failures.append({"chunk": ci, "exc": f"{type(e).__name__}: {e}"})

        if sleep_sec and sleep_sec > 0:
            time.sleep(float(sleep_sec))

    # write batch meta (even if partial failures)
    try:
        now_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        jst = _dt.timezone(_dt.timedelta(hours=9))
        now_jst = _dt.datetime.now(tz=jst).isoformat()
        db_set_meta(db_path, "last_batch_utc", now_utc)
        db_set_meta(db_path, "last_batch_jst", now_jst)
        db_set_meta(db_path, "last_batch_ok", "1")
        db_set_meta(db_path, "last_batch_rows_upserted", str(inserted))
        db_set_meta(db_path, "last_batch_ok_count", str(ok))
        db_set_meta(db_path, "last_batch_fail_count", str(fail))
    except Exception:
        pass

    return {
        "ok": True,
        "db_path": str(db_path),
        "tickers_total": int(len(tickers)),
        "ok_count": int(ok),
        "fail_count": int(fail),
        "rows_upserted": int(inserted),
        "sample_failures": failures,
        "elapsed_sec": float(time.time() - t0),
    }

# eager init (safe)
try:
    init_db(DEFAULT_DB_PATH)
except Exception:
    pass
