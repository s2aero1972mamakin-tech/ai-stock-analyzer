# -*- coding: utf-8 -*-
"""
JPX Swing Auto Scanner — ULTIMATE FAST ENGINE
- Stooq 全銘柄一括DL対応
- セクター強度モデル搭載
- 高速メモリ内処理
- 実戦用ランキング
"""

from __future__ import annotations
import dataclasses
import math
import requests
import pandas as pd
import numpy as np
from io import StringIO
from dataclasses import dataclass
from typing import Dict, List, Optional


# =========================
# PARAMS
# =========================

@dataclass(frozen=True)
class SwingParams:
    require_sma25_over_sma75: bool = True
    rsi_low: float = 40.0
    rsi_high: float = 75.0
    atr_mult_stop: float = 2.0
    tp1_r: float = 1.0
    tp2_r: float = 3.0
    risk_pct: float = 0.01


# =========================
# JPX MASTER
# =========================

def get_jpx_master() -> pd.DataFrame:
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    df = pd.read_excel(url)
    df = df[df["33業種区分"].notna()]
    df["ticker"] = df["コード"].astype(str).str.zfill(4) + ".T"
    df = df.rename(columns={"銘柄名": "name", "33業種区分": "sector"})
    return df[["ticker", "name", "sector"]].drop_duplicates()


# =========================
# STOOQ BULK DOWNLOAD
# =========================

def download_all_stooq() -> pd.DataFrame:
    """
    日本株デイリーを一括取得
    """
    url = "https://stooq.pl/db/l/?g=jp&i=d"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])
    return df


# =========================
# INDICATORS
# =========================

def add_indicators(df):
    df["SMA25"] = df["Close"].rolling(25).mean()
    df["SMA75"] = df["Close"].rolling(75).mean()

    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        (df["High"] - df["Low"]),
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()
    df["RET_3M"] = df["Close"].pct_change(60)
    return df


# =========================
# SECTOR STRENGTH
# =========================

def sector_strength(master, market):
    merged = market.merge(master, left_on="Symbol", right_on="ticker")
    latest = merged.groupby("Symbol").tail(1)

    sec = latest.groupby("sector").agg({
        "RET_3M": "mean",
        "RSI": "mean",
        "ATR": "mean",
        "Volume": "mean"
    }).reset_index()

    sec["score"] = (
        sec["RET_3M"] * 40 +
        sec["RSI"] / 100 * 30 +
        (1 / (sec["ATR"] + 1e-9)) * 20 +
        np.log1p(sec["Volume"]) * 10
    )

    return sec.sort_values("score", ascending=False)


# =========================
# STOCK SCORING
# =========================

def stock_score(row):
    trend = 1 if row["SMA25"] > row["SMA75"] else 0
    rsi_score = 1 - abs(row["RSI"] - 60) / 60
    vol_score = np.log1p(row["Volume"])
    ret_score = row["RET_3M"]

    return (
        trend * 0.3 +
        rsi_score * 0.2 +
        vol_score * 0.2 +
        ret_score * 0.3
    )


# =========================
# MAIN ENGINE
# =========================

def scan_swing_candidates(top_n=10, sector_top_n=5):

    master = get_jpx_master()
    market = download_all_stooq()

    # JPX銘柄のみに絞る
    market = market[market["Symbol"].isin(master["ticker"])]

    # インジケータ
    market = market.groupby("Symbol").apply(add_indicators).reset_index(drop=True)

    # セクター強度算出
    sec_rank = sector_strength(master, market)
    top_sectors = sec_rank.head(sector_top_n)["sector"].tolist()

    # 上位セクター銘柄のみ
    merged = market.merge(master, left_on="Symbol", right_on="ticker")
    latest = merged.groupby("Symbol").tail(1)
    filtered = latest[latest["sector"].isin(top_sectors)].copy()

    filtered["score"] = filtered.apply(stock_score, axis=1)

    ranked = filtered.sort_values("score", ascending=False).head(top_n)

    return {
        "sector_ranking": sec_rank,
        "candidates": ranked[[
            "Symbol", "name", "sector",
            "Close", "RSI", "ATR", "RET_3M", "score"
        ]]
    }
