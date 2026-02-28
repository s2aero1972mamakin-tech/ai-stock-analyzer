# logic.py
# JPX Swing Auto Scanner - STABLE5d FIXED FULL VERSION

import dataclasses
import pandas as pd
import numpy as np
import requests
from dataclasses import dataclass

@dataclass(frozen=True)
class SwingParams:
    pass

def get_jpx_master():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        df = pd.read_excel(url)
        df["ticker"] = df["コード"].astype(str).str.zfill(4) + ".T"
        df = df.rename(columns={"銘柄名": "name", "33業種区分": "sector"})
        return df[["ticker", "name", "sector"]]
    except Exception:
        return pd.DataFrame()

def get_market_data(ticker):
    try:
        sym = ticker.replace(".T", ".jp")
        url = f"https://stooq.pl/q/d/l/?s={sym}&i=d"
        df = pd.read_csv(url)
        return df
    except Exception:
        return pd.DataFrame()

def scan_swing_candidates(budget_yen, top_n=3, progress_callback=None):
    master = get_jpx_master()
    results = []
    stats = {"universe": 0, "pass": 0}

    if master.empty:
        return {"candidates": [], "filter_stats": stats}

    tickers = master["ticker"].tolist()
    stats["universe"] = len(tickers)

    for i, t in enumerate(tickers, start=1):
        if progress_callback:
            progress_callback(i, len(tickers), t)

        df = get_market_data(t)
        if df.empty:
            continue

        price = float(df.iloc[-1]["Close"])
        if price * 100 > budget_yen:
            continue

        stats["pass"] += 1
        results.append({
            "ticker": t,
            "price": price
        })

        if len(results) >= top_n:
            break

    return {"candidates": results, "filter_stats": stats}
