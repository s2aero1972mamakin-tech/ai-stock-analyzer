# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import requests
import math
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

# =============================
# JPX MASTER
# =============================

def get_jpx_master():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    df = pd.read_excel(url)
    df = df[df["33業種区分"].notna()]
    df["ticker"] = df["コード"].astype(str).str.zfill(4) + ".T"
    df = df.rename(columns={"銘柄名":"name","33業種区分":"sector"})
    return df[["ticker","name","sector"]]

# =============================
# STOOQ DATA
# =============================

def fetch_stooq(ticker):
    try:
        code = ticker.replace(".T","") + ".jp"
        url = f"https://stooq.pl/q/d/l/?s={code}&i=d"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        return df
    except:
        return None

# =============================
# INDICATORS
# =============================

def add_indicators(df):
    df["RET_3M"] = df["Close"].pct_change(60)
    df["SMA25"] = df["Close"].rolling(25).mean()
    df["SMA75"] = df["Close"].rolling(75).mean()
    df["VOL_AVG"] = df["Volume"].rolling(20).mean()
    tr = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR"] = tr
    return df

# =============================
# SECTOR STRENGTH
# =============================

def calc_sector_strength(master, data_rows):
    df = pd.DataFrame(data_rows)
    merged = df.merge(master, left_on="Symbol", right_on="ticker")
    sec = merged.groupby("sector")["RET_3M"].mean().reset_index()
    sec["score"] = sec["RET_3M"] * 100
    return sec.sort_values("score", ascending=False)

# =============================
# CORRELATION FILTER
# =============================

def correlation_filter(price_dict, selected, threshold=0.7):
    final = []
    for sym in selected:
        keep = True
        for f in final:
            c = price_dict[sym].pct_change().corr(price_dict[f].pct_change())
            if abs(c) > threshold:
                keep = False
                break
        if keep:
            final.append(sym)
    return final

# =============================
# DRAW DOWN CONTROL
# =============================

def adjust_risk_by_dd(current_dd):
    if current_dd < 0.05:
        return 1.0
    elif current_dd < 0.1:
        return 0.7
    elif current_dd < 0.15:
        return 0.4
    else:
        return 0.0

# =============================
# MAIN SCAN ENGINE
# =============================

def scan_engine(top_n=10, sector_top_n=5):

    master = get_jpx_master()
    tickers = master["ticker"].tolist()[:800]  # 初期負荷軽減

    rows = []
    price_dict = {}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_stooq, tickers))

    for t, df in zip(tickers, results):
        if df is None:
            continue
        df = add_indicators(df)
        latest = df.tail(1)
        if latest.empty:
            continue
        row = latest.iloc[0].to_dict()
        row["Symbol"] = t
        rows.append(row)
        price_dict[t] = df["Close"]

    if not rows:
        return None

    sec_rank = calc_sector_strength(master, rows)
    top_sectors = sec_rank.head(sector_top_n)["sector"].tolist()

    df = pd.DataFrame(rows)
    df = df.merge(master, left_on="Symbol", right_on="ticker")
    df = df[df["sector"].isin(top_sectors)]

    # スコア計算
    df["score"] = (
        df["RET_3M"] * 0.4 +
        (df["SMA25"] > df["SMA75"]).astype(int) * 0.3 +
        np.log1p(df["VOL_AVG"]) * 0.3
    )

    ranked = df.sort_values("score", ascending=False).head(top_n*2)

    selected = ranked["Symbol"].tolist()
    selected = correlation_filter(price_dict, selected)

    final = df[df["Symbol"].isin(selected)].head(top_n)

    return {
        "sector_ranking": sec_rank,
        "candidates": final,
        "price_dict": price_dict
    }

# =============================
# TRADE PLAN
# =============================

def build_orders(df, capital, risk_pct, current_dd):

    risk_adj = adjust_risk_by_dd(current_dd)
    orders = []

    for _, row in df.iterrows():

        price = row["Close"]
        atr = row["ATR"]
        stop = price - atr * 2
        r = price - stop

        risk_budget = capital * risk_pct * risk_adj
        shares = int(risk_budget / r) if r > 0 else 0
        shares = (shares // 100) * 100

        orders.append({
            "Symbol": row["Symbol"],
            "name": row["name"],
            "entry": price,
            "stop": stop,
            "tp1": price + r,
            "tp2": price + r*3,
            "shares": shares
        })

    return pd.DataFrame(orders)
