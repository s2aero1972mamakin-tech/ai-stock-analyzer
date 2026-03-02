
import pandas as pd
import numpy as np
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

def get_jpx_master():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    df = pd.read_excel(url)
    df = df[df["33業種区分"].notna()]
    df["ticker"] = df["コード"].astype(str).str.zfill(4) + ".T"
    df = df.rename(columns={"銘柄名": "name", "33業種区分": "sector"})
    return df[["ticker", "name", "sector"]]

def get_stooq_single(ticker):
    code = ticker.replace(".T","") + ".jp"
    url = f"https://stooq.pl/q/d/l/?s={code}&i=d"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df["RET_3M"] = df["Close"].pct_change(60)
        df["SMA25"] = df["Close"].rolling(25).mean()
        df["SMA75"] = df["Close"].rolling(75).mean()
        tr = (df["High"] - df["Low"]).rolling(14).mean()
        df["ATR"] = tr
        latest = df.tail(1)
        if latest.empty:
            return None
        return latest.iloc[0]
    except:
        return None

def sector_strength(master, rows):
    df = pd.DataFrame(rows)
    merged = df.merge(master, left_on="Symbol", right_on="ticker")
    sec = merged.groupby("sector")["RET_3M"].mean().reset_index()
    sec["score"] = sec["RET_3M"] * 100
    return sec.sort_values("score", ascending=False)

def scan_swing_candidates(top_n=10, sector_top_n=5):
    master = get_jpx_master()
    tickers = master["ticker"].tolist()[:800]  # 負荷軽減

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(get_stooq_single, tickers))

    for t, r in zip(tickers, results):
        if r is None:
            continue
        r = r.to_dict()
        r["Symbol"] = t
        rows.append(r)

    if not rows:
        return {"sector_ranking": pd.DataFrame(), "candidates": pd.DataFrame()}

    sec_rank = sector_strength(master, rows)
    top_sectors = sec_rank.head(sector_top_n)["sector"].tolist()

    df = pd.DataFrame(rows)
    df = df.merge(master, left_on="Symbol", right_on="ticker")
    df = df[df["sector"].isin(top_sectors)]
    df["score"] = df["RET_3M"]

    ranked = df.sort_values("score", ascending=False).head(top_n)

    return {
        "sector_ranking": sec_rank,
        "candidates": ranked[["Symbol","name","sector","Close","ATR","RET_3M","score"]]
    }

def build_trade_plan(price, atr, capital, risk_pct):
    stop = price - atr * 2
    risk_per_share = price - stop
    risk_budget = capital * risk_pct
    shares = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    shares = (shares // 100) * 100
    tp1 = price + risk_per_share
    tp2 = price + risk_per_share * 3
    return {
        "entry": price,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "shares": shares
    }
