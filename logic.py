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
# WALK FORWARD
# =============================

def walk_forward_score(df):

    window_train = 400
    window_test = 120

    scores = []

    for start in range(0, len(df)-window_train-window_test, window_test):

        train = df.iloc[start:start+window_train]
        test = df.iloc[start+window_train:start+window_train+window_test]

        train_ret = train["Close"].pct_change().mean()
        test_ret = test["Close"].pct_change().mean()

        scores.append(test_ret)

    if not scores:
        return 0

    return np.mean(scores)

# =============================
# MONTE CARLO DD
# =============================

def monte_carlo_dd(returns, n_sim=2000):

    max_dds = []

    for _ in range(n_sim):
        shuffled = np.random.permutation(returns)
        equity = np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dds.append(dd.min())

    return np.percentile(max_dds, 95)

# =============================
# PORTFOLIO RISK OF RUIN
# =============================

def portfolio_ror(win_rate, rr, capital_units):

    edge = win_rate - (1-win_rate)/rr
    if edge <= 0:
        return 1.0

    return ((1-edge)/(1+edge))**capital_units

# =============================
# CORRELATION FILTER
# =============================

def correlation_filter(price_dict, selected, threshold=0.7):

    final = []

    for sym in selected:
        keep = True
        for f in final:
            corr = price_dict[sym].pct_change().corr(price_dict[f].pct_change())
            if abs(corr) > threshold:
                keep = False
                break
        if keep:
            final.append(sym)

    return final

# =============================
# MAIN ENGINE
# =============================

def scan_engine(top_n=5, sector_top_n=5):

    master = get_jpx_master()
    tickers = master["ticker"].tolist()[:800]

    rows = []
    price_dict = {}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_stooq, tickers))

    for t, df in zip(tickers, results):

        if df is None or len(df) < 600:
            continue

        df = add_indicators(df)

        latest = df.tail(1)
        if latest.empty:
            continue

        row = latest.iloc[0].to_dict()
        row["Symbol"] = t
        row["wf_score"] = walk_forward_score(df)

        returns = df["Close"].pct_change().dropna().values
        row["mc_dd"] = monte_carlo_dd(returns)

        rows.append(row)
        price_dict[t] = df["Close"]

    if not rows:
        return None

    df_all = pd.DataFrame(rows)

    # Sector strength
    merged = df_all.merge(master, left_on="Symbol", right_on="ticker")
    sec_rank = merged.groupby("sector")["RET_3M"].mean().reset_index()
    sec_rank["score"] = sec_rank["RET_3M"]*100
    sec_rank = sec_rank.sort_values("score", ascending=False)

    top_sectors = sec_rank.head(sector_top_n)["sector"].tolist()

    df_all = merged[merged["sector"].isin(top_sectors)]

    df_all["score"] = (
        df_all["wf_score"]*0.4 +
        df_all["RET_3M"]*0.3 -
        abs(df_all["mc_dd"])*0.3
    )

    ranked = df_all.sort_values("score", ascending=False).head(top_n*2)

    selected = correlation_filter(price_dict, ranked["Symbol"].tolist())

    final = ranked[ranked["Symbol"].isin(selected)].head(top_n)

    return {
        "sector_ranking": sec_rank,
        "candidates": final
    }

# =============================
# ORDER BUILDER
# =============================

def build_orders(df, capital, risk_pct, current_dd, market_vol):

    orders = []

    dd_factor = 1.0
    if current_dd > 0.1:
        dd_factor = 0.6
    if current_dd > 0.15:
        dd_factor = 0.3

    vol_factor = 1.0
    if market_vol > 1.5:
        vol_factor = 0.6

    for _, row in df.iterrows():

        price = row["Close"]
        atr = row["ATR"]

        stop = price - atr*2
        r = price - stop

        risk_budget = capital*risk_pct*dd_factor*vol_factor
        shares = int(risk_budget/r) if r>0 else 0
        shares = (shares//100)*100

        orders.append({
            "Symbol": row["Symbol"],
            "entry": price,
            "stop": stop,
            "tp1": price+r,
            "tp2": price+r*3,
            "shares": shares
        })

    return pd.DataFrame(orders)
