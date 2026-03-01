
# logic.py
import time
import requests
import pandas as pd
from io import StringIO

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

def get_jpx_master():
    try:
        df = pd.read_excel(JPX_URL)
        df["ticker"] = df["コード"].astype(str).str.zfill(4) + ".T"
        return df[["ticker"]]
    except Exception:
        return pd.DataFrame()

def get_market_data(ticker):
    sym = ticker.replace(".T", ".jp")
    url = f"https://stooq.pl/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None, "http_error"
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return None, "empty_data"
        return df, None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception:
        return None, "parse_error"

def scan_swing_candidates(budget_yen, top_n, start_index=0, progress_callback=None, diag=None):

    master = get_jpx_master()
    if master.empty:
        return {"candidates": [], "filter_stats": {}}

    tickers = master["ticker"].tolist()
    total = len(tickers)
    candidates = []

    start_time = time.time()

    for idx in range(start_index, total):
        ticker = tickers[idx]

        if progress_callback:
            progress_callback(idx+1, total)

        if diag:
            diag["cursor_index"] = idx

        fetch_start = time.time()
        df, reason = get_market_data(ticker)
        fetch_sec = time.time() - fetch_start

        if diag:
            diag["timing"]["fetch_sec"] += fetch_sec

        if reason:
            if diag:
                diag["fail_data_reason"][reason] = diag["fail_data_reason"].get(reason, 0) + 1
            continue

        price = float(df.iloc[-1]["Close"])
        if price * 100 > budget_yen:
            continue

        candidates.append({"ticker": ticker, "price": price})

        if len(candidates) >= top_n:
            break

    if diag:
        diag["timing"]["total_sec"] = time.time() - start_time

    return {
        "candidates": candidates,
        "filter_stats": {
            "universe": total,
            "returned": len(candidates)
        }
    }
