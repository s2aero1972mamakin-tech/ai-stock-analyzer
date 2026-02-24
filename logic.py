import pandas as pd
import requests
from dataclasses import dataclass
from io import StringIO

@dataclass
class SwingParams:
    rsi_low: float = 35
    rsi_high: float = 70

def _fetch_stooq(symbol: str):
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    r = requests.get(url, timeout=10)
    if r.status_code != 200 or not r.text.strip():
        return pd.DataFrame()
    df = pd.read_csv(StringIO(r.text))
    if "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def _period_start(period):
    now = pd.Timestamp.now().tz_localize(None)
    if period.endswith("mo"):
        return now - pd.DateOffset(months=int(period[:-2]))
    if period.endswith("y"):
        return now - pd.DateOffset(years=int(period[:-1]))
    return now - pd.DateOffset(months=6)

def get_market_data(symbol, period):
    df = _fetch_stooq(symbol.replace(".T",".jp"))
    if df.empty:
        return df
    start = _period_start(period)
    return df.loc[df.index >= start]

def scan_swing_candidates(budget_yen, period="6mo"):
    universe = ["7203.jp","6758.jp","8306.jp"]
    results = []
    for s in universe:
        df = get_market_data(s, period)
        if df.empty:
            continue
        price = df["Close"].iloc[-1]
        if price * 100 > budget_yen:
            continue
        results.append({"ticker": s, "price": round(price,2)})
    return results
