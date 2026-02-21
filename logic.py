import yfinance as yf
import pandas as pd
from openai import OpenAI
import pytz
import time
from datetime import datetime
import json
import streamlit as st

TOKYO = pytz.timezone("Asia/Tokyo")

@st.cache_data(ttl=86400, show_spinner=False)
def get_jpx_master() -> pd.DataFrame:
    try:
        url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        df = pd.read_excel(url, engine='xlrd')
        df = df[df['33業種区分'] != '-'].copy()
        df['ticker'] = df['コード'].astype(str) + '.T'
        df = df.rename(columns={'銘柄名': 'name', '33業種区分': 'sector'})
        return df[['ticker', 'name', 'sector']]
    except:
        return pd.DataFrame()

def get_company_name(ticker: str) -> str:
    df_master = get_jpx_master()
    if not df_master.empty:
        match = df_master[df_master['ticker'] == ticker]
        if not match.empty:
            return match.iloc[0]['name']
    return ticker

def call_openai(api_key: str, system_prompt: str, user_prompt: str) -> str:
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI API エラー: {str(e)}"

def get_promising_sectors(api_key: str, all_sectors: list) -> list:
    sectors_str = ", ".join(all_sectors)
    system_prompt = "あなたはマクロ経済ストラテジストです。必ずJSONの配列形式のみで回答してください。"
    user_prompt = f"金利・為替から現在有望な業種を2つ選んでください。選択肢: {sectors_str}"
    try:
        res = call_openai(api_key, system_prompt, user_prompt)
        s, e = res.find("["), res.rfind("]")
        if s != -1 and e != -1:
            chosen = json.loads(res[s:e+1])
            return [c for c in chosen if c in all_sectors]
    except: pass
    return ["電気機器", "銀行業"]

def _yahoo_chart(ticker, rng="3mo", interval="1d"):
    try:
        df = yf.download(ticker, period=rng, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return df
    except: return None

def get_market_data(ticker="8306.T", rng="1y", interval="1d"):
    return _yahoo_chart(ticker, rng, interval)

def calculate_indicators(df: pd.DataFrame, benchmark_raw: pd.DataFrame = None) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_25"] = df["Close"].rolling(window=25).mean()
    df["SMA_75"] = df["Close"].rolling(window=75).mean()
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()
    df["SMA_DIFF"] = (df["Close"] - df["SMA_25"]) / df["SMA_25"] * 100
    df["BENCHMARK"] = benchmark_raw["Close"].reindex(df.index, method="ffill") if benchmark_raw is not None else 0.0
    return df.dropna()

def judge_condition(price, sma5, sma25, sma75, rsi):
    short = {"status": "上昇継続 (SMA5上)", "color": "blue"} if price > sma5 else {"status": "勢い鈍化 (SMA5下)", "color": "red"}
    mid = {"status": "静観", "color": "gray"}
    if sma25 > sma75 and rsi < 70: mid = {"status": "上昇トレンド (押し目買い)", "color": "blue"}
    elif rsi <= 30: mid = {"status": "売られすぎ (反発警戒)", "color": "orange"}
    return {"short": short, "mid": mid}

def auto_scan_value_stocks(api_key: str, progress_callback=None):
    df_master = get_jpx_master()
    if df_master.empty: return ["エラー"], []
    all_sectors = df_master['sector'].dropna().unique().tolist()
    target_sectors = get_promising_sectors(api_key, all_sectors)
    target_df = df_master[df_master['sector'].isin(target_sectors)]
    scan_list = target_df.to_dict('records')
    candidates = []
    for i, item in enumerate(scan_list):
        ticker, comp_name = item['ticker'], item['name']
        try:
            if progress_callback: progress_callback(i + 1, len(scan_list), f"{ticker} {comp_name}")
            df = _yahoo_chart(ticker, rng="3mo", interval="1d")
            if df is None or len(df) < 30: continue
            df["SMA_5"], df["SMA_25"] = df["Close"].rolling(window=5).mean(), df["Close"].rolling(window=25).mean()
            delta = df["Close"].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
            rsi = 100 - (100 / (1 + (rs)))
            latest = df.iloc[-1]
            if (latest["SMA_5"] > latest["SMA_25"] and 40 <= rsi <= 60) or (rsi <= 30):
                score = ((latest["Close"] - latest["SMA_25"]) / latest["SMA_25"] * 100) + (70 - rsi)
                candidates.append({"ticker": ticker, "name": comp_name, "price": latest["Close"], "rsi": rsi, "score": score})
        except: continue
    return target_sectors, sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]

def get_ai_analysis(api_key: str, ctx: dict):
    return call_openai(api_key, "あなたは実戦派ファンドマネージャーです。", f"銘柄:{ctx['pair_label']}のデータを分析してください。株価:{ctx['price']}円, RSI:{ctx['rsi']:.1f}, ATR:{ctx['atr']:.2f}")

def get_ai_order_strategy(api_key: str, ctx: dict):
    return call_openai(api_key, "あなたは冷徹な執行責任者です。", f"銘柄:{ctx['pair_label']}のENTRY, LIMIT, STOP価格を算出して命令書を作成してください。株価:{ctx['price']}円, RSI:{ctx['rsi']:.1f}")

def get_ai_portfolio(api_key: str, ctx: dict):
    return call_openai(api_key, "あなたはポートフォリオマネージャーです。", f"銘柄:{ctx['pair_label']}の週末跨ぎの是非を判断してください。")
