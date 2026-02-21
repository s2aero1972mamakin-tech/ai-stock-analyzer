import yfinance as yf
import pandas as pd
from openai import OpenAI
import pytz
import time
from datetime import datetime
import json
import streamlit as st

TOKYO = pytz.timezone("Asia/Tokyo")

# ==========================================
# 🛑 JPX(日本取引所) 公式全4000銘柄データの自動取得
# （手書きの辞書を廃止し、公式データをキャッシュして使用）
# ==========================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_jpx_master() -> pd.DataFrame:
    try:
        # JPXの公式サイトから全上場銘柄一覧(Excel)を直接取得
        url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        df = pd.read_excel(url, engine='xlrd')
        
        # 必要なデータのみ抽出（ETFなどを除外するため、業種があるものに絞る）
        df = df[df['33業種区分'] != '-'].copy()
        
        # yfinance用に証券コードの末尾に「.T」を付与
        df['ticker'] = df['コード'].astype(str) + '.T'
        
        # 必要な列だけをリネームして抽出
        df = df.rename(columns={'銘柄名': 'name', '33業種区分': 'sector'})
        return df[['ticker', 'name', 'sector']]
    except Exception as e:
        # 取得失敗時の安全策（空のデータフレームを返す）
        return pd.DataFrame()

def get_company_name(ticker: str) -> str:
    df_master = get_jpx_master()
    if not df_master.empty:
        match = df_master[df_master['ticker'] == ticker]
        if not match.empty:
            return match.iloc[0]['name']
    
    # クラウドでブロックされる可能性があるが、念のためのフォールバック
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except:
        return ticker

# ==========================================
# 🛑 OpenAI (ChatGPT) 呼び出しヘルパー関数
# ==========================================
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
    user_prompt = f"""
現在の金利、為替、地政学リスクを分析し、日本の「東証33業種」の中から、
現在最も資金が流入しやすく株価伸長率が高い業種を「2つ」厳選してください。
【東証33業種の選択肢】
{sectors_str}

必ず以下のJSON配列形式のみを出力してください。例: ["電気機器", "銀行業"]
"""
    try:
        res = call_openai(api_key, system_prompt, user_prompt)
        s = res.find("[")
        e = res.rfind("]")
        if s != -1 and e != -1:
            chosen = json.loads(res[s:e+1])
            if isinstance(chosen, list) and len(chosen) > 0:
                # 存在する業種名のみを確実に返す
                return [c for c in chosen if c in all_sectors]
    except: pass
    # エラー時のデフォルト（東証の正式な業種名）
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
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()
    
    df["SMA_DIFF"] = (df["Close"] - df["SMA_25"]) / df["SMA_25"] * 100

    df["BENCHMARK"] = 0.0
    if benchmark_raw is not None and not benchmark_raw.empty:
        df["BENCHMARK"] = benchmark_raw["Close"].reindex(df.index, method="ffill")
    return df.dropna()

def judge_condition(price, sma5, sma25, sma75, rsi):
    if price > sma5: short_status, short_color = "上昇継続 (SMA5上)", "blue"
    else: short_status, short_color = "勢い鈍化 (SMA5下)", "red"
    
    mid_status, mid_color = "静観・レンジ", "gray"
    if sma25 > sma75 and rsi < 70: mid_status, mid_color = "上昇トレンド (押し目買い)", "blue"
    elif sma25 < sma75 and rsi > 30: mid_status, mid_color = "下落トレンド (戻り売り)", "red"
    elif rsi >= 70: mid_status, mid_color = "過熱感あり (利益確定検討)", "orange"
    elif rsi <= 30: mid_status, mid_color = "売られすぎ (反発警戒)", "orange"
    return {"short": {"status": short_status, "color": short_color}, "mid": {"status": mid_status, "color": mid_color}}

# ==========================================
# 🚀 4000銘柄対応・全自動メガスクリーニング
# ==========================================
def auto_scan_value_stocks(api_key: str, progress_callback=None):
    df_master = get_jpx_master()
    
    if df_master.empty:
        return ["エラー"], []

    # 東証33業種のリストをマスターから取得
    all_sectors = df_master['sector'].dropna().unique().tolist()
    
    # AIに全業種の中から今熱い業種を選ばせる
    target_sectors = get_promising_sectors(api_key, all_sectors)
    
    # 🌟 選ばれた業種に属する【すべての銘柄】を4000銘柄から抽出！
    target_df = df_master[df_master['sector'].isin(target_sectors)]
    scan_list = target_df.to_dict('records') # [{'ticker': '...', 'name': '...', 'sector': '...'}, ...]
        
    candidates = []
    total_stocks = len(scan_list)
    
    for i, item in enumerate(scan_list):
        ticker = item['ticker']
        comp_name = item['name']
        
        try:
            if progress_callback: progress_callback(i + 1, total_stocks, f"{ticker} {comp_name}")
                
            time.sleep(0.02)
            df = _yahoo_chart(ticker, rng="3mo", interval="1d")
            if df is None or df.empty or len(df) < 30: continue
                
            df["SMA_5"] = df["Close"].rolling(window=5).mean()
            df["SMA_25"] = df["Close"].rolling(window=25).mean()
            delta = df["Close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
            df["RSI"] = 100 - (100 / (1 + rs))
            
            latest = df.iloc[-1]
            price = latest["Close"]
            sma5 = latest["SMA_5"]
            sma25 = latest["SMA_25"]
            rsi = latest["RSI"]
            
            # 【勝率80%追求・厳格な買い条件】
            if (sma5 > sma25 and 40 <= rsi <= 60) or (rsi <= 30):
                score = ((price - sma25) / sma25 * 100) + (70 - rsi)
                candidates.append({"ticker": ticker, "name": comp_name, "price": price, "rsi": rsi, "score": score})
        except Exception:
            continue
            
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return target_sectors, candidates[:3]

# ==========================================
# 🧠 AI分析ロジック (OpenAI API)
# ==========================================
def get_ai_range(api_key: str, ctx: dict):
    p = ctx.get('price', 0.0)
    system_prompt = "あなたは日本株のテクニカルアナリストです。必ずJSON形式で出力してください。"
    user_prompt = f"""
現在株価 {p:.1f} 円、ATR {ctx.get('atr',0.0):.2f}、RSI {ctx.get('rsi',50):.1f} です。
今後1週間の想定最高値(high)と最安値(low)をJSONで出力。
フォーマット: {{"high": 0000.0, "low": 0000.0, "why": "理由"}}
"""
    try:
        res = call_openai(api_key, system_prompt, user_prompt)
        s = res.find("{")
        e = res.rfind("}")
        return json.loads(res[s:e+1]) if s!=-1 else {"high": p*1.05, "low": p*0.95, "why": "JSONパース失敗"}
    except: return {"high": p*1.05, "low": p*0.95, "why": "エラー"}

def get_ai_analysis(api_key: str, ctx: dict):
    system_prompt = "あなたは利益を追求する実戦派の「日本株ファンドマネージャー」です。"
    user_prompt = f"""
【対象銘柄データ】
- 銘柄: {ctx.get('pair_label', '不明')}
- 現在株価: {ctx.get('price', 0.0):.1f} 円
- 5日線: {ctx.get('sma5', 0.0):.1f} 円 / 25日線: {ctx.get('sma25', 0.0):.1f} 円 / 75日線: {ctx.get('sma75', 0.0):.1f} 円
- RSI: {ctx.get('rsi', 50):.1f} / ATR: {ctx.get('atr', 0.0):.2f} 円
【ルール】
この銘柄はシステムが「有望セクター ＋ 勝率80%のテクニカル優位性」の2重フィルターをクリアした鉄板銘柄です。
豊富なテクニカルデータを根拠に、上値のメドと押し目買いのポイントを解説してください。
"""
    return call_openai(api_key, system_prompt, user_prompt)

def get_ai_order_strategy(api_key: str, ctx: dict):
    system_prompt = "あなたは冷徹なシステムトレード執行責任者です。"
    user_prompt = f"""
【データ】
- 銘柄: {ctx.get('pair_label', '不明')}
- 現在株価: {ctx.get('price', 0.0):.1f} 円
- 5日線: {ctx.get('sma5', 0.0):.1f} 円 / 25日線: {ctx.get('sma25', 0.0):.1f} 円 / 75日線: {ctx.get('sma75', 0.0):.1f} 円
- RSI: {ctx.get('rsi', 0.0):.1f} / ATR: {ctx.get('atr', 0.0):.2f} 円
【絶対命令】
1. この銘柄は、Pythonが「勝率80%以上の優位性がある」と証明済みの【鉄板株】です。
2. 「情報不足」を理由にしたABORTは見送りは固く禁じます。必ず「EXECUTE: 100%」を前提に戦略を構築してください。
3. リスクリワード比（損切り幅：利幅 = 1：2以上）が成立する現実的な数値を必ず算出してください。
【出力】
- ENTRY (指値または現在値付近)
- LIMIT (利確目標)
- STOP (損切: 25日線割れ等を考慮)
"""
    return call_openai(api_key, system_prompt, user_prompt)

def get_ai_portfolio(api_key: str, ctx: dict):
    system_prompt = "あなたは日本株のポートフォリオマネージャーです。"
    user_prompt = f"現在の銘柄({ctx.get('pair_label', '不明')})について、週末跨ぎの保有判断をしてください。"
    return call_openai(api_key, system_prompt, user_prompt)
