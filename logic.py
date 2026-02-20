import yfinance as yf
import pandas as pd
import google.generativeai as genai
import pytz
import time
from datetime import datetime
import json

TOKYO = pytz.timezone("Asia/Tokyo")

# ==========================================
# AIモデル取得ヘルパー（動的探索アルゴリズム）
# ==========================================
def get_active_model(api_key: str):
    genai.configure(api_key=api_key)
    try:
        # サーバーに「現在この環境で利用可能なモデル一覧」を直接問い合わせる
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 優先順位（最新Pro ＞ Flash ＞ 旧Pro）で、実際に存在するものだけを自動選択
        for target in ["models/gemini-1.5-pro-latest", "models/gemini-1.5-pro", "models/gemini-1.5-flash", "models/gemini-pro"]:
            if target in available_models:
                return target
                
        # 上記がなくても、リストに使えるものがあればそれを強制使用
        if available_models:
            return available_models[0]
            
        return "models/gemini-1.5-flash" # 取得失敗時の安全なフォールバック
    except Exception:
        return "models/gemini-1.5-flash"

# ==========================================
# データ取得・計算ロジック
# ==========================================
def _yahoo_chart(ticker="8306.T", rng="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=rng, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        return None

def get_market_data(ticker="8306.T", rng="1y", interval="1d"):
    return _yahoo_chart(ticker, rng, interval)

def get_latest_quote(ticker="8306.T"):
    df = _yahoo_chart(ticker, rng="5d", interval="1m")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None

def calculate_indicators(df: pd.DataFrame, benchmark_raw: pd.DataFrame = None) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_25"] = df["Close"].rolling(window=25).mean()
    df["SMA_75"] = df["Close"].rolling(window=75).mean()
    
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
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
    
    return {
        "short": {"status": short_status, "color": short_color},
        "mid": {"status": mid_status, "color": mid_color}
    }

# ==========================================
# AI分析ロジック（安全装置付き）
# ==========================================
def get_ai_range(api_key: str, ctx: dict):
    model_name = get_active_model(api_key)
    model = genai.GenerativeModel(model_name)
    p = ctx.get('price', 0.0)
    prompt = f"""
あなたは日本株のテクニカルアナリストです。
現在の株価 {p:.1f} 円、ATR {ctx.get('atr',0.0):.2f}、RSI {ctx.get('rsi',50):.1f} です。
今後1週間の想定最高値(high)と最安値(low)をJSONで出力してください。
フォーマット: {{"high": 0000.0, "low": 0000.0, "why": "理由"}}
"""
    try:
        res = model.generate_content(prompt).text
        s = res.find("{")
        e = res.rfind("}")
        return json.loads(res[s:e+1]) if s!=-1 else {"high": p*1.05, "low": p*0.95, "why": "JSONパース失敗"}
    except Exception as e:
        return {"high": p*1.05, "low": p*0.95, "why": f"取得エラー: {str(e)}"}

def get_ai_analysis(api_key: str, ctx: dict):
    model_name = get_active_model(api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
あなたは日本株投資に精通した「バリュー株専門のファンドマネージャー」です。
以下のデータから、指定された銘柄の投資判断を下してください。

【対象銘柄データ】
- 銘柄・証券コード: {ctx.get('pair_label', '不明')}
- 現在株価: {ctx.get('price', 0.0):.1f} 円
- 日経平均(参考): {ctx.get('us10y', 0.0):.1f} 円
- ボラティリティ(ATR): {ctx.get('atr', 0.0):.2f} 円
- RSI(14日): {ctx.get('rsi', 50):.1f}
- 25日線乖離率: {ctx.get('sma_diff', 0.0):.2f}%

【分析・戦略作成の絶対ルール】
1. この銘柄が現在押し目買いの好機か、バリュートラップ（下落継続の罠）かを厳しく判定してください。
2. 日経平均の動向と個別株のトレンドを比較してください。
3. 1週間〜1ヶ月のスイングトレードを前提としてください。
4. 【重要】過去の歴史的な上値抵抗線（節目となるキリの良い株価など）を強く意識し、非現実的な一直線の上昇を想定しないでください。
"""
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ 【AI通信エラー】モデル({model_name})の呼び出しに失敗しました。\n詳細: {str(e)}"

def get_ai_order_strategy(api_key: str, ctx: dict):
    model_name = get_active_model(api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
あなたは冷徹な執行責任者です。中間的な回答や迷いは一切不要です。
以下のデータと直近のレポート内容に基づき、具体的な注文戦略を作成してください。

【対象銘柄データ】
- 銘柄・証券コード: {ctx.get('pair_label', '不明')}
- 現在株価: {ctx.get('price', 0.0):.1f} 円
- ボラティリティ(ATR): {ctx.get('atr', 0.0):.2f} 円

【命令】
エントリーの確信が持てる時のみ「EXECUTE: 100%」、少しでも疑念がある時は「ABORT: 0%」と明確に出力してください。
EXECUTEの場合、必ず以下の3点（現実的な数値）を提示してください。
- ENTRY (指値または現在値)
- LIMIT (利確目標: 歴史的節目や抵抗線の「少し手前」に設定すること)
- STOP (損切: ATRを考慮した許容範囲。現値から不自然に離さないこと)
"""
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ 【AI通信エラー】モデル({model_name})の呼び出しに失敗しました。\n詳細: {str(e)}"

def get_ai_portfolio(api_key: str, ctx: dict):
    model_name = get_active_model(api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
あなたは日本株のポートフォリオマネージャーです。
現在の銘柄({ctx.get('pair_label', '不明')})について、週末や月末を跨いで保有（ホールド）すべきか、
それとも金曜日に一旦手仕舞いすべきか、現在の相場環境（株価 {ctx.get('price', 0.0):.1f}円）を元に判断してください。
"""
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ 【AI通信エラー】モデル({model_name})の呼び出しに失敗しました。\n詳細: {str(e)}"

# ==========================================
# 全自動スクリーニング・エンジン（解決策B）
# ==========================================
def auto_scan_value_stocks():
    """
    プログラムが自動で複数銘柄をスキャンし、テクニカル的に「買い」のトップ銘柄を抽出する。
    ※API制限を避けるため、代表的な高配当・低位・大型株をターゲットとする。
    """
    # スキャン対象のユニバース（ここに監視させたい証券コードをいくらでも追加可能）
    scan_list = [
        "8306.T", "8411.T", "7182.T", "8410.T", # 銀行
        "7203.T", "7201.T", "7267.T", "7011.T", # 自動車・重工
        "9432.T", "9984.T", "9433.T", "9434.T", # 通信・IT
        "5020.T", "5401.T", "3407.T", "6178.T", # 素材・郵政
        "2914.T", "8593.T", "8058.T", "8002.T"  # 高配当・商社
    ]
    
    candidates = []
    
    for ticker in scan_list:
        try:
            # 過去3ヶ月のデータを取得（スキャン用なので軽量化）
            df = _yahoo_chart(ticker, rng="3mo", interval="1d")
            if df is None or df.empty or len(df) < 30:
                continue
                
            # テクニカル指標の計算
            df["SMA_25"] = df["Close"].rolling(window=25).mean()
            delta = df["Close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
            df["RSI"] = 100 - (100 / (1 + rs))
            
            latest = df.iloc[-1]
            price = latest["Close"]
            sma25 = latest["SMA_25"]
            rsi = latest["RSI"]
            
            # 【絶対条件】ロボットが「買い候補」とする条件
            # 1. 株価が25日線より上にある（上昇トレンド）または、RSIが40以下（売られすぎの反発狙い）
            # 2. 株価が3000円以下（資金効率を考慮）
            if price <= 3000 and (price > sma25 or rsi <= 40):
                # スコアリング（RSIが低く、かつトレンドが上向きのものを高く評価）
                score = (50 - rsi) + ((price - sma25) / sma25 * 100)
                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "rsi": rsi,
                    "score": score
                })
        except Exception:
            continue
            
    # スコアが高い順に並び替え、トップ3銘柄だけを返す
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:3]
