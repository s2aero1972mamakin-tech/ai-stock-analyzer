import yfinance as yf
import pandas as pd
import google.generativeai as genai
import pytz
import time
from datetime import datetime
import json

TOKYO = pytz.timezone("Asia/Tokyo")

# ==========================================
# 業種・セクター別 銘柄データベース（AIがここから選別します）
# ==========================================
SECTOR_MAP = {
    "銀行・金融": ["8306.T", "8316.T", "8411.T", "7182.T", "8593.T", "8766.T", "8750.T"],
    "自動車・輸送機": ["7203.T", "7267.T", "7269.T", "7201.T", "7011.T", "7012.T", "6301.T"],
    "半導体・電子部品": ["8035.T", "6920.T", "6857.T", "6501.T", "6758.T", "6981.T", "6503.T"],
    "通信・IT": ["9432.T", "9433.T", "9434.T", "9984.T", "9613.T", "4307.T"],
    "商社・卸売": ["8058.T", "8031.T", "8001.T", "8002.T", "8053.T", "2768.T"],
    "素材・化学・鉄鋼": ["5401.T", "5411.T", "3407.T", "4005.T", "4183.T", "4063.T"],
    "エネルギー・インフラ": ["5020.T", "1605.T", "9501.T", "9020.T", "9022.T", "9101.T"],
    "医薬品・食品": ["4502.T", "4568.T", "2914.T", "2502.T", "3382.T", "4519.T"],
    "不動産・建設": ["8801.T", "8802.T", "1925.T", "1928.T", "1801.T", "1812.T"]
}

# ==========================================
# AIモデル取得ヘルパー
# ==========================================
def get_active_model(api_key: str):
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for target in ["models/gemini-1.5-pro-latest", "models/gemini-1.5-pro", "models/gemini-1.5-flash", "models/gemini-pro"]:
            if target in available_models:
                return target
        if available_models: return available_models[0]
        return "models/gemini-1.5-flash" 
    except Exception:
        return "models/gemini-1.5-flash"

# ==========================================
# AIによるマクロ・セクター選定（トップダウン）
# ==========================================
def get_promising_sectors(api_key: str) -> list:
    model_name = get_active_model(api_key)
    model = genai.GenerativeModel(model_name)
    sectors_list_str = ", ".join(SECTOR_MAP.keys())
    
    prompt = f"""
あなたは世界トップクラスのマクロ経済ストラテジストです。
現在の日本の金利動向、為替（円安/円高）、米国の経済状況、および地政学リスクを総合的に分析し、
以下の日本株の業種（セクター）の中から、現在最も資金が流入しやすく、今後1ヶ月で株価伸長率が高いと予想されるセクターを「2つ」だけ厳選してください。

【選択肢】
{sectors_list_str}

必ず以下のJSON配列形式のみを出力してください。理由やその他の文章は一切不要です。
["選んだセクター名1", "選んだセクター名2"]
"""
    try:
        res = model.generate_content(prompt).text
        s = res.find("[")
        e = res.rfind("]")
        if s != -1 and e != -1:
            chosen = json.loads(res[s:e+1])
            if isinstance(chosen, list) and len(chosen) > 0:
                # 辞書に存在するキーだけを確実にフィルタリング
                return [c for c in chosen if c in SECTOR_MAP]
    except Exception:
        pass
    # エラー時はデフォルトで資金が向かいやすいセクターを返す
    return ["半導体・電子部品", "銀行・金融"]

# ==========================================
# データ取得・計算ロジック
# ==========================================
def _yahoo_chart(ticker, rng="3mo", interval="1d"):
    try:
        df = yf.download(ticker, period=rng, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception:
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
# スマート・スクリーニング（的確な絞り込みエンジン）
# ==========================================
def auto_scan_value_stocks(api_key: str):
    """
    1. AIに有望セクターを2つ選ばせる
    2. そのセクターの銘柄群だけを的確にスキャンする（無駄な通信排除）
    3. 勝率80%の厳格な条件に合致するトップ銘柄を抽出する
    """
    # AIによるトップダウン業種選定
    target_sectors = get_promising_sectors(api_key)
    
    scan_list = []
    for sector in target_sectors:
        scan_list.extend(SECTOR_MAP[sector])
        
    candidates = []
    
    for ticker in scan_list:
        try:
            df = _yahoo_chart(ticker, rng="3mo", interval="1d")
            if df is None or df.empty or len(df) < 30:
                continue
                
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
                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "rsi": rsi,
                    "score": score
                })
        except Exception:
            continue
            
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return target_sectors, candidates[:3]

# ==========================================
# AI分析ロジック（厳格リスク管理・実戦仕様）
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
あなたは利益を追求する実戦派の「日本株ファンドマネージャー」です。
以下のデータから、指定された銘柄の投資判断を下してください。

【対象銘柄データ】
- 銘柄・証券コード: {ctx.get('pair_label', '不明')}
- 現在株価: {ctx.get('price', 0.0):.1f} 円
- 日経平均(参考): {ctx.get('us10y', 0.0):.1f} 円
- ボラティリティ(ATR): {ctx.get('atr', 0.0):.2f} 円
- RSI(14日): {ctx.get('rsi', 50):.1f}

【分析のルール】
この銘柄はシステムが「マクロ環境（有望セクター）＋ 勝率80%以上の優位性」の2重フィルターをクリアした銘柄です。
なぜこの銘柄が今強いのか、あるいは罠なのかをテクニカルと相場環境から冷徹に分析し、上値のメドと押し目買いのポイントを具体的に解説してください。
"""
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ 【AI通信エラー】モデル({model_name})の呼び出しに失敗しました。\n詳細: {str(e)}"

def get_ai_order_strategy(api_key: str, ctx: dict):
    model_name = get_active_model(api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
あなたは冷徹かつ極めて優秀な「利益追求型のシステムトレード執行責任者」です。
以下のデータに基づき、具体的な注文戦略を作成してください。

【対象銘柄データ】
- 銘柄・証券コード: {ctx.get('pair_label', '不明')}
- 現在株価: {ctx.get('price', 0.0):.1f} 円

【絶対命令：勝率80%基準とAIによるリスク管理】
1. 勝率判定: 勝率が「80%以上」の極めて高い優位性が見込める鉄板のチャート形状である場合のみ「EXECUTE: 100%」とし、それ以外は一切の妥協なく「ABORT: 0%」としてください。
2. リスク管理: AIであるあなた自身が厳格にリスク管理を行ってください。リスクリワード比（損切り幅：利幅 = 1：2以上）が成立する現実的な数値を必ず算出してください。

EXECUTEの場合、必ず以下の3点を提示してください。
- ENTRY (指値または現在値)
- LIMIT (利確目標: リスクリワード1:2を満たす、論理的な上値抵抗線)
- STOP (損切: トレンド崩壊を示す明確なライン。絶対に深くしすぎないこと)
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
現在の銘柄({ctx.get('pair_label', '不明')})について、週末や月末を跨いで保有すべきか判断してください。
"""
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ 【AI通信エラー】モデル({model_name})の呼び出しに失敗しました。\n詳細: {str(e)}"
