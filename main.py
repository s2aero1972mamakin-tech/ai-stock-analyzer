import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from datetime import datetime
import pytz
import logic  # 日本株用のロジックファイルをインポート

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI日本株アナライザー 2026")
st.title("🤖 AI連携型 日本株 戦略分析ツール (S株対応版・全自動スキャン搭載)")

TOKYO = pytz.timezone("Asia/Tokyo")

# --- セッションステート初期化 ---
if "ai_range" not in st.session_state: st.session_state.ai_range = None
if "quote" not in st.session_state: st.session_state.quote = (None, None)
if "last_ai_report" not in st.session_state: st.session_state.last_ai_report = "" 

# --- APIキー取得 ---
try: default_key = st.secrets.get("GEMINI_API_KEY", "")
except: default_key = ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

# ==========================================
# --- サイドバー設定: 📈 分析対象の銘柄設定 ---
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("📈 分析対象の銘柄設定")

input_mode = st.sidebar.radio("銘柄の指定方法", ["リストから選ぶ (代表銘柄)", "証券コードを直接入力"])

if input_mode == "リストから選ぶ (代表銘柄)":
    predefined_stocks = {
        "三菱UFJ FG (8306)": "8306.T",
        "トヨタ自動車 (7203)": "7203.T",
        "三菱重工業 (7011)": "7011.T",
        "ソフトバンクG (9984)": "9984.T",
        "NTT (9432)": "9432.T",
        "日本製鉄 (5401)": "5401.T",
        "ホンダ (7267)": "7267.T",
        "JT 日本たばこ産業 (2914)": "2914.T"
    }
    pair_label = st.sidebar.selectbox("対象銘柄を選択", list(predefined_stocks.keys()))
    target_ticker = predefined_stocks[pair_label]
else:
    custom_code = st.sidebar.text_input("日本の証券コード (4桁の半角数字)", value="", placeholder="例: 7974 (任天堂)")
    if custom_code == "":
        st.sidebar.info("👆 分析したい銘柄の4桁のコードを入力してください。")
        st.stop()
    elif custom_code.isdigit() and len(custom_code) == 4:
        target_ticker = f"{custom_code}.T"
        pair_label = f"証券コード: {custom_code} (カスタム銘柄)"
    else:
        st.sidebar.error("⚠️ エラー: 4桁の半角数字を入力してください（例: 7203）")
        st.stop()

# ==========================================
# --- サイドバー設定: 🤖 全自動ロボット起動 ---
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 全自動ロボット起動")

if st.sidebar.button("🚀 全自動スキャン＆AI分析を実行"):
    if not api_key:
        st.sidebar.error("API Keyが必要です。")
        st.stop()
        
    with st.spinner("プログラムが市場をスキャン中... (約10〜20秒)"):
        # 1. プログラムによる数学的スクリーニング（全自動発掘）
        top_candidates = logic.auto_scan_value_stocks()
        
    if not top_candidates:
        st.error("現在、システムが買いと判断した銘柄はありません。（相場環境が悪いため待機します）")
        st.stop()
        
    st.success(f"🔥 スキャン完了！ 有力候補 {len(top_candidates)} 銘柄を発見しました。AIによる最終診断を開始します。")
    
    # 2. 発掘された銘柄をAIに連続で診断させる
    for cand in top_candidates:
        t_code = cand["ticker"]
        st.markdown(f"### 🎯 発掘銘柄: {t_code} (現在値: {cand['price']:.1f}円 / RSI: {cand['rsi']:.1f})")
        
        # エラーを完全に回避したコンテキスト生成
        ctx_auto = {
            "pair_label": f"証券コード: {t_code}",
            "price": cand['price'],
            "rsi": cand['rsi'],
            "atr": 0.0,
            "sma_diff": 0.0,
            "us10y": logic.get_latest_quote("^N225") or 0.0  
        }
        
        with st.spinner(f"{t_code} をAIが分析中..."):
            report = logic.get_ai_analysis(api_key, ctx_auto)
            strategy = logic.get_ai_order_strategy(api_key, ctx_auto)
            
            with st.expander(f"📝 {t_code} のAI決済判断（クリックして展開）", expanded=True):
                st.markdown("**【ファンドマネージャー分析】**")
                st.write(report)
                st.markdown("---")
                st.markdown("**【執行責任者の注文戦略】**")
                st.write(strategy)
    
    # 全自動スキャン終了後は、下の個別銘柄チャートを描画せずにここで処理を止める
    st.stop()

# ==========================================
# --- サイドバー設定: 💰 SBI証券 資金管理 ---
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("💰 SBI証券 資金管理 (日本株版)")

capital = st.sidebar.number_input("軍資金 (JPY)", value=300000, step=10000)
risk_percent = st.sidebar.slider("1トレード許容損失 (%)", 1.0, 10.0, 2.0, step=0.1)

risk_amount = capital * (risk_percent / 100.0)
st.sidebar.markdown(f"**1回の許容損失額（最大）**: {risk_amount:,.0f} 円")

stop_loss_width = st.sidebar.number_input("想定損切幅 (円/株)", value=20.0, step=1.0, help="現値からいくら下がったら損切りするか")

if stop_loss_width > 0:
    recommended_shares = math.floor(risk_amount / stop_loss_width)
    recommended_100_units = math.floor(recommended_shares / 100) * 100
    
    st.sidebar.markdown("### 📊 発注推奨株数")
    st.sidebar.info(f"💡 S株(1株単位)での推奨: **{recommended_shares} 株**")
    if recommended_100_units > 0:
        st.sidebar.success(f"💡 単元(100株単位)での推奨: **{recommended_100_units} 株**")
    else:
        st.sidebar.warning("⚠️ 100株単位で買うには許容リスク枠が足りません。SBI証券の「S株（1株単位）」での購入を推奨します。")

st.sidebar.markdown("---")
entry_price = st.sidebar.number_input("保有価格 (保有時のみ)", value=0.0, step=10.0)
trade_type = st.sidebar.selectbox("保有タイプ", ["なし", "買い (現物/信用)", "売り (信用)"])

if st.sidebar.button("🔄 最新クオート更新"):
    st.session_state.quote = logic.get_latest_quote(target_ticker), datetime.now(TOKYO)
    st.session_state.ai_range = None

if st.sidebar.button("📈 AI予想ライン反映"):
    if api_key:
        with st.spinner("AI予想レンジ取得中..."):
            ctx_temp = {"price": st.session_state.quote[0] or 0.0}
            st.session_state.ai_range = logic.get_ai_range(api_key, ctx_temp)
    else: st.sidebar.warning("API Keyが必要です")

# ==========================================
# --- メイン処理 (個別チャート・分析) ---
# ==========================================
benchmark_raw = logic.get_market_data("^N225", rng="1y", interval="1d")
df = logic.get_market_data(target_ticker, rng="1y", interval="1d")

if df is None or df.empty:
    st.error(f"データが取得できませんでした。証券コードが正しいか確認してください: {target_ticker}")
    st.stop()

df = logic.calculate_indicators(df, benchmark_raw)
latest = df.iloc[-1]
curr_price = st.session_state.quote[0] or latest["Close"]

diag = logic.judge_condition(curr_price, latest["SMA_5"], latest["SMA_25"], latest["SMA_75"], latest["RSI"])

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div style='padding:10px; border-radius:5px; background-color:{diag['short']['color']}; color:white;'><b>短期診断 (5日線):</b> {diag['short']['status']}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='padding:10px; border-radius:5px; background-color:{diag['mid']['color']}; color:white;'><b>中期トレンド (RSI/MA):</b> {diag['mid']['status']}</div>", unsafe_allow_html=True)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="株価"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name="SMA 5", line=dict(color='green', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_25'], name="SMA 25", line=dict(color='orange', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_75'], name="SMA 75", line=dict(color='gray', width=2)), row=1, col=1)

if st.session_state.ai_range:
    fig.add_hline(y=st.session_state.ai_range.get("high", curr_price), line_dash="dash", line_color="red", annotation_text="AI予想高値", row=1, col=1)
    fig.add_hline(y=st.session_state.ai_range.get("low", curr_price), line_dash="dash", line_color="green", annotation_text="AI予想安値", row=1, col=1)

if entry_price > 0 and trade_type != "なし":
    color = "blue" if "買い" in trade_type else "magenta"
    fig.add_hline(y=entry_price, line_dash="dot", line_color=color, annotation_text=f"保有 ({trade_type})", row=1, col=1)

if "BENCHMARK" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df['BENCHMARK'], name="日経平均", line=dict(color='cyan', width=1)), row=2, col=1)

fig.update_layout(title=f"{pair_label} - テクニカル分析", xaxis_rangeslider_visible=False, height=700)
st.plotly_chart(fig, use_container_width=True)

ctx = {
    "pair_label": pair_label,
    "price": curr_price,
    "atr": latest["ATR"],
    "rsi": latest["RSI"],
    "sma_diff": latest["SMA_DIFF"],
    "us10y": latest.get("BENCHMARK", 0.0), 
    "panel_short": diag['short']['status'],
    "panel_mid": diag['mid']['status']
}

tab1, tab2, tab3 = st.tabs(["📊 株価詳細レポート", "📝 注文戦略", "💰 ポートフォリオ判断"])

with tab1:
    if st.button("✨ 株価分析レポート生成"):
        if api_key:
            with st.spinner("ファンドマネージャーAIが分析中..."):
                report = logic.get_ai_analysis(api_key, ctx)
                st.session_state.last_ai_report = report 
                st.markdown(report)
        else: st.warning("API Keyを入力してください。")

with tab2:
    if st.button("📝 注文命令書作成"):
        if api_key:
            with st.spinner("AI執行責任者が戦略を策定中..."):
                ctx["last_report"] = st.session_state.last_ai_report
                st.markdown(logic.get_ai_order_strategy(api_key, ctx))
        else: st.warning("API Keyを入力してください。")

with tab3:
    if st.button("💰 週末/月末 ポートフォリオ判断"):
        if api_key:
            with st.spinner("ホールド可否を判定中..."):
                st.markdown(logic.get_ai_portfolio(api_key, ctx))
        else: st.warning("API Keyを入力してください。")
