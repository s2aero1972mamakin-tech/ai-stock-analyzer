import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from datetime import datetime
import pytz
import logic  # 日本株用のロジックファイルをインポート

# ==========================================
# ページ設定と初期化
# ==========================================
st.set_page_config(layout="wide", page_title="AI日本株 全自動ロボット", page_icon="🤖")
st.title("🤖 AI連携型 日本株 全自動システムトレード (勝率80%基準)")

TOKYO = pytz.timezone("Asia/Tokyo")

# セッションステート（状態保持）の初期化
if "target_ticker" not in st.session_state: st.session_state.target_ticker = None
if "pair_label" not in st.session_state: st.session_state.pair_label = None
if "auto_candidates" not in st.session_state: st.session_state.auto_candidates = []
if "ai_range" not in st.session_state: st.session_state.ai_range = None
if "quote" not in st.session_state: st.session_state.quote = (None, None)
if "last_ai_report" not in st.session_state: st.session_state.last_ai_report = "" 

try: default_key = st.secrets.get("GEMINI_API_KEY", "")
except: default_key = ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

# ==========================================
# 🤖 メイン・エンジン：全自動ロボット起動
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 ロボットの起動")

if st.sidebar.button("🔥 マクロ分析＆全自動スキャンを実行", type="primary"):
    if not api_key:
        st.sidebar.error("API Keyが必要です。")
        st.stop()
        
    with st.spinner("AIがマクロ環境(地政学/金利)から最適セクターを絞り込み中..."):
        # AI連携によるスマート・スクリーニング
        target_sectors, top_candidates = logic.auto_scan_value_stocks(api_key)
        
        st.session_state.auto_candidates = top_candidates
        
        if top_candidates:
            # 見つかった第1位の銘柄を、そのままメイン画面に強制セット！
            best = top_candidates[0]
            st.session_state.target_ticker = best["ticker"]
            st.session_state.pair_label = f"🤖 AI発掘 第1位: {best['ticker']} (有望セクター: {'/'.join(target_sectors)})"
            st.sidebar.success(f"トップ銘柄 {best['ticker']} をメイン画面にセットしました！")
        else:
            st.session_state.target_ticker = None
            st.sidebar.error("現在、勝率80%の基準をクリアした銘柄はありません。本日のエントリーは見送ります。")

# 次点候補の表示（スキャン成功時のみ）
if st.session_state.auto_candidates and len(st.session_state.auto_candidates) > 1:
    with st.sidebar.expander("📌 その他の発掘候補 (クリック)"):
        for cand in st.session_state.auto_candidates[1:]:
            st.write(f"- {cand['ticker']} (RSI: {cand['rsi']:.1f})")

# ==========================================
# ⚙️ 手動オーバーライド (ピンポイントで調べたい時用)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 手動分析 (マニュアル指定)")
custom_code = st.sidebar.text_input("日本の証券コード", value="", placeholder="例: 8306")
if st.sidebar.button("手動でセット"):
    if custom_code.isdigit() and len(custom_code) == 4:
        st.session_state.target_ticker = f"{custom_code}.T"
        st.session_state.pair_label = f"手動指定: {custom_code}"
    else:
        st.sidebar.error("4桁の数字を入力してください")

# ==========================================
# 💰 SBI証券 資金管理 (リスクコントロール)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("💰 SBI証券 資金管理")
capital = st.sidebar.number_input("軍資金 (JPY)", value=300000, step=10000)
risk_percent = st.sidebar.slider("1トレード許容損失 (%)", 1.0, 10.0, 2.0, step=0.1)

risk_amount = capital * (risk_percent / 100.0)
st.sidebar.markdown(f"**1回の許容損失額**: {risk_amount:,.0f} 円")

stop_loss_width = st.sidebar.number_input("想定損切幅 (円/株)", value=20.0, step=1.0)
if stop_loss_width > 0:
    recommended_shares = math.floor(risk_amount / stop_loss_width)
    st.sidebar.info(f"💡 S株(1株単位) 推奨ロット: **{recommended_shares} 株**")

# ==========================================
# 🛑 メイン画面（待機状態コントロール）
# ==========================================
# まだ何も銘柄がセットされていない（起動直後）の画面表示
if not st.session_state.target_ticker:
    st.info("👈 左側のメニューから「🔥 マクロ分析＆全自動スキャンを実行」ボタンを押して、ロボットを起動してください。")
    st.stop()

# ==========================================
# 📈 描画とAI分析 (AIがセットした銘柄の処理)
# ==========================================
target_ticker = st.session_state.target_ticker
pair_label = st.session_state.pair_label

benchmark_raw = logic.get_market_data("^N225", rng="1y", interval="1d")
df = logic.get_market_data(target_ticker, rng="1y", interval="1d")

if df is None or df.empty:
    st.error(f"データが取得できませんでした: {target_ticker}")
    st.stop()

df = logic.calculate_indicators(df, benchmark_raw)
latest = df.iloc[-1]
curr_price = st.session_state.quote[0] or latest["Close"]

diag = logic.judge_condition(curr_price, latest["SMA_5"], latest["SMA_25"], latest["SMA_75"], latest["RSI"])

# チャート上部のステータスパネル
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div style='padding:10px; border-radius:5px; background-color:{diag['short']['color']}; color:white;'><b>短期診断 (5日線):</b> {diag['short']['status']}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='padding:10px; border-radius:5px; background-color:{diag['mid']['color']}; color:white;'><b>中期トレンド (RSI/MA):</b> {diag['mid']['status']}</div>", unsafe_allow_html=True)

# Plotlyチャート描画
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="株価"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name="SMA 5", line=dict(color='green', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_25'], name="SMA 25", line=dict(color='orange', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_75'], name="SMA 75", line=dict(color='gray', width=2)), row=1, col=1)

if "BENCHMARK" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df['BENCHMARK'], name="日経平均", line=dict(color='cyan', width=1)), row=2, col=1)

fig.update_layout(title=f"{pair_label} - テクニカル分析", xaxis_rangeslider_visible=False, height=700)
st.plotly_chart(fig, use_container_width=True)

# AI用コンテキスト作成
ctx = {
    "pair_label": pair_label,
    "price": curr_price,
    "atr": latest["ATR"],
    "rsi": latest["RSI"],
    "sma_diff": latest["SMA_DIFF"],
    "us10y": latest.get("BENCHMARK", 0.0)
}

# アクションタブ
tab1, tab2, tab3 = st.tabs(["📝 注文戦略を作成 (EXECUTE)", "📊 株価詳細レポート", "💰 ポートフォリオ判断"])

with tab1:
    st.markdown("### 🤖 執行責任者による最終判断")
    if st.button("📝 注文命令書を発行する", type="primary"):
        if api_key:
            with st.spinner("リスクリワードを計算し、注文戦略を策定中..."):
                st.markdown(logic.get_ai_order_strategy(api_key, ctx))
        else: st.warning("API Keyを入力してください。")

with tab2:
    if st.button("✨ 株価分析レポート生成"):
        if api_key:
            with st.spinner("ファンドマネージャーAIが分析中..."):
                report = logic.get_ai_analysis(api_key, ctx)
                st.session_state.last_ai_report = report 
                st.markdown(report)
        else: st.warning("API Keyを入力してください。")

with tab3:
    if st.button("💰 週末/月末 ポートフォリオ判断"):
        if api_key:
            with st.spinner("ホールド可否を判定中..."):
                st.markdown(logic.get_ai_portfolio(api_key, ctx))
        else: st.warning("API Keyを入力してください。")
