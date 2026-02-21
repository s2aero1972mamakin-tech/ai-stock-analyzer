import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from datetime import datetime
import pytz
import logic

st.set_page_config(layout="wide", page_title="AI日本株 全自動ロボット", page_icon="🤖")
st.title("🤖 ChatGPT連携型 日本株 全自動システムトレード")
st.markdown("※JPX(東証)全4000銘柄公式データを直結。AIが現在の地政学から有望業種を選び、その全銘柄をスキャンします。")

TOKYO = pytz.timezone("Asia/Tokyo")

if "target_ticker" not in st.session_state: st.session_state.target_ticker = None
if "pair_label" not in st.session_state: st.session_state.pair_label = None
if "auto_candidates" not in st.session_state: st.session_state.auto_candidates = []
if "report_strategy" not in st.session_state: st.session_state.report_strategy = ""
if "report_analysis" not in st.session_state: st.session_state.report_analysis = ""
if "report_portfolio" not in st.session_state: st.session_state.report_portfolio = ""

# --- 🔑 APIキー取得（Secrets最優先） ---
st.sidebar.header("🔑 設定")
secret_key = ""
try:
    secret_key = st.secrets["OPENAI_API_KEY"]
except:
    secret_key = ""

api_key = st.sidebar.text_input(
    "OpenAI API Key (sk-...)", 
    value=secret_key, 
    type="password"
)

# ==========================================
# 🚀 ロボットの起動
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 ロボットの起動")
if st.sidebar.button("🔥 全4000銘柄 マクロスキャン開始", type="primary"):
    if not api_key:
        st.sidebar.error("API Keyが必要です。")
        st.stop()
    
    st.info("🤖 AIがセクターを選定し、対象業種の全銘柄を調査中...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total, info_str):
        percent = int((current / total) * 100)
        progress_bar.progress(percent)
        status_text.text(f"🔍 スキャン中... {current} / {total} ({info_str})")

    target_sectors, top_candidates = logic.auto_scan_value_stocks(api_key, progress_callback=update_progress)
    
    progress_bar.empty()
    status_text.empty()
    st.session_state.auto_candidates = top_candidates
    
    if top_candidates:
        best = top_candidates[0]
        st.session_state.target_ticker = best["ticker"]
        st.session_state.pair_label = f"{best['ticker']} {best['name']}"
        st.sidebar.success(f"発掘完了: {st.session_state.pair_label}")
    else:
        st.sidebar.error("基準クリア銘柄なし。")

if st.session_state.auto_candidates and len(st.session_state.auto_candidates) > 1:
    with st.sidebar.expander("📌 その他の期待銘柄"):
        for cand in st.session_state.auto_candidates[1:]:
            st.write(f"- **{cand['ticker']}** {cand['name']} (RSI:{cand['rsi']:.1f})")

# ==========================================
# ⚙️ 手動分析 (マニュアル)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 手動分析 (マニュアル)")
custom_code = st.sidebar.text_input("証券コード4桁", value="", placeholder="例: 8306")
if st.sidebar.button("手動で分析セット"):
    if custom_code.isdigit() and len(custom_code) == 4:
        ticker_str = f"{custom_code}.T"
        st.session_state.target_ticker = ticker_str
        comp_name = logic.get_company_name(ticker_str)
        st.session_state.pair_label = f"{custom_code} {comp_name}"
        st.sidebar.success(f"セット完了: {st.session_state.pair_label}")

# ==========================================
# 💰 SBI証券 資金管理
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("💰 SBI証券 資金管理")
capital = st.sidebar.number_input("軍資金 (JPY)", value=300000)
risk_percent = st.sidebar.slider("1トレード許容損失 (%)", 1.0, 5.0, 2.0)
stop_loss_width = st.sidebar.number_input("想定損切幅 (円/株)", value=20.0)
if stop_loss_width > 0:
    recommended_shares = math.floor((capital * (risk_percent / 100.0)) / stop_loss_width)
    st.sidebar.info(f"💡 推奨ロット: **{recommended_shares} 株**")

# ==========================================
# 📈 メイン表示
# ==========================================
if not st.session_state.target_ticker:
    st.info("👈 スキャンを開始してください。")
    st.stop()

df = logic.get_market_data(st.session_state.target_ticker)
df = logic.calculate_indicators(df)
latest = df.iloc[-1]

# チャート表示
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="株価"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name="5日線", line=dict(color='green', width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_25'], name="25日線", line=dict(color='orange', width=2)))
fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

ctx = {"pair_label": st.session_state.pair_label, "price": latest["Close"], "rsi": latest["RSI"], "atr": latest["ATR"]}

# --- AI診断タブ ---
tab1, tab2, tab3 = st.tabs(["📝 注文戦略", "📊 分析レポート", "💰 ポートフォリオ"])
with tab1:
    if st.button("📝 命令書を発行"):
        st.session_state.report_strategy = logic.get_ai_order_strategy(api_key, ctx)
    st.markdown(st.session_state.report_strategy)

with tab2:
    if st.button("📊 レポートを生成"):
        st.session_state.report_analysis = logic.get_ai_analysis(api_key, ctx)
    st.markdown(st.session_state.report_analysis)

with tab3:
    if st.button("💰 ホールド可否を判断"):
        st.session_state.report_portfolio = logic.get_ai_portfolio(api_key, ctx)
    st.markdown(st.session_state.report_portfolio)

# --- 💾 保存機能 ---
st.markdown("---")
if st.session_state.report_strategy or st.session_state.report_analysis:
    report_text = f"日時: {datetime.now(TOKYO)}\n銘柄: {st.session_state.pair_label}\n株価: {latest['Close']}円\n\n■ 注文戦略:\n{st.session_state.report_strategy}\n\n■ 分析:\n{st.session_state.report_analysis}"
    st.download_button("💾 全分析レポートをテキストで保存", report_text, file_name=f"Report_{st.session_state.target_ticker}.txt")
