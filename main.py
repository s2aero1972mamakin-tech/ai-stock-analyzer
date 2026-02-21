import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from datetime import datetime
import pytz
import logic

# ページ設定
st.set_page_config(layout="wide", page_title="AI日本株 全自動ロボット", page_icon="🤖")
st.title("🤖 ChatGPT連携型 日本株 全自動システムトレード")

TOKYO = pytz.timezone("Asia/Tokyo")

# セッション状態
if "target_ticker" not in st.session_state: st.session_state.target_ticker = None
if "pair_label" not in st.session_state: st.session_state.pair_label = None
if "auto_candidates" not in st.session_state: st.session_state.auto_candidates = []
if "report_strategy" not in st.session_state: st.session_state.report_strategy = ""
if "report_analysis" not in st.session_state: st.session_state.report_analysis = ""
if "report_portfolio" not in st.session_state: st.session_state.report_portfolio = ""

# サイドバー：設定
api_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")

# 🚀 ロボット起動
st.sidebar.subheader("🚀 ロボットの起動")
if st.sidebar.button("🔥 全4000銘柄 スキャン開始", type="primary"):
    if not api_key: st.sidebar.error("API Keyを入力してください。")
    else:
        st.info("🤖 AIが全銘柄をスキャンしています...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        def update_progress(current, total, info):
            progress_bar.progress(int((current / total) * 100))
            status_text.text(f"🔍 調査中... {current}/{total} ({info})")
        sectors, candidates = logic.auto_scan_value_stocks(api_key, progress_callback=update_progress)
        progress_bar.empty()
        status_text.empty()
        st.session_state.auto_candidates = candidates
        if candidates:
            best = candidates[0]
            st.session_state.target_ticker, st.session_state.pair_label = best["ticker"], f"{best['ticker']} {best['name']}"
            st.sidebar.success(f"第1位: {st.session_state.pair_label}")
        else: st.sidebar.error("条件合致なし")

if st.session_state.auto_candidates:
    with st.sidebar.expander("📌 他の候補銘柄"):
        for c in st.session_state.auto_candidates[1:]:
            st.write(f"- {c['ticker']} {c['name']}")

# ⚙️ 手動指定
st.sidebar.markdown("---")
custom_code = st.sidebar.text_input("証券コード4桁", placeholder="例: 8306")
if st.sidebar.button("手動セット"):
    if len(custom_code) == 4:
        st.session_state.target_ticker = f"{custom_code}.T"
        st.session_state.pair_label = f"{custom_code} {logic.get_company_name(st.session_state.target_ticker)}"

# 💰 資金管理
st.sidebar.markdown("---")
cap = st.sidebar.number_input("軍資金 (JPY)", value=300000)
risk = st.sidebar.slider("許容損失 (%)", 1.0, 5.0, 2.0)
stop_w = st.sidebar.number_input("損切幅 (円)", value=20.0)
st.sidebar.info(f"💡 推奨ロット: {math.floor((cap*(risk/100))/stop_w)} 株")

# 📈 メイン表示
if not st.session_state.target_ticker:
    st.info("👈 スキャンを開始してください")
    st.stop()

df = logic.get_market_data(st.session_state.target_ticker)
df = logic.calculate_indicators(df)
latest = df.iloc[-1]

# チャート
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
fig.update_layout(title=st.session_state.pair_label, xaxis_rangeslider_visible=False, height=450)
st.plotly_chart(fig, use_container_width=True)

ctx = {"pair_label": st.session_state.pair_label, "price": latest["Close"], "rsi": latest["RSI"], "atr": latest["ATR"], "sma5": latest["SMA_5"], "sma25": latest["SMA_25"], "sma75": latest["SMA_75"]}

# AI診断タブ
tab1, tab2, tab3 = st.tabs(["📝 注文戦略", "📊 分析レポート", "💰 ポートフォリオ"])
with tab1:
    if st.button("📝 生成"): st.session_state.report_strategy = logic.get_ai_order_strategy(api_key, ctx)
    st.markdown(st.session_state.report_strategy)
with tab2:
    if st.button("📊 生成"): st.session_state.report_analysis = logic.get_ai_analysis(api_key, ctx)
    st.markdown(st.session_state.report_analysis)
with tab3:
    if st.button("💰 生成"): st.session_state.report_portfolio = logic.get_ai_portfolio(api_key, ctx)
    st.markdown(st.session_state.report_portfolio)

# 💾 保存ボタン
st.markdown("---")
if st.session_state.report_strategy or st.session_state.report_analysis:
    report_text = f"日時: {datetime.now(TOKYO)}\n銘柄: {st.session_state.pair_label}\n\n■戦略\n{st.session_state.report_strategy}\n\n■レポート\n{st.session_state.report_analysis}"
    st.download_button("💾 レポートを保存", report_text, file_name=f"Report_{st.session_state.target_ticker}.txt")
