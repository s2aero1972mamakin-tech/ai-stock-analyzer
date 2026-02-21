import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
import logic

st.set_page_config(layout="wide", page_title="AI日本株 全自動ロボット", page_icon="🤖")
st.title("🤖 ChatGPT連携型 日本株 全自動システムトレード")

if "target_ticker" not in st.session_state: st.session_state.target_ticker = None
if "pair_label" not in st.session_state: st.session_state.pair_label = None
if "auto_candidates" not in st.session_state: st.session_state.auto_candidates = []
# レポート保存用セッション
if "report_strategy" not in st.session_state: st.session_state.report_strategy = ""
if "report_analysis" not in st.session_state: st.session_state.report_analysis = ""
if "report_portfolio" not in st.session_state: st.session_state.report_portfolio = ""

api_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")

# --- ロボット起動 ---
if st.sidebar.button("🔥 全4000銘柄 スキャン開始", type="primary"):
    if not api_key: st.error("API Keyが必要です。")
    else:
        st.info("🤖 AIが全銘柄をスキャンしています...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        def update_progress(current, total, info):
            progress_bar.progress(int((current / total) * 100))
            status_text.text(f"🔍 スキャン中... {current}/{total} ({info})")
        target_sectors, top_candidates = logic.auto_scan_value_stocks(api_key, progress_callback=update_progress)
        progress_bar.empty()
        status_text.empty()
        st.session_state.auto_candidates = top_candidates
        if top_candidates:
            best = top_candidates[0]
            st.session_state.target_ticker = best["ticker"]
            st.session_state.pair_label = f"{best['ticker']} {best['name']}"
            st.sidebar.success(f"トップ銘柄: {st.session_state.pair_label}")
        else: st.sidebar.error("条件合致なし")

# --- メイン表示 ---
if not st.session_state.target_ticker:
    st.info("👈 スキャンを開始してください")
    st.stop()

df = logic.get_market_data(st.session_state.target_ticker)
df = logic.calculate_indicators(df)
latest = df.iloc[-1]
ctx = {"pair_label": st.session_state.pair_label, "price": latest["Close"], "rsi": latest["RSI"], "atr": latest["ATR"], "sma5": latest["SMA_5"], "sma25": latest["SMA_25"], "sma75": latest["SMA_75"]}

# チャート描画
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="株価"))
fig.update_layout(title=st.session_state.pair_label, xaxis_rangeslider_visible=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# AI診断
tab1, tab2, tab3 = st.tabs(["📝 注文戦略", "📊 分析レポート", "💰 ポートフォリオ"])

with tab1:
    if st.button("📝 戦略生成"):
        st.session_state.report_strategy = logic.get_ai_order_strategy(api_key, ctx)
    st.markdown(st.session_state.report_strategy)

with tab2:
    if st.button("📊 レポート生成"):
        st.session_state.report_analysis = logic.get_ai_analysis(api_key, ctx)
    st.markdown(st.session_state.report_analysis)

with tab3:
    if st.button("💰 判断生成"):
        st.session_state.report_portfolio = logic.get_ai_portfolio(api_key, ctx)
    st.markdown(st.session_state.report_portfolio)

# --- 💾 保存機能 ---
st.markdown("---")
if st.session_state.report_strategy or st.session_state.report_analysis:
    all_text = f"""【AI分析結果記録】
日付: {datetime.now().strftime('%Y-%m-%d %H:%M')}
銘柄: {st.session_state.pair_label}
株価: {latest['Close']}円

■ 注文戦略:
{st.session_state.report_strategy}

■ 分析レポート:
{st.session_state.report_analysis}

■ ポートフォリオ判断:
{st.session_state.report_portfolio}
"""
    st.download_button(
        label="💾 全分析内容をテキストで保存",
        data=all_text,
        file_name=f"TradeReport_{st.session_state.target_ticker}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
