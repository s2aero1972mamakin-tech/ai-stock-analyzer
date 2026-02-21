import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from datetime import datetime
import pytz
import logic

# ==========================================
# ページ設定と初期化
# ==========================================
st.set_page_config(layout="wide", page_title="AI日本株 全自動ロボット", page_icon="🤖")
st.title("🤖 ChatGPT連携型 日本株 全自動システムトレード")
st.markdown("※JPX(東証)全4000銘柄公式データを直結。AIが現在の地政学から有望業種を選び、その全銘柄をスキャンします。")

# 日本時間設定
TOKYO = pytz.timezone("Asia/Tokyo")

# セッション状態の維持（再描画でデータが消えないように保持）
if "target_ticker" not in st.session_state: st.session_state.target_ticker = None
if "pair_label" not in st.session_state: st.session_state.pair_label = None
if "auto_candidates" not in st.session_state: st.session_state.auto_candidates = []
if "report_strategy" not in st.session_state: st.session_state.report_strategy = ""
if "report_analysis" not in st.session_state: st.session_state.report_analysis = ""
if "report_portfolio" not in st.session_state: st.session_state.report_portfolio = ""

# --- サイドバー：OpenAI設定 ---
st.sidebar.header("🔑 設定")
api_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")

# ==========================================
# 🚀 サイドバー機能①：全自動スキャン
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 ロボットの起動")
if st.sidebar.button("🔥 全4000銘柄 マクロスキャン開始", type="primary"):
    if not api_key:
        st.sidebar.error("OpenAIのAPI Keyが必要です。")
        st.stop()
    
    st.info("🤖 AIがセクターを選定し、対象業種の全銘柄を調査中...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # logic.pyの自動スキャンを呼び出し
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
        st.sidebar.error("本日の基準をクリアする銘柄は見つかりませんでした。")

# 他の候補を表示
if st.session_state.auto_candidates and len(st.session_state.auto_candidates) > 1:
    with st.sidebar.expander("📌 その他の期待銘柄"):
        for cand in st.session_state.auto_candidates[1:]:
            st.write(f"- **{cand['ticker']}** {cand['name']} (RSI:{cand['rsi']:.1f})")

# ==========================================
# ⚙️ サイドバー機能②：手動指定
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
    else:
        st.sidebar.error("4桁の数字を入力してください")

# ==========================================
# 💰 サイドバー機能③：資金管理（SBI証券/S株対応）
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("💰 SBI証券 資金管理")
capital = st.sidebar.number_input("軍資金 (JPY)", value=300000, step=10000)
risk_percent = st.sidebar.slider("1トレード許容損失 (%)", 1.0, 5.0, 2.0, step=0.1)
risk_amount = capital * (risk_percent / 100.0)
st.sidebar.write(f"1回の許容損失額: **{risk_amount:,.0f} 円**")

stop_loss_width = st.sidebar.number_input("想定損切幅 (円/株)", value=20.0, step=1.0)
if stop_loss_width > 0:
    recommended_shares = math.floor(risk_amount / stop_loss_width)
    st.sidebar.info(f"💡 推奨ロット: **{recommended_shares} 株** (S株対応)")

# ==========================================
# 📈 メイン画面：チャート描画とAI診断
# ==========================================
if not st.session_state.target_ticker:
    st.info("👈 左側のメニューから「スキャン開始」ボタンを押すか、証券コードを手動入力してください。")
    st.stop()

# チャートデータの取得
df = logic.get_market_data(st.session_state.target_ticker)
if df is None or df.empty:
    st.error("データの取得に失敗しました。証券コードが正しいか確認してください。")
    st.stop()

df = logic.calculate_indicators(df)
latest = df.iloc[-1]

# 画面上部にサマリー表示
col_name, col_price, col_rsi = st.columns([2, 1, 1])
col_name.subheader(st.session_state.pair_label)
col_price.metric("現在値", f"{latest['Close']:.1f} 円", f"{latest['SMA_DIFF']:.2f}%")
col_rsi.metric("RSI (14日)", f"{latest['RSI']:.1f}")

# チャート描画
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="株価"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name="5日線", line=dict(color='green', width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_25'], name="25日線", line=dict(color='orange', width=2)))
fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# AI診断用コンテキスト
ctx = {
    "pair_label": st.session_state.pair_label,
    "price": latest["Close"],
    "rsi": latest["RSI"],
    "atr": latest["ATR"],
    "sma5": latest["SMA_5"],
    "sma25": latest["SMA_25"],
    "sma75": latest["SMA_75"]
}

# --- AI診断タブエリア ---
st.markdown("### 🧠 AIによる高度診断")
tab1, tab2, tab3 = st.tabs(["📝 注文戦略を作成 (EXECUTE)", "📊 株価詳細レポート", "💰 ポートフォリオ判断"])

with tab1:
    if st.button("📝 命令書を発行", key="btn_strategy"):
        with st.spinner("AI執行責任者が戦略を策定中..."):
            st.session_state.report_strategy = logic.get_ai_order_strategy(api_key, ctx)
    st.markdown(st.session_state.report_strategy)

with tab2:
    if st.button("📊 レポートを生成", key="btn_analysis"):
        with st.spinner("AIファンドマネージャーが分析中..."):
            st.session_state.report_analysis = logic.get_ai_analysis(api_key, ctx)
    st.markdown(st.session_state.report_analysis)

with tab3:
    if st.button("💰 ホールド可否を判断", key="btn_portfolio"):
        with st.spinner("AIアナリストが判定中..."):
            st.session_state.report_portfolio = logic.get_ai_portfolio(api_key, ctx)
    st.markdown(st.session_state.report_portfolio)

# ==========================================
# 💾 保存・出力機能（一番下に配置）
# ==========================================
st.markdown("---")
if st.session_state.report_strategy or st.session_state.report_analysis:
    timestamp = datetime.now(TOKYO).strftime('%Y-%m-%d %H:%M')
    # テキストデータの結合
    all_text = f"""【AI日本株トレード分析記録】
発行日時: {timestamp}
対象銘柄: {st.session_state.pair_label}
現在株価: {latest['Close']:.1f} 円

---------------------------------------
■ 1. 注文戦略 (EXECUTE ORDER)
---------------------------------------
{st.session_state.report_strategy}

---------------------------------------
■ 2. 株価詳細レポート
---------------------------------------
{st.session_state.report_analysis}

---------------------------------------
■ 3. ポートフォリオ判断
---------------------------------------
{st.session_state.report_portfolio}

---------------------------------------
テクニカルデータ参考:
- 25日線乖離率: {latest['SMA_DIFF']:.2f}%
- RSI(14): {latest['RSI']:.2f}
- ATR: {latest['ATR']:.2f}
"""
    # ダウンロードボタン
    st.download_button(
        label="💾 この全分析内容をテキストで保存",
        data=all_text,
        file_name=f"TradeReport_{st.session_state.target_ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )
