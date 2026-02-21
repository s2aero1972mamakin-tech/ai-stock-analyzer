import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from datetime import datetime
import pytz
import logic

# 1. ページ構成の初期化
st.set_page_config(layout="wide", page_title="AI日本株 全自動ロボット", page_icon="🤖")
st.title("🤖 ChatGPT連携型 日本株 全自動システムトレード")
st.markdown("※JPX全4000銘柄公式データ直結。マクロ・テクニカルの2重フィルターによるスクリーニングを実行します。")

TOKYO = pytz.timezone("Asia/Tokyo")

# 2. セッション状態（State）の厳格管理
# これにより、UI操作のたびにスキャンが走り直すのを防ぎます
if "target_ticker" not in st.session_state: st.session_state.target_ticker = None
if "pair_label" not in st.session_state: st.session_state.pair_label = None
if "auto_candidates" not in st.session_state: st.session_state.auto_candidates = []
if "report_strategy" not in st.session_state: st.session_state.report_strategy = ""
if "report_analysis" not in st.session_state: st.session_state.report_analysis = ""
if "report_portfolio" not in st.session_state: st.session_state.report_portfolio = ""

# 3. 🔑 APIキー取得（Secrets優先・サイドバー表示）
st.sidebar.header("🔑 システム設定")
secret_key = st.secrets.get("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input("OpenAI API Key", value=secret_key, type="password")

# 4. 🚀 アルゴリズム①：全自動マクロスキャン
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 全自動スキャン")
if st.sidebar.button("🔥 全4000銘柄 スキャン開始", type="primary"):
    if not api_key:
        st.sidebar.error("API Keyが必要です。")
    else:
        # スキャン実行（logic.py側のキャッシュを利用）
        with st.status("🤖 AIが市場を分析中...", expanded=True) as status:
            st.write("1. 東証33業種から有望セクターを選定中...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, info):
                progress_bar.progress(int((current / total) * 100))
                status_text.text(f"🔍 調査中: {current}/{total} ({info})")
            
            # スキャン本体の呼び出し
            sectors, candidates = logic.auto_scan_value_stocks(api_key, progress_callback=update_progress)
            
            st.session_state.auto_candidates = candidates
            if candidates:
                # 第1位を自動セット
                best = candidates[0]
                st.session_state.target_ticker = best["ticker"]
                st.session_state.pair_label = f"{best['ticker']} {best['name']}"
                # 銘柄が変わるのでレポートはクリア
                st.session_state.report_strategy = ""
                st.session_state.report_analysis = ""
                st.session_state.report_portfolio = ""
                status.update(label="✅ スキャン完了！有望銘柄を特定しました。", state="complete", expanded=False)
            else:
                st.sidebar.error("基準クリア銘柄なし。")
        # 画面をリフレッシュして結果を表示フェーズへ
        st.rerun()

# 候補リストの表示と切替ロジック
if st.session_state.auto_candidates:
    with st.sidebar.expander("📌 発掘された買い候補 (TOP3)"):
        for c in st.session_state.auto_candidates:
            if st.sidebar.button(f"分析：{c['ticker']} {c['name']}", key=f"btn_{c['ticker']}"):
                st.session_state.target_ticker = c['ticker']
                st.session_state.pair_label = f"{c['ticker']} {c['name']}"
                st.session_state.report_strategy = ""
                st.session_state.report_analysis = ""
                st.session_state.report_portfolio = ""
                st.rerun()

# 5. ⚙️ アルゴリズム②：手動コード入力
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ マニュアル分析")
custom_code = st.sidebar.text_input("証券コード4桁", placeholder="例: 8306")
if st.sidebar.button("指定コードをセット"):
    if len(custom_code) == 4:
        t_code = f"{custom_code}.T"
        st.session_state.target_ticker = t_code
        st.session_state.pair_label = f"{custom_code} {logic.get_company_name(t_code)}"
        st.session_state.report_strategy = ""
        st.session_state.report_analysis = ""
        st.session_state.report_portfolio = ""
        st.rerun()

# 6. 💰 アルゴリズム③：SBI証券 位置サイズ計算
st.sidebar.markdown("---")
st.sidebar.subheader("💰 SBI証券 資金管理")
cap = st.sidebar.number_input("運用軍資金 (円)", value=300000, step=10000)
risk = st.sidebar.slider("1トレード許容損失 (%)", 0.5, 5.0, 2.0, step=0.1)
stop_w = st.sidebar.number_input("損切幅 (1株あたりの円)", value=20.0, step=1.0)
if stop_w > 0:
    lot = math.floor((cap * (risk / 100)) / stop_w)
    st.sidebar.info(f"💡 推奨ロット: **{lot} 株** (S株/単元未満対応)")

# ==========================================
# 📈 メイン表示セクション（表示アルゴリズム）
# ==========================================
if not st.session_state.target_ticker:
    st.info("👈 左側のボタンからスキャンを開始するか、証券コードを入力してください。")
    st.stop()

# データ処理
with st.spinner("チャートデータを生成中..."):
    df_raw = logic.get_market_data(st.session_state.target_ticker)
    if df_raw is None or df_raw.empty:
        st.error("データの取得に失敗しました。")
        st.stop()
    df = logic.calculate_indicators(df_raw)
    latest = df.iloc[-1]

# 指標サマリー表示
c1, c2, c3, c4 = st.columns(4)
c1.metric("銘柄", st.session_state.pair_label)
c2.metric("現在値", f"{latest['Close']:.1f}円")
c3.metric("RSI(14)", f"{latest['RSI']:.1f}")
c4.metric("25日線乖離", f"{latest['SMA_DIFF']:.2f}%")

# チャート
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="株価"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name="5日線", line=dict(color='green', width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_25'], name="25日線", line=dict(color='orange', width=2)))
fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# 🧠 AI診断セクション
ctx = {
    "pair_label": st.session_state.pair_label,
    "price": latest["Close"],
    "rsi": latest["RSI"],
    "atr": latest["ATR"],
    "sma5": latest["SMA_5"],
    "sma25": latest["SMA_25"]
}

st.markdown("### 🧠 AIプロフェッショナル診断")
t1, t2, t3 = st.tabs(["📝 注文命令書", "📊 詳細分析", "💰 保有判断"])

with t1:
    if st.button("📝 命令書を発行"):
        st.session_state.report_strategy = logic.get_ai_order_strategy(api_key, ctx)
    st.markdown(st.session_state.report_strategy)

with t2:
    if st.button("📊 分析を実行"):
        st.session_state.report_analysis = logic.get_ai_analysis(api_key, ctx)
    st.markdown(st.session_state.report_analysis)

with t3:
    if st.button("💰 判断を仰ぐ"):
        st.session_state.report_portfolio = logic.get_ai_portfolio(api_key, ctx)
    st.markdown(st.session_state.report_portfolio)

# 💾 保存・レポート出力
st.markdown("---")
if st.session_state.report_strategy or st.session_state.report_analysis:
    all_report = f"""【AI日本株分析レポート】
生成日時: {datetime.now(TOKYO).strftime('%Y-%m-%d %H:%M')}
対象: {st.session_state.pair_label}

■ 注文戦略
{st.session_state.report_strategy}

■ 分析詳細
{st.session_state.report_analysis}

■ 継続保有判断
{st.session_state.report_portfolio}
"""
    st.download_button(
        label="💾 この全分析をテキストで保存",
        data=all_report,
        file_name=f"TradeLog_{st.session_state.target_ticker}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
