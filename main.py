# main.py
# -*- coding: utf-8 -*-
"""
日本株スイング自動スキャン（Streamlit）
- データソース: Stooq
- JPX全銘柄スキャン
- セクター事前絞り込み（任意）
- pullback / breakout 両モード
- 0件時 auto-relax（pullback→breakout + 条件緩和 + sector OFF 再スキャン）
- 診断JSONをサイドバーに「即時」表示 + ダウンロード
- main.py と logic.py のパラメータ不一致でも落ちない（安全に無視して警告表示）
"""

import datetime
import json
import math
import dataclasses
import inspect

import pandas as pd
import streamlit as st

# Plotly is optional; if missing, the app still works without charts.
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None

import logic


APP_BUILD = "2026-02-26T00:00:00Z / stable"


# -------------------------
# Helpers
# -------------------------
def _st_plotly(fig, **kwargs):
    """plotly_chart wrapper for Streamlit versions without use_container_width."""
    try:
        if "use_container_width" in inspect.signature(st.plotly_chart).parameters:
            return st.plotly_chart(fig, use_container_width=True, **kwargs)
    except Exception:
        pass
    return st.plotly_chart(fig, **kwargs)


def build_swing_params(**kwargs):
    """Build SwingParams safely (main.py と logic.py の不一致でも落ちない)."""
    allowed = None
    try:
        if dataclasses.is_dataclass(logic.SwingParams):
            allowed = {f.name for f in dataclasses.fields(logic.SwingParams)}
    except Exception:
        allowed = None

    if not allowed:
        try:
            allowed = set(inspect.signature(logic.SwingParams).parameters.keys())
            allowed.discard("self")
        except Exception:
            allowed = set()

    safe = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted([k for k in kwargs.keys() if k not in safe])

    # mismatch があっても落ちないようにしつつ、気づけるように残す
    if dropped:
        st.session_state["_dropped_param_keys"] = dropped
    else:
        st.session_state.pop("_dropped_param_keys", None)

    return logic.SwingParams(**safe)


def render_diag_sidebar(ph):
    """Render diagnostic JSON panel in sidebar (scan直後に表示)."""
    diag = st.session_state.get("last_scan_diag")
    with ph.container():
        if not diag:
            st.caption("🧾 診断JSON：スキャン実行後にここへ表示されます。")
            return

        with st.expander("🧾 スキャン診断（filter_stats / params / auto_relax）", expanded=False):
            st.caption(str(diag.get("timestamp", "")))
            dropped = st.session_state.get("_dropped_param_keys")
            if dropped:
                st.warning("⚠️ main.py と logic.py のパラメータ不一致があり、無視した項目があります: " + ", ".join(dropped))

            try:
                ts = str(diag.get("timestamp", "")).replace(":", "-").replace(" ", "_")
                st.download_button(
                    "⬇️ 診断JSONをダウンロード",
                    data=json.dumps(diag, ensure_ascii=False, indent=2),
                    file_name=f"scan_diagnostic_{ts}.json",
                    mime="application/json",
                )
            except Exception:
                pass

            st.markdown("**mode**")
            st.code(str(diag.get("mode", "")))

            st.markdown("**filter_stats**")
            st.json(diag.get("filter_stats", {}))

            if diag.get("params_effective"):
                st.markdown("**params_effective**")
                st.json(diag.get("params_effective", {}))

            if diag.get("auto_relax_trace"):
                st.markdown("**auto_relax_trace**")
                st.json(diag.get("auto_relax_trace", []))

            if diag.get("error"):
                st.markdown("**error**")
                st.error(str(diag.get("error")))


# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="JPX Swing Auto Scanner", layout="wide")

st.title("📈 日本株スイング自動スキャン（Stooq）")
st.caption("pullback / breakout  +  0件時 auto-relax  +  診断JSON即時表示")

# Session state defaults
defaults = {
    "auto_candidates": [],
    "scan_meta": {},
    "target_ticker": "",
    "pair_label": "",
    "report_strategy": "",
    "report_analysis": "",
    "report_portfolio": "",
    "last_scan_diag": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------------
# Sidebar (settings)
# -------------------------
st.sidebar.header("⚙️ スキャン設定（期待値最大化）")
st.sidebar.caption(f"build: {APP_BUILD}")

diag_ph = st.sidebar.empty()
render_diag_sidebar(diag_ph)

st.sidebar.markdown("#### 資金/リスク")
budget = st.sidebar.number_input("想定資金（円）", min_value=50_000, max_value=5_000_000, value=300_000, step=50_000)
capital = st.sidebar.number_input("注文書: 運用資金（円）", min_value=50_000, max_value=20_000_000, value=300_000, step=50_000)
risk_pct = st.sidebar.slider("許容損失（1トレード）", 0.1, 3.0, 1.0, step=0.1) / 100.0

st.sidebar.markdown("#### エントリーモード")
entry_mode = st.sidebar.selectbox("モード", ["押し目（pullback）", "ブレイクアウト（breakout）"], index=0)
entry_mode_key = "pullback" if entry_mode.startswith("押し目") else "breakout"

st.sidebar.markdown("#### フィルタ（市場実態に合わせる）")
require_trend = st.sidebar.checkbox("SMA25 > SMA75 を必須（トレンド順張り）", value=True)

rsi_low = st.sidebar.slider("RSI下限", 10.0, 60.0, 40.0, step=1.0)
rsi_high = st.sidebar.slider("RSI上限", 40.0, 90.0, 70.0, step=1.0)

st.sidebar.markdown("#### 押し目条件（SMA25からの乖離%）")
pb_low = st.sidebar.slider("押し目下限（%）", -20.0, 0.0, -8.0, step=0.5)
pb_high = st.sidebar.slider("押し目上限（%）", -15.0, 2.0, -3.0, step=0.5)

st.sidebar.markdown("#### 変動幅（ATR%）")
atr_min = st.sidebar.slider("ATR%下限", 0.5, 12.0, 1.5, step=0.5)
atr_max = st.sidebar.slider("ATR%上限", 1.0, 25.0, 10.0, step=0.5)

st.sidebar.markdown("#### 流動性（出来高/売買代金）")
vol_min = st.sidebar.number_input("平均出来高(20日) 下限", min_value=0, max_value=10_000_000, value=50_000, step=10_000)
turnover_min_yen = st.sidebar.number_input("平均売買代金(20日) 下限（円）", min_value=0, max_value=50_000_000_000, value=0, step=10_000_000)

st.sidebar.markdown("#### ブレイクアウト条件")
breakout_lookback = st.sidebar.slider("高値更新の参照日数", 5, 60, 20, step=1)
breakout_vol_ratio = st.sidebar.slider("出来高倍率（当日/20日平均）", 1.0, 5.0, 1.6, step=0.1)

st.sidebar.markdown("#### 損切/利確（R設計）")
atr_mult = st.sidebar.slider("損切: ATR倍率", 0.8, 3.5, 2.0, step=0.1)
tp1_r = st.sidebar.slider("利確1: +何Rで半分利確", 0.5, 2.0, 1.0, step=0.1)
tp2_r = st.sidebar.slider("利確2: +何Rを狙う", 1.5, 6.0, 3.0, step=0.5)
time_stop_days = st.sidebar.slider("時間切れ（TP1未達で撤退）", 3, 20, 10, step=1)

st.sidebar.markdown("#### バックテスト")
bt_period = st.sidebar.selectbox("バックテスト期間", ["1y", "2y", "3y", "5y"], index=1)
bt_topk = st.sidebar.slider("バックテスト対象（上位K）", 5, 50, 20, step=5)

st.sidebar.markdown("#### セクター事前絞り込み（高速化）")
sector_prefilter = st.sidebar.checkbox("まずセクターで絞り込む（推奨）", value=True)
sector_top_n = st.sidebar.slider("採用する上位セクター数", 2, 12, 6, step=1)
sector_method = st.sidebar.selectbox("絞り込み方式", ["データ（推奨）", "AI＋データ（任意）"], index=0)
sector_method_key = "ai_overlay" if sector_method.startswith("AI") else "quant"

params = build_swing_params(
    rsi_low=float(rsi_low),
    rsi_high=float(rsi_high),
    pullback_low=float(pb_low),
    pullback_high=float(pb_high),
    atr_pct_min=float(atr_min),
    atr_pct_max=float(atr_max),
    vol_avg20_min=float(vol_min),
    turnover_avg20_min_yen=float(turnover_min_yen),
    require_sma25_over_sma75=bool(require_trend),
    entry_mode=str(entry_mode_key),
    atr_mult_stop=float(atr_mult),
    tp1_r=float(tp1_r),
    tp2_r=float(tp2_r),
    time_stop_days=int(time_stop_days),
    breakout_lookback=int(breakout_lookback),
    breakout_vol_ratio=float(breakout_vol_ratio),
    risk_pct=float(risk_pct),
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 全銘柄スキャン")
scan_label = "🔥 スキャン開始（セクター→銘柄スキャン）" if sector_prefilter else "🔥 スキャン開始（JPX全銘柄）"
scan_btn = st.sidebar.button(scan_label, type="primary")

# OpenAI key (optional)
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 AIコメント（任意）")
secret_key = st.secrets.get("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input("OpenAI API Key", value=secret_key, type="password")
st.sidebar.caption("※AI生成は任意。スキャン/バックテスト/注文書はAPIキー不要。")


# -------------------------
# Scan execution
# -------------------------
if scan_btn:
    # First: show "running" diag immediately
    st.session_state["last_scan_diag"] = {
        "timestamp": str(datetime.datetime.now()),
        "mode": "running",
        "filter_stats": {},
        "params_effective": {},
        "auto_relax_trace": [],
    }
    render_diag_sidebar(diag_ph)

    try:
        with st.status("スキャン＆バックテスト中…", expanded=True) as status:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, info):
                pct = int((current / max(1, total)) * 100)
                progress_bar.progress(pct)
                status_text.text(f"🔍 {info} ({current}/{total})")

            res = logic.scan_swing_candidates(
                budget_yen=int(budget),
                top_n=3,
                params=params,
                progress_callback=update_progress,
                backtest_period=bt_period,
                backtest_topk=int(bt_topk),
                sector_prefilter=bool(sector_prefilter),
                sector_top_n=int(sector_top_n),
                sector_method=str(sector_method_key),
                api_key=(api_key if sector_method_key == "ai_overlay" else None),
            )

            candidates = res.get("candidates", [])
            st.session_state.scan_meta = res
            st.session_state.auto_candidates = candidates

            # Persist diagnostics (and show immediately)
            st.session_state["last_scan_diag"] = {
                "timestamp": str(datetime.datetime.now()),
                "mode": res.get("mode"),
                "filter_stats": res.get("filter_stats", {}),
                "params_effective": res.get("params_effective", {}),
                "auto_relax_trace": res.get("auto_relax_trace", []),
                "error": res.get("error"),
            }

            # Update sidebar diagnostic panel immediately in this run
            try:
                render_diag_sidebar(diag_ph)
            except Exception:
                pass

            if candidates:
                best = candidates[0]
                st.session_state.target_ticker = best.get("ticker", "")
                st.session_state.pair_label = f"{best.get('ticker','')} {best.get('name','')}".strip()
                st.session_state.report_strategy = ""
                st.session_state.report_analysis = ""
                st.session_state.report_portfolio = ""
                status.update(label="✅ 完了：候補を抽出しました", state="complete", expanded=False)
            else:
                status.update(label="⚠️ 完了：候補が0件でした（診断JSONを確認してください）", state="complete", expanded=True)

    except Exception as e:
        st.session_state["last_scan_diag"] = {
            "timestamp": str(datetime.datetime.now()),
            "mode": "exception",
            "filter_stats": {},
            "params_effective": {},
            "auto_relax_trace": [],
            "error": f"{type(e).__name__}: {e}",
        }
        render_diag_sidebar(diag_ph)
        st.exception(e)


# -------------------------
# Sidebar: candidate buttons
# -------------------------
if st.session_state.auto_candidates:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 候補（クリックで分析）")
    for c in st.session_state.auto_candidates:
        label = f"{c.get('ticker','')} {c.get('name','')}".strip()
        stats = f"AvgR={c.get('bt_avg_r', 0):.2f} / PF={c.get('bt_pf', 0):.2f} / Trades={c.get('bt_trades', 0)}"
        if st.sidebar.button(f"分析：{label}（{stats}）", key=f"btn_{c.get('ticker','')}"):
            st.session_state.target_ticker = c.get("ticker", "")
            st.session_state.pair_label = label
            st.session_state.report_strategy = ""
            st.session_state.report_analysis = ""
            st.session_state.report_portfolio = ""
            st.rerun()

# Manual ticker
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 マニュアル分析")
custom_code = st.sidebar.text_input("証券コード4桁（例: 8306）", value="")
if st.sidebar.button("指定コードをセット"):
    if len(custom_code.strip()) == 4 and custom_code.strip().isdigit():
        t = f"{custom_code.strip()}.T"
        st.session_state.target_ticker = t
        st.session_state.pair_label = f"{custom_code.strip()} {logic.get_company_name(t)}"
        st.session_state.report_strategy = ""
        st.session_state.report_analysis = ""
        st.session_state.report_portfolio = ""
        st.rerun()


# -------------------------
# Main section
# -------------------------
if not st.session_state.target_ticker:
    st.info("👈 左側でスキャンを実行するか、証券コードを入力してください。")
    st.stop()

ticker = st.session_state.target_ticker

# Load market data for display/backtest
with st.spinner("データ取得＆指標計算中…"):
    df_raw = logic.get_market_data(ticker, period=max(bt_period, "2y"), interval="1d")
    if df_raw is None or df_raw.empty:
        st.error("データの取得に失敗しました。")
        st.stop()
    df = logic.calculate_indicators(df_raw)
    if df.empty:
        st.error("指標計算に失敗しました。")
        st.stop()

latest = df.iloc[-1]
pair_label = st.session_state.pair_label or f"{ticker} {logic.get_company_name(ticker)}"

# Backtest on selected ticker
with st.spinner("バックテスト計算中…"):
    bt = logic.backtest_swing(df, params)

# Trade plan
plan = logic.build_trade_plan(df, params, capital_yen=int(capital), risk_pct=float(risk_pct))

# Header metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("銘柄", pair_label)
m2.metric("現在値", f"{latest['Close']:.1f}円")
m3.metric("RSI(14)", f"{latest['RSI']:.1f}")
m4.metric("ATR(14)", f"{latest['ATR']:.2f}")
m5.metric("PF", f"{bt.profit_factor:.2f}" if math.isfinite(bt.profit_factor) else "inf")
m6.metric("AvgR", f"{bt.expectancy_r:.2f}")

# Regime info
meta = st.session_state.scan_meta or {}
regime_ok = meta.get("regime_ok", None)
if regime_ok is not None:
    st.info(f"地合いフィルタ（N225 > SMA200）: {'✅ OK（買い優位になりやすい）' if regime_ok else '⚠️ NG（逆風になりやすい）'}")

sel = meta.get("selected_sectors") or []
if sel:
    st.info("セクター事前絞り込み: " + " / ".join([str(s) for s in sel]))

# Price chart
if go is not None and make_subplots is not None:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    if "SMA_5" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="SMA5"))
    if "SMA_25" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="SMA25"))
    if "SMA_75" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="SMA75"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=520, margin=dict(l=0, r=0, t=30, b=0))
    _st_plotly(fig)

st.markdown("### 📌 実行タブ")
tab_plan, tab_bt, tab_tune, tab_ai = st.tabs(["📝 注文書（ロジック）", "📈 バックテスト", "🧪 パラメータ検証", "🧠 AIコメント（任意）"])

with tab_plan:
    if not plan:
        st.warning("注文書を生成できませんでした。")
    else:
        st.subheader("注文書（数値）")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("想定エントリー", f"{plan['entry_price']:.1f}円")
        c2.metric("損切（SL）", f"{plan['stop_price']:.1f}円")
        c3.metric("利確1（TP1）", f"{plan['tp1_price']:.1f}円")
        c4.metric("利確2（TP2）", f"{plan['tp2_price']:.1f}円")

        st.write(f"推奨株数: **{plan['shares']}株**（目安） / 1株あたりリスク: {plan['risk_per_share']:.2f}円")
        if plan.get("warning"):
            st.warning(plan["warning"])

        st.markdown("#### 実行手順（例）")
        st.markdown(
            f"""
- **新規買い**：{plan['entry_price']:.1f}円付近（寄成 or 指値）
- **同時に逆指値/指値をセット**：
  - **損切（SL）**：{plan['stop_price']:.1f}円
  - **利確（TP1）**：{plan['tp1_price']:.1f}円で半分利確
- **TP1達成後**：
  - 残り半分の損切を **建値（エントリー価格）** に引き上げ
  - **TP2**：{plan['tp2_price']:.1f}円（+{params.tp2_r:.1f}R）を狙う（もしくはトレーリング）
"""
        )

with tab_bt:
    st.subheader("バックテスト結果（簡易）")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("トレード数", f"{bt.n_trades}")
    c2.metric("勝率", f"{bt.win_rate*100:.1f}%")
    c3.metric("PF", f"{bt.profit_factor:.2f}" if math.isfinite(bt.profit_factor) else "inf")
    c4.metric("AvgR", f"{bt.expectancy_r:.2f}")
    c5.metric("MaxDD", f"{bt.max_drawdown*100:.1f}%")

    if bt.equity_curve_r is not None and not bt.equity_curve_r.empty and go is not None:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bt.equity_curve_r.index, y=bt.equity_curve_r.values, name="Equity (R)"))
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
        _st_plotly(fig2)

    with st.expander("取引ログ（Rベース）"):
        st.dataframe(bt.trades)

with tab_tune:
    st.subheader("パラメータ検証（1銘柄向け）")
    st.caption("PF/AvgR/MaxDDを見ながら、RSIレンジや押し目乖離が『勝率＋利確』に効く帯域を確認します。")
    if st.button("🧪 この銘柄でグリッド検証を実行"):
        with st.spinner("検証中…（数十〜数百回バックテスト）"):
            grid = logic.grid_search_params(df, params)
        if grid.empty:
            st.warning("結果が空でした。")
        else:
            st.dataframe(grid)
            st.caption("上位（scoreが高い）パラメータの傾向を見て、スキャン条件の設計に反映してください。")

with tab_ai:
    st.subheader("AIコメント（任意）")
    st.caption("APIキーを入れた場合のみ生成します。入れなくてもスキャン/分析は動作します。")
    if not api_key:
        st.info("OpenAI API Key を入れると、AIコメントが生成できます（任意）。")
    else:
        ctx = {
            "pair_label": pair_label,
            "price": float(latest["Close"]),
            "rsi": float(latest["RSI"]),
            "atr": float(latest["ATR"]),
            "sma5": float(latest.get("SMA_5", float("nan"))),
            "sma25": float(latest.get("SMA_25", float("nan"))),
            "pf": bt.profit_factor,
            "avg_r": bt.expectancy_r,
            "win_rate": bt.win_rate,
            "entry_price": plan.get("entry_price") if plan else float(latest["Close"]),
            "stop_price": plan.get("stop_price") if plan else float(latest["Close"]),
            "tp1_price": plan.get("tp1_price") if plan else float(latest["Close"]),
            "tp2_price": plan.get("tp2_price") if plan else float(latest["Close"]),
            "time_stop_days": getattr(params, "time_stop_days", 10),
            "shares": plan.get("shares") if plan else 0,
            "max_dd": bt.max_drawdown,
        }

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 市場局面・優位性コメント"):
                with st.spinner("生成中…"):
                    st.session_state.report_analysis = logic.get_ai_market_analysis(api_key, ctx)
            st.text_area("市場局面コメント", value=st.session_state.report_analysis, height=260)

        with col2:
            if st.button("🧾 注文実行（OCO/逆指値）指示"):
                with st.spinner("生成中…"):
                    st.session_state.report_strategy = logic.get_ai_order_strategy(api_key, ctx)
            st.text_area("注文実行コメント", value=st.session_state.report_strategy, height=260)

        if st.button("🧩 ポートフォリオ運用コメント"):
            with st.spinner("生成中…"):
                st.session_state.report_portfolio = logic.get_ai_portfolio(api_key, ctx)
        st.text_area("ポートフォリオコメント", value=st.session_state.report_portfolio, height=220)
