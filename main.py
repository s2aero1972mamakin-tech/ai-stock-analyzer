
# main.py
# ============================================================
# 🤖 ChatGPT連携型 日本株（〜1ヶ月スイング）全自動スキャナ + バックテスト
# - スキャン：JPX全銘柄 → 勝ちやすい局面フィルタ → 上位候補にバックテスト → TOP3表示
# - バックテスト：PF / 平均R / 勝率 / 最大DD / エクイティカーブ
# - 注文書：Rベース（SL/TP1/TP2/時間切れ/建値移動）で数値化
#
# 重要：OpenAI API Key は “AIコメント生成” にのみ使用（スキャン/バックテストは不要）
# ============================================================

import math
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import datetime

import logic
import json

TOKYO = pytz.timezone("Asia/Tokyo")

import inspect
import dataclasses


APP_BUILD = "2026-02-26T05:48:29Z / diagfix"

def _st_plotly(fig, **kwargs):
    """plotly_chart wrapper for Streamlit versions without use_container_width."""
    try:
        if "use_container_width" in inspect.signature(st.plotly_chart).parameters:
            return st.plotly_chart(fig, width='stretch', **kwargs)
    except Exception:
        pass

        render_diag_sidebar(diag_ph)
    return st.plotly_chart(fig, **kwargs)


st.set_page_config(layout="wide", page_title="AI日本株 スイングスキャナ", page_icon="🤖")
st.title("🤖 日本株（〜1ヶ月）スイング：スキャン + バックテスト + 注文書")
st.caption("※勝率だけではなく「利確（平均利益）」も含めた期待値（AvgR / PF）で候補を選別します。")


# -------------------------
# Session state
# -------------------------
def _init_state():
    defaults = {
        "target_ticker": None,
        "pair_label": None,
        "auto_candidates": [],
        "scan_meta": {},
        "report_strategy": "",
        "report_analysis": "",
        "report_portfolio": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# -------------------------
# Helpers (robustness)
# -------------------------
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
            st.markdown(f"**mode**: `{diag.get('mode','')}`")
            if diag.get("relax_level") is not None:
                st.markdown(f"**relax_level**: `{diag.get('relax_level')}`")
            if diag.get("selected_sectors"):
                st.markdown("**selected_sectors**")
                st.json(diag.get("selected_sectors", []))

            st.markdown("**filter_stats**")
            st.json(diag.get("filter_stats", {}))

            if diag.get("params_effective"):
                st.markdown("**params_effective**")
                st.json(diag.get("params_effective", {}))

            if diag.get("auto_relax_trace"):
                st.markdown("**auto_relax_trace**")
                st.json(diag.get("auto_relax_trace", []))

            if diag.get("error"):
                st.error(str(diag.get("error")))

            try:
                ts = str(diag.get("timestamp", "latest"))
                safe_ts = ts.replace(":", "").replace(" ", "_").replace("/", "-")
                diag_json = json.dumps(diag, ensure_ascii=False, indent=2, default=str)
                st.download_button(
                    "⬇️ 診断JSONをダウンロード",
                    data=diag_json.encode("utf-8"),
                    file_name=f"scan_diag_{safe_ts}.json",
                    mime="application/json",
                )
            except Exception as e:
                st.caption(f"診断JSONの生成に失敗: {e}")


# -------------------------
# Sidebar: Settings
# -------------------------
st.sidebar.header("⚙️ スキャン設定（期待値最大化）")
st.sidebar.caption(f"build: {APP_BUILD}")

capital = st.sidebar.number_input("運用軍資金（円）", value=300000, step=10000, min_value=10000)
risk_pct = st.sidebar.slider("1トレード許容損失（%）", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
if risk_pct >= 10.0:
    st.sidebar.warning("⚠️ 許容損失が大きいほど、短期の連敗で資金が急減しやすくなります。")

budget = st.sidebar.number_input("単元（100株）購入上限（円）", value=int(capital), step=10000, min_value=10000)

entry_mode = st.sidebar.selectbox("エントリー型", ["pullback（押し目反発）", "breakout（出来高ブレイク）"], index=0)
entry_mode_key = "pullback" if entry_mode.startswith("pullback") else "breakout"

st.sidebar.markdown("#### フィルタ（勝ちやすい局面）")
rsi_low, rsi_high = st.sidebar.slider("RSI範囲", min_value=10, max_value=90, value=(40, 65), step=1)
pb_low, pb_high = st.sidebar.slider("25日線乖離（%）(押し目用)", min_value=-20.0, max_value=5.0, value=(-6.0, -1.0), step=0.5)
atr_min, atr_max = st.sidebar.slider("ATR%（動く幅）", min_value=0.5, max_value=15.0, value=(1.0, 6.0), step=0.5)
vol_min = st.sidebar.number_input("20日平均出来高 下限（株数）", value=100000, step=10000, min_value=0)

# 売買代金フィルタ（推奨：株価×出来高）
turnover_min_m = st.sidebar.number_input("20日平均 売買代金 下限（百万円）", value=0.0, step=10.0, min_value=0.0)
regime_filter = st.sidebar.checkbox("地合いフィルタ（N225>SMA200 のときだけ）", value=False)
min_trades_bt = st.sidebar.slider("バックテスト最低トレード数（少数トレードの誤差対策）", 0, 30, 8, step=1)
pullback_allow_sma5_trigger = st.sidebar.checkbox("押し目トリガーを緩和（Close>SMA5 も許可）", value=True)

st.sidebar.markdown("#### 出口（利確を伸ばす）")
atr_mult = st.sidebar.slider("損切: ATR倍率", 0.5, 4.0, 1.5, step=0.1)
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
    turnover_avg20_min_yen=float(turnover_min_m) * 1_000_000.0,
    regime_filter=bool(regime_filter),
    min_trades_bt=int(min_trades_bt),
    pullback_allow_sma5_trigger=bool(pullback_allow_sma5_trigger),
    entry_mode=entry_mode_key,
    atr_mult_stop=float(atr_mult),
    tp1_r=float(tp1_r),
    tp2_r=float(tp2_r),
    time_stop_days=int(time_stop_days),
    risk_pct=float(risk_pct),
)


st.sidebar.markdown("---")
st.sidebar.subheader("🚀 全銘柄スキャン")
scan_label = "🔥 スキャン開始（セクター→銘柄スキャン）" if sector_prefilter else "🔥 スキャン開始（JPX全銘柄）"
scan_btn = st.sidebar.button(scan_label, type="primary")
# ---- last scan diagnostics (persist across reruns / shows immediately) ----
diag_ph = st.sidebar.empty()
render_diag_sidebar(diag_ph)

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
    with st.status("スキャン＆バックテスト中…", expanded=True) as status:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total, info):
            pct = int((current / max(1, total)) * 100)
            progress_bar.progress(pct)
            status_text.text(f"🔍 {info} ({current}/{total})")

        try:

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

        except Exception as e:

            res = {

                "regime_ok": True,

                "candidates": [],

                "prelim_count": 0,

                "bt_count": 0,

                "selected_sectors": [],

                "sector_ranking": [],

                "filter_stats": {"exception": str(e)},

                "params_effective": {},

                "auto_relax_trace": [],

                "error": str(e),

            }

            status.update(label=f"❌ スキャン中に例外: {e}", state="error", expanded=True)


        # Persist diagnostics (kept even after st.rerun())
        try:
            st.session_state["last_scan_diag"] = {
                "timestamp": str(datetime.datetime.now()),
                "mode": (res.get("params_effective", {}) or {}).get("entry_mode"),
                "relax_level": res.get("relax_level", 0),
                "selected_sectors": res.get("selected_sectors", []),
                "filter_stats": res.get("filter_stats", {}),
                "params_effective": res.get("params_effective", {}),
                "auto_relax_trace": res.get("auto_relax_trace", []),
                "error": res.get("error"),
            }
        except Exception:
            pass

        # Update sidebar diagnostic panel immediately in this run
        try:
            render_diag_sidebar(diag_ph)
        except Exception:
            pass

        st.session_state.scan_meta = res
        candidates = res.get("candidates", [])
        st.session_state.auto_candidates = candidates

        if candidates:
            best = candidates[0]
            st.session_state.target_ticker = best["ticker"]
            st.session_state.pair_label = f"{best['ticker']} {best['name']}"
            st.session_state.report_strategy = ""
            st.session_state.report_analysis = ""
            st.session_state.report_portfolio = ""
            status.update(label="✅ 完了：候補を抽出しました", state="complete", expanded=False)
        else:
            status.update(label="⚠️ 条件クリア銘柄なし", state="complete", expanded=False)

            err = res.get("error", "")
            relax_level = int(res.get("relax_level", 0))
            params_eff = res.get("params_effective", {})
            stats = res.get("filter_stats", {}) or {}

            st.sidebar.error("条件クリア銘柄が0件でした。下の診断を見て、まずは絞り込み条件を緩めてください。")
            if err:
                st.sidebar.caption(f"理由: {err}")

            # show whether auto-relax was tried
            if relax_level >= 1:
                st.sidebar.warning("自動緩和（条件をゆるめて再スキャン）を1回実施しましたが、まだ0件でした。")

            if params_eff:
                st.sidebar.markdown("**今回スキャンに使われた条件（effective）**")
                st.sidebar.json(params_eff)

            if stats:
                st.sidebar.markdown("**どこで落ちているか（ざっくり）**")
                # show key stats compactly
                keys = ["universe","data_ok","budget_ok","trend_ok","rsi_ok","atr_ok","vol_ok","setup_ok","prelim_pass",
                        "fail_data_short","fail_budget","fail_trend","fail_rsi","fail_atr","fail_vol","fail_setup"]
                compact = {k: stats.get(k) for k in keys if k in stats}
                st.sidebar.json(compact)
            # ダウンロード用（診断JSON）
            try:
                ts_local = datetime.now().strftime('%Y%m%d_%H%M%S')
                diag = {
                    'timestamp': ts_local,
                    'mode': str(params_eff.get('entry_mode', entry_mode_key)),
                    'relax_level': int(relax_level),
                    'params_effective': params_eff,
                    'filter_stats': stats,
                }
                diag_json = json.dumps(diag, ensure_ascii=False, indent=2, default=str)
                st.sidebar.download_button('⬇️ この診断JSONをダウンロード', data=diag_json, file_name=f'scan_diag_{ts_local}.json', mime='application/json')
            except Exception as e:
                st.sidebar.caption(f'診断JSONの生成に失敗: {e}')

            st.sidebar.markdown("---")
            st.sidebar.markdown("**0件になりやすい原因（この順で試してください）**")
            st.sidebar.write("1) エントリー型を **breakout（出来高ブレイク）** に変更")
            st.sidebar.write("2) 20日平均出来高 下限を **100000 → 30000** くらいに下げる")
            st.sidebar.write("3) RSI範囲を **40-65 → 35-70** に広げる")
            st.sidebar.write("4) 押し目乖離を **-6〜-1 → -10〜0** に広げる（pullbackの場合）")
            st.sidebar.write("5) ATR%上限を **6 → 10** に上げる（動く銘柄を許容）")

    st.rerun()


# -------------------------
# Sidebar: candidate picker
# -------------------------
if st.session_state.auto_candidates:
    meta = st.session_state.scan_meta or {}
    prelim_count = meta.get("prelim_count")
    bt_count = meta.get("bt_count")
    if prelim_count is not None:
        st.sidebar.caption(f"スキャン通過: {prelim_count} / バックテスト実施: {bt_count}")

    universe = meta.get("universe")
    if universe is not None:
        st.sidebar.caption(f"走査ユニバース: {universe} 銘柄")

    selected_sectors = meta.get("selected_sectors") or []
    if selected_sectors:
        st.sidebar.caption("セクター絞り込み: " + " / ".join([str(s) for s in selected_sectors]))
        ranking = meta.get("sector_ranking") or []
        if ranking:
            with st.sidebar.expander("📊 セクター強度（上位15）", expanded=False):
                st.dataframe(pd.DataFrame(ranking))

    with st.sidebar.expander("📌 発掘された買い候補 (TOP3)", expanded=True):
        for c in st.session_state.auto_candidates:
            label = f"{c['ticker']} {c['name']}"
            stats = (
                f"AvgR {c.get('bt_avg_r', 0):.2f} / "
                f"PF {c.get('bt_pf', 0):.2f} / "
                f"Win {c.get('bt_win_rate', 0)*100:.0f}% / "
                f"Trades {c.get('bt_trades', 0)}"
            )
            if st.sidebar.button(f"分析：{label}（{stats}）", key=f"btn_{c['ticker']}"):
                st.session_state.target_ticker = c["ticker"]
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
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="SMA5", line=dict(width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="SMA25", line=dict(width=2)))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="SMA75", line=dict(width=1)))
fig.update_layout(xaxis_rangeslider_visible=False, height=520, margin=dict(l=0, r=0, t=30, b=0))
_st_plotly(fig)

st.markdown("### 📌 実行タブ")
tab_plan, tab_bt, tab_tune, tab_ai = st.tabs(["📝 注文書（ロジック）", "📈 バックテスト", "🧪 パラメータ検証", "🧠 AIコメント（任意）"])

with tab_plan:
    if not plan:
        st.warning("注文書を生成できませんでした（ATR未算出など）。")
    else:
        st.subheader("ロジック注文書（期待値型）")
        st.write("**狙い：TP1で勝ちを確保しつつ、残りで+3Rを狙って期待値を作る**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("想定エントリー（目安）", f"{plan['entry_price']:.1f}円")
        c2.metric("損切（SL）", f"{plan['stop_price']:.1f}円")
        c3.metric("利確1（TP1）", f"{plan['tp1_price']:.1f}円")
        c4.metric("利確2（TP2）", f"{plan['tp2_price']:.1f}円")

        st.write(f"- 損切幅（1株あたり）: **{plan['r_yen_per_share']:.2f}円**（ATR×{plan['atr_mult_stop']:.1f}）")
        st.write(f"- 時間切れ: **{plan['time_stop_days']}営業日**でTP1未達なら撤退")
        st.write(f"- 推奨株数（100株単位）: **{plan['shares']}株**（許容損失 {risk_pct:.1f}% = {plan['risk_yen']:.0f}円 目安）")

        if plan["shares"] == 0:
            st.error("⚠️ この銘柄は、設定した損切幅だと100株単位でリスク/資金制約を満たせません。")
            st.write("対策：①損切幅を縮める（ATR倍率↓） ②許容損失%↑（注意） ③株価が低い銘柄を選ぶ ④単元以外（S株等）を使う")

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

    if bt.equity_curve_r is not None and not bt.equity_curve_r.empty:
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
            st.warning("十分なトレード数が出る組合せがありませんでした。期間を延ばす/条件を緩めると改善します。")
        else:
            st.success("検証完了：上位から表示します（AvgR優先）")
            st.dataframe(grid.head(30))
            best = grid.iloc[0].to_dict()
            st.info(f"最良候補（参考）：mode={best['mode']} / RSI={best['rsi']} / 乖離={best['pullback']} / AvgR={best['avg_r']:.2f} / PF={best['pf']:.2f}")

with tab_ai:
    if not api_key:
        st.info("左サイドバーでOpenAI API Keyを入力すると、AIコメントを生成できます（任意）。")
    else:
        ctx = {
            "pair_label": pair_label,
            "price": float(latest["Close"]),
            "rsi": float(latest["RSI"]),
            "atr": float(latest["ATR"]),
            "sma5": float(latest["SMA_5"]),
            "sma25": float(latest["SMA_25"]),
            "pf": float(bt.profit_factor) if math.isfinite(bt.profit_factor) else None,
            "avg_r": float(bt.expectancy_r),
            "win_rate": float(bt.win_rate),
            "max_dd": float(bt.max_drawdown),
            **plan,
        }

        t1, t2, t3 = st.tabs(["📝 AI命令書", "📊 AI分析", "💰 AI保有判断"])
        with t1:
            if st.button("📝 AI命令書を生成"):
                st.session_state.report_strategy = logic.get_ai_order_strategy(api_key, ctx)
            st.markdown(st.session_state.report_strategy)

        with t2:
            if st.button("📊 AI分析を生成"):
                st.session_state.report_analysis = logic.get_ai_analysis(api_key, ctx)
            st.markdown(st.session_state.report_analysis)

        with t3:
            if st.button("💰 AI判断を生成"):
                st.session_state.report_portfolio = logic.get_ai_portfolio(api_key, ctx)
            st.markdown(st.session_state.report_portfolio)


# -------------------------
# Export report
# -------------------------
st.markdown("---")
report = f"""【日本株スイング レポート】
生成日時: {datetime.now(TOKYO).strftime('%Y-%m-%d %H:%M')}
対象: {pair_label}

■ ロジック注文書
entry={plan.get('entry_price')}
stop={plan.get('stop_price')}
tp1={plan.get('tp1_price')}
tp2={plan.get('tp2_price')}
shares={plan.get('shares')}
time_stop_days={plan.get('time_stop_days')}

■ バックテスト（簡易）
trades={bt.n_trades}
win_rate={bt.win_rate}
PF={bt.profit_factor}
AvgR={bt.expectancy_r}
MaxDD={bt.max_drawdown}

■ AI命令書
{st.session_state.report_strategy}

■ AI分析
{st.session_state.report_analysis}

■ AI保有判断
{st.session_state.report_portfolio}
"""
st.download_button(
    label="💾 レポートをテキスト保存",
    data=report,
    file_name=f"TradeLog_{ticker}_{datetime.now(TOKYO).strftime('%Y%m%d_%H%M')}.txt",
    mime="text/plain",
)
