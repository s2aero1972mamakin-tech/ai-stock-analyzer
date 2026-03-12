import os
import json
import time
import traceback
import html
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd

import logic

JST = timezone(timedelta(hours=9))

def render_cards_selected(df: pd.DataFrame):
    for _, r in df.iterrows():
        sym = r.get("銘柄","")
        strat = r.get("推奨方式","")
        title = f"{sym} / {strat}"
        items = []
        for k in ["現在値（終値）","Entry目安","SL目安","TP目安","RR","実質RR","価格更新状態","価格更新メモ","再計算失敗フラグ","売買優先区分","実行優先帯","実行優先スコア","今すぐ発注スコア","単元予算可否","単元推奨可否","単元必要資金(円)","単元想定損失(円)","推奨株数","推奨投資額(円)","想定損失(円)","発注不可理由","selected_now判定","selected_now除外理由","selected_now空理由集計","総合スコア"]:
            if k in df.columns:
                v = r.get(k)
                items.append(f"{k}:{v}")
        st.markdown(
            f'<div class="card"><div class="card-title">{title}</div><div class="kv">'
            + "".join([f"<span>{html.escape(str(x))}</span>" for x in items])
            + "</div></div>",
            unsafe_allow_html=True,
        )

def render_cards_sector(sec: pd.DataFrame):
    for _, r in sec.iterrows():
        st.markdown(
            f'<div class="card"><div class="card-title">#{int(r.get("順位",0))} {html.escape(str(r.get("セクター（33業種）","")))}'
            f'</div><div class="kv"><span>強度:{r.get("強度（中央値）")}</span></div></div>',
            unsafe_allow_html=True,
        )

def render_cards_guide(df: pd.DataFrame):
    for _, r in df.iterrows():
        sym = r.get("銘柄","")
        name = r.get("企業名","")
        strat = r.get("推奨方式","")
        title = f"{sym} / {name} / {strat}"
        items = []
        for k in ["セクター","売買優先区分","実行優先帯","発注単位","単元予算可否","単元推奨可否","推奨株数","推奨投資額(円)","想定損失(円)","Entry目安","SL目安","TP目安","最大保有","Entry状態"]:
            if k in df.columns:
                items.append(f"{k}:{r.get(k)}")
        st.markdown(
            f'<div class="card"><div class="card-title">{title}</div><div class="kv">'
            + "".join([f"<span>{html.escape(str(x))}</span>" for x in items])
            + "</div></div>",
            unsafe_allow_html=True,
        )


def prepare_selected_view(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame(columns=getattr(logic, "LIVE_OUTPUT_SCHEMA", []))
    df = df.copy().reset_index(drop=True)
    if "銘柄" in df.columns and "symbol" not in df.columns: df["symbol"] = df["銘柄"]
    if "銘柄名" in df.columns and "name" not in df.columns: df["name"] = df["銘柄名"]
    if "name" in df.columns and "企業名" not in df.columns: df["企業名"] = df["name"]
    if "セクター" in df.columns and "sector33_name" not in df.columns: df["sector33_name"] = df["セクター"]
    if "推奨方式" in df.columns and "strategy_name" not in df.columns: df["strategy_name"] = df["推奨方式"]
    if "総合スコア" in df.columns and "final_score" not in df.columns: df["final_score"] = df["総合スコア"]
    schema = list(getattr(logic, "LIVE_OUTPUT_SCHEMA", [])) or ["順位","銘柄"]
    if "順位" in df.columns:
        df = df.drop(columns=["順位"], errors="ignore")
    df.insert(0, "順位", range(1, len(df)+1))
    col_map = {"symbol":"銘柄","sector33_name":"セクター","final_score":"総合スコア","strategy_name":"推奨方式"}
    for k,v in list(col_map.items()):
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    for v in schema:
        if v not in df.columns:
            df[v] = None
    df = df[schema]
    try:
        for c in ["企業名","セクター","推奨方式","売買優先区分","実行優先帯","発注単位","単元予算可否","単元推奨可否","単元予算判定理由","単元推奨判定理由","Entry状態","発注不可理由","価格更新状態","価格更新メモ","選定区分","selected_now判定","selected_now除外理由","selected_now空理由集計"]:
            if c in df.columns:
                df[c] = (df[c].astype(str).replace(["None","none","nan","NaN",""], "不明" if c in ["企業名","セクター"] else "").str.strip())
        for c in ["現在値（終値）","Entry目安","SL目安","TP目安","RR","実質RR","再計算失敗フラグ","単元必要資金(円)","単元想定損失(円)","推奨投資額(円)","想定損失(円)","総合スコア","実行優先度","実行優先スコア","今すぐ発注スコア","推奨株数","元Entry目安","元SL目安","元TP目安","元RR","元総合スコア"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
    except Exception:
        pass
    return df


def prepare_execution_views_for_display(live_df: pd.DataFrame, now_df: pd.DataFrame, wait_df: pd.DataFrame):
    live_view = prepare_selected_view(live_df)
    now_view = prepare_selected_view(now_df)
    wait_view = prepare_selected_view(wait_df)

    common_cols = []
    for df_part in [live_view, now_view, wait_view]:
        if isinstance(df_part, pd.DataFrame):
            for c in df_part.columns.tolist():
                if c not in common_cols:
                    common_cols.append(c)

    if not common_cols:
        return live_view, now_view, wait_view

    out = []
    for df_part in [live_view, now_view, wait_view]:
        if not isinstance(df_part, pd.DataFrame):
            df_part = pd.DataFrame()
        df_part = df_part.copy()
        for c in common_cols:
            if c not in df_part.columns:
                df_part[c] = None
        out.append(df_part[common_cols])
    return tuple(out)


def render_selected_section(title: str, caption: str, df: pd.DataFrame, mobile_mode: bool, show_top_n: int, download_name: str | None = None):
    st.subheader(title)
    st.caption(caption)
    if isinstance(df, pd.DataFrame) and len(df):
        if mobile_mode:
            render_cards_selected(df.head(show_top_n))
            with st.expander("表で見る（PC向け）", expanded=False):
                st.dataframe(df, width="stretch")
        else:
            st.dataframe(df, width="stretch")
        if download_name:
            st.download_button(
                f"⬇️ {download_name}",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=download_name,
                mime="text/csv",
                use_container_width=True,
                on_click="ignore",
            )
    else:
        st.info("該当銘柄がありません。")


def prepare_guide_view(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    guide = logic.build_live_linked_guide(df, max_rows=max_rows) if isinstance(df, pd.DataFrame) and len(df) else pd.DataFrame()
    if isinstance(guide, pd.DataFrame) and len(guide):
        try:
            for c in ["銘柄","企業名","セクター","推奨方式","売買優先区分","実行優先帯","発注単位","単元予算可否","単元推奨可否","推奨株数","推奨投資額(円)","想定損失(円)","Entry目安","SL目安","TP目安","最大保有","Entry状態"]:
                if c not in guide.columns:
                    guide[c] = None
            guide = guide[["銘柄","企業名","セクター","推奨方式","売買優先区分","実行優先帯","発注単位","単元予算可否","単元推奨可否","推奨株数","推奨投資額(円)","想定損失(円)","Entry目安","SL目安","TP目安","最大保有","Entry状態"]]
            for c in ["企業名","セクター","推奨方式","売買優先区分","実行優先帯","発注単位","単元予算可否","単元推奨可否","Entry状態"]:
                guide[c] = (guide[c].astype(str).replace(["None","none","nan","NaN",""], "不明" if c in ["企業名","セクター"] else "").str.strip())
            guide = guide.head(int(max_rows)).reset_index(drop=True)
            guide.insert(0, "順位", range(1, len(guide)+1))
        except Exception:
            pass
    return guide


def render_guide_section(title: str, caption: str, guide: pd.DataFrame, mobile_mode: bool, show_top_n: int):
    st.subheader(title)
    st.caption(caption)
    if isinstance(guide, pd.DataFrame) and len(guide):
        if mobile_mode:
            render_cards_guide(guide.head(show_top_n))
            with st.expander("表で見る（PC向け）", expanded=False):
                st.dataframe(guide, width="stretch")
        else:
            st.dataframe(guide, width="stretch")
    else:
        st.info("価格目安の生成に必要なデータが不足しています。")

st.set_page_config(page_title="JPX Swing AI", layout="wide")
if "scan_results_ready" not in st.session_state:
    st.session_state["scan_results_ready"] = False
    st.session_state["scan_diag"] = None
    st.session_state["scan_selected_live"] = pd.DataFrame()
    st.session_state["scan_selected_now"] = pd.DataFrame()
    st.session_state["scan_selected_wait"] = pd.DataFrame()
    st.session_state["scan_now_guide"] = pd.DataFrame()
    st.session_state["scan_wait_guide"] = pd.DataFrame()


# ---- スマホ最適化CSS（Streamlit Cloudでも効く）----
st.markdown(
    """
<style>
/* 全体の余白を少し詰める */
.block-container {padding-top: 1.0rem; padding-bottom: 1.0rem;}
/* タイトルを小さめに */
h1 {font-size: 1.6rem !important; margin-bottom: 0.25rem !important;}
/* モバイルではテーブルが横に溢れやすいので、カード表示優先 */
@media (max-width: 768px){
  h1 {font-size: 1.35rem !important;}
  .stDataFrame {overflow-x: auto;}
}
.small-note {font-size: 0.85rem; opacity: 0.85;}
.card {border: 1px solid rgba(49,51,63,0.2); border-radius: 12px; padding: 12px; margin-bottom: 10px;}
.card-title {font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;}
.kv {display: flex; gap: 10px; flex-wrap: wrap;}
.kv span {background: rgba(49,51,63,0.08); padding: 4px 8px; border-radius: 10px; font-size: 0.88rem;}
</style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# JPX Swing AI")
st.markdown('<div class="small-note">固定条件：最大保有10営業日 / 利確=+1.5ATR(14) / 損切=-1.0ATR(14) — DB: Neon(Postgres)</div>', unsafe_allow_html=True)


# --- Streamlit secrets -> env bridge ---
if "NEON_DATABASE_URL" in st.secrets and not os.environ.get("NEON_DATABASE_URL"):
    os.environ["NEON_DATABASE_URL"] = st.secrets["NEON_DATABASE_URL"]

st.sidebar.header("💰 資金設定")
capital_total = st.sidebar.number_input("運用資金（円）", min_value=50000.0, value=300000.0, step=10000.0)
max_positions = st.sidebar.selectbox("同時保有数（最大）", [1,2,3], index=0, key="max_positions")
mobile_mode = st.sidebar.toggle("📱スマホ表示（カード）", value=True)

show_top_n = int(max_positions)

st.sidebar.header("🗄️ データベース（Neon）")
if st.sidebar.button("🔁 33業種を再同期（JPX）", use_container_width=True):
    with st.spinner("JPXマスタから33業種を更新中..."):
        upd, stt = logic.update_sector33_from_jpx()
    st.sidebar.success(f"33業種 更新: {upd} ({stt})")


db_url_ok, db_msg = logic.check_db_config()
if not db_url_ok:
    st.sidebar.error(db_msg)
    st.sidebar.markdown(
        "**設定方法（Streamlit Cloud）**\n"
        "- App settings → Secrets に以下を追加\n"
        "```toml\n"
        "NEON_DATABASE_URL = \"postgresql://...\"\n"
        "```"
    )
    st.stop()
else:
    st.sidebar.success("DB接続設定OK")

with st.sidebar.expander("🔧 ドライバ診断（psycopg/psycopg2）", expanded=False):
    st.json(logic.driver_diagnostics())

# --- Universe registration (IMPORTANT) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📥 銘柄マスタ登録（初回だけ）")
with st.sidebar.expander("銘柄リストを登録/更新", expanded=False):
    st.write("Neon側に銘柄マスタが **0件だと更新もスキャンもできません**。初回だけ登録してください。")
    st.caption("形式: 1行1銘柄（例: 7203.T） or CSV（列名: ticker / symbol / code）。手動が面倒ならJPX公式の一覧（Excel）から自動取得できます。")
    if st.button("🌐 JPX公式一覧から自動取得して登録", use_container_width=True, key="btn_universe_autofetch"):
        try:
            n, msg = logic.universe_autofetch_from_jpx()
            if n > 0:
                st.success(f"JPXから取得して登録しました: {n} 件")
            else:
                st.error(msg)
        except Exception:
            st.error("自動取得に失敗")
            st.code(traceback.format_exc())
    uploaded = st.file_uploader("CSVをアップロード", type=["csv"], key="upl_universe_csv")
    text_in = st.text_area("または貼り付け（改行区切り）", value="", height=140, key="universe_paste")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("登録する", use_container_width=True):
            try:
                n, msg = logic.universe_register_from_inputs(uploaded, text_in)
                if n > 0:
                    st.success(f"登録しました: {n} 件")
                else:
                    st.error(msg)
            except Exception:
                st.error("登録に失敗")
                st.code(traceback.format_exc())
    with c2:
        tmpl = "ticker\n7203.T\n6758.T\n9984.T\n"
        st.download_button("テンプレCSV", data=tmpl.encode("utf-8"), file_name="universe_template.csv", mime="text/csv", use_container_width=True)

# --- DB status ---
st.sidebar.markdown("---")
st.sidebar.subheader("📌 DB状況")
status = logic.get_db_status()
if status.get("ok"):
    last_ts = status.get("last_update_utc")
    last_jst = None
    if last_ts:
        try:
            dt = datetime.fromisoformat(last_ts.replace("Z","+00:00"))
            last_jst = dt.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            last_jst = str(last_ts)

    st.sidebar.write(f"銘柄マスタ数: **{status.get('universe_count','?')}**")
    st.sidebar.write(f"保存銘柄数(OHLC): **{status.get('symbols_count','?')}**")
    st.sidebar.write(f"保存行数(OHLC): **{status.get('rows_count','?')}**")
    st.sidebar.write(f"最新取引日: **{status.get('latest_trade_date','None')}**")
    st.sidebar.write(f"最終DB更新(JST): **{last_jst or '不明'}**")
    stale_hours = st.sidebar.slider("更新忘れ警告（時間）", 6, 72, 30, 1)
    if status.get("hours_since_update") is not None and status["hours_since_update"] > stale_hours:
        st.sidebar.warning(f"⚠ DB更新が古い可能性：{status['hours_since_update']:.1f} 時間前")
else:
    st.sidebar.warning(status.get("message","DB状態取得に失敗"))

# --- Settings ---
st.sidebar.markdown("---")
st.sidebar.header("⚙ スキャン設定（Cloud最適化 3段階）")
stage0_keep = st.sidebar.slider("Stage0 通過上限（全件→）", 300, 2500, 1200, 100)
stage1_keep = st.sidebar.slider("Stage1 通過上限（→）", 80, 1000, 300, 20)
stage2_keep = st.sidebar.slider("最終選定数（→）", 20, 300, 60, 10)

min_price = st.sidebar.number_input("最低株価（円）", min_value=0.0, value=300.0, step=50.0)
min_avg_volume = st.sidebar.number_input("最低出来高（20日平均）", min_value=0.0, value=30000.0, step=5000.0)
atr_pct_min = st.sidebar.number_input("ATR% 下限", min_value=0.0, value=1.0, step=0.1)
atr_pct_max = st.sidebar.number_input("ATR% 上限", min_value=0.0, value=8.0, step=0.5)

stage2_days = st.sidebar.slider("Stage2 利確評価に使う履歴日数", 60, 365, 180, 5)
stage2_min_bars = st.sidebar.slider("Stage2 最低バー数（短期は暫定評価）", 40, 140, 60, 5)
include_fund = st.sidebar.checkbox("🧾 財務簡易チェック（上位銘柄だけ。バフェット簡易スコア/イベント注意を付与）", value=True)
fund_top_n = st.sidebar.slider("財務/イベント取得数（上位Nのみ）", 0, 60, 20, 5)

st.sidebar.markdown("---")
st.sidebar.header("🔄 DB更新（増分）")
update_days_back = st.sidebar.slider("更新取得日数（直近）", 3, 365, 120, 1)
keep_days = st.sidebar.slider("DB保持日数（古い分は削除）", 120, 2000, 600, 10)
chunk_size = st.sidebar.slider("一括取得チャンク", 20, 300, 120, 10)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🧱 初期化", use_container_width=True):
        try:
            logic.ensure_schema()
            st.success("スキーマOK")
        except Exception:
            st.error("スキーマ作成に失敗")
            st.code(traceback.format_exc())

with col2:
    if st.button("⬇️ 更新", use_container_width=True):
        t0 = time.time()
        try:
            with st.spinner("DB更新中...（銘柄マスタに基づいて取得）"):
                rep = logic.update_db_incremental(days_back=update_days_back, keep_days=keep_days, chunk_size=chunk_size)
            st.success(f"更新完了: {rep.get('upserted_rows',0)}行 / 失敗{rep.get('failed_symbols',0)}")
            st.caption(f"所要: {time.time()-t0:.1f} 秒")
            st.json(rep)
            st.rerun()
        except Exception:
            st.error("DB更新に失敗")
            st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.header("🧾 診断JSON")
last_diag = logic.load_last_diag()
if last_diag:
    st.sidebar.download_button(
        "⬇️ 前回の診断JSONをダウンロード",
        data=json.dumps(last_diag, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="diag_last.json",
        mime="application/json",
        use_container_width=True,
    )
else:
    st.sidebar.info("前回の診断JSONなし（スキャン実行後に保存）")

# --- Main run ---
st.markdown("---")
colA, colB = st.columns([1,1])
with colA:
    run_scan = st.button("🚀 スキャン実行（Stage0→1→2）", use_container_width=True)
with colB:
    show_debug = st.toggle("🛠️ デバッグ表示（traceback）", value=False)

if run_scan:
    t0 = time.time()
    try:
        with st.spinner("スキャン中...（DB読み込み→段階絞り込み→利確スコア）"):
            out = logic.run_scan_3stage(
                capital_total=capital_total,
                max_positions=int(max_positions),

                stage0_keep=stage0_keep,
                stage1_keep=stage1_keep,
                stage2_keep=stage2_keep,
                min_price=min_price,
                min_avg_volume=min_avg_volume,
                atr_pct_min=atr_pct_min,
                atr_pct_max=atr_pct_max,
                stage2_days=stage2_days,
                stage2_min_bars=stage2_min_bars,
                include_fundamentals=include_fund,
                fundamentals_top_n=fund_top_n,
            )
        elapsed = time.time() - t0
        diag = out.get("diag", {})
        diag["elapsed_sec"] = elapsed
        logic.save_last_diag(diag)

        df = out.get("selected")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        try:
            df = logic.refresh_topn_prices_and_recalc(
                df,
                top_n=20,
                capital_total=float(capital_total),
                max_positions=int(max_positions),
            )
        except Exception:
            pass

        try:
            live_df, now_df, wait_df = logic.build_live_execution_views(
                df,
                live_top=20,
                now_top=10,
                wait_top=20,
                now_rr_min=1.00,
                chase_rr_min=1.45,
                wait_rr_min=0.90,
                s_now_rr_min=1.50,
                s_now_max=1,
                chase_now_max=1,
            )
        except Exception:
            live_df, now_df, wait_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_view, now_view, wait_view = prepare_execution_views_for_display(live_df, now_df, wait_df)
        now_guide = prepare_guide_view(now_view, max_rows=int(max_positions))
        wait_guide = prepare_guide_view(wait_view, max_rows=min(10, len(wait_view)) if isinstance(wait_view, pd.DataFrame) else 10)

        st.session_state["scan_results_ready"] = True
        st.session_state["scan_diag"] = diag
        st.session_state["scan_selected_live"] = df_view
        st.session_state["scan_selected_now"] = now_view
        st.session_state["scan_selected_wait"] = wait_view
        st.session_state["scan_now_guide"] = now_guide
        st.session_state["scan_wait_guide"] = wait_guide

    except Exception:
        st.error("スキャン中にエラーが発生しました")
        if show_debug:
            st.code(traceback.format_exc())
        else:
            st.caption("（デバッグ表示をONにすると詳細tracebackを表示します）")

if st.session_state.get("scan_results_ready", False):
    diag = st.session_state.get("scan_diag", {})
    st.success(f"スキャン完了（{float(diag.get('elapsed_sec', 0.0)):.1f}秒） / mode={diag.get('mode','?')}")
    with st.expander("📊 診断（JSON）", expanded=False):
        st.json(diag)

    df_view = st.session_state.get("scan_selected_live", pd.DataFrame())
    now_view = st.session_state.get("scan_selected_now", pd.DataFrame())
    wait_view = st.session_state.get("scan_selected_wait", pd.DataFrame())
    now_guide = st.session_state.get("scan_now_guide", pd.DataFrame())
    wait_guide = st.session_state.get("scan_wait_guide", pd.DataFrame())

    render_selected_section(
        "🏆 AI最終選定銘柄（ライブ再計算後・全20件）",
        "selected_live_top20 / selected_now / selected_wait を同一親DataFrame・同一スキーマで生成しています。先頭は selected_now、次に重複なしの selected_wait、その後ろに残り候補を同じ実行優先順で並べています。",
        df_view,
        mobile_mode,
        show_top_n,
        "selected_live_top20.csv",
    )

    now_empty_reason = ""
    if isinstance(now_view, pd.DataFrame) and len(now_view) == 0:
        for src_df in [df_view, wait_view]:
            if isinstance(src_df, pd.DataFrame) and len(src_df) and "selected_now空理由集計" in src_df.columns:
                vals = src_df["selected_now空理由集計"].astype(str).str.strip()
                vals = vals[(vals != "") & (vals != "nan") & (vals != "None")]
                if len(vals):
                    now_empty_reason = vals.iloc[0]
                    break
        if now_empty_reason:
            st.warning(now_empty_reason)

    render_selected_section(
        "🟢 今すぐ発注ランキング",
        "単元株の発注圏を最優先にした即時発注候補です。追随可は、単元株の発注圏候補が0件のときに限り実質RR>=1.45を最大1件だけ補完します。S株例外も、単元株の発注圏・追随可補完が無い場合に限り実質RR>=1.50を最大1件だけ許可します。空の場合は除外理由集計を上に表示します。",
        now_view,
        mobile_mode,
        show_top_n,
        "selected_now.csv",
    )

    render_selected_section(
        "🟡 押し目待ちランキング",
        "重複なし・見送りなしの監視候補です。selected_now に入らなかった単元株の監視候補を先頭に置き、強いS株候補は補助的に後ろへ残します。",
        wait_view,
        mobile_mode,
        show_top_n,
        "selected_wait.csv",
    )

    render_guide_section(
        "🧭 今すぐ発注の価格目安（Entry/SL/TP）",
        "単元株の発注圏を先頭に表示し、追随可は発注圏候補が無い場合の補完候補だけを表示します。予算内可否と推奨可否も併せて確認できます。",
        now_guide,
        mobile_mode,
        show_top_n,
    )

    render_guide_section(
        "🧭 押し目待ちの価格目安（Entry/SL/TP）",
        "今は飛びつかず監視したい候補です。単元株で入れる強銘柄を優先し、S株は補助候補として後ろに並べます。価格未更新銘柄は今すぐ発注から外れ、総合表の警告列で確認できます。",
        wait_guide,
        mobile_mode,
        show_top_n,
    )

st.markdown("---")
st.caption("※注意：本ツールは投資助言ではありません。最終判断はご自身で行ってください。")
