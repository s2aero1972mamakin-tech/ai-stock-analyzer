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
        for k in ["現在値（終値）","Entry目安","SL目安","TP目安","RR","推奨株数","推奨投資額(円)","想定損失(円)","発注不可理由","総合スコア"]:
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
        sector = r.get("セクター","")
        strat = r.get("推奨方式","")
        title = f"{sym} / {name} / {strat}"
        items = []
        for k in ["セクター","発注単位","推奨株数","推奨投資額(円)","想定損失(円)","Entry目安","SL目安","TP目安","最大保有"]:
            if k in df.columns:
                items.append(f"{k}:{r.get(k)}")
        st.markdown(
            f'<div class="card"><div class="card-title">{title}</div><div class="kv">'
            + "".join([f"<span>{html.escape(str(x))}</span>" for x in items])
            + "</div></div>",
            unsafe_allow_html=True,
        )

st.set_page_config(page_title="JPX Swing AI", layout="wide")

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


include_fund = st.sidebar.checkbox("🧾 財務簡易チェック（上位銘柄のみ。バフェット簡易スコア/イベント注意を付与）", value=True)
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

        st.success(f"スキャン完了（{elapsed:.1f}秒） / mode={diag.get('mode','?')}")
        with st.expander("📊 診断（JSON）", expanded=False):
            st.json(diag)
        # セクター強度ランキングはバックエンド利用のみ（UI非表示）

        st.subheader("🏆 AI最終選定銘柄（利確スコア統合）")
        st.caption("どれを買うか？（利確評価＋資金効率でランキング）")
        st.caption("※ここは **Stage2（固定TP/SL/最大保有）で“利確が再現しやすい順”** に並べた最終ランキングです。")
        df = out.get("selected")
        # --- column bridge (logic.py output may use JP labels) ---
        if isinstance(df, pd.DataFrame) and len(df):
            if "銘柄" in df.columns and "symbol" not in df.columns: df["symbol"] = df["銘柄"]
            if "銘柄名" in df.columns and "name" not in df.columns: df["name"] = df["銘柄名"]
            if "name" in df.columns and "企業名" not in df.columns: df["企業名"] = df["name"]
            if "セクター" in df.columns and "sector33_name" not in df.columns: df["sector33_name"] = df["セクター"]
            if "3ヶ月リターン" in df.columns and "RET_3M" not in df.columns: df["RET_3M"] = df["3ヶ月リターン"]
            if "WF勝率（OOS）" in df.columns and "wf_oos_wr" not in df.columns: df["wf_oos_wr"] = df["WF勝率（OOS）"]
            if "WF損益比RR（OOS）" in df.columns and "wf_oos_rr" not in df.columns: df["wf_oos_rr"] = df["WF損益比RR（OOS）"]
            if "MC DD 5%（推定）" in df.columns and "mc_dd5" not in df.columns: df["mc_dd5"] = df["MC DD 5%（推定）"]
            if "Kelly最適化（f）" in df.columns and "kelly_f" not in df.columns: df["kelly_f"] = df["Kelly最適化（f）"]
            if "AIトレンド" in df.columns and "trend_score" not in df.columns: df["trend_score"] = df["AIトレンド"]
            if "推奨方式" in df.columns and "strategy_name" not in df.columns: df["strategy_name"] = df["推奨方式"]
            if "総合スコア" in df.columns and "final_score" not in df.columns: df["final_score"] = df["総合スコア"]
        if isinstance(df, pd.DataFrame) and len(df):
            df = df.reset_index(drop=True)
            if "順位" in df.columns:

                df = df.drop(columns=["順位"])

            df.insert(0, "順位", range(1, len(df)+1))
            # 表示列を必要最低限に絞る（スマホ前提）
            col_map = {
                "symbol":"銘柄",
                "sector33_name":"セクター",
                "RET_3M":"3ヶ月リターン",
                "wf_oos_wr":"WF勝率（OOS）",
                "wf_oos_rr":"WF損益比RR（OOS）",
                "mc_dd5":"MC DD 5%（推定）",
                "final_score":"総合スコア",
                "strategy_name":"推奨方式",
            }
            for k,v in list(col_map.items()):
                if k in df.columns and v not in df.columns:
                    df[v] = df[k]
            # 列が無い場合は作る（落ちない）
            for v in ["銘柄","企業名","セクター","現在値（終値）","Entry目安","SL目安","TP目安","RR","最大保有","推奨株数","推奨投資額(円)","想定損失(円)","総合スコア","推奨方式","Entry状態","発注不可理由"]:
                if v not in df.columns:
                    df[v] = None
            show_cols = ["順位","銘柄","企業名","セクター","現在値（終値）","Entry目安","SL目安","TP目安","RR","最大保有","推奨株数","推奨投資額(円)","想定損失(円)","総合スコア","推奨方式","Entry状態","発注不可理由"]
            df = df[show_cols]
            try:
                for c in ["企業名","セクター","推奨方式","Entry状態","発注不可理由"]:
                    if c in df.columns:
                        df[c] = (
                            df[c].astype(str)
                            .replace(["None","none","nan","NaN",""], "不明" if c in ["企業名","セクター"] else "")
                            .str.strip()
                        )
                for c in ["現在値（終値）","Entry目安","SL目安","TP目安","RR","推奨投資額(円)","想定損失(円)","総合スコア"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
            except Exception:
                pass


            if mobile_mode:
                render_cards_selected(df.head(show_top_n))
                with st.expander("表で見る（PC向け）", expanded=False):
                    st.dataframe(df, width="stretch")
            else:
                st.dataframe(df, width="stretch")

            st.download_button(
                "⬇️ 選定結果をCSVでダウンロード",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="selected.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning("選定結果が空です（DB更新不足・フィルタが厳しすぎる可能性）")

        st.subheader("🧭 推奨方式と価格目安（Entry/SL/TP）")
        st.caption("どう買うか？（Entry/SL/TP と最大保有日数の具体案）")
        guide = out.get("guide")
        if isinstance(guide, pd.DataFrame) and len(guide):
            try:
                # selected側の銘柄名/セクター/株数/投資額/損失/発注単位をguideへ付与
                if isinstance(df, pd.DataFrame) and len(df) and "銘柄" in guide.columns and "銘柄" in df.columns:
                    extra_cols = [c for c in ["銘柄","企業名","セクター","推奨方式","発注単位","推奨株数","推奨投資額(円)","想定損失(円)","Entry状態"] if c in df.columns]
                    if len(extra_cols) >= 2:
                        guide = guide.merge(df[extra_cols].drop_duplicates(subset=["銘柄"]), on="銘柄", how="left", suffixes=("","_sel"))
                for c in ["企業名","セクター","推奨方式","発注単位","Entry状態"]:
                    if c not in guide.columns:
                        guide[c] = ""
                    guide[c] = (
                        guide[c].astype(str)
                        .replace(["None","none","nan","NaN",""], "不明" if c in ["企業名","セクター"] else "")
                        .str.strip()
                    )
                for c in ["推奨株数","推奨投資額(円)","想定損失(円)","Entry目安","SL目安","TP目安"]:
                    if c not in guide.columns:
                        guide[c] = 0
                order_cols = [c for c in ["銘柄","企業名","セクター","推奨方式","発注単位","推奨株数","推奨投資額(円)","想定損失(円)","Entry目安","SL目安","TP目安","最大保有","Entry状態"] if c in guide.columns]
                guide = guide[order_cols].head(int(max_positions))
            except Exception:
                pass
            guide = guide.reset_index(drop=True)
            guide.insert(0, "順位", range(1, len(guide)+1))
            if mobile_mode:
                render_cards_guide(guide.head(show_top_n))
                with st.expander("表で見る（PC向け）", expanded=False):
                    st.dataframe(guide, width="stretch")
            else:
                st.dataframe(guide, width="stretch")
        else:
            st.info("価格目安の生成に必要なデータが不足しています。")

    except Exception:
        st.error("スキャン中にエラーが発生しました")
        if show_debug:
            st.code(traceback.format_exc())
        else:
            st.caption("（デバッグ表示をONにすると詳細tracebackを表示します）")

st.markdown("---")
st.caption("※注意：本ツールは投資助言ではありません。最終判断はご自身で行ってください。")
