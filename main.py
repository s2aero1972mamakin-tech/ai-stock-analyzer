import os
import json
import time
import traceback
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd

import logic

JST = timezone(timedelta(hours=9))

st.set_page_config(page_title="JPX 利確銘柄抽出AI", layout="wide")

st.title("JPX Swing AI")
st.caption("固定条件：最大保有10営業日 / 利確=+1.5ATR(14) / 損切=-1.0ATR(14)  —  DB: Neon(Postgres)")

# --- Streamlit secrets -> env bridge ---
if "NEON_DATABASE_URL" in st.secrets and not os.environ.get("NEON_DATABASE_URL"):
    os.environ["NEON_DATABASE_URL"] = st.secrets["NEON_DATABASE_URL"]

st.sidebar.header("🗄️ データベース（Neon）")

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
                stage0_keep=stage0_keep,
                stage1_keep=stage1_keep,
                stage2_keep=stage2_keep,
                min_price=min_price,
                min_avg_volume=min_avg_volume,
                atr_pct_min=atr_pct_min,
                atr_pct_max=atr_pct_max,
                stage2_days=stage2_days,
                stage2_min_bars=stage2_min_bars,
            )
        elapsed = time.time() - t0
        diag = out.get("diag", {})
        diag["elapsed_sec"] = elapsed
        logic.save_last_diag(diag)

        st.success(f"スキャン完了（{elapsed:.1f}秒） / mode={diag.get('mode','?')}")
        with st.expander("📊 診断（JSON）", expanded=False):
            st.json(diag)

        st.subheader("🏁 セクター強度ランキング（Stage0）")
        sec = out.get("sector_strength")
        if isinstance(sec, pd.DataFrame) and len(sec):
            st.dataframe(sec, width="stretch")
        else:
            st.info("セクター情報が不足しているため、セクター強度は簡易表示/非表示です。")

        st.subheader("🏆 AI最終選定銘柄（利確スコア統合）")
        df = out.get("selected")
        if isinstance(df, pd.DataFrame) and len(df):
            df = df.reset_index(drop=True)
            df.insert(0, "順位", range(1, len(df)+1))
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
        guide = out.get("guide")
        if isinstance(guide, pd.DataFrame) and len(guide):
            guide = guide.reset_index(drop=True)
            guide.insert(0, "順位", range(1, len(guide)+1))
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
