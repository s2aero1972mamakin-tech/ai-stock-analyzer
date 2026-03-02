# -*- coding: utf-8 -*-
"""
JPX × SBI 半自動トレード（2ファイル完全版）
- main.py : Streamlit UI / もしくは --batch で日次OHLC更新（DB作成）
- logic.py: データ層（SQLite）＋スキャン/最適化/発注CSV生成

実行例:
  1) 日次バッチ（毎朝1回）
     python main.py --batch --db ./jpx_ohlc.sqlite --days 730 --max-workers 6

  2) 運用（Streamlit）
     streamlit run main.py
"""
from __future__ import annotations

import sys
import json
import argparse
import datetime as _dt

import pandas as pd
import logic

APP_BUILD = "2FILE-COMPLETE-2026-03-02"

def _json_dumps(obj) -> str:
    def default(o):
        try:
            if isinstance(o, (pd.Timestamp, _dt.datetime, _dt.date)):
                return str(o)
            if hasattr(o, "item"):
                return o.item()
        except Exception:
            pass
        return str(o)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=default)

def _run_batch(args: argparse.Namespace) -> int:
    rep = logic.daily_batch_update(
        db_path=args.db,
        days=int(args.days),
        max_workers=int(args.max_workers),
        chunk_size=int(args.chunk_size),
        prefer_provider=str(args.provider),
        min_rows=int(args.min_rows),
        sleep_sec=float(args.sleep_sec),
    )
    print(_json_dumps(rep))
    return 0 if rep.get("ok") else 2

def _maybe_run_batch_from_cli() -> None:
    # NOTE: streamlit run では通常 __name__ != "__main__"
    if __name__ != "__main__":
        return
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--batch", action="store_true", help="日次バッチを実行（全銘柄OHLCをDBに保存）")
    p.add_argument("--db", default=logic.DEFAULT_DB_PATH, help="SQLite DBパス")
    p.add_argument("--days", type=int, default=730, help="取得する過去日数（例: 730=約2年）")
    p.add_argument("--max-workers", type=int, default=6, help="並列数（上げすぎ注意）")
    p.add_argument("--chunk-size", type=int, default=120, help="yfinance multi-download のチャンクサイズ")
    p.add_argument("--provider", default="yfinance", choices=["yfinance","stooq","auto"], help="優先データ元")
    p.add_argument("--min-rows", type=int, default=260, help="最低必要本数（不足は失敗扱い）")
    p.add_argument("--sleep-sec", type=float, default=0.0, help="チャンクごとのスリープ（BAN回避）")
    args = p.parse_args()

    if args.batch:
        sys.exit(_run_batch(args))

# ---- CLI batch (must run BEFORE streamlit imports/execution) ----
_maybe_run_batch_from_cli()

# ---- Streamlit UI ----
import streamlit as st  # noqa: E402

st.set_page_config(page_title="SBI 半自動プロ仕様（2ファイル完全版）", layout="wide")
st.title("📈 SBI 半自動プロ仕様（2ファイル完全版）")
st.caption("スキャン→セクター強度→銘柄選択→WF/MC→RoR→DD/市場ボラ制御→注文書CSV。 build: " + APP_BUILD)

prog = st.progress(0)
status = st.empty()
detail = st.empty()

def progress_cb(stage: str, payload: dict):
    stage_map = {
        "market_vol_fetch": 0.02, "universe": 0.05, "fetch": 0.10, "merge": 0.55,
        "sector_strength": 0.60, "preselect": 0.65, "heavy_sims": 0.70, "final_rank": 0.92,
        "done": 1.00, "error": 1.00,
    }
    if stage in stage_map:
        prog.progress(int(stage_map[stage] * 100))

    if stage == "fetch_progress":
        done = payload.get("done", 0); total = payload.get("total", 1)
        ok = payload.get("ok", 0); fail = payload.get("fail", 0)
        prefer_yf = payload.get("prefer_yf", False)
        used_db = payload.get("db_hit", 0)
        frac = 0.10 + 0.40 * (done / max(1, total))
        prog.progress(int(frac * 100))
        status.write(f"📡 取得中: {done}/{total}  OK={ok} / FAIL={fail} / DB_HIT={used_db}  prefer_yfinance={prefer_yf}")
        if payload:
            detail.write(payload)
        return

    if stage == "heavy_progress":
        done = payload.get("done", 0); total = payload.get("total", 1)
        frac = 0.70 + 0.20 * (done / max(1, total))
        prog.progress(int(frac * 100))
        status.write(f"🧠 最適化/推定中: {done}/{total}  heavy_ok={payload.get('heavy_ok')} / heavy_fail={payload.get('heavy_fail')}")
        if payload:
            detail.write(payload)
        return

    status.write(f"⏳ stage: {stage}")
    if payload:
        detail.write(payload)

if "diag_obj" not in st.session_state:
    st.session_state["diag_obj"] = {}
if "diag_text" not in st.session_state:
    st.session_state["diag_text"] = ""

with st.sidebar.expander("🧭 使い方 / 設定の意味（必読）", expanded=True):
    st.markdown(f"""
### 目的
JPX全銘柄を前提に、**スキャン→セクター強度→銘柄選択→WF/MC→RoR→DD/市場ボラ制御→SBI注文書CSV** を出します。

### スキャンが止まらない設計（重要）
- 日中の Streamlit は **ローカルDB（SQLite）を優先**してOHLCを読むため、外部制限で止まりにくいです。
- それでもDBに無い銘柄は自動取得（yfinance優先）しますが、失敗しても **部分結果で必ず診断JSON** を出します。

### 推奨運用
- 毎朝1回: `python main.py --batch` でDB更新（4000銘柄でも安定）
- 日中: `streamlit run main.py` でスキャン/発注

build: `{APP_BUILD}`
""")

st.sidebar.header("⚙️ 設定")
with st.sidebar.form("scan_form", clear_on_submit=False):
    universe_limit = st.slider("スキャン対象上限", 200, 2500, 800, step=100)

    market_filter = st.selectbox("市場フィルタ（JPXマスターに列がある場合のみ有効）", ["ALL","PRIME","STANDARD","GROWTH"], index=1)
    size_filter = st.selectbox("規模フィルタ（列がある場合のみ有効）", ["ALL","LARGE","MID","SMALL"], index=0)
    universe_mode = st.selectbox("ユニバース抽出方式", ["RANDOM_STRATIFIED","HEAD"], index=0)

    min_price = st.number_input("最低株価フィルタ（円）", 0.0, 100000.0, 200.0, step=50.0)
    min_avg_volume = st.number_input("最低出来高（20日平均）", 0.0, 50000000.0, 50000.0, step=10000.0)
    cache_days = st.slider("OHLCキャッシュ/DB鮮度（日）", 0, 14, 2)

    sector_top_n = st.slider("上位セクター数", 2, 12, 6)
    pre_top_m = st.slider("重い計算対象（事前候補M）", 10, 120, 35, step=5)
    top_n = st.slider("最終採用銘柄数", 3, 10, 6)
    corr_threshold = st.slider("相関除外しきい値", 0.3, 0.95, 0.70, step=0.05)

    st.markdown("---")
    capital = st.number_input("運用資金（円）", 100_000, 50_000_000, 1_000_000, step=100_000)
    risk_pct = st.slider("1トレード許容リスク%", 0.3, 3.0, 1.0, step=0.1) / 100.0
    current_dd = st.slider("現在DD%", 0.0, 30.0, 0.0, step=0.5) / 100.0

    st.markdown("---")
    max_workers = st.slider("並列数（多すぎ注意）", 3, 16, 8)
    time_budget_sec = st.slider("タイムバジェット（秒）", 20, 180, 70, step=1)

    run = st.form_submit_button("🔥 スキャン開始", type="primary")

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_market_vol():
    return logic.compute_market_vol_ratio(progress_cb=None)

with st.sidebar.expander("📉 市場ボラ（自動: 1306.T）", expanded=False):
    if st.button("🔄 市場ボラを再取得（必要時のみ）"):
        _cached_market_vol.clear()
    vol_ratio, vol_meta = _cached_market_vol()
    st.write({"vol_ratio": vol_ratio, "meta": vol_meta})

st.sidebar.subheader("🧾 診断JSON")
if st.session_state["diag_text"]:
    st.sidebar.download_button(
        "⬇️ 診断JSONをダウンロード",
        data=st.session_state["diag_text"].encode("utf-8"),
        file_name="diag.json",
        mime="application/json",
    )
else:
    st.sidebar.info("スキャン実行後にここにDLが出ます（失敗でも出ます）。")

st.markdown("## 🧾 診断JSON（ダウンロード）")
if st.session_state["diag_text"]:
    st.download_button(
        "⬇️ 診断JSONをダウンロード（メイン）",
        data=st.session_state["diag_text"].encode("utf-8"),
        file_name="diag.json",
        mime="application/json",
    )
else:
    st.caption("未生成（スキャン後に出ます）")

if run:
    prog.progress(1)
    status.write("開始...")
    detail.empty()

    try:
        with st.spinner("スキャン＆最適化中...（進捗は上に表示）"):
            result = logic.scan_engine(
                universe_limit=int(universe_limit),
                market_filter=str(market_filter),
                size_filter=str(size_filter),
                universe_mode=str(universe_mode),
                min_price=float(min_price),
                min_avg_volume=float(min_avg_volume),
                cache_days=int(cache_days),
                sector_top_n=int(sector_top_n),
                pre_top_m=int(pre_top_m),
                top_n=int(top_n),
                corr_threshold=float(corr_threshold),
                max_workers=int(max_workers),
                time_budget_sec=int(time_budget_sec),
                progress_cb=progress_cb,
            )
    except Exception as e:
        result = {
            "ok": False,
            "error": f"main_exception: {type(e).__name__}",
            "diag": {"stage": "error", "errors": [f"{type(e).__name__}: {e}"]},
            "sector_ranking": pd.DataFrame(),
            "candidates": pd.DataFrame(),
        }

    st.session_state["diag_obj"] = result.get("diag") or {}
    st.session_state["diag_text"] = _json_dumps(st.session_state["diag_obj"])

    if not result.get("ok"):
        st.error("スキャンに失敗/部分終了しました。診断JSONをDLして原因を確認できます。")
        if result.get("error"):
            st.write({"error": result.get("error")})
        st.markdown("### 直近の失敗サンプル（最大25件）")
        st.json((st.session_state["diag_obj"] or {}).get("sample_failures", []))
        st.markdown("### 診断JSON（表示）")
        st.json(st.session_state["diag_obj"], expanded=False)
    else:
        st.success("完了（タイムバジェットで部分終了でも結果は出ます）")

        st.markdown("## 🏆 セクター強度ランキング")
        st.dataframe(result["sector_ranking"], use_container_width=True)

        cands = result["candidates"]
        st.markdown("## 🎯 AI最終選定銘柄（WF/MC後）")
        show_cols = [c for c in [
            "Symbol","name","sector","Close","RET_3M",
            "wf_oos_mean_exp","wf_oos_wr","wf_oos_rr","wf_oos_trades",
            "mc_dd_p05","final_score","wf_best"
        ] if hasattr(cands, "columns") and c in cands.columns]
        st.dataframe(cands[show_cols], use_container_width=True)

        out = logic.build_orders(
            cands,
            capital_yen=int(capital),
            risk_pct_per_trade=float(risk_pct),
            current_dd=float(current_dd),
            vol_ratio=float(vol_ratio),
        )

        st.markdown("## 🧠 ポートフォリオ統合リスク（RoR等）")
        st.json(out["portfolio"])

        st.markdown("## 📝 SBI用 注文書（CSV出力）")
        orders = out["orders"]
        st.dataframe(orders, use_container_width=True)
        csv = orders.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 注文書CSVダウンロード", data=csv, file_name="sbi_orders.csv", mime="text/csv")

        st.markdown("### 診断JSON（表示）")
        st.json(st.session_state["diag_obj"], expanded=False)
else:
    status.info("左の設定を決めて『スキャン開始』を押してください。")
    prog.progress(0)
