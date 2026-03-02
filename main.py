# -*- coding: utf-8 -*-
import json
import datetime as _dt
import streamlit as st
import pandas as pd
import logic

APP_BUILD = "ULTIMATE8-2026-03-02"

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

st.set_page_config(page_title="SBI 半自動プロ仕様（機関レベル）", layout="wide")
st.title("📈 SBI 半自動プロ仕様（機関レベル）")
st.caption("スキャン→セクター強度→銘柄選択→WF/MC→RoR→DD/市場ボラ制御→注文書CSV。 build: " + APP_BUILD)

with st.sidebar.expander("🧭 使い方 / 設定の意味（必読）", expanded=True):
    st.markdown("""
**目的**：AIが「買う候補」を絞り、SBIでの発注はあなたが最終確認して実行（半自動）。

- **スキャン対象上限**：Cloudでは800前後が安定しやすい。増やすほど遅く/ブロックされやすい。  
- **上位セクター数**：強いセクターだけに絞る（4〜7推奨）  
- **事前候補M**：WF/MCをかける銘柄数（ここが一番重い、30〜60推奨）  
- **最終採用銘柄数**：ポートフォリオ銘柄数  
- **相関除外**：0.7なら相関の高い銘柄は同時採用しない（0.6〜0.75推奨）  
- **運用資金/リスク%**：1トレードで失ってよい上限（0.5〜1.0%推奨）  
- **現在DD%**：DDが大きいほどロット縮小  
- **タイムバジェット**：一定秒数で“部分結果を返して”止まらない（Cloud対策）
""")

st.sidebar.header("⚙️ 設定")
universe_limit = st.sidebar.slider("スキャン対象上限", 200, 2500, 800, step=100)
sector_top_n = st.sidebar.slider("上位セクター数", 2, 12, 6)
pre_top_m = st.sidebar.slider("重い計算対象（事前候補M）", 15, 120, 45, step=5)
top_n = st.sidebar.slider("最終採用銘柄数", 3, 10, 6)
corr_threshold = st.sidebar.slider("相関除外しきい値", 0.3, 0.95, 0.70, step=0.05)

st.sidebar.markdown("---")
capital = st.sidebar.number_input("運用資金（円）", 100_000, 50_000_000, 1_000_000, step=100_000)
risk_pct = st.sidebar.slider("1トレード許容リスク%", 0.3, 3.0, 1.0, step=0.1) / 100.0
current_dd = st.sidebar.slider("現在DD%", 0.0, 30.0, 0.0, step=0.5) / 100.0

st.sidebar.markdown("---")
max_workers = st.sidebar.slider("並列数（多すぎ注意）", 4, 16, 10)
time_budget_sec = st.sidebar.slider("タイムバジェット（秒）", 15, 90, 52, step=1)

if "diag" not in st.session_state:
    st.session_state["diag"] = None

st.sidebar.subheader("🧾 診断JSON")
if st.session_state["diag"]:
    ts = str(st.session_state["diag"].get("timestamp_utc", "diag")).replace(":", "-").replace(" ", "_")
    st.sidebar.download_button("⬇️ 診断JSONをダウンロード", data=_json_dumps(st.session_state["diag"]), file_name=f"diag_{ts}.json", mime="application/json")
    with st.sidebar.expander("表示", expanded=False):
        st.sidebar.json(st.session_state["diag"])
else:
    st.sidebar.info("スキャン実行後にここに診断JSONが出ます。")

prog = st.progress(0)
status = st.empty()
detail = st.empty()

def progress_cb(stage: str, payload: dict):
    stage_map = {
        "universe": 0.05,
        "fetch": 0.10,
        "merge": 0.55,
        "sector_strength": 0.60,
        "preselect": 0.65,
        "heavy_sims": 0.70,
        "final_rank": 0.92,
        "done": 1.00,
        "error": 1.00,
        "market_vol_fetch": 0.02,
    }
    if stage in stage_map:
        prog.progress(int(stage_map[stage] * 100))

    if stage == "fetch_progress":
        done = payload.get("done", 0); total = payload.get("total", 1)
        ok = payload.get("ok", 0); fail = payload.get("fail", 0)
        frac = 0.10 + 0.40 * (done / max(1, total))
        prog.progress(int(frac * 100))
        status.write(f"📡 取得中: {done}/{total}  OK={ok} / FAIL={fail}")
        detail.write(payload)
        return

    if stage == "heavy_progress":
        done = payload.get("done", 0); total = payload.get("total", 1)
        frac = 0.70 + 0.20 * (done / max(1, total))
        prog.progress(int(frac * 100))
        status.write(f"🧠 最適化/推定中: {done}/{total}  heavy_ok={payload.get('heavy_ok')} / heavy_fail={payload.get('heavy_fail')}")
        detail.write(payload)
        return

    status.write(f"⏳ stage: {stage}")
    if payload:
        detail.write(payload)

with st.sidebar.expander("📉 市場ボラ（自動: 1306.T）", expanded=False):
    vol_ratio, vol_meta = logic.compute_market_vol_ratio(progress_cb=progress_cb)
    st.write({"vol_ratio": vol_ratio, "meta": vol_meta})

run = st.sidebar.button("🔥 スキャン開始", type="primary")

if run:
    prog.progress(1)
    status.write("開始...")
    detail.empty()

    with st.spinner("スキャン＆最適化中...（進捗は上に表示）"):
        result = logic.scan_engine(
            universe_limit=int(universe_limit),
            sector_top_n=int(sector_top_n),
            pre_top_m=int(pre_top_m),
            top_n=int(top_n),
            corr_threshold=float(corr_threshold),
            max_workers=int(max_workers),
            time_budget_sec=int(time_budget_sec),
            progress_cb=progress_cb,
        )

    st.session_state["diag"] = result.get("diag") or {}

    if not result.get("ok"):
        st.error("スキャンに失敗/部分終了しました。サイドバーから診断JSONをDLできます。")
        if result.get("error"):
            st.write({"error": result.get("error")})
        st.markdown("### 直近の失敗サンプル（最大25件）")
        st.json((st.session_state["diag"] or {}).get("sample_failures", []))
    else:
        st.success("完了")

        st.markdown("## 🏆 セクター強度ランキング")
        st.dataframe(result["sector_ranking"], use_container_width=True)

        cands = result["candidates"]
        st.markdown("## 🎯 AI最終選定銘柄（WF/MC後）")
        show_cols = [c for c in [
            "Symbol","name","sector","Close","RET_3M",
            "wf_oos_mean_exp","wf_oos_wr","wf_oos_rr","wf_oos_trades",
            "mc_dd_p05","final_score","wf_best"
        ] if c in cands.columns]
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

        st.markdown("## 🧾 診断JSON（今回）")
        st.json(st.session_state["diag"], expanded=False)
else:
    status.info("左の設定を決めて『スキャン開始』を押してください。")
