
# -*- coding: utf-8 -*-
import json
import datetime as _dt
import streamlit as st
import pandas as pd
import logic

APP_BUILD = "ULTIMATE7-2026-03-02"
TMP_DIAG_PATH = "/tmp/ai_stock_diag_latest.json"


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


def _save_diag(diag: dict) -> None:
    try:
        with open(TMP_DIAG_PATH, "w", encoding="utf-8") as f:
            f.write(_json_dumps(diag))
    except Exception:
        pass


def _load_diag() -> dict | None:
    try:
        import os
        if os.path.exists(TMP_DIAG_PATH):
            with open(TMP_DIAG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


st.set_page_config(page_title="SBI 半自動プロ仕様（機関レベル）", layout="wide")
st.title("📈 SBI 半自動プロ仕様（機関レベル）")
st.caption("スキャン→セクター強度→銘柄選択→WF/MC→RoR→DD/市場ボラ制御→注文書CSV。 build: " + APP_BUILD)

# Sidebar
st.sidebar.header("⚙️ 設定")
universe_limit = st.sidebar.slider("スキャン対象上限（Cloud安定）", 300, 2500, 1200, step=100)
sector_top_n = st.sidebar.slider("上位セクター数", 2, 12, 6)
pre_top_m = st.sidebar.slider("重い計算対象（事前候補M）", 20, 120, 60, step=5)
top_n = st.sidebar.slider("最終採用銘柄数", 3, 10, 6)
corr_threshold = st.sidebar.slider("相関除外しきい値", 0.3, 0.95, 0.70, step=0.05)

st.sidebar.markdown("---")
capital = st.sidebar.number_input("運用資金（円）", 100_000, 50_000_000, 1_000_000, step=100_000)
risk_pct = st.sidebar.slider("1トレード許容リスク%", 0.3, 3.0, 1.0, step=0.1) / 100.0
current_dd = st.sidebar.slider("現在DD%", 0.0, 30.0, 0.0, step=0.5) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🧾 診断JSON")
diag = _load_diag()
if diag:
    ts = str(diag.get("timestamp", "diag")).replace(":", "-").replace(" ", "_")
    st.sidebar.download_button("⬇️ 診断JSONをダウンロード", data=_json_dumps(diag), file_name=f"diag_{ts}.json", mime="application/json")
    with st.sidebar.expander("表示", expanded=False):
        st.sidebar.json(diag)
else:
    st.sidebar.info("スキャン実行後に診断JSONが生成されます。")

run = st.sidebar.button("🔥 スキャン開始", type="primary")

# Market vol proxy
with st.sidebar.expander("📉 市場ボラ（自動）", expanded=False):
    vol_ratio, vol_meta = logic.compute_market_vol_ratio()
    st.write({"vol_ratio": vol_ratio, "meta": vol_meta})

# Run
if run:
    with st.spinner("スキャン＆最適化中...（Cloudでも止まらない設計）"):
        result = logic.scan_engine(
            universe_limit=int(universe_limit),
            sector_top_n=int(sector_top_n),
            pre_top_m=int(pre_top_m),
            top_n=int(top_n),
            corr_threshold=float(corr_threshold),
        )

    diag = result.get("diag") or {}
    _save_diag(diag)

    if not result.get("ok"):
        st.error("スキャンに失敗しました。サイドバーの診断JSONをDLして確認してください。")
        if result.get("error"):
            st.write({"error": result.get("error")})
        st.markdown("### 直近の失敗サンプル（最大25件）")
        st.json((diag or {}).get("sample_failures", []))
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

        st.markdown("## 🧾 診断JSON（この実行）")
        st.json(diag, expanded=False)
