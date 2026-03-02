# -*- coding: utf-8 -*-
# main.py — Sector-First Trader (Robust)

from __future__ import annotations

import json
import hashlib
import datetime as dt
import pandas as pd
import streamlit as st
import logic

st.set_page_config(layout="wide")
st.title("JPX Sector-First Swing Trader（セクター→銘柄 / 取得堅牢化）")

col1, col2, col3, col4 = st.columns(4)
capital = col1.number_input("資金（円）", value=300000, step=50000, min_value=50000)
top_sectors = col2.number_input("上位セクター数", value=3, step=1, min_value=1, max_value=10)
max_per_sector = col3.number_input("セクターあたり最大銘柄数", value=180, step=20, min_value=40, max_value=600)
lot = col4.number_input("単元株（通常100）", value=100, step=100, min_value=1, max_value=1000)

@st.cache_data(ttl=24*60*60)
def load_jpx_universe() -> pd.DataFrame:
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    df = pd.read_excel(url)
    df = df[df["33業種区分"] != "-"].copy()
    df["ticker"] = df["コード"].astype(str).str.zfill(4) + ".T"
    df = df.rename(columns={"33業種区分": "sector"})
    return df[["ticker","sector"]].dropna()

universe = load_jpx_universe()
st.write(f"ユニバース: {len(universe):,} 銘柄 / セクター: {universe['sector'].nunique()}")

if st.button("🚀 セクター→銘柄スキャン開始", type="primary"):
    with st.spinner("セクター推定 → 上位セクター深掘り…"):
        out = logic.scan_sector_first(
            universe,
            capital_yen=float(capital),
            top_sectors=int(top_sectors),
            max_per_sector=int(max_per_sector),
            lot=int(lot),
        )

    st.success(f"完了: {out['meta']['elapsed_sec']:.1f}s")
    st.subheader("診断（取得状況）")
    st.json(out["meta"]["diagnostics"])

    if out["meta"]["diagnostics"].get("global_fetch_failed"):
        st.error("価格データ取得がほぼ全滅しています。診断の http_status_counts を確認してください（403/429が多いなら取得元がブロックしている可能性大）。")

    payload = json.dumps(out, ensure_ascii=False, indent=2)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    fname = f"scan_result_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest}.json"
    st.download_button("⬇️ 診断JSONをダウンロード", data=payload, file_name=fname, mime="application/json")

    st.subheader("セクター強弱スコア")
    sec_df = pd.DataFrame([{"sector": k, "score": v} for k, v in out["sector_scores"].items()]).sort_values("score", ascending=False)
    st.dataframe(sec_df, use_container_width=True, height=320)

    st.subheader("ランキング（上位50）")
    st.dataframe(pd.DataFrame(out["ranked"][:50]), use_container_width=True, height=440)
