# -*- coding: utf-8 -*-
"""
JPX Swing Auto Scanner — ULTIMATE FAST UI
- Stooq一括DL対応
- セクター強度モデル対応
"""

import streamlit as st
import pandas as pd
import logic   # ← ULTIMATE版に差し替え済前提

st.set_page_config(page_title="JPX Ultimate Scanner", layout="wide")

st.title("🚀 日本株スイング自動スキャン（ULTIMATE）")
st.caption("Stooq一括DL + セクター強度モデル")

# =========================
# Sidebar
# =========================

st.sidebar.header("⚙️ スキャン設定")

top_n = st.sidebar.slider("抽出銘柄数", 5, 30, 10)
sector_top_n = st.sidebar.slider("上位セクター数", 1, 10, 5)

run_btn = st.sidebar.button("🔥 スキャン開始", type="primary")

# =========================
# Run
# =========================

if run_btn:
    with st.spinner("一括DL & 分析中（最大90秒）..."):
        result = logic.scan_swing_candidates(
            top_n=top_n,
            sector_top_n=sector_top_n
        )

    st.success("完了")

    st.markdown("## 🏆 セクター強度ランキング")
    st.dataframe(result["sector_ranking"], use_container_width=True)

    st.markdown("## 🎯 上位銘柄")
    st.dataframe(result["candidates"], use_container_width=True)

else:
    st.info("左のサイドバーからスキャン開始")
