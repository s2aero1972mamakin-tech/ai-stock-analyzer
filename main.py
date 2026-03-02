# -*- coding: utf-8 -*-
import streamlit as st
import logic

st.set_page_config(page_title="SBI 機関レベル運用", layout="wide")

st.title("SBI 半自動プロ運用（機関レベル）")

st.sidebar.header("設定")

top_n = st.sidebar.slider("銘柄数", 3, 10, 5)
sector_top_n = st.sidebar.slider("上位セクター", 1, 8, 5)
capital = st.sidebar.number_input("運用資金", 100000, 20000000, 1000000)
risk_pct = st.sidebar.slider("1トレードリスク%", 0.5, 3.0, 1.0)/100
current_dd = st.sidebar.slider("現在DD%", 0.0, 20.0, 0.0)/100
market_vol = st.sidebar.slider("市場ボラ倍率", 0.5, 2.0, 1.0)

if st.sidebar.button("🔥 スキャン開始"):

    with st.spinner("機関レベル分析中..."):
        result = logic.scan_engine(top_n, sector_top_n)

    if result is None:
        st.error("データ取得失敗")
    else:

        st.subheader("セクターランキング")
        st.dataframe(result["sector_ranking"])

        st.subheader("AI最終選定銘柄")
        st.dataframe(result["candidates"][["Symbol","sector","wf_score","mc_dd","score"]])

        orders = logic.build_orders(
            result["candidates"],
            capital,
            risk_pct,
            current_dd,
            market_vol
        )

        st.subheader("SBI発注用注文書")
        st.dataframe(orders)

        csv = orders.to_csv(index=False).encode("utf-8-sig")
        st.download_button("注文書CSVダウンロード", csv, "sbi_orders.csv")
