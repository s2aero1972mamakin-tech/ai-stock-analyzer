# -*- coding: utf-8 -*-
import streamlit as st
import logic

st.set_page_config(page_title="SBI 半自動プロ仕様", layout="wide")

st.title("🚀 SBI 半自動プロ仕様 AIトレード")

st.sidebar.header("設定")

top_n = st.sidebar.slider("銘柄数", 3, 15, 5)
sector_top_n = st.sidebar.slider("上位セクター", 1, 10, 5)
capital = st.sidebar.number_input("運用資金", 100000, 20000000, 1000000)
risk_pct = st.sidebar.slider("1トレードリスク%", 0.5, 3.0, 1.0)/100
current_dd = st.sidebar.slider("現在DD%", 0.0, 20.0, 0.0)/100

if st.sidebar.button("🔥 スキャン開始"):

    with st.spinner("AI分析中..."):
        result = logic.scan_engine(top_n, sector_top_n)

    if result is None:
        st.error("データ取得失敗")
    else:
        st.subheader("🏆 セクターランキング")
        st.dataframe(result["sector_ranking"])

        st.subheader("🎯 AI選定銘柄")
        st.dataframe(result["candidates"][["Symbol","name","sector","Close","RET_3M","score"]])

        orders = logic.build_orders(
            result["candidates"],
            capital,
            risk_pct,
            current_dd
        )

        st.subheader("📝 SBI発注用注文書")
        st.dataframe(orders)

        csv = orders.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ 注文書CSVダウンロード", csv, "sbi_orders.csv")
