
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import logic

st.set_page_config(page_title="JPX Ultimate Scanner", layout="wide")
st.title("🚀 日本株スイング自動スキャン（ULTIMATE6）")

st.sidebar.header("⚙️ 設定")
top_n = st.sidebar.slider("抽出銘柄数", 5, 30, 10)
sector_top_n = st.sidebar.slider("上位セクター数", 1, 10, 5)
capital = st.sidebar.number_input("運用資金（円）", 100000, 10000000, 500000, step=100000)
risk_pct = st.sidebar.slider("1トレード許容リスク%", 0.5, 3.0, 1.0) / 100

if st.sidebar.button("🔥 スキャン開始", type="primary"):
    with st.spinner("分析中..."):
        result = logic.scan_swing_candidates(top_n=top_n, sector_top_n=sector_top_n)

    st.success("完了")

    st.markdown("## 🏆 セクター強度ランキング")
    st.dataframe(result["sector_ranking"], use_container_width=True)

    st.markdown("## 🎯 上位銘柄")
    st.dataframe(result["candidates"], use_container_width=True)

    # 注文書生成
    orders = []
    for _, row in result["candidates"].iterrows():
        plan = logic.build_trade_plan(
            price=row["Close"],
            atr=row["ATR"],
            capital=capital,
            risk_pct=risk_pct
        )
        plan.update({"Symbol": row["Symbol"], "name": row["name"]})
        orders.append(plan)

    orders_df = pd.DataFrame(orders)
    st.markdown("## 📝 注文書（実行用）")
    st.dataframe(orders_df, use_container_width=True)

    csv = orders_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 注文書CSVダウンロード", csv, "orders.csv", "text/csv")
