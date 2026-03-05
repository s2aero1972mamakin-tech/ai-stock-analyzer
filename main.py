
import streamlit as st
import pandas as pd
from logic import run_scan_3stage

st.set_page_config(page_title="JPX Scanner v11 Global Quant Engine", layout="wide")

st.title("JPX Scanner v11 Global Quant Fund Engine")

with st.sidebar:
    st.header("Scan Settings")
    min_price = st.number_input("Min Price", value=300)
    min_volume = st.number_input("Min Avg Volume", value=100000)
    keep = st.number_input("Stage0 Keep", value=300)

if st.button("Run AI Scan"):
    with st.spinner("Running Institutional Quant Engine..."):
        result = run_scan_3stage(min_price=min_price, min_avg_volume=min_volume, keep=keep)

    if result.get("ok"):
        st.success("Scan Completed")
        st.dataframe(result["selected"], use_container_width=True)

        if "portfolio_mc" in result:
            st.subheader("Portfolio MonteCarlo Risk")
            st.json(result["portfolio_mc"])

        if "diag" in result:
            st.subheader("Diagnostics")
            st.json(result["diag"])
    else:
        st.error("Scan failed")
