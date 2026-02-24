import streamlit as st
import logic

st.set_page_config(layout="wide")
st.title("AI Stock Analyzer (JP Stooq Stable)")

budget = st.sidebar.number_input("Budget (JPY)", value=300000)
period = st.sidebar.selectbox("Period", ["6mo","1y","2y"], index=0)

if st.button("Scan"):
    res = logic.scan_swing_candidates(
        budget_yen=int(budget),
        period=period
    )
    if not res:
        st.warning("No candidates found.")
    else:
        st.dataframe(res, width="stretch")
