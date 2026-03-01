
# main.py
from __future__ import annotations
import dataclasses
import datetime as dt
import json
import os
import traceback
import streamlit as st
import pandas as pd
import logic

APP_BUILD = "STABLE5d-FIXED4"
TMP_DIAG_PATH = "/tmp/ai_stock_scan_diag_latest.json"

def save_diag(d):
    with open(TMP_DIAG_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_diag():
    if os.path.exists(TMP_DIAG_PATH):
        with open(TMP_DIAG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

if "scan_state" not in st.session_state:
    st.session_state.scan_state = "idle"
if "diag" not in st.session_state:
    st.session_state.diag = None
if "resume_index" not in st.session_state:
    st.session_state.resume_index = 0

st.set_page_config(layout="wide")
st.title("JPX Swing Auto Scanner FIXED4")

budget = st.sidebar.number_input("予算", 50000, 10000000, 300000, step=50000)
top_n = st.sidebar.slider("表示銘柄数", 1, 10, 5)

col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("スキャン開始")
resume_btn = col2.button("再開")

if start_btn:
    st.session_state.scan_state = "running"
    st.session_state.resume_index = 0
    diag = {
        "timestamp": str(dt.datetime.now()),
        "status": "running",
        "stage": "scanning",
        "cursor_index": 0,
        "fail_data_reason": {},
        "timing": {"fetch_sec": 0, "total_sec": 0},
        "filter_stats": {},
        "error_log": []
    }
    st.session_state.diag = diag
    save_diag(diag)

if resume_btn:
    saved = load_diag()
    if saved and saved.get("status") == "running":
        st.session_state.diag = saved
        st.session_state.resume_index = saved.get("cursor_index", 0)
        st.session_state.scan_state = "running"

if st.session_state.scan_state == "running":
    progress = st.progress(0)

    try:
        result = logic.scan_swing_candidates(
            budget_yen=int(budget),
            top_n=int(top_n),
            start_index=st.session_state.resume_index,
            progress_callback=lambda i, total: progress.progress(int((i/total)*100)),
            diag=st.session_state.diag
        )

        st.session_state.diag.update({
            "status": "done",
            "stage": "complete",
            "filter_stats": result.get("filter_stats")
        })
        save_diag(st.session_state.diag)
        st.session_state.scan_state = "idle"

        st.success("スキャン完了")
        st.dataframe(pd.DataFrame(result.get("candidates", [])))

    except Exception as e:
        d = st.session_state.diag
        d["status"] = "error"
        d["error"] = str(e)
        d["traceback"] = traceback.format_exc()
        save_diag(d)
        st.error(str(e))

st.sidebar.markdown("### 診断JSON")
if st.session_state.diag:
    st.sidebar.json(st.session_state.diag, expanded=False)
    st.sidebar.download_button(
        "診断JSONダウンロード",
        json.dumps(st.session_state.diag, ensure_ascii=False, indent=2),
        file_name="scan_diag_fixed4.json",
        mime="application/json"
    )
