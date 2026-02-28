# main.py
# JPX Swing Auto Scanner - STABLE5d FIXED FULL VERSION

from __future__ import annotations
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import traceback
import pandas as pd
import streamlit as st
import logic

APP_BUILD = "STABLE5d-2026-02-28-FIXED"
TMP_DIAG_PATH = "/tmp/ai_stock_scan_diag_latest.json"

def _json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)

def _save_diag_tmp(diag):
    with open(TMP_DIAG_PATH, "w", encoding="utf-8") as f:
        f.write(_json_dumps(diag))

def _ensure_state_defaults():
    defaults = {
        "pending_scan": False,
        "auto_candidates": [],
        "partial_candidates": [],
        "scan_meta": {},
        "target_ticker": "",
        "pair_label": "",
        "last_scan_diag": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

st.set_page_config(page_title="JPX Swing Auto Scanner", layout="wide")
_ensure_state_defaults()

st.title("📈 JPX Swing Auto Scanner (Fixed)")

st.sidebar.header("⚙️ 設定")
budget = st.sidebar.number_input("予算(円)", 50000, 5000000, 300000, step=50000)
scan_btn = st.sidebar.button("🔥 スキャン開始")

if scan_btn:
    st.session_state["pending_scan"] = True
    st.session_state["last_scan_diag"] = {
        "timestamp": str(_dt.datetime.now()),
        "status": "running",
        "stage": "queued",
        "progress": {"current": 0, "total": 0},
    }
    _save_diag_tmp(st.session_state["last_scan_diag"])
    st.rerun()

if st.session_state.get("pending_scan"):
    st.session_state["pending_scan"] = False
    progress_bar = st.progress(0)

    try:
        res = logic.scan_swing_candidates(
            budget_yen=int(budget),
            top_n=3,
            progress_callback=lambda c, t, msg: progress_bar.progress(int((c/t)*100))
        )

        st.session_state["auto_candidates"] = res.get("candidates", [])
        st.session_state["last_scan_diag"] = {
            "timestamp": str(_dt.datetime.now()),
            "status": "done",
            "stage": "complete",
            "filter_stats": res.get("filter_stats", {})
        }
        _save_diag_tmp(st.session_state["last_scan_diag"])

    except Exception as e:
        st.session_state["last_scan_diag"] = {
            "timestamp": str(_dt.datetime.now()),
            "status": "error",
            "stage": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        _save_diag_tmp(st.session_state["last_scan_diag"])
        st.error(str(e))

st.markdown("## 🎯 スキャン結果")

cands = st.session_state.get("auto_candidates", [])
if not cands:
    st.info("候補なし")
else:
    st.dataframe(pd.DataFrame(cands), use_container_width=True)
