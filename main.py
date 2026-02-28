# main.py
# -*- coding: utf-8 -*-
"""
JPX Swing Auto Scanner（Fixed3）
- 初期表示で「候補なし」と出さない（"スキャン開始してください" に変更）
- 前回の/tmp診断JSONが残っていても、初回は「未実行」扱いを優先
- 診断JSONはスキャン開始後に生成（途中DL可）
- 中断検知（無更新3分で interrupted）
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import traceback

import pandas as pd
import streamlit as st

import logic

APP_BUILD = "STABLE5d-2026-02-28-FIXED3"
TMP_DIAG_PATH = "/tmp/ai_stock_scan_diag_latest.json"
STALL_SECONDS = 180  # 3 minutes


def _json_dumps(obj) -> str:
    def default(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, (dt.datetime, dt.date)):
            return str(o)
        if hasattr(o, "item"):
            try:
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
        if os.path.exists(TMP_DIAG_PATH):
            with open(TMP_DIAG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _ensure_state():
    defaults = {
        "pending_scan": False,
        "auto_candidates": [],
        "partial_candidates": [],
        "scan_meta": {},
        "last_scan_diag": None,
        "has_scanned_once": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _init_diag(mode: str, params: dict) -> dict:
    now = dt.datetime.now()
    return {
        "timestamp": str(now),
        "updated_at": str(now),
        "status": "running",
        "stage": "queued",
        "mode": mode,
        "progress": {"current": 0, "total": 0, "info": "queued"},
        "filter_stats": {},
        "params_effective": params,
        "auto_relax_trace": [],
        "selected_sectors": [],
        "error": None,
    }


def _maybe_mark_interrupted(diag: dict) -> dict:
    if not diag or diag.get("status") != "running":
        return diag
    try:
        updated_at = diag.get("updated_at")
        if not updated_at:
            return diag
        last = dt.datetime.fromisoformat(updated_at)
        if (dt.datetime.now() - last).total_seconds() > STALL_SECONDS:
            diag["status"] = "interrupted"
            diag["stage"] = "stalled"
            diag["error"] = diag.get("error") or "progress更新が一定時間止まったため interrupted 扱い"
            diag["updated_at"] = str(dt.datetime.now())
            _save_diag(diag)
        return diag
    except Exception:
        return diag


def _diag_sidebar(diag: dict | None):
    st.sidebar.caption(f"build: {APP_BUILD}")
    st.sidebar.subheader("🧾 診断JSON")

    if (not st.session_state.get("has_scanned_once")) and (diag is None):
        st.sidebar.info("未実行です。左の「🔥 スキャン開始」を押してください。")
        if st.sidebar.button("📦 /tmp の診断JSONを読み込む（必要時）"):
            d = _load_diag()
            if d:
                st.session_state["last_scan_diag"] = d
                st.rerun()
        if st.sidebar.button("🧹 /tmp の診断JSONを削除"):
            try:
                if os.path.exists(TMP_DIAG_PATH):
                    os.remove(TMP_DIAG_PATH)
            except Exception:
                pass
            st.sidebar.success("削除しました。")
        return

    if not diag:
        diag = _load_diag()
        if diag:
            st.session_state["last_scan_diag"] = diag

    if not diag:
        st.sidebar.info("スキャン開始後に診断JSONが生成されます（途中でもDL可）。")
        return

    diag = _maybe_mark_interrupted(diag)
    ts = str(diag.get("timestamp", "diag")).replace(":", "-").replace(" ", "_")

    cols = st.sidebar.columns([1, 1])
    if cols[0].button("🧹 クリア"):
        st.session_state["last_scan_diag"] = None
        st.session_state["auto_candidates"] = []
        st.session_state["partial_candidates"] = []
        st.session_state["scan_meta"] = {}
        st.session_state["has_scanned_once"] = False
        try:
            if os.path.exists(TMP_DIAG_PATH):
                os.remove(TMP_DIAG_PATH)
        except Exception:
            pass
        st.rerun()

    if cols[1].button("📦 再読込"):
        d = _load_diag()
        if d:
            st.session_state["last_scan_diag"] = d
        st.rerun()

    st.sidebar.download_button(
        "⬇️ 診断JSONをダウンロード",
        data=_json_dumps(diag),
        file_name=f"scan_diag_{ts}.json",
        mime="application/json",
    )
    st.sidebar.markdown(
        f"- status: **{diag.get('status','')}**\n"
        f"- stage: **{diag.get('stage','')}**\n"
        f"- mode: **{diag.get('mode','')}**\n"
        f"- updated_at: **{diag.get('updated_at','')}**"
    )
    with st.sidebar.expander("表示", expanded=False):
        st.json(diag, expanded=False)


st.set_page_config(page_title="JPX Swing Auto Scanner", layout="wide")
_ensure_state()

st.title("📈 JPX Swing Auto Scanner（Fixed3）")
st.caption("初期表示は『未実行』。スキャン開始後に結果/診断が表示されます。")

_diag_sidebar(st.session_state.get("last_scan_diag"))

st.sidebar.header("⚙️ スキャン設定")
budget = st.sidebar.number_input("想定資金（円）", min_value=50_000, max_value=10_000_000, value=300_000, step=50_000)
top_n = st.sidebar.slider("表示候補数", 1, 10, 3, step=1)
period = st.sidebar.selectbox("期間（簡易）", ["1y", "2y", "3y", "5y"], index=1)
sector_prefilter = st.sidebar.checkbox("セクター事前絞り込み（高速化）", value=True)
sector_top_n = st.sidebar.slider("上位セクター数", 2, 12, 6, step=1)
scan_btn = st.sidebar.button("🔥 スキャン開始", type="primary")

params = logic.SwingParams()

if scan_btn:
    st.session_state["pending_scan"] = True
    st.session_state["auto_candidates"] = []
    st.session_state["partial_candidates"] = []
    st.session_state["scan_meta"] = {}
    st.session_state["has_scanned_once"] = True

    diag = _init_diag(mode=params.entry_mode, params=dataclasses.asdict(params))
    st.session_state["last_scan_diag"] = diag
    _save_diag(diag)
    st.rerun()

if st.session_state.get("pending_scan"):
    st.session_state["pending_scan"] = False

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(current: int, total: int, info: str, partial=None, stats=None):
        pct = int((current / max(1, total)) * 100)
        progress_bar.progress(min(100, max(0, pct)))
        status_text.text(f"🔍 {info} ({current}/{total})")

        d = st.session_state.get("last_scan_diag") or {}
        d["updated_at"] = str(dt.datetime.now())
        d["status"] = "running"
        d["stage"] = "scanning"
        d["progress"] = {"current": int(current), "total": int(total), "info": str(info)}
        if stats is not None:
            d["filter_stats"] = stats
        st.session_state["last_scan_diag"] = d
        _save_diag(d)

        if partial is not None:
            st.session_state["partial_candidates"] = partial

    try:
        with st.status("スキャン中…", expanded=True) as status:
            res = logic.scan_swing_candidates(
                budget_yen=int(budget),
                top_n=int(top_n),
                params=params,
                progress_callback=update_progress,
                period=period,
                sector_prefilter=bool(sector_prefilter),
                sector_top_n=int(sector_top_n),
            )
            st.session_state["scan_meta"] = res
            st.session_state["auto_candidates"] = res.get("candidates", []) or []

            d = st.session_state.get("last_scan_diag") or {}
            d["updated_at"] = str(dt.datetime.now())
            d["status"] = "ok"
            d["stage"] = "done"
            d["filter_stats"] = res.get("filter_stats", {})
            d["selected_sectors"] = res.get("selected_sectors", [])
            d["auto_relax_trace"] = res.get("auto_relax_trace", [])
            d["error"] = res.get("error")
            d["progress"] = {"current": 1, "total": 1, "info": "done"}
            st.session_state["last_scan_diag"] = d
            _save_diag(d)

            if st.session_state["auto_candidates"]:
                status.update(label="✅ 完了", state="complete", expanded=False)
            else:
                status.update(label="⚠️ 完了（候補0件）", state="complete", expanded=True)

    except Exception as e:
        d = st.session_state.get("last_scan_diag") or {}
        d["updated_at"] = str(dt.datetime.now())
        d["status"] = "error"
        d["stage"] = "error"
        d["error"] = f"{type(e).__name__}: {e}"
        d["traceback"] = traceback.format_exc()
        st.session_state["last_scan_diag"] = d
        _save_diag(d)
        st.exception(e)

st.markdown("## 🎯 スキャン結果")

if not st.session_state.get("has_scanned_once"):
    st.info("未実行です。左の「🔥 スキャン開始」を押してください。")
else:
    diag = st.session_state.get("last_scan_diag") or {}
    if diag.get("status") == "running" and diag.get("stage") in ("queued", "scanning"):
        st.warning("スキャン進行中です。途中候補を表示します。")
        partial = st.session_state.get("partial_candidates") or []
        if partial:
            st.dataframe(pd.DataFrame(partial), use_container_width=True)
    else:
        cands = st.session_state.get("auto_candidates") or []
        if not cands:
            st.info("候補なし。サイドバーの診断JSONの filter_stats を確認してください。")
        else:
            st.dataframe(pd.DataFrame(cands), use_container_width=True)
