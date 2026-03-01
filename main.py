# main.py
# -*- coding: utf-8 -*-
"""
JPX Swing Auto Scanner (Stooq) — STABLE5c-2026-02-28 (FULL)
- 診断JSONを「スキャン開始時点」で生成し、途中でも必ずDL可能
- Streamlitの描画順/再実行の罠を回避：queued → calling_scan → done/error の二段階実行
- JPX銘柄ユニバースは、JPX公式の「東証上場銘柄一覧（33業種）」Excelを取得して生成（CSV不要）
"""

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

APP_BUILD = "STABLE5c-2026-02-28"
TMP_DIAG_PATH = "/tmp/ai_stock_scan_diag_latest.json"


def _json_dumps(obj) -> str:
    def default(o):
        try:
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
        except Exception:
            pass
        if isinstance(o, (pd.Timestamp, _dt.datetime, _dt.date)):
            return str(o)
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        return str(o)

    return json.dumps(obj, ensure_ascii=False, indent=2, default=default)


def _save_diag_tmp(diag: dict) -> None:
    try:
        with open(TMP_DIAG_PATH, "w", encoding="utf-8") as f:
            f.write(_json_dumps(diag))
    except Exception:
        pass


def _load_diag_tmp() -> dict | None:
    try:
        if os.path.exists(TMP_DIAG_PATH):
            with open(TMP_DIAG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _ensure_state_defaults():
    defaults = {
        "pending_scan": False,
        "auto_candidates": [],
        "partial_candidates": [],
        "scan_meta": {},
        "target_ticker": "",
        "pair_label": "",
        "last_scan_diag": None,
        "_dropped_param_keys": None,
        "resume_index": 0,
        "resume_requested": False,
        "has_scanned_once": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def build_swing_params_safe(**kwargs) -> logic.SwingParams:
    allowed = {f.name for f in dataclasses.fields(logic.SwingParams)}
    safe = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted([k for k in kwargs.keys() if k not in safe])
    st.session_state["_dropped_param_keys"] = dropped if dropped else None
    return logic.SwingParams(**safe)


def _render_scan_diag_sidebar(slot, *, expanded: bool, title: str):
    tag = hashlib.md5(title.encode("utf-8")).hexdigest()[:10]

    diag = st.session_state.get("last_scan_diag")
    if not diag:
        diag = _load_diag_tmp()
        if diag:
            st.session_state["last_scan_diag"] = diag

    with slot.container():
        st.sidebar.caption(f"build: {APP_BUILD}")
        st.sidebar.subheader(title)

        cols = st.sidebar.columns([1, 1, 2])
        if cols[0].button("🧹 診断をクリア", key=f"btn_clear_diag_{tag}"):
            st.session_state["last_scan_diag"] = None
            _save_diag_tmp({})
            st.rerun()

        if cols[1].button("📦 /tmp から再読込", key=f"btn_reload_diag_{tag}"):
            d = _load_diag_tmp()
            if d:
                st.session_state["last_scan_diag"] = d
            st.rerun()

        if not diag:
            st.sidebar.info("診断JSONはスキャン開始時点で生成されます（途中でもDL可）。")
            return

        ts = str(diag.get("timestamp", "diag")).replace(":", "-").replace(" ", "_")
        st.sidebar.download_button(
            "⬇️ 診断JSONをダウンロード（途中でも可）",
            data=_json_dumps(diag),
            file_name=f"scan_diag_{ts}.json",
            mime="application/json",
            key=f"dl_diag_{tag}",
        )

        st.sidebar.markdown(
            f"- status: **{diag.get('status','')}**\n"
            f"- stage: **{diag.get('stage','')}**\n"
            f"- mode: **{diag.get('mode','')}**\n"
            f"- updated_at: **{diag.get('updated_at','')}**"
        )
        dropped = st.session_state.get("_dropped_param_keys")
        if dropped:
            st.sidebar.warning("main/logicの不一致で無視したパラメータ: " + ", ".join(dropped))

        with st.sidebar.expander("🧾 診断JSON（表示）", expanded=expanded):
            st.json(diag, expanded=False)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="JPX Swing Auto Scanner", layout="wide")
_ensure_state_defaults()

st.title("📈 日本株スイング自動スキャン（Stooq）")
st.caption("JPXユニバースはJPX公式Excelから生成（CSV不要）。診断JSONは常時DL可。")

diag_slot = st.sidebar.empty()
_render_scan_diag_sidebar(diag_slot, expanded=False, title="🧾 診断JSON（常時表示）")

st.sidebar.header("⚙️ スキャン設定")
budget = st.sidebar.number_input("想定資金（円）", min_value=50_000, max_value=5_000_000, value=300_000, step=50_000)
capital = st.sidebar.number_input("運用資金（円）", min_value=50_000, max_value=20_000_000, value=300_000, step=50_000)
risk_pct = st.sidebar.slider("許容損失（1トレード）%", 0.1, 3.0, 1.0, step=0.1) / 100.0

entry_mode_label = st.sidebar.selectbox("モード", ["押し目（pullback）", "ブレイクアウト（breakout）"], index=0)
entry_mode = "pullback" if entry_mode_label.startswith("押し目") else "breakout"

require_trend = st.sidebar.checkbox("SMA25 > SMA75 を必須", value=True)
rsi_low = st.sidebar.slider("RSI下限", 10.0, 60.0, 40.0, step=1.0)
rsi_high = st.sidebar.slider("RSI上限", 40.0, 90.0, 70.0, step=1.0)

pb_low = st.sidebar.slider("押し目下限（SMA25乖離%）", -25.0, 0.0, -8.0, step=0.5)
pb_high = st.sidebar.slider("押し目上限（SMA25乖離%）", -15.0, 5.0, -3.0, step=0.5)

atr_min = st.sidebar.slider("ATR%下限", 0.5, 12.0, 1.5, step=0.5)
atr_max = st.sidebar.slider("ATR%上限", 1.0, 25.0, 10.0, step=0.5)

vol_min = st.sidebar.number_input("平均出来高(20日) 下限", min_value=0, max_value=10_000_000, value=50_000, step=10_000)
turnover_min_yen = st.sidebar.number_input("平均売買代金(20日) 下限（円）", min_value=0, max_value=50_000_000_000, value=0, step=10_000_000)

breakout_lookback = st.sidebar.slider("高値更新参照日数", 5, 60, 20, step=1)
breakout_vol_ratio = st.sidebar.slider("出来高倍率（当日/20日平均）", 1.0, 5.0, 1.6, step=0.1)

atr_mult = st.sidebar.slider("損切: ATR倍率", 0.8, 3.5, 2.0, step=0.1)
tp1_r = st.sidebar.slider("利確1: +何Rで半分利確", 0.5, 2.0, 1.0, step=0.1)
tp2_r = st.sidebar.slider("利確2: +何Rを狙う", 1.5, 6.0, 3.0, step=0.5)
time_stop_days = st.sidebar.slider("時間切れ（日）", 3, 20, 10, step=1)

bt_period = st.sidebar.selectbox("バックテスト期間", ["1y", "2y", "3y", "5y"], index=1)
bt_topk = st.sidebar.slider("バックテスト対象（上位K）", 5, 60, 20, step=5)

sector_prefilter = st.sidebar.checkbox("セクター事前絞り込み（推奨）", value=True)
sector_top_n = st.sidebar.slider("上位セクター数", 2, 12, 6, step=1)

st.sidebar.markdown("---")
scan_btn = st.sidebar.button("🔥 スキャン開始", type="primary")

resume_btn = st.sidebar.button("▶️ 中断から再開（診断JSONの cursor_index から）")
if resume_btn:
    d = st.session_state.get("last_scan_diag") or _load_diag_tmp() or {}
    cur = int(d.get("cursor_index", -1))
    if cur >= 0:
        st.session_state["resume_index"] = cur + 1
        st.session_state["resume_requested"] = True
        st.session_state["pending_scan"] = True
        # running/ interrupted のままでもOK（進捗更新されます）
        if not st.session_state.get("last_scan_diag"):
            st.session_state["last_scan_diag"] = d
        st.rerun()
    else:
        st.sidebar.warning("再開に必要な cursor_index がありません（まず通常スキャンを実行してください）")


params = build_swing_params_safe(
    require_sma25_over_sma75=bool(require_trend),
    entry_mode=str(entry_mode),
    rsi_low=float(rsi_low),
    rsi_high=float(rsi_high),
    pullback_low=float(pb_low),
    pullback_high=float(pb_high),
    atr_pct_min=float(atr_min),
    atr_pct_max=float(atr_max),
    vol_avg20_min=float(vol_min),
    turnover_avg20_min_yen=float(turnover_min_yen),
    breakout_lookback=int(breakout_lookback),
    breakout_vol_ratio=float(breakout_vol_ratio),
    atr_mult_stop=float(atr_mult),
    tp1_r=float(tp1_r),
    tp2_r=float(tp2_r),
    time_stop_days=int(time_stop_days),
    risk_pct=float(risk_pct),
)

# -------- Two-phase execution --------
if scan_btn:
    st.session_state["pending_scan"] = True
    st.session_state["resume_index"] = 0
    st.session_state["resume_requested"] = False
    st.session_state["has_scanned_once"] = True
    st.session_state["last_scan_diag"] = {
        "timestamp": str(_dt.datetime.now()),
        "updated_at": str(_dt.datetime.now()),
        "status": "running",
        "stage": "queued",
        "mode": entry_mode,
        "relax_level": 0,
        "selected_sectors": [],
        "filter_stats": {},
        "params_effective": dataclasses.asdict(params),
        "auto_relax_trace": [],
        "cursor_index": -1,
        "last_ticker": "",
        "fail_data_reason": {},
        "timing": {"fetch_sec": 0.0, "indicators_sec": 0.0, "total_sec": 0.0},
        "progress": {"current": 0, "total": 0, "info": "queued"},
    }
    _save_diag_tmp(st.session_state["last_scan_diag"])
    _render_scan_diag_sidebar(diag_slot, expanded=False, title="🧾 診断JSON（実行開始）")
    st.rerun()

if st.session_state.get("pending_scan"):
    st.session_state["pending_scan"] = False

    diag = st.session_state.get("last_scan_diag") or {}
    diag.update(
        {
            "updated_at": str(_dt.datetime.now()),
            "status": "running",
            "stage": "calling_scan",
            "progress": {"current": 0, "total": 0, "info": "calling_scan"},
        }
    )
    st.session_state["last_scan_diag"] = diag
    _save_diag_tmp(diag)
    _render_scan_diag_sidebar(diag_slot, expanded=False, title="🧾 診断JSON（呼び出し直前）")

    
    progress_bar = st.empty()
status_text = st.empty()

def update_progress(current: int, total: int, info: str, partial=None, stats=None):
        pct = int((current / max(1, total)) * 100)
        progress_bar.progress(min(100, max(0, pct)))
        status_text.text(f"🔍 {info} ({current}/{total})")
        d = st.session_state.get("last_scan_diag") or {}
        d["updated_at"] = str(_dt.datetime.now())
        d["status"] = "running"
        d["stage"] = "scanning"
        d["progress"] = {"current": int(current), "total": int(total), "info": str(info)}
        st.session_state["last_scan_diag"] = d
        _save_diag_tmp(d)
        if stats is not None:
            d["filter_stats"] = stats
            st.session_state["last_scan_diag"] = d
            _save_diag_tmp(d)
        if partial is not None:
            st.session_state["partial_candidates"] = partial

    try:
        with st.status("スキャン＆バックテスト中…", expanded=True) as status:
            res = logic.scan_swing_candidates(
                budget_yen=int(budget),
                start_index=int(st.session_state.get('resume_index', 0)),
                diag=st.session_state.get('last_scan_diag'),
                top_n=3,
                params=params,
                progress_callback=update_progress,
                backtest_period=bt_period,
                backtest_topk=int(bt_topk),
                sector_prefilter=bool(sector_prefilter),
                sector_top_n=int(sector_top_n),
            )

            candidates = res.get("candidates", []) or []
            st.session_state["auto_candidates"] = candidates
            st.session_state["scan_meta"] = res

            diag = st.session_state.get("last_scan_diag") or {}
            diag = {
                **diag,
                "updated_at": str(_dt.datetime.now()),
                "status": "ok",
                "stage": "done",
                "mode": res.get("mode", entry_mode),
                "relax_level": res.get("relax_level", 0),
                "selected_sectors": res.get("selected_sectors", []),
                "filter_stats": res.get("filter_stats", {}),
                "params_effective": res.get("params_effective", dataclasses.asdict(params)),
                "auto_relax_trace": res.get("auto_relax_trace", []),
                "error": res.get("error"),
                "progress": {"current": 1, "total": 1, "info": "done"},
            }
            st.session_state["last_scan_diag"] = diag
            _save_diag_tmp(diag)
            _render_scan_diag_sidebar(diag_slot, expanded=False, title="🧾 診断JSON（完了）")

            if candidates:
                status.update(label="✅ 完了：候補を抽出しました", state="complete", expanded=False)
            else:
                status.update(label="⚠️ 完了：候補が0件でした（診断JSONを確認）", state="complete", expanded=True)

    except Exception as e:
        diag = st.session_state.get("last_scan_diag") or {}
        diag.update(
            {
                "updated_at": str(_dt.datetime.now()),
                "status": "error",
                "stage": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        )
        st.session_state["last_scan_diag"] = diag
        _save_diag_tmp(diag)
        _render_scan_diag_sidebar(diag_slot, expanded=True, title="🧾 診断JSON（例外）")
        st.exception(e)

# -----------------------------
# Results
# -----------------------------
st.markdown("## 🎯 スキャン結果")
st.markdown("## 🎯 スキャン結果")

diag = st.session_state.get("last_scan_diag") or {}
stage = diag.get("stage")
status = diag.get("status")

if status == "running" and stage in ("queued", "calling_scan", "scanning"):
    st.warning("スキャン進行中です。完了すると候補がここに出ます。")
    partial = st.session_state.get("partial_candidates") or []
    if partial:
        st.markdown("### ⏳ 暫定候補（スキャン途中の上位）")
        st.dataframe(pd.DataFrame(partial), use_container_width=True)
else:
    cands = st.session_state.get("auto_candidates") or []
    if not cands:
        st.info("完了しましたが候補は0件でした。サイドバーの診断JSONで fail_* を確認してください。")
    else:
        df = pd.DataFrame(cands)
        st.dataframe(df, use_container_width=True)

        st.markdown("### 候補の選択")
        for c in cands:
            label = f"{c.get('ticker','')} {c.get('name','')}".strip()
            if st.button(f"分析: {label}", key=f"pick_{c.get('ticker','')}"):
                st.session_state["target_ticker"] = c.get("ticker", "")
                st.session_state["pair_label"] = label
                st.rerun()
ticker = st.session_state.get("target_ticker", "")
if ticker:
    st.markdown(f"## 🔎 個別分析: {st.session_state.get('pair_label','')}")
    with st.spinner("データ取得＆指標計算中…"):
        df_raw = logic.get_market_data(ticker, period=max(bt_period, "2y"), interval="1d")
        df_ind = logic.calculate_indicators(df_raw)
    if df_ind.empty:
        st.error("データ取得に失敗しました。")
    else:
        latest = df_ind.iloc[-1]
        st.write(
            {
                "Close": float(latest["Close"]),
                "RSI": float(latest["RSI"]),
                "ATR": float(latest["ATR"]),
                "ATR%": float(latest["ATR_PCT"]),
            }
        )
        bt = logic.backtest_swing(df_ind, params)
        st.write(
            {
                "trades": bt.n_trades,
                "win_rate": bt.win_rate,
                "PF": bt.profit_factor,
                "AvgR": bt.expectancy_r,
                "MaxDD": bt.max_drawdown,
            }
        )
        plan = logic.build_trade_plan(df_ind, params, capital_yen=int(capital), risk_pct=float(risk_pct))
        st.markdown("### 📝 注文書（目安）")
        st.json(plan)
