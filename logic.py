


# logic_fixed28_trend_entry_engine_compat.py
# Drop-in replacement for logic.py
# - Adds self-contained: HH/HL detection, breakout strength, continuation probability, phase-aware EV thresholding,
#   momentum bonus (capped), and breakout gate.
# - Designed to be backward-compatible with existing main.py callers.
#
# NOTE: This module does not depend on ctx keys being passed from main.py; it computes needed signals from price history.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Economic calendar / event risk (optional, free feed)
# ---------------------------------------------------------------------
# This tool is swing-oriented but macro events can cause gaps/slippage.
# We therefore compute an "event_risk_score" from upcoming releases and
# optionally block trades within a high-impact window.
#
# Default feed: Forex Factory weekly export (JSON)
# - https://nfs.faireconomy.media/ff_calendar_thisweek.json
#
# Notes:
# - If the feed is unavailable, we fail safe (event_risk_score=0) and
#   expose status in ctx_out so the UI can warn the operator.
#
import json
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import ssl
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

_FF_CAL_URL_DEFAULT = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_EVENT_CACHE = {
    "ts": 0.0,     # epoch seconds
    "url": None,
    "data": None,  # list
    "err": None,   # str
}



# Persistent file cache (survives Streamlit reruns / transient rate limits)
_EVENT_FILE_CACHE_PATH = Path(tempfile.gettempdir()) / "fx_analyzer_ff_calendar_cache.json"

def _read_event_file_cache(max_age_sec: int = 24 * 3600) -> Optional[List[dict]]:
    try:
        p = _EVENT_FILE_CACHE_PATH
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(obj, dict):
            return None
        ts = float(obj.get("ts", 0.0) or 0.0)
        data = obj.get("data", None)
        if not isinstance(data, list):
            return None
        if ts > 0 and (time.time() - ts) > float(max_age_sec):
            return None
        return data
    except Exception:
        return None

def _write_event_file_cache(url: str, data: List[dict]) -> None:
    try:
        p = _EVENT_FILE_CACHE_PATH
        payload = {"ts": time.time(), "url": url, "data": data}
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _derive_xml_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    if ".json" in u:
        return u.replace(".json", ".xml")
    if u.endswith("/"):
        u = u[:-1]
    # fallback
    return "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

def _fetch_ff_calendar_xml(url: str, timeout: int = 12) -> Optional[List[dict]]:
    """Parse ForexFactory weekly XML export into list[dict] compatible with _compute_event_risk."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (fx-analyzer; event-guard)"})
        ctx = None
        try:
            ctx = ssl.create_default_context()
        except Exception:
            ctx = None
        try:
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                raw = resp.read()
        except TypeError:
            # python without context param
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()

        # XML is often windows-1252
        txt = raw.decode("utf-8", errors="replace")
        # Some servers send the XML declaration with windows-1252; ElementTree can parse bytes too.
        root = ET.fromstring(txt)
        out: List[dict] = []
        ny_tz = ZoneInfo("America/New_York")
        for ev in root.findall(".//event"):
            title = (ev.findtext("title") or "").strip()
            ctry = (ev.findtext("country") or "").strip().upper()
            impact = (ev.findtext("impact") or "").strip()
            date_s = (ev.findtext("date") or "").strip()
            time_s = (ev.findtext("time") or "").strip()
            # Date is typically MM-DD-YYYY in FF XML export
            dt_obj = None
            try:
                mm, dd, yy = date_s.split("-")
                hh, mi = (time_s.split(":") + ["0"])[:2] if time_s else ("0","0")
                dt_obj = datetime(int(yy), int(mm), int(dd), int(hh), int(mi), 0, tzinfo=ny_tz)
            except Exception:
                dt_obj = None
            if dt_obj is None:
                continue
            out.append({
                "title": title,
                "country": ctry,
                "impact": impact,
                "timestamp": float(dt_obj.astimezone(timezone.utc).timestamp()),
            })
        return out if out else None
    except Exception:
        return None
def _pair_to_ccys(pair: str) -> Tuple[str, str]:
    s = (pair or "").upper().replace(" ", "")
    s = s.replace("_", "/")
    if "/" in s:
        a, b = s.split("/", 1)
        return (a[:3], b[:3])
    # fallback: "USDJPY"
    if len(s) >= 6:
        return (s[:3], s[3:6])
    return ("", "")


def _pip_size(pair: str) -> float:
    s = (pair or "").upper()
    # Heuristic: JPY pairs use 0.01, others 0.0001
    return 0.01 if "JPY" in s else 0.0001

def _round_to_pip(x: float, pair: str) -> float:
    try:
        p = _pip_size(pair)
        return float(round(float(x) / p) * p)
    except Exception:
        return float(x)

def _fetch_ff_calendar(url: str, timeout: int = 12, ttl_sec: int = 1800) -> Tuple[Optional[List[dict]], str]:
    """Fetch weekly economic calendar (JSON preferred) with robust caching + XML/file fallback.

    Streamlit Cloud reruns the script frequently. Also, the free FF weekly export can occasionally
    return 429/5xx or time out. We therefore:
      1) use in-memory TTL cache
      2) try JSON fetch
      3) on failure, try XML export (same host)
      4) on failure, fall back to a local file cache (<=24h)
    """
    now = time.time()

    # 1) in-memory cache
    if (_EVENT_CACHE.get("data") is not None) and (_EVENT_CACHE.get("url") == url) and (now - float(_EVENT_CACHE.get("ts", 0.0)) < float(ttl_sec)):
        return _EVENT_CACHE["data"], "cache"

    def _urlopen_bytes(req: Request) -> bytes:
        # Default SSL context; on TLS issues fall back to unverified context
        try:
            ctx = ssl.create_default_context()
        except Exception:
            ctx = None
        try:
            try:
                with urlopen(req, timeout=timeout, context=ctx) as resp:
                    return resp.read()
            except TypeError:
                with urlopen(req, timeout=timeout) as resp:
                    return resp.read()
        except Exception:
            # last resort (some environments)
            try:
                uctx = ssl._create_unverified_context()
                with urlopen(req, timeout=timeout, context=uctx) as resp:
                    return resp.read()
            except Exception:
                raise

    # 2) JSON fetch
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (fx-analyzer; event-guard)", "Accept": "application/json,text/plain,*/*"})
        raw = _urlopen_bytes(req)
        data = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(data, list):
            raise ValueError("calendar json is not a list")
        _EVENT_CACHE.update({"ts": now, "url": url, "data": data, "err": None})
        _write_event_file_cache(url, data)
        return data, "ok"
    except Exception as e_json:
        # 3) XML fallback
        try:
            xml_url = _derive_xml_url(url)
            data_xml = _fetch_ff_calendar_xml(xml_url, timeout=timeout)
            if isinstance(data_xml, list) and data_xml:
                _EVENT_CACHE.update({"ts": now, "url": url, "data": data_xml, "err": f"json_fail={type(e_json).__name__}"})
                _write_event_file_cache(url, data_xml)
                return data_xml, "ok"
        except Exception:
            pass

        # 4) file cache fallback
        cached = _read_event_file_cache(max_age_sec=24 * 3600)
        if isinstance(cached, list) and cached:
            _EVENT_CACHE.update({"ts": now, "url": url, "data": cached, "err": f"json_fail={type(e_json).__name__}: {e_json}"})
            return cached, "cache"

        _EVENT_CACHE.update({"ts": now, "url": url, "data": None, "err": f"{type(e_json).__name__}: {e_json}"})
        return None, "fail"

def _parse_event_dt(item: dict) -> Optional[datetime]:
    """
    Try multiple schemas:
    - timestamp (epoch seconds)
    - datetime / date+time string
    """
    try:
        ts = item.get("timestamp", None)
        if isinstance(ts, (int, float)) and math.isfinite(float(ts)) and float(ts) > 0:
            # Many feeds use seconds; if it's too big, assume ms.
            tsv = float(ts)
            if tsv > 3e12:
                tsv = tsv / 1000.0
            return datetime.fromtimestamp(tsv, tz=timezone.utc)
    except Exception:
        pass

    # Common FF export fields: "date" + "time" (strings)
    dt_str = None
    for k in ("datetime", "date_time", "dt", "dateTime"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            dt_str = v.strip()
            break

    if dt_str is None:
        date_s = item.get("date", None)
        time_s = item.get("time", None)
        if isinstance(date_s, str) and date_s.strip():
            if isinstance(time_s, str) and time_s.strip():
                dt_str = f"{date_s.strip()} {time_s.strip()}"
            else:
                dt_str = date_s.strip()

    if not dt_str:
        return None

    # Parse with pandas (robust) then assume UTC if no tz
    try:
        dt = pd.to_datetime(dt_str, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def _compute_event_risk(
    pair: str,
    *,
    now_tz: str = "Asia/Tokyo",
    horizon_hours: int = 72,
    past_lookback_hours: int = 24,
    hours_scale: float = 24.0,
    norm: float = 3.0,
    impacts: Optional[List[str]] = None,
    high_window_minutes: int = 60,
    url: str = _FF_CAL_URL_DEFAULT,
) -> Dict[str, Any]:
    """
    Swing-oriented economic calendar risk model.

    Returns:
      {
        "ok": bool,
        "status": "ok|cache|fail",
        "err": str|None,
        "score": float,                  # upcoming-only risk score (heuristic)
        "factor": float (0..1),          # normalized risk factor (upcoming-only)
        "window_high": bool,             # high-impact within ±high_window_minutes
        "next_high_hours": float|None,   # hours until next High impact (>=0)
        "last_high_hours": float|None,   # hours since last High impact (>=0)
        "next_any_hours": float|None,    # hours until next (any impact)
        "last_any_hours": float|None,    # hours since last (any impact)
        "impact_ccys": { "USD": {"upcoming": n, "recent": n}, ... },
        "upcoming": [ {dt_utc, currency, impact, title, hours} ... ] (<=10),
        "recent":   [ {dt_utc, currency, impact, title, hours} ... ] (<=10),  # hours is negative (in the past)
      }
    """
    impacts = impacts or ["High", "Medium"]
    a, b = _pair_to_ccys(pair)
    ccys = {a, b}

    data, status = _fetch_ff_calendar(url)
    if not data:
        return {
            "ok": False,
            "status": status,
            "err": _EVENT_CACHE.get("err"),
            "score": 0.0,
            "factor": 0.0,
            "window_high": False,
            "next_high_hours": None,
            "last_high_hours": None,
            "next_any_hours": None,
            "last_any_hours": None,
            "impact_ccys": {},
            "upcoming": [],
            "recent": [],
        }

    tz = ZoneInfo(now_tz)
    now_local = datetime.now(tz=tz)
    now_utc = now_local.astimezone(timezone.utc)

    # weights by impact
    w = {"High": 1.0, "Medium": 0.6, "Low": 0.3, "Holiday": 0.8, "Non-Economic": 0.2}

    events = []
    for it in data:
        if not isinstance(it, dict):
            continue
        cur = str(it.get("currency") or it.get("ccy") or it.get("cur") or "").upper().strip()
        if not cur:
            ctry = str(it.get("country") or "").upper().strip()
            if len(ctry) == 3:
                cur = ctry
        if cur and (cur not in ccys):
            continue

        impact = str(it.get("impact") or "").strip()
        if impacts and (impact not in impacts) and not (impact == "Holiday" and "Holiday" in impacts):
            continue

        dt = _parse_event_dt(it)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)

        hrs = (dt_utc - now_utc).total_seconds() / 3600.0
        if hrs < -float(max(1, int(past_lookback_hours))):
            continue
        if hrs > float(horizon_hours):
            continue

        title = str(it.get("title") or it.get("event") or it.get("name") or "").strip()
        events.append({
            "dt_utc": dt_utc,
            "hours": float(hrs),
            "currency": cur,
            "impact": impact,
            "title": title,
        })

    events.sort(key=lambda x: x["hours"])

    # next/last helpers
    next_high = None
    last_high = None
    next_any = None
    last_any = None
    window_high = False

    # score: upcoming-only
    score = 0.0

    impact_ccys: Dict[str, Dict[str, int]] = {}
    def _inc(cur: str, k: str) -> None:
        if not cur:
            return
        if cur not in impact_ccys:
            impact_ccys[cur] = {"upcoming": 0, "recent": 0}
        impact_ccys[cur][k] = int(impact_ccys[cur].get(k, 0)) + 1

    for ev in events:
        impact = ev["impact"]
        hrs = float(ev["hours"])

        # window check for High impact (±minutes)
        if impact == "High":
            if abs(hrs) * 60.0 <= float(high_window_minutes):
                window_high = True

        if hrs >= 0:
            # upcoming
            if next_any is None:
                next_any = hrs
            if impact == "High" and next_high is None:
                next_high = hrs
            _inc(str(ev.get("currency") or ""), "upcoming")

            denom = 1.0 + (max(0.0, hrs) / max(1e-6, float(hours_scale)))
            score += float(w.get(impact, 0.4)) / denom
        else:
            # recent
            if last_any is None:
                last_any = abs(hrs)
            if impact == "High" and last_high is None:
                last_high = abs(hrs)
            _inc(str(ev.get("currency") or ""), "recent")

    # normalize to factor 0..1 (heuristic)
    factor = _clamp(score / max(1e-6, float(norm)), 0.0, 1.0)

    # trim lists for UI
    upcoming_ui = []
    recent_ui = []
    for ev in events:
        rec = {
            "dt_utc": ev["dt_utc"].isoformat(),
            "hours": float(ev["hours"]),
            "currency": ev["currency"],
            "impact": ev["impact"],
            "title": ev["title"],
        }
        if float(ev["hours"]) >= 0 and len(upcoming_ui) < 10:
            upcoming_ui.append(rec)
        if float(ev["hours"]) < 0 and len(recent_ui) < 10:
            recent_ui.append(rec)

    return {
        "ok": True,
        "status": status,
        "err": None,
        "score": float(score),
        "factor": float(factor),
        "window_high": bool(window_high),
        "next_high_hours": (float(next_high) if next_high is not None else None),
        "last_high_hours": (float(last_high) if last_high is not None else None),
        "next_any_hours": (float(next_any) if next_any is not None else None),
        "last_any_hours": (float(last_any) if last_any is not None else None),
        "impact_ccys": impact_ccys,
        "upcoming": upcoming_ui,
        "recent": recent_ui,
    }

def _compute_weekend_risk(now_tz: str = "Asia/Tokyo") -> float:
    """
    Simple weekend gap risk proxy:
    - Fri evening (>=18:00 local) => 1.0
    - Sat/Sun => 1.0
    else 0.0
    """
    try:
        tz = ZoneInfo(now_tz)
        now = datetime.now(tz=tz)
        wd = now.weekday()  # Mon=0..Sun=6
        if wd >= 5:
            return 1.0
        if wd == 4 and now.hour >= 18:
            return 1.0
        return 0.0
    except Exception:
        return 0.0

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default



def _compute_unrealized_R(side: str, entry: float, sl: float, price: float) -> float:
    """Unrealized PnL in R units (R = initial stop distance). Positive is profit."""
    try:
        side = str(side or "").upper()
        entry = float(entry)
        sl = float(sl)
        price = float(price)
        risk = abs(entry - sl)
        if risk <= 1e-9:
            return 0.0
        if side == "SELL":
            return (entry - price) / risk
        return (price - entry) / risk
    except Exception:
        return 0.0


def _hold_manage_reco(
    pair: str,
    df: pd.DataFrame,
    ctx_in: Dict[str, Any],
    plan_like: Dict[str, Any],
    ev_meta: Dict[str, Any],
    weekend_risk: float,
    weekcross_risk: float,
) -> Dict[str, Any]:
    """Position-holding management rules for swing (event/weekend approach).
    Returns recommendation dict; empty dict if no position info.
    """
    try:
        pos = ctx_in.get("position") or ctx_in.get("pos") or {}
        if not isinstance(pos, dict):
            pos = {}
        pos_open = bool(ctx_in.get("position_open", False) or pos.get("open") or pos.get("is_open") or (len(pos) > 0))
        if not pos_open:
            return {}
        # Get position params (fallback to plan)
        side = str(pos.get("side") or pos.get("pos_side") or plan_like.get("side") or "").upper()
        if side not in ("BUY", "SELL"):
            side = str(plan_like.get("side") or "BUY").upper()
        entry = float(pos.get("entry") or pos.get("entry_price") or pos.get("pos_entry") or plan_like.get("entry") or plan_like.get("entry_price") or 0.0)
        sl = float(pos.get("sl") or pos.get("stop_loss") or pos.get("pos_sl") or plan_like.get("sl") or plan_like.get("stop_loss") or 0.0)
        tp = float(pos.get("tp") or pos.get("take_profit") or pos.get("pos_tp") or plan_like.get("tp") or plan_like.get("take_profit") or 0.0)

        # Current price (user can override; else latest close)
        price = None
        for k in ("current_price", "price", "last_price", "mark_price"):
            if k in pos and pos.get(k) is not None:
                price = pos.get(k)
                break
        if price is None:
            price = ctx_in.get("pos_current_price", None)
        if price is None:
            try:
                price = float(df["Close"].astype(float).iloc[-1]) if isinstance(df, pd.DataFrame) and (not df.empty) else float(entry)
            except Exception:
                price = float(entry)
        price = float(price)

        unrealized_R = pos.get("unrealized_R", None)
        if unrealized_R is None:
            unrealized_R = _compute_unrealized_R(side, entry, sl, price)
        else:
            unrealized_R = float(unrealized_R)

        dd_R = float(pos.get("dd_R") or pos.get("max_dd_R") or pos.get("drawdown_R") or 0.0)

        nh = ev_meta.get("next_high_hours", None)
        try:
            nh_f = (float(nh) if nh is not None else None)
        except Exception:
            nh_f = None

        event_factor = float(ev_meta.get("factor", 0.0) or 0.0)
        window_high = bool(ev_meta.get("window_high", False))

        # ---- Mandatory swing holding rules (not optional) ----
        # Base thresholds (hours)
        no_add_h = float(ctx_in.get("hold_no_add_hours", 48.0) or 48.0)
        reduce_h = float(ctx_in.get("hold_reduce_hours", 18.0) or 18.0)
        be_h = float(ctx_in.get("hold_breakeven_hours", 12.0) or 12.0)
        partial_h = float(ctx_in.get("hold_partial_tp_hours", 18.0) or 18.0)

        # Guardrails
        no_add_h = max(6.0, min(168.0, no_add_h))
        reduce_h = max(3.0, min(72.0, reduce_h))
        be_h = max(1.0, min(48.0, be_h))
        partial_h = max(1.0, min(72.0, partial_h))

        notes: List[str] = []
        actions: List[str] = []

        no_add = False
        if (nh_f is not None) and (nh_f <= no_add_h):
            no_add = True
            notes.append(f"高インパクト指標が{nh_f:.1f}時間以内 → 追加建て禁止（スイングでも実行リスク回避）")
        if float(weekend_risk or 0.0) > 0.0:
            no_add = True
            notes.append("週末ギャップリスク → 追加建て禁止")
        if float(weekcross_risk or 0.0) > 0.0:
            no_add = True
            notes.append("週跨ぎ（木金）リスク → 追加建て禁止")

        # Size shrink recommendation (0.2..1.0)
        reduce_mult = 1.0
        if (nh_f is not None) and (nh_f <= reduce_h):
            # base shrink by event factor
            reduce_mult = min(reduce_mult, float(_clamp(1.0 - 0.60 * event_factor, 0.20, 1.00)))
            if unrealized_R >= 0.20:
                reduce_mult = min(reduce_mult, 0.50)
                actions.append("REDUCE_SIZE")
                notes.append("イベント接近＆含み益あり → 建玉の一部縮退（例：半分）を推奨")
            else:
                reduce_mult = min(reduce_mult, 0.70)
                actions.append("REDUCE_SIZE")
                notes.append("イベント接近 → 建玉縮退を推奨（リスク低減）")

        if float(weekend_risk or 0.0) > 0.0:
            reduce_mult = min(reduce_mult, 0.60)
            if "REDUCE_SIZE" not in actions:
                actions.append("REDUCE_SIZE")
            notes.append("週末前 → 建玉縮退を推奨（ギャップ対策）")

        # Partial take profit (0..1)
        partial_tp = 0.0
        if (nh_f is not None) and (nh_f <= partial_h) and (unrealized_R >= 0.60):
            partial_tp = max(partial_tp, 0.50)
            actions.append("PARTIAL_TP")
            notes.append("含み益0.6R以上＆イベント近接 → 半分利確を推奨")
        if window_high and (unrealized_R >= 0.30):
            partial_tp = max(partial_tp, 0.50)
            if "PARTIAL_TP" not in actions:
                actions.append("PARTIAL_TP")
            notes.append("高インパクト窓内 → 半分利確を推奨（スリッページ/乱高下対策）")

        # Move SL to breakeven / tighten
        move_be = False
        new_sl = None
        if (nh_f is not None) and (nh_f <= be_h) and (unrealized_R >= 0.15):
            move_be = True
            new_sl = float(entry)
            actions.append("MOVE_SL_TO_BE")
            notes.append("イベント接近 → ストップを建値（BE）へ移動を推奨（勝ちを負けにしない）")
        if float(weekend_risk or 0.0) > 0.0 and (unrealized_R >= 0.15):
            move_be = True
            if new_sl is None:
                new_sl = float(entry)
            if "MOVE_SL_TO_BE" not in actions:
                actions.append("MOVE_SL_TO_BE")
            notes.append("週末前 → 建値/浅い利確で防御を推奨")

        # Round new SL to pip
        try:
            if new_sl is not None:
                new_sl = _round_to_pip(float(new_sl), pair)
        except Exception:
            pass

        # Compose
        out = {
            "version": "swing_hold_v1",
            "pair": str(pair),
            "side": side,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "current_price": float(price),
            "unrealized_R": float(unrealized_R),
            "dd_R": float(dd_R),
            "event_next_high_hours": nh_f,
            "event_window_high": bool(window_high),
            "event_risk_factor": float(event_factor),
            "weekend_risk": float(weekend_risk or 0.0),
            "weekcross_risk": float(weekcross_risk or 0.0),
            "no_add": bool(no_add),
            "reduce_size_mult": float(_clamp(reduce_mult, 0.20, 1.00)),
            "partial_tp_ratio": float(_clamp(partial_tp, 0.0, 1.0)),
            "move_sl_to_be": bool(move_be),
            "new_sl_reco": (float(new_sl) if new_sl is not None else None),
            "actions": list(dict.fromkeys(actions)),  # unique preserve order
            "notes": notes,
        }
        return out
    except Exception:
        return {}
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(3, n//2)).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # Simple Wilder ADX
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([(high - low).abs(),
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx.fillna(0.0)

def _slope_norm(s: pd.Series, lookback: int = 10) -> float:
    # Normalized slope over lookback (last - first) / (std * sqrt(n))
    if len(s) < lookback + 1:
        return 0.0
    y = s.iloc[-lookback:].astype(float).values
    if np.all(np.isfinite(y)) is False:
        return 0.0
    dy = y[-1] - y[0]
    sd = float(np.std(y))
    if sd <= 1e-9:
        return 0.0
    return float(dy / (sd * math.sqrt(lookback)))

def _hh_hl_ok(df: pd.DataFrame, n: int = 20) -> bool:
    # crude HH/HL: compare last 2 swing highs and lows via rolling window peaks/valleys
    if len(df) < n + 5:
        return False
    close = df["Close"].astype(float)
    # detect local maxima/minima with a small window
    w = 3
    highs = (close.shift(w) < close) & (close.shift(-w) < close)
    lows = (close.shift(w) > close) & (close.shift(-w) > close)
    hi_idx = close[highs].tail(4).index
    lo_idx = close[lows].tail(4).index
    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return False
    h1, h2 = close.loc[hi_idx[-2]], close.loc[hi_idx[-1]]
    l1, l2 = close.loc[lo_idx[-2]], close.loc[lo_idx[-1]]
    return (h2 > h1) and (l2 > l1)

def _breakout_strength(df: pd.DataFrame, n: int = 20) -> Tuple[bool, float]:
    # breakout if last close exceeds prior n-day high by >= 0.2*ATR
    if len(df) < n + 5:
        return False, 0.0
    close = df["Close"].astype(float)
    hi = df["High"].astype(float).rolling(n).max()
    atr = _atr(df, 14)
    last = close.iloc[-1]
    prev_hi = hi.iloc[-2]  # prior window high
    a = float(atr.iloc[-1]) if len(atr) else 0.0
    if not np.isfinite(prev_hi) or not np.isfinite(last) or a <= 0:
        return False, 0.0
    excess = last - prev_hi
    ok = excess > 0.2 * a
    strength = _clamp(excess / (1.5 * a), 0.0, 1.0) if a > 0 else 0.0
    return bool(ok), float(strength)

def _continuation_prob(df: pd.DataFrame, horizon: int = 5) -> Tuple[float, float]:
    """
    Simple continuation probability model (not ML-heavy):
    - Uses trend slope, adx, and breakout strength to estimate p(continue up) / p(continue down).
    """
    if len(df) < 60:
        return 0.5, 0.5
    close = df["Close"].astype(float)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    slope = _slope_norm(ema20, 12)
    adx = float(_adx(df, 14).iloc[-1])  # 0..100
    breakout_ok, bstr = _breakout_strength(df, 20)

    # Map to 0..1 features
    f_adx = _clamp((adx - 15.0) / 25.0, 0.0, 1.0)
    f_slope = _clamp((slope + 2.0) / 4.0, 0.0, 1.0)  # slope ~[-2,2] -> [0,1]
    f_trend = 0.55 * f_slope + 0.35 * f_adx + 0.10 * bstr
    # direction bias
    bias = 1.0 if ema20.iloc[-1] >= ema50.iloc[-1] else 0.0
    # base probabilities
    p_up = 0.35 + 0.55 * f_trend
    p_dn = 0.35 + 0.55 * (1.0 - f_trend)

    # apply direction bias and breakout
    if bias > 0.5:
        p_up += 0.05 + 0.10 * bstr
        p_dn -= 0.05
    else:
        p_dn += 0.05 + 0.10 * bstr
        p_up -= 0.05

    if breakout_ok:
        if bias > 0.5:
            p_up += 0.06
            p_dn -= 0.03
        else:
            p_dn += 0.06
            p_up -= 0.03

    p_up = _clamp(p_up, 0.05, 0.95)
    p_dn = _clamp(p_dn, 0.05, 0.95)
    return float(p_up), float(p_dn)

def _phase_label(df: pd.DataFrame) -> Tuple[str, float, float]:
    """
    Returns (phase_label, trend_strength 0..1, momentum_score -1..+1)
    """
    if len(df) < 60:
        return "UNKNOWN", 0.0, 0.0
    close = df["Close"].astype(float)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200) if len(close) >= 210 else _ema(close, 100)

    slope20 = _slope_norm(ema20, 12)  # roughly -? .. +?
    adx = float(_adx(df, 14).iloc[-1])
    breakout_ok, bstr = _breakout_strength(df, 20)
    hhhl = _hh_hl_ok(df, 30)

    # strength from ADX and breakout strength and slope magnitude
    s_adx = _clamp((adx - 12.0) / 28.0, 0.0, 1.0)
    s_slope = _clamp(abs(slope20) / 2.0, 0.0, 1.0)
    strength = _clamp(0.55 * s_adx + 0.30 * s_slope + 0.15 * bstr, 0.0, 1.0)

    # momentum: signed slope + small ema alignment
    align = 1.0 if ema20.iloc[-1] > ema50.iloc[-1] else -1.0
    mom = _clamp((slope20 / 2.0) + 0.20 * align, -1.0, 1.0)

    # classify
    if breakout_ok and strength >= 0.35:
        phase = "BREAKOUT_UP" if mom > 0 else "BREAKOUT_DOWN"
    else:
        if strength < 0.25:
            phase = "RANGE"
        else:
            phase = "UP_TREND" if mom > 0 else "DOWN_TREND"

    # refine by HH/HL
    if phase == "UP_TREND" and not hhhl and strength < 0.40:
        phase = "TRANSITION_UP"
    if phase == "DOWN_TREND" and strength < 0.40:
        phase = "TRANSITION_DOWN"

    return phase, float(strength), float(mom)

# ---------------------------------------------------------------------
# Core: strategy output
# ---------------------------------------------------------------------

@dataclass
class StrategyPlan:
    decision: str
    direction: str
    entry: float
    sl: float
    tp: float
    ev_raw: float
    ev_adj: float
    dynamic_threshold: float
    confidence: float
    why: str
    veto: List[str]
    ctx: Dict[str, Any]

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Backward compatible helper. main.py may call this."""
    if df is None or len(df) < 5:
        return {}
    close = df["Close"].astype(float)
    atr = _atr(df, 14)
    adx = _adx(df, 14)
    return {
        "ema20": float(_ema(close, 20).iloc[-1]),
        "ema50": float(_ema(close, 50).iloc[-1]),
        "atr14": float(atr.iloc[-1]) if len(atr) else 0.0,
        "adx14": float(adx.iloc[-1]) if len(adx) else 0.0,
    }


def get_ai_order_strategy(
    price_df: pd.DataFrame = None,
    pair: str = "",
    budget_yen: int = 0,
    context_data: Optional[Dict[str, Any]] = None,
    ext_features: Optional[Dict[str, Any]] = None,
    prefer_long_only: bool = False,
    api_key: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """
    main.py 互換の“完成版”エントリー判定（ctx依存を排除し、内部で必要な特徴量を計算します）。

    完成要件（要点）:
      - 統合スコア（rank_score）で最終ランキングできる
      - 価格構造（HH/HL, ブレイク, トレンド強度, クローズ構造）を最優先
      - 通貨強弱は補助（単独で方向決定しない）
      - マクロは弱いバイアス（NO連発の主因にしない）
      - イベントは「直前の実行リスク」＋「直後の捕獲」を必須化
        * 直前: 成行禁止/縮退/閾値引上げ（ただし見送り地獄にしない）
        * 直後: 0-1h様子見、1-24hブレイク専用ゲートで積極再評価
      - veto乱立を抑え、主因が説明できる
      - NameError / SyntaxError ゼロ（phase_label等の未定義を根絶）
      - RLは出口専用（本関数は入口のみ）

    互換性:
      - main.py が期待する key を返します（expected_R_ev / p_win_ev / veto_reasons / state_probs / ev_contribs 等）
      - 旧key（entry/sl/tp, ev_raw/ev_adj, veto）も残します
      - 追加key: rank_score / final_score / p_eff / event_mode / event_last_high_hours 等
    """

    # --- safety init (NameError防止) ---
    phase = "UNKNOWN"
    phase_label = "UNKNOWN"

    # -----------------------------------------------------------------
    # 1) 引数の吸収（互換）
    # -----------------------------------------------------------------
    if not pair:
        pair = (kwargs.get("pair_label") or kwargs.get("symbol") or kwargs.get("pair") or "") or ""

    if ext_features is None:
        ext_features = kwargs.get("ext") or kwargs.get("external_features") or kwargs.get("ext_meta") or None

    if context_data is None:
        context_data = kwargs.get("ctx") or kwargs.get("context") or kwargs.get("_ctx") or None

    ctx_in = context_data or {}
    ext = ext_features or {}

    df = price_df
    if df is None:
        df = kwargs.get("df") or kwargs.get("price_history") or kwargs.get("price_data")

    if df is None and isinstance(ctx_in, dict):
        df = ctx_in.get("_df") or ctx_in.get("df") or ctx_in.get("price_df") or ctx_in.get("price_history")

    if df is not None and not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            df = None

    # OHLC列名の正規化
    if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
        cols = {c.lower(): c for c in df.columns}
        rename = {}
        for need in ["open", "high", "low", "close", "volume"]:
            if need in cols and cols[need] != need.capitalize():
                rename[cols[need]] = need.capitalize()
        if rename:
            df = df.rename(columns=rename)
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

    need_cols = {"High", "Low", "Close"}
    if df is None or (not isinstance(df, pd.DataFrame)) or len(df) < 60 or (not need_cols.issubset(set(df.columns))):
        debug_cols = []
        try:
            debug_cols = list(df.columns) if isinstance(df, pd.DataFrame) else []
        except Exception:
            debug_cols = []
        thr = float(_clamp(_safe_float((ctx_in or {}).get("min_expected_R", 0.10), 0.10), 0.03, 0.30))
        return {
            "decision": "NO_TRADE",
            "direction": "LONG",
            "side": "BUY",
            "order_type": "—",
            "entry_type": "—",
            "entry": 0.0,
            "entry_price": 0.0,
            "sl": 0.0, "stop_loss": 0.0,
            "tp": 0.0, "take_profit": 0.0,
            "trail_sl": 0.0,
            "extend_factor": 1.0,
            "ev_raw": 0.0, "ev_adj": 0.0,
            "expected_R_ev_raw": 0.0,
            "expected_R_ev_adj": 0.0,
            "expected_R_ev": 0.0,
            "rank_score": 0.0,
            "final_score": 0.0,
            "dynamic_threshold": thr,
            "gate_mode": "NO_DATA",
            "confidence": 0.0,
            "p_win": 0.0,
            "p_eff": 0.0,
            "p_win_ev": 0.0,
            "why": "データ不足",
            "veto": ["データ不足（最低60本必要 / High・Low・Close必須）"],
            "veto_reasons": ["データ不足（最低60本必要 / High・Low・Close必須）"],
            "event_mode": "NO_DATA",
            "event_next_high_hours": None,
            "event_last_high_hours": None,
            "state_probs": {"trend_up": 0.0, "trend_down": 0.0, "range": 0.0, "risk_off": 0.0},
            "ev_contribs": {"trend_up": 0.0, "trend_down": 0.0, "range": 0.0, "risk_off": 0.0},
            "_ctx": {"pair": pair, "len": int(len(df)) if isinstance(df, pd.DataFrame) else 0, "cols": debug_cols},
        }

    df = df.copy()
    close = df["Close"].astype(float)
    last = float(close.iloc[-1])

    # -----------------------------------------------------------------
    # 2) 内部特徴量（ctx依存排除）
    # -----------------------------------------------------------------
    phase, strength, mom = _phase_label(df)
    horizon = int(_safe_float(ctx_in.get("horizon_days", 5), 5))
    p_up, p_dn = _continuation_prob(df, horizon=max(3, horizon))

    breakout_ok, breakout_strength = _breakout_strength(df, 20)
    hhhl_ok = _hh_hl_ok(df, 30)

    # 表示用ラベル（NameError根絶）
    if str(phase) == "RANGE":
        phase_label = "RANGE"
    elif str(phase) in ("UP_TREND", "BREAKOUT_UP", "TRANSITION_UP"):
        phase_label = "UP_TREND"
    elif str(phase) in ("DOWN_TREND", "BREAKOUT_DOWN", "TRANSITION_DOWN"):
        phase_label = "DOWN_TREND"
    else:
        phase_label = str(phase or "UNKNOWN")

    # -----------------------------------------------------------------
    # 3) 方向選択（通貨強弱“単独”は不可。価格構造主導）
    # -----------------------------------------------------------------
    direction = "LONG" if mom >= 0 else "SHORT"
    if prefer_long_only:
        direction = "LONG"
    if phase_label == "UP_TREND" and float(strength) >= 0.28:
        direction = "LONG"
    if phase_label == "DOWN_TREND" and float(strength) >= 0.28:
        direction = "SHORT"

    # RANGEは“端”なら逆張り優先（ただし厳格条件）
    lookback = 20
    recent_low = float(df["Low"].astype(float).tail(lookback).min())
    recent_high = float(df["High"].astype(float).tail(lookback).max())
    span = float(max(1e-9, recent_high - recent_low))
    range_pos = float(_clamp((last - recent_low) / span, 0.0, 1.0))
    if phase_label == "RANGE":
        if range_pos <= 0.30:
            direction = "LONG"
        elif range_pos >= 0.70:
            direction = "SHORT"

    side = "BUY" if direction == "LONG" else "SELL"

    # -----------------------------------------------------------------
    # 4) リスクモデル（SL/TP）: ATRベース
    # -----------------------------------------------------------------
    atr14 = float(_atr(df, 14).iloc[-1])
    atr14 = max(atr14, 1e-6)

    if direction == "LONG":
        sl = min(last - 1.2 * atr14, recent_low - 0.15 * atr14)
        tp = last + (2.2 * atr14 if phase_label in ("UP_TREND",) else 1.6 * atr14)
    else:
        sl = max(last + 1.2 * atr14, recent_high + 0.15 * atr14)
        tp = last - (2.2 * atr14 if phase_label in ("DOWN_TREND",) else 1.6 * atr14)

    entry = last
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 1e-9:
        risk = atr14
    rr = reward / risk

    # -----------------------------------------------------------------
    # 5) 勝率 proxy（モデル）→ confidenceで縮退（p_eff）
    # -----------------------------------------------------------------
    cont_best = max(float(p_up), float(p_dn))
    if direction == "LONG":
        p_win_model = 0.46 + 0.42 * _clamp((float(p_up) - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (float(strength) - 0.5)
    else:
        p_win_model = 0.46 + 0.42 * _clamp((float(p_dn) - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (float(strength) - 0.5)
    p_win_model = float(_clamp(p_win_model, 0.20, 0.80))

    # 信頼度（0..1）
    structure_flag = 1.0 if (breakout_ok or hhhl_ok) else 0.0
    confidence = float(_clamp(
        0.30
        + 0.40 * float(strength)
        + 0.18 * (float(cont_best) - 0.5)
        + 0.08 * _clamp(float(rr - 1.0), -1.0, 2.0)
        + 0.04 * float(structure_flag),
        0.0, 1.0
    ))

    # p_eff: confidenceが低いほど0.5に寄せる（整合崩れ対策）
    conf_k = float(_clamp(confidence / 0.75, 0.0, 1.0))
    p_eff = float(_clamp(0.5 + (p_win_model - 0.5) * conf_k, 0.20, 0.80))

    # EV (R): EV = p*RR - (1-p)*1
    ev_raw = float(p_eff * float(rr) - (1.0 - p_eff) * 1.0)

    # -----------------------------------------------------------------
    # 6) 外部リスク（macro）: 弱いバイアスとして統合（NO連発の主因にしない）
    # -----------------------------------------------------------------
    gr = _safe_float(ext.get("global_risk_index", ext.get("global_risk", ext.get("risk_off", 0.35))), 0.35)
    war = _safe_float(ext.get("war_probability", ext.get("war", 0.0)), 0.0)
    macro_risk = _safe_float(ext.get("macro_risk_score", None), float("nan"))
    if not (isinstance(macro_risk, (int, float)) and math.isfinite(float(macro_risk))):
        macro_risk = _clamp(0.70 * gr + 0.30 * war, 0.0, 1.0)
    else:
        macro_risk = _clamp(float(macro_risk), 0.0, 1.0)

    # 表示用（ev_adj）は弱いペナルティに留める
    risk_penalty = 0.10 + 0.50 * float(macro_risk)   # 0.10..0.60
    ev_adj = float(ev_raw - 0.18 * float(risk_penalty))

    # -----------------------------------------------------------------
    # 6.5) 経済指標/イベント（直前の実行リスク + 直後の捕獲）
    # -----------------------------------------------------------------
    event_guard_enable = bool(ctx_in.get('event_guard_enable', True))
    event_block_window = bool(ctx_in.get('event_block_high_impact_window', True))
    event_horizon_hours = int(_safe_float(ctx_in.get("event_horizon_hours", 168), 168))
    event_past_lookback_hours = int(_safe_float(ctx_in.get("event_past_lookback_hours", 24), 24))
    event_window_minutes = int(_safe_float(ctx_in.get("event_window_minutes", 60), 60))
    event_impacts = ctx_in.get("event_impacts", None)
    if not isinstance(event_impacts, list) or not event_impacts:
        event_impacts = ["High", "Medium"]
    event_calendar_url = str(ctx_in.get("event_calendar_url", _FF_CAL_URL_DEFAULT) or _FF_CAL_URL_DEFAULT)

    ev_meta = {"ok": False, "status": "off", "err": None, "score": 0.0, "factor": 0.0,
               "window_high": False, "next_high_hours": None, "last_high_hours": None,
               "next_any_hours": None, "last_any_hours": None, "upcoming": [], "recent": [], "impact_ccys": {}}
    if event_guard_enable:
        try:
            ev_meta = _compute_event_risk(
                pair,
                now_tz=str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo"),
                horizon_hours=event_horizon_hours,
                past_lookback_hours=event_past_lookback_hours,
                hours_scale=float(ctx_in.get("event_hours_scale", 24.0) or 24.0),
                norm=float(ctx_in.get("event_norm", 3.0) or 3.0),
                impacts=[str(x) for x in event_impacts],
                high_window_minutes=event_window_minutes,
                url=event_calendar_url,
            )
        except Exception as e:
            ev_meta = {"ok": False, "status": "fail", "err": f"{type(e).__name__}: {e}", "score": 0.0, "factor": 0.0,
                       "window_high": False, "next_high_hours": None, "last_high_hours": None,
                       "next_any_hours": None, "last_any_hours": None, "upcoming": [], "recent": [], "impact_ccys": {}}

    weekend_risk = float(_compute_weekend_risk(now_tz=str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo")))

    # Thu/Fri are special for swing entries (weekend gap approaches + event clusters).
    try:
        _tz = ZoneInfo(str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo"))
        _now_local = datetime.now(tz=_tz)
        _wd = int(_now_local.weekday())  # Mon=0..Sun=6
        weekcross_risk = 1.0 if _wd in (3, 4) else 0.0  # Thu/Fri
        weekcross_weekday = _wd
    except Exception:
        weekcross_risk = 0.0
        weekcross_weekday = None

    # event mode classification (swing)
    try:
        next_high = ev_meta.get("next_high_hours", None)
        last_high = ev_meta.get("last_high_hours", None)
        pre_h = float(ctx_in.get("event_preblock_hours", 24.0) or 24.0)
        pre_h = float(_clamp(pre_h, 6.0, 72.0))
    except Exception:
        next_high = None
        last_high = None
        pre_h = 24.0

    event_mode = "NORMAL"
    if bool(ev_meta.get("window_high", False)) and event_block_window:
        event_mode = "EVENT_WINDOW"
    elif (last_high is not None) and (float(last_high) <= 1.0):
        event_mode = "POST_WAIT"          # 0-1h: wait
    elif (last_high is not None) and (float(last_high) <= 24.0):
        event_mode = "POST_BREAKOUT"      # 1-24h: breakout-only gate
    elif (next_high is not None) and (float(next_high) <= float(pre_h)):
        event_mode = "PRE_EVENT"          # upcoming high-impact is close
    else:
        event_mode = "NORMAL"

    # -----------------------------------------------------------------
    # 7) 動的閾値（フェーズ/構造優先 + リスク時に軽く上げる）
    # -----------------------------------------------------------------
    base_thr = _safe_float(ctx_in.get("dynamic_threshold_base", None), float("nan"))
    if not (isinstance(base_thr, (int, float)) and math.isfinite(float(base_thr))):
        base_thr = _safe_float(ctx_in.get("min_expected_R", 0.08), 0.08)
    base_thr = float(_clamp(float(base_thr), 0.03, 0.25))

    thr_mult = 1.0
    if phase_label in ("UP_TREND", "DOWN_TREND"):
        thr_mult -= 0.16 * float(strength)
    if str(phase).startswith("BREAKOUT"):
        thr_mult -= 0.22 * max(float(strength), float(breakout_strength))
    if phase_label == "RANGE":
        thr_mult += 0.10

    dynamic_threshold = float(_clamp(base_thr * thr_mult, 0.02, 0.30))

    # macro bias is weak
    dynamic_threshold = float(_clamp(dynamic_threshold + 0.03 * float(macro_risk), 0.02, 0.30))

    # upcoming event / weekend / weekcross: threshold add (but do not cause perpetual NO)
    try:
        event_thr_add = float(ctx_in.get("event_threshold_add", 0.18) or 0.18)
        event_thr_add = float(_clamp(event_thr_add, 0.10, 0.30))
        weekend_thr_add = float(ctx_in.get("weekend_threshold_add", 0.03) or 0.03)
        weekend_thr_add = float(_clamp(weekend_thr_add, 0.0, 0.20))
        weekcross_thr_add = float(ctx_in.get("weekcross_threshold_add", 0.03) or 0.03)
        weekcross_thr_add = float(_clamp(weekcross_thr_add, 0.0, 0.20))

        ef = float(ev_meta.get("factor", 0.0) or 0.0)
        # POST_BREAKOUTでは“捕獲”を優先し、閾値上乗せを弱める
        if event_mode == "POST_BREAKOUT":
            ef *= 0.40
        dynamic_threshold = float(_clamp(
            dynamic_threshold
            + event_thr_add * ef
            + weekend_thr_add * float(weekend_risk or 0.0)
            + weekcross_thr_add * float(weekcross_risk or 0.0),
            0.02, 0.30
        ))
    except Exception:
        pass

    # -----------------------------------------------------------------
    # 8) モメンタム/通貨強弱（補助のみ、上限あり）
    # -----------------------------------------------------------------
    mom_bonus = 0.0
    if direction == "LONG" and mom > 0:
        mom_bonus = 0.06 * _clamp(float(strength), 0.0, 1.0) * _clamp(float(p_up), 0.0, 1.0)
    if direction == "SHORT" and mom < 0:
        mom_bonus = 0.06 * _clamp(float(strength), 0.0, 1.0) * _clamp(float(p_dn), 0.0, 1.0)
    mom_bonus = float(_clamp(mom_bonus, 0.0, 0.06))

    ccy_strength_proxy = 0.0
    try:
        c = df["Close"].astype(float)
        r20 = (c.iloc[-1] / c.iloc[-21] - 1.0) if len(c) >= 21 else 0.0
        r60 = (c.iloc[-1] / c.iloc[-61] - 1.0) if len(c) >= 61 else 0.0
        vol20 = float(c.pct_change().rolling(20).std().iloc[-1]) if len(c) >= 21 else 0.0
        vol60 = float(c.pct_change().rolling(60).std().iloc[-1]) if len(c) >= 61 else 0.0
        z20 = (r20 / (vol20 + 1e-9)) if vol20 > 0 else 0.0
        z60 = (r60 / (vol60 + 1e-9)) if vol60 > 0 else 0.0
        ccy_strength_proxy = float(_clamp(0.5 * z20 + 0.5 * z60, -1.0, 1.0))
    except Exception:
        ccy_strength_proxy = 0.0

    ccy_bonus = 0.0
    if direction == "LONG" and ccy_strength_proxy > 0:
        ccy_bonus = 0.04 * _clamp(abs(ccy_strength_proxy), 0.0, 1.0)
    elif direction == "SHORT" and ccy_strength_proxy < 0:
        ccy_bonus = 0.04 * _clamp(abs(ccy_strength_proxy), 0.0, 1.0)
    ccy_bonus = float(_clamp(ccy_bonus, 0.0, 0.04))

    ev_gate = float(ev_raw + mom_bonus + ccy_bonus)

    # -----------------------------------------------------------------
    # 9) 構造ゲート（最優先）
    # -----------------------------------------------------------------
    breakout_pass = bool(
        (breakout_ok or hhhl_ok)
        and (float(cont_best) >= 0.57)
        and (max(float(strength), float(breakout_strength)) >= 0.35)
        and (float(macro_risk) <= 0.90)
    )

    # RANGE 端の逆張り（厳格）
    range_edge_setup = False
    try:
        # イベント直前はレンジ逆張りを避ける（事故回避）。直後捕獲はブレイク専用。
        in_pre = (event_mode == "PRE_EVENT")
        if phase_label == "RANGE" and (not in_pre) and (event_mode not in ("EVENT_WINDOW", "POST_WAIT")):
            near_edge = (range_pos <= 0.25) if direction == "LONG" else (range_pos >= 0.75)
            range_edge_setup = bool(
                near_edge
                and (float(rr) >= 1.40)
                and (float(confidence) >= 0.45)
                and (float(cont_best) >= 0.54)
                and (float(macro_risk) <= 0.85)
                and (ev_gate >= float(dynamic_threshold) - 0.02)
            )
    except Exception:
        range_edge_setup = False

    # 全体の構造妥当性
    structure_ok = True
    if phase_label == "RANGE":
        structure_ok = bool(breakout_pass or range_edge_setup)
    else:
        if (float(strength) < 0.18) and not (breakout_ok or hhhl_ok):
            structure_ok = False

    # POST_BREAKOUTはブレイク根拠必須（取り逃がし防止と事故回避を両立）
    if event_mode == "POST_BREAKOUT":
        structure_ok = bool(breakout_ok or hhhl_ok)

    # -----------------------------------------------------------------
    # 10) veto/decision（veto乱立を抑える）
    # -----------------------------------------------------------------
    veto: List[str] = []
    def _veto(msg: str) -> None:
        s = str(msg or "").strip()
        if not s:
            return
        if s not in veto:
            veto.append(s)

    why = ""
    gate_mode = "raw+mom"

    # mandatory event window block
    if event_guard_enable and event_block_window and event_mode == "EVENT_WINDOW":
        gate_mode = "event_block"
        why = f"高インパクト指標の前後（±{event_window_minutes}分）のため見送り"
        _veto(why)
        decision = "NO_TRADE"
    elif event_guard_enable and event_mode == "POST_WAIT":
        gate_mode = "post_wait"
        why = "高インパクト直後0〜1hは様子見（スプレッド/再反転の不確実性）"
        _veto(why)
        decision = "NO_TRADE"
    elif not structure_ok:
        gate_mode = "structure_veto"
        if phase_label == "RANGE":
            why = "レンジ優勢で構造根拠が不足（ブレイク or 端の逆張り条件が未達）"
        else:
            why = "価格構造の根拠が弱い（トレンド強度/HHHL/ブレイクが不足）"
        _veto(why)
        decision = "NO_TRADE"
    else:
        # EV gate (post-breakout has its own rescue)
        if ev_gate >= float(dynamic_threshold):
            decision = "TRADE"
            why = f"EV通過: {ev_gate:+.3f} ≥ 動的閾値 {float(dynamic_threshold):.3f}"
        elif event_mode == "POST_BREAKOUT" and (ev_gate >= float(dynamic_threshold) - 0.08) and float(confidence) >= 0.42:
            decision = "TRADE"
            gate_mode = "post_breakout_rescue"
            why = f"イベント後捕獲（1〜24hブレイク専用）: EV {ev_gate:+.3f} / 閾値 {float(dynamic_threshold):.3f}（救済）"
        elif breakout_pass and (ev_gate >= float(dynamic_threshold) - 0.04):
            decision = "TRADE"
            gate_mode = "breakout_rescue"
            why = f"BREAKOUT通過: EV {ev_gate:+.3f} / 閾値 {float(dynamic_threshold):.3f}（救済）"
        else:
            decision = "NO_TRADE"
            _veto(f"EV不足: {ev_gate:+.3f} < 動的閾値 {float(dynamic_threshold):.3f}")

    # -----------------------------------------------------------------
    # 11) 状態確率 / EV内訳（UI用）
    # -----------------------------------------------------------------
    s_up = max(0.0, float(p_up) * (0.55 + 0.75 * float(strength)) + max(0.0, float(mom)) * 0.10)
    s_dn = max(0.0, float(p_dn) * (0.55 + 0.75 * float(strength)) + max(0.0, -float(mom)) * 0.10)
    s_range = max(0.0, (1.0 - float(strength)) * 0.95 + 0.05)
    s_risk = max(0.0, float(macro_risk) * 1.15 + (1.0 - float(cont_best)) * 0.10)

    tot = s_up + s_dn + s_range + s_risk
    if tot <= 1e-12:
        state_probs = {"trend_up": 0.25, "trend_down": 0.25, "range": 0.25, "risk_off": 0.25}
    else:
        state_probs = {
            "trend_up": float(s_up / tot),
            "trend_down": float(s_dn / tot),
            "range": float(s_range / tot),
            "risk_off": float(s_risk / tot),
        }

    if direction == "LONG":
        r_up = max(0.2, float(rr) * 0.85)
        r_dn = -1.0
    else:
        r_dn = max(0.2, float(rr) * 0.85)
        r_up = -1.0
    r_range = (0.12 * float(rr) - 0.35)
    r_riskoff = -0.75
    ev_contribs = {
        "trend_up": float(state_probs["trend_up"] * r_up),
        "trend_down": float(state_probs["trend_down"] * r_dn),
        "range": float(state_probs["range"] * r_range),
        "risk_off": float(state_probs["risk_off"] * r_riskoff),
    }

    # -----------------------------------------------------------------
    # 12) 統合スコア（単一ランキング指標）
    #   - 価格構造を最優先（structure_weight大）
    #   - EVは次点
    #   - イベント影響（通貨別）はペナルティとして統合 → 非影響通貨ペアが相対的に上位化
    # -----------------------------------------------------------------
    structure_score = (
        0.60 * float(strength)
        + (0.25 * float(breakout_strength) if bool(breakout_ok) else 0.0)
        + (0.15 if bool(hhhl_ok) else 0.0)
        + 0.10 * float(_clamp(abs(float(mom)), 0.0, 1.0))
    )
    if phase_label == "RANGE":
        structure_score *= 0.85
    structure_scaled = float(_clamp(structure_score, 0.0, 1.2) / 1.2)

    ev_scaled = float(_clamp((ev_gate + 0.15) / 1.35, 0.0, 1.0))

    ef_up = float(ev_meta.get("factor", 0.0) or 0.0)
    event_pen = 0.20 * ef_up
    if event_mode == "PRE_EVENT":
        event_pen = 0.28 * ef_up
    if event_mode == "POST_BREAKOUT":
        event_pen = 0.10 * ef_up

    event_pen += 0.08 * float(weekend_risk or 0.0) + 0.08 * float(weekcross_risk or 0.0)
    macro_pen = 0.08 * float(macro_risk)

    rank_score = float(
        2.00 * structure_scaled
        + 1.40 * ev_scaled
        + 0.40 * float(confidence)
        - float(event_pen)
        - float(macro_pen)
    )
    final_score = rank_score

    # -----------------------------------------------------------------
    # 13) ctx（デバッグ/可視化用）
    # -----------------------------------------------------------------
    ctx_out = {
        "pair": pair,
        "phase_label": phase_label,
        "trend_strength": float(strength),
        "momentum_score": float(mom),
        "range_pos": float(range_pos),
        "range_edge_setup": bool(range_edge_setup),
        "ccy_strength_proxy": float(ccy_strength_proxy),
        "ccy_bonus": float(ccy_bonus),
        "cont_p_up": float(p_up),
        "cont_p_dn": float(p_dn),
        "hh_hl_ok": bool(hhhl_ok),
        "breakout_ok": bool(breakout_ok),
        "breakout_strength": float(breakout_strength),
        "breakout_pass": bool(breakout_pass),
        "rr": float(rr),
        "p_win_model": float(p_win_model),
        "p_eff": float(p_eff),
        "macro_risk_score": float(macro_risk),
        "event_mode": str(event_mode),
        "event_risk_score": float(ev_meta.get("score", 0.0) or 0.0),
        "event_risk_factor": float(ev_meta.get("factor", 0.0) or 0.0),
        "event_window_high": bool(ev_meta.get("window_high", False)),
        "event_next_high_hours": (float(ev_meta.get("next_high_hours")) if ev_meta.get("next_high_hours") is not None else None),
        "event_last_high_hours": (float(ev_meta.get("last_high_hours")) if ev_meta.get("last_high_hours") is not None else None),
        "event_feed_status": str(ev_meta.get("status", "") or ""),
        "event_feed_error": str(ev_meta.get("err", "") or ""),
        "event_upcoming": (ev_meta.get("upcoming", []) or []),
        "event_recent": (ev_meta.get("recent", []) or []),
        "event_impact_ccys": (ev_meta.get("impact_ccys", {}) or {}),
        "weekend_risk": float(weekend_risk),
        "weekcross_risk": float(weekcross_risk or 0.0),
        "weekcross_weekday": (int(weekcross_weekday) if weekcross_weekday is not None else None),
        "mom_bonus": float(mom_bonus),
        "dynamic_threshold": float(dynamic_threshold),
        "dynamic_threshold_base": float(base_thr),
        "dynamic_threshold_mult": float(thr_mult),
        "ev_gate": float(ev_gate),
        "structure_scaled": float(structure_scaled),
        "ev_scaled": float(ev_scaled),
        "rank_score": float(rank_score),
        "event_penalty": float(event_pen),
        "macro_penalty": float(macro_pen),
        "len": int(len(df)),
    }

    # -----------------------------------------------------------------
    # 14) 注文方式の提案（直前:成行禁止 / 直後:ブレイク専用）
    # -----------------------------------------------------------------
    order_type = "MARKET"
    entry_type = "MARKET_NOW"
    exec_guard_notes: List[str] = []

    # setup-based suggestion (even for NO_TRADE; UI上は参考として表示可能)
    setup_kind = "TREND"
    if phase_label == "RANGE" and bool(range_edge_setup):
        setup_kind = "RANGE_EDGE"
    elif bool(breakout_pass) or str(phase).startswith("BREAKOUT") or (event_mode == "POST_BREAKOUT"):
        setup_kind = "BREAKOUT"

    try:
        pip = _pip_size(pair)
        atr_for_entry = max(float(atr14), float(pip) * 10.0)
    except Exception:
        pip = 0.01
        atr_for_entry = float(atr14)

    # Base reco by setup
    if setup_kind == "RANGE_EDGE":
        if direction == "LONG":
            new_entry = entry - 0.25 * atr_for_entry
        else:
            new_entry = entry + 0.25 * atr_for_entry
        order_type = "LIMIT"
        entry_type = "LIMIT_PULLBACK"
        exec_guard_notes.append("レンジ端のため、押し目/戻りの指値を推奨")
        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

    elif setup_kind == "BREAKOUT":
        if direction == "LONG":
            new_entry = entry + 0.10 * atr_for_entry
            entry_type = "STOP_BREAKOUT"
        else:
            new_entry = entry - 0.10 * atr_for_entry
            entry_type = "STOP_BREAKDOWN"
        order_type = "STOP"
        exec_guard_notes.append("ブレイク捕獲のため、逆指値（STOP）を推奨")
        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

    # High-impact is close enough → ban MARKET entry (直前:成行禁止)
    event_market_ban_active = False
    event_market_ban_hours = float(ctx_in.get("event_market_ban_hours", 12.0) or 12.0)
    if float(weekcross_risk or 0.0) > 0.0:
        event_market_ban_hours = max(event_market_ban_hours, float(ctx_in.get("weekcross_market_ban_hours", 18.0) or 18.0))
    try:
        nh = ev_meta.get("next_high_hours", None)
        if (nh is not None) and (float(nh) <= float(event_market_ban_hours)):
            event_market_ban_active = True
    except Exception:
        pass

    if decision == "TRADE" and bool(event_market_ban_active) and order_type == "MARKET":
        # If we still ended up MARKET, convert to pending
        try:
            nh = float(ev_meta.get("next_high_hours") or 0.0)
        except Exception:
            nh = None
        if phase_label == "RANGE" and (not breakout_pass):
            # pullback limit
            if direction == "LONG":
                new_entry = entry - 0.25 * atr_for_entry
            else:
                new_entry = entry + 0.25 * atr_for_entry
            order_type = "LIMIT"
            entry_type = "LIMIT_PULLBACK"
            msg = f"高インパクト指標まで{nh:.1f}hのため成行禁止 → 押し目/戻りの指値を提案"
        else:
            # breakout stop
            if direction == "LONG":
                new_entry = entry + 0.10 * atr_for_entry
                entry_type = "STOP_BREAKOUT"
            else:
                new_entry = entry - 0.10 * atr_for_entry
                entry_type = "STOP_BREAKDOWN"
            order_type = "STOP"
            msg = f"高インパクト指標まで{nh:.1f}hのため成行禁止 → ブレイク逆指値を提案"

        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

        exec_guard_notes.append(msg)
        if why:
            why = why + " / " + msg
        else:
            why = msg

    # lot shrink factors (UI/logging)
    try:
        ef = float(ctx_out.get("event_risk_factor", 0.0) or 0.0)
        ctx_out["event_market_ban_active"] = bool(event_market_ban_active)
        ctx_out["event_market_ban_hours"] = float(event_market_ban_hours)
        ctx_out["exec_guard_notes"] = list(exec_guard_notes)
        ctx_out["order_type_reco"] = str(order_type)
        ctx_out["entry_type_reco"] = str(entry_type)
        ctx_out["lot_shrink_event_factor"] = float(_clamp(1.0 - 0.60 * ef, 0.20, 1.00))
        ctx_out["lot_shrink_weekcross_factor"] = (0.75 if float(weekcross_risk or 0.0) > 0.0 else 1.0)
        ctx_out["lot_shrink_weekend_factor"] = (0.60 if float(weekend_risk or 0.0) > 0.0 else 1.0)
    except Exception:
        pass

    # -----------------------------------------------------------------
    # 15) 保有中のイベント接近対応（縮退/一部利確/建値移動/追加禁止）
    # -----------------------------------------------------------------
    hold_manage = _hold_manage_reco(
        pair=str(pair),
        df=df,
        ctx_in=ctx_in,
        plan_like={"side": side, "entry": entry, "sl": sl, "tp": tp},
        ev_meta=(ev_meta or {}),
        weekend_risk=float(weekend_risk or 0.0),
        weekcross_risk=float(weekcross_risk or 0.0),
    )
    if isinstance(hold_manage, dict) and hold_manage:
        try:
            ctx_out["hold_manage"] = hold_manage
        except Exception:
            pass

    # Trail SL: エントリーから0.5R戻し（見せ方用）
    trail_sl = sl
    try:
        dist_sl = abs(entry - sl)
        if dist_sl > 0:
            trail_sl = entry - 0.5 * dist_sl if direction == "LONG" else (entry + 0.5 * dist_sl)
    except Exception:
        trail_sl = sl

    # 返却（main互換キー）
    plan = {
        "decision": str(decision),
        "direction": str(direction),
        "side": str(side),

        "order_type": str(order_type),
        "entry_type": str(entry_type),

        "entry": float(entry),
        "entry_price": float(entry),
        "sl": float(sl),
        "stop_loss": float(sl),
        "tp": float(tp),
        "take_profit": float(tp),

        "trail_sl": float(trail_sl),
        "extend_factor": 1.0,

        "ev_raw": float(ev_raw),
        "ev_adj": float(ev_adj),

        "expected_R_ev_raw": float(ev_raw),
        "expected_R_ev_adj": float(ev_adj),
        "expected_R_ev": float(ev_gate),

        "rank_score": float(rank_score),
        "final_score": float(final_score),

        "dynamic_threshold": float(dynamic_threshold),
        "gate_mode": str(gate_mode),

        "confidence": float(confidence),
        "p_win": float(p_eff),       # UIには縮退後を提示（整合性を優先）
        "p_eff": float(p_eff),
        "p_win_ev": float(p_eff),

        "event_mode": str(event_mode),
        "event_next_high_hours": (float(next_high) if next_high is not None else None),
        "event_last_high_hours": (float(last_high) if last_high is not None else None),

        "why": str(why),
        "veto": list(veto),
        "veto_reasons": list(veto),

        "state_probs": state_probs,
        "ev_contribs": ev_contribs,

        "hold_manage": (hold_manage if isinstance(hold_manage, dict) else {}),
        "_ctx": ctx_out,
    }
    return plan

# End of file
