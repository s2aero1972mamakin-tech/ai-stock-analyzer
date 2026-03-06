

# main.py (v4 integrated, keeps v3 features + adds global risk overlays)
from __future__ import annotations

import os
import time
import csv
import uuid
import math
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

import streamlit as st

# --- Streamlit top-down guard: ensure pair_label exists before any UI blocks ---
try:
    if 'pair_label' not in st.session_state:
        st.session_state['pair_label'] = 'USD/JPY'
except Exception:
    # session_state may not be ready in some edge cases; ignore
    pass
pair_label = None
try:
    pair_label = st.session_state.get('pair_label', 'USD/JPY')
except Exception:
    pair_label = 'USD/JPY'

import requests
import pandas as pd

# ---- JP display helpers (SBI運用向け) ----
def _order_type_jp(order_type: str) -> str:
    t = (order_type or "").strip().upper()
    mp = {"MARKET": "成行", "LIMIT": "指値", "STOP": "逆指値"}
    return mp.get(t, t or "—")

def _entry_type_jp(entry_type: str) -> str:
    t = (entry_type or "").strip().upper()
    mp = {
        "MARKET_NOW": "成行（即時）",
        "MARKET": "成行（即時）",
        "LIMIT_PULLBACK": "指値（押し目/戻り）",
        "LIMIT": "指値",
        "STOP_BREAKOUT": "逆指値（ブレイク）",
        "STOP_BREAKDOWN": "逆指値（ブレイクダウン）",
        "STOP": "逆指値",
    }
    return mp.get(t, t or "—")

# ---- indicators compat (古いlogicでも落とさない) ----
def _compute_indicators_fallback(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        if df is None or df.empty or len(df) < 5:
            return {"ema20": 0.0, "ema50": 0.0, "atr14": 0.0, "adx14": 0.0}
        d = df.copy()
        for c in ["Open", "High", "Low", "Close"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["High", "Low", "Close"])
        close = d["Close"].astype(float)

        # EMA
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

        # ATR(14)
        high = d["High"].astype(float)
        low = d["Low"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else float(tr.mean())

        # ADX(14) (簡易・軽量版)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        atr = tr.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / atr.replace(0, pd.NA))
        minus_di = 100 * (minus_dm.rolling(14).sum() / atr.replace(0, pd.NA))
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA))
        adx14 = dx.rolling(14).mean().iloc[-1] if len(dx) >= 28 else float(dx.dropna().mean() if dx.notna().any() else 0.0)

        return {"ema20": float(ema20), "ema50": float(ema50), "atr14": float(atr14 or 0.0), "adx14": float(adx14 or 0.0)}
    except Exception:
        return {"ema20": 0.0, "ema50": 0.0, "atr14": 0.0, "adx14": 0.0}

def _compute_indicators_compat(df: pd.DataFrame) -> Dict[str, Any]:
    fn = getattr(logic, "compute_indicators", None)
    if callable(fn):
        try:
            out = fn(df)
            if isinstance(out, dict):
                # ensure keys exist
                return {
                    "ema20": float(out.get("ema20", 0.0) or 0.0),
                    "ema50": float(out.get("ema50", 0.0) or 0.0),
                    "atr14": float(out.get("atr14", 0.0) or 0.0),
                    "adx14": float(out.get("adx14", 0.0) or 0.0),
                }
        except Exception:
            pass
    return _compute_indicators_fallback(df)


# ===== RL core (WFA + Q-learning) definitions (must appear before any usage) =====
def _phase_to_id(phase: str) -> int:
    p = str(phase or "").upper()
    if p.startswith("UP"):
        return 1
    if p.startswith("DOWN"):
        return 2
    if p.startswith("BREAK"):
        return 3
    if p.startswith("RANGE"):
        return 4
    return 0

def _bin(x: float, edges: list[float]) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)

class RLExitAgent:
    """Tabular Q-learning exit agent (real RL).

    - States are discretized tuples derived from (phase, trend_strength, edge, unrealized_R, drawdown_R).
    - Actions: HOLD, TRAIL_TIGHT, EXIT, TAKE_PARTIAL (partial treated as EXIT with bonus factor in reward proxy).
    - Learns from historical price paths using a lightweight simulated environment.
    """

    ACTIONS = ["HOLD", "TRAIL_TIGHT", "TAKE_PARTIAL", "EXIT"]

    def __init__(self):
        self.q: dict[str, list[float]] = {}  # state_key -> q-values per action index
        # bins (tunable)
        self.r_edges = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.2, 1.8, 2.5]
        self.dd_edges = [-2.5, -1.8, -1.2, -0.8, -0.5, -0.2, 0.0]
        self.str_edges = [0.15, 0.3, 0.5, 0.7]
        self.edge_edges = [-0.2, -0.1, -0.03, 0.03, 0.1, 0.2]

    def _state_key(self, phase: str, trend_strength: float, edge: float, unreal_R: float, dd_R: float) -> str:
        pid = _phase_to_id(phase)
        sb = _bin(trend_strength, self.str_edges)
        eb = _bin(edge, self.edge_edges)
        rb = _bin(unreal_R, self.r_edges)
        db = _bin(dd_R, self.dd_edges)
        return f"{pid}|{sb}|{eb}|{rb}|{db}"

    def _get_q(self, key: str) -> list[float]:
        if key not in self.q:
            self.q[key] = [0.0 for _ in self.ACTIONS]
        return self.q[key]

    def act(self, phase: str, trend_strength: float, p_up: float, p_dn: float, unrealized_R: float, dd_R: float = 0.0) -> dict:
        p_up = 0.5 if (p_up != p_up) else float(p_up)
        p_dn = 0.5 if (p_dn != p_dn) else float(p_dn)
        edge = float(p_up - p_dn)
        key = self._state_key(phase, trend_strength, edge, float(unrealized_R), float(dd_R))
        qv = self._get_q(key)
        best_i = int(max(range(len(qv)), key=lambda i: qv[i]))
        return {"action": self.ACTIONS[best_i], "q": {a: qv[i] for i, a in enumerate(self.ACTIONS)}, "edge": edge, "state": key}

    def to_json(self) -> dict:
        return {"q": self.q}

    @classmethod
    def from_json(cls, obj: dict) -> "RLExitAgent":
        ag = cls()
        q = obj.get("q", {}) if isinstance(obj, dict) else {}
        if isinstance(q, dict):
            # ensure lists
            ag.q = {k: list(v) for k, v in q.items() if isinstance(v, (list, tuple)) and len(v) == len(ag.ACTIONS)}
        return ag

    def train_from_price(self,
                         df: pd.DataFrame,
                         phase_series: pd.Series,
                         trend_strength_series: pd.Series,
                         p_up_series: pd.Series,
                         p_dn_series: pd.Series,
                         episodes: int = 3000,
                         max_horizon: int = 20,
                         atr_mult_stop: float = 1.5,
                         gamma: float = 0.97,
                         alpha: float = 0.12,
                         eps: float = 0.15,
                         dd_penalty: float = 0.15,
                         time_penalty: float = 0.01,
                         seed: int = 7) -> dict:
        """Real Q-learning training on a simple simulated trade-exit environment."""
        if df is None or df.empty:
            return {"ok": False, "error": "empty_df"}
        d = df.copy()
        d = _coerce_ohlc(d)
        if d.empty or len(d) < 200:
            return {"ok": False, "error": "not_enough_bars"}

        c = d["Close"].astype(float)
        atr = _atr(d, 14).astype(float).fillna(method="bfill").fillna(method="ffill")
        n = len(d)
        rng = np.random.default_rng(seed)

        # align aux series
        ph = phase_series.reindex(d.index).fillna("UNKNOWN").astype(str)
        ts = trend_strength_series.reindex(d.index).fillna(0.0).astype(float)
        pu = p_up_series.reindex(d.index).fillna(0.5).astype(float)
        pdn = p_dn_series.reindex(d.index).fillna(0.5).astype(float)

        # choose random entry points away from the end
        valid_start = np.arange(50, n - (max_horizon + 2))
        if len(valid_start) <= 0:
            return {"ok": False, "error": "no_valid_start"}

        total_steps = 0
        for ep in range(int(episodes)):
            t0 = int(rng.choice(valid_start))
            entry = float(c.iloc[t0])
            stop = entry - float(atr.iloc[t0]) * float(atr_mult_stop)  # long-only exit training proxy
            risk = abs(entry - stop)
            if risk <= 1e-9:
                continue

            # simulate forward
            dd_R = 0.0
            unreal_R = 0.0
            done = False
            t = t0
            # initial state
            edge0 = float(pu.iloc[t] - pdn.iloc[t])
            s_key = self._state_key(ph.iloc[t], float(ts.iloc[t]), edge0, unreal_R, dd_R)

            for step in range(int(max_horizon)):
                t1 = t + 1
                price = float(c.iloc[t1])
                unreal_R = (price - entry) / risk
                dd_R = min(dd_R, unreal_R)  # adverse excursion in R (for long proxy)

                # epsilon-greedy
                qv = self._get_q(s_key)
                if rng.random() < eps:
                    a_i = int(rng.integers(0, len(self.ACTIONS)))
                else:
                    a_i = int(max(range(len(qv)), key=lambda i: qv[i]))
                action = self.ACTIONS[a_i]

                # environment transition + reward
                reward = 0.0
                # trail tight: move stop up (reduce risk), but increases stopout probability (proxied)
                if action == "TRAIL_TIGHT":
                    # tighten by 25% of current risk, but cap to entry (no negative risk)
                    new_stop = stop + 0.25 * (price - stop)
                    new_stop = min(new_stop, entry - 1e-9)
                    if new_stop < stop:
                        stop = new_stop
                        risk = abs(entry - stop)
                        if risk <= 1e-9:
                            risk = abs(entry - (entry - 1e-6))
                            stop = entry - 1e-6
                # exit / take partial ends episode
                if action in ("EXIT", "TAKE_PARTIAL"):
                    realized_R = unreal_R
                    if action == "TAKE_PARTIAL":
                        realized_R = realized_R * 0.75  # partial capture proxy (safer, less reward)
                    reward = realized_R - dd_penalty * abs(min(0.0, dd_R)) - time_penalty * step
                    done = True

                # stopout
                if not done and float(d["Low"].iloc[t1]) <= stop:
                    realized_R = (stop - entry) / risk if risk > 1e-9 else -1.0
                    reward = realized_R - dd_penalty * abs(min(0.0, dd_R)) - time_penalty * step
                    done = True

                # next state
                edge1 = float(pu.iloc[t1] - pdn.iloc[t1])
                s_next = self._state_key(ph.iloc[t1], float(ts.iloc[t1]), edge1, unreal_R, dd_R)

                # Q update
                q_next = self._get_q(s_next)
                td_target = reward if done else (0.0 + gamma * max(q_next))
                qv[a_i] = float(qv[a_i] + alpha * (td_target - qv[a_i]))

                total_steps += 1
                s_key = s_next
                t = t1
                if done:
                    break

        return {"ok": True, "episodes": int(episodes), "steps": int(total_steps), "states": int(len(self.q))}

    def evaluate_policy(self,
                        df: pd.DataFrame,
                        phase_series: pd.Series,
                        trend_strength_series: pd.Series,
                        p_up_series: pd.Series,
                        p_dn_series: pd.Series,
                        episodes: int = 800,
                        max_horizon: int = 20,
                        atr_mult_stop: float = 1.5,
                        dd_penalty: float = 0.15,
                        time_penalty: float = 0.01,
                        seed: int = 11) -> dict:
        """Evaluate current Q-policy (greedy) on the same simulated environment used in training.
        Returns summary stats in R units.
        """
        try:
            if df is None or df.empty:
                return {"ok": False, "error": "empty_df"}
            d = df.copy()
            if "Close" not in d.columns or "Low" not in d.columns:
                return {"ok": False, "error": "missing_ohlc"}
            c = pd.to_numeric(d["Close"], errors="coerce").fillna(method="ffill").fillna(method="bfill")
            low = pd.to_numeric(d["Low"], errors="coerce").fillna(method="ffill").fillna(method="bfill")
            atr = _atr(d, 14).fillna(method="ffill").fillna(method="bfill")
            ph = phase_series.reindex(d.index).fillna("RANGE").astype(str)
            ts = pd.to_numeric(trend_strength_series.reindex(d.index), errors="coerce").fillna(0.0)
            pu = pd.to_numeric(p_up_series.reindex(d.index), errors="coerce").fillna(0.5)
            pdn = pd.to_numeric(p_dn_series.reindex(d.index), errors="coerce").fillna(0.5)

            rng = np.random.default_rng(int(seed))
            valid_start = np.arange(1, max(2, len(d) - int(max_horizon) - 2))
            if len(valid_start) < 10:
                return {"ok": False, "error": "not_enough_bars"}
            ep_R = []
            ep_reward = []
            win = 0
            for _ in range(int(episodes)):
                t0 = int(rng.choice(valid_start))
                entry = float(c.iloc[t0])
                stop = entry - float(atr.iloc[t0]) * float(atr_mult_stop)
                risk = abs(entry - stop)
                if risk <= 1e-9:
                    continue
                dd_R = 0.0
                unreal_R = 0.0
                t = t0
                edge0 = float(pu.iloc[t] - pdn.iloc[t])
                s_key = self._state_key(ph.iloc[t], float(ts.iloc[t]), edge0, unreal_R, dd_R)

                realized_R = 0.0
                reward = 0.0
                done = False
                for step in range(int(max_horizon)):
                    t1 = t + 1
                    price = float(c.iloc[t1])
                    unreal_R = (price - entry) / risk
                    dd_R = min(dd_R, unreal_R)

                    qv = self._get_q(s_key)
                    a_i = int(max(range(len(qv)), key=lambda i: qv[i]))
                    action = self.ACTIONS[a_i]

                    # trail
                    if action == "TRAIL_TIGHT":
                        new_stop = stop + 0.25 * (price - stop)
                        new_stop = min(new_stop, entry - 1e-9)
                        if new_stop < stop:
                            stop = new_stop
                            risk = abs(entry - stop)
                            if risk <= 1e-9:
                                risk = abs(entry - (entry - 1e-6))
                                stop = entry - 1e-6

                    if action in ("EXIT", "TAKE_PARTIAL"):
                        realized_R = unreal_R
                        if action == "TAKE_PARTIAL":
                            realized_R = realized_R * 0.75
                        reward = realized_R - float(dd_penalty) * abs(min(0.0, dd_R)) - float(time_penalty) * step
                        done = True

                    if not done and float(low.iloc[t1]) <= stop:
                        realized_R = (stop - entry) / risk if risk > 1e-9 else -1.0
                        reward = realized_R - float(dd_penalty) * abs(min(0.0, dd_R)) - float(time_penalty) * step
                        done = True

                    edge1 = float(pu.iloc[t1] - pdn.iloc[t1])
                    s_next = self._state_key(ph.iloc[t1], float(ts.iloc[t1]), edge1, unreal_R, dd_R)

                    s_key = s_next
                    t = t1
                    if done:
                        break

                ep_R.append(float(realized_R))
                ep_reward.append(float(reward))
                if realized_R > 0:
                    win += 1

            if not ep_R:
                return {"ok": False, "error": "no_episodes"}
            arrR = np.array(ep_R, dtype=float)
            arrRew = np.array(ep_reward, dtype=float)
            return {
                "ok": True,
                "episodes": int(len(arrR)),
                "avg_R": float(arrR.mean()),
                "win_rate": float((arrR > 0).mean()),
                "pf": float(arrR[arrR > 0].sum() / (abs(arrR[arrR < 0].sum()) + 1e-9)),
                "avg_reward": float(arrRew.mean()),
                "p05_R": float(np.quantile(arrR, 0.05)),
                "p95_R": float(np.quantile(arrR, 0.95)),
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}



# ===== end RL core defs =====


# ---- optional deps ----
try:
    import yfinance as yf
except Exception:
    yf = None

# ---- local modules ----
import logic

# Integrated external features
try:
    import data_layer
except Exception:
    try:
        import data_layer0225 as data_layer  # fallback for local filename
    except Exception:
        data_layer = None

# yfinance rate-limit exception (version dependent)
try:
    from yfinance.exceptions import YFRateLimitError  # type: ignore
except Exception:
    class YFRateLimitError(Exception):
        pass



# =========================
# Early definitions (avoid NameError on Streamlit top-to-bottom execution)
# =========================
def _normalize_pair_label(s: str) -> str:
    s = (s or "").strip().upper().replace(" ", "")
    s = s.replace("-", "/")
    if "/" not in s and len(s) == 6:
        s = s[:3] + "/" + s[3:]
    return s


def _pair_label_to_symbol(pair_label: str) -> str:
    pl = _normalize_pair_label(pair_label)
    mp = getattr(logic, "PAIR_MAP", None)
    if isinstance(mp, dict) and pl in mp:
        return mp[pl]
    fallback = {
        "USD/JPY": "JPY=X",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "AUD/USD": "AUDUSD=X",
        "EUR/JPY": "EURJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "AUD/JPY": "AUDJPY=X",

        "EUR/GBP": "EURGBP=X",

        "AUD/NZD": "AUDNZD=X",

        "EUR/CHF": "EURCHF=X",

        "GBP/AUD": "GBPAUD=X",
    }
    return fallback.get(pl, "JPY=X")


def _pair_label_to_stooq_symbol(pair_label: str) -> Optional[str]:
    pl = _normalize_pair_label(pair_label)
    mapping = {
        "USD/JPY": "usdjpy",
        "EUR/USD": "eurusd",
        "GBP/USD": "gbpusd",
        "AUD/USD": "audusd",
        "EUR/JPY": "eurjpy",
        "GBP/JPY": "gbpjpy",
        "AUD/JPY": "audjpy",

        "EUR/GBP": "eurgbp",

        "AUD/NZD": "audnzd",

        "EUR/CHF": "eurchf",

        "GBP/AUD": "gbpaud",
    }
    return mapping.get(pl)


def _coerce_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    need = ["Open", "High", "Low", "Close"]
    for c in need:
        if c not in d.columns:
            return pd.DataFrame()
    d = d[need].dropna()
    return d


# =========================
# AI Trend Engine (phase classification, continuation probability, similar pattern search, RL-like exit manager)
# =========================
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception:
    LogisticRegression = None
    StandardScaler = None
    Pipeline = None


def _fetch_from_stooq(pair_label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"source": "stooq", "ok": False, "error": None, "interval_used": "1d"}
    sym = _pair_label_to_stooq_symbol(pair_label)
    if not sym:
        meta["error"] = "unsupported_pair_for_stooq"
        return pd.DataFrame(), meta
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        d = pd.read_csv(url)
        if "Date" not in d.columns:
            meta["error"] = "bad_csv"
            return pd.DataFrame(), meta
        d["Date"] = pd.to_datetime(d["Date"])
        d = d.set_index("Date").sort_index()
        d = _coerce_ohlc(d)
        if d.empty:
            meta["error"] = "empty_after_parse"
            return pd.DataFrame(), meta
        meta["ok"] = True
        return d, meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return pd.DataFrame(), meta

@st.cache_data(ttl=60 * 60)  # 1 hour

def fetch_price_history(pair_label: str, symbol: str, period: str, interval: str, prefer_stooq: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust price fetch:
      - prefer stooq for daily (reduces yfinance rate-limit on Streamlit Cloud)
      - else try yfinance, then fallback to stooq daily
    """
    meta: Dict[str, Any] = {"source": "yfinance", "ok": False, "error": None, "fallback": None, "interval_used": interval}

    # Prefer stooq for daily / multi-scan
    if prefer_stooq or interval == "1d":
        df_s, m_s = _fetch_from_stooq(pair_label)
        if not df_s.empty and m_s.get("ok"):
            meta.update({"source": "stooq", "ok": True, "fallback": None, "interval_used": "1d"})
            return df_s, meta
        meta["fallback"] = m_s

    if yf is not None:
        last_err = None
        for attempt in range(2):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                df = _coerce_ohlc(df)
                if df.empty:
                    last_err = "empty_df"
                    raise RuntimeError("empty_df")
                meta["ok"] = True
                return df, meta
            except YFRateLimitError:
                last_err = "YFRateLimitError"
                break
            except Exception as e:
                last_err = f"{type(e).__name__}:{e}"
                time.sleep(0.6 * (attempt + 1))
        meta["error"] = last_err
    else:
        meta["error"] = "yfinance_not_installed"

    # fallback stooq
    df2, m2 = _fetch_from_stooq(pair_label)
    meta["fallback"] = m2
    if not df2.empty and m2.get("ok"):
        meta["source"] = "stooq"
        meta["ok"] = True
        meta["interval_used"] = "1d"
        return df2, meta

    return pd.DataFrame(), meta

@st.cache_data(ttl=60 * 20)

def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(df)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0)


def _compute_phase_strength_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Vectorized phase + strength per bar (no ML, deterministic).

    Phase rules (approx):
      - UP: ema20>ema50 and adx high
      - DOWN: ema20<ema50 and adx high
      - RANGE: adx low
      - TRANSITION: otherwise
    Strength: normalized ADX (0..1) blended with ema slope.
    """
    d = _coerce_ohlc(df.copy())
    if d.empty:
        return pd.Series(dtype=str), pd.Series(dtype=float)
    c = d["Close"].astype(float)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    adx = _adx(d, 14).astype(float)
    adx_n = (adx / 40.0).clip(0.0, 1.0).fillna(0.0)
    slope = (ema20 - ema20.shift(5)) / (d["ATR14"].astype(float).replace(0, np.nan) if "ATR14" in d.columns else (c.rolling(14).std() + 1e-9))
    slope = slope.fillna(0.0).clip(-2.0, 2.0)
    slope_n = (slope.abs() / 2.0).clip(0.0, 1.0)
    strength = (0.65 * adx_n + 0.35 * slope_n).clip(0.0, 1.0)

    phase = pd.Series(index=d.index, dtype=str)
    is_range = adx.fillna(0.0) < 16.0
    is_up = (ema20 > ema50) & (~is_range)
    is_dn = (ema20 < ema50) & (~is_range)
    phase[is_range] = "RANGE"
    phase[is_up] = "UP_TREND"
    phase[is_dn] = "DOWN_TREND"
    phase[phase.isna()] = "TRANSITION"
    return phase, strength


def _compute_cont_p_series(df: pd.DataFrame, horizon: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Cheap continuation probability proxy series based on momentum & volatility.
    This is used only as a feature for RL training; the 'official' continuation model remains separate.
    """
    d = _coerce_ohlc(df.copy())
    if d.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    c = d["Close"].astype(float)
    r = c.pct_change().fillna(0.0)
    mom = (c - c.shift(horizon)) / (c.rolling(14).std() + 1e-9)
    mom = mom.fillna(0.0).clip(-3.0, 3.0)
    # sigmoid to (0..1)
    p_up = 1.0 / (1.0 + np.exp(-1.2 * mom))
    p_dn = 1.0 - p_up
    return pd.Series(p_up, index=d.index), pd.Series(p_dn, index=d.index)


def _rl_exit_reco(agent: Optional[RLExitAgent], phase: str, trend_strength: float, p_up: float, p_dn: float, unrealized_R: float, dd_R: float = 0.0) -> Dict[str, Any]:
    if agent is None:
        return {"action": "HOLD", "note": "RL model not trained yet", "q": {}, "edge": float(p_up - p_dn), "state": ""}
    return agent.act(phase, trend_strength, p_up, p_dn, unrealized_R, dd_R)


def _load_rl_agent(path: str) -> Optional[RLExitAgent]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return RLExitAgent.from_json(obj)
    except Exception:
        return None
    return None


def _save_rl_agent(path: str, agent: RLExitAgent) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(agent.to_json(), f, ensure_ascii=False)
        return True
    except Exception:
        return False



def _wfa_select_rl_coeffs(df: pd.DataFrame,
                          phase_series: pd.Series,
                          trend_strength_series: pd.Series,
                          p_up_series: pd.Series,
                          p_dn_series: pd.Series,
                          grid: Optional[list[dict]] = None,
                          splits: int = 3,
                          train_episodes: int = 2200,
                          eval_episodes: int = 800,
                          max_horizon: int = 20,
                          seed: int = 13) -> dict:
    """Walk-Forward Analysis (WFA) to pick RL reward coefficients.
    Returns best params and fold stats. Keeps it lightweight for Streamlit Cloud.
    """
    try:
        if df is None or df.empty or len(df) < 600:
            return {"ok": False, "error": "not_enough_data"}
        if grid is None:
            grid = []
            for dd in [0.05, 0.10, 0.15, 0.25, 0.35]:
                for tp in [0.0, 0.005, 0.01, 0.02]:
                    for atrm in [1.2, 1.5, 1.8]:
                        grid.append({"dd_penalty": dd, "time_penalty": tp, "atr_mult_stop": atrm})
        # build fold indices (expanding window)
        n = len(df)
        fold = max(120, n // (splits + 2))
        # train ends at: fold*(i+2), test next fold
        folds = []
        for i in range(splits):
            tr_end = fold * (i + 2)
            te_end = min(n, tr_end + fold)
            if te_end - tr_end < 60 or tr_end < 200:
                continue
            folds.append((0, tr_end, tr_end, te_end))
        if not folds:
            # fallback single split 70/30
            tr_end = int(n * 0.7)
            folds = [(0, tr_end, tr_end, n)]

        best = None
        all_rows = []
        for params in grid:
            fold_scores = []
            for (tr0, tr1, te0, te1) in folds:
                df_tr = df.iloc[tr0:tr1].copy()
                df_te = df.iloc[te0:te1].copy()
                ph_tr = phase_series.iloc[tr0:tr1]
                ts_tr = trend_strength_series.iloc[tr0:tr1]
                pu_tr = p_up_series.iloc[tr0:tr1]
                pd_tr = p_dn_series.iloc[tr0:tr1]

                ph_te = phase_series.iloc[te0:te1]
                ts_te = trend_strength_series.iloc[te0:te1]
                pu_te = p_up_series.iloc[te0:te1]
                pd_te = p_dn_series.iloc[te0:te1]

                ag = RLExitAgent()
                r = ag.train_from_price(
                    df_tr, ph_tr, ts_tr, pu_tr, pd_tr,
                    episodes=int(train_episodes),
                    max_horizon=int(max_horizon),
                    atr_mult_stop=float(params["atr_mult_stop"]),
                    dd_penalty=float(params["dd_penalty"]),
                    time_penalty=float(params["time_penalty"]),
                    seed=int(seed + 17),
                )
                if not r.get("ok"):
                    continue
                ev = ag.evaluate_policy(
                    df_te, ph_te, ts_te, pu_te, pd_te,
                    episodes=int(eval_episodes),
                    max_horizon=int(max_horizon),
                    atr_mult_stop=float(params["atr_mult_stop"]),
                    dd_penalty=float(params["dd_penalty"]),
                    time_penalty=float(params["time_penalty"]),
                    seed=int(seed + 19),
                )
                if not ev.get("ok"):
                    continue
                fold_scores.append(float(ev["avg_reward"]))
            if not fold_scores:
                continue
            score = float(np.mean(fold_scores))
            row = {"score": score, **params}
            all_rows.append(row)
            if (best is None) or (score > best["score"]):
                best = row

        if best is None:
            return {"ok": False, "error": "wfa_failed", "tried": int(len(grid))}
        df_rank = pd.DataFrame(all_rows).sort_values("score", ascending=False).head(12)
        return {"ok": True, "best": best, "top": df_rank.to_dict(orient="records"), "folds": folds, "tried": int(len(grid))}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}



def _dominant_state(state_probs: Dict[str, Any]) -> str:
    if not isinstance(state_probs, dict) or not state_probs:
        return "—"
    try:
        return max(state_probs.items(), key=lambda kv: float(kv[1]))[0]
    except Exception:
        return "—"



# =========================
# Utilities
# =========================
PAIR_LIST_CORE = [
    "EUR/USD",
    "GBP/USD",
    "AUD/USD",
    "EUR/JPY",
    "GBP/JPY",
    "AUD/JPY",
]
# 追加ペア（必要なときだけ有効化）
PAIR_LIST_EXTRA = [
    "EUR/GBP",
    "AUD/NZD",
    "EUR/CHF",
    "GBP/AUD",
]
PAIR_LIST_ALL = PAIR_LIST_CORE + PAIR_LIST_EXTRA

# UIのデフォルト選択（= core）
PAIR_LIST_DEFAULT = list(PAIR_LIST_CORE)

# =========================
# Build / Diagnostics
# =========================
APP_BUILD = "fixed28b_20260224"
# ---- EV audit (operator logs) ----
EV_AUDIT_PATH = "logs/ev_audit.csv"

def _ev_audit_append(row: Dict[str, Any], path: str = EV_AUDIT_PATH) -> None:
    """Append one row to ev_audit.csv. Never raises to caller."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = [
            "ts_utc",
            "timeframe_mode",
            "pair",
            "decision_adj",
            "decision_raw",
            "killed_by_adj",
            "ev_raw",
            "ev_adj",
            "dynamic_threshold",
            "dominant_state",
            "confidence",
            "global_risk_index",
            "war_probability",
            "financial_stress",
            "macro_risk_score",
            "risk_off_bump",
        ]
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception:
        # Audit must never break trading UI
        return

# =========================
# External sinks (Webhook / Supabase) for ops logging
# =========================
def _try_post_json(url: str, payload: dict, timeout_s: int = 6):
    """POST JSON and return (ok: bool, status_code: int|None, response_text: str, error: str)."""
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        return (200 <= getattr(r, "status_code", 0) < 300), getattr(r, "status_code", None), (r.text or ""), ""
    except Exception as e:
        return False, None, "", str(e)

def _post_json_webhook(payload: dict):
    """Optional external sink: POST JSON to LOG_WEBHOOK_URL if provided in Streamlit Secrets.
    Returns (ok, status_code, response_text, error). Never raises."""
    try:
        url = st.secrets.get("LOG_WEBHOOK_URL", "")
        if not url:
            return False, None, "", "LOG_WEBHOOK_URL not set"
        return _try_post_json(url, payload, timeout_s=6)
    except Exception as e:
        return False, None, "", str(e)

def _supabase_insert(table: str, row: dict):
    """Optional Supabase sink (REST): requires SUPABASE_URL and SUPABASE_ANON_KEY in secrets.
    Returns (ok, status_code, response_text, error). Never raises."""
    try:
        sb_url = st.secrets.get("SUPABASE_URL", "")
        sb_key = st.secrets.get("SUPABASE_ANON_KEY", "")
        if not sb_url or not sb_key or not table:
            return False, None, "", "Supabase secrets/table not set"
        endpoint = sb_url.rstrip("/") + f"/rest/v1/{table}"
        headers = {
            "apikey": sb_key,
            "Authorization": f"Bearer {sb_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        r = requests.post(endpoint, headers=headers, json=[row], timeout=6)
        ok = 200 <= getattr(r, "status_code", 0) < 300
        return ok, getattr(r, "status_code", None), (r.text or ""), ""
    except Exception as e:
        return False, None, "", str(e)

def _external_log_event(kind: str, row: dict):
    """Send an event to external sinks (webhook and/or Supabase) if configured.
    Returns a dict with per-sink results."""
    payload = {"kind": kind, **row}
    results = {"webhook": None, "supabase": None}

    ok, sc, txt, err = _post_json_webhook(payload)
    results["webhook"] = {"ok": ok, "status_code": sc, "response": (txt[:500] if txt else ""), "error": err}

    try:
        table = st.secrets.get("SUPABASE_LOG_TABLE", "")
    except Exception:
        table = ""
    if table:
        ok2, sc2, txt2, err2 = _supabase_insert(table, payload)
        results["supabase"] = {"ok": ok2, "status_code": sc2, "response": (txt2[:500] if txt2 else ""), "error": err2}

    return results

def _ev_audit_load(path: str = EV_AUDIT_PATH, max_rows: int = 20000) -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                rows.append(row)
                if i >= max_rows:
                    break
        return rows
    except Exception:
        return []

def _ev_audit_summary(rows: List[Dict[str, Any]], days: int = 14) -> Dict[str, Any]:
    """Summarize last N days rows (best-effort; ts_utc ISO8601)."""
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    def parse_ts(s: str):
        try:
            # expecting 2026-02-23T12:34:56Z
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        except Exception:
            return None
    recent = []
    for r in rows:
        ts = parse_ts(str(r.get("ts_utc","")))
        if ts and ts >= cutoff:
            recent.append(r)
    total = len(recent)
    trade = sum(1 for r in recent if str(r.get("decision_adj","")) == "TRADE")
    killed = sum(1 for r in recent if str(r.get("killed_by_adj","")).lower() in ("true","1","yes"))
    return {"days": days, "total": total, "trade": trade, "no_trade": total-trade, "killed_by_adj": killed}


# =========================
# Signal / Trade logs (for real operations)
# =========================
SIGNAL_LOG_PATH = "logs/signals.csv"
TRADE_LOG_PATH = "logs/trades.csv"

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _safe_makedirs_for(path: str) -> None:
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass

def _append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> bool:
    """Append row to CSV with header creation. Returns True on success. Never raises."""
    try:
        _safe_makedirs_for(path)
        file_exists = os.path.exists(path)
        # If file exists but header differs, write to a versioned file instead (avoid corrupting).
        if file_exists:
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    first = f.readline().strip()
                if first and (first.split(",") != fieldnames):
                    base, ext = os.path.splitext(path)
                    path = f"{base}_v2{ext}"
                    file_exists = os.path.exists(path)
            except Exception:
                pass
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})
        return True
    except Exception:
        return False

def _load_csv_df(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _make_signal_id(pair_symbol: str) -> str:
    ps = re.sub(r"[^A-Z0-9_\-]", "", (pair_symbol or "").upper())
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{ps}_{uuid.uuid4().hex[:6]}"

def _norm_side(x: Any) -> str:
    s = str(x or "").upper().strip()
    if s in ("LONG", "BUY"):
        return "LONG"
    if s in ("SHORT", "SELL"):
        return "SHORT"
    return s or "—"

def _float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def _calc_r_multiple(side: str, entry: float, exit_price: float, stop: float) -> Optional[float]:
    """R-multiple based on stop distance. Returns None if cannot compute."""
    try:
        side = _norm_side(side)
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        if side == "LONG":
            return (exit_price - entry) / risk
        if side == "SHORT":
            return (entry - exit_price) / risk
        return None
    except Exception:
        return None

def _build_signal_row(pair_label: str, ctx: Dict[str, Any], feats: Dict[str, Any], plan: Dict[str, Any],
                      price_meta: Dict[str, Any], ext_meta: Dict[str, Any]) -> Dict[str, Any]:
    sym = str(ctx.get("pair_symbol") or _pair_label_to_symbol(pair_label))
    sid = _make_signal_id(sym)
    return {
        "ts_utc": _now_utc_iso(),
        "signal_id": sid,
        "pair": str(pair_label),
        "symbol": sym,
        "timeframe_mode": str(st.session_state.get("timeframe_mode", "")),
        "style_name": str(ctx.get("style_name", "")),
        "priority_mode": str(st.session_state.get("priority_mode", "")),
        "decision": str(plan.get("decision", "")),
        "gate_mode": str(plan.get("gate_mode", "")),
        "ev_raw": _float_or_none(plan.get("expected_R_ev_raw", plan.get("expected_R_ev"))),
        "ev_adj": _float_or_none(plan.get("expected_R_ev_adj", plan.get("expected_R_ev"))),
        "ev_used": _float_or_none(plan.get("expected_R_ev")),
        "dynamic_threshold": _float_or_none(plan.get("dynamic_threshold")),
        "confidence": _float_or_none(plan.get("confidence")),
        "p_win": _float_or_none(plan.get("p_win_ev")),
        "dominant_state": _dominant_state(plan.get("state_probs", {})),
        "direction": _norm_side(plan.get("direction", "")),
        "entry_hint": _float_or_none(plan.get("entry_price")),
        "sl_hint": _float_or_none(plan.get("stop_loss")),
        "tp_hint": _float_or_none(plan.get("take_profit")),
        "trail_sl_hint": _float_or_none(plan.get("trail_sl")),
        "tp_extend_factor": _float_or_none(plan.get("extend_factor")),
        "global_risk_index": _float_or_none(feats.get("global_risk_index")),
        "war_probability": _float_or_none(feats.get("war_probability")),
        "financial_stress": _float_or_none(feats.get("financial_stress")),
        "macro_risk_score": _float_or_none(feats.get("macro_risk_score")),
        "price_close": _float_or_none(ctx.get("price")),
        "price_meta": (json.dumps(price_meta, ensure_ascii=False)[:2000] if isinstance(price_meta, dict) else ""),
        "external_meta": (json.dumps(ext_meta, ensure_ascii=False)[:2000] if isinstance(ext_meta, dict) else ""),
        "why": str(plan.get("why", ""))[:500],
        "veto_reasons": (json.dumps(plan.get("veto_reasons", []), ensure_ascii=False)[:500] if plan.get("veto_reasons") else ""),
    }

def _append_signal(row: Dict[str, Any]) -> bool:
    fieldnames = [
        "ts_utc","signal_id","pair","symbol","timeframe_mode","style_name","priority_mode",
        "decision","gate_mode","ev_raw","ev_adj","ev_used","dynamic_threshold","confidence","p_win","dominant_state",
        "direction","entry_hint","sl_hint","tp_hint","trail_sl_hint","tp_extend_factor",
        "global_risk_index","war_probability","financial_stress","macro_risk_score","price_close",
        "why","veto_reasons","price_meta","external_meta",
    ]
    return _append_csv_row(SIGNAL_LOG_PATH, fieldnames, row)

def _append_trade(row: Dict[str, Any]) -> bool:
    fieldnames = [
        "ts_record_utc","trade_id","signal_id","pair","symbol","side","ts_open_utc","ts_close_utc",
        "entry","exit","stop","take_profit","r_multiple","comment",
    ]
    return _append_csv_row(TRADE_LOG_PATH, fieldnames, row)

def _compute_trade_metrics(df_trades: pd.DataFrame) -> Dict[str, Any]:
    if df_trades is None or df_trades.empty:
        return {"n": 0}
    d = df_trades.copy()
    # numeric coercions
    for c in ["entry","exit","stop","take_profit","r_multiple"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "r_multiple" not in d.columns:
        return {"n": int(len(d))}
    r = d["r_multiple"].dropna()
    if r.empty:
        return {"n": int(len(d))}
    wins = r[r > 0]
    losses = r[r <= 0]
    pf = (wins.sum() / abs(losses.sum())) if (not losses.empty and abs(losses.sum()) > 1e-9) else (float("inf") if wins.sum() > 0 else None)
    cum = r.cumsum()
    dd = (cum - cum.cummax()).min() if len(cum) else 0.0
    return {
        "n": int(len(r)),
        "expectancy_R": float(r.mean()),
        "median_R": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": (float(pf) if pf is not None and math.isfinite(pf) else pf),
        "max_drawdown_R": float(dd),
        "sum_R": float(r.sum()),
    }


def _apply_trend_assist(plan_ui: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Add AI engine fields + optional EV bonus for strong trend continuation."""
    out = dict(plan_ui or {})
    phase = str(ctx.get("phase","UNKNOWN"))
    strength = float(ctx.get("trend_strength", 0.0) or 0.0)
    p_up = float(ctx.get("p_cont_up", float("nan")))
    p_dn = float(ctx.get("p_cont_dn", float("nan")))
    out["phase"] = phase
    out["trend_strength"] = strength
    out["p_cont_up"] = p_up
    out["p_cont_dn"] = p_dn
    # bonus: only when continuation edge is clear
    edge = 0.0
    if (p_up == p_up) and (p_dn == p_dn):
        edge = p_up - p_dn
    bonus = 0.0
    if strength >= 0.55 and edge >= 0.12:
        # up-trend bonus helps avoid missing clean trends; capped for safety
        bonus = float(min(0.06, 0.06 * strength * min(1.0, edge/0.25)))
    out["trend_bonus_R"] = bonus
    # optional: use in display / gate
    use_assist = bool(st.session_state.get("trend_assist_enable", True))
    if use_assist:
        try:
            base_ev = float(out.get("expected_R_ev") or 0.0)
            out["expected_R_ev_with_trend"] = base_ev + bonus
            # If the only reason was 'EV < threshold', allow TRADE when trend bonus bridges the gap (still keeps other vetoes)
            try:
                if str(out.get("decision","")) == "NO_TRADE":
                    thr = float(out.get("dynamic_threshold") or 0.0)
                    evw = float(out.get("expected_R_ev_with_trend") or 0.0)
                    veto = list(out.get("veto_reasons") or [])
                    ev_only = (len(veto)==0) or all(("EV" in str(v) or "期待値" in str(v)) for v in veto)
                    if ev_only and evw >= thr and strength >= 0.55 and edge >= 0.12:
                        out["decision"] = "TRADE"
                        out.setdefault("veto_reasons", [])
                        out["trend_assist_promoted"] = True
            except Exception:
                pass

        except Exception:
            out["expected_R_ev_with_trend"] = out.get("expected_R_ev")
    else:
        out["expected_R_ev_with_trend"] = out.get("expected_R_ev")
    return out


def _estimate_target_zones(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Estimate upside/downside target zones from recent structure.
    Not a prediction; provides probabilistic reference zones.
    """
    try:
        if df is None or (hasattr(df, "empty") and df.empty) or len(df) < 80:
            return {"ok": False}
        d = df.copy()
        for col in ["High", "Low", "Close"]:
            if col not in d.columns:
                return {"ok": False}
        c = pd.to_numeric(d["Close"], errors="coerce").fillna(method="ffill").fillna(method="bfill")
        h = pd.to_numeric(d["High"], errors="coerce").fillna(method="ffill").fillna(method="bfill")
        l = pd.to_numeric(d["Low"], errors="coerce").fillna(method="ffill").fillna(method="bfill")
        last = float(c.iloc[-1])
        atr14 = _atr(d, 14).fillna(method="ffill").fillna(method="bfill")
        atr = float(atr14.iloc[-1])
        if not (atr > 0):
            return {"ok": False}
        # recent structure: highest high in last 20 bars
        look = 20
        recent_high = float(h.iloc[-look:].max())
        recent_low = float(l.iloc[-look:].min())
        # find prior swing low within last 80 bars before the recent_high (basic)
        win = 80
        idx_high = int(h.iloc[-win:].idxmax()) if hasattr(h.iloc[-win:], "idxmax") else None
        # idx_high may be label; convert to position
        try:
            pos_high = int(np.argmax(h.iloc[-win:].to_numpy())) + (len(h) - win)
        except Exception:
            pos_high = len(h) - look
        pos_low = int(np.argmin(l.iloc[max(0, pos_high-60):pos_high].to_numpy())) + max(0, pos_high-60)
        swing_low = float(l.iloc[pos_low])
        swing_high = float(h.iloc[pos_high])

        # fib extensions upward from swing_low->swing_high
        move = max(1e-9, swing_high - swing_low)
        fib127 = swing_low + 1.272 * move
        fib161 = swing_low + 1.618 * move

        # ATR ladder from last
        up1 = last + 1.0 * atr
        up2 = last + 2.0 * atr
        dn1 = last - 1.0 * atr
        dn2 = last - 2.0 * atr

        return {
            "ok": True,
            "last": last,
            "atr": atr,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "swing_low": swing_low,
            "swing_high": swing_high,
            "targets_up": [
                {"name": "ATR+1", "price": up1},
                {"name": "ATR+2", "price": up2},
                {"name": "FIB 1.272", "price": fib127},
                {"name": "FIB 1.618", "price": fib161},
            ],
            "targets_dn": [
                {"name": "ATR-1", "price": dn1},
                {"name": "ATR-2", "price": dn2},
                {"name": "recent_low", "price": recent_low},
            ],
        }
    except Exception:
        return {"ok": False}


def _render_ai_engine_panel(ctx: Dict[str, Any], plan_ui: Dict[str, Any]):
    st.markdown("### 🤖 AIエンジン（フェーズ分類 / 継続確率 / 類似パターン / 利確管理）")
    with st.expander("AI診断（トレンド・継続・パターン・利確）", expanded=False):
        phase = str((ctx.get("phase") if isinstance(ctx, dict) else None) or (ctx.get("phase_label") if isinstance(ctx, dict) else None) or "UNKNOWN")
        strength = float((ctx.get("trend_strength", 0.0) if isinstance(ctx, dict) else 0.0) or 0.0)
        p_up = (ctx.get("p_cont_up") if isinstance(ctx, dict) else None)
        if p_up is None:
            p_up = (ctx.get("cont_p_up") if isinstance(ctx, dict) else float("nan"))
        p_dn = (ctx.get("p_cont_dn") if isinstance(ctx, dict) else None)
        if p_dn is None:
            p_dn = (ctx.get("cont_p_dn") if isinstance(ctx, dict) else float("nan"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("フェーズ", phase)
        c2.metric("トレンド強度", f"{strength:.2f}")
        c3.metric("上昇継続確率", ("—" if p_up != p_up else f"{float(p_up)*100:.0f}%"))
        c4.metric("下落継続確率", ("—" if p_dn != p_dn else f"{float(p_dn)*100:.0f}%"))
        st.caption("※継続確率は、直近10年（取得できる範囲）のOHLCから簡易学習したロジスティック回帰モデル（統計推定）です。")

        # Target zones (not a prediction)
        df_ref = _first_non_none(ctx, ("_df","df","_price_df","_df_price","_df_hist"))
        tz = _estimate_target_zones(df_ref if isinstance(df_ref, pd.DataFrame) else None)
        if tz.get("ok"):
            st.markdown("#### 参考ターゲットゾーン（天井“予知”ではなく、構造・ATRに基づく目安）")
            t_up = pd.DataFrame(tz["targets_up"])
            t_dn = pd.DataFrame(tz["targets_dn"])
            try:
                t_up["price"] = pd.to_numeric(t_up["price"], errors="coerce").round(3)
                t_dn["price"] = pd.to_numeric(t_dn["price"], errors="coerce").round(3)
            except Exception:
                pass
            cc1, cc2, cc3 = st.columns([1,1,2])
            cc1.metric("現在値", f"{tz['last']:.3f}")
            cc2.metric("ATR(14)", f"{tz['atr']:.3f}")
            cc3.metric("直近高値/安値(20本)", f"{tz['recent_high']:.3f} / {tz['recent_low']:.3f}")
            ctu, ctd = st.columns(2)
            with ctu:
                st.write("上方向（候補）")
                st.dataframe(t_up, use_container_width=True, hide_index=True)
            with ctd:
                st.write("下方向（候補）")
                st.dataframe(t_dn, use_container_width=True, hide_index=True)

        # Similar patterns
        patt = ctx.get("_similar_patterns_df")
        if isinstance(patt, pd.DataFrame) and not patt.empty:
            st.markdown("#### 類似パターン（直近30本と似た局面）")
            st.dataframe(patt, use_container_width=True)
            try:
                avg = float(pd.to_numeric(patt["fwd_R_atr"], errors="coerce").mean())
                st.caption(f"類似局面の平均（{int(ctx.get('horizon_days',5))}日先）: {avg:+.2f} ATR-R")
            except Exception:
                pass


# True RL exit agent (Q-learning) — trains from history and persists model
st.markdown("#### 強化学習（RL）利確管理：学習済みポリシー")
st.caption("これは **学習するRL** です（Q-learning）。ボタンで学習→モデル保存→以後は学習済みポリシーで提案します。")

rl_dir = "logs"
os.makedirs(rl_dir, exist_ok=True)
_pair_for_rl_path = (st.session_state.get("pair_label")
                    or st.session_state.get("selected_pair")
                    or st.session_state.get("pair")
                    or "AUD/JPY")
rl_path = os.path.join(rl_dir, f"rl_exit_{str(_pair_for_rl_path).replace('/','')}_1d.json")

# load agent (best-effort)
agent = _load_rl_agent(rl_path)

cA, cB, cC = st.columns([1,1,2])
with cA:
    episodes = st.number_input("学習エピソード数（多いほど精度↑/時間↑）", min_value=200, max_value=20000, value=3000, step=200)
with cB:
    max_h = st.number_input("最大保有（学習）日数", min_value=5, max_value=60, value=20, step=1)
with cC:
    st.write("**モデル状態**")
    st.code("trained" if (agent is not None and len(getattr(agent,'q',{}))>0) else "not trained", language="text")


cT1, cT2 = st.columns([1,1])
with cT1:
    do_wfa = st.button("🧠 RLを自動WFA学習（推奨：係数も自動選定）", key="train_rl_wfa")
with cT2:
    do_quick = st.button("🧠 RLを学習（手動係数・高速）", key="train_rl_quick")

if do_wfa or do_quick:
    with st.spinner("RL学習中...（初回は少し時間がかかります）"):
        # pair_label may not be defined yet in top-down execution; re-fetch safely
        try:
            pair_label = st.session_state.get('pair_label', 'USD/JPY')
        except Exception:
            pair_label = 'USD/JPY'
        df_rl, meta_rl = fetch_price_history(pair_label, _pair_label_to_symbol(pair_label), period="10y", interval="1d", prefer_stooq=True)
        if df_rl is None or df_rl.empty:
            st.error(f"価格データ取得に失敗: {meta_rl}")
        else:
            ph_s, st_s = _compute_phase_strength_series(df_rl)
            horizon_days = int(st.session_state.get('horizon_days', 5) or 5)
            pu_s, pd_s = _compute_cont_p_series(df_rl, horizon=max(3, horizon_days))
            best_params = {"dd_penalty": 0.15, "time_penalty": 0.01, "atr_mult_stop": 1.5}
            if do_wfa:
                wfa = _wfa_select_rl_coeffs(
                    df_rl, ph_s, st_s, pu_s, pd_s,
                    splits=3,
                    train_episodes=max(800, int(episodes * 0.6)),
                    eval_episodes=800,
                    max_horizon=int(max_h),
                )
                if not wfa.get("ok"):
                    st.warning(f"WFAが失敗したため、既定係数で学習します: {wfa}")
                else:
                    best_params = dict(wfa["best"])
                    st.success(f"WFA選定係数: dd_penalty={best_params['dd_penalty']}, time_penalty={best_params['time_penalty']}, atr_mult_stop={best_params['atr_mult_stop']}（score={best_params['score']:+.4f}）")
                    try:
                        st.dataframe(pd.DataFrame(wfa.get("top", [])), use_container_width=True)
                    except Exception:
                        pass

            ag = RLExitAgent()
            res = ag.train_from_price(
                df_rl, ph_s, st_s, pu_s, pd_s,
                episodes=int(episodes),
                max_horizon=int(max_h),
                atr_mult_stop=float(best_params.get("atr_mult_stop", 1.5)),
                dd_penalty=float(best_params.get("dd_penalty", 0.15)),
                time_penalty=float(best_params.get("time_penalty", 0.01)),
            )
            if res.get("ok"):
                _save_rl_agent(rl_path, ag)
                st.success(f"RL学習完了: {res}")
                agent = ag
                st.session_state[f"_rl_best_params_{pair_label}"] = best_params
            else:
                st.error(f"RL学習失敗: {res}")
st.divider()

# --- ensure variables exist (avoid NameError) ---
_ctx_for_rl = locals().get("ctx", None)
if not isinstance(_ctx_for_rl, dict):
    _ctx_for_rl = {}
phase = str(locals().get("phase", None) or _ctx_for_rl.get("phase", "UNKNOWN"))
strength = float(locals().get("strength", None) or _ctx_for_rl.get("trend_strength", 0.0) or 0.0)
p_up = locals().get("p_up", None)
p_dn = locals().get("p_dn", None)
try:
    if p_up is None:
        p_up = float(_ctx_for_rl.get("p_up", float("nan")))
    if p_dn is None:
        p_dn = float(_ctx_for_rl.get("p_dn", float("nan")))
except Exception:
    p_up, p_dn = float("nan"), float("nan")
ur = st.number_input("現在の含み損益（R）※損切幅=1R", value=0.0, step=0.1, format="%.2f")
dd_in = st.number_input("含み損の最大（R）※マイナス（任意）", value=0.0, step=0.1, format="%.2f")
reco = _rl_exit_reco(agent, phase, strength,
                     float(p_up) if p_up==p_up else float('nan'),
                     float(p_dn) if p_dn==p_dn else float('nan'),
                     float(ur), float(dd_in))
st.write(f"推奨アクション: **{reco.get('action','HOLD')}**  （edge={reco.get('edge',0.0):+.2f} / state={reco.get('state','')}）")
if reco.get("note"):
    st.info(reco.get("note"))
st.json(reco.get("q", {}))




def _render_hold_manage_panel(pair_label: str, ctx_in: Dict[str, Any], plan_ui: Dict[str, Any], feats: Dict[str, Any], keys: Dict[str, str]):
    """保有中のイベント接近対応（縮退/一部利確/建値移動/新規追加禁止）を表示。
    - 実際の建玉をアプリが自動で把握することはできないので、運用者が「保有開始/解除」を押して管理します。
    - ルール自体は logic.py 側に統合済み（swing_hold_v1）。
    """
    try:
        st.markdown("### 📌 保有ポジション管理（イベント接近時の対応）")
        st.caption("保有中だけ使います。**イベント接近・週末/週跨ぎ**では、\n- 追加建て禁止\n- 建玉縮退\n- 一部利確\n- 建値（BE）へストップ移動\nを自動で推奨します（発注はしません）。")

        # ctx_in に _df が入っている前提（_build_ctx で付与）
        df_ref = _first_non_none(ctx_in, ("_df", "df", "_price_df", "_df_price", "_df_hist"))
        if not isinstance(df_ref, pd.DataFrame) or df_ref.empty:
            st.info("価格データが無いので保有管理は表示できません。")
            return

        # Session state storage
        pos_key = "open_position"
        pos = st.session_state.get(pos_key, None)
        if not isinstance(pos, dict):
            pos = None

        # Current plan snapshot (for register)
        side_plan = str(plan_ui.get("side") or ("BUY" if str(plan_ui.get("direction")) == "LONG" else "SELL"))
        entry_plan = float(plan_ui.get("entry") or plan_ui.get("entry_price") or ctx_in.get("price") or 0.0)
        sl_plan = float(plan_ui.get("sl") or plan_ui.get("stop_loss") or 0.0)
        tp_plan = float(plan_ui.get("tp") or plan_ui.get("take_profit") or 0.0)

        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            if st.button("📌 このプランで保有開始として登録", key=f"pos_register_{pair_label}"):
                st.session_state[pos_key] = {
                    "pair": str(pair_label),
                    "side": str(side_plan).upper(),
                    "entry": float(entry_plan),
                    "sl": float(sl_plan),
                    "tp": float(tp_plan),
                    "opened_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                st.success("保有ポジションとして登録しました（このアプリは発注しません）。")
                st.rerun()
        with c2:
            if st.button("🧹 保有解除", key=f"pos_clear_{pair_label}"):
                if pos_key in st.session_state:
                    st.session_state.pop(pos_key, None)
                st.success("保有ポジションを解除しました。")
                st.rerun()
        with c3:
            st.caption("※実際にポジションを持った時だけ『保有開始』を押してください。解除を忘れると管理提案が残ります。")

        # Only show when position exists and matches pair
        pos = st.session_state.get(pos_key, None)
        if not isinstance(pos, dict) or str(pos.get("pair")) != str(pair_label):
            st.info("このペアの保有ポジションが未登録です。保有中なら『保有開始として登録』を押してください。") 
            return

        # Current price (auto + override)
        try:
            px_auto = float(df_ref["Close"].astype(float).iloc[-1])
        except Exception:
            px_auto = float(ctx_in.get("price") or entry_plan or 0.0)
        step = 0.001 if "/JPY" not in str(pair_label) else 0.001  # 最低限（UI崩れ防止）
        px_now = st.number_input("現在値（自動入力。必要なら修正）", value=float(px_auto), step=float(step), format="%.6f")
        pos2 = dict(pos)
        pos2["current_price"] = float(px_now)
        pos2["open"] = True

        # Call logic again for holding management recommendation (manage_only)
        ctx2 = dict(ctx_in or {})
        ctx2["position"] = pos2
        ctx2["position_open"] = True
        ctx2["manage_only"] = True

        try:
            plan2 = logic.get_ai_order_strategy(price_df=df_ref, pair=pair_label, context_data=ctx2, ext_features=feats, api_key=keys.get("OPENAI_API_KEY", ""))
            hold = plan2.get("hold_manage", {}) or {}
        except Exception as e:
            st.error(f"保有管理の計算でエラー: {e}")
            return

        if not isinstance(hold, dict) or not hold:
            st.info("保有管理の推奨はありません（position 情報不足、またはガード条件に該当なし）。")
            return

        # Summary
        side_jp = "買い" if str(hold.get("side")).upper() == "BUY" else "売り"
        ur = float(hold.get("unrealized_R") or 0.0)
        nh = hold.get("event_next_high_hours", None)

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("保有", f"{side_jp}")
        cc2.metric("含み損益（R）", f"{ur:+.2f}")
        cc3.metric("追加建て", "禁止" if bool(hold.get("no_add")) else "許可")
        cc4.metric("次の高インパクト", ("—" if nh is None else f"{float(nh):.1f}h"))

        # Recommended actions
        st.markdown("#### 推奨アクション（発注は運用者が実行）")
        actions = hold.get("actions", []) or []
        notes = hold.get("notes", []) or []

        # Convert into human-readable recommendations
        rec_lines = []
        if bool(hold.get("no_add")):
            rec_lines.append("- ✅ **新規追加は禁止**（イベント/週末/週跨ぎガード）")
        mult = float(hold.get("reduce_size_mult") or 1.0)
        if mult < 0.999:
            rec_lines.append(f"- ✅ **建玉縮退**：推奨サイズ係数 ×{mult:.2f}（例：半分利確/縮小）")
        pt = float(hold.get("partial_tp_ratio") or 0.0)
        if pt > 0.0:
            rec_lines.append(f"- ✅ **一部利確**：{pt*100:.0f}% を目安")
        if bool(hold.get("move_sl_to_be")) and hold.get("new_sl_reco") is not None:
            rec_lines.append(f"- ✅ **建値移動/防御**：SL を {float(hold.get('new_sl_reco')):.3f} 付近へ（BE）")
        if not rec_lines:
            rec_lines.append("- （特別な推奨はありません）")

        st.markdown("\n".join(rec_lines))

        if notes:
            with st.expander("根拠（メモ）", expanded=False):
                for n in notes:
                    st.write("- " + str(n))

    except Exception:
        # Never break the main screen because of this optional panel
        return



def _render_logging_panel(pair_label: str, plan_ui: Dict[str, Any], ctx: Dict[str, Any], feats: Dict[str, Any],
                          price_meta: Dict[str, Any], ext_meta: Dict[str, Any]):
    """UI: save signal + record trade outcome. Keeps everything optional and non-blocking."""
    st.markdown("### 📝 シグナル/損益ログ（運用用）")
    with st.expander("📝 保存（signals / trades）+ 外部Sink（Webhook/Supabase）", expanded=False):
        row = _build_signal_row(pair_label, ctx, feats, plan_ui, price_meta=price_meta, ext_meta=ext_meta)
        st.caption("運用の第一歩：**『シグナル保存 → 決済後に損益保存 → パフォーマンス自動集計』** の流れを固定します。")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ このシグナルを保存", key=f"save_signal_{row['signal_id']}"):
                ok_local = _append_signal(row)
                st.session_state["last_signal_id"] = row["signal_id"]
                st.session_state["last_signal_row"] = row
                st.session_state["last_plan"] = plan_ui
                st.session_state["last_ctx"] = ctx
                st.session_state["last_feats"] = feats
                # external sinks (best-effort)
                ext = _external_log_event("signal", row)
                if ok_local:
                    st.success(f"保存OK: signal_id={row['signal_id']}")
                else:
                    st.warning("ローカル保存に失敗（外部Sinkの結果を確認してください）")
                st.json(ext)
        with c2:
            st.write("**signal_id**")
            st.code(row["signal_id"])
            st.caption("このIDを決済記録に紐づけると分析が強くなります。")

        st.divider()
        st.subheader("決済後：トレード結果を保存（Rの自動計算）")

        default_sid = str(st.session_state.get("last_signal_id", row["signal_id"]))
        default_side = _norm_side(plan_ui.get("direction", "LONG"))
        default_entry = _float_or_none(plan_ui.get("entry_price")) or _float_or_none(ctx.get("price")) or 0.0
        default_stop = _float_or_none(plan_ui.get("stop_loss")) or 0.0
        default_tp = _float_or_none(plan_ui.get("take_profit")) or 0.0

        form_key = f"trade_close_{pair_label.replace('/','_')}"
        with st.form(form_key, clear_on_submit=False):
            signal_id = st.text_input("signal_id（任意だが推奨）", value=default_sid)
            side = st.selectbox("方向", ["LONG", "SHORT"], index=0 if default_side == "LONG" else 1)
            entry = st.number_input("Entry（約定値）", value=float(default_entry), format="%.5f")
            exit_price = st.number_input("Exit（決済値）", value=float(default_tp or default_entry), format="%.5f")
            stop = st.number_input("Stop（実際のSL）", value=float(default_stop or default_entry), format="%.5f")
            take_profit = st.number_input("TP（実際のTP）", value=float(default_tp or default_entry), format="%.5f")
            ts_open_utc = st.text_input("Open時刻（UTC, ISO）", value=_now_utc_iso())
            ts_close_utc = st.text_input("Close時刻（UTC, ISO）", value=_now_utc_iso())
            comment = st.text_input("メモ（任意）", value="")
            submitted = st.form_submit_button("💾 決済結果を保存")

        if submitted:
            r_mult = _calc_r_multiple(side, float(entry), float(exit_price), float(stop))
            trade_row = {
                "ts_record_utc": _now_utc_iso(),
                "trade_id": uuid.uuid4().hex[:12],
                "signal_id": str(signal_id),
                "pair": str(pair_label),
                "symbol": str(ctx.get("pair_symbol") or _pair_label_to_symbol(pair_label)),
                "side": _norm_side(side),
                "ts_open_utc": str(ts_open_utc),
                "ts_close_utc": str(ts_close_utc),
                "entry": float(entry),
                "exit": float(exit_price),
                "stop": float(stop),
                "take_profit": float(take_profit),
                "r_multiple": (float(r_mult) if r_mult is not None else ""),
                "comment": str(comment)[:500],
            }
            ok_local = _append_trade(trade_row)
            ext = _external_log_event("trade_close", trade_row)
            if ok_local:
                st.success(f"保存OK: trade_id={trade_row['trade_id']} / R={r_mult if r_mult is not None else '—'}")
            else:
                st.warning("ローカル保存に失敗（外部Sinkの結果を確認してください）")
            st.json(ext)

        st.divider()
        st.caption("ローカル保存先: logs/signals.csv / logs/trades.csv（Streamlit Cloudではデプロイで消える場合があるため、外部Sinkを併用推奨）")

# =========================
# Operator-friendly labels
# =========================
STATE_LABELS_JA = {
    "trend_up": "上昇トレンド優勢",
    "trend_down": "下降トレンド優勢",
    "range": "レンジ（往復）優勢",
    "risk_off": "リスクオフ（荒れ/急変）",
}

def _state_label_full(key: str) -> str:
    k = str(key or "")
    ja = STATE_LABELS_JA.get(k, k)
    return f"{ja} ({k})" if k and ja != k else ja

def _bucket_01(v: float) -> str:
    """
    0-1 のリスク値を「低/中/高」に丸める（表示用）。
    ※見た目の赤/黄/緑は“時間軸”と“運用スタイル”で少し動かす（見送りを強制しない）。
    """
    try:
        x = float(v)
    except Exception:
        return "—"
    if x != x:
        return "—"

    # 時間軸（horizon_days）でしきい値を調整
    try:
        hd = int(globals().get("horizon_days", 3))
    except Exception:
        hd = 3

    # base thresholds
    if hd <= 1:         # スキャ
        t1, t2 = 0.30, 0.55
    elif hd <= 4:       # デイトレ
        t1, t2 = 0.33, 0.66
    else:               # スイング
        t1, t2 = 0.40, 0.75

    # スタイル補正（表示だけ）
    style = str(globals().get("style_name", "標準") or "標準")
    if style == "保守":
        t1 -= 0.05
        t2 -= 0.05
    elif style == "攻撃":
        t1 += 0.05
        t2 += 0.05

    t1 = max(0.05, min(0.90, t1))
    t2 = max(t1 + 0.05, min(0.95, t2))

    if x < t1:
        return "低（平常）"
    if x < t2:
        return "中（警戒）"
    return "高（危険）"



def _jp_decision(decision: str) -> str:
    """Decision label for operators (JP)."""
    d = str(decision or "").upper()
    mapping = {
        "TRADE": "エントリー可",
        "NO_TRADE": "見送り",
        "HOLD": "様子見",
        "WAIT": "待機",
        "PAUSE": "一時停止",
    }
    return mapping.get(d, d or "—")


def _lot_multiplier(global_risk_index: Any, alpha: Any, floor: float = 0.2, ceil: float = 1.0) -> float:
    """Recommended lot multiplier (display-only). Safe (never raises NameError)."""
    try:
        r = float(global_risk_index)
    except Exception:
        return 1.0
    try:
        a = float(alpha)
    except Exception:
        a = 0.0

    # sanitize
    if r != r or r is None:
        return 1.0
    if a != a or a is None:
        a = 0.0
    r = max(0.0, min(1.0, r))
    a = max(0.0, min(2.0, a))

    x = 1.0 - a * r
    if x < floor:
        x = floor
    if x > ceil:
        x = ceil
    return float(x)


def _apply_sbi_minlot_guard(plan: dict, *, sbi_min_lot: int = 1) -> dict:
    """SBI（最小1建・小数不可）を前提に、縮退が効かない局面をAI側で『見送り』に倒す。

    目的：
      - 画面が「エントリー可」と出ても、実際は縮退できず事故りやすいケースを可視化＆抑制
      - 根拠を“数値”で説明（SBI補正EV、EV余裕、必要余裕、実質リスク倍率）

    方針（過度に見送り地獄にしないための段階制）：
      - hard: 推奨ロット係数が極端に低い（<0.35）→ ほぼ確実に縮退必須なので見送り
      - soft: 係数が低い（<0.55）かつ、EV余裕が薄い/イベント近接ガード中 → 見送り
      - それ以外は元の decision を尊重
    """
    try:
        if not isinstance(plan, dict):
            return plan
        decision = str(plan.get("decision", "NO_TRADE"))
        # 既に見送りならそのまま
        if decision != "TRADE":
            plan["_sbi_exec_lots"] = 0
            return plan

        ctx = plan.get("_ctx", {}) if isinstance(plan.get("_ctx", {}), dict) else {}
        ev = float(plan.get("expected_R_ev") or 0.0)
        dyn = float(plan.get("dynamic_threshold") or 0.0)
        lot_mult = float(plan.get("_lot_multiplier_reco") or 1.0)
        lot_mult = max(0.000001, min(1.0, lot_mult))

        # SBIは最小1建。ここでは「基準建玉=1」を前提に、縮退不能による実質リスク倍率を算出。
        base_lots = 1.0
        desired_lots = base_lots * lot_mult
        exec_lots = max(float(sbi_min_lot), float(int(round(desired_lots))))
        # roundで0になる可能性があるので再ガード
        exec_lots = max(float(sbi_min_lot), exec_lots)

        risk_x = float(exec_lots) / max(desired_lots, 1e-9)  # 例: 0.269→1建=3.7倍
        ev_sbi = ev / max(risk_x, 1e-9)  # ＝ ev * desired/exec
        ev_margin = ev - dyn

        # 必要余裕（数値基準）
        req_margin = 0.03  # 通常時の最低余裕
        event_ban = bool(ctx.get("event_market_ban_active", False))
        if event_ban:
            req_margin += 0.03  # イベント近接中は余裕を上乗せ（見送り地獄防止のため控えめ）
        if float(ctx.get("weekcross_risk") or 0.0) > 0.0:
            req_margin += 0.02
        if float(ctx.get("weekend_risk") or 0.0) > 0.0:
            req_margin += 0.03
        # 縮退不能の度合い（実質リスク倍率）を余裕に反映（上限0.10R）
        req_margin += min(0.10, 0.04 * max(0.0, risk_x - 1.0))

        hard_veto = (lot_mult < 0.35)
        soft_veto = (lot_mult < 0.55) and (event_ban or (ev_margin < req_margin) or (ev_sbi < dyn))

        # 情報はplanに格納（UI表示用）
        plan["_sbi"] = {
            "min_lot": int(sbi_min_lot),
            "base_lots": base_lots,
            "desired_lots": float(desired_lots),
            "exec_lots": int(exec_lots),
            "risk_x": float(risk_x),
            "ev_sbi": float(ev_sbi),
            "ev_margin": float(ev_margin),
            "ev_margin_req": float(req_margin),
            "event_ban": bool(event_ban),
            "decision_sbi": "NO_TRADE" if (hard_veto or soft_veto) else "TRADE",
        }
        plan["_sbi_exec_lots"] = int(exec_lots) if not (hard_veto or soft_veto) else 0

        if hard_veto or soft_veto:
            reasons = []
            reasons.append(f"SBI最小{int(sbi_min_lot)}建のため縮退不可（推奨係数{lot_mult:.3f}→実質リスク×{risk_x:.2f}）")
            reasons.append(f"SBI補正EV {ev_sbi:+.3f} / 閾値 {dyn:.3f}")
            reasons.append(f"EV余裕 {ev_margin:+.3f} / 必要余裕 {req_margin:.3f}" + ("（イベント近接）" if event_ban else ""))

            plan["_decision_original"] = plan.get("decision")
            plan["decision"] = "NO_TRADE"
            plan["_decision_override_reason"] = "SBI最小1建ガード: " + " / ".join(reasons)

            # veto表示にも反映（既存の理由を壊さない）
            vr = plan.get("veto_reasons")
            if not isinstance(vr, list):
                vr = []
            for r in reasons:
                vr.append(r)
            plan["veto_reasons"] = vr
            plan["veto"] = vr

            # why も補足
            plan["why"] = " / ".join(reasons[:2])

        return plan
    except Exception:
        # 失敗してもアプリを落とさない（安全側：元のplanを返す）
        return plan



def _apply_swing_lot_guards(lot_mult: float, plan: Any) -> float:
    """Apply mandatory swing risk shrink to lot multiplier (display-only)."""
    try:
        x = float(lot_mult)
    except Exception:
        x = 1.0

    ctx = {}
    try:
        if isinstance(plan, dict):
            ctx = plan.get("_ctx", {}) if isinstance(plan.get("_ctx", {}), dict) else {}
    except Exception:
        ctx = {}

    # Event density shrink (mandatory)
    try:
        ef = float(ctx.get("event_risk_factor") or 0.0)
        ef = max(0.0, min(1.0, ef))
        x *= max(0.20, 1.0 - 0.60 * ef)
    except Exception:
        pass

    # Thu/Fri week-cross rule (mandatory)
    try:
        wc = float(ctx.get("weekcross_risk") or 0.0)
        if wc > 0.0:
            x *= 0.75
    except Exception:
        pass

    # Weekend gap risk
    try:
        wknd = float(ctx.get("weekend_risk") or 0.0)
        if wknd > 0.0:
            x *= 0.60
    except Exception:
        pass

    # clamp
    if x < 0.20:
        x = 0.20
    if x > 1.00:
        x = 1.00
    return float(x)



def _jp_side(side: str) -> str:
    s = str(side or "").upper()
    if s in ("BUY", "LONG"):
        return "買い"
    if s in ("SELL", "SHORT"):
        return "売り"
    return "—"

def _jp_order_kind(kind: str) -> str:
    k = str(kind or "").upper()

    # finer-grain entry types (internal codes)
    if k in ("LIMIT_PULLBACK",):
        return "指値（押し目/戻り）"
    if k in ("STOP_BREAKOUT", "STOP_BREAKDOWN"):
        return "逆指値（ブレイク）"

    # generic kinds
    if k in ("MARKET", "MKT", "MARKET_NOW", "NOW") or k.startswith("MARKET"):
        return "成行"
    if k in ("LIMIT", "LMT") or k.startswith("LIMIT"):
        return "指値"
    if k in ("STOP", "STP", "STOP_MARKET") or k.startswith("STOP"):
        return "逆指値"
    return "—"

def _infer_entry_order_kind(direction: str, entry: float, current_price: float, tol_ratio: float = 0.00015) -> str:
    """entry と current の位置関係から、発注種別（成行/指値/逆指値）を推定する。
    tol_ratio は“ほぼ同値”を成行扱いにする許容比率（例: 0.00015=0.015%）。
    """
    try:
        e = float(entry)
        c = float(current_price)
        if c <= 0:
            return "MARKET"
        if abs(e - c) / c <= tol_ratio:
            return "MARKET"
        d = str(direction or "").upper()
        if d == "LONG":
            return "LIMIT" if e < c else "STOP"
        if d == "SHORT":
            return "LIMIT" if e > c else "STOP"
        return "MARKET"
    except Exception:
        return "MARKET"

def _jp_order_scheme(has_entry: bool, has_tp: bool, has_sl: bool) -> str:
    # 一般的な店頭FXの発注方式表記に合わせる
    if has_entry and has_tp and has_sl:
        return "IFDOCO"
    if has_entry and (has_tp or has_sl):
        return "IFD"
    if (not has_entry) and has_tp and has_sl:
        return "OCO"
    return "—"

def _first_non_none(d: dict, keys: tuple[str, ...]):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None

def _fmt_price(x: Any, decimals: int = 3) -> str:
    """Format price for UI: max `decimals` places, trim trailing zeros. Internal calc stays full precision."""
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return str(x)
    s = f"{v:.{decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return s


def _action_hint(global_risk: float, war: float, fin: float, macro: float, bs_flag: bool, gov_enabled: bool) -> str:
    """
    運用者向けの“日本語の次の行動”だけを返す（強制停止はしない）。
    しきい値は時間軸（horizon_days）とスタイル（style_name）で少し動かす。
    """
    if bs_flag or (not gov_enabled):
        return "🛑 新規エントリー停止（強制ガード発動）"

    g = float(global_risk or 0.0)
    w = float(war or 0.0)
    f = float(fin or 0.0)
    m = float(macro or 0.0)

    # ベース：デイトレ想定
    hi_g, hi_w, hi_f, hi_m = 0.80, 0.60, 0.80, 0.80
    mid_g, mid_w, mid_f, mid_m = 0.55, 0.35, 0.55, 0.55

    # 時間軸補正（スキャは敏感、スイングは鈍感）
    try:
        hd = int(globals().get("horizon_days", 3))
    except Exception:
        hd = 3

    if hd <= 1:  # スキャ
        hi_g, hi_w, hi_f, hi_m = 0.70, 0.55, 0.70, 0.70
        mid_g, mid_w, mid_f, mid_m = 0.45, 0.30, 0.45, 0.45
    elif hd >= 7:  # スイング
        hi_g, hi_w, hi_f, hi_m = 0.85, 0.65, 0.85, 0.85
        mid_g, mid_w, mid_f, mid_m = 0.60, 0.40, 0.60, 0.60

    # スタイル補正（保守は厳しめ、攻撃は緩め）
    style = str(globals().get("style_name", "標準") or "標準")
    delta = -0.05 if style == "保守" else (0.05 if style == "攻撃" else 0.0)
    hi_g, hi_f, hi_m = hi_g + delta, hi_f + delta, hi_m + delta
    mid_g, mid_f, mid_m = mid_g + delta, mid_f + delta, mid_m + delta
    # war は過敏になりやすいので控えめに
    hi_w = hi_w + (delta * 0.5)
    mid_w = mid_w + (delta * 0.5)

    if (g >= hi_g) or (w >= hi_w) or (f >= hi_f) or (m >= hi_m):
        return "🔴 高リスク：見送り推奨（入るならロット最小・短期・監視必須）"
    if (g >= mid_g) or (w >= mid_w) or (f >= mid_f) or (m >= mid_m):
        return "🟡 警戒：ロット縮小/回数制限（見送り増は正常）"
    return "🟢 平常：通常運用（ただし重要指標/要人発言/週末は別途警戒）"


def _normalize_pair_label(s: str) -> str:
    s = (s or "").strip().upper().replace(" ", "")
    s = s.replace("-", "/")
    if "/" not in s and len(s) == 6:
        s = s[:3] + "/" + s[3:]
    return s

def _load_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return os.getenv(name, default) or default

def _pair_label_to_symbol(pair_label: str) -> str:
    pl = _normalize_pair_label(pair_label)
    mp = getattr(logic, "PAIR_MAP", None)
    if isinstance(mp, dict) and pl in mp:
        return mp[pl]
    fallback = {
        "USD/JPY": "JPY=X",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "AUD/USD": "AUDUSD=X",
        "EUR/JPY": "EURJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "AUD/JPY": "AUDJPY=X",
    }
    return fallback.get(pl, "JPY=X")

def _pair_label_to_stooq_symbol(pair_label: str) -> Optional[str]:
    pl = _normalize_pair_label(pair_label)
    mapping = {
        "USD/JPY": "usdjpy",
        "EUR/USD": "eurusd",
        "GBP/USD": "gbpusd",
        "AUD/USD": "audusd",
        "EUR/JPY": "eurjpy",
        "GBP/JPY": "gbpjpy",
        "AUD/JPY": "audjpy",
    }
    return mapping.get(pl)

def _coerce_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    need = ["Open", "High", "Low", "Close"]
    for c in need:
        if c not in d.columns:
            return pd.DataFrame()
    d = d[need].dropna()
    return d


# =========================
# AI Trend Engine (phase classification, continuation probability, similar pattern search, RL-like exit manager)
# =========================
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception:
    LogisticRegression = None
    StandardScaler = None
    Pipeline = None

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(df)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0)

def _phase_classify(df: pd.DataFrame) -> Dict[str, Any]:
    """Return phase label + trend strength (0-1-ish)."""
    if df is None or df.empty or len(df) < 60:
        return {"phase": "UNKNOWN", "trend_strength": 0.0, "adx": float("nan")}
    d = df.copy()
    c = d["Close"]
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema200 = _ema(c, 200) if len(c) >= 220 else _ema(c, 100)
    adx14 = _adx(d, 14)
    slope20 = (ema20 - ema20.shift(5)) / (5 * _atr(d, 14).replace(0, np.nan))
    slope = float(slope20.iloc[-1]) if not np.isnan(slope20.iloc[-1]) else 0.0
    adx_v = float(adx14.iloc[-1])
    # strength: combine ADX and slope magnitude
    strength = float(min(1.0, (max(0.0, adx_v - 10) / 30.0) * min(1.0, abs(slope))))
    up = (ema20.iloc[-1] > ema50.iloc[-1]) and (ema50.iloc[-1] > ema200.iloc[-1]) and (slope > 0.15)
    dn = (ema20.iloc[-1] < ema50.iloc[-1]) and (ema50.iloc[-1] < ema200.iloc[-1]) and (slope < -0.15)
    # range if weak trend
    if adx_v < 15 and abs(slope) < 0.10:
        phase = "RANGE"
        strength = float(min(0.4, strength))
    elif up and adx_v >= 18:
        phase = "UP_TREND"
    elif dn and adx_v >= 18:
        phase = "DOWN_TREND"
    else:
        # breakout-ish: 20d high/low break with adx rising
        hh20 = c.rolling(20).max()
        ll20 = c.rolling(20).min()
        if c.iloc[-1] >= hh20.iloc[-2] and adx_v >= float(adx14.iloc[-6]):
            phase = "BREAKOUT_UP"
            strength = max(strength, 0.55)
        elif c.iloc[-1] <= ll20.iloc[-2] and adx_v >= float(adx14.iloc[-6]):
            phase = "BREAKOUT_DOWN"
            strength = max(strength, 0.55)
        else:
            phase = "TRANSITION"
    return {"phase": phase, "trend_strength": strength, "adx": adx_v, "slope20_atr": slope}

def _make_continuation_dataset(df: pd.DataFrame, horizon: int = 5, k_atr: float = 0.6) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build features X and labels y_up / y_dn for continuation within horizon days."""
    d = df.copy()
    c = d["Close"]
    atr14 = _atr(d, 14)
    adx14 = _adx(d, 14)
    rsi14 = _rsi(c, 14)
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ret1 = c.pct_change(1)
    ret3 = c.pct_change(3)
    ret5 = c.pct_change(5)
    atrp = (atr14 / c).replace([np.inf, -np.inf], np.nan)
    emar = (ema20 / ema50 - 1.0).replace([np.inf, -np.inf], np.nan)
    # forward move in ATR units
    fwd = c.shift(-horizon) - c
    move_atr = (fwd / atr14.replace(0, np.nan))
    y_up = (move_atr >= k_atr).astype(int)
    y_dn = (move_atr <= -k_atr).astype(int)
    X = pd.DataFrame({
        "ret1": ret1, "ret3": ret3, "ret5": ret5,
        "atrp": atrp,
        "adx": adx14/50.0,
        "rsi": (rsi14-50.0)/50.0,
        "emar": emar,
    }).dropna()
    y_up = y_up.loc[X.index]
    y_dn = y_dn.loc[X.index]
    # drop last horizon where labels unknown
    X = X.iloc[:-horizon] if len(X) > horizon else X.iloc[:0]
    y_up = y_up.iloc[:-horizon] if len(y_up) > horizon else y_up.iloc[:0]
    y_dn = y_dn.iloc[:-horizon] if len(y_dn) > horizon else y_dn.iloc[:0]
    return X, y_up, y_dn

@st.cache_data(show_spinner=False, ttl=60*60)
def _fit_continuation_models(pair_label: str, interval: str, df: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
    """Fit per-pair logistic models (up/down). Cached."""
    if LogisticRegression is None or df is None or df.empty or len(df) < 400:
        return {"ok": False}
    X, y_up, y_dn = _make_continuation_dataset(df, horizon=horizon)
    if X.empty or y_up.nunique() < 2 or y_dn.nunique() < 2:
        return {"ok": False}
    model_up = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200, class_weight="balanced"))])
    model_dn = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200, class_weight="balanced"))])
    model_up.fit(X, y_up)
    model_dn.fit(X, y_dn)
    return {"ok": True, "model_up": model_up, "model_dn": model_dn, "feature_cols": list(X.columns)}

def _predict_continuation(pair_label: str, interval: str, df: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"p_up": float("nan"), "p_dn": float("nan"), "ok": False}
    fitted = _fit_continuation_models(pair_label, interval, df, horizon=horizon)
    if not fitted.get("ok"):
        return {"p_up": float("nan"), "p_dn": float("nan"), "ok": False}
    cols = fitted["feature_cols"]
    X_latest, _, _ = _make_continuation_dataset(df, horizon=horizon)
    if X_latest.empty:
        # build latest row manually
        X_latest = _make_continuation_dataset(df, horizon=1)[0]
    if X_latest.empty:
        return {"p_up": float("nan"), "p_dn": float("nan"), "ok": False}
    row = X_latest[cols].iloc[[-1]]
    p_up = float(fitted["model_up"].predict_proba(row)[0,1])
    p_dn = float(fitted["model_dn"].predict_proba(row)[0,1])
    return {"p_up": p_up, "p_dn": p_dn, "ok": True}

def _similar_pattern_search(df: pd.DataFrame, window: int = 30, horizon: int = 5, topk: int = 5) -> pd.DataFrame:
    """Find similar windows using simple normalized indicator sequence distance."""
    if df is None or df.empty or len(df) < window + horizon + 100:
        return pd.DataFrame()
    d = df.copy()
    c = d["Close"]
    atr14 = _atr(d, 14)
    rsi14 = _rsi(c, 14)
    adx14 = _adx(d, 14)
    # features per bar
    feat = pd.DataFrame({
        "ret": c.pct_change().fillna(0.0),
        "rsi": (rsi14-50)/50.0,
        "adx": (adx14/50.0).clip(0,2),
        "atrp": (atr14/c).fillna(method="bfill").fillna(0.0),
    }).fillna(0.0)
    # build last window vector
    last = feat.iloc[-window:].to_numpy().reshape(-1)
    last = (last - last.mean()) / (last.std() + 1e-9)
    candidates = []
    # exclude overlap near end
    max_i = len(feat) - window - horizon - 5
    for i in range(60, max_i):
        seg = feat.iloc[i:i+window].to_numpy().reshape(-1)
        seg = (seg - seg.mean()) / (seg.std() + 1e-9)
        # cosine distance
        num = float(np.dot(last, seg))
        den = float((np.linalg.norm(last)+1e-9)*(np.linalg.norm(seg)+1e-9))
        sim = num/den
        fwd_ret = float((c.iloc[i+window+horizon-1] - c.iloc[i+window-1]) / (atr14.iloc[i+window-1] + 1e-9))
        candidates.append((sim, i, d.index[i+window-1], fwd_ret))
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda x: x[0], reverse=True)
    rows=[]
    for sim, i, end_dt, fwdR in candidates[:topk]:
        rows.append({"similarity": sim, "pattern_end": str(end_dt), "fwd_R_atr": fwdR})
    return pd.DataFrame(rows)

def _load_rl_agent(path: str) -> Optional[RLExitAgent]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return RLExitAgent.from_json(obj)
    except Exception:
        return None
    return None

def _save_rl_agent(path: str, agent: RLExitAgent) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(agent.to_json(), f, ensure_ascii=False)
        return True
    except Exception:
        return False


def _wfa_select_rl_coeffs(df: pd.DataFrame,
                          phase_series: pd.Series,
                          trend_strength_series: pd.Series,
                          p_up_series: pd.Series,
                          p_dn_series: pd.Series,
                          grid: Optional[list[dict]] = None,
                          splits: int = 3,
                          train_episodes: int = 2200,
                          eval_episodes: int = 800,
                          max_horizon: int = 20,
                          seed: int = 13) -> dict:
    """Walk-Forward Analysis (WFA) to pick RL reward coefficients.
    Returns best params and fold stats. Keeps it lightweight for Streamlit Cloud.
    """
    try:
        if df is None or df.empty or len(df) < 600:
            return {"ok": False, "error": "not_enough_data"}
        if grid is None:
            grid = []
            for dd in [0.05, 0.10, 0.15, 0.25, 0.35]:
                for tp in [0.0, 0.005, 0.01, 0.02]:
                    for atrm in [1.2, 1.5, 1.8]:
                        grid.append({"dd_penalty": dd, "time_penalty": tp, "atr_mult_stop": atrm})
        # build fold indices (expanding window)
        n = len(df)
        fold = max(120, n // (splits + 2))
        # train ends at: fold*(i+2), test next fold
        folds = []
        for i in range(splits):
            tr_end = fold * (i + 2)
            te_end = min(n, tr_end + fold)
            if te_end - tr_end < 60 or tr_end < 200:
                continue
            folds.append((0, tr_end, tr_end, te_end))
        if not folds:
            # fallback single split 70/30
            tr_end = int(n * 0.7)
            folds = [(0, tr_end, tr_end, n)]

        best = None
        all_rows = []
        for params in grid:
            fold_scores = []
            for (tr0, tr1, te0, te1) in folds:
                df_tr = df.iloc[tr0:tr1].copy()
                df_te = df.iloc[te0:te1].copy()
                ph_tr = phase_series.iloc[tr0:tr1]
                ts_tr = trend_strength_series.iloc[tr0:tr1]
                pu_tr = p_up_series.iloc[tr0:tr1]
                pd_tr = p_dn_series.iloc[tr0:tr1]

                ph_te = phase_series.iloc[te0:te1]
                ts_te = trend_strength_series.iloc[te0:te1]
                pu_te = p_up_series.iloc[te0:te1]
                pd_te = p_dn_series.iloc[te0:te1]

                ag = RLExitAgent()
                r = ag.train_from_price(
                    df_tr, ph_tr, ts_tr, pu_tr, pd_tr,
                    episodes=int(train_episodes),
                    max_horizon=int(max_horizon),
                    atr_mult_stop=float(params["atr_mult_stop"]),
                    dd_penalty=float(params["dd_penalty"]),
                    time_penalty=float(params["time_penalty"]),
                    seed=int(seed + 17),
                )
                if not r.get("ok"):
                    continue
                ev = ag.evaluate_policy(
                    df_te, ph_te, ts_te, pu_te, pd_te,
                    episodes=int(eval_episodes),
                    max_horizon=int(max_horizon),
                    atr_mult_stop=float(params["atr_mult_stop"]),
                    dd_penalty=float(params["dd_penalty"]),
                    time_penalty=float(params["time_penalty"]),
                    seed=int(seed + 19),
                )
                if not ev.get("ok"):
                    continue
                fold_scores.append(float(ev["avg_reward"]))
            if not fold_scores:
                continue
            score = float(np.mean(fold_scores))
            row = {"score": score, **params}
            all_rows.append(row)
            if (best is None) or (score > best["score"]):
                best = row

        if best is None:
            return {"ok": False, "error": "wfa_failed", "tried": int(len(grid))}
        df_rank = pd.DataFrame(all_rows).sort_values("score", ascending=False).head(12)
        return {"ok": True, "best": best, "top": df_rank.to_dict(orient="records"), "folds": folds, "tried": int(len(grid))}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _rl_exit_reco(agent: Optional[RLExitAgent], phase: str, trend_strength: float, p_up: float, p_dn: float, unrealized_R: float, dd_R: float = 0.0) -> Dict[str, Any]:
    if agent is None:
        return {"action": "HOLD", "note": "RL model not trained yet", "q": {}, "edge": float(p_up - p_dn), "state": ""}
    return agent.act(phase, trend_strength, p_up, p_dn, unrealized_R, dd_R)

def _compute_phase_strength_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Vectorized phase + strength per bar (no ML, deterministic).

    Phase rules (approx):
      - UP: ema20>ema50 and adx high
      - DOWN: ema20<ema50 and adx high
      - RANGE: adx low
      - TRANSITION: otherwise
    Strength: normalized ADX (0..1) blended with ema slope.
    """
    d = _coerce_ohlc(df.copy())
    if d.empty:
        return pd.Series(dtype=str), pd.Series(dtype=float)
    c = d["Close"].astype(float)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    adx = _adx(d, 14).astype(float)
    adx_n = (adx / 40.0).clip(0.0, 1.0).fillna(0.0)
    slope = (ema20 - ema20.shift(5)) / (d["ATR14"].astype(float).replace(0, np.nan) if "ATR14" in d.columns else (c.rolling(14).std() + 1e-9))
    slope = slope.fillna(0.0).clip(-2.0, 2.0)
    slope_n = (slope.abs() / 2.0).clip(0.0, 1.0)
    strength = (0.65 * adx_n + 0.35 * slope_n).clip(0.0, 1.0)

    phase = pd.Series(index=d.index, dtype=str)
    is_range = adx.fillna(0.0) < 16.0
    is_up = (ema20 > ema50) & (~is_range)
    is_dn = (ema20 < ema50) & (~is_range)
    phase[is_range] = "RANGE"
    phase[is_up] = "UP_TREND"
    phase[is_dn] = "DOWN_TREND"
    phase[phase.isna()] = "TRANSITION"
    return phase, strength

def _compute_cont_p_series(df: pd.DataFrame, horizon: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Cheap continuation probability proxy series based on momentum & volatility.
    This is used only as a feature for RL training; the 'official' continuation model remains separate.
    """
    d = _coerce_ohlc(df.copy())
    if d.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    c = d["Close"].astype(float)
    r = c.pct_change().fillna(0.0)
    mom = (c - c.shift(horizon)) / (c.rolling(14).std() + 1e-9)
    mom = mom.fillna(0.0).clip(-3.0, 3.0)
    # sigmoid to (0..1)
    p_up = 1.0 / (1.0 + np.exp(-1.2 * mom))
    p_dn = 1.0 - p_up
    return pd.Series(p_up, index=d.index), pd.Series(p_dn, index=d.index)

def _fetch_from_stooq(pair_label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"source": "stooq", "ok": False, "error": None, "interval_used": "1d"}
    sym = _pair_label_to_stooq_symbol(pair_label)
    if not sym:
        meta["error"] = "unsupported_pair_for_stooq"
        return pd.DataFrame(), meta
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        d = pd.read_csv(url)
        if "Date" not in d.columns:
            meta["error"] = "bad_csv"
            return pd.DataFrame(), meta
        d["Date"] = pd.to_datetime(d["Date"])
        d = d.set_index("Date").sort_index()
        d = _coerce_ohlc(d)
        if d.empty:
            meta["error"] = "empty_after_parse"
            return pd.DataFrame(), meta
        meta["ok"] = True
        return d, meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return pd.DataFrame(), meta

@st.cache_data(ttl=60 * 60)  # 1 hour
def fetch_price_history(pair_label: str, symbol: str, period: str, interval: str, prefer_stooq: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust price fetch:
      - prefer stooq for daily (reduces yfinance rate-limit on Streamlit Cloud)
      - else try yfinance, then fallback to stooq daily
    """
    meta: Dict[str, Any] = {"source": "yfinance", "ok": False, "error": None, "fallback": None, "interval_used": interval}

    # Prefer stooq for daily / multi-scan
    if prefer_stooq or interval == "1d":
        df_s, m_s = _fetch_from_stooq(pair_label)
        if not df_s.empty and m_s.get("ok"):
            meta.update({"source": "stooq", "ok": True, "fallback": None, "interval_used": "1d"})
            return df_s, meta
        meta["fallback"] = m_s

    if yf is not None:
        last_err = None
        for attempt in range(2):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                df = _coerce_ohlc(df)
                if df.empty:
                    last_err = "empty_df"
                    raise RuntimeError("empty_df")
                meta["ok"] = True
                return df, meta
            except YFRateLimitError:
                last_err = "YFRateLimitError"
                break
            except Exception as e:
                last_err = f"{type(e).__name__}:{e}"
                time.sleep(0.6 * (attempt + 1))
        meta["error"] = last_err
    else:
        meta["error"] = "yfinance_not_installed"

    # fallback stooq
    df2, m2 = _fetch_from_stooq(pair_label)
    meta["fallback"] = m2
    if not df2.empty and m2.get("ok"):
        meta["source"] = "stooq"
        meta["ok"] = True
        meta["interval_used"] = "1d"
        return df2, meta

    return pd.DataFrame(), meta

@st.cache_data(ttl=60 * 20)
def fetch_external(pair_label: str, keys: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Integrated external features. Never crashes.
    """
    base = {
        "news_sentiment": 0.0,
        "cpi_surprise": 0.0,
        "nfp_surprise": 0.0,
        "rate_diff_change": 0.0,
        "macro_risk_score": 0.0,
        "global_risk_index": 0.0,
        "war_probability": 0.0,
        "financial_stress": 0.0,
        "gdelt_war_count_1d": 0.0,
        "gdelt_finance_count_1d": 0.0,
        "vix": float("nan"),
        "dxy": float("nan"),
        "us10y": float("nan"),
        "jp10y": float("nan"),
        "av_inflation": float("nan"),
        "av_unemployment": float("nan"),
        "av_fed_funds_rate": float("nan"),
        "av_treasury_10y": float("nan"),
        "av_macro_risk": 0.0,
    }
    if data_layer is None:
        return base, {"ok": False, "error": "data_layer_import_failed"}
    if not hasattr(data_layer, "fetch_external_features"):
        return base, {"ok": False, "error": "data_layer_missing_fetch_external_features", "file": getattr(data_layer, "__file__", "unknown")}
    try:
        feats, meta = data_layer.fetch_external_features(pair_label, keys=keys)  # type: ignore
        out = base.copy()
        out.update({k: float(v) for k, v in (feats or {}).items() if k in out and v is not None})
        return out, meta if isinstance(meta, dict) else {"ok": True}
    except Exception as e:
        return base, {"ok": False, "error": f"fetch_external_failed:{type(e).__name__}", "detail": str(e)}


def _parts_status_table(meta: Dict[str, Any]) -> pd.DataFrame:
    parts = (meta or {}).get("parts", {}) if isinstance(meta, dict) else {}
    rows: List[Dict[str, Any]] = []

    def _summarize_detail(detail: Any) -> Tuple[Optional[bool], str, Optional[str]]:
        """Return (ok_override, detail_str, err_override)"""
        if isinstance(detail, dict) and any(isinstance(v, dict) and ('ok' in v) for v in detail.values()):
            nested_ok_bits = []
            nested_errs = []
            all_ok = True
            any_ok_field = False
            for k, v in detail.items():
                if not isinstance(v, dict):
                    continue
                vok = v.get("ok", None)
                if vok is not None:
                    any_ok_field = True
                    all_ok = all_ok and bool(vok)
                nested_ok_bits.append(f"{k}:{'ok' if vok else 'ng'}")
                if v.get("error"):
                    nested_errs.append(f"{k}:{v.get('error')}")
            ok_override = all_ok if any_ok_field else None
            d = ", ".join(nested_ok_bits)[:120]
            e = "; ".join(nested_errs)[:160] if nested_errs else None
            return ok_override, d, e

        if isinstance(detail, dict):
            # compact key:val view
            bits = []
            for k, v in list(detail.items())[:12]:
                if isinstance(v, dict):
                    # keys row etc.
                    if "present" in v:
                        mark = "✓" if v.get("present") else "×"
                        used = v.get("used", None)
                        if used:
                            u = str(used)
                            tag = {"keys": "ui", "secrets": "sec", "env": "env"}.get(u, u)
                            bits.append(f"{k}:{mark}({tag})")
                        else:
                            bits.append(f"{k}:{mark}")
                    elif "ok" in v:
                        bits.append(f"{k}:{'ok' if v.get('ok') else 'ng'}")
                    else:
                        bits.append(f"{k}:…")
                else:
                    bits.append(f"{k}:{v}")
            return None, ", ".join(bits)[:120], None

        if isinstance(detail, str):
            return None, detail[:120], None

        return None, "", None

    if isinstance(parts, dict):
        for name, p in parts.items():
            ok: Optional[bool] = None
            err: Optional[str] = None
            extra = ""

            if isinstance(p, dict):
                ok = p.get("ok")
                err = p.get("error")

                # Prefer p['detail'] for summary
                detail = p.get("detail", None)

                ok2, extra2, err2 = _summarize_detail(detail)
                if extra2:
                    extra = extra2
                # If nested says ng but ok was True, override to False (conservative)
                if ok2 is not None:
                    ok = ok2 if ok is None or ok is True else ok
                if err2 and not err:
                    err = err2

                # Fallback: summarize nested dicts directly in p if detail empty
                if not extra:
                    nested = {k: v for k, v in p.items() if isinstance(v, dict)}
                    ok3, extra3, err3 = _summarize_detail(nested)
                    if extra3:
                        extra = extra3
                    if ok3 is not None:
                        ok = ok3 if ok is None or ok is True else ok
                    if err3 and not err:
                        err = err3

                n = p.get("n", None)
                if n is not None:
                    extra = (extra + f" n={n}").strip()
            rows.append({"source": name, "ok": ok, "error": err, "detail": extra})

    if not rows:
        rows = [{"source": "external", "ok": False, "error": "no_meta_parts", "detail": ""}]
    return pd.DataFrame(rows)





def _recommend_min_expected_R_from_audit(rows: List[Dict[str, Any]], target_trade_rate: float = 0.20) -> Dict[str, Any]:
    """Recommend base min_expected_R using ev_audit rows.

    dynamic_threshold = base_threshold * mult
    mult = 1 + 0.8*macro + 1.0*global + 0.6*war + 0.6*fin

    For EV_RAW gate, TRADE if:
      ev_raw >= base_threshold * mult
    => base_threshold <= ev_raw / mult

    To target trade rate r, choose base_threshold as the (1-r) quantile
    of (ev_raw/mult) over recent rows.
    """
    try:
        r = float(target_trade_rate)
        r = max(0.01, min(0.80, r))
    except Exception:
        r = 0.20

    vals: List[float] = []
    for row in rows:
        try:
            ev_raw = float(row.get("ev_raw", ""))
            macro = float(row.get("macro_risk_score", 0.0) or 0.0)
            global_risk = float(row.get("global_risk_index", 0.0) or 0.0)
            war = float(row.get("war_probability", 0.0) or 0.0)
            fin = float(row.get("financial_stress", 0.0) or 0.0)
            mult = 1.0 + 0.8*macro + 1.0*global_risk + 0.6*war + 0.6*fin
            mult = max(1.0, float(mult))
            vals.append(ev_raw / mult)
        except Exception:
            continue

    if len(vals) < 5:
        return {"ok": False, "n": len(vals), "recommended": None, "reason": "ログ件数が不足（5件以上で暫定推奨）"}

    vals_sorted = sorted(vals)
    q = 1.0 - r
    k = int(round(q * (len(vals_sorted) - 1)))
    k = max(0, min(len(vals_sorted) - 1, k))
    rec = float(vals_sorted[k])

    # Guard rails for swing
    rec = float(max(0.01, min(0.12, rec)))

    return {"ok": True, "n": len(vals), "recommended": rec, "target_trade_rate": r}

def _style_defaults(style_name: str) -> Dict[str, Any]:
    # Presets: avoid manual tuning
    if style_name == "保守":
        return {"min_expected_R": 0.12, "horizon_days": 7}
    if style_name == "攻撃":
        return {"min_expected_R": 0.03, "horizon_days": 5}
    return {"min_expected_R": 0.07, "horizon_days": 7}  # 標準

def _build_ctx(pair_label: str, df: pd.DataFrame, feats: Dict[str, float], horizon_days: int, min_expected_R: float, style_name: str,
               governor_cfg: Dict[str, Any]) -> Dict[str, Any]:
    indicators = _compute_indicators_compat(df)
    ctx: Dict[str, Any] = {}
    ctx.update(indicators)
    ctx.update(feats)
    # --- AI Trend Engine features (phase / continuation / similar patterns) ---
    try:
        phase_info = _phase_classify(df)
        cont = _predict_continuation(pair_label, "1d", df, horizon=max(3, int(horizon_days)))
        patt = _similar_pattern_search(df, window=30, horizon=max(3, int(horizon_days)), topk=5)
        ctx["phase"] = phase_info.get("phase", "UNKNOWN")
        ctx["trend_strength"] = float(phase_info.get("trend_strength", 0.0) or 0.0)
        ctx["adx14"] = float(phase_info.get("adx", float("nan")))
        ctx["p_cont_up"] = float(cont.get("p_up", float("nan")))
        ctx["p_cont_dn"] = float(cont.get("p_dn", float("nan")))
        # stash patterns for UI
        ctx["_similar_patterns_df"] = patt
    except Exception:
        ctx["phase"] = "UNKNOWN"
        ctx["trend_strength"] = 0.0
        ctx["p_cont_up"] = float("nan")
        ctx["p_cont_dn"] = float("nan")
    ctx["pair_label"] = pair_label
    ctx["pair_symbol"] = _pair_label_to_symbol(pair_label)
    ctx["price"] = float(df["Close"].iloc[-1])
    ctx["horizon_days"] = int(horizon_days)
    ctx["min_expected_R"] = float(min_expected_R)
    ctx["style_name"] = style_name
    # Capital Governor inputs (user provided / optional)
    ctx.update(governor_cfg)
    # stash recent bars for UI (target zones / reference chart)
    try:
        ctx["_df"] = df.tail(320).copy()
    except Exception:
        pass
    return ctx

def _dominant_state(state_probs: Dict[str, Any]) -> str:
    if not isinstance(state_probs, dict) or not state_probs:
        return "—"
    try:
        return max(state_probs.items(), key=lambda kv: float(kv[1]))[0]
    except Exception:
        return "—"


def _render_top_trade_panel(pair_label: str, plan: Dict[str, Any], current_price: float):
    """
    運用者が「いま実行すべきか」を迷わないための最上段パネル。
    - NO_TRADE は「エントリー禁止」ではなく「この条件では期待値が薄い/危険寄りなので見送り推奨」
    - ロット係数は“推奨値（表示）”。実際の発注に反映するかは運用者が決める。
    """
    decision = str(plan.get("decision", "NO_TRADE"))
    decision_jp = _jp_decision(decision)
    expected_R_ev = float(plan.get("expected_R_ev") or 0.0)
    p_win_ev = float(plan.get("p_win_ev") or 0.0)
    confidence = float(plan.get("confidence") or 0.0)
    dyn_th = float(plan.get("dynamic_threshold") or 0.0)
    lot_mult = float(plan.get("_lot_multiplier_reco") or 1.0)

    # override info (manual kill switch / outage)
    orig = plan.get("_decision_original")
    ovr = plan.get("_decision_override_reason")

    # モバイル前提：6列だと潰れるため、2列×3段で表示
    r1c1, r1c2 = st.columns(2)
    r1c1.metric("ペア", pair_label)
    r1c2.metric("最終判断", decision_jp)

    r2c1, r2c2 = st.columns(2)
    r2c1.metric("期待値EV (R)", f"{expected_R_ev:+.3f}")
    r2c2.metric("動的閾値", f"{dyn_th:.3f}")

    # 判定に使ったEVが「生EV」か「リスク調整後EV」かを明示（検証用）
    ev_raw = plan.get("expected_R_ev_raw", None)
    ev_adj = plan.get("expected_R_ev_adj", None)
    gate_mode = plan.get("gate_mode", None)
    if ev_raw is not None and ev_adj is not None and gate_mode:
        st.caption(f"判定モード: {gate_mode} / EV(生)={ev_raw:+.3f} / EV(調整後)={ev_adj:+.3f}")

    r3c1, r3c2 = st.columns(2)
    r3c1.metric("信頼度", f"{confidence:.2f}")
    # SBI想定：最小1建・小数建て不可のため、ロットは「整数建玉」で表示
    sbi_lots = int(plan.get("_sbi_exec_lots") or (0 if decision == "NO_TRADE" else 1))
    r3c2.metric("推奨建玉（SBI）", f"{sbi_lots}建")

    # SBIガードの数値根拠（必要なときだけ）
    sbi_info = plan.get("_sbi") if isinstance(plan.get("_sbi"), dict) else None
    if isinstance(sbi_info, dict) and sbi_info:
        try:
            ev_sbi = float(sbi_info.get("ev_sbi") or 0.0)
            risk_x = float(sbi_info.get("risk_x") or 1.0)
            ev_margin = float(sbi_info.get("ev_margin") or 0.0)
            req = float(sbi_info.get("ev_margin_req") or 0.0)
            if decision == "NO_TRADE":
                st.caption(f"SBI最小1建ガード：実質リスク×{risk_x:.2f} / SBI補正EV {ev_sbi:+.3f} / EV余裕 {ev_margin:+.3f}（必要 {req:.3f}）")
            else:
                # TRADEでも縮退推奨が残る場合は注意喚起だけ表示
                if float(plan.get("_lot_multiplier_reco") or 1.0) < 0.80:
                    st.caption(f"SBI参考：縮退推奨 係数{float(plan.get('_lot_multiplier_reco') or 1.0):.3f}（実質リスク×{risk_x:.2f}）")
        except Exception:
            pass

    if orig is not None and ovr:
        st.warning(f"判断は上書きされています：{_jp_decision(str(orig))} → {decision_jp}（理由：{ovr}）")

    st.caption(
        "EV (R) は『損切り幅=1R』基準の期待値です。"
        "動的閾値は危険時に上がります（見送りが増えるのは仕様）。"
        "縮退係数（参考）は“連続補正”で急変しません。SBIは最小1建のため係数は警戒度として見てください。"
    )


    if decision != "NO_TRADE":
        # ---- 表示は運用者向けに日本語化（略語は出さない） ----
        direction = str(plan.get("direction", "") or "").upper()
        side_raw = plan.get("side", "—")
        entry = plan.get("entry", None)
        sl = plan.get("stop_loss", None)
        tp = plan.get("take_profit", None)

        # 発注種別（成行/指値/逆指値）は、現在値との位置関係から推定（スイング運用の実務表記）
        entry_kind_src = str(plan.get("entry_type") or plan.get("order_type") or "")
        entry_kind = entry_kind_src if _jp_order_kind(entry_kind_src) != "—" else _infer_entry_order_kind(direction, float(entry or current_price), float(current_price))
        entry_kind_jp = _jp_order_kind(entry_kind)

        # 発注方式（IFD/OCO/IFDOCO）を明示
        has_tp = (tp is not None) and (float(tp or 0.0) != 0.0)
        has_sl = (sl is not None) and (float(sl or 0.0) != 0.0)
        scheme = _jp_order_scheme(True, has_tp, has_sl)

        st.success("✅ エントリー候補（このアプリは発注しません。発注は運用者が実行）")

        # 信頼度が中程度以下のときは注意喚起（急にエントリーになった不安を軽減）
        if confidence < 0.55:
            st.info(
                "⚠️ 信頼度は中程度です。エントリー可でも“確定”ではありません。"
                "（フェーズ/継続確率/構造判定が更新され、条件を満たした可能性があります）"
            )

        st.markdown(f"""
- **売買**: {_jp_side(side_raw)}
- **エントリー注文**: {entry_kind_jp}
- **注文方式**: {scheme}（エントリー成立後、TP/SLは **OCO** で同時に置く想定）
- **エントリー価格**: {_fmt_price(entry)}
- **損切り(SL)**: {_fmt_price(sl)}
- **利確(TP)**: {_fmt_price(tp)}
""")
        st.caption(f"参考：勝率推定 p_win={p_win_ev:.2f}（あくまでモデル推定）。")
    else:
        st.warning("⏸ 見送り（NO_TRADE）")

        # --- 要件: NO_TRADEでも「注文方式（指値/逆指値/IFD/OCO/IFDOCO）」を必ず表示 ---
        direction = str(plan.get("direction", "") or "").upper()
        side_raw = plan.get("side", "—")
        entry = plan.get("entry", None)
        sl = plan.get("stop_loss", None)
        tp = plan.get("take_profit", None)

        entry_kind_src = str(plan.get("entry_type") or plan.get("order_type") or "")
        entry_kind = entry_kind_src if _jp_order_kind(entry_kind_src) != "—" else _infer_entry_order_kind(direction, float(entry or current_price), float(current_price))
        entry_kind_jp = _jp_order_kind(entry_kind)

        has_tp = (tp is not None) and (float(tp or 0.0) != 0.0)
        has_sl = (sl is not None) and (float(sl or 0.0) != 0.0)
        scheme = _jp_order_scheme(True, has_tp, has_sl)

        st.markdown(f"""
- **売買候補（参考）**: {_jp_side(side_raw)}
- **エントリー注文（参考）**: {entry_kind_jp}
- **注文方式（参考）**: {scheme}
- **参考エントリー価格**: {_fmt_price(entry)}
- **参考損切り(SL)**: {_fmt_price(sl)}
- **参考利確(TP)**: {_fmt_price(tp)}
""")


        # 表示は「主因」を1行に統一（重複を避ける）
        why = str(plan.get("why", "") or "").strip()
        veto = plan.get("veto_reasons", None)

        primary = ""
        if why:
            primary = why
        elif isinstance(veto, (list, tuple)) and len(veto) > 0:
            primary = str(veto[0])
        else:
            primary = "—"

        st.markdown(f"**主因**: {primary}")
        st.caption(f"判定値：EV {expected_R_ev:+.3f}  <  動的閾値 {dyn_th:.3f}")

        with st.expander("詳細（見送り理由）", expanded=False):
            if why:
                st.markdown(f"- **理由（why）**: {why}")
            if isinstance(veto, (list, tuple)) and len(veto) > 0:
                st.markdown("- **veto内訳**:")
                st.write(list(veto))



def _render_risk_dashboard(plan: Dict[str, Any], feats: Dict[str, float], ext_meta: Optional[Dict[str, Any]] = None):
    """
    運用者が「危険度」と「データ品質」を見て、実行/見送り/ロット縮小を判断できるパネル。
    数字だけで終わらず、日本語の“意味”と“次の行動”をセットで出す。
    さらに、外部APIの取得状態（401/403/429/timeout等）を同じ場所で確認できるようにする。
    """
    bs = plan.get("black_swan", {}) or {}
    gov = plan.get("governor", {}) or {}
    overlay = plan.get("overlay_meta", {}) or {}
    ext_meta = ext_meta or {}

    global_risk = float(feats.get("global_risk_index", 0.0) or 0.0)
    war = float(feats.get("war_probability", 0.0) or 0.0)
    fin = float(feats.get("financial_stress", 0.0) or 0.0)
    macro = float(feats.get("macro_risk_score", 0.0) or 0.0)
    news = float(feats.get("news_sentiment", 0.0) or 0.0)

    bs_flag = bool(bs.get("flag", False))
    bs_level = str(bs.get("level", "") or "")
    gov_enabled = bool(gov.get("enabled", True))

    # data quality
    q_level = ""
    q_reasons: List[str] = []
    try:
        parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}
        q = parts.get("quality", {}) if isinstance(parts, dict) else {}
        qd = (q.get("detail", {}) or {}) if isinstance(q, dict) else {}
        q_level = str(qd.get("level", "") or "")
        q_reasons = [str(x) for x in (qd.get("reasons", []) or [])]
    except Exception:
        q_level = ""

    st.markdown("### リスク/ガード（運用判断）")

    # Quality banner
    if q_level == "OUTAGE":
        st.error("🚨 外部データ品質：OUTAGE（主要ソースが取れていない可能性）")
        if q_reasons:
            st.caption("理由: " + " / ".join(q_reasons[:6]))
    elif q_level == "DEGRADED":
        st.warning("⚠️ 外部データ品質：DEGRADED（一部ソース欠け）")
        if q_reasons:
            st.caption("理由: " + " / ".join(q_reasons[:6]))
    else:
        st.success("✅ 外部データ品質：OK（主要ソースが揃っています）")

    # Main risk meters
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("相場全体リスク", f"{global_risk:.2f}", help="0〜1。高いほど荒れやすい")
    c2.metric("地政学リスク", f"{war:.2f}", help="0〜1。戦争/紛争の悪化確率（推定）")
    c3.metric("金融ストレス", f"{fin:.2f}", help="0〜1。信用/金融不安の強さ（推定）")
    c4.metric("マクロ不安", f"{macro:.2f}", help="0〜1。VIX/DXY/金利等からの合成")
    c5.metric("ニュースムード", f"{news:.2f}", help="0〜1。高いほどネガ寄り（定義は実装依存）")

    # ---- event / weekend risk (pair-specific; returned from logic in plan["_ctx"]) ----
    ctxp = plan.get("_ctx", {}) if isinstance(plan.get("_ctx", {}), dict) else {}
    try:
        ev_factor = float(ctxp.get("event_risk_factor", 0.0) or 0.0)
        ev_score = float(ctxp.get("event_risk_score", 0.0) or 0.0)
    except Exception:
        ev_factor, ev_score = 0.0, 0.0
    ev_window = bool(ctxp.get("event_window_high", False))
    next_high = ctxp.get("event_next_high_hours", None)
    try:
        weekend_r = float(ctxp.get("weekend_risk", 0.0) or 0.0)
    except Exception:
        weekend_r = 0.0
    feed_status = str(ctxp.get("event_feed_status", "") or "")
    feed_err = str(ctxp.get("event_feed_error", "") or "")

    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("イベント近接リスク", f"{ev_factor:.2f}", help="0〜1。高/中インパクト指標が近いほど上がります（無料カレンダー）。")
    ec2.metric("高インパクト窓", "ON" if ev_window else "OFF", help="発表前後は強制見送り(設定ON時)。")
    ec3.metric("週末ギャップ", f"{weekend_r:.2f}", help="金曜夜〜週末はギャップ/窓開けリスクが上がります（簡易）。")

    if isinstance(next_high, (int, float)) and (next_high == next_high):
        st.caption(f"次の高インパクトまで: 約 {float(next_high):.1f} 時間")
    if feed_status and feed_status not in ("ok", "cache"):
        st.warning(f"イベントカレンダー取得: {feed_status}（イベントリスク=0扱い）")
        if feed_err:
            st.caption(f"理由: {feed_err}")

    st.caption(
        f"判定：相場全体={_bucket_01(global_risk)} / 地政学={_bucket_01(war)} / 金融={_bucket_01(fin)} / マクロ={_bucket_01(macro)}"
    )

    # Black swan / governor
    if bs_flag:
        st.error(f"🟥 Black Swan Guard: ON（{bs_level}）")
        rs = bs.get("reasons", []) or []
        if rs:
            st.write(rs)
    else:
        st.info("🟩 Black Swan Guard: OFF（通常）")

    if not gov_enabled:
        st.error("🛑 Capital Governor: 停止（DD/損失/連敗条件に抵触）")
        rs = gov.get("reasons", []) or []
        if rs:
            st.write(rs)
    else:
        st.info("✅ Capital Governor: OK（停止条件に非該当）")

    # Next action hint
    hint = _action_hint(global_risk, war, fin, macro, bs_flag, gov_enabled)
    st.markdown(f"#### 次のアクション（提案）\n- {hint}")
    st.caption(f"ガード設定: {str(guard_apply)} / SBI最小1建前提で『縮退不能』は自動で見送りに倒す場合があります。")

    # Overlay notes (debug-level)
    if isinstance(overlay, dict) and overlay:
        adj = overlay.get("risk_adjustment", {})
        if isinstance(adj, dict) and adj:
            st.caption(
                f"（内部補正）risk_adjustment: global={adj.get('global_risk')} war={adj.get('war')} fin={adj.get('fin')} macro={adj.get('macro')}"
            )

    # ---- status table (root-cause) ----
    try:
        show_diag_default = (str(q_level) in ("OUTAGE","DEGRADED"))
        with st.expander("🧪 外部データ取得ステータス（診断・原因究明）", expanded=show_diag_default or bool(show_debug)):
            st.markdown("#### 外部データ取得ステータス（0固定/異常が出たら最優先でここ）")
            st.caption("OKでも中身が空/一部失敗があり得ます。error / detail を必ず見ます。")
            df = _parts_status_table(ext_meta)

            # ---- Economic calendar (event guard) ----
            try:
                st.markdown("#### 📅 経済指標カレンダー（イベントリスク）")
                if feed_status in ("ok", "cache"):
                    st.success(f"カレンダー取得: {feed_status} / score={ev_score:.3f} / factor={ev_factor:.2f}")
                elif feed_status == "off":
                    st.info("カレンダー: OFF（設定でONにできます）")
                else:
                    st.warning(f"カレンダー取得: {feed_status or 'unknown'}（イベントリスク=0扱い）")
                    if feed_err:
                        st.caption(f"理由: {feed_err}")

                ev_list = ctxp.get("event_upcoming", []) if isinstance(ctxp, dict) else []
                if isinstance(ev_list, list) and ev_list:
                    # show top items in local time
                    rows = []
                    tzname = str(ctxp.get("event_timezone") or st.session_state.get("event_timezone","Asia/Tokyo") or "Asia/Tokyo")
                    try:
                        from zoneinfo import ZoneInfo
                        tz = ZoneInfo(tzname)
                    except Exception:
                        tz = None
                    for it in ev_list[:8]:
                        try:
                            dt_utc = str(it.get("dt_utc") or "")
                            dt = pd.to_datetime(dt_utc, utc=True, errors="coerce")
                            if pd.isna(dt):
                                continue
                            if tz is not None:
                                dt_local = dt.tz_convert(tz)
                                dt_show = dt_local.strftime("%m/%d %H:%M")
                            else:
                                dt_show = dt.strftime("%m/%d %H:%M UTC")
                            rows.append({
                                "時刻": dt_show,
                                "通貨": str(it.get("currency") or ""),
                                "重要度": str(it.get("impact") or ""),
                                "内容": str(it.get("title") or "")[:60],
                                "あと(h)": f"{float(it.get('hours') or 0.0):.1f}",
                            })
                        except Exception:
                            continue
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception:
                pass


            # Runtime fingerprint: proof of what is actually running
            runtime_line = ""
            try:
                dl_file = getattr(data_layer, "__file__", "IMPORT_FAILED")
                dl_build = getattr(data_layer, "DATA_LAYER_BUILD", "?")
                sha12 = "unknown"
                try:
                    import hashlib as _hashlib
                    with open(dl_file, "rb") as _f:
                        sha12 = _hashlib.sha256(_f.read()).hexdigest()[:12]
                except Exception:
                    pass
                runtime_line = f"main={APP_BUILD}, data_layer={dl_build}, sha12={sha12}, file={dl_file}"
                rows = [{"source": "runtime", "ok": True, "error": None, "detail": runtime_line}]
                df = pd.concat([pd.DataFrame(rows), df], ignore_index=True)
            except Exception:
                pass

            if runtime_line:
                st.text_area("実行中コード指紋（コピー用）", value=runtime_line, height=70)

            meaning = {
                "runtime": "実行中のコード指紋（build/sha）",
                "keys": "キー検出状況（secrets/ui/env）",
                "fred": "VIX/DXY/金利（マクロ系）",
                "te": "経済指標カレンダー（CPI/NFP等）",
                "gdelt": "紛争/金融ニュース量（無料）",
                "newsapi": "記事見出しセンチメント",
                "openai": "LLMによる地政学/危機推定（JSON）",
                "alpha_vantage": "マクロ補助（Alpha Vantage）",
                "risk_values": "リスク値（最終：運用判断の基準）",
                "quality": "外部データ品質（OK/DEGRADED/OUTAGE）",
                "build": "data_layer の build 文字列",
            }
            if "source" in df.columns:
                df["意味"] = df["source"].map(meaning).fillna("")

            st.dataframe(df, use_container_width=True)

            try:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 ステータスCSVをダウンロード（検証用）",
                    data=csv,
                    file_name="risk_status_export.csv",
                    mime="text/csv",
                )
            except Exception:
                pass


        with st.expander("📒 EV監査ログ（直近2週間：rawは良いのにadjで潰れる/閾値で潰れるを可視化）", expanded=False):
            rows = _ev_audit_load()

            # --- 自動最適化（min_expected_R 推奨） ---
            target_rate = 0.10 if st.session_state.get('priority_mode') == '勝率優先（見送り増）' else 0.20
            rec = _recommend_min_expected_R_from_audit(rows, target_trade_rate=target_rate) if rows else {"ok": False, "n": 0, "recommended": None, "reason": "ログなし"}
            colA, colB = st.columns([2, 1])
            with colA:
                if rec.get("ok"):
                    st.success(f"自動推奨 min_expected_R（目標TRADE率 {int(rec['target_trade_rate']*100)}%）: **{rec['recommended']:.3f}R**（n={rec['n']}）")
                else:
                    st.info(f"自動推奨 min_expected_R: まだ算出できません（{rec.get('reason','')} / n={rec.get('n',0)}）")
            with colB:
                if rec.get("ok"):
                    if st.button("この推奨値を適用", use_container_width=True):
                        st.session_state["min_expected_R_override"] = float(rec["recommended"])
                        st.toast(f"min_expected_R を {rec['recommended']:.3f} に固定しました（自動最適化）")
                if st.button("自動最適化を解除", use_container_width=True):
                    st.session_state["min_expected_R_override"] = None
                    st.toast("min_expected_R の自動最適化を解除しました（プリセットに戻ります）")
            st.caption("※自動最適化は『直近ログのEV_raw分布』から、目標TRADE率になるよう min_expected_R を推奨します。外部リスクでEVを削らず、閾値だけ調整します。")
            # --- /自動最適化 ---
            summ = _ev_audit_summary(rows, days=14)
            st.write(f"直近{summ['days']}日：判定 {summ['total']}件 / TRADE {summ['trade']}件 / NO_TRADE {summ['no_trade']}件")
            st.write(f"直近{summ['days']}日：**『EV_rawならTRADEだが EV_adjならNO_TRADE』= {summ['killed_by_adj']}件**（二重ブレーキ確認用）")

            # ---- Optimization hint (data-driven) ----
            if rows and summ["total"] >= 20:
                try:
                    import pandas as _pd
                    df_opt = _pd.DataFrame(rows)
                    # coerce
                    for c in ["ev_raw","ev_adj","dynamic_threshold","macro_risk_score","confidence"]:
                        if c in df_opt.columns:
                            df_opt[c] = _pd.to_numeric(df_opt[c], errors="coerce")
                    # focus on recent 14d already counted by summary: we re-filter by ts_utc
                    df_opt = df_opt.dropna(subset=["ev_raw","dynamic_threshold"])
                    # eligible: confident trend states (avoid pure risk_off)
                    if "dominant_state" in df_opt.columns:
                        df_opt = df_opt[df_opt["dominant_state"].astype(str).isin(["trend_up","trend_down","range","risk_off"])]

                    # Estimate base_threshold used by logic (swing-normal assumes: dyn = base*(1+0.20*macro))
                    if "macro_risk_score" in df_opt.columns and df_opt["macro_risk_score"].notna().any():
                        macro = df_opt["macro_risk_score"].fillna(0.0).clip(lower=0.0)
                        base_est = (df_opt["dynamic_threshold"] / (1.0 + 0.20*macro)).clip(lower=0.0)
                    else:
                        base_est = df_opt["dynamic_threshold"]

                    base_est_med = float(base_est.dropna().median()) if base_est.notna().any() else None

                    st.markdown("#### 🎯 しきい値（min_expected_R）の最適化提案（直近ログから推定）")
                    if base_est_med is not None:
                        st.caption(f"現在の推定 base_threshold（中央値）: {base_est_med:.3f}")

                    # Target: at least 4 TRADE / 14d (≈2/週) as a sane starting point
                    target_trades = st.number_input("目標TRADE件数（直近14日）", min_value=0, max_value=500, value=4, step=1)
                    # For each row, trade if base <= ev_raw / (1+0.20*macro)
                    if "macro_risk_score" in df_opt.columns and df_opt["macro_risk_score"].notna().any():
                        macro = df_opt["macro_risk_score"].fillna(0.0).clip(lower=0.0)
                        ratios = (df_opt["ev_raw"] / (1.0 + 0.20*macro)).replace([_pd.NA, _pd.NaT], _pd.NA)
                    else:
                        ratios = df_opt["ev_raw"]

                    ratios = _pd.to_numeric(ratios, errors="coerce").dropna()
                    ratios = ratios[ratios > 0]  # only meaningful positive EV_raw

                    if len(ratios) >= 10 and target_trades > 0:
                        ratios_sorted = ratios.sort_values(ascending=False).reset_index(drop=True)
                        k = int(min(target_trades, len(ratios_sorted)))
                        reco_base = float(ratios_sorted.iloc[k-1])
                        # simulate
                        sim_trades = int((ratios >= reco_base).sum())
                        st.write(f"推奨 min_expected_R（ベース）: **{reco_base:.3f}** （この値なら直近ログ上の予測TRADE: {sim_trades}件/14日）")
                        st.caption("※これは『過去ログに対するシミュレーション』です。市場が変われば変動します。")
                        if st.button("この推奨値を『min_expected_R』に適用（診断用）"):
                            st.session_state["min_expected_R"] = float(reco_base)
                            st.success("min_expected_R を推奨値に更新しました。画面を再計算してください。")
                    else:
                        st.caption("ログがまだ少ないため、推奨値の推定ができません（目安：10件以上）。")
                except Exception:
                    st.caption("最適化提案の計算に失敗しました（ログ形式の違いの可能性）。")
            # ---- /Optimization hint ----
            if rows:
                st.caption("※直近200件を表示。必要ならCSVをダウンロードして集計してください。")
                # show latest first
                import pandas as _pd
                df_a = _pd.DataFrame(list(reversed(rows))[:200])
                st.dataframe(df_a, use_container_width=True)
                try:
                    csv_a = _pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 EV監査ログCSVをダウンロード（ev_audit.csv）",
                        data=csv_a,
                        file_name="ev_audit.csv",
                        mime="text/csv",
                    )
                except Exception:
                    pass
            else:
                st.info("監査ログはまだありません。判定を数回実行すると蓄積されます（デプロイし直すと消える場合があります）。")

    
    except Exception:
        # status table is best-effort; never break trading UI
        pass

# =========================
# UI
# =========================
st.set_page_config(page_title="FX EV判断", layout="centered", initial_sidebar_state="collapsed")

# --- Global defaults to avoid NameError under Streamlit top-down execution ---
if "pair_label" not in st.session_state:
    st.session_state["pair_label"] = "USD/JPY"
pair_label = st.session_state.get("pair_label", "USD/JPY")

st.title("FX EV判断")

with st.sidebar:
    st.header("運用操作（見る順）")
    st.caption("普段は上から順に。『安全/診断/詳細』は折りたたんであります。")

    mode = st.selectbox("モード", ["相場全体から最適ペアを自動抽出（推奨）", "単一ペア最適化（徹底）"], index=0)
    trade_axis = st.selectbox("時間軸（保有期間）", ["スイング（1週〜1ヶ月）", "中長期（1〜3ヶ月）"], index=0)
    style_name = st.selectbox("運用スタイル（見送りライン）", ["標準", "保守", "攻撃"], index=0)
    priority = st.selectbox("優先度", ["バランス（推奨）", "勝率優先（見送り増）"], index=0)
    st.session_state["trend_assist_enable"] = st.checkbox("🤖 トレンド捕獲アシスト（実験）", value=False, help="強いトレンド局面では期待値に小さな加点を入れて、取り逃しを減らします。")
    st.session_state['priority_mode'] = priority

    # 時間軸プリセット（詳細設定で上書き可）
    # ※「ポジショントレード」はUI名ではなく“保有期間”の呼び方です。1ヶ月寄りなら「中長期」や interval=1wk を推奨。
    if "中長期" in trade_axis:
        period = "max"
        interval = "1wk"
        horizon_mode = "月（推奨）"
        horizon_days = 30
    else:  # スイング（1週〜1ヶ月）
        period = "10y"
        interval = "1d"
        horizon_mode = "週〜月（推奨）"
        horizon_days = 14

    preset = _style_defaults(style_name)
    min_expected_R = float(preset["min_expected_R"])

    # Optional override (from EV監査ログの自動最適化)
    if "min_expected_R_override" in st.session_state and st.session_state["min_expected_R_override"] is not None:
        try:
            min_expected_R = float(st.session_state["min_expected_R_override"])
        except Exception:
            pass

    st.caption(
        f"見送りライン（min_expected_R）: {min_expected_R:.3f}R / 想定期間: {horizon_days}日 / 価格: {period}・{interval}"
        + ("（自動最適化適用）" if "min_expected_R_override" in st.session_state and st.session_state["min_expected_R_override"] is not None else "")
    )

    with st.expander("🛡️ 安全/ガード（非常時だけ）", expanded=False):
        outage_policy = st.selectbox("外部データ全滅時の扱い", ["表示のみ（推奨：機会を殺さない）", "強制見送り（安全優先）"], index=0)
        guard_apply = st.selectbox(
            "ガードの反映（UIだけ）",
            ["表示のみ（推奨）", "縮退係数を表示（参考）", "品質OUTAGE時のみ見送り（安全）"],
            index=0,
        )
        lot_risk_alpha = st.slider("推奨ロット係数の強さ（α）", 0.0, 1.0, 0.35, 0.05, help="lot_mult = clamp(1 - α*global_risk_index, 0.2, 1.0)")

        # 📅 Economic events guard (recommended ON)
        st.markdown("---")
        st.markdown("##### 📅 経済指標/イベント リスクガード")
        st.session_state["event_guard_enable"] = st.checkbox(
            "経済指標の近接をリスクに反映（必須）",
            value=True,
            disabled=True,
            help="無料の経済指標カレンダー（週次JSON）から、直近の高/中インパクト指標を取得して閾値・見送りに反映します。"
        )
        st.caption("※スイング運用の安全装置として、イベントガード／レンジ時プレブロック／成行禁止（高インパクト近接）／イベント密度によるロット縮退／木金ルールは常時有効です。")
        st.session_state["event_block_high_impact_window"] = st.checkbox(
            "高インパクト指標の前後は強制見送り（±60分・必須）",
            value=True,
            disabled=True,
            help="発表前後はスプレッド拡大/ギャップ/急変が起きやすいため、運用上の安全弁として推奨。"
        )
        st.session_state["event_window_minutes"] = st.slider("高インパクト窓（分）", 15, 180, 60, 5)
        st.session_state["event_horizon_hours"] = st.slider("先読み時間（h）", 24, 168, 168, 6, help="この時間範囲のイベントだけを評価します。")
        # --- Swing (1週〜1か月) 向けの調整 ---
        # 数時間のイベントは「デイトレ判定」ではなく、"エントリー直後の実行リスク(滑り/急変/初期SL触れ)" を抑えるためのものです。
        # スイング運用では、影響を「数時間」ではなく「1〜2日スケール」でも加味するのが現実的なので、減衰スケールを用意します。
        st.session_state["event_hours_scale"] = st.slider(
            "スイング用：イベント影響の減衰スケール（h）",
            4, 72, 24, 2,
            help="24hなら『数時間』だけでなく1〜2日スケールでもイベントリスクを加味します。小さいほどデイトレ寄り、大きいほどスイング寄り。"
        )
        st.session_state["event_threshold_add"] = st.slider(
            "イベントによる動的閾値の上乗せ（最大R）",
            0.0, 0.30, 0.12, 0.01,
            help="イベント密度が高いほど見送りが増えます。スイングは0.10〜0.18が目安。"
        )
        st.session_state["event_preblock_range_enable"] = st.checkbox(
            "レンジ優勢かつ高インパクトが近いときは見送り（必須）",
            value=True,
            disabled=True,
            help="RANGE局面は指標でレンジ抜け→急反転が起きやすいため、根拠が弱い場合は発表後に再判定します。"
        )
        st.session_state["event_preblock_hours"] = st.slider(
            "上記見送りの判定時間（h）",
            0, 48, 24, 1,
            help="スイング運用でも『エントリー直後の急変』は致命傷になり得るため、24h程度を推奨。"
        )

        st.session_state["event_impacts"] = st.multiselect(
            "対象インパクト",
            ["High", "Medium", "Low", "Holiday"],
            default=["High", "Medium"],
            help="Lowまで入れると見送りが増えがちです。まずは High/Medium 推奨。"
        )
        with st.expander("（上級）イベントソース設定", expanded=False):
            st.session_state["event_timezone"] = st.text_input("表示タイムゾーン", value="Asia/Tokyo")
            st.session_state["event_norm"] = st.slider(
                "（上級）イベントリスク正規化係数（大きいほど効きが弱い）",
                1.0, 8.0, 3.0, 0.25,
                help="イベントスコアを0..1に正規化する係数です。値を上げるとイベントによる見送りが減ります。"
            )

            st.session_state["event_calendar_url"] = st.text_input(
                "カレンダーURL（JSON）",
                value="https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                help="既定: Forex Factory 週次エクスポート（無料）。取得不可の時はイベントリスク=0で動作します。"
            )

        force_no_trade_env = (os.getenv("FORCE_NO_TRADE", "") or "").strip().lower() in ("1","true","yes","on")
        force_no_trade = st.checkbox("🛑 手動緊急停止（最終判断を全てNO_TRADE）", value=force_no_trade_env)

    with st.expander("🔑 APIキー（任意・入れた分だけ強くなる）", expanded=False):
        openai_key = st.text_input("OPENAI_API_KEY（地政学LLM）", value=_load_secret("OPENAI_API_KEY", ""), type="password")
        news_key = st.text_input("NEWSAPI_KEY（記事取得）", value=_load_secret("NEWSAPI_KEY", ""), type="password")
        te_key = st.text_input("TRADING_ECONOMICS_KEY（経済指標）", value=_load_secret("TRADING_ECONOMICS_KEY", ""), type="password")
        fred_key = st.text_input("FRED_API_KEY（金利/VIX/DXY）", value=_load_secret("FRED_API_KEY", ""), type="password")
        av_key = st.text_input("ALPHAVANTAGE_API_KEY（マクロ補助/予備）", value=_load_secret("ALPHAVANTAGE_API_KEY", ""), type="password")
        st.caption("※ChatGPT利用とOpenAI APIは別物です。OpenAIは課金/権限が無いと401になります。")

    with st.expander("Capital Governor（本気運用の安全装置）", expanded=False):
        max_dd = st.slider("最大DD（停止）", 0.05, 0.30, 0.15, 0.01)
        daily_stop = st.slider("日次損失（停止）", 0.01, 0.10, 0.03, 0.01)
        max_streak = st.slider("連敗停止", 2, 12, 5, 1)
        equity_dd = st.number_input("現在DD（運用者入力）", value=0.0, step=0.01, help="0.10=10%DD")
        daily_loss = st.number_input("本日損失率（運用者入力）", value=0.0, step=0.01)
        losing_streak = st.number_input("連敗数（運用者入力）", value=0, step=1)

    with st.expander("🔧 詳細/診断（普段は不要）", expanded=False):
        # プリセットの上書き
        period = st.selectbox("価格期間（上書き）", ["1y", "2y", "5y", "10y"], index=["1y","2y","5y","10y"].index(period))
        interval = st.selectbox("価格間隔（上書き）", ["1d", "1wk", "1h"], index=["1d","1wk","1h"].index(interval))
        show_meta = st.checkbox("取得メタ表示（検証用）", value=False)
        show_debug = st.checkbox("デバッグ表示（検証用）", value=False)
        allow_override = st.checkbox("EV閾値/想定期間を手動上書き", value=False)
        if allow_override:
            min_expected_R = st.slider("min_expected_R", 0.0, 0.30, float(min_expected_R), 0.01)
            horizon_days = st.slider("horizon_days", 1, 30, int(horizon_days), 1)
        pair_custom = st.multiselect("スキャン対象（任意）", PAIR_LIST_ALL, default=PAIR_LIST_DEFAULT)

        if st.button("🔄 キャッシュクリアして再取得"):
            st.cache_data.clear()
            st.rerun()
period = locals().get("period", "10y")
interval = locals().get("interval", "1d")
show_meta = locals().get("show_meta", False)
show_debug = locals().get("show_debug", False)
pair_custom = locals().get("pair_custom", PAIR_LIST_DEFAULT)


guard_apply = locals().get("guard_apply", "表示のみ（推奨）")
lot_risk_alpha = float(locals().get("lot_risk_alpha", 0.35))
force_no_trade = bool(locals().get("force_no_trade", False))

keys = {
    "OPENAI_API_KEY": (locals().get("openai_key","") or "").strip(),
    "NEWSAPI_KEY": (locals().get("news_key","") or "").strip(),
    "TRADING_ECONOMICS_KEY": (locals().get("te_key","") or "").strip(),
    "FRED_API_KEY": (locals().get("fred_key","") or "").strip(),
    "ALPHAVANTAGE_API_KEY": (locals().get("av_key","") or "").strip(),
}

governor_cfg = {
    "max_drawdown_limit": float(locals().get("max_dd", 0.15)),
    "daily_loss_limit": float(locals().get("daily_stop", 0.03)),
    "max_losing_streak": int(locals().get("max_streak", 5)),
    "equity_drawdown": float(locals().get("equity_dd", 0.0)),
    "daily_loss": float(locals().get("daily_loss", 0.0)),
    "losing_streak": int(locals().get("losing_streak", 0)),
}

tabs = st.tabs(["🟢 AUTO判断", "🧪 バックテスト（WFA）", "📊 パフォーマンス", "📘 使い方"])

# =========================
# Tab 1: AUTO
# =========================
with tabs[0]:
    st.subheader("最終判断（ここだけ見ればOK）")

    if "相場全体" in mode:
        pairs = [_normalize_pair_label(p) for p in (pair_custom or PAIR_LIST_DEFAULT)]
        pairs = [p for p in pairs if p]
        if not pairs:
            st.error("スキャン対象が空です。")
            st.stop()

        st.caption("複数ペアを同じロジックで評価し、EV最大のペアを自動選択します（日足はStooq優先で安定化）。")

        rows: List[Dict[str, Any]] = []
        # 外部リスクはグローバル（ペア依存しない）なので、マルチペアでも1回だけ取得
        feats_global, ext_meta_global = fetch_external("GLOBAL", keys=keys)
        for p in pairs:
            sym = _pair_label_to_symbol(p)
            df, price_meta = fetch_price_history(p, sym, period=period, interval=interval, prefer_stooq=(str(interval)=="1d"))
            if df.empty:
                rows.append({"pair": p, "EV": None, "decision": "NO_DATA", "confidence": None, "dom_state": None})
                continue

            feats, ext_meta = feats_global, ext_meta_global
            ctx = _build_ctx(p, df, feats, horizon_days=int(horizon_days), min_expected_R=float(min_expected_R), style_name=style_name, governor_cfg=governor_cfg)
            # Event guard settings (passed to logic)
            try:
                ctx.update({
                    "event_guard_enable": bool(st.session_state.get("event_guard_enable", True)),
                    "event_block_high_impact_window": bool(st.session_state.get("event_block_high_impact_window", True)),
                    "event_window_minutes": int(st.session_state.get("event_window_minutes", 60) or 60),
                    "event_horizon_hours": int(st.session_state.get("event_horizon_hours", 72) or 72),
                    "event_impacts": list(st.session_state.get("event_impacts", ["High","Medium"]) or ["High","Medium"]),
                    "event_timezone": str(st.session_state.get("event_timezone","Asia/Tokyo") or "Asia/Tokyo"),
                    "event_calendar_url": str(st.session_state.get("event_calendar_url","https://nfs.faireconomy.media/ff_calendar_thisweek.json") or "https://nfs.faireconomy.media/ff_calendar_thisweek.json"),
                    "event_hours_scale": float(st.session_state.get("event_hours_scale", 24.0) or 24.0),
                    "event_norm": float(st.session_state.get("event_norm", 3.0) or 3.0),
                    "event_threshold_add": float(st.session_state.get("event_threshold_add", 0.12) or 0.12),
                    "event_preblock_range_enable": bool(st.session_state.get("event_preblock_range_enable", True)),
                    "event_preblock_hours": float(st.session_state.get("event_preblock_hours", 24.0) or 24.0),

                })
            except Exception:
                pass
            plan = logic.get_ai_order_strategy(price_df=df, pair=p, context_data=ctx, ext_features=feats, api_key=keys.get("OPENAI_API_KEY",""))


            
            # --- Win-rate focus soft gate (display/decision layer) ---
            # Does not change internal calculations; only tightens final permission when user selects 勝率優先.
            try:
                if str(st.session_state.get('priority_mode','')) == '勝率優先（見送り増）':
                    # 勝率優先 = 「p_winをハード閾値で切り捨て」ではなく、
                    # 低p_winを"強いEVでのみ許可"する階段ゲートにする（ゼロ化を防ぐ）
                    p_win = float(plan.get("p_win_ev", 0.0) or 0.0)
                    conf = float(plan.get("confidence", 0.0) or 0.0)
                    ev_raw = float(plan.get("expected_R_ev_raw", plan.get("expected_R_ev", 0.0)) or 0.0)
                    thr = float(plan.get("dynamic_threshold", 0.0) or 0.0)

                    # ゲート設定（固定・後方互換）
                    conf_min = 0.55
                    hard_pwin_min = 0.50
                    soft_pwin_min = 0.54
                    ev_margin_if_soft = 0.02  # p_winが弱いときは、EVで補強されている場合のみ許可

                    veto = None
                    if conf < conf_min:
                        veto = f"勝率優先フィルタ: conf={conf:.2f} < {conf_min:.2f}"
                    elif p_win < hard_pwin_min:
                        veto = f"勝率優先フィルタ: p_win={p_win:.2f} < {hard_pwin_min:.2f}"
                    elif p_win < soft_pwin_min and ev_raw < (thr + ev_margin_if_soft):
                        veto = f"勝率優先フィルタ: p_win={p_win:.2f}（弱）→ EV補強不足 {ev_raw:+.3f} < {thr + ev_margin_if_soft:.3f}"

                    if str(plan.get("decision","")) == "TRADE" and veto:
                        plan["decision"] = "NO_TRADE"
                        reasons = list(plan.get("veto_reasons", []) or [])
                        reasons.append(veto)
                        plan["veto_reasons"] = reasons
                        plan["why"] = reasons[-1]
            except Exception:
                pass
# ---- EV audit row (for 2-week verification) ----
            try:
                overlay = plan.get("overlay_meta", {}) or {}
                ev_raw = plan.get("expected_R_ev_raw", plan.get("expected_R_ev", 0.0))
                ev_adj = plan.get("expected_R_ev_adj", plan.get("expected_R_ev", 0.0))
                thr = plan.get("dynamic_threshold", 0.0)
                # "killed" means: would TRADE on raw, but would be NO_TRADE on adj (for analysis)
                decision_raw = "TRADE" if float(ev_raw) >= float(thr) else "NO_TRADE"
                decision_adj = "TRADE" if float(ev_adj) >= float(thr) else "NO_TRADE"
                killed_by_adj = (decision_raw == "TRADE" and decision_adj != "TRADE")
                _ev_audit_append({
                    "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "timeframe_mode": str(st.session_state.get("timeframe_mode", "")),
                    "pair": str(p),
                    "decision_adj": str(plan.get("decision","")),
                    "decision_raw": decision_raw,
                    "killed_by_adj": str(bool(killed_by_adj)),
                    "ev_raw": float(ev_raw),
                    "ev_adj": float(ev_adj),
                    "dynamic_threshold": float(thr),
                    "dominant_state": str(plan.get("dominant_state","")),
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "global_risk_index": float(overlay.get("global_risk_index", 0.0) or 0.0),
                    "war_probability": float(overlay.get("war_probability", 0.0) or 0.0),
                    "financial_stress": float(overlay.get("financial_stress", 0.0) or 0.0),
                    "macro_risk_score": float(overlay.get("macro_risk_score", 0.0) or 0.0),
                    "risk_off_bump": float(overlay.get("risk_off_bump", 0.0) or 0.0),
                })
            except Exception:
                pass
            # ---- /EV audit row ----
            # ---- operator guard (UI-level; default display-only) ----

            lot_mult = _lot_multiplier(feats.get("global_risk_index", 0.0), lot_risk_alpha)

            decision_override = None

            override_reason = ""

            # 手動緊急停止は最優先

            if force_no_trade:

                decision_override = "NO_TRADE"

                override_reason = "手動緊急停止"

            # 品質OUTAGE時のみ見送り（安全）

            try:

                parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}

                level = str((((parts.get("quality", {}) or {}).get("detail", {}) or {}).get("level", "") or ""))

            except Exception:

                level = ""

            if decision_override is None and ("品質OUTAGE時のみ見送り" in str(guard_apply)) and level == "OUTAGE":

                decision_override = "NO_TRADE"

                override_reason = "外部データ品質OUTAGE"

            if decision_override is not None:

                plan = dict(plan or {})

                plan["_decision_original"] = plan.get("decision")

                plan["decision"] = decision_override

                plan["_decision_override_reason"] = override_reason

            plan = dict(plan or {})

            plan["_lot_multiplier_reco"] = float(_apply_swing_lot_guards(lot_mult, plan))


            # SBI最小1建ガード（縮退できない局面はAI側で「見送り」に倒す）

            plan = _apply_sbi_minlot_guard(plan, sbi_min_lot=1)
            plan_ui = plan
            try:
                parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}
                level = str(((parts.get("quality", {}) or {}).get("detail", {}) or {}).get("level", "") or "")
                if "強制見送り" in str(locals().get("outage_policy","")) and level == "OUTAGE":
                    # UI only: do not change logic.py; just stop recommending entries when blind
                    plan_ui = dict(plan or {})
                    plan_ui["decision"] = "NO_TRADE"
                    vr = list(plan_ui.get("veto_reasons") or [])
                    if "DATA_OUTAGE" not in vr:
                        vr.append("DATA_OUTAGE（外部データ全滅）")
                    plan_ui["veto_reasons"] = vr
            except Exception:
                plan_ui = plan

            ev = float(plan.get("expected_R_ev") or 0.0)
            score = float(plan.get("rank_score") or plan.get("final_score") or ev)
            decision = str(plan.get("decision") or "NO_TRADE")
            conf = float(plan.get("confidence") or 0.0)
            dom = _dominant_state(plan.get("state_probs", {}))

            rows.append({
                "pair": p,
                "score": score,
                "EV": ev,
                "decision": decision,
                "confidence": conf,
                "dom_state": dom,
                "_plan": plan,
                "_plan_ui": plan_ui,
                "_ctx": ctx,
                "_feats": feats,
                "_price_meta": price_meta,
                "_ext_meta": ext_meta,
            })

        ranked = [r for r in rows if isinstance(r.get("score"), (int, float))]
        ranked.sort(key=lambda r: float(r["score"]), reverse=True)
        if not ranked:
            st.error("有効なペアがありません（全てNO_DATA）。")
            st.dataframe(pd.DataFrame(rows)[["pair", "decision"]], use_container_width=True)
            st.stop()

        # --- AI selection: pick ONE pair to act on (SBIガード込みのTRADE候補から) ---
        trade_ranked = [r for r in ranked if str(r.get("decision", "")) == "TRADE"]
        trade_ranked.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

        if trade_ranked:
            best = trade_ranked[0]
            st.markdown(f"## 🧠 AI選択：本日の実行ペア  **{best['pair']}**")
        else:
            best = ranked[0]
            st.markdown("## 🧠 AI選択：本日は **見送り**（トレード候補なし）")
            st.caption("※ SBI最小1建ガード・イベント近接・閾値条件により、相場全体で見ても全ペアが見送り判定です。")

        plan = best["_plan"]
        plan_ui_best = best.get("_plan_ui", plan)
        plan_ui_best = _apply_trend_assist(plan_ui_best, best.get("_ctx", {}))
        feats = best["_feats"]
        price = float(best["_ctx"].get("price", 0.0))

        # 見送りの日は「詳細は必要なときだけ」開けるようにする
        details_box = st.container() if trade_ranked else st.expander("参考（最良候補の診断・ログ）", expanded=False)
        with details_box:
            # Top panel must show entry format + price (user request)
            _render_top_trade_panel(best["pair"], plan_ui_best, price)

            # Risk dashboard (new)
            _render_risk_dashboard(plan_ui_best, feats, ext_meta=best.get("_ext_meta", {}))

            _render_ai_engine_panel(best.get("_ctx", {}), plan_ui_best)

            # Holding management (event/weekend approach)
            _render_hold_manage_panel(best["pair"], best.get("_ctx", {}), plan_ui_best, feats, keys)

            # Logging (optional)
            _render_logging_panel(best["pair"], plan_ui_best, best.get("_ctx", {}), feats, best.get("_price_meta", {}), best.get("_ext_meta", {}))

            # 「ランキング」は“トレード可能が複数あるときだけ”表示（上位3まで）
            if len(trade_ranked) >= 2:
                st.markdown("### トレード可能候補（上位3）")
                view = []
                for r in trade_ranked[:3]:
                    p = r.get("_plan_ui", r.get("_plan", {})) or {}
                    view.append({
                        "pair": r["pair"],
                        "score": float(r.get("score", 0.0) or 0.0),
                        "EV(R)": float(r.get("EV", 0.0) or 0.0),
                        "confidence": float(r.get("confidence", 0.0) or 0.0),
                        "注文": _order_type_jp(p.get("order_type", "")),
                        "エントリー": _entry_type_jp(p.get("entry_type", "")),
                        "event_mode": str(p.get("event_mode", "")),
                    })
                st.dataframe(pd.DataFrame(view), use_container_width=True)
                st.caption("※ 1つしかトレード候補がない日は表示しません。")

            # デバッグ用途：全ペア一覧は必要なときだけ（通常は非表示）
            if bool(show_debug):
                with st.expander("デバッグ：全ペア一覧（参考）", expanded=False):
                    view_all = [{
                        "pair": r["pair"],
                        "score": float(r.get("score", r.get("EV", 0.0))),
                        "EV": float(r["EV"]),
                        "decision": r["decision"],
                        "confidence": float(r["confidence"]),
                        "dominant_state": _state_label_full(r["dom_state"]),
                        "global_risk": float(r["_feats"].get("global_risk_index", 0.0)),
                        "war": float(r["_feats"].get("war_probability", 0.0)),
                    } for r in ranked]
                    st.dataframe(pd.DataFrame(view_all), use_container_width=True)

            st.markdown("### EV内訳（AI選択ペア）")

            # モバイル前提：棒グラフだけだと「0の棒が見えない」ため、
            # まず表で「確率(%) / 寄与EV(R)」を明示し、図は折りたたみにします。
            ev_contribs = (plan.get("ev_contribs", {}) or {})
            state_probs = (plan.get("state_probs", {}) or {})
            _states = ["trend_up", "trend_down", "range", "risk_off"]

            rows_ev = []
            for st_name in _states:
                c = float((ev_contribs or {}).get(st_name, 0.0) or 0.0)
                p = float((state_probs or {}).get(st_name, 0.0) or 0.0)
                rows_ev.append({
                    "状態": _state_label_full(st_name),
                    "確率(%)": round(p * 100.0, 1),
                    "寄与EV(R)": round(c, 4),
                })

            cdf = pd.DataFrame(rows_ev)
            if not cdf.empty:
                st.dataframe(cdf, use_container_width=True)
                st.caption("※ 棒が見えない＝寄与が0付近です。未計算ではありません。")

                with st.expander("棒グラフ（参考）", expanded=False):
                    st.bar_chart(cdf.set_index("状態")[["寄与EV(R)"]])
            else:
                st.info("EV内訳が空です。")

            with st.expander("詳細（AI選択ペア）", expanded=False):
                st.json({"plan": plan})
                if show_debug:
                    st.json({"ctx": best["_ctx"], "feats": feats})
                if show_meta:
                    st.json({"price_meta": best.get("_price_meta", {}), "external_meta": best.get("_ext_meta", {})})
    else:
        # 単一ペアはプルダウン（ユーザビリティ改善）
        common_pairs = [
            'USD/JPY','AUD/JPY','EUR/USD','EUR/JPY','GBP/JPY','GBP/USD','AUD/USD','USD/CHF','NZD/JPY','CAD/JPY',
            'USD/CAD','EUR/GBP','EUR/AUD','AUD/NZD','CHF/JPY','NZD/USD','CAD/CHF','EUR/CHF','GBP/CHF','AUD/CHF',
        ]
        sel = st.selectbox('通貨ペア（単一最適化）', options=common_pairs + ['（カスタム入力）'], index=0)
        if sel == '（カスタム入力）':
            custom = st.text_input('カスタム通貨ペア（例: USD/JPY）', value=st.session_state.get('pair_label','USD/JPY'))
            pair_label = _normalize_pair_label(custom)
        else:
            pair_label = _normalize_pair_label(sel)
        st.session_state["pair_label"] = pair_label
        symbol = _pair_label_to_symbol(pair_label)

        df, price_meta = fetch_price_history(pair_label, symbol, period=period, interval=interval, prefer_stooq=(str(interval)=="1d"))
        if df.empty:
            st.error("価格データ取得に失敗しました。")
            st.json(price_meta)
            st.stop()

        feats, ext_meta = fetch_external(pair_label, keys=keys)
        ctx = _build_ctx(pair_label, df, feats, horizon_days=int(horizon_days), min_expected_R=float(min_expected_R), style_name=style_name, governor_cfg=governor_cfg)
        # Event guard settings (passed to logic)
        try:
            ctx.update({
                "event_guard_enable": bool(st.session_state.get("event_guard_enable", True)),
                "event_block_high_impact_window": bool(st.session_state.get("event_block_high_impact_window", True)),
                "event_window_minutes": int(st.session_state.get("event_window_minutes", 60) or 60),
                "event_horizon_hours": int(st.session_state.get("event_horizon_hours", 72) or 72),
                "event_impacts": list(st.session_state.get("event_impacts", ["High","Medium"]) or ["High","Medium"]),
                "event_timezone": str(st.session_state.get("event_timezone","Asia/Tokyo") or "Asia/Tokyo"),
                "event_calendar_url": str(st.session_state.get("event_calendar_url","https://nfs.faireconomy.media/ff_calendar_thisweek.json") or "https://nfs.faireconomy.media/ff_calendar_thisweek.json"),
                    "event_hours_scale": float(st.session_state.get("event_hours_scale", 24.0) or 24.0),
                    "event_norm": float(st.session_state.get("event_norm", 3.0) or 3.0),
                    "event_threshold_add": float(st.session_state.get("event_threshold_add", 0.12) or 0.12),
                    "event_preblock_range_enable": bool(st.session_state.get("event_preblock_range_enable", True)),
                    "event_preblock_hours": float(st.session_state.get("event_preblock_hours", 24.0) or 24.0),

            })
        except Exception:
            pass
        plan = logic.get_ai_order_strategy(price_df=df, pair=p, context_data=ctx, ext_features=feats, api_key=keys.get("OPENAI_API_KEY",""))

        # ---- operator guard (UI-level; default display-only) ----

        lot_mult = _lot_multiplier(feats.get("global_risk_index", 0.0), lot_risk_alpha)

        decision_override = None

        override_reason = ""

        # 手動緊急停止は最優先

        if force_no_trade:

            decision_override = "NO_TRADE"

            override_reason = "手動緊急停止"

        # 品質OUTAGE時のみ見送り（安全）

        try:

            parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}

            level = str((((parts.get("quality", {}) or {}).get("detail", {}) or {}).get("level", "") or ""))

        except Exception:

            level = ""

        if decision_override is None and ("品質OUTAGE時のみ見送り" in str(guard_apply)) and level == "OUTAGE":

            decision_override = "NO_TRADE"

            override_reason = "外部データ品質OUTAGE"

        if decision_override is not None:

            plan = dict(plan or {})

            plan["_decision_original"] = plan.get("decision")

            plan["decision"] = decision_override

            plan["_decision_override_reason"] = override_reason

        plan = dict(plan or {})

        plan["_lot_multiplier_reco"] = float(_apply_swing_lot_guards(lot_mult, plan))


        # SBI最小1建ガード（縮退できない局面はAI側で「見送り」に倒す）

        plan = _apply_sbi_minlot_guard(plan, sbi_min_lot=1)
        plan_ui = plan
        try:
            parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}
            level = str(((parts.get("quality", {}) or {}).get("detail", {}) or {}).get("level", "") or "")
            if "強制見送り" in str(locals().get("outage_policy","")) and level == "OUTAGE":
                plan_ui = dict(plan or {})
                plan_ui["decision"] = "NO_TRADE"
                vr = list(plan_ui.get("veto_reasons") or [])
                if "DATA_OUTAGE" not in vr:
                    vr.append("DATA_OUTAGE（外部データ全滅）")
                plan_ui["veto_reasons"] = vr
        except Exception:
            plan_ui = plan

        plan_ui = _apply_trend_assist(plan_ui, ctx)

        price = float(ctx.get("price", 0.0))
        _render_top_trade_panel(pair_label, plan_ui, price)
        _render_risk_dashboard(plan_ui, feats, ext_meta=ext_meta)

        _render_ai_engine_panel(ctx, plan_ui)

        _render_logging_panel(pair_label, plan_ui, ctx, feats, price_meta, ext_meta)

        st.markdown("### EV内訳（何がEVを潰しているか）")
        ev_contribs = plan.get("ev_contribs", {}) or {}
        if isinstance(ev_contribs, dict) and ev_contribs:
            cdf = pd.DataFrame([{"state": k, "contrib_R": float(v)} for k, v in ev_contribs.items()]).sort_values("contrib_R")
            cdf["state_label"] = cdf["state"].apply(_state_label_full)
            st.bar_chart(cdf.set_index("state_label")[["contrib_R"]])
        else:
            st.info("EV内訳が空です。")

        with st.expander("詳細", expanded=False):
            st.json(plan.get("state_probs", {}))
            if show_debug:
                st.json({"ctx": ctx, "feats": feats})
            if show_meta:
                st.json({"price_meta": price_meta, "external_meta": ext_meta})

# =========================
# Tab 2: Backtest (keep existing)
# =========================
with tabs[1]:
    st.subheader("ウォークフォワード（WFA）バックテスト")
    st.caption("方向性確認用（コスト・スリップ未反映）。バックテストは“残す”方針。")

    colA, colB, colC = st.columns(3)
    with colA:
        bt_pair = st.selectbox("バックテスト対象ペア", PAIR_LIST_ALL, index=0)
        bt_period = st.selectbox("BT期間", ["5y", "10y"], index=1)
        train_years = st.number_input("train_years", min_value=1, max_value=8, value=3, step=1)
    with colB:
        test_months = st.number_input("test_months", min_value=1, max_value=24, value=6, step=1)
        bt_horizon = st.number_input("horizon_days", min_value=1, max_value=14, value=int(horizon_days), step=1)
    with colC:
        bt_min_ev = st.slider("min_expected_R", 0.0, 0.3, float(min_expected_R), 0.01)

    run = st.button("バックテスト実行", type="primary")
    if run:
        try:
            import backtest_ev_v1
            sym = _pair_label_to_symbol(bt_pair)
            wf_df, summ = backtest_ev_v1.run_backtest(
                pair_symbol=sym,
                period=bt_period,
                horizon_days=int(bt_horizon),
                train_years=int(train_years),
                test_months=int(test_months),
                min_expected_R=float(bt_min_ev),
            )
            st.markdown("### サマリー")
            st.json(summ)
            st.markdown("### WFA結果")
            st.dataframe(wf_df, use_container_width=True)

            csv = wf_df.to_csv(index=False).encode("utf-8")
            st.download_button("CSVダウンロード", data=csv, file_name=f"ev_wfa_{bt_pair.replace('/','_')}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"バックテストでエラー: {type(e).__name__}: {e}")


# =========================
# Tab 3: Performance
# =========================
with tabs[2]:
    st.subheader("パフォーマンス（損益ログから自動集計）")
    st.caption("signals / trades を保存していれば、ここで期待値・勝率・ドローダウンを自動で見えます。")

    df_s = _load_csv_df(SIGNAL_LOG_PATH)
    df_t = _load_csv_df(TRADE_LOG_PATH)

    m = _compute_trade_metrics(df_t)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("トレード数", f"{m.get('n',0)}")
    if m.get("n",0) > 0:
        c2.metric("期待値（平均R）", f"{m.get('expectancy_R',0.0):+.3f}")
        c3.metric("勝率", f"{m.get('win_rate',0.0)*100:.1f}%")
        pf = m.get("profit_factor")
        c4.metric("PF", ("∞" if pf == float("inf") else (f"{pf:.2f}" if isinstance(pf,(int,float)) else "—")))
    else:
        c2.metric("期待値（平均R）", "—")
        c3.metric("勝率", "—")
        c4.metric("PF", "—")

    if m.get("n",0) > 0 and "r_multiple" in df_t.columns:
        d = df_t.copy()
        d["r_multiple"] = pd.to_numeric(d["r_multiple"], errors="coerce")
        d = d.dropna(subset=["r_multiple"])
        d["cum_R"] = d["r_multiple"].cumsum()
        st.line_chart(d.set_index(pd.RangeIndex(len(d)))["cum_R"])
        st.caption(f"最大DD（R）: {m.get('max_drawdown_R',0.0):.3f} / 総R: {m.get('sum_R',0.0):+.3f}")

    st.markdown("### 直近のトレード（trades）")
    st.dataframe(df_t.tail(200), use_container_width=True)

    st.markdown("### 直近のシグナル（signals）")
    st.dataframe(df_s.tail(200), use_container_width=True)

    # downloads
    try:
        if not df_t.empty:
            st.download_button("trades.csv をダウンロード", data=df_t.to_csv(index=False).encode("utf-8"),
                               file_name="trades.csv", mime="text/csv")
        if not df_s.empty:
            st.download_button("signals.csv をダウンロード", data=df_s.to_csv(index=False).encode("utf-8"),
                               file_name="signals.csv", mime="text/csv")
    except Exception:
        pass


# =========================
# Tab 3: Guide
# =========================
with tabs[3]:
    st.markdown("""
# 📘 運用者向け・画面の見方（メイン/サイドバー）

このツールは **「期待値（EV）を最大化しつつ、外部リスクで“止める/弱める”」** ための運用パネルです。  
**迷ったら** → AUTO判断タブの **「最終判断」→「リスクダッシュボード」→「外部データ取得ステータス」** の順に見てください。

---

## 1) サイドバー（左）の機能
### モード
- **相場全体から最適ペアを自動抽出（推奨）**：複数ペアを走査し、EVが最大のペアを出します（運用向け）
- **単一ペア最適化（徹底）**：指定ペアだけを深く見る（検証/研究向け）

### 運用スタイル（標準/保守/攻撃）
- **保守**：見送りラインが高く、厳選（資金大きい/イベント多い時）
- **標準**：バランス
- **攻撃**：見送りラインが低く、回転（検証や小さめ資金向け）

### 想定期間（週/日）
- **週（推奨）**：ノイズに強く、判断が安定
- **日**：短期トレード寄り（シグナルは速いがブレやすい）

### APIキー（任意：入れた分だけ“外部リスク”が精密）
- **FRED**：VIX/DXY/金利（マクロ・不安定度）
- **NewsAPI**：記事見出しセンチメント
- **TradingEconomics**：CPI/NFPなど（ただし無料キーは国制限で403になりがち）
- **OpenAI**：LLMが地政学/危機確率を推定（JSON）→ **GlobalRisk/WarProb に反映**

### Capital Governor（本気運用の安全装置）
- 最大DD/日次損失/連敗が閾値を超えると **強制停止**します（運用者が入力）

---

## 2) メインパネル（AUTO判断タブ）の見方
### 最終判断（ここだけ見ればOK）
- **TRADE**：推奨エントリー（Entry/SL/TP）が出ます
- **NO_TRADE**：見送り。理由（veto）が出ます（EV不足/リスク過多/ガバナー停止など）

### 期待値EV (R) / 動的閾値 / 信頼度
- **EV (R)**：1R（＝損切り幅）を基準にした「1回あたりの期待値」  
  例）EV=+0.07 → 1回の取引で **平均 +0.07R** を狙う設計
- **動的閾値**：相場が危険になるほど上がる “見送りライン”  
  → 危険時に見送りが増えるのは **仕様**
- **信頼度**：モデルの確信度（0〜1）。低いほど慎重に。

### 🛡️ リスクダッシュボード（運用の心臓部）
- **総合リスク / 戦争・地政学 / 金融ストレス / マクロ不確実性** を 0〜1 で表示  
  0=平常 / 1=危機。**低/中/高** と **推奨アクション** が併記されます。

### 外部データ取得ステータス（原因究明）
- 0固定や異常値の原因は、ここに **http_401/403/429/timeout** として出ます。
- **keys 行**：キーがどこから読めたか（sec/ui/env）を表示します。

### EV内訳（棒グラフ）
- 相場タイプ（上昇/下降/レンジ/リスクオフ）の **どれがEVを押し上げ/押し下げ**しているかの内訳です。  
  「リスクオフが大きくマイナス」なら、見送りになりやすいのは正常です。

---

## 3) よくあるトラブルと対処
- **OpenAI 401**：APIキー/課金/権限（ChatGPT契約とは別）
- **TradingEconomics 403**：無料キー国制限（仕様寄り）
- **GDELT timeout/429**：ネットワーク到達性 or 間隔制御不足（キャッシュ/リトライで緩和）
""")

def _profit_max_reco(plan: dict) -> dict:
    """Compute an extended take-profit and a simple trail-stop suggestion (advisory)."""
    try:
        side = (plan.get("side") or "").upper()
        entry = float(plan.get("entry") or 0.0)
        sl = float(plan.get("stop_loss") or 0.0)
        tp = float(plan.get("take_profit") or 0.0)
        conf = float(plan.get("confidence") or 0.0)
        dom = str(plan.get("dominant_state") or "")
        overlay = plan.get("overlay_meta") or {}
        gri = float(overlay.get("global_risk_index") or 0.0)
        war = float(overlay.get("war_probability") or 0.0)
        fin = float(overlay.get("financial_stress") or 0.0)

        risk = max(0.0, min(1.0, 0.5*gri + 0.3*war + 0.2*fin))
        extend = 1.0 + 0.9 * max(0.0, conf - 0.5) * (1.0 - risk)
        if "risk_off" in dom:
            extend = 1.0
        extend = max(1.0, min(1.8, extend))

        dist_tp = abs(tp - entry)
        if dist_tp <= 0:
            tp_ext = tp
        else:
            tp_ext = entry + dist_tp * extend if side == "BUY" else (entry - dist_tp * extend if side == "SELL" else tp)

        dist_sl = abs(entry - sl)
        if dist_sl <= 0:
            trail_sl = sl
        else:
            trail_sl = entry + 0.5*dist_sl if side == "BUY" else (entry - 0.5*dist_sl if side == "SELL" else sl)

        return {"tp_ext": tp_ext, "trail_sl": trail_sl, "extend_factor": extend}
    except Exception:
        return {"tp_ext": plan.get("take_profit"), "trail_sl": plan.get("stop_loss"), "extend_factor": 1.0}



# --- Webhook Diagnostics (fixed27) ---
with st.expander("🔧 Webhook診断（送信テスト/失敗理由の表示）", expanded=False):
    url = ""
    try:
        url = st.secrets.get("LOG_WEBHOOK_URL", "")
    except Exception:
        url = ""

    if url:
        masked = url[:32] + "..." + url[-12:] if len(url) > 48 else url
        st.write(f"LOG_WEBHOOK_URL: `{masked}`")
    else:
        st.warning("LOG_WEBHOOK_URL が Secrets に設定されていません。B1(Webhook) は無効です。")

    test_payload = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pair": st.session_state.get("selected_pair", ""),
        "decision": "TEST",
        "ev_raw": 0.123,
        "ev_adj": 0.045,
        "dynamic_threshold": 0.010,
        "gate_mode": "EV_RAW",
    }

    colA, colB = st.columns(2)
    with colA:
        if st.button("Webhookへテスト送信", use_container_width=True):
            if not url:
                st.error("LOG_WEBHOOK_URL が未設定です。")
            else:
                res = _external_log_event("debug_test", test_payload)
                st.session_state["last_webhook_result"] = res
                wh = (res or {}).get("webhook") or {}
                if wh.get("ok"):
                    st.success(f"Webhook送信OK (HTTP {wh.get('status_code')})")
                else:
                    st.error(f"Webhook送信NG: {wh.get('error') or 'unknown'} (HTTP {wh.get('status_code')})")
                sb = (res or {}).get("supabase")
                if sb is not None:
                    if sb.get("ok"):
                        st.success(f"Supabase INSERT OK (HTTP {sb.get('status_code')})")
                    else:
                        st.error(f"Supabase INSERT NG: {sb.get('error') or 'unknown'} (HTTP {sb.get('status_code')})")
    with colB:
        st.caption("送信payload（確認用）")
        st.json({"kind": "debug_test", **test_payload})

    if "last_webhook_result" in st.session_state:
        st.caption("直近の送信結果（デバッグ）")
        st.json(st.session_state["last_webhook_result"])
# --- /Webhook Diagnostics ---
