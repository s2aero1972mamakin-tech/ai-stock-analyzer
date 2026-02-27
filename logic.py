
# logic.py
# ============================================================
# 日本株スイング（〜1ヶ月）向け：スキャン＋バックテスト＋注文書生成
# 目的：勝率だけでなく「期待値（Expectancy）= 勝率×平均利益 - 負け率×平均損失」を最大化する設計
#
# - スキャンはJPXマスター（全銘柄）をユニバースにし、流動性/トレンド/押し目 or ブレイクで候補抽出
# - 上位候補に対して簡易バックテスト（2年推奨）を走らせ、PF・平均R・最大DDを算出して再ランキング
# - 注文書はRベース（SL/TP1/TP2/時間切れ/建値移動）で数値化
#
# 注：データは yfinance 取得（無料APIのため遅延/欠損が起きる場合があります）
# ============================================================

from __future__ import annotations

import math
import re
import json
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def _stooq_symbol(ticker: str) -> str:
    """Convert common JP tickers to Stooq symbols.
    Examples:
      '7868.T' -> '7868.jp'
      '7868'   -> '7868.jp'
      '^N225'  -> '^nkx'  (Nikkei 225 on Stooq)
    """
    t = str(ticker).strip()
    if t.upper() in {"^N225", "N225", "NIKKEI"}:
        return "^nkx"
    if t.endswith(".T") or t.endswith(".t"):
        return t[:-2].lower() + ".jp"
    if re.fullmatch(r"\d{4}", t):
        return t + ".jp"
    return t.lower()

# Backward-compatible alias
def _to_stooq_symbol(ticker: str) -> str:
    return _stooq_symbol(ticker)



def _period_to_start(period: str) -> pd.Timestamp:
    """Convert lookback period string (e.g. '6mo','1y','2y') to a tz-naive start timestamp (00:00)."""
    now = pd.Timestamp.utcnow().normalize()
    # pd.Timestamp.utcnow() is usually tz-naive, but be defensive
    if now.tzinfo is not None:
        now = now.tz_convert(None)

    p = str(period).strip().lower()
    if p.endswith("mo"):
        n = int(re.sub(r"[^0-9]", "", p) or "6")
        start = now - pd.DateOffset(months=n)
    elif p.endswith("y"):
        n = int(re.sub(r"[^0-9]", "", p) or "2")
        start = now - pd.DateOffset(years=n)
    elif p.endswith("d"):
        n = int(re.sub(r"[^0-9]", "", p) or "180")
        start = now - pd.DateOffset(days=n)
    else:
        # default 2y
        start = now - pd.DateOffset(years=2)

    start = pd.Timestamp(start).normalize()
    if start.tzinfo is not None:
        start = start.tz_convert(None)
    return start

def _fetch_stooq_ohlc(
    symbol: str,
    start: Optional[datetime.date] = None,
    end: Optional[datetime.date] = None,
    retry_count: int = 3,
    pause: float = 0.25,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV from Stooq CSV endpoint.

    Uses optional date range parameters (d1/d2) to reduce payload size.
    - base endpoint: https://stooq.com/q/d/l/
    - params: s=<symbol>, i=d, d1=YYYYMMDD, d2=YYYYMMDD
    """
    base_url = "https://stooq.com/q/d/l/"
    sess = session or _STOOQ_SESSION

    params = {"s": symbol, "i": "d"}
    if start is not None:
        try:
            params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
        except Exception:
            pass
    if end is not None:
        try:
            params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
        except Exception:
            pass

    last_exc: Optional[Exception] = None
    for attempt in range(int(max(0, retry_count)) + 1):
        try:
            r = sess.get(
                base_url,
                params=params,
                timeout=(10, 25),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                return df
            if "Date" not in df.columns:
                return pd.DataFrame()

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]

            cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
            return df[cols]
        except Exception as e:
            last_exc = e
            if attempt < int(max(0, retry_count)):
                # exponential backoff with jitter-ish
                try:
                    time.sleep(float(pause) * (2 ** attempt))
                except Exception:
                    pass
                continue
            return pd.DataFrame()


import pytz
import streamlit as st
import requests
import io
import time
from openai import OpenAI

TOKYO = pytz.timezone("Asia/Tokyo")


# -------------------------
# 0) JPXマスター
# -------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_jpx_master() -> pd.DataFrame:
    """
    JPXが公開している「東証上場銘柄一覧（33業種）」に依存。
    取得失敗時は空DataFrameを返す（UI側でエラーハンドリング）。
    """
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        # pandasが適切なengineを選ぶこともあるので、まずは素直に読む
        df = pd.read_excel(url)
    except Exception:
        # 環境差でengine問題が出ることがあるので、BytesIO経由で再試行
        try:
            import requests  # type: ignore
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_excel(BytesIO(r.content), engine="xlrd")
        except Exception:
            return pd.DataFrame()

    # 想定列がない場合は空を返す
    needed = {"コード", "銘柄名", "33業種区分"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.copy()
    df = df[df["33業種区分"].notna()]
    df = df[df["33業種区分"] != "-"]
    df["ticker"] = df["コード"].astype(str).str.zfill(4) + ".T"
    df = df.rename(columns={"銘柄名": "name", "33業種区分": "sector"})
    return df[["ticker", "name", "sector"]].drop_duplicates().reset_index(drop=True)


def get_company_name(ticker: str) -> str:
    m = get_jpx_master()
    if m.empty:
        return ticker
    row = m[m["ticker"] == ticker]
    if row.empty:
        return ticker
    return str(row.iloc[0]["name"])


# -------------------------
# 1) OpenAI 呼び出し（任意）
# -------------------------
def _call_openai(api_key: str, system_prompt: str, user_prompt: str) -> str:
    try:
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return res.choices[0].message.content or ""
    except Exception as e:
        return f"⚠️ OpenAI API エラー: {e}"


def get_ai_analysis(api_key: str, ctx: dict) -> str:
    return _call_openai(
        api_key,
        "あなたは実戦派の株式ストラテジストです。曖昧な表現を避け、箇条書き中心で要点とリスクを短く提示してください。",
        f"""対象:{ctx.get('pair_label')}
現在値:{ctx.get('price'):.2f}円
RSI:{ctx.get('rsi'):.1f}
ATR:{ctx.get('atr'):.2f}
SMA5:{ctx.get('sma5'):.2f}
SMA25:{ctx.get('sma25'):.2f}
バックテスト: PF={ctx.get('pf')}, AvgR={ctx.get('avg_r')}, WinRate={ctx.get('win_rate')}
この銘柄を「1週間〜1ヶ月のスイング」で取引する前提で、優位性が出る局面・避けたい局面・注意点を提案してください。
""",
    )


def get_ai_order_strategy(api_key: str, ctx: dict) -> str:
    return _call_openai(
        api_key,
        "あなたは冷徹な執行責任者です。必ず数値を含め、注文実行の手順（何をいつ置くか）を明確に書いてください。",
        f"""対象:{ctx.get('pair_label')}
現在値:{ctx.get('price'):.2f}円
想定エントリー:{ctx.get('entry_price'):.2f}円
損切:{ctx.get('stop_price'):.2f}円
利確1:{ctx.get('tp1_price'):.2f}円
利確2:{ctx.get('tp2_price'):.2f}円
時間切れ:{ctx.get('time_stop_days')}営業日
推奨株数:{ctx.get('shares')}
PF={ctx.get('pf')}, AvgR={ctx.get('avg_r')}, WinRate={ctx.get('win_rate')}
この条件で、単元株（100株単位）を前提に、OCO/逆指値が使える一般的な国内株注文として命令書を作成してください。
""",
    )


def get_ai_portfolio(api_key: str, ctx: dict) -> str:
    return _call_openai(
        api_key,
        "あなたはポートフォリオマネージャーです。短期スイングにおける『保有継続/撤退/追加』判断をロジック優先で提案してください。",
        f"""対象:{ctx.get('pair_label')}
PF={ctx.get('pf')}, AvgR={ctx.get('avg_r')}, WinRate={ctx.get('win_rate')}, MaxDD={ctx.get('max_dd')}
現状の含み益/含み損が無い前提で、週末跨ぎやイベント前後での方針（持つ/持たない）を提案してください。
""",
    )


# -------------------------
# 2) 価格データ取得
# -------------------------
def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # multiindex columns -> flatten
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance multi-index: (Field, Ticker)
        # we'll handle per-ticker extraction elsewhere
        return df
    # timezone fix
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df



def get_market_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """価格データを取得（現在は Stooq を使用）。日本株のみ前提。

    - ticker: "7868.T" など（内部で Stooq 向けに変換）
    - period: "6mo", "1y", "2y" など
    - interval: 現状 "1d" 想定（Stooq の日足）
    """
    # Determine desired date window first (to reduce payload)
    start_ts = _period_to_start(period)
    try:
        start_date = pd.Timestamp(start_ts).date()
    except Exception:
        start_date = None

    try:
        end_date = datetime.datetime.now(TOKYO).date()
    except Exception:
        end_date = None

    try:
        stooq_symbol = _to_stooq_symbol(ticker)
        df = _fetch_stooq_ohlc(stooq_symbol, start=start_date, end=end_date)
    except Exception:
        return pd.DataFrame()


    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure datetime index (tz-naive)
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[~df.index.isna()]
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        return pd.DataFrame()

    df = df.sort_index()

    # Slice by period
    start_ts = _period_to_start(period)
    try:
        start_ts = pd.Timestamp(start_ts)
        # Normalize timezone so we can compare reliably with tz-naive daily indices (Stooq etc.)
        if getattr(start_ts, "tz", None) is not None:
            start_ts = start_ts.tz_convert(None)
    except Exception:
        return df

    return df.loc[df.index >= start_ts].copy()


def get_benchmark_data(ticker: str = "^N225", period: str = "2y") -> pd.DataFrame:
    """Benchmark data via current market data source (default Nikkei 225).

    Args:
        ticker: Benchmark symbol (e.g., "^N225"). Kept for backward compatibility with older call sites.
        period: Lookback window like "6mo", "1y", "2y".
    """
    return get_market_data(ticker, period=period)

def calculate_indicators(df: pd.DataFrame, include_sma200: bool = False) -> pd.DataFrame:
    """指標計算。
    重要: 6mo 等の短い期間では SMA200 が作れないため、include_sma200=False で呼ぶこと。

    追加（安定化）:
    - 売買代金（TURNOVER = Close*Volume）とその20日平均を計算（流動性フィルタ用）
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # Guard: ensure required columns exist
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in d.columns:
            return pd.DataFrame()

    # SMAs
    for w in (5, 10, 25, 75):
        d[f"SMA_{w}"] = d["Close"].rolling(window=w).mean()
    if include_sma200:
        d["SMA_200"] = d["Close"].rolling(window=200).mean()

    # RSI(14) - EWMA smoothing
    delta = d["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    d["RSI"] = 100 - (100 / (1 + rs))

    # ATR(14)
    tr = pd.concat(
        [
            (d["High"] - d["Low"]),
            (d["High"] - d["Close"].shift()).abs(),
            (d["Low"] - d["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.rolling(window=14).mean()
    d["ATR_PCT"] = (d["ATR"] / d["Close"]) * 100

    # Trend helpers
    d["SMA_DIFF"] = (d["Close"] - d["SMA_25"]) / d["SMA_25"] * 100
    d["SMA25_SLOPE5"] = d["SMA_25"].pct_change(5) * 100
    d["SMA75_SLOPE10"] = d["SMA_75"].pct_change(10) * 100

    # Liquidity (shares)
    d["VOL_AVG_20"] = d["Volume"].rolling(window=20).mean()
    d["VOL_RATIO"] = d["Volume"] / d["VOL_AVG_20"]

    # Liquidity (JPY turnover)
    d["TURNOVER"] = d["Close"] * d["Volume"]
    d["TURNOVER_AVG_20"] = d["TURNOVER"].rolling(window=20).mean()
    d["TURNOVER_RATIO"] = d["TURNOVER"] / d["TURNOVER_AVG_20"]

    # Range
    d["HIGH_20"] = d["High"].rolling(20).max()
    d["LOW_20"] = d["Low"].rolling(20).min()

    # dropna only on required columns (avoid wiping out short-period data)
    base_cols = [
        "Close", "High", "Low", "Volume",
        "SMA_25", "SMA_75", "RSI", "ATR", "ATR_PCT",
        "SMA_DIFF", "VOL_AVG_20", "VOL_RATIO",
        "TURNOVER_AVG_20", "TURNOVER_RATIO",
        "HIGH_20", "LOW_20", "SMA25_SLOPE5", "SMA75_SLOPE10",
    ]
    if include_sma200:
        base_cols.append("SMA_200")

    d = d.dropna(subset=base_cols).copy()
    return d


def judge_condition(price: float, sma5: float, sma25: float, sma75: float, rsi: float) -> dict:
    short = {"status": "上昇継続 (SMA5上)", "color": "blue"} if price > sma5 else {"status": "勢い鈍化 (SMA5下)", "color": "red"}
    mid = {"status": "静観", "color": "gray"}
    if sma25 > sma75 and rsi < 70:
        mid = {"status": "上昇トレンド (押し目買い)", "color": "blue"}
    elif rsi <= 30:
        mid = {"status": "売られすぎ (反発警戒)", "color": "orange"}
    return {"short": short, "mid": mid}


# -------------------------
# 4) 戦略パラメータ
# -------------------------
@dataclass(frozen=True)
class SwingParams:
    # --------------------
    # Entry / Universe filters
    # --------------------
    rsi_low: float = 40.0
    rsi_high: float = 65.0

    # Pullback (SMA25乖離 %)
    pullback_low: float = -6.0     # 下限（より下げ過ぎは避ける）
    pullback_high: float = -1.0    # 上限

    # Volatility (ATR% = ATR/Close*100)
    atr_pct_min: float = 1.0
    atr_pct_max: float = 6.0

    # Liquidity filters
    vol_avg20_min: float = 100_000.0          # 出来高（株数）20日平均の下限（後方互換）
    turnover_avg20_min_yen: float = 0.0       # 売買代金（円）20日平均の下限（0で無効）

    # Trend filters
    require_sma25_over_sma75: bool = True
    regime_filter: bool = False               # Trueなら「N225がSMA200上」のときだけロング候補を出す（0件時はauto-relaxでOFF化）

    # Entry trigger type: "pullback" or "breakout"
    entry_mode: str = "pullback"

    # Pullback trigger relaxation (頻度を落としすぎないための安全弁)
    pullback_allow_sma5_trigger: bool = True  # Close>昨日高 だけでなく Close>SMA5 でも可（勝率はStage2で選別）

    # Exit rules
    atr_mult_stop: float = 1.5
    tp1_r: float = 1.0
    tp2_r: float = 3.0
    time_stop_days: int = 10

    # Breakout config
    breakout_lookback: int = 20
    breakout_vol_ratio: float = 1.5

    # Backtest / ranking stabilization
    min_trades_bt: int = 8                   # 少数トレードの“事故”を除外
    score_shrink_k: float = 20.0             # expectancy の縮小係数（大きいほど少数トレードを強く抑制）
    score_pf_clip: float = 4.0               # PF上限（inf対策）
    score_dd_ref: float = 0.20               # DD罰則の基準（20%）

    # Risk sizing (UI can override)
    risk_pct: float = 2.0


# -------------------------
# 5) 簡易バックテスト（Rベース）
# -------------------------
@dataclass
class BacktestResult:
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    avg_win_r: float
    avg_loss_r: float
    max_drawdown: float
    equity_curve_r: pd.Series
    trades: pd.DataFrame


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if not dd.empty else 0.0


def backtest_swing(df_ind: pd.DataFrame, params: SwingParams) -> BacktestResult:
    """
    日足での簡易バックテスト。
    - エントリーは「条件成立日の翌日Open」で約定したと仮定（実務に近い）
    - SL=ATR*mult、TP1=+1Rで半分利確→残りは建値へ、TP2=+3R or トレーリング（SMA10割れ）
    - time_stop_days でTP1未達なら撤退
    """
    if df_ind is None or df_ind.empty:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, pd.Series(dtype=float), pd.DataFrame())

    d = df_ind.copy()

    # signals
    # slope filter: SMA25 is rising
    slope_ok = d["SMA25_SLOPE5"] > 0.0

    if params.entry_mode == "pullback":
        # 押し目は SMA25 の下に潜ることがあるため Close>SMA25 は要求しない
        trend_ok = (d["Close"] > d["SMA_75"]) & slope_ok
        if params.require_sma25_over_sma75:
            trend_ok &= d["SMA_25"] > d["SMA_75"]
    else:
        # ブレイクは強い局面：Close>SMA25
        trend_ok = (d["Close"] > d["SMA_25"]) & slope_ok
        if params.require_sma25_over_sma75:
            trend_ok &= d["SMA_25"] > d["SMA_75"]

    rsi_ok = (d["RSI"] >= params.rsi_low) & (d["RSI"] <= params.rsi_high)
    atr_ok = (d["ATR_PCT"] >= params.atr_pct_min) & (d["ATR_PCT"] <= params.atr_pct_max)
    vol_ok = d["VOL_AVG_20"] >= params.vol_avg20_min

    if params.entry_mode == "pullback":
        pull_ok = (d["SMA_DIFF"] >= params.pullback_low) & (d["SMA_DIFF"] <= params.pullback_high)
        # 反発確認：終値が前日高値を上抜け
        trigger = d["Close"] > d["High"].shift(1)
        signal = trend_ok & rsi_ok & atr_ok & vol_ok & pull_ok & trigger
    else:
        lb = params.breakout_lookback
        # ブレイク：終値が過去lb日高値更新（当日を含めない）
        prev_high = d["High"].rolling(lb).max().shift(1)
        breakout = d["Close"] > prev_high
        vr_ok = d["VOL_RATIO"] >= params.breakout_vol_ratio
        signal = trend_ok & rsi_ok & atr_ok & vol_ok & breakout & vr_ok

    # entry on next day open
    entries = signal.shift(1).fillna(False)

    trades = []
    equity = [1.0]  # start at 1.0R equity baseline
    equity_dates = [d.index[0]]

    in_pos = False
    entry_price = stop_price = tp1 = tp2 = np.nan
    entry_idx = None
    hit_tp1 = False
    remaining = 1.0  # position fraction

    for i in range(1, len(d)):
        date = d.index[i]
        row = d.iloc[i]

        # update equity curve daily (mark-to-market not needed; we keep step on trade exits)
        # We'll append last equity with current date for chart continuity.
        equity.append(equity[-1])
        equity_dates.append(date)

        if not in_pos:
            if bool(entries.iloc[i]):
                # enter
                ep = float(row["Open"])
                atr = float(row["ATR"])
                if not np.isfinite(ep) or not np.isfinite(atr) or atr <= 0:
                    continue
                r = atr * params.atr_mult_stop
                sp = ep - r
                tp1_ = ep + params.tp1_r * r
                tp2_ = ep + params.tp2_r * r

                in_pos = True
                entry_price = ep
                stop_price = sp
                tp1 = tp1_
                tp2 = tp2_
                entry_idx = i
                hit_tp1 = False
                remaining = 1.0
            continue

        # if in position: evaluate exit within day (low/high)
        low = float(row["Low"])
        high = float(row["High"])
        close = float(row["Close"])

        r_unit = (entry_price - stop_price)
        if r_unit <= 0:
            # abnormal, exit
            in_pos = False
            continue

        # time stop check (TP1未達)
        days_in = i - (entry_idx or i)
        time_stop = (days_in >= params.time_stop_days) and (not hit_tp1)

        # 1) stop hit
        stop_hit = low <= stop_price

        # 2) TP1/TP2 hit
        tp1_hit = (not hit_tp1) and (high >= tp1)
        tp2_hit = high >= tp2

        # 3) trailing for remaining after TP1 (close < SMA10)
        trail_hit = hit_tp1 and (close < float(row["SMA_10"]))

        # Execution priority within a day is ambiguous.
        # Conservative assumption:
        # - If stop and TP are both touched same day, assume stop first (worse).
        if stop_hit:
            # full size stop if TP1 not hit yet; else remaining stopped at breakeven? (stop moved)
            # We move stop to breakeven after TP1, so stop_price should be entry_price in that state.
            exit_price = stop_price
            pnl_r = (exit_price - entry_price) / r_unit  # negative or 0
            # realized pnl in R for remaining fraction
            total_r = pnl_r * remaining
            # if TP1 already realized, that part is stored separately in trade record
            # We'll finalize trade now.
            tr = {
                "entry_date": d.index[entry_idx] if entry_idx is not None else date,
                "exit_date": date,
                "entry": entry_price,
                "exit": exit_price,
                "mode": params.entry_mode,
                "tp1_hit": hit_tp1,
                "tp2_hit": False,
                "days": days_in,
                "r": total_r + (0.5 * params.tp1_r if hit_tp1 else 0.0),
            }
            trades.append(tr)
            equity[-1] = equity[-2] + tr["r"]
            in_pos = False
            continue

        if tp1_hit:
            # take half profit at TP1
            hit_tp1 = True
            remaining = 0.5
            # move stop to breakeven
            stop_price = entry_price

        if tp2_hit:
            # exit remaining at TP2
            exit_price = tp2
            pnl_r = (exit_price - entry_price) / r_unit
            total_r = pnl_r * remaining
            tr = {
                "entry_date": d.index[entry_idx] if entry_idx is not None else date,
                "exit_date": date,
                "entry": entry_price,
                "exit": exit_price,
                "mode": params.entry_mode,
                "tp1_hit": hit_tp1,
                "tp2_hit": True,
                "days": days_in,
                "r": total_r + (0.5 * params.tp1_r if hit_tp1 else 0.0),
            }
            trades.append(tr)
            equity[-1] = equity[-2] + tr["r"]
            in_pos = False
            continue

        if trail_hit:
            # exit remaining at close
            exit_price = close
            pnl_r = (exit_price - entry_price) / r_unit
            total_r = pnl_r * remaining
            tr = {
                "entry_date": d.index[entry_idx] if entry_idx is not None else date,
                "exit_date": date,
                "entry": entry_price,
                "exit": exit_price,
                "mode": params.entry_mode,
                "tp1_hit": hit_tp1,
                "tp2_hit": False,
                "days": days_in,
                "r": total_r + (0.5 * params.tp1_r if hit_tp1 else 0.0),
            }
            trades.append(tr)
            equity[-1] = equity[-2] + tr["r"]
            in_pos = False
            continue

        if time_stop:
            # exit full at close
            exit_price = close
            pnl_r = (exit_price - entry_price) / r_unit
            tr = {
                "entry_date": d.index[entry_idx] if entry_idx is not None else date,
                "exit_date": date,
                "entry": entry_price,
                "exit": exit_price,
                "mode": params.entry_mode,
                "tp1_hit": False,
                "tp2_hit": False,
                "days": days_in,
                "r": pnl_r,  # full size (TP1未達)
            }
            trades.append(tr)
            equity[-1] = equity[-2] + tr["r"]
            in_pos = False
            continue

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.Series(equity, index=pd.to_datetime(equity_dates)).astype(float)
    equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]

    if trades_df.empty:
        return BacktestResult(0, 0, 0, 0, 0, 0, _max_drawdown(equity_curve), equity_curve, trades_df)

    wins = trades_df[trades_df["r"] > 0]
    losses = trades_df[trades_df["r"] < 0]
    win_rate = float(len(wins) / len(trades_df)) if len(trades_df) else 0.0
    gross_profit = float(wins["r"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["r"].sum()) if not losses.empty else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy = float(trades_df["r"].mean())
    avg_win = float(wins["r"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["r"].mean()) if not losses.empty else 0.0  # negative
    max_dd = _max_drawdown(equity_curve)

    return BacktestResult(
        n_trades=int(len(trades_df)),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_r=expectancy,
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
        max_drawdown=max_dd,
        equity_curve_r=equity_curve,
        trades=trades_df,
    )


# -------------------------
# 6) パラメータ簡易検証（単一銘柄）
# -------------------------
def grid_search_params(df_ind: pd.DataFrame, base: SwingParams) -> pd.DataFrame:
    """
    1銘柄に対して、軽量なグリッドでパラメータ感度を見る。
    PFとExpectancyを見て、"勝率＋利確"が両立するレンジを見つける用途。
    """
    if df_ind is None or df_ind.empty:
        return pd.DataFrame()

    rsi_lows = [35, 40, 45]
    rsi_highs = [60, 65, 70]
    pb_lows = [-8, -6, -4]
    pb_highs = [-3, -2, -1]
    modes = ["pullback", "breakout"]

    rows = []
    for mode in modes:
        for rl in rsi_lows:
            for rh in rsi_highs:
                if rl >= rh:
                    continue
                for pl in pb_lows:
                    for ph in pb_highs:
                        if pl >= ph:
                            continue
                        params = SwingParams(
                            rsi_low=float(rl),
                            rsi_high=float(rh),
                            pullback_low=float(pl),
                            pullback_high=float(ph),
                            atr_pct_min=base.atr_pct_min,
                            atr_pct_max=base.atr_pct_max,
                            vol_avg20_min=base.vol_avg20_min,
                            require_sma25_over_sma75=base.require_sma25_over_sma75,
                            entry_mode=mode,
                            atr_mult_stop=base.atr_mult_stop,
                            tp1_r=base.tp1_r,
                            tp2_r=base.tp2_r,
                            time_stop_days=base.time_stop_days,
                            breakout_lookback=base.breakout_lookback,
                            breakout_vol_ratio=base.breakout_vol_ratio,
                            risk_pct=base.risk_pct,
                        )
                        bt = backtest_swing(df_ind, params)
                        if bt.n_trades < 8:  # サンプルが少なすぎる組は除外
                            continue
                        rows.append(
                            {
                                "mode": mode,
                                "rsi": f"{rl}-{rh}",
                                "pullback": f"{pl}〜{ph}",
                                "trades": bt.n_trades,
                                "win_rate": bt.win_rate,
                                "pf": bt.profit_factor,
                                "avg_r": bt.expectancy_r,
                                "max_dd": bt.max_drawdown,
                            }
                        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 期待値優先 + PFで二次選別
    out = out.sort_values(["avg_r", "pf", "win_rate"], ascending=[False, False, False]).reset_index(drop=True)
    return out


# -------------------------
# 7) スキャン（全銘柄→候補→上位バックテスト→TOP3）
# -------------------------
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _download_chunk_ohlcv(tickers: Tuple[str, ...], period: str = "6mo") -> pd.DataFrame:
    """Download OHLCV for multiple tickers via Stooq and return MultiIndex columns.
    Columns: level0=ticker, level1=field.
    """
    frames: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = get_market_data(t, period=period)
        if df is None or df.empty:
            continue
        frames[t] = df

    if not frames:
        return pd.DataFrame()

    all_idx = sorted(set().union(*[set(df.index) for df in frames.values()]))
    out = {}
    for t, df in frames.items():
        df2 = df.reindex(all_idx)
        for field in df2.columns:
            out[(t, field)] = df2[field].values

    out_df = pd.DataFrame(out, index=pd.to_datetime(all_idx))
    out_df.columns = pd.MultiIndex.from_tuples(out_df.columns, names=["Ticker", "Field"])
    return out_df.sort_index()

def _extract_one(df_multi: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    _download_chunk_ohlcv() で作った MultiIndex 列から、指定 ticker の OHLCV を抜き出す。
    - 期待する列は (ticker, field) または (field, ticker) のどちらでも来うる
    - ticker 表記ゆれ（例: 7203.T / 7203.JP / 7203）を吸収する
    - 見つからない場合、MultiIndex 全体を返すと下流が壊れるので **空 DF** を返す
    """
    if df_multi is None or df_multi.empty:
        return pd.DataFrame()

    cols = df_multi.columns

    # Single-level columns (already one ticker's frame)
    if not isinstance(cols, pd.MultiIndex):
        return df_multi.copy()

    def _norm(sym: str) -> str:
        s = str(sym).strip().lower()
        # drop common suffixes
        s = re.sub(r"\.(t|jp|us|ns|to|l|hk|ss|sz)$", "", s)
        # keep only first 4 digits for JP stocks if present
        m = re.match(r"^(\d{4})", s)
        if m:
            return m.group(1)
        return s.lstrip("^")

    target_norm = _norm(ticker)
    # candidate norms: original, stripped, etc.
    cand_norms = {target_norm, _norm(ticker.replace(".T", "")), _norm(ticker.replace(".JP", ""))}
    cand_norms = {c for c in cand_norms if c}

    # Build maps for each level
    lvl0 = list(cols.get_level_values(0))
    lvl1 = list(cols.get_level_values(1)) if cols.nlevels >= 2 else []

    map0 = {}
    for v in set(lvl0):
        map0.setdefault(_norm(v), []).append(v)

    map1 = {}
    for v in set(lvl1):
        map1.setdefault(_norm(v), []).append(v)

    # try to decide layout:
    # layout A: (ticker, field) -> level0 has tickers, level1 has fields like Open/High/Low/Close/Volume
    # layout B: (field, ticker) -> level0 has fields, level1 has tickers
    fields = {"open", "high", "low", "close", "volume", "adj close", "adj_close", "adjclose"}

    lvl0_is_fieldish = any(_norm(v) in fields for v in lvl0)
    lvl1_is_fieldish = any(_norm(v) in fields for v in lvl1)

    # pick matching label for ticker from appropriate level
    chosen = None
    if lvl0_is_fieldish and not lvl1_is_fieldish:
        # (field, ticker) => ticker in level1
        for cn in cand_norms:
            if cn in map1:
                chosen = map1[cn][0]
                break
        if chosen is None:
            return pd.DataFrame()
        try:
            out = df_multi.xs(chosen, level=1, axis=1)
        except Exception:
            return pd.DataFrame()
        return out.dropna(how="all").copy()
    else:
        # default: (ticker, field) => ticker in level0
        for cn in cand_norms:
            if cn in map0:
                chosen = map0[cn][0]
                break
        if chosen is None:
            # last resort: direct match
            if ticker in set(lvl0):
                chosen = ticker
            else:
                return pd.DataFrame()
        try:
            out = df_multi.xs(chosen, level=0, axis=1)
        except Exception:
            return pd.DataFrame()
        return out.dropna(how="all").copy()
def _regime_is_ok() -> bool:
    bench = calculate_indicators(get_benchmark_data("^N225", period="5y"), include_sma200=True)
    if bench.empty:
        return True  # ベンチが取れないならスキャンは継続（保守的に止めない）
    last = bench.iloc[-1]
    return bool(last["Close"] > last["SMA_200"])



# -------------------------
# 6.5) セクター事前絞り込み（高速化 & 勝ちやすい地合い寄せ）
# -------------------------
def _parse_sector_list(text: str, allowed: List[str]) -> List[str]:
    """
    AIの出力テキストからセクター名を抽出（許可リストにあるものだけ返す）
    期待フォーマット: 改行区切り or 箇条書き
    """
    if not text:
        return []
    allowed_set = set(allowed)
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip().lstrip("-").lstrip("・").strip()
        if not s:
            continue
        # "○○（理由）" のような括弧を削る
        s = re.sub(r"[（(].*?[）)]", "", s).strip()
        if s in allowed_set and s not in out:
            out.append(s)
    return out


def rank_sectors_quant(
    budget_yen: int,
    vol_avg20_min: float,
    turnover_avg20_min_yen: float = 0.0,
    period: str = "6mo",
    lookback_days: int = 63,      # 約3ヶ月
    reps_per_sector: int = 30,     # 各セクターからサンプルする最大銘柄数（多いほど精度↑・重さ↑）
    min_reps: int = 8,             # セクター評価に必要な最小サンプル
) -> pd.DataFrame:
    """
    数値で“強いセクター（相対強度）”を作る（安定版）:
    - 各セクターから「ティッカー順の先頭」ではなく、セクター内を“均等サンプル”して代表を作る（偏り低減）
    - 3ヶ月リターン（中央値）とボラ（標準偏差）でスコア化
    - 予算（100株）と流動性（出来高 / 売買代金）を反映（実際に買える範囲に寄せる）
    """
    master = get_jpx_master()
    if master.empty:
        return pd.DataFrame()

    # per-sector ticker list
    sector_map: Dict[str, List[str]] = {}
    for _, r in master.iterrows():
        sector_map.setdefault(str(r["sector"]), []).append(str(r["ticker"]))
    for k in list(sector_map.keys()):
        sector_map[k] = sorted(list(dict.fromkeys(sector_map[k])))

    def _sample_even(lst: List[str], k: int) -> List[str]:
        if not lst:
            return []
        if len(lst) <= k:
            return lst
        idxs = np.linspace(0, len(lst) - 1, k).astype(int)
        idxs = sorted(set(int(i) for i in idxs))
        return [lst[i] for i in idxs]

    # cap sampling size for safety
    k_per_sector = int(min(max(int(reps_per_sector), 15), 60))

    reps_rows: List[Tuple[str, str]] = []  # (ticker, sector)
    rep_tickers: List[str] = []
    for sector, lst in sector_map.items():
        sampled = _sample_even(lst, k_per_sector)
        for t in sampled:
            reps_rows.append((t, sector))
            rep_tickers.append(t)

    rep_tickers = list(dict.fromkeys(rep_tickers))
    if not rep_tickers:
        return pd.DataFrame()

    # benchmark（相対強度用）
    bench = get_benchmark_data("^N225", period=period)
    bench_close = bench.get("Close", pd.Series(dtype=float)).dropna()
    bench_ret = np.nan
    if len(bench_close) > lookback_days:
        bench_ret = float(bench_close.iloc[-1] / bench_close.iloc[-lookback_days - 1] - 1)

    # sector -> list of (ret, vol, vol_avg20, turnover_avg20)
    bucket: Dict[str, List[Tuple[float, float, float, float]]] = {}

    stats = {
        "rep_total": len(rep_tickers),
        "budget_ok": 0,
        "fail_budget": 0,
        "data_ok": 0,
        "fail_data_short": 0,
        "fail_liquidity": 0,
    }

    # map ticker->sector for quick lookup
    t2s = {t: s for t, s in reps_rows}

    for c in _chunk(rep_tickers, 50):
        df_multi = _download_chunk_ohlcv(tuple(c), period=period)
        if df_multi is None or df_multi.empty:
            continue

        for t in c:
            df_t = _extract_one(df_multi, t)
            if df_t is None or df_t.empty:
                continue

            if len(df_t) < max(lookback_days + 5, 25):
                stats["fail_data_short"] += 1
                continue

            close = df_t["Close"].dropna()
            if len(close) <= lookback_days:
                stats["fail_data_short"] += 1
                continue

            price = float(close.iloc[-1])
            if not np.isfinite(price) or price <= 0:
                continue

            # 100株が予算内
            if price * 100 > budget_yen:
                stats["fail_budget"] += 1
                continue
            stats["budget_ok"] += 1

            vol_avg20 = float(df_t["Volume"].rolling(20).mean().iloc[-1])
            turnover_avg20 = float((df_t["Close"] * df_t["Volume"]).rolling(20).mean().iloc[-1])

            # liquidity gates
            if not np.isfinite(vol_avg20) or vol_avg20 < float(vol_avg20_min):
                stats["fail_liquidity"] += 1
                continue
            if float(turnover_avg20_min_yen) > 0 and (not np.isfinite(turnover_avg20) or turnover_avg20 < float(turnover_avg20_min_yen)):
                stats["fail_liquidity"] += 1
                continue

            ret_3m = float(close.iloc[-1] / close.iloc[-lookback_days - 1] - 1)

            rets = close.pct_change().dropna()
            vol = float(rets.tail(lookback_days).std()) if len(rets) >= 5 else float("nan")
            if not np.isfinite(vol):
                continue

            sector = str(t2s.get(t, ""))
            if not sector:
                continue

            bucket.setdefault(sector, []).append((ret_3m, vol, vol_avg20, turnover_avg20))
            stats["data_ok"] += 1

    rows = []
    for sector, arr in bucket.items():
        if len(arr) < int(min_reps):
            continue
        ret_med = float(np.median([a[0] for a in arr]))
        vol_med = float(np.median([a[1] for a in arr]))
        vol_med20 = float(np.median([a[2] for a in arr]))
        to_med20 = float(np.median([a[3] for a in arr]))
        excess = float(ret_med - bench_ret) if np.isfinite(bench_ret) else ret_med

        # score: relative strength (excess) primarily, lightly penalize volatility
        score = excess * 100.0 - vol_med * 50.0

        rows.append(
            {
                "sector": sector,
                "n": int(len(arr)),
                "ret_3m": ret_med,
                "excess_vs_n225_3m": excess,
                "vol_3m": vol_med,
                "vol_avg20": vol_med20,
                "turnover_avg20_yen": to_med20,
                "score": score,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    try:
        out.attrs["rank_stats"] = stats
    except Exception:
        pass
    return out

def select_hot_sectors(
    sector_rank: pd.DataFrame,
    top_n: int,
    method: str = "quant",              # "quant" or "ai_overlay"
    api_key: Optional[str] = None,
) -> List[str]:
    """
    - quant: セクタースコア上位 top_n を採用（推奨）
    - ai_overlay: quant上位（最大15）をAIに渡し、top_nに絞る（任意）
    """
    if sector_rank is None or sector_rank.empty:
        return []

    quant = sector_rank["sector"].head(max(top_n, 1)).tolist()

    if method != "ai_overlay" or not api_key:
        return quant[:top_n]

    # AIは“未来予言”ではなく「数値に基づく絞り込み」だけさせる
    allowed = sector_rank["sector"].head(15).tolist()
    table = sector_rank.head(15)[["sector", "excess_vs_n225_3m", "vol_3m", "n"]].to_dict("records")

    prompt = f"""以下は日本株（33業種）の“直近3ヶ月の相対強度(対N225超過)”ランキングです。
あなたの仕事は、短期〜1ヶ月での順張り（押し目/ブレイク）に相性が良いセクターを {top_n} 個に絞ることです。

制約：
- “未来を断言”しない。ここにある数値（相対強度とボラ）に基づき選ぶ。
- 出力はセクター名だけを改行区切りで。理由や文章は不要。

データ（上位15）:
{json.dumps(table, ensure_ascii=False)}
"""

    txt = _call_openai(
        api_key,
        "あなたは定量投資のリサーチャーです。入力の数値以外を持ち込まず、指定形式でのみ出力してください。",
        prompt,
    )
    sectors_ai = _parse_sector_list(txt, allowed)
    if not sectors_ai:
        return quant[:top_n]
    return sectors_ai[:top_n]

def _score_prelim(
    latest: pd.Series,
    params: SwingParams,
    *,
    prev_high: Optional[float] = None,
    pullback_trigger: str = "yday_high",  # "yday_high" or "sma5"
) -> float:
    """バックテスト前の事前スコア（軽い優先度付け）。
    目的は「Stage2に回す上位Kの質を上げる」ことであり、最終判断はバックテストに委ねる。

    - pullback: 乖離の“ちょうど良さ” + 流動性 + 過熱しないRSI + 低〜中ボラ + トレンド強度
    - breakout : ブレイク余裕（ATR基準） + 流動性 + トレンド強度 + RSI + ATR%

    Args:
        latest: indicators計算済みの最終行
        prev_high: breakoutの直近高値（当日を含めない）
        pullback_trigger: pullbackのトリガー種別（スコアに微小反映）
    """

    def sf(x, default=np.nan) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    mode = str(getattr(params, "entry_mode", "pullback"))
    close = sf(latest.get("Close", np.nan))
    rsi = sf(latest.get("RSI", np.nan))
    atr = sf(latest.get("ATR", np.nan))
    atr_pct = sf(latest.get("ATR_PCT", np.nan))
    sma_diff = sf(latest.get("SMA_DIFF", np.nan))
    vol_ratio = sf(latest.get("VOL_RATIO", 1.0), default=1.0)
    to_ratio = sf(latest.get("TURNOVER_RATIO", np.nan))

    # Liquidity score (prefer turnover_ratio when available)
    liq_ratio = to_ratio if np.isfinite(to_ratio) else vol_ratio
    liq_score = min(2.0, max(0.0, liq_ratio)) / 2.0  # 0..1

    # Trend strength: (SMA25/SMA75 - 1) capped
    sma25 = sf(latest.get("SMA_25", np.nan))
    sma75 = sf(latest.get("SMA_75", np.nan))
    ts = 0.0
    if np.isfinite(sma25) and np.isfinite(sma75) and sma75 > 0:
        ts = max(0.0, min(0.20, (sma25 / sma75) - 1.0)) / 0.20  # 0..1

    # RSI score (centered)
    rsi_center = (float(params.rsi_low) + float(params.rsi_high)) / 2.0
    rsi_span = max(5.0, float(params.rsi_high) - float(params.rsi_low))
    rsi_score_center = 1.0 - min(1.0, abs(rsi - rsi_center) / (rsi_span / 2.0 + 5.0))

    # ATR score: prefer lower within range (too高いと飛びやすい)
    if float(params.atr_pct_max) > float(params.atr_pct_min):
        atr_score_low = 1.0 - min(
            1.0,
            max(0.0, atr_pct - float(params.atr_pct_min)) / max(0.5, float(params.atr_pct_max) - float(params.atr_pct_min)),
        )
    else:
        atr_score_low = 0.5

    if mode == "pullback":
        pb_center = (float(params.pullback_low) + float(params.pullback_high)) / 2.0
        pb_span = max(1.0, abs(float(params.pullback_low) - float(params.pullback_high)))
        pb_score = 1.0 - min(1.0, abs(sma_diff - pb_center) / (pb_span / 2.0 + 1.0))

        trig_bonus = 0.03 if str(pullback_trigger) == "yday_high" else 0.0

        score = (
            45.0 * pb_score
            + 20.0 * liq_score
            + 15.0 * rsi_score_center
            + 10.0 * atr_score_low
            + 10.0 * ts
            + 50.0 * trig_bonus
        )
        return float(score)

    # breakout
    br = 0.0
    if prev_high is not None and np.isfinite(prev_high) and np.isfinite(atr) and atr > 0 and np.isfinite(close):
        br = max(0.0, min(2.0, (close - float(prev_high)) / atr)) / 2.0  # 0..1

    # For breakout, slightly prefer RSI above center (within allowed band)
    rsi_score = 0.5 * rsi_score_center + 0.5 * max(0.0, min(1.0, (rsi - rsi_center + 5.0) / 20.0))

    # ATR score for breakout: prefer mid (too低いと抜けない/ too高いと飛ぶ)
    atr_center = (float(params.atr_pct_min) + float(params.atr_pct_max)) / 2.0
    atr_span = max(0.5, float(params.atr_pct_max) - float(params.atr_pct_min))
    atr_score_mid = 1.0 - min(1.0, abs(atr_pct - atr_center) / (atr_span / 2.0 + 0.5))

    score = 35.0 * br + 25.0 * liq_score + 20.0 * ts + 10.0 * rsi_score + 10.0 * atr_score_mid
    return float(score)


def _rank_score(bt: BacktestResult, years: float, params: SwingParams) -> float:
    """Stage2ランキング用の安定スコア（勝率×利確幅×頻度 を同時に設計）。
    - expectancy(R) を主軸にしつつ、少数トレードを縮小し、頻度・PF・DDで整える。
    """
    n = float(max(0, int(bt.n_trades)))
    if n <= 0:
        return float("-inf")

    # Expectancy shrinkage
    shrink_k = float(getattr(params, "score_shrink_k", 20.0))
    shrink = n / (n + max(1.0, shrink_k))
    e_adj = float(bt.expectancy_r) * shrink

    # Frequency (trades per year)
    yrs = float(max(1e-6, years))
    freq = n / yrs
    freq_boost = 1.0 + float(np.log1p(max(0.0, freq)))

    # PF clipping / normalization
    pf_clip = float(getattr(params, "score_pf_clip", 4.0))
    pf = float(bt.profit_factor) if np.isfinite(float(bt.profit_factor)) else 0.0
    pf_c = max(0.0, min(pf_clip, pf))
    pf_norm = 0.5 + 0.5 * (pf_c / max(1e-6, pf_clip))

    # DD penalty (exp)
    dd_ref = float(getattr(params, "score_dd_ref", 0.20))
    dd = float(bt.max_drawdown) if np.isfinite(float(bt.max_drawdown)) else 0.0
    dd_abs = abs(min(0.0, dd))
    dd_pen = float(np.exp(-dd_abs / max(1e-6, dd_ref)))

    return float(e_adj * freq_boost * pf_norm * dd_pen)

def scan_swing_candidates(
    budget_yen: int = 300_000,
    top_n: int = 3,
    params: Optional[SwingParams] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    backtest_period: str = "2y",
    backtest_topk: int = 20,
    sector_prefilter: bool = True,
    sector_top_n: int = 6,
    sector_method: str = "quant",   # "quant" or "ai_overlay"
    api_key: Optional[str] = None,
    relax_level: int = 0,
    auto_relax_trace: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    返り値:
      {
        "regime_ok": bool,
        "candidates": [ {ticker,name,price,rsi,score_pre, bt:{...}} ...] (top_n)
        "prelim_count": int,
        "bt_count": int
      }
    """
    params = params or SwingParams()
    trace: List[Dict[str, object]] = list(auto_relax_trace) if auto_relax_trace else []

    master_all = get_jpx_master()
    if master_all.empty:
        return {
            "regime_ok": True,
            "candidates": [],
            "prelim_count": 0,
            "bt_count": 0,
            "selected_sectors": [],
            "sector_ranking": [],
            "universe": 0,
            "filter_stats": {"error": "JPX master empty"},
            "auto_relax_trace": trace,
            "error": "JPXマスター取得に失敗しました",
        }

    # --- セクター事前絞り込み（任意）---
    sector_rank_records: List[Dict[str, object]] = []
    selected_sectors: List[str] = []

    master = master_all
    if sector_prefilter:
        if progress_callback:
            progress_callback(0, 1, "セクター強度を計算中…")

        sector_rank = rank_sectors_quant(
            budget_yen=int(budget_yen),
            vol_avg20_min=float(params.vol_avg20_min),
            turnover_avg20_min_yen=float(getattr(params, "turnover_avg20_min_yen", 0.0) or 0.0),
            period="6mo",
            lookback_days=63,
            reps_per_sector=30,
            min_reps=8,
        )

        if sector_rank is not None and not sector_rank.empty:
            sector_rank_records = sector_rank.head(15).to_dict("records")

            method_key = "ai_overlay" if str(sector_method).lower().startswith("ai") else "quant"
            selected_sectors = select_hot_sectors(
                sector_rank=sector_rank,
                top_n=int(sector_top_n),
                method=method_key,
                api_key=api_key,
            )
            if selected_sectors:
                master = master_all[master_all["sector"].isin(selected_sectors)].copy()

    # 絞り込みで銘柄ゼロになったらフォールバック
    if master is None or master.empty:
        master = master_all
        selected_sectors = []

    tickers = master["ticker"].tolist()

    # Regime check (N225 > SMA200)
    regime_ok = _regime_is_ok()

    # Optional hard regime filter (auto-relax will disable it once)
    if bool(getattr(params, "regime_filter", False)) and (not regime_ok):
        trace.append(
            {
                "step": "regime_filter_block",
                "regime_ok": False,
                "action": "auto_relax_off" if relax_level == 0 else "blocked",
            }
        )
        if relax_level == 0:
            relaxed = replace(params, regime_filter=False)
            trace.append(
                {
                    "step": "auto_relax",
                    "reason": "regime_filter",
                    "from": {"regime_filter": True},
                    "to": {"regime_filter": False},
                }
            )
            return scan_swing_candidates(
                budget_yen=budget_yen,
                top_n=top_n,
                params=relaxed,
                progress_callback=progress_callback,
                backtest_period=backtest_period,
                backtest_topk=backtest_topk,
                sector_prefilter=False,  # 緩和時はまず全体で拾う
                sector_top_n=sector_top_n,
                sector_method=sector_method,
                api_key=api_key,
                relax_level=1,
                auto_relax_trace=trace,
            )

        return {
            "regime_ok": regime_ok,
            "candidates": [],
            "prelim_count": 0,
            "bt_count": 0,
            "selected_sectors": selected_sectors,
            "sector_ranking": sector_rank_records,
            "universe": len(tickers),
            "filter_stats": {"universe": len(tickers), "regime_block": 1},
            "auto_relax_trace": trace,
            "relax_level": relax_level,
            "params_effective": {"entry_mode": params.entry_mode, "regime_filter": True},
            "error": "地合いフィルタ（N225>SMA200）により候補を出しませんでした",
        }

    # Stage 1: preliminary scan with chunk download (6mo)
    prelim: List[Dict[str, object]] = []

    stats = {
        "universe": len(tickers),
        "chunks": 0,
        "data_ok": 0,
        "budget_ok": 0,
        "trend_ok": 0,
        "rsi_ok": 0,
        "atr_ok": 0,
        "vol_ok": 0,
        "turnover_ok": 0,
        "setup_ok": 0,
        "prelim_pass": 0,
        "fail_data_short": 0,
        "fail_budget": 0,
        "fail_trend": 0,
        "fail_rsi": 0,
        "fail_atr": 0,
        "fail_vol": 0,
        "fail_turnover": 0,
        "fail_setup": 0,
        "bt_tried": 0,
        "fail_bt_trades": 0,
        "ranked_pass": 0,
    }

    chunks = _chunk(tickers, 50)
    total = len(chunks)
    stats["chunks"] = total

    for ci, c in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(ci, total, f"スキャン中 {ci}/{total}")

        df_multi = _download_chunk_ohlcv(tuple(c), period="6mo")
        if df_multi is None or df_multi.empty:
            continue

        for t in c:
            df_t = _extract_one(df_multi, t)
            if df_t is None or df_t.empty or len(df_t) < 90:
                stats["fail_data_short"] += 1
                continue

            ind = calculate_indicators(df_t, include_sma200=False)
            if ind.empty:
                stats["fail_data_short"] += 1
                continue

            stats["data_ok"] += 1
            latest = ind.iloc[-1]

            price = float(latest["Close"])
            if price * 100 > float(budget_yen):
                stats["fail_budget"] += 1
                continue
            stats["budget_ok"] += 1

            # --- trend gate ---
            slope5 = float(latest.get("SMA25_SLOPE5", np.nan))
            slope_ok = bool(np.isfinite(slope5) and slope5 > 0.0)

            if str(params.entry_mode) == "pullback":
                # 押し目では「SMA25の下」に潜るのが普通なので、Close>SMA25は要求しない
                trend_ok = True
                if bool(params.require_sma25_over_sma75):
                    trend_ok = trend_ok and bool(latest["SMA_25"] > latest["SMA_75"])
                # 大局は上（SMA75の上）
                trend_ok = trend_ok and bool(price > latest["SMA_75"])
                trend_ok = trend_ok and slope_ok
            else:
                # ブレイクは強いのでSMA25の上を要求
                trend_ok = bool(price > latest["SMA_25"])
                if bool(params.require_sma25_over_sma75):
                    trend_ok = trend_ok and bool(latest["SMA_25"] > latest["SMA_75"])
                trend_ok = trend_ok and slope_ok

            if not trend_ok:
                stats["fail_trend"] += 1
                continue
            stats["trend_ok"] += 1

            # --- RSI gate ---
            rsi = float(latest["RSI"])
            if not (float(params.rsi_low) <= rsi <= float(params.rsi_high)):
                stats["fail_rsi"] += 1
                continue
            stats["rsi_ok"] += 1

            # --- ATR gate ---
            atr_pct = float(latest["ATR_PCT"])
            if not (float(params.atr_pct_min) <= atr_pct <= float(params.atr_pct_max)):
                stats["fail_atr"] += 1
                continue
            stats["atr_ok"] += 1

            # --- liquidity gates ---
            vol_avg20 = float(latest["VOL_AVG_20"])
            if not (np.isfinite(vol_avg20) and vol_avg20 >= float(params.vol_avg20_min)):
                stats["fail_vol"] += 1
                continue
            stats["vol_ok"] += 1

            turnover_avg20 = float(latest.get("TURNOVER_AVG_20", np.nan))
            to_min = float(getattr(params, "turnover_avg20_min_yen", 0.0) or 0.0)
            if to_min > 0:
                if not (np.isfinite(turnover_avg20) and turnover_avg20 >= to_min):
                    stats["fail_turnover"] += 1
                    continue
            stats["turnover_ok"] += 1

            # --- setup / trigger ---
            score_pre = float("nan")
            trigger_kind = ""

            if str(params.entry_mode) == "pullback":
                sma_diff = float(latest["SMA_DIFF"])
                pb_ok = float(params.pullback_low) <= sma_diff <= float(params.pullback_high)

                # trigger:
                trigger_yday = bool(ind["Close"].iloc[-1] > ind["High"].iloc[-2])
                trigger_sma5 = bool(getattr(params, "pullback_allow_sma5_trigger", True)) and bool(ind["Close"].iloc[-1] > latest["SMA_5"])
                trigger = bool(trigger_yday or trigger_sma5)
                trigger_kind = "yday_high" if trigger_yday else ("sma5" if trigger_sma5 else "")

                if not (pb_ok and trigger):
                    stats["fail_setup"] += 1
                    continue
                stats["setup_ok"] += 1

                score_pre = _score_prelim(latest, params, pullback_trigger=trigger_kind or "sma5")
            else:
                lb = int(params.breakout_lookback)
                if lb < 5:
                    lb = 5

                # prev_high: exclude today
                if len(ind) > lb + 1:
                    prev_high = float(ind["High"].iloc[-lb-1:-1].max())
                else:
                    prev_high = float("nan")

                breakout = bool(np.isfinite(prev_high) and price > prev_high)

                # volume gate: use the larger of vol_ratio and turnover_ratio (when available)
                vr = float(latest.get("VOL_RATIO", 1.0))
                tor = float(latest.get("TURNOVER_RATIO", np.nan))
                liq_ratio = tor if np.isfinite(tor) else vr

                vol_gate = bool(liq_ratio >= float(params.breakout_vol_ratio))

                if not (breakout and vol_gate):
                    stats["fail_setup"] += 1
                    continue
                stats["setup_ok"] += 1

                score_pre = _score_prelim(latest, params, prev_high=prev_high)

            # append prelim
            row_m = master[master["ticker"] == t].iloc[0]
            prelim.append(
                {
                    "ticker": t,
                    "name": str(row_m["name"]),
                    "sector": str(row_m["sector"]),
                    "price": price,
                    "rsi": rsi,
                    "atr": float(latest["ATR"]),
                    "atr_pct": atr_pct,
                    "vol_avg20": vol_avg20,
                    "turnover_avg20_yen": turnover_avg20,
                    "score_pre": float(score_pre) if np.isfinite(float(score_pre)) else 0.0,
                    "trigger": trigger_kind,
                }
            )

    prelim_count = len(prelim)
    stats["prelim_pass"] = prelim_count

    if prelim_count == 0:
        # 自動緩和（1回だけ）：条件が厳しすぎて “0件” になるのを避ける
        if relax_level == 0:
            relaxed = replace(
                params,
                require_sma25_over_sma75=False,
                regime_filter=False,
                rsi_low=max(15.0, float(params.rsi_low) - 7.0),
                rsi_high=min(90.0, float(params.rsi_high) + 7.0),
                pullback_low=float(params.pullback_low) - 5.0,
                pullback_high=float(params.pullback_high) + 3.0,
                atr_pct_min=max(0.5, float(params.atr_pct_min) - 0.8),
                atr_pct_max=min(18.0, float(params.atr_pct_max) + 5.0),
                vol_avg20_min=max(20_000.0, float(params.vol_avg20_min) * 0.5),
                turnover_avg20_min_yen=max(0.0, float(getattr(params, "turnover_avg20_min_yen", 0.0) or 0.0) * 0.5),
            )

            # pullbackで0件のときは breakout に切替（短期で候補を作りやすい）
            if str(params.entry_mode) == "pullback":
                relaxed = replace(
                    relaxed,
                    entry_mode="breakout",
                    breakout_vol_ratio=max(1.15, float(params.breakout_vol_ratio) - 0.35),
                )

            trace.append(
                {
                    "step": "auto_relax",
                    "reason": "prelim_zero",
                    "from": {
                        "entry_mode": str(params.entry_mode),
                        "require_sma25_over_sma75": bool(params.require_sma25_over_sma75),
                        "rsi": [float(params.rsi_low), float(params.rsi_high)],
                        "pullback": [float(params.pullback_low), float(params.pullback_high)],
                        "atr_pct": [float(params.atr_pct_min), float(params.atr_pct_max)],
                        "vol_avg20_min": float(params.vol_avg20_min),
                        "turnover_avg20_min_yen": float(getattr(params, "turnover_avg20_min_yen", 0.0) or 0.0),
                        "breakout_vol_ratio": float(params.breakout_vol_ratio),
                        "regime_filter": bool(getattr(params, "regime_filter", False)),
                    },
                    "to": {
                        "entry_mode": str(relaxed.entry_mode),
                        "require_sma25_over_sma75": bool(relaxed.require_sma25_over_sma75),
                        "rsi": [float(relaxed.rsi_low), float(relaxed.rsi_high)],
                        "pullback": [float(relaxed.pullback_low), float(relaxed.pullback_high)],
                        "atr_pct": [float(relaxed.atr_pct_min), float(relaxed.atr_pct_max)],
                        "vol_avg20_min": float(relaxed.vol_avg20_min),
                        "turnover_avg20_min_yen": float(getattr(relaxed, "turnover_avg20_min_yen", 0.0) or 0.0),
                        "breakout_vol_ratio": float(relaxed.breakout_vol_ratio),
                        "regime_filter": bool(getattr(relaxed, "regime_filter", False)),
                    },
                }
            )

            return scan_swing_candidates(
                budget_yen=budget_yen,
                top_n=top_n,
                params=relaxed,
                progress_callback=progress_callback,
                backtest_period=backtest_period,
                backtest_topk=backtest_topk,
                sector_prefilter=False,
                sector_top_n=sector_top_n,
                sector_method=sector_method,
                api_key=api_key,
                relax_level=1,
                auto_relax_trace=trace,
            )

        return {
            "regime_ok": regime_ok,
            "candidates": [],
            "prelim_count": 0,
            "bt_count": 0,
            "selected_sectors": selected_sectors,
            "sector_ranking": sector_rank_records,
            "universe": len(tickers),
            "filter_stats": stats,
            "auto_relax_trace": trace,
            "relax_level": relax_level,
            "params_effective": {"entry_mode": params.entry_mode},
            "error": "候補が0件でした（条件が厳しい/データ欠損の可能性）",
        }

    # Backtest topK (keep a bit more than top_n)
    prelim_sorted = sorted(prelim, key=lambda x: float(x.get("score_pre", 0.0)), reverse=True)[: max(int(backtest_topk), int(top_n))]
    stats["bt_tried"] = len(prelim_sorted)

    ranked: List[Dict[str, object]] = []
    for i, item in enumerate(prelim_sorted, start=1):
        if progress_callback:
            progress_callback(i, stats["bt_tried"], f"バックテスト {item['ticker']}")

        df = get_market_data(item["ticker"], period=backtest_period, interval="1d")
        ind = calculate_indicators(df, include_sma200=False)
        if ind.empty:
            continue

        bt = backtest_swing(ind, params)

        # stabilize: require minimum trades (avoid sample noise)
        if int(bt.n_trades) < int(getattr(params, "min_trades_bt", 8)):
            stats["fail_bt_trades"] += 1
            continue

        years = max(0.5, float(len(ind)) / 252.0)
        bt_score = _rank_score(bt, years, params)

        ranked.append(
            {
                **item,
                "bt_trades": bt.n_trades,
                "bt_win_rate": bt.win_rate,
                "bt_pf": bt.profit_factor,
                "bt_avg_r": bt.expectancy_r,
                "bt_max_dd": bt.max_drawdown,
                "bt_years": years,
                "bt_score": bt_score,
            }
        )

    stats["ranked_pass"] = len(ranked)

    if not ranked:
        # まだ0件なら、min_tradesだけ緩める（1回だけ）
        if relax_level == 0 and int(getattr(params, "min_trades_bt", 8)) > 3:
            relaxed = replace(params, min_trades_bt=max(3, int(getattr(params, "min_trades_bt", 8)) // 2))
            trace.append(
                {
                    "step": "auto_relax",
                    "reason": "bt_trades_zero",
                    "from": {"min_trades_bt": int(getattr(params, "min_trades_bt", 8))},
                    "to": {"min_trades_bt": int(getattr(relaxed, "min_trades_bt", 3))},
                }
            )
            return scan_swing_candidates(
                budget_yen=budget_yen,
                top_n=top_n,
                params=relaxed,
                progress_callback=progress_callback,
                backtest_period=backtest_period,
                backtest_topk=backtest_topk,
                sector_prefilter=False,
                sector_top_n=sector_top_n,
                sector_method=sector_method,
                api_key=api_key,
                relax_level=1,
                auto_relax_trace=trace,
            )

        return {
            "regime_ok": regime_ok,
            "candidates": [],
            "prelim_count": prelim_count,
            "bt_count": 0,
            "selected_sectors": selected_sectors,
            "sector_ranking": sector_rank_records,
            "universe": len(tickers),
            "filter_stats": stats,
            "auto_relax_trace": trace,
            "relax_level": relax_level,
            "params_effective": {"entry_mode": params.entry_mode},
            "error": "バックテストで有効な候補が0件でした（トレード数不足/データ欠損の可能性）",
        }

    # Final ranking by stable score -> expectancy -> PF -> trades
    ranked = sorted(
        ranked,
        key=lambda x: (
            float(x.get("bt_score", float("-inf"))),
            float(x.get("bt_avg_r", 0.0)),
            float(x.get("bt_pf", 0.0)) if np.isfinite(float(x.get("bt_pf", 0.0))) else 0.0,
            float(x.get("bt_trades", 0)),
        ),
        reverse=True,
    )

    return {
        "regime_ok": regime_ok,
        "selected_sectors": selected_sectors,
        "sector_ranking": sector_rank_records,
        "universe": len(tickers),
        "candidates": ranked[: int(top_n)],
        "prelim_count": prelim_count,
        "bt_count": len(ranked),
        "filter_stats": stats,
        "auto_relax_trace": trace,
        "relax_level": relax_level,
        "params_effective": {
            "entry_mode": str(params.entry_mode),
            "require_sma25_over_sma75": bool(params.require_sma25_over_sma75),
            "regime_filter": bool(getattr(params, "regime_filter", False)),
            "rsi": [float(params.rsi_low), float(params.rsi_high)],
            "pullback": [float(params.pullback_low), float(params.pullback_high)],
            "atr_pct": [float(params.atr_pct_min), float(params.atr_pct_max)],
            "vol_avg20_min": float(params.vol_avg20_min),
            "turnover_avg20_min_yen": float(getattr(params, "turnover_avg20_min_yen", 0.0) or 0.0),
            "breakout_vol_ratio": float(params.breakout_vol_ratio),
            "breakout_lookback": int(params.breakout_lookback),
            "min_trades_bt": int(getattr(params, "min_trades_bt", 8)),
            "score_shrink_k": float(getattr(params, "score_shrink_k", 20.0)),
            "score_pf_clip": float(getattr(params, "score_pf_clip", 4.0)),
            "score_dd_ref": float(getattr(params, "score_dd_ref", 0.20)),
        },
    }


# Backward-compatible alias for main.py

# Backward-compatible alias for main.py
def auto_scan_value_stocks(api_key: str, progress_callback=None):
    """
    旧main.py互換：戻り値は (sectors, candidates)。
    v4以降は「セクター事前絞り込み」を内部で行うため、選ばれたセクター一覧も返す。
    """
    # api_key is not required for scan; kept for compatibility
    res = scan_swing_candidates(
        budget_yen=300_000,
        top_n=3,
        params=SwingParams(entry_mode="pullback"),
        progress_callback=progress_callback,
        backtest_period="2y",
        backtest_topk=20,
    )
    return res.get("selected_sectors", []), res.get("candidates", [])


# -------------------------
# 8) 注文書（アルゴリズム版）
# -------------------------
def build_trade_plan(
    df_ind: pd.DataFrame,
    params: SwingParams,
    capital_yen: int,
    risk_pct: float,
    allow_partial_tp: bool = True,
) -> Dict[str, object]:
    """
    現在の指標から、次のエントリーを想定した注文書を数値で生成。
    - entry: 次営業日の寄り（推定）として、最新Closeをベースに提示（実務は指値や寄成）
    - stop: ATR*mult
    - tp1/tp2: Rベース
    - shares: 100株単位で、リスク額・資金額の両方を満たす最大株数を算出
    """
    if df_ind is None or df_ind.empty:
        return {}

    last = df_ind.iloc[-1]
    price = float(last["Close"])
    atr = float(last["ATR"])
    if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
        return {}

    r = atr * params.atr_mult_stop
    entry = price  # proxy
    stop = entry - r
    tp1 = entry + params.tp1_r * r
    tp2 = entry + params.tp2_r * r

    # position sizing
    risk_yen = capital_yen * (risk_pct / 100.0)
    per_share_risk = max(1e-9, entry - stop)
    raw_shares = math.floor(risk_yen / per_share_risk)

    # unit: 100 shares
    unit_shares = (raw_shares // 100) * 100
    if unit_shares < 100:
        unit_shares = 0

    # capital constraint
    if unit_shares > 0:
        max_by_cap = (capital_yen // (entry * 100)) * 100
        unit_shares = int(min(unit_shares, max_by_cap))

    plan = {
        "entry_price": float(entry),
        "stop_price": float(stop),
        "tp1_price": float(tp1),
        "tp2_price": float(tp2),
        "r_yen_per_share": float(per_share_risk),
        "risk_yen": float(risk_yen),
        "shares": int(unit_shares),
        "time_stop_days": int(params.time_stop_days),
        "mode": params.entry_mode,
        "atr_mult_stop": float(params.atr_mult_stop),
        "tp1_r": float(params.tp1_r),
        "tp2_r": float(params.tp2_r),
    }
    return plan
