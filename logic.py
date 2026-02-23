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
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
        df = _normalize_yf_df(df)
        if isinstance(df.columns, pd.MultiIndex):
            # sometimes single ticker still returns multi-index
            df.columns = df.columns.droplevel(1)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_data(symbol: str = "^N225", period: str = "5y") -> pd.DataFrame:
    return get_market_data(symbol, period=period, interval="1d")


# -------------------------
# 3) 指標計算
# -------------------------
def calculate_indicators(df: pd.DataFrame, include_sma200: bool = False) -> pd.DataFrame:
    """指標計算。
    重要: 6mo 等の短い期間では SMA200 が作れないため、include_sma200=False で呼ぶこと。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # SMAs
    for w in (5, 10, 25, 75):
        d[f"SMA_{w}"] = d["Close"].rolling(window=w).mean()
    if include_sma200:
        d["SMA_200"] = d["Close"].rolling(window=200).mean()

    # RSI(14)
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

    # Volume
    d["VOL_AVG_20"] = d["Volume"].rolling(window=20).mean()
    d["VOL_RATIO"] = d["Volume"] / d["VOL_AVG_20"]

    # Range
    d["HIGH_20"] = d["High"].rolling(20).max()
    d["LOW_20"] = d["Low"].rolling(20).min()

    # dropna only on required columns (avoid wiping out short-period data)
    base_cols = ["Close", "High", "Low", "Volume", "SMA_25", "SMA_75", "RSI", "ATR", "ATR_PCT",
                 "SMA_DIFF", "VOL_AVG_20", "VOL_RATIO", "HIGH_20", "LOW_20", "SMA25_SLOPE5", "SMA75_SLOPE10"]
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
    # entry filters
    rsi_low: float = 40.0
    rsi_high: float = 65.0
    pullback_low: float = -6.0     # SMA25乖離の下限（より下げ過ぎは避ける）
    pullback_high: float = -1.0    # SMA25乖離の上限
    atr_pct_min: float = 1.0
    atr_pct_max: float = 6.0
    vol_avg20_min: float = 100_000.0

    # trend filters
    require_sma25_over_sma75: bool = True

    # entry trigger type: "pullback" or "breakout"
    entry_mode: str = "pullback"

    # exit rules
    atr_mult_stop: float = 1.5
    tp1_r: float = 1.0
    tp2_r: float = 3.0
    time_stop_days: int = 10

    # breakout config
    breakout_lookback: int = 20
    breakout_vol_ratio: float = 1.5

    # risk sizing
    risk_pct: float = 2.0  # for position sizing (UI can override)


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


@st.cache_data(ttl=3600, show_spinner=False)
def _download_chunk_ohlcv(tickers: Tuple[str, ...], period: str = "6mo") -> pd.DataFrame:
    # yfinance multi ticker download
    try:
        df = yf.download(
            tickers=list(tickers),
            period=period,
            interval="1d",
            progress=False,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
        return _normalize_yf_df(df)
    except Exception:
        return pd.DataFrame()


def _extract_one(df_multi: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df_multi is None or df_multi.empty:
        return pd.DataFrame()
    if isinstance(df_multi.columns, pd.MultiIndex):
        # columns: (Field, Ticker) or (Ticker, Field) depending on yfinance; handle both
        lvl0 = df_multi.columns.get_level_values(0)
        lvl1 = df_multi.columns.get_level_values(1)
        if ticker in set(lvl0):
            sub = df_multi[ticker].copy()
            sub.columns.name = None
            return sub.dropna()
        if ticker in set(lvl1):
            sub = df_multi.xs(ticker, axis=1, level=1).copy()
            sub.columns.name = None
            return sub.dropna()
    # single ticker
    return df_multi.copy().dropna()


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


@st.cache_data(ttl=3600, show_spinner=False)
def rank_sectors_quant(
    budget_yen: int,
    vol_avg20_min: float,
    period: str = "6mo",
    lookback_days: int = 63,      # 約3ヶ月
    reps_per_sector: int = 30,     # 各セクター代表銘柄数（多いほど精度↑・重さ↑）
    min_reps: int = 8,             # セクター評価に必要な最小サンプル
) -> pd.DataFrame:
    """
    「AIで未来を当てる」ではなく、まずは数値で“強いセクター（相対強度）”を作る。
    - 各セクターから代表銘柄(reps)を抽出
    - 3ヶ月リターン（中央値）とボラ（標準偏差）でスコア化
    - 予算（100株）と出来高下限も反映（あなたが実際に買える範囲に寄せる）
    """
    master = get_jpx_master()
    if master.empty:
        return pd.DataFrame()

    reps = master.sort_values("ticker").groupby("sector", as_index=False).head(reps_per_sector).copy()
    rep_tickers = reps["ticker"].tolist()
    if not rep_tickers:
        return pd.DataFrame()

    # benchmark（相対強度用）
    bench = get_benchmark_data("^N225", period=period)
    bench_close = bench.get("Close", pd.Series(dtype=float)).dropna()
    bench_ret = np.nan
    if len(bench_close) > lookback_days:
        bench_ret = float(bench_close.iloc[-1] / bench_close.iloc[-lookback_days - 1] - 1)

    # sector -> list of (ret, vol, vol_avg20)
    bucket: Dict[str, List[Tuple[float, float, float]]] = {}

    # debug stats for sector ranking (kept minimal)
    stats = {
        'rep_total': len(rep_tickers),
        'budget_ok': 0,
        'fail_budget': 0,
        'data_ok': 0,
        'fail_data_short': 0,
    }

    # download reps in chunks
    for c in _chunk(rep_tickers, 50):
        df_multi = _download_chunk_ohlcv(tuple(c), period=period)
        if df_multi is None or df_multi.empty:
            continue

        for t in c:
            df_t = _extract_one(df_multi, t)
            if df_t is None or df_t.empty:
                continue

            # need enough bars for vol + lookback + vol_avg20
            if len(df_t) < max(lookback_days + 5, 25):
                continue

            close = df_t["Close"].dropna()
            if len(close) <= lookback_days:
                continue

            price = float(close.iloc[-1])
            if not np.isfinite(price) or price <= 0:
                continue

            # 100株が予算内
            if price * 100 > budget_yen:
                stats['fail_budget'] += 1
                continue
            stats['budget_ok'] += 1

            vol_avg20 = float(df_t["Volume"].rolling(20).mean().iloc[-1])
            if not np.isfinite(vol_avg20) or vol_avg20 < vol_avg20_min:
                continue

            ret_3m = float(close.iloc[-1] / close.iloc[-lookback_days - 1] - 1)

            rets = close.pct_change().dropna()
            vol = float(rets.tail(lookback_days).std()) if len(rets) >= 5 else float("nan")
            if not np.isfinite(vol):
                continue

            # sector lookup
            sector_row = master[master["ticker"] == t]
            if sector_row.empty:
                continue
            sector = str(sector_row.iloc[0]["sector"])
            bucket.setdefault(sector, []).append((ret_3m, vol, vol_avg20))

    rows = []
    for sector, arr in bucket.items():
        if len(arr) < min_reps:
            continue
        ret_med = float(np.median([a[0] for a in arr]))
        vol_med = float(np.median([a[1] for a in arr]))
        vol_med20 = float(np.median([a[2] for a in arr]))
        excess = float(ret_med - bench_ret) if np.isfinite(bench_ret) else ret_med

        # スコア：相対強度（超過リターン）を主に、ボラで軽く罰則
        score = excess * 100.0 - vol_med * 50.0

        rows.append(
            {
                "sector": sector,
                "n": int(len(arr)),
                "ret_3m": ret_med,
                "excess_vs_n225_3m": excess,
                "vol_3m": vol_med,
                "vol_avg20": vol_med20,
                "score": score,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    # attach stats for optional UI display
    try:
        out.attrs['rank_stats'] = stats
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

def _score_prelim(latest: pd.Series, params: SwingParams) -> float:
    """
    事前スコア（バックテスト前）：
    - 押し目の深さ（ちょうど良いレンジに近いほど）
    - 出来高比（大きいほど）
    - RSIが中庸ほど（過熱しすぎない）
    """
    sma_diff = float(latest["SMA_DIFF"])
    vol_ratio = float(latest.get("VOL_RATIO", 1.0))
    rsi = float(latest["RSI"])

    # pullback: range center
    pb_center = (params.pullback_low + params.pullback_high) / 2
    pb_score = 1.0 - min(1.0, abs(sma_diff - pb_center) / max(1.0, abs(params.pullback_low - pb_center)))

    rsi_center = (params.rsi_low + params.rsi_high) / 2
    rsi_score = 1.0 - min(1.0, abs(rsi - rsi_center) / max(1.0, abs(params.rsi_high - rsi_center)))

    vol_score = min(2.0, max(0.0, vol_ratio)) / 2.0

    atr_pct = float(latest["ATR_PCT"])
    atr_center = (params.atr_pct_min + params.atr_pct_max) / 2
    atr_score = 1.0 - min(1.0, abs(atr_pct - atr_center) / max(0.5, abs(params.atr_pct_max - atr_center)))

    return float(50 * pb_score + 30 * vol_score + 10 * rsi_score + 10 * atr_score)


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


    master_all = get_jpx_master()
    if master_all.empty:
        return {"regime_ok": True, "candidates": [], "prelim_count": 0, "bt_count": 0, "selected_sectors": [], "sector_ranking": [], "error": "JPXマスター取得に失敗しました"}

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
            period="6mo",
            lookback_days=63,
            reps_per_sector=30,
            min_reps=8,
        )

        if sector_rank is not None and not sector_rank.empty:
            # UI表示用（上位15）
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

    regime_ok = _regime_is_ok()

    # Stage 1: preliminary scan with chunk download (6mo)
    prelim: List[Dict[str, object]] = []

    # debug stats: where candidates are filtered out
    stats = {
        'universe': len(tickers),
        'chunks': total if 'total' in locals() else None,
        'data_ok': 0,
        'budget_ok': 0,
        'trend_ok': 0,
        'rsi_ok': 0,
        'atr_ok': 0,
        'vol_ok': 0,
        'setup_ok': 0,
        'prelim_pass': 0,
        'fail_data_short': 0,
        'fail_budget': 0,
        'fail_trend': 0,
        'fail_rsi': 0,
        'fail_atr': 0,
        'fail_vol': 0,
        'fail_setup': 0,
    }
    chunks = _chunk(tickers, 50)

    total = len(chunks)
    stats['chunks'] = total
    for ci, c in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(ci, total, f"スキャン中 {ci}/{total}")

        df_multi = _download_chunk_ohlcv(tuple(c), period="6mo")
        if df_multi is None or df_multi.empty:
            continue

        for t in c:
            df_t = _extract_one(df_multi, t)
            if df_t is None or df_t.empty or len(df_t) < 90:
                continue
            ind = calculate_indicators(df_t, include_sma200=False)
            if ind.empty:
                stats['fail_data_short'] += 1
                continue
            stats['data_ok'] += 1
            latest = ind.iloc[-1]

            price = float(latest["Close"])
            # 100株の投資単位が予算内か
            if price * 100 > budget_yen:
                stats['fail_budget'] += 1
                continue
            stats['budget_ok'] += 1

            # filters
            # trend / regime (entry_modeで条件を分ける)
            slope5 = float(latest.get("SMA25_SLOPE5", np.nan))
            slope_ok = bool(np.isfinite(slope5) and slope5 > 0.0)

            if params.entry_mode == "pullback":
                # 押し目では「SMA25の下」に潜るのが普通なので、Close>SMA25は要求しない
                trend_ok = True
                if params.require_sma25_over_sma75:
                    trend_ok = trend_ok and bool(latest["SMA_25"] > latest["SMA_75"])
                # 大局は上（SMA75の上）
                trend_ok = trend_ok and bool(price > latest["SMA_75"])
                trend_ok = trend_ok and slope_ok
            else:
                # ブレイクは強いのでSMA25の上を要求
                trend_ok = bool(price > latest["SMA_25"])
                if params.require_sma25_over_sma75:
                    trend_ok = trend_ok and bool(latest["SMA_25"] > latest["SMA_75"])
                trend_ok = trend_ok and slope_ok

            rsi = float(latest["RSI"])
            rsi_ok = params.rsi_low <= rsi <= params.rsi_high

            atr_pct = float(latest["ATR_PCT"])
            atr_ok = params.atr_pct_min <= atr_pct <= params.atr_pct_max

            vol_avg20 = float(latest["VOL_AVG_20"])
            vol_ok = vol_avg20 >= params.vol_avg20_min

            if not trend_ok:
                stats['fail_trend'] += 1
                continue
            stats['trend_ok'] += 1

            if not rsi_ok:
                stats['fail_rsi'] += 1
                continue
            stats['rsi_ok'] += 1

            if not atr_ok:
                stats['fail_atr'] += 1
                continue
            stats['atr_ok'] += 1

            if not vol_ok:
                stats['fail_vol'] += 1
                continue
            stats['vol_ok'] += 1

            if params.entry_mode == "pullback":
                sma_diff = float(latest["SMA_DIFF"])
                pb_ok = params.pullback_low <= sma_diff <= params.pullback_high
                # trigger: close > yesterday high
                trigger = bool(ind["Close"].iloc[-1] > ind["High"].iloc[-2])
                if not (pb_ok and trigger):
                    stats['fail_setup'] += 1
                    continue
                stats['setup_ok'] += 1
            else:
                lb = params.breakout_lookback
                prev_high = float(ind["High"].iloc[-lb-1:-1].max()) if len(ind) > lb+1 else float("nan")
                breakout = bool(price > prev_high) if np.isfinite(prev_high) else False
                vr = float(latest["VOL_RATIO"])
                if not (breakout and vr >= params.breakout_vol_ratio):
                    stats['fail_setup'] += 1
                    continue
                stats['setup_ok'] += 1

            score_pre = _score_prelim(latest, params)

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
                    "score_pre": score_pre,
                }
            )

    prelim_count = len(prelim)
    stats['prelim_pass'] = prelim_count
    if prelim_count == 0:
        # 自動緩和（1回だけ）：条件が厳しすぎて “0件” になるのを避ける
        if relax_level == 0:
            relaxed = SwingParams()
            # copy current
            for k in [
                "rsi_low","rsi_high","pullback_low","pullback_high","atr_pct_min","atr_pct_max","vol_avg20_min",
                "require_sma25_over_sma75","entry_mode","atr_mult_stop","tp1_r","tp2_r","time_stop_days",
                "breakout_lookback","breakout_vol_ratio","risk_pct"
            ]:
                setattr(relaxed, k, getattr(params, k))

            # relax knobs
            relaxed.require_sma25_over_sma75 = False
            relaxed.rsi_low = max(20.0, float(params.rsi_low) - 5.0)
            relaxed.rsi_high = min(85.0, float(params.rsi_high) + 5.0)
            relaxed.pullback_low = float(params.pullback_low) - 4.0
            relaxed.pullback_high = float(params.pullback_high) + 2.0
            relaxed.atr_pct_min = max(0.5, float(params.atr_pct_min) - 0.5)
            relaxed.atr_pct_max = min(15.0, float(params.atr_pct_max) + 4.0)
            relaxed.vol_avg20_min = max(20_000.0, float(params.vol_avg20_min) * 0.5)

            # pullbackで0件のときは breakout に切替（短期で候補を作りやすい）
            if str(params.entry_mode) == "pullback":
                relaxed.entry_mode = "breakout"
                relaxed.breakout_vol_ratio = max(1.2, float(params.breakout_vol_ratio) - 0.3)

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
            "error": "候補が0件でした（条件が厳しい/データ取得失敗の可能性）",
        }

    prelim_sorted = sorted(prelim, key=lambda x: float(x["score_pre"]), reverse=True)[: max(backtest_topk, top_n)]
    bt_count = len(prelim_sorted)

    # Stage 2: backtest topK and re-rank by expectancy/PF
    ranked: List[Dict[str, object]] = []
    for i, item in enumerate(prelim_sorted, start=1):
        if progress_callback:
            progress_callback(i, bt_count, f"バックテスト {item['ticker']}")

        df = get_market_data(item["ticker"], period=backtest_period, interval="1d")
        ind = calculate_indicators(df, include_sma200=False)
        bt = backtest_swing(ind, params)

        ranked.append(
            {
                **item,
                "bt_trades": bt.n_trades,
                "bt_win_rate": bt.win_rate,
                "bt_pf": bt.profit_factor,
                "bt_avg_r": bt.expectancy_r,
                "bt_max_dd": bt.max_drawdown,
            }
        )

    # rank: expectancy -> PF -> trades
    ranked = sorted(
        ranked,
        key=lambda x: (float(x["bt_avg_r"]), float(x["bt_pf"]) if np.isfinite(float(x["bt_pf"])) else 0.0, float(x["bt_trades"])),
        reverse=True,
    )

    return {
        "regime_ok": regime_ok,
        "selected_sectors": selected_sectors,
        "sector_ranking": sector_rank_records,
        "universe": len(tickers),
        "candidates": ranked[:top_n],
        "prelim_count": prelim_count,
        "bt_count": bt_count,
        "filter_stats": stats,
        "relax_level": relax_level,
        "params_effective": {
            "entry_mode": params.entry_mode,
            "require_sma25_over_sma75": params.require_sma25_over_sma75,
            "rsi": [params.rsi_low, params.rsi_high],
            "pullback": [params.pullback_low, params.pullback_high],
            "atr_pct": [params.atr_pct_min, params.atr_pct_max],
            "vol_avg20_min": params.vol_avg20_min,
            "breakout_vol_ratio": params.breakout_vol_ratio,
        },
    }


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
