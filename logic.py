# -*- coding: utf-8 -*-
"""
logic.py  (Buffett-Style Value + Timing, bug-fixed, backward-compatible)

このファイルは main.py から呼ばれる「ロジック層」です。
- 既存の関数シグネチャ（main.py依存）を維持
- RSI比較のSeriesバグを修正（全銘柄スキップ問題の根本原因）
- 「バフェット型（割安×質×安全性）」のスコアリングを追加
- スキャン結果が0になり続けるのを避けるため、厳格ヒットが0でも「上位候補TOP3」を返す設計
  ※候補dictに hit=True/False を付与（main.pyは無視しても動く）

注意:
- yfinanceの財務指標は銘柄により欠損が多いです（特に日本株）。
  欠損は「除外」ではなく「中立スコア」扱いにして、候補が消えないようにしています。
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from openai import OpenAI

TOKYO = pytz.timezone("Asia/Tokyo")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -------------- ユーティリティ --------------

def _now_tokyo() -> datetime:
    return datetime.now(TOKYO)

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            if math.isnan(x):
                return None
            return float(x)
        xf = float(x)
        if math.isnan(xf):
            return None
        return xf
    except Exception:
        return None

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# -------------- JPXマスター --------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_jpx_master() -> pd.DataFrame:
    """
    JPX公式XLSから「ticker/name/sector/market」を作る。
    失敗したら空DataFrame。
    """
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        try:
            df = pd.read_excel(url, engine="xlrd")
        except Exception:
            df = pd.read_excel(url)

        if "33業種区分" not in df.columns or "コード" not in df.columns or "銘柄名" not in df.columns:
            return pd.DataFrame()

        df = df[df["33業種区分"] != "-"].copy()

        market_col_candidates = ["市場・商品区分", "市場区分", "市場"]
        market_col = next((c for c in market_col_candidates if c in df.columns), None)
        if market_col is None:
            df["market"] = ""
        else:
            df["market"] = df[market_col].astype(str)

        df["ticker"] = df["コード"].astype(str) + ".T"
        df = df.rename(columns={"銘柄名": "name", "33業種区分": "sector"})

        out = df[["ticker", "name", "sector", "market"]].copy()
        out["sector"] = out["sector"].astype(str)
        out["name"] = out["name"].astype(str)
        out["ticker"] = out["ticker"].astype(str)
        return out

    except Exception:
        return pd.DataFrame()

def get_company_name(ticker: str) -> str:
    df_master = get_jpx_master()
    if not df_master.empty:
        match = df_master[df_master["ticker"] == ticker]
        if not match.empty:
            return str(match.iloc[0]["name"])
    return ticker

# -------------- OpenAI --------------

def call_openai(api_key: str, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(temperature),
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI API エラー: {str(e)}"

def get_promising_sectors(api_key: str, all_sectors: List[str], n: int = 6) -> List[str]:
    """
    マクロ観点で有望業種を選ぶ（ブレ抑制: temperature=0）。
    OpenAIが不調でも、必ずフォールバックで動作。
    """
    fallback = ["電気機器", "情報・通信業", "機械", "医薬品", "銀行業", "小売業"]
    fallback = [s for s in fallback if s in all_sectors] or all_sectors[: min(len(all_sectors), n)]

    if not api_key:
        return fallback[:n]

    sectors_str = ", ".join(all_sectors)
    system_prompt = (
        "あなたはマクロ経済ストラテジストです。"
        "必ず JSON 配列のみで回答してください。余計な文章は禁止。"
    )
    user_prompt = (
        f"金利・為替・景気循環の観点で、現在の日本株で相対的に有望と思われる業種を{n}個選んでください。"
        f"候補（33業種区分）: {sectors_str}\n"
        f"出力例: [\"電気機器\", \"銀行業\", ...]"
    )

    try:
        res = call_openai(api_key, system_prompt, user_prompt, temperature=0.0)
        s, e = res.find("["), res.rfind("]")
        if s != -1 and e != -1:
            chosen = json.loads(res[s : e + 1])
            chosen = [c for c in chosen if isinstance(c, str)]
            chosen = [c for c in chosen if c in all_sectors]
            if chosen:
                for f in fallback:
                    if len(chosen) >= n:
                        break
                    if f not in chosen:
                        chosen.append(f)
                return chosen[:n]
    except Exception:
        pass

    return fallback[:n]

# -------------- Market Data / Indicators --------------

def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinanceの返り値は環境で列形式が違うことがあるので正規化。
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # 念のため
        df.columns = df.columns.droplevel(-1)
    return df

def _yahoo_chart(ticker: str, rng: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=rng, interval=interval, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return None
        return _normalize_ohlcv_columns(df)
    except Exception:
        return None

def _bulk_yahoo_chart(tickers: List[str], rng: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    まとめてダウンロードしてリクエスト数を削減。
    失敗しやすいので、失敗時は空dictを返し、呼び出し側で個別取得にフォールバック。
    """
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    try:
        df = yf.download(
            tickers,
            period=rng,
            interval=interval,
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if df is None or df.empty:
            return out

        # 単一tickerだと通常形になる
        if not isinstance(df.columns, pd.MultiIndex):
            out[tickers[0]] = _normalize_ohlcv_columns(df)
            return out

        # MultiIndex: (ticker, field) or (field, ticker)
        lv0 = set(df.columns.get_level_values(0).astype(str))
        lv1 = set(df.columns.get_level_values(1).astype(str))

        for t in tickers:
            try:
                if t in lv0:
                    sub = df[t].copy()
                elif t in lv1:
                    sub = df.xs(t, level=1, axis=1).copy()
                else:
                    continue
                # 欠損列があると後続が死ぬので必須列チェック
                need = {"Open", "High", "Low", "Close", "Volume"}
                if not need.issubset(set(map(str, sub.columns))):
                    continue
                out[t] = sub
            except Exception:
                continue

        return out

    except Exception:
        return {}

def get_market_data(ticker: str = "8306.T", rng: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    return _yahoo_chart(ticker, rng, interval)

def calculate_indicators(df: pd.DataFrame, benchmark_raw: pd.DataFrame = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()

    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_25"] = df["Close"].rolling(window=25).mean()
    df["SMA_75"] = df["Close"].rolling(window=75).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = (-delta.clip(upper=0)).abs()

    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean() + 1e-9
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    tr = pd.concat(
        [
            (df["High"] - df["Low"]),
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    df["SMA_DIFF"] = (df["Close"] - df["SMA_25"]) / df["SMA_25"] * 100

    if benchmark_raw is not None and isinstance(benchmark_raw, pd.DataFrame) and not benchmark_raw.empty:
        df["BENCHMARK"] = benchmark_raw["Close"].reindex(df.index, method="ffill")
    else:
        df["BENCHMARK"] = 0.0

    return df.dropna()

def judge_condition(price, sma5, sma25, sma75, rsi):
    short = {"status": "上昇継続 (SMA5上)", "color": "blue"} if price > sma5 else {"status": "勢い鈍化 (SMA5下)", "color": "red"}
    mid = {"status": "静観", "color": "gray"}
    if sma25 > sma75 and rsi < 70:
        mid = {"status": "上昇トレンド (押し目買い)", "color": "blue"}
    elif rsi <= 30:
        mid = {"status": "売られすぎ (反発警戒)", "color": "orange"}
    return {"short": short, "mid": mid}

# -------------- Fundamentals (Buffett-ish) --------------

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_FUND_CACHE_PATH = _CACHE_DIR / "fundamentals_cache.json"

def _load_fund_cache() -> Dict[str, Any]:
    try:
        if _FUND_CACHE_PATH.exists():
            return json.loads(_FUND_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_fund_cache(cache: Dict[str, Any]) -> None:
    try:
        _FUND_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _fetch_fundamentals_yf(ticker: str) -> Dict[str, Any]:
    """
    yfinanceから財務指標を取得（欠損多い前提）。
    失敗しても例外にせず空dictを返す。
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        data = {
            "trailingPE": _safe_float(info.get("trailingPE")),
            "forwardPE": _safe_float(info.get("forwardPE")),
            "priceToBook": _safe_float(info.get("priceToBook")),
            "returnOnEquity": _safe_float(info.get("returnOnEquity")),
            "debtToEquity": _safe_float(info.get("debtToEquity")),
            "currentRatio": _safe_float(info.get("currentRatio")),
            "operatingMargins": _safe_float(info.get("operatingMargins")),
            "profitMargins": _safe_float(info.get("profitMargins")),
            "freeCashflow": _safe_float(info.get("freeCashflow")),
            "operatingCashflow": _safe_float(info.get("operatingCashflow")),
            "marketCap": _safe_float(info.get("marketCap")),
            "dividendYield": _safe_float(info.get("dividendYield")),
        }

        # 10や12が入ってくるケース（%）対策
        roe = data.get("returnOnEquity")
        if roe is not None and roe > 1.0:
            data["returnOnEquity"] = roe / 100.0

        # 倍率で返るケースを%寄りに補正
        dte = data.get("debtToEquity")
        if dte is not None and 0 < dte < 10:
            data["debtToEquity"] = dte * 100.0

        return {k: v for k, v in data.items() if v is not None}
    except Exception:
        return {}

def get_fundamentals_cached(ticker: str, max_age_days: int = 14) -> Dict[str, Any]:
    cache = _load_fund_cache()
    rec = cache.get(ticker)
    now = _now_tokyo()

    if isinstance(rec, dict):
        ts = rec.get("ts")
        data = rec.get("data")
        try:
            if ts and data:
                ts_dt = datetime.fromisoformat(ts)
                if now - ts_dt < timedelta(days=max_age_days):
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass

    data = _fetch_fundamentals_yf(ticker)
    cache[ticker] = {"ts": now.isoformat(), "data": data}
    _save_fund_cache(cache)
    return data

def _value_quality_score(f: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Buffett-ish: 割安 + 質 + 安全性 を 0..100 に正規化して合成。
    欠損は中立（50付近）に寄せる。
    """
    TARGET_PE = 18.0
    TARGET_PB = 1.5
    TARGET_ROE = 0.10
    MAX_D2E = 200.0
    MIN_MARGIN = 0.05
    MIN_DIV = 0.01

    pe = f.get("trailingPE") or f.get("forwardPE")
    pb = f.get("priceToBook")
    roe = f.get("returnOnEquity")
    d2e = f.get("debtToEquity")
    opm = f.get("operatingMargins")
    pm = f.get("profitMargins")
    fcf = f.get("freeCashflow")
    divy = f.get("dividendYield")

    if pe is None:
        pe_score = 50.0
    else:
        pe_score = 100.0 * _clamp((TARGET_PE / max(pe, 1e-9)), 0.0, 2.0) / 2.0
        pe_score = _clamp(pe_score, 0.0, 100.0)

    if pb is None:
        pb_score = 50.0
    else:
        pb_score = 100.0 * _clamp((TARGET_PB / max(pb, 1e-9)), 0.0, 2.0) / 2.0
        pb_score = _clamp(pb_score, 0.0, 100.0)

    if roe is None:
        roe_score = 50.0
    else:
        roe_score = 100.0 * _clamp(roe / TARGET_ROE, 0.0, 2.0) / 2.0
        roe_score = _clamp(roe_score, 0.0, 100.0)

    margin = opm if opm is not None else pm
    if margin is None:
        margin_score = 50.0
    else:
        margin_score = 100.0 * _clamp(margin / MIN_MARGIN, 0.0, 2.0) / 2.0
        margin_score = _clamp(margin_score, 0.0, 100.0)

    if d2e is None:
        d2e_score = 50.0
    else:
        d2e_score = 100.0 * _clamp((MAX_D2E - d2e) / MAX_D2E, -1.0, 1.0)
        d2e_score = _clamp(d2e_score, 0.0, 100.0)

    if fcf is None:
        fcf_score = 50.0
    else:
        fcf_score = 80.0 if fcf > 0 else 20.0

    if divy is None:
        div_score = 50.0
    else:
        div_score = 100.0 * _clamp(divy / MIN_DIV, 0.0, 2.0) / 2.0
        div_score = _clamp(div_score, 0.0, 100.0)

    value = 0.5 * pe_score + 0.5 * pb_score
    quality = 0.6 * roe_score + 0.4 * margin_score
    safety = 0.7 * d2e_score + 0.3 * fcf_score
    shareholder = div_score

    total = 0.45 * value + 0.30 * quality + 0.20 * safety + 0.05 * shareholder

    debug = {
        "pe": pe, "pb": pb, "roe": roe, "d2e": d2e, "margin": margin, "fcf": fcf, "divy": divy,
        "pe_score": pe_score, "pb_score": pb_score, "roe_score": roe_score, "margin_score": margin_score,
        "d2e_score": d2e_score, "fcf_score": fcf_score, "div_score": div_score,
        "value_score": value, "quality_score": quality, "safety_score": safety,
    }
    return float(_clamp(total, 0.0, 100.0)), debug

def _timing_score(latest_close: float, sma25: float, sma75: float, sma200: float, rsi: float, sma_diff: float) -> float:
    score = 50.0

    if sma200 > 0:
        score += 20.0 if latest_close > sma200 else -20.0

    score += 10.0 if sma25 > sma75 else -10.0

    ideal = -2.0
    dist = abs(sma_diff - ideal)
    score += max(0.0, 20.0 - dist * 3.0)

    if rsi <= 30:
        score += 15.0
    elif 35 <= rsi <= 55:
        score += 10.0
    elif rsi >= 70:
        score -= 20.0

    return float(_clamp(score, 0.0, 100.0))

def _liquidity_filter(df: pd.DataFrame) -> Tuple[bool, float]:
    if df is None or df.empty or "Volume" not in df.columns:
        return True, 0.0

    vol20 = _safe_float(df["Volume"].tail(20).mean())
    if vol20 is None:
        return True, 0.0

    ok = vol20 >= 100000
    liq_score = 100.0 * _clamp(math.log10(vol20 + 1) / 6.0, 0.0, 1.0)
    return ok, float(_clamp(liq_score, 0.0, 100.0))

# -------------- Auto Scan --------------

def auto_scan_value_stocks(api_key: str, progress_callback=None):
    """
    main.py互換:
      return (target_sectors, candidates_top3)

    candidates dict:
      ticker, name, price, rsi, score
    + optional:
      hit, value_score, timing_score, liquidity_score, pb, pe, roe, d2e, fcf
    """
    df_master = get_jpx_master()
    if df_master.empty:
        return ["エラー"], []

    # 市場フィルタ（東証1部相当をPrime/Standardに寄せる）
    df_universe = df_master.copy()
    if "market" in df_universe.columns and df_universe["market"].astype(str).str.len().mean() > 0:
        m = df_universe["market"].astype(str)
        tmp = df_universe[m.str.contains("プライム|Prime|スタンダード|Standard", regex=True, na=False)].copy()
        # フィルタ結果が空なら、ここは無理に絞らない（環境/JPX列仕様差で0件になるのを防ぐ）
        if not tmp.empty:
            df_universe = tmp

    all_sectors = df_universe["sector"].dropna().unique().tolist()
    target_sectors = get_promising_sectors(api_key, all_sectors, n=6)

    target_df = df_universe[df_universe["sector"].isin(target_sectors)].copy()
    if target_df.empty:
        target_df = df_universe.copy()

    scan_list = target_df.to_dict("records")
    tickers = [x["ticker"] for x in scan_list]

    # 価格データはまとめて取りに行く（失敗時は個別にフォールバック）
    price_frames: Dict[str, pd.DataFrame] = {}
    CHUNK = 40  # 大きすぎると失敗しやすい。環境により調整
    for chunk_idx, chunk in enumerate(_chunked(tickers, CHUNK)):
        if progress_callback:
            progress_callback(min((chunk_idx + 1) * CHUNK, len(tickers)), len(tickers), f"price bulk {chunk_idx+1}")

        bulk = _bulk_yahoo_chart(chunk, rng="1y", interval="1d")
        if bulk:
            price_frames.update(bulk)
        else:
            for t in chunk:
                df = _yahoo_chart(t, rng="1y", interval="1d")
                if df is not None and not df.empty:
                    price_frames[t] = df
                time.sleep(0.05)

        time.sleep(0.1)

    # pre-rank: 価格 + 流動性 + タイミング
    pre_rank = []
    for i, item in enumerate(scan_list):
        ticker, comp_name = item["ticker"], item["name"]
        try:
            if progress_callback:
                progress_callback(i + 1, len(scan_list), f"{ticker} {comp_name}")

            df = price_frames.get(ticker)
            if df is None or df.empty or len(df) < 210:
                continue

            ind = calculate_indicators(df)
            if ind is None or ind.empty:
                continue

            latest = ind.iloc[-1]
            latest_close = float(latest["Close"])
            rsi = float(latest["RSI"])
            sma25 = float(latest["SMA_25"])
            sma75 = float(latest["SMA_75"])
            sma200 = float(latest.get("SMA_200", float("nan")))
            sma_diff = float(latest["SMA_DIFF"])

            liq_ok, liq_score = _liquidity_filter(ind)
            if not liq_ok:
                continue

            t_score = _timing_score(latest_close, sma25, sma75, sma200, rsi, sma_diff)

            pre_rank.append({
                "ticker": ticker,
                "name": comp_name,
                "price": latest_close,
                "rsi": rsi,
                "timing_score": t_score,
                "liquidity_score": liq_score,
                "sma_diff": sma_diff,
                "sma25": sma25,
                "sma75": sma75,
                "sma200": sma200,
            })
        except Exception:
            continue

    if not pre_rank:
        return target_sectors, []

    pre_rank = sorted(pre_rank, key=lambda x: (x["timing_score"] + x["liquidity_score"]), reverse=True)
    TOP_FOR_FUND = min(200, len(pre_rank))
    short_list = pre_rank[:TOP_FOR_FUND]

    candidates = []
    near_miss = []

    for j, base in enumerate(short_list):
        ticker = base["ticker"]
        try:
            f = get_fundamentals_cached(ticker, max_age_days=14)
            vq_score, _ = _value_quality_score(f)

            timing_score = float(base["timing_score"])
            liq_score = float(base["liquidity_score"])

            total_score = 0.65 * vq_score + 0.25 * timing_score + 0.10 * liq_score

            pe = f.get("trailingPE") or f.get("forwardPE")
            pb = f.get("priceToBook")
            roe = f.get("returnOnEquity")
            d2e = f.get("debtToEquity")
            fcf = f.get("freeCashflow")

            strict_value_ok = (
                (pb is not None and pb <= 1.5) and
                (pe is not None and pe <= 18.0) and
                (roe is not None and roe >= 0.08) and
                (d2e is None or d2e <= 250.0) and
                (fcf is None or fcf > 0)
            )

            sma200 = base["sma200"]
            strict_timing_ok = (
                (35.0 <= base["rsi"] <= 60.0) and
                (-8.0 <= base["sma_diff"] <= 3.0) and
                (base["price"] > (sma200 if not math.isnan(sma200) else 0.0))
            )

            hit = bool(strict_value_ok and strict_timing_ok)

            out = {
                "ticker": ticker,
                "name": base["name"],
                "price": float(base["price"]),
                "rsi": float(base["rsi"]),
                "score": float(total_score),
                "hit": hit,
                "value_score": float(vq_score),
                "timing_score": float(timing_score),
                "liquidity_score": float(liq_score),
                "pb": _safe_float(pb),
                "pe": _safe_float(pe),
                "roe": _safe_float(roe),
                "d2e": _safe_float(d2e),
                "fcf": _safe_float(fcf),
            }

            if hit:
                candidates.append(out)
            else:
                near_miss.append(out)

        except Exception:
            continue

        if (j + 1) % 40 == 0:
            time.sleep(0.05)

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    if candidates:
        return target_sectors, candidates[:3]

    near_miss = sorted(near_miss, key=lambda x: x["score"], reverse=True)
    return target_sectors, near_miss[:3]

# -------------- AI report generation --------------

def get_ai_analysis(api_key: str, ctx: dict) -> str:
    return call_openai(
        api_key,
        "あなたは実戦派ファンドマネージャーです。与えられた数値から、短期/中期の注意点と売買判断の論点を簡潔に述べてください。",
        f"銘柄:{ctx['pair_label']} 株価:{ctx['price']}円 RSI:{ctx['rsi']:.1f} ATR:{ctx['atr']:.2f} SMA5:{ctx['sma5']:.1f} SMA25:{ctx['sma25']:.1f}",
        temperature=0.2,
    )

def get_ai_order_strategy(api_key: str, ctx: dict) -> str:
    return call_openai(
        api_key,
        "あなたは冷徹な執行責任者です。リスク管理込みで、Entry/Limit/Stop/利確案を具体的な数値で提示してください。",
        f"銘柄:{ctx['pair_label']} 株価:{ctx['price']}円 RSI:{ctx['rsi']:.1f} ATR:{ctx['atr']:.2f}",
        temperature=0.2,
    )

def get_ai_portfolio(api_key: str, ctx: dict) -> str:
    return call_openai(
        api_key,
        "あなたはポートフォリオマネージャーです。週末跨ぎ/決算/材料のリスクも踏まえ、保有継続の是非を述べてください。",
        f"銘柄:{ctx['pair_label']} 株価:{ctx['price']}円 RSI:{ctx['rsi']:.1f} ATR:{ctx['atr']:.2f}",
        temperature=0.2,
    )
