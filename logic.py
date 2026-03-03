import os
import json
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

JST = timezone(timedelta(hours=9))

RATE_LIMIT_SLEEP_SEC = float(os.environ.get('RATE_LIMIT_SLEEP_SEC','0.4'))

LAST_DIAG_PATH = "last_diag.json"

# --- DB URL ---
def _get_db_url() -> Optional[str]:
    return os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")

def check_db_config() -> Tuple[bool, str]:
    url = _get_db_url()
    if not url or "postgres" not in url:
        return False, "Neonの接続URLが見つかりません（NEON_DATABASE_URL）。"
    return True, "ok"

def driver_diagnostics() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": None, "psycopg": None, "psycopg2": None}
    try:
        import sys
        info["python"] = sys.version
    except Exception:
        pass
    try:
        import psycopg  # type: ignore
        info["psycopg"] = getattr(psycopg, "__version__", "unknown")
    except Exception as e:
        info["psycopg"] = {"error": repr(e)}
    try:
        import psycopg2  # type: ignore
        info["psycopg2"] = getattr(psycopg2, "__version__", "unknown")
    except Exception as e:
        info["psycopg2"] = {"error": repr(e)}
    return info

def _connect():
    url = _get_db_url()
    if not url:
        raise RuntimeError("NEON_DATABASE_URL not set")

    psycopg_err = None
    psycopg2_err = None

    try:
        import psycopg  # type: ignore
        return psycopg.connect(url)
    except Exception as e:
        psycopg_err = repr(e)

    try:
        import psycopg2  # type: ignore
        return psycopg2.connect(url)
    except Exception as e:
        psycopg2_err = repr(e)

    raise RuntimeError(
        "Postgresドライバをimport/接続できませんでした。\n"
        f"psycopg error: {psycopg_err}\n"
        f"psycopg2 error: {psycopg2_err}"
    )

def ensure_schema() -> None:
    """DBスキーマを安全に作成/拡張（後方互換）"""
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ohlc_daily (
            symbol TEXT NOT NULL,
            trade_date DATE NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            adj_close DOUBLE PRECISION,
            volume BIGINT,
            PRIMARY KEY(symbol, trade_date)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS kv_meta (
            k TEXT PRIMARY KEY,
            v TEXT
        );
        """)
        # 銘柄マスタ（JPX公式一覧から自動取得する想定）
        cur.execute("""
        CREATE TABLE IF NOT EXISTS universe_symbols (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            market TEXT,
            sector33_code TEXT,
            sector33_name TEXT,
            sector17_code TEXT,
            sector17_name TEXT,
            scale_code TEXT,
            scale_name TEXT,
            updated_utc TIMESTAMP DEFAULT NOW()
        );
        """)
        # 既存DBの後方互換（古いテーブルに列を追加）
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS name TEXT;")
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS market TEXT;")
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS sector33_code TEXT;")
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS sector33_name TEXT;")
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS sector17_code TEXT;")
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS sector17_name TEXT;")
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS scale_code TEXT;")
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS scale_name TEXT;")
        cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS updated_utc TIMESTAMP DEFAULT NOW();")
        # 財務/イベント（簡易）キャッシュ
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_cache (
            symbol TEXT PRIMARY KEY,
            asof_date DATE,
            payload_json TEXT,
            updated_utc TIMESTAMP DEFAULT NOW()
        );
        """)
    conn.commit()
    conn.close()


def _set_meta(conn, k: str, v: str):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO kv_meta(k,v) VALUES(%s,%s)
        ON CONFLICT (k) DO UPDATE SET v=EXCLUDED.v;
        """, (k, v))

def _get_meta(conn, k: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT v FROM kv_meta WHERE k=%s;", (k,))
        row = cur.fetchone()
    return row[0] if row else None

def universe_count() -> int:
    ensure_schema()
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM universe_symbols;")
        n = int(cur.fetchone()[0] or 0)
    conn.close()
    return n

def universe_upsert(symbols: List[str]) -> int:
    """後方互換: symbols(文字列)のみを登録。"""
    ensure_schema()
    rows = []
    for s in symbols:
        s = str(s).strip()
        if not s:
            continue
        s2 = s.replace(" ", "").upper()
        if re.fullmatch(r"\d{4}", s2):
            s2 = s2 + ".T"
        rows.append({"symbol": s2})
    if not rows:
        return 0
    return universe_upsert_rows(rows)

def universe_upsert_rows(rows: List[Dict[str, Any]]) -> int:
    """新: symbol + メタ（市場/33業種/銘柄名など）を upsert"""
    ensure_schema()
    # normalize + unique by symbol (preserve order)
    seen = set()
    norm_rows: List[Dict[str, Any]] = []
    for r in rows:
        sym = str(r.get("symbol","")).strip().replace(" ", "").upper()
        if not sym:
            continue
        if re.fullmatch(r"\d{4}", sym):
            sym = sym + ".T"
        if sym in seen:
            continue
        seen.add(sym)
        rr = dict(r)
        rr["symbol"] = sym
        norm_rows.append(rr)

    if not norm_rows:
        return 0

    conn = _connect()
    with conn.cursor() as cur:
        for r in norm_rows:
            cur.execute(
                """
                INSERT INTO universe_symbols(symbol, name, market, sector33_code, sector33_name, sector17_code, sector17_name, scale_code, scale_name, updated_utc)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
                ON CONFLICT(symbol) DO UPDATE SET
                    name=COALESCE(EXCLUDED.name, universe_symbols.name),
                    market=COALESCE(EXCLUDED.market, universe_symbols.market),
                    sector33_code=COALESCE(EXCLUDED.sector33_code, universe_symbols.sector33_code),
                    sector33_name=COALESCE(EXCLUDED.sector33_name, universe_symbols.sector33_name),
                    sector17_code=COALESCE(EXCLUDED.sector17_code, universe_symbols.sector17_code),
                    sector17_name=COALESCE(EXCLUDED.sector17_name, universe_symbols.sector17_name),
                    scale_code=COALESCE(EXCLUDED.scale_code, universe_symbols.scale_code),
                    scale_name=COALESCE(EXCLUDED.scale_name, universe_symbols.scale_name),
                    updated_utc=NOW();
                """,
                (
                    r.get("symbol"),
                    r.get("name"),
                    r.get("market"),
                    r.get("sector33_code"),
                    r.get("sector33_name"),
                    r.get("sector17_code"),
                    r.get("sector17_name"),
                    r.get("scale_code"),
                    r.get("scale_name"),
                ),
            )
    conn.commit()
    conn.close()
    return len(norm_rows)

def universe_load_symbols(limit: Optional[int] = None) -> List[str]:
    ensure_schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            if limit is None:
                cur.execute("SELECT symbol FROM universe_symbols ORDER BY symbol;")
            else:
                cur.execute("SELECT symbol FROM universe_symbols ORDER BY symbol LIMIT %s;", (int(limit),))
            rows = cur.fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows if r and r[0]]

    ensure_schema()
    conn = _connect()
    with conn.cursor() as cur:
        if limit:
            cur.execute("SELECT symbol FROM universe_symbols ORDER BY symbol LIMIT %s;", (int(limit),))
        else:
            cur.execute("SELECT symbol FROM universe_symbols ORDER BY symbol;")
        rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def universe_register_from_inputs(uploaded_csv_file, pasted_text: str):
    symbols: List[str] = []
    if uploaded_csv_file is not None:
        try:
            df = pd.read_csv(uploaded_csv_file)
            cols = [c.lower() for c in df.columns]
            pick = None
            for c in ["ticker", "symbol", "code"]:
                if c in cols:
                    pick = df.columns[cols.index(c)]
                    break
            if pick is None:
                return 0, "CSVに ticker / symbol / code 列が見つかりません。"
            symbols += [str(x).strip() for x in df[pick].dropna().tolist()]
        except Exception as e:
            return 0, f"CSV読込に失敗: {e}"

    if pasted_text and pasted_text.strip():
        for line in pasted_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in re.split(r"[\s,]+", line) if p.strip()]
            symbols += parts

    if not symbols:
        return 0, "入力が空です。CSVをアップロードするか、銘柄を貼り付けてください。"

    try:
        n = universe_upsert(symbols)
        return n, "ok"
    except Exception as e:
        return 0, f"DB登録に失敗: {e}"


def universe_load_meta() -> pd.DataFrame:
    """universe_symbols からメタ（市場/33業種など）を読み込む。無ければ空DF。"""
    ensure_schema()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, name, market, sector33_code, sector33_name, sector17_code, sector17_name, scale_code, scale_name FROM universe_symbols;")
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=["symbol","name","market","sector33_code","sector33_name","sector17_code","sector17_name","scale_code","scale_name"])
    return pd.DataFrame(rows, columns=["symbol","name","market","sector33_code","sector33_name","sector17_code","sector17_name","scale_code","scale_name"])

def universe_autofetch_from_jpx() -> Tuple[int, str]:
    """JPX公式サイトの『東証上場銘柄一覧（Excel）』を取得し、銘柄マスタ（universe_symbols）に登録する。

    - 4桁コードは自動で .T を付与
    - 可能なら 33業種/市場区分/規模 なども保存（Stage0のセクター判定に使用）
    """
    ensure_schema()

    url_candidates = [
        "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls",
    ]

    last_err = None
    content = None
    for url in url_candidates:
        try:
            import requests  # type: ignore
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            content = r.content
            break
        except Exception as e:
            last_err = e

    if content is None:
        return 0, f"JPX一覧のダウンロードに失敗しました: {last_err}"

    try:
        import io
        bio = io.BytesIO(content)
        df = pd.read_excel(bio)
    except Exception as e:
        return 0, f"Excel読込に失敗しました（環境依存）: {e}"

    # 列名の推定（JPXの data_j.xls 前提）
    col_code = None
    col_name = None
    col_market = None
    col_s33_code = None
    col_s33_name = None
    col_s17_code = None
    col_s17_name = None
    col_scale_code = None
    col_scale_name = None

    for c in df.columns:
        cs = str(c).strip()
        if cs in ["コード", "銘柄コード"]:
            col_code = c
        elif cs in ["銘柄名", "会社名", "名称"]:
            col_name = c
        elif cs in ["市場・商品区分", "市場区分", "市場"]:
            col_market = c
        elif cs in ["33業種コード", "33業種"]:
            col_s33_code = c
        elif cs in ["33業種区分", "33業種区分（日本語）"]:
            col_s33_name = c
        elif cs in ["17業種コード", "17業種"]:
            col_s17_code = c
        elif cs in ["17業種区分"]:
            col_s17_name = c
        elif cs in ["規模コード", "規模"]:
            col_scale_code = c
        elif cs in ["規模区分"]:
            col_scale_name = c

    # fallback: 4桁比率でコード列推定
    if col_code is None:
        for c in df.columns:
            ser = df[c].dropna().astype(str).str.replace(r"\D", "", regex=True)
            if len(ser) == 0:
                continue
            ratio = ser.str.fullmatch(r"\d{4}").mean()
            if ratio > 0.6:
                col_code = c
                break

    if col_code is None:
        return 0, "銘柄コード列を特定できませんでした。手動CSVでの登録をお使いください。"

    def _norm_code(x: Any) -> Optional[str]:
        s = "" if x is None else str(x)
        s = re.sub(r"\D", "", s)
        if re.fullmatch(r"\d{4}", s):
            return s + ".T"
        return None

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = _norm_code(r.get(col_code))
        if not sym:
            continue
        row = {"symbol": sym}
        if col_name is not None:
            v = r.get(col_name)
            if pd.notna(v):
                row["name"] = str(v).strip()
        if col_market is not None:
            v = r.get(col_market)
            if pd.notna(v):
                row["market"] = str(v).strip()
        if col_s33_code is not None:
            v = r.get(col_s33_code)
            if pd.notna(v):
                row["sector33_code"] = str(v).strip()
        if col_s33_name is not None:
            v = r.get(col_s33_name)
            if pd.notna(v):
                row["sector33_name"] = str(v).strip()
        if col_s17_code is not None:
            v = r.get(col_s17_code)
            if pd.notna(v):
                row["sector17_code"] = str(v).strip()
        if col_s17_name is not None:
            v = r.get(col_s17_name)
            if pd.notna(v):
                row["sector17_name"] = str(v).strip()
        if col_scale_code is not None:
            v = r.get(col_scale_code)
            if pd.notna(v):
                row["scale_code"] = str(v).strip()
        if col_scale_name is not None:
            v = r.get(col_scale_name)
            if pd.notna(v):
                row["scale_name"] = str(v).strip()
        rows.append(row)

    if not rows:
        return 0, "銘柄コードが抽出できませんでした。"

    n = universe_upsert_rows(rows)
    return n, "ok"


def get_db_status() -> Dict[str, Any]:
    try:
        ensure_schema()
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM universe_symbols;")
            uni = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(DISTINCT symbol) FROM ohlc_daily;")
            symbols_count = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM ohlc_daily;")
            rows_count = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT MAX(trade_date) FROM ohlc_daily;")
            latest_trade_date = cur.fetchone()[0]
            last_update_utc = _get_meta(conn, "last_update_utc")
        conn.close()

        hours_since = None
        if last_update_utc:
            try:
                dt = datetime.fromisoformat(last_update_utc.replace("Z","+00:00"))
                hours_since = (datetime.now(timezone.utc) - dt).total_seconds()/3600.0
            except Exception:
                hours_since = None

        return {
            "ok": True,
            "universe_count": uni,
            "symbols_count": symbols_count,
            "rows_count": rows_count,
            "latest_trade_date": str(latest_trade_date) if latest_trade_date else None,
            "last_update_utc": last_update_utc,
            "hours_since_update": hours_since,
        }
    except Exception as e:
        return {"ok": False, "message": str(e)}

# --- DB update ---
def _fetch_from_yf(symbols: List[str], start: str) -> Dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=" ".join(symbols),
        start=start,
        group_by="ticker",
        threads=True,
        auto_adjust=False,
        progress=False,
    )
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in data.columns.get_level_values(0):
                df = data[sym].copy()
                df.dropna(how="all", inplace=True)
                out[sym] = df
    else:
        if len(symbols) == 1:
            df = data.copy()
            df.dropna(how="all", inplace=True)
            out[symbols[0]] = df
    return out

def update_db_incremental(days_back: int = 14, keep_days: int = 400, chunk_size: int = 200) -> Dict[str, Any]:
    ensure_schema()
    universe = universe_load_symbols()
    if not universe:
        return {"upserted_rows": 0, "failed_symbols": 0, "message": "銘柄マスタが0件です。先に『銘柄マスタ登録』を行ってください。"}

    start_dt = (datetime.now(timezone.utc) - timedelta(days=days_back + 5)).date()
    start = str(start_dt)

    upserted = 0
    failed = 0
    sample_fail: List[str] = []

    conn = _connect()
    try:
        for i in range(0, len(universe), chunk_size):
            chunk = universe[i:i+chunk_size]
            try:
                fetched = _fetch_from_yf(chunk, start=start)
            except Exception:
                fetched = {}
            if not fetched:
                failed += len(chunk)
                if len(sample_fail) < 50:
                    sample_fail.extend(chunk[: min(10, len(chunk))])
                # Cloud対策：過剰リクエスト抑制
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                continue

            rows = []
            for sym, df in fetched.items():
                if df is None or df.empty:
                    failed += 1
                    if len(sample_fail) < 50:
                        sample_fail.append(sym)
                    continue
                df = df.reset_index()
                for _, r in df.iterrows():
                    d = r.get("Date")
                    if pd.isna(d):
                        continue
                    rows.append((
                        sym,
                        pd.to_datetime(d).date(),
                        float(r["Open"]) if pd.notna(r.get("Open")) else None,
                        float(r["High"]) if pd.notna(r.get("High")) else None,
                        float(r["Low"]) if pd.notna(r.get("Low")) else None,
                        float(r["Close"]) if pd.notna(r.get("Close")) else None,
                        float(r.get("Adj Close")) if pd.notna(r.get("Adj Close")) else None,
                        int(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
                    ))
            if rows:
                with conn.cursor() as cur:
                    cur.executemany("""
                    INSERT INTO ohlc_daily(symbol, trade_date, open, high, low, close, adj_close, volume)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(symbol, trade_date) DO UPDATE SET
                        open=EXCLUDED.open,
                        high=EXCLUDED.high,
                        low=EXCLUDED.low,
                        close=EXCLUDED.close,
                        adj_close=EXCLUDED.adj_close,
                        volume=EXCLUDED.volume;
                    """, rows)
                upserted += len(rows)
                conn.commit()

            # Cloud対策：過剰リクエスト抑制
            time.sleep(RATE_LIMIT_SLEEP_SEC)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).date()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ohlc_daily WHERE trade_date < %s;", (cutoff,))
        _set_meta(conn, "last_update_utc", datetime.now(timezone.utc).isoformat().replace("+00:00","Z"))
        conn.commit()
    finally:
        conn.close()

    return {"upserted_rows": upserted, "failed_symbols": failed, "sample_failures": sample_fail[:20], "start": start, "keep_days": keep_days, "chunk_size": chunk_size}

# --- Indicators & scan (unchanged core) ---
def _atr14(df: pd.DataFrame) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def evaluate_swing_fixed(df: pd.DataFrame, max_hold_days: int = 10, tp_atr: float = 1.5, sl_atr: float = 1.0, min_bars: int = 60, atr_pct: float | None = None) -> Dict[str, Any]:
    d = df.copy()
    d["ATR14"] = _atr14(d)
    d = d.dropna()
    # 履歴が短い場合でも「暫定評価」で落とさない（Cloudで最初は履歴が薄くなりがち）
    if len(d) < int(min_bars):
        tf = _trend_features(df)
        ap = float(atr_pct) if atr_pct is not None and np.isfinite(atr_pct) else np.nan
        # 暫定スコア：トレンド(20/60日)を加点、ボラ(ATR%)を軽く減点
        score = 0.55*(tf.get("ret_60", np.nan) if np.isfinite(tf.get("ret_60", np.nan)) else 0.0) + 0.35*(tf.get("ret_20", np.nan) if np.isfinite(tf.get("ret_20", np.nan)) else 0.0)
        if np.isfinite(ap):
            score -= 0.02*(ap)
        return {"ok": True, "trials": 0, "tp_hit_rate": float("nan"), "ev_r": float("nan"), "avg_tp_days": float(max_hold_days), "avg_dd_r": float("nan"), "swing_score": float(score), "note": "履歴不足のため暫定評価"}

    wins = 0
    losses = 0
    hit_days = []
    dd_r = []

    H = int(max_hold_days)
    for i in range(0, len(d)-H-1):
        entry = float(d["Close"].iloc[i])
        atr = float(d["ATR14"].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
        tp = entry + tp_atr*atr
        sl = entry - sl_atr*atr
        fut = d.iloc[i+1:i+1+H]
        mae = (float(fut["Low"].min()) - entry) / atr
        dd_r.append(mae)

        outcome = None
        for j in range(len(fut)):
            if float(fut["High"].iloc[j]) >= tp:
                outcome = ("win", j+1); break
            if float(fut["Low"].iloc[j]) <= sl:
                outcome = ("loss", None); break
        if outcome is None:
            outcome = ("loss", None)

        if outcome[0] == "win":
            wins += 1
            hit_days.append(outcome[1])
        else:
            losses += 1

    n = wins + losses
    if n == 0:
        return {"ok": False}
    p_win = wins / n
    ev_r = p_win*tp_atr - (1-p_win)*sl_atr
    avg_tp_days = float(np.mean(hit_days)) if hit_days else float(H)
    avg_dd_r = float(abs(np.mean(dd_r))) if dd_r else 0.0
    score = 0.45*ev_r + 0.25*p_win + 0.15*(1.0/max(avg_tp_days,1.0)) - 0.15*avg_dd_r

    return {"ok": True, "trials": int(n), "tp_hit_rate": float(p_win), "ev_r": float(ev_r), "avg_tp_days": float(avg_tp_days), "avg_dd_r": float(avg_dd_r), "swing_score": float(score)}

def _trend_features(df: pd.DataFrame) -> Dict[str, float]:
    c = df["Close"].astype(float)
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    ret_20 = (c.iloc[-1]/c.iloc[-21]-1.0) if len(c) > 21 else np.nan
    ret_60 = (c.iloc[-1]/c.iloc[-61]-1.0) if len(c) > 61 else np.nan
    slope20 = (ma20.iloc[-1]-ma20.iloc[-6])/(abs(ma20.iloc[-6])+1e-12) if len(ma20) > 6 else np.nan
    return {"ret_20": float(ret_20) if np.isfinite(ret_20) else np.nan, "ret_60": float(ret_60) if np.isfinite(ret_60) else np.nan, "slope20": float(slope20) if np.isfinite(slope20) else np.nan}

def _pick_strategy(features: Dict[str, float], atr_pct: float) -> str:
    r20 = features.get("ret_20", np.nan)
    s20 = features.get("slope20", np.nan)
    if np.isfinite(r20) and r20 > 0.08 and np.isfinite(s20) and s20 > 0:
        return "順張り（ブレイク/上昇トレンド）"
    if np.isfinite(r20) and r20 < -0.06:
        return "逆張り（リバウンド狙い）"
    if np.isfinite(atr_pct) and atr_pct < 1.5:
        return "レンジ（小動き）"
    return "押し目買い（回復狙い）"

def _read_latest_dates(conn) -> Tuple[Optional[str], Optional[str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(trade_date) FROM ohlc_daily;")
        latest = cur.fetchone()[0]
        if not latest:
            return None, None
        cur.execute("SELECT MAX(trade_date) FROM ohlc_daily WHERE trade_date < %s;", (latest,))
        prev = cur.fetchone()[0]
    return str(latest), str(prev) if prev else None

def fetch_last_n_days(symbols: List[str], n_days: int = 80) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    ensure_schema()
    conn = _connect()
    try:
        latest, _ = _read_latest_dates(conn)
        if not latest:
            return pd.DataFrame()
        cutoff = (datetime.fromisoformat(latest).date() - timedelta(days=int(n_days*2))).isoformat()
        with conn.cursor() as cur:
            cur.execute("""
            SELECT symbol, trade_date, open, high, low, close, volume
            FROM ohlc_daily
            WHERE symbol = ANY(%s) AND trade_date >= %s
            ORDER BY symbol, trade_date;
            """, (symbols, cutoff))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["Symbol","Date","Open","High","Low","Close","Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def stage0_select(min_price: float, min_avg_volume: float, keep: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Stage0: DBにある最新2営業日のスナップショット+20日平均出来高だけで全件を軽量スクリーニング。
    33業種（JPX公式一覧）をDBに持てている場合は、セクター強度ランキングも出す。
    """
    ensure_schema()
    universe = universe_load_symbols()
    t0 = time.time()
    conn = _connect()
    try:
        latest, prev = _read_latest_dates(conn)
        if not latest or not prev:
            return pd.DataFrame(), pd.DataFrame(), {"ok": False, "reason": "db_has_no_latest_prev"}

        with conn.cursor() as cur:
            cur.execute(
                """
                WITH last2 AS (
                    SELECT symbol, trade_date, close, volume
                    FROM ohlc_daily
                    WHERE trade_date IN (%s, %s)
                ),
                piv AS (
                    SELECT symbol,
                           MAX(CASE WHEN trade_date=%s THEN close END) AS close_latest,
                           MAX(CASE WHEN trade_date=%s THEN close END) AS close_prev
                    FROM last2
                    GROUP BY symbol
                )
                SELECT symbol, close_latest, close_prev
                FROM piv;
                """,
                (latest, prev, latest, prev),
            )
            snap = cur.fetchall()

            cutoff20 = (datetime.fromisoformat(latest).date() - timedelta(days=40)).isoformat()
            cur.execute(
                """
                SELECT symbol, AVG(volume)::float AS avg_vol20
                FROM ohlc_daily
                WHERE symbol = ANY(%s) AND trade_date >= %s
                GROUP BY symbol;
                """,
                (universe, cutoff20),
            )
            vol20 = cur.fetchall()
    finally:
        conn.close()

    snap_df = pd.DataFrame(snap, columns=["symbol", "close_latest", "close_prev"])
    vol20_df = pd.DataFrame(vol20, columns=["symbol", "avg_vol20"])
    df = snap_df.merge(vol20_df, on="symbol", how="left")
    df["pct_change_1d"] = (df["close_latest"] / (df["close_prev"] + 1e-12) - 1.0) * 100.0

    df = df[pd.notna(df["close_latest"])]
    df = df[df["close_latest"] >= float(min_price)]
    df = df[pd.notna(df["avg_vol20"])]
    df = df[df["avg_vol20"] >= float(min_avg_volume)]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"ok": True, "latest_trade_date": latest, "elapsed_sec": time.time() - t0, "stage0_candidates": 0}

    # Stage0 基本スコア（OHLCを追加取得しない軽量スコア）
    df["stage0_score"] = 0.7 * df["pct_change_1d"].fillna(0.0) + 0.3 * np.log10(df["avg_vol20"].fillna(1.0))

    # 33業種メタ（あれば）
    meta_df = universe_load_meta()
    if not meta_df.empty and "sector33_name" in meta_df.columns:
        df = df.merge(meta_df[["symbol", "sector33_name"]], on="symbol", how="left")
    df["sector33_name"] = df.get("sector33_name").fillna("不明")

    sec = (
        df.groupby("sector33_name", dropna=False)["stage0_score"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    sec.columns = ["セクター（33業種）", "強度（中央値）"]
    sec.insert(0, "順位", range(1, len(sec) + 1))

    # セクター比率を維持しつつ上位抽出
    keep = int(keep)
    df2 = df.sort_values(["sector33_name", "stage0_score"], ascending=[True, False]).copy()
    counts = df2["sector33_name"].value_counts()
    total = int(counts.sum()) if len(counts) else 0
    min_per = 3
    alloc: Dict[str, int] = {}
    if total > 0:
        for sn, cnt in counts.items():
            alloc[sn] = max(min_per, int(round(keep * (cnt / total))))

        def _sum_alloc() -> int:
            return int(sum(alloc.values()))

        while _sum_alloc() > keep:
            k_max = max(alloc, key=lambda k: alloc[k])
            if alloc[k_max] > min_per:
                alloc[k_max] -= 1
            else:
                break
        while _sum_alloc() < keep:
            k_max = max(alloc, key=lambda k: counts.get(k, 0))
            alloc[k_max] += 1

    picks = []
    for sn, k in alloc.items():
        picks.append(df2[df2["sector33_name"] == sn].head(int(k)))
    picked = pd.concat(picks, ignore_index=True) if picks else df2.head(keep)

    picked = picked.sort_values("stage0_score", ascending=False)
    picked_syms = set(picked["symbol"].tolist())
    if len(picked) < keep:
        rest = df2[~df2["symbol"].isin(picked_syms)].sort_values("stage0_score", ascending=False)
        picked = pd.concat([picked, rest.head(keep - len(picked))], ignore_index=True)

    picked = picked.sort_values("stage0_score", ascending=False).head(keep).copy()

    out = picked.rename(
        columns={
            "symbol": "銘柄",
            "close_latest": "現在値（終値）",
            "pct_change_1d": "前日比（%）",
            "avg_vol20": "平均出来高20日",
        }
    )[["銘柄", "現在値（終値）", "前日比（%）", "平均出来高20日", "stage0_score", "sector33_name"]]
    out["stage0_score"] = out["stage0_score"].astype(float).round(4)

    meta = {"ok": True, "latest_trade_date": latest, "elapsed_sec": time.time() - t0, "stage0_candidates": int(len(out))}
    return out, sec, meta

def stage1_select(stage0_df: pd.DataFrame, keep: int, atr_pct_min: float, atr_pct_max: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()
    syms = stage0_df["銘柄"].astype(str).tolist()
    hist = fetch_last_n_days(syms, n_days=80)
    if hist.empty:
        return pd.DataFrame(), {"ok": False}
    rows = []
    for sym, g in hist.groupby("Symbol"):
        g = g.sort_values("Date")
        if len(g) < 60:
            continue
        atr = _atr14(g)
        atr_last = float(atr.iloc[-1]) if len(atr) else np.nan
        close_last = float(g["Close"].iloc[-1])
        atr_pct = (atr_last/(close_last+1e-12))*100.0 if np.isfinite(atr_last) else np.nan
        if np.isfinite(atr_pct) and (atr_pct < float(atr_pct_min) or atr_pct > float(atr_pct_max)):
            continue
        feats = _trend_features(g)
        score = 0.40*(feats.get("ret_20",0.0) if np.isfinite(feats.get("ret_20",np.nan)) else 0.0) + 0.20*(feats.get("ret_60",0.0) if np.isfinite(feats.get("ret_60",np.nan)) else 0.0) + 0.20*(feats.get("slope20",0.0) if np.isfinite(feats.get("slope20",np.nan)) else 0.0) + 0.20*(atr_pct/10.0 if np.isfinite(atr_pct) else 0.0)
        rows.append({"銘柄": sym, "ATR%": atr_pct, "20日騰落": feats.get("ret_20",np.nan), "60日騰落": feats.get("ret_60",np.nan), "MA20傾き": feats.get("slope20",np.nan), "stage1_score": score, "現在値（終値）": close_last})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), {"ok": False}
    df = df.sort_values("stage1_score", ascending=False).head(int(keep)).copy()
    df["ATR%"] = df["ATR%"].round(3)
    df["20日騰落"] = (df["20日騰落"]*100.0).round(2)
    df["60日騰落"] = (df["60日騰落"]*100.0).round(2)
    df["MA20傾き"] = (df["MA20傾き"]*100.0).round(3)
    df["stage1_score"] = df["stage1_score"].round(4)
    df["現在値（終値）"] = df["現在値（終値）"].round(2)
    return df, {"ok": True, "elapsed_sec": time.time()-t0, "stage1_candidates": int(len(df))}

def _fundamentals_get_cached(symbol: str) -> Optional[Dict[str, Any]]:
    """yfinanceの info/calendar を使った簡易財務/イベント。RateLimit時はNone。"""
    ensure_schema()
    symbol = str(symbol).strip().upper()
    if not symbol:
        return None

    today = datetime.now(JST).date()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT asof_date, payload_json FROM fundamentals_cache WHERE symbol=%s;", (symbol,))
            row = cur.fetchone()
        if row and row[0] == today and row[1]:
            try:
                return json.loads(row[1])
            except Exception:
                pass
    finally:
        conn.close()

    # fetch
    try:
        import yfinance as yf  # type: ignore
        tk = yf.Ticker(symbol)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        cal = {}
        try:
            cal = getattr(tk, "calendar", None)
            if hasattr(cal, "to_dict"):
                cal = cal.to_dict()
            if isinstance(cal, pd.DataFrame):
                cal = cal.to_dict()
        except Exception:
            cal = {}

        payload = {
            "symbol": symbol,
            "asof": str(today),
            "info": info,
            "calendar": cal,
        }
    except Exception:
        return None

    # cache
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fundamentals_cache(symbol, asof_date, payload_json, updated_utc)
                VALUES(%s,%s,%s,NOW())
                ON CONFLICT(symbol) DO UPDATE SET asof_date=EXCLUDED.asof_date, payload_json=EXCLUDED.payload_json, updated_utc=NOW();
                """,
                (symbol, today, json.dumps(payload, ensure_ascii=False)),
            )
        conn.commit()
    finally:
        conn.close()
    return payload

def _buffett_score(payload: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """超簡易の“バフェット風”評価（データが無ければNone）。"""
    info = payload.get("info") or {}
    try:
        roe = info.get("returnOnEquity")
        pm = info.get("profitMargins")
        de = info.get("debtToEquity")
        fcf = info.get("freeCashflow")
        mc = info.get("marketCap")
        score = 0.0
        used = 0
        if isinstance(roe, (int,float)) and roe == roe:
            score += max(-1.0, min(1.0, float(roe))) * 1.5
            used += 1
        if isinstance(pm, (int,float)) and pm == pm:
            score += max(-1.0, min(1.0, float(pm))) * 1.0
            used += 1
        if isinstance(de, (int,float)) and de == de:
            # 低いほど良い
            score += max(-1.0, min(1.0, (1.0 - float(de)/200.0))) * 1.0
            used += 1
        if isinstance(fcf, (int,float)) and isinstance(mc, (int,float)) and mc and mc == mc and fcf == fcf:
            # FCF利回りっぽい
            yld = float(fcf)/float(mc)
            score += max(-1.0, min(1.0, yld*10.0)) * 1.0
            used += 1
        if used == 0:
            return None, "財務データ不足"
        return round(score, 3), f"指標使用={used}"
    except Exception:
        return None, "計算失敗"

def _event_alert(payload: Dict[str, Any], days_earn: int = 10, days_exdiv: int = 7) -> str:
    """決算/権利落ちなど、取得できる範囲での注意喚起。"""
    info = payload.get("info") or {}
    today = datetime.now(JST).date()
    notes = []
    # earningsDate
    ed = info.get("earningsDate") or info.get("earningsTimestamp")
    try:
        if isinstance(ed, (list, tuple)) and ed:
            ed = ed[0]
        if isinstance(ed, (int,float)):
            d = datetime.fromtimestamp(ed, tz=timezone.utc).astimezone(JST).date()
            if 0 <= (d - today).days <= days_earn:
                notes.append(f"決算接近({d})")
        elif isinstance(ed, str):
            d = pd.to_datetime(ed, errors="coerce")
            if pd.notna(d):
                d = d.tz_localize("UTC") if getattr(d, "tzinfo", None) is None else d
                d = d.tz_convert(JST).date()
                if 0 <= (d - today).days <= days_earn:
                    notes.append(f"決算接近({d})")
    except Exception:
        pass
    # exDividendDate
    exd = info.get("exDividendDate")
    try:
        if isinstance(exd, (int,float)):
            d = datetime.fromtimestamp(exd, tz=timezone.utc).astimezone(JST).date()
            if 0 <= (d - today).days <= days_exdiv:
                notes.append(f"権利落ち接近({d})")
    except Exception:
        pass
    return " / ".join(notes) if notes else "-"

def stage2_rank(stage1_df: pd.DataFrame, keep: int, stage2_days: int = 180, min_bars: int = 60, include_fundamentals: bool = True, fundamentals_top_n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()
    syms = stage1_df["銘柄"].astype(str).tolist()
    hist = fetch_last_n_days(syms, n_days=int(stage2_days))
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame(), {"ok": False}

    rows, guides = [], []
    errors, sample_fail = 0, []
    diag = {"hist_symbols": int(len(syms)), "hist_days": int(stage2_days), "ok_eval": 0, "fallback_eval": 0, "skipped_no_atr": 0}

    for sym, g in hist.groupby("Symbol"):
        try:
            g = g.sort_values("Date")
            atr_pct = float(stage1_df.loc[stage1_df["銘柄"]==sym, "ATR%"].iloc[0]) if len(stage1_df.loc[stage1_df["銘柄"]==sym]) else np.nan
            res = evaluate_swing_fixed(g, min_bars=int(min_bars), atr_pct=atr_pct)
            if not res.get("ok"):
                continue
            if res.get("trials", 0) == 0:
                diag["fallback_eval"] += 1
            else:
                diag["ok_eval"] += 1
            close_last = float(g["Close"].iloc[-1])
            atr_series = _atr14(g)
            atr_last = float(atr_series.dropna().iloc[-1]) if len(atr_series.dropna()) else float("nan")
            tp = close_last + 1.5*atr_last
            sl = close_last - 1.0*atr_last
            atr_pct = float(stage1_df.loc[stage1_df["銘柄"]==sym, "ATR%"].iloc[0]) if len(stage1_df.loc[stage1_df["銘柄"]==sym]) else np.nan
            strat = _pick_strategy(_trend_features(g), atr_pct)

            rows.append({"銘柄": sym, "推奨方式": strat, "現在値（終値）": round(close_last,2), "TP目安": round(tp,2), "SL目安": round(sl,2),
                         "TP到達率": res["tp_hit_rate"], "期待値EV(R)": res["ev_r"], "平均利確日数": res["avg_tp_days"],
                         "平均逆行(R)": res["avg_dd_r"], "検証回数": res.get("trials",0), "利確スコア": res.get("swing_score", float("nan")), "備考": res.get("note","")})
            guides.append({"銘柄": sym, "推奨方式": strat, "Entry目安": round(close_last,2), "SL目安": round(sl,2), "TP目安": round(tp,2), "最大保有": "10営業日"})
        except Exception:
            errors += 1
            if len(sample_fail) < 20:
                sample_fail.append(sym)

    df = pd.DataFrame(rows)
    guide = pd.DataFrame(guides)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"ok": False, "reason": "no_pass", "diag": diag, "errors": errors, "sample_failures": sample_fail}

    df["TP到達率"] = df["TP到達率"].clip(0,1)
    df["期待値EV(R)"] = df["期待値EV(R)"].clip(-5,5)
    df["平均逆行(R)"] = df["平均逆行(R)"].clip(0,10)
    df["平均利確日数"] = df["平均利確日数"].clip(1,10)

    df["総合スコア"] = 0.55*df["利確スコア"] + 0.20*df["TP到達率"] + 0.15*df["期待値EV(R)"] - 0.10*df["平均逆行(R)"]
    df = df.sort_values("総合スコア", ascending=False).head(int(keep)).copy()

# --- 財務/イベント（簡易）を上位Nだけ付与（RateLimit時はスキップ） ---
    if include_fundamentals and fundamentals_top_n and len(df):
        topn = int(min(len(df), max(0, fundamentals_top_n)))
        scores = []
        alerts = []
        memos = []
        for i, sym in enumerate(df["銘柄"].tolist()):
            if i >= topn:
                scores.append(None); alerts.append("-"); memos.append("未取得（上位のみ）"); 
                continue
            payload = _fundamentals_get_cached(sym)
            if payload is None:
                scores.append(None); alerts.append("-"); memos.append("取得失敗（RateLimit等）")
                continue
            sc, memo = _buffett_score(payload)
            scores.append(sc)
            alerts.append(_event_alert(payload))
            memos.append(memo)
        df["バフェット簡易スコア"] = scores
        df["イベント注意"] = alerts
        df["財務メモ"] = memos
    else:
        df["バフェット簡易スコア"] = None
        df["イベント注意"] = "-"
        df["財務メモ"] = "OFF"

    for c in ["利確スコア","TP到達率","期待値EV(R)","平均利確日数","平均逆行(R)","総合スコア"]:
        df[c] = df[c].astype(float).round(4)
    df["TP到達率"] = (df["TP到達率"]*100.0).round(2)

    cols = ["銘柄","推奨方式","現在値（終値）","TP目安","SL目安","TP到達率","期待値EV(R)","平均利確日数","平均逆行(R)","検証回数","利確スコア","総合スコア","バフェット簡易スコア","イベント注意","財務メモ"]
    df = df[cols]

    meta = {"ok": True, "elapsed_sec": time.time()-t0, "errors": errors, "sample_failures": sample_fail, "stage2_selected": int(len(df)), "diag": diag}
    return df, guide, meta

def run_scan_3stage(stage0_keep: int = 1200, stage1_keep: int = 300, stage2_keep: int = 60,
                    min_price: float = 300.0, min_avg_volume: float = 30000.0,
                    atr_pct_min: float = 1.0, atr_pct_max: float = 8.0,
                    stage2_days: int = 180, stage2_min_bars: int = 60,
                    include_fundamentals: bool = True, fundamentals_top_n: int = 20) -> Dict[str, Any]:
    ensure_schema()
    diag: Dict[str, Any] = {"ok": True, "stage": "start", "mode": "stable", "errors": [], "sample_failures": []}

    s0, sec, m0 = stage0_select(min_price=min_price, min_avg_volume=min_avg_volume, keep=stage0_keep)
    diag["stage0"] = m0
    if s0.empty:
        diag["stage"] = "done"; diag["mode"] = "degraded"
        diag["errors"].append("Stage0 empty: DB更新不足 or フィルタが厳しすぎます")
        return {"selected": pd.DataFrame(), "guide": pd.DataFrame(), "sector_strength": sec, "diag": diag}

    s1, m1 = stage1_select(s0, keep=stage1_keep, atr_pct_min=atr_pct_min, atr_pct_max=atr_pct_max)
    diag["stage1"] = m1
    if s1.empty:
        diag["stage"] = "done"; diag["mode"] = "degraded"
        diag["errors"].append("Stage1 empty: ATR%条件/履歴不足")
        return {"selected": pd.DataFrame(), "guide": pd.DataFrame(), "sector_strength": sec, "diag": diag}

    sel, guide, m2 = stage2_rank(s1, keep=stage2_keep, stage2_days=int(stage2_days), min_bars=int(stage2_min_bars), include_fundamentals=include_fundamentals, fundamentals_top_n=int(fundamentals_top_n))
    diag["stage2"] = m2
    if sel.empty:
        diag["mode"] = "degraded"
        diag["errors"].append("Stage2 empty: 利確評価失敗")
    diag["stage"] = "done"
    return {"selected": sel, "guide": guide, "sector_strength": sec, "diag": diag}

# --- diag persistence ---
def save_last_diag(diag: Dict[str, Any]) -> None:
    try:
        with open(LAST_DIAG_PATH, "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_last_diag() -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(LAST_DIAG_PATH):
            with open(LAST_DIAG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None
