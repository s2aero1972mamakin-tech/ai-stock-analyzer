import os
import json
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple


def compute_buffett_score(payload: dict) -> Optional[float]:
    """Simple heuristic score 0..1. Missing -> None."""
    if not isinstance(payload, dict) or not payload:
        return None
    def g(*keys):
        for k in keys:
            if k in payload and payload[k] not in (None, "", "None"):
                return payload[k]
        return None
    try:
        scores=[]
        pe = g("trailingPE","pe","PER","per")
        pb = g("priceToBook","pb","PBR","pbr")
        de = g("debtToEquity","debt_equity","DE")
        margin = g("profitMargins","netMargin","margin")
        roe = g("returnOnEquity","roe")
        if pe is not None:
            pe=float(pe)
            if pe>0: scores.append(max(0.0, min(1.0, (30.0-pe)/30.0)))
        if pb is not None:
            pb=float(pb)
            if pb>0: scores.append(max(0.0, min(1.0, (3.0-pb)/3.0)))
        if de is not None:
            de=float(de)
            scores.append(max(0.0, min(1.0, (200.0-de)/200.0)))
        if margin is not None:
            margin=float(margin)
            if margin<=1.0: scores.append(max(0.0, min(1.0, margin/0.15)))
            else: scores.append(max(0.0, min(1.0, (margin/100.0)/0.15)))
        if roe is not None:
            roe=float(roe)
            if roe<=1.0: scores.append(max(0.0, min(1.0, roe/0.15)))
            else: scores.append(max(0.0, min(1.0, (roe/100.0)/0.15)))
        if not scores:
            return None
        return float(sum(scores)/len(scores))
    except Exception:
        return None


def _sym_key(sym: str) -> str:
    """Join key for JP symbols: strip suffix and non-digits, keep last 4 digits, zfill(4)."""
    s = (sym or "").strip()
    if not s:
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return s
    if len(digits) >= 4:
        digits = digits[-4:]
    return digits.zfill(4)


def _norm_symbol(sym: str) -> str:
    """Normalize symbol to JP format: '1301' -> '1301.T'."""
    s = (sym or "").strip()
    if not s:
        return s
    if "." in s:
        return s
    if s.isdigit():
        return s.zfill(4) + ".T"
    return s


def _json_safe(obj: Any):
    """json.dumps default helper (date/datetime/numpy)."""
    try:
        import datetime as _dt
        if isinstance(obj, (_dt.date, _dt.datetime)):
            return obj.isoformat()
    except Exception:
        pass
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    return str(obj)

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
        # --- universe_symbols columns migration (idempotent) ---
        try:
            cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS name TEXT;")
            cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS market TEXT;")
            cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS sector33_code TEXT;")
            cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS sector33_name TEXT;")
            cur.execute("ALTER TABLE universe_symbols ADD COLUMN IF NOT EXISTS lot_size INTEGER DEFAULT 100;")
        except Exception:
            pass

    conn.commit()
    conn.close()


_universe_col_cache: Dict[str, bool] = {}

def _has_universe_col(col: str) -> bool:
    if col in _universe_col_cache:
        return _universe_col_cache[col]
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT 1
                        FROM information_schema.columns
                        WHERE table_name='universe_symbols' AND column_name=%s
                        LIMIT 1;""",
                    (col,),
                )
                ok = cur.fetchone() is not None
        finally:
            conn.close()
    except Exception:
        ok = False
    _universe_col_cache[col] = ok
    return ok



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


def universe_upsert(symbols: List[str], meta: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    """Upsert universe symbols with optional metadata."""
    ensure_schema()
    meta = meta or {}

    norm: List[str] = []
    for s in symbols:
        s = str(s).strip()
        if not s:
            continue
        s2 = s.replace(" ", "").upper()
        if re.fullmatch(r"\d{4}", s2):
            s2 = s2 + ".T"
        norm.append(s2)

    seen = set()
    uniq: List[str] = []
    for s in norm:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    if not uniq:
        return 0

    conn = _connect()
    with conn.cursor() as cur:
        for sym in uniq:
            # meta は '1301' / '1301.T' どちらのキーでも来うるので吸収する
            base = sym[:-2] if sym.endswith(".T") else sym
            m = meta.get(sym) or meta.get(base) or meta.get(base + ".T") or {}
            cur.execute(
                """INSERT INTO universe_symbols(symbol, name, market, sector33_code, sector33_name, lot_size, updated_utc)
                   VALUES(%s,%s,%s,%s,%s,%s,NOW())
                   ON CONFLICT(symbol) DO UPDATE SET
                     name=COALESCE(EXCLUDED.name, universe_symbols.name),
                     market=COALESCE(EXCLUDED.market, universe_symbols.market),
                     sector33_code=COALESCE(EXCLUDED.sector33_code, universe_symbols.sector33_code),
                     sector33_name=COALESCE(EXCLUDED.sector33_name, universe_symbols.sector33_name),
                     lot_size=COALESCE(EXCLUDED.lot_size, universe_symbols.lot_size),
                     updated_utc=NOW();""",
                (sym, m.get("name"), m.get("market"), m.get("sector33_code"), m.get("sector33_name"), int(m.get("lot_size") or 100)),
            )
    conn.commit()
    conn.close()
    return len(uniq)

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

    def _norm_col(x: Any) -> str:
        return str(x).strip().replace("（", "(").replace("）", ")")

    # まずは“完全一致”に近いものを拾う
    for c in df.columns:
        cs = _norm_col(c)
        if cs in ["コード", "銘柄コード", "Code"]:
            col_code = c
        elif cs in ["銘柄名", "会社名", "名称", "銘柄名(日本語)"]:
            col_name = c
        elif cs in ["市場・商品区分", "市場区分", "市場", "市場・商品区分(日本語)"]:
            col_market = c
        elif cs in ["33業種コード", "33業種コード(東証)", "33業種コード（東証）"]:
            col_s33_code = c
        elif cs in ["33業種区分", "33業種名", "33業種区分(日本語)", "33業種区分（日本語）"]:
            col_s33_name = c
        elif cs in ["17業種コード", "17業種コード(東証)", "17業種コード（東証）"]:
            col_s17_code = c
        elif cs in ["17業種区分", "17業種名", "17業種区分(日本語)", "17業種区分（日本語）"]:
            col_s17_name = c
        elif cs in ["規模コード", "規模"]:
            col_scale_code = c
        elif cs in ["規模区分", "規模区分(日本語)", "規模区分（日本語）"]:
            col_scale_name = c

    # 次に“部分一致”で補完（JPXの列名が微妙に変わることがあるため）
    def _find_by_contains(must: List[str], must_not: List[str] = []) -> Optional[Any]:
        for c in df.columns:
            cs = _norm_col(c)
            ok = all(k in cs for k in must) and all(k not in cs for k in must_not)
            if ok:
                return c
        return None

    if col_s33_code is None:
        col_s33_code = _find_by_contains(["33", "業種", "コード"])
    if col_s33_name is None:
        col_s33_name = _find_by_contains(["33", "業種"], must_not=["コード"])
    if col_market is None:
        col_market = _find_by_contains(["市場"])
    if col_name is None:
        col_name = _find_by_contains(["銘柄", "名"]) or _find_by_contains(["名称"])
    if col_scale_code is None:
        col_scale_code = _find_by_contains(["規模", "コード"])
    if col_scale_name is None:
        col_scale_name = _find_by_contains(["規模"], must_not=["コード"])
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
    update_sector33_from_jpx()
    return n, "ok"





def update_sector33_from_jpx() -> Tuple[int, str]:
    """
    Download JPX listed issues master (Excel) and update universe_symbols.sector33_*.
    Returns (updated_rows, status_str).
    """
    ensure_schema()

    # JPX official file (frequently updated). We keep robust fallbacks.
    urls = [
        "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls",
        "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls?download=1",
        "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html",
    ]

    # try direct excel first
    df = None
    last_err = ""
    for u in urls[:2]:
        try:
            df = pd.read_excel(u)
            if df is not None and len(df) > 1000:
                break
        except Exception as e:
            last_err = str(e)
            df = None

    # if failed, try to scrape the page for xls link
    if df is None:
        try:
            import pandas as _pd
            import requests
            import re as _re
            html = requests.get(urls[2], timeout=20).text
            m = _re.search(r'href="([^"]+data_j\.xls[^"]*)"', html)
            if m:
                link = m.group(1)
                if link.startswith("/"):
                    link = "https://www.jpx.co.jp" + link
                df = _pd.read_excel(link)
        except Exception as e:
            last_err = str(e)
            df = None

    if df is None or df.empty:
        return 0, f"jpx_master_download_failed: {last_err}"

    # normalize columns（JPXの列名は微妙に変わることがあるので“部分一致”も使う）
    def _norm_col(x: Any) -> str:
        return str(x).strip().replace("（", "(").replace("）", ")")

    cols = {_norm_col(c): c for c in df.columns}

    # required: code
    code_col = None
    for k in ["コード", "銘柄コード", "Code"]:
        if k in cols:
            code_col = cols[k]
            break
    if code_col is None:
        # fallback: 4桁比率で推定
        for c in df.columns:
            ser = df[c].dropna().astype(str).str.replace(r"\D", "", regex=True)
            if len(ser) == 0:
                continue
            if (ser.str.fullmatch(r"\d{4}").mean() or 0.0) > 0.6:
                code_col = c
                break
    if code_col is None:
        code_col = df.columns[0]

    def _find_by_contains(must: List[str], must_not: List[str] = []) -> Optional[Any]:
        for c in df.columns:
            cs = _norm_col(c)
            ok = all(k in cs for k in must) and all(k not in cs for k in must_not)
            if ok:
                return c
        return None

    sec_code_col = None
    sec_name_col = None
    market_col = None
    name_col = None

    # exact-ish
    for c in df.columns:
        s = _norm_col(c)
        if s in ["33業種コード", "33業種コード(東証)", "33業種コード（東証）"]:
            sec_code_col = c
        elif s in ["33業種区分", "33業種名", "33業種区分(日本語)", "33業種区分（日本語）"]:
            sec_name_col = c
        elif s in ["市場・商品区分", "市場区分", "市場"]:
            market_col = c
        elif s in ["銘柄名", "名称", "会社名"]:
            name_col = c

    # contains fallback
    if sec_code_col is None:
        sec_code_col = _find_by_contains(["33", "業種", "コード"])
    if sec_name_col is None:
        sec_name_col = _find_by_contains(["33", "業種"], must_not=["コード"])
    if market_col is None:
        market_col = _find_by_contains(["市場"])
    if name_col is None:
        name_col = _find_by_contains(["銘柄", "名"]) or _find_by_contains(["名称"])
# if sector columns not found, we can't update
    if sec_code_col is None and sec_name_col is None:
        return 0, "jpx_master_has_no_sector33_columns"

    tmp_cols = [code_col]
    for c in [name_col, market_col, sec_code_col, sec_name_col]:
        if c is not None and c not in tmp_cols:
            tmp_cols.append(c)
    tmp = df[tmp_cols].copy()
    tmp[code_col] = tmp[code_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(4)
    tmp = tmp[tmp[code_col].str.fullmatch(r"\d{4}")].copy()
    tmp["symbol"] = tmp[code_col] + ".T"

    # build update rows
    rows = []
    for _, r in tmp.iterrows():
        sym = r["symbol"]
        sec_code = str(r[sec_code_col]).strip() if sec_code_col is not None and pd.notna(r.get(sec_code_col)) else None
        sec_name = str(r[sec_name_col]).strip() if sec_name_col is not None and pd.notna(r.get(sec_name_col)) else None
        name = str(r[name_col]).strip() if name_col is not None and pd.notna(r.get(name_col)) else None
        market = str(r[market_col]).strip() if market_col is not None and pd.notna(r.get(market_col)) else None
        rows.append((sym, name, market, sec_code, sec_name))

    conn = _connect()
    updated = 0
    try:
        with conn.cursor() as cur:
            # update in chunks; only overwrite when null/empty
            chunk = 2000
            for i in range(0, len(rows), chunk):
                part = rows[i:i+chunk]
                cur.executemany(
                    """UPDATE universe_symbols
                       SET
                         name = COALESCE(universe_symbols.name, %s),
                         market = COALESCE(universe_symbols.market, %s),
                         sector33_code = CASE
                           WHEN universe_symbols.sector33_code IS NULL OR universe_symbols.sector33_code='' THEN %s
                           ELSE universe_symbols.sector33_code END,
                         sector33_name = CASE
                           WHEN universe_symbols.sector33_name IS NULL OR universe_symbols.sector33_name='' THEN %s
                           ELSE universe_symbols.sector33_name END,
                         updated_utc = NOW()
                       WHERE symbol = %s;""",
                    [(name, market, sec_code, sec_name, sym) for (sym, name, market, sec_code, sec_name) in part],
                )
                updated += cur.rowcount
        conn.commit()
    finally:
        conn.close()

    return updated, "ok"

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

def _safe_ret(series: pd.Series, lookback: int) -> float:
    try:
        s = pd.to_numeric(series, errors="coerce")
        if len(s) <= int(lookback):
            return np.nan
        prev = float(s.iloc[-(int(lookback) + 1)])
        cur = float(s.iloc[-1])
        if (not np.isfinite(prev)) or abs(prev) < 1e-12 or (not np.isfinite(cur)):
            return np.nan
        return float(cur / prev - 1.0)
    except Exception:
        return np.nan


def _safe_slope(series: pd.Series, lookback: int) -> float:
    try:
        s = pd.to_numeric(series, errors="coerce")
        if len(s) <= int(lookback):
            return np.nan
        prev = float(s.iloc[-(int(lookback) + 1)])
        cur = float(s.iloc[-1])
        if (not np.isfinite(prev)) or abs(prev) < 1e-12 or (not np.isfinite(cur)):
            return np.nan
        return float((cur - prev) / (abs(prev) + 1e-12))
    except Exception:
        return np.nan


def _clip01(x: float, default: float = 0.0) -> float:
    try:
        if not np.isfinite(x):
            return float(default)
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return float(default)


def _short_reversal_label(features: Dict[str, float]) -> str:
    score = pd.to_numeric(pd.Series([features.get("reversal_score", np.nan)]), errors="coerce").iloc[0]
    confirmed = bool(features.get("reversal_confirmed", False))
    if confirmed:
        return "確認"
    if np.isfinite(score) and float(score) >= 0.46:
        return "準確認"
    return "未確認"


def _build_trend_audit(features: Dict[str, float]) -> str:
    if not isinstance(features, dict) or not features:
        return ""
    parts = []
    if bool(features.get("strong_uptrend", False)):
        parts.append("上昇基調")
    elif bool(features.get("uptrend_bias", False)):
        parts.append("中期上向き")
    elif bool(features.get("strong_downtrend", False)):
        parts.append("下降継続警戒")
    elif bool(features.get("trend_headwind", False)):
        parts.append("下向き圧力あり")
    else:
        parts.append("方向感中立")

    parts.append(f"短期反転:{_short_reversal_label(features)}")

    pullback = pd.to_numeric(pd.Series([features.get("pullback_from_high20", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(pullback):
        parts.append(f"20日高値比{float(pullback)*100:.1f}%")

    pos20 = pd.to_numeric(pd.Series([features.get("pos_vs_ma20", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(pos20):
        parts.append(f"20日線比{float(pos20)*100:.1f}%")

    if bool(features.get("breakout_3", False)):
        parts.append("3日高値更新")
    return " / ".join(parts)


def _build_strategy_audit(features: Dict[str, float], strategy_name: str = "") -> str:
    if not isinstance(features, dict) or not features:
        return strategy_name or ""
    tags = []
    if bool(features.get("strong_uptrend", False)):
        tags.append("5/20/60日線が上向き")
    elif bool(features.get("uptrend_bias", False)):
        tags.append("20日線が60日線を上回る")
    elif bool(features.get("trend_headwind", False)):
        tags.append("20日線が弱く下降圧力あり")

    if bool(features.get("pullback_candidate", False)):
        tags.append("上昇基調内の押し目")
    if bool(features.get("reversal_confirmed", False)):
        tags.append("短期反転を確認")
    elif _short_reversal_label(features) == "準確認":
        tags.append("短期反転は準確認")
    else:
        tags.append("短期反転は未確認")

    pullback = pd.to_numeric(pd.Series([features.get("pullback_from_high20", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(pullback):
        tags.append(f"20日高値比{float(pullback)*100:.1f}%")
    return " / ".join(tags[:5])


def _trend_features(df: pd.DataFrame) -> Dict[str, float]:
    d = df.copy()
    if "Date" in d.columns:
        d = d.sort_values("Date")
    c = pd.to_numeric(d["Close"], errors="coerce")
    h = pd.to_numeric(d["High"], errors="coerce") if "High" in d.columns else c.copy()
    l = pd.to_numeric(d["Low"], errors="coerce") if "Low" in d.columns else c.copy()
    if "Volume" in d.columns:
        v = pd.to_numeric(d["Volume"], errors="coerce")
    else:
        v = pd.Series(np.nan, index=d.index, dtype=float)

    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()

    ret_3 = _safe_ret(c, 3)
    ret_5 = _safe_ret(c, 5)
    ret_10 = _safe_ret(c, 10)
    ret_20 = _safe_ret(c, 20)
    ret_60 = _safe_ret(c, 60)
    slope5 = _safe_slope(ma5, 3)
    slope20 = _safe_slope(ma20, 5)
    slope60 = _safe_slope(ma60, 10)

    close_last = float(c.iloc[-1]) if len(c) else np.nan
    ma5_last = float(ma5.iloc[-1]) if len(ma5) else np.nan
    ma20_last = float(ma20.iloc[-1]) if len(ma20) else np.nan
    ma60_last = float(ma60.iloc[-1]) if len(ma60) else np.nan

    hh3_prev = float(h.shift(1).rolling(3).max().iloc[-1]) if len(h) >= 4 else np.nan
    hh10_prev = float(h.shift(1).rolling(10).max().iloc[-1]) if len(h) >= 11 else np.nan
    hh20 = float(h.rolling(20).max().iloc[-1]) if len(h) >= 20 else np.nan
    ll10 = float(l.rolling(10).min().iloc[-1]) if len(l) >= 10 else np.nan
    ll20 = float(l.rolling(20).min().iloc[-1]) if len(l) >= 20 else np.nan

    vol_ratio_5 = np.nan
    try:
        vol5 = float(v.rolling(5).mean().iloc[-1]) if len(v) >= 5 else np.nan
        vol_last = float(v.iloc[-1]) if len(v) else np.nan
        if np.isfinite(vol5) and vol5 > 0 and np.isfinite(vol_last):
            vol_ratio_5 = float(vol_last / vol5)
    except Exception:
        vol_ratio_5 = np.nan

    pos_vs_ma5 = (close_last / ma5_last - 1.0) if np.isfinite(close_last) and np.isfinite(ma5_last) and abs(ma5_last) > 1e-12 else np.nan
    pos_vs_ma20 = (close_last / ma20_last - 1.0) if np.isfinite(close_last) and np.isfinite(ma20_last) and abs(ma20_last) > 1e-12 else np.nan
    pos_vs_ma60 = (close_last / ma60_last - 1.0) if np.isfinite(close_last) and np.isfinite(ma60_last) and abs(ma60_last) > 1e-12 else np.nan
    pullback_from_high20 = (close_last / hh20 - 1.0) if np.isfinite(close_last) and np.isfinite(hh20) and abs(hh20) > 1e-12 else np.nan
    rebound_from_low10 = (close_last / ll10 - 1.0) if np.isfinite(close_last) and np.isfinite(ll10) and abs(ll10) > 1e-12 else np.nan
    rebound_from_low20 = (close_last / ll20 - 1.0) if np.isfinite(close_last) and np.isfinite(ll20) and abs(ll20) > 1e-12 else np.nan

    breakout_3 = bool(np.isfinite(close_last) and np.isfinite(hh3_prev) and close_last > hh3_prev * 1.001)
    breakout_10 = bool(np.isfinite(close_last) and np.isfinite(hh10_prev) and close_last > hh10_prev * 1.002)
    above_ma5 = bool(np.isfinite(pos_vs_ma5) and pos_vs_ma5 >= 0.0)
    above_ma20 = bool(np.isfinite(pos_vs_ma20) and pos_vs_ma20 >= 0.0)
    above_ma60 = bool(np.isfinite(pos_vs_ma60) and pos_vs_ma60 >= 0.0)

    ma_stack_up = bool(np.isfinite(ma5_last) and np.isfinite(ma20_last) and np.isfinite(ma60_last) and ma5_last >= ma20_last >= ma60_last)
    ma_stack_down = bool(np.isfinite(ma5_last) and np.isfinite(ma20_last) and np.isfinite(ma60_last) and ma5_last <= ma20_last <= ma60_last)

    uptrend_bias = bool(
        np.isfinite(ma20_last) and np.isfinite(ma60_last)
        and ma20_last >= ma60_last
        and (not np.isfinite(slope20) or slope20 > -0.004)
        and (not np.isfinite(ret_60) or ret_60 > 0.02)
    )
    strong_uptrend = bool(
        uptrend_bias
        and ma_stack_up
        and np.isfinite(ret_20) and ret_20 > 0.05
        and np.isfinite(slope20) and slope20 > 0.0
    )
    trend_headwind = bool(
        np.isfinite(ma20_last) and np.isfinite(ma60_last)
        and ma20_last < ma60_last
        and np.isfinite(slope20) and slope20 < 0.0
        and np.isfinite(ret_20) and ret_20 < 0.0
    )
    strong_downtrend = bool(
        trend_headwind
        and ma_stack_down
        and np.isfinite(ret_60) and ret_60 < -0.04
    )
    pullback_candidate = bool(
        uptrend_bias
        and np.isfinite(pullback_from_high20) and (-0.18 <= pullback_from_high20 <= -0.01)
        and (not np.isfinite(pos_vs_ma60) or pos_vs_ma60 > -0.035)
    )

    reversal_raw = 0.0
    reversal_raw += 1.00 if above_ma5 else 0.0
    reversal_raw += 0.85 if (np.isfinite(ret_3) and ret_3 > 0.0) else 0.0
    reversal_raw += 0.75 if (np.isfinite(slope5) and slope5 > 0.0) else 0.0
    reversal_raw += 0.85 if breakout_3 else 0.0
    reversal_raw += 0.45 if (np.isfinite(vol_ratio_5) and vol_ratio_5 >= 1.05) else 0.0
    reversal_raw += 0.35 if (np.isfinite(pos_vs_ma20) and pos_vs_ma20 >= -0.015) else 0.0
    reversal_score = _clip01(reversal_raw / 4.25, default=0.0)

    reversal_confirmed = bool(
        reversal_score >= 0.62
        and (above_ma5 or breakout_3)
        and (not np.isfinite(slope5) or slope5 > -0.002)
    )

    return {
        "ret_3": float(ret_3) if np.isfinite(ret_3) else np.nan,
        "ret_5": float(ret_5) if np.isfinite(ret_5) else np.nan,
        "ret_10": float(ret_10) if np.isfinite(ret_10) else np.nan,
        "ret_20": float(ret_20) if np.isfinite(ret_20) else np.nan,
        "ret_60": float(ret_60) if np.isfinite(ret_60) else np.nan,
        "slope5": float(slope5) if np.isfinite(slope5) else np.nan,
        "slope20": float(slope20) if np.isfinite(slope20) else np.nan,
        "slope60": float(slope60) if np.isfinite(slope60) else np.nan,
        "ma5": float(ma5_last) if np.isfinite(ma5_last) else np.nan,
        "ma20": float(ma20_last) if np.isfinite(ma20_last) else np.nan,
        "ma60": float(ma60_last) if np.isfinite(ma60_last) else np.nan,
        "pos_vs_ma5": float(pos_vs_ma5) if np.isfinite(pos_vs_ma5) else np.nan,
        "pos_vs_ma20": float(pos_vs_ma20) if np.isfinite(pos_vs_ma20) else np.nan,
        "pos_vs_ma60": float(pos_vs_ma60) if np.isfinite(pos_vs_ma60) else np.nan,
        "pullback_from_high20": float(pullback_from_high20) if np.isfinite(pullback_from_high20) else np.nan,
        "rebound_from_low10": float(rebound_from_low10) if np.isfinite(rebound_from_low10) else np.nan,
        "rebound_from_low20": float(rebound_from_low20) if np.isfinite(rebound_from_low20) else np.nan,
        "vol_ratio_5": float(vol_ratio_5) if np.isfinite(vol_ratio_5) else np.nan,
        "breakout_3": breakout_3,
        "breakout_10": breakout_10,
        "above_ma5": above_ma5,
        "above_ma20": above_ma20,
        "above_ma60": above_ma60,
        "ma_stack_up": ma_stack_up,
        "ma_stack_down": ma_stack_down,
        "uptrend_bias": uptrend_bias,
        "strong_uptrend": strong_uptrend,
        "trend_headwind": trend_headwind,
        "strong_downtrend": strong_downtrend,
        "pullback_candidate": pullback_candidate,
        "reversal_score": float(reversal_score),
        "reversal_confirmed": reversal_confirmed,
    }


def _pick_strategy(features: Dict[str, float], atr_pct: float) -> str:
    r20 = pd.to_numeric(pd.Series([features.get("ret_20", np.nan)]), errors="coerce").iloc[0]
    r60 = pd.to_numeric(pd.Series([features.get("ret_60", np.nan)]), errors="coerce").iloc[0]
    reversal_score = pd.to_numeric(pd.Series([features.get("reversal_score", np.nan)]), errors="coerce").iloc[0]
    pullback = pd.to_numeric(pd.Series([features.get("pullback_from_high20", np.nan)]), errors="coerce").iloc[0]

    strong_uptrend = bool(features.get("strong_uptrend", False))
    uptrend_bias = bool(features.get("uptrend_bias", False))
    pullback_candidate = bool(features.get("pullback_candidate", False))
    reversal_confirmed = bool(features.get("reversal_confirmed", False))
    breakout_10 = bool(features.get("breakout_10", False))
    trend_headwind = bool(features.get("trend_headwind", False))
    strong_downtrend = bool(features.get("strong_downtrend", False))

    if strong_uptrend and breakout_10 and (not np.isfinite(r20) or r20 > 0.03):
        return "順張り（ブレイク/上昇トレンド）"

    if pullback_candidate and (reversal_confirmed or (np.isfinite(reversal_score) and reversal_score >= 0.46)):
        return "押し目買い（回復狙い）"

    if (np.isfinite(r20) and r20 < -0.06) and (reversal_confirmed or (np.isfinite(reversal_score) and reversal_score >= 0.58)) and not strong_downtrend:
        return "逆張り（リバウンド狙い）"

    if np.isfinite(atr_pct) and atr_pct < 1.5 and not strong_uptrend and not trend_headwind:
        return "レンジ（小動き）"

    if uptrend_bias and np.isfinite(pullback) and pullback <= -0.01:
        return "押し目買い（回復狙い）"

    if reversal_confirmed and not strong_downtrend and (not np.isfinite(r60) or r60 > -0.05):
        return "逆張り（リバウンド狙い）"

    if strong_uptrend:
        return "順張り（ブレイク/上昇トレンド）"

    return "レンジ（小動き）"

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
    conn_u = _connect()
    try:
        with conn_u.cursor() as cur_u:
            # lot_size列が無いDBでも落ちないように吸収
            try:
                if _has_universe_col("lot_size"):
                    cur_u.execute(
                        "SELECT symbol, name, sector33_name, sector33_code, COALESCE(lot_size,100) "
                        "FROM universe_symbols WHERE symbol = ANY(%s);",
                        (universe,),
                    )
                else:
                    cur_u.execute(
                        "SELECT symbol, name, sector33_name, sector33_code, 100 "
                        "FROM universe_symbols WHERE symbol = ANY(%s);",
                        (universe,),
                    )
            except Exception:
                cur_u.execute(
                    "SELECT symbol, name, sector33_name, sector33_code, 100 "
                    "FROM universe_symbols WHERE symbol = ANY(%s);",
                    (universe,),
                )
            urows = cur_u.fetchall()
    finally:
        conn_u.close()
    u_df = pd.DataFrame(columns=['symbol','name','sector33_name','sector33_code','lot_size'])
    if urows:
        # urows の列数はDBスキーマ/SELECTにより変わる可能性があるため動的に吸収
        first_len = len(urows[0])
        if first_len >= 5:
            u_df = pd.DataFrame([tuple(r[:5]) for r in urows], columns=['symbol','name','sector33_name','sector33_code','lot_size'])
        elif first_len == 4:
            u_df = pd.DataFrame(urows, columns=['symbol','sector33_name','sector33_code','lot_size'])
            u_df['name'] = ''
        elif first_len == 3:
            u_df = pd.DataFrame(urows, columns=['symbol','sector33_name','sector33_code'])
            u_df['name'] = ''
            u_df['lot_size'] = 100
        elif first_len == 2:
            u_df = pd.DataFrame(urows, columns=['symbol','name'])
            u_df['sector33_name'] = '不明'
            u_df['sector33_code'] = ''
            u_df['lot_size'] = 100
        elif first_len == 1:
            u_df = pd.DataFrame(urows, columns=['symbol'])
            u_df['name'] = ''
            u_df['sector33_name'] = '不明'
            u_df['sector33_code'] = ''
            u_df['lot_size'] = 100
    u_df['symbol'] = u_df['symbol'].astype(str).map(_norm_symbol)
    u_df['_key'] = u_df['symbol'].astype(str).map(_sym_key)
    ukeys = set([_sym_key(x) for x in universe])
    u_df = u_df[u_df['_key'].isin(ukeys)].copy()
    u_df['_sym_norm'] = u_df['symbol'].astype(str).map(_norm_symbol)
    df["_key"] = df["symbol"].astype(str).map(_sym_key)
    df = df.merge(u_df.drop(columns=["symbol","_sym_norm"], errors="ignore"), left_on="_key", right_on="_key", how="left")
    df["sector33_name"] = df.get("sector33_name").replace("", None).fillna("不明")
    df["name"] = df.get("name").fillna("")
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
        df = df.merge(
            meta_df[["symbol","sector33_name"]],
            on="symbol",
            how="left",
            suffixes=("","_meta"),
        )
        if "sector33_name_meta" in df.columns:
            df["sector33_name"] = df["sector33_name"].fillna(df["sector33_name_meta"])
            df.drop(columns=["sector33_name_meta"], inplace=True)
    if "sector33_name" not in df.columns:
        df["sector33_name"] = "不明"
    else:
        df["sector33_name"] = df["sector33_name"].fillna("不明")

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
    )[["銘柄", "現在値（終値）", "前日比（%）", "平均出来高20日", "stage0_score", "name", "sector33_name"]]
    out = out.rename(columns={"name":"銘柄名","sector33_name":"セクター"})
    out["stage0_score"] = out["stage0_score"].astype(float).round(4)

    meta = {"ok": True, "latest_trade_date": latest, "elapsed_sec": time.time() - t0, "stage0_candidates": int(len(out))}
    return out, sec, meta

def stage1_select(stage0_df: pd.DataFrame, keep: int, atr_pct_min: float, atr_pct_max: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()
    # stage0 からセクター/銘柄名を引き継ぐ（無ければ空）
    sec_map = {}
    name_map = {}
    try:
        if "セクター" in stage0_df.columns:
            sec_map = stage0_df.set_index("銘柄")["セクター"].to_dict()
        elif "sector33_name" in stage0_df.columns:
            sec_map = stage0_df.set_index("銘柄")["sector33_name"].fillna("不明").to_dict()
        if "銘柄名" in stage0_df.columns:
            name_map = stage0_df.set_index("銘柄")["銘柄名"].fillna("").to_dict()
        elif "name" in stage0_df.columns:
            name_map = stage0_df.set_index("銘柄")["name"].fillna("").to_dict()
    except Exception:
        pass

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
        rows.append({"銘柄": sym, "企業名": name_map.get(sym, ""), "セクター": sec_map.get(sym, "不明"), "ATR%": atr_pct, "20日騰落": feats.get("ret_20",np.nan), "60日騰落": feats.get("ret_60",np.nan), "MA20傾き": feats.get("slope20",np.nan), "stage1_score": score, "現在値（終値）": close_last})
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
                (symbol, today, json.dumps(payload, ensure_ascii=False, default=_json_safe)),
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


# =========================
# COMPLETE AI HELPERS
# =========================
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def detect_trend_ai_01(g: pd.DataFrame) -> float:
    """AIトレンド検出: 0..1"""
    try:
        c = g["Close"].astype(float)
        if len(c) < 70:
            return 0.5
        ema20 = _ema(c, 20)
        ema60 = _ema(c, 60)
        dir_up = 1.0 if float(ema20.iloc[-1]) >= float(ema60.iloc[-1]) else -1.0
        r20 = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1.0) if len(c) > 21 else 0.0
        slope = (float(ema20.iloc[-1]) - float(ema20.iloc[-6])) / (abs(float(ema20.iloc[-6])) + 1e-12)
        x = 0.0
        x += 0.55 * (1.0 / (1.0 + np.exp(-10.0 * r20)) - 0.5) * 2.0
        x += 0.25 * np.tanh(8.0 * slope)
        x += 0.20 * (1.0 if dir_up > 0 else -1.0)
        return float(max(0.0, min(1.0, (x + 1.0) / 2.0)))
    except Exception:
        return 0.5

def walk_forward_oos(g: pd.DataFrame, hold_days: int = 10, tp_atr: float = 1.5, sl_atr: float = 1.0,
                     train_days: int = 120, test_days: int = 40, step_days: int = 40):
    """WalkForward OOS: (winrate, rr). 失敗時は (nan,nan)"""
    try:
        d = g.copy().dropna(subset=["Close","High","Low"])
        d["ATR14"] = _atr14(d)
        d = d.dropna()
        if len(d) < (train_days + test_days + 40):
            return float("nan"), float("nan")
        wins = 0
        losses = 0
        start = 0
        while start + train_days + test_days < len(d):
            train = d.iloc[start:start+train_days]
            test = d.iloc[start+train_days:start+train_days+test_days]
            ema20 = _ema(train["Close"].astype(float), 20)
            ema60 = _ema(train["Close"].astype(float), 60)
            direction_long = bool(float(ema20.iloc[-1]) >= float(ema60.iloc[-1]))
            H = int(hold_days)
            for i in range(0, len(test)-H-1):
                entry = float(test["Close"].iloc[i])
                atr = float(test["ATR14"].iloc[i])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                fut = test.iloc[i+1:i+1+H]
                if direction_long:
                    tp = entry + tp_atr*atr
                    sl = entry - sl_atr*atr
                    hit = None
                    for j in range(len(fut)):
                        if float(fut["High"].iloc[j]) >= tp:
                            hit = "win"; break
                        if float(fut["Low"].iloc[j]) <= sl:
                            hit = "loss"; break
                else:
                    tp = entry - tp_atr*atr
                    sl = entry + sl_atr*atr
                    hit = None
                    for j in range(len(fut)):
                        if float(fut["Low"].iloc[j]) <= tp:
                            hit = "win"; break
                        if float(fut["High"].iloc[j]) >= sl:
                            hit = "loss"; break
                if hit == "win":
                    wins += 1
                else:
                    losses += 1
            start += step_days
        n = wins + losses
        if n <= 0:
            return float("nan"), float("nan")
        return float(wins/n), float(tp_atr/max(sl_atr,1e-9))
    except Exception:
        return float("nan"), float("nan")

def montecarlo_dd5(p_win: float, rr: float, trades: int = 60, paths: int = 400, seed: int = 7) -> float:
    """MonteCarlo DD（R単位）5%点。負の値。"""
    try:
        if not np.isfinite(p_win) or not np.isfinite(rr) or trades < 10:
            return float("nan")
        p = float(max(0.01, min(0.99, p_win)))
        b = float(max(0.2, rr))
        rng = np.random.default_rng(seed)
        dds = []
        for _ in range(int(paths)):
            wins = rng.random(int(trades)) < p
            r = np.where(wins, b, -1.0).astype(float)
            eq = np.cumsum(r)
            peak = np.maximum.accumulate(eq)
            dd = eq - peak
            dds.append(float(np.min(dd)))
        return float(np.quantile(np.array(dds, dtype=float), 0.05))
    except Exception:
        return float("nan")

def kelly_fraction(p_win: float, rr: float) -> float:
    """Kelly最適化（保守クリップ0..0.25）"""
    try:
        if not np.isfinite(p_win) or not np.isfinite(rr) or rr <= 0:
            return 0.0
        p = float(p_win); q = 1.0 - p; b = float(rr)
        f = (b*p - q) / b
        return float(max(0.0, min(0.25, f)))
    except Exception:
        return 0.0

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
            feats = _trend_features(g)
            strat = _pick_strategy(feats, atr_pct)
            reversal_score = pd.to_numeric(pd.Series([feats.get("reversal_score", np.nan)]), errors="coerce").iloc[0]
            reversal_label = _short_reversal_label(feats)
            trend_audit = _build_trend_audit(feats)
            strategy_audit = _build_strategy_audit(feats, strat)

            rows.append({"銘柄": sym, "企業名": (stage1_df.loc[stage1_df["銘柄"]==sym, "企業名"].iloc[0] if ("企業名" in stage1_df.columns and len(stage1_df.loc[stage1_df["銘柄"]==sym])) else ""), "セクター": (stage1_df.loc[stage1_df["銘柄"]==sym, "セクター"].iloc[0] if ("セクター" in stage1_df.columns and len(stage1_df.loc[stage1_df["銘柄"]==sym])) else "不明"), "推奨方式": strat, "現在値（終値）": round(close_last,2), "TP目安": round(tp,2), "SL目安": round(sl,2),
                         "TP到達率": res["tp_hit_rate"], "期待値EV(R)": res["ev_r"], "平均利確日数": res["avg_tp_days"],
                         "平均逆行(R)": res["avg_dd_r"], "検証回数": res.get("trials",0), "利確スコア": res.get("swing_score", float("nan")), "短期反転スコア": float(round(reversal_score, 4)) if np.isfinite(reversal_score) else np.nan,
                         "短期反転確認": reversal_label, "戦略判定根拠": strategy_audit, "トレンド監査": trend_audit, "備考": res.get("note","")})
            guides.append({"銘柄": sym, "企業名": (stage1_df.loc[stage1_df["銘柄"]==sym, "企業名"].iloc[0] if ("企業名" in stage1_df.columns and len(stage1_df.loc[stage1_df["銘柄"]==sym])) else ""), "セクター": (stage1_df.loc[stage1_df["銘柄"]==sym, "セクター"].iloc[0] if ("セクター" in stage1_df.columns and len(stage1_df.loc[stage1_df["銘柄"]==sym])) else "不明"), "推奨方式": strat, "Entry目安": round(close_last,2), "SL目安": round(sl,2), "TP目安": round(tp,2), "最大保有": "10営業日", "短期反転確認": reversal_label, "トレンド監査": trend_audit})
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
    # --- COMPLETE AI: WalkForward / MonteCarlo / Kelly (budgeted) ---
    try:
        # Streamlit Cloud対策：重い計算は上位N件だけ＆時間予算で打ち切り
        top_n = int(os.environ.get("AI_EXTRA_TOP_N", "8"))
        time_budget_s = float(os.environ.get("AI_EXTRA_BUDGET_S", "18"))
        _t_ai0 = time.time()

        n_all = int(len(df))
        n_do = max(0, min(n_all, top_n))

        wf_wr_list = [float("nan")] * n_all
        wf_rr_list = [float("nan")] * n_all
        mc_list    = [float("nan")] * n_all
        k_list     = [float("nan")] * n_all

        syms = df["銘柄"].astype(str).tolist()
        for i_sym, sym2 in enumerate(syms[:n_do]):
            if (time.time() - _t_ai0) > time_budget_s:
                break
            g2 = hist[hist["Symbol"] == sym2].sort_values("Date")
            wr, rr = walk_forward_oos(
                g2,
                hold_days=10,
                tp_atr=1.5,
                sl_atr=1.0,
                train_days=120,
                test_days=40,
                step_days=40,
            )
            wf_wr_list[i_sym] = wr
            wf_rr_list[i_sym] = rr
            mc_list[i_sym] = montecarlo_dd5(wr, rr, trades=60, paths=250, seed=7)  # paths軽量化
            k_list[i_sym] = kelly_fraction(wr, rr)

        df["WF勝率（OOS）"] = wf_wr_list
        df["WF損益比RR（OOS）"] = wf_rr_list
        df["MC DD 5%（推定）"] = mc_list
        df["Kelly最適化（f）"] = k_list

        # ボラ調整（ATR%があれば）
        if "ATR%" in stage1_df.columns:
            atr_map = stage1_df.set_index("銘柄")["ATR%"].to_dict()
            atrs = df["銘柄"].map(lambda s: float(atr_map.get(s, float("nan"))))
        else:
            atrs = float("nan")
        vol_adj = pd.to_numeric(atrs, errors="coerce")
        vol_adj = (3.0 / vol_adj).where(np.isfinite(vol_adj) & (vol_adj > 0), 1.0).clip(0.35, 1.65)

        # AI総合スコア（置換：欠損は中立値）
        ret3m = pd.to_numeric(df.get("3ヶ月リターン", np.nan), errors="coerce").fillna(0.0)
        trend = pd.to_numeric(df.get("AIトレンド", np.nan), errors="coerce").fillna(0.5)
        wf = pd.to_numeric(df.get("WF勝率（OOS）", np.nan), errors="coerce").fillna(0.5)
        rr = pd.to_numeric(df.get("WF損益比RR（OOS）", np.nan), errors="coerce").fillna(1.5)
        dd5 = pd.to_numeric(df.get("MC DD 5%（推定）", np.nan), errors="coerce").fillna(-5.0)
        dd_score = 1.0/(1.0+np.exp(-(dd5+3.0)))
        kelly = pd.to_numeric(df.get("Kelly最適化（f）", np.nan), errors="coerce").fillna(0.0)

        base = (
            0.28 * np.tanh(4.0*ret3m) +
            0.22 * (trend-0.5)*2.0 +
            0.22 * (wf-0.5)*2.0 +
            0.10 * np.tanh(rr-1.0) +
            0.12 * (dd_score-0.5)*2.0 +
            0.06 * (kelly/0.25)
        )
        df["総合スコア"] = (base * vol_adj).astype(float)
        df = df.sort_values("総合スコア", ascending=False).copy()
    except Exception:
        pass

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
        if c in df.columns:
            df[c] = df[c].astype(float).round(4)
    df["TP到達率"] = (df["TP到達率"]*100.0).round(2)

    # 表示用: セクター/銘柄名（無い場合は空）
    if "セクター" not in df.columns:
        if "sector33_name" in df.columns:
            df["セクター"] = df["sector33_name"].fillna("不明")
        else:
            df["セクター"] = "不明"
    if "銘柄名" not in df.columns:
        if "name" in df.columns:
            df["銘柄名"] = df["name"].fillna("")
        else:
            df["銘柄名"] = ""

    cols = ["銘柄","銘柄名","セクター","3ヶ月リターン","WF勝率（OOS）","WF損益比RR（OOS）","MC DD 5%（推定）","総合スコア","推奨方式","Kelly最適化（f）","AIトレンド","現在値（終値）","TP目安","SL目安","TP到達率","期待値EV(R)","平均利確日数","平均逆行(R)","検証回数","利確スコア","バフェット簡易スコア","イベント注意","財務メモ"]
    # --- column safety: ensure optional AI columns exist (avoid KeyError) ---
    for _c in ["3ヶ月リターン","AIトレンド","WF勝率（OOS）","WF損益比RR（OOS）","MC DD 5%（推定）","Kelly最適化（f）"]:
        if _c not in df.columns:
            df[_c] = np.nan

    # keep column order; missing columns become NaN
    df = df.reindex(columns=cols)


    meta = {"ok": True, "elapsed_sec": time.time()-t0, "errors": errors, "sample_failures": sample_fail, "stage2_selected": int(len(df)), "diag": diag}
    return df, guide, meta


def run_scan_3stage(
    stage0_keep: int = 1200,
    stage1_keep: int = 300,
    stage2_keep: int = 60,
    min_price: float = 300.0,
    min_avg_volume: float = 30000.0,
    atr_pct_min: float = 1.0,
    atr_pct_max: float = 8.0,
    # Stage2 profit-eval config
    stage2_days: int = 180,
    stage2_min_bars: int = 60,
    include_fundamentals: bool = True,
    fundamentals_top_n: int = 20,
    # Stage3 capital optimization
    capital_total: float = 300000.0,
    max_positions: int = 1,
    capital_mode: str = "growth",
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    4000銘柄網羅の3段階スキャン + 資金効率最適化(Stage3)。
    main.py 側から未知の引数が渡っても落ちないよう **_ignored を受ける。
    """
    ensure_schema()
    t0 = time.time()
    diag: Dict[str, Any] = {
        "ok": True,
        "stage": "start",
        "mode": "stable",
        "errors": [],
        "sample_failures": [],
    }

    # Stage0
    s0, sec, m0 = stage0_select(min_price=min_price, min_avg_volume=min_avg_volume, keep=int(stage0_keep))
    diag["stage0"] = m0
    if s0.empty:
        diag["stage"] = "done"
        diag["mode"] = "degraded"
        diag["errors"].append("Stage0 empty: DB更新不足 or フィルタが厳しすぎます")
        diag["elapsed_sec"] = time.time() - t0
        return {"selected": pd.DataFrame(), "guide": pd.DataFrame(), "sector_strength": sec, "diag": diag}

    # Stage1
    s1, m1 = stage1_select(s0, keep=int(stage1_keep), atr_pct_min=float(atr_pct_min), atr_pct_max=float(atr_pct_max))
    diag["stage1"] = m1
    if s1.empty:
        diag["stage"] = "done"
        diag["mode"] = "degraded"
        diag["errors"].append("Stage1 empty: ATR%条件/履歴不足")
        diag["elapsed_sec"] = time.time() - t0
        return {"selected": pd.DataFrame(), "guide": pd.DataFrame(), "sector_strength": sec, "diag": diag}

    # Stage2
    sel, guide, m2 = stage2_rank(
        s1,
        keep=int(stage2_keep),
        stage2_days=int(stage2_days),
        min_bars=int(stage2_min_bars),
        include_fundamentals=bool(include_fundamentals),
        fundamentals_top_n=int(fundamentals_top_n),
    )
    diag["stage2"] = m2
    if sel.empty:
        diag["mode"] = "degraded"
        diag["errors"].append("Stage2 empty: 利確評価失敗")
        diag["stage"] = "done"
        diag["elapsed_sec"] = time.time() - t0
    
    # Stage3: capital efficiency + execution sizing (v17.7)
    try:
        # merge guide into selected
        try:
            if isinstance(guide, pd.DataFrame) and not guide.empty:
                _g = guide.copy()
                keep_cols = [c for c in ["銘柄","企業名","セクター","推奨方式","Entry目安","SL目安","TP目安","最大保有"] if c in _g.columns]
                _g = _g[keep_cols].drop_duplicates(subset=["銘柄"])
                sel = sel.merge(_g, on="銘柄", how="left", suffixes=("","_guide"))
                for c in ["企業名","セクター","推奨方式","Entry目安","SL目安","TP目安","最大保有"]:
                    cg = f"{c}_guide"
                    if cg in sel.columns:
                        if c not in sel.columns:
                            sel[c] = sel[cg]
                        else:
                            sel[c] = sel[c].where(pd.notna(sel[c]), sel[cg])
                        sel.drop(columns=[cg], inplace=True)
        except Exception as _e_merge:
            diag["errors"].append(f"Stage3 guide merge warning: {type(_e_merge).__name__}: {_e_merge}")

        # company/sector from universe meta via sym_key
        try:
            meta_df = universe_load_meta()
            if isinstance(meta_df, pd.DataFrame) and not meta_df.empty:
                sel = _merge_meta_by_symkey(sel, meta_df)
        except Exception as _e_meta:
            diag["errors"].append(f"Stage3 meta warning: {type(_e_meta).__name__}: {_e_meta}")

        # AI auto order unit / size
        risk_budget = float(capital_total) * 0.01
        budget_per_pos = float(capital_total) / max(int(max_positions), 1)

        shares_list = []
        invest_list = []
        loss_list = []
        reason_list = []
        rr_list = []
        status_list = []
        unit_list = []

        for _, r in sel.iterrows():
            entry = float(r.get("Entry目安", r.get("現在値（終値）", 0)) or 0)
            sl = float(r.get("SL目安", 0) or 0)
            tp = float(r.get("TP目安", 0) or 0)
            current = float(r.get("現在値（終値）", 0) or 0)

            risk_per_share = max(entry - sl, 0.0)
            shares = 0
            invest = 0.0
            loss = 0.0
            reason = ""
            unit = "S株"

            if entry <= 0 or sl <= 0 or tp <= 0:
                reason = "エントリー未計算"
            elif risk_per_share <= 0:
                reason = "損切幅不正"
            else:
                shares_risk = int(risk_budget // risk_per_share)
                shares_cap = int(budget_per_pos // entry) if entry > 0 else 0
                shares100 = (min(shares_risk, shares_cap) // 100) * 100
                shares1 = min(shares_risk, shares_cap)
                if shares100 > 0:
                    shares = shares100
                    unit = "単元株"
                else:
                    shares = shares1
                    unit = "S株"
                invest = shares * entry
                loss = shares * risk_per_share
                if shares <= 0:
                    reason = "資金不足または単元未満"

            rr = ((tp - entry) / risk_per_share) if (risk_per_share > 0 and tp > 0) else np.nan
            status = "ブレイク済" if (current >= entry and entry > 0) else "待機"

            shares_list.append(int(shares))
            invest_list.append(float(invest))
            loss_list.append(float(loss))
            reason_list.append(reason)
            rr_list.append(rr)
            status_list.append(status)
            unit_list.append(unit)

        sel["発注単位"] = unit_list
        sel["推奨株数"] = shares_list
        sel["推奨投資額(円)"] = np.round(invest_list, 0)
        sel["想定損失(円)"] = np.round(loss_list, 0)
        sel["発注不可理由"] = reason_list
        sel["RR"] = np.round(rr_list, 3)
        sel["Entry状態"] = status_list

        def _z(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if x.size == 0:
                return x
            mn = np.nanmin(x)
            mx = np.nanmax(x)
            if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) <= 1e-12:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        entry_arr = pd.to_numeric(_safe_series(sel, "Entry目安", 0), errors="coerce").fillna(0).values
        tp_arr = pd.to_numeric(_safe_series(sel, "TP目安", 0), errors="coerce").fillna(0).values
        invest_arr = pd.to_numeric(_safe_series(sel, "推奨投資額(円)", 0), errors="coerce").fillna(0).values
        prof_arr = np.maximum(tp_arr - entry_arr, 0) * pd.to_numeric(_safe_series(sel, "推奨株数", 0), errors="coerce").fillna(0).values
        sel["想定利確額(円)"] = np.round(prof_arr, 0)
        sel["資金効率(利確/投入)"] = np.where(invest_arr > 0, (prof_arr / np.maximum(invest_arr, 1))*100.0, 0.0)

        eff_n = _z(pd.to_numeric(sel["資金効率(利確/投入)"], errors="coerce").fillna(0).values)
        daily_n = _z(pd.to_numeric(_safe_series(sel, "想定日次利確(円/日)", 0), errors="coerce").fillna(0).values)
        ev_n = _z(pd.to_numeric(_safe_series(sel, "期待値EV(R)", 0), errors="coerce").fillna(0).values)
        wr_n = _z(pd.to_numeric(_safe_series(sel, "TP到達率", 0), errors="coerce").fillna(0).values)
        dd_n = 1.0 - _z(pd.to_numeric(_safe_series(sel, "平均逆行(R)", 0), errors="coerce").fillna(0).values)
        buff_n = _z(pd.to_numeric(_safe_series(sel, "バフェット簡易スコア", 0), errors="coerce").fillna(0).values)

        sel["資金最適スコア"] = np.round(
            0.30 * eff_n + 0.18 * daily_n + 0.18 * ev_n + 0.14 * wr_n + 0.10 * dd_n + 0.10 * buff_n,
            6,
        )
        sort_cols = ["資金最適スコア","総合スコア"] if "総合スコア" in sel.columns else ["資金最適スコア"]
        sel = sel.sort_values(sort_cols, ascending=False).reset_index(drop=True)
        sel.insert(0, "順位", range(1, len(sel) + 1))
    except Exception as e:
        diag["mode"] = "degraded"
        diag["errors"].append(f"Stage3 warning: 資金最適化/発注計画で例外: {type(e).__name__}: {e}")

    # final cleanup
    for c in ["企業名","セクター","発注不可理由","Entry状態","発注単位"]:
        if c not in sel.columns:
            sel[c] = ""
        sel[c] = sel[c].fillna("").astype(str)
        if c in ["企業名","セクター"]:
            sel[c] = sel[c].replace(["","None","none","nan","NaN"], "不明")
        else:
            sel[c] = sel[c].replace(["None","none","nan","NaN"], "")
    for c in ["現在値（終値）","Entry目安","SL目安","TP目安","RR","推奨投資額(円)","想定損失(円)","総合スコア"]:
        if c not in sel.columns:
            sel[c] = 0.0
        sel[c] = pd.to_numeric(sel[c], errors="coerce").fillna(0.0)
    if "推奨株数" not in sel.columns:
        sel["推奨株数"] = 0
    sel["推奨株数"] = pd.to_numeric(sel["推奨株数"], errors="coerce").fillna(0).astype(int)
    if "最大保有" not in sel.columns:
        sel["最大保有"] = "10営業日"
    sel["最大保有"] = sel["最大保有"].fillna("10営業日").astype(str)

    diag["stage"] = "done"
    diag["elapsed_sec"] = time.time() - t0
    guide = sel[[c for c in ["銘柄","企業名","セクター","推奨方式","発注単位","推奨株数","推奨投資額(円)","想定損失(円)","Entry目安","SL目安","TP目安","最大保有","Entry状態"] if c in sel.columns]].copy() if isinstance(sel, pd.DataFrame) and not sel.empty else guide
    return {"selected": sel, "guide": guide, "sector_strength": sec, "diag": diag}

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


def _safe_series(df: pd.DataFrame, key: str, default=0):
    import pandas as pd
    if key not in df.columns:
        df[key] = pd.Series([default] * len(df), index=df.index)
    s = df[key]
    if not isinstance(s, pd.Series):
        s = pd.Series([default] * len(df), index=df.index)
        df[key] = s
    return s

def _clean_text(x, default=""):
    if x is None:
        return default
    s = str(x).strip()
    if s.lower() in ["none","nan","nat",""]:
        return default
    return s

def _norm_sector(x):
    return _clean_text(x, "不明")

def _merge_meta_by_symkey(sel: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    if sel is None or sel.empty:
        return sel
    out = sel.copy()
    out["_key"] = out["銘柄"].astype(str).map(_sym_key)
    if "企業名" not in out.columns:
        out["企業名"] = ""
    if "銘柄名" in out.columns:
        out["企業名"] = out["企業名"].where(out["企業名"].astype(str).str.strip() != "", out["銘柄名"].fillna("").astype(str))
    if "name" in out.columns:
        out["企業名"] = out["企業名"].where(out["企業名"].astype(str).str.strip() != "", out["name"].fillna("").astype(str))
    if "セクター" not in out.columns:
        out["セクター"] = ""
    if "sector33_name" in out.columns:
        out["セクター"] = out["セクター"].where(out["セクター"].astype(str).str.strip() != "", out["sector33_name"].fillna("").astype(str))

    if meta_df is not None and isinstance(meta_df, pd.DataFrame) and not meta_df.empty and "symbol" in meta_df.columns:
        m = meta_df.copy()
        m["_key"] = m["symbol"].astype(str).map(_sym_key)
        keep = [c for c in ["_key","name","sector33_name"] if c in m.columns]
        m = m[keep].drop_duplicates(subset=["_key"])
        out = out.merge(m, on="_key", how="left", suffixes=("","_meta"))
        if "name_meta" in out.columns:
            out["企業名"] = out["企業名"].where(out["企業名"].astype(str).str.strip() != "", out["name_meta"].fillna("").astype(str))
        if "sector33_name_meta" in out.columns:
            out["セクター"] = out["セクター"].where(out["セクター"].astype(str).str.strip() != "", out["sector33_name_meta"].fillna("").astype(str))

    out["企業名"] = out["企業名"].apply(lambda x: _clean_text(x, "不明"))
    out["セクター"] = out["セクター"].apply(_norm_sector)
    return out.drop(columns=["_key"], errors="ignore")




# ==============================
# v18 AI MODULES
# ==============================

def breakout_quality_ai(df):
    try:
        ma20 = df["Close"].rolling(20).mean()
        ma60 = df["Close"].rolling(60).mean()
        vol20 = df["Volume"].rolling(20).mean()

        cond_trend = (df["Close"].iloc[-1] > ma20.iloc[-1]) and (df["Close"].iloc[-1] > ma60.iloc[-1])
        cond_vol = df["Volume"].iloc[-1] > 1.5 * vol20.iloc[-1]
        cond_break = df["High"].iloc[-1] >= df["High"].rolling(20).max().iloc[-1]

        score = 0
        if cond_trend: score += 0.4
        if cond_vol: score += 0.3
        if cond_break: score += 0.3
        return score
    except Exception:
        return 0.0


def liquidity_trap_filter(avg_volume):
    if avg_volume < 100000:
        return -0.2
    return 0.0


def atr_volatility_ai(atr_pct):
    try:
        return 1 - abs(atr_pct - 3)/5
    except:
        return 0


def earnings_risk_ai(symbol):
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal is None or len(cal) == 0:
            return False
        ed = cal.index[0]
        import datetime
        today = datetime.datetime.utcnow().date()
        diff = (ed.date() - today).days
        if -2 <= diff <= 5:
            return True
        return False
    except Exception:
        return False


def dynamic_tp_engine(entry, atr):
    tp1 = entry + 1.0 * atr
    tp2 = entry + 2.0 * atr
    trailing = atr
    return {
        "tp1": tp1,
        "tp2": tp2,
        "trail": trailing
    }






def _recalc_live_execution_plan(df_plan: pd.DataFrame, capital_total: float = 300000.0, max_positions: int = 1) -> pd.DataFrame:
    """ライブ再計算後の順位に対して、発注単位/株数/投資額/損失を再計算する。

    v5_exec:
    - 単元株は「予算内可否」と「推奨可否」を分離して可視化
    - selected_now は単元株主力候補中心で使えるよう、基礎列を整備
    - 後方互換のため 単元株可否 は 単元推奨可否 の別名として残す
    """
    import numpy as np
    import pandas as pd

    if df_plan is None or len(df_plan) == 0:
        return df_plan

    out = df_plan.copy()
    budget_per_pos = float(capital_total) / max(int(max_positions), 1)
    fraction_risk_budget = float(capital_total) * 0.01
    unit_risk_budget = float(capital_total) * 0.04

    shares_list = []
    invest_list = []
    loss_list = []
    reason_list = []
    unit_list = []
    rr_list = []
    lot_size_list = []
    lot_cost_list = []
    lot_loss_list = []
    lot_budget_ok_list = []
    lot_reco_ok_list = []
    lot_budget_reason_list = []
    lot_reco_reason_list = []
    priority_list = []

    for _, r in out.iterrows():
        entry = float(pd.to_numeric(pd.Series([r.get("Entry目安", r.get("現在値（終値）", 0))]), errors="coerce").fillna(0).iloc[0])
        sl = float(pd.to_numeric(pd.Series([r.get("SL目安", 0)]), errors="coerce").fillna(0).iloc[0])
        tp = float(pd.to_numeric(pd.Series([r.get("TP目安", 0)]), errors="coerce").fillna(0).iloc[0])
        existing_reason_raw = r.get("発注不可理由", "")
        existing_reason = str(existing_reason_raw).strip() if existing_reason_raw is not None else ""
        if existing_reason.lower() in ("nan", "none"):
            existing_reason = ""
        entry_status_raw = r.get("Entry状態", "")
        entry_status = str(entry_status_raw).strip() if entry_status_raw is not None else ""
        if entry_status.lower() in ("nan", "none"):
            entry_status = ""

        lot_size = int(pd.to_numeric(pd.Series([r.get("lot_size", 100)]), errors="coerce").fillna(100).iloc[0])
        if lot_size <= 0:
            lot_size = 100

        risk_per_share = max(entry - sl, 0.0)
        shares = 0
        invest = 0.0
        loss = 0.0
        reason = existing_reason
        unit = "S株"
        priority = "保留"

        lot_cost = float(lot_size) * max(entry, 0.0)
        lot_loss = float(lot_size) * max(risk_per_share, 0.0)
        lot_budget_ok = bool(entry > 0 and lot_cost <= budget_per_pos + 1e-9)
        lot_risk_ok = bool(risk_per_share > 0 and lot_loss <= unit_risk_budget + 1e-9)
        lot_reco_ok = bool(lot_budget_ok and lot_risk_ok)

        if lot_budget_ok:
            lot_budget_reason = "可"
        elif entry <= 0:
            lot_budget_reason = "単元の必要資金を計算できない"
        else:
            lot_budget_reason = "単元の必要資金が予算超過"

        if lot_reco_ok:
            lot_reco_reason = "可"
        elif not lot_budget_ok and not lot_risk_ok:
            lot_reco_reason = "単元の必要資金・想定損失が条件超過"
        elif not lot_budget_ok:
            lot_reco_reason = "単元の必要資金が予算超過"
        elif not lot_risk_ok:
            lot_reco_reason = "単元の想定損失が上限超過"
        else:
            lot_reco_reason = "条件未達"

        forced_block = False
        if existing_reason:
            forced_block = True
        if entry_status.startswith("見送り（価格未更新") or entry_status.startswith("価格未更新"):
            forced_block = True
            if not reason:
                reason = "ライブ価格未更新"

        if not forced_block:
            if entry <= 0 or sl <= 0 or tp <= 0:
                reason = "エントリー未計算"
                priority = "計算不可"
                lot_reco_ok = False
                lot_reco_reason = "エントリー未計算"
            elif risk_per_share <= 0:
                reason = "損切幅不正"
                priority = "計算不可"
                lot_reco_ok = False
                lot_reco_reason = "損切幅不正"
            else:
                shares_cap = int(budget_per_pos // entry) if entry > 0 else 0
                shares_risk_fraction = int(fraction_risk_budget // risk_per_share) if risk_per_share > 0 else 0
                shares_risk_unit = int(unit_risk_budget // risk_per_share) if risk_per_share > 0 else 0

                if lot_reco_ok:
                    shares_lot = (min(shares_cap, max(shares_risk_unit, lot_size)) // lot_size) * lot_size
                    shares = shares_lot if shares_lot >= lot_size else lot_size
                    unit = "単元株"
                    priority = "単元株主力候補"
                else:
                    shares = min(shares_risk_fraction, shares_cap)
                    unit = "S株"
                    if shares > 0:
                        priority = "S株補助候補"
                    else:
                        priority = "資金保留"
                        if not reason:
                            reason = "資金不足または損失条件未達"

                invest = shares * entry
                loss = shares * risk_per_share
        else:
            lot_reco_ok = False
            if not lot_reco_reason or lot_reco_reason == "可":
                lot_reco_reason = reason or "発注保留"
            priority = "保留"

        if forced_block and not reason:
            reason = "発注保留"

        rr = ((tp - entry) / risk_per_share) if (risk_per_share > 0 and tp > 0) else np.nan

        shares_list.append(int(shares))
        invest_list.append(float(invest))
        loss_list.append(float(loss))
        reason_list.append(reason)
        unit_list.append(unit)
        rr_list.append(float(rr) if np.isfinite(rr) else np.nan)
        lot_size_list.append(int(lot_size))
        lot_cost_list.append(float(lot_cost))
        lot_loss_list.append(float(lot_loss))
        lot_budget_ok_list.append("可" if lot_budget_ok else "不可")
        lot_reco_ok_list.append("可" if lot_reco_ok else "不可")
        lot_budget_reason_list.append(lot_budget_reason)
        lot_reco_reason_list.append(lot_reco_reason)
        priority_list.append(priority)

    out["発注単位"] = unit_list
    out["推奨株数"] = shares_list
    out["推奨投資額(円)"] = np.round(invest_list, 0)
    out["想定損失(円)"] = np.round(loss_list, 0)
    out["発注不可理由"] = reason_list
    out["RR"] = np.round(rr_list, 3)
    out["単元株数"] = lot_size_list
    out["単元必要資金(円)"] = np.round(lot_cost_list, 0)
    out["単元想定損失(円)"] = np.round(lot_loss_list, 0)
    out["単元予算可否"] = lot_budget_ok_list
    out["単元推奨可否"] = lot_reco_ok_list
    out["単元予算判定理由"] = lot_budget_reason_list
    out["単元推奨判定理由"] = lot_reco_reason_list
    out["単元株可否"] = out["単元推奨可否"]
    out["単元判定理由"] = out["単元推奨判定理由"]
    out["売買優先区分"] = priority_list

    if "最大保有" not in out.columns:
        out["最大保有"] = "10営業日"
    out["最大保有"] = out["最大保有"].fillna("10営業日").astype(str)
    return out



def _clean_exec_text(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\u3000", " ").strip()
    if s in ("nan", "NaN", "None", "none", "<NA>", "N/A", "null", "NULL"):
        return ""
    return re.sub(r"\s+", " ", s)


def _execution_band_label(code: int) -> str:
    return {
        90: "単元株即時候補",
        80: "単元株監視候補(押し目)",
        70: "単元株監視候補",
        60: "単元株追随候補(厳格)",
        50: "S株即時候補(例外)",
        40: "S株監視候補(押し目)",
        35: "S株発注圏候補(最終補完)",
        34: "S株追随候補(最終補完)",
        30: "S株監視候補",
        10: "発注保留",
        0: "見送り",
    }.get(int(code), "見送り")


def _derive_execution_band(
    row_like: Dict[str, Any],
    strict_chase_rr: float = 1.30,
    strict_s_rr: float = 1.35,
    now_rr_min: float = 1.00,
) -> Tuple[int, str, Dict[str, Any]]:
    import pandas as pd
    import numpy as np
    import re

    status = _clean_exec_text(row_like.get("Entry状態", ""))
    unit = _clean_exec_text(row_like.get("発注単位", ""))
    reason = _clean_exec_text(row_like.get("発注不可理由", ""))
    reversal_tag = _clean_exec_text(row_like.get("短期反転確認", "")) or "未確認"
    trend_text = _clean_exec_text(row_like.get("トレンド監査", ""))
    refresh_state = _clean_exec_text(row_like.get("価格更新状態", ""))
    refresh_memo = _clean_exec_text(row_like.get("価格更新メモ", ""))

    rr = pd.to_numeric(pd.Series([row_like.get("実質RR", np.nan)]), errors="coerce").iloc[0]
    fail_flag = pd.to_numeric(pd.Series([row_like.get("再計算失敗フラグ", 0)]), errors="coerce").fillna(0).iloc[0]
    shares = pd.to_numeric(pd.Series([row_like.get("推奨株数", 0)]), errors="coerce").fillna(0).iloc[0]
    reversal_score = pd.to_numeric(pd.Series([row_like.get("短期反転スコア", np.nan)]), errors="coerce").iloc[0]

    refresh_ok = bool(fail_flag <= 0) and (refresh_state == "" or ("更新成功" in refresh_state and "失敗" not in refresh_state))
    shares_ok = bool(shares > 0)
    budget_ok = _clean_exec_text(row_like.get("単元予算可否", "")) == "可"
    reco_ok = _clean_exec_text(row_like.get("単元推奨可否", row_like.get("単元株可否", ""))) == "可"
    reversal_ok = reversal_tag in ("確認", "準確認")
    reversal_strict = reversal_tag == "確認"
    trend_headwind = bool(re.search(r"下降継続|下向き圧力|弱い|調整継続", trend_text))
    strong_headwind = bool(re.search(r"下降継続", trend_text))
    actionable = refresh_ok and shares_ok and (reason == "") and status != "" and (not status.startswith("見送り"))

    soft_s_rr = min(float(strict_s_rr), max(1.20, float(strict_s_rr) - 0.20))
    soft_s_chase_rr = max(1.25, min(float(strict_chase_rr), float(strict_s_rr) - 0.05))

    code = 0
    if not actionable:
        code = 10 if (refresh_ok and reason != "") else 0
    elif unit == "単元株":
        if budget_ok and reco_ok:
            if status == "発注圏" and reversal_ok and ((not trend_headwind) or reversal_strict) and (pd.isna(rr) or rr >= max(float(now_rr_min), 1.00)):
                code = 90
            elif status == "追随可" and reversal_strict and (not trend_headwind) and (not pd.isna(rr)) and rr >= float(strict_chase_rr):
                code = 60
            elif status == "押し目待ち" and (pd.isna(rr) or rr >= 0.90) and (reversal_ok or (not strong_headwind)):
                code = 80
            elif status == "様子見" and (pd.isna(rr) or rr >= 0.95) and (reversal_ok or ((not strong_headwind) and (pd.isna(reversal_score) or reversal_score >= 0.42))):
                code = 70
    elif unit == "S株":
        if status == "発注圏" and reversal_strict and (not trend_headwind) and (not pd.isna(rr)) and rr >= float(strict_s_rr):
            code = 50
        elif status == "発注圏" and reversal_ok and (pd.isna(rr) or rr >= soft_s_rr):
            code = 35
        elif status == "追随可" and reversal_strict and (not trend_headwind) and (not pd.isna(rr)) and rr >= soft_s_chase_rr:
            code = 34
        elif status == "押し目待ち" and reversal_ok and (pd.isna(rr) or rr >= 0.95):
            code = 40
        elif status in ("様子見", "追随可") and reversal_ok and (pd.isna(rr) or rr >= 1.00) and (not strong_headwind):
            code = 30

    meta = {
        "status": status,
        "unit": unit,
        "reason": reason,
        "rr": float(rr) if pd.notna(rr) else np.nan,
        "refresh_ok": refresh_ok,
        "refresh_state": refresh_state,
        "refresh_memo": refresh_memo,
        "shares_ok": shares_ok,
        "budget_ok": budget_ok,
        "reco_ok": reco_ok,
        "reversal_ok": reversal_ok,
        "reversal_strict": reversal_strict,
        "reversal_score": float(reversal_score) if pd.notna(reversal_score) else np.nan,
        "trend_headwind": trend_headwind,
        "strong_headwind": strong_headwind,
        "actionable": actionable,
    }
    return int(code), _execution_band_label(int(code)), meta


def _compute_live_execution_priority(df: pd.DataFrame, strict_chase_rr: float = 1.30, strict_s_rr: float = 1.35) -> pd.DataFrame:
    """ライブ再計算後の実行優先度を付与する。selected_live_top20 の並び替えにも使う。"""
    import pandas as pd
    import numpy as np

    if df is None or len(df) == 0:
        return df

    out = df.copy()
    for c in ["Entry状態", "発注単位", "発注不可理由", "売買優先区分", "単元予算可否", "単元推奨可否", "単元株可否", "短期反転確認", "トレンド監査", "価格更新状態", "価格更新メモ"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].map(_clean_exec_text)
    for c in ["実質RR", "総合スコア", "再計算失敗フラグ", "短期反転スコア"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    band_codes = []
    actionable_flags = []
    for _, row in out.iterrows():
        code, _, meta = _derive_execution_band(
            row.to_dict(),
            strict_chase_rr=float(strict_chase_rr),
            strict_s_rr=float(strict_s_rr),
            now_rr_min=1.00,
        )
        band_codes.append(int(code))
        actionable_flags.append(bool(meta.get("actionable", False)))

    out["実行優先度"] = pd.Series(band_codes, index=out.index, dtype="int64")
    out["実行優先帯"] = out["実行優先度"].map(_execution_band_label).fillna("見送り")

    rr = out["実質RR"].fillna(0.0)
    base = out["総合スコア"].fillna(0.0)
    reversal_score = out["短期反転スコア"].fillna(0.0).clip(lower=0.0, upper=1.0)
    reversal_label = out["短期反転確認"].replace("", "未確認")
    unit_budget_ok = out.get("単元予算可否", pd.Series([""] * len(out), index=out.index)).astype(str).eq("可")
    unit_reco_ok = out.get("単元推奨可否", pd.Series([""] * len(out), index=out.index)).astype(str).eq("可")
    trend_text = out.get("トレンド監査", pd.Series([""] * len(out), index=out.index)).astype(str)
    trend_headwind = trend_text.str.contains("下降継続|下向き圧力|調整継続", regex=True, na=False)
    trend_tailwind = trend_text.str.contains("上昇基調|中期上向き", regex=True, na=False)
    actionable_series = pd.Series(actionable_flags, index=out.index, dtype=bool)

    rr_norm = rr.clip(lower=0.0, upper=2.5) / 2.5
    base_span = float(base.max() - base.min()) if len(base) else 0.0
    if base_span > 1e-12:
        base_norm = (base - float(base.min())) / base_span
    else:
        base_norm = pd.Series(0.5, index=out.index, dtype=float)

    reversal_bonus = reversal_label.map({"確認": 0.24, "準確認": 0.10, "未確認": -0.12}).fillna(0.0)
    headwind_penalty = trend_headwind.astype(float) * 0.12
    tailwind_bonus = trend_tailwind.astype(float) * 0.08
    unit_bonus = out.get("発注単位", pd.Series([""] * len(out), index=out.index)).astype(str).map({"単元株": 0.10, "S株": -0.04}).fillna(0.0)
    status_bonus = out.get("Entry状態", pd.Series([""] * len(out), index=out.index)).astype(str).map({"発注圏": 0.18, "追随可": 0.08, "押し目待ち": 0.04, "様子見": -0.03}).fillna(0.0)
    actionable_bonus = actionable_series.astype(float) * 0.10

    out["実行優先スコア"] = (
        out["実行優先度"].astype(float)
        + 0.52 * rr_norm.fillna(0.0)
        + 0.20 * base_norm.fillna(0.0)
        + 0.18 * reversal_score.fillna(0.0)
        + reversal_bonus
        + 0.06 * unit_budget_ok.astype(float)
        + 0.04 * unit_reco_ok.astype(float)
        + actionable_bonus
        + status_bonus
        + unit_bonus
        + tailwind_bonus
        - headwind_penalty
    )
    return out


def _ensure_profit_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    for c in ["Entry目安", "TP目安", "推奨株数", "想定損失(円)"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    width = out["TP目安"] - out["Entry目安"]
    profit = width * out["推奨株数"].fillna(0)
    loss = pd.to_numeric(out.get("想定損失(円)", 0), errors="coerce").fillna(0)

    out["利確利幅(円/株)"] = width.round(2)
    out["想定利益(円)"] = profit.round(0)
    out["期待純益(円)"] = (profit.fillna(0) - loss).round(0)
    return out


LIVE_OUTPUT_SCHEMA = [
    "順位","銘柄","企業名","セクター","推奨方式","売買優先区分","実行優先帯","実行優先度","実行優先スコア","今すぐ発注スコア",
    "発注単位","単元予算可否","単元推奨可否","単元株可否","単元必要資金(円)","単元想定損失(円)","単元予算判定理由","単元推奨判定理由",
    "現在値（終値）","Entry目安","SL目安","TP目安","利確利幅(円/株)","想定利益(円)","期待純益(円)","RR","実質RR","価格更新状態","価格更新メモ","再計算失敗フラグ",
    "最大保有","推奨株数","推奨投資額(円)","想定損失(円)","総合スコア","Entry状態","発注不可理由","短期反転スコア","短期反転確認","戦略判定根拠","トレンド監査","Entry監査メモ",
    "選定区分","selected_now判定","selected_now除外理由","selected_now空理由集計",
    "元Entry目安","元SL目安","元TP目安","元RR","元総合スコア",
]


def _standardize_live_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame(columns=LIVE_OUTPUT_SCHEMA)

    out = _ensure_profit_display_columns(df.copy().reset_index(drop=True))
    if "name" in out.columns and "企業名" not in out.columns:
        out["企業名"] = out["name"]
    if "銘柄名" in out.columns and "企業名" not in out.columns:
        out["企業名"] = out["銘柄名"]
    if "sector33_name" in out.columns and "セクター" not in out.columns:
        out["セクター"] = out["sector33_name"]
    if "strategy_name" in out.columns and "推奨方式" not in out.columns:
        out["推奨方式"] = out["strategy_name"]

    text_defaults = {
        "銘柄":"", "企業名":"不明", "セクター":"不明", "推奨方式":"", "売買優先区分":"", "実行優先帯":"",
        "発注単位":"", "単元予算可否":"", "単元推奨可否":"", "単元株可否":"", "単元予算判定理由":"", "単元推奨判定理由":"",
        "価格更新状態":"", "価格更新メモ":"", "最大保有":"10営業日", "Entry状態":"", "発注不可理由":"", "短期反転確認":"", "戦略判定根拠":"", "トレンド監査":"", "Entry監査メモ":"",
        "選定区分":"", "selected_now判定":"", "selected_now除外理由":"", "selected_now空理由集計":"",
    }
    numeric_defaults = {
        "実行優先度":np.nan, "実行優先スコア":np.nan, "今すぐ発注スコア":np.nan,
        "単元必要資金(円)":np.nan, "単元想定損失(円)":np.nan,
        "現在値（終値）":np.nan, "Entry目安":np.nan, "SL目安":np.nan, "TP目安":np.nan, "利確利幅(円/株)":np.nan, "想定利益(円)":np.nan, "期待純益(円)":np.nan, "RR":np.nan, "実質RR":np.nan,
        "再計算失敗フラグ":np.nan, "推奨株数":np.nan, "推奨投資額(円)":np.nan, "想定損失(円)":np.nan, "総合スコア":np.nan, "短期反転スコア":np.nan,
        "元Entry目安":np.nan, "元SL目安":np.nan, "元TP目安":np.nan, "元RR":np.nan, "元総合スコア":np.nan,
    }

    for c, default in {**text_defaults, **numeric_defaults}.items():
        if c not in out.columns:
            out[c] = default

    for c in text_defaults:
        out[c] = out[c].fillna(text_defaults[c]).astype(str)
        if c in ["企業名", "セクター"]:
            out[c] = out[c].replace(["", "None", "none", "nan", "NaN"], text_defaults[c])
        else:
            out[c] = out[c].replace(["None", "none", "nan", "NaN"], "")

    for c in numeric_defaults:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "順位" in out.columns:
        out = out.drop(columns=["順位"], errors="ignore")
    out.insert(0, "順位", range(1, len(out) + 1))
    return out[[c for c in LIVE_OUTPUT_SCHEMA if c in out.columns or c == "順位"]]



def _split_live_rankings_core(
    df_selected: pd.DataFrame,
    now_top: int = 10,
    wait_top: int = 20,
    now_rr_min: float = 1.00,
    chase_rr_min: float = 1.45,
    wait_rr_min: float = 0.90,
    s_now_rr_min: float = 1.50,
    s_now_max: int = 1,
    chase_now_max: int = 1,
    recompute_priority: bool = True,
):
    import pandas as pd
    import numpy as np

    empty = pd.DataFrame()
    if df_selected is None or len(df_selected) == 0:
        return empty, empty

    work = _compute_live_execution_priority(
        df_selected,
        strict_chase_rr=float(chase_rr_min),
        strict_s_rr=float(s_now_rr_min),
    ).copy() if recompute_priority else df_selected.copy()

    for c in ["実行優先度", "実行優先スコア", "実質RR", "総合スコア", "短期反転スコア", "推奨投資額(円)", "想定損失(円)", "推奨株数", "再計算失敗フラグ"]:
        if c not in work.columns:
            work[c] = np.nan
        work[c] = pd.to_numeric(work[c], errors="coerce")
    for c in ["発注単位", "Entry状態", "短期反転確認", "トレンド監査"]:
        if c not in work.columns:
            work[c] = ""
        work[c] = work[c].map(_clean_exec_text)

    band_code = pd.to_numeric(work.get("実行優先度", 0), errors="coerce").fillna(0).astype(int)

    def _sort_candidates(df_part: pd.DataFrame) -> pd.DataFrame:
        if not len(df_part):
            return df_part
        return df_part.sort_values(
            ["実行優先度", "実行優先スコア", "実質RR", "短期反転スコア", "総合スコア"],
            ascending=[False, False, False, False, False],
            na_position="last",
        )

    def _take_band(target_code: int, limit, used_idx: set) -> pd.DataFrame:
        cand = _sort_candidates(work.loc[band_code.eq(int(target_code)) & (~work.index.isin(list(used_idx)))].copy())
        if not len(cand):
            return work.iloc[0:0].copy()
        if limit is None:
            return cand.copy()
        return cand.head(max(int(limit), 0)).copy()

    now_frames = []
    used_idx = set()
    remaining_slots = None if now_top is None else max(int(now_top), 0)

    def _append_now(df_part: pd.DataFrame):
        nonlocal remaining_slots
        if not isinstance(df_part, pd.DataFrame) or not len(df_part):
            return
        now_frames.append(df_part)
        used_idx.update(df_part.index.tolist())
        if remaining_slots is not None:
            remaining_slots = max(remaining_slots - len(df_part), 0)

    _append_now(_take_band(90, remaining_slots, used_idx))
    has_core = sum(len(x) for x in now_frames) > 0

    if (not has_core) and max(int(chase_now_max or 0), 0) > 0 and (remaining_slots is None or remaining_slots > 0):
        limit = max(int(chase_now_max or 0), 0) if remaining_slots is None else min(max(int(chase_now_max or 0), 0), remaining_slots)
        _append_now(_take_band(60, limit, used_idx))
        has_core = sum(len(x) for x in now_frames) > 0

    if (not has_core) and max(int(s_now_max or 0), 0) > 0 and (remaining_slots is None or remaining_slots > 0):
        limit = max(int(s_now_max or 0), 0) if remaining_slots is None else min(max(int(s_now_max or 0), 0), remaining_slots)
        _append_now(_take_band(50, limit, used_idx))
        has_core = sum(len(x) for x in now_frames) > 0

    if (not has_core) and (remaining_slots is None or remaining_slots > 0):
        _append_now(_take_band(35, 1 if remaining_slots is None else min(1, remaining_slots), used_idx))
        has_core = sum(len(x) for x in now_frames) > 0

    if (not has_core) and (remaining_slots is None or remaining_slots > 0):
        _append_now(_take_band(34, 1 if remaining_slots is None else min(1, remaining_slots), used_idx))
        has_core = sum(len(x) for x in now_frames) > 0

    now_df = pd.concat(now_frames, axis=0, ignore_index=False, sort=False) if now_frames else work.iloc[0:0].copy()

    if len(now_df):
        def _norm01(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            lo = s.min(skipna=True)
            hi = s.max(skipna=True)
            if pd.isna(lo) or pd.isna(hi) or hi <= lo:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - lo) / (hi - lo)

        base_norm = _norm01(now_df["総合スコア"])
        rr_norm = _norm01(now_df["実質RR"].fillna(0.0))
        exec_norm = _norm01(now_df["実行優先スコア"].fillna(0.0))
        rev_series = pd.to_numeric(now_df.get("短期反転スコア", pd.Series([0.0] * len(now_df), index=now_df.index)), errors="coerce").fillna(0.0)
        reversal_norm = _norm01(rev_series)
        reversal_label = now_df.get("短期反転確認", pd.Series(["未確認"] * len(now_df), index=now_df.index)).astype(str).replace("", "未確認")
        status_bonus = now_df["Entry状態"].astype(str).map({"発注圏": 1.20, "追随可": -0.25}).fillna(0.0)
        reversal_bonus = reversal_label.map({"確認": 0.42, "準確認": 0.14, "未確認": -0.30}).fillna(0.0)
        unit_bonus = now_df["発注単位"].astype(str).map({"単元株": 0.46, "S株": -0.14}).fillna(0.0)
        priority_bonus = now_df.get("売買優先区分", pd.Series([""] * len(now_df), index=now_df.index)).astype(str).map({"単元株主力候補": 0.28, "S株補助候補": -0.10}).fillna(0.0)
        trend_bonus = now_df.get("トレンド監査", pd.Series([""] * len(now_df), index=now_df.index)).astype(str).map(lambda x: 0.12 if ("上昇基調" in x or "中期上向き" in x) else (-0.18 if "下降継続" in x else 0.0))

        now_df["今すぐ発注スコア"] = (
            0.42 * exec_norm
            + 0.18 * rr_norm
            + 0.08 * base_norm
            + 0.10 * reversal_norm
            + 0.14 * status_bonus
            + reversal_bonus
            + unit_bonus
            + priority_bonus
            + trend_bonus
        )
        now_df = now_df.sort_values(
            ["実行優先度", "今すぐ発注スコア", "実質RR", "総合スコア"],
            ascending=[False, False, False, False],
            na_position="last",
        ).copy()
        if now_top is not None:
            now_df = now_df.head(int(now_top)).copy()

    wait_band_codes = {80, 70, 40, 30}
    wait_df = _sort_candidates(work.loc[(~work.index.isin(list(used_idx))) & band_code.isin(list(wait_band_codes))].copy())
    if len(wait_df) and "銘柄" in wait_df.columns and len(now_df) and "銘柄" in now_df.columns:
        wait_df = wait_df.loc[~wait_df["銘柄"].astype(str).isin(now_df["銘柄"].astype(str))].copy()
    if len(wait_df):
        if "銘柄" in wait_df.columns:
            wait_df = wait_df.drop_duplicates(subset=["銘柄"], keep="first")
        if wait_top is not None:
            wait_df = wait_df.head(int(wait_top)).copy()

    return now_df, wait_df


def _format_now_empty_reason_summary(work: pd.DataFrame) -> str:
    import pandas as pd
    if work is None or not isinstance(work, pd.DataFrame) or len(work) == 0:
        return "selected_now候補なし"
    reasons = work.get("selected_now除外理由", pd.Series(dtype=str)).astype(str).str.strip()
    reasons = reasons[(reasons != "") & (reasons != "採用") & (reasons != "監視候補")]
    counts = reasons.value_counts()
    if counts.empty:
        return "selected_now候補なし"
    parts = [f"{idx}:{int(val)}件" for idx, val in counts.head(5).items()]
    return "selected_nowが空の理由集計 / " + " / ".join(parts)



def _annotate_execution_membership(
    work: pd.DataFrame,
    now_ids: list,
    wait_ids: list,
    now_rr_min: float = 1.00,
    chase_rr_min: float = 1.45,
    s_now_rr_min: float = 1.50,
) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    out = work.copy()
    for c in ["選定区分", "selected_now判定", "selected_now除外理由", "selected_now空理由集計"]:
        if c not in out.columns:
            out[c] = ""
    out["選定区分"] = "selected_live_only"
    out["selected_now判定"] = "対象外"
    out["selected_now除外理由"] = ""
    out["selected_now空理由集計"] = ""

    if "__row_id__" not in out.columns:
        out["__row_id__"] = np.arange(len(out), dtype=int)

    band_code = pd.to_numeric(out.get("実行優先度", pd.Series([0] * len(out), index=out.index)), errors="coerce").fillna(0).astype(int)
    now_id_set = set(int(x) for x in now_ids)
    wait_id_set = set(int(x) for x in wait_ids)
    unit_core_available = bool(band_code.eq(90).any())

    band_reason_map = {
        80: "監視候補（押し目待ち）",
        70: "監視候補（様子見）",
        40: "監視候補（押し目待ち）",
        30: "監視候補（様子見）",
        90: "発注圏だが上位枠外",
        60: "単元株発注圏優先のため追随可抑制" if unit_core_available else "追随可上限超過",
        50: "単元株発注圏優先のためS株例外不採用" if unit_core_available else "S株例外/最終補完上限超過",
        35: "単元株発注圏優先のためS株例外不採用" if unit_core_available else "S株発注圏候補（最終補完未採用）",
        34: "単元株発注圏優先のためS株追随候補不採用" if unit_core_available else "S株追随候補（最終補完未採用）",
        10: "発注保留",
        0: "対象外",
    }

    for idx, row in out.iterrows():
        row_id = int(pd.to_numeric(row.get("__row_id__"), errors="coerce"))
        stt = _clean_exec_text(row.get("Entry状態", ""))
        rsn = _clean_exec_text(row.get("発注不可理由", ""))
        fail_flag = pd.to_numeric(pd.Series([row.get("再計算失敗フラグ")]), errors="coerce").fillna(0).iloc[0]
        shares = pd.to_numeric(pd.Series([row.get("推奨株数")]), errors="coerce").fillna(0).iloc[0]
        code = int(pd.to_numeric(pd.Series([row.get("実行優先度", 0)]), errors="coerce").fillna(0).iloc[0])

        if row_id in now_id_set:
            out.at[idx, "選定区分"] = "selected_now"
            out.at[idx, "selected_now判定"] = "採用"
            out.at[idx, "selected_now除外理由"] = "採用"
            continue
        if row_id in wait_id_set:
            out.at[idx, "選定区分"] = "selected_wait"
            out.at[idx, "selected_now判定"] = "監視候補"

        if fail_flag > 0:
            out.at[idx, "selected_now除外理由"] = "再計算失敗/価格未更新"
            continue
        if shares <= 0:
            out.at[idx, "selected_now除外理由"] = "推奨株数0"
            continue
        if rsn:
            out.at[idx, "selected_now除外理由"] = rsn
            continue
        if stt.startswith("見送り"):
            out.at[idx, "selected_now除外理由"] = stt or "見送り"
            continue

        reason = band_reason_map.get(code, "対象外")
        if code == 0 and stt in ["発注圏", "追随可", "押し目待ち", "様子見"]:
            unit = _clean_exec_text(row.get("発注単位", ""))
            reason = f"優先帯未付与（{stt}/{unit}）"
        out.at[idx, "selected_now除外理由"] = reason

    if len(now_id_set) == 0:
        summary = _format_now_empty_reason_summary(out)
        out["selected_now空理由集計"] = summary
    return out


def build_live_linked_guide(df_selected: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    """selected から live 再計算後の guide を作る。"""
    import pandas as pd
    guide_cols = ["銘柄","企業名","セクター","推奨方式","売買優先区分","実行優先帯","発注単位","単元予算可否","単元推奨可否","推奨株数","推奨投資額(円)","想定利益(円)","期待純益(円)","想定損失(円)","Entry目安","SL目安","TP目安","最大保有","Entry状態","短期反転確認","トレンド監査","Entry監査メモ"]
    if df_selected is None or len(df_selected) == 0:
        return pd.DataFrame(columns=guide_cols)

    cols = [c for c in ["銘柄","企業名","セクター","推奨方式","売買優先区分","実行優先帯","発注単位","単元予算可否","単元推奨可否","単元株可否","推奨株数","推奨投資額(円)","想定利益(円)","期待純益(円)","想定損失(円)","Entry目安","SL目安","TP目安","最大保有","Entry状態","短期反転確認","トレンド監査","Entry監査メモ"] if c in df_selected.columns]
    guide = df_selected[cols].copy()
    if max_rows is not None:
        guide = guide.head(int(max_rows)).copy()
    return guide


def refresh_topn_prices_and_recalc(
    df_selected,
    top_n=20,
    tp_atr=1.5,
    sl_atr=1.0,
    capital_total=300000.0,
    max_positions=1,
    effective_rr_floor=1.0,
    chase_rr_floor=1.15,
):
    """上位N銘柄だけ現在値を再取得し、
    - 現在値基準で Entry/TP/SL を再計算
    - 元のTP/SLに対する 実質RR を算出
    - Entry状態を再定義
    - ライブ再計算後スコアで再ランキング
    - 価格未更新銘柄を強警告＋大幅減点
    - selected_live_top20 自体も実行優先順へ再整列
    して返す。
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np

    if df_selected is None or len(df_selected) == 0:
        return df_selected

    out = df_selected.copy()
    work = out.head(int(top_n)).copy()

    for c in [
        "銘柄",
        "現在値（終値）",
        "Entry目安",
        "SL目安",
        "TP目安",
        "RR",
        "総合スコア",
        "推奨株数",
    ]:
        if c not in work.columns:
            work[c] = np.nan
    for c in ["発注不可理由", "Entry状態", "短期反転確認", "戦略判定根拠", "トレンド監査", "Entry監査メモ"]:
        if c not in work.columns:
            work[c] = ""
        work[c] = work[c].astype(object)
    if "短期反転スコア" not in work.columns:
        work["短期反転スコア"] = np.nan

    if "元Entry目安" not in work.columns:
        work["元Entry目安"] = pd.to_numeric(work["Entry目安"], errors="coerce")
    if "元SL目安" not in work.columns:
        work["元SL目安"] = pd.to_numeric(work["SL目安"], errors="coerce")
    if "元TP目安" not in work.columns:
        work["元TP目安"] = pd.to_numeric(work["TP目安"], errors="coerce")
    if "元RR" not in work.columns:
        work["元RR"] = pd.to_numeric(work["RR"], errors="coerce")
    if "元総合スコア" not in work.columns:
        work["元総合スコア"] = pd.to_numeric(work["総合スコア"], errors="coerce")

    symbols = []
    for s in work["銘柄"].tolist():
        s = str(s).strip()
        if not s:
            symbols.append("")
            continue
        if not s.endswith(".T") and s.isdigit():
            s = s + ".T"
        symbols.append(s)

    valid_symbols = [s for s in symbols if s]

    trend_feature_map = {}
    try:
        hist_recent = fetch_last_n_days(valid_symbols, n_days=90)
        if isinstance(hist_recent, pd.DataFrame) and len(hist_recent):
            for sym_hist, g_hist in hist_recent.groupby("Symbol"):
                try:
                    sym_key = str(sym_hist).strip()
                    if sym_key and sym_key.isdigit():
                        sym_key = sym_key + ".T"
                    trend_feature_map[sym_key] = _trend_features(g_hist.sort_values("Date"))
                except Exception:
                    continue
    except Exception:
        trend_feature_map = {}

    price_map = {}
    fetch_error_map = {}
    try:
        data = yf.download(
            " ".join(valid_symbols),
            period="1d",
            interval="1m",
            progress=False,
            group_by="ticker",
            threads=True,
            auto_adjust=False,
        )
    except Exception as e:
        data = None
        for sym in valid_symbols:
            fetch_error_map[sym] = repr(e)

    if data is not None:
        for sym in valid_symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    ser = data[sym]["Close"].dropna()
                else:
                    ser = data["Close"].dropna()
                if len(ser):
                    px = float(ser.iloc[-1])
                    if np.isfinite(px) and px > 0:
                        price_map[sym] = px
                        continue
                fetch_error_map[sym] = "close_empty"
            except Exception as e:
                fetch_error_map[sym] = repr(e)

    rr_floor = float(effective_rr_floor) if np.isfinite(effective_rr_floor) else 1.0
    chase_floor = max(rr_floor, float(chase_rr_floor) if np.isfinite(chase_rr_floor) else rr_floor)

    def _status_from_live(current, old_entry, old_tp, effective_rr, live_price_ok, features):
        label = _short_reversal_label(features)
        reversal_score = pd.to_numeric(pd.Series([features.get("reversal_score", np.nan)]), errors="coerce").iloc[0]
        trend_headwind = bool(features.get("trend_headwind", False))
        strong_downtrend = bool(features.get("strong_downtrend", False))
        pullback_candidate = bool(features.get("pullback_candidate", False))
        reversal_confirmed = bool(features.get("reversal_confirmed", False))
        pos_vs_ma20 = pd.to_numeric(pd.Series([features.get("pos_vs_ma20", np.nan)]), errors="coerce").iloc[0]

        if not live_price_ok:
            return "見送り（価格未更新）", "ライブ価格を取得できませんでした"

        if not np.isfinite(current) or current <= 0:
            return "不明", "ライブ価格が不明です"

        if np.isfinite(old_tp) and current >= old_tp * 0.995:
            return "見送り（利確圏）", "元TPに近く、新規エントリー妙味が薄い状態です"

        dev = np.nan
        if np.isfinite(old_entry) and old_entry > 0:
            dev = (current / old_entry) - 1.0

        if np.isfinite(effective_rr) and effective_rr < min(rr_floor, 0.90):
            return "見送り（利幅薄い）", f"実質RRが不足しています ({float(effective_rr):.2f})"

        if strong_downtrend and not reversal_confirmed:
            return "様子見", "下降継続が強く、短期反転が未確認です"

        if trend_headwind and not reversal_confirmed:
            if np.isfinite(dev) and dev <= -0.01:
                return "押し目待ち", "まだ下降圧力が残るため反転確認待ちです"
            return "様子見", "下降圧力が残り、短期反転が未確認です"

        if pullback_candidate and label in ["確認", "準確認"]:
            if np.isfinite(dev) and abs(dev) <= 0.012 and (not np.isfinite(effective_rr) or effective_rr >= rr_floor):
                return "発注圏", "上昇基調内の押し目で、短期反転も確認できています"
            if np.isfinite(dev) and dev < -0.012 and (not np.isfinite(effective_rr) or effective_rr >= min(rr_floor, 0.90)):
                return "押し目待ち", "上昇基調内の押し目だが、まだ引き付け余地があります"
            if np.isfinite(dev) and 0.012 < dev <= 0.025 and (not np.isfinite(effective_rr) or effective_rr >= chase_floor) and label == "確認":
                return "追随可", "反転確認後の上放れで、厳格な追随条件を満たしています"

        if reversal_confirmed:
            if np.isfinite(dev) and abs(dev) <= 0.010 and (not np.isfinite(effective_rr) or effective_rr >= rr_floor):
                return "発注圏", "短期反転を確認し、旧Entry近辺で再評価しても妙味があります"
            if np.isfinite(dev) and 0.010 < dev <= 0.022 and (not np.isfinite(effective_rr) or effective_rr >= chase_floor):
                return "追随可", "短期反転確認後で、追随エントリー許容帯です"
            if np.isfinite(dev) and dev < -0.010 and (not np.isfinite(effective_rr) or effective_rr >= min(rr_floor, 0.90)):
                return "押し目待ち", "反転兆候はあるが、もう一段の押しを待ちたい位置です"

        if np.isfinite(reversal_score) and reversal_score >= 0.46:
            if np.isfinite(dev) and dev < -0.005:
                return "押し目待ち", "反転は準確認で、エントリーは引き付け優先です"
            if np.isfinite(pos_vs_ma20) and pos_vs_ma20 >= -0.01:
                return "様子見", "反転は準確認だが、終値での裏付けがもう一段ほしい状態です"

        return "様子見", "短期反転の確度が十分ではなく、監視優先です"

    def _status_bonus(s, reversal_label):
        base = {
            "発注圏": 0.18,
            "追随可": 0.02,
            "押し目待ち": 0.06,
            "様子見": -0.03,
            "見送り（利幅薄い）": -0.18,
            "見送り（利確圏）": -0.20,
            "見送り（価格未更新）": -0.60,
            "不明": -0.10,
        }.get(str(s), 0.0)
        extra = {"確認": 0.08, "準確認": 0.02, "未確認": -0.08}.get(str(reversal_label), 0.0)
        return base + extra

    live_scores = []
    eff_rrs = []
    statuses = []
    entry_audits = []
    reversal_scores = []
    reversal_labels = []
    trend_audits = []
    strategy_audits = []
    live_price_flags = []
    live_price_states = []
    live_price_notes = []

    for i, row in work.iterrows():
        sym = str(row.get("銘柄", "")).strip()
        sym_yf = sym if sym.endswith(".T") else (sym + ".T" if sym.isdigit() else sym)
        current_live = price_map.get(sym_yf, np.nan)
        live_price_ok = bool(np.isfinite(current_live) and current_live > 0)
        current_display = current_live if live_price_ok else pd.to_numeric(pd.Series([row.get("現在値（終値）")]), errors="coerce").fillna(0).iloc[0]

        old_entry = pd.to_numeric(pd.Series([row.get("元Entry目安", row.get("Entry目安"))]), errors="coerce").iloc[0]
        old_sl = pd.to_numeric(pd.Series([row.get("元SL目安", row.get("SL目安"))]), errors="coerce").iloc[0]
        old_tp = pd.to_numeric(pd.Series([row.get("元TP目安", row.get("TP目安"))]), errors="coerce").iloc[0]
        old_score = pd.to_numeric(pd.Series([row.get("元総合スコア", row.get("総合スコア"))]), errors="coerce").iloc[0]
        shares = pd.to_numeric(pd.Series([row.get("推奨株数", 0)]), errors="coerce").fillna(0).iloc[0]

        trend_info = trend_feature_map.get(sym_yf, {}) or {}
        if not trend_info and sym in trend_feature_map:
            trend_info = trend_feature_map.get(sym, {}) or {}

        atr = np.nan
        if np.isfinite(old_entry) and np.isfinite(old_sl) and old_entry > old_sl:
            atr = (old_entry - old_sl) / max(float(sl_atr), 1e-9)

        if live_price_ok and np.isfinite(current_live) and current_live > 0 and np.isfinite(atr) and atr > 0:
            new_entry = float(current_live)
            new_sl = float(current_live - sl_atr * atr)
            new_tp = float(current_live + tp_atr * atr)
            nominal_rr = (new_tp - new_entry) / max(new_entry - new_sl, 1e-9)
            work.at[i, "現在値（終値）"] = round(new_entry, 2)
            work.at[i, "Entry目安"] = round(new_entry, 2)
            work.at[i, "SL目安"] = round(new_sl, 2)
            work.at[i, "TP目安"] = round(new_tp, 2)
            work.at[i, "RR"] = round(nominal_rr, 3)
            if shares > 0:
                work.at[i, "推奨投資額(円)"] = round(float(shares) * new_entry, 0)
                work.at[i, "想定損失(円)"] = round(float(shares) * max(new_entry - new_sl, 0.0), 0)
        else:
            if np.isfinite(current_display) and current_display > 0:
                work.at[i, "現在値（終値）"] = round(float(current_display), 2)

        effective_rr = np.nan
        if live_price_ok and np.isfinite(current_live) and np.isfinite(old_tp) and np.isfinite(old_sl):
            risk_now = max(float(current_live) - float(old_sl), 0.0)
            reward_now = max(float(old_tp) - float(current_live), 0.0)
            if risk_now > 0:
                effective_rr = reward_now / risk_now
        eff_rrs.append(effective_rr)

        reversal_label = _short_reversal_label(trend_info) if trend_info else str(row.get("短期反転確認", "") or "")
        trend_audit = _build_trend_audit(trend_info) if trend_info else str(row.get("トレンド監査", "") or "")
        strategy_audit = _build_strategy_audit(trend_info, str(row.get("推奨方式", "") or "")) if trend_info else str(row.get("戦略判定根拠", "") or "")
        reversal_score = pd.to_numeric(pd.Series([trend_info.get("reversal_score", row.get("短期反転スコア", np.nan)) if trend_info else row.get("短期反転スコア", np.nan)]), errors="coerce").iloc[0]

        status, entry_audit = _status_from_live(current_live, old_entry, old_tp, effective_rr, live_price_ok, trend_info)
        statuses.append(status)
        entry_audits.append(entry_audit)
        reversal_scores.append(float(round(reversal_score, 4)) if np.isfinite(reversal_score) else np.nan)
        reversal_labels.append(reversal_label)
        trend_audits.append(trend_audit)
        strategy_audits.append(strategy_audit)

        note = ""
        if live_price_ok:
            live_price_flags.append(0)
            live_price_states.append("更新成功")
        else:
            live_price_flags.append(1)
            live_price_states.append("更新失敗")
            raw_note = str(fetch_error_map.get(sym_yf, "no_live_price"))
            note = raw_note[:160]
            live_price_notes.append(note)
            work.at[i, "発注不可理由"] = "ライブ価格再取得失敗"
        if live_price_ok:
            live_price_notes.append(note)

        base = float(old_score) if np.isfinite(old_score) else 0.0
        rr_component = min(max(float(effective_rr) if np.isfinite(effective_rr) else 0.0, 0.0), 2.5) / 2.5
        ext_penalty = 0.0
        if live_price_ok and np.isfinite(current_live) and np.isfinite(old_entry) and old_entry > 0:
            ext = max((float(current_live) / float(old_entry)) - 1.0, 0.0)
            ext_penalty = min(ext, 0.08) * 1.6
        reversal_component = min(max(float(reversal_score) if np.isfinite(reversal_score) else 0.0, 0.0), 1.0)
        live_score = 0.52 * base + 0.22 * rr_component + 0.10 * reversal_component + _status_bonus(status, reversal_label) - ext_penalty
        if trend_info and bool(trend_info.get("strong_downtrend", False)) and reversal_label != "確認":
            live_score -= 0.12
        if not live_price_ok:
            live_score -= 0.50
        live_scores.append(float(live_score))

    work["実質RR"] = np.round(eff_rrs, 3)
    work["Entry状態"] = statuses
    work["Entry監査メモ"] = entry_audits
    work["短期反転スコア"] = reversal_scores
    work["短期反転確認"] = reversal_labels
    work["トレンド監査"] = trend_audits
    work["戦略判定根拠"] = strategy_audits
    work["総合スコア"] = np.round(live_scores, 6)
    work["再計算失敗フラグ"] = live_price_flags
    work["価格更新状態"] = live_price_states
    work["価格更新メモ"] = live_price_notes

    work = _recalc_live_execution_plan(work, capital_total=float(capital_total), max_positions=int(max_positions))
    work = _compute_live_execution_priority(work, strict_chase_rr=max(chase_floor, 1.30), strict_s_rr=max(rr_floor, 1.35))
    work = work.sort_values(
        ["実行優先度", "実行優先スコア", "実質RR", "短期反転スコア", "総合スコア", "再計算失敗フラグ"],
        ascending=[False, False, False, False, False, True],
        na_position="last"
    ).reset_index(drop=True)
    work["順位"] = np.arange(1, len(work) + 1)

    tail = out.iloc[int(top_n):].copy() if len(out) > int(top_n) else out.iloc[0:0].copy()
    if len(tail):
        tail = tail.reset_index(drop=True)
    merged = pd.concat([work, tail], ignore_index=True, sort=False)
    return merged.head(int(top_n)).copy()


def split_live_rankings(
    df_selected: pd.DataFrame,
    now_top: int = 10,
    wait_top: int = 20,
    now_rr_min: float = 1.00,
    chase_rr_min: float = 1.40,
    wait_rr_min: float = 0.90,
    s_now_rr_min: float = 1.35,
    s_now_max: int = 1,
    chase_now_max: int = 1,
):
    """ライブ再計算後の selected を、build_live_execution_views と同一親DataFrame基準で now/wait に分割して返す。"""
    _, now_df, wait_df = build_live_execution_views(
        df_selected,
        live_top=max(int(now_top or 0) + int(wait_top or 0), 20),
        now_top=now_top,
        wait_top=wait_top,
        now_rr_min=now_rr_min,
        chase_rr_min=chase_rr_min,
        wait_rr_min=wait_rr_min,
        s_now_rr_min=s_now_rr_min,
        s_now_max=s_now_max,
        chase_now_max=chase_now_max,
    )
    return now_df, wait_df


def build_live_execution_views(
    df_selected: pd.DataFrame,
    live_top: int = 20,
    now_top: int = 10,
    wait_top: int = 20,
    now_rr_min: float = 1.00,
    chase_rr_min: float = 1.45,
    wait_rr_min: float = 0.90,
    s_now_rr_min: float = 1.50,
    s_now_max: int = 1,
    chase_now_max: int = 1,
):
    """selected_live_top20 / selected_now / selected_wait を同一親DataFrame・同一スキーマで再構成する。"""
    import pandas as pd
    import numpy as np

    empty = pd.DataFrame(columns=LIVE_OUTPUT_SCHEMA)
    if df_selected is None or len(df_selected) == 0:
        return empty, empty, empty

    work = _compute_live_execution_priority(
        df_selected,
        strict_chase_rr=float(chase_rr_min),
        strict_s_rr=float(s_now_rr_min),
    ).copy().reset_index(drop=True)
    work["__row_id__"] = np.arange(len(work), dtype=int)

    now_raw, wait_raw = _split_live_rankings_core(
        work,
        now_top=now_top,
        wait_top=wait_top,
        now_rr_min=now_rr_min,
        chase_rr_min=chase_rr_min,
        wait_rr_min=wait_rr_min,
        s_now_rr_min=s_now_rr_min,
        s_now_max=s_now_max,
        chase_now_max=chase_now_max,
        recompute_priority=False,
    )

    def _row_ids(df_part: pd.DataFrame):
        if not isinstance(df_part, pd.DataFrame) or len(df_part) == 0 or "__row_id__" not in df_part.columns:
            return []
        return pd.to_numeric(df_part["__row_id__"], errors="coerce").dropna().astype(int).tolist()

    now_ids = _row_ids(now_raw)
    wait_ids = [x for x in _row_ids(wait_raw) if x not in set(now_ids)]

    if len(now_ids) == 0 and int(now_top or 0) > 0:
        rescue = work.copy()
        rescue_codes = pd.to_numeric(rescue.get("実行優先度", pd.Series([0] * len(rescue), index=rescue.index)), errors="coerce").fillna(0).astype(int)
        rescue = rescue.loc[rescue_codes.isin([90, 60, 50, 35, 34])].copy()
        if len(rescue):
            rescue = rescue.sort_values(["実行優先度", "実行優先スコア", "実質RR", "短期反転スコア", "総合スコア"], ascending=[False, False, False, False, False], na_position="last")
            now_ids = pd.to_numeric(rescue["__row_id__"], errors="coerce").dropna().astype(int).tolist()[: max(1, int(now_top))]
            wait_ids = [x for x in wait_ids if x not in set(now_ids)]

    annotated = _annotate_execution_membership(
        work,
        now_ids=now_ids,
        wait_ids=wait_ids,
        now_rr_min=float(now_rr_min),
        chase_rr_min=float(chase_rr_min),
        s_now_rr_min=float(s_now_rr_min),
    )

    if len(annotated):
        selection_rank = annotated["選定区分"].map({"selected_now": 3, "selected_wait": 2, "selected_live_only": 1}).fillna(0)
        annotated = annotated.assign(__selection_rank__=selection_rank)
        annotated = annotated.sort_values(
            ["__selection_rank__", "実行優先度", "実行優先スコア", "実質RR", "総合スコア"],
            ascending=[False, False, False, False, False],
            na_position="last",
        ).copy()
    else:
        annotated["__selection_rank__"] = []

    parent = annotated.set_index("__row_id__", drop=False) if len(annotated) else annotated.copy()

    live_order = now_ids + wait_ids
    if len(parent):
        remaining = [int(x) for x in parent.index.tolist() if int(x) not in set(live_order)]
        live_order = live_order + remaining
    if live_top is not None:
        live_order = live_order[: int(live_top)]

    live_raw = parent.loc[[i for i in live_order if i in parent.index]].copy() if len(parent) else annotated.iloc[0:0].copy()
    now_raw = parent.loc[[i for i in now_ids if i in parent.index]].copy() if len(parent) and now_ids else annotated.iloc[0:0].copy()
    wait_raw = parent.loc[[i for i in wait_ids if i in parent.index]].copy() if len(parent) and wait_ids else annotated.iloc[0:0].copy()

    if len(wait_raw) and "銘柄" in wait_raw.columns and len(now_raw) and "銘柄" in now_raw.columns:
        wait_raw = wait_raw.loc[~wait_raw["銘柄"].astype(str).isin(now_raw["銘柄"].astype(str))].copy()
    if len(wait_raw):
        wait_raw = wait_raw.loc[~wait_raw["Entry状態"].astype(str).str.startswith("見送り")].copy()
        wait_raw = wait_raw.loc[wait_raw.get("実行優先帯", pd.Series([""] * len(wait_raw), index=wait_raw.index)).astype(str) != "見送り"].copy()
        if "銘柄" in wait_raw.columns:
            wait_raw = wait_raw.drop_duplicates(subset=["銘柄"], keep="first").copy()

    drop_helper_cols = ["__row_id__", "__selection_rank__"]
    live_df = _standardize_live_output_schema(live_raw.drop(columns=drop_helper_cols, errors="ignore"))
    now_df = _standardize_live_output_schema(now_raw.drop(columns=drop_helper_cols, errors="ignore"))
    wait_df = _standardize_live_output_schema(wait_raw.drop(columns=drop_helper_cols, errors="ignore"))

    if len(now_df) == 0:
        summary = _format_now_empty_reason_summary(live_df)
        if len(live_df):
            live_df["selected_now空理由集計"] = summary
        if len(wait_df):
            wait_df["selected_now空理由集計"] = summary

    if len(now_df) and len(wait_df) and "銘柄" in now_df.columns and "銘柄" in wait_df.columns:
        wait_df = wait_df.loc[~wait_df["銘柄"].astype(str).isin(now_df["銘柄"].astype(str))].copy()
        wait_df = _standardize_live_output_schema(wait_df)
    if len(live_df) and len(now_df) and "銘柄" in live_df.columns and "銘柄" in now_df.columns:
        now_df = now_df.loc[now_df["銘柄"].astype(str).isin(live_df["銘柄"].astype(str))].copy()
        now_df = _standardize_live_output_schema(now_df)
    if len(live_df) and len(wait_df) and "銘柄" in live_df.columns and "銘柄" in wait_df.columns:
        wait_df = wait_df.loc[wait_df["銘柄"].astype(str).isin(live_df["銘柄"].astype(str))].copy()
        wait_df = _standardize_live_output_schema(wait_df)

    return live_df, now_df, wait_df

