"""External US term-premium benchmark loaders (ACM, Kim-Wright, DKW).

Pulled alongside the underlying rates so the US series can be plotted side-by-side
with our estimate (and used as validation references — DKW in particular gives real
term-premium and inflation-risk-premium benchmarks for the joint nominal/real model).

Each loader mirrors ``validation/sf_fed.py``: an ``os.environ``-overridable URL, a
raw-file + parquet cache under ``validation/reports/``, and a cached-fallback when
the network/egress fails (raising a clear error only on a cold cache). All emit a
tidy frame in DECIMAL with a ``date`` datetime column, an integer ``maturity`` (years)
column, and one column per quantity.

Christensen-Rudebusch (SF Fed) stays in ``validation/sf_fed.py``.
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from validation import config as cfg

logger = logging.getLogger(__name__)
_TIMEOUT = 120
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}


# --------------------------------------------------------------------------- io
def _download_curl(url: str) -> bytes:
    """Fetch via the curl CLI (reliable through this environment's egress proxy for
    hosts where the requests/urllib stack times out, e.g. FRED's on-the-fly CSV)."""
    import subprocess
    out = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
         "-A", _HEADERS["User-Agent"], url],
        capture_output=True, timeout=_TIMEOUT + 10)
    if out.returncode != 0:
        raise RuntimeError(f"curl failed ({out.returncode}): {out.stderr[:200]!r}")
    return out.stdout


def _download_requests(url: str) -> bytes:
    import requests
    r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.content


def _download_bytes(url: str, retries: int = 4) -> bytes:
    """GET url as bytes with retry/backoff. curl is the primary transport (it is
    uniformly reliable through this environment's egress proxy, whereas requests
    times out on some hosts e.g. FRED); requests is the fallback for portability."""
    import time
    last = None
    for attempt in range(retries):
        for fetch in (_download_curl, _download_requests):
            try:
                return fetch(url)
            except Exception as exc:                               # noqa: BLE001
                last = exc
        if attempt < retries - 1:
            time.sleep(2 ** attempt)                               # 1,2,4,8s
    raise last


def _cached(parquet_path, refresh: bool, build_fn, label: str) -> pd.DataFrame:
    """Return the parsed benchmark frame, using the parquet cache unless ``refresh``;
    on any build/network failure fall back to the cache, else raise."""
    if not refresh and parquet_path.exists():
        return pd.read_parquet(parquet_path)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = build_fn()
        df.to_parquet(parquet_path)
        logger.info("%s: %d rows %s..%s, maturities %s", label, len(df),
                    df["date"].min().date(), df["date"].max().date(),
                    sorted(df["maturity"].unique()))
        return df
    except Exception as exc:                                       # noqa: BLE001
        if parquet_path.exists():
            logger.warning("%s fetch failed (%s); using cached parquet", label, exc)
            return pd.read_parquet(parquet_path)
        raise RuntimeError(
            f"{label}: fetch failed ({exc}) and no cache at {parquet_path}. "
            f"Check egress to the source host, or set the *_URL override.") from exc


# --------------------------------------------------------------------------- ACM
def parse_acm_xls(content: bytes) -> pd.DataFrame:
    """Parse the ACM .xls workbook bytes -> tidy (date, maturity, term_premium,
    expected_rate, fitted_yield) in DECIMAL. Pure (no network) for testability."""
    xl = pd.ExcelFile(io.BytesIO(content))          # legacy .xls (needs xlrd)
    sheet = next((s for s in xl.sheet_names if s.strip().lower() == "acm daily"),
                 xl.sheet_names[-1])
    raw = pd.read_excel(xl, sheet_name=sheet)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"], format="%d-%b-%Y", errors="coerce")
    raw = raw.dropna(subset=["date"])
    # ACMTP=term premium, ACMRNY=risk-neutral (expected avg short rate), ACMY=fitted.
    frames = []
    for n in range(1, 11):
        cols = {"term_premium": f"ACMTP{n:02d}", "expected_rate": f"ACMRNY{n:02d}",
                "fitted_yield": f"ACMY{n:02d}"}
        if not all(c in raw.columns for c in cols.values()):
            continue
        sub = raw[["date", *cols.values()]].copy()
        sub.columns = ["date", *cols.keys()]
        sub["maturity"] = float(n)
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    for q in ("term_premium", "expected_rate", "fitted_yield"):
        df[q] = pd.to_numeric(df[q], errors="coerce") / 100.0        # percent -> decimal
    return df[["date", "maturity", "term_premium", "expected_rate", "fitted_yield"]]


def _build_acm() -> pd.DataFrame:
    content = _download_bytes(cfg.ACM_URL)
    cfg.ACM_RAW_PATH.write_bytes(content)
    return parse_acm_xls(content)


def load_acm(refresh: bool = False) -> pd.DataFrame:
    """ACM (NY Fed) term premium / risk-neutral / fitted yields, 1-10y, daily (decimal)."""
    return _cached(cfg.ACM_PARQUET_PATH, refresh, _build_acm, "ACM")


# -------------------------------------------------------------------- Kim-Wright
def _build_kim_wright() -> pd.DataFrame:
    frames = []
    for n, sid in enumerate(cfg.KIM_WRIGHT_IDS, start=1):
        text = _download_bytes(cfg.KIM_WRIGHT_FRED_BASE + sid).decode("utf-8")
        s = pd.read_csv(io.StringIO(text))
        s.columns = ["date", "val"]
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s["val"] = pd.to_numeric(s["val"], errors="coerce")
        s = s.dropna(subset=["date"])
        s["maturity"] = float(n)
        frames.append(s.rename(columns={"val": "term_premium"}))
    df = pd.concat(frames, ignore_index=True)
    df["term_premium"] = df["term_premium"] / 100.0                  # percent -> decimal
    return df[["date", "maturity", "term_premium"]].dropna(subset=["term_premium"])


def load_kim_wright(refresh: bool = False) -> pd.DataFrame:
    """Kim-Wright (Fed Board, via FRED THREEFYTP01..10) term premium, 1-10y (decimal)."""
    return _cached(cfg.KIM_WRIGHT_PARQUET_PATH, refresh, _build_kim_wright, "Kim-Wright")


# --------------------------------------------------------------------------- DKW
_DKW_QUANTITIES = {                       # canonical name -> DKW column stem
    "real_term_premium": "real.term.prem",
    "inflation_risk_premium": "inflation.risk.prem",
    "expected_inflation": "exp.inflation",
    "expected_real_short_rate": "exp.real.short.rate",
    "tips_liquidity_premium": "tips.liq.prem",
    "nominal_yield": "nominal.yield.fitted",
}


def parse_dkw_csv(text: str) -> pd.DataFrame:
    """Parse the DKW_updates.csv text (multi-line preamble + wide 5/10/5f5 columns)
    -> tidy (date, maturity in {5,10}, term_premium=real+IRP, real_term_premium,
    inflation_risk_premium, expected_inflation, expected_real_short_rate, nominal_yield),
    DECIMAL. Pure (no network) for testability."""
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines)
               if l.replace('"', "").strip().lower().startswith("date,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    frames = []
    for mat in (5, 10):                    # spot decompositions (skip 5f5 forward)
        cols = {q: f"{stem}.{mat}" for q, stem in _DKW_QUANTITIES.items()}
        if not all(c in df.columns for c in cols.values()):
            continue
        sub = df[["date", *cols.values()]].copy()
        sub.columns = ["date", *cols.keys()]
        for q in _DKW_QUANTITIES:
            sub[q] = pd.to_numeric(sub[q], errors="coerce") / 100.0   # percent -> decimal
        # nominal term premium = real term premium + inflation risk premium
        sub["term_premium"] = sub["real_term_premium"] + sub["inflation_risk_premium"]
        sub["maturity"] = float(mat)
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    cols = ["date", "maturity", "term_premium", *_DKW_QUANTITIES.keys()]
    return df[cols]


def _build_dkw() -> pd.DataFrame:
    text = _download_bytes(cfg.DKW_URL).decode("utf-8", errors="replace")
    cfg.DKW_RAW_PATH.write_text(text, encoding="utf-8")
    return parse_dkw_csv(text)


def load_dkw(refresh: bool = False) -> pd.DataFrame:
    """DKW (Fed Board) 5y/10y decomposition: nominal TP (=real TP + IRP), real TP,
    inflation-risk premium, expected inflation, expected real short rate (decimal)."""
    return _cached(cfg.DKW_PARQUET_PATH, refresh, _build_dkw, "DKW")


# ------------------------------------------------------------------------- ATSIX
def parse_atsix_xlsx(content: bytes) -> pd.DataFrame:
    """Parse the ATSIX_Vintages.xlsx `InfExp` sheet -> tidy (date, maturity in years,
    expected_inflation) in DECIMAL. Columns are `infexp{h}` for horizons h=3..120
    months; `infexp{h}` is the expected AVERAGE inflation over the next h months.
    Pure (no network) for testability."""
    xl = pd.ExcelFile(io.BytesIO(content))
    sheet = next((s for s in xl.sheet_names if s.strip().lower() == "infexp"),
                 xl.sheet_names[0])
    raw = pd.read_excel(xl, sheet_name=sheet)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    cols = [c for c in raw.columns if str(c).lower().startswith("infexp")]
    long = raw.melt(id_vars=["date"], value_vars=cols,
                    var_name="col", value_name="expected_inflation")
    months = long["col"].str.extract(r"(\d+)").astype(float)[0]
    long["maturity"] = months / 12.0                      # months -> years
    long["expected_inflation"] = pd.to_numeric(
        long["expected_inflation"], errors="coerce") / 100.0   # percent -> decimal
    long = long.dropna(subset=["expected_inflation"])
    return long[["date", "maturity", "expected_inflation"]].sort_values(
        ["date", "maturity"]).reset_index(drop=True)


def _build_atsix() -> pd.DataFrame:
    content = _download_bytes(cfg.ATSIX_URL)
    cfg.ATSIX_RAW_PATH.write_bytes(content)
    return parse_atsix_xlsx(content)


def load_atsix(refresh: bool = False) -> pd.DataFrame:
    """ATSIX (Philadelphia Fed) survey-based expected-inflation term structure,
    horizons 3-120 months, monthly 1998+ (decimal, horizon-average)."""
    return _cached(cfg.ATSIX_PARQUET_PATH, refresh, _build_atsix, "ATSIX")


def atsix_panel(grid_years, dates=None, refresh: bool = False) -> pd.DataFrame:
    """Wide expected-inflation panel (index=date, columns=grid_years, DECIMAL) for
    anchoring the joint model. ``grid_years`` are matched to the nearest available
    ATSIX horizon (they line up exactly for integer years: 2y=infexp24, ...).
    If ``dates`` is given, reindex to them by month (ATSIX is monthly, dated to the
    1st; caller's month-end dates map by year-month)."""
    df = load_atsix(refresh=refresh)
    grid = np.asarray(grid_years, float)
    avail = np.sort(df["maturity"].unique())
    cols = {}
    for g in grid:
        near = avail[int(np.argmin(np.abs(avail - g)))]
        s = df[np.isclose(df["maturity"], near)].set_index("date")["expected_inflation"]
        cols[g] = s
    panel = pd.DataFrame(cols).sort_index()
    panel.columns = grid
    if dates is not None:
        dates = pd.DatetimeIndex(dates)
        by_month = panel.copy()
        by_month.index = panel.index.to_period("M")
        out = by_month.reindex(dates.to_period("M"))
        out.index = dates
        return out
    return panel


# --------------------------------------------------------------------------- all
LOADERS = {"acm": load_acm, "kim_wright": load_kim_wright, "dkw": load_dkw,
           "atsix": load_atsix}


def refresh_all_us_benchmarks(refresh: bool = True) -> dict:
    """Fetch/refresh every US benchmark; returns {name: frame or None}. A source that
    fails on a cold cache is logged and skipped (returns None) rather than aborting
    the others — the daily job retries it next run, and the viz attach is guarded."""
    out = {}
    for name, fn in LOADERS.items():
        try:
            out[name] = fn(refresh=refresh)
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("benchmark %s unavailable this run: %s", name, exc)
            out[name] = None
    return out


def _main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch/inspect US TP benchmarks")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--which", default="all", choices=["all", *LOADERS])
    args = ap.parse_args()
    names = list(LOADERS) if args.which == "all" else [args.which]
    for name in names:
        df = LOADERS[name](refresh=args.refresh)
        last = df[df["maturity"] == df["maturity"].max()].dropna().tail(1)
        print(f"{name}: {len(df)} rows {df['date'].min().date()}..{df['date'].max().date()} "
              f"maturities {sorted(df['maturity'].unique())}")
        if len(last):
            print("  latest:", {k: round(float(v), 4) for k, v in last.iloc[0].items()
                                 if k not in ("date",)})


if __name__ == "__main__":
    _main()
