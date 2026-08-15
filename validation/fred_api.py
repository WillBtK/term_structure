"""Shared FRED access — the official API (api.stlouisfed.org) when ``FRED_API_KEY``
is set, falling back to the keyless fredgraph CSV otherwise.

The keyless ``fredgraph.csv`` endpoint is flaky through some egress proxies (HTTP/2
resets, long timeouts); the official JSON API with a key is materially more reliable
and also exposes series/category metadata. Both paths retry with backoff. Used by the
Kim-Wright benchmark loader, the TIPS 'noise' liquidity gauge, and the shadow-rate
validation's fed-funds anchor.
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)

FRED_API = "https://api.stlouisfed.org/fred"
FREDGRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
_TIMEOUT = 40


def have_key() -> bool:
    return bool(os.environ.get("FRED_API_KEY"))


def _key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    return key


def fred_get(path: str, retries: int = 4, **params) -> dict:
    """GET a FRED API JSON endpoint (e.g. '/series/observations') with retry/backoff.
    Requires FRED_API_KEY."""
    params.update(api_key=_key(), file_type="json")
    url = FRED_API + path + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
                return json.load(r)
        except Exception as exc:                                   # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"FRED API failed for {path}: {last}")


def _fredgraph_csv(series_id: str, retries: int = 4) -> pd.Series:
    """Keyless fallback: parse fredgraph.csv via curl (more reliable than urllib
    through the egress proxy for this host)."""
    import subprocess
    last = None
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
                 "-A", "Mozilla/5.0", FREDGRAPH_CSV + series_id],
                capture_output=True, timeout=_TIMEOUT + 10)
            if out.returncode != 0:
                raise RuntimeError(f"curl {out.returncode}: {out.stderr[:120]!r}")
            df = pd.read_csv(io.BytesIO(out.stdout))
            df.columns = ["date", "val"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["val"] = pd.to_numeric(df["val"], errors="coerce")
            return df.dropna(subset=["date"]).set_index("date")["val"].dropna()
        except Exception as exc:                                   # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"fredgraph CSV failed for {series_id}: {last}")


def fred_series(series_id: str) -> pd.Series:
    """A FRED series as a float Series indexed by date (missing values dropped).
    Uses the official API when a key is present, else the keyless CSV fallback."""
    if have_key():
        js = fred_get("/series/observations", series_id=series_id)
        rows = [(pd.Timestamp(o["date"]), float(o["value"]))
                for o in js["observations"] if o["value"] not in (".", "", None)]
        if not rows:
            return pd.Series(dtype=float)
        idx, vals = zip(*rows)
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name=series_id)
    return _fredgraph_csv(series_id).rename(series_id)
