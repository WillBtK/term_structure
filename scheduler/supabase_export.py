"""Supabase-ready export of the daily term-premium dataset (+ benchmarks).

Writes upload-ready, long-form tables from the canonical daily dataset and the
cached US benchmarks, and — when Supabase credentials are configured — upserts them.
It is **inert without credentials**: it always writes the staging files and logs a
clear message, so it is safe to run daily now and "just works" once the user sets
``SUPABASE_URL`` / ``SUPABASE_KEY`` (and ``supabase.co`` is allowlisted).

Tables (idempotent upsert keys in parens):
  term_premia (country, instrument, date, tenor) — canonical decimal decomposition
  benchmarks  (source, date, maturity, metric)   — external US TP model estimates

Usage:
    python -m scheduler.supabase_export           # write staging files (+ upload if creds)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger("scheduler.supabase_export")
ROOT = Path(__file__).resolve().parent.parent
CANON = ROOT / "viz" / "data_daily.parquet"
STAGE = ROOT / "scheduler" / "supabase_stage"       # gitignored; transient upload files
TP_TABLE, BENCH_TABLE = "term_premia", "benchmarks"


def _benchmarks_long() -> pd.DataFrame:
    """All cached US benchmarks as tidy (source, date, maturity, metric, value)."""
    from validation import config as vcfg
    from validation import benchmarks as bm
    from validation import sf_fed
    specs = [
        ("acm", vcfg.ACM_PARQUET_PATH, lambda: bm.load_acm(refresh=False)),
        ("kim_wright", vcfg.KIM_WRIGHT_PARQUET_PATH, lambda: bm.load_kim_wright(refresh=False)),
        ("dkw", vcfg.DKW_PARQUET_PATH, lambda: bm.load_dkw(refresh=False)),
        ("cr", vcfg.SF_FED_PARQUET_PATH, lambda: sf_fed.load_sf_fed_series(refresh=False)),
    ]
    frames = []
    for name, cache, load in specs:
        if not cache.exists():
            continue
        try:
            df = load()
        except Exception:                                          # noqa: BLE001
            continue
        metrics = [c for c in df.columns if c not in ("date", "maturity")]
        long = df.melt(id_vars=["date", "maturity"], value_vars=metrics,
                       var_name="metric", value_name="value").dropna(subset=["value"])
        long.insert(0, "source", name)
        frames.append(long)
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["source", "date", "maturity", "metric", "value"]))


def build_exports() -> dict:
    """Write the staging CSVs; return {table: (path, n_rows)}."""
    STAGE.mkdir(parents=True, exist_ok=True)
    out = {}
    if CANON.exists():
        tp = pd.read_parquet(CANON)                                # already long, decimal
        p = STAGE / "term_premia.csv"; tp.to_csv(p, index=False)
        out[TP_TABLE] = (p, len(tp))
    else:
        logger.warning("canonical %s missing — run viz.export_daily first", CANON)
    bench = _benchmarks_long()
    p = STAGE / "benchmarks.csv"; bench.to_csv(p, index=False)
    out[BENCH_TABLE] = (p, len(bench))
    return out


def maybe_upload(exports: dict) -> bool:
    """Upsert the staged tables to Supabase iff SUPABASE_URL/KEY are set and the
    client is importable; otherwise log an inert message and return False."""
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        logger.info("SUPABASE_URL/KEY unset — wrote staging files only "
                    "(set creds + allowlist supabase.co to enable upload).")
        return False
    try:
        from supabase import create_client
    except ImportError:
        logger.warning("supabase client not installed (`pip install supabase`); "
                       "staging files written, upload skipped.")
        return False
    client = create_client(url, key)
    keys = {TP_TABLE: ["country", "instrument", "date", "tenor"],
            BENCH_TABLE: ["source", "date", "maturity", "metric"]}
    for table, (path, _) in exports.items():
        df = pd.read_csv(path)
        df["date"] = df["date"].astype(str)
        # chunked upsert (idempotent on the table's primary key)
        for i in range(0, len(df), 5000):
            client.table(table).upsert(
                df.iloc[i:i + 5000].to_dict("records"),
                on_conflict=",".join(keys[table])).execute()
        logger.info("upserted %d rows -> %s", len(df), table)
    return True


def run() -> dict:
    exports = build_exports()
    uploaded = maybe_upload(exports)
    return {"tables": {t: {"path": str(p), "rows": n} for t, (p, n) in exports.items()},
            "uploaded": uploaded}


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    argparse.ArgumentParser(description="Supabase-ready export of the daily dataset").parse_args()
    summary = run()
    print("\n=== supabase export ===")
    for t, info in summary["tables"].items():
        print(f"  {t}: {info['rows']:,} rows -> {info['path']}")
    print(f"uploaded to Supabase: {summary['uploaded']}")


if __name__ == "__main__":
    _main()
