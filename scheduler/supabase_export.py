"""Supabase-ready export of the daily term-premium dataset (+ benchmarks).

Writes upload-ready, long-form tables from the canonical daily dataset and the
cached US benchmarks, and — when Supabase credentials are configured — upserts them.
It is **inert without credentials**: it always writes the staging files and logs a
clear message, so it is safe to run daily now and "just works" once the user sets
``SUPABASE_URL`` / ``SUPABASE_KEY`` (and ``supabase.co`` is allowlisted).

Tables (idempotent upsert keys in parens):
  term_premia (country, instrument, date, tenor) — canonical decimal decomposition
  benchmarks  (source, date, maturity, metric)   — external US TP model estimates

It also publishes the self-contained Atlas (``viz/artifact.html``) to a PUBLIC
Supabase Storage bucket (default ``atlas``, override with ``SUPABASE_ATLAS_BUCKET``)
as ``index.html`` — a stable, auto-updating public URL while the GitHub repo stays
private. This too is inert without credentials.

Usage:
    python -m scheduler.supabase_export           # write staging files (+ upload if creds)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger("scheduler.supabase_export")
ROOT = Path(__file__).resolve().parent.parent
CANON = ROOT / "viz" / "data_daily.parquet"
STAGE = ROOT / "scheduler" / "supabase_stage"       # gitignored; transient upload files
TP_TABLE, BENCH_TABLE, GF_TABLE = "term_premia", "benchmarks", "global_factor"
ATLAS_HTML = ROOT / "viz" / "artifact.html"
ATLAS_BUCKET = os.environ.get("SUPABASE_ATLAS_BUCKET") or "atlas"
ATLAS_OBJECT = "index.html"


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
    # TIPS curve-fitting "noise" (liquidity gauge) — a standalone daily product, not a
    # TP model. Curve-wide (no maturity), so maturity=0 sentinel; value in decimal.
    if vcfg.TIPS_NOISE_PARQUET.exists():
        try:
            from validation.tips_liquidity import build_noise
            nz = build_noise(refresh=False, freq="D")[["date", "noise"]].dropna()
            nz = nz.rename(columns={"noise": "value"})
            nz["source"] = "tips_noise"; nz["maturity"] = 0.0; nz["metric"] = "rmse"
            frames.append(nz[["source", "date", "maturity", "metric", "value"]])
        except Exception:                                          # noqa: BLE001
            pass
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
    gf = ROOT / "viz" / "global_factor.parquet"
    if gf.exists():
        g = pd.read_parquet(gf)                                    # long: date,country,kind,tenor,value
        p = STAGE / "global_factor.csv"; g.to_csv(p, index=False)
        out[GF_TABLE] = (p, len(g))
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
            BENCH_TABLE: ["source", "date", "maturity", "metric"],
            GF_TABLE: ["date", "country", "kind", "tenor"]}
    any_ok = False
    for table, (path, _) in exports.items():
        df = pd.read_csv(path)
        df["date"] = df["date"].astype(str)
        try:
            # chunked upsert (idempotent on the table's primary key)
            for i in range(0, len(df), 5000):
                client.table(table).upsert(
                    df.iloc[i:i + 5000].to_dict("records"),
                    on_conflict=",".join(keys[table])).execute()
        except Exception as exc:                                    # noqa: BLE001
            # A missing table is fine: the DB tables are optional (the Atlas is
            # published to Storage independently). Log and keep going.
            logger.warning("skipping table %r (create it in Supabase to enable): %s",
                           table, exc)
            continue
        any_ok = True
        logger.info("upserted %d rows -> %s", len(df), table)
    return any_ok


def maybe_upload_atlas() -> str | None:
    """Publish the self-contained Atlas to a PUBLIC Supabase Storage bucket and
    return its public URL. Inert (returns None) without creds / client / artifact."""
    import urllib.request
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        logger.info("SUPABASE_URL/KEY unset — Atlas not published to Storage.")
        return None
    if not ATLAS_HTML.exists():
        logger.warning("Atlas %s missing — run the daily build first; skipping publish.", ATLAS_HTML)
        return None
    # The repo's artifact.html is "content-only" (no <!doctype>/<html>/<head>/<body>) —
    # exactly what the claude.ai Artifact tool expects, since it wraps the file itself at
    # publish time. Served raw, a browser falls into quirks mode and has no charset hint
    # (this page uses λ, φ, ≈, — heavily). Wrap it in a minimal standards-mode skeleton so
    # the standalone Supabase page renders IDENTICALLY to the claude.ai Artifact.
    raw = ATLAS_HTML.read_text(encoding="utf-8")
    # A per-build marker guarantees the bytes (and thus the etag) differ every upload.
    # Supabase's CDN caches keyed on the etag; identical bytes across re-uploads keep it
    # serving the ORIGINAL cached response (whose Content-Type was text/plain from the
    # first-ever storage-client upload), so a stable etag would never pick up text/html.
    marker = (f"<!-- atlas build {os.environ.get('GITHUB_SHA', 'local')[:8]}"
              f" run {os.environ.get('GITHUB_RUN_ID', '0')} -->\n")
    doc = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           + raw + '\n</html>\n' + marker)
    data = doc.encode("utf-8")
    base = url.rstrip("/")
    endpoint = f"{base}/storage/v1/object/{ATLAS_BUCKET}/{ATLAS_OBJECT}"
    auth = {"Authorization": f"Bearer {key}", "apikey": key}
    # Delete first so we CREATE fresh (Supabase keeps an object's stored mimetype across
    # upsert-updates). DELETE via raw REST (works reliably).
    try:
        with urllib.request.urlopen(
                urllib.request.Request(endpoint, method="DELETE", headers=auth), timeout=30) as r:
            logger.info("pre-delete of %s -> HTTP %s", ATLAS_OBJECT, r.status)
    except Exception as exc:                                        # noqa: BLE001
        logger.info("pre-delete of %s skipped (%s) — will create fresh", ATLAS_OBJECT, exc)
    # Upload via the OFFICIAL storage client: it populates the object's metadata.mimetype,
    # which is the field Supabase's serving layer reads for the response Content-Type. A raw
    # POST only sets the top-level content_type column (serving ignores it → text/plain, so
    # the browser shows source). Use a BARE "text/html" (a "; charset" param can defeat the
    # mime match). The <meta charset> in the doc handles encoding.
    try:
        from supabase import create_client
        client = create_client(url, key)
        client.storage.from_(ATLAS_BUCKET).upload(
            path=ATLAS_OBJECT, file=data,
            file_options={"content-type": "text/html", "cache-control": "300", "upsert": "true"})
    except Exception as exc:                                        # noqa: BLE001
        logger.error("Atlas upload to bucket %r failed (does the PUBLIC bucket exist?): %s",
                     ATLAS_BUCKET, exc)
        return None
    public_url = f"{base}/storage/v1/object/public/{ATLAS_BUCKET}/{ATLAS_OBJECT}"
    logger.info("published Atlas (%d KB) -> %s", len(data) // 1024, public_url)
    # Verify the ORIGIN mimetype (authenticated info endpoint, bypasses the public CDN,
    # which caches the Content-Type for up to cache-control seconds after a re-upload —
    # so an immediate public HEAD can read a stale value).
    try:
        info = urllib.request.Request(
            f"{base}/storage/v1/object/info/{ATLAS_BUCKET}/{ATLAS_OBJECT}", headers=auth)
        with urllib.request.urlopen(info, timeout=30) as r:
            logger.info("origin info JSON = %s", r.read().decode("utf-8", "replace")[:400])
    except Exception as exc:                                        # noqa: BLE001
        logger.info("origin mimetype verify skipped: %s", exc)
    # Full GET (no Range → a real 200, like a browser) reads the true served Content-Type.
    try:
        with urllib.request.urlopen(urllib.request.Request(public_url), timeout=60) as r:
            logger.info("served (full GET) Content-Type = %s", r.headers.get("Content-Type"))
    except Exception as exc:                                        # noqa: BLE001
        logger.info("Content-Type verify skipped: %s", exc)
    # DIAGNOSTIC: upload a brand-new (never-CDN-cached) probe path and full-GET it. If the
    # probe serves text/html but index.html serves text/plain, the problem is a stale CDN
    # edge cache on index.html (not Supabase serving html wrong). Probe is deleted after.
    try:
        from supabase import create_client as _cc
        probe = f"probe-{os.environ.get('GITHUB_RUN_ID', 'x')}.html"
        _cc(url, key).storage.from_(ATLAS_BUCKET).upload(
            path=probe, file=b"<!doctype html><meta charset=utf-8><title>probe</title><h1>probe</h1>",
            file_options={"content-type": "text/html", "upsert": "true"})
        with urllib.request.urlopen(
                f"{base}/storage/v1/object/public/{ATLAS_BUCKET}/{probe}", timeout=30) as r:
            logger.info("PROBE fresh-path served Content-Type = %s", r.headers.get("Content-Type"))
        urllib.request.urlopen(urllib.request.Request(
            f"{base}/storage/v1/object/{ATLAS_BUCKET}/{probe}", method="DELETE", headers=auth), timeout=30)
    except Exception as exc:                                        # noqa: BLE001
        logger.info("probe skipped: %s", exc)
    return public_url


def run() -> dict:
    exports = build_exports()
    uploaded = maybe_upload(exports)
    atlas_url = maybe_upload_atlas()
    return {"tables": {t: {"path": str(p), "rows": n} for t, (p, n) in exports.items()},
            "uploaded": uploaded, "atlas_url": atlas_url}


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
