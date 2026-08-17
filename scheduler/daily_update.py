"""Daily update driver — the FAST path (no re-estimation).

Runs the operational daily cadence from the top-level README: refresh every source
cache (rate curves + US benchmarks), then rebuild the canonical daily dataset, the
artifact payload, and the artifact HTML using the *committed* cached parameters
(`viz/params/*.json`). Re-estimation is deliberately NOT here — it is slow and
convergence-fragile, and lives in `scheduler/reestimate.py` on a weekly cadence.

Each source refresh is fault-tolerant: a fetch failure falls back to that source's
cache (the loaders handle this) and is recorded in the run summary rather than
aborting the run. Idempotent; safe to re-run.

Usage:
    python -m scheduler.daily_update            # refresh + rebuild
    python -m scheduler.daily_update --no-fetch # rebuild from existing caches only
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("scheduler.daily_update")
ROOT = Path(__file__).resolve().parent.parent


# (label, callable) — each refreshes one source cache with refresh=True.
def _refreshers():
    import numpy as np
    from us.fetch_gsw import load_gsw_params
    from us.tips import load_tips_params
    from germany.fetch_bundesbank import load_params as de_load
    from japan.fetch_mof import load_yields as jp_load
    from uk.fetch_boe import load_curve as uk_load
    from switzerland.fetch_snb import load_yields as ch_load
    from canada.fetch_boc import load_yields as ca_load
    from validation import benchmarks as bench
    from validation.sf_fed import load_sf_fed_series
    from validation.tips_liquidity import build_noise
    from validation.bund_liquidity import build_noise as build_bund_noise
    return [
        ("US GSW (nominal)", lambda: load_gsw_params(refresh=True)),
        ("US TIPS (real)", lambda: load_tips_params(refresh=True)),
        ("Germany (daily)", lambda: de_load(freq="D", refresh=True)),
        ("Japan (daily)", lambda: jp_load(refresh=True)),   # MoF JGB is daily-only (no freq arg)
        ("UK nominal (daily)", lambda: uk_load("nominal", refresh=True, freq="D")),
        ("UK real (daily)", lambda: uk_load("real", refresh=True, freq="D")),
        ("Switzerland (daily)", lambda: ch_load(refresh=True, freq="D")),
        ("Canada (daily)", lambda: ca_load(refresh=True, freq="D")),   # BoC ZC curve (daily-only)
        ("Benchmark: C&R (SF Fed)", lambda: load_sf_fed_series(refresh=True)),
        ("Benchmarks: ACM/KW/DKW", lambda: bench.refresh_all_us_benchmarks(refresh=True)),
        # Standalone daily product (not just a model input): the TIPS curve-fitting
        # "noise" / liquidity gauge — a free substitute for the vendor RMSE series.
        # Needs FRED_API_KEY; guarded like every other source (cache fallback).
        ("TIPS noise (liquidity gauge)", lambda: build_noise(refresh=True, freq="D")),
        # German Bund analogue: RMS deviation of individual Bund yields from a smooth
        # curve (Bundesbank per-ISIN monthly workbooks). Guarded; cache fallback.
        ("Bund noise (liquidity gauge)", lambda: build_bund_noise(refresh=True, freq="D")),
    ]


def refresh_sources() -> dict:
    """Refresh every source cache; return {label: 'ok' | 'error: ...'}."""
    status = {}
    for label, fn in _refreshers():
        try:
            fn()
            status[label] = "ok"
            logger.info("refreshed %s", label)
        except Exception as exc:                                   # noqa: BLE001
            status[label] = f"error: {exc}"
            logger.warning("refresh FAILED for %s: %s (using cache if present)", label, exc)
    return status


def _run_step(mod: str, timeout: int = 900) -> None:
    """Run a viz build step as `python -m <mod>` (isolated; raises on failure)."""
    logger.info("running %s", mod)
    env = {"PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    import os
    r = subprocess.run([sys.executable, "-m", mod], cwd=str(ROOT),
                       env={**os.environ, **env}, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{mod} failed (rc={r.returncode}):\n{r.stderr[-1500:]}")


def rebuild_viz() -> None:
    """Regenerate canonical daily -> derived global factor -> thinned payload -> artifact
    HTML (fast path)."""
    _run_step("viz.export_daily")
    try:
        _run_step("viz.global_factor")       # derived from the canonical TP; never block
    except Exception as exc:                 # noqa: BLE001
        logger.warning("global_factor step skipped: %s", exc)
    _run_step("viz.thin_daily")
    _run_step("viz.build_artifact")


def run(fetch: bool = True) -> dict:
    t0 = time.time()
    status = refresh_sources() if fetch else {"(skipped fetch)": "ok"}
    rebuild_viz()
    dt = time.time() - t0
    ok = sum(v == "ok" for v in status.values())
    logger.info("daily update done in %.0fs — %d/%d sources ok", dt, ok, len(status))
    return {"elapsed_s": round(dt, 1), "sources": status}


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Daily term-premium update (fast path)")
    ap.add_argument("--no-fetch", action="store_true", help="rebuild from existing caches only")
    args = ap.parse_args()
    summary = run(fetch=not args.no_fetch)
    print("\n=== daily update summary ===")
    for k, v in summary["sources"].items():
        print(f"  {'OK ' if v == 'ok' else 'ERR'} {k}" + ("" if v == "ok" else f"  ({v})"))
    print(f"elapsed {summary['elapsed_s']}s")


if __name__ == "__main__":
    _main()
