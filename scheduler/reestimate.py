"""Slow re-estimation driver — the WEEKLY/MONTHLY cadence (NOT the daily job).

Re-estimates each market/instrument's state-space parameters and rewrites the cached
`viz/params/*.json` that the daily filter-forward path reuses. Per the README
cadence design: warm-start from the existing parameter set where supported, and
**fall back to the last valid params on failure** (never overwrite a good JSON with a
broken fit). Estimation is nonlinear and convergence-fragile, so this is run
deliberately (weekly, manually or on a slow cron) and is kept OFF the daily job.

Usage:
    python -m scheduler.reestimate                 # re-estimate all, warm-started
    python -m scheduler.reestimate --only us_prod  # a single spec
    python -m scheduler.reestimate --refresh       # refresh source caches first
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("scheduler.reestimate")
ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "viz" / "params"


# params file -> (module path, function name). Each returns the run_production_tp
# dict whose ["prod"] is a serializable ProductionParams.
SPECS = {
    "us_prod":    ("us.production_tp", "run_production_tp"),
    "us_real":    ("us.real_tp", "run_real_tp"),
    "de_prod":    ("germany.production_tp", "run_germany_production"),
    "japan_prod": ("japan.production_tp", "run_japan_production"),
    "uk_nom":     ("uk.production_tp", "run_uk_nominal"),
    "uk_real":    ("uk.production_tp", "run_uk_real"),
    "ch_prod":    ("switzerland.production_tp", "run_switzerland_production"),
}


def _reestimate_one(name: str) -> str:
    import importlib
    from us.production_tp import ProductionParams
    mod_name, fn_name = SPECS[name]
    fn = getattr(importlib.import_module(mod_name), fn_name)
    out_path = PARAMS / f"{name}.json"
    # warm-start from the existing params where the wrapper accepts it.
    warm = None
    if out_path.exists():
        try:
            warm = ProductionParams.from_json(out_path).params
        except Exception:                                          # noqa: BLE001
            warm = None
    try:
        try:
            res = fn(warm_start=warm) if warm is not None else fn()
        except TypeError:                                          # wrapper has no warm_start kw
            res = fn()
        prod = res["prod"]
        # sanity: don't overwrite with a degenerate fit
        import numpy as np
        if not np.all(np.isfinite(prod.params.K)):
            raise ValueError("non-finite K in re-estimated params")
        prod.to_json(out_path)
        return "ok"
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("re-estimate FAILED for %s: %s — keeping existing %s",
                       name, exc, out_path.name)
        return f"error (kept cached): {exc}"


def run(only: str | None = None, refresh: bool = False) -> dict:
    if refresh:
        from scheduler.daily_update import refresh_sources
        refresh_sources()
    names = [only] if only else list(SPECS)
    status = {}
    for name in names:
        logger.info("re-estimating %s ...", name)
        status[name] = _reestimate_one(name)
    return status


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Weekly re-estimation of viz/params/*.json")
    ap.add_argument("--only", choices=list(SPECS), default=None)
    ap.add_argument("--refresh", action="store_true", help="refresh source caches first")
    args = ap.parse_args()
    status = run(only=args.only, refresh=args.refresh)
    print("\n=== re-estimation summary ===")
    for k, v in status.items():
        print(f"  {'OK ' if v == 'ok' else 'ERR'} {k}" + ("" if v == "ok" else f"  ({v})"))
    print("Re-run scheduler.daily_update (or the daily job) to propagate into the Atlas.")


if __name__ == "__main__":
    _main()
