"""Daily build of the global term-premium factor for the Atlas + exports.

Reads the canonical TP panel (`viz/data_daily.parquet`), fits the 1-factor DFM
(`model.global_factor`, warm-started from the cached params for speed/stability), and
writes a long-form `viz/global_factor.parquet` (the global factor + per-market fitted /
residual / observed) plus a `viz/global_factor_meta.json` (loadings, variance shares,
the PCA ladder). Month-end resolution — the factor is a slow common cycle; keeps the
Atlas payload small while a double-ended slider still zooms the full history.

Inserted into `scheduler/daily_update.py::rebuild_viz()` after `viz.export_daily`. The
inline EM fit is cheap (~seconds) and warm-started, so no separate slow-cadence step is
needed; it never raises (guarded) so a cold run can't block the daily job.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model import global_factor as gf

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
OUT_PARQUET = HERE / "global_factor.parquet"
OUT_META = HERE / "global_factor_meta.json"
PARAMS = HERE / "params" / "global_factor.json"
TENOR = 10.0
# Markets in the global factor (Switzerland excluded — negligible market liquidity).
# The actual order/set comes from model.global_factor.load_tp_panel (GF_ORDER); this
# mirror is kept only for reference. The panel is month-end and backfilled (DE→1972,
# JP 9y→10y→1974), clipped to ≥3 markets (≥1974).
ORDER = ["US", "DE", "JP", "UK", "CA"]


def _pca_ladder(Y: np.ndarray) -> list:
    """PCA variance-explained ladder (%) on the balanced standardized panel."""
    bal = Y[np.all(np.isfinite(Y), axis=1)]
    if bal.shape[0] < 10:
        return []
    Z = (bal - bal.mean(0)) / bal.std(0)
    w = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    return list(np.round(w / w.sum() * 100, 1))


def _month_end(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.groupby([d.index.year, d.index.month]).tail(1)
    d.index = d.index.to_period("M").to_timestamp("M")
    return d


def build() -> pd.DataFrame:
    panel = gf.load_tp_panel(TENOR)
    Y = panel.to_numpy(float)
    names = list(panel.columns)
    overlap = np.all(np.isfinite(Y), axis=1)
    warm = gf.DFMParams.from_json(PARAMS) if PARAMS.exists() else None
    p, diag = gf.estimate_dfm(Y, names, overlap_mask=overlap, warm=warm)
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    p.to_json(PARAMS)                                          # warm-start next run
    dec = gf.filter_forward(p, Y, smoothed=True)              # smoothed factor for display
    G = pd.Series(dec["G"], index=panel.index)
    fitted = pd.DataFrame(dec["fitted"], index=panel.index, columns=names)
    resid = pd.DataFrame(dec["residual"], index=panel.index, columns=names)
    obs = panel

    # long-form parquet (decimal; factor is standardized/unitless)
    frames = [pd.DataFrame({"date": G.index, "country": "GLOBAL", "kind": "factor",
                            "tenor": TENOR, "value": G.to_numpy()})]
    for cc in names:
        for kind, src in (("fitted", fitted), ("residual", resid), ("observed_tp", obs)):
            frames.append(pd.DataFrame({"date": src.index, "country": cc, "kind": kind,
                                        "tenor": TENOR, "value": src[cc].to_numpy()}))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    out.to_parquet(OUT_PARQUET)

    meta = {"asof": str(panel.index.max().date()), "tenor": TENOR, "names": names,
            "phi": round(p.phi, 5), "pca_corr": round(diag["pca_corr"], 3),
            "loglik": round(diag["loglik"], 1),
            "balanced_start": str(panel.index[overlap].min().date()),
            "pca_ladder": _pca_ladder(Y),
            "markets": {cc: {"beta": round(float(p.beta[i]), 5),
                             "global_share": round(float(p.communality[i]) * 100, 1)}
                        for i, cc in enumerate(names)}}
    OUT_META.write_text(json.dumps(meta, indent=2))
    logger.info("global factor: %d rows, PC1 share %s, mean communality %.0f%% -> %s",
                len(out), meta["pca_ladder"][:1], np.mean(p.communality) * 100, OUT_PARQUET.name)
    return out


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        build()
    except Exception as exc:                                   # noqa: BLE001 — never block the daily job
        logger.warning("global_factor build skipped: %s", exc)


if __name__ == "__main__":
    _main()
