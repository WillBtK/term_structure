"""Daily build of the global cycle factors for the Atlas + exports.

Reads the canonical panel (`viz/data_daily.parquet`), fits a 1-factor DFM
(`model.global_factor`) for each **combination of measure and curve point**, and writes a
long-form `viz/global_factor.parquet` (the common factor G_t plus each market's observed
/ cycle-implied fitted / local residual) with a `viz/global_factor_meta.json` carrying the
loadings, variance shares and PCA ladder per combination.

Combinations (see COMBOS): three measures — actual yield, expected rate, term premium —
each at 5y / 10y / 20y / 30y and at the **5y5y forward**. The forward is derived from the
two reported zero legs (`model.forwards`); because that transform is linear the three
measures stay additive there, so a forward term premium needs no re-estimation.

Two coverage caveats, both honest rather than papered over:
  * **30y excludes the UK** — the BoE grid we ingest tops out at 20y (its 30y point only
    begins 2016), so the 30y factor is a four-market cycle. 20y is the well-identified
    long point: all five markets, balanced from 1997.
  * **30y has no historical backfill** — the DE/JP overlays reach 10y, so 5y/10y/5y5y
    keep their 1970s history while 20y/30y start when the modern curves do.

Month-end estimation and display throughout. This is a modelling choice, not a size
compromise (delivery is Supabase + Cloudflare, so page weight is a soft constraint):
  * the common factor is a slow cycle, while daily term-premium changes are dominated by
    curve-fitting and measurement noise — monthly averages that away;
  * on daily data the factor's AR(1) coefficient sits at ~0.999, effectively a unit root
    and weakly identified; at month-end the persistence is far better conditioned;
  * the deep history is *natively monthly* anyway — the pre-1997 DE/JP backfills are
    monthly overlays, so daily resolution there was interpolation, not information;
  * monthly is the standard frequency for this class of common-factor model.
A double-ended slider still zooms the full history.

Inserted into `scheduler/daily_update.py::rebuild_viz()` after `viz.export_daily`. Each EM
fit is cheap and warm-started from the previous run; the whole step is guarded so a cold
or failing combo can never block the daily job.
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
FREQ = "M"                     # month-end (slow cycle; keeps the payload sane)
TENOR = 10.0                   # the default combo surfaced first in the Atlas
ORDER = ["US", "DE", "JP", "UK", "CA"]      # Switzerland excluded — negligible liquidity
_NO_UK = ["US", "DE", "JP", "CA"]           # 30y: the BoE grid we ingest stops at 20y

MEASURE_LABEL = {"observed": "Actual yield", "expected": "Expected rate",
                 "term_premium": "Term premium"}

# (measure, tenor, fwd_start, markets) -> one DFM fit each.
COMBOS = [(m, t, f, mk)
          for m in ("term_premium", "expected", "observed")
          for (t, f, mk) in ((5.0, 0.0, ORDER), (10.0, 0.0, ORDER), (20.0, 0.0, ORDER),
                             (30.0, 0.0, _NO_UK), (5.0, 5.0, ORDER))]


def combo_key(measure: str, tenor: float, fwd_start: float) -> str:
    """Stable id, e.g. ``term_premium@10y`` or ``expected@5y5y``."""
    from model.forwards import FORWARD_LABEL
    return f"{measure}@{FORWARD_LABEL(fwd_start, tenor)}"


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


def _fit_one(measure, tenor, fwd_start, markets, warm_all):
    """Fit one combination; returns (long-form rows DataFrame, meta dict) or (None, None)
    when the panel is too thin to identify a common factor."""
    key = combo_key(measure, tenor, fwd_start)
    panel = gf.load_measure_panel(measure, tenor=tenor, fwd_start=fwd_start,
                                  freq=FREQ, markets=markets)
    panel = panel.dropna(how="all")
    names = list(panel.columns)
    if panel.empty or len(names) < 3:
        logger.warning("global cycle %s: panel too thin (%d markets) -> skipped", key, len(names))
        return None, None
    Y = panel.to_numpy(float)
    overlap = np.all(np.isfinite(Y), axis=1)
    if overlap.sum() < 24:                       # need a real balanced window to anchor the fit
        logger.warning("global cycle %s: only %d balanced months -> skipped", key, int(overlap.sum()))
        return None, None
    warm = None
    if isinstance(warm_all, dict) and key in warm_all:
        try:
            warm = gf.DFMParams.from_dict(warm_all[key])
        except Exception:                        # noqa: BLE001
            warm = None
    p_, diag = gf.estimate_dfm(Y, names, overlap_mask=overlap, warm=warm)
    dec = gf.filter_forward(p_, Y, smoothed=True)
    G = pd.Series(dec["G"], index=panel.index)
    fitted = pd.DataFrame(dec["fitted"], index=panel.index, columns=names)
    resid = pd.DataFrame(dec["residual"], index=panel.index, columns=names)

    frames = [pd.DataFrame({"date": G.index, "combo": key, "country": "GLOBAL",
                            "kind": "factor", "value": G.to_numpy()})]
    for cc in names:
        for kind, src in (("fitted", fitted), ("residual", resid), ("observed", panel)):
            frames.append(pd.DataFrame({"date": src.index, "combo": key, "country": cc,
                                        "kind": kind, "value": src[cc].to_numpy()}))
    rows = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    from model.forwards import FORWARD_LABEL
    meta = {"measure": measure, "measure_label": MEASURE_LABEL[measure],
            "point": FORWARD_LABEL(fwd_start, tenor), "tenor": tenor,
            "fwd_start": fwd_start, "names": names,
            "phi": round(float(p_.phi), 5), "pca_corr": round(float(diag["pca_corr"]), 3),
            "balanced_start": str(panel.index[overlap].min().date()),
            "pca_ladder": _pca_ladder(Y),
            "markets": {cc: {"beta": round(float(p_.beta[i]), 5),
                             "global_share": round(float(p_.communality[i]) * 100, 1)}
                        for i, cc in enumerate(names)}}
    return rows, (meta, p_)


def build() -> pd.DataFrame:
    """Fit every (measure x curve point) combination and write the parquet + meta."""
    warm_all = {}
    if PARAMS.exists():
        try:
            warm_all = json.loads(PARAMS.read_text())
        except Exception:                                          # noqa: BLE001
            warm_all = {}
    all_rows, metas, params_out = [], {}, {}
    for measure, tenor, fwd_start, markets in COMBOS:
        key = combo_key(measure, tenor, fwd_start)
        try:
            rows, res = _fit_one(measure, tenor, fwd_start, markets, warm_all)
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("global cycle %s failed (%s) -> skipped", key, exc)
            continue
        if rows is None:
            continue
        meta, p_ = res
        all_rows.append(rows)
        metas[key] = meta
        params_out[key] = p_.to_dict()                             # warm-start next run
        logger.info("global cycle %-22s %d markets, PC1 %s%%, mean communality %.0f%%, from %s",
                    key, len(meta["names"]), meta["pca_ladder"][:1],
                    np.mean([m["global_share"] for m in meta["markets"].values()]),
                    meta["balanced_start"])
    if not all_rows:
        raise RuntimeError("no global-cycle combination could be fitted")
    out = pd.concat(all_rows, ignore_index=True)
    out.to_parquet(OUT_PARQUET)
    default = combo_key("term_premium", TENOR, 0.0)
    OUT_META.write_text(json.dumps(
        {"asof": str(pd.to_datetime(out["date"]).max().date()), "freq": FREQ,
         "default": default if default in metas else sorted(metas)[0],
         "measure_labels": MEASURE_LABEL, "combos": metas}, indent=2))
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(params_out, indent=2))
    logger.info("global cycle: %d combos, %d rows -> %s", len(metas), len(out), OUT_PARQUET.name)
    return out


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        build()
    except Exception as exc:                                   # noqa: BLE001 — never block the daily job
        logger.warning("global_factor build skipped: %s", exc)


if __name__ == "__main__":
    _main()
