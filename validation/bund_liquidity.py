"""Price-based German Bund liquidity proxy — the Hu-Pan-Wang (2013) "noise" measure
applied to the cross-section of individual listed Federal securities.

The Bund analogue of the US gauges (`validation/ust_liquidity.py`,
`validation/tips_liquidity.py`): per trading day, fit a smooth curve to the yields of
the individual outstanding Bunds/Bobls/Schätze and take the residual standard error.
When bond-specific liquidity is scarce, individual issues scatter further from any
smooth curve and the "noise" rises — a free daily illiquidity gauge from public data.

The fit itself is `validation/noise_core.residual_se`, shared with the other two: a
Svensson form profiled over a grid of decays, with the coupon regressed out and a
degrees-of-freedom correction. That replaced a single-fixed-decay Nelson-Siegel with a
plain RMS, which — measured on cross-sections priced exactly off a known curve, where
the true noise is zero — was reporting several basis points of its own error as
illiquidity. See `noise_core` for the decomposition of what it was costing.

Data: Deutsche Bundesbank "Prices and yields of listed Federal securities" — daily
reference price + yield per ISIN, published as monthly workbooks (see
`germany.fetch_bund_prices`). This module is the estimation core and is independent
of the fetch format: `noise_from_panel` takes a tidy per-security panel and is unit
-tested offline; `build_noise` wires in the (CI-fetched) Bundesbank workbooks.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from validation.noise_core import residual_se

logger = logging.getLogger(__name__)

_MIN_RESID_MAT = 1.0      # drop near-maturity issues (yields noisy at the very short end)
_MAX_RESID_MAT = 30.0
_MIN_BONDS = 10           # need comfortably more bonds than the fitted parameters


def _ns_design(tau: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """Nelson-Siegel (Diebold-Li) regressors at a fixed decay.

    Retained only for the diagnostics in ``germany.fetch_bund_prices``; the gauge itself
    now uses the profiled-decay Svensson fit in ``validation.noise_core`` (see the
    module docstring for what the single-decay version was costing).
    """
    lt = lam * np.asarray(tau, dtype=float)
    lt = np.where(lt == 0.0, 1e-9, lt)
    s = (1.0 - np.exp(-lt)) / lt
    return np.column_stack([np.ones_like(lt), s, s - np.exp(-lt)])


def noise_from_panel(panel: pd.DataFrame,
                     min_mat: float = _MIN_RESID_MAT, max_mat: float = _MAX_RESID_MAT,
                     min_bonds: int = _MIN_BONDS) -> pd.DataFrame:
    """Compute the daily 'noise' series from a per-security panel.

    ``panel`` columns: ``date`` (datetime), ``maturity`` (datetime; the security's
    redemption date), ``yield`` (DECIMAL), and optionally ``coupon`` (regressed out
    when present for every bond that day — see ``noise_core``). Returns
    (date, noise, n_bonds) in decimal, one row per trading day with enough bonds.
    Independent of the fetch format so it is unit-testable offline.
    """
    df = panel.dropna(subset=["yield"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["maturity"] = pd.to_datetime(df["maturity"])
    has_coupon = "coupon" in df.columns and df["coupon"].notna().any()
    recs = []
    for dt, g in df.groupby("date"):
        resid_mat = (g["maturity"] - dt).dt.days.to_numpy() / 365.25
        keep = (resid_mat >= min_mat) & (resid_mat <= max_mat)
        tau = resid_mat[keep]
        y = g["yield"].to_numpy()[keep]
        if tau.size < min_bonds:
            continue
        cpn = g["coupon"].to_numpy()[keep] if has_coupon else None
        if cpn is not None and not np.isfinite(cpn).all():
            cpn = None                       # a partial coupon column is worse than none
        recs.append({"date": dt, "noise": residual_se(tau, y, coupon=cpn),
                     "n_bonds": int(tau.size)})
    if not recs:
        return pd.DataFrame(columns=["date", "noise", "n_bonds"])
    return pd.DataFrame(recs).dropna().sort_values("date").reset_index(drop=True)


def drop_inflation_linked(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only conventional NOMINAL Federal securities for the noise fit.

    The workbook mixes nominal Bunds/Bobls/Schätze with a handful of inflation-linked
    issues. Linkers quote REAL yields that sit percent below the nominal curve, so a
    single cross-sectional fit sees them as ~200 bp residuals and the 'noise' balloons
    to ~40 bp — an artefact, not illiquidity. Dropping them brings the gauge to the
    single-digit bp expected for the highly liquid Bund curve.

    The description is spelt differently in the two published layouts — the current one
    writes ``DBRI 0.1 04/15/46 I/L``, the older one ``Bund 15 index.`` — so both markers
    are matched. A no-op when the panel carries no ``name`` column (older cache).
    """
    if "name" not in panel.columns:
        return panel
    nm = panel["name"].astype(str)
    is_linker = (nm.str.match(r"\s*DBRI\b", case=False)
                 | nm.str.contains(r"I/L", case=False)
                 | nm.str.contains(r"index", case=False)
                 | nm.str.contains(r"\binfl", case=False))
    return panel.loc[~is_linker].copy()


def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["ym"] = d["date"].dt.to_period("M")
    m = d.groupby("ym").agg(noise=("noise", "mean"), n_bonds=("n_bonds", "last")).reset_index()
    m["date"] = m["ym"].dt.to_timestamp("M")
    return m[["date", "noise", "n_bonds"]]


def build_noise(refresh: bool = False, freq: str = "M") -> pd.DataFrame:
    """The Bund 'noise' liquidity proxy in DECIMAL: per date, RMS deviation of individual
    Bund yields from a smooth fit. Returns (date, noise, n_bonds); ``freq='M'`` gives
    month-end, 'D' daily. Cached to parquet. Requires the Bundesbank per-security
    workbooks via ``germany.fetch_bund_prices`` (fetched in the daily CI job)."""
    from validation import config as cfg
    path = cfg.BUND_NOISE_PARQUET
    if not refresh and path.exists():
        out = pd.read_parquet(path)
        return out if freq == "D" else _to_monthly(out)
    from germany.fetch_bund_prices import load_bund_yields
    raw = drop_inflation_linked(load_bund_yields(refresh=refresh))
    daily = noise_from_panel(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(path)
    if len(daily):
        logger.info("Bund noise: %d daily obs %s..%s, latest %.1f bp (%d bonds)",
                    len(daily), daily["date"].min().date(), daily["date"].max().date(),
                    daily["noise"].iloc[-1] * 1e4, daily["n_bonds"].iloc[-1])
    return daily if freq == "D" else _to_monthly(daily)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse
    ap = argparse.ArgumentParser(description="German Bund liquidity 'noise' proxy")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    m = build_noise(refresh=args.refresh, freq="M")
    if not len(m):
        print("no Bund noise series (fetch produced no data)")
        return
    print(f"Bund noise (monthly): {len(m)} obs {m['date'].min().date()}..{m['date'].max().date()}")
    print((m.tail(3).assign(bp=lambda d: (d["noise"] * 1e4).round(1))
           [["date", "bp", "n_bonds"]].to_string(index=False)))


if __name__ == "__main__":
    _main()
