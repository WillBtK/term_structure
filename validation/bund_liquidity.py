"""Price-based German Bund liquidity proxy — the Hu-Pan-Wang (2013) "noise" measure
applied to the cross-section of individual listed Federal securities.

The Bund analogue of the US TIPS/UST noise gauge (`validation/tips_liquidity.py`):
per trading day, fit a smooth (fixed-lambda Nelson-Siegel) curve to the yields of the
individual outstanding Bunds/Bobls/Schätze and take the RMS fitting residual. When
bond-specific liquidity is scarce, individual issues scatter further from any smooth
curve and the "noise" rises — a free daily illiquidity gauge from public data.

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

logger = logging.getLogger(__name__)

_NS_LAMBDA = 0.5           # fixed Nelson-Siegel decay (Bund curve spans ~0-30y)
_MIN_RESID_MAT = 1.0      # drop near-maturity issues (yields noisy at the very short end)
_MAX_RESID_MAT = 30.0
_MIN_BONDS = 6            # need > n_params(=3) for a meaningful RMS


def _ns_design(tau: np.ndarray, lam: float = _NS_LAMBDA) -> np.ndarray:
    """Nelson-Siegel (Diebold-Li) regressors [1, slope, curvature] at tenors ``tau``
    for a fixed decay ``lam`` — so the curve fit is ordinary least squares."""
    lt = lam * np.asarray(tau, dtype=float)
    lt = np.where(lt == 0.0, 1e-9, lt)
    s = (1.0 - np.exp(-lt)) / lt
    c = s - np.exp(-lt)
    return np.column_stack([np.ones_like(lt), s, c])


def _rms_residual(tau: np.ndarray, y: np.ndarray) -> float:
    """RMS residual of a fixed-λ Nelson-Siegel OLS fit to (tau, y) — the cross-sectional
    'noise'. NaN if too few bonds."""
    tau = np.asarray(tau, dtype=float)
    y = np.asarray(y, dtype=float)
    if tau.size < _MIN_BONDS:
        return np.nan
    X = _ns_design(tau)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sqrt(np.mean(resid ** 2)))


def noise_from_panel(panel: pd.DataFrame,
                     min_mat: float = _MIN_RESID_MAT, max_mat: float = _MAX_RESID_MAT,
                     min_bonds: int = _MIN_BONDS) -> pd.DataFrame:
    """Compute the daily 'noise' series from a per-security panel.

    ``panel`` columns: ``date`` (datetime), ``maturity`` (datetime; the security's
    redemption date), ``yield`` (DECIMAL). Returns (date, noise, n_bonds) in decimal,
    one row per trading day with enough bonds. Independent of the fetch format so it
    is unit-testable offline.
    """
    df = panel.dropna(subset=["yield"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["maturity"] = pd.to_datetime(df["maturity"])
    recs = []
    for dt, g in df.groupby("date"):
        resid_mat = (g["maturity"] - dt).dt.days.to_numpy() / 365.25
        keep = (resid_mat >= min_mat) & (resid_mat <= max_mat)
        tau = resid_mat[keep]
        y = g["yield"].to_numpy()[keep]
        if tau.size < min_bonds:
            continue
        recs.append({"date": dt, "noise": _rms_residual(tau, y), "n_bonds": int(tau.size)})
    if not recs:
        return pd.DataFrame(columns=["date", "noise", "n_bonds"])
    return pd.DataFrame(recs).dropna().sort_values("date").reset_index(drop=True)


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
    raw = load_bund_yields(refresh=refresh)
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
