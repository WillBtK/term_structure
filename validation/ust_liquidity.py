"""Price-based U.S. **nominal** Treasury liquidity proxy — the Hu-Pan-Wang (2013)
"noise" measure on the coupon-Treasury cross-section.

This is HPW's original object. Their measure is a nominal Treasury one; the two gauges
already in the repo are variations on it — `validation/tips_liquidity.py` restricts it
to TIPS (to identify the joint model's inflation-risk premium) and
`validation/bund_liquidity.py` ports it to German Federal securities. What was missing
was the US nominal benchmark itself, and the reason is that no free source publishes
per-security Treasury *yields*: TreasuryDirect's FedInvest file publishes end-of-day
**prices**. `model/bond_math.py` closes that gap — coupon schedule, actual/actual
accrual, and a price -> yield inversion on the semiannual bond-equivalent basis every
quoted Treasury yield uses.

Two residual definitions are computed each day:

``noise_gsw`` (headline) — RMS deviation from the **published GSW Svensson curve**:
    turn the parameters into a discount curve, price each bond off it, invert to a
    yield, and difference. This is the accurate one, and measurably so: on a synthetic
    cross-section priced exactly off a known curve — true noise zero — it reads 0.2 bp,
    because it fits nothing and so has no functional-form error of its own. It is also
    the stricter reading, since GSW deliberately exclude on-the-run and first-off-the-run
    issues from their fit, so a rich on-the-run shows up as a real (negative) residual
    instead of being absorbed.

``noise`` (comparability) — residual standard error of a Svensson fit to the observed
    yields, profiled over a small grid of decay pairs. Fitting the level ourselves
    absorbs any constant offset between FedInvest end-of-day prices and the quotes GSW
    was built from, so this is the column that lines up with the Bund and TIPS gauges.
    It is the weaker of the two by construction, and the synthetic panel says by how
    much: a single fixed decay (what those two gauges use) leaves 2-9 bp of pure
    functional-form misfit, varying with the shape of the curve, and the mixed-coupon
    cross-section adds ~2 bp more because yield to maturity is not a function of
    maturity alone. Profiling over a decay grid and carrying a coupon regressor brings
    it to ~0.6 bp. It is kept because GSW publish in weekly batches, so the newest days
    have no curve to reference.

Agreement between the two validates the bond math: a mistake in accrual or compounding
would move them apart.

"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from validation.noise_core import residual_se

logger = logging.getLogger(__name__)

_MIN_RESID_MAT = 1.0       # below 1y the yield is very sensitive to the price tick
_MAX_RESID_MAT = 30.0
_MIN_BONDS = 10            # need comfortably more bonds than the 6 fitted parameters

# The curve-free fit lives in validation/noise_core: profiled-decay Svensson with a
# coupon regressor and a degrees-of-freedom correction, shared with the Bund and TIPS
# gauges. Only the curve-referenced residual below is specific to this market.


def _rms(x: np.ndarray) -> float:
    """Plain RMS — correct for the curve-referenced measure, which fits nothing."""
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan


def gsw_model_yields(day: pd.DataFrame, params: pd.Series) -> np.ndarray:
    """Yield each bond in ``day`` would carry if it priced exactly off the GSW curve.

    ``params`` is one row of GSW Svensson parameters (BETA0-3 in percent, TAU1-2 in
    years). GSW zeros are continuously compounded, so the discount factor is
    ``exp(-y(t) t)``; the model clean price is then inverted back to a semiannual yield
    so model and observed are on the identical basis and the difference is meaningful.
    """
    from model.bond_math import (disc_from_zero, price_from_curve, ytm_from_price)
    from us.reconstruct import svensson_yield

    def zero(t):
        return svensson_yield(np.maximum(np.asarray(t, dtype=float), 1e-6),
                              params["BETA0"], params["BETA1"], params["BETA2"],
                              params["BETA3"], params["TAU1"], params["TAU2"]) / 100.0

    disc = disc_from_zero(zero)
    out = []
    for settle, mat, cpn in zip(day["date"], day["maturity"], day["coupon"]):
        p = price_from_curve(settle, mat, float(cpn), disc)
        out.append(ytm_from_price(p, settle, mat, float(cpn)))
    return np.asarray(out, dtype=float)


def noise_from_panel(panel: pd.DataFrame, gsw_params: pd.DataFrame | None = None,
                     min_mat: float = _MIN_RESID_MAT, max_mat: float = _MAX_RESID_MAT,
                     min_bonds: int = _MIN_BONDS) -> pd.DataFrame:
    """Daily noise from a per-security panel.

    ``panel`` columns: ``date``, ``maturity`` (datetimes), ``coupon`` (annual rate in
    PERCENT) and either ``yield`` (decimal) or ``price`` (clean, per 100 face) — a
    missing ``yield`` is computed from the price. ``gsw_params`` is the date-indexed
    Svensson frame; when given, the GSW cross-check column is added.

    Returns ``(date, noise, n_bonds[, noise_gsw])`` in DECIMAL.
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["maturity"] = pd.to_datetime(df["maturity"])
    if "yield" not in df.columns:
        from model.bond_math import yields_from_prices
        df = yields_from_prices(df)
    df = df.dropna(subset=["yield", "coupon"])

    recs = []
    for dt, g in df.groupby("date"):
        tau = (g["maturity"] - dt).dt.days.to_numpy() / 365.25
        keep = (tau >= min_mat) & (tau <= max_mat)
        if keep.sum() < min_bonds:
            continue
        g = g.loc[keep]
        tau, y = tau[keep], g["yield"].to_numpy()
        rec = {"date": dt,
               "noise": residual_se(tau, y, coupon=g["coupon"].to_numpy()),
               "n_bonds": int(keep.sum())}
        if gsw_params is not None and dt in gsw_params.index:
            resid = y - gsw_model_yields(g, gsw_params.loc[dt])
            good = np.isfinite(resid)
            # A curve that cannot price the day's bonds at all is a params problem, not
            # a liquidity reading — leave it NaN rather than report a partial basket.
            rec["noise_gsw"] = _rms(resid[good]) if good.sum() >= min_bonds else np.nan
        recs.append(rec)
    cols = ["date", "noise", "n_bonds"] + (["noise_gsw"] if gsw_params is not None else [])
    if not recs:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(recs).dropna(subset=["noise"])
            .sort_values("date").reset_index(drop=True)[cols])


def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["ym"] = d["date"].dt.to_period("M")
    agg = {"noise": ("noise", "mean"), "n_bonds": ("n_bonds", "last")}
    if "noise_gsw" in d.columns:
        agg["noise_gsw"] = ("noise_gsw", "mean")
    m = d.groupby("ym").agg(**agg).reset_index()
    m["date"] = m["ym"].dt.to_timestamp("M")
    return m[[c for c in ("date", "noise", "noise_gsw", "n_bonds") if c in m.columns]]


def _read_store() -> pd.DataFrame:
    from validation import config as cfg
    path = cfg.UST_NOISE_STORE
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("unreadable UST noise store %s: %s", path, exc)
    return pd.DataFrame(columns=["date", "noise", "noise_gsw", "n_bonds"])


def build_noise(refresh: bool = False, freq: str = "M",
                start: str | None = None, end: str | None = None,
                lookback_days: int = 20) -> pd.DataFrame:
    """The US nominal Treasury 'noise' proxy in DECIMAL, appended incrementally.

    The derived series is VERSIONED (``UST_NOISE_STORE``) while the raw price panel is
    a rebuildable cache: FedInvest serves any past date, so nothing is lost by
    discarding prices, but re-crawling a decade of them on every ephemeral CI run is
    not viable. So a normal run fetches only the days after the store's last date (plus
    a short ``lookback_days`` overlap, in case a recent day was published late) and
    appends. Pass an explicit ``start`` to back-fill a deeper window.
    """
    from validation import config as cfg
    store = _read_store()
    if not refresh and start is None and len(store):
        return store if freq == "D" else _to_monthly(store)

    if start is None and len(store):
        start = (pd.to_datetime(store["date"]).max()
                 - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    start = start or cfg.UST_PRICES_START

    from us.fetch_fedinvest import load_prices, nominal_coupon_bonds
    from us.fetch_gsw import load_gsw_params
    raw = nominal_coupon_bonds(load_prices(start=start, end=end))
    if len(raw):
        # load_prices returns the whole cache, not just the requested window. Bound it
        # to the window so a chunked back-fill run recomputes only its own days.
        dts = pd.to_datetime(raw["date"])
        keep = dts >= pd.Timestamp(start)
        if end is not None:
            keep &= dts <= pd.Timestamp(end)
        raw = raw[keep]
    try:
        gsw = load_gsw_params(refresh=False)
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("GSW params unavailable (%s); headline noise only", exc)
        gsw = None
    fresh = noise_from_panel(raw, gsw_params=gsw)

    daily = (pd.concat([store, fresh], ignore_index=True)
             .sort_values("date").drop_duplicates("date", keep="last")
             .reset_index(drop=True)) if len(fresh) else store
    if len(daily):
        for path in (cfg.UST_NOISE_STORE, cfg.UST_NOISE_PARQUET):
            path.parent.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(path)
        last = daily.iloc[-1]
        logger.info("UST noise: %d daily obs %s..%s (+%d new), latest %.1f bp (%d bonds)",
                    len(daily), pd.to_datetime(daily["date"]).min().date(),
                    pd.to_datetime(daily["date"]).max().date(), len(daily) - len(store),
                    last["noise"] * 1e4, last["n_bonds"])
        if "noise_gsw" in daily.columns and daily["noise_gsw"].notna().any():
            both = daily.dropna(subset=["noise_gsw"])
            logger.info("  vs GSW curve: latest %.1f bp, corr(self-fit, GSW) = %.3f",
                        both["noise_gsw"].iloc[-1] * 1e4,
                        both["noise"].corr(both["noise_gsw"]))
    return daily if freq == "D" else _to_monthly(daily)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse
    ap = argparse.ArgumentParser(description="US nominal Treasury liquidity 'noise'")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()
    m = build_noise(refresh=args.refresh or bool(args.start), freq="M",
                    start=args.start, end=args.end)
    if not len(m):
        print("no UST noise series (fetch produced no data)")
        return
    print(f"UST noise (monthly): {len(m)} obs {m['date'].min().date()}..{m['date'].max().date()}")
    show = m.tail(6).assign(bp=lambda d: (d["noise"] * 1e4).round(1))
    if "noise_gsw" in show.columns:
        show = show.assign(bp_gsw=lambda d: (d["noise_gsw"] * 1e4).round(1))
    print(show[[c for c in ("date", "bp", "bp_gsw", "n_bonds") if c in show.columns]]
          .to_string(index=False))


if __name__ == "__main__":
    _main()
