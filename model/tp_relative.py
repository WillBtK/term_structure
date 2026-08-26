"""Term premium as a US level plus a country spread.

Four independent driver models fit each market's LEVEL respectably and then explain
almost none of its CHANGE -- which is the only thing a scenario ever uses. Regressed on
twelve-month changes the standalone models return R2 0.046 for the UK, 0.062 for Germany
and 0.081 for Japan. They describe where a premium has been, not where it is going.

Putting the US in as the common level fixes that, because the US is the global rate
setter and the literature has said so for two decades. On twelve-month changes the US
term premium alone explains 0.57 of the UK, 0.49 of Germany and 0.30 of Japan, and
adding each market's drivers RELATIVE to the US takes those to 0.66, 0.56 and 0.35. On
levels the same structure lifts the UK from 0.62 to 0.84, Germany from 0.82 to 0.93 and
Japan from 0.82 to 0.91, roughly halving the RMSE in each.

    TP_i = a_i + b_i * TP_US + c_i * (driver_i - driver_US)

Two things to be careful about when selling this, neither of which is a reason not to
build it.

The first is that 93.8% of variance in PC1 is a LEVEL artefact. It is the shared collapse
in premia since the 1980s, and no ten-year window comes near it: rolling PC1 runs 0.51 in
1995, 0.90 through 2002-03, 0.58 in 2011, 0.85 in 2017, 0.60 in 2022 and 0.80 today,
averaging about 0.74 with no trend either way. On monthly CHANGES it is 0.53. So the
common factor is real and currently near the top of its range -- but it is a cyclical 60
to 90 percent, not a structural 94.

The second is that common DIRECTION is not common AMPLITUDE. The US betas have fallen
even as PC1 has cycled back up: on 2010+ levels the UK reads 0.65, Germany 0.54 and Japan
0.24, against 0.98, 0.89 and 0.78 over the full sample, and on twelve-month changes since
2010 Japan's beta is 0.12. Japan has very nearly decoupled, which is what yield-curve
control was for.

The structure wins by MORE on the recent sample (2010+ levels: UK 0.53 -> 0.82, Germany
0.35 -> 0.76, Japan 0.62 -> 0.83), and it wins because the relative drivers capture that
divergence in amplitude. A committee should be told the US sets the direction and each
market's own QE and fiscal path decide how far it travels -- not that the four markets
have become one.

Which is why each market is now fitted on its OWN policy regime (``REGIME_START``) and
not on everything available. Germany used to start in 1991-04 and returned a beta of
1.296 -- above one, so a US premium move was AMPLIFIED into the Bund -- while scoring
R2 = -0.245 on 2010+, worse than predicting the mean. On the euro sample the beta is
0.964 and Japan, cut at 2000 to match its production re-estimation, falls from 0.739 to
0.482 with recent fit improving from 0.294 to 0.564. The shipped betas are therefore
UK 0.888, Germany 0.964, Japan 0.482.

The fiscal channel is the same object in every market: the FIVE-YEAR-AHEAD OFFICIAL
projected deficit (CBO for the US, OBR / AMECO / Cabinet Office for the spreads --
``macro.fiscal_projections``), entering the spread as the relative (own minus US)
projection. No spread market has the vintage archive that identified the US coefficient,
so all three borrow it -- +0.52pp of premium per point of projected deficit -- as an
adjustment measured from today's relative reading: worth nothing at the base case, alive
in any scenario that moves one side's deficit path against the other's. Relative debt
stock, which used to fill this slot, is gone: the flow-vs-stock evidence says the
projected balance is the priced object, and running both would double-count one story.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from model.tp_drivers import (CORE, DRIVERS, MARKETS, RESIDUAL_PERSISTENCE, SIGNS,
                              build_panel, design, fit_market)

logger = logging.getLogger(__name__)

SPREAD_MARKETS = tuple(c for c in MARKETS if c != "US")

# The US beta is bounded below by zero -- a market whose premium moves AGAINST the US
# level is not a spread market -- and above by 1.5, which no sample comes near.
BETA_BOUNDS = (0.0, 1.5)
MIN_T = 2.0
MIN_OBS = 60          # below this a regime-trimmed sample is not worth the trade

# --- Where each market's relative fit is allowed to start ------------------------
# The rest of this repo estimates every market on ONE monetary-policy regime, so that
# no structural break sits inside the sample (README, "Modern production TP vs
# historical exhibit"). The relative model was not held to that rule, and Germany paid
# for it: its pair began in 1991-04, which put reunification, the EMS crisis and the
# convergence trade inside the fit, and -- worse -- 76 of those months are SPLICED
# historical-exhibit premia carried in with a +2.06pp level offset at the 1997-08 seam,
# not production estimates at all.
#
# The damage was measurable rather than theoretical. Fitted on the full sample the
# German model scored R2 = -0.245 on 2010+: worse than predicting the mean, against
# 0.732 for the UK on the same test. Beta came out at 1.296 -- above one, so a US
# premium move was amplified on its way into the Bund -- and that single number was
# most of what drove the German premium to zero in a scenario where the US premium
# collapsed.
#
# Two further reasons the full-sample number was too high, both of which survive this
# fix and belong in any discussion of it: the fit is on LEVELS, where beta falls from
# 1.24 to 0.94 to 0.54 as the sample shortens while the twelve-month-CHANGES beta
# barely moves (0.64 / 0.67 / 0.61) -- the signature of a regression loading on the
# shared secular decline in premia rather than on co-movement; and where the sign
# constraint kills a relative channel, its work is absorbed into beta, which is why
# the constrained fit (1.296) sits above simple OLS (1.24).
REGIME_START = {
    "DE": "1999-01-01",   # the euro. Production premia begin 1997-08; before that the
                          # series is the spliced historical exhibit.
    "JP": "2000-01-01",   # matches the production re-cut on the regime-clean 2000-05+
                          # sample (validation/FINDINGS_japan.md)
    "UK": None,           # the pair already starts 2004-10; nothing to trim
}


@dataclass(frozen=True)
class RelativeModel:
    """One market's premium as a beta to the US plus relative drivers."""
    country: str
    intercept: float
    beta_us: float
    coef: dict                       # relative-driver coefficients, FITTED
    tstat: dict = field(default_factory=dict)
    # Channels this market's own history cannot identify. Each is applied to the
    # relative driver measured FROM TODAY, so it contributes nothing at the base case --
    # the historical fit stays honest -- and still responds when a scenario moves it.
    adjust: dict = field(default_factory=dict)
    base: dict = field(default_factory=dict)
    borrowed: tuple = ()             # channels whose coefficient is taken from the US
    r2: float = float("nan")
    r2_recent: float = float("nan")
    rmse: float = float("nan")
    n: int = 0

    def premium(self, us: dict, own: dict, tp_us: float) -> float:
        """This market's modelled premium, given the US premium and both driver sets."""
        rel = _relative(own, us)
        for k in self.adjust:                       # e.g. fiscal, outside DRIVERS
            if k not in rel:                        # missing on either side -> at base,
                rel[k] = (float(own[k]) - float(us[k])   # contributing nothing
                          if k in own and k in us else self.base[k])
        return (self.intercept + self.beta_us * float(tp_us)
                + sum(self.coef.get(k, 0.0) * rel[k] for k in DRIVERS)
                + sum(g * (rel[k] - self.base[k]) for k, g in self.adjust.items()))

    def scenario(self, us_today: dict, us_scenario: dict, own_today: dict,
                 own_scenario: dict, tp_us_today: float, tp_us_scenario: float,
                 tp_today: float, horizon: float = 0.0,
                 rho: float = RESIDUAL_PERSISTENCE,
                 view_weight: float = 1.0) -> float:
        """Anchored on today's observed premium, exactly as the US model is.

        The model supplies the level from the drivers and today supplies what they
        cannot explain, with that residual decaying at ``rho`` per year rather than
        persisting for ever -- see ``model.tp_drivers.RESIDUAL_PERSISTENCE`` for why,
        and for the evidence that Germany and Japan prefer no decay at all while the US
        and UK clearly prefer some.

        At the zero horizon this returns today's premium exactly for any ``rho``
        PROVIDED ``view_weight`` is the engine's ``1 - lambda**h``, which is zero
        there. Unweighted it returns today's premium plus the whole move in fair
        value, because the drivers at h=0 already read the user's full path; the
        engine used to paper over that by writing ``tp[0] = tp_today`` at the call
        site instead of calling this.

        ``view_weight`` phases the scenario's move in fair value in at the same rate
        the expectations leg phases the rate view in -- see
        ``model.tp_drivers.DriverModel.scenario`` for why one view must price at one
        speed.

        The move has to be SPLIT before it is weighted. The premium is linear in its
        parts, so

            premium(s) - premium(t) = beta * (tp_us_s - tp_us_t)      <- US channel
                                    + [own relative drivers moving]   <- own channel

        and the US premium arriving in ``tp_us_scenario`` has already been blended by
        the US model. Weighting the whole move would damp that channel a second time,
        which would make every spread market converge more slowly than the US for no
        reason. So the US channel passes through at full weight and only this market's
        own relative drivers carry ``view_weight``. Evaluating both premia at the same
        ``tp_us_today`` is what isolates the own channel.

        ``view_weight=1.0`` reproduces the unblended result exactly.
        """
        residual = float(tp_today) - self.premium(us_today, own_today, tp_us_today)
        w = float(view_weight)
        us_move = self.beta_us * (float(tp_us_scenario) - float(tp_us_today))
        own_move = (self.premium(us_scenario, own_scenario, tp_us_today)
                    - self.premium(us_today, own_today, tp_us_today))
        return (float(tp_today) + us_move + own_move * w
                + residual * (rho ** float(horizon) - 1.0))


def _relative(own: dict, us: dict) -> dict:
    """Driver differences, with inflation kinked at the target on each side first."""
    from model.tp_drivers import TARGET_KINK
    out = {}
    for k in DRIVERS:
        a, b = float(own[k]), float(us[k])
        if k == "expected_inflation":
            a, b = max(a - TARGET_KINK, 0.0), max(b - TARGET_KINK, 0.0)
        out[k] = a - b
    return out


def build_pair(country: str, panels: dict | None = None) -> pd.DataFrame:
    """One market joined to the US, with the relative drivers already differenced."""
    panels = panels or {}
    own = panels.get(country)
    own = build_panel(country) if own is None else own
    us = panels.get("US")
    us = build_panel("US") if us is None else us
    j = own.join(us[["tp"] + list(DRIVERS)].add_prefix("us_"), how="inner")
    x, xu = design(j), design(j[[f"us_{k}" for k in DRIVERS]]
                              .rename(columns={f"us_{k}": k for k in DRIVERS}))
    for k in DRIVERS:
        j[f"rel_{k}"] = x[k] - xu[k]
    return j


def fit_relative(country: str, pair: pd.DataFrame | None = None,
                 us_model=None) -> RelativeModel:
    """Sign-constrained fit of one market's spread to the US.

    A channel is FITTED only where this market's own history identifies it. Where the
    sign constraint pins a core channel at zero -- Germany's relative inflation, say --
    forcing a number into the historical fit is what wrecked the earlier attempt, so
    those channels are carried as ADJUSTMENTS measured from today's relative driver:
    worth nothing at the base case, which leaves the fit reporting what the data
    actually supports, and still alive when a scenario moves them. The borrowed
    coefficient is the US's own, because a relative driver's coefficient IS the
    response to a move in this market's driver with the US held still.

    The fiscal channel is ALWAYS an adjustment, never fitted: the driver is the
    five-year-ahead official projected deficit relative to the US, no spread market
    has the vintage archive that would let its own history identify a coefficient,
    and the borrowed US sensitivity (+0.52 per point) is applied from today's
    relative reading. Relative debt stock no longer enters anywhere -- the projected
    balance is the priced fiscal object, and carrying both would double-count.
    """
    p = build_pair(country) if pair is None else pair
    start = REGIME_START.get(country)
    if start is not None:
        before = len(p)
        p = p[p.index >= start]
        if len(p) < MIN_OBS:
            logger.warning("%s: regime start %s leaves only %d observations; "
                           "falling back to the full sample", country, start, len(p))
            p = build_pair(country) if pair is None else pair
        elif len(p) < before:
            logger.info("%s: relative fit on %s+ (%d of %d months; the rest predates "
                        "the regime)", country, start[:4], len(p), before)
    us_model = us_model or fit_market("US")
    y = p["tp"].values
    base = {k: float(p[f"rel_{k}"].iloc[-1]) for k in CORE}

    coef, tstat, b_us, a = _solve(p, list(CORE), y)
    dead = [k for k in CORE if abs(coef[k]) < 1e-9 or tstat[k] < MIN_T]
    if dead:                                    # refit on what is left, cleanly
        live = [k for k in CORE if k not in dead]
        coef, tstat, b_us, a = _solve(p, live, y)

    adjust, borrowed = {}, []
    for k in dead:
        adjust[k] = us_model.coef[k]
        borrowed.append(k)
    fiscal = _relative_fiscal_base(country)
    if fiscal is not None and "fiscal" in us_model.adjust:
        adjust["fiscal"] = us_model.adjust["fiscal"]
        base["fiscal"] = fiscal
        borrowed.append("fiscal")

    r = y - _hat(p, coef, b_us, a)
    recent = np.asarray(p.index >= "2010-01-01")
    return RelativeModel(
        country=country, intercept=a, beta_us=b_us, coef=coef, tstat=tstat,
        adjust=adjust, base=base, borrowed=tuple(borrowed),
        r2=float(1 - (r @ r) / ((y - y.mean()) ** 2).sum()),
        r2_recent=float(1 - (r[recent] ** 2).sum()
                        / ((y[recent] - y[recent].mean()) ** 2).sum()),
        rmse=float(np.sqrt((r ** 2).mean())), n=len(p))


def _relative_fiscal_base(country: str) -> float | None:
    """Today's relative (own minus US) five-year-ahead projected deficit, from the
    committed official projections. None -- and no fiscal channel -- when a side is
    unavailable, so an offline fit degrades rather than fails."""
    try:
        from macro.fiscal_projections import five_year_ahead
        return float(five_year_ahead(country) - five_year_ahead("US"))
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("%s: no fiscal projection (%s); spread runs without the "
                       "fiscal channel", country, exc)
        return None


def _solve(p: pd.DataFrame, keys: list, y: np.ndarray) -> tuple:
    A = np.column_stack([np.ones(len(p)), p["us_tp"].values]
                        + [p[f"rel_{k}"].values for k in keys])
    lo = np.array([-np.inf, BETA_BOUNDS[0]]
                  + [0.0 if SIGNS[k] > 0 else -np.inf for k in keys])
    hi = np.array([np.inf, BETA_BOUNDS[1]]
                  + [np.inf if SIGNS[k] > 0 else 0.0 for k in keys])
    b = lsq_linear(A, y, bounds=(lo, hi)).x
    r = y - A @ b
    se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * (r @ r) / max(len(y) - A.shape[1], 1))
    coef = {k: float(b[i + 2]) for i, k in enumerate(keys)}
    tstat = {k: float(abs(b[i + 2] / se[i + 2])) if se[i + 2] > 0 else 0.0
             for i, k in enumerate(keys)}
    return coef, tstat, float(b[1]), float(b[0])


def _hat(p: pd.DataFrame, coef: dict, beta: float, a: float) -> np.ndarray:
    """The FITTED premium. Adjustments are deliberately absent: they are measured from
    today, so over history they would impose swings that never happened -- 2.5bp against
    Japan's hundred-point climb in relative debt is 2.5pp of premium the market never
    charged. They contribute nothing at today's values and so change nothing here, and
    everything in a scenario, which is exactly where they belong."""
    out = a + beta * p["us_tp"].values
    for k, g in coef.items():
        out = out + g * p[f"rel_{k}"].values
    return out


def fit_all_relative(markets: tuple = SPREAD_MARKETS) -> dict:
    us_model = fit_market("US")
    panels = {c: build_panel(c) for c in ("US",) + tuple(markets)}
    return {c: fit_relative(c, build_pair(c, panels), us_model) for c in markets}


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    us_model = fit_market("US")
    print(f"US, the common level: R2 {us_model.r2:.3f}  post-2000 "
          f"{us_model.r2_post2000:+.3f}  RMSE {us_model.rmse:.2f}pp  (unchanged)\n")
    panels = {c: build_panel(c) for c in MARKETS}
    for c in SPREAD_MARKETS:
        p = build_pair(c, panels)
        m = fit_relative(c, p, us_model)
        print(f"{c} = a + b*US + relative drivers,  n={m.n}, "
              f"{p.index[0].date()}..{p.index[-1].date()}")
        print(f"  R2 {m.r2:.3f}   2010+ {m.r2_recent:+.3f}   RMSE {m.rmse:.2f}pp")
        print(f"  {'US term premium':24} {m.beta_us:+.3f}")
        for k in m.coef:
            print(f"  {'rel ' + k:24} {m.coef[k]:+.4f}  (t={m.tstat[k]:5.1f})")
        for k, g in m.adjust.items():
            print(f"  {'rel ' + k:24} {g:+.4f}  from US, applied from today"
                  + (f"  (base {m.base[k]:+.2f})" if k == "fiscal" else ""))
        print()


if __name__ == "__main__":
    _main()
