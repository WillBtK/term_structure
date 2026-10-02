"""From a macro forecast to a ten-year yield and a bond return.

One instrument -- the ten-year bond -- in four markets, two horizons: the yield in
three years' time and in ten. The user supplies, per country, annual projections of
the policy rate, headline inflation and real growth. Growth is informational only: it
is assumed to have landed on the policy-rate forecast already, which is treated as
exogenous. Inflation is headline, on the stated basis that beyond base effects
headline equals core.

The chain, every link anchored to today:

  0. WHAT THE FORECAST IS BUILT TO DO. A year-end forecast should move with the house's
     calls and not with the market's noise, without ignoring a repricing. Every choice
     below was made on a real-time backtest against no change, 1992-2026
     (validation/FINDINGS_yield_forecast_backtest.md): today's yield and premium are
     common to every scenario, so the spreads between scenarios move only when their
     calls do.

  1. TERM PREMIUM.  US: the driver model's fair value on the SCENARIO's drivers, plus
     the part of today's premium those drivers cannot explain, fading at
     ``RESIDUAL_FADE_US`` (0.5) a year -- leaning against that residual cut 4-5 quarter
     error by 7-9% in real time. Spread markets: today's premium, plus the US premium's
     move at ``FORECAST_BETA`` (the fitted beta, except Japan at 0.15, which is then the
     whole US effect), plus their own drivers relative to the US, with the premium's
     response to an inflation call scaled to what yields have shown
     (``INFLATION_RESPONSE``). Their own unexplained premium is held: leaning against it
     made their forecasts worse. Two drivers come straight from the scenario: long-run
     expected inflation is the average of the projected inflation path over the bond's
     life, and inflation
     uncertainty is the fitted EWMA volatility propagated through the scenario months
     and measured AGAINST A STATUS-QUO BASELINE -- the same history extended with
     inflation simply continuing at its latest reading. Without the baseline, any
     smooth path mechanically decays the EWMA and a "no change" scenario quietly
     compresses the premium by 40bp over a decade: a hidden view. Differencing
     against the baseline removes it, so uncertainty moves only insofar as the
     scenario's path departs from where inflation already is -- swings raise it,
     a smooth return to target after a spike genuinely lowers it, and extending
     the status quo changes nothing. Ownership holds at today unless the scenario
     overrides it; debt/GDP defaults to the committed WEO/national-source projection
     (``macro.debt_gdp``), also overridable.

  2. YIELD.  y(h) = TP(h) + Pbar(h), Pbar(h) the expected average short rate over the
     ten years from h:

         Pbar(h) = curve + [view(h) - curve] * (1 - lam ** h)

     ``curve`` is the ten-year average today's market prices (y0 - TP0), ``view(h)`` the
     scenario's policy path averaged over [h, h+10] and put on the curve's own short-rate
     basis (``POLICY_BASIS``), and ``lam`` the speed at which a view of this kind is
     priced (``convergence_for``: the house forecast slowest). There is NO drift of the
     market's own: the curve's priced drift did not materialise on average, and adding it
     made forecasts worse. At h = 0, y(0) reproduces today's yield exactly.

  3. RETURN.  A constant-maturity ten-year position held H years with the yield
     moving linearly from y(0) to y(H) earns, annualised (Leibowitz-Bova-Kogelman),

         R = y0 + (dy/H) * ((H+1)/2 - D)

     with D the Macaulay duration of a ten-year par bond at today's yield. Carry
     accrues at each year's ending yield, which gives their convergence result: at
     H = 2D-1 the return equals the starting yield regardless of where yields go.
     The inflation-adjusted return deflates geometrically by the average projected
     inflation over the holding period.

Everything here is deliberately annual-granularity and approximately right: the
sources of error in a three-to-ten-year macro projection dwarf anything a monthly
bootstrap would add.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model.tp_drivers import DRIVERS, DriverModel
from model.tp_relative import RelativeModel

logger = logging.getLogger(__name__)

TENOR = 10.0
HORIZONS = (3, 10)
# Share of a rate view still UNEXPRESSED after a year -- the expectations-leg twin of
# RESIDUAL_PERSISTENCE. 0.8 puts about half a stated view into the three-year horizon
# and nine tenths into the ten-year one. See the module docstring for why it is not 0.
#
# Unlike RESIDUAL_PERSISTENCE this is NOT fitted, because convergence to a forecast with
# no track record cannot be measured directly. Two proxies were run
# (validation/view_convergence.py, FINDINGS_view_convergence.md): the SPF long-run bill
# forecast disagrees with the curve so rarely -- sd 0.21pp, |gap| > 25bp in one year of
# thirty-five -- that it identifies nothing; and a perfect-foresight error-correction
# test bounds lambda at 0.86-0.92, biased toward persistence and resting on about four
# independent windows. 0.8 sits just inside that bound on the faster-convergence side.
# It is a reasoned setting in a wide band, which is why the sheet exposes it as a dial.
VIEW_CONVERGENCE = 0.8

# --- How fast the market learns a state it is living through ---------------------
# The weight on your path is 1 - lam**h, and what it measures is NOT how persuasive
# your view is. A scenario other than the baseline is a conditional statement -- "if
# the world is stagflationary, how do bonds price?" -- so there is no disagreement to
# decay. What there is, is learning: the state has changed, but the market has no
# foresight and prices it as it unfolds. That is a measurable object, and
# validation/growth_drawdowns.py measures it, fitting 1 - lam**h to the market's own
# ten-year expected-rate leg through episodes where a state did materialise.
#
# It differs by regime, which one constant cannot express: an unambiguous state is
# learned fast, an inflationary one slowly, because the central bank cannot ease into
# it and the path stays genuinely murky. A scenario is classified by its OWN mean
# inflation over the projection grid rather than by name, so the rule travels to
# scenarios nobody has written yet.
#
# The baseline gets the same treatment as every other state. Under a learning reading
# there is only one process: the market's prior is today's curve -- which is why h=0
# still reproduces today's yield, since 1 - lam**0 = 0 for any lam -- and lam is how
# fast information arrives, not whose opinion it is.
#
# The BASELINE is the exception, and it is a real one rather than a convenience. A
# conditional scenario has a forcing event: the state happens, and the market cannot
# not see it, which is why the fitted speeds apply. The baseline is our FORECAST, made
# on the same information set the market already has. Nothing arrives to force
# agreement -- the market has seen what we have seen and concluded something else --
# and there is a genuine chance we are simply wrong. So it converges on the slower
# disagreement rate, and a scenario carries `is_view` to say which it is.
LEARNING = {"low": 0.40, "high": 0.685,   # fitted; see FINDINGS_growth_drawdowns.md
            "view": VIEW_CONVERGENCE}     # our own forecast: disagreement, not learning
LEARNING_INFL_SPLIT = 4.0                 # % mean inflation dividing the two regimes

# --- The year-end construction's parameters (validation/FINDINGS_yield_forecast_backtest.md).
# The scenario sheet reads these from its payload (scenarios.build_sheet), so the
# browser and this module cannot drift apart.
# Annual survival of the part of today's US premium the drivers cannot explain.
RESIDUAL_FADE_US = 0.5
# House policy rate -> the government curve's own short rate (validation/policy_basis.py,
# validation/survey_basis.py). US: the house rate is the fed funds LOWER bound.
POLICY_BASIS = {"US": 0.277, "UK": -0.185, "DE": -0.214, "JP": -0.102}
# Share of a US premium move reaching a spread market's forecast, where it differs from
# the fitted level beta (validation/backtest_japan_beta.py).
FORECAST_BETA = {"JP": 0.15}
# Realised / modelled premium response to an inflation call
# (validation/backtest_inflation_channel.py).
INFLATION_RESPONSE = {"US": 1.0, "UK": 0.15, "DE": 0.75, "JP": 0.0}


def forecast_params() -> dict:
    """What the scenario sheet needs to run the same construction."""
    return {"rho_us": RESIDUAL_FADE_US, "policy_basis": dict(POLICY_BASIS),
            "forecast_beta": dict(FORECAST_BETA),
            "inflation_response": dict(INFLATION_RESPONSE)}


def convergence_for(scenario, learning: dict | None = None,
                    split: float = LEARNING_INFL_SPLIT) -> float:
    """The convergence rate this scenario implies.

    A scenario flagged ``is_view`` is our forecast and gets the slower disagreement
    rate; anything else is a conditional state and gets the fitted learning speed for
    its own inflation regime.
    """
    lam = dict(LEARNING if learning is None else learning)
    if getattr(scenario, "is_view", False):
        return lam["view"]
    path = getattr(scenario, "inflation", None) or {}
    if not path:
        return lam["low"]
    mean_pi = sum(float(v) for v in path.values()) / len(path)
    return lam["high"] if mean_pi >= split else lam["low"]

REQUIRED_COLUMNS = ("year", "policy_rate", "inflation", "real_growth")


@dataclass(frozen=True)
class Scenario:
    """One market's annual projections, all in percent, keyed by projection year
    (1 = the first full year ahead). ``real_growth`` is carried but never used in the
    arithmetic -- it is the story behind the policy path, not a second input to it.

    ``deficit`` (deficit-positive, % of GDP) drives the fiscal channel in every
    market: the US directly, the spread markets through the RELATIVE (own minus US)
    five-year-ahead projected deficit. Entering nothing runs each market's official
    projection (CBO / OBR / AMECO / Cabinet Office, ``macro.fiscal_projections``).
    One constraint worth restating from the evidence: the fiscal effect is a
    REAL-rate effect -- the inflation-risk premium does not respond -- so widening
    the deficit is not by itself a reason to also raise the inflation path."""
    country: str
    policy_rate: dict
    inflation: dict
    real_growth: dict = field(default_factory=dict)
    price_insensitive: dict = field(default_factory=dict)   # optional override, % GDP
    debt_gdp: dict = field(default_factory=dict)            # optional override, % GDP
    deficit: dict = field(default_factory=dict)             # optional override, % GDP
    # The deficit the premium PRICES: ``deficit`` with the cycle taken out
    # (macro.fiscal_rules.structural_deficit). Set by ``run``; empty = price ``deficit``.
    deficit_priced: dict = field(default_factory=dict)
    # True for the house forecast, False for a conditional "if the world is X" state.
    # It selects the convergence rate and nothing else -- see convergence_for.
    is_view: bool = False
    # True when the paths come from CALENDAR years (the sheet / a calendar CSV): the
    # inflation path is then keyed at each year's START (a calendar-year average governs
    # its own year) and the uncertainty driver reads it month by calendar month.
    calendar: bool = False

    def __post_init__(self):
        for name in ("policy_rate", "inflation"):
            path = getattr(self, name)
            if not path:
                raise ValueError(f"{self.country}: scenario needs a {name} path")
            years = sorted(path)
            # a calendar inflation path is keyed at year STARTS, so the current year
            # sits at a negative horizon (it began before today); it must still end ahead
            floor = -1.0 if (self.calendar and name == "inflation") else 0.0
            if years[0] <= floor:
                raise ValueError(f"{self.country}: {name} years must be positive "
                                 f"horizons ahead of today, got {years}")
            if len(set(years)) != len(years):
                raise ValueError(f"{self.country}: {name} has duplicate years")


@dataclass(frozen=True)
class MarketState:
    """Today, as the data stands: the anchor for every projection."""
    country: str
    y_today: float                    # observed 10y yield, %
    tp_today: float                   # estimated 10y term premium, %
    drivers_today: dict               # the four driver values the panel ends on
    core_history: pd.Series           # monthly YoY core inflation, %, for the EWMA
    # The market's OWN expected average short rate by tenor, {tenor_years: percent}
    # -- the expectations leg of the published decomposition, y(tau) - TP(tau). It is
    # what lets a projection start from the market's forward view rather than from a
    # flat reading of today's ten-year average.
    expected_curve: dict = field(default_factory=dict)


def step(path: dict, year: float) -> float:
    """The path value in a given projection year: annual steps, held flat beyond the
    last projection and anchored before year 1 at the year-1 value."""
    years = sorted(path)
    if year <= years[0]:
        return float(path[years[0]])
    if year >= years[-1]:
        return float(path[years[-1]])
    # the entry in force at this time: the last one dated at or before it. Written
    # this way rather than by flooring so the path may be keyed by fractional
    # horizons -- which is what calendar year-ends become.
    prev = years[0]
    for y in years:
        if y > year:
            break
        prev = y
    return float(path[prev])


def forward_average(path: dict, start: int, span: int = int(TENOR)) -> float:
    """Average of the path over ``span`` years beginning after year ``start`` -- the
    expectations leg of a bond bought at horizon ``start``."""
    return float(np.mean([step(path, start + k) for k in range(1, span + 1)]))


def path_average(path: dict, start: float, span: float = TENOR, step_y: float = 0.25) -> float:
    """Average of a path over [start, start+span], quarterly -- what a bond bought at
    ``start`` lives through. ``forward_average`` samples one point a year from start+1,
    which skips the first year: right for the official deficit's annual keys, wrong for
    a rate or inflation path held from day one. Mirrors the sheet's ``avgPath``."""
    xs = np.arange(start + step_y / 2, start + span, step_y)
    return float(np.mean([step(path, float(x)) for x in xs]))


def market_forward_average(curve: dict, start: float, span: float = TENOR) -> float:
    """The MARKET's expected average short rate over ``span`` years beginning at
    ``start``, read off the published expected-rate curve.

    ``curve`` maps tenor to the expected average short rate over that tenor, so
    ``C(t) = t * curve(t)`` is the cumulative expected short rate and the average over
    a forward window is ``(C(start+span) - C(start)) / span``. Tenors between grid
    points are linearly interpolated; beyond the longest tenor the curve is held flat,
    which is the honest reading of a curve that has run out of data.

    This is the anchor the projection starts from at every horizon, and it is NOT the
    same number at each one: the US curve prices an average of 3.11% over the next ten
    years but only 2.82% over the ten years starting in three, because it expects
    policy to fall. Anchoring every horizon on the spot ten-year average would smuggle
    that decline in as a view the user never expressed.
    """
    if not curve:
        return float("nan")
    keys = sorted(curve, key=float)
    ts = [float(t) for t in keys]
    vs = [float(curve[k]) for k in keys]

    def cum(t: float) -> float:
        """Cumulative expected short rate to t. By definition C(t) = t * e(t), so
        interpolating the AVERAGE curve and multiplying is both exact at the nodes and
        smooth between them -- interpolating C directly would instead make the implied
        one-year forwards piecewise-constant, which shows up as repeated values in a
        projected policy path. Beyond the longest tenor the average is held flat."""
        if t <= 0.0:
            return 0.0
        return t * float(np.interp(t, ts, vs))      # np.interp clamps outside the grid

    return (cum(start + span) - cum(start)) / span


def policy_view(state: MarketState, scenario: Scenario, horizon: float) -> float:
    """The scenario's policy path averaged over the ten years from ``horizon``, on the
    curve's own short-rate basis."""
    return path_average(scenario.policy_rate, horizon) + POLICY_BASIS.get(state.country, 0.0)


def rate_leg(state: MarketState, scenario: Scenario, horizon: float,
             lam: float = VIEW_CONVERGENCE) -> float:
    """The expectations leg at a horizon: today's priced ten-year average moving toward
    the scenario's, with no drift of the market's own.

        Pbar(h) = curve + [view(h) - curve] * (1 - lam ** h)

    ``lam = 0`` prices the whole view at once; ``lam = 1`` holds today's priced level.
    """
    curve = state.y_today - state.tp_today
    w = 1.0 - float(lam) ** max(float(horizon), 0.0)
    return float(curve + (policy_view(state, scenario, horizon) - curve) * w)


def _ewma_vol_at(hist: pd.Series, monthly_path: list, hl: int) -> float:
    future = pd.Series(monthly_path,
                       index=pd.date_range(hist.index[-1] + pd.DateOffset(months=1),
                                           periods=len(monthly_path), freq="MS"))
    s = pd.concat([hist, future])
    mean = s.ewm(halflife=hl).mean()
    return float(np.sqrt(((s - mean) ** 2).ewm(halflife=hl).mean()).iloc[-1])


def propagated_uncertainty(state: MarketState, scenario: Scenario, horizon: int,
                           halflife: int | None = None) -> float:
    """The EWMA inflation-volatility driver at the horizon, anchored to today.

    The realised core history is extended with the projected annual inflation --
    twelve identical monthly readings per projection year, since the scenario is a
    view about levels, not month-to-month noise. The raw EWMA of that spliced series
    would decay under ANY smooth path, so a no-change scenario would quietly
    compress the premium: a hidden view. It is therefore differenced against a
    status-quo baseline (inflation continuing at its latest reading) and applied to
    today's driver value:

        u(h) = u_today + [ vol(history + scenario, h) - vol(history + status quo, h) ]

    Extending the status quo changes nothing; swings add exactly the volatility they
    imply; a smooth return to target after a spike genuinely reduces it. Floored at
    zero, since a volatility is one.
    """
    from macro.fetch_core_cpi import UNCERTAINTY_HALFLIFE
    hl = halflife or UNCERTAINTY_HALFLIFE
    hist = state.core_history.dropna()
    # months to the horizon; a fractional (year-end) horizon takes its part-month, as
    # the sheet's loop does
    n = int(np.ceil(float(horizon) * 12 - 1e-9))
    if getattr(scenario, "calendar", False):
        # each month after the last core print takes its CALENDAR year's inflation:
        # dated at its midpoint, read off the year-start-keyed path (as the sheet does)
        last = pd.Timestamp(hist.index[-1])
        a0 = last.year + (last.month - 1) / 12.0
        now = pd.Timestamp.today()
        today_frac = now.year + (now.dayofyear - 1) / 365.25
        scen = [step(scenario.inflation, a0 + (m + 1) / 12.0 + 1 / 24.0 - today_frac)
                for m in range(n)]
    else:
        scen = [step(scenario.inflation, 1 + m // 12) for m in range(n)]
    base = [float(hist.iloc[-1])] * n
    delta = _ewma_vol_at(hist, scen, hl) - _ewma_vol_at(hist, base, hl)
    return max(state.drivers_today["inflation_uncertainty"] + delta, 0.0)


def drivers_at(state: MarketState, scenario: Scenario, horizon: int,
               debt_default: dict | None = None,
               deficit_default: dict | None = None) -> dict:
    """The driver values the premium models see at a horizon."""
    out = {
        "expected_inflation": path_average(scenario.inflation, horizon),
        "inflation_uncertainty": propagated_uncertainty(state, scenario, horizon),
        "price_insensitive": (step(scenario.price_insensitive, horizon)
                              if scenario.price_insensitive
                              else state.drivers_today["price_insensitive"]),
    }
    if scenario.debt_gdp:
        out["debt_gdp"] = step(scenario.debt_gdp, horizon)
    elif debt_default:
        out["debt_gdp"] = step(debt_default, horizon)
    else:
        out["debt_gdp"] = state.drivers_today["debt_gdp"]
    if "fiscal" in state.drivers_today:
        # A ten-year bond bought at horizon h prices the fiscal path over its WHOLE
        # life, so the driver moves with the ten-year forward average of the projected
        # deficit, not with whichever single calendar year sits h+5 ahead. Reading one
        # year swung US fair value 43bp purely on CBO's year-8 bump -- calendar
        # idiosyncrasy (expiring provisions and the like), not a change in trajectory.
        #
        # The LEVEL stays the estimation object -- today's five-year-ahead reading, the
        # statistic the coefficient was measured on -- and what moves it is the gap
        # between the scenario's trajectory average and the OFFICIAL baseline's. The
        # reference is deliberately the official path rather than the scenario's own
        # first year: entering a deficit two points wider than the baseline everywhere
        # is a real fiscal view and must register, while running the baseline unchanged
        # must contribute exactly nothing.
        path = scenario.deficit_priced or scenario.deficit or deficit_default
        ref = deficit_default or scenario.deficit
        today = state.drivers_today["fiscal"]
        out["fiscal"] = (today + forward_average(path, horizon) - forward_average(ref, 0)
                         if path else today)
    return out


def leibowitz_return(y0: float, yH: float, hold_years: int, duration: float) -> float:
    """Annualised nominal return, %, on a constant-maturity position (see module doc)."""
    dy = yH - y0
    return y0 + (dy / hold_years) * ((hold_years + 1) / 2.0 - duration)


def real_return(nominal_pct: float, scenario: Scenario, hold_years: int) -> float:
    """Geometrically deflated by the average projected inflation over the hold."""
    pi = np.mean([step(scenario.inflation, k) for k in range(1, hold_years + 1)])
    return ((1 + nominal_pct / 100.0) / (1 + pi / 100.0) - 1) * 100.0


@dataclass(frozen=True)
class ScenarioResult:
    country: str
    horizon: int                      # years ahead
    tp: float                         # projected 10y term premium, %
    policy_avg: float                 # 10y-forward average policy rate at horizon, %
    yield_10y: float                  # projected 10y yield, % == tp + policy_avg
    nominal_return: float             # annualised, buying today and holding `horizon`
    real_return: float
    duration: float
    curve_expected: float             # what today's curve prices: y(0) - TP(0), %
    policy_gap: float                 # your view today minus the curve's, pp
    # The expectations leg, opened up: where the market's own forward view sits at this
    # horizon, where the user's path sits, and how much of the difference is priced in.
    market_policy_avg: float = float("nan")
    view_policy_avg: float = float("nan")
    view_weight: float = float("nan")
    carried: float = float("nan")         # the unexplained premium still in tp, %


def run_market(state: MarketState, scenario: Scenario, tp_path: dict,
               lam: float = VIEW_CONVERGENCE, horizons=HORIZONS,
               carried: dict | None = None) -> list:
    """Assemble yields and returns from an already-computed premium path.

    ``tp_path`` maps horizon (0 included) to the premium in percent -- computed by the
    caller because the US and the spread markets use different models.
    """
    from model.bond_math import macaulay_duration
    today = pd.Timestamp.today().normalize()
    dur = macaulay_duration(today, today + pd.DateOffset(years=int(TENOR)),
                            coupon_pct=state.y_today, ytm=state.y_today / 100.0)
    curve = state.y_today - state.tp_today          # the market's expected average
    gap = policy_view(state, scenario, 0) - curve
    out = []
    for h in horizons:
        pbar = rate_leg(state, scenario, h, lam)
        y = tp_path[h] + pbar                       # the identity, not a difference
        hold = max(float(h), 1e-9)
        nom = leibowitz_return(state.y_today, y, hold, dur)
        out.append(ScenarioResult(
            country=state.country, horizon=h, tp=tp_path[h], policy_avg=pbar,
            yield_10y=y, nominal_return=nom,
            real_return=real_return(nom, scenario, max(int(round(hold)), 1)), duration=dur,
            curve_expected=curve, policy_gap=gap,
            market_policy_avg=market_forward_average(state.expected_curve, h),
            view_policy_avg=policy_view(state, scenario, h),
            view_weight=1.0 - float(lam) ** max(float(h), 0.0),
            carried=(carried or {}).get(h, float("nan"))))
    return out


def run(us_model: DriverModel, spread_models: dict, states: dict, scenarios: dict,
        debt_defaults: dict | None = None, deficit_defaults: dict | None = None,
        rho: float = RESIDUAL_FADE_US,
        view_convergence: float | None = None,
        growth_defaults: dict | None = None, horizons=HORIZONS) -> dict:
    """All markets, at ``horizons`` (years ahead; ``year_end_horizons`` for year-ends).
    The US premium path is computed first; every spread market follows it at its
    forecasting beta, plus its own drivers (module docstring, step 1).

    ``rho`` is the annual survival of the part of today's US premium the drivers cannot
    explain; it reaches the spread markets only through the US premium they follow.

    ``growth_defaults`` ({market: {years ahead: baseline real growth}}) switches on
    the endogenous deficit (``macro.fiscal_rules``): a scenario that states a growth
    view but no fiscal view gets its deficit path derived from the official
    projection via each market's semi-elasticity. An explicit ``deficit`` on the
    scenario switches the rule off for that market -- a stated fiscal view is the
    whole fiscal path (policy-driven growth is entered that way, never rule-plus-
    override). The premium reference stays the OFFICIAL path either way, and what
    the premium prices is the STRUCTURAL deficit -- the path with the cycle taken back
    out (``macro.fiscal_rules.structural_deficit``).
    """
    debt_defaults = debt_defaults or {}
    deficit_defaults = deficit_defaults or {}
    if growth_defaults:
        from dataclasses import replace as _replace

        from macro.fiscal_rules import (adjusted_deficit, extend_stated_deficit,
                                        structural_deficit)
        scenarios = dict(scenarios)
        for c, sc in scenarios.items():
            derived = (adjusted_deficit(c, sc.real_growth, growth_defaults[c],
                                        deficit_defaults[c])
                       if sc.real_growth and growth_defaults.get(c)
                       and deficit_defaults.get(c) else deficit_defaults.get(c))
            if not sc.deficit and derived and sc.real_growth and growth_defaults.get(c):
                sc = _replace(sc, deficit=derived)
            elif sc.deficit and derived:
                sc = _replace(sc, deficit=extend_stated_deficit(sc.deficit, derived))
            # Price the STRUCTURAL deficit, derived or stated alike: the coefficients
            # were estimated on projections that assume the gap has closed.
            if sc.deficit and sc.real_growth and growth_defaults.get(c):
                sc = _replace(sc, deficit_priced=structural_deficit(
                    c, sc.deficit, sc.real_growth, growth_defaults[c]))
            scenarios[c] = sc
    us_state, us_scen = states["US"], scenarios["US"]
    # Each market's own scenario sets its own learning speed, unless the caller pins
    # one (the sheet's dial, and every test that wants a fixed weight).
    lam_of = {c: (float(view_convergence) if view_convergence is not None
                  else convergence_for(sc)) for c, sc in scenarios.items()}
    hs = sorted(set(horizons) | {0})

    def drv(state, sc, c, h):
        return drivers_at(state, sc, h, debt_defaults.get(c), deficit_defaults.get(c))

    # US: fair value on the scenario's own drivers + today's unexplained part, fading.
    us_drv = {h: drv(us_state, us_scen, "US", h) for h in hs}
    us_resid = us_state.tp_today - us_model.premium(**us_drv[0])
    us_tp = {h: (us_state.tp_today if h <= 0
                 else us_model.premium(**us_drv[h]) + us_resid * rho ** float(h)) for h in hs}
    results = {"US": run_market(us_state, us_scen, us_tp, lam_of["US"], horizons,
                                {h: us_resid * rho ** max(float(h), 0.0) for h in hs})}

    def pick(a, b):
        out = dict(a)
        out.update({k: b[k] for k in ("expected_inflation", "inflation_uncertainty")})
        return out

    for c, m in spread_models.items():
        if c not in scenarios:
            continue
        st, sc = states[c], scenarios[c]
        beta = FORECAST_BETA.get(c, m.beta_us)
        s_inf = INFLATION_RESPONSE.get(c, 1.0)
        own0 = drv(st, sc, c, 0)
        tp0_us = us_tp[0]
        p0 = m.premium(us_drv[0], own0, tp0_us)
        tp = {0: st.tp_today}
        for h in hs:
            if h <= 0:
                continue
            own_h = drv(st, sc, c, h)
            # where a forecasting beta replaces the fitted one, the US reaches this
            # market only through it: relative drivers read against the US held at today
            us_ref = us_drv[0] if c in FORECAST_BETA else us_drv[h]
            own = m.premium(us_ref, own_h, tp0_us) - p0
            infl = m.premium(pick(us_drv[0], us_ref), pick(own0, own_h), tp0_us) - p0
            tp[h] = st.tp_today + beta * (us_tp[h] - tp0_us) + own - (1.0 - s_inf) * infl
        results[c] = run_market(st, sc, tp, lam_of[c], horizons,
                                {h: st.tp_today - p0 for h in hs})
    return results


def year_end_horizons(n: int = 10, today: "pd.Timestamp | None" = None) -> dict:
    """{calendar year: horizon in years} for this year's end and the next ``n - 1``."""
    now = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    frac = now.year + (now.dayofyear - 1) / 365.25
    return {y: round((y + 1) - frac, 4) for y in range(now.year, now.year + n)}


# ------------------------------------------------------------------ repo-data wiring

def load_scenario(path: "Path | str", cpi_ytd: "dict | None" = None) -> Scenario:
    """A scenario CSV: columns ``year, policy_rate, inflation, real_growth`` (percent,
    year 1 = first full year ahead), optional ``price_insensitive`` / ``debt_gdp`` /
    ``fiscal_balance`` (deficit negative, as the sheet displays and exports) or the
    older deficit-positive ``deficit``.

    In CALENDAR years, inflation and growth are calendar-year averages of y/y rates and
    the policy rate a year-end level. Inflation is keyed at each year's start; for the
    current year the published months are history and the rest is priced at the rate
    that delivers the stated average (``cpi_ytd``, from
    ``macro.fetch_headline_cpi.year_to_date``; looked up by file stem when omitted)."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df = df.dropna(subset=["year"]).astype({"year": int}).set_index("year")
    # The sheet exports CALENDAR years, which is what a reader wants to see. Convert
    # them to horizons ahead of today -- year-end Y is (Y + 1) - today. A file written
    # in the old years-ahead convention (small integers) is taken as-is.
    now = pd.Timestamp.today()
    today_frac = now.year + (now.dayofyear - 1) / 365.25
    calendar = df.index.min() > 1900

    def key(y: int) -> float:
        return round((int(y) + 1) - today_frac, 4) if calendar else float(y)

    def col(name):
        return {key(y): float(v) for y, v in df[name].dropna().items()} \
            if name in df else {}

    def inflation_col():
        if "inflation" not in df:
            return {}
        if not calendar:
            return col("inflation")
        from macro.fetch_headline_cpi import remaining_rate, year_to_date
        ytd = cpi_ytd
        if ytd is None:
            try:
                ytd = year_to_date(path.stem.upper()[:2], now.year)
            except Exception:                                       # noqa: BLE001
                ytd = {}
        out = {}
        for y, v in df["inflation"].dropna().items():
            v = float(v)
            if int(y) == (ytd or {}).get("year"):
                v = remaining_rate(v, ytd)
            out[round(int(y) - today_frac, 4)] = v       # the calendar year's START
        return out
    # Fiscal comes in on either sign convention: ``fiscal_balance`` (deficit
    # negative, what the sheet shows and exports) is negated to the internal
    # deficit-positive convention the coefficient was measured on.
    deficit = ({k: -v for k, v in col("fiscal_balance").items()}
               if "fiscal_balance" in df else col("deficit"))
    # ...but the deficit is keyed by CALENDAR-YEAR INDEX (0 = this year), the way the
    # official projections it is compared against are. Year-end keying put a stated
    # 2027 at horizon 1.35, so the pricing read 2026 at year 1 -- a year behind the
    # official path it is measured against.
    if calendar:
        deficit = {int(round(k - 1 + today_frac)) - now.year: v for k, v in deficit.items()}
    return Scenario(country=path.stem.upper(),
                    policy_rate=col("policy_rate"), inflation=inflation_col(),
                    real_growth=col("real_growth"),
                    price_insensitive=col("price_insensitive"),
                    debt_gdp=col("debt_gdp"), deficit=deficit, calendar=calendar)


def deficit_defaults(markets=("US", "UK", "DE", "JP")) -> dict:
    """Each market's official projected-deficit path as its fiscal default -- CBO,
    OBR, AMECO, Cabinet Office -- so entering nothing runs the official projection,
    visible in the sheet's grid rather than hidden."""
    from macro.fiscal_projections import path
    out = {}
    for c in markets:
        try:
            out[c] = path(c)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("no deficit path for %s (%s); fiscal holds at today", c, exc)
    return out


def growth_defaults(markets=("US", "UK", "DE", "JP")) -> dict:
    """Each market's WEO baseline real-growth path ({years ahead: percent}) -- the
    reference the endogenous-deficit rule differences a stated growth view against."""
    from macro.fiscal_rules import growth_defaults as _gd
    return _gd(markets)


def latest_states(markets=("US", "UK", "DE", "JP")) -> dict:
    """Today's anchors from the repo's own data."""
    from macro.fetch_core_cpi import core_inflation_yoy
    from model.tp_drivers import build_panel
    allobs = pd.read_parquet(Path(__file__).resolve().parent.parent
                             / "viz" / "data_daily.parquet")
    allobs = allobs[allobs["instrument"] == "nominal"]
    obs = allobs[allobs["tenor"] == TENOR]
    core = core_inflation_yoy()
    out = {}
    for c in markets:
        p = build_panel(c)
        last = obs[obs["country"] == c].sort_values("date").iloc[-1]
        keys = [k for k in (*DRIVERS, "fiscal") if k in p]
        drivers = {k: float(p[k].dropna().iloc[-1]) for k in keys}
        if "fiscal" not in drivers:      # spread markets: today's official 5y-ahead
            try:
                from macro.fiscal_projections import five_year_ahead
                drivers["fiscal"] = float(five_year_ahead(c))
            except Exception as exc:                            # noqa: BLE001
                logger.warning("no fiscal projection for %s (%s)", c, exc)
        mk = allobs[allobs["country"] == c]
        curve = (mk[mk["date"] == mk["date"].max()]
                 .dropna(subset=["expected"]).set_index("tenor")["expected"] * 100.0)
        # tp_today comes from the SAME daily observation as y_today and the expected
        # curve, not from the monthly panel's last row. The panel ends where its
        # DRIVERS end -- UK core CPI runs a month or two behind the curve -- so taking
        # the premium from there paired a July premium with an August yield and broke
        # y(0) = TP(0) + Pbar(0) by 6bp in the UK. The panel is what the model is
        # FITTED on; today's anchor is today's curve.
        out[c] = MarketState(
            country=c, y_today=float(last["observed"]) * 100.0,
            tp_today=float(last["term_premium"]) * 100.0,
            drivers_today=drivers, core_history=core[c],
            expected_curve={float(t): float(v) for t, v in curve.sort_index().items()})
    return out


def debt_projections(markets=("US", "UK", "DE", "JP")) -> dict:
    """Default debt/GDP paths from macro.debt_gdp, re-keyed to projection years."""
    from macro.debt_gdp import projection
    y0 = pd.Timestamp.today().year
    out = {}
    for c in markets:
        try:
            proj = projection(c)
            out[c] = {int(y) - y0: float(v) for y, v in proj.items() if int(y) > y0}
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("no debt projection for %s (%s); holding today's ratio", c, exc)
    return {c: v for c, v in out.items() if v}


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    from model.tp_drivers import build_panel, fit_market
    from model.tp_relative import build_pair, fit_relative

    panels = {c: build_panel(c) for c in ("US", "UK", "DE", "JP")}
    us_model = fit_market("US", panels["US"])
    spread = {c: fit_relative(c, build_pair(c, panels), us_model)
              for c in ("UK", "DE", "JP")}
    states = latest_states()
    debt = debt_projections()

    # An illustrative central scenario: policy glides to neutral, inflation settles
    # a touch above target. Replace with CSVs via load_scenario for real use.
    demo = {
        "US": Scenario("US", policy_rate={1: 3.9, 2: 3.4, 3: 3.1},
                       inflation={1: 2.7, 2: 2.4, 3: 2.3}, real_growth={1: 1.8}),
        "UK": Scenario("UK", policy_rate={1: 3.5, 2: 3.25, 3: 3.0},
                       inflation={1: 2.8, 2: 2.3, 3: 2.1}, real_growth={1: 1.2}),
        "DE": Scenario("DE", policy_rate={1: 2.0, 2: 2.0, 3: 2.0},
                       inflation={1: 2.1, 2: 2.0, 3: 2.0}, real_growth={1: 1.0}),
        "JP": Scenario("JP", policy_rate={1: 0.9, 2: 1.2, 3: 1.5},
                       inflation={1: 2.2, 2: 2.0, 3: 1.9}, real_growth={1: 0.8}),
    }
    import sys
    rho = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--rho=")),
                     RESIDUAL_FADE_US))
    print(f"US residual fade rho = {rho}/yr "
          f"({rho:.0%} of today's unexplained US premium survives one year, {rho ** 3:.0%} three)")
    res = run(us_model, spread, states, demo, debt, deficit_defaults(), rho=rho,
              growth_defaults=growth_defaults())
    for c, rows in res.items():
        st = states[c]
        print(f"\n{c}: 10y today {st.y_today:.2f}%  (TP {st.tp_today:+.2f}%),  "
              f"D={rows[0].duration:.1f}y")
        print(f"  curve prices an expected average policy rate of {rows[0].curve_expected:.2f}%;"
              f" your path today averages {rows[0].curve_expected + rows[0].policy_gap:.2f}%"
              f" ({rows[0].policy_gap:+.2f}pp)")
        for r in rows:
            print(f"  {r.horizon:>2}y horizon: TP {r.tp:+.2f}% + avg policy "
                  f"{r.policy_avg:.2f}%  =  10y yield {r.yield_10y:.2f}%   "
                  f"return {r.nominal_return:+.2f}% nom / {r.real_return:+.2f}% real p.a.")
            print(f"      expectations leg: market {r.market_policy_avg:.2f}% -> "
                  f"your {r.view_policy_avg:.2f}%, {r.view_weight:.0%} of the view in")


if __name__ == "__main__":
    _main()
