"""From a macro forecast to a ten-year yield and a bond return.

One instrument -- the ten-year bond -- in four markets, two horizons: the yield in
three years' time and in ten. The user supplies, per country, annual projections of
the policy rate, headline inflation and real growth. Growth is informational only: it
is assumed to have landed on the policy-rate forecast already, which is treated as
exogenous. Inflation is headline, on the stated basis that beyond base effects
headline equals core.

The chain, every link anchored to today:

  1. TERM PREMIUM.  The driver models (``model.tp_drivers`` for the US,
     ``model.tp_relative`` for the spread markets) translate the scenario's driver
     values into a premium CHANGE from today; today's observed premium supplies the
     level. Two drivers come straight from the scenario: long-run expected inflation
     is the ten-year forward average of the projected inflation path, and inflation
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

  2. YIELD.  y(h) = y(0) + [TP(h) - TP(0)] + [Pbar(h) - Pbar(0)], where Pbar(t) is
     the average of the user's policy path over the ten years from t, the path held
     flat beyond its last projected year (their terminal IS their long-run view).
     The same anchoring logic as the premium: the model never restates today's
     yield, it only moves it by what the scenario changes. A ten-year-ahead yield
     needs policy expectations to year twenty, which the flat extension provides.

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
REQUIRED_COLUMNS = ("year", "policy_rate", "inflation", "real_growth")


@dataclass(frozen=True)
class Scenario:
    """One market's annual projections, all in percent, keyed by projection year
    (1 = the first full year ahead). ``real_growth`` is carried but never used in the
    arithmetic -- it is the story behind the policy path, not a second input to it."""
    country: str
    policy_rate: dict
    inflation: dict
    real_growth: dict = field(default_factory=dict)
    price_insensitive: dict = field(default_factory=dict)   # optional override, % GDP
    debt_gdp: dict = field(default_factory=dict)            # optional override, % GDP

    def __post_init__(self):
        for name in ("policy_rate", "inflation"):
            path = getattr(self, name)
            if not path:
                raise ValueError(f"{self.country}: scenario needs a {name} path")
            years = sorted(path)
            if years[0] != 1 or years != list(range(1, len(years) + 1)):
                raise ValueError(f"{self.country}: {name} years must run 1..N "
                                 f"without gaps, got {years}")


@dataclass(frozen=True)
class MarketState:
    """Today, as the data stands: the anchor for every projection."""
    country: str
    y_today: float                    # observed 10y yield, %
    tp_today: float                   # estimated 10y term premium, %
    drivers_today: dict               # the four driver values the panel ends on
    core_history: pd.Series           # monthly YoY core inflation, %, for the EWMA


def step(path: dict, year: float) -> float:
    """The path value in a given projection year: annual steps, held flat beyond the
    last projection and anchored before year 1 at the year-1 value."""
    years = sorted(path)
    if year <= years[0]:
        return float(path[years[0]])
    if year >= years[-1]:
        return float(path[years[-1]])
    return float(path[int(np.floor(year))])


def forward_average(path: dict, start: int, span: int = int(TENOR)) -> float:
    """Average of the path over ``span`` years beginning after year ``start`` -- the
    expectations leg of a bond bought at horizon ``start``."""
    return float(np.mean([step(path, start + k) for k in range(1, span + 1)]))


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
    n = horizon * 12
    scen = [step(scenario.inflation, 1 + m // 12) for m in range(n)]
    base = [float(hist.iloc[-1])] * n
    delta = _ewma_vol_at(hist, scen, hl) - _ewma_vol_at(hist, base, hl)
    return max(state.drivers_today["inflation_uncertainty"] + delta, 0.0)


def drivers_at(state: MarketState, scenario: Scenario, horizon: int,
               debt_default: dict | None = None) -> dict:
    """The four driver values the premium model sees at a horizon."""
    out = {
        "expected_inflation": forward_average(scenario.inflation, horizon),
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
    yield_10y: float                  # projected 10y yield, %
    nominal_return: float             # annualised, buying today and holding `horizon`
    real_return: float
    duration: float


def run_market(state: MarketState, scenario: Scenario, tp_path: dict,
               debt_default: dict | None = None) -> list:
    """Assemble yields and returns from an already-computed premium path.

    ``tp_path`` maps horizon (0 included) to the premium in percent -- computed by the
    caller because the US and the spread markets use different models.
    """
    from model.bond_math import macaulay_duration
    today = pd.Timestamp.today().normalize()
    dur = macaulay_duration(today, today + pd.DateOffset(years=int(TENOR)),
                            coupon_pct=state.y_today, ytm=state.y_today / 100.0)
    pbar0 = forward_average(scenario.policy_rate, 0)
    out = []
    for h in HORIZONS:
        pbar = forward_average(scenario.policy_rate, h)
        y = state.y_today + (tp_path[h] - tp_path[0]) + (pbar - pbar0)
        nom = leibowitz_return(state.y_today, y, h, dur)
        out.append(ScenarioResult(
            country=state.country, horizon=h, tp=tp_path[h], policy_avg=pbar,
            yield_10y=y, nominal_return=nom,
            real_return=real_return(nom, scenario, h), duration=dur))
    return out


def run(us_model: DriverModel, spread_models: dict, states: dict, scenarios: dict,
        debt_defaults: dict | None = None) -> dict:
    """All markets. The US premium path is computed first; every spread market's
    premium is a beta on it plus its drivers relative to the US, both at the same
    horizon, exactly as fitted."""
    debt_defaults = debt_defaults or {}
    us_state, us_scen = states["US"], scenarios["US"]
    us_drv = {0: us_state.drivers_today}
    us_drv.update({h: drivers_at(us_state, us_scen, h, debt_defaults.get("US"))
                   for h in HORIZONS})
    us_tp = {0: us_state.tp_today}
    us_tp.update({h: us_model.scenario(us_drv[0], us_drv[h], us_state.tp_today)
                  for h in HORIZONS})

    results = {"US": run_market(us_state, us_scen, us_tp, debt_defaults.get("US"))}
    for c, m in spread_models.items():
        if c not in scenarios:
            continue
        st, sc = states[c], scenarios[c]
        tp = {0: st.tp_today}
        for h in HORIZONS:
            own_h = drivers_at(st, sc, h, debt_defaults.get(c))
            tp[h] = m.scenario(us_drv[0], us_drv[h], st.drivers_today, own_h,
                               us_tp[0], us_tp[h], st.tp_today)
        results[c] = run_market(st, sc, tp, debt_defaults.get(c))
    return results


# ------------------------------------------------------------------ repo-data wiring

def load_scenario(path: "Path | str") -> Scenario:
    """A scenario CSV: columns ``year, policy_rate, inflation, real_growth`` (percent,
    year 1 = first full year ahead), optional ``price_insensitive`` / ``debt_gdp``."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df = df.dropna(subset=["year"]).astype({"year": int}).set_index("year")
    def col(name):
        return {int(y): float(v) for y, v in df[name].dropna().items()} \
            if name in df else {}
    return Scenario(country=path.stem.upper(),
                    policy_rate=col("policy_rate"), inflation=col("inflation"),
                    real_growth=col("real_growth"),
                    price_insensitive=col("price_insensitive"),
                    debt_gdp=col("debt_gdp"))


def latest_states(markets=("US", "UK", "DE", "JP")) -> dict:
    """Today's anchors from the repo's own data."""
    from macro.fetch_core_cpi import core_inflation_yoy
    from model.tp_drivers import build_panel
    obs = pd.read_parquet(Path(__file__).resolve().parent.parent
                          / "viz" / "data_daily.parquet")
    obs = obs[(obs["instrument"] == "nominal") & (obs["tenor"] == TENOR)]
    core = core_inflation_yoy()
    out = {}
    for c in markets:
        p = build_panel(c)
        last = obs[obs["country"] == c].sort_values("date").iloc[-1]
        out[c] = MarketState(
            country=c, y_today=float(last["observed"]) * 100.0,
            tp_today=float(p["tp"].iloc[-1]),
            drivers_today={k: float(p[k].iloc[-1]) for k in DRIVERS},
            core_history=core[c])
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
    res = run(us_model, spread, states, demo, debt)
    for c, rows in res.items():
        st = states[c]
        print(f"\n{c}: 10y today {st.y_today:.2f}%  (TP {st.tp_today:+.2f}%),  "
              f"D={rows[0].duration:.1f}y")
        for r in rows:
            print(f"  {r.horizon:>2}y horizon: TP {r.tp:+.2f}%  avg policy "
                  f"{r.policy_avg:.2f}%  ->  10y yield {r.yield_10y:.2f}%   "
                  f"return {r.nominal_return:+.2f}% nom / {r.real_return:+.2f}% real p.a.")


if __name__ == "__main__":
    _main()
