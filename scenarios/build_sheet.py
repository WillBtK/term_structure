"""Build the scenario sheet's data payload.

The input sheet runs entirely in the browser -- there is no server to call -- so
everything it needs travels with it: today's yield and term premium per market, the
fitted coefficients of both model families, the EWMA state at the end of the core-CPI
history, the committed debt projection, and annual history for the three charts.

The EWMA state is the one piece that needs justifying. Propagating inflation
uncertainty through a scenario needs the whole core history, which is far too much to
ship. It is not necessary: pandas' exponentially weighted moments converge to their
recursive form, so the mean and variance at the end of history are a sufficient
statistic. Continuing from ``(m, v)`` with ``m <- lam*m + (1-lam)*x`` and
``v <- lam*v + (1-lam)*(x-m)^2`` reproduces a full recomputation to within 1e-9,
verified against pandas over three test paths. Two numbers replace seventy years.

The rest of the payload is rounded to 3-4 decimals, which is why the browser and
``scenarios.engine`` agree to about 0.2 basis points rather than exactly.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "sheet_payload.json"
MARKETS = ("US", "UK", "DE", "JP")
HIST_FROM = "1990"

# --- Putting UK breakevens on a CPI footing ------------------------------------
# Index-linked gilts pay RPI, which has run above CPI for two reasons: the Carli
# formula in RPI's elementary aggregates, and coverage/weight differences. The gap
# has averaged roughly a point; 0.9pp is the middle of the usual 0.7-1.0 range and
# close to what inflation swaps traded as the "wedge".
#
# It is NOT a flat adjustment, and the reason matters. The November 2020 decision
# changes the RPI INDEX ITSELF from February 2030, not the terms of new issues: every
# RPI-linked cashflow accruing after that date -- on gilts already in issue as much as
# on anything sold later -- is calculated on CPIH methodology. There is no
# grandfathering and, explicitly, no compensation, which is why pension trustees
# sought judicial review in 2022 and lost. The date follows from the consent lock in
# the Statistics and Registration Service Act: the Authority needed the Chancellor's
# agreement while any "relevant" index-linked gilt remained in issue, and the last of
# those matures in 2030.
#
# So the forward curve genuinely should be fully converged by then, and a flat
# deduction would understate long-dated UK expectations by most of a point. The wedge
# is applied only to the share of each forward year falling before the reform. Beyond
# it RPI is CPIH, which differs from CPI only marginally (owner-occupier housing).
UK_RPI_WEDGE = 0.9          # percentage points of RPI over CPI, pre-reform
RPI_REFORM = 2030.083       # February 2030


def annual(s: pd.Series, start: str = HIST_FROM) -> dict:
    a = s.dropna().loc[start:].resample("YS").mean().dropna()
    return {"years": [int(t.year) for t in a.index],
            "v": [round(float(v), 2) for v in a]}


def monthly(s: pd.Series, start: str = HIST_FROM) -> dict:
    """A monthly series as a start date plus values -- no per-point dates, which keeps
    forty years of four series per market inside the payload budget. Gaps are kept as
    nulls so the chart breaks the line rather than inventing a straight segment."""
    a = s.loc[start:].resample("MS").mean()
    if a.dropna().empty:
        return {"t0": None, "v": []}
    a = a.loc[a.first_valid_index():a.last_valid_index()]
    t0 = a.index[0]
    return {"t0": round(t0.year + (t0.month - 1) / 12.0, 4),
            "v": [None if pd.isna(v) else round(float(v), 2) for v in a]}


def _cpi_adjust(country: str, path: list, t0: float) -> tuple:
    """Return (path on a CPI footing, footnote). ``path[k]`` is the forward inflation
    rate for the year running from ``t0 + k``."""
    if country != "UK" or not path:
        return path, ""
    out = []
    for k, v in enumerate(path):
        a, b = t0 + k, t0 + k + 1
        pre = max(0.0, min(b, RPI_REFORM) - a) / (b - a)     # share still on RPI rules
        out.append(round(v - UK_RPI_WEDGE * pre, 3))
    return out, (f"Index-linked gilts pay RPI, so {UK_RPI_WEDGE:.1f}pp is deducted for "
                 "the portion of each forward year before February 2030. The 2020 reform "
                 "changes the RPI index itself, so every RPI-linked cashflow after that "
                 "date accrues on CPIH methodology &mdash; on gilts already in issue, not "
                 "just new ones &mdash; and no wedge is applied beyond it.")


def _curve_path(curve: dict, years: int = 12) -> list:
    """Year-by-year forwards implied by a curve of AVERAGES over 0..tau, via
    C(t) = t*curve(t). Used for the market's policy path and for breakeven inflation."""
    from scenarios.engine import market_forward_average
    return [] if not curve else [round(market_forward_average(curve, k, span=1.0), 3)
                                 for k in range(years)]


def _forward_tens(observed: dict, horizons: int = 11) -> list:
    """The market's own forward ten-year yield at each horizon -- what the curve already
    prices for the bond we are projecting, and so the reference our path has to beat.
    Same cumulative identity, applied to observed yields rather than expected rates."""
    from scenarios.engine import market_forward_average
    if not observed:
        return []
    return [round(market_forward_average(observed, h, span=10.0), 3)
            for h in range(horizons)]


def _deficit_history(country: str) -> list:
    """Deficit history, percent of GDP, deficit-positive, as [[year, value], ...].

    The US series is the DRIVER's own history -- a hundred vintages of CBO's
    five-year-ahead projection, which is the statistic the coefficient was measured on.
    The others have no vintage archive (the recorded upgrade path), so they show the
    realised balance from the same official source that supplies their projection.
    Both are deficits in percent of GDP; the chart says which is which.
    """
    try:
        if country == "US":
            from macro.fetch_cbo import load
            s = load()
            return [[round(d.year + (d.month - 1) / 12.0, 2), round(float(v), 2)]
                    for d, v in s.items()]
        from macro.fiscal_projections import FILES
        d = pd.read_csv(FILES[country])
        now = pd.Timestamp.today().year
        return [[int(r["year"]), round(-float(r["balance_pct_gdp"]), 2)]
                for _, r in d.iterrows() if int(r["year"]) <= now]
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("no deficit history for %s (%s)", country, exc)
        return []


def build() -> dict:
    warnings.filterwarnings("ignore")
    from macro.fetch_core_cpi import UNCERTAINTY_HALFLIFE as HL, core_inflation_yoy
    from model.bond_math import macaulay_duration
    from model.tp_drivers import build_panel, fit_market
    from model.tp_relative import build_pair, fit_relative
    from scenarios.engine import (TENOR, VIEW_CONVERGENCE, debt_projections,
                                  deficit_defaults, latest_states,
                                  market_forward_average)

    panels = {c: build_panel(c) for c in MARKETS}
    allobs = pd.read_parquet(HERE.parent / "viz" / "data_daily.parquet")
    us_m = fit_market("US", panels["US"])
    spread = {c: fit_relative(c, build_pair(c, panels), us_m)
              for c in MARKETS if c != "US"}
    states, debt = latest_states(MARKETS), debt_projections(MARKETS)
    deficits = deficit_defaults(MARKETS)
    core = core_inflation_yoy()
    obs = pd.read_parquet(HERE.parent / "viz" / "data_daily.parquet")
    obs = obs[obs["instrument"] == "nominal"]
    today = pd.Timestamp.today().normalize()

    now = pd.Timestamp.today()
    year_frac = now.year + (now.dayofyear - 1) / 365.25
    from macro import weo
    from macro.fiscal_rules import GAP_CLOSURE, SEMI_ELASTICITY
    growth_base = {}
    for c in MARKETS:
        try:        # keyed by CALENDAR YEAR; the page maps years to horizons itself
            growth_base[c] = {y: v for y, v in weo.growth(c).items()
                              if y >= pd.Timestamp.today().year}
        except Exception:                                           # noqa: BLE001
            pass
    payload = {"halflife": HL, "tenor": TENOR, "lam": 0.5 ** (1.0 / HL),
               "asof": str(obs["date"].max().date()), "markets": {},
               "view_convergence": VIEW_CONVERGENCE,
               "year0": int(now.year), "year_frac": round(float(year_frac), 3),
               # the endogenous-deficit rule: deficit(t) = official(t) - eps*gap(t),
               # the gap cumulating growth deviations vs the WEO baseline and closing
               # at GAP_CLOSURE per year (macro.fiscal_rules has the provenance)
               "fiscal_rule": {"eps": SEMI_ELASTICITY, "closure": GAP_CLOSURE}}
    for c in MARKETS:
        st, hist = states[c], core[c].dropna()
        m = hist.ewm(halflife=HL).mean()
        v = ((hist - m) ** 2).ewm(halflife=HL).mean()
        mk = obs[obs["country"] == c]
        short = mk[mk["tenor"] == sorted(mk["tenor"].unique())[0]]
        ten = mk[mk["tenor"] == TENOR].set_index("date")["observed"].sort_index() * 100
        shortm = short.set_index("date")["observed"].sort_index() * 100
        # today's nominal curve, and the breakeven curve where a real curve exists
        cm = allobs[(allobs["country"] == c) & (allobs["date"] == mk["date"].max())]
        nom = (cm[cm["instrument"] == "nominal"].dropna(subset=["observed"])
               .set_index("tenor")["observed"] * 100).sort_index()
        rl = (cm[cm["instrument"] == "real"].dropna(subset=["observed"])
              .set_index("tenor")["observed"] * 100).sort_index()
        be = {}
        for t in rl.index:                       # breakeven = nominal - real, by tenor
            if len(nom):
                be[float(t)] = float(np.interp(float(t), nom.index.values.astype(float),
                                               nom.values) - rl.loc[t])
        dh = _deficit_history(c)
        be_path, be_note = _cpi_adjust(c, _curve_path(be) if be else [], year_frac)
        if be_path and not be_note:
            be_note = "Breakevens are CPI-linked; no index adjustment is applied."
        if be_path:
            be_note += (" They still carry an inflation risk premium, so they sit above "
                        "expected inflation" +
                        (", and UK linkers are further supported by structural "
                         "liability-driven demand; neither is adjusted away here."
                         if c == "UK" else "; that is not adjusted away here."))
        payload["markets"][c] = {
            "y_today": round(st.y_today, 3), "tp_today": round(st.tp_today, 3),
            "drivers": {k: round(float(x), 3) for k, x in st.drivers_today.items()},
            "duration": round(macaulay_duration(
                today, today + pd.DateOffset(years=int(TENOR)),
                st.y_today, st.y_today / 100.0), 3),
            "ewma": {"m": round(float(m.iloc[-1]), 4),
                     "v": round(float(v.iloc[-1]), 6),
                     "last": round(float(hist.iloc[-1]), 3),
                     "asof": str(hist.index[-1].date())},
            "deficit_proj": {str(k): round(float(x), 2)
                             for k, x in sorted(deficits.get(c, {}).items())},
            # WEO baseline real growth, {years ahead: %} -- the reference the
            # endogenous-deficit rule differences the growth row against
            "growth_base": {str(k): round(float(v), 3)
                            for k, v in sorted(growth_base.get(c, {}).items())},
            # the market's own expected-rate curve, and the year-by-year policy path
            # implied by it -- the sheet's genuinely neutral opening
            "expected_curve": {str(t): round(float(v), 4)
                               for t, v in sorted(st.expected_curve.items())},
            "market_policy_path": _curve_path(st.expected_curve),
            # the market's own forward ten-year yield -- the benchmark for our path
            "fwd10": _forward_tens({float(t): float(v) for t, v in nom.items()}),
            # market-implied inflation from breakevens, where a real curve exists
            "be_path": be_path, "be_note": be_note,
            "deficit_hist": dh,
            "hist": {"core": annual(core[c]), "short": annual(shortm),
                     "ten": annual(ten)},
            "mhist": {"core": monthly(core[c]), "short": monthly(shortm),
                      "ten": monthly(ten), "tp": monthly(panels[c]["tp"])},
            "model": ({"kind": "us", "intercept": us_m.intercept, "coef": us_m.coef,
                       "adjust": us_m.adjust, "base": us_m.base}
                      if c == "US" else
                      {"kind": "spread", "intercept": spread[c].intercept,
                       "beta_us": spread[c].beta_us, "coef": spread[c].coef,
                       "adjust": spread[c].adjust, "base": spread[c].base}),
        }
    return payload


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    for c, m in payload["markets"].items():
        print(f"{c}: 10y {m['y_today']:.2f}%  TP {m['tp_today']:+.2f}%  D {m['duration']:.1f}y"
              f"  core {m['ewma']['last']:+.2f}% as of {m['ewma']['asof']}")
    print(f"\nwrote {OUT} ({len(OUT.read_text())} bytes)")


if __name__ == "__main__":
    _main()
