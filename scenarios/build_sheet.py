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

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "sheet_payload.json"
MARKETS = ("US", "UK", "DE", "JP")
HIST_FROM = "1990"


def annual(s: pd.Series, start: str = HIST_FROM) -> dict:
    a = s.dropna().loc[start:].resample("YS").mean().dropna()
    return {"years": [int(t.year) for t in a.index],
            "v": [round(float(v), 2) for v in a]}


def build() -> dict:
    warnings.filterwarnings("ignore")
    from macro.fetch_core_cpi import UNCERTAINTY_HALFLIFE as HL, core_inflation_yoy
    from model.bond_math import macaulay_duration
    from model.tp_drivers import build_panel, fit_market
    from model.tp_relative import build_pair, fit_relative
    from scenarios.engine import (TENOR, debt_projections, deficit_defaults,
                                  latest_states)

    panels = {c: build_panel(c) for c in MARKETS}
    us_m = fit_market("US", panels["US"])
    spread = {c: fit_relative(c, build_pair(c, panels), us_m)
              for c in MARKETS if c != "US"}
    states, debt = latest_states(MARKETS), debt_projections(MARKETS)
    deficits = deficit_defaults(MARKETS)
    core = core_inflation_yoy()
    obs = pd.read_parquet(HERE.parent / "viz" / "data_daily.parquet")
    obs = obs[obs["instrument"] == "nominal"]
    today = pd.Timestamp.today().normalize()

    payload = {"halflife": HL, "tenor": TENOR, "lam": 0.5 ** (1.0 / HL),
               "asof": str(obs["date"].max().date()), "markets": {}}
    for c in MARKETS:
        st, hist = states[c], core[c].dropna()
        m = hist.ewm(halflife=HL).mean()
        v = ((hist - m) ** 2).ewm(halflife=HL).mean()
        mk = obs[obs["country"] == c]
        short = mk[mk["tenor"] == sorted(mk["tenor"].unique())[0]]
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
            "debt_proj": {str(k): round(x, 1) for k, x in sorted(debt.get(c, {}).items())},
            "deficit_proj": {str(k): round(float(x), 2)
                             for k, x in sorted(deficits.get(c, {}).items())},
            "hist": {
                "core": annual(core[c]),
                "short": annual(short.set_index("date")["observed"].sort_index() * 100),
                "ten": annual(mk[mk["tenor"] == TENOR]
                              .set_index("date")["observed"].sort_index() * 100)},
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
