# Scenario tool

A macro forecast in, a ten-year yield and a bond return out, for the US, UK, Germany
and Japan. One instrument — the ten-year bond — at two horizons: three years and ten.

## Inputs

Three annual paths per market, in percent, by year ahead (1 = the first full year):
**policy rate**, **headline inflation**, **real growth**. Growth is carried but never
enters the arithmetic — it is the story behind the policy path, which is treated as
exogenous. Inflation is headline, on the stated basis that beyond base effects headline
equals core. The last projected year is held flat beyond the grid: your terminal *is*
your long-run view.

```
scenarios/us.csv
year,policy_rate,inflation,real_growth
1,3.9,2.7,1.8
2,3.4,2.4,1.7
3,3.1,2.3,1.7
```

Optional `price_insensitive` and `debt_gdp` columns override the defaults (ownership
holds at today; debt follows the committed WEO/national projection).

## Running it

```bash
python -m scenarios.engine          # the demo scenario, all four markets
python -m scenarios.build_sheet     # refresh the browser payload
```

```python
from scenarios.engine import load_scenario, latest_states, debt_projections, run
```

## The chain

| step | what happens |
|---|---|
| **Term premium** | the driver models price the scenario's *change*; today's observed premium supplies the level |
| **Yield** | `y(h) = y(0) + ΔTP + ΔP̄`, `P̄` the ten-year average of your policy path |
| **Return** | Leibowitz constant-maturity: `R = y₀ + (Δy/H)·((H+1)/2 − D)`, deflated by your own inflation path for the real figure |

Anchoring is the discipline: the model supplies changes, never levels, so whatever it
cannot explain today it carries forward untouched instead of forecasting away.

## The two inflation channels

Your single inflation path feeds two drivers that sit at different levels today, so no
flat path zeroes both — and that is correct rather than a defect.

* **Long-run expectation** is the ten-year forward average of your path, measured
  against today's *surveyed* expectation (US: 2.3%).
* **Uncertainty** is the fitted EWMA volatility propagated through your path and
  measured against a *status-quo baseline* — inflation continuing at its latest
  realised reading (US core: 2.8%).

Without that baseline the EWMA decays under any smooth path and a no-change scenario
quietly compresses the premium ~40bp over a decade: a hidden view. With it, extending
the status quo moves nothing and only your own departures count.

## The sheet

`scenarios/sheet/index.html` is the published input page — country tabs, an editable
grid, your paths drawn against history, and live yield/return. It computes entirely in
the browser from `scenarios/data/sheet_payload.json` (built by `build_sheet.py`), and
agrees with `scenarios.engine` to about **0.2 basis points**; the gap is payload
rounding, nothing else. Export writes a CSV in exactly the format `load_scenario` reads.

To rebuild and republish: `python -m scenarios.build_sheet`, regenerate the page's
inlined payload, publish `scenarios/sheet/index.html`.

## What this is, and is not

A **conditional mapping**: you supply the scenario, the tool prices it. Not a forecaster
— see `validation/FINDINGS_scenario_oos.md`, where matching the random walk at three
years is the expected result. Present sensitivities, never forecasts; the scenarios
carry the uncertainty. Japan is directional only: its beta on the US premium is 0.74
over the full sample against 0.12 on changes since 2010.
