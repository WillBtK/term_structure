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

Optional `price_insensitive`, `debt_gdp` and `deficit` columns override the defaults
(ownership holds at today; the deficit follows each market's committed official
projection — CBO, OBR, European Commission AMECO, Cabinet Office, all deficit-positive
% of GDP via `macro.fiscal_projections`). The fiscal channel prices the five-year-ahead
projected deficit: directly for the US, and relative to the US path for the spread
markets, at the US-measured sensitivity. It is a real-rate effect — widening the
deficit is not by itself a reason to also raise the inflation path.

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
| **Yield** | `y(h) = TP(h) + P̄(h)`, `P̄` the ten-year average of your policy path |
| **Return** | Leibowitz constant-maturity: `R = y₀ + (Δy/H)·((H+1)/2 − D)`, deflated by your own inflation path for the real figure |

**Only the premium is anchored**, and the asymmetry is deliberate. The driver model
cannot explain today's premium level and should not pretend to, so its residual is
carried forward untouched. The expectations leg is different: your policy path *is* your
forecast of the expected average short rate, so taking only its change and discarding
its level would throw away the view you came to express.

Today's curve prices an expected average of `y(0) − TP(0)` — 3.11% for the US against a
4.75% yield and a 1.63% premium. A path averaging 3.50% says the market is 39bp too low,
and that belongs in the answer: it raises every projected yield and costs return. Every
result carries `policy_gap` so the disagreement is visible rather than buried, and an
unchanged scenario returns today's yield **only when your path matches what is priced**
— which is where the sheet opens.

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
