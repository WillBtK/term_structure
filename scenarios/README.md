# Scenario tool

A macro forecast in, a ten-year yield and a bond return out, for the US, UK, Germany
and Japan. One instrument — the ten-year bond — at two horizons: three years and ten.

## Inputs

Annual paths per market, in percent, **at each calendar year end**: **policy rate**,
**headline inflation**, **real growth**. Growth is carried but never enters the
arithmetic — it is the story behind the policy path, which is treated as exogenous.
Inflation is headline, on the stated basis that beyond base effects headline equals
core. The last entered year is held flat beyond the grid, so whatever year you hold
flat from becomes your long-run view.

```
scenarios/us.csv
year,policy_rate,inflation,real_growth,deficit
2026,3.9,2.7,1.8,5.87
2027,3.4,2.4,1.7,5.83
2028,3.1,2.3,1.7,5.99
```

The grid **opens on the market's own expected policy path**, derived from the published
expected-rate curve, so an untouched column is not a view. `load_scenario` converts
calendar years to horizons ahead of today (year-end *Y* is `(Y+1) − today`); a file
written in the older years-ahead convention still loads unchanged.

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
| **Yield** | `y(h) = TP(h) + P̄(h)`, `P̄` blending the market's forward expectation toward your path |
| **Return** | Leibowitz constant-maturity: `R = y₀ + (Δy/H)·((H+1)/2 − D)`, deflated by your own inflation path for the real figure |

**Both legs start from the market**, and converge at rates you control. That symmetry is
the point. The premium leg carries today's unexplained residual forward, decaying at `ρ`
toward the model's fair value. The expectations leg does the same with your rate view:

```
P̄(h) = mkt(h) + [your(h) − mkt(h)] · (1 − λ^h)
```

`mkt(h)` is the market's **own** expected average over the ten years from `h`, read off
the published expected-rate curve — and it is not the same number at every horizon. The
US curve prices 3.11% over the next ten years but 2.75% over the ten years starting in
three, because it expects policy to fall. Anchoring every horizon on the spot ten-year
average would smuggle that decline in as a view nobody expressed.

Without this, a three-year projection paired the **market's** term premium with your
**entire** rate view — an inconsistency that showed up as an implausibly high ten-year
yield at short horizons. `λ = 0.8` puts about half a stated view into the three-year
horizon and nine tenths into the ten-year one; `λ = 0` restores take-it-literally
behaviour and `λ = 1` projects the market's own forward curve. At `h = 0` no view is in
yet, so `y(0) = TP(0) + P̄(0)` reproduces today's yield exactly. Every result carries
`market_policy_avg`, `view_policy_avg` and `view_weight` so the blend is inspectable.

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

`scenarios/sheet/index.html` is the published input page. The headline sits at the top:
the ten-year yield's monthly history, your projected path, and **the curve's own forward
ten-year yield** (`fwd10`, from `C(t)=t·y(t)` on the nominal curve) — the benchmark a
view has to beat — with the result cards beside it. Below that, a calendar-year entry
grid whose top row is the **derived ten-year yield** (hover a cell for the term-premium
split), a hold-flat-from selector, and four half-width charts: policy rate against the
market's implied path, inflation against the market breakeven, term premium, and
deficit. All history is monthly with hover readouts.

**UK breakevens are put on a CPI footing before display.** Index-linked gilts pay RPI,
which has run roughly a point above CPI (the Carli formula effect plus coverage and
weight differences); `UK_RPI_WEDGE = 0.9pp` is the middle of the usual 0.7–1.0 range.
The deduction is *not* flat: the 2020 UK Statistics Authority reform aligns RPI to CPIH
from **February 2030**, so only the pre-reform share of each forward year is adjusted
and the straddling year is pro-rated. A flat point would understate long-dated UK
expectations by most of a point — and the correction matters for reading the curve: the
raw RPI forwards fall from 3.9% to 3.1%, which looks like expected disinflation, but on
a CPI footing the path is a roughly flat 3%. The apparent decline was mostly the reform.
The footnote under the chart states the adjustment; breakevens still carry an inflation
risk premium, which is not adjusted away. It computes entirely in
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
