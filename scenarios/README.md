# Scenario tool

A macro forecast in, a ten-year yield and a bond return out, for the US, UK, Germany
and Japan. One instrument — the ten-year bond — at two horizons: three years and ten,
plus the US thirty-year in the comparison table (below).

## Inputs

Annual paths per market, in percent, **at each calendar year end**: **policy rate**,
**headline inflation**, **real growth**. Growth is carried but never enters the
arithmetic — it is the story behind the policy path, which is treated as exogenous.
Inflation is headline, on the stated basis that beyond base effects headline equals
core. The last entered year is held flat beyond the grid, so whatever year you hold
flat from becomes your long-run view.

```
scenarios/us.csv
year,policy_rate,inflation,real_growth,fiscal_balance
2026,3.9,2.7,1.8,-5.87
2027,3.4,2.4,1.7,-5.83
2028,3.1,2.3,1.7,-5.99
```

The grid **opens on the market's own expected policy path** (derived from the
published expected-rate curve) **and on the WEO macro baseline for growth and
inflation** — the house starting point, one vintage across all four markets.
`load_scenario` converts
calendar years to horizons ahead of today (year-end *Y* is `(Y+1) − today`); a file
written in the older years-ahead convention still loads unchanged.

Optional `price_insensitive`, `debt_gdp` and `fiscal_balance` columns override the
defaults. Fiscal figures are entered and displayed as the **balance** — deficit
negative — while the engine's internal convention stays deficit-positive, the basis
the coefficient was measured on (a deficit-positive `deficit` column from older files
still loads). Ownership holds at today; the fiscal default follows each market's
committed official projection — CBO, OBR, European Commission AMECO extended with the
IMF WEO shape, Cabinet Office, via `macro.fiscal_projections`. The fiscal channel
prices the five-year-ahead projected deficit: directly for the US, and relative to
the US path for the spread markets, at the US-measured sensitivity. It is a real-rate
effect — widening the deficit is not by itself a reason to also raise the inflation
path.

**The fiscal path is derived unless you state it.** A scenario with a growth view and
no fiscal column gets its deficit endogenously (`macro.fiscal_rules`): growth entered
below the baseline widens the official path via each market's own institution's
rule of thumb — CBO's rules-of-thumb workbook for the US (ε = 0.36), the OBR's
cyclical-adjustment coefficient for the UK (0.70), the European Commission's
semi-elasticity for Germany (0.504, recovered exactly from AMECO's own
decomposition), the OECD's for Japan (0.45) — applied to an output gap cumulated
from the growth deviations and closing at 30%/yr, so a one-off miss washes out and
only a sustained view leaves a mark. The baseline the deviation is measured against
is the **deficit source's own growth projection** (CBO / OBR / AMECO+WEO / Cabinet
Office, `macro.fiscal_projections.growth_baseline`) — the assumptions the official
deficit was conditioned on. Because the grid opens on WEO growth, the untouched
balance row shows the official projection **re-aligned to WEO growth** through the
sensitivities; typing the source's own growth numbers recovers the official path
exactly. The rule assumes growth deviations are
economy-driven; deficit-financed stimulus is a fiscal view and is entered in the
balance row (or `fiscal_balance` column) directly, which switches the rule off — the
override is the whole fiscal path, never rule-plus-override. Inflation is deliberately excluded (its budget
effect flips sign with the rate response, and it already enters the premium twice),
and so is the interest-cost feedback of our own yields onto the deficit (loop gain
~0.45 — documented, not wired). Evidence and provenance:
`validation/FINDINGS_endogenous_deficit.md`.

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
market's implied path, inflation against the market breakeven, term premium, and the
fiscal balance (deficit negative). All history is monthly with hover readouts. The
growth and inflation rows open on the WEO baseline and the balance row is
**derived** from the growth row
(italic, ⚙) until you type over it (✎ — click the row label to restore the rule);
while derived, the exported CSV omits the fiscal column so the engine re-derives the
identical full path.

It computes entirely in the browser from `scenarios/data/sheet_payload.json` (built by
`build_sheet.py`) and agrees with `scenarios.engine` to about **0.2 basis points**; the
gap is payload rounding, nothing else. Export writes a CSV in exactly the format
`load_scenario` reads.

### UK breakevens are put on a CPI footing before display

Index-linked gilts pay RPI, which has run roughly a point above CPI — the Carli formula
effect in RPI's elementary aggregates plus coverage and weight differences.
`UK_RPI_WEDGE = 0.9pp` is the middle of the usual 0.7–1.0 range.

The deduction is **not flat**, because the 2020 reform changes the *index itself* from
**February 2030**: every RPI-linked cashflow after that date accrues on CPIH
methodology, on gilts already in issue as much as on new ones. There is no
grandfathering and no compensation — pension trustees sought judicial review in 2022 and
lost — and the date follows from the consent lock in the Statistics and Registration
Service Act, which the Authority needed while any "relevant" index-linked gilt remained
in issue; the last of those matures in 2030. **So the forward curve genuinely should be
fully converged by then.** Only the pre-reform share of each forward year is adjusted,
and the straddling year is pro-rated.

This matters for reading the curve, not just for tidiness: the raw RPI forwards fall
from 3.9% to 3.1%, which looks like expected disinflation, but on a CPI footing the path
is a roughly flat 3%. The apparent decline was mostly the reform.

Two things are deliberately **not** adjusted away, and the footnote says so: breakevens
carry an inflation risk premium, and UK linkers are further supported by structural
liability-driven demand. Together those are why the adjusted UK line still reads above
target.

To rebuild and republish: `python -m scenarios.build_sheet`, regenerate the page's
inlined payload, publish `scenarios/sheet/index.html`.

## What this is, and is not

A **conditional mapping**: you supply the scenario, the tool prices it. Not a forecaster
— see `validation/FINDINGS_scenario_oos.md`, where matching the random walk at three
years is the expected result. Present sensitivities, never forecasts; the scenarios
carry the uncertainty. Japan is directional only: its beta on the US premium is 0.74
over the full sample against 0.12 on changes since 2010.

## Named scenarios

The sheet holds five independent scenarios, switched by the bar above the market tabs:
**Baseline**, **Goldilocks**, **High growth, high inflation**, **Stagflation** and
**Hard landing**. Each is a complete copy of the input state for every market, so
switching is instant and nothing leaks across; a dot on the bar marks the ones that
have been moved off their opening values.

`Baseline` opens where the sheet has always opened — each market's own expected policy
path, the IMF WEO macro baseline for growth and inflation, and the official fiscal
balance re-aligned to it. The other four start as copies until their numbers are
entered.

**Pre-populating from a CSV.** `python -m scenarios.load_presets views.csv` parses a
scenario file and writes the sheet's `PRESETS` block. `--check` parses and reports
without touching the sheet. Long format:

```csv
scenario,market,row,year,value
Stagflation,US,inflation,2027,4.5
Stagflation,US,policy_rate,2027,5.5
```

A wide layout with one column per year is detected and reshaped automatically, which is
usually how a forecast table arrives. Market names, row names and scenario names are
matched loosely — `United States`, `Real GDP growth`, `high growth high inflation` all
resolve — but anything unrecognised **raises** rather than being quietly dropped, and a
blank value is skipped rather than read as zero.

Rows are `policy_rate`, `inflation`, `real_growth` and `fiscal_balance`; the first three
are percent, the balance is percent of GDP with **deficit negative**, matching what the
sheet shows. Supplying a balance takes that market's row manual, so the growth rule
stops driving it — supply growth instead and let the rule do the work.

**A view stated for part of the grid runs to the end of it.** The last year you give is
carried forward. Stating policy at 2% for 2027–28 and nothing after means 2% from 2027
onward, not a dip and a snap back to the market's path in 2029. State the later years
explicitly to say something different. Years you never touch on a row keep the baseline
opening, so a scenario can state a view on two rows and stay neutral on the rest.

**What the reader gets.** A comparison table above the grid runs all five side by side
for the selected market: the ten-year yield at three and ten years, the difference from
Baseline in basis points, and the annualised nominal and real return. **For the US each
scenario gets a second row, the thirty-year** — a genuine second fit of the same model
on the thirty-year premium (`viz/data_daily.parquet` carries US 30y nominal from 1985;
R² 0.90 on 488 months), with the expected-rate leg averaged over thirty years rather
than ten so the identity `y = premium + expected average short rate` still closes on
today's yield to a hundredth of a basis point. The drivers are the same series at the
same definitions — expected inflation is a ten-year survey measure, the fiscal driver
the five-year-ahead projection — because that is what the coefficients are estimated
against at every tenor; holding them fixed is what lets the thirty-year loadings be read
as *how much more the long end moves for the same shift*. They differ substantially: the
long end loads less on the LEVEL of expected inflation (+0.72 against +1.01) and far
more on inflation UNCERTAINTY (+1.81 against +0.81), which is the sensible reading — a
thirty-year bond is paid for the width of the distribution, not its centre. Each bond is
compared against its own Baseline; a thirty-year yield is not a spread to the ten.

Only the US. The other three price off the US premium through the relative model in
`model/tp_relative.py`, and that model is fitted at ten years, so a thirty-year Bund or
gilt would need its own relative fit — not claimed here. Typed numbers
persist in the browser between visits and are dropped automatically if the underlying
data vintage changes, so a stored scenario can never mix last month's market path with
this month's charts. Export writes all five as one CSV, each block readable by
`load_scenario` unchanged.
