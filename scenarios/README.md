# Scenario tool

A macro forecast in, a ten-year yield and a bond return out, for the US, UK, Germany
and Japan. One instrument — the ten-year bond — at every calendar year end to ten years
out (the house year-end forecasts), and at three and ten years for returns.

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
python -m scenarios.year_end_10y --scenario all --round   # year-end 10y table, off the sheet
python -m scenarios.load_presets scenarios/data/{us,uk,de,jp}_scenarios.csv   # CSVs -> sheet
python -m scenarios.engine          # the demo scenario, all four markets
python -m scenarios.build_sheet     # refresh the browser payload
python -m validation.sheet_engine_parity   # engine vs sheet, year-end horizons
```

```python
from scenarios.engine import load_scenario, latest_states, debt_projections, run
```

## The chain

The construction was chosen on a real-time backtest of year-end forecasts against no
change, 1992-2026 (`validation/FINDINGS_yield_forecast_backtest.md`). It is built so a
forecast moves with the house's calls and not with the market's noise, without
ignoring a repricing: today's yield and premium are common to every scenario, so the
spreads between scenarios move only when their calls do.

| step | what happens |
|---|---|
| **Rate leg** | today's priced ten-year average converging on your policy path: `P̄(h) = curve + [view(h) − curve]·(1 − λ^h)`, **no drift of the market's own** |
| **US premium** | the driver model's fair value on *your* drivers, plus today's unexplained part fading at `ρ = 0.5` a year |
| **Spread premia** | today's premium + the US premium's move at the forecasting beta + their own drivers, the inflation channel scaled to what yields have shown; their own unexplained part held |
| **Return** | Leibowitz constant-maturity: `R = y₀ + (Δy/H)·((H+1)/2 − D)`, deflated by your own inflation path for the real figure |

`curve = y₀ − TP₀` is what the market prices today and `view(h)` your policy path
averaged over `[h, h+10]`, **put on the curve's own short-rate basis**: the house US
rate is the fed funds *lower bound* and the fitted 3m runs 0.152 above the target
midpoint, so +0.277; Bund 3m vs the ECB deposit rate −0.21, gilt 6m vs Bank Rate −0.19,
JGB 1y vs the call rate −0.10, each net of the policy change realised over the tenor
(`validation/policy_basis.py`). Unconverted, the gap would be booked as premium.

Why each piece (all real time, error relative to no change at 4-5 quarters):

* **No market drift.** Adding the curve's priced drift made forecasts worse in every
  market; the drift did not materialise on average.
* **US residual fades at 0.5.** Leaning against the part of the US premium the drivers
  cannot explain cut error 7-9% (both halves of the sample), and passed into gilts and
  Bunds at their beta it helped there too (4-7%). Reverting to a trailing mean did not.
* **Spread markets' own residual is held.** Leaning against it was 5-22% worse.
* **Japan's forecasting beta is 0.15** (0.48 in the level fit): best on JGB yield
  changes since 2006 and before yield-curve control; 0.48 was no better than no change.
  The quarters since YCC ended point higher, so it is under review. Japan reaches the
  US *only* through this beta: its relative drivers are read against the US held at
  today.
* **Inflation response.** A correct inflation call paid through the premium at the
  model's coefficient in the US, at 0.75 of it in the euro area, and not at all in the
  UK (0.15) or Japan (0) — there it works through the policy path.
* **A correct rate call is worth most** (7-23% less error); the house view is phased in
  at 20% a year.

The parameters live in `scenarios.engine` (`RESIDUAL_FADE_US`, `POLICY_BASIS`,
`FORECAST_BETA`, `INFLATION_RESPONSE`) and travel to the sheet in its payload, so the two
cannot drift apart. At `h = 0` no view is in yet and `y(0)` is today's yield.

**λ is fitted, and it is not about persuasion.** It used to be asserted at 0.8 and
justified as how quickly your view reaches the market. That justification does not
survive the way these scenarios are actually used: everything except the baseline is a
*conditional* statement — "if the world is stagflationary, how do bonds price?" — and
there is no disagreement to decay in a conditional. What there is, is **learning**. The
state has changed and the market has no foresight, so it prices the state as it unfolds.

That is measurable, and `validation/growth_drawdowns.py` measures it on episodes where
a state did materialise:

| regime | λ | priced at 1y | at 3y |
|---|---|---|---|
| calm | **0.40** | 60% | 94% |
| inflationary | **0.685** | 32% | 68% |
| *previously asserted* | *0.80* | *20%* | *49%* |

A calm state is priced about three times faster than the old constant assumed; an
inflationary one more slowly, because the central bank cannot ease into it and the path
stays murky. One constant cannot say both, so each scenario is classified by **its own
mean inflation over the grid** (`engine.convergence_for`) rather than by name — the rule
travels to scenarios nobody has written yet.

**The baseline is the exception, and it is a real one.** A conditional scenario has a
forcing event: the state happens, and the market cannot not see it. The baseline is our
*forecast*, made on the same information the market already has. Nothing arrives to
force agreement — the market has seen what we have seen and concluded something else —
and there is a genuine chance we are simply wrong. So it keeps the slower disagreement
rate, `λ = 0.8`, and a scenario carries `is_view` to say which of the two it is.

**Your typed paths are untouched.** At long horizons the yield still converges on
exactly what you entered — policy at 1% still delivers 1%. Only the route there is
faster. The dial above the grid overrides λ by hand (`0` = perfect foresight, `1` =
hold today's priced level); leaving it on *fitted* uses the table.

## The two inflation channels

Your single inflation path feeds two drivers that sit at different levels today, so no
flat path zeroes both — and that is correct rather than a defect.

* **Long-run expectation** is the average of your path over the bond's ten-year life.
  Today's reading uses your path too, so any gap to the surveyed expectation sits in the
  unexplained premium and fades with it (US) or is held (spread markets).
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
`build_sheet.py`) and agrees with `scenarios.engine` to **0.05 basis points** at every
year-end horizon (`validation/sheet_engine_parity.py`); the gap is payload rounding. Export writes a CSV in exactly the format
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

The Baseline is the **house forecast**; the other four are **conditional** ("if the
world is stagflationary, how do bonds price?") and are priced faster, because the state
happens and the market sees it. A forecast is only as good as its calls: nothing
published beats no change at a year, and the one piece of the model that does on its
own — fading today's unexplained US premium — is built in. Everything else is your
rate and inflation calls, priced.

## Named scenarios

The sheet holds five independent scenarios, switched by the bar above the market tabs:
**Baseline**, **Goldilocks**, **High Growth High Inflation**, **Stagflation** and
**Hard Landing**. Each is a complete copy of the input state for every market, so
switching is instant and nothing leaks across; a dot on the bar marks the ones that
have been moved off their opening values.

`Baseline` is the house forecast for growth, CPI inflation and the policy rate in each
market (`scenarios/data/*_scenarios.csv`, Baseline rows; revised 2026-10-02).

**Definitions.** Inflation is the calendar-year average of monthly year-on-year rates
(seasonally adjusted headline CPI; for the German market, euro-area HICP); growth the average of the four quarterly year-on-year
rates; the policy rate a year-end level. Each inflation value therefore governs its own
year. For the current year the published months are history: the sheet prices the
remaining months at the rate that delivers the stated average,
`(12 x average - sum of published months) / months remaining`, using seasonally
adjusted CPI (`macro/fetch_headline_cpi.py`: BLS CPI-U SA; ECB euro-area HICP SA for the
German market; OECD all-items SA for Japan; ONS CPI for the UK, adjusted here with STL
because the ONS publishes no SA CPI), refreshed every daily run. The sheet shows
the split under the grid, and does the same for growth from the published GDP quarters
(`macro/fetch_quarterly_gdp.py`; euro-area GDP for the German market).

The alternative scenarios' 2026 growth and inflation are the Baseline's, except
Stagflation and Hard Landing, which state their 2026 rates for the **unpublished periods
only** — the `growth_rest` / `inflation_rest` columns, which `load_presets` turns into a
`remaining` block; the sheet rebuilds the annual figure on every data refresh. Their
2026 year-end policy rates are capped by the meetings left: 50bp a meeting in
Stagflation and Hard Landing, 25bp in the other two, or the stated target if nearer.
The US policy rate is the fed funds **lower bound**. Beyond the
sheet's 2026-31 grid the 2031 values hold, which matches the forecast's flat tails to
2045. The fiscal balance follows the official projection, re-aligned to the growth row.
After changing presets, refresh the deck's scenario slides with
`python -m presentation.build_scenario_slides`.

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

**The headline chart shows all five.** The ten-year under every scenario, plotted
against the curve's own forward ten-year &mdash; the benchmark any view has to beat.
Two lines are meant to be read first: the **market forward** and **Baseline**. The
other three sit in a recessive ink at half the weight, present for the range they
imply rather than for their individual paths, and named by direct label at the line
end rather than by colour. That is a deliberate choice, not a shortcut: there are not
five hues that stay apart under colour-blindness alongside the history and market
colours already in use (checked, not eyeballed &mdash; the worst pair came out at
&Delta;E 9 against a floor of 15), so the label carries identity and hue is spent on
the two series the chart is actually about. Whichever scenario is *selected* is drawn
at full weight in the ink so that editing it is visible. The window starts in 2010
rather than 1995 like the other charts, because a ten-year fan of five paths is
unreadable squeezed into the right quarter of the plot.

**What the reader gets.** A comparison table above the grid runs all five side by side
for the selected market: the ten-year yield at three and ten years, the difference from
Baseline, and the annualised nominal and real return. What the curve already prices for
the same bond is the dashed line on the chart above, so it is not repeated as a column.

**The table shows the ten-year and only the ten-year.** A second row per scenario
doubles its height for a number that is a cross-check rather than the argument. The
thirty-year fit is still built and still shipped in the payload, and `project` still
takes an alternate tenor, so restoring the row is a display change and not a modelling
one.

That fit is a genuine second estimate of the same model on the thirty-year premium
(`viz/data_daily.parquet` carries US 30y nominal from 1985; R² 0.90 on 488 months),
with the expected-rate leg averaged over thirty years rather than ten so the identity
`y = premium + expected average short rate` still closes on today's yield to a
hundredth of a basis point. The drivers are the same series at the same definitions —
expected inflation is a ten-year survey measure, the fiscal driver the five-year-ahead
projection — because that is what the coefficients are estimated against at every
tenor; holding them fixed is what lets the thirty-year loadings be read as *how much
more the long end moves for the same shift*. They differ substantially: the long end
loads less on the LEVEL of expected inflation (+0.72 against +1.01) and far more on
inflation UNCERTAINTY (+1.81 against +0.81), which is the sensible reading — a
thirty-year bond is paid for the width of the distribution, not its centre.

Only the US has one. The other three price off the US premium through the relative
model in `model/tp_relative.py`, and that model is fitted at ten years, so a
thirty-year Bund or gilt would need its own relative fit — not claimed here. Typed numbers
persist in the browser between visits and are dropped automatically if the underlying
data vintage changes, so a stored scenario can never mix last month's market path with
this month's charts. Export writes all five as one CSV, each block readable by
`load_scenario` unchanged.
