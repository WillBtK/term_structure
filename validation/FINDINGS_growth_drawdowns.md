# What the curve actually does in a growth drawdown

**The question.** The scenario engine converges monotonically toward a terminal fair
value. Real downturns do not — yields fall, trough, and recover. Before deciding
whether that is a defect in the engine or in the intuition, ask the data: how far does
the ten-year move, when does it trough, how much comes back, and **which leg does it**.

**Method.** `validation/growth_drawdowns.py`. An episode starts at the unemployment
rate's **cycle trough** — the lowest reading in a contiguous run of months that each
precede a rise of ≥1.0pp within 24 months. Sahm-style, applied unchanged to
US / DE / UK / JP / CA, dating the last good month rather than a committee's later
verdict. Episodes within 48 months are collapsed. Changes are from the episode start,
in basis points, from `viz/data_daily.parquet`; the identity
`observed = expected + term premium` holds to 0.0bp in every market, and each tenor's
three series are averaged over rows where all three exist, so the split is exact.

> **The dating rule matters more than it looks.** An earlier cut took the *first*
> qualifying month rather than the run's minimum. On a slow expansion that dates the
> episode years before the actual low and materially changed the answer — the ten-year
> trough moved from −57bp to −38bp and the premium's sign flipped. The cycle trough is
> the right anchor and is what is reported here.

## A drawdown is not one object

Splitting on core inflation at the episode start (4% threshold) separates two
responses that have almost nothing in common. **This is the same distinction as the
scenario sheet's Hard Landing versus its Stagflation**, and averaging them together
produces a number that describes neither.

### Low-inflation drawdowns — 12 episodes (2000, 2007, 2019 vintages)

| months | 2y | 10y | 10y: expected | 10y: premium |
|---|---|---|---|---|
| 6 | −54 | −36 | −20 | −17 |
| 12 | −88 | −45 | −39 | −6 |
| **18** | **−185** | **−80** | **−77** | **−2** |
| 24 | −177 | −48 | −87 | +39 |
| 36 | −128 | −22 | −62 | +40 |

A clean V. The ten-year troughs at **−80bp, eighteen months in**, and is back to
−22bp by three years. At the trough it is **almost entirely the expectations leg**
(−77 of −80); the premium does essentially nothing (−2bp), then turns *positive* as
the recovery prices in.

### High-inflation drawdowns — 7 episodes (1968, 1973, 1979, 1989–90, 2023)

| months | 2y | 10y | 10y: expected | 10y: premium |
|---|---|---|---|---|
| 6 | +56 | **+48** | +18 | +31 |
| 12 | −5 | +31 | −44 | +75 |
| 18 | −16 | +46 | −56 | +102 |
| 24 | −22 | +46 | −52 | +98 |
| 36 | −188 | −28 | −140 | +112 |

**Yields go up.** The ten-year is +46bp two years into the downturn, and the premium
is doing all of it (+102bp) against an expectations leg that is already falling. There
is no V; there is an inverted one, and the fall only arrives in year three.

## What this settles

**1. Do not add a cyclical growth channel to the premium.** Now rejected three ways:
the regression (unemployment gap, its 12-month change, and its level all fail inside
the US premium regression — R² 0.930 → 0.931, none significant, the near-significant
one with the wrong sign); the low-inflation event study (premium moves −2bp at the
trough, i.e. not at all); and the high-inflation one (premium moves +102bp — the
opposite sign to flight-to-quality). The premium responds to **inflation**, not to
growth, which is exactly what the driver model already says.

**2. The engine has the right mechanism for both scenarios.** In a hard landing the
shape is the expectations leg, which is the user's policy path — so a V-shaped input
produces a V-shaped output, and the sheet already does this (4.74 → 4.65 at h=1 →
4.73 → 4.84 → 5.01 on a test path). In stagflation the premium does the work through
expected inflation and inflation uncertainty, which the model has. Nothing structural
is missing.

**3. The dilution argument is confirmed quantitatively.** In low-inflation drawdowns
the two-year falls **2.3×** as far as the ten-year (−185 vs −80 at the trough). A
dated two-year event averaged into a ten-year expected short rate is necessarily
small. If the committee wants to see the recession in a bond, it has to be a short
bond.

**4. The reporting horizons straddle the trough.** Compare shows 3y and 10y. The
low-inflation trough is at **18 months** and two thirds of it has unwound by 36. A
1y/2y column would show it; 3y systematically understates it.

**5. Calibration checks for the current scenarios.** History puts a hard landing at
roughly **−80bp on the ten-year at 18 months, −22bp at three years**. The sheet's Hard
Landing currently shows −118bp at three years, because its input path holds policy at
1% forever — a permanent-stagnation view, not a shock-and-recovery one. Stagflation
shows +478bp at ten years against a historical +46bp at two years; the input path is
far more extreme than anything in the sample, so this is a statement about the input,
not a model error — but it is the number that will be challenged.

## The learning speed, fitted

The scenario engine's weight on your path is `1 - lam**h`. It used to be justified as
how *persuasive* your view is, with `lam = 0.8` asserted. That justification does not
survive the conditional reading: a scenario other than the baseline is not a forecast
anyone has to be persuaded of, it is the statement "if the world is X, how do bonds
price?". There is no disagreement to decay. What there is, is **learning** — the state
has changed and the market prices it as it unfolds, with no foresight.

That is measurable, and these episodes measure it: states that did materialise, learned
in real time. Fitting `1 - lam**h` to the market's own ten-year expected-rate leg over
the rising limb of each regime (`learning_speed()`):

| regime | λ | priced at 1y | at 3y | at 10y |
|---|---|---|---|---|
| low inflation | **0.40** | 60% | 94% | >99% |
| high inflation | **0.685** | 32% | 68% | 98% |
| *previously asserted* | *0.80* | *20%* | *49%* | *89%* |

The market prices a calm state roughly **three times faster** than the old constant
assumed, and an inflationary one more slowly than a calm one — because the central bank
cannot ease into it and the path stays genuinely murky. One constant cannot express
that, which is why the engine now classifies each scenario by **its own mean inflation
over the projection grid** (`engine.convergence_for`) rather than by name.

The baseline is treated as a state like any other. Under a learning reading there is
only one process: the market's prior is today's curve — which is why `h=0` still
reproduces today's yield, since `1 - lam**0 = 0` for any λ — and λ is how fast
information arrives, not whose opinion it is.

**What this does not change.** Your typed paths are untouched. At long horizons the
yield still converges on exactly what you entered; policy at 1% still delivers 1%. Only
the route there is faster. On the current US scenarios the three-year column moves by
+50 to +309bp and the ten-year by −10 to +75bp, which is the right signature for a
change that is about the path and not the destination.

## Caveats

Small samples: 12 low-inflation and 7 high-inflation episodes, and the 30y and 60-month
cells thinner still. The 60-month column is contaminated by the secular decline in
rates across the sample — read it as directional. Term premia before each market's
regime start use extrapolated parameters (see the README's regime discussion); the US
1968 and 1973 episodes sit in that window. Japan's expectations leg is anchored near
the bound, so its decomposition loads almost everything onto the premium — a property
of the ELB, not of Japanese recessions. The 4% inflation split is a judgement; the two
groups separate cleanly enough that the exact threshold is not doing the work, but it
has not been tuned or tested for robustness. The fitted lambdas inherit the
normalisation caveat above: taking the trough as the full move understates it if the
state persists past the window, and a larger full move implies a HIGHER lambda (slower
learning). That assumption, not the decay rate, is the thing to put in front of a
committee.
