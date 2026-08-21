# The inflation shape of the term premium — the production relationship

Reproduce with `python -m validation.high_inflation_identification` for the
identification evidence; the fitted shape lives in `model/tp_inflation.py`.

## The specification, and why it mixes horizons on purpose

| | sample | why |
|---|---|---|
| **inflation shape** | full history (1961-74 onward per market) | the high-inflation branch cannot be seen any other way |
| **balance-sheet control** | regime-clean window only | it is a control on a recent, well-measured episode; it means nothing earlier |

This is a deliberate trade, not an oversight. Inside each market's regime-clean
production window there are 295 monthly observations above 3.5% inflation — but **every
one of them is a disinflation**, 1984-2000. Nothing from 2021-24 clears the bar. Fitting
the shape only on clean data would mean never observing an inflation *ascent* and never
observing anything above about 6%. The premium series before each market's regime-clean
start is noisier — it comes from filtering the production model over data its parameters
were not estimated on — but it is the only place the upper branch exists. A noisier
regressand over the range that matters beats a clean one that omits it.

The balance sheet gets the opposite treatment for the same reason: it is only measurable,
and only meaningful, inside the modern window.

## The shape

Two kinks, both **imposed** rather than searched:

- **2%** — the inflation target in all four markets. The story is "flat while inflation
  is at target, rising once it clears", and the threshold is one every reader already
  knows.
- **5%** — where the marginal effect drops. Searched freely it lands at 4.5-5.0% for the
  US and UK; imposing 5% costs almost nothing and stops thinner markets choosing
  implausible break points (a free search puts Japan's at 2.75% and implies a +4.59%
  premium at 3% inflation).

The **slope above 5% is shared across markets**, estimated where it is identified. The US
and UK have 158 and 177 observations above 5%; Germany has **34**, and left to itself its
upper slope comes out negative — implying the premium *falls* as inflation rises past 5%,
and turning the German 3-year premium to −0.39% at 6%. That is not a finding, it is a fit
running out of data. Each market keeps its own intercept and its own 2-to-5% sensitivity,
which is where the real cross-market differences are and where every market has data.

## Monotonicity is imposed, not hoped for

The unconstrained fit says the marginal effect above 5% is **+0.339** at ten years and
**+0.107** at three — the second close enough to zero that another point of inflation
would add 11bp. Taken literally that means inflation risk stops being priced once
inflation is high, which no investor believes and no committee would accept.

It is also exactly what our sample would produce whether or not it were true. **Every
high-inflation observation we have is a disinflation** with improving credibility, and
that is the wrong analogue for a scenario going the other way.

So the high-inflation slope is **floored at one third of each market's own 2-to-5%
slope**. This is a judgement, applied openly and in one place (`HIGH_SLOPE_FLOOR`). It
costs almost nothing in fit — the ten-year US R² moves 0.822 to 0.824 — and it makes the
curve non-decreasing in inflation **by construction**, which is asserted in the tests
rather than left to luck. It binds at three years for every market and at ten years for
the US and Germany.

## Fitted, 10-year term premium

| market | n | at target | slope 2-5% | slope >5% | R² | 3% | 4% | 5% | 6% | 8% |
|---|---|---|---|---|---|---|---|---|---|---|
| US | 659 | −0.07% | +1.135 | +0.386 | 0.824 | +1.07 | +2.20 | +3.34 | +3.72 | +4.50 |
| UK | 570 | +0.52% | +0.883 | +0.339 | 0.767 | +1.41 | +2.29 | +3.17 | +3.51 | +4.19 |
| DE | 646 | +0.96% | +1.412 | +0.472 | 0.546 | +2.37 | +3.78 | +5.19 | +5.67 | +6.61 |
| JP | 622 | +1.26% | +0.807 | +0.339 | 0.479 | +2.06 | +2.87 | +3.68 | +4.02 | +4.69 |

## Fitted, 3-year term premium

| market | n | at target | slope 2-5% | slope >5% | R² | 3% | 4% | 5% | 6% | 8% |
|---|---|---|---|---|---|---|---|---|---|---|
| US | 781 | −0.13% | +0.589 | +0.212 | 0.591 | +0.46 | +1.05 | +1.64 | +1.85 | +2.27 |
| UK | 570 | +0.09% | +0.406 | +0.155 | 0.496 | +0.50 | +0.91 | +1.31 | +1.47 | +1.78 |
| DE | 646 | +0.34% | +0.842 | +0.283 | 0.371 | +1.18 | +2.02 | +2.87 | +3.15 | +3.72 |
| JP | 620 | +0.37% | +0.382 | +0.139 | 0.381 | +0.75 | +1.13 | +1.51 | +1.65 | +1.93 |

## The sentence this is for

> The term premium is flat while inflation sits at target. Between 2% and 5% it rises
> roughly 0.8 to 1.4 percentage points per point of inflation, depending on the market.
> Above 5% the marginal effect drops to about a third of a point — by then the regime has
> already repriced.

## Memory

Inflation enters as a two-year EWMA of core. That is long enough to mean "the prevailing
regime" and short enough that the 2021-24 episode registers as the high-inflation event
it was rather than being averaged to 3.1%. **The memory parameter does not affect the
scenario output** — a settled inflation path is its own EWMA — it only decides which
historical episodes count as high-inflation when fitting.

## What this is and is not

1. **Storytelling with support, not a description.** Projecting a premium ten years out
   from a macro scenario carries errors far larger than the third decimal of a slope. The
   shape is meant to be defensible and roughly right, and the numbers should be read to
   one decimal place at most.

2. **Level, not dynamics — and now demonstrated rather than asserted.** The German
   relationship fitted on 1984+ and extrapolated back to 1972-83 lands within **0.73pp**
   of the independent regime-switching exhibit on the level, with a correlation of
   **+0.111** on the path. It anchors where the premium settles; it does not track it.

3. **The upper branch comes from disinflations, and is floored for that reason.** Every
   high-inflation observation is an inflation *descent* with improving credibility. An
   ascent is a different experiment we have not observed. The floor is the honest
   response: it says we do not believe the flat reading, and states by how much.

4. **The balance sheet is a control, not a driver.** It improves the QE-era fit modestly,
   is worth roughly 2bp under a neutral policy, and belongs nowhere near a committee
   slide. See `FINDINGS_supply.md`.

5. **Germany and Japan are the weaker fits** (R² 0.48-0.55 at 10y against 0.77-0.82 for
   the US and UK), and both borrow their upper branch. Treat their high-inflation
   scenarios as the least reliable output of the tool.
