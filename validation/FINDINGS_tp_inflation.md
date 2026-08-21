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

## Fitted, 10-year term premium

Shared marginal effect above 5%: **+0.339**

| market | n | at target | slope 2-5% | R² | 2% | 3% | 4% | 5% | 6% |
|---|---|---|---|---|---|---|---|---|---|
| US | 659 | −0.08% | +1.158 | 0.822 | −0.08 | +1.08 | +2.24 | +3.40 | +3.73 |
| UK | 570 | +0.52% | +0.883 | 0.767 | +0.52 | +1.41 | +2.29 | +3.17 | +3.51 |
| DE | 646 | +0.96% | +1.417 | 0.547 | +0.96 | +2.37 | +3.79 | +5.21 | +5.55 |
| JP | 622 | +1.26% | +0.807 | 0.479 | +1.26 | +2.06 | +2.87 | +3.68 | +4.02 |

## Fitted, 3-year term premium

Shared marginal effect above 5%: **+0.107**

| market | n | at target | slope 2-5% | R² | 2% | 3% | 4% | 5% | 6% |
|---|---|---|---|---|---|---|---|---|---|
| US | 781 | −0.15% | +0.636 | 0.590 | −0.15 | +0.48 | +1.12 | +1.76 | +1.86 |
| UK | 570 | +0.08% | +0.466 | 0.498 | +0.08 | +0.55 | +1.01 | +1.48 | +1.59 |
| DE | 646 | +0.34% | +0.849 | 0.375 | +0.34 | +1.19 | +2.03 | +2.88 | +2.99 |
| JP | 620 | +0.36% | +0.416 | 0.405 | +0.36 | +0.78 | +1.20 | +1.61 | +1.72 |

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

3. **The upper branch comes from disinflations.** Every high-inflation observation in the
   clean windows is an inflation *descent* with improving credibility. An ascent with
   deteriorating credibility is a different experiment that we have not observed, and the
   flattening above 5% may not survive one.

4. **The balance sheet is a control, not a driver.** It improves the QE-era fit modestly,
   is worth roughly 2bp under a neutral policy, and belongs nowhere near a committee
   slide. See `FINDINGS_supply.md`.

5. **Germany and Japan are the weaker fits** (R² 0.48-0.55 at 10y against 0.77-0.82 for
   the US and UK), and both borrow their upper branch. Treat their high-inflation
   scenarios as the least reliable output of the tool.
