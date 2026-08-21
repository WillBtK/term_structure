# Okun's "mirage of steady inflation": level → inflation uncertainty

**Purpose.** The scenario framework forecasts only growth, inflation and policy rates.
The term premium compensates inflation *uncertainty*, not the inflation *level* — and
uncertainty is not forecast. Okun (1971, BPEA) argued that steady inflation is a mirage:
as inflation rises it also becomes more variable and less predictable. If that holds, the
level the scenario already fixes implies the uncertainty it does not.

This note tests it on core CPI across the six Atlas markets.

**Core, not headline** — deliberately. The object is a slow-moving inflation *regime*.
Headline injects energy swings a trailing window cannot distinguish from a genuine
regime change, so a headline reading would report "uncertainty" that is really the oil
price.

Reproduce with `python -m validation.okun_mirage`.

## Data

Monthly core CPI indices, US from BLS (`CPILFESL`, 1957+, current), the other five from
the OECD MEI "all items non-food non-energy" series (1955-1970 starts). Regime pairs are
the trailing **10-year** mean (`level`) and standard deviation (`vol`) of year-on-year
core inflation, both in percentage points. 3,883 monthly observations → **327 annual**,
1965-2025.

Fits are on annual observations. The rolling windows overlap, so a monthly fit would
imply roughly twelve times the sample size actually present.

The panel spans genuine deflation (JP −0.5%) to double digits (UK 13.7%), which is what
identifies the relationship.

## Result: a hockey stick, not a smooth convexity

Non-parametric first — mean 10y core-inflation vol by 10y mean level, no functional form
imposed:

| mean level | −0.2% | 0.4% | 1.6% | 2.5% | 3.5% | 4.9% | 6.9% | 9.7% |
|---|---|---|---|---|---|---|---|---|
| **mean vol** | 0.62 | 0.67 | 0.63 | 1.29 | 1.57 | 2.06 | 2.74 | 4.19 |

Volatility is **flat at ~0.65pp from deflation all the way to ~2%**, then rises roughly
linearly. Fitted:

```
vol = 0.61 + 0.41 × max(level − 1.25, 0)      R² = 0.775, n = 327
```

against R² = 0.757 linear and 0.760 quadratic. Pooled corr(level, vol) = **0.869**.

**This shape is the economically interesting part.** It says moving from 0% to 2%
inflation earns almost *no* extra inflation risk premium, while moving from 2% to 4%
earns a lot. That is coherent: while inflation is at or below target the anchor holds and
uncertainty is small-but-irreducible; it is credibility erosion *above* target that
generates uncertainty.

## Robustness

**Every market, same sign** (kink fixed at 1.25):

| | CA | CH | DE | JP | UK | US |
|---|---|---|---|---|---|---|
| floor | 0.75 | 0.65 | 0.72 | 0.66 | 0.65 | 0.28 |
| slope | 0.27 | 0.51 | 0.32 | 0.50 | 0.42 | 0.41 |
| R² | 0.693 | 0.729 | 0.573 | 0.865 | 0.897 | 0.773 |

**Not driven by any one market.** Leave-one-out slopes span 0.38-0.44, floors 0.54-0.67.

**Window.** A 5y regime window gives the same shape with a lower floor (0.52) and
slightly flatter slope (0.33, R² 0.707). 10y is the reported spec because a terminal-state
scenario is asking about a regime, not a cycle.

## Caveats — read before relying on this

1. **The breakpoint is weakly identified.** R² is essentially flat across kinks of
   0.5-1.5 (0.767-0.775). Treat it as "somewhere around 1-1.5%", not 1.25. Anywhere in
   that range is defensible; a kink at 2.5%+ is not (R² falls away).

2. **The slope is era-dependent, and this is the largest open question.** Pre-1995:
   slope 0.40. Post-1995: slope **0.71** with a higher kink (1.75). The modern
   relationship looks *steeper* — but post-1995 contains little high-inflation variation,
   so that slope is identified almost entirely off 2021-24 and is correspondingly fragile
   (R² 0.521 vs 0.680). The full-sample 0.41 is better identified; the modern subsample is
   a live argument that it understates. **Run both as a scenario sensitivity** rather than
   picking one.

3. **Realised volatility is a proxy for ex-ante uncertainty, not the same thing.**
   What a premium compensates is uncertainty *looking forward*; what is measured here is
   variability *looking back*. `OkunParams.anchored_to()` exists for this reason: it
   re-levels the map onto a directly observed forward-looking reading (e.g. current
   cross-sectional dispersion of long-horizon SPF forecasts) while leaving the
   historically-estimated slope untouched.

4. **This is a mapping, not a causal claim.** High inflation and high inflation
   volatility are plausibly both symptoms of poor monetary control. That is fine for the
   use here — projecting uncertainty from a scenario's inflation level — but it is not
   evidence that lowering inflation *causes* lower uncertainty.

5. **Coverage.** The OECD feed was discontinued: most series end 2025, Japan's at
   2021-06. Immaterial for estimation (identification comes from the high-inflation era),
   and it is why anchoring to a current observed reading is a separate step rather than
   read off the tail of this panel.

## What it delivers to the scenario model

| terminal core inflation | implied uncertainty |
|---|---|
| 0.0% | 0.61pp |
| 1.0% | 0.61pp |
| 2.0% | 0.91pp |
| 2.5% | 1.12pp |
| 3.0% | 1.32pp |
| 4.0% | 1.73pp |
| 5.0% | 2.14pp |
| 6.0% | 2.55pp |

A scenario's terminal inflation now implies an inflation uncertainty, which is the input
the stock-bond-correlation channel (Brixton et al. 2023, JPM) requires and which the
scenario cannot otherwise supply.

**Reference:** Okun, A. (1971), "The Mirage of Steady Inflation", *BPEA* 2:1971.
Mechanism in Ball, L. (1992), "Why does high inflation raise inflation uncertainty?",
*JME*.
