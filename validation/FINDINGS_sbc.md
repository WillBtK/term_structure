# The stock-bond correlation channel: macro volatility → correlation → term premium

**Purpose.** A scenario model that maps macro forecasts straight to a term premium is
untestable end to end — you cannot check a premium you had to model in order to see. The
stock-bond correlation (SBC) is *observable*, so routing through it splits the problem
into two separately falsifiable links and puts a familiar object in front of the reader
("bonds stopped hedging") rather than an unfamiliar one.

This note tests the first link. Reproduce with `python -m validation.sbc_macro`.

## The model being tested

Brixton et al. (2023, *JPM*): equities and bonds have opposite-sign exposure to growth
news and same-sign exposure to inflation news, so

```
cov(s,b) = b_sg*b_bg*var(g)  +  b_spi*b_bpi*var(pi)  +  (cross term)
             ^ NEGATIVE                ^ POSITIVE
```

Growth volatility pushes the correlation down (bonds hedge); inflation volatility pushes
it up (bonds stop hedging). Testable prediction: **inflation-vol coefficient positive,
growth-vol coefficient negative.**

## Data

Realised SBC from trailing 10y correlation of monthly equity returns (OECD share price
indices) and 10y bond returns (carry − duration × Δy, from the Atlas's own yield panel).
σ_π from core CPI, σ_g from industrial production — IP rather than GDP because FRED's
real GDP starts 1991-95 for DE/JP/CH, *after* the regime change that identifies the
relationship. 154 annual observations, six markets, 1981-2023.

The realised US SBC reproduces the documented history: +0.11 to +0.21 through the
1980s-90s, turning negative around 2000, −0.23 to −0.33 in 2011-20, back to −0.04 now.

## Result 1: the sign structure holds

| | pooled | CA | CH | DE | JP | UK | US |
|---|---|---|---|---|---|---|---|
| σ_π² coeff | **+0.030** | +0.006 | +0.135 | +0.148 | **−0.026** | +0.004 | +0.025 |
| σ_g² coeff | **−0.004** | −0.025 | −0.002 | −0.002 | −0.002 | −0.017 | −0.012 |
| R² | 0.432 | 0.833 | 0.577 | 0.957 | 0.374 | 0.776 | 0.484 |

Growth-vol is negative in **all six**; inflation-vol is positive in **five of six**.
Japan is the exception, and plausibly so — deflation, ZIRP and YCC make JGBs behave
unlike a normal duration asset.

## Result 2: the one-variable ratio form is better *and* simpler

| | pooled | CA | CH | DE | JP | UK | US |
|---|---|---|---|---|---|---|---|
| slope on σ_π/σ_g | **+0.936** | +1.483 | +0.727 | +1.847 | +0.601 | +0.550 | +0.810 |
| R² | **0.530** | 0.637 | 0.475 | 0.814 | 0.228 | 0.660 | 0.536 |

Pooled R² rises from 0.432 to **0.530** on one variable instead of three — the three
structural terms are collinear over a 40-year panel and the cross-term is weakly
identified. Every market has a positive slope.

US R² of 0.536 against AQR's reported ~70% is a partial replication, and the gap is
explicable: IP rather than GDP growth, price rather than total returns, an OECD share
index rather than MSCI, and a duration-approximated bond return.

## Result 3 — the important caveat: this explains the REGIME, not the wiggles

| | intercept | slope | R² | n |
|---|---|---|---|---|
| ≤2000 | −0.046 | +0.468 | **0.241** | 33 |
| 2001+ | −0.309 | +1.006 | **0.237** | 121 |

Within either era the model explains only ~24%, against 53% pooled. **Most of the pooled
fit comes from the level shift between eras, not from tracking within them.** AQR say the
same of their own version ("less successful at explaining short-term fluctuations").

This bounds the use case precisely, and it happens to match what the scenario framework
needs: it is a **terminal-state** tool. It can answer "in a world that settles at 4%
inflation, do bonds still hedge?" It cannot answer "where is the correlation next
quarter?" Do not let it be used for the second question.

## Result 4: the full chain

Terminal core inflation → (Okun) inflation uncertainty → (ratio) correlation, with growth
volatility held at the panel median (4.43pp, IP-based):

| terminal core inflation | inflation uncertainty | σ_π/σ_g | implied SBC |
|---|---|---|---|
| 0.0% | 0.61pp | 0.14 | −0.16 |
| 1.0% | 0.61pp | 0.14 | −0.16 |
| 2.0% | 0.91pp | 0.21 | −0.10 |
| 2.5% | 1.12pp | 0.25 | −0.05 |
| **3.0%** | 1.32pp | 0.30 | **−0.01** |
| 4.0% | 1.73pp | 0.39 | +0.08 |
| 5.0% | 2.14pp | 0.48 | +0.16 |
| 6.0% | 2.55pp | 0.57 | +0.25 |

**The correlation crosses zero at roughly 3% terminal core inflation.** Below it bonds
diversify; above it they stop. That is a single, defensible sentence for an investment
committee, and it drops out of the data rather than being imposed.

Note the flat 0-1% region is the Okun hockey stick showing through: no extra inflation
risk is priced until inflation clears the anchor.

## Caveats

1. **σ_g must be measured the same way it was estimated.** The ratio is scale-sensitive:
   IP-based growth vol (median 4.43pp) is far higher than GDP-based would be, so
   substituting a GDP-based σ_g into a fit estimated on IP would roughly double the ratio
   and badly overstate the implied correlation. The chain is internally consistent; do
   not mix proxies.

2. **Growth volatility is held constant in the projection.** Defensible — AQR's own
   forward-looking discussion treats inflation uncertainty as the swing factor — but it
   is an assumption, and σ_g is doing real work in the denominator.

3. **Regime-level tool only.** See Result 3.

4. **Japan is a genuine miss** on the structural form and the weakest ratio fit. Any
   Japan scenario output should carry that caveat explicitly.

5. **Overlapping windows.** Fits are on annual observations, but 10y rolling windows mean
   the effective sample is smaller still than n=154 suggests. Treat R² as descriptive.
