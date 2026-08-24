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
relationship. **198** annual observations, six markets, 1981-2024.

Two history extensions matter for the result. Germany's 10y yield comes from the
**monthly Bundesbank Svensson parameters** before 1997 (`macro.tp_history`), and the UK's
industrial production from the **ONS's own K222** ("IOP: B-E: PRODUCTION: CVMSA",
quarterly, 1948 Q1+) rather than the OECD quarterly series (1997+). Without them DE and
UK began only in 2007 — inside a single correlation regime — and both showed no
relationship at all. The German splice is clean: the reconstructed monthly 10y matches
the production daily value at the seam to **0.000pp**, confirming they are the same curve
object.

The UK series is read straight from the ONS rather than through FRED's mirror of it
(`GBRPROINDMISMEI`), for two reasons. The mirror is a **stale vintage ending 2024-03**,
where the primary source runs to 2026 Q2; and the ONS publishes the **quarterly**
aggregate itself, so the UK now arrives at the same frequency as the other five and needs
no resampling assumption of ours. Where the data is settled the two are the same series —
year-on-year growth correlates 0.999 pre-1970 and 1.000 for 1970-90 — and they diverge
only around COVID, where revisions have not reached the mirror.

The realised US SBC reproduces the documented history: +0.11 to +0.21 through the
1980s-90s, turning negative around 2000, −0.23 to −0.33 in 2011-20, back to −0.04 now.

## Result 1: the sign structure holds

| | pooled | CA | CH | DE | JP | UK | US |
|---|---|---|---|---|---|---|---|
| σ_π² coeff | **+0.030** | +0.006 | +0.135 | +0.087 | **−0.026** | +0.047 | +0.025 |
| σ_g² coeff | **−0.004** | −0.025 | −0.002 | −0.005 | −0.002 | −0.028 | −0.012 |
| R² | 0.481 | 0.833 | 0.577 | 0.724 | 0.374 | 0.876 | 0.484 |

Growth-vol is negative in **all six**; inflation-vol is positive in **five of six**.
Japan is the exception, and plausibly so — deflation, ZIRP and YCC make JGBs behave
unlike a normal duration asset.

## Result 2: the one-variable ratio form is better *and* simpler

| | pooled | CA | CH | DE | JP | UK | US |
|---|---|---|---|---|---|---|---|
| slope on σ_π/σ_g | **+0.617** | +1.483 | +0.727 | +0.978 | +0.601 | +0.490 | +0.810 |
| R² | **0.534** | 0.637 | 0.475 | 0.670 | 0.228 | **0.926** | 0.536 |

Pooled R² rises from 0.481 to **0.534** on one variable instead of three — the three
structural terms are collinear over a 40-year panel and the cross-term is weakly
identified. Every market has a positive slope.

US R² of 0.536 against AQR's reported ~70% is a partial replication, and the gap is
explicable: IP rather than GDP growth, price rather than total returns, an OECD share
index rather than MSCI, and a duration-approximated bond return.

## Result 3 — the important caveat: this explains the REGIME, not the wiggles

| | intercept | slope | R² | n |
|---|---|---|---|---|
| ≤2000 | +0.034 | +0.245 | **0.238** | 64 |
| 2001+ | −0.275 | +0.753 | **0.334** | 134 |

Within either era the model explains 24-33%, against 53% pooled. **Most of the pooled
fit comes from the level shift between eras, not from tracking within them.** AQR say the
same of their own version ("less successful at explaining short-term fluctuations").

This bounds the use case precisely, and it happens to match what the scenario framework
needs: it is a **terminal-state** tool. It can answer "in a world that settles at 4%
inflation, do bonds still hedge?" It cannot answer "where is the correlation next
quarter?" Do not let it be used for the second question.

## Result 4: the full chain

Terminal core inflation → (Okun) inflation uncertainty → (ratio) correlation, with growth
volatility held at the panel median (4.13pp, IP-based):

| terminal core inflation | inflation uncertainty | σ_π/σ_g | implied SBC |
|---|---|---|---|
| 0.0% | 0.61pp | 0.15 | −0.13 |
| 1.0% | 0.61pp | 0.15 | −0.13 |
| 2.0% | 0.91pp | 0.22 | −0.09 |
| 2.5% | 1.12pp | 0.27 | −0.06 |
| **3.0%** | 1.32pp | 0.32 | **−0.03** |
| 4.0% | 1.73pp | 0.42 | +0.03 |
| 5.0% | 2.14pp | 0.52 | +0.10 |
| 6.0% | 2.55pp | 0.62 | +0.16 |

**The correlation crosses zero at roughly 3% terminal core inflation.** Below it bonds
diversify; above it they stop. That is a single, defensible sentence for an investment
committee, and it drops out of the data rather than being imposed.

Note the flat 0-1% region is the Okun hockey stick showing through: no extra inflation
risk is priced until inflation clears the anchor.

## Caveats

1. **σ_g must be measured the same way it was estimated.** The ratio is scale-sensitive:
   IP-based growth vol (median 4.13pp) is far higher than GDP-based would be, so
   substituting a GDP-based σ_g into a fit estimated on IP would roughly double the ratio
   and badly overstate the implied correlation. The chain is internally consistent; do
   not mix proxies.

2. **Growth volatility is held constant in the projection.** Defensible — AQR's own
   forward-looking discussion treats inflation uncertainty as the swing factor — but it
   is an assumption, and σ_g is doing real work in the denominator.

3. **Regime-level tool only.** See Result 3.

4. **Japan is the one genuine miss** — still the wrong sign on the structural form and
   the weakest ratio fit (R² 0.228) after the history extensions. Deflation, ZIRP and YCC
   plausibly explain it, but any Japan scenario output should carry the caveat explicitly.

   DE and UK, by contrast, *were* misses and are no longer: both were sample-span
   artefacts, and extending their histories moved them from ~0 to +0.72/+0.88 (structural
   R²). That is the cleanest available evidence that the earlier zeros reflected a lack of
   regime variation rather than a failure of the model.

5. **Overlapping windows.** Fits are on annual observations, but 10y rolling windows mean
   the effective sample is smaller still than n=198 suggests. Treat R² as descriptive.

---

## Extension — the US correlation back to 1881 (Shiller)

**Why.** Everything above rests on a panel starting in 1981, which contains exactly one
inflation regime change: the disinflation. A relationship identified off a single
turning point is a story about that turning point. The claim being made — that the
stock-bond correlation is positive when inflation is the dominant risk and negative when
growth is — is a claim about regimes, and needs more than one regime to test.

**How.** `macro/shiller.py` reads Shiller's `ie_data.xls` (monthly S&P price, dividend,
CPI and the long government yield from 1871), builds the equity total return directly,
and approximates the constant-maturity ten-year bond return as `y/12 − D_mod·Δy` with
duration recomputed each month. Rolling 120-month correlation: **1881-01 to 2026-08,
1,748 observations.**

| Era | Regime | Correlation |
|---|---|---|
| 1881–1914 | Classical gold standard | +0.11 |
| 1915–1929 | First World War inflation | +0.02 |
| 1930–1945 | Depression and war | +0.08 |
| 1946–1965 | Bretton Woods | −0.08 |
| **1966–1982** | **The Great Inflation** | **+0.28** |
| 1983–1999 | Disinflation | +0.23 |
| **2000–2020** | **Inflation targeting** | **−0.26** |
| 2021– | Inflation returns | −0.02 |

Eras are named for what *inflation* did, not for the correlation, so the table is not
drawn around its own answer. The window is **centred** for this table: a trailing
ten-year window dates a regime by when it closed, which puts the Great Inflation's
correlation in the 1980s and 90s and makes the disinflation look like the higher-
correlation era (+0.32 against +0.19) when the opposite is true. The plotted series
keeps the trailing convention, because that is what an investor could have observed at
the time and because it runs to today.

**Verdict.** The sign pattern the ratio model predicts is present in every regime the
extended sample covers, and the two largest-magnitude cells are the two the theory
picks out. This is materially stronger evidence than the 1981+ panel could give, and it
is the reason the deck now shows this series rather than the 40-year one.

**Caveats.** (a) The bond return is a first-order approximation — no convexity, par
bond, no roll down a non-flat curve. Adding convexity moves the rolling correlation by
at most 0.006 (Feb 1980; mean 0.0008), so it is omitted rather than carried as false
precision. (b) Pre-war equity and CPI data are monthly averages of sparse underlying
observations; the 19th-century cells should be read as "small and unstable", not as
precise. (c) US only. The cross-sectional evidence across six markets is what the
scatter above provides; this adds depth in one market, not breadth.
