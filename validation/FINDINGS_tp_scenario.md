# Scenario → term premium: the assembled chain

```
terminal core inflation
  → (Okun 1971)            inflation uncertainty      FINDINGS_okun.md
  → (Brixton et al. 2023)  stock-bond correlation     FINDINGS_sbc.md
  → (fair value)           equilibrium term premium   ← this note
  → (error correction)     term premium at horizon h  ← this note
```

Reproduce with `python -m validation.tp_scenario_fit`.

## The final link

Term premium against the stock-bond correlation, annual observations, 1981-2023:

| | pooled | CA | CH | DE | JP | UK | US |
|---|---|---|---|---|---|---|---|
| corr(TP 3y, SBC) | **+0.60** | +0.73 | +0.49 | +0.68 | +0.84 | +0.61 | +0.69 |
| corr(TP 10y, SBC) | **+0.63** | +0.80 | +0.53 | +0.76 | +0.82 | +0.62 | +0.73 |

**All six markets are positive, and getting there was the finding.** In the first cut DE
and UK correlated ~0.00 and were excluded. Both turned out to be sample-span artefacts
rather than contradictions: their panels began only in 2007 and therefore sat entirely
inside one correlation regime. Extending them fixed both —

- **UK**: industrial production from `GBRPROINDMISMEI` (1948+) instead of the OECD
  quarterly series (1997+). Sample 2007→1989, correlation +0.01 → **+0.62**.
- **Germany**: the binding constraint was never the premium but the **yield**, since the
  correlation is built from bond returns and the daily Bund curve starts 1997. The 10y is
  now reconstructed from the **monthly Bundesbank Svensson parameters** (1972+). Sample
  2007→1982, correlation −0.14 → **+0.76**. The splice is verified clean: reconstructed
  monthly and production daily agree at the seam to **0.000pp**.

This is the strongest evidence in the whole chain that the earlier zeros reflected
missing regime variation rather than a broken model — the prediction was made *before*
the data was extended, and extending it moved both markets exactly as predicted. Fitted
parameters now use **all six**.

## Fair value

| tenor | fair value | R² |
|---|---|---|
| 3y | TP\* = +0.63 + 2.97·SBC − 0.004·gap | 0.359 |
| 10y | TP\* = +1.65 + 5.12·SBC − 0.027·gap | 0.408 |

n=197 (was 154 before the history extensions).

The SBC loading is **larger at 10y than 3y** (5.12 vs 2.97), which is the monotonicity
check this model has to pass: longer duration is more exposed to the inflation-risk
channel. It was not imposed.

The output gap enters with the right sign (slack lowers the premium) but adds almost
nothing: R² 0.358→0.359 at 3y (+0.001) and 0.397→0.408 at 10y (+0.012). That is an honest
negative on a *crude* proxy — this uses detrended industrial production, not the
forecast-appended two-sided filter intended for production. Worth retrying with the proper
gap; not worth claiming now.

## Error correction — and a correction to an earlier guess

| tenor | λ (per year) | half-life | closed by 3y | closed by 10y |
|---|---|---|---|---|
| 3y | 0.417 | 1.3y | **80%** | 100% |
| 10y | 0.264 | 2.3y | **60%** | 95% |

I earlier guessed that a 3-year horizon would not be long enough to converge. The data
says that is right for the 10y tenor (60%, well short) but much less so for the 3y tenor
(80%). So:

- **The 10-year-horizon answer is fair value** at both tenors — starting point is
  irrelevant.
- **The 3-year-horizon answer still depends on where the premium is today**, materially
  so at the 10y tenor.

## The output

σ_g at the panel median (4.16pp), US starting points:

**10y term premium**

| terminal core inflation | σ_π | implied SBC | fair value | TP @3y | TP @10y |
|---|---|---|---|---|---|
| 1.0% | 0.61pp | −0.14 | 0.96% | 0.96% | 0.96% |
| 2.0% | 0.91pp | −0.09 | 1.21% | 1.11% | 1.20% |
| 3.0% | 1.32pp | −0.02 | 1.55% | 1.32% | 1.52% |
| 4.0% | 1.73pp | +0.05 | 1.89% | 1.52% | 1.84% |
| 5.0% | 2.14pp | +0.11 | 2.23% | 1.72% | 2.17% |

**3y term premium**

| terminal core inflation | fair value | TP @3y | TP @10y |
|---|---|---|---|
| 2.0% | 0.37% | 0.20% | 0.37% |
| 3.0% | 0.57% | 0.36% | 0.56% |
| 4.0% | 0.77% | 0.51% | 0.76% |

Roughly **+0.34pp of 10y term premium per 1pp of terminal core inflation** above the Okun
kink, and about +0.20pp at the 3y tenor. Below ~1.25% inflation the hockey stick flattens
and further disinflation buys nothing.

## Caveats

1. **This explains ~40% of term premium variation, not all of it.** R² 0.36-0.41. The
   macro-uncertainty channel is one driver among several — QE/QT and bond supply are the
   obvious absentees and are deliberately still to come. Do not present the output as a
   complete account.

2. **Persistent regressor, persistent regressand.** Both TP and SBC are near-unit-root
   over 40 years, so the level R² is flattered and conventional t-stats would be badly
   overstated (Bauer-Hamilton 2018). The error-correction form is the partial defence;
   proper small-sample inference has **not** been run and is the next validation step.

3. **σ_g scale sensitivity carries through the whole chain.** The correlation model is a
   ratio, so a differently-scaled growth-volatility proxy rescales everything downstream.
   Use the same IP-based measure the fit used.

4. **The Okun era-dependence propagates.** `FINDINGS_okun.md` records a full-sample slope
   of 0.41 against 0.71 post-1995. The 0.71 variant would roughly *double* the
   inflation sensitivity of every number above. Run it as the stated sensitivity.

5. **Regime-level tool.** Inherited from the correlation link: this answers "where does
   the premium settle in a world that looks like this", not "where is it next quarter".

6. **Japan remains the one market whose correlation link misfits** (see
   `FINDINGS_sbc.md`): its inflation-volatility coefficient has the wrong sign, plausibly
   because deflation, ZIRP and YCC make JGBs behave unlike a normal duration asset. Its
   *term premium* correlates strongly with the SBC (+0.82), so it is the macro→correlation
   step that is weak, not this one — but flag it in any Japan scenario output.
