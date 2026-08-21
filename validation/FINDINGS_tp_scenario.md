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
| corr(TP 3y, SBC) | **+0.58** | +0.73 | +0.49 | −0.03 | +0.84 | +0.00 | +0.69 |
| corr(TP 10y, SBC) | **+0.59** | +0.80 | +0.53 | −0.14 | +0.82 | +0.01 | +0.73 |

**The DE/UK zeros are a sample-span artefact, not a contradiction.** Their panels start
in 2007 — the Bund daily curve begins 1997 and UK industrial production 1997, and the
10-year windows consume the first decade — so both sit *entirely inside one correlation
regime*. That is exactly the case `FINDINGS_sbc.md` predicts the model cannot speak to.
Every market whose sample spans the regime change (US 1981-, CA/JP/CH 1996-98-) shows a
strong positive relationship. Fitted parameters therefore use **US, CA, JP, CH only**.

## Fair value

| tenor | fair value | R² |
|---|---|---|
| 3y | TP\* = +0.67 + 2.88·SBC − 0.016·gap | 0.387 |
| 10y | TP\* = +1.71 + 4.99·SBC − 0.041·gap | 0.417 |

The SBC loading is **larger at 10y than 3y** (4.99 vs 2.88), which is the monotonicity
check this model has to pass: longer duration is more exposed to the inflation-risk
channel. It was not imposed.

The output gap enters with the right sign (slack lowers the premium) but adds almost
nothing — R² 0.336→0.337 at 3y. That is an honest negative on a *crude* proxy: this uses
detrended industrial production, not the forecast-appended two-sided filter intended for
production. Worth retrying with the proper gap; not worth claiming now.

## Error correction — and a correction to an earlier guess

| tenor | λ (per year) | half-life | closed by 3y | closed by 10y |
|---|---|---|---|---|
| 3y | 0.493 | 1.0y | **87%** | 100% |
| 10y | 0.315 | 1.8y | **68%** | 98% |

I earlier guessed that a 3-year horizon would not be long enough to converge. The data
says that is right for the 10y tenor (only ~two-thirds of the way) but wrong for the 3y
tenor (~87%, essentially there). So:

- **The 10-year-horizon answer is fair value** at both tenors — starting point is
  irrelevant.
- **The 3-year-horizon answer still depends on where the premium is today**, materially
  so at the 10y tenor.

## The output

σ_g at the panel median (4.43pp), US starting points:

**10y term premium**

| terminal core inflation | σ_π | implied SBC | fair value | TP @3y | TP @10y |
|---|---|---|---|---|---|
| 1.0% | 0.61pp | −0.16 | 0.91% | 0.93% | 0.91% |
| 2.0% | 0.91pp | −0.10 | 1.23% | 1.15% | 1.23% |
| 3.0% | 1.32pp | −0.01 | 1.67% | 1.44% | 1.65% |
| 4.0% | 1.73pp | +0.08 | 2.10% | 1.73% | 2.07% |
| 5.0% | 2.14pp | +0.16 | 2.53% | 2.03% | 2.49% |

**3y term premium**

| terminal core inflation | fair value | TP @3y | TP @10y |
|---|---|---|---|
| 2.0% | 0.39% | 0.28% | 0.39% |
| 3.0% | 0.64% | 0.49% | 0.64% |
| 4.0% | 0.89% | 0.71% | 0.89% |

Roughly **+0.43pp of 10y term premium per 1pp of terminal core inflation** above the Okun
kink, and about +0.25pp at the 3y tenor. Below ~1.25% inflation the hockey stick flattens
and further disinflation buys nothing.

## Caveats

1. **This explains ~40% of term premium variation, not all of it.** R² 0.39-0.42. The
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

6. **DE and UK have no usable estimate** of their own and currently inherit the
   long-sample parameters. Flag that in any DE/UK scenario output.
