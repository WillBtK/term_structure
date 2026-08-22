# Can the driver models be trusted forward? — out-of-sample evidence

`validation/tp_scenario_oos.py`. Two tests: fit frozen at end-2015 and driven with
realized drivers for a decade; and expanding-window annual refits projecting 36 months
ahead, scored against the random walk. Reproduce with
`python -m validation.tp_scenario_oos`.

|        | frozen-2015 RMSE | end-point error | 3y model RMSE | 3y RW RMSE | ratio |
|--------|-----------------|-----------------|---------------|------------|-------|
| **US** | 0.72pp          | −0.25pp         | 0.95pp        | 0.97pp     | 0.98  |
| **UK** | 0.58pp          | −0.33pp         | 0.95pp        | 1.00pp     | 0.95  |
| **DE** | 1.44pp          | −1.17pp         | 1.25pp        | 1.00pp     | 1.24  |
| **JP** | 1.69pp          | −2.56pp         | 0.95pp        | 0.50pp     | 1.88  |

## Verdict

**Credible as a conditional scenario mapping; not an unconditional forecaster — and it
must never be presented as one.** The user supplies the macro scenario; the model
prices it. Matching the random walk at three years (US 0.98, UK 0.95) is the expected
result: nothing in this literature beats no-change unconditionally at that horizon
(Thornton–Valente 2012). Anchoring is what makes that safe — an unchanged scenario
returns today's premium exactly, so the model adds no drift of its own, only the
translation of deliberate scenario deviations.

## Why DE and JP fail the frozen test

Identification, not specification. Frozen at 2015, Germany's fit had seen **zero
months of PSPP** (it began March 2015) and Japan's had never seen BoJ holdings above
~30% of GDP; both extrapolated coefficients into driver ranges outside their samples.
Japan's random-walk RMSE of 0.50pp is yield-curve control itself — no-change was
enforced policy, unbeatable and uninformative as a benchmark. Today's fits contain a
complete QE expansion **and** contraction, exactly the variation those coefficients
needed. This defence is fair for the ownership channel and should be retired the day
either market's coefficient moves materially on new data.

## External validity (the checks an IC can verify independently)

- QE sensitivity **−6.5bp per % of GDP** vs the event-study range of ~4–8bp
  (Ihrig et al. 2018; Williams 2014 survey).
- Fed and foreign-official coefficients land within a whisker of each other
  (−0.069 / −0.064) with no constraint forcing it — consistent with one mechanism:
  price-insensitive duration absorption.
- Fiscal at +1.4bp (US, fitted) to +2.5bp (calibrated) per debt point vs Laubach
  (2009) 3–4bp and the CBO review's 2–3bp.
- US-level-plus-spread structure per the spillover literature (Rey;
  Miranda-Agrippino–Rey; Diebold–Li–Yue; Bauer–Neely — US shocks move foreign yields
  mostly through term premia).

## Standing caveats for use

1. Present sensitivities, never forecasts. Scenarios carry the uncertainty (agreed:
   no bands to the IC).
2. Japan is directional only: β on the US premium is 0.74 full-sample vs 0.12 on
   post-2010 changes. With YCC ended the forward case is migration back toward the
   higher number — a judgement, not an estimate.
3. Overlapping 36m windows mean the rolling RMSEs carry wide standard errors
   (n = 10–19 per market).

---

# How persistent is the unexplained premium?

Full anchoring — carrying today's residual whole to every horizon — asserts that every
shock to the term premium is permanent. For a fair-value model that is an uncomfortable
claim: a temporary spike would be projected forward for a decade. The opposite, reverting
to the model's own fair value, asserts a mispricing: for the US today, that the ten-year
yield is **71bp rich**. Neither is obviously right, so it was measured.

`validation/tp_scenario_oos.py::residual_persistence_grid`. Real-time annual refits, so
the residual is the one a user would actually have seen. RMSE in pp:

| horizon | market | n | ρ=1.0 | ρ=0.9 | ρ=0.8 | ρ=0.7 | ρ=0.5 | ρ=0.0 |
|---|---|---|---|---|---|---|---|---|
| 3y | US | 19 | 0.95 | 0.86 | 0.80 | 0.76 | 0.72 | **0.71** |
| | UK | 10 | 0.84 | 0.80 | 0.78 | **0.77** | 0.77 | 0.77 |
| | DE | 14 | 1.10 | 1.06 | **1.05** | 1.05 | 1.06 | 1.08 |
| | JP | 14 | 0.75 | 0.71 | 0.69 | **0.68** | 0.69 | 0.71 |
| 10y | US | 12 | 0.94 | 0.79 | **0.78** | 0.78 | 0.78 | 0.78 |
| | UK | 3 | 0.77 | 0.66 | 0.64 | 0.64 | 0.64 | **0.64** |
| | DE | 7 | **1.10** | 1.43 | 1.57 | 1.62 | 1.63 | 1.64 |
| | JP | 7 | **1.11** | 1.17 | 1.22 | 1.24 | 1.25 | 1.25 |
| | *mean* | | 0.944 | **0.935** | 0.941 | 0.943 | 0.943 | 0.945 |
| | *worst cell* | | **1.11** | 1.43 | 1.57 | 1.62 | 1.64 | 1.64 |

## What it says

**The two markets whose fair value is well identified want reversion; the two that lean
on borrowed and calibrated coefficients want none.** The US and UK improve at both
horizons, materially (US 3y 0.95 → 0.71). Germany and Japan are indifferent at three
years and clearly prefer full anchoring at ten — reverting *to* an unreliable level is
worse than staying at the observed one. Averaged over all eight cells the choice is
close to a wash, while the worst cell degrades from 1.11 to 1.64 as reversion increases.

So the honest verdict is that the evidence does not license a strong default in either
direction. **ρ = 0.9 a year** is the aggregate optimum: a half-life near six and a half
years, keeping 73% of the residual at three years and 35% at ten. It concedes the
conceptual point — no shock is treated as permanent — without asserting a large
mispricing, and it is exposed as a control on the scenario sheet rather than buried.
A user who judges a particular shock temporary should turn it down.

## A trap worth recording

Estimated **in sample**, the residual looks far more mean-reverting than it is: a
full-sample AR(1) puts the US half-life at four months and a positive residual predicts
the premium falling over the next three years with t = −15.5. That is largely an
artefact — the regression forces the residual orthogonal to the drivers over the whole
span, including the future. The real-time numbers above are three to ten times less
enthusiastic about reversion, and they are the ones to quote.
