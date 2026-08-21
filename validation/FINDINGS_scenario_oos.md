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
