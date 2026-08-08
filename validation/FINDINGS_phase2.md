# Phase 2 findings — US shadow-rate AFNS (Christensen-Rudebusch)

Full-sample estimation: `us.run_shadow.run_shadow(freq="monthly", start="1987-01-01",
elb=-0.0035)` — EKF/QML, 475 monthly obs, **converged** (loglik 20286),
**yield fit 3.7 bp**. Math verified term-by-term against CR WP 2013-07 (see
`model/shadow_rate.py`). Estimated: λ=0.512, κ=[0.033, 0.125, 0.436],
θ=[6.07%, −3.34%, −2.85%], σ=[0.66%, 1.11%, 2.71%], h=6.4 bp.

## Internal validation (README's core expectation) — PASS

The ELB "wedge" (fitted ELB yield − same-state unbounded/plain-AFNS yield),
mean |·| over maturities:

| regime | mean &#124;wedge&#124; |
|---|---|
| away from the bound (normal) | **1.8 bp** |
| ZLB windows (2008-15, 2020-21) | **21.5 bp** |

So the model **collapses to plain AFNS away from the bound and diverges during
the ZLB** — exactly as the README requires. Year-by-year the wedge is ~0.5-1.6 bp
in 1998-2007 and 2022-2026, peaking ~30 bp (2011-12) and ~45 bp (2020).

## External directional cross-checks

| benchmark | corr (full sample) | our trough | their trough |
|---|---|---|---|
| Effective fed funds (FRED) | 0.986 *(normal-times)* | — | — |
| Wu-Xia (Atlanta Fed) | **0.973** | −0.08% | −2.99% |
| Krippner (ljkmfa.com) | **0.942** | −0.08% | −4.14% |

- **Turning points / direction: strong.** Our shadow rate correlates 0.97 / 0.94
  with Wu-Xia / Krippner over the full sample and 0.986 with fed funds when rates
  are positive — the README's "plausibility on turning points" is well met.
- **Level at the ZLB trough: our shadow rate is much shallower.** It bottoms at
  ~0 (min −0.08%, 2009-10; above fed funds in ~76% of ZLB months) versus Wu-Xia
  −3% and Krippner −4%. The README anticipates divergence here ("methodology
  differs; use for plausibility on level and turning points, not a target to
  match exactly"), but our shadow rate is shallow even relative to CR's own AFNS
  estimates.

## Interpretation and honest limitation

Away from the bound the model behaves exactly as intended (collapses to AFNS;
tracks policy). During the ZLB the effective-lower-bound effect shows up mainly
in the **long-yield wedge** (10-45 bp) rather than in a deeply negative shadow
**short** rate: our GSW short-end inputs stayed slightly positive and the fit
(3.7 bp) attributes the ZLB largely to the option/vol term, leaving the shadow
policy rate near zero. Wu-Xia and Krippner (different models/inputs) imply much
deeper shadow rates. This is the known high model-sensitivity of shadow-rate
*levels* (README: ELB/shadow estimates are "highly sensitive"), but the
shallowness is a limitation worth refining — candidate levers: a full (non-
diagonal) K^P, the ELB value, or the weak identification of the shadow level
below the bound (our synthetic test already showed ~90 bp state RMSE deep below
the bound).

## Not yet done
- **ELB sensitivity** (README: run 2-3 bound values, confirm the profile is
  stable in shape) — deferred; each value is a full EKF re-estimation.
- Daily-frequency estimation (monthly used for tractability of the nonlinear EKF).
