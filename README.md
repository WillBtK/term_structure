# Cross-Country Term Premium Estimation

## Objective

Estimate government bond term premia consistently across the United States,
Germany, Japan, and the United Kingdom, using a single model specification
across all four markets, while correctly handling periods of zero or negative
short-term rates (Japan since the mid-1990s; Germany/euro area since 2014).

## Model

**Arbitrage-free Nelson-Siegel (AFNS)**, extended with the **Christensen-Rudebusch
shadow-rate specification** for the zero/negative lower bound.

- Base repo to fork: https://github.com/snejens/arbfree_dynamic_nelson_siegel
  (AFNS via Kalman filter / MLE, following Christensen, Diebold, and Rudebusch 2011)
- Shadow-rate extension follows: Christensen, Jens H. E., and Glenn D. Rudebusch.
  2015. "Estimating Shadow-Rate Term Structure Models with Near-Zero Yields."
  *Journal of Financial Econometrics* 13 (2): 226-59.
- One model, one codebase, run identically across all four countries. Do not
  fork the model per country — only the data ingestion differs.

## Effective lower bound (ELB): values and rationale

The ELB is a **calibrated input, not an estimated parameter** — this is standard
practice in the literature (Christensen-Rudebusch note it can be replaced with
"some appropriate, possibly time-varying, negative epsilon"; Bauer and Rudebusch
2016 and Krippner 2015a both document that shadow-rate estimates are highly
sensitive to this choice). Treat it as a sensitivity parameter, not a fact.

| Country | Central ELB value | Sensitivity band | Basis |
|---|---|---|---|
| US | -0.35% | fixed | Standard currency-storage-cost floor |
| UK | -0.35% | fixed | Standard currency-storage-cost floor |
| Germany | -1.0% | fixed | Reversal-rate estimate (ECB) |
| Japan | -0.75% | -1.0% to -0.5% | Reversal-rate estimate (BoJ range) |

Use **one constant bound per country for the full sample** — do not build a
time-varying or step-function bound unless sensitivity testing shows the
constant-bound result is implausible (e.g., the shadow rate pinned to the floor
through the negative-rate period with no room to reflect subsequent tightening).
If that escalation is needed, see: Kortela, Tomi. 2016. "A Shadow Rate Model
with Time-Varying Lower Bound of Interest Rates." Bank of Finland Research
Discussion Paper 19/2016.

Away from the bound, the shadow-rate model collapses to the standard AFNS
model regardless of the bound value chosen — so the constant bound does not
need to be "switched on" at a specific date; it can run across the full
sample without distorting periods when rates are comfortably positive.

Run each country's shadow rate at 2-3 bound values within its band and confirm
the profile is stable in shape (even if the level shifts) before treating any
single run as final.

## Data sources, per country

| Country | Source | Method | What to pull |
|---|---|---|---|
| US | Federal Reserve Board (GSW) | Svensson, parametric | Daily parameters (beta0-3, tau1-2); reconstruct yield grid on demand |
| Germany | Bundesbank | Svensson, parametric | Daily parameters since Aug 1997 (monthly since Sept 1972) |
| UK | Bank of England | VRP spline, non-parametric | Full published yield curve grid, daily since Jan 1979, out to 20y |
| Japan | MOF / BoJ | Spline-based (per BIS Papers no. 25) | **Confirm current source and method directly before building — not verified as closely as the other three.** |

Target a common maturity grid across countries for comparability, e.g.
3m / 6m / 1y / 2y / 3y / 5y / 7y / 10y.

## Validation benchmarks

- **US, non-shadow-rate AFNS baseline**: SF Fed "Treasury Yield Premiums"
  (https://www.frbsf.org/research-and-insights/data-and-indicators/treasury-yield-premiums/).
  Same model family (plain AFNS, no ZLB constraint), same GSW input data —
  use this to validate the AFNS implementation itself before adding the
  shadow-rate extension. Expect close tracking throughout.
- **All four countries, directional cross-check**: Leo Krippner's published
  series (https://www.ljkmfa.com/visitors/). Methodology differs (his current
  model is three-factor, OIS-augmented, survey-based natural rate) — use for
  plausibility on level and turning points, not as a target to match exactly.
- Expect the shadow-rate model to **diverge** from both plain-AFNS and from
  Krippner during each country's ZLB/negative-rate window — that divergence
  is expected, not a bug.

## Estimation cadence (operational design)

Do not force a full nonlinear Kalman filter MLE re-optimization every day —
convergence risk is real for shadow-rate models. Instead:

- Re-estimate state-space parameters on a **slower cadence** (weekly or monthly).
- Run the Kalman filter **forward daily** with the fixed parameters to produce
  the daily-updated premium series.
- **Warm-start** each re-estimation from the previous parameter vector.
- Log the likelihood at convergence; **fall back to the last valid parameter
  set** if an optimization run fails, rather than publishing a broken estimate.

## Build sequence

1. **US** — best data, cleanest external benchmark (SF Fed). Validate plain
   AFNS here first, then add the shadow-rate extension.
2. **Germany** — next-cleanest data; negative-rate period present but shorter
   and less extreme than Japan's.
3. **Japan** — longest and most demanding ZLB/negative-rate history. Highest
   payoff, highest estimation risk. Confirm data source/method before starting.
4. **UK** — mildest ZLB period of the four; useful final check that the
   pipeline generalizes without further adjustment.

## Repo structure

```
/model          # shared AFNS + shadow-rate extension (forked from snejens/arbfree_dynamic_nelson_siegel)
/us             # data fetcher + country-specific config
/germany
/japan
/uk
/validation     # SF Fed and Krippner benchmark comparisons
/scheduler      # cadence logic: weekly/monthly re-estimation, daily filter-forward, warm-start, fallback
```

## Session guidance

- Work through the build sequence one country at a time; do not attempt all
  four unsupervised in a single session.
- Flag any convergence failure, implausible shadow-rate path, or benchmark
  divergence explicitly rather than silently proceeding.
- The ELB values above are provisional and open to revision on sensitivity
  testing — do not treat them as fixed facts.
