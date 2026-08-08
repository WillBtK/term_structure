# Cross-Country Term Premium Estimation

## Objective

Estimate government bond term premia consistently across the United States,
Germany, Japan, and the United Kingdom, using a single model specification
across all four markets, while correctly handling periods of zero or negative
short-term rates (Japan since the mid-1990s; Germany/euro area since 2014).

## Model

**Arbitrage-free Nelson-Siegel (AFNS)**, extended with the **Christensen-Rudebusch
shadow-rate specification** for the zero/negative lower bound.

- Engine: [`TermStructureModels.jl`](https://github.com/econPreference/TermStructureModels.jl)
  (MIT) — a Gaussian affine model (JSZ constrained by the AFNS restriction of
  Christensen, Diebold, and Rudebusch 2011) with native term-premium
  decomposition. We initially built on the Python repo
  `snejens/arbfree_dynamic_nelson_siegel`, but replaced it: it tracked the SF Fed
  benchmark far worse (2y term-premium correlation 0.43 vs 0.85) and carried no
  license. See `model/README.md` and `validation/FINDINGS_phase1e.md`.
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

**Deriving the ELB from data (optional).** The bound can also be *estimated* by
ML as a free parameter (Christensen-Rudebusch), via `ekf.estimate(...,
estimate_elb=True)`. It is only identified when yields actually approach the
bound: for the US — which barely touched it — the ML ELB is weakly identified and
the calibration + sensitivity band remains the reporting basis; for Germany and
Japan, which went clearly negative, ML estimation is well identified and is the
primary method. The revealed bound abroad (SNB/ECB/BoJ reached roughly
-0.75/-0.50/-0.10% without triggering large-scale cash conversion) is the better
anchor than a retail currency-storage-cost calculation: the relevant floor is the
cost of warehousing *institutional-scale* cash, which "digital cash" makes more,
not less, binding. See `validation/FINDINGS_phase3.md`.

## Sample robustness, historical scope, and regimes (Phase 3)

Like ACM (and unlike survey-anchored models such as D'Amico-Kim-Wei), a
yields-only affine model's expected-rate/premium split hinges on the estimated
persistence of the short rate under P, which is weakly identified and biased
toward *too little* persistence in finite samples — so the premium is understated,
most severely in the highly persistent 1970s. We address this **without** giving
up portability (surveys are US-rich but patchy across the four markets, so they
stay out of the core; they remain an optional US-only overlay):

- **Bias-correction** (Bauer-Rudebusch-Wu 2012; `model/bias_correction.py`): a
  Kilian bootstrap correction of the VAR(1) persistence, yields-only and portable.
  On the US it lifts the short-to-medium term-premium correlation with SF Fed
  substantially (e.g. 2y 0.28→0.79) — see `validation/FINDINGS_phase3.md`.
- **Historical scope to 1961** (`us/reconstruct.py`): the pre-1980 four-parameter
  Nelson-Siegel regime is reconstructed and the early long end is kept ragged
  (10y from 1971); the state-space filter (`model/ekf.py`) masks missing
  maturities per date.
- **Regime-switching** (`model/regime_switching.py`): a Markov-switching AFNS
  (Kim 1994 filter) lets the P-dynamics switch, giving the high-inflation regime
  its own persistence, volatility, and term-premium level — the stagflation
  question.

Engine note: the production plain-AFNS term premium is validated against SF Fed
via `TermStructureModels.jl`; the **Python EKF** (`model/ekf.py`) is the workhorse
for the shadow-rate, historical, bias-corrected, and regime extensions because it
exposes `K^P`, handles ragged data, and can estimate the ELB. One model family,
one codebase, across all four countries.

## Regime-switching extension & the 2008 splice

The full 1961-present US term premium is produced by **two separately-estimated
models joined at a fixed splice date**, *not* one unified model:

- **Pre-2008-12-31 — regime-switching AFNS** (`model/regime_switching.py`): a
  2-regime Markov-switching AFNS on the P-dynamics, estimated by the Kim (1994)
  filter (Hamilton regime filter + Kalman + collapsing). Common arbitrage-free
  Q-loadings (each regime is AFNS-arbitrage-free; regime-switching risk is left
  unpriced — the tractable choice, full Dai-Singleton-Yang being the ideal). This
  leg is **frozen history** — never re-estimated. Output = regime-mixed factors
  and term premium (weighted by the smoothed regime probabilities).
- **2008-12-31 onward — shadow-rate AFNS** (`us/run_shadow.py`): the **live**
  model (re-estimate slow, filter forward daily), which respects the ELB. Its term
  premium uses the expected-average *floored* short rate under P
  (`term_premium.expected_floored_short_rate_yields`).

**Why two models, not one.** Combining the Kim collapsing approximation with the
shadow-rate `max()` nonlinearity in a single estimation compounds two
approximate/nonlinear techniques (much harder to converge reliably) and breaks the
weekly/monthly re-estimate + daily-filter-forward cadence above. The splice date
**2008-12-31** is the ZLB onset — the first date the shadow model does something
the regime model can't; both legs agree on the positive-rate run-up, so the seam
is clean, and it cleanly separates frozen history from the live model.

**Common λ.** The regime leg fixes the NS decay λ to the shadow leg's estimate so
the level/slope/curvature factors are on the same basis at the seam;
`us/splice.py::continuity_check` reports the two legs' L/S/C agreement over the
1987-2008 overlap and the factor/term-premium values on both sides of the seam.
Away from the bound the shadow term premium collapses to the plain-AFNS term
premium, so the legs coincide over the overlap by construction.

**Volcker window (Oct 1979 - Oct 1982).** The non-borrowed-reserves operating
procedure produced short-rate volatility from a different policy instrument. The
2-regime model captures this window *within* its high-volatility regime
(P(high-vol) ≈ 0.98 there) — a **third dedicated regime is not needed**.

**Validation.** The Kim filter was validated before wiring it in — against an
exact path-enumeration filter (collapsing correct to ~1e-4 loglik / ~1e-5 regime
prob) and against the Ang-Bekaert (2002) / Levant-Ma (2017) empirical pattern
(clean low-vol post-1985; high-vol confined to the 1973-75 oil shock and 1979-82
Volcker). See `validation/FINDINGS_kim_validation.md` and
`validation/FINDINGS_phase4.md`.

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
