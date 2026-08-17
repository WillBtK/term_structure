# Cross-Country Term Premium Estimation

## Objective

Estimate government bond term premia consistently across the United States,
Germany, Japan, the United Kingdom, Switzerland, and Canada, using a single model
specification across all six markets, while correctly handling periods of zero or
negative short-term rates (Japan since the mid-1990s; Germany/euro area since 2014;
Switzerland since 2015; Canada grazed its ~0.25% lower bound in 2009–10 and 2020–21).

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
- One model, one codebase, run identically across all five countries. Do not
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
| Germany | -0.80% (ML) | -1.0% calibration | ML-estimated; validates Praet/ECB reversal-rate region |
| Japan | -0.17% (ML) | start -0.20% | ML-estimated; lands at the NIRP policy rate (-0.10%) |

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
up portability (surveys are US-rich but patchy across the five markets, so they
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
one codebase, across all five countries.

## Two deliverables: the modern production TP and the historical exhibit

There is **no single 1961-present term-premium series** — deliberately. A single
model spanning 1961-today would estimate one physical mean θ and persistence K^P
over the whole span, pulling them toward the 1970s-80s high-inflation world and so
**contaminating *today's* term premium** (today's expected-rate path mean-reverts
toward a long-run level that no longer applies). That is the long-sample side of
the sample-sensitivity tradeoff; the short-sample side is finite-sample persistence
bias. We split the output by what each is good for:

### Modern / production term premium — `us/production_tp.py`
The **headline, cross-country-comparable** series: the **shadow-rate AFNS estimated
on 1987+** (ELB-respecting; reduces to plain AFNS when rates are high) **with
Bauer-Rudebusch-Wu bias-correction** of the P-persistence. The modern sample keeps
1970s-80s dynamics *out* of θ/K^P (no contamination); BRW then fixes the
finite-sample persistence bias a short sample would otherwise carry (in-sample it
lifts the short-end SF-Fed term-premium correlation from ~0.14 to ~0.83). Its term
premium uses the expected-average *floored* short rate under P
(`term_premium.expected_floored_short_rate_yields`). This is the "term premium
today" and the series used for cross-country comparison. Estimated on the
slow-cadence / daily-filter-forward schedule above.

### Historical / stagflation exhibit — separate, clearly labeled
The 1961+ regime-switching and single-regime work is kept as a **separate historical
exhibit**, *not* joined to the modern series:
- **Regime dating** — `model/regime_switching.py` (2-regime Markov-switching AFNS,
  Kim 1994 filter): *when* the high-inflation / high-volatility regime prevailed
  (1973-75 oil shock, 1979-82 Volcker; clean low-vol Great Moderation after).
- **Historical level** — a single-regime 1961+ bias-corrected fit puts the 1970s
  stagflation 10y term premium at a **moderate ~1.3%** (the premium peaked later,
  early-1980s to early-1990s). See `validation/FINDINGS_phase3.md`.

**Why not a 2008 splice.** An earlier design spliced a pre-2008 regime model to the
post-2008 shadow model at 2008-12-31. It produced a large term-premium discontinuity
at the seam (243 bp) *and* — via the pre-2008 leg — still risked contaminating the
modern estimate. The splice code is retained in `us/splice.py` as an internal option
(with the Kim-filter and continuity machinery), but it is **not the product**.
Keeping today's estimate on a modern sample is both cleaner and more correct.

**Kim-filter validation (for the exhibit).** Before any use, the Kim filter was
validated against an exact path-enumeration filter (collapsing correct to ~1e-4
loglik / ~1e-5 regime prob) and against the Ang-Bekaert (2002) / Levant-Ma (2017)
empirical pattern (Volcker P(high-vol) ≈ 0.98 — captured within the 2-regime model,
so no third regime needed). See `validation/FINDINGS_kim_validation.md`,
`FINDINGS_phase4.md`, and `FINDINGS_phase5.md`.

## Data sources, per country

| Country | Source | Method | What to pull |
|---|---|---|---|
| US | Federal Reserve Board (GSW) | Svensson, parametric | Daily parameters (beta0-3, tau1-2); reconstruct yield grid on demand |
| Germany | Bundesbank | Svensson, parametric | Daily parameters since Aug 1997 (monthly since Sept 1972) |
| UK | Bank of England | VRP spline, non-parametric | Full published yield curve grid (nominal + real), daily since 1979 (real 1985), out to 20y+ |
| Japan | MOF / BoJ | Spline-based (per BIS Papers no. 25) | MoF JGB spot yields, ragged panel 1986+ |
| Switzerland | SNB | `rendoblim` Svensson spot curve | Daily CHF spot rates, 1988+ (ragged long end from 1997-12) |
| Canada | Bank of Canada | Fitted zero-coupon curve (`lookup_yield_curve.php` CSV) | Daily CAD zero-coupon yields 0.25–30y, 1986+ (30y from 1991) |

US real (TIPS, GSW feds200805, 2004+) and UK real (index-linked gilts, 1985+) are
also pulled for the inflation-linked leg. US benchmark/survey series (ACM,
Kim-Wright, DKW, C&R, ATSIX, and the TIPS liquidity gauge) come via `/validation`.

Target a common maturity grid across countries for comparability, e.g.
3m / 6m / 1y / 2y / 3y / 5y / 7y / 10y.

## Validation benchmarks

- **US, non-shadow-rate AFNS baseline**: SF Fed "Treasury Yield Premiums"
  (https://www.frbsf.org/research-and-insights/data-and-indicators/treasury-yield-premiums/).
  Same model family (plain AFNS, no ZLB constraint), same GSW input data —
  use this to validate the AFNS implementation itself before adding the
  shadow-rate extension. Expect close tracking throughout.
- **All five countries, directional cross-check**: Leo Krippner's published
  series (https://www.ljkmfa.com/visitors/). Methodology differs (his current
  model is three-factor, OIS-augmented, survey-based natural rate) — use for
  plausibility on level and turning points, not as a target to match exactly.
- Expect the shadow-rate model to **diverge** from both plain-AFNS and from
  Krippner during each country's ZLB/negative-rate window — that divergence
  is expected, not a bug.
- **US external benchmarks** (`validation/benchmarks.py`, pulled alongside the
  rates): **ACM** (NY Fed, 5-factor), **Kim-Wright** (Fed Board, FRED
  `THREEFYTP1..10`), and **DKW** (D'Amico-Kim-Wei; also supplies real-TP,
  inflation-risk-premium and TIPS-liquidity references) — plus C&R above. The
  Atlas "Cross-check" panel overlays our US term premium on all four; the spread
  between these independent models is the honest uncertainty band on "the" US term
  premium. Each loader mirrors `sf_fed.py` (env-URL, parquet cache, cached-fallback).
- **FRED access** goes through `validation/fred_api.py` — the official API
  (`api.stlouisfed.org`) when `FRED_API_KEY` is set, else the keyless
  `fredgraph.csv` fallback. The official path is materially more reliable through
  the egress proxy (the keyless CSV times out) and exposes series/category
  metadata. (Switching to it also surfaced a latent bug: the Kim-Wright IDs are
  `THREEFYTP1..10`, not the zero-padded `THREEFYTP01` — now fixed and populated.)
- **ATSIX survey anchor** (`validation/benchmarks.py::load_atsix`): the
  Philadelphia Fed *Aruoba Term Structure of Inflation Expectations* — a free,
  monthly (1998+), survey-based **expected-inflation term structure** (it blends
  SPF + Blue Chip + Michigan, so it carries the Blue-Chip signal that is otherwise
  paywalled). Used to anchor the joint model's expected inflation (below); it is
  **US-only**, so it stays out of the portable cross-country core.
- **TIPS liquidity gauge** (`validation/tips_liquidity.py`): a standalone,
  price-based, **non-circular** liquidity measure — the Hu-Pan-Wang (2013) "noise"
  statistic (RMS deviation of every outstanding TIPS from a smooth fitted curve),
  built from FRED's individual TIPS securities (`DTP*`) via the official API. It
  peaks at ~45 bp in Dec-2008 (the post-Lehman TIPS freeze) and sits ~8 bp now —
  a free daily substitute for the vendor liquidity/RMSE series, shipped as its own
  Atlas panel and daily export. (No nominal-UST analogue: FRED carries individual
  TIPS but only the *smooth* nominal curve; a nominal cross-section is CRSP/Bloomberg,
  i.e. paid.)

## Estimation cadence (operational design)

Do not force a full nonlinear Kalman filter MLE re-optimization every day —
convergence risk is real for shadow-rate models. Instead:

- Re-estimate state-space parameters on a **slower cadence** (weekly or monthly).
- Run the Kalman filter **forward daily** with the fixed parameters to produce
  the daily-updated premium series.
- **Warm-start** each re-estimation from the previous parameter vector.
- Log the likelihood at convergence; **fall back to the last valid parameter
  set** if an optimization run fails, rather than publishing a broken estimate.

**This is what makes daily publishing cheap.** Only the *estimation* (the
Nelder-Mead optimization) is slow — minutes cold, faster warm-started — and it is
amortized on the weekly/monthly cadence in the background. The **daily** operation
is a single Kalman/EKF pass with *fixed, cached* parameters, which is fast:

| operation | cost (per country) |
|---|---|
| daily filter-forward, last-12-month window | **~20 ms** |
| full-history monthly re-filter + decompose (~40y) | ~0.5 s |
| full-history *daily* re-filter + decompose (~28y) | ~8 s |

So publishing the term premium daily across all five markets is well under a
minute even re-filtering the whole history each day, and milliseconds with a
rolling window. Implementation: `us.production_tp.ProductionParams` (the cached,
JSON-serializable parameter set produced by a re-estimation) + `filter_forward`
(the fast daily decomposition). Estimation `restarts` default is **1** (warm-started
re-estimation converges in one pass; multi-start is opt-in for cold fits).

**Implemented in `/scheduler`.** `daily_update.py` is the fast daily path (refresh
all source caches + benchmarks → `export_daily` → `thin_daily` → `build_artifact`,
cached params, no estimation); `reestimate.py` is the slow weekly re-estimation
(warm-started, falls back to the last valid `viz/params/*.json` on failure);
`supabase_export.py` writes/ upserts the Supabase-ready tables. A daily Routine
(cron, **00:00 UTC = 7pm EST**) runs the fast path, commits+pushes `viz/data.json`
and `viz/artifact.html`, and republishes the Atlas. See `scheduler/README.md`.

## Build sequence

1. **US** — DONE. Modern production TP (1987+ shadow AFNS + BRW) validated vs
   SF Fed; separate 1961+ historical/regime exhibit. See `validation/FINDINGS_phase5.md`.
2. **Germany** — DONE. Bundesbank Svensson data (`/germany`), reconstruction
   < 0.6 bp vs published spots; euro-era (1999+) production TP; the ELB is well
   identified and validates the −1.0% (Praet) calibration. See
   `validation/FINDINGS_germany.md`.
3. **Japan** — DONE. MoF JGB spot yields (`/japan`), ragged 1986+ panel; ML ELB
   −0.17% (validates the NIRP anchor); YCC measurement-downweight on the pegged
   10y/7y. The **realised 10y term premium is negative through QQE/YCC** (min
   −0.44% in 2020) and normalises post-YCC (now the highest of the five markets
   under the mean-reversion floor — see the correction note below). This leg
   surfaced decomposition fixes applied pipeline-wide (diagonal K^P +
   sample-mean θ for the expectations leg). See `validation/FINDINGS_japan.md`
   and `validation/FINDINGS_mean_reversion.md`.
4. **UK** — DONE. BoE Government Liability Curve (`/uk`, direct spot rates,
   1970→2026), mild fixed ELB (weakly identified). Modern production TP on the
   inflation-targeting era (1992+); 10y TP ~1.5% (among the highest, reflecting
   gilt/fiscal premia). See `validation/FINDINGS_uk.md`.
5. **Switzerland** — DONE. SNB `rendoblim` CHF spot curve (`/switzerland`,
   direct spot rates, 1988+, ragged long end from 1997-12), fixed ELB −0.75%
   (the revealed SNB floor). 10y term premium the **lowest of the five** (deepest
   safe-haven, currently negative). Added as the 5th market in the atlas.
6. **Canada** — DONE. Bank of Canada fitted **zero-coupon** curve (`/canada`,
   `lookup_yield_curve.php` CSV, direct spot yields 0.25–30y, 1986+, 30y from 1991),
   shadow-rate + BRW on the 1991+ inflation-targeting sample; ELB calibrated near 0%
   and ML-refined (weakly identified — Canada only grazed the bound). 6th market in
   the atlas, and included in the global-cycle factor (a deep, liquid G7 market).

### Inflation-linked extension (US + UK real curves)
**Real** term premia (US TIPS `us/tips.py`, GSW feds200805, 2004+; UK
index-linked gilts, 1985+), estimated with plain AFNS — real rates have **no ELB**
("no real cash", bound pushed to −100%). These two real term premia are solid and
are **what the Atlas publishes**: a "nominal vs inflation-protected" panel puts the
US nominal and TIPS term premia side by side over their overlap (2004+).

What the Atlas deliberately **does not** publish is the further split of the
breakeven into *expected inflation* + an *inflation-risk premium* (IRP). The
breakeven itself (nominal − real yield) is a clean market number, but the
"two separate AFNS, subtract" split (`us/inflation.py`, IRP = nominal TP − real
TP) is **too noisy to headline**: the implied 10y expected-inflation series swings
from −1.3% to +5.4% (std 1.27) even though its *latest/average* value lands near
the 2% target — the split is the small difference of two independently-estimated
premia, and it inherits both models' noise. So we report the two term premia and
stop there. The UK split is worse still (RPI-CPI basis + LDI-driven negative real
term premia). See `validation/FINDINGS_inflation.md`.

### Joint nominal/real model & the inflation decomposition — exploratory (not shipped)
A single-kernel **joint** AFNS (`model/joint_afns.py`, `us/joint_us.py`) prices
both the nominal and TIPS curves at once, to try to extract a better IRP than
"subtract". Extensions built and evaluated: a **TIPS liquidity factor**, an
**ATSIX survey anchor** on expected inflation, and a price-based **TIPS "noise"
liquidity proxy**. Outcome (US, vs the Fed Board's DKW model):

- **Real term premium** co-moves with DKW at **0.98**, and **expected inflation**
  tracks the ATSIX survey (both are the usable results).
- **The IRP is not yet resolved.** Its high-frequency correlation with DKW stays
  weak because the IRP is the small residual of two large premia, and separating
  the **TIPS liquidity level** (a pricing-side quantity) from the real term premium
  needs a direct liquidity observable — the TIPS–inflation-swap basis, which is
  **paid data**. DKW's own liquidity series would work but is **circular** for
  validation. The free price-based noise proxy is a crisis-frictions measure, not
  the persistent premium level, so it does not close the gap.

So the joint model stays a **documented research spike**, not surfaced in the
public Atlas. Full write-up: `validation/FINDINGS_joint_us.md`.
**UK joint model: on hold** — runnable (both curves exist) but not validatable to
the US standard (no ATSIX analogue, an RPI→CPI wedge to model, a larger unidentified
LDI/convenience premium, and no public benchmark to check against).

### Cross-country nominal 10y term premium (production, latest)
| Switzerland | US | Germany | UK | Japan |
|---|---|---|---|---|
| −0.4% | +1.2% | +1.3% | +1.5% | +1.7% |

Ordering CH < US < DE < UK < JP: Switzerland lowest (deepest safe-haven), the
euro/dollar core next, gilt/fiscal and post-YCC Japanese premia highest. These
are the **mean-reversion-floor** numbers — see the note below.

> **Correction (mean-reversion floor).** The term-premium numbers scattered
> through the per-country findings below predate a decomposition fix. The
> yields-only ML persistence landed near a unit root, so the expected-average
> short rate tracked the policy rate almost one-for-one and the 10y premium
> **collapsed negative through the 2022-23 hiking cycle** — an artefact, not a
> signal (the SF Fed Christensen-Rudebusch series stayed ~+1% and reads 1.19%
> now). The fix floors the P-persistence at a calibrated ~3.5-year half-life
> (`term_premium.mean_reversion_floor`, replacing Bauer-Rudebusch-Wu for the
> expectations leg; `run_production_tp(persistence="mrfloor")`, the default). It
> is a post-estimation decomposition change — the shadow-rate AFNS fit, the ELB,
> and the shadow short rate are untouched. The authoritative corrected numbers
> and the US-vs-SF-Fed calibration are in **`validation/FINDINGS_mean_reversion.md`**.

## Repo structure

```
/model          # shared AFNS + CR shadow-rate + EKF; joint nominal/real spike (joint_afns.py)
/us             # US GSW nominal + TIPS real fetchers, drivers, inflation decomposition
/germany
/japan
/uk
/switzerland    # SNB CHF spot curve (5th market)
/canada         # Bank of Canada fitted zero-coupon curve (6th market)
/validation     # benchmarks (SF Fed / ACM / Kim-Wright / DKW), ATSIX survey anchor,
                #   TIPS liquidity gauge (tips_liquidity.py), FRED API helper (fred_api.py)
/scheduler      # cadence: weekly/monthly re-estimation, daily filter-forward, Supabase export
/viz            # canonical daily dataset -> thinned payload -> self-contained Atlas artifact
```

## Session guidance

- Work through the build sequence one country at a time; do not attempt all
  four unsupervised in a single session.
- Flag any convergence failure, implausible shadow-rate path, or benchmark
  divergence explicitly rather than silently proceeding.
- The ELB values above are provisional and open to revision on sensitivity
  testing — do not treat them as fixed facts.
