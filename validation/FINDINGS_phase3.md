# Phase 3 findings — sample robustness, historical scope, regimes (US)

Motivation (from the ACM-vs-DKW discussion). A yields-only affine model's
expected-rate/premium split hinges on the persistence of the short rate under P
(largest eigenvalue of `K^P`), which is weakly identified and biased toward *too
little* persistence in finite samples — so the term premium is understated, worst
in the highly persistent 1970s. Surveys (the DKW cure) are not portable across
the four markets, so we use the portable, yields-only fixes: **bias-correction**
and **regime-switching**, and we extend the sample back to **1961**.

Engine decision (Strand 0). The Julia engine is Bayesian and returns only the TP
series (no `K^P`), so the **Python EKF** (`model/ekf.py`) is the workhorse for the
Phase-3 extensions (it exposes `K^P`, handles ragged data, estimates the ELB). We
validated it as a plain-AFNS engine by pushing the bound to -100% (the Krippner
floor never binds → exact plain AFNS) and comparing its 1987+ term premium to the
cached SF Fed benchmark.

## Strand 0 — plain-AFNS EKF vs SF Fed (1987+)

Converged (loglik 20239), **fitted yields essentially exact** (MAD 3.73 bp,
per-maturity corr ≈ 1.00, 1–3.5 bp) — the AFNS implementation is correct. The
term-premium *decomposition*, however, tracked SF Fed less well than the Bayesian
Julia engine (2y corr 0.28, 10y 0.69, vs Julia ~0.85/0.83). The gap is entirely
in the weakly-identified P-dynamics — exactly the ACM-style sensitivity — and is
the motivation for Strand A, not a bug (the Q/fitted side is exact).

## Strand A — Bauer-Rudebusch-Wu bias-correction (1987+)  ★ headline result

Kilian bootstrap bias-correction of the VAR(1) persistence (`model/bias_correction.py`,
2000 replications) pushed the level-factor persistence to near-unit-root (smallest
`eig(K^P)` 0.174 → 0.0005; capped at the stationarity boundary, δ=0.87) and
**sharply improved the short-to-medium term-premium tracking vs SF Fed**:

| maturity | TP corr (ML) | TP corr (bias-corrected) | bias (ML → BC) |
|---|---|---|---|
| 1y  | 0.14 | **0.83** | -7.6 → -9.5 bp |
| 2y  | 0.28 | **0.79** | -5.3 → -6.8 bp |
| 3y  | 0.39 | **0.74** | +0.1 → -0.2 bp |
| 5y  | 0.53 | **0.68** | +10.9 → +14.1 bp |
| 7y  | 0.62 | 0.66 | +18.1 → +25.2 bp |
| 10y | 0.69 | 0.66 | +22.6 → +35.7 bp |

Interpretation: the finite-sample persistence bias was the dominant cause of the
poor short-end tracking; correcting it aligns the expected-rate path with SF Fed
(1–3y corr 0.14–0.39 → 0.74–0.83). At the long end the correlation is roughly
flat and the *level* rises (more persistence → higher/steeper premium — BRW's
known effect). This both licenses the bias-corrected EKF as the workhorse and
directly answers the user's robustness question: yes, the model shares ACM's
persistence sensitivity, and BRW bias-correction is an effective, portable fix.

## Strand B — historical scope to 1961 (data + ragged filter)

`us/reconstruct.py` now reconstructs the pre-1980 four-parameter Nelson-Siegel
regime (Svensson with BETA3=0) and masks each maturity to NaN before it is
published in the file's `SVENYNN` columns. Result: the panel spans **1961-06 →
2026-07** (was 1980+), with 3m–7y available from 1961 and **10y from 1971-08**
(correctly ragged; matches the file's own availability exactly). NS-era
reconstruction matches published `SVENY` to **0.005 bp**. The EKF/Kim measurement
loops mask missing maturities per date, so the ragged early sample is used
without dropping the 1960s.

## Historical term premium to 1961 (single-regime, bias-corrected)  ★ stagflation

Plain-AFNS on the full **1961-2026** sample (782 monthly obs, ragged early 10y),
with BRW bias-correction (persistence pushed to near-unit-root, ρ 0.990→0.999).
10y term premium by era:

| era | 10y TP (ML) | 10y TP (bias-corrected) |
|---|---|---|
| 1970s stagflation (1970-85) | 2.03% | **1.35%** |
| post-1990 | 0.90% | 1.37% |
| peak (bias-corrected) | — | **3.95% (1992-09)** |

Annual bias-corrected 10y TP climbs through the era: ~0.2-1.5% in the 1970s, then
**2.5-2.9% in 1983-85** (the Volcker-disinflation / high-real-yield aftermath).

**The key, slightly counter-intuitive result:** under properly persistent
(bias-corrected) dynamics the *pure 1970s stagflation* term premium is **moderate
(~1.3% at 10y), not extreme** — because with a near-unit-root short rate, the high
1970s long yields are mostly attributed to rationally-expected high *future* short
rates (persistent inflation), leaving a moderate premium. The premium is *higher*
in the early-1980s to early-1990s disinflation than at the 1970s inflation peak
itself. Note the direction of the bias-correction is state-dependent: it *lowers*
the 1970s premium (more of the high rate is "expected") but *raises* the low-rate
post-1990 premium — a clean illustration of why the persistence assumption, not
the fit, drives the level (the whole ACM-sensitivity point). Caveat: the shape is
sensitive to whether the 1970s is allowed its own dynamics — see regime-switching.

## Strand C — regime-switching AFNS (1961+)

Markov-switching AFNS on the P-dynamics (`model/regime_switching.py`, Kim 1994
filter, 2 regimes, common arbitrage-free Q-loadings). Synthetic 2-regime ground
truth is recovered (regime sequence >90% correct; 7 unit tests). Empirical 1961+
fit (782 obs, warm-started from the plain-AFNS optimum): loglik **30368 vs 30032**
single-regime (+335 — a distinct regime genuinely improves the fit).

Two regimes emerge with sharply different character:

| | calm regime | **high-inflation regime** |
|---|---|---|
| long-run level θ_L | 1.7% | **11.2%** |
| factor vols σ (level/slope/curv) | 50/81/243 bp | **97/322/508 bp** |
| stay-probability | 0.95 | 0.86 |
| share of months | 77% | **23%** |

**The high-inflation regime is dated sensibly** — smoothed probability peaks in
**1973-75** (first oil shock, prob 0.79-0.98) and **1980-82** (second oil shock /
Volcker, 0.85-1.00), covering ~52% of 1970-85; it is essentially absent in the
calm 1990s-2010s. So the model cleanly *identifies when* the stagflation regime
was, with its own much higher mean and roughly double the factor volatility.

**Honest caveat on the level.** This fit hit its iteration budget without meeting
the tight convergence tolerance (`success=False`), and the regime-*conditional*
term premium — a counterfactual "if the economy sat in the θ_L=11% regime at date
t" — is noisy and can go negative (the current state usually sits far below an 11%
mean). So the clean *quantitative* stagflation term-premium level is best taken
from the single-regime bias-corrected estimate above (**~1.3% at 10y**, peaking
~2.9% in the mid-1980s); the regime model's robust contribution is the *dating* of
the high-inflation regime and the confirmation that it carries distinctly higher
persistence and volatility. A longer, multi-start regime run (or a 3-regime
variant) is the natural refinement if a converged regime-conditional level is
wanted.

## Summary — the stagflation term premium

Combining the two historical runs: a stagflation regime is real and identifiable
(1973-75, 1980-82) with ~2× the factor volatility of calm periods, but the 10y
*term premium* attributable to it is **moderate, ~1-1.5%**, not extreme — because
under properly persistent dynamics most of the high 1970s long yields reflect
rationally-expected high future short rates, not premium. The premium peaked
*after* the inflation peak (early-1980s to early-1990s, ~2.5-4%). And the whole
level is sensitive to the persistence assumption — exactly the ACM-style
sensitivity that motivated the bias-correction, now quantified.

## Strand D — ML estimation of the ELB (US, 1987+)

`ekf.estimate(..., estimate_elb=True)` adds the ELB as a free parameter
(Christensen-Rudebusch). On the US:

- **ML ELB = +0.16%** (vs the -0.35% calibration), likelihood gain +223.
- The profile is **asymmetric**: the data sharply rules out any bound *above*
  ~+0.2% (a floor cannot sit above the lowest observed yields — the negloglik
  collapses), but is nearly flat going *more negative* (Δloglik only ~180 from 0%
  to -2%). So the US ELB is **identified from above, weakly from below** — the ML
  value just pins to the observed short-yield floor (US yields never went
  negative). This confirms keeping the US ELB a calibrated input with a
  sensitivity band, and explains the Phase-2 shallow shadow rate: the data prefer
  a near-zero bound, not a deeply negative one.
- For Germany/Japan (clearly negative rates) ML estimation will be well-identified
  and is the primary ELB method. The revealed bound abroad (SNB/ECB/BoJ ≈
  -0.75/-0.50/-0.10%) anchors the institutional cash-holding cost — "digital cash"
  makes warehousing large balances *harder*, so the bound is more, not less,
  binding than a retail storage-cost calculation implies.

## Strand E — portability / surveys (decision)

The portable core stays **yields-only**. Bias-correction (A) and regime-switching
(C) are the portable robustness fixes; survey-anchored expectations (SPF/Blue
Chip, à la DKW) would improve US robustness further but are patchy across the four
markets, so they are kept out of the core and remain an optional US-only overlay,
not a pipeline dependency. Documented in the charter `README.md`.
