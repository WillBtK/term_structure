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

## Strand C — regime-switching AFNS (1961+)  [run in progress]

Markov-switching AFNS on the P-dynamics (`model/regime_switching.py`, Kim 1994
filter, 2 regimes, common arbitrage-free Q-loadings). Synthetic 2-regime ground
truth is recovered (regime sequence classified >90% correct). Empirical 1961+
estimation and the regime-conditional / stagflation term premium are being
estimated; results to be appended here.

## Strand D — ML estimation of the ELB  [run in progress]

`ekf.estimate(..., estimate_elb=True)` adds the ELB as a free parameter
(Christensen-Rudebusch). Likelihood-profile tests confirm identification when
yields approach the bound. US identification is expected to be weak (the US
barely hit the bound); the ML value, its profile, and the comparison to the
-0.35% calibration are being estimated and will be appended. For Germany/Japan
(clearly negative rates) this becomes the primary ELB method.

## Strand E — portability / surveys (decision)

The portable core stays **yields-only**. Bias-correction (A) and regime-switching
(C) are the portable robustness fixes; survey-anchored expectations (SPF/Blue
Chip, à la DKW) would improve US robustness further but are patchy across the four
markets, so they are kept out of the core and remain an optional US-only overlay,
not a pipeline dependency. Documented in the charter `README.md`.
