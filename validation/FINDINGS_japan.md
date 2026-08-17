# Japan (JGB) term premium — findings

> **⚠ Superseded sample & numbers.** This doc describes the original **1986-07**
> production sample. Japan's production sample now starts **2000-05** — the
> regime-clean modern sample dated by a 2-regime Kim filter on the full 1974+
> curve (permanent level-regime break at May 2000; 1986 sat 14 years *inside* the
> old high-level regime). Under the 2000+ fit the latest 10y premium is **+2.15%**
> (a lower deflation-era θ anchor lifts it vs the old sample — read the level as
> anchor-sensitive given the 2024+ BoJ exit). The 1974–2000 high-inflation era is
> now a **separate labelled exhibit** (10y TP ~+3.4% mean / +4.9% peak). See
> **`FINDINGS_japan_regimes.md`** (regime dating + the 2000 decision) and
> **`FINDINGS_mean_reversion.md`** (θ-anchoring). The QQE/YCC narrative, ELB, and
> YCC-downweight methodology below still stand.

Third country. Same engine as the US/Germany (`/model` unforked); the data layer
(`japan/`) and two Japan-specific handling steps (ML-estimated ELB, YCC
downweight) differ. Two Japan challenges were flagged up front — the long
ZLB/negative-rate era and the deliberate QQE/YCC distortion — and both are
handled explicitly below.

## Data
Ministry of Finance JGB spot yields (`jgbcm_all.csv`), published **directly by
maturity** (no Svensson reconstruction), Shift-JIS, Japanese-era dates. Monthly
panel on `[1,2,3,5,7,10,20,30]`y, **1986-07 → 2026-07** (481 obs), genuinely
ragged: 10y from 1986-07, 20y from 1986-12, 30y from 1999-09 (per-date maturity
masking in the EKF). Sample start 1986-07 covers the **1989-90 bubble peak/collapse**
and the **1998 Trust Fund Bureau (VaR) shock**. Yield-fit MAD **4.8 bp**.

## Effective lower bound — ML-estimated, validates the NIRP anchor
JGB yields sat clearly at the bound for years, so the ELB is well identified. ML
estimate = **−0.17%**, right at the NIRP policy rate (−0.10%) and inside the
revealed short-yield floor (−0.1% to −0.35%). We start at −0.20% and estimate; the
data pins it near NIRP, not at the deeper −0.75% reversal-rate concept.

## QQE / Yield Curve Control handling
Under YCC (2016-09 → 2024-03) the BoJ price-pegged the 10y. We **downweight** the
pegged tenors (7y/10y) in the measurement over that window (heaviest while the
band was tight ±0.10%, tapering as it widened to ±0.25/0.50/1.0%), so the peg does
not distort the inferred latent factors — the long end (20y/30y, never pegged)
still identifies the curve. The term premium is then reported against **observed**
yields (`observed − expected-average-floored-short-rate`), i.e. the *realised,
policy-suppressed* premium — which is exactly where the peg shows up: the observed
10y sat **below** the model-fair fitted 10y under YCC, driving the realised
premium negative.

## Headline result — realised 10y term premium (bias-corrected)

The pattern the project was built to show — **negative term premia out to 10y
under QQE/YCC, normalising as YCC ended and the BoJ shifted to hikes/QT:**

| event | date | 10y TP | 5y TP | 2y TP | expected short (10y) | observed 10y |
|---|---|---|---|---|---|---|
| bubble peak | 1990-12 | +0.29% | +0.05% | −0.08% | 6.33% | 6.62% |
| TFB (VaR) shock | 1998-12 | +1.19% | +0.77% | +0.23% | 0.93% | 2.12% |
| QQE start | 2013-04 | **−0.19%** | −0.41% | −0.39% | 0.80% | 0.61% |
| NIRP | 2016-02 | **−0.38%** | −0.43% | −0.30% | 0.32% | −0.06% |
| YCC | 2019-08 | **−0.18%** | −0.22% | −0.14% | −0.10% | −0.28% |
| YCC band widened | 2023-01 | −0.03% | −0.18% | −0.20% | 0.54% | 0.51% |
| YCC exit | 2024-03 | +0.08% | −0.14% | −0.15% | 0.67% | 0.75% |
| latest | 2026-07 | **+0.89%** | +0.42% | +0.12% | 1.91% | 2.80% |

- **10y realised TP is negative in 81% of QQE/YCC months** (2013-04..2024-03),
  averaging −0.07% over the YCC window, min **−0.42% (Jan 2016)**.
- **Negative across the curve** under QQE/YCC (5y and 2y troughs ≈ −0.43%/−0.39%),
  i.e. suppression "out to 10y", not just at the peg.
- **Normalises monotonically** off the trough: NIRP −0.38% → 2019 −0.18% → 2023
  band-widening −0.03% → 2024 exit +0.08% → **+0.89% now** (BoJ hiking, JGB 10y
  ≈ 2.8%). Post-YCC 10y TP averages +0.28%.
- The **1998 TFB shock** shows a positive 10y premium spike (+1.19%) — the
  documented VaR-shock curve steepening — captured by the model.

## Two decomposition fixes this leg surfaced (apply pipeline-wide)
Japan's near-unit-root ZLB factors exposed two latent bugs in the shared
expected-rate decomposition; both are now fixed in `model`/`us.production_tp` and
covered by tests, and the US/Germany production series were re-run on the
corrected path:
1. **Diagonal bias-correction.** The BRW correction re-fitted an *unrestricted*
   VAR(1) on the filtered factors; on collinear near-unit-root factors that
   produced a full K^P with **complex eigenvalues** and a distorted (declining)
   expected-rate path. The correction now preserves the independent-factor
   (diagonal K^P) production spec (`bias_correction.bias_correct(..., diagonal=True)`),
   giving real, positive mean-reversion.
2. **Sample-mean θ for the expectations leg.** The ML transition mean θ is
   unidentified in a near-unit-root sample (Japan's implied long-run short rate
   came out ~17%); feeding it to the expectations leg blew up the premium. Per BRW
   the expectations-leg long-run mean is the **sample mean** of the filtered
   factors (0.75% here), held separately in `ProductionParams.theta_expect`. The
   ML θ is still used for the (θ-insensitive) filter transition.

Before the fixes the Japan 10y TP came out a spurious +2.9% today with *no*
negative YCC episodes; after, it is the economically coherent series above.

## Validation
No Japanese SF-Fed-equivalent term-premium benchmark exists, so validation rests
on: (1) MoF data/ragged availability matching the published first-dates exactly;
(2) yield-fit MAD 4.8 bp; (3) the ML ELB (−0.17%) landing at the NIRP anchor;
(4) the YCC downweight test (downweight ≈ drop the pegged datum,
`model/tests/test_ekf.py`); (5) the realised-premium pattern reproducing the
known QQE/YCC compression and post-YCC normalisation; (6) directional check vs the
Krippner Japan shadow short rate where reachable (`economy="JP"`).
