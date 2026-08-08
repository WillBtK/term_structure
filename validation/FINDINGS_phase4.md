# Phase 4 findings — regime/shadow two-model splice (US)

The full 1961-present US term premium is produced by two separately-estimated
models joined at **2008-12-31**: a regime-switching AFNS (pre-seam, frozen) and
the shadow-rate AFNS (post-seam, live). Rationale, decisions, and the Kim-filter
validation are in `README.md` and `validation/FINDINGS_kim_validation.md`.

Legs (monthly): shadow AFNS 1987-2026 (loglik 20286, converged, λ=0.512); regime
AFNS 1961-2008 with **λ fixed to 0.512** (loglik 21505, `success=False` — NM
plateaued at the budget, as in every long-sample regime fit). Regimes: a
high-vol/persistent regime (σ 112/365/544 bp, κ_L≈0.001) and a calmer one
(σ 51/109/286 bp).

## Continuity check (the deliverable) — factors continuous, TP is NOT

Over the 1987-2008 overlap (264 months) the two legs' filtered factors co-move
very tightly, but with non-trivial level gaps:

| factor | corr | mean gap | max gap |
|---|---|---|---|
| L | 0.997 | 8.5 bp | 66.8 bp |
| S | 0.998 | 7.8 bp | 70.8 bp |
| C | 0.995 | 19.2 bp | 79.9 bp |

At the seam (2008-12-31):

| | L | S | C | 10y TP |
|---|---|---|---|---|
| regime leg | 6.33% | −5.71% | −8.41% | **−1.06%** |
| shadow leg | 5.66% | −5.00% | −8.52% | **+1.38%** |
| gap | +66.8 bp | −70.8 bp | +11.2 bp | **−243 bp** |

**Read honestly:**
- The **short rate (L+S) is continuous** across the seam (~4 bp: regime 0.62% vs
  shadow 0.66%). The L and S gaps (~±70 bp) are an offsetting level/slope
  *rotation* between the two independently-estimated models, not a jump in the
  economically meaningful short rate. Factor correlations 0.995+ over 22 years.
- The **10y term premium is discontinuous at the seam (−243 bp)**: the regime leg
  puts it at −1.06%, the shadow leg at +1.38%. This is a genuine discontinuity to
  flag before acceptance, driven by (a) the two legs having different P-dynamics /
  expected-rate paths, (b) the regime-*mixed* TP being the noisy quantity already
  flagged in Phase 3 (a crisis-date value pulled by the persistent regime), and
  (c) the regime leg's non-convergence. Per the charter we **report this rather
  than force it**.

## Interpretation and options

The splice is clean where it matters operationally: the **live shadow leg governs
the entire ZLB/post-2008 era** (the reason the two-model architecture exists), and
the short-rate factor is continuous. The term-premium *level* is not continuous at
the seam because two independently-estimated models disagree on the expected-rate
decomposition — exactly the persistence-sensitivity documented throughout Phase 3.

Options (for the user to decide — not forced here):
1. **Accept** the splice with the discontinuity documented; use the shadow leg for
   2008+ (validated vs SF Fed) and the regime leg for the historical regime
   narrative, noting the ~2.4pp seam gap in the 10y TP level.
2. **Improve the regime leg**: a converged (multi-start, longer) or bias-corrected
   regime fit would likely shrink the gap; the regime-mixed TP is the weak link.
3. **Level-align at the seam**: report the pre-2008 TP as a *deviation* spliced
   onto the shadow level at 2008-12 (removes the level jump; keeps the historical
   shape). A reporting choice, not a re-estimation.

## Artifacts
- Continuous series: `us/splice.py` → `run_splice()`; `spliced` frame
  `(date, L, S, C, shadow_short_rate, tp_<mat>y)` 1961-2026 (782 months).
- Continuity numbers above from `splice.continuity_check`.
