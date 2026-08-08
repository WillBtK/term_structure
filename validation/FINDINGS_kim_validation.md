# Kim-filter validation (regime-switching AFNS)

Before wiring the Markov-switching AFNS into any pipeline, the Kim (1994) filter
was validated in isolation — because a collapsing-approximation bug runs without
error and produces plausible-but-wrong output ("it runs" is not evidence).

## A. Correctness — Kim vs exact path-enumeration filter

The Kim filter is compared against an **exact** filter that enumerates every
regime path and reuses the *identical* per-branch Kalman math (`rs._kalman_branch`),
so the only difference is Kim's collapse. Tests in
`model/tests/test_regime_switching.py` (`test_kim_matches_exact_path_enumeration`,
`test_kim_degenerate_identical_regimes_is_flat_and_exact`,
`test_kim_loglik_relabel_invariant`).

| check | result |
|---|---|
| loglik, Kim vs exact (T=8/10/12) | \|Δ\| ≈ 1e-4 (1.4e-4, 1.2e-5, 3.3e-5) |
| filtered regime prob, Kim vs exact | max \|Δ\| ≈ 1e-5-1e-10 |
| identical regimes → filtered prob | pinned to stationary dist to 1e-15 |
| identical regimes → loglik vs exact | \|Δ\| ≈ 1e-13 |
| loglik under 0↔1 relabel | unchanged to 0.0 |

The collapsing approximation is implemented correctly — no wrong-axis / wrong-weight
/ i↔j-swap bug.

## B. Empirical reproduction — Ang-Bekaert (2002) / Levant-Ma (2017)

Fit the 2-regime AFNS to US Treasury (GSW) yields **1972-2007**, monthly, 8-point
grid; three starts, best loglik kept (17238 vs 17240 — consistent). The high-vol
regime has σ = 93/492/569 bp (level/slope/curv) vs the calm regime's 55/106/272 bp.

Smoothed P(high-volatility regime), annual:

| period | P(high-vol) |
|---|---|
| 1973-75 (first oil shock) | 0.67 / 0.91 / 0.71 |
| 1976-78 (calm) | ~0.00 |
| **1979-82 (Volcker)** | 0.24 / 0.96 / 1.00 / 0.86 |
| 1983-86 | 0.00 / 0.49 / 0.44 / 0.33 |
| 1987-2007 | ~0.00 (Great Moderation) |
| mean pre-1985 | 0.45 |
| **mean post-1985** | **0.035** |

The **post-1985 low-vol dominance is textbook-clean** and the high-vol regime is
confined to pre-1985, landing exactly on the two genuinely volatile episodes —
matching what Ang-Bekaert and Levant-Ma find (high-vol *clustered* pre-1985,
dominated by Volcker). The one nuance vs a literal "all pre-1985 is high-vol": the
calm mid-1970s (1976-78) are correctly classified low-vol, so the pre-1985 mean is
0.45, not a majority — arguably more faithful than a uniform split. Reassuringly,
this vol-identified regime dates the *same episodes* as the earlier level-identified
run, so the regime timing is robust across specifications.

## C. Volcker window (Oct 1979 - Oct 1982)

The Fed's non-borrowed-reserves operating procedure produced short-rate volatility
from a different policy *instrument*. The 2-regime model **fits this window well**:
mean P(high-vol) over Oct-1979-Oct-1982 = **0.98**. It is captured within the
existing high-vol regime (which also picks up 1973-75, so it is a genuine
volatility regime, not a Volcker dummy). **A third dedicated regime is not needed.**

## Verdict

Correctness is beyond doubt (A). The empirical classification (B, C) is
economically sensible and literature-consistent. Caveats reported honestly: the
optimizer plateaus without meeting the tight NM tolerance (`success=False`) though
multi-start consistency indicates a stable optimum; and pre-1985 high-vol is
episodic, not uniform. The filter is trusted for the Phase-4 splice.
