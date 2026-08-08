# Phase 1f — 2y/10y validation verdict (Julia engine vs SF Fed)

End-to-end run: `us.engine.run(freq="monthly", start="1987-01-01",
maturities_years=(2,10), iteration=6000, burnin=1000)` (TermStructureModels.jl)
→ `validation.compare_afns.run_comparison` vs SF Fed. Overlap 1998–2026, n=343.
Report artifacts in `validation/reports/` (gitignored).

## Metrics vs SF Fed

| quantity | 2y corr | 2y mad | 10y corr | 10y mad | 10y bias |
|---|---|---|---|---|---|
| fitted yield | 0.99994 | 1.6 bp | 0.99948 | 6.9 bp | +6.5 bp |
| expected rate | 0.986 | 23.7 bp | 0.882 | 46.9 bp | −12.9 bp |
| term premium | 0.851 | 23.7 bp | 0.812 | 47.8 bp | +19.3 bp |

## Gate verdict: **FAIL** (term premium)

Realistic bar = corr ≥ 0.90 **and** |bias| ≤ 25 bp at 2y/10y.
- **Level (bias): PASS** — 2y +5.7 bp, 10y +19.3 bp (both ≤ 25 bp).
- **Co-movement (corr): FAIL** — 2y 0.851, 10y 0.812 (both < 0.90).

## Reading

- **The AFNS implementation is validated.** Fitted yields reproduce SF Fed to
  1.6 bp (2y) / 6.9 bp (10y), corr ~0.9995 — the README's stated purpose for this
  benchmark ("validate the AFNS implementation itself") is met decisively.
- **The term premium tracks closely on level but not to a 0.90 correlation.**
  This is the best an independent estimator achieves here (Phase 1e: even the
  purpose-built Julia package lands ~0.81–0.85 vs SF Fed) — the residual is the
  intrinsic weak-identification of the P-dynamics; SF Fed's specific estimate is
  not reproducible to 0.90 corr without their exact (non-public) choices.

## Decision point
Two defensible readings of "validation":
1. **Accept** — the AFNS engine is validated (yields) and the premium tracks in
   level and direction; the 0.90-corr bar is stricter than an independent
   re-estimation can meet. Proceed to the shadow-rate extension.
2. **Push further** — try to raise correlation (e.g. match SF Fed's exact
   sample/frequency more closely, or their normalization), accepting it may stay
   capped near ~0.85 by identification.
Per the README, do not start the shadow-rate extension until this is accepted.
