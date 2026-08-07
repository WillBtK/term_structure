# Phase 1e findings — independent third estimate (TermStructureModels.jl)

To determine whether the term-premium gap vs SF Fed is intrinsic or specific to
our vendored engine, we ran an **independent** MIT-licensed implementation
(`TermStructureModels.jl`, JSZ constrained by the AFNS restriction, Bayesian
MCMC) on the **same** GSW monthly panel (1987–2026), and compared all estimates
to SF Fed at 2y/10y.

## Results (term premium vs SF Fed, same GSW data)

| estimate | 2y corr | 2y mad | 10y corr | 10y bias | 10y mad |
|---|---|---|---|---|---|
| Julia, P-VAR(11) (marginal-lik. optimal) | 0.851 | 23.8 bp | 0.816 | +20.3 bp | 47.7 bp |
| Julia, P-VAR(1) (matched to SF Fed spec) | 0.851 | 23.0 bp | 0.829 | +15.1 bp | 52.2 bp |
| **Ours (snejens MLE, VAR(1))** | **0.432** | 46.6 bp | **0.774** | +38.3 bp | 66.1 bp |

(Julia compared on 343 monthly overlap points; ours on 7134 daily. Frequency is
not the driver: our *monthly* 10y corr was 0.756, ~ our daily 0.774.)

## Attribution (both questions answered)

1. **P-dynamics lag richness is NOT the cause.** Julia VAR(1) ≈ Julia VAR(11)
   (2y 0.851 = 0.851; 10y 0.829 vs 0.816). Matching SF Fed's VAR(1) spec does
   not degrade Julia's tracking — so the earlier confound is ruled out.
2. **Our vendored engine is the main weak link.** With identical spec (VAR(1)
   AFNS) and identical data, the purpose-built Julia package tracks SF Fed
   **much** better than our engine — decisively at 2y (0.85 vs 0.43) and
   clearly at 10y (0.83 vs 0.77), with about half the level bias (10y +15 vs
   +38 bp). Our engine's short-end P-dynamics (2y) are the worst spot.
3. **A residual intrinsic ceiling remains.** Even the purpose-built Julia
   estimate reaches only ~0.83–0.85 vs SF Fed, not ~1.0. An independent
   re-estimation of a weakly-identified premium cannot perfectly reproduce SF
   Fed's — so *some* gap is intrinsic, but most of ours is the engine.

## Likely cause in our engine
The vendored MLE lands at a very persistent diagonal `K^P`
([0.105, 0.105, 0.88]); the near-unit-root level/slope makes the 2y expected-rate
path too smooth, so the 2y premium mistracks. The Julia Bayesian estimator
regularizes `K^P` via priors and handles the weak identification better.

## Implication / decision
The `snejens` MLE engine (also **unlicensed**) is now shown to be the weaker,
legally encumbered component, while `TermStructureModels.jl` is **MIT-licensed,
purpose-built, tracks SF Fed better, and is already installed and running
end-to-end** in this environment on our GSW data. This materially strengthens
the case for adopting it as the estimation/term-premium engine — reversing the
earlier "reference only" stance, which was taken when we believed our engine was
fine. To be decided with the project owner.

## Validity / reproducibility
- Julia driver: `scratchpad/julia_tp.jl` (yields-only AFNS, `tau_n` in months,
  `data_scale=1200`, `TP_tau=[24,120]`, 6000 draws / 1000 burn-in).
- Our decomposition remains independently validated (synthetic 3e-7 bp), and our
  fitted yields still match SF Fed to 3–4 bp — the curve-fit side is not in
  question; only the P-split from the vendored MLE is.
