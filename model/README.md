# `/model` — shared term-structure engine

The estimation/term-premium engine for the project. Per the top-level
`README.md` there is **one model, one codebase**: only per-country data ingestion
(`/us`, `/germany`, ...) differs.

## Engine: `TermStructureModels.jl` (Julia, MIT)

We use [`TermStructureModels.jl`](https://github.com/econPreference/TermStructureModels.jl)
— a Gaussian affine term-structure model (JSZ constrained by the AFNS
restriction of Christensen–Diebold–Rudebusch 2011), estimated by Bayesian MCMC,
which natively decomposes yields into an expected-average-short-rate component
and a term premium.

**Why this engine.** We first built on a vendored Python AFNS repo
(`snejens/arbfree_dynamic_nelson_siegel`) but a like-for-like comparison
(`validation/FINDINGS_phase1e.md`) showed that engine tracked the SF Fed
benchmark far worse than this one (2y term-premium correlation 0.43 vs 0.85) and
it carried no license. It has been removed; this MIT-licensed, purpose-built
package is the engine.

### Layout
- `engine_julia/` — pinned Julia project: `Project.toml` + `Manifest.toml`
  (TermStructureModels v0.6.0, CSV, DataFrames) and `estimate_tp.jl`, the driver
  that reads a yield-panel CSV (percent) and writes the canonical term-premium
  frame (decimal). Called from Python via `us/engine.py`.
- `term_premium.py` — our own pure-numpy/scipy AFNS decomposition math, validated
  against synthetic ground truth (3e-7 bp) and kept as an independent
  **cross-check** of the Julia output. Not on the production path.
- `tests/` — unit tests for `term_premium.py`.

### Data contract
`us/engine.estimate_term_premium(panel)` consumes a `DataFrame` indexed by date,
columns = maturities in years, values = zero yields in **decimal**, and returns
the canonical long frame `date, maturity, term_premium, expected_rate,
fitted_yield` (decimal) — consumed by `validation/compare_afns`.

## Environment / reproducibility

Requires Julia (installed via juliaup) plus the pinned project.

```bash
# 1) Install Julia (juliaup). Needs egress to the Julia channels:
#    *.julialang.org (installer + package servers) and *.julialang.net (storage).
curl -fsSL https://install.julialang.org | sh -s -- -y
# 2) Instantiate the pinned engine project (packages resolve from Manifest.toml):
julia --project=model/engine_julia -e 'using Pkg; Pkg.instantiate()'
```

The Python side (`us/engine.py`) finds julia via `JULIA_BIN` or
`~/.juliaup/bin/julia`, and honors `JULIA_DEPOT_PATH`. First run precompiles the
Julia packages (a few minutes); subsequent runs are fast.
