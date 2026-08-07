# License status of the engine — RESOLVED

The project originally vendored `snejens/arbfree_dynamic_nelson_siegel`, which
had **no license** (all rights reserved) — a tracked legal risk.

That code has been **removed** from the repository (git history retains it). The
engine is now [`TermStructureModels.jl`](https://github.com/econPreference/TermStructureModels.jl),
which is **MIT-licensed**, so the open-item is closed. See
`validation/FINDINGS_phase1e.md` for why the switch was made (it also tracks the
SF Fed benchmark materially better).

Our own decomposition math in `term_premium.py` is original to this project and
carries the repository's license.
