# `/model` — shared AFNS engine (vendored)

This is the single, country-agnostic estimation engine for the whole project.
Per the top-level `README.md`, there is **one model, one codebase**: only the
per-country data ingestion (`/us`, `/germany`, ...) differs — never fork the
model per country.

## Provenance

Vendored from `snejens/arbfree_dynamic_nelson_siegel` @ `2a402a8` (2024-05-19).
It implements the arbitrage-free Dynamic (Generalized) Nelson-Siegel model of
Christensen, Diebold & Rudebusch (2011), estimated by Kalman filter + MLE, with
an optional Svensson (5-factor, `n=5`) extension alongside the 3-factor (`n=3`)
form.

> **License:** upstream has none. See `LICENSE.NOTE.md` — do not distribute
> externally until a license is granted.

## Layout

- `arbfree_dyn_ns/` — the vendored Python package (verbatim except `config.py`).
- `arbfree_dyn_ns/config_upstream.py.orig` — the original upstream `config.py`.
- `requirements.lock.txt` — upstream's exact late-2023 pins. We reproduce these
  first for numerical fidelity during SF Fed validation, then upgrade later.

## Key modules and the estimation interface

- `nss.py` — Nelson-Siegel-Svensson factor loadings; `B_matrix(lambda, grid,
  lambda_svensson)` and `C_vector(...)` build the affine yield map `y = B x + C`.
  ⚠️ **Known upstream bug:** the numeric `ns_yield()` computes the β₃ (Svensson)
  term with denominator `lambda_ * T` instead of `lambda_svensson * T` (compare
  the correct symbolic `nss_formula`). Do **not** reuse `ns_yield()` for Svensson
  yield reconstruction; `/us/reconstruct.py` implements the correct formula
  standalone. This does not affect the Kalman/MLE path, which uses `B_matrix`.
- `kalman.py` — Kalman filter (numba-jit).
- `optimization.py` — `kalman_optimize(...)`: the MLE driver over the state-space
  parameters. Imports IPython for progress display.
- `curves.py`, `packing.py`, `utils.py` — curve helpers, parameter (un)packing,
  utilities.
- `buba_data.py`, `lw_data.py` — upstream's Bundesbank / Krippner loaders. Kept
  for reference; **not used** by the US pipeline. They illustrate the data
  contract our loaders follow.

### Data contract (what a country loader must produce)

`optimization.kalman_optimize` consumes `synth_yields`: a `pandas.DataFrame`
indexed by observation date, with **columns = maturities in years** (the grid)
and **values = zero-coupon yields in decimal** (e.g. `0.0423`, not `4.23`).
`/us` produces exactly this on the common grid `[0.25, 0.5, 1, 2, 3, 5, 7, 10]`.

See `main.py` (`synth_yields_for_main`, `initial_solution`) for how upstream
wires a data source, an initial parameter guess, and `kalman_optimize` together
— the US driver follows the same shape with our grid and `data_source="us"`.

## Environment

Requires Python ≤3.12 (numba 0.58.1 / llvmlite 0.41.1 pins). Set
`ARBFREE_BASE_PATH` to choose where caches/logs/pickles are written (defaults to
`~/.cache/arbfree-dyn-ns`).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r model/requirements.lock.txt
```
