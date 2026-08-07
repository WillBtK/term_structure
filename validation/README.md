# /validation — benchmark comparison

Phase-1 **gate**: the plain (non-shadow-rate) AFNS fit on US GSW data must track
the SF Fed "Treasury Yield Premiums" series before we add the shadow-rate
extension (top-level README). Same model family, same GSW input → expect close
tracking.

## Modules

- `config.py` — quantities, gate maturities (2y, 10y — the only maturities the
  SF Fed publishes a decomposition for), pass criteria
  (correlation ≥ 0.98, mean |diff| ≤ 15 bp — **proposed, confirm**), paths, URL.
- `sf_fed.py` — download + normalize the SF Fed file into the canonical tidy
  frame. Parsing is driven by an explicit `ColumnSpec` so it adapts to the real
  file's headers without touching comparison code.
- `compare_afns.py` — pure, tested core: `align` → `metrics` → `evaluate_gate`,
  plus optional overlay plots and a Markdown/CSV report (`run_comparison`).
- `tests/` — offline tests of the comparison core and the SF Fed reshape.

## Canonical frame schema

Both sides are long-format frames with columns
`date, maturity, term_premium, expected_rate, fitted_yield` (all **decimal**).

## How the two sides are produced

- **Benchmark (SF Fed):** `sf_fed.load_sf_fed_series(refresh=True)` downloads and
  parses `FRBSF_Term_Model_Data.xlsx` (verified 2026-08). It reads the model
  **Fit** sheets (`Fit Term Premium` / `Fit Avg Expected Rate` /
  `Fit Treasury Yields`, columns `FTERMP/FEXPTR/FYIELDnnYR`) into the canonical
  frame — daily, decimal, integer maturities 1..10y (overlap with our grid:
  {1,2,3,5,7,10}y).
- **Ours (AFNS):** run the vendored `/model` engine on the US GSW yield panel
  (`us.fetch_gsw.build_yield_panel`) and decompose the fit into the risk-neutral
  expected-rate path and the term premium, emitted in the canonical schema.
  This model-integration step needs the full numba/sympy/IPython stack and is the
  **next step**.

### Data layer validated against the benchmark's inputs

Our GSW reconstruction matches SF Fed's *observed* input yields
(`Observed Treasury Yields` sheet — same 8-point grid) to **~0.006 bp mean**
across {1,2,3,5,7,10}y, confirming units/maturity conventions align on both
sides before any model run.

## Run

```bash
python -m pytest validation/tests -q            # offline core tests
python -m us.fetch_gsw --refresh                 # cache real GSW panel
# benchmark side (real data):
python - <<'PY'
from validation.sf_fed import load_sf_fed_series
bench = load_sf_fed_series(refresh=True)          # canonical SF Fed frame
print(bench.head())
PY
# full comparison, once the /model side produces `ours`:
python - <<'PY'
from validation.sf_fed import load_sf_fed_series
from validation.compare_afns import run_comparison
bench = load_sf_fed_series()
ours  = ...  # from the /model run on the US panel (canonical schema)
res = run_comparison(ours, bench)
print("PASS" if res["gate"]["passed"] else "FAIL", res["report_path"])
PY
```

Do **not** proceed to the shadow-rate extension until this gate passes.
`matplotlib` is optional (plots are skipped with a warning if absent);
`tabulate` improves the report table but is not required.

## Data access (federalreserve.gov + frbsf.org)

Both source hosts (`www.federalreserve.gov`, `www.frbsf.org`) are now allowlisted
in the environment's egress policy and reachable. If a future environment blocks
them again, the proxy returns 403/407 (see `/root/.ccr/README.md`); re-add the
exact `www.` hosts under the environment's **Network access → Custom**
(https://code.claude.com/docs/en/cloud-environments) and start a fresh session —
allowlists match exact hostnames, so the apex domain alone does not cover `www.`.

Do not substitute an unofficial mirror: the SF Fed benchmark has no reliable
mirror and GSW parameters live only on federalreserve.gov (FRED's `THREEFY*` is
the different Kim-Wright model). `GSW_URL` / `SF_FED_URL` overrides exist only as
a documented escape hatch.

## Runbook (data is live)

1. **GSW → yield panel.** `python -m us.fetch_gsw --refresh` — the printed
   reconstruction cross-check is **< 0.04 bp** vs published `SVENY`.
   `us.fetch_gsw.build_yield_panel()` gives the engine-ready panel (1980–present,
   8-point grid).
2. **SF Fed benchmark.** `load_sf_fed_series(refresh=True)` caches the canonical
   frame from `FRBSF_Term_Model_Data.xlsx` (1998–present, 1..10y).
3. **Model run + decomposition** (the remaining hard core): estimate plain AFNS
   on the US panel via `/model`, decompose into expected-rate path + term
   premium in the canonical schema, then `run_comparison(ours, bench)` and check
   the 2y/10y gate. Provision a Python ≤3.12 venv with
   `model/requirements.lock.txt` (numba/sympy/IPython) for this step.
