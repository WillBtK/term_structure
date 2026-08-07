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
- **Ours (AFNS):** `us.engine.run(freq, start, end)` builds the US GSW panel and
  runs the Julia engine (`model/engine_julia`, TermStructureModels.jl) to produce
  the term-premium decomposition in the canonical schema. (Our own
  `model/term_premium.py` provides an independent pure-Python cross-check.)

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
# full comparison (Julia engine produces `ours`):
python - <<'PY'
from us.engine import run
from validation.sf_fed import load_sf_fed_series
from validation.compare_afns import run_comparison
ours  = run(freq="monthly", start="1987-01-01")   # Julia engine -> canonical frame
bench = load_sf_fed_series()
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
3. **Model run + decomposition.** `us.engine.run(...)` estimates the AFNS via the
   Julia engine and returns the canonical frame; `run_comparison(ours, bench)`
   checks the 2y/10y term-premium gate. Requires Julia (see `model/README.md`).
   Current result (monthly 1987+): fitted yields ~3–4 bp vs SF Fed; term premium
   2y corr 0.85, 10y corr 0.83 (see `FINDINGS_phase1e.md`).
