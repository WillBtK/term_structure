# /validation — benchmark comparison

Phase-1 **gate**: the plain (non-shadow-rate) AFNS fit on US GSW data must track
the SF Fed "Treasury Yield Premiums" series before we add the shadow-rate
extension (top-level README). Same model family, same GSW input → expect close
tracking.

## Modules

- `config.py` — quantities, gate maturities (5y, 10y), pass criteria
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

- **Benchmark (SF Fed):** `sf_fed.load_sf_fed_series(spec, refresh=True)`.
  ⚠️ The exact download URL and column layout are **not yet verified** —
  `frbsf.org` is egress-blocked in this environment. Set `SF_FED_URL` and fill a
  `ColumnSpec` matching the live file's headers, then confirm the reconstruction
  before trusting numbers.
- **Ours (AFNS):** run the vendored `/model` engine on the US GSW yield panel
  (`us.fetch_gsw.build_yield_panel`) and decompose the fit into the risk-neutral
  expected-rate path and the term premium, emitted in the canonical schema.
  This model-integration step needs the full numba/sympy/IPython stack and an
  egress-open environment (GSW + SF Fed), so it is the **next step**, not run
  here.

## Run

```bash
python -m pytest validation/tests -q            # offline core tests
# once both sides are available:
python - <<'PY'
from validation.sf_fed import load_sf_fed_series, ColumnSpec
from validation.compare_afns import run_comparison
bench = load_sf_fed_series(spec=ColumnSpec(...), refresh=True)
ours  = ...  # from the /model run on the US panel
res = run_comparison(ours, bench)
print("PASS" if res["gate"]["passed"] else "FAIL", res["report_path"])
PY
```

Do **not** proceed to the shadow-rate extension until this gate passes.
`matplotlib` is optional (plots are skipped with a warning if absent);
`tabulate` improves the report table but is not required.
