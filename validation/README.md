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

## Unblocking the real data (federalreserve.gov + frbsf.org)

Both source hosts are denied by this environment's **egress policy** (the proxy
returns 403/407; see `/root/.ccr/README.md`, which says to report the blocked
host, not route around it). GitHub / PyPI / FRED are reachable; these two are
not. The fix is a configuration change on the Claude Code cloud environment:

1. Edit the environment → **Network access → Custom** and add
   `federalreserve.gov` and `frbsf.org` (include the `www.` hosts).
   Docs: https://code.claude.com/docs/en/cloud-environments
2. Note the known propagation gap: set this on the **environment's** Network
   access, not only the account-level "Additional allowed domains" (claude-code
   issues #19087, #71629).
3. Egress policy is fixed at container start — **start a fresh session** after
   changing it.

Do not substitute an unofficial GitHub mirror: the SF Fed benchmark has no
reliable mirror (the formal gate needs `frbsf.org` regardless), and GSW
parameters live only on federalreserve.gov (FRED's `THREEFY*` is the different
Kim-Wright model). The fetchers accept `GSW_URL` / `SF_FED_URL` overrides only
as a documented, caveated escape hatch.

## First-access runbook (once the hosts are allowlisted, fresh session)

1. **GSW → yield panel.** `python -m us.fetch_gsw --refresh` — confirm the
   printed reconstruction cross-check is **< 1 bp** vs published `SVENY10`.
   `us.fetch_gsw.build_yield_panel()` then gives the engine-ready panel.
2. **SF Fed benchmark.** Fetch and inspect the real headers, then finalize the
   spec (only 2y/10y are published):
   ```python
   import pandas as pd
   from validation.sf_fed import sf_fed_spec_template, normalize
   raw = pd.read_excel("<downloaded file>")   # or read_csv
   print(list(raw.columns))                    # read real headers
   spec = sf_fed_spec_template()               # replace *_PLACEHOLDER names
   bench = normalize(raw, spec)                # -> canonical 2y/10y series
   ```
   Set `SF_FED_URL` and use `load_sf_fed_series(spec=..., refresh=True)` to cache.
3. **Model run + decomposition** (the deferred hard core): estimate plain AFNS
   on the US panel via `/model`, decompose into expected-rate path + term
   premium in the canonical schema, then `run_comparison(ours, bench)` and check
   the 2y/10y gate. Provision a Python ≤3.12 venv with
   `model/requirements.lock.txt` (numba/sympy/IPython) for this step.
