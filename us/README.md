# /us — US data ingestion (GSW Svensson → yield grid)

Produces the engine's data contract for the US: a date × maturity panel of
zero-coupon yields (decimal) on the common grid `[0.25, 0.5, 1, 2, 3, 5, 7, 10]`
years, reconstructed daily from the Fed Board GSW Svensson parameters.

## Modules

- `config.py` — US settings: `COMMON_GRID`, `ELB = -0.0035` (fixed, for the
  later shadow-rate step), `GSW_URL` (feds200628; override via env), cache paths.
- `fetch_gsw.py` — download + cache the GSW file (`BETA0-3`, `TAU1-2`, plus
  published `SVENYNN` for cross-check). Caches raw CSV + cleaned parquet under
  `data/`. Falls back to cache when the download is blocked.
- `reconstruct.py` — standalone, correct Svensson zero-yield formula →
  `reconstruct_grid()` returns the decimal panel; `cross_check_against_published()`
  verifies reconstruction against the file's own `SVENYNN`.
- `tests/` — offline unit tests (golden values, short-maturity limit, decimal
  convention, invalid-row dropping, the β₃/τ₂ regression guard, cross-check).

## Usage

```bash
python -m us.fetch_gsw --refresh     # download + cache; prints cross-check
python -m pytest us/tests -q         # offline tests (no network)
```

```python
from us.fetch_gsw import build_yield_panel
panel = build_yield_panel()          # date x maturity, decimal yields -> feed /model
```

## Conventions & notes

- GSW `BETA0-3` are in **percent**, `TAU1-2` in **years**; the output panel is
  in **decimal** (÷100) to match `/model`'s contract.
- Svensson yield: `y(n) = b0 + b1·L1(n/τ1) + b2·(L1(n/τ1)−e^(−n/τ1)) +
  b3·(L1(n/τ2)−e^(−n/τ2))`, with `L1(x)=(1−e^(−x))/x`. Implemented standalone
  because the vendored engine's numeric `ns_yield` has a β₃-denominator bug
  (see `model/README.md`); a test guards against reintroducing it.
- **Egress:** `www.federalreserve.gov` is blocked in some environments. Refresh
  the cache where egress is open, or set `GSW_URL` to a mirror. The reconstructed
  10y is cross-checked against the published `SVENY10` (expected < 1 bp).
