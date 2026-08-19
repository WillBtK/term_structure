# Germany neutral-anchor inputs

Inputs to `germany/neutral_anchor.py` — the shifting endpoint for the German
expected-rate path. All files are `date,value` CSVs with **values in percent** and
`date` as the **release/observation date** (not the forecast horizon), e.g.

```
date,value
1990-04-30,3.2
1990-10-31,3.1
```

The anchor is **inert** until at least one of `consensus_*` or `ferreira_rstar_ea.csv`
is present, so the pipeline never breaks on a missing file.

## Auto-maintained (do not edit — overwritten weekly by CI)

| file | source | cadence | span |
|---|---|---|---|
| `ferreira_rstar_ea.csv` | Fed FEDS Note, "Real-Time Global Longer-Run Neutral Rates", `EA` column | semi-annual | 1960+ |
| `ecb_spf_infl_lt.csv` | ECB Data Portal, SPF `Q.U2.HICP.POINT.LT.Q.AVG` (average point forecast, longer-term horizon) | quarterly | 1999+ |

Refreshed by `.github/workflows/refresh-anchors.yml` via `python -m germany.fetch_neutral`.
A failed fetch keeps the committed file rather than overwriting it.

## Provided manually — proprietary, one-off (NEVER auto-refreshed)

| file | what it is | needed span |
|---|---|---|
| `consensus_shortrate_ea_6_10y.csv` | Consensus Economics **6–10 year ahead** forecast of the euro-area **3-month interbank rate** | 2016-10+ |
| `consensus_cpi_de_6_10y.csv` | Consensus Economics **6–10 year ahead** forecast of **German CPI inflation** | 1990+ |

These are Consensus Economics data: this repo is private, and only the *derived* term
premium is ever published — never the raw survey.

**They are deliberately allowed to go stale.** That is what the degradation ladder is
for. While the short-rate file is inside its staleness window (`STALENESS_DAYS = 550`)
it is used directly; past that, the anchor falls through to the public bridge
`n = ferreira_rstar_ea + pi^e`, level-calibrated to the Consensus short rate over their
overlap so the two rungs join without a step. The CPI file extends the same way, via the
ECB SPF. So a one-off snapshot anchors the history and the public rungs keep the current
edge live indefinitely, with no manual step and no silent freeze.

## Will it carry forward? — measure it, don't assume it

There is **no public, machine-readable, long-horizon euro-area short-rate survey** to
replace rung 1 when the Consensus snapshot lapses (the ECB Data Portal carries SPF,
Bank Lending, Consumer Expectations and Money Market surveys — none of which forecasts
interest rates at a 6-10 year horizon; the Survey of Monetary Analysts is published as
documents, not data). So rung 1 *will* go stale, and everything then rests on the public
bridge. Whether that hand-off preserves the meaning of the anchor is an empirical
question, and `model.neutral_anchor.bridge_fidelity` answers it:

```bash
python -c "
from model.neutral_anchor import bridge_fidelity
from germany.neutral_anchor import _spec
print(bridge_fidelity(_spec()))"
```

It rebuilds the bridge with the direct survey withheld and compares the two over their
overlap, reporting the gap **after** the level calibration the ladder already applies —
so a constant offset is removed by construction and only genuine drift shows up. A small
`mean_abs_gap_bp` with high `corr` means the bridge is a faithful stand-in and the
anchor will not change character on the day the survey lapses. A large or drifting gap
is the warning that it would, and would argue for shortening `STALENESS_DAYS` so the
transition happens while it can still be seen against the survey.

## Checking what the anchor is actually doing

```bash
python -c "
import pandas as pd
from germany.neutral_anchor import build_nominal_neutral
v, m = build_nominal_neutral(pd.date_range('1990-01-31', '2026-08-31', freq='M'))
print({k: v for k, v in m.items() if k != 'src_by_date'})
print('latest endpoint: %.2f%%' % (v[-1] * 100))"
```

`meta['source']` names the active rung, `meta['pi_e']` the inflation leg,
`meta['calib_offset_pct']` the level match between bridge and direct survey, and
`meta['asof']` the last point-in-time survey release pinning today's endpoint.
