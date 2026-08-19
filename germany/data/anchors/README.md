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
| `ecb_spf_rate_lt.csv` | ECB Data Portal, SPF `Q.U2.ASSU.IR.LT.Q.AVG` — forecasters' longer-term **interest-rate assumption** | quarterly | 2025+ |

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

## Will it carry forward? — yes, and it is measurable

There **is** a public, machine-readable, long-horizon euro-area short-rate expectation:
the SPF's own longer-term **interest-rate assumption**, `Q.U2.ASSU.IR.LT.Q.AVG`. It is
not obvious from the flow catalogue — it sits under the `ASSU` ("Assumptions") topic
rather than anywhere named like a forecast — which is why an earlier pass through this
missed it and wrongly concluded no such series existed.

Two properties make it the right carry-forward rather than just another fallback:

* **Same concept as the Consensus series.** The SPF "interest rate" variable is a
  3-month market rate, not the policy rate — its near-term counterpart
  (`Q.U2.ASSU.IR.P3M.Q.AVG`, 99 rounds back to 2002) tracks 3m EURIBOR closely. So it
  splices like-for-like onto a Consensus 3m-interbank forecast.
* **Different horizon.** SPF's longer term is 4-5 calendar years ahead; Consensus is
  6-10. The ladder level-bridges over their overlap, which absorbs the horizon gap — and
  **refuses to extend at all** with fewer than 6 months of overlap, since the offset
  could not be identified and splicing at face value would silently step the endpoint.

It starts only in **2025-Q1** (7 rounds so far), so it cannot backfill history — the
Consensus snapshot still carries 2016-2025. What it does is keep the DIRECT rung alive
after that snapshot lapses, instead of dropping straight to `r* + pi^e`.

The bridge beneath both remains the deepest fallback. Whether that hand-off preserves the meaning of the anchor is an empirical
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
