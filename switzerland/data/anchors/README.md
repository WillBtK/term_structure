# Switzerland neutral-rate anchor — survey rungs

Seed data for the shifting-endpoint ladder (`model/neutral_anchor.py`), delivered
2026-08-20 as a ONE-OFF from Consensus Economics long-term surveys (6-10 years
ahead). Cannot be refreshed; public rungs carry the anchor forward.

| file | variable | coverage |
|---|---|---|
| `consensus_cpi_ch_6_10y.csv` | CPI inflation | 1998-10 .. 2026-07 |
| `consensus_shortrate_ch_6_10y.csv` | 3-month interbank | 2017-01 .. 2026-07 |

Format: `date,value` — survey publication date, percent p.a. Semiannual to 2013,
quarterly from 2014. The CPI column passed the post-2021 plausibility test (mild
bump 1.03->1.18, easing to ~1.0 — consistent with Switzerland's small shock).

PROPRIETARY: commit privately only. Publish derived premia, never these inputs.
