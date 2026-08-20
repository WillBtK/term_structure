# UK neutral-rate anchor — survey rungs

Seed data for the shifting-endpoint ladder (`model/neutral_anchor.py`), delivered
2026-08-20 as a ONE-OFF from Consensus Economics long-term surveys (6-10 years
ahead). This data cannot be refreshed — the ladder's public rungs carry the anchor
forward and `bridge_fidelity()` decides whether that hand-off is trustworthy.

| file | variable | coverage | notes |
|---|---|---|---|
| `consensus_cpi_uk_6_10y.csv` | CPI inflation | 2004-10 .. 2026-07 | starts with CPI targeting |
| `consensus_rpi_uk_6_10y.csv` | RPI inflation | 1990-04 .. 2026-07 | wedge vs CPI ~1.0pp (2017+) |
| `consensus_shortrate_uk_6_10y.csv` | 3-month interbank | 2017-01 .. 2026-07 | |

Format: `date,value` — survey publication date, percent p.a. Semiannual (Apr/Oct)
to 2013, quarterly from 2014. Values to 2004 are published roundings (0.1pp);
later values are full-precision panel means.

PROPRIETARY: commit privately only (this repo is private). Publish derived
premia, never these inputs.
