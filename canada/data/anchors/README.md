# Canada neutral-rate anchor — survey rungs

Seed data for the shifting-endpoint ladder (`model/neutral_anchor.py`), delivered
2026-08-20 as a ONE-OFF from Consensus Economics long-term surveys (6-10 years
ahead). Cannot be refreshed; public rungs carry the anchor forward.

| file | variable | coverage |
|---|---|---|
| `consensus_cpi_ca_6_10y.csv` | CPI inflation | 1990-04 .. 2026-07 |
| `consensus_shortrate_ca_6_10y.csv` | 3-month T-bill | 2017-01 .. 2026-07 |

Format: `date,value` — survey publication date, percent p.a. Semiannual to 2013,
quarterly from 2014. One correction applied on ingest: the 2026-04-13 T-bill cell
of the re-extraction duplicated the 2026-07-13 value; the distinct reading from
the first delivery (2.4241) was restored.

PROPRIETARY: commit privately only. Publish derived premia, never these inputs.
