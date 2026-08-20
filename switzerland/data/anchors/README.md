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


## KOF Consensus Forecast (public, auto-refreshed — added 2026-08-20)

No account is required: the KOF TSDB API v2 serves its public tier anonymously
(`access_type=public`); the login wall guards only non-public series. Two medians are
kept current weekly by `switzerland/fetch_neutral.py`:

| file | series | horizon | coverage |
|---|---|---|---|
| `kof_cpi_5y.csv` | CPI expectation | **5 years ahead** | 2015-Q1+ quarterly |
| `kof_3m_12m.csv` | 3-month rate expectation | 12 months ahead | 2001-Q1+ quarterly |

KOF's 5-year block covers CPI/GDP/unemployment only — there is NO long-horizon
interest-rate question, so neither series is wired into the endpoint construction.
The recorded options for the Swiss carry-forward once the Consensus snapshot goes
stale: (a) frozen hold (current behaviour); (b) live KOF pi_5y plus a constant real
component calibrated on the 2017-2026 survey overlap — survey-only, no model r*,
but the constant embeds the calibration window. Decision pending.
Cross-check at handover: KOF pi_5y 0.90% vs Consensus 6-10y CPI 0.97% (7bp apart).
