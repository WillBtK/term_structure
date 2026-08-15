# /scheduler — operational cadence

Implements the estimation-cadence design from the top-level `README.md`: a **fast
daily** filter-forward + publish, and a **slow weekly** re-estimation, cleanly
separated so the daily job never runs the convergence-fragile optimizer.

## Scripts

| Script | Cadence | What it does |
|---|---|---|
| `daily_update.py` | **daily** | Refresh every source cache (5 rate curves + ACM/Kim-Wright/DKW/C&R benchmarks), then rebuild `viz/data_daily.parquet` → `viz/data.json` → `viz/artifact.html` via the **cached** params (`viz/params/*.json`). No estimation. Per-source fault-tolerant (a failed fetch falls back to that source's cache and is noted in the summary). |
| `reestimate.py` | **weekly/monthly** | Re-estimate each market's state-space params and rewrite `viz/params/*.json`, **warm-started** from the existing params, **falling back to the last valid JSON on failure**. Slow + reclaim-fragile → run deliberately, off the daily job. |
| `supabase_export.py` | daily (with the daily job) | Write upload-ready long-form CSVs (`term_premia`, `benchmarks`) and, when `SUPABASE_URL`/`SUPABASE_KEY` are set (+ `supabase.co` allowlisted), upsert them. **Inert without credentials** — writes staging files and logs. |

## Daily schedule + publish

A Claude Code Remote **Routine** (cron trigger) fires a fresh session at **00:00 UTC
(= 7pm EST / 8pm EDT, fixed)** which:
1. `python -m scheduler.daily_update` — refresh + rebuild (fast path),
2. commits & pushes `viz/data.json`, `viz/artifact.html` (data caches are gitignored)
   to `claude/us-yield-curve-plan-ryw3is`,
3. `python -m scheduler.supabase_export` — stage (+ upload if creds),
4. republishes the Atlas to its existing artifact URL.

00:00 UTC is fixed (a 1-hour seasonal drift under US daylight saving is accepted);
it is late enough in the US day that same-day GSW / ACM / DKW / FRED updates have
posted. A fired session inherits this environment's repo write access, the source
allowlist, and the Artifact tool. (A GitHub Action could run steps 1–3 with open
egress, but cannot republish the Artifact, so a fired session is the single
mechanism.)

## Manual runs

```bash
export PYTHONPATH=$(pwd)
python -m scheduler.daily_update            # full daily refresh + rebuild
python -m scheduler.daily_update --no-fetch # rebuild from existing caches only
python -m scheduler.reestimate              # weekly: re-estimate all viz/params/*.json
python -m scheduler.reestimate --only us_prod
python -m scheduler.supabase_export         # stage (+ upload if creds set)
```

## Notes
- The daily job is a single EKF filter-forward pass per market with fixed params —
  fast (~seconds per market) and robust. Only re-estimation is slow; keep it weekly.
- FRED egress through the environment proxy is intermittent (HTTP/2 stream resets);
  the Kim-Wright benchmark cache fills in on a later daily run and its Atlas trace
  appears automatically once present.
