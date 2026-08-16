# /scheduler — operational cadence

Implements the estimation-cadence design from the top-level `README.md`: a **fast
daily** filter-forward + publish, and a **slow weekly** re-estimation, cleanly
separated so the daily job never runs the convergence-fragile optimizer.

## Scripts

| Script | Cadence | What it does |
|---|---|---|
| `daily_update.py` | **daily** | Refresh every source cache (5 rate curves + ACM/Kim-Wright/DKW/C&R benchmarks + the **TIPS liquidity "noise" gauge**), then rebuild `viz/data_daily.parquet` → `viz/data.json` → `viz/artifact.html` via the **cached** params (`viz/params/*.json`). No estimation. Per-source fault-tolerant (a failed fetch falls back to that source's cache and is noted in the summary). FRED-based sources (Kim-Wright, TIPS noise) use the official API when `FRED_API_KEY` is set. |
| `reestimate.py` | **weekly/monthly** | Re-estimate each market's state-space params and rewrite `viz/params/*.json`, **warm-started** from the existing params, **falling back to the last valid JSON on failure**. Slow + reclaim-fragile → run deliberately, off the daily job. |
| `supabase_export.py` | daily (with the daily job) | Write upload-ready long-form CSVs (`term_premia`, `benchmarks` — the latter includes the TIPS noise gauge as `source="tips_noise"`) and, when `SUPABASE_URL`/`SUPABASE_KEY` are set (+ `supabase.co` allowlisted), upsert them **and** publish the self-contained Atlas (`viz/artifact.html`) to a **PUBLIC Supabase Storage bucket** as `index.html`. **Inert without credentials** — writes staging files and logs. |

## Daily schedule + publish

**GitHub Actions + Supabase Storage** (`.github/workflows/daily-update.yml`), cron
**00:00 UTC (= 7pm EST / 8pm EDT, fixed)**. On a GitHub runner (open egress — no
proxy flakiness, so FRED/NY-Fed/etc. all resolve) it runs `daily_update`, then
`supabase_export`. When the Supabase secrets are set, `supabase_export` upserts the
dataset tables **and uploads `viz/artifact.html` to a public Supabase Storage
bucket** — a stable, auto-updating public URL you can share **while this GitHub repo
stays private** (GitHub Pages would require a public repo on the free tier, which is
why Pages was dropped). No Artifact tool and no daily 13 MB git commit needed.

One-time owner setup: enable Actions; add repo secrets `SUPABASE_URL` /
`SUPABASE_KEY`; in Supabase create a **public** Storage bucket named `atlas` (or set
the `SUPABASE_ATLAS_BUCKET` secret to your bucket name). The public URL is then
`<SUPABASE_URL>/storage/v1/object/public/<bucket>/index.html`. Scheduled runs fire
only from the **default branch**, so it stays dormant until this work merges; use
"Run workflow" to trigger manually meanwhile.

**Why not auto-republish the claude.ai Artifact:** neither a GitHub runner nor a
Claude Code Remote fired session has the **Artifact tool** (verified — the tool is
only in interactive sessions), so the live *claude.ai* Atlas is refreshed
interactively (or you share the Supabase Storage URL, which updates automatically).
00:00 UTC is fixed (1-hour seasonal DST drift accepted); it is late enough that
same-day GSW / ACM / DKW / FRED updates have posted.

**Fallback (disabled): a Claude Code Remote Routine** (`trig_017cXP2SMdLX3MiP7Hc7tzs4`,
currently disabled). It fires a fresh session that runs `daily_update` +
`supabase_export` + git-push (data-only; no republish). It works without GitHub
setup but uses the flakier sandbox egress and can't publish the live Atlas — re-enable
it via `update_trigger(enabled=true)` only if you prefer it over GitHub Actions.

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
