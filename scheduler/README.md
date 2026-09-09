# /scheduler — operational cadence

Implements the estimation-cadence design from the top-level `README.md`: a **fast
daily** filter-forward + publish, and a **slow weekly** re-estimation, cleanly
separated so the daily job never runs the convergence-fragile optimizer.

## Scripts

| Script | Cadence | What it does |
|---|---|---|
| `daily_update.py` | **daily** | Refresh every source cache (5 rate curves + ACM/Kim-Wright/DKW/C&R benchmarks + the **TIPS liquidity "noise" gauge**), then rebuild `viz/data_daily.parquet` → `viz/data.json` → `viz/artifact.html` via the **cached** params (`viz/params/*.json`). No estimation. Per-source fault-tolerant (a failed fetch falls back to that source's cache and is noted in the summary). FRED-based sources (Kim-Wright, TIPS noise) use the official API when `FRED_API_KEY` is set. |
| `reestimate.py` | **weekly/monthly** | Re-estimate each market's state-space params and rewrite `viz/params/*.json`, **warm-started** from the existing params, **falling back to the last valid JSON on failure**. Slow + reclaim-fragile → run deliberately, off the daily job. |
| `freshness.py` | daily (last step) | **Asserts the published data is actually recent.** Judges each market/instrument against *its own* recent publication cadence, so a slow publisher is not mistaken for a dead one, and annotates + exits non-zero on a breach. See below. |
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
the `SUPABASE_ATLAS_BUCKET` secret to your bucket name). Scheduled runs fire only
from the **default branch**, so it stays dormant until this work merges; use "Run
workflow" to trigger manually meanwhile.

### Serving the rendering page — a Cloudflare Worker (not Supabase)

Supabase serves user-uploaded HTML as `text/plain` on its default `*.supabase.co`
domains — **both** Storage *and* Edge Functions (an anti-abuse policy) — so a
browser hitting either shows source, not the rendered page (rendering would need a
paid Supabase custom domain). The public, rendering URL is instead a tiny
**Cloudflare Worker** (`cloudflare/atlas-worker.js`) that reads the latest uploaded
page from the public Supabase Storage object and re-serves it as real `text/html`
on `*.workers.dev` (free, and HTML is not downgraded there):

    https://term-premium-atlas.<your-subdomain>.workers.dev

Deploy it **once** by pasting `cloudflare/atlas-worker.js` into a new Worker
(Cloudflare dashboard → Compute → Workers → Create) — see `cloudflare/README.md`.
It reads live from Storage, so every nightly data update flows through with no
redeploy. The Supabase data pipeline (Storage upload + DB tables) is unchanged;
the Worker is only the render layer.

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

## The scenario sheet behind Cloudflare Access

`supabase_export.maybe_upload_sheet()` republishes `scenarios/sheet/index.html`
each run — with a freshly rebuilt payload — to a PRIVATE Storage bucket
(`SUPABASE_SHEET_BUCKET`, default `sheet`). A Cloudflare Worker serves it on a
first-party hostname behind Cloudflare Access; setup runbook in `web/README.md`.
Inert without `SUPABASE_URL`/`SUPABASE_KEY`, like everything else here.


## Freshness: why a green run is not a fresh run

Every fetch in `daily_update.py` is guarded with a cache fallback — deliberately, so
one dead source cannot stop the other five publishing. The cost is that **a source
which quietly stops serving new days is indistinguishable from success**: the job is
green, the Atlas rebuilds, and the numbers simply stop moving. This has bitten twice.
The SNB served a frozen cube for a year with HTTP 200. The scenario sheet served a
payload stamped 2026-08-19 for three weeks because a FRED call failed inside a `try`
and the code fell back to uploading the *committed* page.

`freshness.py` is the check that turns that silence into a signal. It asks two
questions, because they fail differently:

* **Absolute** — is this series behind *today*? This is what catches a whole-pipeline
  freeze, and it is precisely what `daily_update.audit_staleness` cannot see: that
  check measures each market against the freshest market, so when everything stops
  together, no market is behind any other and nothing is reported.
* **Relative** — is this series behind the *other* markets? Catches one source dying
  while the rest keep moving.

**The budget is derived, not declared.** Markets publish on different rhythms —
Treasury same-day, the BoE and MoF at T+1, the SNB slower — and each takes its own
public holidays. A fixed "stale after N days" rule would either cry wolf on
Switzerland or sleep through a dead US feed. So each series is allowed the longest
quiet spell it has actually had lately, plus a grace margin: it is stale when it has
gone quiet for materially longer than it normally does. No per-market table to
maintain, and it adapts when a publisher changes cadence.

```bash
python -m scheduler.freshness            # print the report
python -m scheduler.freshness --check    # exit 1 if anything is stale (the CI gate)
```

### Where it sits in the workflow, and why it is last

It runs **after the publish** and **after the Bund commit**, for two separate reasons:

* *After the publish*, because publishing the best data available still beats
  publishing nothing. The check must not gate the publish — it turns the **run** red
  afterwards and annotates each offending series in the run summary.
* *After the Bund commit*, because a non-zero exit aborts the job, and the Bundesbank
  per-ISIN cross-section is the one store whose history **cannot be re-fetched**.
  Failing before that step would discard a trading day of it permanently — a far worse
  outcome than a stale feed.

`supabase_export.py`'s sheet-payload fallback is annotated the same way, so a sheet
that silently reverts to the committed page shows up in the run UI rather than at
line 700 of a log nobody opens.
