# Serving the scenario sheet behind Cloudflare Access

The office-friendly deployment: the sheet lives in a **private** Supabase Storage
bucket, a **Cloudflare Worker** on your own hostname serves it, and **Cloudflare
Access** authenticates in front. Plain HTTPS on a first-party domain, no iframes,
no third-party runtime — the things corporate proxies (zscaler) object to about a
claude.ai Artifact are all gone. The page itself is unchanged: fully
self-contained, computes in the browser, nothing leaves it.

```
browser ──(Access login)── Cloudflare Worker ──(bearer key)── Supabase Storage
                                                              private bucket "sheet"
        ▲ the only door                                        ▲ not publicly readable
```

The daily pipeline (`scheduler/daily_update` → `scheduler/supabase_export`) already
runs in GitHub Actions with the Supabase secrets; `maybe_upload_sheet()` now
re-splices a fresh payload into the page and uploads it each run, so the office
sheet updates itself daily with no redeploys.

## One-time setup

### 1. Supabase (~2 minutes)

* Dashboard → Storage → **New bucket** → name `sheet`, **Public bucket OFF**.
* Recommended: let the **anon** key read this one bucket (so the Worker doesn't
  need the service key). SQL editor:

  ```sql
  create policy "anon reads sheet bucket"
  on storage.objects for select to anon
  using (bucket_id = 'sheet');
  ```

* First publish (or wait for the next scheduled run):

  ```bash
  SUPABASE_URL=... SUPABASE_KEY=<service key> python -m scheduler.supabase_export
  ```

### 2. Cloudflare Worker (~3 minutes)

```bash
cd web
npx wrangler login
npx wrangler secret put SUPABASE_URL      # https://<project>.supabase.co
npx wrangler secret put SUPABASE_KEY      # the anon key (with the policy above)
npx wrangler deploy
```

Then attach a hostname on a zone in your Cloudflare account (Workers → your worker
→ Settings → Domains & Routes → **Add custom domain**, e.g. `sheet.example.com`),
or uncomment the `routes` line in `wrangler.toml` and redeploy.

### 3. Cloudflare Access (~3 minutes)

Zero Trust dashboard → Access → Applications → **Add an application** →
Self-hosted:

* Application domain: the hostname from step 2.
* Policy: *Allow* → Include → **Emails** → your address(es). The free One-Time-PIN
  login works with any mailbox; add your IdP (Azure AD / Google) instead if the
  office uses one.
* Session duration to taste (e.g. 1 week).

That's it: the hostname now 302s to the Access login; the Worker never sees an
unauthenticated request, and the bucket never sees anything but the Worker.

If the zone isn't on Cloudflare yet, the fallback is the `workers.dev` URL with
Access enabled on it (Workers → Settings → Domains & Routes → workers.dev →
*Enable Cloudflare Access*) — functional, but a custom domain looks better to
corporate allowlists.

## Operational notes

* **Updates**: every daily-update run republishes the sheet with a fresh payload;
  the Worker caches for 5 minutes. Manual refresh: rerun `supabase_export` (or the
  GitHub Action's *workflow_dispatch*).
* **The Atlas stays where it is** (public bucket) — move it behind the same Worker
  later by pointing a second route at bucket `atlas` if wanted.
* **Rollback**: the committed `scenarios/sheet/index.html` is always the source of
  truth; the artifact publish and this deployment are two views of the same file.
* **CSV export** uses a browser-side Blob download; if the office proxy blocks
  those, the recorded next step is saving scenarios to a Supabase table instead —
  the migration and UI are deliberately not built until needed.
