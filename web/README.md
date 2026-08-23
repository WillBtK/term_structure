# Putting the scenario sheet behind Cloudflare Access — browser-only setup

Everything here is done in three websites: **supabase.com**, **github.com** and
**dash.cloudflare.com**. No command line needed.

What you are building: the sheet is stored in a private folder ("bucket") inside
your Supabase project, a small Cloudflare Worker serves it at an address like
`sheet.yourdomain.com`, and Cloudflare Access makes people log in before they can
see it. Because the bucket is private, the login page is the only way in. The
daily pipeline you already run on GitHub re-uploads the sheet with fresh numbers
every day, so once this is set up you never touch it again.

```
you (after logging in) ── Cloudflare Worker ── private Supabase bucket "sheet"
```

## Step 1 — Supabase (about 2 minutes)

1. Go to **supabase.com**, open your project.
2. Left sidebar → **Storage** → **New bucket**. Name it `sheet`. Leave
   **"Public bucket" switched OFF**. Save.
3. Left sidebar → **SQL Editor** → **New query**. Paste this and press **Run**
   (it lets the "anon" key read this one bucket and nothing else):

   ```sql
   create policy "anon reads sheet bucket"
   on storage.objects for select to anon
   using (bucket_id = 'sheet');
   ```

4. Copy two values into a notepad for Step 3:
   * **Project URL** — Settings (gear icon) → **Data API** → "Project URL"
     (looks like `https://abcdefgh.supabase.co`).
   * **anon key** — Settings → **API Keys** → the key labelled `anon` / `public`
     (the long one; this is the *publishable* key, not the secret one).

## Step 2 — Publish the sheet into the bucket (one click)

Go to the GitHub repository → **Actions** tab → **Daily data update** →
**Run workflow** button (top right) → green **Run workflow**. Wait for the run to
go green (~15 minutes). The pipeline already has your Supabase credentials from
the Atlas setup; this run uploads the sheet into the new bucket. (It will also
happen automatically every day from now on.)

## Step 3 — Cloudflare Worker (about 3 minutes)

1. Go to **dash.cloudflare.com** → **Workers & Pages** → **Create** →
   **Create Worker**. Name it `term-premium-sheet`, press **Deploy** (it deploys
   a placeholder).
2. Press **Edit code**. In another browser tab, open `web/worker.js` in this
   GitHub repository, press the **Raw** button, select all, copy. Back in the
   Cloudflare editor: select all, paste over the placeholder, press **Deploy**.
3. Back on the worker's page → **Settings** → **Variables and Secrets** →
   **Add**:
   * Type **Secret**, name `SUPABASE_URL`, value = the Project URL from Step 1.
   * Type **Secret**, name `SUPABASE_KEY`, value = the anon key from Step 1.

   Save and deploy when prompted.
4. Give it an address — **Settings** → **Domains & Routes**:
   * Best: **Add** → **Custom domain** → e.g. `sheet.yourdomain.com` (works if
     your domain is on Cloudflare).
   * Otherwise: use the `….workers.dev` address it already has.

At this point the address should show the sheet — but open to anyone who knows
it. Step 4 closes that.

## Step 4 — Cloudflare Access (about 3 minutes)

1. Go to **one.dash.cloudflare.com** (Zero Trust). First visit asks you to pick a
   free team name — anything.
2. **Access** → **Applications** → **Add an application** → **Self-hosted**.
3. Application domain: the address from Step 3 (the custom domain, or the
   workers.dev one). Name it "Term premium sheet".
4. Add a policy: Action **Allow** → Include → **Emails** → your email address
   (add colleagues' addresses here too, or your office login provider if you use
   one). Save.

Done. Visiting the address now shows a Cloudflare login page — enter your email,
receive a one-time code, and you're in. Sessions last as long as you set (a week
is a comfortable default).

## Day to day

* **The sheet updates itself** — every daily pipeline run re-uploads it with
  fresh data. Nothing to redo.
* **If the numbers look stale**, re-run Step 2's workflow button.
* **The Atlas is unaffected** (it stays on its public link).
* **If the office proxy also blocks the "Export CSV" download**, say so — the
  next step recorded for that is saving scenarios into Supabase instead of
  downloading a file, and it's deliberately not built until needed.

<details><summary>For a machine with a command line (optional, not required)</summary>

```bash
cd web
npx wrangler login
npx wrangler secret put SUPABASE_URL
npx wrangler secret put SUPABASE_KEY
npx wrangler deploy
```
</details>
