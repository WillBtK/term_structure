# Cloudflare Worker — Atlas render layer

Supabase serves user-uploaded HTML as `text/plain` on its default `*.supabase.co`
domains (**both** Storage and Edge Functions — an anti-abuse policy), so the Atlas
page shows source instead of rendering. Cloudflare Workers serve on `*.workers.dev`,
which returns HTML correctly. `atlas-worker.js` fetches the latest Atlas that the
daily pipeline uploads to the public Supabase Storage bucket and re-serves it as
real `text/html`.

The Supabase data pipeline is unchanged — this Worker is only the render layer.

## Deploy (once, ~2 minutes, no CLI / token)

1. Sign in to Cloudflare (free) → **Workers & Pages** → **Create** → **Create Worker**.
2. Name it e.g. `term-premium-atlas`; click **Deploy** (deploys a placeholder).
3. Click **Edit code**, delete the placeholder, paste all of `atlas-worker.js`, **Deploy**.
4. Your public URL is shown at the top, e.g.
   `https://term-premium-atlas.<your-subdomain>.workers.dev`.

That URL renders the Atlas and **auto-updates nightly** (it reads live from Storage,
so no redeploy is needed when the data changes). Free tier is 100k requests/day.

## Notes
- If you rename the Storage bucket, update `SRC` in `atlas-worker.js` and redeploy.
- To use your own domain later, add a Route/Custom Domain to the Worker in Cloudflare.
