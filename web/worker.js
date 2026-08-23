/* Serves the Term Premium scenario sheet from a PRIVATE Supabase Storage bucket.
 *
 * The Worker is the only thing that can read the bucket (it authenticates with a
 * key held as a Worker secret), and Cloudflare Access sits in front of the Worker's
 * hostname — so Access is the single door to the page. See web/README.md for the
 * full setup runbook.
 *
 * Secrets / vars (wrangler secret put …):
 *   SUPABASE_URL   e.g. https://abcdefgh.supabase.co
 *   SUPABASE_KEY   a key that can read the bucket — the anon key with a storage
 *                  read policy on the bucket (recommended; see README), or the
 *                  service_role key (works, but broader than needed)
 * Optional vars (wrangler.toml [vars]):
 *   SHEET_BUCKET   default "sheet"
 *   SHEET_OBJECT   default "index.html"
 */
export default {
  async fetch(request, env) {
    if (request.method !== "GET" && request.method !== "HEAD")
      return new Response("method not allowed", { status: 405 });

    const bucket = env.SHEET_BUCKET || "sheet";
    const object = env.SHEET_OBJECT || "index.html";
    const origin = `${env.SUPABASE_URL.replace(/\/$/, "")}/storage/v1/object/${bucket}/${object}`;

    const upstream = await fetch(origin, {
      headers: { Authorization: `Bearer ${env.SUPABASE_KEY}`, apikey: env.SUPABASE_KEY },
      // Let Cloudflare's edge cache absorb repeat loads for a few minutes; the
      // sheet only changes once a day when the pipeline republishes it.
      cf: { cacheTtl: 300, cacheEverything: true },
    });
    if (!upstream.ok)
      return new Response(`upstream ${upstream.status} — is the bucket/object present?`,
                          { status: 502 });

    return new Response(upstream.body, {
      status: 200,
      headers: {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "private, max-age=300",
        // the page is fully self-contained except Google Fonts
        "content-security-policy":
          "default-src 'self' 'unsafe-inline'; img-src 'self' data:; " +
          "font-src https://fonts.gstatic.com; style-src 'self' 'unsafe-inline' " +
          "https://fonts.googleapis.com; connect-src 'none'; frame-ancestors 'none'",
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
      },
    });
  },
};
