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
    // Nothing in here is allowed to crash into Cloudflare's opaque "Error 1101"
    // page: every failure comes back as readable text naming the fix.
    try {
      return await serve(request, env);
    } catch (err) {
      return new Response(
        "Worker error: " + (err && err.message ? err.message : String(err)) +
        "\nCheck the SUPABASE_URL secret is the full URL including https:// " +
        "(e.g. https://abcdefgh.supabase.co), then press Deploy.",
        { status: 500 });
    }
  },
};

async function serve(request, env) {
    if (request.method !== "GET" && request.method !== "HEAD")
      return new Response("method not allowed", { status: 405 });

    // Plain-English failures: each of these tells you which setup step to revisit.
    if (!env.SUPABASE_URL || !env.SUPABASE_KEY)
      return new Response(
        "Setup incomplete: add the two secrets SUPABASE_URL and SUPABASE_KEY " +
        "under this worker's Settings -> Variables and Secrets, then Deploy. " +
        "(Values: your Supabase Project URL and the anon/public API key.)",
        { status: 500 });

    const bucket = env.SHEET_BUCKET || "sheet";
    // /deck serves the presentation from the same private bucket; anything else
    // serves the scenario sheet
    const path = new URL(request.url).pathname;
    const object = path.startsWith("/deck") ? "deck.html"
                 : path.startsWith("/summary") ? "summary.html"
                 : (env.SHEET_OBJECT || "index.html");
    // tolerate a URL pasted without the scheme
    let base = env.SUPABASE_URL.trim().replace(/\/+$/, "");
    if (!/^https?:\/\//.test(base)) base = "https://" + base;
    const origin = `${base}/storage/v1/object/${bucket}/${object}`;

    const upstream = await fetch(origin, {
      headers: { Authorization: `Bearer ${env.SUPABASE_KEY}`, apikey: env.SUPABASE_KEY },
      // Let Cloudflare's edge cache absorb repeat loads for a few minutes; the
      // sheet only changes once a day when the pipeline republishes it.
      cf: { cacheTtl: 300, cacheEverything: true },
    });
    if (upstream.status === 404 || upstream.status === 400)
      return new Response(
        `Supabase says ${upstream.status}: the file "${object}" isn't in bucket ` +
        `"${bucket}" yet. Check the bucket exists with exactly that name, and run ` +
        "the GitHub Action (Actions -> Daily data update -> Run workflow) to " +
        "upload the sheet.",
        { status: 502 });
    if (upstream.status === 401 || upstream.status === 403)
      return new Response(
        `Supabase says ${upstream.status}: the key can't read the bucket. Check ` +
        "the SQL policy from web/README step 1.3 was run, and that SUPABASE_KEY " +
        "is the anon/public key from Settings -> API Keys.",
        { status: 502 });
    if (!upstream.ok)
      return new Response(`upstream ${upstream.status} from Supabase`, { status: 502 });

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
}
