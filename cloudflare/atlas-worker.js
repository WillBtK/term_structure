// Term Premium Atlas — Cloudflare Worker serving layer.
//
// WHY: Supabase serves user HTML as text/plain on its default *.supabase.co
// domains (both Storage and Edge Functions), so the page shows source, not the
// rendered app. Cloudflare Workers serve on *.workers.dev, which returns HTML
// correctly. This Worker fetches the latest Atlas that the daily pipeline
// uploads to the public Supabase Storage bucket and re-serves it as text/html.
//
// Deploy ONCE (Cloudflare dashboard -> Workers & Pages -> Create Worker ->
// paste this -> Deploy). It reads live from Storage, so every nightly data
// refresh flows through with no redeploy. Public URL:
//   https://<worker-name>.<your-subdomain>.workers.dev
//
// The Supabase data pipeline (Storage upload + DB tables) is unchanged; this is
// purely the render layer on a host that allows HTML.

const SRC =
  "https://xzhoykybwlkefgvgwnii.supabase.co/storage/v1/object/public/atlas/index.html";

export default {
  async fetch(_request) {
    let upstream;
    try {
      upstream = await fetch(SRC, { cf: { cacheTtl: 300, cacheEverything: true } });
    } catch (e) {
      return new Response("Atlas fetch failed: " + e, { status: 502 });
    }
    if (!upstream.ok) {
      return new Response("Atlas unavailable (upstream " + upstream.status + ")", { status: 502 });
    }
    const html = await upstream.text();
    return new Response(html, {
      status: 200,
      headers: {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "public, max-age=300",
      },
    });
  },
};
