// Term Premium Atlas — public HTML serving layer.
//
// Supabase Storage serves user-uploaded HTML as text/plain (an anti-abuse policy),
// so a browser hitting the Storage URL sees source instead of the rendered page.
// This Edge Function proxies the latest uploaded Atlas and returns it with the
// correct `text/html` Content-Type. Deploy ONCE — it always serves the newest
// daily build because it reads live from Storage (the daily job overwrites the
// object each night). No redeploy needed when the data updates.
//
// Public URL after deploy:  https://<project-ref>.supabase.co/functions/v1/atlas

const BUCKET = Deno.env.get("ATLAS_BUCKET") ?? "atlas";
const OBJECT = "index.html";
const BASE = Deno.env.get("SUPABASE_URL"); // auto-injected in Edge Functions

Deno.serve(async (_req: Request) => {
  if (!BASE) return new Response("SUPABASE_URL not set", { status: 500 });
  const src = `${BASE}/storage/v1/object/public/${BUCKET}/${OBJECT}`;
  let upstream: Response;
  try {
    upstream = await fetch(src);
  } catch (e) {
    return new Response(`Atlas fetch failed: ${e}`, { status: 502 });
  }
  if (!upstream.ok || !upstream.body) {
    return new Response(`Atlas not available (upstream ${upstream.status})`, { status: 502 });
  }
  // Stream the bytes through, only rewriting the Content-Type (and a short cache).
  return new Response(upstream.body, {
    status: 200,
    headers: {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "public, max-age=300",
    },
  });
});
