// Term Premium Atlas — public HTML serving layer.
//
// Supabase Storage serves user-uploaded HTML as text/plain (an anti-abuse policy),
// so a browser hitting the Storage URL sees source instead of the rendered page.
// This Edge Function fetches the latest uploaded Atlas from Storage and returns it
// with the correct `text/html` Content-Type. Deploy ONCE — it always serves the
// newest daily build (it reads live from Storage).
//
// Public URL after deploy:  https://<project-ref>.supabase.co/functions/v1/atlas

const BUCKET = Deno.env.get("ATLAS_BUCKET") ?? "atlas";
const OBJECT = "index.html";
const BASE = Deno.env.get("SUPABASE_URL"); // auto-injected in Edge Functions

Deno.serve(async (_req: Request) => {
  if (!BASE) return new Response("SUPABASE_URL not set", { status: 500 });
  const src = `${BASE}/storage/v1/object/public/${BUCKET}/${OBJECT}`;
  let html: string;
  try {
    const upstream = await fetch(src);
    if (!upstream.ok) {
      return new Response(`Atlas not available (upstream ${upstream.status})`, { status: 502 });
    }
    html = await upstream.text();
  } catch (e) {
    return new Response(`Atlas fetch failed: ${e}`, { status: 502 });
  }
  const headers = new Headers();
  headers.set("content-type", "text/html; charset=utf-8");
  headers.set("cache-control", "public, max-age=300");
  headers.set("x-atlas-fn", "1"); // marker so we can confirm this function served the response
  return new Response(html, { status: 200, headers });
});
