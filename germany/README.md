# /germany — placeholder (not built in Phase 1)

Per the top-level `README.md` **Build sequence**, this country is deferred.
Phase 1 delivers the **US** leg only (`/us`, `/model`, `/validation`).

When this country is built, it will contain a data loader that produces the same
data contract the engine expects — a `DataFrame` indexed by date, columns =
maturities in years on the common grid `[0.25, 0.5, 1, 2, 3, 5, 7, 10]`, values
= zero-coupon yields in decimal — mirroring `/us`. Do not fork `/model`.

See the top-level `README.md` for this country's data source, method, and
effective-lower-bound value.
