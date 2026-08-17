"""Global term-premium factor — a reduced-form dynamic factor model (DFM).

Decomposes the five markets' term-premium series into a single **global** cyclical
factor plus market-specific **local** residuals:

    z_{i,t} = λ_i · G_t + e_{i,t},   e_{i,t} ~ N(0, r_i)   (diagonal R)
    G_t     = φ · G_{t-1} + u_t,     u_t   ~ N(0, q),  q = 1 − φ²   (Var(G)=1)

where z is the per-market term premium standardized over the balanced-overlap window.
Estimated by Kalman filter + RTS smoother + EM (closed-form M-step), which handles the
**ragged** panel (markets appear at different dates) by masking missing measurements —
so full history is usable, with G simply less identified where few markets exist.

This is a *reduced-form* model on the already-estimated TP series (from the per-country
AFNS + shadow-rate pipeline); it does NOT re-estimate any yield curve. The global factor
is the common risk-compensation cycle; the local residual `TP_i − (α_i + β_i·G_t)` is a
relative-value read (which curve is cheap/rich vs the global cycle). See
`validation/global_factor_rv.py` for the portfolio diagnostic and
`validation/FINDINGS_global_factor.md`.

Pure numpy (no I/O); the ``__main__`` runs it on the canonical dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

_PHI_MAX = 0.99995


@dataclass
class DFMParams:
    lam: np.ndarray        # (N,) loadings λ_i in standardized space
    phi: float             # AR(1) coefficient of the latent factor
    r: np.ndarray          # (N,) idiosyncratic variances (standardized space)
    mu: np.ndarray         # (N,) per-market means (raw TP units) for raw mapping
    sd: np.ndarray         # (N,) per-market SDs (raw TP units)
    names: list             # market order, e.g. ["US","UK","DE","JP","CH"]

    @property
    def n(self) -> int:
        return len(self.lam)

    @property
    def beta(self) -> np.ndarray:
        """Raw-unit loading β_i = λ_i·s_i (Var(G)=1)."""
        return self.lam * self.sd

    @property
    def alpha(self) -> np.ndarray:
        """Raw-unit intercept α_i = μ_i (since E[G]=0)."""
        return self.mu.copy()

    @property
    def communality(self) -> np.ndarray:
        """Share of each market's (standardized) TP variance explained by G:
        λ_i²·Var(G)/(λ_i²·Var(G)+r_i) with Var(G)=1."""
        return self.lam ** 2 / (self.lam ** 2 + self.r)

    def to_json(self, path) -> None:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}
        from pathlib import Path
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path) -> "DFMParams":
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        return cls(lam=np.array(d["lam"]), phi=float(d["phi"]), r=np.array(d["r"]),
                   mu=np.array(d["mu"]), sd=np.array(d["sd"]), names=list(d["names"]))


# --------------------------------------------------------------------- standardize
def standardize(Y: np.ndarray, overlap_mask: np.ndarray | None = None):
    """Standardize each column of Y (T,N) to zero mean / unit SD using the rows in
    ``overlap_mask`` (default: all finite). NaNs are preserved. Returns (Z, mu, sd)."""
    Y = np.asarray(Y, float)
    T, N = Y.shape
    mu = np.empty(N); sd = np.empty(N)
    for i in range(N):
        col = Y[:, i]
        m = np.isfinite(col) & (overlap_mask if overlap_mask is not None else True)
        vals = col[m]
        mu[i] = np.nanmean(vals) if vals.size else 0.0
        sd[i] = np.nanstd(vals) if vals.size else 1.0
        if not np.isfinite(sd[i]) or sd[i] < 1e-12:
            sd[i] = 1.0
    Z = (Y - mu) / sd
    return Z, mu, sd


# ------------------------------------------------------------------- Kalman (scalar)
def _kalman_filter(Z, lam, phi, r, q):
    """Scalar-state Kalman filter over Z (T,N) with ragged NaN masking. Measurements
    are conditionally independent given the scalar state (diagonal R), so each observed
    series is applied as a sequential 1-D update. Returns filtered means/vars, the
    one-step predicted means/vars (for the smoother), and the observed-data loglik."""
    T, N = Z.shape
    g_f = np.empty(T); P_f = np.empty(T)
    g_p = np.empty(T); P_p = np.empty(T)
    ll = 0.0
    g, P = 0.0, 1.0                                   # stationary prior: mean 0, var 1
    for t in range(T):
        gp = phi * g; Pp = phi * phi * P + q          # predict
        g_p[t] = gp; P_p[t] = Pp
        g, P = gp, Pp
        row = Z[t]
        for i in range(N):
            z = row[i]
            if not np.isfinite(z):
                continue
            s = lam[i] * lam[i] * P + r[i]            # innovation variance
            v = z - lam[i] * g                        # innovation
            ll += -0.5 * (np.log(2 * np.pi) + np.log(s) + v * v / s)
            k = P * lam[i] / s                         # gain
            g = g + k * v
            P = (1.0 - k * lam[i]) * P
        g_f[t] = g; P_f[t] = P
    return g_f, P_f, g_p, P_p, ll


def _rts_smoother(g_f, P_f, g_p, P_p, phi):
    """RTS smoother → smoothed means, vars, and lag-one covariances."""
    T = g_f.shape[0]
    g_s = g_f.copy(); P_s = P_f.copy(); P_lag = np.zeros(T)     # P_lag[t]=Cov(G_t,G_{t-1}|Y)
    for t in range(T - 2, -1, -1):
        J = P_f[t] * phi / P_p[t + 1] if P_p[t + 1] > 0 else 0.0
        g_s[t] = g_f[t] + J * (g_s[t + 1] - g_p[t + 1])
        P_s[t] = P_f[t] + J * J * (P_s[t + 1] - P_p[t + 1])
        P_lag[t + 1] = J * P_s[t + 1]
    return g_s, P_s, P_lag


# --------------------------------------------------------------------------- PCA
def pca_first_factor(Z_bal: np.ndarray):
    """First principal component of the balanced (no-NaN rows) standardized panel.
    Returns (scores g (T,), loadings (N,)) with the sign anchored so loadings[0]>0."""
    Z = np.asarray(Z_bal, float)
    good = np.all(np.isfinite(Z), axis=1)
    Zc = Z[good]
    C = np.cov(Zc, rowvar=False)
    w, V = np.linalg.eigh(C)
    v = V[:, -1]                                       # top eigenvector
    if v[0] < 0:
        v = -v
    scores = np.full(Z.shape[0], np.nan)
    scores[good] = Zc @ v
    return scores, v


# --------------------------------------------------------------------------- EM
def estimate_dfm(Y: np.ndarray, names, overlap_mask=None, *, max_iter=500, tol=1e-7,
                 warm: DFMParams | None = None) -> tuple:
    """EM estimate of the 1-factor DFM on the raw TP panel Y (T,N). Standardizes
    internally (over ``overlap_mask``), warm-starts from PCA (or ``warm``), iterates
    Kalman+RTS+closed-form M-step to loglik convergence, and sign-anchors λ[0]>0.
    Returns (DFMParams, dict of diagnostics)."""
    Y = np.asarray(Y, float)
    T, N = Y.shape
    Z, mu, sd = standardize(Y, overlap_mask)
    # --- init (PCA warm start on the balanced sub-panel) ---
    g_pca, load_pca = pca_first_factor(Z)
    if warm is not None and warm.n == N:
        lam = warm.lam.copy(); phi = float(warm.phi); r = warm.r.copy()
    else:
        lam = load_pca * np.nanstd(g_pca[np.isfinite(g_pca)])   # scale loadings to PC sd
        lam = np.where(np.isfinite(lam) & (np.abs(lam) > 1e-6), lam, 0.5)
        gg = g_pca[np.isfinite(g_pca)]
        phi = float(np.clip(np.corrcoef(gg[:-1], gg[1:])[0, 1], 0.0, _PHI_MAX)) if gg.size > 2 else 0.9
        r = np.clip(1.0 - lam ** 2, 0.05, 0.99)
    q = 1.0 - phi * phi
    prev_ll = -np.inf; n_iter = 0
    for n_iter in range(1, max_iter + 1):
        g_f, P_f, g_p, P_p, ll = _kalman_filter(Z, lam, phi, r, q)
        g_s, P_s, P_lag = _rts_smoother(g_f, P_f, g_p, P_p, phi)
        EG = g_s
        EG2 = P_s + g_s ** 2                            # E[G_t²|Y]
        EGG = P_lag + g_s * np.r_[0.0, g_s[:-1]]        # E[G_t G_{t-1}|Y] (t>=1)
        # M-step: loadings + idiosyncratic variances (per market, over observed rows)
        for i in range(N):
            obs = np.isfinite(Z[:, i])
            zi = Z[obs, i]; g = EG[obs]; g2 = EG2[obs]
            denom = g2.sum()
            lam[i] = (zi * g).sum() / denom if denom > 0 else 0.0
            resid2 = (zi ** 2 - 2 * lam[i] * zi * g + lam[i] ** 2 * g2).sum()
            r[i] = max(resid2 / max(obs.sum(), 1), 1e-4)
        # M-step: AR(1) coefficient (then re-pin q = 1-φ² for unit factor variance)
        num = EGG[1:].sum(); den = EG2[:-1].sum()
        phi = float(np.clip(num / den if den > 0 else 0.0, -_PHI_MAX, _PHI_MAX))
        q = 1.0 - phi * phi
        if abs(ll - prev_ll) < tol * (1 + abs(prev_ll)):
            prev_ll = ll
            break
        prev_ll = ll
    # sign-anchor: G loads positively on the first market (US)
    if lam[0] < 0:
        lam = -lam
    p = DFMParams(lam=lam, phi=phi, r=r, mu=mu, sd=sd, names=list(names))
    # PCA cross-check correlation (on the balanced window)
    g_s_final, _, _ = _rts_smoother(*_kalman_filter(Z, lam, phi, r, q)[:4], phi)
    good = np.isfinite(g_pca)
    pca_corr = float(np.corrcoef(g_s_final[good], g_pca[good])[0, 1]) if good.sum() > 2 else np.nan
    diag = {"loglik": float(prev_ll), "n_iter": n_iter, "phi": phi,
            "halflife_days": float(np.log(2) / -np.log(phi)) if 0 < phi < 1 else np.inf,
            "pca_corr": abs(pca_corr), "communality": p.communality.tolist()}
    return p, diag


# --------------------------------------------------------------------- decompose
def filter_forward(p: DFMParams, Y: np.ndarray, smoothed: bool = False) -> dict:
    """Given fixed params, return the factor path and the raw-unit decomposition.
    ``smoothed=False`` (default) gives the **real-time filtered** factor (no look-ahead —
    use this for the RV signal); ``smoothed=True`` gives the full-sample smoothed factor
    (less noisy — use for in-sample description/plots)."""
    Z = (np.asarray(Y, float) - p.mu) / p.sd
    q = 1.0 - p.phi * p.phi
    g_f, P_f, g_p, P_p, _ = _kalman_filter(Z, p.lam, p.phi, p.r, q)
    G = _rts_smoother(g_f, P_f, g_p, P_p, p.phi)[0] if smoothed else g_f
    fitted = p.alpha[None, :] + G[:, None] * p.beta[None, :]     # (T,N) raw TP units
    residual = np.asarray(Y, float) - fitted
    return {"G": G, "fitted": fitted, "residual": residual}


def variance_decomposition(p: DFMParams) -> dict:
    """Per-market share of TP variance explained by the global factor (communality)."""
    return {"names": list(p.names), "global_share": p.communality.tolist(),
            "beta": p.beta.tolist(), "alpha": p.alpha.tolist()}


# --------------------------------------------------------------------------- driver
# Global-factor market set: Switzerland is EXCLUDED — its government-bond market
# liquidity is negligible next to the other four, so its 10y term premium is a noisy
# member of a common cyclical factor. Order is the display order.
GF_ORDER = ["US", "DE", "JP", "UK"]


def _month_end(s):
    """Last observation per calendar month, indexed at the month-end timestamp."""
    import pandas as pd
    s = s.dropna().sort_index()
    if s.empty:
        return s
    m = s.groupby(s.index.to_period("M")).tail(1)
    m.index = m.index.to_period("M").to_timestamp("M")
    return m


def splice_backfill(prod, hist):
    """Extend a modern month-end series ``prod`` backwards with ``hist`` via a ONE-TIME
    level match at the earliest overlap month: ``offset = prod[t0] - hist[t0]`` where
    ``t0`` is the first month both are present; the returned series is ``prod`` where it
    exists and ``hist + offset`` before that. Both inputs month-end, same units (decimal).
    Returns ``prod`` unchanged if there is no overlap."""
    import pandas as pd
    prod, hist = _month_end(prod), _month_end(hist)
    if prod.empty or hist.empty:
        return prod
    common = prod.index.intersection(hist.index)
    if len(common) == 0:
        return prod
    t0 = common.min()
    offset = float(prod.loc[t0] - hist.loc[t0])
    pre = hist[hist.index < prod.index.min()] + offset
    return pd.concat([pre, prod]).sort_index()


def _load_hist_tp(fname, tenor_key):
    """Read a monthly historical-overlay JSON (viz/history) as a month-end DECIMAL series
    for one tenor; returns None if absent/unreadable."""
    import json, pandas as pd
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "viz" / "history" / fname
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text())
        vals = d["tenors"][tenor_key]                       # percent
        s = pd.Series([None if v is None else float(v) / 100.0 for v in vals],
                      index=pd.to_datetime(d["dates"]), dtype=float).dropna()
        return _month_end(s)
    except Exception:                                       # noqa: BLE001
        return None


def load_tp_panel(tenor: float = 10.0, canon_path=None, extend: bool = True,
                  min_markets: int = 3, min_start: str = "1974-01-01", freq: str = "D"):
    """Wide (date × market) 10y nominal term-premium panel (decimal), markets
    US,DE,JP,UK (Switzerland excluded — negligible liquidity).

    ``freq='D'`` (default) builds a **daily** panel at the highest frequency available —
    each market's daily curve where it exists, and, before that, the monthly historical
    backfill **interpolated to daily** (linear in time). ``freq='M'`` gives the month-end
    panel. With ``extend`` (default), two markets are backfilled before their modern daily
    curve begins, via a one-time level splice (:func:`splice_backfill`):
      * DE 10y ← the 1972+ monthly historical overlay (germany_hist_tp.json).
      * JP 10y ← the 1974+ **9y** monthly overlay (japan_hist_tp.json); MoF's 10y
        starts only 1986-07, so the 9y premium (back to 1974) is spliced onto the 10y.
    The panel is clipped to the first date with at least ``min_markets`` markets present
    (and to ``min_start`` — pre-1974 only US+DE exist, which is too thin for a common
    factor)."""
    import pandas as pd
    from pathlib import Path
    canon_path = canon_path or (Path(__file__).resolve().parent.parent / "viz" / "data_daily.parquet")
    df = pd.read_parquet(canon_path)
    df["date"] = pd.to_datetime(df["date"])
    daily = {}
    for cc in GF_ORDER:
        s = df[(df.country == cc) & (df.instrument == "nominal") & (np.isclose(df.tenor, tenor))]
        s = s.set_index("date")["term_premium"].sort_index()
        daily[cc] = s[~s.index.duplicated()].dropna()
    back = {}                                                    # cc -> monthly continuous spliced series
    if extend:
        de_hist = _load_hist_tp("germany_hist_tp.json", "%g" % tenor)
        if de_hist is not None:
            back["DE"] = splice_backfill(_month_end(daily["DE"]), de_hist)
        jp_hist = _load_hist_tp("japan_hist_tp.json", "9")
        if jp_hist is not None:
            back["JP"] = splice_backfill(_month_end(daily["JP"]), jp_hist)

    if freq in ("M", "monthly"):
        cols = {cc: (back[cc] if cc in back else _month_end(daily[cc])) for cc in GF_ORDER}
        panel = pd.DataFrame(cols).sort_index()
    else:                                                        # daily
        allidx = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in daily.values()])))
        allidx = allidx[allidx >= pd.Timestamp(min_start)]
        cols = {}
        for cc in GF_ORDER:
            s = daily[cc].reindex(allidx)
            if cc in back and len(daily[cc]):
                pstart = daily[cc].index.min()
                bf = back[cc]                                    # monthly (month-end) continuous
                bf_daily = (bf.reindex(allidx.union(bf.index)).sort_index()
                            .interpolate(method="time").reindex(allidx))
                pre = allidx < pstart
                s = s.where(~pre, bf_daily.where(pre))
            cols[cc] = s
        panel = pd.DataFrame(cols, index=allidx).sort_index()
    ok = panel.notna().sum(axis=1) >= min_markets
    if ok.any():
        start = max(panel.index[ok].min(), pd.Timestamp(min_start))
        panel = panel[panel.index >= start]
    return panel


def _main() -> None:
    import argparse, logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Global term-premium DFM (research run)")
    ap.add_argument("--tenor", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="write DFMParams JSON here")
    args = ap.parse_args()
    import pandas as pd
    panel = load_tp_panel(args.tenor)
    Y = panel.to_numpy(float)
    names = list(panel.columns)
    overlap = np.all(np.isfinite(Y), axis=1)           # balanced-window mask for scaling
    print(f"panel {panel.index.min().date()}..{panel.index.max().date()} | {Y.shape[0]} days | "
          f"markets {names} | balanced from {panel.index[overlap].min().date()} "
          f"({int(overlap.sum())} days)")
    p, diag = estimate_dfm(Y, names, overlap_mask=overlap)
    print(f"\nphi={p.phi:.5f} (half-life {diag['halflife_days']:.0f}d)  loglik={diag['loglik']:.1f}  "
          f"iters={diag['n_iter']}  PCA corr={diag['pca_corr']:.3f}")
    print("\nmarket   beta    global_share")
    for i, nm in enumerate(names):
        print(f"  {nm:<5} {p.beta[i]:+.3f}   {p.communality[i]*100:5.1f}%")
    dec = filter_forward(p, Y, smoothed=True)
    G = pd.Series(dec["G"], index=panel.index)
    print(f"\nglobal factor G: latest {G.iloc[-1]:+.2f} (std units); "
          f"mean communality {np.mean(p.communality)*100:.0f}%")
    # residual = local (relative-value) read, at the latest date all five report
    resid = pd.DataFrame(dec["residual"], index=panel.index, columns=names)
    full = resid.dropna()
    res = full.iloc[-1] * 100
    print(f"local residual (bp, + = cheap vs global cycle) at {full.index[-1].date()}:")
    print("  " + "  ".join(f"{nm} {res[nm]:+.0f}" for nm in names))
    if args.out:
        p.to_json(args.out); print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _main()
