"""Tests for the global term-premium DFM (synthetic recovery + invariants)."""
import numpy as np
import pytest

from model import global_factor as gf


def _simulate(T=1500, phi=0.98, lam=(0.9, 0.7, 0.8, 0.5, 0.6), r=None, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.asarray(lam, float); N = lam.size
    r = np.asarray(r if r is not None else 0.4 * np.ones(N), float)
    q = 1.0 - phi * phi
    G = np.empty(T); G[0] = rng.normal(0, 1)
    for t in range(1, T):
        G[t] = phi * G[t - 1] + rng.normal(0, np.sqrt(q))
    Z = G[:, None] * lam[None, :] + rng.normal(0, np.sqrt(r), size=(T, N))
    return Z, G, lam, r


def test_recovers_known_factor_and_loadings():
    Z, G, lam, r = _simulate(seed=1)
    names = list("ABCDE")
    p, diag = gf.estimate_dfm(Z, names, overlap_mask=np.ones(len(Z), bool))
    dec = gf.filter_forward(p, Z, smoothed=True)
    # factor recovered (up to sign, which is anchored to market 0 loading)
    corr = abs(np.corrcoef(dec["G"], G)[0, 1])
    assert corr > 0.98, corr
    # loadings recovered in the right order/relative size
    assert np.corrcoef(p.lam, lam)[0, 1] > 0.95
    # AR(1) persistence in the right neighbourhood
    assert abs(p.phi - 0.98) < 0.03, p.phi


def test_idiosyncratic_floor_prevents_heywood_collapse():
    """A near-noiseless, highly-correlated series must NOT collapse the factor onto
    itself (communality -> 1, ~zero residual) — the Canada/global-cycle pathology."""
    # market 0 is almost a scaled copy of the factor (tiny idiosyncratic noise)
    Z, G, lam, r = _simulate(r=[1e-4, 0.4, 0.4, 0.4, 0.4], seed=7)
    p, _ = gf.estimate_dfm(Z, list("ABCDE"), overlap_mask=np.ones(len(Z), bool))
    # the floor caps communality below 1 (no Heywood collapse) for every market
    assert np.all(p.communality < 0.99), p.communality.tolist()
    # and the near-degenerate market's uniqueness is held at (or above) the floor
    assert p.r[0] >= gf._R_FLOOR - 1e-9, (p.r[0], gf._R_FLOOR)


def test_variance_shares_in_unit_interval():
    Z, *_ = _simulate(seed=2)
    p, _ = gf.estimate_dfm(Z, list("ABCDE"), overlap_mask=np.ones(len(Z), bool))
    comm = p.communality
    assert np.all(comm >= 0) and np.all(comm <= 1)
    # a high-loading, low-noise market has a higher global share than a noisy one
    assert comm[0] > comm[3]


def test_residual_orthogonal_to_factor():
    Z, *_ = _simulate(seed=3)
    p, _ = gf.estimate_dfm(Z, list("ABCDE"), overlap_mask=np.ones(len(Z), bool))
    dec = gf.filter_forward(p, Z, smoothed=True)
    G = dec["G"]
    for i in range(p.n):
        c = abs(np.corrcoef(dec["residual"][:, i], G)[0, 1])
        assert c < 0.15, (i, c)          # residual ~ orthogonal to the common factor


def test_ragged_panel_runs_and_matches_on_overlap():
    Z, G, lam, r = _simulate(seed=4)
    Zr = Z.copy()
    Zr[:400, 2] = np.nan                 # market C only appears after t=400
    Zr[:200, 4] = np.nan                 # market E after t=200
    overlap = np.all(np.isfinite(Zr), axis=1)
    p, diag = gf.estimate_dfm(Zr, list("ABCDE"), overlap_mask=overlap)
    assert np.isfinite(diag["loglik"])
    dec = gf.filter_forward(p, Zr, smoothed=True)
    # factor on the fully-observed tail still tracks the truth
    tail = slice(500, None)
    assert abs(np.corrcoef(dec["G"][tail], G[tail])[0, 1]) > 0.95


def test_sign_anchor_market0_positive():
    for seed in (5, 6, 7):
        Z, *_ = _simulate(lam=(-0.9, 0.7, -0.5, 0.8, 0.6), seed=seed)
        p, _ = gf.estimate_dfm(Z, list("ABCDE"), overlap_mask=np.ones(len(Z), bool))
        assert p.lam[0] > 0, (seed, p.lam[0])


def test_dfm_matches_pca_on_balanced():
    Z, *_ = _simulate(seed=8)
    p, diag = gf.estimate_dfm(Z, list("ABCDE"), overlap_mask=np.ones(len(Z), bool))
    assert diag["pca_corr"] > 0.90, diag["pca_corr"]   # noisy synthetic; higher on real data


def test_json_roundtrip():
    Z, *_ = _simulate(seed=9)
    p, _ = gf.estimate_dfm(Z, list("ABCDE"), overlap_mask=np.ones(len(Z), bool))
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "gf.json"); p.to_json(fp)
        q = gf.DFMParams.from_json(fp)
    assert np.allclose(p.lam, q.lam) and np.isclose(p.phi, q.phi)
    assert q.names == p.names
