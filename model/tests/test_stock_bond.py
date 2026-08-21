"""Offline tests for the stock-bond correlation channel (no network)."""
import numpy as np
import pandas as pd
import pytest

from model.stock_bond import SBCParams, bond_return_proxy, fit_sbc, realised_sbc


def test_bond_return_is_negative_in_yield_changes():
    """The sign that makes bonds bonds: yields up, prices down."""
    y = pd.Series([4.0, 4.0, 5.0, 4.0])
    r = bond_return_proxy(y, duration=8.0)
    assert r.iloc[1] == pytest.approx(4.0 / 12)          # unchanged yield -> carry only
    assert r.iloc[2] < 0 and r.iloc[3] > 0               # +100bp hurts, -100bp helps
    assert r.iloc[2] == pytest.approx(4.0 / 12 - 8.0 * 1.0)


def test_recovers_a_known_ratio_relationship():
    rng = np.random.default_rng(0)
    # Keep the ratio inside the range where the true line stays a legal correlation.
    # Generating through predict() would CLIP at +/-1 and destroy the very linearity
    # the fit is being asked to recover, so build from the raw line instead.
    sig_pi = rng.uniform(0.3, 3.0, 300)
    sig_g = rng.uniform(2.0, 8.0, 300)
    sbc = -0.55 + 0.90 * (sig_pi / sig_g) + rng.normal(0, 0.02, 300)
    assert np.abs(sbc).max() < 1.0, "test data must not need clipping"
    got = fit_sbc(sbc, sig_pi, sig_g)
    assert got.intercept == pytest.approx(-0.55, abs=0.05)
    assert got.slope == pytest.approx(0.90, abs=0.05)
    assert got.r2 > 0.95


def test_inflation_vol_raises_and_growth_vol_lowers_the_correlation():
    """The AQR sign structure -- the whole economic content of the channel."""
    p = SBCParams(intercept=-0.55, slope=0.90)
    base = p.predict(1.0, 3.0)
    assert p.predict(2.0, 3.0) > base      # more inflation vol -> bonds hedge worse
    assert p.predict(1.0, 6.0) < base      # more growth vol   -> bonds hedge better


def test_prediction_cannot_leave_the_legal_correlation_range():
    p = SBCParams(intercept=-0.55, slope=0.90)
    assert p.predict(50.0, 0.5) == pytest.approx(1.0)     # absurd ratio, still a corr
    assert p.predict(0.01, 50.0) >= -1.0


def test_realised_sbc_recovers_a_constructed_correlation():
    n = 400
    rng = np.random.default_rng(3)
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    # yields wander; equity built to move WITH bond returns (i.e. positive SBC)
    y = pd.Series(5.0 + np.cumsum(rng.normal(0, 0.05, n)), index=idx)
    rb = bond_return_proxy(y).fillna(0.0)
    eq = pd.Series(np.exp(np.cumsum(0.3 * rb.values / 100 + rng.normal(0, 0.002, n))), index=idx)
    out = realised_sbc(eq, y, window=120)
    assert len(out) > 0 and out.iloc[-1] > 0.3


def test_fit_refuses_a_sample_too_small_to_mean_anything():
    with pytest.raises(ValueError, match="at least 8"):
        fit_sbc([0.1, 0.2], [1.0, 1.0], [2.0, 2.0])
