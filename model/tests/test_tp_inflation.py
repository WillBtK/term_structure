"""The inflation shape, and the constraint that keeps thin markets honest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.tp_inflation import (HIGH_SLOPE_FLOOR, TARGET_KINK, UPPER_KINK,
                                InflationShape, ewma_level, fit_constrained,
                                fit_shape)


def _series(pi, tp):
    idx = pd.date_range("1980-01-01", periods=len(pi), freq="MS")
    return pd.Series(tp, index=idx), pd.Series(pi, index=idx)


def test_premium_is_flat_at_and_below_target():
    s = InflationShape(0.5, 1.0, -0.6, 5.0, 0.9, 100)
    assert s.premium(0.0) == s.premium(1.0) == s.premium(2.0) == 0.5


def test_the_two_branches_have_the_advertised_slopes():
    s = InflationShape(0.5, 1.0, -0.6, 5.0, 0.9, 100)
    assert abs((s.premium(4.0) - s.premium(3.0)) - 1.0) < 1e-12
    assert abs((s.premium(6.0) - s.premium(5.0)) - s.slope_high) < 1e-12
    assert abs(s.slope_high - 0.4) < 1e-12


def test_fit_recovers_a_known_two_branch_shape():
    pi = np.linspace(0.5, 9.0, 400)
    true = InflationShape(0.4, 1.2, -0.9, 5.0, np.nan, 0)
    tp, pis = _series(pi, true.premium(pi))
    got = fit_shape(tp, pis)
    assert abs(got.kink2 - 5.0) < 0.3
    assert abs(got.slope_mid - 1.2) < 0.05
    assert abs(got.slope_high - 0.3) < 0.05


def test_shared_upper_branch_rescues_a_market_with_no_high_inflation_data():
    """The Germany case: a market whose sample barely clears the upper kink must not be
    allowed to fit its own upper slope, or the premium turns DOWN as inflation rises."""
    rng = np.random.default_rng(0)
    truth = InflationShape(0.5, 1.1, -0.8, UPPER_KINK, np.nan, 0)
    wide = np.linspace(0.5, 9.0, 500)                       # identifies the upper branch
    thin = np.linspace(0.5, 5.4, 500)                       # barely clears it
    tpm, pim = {}, {}
    for name, pi in (("US", wide), ("UK", wide), ("DE", thin)):
        noise = rng.normal(0, 0.05 if name != "DE" else 0.6, len(pi))
        tpm[name], pim[name] = _series(pi, truth.premium(pi) + noise)
    out = fit_constrained(tpm, pim, identified=("US", "UK"))
    shared = out.pop("_shared_high_fitted")
    assert abs(shared - truth.slope_high) < 0.1     # the UNfloored estimate
    for c, s in out.items():
        assert s.slope_high > 0, f"{c} upper branch went negative"
        assert s.premium(6.0) > s.premium(5.0), f"{c} premium falls as inflation rises"


def test_ewma_memory_is_short_enough_for_a_recent_spike_to_register():
    idx = pd.date_range("2000-01-01", periods=360, freq="MS")
    pi = pd.Series(2.0, index=idx)
    pi.iloc[-40:-16] = 6.0                                   # a two-year spike
    e = ewma_level(pi)
    assert e.max() > 4.0, "a two-year 6% episode must not be averaged away"


def test_settled_inflation_is_its_own_ewma():
    """The scenario path flatlines, so the memory parameter cannot matter there."""
    idx = pd.date_range("2000-01-01", periods=400, freq="MS")
    flat = pd.Series(4.0, index=idx)
    assert abs(ewma_level(flat).iloc[-1] - 4.0) < 1e-6


def test_the_premium_never_falls_as_inflation_rises():
    """The constraint that matters most. A premium that declines in a rising-inflation
    world is not something any investor would accept, however well it fits."""
    rng = np.random.default_rng(3)
    # A market whose data genuinely suggests a FLAT or falling upper branch.
    wide = np.linspace(0.5, 9.0, 500)
    truth = InflationShape(0.5, 1.2, -1.35, UPPER_KINK, np.nan, 0)   # upper slope -0.15
    tpm, pim = {}, {}
    for name in ("US", "UK", "DE"):
        idx = pd.date_range("1980-01-01", periods=len(wide), freq="MS")
        tpm[name] = pd.Series(truth.premium(wide) + rng.normal(0, .05, len(wide)), index=idx)
        pim[name] = pd.Series(wide, index=idx)
    out = fit_constrained(tpm, pim, identified=("US", "UK"))
    out.pop("_shared_high_fitted")
    grid = np.arange(0.0, 12.01, 0.25)
    for c, s in out.items():
        assert s.slope_high >= HIGH_SLOPE_FLOOR * s.slope_mid - 1e-9, c
        p = s.premium(grid)
        assert np.all(np.diff(p) >= -1e-9), f"{c} premium falls as inflation rises"
