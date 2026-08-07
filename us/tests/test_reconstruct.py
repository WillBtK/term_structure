"""Tests for the standalone Svensson reconstruction (no network required).

Golden values were computed independently with plain-Python ``math`` for the
parameter set (b0,b1,b2,b3,t1,t2) = (3.0, -1.0, 2.0, -0.5, 1.5, 8.0):

    n=0.25 -> 2.220494569734765   (percent)
    n=10   -> 3.0051181545012904  (percent)
    n->0   -> b0 + b1 = 2.0       (percent)
"""
import numpy as np
import pandas as pd
import pytest

from us.reconstruct import (
    cross_check_against_published,
    reconstruct_grid,
    svensson_yield,
)

PARAMS = dict(beta0=3.0, beta1=-1.0, beta2=2.0, beta3=-0.5, tau1=1.5, tau2=8.0)


def test_golden_values():
    assert svensson_yield(0.25, **PARAMS) == pytest.approx(2.220494569734765, abs=1e-12)
    assert svensson_yield(10.0, **PARAMS) == pytest.approx(3.0051181545012904, abs=1e-12)


def test_short_maturity_limit():
    # As n -> 0, all loadings -> the level+slope intercept: y -> b0 + b1.
    assert svensson_yield(1e-9, **PARAMS) == pytest.approx(3.0 - 1.0, abs=1e-6)


def test_loading_zero_maturity_is_finite():
    # Exactly n=0 must not blow up (handled limit), returns b0+b1.
    assert np.isfinite(svensson_yield(0.0, **PARAMS))
    assert svensson_yield(0.0, **PARAMS) == pytest.approx(2.0, abs=1e-12)


def test_beta3_term_uses_tau2_not_tau1():
    # Regression guard against the upstream ns_yield bug (β3 term divided by
    # lambda_ instead of lambda_svensson). With tau1 != tau2 and only beta3
    # active, the yield must depend on tau2. The buggy form would instead use
    # tau1 in the (1-e^-x)/x factor.
    base = dict(beta0=0.0, beta1=0.0, beta2=0.0, beta3=1.0, tau1=1.0, tau2=5.0)
    correct = svensson_yield(4.0, **base)
    x2 = 4.0 / 5.0
    expected = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    assert correct == pytest.approx(expected, abs=1e-12)
    # And it must NOT equal the buggy variant that uses tau1 in the denominator.
    x1 = 4.0 / 1.0
    buggy = (1 - np.exp(-x2)) / x1 - np.exp(-x2)
    assert abs(correct - buggy) > 1e-6


def test_reconstruct_grid_shape_and_decimal():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    params = pd.DataFrame(
        {
            "BETA0": [3.0, 3.1],
            "BETA1": [-1.0, -1.1],
            "BETA2": [2.0, 1.9],
            "BETA3": [-0.5, -0.4],
            "TAU1": [1.5, 1.4],
            "TAU2": [8.0, 8.5],
        },
        index=idx,
    )
    grid = np.array([0.25, 1.0, 10.0])
    panel = reconstruct_grid(params, grid)
    assert panel.shape == (2, 3)
    assert list(panel.columns) == list(grid)
    # First row, 10y should equal the golden value / 100 (decimal).
    assert panel.iloc[0][10.0] == pytest.approx(3.0051181545012904 / 100.0, abs=1e-14)
    # Yields are decimals, so a ~3% yield is ~0.03, not ~3.
    assert 0.0 < panel.iloc[0][10.0] < 0.10


def test_invalid_rows_dropped():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    params = pd.DataFrame(
        {
            "BETA0": [3.0, np.nan, 3.0],   # row 2 has a NaN param
            "BETA1": [-1.0, -1.0, -1.0],
            "BETA2": [2.0, 2.0, 2.0],
            "BETA3": [-0.5, -0.5, -0.5],
            "TAU1": [1.5, 1.5, -1.0],      # row 3 has a non-positive tau
            "TAU2": [8.0, 8.0, 8.0],
        },
        index=idx,
    )
    panel = reconstruct_grid(params, np.array([1.0]))
    assert len(panel) == 1
    assert panel.index[0] == idx[0]


def test_cross_check_against_self_consistent_published():
    # Build synthetic "published" SVENY columns from the same formula; the
    # cross-check must then report ~0 bp difference and within_tol=True.
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    params = pd.DataFrame(
        {
            "BETA0": [3.0, 3.1],
            "BETA1": [-1.0, -1.1],
            "BETA2": [2.0, 1.9],
            "BETA3": [-0.5, -0.4],
            "TAU1": [1.5, 1.4],
            "TAU2": [8.0, 8.5],
        },
        index=idx,
    )
    for m in (1, 5, 10):
        params[f"SVENY{m:02d}"] = reconstruct_grid(params, np.array([float(m)])).iloc[:, 0] * 100.0
    check = cross_check_against_published(params)
    assert set(check["maturity"]) == {1.0, 5.0, 10.0}
    assert bool(check["within_tol"].all())
    assert check["max_abs_diff_bp"].max() < 1e-6
