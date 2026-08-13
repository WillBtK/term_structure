"""Tests for the production TP assembly (estimation-free)."""
import numpy as np
import pandas as pd

from model import bias_correction as bcm
from model import shadow_rate as sr
from us.production_tp import _assemble_tp

GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


def _synth():
    K = np.diag([0.06, 0.30, 0.60])
    theta = np.array([0.04, -0.015, 0.0])
    Sigma = np.diag([0.008, 0.010, 0.012])
    dates = pd.date_range("1990-01-31", periods=60, freq="M")
    rng = np.random.default_rng(0)
    states = theta + 0.01 * rng.standard_normal((60, 3))
    pre = sr.precompute(0.5, Sigma, -0.0035, GRID, n_per_year=24)
    fitted = np.array([sr.elb_yields(states[t], pre) for t in range(60)])
    return states, fitted, K, theta, Sigma, dates


def test_assemble_schema_and_units():
    states, fitted, K, theta, Sigma, dates = _synth()
    frame = _assemble_tp(states, fitted, K, theta, Sigma, -0.0035, GRID, dates)
    assert set(frame.columns) == {"date", "maturity", "term_premium",
                                  "expected_rate", "fitted_yield"}
    assert len(frame) == 60 * len(GRID)
    assert sorted(frame["maturity"].unique()) == list(GRID)
    # decimal units: yields order ~few percent, not tens
    assert frame["fitted_yield"].abs().max() < 0.5
    # TP = fitted - expected exactly
    assert np.allclose(frame["term_premium"],
                       frame["fitted_yield"] - frame["expected_rate"])


def test_bias_correction_raises_persistence_in_assembly_inputs():
    # The headline uses BRW-corrected K^P; confirm the correction is applied and
    # raises persistence on a near-unit-root synthetic factor series.
    rng = np.random.default_rng(1)
    n = 400
    Phi = np.diag([0.99, 0.97, 0.95])
    x = np.zeros((n, 3))
    for t in range(1, n):
        x[t] = Phi @ x[t - 1] + 0.01 * rng.standard_normal(3)
    bc = bcm.bias_correct(x, dt=1.0 / 12, n_boot=200, seed=3)
    assert bc.rho_bc >= bc.rho_hat
    assert bc.rho_bc < 1.0


def test_assemble_headline_vs_uncorrected_differ():
    states, fitted, K, theta, Sigma, dates = _synth()
    bc = bcm.bias_correct(states, dt=1.0 / 12, n_boot=200, seed=0)
    head = _assemble_tp(states, fitted, bc.K_bc, theta, Sigma, -0.0035, GRID, dates)
    unc = _assemble_tp(states, fitted, K, theta, Sigma, -0.0035, GRID, dates)
    # same fitted yields, different expected-rate/TP (different K^P)
    assert np.allclose(head["fitted_yield"], unc["fitted_yield"])
    assert not np.allclose(head["term_premium"], unc["term_premium"])
