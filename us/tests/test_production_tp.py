"""Tests for the production TP assembly (estimation-free)."""
import numpy as np
import pandas as pd

from model import bias_correction as bcm
from model import ekf
from model import shadow_rate as sr
from us.production_tp import _assemble_tp, ProductionParams, filter_forward

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


def test_production_params_json_roundtrip(tmp_path):
    p = ekf.ShadowParams.from_diagonal(kappa=[0.06, 0.30, 0.60], theta=[0.04, -0.015, 0.0],
                                       sigma=[0.008, 0.010, 0.012], lambda_=0.5, h=0.0006)
    prod = ProductionParams(params=p, K_bc=np.diag([0.01, 0.30, 0.60]), elb=-0.0035, n_per_year=24)
    path = tmp_path / "prod.json"
    prod.to_json(path)
    q = ProductionParams.from_json(path)
    assert np.allclose(q.params.K, p.K) and np.allclose(q.params.theta, p.theta)
    assert np.allclose(q.params.sigma, p.sigma) and np.isclose(q.params.lambda_, p.lambda_)
    assert np.allclose(q.K_bc, prod.K_bc) and q.elb == prod.elb and q.n_per_year == 24
    # theta_expect defaults to None -> expectations θ falls back to the transition θ
    assert q.theta_expect is None
    assert np.allclose(q.theta_for_expectations(), p.theta)


def test_theta_expect_roundtrip_and_overrides():
    p = ekf.ShadowParams.from_diagonal(kappa=[0.06, 0.30, 0.60], theta=[0.9, 0.9, 0.0],
                                       sigma=[0.008, 0.010, 0.012], lambda_=0.5, h=0.0006)
    te = np.array([0.02, -0.01, 0.0])
    prod = ProductionParams(params=p, K_bc=np.diag([0.01, 0.30, 0.60]), elb=-0.0035,
                            n_per_year=24, theta_expect=te)
    assert np.allclose(prod.theta_for_expectations(), te)     # not the (silly) transition θ
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "p.json")
    prod.to_json(path)
    q = ProductionParams.from_json(path)
    assert np.allclose(q.theta_expect, te)


def test_filter_forward_is_the_fast_daily_path():
    # filter_forward uses cached params (no estimation) and reproduces the canonical
    # frame; it is the daily publish operation. The premium is now defined against
    # the observed panel, so the frame carries observed_yield + term_premium_fitted.
    states, fitted, K, theta, Sigma, dates = _synth()
    p = ekf.ShadowParams(K=K, theta=theta, sigma=np.diag(Sigma), lambda_=0.5, h=0.0006, elb=-0.0035)
    prod = ProductionParams(params=p, K_bc=K, elb=-0.0035, n_per_year=24)
    panel = pd.DataFrame(fitted, index=dates, columns=GRID)  # yields as a panel
    frame = filter_forward(prod, panel, dt=1.0 / 12)
    assert set(frame.columns) == {"date", "maturity", "term_premium", "expected_rate",
                                  "fitted_yield", "term_premium_fitted", "observed_yield"}
    assert len(frame) == len(dates) * len(GRID)
    # realised premium = observed - expected
    assert np.allclose(frame["term_premium"], frame["observed_yield"] - frame["expected_rate"])


def test_assemble_observed_based_premium_and_ycc_negative():
    # With an observed panel, term_premium is observed-based (realised). Simulate a
    # peg that pushes the observed 10y BELOW the model-fair fitted 10y: the realised
    # premium then drops below the model-fair premium at that maturity/date — the
    # Japan-YCC mechanism.
    states, fitted, K, theta, Sigma, dates = _synth()
    observed = fitted.copy()
    j10 = list(GRID).index(10.0)
    observed[:, j10] -= 0.0030            # peg 30 bp below fair on the 10y
    frame = _assemble_tp(states, fitted, K, theta, Sigma, -0.0035, GRID, dates,
                         observed=observed)
    assert {"observed_yield", "term_premium_fitted"} <= set(frame.columns)
    t10 = frame[frame.maturity == 10.0]
    # realised premium sits ~30 bp below the model-fair premium on the pegged 10y
    assert np.allclose(t10["term_premium_fitted"] - t10["term_premium"], 0.0030)
    # away from the peg (2y) the two coincide
    t2 = frame[frame.maturity == 2.0]
    assert np.allclose(t2["term_premium"], t2["term_premium_fitted"])


def test_assemble_headline_vs_uncorrected_differ():
    states, fitted, K, theta, Sigma, dates = _synth()
    bc = bcm.bias_correct(states, dt=1.0 / 12, n_boot=200, seed=0)
    head = _assemble_tp(states, fitted, bc.K_bc, theta, Sigma, -0.0035, GRID, dates)
    unc = _assemble_tp(states, fitted, K, theta, Sigma, -0.0035, GRID, dates)
    # same fitted yields, different expected-rate/TP (different K^P)
    assert np.allclose(head["fitted_yield"], unc["fitted_yield"])
    assert not np.allclose(head["term_premium"], unc["term_premium"])
