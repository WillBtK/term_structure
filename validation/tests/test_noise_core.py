"""The shared cross-sectional fit — the corrections it exists to make.

Each test builds a cross-section whose TRUE noise is known, so anything the estimator
reports beyond that is its own error. These are the measurements that justified
replacing the single-fixed-decay Nelson-Siegel fit the gauges started with.
"""
import numpy as np
import pytest

from us.reconstruct import svensson_yield
from validation.noise_core import residual_se, svensson_design

# Four curve shapes a government market plausibly takes (GSW units: betas percent).
SHAPES = [
    (4.2, -1.6, -1.0, 0.8, 1.5, 8.0),      # normal, mild hump
    (2.0, -3.0, 4.0, -2.0, 0.8, 4.0),      # steep, big short-end curvature
    (6.0, 1.0, -2.0, 3.0, 2.5, 14.0),      # high level, inverted front
    (3.0, -4.5, 6.0, -4.0, 0.5, 20.0),     # extreme curvature at both ends
]
TAU = np.linspace(1.0, 30.0, 24)


def _zero(tau, shape):
    return svensson_yield(tau, *shape) / 100.0


def _fixed_ns_rms(tau, y, lam=0.5):
    """The ORIGINAL specification: 3-factor Nelson-Siegel at one pinned decay, plain
    RMS. Kept here only as the baseline the corrections are measured against."""
    lt = np.maximum(lam * tau, 1e-9)
    s = (1.0 - np.exp(-lt)) / lt
    X = np.column_stack([np.ones_like(tau), s, s - np.exp(-lt)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sqrt(np.mean((y - X @ beta) ** 2)))


@pytest.mark.parametrize("shape", SHAPES)
def test_a_curve_with_no_noise_reads_as_no_noise(shape):
    """The headline correction. On yields lying exactly on a Svensson curve the true
    noise is zero, yet the single-fixed-decay fit reports basis points of it — and the
    amount depends on the shape of the curve, so it is not even a constant one could
    subtract out."""
    y = _zero(TAU, shape)
    assert residual_se(TAU, y) * 1e4 < 1.0
    assert _fixed_ns_rms(TAU, y) * 1e4 > 1.5       # the specification being replaced


def test_the_fixed_decay_error_varies_with_the_shape_of_the_curve():
    errs = [_fixed_ns_rms(TAU, _zero(TAU, s)) * 1e4 for s in SHAPES]
    assert max(errs) > 3 * min(errs)               # not a removable constant offset


def test_the_coupon_effect_is_absorbed():
    """Two bonds maturing the same day with different coupons trade at different YTMs,
    so a curve through a mixed-coupon cross-section is mis-specified. Here the coupon
    effect is injected directly: a known per-unit-coupon tilt on an otherwise clean
    curve. Regressing the coupon out must remove it."""
    rng = np.random.default_rng(0)
    y = _zero(TAU, SHAPES[0])
    coupon = rng.uniform(0.5, 6.0, TAU.size)
    y_mixed = y + 3e-4 * coupon                    # 3 bp of tilt per point of coupon
    assert residual_se(TAU, y_mixed) * 1e4 > 3.0                     # ignored
    assert residual_se(TAU, y_mixed, coupon=coupon) * 1e4 < 1.0      # absorbed


def test_a_constant_coupon_column_is_not_charged_a_degree_of_freedom():
    """A collinear coupon column carries no information, so it must not inflate the
    estimate by consuming a degree of freedom."""
    y = _zero(TAU, SHAPES[0])
    flat = np.full(TAU.size, 2.5)
    assert residual_se(TAU, y, coupon=flat) == pytest.approx(residual_se(TAU, y))


def test_the_estimate_lands_on_the_injected_scatter():
    """With a known amount of scatter injected on top of a real curve, the estimate
    should report roughly that amount. Profiling the decays costs a little — picking
    the best of 42 grids is itself a fit — so it sits a few percent low, not the ~30%
    the uncorrected specification is off by."""
    rng = np.random.default_rng(7)
    sd_bp = 8.0
    got = [residual_se(TAU, _zero(TAU, SHAPES[k % 4])
                       + rng.normal(0.0, sd_bp / 1e4, TAU.size)) * 1e4
           for k in range(120)]
    assert np.mean(got) == pytest.approx(sd_bp, rel=0.10)


def test_the_fixed_decay_error_adds_to_whatever_scatter_is_there():
    """The functional-form misfit does not cancel against the small-sample shrinkage —
    it dominates it. So the uncorrected gauge reads HIGH on quiet days, and by an
    amount that depends on the shape of the curve rather than on the market."""
    rng = np.random.default_rng(7)
    sd_bp = 8.0
    ours, plain = [], []
    for k in range(120):
        y = _zero(TAU, SHAPES[k % 4]) + rng.normal(0.0, sd_bp / 1e4, TAU.size)
        ours.append(residual_se(TAU, y) * 1e4)
        plain.append(_fixed_ns_rms(TAU, y) * 1e4)
    assert np.mean(plain) > np.mean(ours)
    assert np.mean(plain) > sd_bp                  # reports more noise than was injected


def test_the_degrees_of_freedom_correction_removes_the_shrinkage():
    """Isolate the third correction: same fit, RMS over n versus over the residual
    degrees of freedom. Dividing by n understates the scatter, and worse on thin
    cross-sections — which are usually the stressed ones."""
    rng = np.random.default_rng(11)
    sd = 8.0 / 1e4
    for n in (14, 24):
        tau = np.linspace(1.0, 30.0, n)
        corrected, uncorrected = [], []
        for k in range(150):
            y = _zero(tau, SHAPES[k % 4]) + rng.normal(0.0, sd, n)
            X = svensson_design(tau, 0.7, 0.15)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            corrected.append(np.sqrt(np.sum(r ** 2) / (n - 4)))
            uncorrected.append(np.sqrt(np.mean(r ** 2)))
        ratio = np.mean(uncorrected) / np.mean(corrected)
        assert ratio == pytest.approx(np.sqrt((n - 4) / n), rel=0.02)
        assert np.mean(corrected) == pytest.approx(sd, rel=0.06)


def test_weights_scale_the_fit_and_the_residual_together():
    """A bond weighted to zero must not affect the answer at all — that is what makes
    the TIPS taper safe against basket composition changes."""
    y = _zero(TAU, SHAPES[1]) 
    tau2 = np.r_[TAU, 1.2]
    y2 = np.r_[y, y[0] + 0.01]                     # a 100 bp outlier...
    w2 = np.r_[np.ones(TAU.size), 0.0]             # ...given zero weight
    assert residual_se(tau2, y2, weights=w2) == pytest.approx(residual_se(TAU, y))


def test_too_little_to_fit_returns_nan_rather_than_a_number():
    assert np.isnan(residual_se(TAU[:4], _zero(TAU[:4], SHAPES[0])))
    assert np.isnan(residual_se(np.array([2.0, 5.0]), np.array([0.01, 0.02])))


def test_design_matrix_shape_and_limits():
    X = svensson_design(TAU, 0.7, 0.15)
    assert X.shape == (TAU.size, 4)
    assert np.allclose(X[:, 0], 1.0)               # level loading
    # slope loading -> 1 at the short end, -> 0 at the long end
    assert svensson_design(np.array([1e-6]), 0.7, 0.15)[0, 1] == pytest.approx(1.0, abs=1e-4)
    assert svensson_design(np.array([1e4]), 0.7, 0.15)[0, 1] == pytest.approx(0.0, abs=1e-3)
