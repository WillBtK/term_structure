"""Tests for the global-factor relative-value diagnostic (plumbing + invariants)."""
import numpy as np
import pandas as pd

from validation import global_factor_rv as rv


def _panel(T=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=T, freq="M").to_timestamp("M")
    cols = ["US", "UK", "DE", "JP", "CH"]
    return idx, cols, rng


def test_weights_dollar_neutral_unit_gross():
    row = pd.Series([3.0, 1.0, 2.0, -1.0, 0.5], index=list("ABCDE"))
    w = rv._xs_weights(row)
    assert abs(w.sum()) < 1e-12                 # dollar-neutral
    assert abs(np.abs(w).sum() - 1.0) < 1e-9    # unit gross
    assert w.idxmax() == "A" and w.idxmin() == "D"   # long highest, short lowest


def test_excess_returns_are_cross_sectionally_demeaned():
    idx, cols, rng = _panel()
    Y10 = pd.DataFrame(0.03 + 0.01 * rng.standard_normal((len(idx), 5)), index=idx, columns=cols)
    SH = pd.DataFrame(0.02 + 0.005 * rng.standard_normal((len(idx), 5)), index=idx, columns=cols)
    rx = rv.excess_returns(Y10, SH)
    # each row sums to ~0 across markets (demeaned) where defined
    assert np.allclose(rx.dropna().sum(axis=1), 0.0, atol=1e-12)


def test_common_level_shock_largely_cancels():
    """A common yield move across all markets is *largely* absorbed by the
    cross-sectional demean — a first-order currency/level hedge. It is not exact:
    residual leakage comes only from duration dispersion (D=10/(1+y) differs by
    market), so the demeaned change is ~1bp vs the ~4.5% raw price impact."""
    idx, cols, rng = _panel(seed=2)
    Y10 = pd.DataFrame(0.03 + 0.01 * rng.standard_normal((len(idx), 5)), index=idx, columns=cols)
    SH = pd.DataFrame(0.02 + np.zeros((len(idx), 5)), index=idx, columns=cols)
    rx0 = rv.excess_returns(Y10, SH)
    Y10b = Y10.copy(); Y10b.iloc[50] += 0.005          # common +50bp shock at t=50
    rx1 = rv.excess_returns(Y10b, SH)
    raw_impact = 9.5 * 0.005                            # ≈ duration × shock ≈ 4.75%
    leak = float(np.abs(rx1.iloc[49].dropna() - rx0.iloc[49].dropna()).max())
    assert leak < 0.05 * raw_impact                    # >95% of the common shock hedged out
    # with equal yields (equal durations) it cancels exactly:
    Yflat = pd.DataFrame(0.03, index=idx, columns=cols)
    r0 = rv.excess_returns(Yflat, SH)
    Yf = Yflat.copy(); Yf.iloc[50] += 0.005
    r1 = rv.excess_returns(Yf, SH)
    assert np.allclose(r0.iloc[49].dropna(), r1.iloc[49].dropna(), atol=1e-12)


def test_injected_signal_predicts_returns():
    """If the signal mechanically leads relative returns, the backtest recovers a
    positive IC/IR — confirms the alignment (signal_t predicts rx over [t,t+1])."""
    idx, cols, rng = _panel(seed=3)
    T = len(idx)
    sig = pd.DataFrame(rng.standard_normal((T, 5)), index=idx, columns=cols)
    sig = sig.sub(sig.mean(axis=1), axis=0)            # cross-sectionally demeaned
    noise = 0.2 * rng.standard_normal((T, 5))
    rx = sig * 0.01 + pd.DataFrame(noise * 0.01, index=idx, columns=cols)
    rx = rx.sub(rx.mean(axis=1), axis=0)
    bt = rv.backtest(sig, rx)
    assert bt["pooled_IC"] > 0.3 and bt["IR"] > 0.5
    # a reversed signal flips the IR sign
    bt_rev = rv.backtest(-sig, rx)
    assert bt_rev["IR"] < 0
