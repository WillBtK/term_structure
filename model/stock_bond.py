"""Stock-bond correlation from the relative volatility of inflation and growth.

Brixton, Brooks, Hecht, Ilmanen, Maloney and McQuinn (2023, *JPM*), "A Changing
Stock-Bond Correlation": equities and bonds have OPPOSITE-sign exposure to growth news
and SAME-sign exposure to inflation news. Writing returns as

    r_stock = b_sg * g + b_spi * pi        (b_sg > 0, b_spi < 0)
    r_bond  = b_bg * g + b_bpi * pi        (b_bg < 0, b_bpi < 0)

the stock-bond covariance is

    cov(s,b) = b_sg*b_bg * var(g)  +  b_spi*b_bpi * var(pi)
               + (b_sg*b_bpi + b_spi*b_bg) * cov(g,pi)

The first product is NEGATIVE (opposite signs) so growth volatility pushes the
correlation down — bonds hedge. The second is POSITIVE (both negative) so inflation
volatility pushes it up — bonds stop hedging. The cross-term's sign is ambiguous. Hence
the paper's headline: the correlation depends "not on the level of inflation, but on the
relative volatility of growth and inflation and the correlation between them".

**Why this module is in the chain.** The stock-bond correlation is OBSERVABLE. A
scenario model that maps macro forecasts to a term premium is otherwise untestable end
to end — you cannot check a premium you had to model in order to see. Routing through
the correlation splits the problem into two separately falsifiable links, and puts a
familiar object ("bonds stopped hedging") in front of the reader instead of an
unfamiliar one.

**The ratio form is what we use.** Empirically the single variable sigma_pi/sigma_g
outperforms the three-term structural regression out of the box (pooled R2 0.53 against
0.43 — see ``validation/FINDINGS_sbc.md``), because the three terms are collinear over a
40-year panel and the cross-term is weakly identified. One signed variable that fits
better is also the one an investment committee can hold in its head.

Pure: no I/O, no network.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def bond_return_proxy(yield_pct: pd.Series, duration: float = 8.0) -> pd.Series:
    """Monthly bond total return (%) from a 10y yield: carry minus duration times dy.

    A first-order approximation, which is all a CORRELATION needs — the omitted convexity
    term is second order and does not change the sign structure the model is about.
    """
    y = pd.Series(yield_pct).astype(float)
    return y.shift(1) / 12.0 - duration * y.diff()


def realised_sbc(equity_index: pd.Series, yield_pct: pd.Series,
                 window: int = 120, duration: float = 8.0) -> pd.Series:
    """Trailing correlation of monthly equity and bond returns."""
    re = np.log(pd.Series(equity_index).astype(float)).diff() * 100.0
    rb = bond_return_proxy(yield_pct, duration)
    df = pd.DataFrame({"re": re, "rb": rb}).dropna()
    return df["re"].rolling(window).corr(df["rb"]).dropna()


@dataclass(frozen=True)
class SBCParams:
    """``sbc = intercept + slope * (sigma_pi / sigma_g)``."""

    intercept: float
    slope: float
    r2: float = float("nan")
    n: int = 0

    def predict(self, sigma_pi, sigma_g):
        """Correlation implied by an inflation/growth volatility pair.

        Clipped to [-1, 1]: the fit is linear and unconstrained, so an extreme scenario
        can push it outside the range a correlation can legally take.
        """
        ratio = np.asarray(sigma_pi, dtype=float) / np.asarray(sigma_g, dtype=float)
        out = np.clip(self.intercept + self.slope * ratio, -1.0, 1.0)
        return float(out) if out.ndim == 0 else out


def fit_sbc(sbc, sigma_pi, sigma_g) -> SBCParams:
    """Least squares of realised correlation on the volatility ratio.

    Fit on ANNUAL observations: the inputs are overlapping rolling windows, so a monthly
    fit would overstate the sample size by roughly twelvefold.
    """
    sbc = np.asarray(sbc, dtype=float)
    ratio = np.asarray(sigma_pi, dtype=float) / np.asarray(sigma_g, dtype=float)
    ok = np.isfinite(sbc) & np.isfinite(ratio)
    sbc, ratio = sbc[ok], ratio[ok]
    if len(sbc) < 8:
        raise ValueError(f"need at least 8 observations to fit, got {len(sbc)}")
    X = np.column_stack([np.ones(len(ratio)), ratio])
    beta, *_ = np.linalg.lstsq(X, sbc, rcond=None)
    resid = sbc - X @ beta
    denom = ((sbc - sbc.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / denom if denom > 0 else float("nan")
    return SBCParams(beta[0], beta[1], r2, len(sbc))
