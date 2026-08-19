"""Forward zero rates derived from the reported zero-coupon curve.

Every yield in this project is a **continuously compounded zero-coupon** rate (that is
what the AFNS state-space prices and what GSW/BoE/MoF/SNB/BoC publish), so the forward
rate between two grid points is exactly linear in the two zero rates:

    f(t1, t2) = (y(t2)*t2 - y(t1)*t1) / (t2 - t1)          t2 > t1 >= 0

with the spot rate recovered at ``t1 = 0``. The familiar 5y5y forward is the case
``f(5, 10) = 2*y(10) - y(5)``.

Why this matters for the decomposition
--------------------------------------
The transform is **linear in the yields**, and the decomposition is additive
(``observed = expected + term_premium``), so applying it to each leg is consistent:

    f_obs(t1,t2) = f_exp(t1,t2) + f_tp(t1,t2)

i.e. the forward term premium IS the forward of the term premium — no re-estimation and
no re-decomposition is needed to read the model at forward points. (Exact for
continuously compounded rates; for an annually compounded convention the equivalent is
``((1+y2)**t2 / (1+y1)**t1)**(1/(t2-t1)) - 1``, which is NOT linear and would break
this additivity — hence the continuous convention is load-bearing, not cosmetic.)
"""
from __future__ import annotations

import numpy as np

__all__ = ["forward_rate", "forward_from_curve", "FORWARD_LABEL"]


def forward_rate(y1, t1: float, y2, t2: float):
    """Continuously compounded forward zero rate between ``t1`` and ``t2`` years.

    ``y1``/``y2`` are the zero rates (any array-like, DECIMAL) at ``t1``/``t2``. With
    ``t1 == 0`` this returns ``y2`` (the spot rate). Raises when ``t2 <= t1``."""
    t1, t2 = float(t1), float(t2)
    if not t2 > t1:
        raise ValueError(f"forward needs t2 > t1 (got t1={t1}, t2={t2})")
    y2 = np.asarray(y2, dtype=float)
    if t1 <= 0.0:
        return y2
    y1 = np.asarray(y1, dtype=float)
    return (y2 * t2 - y1 * t1) / (t2 - t1)


def forward_from_curve(curve, start: float, tenor: float):
    """Forward rate of length ``tenor`` starting ``start`` years ahead, taken from a wide
    (date x maturity) frame of zero rates.

    Both legs must be present as columns (``start`` and ``start + tenor``) — we do not
    interpolate, so the result is derived strictly from reported grid points. Returns a
    Series indexed like ``curve``; raises KeyError when a leg is missing."""
    t1, t2 = float(start), float(start) + float(tenor)
    cols = {float(c): c for c in curve.columns}
    if t2 not in cols or (t1 > 0 and t1 not in cols):
        missing = [t for t in ((t1,) if t1 > 0 else ()) + (t2,) if t not in cols]
        raise KeyError(f"curve lacks maturities {missing} (has {sorted(cols)})")
    y2 = curve[cols[t2]]
    if t1 <= 0.0:
        return y2.rename(FORWARD_LABEL(start, tenor))
    out = forward_rate(curve[cols[t1]].to_numpy(), t1, y2.to_numpy(), t2)
    import pandas as pd
    return pd.Series(out, index=curve.index, name=FORWARD_LABEL(start, tenor))


def FORWARD_LABEL(start: float, tenor: float) -> str:
    """Market shorthand: ``5y5y`` for a 5-year rate starting in 5 years; ``10y`` spot."""
    g = lambda v: ("%g" % float(v))
    return f"{g(tenor)}y" if float(start) <= 0 else f"{g(start)}y{g(tenor)}y"
