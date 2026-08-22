# The common factor, and why it must not be applied naively

Numbers behind §3 of the methodology artifact ("Anatomy of the Term Premium").
Monthly 10y term premia from `viz/data_daily.parquet`, common G4 sample 1997-08..2026-08.

## The backbone

First principal component's share of variance: **0.81 on levels, 0.71 on 12-month
changes**. Rolling 120-month window: 0.66 at the start (2007), trough **0.58 around
2011**, **0.84 today** — the common share cycles, it does not trend down.

## The thesis: the loose episodes are priced distortions, not decay

The episodes where the common share loosens coincide with divergence in the relative
drivers — price-insensitive ownership first (Japan's relative-to-US ownership runs from
−25pp of GDP pre-QQE to **+66pp** under YCC), and post-2008, projected fiscal paths.
These are omitted variables, not exceptions, and the spread model includes them.

The demonstration: R² on levels of a naive full-sample beta to the US premium vs the
current spread specification (beta + relative drivers), evaluated full-sample and on
2010+ (against the subwindow mean — a harsh metric):

| market | β | full naive | full ours | 2010+ naive | 2010+ ours |
|---|---|---|---|---|---|
| UK | 0.89 | 0.37 | **0.78** | 0.31 | **0.73** |
| DE | 1.30 | 0.73 | **0.80** | −0.64 | **−0.25** |
| JP | 0.74 | 0.53 | **0.78** | −1.35 | **+0.29** |

**Japan is the proof.** Post-2010 a naive US beta is not weak but catastrophically wrong
(R² −1.35, worse than knowing nothing) — that is yield-curve control, the largest
relative-ownership divergence in the sample. The relative drivers restore a positive
fit. If the low-PC1 episodes were genuine decoupling noise, adding relative QE and
fiscal could not recover it.

Honest weak spot: Germany. Its 2010+ premium sat in a narrow band that a full-sample
fit measures poorly under either specification; the relative drivers halve the miss
(−0.64 → −0.25) but do not fix it.

Note the changes-frequency comparison is NOT quoted: at 12-month changes both
specifications are noisy and the full model does not consistently beat the naive one —
the relative drivers earn their keep on levels, which is the fit target.

Reproduce: the extraction lives in the session scratchpad
(`extract3.py`); the computation is ~30 lines — PCA eigenvalue shares on standardized
panels, and lstsq beta-only vs `model.tp_relative._hat` fitted values.
