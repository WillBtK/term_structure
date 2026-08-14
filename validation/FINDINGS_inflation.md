# Real term premia & the inflation-risk premium (US + UK) — findings

> **⚠ Superseded numbers.** The **real** term-premium levels here predate the
> **mean-reversion floor** fix (it applies to real curves too, since the
> plain-AFNS expectations leg had the same near-unit-root persistence). Corrected
> real levels: US TIPS 10y ≈ +1.26%, UK index-linked 10y ≈ +1.24% — see
> **`FINDINGS_mean_reversion.md`**. The breakeven / inflation-risk-premium
> decomposition method and the RPI-CPI / LDI caveats below still stand, but the
> IRP = nominal-TP − real-TP levels shift with the corrected inputs.

The inflation-linked leg: **real** term premia (US TIPS, UK index-linked gilts)
and the **breakeven / inflation-risk-premium** decomposition, via the "two
separate AFNS, subtract" approach (chosen with the user). Real rates have **no
ELB** (no "real cash"), so the real TP is a plain-AFNS decomposition (bound pushed
to −100%, never binds); everything else reuses the shared pipeline (BRW-auto +
sample-mean-θ + observed-based premium).

Identities (Fisher): breakeven = nominal − real yield = E[avg inflation] +
(nominal TP − real TP); so **IRP = nominal TP − real TP**, and implied
**E[avg inflation] = breakeven − IRP**.

**Matched samples.** The IRP is only meaningful if nominal and real are estimated
on the *same* sample (different sample-mean short rates make the TP levels
non-comparable). So each IRP uses a matched pair — **US 2004+**, **UK 1992+** —
while the standalone nominal headlines keep their validated samples (US 1987+, UK
1992+). Real fits: US MAD 1.9 bp, UK MAD 1.4 bp (excellent).

## US — a clean decomposition

Real 10y TP **+0.92%**; nominal(2004+) 10y TP +0.98% → **10y IRP ≈ +0.06%**, rising
to ~+0.18% at 5y. Latest (2026-07):

| tenor | nominal TP | real TP | **IRP** | breakeven | E[inflation] |
|---|---|---|---|---|---|
| 2y | 0.19% | 0.07% | **0.11%** | 1.93% | 1.82% |
| 5y | 0.52% | 0.34% | **0.18%** | 2.33% | 2.15% |
| 10y | 0.98% | 0.92% | **0.06%** | 2.33% | 2.28% |

The implied expected inflation (~2.1-2.3%) sits **right at the Fed's 2% target plus
a small premium** — a strong external sanity check that the decomposition is
sensible. The 10y IRP history swings with the **TIPS liquidity premium** (the GSW
caveat): +1.1% in the 2008-12 crisis (TIPS sold off, breakeven collapsed to 0.4%),
−1.8% in the 2020-03 COVID dash-for-cash. These swings are the liquidity
differential embedded in the IRP, not pure inflation-risk repricing — flagged, and
the reason a liquidity factor (D'Amico-Kim-Wei) is the documented rigorous
extension.

## UK — large IRP, and an honest limitation

Latest (2026-07): nominal 10y TP +1.49% (the highest of the four markets), **real
10y TP −0.46%**, so the naive 10y IRP is **+1.95%** — far larger than the US:

| tenor | nominal TP | real TP | IRP | breakeven | E[inflation] |
|---|---|---|---|---|---|
| 5y | 0.89% | −0.64% | 1.53% | 3.44% | 1.91% |
| 10y | 1.49% | −0.46% | 1.95% | 3.22% | 1.28% |
| 20y | 2.04% | −0.18% | 2.22% | 3.30% | 1.08% |

**This IRP is over-stated, and the tell is the implied expected inflation of
~1.1-1.3% at 10-20y — implausibly low for the UK.** Two structural distortions,
both larger than the US liquidity caveat:
1. **RPI, not CPI.** UK index-linked gilts (and hence BoE breakevens) are
   **RPI**-linked; RPI has run ~0.8-1% above CPI. So the ~3.2% breakeven is an RPI
   breakeven — comparable to the US ~2.3% CPI breakeven once the wedge is removed,
   but the decomposition treats it as-is.
2. **LDI / pension demand depresses the real TP.** Structural liability-driven
   demand from UK pension funds for long index-linked gilts pushes real yields —
   and the real term premium — **negative** (−0.46% at 10y). Because IRP = nominal
   TP − real TP, that depressed real TP inflates the IRP and pushes implied
   expected inflation too low.

So the separate-AFNS IRP works cleanly for the US but is **contaminated for the
UK** by the RPI-CPI basis and the LDI real-yield distortion. Reported with the
caveat; the fixes (an RPI→CPI adjustment and a convenience/LDI factor, or a joint
nominal-real model) are documented extensions, not built here.

The UK real TP history is itself informative: **+1.8% during the Sep-2022 LDI
crisis** (forced pension selling spiked real yields) reverting to −0.5% as the
Bank's intervention and demand normalised — the mirror image of the nominal 10y TP
briefly going negative (−0.68%) in the same mini-budget gilt rout.

## Bottom line
- **US real TP + IRP: a validated success** — E[inflation] lands at target, IRP
  small and positive, liquidity swings understood.
- **UK: real TP and nominal TP are each sensible and useful; the *IRP split* is
  distorted** by RPI basis + LDI, and is reported as an upper bound on inflation
  risk with the mechanism spelled out — a genuine limit of the portable
  yields-only, separate-model approach for a market with strong structural real
  demand.
