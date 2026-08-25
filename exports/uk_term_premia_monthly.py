"""Monthly UK term premia — same schema as the US export, read by the OBR model
(WillBtK/OBR_Model, forward refinancing mode).

Reuses ``exports.term_premia_monthly.build``, which already decomposes any wired market
by ``country``; this just points it at the UK and writes the CSV. Runs in the daily
workflow after the canonical ``viz/data_daily.parquet`` has been rebuilt.

    python -m exports.uk_term_premia_monthly
"""
from __future__ import annotations

from pathlib import Path

from exports.term_premia_monthly import build

OUT = Path(__file__).resolve().parent / "uk_term_premia_monthly.csv"


def _main() -> None:
    m = build("UK")
    OUT.write_text(m.to_csv())
    tp = [c for c in m.columns if c.startswith("tp")]
    rtp = [c for c in m.columns if c.startswith("rtp")]
    print(f"wrote {OUT.name}  ({len(m)} months, "
          f"{str(m.index.min())[:7]}..{str(m.index.max())[:7]})")
    print(f"  nominal {len(tp):>2} tenors: {' '.join(tp)}")
    print(f"  real    {len(rtp):>2} tenors: {' '.join(rtp)}")


if __name__ == "__main__":
    _main()
