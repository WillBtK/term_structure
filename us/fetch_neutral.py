"""Fetch the US endpoint surveys — Philadelphia Fed SPF ``BILL10`` and ``CPI10``.

These are the same inputs Kim-Wright uses to anchor expectations: the 10-year
annual-average 3-month T-bill return forecast (1992+) and the median 10-year CPI
forecast (1991Q4+), both public and maintained by the Philadelphia Fed indefinitely.
CPI10 is quarterly; BILL10 is asked in the FIRST-QUARTER round only, so the nominal
anchor is bridged between Februaries by the NY Fed SME (``us.fetch_sme``,
``us.anchor.bridged_forward_reads``) — so unlike the other markets the US never needs a proprietary seed or a
hand-off; the living survey IS the rung.

One caveat carried through to the wiring (``viz/export_daily``): both are 0-10y
AVERAGES, not 6-10y-ahead readings, so they discipline the model's expected 10y
AVERAGE rate rather than its terminal endpoint. The export applies them through an
average-matching transform instead of substituting them for theta directly.

MEANS, not medians. The SPF median clusters on round numbers (BILL10's median sat at
2.50 for 2014-17 and lands on 3.00 in 2011 and 2026), so it steps when enough
forecasters cross a round number and ignores movement either side of it. The cross-
forecaster mean moves continuously with the panel. Mean minus median on the 2-10y
forward: -0.05pp on average since 1992, sd 0.15pp per round.

Best-effort like the other anchor fetchers: never raises, never overwrites a good
committed file with a bad parse.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
_BASE = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
         "survey-of-professional-forecasters/data-files/files/{stat}_{var}_level.xlsx")
STAT = "mean"                    # cross-forecaster mean; see module docstring
FILES = {"bill10": "spf_bill10.csv", "cpi10": "spf_cpi10.csv",
         "tbill_y1": "spf_tbill_y1.csv", "tbill_y2": "spf_tbill_y2.csv"}


def _fetch_spf_level(var: str, out_name: str, out_dir: "Path | None" = None) -> bool:
    """One SPF mean-level file -> committed date,value CSV (percent).

    The workbook is (YEAR, QUARTER, <VAR>); the survey is fielded mid-quarter, so the
    observation is dated to the quarter's middle month's 15th — close enough to the
    actual deadline for a 6-10y object, and stable across releases.
    """
    out = Path(out_dir) if out_dir is not None else ANCHOR_DIR
    try:
        import requests
        r = requests.get(_BASE.format(stat=STAT, var=var), timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        import io
        df = pd.read_excel(io.BytesIO(r.content))
        cols = {c.strip().upper(): c for c in df.columns}
        y, q = cols["YEAR"], cols["QUARTER"]
        vc = [c for c in df.columns if c not in (y, q)][-1]
        df = df.dropna(subset=[vc])
        dates = pd.to_datetime(dict(year=df[y].astype(int),
                                    month=df[q].astype(int) * 3 - 1, day=15))
        s = pd.Series(pd.to_numeric(df[vc], errors="coerce").to_numpy(), index=dates).dropna()
        if len(s) < 20 or not (0.0 <= s.median() <= 10.0):
            raise ValueError(f"implausible parse: n={len(s)} median={s.median()}")
        out.mkdir(parents=True, exist_ok=True)
        s.rename("value").rename_axis("date").round(4).to_csv(out / out_name)
        logger.info("SPF %s: %d obs %s..%s (latest %.2f%%)", var.upper(),
                    len(s), s.index.min().date(), s.index.max().date(), s.iloc[-1])
        return True
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("SPF %s fetch failed (%s) — keeping committed file", var, exc)
        return False


def fetch_spf_tbill_annuals(out_dir: "Path | None" = None) -> bool:
    """The near-year T-bill forecasts (TBILLA = current year, TBILLB = next year),
    quarterly since 1992 alongside BILL10's survey rounds. These are what turn the
    0-10y BILL10 average into a 2-10y FORWARD by pure arithmetic — the near path is
    surveyed, so stripping it involves no model (see ``us.anchor.survey_forward``)."""
    out = Path(out_dir) if out_dir is not None else ANCHOR_DIR
    try:
        import requests, io
        r = requests.get(_BASE.format(stat=STAT, var="tbill"), timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content))
        cols = {c.strip().upper(): c for c in df.columns}
        y, q = cols["YEAR"], cols["QUARTER"]
        dates = pd.to_datetime(dict(year=df[y].astype(int),
                                    month=df[q].astype(int) * 3 - 1, day=15))
        ok = 0
        for var, name in (("TBILLA", FILES["tbill_y1"]), ("TBILLB", FILES["tbill_y2"])):
            sv = pd.Series(pd.to_numeric(df[cols[var]], errors="coerce").to_numpy(),
                           index=dates).dropna()
            if len(sv) < 40 or not (-1.0 <= sv.median() <= 12.0):
                raise ValueError(f"implausible {var}: n={len(sv)} median={sv.median()}")
            out.mkdir(parents=True, exist_ok=True)
            sv.rename("value").rename_axis("date").round(4).to_csv(out / name)
            logger.info("SPF %s: %d obs %s..%s (latest %.2f%%)", var,
                        len(sv), sv.index.min().date(), sv.index.max().date(), sv.iloc[-1])
            ok += 1
        return ok == 2
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("SPF TBILL annuals fetch failed (%s) — keeping committed files", exc)
        return False


def fetch_spf_bill10(out_dir=None) -> bool:
    return _fetch_spf_level("bill10", FILES["bill10"], out_dir)


def fetch_spf_cpi10(out_dir=None) -> bool:
    return _fetch_spf_level("cpi10", FILES["cpi10"], out_dir)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_spf_bill10()
    fetch_spf_cpi10()
    fetch_spf_tbill_annuals()


if __name__ == "__main__":
    _main()
