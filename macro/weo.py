"""IMF World Economic Outlook, one same-vintage macro baseline for all four markets.

The endogenous-deficit rule (``macro.fiscal_rules``) needs a growth baseline to
difference the user's growth path against, and the honest baseline is one vintage on
one methodology for every market. WEO is the only source that provides growth,
inflation and the fiscal balance together for all four, to t+5 or further, machine
readable. The route that works is the IMF's SDMX 2.1 API (``api.imf.org`` -- the same
host ``supply.price_insensitive`` already uses for COFER); the Datamapper front door
on ``www.imf.org`` sits behind Akamai bot protection and is NOT reachable from this
environment.

Division of labour, decided deliberately:

* **Growth baseline** for the gap calculation: WEO, all four markets.
* **Deficit level**: stays with each market's own institution (CBO / OBR / AMECO /
  Cabinet Office) because the fiscal coefficient (+51.9bp per point) was measured on
  the CBO-family current-law basis, and WEO's staff-judgement basis is a different
  object -- it diverges from CBO by 0.75pp of GDP at t+5 for the US and from the
  Cabinet Office by ~1.3pp for Japan (documented in
  ``validation/FINDINGS_endogenous_deficit.md``).
* **Germany is the exception**: AMECO's two-year horizon is extended with the WEO
  shape (level-matched at the seam) in ``macro.fiscal_projections`` -- the two agree
  to ~0.1-0.2pp where they overlap, so the splice is nearly invisible today.

Values are cached to a committed CSV so everything runs offline; a failed fetch keeps
the committed file.
"""
from __future__ import annotations

import io
import logging
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "weo_g4.csv"

SDMX_DATA = "https://api.imf.org/external/sdmx/2.1/data/IMF.RES,WEO"
ISO = {"US": "USA", "UK": "GBR", "DE": "DEU", "JP": "JPN"}
# WEO indicator codes: general government net lending (% GDP, surplus-positive),
# real GDP growth (%), CPI inflation (%).
INDICATORS = {"balance": "GGXCNL_NGDP", "growth": "NGDP_RPCH", "inflation": "PCPIPCH"}

_TIMEOUT = 120


def _fetch_csv() -> pd.DataFrame:
    key = "+".join(ISO.values()) + "." + "+".join(INDICATORS.values()) + ".A"
    url = f"{SDMX_DATA}/{key}?startPeriod=2015"
    out = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
         "-A", "Mozilla/5.0",
         "-H", "Accept: application/vnd.sdmx.data+csv;version=1.0.0", url],
        capture_output=True, timeout=_TIMEOUT + 30)
    if out.returncode != 0:
        raise RuntimeError(f"curl {out.returncode}: {out.stderr[:150]!r}")
    d = pd.read_csv(io.BytesIO(out.stdout))
    iso_rev = {v: k for k, v in ISO.items()}
    ind_rev = {v: k for k, v in INDICATORS.items()}
    d = d[["COUNTRY", "INDICATOR", "TIME_PERIOD", "OBS_VALUE"]].dropna()
    d["country"] = d["COUNTRY"].map(iso_rev)
    d["indicator"] = d["INDICATOR"].map(ind_rev)
    d = d.dropna(subset=["country", "indicator"])
    return (d.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "value"})
             [["country", "indicator", "year", "value"]]
             .astype({"year": int, "value": float})
             .sort_values(["country", "indicator", "year"]))


def refresh() -> None:
    try:
        d = _fetch_csv()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(CACHE, index=False)
        logger.info("WEO: %d rows, %d..%d", len(d), d.year.min(), d.year.max())
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("WEO refresh failed (%s) -- keeping committed file", exc)


def series(country: str, indicator: str) -> dict:
    """{calendar year: value} from the committed cache. ``balance`` is returned as
    published (surplus-positive); use :func:`deficit` for the deficit-positive sign."""
    if not CACHE.is_file():
        raise FileNotFoundError("no committed WEO cache; run python -m macro.weo")
    d = pd.read_csv(CACHE)
    d = d[(d.country == country) & (d.indicator == indicator)]
    if d.empty:
        raise KeyError(f"WEO cache has no {country}/{indicator}")
    return {int(r.year): float(r.value) for r in d.itertuples()}


def deficit(country: str) -> dict:
    return {y: -v for y, v in series(country, "balance").items()}


def growth(country: str) -> dict:
    return series(country, "growth")


def inflation(country: str) -> dict:
    return series(country, "inflation")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    refresh()
    for c in ISO:
        g, d = growth(c), deficit(c)
        proj = {y: v for y, v in d.items() if y >= 2026}
        print(f"{c}: growth 2026 {g.get(2026, float('nan')):+.2f}%   deficit "
              + "  ".join(f"{y}:{v:.2f}" for y, v in sorted(proj.items())))


if __name__ == "__main__":
    _main()
