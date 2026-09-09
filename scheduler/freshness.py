"""Does the published dataset actually contain recent data?

A green run is not a fresh run. Every fetch in ``daily_update`` is guarded with a
cache fallback — deliberately, so one dead source cannot stop the other five
publishing — which means a source that quietly stops serving new days looks
*identical to success*: the job is green, the Atlas rebuilds, and the numbers simply
stop moving. The SNB served a frozen cube for a year with HTTP 200. The scenario
sheet served a payload stamped 2026-08-19 for three weeks because a FRED call failed
inside a ``try``. Neither showed up as a failure anywhere.

This module is the check that turns silence into a signal. Two questions, because
they fail differently:

**Absolute** — is this series behind *today*? This is the one that catches a
whole-pipeline freeze, and it is exactly what ``daily_update.audit_staleness``
cannot see: that check measures each market against the freshest market, so if
everything stops together, every market is equally stale and nothing is reported.

**Relative** — is this series behind the *other* markets? Catches one source dying
while the rest keep moving, which the absolute check can miss when a market's own
cadence is genuinely slow.

No hand-set per-market budget
-----------------------------
Markets publish on different rhythms — Treasury same-day, the BoE and MoF at T+1,
the SNB slower, and every one of them takes its own public holidays. A fixed
"stale after N days" threshold would either cry wolf on Switzerland or sleep through
a dead US feed. So the budget is **derived from each series' own recent behaviour**:
the longest quiet spell it has had lately, plus a grace margin. A series is stale
when it has gone quiet for materially longer than it normally does — which needs no
per-market table, adapts automatically when a publisher changes cadence, and cannot
be invalidated by a holiday the code does not know about.

Usage:
    python -m scheduler.freshness            # print the report
    python -m scheduler.freshness --check    # exit 1 if anything is stale (CI gate)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("scheduler.freshness")
ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "viz" / "data_daily.parquet"

CADENCE_WINDOW_DAYS = 180   # how much recent history defines "how this series behaves"
GRACE_BUSINESS_DAYS = 2     # slack on top of its own worst recent gap
MIN_BUDGET_DAYS = 3         # floor: never alarm on a normal weekend + one holiday
RELATIVE_STALE_DAYS = 14    # calendar days behind the freshest series (the second check)

# BATCH PUBLISHERS — where the derived budget is wrong, and why.
#
# Deriving the budget from observation spacing silently assumes a source publishes each
# day's data on that day. For a source that publishes DAILY data in MONTHLY BATCHES the
# two quantities come apart completely: the SNB `rendoblim` cube contains every business
# day (so consecutive observations are one business day apart, giving a ~5-day budget)
# but is refreshed roughly monthly, covering through the previous month end. Verified
# against the source: on 2026-08-19 it ended 2026-07-31; on 2026-09-09 it ended
# 2026-08-31. Judged on its observation spacing it is "stale" for three weeks out of
# every four -- crying wolf on precisely the market the derived rule was meant to spare.
#
# Publication lag cannot be read off the observation dates, so for these it has to be
# stated: one publication period plus slack. The value alarms only if the source misses
# a whole further period, which is a real failure rather than its normal rhythm.
BATCH_PUBLISHERS = {
    "CH:nominal": (30, "SNB rendoblim cube: daily observations, refreshed ~monthly"),
}


def business_days_between(start: date, end: date) -> int:
    """Business days from ``start`` to ``end`` (weekends excluded, holidays not — the
    grace margin absorbs those, which is why the budget is derived rather than fixed)."""
    return int(np.busday_count(np.datetime64(start, "D"), np.datetime64(end, "D")))


def cadence_budget(dates, window_days: int = CADENCE_WINDOW_DAYS,
                   grace: int = GRACE_BUSINESS_DAYS,
                   floor: int = MIN_BUDGET_DAYS) -> int:
    """How long this series is allowed to go quiet, in business days, judged by how
    long it *normally* goes quiet.

    Takes the largest business-day gap between consecutive observations over the
    trailing ``window_days`` and adds ``grace``. A daily curve earns a small budget;
    a series that genuinely publishes weekly earns a larger one, without anyone
    having to write that down per market.
    """
    d = sorted({pd.Timestamp(x).date() for x in dates})
    if len(d) < 2:
        return max(floor, grace)
    cutoff = d[-1] - pd.Timedelta(days=window_days).to_pytimedelta()
    recent = [x for x in d if x >= cutoff]
    if len(recent) < 2:
        recent = d[-2:]
    worst = max(business_days_between(a, b) for a, b in zip(recent, recent[1:]))
    return max(floor, worst + grace)


@dataclass
class Verdict:
    key: str
    last: date
    behind: int          # business days behind as_of
    budget: int          # its allowance: derived from cadence, or a stated batch period
    behind_newest: int   # calendar days behind the freshest series
    stale: bool          # absolute breach
    lagging: bool        # relative breach
    basis: str = "cadence"   # where the budget came from, for the report

    @property
    def status(self) -> str:
        if self.stale:
            return "STALE"
        return "LAGGING" if self.lagging else "ok"


def series_last_dates(path: Path = DAILY) -> dict:
    """{"US:nominal": [dates...]} from the canonical dataset."""
    df = pd.read_parquet(path, columns=["country", "instrument", "date"])
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for (c, inst), g in df.groupby(["country", "instrument"], observed=True):
        out[f"{c}:{inst}"] = sorted(set(g["date"].dt.date))
    return out


def audit(series: dict, as_of: date | None = None,
          relative_days: int = RELATIVE_STALE_DAYS, **budget_kw) -> list:
    """Verdict per series, worst first."""
    as_of = as_of or date.today()
    if not series:
        return []
    newest = max(d[-1] for d in series.values() if d)
    out = []
    for key, dates in series.items():
        if not dates:
            continue
        last = dates[-1]
        if key in BATCH_PUBLISHERS:
            budget, basis = BATCH_PUBLISHERS[key]
        else:
            budget, basis = cadence_budget(dates, **budget_kw), "cadence"
        behind = business_days_between(last, as_of)
        behind_newest = (newest - last).days
        out.append(Verdict(key=key, last=last, behind=behind, budget=budget,
                           behind_newest=behind_newest,
                           stale=behind > budget,
                           # A batch publisher is *expected* to trail the daily markets
                           # by up to a full period, so the relative check would fire on
                           # it every month for no reason. Its absolute budget above is
                           # the meaningful test.
                           lagging=(basis == "cadence") and behind_newest > relative_days,
                           basis=basis))
    return sorted(out, key=lambda v: (-v.behind, v.key))


def annotate(level: str, message: str) -> None:
    """Emit a GitHub Actions annotation when running in CI, so a problem lands in the
    run's summary UI instead of at line 700 of a log nobody opens. A plain log line
    everywhere else."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message}", flush=True)
    (logger.error if level == "error" else logger.warning)(message)


def report(verdicts: list, annotate_ci: bool = True) -> str:
    """Human-readable table; annotates each breach when in CI."""
    if not verdicts:
        return "freshness: no series found"
    width = max(len(v.key) for v in verdicts)
    lines = [f"{'series'.ljust(width)}  last         behind  budget  vs newest  status  basis",
             "-" * (width + 54)]
    for v in verdicts:
        lines.append(f"{v.key.ljust(width)}  {v.last}  {v.behind:>6}  {v.budget:>6}"
                     f"  {v.behind_newest:>9}  {v.status:<7} {v.basis}")
        if annotate_ci and v.stale:
            why = ("its own recent cadence" if v.basis == "cadence"
                   else f"its publication period ({v.basis})")
            annotate("error", f"STALE: {v.key} ends {v.last} — {v.behind} business days "
                              f"behind today, against a {v.budget}-day budget from "
                              f"{why}. The upstream source has probably stopped "
                              f"publishing or changed shape.")
        elif annotate_ci and v.lagging:
            annotate("warning", f"LAGGING: {v.key} ends {v.last} — {v.behind_newest} days "
                                f"behind the freshest market. Within its own cadence, so "
                                f"this is a slow publisher rather than a dead one.")
    return "\n".join(lines)


def check(path: Path = DAILY, as_of: date | None = None) -> tuple:
    """(verdicts, n_stale). Reads the canonical dataset."""
    verdicts = audit(series_last_dates(path), as_of=as_of)
    return verdicts, sum(v.stale for v in verdicts)


def _main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any series is stale (CI gate)")
    ap.add_argument("--path", type=Path, default=DAILY)
    a = ap.parse_args()
    if not a.path.exists():
        msg = f"canonical dataset missing: {a.path} — nothing to check"
        annotate("error", msg)
        raise SystemExit(1 if a.check else 0)
    verdicts, n_stale = check(a.path)
    print("\n=== data freshness ===")
    print(report(verdicts))
    print(f"{n_stale}/{len(verdicts)} series stale")
    if a.check and n_stale:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
