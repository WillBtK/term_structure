"""BoE Market Participants Survey neutral-rate parsing (uk.fetch_neutral)."""
import io

import pandas as pd

from uk import fetch_neutral as F

PAGE = ("<p>1bi) And where do you see the level of Bank Rate at which monetary policy is "
        "neither expansionary nor contractionary (often referred to as the neutral, "
        "natural or equilibrium rate)?</p>")


def _book(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=name, header=False, index=False)
    return buf.getvalue()


HDR = ["25th percentile", "50th percentile", "75th percentile", "Number of responses"]


def test_label_found_by_wording_across_numbering():
    assert F.neutral_label(PAGE) == "1bi"
    assert F.neutral_label(PAGE.replace("1bi)", "1g)")) == "1g"
    assert F.neutral_label(PAGE.replace("1bi)", "1b(i))")) == "1bi"
    assert F.neutral_label("<p>1b) Where do you see Bank Rate in five years?</p>") is None


def test_reads_the_labelled_sheet_not_a_lookalike():
    book = _book({"Q1b": [HDR, [3.5, 3.75, 4.0, 90]],        # a different question
                  "Q1bi": [HDR, [3.0, 3.25, 3.5, 91]]})
    assert F._round_value(book, "1bi") == (3.0, 3.25, 3.5)
    assert F._round_value(book, "1g") is None


def test_rejects_implausible_quartiles():
    book = _book({"Q1b": [HDR, [3.5, 3.25, 3.0, 91]]})       # out of order
    assert F._round_value(book, "1b") is None
