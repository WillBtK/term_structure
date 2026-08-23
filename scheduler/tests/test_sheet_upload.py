"""The sheet-to-Storage path: inert without creds, resplice logic sound."""
import os
from unittest import mock

from scheduler.supabase_export import SHEET_HTML, _fresh_sheet_html, maybe_upload_sheet


def test_inert_without_credentials():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SUPABASE_URL", None)
        os.environ.pop("SUPABASE_KEY", None)
        assert maybe_upload_sheet() is False


def test_fresh_sheet_carries_a_payload_and_the_full_page():
    h = _fresh_sheet_html()
    assert "const D = {" in h and '"asof"' in h
    assert "<title>" in h and "ruleDeficitFull" in h
    assert len(h) > 50_000


def test_committed_page_has_the_splice_anchor():
    import re
    raw = SHEET_HTML.read_text(encoding="utf-8")
    assert re.search(r"const D = \{.*?\};\n", raw, flags=re.S)
