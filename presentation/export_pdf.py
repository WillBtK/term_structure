"""Render the deck and the executive summary to PDF.

Both pages are a JavaScript slide engine writing into a single 1280x720 ``#stage``
that is scaled to the window -- there is no print stylesheet and no scrolling
document to print. So the export drives the engine itself: for each slide, set the
index, render every build step at once, and print that one stage at its native
size. One PDF page per slide, landscape 1280x720, backgrounds on.

    python -m presentation.export_pdf            # both, into presentation/dist/
    python -m presentation.export_pdf --deck     # just the long deck
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"

# (source page, output name). The stems are what a recipient sees in a mail client,
# so they name the document rather than the file that produced it.
DOCS = {
    "deck": (HERE / "index.html", "term-premium-from-macro-views-to-bond-yields.pdf"),
    "summary": (HERE / "summary.html", "term-premium-executive-summary.pdf"),
}

W, H = 1280, 720

CHROME = Path("/opt/pw-browsers/chromium")

# The stage is centred and scaled to the viewport by fit(); for print it has to sit
# at the origin at 1:1 or the page picks up the letterbox. The HUD is a presenter
# affordance, not part of the document.
PRINT_CSS = """
html,body{overflow:visible!important;background:#fff!important;margin:0!important}
#frame{position:static!important;display:block!important}
#stage{transform:none!important;border:none!important;margin:0!important}
.hud{display:none!important}
"""

# Slide n of N: set the index, fully reveal the builds, render without animation.
# The two pages use different slide shapes -- the deck's are {steps, render}, the
# summary's are bare functions -- so only touch K when there is a K to touch.
GOTO = """(i)=>{
  S=i;
  const s=SLIDES[i];
  if(typeof s!=='function'){ K=s.steps; }
  render(false);
  return SLIDES.length;
}"""


def export(src: Path, out: Path) -> int:
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    with sync_playwright() as pw:
        # The container ships one pinned Chromium under PLAYWRIGHT_BROWSERS_PATH which
        # need not match the Python package's expected build; point at it directly
        # rather than triggering a download that the egress policy would refuse.
        browser = pw.chromium.launch(executable_path=CHROME if CHROME.exists() else None)
        page = browser.new_page(viewport={"width": W, "height": H})
        # Screen media (there is no print stylesheet) and no motion, so a slide is
        # captured in its settled state rather than mid-transition.
        page.emulate_media(media="screen", reduced_motion="reduce")
        page.goto(src.as_uri(), wait_until="networkidle")
        n = int(page.evaluate("SLIDES.length"))
        page.add_style_tag(content=PRINT_CSS)
        for i in range(n):
            page.evaluate(GOTO, i)
            page.wait_for_timeout(250)
            # page_ranges: a stage that overflows its box by a sub-pixel spills a
            # blank second sheet, and one slide is always one page.
            pdf = page.pdf(width=f"{W}px", height=f"{H}px", print_background=True,
                           page_ranges="1",
                           margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
            writer.append(PdfReader(io.BytesIO(pdf)))
        browser.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        writer.write(fh)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", action="store_true", help="long deck only")
    ap.add_argument("--summary", action="store_true", help="executive summary only")
    ap.add_argument("--out", type=Path, default=DIST)
    a = ap.parse_args()
    want = [k for k in DOCS if getattr(a, k)] or list(DOCS)
    for key in want:
        src, name = DOCS[key]
        out = a.out / name
        n = export(src, out)
        print(f"{key:8s} {n:2d} pages  {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
