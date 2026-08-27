"""Build the executive summary as a PowerPoint.

Two steps, because the deck is a browser page and PowerPoint is not: rasterise the
one exhibit PowerPoint cannot draw natively (a shaded band between two lines), then
hand the figures and that image to ``pptx_gen.js``. Everything else lands as native,
editable text.

    python -m presentation.build_pptx
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from presentation.export_pdf import CHROME, GOTO, PRINT_CSS

HERE = Path(__file__).resolve().parent
CHART = HERE / "pptx" / "chain_chart.png"
OUT = HERE / "dist" / "term-premium-executive-summary.pptx"
SCALE = 3          # so the chart survives a projector and a print


def render_chart() -> Path:
    """Slide 1's chart, at SCALE, with room for the labels that sit outside its box."""
    from playwright.sync_api import sync_playwright

    CHART.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=str(CHROME) if CHROME.exists() else None)
        page = browser.new_page(viewport={"width": 1280, "height": 720},
                                device_scale_factor=SCALE)
        page.emulate_media(media="screen", reduced_motion="reduce")
        page.goto((HERE / "summary.html").as_uri(), wait_until="networkidle")
        page.add_style_tag(content=PRINT_CSS)
        page.evaluate(GOTO, 0)
        page.wait_for_timeout(300)
        box = page.evaluate("()=>{const r=document.querySelector('svg')"
                            ".getBoundingClientRect();"
                            "return {x:r.x,y:r.y,w:r.width,h:r.height};}")
        # The note above the plot and the end labels to its right are drawn outside the
        # svg's own box (overflow is visible), so the clip is padded to keep them.
        page.screenshot(path=str(CHART),
                        clip={"x": box["x"] - 6, "y": box["y"] - 22,
                              "width": box["w"] + 16, "height": box["h"] + 34})
        browser.close()
    return CHART


def main() -> None:
    render_chart()
    subprocess.run(["node", str(HERE / "pptx_gen.js"), str(OUT)], check=True)
    print(f"{OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
