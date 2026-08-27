/* The executive summary as a PowerPoint.
 *
 * The HTML original is the design of record, so this is a transcription rather than
 * a redesign: same words, same figures (both read from the deck payload, not typed
 * in), same Tufte-ish typography, same 16:9 geometry. The stage is 1280x720 CSS px
 * and a wide PowerPoint slide is 13.333x7.5in at 96dpi -- exactly the same box --
 * so every coordinate below is the measured browser pixel divided by 96.
 *
 * Text is native and editable. The one exception is slide 1's chart, a hand-built
 * SVG with a shaded band between two lines that PowerPoint has no native chart for;
 * build_pptx.py rasterises it at 3x and it lands here as an image.
 */
const pptx = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const HERE = __dirname;
const P = JSON.parse(fs.readFileSync(path.join(HERE, "data", "deck_payload.json"), "utf8"));
const CHART = path.join(HERE, "pptx", "chain_chart.png");
const OUT = process.argv[2] || path.join(HERE, "dist", "term-premium-executive-summary.pptx");

const px = (v) => v / 96;                    // browser pixel -> inch
const C = {                                  // the page's own custom properties
  stage: "FBFAF8", card: "FFFFFF", ink: "191C1F", ink2: "4C5257",
  muted: "797F85", rule: "E4E0D9",
  us: "2F6FD4", uk: "C2410C", de: "157F5B", jp: "8C2C86",
};
// Cambria for the humanist serif and Courier New for the figures: both ship with
// Office and render at their true widths, so what is checked here is what opens
// on someone else's machine.
const SERIF = "Cambria", MONO = "Courier New";

const M = px(62);                            // the page's left and right margin
const W = px(1156);                          // content width between them

const deck = new pptx();
deck.layout = "LAYOUT_WIDE";                 // 13.333 x 7.5in
deck.author = "Term Premium Atlas";
deck.title = "Term Premium Atlas - executive summary";

/* Runs: "*bold*" segments inside a string become bold ink, the rest body colour. */
function runs(s, { size, color = C.ink2, bold = C.ink, face = SERIF }) {
  return s.split(/\*([^*]+)\*/).map((part, i) => ({
    text: part,
    options: { fontFace: face, fontSize: size,
               color: i % 2 ? bold : color, bold: i % 2 === 1 },
  })).filter((r) => r.text.length);
}

/* Every slide carries the same head and foot. */
function shell(kicker, title, take, takeW = px(1010)) {
  const s = deck.addSlide();
  s.background = { color: C.stage };
  s.addText(kicker.toUpperCase(), {
    x: M, y: px(36), w: W, h: px(18), isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 9, color: C.muted, charSpacing: 1.4 });
  s.addText(title, {
    x: M, y: px(58), w: W, h: px(46), isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 25.5, color: C.ink });
  s.addText(runs(take, { size: 11.5 }), {
    x: M, y: px(105), w: takeW, h: px(64), isTextBox: true, margin: 0,
    lineSpacing: 17.5, valign: "top" });
  s.addText("Term Premium Atlas  -  executive summary", {
    x: M, y: px(686), w: px(500), h: px(16), isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 8.5, color: C.muted });
  s.addText(P.asof_label, {
    x: px(718), w: px(500), y: px(686), h: px(16), isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 8.5, color: C.muted, align: "right" });
  return s;
}

/* A bordered white panel, the page's one container device. */
function panel(s, x, y, w, h) {
  s.addShape(deck.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.04,
    fill: { color: C.card }, line: { color: C.rule, width: 0.75 } });
}
function tag(s, x, y, text) {
  s.addText(text.toUpperCase(), {
    x, y, w: px(400), h: px(14), isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 8.5, color: C.muted, charSpacing: 1 });
}

/* ---- 1: what it is ------------------------------------------------------- */
const t = P.today.US;
const expected = (t.y - t.tp).toFixed(2);

const s1 = shell("Executive summary - 1 of 2",
  "Turning macro views into bond yields, across the G4",
  "A bond yield is the policy rate the market expects, *plus a term premium* - and the "
  + "premium is most of what moves long yields. Our macro process gives us growth, "
  + "inflation and policy rates; this framework converts them into a *ten-year yield and "
  + "a total return* for the US, Germany, Japan and the UK.");

const STEPS = [
  "*We measure the premium* - the same way in all four markets - a curve model anchored "
  + "on surveys, corrected for the zero-rate years and for regime changes",
  "*The US sets the level* - the four premia share one common factor worth about *80%* "
  + "of their variation, and the US leads it",
  "*Three drivers explain the US* - *inflation* and the volatility it brings, the "
  + "*projected deficit*, and how much is held by *price-insensitive buyers*",
  "*Other markets are the US plus a difference* - relative inflation, relative ownership, "
  + "relative fiscal position",
  "*Your macro path prices out* - the deficit follows your growth view; both legs converge "
  + "from today's market pricing, never jumping to the answer",
];
STEPS.forEach((text, i) => {
  const y = px(314 + i * 48.2);
  s1.addShape(deck.ShapeType.roundRect, {
    x: M, y, w: px(19), h: px(19), rectRadius: 0.15, fill: { color: C.muted }, line: null });
  s1.addText(String(i + 1), {
    x: M, y, w: px(19), h: px(19), isTextBox: true, margin: 0, align: "center",
    valign: "middle", fontFace: MONO, fontSize: 8, color: C.stage });
  s1.addText(runs(text, { size: 10 }), {
    x: px(93), y: px(311 + i * 48.2), w: px(539), h: px(44),
    isTextBox: true, margin: 0, lineSpacing: 14.6, valign: "top" });
});

// The chart: a shaded band between the ten-year yield and expected policy, which is
// the term premium made visible. No native PowerPoint chart draws that, so it comes
// across as a 3x raster of the page's own SVG.
s1.addImage({ path: CHART, x: px(652), y: px(220), w: px(576), h: px(334) });

panel(s1, px(658), px(542), px(560), px(78));
tag(s1, px(675), px(557), "Today, United States");
s1.addText([
  { text: "10y yield ", options: { fontFace: MONO, fontSize: 9.5, color: C.ink2 } },
  { text: t.y.toFixed(2) + "%", options: { fontFace: MONO, fontSize: 9.5, color: C.ink, bold: true } },
  { text: "  =  expected policy ", options: { fontFace: MONO, fontSize: 9.5, color: C.ink2 } },
  { text: expected + "%", options: { fontFace: MONO, fontSize: 9.5, color: C.ink, bold: true } },
  { text: "  +  term premium ", options: { fontFace: MONO, fontSize: 9.5, color: C.ink2 } },
  { text: t.tp.toFixed(2) + "%", options: { fontFace: MONO, fontSize: 9.5, color: C.us, bold: true } },
], { x: px(675), y: px(578), w: px(530), h: px(26), isTextBox: true, margin: 0 });

/* ---- 2: why believe it --------------------------------------------------- */
const bp = Math.round(P.fiscal.coef / 10) * 10;

const s2 = shell("Executive summary - 2 of 2",
  "What it rests on, and what it does not claim",
  "Four pieces of evidence, each measured on our own data across the four markets and "
  + "forty years - and three things we would not want inferred from it.");

// The browser lays these out with flex and leaves the lower third of the slide empty;
// on a printed page that reads as an accident, so the rows are spaced to sit in it.
const EV = [
  ["~80%", C.us, 112, "Term premia move together",
   "One common factor explains about 80% of the variation across the four markets. The "
   + "share moves through time, which is why we model the differences rather than assume "
   + "them away."],
  ["1-1.5%", C.jp, 112, "Inflation's effect is non-linear",
   "Below roughly 1 to 1.5% inflation, volatility does not respond to the level. Above "
   + "it, it does - Okun's 1970 result, still holding across six markets and sixty years."],
  ["~" + bp + "bp", C.uk, 112, "Deficits price, but only since 2008",
   "A point of GDP on the projected deficit is worth roughly " + bp + " basis points of "
   + "term premium post-2008. Before 2008 the market ignored it."],
  ["observable", C.de, 160, "The chain can be checked",
   "Inflation volatility drives the stock-bond correlation across six markets - a link we "
   + "can verify against something the market prices directly, rather than only against "
   + "our own model."],
];
EV.forEach(([key, color, keyW, head, body], i) => {
  const cx = i % 2 ? px(648) : px(62);
  const cy = i < 2 ? px(180) : px(322);
  panel(s2, cx, cy, px(570), px(126));
  s2.addText(key, {
    x: cx + px(18), y: cy + px(16), w: px(keyW), h: px(30), isTextBox: true, margin: 0,
    fontFace: MONO, fontSize: 18.5, color });
  s2.addText([
    { text: head + "\n", options: { fontFace: SERIF, fontSize: 10, color: C.ink, bold: true } },
    { text: body, options: { fontFace: SERIF, fontSize: 9.75, color: C.ink2 } },
  ], { x: cx + px(30 + keyW), y: cy + px(14), w: px(570 - 48 - keyW), h: px(100),
       isTextBox: true, margin: 0, lineSpacing: 14 });
});

panel(s2, M, px(484), W, px(104));
tag(s2, M + px(17), px(500), "What it is not");
s2.addText(runs(
  "It *prices scenarios; it does not forecast* - out of sample at three years it does no "
  + "better than assuming yields stay put, which is the honest result. *Japan is "
  + "directional only.* And the speed at which the market converges to a stated view is a "
  + "*judgement we expose as a dial*, not a number we measured. Present it as a "
  + "sensitivity, never as a point forecast.", { size: 9.75 }),
  { x: M + px(17), y: px(520), w: W - px(34), h: px(56),
    isTextBox: true, margin: 0, lineSpacing: 14.5 });

fs.mkdirSync(path.dirname(OUT), { recursive: true });
deck.writeFile({ fileName: OUT }).then(() => console.log("wrote " + OUT));
