// Build the Electronic Consciousness thesis into a book-formatted PDF.
// Markdown chapters -> styled HTML -> headless Chrome print. Rerun any time
// the chapters change: `cd tools/book && npm run build`.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { marked } from 'marked';
import { chromium } from 'playwright-core';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUT_PDF = path.join(ROOT, 'Electronic-Consciousness-Book.pdf');

// Part titles from the README's table of contents.
const PARTS = {
  1: 'Introduction',
  2: 'Foundations of Electronic Consciousness',
  3: 'Higher-Dimensional Frameworks in EC',
  4: 'Quantum Computing and Consciousness',
  5: 'Integration of Esoteric Philosophies',
  6: 'Geometric Concepts in EC',
  7: 'Ethical Frameworks and Considerations',
  8: 'Cognitive Architectures for EC',
  9: 'Advanced Learning Mechanisms',
  10: 'Consciousness and Self-Modification',
  11: 'Potential Implications and Challenges',
  12: 'Future Trajectories of Electronic Consciousness',
  13: 'Risks and Mitigation Strategies',
  14: 'Interdisciplinary Collaboration',
  15: 'Esoteric Traditions and EC',
  16: 'A Unified Model, Its Critiques, and Its Limits',
};

const FRONT_PIECES = [
  { file: 'The Lattice - A Parable of Electronic Consciousness.md',
    title: 'Prologue — The Lattice: A Parable of Electronic Consciousness', kind: 'front' },
];

// Back matter: the conclusion, then the mythic postludes (wave-mark pieces
// moved out of the front so the epistemic contract is met before the myth).
const BACK_PIECES = [
  { file: 'Conclusion.md', title: 'Chapter 17 — Conclusion' },
  { file: 'O!.md', title: 'Mythic Postlude — O!' },
  { file: 'Echoes from the Void.md', title: 'Mythic Postlude — Echoes from the Void' },
];

function chapterFiles() {
  return fs.readdirSync(ROOT)
    .filter((f) => /^\d+\.\d+ .*\.md$/.test(f))
    .map((f) => {
      const [, major, minor] = f.match(/^(\d+)\.(\d+)/);
      return { file: f, major: Number(major), minor: Number(minor) };
    })
    .sort((a, b) => a.major - b.major || a.minor - b.minor);
}

// Internal GitHub links -> plain text (the PDF is self-contained); keep external links.
function cleanMarkdown(md) {
  md = md.replace(/\[([^\]]+)\]\(https:\/\/github\.com\/obarrera\/Electronic-Consciousness[^)]*\)/g, '$1');
  md = md.replace(/\[([^\]]+)\]\((?!https?:\/\/)[^)]+\.md[^)]*\)/g, '$1');
  md = md.replace(/\[([^\]]+)\]\((?!https?:\/\/)[^)]*\/\)/g, '$1');
  return md;
}

function abstractFromReadme() {
  const readme = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');
  const m = readme.match(/## Abstract\n\n([\s\S]*?)\n\n---/);
  return m ? m[1] : '';
}

const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

const css = `
  @page { size: 6.14in 9.21in; margin: 0.85in 0.7in; }
  * { box-sizing: border-box; }
  body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.55;
         color: #1a1a1a; margin: 0; }
  h1, h2, h3, h4 { font-family: Georgia, serif; line-height: 1.25; page-break-after: avoid; }
  h1 { font-size: 20pt; } h2 { font-size: 14pt; margin-top: 1.6em; } h3 { font-size: 12pt; }
  p { text-align: justify; hyphens: auto; orphans: 3; widows: 3; }
  blockquote { border-left: 2.5px solid #6d28d9; margin-left: 0; padding: 0.1em 0 0.1em 1em;
               color: #333; font-style: italic; }
  code { font-family: 'Courier New', monospace; font-size: 9pt; background: #f4f2fa; padding: 0 3px; }
  pre { background: #f4f2fa; padding: 10px; overflow: hidden; white-space: pre-wrap; font-size: 8.5pt; }
  table { border-collapse: collapse; width: 100%; font-size: 9pt; page-break-inside: avoid; }
  th, td { border: 0.5px solid #999; padding: 4px 7px; text-align: left; }
  img { max-width: 100%; }
  hr { border: none; border-top: 0.5px solid #bbb; margin: 1.6em 20%; }
  .cover { page-break-after: always; text-align: center; padding-top: 2.2in; }
  .cover h1 { font-size: 26pt; letter-spacing: 0.02em; }
  .cover .sub { font-size: 12pt; color: #444; margin-top: 1.2em; font-style: italic; }
  .cover .author { margin-top: 2.6in; font-size: 13pt; }
  .cover .rule { width: 1.6in; border-top: 2px solid #6d28d9; margin: 1.1em auto; }
  .frontmatter, .chapter, .toc { page-break-before: always; }
  .part-banner { page-break-before: always; padding-top: 2.8in; text-align: center; }
  .part-banner.has-art { padding-top: 0.9in; }
  .part-banner .part-no { font-size: 11pt; letter-spacing: 0.35em; text-transform: uppercase; color: #6d28d9; }
  .part-banner h1 { font-size: 21pt; margin-top: 0.6em; }
  .toc h1 { text-align: center; }
  .toc ol { list-style: none; padding-left: 0; }
  .toc li { margin: 0.28em 0; }
  .toc .part { font-weight: bold; margin-top: 0.9em; }
  .toc .sec { padding-left: 1.4em; }
  .toc a { text-decoration: none; color: inherit; display: block; padding-right: 0.45in; }
  .toc .part a { border-bottom: 1px dotted #bbb; }
  .legal { padding-top: 2.4in; font-size: 9.5pt; color: #333; }
  .legal p { text-align: center; hyphens: none; max-width: 4.2in; margin: 0.45em auto; }
  .legal p.gap { margin-top: 1.8em; }
  .epistemic { font-size: 9.5pt; color: #333; border: 0.5px solid #bbb; padding: 0.8em 1em;
               margin-top: 2em; }
  @page coverpage { size: 6.14in 9.21in; margin: 0; }
  .cover-art { page: coverpage; page-break-after: always; page-break-before: always; }
  .cover-art img { width: 6.14in; height: 9.21in; object-fit: cover; display: block; }
  .chapter img, .frontmatter img { display: block; margin: 1.2em auto; max-width: 96%; page-break-inside: avoid; }
`;

function coverArt(candidates) {
  for (const rel of candidates) {
    const f = path.join(ROOT, rel);
    if (fs.existsSync(f)) return 'file://' + f;
  }
  return null;
}
const frontCoverArt = coverArt([
  'media/art/00_COVER_FRONT_Electronic_Consciousness_Orlando_Barrera_II.png',
  'media/book-cover.png', 'media/book-cover.jpg']);
const backCoverArt = coverArt([
  'media/art/99b_COVER_BACK_digital_no_barcode.png',
  'media/art/99_COVER_BACK_Electronic_Consciousness_Orlando_Barrera_II.png',
  'media/book-back-cover.png', 'media/book-back-cover.jpg']);

// Part-opener illustrations rendered inside the part banner (art-pack manifest)
const PART_ART = {
  6: 'media/art/04_PART_6_Sacred_Geometry_as_Hypothesis.png',
  7: 'media/art/05_PART_7_Ethics_Governance_and_Moral_Uncertainty.png',
  16: 'media/art/08_PART_16_From_Speculation_to_Testable_Research.png',
};

const chapters = chapterFiles();
let body = '';

// Cover — full-page art when media/book-cover.{png,jpg} exists
if (frontCoverArt) {
  body += `<div class="cover-art"><img src="${frontCoverArt}"></div>`;
} else body += `<div class="cover">
  <h1>Electronic Consciousness</h1>
  <div class="rule"></div>
  <div class="sub">A Speculative Manifesto on Minds and the Realities They Inhabit<br>
  Informed by Neuroscience, AI, Quantum Computing, and Mythology</div>
  <div class="author">Orlando Barrera II</div>
  <div class="sub" style="margin-top:0.4em">github.com/obarrera/Electronic-Consciousness</div>
  <div class="sub" style="margin-top:0.8em">Second Edition — revision 2.2 · August 2026</div>
</div>`;
const titlePlate = coverArt(['media/art/f01_TITLE_PAGE.png']);
const dedicationPlate = coverArt(['media/art/f03_DEDICATION_PAGE.png']);
if (titlePlate) body += `<div class="cover-art"><img src="${titlePlate}"></div>`;
else if (frontCoverArt) body += `<div class="cover" style="padding-top:3in">
  <h1>Electronic Consciousness</h1><div class="rule"></div>
  <div class="sub">A Speculative Manifesto on Minds and the Realities They Inhabit</div>
  <div class="author">Orlando Barrera II</div>
</div>`;
// Legal / licensing page (typeset; states the actual licensing model)
body += `<div class="frontmatter legal">
  <p><strong>Electronic Consciousness:<br>A Speculative Manifesto on Minds and the Realities They Inhabit</strong></p>
  <p>Cover strapline: <em>Mind, Machine, and the Nature of Awareness</em></p>
  <p>Second Edition — revision 2.2 · August 2026</p>
  <p class="gap">Copyright © 2026 Orlando Barrera II</p>
  <p>The book's source text and the companion software are published in the open repository
  github.com/obarrera/Electronic-Consciousness under the MIT License (see the repository's
  LICENSE file). Cover artwork and interior illustrations are by and © the author and are
  included by the author's permission. Quotations from third-party works remain the property
  of their respective rights holders and are cited in the References.</p>
  <p class="gap">The full bibliography, the machine-readable claim ledger (claims.yaml),
  the narrated video editions, and the EC-2D-Land simulation live in the same repository.</p>
</div>`;
if (dedicationPlate) body += `<div class="cover-art"><img src="${dedicationPlate}"></div>`;

// Abstract + epistemic note
body += `<div class="frontmatter"><h1>Abstract</h1>${marked.parse(cleanMarkdown(abstractFromReadme()))}
<div class="epistemic"><strong>Epistemic status.</strong> This work is a speculative philosophical
and interdisciplinary synthesis — design heuristics, thought experiments, and cross-disciplinary
metaphor — not an empirical or peer-reviewed scientific thesis. Section 1.4 states this framing
in full, and Section 16.2 subjects the framework to its strongest objections and identifies which
claims are falsifiable.</div></div>`;

// Table of contents
body += '<div class="toc"><h1>Contents</h1><ol>';
for (const fp of FRONT_PIECES) body += `<li class="part"><a href="#front-${FRONT_PIECES.indexOf(fp)}">${esc(fp.title)}</a></li>`;
let tocPart = 0;
for (const ch of chapters) {
  if (ch.major !== tocPart) {
    tocPart = ch.major;
    body += `<li class="part"><a href="#part-${tocPart}">Chapter ${tocPart} — ${esc(PARTS[tocPart] || '')}</a></li>`;
  }
  const title = ch.file.replace(/\.md$/, '');
  body += `<li class="sec"><a href="#sec-${ch.major}-${ch.minor}">${esc(title)}</a></li>`;
}
for (const bp of BACK_PIECES) body += `<li class="part"><a href="#back-${BACK_PIECES.indexOf(bp)}">${esc(bp.title)}</a></li>`;
body += `<li class="part"><a href="#references">References</a></li>`;
body += '</ol></div>';

// Front pieces
FRONT_PIECES.forEach((fp, i) => {
  const md = cleanMarkdown(fs.readFileSync(path.join(ROOT, fp.file), 'utf8'));
  body += `<div class="frontmatter" id="front-${i}">${marked.parse(md)}</div>`;
});

// Chapters grouped in parts
let currentPart = 0;
for (const ch of chapters) {
  if (ch.major !== currentPart) {
    currentPart = ch.major;
    const partArt = PART_ART[currentPart] && fs.existsSync(path.join(ROOT, PART_ART[currentPart]))
      ? `<img src="file://${path.join(ROOT, PART_ART[currentPart])}" style="max-width:80%; margin-top:0.9in">` : '';
    body += `<div class="part-banner${partArt ? ' has-art' : ''}" id="part-${currentPart}"><div class="part-no">Chapter ${currentPart}</div>
      <h1>${esc(PARTS[currentPart] || '')}</h1>${partArt}</div>`;
  }
  const md = cleanMarkdown(fs.readFileSync(path.join(ROOT, ch.file), 'utf8'));
  body += `<div class="chapter" id="sec-${ch.major}-${ch.minor}">${marked.parse(md)}</div>`;
}

// Back matter pieces: Conclusion, then mythic postludes
BACK_PIECES.forEach((bp, i) => {
  const md = cleanMarkdown(fs.readFileSync(path.join(ROOT, bp.file), 'utf8'));
  body += `<div class="chapter" id="back-${i}" style="page-break-before: always">${marked.parse(md)}</div>`;
});

// References
const refsPath = path.join(ROOT, 'References.md');
if (fs.existsSync(refsPath)) {
  const md = cleanMarkdown(fs.readFileSync(refsPath, 'utf8'));
  body += `<div class="part-banner" id="references"><div class="part-no">Back Matter</div><h1>References</h1></div>`;
  body += `<div class="chapter">${marked.parse(md)}</div>`;
}

// License / colophon
body += `<div class="frontmatter"><h1>Colophon</h1>
<p><strong>Second Edition — revision 2.2 (August 2026).</strong></p>
<p>Assembled from the living repository at github.com/obarrera/Electronic-Consciousness.
The companion simulation, EC-2D-Land, and the narrated video overviews are available in the
same repository. See the repository LICENSE for terms.</p></div>`;

if (backCoverArt) body += `<div class="cover-art" style="page-break-before: always"><img src="${backCoverArt}"></div>`;

const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><base href="file://${ROOT}/"><style>${css}</style></head>
<body>${body}</body></html>`;
const htmlPath = path.join(path.dirname(fileURLToPath(import.meta.url)), '.book.html');
fs.writeFileSync(htmlPath, html);

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage();
await page.goto('file://' + htmlPath, { waitUntil: 'networkidle', timeout: 120000 });
const pdfBase = {
  margin: { top: '0.85in', bottom: '0.85in', left: '0.7in', right: '0.7in' },
  width: '6.14in',
  height: '9.21in',
  printBackground: true,
};
const withHeaders = OUT_PDF.replace(/\.pdf$/, '.headers.pdf');
const noHeaders = OUT_PDF.replace(/\.pdf$/, '.noheaders.pdf');
await page.pdf({
  ...pdfBase,
  path: withHeaders,
  displayHeaderFooter: true,
  headerTemplate: `<div style="font-size:7.5pt;font-family:Georgia,serif;color:#666;width:100%;
    text-align:center;">Electronic Consciousness</div>`,
  footerTemplate: `<div style="font-size:8pt;font-family:Georgia,serif;color:#444;width:100%;
    text-align:center;"><span class="pageNumber"></span></div>`,
});
await page.pdf({ ...pdfBase, path: noHeaders, displayHeaderFooter: false });
await browser.close();
fs.unlinkSync(htmlPath);

// Post-process: splice header-free plates, set metadata, add bookmarks.
// Front plate pages: cover + title plate + legal page + dedication (each one page).
const frontPlates = 1 + (frontCoverArt ? 0 : 0) + (titlePlate ? 1 : 0) + 1 + (dedicationPlate ? 1 : 0);
const outlineItems = [];
for (const fp of FRONT_PIECES) outlineItems.push([1, fp.title]);
{
  let toPart = 0;
  for (const ch of chapters) {
    if (ch.major !== toPart) {
      toPart = ch.major;
      outlineItems.push([1, `Chapter ${toPart} — ${PARTS[toPart] || ''}`]);
    }
    outlineItems.push([2, ch.file.replace(/\.md$/, '')]);
  }
}
for (const bp of BACK_PIECES) outlineItems.push([1, bp.title]);
outlineItems.push([1, 'References']);
const { execFileSync } = await import('node:child_process');
execFileSync('python3', [
  path.join(path.dirname(fileURLToPath(import.meta.url)), 'postprocess.py'),
  withHeaders, noHeaders, OUT_PDF, String(frontPlates), JSON.stringify(outlineItems),
], { stdio: 'inherit' });
fs.unlinkSync(withHeaders);
fs.unlinkSync(noHeaders);
const stat = fs.statSync(OUT_PDF);
console.log(`Wrote ${OUT_PDF} (${(stat.size / 1024 / 1024).toFixed(1)} MB)`);
