#!/usr/bin/env python3
"""Book PDF post-processor.

Splices header-free front plates (cover, title, legal, dedication) and the
back cover from the no-headers render into the with-headers body render,
sets document metadata, and builds a hierarchical outline (bookmarks) from
the table of contents' internal link annotations.

Usage: postprocess.py with_headers.pdf no_headers.pdf out.pdf N_FRONT_PLATES TITLES_JSON
"""
import json
import sys

from pypdf import PdfReader, PdfWriter
from pypdf.generic import IndirectObject

with_headers, no_headers, out_path, n_front, titles_json = sys.argv[1:6]
n_front = int(n_front)
titles = json.loads(titles_json)

hdr = PdfReader(with_headers)
plain = PdfReader(no_headers)
n_pages = len(hdr.pages)
assert len(plain.pages) == n_pages, "renders differ in pagination"

writer = PdfWriter()
for i in range(n_pages):
    src = plain if (i < n_front or i == n_pages - 1) else hdr
    writer.add_page(src.pages[i])

writer.add_metadata({
    "/Title": "Electronic Consciousness: A Speculative Manifesto on Minds and the Realities They Inhabit",
    "/Author": "Orlando Barrera II",
    "/Subject": "Artificial intelligence, consciousness, philosophy of mind, and speculative research",
    "/Keywords": ("electronic consciousness, AI consciousness, philosophy of mind, "
                   "global workspace, integrated information, sacred geometry, "
                   "machine consciousness, speculative philosophy"),
    "/Producer": "build_book.mjs (Chromium print) + pypdf postprocess",
})

# ---- Outline from the TOC's link annotations -------------------------------
# The TOC is the only place with this many internal GoTo links in sequence;
# annotation order follows content order, matching the titles list.

def _page_index_by_id(reader, target_id):
    for idx, page in enumerate(reader.pages):
        if page.indirect_reference.idnum == target_id:
            return idx
    return None


def dest_page_index(reader, annot, named):
    obj = annot.get_object()
    dest = None
    if "/Dest" in obj:
        dest = obj["/Dest"]
    elif "/A" in obj:
        action = obj["/A"].get_object()
        if action.get("/S") == "/GoTo":
            dest = action.get("/D")
    if dest is None:
        return None
    dest = dest.get_object() if hasattr(dest, "get_object") else dest
    if isinstance(dest, (str, bytes)):  # named destination (Chromium)
        d = named.get(str(dest))
        if d is not None and d.page is not None:
            return _page_index_by_id(reader, d.page.idnum)
        return None
    target = dest[0] if isinstance(dest, list) else dest
    if isinstance(target, IndirectObject):
        return _page_index_by_id(reader, target.idnum)
    return None

# Collect internal-link destinations in order, from the first 12 pages
named = hdr.named_destinations
dests = []
for pidx in range(min(12, n_pages)):
    page = hdr.pages[pidx]
    annots = page.get("/Annots")
    for annot in (annots.get_object() if annots is not None else []) or []:
        d = dest_page_index(hdr, annot, named)
        if d is not None:
            dests.append(d)

# Re-register named destinations (add_page does not copy the catalog name
# tree, which would leave every internal link in the book dead).
readded = 0
for name, d in (hdr.named_destinations or {}).items():
    if d.page is None:
        continue
    idx = _page_index_by_id(hdr, d.page.idnum)
    if idx is not None:
        writer.add_named_destination(name.lstrip("/"), idx)
        readded += 1
print(f"named destinations restored: {readded}")

if len(dests) >= len(titles):
    parent = None
    for (level, title), page_idx in zip(titles, dests[: len(titles)]):
        if level == 1:
            parent = writer.add_outline_item(title, page_idx)
        else:
            writer.add_outline_item(title, page_idx, parent=parent)
    print(f"outline: {len(titles)} bookmarks added (nested)")
else:
    print(f"outline: skipped (found {len(dests)} TOC links, expected {len(titles)})")

with open(out_path, "wb") as fh:
    writer.write(fh)

# ---- Stamp TOC page numbers (Chromium lacks target-counter support) --------
import fitz  # pymupdf

doc = fitz.open(out_path)
names = doc.resolve_names() if hasattr(doc, "resolve_names") else {}
def link_page(l):
    if l.get("page", -1) >= 0:
        return l["page"]
    name = l.get("nameddest") or l.get("name")
    for key in (name, f"/{name}") if name else ():
        if key in names:
            return names[key].get("page", -1)
    return -1
stamped = 0
for pidx in range(min(12, len(doc))):
    page = doc[pidx]
    links = [dict(l, page=link_page(l)) for l in page.get_links()]
    links = [l for l in links if l["page"] >= 0]
    if not links:
        continue
    right_edge = page.rect.width - 0.7 * 72  # inside the right margin
    for l in links:
        num = str(l["page"] + 1)
        y = l["from"].y1 - 1.5
        w = fitz.get_text_length(num, fontname="tiro", fontsize=10)
        page.insert_text((right_edge - w, y), num, fontname="tiro", fontsize=10,
                         color=(0.1, 0.1, 0.1))
        stamped += 1
doc.saveIncr()
print(f"toc: {stamped} page numbers stamped")
print(f"postprocess complete: {n_pages} pages, {n_front} front plates + back cover header-free")
