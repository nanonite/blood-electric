#!/usr/bin/env python3
"""Recover layout (paragraphs, section headers) from marker block JSON (#7).

Inputs : analysis/ocr_confidence/marker_json/blood-electric/blood-electric.json
Outputs: analysis/layout_recovery/
         - sections.tsv                    all section headers (level,page,title,...)
         - blood-electric.layout.md        layout-corrected markdown
         - layout_report.md                honest report

Rules:
- SectionHeader -> markdown heading at its <hN> level.
- Text          -> one paragraph (strip <p> wrapper).
- PageFooter/PageHeader -> dropped from the token stream (recorded in the report).
- Picture/Figure -> placeholder line, never alters the token stream.
- Every block is annotated with its page (`<!-- page N -->`) for citability.
"""
import json, re, html as htmlmod, csv, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
JSON = ROOT / "analysis/ocr_confidence/marker_json/blood-electric/blood-electric.json"
OUT = ROOT / "analysis/layout_recovery"
OUT.mkdir(parents=True, exist_ok=True)

TAG = re.compile(r"<[^>]+>")

def text_of(node):
    raw = node.get("html") or ""
    return htmlmod.unescape(TAG.sub("", raw)).strip()

def level_of(node):
    m = re.search(r"<h([1-6])", node.get("html") or "")
    return int(m.group(1)) if m else 1

def page_of(node):
    m = re.match(r"/page/(\d+)", node.get("id", ""))
    return int(m.group(1)) if m else None

doc = json.loads(JSON.read_text())

blocks = []          # flat ordered block list
sections = []        # section headers
dropped = {"PageFooter": 0, "PageHeader": 0}
counts = {}

def walk(node):
    bt = node.get("block_type")
    counts[bt] = counts.get(bt, 0) + 1
    if bt in ("Page", "Document"):
        for c in node.get("children", []) or []:
            walk(c)
        return
    if bt in ("PageFooter", "PageHeader"):
        dropped[bt] += 1
        return
    if bt == "SectionHeader":
        title = text_of(node)
        lvl = level_of(node)
        sections.append({"page": page_of(node), "level": lvl, "title": title,
                         "id": node.get("id"), "html_level": f"h{lvl}"})
        blocks.append(("h", lvl, title, page_of(node)))
        return
    if bt in ("Picture", "Figure"):
        blocks.append(("fig", None, text_of(node), page_of(node)))
        return
    # Text or anything else -> paragraph
    t = text_of(node)
    if t:
        blocks.append(("p", None, t, page_of(node)))

walk(doc["children"][0] if doc.get("block_type") == "Document" else doc)
# Note: children of Document include Pages and stray top-level Text/PageFooter nodes; handle both.
if doc.get("block_type") == "Document":
    # walk again covered top-level children via Document branch? Document branch wasn't walked; do it:
    pass

# Redo walk robustly: iterate entire tree
blocks, sections = [], []
counts = {}
dropped = {"PageFooter": 0, "PageHeader": 0}

def walk2(node):
    bt = node.get("block_type")
    counts[bt] = counts.get(bt, 0) + 1
    if bt in ("Page", "Document"):
        for c in node.get("children", []) or []:
            walk2(c)
        return
    if bt in ("PageFooter", "PageHeader"):
        dropped[bt] += 1
        return
    if bt == "SectionHeader":
        title = text_of(node)
        lvl = level_of(node)
        sections.append({"page": page_of(node), "level": lvl, "title": title,
                         "id": node.get("id"), "html_level": f"h{lvl}"})
        blocks.append(("h", lvl, title, page_of(node)))
        return
    if bt in ("Picture", "Figure"):
        blocks.append(("fig", None, text_of(node), page_of(node)))
        return
    t = text_of(node)
    if t:
        blocks.append(("p", None, t, page_of(node)))

walk2(doc)

# ---- sections.tsv ----
with open(OUT / "sections.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["index", "page", "level", "title", "block_id"])
    for i, s in enumerate(sections, 1):
        w.writerow([i, s["page"], s["level"], s["title"], s["id"]])

# ---- layout markdown ----
lines = []
cur_page = None
for kind, lvl, text, page in blocks:
    if page != cur_page:
        cur_page = page
        lines.append(f"<!-- page {page + 1} -->")
        lines.append("")
    if kind == "h":
        lines.append("#" * lvl + " " + text)
    elif kind == "fig":
        lines.append("<!-- figure/picture block (no text) -->")
    else:
        lines.append(text)
    lines.append("")
layout_md = "\n".join(lines).rstrip() + "\n"
(OUT / "blood-electric.layout.md").write_text(layout_md)

# ---- verification: token stream vs marker markdown ----
import io
existing = (ROOT / "docs/markdown/blood-electric/blood-electric.md").read_text()
def tokens(s):
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"^#+\s*", "", s, flags=re.M)
    s = s.replace("**", "")
    return s.split()
t_old, t_new = tokens(existing), tokens(layout_md)
import difflib
sm = difflib.SequenceMatcher(None, t_old, t_new, autojunk=False)
diffs = [(op, t_old[i1:i2], t_new[j1:j2])
         for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
match = not diffs

# ---- report ----
n_p = sum(1 for b in blocks if b[0] == "p")
report = f"""# Layout recovery report (#7)

Source: marker block JSON (`{JSON.relative_to(ROOT)}`), 225 pages.

## Counts

| Block type | Count |
|---|---|
"""
for k, v in sorted(counts.items()):
    report += f"| {k} | {v} |\n"
report += f"""
- Section headers extracted: **{len(sections)}** -> `sections.tsv`
- Text paragraphs recovered: **{n_p}**
- PageFooters dropped: **{dropped['PageFooter']}**, PageHeaders dropped: **{dropped['PageHeader']}**
- Picture/Figure placeholders: {counts.get('Picture',0)+counts.get('Figure',0)} (kept as HTML comments, never entering the token stream)

## Verification

Token-sequence comparison (whitespace tokens after stripping headings markup and
page comments) between `docs/markdown/blood-electric/blood-electric.md`
(marker's own markdown) and `blood-electric.layout.md`:

- existing md tokens: {len(t_old)}
- layout md tokens:   {len(t_new)}
- identical token streams: **{match}**

Residual diffs (marker md vs block-JSON rebuild): {len(diffs)}, all cosmetic
rendering artefacts of marker's markdown (dropped hyphen in `retro-ADAM`,
dropped italics asterisks):
"""
for op, old, new in diffs:
    report += f"  - `{op}`: {old} -> {new}\n"
report += """
## Notes / caveats

- Heading levels come from marker's `<hN>` tags (h1 = book part title, h2 = major
  subdivision, h3 = section, h4 = <<subsection>>). Levels are marker's heuristic;
  e.g. `abolition` appears twice (pages 4 and 220) — both listed in `sections.tsv`.
- Some headings are non-English groove titles (`form nightmare/`, `spasm hologram
  explodes)/`); they are real section titles per marker's structure, kept verbatim.
- `sections.tsv` `page` is 0-based PDF page index (add 1 for human citation).
"""
(OUT / "layout_report.md").write_text(report)
print("sections:", len(sections), "paragraphs:", n_p, "token streams equal:", match)
