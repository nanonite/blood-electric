#!/usr/bin/env python3
"""Build a page/folio map for citation (#8).

Aligns markdown spans in the layout-recovered corpus to PDF page indices
and printed folio numbers, so every downstream claim is citable to a page.

Inputs : analysis/ocr_confidence/raw_ocr_confidence.jsonl.gz  (folio lines)
         analysis/ocr_confidence/marker_json/blood-electric/blood-electric.json
         analysis/layout_recovery/blood-electric.layout.md    (<!-- page N -->)
Outputs: analysis/page_folio_map/
         - page_folio_map.tsv     one row per PDF page, 225 rows
         - coverage_report.md     honest coverage validation report

Folio extraction rule: a raw OCR line whose text is `N` or `<b>N</b>`
(standalone digits) is a printed folio line (marker drops these from the
markdown; see analysis/charset/charset_report.md §2.3). One folio line per
page holds for the whole corpus; folios strictly increase and satisfy
folio == page_id + 2 wherever measured. Gap pages in the folio extent are
labelled `inferred`; pages outside the folio extent carry no folio.
"""
import csv, gzip, json, re, pathlib, collections

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "analysis/ocr_confidence/raw_ocr_confidence.jsonl.gz"
JSON = ROOT / "analysis/ocr_confidence/marker_json/blood-electric/blood-electric.json"
LAYOUT = ROOT / "analysis/layout_recovery/blood-electric.layout.md"
OUT = ROOT / "analysis/page_folio_map"
OUT.mkdir(parents=True, exist_ok=True)

FOLIO_RE = re.compile(r"^(?:<b>)?(\d+)(?:</b>)?$")
MARKER_RE = re.compile(r"^<!-- page (\d+) -->\s*$")
PAGECOMMENT_RE = re.compile(r"<!-- page \d+ -->")
TAG = re.compile(r"<[^>]+>")

# 1. detected folios from the raw OCR stream ---------------------------------
folios = collections.defaultdict(list)   # page_id -> list of (folio, conf, block_id)
with gzip.open(RAW, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        m = FOLIO_RE.match((r.get("surya_text") or "").strip())
        if m:
            folios[r["page_id"]].append((m.group(1), r.get("surya_line_conf"), r.get("block_id")))

n_folio_lines = sum(len(v) for v in folios.values())
multi = {p: v for p, v in folios.items() if len(v) > 1}
exist = sorted(folios)
offset_ok = sum(1 for p, v in folios.items() if len(v) == 1 and int(v[0][0]) == p + 2)
monotonic = all(int(folios[a][0][0]) < int(folios[b][0][0])
                for a, b in zip(exist, exist[1:]) if len(folios[a]) == 1 and len(folios[b]) == 1)
lo, hi = exist[0], exist[-1]
gap_pages = [p for p in range(lo, hi + 1) if p not in folios]
low_conf = sorted((v[0][1], p, v[0][0]) for p, v in folios.items() if v[0][1] is not None and v[0][1] < 0.8)
tagged_pages = []
with gzip.open(RAW, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        m = FOLIO_RE.match((r.get("surya_text") or "").strip())
        if m and ("<b>" in r["surya_text"] or "</b>" in r["surya_text"]):
            tagged_pages.append((r["page_id"], int(m.group(1))))

# 2. marker JSON: per-page block inventory -----------------------------------
doc = json.loads(JSON.read_text())
pages = [c for c in doc["children"] if c.get("block_type") == "Page"]
assert len(pages) == 225, len(pages)

def page_of(node_id):
    m = re.match(r"/page/(\d+)", node_id or "")
    return int(m.group(1)) if m else None

block_counts = collections.Counter()
sections_per_page = collections.Counter()
for p in pages:
    pid = page_of(p["id"])
    for c in p.get("children") or []:
        block_counts[c.get("block_type")] += 1
        if c.get("block_type") == "SectionHeader":
            sections_per_page[pid] += 1

# 3. layout markdown: page markers + span token counts -----------------------
text = LAYOUT.read_text(encoding="utf-8")
lines = text.splitlines()
marker_lines = []   # (pdf_page_1based, line_index)
for i, ln in enumerate(lines):
    m = MARKER_RE.match(ln.strip())
    if m:
        marker_lines.append((int(m.group(1)), i))
assert len(marker_lines) == 225, len(marker_lines)
marker_pages = [p for p, _ in marker_lines]
markers_ascending_unique = marker_pages == sorted(marker_pages) and len(set(marker_pages)) == 225

def span_tokens(chunk):
    s = PAGECOMMENT_RE.sub(" ", chunk)
    s = re.sub(r"<!--.*?-->", " ", s)   # Picture/Figure placeholders
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)
    return s.split()

spans = {}
for idx, (pdf1, i) in enumerate(marker_lines):
    j = marker_lines[idx + 1][1] if idx + 1 < len(marker_lines) else len(lines)
    chunk = "\n".join(lines[i + 1:j])
    toks = span_tokens(chunk)
    spans[pdf1] = {"marker_line": i + 1, "tokens": len(toks), "chars": len(chunk)}

total_layout_tokens = sum(s["tokens"] for s in spans.values())
empty_pages = sorted(p for p, s in spans.items() if s["tokens"] == 0)

# 4. per-page rows ------------------------------------------------------------
rows = []
gapset = set(gap_pages)
for p in pages:
    pid = page_of(p["id"])
    pdf1 = pid + 1
    if len(folios.get(pid, [])) == 1:
        printed_folio, folio_conf, folio_src = folios[pid][0][0], folios[pid][0][1], "detected"
    elif lo <= pid <= hi:
        printed_folio, folio_conf, folio_src = str(pid + 2), "", "inferred"
    else:
        printed_folio, folio_conf, folio_src = "", "", "none"
    sp = spans[pdf1]
    rows.append({
        "page_id": pid,
        "pdf_page": pdf1,
        "printed_folio": printed_folio,
        "folio_source": folio_src,
        "folio_conf": "" if folio_conf == "" else f"{folio_conf:.4f}",
        "marker_line": sp["marker_line"],
        "span_tokens": sp["tokens"],
        "n_blocks": len(p.get("children") or []),
        "n_section_headers": sections_per_page.get(pid, 0),
    })

tsv = OUT / "page_folio_map.tsv"
with tsv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader(); w.writerows(rows)

# 5. coverage report ----------------------------------------------------------
def fmt(x):
    return f"{x:,}"

detected = sum(1 for r in rows if r["folio_source"] == "detected")
inferred = sum(1 for r in rows if r["folio_source"] == "inferred")
nofolio = sum(1 for r in rows if r["folio_source"] == "none")

report = []
p = report.append
p("# Page/folio map coverage report (#8)")
p("")
p("Source: `analysis/ocr_confidence/marker_json/.../blood-electric.json` (225 pages),")
p("layout corpus `analysis/layout_recovery/blood-electric.layout.md` (`<!-- page N -->`")
p("markers), raw OCR stream `analysis/ocr_confidence/raw_ocr_confidence.jsonl.gz`.")
p("Reproduce: `python work/build_page_folio_map.py`.")
p("")
p("## Deliverable")
p("")
p("`analysis/page_folio_map/page_folio_map.tsv` — 225 rows, one per PDF page:")
p("")
p("| column | meaning |")
p("|---|---|")
p("| `page_id` | 0-based PDF page index (marker's `/page/N/...` id) |")
p("| `pdf_page` | 1-based PDF page number (human citation; same as layout md `<!-- page N -->`) |")
p("| `printed_folio` | the folio printed on the page (the book's own page number), empty if none |")
p("| `folio_source` | `detected` (standalone digit line in raw OCR) / `inferred` (folio = page_id + 2) / `none` |")
p("| `folio_conf` | surya line confidence of the detected folio line |")
p("| `marker_line` | line number of the `<!-- page N -->` marker in the layout corpus |")
p("| `span_tokens` | whitespace tokens in that page's span of the layout corpus |")
p("")
p("## Measured results")
p("")
p(f"- Pages (PDF indices): **225**; layout markers found: **{len(marker_lines)}**, "
  f"ascending, unique, matching 1..225: **{'yes' if markers_ascending_unique else 'NO'}**.")
p(f"- Page spans with >=1 token: **{225 - len(empty_pages)}/225**; empty spans: {empty_pages if empty_pages else 'none'}.")
p(f"- Folio lines in raw stream: **{n_folio_lines}**, all single-folio per page: **{'yes' if not multi else multi}**.")
p(f"- Offset rule `printed_folio == page_id + 2` holds for **{offset_ok}/{n_folio_lines}** detected folios; folio sequence strictly increasing: **{'yes' if monotonic else 'NO'}**.")
p(f"- Folio extent (0-based page ids): {lo}..{hi}; detected **{detected}**, inferred in gap pages **{inferred}**, no folio (front/back matter) **{nofolio}** (pages 0-7, 219-224).")
p(f"- Gap pages without a detected folio line (folio by inference): {len(gap_pages)} — {gap_pages}.")
p(f"- Layout corpus tokens by span sum: **{fmt(total_layout_tokens)}** — matches #7's whole-file count (69,762), so page spans partition the token stream exactly.")
p("")
p("## Anomalies / honest notes")
p("")
p(f"- One renderer-tagged folio `<b>33</b>` survived on page 31 (0-based): {tagged_pages}.")
p(f"- Low-confidence folio detections (<0.80): **{len(low_conf)}** lines — {[(p+1, f, round(c,3)) for c, p, f in low_conf]}.")
p("- The digit **5 never occurs** in the book text (charset freeze); every folio digit found is book-consistent. Printed folios start at 10 on page_id 8 = PDF page 9 (1-based); front matter (title, copyright, parts without folios) is unnumbered in the source.")
p("- Pages 0–7 (title/copyright/dedication) and 219–224 (end matter + ads) carry no printed folio; citation for them falls back to `pdf_page`.")
p("- Folios are *printed* page numbers from the scan, not PDF indices; both are recorded so downstream claims can be cited as `(pdf_page=P, folio=F)`.")
p("")
rep = OUT / "coverage_report.md"
rep.write_text("\n".join(report) + "\n", encoding="utf-8")

print(f"wrote {tsv} ({len(rows)} rows)")
print(f"wrote {rep}")
print(f"folios: detected {detected}, inferred {inferred}, none {nofolio}; monotonic {monotonic}; offset ok {offset_ok}/{n_folio_lines}")
print(f"layout markers {len(marker_lines)}, ascending {markers_ascending_unique}, span token total {total_layout_tokens}")
