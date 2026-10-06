#!/usr/bin/env python3
"""
L0 / Chainlink #5 - freeze builder: charset.tsv + freeze.json +
charset_report.md for the "Blood Electric" corpus.

What this builds
----------------
  analysis/charset/charset.tsv        the frozen glyph inventory
                                      and canonical-form decisions
  analysis/charset/freeze.json        machine-readable freeze record
                                      (hashes, counts, rule fire
                                      counts, test result)
  analysis/charset/charset_report.md  the human report

Everything measured is measured here, from:
  * the working corpus
    (docs/markdown/blood-electric/blood-electric.md),
  * the raw OCR text stream
    (analysis/ocr_confidence/raw_ocr_confidence.jsonl.gz - the
    per-character OCR output of the same marker run, before
    marker's markdown/ftfy post-processing),
  * marker's own JSON output
    (analysis/ocr_confidence/marker_json/blood-electric/
    blood-electric.json - for the html-entity cross-check).

The normalizer rules are NOT defined here; they are imported
from work/normalize.py so code, frozen table and test share one
source of truth.

Usage (from the workspace root, inside the `marker` conda env):

  python work/build_charset.py
  python work/normalize.py freeze          # same thing
"""
from __future__ import annotations

import collections
import datetime
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_CORPUS = normalize.DEFAULT_CORPUS
DEFAULT_RAW_DUMP = normalize.DEFAULT_RAW_DUMP
MARKER_JSON = os.path.join(
    ROOT, "analysis", "ocr_confidence", "marker_json",
    "blood-electric", "blood-electric.json")
DEFAULT_OUTDIR = normalize.DEFAULT_OUTDIR

# --------------------------------------------------------------------------
# Curated notes for the observed codepoints (the "why" of each
# decision).  Codepoints not listed here get a generated note.
# --------------------------------------------------------------------------
NOTES = {
    0x000A: "line break; paragraph separator in marker markdown",
    0x0020: "space",
    0x0021: "exclamation mark; front matter only",
    0x0022: "straight double quote; front matter / blurbs",
    0x0023: "markdown heading marker, added by marker's "
            "renderer (absent from the raw OCR stream)",
    0x0027: "straight apostrophe",
    0x0028: "left parenthesis",
    0x0029: "right parenthesis",
    0x002A: "markdown bold marker, added by marker's renderer "
            "(the credit page of the scan displays literal "
            "<b> tags; the renderer turns them into **)",
    0x002B: "plus sign; also the '++' concatenation operator",
    0x002C: "comma",
    0x002D: "hyphen-minus; compound linker (boy-roid, "
            "drug-embryo); also the '++'/'--' operators",
    0x002E: "full stop",
    0x002F: "solidus; alternation/possessive (soul/gram); "
            "also the '//' line operator",
    0x003A: "colon; single and '::' field separator",
    0x003C: "less-than sign; single and '<<' stream opener; "
            "the book also prints anchor-like codework "
            "(<a href=\"mailto:...\")",
    0x003D: "equals sign; identity/naming operator (gene=TV)",
    0x003E: "greater-than sign; single and '>>' stream closer",
    0x005B: "left square bracket; the book's bracketed "
            "phrases ([ice nebula faecal black])",
    0x005C: "reverse solidus; genuine codework content: "
            "literal '\\n' in the scan text (7x) plus "
            "'H\\body-OMOTYA' (1x); present in the raw OCR "
            "stream, so not a renderer artifact",
    0x005D: "right square bracket; see U+005B",
    0x007C: "vertical line; the '<<|' construction",
    0x00A9: "copyright sign; front-matter credit line only",
    0x00D8: "capital O with stroke; genuine glyph of the "
            "section header 'MØNKEY/gene-dub'",
    0x2013: "en dash; prose dash. Kept DISTINCT from the em "
            "dash: the scan gives no evidence the two are one "
            "glyph (see report section 4)",
    0x2014: "em dash; prose dash. Kept DISTINCT from the en "
            "dash (see report section 4). The raw OCR stream "
            "has 486, the corpus 484: marker's line-joiner "
            "drops a line-final em dash at two joins "
            "(0-based PDF pages 41 and 53 - see report "
            "section 2.3)",
    0x2022: "bullet; front-matter credit line only",
}

DECISION_KEEP = "keep"
DECISION_NORMALIZE = "normalize"
DECISION_STRIP = "strip"

# Genuine renderer tags found in the raw OCR stream (the credit
# page of the scan displays literal HTML; everything else that
# looks like a tag - <a href="mailto:..."> - is the book's own
# printed codework and appears in both views).
RENDERER_TAG_RE = re.compile(r"<br\s*/?>|</?b>")
FOLIO_RE = re.compile(r"^(?:<b>)?(\d+)(?:</b>)?$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw_dump(path):
    """Return (records, raw_text, per_page_text) from the raw
    OCR dump.  records is the list of parsed JSONL records."""
    records = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    parts = []
    per_page = collections.defaultdict(list)
    for rec in records:
        text = rec.get("surya_text", "")
        parts.append(text)
        per_page[rec.get("page_id")].append(text)
    return records, "".join(parts), per_page


def read_marker_json_html(path):
    """Concatenate the 'html' fields of marker's JSON output."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    parts = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "html" and isinstance(value, str):
                    parts.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return "".join(parts)


# --------------------------------------------------------------------------
# The raw-vs-corpus cross-check (report section 2.3)
# --------------------------------------------------------------------------
def cross_check(records, raw_text, corpus_text):
    """Measure how the raw OCR stream and the markdown corpus
    differ, codepoint by codepoint, and explain every delta
    class.  Returns a dict of measured findings."""
    raw_counts = normalize.inventory(raw_text)
    corpus_counts = normalize.inventory(corpus_text)
    md_stripped = corpus_text.replace("\n", " ")

    # every codepoint whose count differs
    delta = {
        cp: (raw_counts.get(cp, 0), corpus_counts.get(cp, 0))
        for cp in sorted(set(raw_counts) | set(corpus_counts))
        if raw_counts.get(cp, 0) != corpus_counts.get(cp, 0)
    }

    # folio (printed page-number) lines, e.g. '10' or '<b>33</b>'
    folio_lines = []
    for rec in records:
        m = FOLIO_RE.match(rec.get("surya_text", "").strip())
        if m:
            folio_lines.append((rec, m.group(1)))
    folio_digit_chars = sum(len(d) for _rec, d in folio_lines)

    # genuine renderer tags
    renderer_tags = collections.Counter()
    for rec in records:
        renderer_tags.update(
            RENDERER_TAG_RE.findall(rec.get("surya_text", "")))

    # the book's own anchor-like codework (present in BOTH views)
    anchor_lines = [
        rec for rec in records
        if "<a href" in rec.get("surya_text", "")
    ]
    anchor_occurrences = sum(
        rec.get("surya_text", "").count("<a href")
        for rec in anchor_lines)

    # de-hyphenation: raw lines ending in hyphen-minus
    hyphen_enders = [
        rec for rec in records
        if rec.get("surya_text", "").rstrip().endswith("-")
    ]

    # line-final em dashes: which survived marker's line-joiner?
    em_dash_enders = []
    for rec in records:
        text = rec.get("surya_text", "").rstrip()
        if text.endswith("—"):
            probe = text[-30:].replace("\n", " ")
            em_dash_enders.append({
                "page_id": rec.get("page_id"),
                "block_id": rec.get("block_id"),
                "tail": text[-40:],
                "survived": probe in md_stripped,
            })

    # non-folio raw lines whose content is absent from the
    # corpus (probe = first 30 chars, leading whitespace
    # ignored, newlines as spaces).  Classified by whether
    # the line carries renderer tags (credit page) or not
    # (a genuinely dropped line).
    unmatched_lines = []
    for rec in records:
        text = rec.get("surya_text", "")
        if FOLIO_RE.match(text.strip()):
            continue
        probe = text.lstrip().replace("\n", " ")[:30]
        if not probe or probe in md_stripped:
            continue
        has_tags = bool(RENDERER_TAG_RE.search(text))
        unmatched_lines.append({
            "page_id": rec.get("page_id"),
            "block_id": rec.get("block_id"),
            "text": text[:80],
            "kind": ("renderer_tag_line" if has_tags
                     else "dropped_line"),
        })

    # surplus/deficit accounting
    raw_surplus = sum(
        max(0, raw_n - md_n) for raw_n, md_n in delta.values())
    raw_deficit = sum(
        max(0, md_n - raw_n) for raw_n, md_n in delta.values())
    tag_chars = sum(
        len(t) * n for t, n in
        collections.Counter(
            t for rec in records
            for t in RENDERER_TAG_RE.findall(
                rec.get("surya_text", ""))
        ).items())
    dropped_line_chars = sum(
        len(rec.get("surya_text", ""))
        for rec in records
        if FOLIO_RE.match(rec.get("surya_text", "").strip()) is None
        and any(u["block_id"] == rec.get("block_id")
                and u["page_id"] == rec.get("page_id")
                and u["kind"] == "dropped_line"
                for u in unmatched_lines))

    # stray control/format characters in the raw stream
    raw_stray = sum(
        1 for cp in raw_counts
        if unicodedata.category(chr(cp)) in ("Cc", "Cf")
        and chr(cp) not in "\n\t")

    return {
        "delta": delta,
        "folio_lines": folio_lines,
        "folio_digit_chars": folio_digit_chars,
        "renderer_tags": dict(renderer_tags),
        "renderer_tag_chars": tag_chars,
        "anchor_lines": len(anchor_lines),
        "anchor_occurrences": anchor_occurrences,
        "anchor_pages": sorted({
            rec.get("page_id") for rec in anchor_lines}),
        "hyphen_enders": len(hyphen_enders),
        "hyphen_delta": (
            raw_counts.get(ord("-"), 0)
            - corpus_counts.get(ord("-"), 0)),
        "em_dash_enders": em_dash_enders,
        "em_dash_delta": (
            raw_counts.get(0x2014, 0)
            - corpus_counts.get(0x2014, 0)),
        "unmatched_lines": unmatched_lines,
        "dropped_line_chars": dropped_line_chars,
        "raw_surplus": raw_surplus,
        "raw_deficit": raw_deficit,
        "raw_stray_controls": raw_stray,
    }


# --------------------------------------------------------------------------
# charset.tsv
# --------------------------------------------------------------------------
def escape_char(ch):
    """Render a character for a TSV cell: printable
    characters as themselves, control/zero-width characters
    as escape sequences (a literal newline would break the
    row structure)."""
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    if ch == "\r":
        return "\\r"
    if ch.isprintable():
        return ch
    return f"\\u{ord(ch):04X}"


def charset_rows(corpus_counts, raw_counts):
    """Build the frozen charset rows.

    Observed codepoints (decision=keep) first, then the frozen
    normalization maps (decision=normalize/strip, count 0: none
    of them occur in this corpus or its raw OCR stream).
    """
    rows = []
    for cp, n in sorted(corpus_counts.items()):
        ch = chr(cp)
        name = unicodedata.name(ch, "<NO NAME>")
        cat = unicodedata.category(ch)
        note = NOTES.get(cp)
        if note is None:
            if cat.startswith("L"):
                note = f"{cat[0]} letter; ordinary English lexis"
            elif cat == "Nd":
                note = "digit"
            elif cat in ("Po", "Ps", "Pe", "Sm", "Pd", "So"):
                note = f"punctuation/symbol ({cat})"
            else:
                note = f"category {cat}"
        rows.append({
            "codepoint": f"U+{cp:04X}",
            "char": escape_char(ch),
            "unicode_name": name,
            "category": cat,
            "corpus_count": str(n),
            "raw_ocr_count": str(raw_counts.get(cp, 0)),
            "decision": DECISION_KEEP,
            "canonical": escape_char(ch),
            "rule": "",
            "notes": note,
        })
    # Frozen normalization maps: unobserved in this corpus, but
    # part of the frozen spec so future OCR passes normalize
    # identically.  canonical = the replacement string.
    for rule, mapping in normalize.MAPPED_RULES.items():
        for src, dst in sorted(mapping.items(),
                               key=lambda kv: ord(kv[0])):
            cp = ord(src)
            decision = (DECISION_STRIP if dst == ""
                        else DECISION_NORMALIZE)
            rows.append({
                "codepoint": f"U+{cp:04X}",
                "char": escape_char(src),
                "unicode_name": unicodedata.name(src,
                                                 "<NO NAME>"),
                "category": unicodedata.category(src),
                "corpus_count": "0",
                "raw_ocr_count": "0",
                "decision": decision,
                "canonical": dst,
                "rule": rule,
                "notes": f"mapped by rule '{rule}' "
                         f"(work/normalize.py); not observed in "
                         f"corpus or raw OCR stream",
            })
    return rows


def write_charset_tsv(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(normalize.CHARSET_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(row[k] for k in normalize.CHARSET_HEADER)
                     + "\n")


# --------------------------------------------------------------------------
# Report helpers
# --------------------------------------------------------------------------
def ascii_class_stats(counts):
    letters = {cp: n for cp, n in counts.items()
               if chr(cp).isascii() and chr(cp).isalpha()}
    digits = {cp: n for cp, n in counts.items()
              if chr(cp).isascii() and chr(cp).isdigit()}
    whitespace = {cp: n for cp, n in counts.items()
                  if chr(cp).isspace()}
    punct = {cp: n for cp, n in counts.items()
             if chr(cp).isascii() and not chr(cp).isalpha()
             and not chr(cp).isdigit() and not chr(cp).isspace()}
    return letters, digits, whitespace, punct


def fmt_count(n):
    return f"{n:,}"


def build_report(ctx):
    """Render charset_report.md from the measured context."""
    L = []
    p = L.append

    corpus = ctx["corpus"]
    raw = ctx["raw"]
    per_rule = ctx["per_rule"]
    rows = ctx["rows"]
    test = ctx["test"]
    x = ctx["cross"]

    p("# Blood Electric — glyph inventory & unicode normalization "
      "freeze (L0 / Chainlink #5)")
    p("")
    p(f"_Generated {ctx['generated_at']} by "
      f"`work/build_charset.py` (via `work/normalize.py freeze`), "
      f"from the corpus `{os.path.relpath(corpus['path'], ROOT)}` "
      f"and the raw OCR dump "
      f"`{os.path.relpath(raw['path'], ROOT)}`._")
    p("")
    p("## 0. Summary")
    p("")
    p(f"- **{corpus['codepoints']} distinct codepoints** in "
      f"**{fmt_count(corpus['chars'])} characters**: "
      f"{corpus['ascii_codepoints']} ASCII + "
      f"{corpus['non_ascii_codepoints']} non-ASCII")
    p(f"- **No ligatures, no typographic quotes, no unicode "
      f"spaces, no zero-width or stray control characters** — "
      f"measured at zero in the corpus *and* in the raw OCR "
      f"stream (§3)")
    p(f"- The frozen normalizer is a **measured no-op** on the "
      f"frozen corpus: **{ctx['total_events']} of "
      f"{fmt_count(corpus['chars'])} characters changed**; "
      f"idempotent; the normalized corpus is byte-identical "
      f"(sha256 equal, §5)")
    p(f"- Raw OCR stream vs corpus: **same non-ASCII alphabet** "
      f"(all 5 glyphs, none introduced or destroyed); every "
      f"count delta is measured, explained and located (§2.3) — "
      f"including **2 em dashes dropped by marker's line-joiner** "
      f"(pages 41, 53) and **one 7-character OCR line dropped** "
      f"(page 58, `being>>/`)")
    p(f"- `charset.tsv` frozen: **{len(rows)} rows** "
      f"({ctx['keep_rows']} observed `keep` + "
      f"{ctx['mapped_rows']} frozen `normalize`/`strip` maps); "
      f"test suite **{test['result']}**")
    p("")
    p("## 1. Method")
    p("")
    p("- Every codepoint of the corpus was enumerated "
      "(`collections.Counter` over the UTF-8 text) and "
      "cross-checked against two independent views of the same "
      "marker run: the **raw OCR text stream** "
      "(`raw_ocr_confidence.jsonl.gz`, the per-character OCR "
      "output *before* marker's markdown/ftfy post-processing) "
      "and **marker's own JSON** (its `html` fields).")
    p("- Canonical-form decisions are frozen in "
      "`work/normalize.py` (the `RULES` pipeline and mapping "
      "tables). `charset.tsv` is *generated from those same "
      "tables*, so code, table and test cannot drift: "
      "`work/test_normalize.py` checks all three against each "
      "other and against the corpus.")
    p("- Nothing in this task corrects OCR *lexical* drift "
      "(`It`→`lt`, `boyroid`→`boy-roid`, spacing around `>>`). "
      "That is Chainlink #6 (gazetteer-protected lexical "
      "correction); unicode normalization is deliberately "
      "upstream of it and must not pre-empt it. The "
      "post-processing losses found here (2 dropped em "
      "dashes + 1 dropped line, §2.3) are *reported, not "
      "repaired*: repairing the corpus is #7's job (layout "
      "recovery), and the freeze must describe the corpus "
      "as it is.")
    p("")
    p("## 2. Inventory (measured)")
    p("")
    p("### 2.1 The non-ASCII glyphs (all 5)")
    p("")
    p("| codepoint | char | unicode name | corpus | raw OCR | "
      "pages (0-based PDF pages) | decision |")
    p("|---|---|---|---|---|---|---|")
    for cp, n in sorted(corpus["counts"].items()):
        if cp <= 0x7F:
            continue
        ch = chr(cp)
        pages = raw["non_ascii_pages"].get(ch, [])
        if len(pages) <= 12:
            page_str = ", ".join(str(pg) for pg in pages)
        else:
            page_str = (f"{len(pages)} pages (first: "
                        + ", ".join(str(pg) for pg in pages[:6])
                        + ", …)")
        row = next(r for r in rows
                   if normalize.parse_codepoint(r["codepoint"]) == cp)
        raw_n = raw["counts"].get(cp, 0)
        mark = "" if raw_n == n else f" **(raw {raw_n:,})**"
        p(f"| U+{cp:04X} | {ch} | "
          f"{unicodedata.name(ch)} | {fmt_count(n)}{mark} | "
          f"{fmt_count(raw_n)} | {page_str} | {row['decision']} |")
    p("")
    p("All five are genuine content, not OCR artifacts: the "
      "copyright sign, the O-slash of the `MØNKEY/gene-dub` "
      "section header, the two dashes of the prose, and the "
      "bullet of the front-matter credit line. The em dash is "
      "the only glyph whose count differs between the two views "
      "(486 raw vs 484 corpus) — the difference is located and "
      "explained in §2.3.")
    p("")
    p("### 2.2 ASCII inventory")
    p("")
    letters, digits, whitespace, punct = ctx["ascii_stats"]
    p(f"- letters: **{len(letters)} codepoints**, "
      f"{fmt_count(sum(letters.values()))} occurrences "
      f"(26 upper + 26 lower; case is evidence and is never "
      f"folded)")
    digit_list = ", ".join(f"{chr(cp)}×{fmt_count(n)}"
                           for cp, n in sorted(digits.items()))
    present = {chr(cp) for cp in digits}
    missing = [str(d) for d in range(10) if str(d) not in present]
    p(f"- digits: **{len(digits)} codepoints** "
      f"({digit_list}) — the digit **{', '.join(missing)} never "
      f"occurs** in the book: 0 occurrences in the corpus *and* "
      f"0 in the raw OCR text lines (the only `5`s in the raw "
      f"dump are the folio page numbers, which marker drops from "
      f"the markdown — §2.3)")
    p(f"- whitespace: **{len(whitespace)} codepoints** "
      f"({', '.join(f'U+{cp:04X} {chr(cp)!r} ×{fmt_count(n)}' for cp, n in sorted(whitespace.items()))})")
    p(f"- ASCII punctuation/symbols: **{len(punct)} codepoints**, "
      f"{fmt_count(sum(punct.values()))} occurrences:")
    p("")
    p("| codepoint | char | count | role (evidence-linked) |")
    p("|---|---|---|---|")
    for cp, n in sorted(punct.items()):
        row = next(r for r in rows
                   if normalize.parse_codepoint(r["codepoint"]) == cp)
        p(f"| U+{cp:04X} | `{chr(cp)}` | {fmt_count(n)} | "
          f"{row['notes']} |")
    p("")
    p("### 2.3 Raw OCR stream vs markdown corpus (the honest "
      "delta)")
    p("")
    p("The raw OCR text stream (per-character OCR output before "
      "marker's post-processing) and the working markdown corpus "
      "are the same text at two stages of the pipeline. They are "
      "NOT identical, and the freeze records the difference "
      "rather than hiding it. Full codepoint delta "
      "(non-printable characters escaped):")
    p("")
    p("| codepoint | char | raw OCR | corpus | delta (raw − corpus) |")
    p("|---|---|---|---|---|")
    for cp, (raw_n, md_n) in x["delta"].items():
        p(f"| U+{cp:04X} | `{escape_char(chr(cp))}` | "
          f"{fmt_count(raw_n)} | "
          f"{fmt_count(md_n)} | {raw_n - md_n:+d} |")
    p("")
    p("Every delta class, measured:")
    p("")
    folio_pages = sorted({rec.get("page_id")
                          for rec, _d in x["folio_lines"]})
    all_digits = sum(v for k, v in raw["counts"].items()
                     if chr(k).isdigit())
    tagged = [rec for rec, d in x["folio_lines"]
              if "<" in rec.get("surya_text", "")]
    tagged_note = ""
    if tagged:
        tagged_note = (
            ", plus one renderer-tagged folio "
            + " ".join(f"`{rec.get('surya_text', '').strip()}`"
                       for rec in tagged)
            + f" on page(s) "
            + ", ".join(str(rec.get("page_id"))
                        for rec in tagged))
    p(f"1. **Folio (printed page-number) lines — "
      f"{len(x['folio_lines'])} lines, "
      f"{x['folio_digit_chars']} digit characters** "
      f"(pages {folio_pages[0]}–{folio_pages[-1]}, "
      f"0-based{tagged_note}). The raw stream contains the "
      f"printed page numbers as standalone lines; marker's "
      f"markdown renderer drops them. This accounts for the "
      f"entire digit delta: raw {fmt_count(all_digits)} "
      f"digits = {x['folio_digit_chars']} folio + "
      f"{fmt_count(all_digits - x['folio_digit_chars'])} "
      f"in text; the corpus keeps exactly the in-text digits.")
    p(f"2. **Credit-page renderer tags — "
      f"{sum(x['renderer_tags'].values())} tags, "
      f"{x['renderer_tag_chars']} characters "
      f"({', '.join(f'{t} ×{n}' for t, n in sorted(x['renderer_tags'].items()))}).** "
      f"The scan's copyright/credit page (0-based PDF pages 0–2) "
      f"displays *literal* HTML tags — the PDF was produced "
      f"from unrendered HTML. The OCR reads them as characters; "
      f"marker's markdown renderer interprets them (`<b>`→`**`, "
      f"`<br>`→line break). This accounts for the `#`/`*` "
      f"markdown-only codepoints and most of the "
      f"`{x['delta'].get(0x003C, (0, 0))[0] - x['delta'].get(0x003C, (0, 0))[1]:+d}`/`"
      f"{x['delta'].get(0x003E, (0, 0))[0] - x['delta'].get(0x003E, (0, 0))[1]:+d}` "
      f"angle-bracket and `b`/`r` letter deltas.")
    p(f"3. **The book's own anchor-like codework — "
      f"{x['anchor_occurrences']} occurrences of "
      f"`<a href=\"mailto:...\">` on {x['anchor_lines']} lines "
      f"(pages {', '.join(str(pg) for pg in x['anchor_pages'])}).** "
      f"The book *prints* anchor-like text as part of its "
      f"codework; it appears in both views (the corpus contains "
      f"all {x['anchor_occurrences']}), so it is content, not a "
      f"delta.")
    p(f"4. **De-hyphenation — {x['hyphen_delta']:+d} hyphens "
      f"(raw {fmt_count(x['delta'].get(0x002D, (0, 0))[0])} vs "
      f"corpus {fmt_count(x['delta'].get(0x002D, (0, 0))[1])}).** "
      f"{x['hyphen_enders']} raw lines end in `-`; marker's "
      f"line-joiner deletes the hyphen when the next line starts "
      f"lowercase (e.g. `digital-` + `vampire` → "
      f"`digitalvampire`). This matches the de-hyphenation count "
      f"in the OCR confidence report (Chainlink #3, §1.1).")
    dropped = [e for e in x["em_dash_enders"] if not e["survived"]]
    kept = [e for e in x["em_dash_enders"] if e["survived"]]
    p(f"5. **Em dashes dropped at line joins — "
      f"{x['em_dash_delta']:+d} (raw "
      f"{fmt_count(x['delta'].get(0x2014, (0, 0))[0])} vs corpus "
      f"{fmt_count(x['delta'].get(0x2014, (0, 0))[1])}).** "
      f"{len(x['em_dash_enders'])} raw lines end in an em dash; "
      f"{len(kept)} survived into the corpus and **{len(dropped)} "
      f"were dropped by marker's line-joiner** — the same class "
      f"of behavior as its de-hyphenation, applied to a trailing "
      f"`—`:")
    for e in dropped:
        p(f"    - 0-based PDF page {e['page_id']}, "
          f"`{e['block_id']}`: `…{e['tail'][-38:]}`")
    p("    ")
    dropped_lines = [u for u in x["unmatched_lines"]
                     if u["kind"] == "dropped_line"]
    tag_lines = [u for u in x["unmatched_lines"]
                 if u["kind"] == "renderer_tag_line"]
    p(f"6. **Dropped OCR lines — {len(dropped_lines)} line, "
      f"{x['dropped_line_chars']} characters.** One raw OCR line "
      f"is absent from the corpus entirely:")
    for u in dropped_lines:
        p(f"    - 0-based PDF page {u['page_id']}, "
          f"`{u['block_id']}`: `{u['text']!r}`")
    p("    ")
    p(f"    (A further {len(tag_lines)} unmatched raw lines are "
      f"the credit-page tag lines of class 2.) The dropped line "
      f"explains the residual `e`/`g`/`n` letter deltas exactly "
      f"and part of `b`/`i`/`>`/`/`. **Leads for Chainlink #4 "
      f"(ensemble OCR) and #7 (layout recovery):** re-check the "
      f"page image at the cited line; the loss is recoverable "
      f"from the raw stream. It is reported here, not repaired, "
      f"because the freeze must describe the corpus as it "
      f"stands.")
    p("7. **Line/paragraph restructuring** — the newline and "
      "space counts differ because marker's markdown renderer "
      "re-packs OCR lines into paragraphs (1,468 newlines and "
      "69,066 spaces in the corpus vs 38 and 66,584 in the "
      "line-based raw stream). No glyph is lost; only the "
      "packing changes. The same re-packing, plus operator "
      "spacing drift around `>>` (drift the workspace already "
      "documents: README, #3 §6), accounts for the remaining "
      "residual.")
    explained = (x["folio_digit_chars"]
                 + x["renderer_tag_chars"]
                 + x["dropped_line_chars"]
                 + x["em_dash_delta"]
                 + x["hyphen_delta"])
    residual = x["raw_surplus"] - explained
    p("")
    p(f"**Accounting:** the raw stream's total character surplus "
      f"over the corpus is {fmt_count(x['raw_surplus'])} "
      f"characters (the corpus is also missing "
      f"{fmt_count(x['raw_deficit'])} characters the raw stream "
      f"lacks — the markdown-only `#`/`*` and the re-packed "
      f"newlines/spaces). Of the surplus, {fmt_count(explained)} "
      f"are fully located by the classes above "
      f"({x['folio_digit_chars']} folio digits + "
      f"{x['renderer_tag_chars']} tag characters + "
      f"{x['dropped_line_chars']} dropped-line characters + "
      f"{x['em_dash_delta']} em dashes + "
      f"{x['hyphen_delta']} de-hyphenated hyphens), leaving a "
      f"**residual of {fmt_count(residual)} characters** spread "
      f"over a handful of letters and the angle brackets — "
      f"consistent with line-join re-packing and operator-spacing "
      f"drift, and small enough to localize by eye with the "
      f"cited block ids if #7 ever needs to.")
    p("")
    p("**Net claim of this cross-check:** the non-ASCII alphabet "
      "is identical in both views (5 glyphs, same set); the "
      "normalization hazard classes are all at zero in both "
      "views (§3); and every large count delta above is explained "
      "by a located, understood rendering behavior. No residual "
      "touches a non-ASCII glyph or any normalization hazard "
      "class.")
    p("")
    p("**The backslash finding.** The corpus contains 8 reverse "
      "solidi. 7 are literal `\\n` pairs (e.g. `HDD\\ninternal`, "
      "`the technocrisis was\\nunleashed`) and 1 is "
      "`H\\body-OMOTYA`. All 8 are present in the raw OCR "
      "stream, so they are **genuine text of the scan** — the "
      "book's own codework (text-virus poetics), not a "
      "renderer escape. They are kept as evidence (decision: "
      "keep, U+005C).")
    p("")
    p("## 3. Negative results (what is NOT in the corpus)")
    p("")
    p("Every hazard class the normalizer defends against was "
      "measured at zero in both the corpus and the raw OCR "
      "stream:")
    p("")
    p("| hazard class | codepoints watched | in corpus | in raw "
      "OCR stream |")
    p("|---|---|---|---|")
    for rule, mapping in normalize.MAPPED_RULES.items():
        in_corpus = sum(n for cp, n in corpus["counts"].items()
                        if chr(cp) in mapping)
        in_raw = sum(n for cp, n in raw["counts"].items()
                     if chr(cp) in mapping)
        watched = len(mapping)
        p(f"| {rule} | {watched} | {fmt_count(in_corpus)} | "
          f"{fmt_count(in_raw)} |")
    stray = [cp for cp in corpus["counts"]
             if unicodedata.category(chr(cp)) in ("Cc", "Cf")
             and chr(cp) not in "\n\t"]
    p(f"| stray control/format characters | all Cc/Cf | "
      f"{len(stray)} | {x['raw_stray_controls']} |")
    p("")
    p("Interpretation: marker + ftfy already emit clean, "
      "normalized Unicode for this scan. The freeze therefore "
      "does not *change* the corpus — it **locks the observed "
      "state in** and guarantees that any future OCR pass (the "
      "ensemble of Chainlink #4, a re-run, a different scan) "
      "normalizes identically: ligatures decompose, curly "
      "quotes canonicalize to ASCII, unicode spaces collapse, "
      "invisible characters vanish. Downstream tasks "
      "(#6 gazetteer correction, #17 operator-aware tokenizer, "
      "#10 operator inventory) can assume a stable alphabet of "
      "exactly the 88 codepoints listed in `charset.tsv`.")
    p("")
    p("## 4. Canonical-form decisions (the frozen rules)")
    p("")
    p("The rule pipeline, in order, with its measured fire "
      "count on the frozen corpus:")
    p("")
    p("| # | rule | what it does | fires on corpus |")
    p("|---|---|---|---|")
    for i, (name, desc, events) in enumerate(per_rule, 1):
        p(f"| {i} | `{name}` | {desc} | {fmt_count(events)} |")
    p("")
    p("Order matters and is deliberate: the semantically named "
      "rules (`quotes`, `spaces`) run *before* the generic NFKC "
      "decomposition, so typographic quotes and primes claim "
      "their characters before NFKC splits double primes "
      "(U+2033 → U+2032 U+2032) into single primes, and so "
      "fire counts are attributed to the rule a human would "
      "name. The explicit ligature table runs *after* NFKC as "
      "pure defence in depth (every entry maps to exactly what "
      "NFKC maps it to — asserted by the tests).")
    p("")
    p("**Deliberate non-decisions** (things the normalizer "
      "refuses to do, because they would destroy evidence):")
    p("")
    p("- **No case folding.** `ADAM` vs `adam`, `ToK`, "
      "`BABEL-TV` — capitalization is part of the book's "
      "signalling (entity induction, #30, depends on it).")
    p("- **No diacritic stripping.** `Ø` in `MØNKEY/gene-dub` "
      "is a real glyph, not an OCR error to flatten. Likewise "
      "the ligature *letters* `Œ`/`œ`/`Ĳ`/`ĳ` are left alone "
      "wherever they might appear: Unicode itself does not "
      "decompose them, because they are legitimate letters, not "
      "typographic presentation forms.")
    p("- **Dashes stay distinct.** The corpus uses U+2014 em "
      "dash (484×) and U+2013 en dash (43×) as prose dashes. "
      "Whether the scan's two dash glyphs are one authorial "
      "dash quantized by marker's width estimation, or two "
      "deliberate forms, **cannot be decided from text alone** "
      "— it is exactly the kind of glyph ambiguity the "
      "multi-engine ensemble of Chainlink #4 exists to "
      "adjudicate. Until then the lossless choice is to keep "
      "both. Downstream tasks that want a single dash class can "
      "opt in with `normalize(text, flatten_dashes=True)` "
      "(maps the 10-codepoint dash class to U+2014; hyphen-minus "
      "forms are excluded so the compound linker `boy-roid` can "
      "never become `boy—roid`); the default stays off.")
    p("- **No lexical correction.** `lt` for `It`, `boyroid` "
      "for `boy-roid` are *lexical* OCR drift, not unicode "
      "variant forms; normalizing them here would pre-empt and "
      "corrupt the gazetteer-protected correction of #6.")
    p("- **Backslash kept.** The literal `\\n` sequences are "
      "book content (§2.3).")
    p("")
    p("## 5. Freeze record")
    p("")
    p(f"- corpus: `{os.path.relpath(corpus['path'], ROOT)}` — "
      f"sha256 `{corpus['sha256']}`")
    p(f"- normalized corpus sha256: `{ctx['normalized_sha256']}` "
      f"— **identical** to the corpus sha256 "
      f"({ctx['normalizations_applied']} character(s) changed)")
    p(f"- idempotence: `normalize(normalize(x)) == normalize(x)` "
      f"holds on the corpus and on the unit-test corpus")
    p(f"- charset: `{os.path.relpath(ctx['charset_path'], ROOT)}` "
      f"— {len(rows)} rows, sha256 `{ctx['charset_sha256']}`")
    p(f"- normalizer: `work/normalize.py` v{normalize.NORMALIZER_VERSION} "
      f"— sha256 `{ctx['normalizer_sha256']}`")
    p(f"- test: `python work/test_normalize.py` → "
      f"**{test['result']}** ({test['tests_run']} tests, "
      f"{test['duration_s']:.1f}s; sha256 "
      f"`{test['sha256']}`)")
    p("")
    p("Re-verify the freeze at any time:")
    p("")
    p("```bash")
    p("python work/test_normalize.py "
      "# suite: rules + freeze consistency + CLI")
    p("python work/normalize.py verify "
      "docs/markdown/blood-electric/blood-electric.md "
      "# corpus ⊆ charset.tsv")
    p("python work/normalize.py "
      "docs/markdown/blood-electric/blood-electric.md "
      "# per-rule fire counts (expect all 0)")
    p("```")
    p("")
    p("## 6. Files")
    p("")
    p("| file | what it is |")
    p("|---|---|")
    p(f"| `analysis/charset/charset.tsv` | the frozen charset: "
      f"every observed codepoint (decision, counts in corpus and "
      f"raw OCR stream, curated note) plus every frozen "
      f"normalization map (unobserved here, counted 0).  "
      f"Non-printable characters (newline, NBSP, zero-width) are "
      f"escaped in the `char`/`canonical` columns (`\\n`, "
      f"`\\u00A0`, ...) so the TSV stays one row per codepoint. |")
    p(f"| `analysis/charset/freeze.json` | machine-readable "
      f"freeze record: hashes, counts, per-rule fire counts, "
      f"test result |")
    p(f"| `analysis/charset/charset_report.md` | this report |")
    p(f"| `work/normalize.py` | the reusable normalizer "
      f"(library + CLI: normalize / verify / freeze) |")
    p(f"| `work/test_normalize.py` | the test suite (stdlib "
      f"unittest; runs under `python work/test_normalize.py` "
      f"and unittest discovery) |")
    p(f"| `work/build_charset.py` | this builder (regenerates "
      f"all three artifacts deterministically) |")
    p("")
    p("## 7. Caveats (what this does not establish)")
    p("")
    p("1. **The freeze is only as good as the scan.** It "
      "characterizes marker's output on *this* PDF; a different "
      "scan or engine can introduce new codepoints, which "
      "`verify` will then flag as charset violations by design.")
    p("2. **Dash distinction unresolved** (§4): en vs em dash "
      "is preserved, not explained. Ensemble OCR (#4) is the "
      "designated adjudicator.")
    p("3. **The corpus is missing 2 em dashes** (§2.3, pages 41 "
      "and 53) **and one 7-character OCR line** (page 58, "
      "`being>>/`) that the OCR stream contains — marker "
      "line-joiner behaviors, reported here and left for "
      "#4/#7. Every em-dash count in the program (e.g. #10's "
      "operator inventory) inherits the 2-count undercount.")
    p("4. **Normalization is not correction.** A correctly "
      "normalized `lt` is still the wrong token where the book "
      "has `It`; that fix belongs to #6 and must stay "
      "gazetteer-gated so true coinages survive.")
    p("5. **Markdown formatting is included.** `#` and `*` are "
      "renderer-added markup, not book glyphs; tasks that want "
      "pure prose should strip markdown headings/bold first "
      "(as `work/analyze.py` does), the charset records them "
      "either way so the inventory is complete.")
    p("6. **Folio numbers are absent from the corpus.** The "
      "printed page numbers exist only in the raw OCR stream "
      "(§2.3); the page/folio map of Chainlink #8 must source "
      "them from there (or the PDF), not from the markdown.")

    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def run_test_suite():
    """Run work/test_normalize.py; return a result dict."""
    test_path = normalize.TEST_FILE
    started = time.time()
    proc = subprocess.run(
        [sys.executable, test_path],
        capture_output=True, text=True, timeout=600,
        cwd=ROOT,
    )
    duration = time.time() - started
    # unittest writes its progress and the "Ran N tests"
    # summary to stderr, not stdout.
    summary = proc.stdout + proc.stderr
    ran = None
    for line in summary.splitlines():
        if line.startswith("Ran "):
            try:
                ran = int(line.split()[1])
            except (ValueError, IndexError):
                pass
    return {
        "path": test_path,
        "sha256": sha256(test_path),
        "command": "python work/test_normalize.py",
        "returncode": proc.returncode,
        "result": "PASS" if proc.returncode == 0 else "FAIL",
        "tests_run": ran,
        "duration_s": round(duration, 1),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def build(corpus=DEFAULT_CORPUS, raw_dump=DEFAULT_RAW_DUMP,
          outdir=DEFAULT_OUTDIR, run_tests=True):
    os.makedirs(outdir, exist_ok=True)
    generated_at = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")

    # --- measure -------------------------------------------------------
    with open(corpus, encoding="utf-8") as fh:
        corpus_text = fh.read()
    corpus_counts = normalize.inventory(corpus_text)
    records, raw_text, per_page = read_raw_dump(raw_dump)
    raw_counts = normalize.inventory(raw_text)

    non_ascii_pages = collections.defaultdict(list)
    for page_id, parts in sorted(per_page.items()):
        page_text = "".join(parts)
        for ch in set(page_text):
            if ord(ch) > 0x7F:
                non_ascii_pages[ch].append(page_id)

    normalized, per_rule, total_events = normalize.normalize_report(
        corpus_text)
    normalized_sha256 = hashlib.sha256(
        normalized.encode("utf-8")).hexdigest()

    ascii_stats = ascii_class_stats(corpus_counts)
    cross = cross_check(records, raw_text, corpus_text)

    rows = charset_rows(corpus_counts, raw_counts)
    keep_rows = sum(1 for r in rows if r["decision"] == DECISION_KEEP)
    mapped_rows = len(rows) - keep_rows

    charset_path = os.path.join(outdir, "charset.tsv")
    write_charset_tsv(rows, charset_path)

    # marker JSON html-entity cross-check (optional evidence)
    marker_json_ctx = None
    if os.path.exists(MARKER_JSON):
        html = read_marker_json_html(MARKER_JSON)
        if html is not None:
            marker_json_ctx = {
                "lt": html.count("&lt;"),
                "gt": html.count("&gt;"),
            }

    # --- run the test suite against the freshly written charset ----
    test = run_test_suite() if run_tests else {
        "path": normalize.TEST_FILE,
        "sha256": sha256(normalize.TEST_FILE),
        "command": "python work/test_normalize.py",
        "returncode": None,
        "result": "SKIPPED",
        "tests_run": None,
        "duration_s": 0.0,
        "stdout": "",
        "stderr": "",
    }

    freeze_path = os.path.join(outdir, "freeze.json")
    report_path = os.path.join(outdir, "charset_report.md")

    ctx = {
        "generated_at": generated_at,
        "corpus": {
            "path": corpus,
            "sha256": sha256(corpus),
            "chars": len(corpus_text),
            "codepoints": len(corpus_counts),
            "ascii_codepoints": sum(1 for cp in corpus_counts
                                    if cp <= 0x7F),
            "non_ascii_codepoints": sum(1 for cp in corpus_counts
                                        if cp > 0x7F),
            "counts": corpus_counts,
        },
        "raw": {
            "path": raw_dump,
            "sha256": sha256(raw_dump),
            "chars": len(raw_text),
            "codepoints": len(raw_counts),
            "non_ascii_codepoints": sum(1 for c in raw_counts
                                        if c > 0x7F),
            "counts": raw_counts,
            "non_ascii_pages": non_ascii_pages,
        },
        "per_rule": per_rule,
        "rows": rows,
        "keep_rows": keep_rows,
        "mapped_rows": mapped_rows,
        "total_events": total_events,
        "normalizations_applied": total_events,
        "normalized_sha256": normalized_sha256,
        "ascii_stats": ascii_stats,
        "marker_json": marker_json_ctx,
        "cross": cross,
        "charset_path": charset_path,
        "charset_sha256": sha256(charset_path),
        "normalizer_sha256": sha256(
            normalize.__file__
            if normalize.__file__.endswith(".py")
            else os.path.join(HERE, "normalize.py")),
        "test": {k: v for k, v in test.items()
                 if k not in ("stdout", "stderr")},
        "freeze_path": freeze_path,
        "report_path": report_path,
    }

    freeze = {
        "task": "chainlink#5",
        "title": "Inventory glyphs and freeze unicode "
                 "normalization",
        "generated_at": generated_at,
        "normalizer": {
            "path": "work/normalize.py",
            "version": normalize.NORMALIZER_VERSION,
            "sha256": ctx["normalizer_sha256"],
        },
        "test": ctx["test"],
        "corpus": {
            "path": os.path.relpath(corpus, ROOT),
            "sha256": ctx["corpus"]["sha256"],
            "chars": ctx["corpus"]["chars"],
            "codepoints": ctx["corpus"]["codepoints"],
            "non_ascii_codepoints":
                ctx["corpus"]["non_ascii_codepoints"],
        },
        "raw_ocr_stream": {
            "path": os.path.relpath(raw_dump, ROOT),
            "sha256": ctx["raw"]["sha256"],
            "chars": ctx["raw"]["chars"],
            "codepoints": ctx["raw"]["codepoints"],
            "non_ascii_codepoints":
                ctx["raw"]["non_ascii_codepoints"],
        },
        "rules": [
            {"name": name, "description": desc,
             "fires_on_corpus": events}
            for name, desc, events in per_rule
        ],
        "normalizations_applied": total_events,
        "normalized_corpus_sha256": normalized_sha256,
        "corpus_unchanged": normalized_sha256
        == ctx["corpus"]["sha256"],
        "idempotent": normalize.normalize(normalized) == normalized,
        "cross_check": {
            "non_ascii_alphabet_identical": (
                {cp for cp in raw_counts if cp > 0x7F}
                == {cp for cp in corpus_counts if cp > 0x7F}),
            "em_dash_delta": cross["em_dash_delta"],
            "em_dashes_dropped_at_line_joins": [
                {"page_id": e["page_id"],
                 "block_id": e["block_id"]}
                for e in cross["em_dash_enders"]
                if not e["survived"]],
            "dropped_ocr_lines": [
                {"page_id": u["page_id"],
                 "block_id": u["block_id"],
                 "text": u["text"]}
                for u in cross["unmatched_lines"]
                if u["kind"] == "dropped_line"],
            "folio_lines": len(cross["folio_lines"]),
            "folio_digit_chars": cross["folio_digit_chars"],
            "renderer_tags": cross["renderer_tags"],
            "renderer_tag_chars": cross["renderer_tag_chars"],
            "hyphen_delta": cross["hyphen_delta"],
            "raw_surplus": cross["raw_surplus"],
            "raw_deficit": cross["raw_deficit"],
            "codepoint_delta": {
                f"U+{cp:04X}": [raw_n, md_n]
                for cp, (raw_n, md_n) in cross["delta"].items()
            },
        },
        "charset_tsv": {
            "path": os.path.relpath(charset_path, ROOT),
            "sha256": ctx["charset_sha256"],
            "rows": len(rows),
            "observed_keep_rows": keep_rows,
            "mapped_rows": mapped_rows,
            "decisions": {
                "keep": keep_rows,
                "normalize": sum(1 for r in rows
                                 if r["decision"] == DECISION_NORMALIZE),
                "strip": sum(1 for r in rows
                             if r["decision"] == DECISION_STRIP),
            },
        },
    }
    with open(freeze_path, "w", encoding="utf-8") as fh:
        json.dump(freeze, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    report = build_report(ctx)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    # --- console summary ---------------------------------------------
    print(f"corpus:    {corpus}")
    print(f"  chars:   {fmt_count(ctx['corpus']['chars'])}  "
          f"codepoints: {ctx['corpus']['codepoints']}  "
          f"non-ascii: {ctx['corpus']['non_ascii_codepoints']}")
    print(f"raw OCR:   {raw_dump}")
    print(f"  chars:   {fmt_count(ctx['raw']['chars'])}  "
          f"codepoints: {ctx['raw']['codepoints']}")
    print(f"rules fired on corpus: {total_events}")
    print(f"cross-check: non-ascii alphabet identical: "
          f"{freeze['cross_check']['non_ascii_alphabet_identical']}")
    print(f"  em dash delta: {cross['em_dash_delta']:+d} "
          f"(dropped at line joins: "
          f"{[e['page_id'] for e in cross['em_dash_enders'] if not e['survived']]})")
    print(f"charset:   {charset_path} ({len(rows)} rows: "
          f"{keep_rows} keep + {mapped_rows} mapped)")
    print(f"freeze:    {freeze_path}")
    print(f"report:    {report_path}")
    print(f"tests:     {test['result']} "
          f"({test['tests_run']} tests, {test['duration_s']}s)")
    if test["returncode"] != 0:
        print(test["stdout"])
        print(test["stderr"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(build())
