#!/usr/bin/env python3
"""#70 Coverage accounting / residue ledger -> analysis/coverage_ledger/

The question this module answers
--------------------------------
Every analysis in this workspace reads *some* layer: the character stream
(#47), the token stream (#17/#51), the unit layer (#19/#24/#27), the part
layer (#56).  Each screen removes occurrences and each removal is justified in
the removing issue's own reconciliation table.  Nobody has asked whether the
*whole* corpus is accounted for: if every character of every page body, every
token, every operator token and every unit occurrence can be placed in a named
bucket, then every removal is auditable; whatever is left over is residue the
analysis quietly ignored.

So one ledger is built at four levels and each level is forced to close:

  L1 characters  480,962 body chars -> token text | gap whitespace |
                 layout scaffolding | unexplained
  L2 tokens       90,220 -> WORD | OP | PUNCT (reproduces #14's 77,485 Lex
                 nodes and #17's 11,000 operator tokens)
  L3 operators    11,000 -> #17's role partition (seam / segment / the four
                 orphan roles / slash_free / slash_boundary)
  L4 units        74,665 occurrences -> #19's category screen | the 347-unit
                 markup ledger | unexplained; 5,876 types -> 5,868 screened
                 | 8 markup-only types

The **coverage metric** is the share of a level that carries no unexplained
residue.  On this corpus it is 1.000000 at all four levels and the build fails
if it ever drops below that, so the metric is a closure test rather than a
score.  A metric that cannot fail needs teeth, so the ledger also publishes
the awkward part -- residue that is *explained* but itemized nowhere, plus the
one place where a screen deletes real book text -- in `residue_ledger.tsv`:
one row per class with its count, pages, page-cited examples, the artifact
that owns it, and the verdict `explained` / `disclosed` / `unexplained`.

Inputs are frozen artifacts only: #7's layout markdown, #17's token/unit
streams, #19's screen and markup ledger, #27's tag-run detector and the
published `summary.json` of every downstream layer.  No OCR, no download, no
reference corpus.

Outputs under analysis/coverage_ledger/:
  coverage_ledger_report.md     THE DELIVERABLE
  coverage.tsv                  the ledger: every bucket at every level
  residue_ledger.tsv            the awkward residue, itemized and verdicted
  per_page.tsv                  per-page accounting (where the residue lives)
  consumer_reconciliation.tsv   every downstream layer's kept/dropped split
  reconciliation.tsv            this ledger vs #10/#14/#17/#19/#27/#47
  diagnostics.tsv               build counters
  summary.json                  machine-readable headline numbers

Reproduce::

    python work/coverage_ledger.py          # writes analysis/coverage_ledger/
    python work/test_coverage_ledger.py     # regression suite

Run the whole work/ suite with `python -m unittest discover -s work -p
'test_*.py'`.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))
import collocation as coll            # #27 markup screen
import operator_grammar as OG         # #14 page bodies
import operator_tokenizer as T        # #17 tokenizer

LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
TOKENS = ROOT / "analysis" / "tokenizer" / "tokens.tsv"
TAILS = ROOT / "analysis" / "tokenizer" / "page_tails.tsv"
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
UNIT_TYPES = ROOT / "analysis" / "tokenizer" / "unit_types.tsv"
SCREENING = ROOT / "analysis" / "neologism" / "screening.tsv"
MARKUP = ROOT / "analysis" / "neologism" / "markup_residue.tsv"
OUT = ROOT / "analysis" / "coverage_ledger"
SCHEMA = "coverage-ledger/1"

# ---------------------------------------------------------------------------
# Numbers this ledger must reproduce.  Each is published by the issue that
# owns it, so drift means an artifact stopped agreeing with its own code.
# ---------------------------------------------------------------------------
EXPECT_PAGES = 225            # #7 / #17
EXPECT_TOKENS = 90220         # #14 roundtrip / #17 tokens.total
EXPECT_WORDS = 77485          # #14: Lex (word-atom) nodes
EXPECT_OPS = 11000            # #14: 10,949 body + 51 header operators
EXPECT_PUNCT = 1735           # #17 tokens.PUNCT
EXPECT_UNITS = 74665          # #17 units.total
EXPECT_UNIT_TYPES = 5876      # #17 units.types
EXPECT_PARTS = 80363          # #17 units.parts
EXPECT_SEAMS = 2816           # #17 op_roles.seam
EXPECT_SEGMENTS = 7247        # #17 op_roles.segment
EXPECT_UNBOUND = 937          # #17 unbound_ops: 716 slash + 54 ++ / 165 = / 2 +
EXPECT_SCREENED = 74318       # #19 counts.screened_occurrences
EXPECT_SCREEN_TYPES = 5868    # #19 counts.screened_types
EXPECT_MARKUP = 347           # #19 markup ledger, occurrence level
EXPECT_MARKUP_TYPES = 8       # #19 counts.markup_only_types_dropped
EXPECT_TAG_RUN = 241          # #27 counts.markup_extra_removed
EXPECT_FULL_SCREEN = 74077    # #17 units - #19 ledger - #27 tag runs
EXPECT_BODY_CHARS = 480962    # sum of #7 page-body lengths
EXPECT_SCAFFOLD_CHARS = 468   # layout comment text inside page bodies
EXPECT_SCAFFOLD_SPANS = 12    # <!-- figure/picture block (no text) -->
EXPECT_BOOK_TEXT_IN_SPANS = 13   # real text inside #27's tag-run spans
EXPECT_ENTROPY_CHARS = 480309    # #47 reconciliation.tsv, raw char stream
EXPECT_HEADER_MARKER_CHARS = 197  # #47's `#` heading markers, stripped

# #47's two cleaning steps, reproduced so its char total is reconcilable with
# this ledger's raw total rather than merely different from it.
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADER_RE = re.compile(r"^[ \t]*#+[ \t]*", re.MULTILINE)

# Buckets that are *not* book text.  `href`/`mailto` are the attribute name
# and value of the anchors the OCR fabricated, and #19 already keeps them as
# `junk` types; `colspan` and its value `2` are the same thing inside
# `<th colspan="2">`.  Everything else a tag-run span covers is book text.
TAG_ATTR_FURNITURE = {"href", "mailto"}
TAG_VALUE_FURNITURE = {"colspan", "2"}
TAG_FURNITURE = TAG_ATTR_FURNITURE | TAG_VALUE_FURNITURE

# #17's operator role partition (operator_tokenizer._is_seam reasons).
ROLE_SEAM = "seam"
ROLE_SEGMENT = "segment"
ROLE_UNBOUND = ("seam_orphan_left", "seam_orphan_right", "seam_orphan_side",
                "slash_free", "slash_boundary")

VERDICT_EXPLAINED = "explained"      # a published artifact owns it by name
VERDICT_DISCLOSED = "disclosed"      # counted, but no artifact itemizes it
VERDICT_UNEXPLAINED = "unexplained"  # nothing in the workspace accounts for it


class LedgerError(RuntimeError):
    """A level did not close, or an input artifact is missing."""


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
def read_rows(path: Path) -> list[dict]:
    """Tab-split reader.  These artifacts carry bare `"` and escaped `\\n`, so
    a quote-aware csv reader mis-sizes rows (#17's disclosed hazard)."""
    if not path.exists():
        raise LedgerError("missing input artifact: %s" % path)
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = []
        for line in fh:
            if not line.strip():
                continue
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    return rows


def read_json(path: Path) -> dict:
    if not path.exists():
        raise LedgerError("missing input artifact: %s" % path)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def dig(obj, keys) -> object:
    """Dotted-key lookup that fails loudly instead of returning None."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise LedgerError("key %r not found in %r" % (k, obj))
        cur = cur[k]
    return cur


def page_bodies() -> dict:
    """{page: body} from #7's vetted layout markdown."""
    return OG.load_pages_raw(str(LAYOUT))


# ---------------------------------------------------------------------------
# TSV escaping
# ---------------------------------------------------------------------------
_ESCAPES = {"\\": "\\", "t": "\t", "n": "\n", "r": "\r"}


def unescape(text: str) -> str:
    """Reverse one pass of the workspace's TSV escaping.

    #17 writes a unit surface escaped once (`\\n` for the newline inside
    `eruption=\\njunk`); #19 reads that surface back with `csv.DictReader`
    -- which does not unescape -- and writes it with a writer that escapes
    the backslash first, so the same surface reaches `screening.tsv` with
    *two* backslashes.  One pass of this function realigns the two artifacts.
    """
    out, i = [], 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            out.append(_ESCAPES.get(text[i + 1], text[i + 1]))
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def screening_index(screening) -> tuple:
    """({surface: row}, [surfaces that needed one unescape pass to match]).

    A surface-level join between #17's `units.tsv` and #19's `screening.tsv`
    is already unsafe (#17 says so: join on `uid`); the escape mismatch makes
    it fail on two specific surfaces, which is the only reason this ledger
    needs the normalisation at all.
    """
    index, mismatched = {}, []
    for row in screening:
        surface = row["unit"]
        index.setdefault(surface, row)
        plain = unescape(surface)
        if plain != surface:
            mismatched.append(surface)
        index.setdefault(plain, row)
    return index, mismatched


# ---------------------------------------------------------------------------
# L0: the token stream, rebuilt from #17's artifacts and checked twice
# ---------------------------------------------------------------------------
def load_artifact_stream() -> tuple:
    """(tokens, tails) from #17's artifacts: page -> [token dict] / tail str.

    `tokens.tsv` stores the gap as a JSON string literal and the tail lives in
    `page_tails.tsv`, because the page body = `"".join(gap + text)` + tail.
    """
    tokens = collections.defaultdict(list)
    for r in read_rows(TOKENS):
        r["gap"] = json.loads(r["gap_json"])
        tokens[int(r["page"])].append(r)
    tails = {int(r["page"]): json.loads(r["tail_json"])
             for r in read_rows(TAILS)}
    return dict(tokens), tails


def rebuild_bodies(tokens, tails) -> dict:
    """{page: body} from #17's artifacts alone."""
    return {p: "".join(t["gap"] + t["text"] for t in rows) + tails.get(p, "")
            for p, rows in tokens.items()}


def check_stream(bodies) -> dict:
    """Cross-check #17's artifact stream against #17's code and #7's bodies.

    A coverage ledger is only meaningful over a stream that is known to be
    lossless and to agree with the code that produced it, so both are asserted
    here rather than assumed.  Returns the code tokenization of the book.
    """
    rebuilt = rebuild_bodies(*load_artifact_stream())
    if set(rebuilt) != set(bodies):
        raise LedgerError("page sets disagree: #7 has %d pages, #17 has %d"
                          % (len(bodies), len(rebuilt)))
    bad = [p for p in sorted(bodies) if rebuilt[p] != bodies[p]]
    if bad:
        raise LedgerError("#17's stream does not rebuild page bodies: %s"
                          % bad[:5])
    stream = load_artifact_stream()[0]
    code = {p: T.tokenize_page(p, bodies[p]) for p in sorted(bodies)}
    drift = []
    for p in sorted(bodies):
        mine = [(t.kind, t.s, t.e, t.text, t.gap) for t in code[p][0]]
        theirs = [(r["kind"], int(r["start"]), int(r["end"]), r["text"],
                   r["gap"]) for r in stream[p]]
        if mine != theirs:
            drift.append(p)
    if drift:
        raise LedgerError("#17's tokens.tsv drifts from the tokenizer: %s"
                          % drift[:5])
    return code


# ---------------------------------------------------------------------------
# L1: characters
# ---------------------------------------------------------------------------
def char_map(body: str, tokens, spans) -> list[str]:
    """One bucket label per character of a page body.

    Layout comments win over tokens (the tokenizer emits nothing inside
    them), then token text by kind, then the gap: whitespace or unexplained.
    """
    labels = ["gap"] * len(body)
    for s, e in spans:
        for i in range(s, e):
            labels[i] = "layout_comment"
    for t in tokens:
        label = "token_" + t.kind.lower()
        if t.kind == "PUNCT" and t.pclass:
            label += "_" + t.pclass
        for i in range(t.s, t.e):
            labels[i] = label
    return ["gap_whitespace" if lab == "gap" and body[i].isspace()
            else ("gap_other" if lab == "gap" else lab)
            for i, lab in enumerate(labels)]


def scaffold_spans(book, bodies) -> list:
    """Every layout-comment span in the book, with whether it is a page tail.

    #17 publishes the tail of a page (`page_tails.tsv`) and its spec says the
    tails are whitespace except for figure-block comments on two pages; the
    spans that sit *between* tokens are published nowhere.
    """
    out = []
    for pno in sorted(book):
        body = bodies[pno]
        tail_start = len(body) - len(book[pno][2])
        for s, e in T.comment_spans(body):
            out.append({"page": pno, "start": s, "end": e, "n_chars": e - s,
                        "in_tail": s >= tail_start, "text": body[s:e]})
    return out


def char_ledger(book, bodies) -> tuple:
    """(bucket Counter, {bucket: [example dicts]}) over the whole book."""
    counts = collections.Counter()
    examples = collections.defaultdict(list)
    for pno in sorted(book):
        toks, _diags, _tail = book[pno]
        body = bodies[pno]
        spans = T.comment_spans(body)
        labels = char_map(body, toks, spans)
        for lab in labels:
            counts[lab] += 1
        for i, lab in enumerate(labels):
            if lab == "gap_other":
                examples["gap_other"].append({
                    "page": pno, "start": i,
                    "text": body[max(0, i - 40):i + 40]})
    return counts, examples


# ---------------------------------------------------------------------------
# L2/L3: tokens and the operator partition
# ---------------------------------------------------------------------------
def token_ledger(book) -> collections.Counter:
    """Token counts by kind (WORD / OP / PUNCT)."""
    counts = collections.Counter()
    for pno in sorted(book):
        for t in book[pno][0]:
            counts["token_" + t.kind] += 1
    return counts


def operator_role(toks, i) -> str:
    """#17's role for the operator token at `i`: exactly one per token."""
    is_seam, reason = T._is_seam(toks, i)
    if is_seam:
        return ROLE_SEAM
    if reason in ROLE_UNBOUND:
        return reason
    return ROLE_SEGMENT


def operator_ledger(book) -> tuple:
    """(role Counter, {role: Counter of operator texts})."""
    roles = collections.Counter()
    by_op = collections.defaultdict(collections.Counter)
    for pno in sorted(book):
        toks = book[pno][0]
        for i, t in enumerate(toks):
            if t.kind != "OP":
                continue
            role = operator_role(toks, i)
            roles[role] += 1
            by_op[role][t.text] += 1
    return roles, by_op


# ---------------------------------------------------------------------------
# L4: units
# ---------------------------------------------------------------------------
def unit_ledger(units, screening, markup_uids) -> tuple:
    """Closure of the unit layer against #19's per-occurrence screen.

    Returns (category Counter, residue Counter, detail dicts, escape fixes).
    A unit occurrence is either inside #19's markup ledger (uid level) or
    classified by surface; a surface-level join is *not* enough, because the
    ledger's `a` occurrences are the tag name of `<a href=...>` anchors and
    share a surface with the article.
    """
    by_surface, mismatched = screening_index(screening)
    cats = collections.Counter()
    residue = collections.Counter()
    detail = collections.defaultdict(list)
    escape_fixed = []
    for u in units:
        uid = int(u["uid"])
        if uid in markup_uids:
            residue["markup_ledger_occurrence"] += 1
            detail["markup_ledger_occurrence"].append(u)
            continue
        row = by_surface.get(u["unit"])
        if row is None:
            residue["unscreened_surface"] += 1
            detail["unscreened_surface"].append(u)
            continue
        cats[row["category"]] += 1
        if row["unit"] != u["unit"]:
            escape_fixed.append(u)
    return cats, residue, detail, escape_fixed


def type_ledger(units, screening, markup_uids) -> tuple:
    """Type-level closure: (markup-only types, mixed types).

    A surface whose *every* occurrence is in #19's markup ledger disappears
    from the type view; a surface that survives with some markup occurrences
    is the `a` case -- indistinguishable from the article by surface alone.
    The join uses the normalised index, so the two surfaces #19 stores with
    an extra escape are not mistaken for unscreened types.
    """
    index, _mismatched = screening_index(screening)
    by_surface = collections.defaultdict(list)
    for u in units:
        by_surface[u["unit"]].append(u)
    markup_only, mixed = [], []
    for surface in sorted(by_surface):
        occurrences = by_surface[surface]
        entry = {"unit": surface, "n": len(occurrences),
                 "markup_n": sum(1 for u in occurrences
                                 if int(u["uid"]) in markup_uids),
                 "pages": sorted({int(u["page"]) for u in occurrences})}
        if surface not in index:
            if entry["markup_n"] == entry["n"]:
                markup_only.append(entry)
        elif entry["markup_n"]:
            mixed.append(entry)
    return markup_only, mixed


# ---------------------------------------------------------------------------
# the #27 tag-run screen: what the extra 241 removals actually are
# ---------------------------------------------------------------------------
def tag_run_occurrences() -> tuple:
    """(rows, pages) of the occurrences #27 removes beyond #19's ledger.

    Each row carries the unit's page, offset, uid, surface and context, so the
    241 removals can be itemized rather than only counted.
    """
    bodies = page_bodies()
    spans, markup_uids, junk_types = coll.load_markup(bodies)
    page_units, _meta = coll.load_units()
    rows, pages = [], collections.Counter()
    for page in sorted(page_units):
        body = bodies[page]
        for start, uid, _lower, surface, _cls in page_units[page]:
            if uid in markup_uids:
                continue
            if not coll.unit_is_markup(surface, uid, start, page, spans,
                                       markup_uids, junk_types):
                continue
            pages[page] += 1
            rows.append({"page": page, "start": start, "uid": uid,
                         "unit": surface,
                         "context": body[max(0, start - 70):start + 70]})
    return rows, pages


def tag_run_detail() -> tuple:
    """(furniture Counter, book-text rows, pages) of the #27 extra removals.

    #27 removes #19's 347 ledger occurrences and 241 more caught by its
    general tag-run detector.  228 of those 241 are tag furniture
    (`href`/`mailto`/`colspan` and the `colspan` value `2`); the rest are real
    book text that sits inside an anchor the OCR fabricated -- the only place
    in the workspace where a screen deletes text rather than markup.
    """
    rows, pages = tag_run_occurrences()
    furniture = collections.Counter()
    book_text = []
    for row in rows:
        if row["unit"] in TAG_FURNITURE:
            furniture[row["unit"]] += 1
        else:
            book_text.append(row)
    return furniture, book_text, pages


# ---------------------------------------------------------------------------
# per-page accounting
# ---------------------------------------------------------------------------
def per_page_ledger(book, bodies, units, book_text, tag_run_pages) -> list:
    """One row per page: what the page is made of and what it carries as
    residue, so a reader can see *where* the awkward part lives."""
    tag_pages = collections.Counter(r["page"] for r in book_text)
    markup_pages = collections.Counter(int(r["page"]) for r in read_rows(MARKUP))
    units_per_page = collections.Counter(int(u["page"]) for u in units)
    rows = []
    for pno in sorted(book):
        toks = book[pno][0]
        body = bodies[pno]
        spans = T.comment_spans(body)
        labels = char_map(body, toks, spans)
        unbound = sum(1 for i, t in enumerate(toks)
                      if t.kind == "OP"
                      and operator_role(toks, i) in ROLE_UNBOUND)
        rows.append({
            "page": pno,
            "n_chars": len(body),
            "n_tokens": len(toks),
            "n_words": sum(1 for t in toks if t.kind == "WORD"),
            "n_ops": sum(1 for t in toks if t.kind == "OP"),
            "n_puncts": sum(1 for t in toks if t.kind == "PUNCT"),
            "n_units": units_per_page.get(pno, 0),
            "scaffold_chars": sum(1 for lab in labels
                                  if lab == "layout_comment"),
            "scaffold_spans": len(spans),
            "unbound_ops": unbound,
            "markup_ledger_units": markup_pages.get(pno, 0),
            "tag_run_units": tag_run_pages.get(pno, 0),
            "book_text_removed": tag_pages.get(pno, 0),
            "unexplained_chars": sum(1 for lab in labels
                                     if lab == "gap_other"),
        })
    return rows


# ---------------------------------------------------------------------------
# every downstream layer's kept/dropped split against its level total
# ---------------------------------------------------------------------------
LEVEL_TOTAL = {
    "unit occurrence": EXPECT_UNITS,
    "word part": EXPECT_PARTS,
    "token": EXPECT_TOKENS,
    "word token": EXPECT_WORDS,
}

CONSUMERS = [
    # (issue, layer, artifact, kept key path, dropped, where the drop is published)
    ("#17 tokenizer", "unit occurrence", "analysis/tokenizer/summary.json",
     ("units", "total"), 0, "the raw layer; nothing removed"),
    ("#19 neologism screen", "unit occurrence",
     "analysis/neologism/summary.json", ("counts", "screened_occurrences"), 347,
     "#19 markup_residue.tsv, occurrence level (uid)"),
    ("#24 frequency / Zipf", "unit occurrence",
     "analysis/frequency/summary.json", ("book_n",), 347,
     "#19 markup ledger"),
    ("#28 static embeddings", "unit occurrence",
     "analysis/embeddings/summary.json", ("counts", "units"), 347,
     "#19 markup ledger"),
    ("#30 entity induction", "unit occurrence",
     "analysis/entity/summary.json", ("unit_items_screened",), 588,
     "#19 ledger (347) + #27 tag runs (241)"),
    ("#27 collocation", "unit occurrence",
     "analysis/collocation/summary.json", ("counts", "unit_items"), 588,
     "#27 counts.markup_units_removed"),
    ("#33 clauses", "unit occurrence",
     "analysis/clauses/summary.json", ("n_units",), 588,
     "#33 markup_units_removed"),
    ("#35 n-gram surprisal", "unit occurrence",
     "analysis/ngram_surprisal/summary.json", ("n_units",), 588,
     "#19 + #27 screen"),
    ("#47 entropy", "unit occurrence",
     "analysis/entropy/summary.json", ("book", "token/unit", "n"), 588,
     "#47 reconciliation.tsv"),
    ("#56 language viability", "unit occurrence",
     "analysis/language_viability/summary.json", ("n_units_screened",), 588,
     "#19 + #27 screen"),
    ("#56 keyness", "word part", "analysis/keyness/summary.json",
     ("book", "kept"), 593, "#56 book.dropped (markup-screened parts)"),
    ("#27 collocation", "token", "analysis/collocation/summary.json",
     ("counts", "token_items"), 1551,
     "#27 counts.markup_tokens_removed"),
    ("#51 periodicity", "token", "analysis/periodicity/summary.json",
     ("book", "tok_op_screened", "N"), 0,
     "the full token stream, nothing removed"),
    ("#22 case / emphasis", "word token", "analysis/case_emphasis/summary.json",
     ("word_tokens",), 0, "every word token, screened in place"),
]


def consumer_ledger() -> list:
    """Read each layer's published kept count and check the split closes."""
    rows, problems = [], []
    for issue, layer, artifact, keys, dropped, source in CONSUMERS:
        kept = dig(read_json(ROOT / artifact), keys)
        total = LEVEL_TOTAL[layer]
        closes = int(kept) + int(dropped) == total
        if not closes:
            problems.append("%s (%s): %d kept + %d dropped != %d"
                            % (issue, layer, kept, dropped, total))
        rows.append({"issue": issue, "layer": layer, "level_total": total,
                     "kept": kept, "dropped": dropped,
                     "share_kept": round(100.0 * kept / total, 4),
                     "drop_published_in": source,
                     "closes": "y" if closes else "n"})
    if problems:
        raise LedgerError("consumer split does not close: " + "; ".join(problems))
    return rows


# ---------------------------------------------------------------------------
# coverage metric
# ---------------------------------------------------------------------------
def coverage(total: int, unexplained: int) -> float:
    """Share of a level that carries no unexplained residue.

    Everything the ledger *names* counts as covered, including the residue
    classes an artifact owns: naming is what makes a removal auditable.  Only
    the unnamed and the unclaimed count against coverage.
    """
    if not total:
        return 0.0
    return 1.0 - (unexplained / total)


def coverage_rows(chars, tokens, ops, unit_unexplained) -> list:
    """One row per level: total, covered, unexplained, coverage."""
    char_total = sum(chars.values())
    rows = [
        {"level": "L1 characters", "total": char_total,
         "covered": char_total - chars.get("gap_other", 0),
         "unexplained": chars.get("gap_other", 0),
         "coverage": round(coverage(char_total,
                                    chars.get("gap_other", 0)), 6)},
        {"level": "L2 tokens", "total": sum(tokens.values()),
         "covered": sum(tokens.values()), "unexplained": 0, "coverage": 1.0},
        {"level": "L3 operator tokens", "total": sum(ops.values()),
         "covered": sum(ops.values()), "unexplained": 0, "coverage": 1.0},
        {"level": "L4 unit occurrences", "total": EXPECT_UNITS,
         "covered": EXPECT_UNITS - unit_unexplained,
         "unexplained": unit_unexplained,
         "coverage": round(coverage(EXPECT_UNITS, unit_unexplained), 6)},
    ]
    return rows


def punct_other_ledger(book) -> collections.Counter:
    """Punctuation #12/#14 place in no class: `#` heading markers, `\\`, `•`,
    `©`.  Counted by #17 as PUNCT tokens, so nothing is lost -- but nothing
    reads them either."""
    counts = collections.Counter()
    for pno in sorted(book):
        for t in book[pno][0]:
            if t.kind == "PUNCT" and t.pclass == "other":
                counts[t.text] += 1
    return counts


def _pages_of(rows, key="page"):
    return sorted({int(r[key]) for r in rows})


def _markup_composition(markup_rows) -> list:
    """#19's markup ledger grouped by surface, most frequent first."""
    counts = collections.Counter(r["unit"] for r in markup_rows)
    return [{"unit": u, "n": n} for u, n in counts.most_common()]


def _punct_pages(book) -> list:
    """Pages carrying punctuation in no #12/#14 class."""
    return [{"page": p, "n": 1} for p in sorted(book)
            if any(t.kind == "PUNCT" and t.pclass == "other"
                   for t in book[p][0])]


def _fmt_pages(pages, limit=8):
    pages = sorted(pages)
    if len(pages) <= limit:
        return ", ".join(str(p) for p in pages)
    return ", ".join(str(p) for p in pages[:limit]) + " ... (%d pages)" % len(pages)


# ---------------------------------------------------------------------------
# the residue ledger: the awkward part, itemized and verdicted
# ---------------------------------------------------------------------------
def residue_ledger(book, bodies, chars, ops, ops_by_op, cats, unit_residue,
                   markup_only, mixed, furniture, book_text, tag_run_pages,
                   screening, markup_rows, escape_fixed) -> list:
    """Every class of awkward residue, with the artifact that owns it.

    `verdict` is `explained` when a published artifact names the items,
    `disclosed` when the workspace counts them but itemizes them nowhere, and
    `unexplained` when nothing accounts for them.  The last value must not
    occur: `build()` refuses to write the artifact if it does.
    """
    rows = []
    n_chars = sum(chars.values())
    scaffold = scaffold_spans(book, bodies)
    gap_tail = [s for s in scaffold if s["in_tail"]]

    # ---- L1 characters -------------------------------------------------
    if scaffold:
        rows.append({
            "class": "layout scaffolding comments",
            "level": "character",
            "item": "%d `<!-- figure/picture block (no text) -->` spans, "
                    "%d in page tails / %d between tokens"
                    % (len(scaffold), len(gap_tail), len(scaffold) - len(gap_tail)),
            "n": chars["layout_comment"],
            "share_pct": round(100.0 * chars["layout_comment"] / n_chars, 4),
            "pages": _fmt_pages(_pages_of(scaffold)),
            "example": "p%d offset %d" % (scaffold[0]["page"],
                                          scaffold[0]["start"]),
            "owner": "#17 page_tails.tsv (tail spans); this ledger (all spans)",
            "verdict": VERDICT_DISCLOSED,
            "note": "Marker's figure-block placeholders. The tokenizer emits "
                    "no tokens inside them, so they survive only as gap text; "
                    "#17's spec discloses them qualitatively ('whitespace only "
                    "except for the two figure-block comments') and never "
                    "lists them. Six of the twelve sit between tokens."})
    punct_other = punct_other_ledger(book)
    if punct_other:
        rows.append({
            "class": "punctuation in no operator class",
            "level": "character",
            "item": ", ".join("%s x%d" % (repr(k), v)
                              for k, v in sorted(punct_other.items())),
            "n": sum(punct_other.values()),
            "share_pct": round(100.0 * sum(punct_other.values()) / n_chars, 4),
            "pages": _fmt_pages(sorted({int(r["page"]) for r in
                                        _punct_pages(book)})),
            "example": "`#` is the markdown heading marker; #12/#14 model only "
                       "the book's own operators",
            "owner": "#17 PUNCT tokens (pclass `other`); #47 strips them "
                     "(197 chars) before its char stream",
            "verdict": VERDICT_EXPLAINED,
            "note": "Counted, and none of it is deleted: #47's char total "
                    "(480,309) is this ledger's raw total (480,962) less the "
                    "468 scaffolding chars, less the 197 marker chars, plus 12 "
                    "replacement spaces."})

    # ---- L3 operators --------------------------------------------------
    unbound = {r: ops.get(r, 0) for r in ROLE_UNBOUND}
    n_unbound = sum(unbound.values())
    rows.append({
        "class": "operator tokens that bind nothing",
        "level": "operator token",
        "item": ", ".join("%s %d" % (r, n) for r, n in unbound.items() if n),
        "n": n_unbound,
        "share_pct": round(100.0 * n_unbound / max(1, sum(ops.values())), 3),
        "pages": "",
        "example": "; ".join("%s: %s" % (r, ", ".join(
            "%s x%d" % (repr(t), c) for t, c in sorted(
                ops_by_op.get(r, {}).items(), key=lambda kv: -kv[1])))
            for r in ROLE_UNBOUND if ops_by_op.get(r)),
        "owner": "#17 diagnostics.tsv (seam_orphan_* / slash_free / "
                 "slash_boundary rows)",
        "verdict": VERDICT_EXPLAINED,
        "note": "A seam operator with no word on one side, or a `/` that is "
                "not glued to the right. Kept as OP tokens with exact "
                "offsets, never repaired: the tokenizer deletes nothing."})

    # ---- L4 units ------------------------------------------------------
    markup_pages = _pages_of(markup_rows)
    composition = _markup_composition(markup_rows)
    rows.append({
        "class": "markup-residue unit occurrences",
        "level": "unit occurrence",
        "item": "#19 markup ledger, uid level: %s" % ", ".join(
            "%s x%d" % (r["unit"], r["n"]) for r in composition),
        "n": unit_residue["markup_ledger_occurrence"],
        "share_pct": round(100.0 * unit_residue["markup_ledger_occurrence"]
                           / EXPECT_UNITS, 4),
        "pages": _fmt_pages(markup_pages),
        "example": "; ".join("%s x%d" % (r["unit"], r["n"])
                             for r in composition),
        "owner": "#19 markup_residue.tsv (347 rows, one per occurrence)",
        "verdict": VERDICT_EXPLAINED,
        "note": "Every occurrence is published with its uid, so the removal "
                "is per-occurrence and auditable; a surface-level join "
                "cannot reproduce it."})
    if markup_only:
        rows.append({
            "class": "markup-only unit types",
            "level": "unit type",
            "item": ", ".join("%s x%d" % (r["unit"], r["n"])
                              for r in markup_only),
            "n": sum(r["n"] for r in markup_only),
            "share_pct": round(100.0 * sum(r["n"] for r in markup_only)
                               / EXPECT_UNITS, 4),
            "pages": _fmt_pages({p for r in markup_only for p in r["pages"]}),
            "example": "p%d offset 995: unit 'th' inside '<th>'"
                       % markup_only[0]["pages"][0],
            "owner": "#19 counts.markup_only_types_dropped (8) + this ledger "
                     "per type",
            "verdict": VERDICT_EXPLAINED,
            "note": "Surfaces whose every occurrence is markup, so the type "
                    "leaves the 5,876-type vocabulary entirely."})
    if mixed:
        rows.append({
            "class": "markup occurrences on a surviving surface",
            "level": "unit occurrence",
            "item": ", ".join("%s x%d of %d" % (r["unit"], r["markup_n"],
                                                r["n"]) for r in mixed),
            "n": sum(r["markup_n"] for r in mixed),
            "share_pct": round(100.0 * sum(r["markup_n"] for r in mixed)
                               / EXPECT_UNITS, 4),
            "pages": _fmt_pages({p for r in mixed for p in r["pages"]}),
            "example": "p92: the `a` of `</a>` inside a mailto anchor",
            "owner": "#19 markup_residue.tsv (uid level)",
            "verdict": VERDICT_EXPLAINED,
            "note": "The hazard this ledger exists for: #17's unit_types.tsv "
                    "advertises `a` 1,126 times on 214 pages, but 201 of "
                    "those are tag names. Only the uid join removes them."})
    junk = [r for r in screening if r["category"] == "junk"]
    if junk:
        rows.append({
            "class": "tag furniture kept as English by the screen",
            "level": "unit occurrence",
            "item": ", ".join("%s x%d" % (r["unit"], int(r["n"]))
                              for r in sorted(junk, key=lambda r: -int(r["n"]))),
            "n": sum(int(r["n"]) for r in junk),
            "share_pct": round(100.0 * sum(int(r["n"]) for r in junk)
                               / EXPECT_UNITS, 4),
            "pages": _fmt_pages({int(r["first_page"]) for r in junk}),
            "example": "`href` x%d and `mailto` x%d are wordfreq-recognised "
                       "English words, so #19 classes them `junk` but keeps "
                       "them" % (int(junk[0]["n"]), int(junk[1]["n"])),
            "owner": "#19 screening.tsv category `junk`; #27 counts."
                     "markup_extra_removed (241) removes them again",
            "verdict": VERDICT_EXPLAINED,
            "note": "Two artifacts disagree about the same 202 occurrences by "
                    "design: #19 publishes them, #27 screens them out."})
    if furniture:
        attr = sum(v for k, v in furniture.items() if k in TAG_ATTR_FURNITURE)
        value = sum(v for k, v in furniture.items() if k in TAG_VALUE_FURNITURE)
        rows.append({
            "class": "attribute names and values inside tag runs",
            "level": "unit occurrence",
            "item": ", ".join("%s x%d" % (repr(k), v) for k, v
                              in sorted(furniture.items())),
            "n": attr + value,
            "share_pct": round(100.0 * (attr + value) / EXPECT_UNITS, 4),
            "pages": _fmt_pages(tag_run_pages),
            "example": "`<th colspan=\"2\">`: the attribute name (%d) and its "
                       "value (%d); `<a href=\"mailto:...\">`: the name (%d) "
                       "and the value (%d)"
                       % (furniture.get("colspan", 0), furniture.get("2", 0),
                          furniture.get("href", 0), furniture.get("mailto", 0)),
            "owner": "#27 find_markup_spans (the general tag-run detector)",
            "verdict": VERDICT_EXPLAINED,
            "note": "Invisible to #19's screen, which matches whole tags and "
                    "not the runs they sit in. The %d `href`/`mailto` "
                    "occurrences overlap the `junk` row above (the same "
                    "occurrences seen from #19's side); the %d "
                    "`colspan`/`2` occurrences are #27's alone."
                    % (attr, value)})
    if book_text:
        rows.append({
            "class": "book text deleted by the tag-run screen",
            "level": "unit occurrence",
            "item": ", ".join("%s x%d" % (k, v) for k, v in sorted(
                collections.Counter(r["unit"] for r in book_text).items())),
            "n": len(book_text),
            "share_pct": round(100.0 * len(book_text) / EXPECT_UNITS, 4),
            "pages": _fmt_pages(_pages_of(book_text)),
            "example": "p158 offset 2382: %r" % book_text[0]["context"][:70],
            "owner": "no artifact; itemized by this ledger",
            "verdict": VERDICT_DISCLOSED,
            "note": "The only place in the workspace where a screen removes "
                    "real text rather than markup: the OCR wrapped printed "
                    "book text in `<a href=\"...\">`, and #27's span rule "
                    "deletes the anchor's href value with the tag. On p219 "
                    "the deleted text is a duplicate of the visible anchor "
                    "text that follows, and on p158 likewise; whether these "
                    "13 occurrences should be restored (span narrowing, not "
                    "a wider screen) is #6/#19's call."})
    if escape_fixed:
        rows.append({
            "class": "artifact escape mismatch",
            "level": "unit occurrence",
            "item": ", ".join("%s (p%s, uid %s)"
                              % (r["unit"], r["page"], r["uid"])
                              for r in escape_fixed),
            "n": len(escape_fixed),
            "share_pct": round(100.0 * len(escape_fixed) / EXPECT_UNITS, 4),
            "pages": _fmt_pages(_pages_of(escape_fixed)),
            "example": "; ".join("%s (p%s, uid %s)"
                                 % (r["unit"], r["page"], r["uid"])
                                 for r in escape_fixed),
            "owner": "this ledger's normalisation; the mismatch is #19's "
                     "writer",
            "verdict": VERDICT_DISCLOSED,
            "note": "Two unit surfaces cross a newline after a seam. #17 "
                    "writes the surface escaped once; #19 reads it back with "
                    "`csv.DictReader` (no unescaping) and writes it with a "
                    "writer that escapes the backslash again, so "
                    "`screening.tsv` stores two backslashes where `units.tsv` "
                    "stores one. A surface-level join across the two "
                    "artifacts loses exactly these %d occurrences, which is "
                    "the concrete reason #17 says to join on `uid`. This "
                    "ledger normalises the escape and closes the level "
                    "without it." % len(escape_fixed)})
    ocr = [r for r in screening if r["category"] == "ocr_error"]
    if ocr:
        rows.append({
            "class": "OCR-error units kept in the unit layer",
            "level": "unit occurrence",
            "item": "%d types, %s" % (len(ocr), ", ".join(
                r["unit"] for r in sorted(ocr, key=lambda r: -int(r["n"]))[:6])),
            "n": sum(int(r["n"]) for r in ocr),
            "share_pct": round(100.0 * sum(int(r["n"]) for r in ocr)
                               / EXPECT_UNITS, 4),
            "pages": _fmt_pages({int(r["first_page"]) for r in ocr}),
            "example": "kept, not corrected: #6 owns the correction",
            "owner": "#6 corrections + #19 category `ocr_error`",
            "verdict": VERDICT_EXPLAINED,
            "note": "Residue that is *kept* rather than dropped: #6's fixes "
                    "are applied in the lexical-correction layer, not in the "
                    "unit layer, so every downstream statistic still sees "
                    "them."})
    unexplained_chars = chars.get("gap_other", 0)
    if unexplained_chars or unit_residue["unscreened_surface"]:
        rows.append({
            "class": "unexplained residue",
            "level": "mixed",
            "item": "%d gap characters, %d unscreened unit surfaces"
                    % (unexplained_chars, unit_residue["unscreened_surface"]),
            "n": unexplained_chars + unit_residue["unscreened_surface"],
            "share_pct": 0.0, "pages": "", "example": "", "owner": "",
            "verdict": VERDICT_UNEXPLAINED,
            "note": "The ledger did not close; the build must fail."})
    return rows


# ---------------------------------------------------------------------------
# reconciliation against the artifacts this ledger reads
# ---------------------------------------------------------------------------
def entropy_char_total(bodies, chars) -> tuple:
    """Reproduce #47's raw char total from this ledger's raw total.

    #47 replaces every layout comment with one space, strips the markdown
    heading markers and case-folds; nothing else changes, so its published
    total must be `raw - comment_chars + comment_spans - marker_chars`.
    Returns (total, marker_chars, comment_spans).
    """
    comment_chars = 0
    comment_spans = 0
    marker_chars = 0
    for body in bodies.values():
        hits = list(_COMMENT_RE.finditer(body))
        comment_chars += sum(h.end() - h.start() for h in hits)
        comment_spans += len(hits)
        marker_chars += sum(m.end() - m.start()
                            for m in _HEADER_RE.finditer(body))
    total = (sum(chars.values()) - comment_chars + comment_spans
             - marker_chars)
    return total, marker_chars, comment_spans


def reconciliation_rows(bodies, chars, tokens, ops, cats, unit_residue,
                        markup_only, furniture, book_text, scaffold) -> list:
    """Every count this ledger re-derives, against the issue that published
    it.  A `MISMATCH` here fails the build."""
    tok = read_json(ROOT / "analysis" / "tokenizer" / "summary.json")
    ent_total, marker_chars, comment_spans = entropy_char_total(bodies, chars)
    out = [
        ("pages", EXPECT_PAGES, tok["n_pages"], "#7 / #17"),
        ("body characters", EXPECT_BODY_CHARS, sum(chars.values()),
         "sum of #7 page-body lengths"),
        ("tokens", EXPECT_TOKENS, sum(tokens.values()), "#14 / #17"),
        ("word tokens", EXPECT_WORDS, tokens["token_WORD"],
         "#14 Lex nodes / #22"),
        ("operator tokens", EXPECT_OPS, tokens["token_OP"],
         "#14 10,949 body + 51 header"),
        ("punctuation tokens", EXPECT_PUNCT, tokens["token_PUNCT"], "#17"),
        ("operator role `seam`", EXPECT_SEAMS, ops[ROLE_SEAM], "#17"),
        ("operator role `segment`", EXPECT_SEGMENTS, ops[ROLE_SEGMENT], "#17"),
        ("unbound operator tokens", EXPECT_UNBOUND,
         sum(ops.get(r, 0) for r in ROLE_UNBOUND), "#17 unbound_ops"),
        ("unit occurrences", EXPECT_UNITS, EXPECT_UNITS, "#17 units.total"),
        ("screened unit occurrences", EXPECT_SCREENED, sum(cats.values()),
         "#19 counts.screened_occurrences"),
        ("markup-ledger occurrences", EXPECT_MARKUP,
         unit_residue["markup_ledger_occurrence"],
         "#19 counts.markup_total_occurrences"),
        ("markup-only unit types", EXPECT_MARKUP_TYPES, len(markup_only),
         "#19 counts.markup_only_types_dropped"),
        ("tag-run extra removals", EXPECT_TAG_RUN,
         sum(furniture.values()) + len(book_text),
         "#27 counts.markup_extra_removed"),
        ("book text inside tag runs", EXPECT_BOOK_TEXT_IN_SPANS,
         len(book_text), "this ledger (unpublished elsewhere)"),
        ("scaffolding comment chars", EXPECT_SCAFFOLD_CHARS,
         chars["layout_comment"], "#17 page_tails.tsv + gaps"),
        ("scaffolding comment spans", EXPECT_SCAFFOLD_SPANS,
         comment_spans, "#17 spec, qualitative only"),
        ("#47 raw char total", EXPECT_ENTROPY_CHARS, ent_total,
         "#47 reconciliation.tsv (comments -> space, markers stripped)"),
        ("#47 heading-marker chars", EXPECT_HEADER_MARKER_CHARS,
         marker_chars, "#47 clean_page()"),
    ]
    rows = []
    for quantity, expected, here, owner in out:
        rows.append({"quantity": quantity, "earlier": "{:,}".format(expected),
                     "here": "{:,}".format(here), "owner": owner,
                     "match": "ok" if expected == here
                              else "MISMATCH (%d != %d)" % (expected, here)})
    if any(r["match"] != "ok" for r in rows):
        raise LedgerError("reconciliation mismatch: " + "; ".join(
            "%s %s" % (r["quantity"], r["match"])
            for r in rows if r["match"] != "ok"))
    return rows


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def _cell(value) -> str:
    text = "" if value is None else str(value)
    return (text.replace("\\", "\\\\").replace("\t", "\\t")
                .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(name: str, header: list, rows: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(_cell(row.get(h, "")) for h in header) + "\n")


COVERAGE_HEADER = ["level", "bucket", "kind", "n", "share_pct", "status",
                   "owner"]

BUCKET_STATUS = {
    # char buckets
    "token_word": ("book text", "#17 WORD tokens"),
    "token_op": ("book text", "#17 OP tokens (#12/#14 operators)"),
    "gap_whitespace": ("layout", "whitespace between tokens"),
    "layout_comment": ("residue", "marker figure-block scaffolding"),
    "gap_other": ("UNEXPLAINED", "nothing"),
    # token buckets
    "token_WORD": ("book text", "#14 Lex nodes"),
    "token_OP": ("book text", "#10/#12 operator inventory"),
    "token_PUNCT": ("book text", "ordinary punctuation"),
}
for _cls in ("dash", "colon", "bracket", "quote", "period", "comma", "bang",
             "semi", "apos", "query", "other"):
    BUCKET_STATUS["token_punct_" + _cls] = (
        "book text", "#17 pclass `%s`" % _cls)


def coverage_table(chars, tokens, ops, cats, unit_residue) -> list:
    """The ledger itself: every bucket at every level, closing to its total."""
    rows = []
    for bucket in sorted(chars):
        status, owner = BUCKET_STATUS.get(bucket, ("UNEXPLAINED", "nothing"))
        rows.append({"level": "L1 characters", "bucket": bucket,
                     "kind": status, "n": chars[bucket],
                     "share_pct": round(100.0 * chars[bucket]
                                        / max(1, sum(chars.values())), 4),
                     "status": "accounted" if status != "UNEXPLAINED"
                               else "unexplained",
                     "owner": owner})
    for bucket in sorted(tokens):
        rows.append({"level": "L2 tokens", "bucket": bucket,
                     "kind": BUCKET_STATUS.get(bucket, ("book text", "#17"))[0],
                     "n": tokens[bucket],
                     "share_pct": round(100.0 * tokens[bucket]
                                        / max(1, sum(tokens.values())), 4),
                     "status": "accounted",
                     "owner": BUCKET_STATUS.get(bucket, ("", "#17"))[1]})
    for role in sorted(ops):
        rows.append({"level": "L3 operator tokens", "bucket": role,
                     "kind": "operator role",
                     "n": ops[role],
                     "share_pct": round(100.0 * ops[role]
                                        / max(1, sum(ops.values())), 4),
                     "status": "accounted",
                     "owner": "#17 operator_tokenizer._is_seam"})
    for category in sorted(cats):
        rows.append({"level": "L4 unit occurrences", "bucket": category,
                     "kind": "unit category", "n": cats[category],
                     "share_pct": round(100.0 * cats[category] / EXPECT_UNITS, 4),
                     "status": "accounted",
                     "owner": "#19 screening.tsv"})
    for bucket in sorted(unit_residue):
        rows.append({"level": "L4 unit occurrences", "bucket": bucket,
                     "kind": "unit residue", "n": unit_residue[bucket],
                     "share_pct": round(100.0 * unit_residue[bucket]
                                        / EXPECT_UNITS, 4),
                     "status": "unexplained" if bucket == "unscreened_surface"
                               else "accounted",
                     "owner": "#19 markup_residue.tsv"
                              if bucket == "markup_ledger_occurrence"
                              else "nothing"})
    return rows


# ---------------------------------------------------------------------------
# closure assertions
# ---------------------------------------------------------------------------
def check_types(units, screening, markup_only, tag_run_total,
                expected_types=EXPECT_UNIT_TYPES,
                expected_screened=EXPECT_SCREEN_TYPES,
                expected_full_screen=EXPECT_FULL_SCREEN,
                expected_markup=EXPECT_MARKUP,
                against_artifact: bool = True) -> dict:
    """Type-level closure and the arithmetic that links the three screens.

    5,876 raw types = 5,868 screened + 8 markup-only, and 74,665 raw
    occurrences - 347 (#19's ledger) - 241 (#27's tag runs) = 74,077, the count
    every full-screen layer publishes.  Both are this ledger's own invariants,
    so they are checked rather than assumed; the expectations are parameters so
    a synthetic corpus can exercise the same closure.
    """
    surfaces = {u["unit"] for u in units}
    problems = []
    if against_artifact:
        published = {r["unit_type"] for r in read_rows(UNIT_TYPES)}
        if published != surfaces:
            problems.append("unit_types.tsv and units.tsv disagree on %d "
                            "surfaces" % len(published ^ surfaces))
    if len(surfaces) != expected_types:
        problems.append("unit types %d != %d" % (len(surfaces),
                                                 expected_types))
    if len(screening) != expected_screened:
        problems.append("screened types %d != %d" % (len(screening),
                                                     expected_screened))
    if len(screening) + len(markup_only) != len(surfaces):
        problems.append("type level does not close: %d + %d != %d"
                        % (len(screening), len(markup_only), len(surfaces)))
    full_screen = len(units) - expected_markup - tag_run_total
    if full_screen != expected_full_screen:
        problems.append("full screen %d != %d" % (full_screen,
                                                 expected_full_screen))
    if problems:
        raise LedgerError("the type level did not close: " + "; ".join(problems))
    return {"unit_types": len(surfaces), "screened_types": len(screening),
            "markup_only_types": len(markup_only),
            "full_screen_occurrences": full_screen}


def check_closed(chars, tokens, ops, cats, unit_residue, residue_rows) -> dict:
    """Every level must close, and the residue ledger must not carry an
    `unexplained` verdict.  Returns the per-level totals for the summary."""
    char_total = sum(chars.values())
    unexplained_chars = chars.get("gap_other", 0)
    problems = []
    if char_total != EXPECT_BODY_CHARS:
        problems.append("character total %d != %d" % (char_total,
                                                    EXPECT_BODY_CHARS))
    if unexplained_chars:
        problems.append("%d gap characters in no bucket" % unexplained_chars)
    if sum(tokens.values()) != EXPECT_TOKENS:
        problems.append("token total %d != %d" % (sum(tokens.values()),
                                                 EXPECT_TOKENS))
    if sum(ops.values()) != EXPECT_OPS:
        problems.append("operator total %d != %d" % (sum(ops.values()),
                                                    EXPECT_OPS))
    if sum(cats.values()) + sum(unit_residue.values()) != EXPECT_UNITS:
        problems.append("unit occurrences do not close: %d + %d != %d"
                        % (sum(cats.values()), sum(unit_residue.values()),
                           EXPECT_UNITS))
    unscreened = unit_residue.get("unscreened_surface", 0)
    if unscreened:
        problems.append("%d unit occurrences match no screening row"
                        % unscreened)
    unexplained = [r for r in residue_rows
                   if r["verdict"] == VERDICT_UNEXPLAINED]
    if unexplained or unexplained_chars:
        problems.append("%d residue rows and %d characters are unexplained: %s"
                        % (len(unexplained), unexplained_chars,
                           "; ".join(r["class"] for r in unexplained)))
    if problems:
        raise LedgerError("the ledger did not close: " + "; ".join(problems))
    return {"characters": char_total, "tokens": sum(tokens.values()),
            "operator_tokens": sum(ops.values()),
            "unit_occurrences": (sum(cats.values())
                                 + sum(unit_residue.values())),
            "residue_rows": len(residue_rows),
            "residue_by_level": _residue_by_level(residue_rows),
            "residue_occurrences": sum(r["n"] for r in residue_rows
                                       if r["level"] == "unit occurrence")}


def _residue_by_level(residue_rows) -> dict:
    """Residue occurrences per level; levels are not added together."""
    out = collections.Counter()
    for r in residue_rows:
        out[r["level"]] += int(r["n"])
    return dict(out)


# ---------------------------------------------------------------------------
# the deliverable
# ---------------------------------------------------------------------------
def write_report(path: Path, cov, coverage_table_rows, residue_rows,
                 consumers, recon_rows, totals, book_text) -> None:
    by_level = collections.Counter()
    for r in residue_rows:
        by_level[r["level"]] += int(r["n"])
    L = []
    A = L.append
    A("# Coverage accounting / residue ledger (#70)\n")
    A("Does the analysis account for every token and every operator, or does "
      "it quietly drop the awkward residue?  One ledger is built over the "
      "whole corpus at four levels, and each level is forced to close.\n")
    A("**Verdict: it accounts for everything, and the residue is itemized.** "
      "Every one of the 480,962 page-body characters, all 90,220 tokens, all "
      "11,000 operator tokens and all 74,665 unit occurrences land in a named "
      "bucket, so the coverage metric is 1.000000 at every level and the "
      "build fails if it ever drops.  The awkward part is therefore not "
      "hidden residue but *named* residue: %d classes across the levels -- "
      "%s characters of layout scaffolding and unclassifiable punctuation, "
      "%s operator tokens that bind nothing, and %s unit-occurrence entries "
      "(one class deliberately overlaps another: the 202 `href`/`mailto` "
      "occurrences are #19 `junk` *and* #27 tag-run furniture).  Two classes "
      "are itemized by no artifact until this ledger: the 13 occurrences of "
      "real book text that #27's tag-run screen deletes, and the 2 unit "
      "surfaces that #17 and #19 store with a different number of "
      "backslashes.\n"
      % (len(residue_rows),
         "{:,}".format(by_level.get("character", 0)),
         "{:,}".format(by_level.get("operator token", 0)),
         "{:,}".format(by_level.get("unit occurrence", 0)
                       + by_level.get("unit type", 0))))
    A("## 1. The coverage metric\n")
    A("| level | total | covered | unexplained | coverage |")
    A("|---|---:|---:|---:|---:|")
    for r in cov:
        A("| %s | %s | %s | %s | %.6f |"
          % (r["level"], "{:,}".format(r["total"]),
             "{:,}".format(r["covered"]), "{:,}".format(r["unexplained"]),
             r["coverage"]))
    A("")
    A("Coverage counts a bucket as covered when the ledger can *name* it, "
      "including the residue classes an artifact owns: naming a removal is "
      "what makes it auditable.  Only the unnamed and the unclaimed count "
      "against coverage, and on this corpus there are none.  The metric is a "
      "closure test, not a score -- a level that silently lost 0.1% of its "
      "items would still read 0.999000, which is why the tests plant residue "
      "and require the detector to fire (`work/test_coverage_ledger.py`).\n")

    A("## 2. What the buckets are\n")
    A("### L1 characters -- %s page-body characters\n"
      % "{:,}".format(totals["characters"]))
    A("| bucket | chars | share | status |")
    A("|---|---:|---:|---|")
    for r in coverage_table_rows:
        if r["level"] != "L1 characters":
            continue
        A("| `%s` | %s | %.3f%% | %s |"
          % (r["bucket"], "{:,}".format(r["n"]), r["share_pct"],
             r["status"]))
    A("")
    A("The page body is reconstructed from #17's artifacts alone "
      "(`\"\".join(gap + text)` + `page_tails.tsv`), and both directions are "
      "asserted before the ledger is built: the reconstruction equals #7's "
      "layout markdown on all 225 pages, and #17's `tokens.tsv` equals what "
      "`operator_tokenizer.tokenize_page` emits.  A ledger over a drifting "
      "stream would be worthless.\n")
    A("The only non-token, non-whitespace characters in the book are the %d "
      "figure-block scaffolding comments (%d chars on pages %s).  #47's char "
      "total (480,309) is reconcilable rather than merely different: it "
      "replaces each comment with one space (468 chars -> 12), strips the "
      "markdown heading markers (197 chars) and case-folds.\n"
      % (EXPECT_SCAFFOLD_SPANS, EXPECT_SCAFFOLD_CHARS, "1, 224, 225"))
    A("### L2/L3 tokens and operator roles\n")
    A("| level | bucket | n | share |")
    A("|---|---|---:|---:|")
    for r in coverage_table_rows:
        if r["level"] in ("L2 tokens", "L3 operator tokens"):
            A("| %s | `%s` | %s | %.3f%% |"
              % (r["level"], r["bucket"], "{:,}".format(r["n"]),
                 r["share_pct"]))
    A("")
    A("The token partition reproduces #14's 77,485 Lex nodes and #17's "
      "11,000 operators (10,949 body + 51 header).  Every one of the 11,000 "
      "operator tokens carries exactly one #17 role, so the operator level "
      "closes by construction: 2,816 seams and 7,247 segment operators that "
      "do work, and %d that bind nothing.\n" % EXPECT_UNBOUND)
    A("### L4 unit occurrences\n")
    A("| bucket | occurrences | share |")
    A("|---|---:|---:|")
    for r in coverage_table_rows:
        if r["level"] == "L4 unit occurrences":
            A("| `%s` | %s | %.3f%% |"
              % (r["bucket"], "{:,}".format(r["n"]), r["share_pct"]))
    A("")
    A("74,318 occurrences carry a #19 category and 347 sit in #19's "
      "per-occurrence markup ledger: `74,318 + 347 = 74,665`.  At the type "
      "level `5,868 + 8 = 5,876`.  Both directions close, and neither closes "
      "on a surface-only join -- that is the point of §3.\n")

    A("## 3. The residue ledger\n")
    A("| class | level | n | share | verdict | owner |")
    A("|---|---|---:|---:|---|---|")
    for r in residue_rows:
        A("| %s | %s | %s | %.4f%% | **%s** | %s |"
          % (r["class"], r["level"], "{:,}".format(r["n"]), r["share_pct"],
             r["verdict"], r["owner"]))
    A("")
    for r in residue_rows:
        pages_note = ("pages: " + r["pages"]) if r["pages"] else "no page list"
        A("- **%s** (%s, %s occurrences). %s  \n  %s  \n  _%s; %s_"
          % (r["class"], r["level"], "{:,}".format(r["n"]), r["item"],
             r["note"], r["verdict"], pages_note))
    A("")
    A("`explained` means a published artifact names the items; `disclosed` "
      "means the workspace counts them but itemizes them nowhere.  There is "
      "no `unexplained` row, and the build fails if one ever appears.\n")

    A("## 4. The one screen that deletes book text\n")
    A("#27 removes #19's 347 ledger occurrences and 241 more through its "
      "general tag-run detector.  The ledger decomposes those 241: 228 are "
      "tag furniture (`href` 102, `mailto` 100, `colspan` 13 and the "
      "`colspan` value `2` 13) and **13 are real book text** that the OCR "
      "wrapped in a fabricated anchor:\n")
    A("| page | offset | unit | context |")
    A("|---:|---:|---|---|")
    for r in book_text:
        A("| %s | %s | `%s` | `%s` |"
          % (r["page"], r["start"], r["unit"],
             r["context"].replace("`", "'")[:78]))
    A("")
    A("The printed page reads `<<a href=\"atrocity genomics of "
      "retro-ADAM\">atrocity genomics of retro-ADAM` (p158) and `<<a "
      "href=\"mailto:konganic-eye of cyberBuddha\">konganic-eye of "
      "cyberBuddha` (p219, three times): the anchor's href *value* is a copy "
      "of the visible text that follows, which is why deleting it is "
      "defensible.  But the deletion is never itemized, and the 74,318 vs "
      "74,077 gap that #27 publishes as \"241 tag-furniture occurrences\" is "
      "13 text occurrences larger than that description.  The fix is span "
      "narrowing (screen the tag, keep the href value) and it is #6/#19's "
      "call, not this ledger's.\n")

    A("## 5. Every downstream layer's split\n")
    A("| issue | layer | level total | kept | dropped | kept % | drop published in |")
    A("|---|---|---:|---:|---:|---:|---|")
    for r in consumers:
        A("| %s | %s | %s | %s | %s | %.3f%% | %s |"
          % (r["issue"], r["layer"], "{:,}".format(r["level_total"]),
             "{:,}".format(int(r["kept"])), "{:,}".format(int(r["dropped"])),
             r["share_kept"], r["drop_published_in"]))
    A("")
    A("Each layer keeps a different share of the same 74,665 units, and each "
      "records its own drop.  The ledger's contribution is that the "
      "arithmetic is now checked in one place: `kept + dropped = level "
      "total` for all %d rows, and a layer that changed its split without "
      "saying so fails this build.\n" % len(consumers))

    A("## 6. Reconciliation\n")
    A("Every count this ledger re-derives, against the issue that published "
      "it.  A mismatch fails the build.\n")
    A("| quantity | published | here | owner | match |")
    A("|---|---:|---:|---|---|")
    for r in recon_rows:
        A("| %s | %s | %s | %s | %s |"
          % (r["quantity"], r["earlier"], r["here"], r["owner"], r["match"]))
    A("")

    A("## 7. Method\n")
    A("Inputs are frozen artifacts only -- #7's layout markdown, #17's "
      "token/unit streams, #19's screen and markup ledger, #27's tag-run "
      "detector, and the published `summary.json` of every downstream layer. "
      "No OCR is run, nothing is downloaded and no reference corpus is "
      "consulted, so this ledger adds no new external dependency to the "
      "program (#78-#80 stay parked).  Character buckets come from one pass "
      "over each page body: layout comments win over tokens (the tokenizer "
      "emits nothing inside them), then token text by kind and #17's "
      "punctuation class, then the gap.  Operator roles are re-derived with "
      "#17's own `_is_seam` rule rather than read from its summary, so the "
      "published partition is tested, not assumed.\n")

    A("## 8. Limits\n")
    A("- **The ledger names; it does not repair.**  Nothing here corrects a "
      "corpus, changes a screen or re-opens a closed decision.  Where the "
      "verdict is `disclosed` the recommendation is an artifact, not a fix.")
    A("- **`explained` is not `correct`.**  #19 keeping `href`/`mailto` as "
      "`junk` and #27 removing them are both defensible; the ledger only "
      "makes the disagreement visible in one place.")
    A("- **Coverage is level-local.**  A level at 1.000000 says nothing "
      "about whether the *reading* built on it is right; that is #73's "
      "triangulation and #56's viability assessment.")
    A("- **The figure-only pages stay in.**  Pages 1, 224 and 225 carry 10, "
      "2 and 2 tokens; every per-page statistic includes them, and this "
      "ledger does not decide whether it should.\n")

    A("## 9. Artifacts\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `coverage_ledger_report.md` | THE DELIVERABLE |")
    A("| `coverage.tsv` | the ledger: every bucket at every level |")
    A("| `residue_ledger.tsv` | the awkward residue, itemized and verdicted |")
    A("| `per_page.tsv` | per-page accounting (where the residue lives) |")
    A("| `consumer_reconciliation.tsv` | every downstream layer's split |")
    A("| `reconciliation.tsv` | this ledger vs #7/#14/#17/#19/#27/#47 |")
    A("| `diagnostics.tsv` | build counters |")
    A("| `summary.json` | machine-readable headline numbers |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(out=Path(OUT), quiet=False) -> dict:
    """Build the ledger and every artifact under `out`."""
    global OUT
    OUT = Path(out)
    bodies = page_bodies()
    if len(bodies) != EXPECT_PAGES:
        raise LedgerError("expected %d pages, #7 gives %d"
                          % (EXPECT_PAGES, len(bodies)))
    book = check_stream(bodies)

    chars, _examples = char_ledger(book, bodies)
    tokens = token_ledger(book)
    ops, ops_by_op = operator_ledger(book)

    units = read_rows(UNITS)
    screening = read_rows(SCREENING)
    markup_rows = read_rows(MARKUP)
    markup_uids = {int(r["uid"]) for r in markup_rows}
    cats, unit_residue, _detail, escape_fixed = unit_ledger(units, screening,
                                                           markup_uids)
    markup_only, mixed = type_ledger(units, screening, markup_uids)
    furniture, book_text, tag_run_pages = tag_run_detail()
    scaffold = scaffold_spans(book, bodies)
    consumers = consumer_ledger()
    per_page = per_page_ledger(book, bodies, units, book_text, tag_run_pages)
    cov = coverage_rows(chars, tokens, ops, unit_residue["unscreened_surface"])
    residue_rows = residue_ledger(book, bodies, chars, ops, ops_by_op, cats,
                                  unit_residue, markup_only, mixed, furniture,
                                  book_text, tag_run_pages, screening,
                                  markup_rows, escape_fixed)
    totals = check_closed(chars, tokens, ops, cats, unit_residue, residue_rows)
    types = check_types(units, screening, markup_only,
                        sum(furniture.values()) + len(book_text))
    recon_rows = reconciliation_rows(bodies, chars, tokens, ops, cats,
                                     unit_residue, markup_only, furniture,
                                     book_text, scaffold)
    table = coverage_table(chars, tokens, ops, cats, unit_residue)

    write_tsv("coverage.tsv", COVERAGE_HEADER, table)
    write_tsv("residue_ledger.tsv",
              ["class", "level", "item", "n", "share_pct", "pages", "example",
               "owner", "verdict", "note"], residue_rows)
    write_tsv("per_page.tsv",
              ["page", "n_chars", "n_tokens", "n_words", "n_ops", "n_puncts",
               "n_units", "scaffold_chars", "scaffold_spans", "unbound_ops",
               "markup_ledger_units", "tag_run_units", "book_text_removed",
               "unexplained_chars"], per_page)
    write_tsv("consumer_reconciliation.tsv",
              ["issue", "layer", "level_total", "kept", "dropped",
               "share_kept", "drop_published_in", "closes"], consumers)
    write_tsv("reconciliation.tsv",
              ["quantity", "earlier", "here", "owner", "match"], recon_rows)
    write_tsv("diagnostics.tsv", ["diag", "value"], [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "pages", "value": len(bodies)},
        {"diag": "body_chars", "value": totals["characters"]},
        {"diag": "unexplained_chars", "value": chars.get("gap_other", 0)},
        {"diag": "unbound_operator_tokens", "value": EXPECT_UNBOUND},
        {"diag": "markup_only_types", "value": len(markup_only)},
        {"diag": "tag_run_furniture", "value": sum(furniture.values())},
        {"diag": "book_text_in_tag_runs", "value": len(book_text)},
        {"diag": "residue_rows", "value": totals["residue_rows"]},
        {"diag": "residue_unit_occurrences",
         "value": totals["residue_occurrences"]},
        {"diag": "residue_operator_tokens",
         "value": totals["residue_by_level"].get("operator token", 0)},
        {"diag": "residue_characters",
         "value": totals["residue_by_level"].get("character", 0)},
        {"diag": "unexplained_residue_rows",
         "value": sum(1 for r in residue_rows
                      if r["verdict"] == VERDICT_UNEXPLAINED)},
    ])
    summary = {
        "schema": SCHEMA,
        "pages": len(bodies),
        "levels": {
            "characters": totals["characters"],
            "tokens": totals["tokens"],
            "operator_tokens": totals["operator_tokens"],
            "unit_occurrences": totals["unit_occurrences"],
        },
        "coverage": cov,
        "unexplained": {
            "characters": chars.get("gap_other", 0),
            "tokens": 0,
            "operator_tokens": 0,
            "unit_occurrences": unit_residue["unscreened_surface"],
        },
        "buckets": {r["bucket"]: r["n"] for r in table},
        "residue": {
            "n_rows": totals["residue_rows"],
            "n_occurrences": totals["residue_occurrences"],
            "by_level": totals["residue_by_level"],
            "n_unexplained": sum(1 for r in residue_rows
                                 if r["verdict"] == VERDICT_UNEXPLAINED),
            "by_class": {r["class"]: r["n"] for r in residue_rows},
        },
        "consumers": consumers,
        "closure": dict(totals, types=types),
        "determinism": "summary.json carries no timestamp; a rebuild is "
                       "byte-identical except the `generated` line of the "
                       "report",
    }
    (OUT.mkdir(parents=True, exist_ok=True))
    with (OUT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    write_report(OUT / "coverage_ledger_report.md", cov, table, residue_rows,
                 consumers, recon_rows, totals, book_text)
    if not quiet:
        print("[#70] ledger closed: %s chars / %s tokens / %s operators / "
              "%s units; %d residue rows, %d unexplained"
              % ("{:,}".format(totals["characters"]),
                 "{:,}".format(totals["tokens"]),
                 "{:,}".format(totals["operator_tokens"]),
                 "{:,}".format(totals["unit_occurrences"]),
                 totals["residue_rows"],
                 summary["residue"]["n_unexplained"]))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        build(Path(args.out), quiet=args.quiet)
    except LedgerError as exc:
        print("coverage ledger failed: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
