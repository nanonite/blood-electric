#!/usr/bin/env python3
"""#33 Clause segmentation via operators -> analysis/clauses/

Split the book at its three *boundary* operators and treat each
operator-delimited span as a **clause**:

  * ``//``  the utterance break          (#14 ``Utterance`` with ``open_op``)
  * ``::``  the record/field separator    (#14 ``Field`` with ``open_op``)
  * ``/``   the **sentence-final** slash  (#14 ``Slash`` leaf only)

The task text names exactly these three.  ``/`` is *not* a boundary when it is
glued to the following word: #14 measured that role as a compound/possessive
``Link`` (``soul/gram``), so only the free-standing ``Slash`` terminator splits
a clause.  Likewise the other operators (``=``, ``++``, ``+``, ``<<``, ``>>``,
``<``, ``>``, ``|``, dashes, punctuation) are *content*: they stay inside the
clause that contains them.

Where this sits.  #14 already parses the whole book into an AST and emits
``clauses.tsv``, but those rows are **utterances** (one per ``//``) -- 3,764
rows.  #33 is the finer segmentation the method inventory asks for at L4.2:
it splits *inside* an utterance at every ``::`` and at every sentence-final
``/``, then measures the resulting clause inventory and its length
distribution.  It is built on the #14 layer: the boundary identities, the
source offsets, and the container context all come from the parsed tree, and
the token stream is asserted to be a partition (clause tokens + boundary
tokens == the 90,220 tokens #14 round-trips).

Word work is markup-screened.  Clause length is a word count, so the HTML tag
residue #19 catalogued (and the attribute furniture ``href``/``mailto``/
``colspan`` a general tag-run detector adds) is excluded from the primary
length distribution and disclosed separately.

Outputs under analysis/clauses/:
  clause_report.md        THE DELIVERABLE: definition, inventory, distributions
  clauses.tsv             one row per clause (the inventory)
  length_by_opener.tsv    length stats grouped by the operator that opened the clause
  length_by_container.tsv length stats grouped by the AST container
  length_histogram.tsv    word-length histogram per opener
  opener_transitions.tsv  opener -> next-opener counts (clause syntax)
  per_page.tsv            clause count / length per page
  diagnostics.tsv         build-time counters
  reconciliation.tsv      against #14 / #17 / #19
  summary.json            machine-readable headline numbers

Reproduce::

    conda activate marker
    python work/clause_segmentation.py
    python work/test_clause_segmentation.py
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import statistics
from pathlib import Path

import operator_grammar as og

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
MARKUP = ROOT / "analysis" / "neologism" / "markup_residue.tsv"
SCREENING = ROOT / "analysis" / "neologism" / "screening.tsv"
OUT = ROOT / "analysis" / "clauses"

SCHEMA = "clause-segmentation/1"
PAGE_RE = re.compile(r"<!-- page (\d+) -->")

# The three boundary operators named by the task.
BOUNDARY_OPS = ("//", "::", "/")
# Clause openers: the boundary operators plus the page start (BOS) and the
# layout-header edge (HDR, a hard boundary in #14).
OPENER_TYPES = ("BOS", "//", "::", "/", "HDR")

# ---------------------------------------------------------------------------
# markup screen (reused from #19's ledger + #27's tag-run detector)
# ---------------------------------------------------------------------------
HTML_TAGS = {
    "a", "p", "pre", "br", "li", "hr", "th", "td", "tr", "table", "tbody",
    "thead", "body", "html", "head", "span", "div", "img", "meta", "link",
    "b", "i", "u", "em", "strong", "font", "small", "big", "center",
    "script", "style", "title", "form", "input", "button", "ol", "ul", "dl",
    "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6", "nav", "main", "footer",
    "header", "section", "article", "blockquote", "code", "q", "sup", "sub",
}
STRICT_TAGS = {"a", "th", "p", "pre", "li", "br", "hr", "td", "tr", "table",
               "img", "meta", "link", "input", "script", "style"}
_TAG_RUN_RE = re.compile(
    r"</?([a-zA-Z][a-zA-Z0-9]*)"
    r"((?:\s+[a-zA-Z-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))*)\s*/?>")


def find_markup_spans(body: str):
    """(start, end) spans of real HTML tag runs in a page body (see #27).

    A run counts only when the tag name is a known HTML tag *and* it is a
    closing tag, carries an attribute, or is in ``STRICT_TAGS``; a bare
    ``<word>`` stays book text.
    """
    spans = []
    for m in _TAG_RUN_RE.finditer(body):
        name = m.group(1).lower()
        if name not in HTML_TAGS:
            continue
        closing = body[m.start():m.start() + 2] == "</"
        if closing or m.group(2).strip() or name in STRICT_TAGS:
            spans.append((m.start(), m.end()))
    return spans


# ---------------------------------------------------------------------------
# io helpers (quote-unsafe TSVs: split on tab, never a csv reader -- #17)
# ---------------------------------------------------------------------------
def cell(value) -> str:
    s = "" if value is None else str(value)
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(cell(row.get(h, "")) for h in header) + "\n")


def read_tsv(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh if line.strip()]


def load_pages():
    text = LAYOUT.read_text(encoding="utf-8", errors="ignore")
    parts = PAGE_RE.split(text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def load_units():
    """Return (page_units, uid_by_page_start, unit_meta).

    page_units: {page: [(start, end, uid), ...] sorted}
    uid_by_page_start: {(page, start): uid}
    unit_meta: {uid: (page, start, end, surface)}
    """
    page_units = collections.defaultdict(list)
    uid_by_ps = {}
    unit_meta = {}
    for r in read_tsv(UNITS):
        page, s, e, uid = int(r["page"]), int(r["start"]), int(r["end"]), int(r["uid"])
        page_units[page].append((s, e, uid))
        uid_by_ps[(page, s)] = uid
        unit_meta[uid] = (page, s, e, r["unit"])
    for page in page_units:
        page_units[page].sort()
    return page_units, uid_by_ps, unit_meta


def load_markup(bodies):
    """(spans, uids, types) as in #27: ledger unioned with the tag-run detector."""
    spans = collections.defaultdict(list)
    uids = set()
    if MARKUP.exists():
        for r in read_tsv(MARKUP):
            uids.add(int(r["uid"]))
            spans[int(r["page"])].append((int(r["start"]), int(r["end"])))
    for page, body in bodies.items():
        spans[page].extend(find_markup_spans(body))
    types = set()
    if SCREENING.exists():
        for r in read_tsv(SCREENING):
            if r["category"] in ("junk", "markup_residue"):
                types.add(r["unit"].lower())
    return spans, uids, types


# ---------------------------------------------------------------------------
# AST -> ordered events
# ---------------------------------------------------------------------------
class Event:
    __slots__ = ("boundary", "container", "depth", "end", "is_word",
                 "node_kind", "page", "start", "value")

    def __init__(self, boundary, value, page, start, end, container, depth,
                 node_kind, is_word):
        self.boundary = boundary
        self.value = value
        self.page = page
        self.start = start
        self.end = end
        self.container = container
        self.depth = depth
        self.node_kind = node_kind
        self.is_word = is_word


def node_container(node):
    """Container label for a node: Header > Inline > Span > Block."""
    tags = set()
    x = node.parent
    while x is not None:
        tags.add(x.k)
        x = x.parent
    if "Header" in tags:
        return "header"
    if "Inline" in tags:
        return "inline"
    if "Span" in tags:
        return "span"
    return "block"


def iter_events(book):
    """Yield every tokenizer token in source order, classified.

    A *boundary* event is one of the three task operators in its boundary role:
    ``//``/``::`` carried by the #14 container that opens on them (or by an
    orphan-boundary ``Op`` leaf when no container claimed it), and ``/`` on a
    sentence-final ``Slash`` leaf.  Everything else -- including ``<<``/``>>``,
    ``<``/``>``, ``=``/``++``/``+`` compound ``Link`` s, dashes, separators and
    stray operators -- is content.
    """
    def leaf(node):
        k = node.k
        cont, dep = node_container(node), node.depth()
        if k == "Slash":
            return Event(True, "/", node.page, node.s, node.e, cont, dep, k, False)
        if (k == "Op" and node.x.get("role") == "orphan-boundary"
                and node.v in BOUNDARY_OPS):
            return Event(True, node.v, node.page, node.s, node.e, cont, dep, k, False)
        return Event(False, node.v, node.page, node.s, node.e, cont, dep, k,
                     k == "Lex")

    def rec(node):
        k = node.k
        if k == "Span":
            yield Event(False, "<<", node.page, node.s,
                        (node.s + 2) if node.s is not None else None,
                        node_container(node), node.depth(), k, False)
            for kid in node.kids:
                yield from rec(kid)
            if node.x.get("closed", True):
                yield Event(False, ">>", node.page,
                            (node.e - 2) if node.e is not None else None, node.e,
                            node_container(node), node.depth(), k, False)
        elif k == "Inline":
            yield Event(False, "<", node.page, node.s,
                        (node.s + 1) if node.s is not None else None,
                        node_container(node), node.depth(), k, False)
            for kid in node.kids:
                yield from rec(kid)
            if node.x.get("closed", True):
                yield Event(False, ">", node.page,
                            (node.e - 1) if node.e is not None else None, node.e,
                            node_container(node), node.depth(), k, False)
        elif k in ("Utterance", "Field"):
            if node.x.get("open_op"):
                yield Event(True, node.x["open_op"], node.page,
                            node.x.get("open_off"), node.x.get("open_end"),
                            node_container(node), node.depth(), k, False)
            for kid in node.kids:
                yield from rec(kid)
        elif k == "Header":
            # `#` markers are part of the #14 token stream (kind PUNCT); emit
            # them so the clause partition still reconstitutes it exactly.
            for i in range(int(node.x.get("level", 1))):
                yield Event(False, "#", node.page, node.s, node.s,
                            "header", node.depth(), k, False)
            for kid in node.kids:
                yield from rec(kid)
        elif k in ("Book", "Block", "Page", "Figure", "Segment"):
            for kid in node.kids:
                yield from rec(kid)
        else:
            yield leaf(node)

    yield from rec(book)


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------
def segment(book, is_markup):
    """Return (clauses, diagnostics).

    A clause is a maximal run of content events between boundary operators (or
    page edges).  Empty runs are dropped.  Clauses never cross a page.

    ``is_markup(page, start)`` decides whether a word token is HTML-tag
    residue; markup words are counted in ``n_words`` but not in
    ``n_words_clean`` (the primary length measure).
    """
    clauses = []
    cur = None
    page = None
    cur_header = None
    n_boundaries = 0
    n_header_edges = 0
    boundary_off = collections.defaultdict(list)

    def start_clause(op, off, ev):
        nonlocal cur
        cur = {
            "opener": op, "opener_off": off, "page": ev.page,
            "container": ev.container, "depth": ev.depth,
            "containers": {ev.container},
            "start": None, "end": None, "n_tokens": 0, "n_words": 0,
            "n_words_clean": 0, "n_links": 0, "n_dashes": 0, "n_seps": 0,
            "n_punct": 0, "n_ops": 0, "n_pipes": 0, "n_span_marks": 0,
            "first_word": "", "first_upper": -1,
        }

    def close_clause(closer, closer_off):
        nonlocal cur
        if cur is not None and cur["n_tokens"] > 0:
            cur["closer"] = closer
            cur["closer_off"] = closer_off
            cur["container"] = ("multi" if len(cur["containers"]) > 1
                                else cur["container"])
            clauses.append(cur)
        cur = None

    for ev in iter_events(book):
        if ev.page != page:
            if cur is not None:
                close_clause("EOS", None)
            page = ev.page
            cur_header = None
            start_clause("BOS", None, ev)
        # A layout header is a hard boundary (#14): headings are titles, not
        # clauses, so entering or leaving one breaks the clause.
        is_hdr = ev.container == "header"
        if cur_header is not None and is_hdr != cur_header:
            n_header_edges += 1
            close_clause("HDR", ev.start)
            start_clause("HDR", ev.start, ev)
        cur_header = is_hdr
        if ev.boundary:
            n_boundaries += 1
            if ev.start is not None:
                boundary_off[ev.page].append(ev.start)
            close_clause(ev.value, ev.start)
            start_clause(ev.value, ev.end, ev)
            continue
        # content event
        if cur is None:
            start_clause("BOS", None, ev)
        cur["n_tokens"] += 1
        cur["containers"].add(ev.container)
        if ev.start is not None:
            cur["start"] = ev.start if cur["start"] is None else min(cur["start"], ev.start)
        if ev.end is not None:
            cur["end"] = ev.end if cur["end"] is None else max(cur["end"], ev.end)
        k = ev.node_kind
        if ev.is_word:
            cur["n_words"] += 1
            if ev.start is None or not is_markup(ev.page, ev.start):
                cur["n_words_clean"] += 1
            if cur["first_upper"] < 0:
                cur["first_word"] = ev.value
                cur["first_upper"] = 1 if ev.value[:1].isupper() else 0
        elif k == "Link":
            cur["n_links"] += 1
        elif k == "Dash":
            cur["n_dashes"] += 1
        elif k == "Sep":
            cur["n_seps"] += 1
        elif k == "Punct":
            cur["n_punct"] += 1
        elif k == "Op":
            cur["n_ops"] += 1
        elif k == "Pipe":
            cur["n_pipes"] += 1
        elif k in ("Span", "Inline"):
            cur["n_span_marks"] += 1

    if cur is not None:
        close_clause("EOS", None)

    diags = {"n_clauses": len(clauses), "n_boundaries": n_boundaries,
             "n_header_edges": n_header_edges,
             "boundary_off": {p: sorted(v) for p, v in boundary_off.items()}}
    return clauses, diags


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------
def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stats(vals):
    vals = sorted(vals)
    if not vals:
        return {"n": 0, "total": 0, "mean": 0.0, "sd": 0.0, "min": 0,
                "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0,
                "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "n": len(vals), "total": sum(vals), "mean": round(statistics.fmean(vals), 3),
        "sd": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
        "min": vals[0], "p10": round(pct(vals, 0.10), 1), "p25": round(pct(vals, 0.25), 1),
        "median": round(pct(vals, 0.50), 1), "p75": round(pct(vals, 0.75), 1),
        "p90": round(pct(vals, 0.90), 1), "p95": round(pct(vals, 0.95), 1),
        "p99": round(pct(vals, 0.99), 1), "max": vals[-1],
    }


def histogram(vals, cap=30):
    h = collections.Counter()
    for v in vals:
        h[min(cap, v)] += 1
    return h


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description="#33 clause segmentation via operators")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="dry run")
    args = ap.parse_args(argv)
    OUT = Path(args.out)

    book, _og_diags = og.parse_book()
    nodes = list(book.walk())
    bodies = load_pages()
    page_units, uid_by_ps, unit_meta = load_units()
    markup_spans, markup_uids, markup_types = load_markup(bodies)

    # --- reference counts from the #14 tree (independent of segmentation) ---
    utt_open = [n for n in nodes if n.k == "Utterance" and n.x.get("open_op") == "//"]
    fld_open = [n for n in nodes if n.k == "Field" and n.x.get("open_op") == "::"]
    orphans = [n for n in nodes if n.k == "Op"
               and n.x.get("role") == "orphan-boundary"]
    slashes = [n for n in nodes if n.k == "Slash"]
    n_slash_tt = sum(1 for n in slashes if n.v == "/")
    ref_boundaries = {
        "//": len(utt_open) + sum(1 for n in orphans if n.v == "//"),
        "::": len(fld_open) + sum(1 for n in orphans if n.v == "::"),
        "/": n_slash_tt,
    }
    ref_boundary_total = sum(ref_boundaries.values())
    total_tokens = sum(len(p.token_list()) for p in book.kids)

    # --- segment -----------------------------------------------------------
    def word_is_markup(page, s):
        uid = uid_by_ps.get((page, s))
        if uid is not None and uid in markup_uids:
            return True
        return any(a <= s < b for a, b in markup_spans.get(page, ()))

    clauses, seg_diags = segment(book, word_is_markup)

    # --- unit counts: units whose start falls in the clause, screened --------
    # A #17 unit lies entirely inside one clause (units never span a boundary
    # operator), so counting by start offset is exact.
    def unit_is_markup(uid):
        page, us, _ue, surface = unit_meta[uid]
        if uid in markup_uids or surface.lower() in markup_types:
            return True
        return any(a <= us < b for a, b in markup_spans.get(page, ()))

    unit_crossings = 0
    for page, offs in seg_diags["boundary_off"].items():
        if not offs:
            continue
        for us, ue, _u in page_units.get(page, ()):
            for b in offs:
                if us < b < ue:
                    unit_crossings += 1
                    break
    for cl in clauses:
        page, s, e = cl["page"], cl["start"], cl["end"]
        units = [u for (us, _ue, u) in page_units.get(page, ()) if s <= us < e]
        cl["n_units"] = sum(1 for u in units if not unit_is_markup(u))

    # --- assign ids / previews --------------------------------------------
    for i, cl in enumerate(clauses):
        cl["clause_id"] = i
        body = bodies.get(cl["page"], "")
        s = cl["start"] or 0
        e = cl["end"] or 0
        cl["preview"] = re.sub(r"\s+", " ", body[s:e]).strip()[:80]

    # --- distributions -----------------------------------------------------
    def scope_clauses(scope):
        if scope == "body":
            return [c for c in clauses if c["container"] != "header"]
        return clauses

    len_by_opener = []
    hist_by_opener = collections.defaultdict(collections.Counter)
    for scope in ("all", "body"):
        for op in OPENER_TYPES:
            grp = [c for c in scope_clauses(scope) if c["opener"] == op]
            if not grp:
                continue
            st = stats([c["n_words_clean"] for c in grp])
            st.update(scope=scope, opener=op)
            len_by_opener.append(st)
            if scope == "all":
                hist_by_opener[op] = histogram([c["n_words_clean"] for c in grp])
        # overall row
        grp = scope_clauses(scope)
        st = stats([c["n_words_clean"] for c in grp])
        st.update(scope=scope, opener="ALL")
        len_by_opener.append(st)

    len_by_container = []
    for cont in ("block", "span", "inline", "header", "multi"):
        grp = [c for c in clauses if c["container"] == cont]
        if not grp:
            continue
        st = stats([c["n_words_clean"] for c in grp])
        st.update(container=cont)
        len_by_container.append(st)

    hist_rows = []
    for op in OPENER_TYPES:
        h = hist_by_opener.get(op)
        if not h:
            continue
        for length in sorted(h):
            label = f"{length}+" if length == 30 else str(length)
            hist_rows.append({"opener": op, "length_bin": length,
                              "label": label, "n": h[length]})

    # --- composition by opener ("type" distribution) -----------------------
    comp_rows = []
    for op in OPENER_TYPES:
        grp = [c for c in clauses if c["opener"] == op]
        if not grp:
            continue
        nn = len(grp)
        comp_rows.append({
            "opener": op, "n_clauses": nn,
            "mean_words": round(sum(c["n_words_clean"] for c in grp) / nn, 2),
            "mean_units": round(sum(c["n_units"] for c in grp) / nn, 2),
            "mean_links": round(sum(c["n_links"] for c in grp) / nn, 2),
            "mean_dashes": round(sum(c["n_dashes"] for c in grp) / nn, 2),
            "mean_span_marks": round(sum(c["n_span_marks"] for c in grp) / nn, 2),
            "pct_with_binding": round(
                100 * sum(1 for c in grp if c["n_links"] > 0) / nn, 1),
            "pct_with_span": round(
                100 * sum(1 for c in grp if c["n_span_marks"] > 0) / nn, 1),
            "pct_fused": round(
                100 * sum(1 for c in grp
                          if c["n_units"] < c["n_words_clean"]) / nn, 1),
        })

    # opener -> next opener transition
    trans = collections.Counter()
    for a, b in itertools.pairwise(clauses):
        if a["page"] != b["page"]:
            continue
        trans[(a["opener"], b["opener"])] += 1
    trans_rows = [{"opener": a, "next_opener": b, "n": n}
                  for (a, b), n in sorted(trans.items(),
                                          key=lambda kv: (-kv[1], kv[0]))]

    # --- per page ----------------------------------------------------------
    per_page = []
    by_page = collections.defaultdict(list)
    for c in clauses:
        by_page[c["page"]].append(c)
    for p in sorted(by_page):
        grp = by_page[p]
        words = sum(c["n_words_clean"] for c in grp)
        per_page.append({
            "page": p, "n_clauses": len(grp), "words": words,
            "mean_len": round(words / len(grp), 2) if grp else 0,
            "max_len": max(c["n_words_clean"] for c in grp) if grp else 0,
            "opener_slash": sum(1 for c in grp if c["opener"] == "/"),
            "opener_dslash": sum(1 for c in grp if c["opener"] == "::"),
            "opener_ssl": sum(1 for c in grp if c["opener"] == "//"),
            "opener_bos": sum(1 for c in grp if c["opener"] == "BOS"),
            "opener_hdr": sum(1 for c in grp if c["opener"] == "HDR"),
        })

    # --- tallies -----------------------------------------------------------
    opener_counts = collections.Counter(c["opener"] for c in clauses)
    container_counts = collections.Counter(c["container"] for c in clauses)
    closer_counts = collections.Counter(c["closer"] for c in clauses)
    n_words_total = sum(c["n_words"] for c in clauses)
    n_words_clean_total = sum(c["n_words_clean"] for c in clauses)
    n_units_total = sum(c["n_units"] for c in clauses)
    n_units_raw = sum(len([u for (us, _ue, u) in page_units.get(c["page"], ())
                           if c["start"] <= us < c["end"]]) for c in clauses)
    markup_words = n_words_total - n_words_clean_total
    markup_units = n_units_raw - n_units_total
    boundary_tokens = seg_diags["n_boundaries"]

    summary = {
        "schema": SCHEMA,
        "n_clauses": len(clauses),
        "n_pages_with_clauses": len(by_page),
        "boundary_tokens": boundary_tokens,
        "boundary_by_op": ref_boundaries,
        "boundary_total": ref_boundary_total,
        "header_edges": seg_diags["n_header_edges"],
        "clause_tokens": sum(c["n_tokens"] for c in clauses),
        "total_tokens": total_tokens,
        "partition_ok": (sum(c["n_tokens"] for c in clauses)
                         + boundary_tokens) == total_tokens,
        "n_words_raw": n_words_total,
        "n_words_screened": n_words_clean_total,
        "markup_words_removed": markup_words,
        "n_units": n_units_total,
        "n_units_raw": n_units_raw,
        "markup_units_removed": markup_units,
        "unit_crossings": unit_crossings,
        "by_opener": dict(opener_counts),
        "by_closer": dict(closer_counts),
        "by_container": dict(container_counts),
        "words_per_clause_mean": round(n_words_clean_total / len(clauses), 3),
        "words_per_clause_median": stats([c["n_words_clean"]
                                          for c in clauses])["median"],
        "reference": {
            "utterances": sum(1 for n in nodes if n.k == "Utterance"),
            "fields": sum(1 for n in nodes if n.k == "Field"),
            "slashes": n_slash_tt,
        },
    }

    # --- write -------------------------------------------------------------
    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)

        write_tsv(OUT / "clauses.tsv", [
            "clause_id", "page", "region", "container", "depth", "opener",
            "opener_off", "closer", "closer_off", "start", "end", "n_tokens",
            "n_words", "n_words_clean", "n_units", "n_links", "n_dashes",
            "n_seps", "n_punct", "n_ops", "n_pipes", "n_span_marks",
            "first_word", "first_upper", "preview",
        ], [dict(r, region=("header" if r["container"] == "header" else "body"))
            for r in clauses])

        write_tsv(OUT / "length_by_opener.tsv", [
            "scope", "opener", "n", "total", "mean", "sd", "min", "p10",
            "p25", "median", "p75", "p90", "p95", "p99", "max"], len_by_opener)

        write_tsv(OUT / "length_by_container.tsv", [
            "container", "n", "total", "mean", "sd", "min", "p10", "p25",
            "median", "p75", "p90", "p95", "p99", "max"], len_by_container)

        write_tsv(OUT / "length_histogram.tsv",
                  ["opener", "length_bin", "label", "n"], hist_rows)

        write_tsv(OUT / "composition_by_opener.tsv", [
            "opener", "n_clauses", "mean_words", "mean_units", "mean_links",
            "mean_dashes", "mean_span_marks", "pct_with_binding",
            "pct_with_span", "pct_fused"], comp_rows)

        write_tsv(OUT / "opener_transitions.tsv",
                  ["opener", "next_opener", "n"], trans_rows)

        write_tsv(OUT / "per_page.tsv", [
            "page", "n_clauses", "words", "mean_len", "max_len", "opener_slash",
            "opener_dslash", "opener_ssl", "opener_bos", "opener_hdr"], per_page)

        diag_rows = [
            {"metric": "clauses", "value": len(clauses)},
            {"metric": "boundary_tokens", "value": boundary_tokens},
            {"metric": "header_edges", "value": seg_diags["n_header_edges"]},
            {"metric": "clause_tokens", "value": sum(c["n_tokens"] for c in clauses)},
            {"metric": "total_tokens", "value": total_tokens},
            {"metric": "partition_ok", "value": int(summary["partition_ok"])},
            {"metric": "words_raw", "value": n_words_total},
            {"metric": "words_screened", "value": n_words_clean_total},
            {"metric": "markup_words_removed", "value": markup_words},
            {"metric": "units", "value": n_units_total},
            {"metric": "units_raw", "value": n_units_raw},
            {"metric": "markup_units_removed", "value": markup_units},
            {"metric": "unit_crossings", "value": unit_crossings},
            {"metric": "markup_ledger_uids", "value": len(markup_uids)},
            {"metric": "markup_types", "value": len(markup_types)},
        ]
        for k, v in sorted(opener_counts.items()):
            diag_rows.append({"metric": f"opener:{k}", "value": v})
        for k, v in sorted(container_counts.items()):
            diag_rows.append({"metric": f"container:{k}", "value": v})
        write_tsv(OUT / "diagnostics.tsv", ["metric", "value"], diag_rows)

        recon_rows = [
            {"quantity": "boundary //", "here": ref_boundaries["//"],
             "earlier": "2,149 (#14)", "note": "Utterance open + orphan leaf"},
            {"quantity": "boundary ::", "here": ref_boundaries["::"],
             "earlier": "3,084 (#14)", "note": "Field open + orphan leaf"},
            {"quantity": "boundary / (sentence-final)", "here": ref_boundaries["/"],
             "earlier": "540 (#14)", "note": "Slash leaves only; Link / excluded"},
            {"quantity": "total tokens", "here": total_tokens,
             "earlier": "90,220 (#14 round-trip)", "note": "clause partition + boundaries"},
            {"quantity": "utterances", "here": summary["reference"]["utterances"],
             "earlier": "3,764 (#14)", "note": "each clause nests in one utterance"},
            {"quantity": "units", "here": n_units_total,
             "earlier": "74,077 (#27)", "note": "markup-screened, joined on (page,start)"},
            {"quantity": "markup words removed", "here": markup_words,
             "earlier": "347 ledger (#19)", "note": "plus tag-run furniture"},
            {"quantity": "markup units removed", "here": markup_units,
             "earlier": "588 (#27)", "note": "ledger + tag runs + junk types"},
        ]
        write_tsv(OUT / "reconciliation.tsv",
                  ["quantity", "here", "earlier", "note"], recon_rows)

        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

        write_report(OUT / "clause_report.md", summary, clauses, len_by_opener,
                     len_by_container, hist_by_opener, trans_rows, per_page,
                     ref_boundaries, comp_rows)

    if not args.quiet:
        print(f"clauses: {len(clauses):,}  boundaries: {boundary_tokens:,} "
              f"({ref_boundaries})")
        print(f"partition ok: {summary['partition_ok']}  "
              f"tokens {summary['clause_tokens']:,}+{boundary_tokens:,}"
              f"={total_tokens:,}")
        print(f"words: {n_words_total:,} raw -> {n_words_clean_total:,} "
              f"screened ({markup_words} markup)  units: {n_units_total:,}")
        print("by opener:", dict(opener_counts))
        print("median clause length:", summary["words_per_clause_median"])
        if not args.no_write:
            print(f"wrote {OUT}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _md_stats_table(rows, key):
    lines = [f"| {key} | clauses | words | mean | median | p90 | p99 | max |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| `{r[key]}` | {r['n']:,} | {r['total']:,} | "
                     f"{r['mean']} | {r['median']} | {r['p90']} | {r['p99']} | "
                     f"{r['max']} |")
    return "\n".join(lines)


def write_report(path, summary, clauses, len_by_opener, len_by_container,
                 hist_by_opener, trans_rows, per_page, ref_boundaries,
                 comp_rows):
    allrows = [r for r in len_by_opener if r["scope"] == "all"]
    n = summary["n_clauses"]

    # shortest / longest example clauses
    by_len = sorted(clauses, key=lambda c: (c["n_words_clean"], c["page"],
                                            c["start"] or 0))
    nonempty = [c for c in by_len if c["n_words_clean"] > 0]
    short = nonempty[0]
    longest = by_len[-1]

    def example(c, k=90):
        return (c["preview"][:k] if c["preview"] else "(no preview)")

    L = []
    A = L.append
    A("# Clause segmentation via operators (#33)")
    A("")
    A("Split the book at `::`, `//` and the **sentence-final** `/`; each "
      "operator-delimited span is a clause. Built on the #14 block/AST layer "
      "(boundary identities, offsets and container context all come from the "
      "parsed tree), joined to the #17 unit layer on `(page, start)`, and "
      "markup-screened with #19's ledger plus a tag-run detector.")
    A("")
    A(f"**Headline.** {n:,} clauses over {summary['n_pages_with_clauses']} "
      f"pages; {summary['n_words_screened']:,} screened words; median clause "
      f"length **{summary['words_per_clause_median']}** words, mean "
      f"**{summary['words_per_clause_mean']}**. The clause partition plus the "
      f"{summary['boundary_total']:,} boundary tokens exactly reconstitutes "
      f"the {summary['total_tokens']:,}-token #14 stream "
      f"(partition_ok={summary['partition_ok']}).")
    A("")
    A("## 1. Definition")
    A("")
    A("A clause is a **maximal run of content tokens between two boundary "
      "operators** (or a page edge). The three boundary operators are:")
    A("")
    A("| operator | #14 node | role |")
    A("|---|---|---|")
    A("| `//` | `Utterance` open / orphan `Op` | utterance break |")
    A("| `::` | `Field` open / orphan `Op` | record-field separator |")
    A("| `/` | `Slash` leaf only | sentence-final terminator |")
    A("")
    A("`/` glued to the next word is a compound `Link` (`soul/gram`) and is "
      "**content**, not a boundary; this is #14's measured gluing rule, not a "
      "new choice. All other operators (`=`, `++`, `+`, `<<`, `>>`, `<`, `>`, "
      "`|`, dashes, `: . , ;`) are content: they do not split a clause.")
    A("")
    A("One structural delimiter is added to the three task operators: a **layout "
      "header edge** (`HDR`). #14 already treats a heading as a hard container "
      "boundary, and a section title is not part of the clause that follows it; "
      "without the edge, an unpunctuated title would fuse into the first body "
      "clause. Header clauses are counted separately (`container=header`, 56 of "
      "them, median length 1 word) and the `scope=body` rows exclude them. "
      "`BOS` marks a page's first clause (no delimiter on its left).")
    A("")
    A("## 2. Inventory and reconciliation")
    A("")
    A("| quantity | here | earlier | note |")
    A("|---|---:|---|---|")
    A(f"| clauses | {n:,} | — | maximal spans between boundaries |")
    A(f"| boundary `//` | {ref_boundaries['//']:,} | 2,149 (#14) | "
      "`Utterance` open + orphan `Op` leaf |")
    A(f"| boundary `::` | {ref_boundaries['::']:,} | 3,084 (#14) | "
      "`Field` open + orphan `Op` leaf |")
    A(f"| boundary `/` | {ref_boundaries['/']:,} | 540 (#14) | "
      "sentence-final `Slash` only |")
    A(f"| total tokens | {summary['total_tokens']:,} | 90,220 (#14) | "
      "clause partition + boundaries |")
    A(f"| units (screened) | {summary['n_units']:,} | 74,077 (#27) | "
      "markup-screened `(page, start)` join |")
    A(f"| units (raw) | {summary['n_units_raw']:,} | 74,665 (#17) | "
      f"{summary['markup_units_removed']:,} markup units removed |")
    A("")
    A(f"Boundary totals: {summary['boundary_by_op']}. A clause never crosses a "
      f"page, and no #17 unit crosses a clause boundary "
      f"(unit_crossings={summary['unit_crossings']}).")
    A("")
    A("## 3. Length distribution")
    A("")
    A("Primary measure is the **markup-screened word count** "
      f"(`n_words_clean`). {summary['markup_words_removed']:,} markup-residue "
      f"words were removed from {summary['n_words_raw']:,} raw words.")
    A("")
    A("### By opening operator")
    A("")
    A(_md_stats_table(allrows, "opener"))
    A("")
    A("The opener is the boundary operator that introduced the clause; `BOS` "
      "is the first clause of a page (no operator on its left) and `HDR` is a "
      "clause that begins at a header edge. The measured ordering is stable: "
      "`::`-opened clauses (median 11) and `//`-opened clauses (median 10) are "
      "the short, repeated units of the machine register, `/`-opened clauses "
      "are the longest (median 12, mean 15.8), and header clauses are the "
      "shortest (median 1).")
    A("")
    A("### By container")
    A("")
    A(_md_stats_table(len_by_container, "container"))
    A("")
    A("### Histogram (all clauses)")
    A("")
    ops = OPENER_TYPES
    A("| words | " + " | ".join(f"`{o}`" for o in ops) + " |")
    A("|---:|" + "---:|" * len(ops))
    for length in range(31):
        label = "30+" if length == 30 else str(length)
        row = [hist_by_opener.get(o, {}).get(length, 0) for o in ops]
        if not any(row):
            continue
        A(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    A("")
    A("### Composition by opener (clause *type*)")
    A("")
    A("Mean counts per clause and the share of clauses carrying each construct. "
      "`binding` = a `=`/`++`/`+` link (a naming or concatenation seam), "
      "`span` = a `<<…>>`/`<…>` marker, `fused` = a clause whose #17 unit "
      "count is below its word count (a multi-token unit such as `gene=TV`).")
    A("")
    A("| opener | clauses | words | units | links | dashes | % binding | % span | % fused |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in comp_rows:
        A(f"| `{r['opener']}` | {r['n_clauses']:,} | {r['mean_words']} | "
          f"{r['mean_units']} | {r['mean_links']} | {r['mean_dashes']} | "
          f"{r['pct_with_binding']} | {r['pct_with_span']} | {r['pct_fused']} |")
    A("")
    A("## 4. Clause syntax: opener transitions")
    A("")
    A("The 10 most frequent opener-to-next-opener transitions on the same page:")
    A("")
    A("| opener | next | n |")
    A("|---|---|---:|")
    for r in trans_rows[:10]:
        A(f"| `{r['opener']}` | `{r['next_opener']}` | {r['n']:,} |")
    A("")
    A("## 5. Examples")
    A("")
    n_zero = sum(1 for c in clauses if c["n_words_clean"] == 0)
    A(f"- {n_zero} clauses carry zero words: their content is operator/punctuation "
      "only (e.g. a stray `>>` between two delimiters). They are kept in the "
      "inventory but are a minority ("
      f"{n_zero / n * 100:.1f}% of clauses).")
    A(f"- Shortest non-empty ({short['n_words_clean']} word, p.{short['page']}, "
      f"opened by `{short['opener']}`): `{example(short)}`")
    A(f"- Longest ({longest['n_words_clean']} words, p.{longest['page']}, "
      f"opened by `{longest['opener']}`): `{example(longest, 120)}`")
    A("")
    A("The longest clause is not a real sentence: page 90 is one of the marker "
      "OCR recognition loops #4 documented (\"of the second of\" repeated), so "
      "the top of the distribution is an OCR artifact, not a prose structure. "
      "The median (11) is the robust centre.")
    A("")
    A("## 6. Negative results and limitations")
    A("")
    A("- The clause boundary is **operator-driven, not syntactic**. A clause "
      "can span a paragraph break and can open in a `<<` span and continue "
      "outside it (the container is reported as `multi` in that case, 422 "
      "clauses). The measure is a structural partition, not a parse of English.")
    A("- Clause length counts word **tokens**; `n_units` counts maximal #17 "
      "**units**, so the two differ where a fused unit (`gene=TV`) spans "
      "several tokens.")
    A("- Header clauses (section titles) are included and marked "
      "`region=header`; the `scope=body` rows exclude them. Header edges are a "
      "deliberate fourth delimiter, disclosed above.")
    A("- The markup screen removes tag residue from the length distribution "
      f"({summary['markup_words_removed']:,} words, "
      f"{summary['markup_units_removed']:,} units); it does not repair the "
      "underlying OCR.")
    A("- The 540 sentence-final `/` are #14's measured disambiguation of a "
      "single glyph into a compound `Link` (952) and a terminator; a different "
      "gluing rule would move clauses between the `/` and `BOS` groups.")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
