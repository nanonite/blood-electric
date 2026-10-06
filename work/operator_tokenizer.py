#!/usr/bin/env python3
"""Operator-aware tokenizer - Chainlink #17.

Standard tokenizers throw away exactly the information this corpus is made of.
`gene=TV`, `soul/gram` and `suck=blood` are single lexical units whose *parts*
and whose *seams* (`=`, `/`) are separately meaningful; a whitespace splitter
keeps the unit but not the parts, a word-tokenizer keeps the parts but not the
unit, and a `\w+` tokenizer keeps neither and throws the operator away.  This
module emits all three levels at once, on the token stream #10/#12/#14 already
established, so nothing has to be re-guessed downstream.

Grammar of a *unit* (maximal lexical run), L2 of docs/meaning-extraction-methods.md:

    stream  = { region } ;
    region  = header | para ;
    token   = word | op | punct ;
    unit    = word { seam word } ;            (* maximal; `word` includes
                                                internal-hyphen compounds *)
    seam    = "=" | "++" | "+" | "/" ;        (* lexical join, #12 *)
    join    = "-" | "—" | "–" ;        (* word-internal or glued dash *)
    part    = word | seam | join ;    (* the unit's own sub-strings *)

Seam rules, each one #12's measured signature (not a guess):
  * `=`, `++`, `+`  seam iff a WORD token sits on **both** sides (#12: `=` is
    glued-L 0.98 / glued-R 0.95, interior 0.92 - a tight lexical linker).
  * `/`            seam iff it is **glued to a following word** and a word
    follows (#14's measured rule: 952 Link nodes vs 540 sentence-final Slash
    nodes; #12: dual role, compound/possessive + sentence-final).
  * `::`, `//`, `<<`, `>>`, `<`, `>`, `|`, dashes, `. , ; :`  never seams:
    they are *segment* operators (#14's precedence: span > utterance > field >
    segment > atom/link), so they end a unit.

Three views are emitted for every occurrence, all from one pass:
  1. `tokens` - the lossless token stream.  Each token carries the exact
     inter-token gap, so `"".join(gap + text)` rebuilds the page body
     byte-for-byte (asserted at build time, 225/225 pages).
  2. `units`  - maximal `word (seam word)*` runs: `gene=TV`, `soul/gram`,
     `gene=TV/spasm`, `boy-roid`.
  3. `parts`  - the unit's sub-strings with roles `w` (word), `seam`
     (operator), `join` (a word-internal or glued dash), each with its own
     source offsets (`part_index.tsv`).

Optional enrichment: when #14's AST is importable, every token/unit is located
in the operator tree (`ast_path`, `container`, `in_span`) by character offset,
so a unit can be read in its `<<…>>` / `//` / `::` context without re-parsing.
The tokenizer itself never imports the parser at call time.

Outputs under analysis/tokenizer/ (the path named in the #17 task text):
  tokenizer_spec.md       THE DELIVERABLE: grammar, schema, evaluation
  tokens.tsv              the lossless token stream, unit-annotated
  tokens.jsonl            the same stream, one JSON object per token
  units.tsv               one row per lexical unit (the analyzable unit)
  unit_types.tsv          unit type inventory (surface -> n, pages, class)
  part_index.tsv          one row per unit part (w / seam / join)
  seam_loss.tsv           standard-tokenizer comparison, per unit class
  cases.tsv               named constructs, page-cited, with all three views
  case_examples.txt       readable renderings of each case
  diagnostics.tsv         tokenizer pressure ledger (orphans, spaced seams...)
  reconciliation.tsv      this tokenizer vs #10 / #12 / #14 / first pass
  summary.json            machine-readable headline numbers

Tests: work/test_operator_tokenizer.py (regression suite; run the whole work/
suite with `python -m unittest discover -s work -p 'test_*.py'`).
"""
import bisect
import collections
import csv
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from operator_inventory import OPS                                  # #10
from operator_functions import tokenize as base_tokenize            # #12
import operator_grammar as OG                                       # #14
import operator_ngrams as NG                                         # #11

SRC = "analysis/layout_recovery/blood-electric.layout.md"
OUT = "analysis/tokenizer"

# ---------------------------------------------------------------------------
# Numbers this tokenizer must reproduce (see analysis/block_parser/grammar_spec.md
# §3 and analysis/operator_inventory/operator_report.md).  A drift here means
# the tokenizer stopped agreeing with the operator inventory, not that the book
# changed, so every one of them is a build-time assertion.
# ---------------------------------------------------------------------------
EXPECTED_PAGES = 225
EXPECTED_OPS_BODY = 10949        # #10/#12, page bodies (headers excluded)
EXPECTED_OPS_HEADER = 51         # #14 §3: operator tokens living in headings
EXPECTED_OPS_ALL = EXPECTED_OPS_BODY + EXPECTED_OPS_HEADER
EXPECTED_WORDS = 77485           # #14: 77,485 Lex (word-atom) nodes
EXPECTED_OPS_TOKENS = {          # op -> count in page bodies (from #14 §3)
    "<<": 412, ">>": 386, "//": 2149, "::": 3083, "++": 198, "=": 2047,
    "/": 1484, "<": 381, ">": 777, "+": 15, "|": 17,
}
EXPECTED_LINK_NODES = {          # #14 summary.json: Link nodes, whole book
    "=": 2048, "++": 198, "+": 15, "/": 952,
}
EXPECTED_SLASH_NODES = 540       # #14: sentence-final `/` terminators
EXPECTED_FREE_DASH = 696         # #13/#14: whitespace-bounded dash sites

# Seam classes: an operator that can bind two words into one lexical unit.
SEAM_BOTH = ("=", "++", "+")     # need a word on each side
SEAM_RIGHT = ("/",)              # need a glued word on the right (#14 rule)
# Operators that end a segment and therefore *never* bind (#14 precedence).
SEGMENT_OPS = ("::", "//", "<<", ">>", "<", ">", "|")
DASH_CHARS = ("-", "—", "–")
# ---------------------------------------------------------------------------
# Source / OCR markup residue.  The page bodies still carry real HTML that
# marker bridged out of the source PDF: page furniture (`<th>`, `<pre>`, `<p>`,
# `<li>`, `<br/>`, `<hr/>`) and a block of `<a href="mailto:...">` anchors on
# the copyright page.  Every such tag contributes its *name* as a WORD token
# (`a`, `th`, `p`, `pre`, `li`, `br`, `hr`), so it would otherwise enter the
# lexicon as a one- or two-character "word".  The book's own angle-bracket forms
# (`<<…>>`, `<the meridian>`) are not tags and are not matched.
HTML_TAGS = {"a", "abbr", "address", "article", "aside", "b", "base",
             "blockquote", "body", "br", "button", "caption", "cite", "code",
             "col", "dd", "div", "dl", "dt", "em", "figcaption", "figure",
             "font", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
             "head", "header", "hr", "html", "i", "img", "input", "kbd",
             "li", "link", "main", "meta", "nav", "noscript", "ol", "option",
             "p", "pre", "q", "s", "samp", "script", "section", "select",
             "small", "span", "strike", "strong", "style", "sub", "sup",
             "table", "tbody", "td", "textarea", "tfoot", "th", "thead",
             "title", "tr", "tt", "u", "ul", "var"}
RE_HTML_TAG = re.compile(r"</?([A-Za-z]{1,12})/?>")
PUNCT_CLASS = {",": "comma", ".": "period", ";": "semi", ":": "colon",
               "!": "bang", "?": "query", '"': "quote", "'": "apos",
               "(": "bracket", ")": "bracket", "[": "bracket", "]": "bracket"}
CONTAINERS = ("Span", "Utterance", "Field", "Inline", "Header", "Block", "Segment")

SCHEMA = "operator-aware-tokenizer/1"


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------
class Tok:
    """One token.  `gap` is the exact source text between the previous token's
    end and this token's start, which makes the stream losslessly invertible."""

    __slots__ = ("i", "page", "region", "item", "hlevel", "s", "e", "kind",
                 "text", "gap", "glue_L", "glue_R", "upper", "num",
                 "has_digit", "caps", "left_op", "pclass", "unit", "pos",
                 "ast_path", "container")

    def __init__(self, i, page, region, item, hlevel, s, e, kind, text, gap,
                 glue_L=False, glue_R=False, left_op="", pclass=""):
        self.i = i
        self.page = page
        self.region = region          # 'header' | 'para'
        self.item = item              # layout-item index: a seam never crosses it
        self.hlevel = hlevel or 0
        self.s = s
        self.e = e
        self.kind = kind              # 'WORD' | 'OP' | 'PUNCT'
        self.text = text
        self.gap = gap
        self.glue_L = glue_L          # no whitespace before
        self.glue_R = glue_R          # no whitespace after
        self.upper = text[:1].isupper()
        self.num = kind == "WORD" and text.isdigit()
        self.has_digit = kind == "WORD" and any(c.isdigit() for c in text)
        self.caps = kind == "WORD" and any(c.isupper() for c in text[1:])
        self.left_op = left_op        # nearest operator token immediately left
        self.pclass = pclass          # PUNCT only: comma / dash / bracket ...
        self.unit = None              # Unit id
        self.pos = -1                 # index inside the unit (0 = first word)
        self.ast_path = ""
        self.container = ""

    def obj(self):
        d = {"tid": self.i, "page": self.page, "region": self.region,
             "kind": self.kind, "text": self.text, "start": self.s,
             "end": self.e}
        if self.item >= 0:
            d["item"] = self.item
        if self.hlevel:
            d["hlevel"] = self.hlevel
        if self.gap:
            d["gap"] = self.gap
        if self.kind == "WORD":
            for flag in ("glue_L", "glue_R", "upper", "num", "has_digit", "caps"):
                if getattr(self, flag):
                    d[flag] = True
        elif self.kind == "PUNCT":
            d["pclass"] = self.pclass or "other"
            d["glue_L"] = self.glue_L
            d["glue_R"] = self.glue_R
        else:
            d["glue_L"] = self.glue_L
            d["glue_R"] = self.glue_R
        if self.left_op:
            d["left_op"] = self.left_op
        if self.unit is not None:
            d["unit"] = self.unit
            d["pos"] = self.pos
        if self.ast_path:
            d["ast"] = self.ast_path
        return d


class Unit:
    """A maximal lexical run: `word (seam word)*`, plus its parts."""

    __slots__ = ("id", "page", "region", "s", "e", "text", "toks", "parts",
                 "ops", "sig", "cls", "n_hyp", "n_join", "ast_path", "container",
                 "in_span", "in_utt", "in_field")

    def __init__(self, uid, toks):
        self.id = uid
        self.toks = toks
        self.page = toks[0].page
        self.region = toks[0].region
        self.s = toks[0].s
        self.e = toks[-1].e
        # The unit's surface is the exact source slice: gaps between tokens are
        # kept, so `gene = TV` (spaced) stays distinct from `gene=TV`.
        self.text = "".join(
            (t.gap if k else "") + t.text for k, t in enumerate(toks))
        self.ops = [t.text for t in toks if t.kind == "OP"]
        self.parts = []              # (text, role) role in w / seam / join
        for t in toks:
            if t.kind == "OP":
                self.parts.append((t.text, "seam"))
            elif t.kind == "PUNCT" and t.pclass == "dash":
                # a dash glued to a word on at least one side: #12/#14's
                # compound join (`ice—our`), not a separator
                self.parts.append((t.text, "join"))
            elif "-" in t.text:
                chunks = t.text.split("-")
                for j, ch in enumerate(chunks):
                    if j:
                        self.parts.append(("-", "join"))
                    if ch:
                        self.parts.append((ch, "w"))
            else:
                self.parts.append((t.text, "w"))
        self.sig = "".join(self.ops)
        self.n_join = sum(1 for _t, r in self.parts if r == "join")
        self.n_hyp = sum(1 for _t, r in self.parts
                         if r == "join" and _t == "-")
        self.cls = unit_class(self)
        self.ast_path = ""
        self.container = ""
        self.in_span = self.in_utt = self.in_field = ""

    @property
    def n_parts(self):
        return sum(1 for _t, r in self.parts if r == "w")

    def obj(self):
        return {"uid": self.id, "page": self.page, "region": self.region,
                "text": self.text, "start": self.s, "end": self.e,
                "parts": [{"text": t, "role": r} for t, r in self.parts],
                "sig": self.sig, "class": self.cls, "n_words": self.n_parts,
                "ast": self.ast_path or None, "container": self.container or None}


def unit_class(u):
    """Name a unit's shape from its seams.  Descriptive, not interpretive."""
    if u.ops:
        kinds = set(u.ops)
        if len(kinds) > 1:
            base = "mixed"
        else:
            base = {"=": "eq", "/": "slash", "++": "cat", "+": "plus"}[u.ops[0]]
        return base + ("+join" if u.n_join else "")
    return "join" if u.n_join else "plain"


def part_spans(u):
    """`[(start, end, text, role)]` for a unit's parts, with exact source offsets.

    A hyphen inside a WORD token is a `join` part, so the sub-strings get their
    own offsets (`cold-blooded` -> `cold` [s, s+4), `-` [s+4, s+5), `blooded`).
    The text sequence must equal `Unit.parts`; the build asserts it, so the
    offset view and the role view cannot drift apart.
    """
    out = []
    for t in u.toks:
        if t.kind == "WORD" and "-" in t.text:
            off = t.s
            for j, chunk in enumerate(t.text.split("-")):
                if j:
                    out.append((off, off + 1, "-", "join"))
                    off += 1
                if chunk:
                    out.append((off, off + len(chunk), chunk, "w"))
                    off += len(chunk)
            continue
        if t.kind == "OP":
            role = "seam"
        elif t.kind == "PUNCT" and t.pclass == "dash":
            role = "join"
        else:
            role = "w"
        out.append((t.s, t.e, t.text, role))
    return out


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def html_tag_runs(body):
    """Offsets of real HTML tag runs in a page body (the book's own
    angle-bracket forms such as `<<…>>` and `<the meridian>` are not tags)."""
    return [(m.start(), m.end(), m.group(0)) for m in RE_HTML_TAG.finditer(body)
            if m.group(1).lower() in HTML_TAGS]


def comment_spans(body):
    return [(m.start(), m.end()) for m in re.finditer(r"<!--.*?-->", body, re.S)]


def tokenize_page(pno, body):
    """Tokenize one page body.  Layout placeholder comments yield no tokens but
    their text survives inside the recorded gaps (losslessness).

    Returns (tokens, diagnostics, tail) where `tail` is the page text after the
    last token.
    """
    cspans = comment_spans(body)
    items = [it for it in OG.items_of_body(body) if it.kind != "figure"]
    regions = [(it.s, it.e, it.kind, it.level or 0) for it in items]
    starts = [r[0] for r in regions]

    def region_of(off):
        """(region kind, header level, layout-item index) for a page offset."""
        i = bisect.bisect_right(starts, off) - 1
        if 0 <= i < len(regions) and off < regions[i][1]:
            return regions[i][2], regions[i][3], i
        return "para", 0, -1

    toks, diags, prev_e = [], [], 0
    for n, (kind, s, e, val) in enumerate(base_tokenize(body)):
        if cspans and any(cs <= s < ce for cs, ce in cspans):
            continue                      # layout scaffolding, not book text
        region, hlevel, item = region_of(s)
        if kind == "WORD":
            tk, pclass = "WORD", ""
        elif kind == "OP":
            tk, pclass = "OP", ""
        else:
            tk = "PUNCT"
            pclass = ("dash" if val in DASH_CHARS
                      else PUNCT_CLASS.get(val, "other"))
        glue_L = bool(toks) and toks[-1].e == s
        left_op = toks[-1].text if toks and toks[-1].kind == "OP" else ""
        t = Tok(len(toks), pno, region, item, hlevel, s, e, tk, val,
                body[prev_e:s], glue_L=glue_L, left_op=left_op, pclass=pclass)
        toks.append(t)
        prev_e = e
    for i, t in enumerate(toks):
        if i + 1 < len(toks):
            t.glue_R = toks[i + 1].s == t.e
    return toks, diags, body[prev_e:]


def rebuild(toks, tail):
    """The losslessness check: gaps + token texts == the page body."""
    return "".join(t.gap + t.text for t in toks) + tail


def tokenize_book(src=SRC):
    """{page: (tokens, diags, tail)} over the whole book, in page order."""
    bodies = OG.load_pages_raw(src)
    return {p: tokenize_page(p, bodies[p]) for p in sorted(bodies)}


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def _is_seam(toks, i):
    """Is toks[i] (an OP) a lexical seam?  Returns (bool, reason)."""
    t = toks[i]
    if t.kind != "OP":
        return False, "not-op"
    if t.text in SEGMENT_OPS:
        return False, "segment-op"
    # A seam never crosses a layout item: a blank line or a heading is not
    # glue, so `alpha =` / `beta` on separate paragraphs is two units.
    same = ((i == 0 or toks[i - 1].item == t.item)
            and (i + 1 >= len(toks) or toks[i + 1].item == t.item))
    if t.text in SEAM_BOTH:
        # `=` `++` `+`: bind only between two words inside one region.  A
        # paragraph break is not glue, so `head =\n body` is a terminator pair,
        # not a binding - the same rule #14 applies to `/`.
        if i == 0 or i + 1 >= len(toks) or not same:
            return False, "seam_orphan_side"
        if toks[i - 1].kind != "WORD":
            return False, "seam_orphan_left"
        if toks[i + 1].kind != "WORD":
            return False, "seam_orphan_right"
        return True, ""
    if t.text in SEAM_RIGHT:
        # `/`: #14's measured rule - glued to the next token AND a word follows.
        # Unglued it is the sentence-final terminator, which is #12's second
        # `/` role.  #14's Link test does not look left or check the layout
        # item, so those two cases are kept as separate reasons here: they are
        # still Link nodes in #14's arithmetic, they just do not bind a unit.
        if not t.glue_R or i + 1 >= len(toks) or toks[i + 1].kind != "WORD":
            return False, ("seam_orphan_right"
                           if t.glue_R and i + 1 < len(toks)
                           and toks[i + 1].kind != "WORD" else "slash_free")
        if not same:
            return False, "slash_boundary"
        if i == 0 or toks[i - 1].kind != "WORD":
            return False, "seam_orphan_left"
        return True, ""
    return False, "non-seam-op"


def _dash_cuts(toks, j):
    """True when the dash at `j` is a separator rather than a compound join.

    #12's test: a dash is *free-standing* (whitespace-bounded) when it is a
    separator.  #14 applies the same test to all three dash characters and
    keeps the 19 glued ones as `Punct` leaves.  This tokenizer is stricter than
    #14 in one direction: a dash joins two words into a unit only when it is
    glued on **both** sides (`ice—our`, `channel—the` - one lexical item).
    A dash with whitespace on either side (`scope— the`, `word - word`) cuts,
    which is the conservative reading and keeps the 696 whitespace-bounded
    dash sites of #13/#14 intact.
    """
    t = toks[j]
    return not (t.glue_L and t.glue_R)


def build_units(toks):
    """Maximal `word (seam word | glued-dash word)*` runs.

    Returns (units, diagnostics).
    """
    units, diags, i, n = [], [], 0, len(toks)
    uid = 0
    while i < n:
        if toks[i].kind != "WORD":
            i += 1
            continue
        j = i + 1
        while j < n:
            if toks[j].kind == "OP":
                if not _is_seam(toks, j)[0]:
                    break
                j += 2                  # seam + its right word
                continue
            # a dash glued to a word on either side is #14's compound join
            # (`ice—our`), not a separator: it keeps the unit going.
            if toks[j].kind == "PUNCT" and toks[j].pclass == "dash" \
                    and not _dash_cuts(toks, j) and j + 1 < n \
                    and toks[j + 1].kind == "WORD" \
                    and toks[j].item == toks[j + 1].item:
                j += 2                  # glued dash + its right word
                continue
            break
        u = Unit(uid, toks[i:j])
        units.append(u)
        for k, t in enumerate(toks[i:j]):
            t.unit = u.id
            t.pos = k
        uid += 1
        i = j
    return units, diags


def classify_op_tokens(toks):
    """Every operator token gets exactly one role, from a full scan.

    This is the diagnostics ledger's source: it does not stop at the first
    non-seam, so an operator that sits after a non-word (e.g. `::/to`) is still
    accounted for.  Roles:
      seam            - bound two words into a unit
      segment         - a #14 segment operator (`::` `//` `<<` `>>` `<` `>` `|`)
      slash_free      - whitespace-bounded `/`: #14's sentence-final terminator
      seam_orphan_left  - a seam operator with no word on its left
      seam_orphan_right - a seam operator with no word on its right
    """
    out = []
    for i, t in enumerate(toks):
        if t.kind != "OP":
            continue
        ok, reason = _is_seam(toks, i)
        if ok:
            role = "seam"
        elif t.text in SEGMENT_OPS:
            role = "segment"
        else:
            role = reason
        out.append({"page": t.page, "start": t.s, "op": t.text, "role": role,
                    "upper": t.upper})
    return out


def check_unit_invariants(toks, units):
    """Maximality and containment: a unit is `word (seam word)*`, nothing else.

    Returns a list of violation strings (empty = clean).
    """
    bad = []
    for u in units:
        for k, t in enumerate(u.toks):
            if k % 2 == 0:
                if t.kind != "WORD":
                    bad.append(f"unit {u.id} p{u.page}: even slot {k} is "
                               f"{t.kind}, expected WORD")
            elif t.kind == "OP":
                ok, _r = _is_seam(u.toks, k)
                if not ok:
                    bad.append(f"unit {u.id} p{u.page}: slot {k} "
                               f"{t.text!r} is not a seam")
            elif not (t.kind == "PUNCT" and t.pclass == "dash"):
                bad.append(f"unit {u.id} p{u.page}: slot {k} is {t.kind} "
                           f"{t.text!r}, expected a seam or a glued dash")
        if u.toks[0].s != u.s or u.toks[-1].e != u.e:
            bad.append(f"unit {u.id} p{u.page}: offsets disagree with its tokens")
    # maximality: the token after a unit must not be a seam onto the unit
    by_page = collections.defaultdict(list)
    for t in toks:
        by_page[t.page].append(t)
    for pno, ts in by_page.items():
        uid_of = {id(t): t.unit for t in ts}
        for idx, t in enumerate(ts):
            if t.kind == "OP" and uid_of[id(t)] is None and idx + 1 < len(ts):
                ok, _r = _is_seam(ts, idx)
                if ok:
                    bad.append(f"p{pno}: seam {t.text}@{t.s} not bound to a unit")
    return bad


# ---------------------------------------------------------------------------
# AST enrichment (optional; #14)
# ---------------------------------------------------------------------------
class AstIndex:
    """Offset -> operator-tree container lookup, built from #14's AST."""

    def __init__(self, src=SRC):
        book, _diags = OG.parse_book(src)
        acc = collections.defaultdict(list)
        for p in book.kids:
            acc[p.page] = self._page_intervals(p)
        self.by_page = {}
        for p, iv in acc.items():
            rows = sorted(iv, key=lambda r: (r[0], r[1]))
            self.by_page[p] = (rows, [r[0] for r in rows])

    @staticmethod
    def _page_intervals(page):
        out = []

        def walk(n, path):
            here = path + [n.k]
            if n.k in CONTAINERS and n.s is not None and n.e is not None:
                out.append((n.s, n.e, ">".join(here), n.k))
            for kid in n.kids:
                walk(kid, here)

        walk(page, ["Book"])
        return out

    STRUCT = ("Span", "Utterance", "Field", "Inline", "Header", "Block")

    def lookup(self, pno, off):
        """(path, container) for the deepest container whose span covers `off`.

        `path` includes the `Segment` layer; `container` is the outermost
        structural layer (#14's vocabulary) so downstream joins do not depend
        on segment subdivision.
        """
        ent = self.by_page.get(pno)
        if not ent:
            return "", ""
        rows, starts = ent
        hi = bisect.bisect_right(starts, off)
        best = None
        best_key = None
        for s, e, path, kind in rows[:hi]:
            if s <= off < e:
                key = (path.count(">"), s)
                if best_key is None or key > best_key:
                    best_key, best = key, (path, kind)
        if best is None:
            return "", ""
        path, kind = best
        for lvl in reversed(self.STRUCT):
            if f">{lvl}" in path:
                return path, lvl
        return path, kind


def annotate(toks, units, index):
    """Attach the operator-tree path / container to tokens and units."""
    if index is None:
        return 0
    cache = {}
    for t in toks:
        key = (t.page, t.s)
        hit = cache.get(key)
        if hit is None:
            hit = index.lookup(t.page, t.s)
            cache[key] = hit
        t.ast_path, t.container = hit
    for u in units:
        t = u.toks[0]
        u.ast_path = t.ast_path
        u.container = t.container
        path = t.ast_path or ""
        u.in_span = "y" if ">Span" in path or ">Inline" in path else ""
        u.in_utt = "y" if ">Utterance" in path else ""
        u.in_field = "y" if ">Field" in path else ""
    return len(cache)


# ---------------------------------------------------------------------------
# baseline (standard) tokenizers - what the corpus looks like to other tools
# ---------------------------------------------------------------------------
RE_WS = re.compile(r"\S+")
RE_WORDPUNCT = re.compile(r"\w+|[^\w\s]+", re.UNICODE)
RE_SKLEARN = re.compile(r"(?u)\b\w\w+\b")        # sklearn CountVectorizer default
RE_ALPHA = re.compile(r"[A-Za-z]+")
RE_BASELINE = {"ws": RE_WS, "wordpunct": RE_WORDPUNCT, "sklearn": RE_SKLEARN,
               "alpha": RE_ALPHA}


def baseline_spans(name, text):
    """`{(start, end): token text}` - the same four tokenizers, with offsets.

    The evaluation in 4.1 scores a unit / part / seam / operator by **source
    span**, not by string membership: an item counts as kept only when the
    baseline emits one token covering exactly that item's own offsets in that
    page.  A page-level bag-of-strings test would let a same-page match
    elsewhere in the page stand in for the item, and it would score
    `CountVectorizer` wrong in the other direction too (the real library
    lower-cases, so `Soul/gram` is not in its token set).  Offsets remove both
    effects: a span determines its own text, so no case folding can hide a unit.

    `sklearn`'s offsets come from its own default `token_pattern`;
    `baseline_span_check()` asserts token-for-token agreement with the
    installed `CountVectorizer` (whose only difference is case), so the offsets
    are the library's, not a look-alike's.
    """
    rx = RE_BASELINE[name]
    return {(m.start(), m.end()): m.group(0) for m in rx.finditer(text)}


def baseline_span_check(name, text):
    """Is `baseline_spans(name, text)` the same tokenization as `baseline_tokens`?

    Returns True when the two agree token-for-token in length and offset.
    For `sklearn` the texts are compared case-folded, because that is the one
    documented difference between the default pattern and the library.
    """
    spans = sorted(baseline_spans(name, text))
    toks = baseline_tokens(name, text)
    if len(spans) != len(toks):
        return False
    if name == "sklearn":
        return [text[s:e].lower() for s, e in spans] == list(toks)
    return [text[s:e] for s, e in spans] == list(toks)


def baseline_tokens(name, text):
    """Tokenize `text` the way a standard tool would.

    ws         - `str.split()`, the naive baseline
    wordpunct  - NLTK's `wordpunct_tokenize` scheme (re-implemented; nltk is
                 not installed in the `marker` env), `\\w+|[^\\w\\s]+`
    sklearn    - `sklearn.feature_extraction.text.CountVectorizer`'s default
                 token_pattern `(?u)\\b\\w\\w+\\b` (the real library when it
                 imports, the same regex otherwise)
    alpha      - `[A-Za-z]+`, the strictest letters-only view
    """
    if name == "ws":
        return text.split()
    if name == "wordpunct":
        return RE_WORDPUNCT.findall(text)
    if name == "sklearn":
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer()
            return cv.build_analyzer()(text)
        except Exception:
            return RE_SKLEARN.findall(text)
    if name == "alpha":
        return RE_ALPHA.findall(text)
    raise ValueError(name)


BASELINES = ("ws", "wordpunct", "sklearn", "alpha")


def baseline_uses_real_lib():
    """Is the `sklearn` baseline the real library, or its regex stand-in?"""
    try:
        import sklearn.feature_extraction.text as _sk
        return hasattr(_sk, "CountVectorizer")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def two_proportion_z(a, n1, b, n2):
    p1, p2 = a / n1, b / n2
    p = (a + b) / (n1 + n2)
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    return (p1 - p2) / se if se else 0.0


def _stripped(pno, src=SRC):
    """#12's page view: layout comments and markdown header lines removed."""
    body = OG.load_pages_raw(src)[pno]
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    return re.sub(r"^#+[ \t]+.*$", "", body, flags=re.M)


def n12_style_pairs(pages):
    """#12's linker-inventory method, recomputed here so the named-construct
    reconciliation can be stated exactly rather than guessed.

    #12 (`work/operator_functions.linker_evidence`) matches
    `word (sep word)+` with a *regex over the stripped page text*, so a pair
    inside a longer chain (`gene=TV/spasm`) still counts.  This tokenizer cuts
    maximal units, so a chained construct is one longer unit and its substring
    is no longer a unit of its own.  Both numbers are reported; the difference
    is the chain effect, and it is the reason the two disagree.
    """
    raw = "\n".join(pages[p] for p in sorted(pages))
    word = r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?"
    out = {}
    for sep, label, chain in (("=", "equals", False), ("/", "slash", True),
                             ("+", "plus", False), ("++", "plusplus", False),
                             ("-", "hyphen", False)):
        # #12 uses a *pair* regex everywhere except `/`, where it uses a chain
        # (slash chains are the common case: `soul/gram`, `a/b/c`).
        pat = re.compile(word + (f"(?:{re.escape(sep)}{word})+"
                                 if chain else f"{re.escape(sep)}{word}"))
        found = pat.findall(raw)
        out[label] = {"n": len(found), "unique": len(set(found)),
                      "counter": collections.Counter(found)}
    return out


def substring_host_counts(units, bodies, needle):
    """How many raw occurrences of `needle` are (a) a whole unit, (b) inside a
    longer unit.  Page-cited, and used to explain #12 vs this tokenizer."""
    whole = inside = orphan = 0
    hosts = collections.Counter()
    first = {}
    for pno in sorted(bodies):
        hits = [u for u in units if u.page == pno]
        starts = [u.s for u in hits]
        for m in re.finditer(re.escape(needle), bodies[pno]):
            off = m.start()
            i = bisect.bisect_right(starts, off) - 1
            hit = None
            for cand in reversed(hits[max(0, i - 4):i + 1]):
                if cand.s <= off < cand.e:
                    hit = cand
                    break
            if hit is None:
                orphan += 1
                continue
            if hit.text == needle:
                whole += 1
                first.setdefault(pno, (hit.s, hit.ast_path))
            else:
                inside += 1
                hosts[hit.text] += 1
    return {"needle": needle, "raw": whole + inside + orphan, "whole_unit": whole,
            "inside_longer_unit": inside, "not_a_unit": orphan,
            "hosts": hosts.most_common(8), "first": first}


def unit_type_rows(units):
    """Type inventory of units: surface -> count, pages, class, parts."""
    agg = collections.defaultdict(lambda: {"n": 0, "pages": collections.Counter(),
                                          "cls": "", "parts": ()})
    for u in units:
        a = agg[u.text]
        a["n"] += 1
        a["pages"][u.page] += 1
        a["cls"] = u.cls
        a["parts"] = tuple(t for t, r in u.parts if r == "w")
    rows = []
    for text, a in agg.items():
        first = next(u for u in units if u.text == text)
        rows.append({"unit_type": text, "lower": text.lower(), "n": a["n"],
                     "n_pages": len(a["pages"]), "class": a["cls"],
                     "n_words": len(a["parts"]),
                     "parts": "|".join(a["parts"]),
                     "first_page": first.page, "first_start": first.s,
                     "ast": first.ast_path,
                     "page_list": ",".join(str(p) for p in sorted(a["pages"]))})
    rows.sort(key=lambda r: (-r["n"], r["unit_type"]))
    return rows


# Curated constructs, asserted present at build time. `kind="unit"` requires the
# needle to be a whole unit; `kind="token"` marks a case about a token that is
# deliberately *not* a unit (e.g. the `</p>` OCR artifact).
CASES = (
    ("C1", "gene=TV", "the issue's naming unit: `=` as identity linker (#12)",
     "unit"),
    ("C2", "soul/gram", "the issue's compound unit: glued `/` (#12 dual role)",
     "unit"),
    ("C3", "suck=blood", "the issue's naming unit inside a `<<…>>` span", "unit"),
    ("C4", "gene=TV/spasm", "`=` and `/` chained inside one unit (2 seams)", "unit"),
    ("C5", "boy-roid", "hyphen compound: one WORD token, two `w` parts", "unit"),
    ("C6", "vital=serum", "second `=` naming unit", "unit"),
    ("C7", "retro-ADAM", "hyphen compound with a capitalised part", "unit"),
    ("C8", "cold-blooded", "two-hyphen compound, three `w` parts", "unit"),
    ("C9", "1-84068-060-1", "numeric hyphen chain: digits belong to the lexical "
                            "unit, not to a separate channel", "unit"),
    ("C10", "</p>", "source-HTML page furniture that survived OCR: the `/` has "
                    "no WORD on its left, so it never becomes a seam, and the "
                    "letter between `<` and `>` is kept as a unit "
                    "(`html_tag_residue`)", "token"),
    ("C11", "//++", "`++` opening a segment: seam with no left word", "token"),
    ("C12", "Soul/gram", "case variant of `soul/gram`, kept as its own type",
     "unit"),
    ("C13", "vital=serums", "seam unit whose right part is a plural", "unit"),
    ("C14", "acidHUMAN-ISM", "hyphen compound with an all-caps left part",
     "unit"),
    ("C15", "soul/gram::the", "a unit bounded by a `::` record field: the unit "
                             "stops at `::`, the field carries on", "token"),
    ("C16", "ANDROID-beat-ANDROID-vital-fake-ANDROID-resuscitation-second-"
            "respiration-ANDROID-hypertext-desire-mechanism-emotional-invasion-"
            "ANDROID-dog-X-joint-spectre-void",
     "longest unit in the book (p216): one hyphen chain, 21 parts", "unit"),
)


def case_rows(units, bodies):
    """Named constructs from the issue text plus their siblings, page-cited.

    Each case is asserted present at build time (the caller checks that every
    case id was located), so a change in tokenization cannot silently drop one.
    """
    cases = []
    seen = set()

    unit_ix = collections.defaultdict(list)
    for u in units:
        unit_ix[u.page].append(u)

    def unit_at(pno, off):
        """The innermost unit covering `off`."""
        cands = [u for u in unit_ix[pno] if u.s <= off < u.e]
        return min(cands, key=lambda u: (u.e - u.s, u.s)) if cands else None

    def add(cid, needle, note, kind="unit"):
        """First *exact* occurrence of `needle` that is itself a unit.

        `kind="token"`: the case is about a token, not a unit (e.g. the `</p>`
        OCR artifact, whose `/` never binds), so no unit match is required.
        """
        for pno in sorted(bodies):
            b = bodies[pno]
            for m in re.finditer(re.escape(needle), b):
                off = m.start()
                hit = unit_at(pno, off)
                if kind == "unit" and (hit is None or hit.text != needle):
                    continue
                key = (cid, pno, m.start())
                if key in seen:
                    continue
                seen.add(key)
                anchor = hit if kind == "unit" else None
                s = anchor.s if anchor else off
                e = anchor.e if anchor else off + len(needle)
                ctx = b[max(0, s - 45):s].replace("\n", "\\n")
                ctx2 = b[e:e + 45].replace("\n", "\\n")
                if anchor is not None:
                    parts = "|".join(f"{t}:{r}" for t, r in anchor.parts)
                    unit_txt, cls, sig, nw = (anchor.text, anchor.cls,
                                              anchor.sig, anchor.n_parts)
                else:
                    parts, unit_txt, cls, sig, nw = "", needle, "not-a-unit", "", 0
                cases.append({
                    "case": cid, "kind": kind, "needle": needle,
                    "page": pno, "start": s, "end": e, "unit": unit_txt,
                    "class": cls, "sig": sig, "n_words": nw, "parts": parts,
                    "container": (anchor.container if anchor else "")
                    or (hit.container if hit else ""),
                    "ast": (anchor.ast_path if anchor
                            else (hit.ast_path if hit else "")),
                    "left_ctx": ctx, "right_ctx": ctx2, "note": note})
                break
        return None

    for cid, needle, note, kind in CASES:
        add(cid, needle, note, kind)
    return cases


# ---------------------------------------------------------------------------
# artifact writers
# ---------------------------------------------------------------------------
def dump_tsv(name, rows, cols, out):
    with open(f"{out}/{name}", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(_cell(r.get(c, "")) for c in cols) + "\n")


def md_code(s):
    """A markdown code span; a backtick needs a spaced double fence."""
    return f"`` {s} ``" if "`" in s else f"`{s}`"


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "y" if v else "n"
    return str(v).replace("\t", " ").replace("\n", "\\n")


def _one_per_case(rows):
    """First occurrence of each case id, in CASES order."""
    seen, out = set(), []
    for cid, _n, _note, _kind in CASES:
        for r in rows:
            if r["case"] == cid and cid not in seen:
                seen.add(cid)
                out.append(r)
                break
    return out


def write_spec(S, out):
    """tokenizer_spec.md - assembled from the measured summary (no hand numbers)."""
    hdr = S["seam_loss"]["baselines"]
    L = []
    A = L.append
    A("# Operator-aware tokenizer (#17)\n")
    A(f"Source: `{S['src']}` - the vetted layout markdown (#7), {S['n_pages']} pages. "
      f"Token stream from #10/#12 (`work/operator_functions.tokenize`), operator "
      f"classes from #12, structural context from #14's AST. Reproducible via "
      f"`python work/operator_tokenizer.py`; regression suite "
      f"`work/test_operator_tokenizer.py`.\n")
    A("## 1. Why a standard tokenizer is the wrong tool here\n")
    A("The three units named in the issue are one lexical item each, and each has "
      "a seam and parts that are separately meaningful:\n")
    for cid, needle in (("C1", "gene=TV"), ("C2", "soul/gram"), ("C3", "suck=blood")):
        row = next((c for c in S["case_rows_all"] if c["case"] == cid), None)
        if row:
            A(f"- `{needle}` - p{row['page']} offsets {row['start']}-{row['end']}, "
              f"`{row['ast'] or 'no container'}`, parts `{row['parts']}`")
    A("")
    A("The last part is the point: the unit carries both the whole lexical item "
      "and the parts a downstream analysis needs, and the seam token between "
      "them is never thrown away.\n")
    A("Whitespace splitting keeps the unit and loses the parts; a word-tokenizer "
      "keeps the parts and loses the unit; a `\\w+` tokenizer loses the operator "
      "outright. Measured over every unit occurrence in the book (§4.1):\n")
    A("| tokenizer | tokens | unit kept | parts kept | seams kept |")
    A("|---|---:|---:|---:|---:|")
    for b in BASELINES:
        h = hdr.get(b)
        if not h:
            continue
        A(f"| `{b}`{'' if b != 'sklearn' else ' (CountVectorizer default)'} "
          f"| {h['n_tokens']:,} | {h['unit_kept_rate']*100:.1f}% | "
          f"{h['parts_kept_rate']*100:.1f}% | {h['seams_kept_rate']*100:.1f}% |")
    A(f"| **`operator-aware` (this)** | {S['tokens']['total']:,} | "
      f"100.0% | 100.0% | 100.0% |")
    A("")
    A("`ws` keeps the unit but never emits a part on its own, so the parts are "
      "unreachable; `wordpunct` emits every part and every seam but no unit, so "
      "the unit is unreachable; `sklearn`/`alpha` additionally drop the "
      "operators. This tokenizer emits all three views from one pass, which is "
      "the whole point of the issue.\n")
    A("## 2. The unit grammar\n")
    A("```ebnf")
    A("stream  = { region } ;")
    A("region  = header | para ;          (* layout placeholder comments are")
    A("                                       not book text: #7/#10/#14 *)")
    A("token   = word | op | punct ;")
    A("unit    = word { seam word | join word } ;   (* maximal *)")
    A("seam    = \"=\" | \"++\" | \"+\" | \"/\" ;   (* #12 linkers *)")
    A("join    = \"-\" | \"—\" | \"–\" ;        (* word-internal or glued dash *)")
    A("part    = word | seam | join ;    (* roles w / seam / join *)")
    A("```")
    A("")
    A("`unit` is maximal: a unit ends at the first token that is neither a seam "
      "onto a following word nor a dash glued to one. Precedence is inherited from #14 (span > utterance > "
      "field > segment > atom/link): `::`, `//`, `<<`, `>>`, `<`, `>`, `|`, the "
      "three dashes and `. , ; :` are **segment** operators and never bind, so "
      "`soul/gram::head` is two units plus a field boundary, never one unit.\n")
    A("### 2.1 Seam rules (each one a #12 measurement, not a guess)\n")
    A("| seam | rule | evidence |")
    A("|---|---|---|")
    A("| `=`, `++`, `+` | WORD on both sides | #12: `=` glued-L 0.98, glued-R "
      "0.95, interior 0.92 - a tight lexical linker |")
    A("| `/` | glued to a following WORD | #14: 952 `/` Link nodes vs 540 "
      "sentence-final `Slash` nodes; #12: dual role (compound/possessive + "
      "sentence-final) |")
    A("| everything else | never a seam | #12: `::` 0.78 interior but 13% "
      "upper-next = phrase-level; `//` 53% upper-next = sentence break |")
    A("")
    A("A binding seam also needs a WORD on its **left** and must stay inside one "
      "layout item (a blank line or a heading is not glue). That is stricter "
      "than #14's Link test, which only looks right, so the gap is reported "
      "rather than absorbed: "
      f"{S['unbound_ops'].get('/', 0):,} "
      "`/` tokens do not bind here - "
      f"{S['op_roles_by_op'].get('seam_orphan_left|/', 0) + S['op_roles_by_op'].get('slash_boundary|/', 0):,} "
      "of them are #14 Link nodes with no left WORD or crossing a layout item "
      "(`seam_orphan_left`, `slash_boundary`), and the remaining "
      f"{S['op_roles_by_op'].get('slash_free|/', 0) + S['op_roles_by_op'].get('seam_orphan_right|/', 0):,} "
      "are #14's sentence-final `Slash` nodes. All of them are rows in the "
      "ledger in 5.\n")
    A("### 2.2 Losslessness\n")
    A(f"Every token carries the exact source text between it and the previous "
      f"token, so `\"\".join(gap + text)` plus the page tail rebuilds the page "
      f"body byte-for-byte: "
      f"**{S['lossless']['ok']}/{S['n_pages']} pages** "
      f"({S['tokens']['total']:,} tokens). Header `#` runs and layout comments "
      f"live in gaps, so dropping them from the token stream loses no source "
      f"text. This is the tokenizer's primary correctness criterion, the same "
      f"one #14 used for its AST.\n")
    A("In `tokens.tsv` that column is `gap_json`, a JSON string literal "
      "(`json.loads` it) - a raw gap can contain a newline, so a bare field "
      "would break a line-oriented reader; `tokens.jsonl` carries the same "
      "field already decoded, and `units.tsv` carries the unit surface with "
      "`\\n`/tab escaped.\n")
    A(f"The tail is the page text after its last token - the one span no gap can "
      f"carry - so it is published in `page_tails.tsv` "
      f"(`page`, `last_token_end`, `tail_json`, `n_chars`): {S['tail_chars']:,} "
      f"characters over {S['n_pages']} pages, whitespace only except for the two "
      f"figure-block comments on pages {S['tail_comment_pages']}. Page body = "
      f"`\"\".join(gap + text)` + `tail_json`, checkable from the artifacts "
      f"alone.\n")
    A(f"**Read `tokens.tsv` with a plain `line.split(\"\\t\")`, not a quote-aware "
      f"reader.** The corpus contains {S['n_quote_tokens']} bare `\"` tokens (on "
      f"pages {S['quote_token_pages']}), so `csv.DictReader(delimiter=\"\\t\")` - "
      f"the obvious way to read a `.tsv` - mis-sizes {S['n_csv_bad_rows']} rows "
      f"from the first one on and silently loses fields. `text_json` (the token "
      f"text as a JSON string literal, like `gap_json`) is the quote-safe column, "
      f"and `tokens.jsonl` is quote-safe end to end.\n")
    A("## 3. Inventory\n")
    tk = S["tokens"]
    u = S["units"]
    A("| level | occurrences | types | notes |")
    A("|---|---:|---:|---|")
    A(f"| tokens | {tk['total']:,} | {tk['types']:,} | "
      f"{tk['WORD']:,} WORD + {tk['OP']:,} OP + {tk['PUNCT']:,} PUNCT |")
    A(f"| units | {u['total']:,} | {u['types']:,} | "
      f"maximal `word (seam word)*` runs |")
    A(f"| ... of which fused (seam or dash join) | {u['fused_occurrences']:,} | "
      f"{u['fused_types']:,} | invisible to a word-level count |")
    A(f"| ... of which seam-bearing | {u['seam_occurrences']:,} | "
      f"{u['seam_types']:,} | the `/`, `=`, `++`, `+` units |")
    A(f"| unit word parts | {u['parts']:,} | {u['part_types']:,} | "
      f"distinct word strings inside units |")
    A(f"| ... of which in multi-part units | {u['multi_word_occurrences']:,} | | "
      f"a unit with two or more word parts |")
    A(f"| seams (operators doing lexical work) | {u['seams']:,} | "
      f"{len(S['seams']):,} operator kinds | |")
    A(f"| dash joins (`-`/`—`/`–` inside a unit) | {u['joins']:,} | | "
      f"exposed as `join` parts; {u['hyphens']:,} of them ASCII `-` |")
    A("")
    A("Unit shapes:\n")
    A("| class | units | share | distinct types |")
    A("|---|---:|---:|---:|")
    for r in S["class_rows"]:
        A(f"| `{r['class']}` | {r['n']:,} | {r['share']*100:.1f}% | {r['types']:,} |")
    A("")
    A(f"Chain lengths (words per unit): {S['units']['chain_hist']}\n")
    A(f"Seam counts per unit: {S['units']['seams_per_unit_hist']}\n")
    A("## 4. Evaluation\n")
    A("### 4.1 Seam loss under standard tokenizers\n")
    A("Measured over all unit occurrences. **An item counts as kept only when the "
      "baseline emits one token covering exactly that item's own source span on "
      "its own page** - offset-exact, not string membership. Two reasons it is "
      "spans and not strings: a page-level bag would let a same-page match "
      "elsewhere stand in for the item, and it scores the real "
      "`CountVectorizer` in the wrong direction (the library lower-cases, so "
      "`Soul/gram` is simply absent from its token set and would be scored as "
      "lost). A span determines its own text, so case folding cannot hide a "
      "unit. The `sklearn` offsets are its own default `token_pattern`, asserted "
      "token-for-token against the installed `CountVectorizer` on all "
      f"{S['n_pages']} pages (the only difference is case):\n")
    A("Rates, with the raw counts in parentheses:")
    A("")
    A("| baseline | tokens | unit kept | word parts kept | seams kept | "
      "operator tokens kept |")
    A("|---|---:|---:|---:|---:|---:|")
    for b in BASELINES:
        h = hdr.get(b)
        if h:
            A(f"| `{b}` | {h['n_tokens']:,} | "
              f"{h['unit_kept_rate']*100:.1f}% ({h['unit_kept']:,}/"
              f"{h['n_units']:,}) | "
              f"{h['parts_kept_rate']*100:.1f}% ({h['parts_kept']:,}/"
              f"{h['n_parts']:,}) | "
              f"{h['seams_kept_rate']*100:.1f}% ({h['seams_kept']:,}/"
              f"{h['n_seams']:,}) | "
              f"{h['ops_kept_rate']*100:.1f}% ({h['ops_kept']:,}/"
              f"{h['n_ops']:,}) |")
    A(f"| **`operator-aware`** | {tk['total']:,} | 100.0% "
      f"({S['units']['total']:,}/{S['units']['total']:,}) | 100.0% "
      f"({S['units']['parts']:,}/{S['units']['parts']:,}) | 100.0% "
      f"({S['units']['seams']:,}/{S['units']['seams']:,}) | 100.0% "
      f"({tk['OP']:,}/{tk['OP']:,}) |")
    A("")
    A("The rightmost column is where the standards lose the operator layer. "
      "`wordpunct` emits `[^\\w\\s]+` runs, so a seam *string* survives - but it "
      "merges adjacent operators into one token (`=+`, `<<>>`), so only "
      f"{hdr['wordpunct']['ops_kept']:,} of {hdr['wordpunct']['n_ops']:,} "
      f"operator tokens ({hdr['wordpunct']['ops_kept_rate']*100:.1f}%) come back "
      "as one token spanning exactly that operator. `sklearn` and `alpha` "
      f"recover none. `ws` recovers {hdr['ws']['ops_kept']:,}, and only where an "
      "operator is a whole whitespace-delimited token of its own.\n")
    fus0 = hdr[BASELINES[0]]["fused"]
    A("Restricted to the **fused** units - the "
      f"{fus0['n']:,} occurrences that actually carry a seam or a hyphen join, "
      "which is the whole point of the issue:")
    A("")
    A("| baseline | unit kept | word parts kept | seams kept |")
    A("|---|---:|---:|---:|")
    for b in BASELINES:
        f = hdr[b]["fused"]
        A(f"| `{b}` | {f['unit_kept']/f['n']*100:.1f}% "
          f"({f['unit_kept']:,}/{f['n']:,}) | "
          f"{f['parts_kept']/f['parts']*100:.1f}% "
          f"({f['parts_kept']:,}/{f['parts']:,}) | "
          f"{f['seams_kept']/f['seams']*100:.1f}% "
          f"({f['seams_kept']:,}/{f['seams']:,}) |")
    A(f"| **`operator-aware`** | 100.0% ({fus0['n']:,}/{fus0['n']:,}) | "
      f"100.0% ({fus0['parts']:,}/{fus0['parts']:,}) | 100.0% "
      f"({fus0['seams']:,}/{fus0['seams']:,}) |")
    A("")
    A("This is the finding the issue asks for. `ws` keeps a fused unit as one "
      f"token but loses {100 - hdr['ws']['fused']['parts_kept']/fus0['parts']*100:.0f}% "
      "of its word parts, so the parts are unreachable. `wordpunct` keeps "
      f"{hdr['wordpunct']['fused']['parts_kept']/fus0['parts']*100:.1f}% of them "
      "and every seam as a token, but splits the unit, so the unit is "
      "unreachable: a word-frequency list built on it can never count the "
      f"{S['named']['soul/gram']} occurrences that *are* one lexical item as one "
      f"item - and "
      f"`sklearn`/`alpha` lose the operators outright. Each standard tokenizer "
      "keeps at most two of the three views; this one keeps all three from a "
      "single pass.\n")
    A("By unit class (share of occurrences where the *whole* unit survives as one "
      "baseline token):\n")
    A("| class | units | " + " | ".join(f"`{b}`" for b in BASELINES) + " |")
    A("|---|---:|" + "---:|" * len(BASELINES))
    for r in S["class_seam_rows"]:
        A(f"| `{r['class']}` | {r['n']:,} | " +
          " | ".join(f"{r['kept'][b]*100:.0f}%" for b in BASELINES) + " |")
    A("")
    A("### 4.2 Reconciliation with the earlier issues\n")
    A("| quantity | earlier | this tokenizer |")
    A("|---|---:|---:|")
    for r in S["recon_rows"]:
        A(f"| {r['what']} | {r['earlier']} | {r['here']} |")
    A("")
    A("### 4.3 The named constructs, page-cited\n")
    A("| case | construct | kind | page | offsets | unit | parts | container |")
    A("|---|---|---|---:|---|---|---|---|")
    for c in S["case_rows_head"]:
        A(f"| {c['case']} | `{c['needle']}` | {c['kind']} | {c['page']} | "
          f"{c['start']}-{c['end']} | `{c['unit']}` | `{c['parts'] or '-'}` | "
          f"`{c['container'] or '-'}` |")
    A("")
    A(f"All {len(S['case_rows_all'])} cases, with their context lines and the four "
      f"baseline tokenizations of the same span, are in `cases.tsv` and "
      f"`case_examples.txt`.\n")
    A("### 4.4 The named constructs, with the #12 discrepancy explained\n")
    A("| construct | raw | whole unit | inside a longer unit | #12 (recomputed) | longer units it sits in |")
    A("|---|---:|---:|---:|---:|---|")
    for r in S["named_rows"]:
        A(f"| `{r['needle']}` | {r['raw']} | {r['whole_unit']} | "
          f"{r['inside_longer_unit']} | {r['prior']} "
          f"(recomputed {r['prior_recomputed']}) | "
          f"{r['host_units'][:70] or '-'} |")
    A("")
    A("The gap is not a disagreement about the text: it is what a *unit* means. "
      "#12 counted `word=word` and `word/word` **regex pairs**, so `gene=TV` "
      "inside `gene=TV/spasm` is a pair. This tokenizer cuts **maximal** units, "
      "so that string is one unit whose seam chain is `=/` and `gene=TV` is no "
      "longer a unit in its own right. #12's numbers recomputed here with its own "
      "method match its published inventory, which confirms the delta is the "
      "maximality rule and not a tokenization error. Both views are published: "
      "`units.tsv` (maximal) and `named_constructs.tsv` (per-construct "
      "breakdown).\n")
    A("### 4.5 What the units add that word counts do not\n")
    for line in S["insight_lines"]:
        A(f"- {line}")
    A("")
    A("## 5. Diagnostics (nothing silently repaired)\n")
    A("Every one of the "
      f"{S['op_roles'].get('seam', 0) + S['op_roles'].get('segment', 0) + sum(n for k, n in S['op_roles'].items() if k not in ('seam', 'segment')):,} "
      "operator tokens gets exactly one role, so the ledger's rows add up to the "
      "token stream and nothing is dropped without a row:\n")
    A("| role | n | pages | meaning | first page-cited example |")
    A("|---|---:|---:|---|---|")
    for r in S["diag_rows"]:
        A(f"| `{r['diag']}` | {r['n']} | {r['pages']} | {r['meaning']} | "
          f"{r['example']} |")
    A("")
    A("Roles `seam` and `segment` are listed for reconciliation, not as problems: "
      f"{S['op_roles'].get('seam', 0):,} seams bind and "
      f"{S['op_roles'].get('segment', 0):,} segment operators cut. The three "
      "`seam_orphan_*` / `slash_free` rows are the residue, and every one of them "
      "is still present in `tokens.tsv` as an OP token with its exact offsets. The "
      "`slash_free` … `seam_orphan_side` rows plus the two `(role)` rows are the "
      f"operator partition: they add up to the {S['tokens']['OP']:,} operator "
      "tokens. `seam_spaced` is a *flag* on seam tokens rather than a role, and "
      "`html_tag_residue` is a unit-level artifact, so neither is in it. That "
      "artifact row is worth reading: the page bodies still carry the source "
      "PDF's own markup - page furniture (`<th>`, `<pre>`, `<li>`, `<br/>`, "
      "`<hr/>`) and a block of `mailto:` anchors (`</a>`) - so 229 tag names "
      "arrive as one- or two-character units. They are counted here, not "
      "repaired: this tokenizer deletes nothing, and #6/#19 own that decision.\n")
    if S.get("never_fired"):
        A("Never fired on this corpus (negative result): "
          + ", ".join(f"`{d}`" for d in S["never_fired"]) + "\n")
    A("## 6. Artifacts\n")
    A("| file | contents |")
    A("|---|---|")
    for name, desc in S["artifact_rows"]:
        A(f"| `{name}` | {desc} |")
    A("")
    A("## 7. Scope and honest limits\n")
    for line in S["limit_lines"]:
        A(f"- {line}")
    A("")
    A("## 8. Downstream contract\n")
    for line in S["downstream_lines"]:
        A(f"- {line}")
    A("")
    A("---\n*Generated by `work/operator_tokenizer.py` (schema "
      f"`{SCHEMA}`).*")
    with open(f"{out}/tokenizer_spec.md", "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
DIAG_MEANING = {
    "seam_orphan_left": "a seam operator with no WORD on its left, so it binds "
                        "nothing (e.g. `::/to`, or the `/` of a `</p>` tag): "
                        "kept as an OP token, never a seam",
    "seam_orphan_right": "a seam operator with no WORD on its right (segment, "
                         "page or comment end): kept as an OP token",
    "slash_free": "whitespace-bounded `/`: #14's sentence-final terminator "
                  "(540 Slash nodes), never part of a unit",
    "seam_spaced": "a seam with whitespace on one side (still two words, so it "
                   "binds): counted, and flagged so the glue measurement stays "
                   "auditable",
    "seam_orphan_side": "an `=`/`++`/`+` at a layout-item boundary (blank line or "
                        "heading) or at the edge of the page: binds nothing",
    "slash_boundary": "a `/` glued to a following word but crossing a layout "
                      "item: #14 counts it a Link node, this tokenizer does not "
                      "bind it into a unit",
    "unit_straddles_boundary": "a unit containing a `::`/`//`/delimiter - must "
                               "be 0",
    "empty_unit": "a unit with no word part - must be 0",
    "html_tag_residue": "a unit produced *inside* a real HTML tag the source "
                        "carried (page furniture, `mailto:` anchors): the tag's "
                        "own name becomes a 1-2 character 'word'. Kept, not "
                        "repaired - #6/#19 own that decision",
}


def build(src=SRC, out=OUT, with_ast=True):
    os.makedirs(out, exist_ok=True)
    bodies = OG.load_pages_raw(src)
    pages = sorted(bodies)
    all_toks, page_tails, page_units = {}, {}, {}
    units, diags = [], []
    lossless_bad = []
    for pno in pages:
        toks, _d, tail = tokenize_page(pno, bodies[pno])
        page_tails[pno] = tail
        if rebuild(toks, tail) != bodies[pno]:
            lossless_bad.append(pno)
        all_toks[pno] = toks
        us, _d = build_units(toks)
        page_units[pno] = us
        units.extend(us)
    if lossless_bad:
        raise SystemExit(f"losslessness failure on pages {lossless_bad[:5]}")

    # ---- AST enrichment (#14) ------------------------------------------
    index = None
    if with_ast:
        try:
            index = AstIndex(src)
        except Exception as exc:                      # pragma: no cover
            print(f"warn: AST enrichment unavailable ({exc})")
    annotate([t for pno in pages for t in all_toks[pno]], units, index)

    # ---- unit-index fix: uid must be book-global -------------------------
    for k, u in enumerate(units):
        u.id = k

    violations = check_unit_invariants(
        [t for pno in pages for t in all_toks[pno]], units)
    if violations:
        raise SystemExit("unit invariant failure: " + violations[0])

    # The offset view (`part_spans`) and the role view (`Unit.parts`) must agree
    # text-for-text, and every part must be the exact slice at its own offsets -
    # otherwise the evaluation in 4.1 would be scoring the wrong things.
    for u in units:
        ps = part_spans(u)
        if [x[2] for x in ps] != [t for t, _r in u.parts]:
            raise SystemExit(f"part_spans/parts disagree on unit {u.id} p{u.page}")
        for s, e, txt, _r in ps:
            if bodies[u.page][s:e] != txt:
                raise SystemExit(f"part span mismatch in unit {u.id} p{u.page}")
        # `:` and `|` are segment operators, so they can never sit inside a unit:
        # that is what keeps the `parts` column (bare texts joined by `|`) and
        # the `text:role` notation of `cases.tsv` unambiguous to split on.
        if any(("|" in t or ":" in t) for t, _r in u.parts):
            raise SystemExit(f"part text carries a `parts` separator: unit {u.id}")

    # ---- inventories -----------------------------------------------------
    flat = [t for pno in pages for t in all_toks[pno]]
    kinds = collections.Counter(t.kind for t in flat)
    ops_all = collections.Counter(t.text for t in flat if t.kind == "OP")
    ops_body = collections.Counter(t.text for pno in pages for t in all_toks[pno]
                                   if t.kind == "OP" and t.region == "para")
    ops_head = collections.Counter(t.text for pno in pages for t in all_toks[pno]
                                   if t.kind == "OP" and t.region == "header")
    n_words = kinds["WORD"]
    assert n_words == EXPECTED_WORDS, \
        f"word-token drift: {n_words} != {EXPECTED_WORDS} (#14 Lex nodes)"
    assert sum(ops_body.values()) == EXPECTED_OPS_BODY, \
        f"body operator drift: {sum(ops_body.values())} != {EXPECTED_OPS_BODY}"
    assert sum(ops_head.values()) == EXPECTED_OPS_HEADER, \
        f"header operator drift: {sum(ops_head.values())} != {EXPECTED_OPS_HEADER}"
    for op, n in EXPECTED_OPS_TOKENS.items():
        assert ops_body[op] == n, \
            f"operator {op} drift in page bodies: {ops_body[op]} != {n} (#14 §3)"

    seam_ops = collections.Counter()
    for u in units:
        seam_ops.update(u.ops)
    n_seams = sum(seam_ops.values())
    # Every operator token gets exactly one role (full scan, no early exit), so
    # the ledger is complete and the parts sum back to the 11,000 tokens.
    roles = [r for pno in pages for r in classify_op_tokens(all_toks[pno])]
    role_n = collections.Counter(r["role"] for r in roles)
    role_op = collections.Counter((r["role"], r["op"]) for r in roles)
    unbound = collections.Counter(
        t.text for t in flat if t.kind == "OP"
        and t.unit is None and t.text not in SEGMENT_OPS)
    n_segment_ops = sum(1 for t in flat if t.kind == "OP"
                        and t.text in SEGMENT_OPS)
    assert len(roles) == kinds["OP"], "operator role scan lost a token"
    assert sum(role_n.values()) == kinds["OP"], "operator roles do not add up"
    assert role_n["seam"] + role_n["segment"] + \
        sum(n for r, n in role_n.items()
            if r not in ("seam", "segment")) == kinds["OP"], \
        "operator roles must partition the operator tokens"
    assert role_n["segment"] == n_segment_ops, \
        f"segment role {role_n['segment']} != segment-op tokens {n_segment_ops}"
    assert sum(n for r, n in role_n.items()
               if r not in ("seam", "segment")) == sum(unbound.values()), \
        "every unbound link-op token must have a non-segment ledger role"
    assert role_n["seam"] == n_seams, \
        f"seam role count {role_n['seam']} != seams in units {n_seams}"

    free_dash = sum(1 for t in flat if t.kind == "PUNCT"
                    and t.pclass == "dash" and not t.glue_L and not t.glue_R)
    assert free_dash == EXPECTED_FREE_DASH, \
        f"free dash drift: {free_dash} != {EXPECTED_FREE_DASH} (#13/#14)"

    # #11's *ordered* operator stream, page for page.  #10/#11 tokenize
    # operators with a greedy longest match over `OPS`; #12 (and therefore this
    # tokenizer) uses its own operator regex.  Matching totals would not prove
    # the two agree, so the sequences themselves are compared: if #15 or a future
    # parse changes the operator tokenization, this fails here instead of
    # quietly producing two different operator streams.
    n11_stream = {p: [op for run in runs for op in run]
                  for p, runs in NG.page_runs(src).items()}
    tok_stream = {p: [t.text for t in all_toks[p]
                      if t.kind == "OP" and t.region == "para"] for p in pages}
    n11_pages_diff = [p for p in pages if n11_stream.get(p) != tok_stream[p]]
    n11_total = sum(len(v) for v in n11_stream.values())
    assert not n11_pages_diff, (
        f"#11 ordered operator stream differs on pages {n11_pages_diff[:5]}")

    # Seam counts must equal #14's Link-node counts: #14 emits a Link node for
    # every `=`/`++`/`+` (left/right may be missing) and for `/` only when
    # glued-right with a following word.  So the arithmetic is
    #     seams + orphan-side seam ops = Link nodes
    # with the orphan side distributed exactly as #12's glue profile predicts.
    orph_all = {op: collections.Counter() for op in EXPECTED_LINK_NODES}
    for r in roles:
        if r["op"] in orph_all and r["role"] not in ("segment", "seam"):
            orph_all[r["op"]][r["role"]] += 1
    # Which orphan roles belong to #14's Link side and which to its Slash side.
    # For `=`/`++`/`+` every link operator is a Link node; for `/` only the ones
    # that are glued right onto a following word are (a `/` with nothing to its
    # right, or a whitespace-bounded one, is #14's sentence-final Slash).
    LINK_SIDE = {"/": ("seam_orphan_left", "slash_boundary")}
    for op, n in EXPECTED_LINK_NODES.items():
        orph = orph_all[op]
        link_orph = orph[LINK_SIDE[op][0]] + orph[LINK_SIDE[op][1]] \
            if op in LINK_SIDE else sum(orph.values())
        assert seam_ops[op] + link_orph == n, (
            f"`{op}` seam drift: seams={seam_ops[op]} orphans={dict(orph)} "
            f"vs #14 Link {n}")
    slash_terminators = (orph_all["/"]["slash_free"]
                         + orph_all["/"]["seam_orphan_right"])
    assert slash_terminators == EXPECTED_SLASH_NODES, \
        f"slash terminator drift: {slash_terminators} != {EXPECTED_SLASH_NODES}"

    # ---- structural invariants ------------------------------------------
    boundary_cross = 0
    seg = ("::", "//", "<<", ">>", "<", ">", "|")
    for u in units:
        for t in u.toks[1:]:
            if t.kind == "OP" and t.text in seg:
                boundary_cross += 1
    if boundary_cross:
        diags.append({"diag": "unit_straddles_boundary", "page": 0, "start": 0,
                      "op": "", "note": f"{boundary_cross} units contain a "
                                        f"segment operator"})
    empty_units = sum(1 for u in units if u.n_parts == 0)
    if empty_units:
        diags.append({"diag": "empty_unit", "page": 0, "start": 0, "op": "",
                      "note": f"{empty_units} units have no word part"})

    # markup residue (see HTML_TAGS): word tokens sitting inside a real HTML
    # tag run in a page body - one per tag, the tag's own name
    html_runs = {pno: html_tag_runs(bodies[pno]) for pno in pages}
    n_html_runs = sum(len(v) for v in html_runs.values())
    html_art = []
    for pno in pages:
        runs = html_runs[pno]
        for t in all_toks[pno]:
            if t.kind != "WORD":
                continue
            for s, e, tag in runs:
                if s <= t.s < e:
                    html_art.append({"page": pno, "start": t.s, "text": t.text,
                                     "tag": tag})
                    break
    assert len(html_art) == n_html_runs, (
        f"HTML tag runs {n_html_runs} != word tokens inside them "
        f"{len(html_art)}: the residue ledger would be wrong")
    html_tag_counts = collections.Counter(a["tag"] for a in html_art)

    # A seam with whitespace on one side still binds (two words), but it is not
    # glued, so it is flagged and reported separately: #12's `=` glue rates
    # (0.98/0.95) leave room for exactly this residue.
    spaced = [(t.page, t.s, t.text) for t in flat if t.kind == "OP"
              and t.unit is not None and (not t.glue_L or not t.glue_R)]

    # ---- classes, chains -------------------------------------------------
    cls_n = collections.Counter(u.cls for u in units)
    cls_types = collections.defaultdict(set)
    for u in units:
        cls_types[u.cls].add(u.text)
    class_rows = [{"class": c, "n": n, "share": n / len(units),
                   "types": len(cls_types[c])}
                  for c, n in cls_n.most_common()]
    chain_hist = collections.Counter(u.n_parts for u in units)
    seams_per_unit = collections.Counter(len(u.ops) for u in units)

    # ---- baseline seam loss ---------------------------------------------
    # Scored by source span (see `baseline_spans`): an item survives a baseline
    # only when that baseline emits one token covering exactly its own offsets
    # on the same page.
    loss_rows = []
    for bname in BASELINES:
        kept_unit = kept_parts = kept_seam = 0
        n_tok_b = n_seams_t = n_parts_seen = 0
        cls_kept = collections.Counter()
        cls_tot = collections.Counter()
        fused_u = fused_ok = 0
        fused_p = fused_p_ok = 0
        fused_s = fused_s_ok = 0
        op_kept = 0
        for pno in pages:
            text = bodies[pno]
            btoks = baseline_tokens(bname, text)
            if not baseline_span_check(bname, text):
                raise SystemExit(
                    f"baseline span/token mismatch for {bname} on page {pno}")
            bspans = baseline_spans(bname, text)
            n_tok_b += len(btoks)
            for t in all_toks[pno]:
                if t.kind == "OP" and (t.s, t.e) in bspans:
                    op_kept += 1
            for u in page_units[pno]:
                n_seams_t += len(u.ops)
                cls_tot[u.cls] += 1
                hit_u = (u.s, u.e) in bspans
                fused = bool(u.ops or u.n_join)
                if hit_u:
                    kept_unit += 1
                    cls_kept[u.cls] += 1
                for s, e, _txt, role in part_spans(u):
                    if role != "w":
                        continue
                    n_parts_seen += 1
                    ok = (s, e) in bspans
                    kept_parts += ok
                    if fused:
                        fused_p += 1
                        fused_p_ok += ok
                for t in u.toks:
                    if t.kind == "OP":
                        ok = (t.s, t.e) in bspans
                        kept_seam += ok
                        if fused:
                            fused_s += 1
                            fused_s_ok += ok
                if fused:
                    fused_u += 1
                    fused_ok += hit_u
        loss_rows.append({
            "baseline": bname, "n_tokens": n_tok_b,
            "unit_kept": kept_unit, "n_units": len(units),
            "unit_kept_rate": kept_unit / len(units),
            "parts_kept": kept_parts, "n_parts": n_parts_seen,
            "parts_kept_rate": kept_parts / n_parts_seen if n_parts_seen else 0.0,
            "seams_kept": kept_seam, "n_seams": n_seams_t,
            "seams_kept_rate": kept_seam / n_seams_t if n_seams_t else 0.0,
            "ops_kept": op_kept, "n_ops": kinds["OP"],
            "class_kept": {c: (cls_kept[c], cls_tot[c]) for c in cls_tot},
            "fused": {"n": fused_u, "unit_kept": fused_ok,
                      "parts": fused_p, "parts_kept": fused_p_ok,
                      "seams": fused_s, "seams_kept": fused_s_ok},
        })
        loss_rows[-1]["ops_kept_rate"] = (op_kept / kinds["OP"]
                                          if kinds["OP"] else 0.0)
    class_seam_rows = []
    for r in class_rows:
        c = r["class"]
        kept = {}
        for L in loss_rows:
            k, t = L["class_kept"].get(c, (0, 0))
            kept[L["baseline"]] = (k / t) if t else 0.0
        class_seam_rows.append({"class": c, "n": r["n"], "kept": kept})

    # ---- unit type inventory, part index, cases --------------------------
    ut_rows = unit_type_rows(units)
    part_rows = []
    for u in units:
        for pos, (s, e, txt, role) in enumerate(part_spans(u)):
            part_rows.append({"uid": u.id, "page": u.page, "unit": u.text,
                              "start": s, "end": e, "pos": pos,
                              "role": role, "text": txt})
    case_rows_ = case_rows(units, bodies)
    got_ids = {c["case"] for c in case_rows_}
    want_ids = {c[0] for c in CASES}
    if got_ids != want_ids:
        raise SystemExit(f"spot cases not located: {sorted(want_ids - got_ids)}")

    # ---- reconciliation table -------------------------------------------
    S_units_parts = sum(u.n_parts for u in units)
    S_units_joined = sum(1 for u in units if u.n_join)
    S_units_hyph_tokens = sum(1 for u in units for t in u.toks
                              if t.kind == "WORD" and "-" in t.text)
    S_units_hyphens = sum(u.n_hyp for u in units)
    recon = []
    recon.append({"what": "pages", "earlier": f"{EXPECTED_PAGES} (#7/#10/#14)",
                  "here": str(len(pages))})
    recon.append({"what": "word tokens (WORD)", "earlier": f"{EXPECTED_WORDS:,} (#14 Lex)",
                  "here": f"{n_words:,}"})
    recon.append({"what": "operator tokens, page bodies",
                  "earlier": f"{EXPECTED_OPS_BODY:,} (#10/#12)", "here": f"{sum(ops_body.values()):,}"})
    recon.append({"what": "operator tokens, headers",
                  "earlier": f"{EXPECTED_OPS_HEADER} (#14 §3)", "here": f"{sum(ops_head.values()):,}"})
    recon.append({"what": "operator tokens, whole book",
                  "earlier": f"{EXPECTED_OPS_ALL:,} (#10 + #14 headers)",
                  "here": f"{sum(ops_all.values()):,}"})
    for op in ("=", "/", "++", "+"):
        orph = orph_all[op]
        link_orph = sum(orph[r] for r in LINK_SIDE.get(op, tuple(orph)))
        here = f"{seam_ops[op]:,} seams"
        if op == "/":
            here += (f" + {link_orph} Link-side orphans ("
                     + ", ".join(f"{r} x{orph[r]}"
                                 for r in LINK_SIDE["/"] if orph[r])
                     + f") = {EXPECTED_LINK_NODES['/']:,} Link nodes; "
                     f"{slash_terminators:,} free/right-orphan `/` = "
                     f"{EXPECTED_SLASH_NODES} Slash nodes")
        else:
            here += f" + {link_orph} orphan-sided = {EXPECTED_LINK_NODES[op]:,}"
        recon.append({"what": f"`{op}` seams vs #14 Link/Slash nodes",
                      "earlier": f"{EXPECTED_LINK_NODES[op]:,} Link"
                                 + (f" / {EXPECTED_SLASH_NODES} Slash"
                                    if op == "/" else ""),
                      "here": here})
    ops_absent = [op for op in OPS if not ops_all.get(op)]
    recon.append({"what": "#10 operator set vs this token stream",
                  "earlier": f"{len(OPS)} operators (#10 `OPS`; #12 audits "
                             f"{len(ops_absent)} as absent from page bodies)",
                  "here": f"{len(OPS) - len(ops_absent)} present with the counts "
                          f"in `ops_all`; absent (0 tokens): "
                          + ", ".join(md_code(o) for o in ops_absent)
                          + " - #12 showed these occur only inside layout "
                            "placeholder comments, which yield no token"})
    recon.append({"what": "free-standing dashes",
                  "earlier": f"{EXPECTED_FREE_DASH} (#13/#14)", "here": f"{free_dash:,}"})
    recon.append({"what": "ordered operator stream vs #11",
                  "earlier": f"{n11_total:,} ops over 225 pages (#11 "
                             "`operator_stream.tsv`; #10's greedy longest-match "
                             "operator tokenization)",
                  "here": (f"identical operator-for-operator on all "
                           f"{len(pages)} pages (page bodies, order preserved)"
                           if not n11_pages_diff
                           else f"differs on {len(n11_pages_diff)} pages")})
    # named constructs: #12's number vs this tokenizer's, with the arithmetic
    n12 = n12_style_pairs({p: _stripped(p) for p in pages})
    named_rows = []
    for needle, prior, prior_src in (("gene=TV", 147, "#12 joined_pairs.tsv"),
                                     ("vital=serum", 130, "#12 joined_pairs.tsv"),
                                     ("suck=blood", 129, "#12 joined_pairs.tsv"),
                                     ("soul/gram", 317, "#12 chain inventory")):
        h = substring_host_counts(units, bodies, needle)
        h["prior"] = prior
        h["prior_src"] = prior_src
        # #12's own number, recomputed here so the delta is attributable
        h["prior_recomputed"] = n12["slash" if needle.endswith("gram")
                                    else "equals"]["counter"][needle]
        h["host_units"] = ";".join(f"{t}x{c}" for t, c in h["hosts"])
        h["first_page"] = min(h["first"]) if h["first"] else ""
        h["first_start"] = h["first"].get(h["first_page"], ("", ""))[0] \
            if h["first"] else ""
        del h["hosts"], h["first"]
        named_rows.append(h)
        recon.append({"what": f"`{needle}` (whole units)",
                      "earlier": f"{prior} ({prior_src}; recomputed here "
                                 f"{h['prior_recomputed']})",
                      "here": f"{h['whole_unit']} of {h['raw']} raw occurrences"})
    for h in named_rows:
        recon.append({"what": f"`{h['needle']}` inside a longer unit",
                      "earlier": "counted as its own pair by #12",
                      "here": f"{h['inside_longer_unit']}"
                               + (f" ({h['host_units']})"
                                  if h["host_units"] else "")})
    recon.append({"what": "hyphen compounds (units / tokens)",
                  "earlier": f"{n12['hyphen']['n']:,} (#12 regex `word-word` "
                             f"matches over stripped text; recomputed here "
                             f"identical)",
                  "here": f"{S_units_joined:,} units carry a dash join "
                          f"({sum(1 for u in units if u.n_hyp):,} with a word-"
                          f"internal `-`); "
                          f"{S_units_hyph_tokens:,} hyphenated WORD tokens "
                          f"holding {S_units_hyphens:,} `-` characters"})
    recon.append({"what": "hyphen compounds (why the numbers differ)",
                  "earlier": "regex match = one `word-word` pair, non-overlapping",
                  "here": f"a k-hyphen token (`cold-blooded-machine`) holds k "
                          f"hyphens but yields only {math.ceil(3/2)} regex "
                          f"match(es), so {n12['hyphen']['n']:,} matches < "
                          f"{S_units_hyphens:,} hyphens; this tokenizer counts "
                          f"tokens *and* joins, so all three numbers are "
                          f"published"})
    recon.append({"what": "word tokens vs the first pass",
                  "earlier": "74,760 (first pass, `docs/markdown/…md`, "
                             "regex `[A-Za-z][A-Za-z0-9]*([-=/][A-Za-z0-9]+)*` "
                             "so units *and* their parts fused into one count)",
                  "here": f"{n_words:,} WORD tokens (hyphen compounds joined) "
                          f"+ {len(units):,} maximal units + "
                          f"{S_units_parts:,} unit word parts - three "
                          f"different denominators, hence the gap"})
    recon.append({"what": "PUNCT tokens",
                  "earlier": "not counted by #10/#12 (they scan operators only)",
                  "here": f"{kinds['PUNCT']:,} (`tokens.tsv`, class in "
                          f"`pclass`); {free_dash:,} free dashes reproduce "
                          f"#13/#14"})
    for r in recon:
        assert "drift" not in r["here"], r

    # ---- diagnostics summary --------------------------------------------
    # Ledger rows: the non-seam roles come from the full operator scan (one row per
    # role, with a page-cited example), the invariant checks from `diags`.
    op_examples = {}
    for r in roles:
        op_examples.setdefault(r["role"], r)
    diag_rows = []
    for role in ("slash_free", "slash_boundary", "seam_orphan_left",
                 "seam_orphan_right", "seam_orphan_side", "seam_spaced"):
        if role == "seam_spaced":
            if spaced:
                diag_rows.append({"diag": role, "n": len(spaced),
                                  "pages": len({p for p, _s, _o in spaced}),
                                  "meaning": DIAG_MEANING[role],
                                  "example": (f"p{spaced[0][0]} offset "
                                              f"{spaced[0][1]} ({spaced[0][2]})")})
            continue
        r = op_examples.get(role)
        if not r:
            continue
        by_op = ", ".join(f"`{o}`x{n}" for (ro, o), n in sorted(role_op.items())
                          if ro == role)
        diag_rows.append({"diag": role, "n": role_n[role],
                          "pages": len({x["page"] for x in roles
                                        if x["role"] == role}),
                          "meaning": DIAG_MEANING[role] + f" ({by_op})",
                          "example": (f"p{r['page']} offset {r['start']}: "
                                      f"{r['op']} ({r['role']})")})
    for d in diags:
        diag_rows.append({"diag": d["diag"], "n": 1, "pages": 1,
                          "meaning": DIAG_MEANING.get(d["diag"], ""),
                          "example": d["note"]})
    if html_art:
        tags_desc = ", ".join(f"`{t}`x{n}" for t, n in html_tag_counts.most_common())
        diag_rows.append({"diag": "html_tag_residue", "n": len(html_art),
                          "pages": len({a["page"] for a in html_art}),
                          "meaning": DIAG_MEANING["html_tag_residue"]
                                     + f" ({tags_desc})",
                          "example": (f"p{html_art[0]['page']} offset "
                                      f"{html_art[0]['start']}: unit "
                                      f"{html_art[0]['text']!r} inside "
                                      f"{html_art[0]['tag']!r}")})
        assert sum(1 for r in diag_rows if r["diag"] == "html_tag_residue") == 1, \
            "html_tag_residue reported twice"
    assert sum(1 for r in diag_rows if r["diag"] == "seam_spaced") <= 1, \
        "seam_spaced reported twice"
    fired = {r["diag"] for r in diag_rows}
    never_fired = [d for d in DIAG_MEANING if d not in fired]
    # `seam` and `segment` are roles, not anomalies: assert they are not
    # reported as diagnostics and keep the ledger's totals reconcilable.
    diag_rows.append({"diag": "(role) seam", "n": role_n["seam"],
                      "pages": len({u.page for u in units if u.ops}),
                      "meaning": "operator bound two words into a unit (not an "
                                 "anomaly)",
                      "example": f"p{units[0].page} offset {units[0].s}"})
    diag_rows.append({"diag": "(role) segment",
                      "n": role_n["segment"], "pages": len({x["page"] for x in roles
                                                           if x["role"] == "segment"}),
                      "meaning": "#14 segment operator: ends a unit, never binds "
                                 "(not an anomaly)",
                      "example": f"p{op_examples['segment']['page']} offset "
                                 f"{op_examples['segment']['start']}: "
                                 f"{op_examples['segment']['op']}"})

    # ---- insight lines (computed, not asserted prose) --------------------
    hyp_types = {r["unit_type"] for r in ut_rows if r["class"].startswith("join")}
    seam_types = {r["unit_type"] for r in ut_rows if r["class"].split("+")[0]
                  not in ("plain", "join")}
    fused_types = len(hyp_types | seam_types)
    case_lower = {}
    for r in ut_rows:
        case_lower.setdefault(r["lower"], set()).add(r["unit_type"])
    case_var = {k: sorted(v) for k, v in case_lower.items() if len(v) > 1}
    n_case_var_units = sum(1 for u in units if u.text.lower() in case_var
                           and len(case_var[u.text.lower()]) > 1)
    # case-variant groups restricted to fused units: the ones where case folding
    # would merge a *seam* unit with another seam unit
    seam_var = {k: v for k, v in case_var.items()
                if any(any(t in s for s in seam_types) for t in v)}
    n_seam_var_units = sum(1 for u in units if u.text.lower() in seam_var)
    unit_ttr = len(ut_rows) / len(units)
    word_types = len({w for u in units for w, r in u.parts if r == "w"})
    n_seam_units = sum(1 for u in units if u.ops)
    n_fused = sum(1 for u in units if u.ops or u.n_join)
    n_multi = sum(1 for r in ut_rows if r["n_words"] > 1)
    n_multi_occ = sum(1 for u in units if u.n_parts > 1)
    seam_rate = n_seam_units / len(units)
    insight_lines = [
        f"{n_fused:,} unit occurrences ({n_fused/len(units)*100:.1f}%) are "
        f"**fused** - they carry a seam or a hyphen join - and a word-level "
        f"count cannot see them at all: `gene=TV` is one WORD token here, so "
        f"only the unit view knows it exists.",
        f"Unit-level TTR {unit_ttr:.4f} ({len(ut_rows):,} types) against "
        f"part-level TTR {word_types/len(units):.4f} ({word_types:,} types) over "
        f"the same {len(units):,} occurrences: cutting units apart *raises* the "
        f"type/token ratio, because the same part recurs in different seam "
        f"contexts. The seams carry lexical distinctions, not just punctuation.",
        f"{n_multi:,} unit types have more than one word part "
        f"({n_multi_occ:,} occurrences), and {len(seam_types):,} types contain at "
        f"least one seam - so any part-level statistic computed without the unit "
        f"layer is counting a *different* lexicon from the one a reader sees.",
        f"{n_case_var_units:,} occurrences belong to a lower-cased group with more "
        f"than one spelling ({len(case_var)} groups overall, "
        f"{len(seam_var)} of them seam-bearing / {n_seam_var_units:,} occurrences). "
        f"Case folding merges `Soul/gram` into `soul/gram`; whether that is right "
        f"is #19's call, but the tokenizer keeps both spellings apart so the "
        f"decision stays visible.",
        f"Only {seam_rate*100:.1f}% of unit occurrences contain a seam, so the "
        f"unit layer is mostly a *safety net*: it changes nothing for the plain "
        f"majority and rescues the fused minority. That asymmetry is why a "
        f"word-level count loses real information instead of merely adding noise.",
    ]

    # ---- artifact-format hazards, measured so they cannot grow silently ---
    quote_tokens = [t for t in flat if '"' in t.text]
    quote_pages = sorted({t.page for t in quote_tokens})
    n_quote_tokens = len(quote_tokens)
    tails = {p: page_tails[p] for p in pages}

    limit_lines = [
        "**The unit definition is fitted to this corpus.** There is no held-out "
        "text; the seam rules are #12's signatures measured on this book, not a "
        "general tokenizer.",
        "**A unit is a tokenization unit, not a word.** `soul/gram` is emitted as "
        "one unit *because* the book writes it that way; whether it is one word, a "
        "portmanteau, two words or a citation is #18/#19's question, not this "
        "issue's. The tokenizer only guarantees the unit is recoverable.",
        "**Hyphen joining is inherited, not re-decided.** #12's tokenizer keeps "
        "`boy-roid` as one WORD token; this tokenizer keeps that decision and "
        "additionally exposes the split as `join` parts, so a downstream "
        "decomposition (#18) can re-join or re-split without re-tokenizing.",
        "**Unbound seams are counted, not repaired.** "
        f"{seam_ops['/']:,} `/`, {seam_ops['=']:,} `=`, {seam_ops['++']:,} `++` "
        f"and {seam_ops['+']:,} `+` operator tokens bind two words into a unit "
        f"({n_seams:,} seams); the other {sum(unbound.values()):,} link-operator "
        f"tokens bind nothing and each class of them is a row in "
        f"`diagnostics.tsv`.",
        "**The `//`-boundary reading is inherited.** Units do not cross `::` or "
        "`//` because #14 placed them there; if #15's Rosetta comparison shows a "
        "different layering, unit boundaries move with it.",
        "**Numeric channel not interpreted.** Tokens are flagged `num` / "
        "`has_digit` but no value parsing is attempted - that is #21.",
        "**The seam-loss comparison is offset-exact, and it still flatters the "
        "baselines.** An item counts as kept only if the baseline emits a token "
        "covering exactly its span, which is the weakest defensible reading of "
        "\"kept\". It says nothing about downstream utility: `wordpunct` keeps "
        "every seam *string* as a token, and a downstream consumer could "
        "re-assemble units from parts plus seams - at the cost of the rule this "
        "tokenizer exists to state once.",
    ]

    artifact_rows = [
        ("tokenizer_spec.md", "THE DELIVERABLE: grammar, schema, evaluation"),
        ("tokens.tsv", "the lossless token stream, unit-annotated"),
        ("tokens.jsonl", "the same stream, one JSON object per token"),
        ("page_tails.tsv", "the page text after each page's last token"),
        ("units.tsv", "one row per lexical unit (the analyzable unit)"),
        ("unit_types.tsv", "unit type inventory (surface -> n, pages, class)"),
        ("part_index.tsv", "one row per unit part (w / seam / join) with offsets"),
        ("seam_loss.tsv", "standard-tokenizer comparison, overall + per class"),
        ("cases.tsv", "named constructs, page-cited, with all three views"),
        ("case_examples.txt", "readable rendering: context, unit, parts, baselines"),
        ("diagnostics.tsv", "tokenizer pressure ledger"),
        ("reconciliation.tsv", "this tokenizer vs #10 / #12 / #14 / first pass"),
        ("token_stats.tsv", "per-class and per-op token statistics"),
        ("summary.json", "machine-readable headline numbers"),
    ]

    # ---- write artifacts -------------------------------------------------
    with open(f"{out}/tokens.tsv", "w") as f:
        # Two columns carry JSON string literals rather than raw text:
        #   * `gap_json` - the source text between two tokens; it can contain a
        #     blank line or a tab, which would break a line-oriented reader.
        #   * `text_json` - the token text, for the same reason in the other
        #     direction: the corpus contains bare `"` tokens (136 of them, on
        #     pages 47, 92, 158, 209, 219, 224), so a *quote-aware* reader
        #     (`csv.DictReader(delimiter="\t")`, pandas, R) silently mis-sizes
        #     110 rows from p47 on.  Read this file with a plain `split("\t")`,
        #     or take `text_json` / `gap_json`, or use `tokens.jsonl`.
        f.write("tid\tpage\tregion\titem\thlevel\tkind\tstart\tend\ttext\t"
                "text_json\tgap_json\tglue_L\tglue_R\tupper\tnum\thas_digit\tcaps\t"
                "left_op\tpclass\tunit\tupos\tast_path\tcontainer\n")
        for t in flat:
            f.write(f"{t.i}\t{t.page}\t{t.region}\t{t.item}\t{t.hlevel}\t"
                    f"{t.kind}\t{t.s}\t"
                    f"{t.e}\t{_cell(t.text)}\t"
                    f"{json.dumps(t.text, ensure_ascii=False)}\t"
                    f"{json.dumps(t.gap, ensure_ascii=False)}\t"
                    f"{'y' if t.glue_L else ''}\t{'y' if t.glue_R else ''}\t"
                    f"{'y' if t.upper else ''}\t{'y' if t.num else ''}\t"
                    f"{'y' if t.has_digit else ''}\t{'y' if t.caps else ''}\t"
                    f"{_cell(t.left_op)}\t{t.pclass}\t"
                    f"{t.unit if t.unit is not None else ''}\t{t.pos}\t"
                    f"{_cell(t.ast_path)}\t{t.container}\n")
    with open(f"{out}/tokens.jsonl", "w") as f:
        for t in flat:
            f.write(json.dumps(t.obj(), separators=(",", ":"),
                               ensure_ascii=False) + "\n")
    # The text after a page's last token is not carried by any token's gap, so
    # `"".join(gap + text)` alone stops one token short of the page end.
    # Publishing the tail closes that: page body == tokens + tail, which is what
    # makes the losslessness claim checkable from the artifacts alone.
    with open(f"{out}/page_tails.tsv", "w") as f:
        f.write("page\tlast_token_end\ttail_json\tn_chars\n")
        for pno in pages:
            tail = page_tails[pno]
            f.write(f"{pno}\t{all_toks[pno][-1].e}\t"
                    f"{json.dumps(tail, ensure_ascii=False)}\t{len(tail)}\n")

    # The quote hazard is measured on the file as it actually lands on disk, not
    # on the in-memory token list, so a change to the writer cannot hide it.
    n_csv_bad = 0
    with open(f"{out}/tokens.tsv") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) != 23 and row and row[0] != "tid":
                n_csv_bad += 1

    downstream_lines = [
        "`units.tsv` (uid, page, offsets, text, sig, class, parts, ast path) is the "
        "join key for #18 (decomposition), #19 (neologisms), #33 (clauses) and "
        "#52 (operator-word MI): join on `uid`/`page`+`start`, never on the token "
        "index.",
        "`tokens.tsv` carries the lossless `gap_json` column; anything that "
        "rebuilds text should decode it rather than re-joining with spaces (the "
        "book has no space after `::`, `//`, `=` in most occurrences), and "
        "should append the page's `page_tails.tsv` row to reach the page end.",
        "**Read the `.tsv` artifacts with `line.split(\"\\t\")`.** The corpus "
        f"carries {n_quote_tokens} bare `\"` tokens, so a quote-aware reader "
        f"(`csv.DictReader(delimiter=\"\\t\")`) mis-sizes {n_csv_bad} rows of "
        f"`tokens.tsv` from page {quote_pages[0]} on. `tokens.jsonl` is the "
        "quote-safe equivalent of the whole stream.",
        "`units.tsv`'s `parts` column is bare part texts joined by `|`; the "
        "`text:role` notation used in this spec and in `cases.tsv` carries the "
        "role (`w` / `seam` / `join`). Both are safe to split because `:` and "
        "`|` are segment operators and can never sit inside a unit - asserted at "
        "build time. `part_index.tsv` carries every part's role **and its own "
        "source offsets** (`start`, `end`): join it on `uid`+`pos`, and a part "
        "can be quoted by page and offsets without any string search.",
        f"{len(seam_types):,} unit types are opaque to a "
        "word-frequency list - #19 should screen units, not words, or it will "
        "re-invent this tokenizer.",
        "Case is preserved on every token and unit (`upper`, `caps` flags); "
        "case-folded views must be derived, not assumed.",
        f"A unit's `text` is the exact source slice, spacing included: "
        f"{len(spaced)} units have a seam with whitespace on one side "
        f"(`gene = TV`, `eruption=\\njunk`), so concatenating `parts` reproduces "
        f"the surface only modulo whitespace. Quote `text`, decompose `parts`.",
        "The `html_tag_residue` row of `diagnostics.tsv` lists every unit that "
        "came from inside a real HTML tag; #19 must screen those before "
        "treating a 1-2 character unit as a neologism.",
        "The `//`/`::`-boundary invariant is asserted at build time, so if a future "
        "parse disagrees, the build fails rather than silently re-cutting units.",
    ]
    with open(f"{out}/units.tsv", "w") as f:
        f.write("uid\tpage\tregion\tstart\tend\tunit\tn_words\tsig\tclass\t"
                "parts\tin_span\tin_utterance\tin_field\tcontainer\tast_path\n")
        for u in units:
            f.write(f"{u.id}\t{u.page}\t{u.region}\t{u.s}\t{u.e}\t"
                    f"{_cell(u.text)}\t"
                    f"{u.n_parts}\t{u.sig}\t{u.cls}\t"
                    f"{'|'.join(t for t, r in u.parts)}\t{u.in_span}\t"
                    f"{u.in_utt}\t{u.in_field}\t{u.container}\t{_cell(u.ast_path)}\n")
    dump_tsv("unit_types.tsv", ut_rows,
             ["unit_type", "lower", "n", "n_pages", "class", "n_words", "parts",
              "first_page", "first_start", "ast", "page_list"], out)
    dump_tsv("part_index.tsv", part_rows,
             ["uid", "page", "unit", "start", "end", "pos", "role", "text"], out)
    dump_tsv("seam_loss.tsv", loss_rows,
             ["baseline", "n_tokens", "unit_kept", "n_units", "unit_kept_rate",
              "parts_kept", "n_parts", "parts_kept_rate", "seams_kept", "n_seams",
              "seams_kept_rate", "ops_kept", "n_ops", "ops_kept_rate"], out)
    dump_tsv("named_constructs.tsv", named_rows,
             ["needle", "prior", "prior_src", "prior_recomputed", "raw",
              "whole_unit", "inside_longer_unit", "not_a_unit",
              "host_units", "first_page", "first_start"], out)
    dump_tsv("cases.tsv", case_rows_,
             ["case", "kind", "needle", "page", "start", "end", "unit", "class",
              "sig", "n_words", "parts", "container", "ast", "left_ctx",
              "right_ctx", "note"], out)
    dump_tsv("diagnostics.tsv", diag_rows,
             ["diag", "n", "pages", "meaning", "example"], out)
    dump_tsv("reconciliation.tsv", recon, ["what", "earlier", "here"], out)

    with open(f"{out}/case_examples.txt", "w") as f:
        f.write("# Operator-aware tokenizer: named constructs (#17)\n")
        f.write("# each case: page-cited context, the unit, its parts, and how\n")
        f.write("# four standard tokenizers see the same span of text.\n")
        for c in case_rows_:
            pno = c["page"]
            b = bodies[pno]
            lo = max(0, c["start"] - 40)
            hi = min(len(b), c["end"] + 40)
            frag = b[lo:hi]
            f.write(f"\n{c['case']}  {c['needle']}   [{c['kind']}]\n")
            f.write(f"  page {pno} offsets {c['start']}-{c['end']}  container "
                    f"{c['container'] or '-'}  {c['ast'] or '-'}\n")
            f.write(f"  context : ...{frag}...\n")
            if c["kind"] == "unit":
                f.write(f"  unit    : {c['unit']}  (class {c['class']}, sig "
                        f"'{c['sig']}', {c['n_words']} word part(s))\n")
                f.write(f"  parts   : {c['parts']}\n")
            else:
                f.write(f"  token   : {c['needle']} - not a unit by "
                        f"construction\n")
            f.write("  --- how standard tokenizers see the same span ---\n")
            for bname in BASELINES:
                bt = baseline_tokens(bname, frag)
                f.write(f"  {bname:<9}: {' '.join(bt)}\n")
            f.write(f"  note    : {c['note']}\n")

    tok_stats = []
    for k, n in kinds.most_common():
        tok_stats.append({"group": "kind", "key": k, "n": n})
    for op, n in ops_all.most_common():
        tok_stats.append({"group": "op_all", "key": op, "n": n})
    for op, n in ops_body.most_common():
        tok_stats.append({"group": "op_body", "key": op, "n": n})
    for op, n in ops_head.most_common():
        tok_stats.append({"group": "op_header", "key": op, "n": n})
    for op, n in seam_ops.most_common():
        tok_stats.append({"group": "seam", "key": op, "n": n})
    for c, n in cls_n.most_common():
        tok_stats.append({"group": "unit_class", "key": c, "n": n})
    for pc, n in collections.Counter(
            t.pclass for t in flat if t.kind == "PUNCT").most_common():
        tok_stats.append({"group": "punct_class", "key": pc, "n": n})
    for n_parts_, n in sorted(chain_hist.items()):
        tok_stats.append({"group": "chain_words", "key": n_parts_, "n": n})
    for k, n in sorted(seams_per_unit.items()):
        tok_stats.append({"group": "seams_per_unit", "key": k, "n": n})
    dump_tsv("token_stats.tsv", tok_stats, ["group", "key", "n"], out)

    # ---- artifact-format hazards, measured so they cannot grow silently ---
    S = {
        "schema": SCHEMA,
        "src": src,
        "n_pages": len(pages),
        "lossless": {"ok": len(pages) - len(lossless_bad), "total": len(pages),
                   "bad": lossless_bad},
        "tail_chars": sum(len(v) for v in tails.values()),
        "tail_nonblank_pages": sorted(p for p, v in tails.items() if v.strip()),
        "tail_comment_pages": sorted(p for p, v in tails.items()
                                     if "<!--" in v),
        "n_quote_tokens": n_quote_tokens,
        "quote_token_pages": quote_pages,
        "n_csv_bad_rows": n_csv_bad,
        "pages": pages,
        "free_dash": free_dash,
        "tokens": {"total": len(flat), "types": len({t.text for t in flat}),
                   "WORD": kinds["WORD"], "OP": kinds["OP"],
                   "PUNCT": kinds["PUNCT"],
                   "body": len(flat) and sum(1 for t in flat
                                             if t.region == "para"),
                   "header": sum(1 for t in flat if t.region == "header")},
        "units": {
            "total": len(units),
            "types": len(ut_rows),
            "parts": sum(u.n_parts for u in units),
            "part_types": word_types,
            "seams": n_seams,
            "seam_types": len(seam_types),
            "hyphens": sum(u.n_hyp for u in units),
            "hyphenated_word_tokens": sum(
                1 for u in units for t in u.toks
                if t.kind == "WORD" and "-" in t.text),
            "hyphen_types": len({t.text for u in units for t in u.toks
                                 if t.kind == "WORD" and "-" in t.text}),
            "glued_dash_in_units": sum(
                1 for u in units for t in u.toks
                if t.kind == "PUNCT" and t.pclass == "dash"),
            "n12_regex_hyphen": n12["hyphen"]["n"],
            "multi_word_units": sum(1 for r in ut_rows if r["n_words"] > 1),
            "fused_types": fused_types,
            "fused_occurrences": sum(1 for u in units if u.ops or u.n_join),
            "joins": sum(u.n_join for u in units),
            "seam_occurrences": n_seam_units,
            "multi_word_occurrences": sum(1 for u in units if u.n_parts > 1),
            "chain_hist": {str(k): v for k, v in sorted(chain_hist.items())},
            "seams_per_unit_hist": {str(k): v for k, v in
                                    sorted(seams_per_unit.items())},
            "median_words": statistics.median(u.n_parts for u in units),
            "mean_words": statistics.mean(u.n_parts for u in units),
            "ttr": unit_ttr,
            "part_ttr": word_types / len(units),
            "case_variant_groups": len(case_var),
            "case_variant_occurrences": n_case_var_units,
            "case_variant_examples": {k: v for k, v in
                                      list(case_var.items())[:10]},
            "seam_case_variant_groups": len(seam_var),
            "seam_case_variant_occurrences": n_seam_var_units,
        },
        "n11_stream": {"total": n11_total, "pages": len(pages),
                       "pages_differ": n11_pages_diff, "identical": not n11_pages_diff},
        "seams": {op: n for op, n in seam_ops.most_common()},
        "unbound_ops": dict(unbound),
        "ops_body": dict(ops_body),
        "ops_header": dict(ops_head),
        "ops_all": dict(ops_all),
        "class_rows": class_rows,
        "seam_loss": {"baselines": {L["baseline"]: L for L in loss_rows},
                      "n_baselines": len(loss_rows),
                      "baselines_real_lib": baseline_uses_real_lib()},
        "class_seam_rows": class_seam_rows,
        "recon_rows": recon,
        "case_rows_head": _one_per_case(case_rows_),
        "case_rows_all": case_rows_,
        "diag_rows": diag_rows,
        "never_fired": never_fired,
        "op_roles": dict(role_n),
        "op_roles_by_op": {f"{r}|{o}": n for (r, o), n in sorted(role_op.items())},
        "n_segment_ops": n_segment_ops,
        "insight_lines": insight_lines,
        "limit_lines": limit_lines,
        "downstream_lines": downstream_lines,
        "artifact_rows": artifact_rows,
        "named": {r["needle"]: r["whole_unit"] for r in named_rows},
        "named_rows": named_rows,
        "n12_pairs": {k: {"n": v["n"], "unique": v["unique"]}
                      for k, v in n12.items()},
    }
    with open(f"{out}/summary.json", "w") as f:
        json.dump(S, f, indent=1, ensure_ascii=False)
    write_spec(S, out)
    return S


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    src = argv[0] if argv else SRC
    out = argv[1] if len(argv) > 1 else OUT
    S = build(src, out)
    print(f"pages={S['n_pages']} lossless={S['lossless']['ok']}/{S['n_pages']} "
          f"tokens={S['tokens']['total']} words={S['tokens']['WORD']} "
          f"ops={S['tokens']['OP']}")
    print(f"units={S['units']['total']} types={S['units']['types']} "
          f"parts={S['units']['parts']} seams={S['units']['seams']} "
          f"hyphens={S['units']['hyphens']}")
    print("named units: "
          + " ".join(f"{r['needle']}={r['whole_unit']}/{r['raw']}raw"
                     for r in S["named_rows"]))
    for L in S["seam_loss"]["baselines"].values():
        print(f"  {L['baseline']:<10} tokens={L['n_tokens']:>7} "
              f"unit_kept={L['unit_kept_rate']*100:5.1f}% "
              f"parts_kept={L['parts_kept_rate']*100:5.1f}% "
              f"seams_kept={L['seams_kept_rate']*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())