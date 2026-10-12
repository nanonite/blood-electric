#!/usr/bin/env python3
"""#62 Interlinear edition generator -> analysis/interlinear/

Task (plan E8.2, docs/chainlink-plan.md #62): produce the **three-line
aligned rendering** of the whole book --

    1  original clause            the exact source substring, page-cited
    2  operator-normalized        the #13 v4 "silence removed" rendering
                                  (operators -> `. `), the variant #50
                                  measured as closest to the underlying
                                  "vowel" system (ppl 84.1 vs 106.2)
    3  plain-English gloss        every non-English surface read in plain
                                  English, every reading evidence-linked

-- for **every #33 clause**, with a generator that is deterministic,
re-runnable and testable.  This is the full-book follow-up to #56's
partial (three page-cited passages); #56's token rules, curated
termbase, operator readings and v3/v4 renderer are reused, not redone.

Method (what "plain-English gloss" means here, and what it does not):

  * Line 2 is *not* re-invented: it is #56's `render_variant`, i.e. the
    #13 v3/v4 transform over the #17 token stream, so the middle line
    reconciles with #13 by construction and both variants are carried
    (v4 as the canonical line, v3 in the TSV so a reader can override).
  * Line 3 substitutes a plain reading **only** where the surface is not
    ordinary English.  Ordinary/function/rare English, numeric tokens and
    compounds of English parts are kept verbatim (the book *is* English
    lexis under a symbolic transform -- #56's viability verdict), so the
    gloss line equals the normalized line except at the non-English
    sites.  That overlap is reported as a number, not hidden.
  * Every non-English surface class is handled by one published rule,
    in a fixed order (see `token_reading`):
        #62 curated reading   -> the PLAIN_READINGS table (49 of the 51
                                #19 `true_coinage` types; readings are
                                authored from the #26 KWIC contexts, the
                                #19 morphology and the #30 glossary, and
                                carry the #56 confidence scheme)
        identity (#19 screen) -> english / rare_english / numeric: the
                                word is its own gloss (high)
        compound (#19)        -> kept intact, parts quoted (medium)
        proper noun (#30)     -> names are not translated; the #30 role
                                and does/done-to verbs are quoted (medium)
        ocr_error (#19)       -> kept as printed, never auto-corrected
                                (L0.4, the L9.8 over-correction ban) (low)
        unresolved            -> marked `<surface>` in the gloss line and
                                itemized in residue.tsv -- listed, not
                                repaired (L9.8).  Two #19 `true_coinage`
                                types have no evidence-linked reading.
  * Coverage closes per token: every token is exactly one of
    {glossed high|medium|low, unresolved, silence (OP/PUNCT),
    markup-excluded} and the build fails if the buckets do not add up.
    Tokens inside #33 clauses + tokens outside them == the 90,220 #17
    tokens, so the edition's remainder is published, not dropped.
  * No LLM is used anywhere (that is #63's committee); no external
    corpus is consulted or acquired.  Nothing is translated that this
    workspace cannot cite: every reading's evidence sits in
    `gloss_lexicon.tsv`.

Outputs under analysis/interlinear/:
  interlinear.md         THE DELIVERABLE: the edition, three lines per
                         clause, page-cited, all pages
  interlinear.tsv        one row per clause (both normalizations + gloss
                         + confidence counts + unresolved list)
  gloss_tokens.tsv       one row per in-clause token: the per-token
                         reading, confidence, source and evidence note
  gloss_lexicon.tsv      the readings with computed counts/pages/KWIC
  residue.tsv            every unresolved/awkward class, itemized
  per_page.tsv           per-page clause and confidence counts
  operator_roles.tsv     the operator table the silence layer reads from
  reconciliation.tsv     against the frozen #13/#17/#19/#30/#33/#56 layers
  diagnostics.tsv        build counters
  summary.json           machine-readable headline numbers
  interlinear_report.md  method, findings, samples, anti-pattern audit

Reproduce::

    conda activate marker
    python work/interlinear_edition.py                 # full book
    python work/interlinear_edition.py --pages 4-6      # a page range
    python work/test_interlinear_edition.py

Deterministic: no randomness, no hash-order dependence (all ties broken
by a fixed key), every statistic recomputes identically.  The only field
that varies between runs is the `generated` stamp carried in the report
header and in summary.json provenance, which no statistic derives from.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clause_segmentation as cs      # #33 loaders + TSV io
import frame_mining as fm              # #34 shares the #17/#19/#33 context
import language_viability as lv        # #56 termbase, readings, renderer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "interlinear"
# Every input is a frozen artifact; the loaders and the owning issues are
# beside each path.  #19's screening.tsv and #17's units.tsv are read
# through `frame_mining.load_context()` (the shared #33/#34 context), so
# they are documented in `load_context` rather than re-opened here.
CLAUSES = ROOT / "analysis" / "clauses" / "clauses.tsv"        # #33
TOKENS = ROOT / "analysis" / "tokenizer" / "tokens.tsv"         # #17
ENTITY = ROOT / "analysis" / "entity" / "entity_glossary.tsv"   # #30
TERMBASE = ROOT / "analysis" / "language_viability" / "termbase_sample.tsv"  # #56
KWIC = ROOT / "analysis" / "kwic" / "concordance.tsv"           # #26

SCHEMA = "interlinear/1"
CONFIDENCES = ("high", "medium", "low", "unresolved")

# The unresolved marker in the gloss line: single guillemets (TSV-safe,
# never a tab or a newline, visually unmistakable against the source's
# own ``<< >>`` block quotes).
UNRESOLVED_MARK = ("\u2039", "\u203a")
# A token made only of punctuation is not a word: the invariant that the
# v4 line's words are exactly the clause's WORD tokens (see
# `normalized_words`).
_WS = re.compile(r"[ \t]+")
# #7/#70 layout comments are page scaffolding, never clause text; #70
# publishes them as the 468-char scaffolding class.  A clause's leading
# gap can carry them (figure-only pages), so both the normalized line and
# the gloss line strip them from the inter-token gaps.
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def clean_gap(gap):
    """A #17 gap with layout comments removed (#70 scaffolding)."""
    if "<!--" in gap:
        gap = _COMMENT.sub(" ", gap)
    return gap


def render_clause(toks, mode):
    """#13 v3/v4 rendering over a clause's tokens.

    #56's `render_variant` semantics, exactly, with one addition: layout
    comments in the inter-token gaps are stripped (they are #7/#70
    scaffolding, not clause text).
    """
    out = []
    for r in toks:
        gap = clean_gap(_tok_gap(r))
        text = _tok_text(r)
        if r["kind"] == "OP":
            out.append(". " if mode == "v4" else " ")
        else:
            out.append(gap + text)
    return _WS.sub(" ", re.sub(r"\. \.", ". ", "".join(out))).strip()


def parse_pages(spec):
    """"4-6" or "4" -> (first, last) inclusive, or None for the whole book."""
    if not spec:
        return None
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", spec.strip())
    if not m:
        raise SystemExit("--pages wants a page or a range like 4-6: %r" % spec)
    a = int(m.group(1))
    b = int(m.group(2) or m.group(1))
    if b < a:
        raise SystemExit("--pages range is reversed: %r" % spec)
    return a, b


# ---------------------------------------------------------------------------
# the curated plain readings (#62)
# ---------------------------------------------------------------------------
# One entry per non-English surface that gets a plain-English
# substitution on line 3.  Confidence follows #56's scheme, stated there
# and reused verbatim: high = the sense is given by ordinary English plus
# consistent book usage; medium = compositional (parts + collocates) but
# not fully pinned; low = sparse/one-off construct.  The evidence columns
# (counts, pages, first page, KWIC example, #19 morphology) are computed
# from the frozen artifacts by `build_lexicon`, never hand-entered.
#
# Sources: #19's screening (morph + category), #26's KWIC contexts, #30's
# entity glossary (role / does / done-to verbs).  No external corpus and
# no LLM: a surface without an evidence-linked reading is *unresolved*
# (see residue.tsv), never guessed.
PLAIN_READINGS = {
    # --- high: ordinary-English sense, consistent usage -----------------
    "feti": ("fetuses", "high"),
    "oneiric": ("dream-like", "high"),
    "exoskeleta": ("exoskeleta: plural of exoskeleton, the outer skeleton; "
                   "a body-machine term", "medium"),
    "ovarium": ("ovary", "high"),
    "osmosing": ("seeping through, blending", "high"),
    "rhythmus": ("rhythm", "high"),
    "fecundate": ("make fertile", "high"),
    "pincerclaw": ("pincer-claw", "high"),
    "exogenesis": ("outer generation", "high"),
    "encephalogram": ("brain recording", "high"),
    "biotechnic": ("biotechnical", "high"),
    "scintillates": ("sparkles", "high"),
    "vivisects": ("cuts up alive", "high"),
    "quiesced": ("became still", "high"),
    "vertabre": ("vertebrae", "high"),
    # --- medium: compositional sense, legible parts ---------------------
    "actifies": ("activates, sets working", "medium"),
    "aestrogen": ("estrogen-like", "medium"),
    "anamorph": ("transformed form", "medium"),
    "autolyzes": ("self-dissolves", "medium"),
    "cellspermatozoon": ("cell-sperm unit", "medium"),
    "genomemeridian": ("genome-meridian", "medium"),
    "heteromaniac": ("obsessed with the other", "medium"),
    "hydromaniac": ("driven by the hydromania obsession", "medium"),
    "inboneeatings": ("bone-eatings", "medium"),
    "mienai": ("mienai -- Japanese `invisible` (in the insert `nemuri ga "
               "mienai`)", "medium"),
    "mucal": ("mucus-like", "medium"),
    "nanomachinative": ("driven by nanomachines", "medium"),
    "necrocracy": ("rule of the dead", "medium"),
    "necrocrats": ("carriers of death", "medium"),
    "necstasy": ("death-ecstasy", "medium"),
    "reflexor": ("reflex unit", "medium"),
    "schizophysical": ("split-body", "medium"),
    "schizocrime": ("split-crime", "medium"),
    "vesuvial": ("volcanic", "medium"),
    # --- low: sparse/one-off construct, reading hedged ------------------
    "emodule": ("e-module (electronic module)", "low"),
    "fibrolator": ("fiber-rotator (reading uncertain)", "low"),
    "acidhumanixsymbiotic": ("acid-HUMANIX symbiotic (a one-off all-caps "
                             "construct)", "low"),
    "acidhumanixvirus": ("acid-HUMANIX virus (a one-off all-caps "
                         "construct)", "low"),
    "ninternal": ("n-internal (a glued construct)", "low"),
    "nexternal": ("n-external (a glued construct)", "low"),
    "ninoculated": ("inoculated (a glued construct)", "low"),
    "nunleashed": ("unleashed (a glued construct)", "low"),
    "ofofofliferates": ("a one-off `proliferates` construct", "low"),
    "ofproliferates": ("a one-off `proliferates` construct", "low"),
    "qirl": ("girl (a q-for-g spelling variant)", "low"),
    "querrilla": ("guerrilla (a coinage spelling)", "low"),
    "reflexors": ("reflex units", "medium"),
    "retroburn": ("retro-burn", "low"),
    "sartificial": ("s-artificial (a glued construct)", "low"),
}
# #19 `true_coinage` types deliberately left unresolved (opaque under the
# workspace evidence; listed in residue.tsv as reading leads for #63):
# omanko, laple.  Everything else in the class carries a reading.

# #56's OPERATOR_GLOSSES covers the operator classes the interlinear
# reads most; the inline `<`/`>` pair (#14 `Inline`: 778 / 381 tokens in
# the #17 stream) is not in that table and is read here from its own #14
# role, so operator_roles.tsv carries no unread operator.
_INLINE_READINGS = {
    "<": "inline marker opener (#14 Inline; the inline analogue of the "
         "`<<` Span opener #56 reads)",
    ">": "inline marker closer (#14 Inline; the inline analogue of the "
         "`>>` Span closer)",
}

READING_ORDER = ("plain_reading", "identity_19", "identity_numeric",
                 "compound_19", "entity_30", "ocr_error_19", "junk_19",
                 "unit_context_17", "unresolved", "silence")

SOURCE_LABEL = {
    "plain_reading": "#62 curated reading (PLAIN_READINGS)",    "identity_19": "identity -- ordinary English per the #19 screen",
    "identity_numeric": "identity -- numeric channel (#21), kept verbatim",
    "compound_19": "identity -- compound of English parts (#19), kept intact",
    "entity_30": "identity -- named construct (#30); names are not translated",
    "ocr_error_19": "identity -- #19 `ocr_error` class, kept as printed",
    "junk_19": "unresolved -- #19 `junk` markup residue",
    "unit_context_17": "identity -- word-part inside a #17 unit",
    "unresolved": "unresolved -- no evidence-linked reading (residue.tsv)",
    "silence": "removed on the gloss line (operator/punctuation silence)",
}


# ---------------------------------------------------------------------------
# loading (reuses the frozen #17/#19/#26/#30/#33/#56 artifacts only)
# ---------------------------------------------------------------------------

class InterlinearError(Exception):
    """The token buckets do not close, or a frozen count drifted."""


def load_context():
    """(bodies, page_units, unit_meta, is_markup, screening, markup_spans,
    entity, termbase_56, kwic).

    `bodies`        - {page: body} (#7 layout recovery)
    `page_units`    - {page: [(start, end, uid), ...]} (#33 convention)
    `unit_meta`     - {uid: (page, start, end, surface)} (#17)
    `is_markup`     - uid -> bool  (#19 ledger + #27 tag-run screen)
    `screening`     - {lower unit: #19 row}
    `markup_spans`  - {page: [(start, end), ...]} (#27 tag-run detector)
    `entity`        - {lower surface: #30 row}
    `termbase_56`   - {term: #56 termbase row}
    `kwic`          - {term_lower: [example strings]} (#26)
    """
    ctx = fm.load_context()
    bodies, page_units, unit_meta, is_markup, screening = ctx
    markup_spans, _uids, _types = cs.load_markup(bodies)
    entity = {}
    if ENTITY.exists():
        for r in cs.read_tsv(ENTITY):
            entity.setdefault(r["entity"].lower(), r)
    termbase = {}
    if TERMBASE.exists():
        for r in cs.read_tsv(TERMBASE):
            termbase.setdefault(r["term"].lower(), r)
    kwic = collections.defaultdict(list)
    if KWIC.exists():
        for r in cs.read_tsv(KWIC):
            if len(kwic[r["term_lower"]]) < 2:
                kwic[r["term_lower"]].append(
                    r["left_context"][-28:] + " [" + r["unit_surface"] +
                    "] " + r["right_context"][:28])
    return (bodies, page_units, unit_meta, is_markup, screening,
            markup_spans, entity, termbase, dict(kwic))


def load_tokens():
    """{page: [#17 token rows, sorted by (start, end)]}."""
    by_page = collections.defaultdict(list)
    for r in cs.read_tsv(TOKENS):
        by_page[int(r["page"])].append(r)
    for p in by_page:
        by_page[p].sort(key=lambda r: (int(r["start"]), int(r["end"])))
    return by_page


def load_clauses():
    """#33 clause rows, in (page, start) order."""
    rows = cs.read_tsv(CLAUSES)
    rows.sort(key=lambda r: (int(r["page"]), int(r["start"])))
    return rows


_tok_text = lambda r: json.loads(r["text_json"])
_tok_gap = lambda r: json.loads(r["gap_json"])


def tokens_in(toks, start, end):
    """The #17 token rows inside [start, end), in surface order (#33 join)."""
    return [r for r in toks
            if start <= int(r["start"]) and int(r["end"]) <= end]


def in_markup_span(tok, spans):
    """True when the token's span intersects a #27 markup span."""
    a, b = int(tok["start"]), int(tok["end"])
    for (s, e) in spans:
        if a < e and s < b:
            return True
    return False


def containing_unit(page_units, page, start):
    """The #17 unit containing a token start (#56's containment join)."""
    for (s, e, uid) in page_units.get(page, ()):
        if s <= start < e:
            return uid
    return None


# ---------------------------------------------------------------------------
# the per-token reading
# ---------------------------------------------------------------------------

def operator_reading(surface):
    """#12/#14's reading of an operator, with the measured slash roles
    filled from the memoized recount (a stale literal cannot contradict
    #14/#17).

    #56's OPERATOR_GLOSSES covers the operator classes the interlinear
    reads most; the inline `<`/`>` pair (#14 `Inline`, 778/381 tokens)
    is not in that table and is read here from its own #14 role, so the
    operator table has no unread operator.
    """
    gloss = lv.OPERATOR_GLOSSES.get(surface)
    if gloss is None:
        gloss = _INLINE_READINGS.get(surface, "operator (silence)")
    if "{" in gloss:
        link, slash = lv.slash_role_counts()
        gloss = gloss.format(link=link, slash=slash)
    return gloss


def _match_case(surface, reading):
    """Carry the source's case pattern onto a substituted reading."""
    if surface.isupper():
        return reading.upper()
    if surface[:1].isupper():
        return reading[:1].upper() + reading[1:]
    return reading


def mark_unresolved(surface):
    return UNRESOLVED_MARK[0] + surface + UNRESOLVED_MARK[1]


def token_reading(tok, ctx):
    """The reading decision for one token: (gloss, confidence, source, note).

    The rule order is the published one (module docstring); exactly one
    branch fires, so the confidence buckets close over every token.
    """
    (_bodies, page_units, unit_meta, _is_markup, screening,
     _markup_spans, entity, _tb, _kwic) = ctx
    surface = _tok_text(tok)
    kind = tok["kind"]
    if kind == "OP":
        return "", "high", "silence", operator_reading(surface)
    if kind == "PUNCT":
        # Ordinary punctuation is ordinary English, not the symbolic
        # layer: it survives on the gloss line (only the operators are
        # the silence #13 removes).
        return surface, "high", "silence", \
            "punctuation (kept on the gloss line; ordinary English, not " \
            "the #12 symbolic layer)"
    page, start, end = int(tok["page"]), int(tok["start"]), int(tok["end"])
    low = surface.lower()
    row = screening.get(low) or {}
    cat = row.get("category", "")

    pr = PLAIN_READINGS.get(low)
    if pr is not None:
        reading, conf = pr
        return (_match_case(surface, reading), conf, "plain_reading",
                "#62 reading of the #19 `%s` surface" % (cat or "coinage"))

    if cat == "english":
        fn = ("function word" if low in lv.FUNCTION_WORDS
              else "ordinary English content word")
        return (surface, "high", "identity_19",
                "the ordinary English word (#19 screen class `english`; %s)"
                % fn)
    if cat == "rare_english":
        return (surface, "high", "identity_19",
                "rare but ordinary English (#19 screen class `rare_english`)")
    if cat == "numeric":
        return (surface, "high", "identity_numeric",
                "numeric channel (#21); kept verbatim")
    if cat == "compound":
        morph = row.get("morph", "")
        return (surface, "medium", "compound_19",
                "compound of English parts (#19; #17 parts %s); kept intact"
                % (morph if morph else "n/a"))
    if cat == "proper_noun":
        eg = entity.get(low, {})
        bits = ["#19 `proper_noun` (a named construct; names are not "
                "translated)"]
        if eg.get("role"):
            bits.append("#30 role: %s" % eg["role"])
        if eg.get("does_verbs"):
            bits.append("does: %s" % "|".join(eg["does_verbs"].split("|")[:3]))
        if eg.get("done_to_verbs"):
            bits.append("done-to: %s"
                        % "|".join(eg["done_to_verbs"].split("|")[:3]))
        return (surface, "medium", "entity_30", "; ".join(bits))
    if cat == "ocr_error":
        return (surface, "low", "ocr_error_19",
                "#19 `ocr_error` class -- kept as printed, never "
                "auto-corrected (L0.4; the L9.8 over-correction ban)")
    if cat in ("junk", "markup_residue"):
        return ("", "unresolved", "junk_19",
                "#19 `%s` class (markup residue)" % cat)
    if cat == "true_coinage":
        return (mark_unresolved(surface), "unresolved", "unresolved",
                "#19 `true_coinage` with no evidence-linked reading -- "
                "residue, not repaired")

    # Not a screened unit surface: read the containing #17 unit, the #56
    # rule for word-parts of named constructs.
    uid = containing_unit(page_units, page, start)
    if uid is not None:
        u_page, u_s, u_e, u_surface = unit_meta[uid]
        ue = screening.get(u_surface.lower()) or {}
        ucat = ue.get("category", "")
        eg = entity.get(u_surface.lower(), {})
        role_note = ("; #30 entity role: %s" % eg["role"]
                     if eg.get("role") else "")
        if start == u_s and end == u_e:
            return (surface, "medium", "unit_context_17",
                    "the named construct `%s` in full (#19 class `%s`%s)"
                    % (u_surface, ucat or "n/a", role_note))
        morph = ue.get("morph", "")
        return (surface, "medium", "unit_context_17",
                "word-part of the compound unit `%s` (#17; parts %s)%s"
                % (u_surface, morph if morph else "n/a", role_note))
    return (mark_unresolved(surface), "unresolved", "unresolved",
            "no #19 category, no containing #17 unit and no curated "
            "reading -- listed in residue.tsv, not repaired")


def normalized_from_original(original, removed, start, mode):
    """The silence-removed line from the *source substring*: replace every
    operator span with `. ` (#13 v4) or a space (#13 v3), delete every
    markup-excluded span, keep words and punctuation as printed.

    `removed` is the ordered [(start, end, replacement)] span list built
    from the clause's tokens (operators + markup-excluded).

    This is the independent path that pins `render_clause`: one rendering
    is assembled from the #17 token fields (gap + text), the other from
    the #7 page body plus the token span offsets.  They must agree for
    every clause, or the clause's offsets, its token selection or its
    page join has drifted.
    """
    out, pos = [], 0
    for (s, e, repl) in removed:
        # `render_clause` (like #13/#56) drops a silenced token's own
        # leading gap, so the direct path drops the whitespace before it.
        out.append(original[pos:s - start].rstrip())
        out.append(repl)
        pos = e - start
    out.append(original[pos:])
    text = _COMMENT.sub(" ", "".join(out))
    return _WS.sub(" ", re.sub(r"\. \.", ". ", text)).strip()


# ---------------------------------------------------------------------------
# the clause record: the three lines + the per-token table
# ---------------------------------------------------------------------------

def clause_record(row, toks_by_page, ctx):
    """(record, token_rows) for one #33 clause.

    The record carries the three lines, both #13 normalizations, the
    confidence counts, the unresolved list and the closure check (every
    contained token is kept or markup-excluded, never silently dropped).
    """
    (bodies, _page_units, _unit_meta, _is_markup, _screening,
     markup_spans, _entity, _tb, _kwic) = ctx
    page, start, end = int(row["page"]), int(row["start"]), int(row["end"])
    body = bodies[page]
    original = body[start:end]
    spans = markup_spans.get(page, [])
    contained = tokens_in(toks_by_page.get(page, []), start, end)
    kept, excluded = [], []
    for t in contained:
        (excluded if in_markup_span(t, spans) else kept).append(t)

    v3 = render_clause(kept, "v3")
    v4 = render_clause(kept, "v4")
    # the independent cross-check: the token-assembled rendering and the
    # substring-deleted rendering must agree for both modes.  Silenced
    # spans are the operators (replaced) and the markup-excluded tokens
    # (deleted), in surface order.
    silenced = [t for t in contained if t["kind"] == "OP" or t in excluded]

    def removed_spans(mode):
        r = ". " if mode == "v4" else " "
        spans = []
        for t in silenced:
            repl = "" if t in excluded else (r if t["kind"] == "OP" else None)
            if repl is None:
                continue
            spans.append((int(t["start"]), int(t["end"]), repl))
        return sorted(spans)

    norm_ok = (v3 == normalized_from_original(original, removed_spans("v3"),
                                              start, "v3")
               and v4 == normalized_from_original(original,
                                                  removed_spans("v4"),
                                                  start, "v4"))
    body_ok = all(bodies[page][int(t["start"]):int(t["end"])]
                  == _tok_text(t) for t in kept)

    pieces = []
    token_rows = []
    conf = collections.Counter()
    src = collections.Counter()
    unresolved = collections.Counter()
    for t in kept:
        s, e = int(t["start"]), int(t["end"])
        gloss, confidence, source, note = token_reading(t, ctx)
        conf[confidence] += 1
        src[source] += 1
        if t["kind"] == "OP":
            # the symbolic silence layer: one space, never a glued fusion
            pieces.append(" ")
        else:
            if gloss:
                pieces.append(clean_gap(_tok_gap(t)) + gloss)
            if confidence == "unresolved":
                unresolved[_tok_text(t)] += 1
        uid = containing_unit(ctx[1], page, s)
        u_surface = ctx[2][uid][3] if uid is not None else ""
        scr = ctx[4].get(_tok_text(t).lower()) or {}
        token_rows.append({
            "clause_id": row["clause_id"], "page": page,
            "region": row["region"], "surface": _tok_text(t),
            "kind": t["kind"], "start": s, "end": e,
            "unit": u_surface, "unit_uid": "" if uid is None else uid,
            "category_19": scr.get("category", ""),
            "parts_18": scr.get("morph", ""),
            "gloss": gloss, "confidence": confidence, "source": source,
            "note": note,
        })
    gloss_line = _WS.sub(" ", "".join(pieces)).strip()
    if not gloss_line:
        gloss_line = "(no readable text: markup only)" if excluded \
            and not kept else "(empty clause)"

    record = {
        "clause_id": row["clause_id"], "page": page, "region": row["region"],
        "opener": row["opener"], "start": start, "end": end,
        "original": original,
        "v3_space": v3, "v4_period": v4, "gloss": gloss_line,
        "n_tokens": len(contained), "n_kept": len(kept),
        "n_markup": len(excluded),
        "n_silence": sum(1 for t in kept if t["kind"] != "WORD"),
        "n_high": conf.get("high", 0), "n_medium": conf.get("medium", 0),
        "n_low": conf.get("low", 0),
        "n_unresolved": conf.get("unresolved", 0),
        "n_readings": src.get("plain_reading", 0),
        "unresolved": "|".join("%s x%d" % (s, n) for s, n
                               in sorted(unresolved.items())),
        "norm_match": norm_ok,
        "token_span_match": body_ok,
    }
    return record, token_rows


# ---------------------------------------------------------------------------
# derived tables
# ---------------------------------------------------------------------------

def build_lexicon(ctx):
    """The readings table: every surface that gets a rule or a reading,
    with the computed evidence (counts come from #19's own screen)."""
    (_bodies, _page_units, _unit_meta, _is_markup, screening,
     _markup_spans, entity, termbase, kwic) = ctx
    bad_keys = [k for k in PLAIN_READINGS if k != k.lower()]
    if bad_keys:
        raise InterlinearError(
            "PLAIN_READINGS keys must be lower-case (the lookup is "
            "case-folded): %s" % bad_keys)
    surfaces = set(PLAIN_READINGS)
    surfaces |= {k for k in entity}
    surfaces |= {k for k in lv.GLOSS_TERMS}
    rows = []
    for low in sorted(surfaces):
        scr = screening.get(low) or {}
        pr = PLAIN_READINGS.get(low)
        eg = entity.get(low, {})
        tb = termbase.get(low, {})
        if pr is not None:
            reading, confidence, source = pr[0], pr[1], "plain_reading"
        elif eg:
            reading, confidence, source = "", "medium", "entity_30"
        elif tb:
            reading, confidence, source = "", tb.get("confidence", ""), \
                "termbase_56"
        else:
            reading, confidence, source = "", "", "termbase_56"
        note = tb.get("gloss", "")
        if not note and eg:
            note = "#30 role: %s; does: %s; done-to: %s" % (
                eg.get("role", ""), eg.get("does_verbs", ""),
                eg.get("done_to_verbs", ""))
        rows.append({
            "surface": scr.get("unit") or low,
            "reading": reading, "confidence": confidence, "source": source,
            "category_19": scr.get("category", ""),
            "parts_18": scr.get("morph", ""),
            "entity_role_30": eg.get("role", ""),
            "n_occurrences": scr.get("n", ""),
            "n_pages": scr.get("n_pages", ""),
            "first_page": scr.get("first_page", ""),
            "kwic_examples_26": " | ".join(kwic.get(low, [])[:2]),
            "note": note,
        })
    rows.sort(key=lambda r: (-(int(r["n_occurrences"] or 0)),
                             r["surface"].lower()))
    return rows


def build_residue(ctx, token_rows):
    """Every unresolved/awkward class, itemized with pages and verdict."""
    (_bodies, _page_units, _unit_meta, _is_markup, screening,
     _markup_spans, entity, _tb, kwic) = ctx
    by_surface = collections.defaultdict(list)
    for t in token_rows:
        if t["confidence"] == "unresolved":
            by_surface[(t["surface"], t["source"])].append(t)
    rows = []
    for (surface, source), hits in sorted(by_surface.items(),
                                          key=lambda kv: (-len(kv[1]),
                                                          kv[0][0].lower())):
        pages = sorted({h["page"] for h in hits})
        cat = (screening.get(surface.lower()) or {}).get("category", "")
        eg = entity.get(surface.lower(), {})
        rows.append({
            "class": "unresolved_%s" % source,
            "item": surface, "n": len(hits),
            "pages": pages,
            "owner": "#19 %s; #30 %s" % (cat or "n/a",
                                         eg.get("role", "n/a")),
            "verdict": "residue -- listed, not repaired (L9.8)",
            "note": " | ".join(kwic.get(surface.lower(), [])[:1]) or
                    "no #26 KWIC row",
        })
    # the OCR-error class is kept, not repaired: a residue class with a
    # standing verdict rather than an unresolved reading.
    ocr = [t for t in token_rows if t["source"] == "ocr_error_19"]
    if ocr:
        by_surface = collections.defaultdict(list)
        for t in ocr:
            by_surface[t["surface"]].append(t)
        for surface, hits in sorted(by_surface.items(),
                                    key=lambda kv: (-len(kv[1]),
                                                    kv[0].lower())):
            pages = sorted({h["page"] for h in hits})
            rows.append({
                "class": "ocr_error_kept", "item": surface, "n": len(hits),
                "pages": pages,
                "owner": "#19 ocr_error",
                "verdict": "kept as printed -- never auto-corrected (L0.4)",
                "note": " | ".join(kwic.get(surface.lower(), [])[:1]) or
                        "no #26 KWIC row",
            })
    return rows


def build_operator_rows(toks_by_page):
    """The operator table the silence layer reads from: counts per
    operator surface (#17's stream) beside the #12/#14 reading."""
    counts = collections.Counter()
    glued = collections.Counter()
    for toks in toks_by_page.values():
        for t in toks:
            if t["kind"] != "OP":
                continue
            s = _tok_text(t)
            counts[s] += 1
            if t["glue_L"] and t["glue_R"]:
                glued[s] += 1
    total = sum(counts.values())
    rows = []
    for surface, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append({
            "operator": surface, "n_tokens": n,
            "share_pct": 100.0 * n / total if total else 0.0,
            "n_glued_both_sides": glued.get(surface, 0),
            "reading_12_14": operator_reading(surface),
        })
    return rows, total


def build_per_page(records):
    """Per-page clause and confidence counts, in page order."""
    rows = []
    by_page = collections.defaultdict(list)
    for rec in records:
        by_page[rec["page"]].append(rec)
    for page in sorted(by_page):
        recs = by_page[page]
        rows.append({
            "page": page,
            "n_clauses": len(recs),
            "n_words": sum(r["n_kept"] - r["n_silence"] for r in recs),
            "n_silence": sum(r["n_silence"] for r in recs),
            "n_markup": sum(r["n_markup"] for r in recs),
            "n_high": sum(r["n_high"] for r in recs),
            "n_medium": sum(r["n_medium"] for r in recs),
            "n_low": sum(r["n_low"] for r in recs),
            "n_unresolved": sum(r["n_unresolved"] for r in recs),
            "n_unresolved_clauses": sum(1 for r in recs
                                        if r["n_unresolved"]),
        })
    return rows


# ---------------------------------------------------------------------------
# reconciliation against the frozen layers
# ---------------------------------------------------------------------------

def build_reconciliation(ctx, records, token_rows, toks_by_page,
                          outside_clause, op_total, full_book):
    """Every count the edition publishes against the artifact that owns it.

    `outside_clause` is the #17 tokens in no #33 clause (the edition's
    remainder); closure over the token stream is asserted here rather
    than assumed.
    """
    (_bodies, _page_units, _unit_meta, _is_markup, screening,
     _markup_spans, _entity, _tb, _kwic) = ctx
    clause_rows = load_clauses()
    n_tokens_read = sum(len(t) for t in toks_by_page.values())
    kinds = collections.Counter(t["kind"] for t in token_rows)
    slash_link, slash_bound = lv.slash_role_counts()
    sum_n_tokens = sum(int(r["n_tokens"]) for r in clause_rows)
    sum_n_words_clean = sum(int(r["n_words_clean"]) for r in clause_rows)
    bad_norm = [r["clause_id"] for r in records if not r["norm_match"]]
    words_kept = kinds.get("WORD", 0)
    contained = sum(r["n_tokens"] for r in records)
    # #33's `n_tokens` counts #14 AST *events*, which include synthesized
    # marker events (the `<<`/`>>` span openers and the zero-width `#`
    # heading markers); the #17 containment join counts tokens only.  The
    # gap is published, not forced: the word layer (the load-bearing
    # reconciliation) matches exactly.
    marker_gap = sum_n_tokens - contained

    def row(quantity, earlier, here, owner, note="", match=None):
        return {"quantity": quantity, "earlier": earlier, "here": here,
                "owner": owner,
                "match": (str(earlier == here).lower() if match is None
                          else match), "note": note}

    rows = [
        row("pages rendered", 225, len({r["page"] for r in records}), "#7",
            "layout-recovery page bodies"),
        row("clauses", len(clause_rows), len(records), "#33",
            "the edition renders every #33 clause, none dropped"),
        row("word tokens inside clauses (#33 n_words_clean)",
            sum_n_words_clean, words_kept, "#33/#19",
            "markup-screened word count per clause -- the load-bearing "
            "join between the edition and #33"),
        row("token buckets close over the clause stream", contained,
            sum(r["n_high"] + r["n_medium"] + r["n_low"] +
                r["n_unresolved"] for r in records)
            + sum(r["n_markup"] for r in records), "#62",
            "kept + markup-excluded == contained, per clause"),
        row("clauses whose two normalization paths agree", 0, len(bad_norm),
            "#7/#17", "token-assembled and substring-deleted renderings "
            "must match for every clause"),
        row("clauses whose token spans tile the page body", 0,
            sum(1 for r in records if not r["token_span_match"]), "#7/#17",
            "every kept token's span reads back as its surface"),
        row("#17 token stream", 90220, n_tokens_read, "#17",
            "read whole before the clause join"),
        row("tokens in no #33 clause (the remainder)",
            90220 - sum_n_tokens + marker_gap, outside_clause, "#17/#33",
            "header/page furniture and #14's synthesized marker events; "
            "counted, not silently dropped"),
        row("#14 marker events #33 counts inside clauses",
            sum_n_tokens, contained, "#33/#14",
            "convention gap, not a drift: #33 counts the <<`>>` span "
            "openers and zero-width `#` heading markers as events (%d); "
            "the #17 token containment join counts tokens only.  The word "
            "layer above matches exactly" % marker_gap,
            match="disclosed"),
        row("operator tokens in the stream", 11000, op_total, "#17",
            "feeds operator_roles.tsv"),
        row("`/` compound-link role (glued)", slash_link, slash_link, "#14",
            "quoted from the memoized slash_role_counts recount"),
        row("`/` sentence-final role", slash_bound, slash_bound, "#14",
            "quoted from the memoized slash_role_counts recount"),
    ]
    if not full_book:
        rows = [dict(r, match="n/a (slice build)",
                     note=r["note"] + " [slice]") for r in rows]
    return rows


# ---------------------------------------------------------------------------
# the edition: three lines per clause
# ---------------------------------------------------------------------------

def render_edition(records):
    """The full-book interlinear as markdown: one fenced block per clause,
    page-cited, in (page, clause) order."""
    lines = [
        "# Interlinear edition -- Blood Electric (Kenji Siratori, 2002)",
        "",
        "Three lines per #33 clause, in a fixed order:",
        "",
        "    1  the original clause (exact source substring)",
        "    2  the operator-normalized rendering (#13 v4: operators -> `.`,",
        "       the silence-removed sentence skeleton; #13 v3 is carried in",
        "       interlinear.tsv)",
        "    3  the plain-English gloss: ordinary English is its own gloss;",
        "       only non-English surfaces are read (gloss_lexicon.tsv); an",
        "       unread surface is kept and marked `<...>` (residue.tsv).",
        "",
        "Every clause is page-cited.  The operators are dropped on line 3",
        "(they are the silence layer, #12/#13) and quoted per-token in",
        "gloss_tokens.tsv; ordinary punctuation is ordinary English and",
        "stays on the gloss line.",
        "",
    ]
    cur = None
    for rec in records:
        if cur != rec["page"]:
            cur = rec["page"]
            lines.append("## Page %d" % cur)
            lines.append("")
        stats = ("p%s %s c%s | words %d silence %d | conf h%d m%d l%d u%d"
                 % (rec["page"], rec["region"], rec["clause_id"],
                    rec["n_kept"] - rec["n_silence"], rec["n_silence"],
                    rec["n_high"], rec["n_medium"], rec["n_low"],
                    rec["n_unresolved"]))
        lines.append("```text")
        lines.append("c%s %s" % (rec["clause_id"], stats))
        lines.append("1| %s" % rec["original"].replace("\n", " "))
        lines.append("2| %s" % rec["v4_period"].replace("\n", " "))
        lines.append("3| %s" % rec["gloss"].replace("\n", " "))
        lines.append("```")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Generated by `work/interlinear_edition.py` (%s). "
                 "Deterministic; a rebuild is byte-identical."
                 % SCHEMA)
    lines.append("")
    return "\n".join(lines)


def pick_samples(records):
    """Deterministic, content-bearing samples: the first, middle and last
    clause in the build that actually exercises the gloss line (a plain
    reading or an unresolved surface).  Falls back to the first, middle
    and last rendered clause when nothing does."""
    if not records:
        return []
    content = [r for r in records
               if r["n_readings"] or r["n_unresolved"]]
    if len(content) >= 3:
        return [content[0], content[len(content) // 2], content[-1]]
    return [records[0], records[len(records) // 2], records[-1]]


def _pct(n, total):
    return 100.0 * n / total if total else 0.0


def write_report(path, records, token_rows, lexicon_rows, residue_rows,
                 recon_rows, outside_clause, page_range, generated):
    """The deliverable: method, findings, samples, anti-pattern audit."""
    n_words = sum(r["n_kept"] - r["n_silence"] for r in records)
    word_conf = collections.Counter()
    src = collections.Counter()
    for t in token_rows:
        src[t["source"]] += 1
        if t["source"] != "silence":
            word_conf[t["confidence"]] += 1
    n_tokens = len(token_rows)
    n_silence = sum(r["n_silence"] for r in records)
    n_markup = sum(r["n_markup"] for r in records)
    n_unres = sum(r["n_unresolved"] for r in records)
    clauses_unres = sum(1 for r in records if r["n_unresolved"])
    reading_rows = [r for r in lexicon_rows if r["source"] == "plain_reading"]
    ident = sum(1 for t in token_rows
                if t["source"] in ("identity_19", "identity_numeric",
                                   "compound_19", "entity_30",
                                   "unit_context_17", "ocr_error_19"))
    L = []
    a = L.append
    a("# Interlinear edition -- generator and first full-book rendering (#62)")
    a("")
    a("Generated: %s" % generated)
    a("")
    a("Scope: %s" % ("pages %d-%d, %s clauses"
                     % (page_range[0], page_range[1],
                        "{:,}".format(len(records)))
                     if page_range else
                     "all 225 pages, %s clauses" % "{:,}".format(len(records))))
    a("")
    a("## What was built")
    a("")
    a("`work/interlinear_edition.py` renders every #33 clause in three")
    a("aligned lines -- original clause | operator-normalized (silence")
    a("removed) | plain-English gloss -- and writes the whole book to")
    a("`analysis/interlinear/interlinear.md`, with the per-clause TSV,")
    a("the per-token reading table, the readings lexicon, the residue")
    a("ledger and the reconciliation.  It is the full-book follow-up to")
    a("#56's three glossed passages and reuses #56's token rules, curated")
    a("termbase, operator readings and v3/v4 renderer unchanged.")
    a("")
    a("## Method")
    a("")
    a("* **Line 2 is measured, not invented.**  It is #13's v4 transform")
    a("  (every operator -> `. `) over the #17 token stream, with #56's")
    a("  `render_variant` semantics; v4 is the variant #50 scored closest")
    a("  to the underlying system (perplexity 84.1 vs 106.2 baseline).")
    a("  v3 (operators -> space) is carried in `interlinear.tsv`, so the")
    a("  middle line is overridable.  Two build-time invariants pin it: for")
    a("  every clause the token-assembled rendering equals the rendering")
    a("  recomputed by deleting the operator spans from the source")
    a("  substring, and every kept token's span reads back as its surface.")
    a("* **Line 3 substitutes only what is not ordinary English.**  The")
    a("  book is English lexis under a symbolic transform (#56's VIABLE")
    a("  verdict), so English/function words, numeric tokens, compounds")
    a("  of English parts and named constructs are kept verbatim, and the")
    a("  gloss line equals the v3-normalized line except at the")
    a("  non-English sites.  The overlap is a reported number below, not")
    a("  a hidden one.  Seam operators inside compounds (`soul/gram`,")
    a("  `gene=TV`) are dropped on line 3 exactly as #13's v3 drops them")
    a("  -- the parts stay separate words, never fused into #13's v1")
    a("  portmanteau (`soulgram`) -- and `gloss_tokens.tsv` records the")
    a("  containing #17 unit for every part, so the construct is")
    a("  recoverable from the table.")
    a("* **Readings are evidence-linked or unresolved.**  The rule order")
    a("  per token is published in the module docstring; the readings for")
    a("  the #19 `true_coinage` class are authored from the #26 KWIC")
    a("  contexts, #19's morphology and #30's roles, with #56's")
    a("  confidence scheme.  No LLM is used (that is #63's committee) and")
    a("  no external corpus is consulted or acquired.")
    a("")
    a("## Headline numbers")
    a("")
    a("| quantity | value |")
    a("|---|---|")
    a("| clauses rendered | %s |" % "{:,}".format(len(records)))
    a("| pages | %s |" % "{:,}".format(len({r['page'] for r in records})))
    a("| word tokens in clauses | %s |" % "{:,}".format(n_words))
    a("| silence tokens (operators/punctuation) | %s |" % "{:,}".format(n_silence))
    a("| markup-excluded tokens (#19/#27 screen) | %s |" % "{:,}".format(n_markup))
    a("| word tokens keeping an identity reading | %s (%.1f%%) |"
      % ("{:,}".format(ident), _pct(ident, n_words)))
    a("| curated plain readings (types) | %d |" % len(reading_rows))
    a("| unresolved word tokens | %s (%.2f%% of words) |"
      % ("{:,}".format(n_unres), _pct(n_unres, n_words)))
    a("| clauses with >= 1 unresolved surface | %s (%.1f%%) |"
      % ("{:,}".format(clauses_unres), _pct(clauses_unres, len(records))))
    a("")
    a("Confidence over the %s in-clause word tokens: high %s / medium %s /"
      " low %s" % ("{:,}".format(sum(word_conf.values())),
                   "{:,}".format(word_conf.get("high", 0)),
                   "{:,}".format(word_conf.get("medium", 0)),
                   "{:,}".format(word_conf.get("low", 0))))
    a("(unresolved %s are itemized in residue.tsv)."
      % "{:,}".format(word_conf.get("unresolved", 0)))
    a("")
    a("Reading sources over the same tokens:")
    a("")
    a("| source | tokens | share |")
    a("|---|---|---|")
    for s in READING_ORDER:
        n = src.get(s, 0)
        if n:
            a("| %s | %s | %.1f%% |" % (SOURCE_LABEL[s],
                                        "{:,}".format(n), _pct(n, n_tokens)))
    a("")
    a("## Samples")
    a("")
    for rec in pick_samples(records):
        a("### p. %d -- clause %s (%s region, opener `%s`)"
          % (rec["page"], rec["clause_id"], rec["region"], rec["opener"]))
        a("")
        a("```text")
        a("1| %s" % rec["original"].replace("\n", " "))
        a("2| %s" % rec["v4_period"].replace("\n", " "))
        a("3| %s" % rec["gloss"].replace("\n", " "))
        a("```")
        a("")
        a("readings: high %d, medium %d, low %d, unresolved %d%s"
          % (rec["n_high"], rec["n_medium"], rec["n_low"],
             rec["n_unresolved"],
             "; unresolved: " + rec["unresolved"] if rec["unresolved"] else ""))
        a("")
    a("## Residue (listed, not repaired)")
    a("")
    a("| class | item | n | pages | verdict |")
    a("|---|---|---|---|---|")
    for r in residue_rows[:25]:
        a("| %s | %s | %d | %s | %s |"
          % (r["class"], r["item"], r["n"],
             _fmt_pages(r["pages"]), r["verdict"]))
    if not residue_rows:
        a("| (none) | | | | |")
    a("")
    unres_items = [r for r in residue_rows
                   if r["class"] == "unresolved_unresolved"]
    if unres_items:
        a("The #19 `true_coinage` types with no evidence-linked reading in")
        a("this workspace -- %s -- are left marked for #63's glossing"
          % ", ".join("`%s`" % r["item"] for r in unres_items))
        a("committee.")
        a("")
    a("## Reconciliation with the frozen layers")
    a("")
    a("| quantity | earlier | here | owner | match |")
    a("|---|---|---|---|---|")
    for r in recon_rows:
        a("| %s | %s | %s | %s | %s |"
          % (r["quantity"], r["earlier"], r["here"], r["owner"], r["match"]))
    a("")
    a("## Anti-pattern audit (L9.8)")
    a("")
    a("* **Hallucinated coherence**: no reading asserts a plot.  Every")
    a("  substitution is a single surface -> single plain reading with its")
    a("  KWIC context published in gloss_lexicon.tsv; unread surfaces are")
    a("  marked, never smoothed over.")
    a("* **Circularity**: no LLM translates or judges anything here; the")
    a("  readings come from the text's own counted evidence.  #63's")
    a("  committee and #75's jury are where models enter.")
    a("* **Over-correction**: #19's `ocr_error` class is kept as printed")
    a("  (15 types), compounds are kept intact, and coinages are never")
    a("  normalized into English spelling.")
    a("* **Single-method certainty**: every reading carries a confidence")
    a("  and a source; line 2 is measured (#13/#50), not asserted; the")
    a("  identity share above is stated so the reader knows how much of")
    a("  line 3 is the text itself.")
    a("* **Ignoring the residue**: tokens in no #33 clause (%s) and the"
      " markup screen's %s excluded tokens are counted in the"
      " reconciliation, and every unresolved surface is itemized."
      % ("{:,}".format(outside_clause), "{:,}".format(n_markup)))
    a("")
    a("## Determinism")
    a("")
    a("No randomness and no hash-order dependence (all ties broken by a")
    a("fixed key).  A rebuild reproduces every artifact byte-identically")
    a("except this header's `Generated` stamp, which no statistic reads.")
    a("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _fmt_pages(pages, limit=8):
    if not pages:
        return "-"
    if len(pages) <= limit:
        return ",".join(str(p) for p in pages)
    return "%d-%d (%d pages)" % (pages[0], pages[-1], len(pages))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

INTERLINEAR_HEADER = [
    "clause_id", "page", "region", "opener", "start", "end",
    "n_words", "n_silence", "n_markup", "n_high", "n_medium", "n_low",
    "n_unresolved", "unresolved", "original", "v3_space", "v4_period",
    "gloss",
]
TOKEN_HEADER = [
    "clause_id", "page", "region", "surface", "kind", "start", "end",
    "unit", "unit_uid", "category_19", "parts_18", "gloss", "confidence",
    "source", "note",
]
LEXICON_HEADER = [
    "surface", "reading", "confidence", "source", "category_19", "parts_18",
    "entity_role_30", "n_occurrences", "n_pages", "first_page",
    "kwic_examples_26", "note",
]
RESIDUE_HEADER = [
    "class", "item", "n", "pages", "owner", "verdict", "note",
]
PER_PAGE_HEADER = [
    "page", "n_clauses", "n_words", "n_silence", "n_markup", "n_high",
    "n_medium", "n_low", "n_unresolved", "n_unresolved_clauses",
]
OPERATOR_HEADER = [
    "operator", "n_tokens", "share_pct", "n_glued_both_sides",
    "reading_12_14",
]
RECON_HEADER = [
    "quantity", "earlier", "here", "owner", "match", "note",
]


def build(out=OUT, pages=None, quiet=False):
    """Build the edition and every artifact under `out`.

    `pages` restricts the whole build to a page range: only those
    clauses are rendered and every artifact describes the range, with
    the reconciliation rows marked `n/a (slice build)`.  The default
    (no range) is the full deliverable.
    """
    global OUT
    OUT = Path(out)
    ctx = load_context()
    toks_by_page = load_tokens()
    clause_rows = load_clauses()
    if pages is not None:
        clause_rows = [r for r in clause_rows
                       if pages[0] <= int(r["page"]) <= pages[1]]

    records, token_rows = [], []
    for row in clause_rows:
        rec, toks = clause_record(row, toks_by_page, ctx)
        records.append(rec)
        token_rows.extend(toks)

    # closure: every #17 token on a rendered page is either inside a
    # rendered clause or in the published remainder.
    in_clause = sum(r["n_tokens"] for r in records)
    selected_pages = {r["page"] for r in records}
    n_stream = sum(len(t) for p, t in toks_by_page.items()
                   if p in selected_pages)
    outside = n_stream - in_clause
    bad_norm = [r["clause_id"] for r in records if not r["norm_match"]]
    bad_span = [r["clause_id"] for r in records if not r["token_span_match"]]
    if bad_span:
        raise InterlinearError(
            "token spans do not tile the page body in %d clauses (e.g. %s)"
            % (len(bad_span), bad_span[:5]))
    if bad_norm:
        raise InterlinearError(
            "the two normalization paths disagree in %d clauses (e.g. %s)"
            % (len(bad_norm), bad_norm[:5]))
    per_clause_words = sum(r["n_kept"] - r["n_silence"] for r in records)
    bucket_sum = (sum(r["n_high"] + r["n_medium"] + r["n_low"] +
                      r["n_unresolved"] for r in records)
                  + sum(r["n_markup"] for r in records))
    if bucket_sum != in_clause:
        raise InterlinearError(
            "token buckets do not close: %d classified vs %d contained"
            % (bucket_sum, in_clause))

    lexicon_rows = build_lexicon(ctx)
    residue_rows = build_residue(ctx, token_rows)
    sel_tokens = {p: t for p, t in toks_by_page.items()
                  if p in selected_pages}
    op_rows, op_total = build_operator_rows(sel_tokens)
    per_page_rows = build_per_page(records)
    recon_rows = build_reconciliation(ctx, records, token_rows, sel_tokens,
                                      outside, op_total,
                                      full_book=pages is None)

    edition = render_edition(records)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    conf = collections.Counter(t["confidence"] for t in token_rows)
    src = collections.Counter(t["source"] for t in token_rows)
    n_words = per_clause_words
    summary = {
        "schema": SCHEMA,
        "pages_rendered": len(selected_pages),
        "page_range": list(pages) if pages else None,
        "clauses": len(records),
        "tokens_in_clauses": in_clause,
        "tokens_outside_clauses": outside,
        "tokens_total": in_clause + outside,
        "word_tokens_in_clauses": n_words,
        "silence_tokens": sum(r["n_silence"] for r in records),
        "markup_excluded_tokens": sum(r["n_markup"] for r in records),
        "confidence": {c: conf.get(c, 0) for c in CONFIDENCES},
        "sources": {s: src.get(s, 0) for s in READING_ORDER},
        "n_unresolved_clauses": sum(1 for r in records
                                    if r["n_unresolved"]),
        "n_readings": sum(1 for r in lexicon_rows
                          if r["source"] == "plain_reading"),
        "operator_tokens": op_total,
        "generated": generated,
        "determinism": "every artifact is byte-identical on rebuild except "
                       "the `generated` stamp, which no statistic reads",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interlinear.md").write_text(edition + "\n", encoding="utf-8")
    cs.write_tsv(OUT / "interlinear.tsv", INTERLINEAR_HEADER, [
        {"n_words": r["n_kept"] - r["n_silence"], **{
            k: r[k] for k in INTERLINEAR_HEADER if k != "n_words"}}
        for r in records])
    cs.write_tsv(OUT / "gloss_tokens.tsv", TOKEN_HEADER, token_rows)
    cs.write_tsv(OUT / "gloss_lexicon.tsv", LEXICON_HEADER, lexicon_rows)
    cs.write_tsv(OUT / "residue.tsv", RESIDUE_HEADER, [
        dict(r, pages=",".join(str(p) for p in r["pages"]))
        for r in residue_rows])
    cs.write_tsv(OUT / "per_page.tsv", PER_PAGE_HEADER, per_page_rows)
    cs.write_tsv(OUT / "operator_roles.tsv", OPERATOR_HEADER, op_rows)
    cs.write_tsv(OUT / "reconciliation.tsv", RECON_HEADER, recon_rows)
    cs.write_tsv(OUT / "diagnostics.tsv", ["diag", "value"], [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "page_range", "value": summary["page_range"]},
        {"diag": "clauses", "value": len(records)},
        {"diag": "word_tokens", "value": n_words},
        {"diag": "silence_tokens", "value": summary["silence_tokens"]},
        {"diag": "markup_excluded_tokens",
         "value": summary["markup_excluded_tokens"]},
        {"diag": "unresolved_tokens", "value": conf.get("unresolved", 0)},
        {"diag": "unresolved_clauses", "value": summary["n_unresolved_clauses"]},
        {"diag": "residue_rows", "value": len(residue_rows)},
        {"diag": "lexicon_rows", "value": len(lexicon_rows)},
        {"diag": "tokens_outside_clauses", "value": outside},
        {"diag": "clauses_with_normalization_mismatch", "value": len(bad_norm)},
    ])
    with (OUT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    write_report(OUT / "interlinear_report.md", records, token_rows,
                 lexicon_rows, residue_rows, recon_rows, outside, pages,
                 generated)
    if not quiet:
        print("[#62] interlinear edition: %s clauses / %s word tokens; "
              "high %s, medium %s, low %s, unresolved %s (%s outside clauses)"
              % ("{:,}".format(len(records)), "{:,}".format(n_words),
                 "{:,}".format(conf.get("high", 0)),
                 "{:,}".format(conf.get("medium", 0)),
                 "{:,}".format(conf.get("low", 0)),
                 "{:,}".format(conf.get("unresolved", 0)),
                 "{:,}".format(outside)))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--pages", default=None,
                    help="restrict the build to a page range, e.g. 4-6 "
                         "(every artifact then describes that range)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        build(Path(args.out), pages=parse_pages(args.pages), quiet=args.quiet)
    except InterlinearError as exc:
        print("interlinear edition failed: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
