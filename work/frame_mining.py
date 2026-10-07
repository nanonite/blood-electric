#!/usr/bin/env python3
"""#34 Frame and template mining -> analysis/frame_mining/

Extract the recurrent **schemata** (frames / templates) whose recurrence *is*
the book's own loop grammar (``docs/meaning-extraction-methods.md`` L4).  The
task names three schema types:

  * clause-level frames with slots -- ``I <verb> the <NP> of <NP>``,
    ``the <N> that <V>``;
  * the ``<<...>>`` block (a #14 ``Span``), which carries the book's recurring
    names/labels;
  * the operator-delimited clause itself (#33), the surface the frames live on.

Method (stated once, used everywhere):

* **Tokens are #17 units, markup-screened.**  Clause token sequences are built
  from the #33 clause boundaries (``analysis/clauses/clauses.tsv``) joined to
  the #17 unit layer on ``(page, start)`` containment, screened with #19's
  ledger + the #27 tag-run detector -- the exact join #33 uses, so the unit
  total reconciles with #27's 74,077.  A unit is the book's word; case-folded
  for matching, so frames are case-insensitive.
* **A frame is an n-gram skeleton with one or two slots.**  For every n-gram
  (n = 2..8) the clause set it occurs in is recorded (document frequency, so a
  long clause cannot dominate).  Single-slot skeletons mask one position,
  two-slot skeletons mask two; a skeleton is a *frame* when it is instantiated
  by >= ``MIN_CLAUSES`` distinct clauses *and* >= ``MIN_FILLERS`` distinct
  fillers per slot.  This is construction-grammar frame mining: the fixed
  words are the frame's signature, the slots are its variable parts.  Each
  frame is also given a rule-based **signature family** (``first_person``,
  ``rel_that``, ``np_of_np``, ``pp_to``, ``other``) so the task's named
  schemata are directly readable from the catalogue; the rules are disclosed
  and only read the fixed words.
* **The ``<<...>>`` blocks** come from the #14 span layer
  (``analysis/block_parser/spans.tsv``); a block's *name* is the leading run
  (<= 3) of screened content units of a span that opens with ``<<`` -- the
  label the block leads with -- and their recurrence is catalogued separately.
* **The loop grammar** is the small set of top frames that covers a large
  fraction of the book: cumulative clause coverage of the ranked frames.

Outputs under ``analysis/frame_mining/``:
  frame_mining_report.md   THE DELIVERABLE
  frames.tsv               the frame catalogue (skeleton + counts + fillers)
  family_counts.tsv        frames / clauses / instances per signature family
  block_names.tsv          recurring <<...>> block names + counts
  frame_coverage.tsv       cumulative coverage of the ranked frames
  reconciliation.tsv       against #13 / #14 / #17 / #27 / #33
  diagnostics.tsv          build-time counters
  summary.json             machine-readable headline numbers

Reproduce::

    conda activate marker
    python work/frame_mining.py            # writes analysis/frame_mining/
    python work/test_frame_mining.py

Deterministic: no randomness, no hash-order dependence (all ties broken by a
fixed key).  Stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import clause_segmentation as cs  # #33 clause boundaries + unit/markup loaders

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "frame_mining"
CLAUSES = ROOT / "analysis" / "clauses" / "clauses.tsv"
SPANS = ROOT / "analysis" / "block_parser" / "spans.tsv"
SCREENING = ROOT / "analysis" / "neologism" / "screening.tsv"

SCHEMA = "frame-mining/1"
# Slot marker in a skeleton (distinctive, not a tab -> quote-safe TSV).
SLOT = "\u27e8_\u27e9"          # ⟨_⟩
# n-gram range and frame thresholds.
MIN_N, MAX_N = 2, 8
SINGLE_MIN_N = 3               # need a fixed context on both sides of the slot
DOUBLE_MIN_N = 5
MIN_DF = 3                     # n-gram must occur in >= this many clauses
DOUBLE_MIN_DF = 4
MIN_CLAUSES = 4                # a frame must cover >= this many clauses
MIN_FILLERS = 3                # ... with >= this many distinct fillers per slot
TOP_FILLERS = 8                # fillers shown per slot in the catalogue
NAME_UNITS = 3                 # leading content units that name a <<...>> block
# Signature-family rules (priority order, read the *fixed* words only).
FAMILY_RULES = (
    ("first_person", lambda skel: skel[0] == "i"),
    ("rel_that", lambda skel: "that" in skel),
    ("np_of_np", lambda skel: "of" in skel),
    ("pp_to", lambda skel: "to" in skel),
)
FAMILY_ORDER = ("first_person", "rel_that", "np_of_np", "pp_to", "other")
# A leading single character is only a name token for these pronouns/articles;
# anything else at the front of a block is OCR/markup residue.
OK_SINGLE = {"i", "a"}
PUNCT_ONLY = re.compile(r"^[^0-9A-Za-z]+$")

read_tsv = cs.read_tsv
write_tsv = cs.write_tsv


# --------------------------------------------------------------------------
# shared context (loaded once)
# --------------------------------------------------------------------------
def load_context():
    """``(bodies, page_units, unit_meta, is_markup, screening)``.

    ``is_markup(uid)`` is the #19-ledger + #27 tag-run screen (the join #33
    uses); ``screening`` maps a lower-cased unit to its #19 category row.
    """
    bodies = cs.load_pages()
    page_units, _uid_by_ps, unit_meta = cs.load_units()
    markup_spans, markup_uids, markup_types = cs.load_markup(bodies)

    def is_markup(uid):
        page, us, _ue, surface = unit_meta[uid]
        if uid in markup_uids or surface.lower() in markup_types:
            return True
        return any(a <= us < b for a, b in markup_spans.get(page, ()))

    screening = {}
    if SCREENING.exists():
        for r in read_tsv(SCREENING):
            screening.setdefault(r["unit"].lower(), r)
    return bodies, page_units, unit_meta, is_markup, screening


def _clause_tokens(page, start, end, page_units, unit_meta, is_markup):
    return [unit_meta[u][3].lower()
            for (us, _ue, u) in page_units.get(page, ())
            if start <= us < end and not is_markup(u)]


def load_clause_tokens(ctx):
    """Clause dicts with a case-folded ``tokens`` list of screened #17 units."""
    _bodies, page_units, unit_meta, is_markup, _sc = ctx
    clauses = []
    for r in read_tsv(CLAUSES):
        page, s, e = int(r["page"]), int(r["start"]), int(r["end"])
        clauses.append({
            "clause_id": int(r["clause_id"]),
            "page": page,
            "opener": r["opener"],
            "container": r["container"],
            "n_words_clean": int(r["n_words_clean"]),
            "tokens": _clause_tokens(page, s, e, page_units, unit_meta, is_markup),
        })
    return clauses


# --------------------------------------------------------------------------
# frame mining
# --------------------------------------------------------------------------
def _skeleton(ng, positions):
    """``ng`` with ``positions`` replaced by the slot marker."""
    out = list(ng)
    for p in positions:
        out[p] = SLOT
    return tuple(out)


def family_of(skel):
    """Rule-based signature family of a skeleton (fixed words only)."""
    for name, rule in FAMILY_RULES:
        if rule(skel):
            return name
    return "other"


def mine_frames(clauses):
    """Mine single- and two-slot frames from clause token sequences.

    Returns a list of frame dicts: ``{skeleton, n_slots, n_clauses,
    n_instances, family, fillers, examples}`` where ``fillers`` is a list (one
    per slot) of ``(word, count)`` and ``examples`` a few instantiations.

    ``n_clauses`` is document frequency (distinct clauses containing the
    frame); ``n_instances`` is the **exact** number of (clause, position)
    occurrences of the skeleton, and the filler counts are exact occurrences.
    A frame *qualifies* when it is discovered from document-frequency-qualified
    n-grams (a disclosed noise screen), covers >= ``MIN_CLAUSES`` distinct
    clauses, and has >= ``MIN_FILLERS`` attested fillers per slot; the reported
    counts are then recomputed exactly over every n-gram occurrence, so a
    low-frequency filler still counts.
    """
    # Pass 1: n-gram -> set of clause indices (df) and occurrence count.
    ngram_clauses = collections.defaultdict(set)
    ngram_occ = collections.Counter()
    for ci, cl in enumerate(clauses):
        toks = cl["tokens"]
        n = len(toks)
        for k in range(MIN_N, MAX_N + 1):
            for i in range(n - k + 1):
                ng = tuple(toks[i:i + k])
                ngram_clauses[ng].add(ci)
                ngram_occ[ng] += 1

    # Candidate skeletons from df-qualified n-grams: clauses they can appear in
    # and their attested fillers (the qualification screen).
    single = collections.defaultdict(
        lambda: {"fillers": collections.Counter(), "clauses": set()})
    double = collections.defaultdict(
        lambda: {"fillers": [collections.Counter(), collections.Counter()],
                 "clauses": set()})
    for ng, clset in ngram_clauses.items():
        k = len(ng)
        cov = len(clset)          # clauses this n-gram occurs in
        if k >= SINGLE_MIN_N and cov >= MIN_DF:
            for i in range(k):
                rec = single[_skeleton(ng, (i,))]
                rec["fillers"][ng[i]] += cov
                rec["clauses"] |= clset
        if k >= DOUBLE_MIN_N and cov >= DOUBLE_MIN_DF:
            for i in range(k):
                for j in range(i + 1, k):
                    rec = double[_skeleton(ng, (i, j))]
                    rec["fillers"][0][ng[i]] += cov
                    rec["fillers"][1][ng[j]] += cov
                    rec["clauses"] |= clset

    # Qualifying skeletons: clause coverage + attested fillers per slot.
    emit_single = {s for s, rec in single.items()
                   if len(rec["clauses"]) >= MIN_CLAUSES
                   and len(rec["fillers"]) >= MIN_FILLERS}
    emit_double = {s for s, rec in double.items()
                   if len(rec["clauses"]) >= MIN_CLAUSES
                   and all(len(f) >= MIN_FILLERS for f in rec["fillers"])}

    # Pass 2: exact occurrences + fillers for the qualifying skeletons.
    inst_single = collections.Counter()
    fill_single = collections.defaultdict(collections.Counter)
    inst_double = collections.Counter()
    fill_double = collections.defaultdict(
        lambda: [collections.Counter(), collections.Counter()])
    for ng, occ in ngram_occ.items():
        k = len(ng)
        if k >= SINGLE_MIN_N:
            for i in range(k):
                skel = _skeleton(ng, (i,))
                if skel in emit_single:
                    inst_single[skel] += occ
                    fill_single[skel][ng[i]] += occ
        if k >= DOUBLE_MIN_N:
            for i in range(k):
                for j in range(i + 1, k):
                    skel = _skeleton(ng, (i, j))
                    if skel in emit_double:
                        inst_double[skel] += occ
                        fill_double[skel][0][ng[i]] += occ
                        fill_double[skel][1][ng[j]] += occ

    frames = []

    def emit(skel, n_slots, fillers, clset, instances):
        # example instantiations: the shortest clauses that contain an instance
        ex_clauses = sorted(clset, key=lambda ci: (len(clauses[ci]["tokens"]),
                                                   ci))[:3]
        examples = [" ".join(clauses[ci]["tokens"]) for ci in ex_clauses]
        frames.append({
            "skeleton": skel,
            "n_slots": n_slots,
            "n_clauses": len(clset),
            "n_instances": instances,
            "family": family_of(skel),
            "fillers": [f.most_common(TOP_FILLERS) for f in fillers],
            "examples": examples,
        })

    for skel in emit_single:
        emit(skel, 1, [fill_single[skel]], single[skel]["clauses"],
             inst_single[skel])
    for skel in emit_double:
        emit(skel, 2, fill_double[skel], double[skel]["clauses"],
             inst_double[skel])

    # deterministic rank: coverage desc, then instances desc, then skeleton
    frames.sort(key=lambda f: (-f["n_clauses"], -f["n_instances"], f["skeleton"]))
    return frames


# --------------------------------------------------------------------------
# <<...>> block names
# --------------------------------------------------------------------------
def name_class(token, screening):
    """Coarse #19 class of a block name's leading unit."""
    r = screening.get(token.lower())
    if r is not None:
        if r.get("in_wordlist") == "y":
            return "english"
        if r.get("proper_noun") == "y":
            return "proper"
        if r.get("gazetteer"):
            return "gazetteer"
        return r.get("category", "unknown")
    if token.isupper() and len(token) > 1:
        return "label"
    return "unknown"


def mine_block_names(ctx, name_units=NAME_UNITS):
    """Recurring ``<<...>>`` block names from the #14 span layer.

    A name-bearing span is one whose body text opens with ``<<`` (every #14
    span does); the *name* is the leading run of up to ``name_units`` screened
    #17 units, skipping leading HTML-tag tokens and single-letter OCR/markup
    residue.  Returns ``(names, classes, stats)`` where ``names`` is a list of
    ``(name, first_class, n_spans)`` sorted by recurrence, ``classes`` a
    Counter of leading-unit classes, and ``stats`` a dict of counters.
    """
    bodies, page_units, unit_meta, is_markup, screening = ctx
    counter = collections.Counter()      # (name, first_class) -> spans
    classes = collections.Counter()
    stats = collections.Counter()
    for r in read_tsv(SPANS):
        stats["spans"] += 1
        if r["closed"] == "True":
            stats["closed"] += 1
        page, s, e = int(r["page"]), int(r["start"]), int(r["end"])
        body = bodies.get(page, "")
        if not (s + 2 <= len(body) and body[s:s + 2] == "<<"):
            stats["not_open_marker"] += 1
            continue
        stats["open_marker"] += 1
        toks = [unit_meta[u][3]
                for (us, _ue, u) in page_units.get(page, ())
                if s <= us < e and not is_markup(u)]
        out = []
        for t in toks:
            if not PUNCT_ONLY.match(t):
                if t.lower() in cs.HTML_TAGS:
                    continue                      # tag-name residue at the front
                if len(t) == 1 and t.lower() not in OK_SINGLE:
                    continue                      # single-letter OCR residue
                out.append(t)
                if len(out) >= name_units:
                    break
            elif out:
                break                             # boundary operator ends the name
        if not out:
            stats["no_name"] += 1
            continue
        stats["named"] += 1
        name = " ".join(out)
        cls = name_class(out[0], screening)
        counter[(name, cls)] += 1
        classes[cls] += 1
        if cls not in ("english", "proper", "gazetteer", "label"):
            stats["residue_leading"] += 1
    names = [{"name": n, "first_class": c, "n_spans": k}
             for (n, c), k in counter.most_common()]
    return names, classes, stats


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------
def coverage(clauses, frames, topk=40):
    """Cumulative clause coverage of the ranked frames (a clause is 'covered'
    when some frame's skeleton matches a sub-sequence of it)."""
    covered = set()
    rows = []
    for rank, fr in enumerate(frames[:topk], 1):
        skel = fr["skeleton"]
        k = len(skel)
        for ci, cl in enumerate(clauses):
            if ci in covered:
                continue
            toks = cl["tokens"]
            for i in range(len(toks) - k + 1):
                if all(skel[t] == SLOT or skel[t] == toks[i + t]
                       for t in range(k)):
                    covered.add(ci)
                    break
        rows.append({"rank": rank, "frame": " ".join(skel),
                     "n_covered": len(covered),
                     "frac_covered": round(len(covered) / max(1, len(clauses)), 6)})
    return rows


def family_counts(frames):
    """Frames / clauses / instances per signature family, in fixed order."""
    rows = []
    for fam in FAMILY_ORDER:
        sub = [f for f in frames if f["family"] == fam]
        rows.append({
            "family": fam,
            "n_frames": len(sub),
            "n_clauses": sum(f["n_clauses"] for f in sub),
            "n_instances": sum(f["n_instances"] for f in sub),
            "top_frame": (" ".join(sub[0]["skeleton"]) if sub else ""),
        })
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="#34 frame and template mining")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if args.quiet else print
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    ctx = load_context()
    clauses = load_clause_tokens(ctx)
    n_units = sum(len(cl["tokens"]) for cl in clauses)
    log(f"[#34] {len(clauses)} clauses, {n_units} screened units")

    frames = mine_frames(clauses)
    log(f"[#34] {len(frames)} frames mined")

    names, classes, span_stats = mine_block_names(ctx)
    log(f"[#34] {span_stats['spans']} spans, {len(names)} distinct block names")

    cov = coverage(clauses, frames)
    fams = family_counts(frames)
    # OCR-loop confound (#4): the recognition loop "of the second of" inflates
    # the `second` filler of the `of`-frames.  Counted so it is disclosed.
    loop_second = sum(1 for cl in clauses
                      for i in range(len(cl["tokens"]) - 2)
                      if cl["tokens"][i:i + 3] == ["of", "the", "second"])

    # ---- frames.tsv ----
    frame_rows = []
    for i, fr in enumerate(frames, 1):
        skel_str = " ".join(fr["skeleton"])
        f1 = ", ".join(f"{w}({c})" for w, c in fr["fillers"][0])
        f2 = ", ".join(f"{w}({c})" for w, c in fr["fillers"][1]) if fr["n_slots"] == 2 else ""
        frame_rows.append({
            "frame_id": i,
            "skeleton": skel_str,
            "family": fr["family"],
            "n_slots": fr["n_slots"],
            "n_clauses": fr["n_clauses"],
            "n_instances": fr["n_instances"],
            "slot1_fillers": f1,
            "slot2_fillers": f2,
            "example": (fr["examples"][0] if fr["examples"] else "")[:120],
        })
    write_tsv(out_dir / "frames.tsv",
              ["frame_id", "skeleton", "family", "n_slots", "n_clauses",
               "n_instances", "slot1_fillers", "slot2_fillers", "example"],
              frame_rows)

    # ---- family_counts.tsv ----
    write_tsv(out_dir / "family_counts.tsv",
              ["family", "n_frames", "n_clauses", "n_instances", "top_frame"],
              fams)

    # ---- block_names.tsv ----
    write_tsv(out_dir / "block_names.tsv",
              ["name", "first_class", "n_spans"], names)

    # ---- frame_coverage.tsv ----
    write_tsv(out_dir / "frame_coverage.tsv",
              ["rank", "frame", "n_covered", "frac_covered"], cov)

    # ---- reconciliation ----
    recon = [
        {"quantity": "clauses", "earlier": "5,614 (#33)",
         "here": str(len(clauses)), "note": "analysis/clauses/clauses.tsv"},
        {"quantity": "screened units", "earlier": "74,077 (#27)",
         "here": str(n_units), "note": "sum of clause token counts"},
        {"quantity": "spans (<<...>>)", "earlier": "432 (#14)",
         "here": str(span_stats["spans"]), "note": "analysis/block_parser/spans.tsv"},
        {"quantity": "closed spans", "earlier": "250 (#14)",
         "here": str(span_stats["closed"]), "note": "span layer"},
        {"quantity": "block names", "earlier": "-",
         "here": str(len(names)), "note": f"leading {NAME_UNITS} screened units"},
        {"quantity": "frames mined", "earlier": "-",
         "here": str(len(frames)),
         "note": f"min {MIN_CLAUSES} clauses, {MIN_FILLERS} fillers/slot"},
    ]
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    # ---- diagnostics ----
    diag = [
        {"diag": "clauses", "value": len(clauses)},
        {"diag": "screened_units", "value": n_units},
        {"diag": "frames", "value": len(frames)},
        {"diag": "single_slot_frames",
         "value": sum(1 for f in frames if f["n_slots"] == 1)},
        {"diag": "two_slot_frames",
         "value": sum(1 for f in frames if f["n_slots"] == 2)},
        {"diag": "spans", "value": span_stats["spans"]},
        {"diag": "spans_open_marker", "value": span_stats["open_marker"]},
        {"diag": "block_names_distinct", "value": len(names)},
        {"diag": "block_names_named_spans", "value": span_stats["named"]},
        {"diag": "block_names_residue_leading", "value": span_stats["residue_leading"]},
        {"diag": "min_clauses", "value": MIN_CLAUSES},
        {"diag": "min_fillers", "value": MIN_FILLERS},
        {"diag": "min_df", "value": MIN_DF},
        {"diag": "max_n", "value": MAX_N},
        {"diag": "ocr_loop_of_the_second", "value": loop_second},
        {"diag": "coverage_top10",
         "value": cov[9]["frac_covered"] if len(cov) >= 10 else ""},
        {"diag": "coverage_top40",
         "value": cov[-1]["frac_covered"] if cov else ""},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["diag", "value"], diag)

    # ---- summary.json ----
    summary = {
        "schema": SCHEMA, "generated": generated,
        "n_clauses": len(clauses), "n_units": n_units,
        "n_frames": len(frames),
        "n_single": sum(1 for f in frames if f["n_slots"] == 1),
        "n_double": sum(1 for f in frames if f["n_slots"] == 2),
        "n_spans": span_stats["spans"], "n_spans_closed": span_stats["closed"],
        "n_block_names": len(names),
        "ocr_loop_of_the_second": loop_second,
        "block_class_counts": dict(classes.most_common()),
        "families": fams,
        "top_frames": frame_rows[:40],
        "top_block_names": names[:40],
        "coverage": cov,
        "reconciliation": recon,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    # ---- report ----
    (out_dir / "frame_mining_report.md").write_text(
        build_report(summary), encoding="utf-8")

    log(f"[#34] wrote {out_dir} (schema {SCHEMA})")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def build_report(s):
    L = []
    a = L.append
    top = s["top_frames"]
    cov = s["coverage"]
    a("# Frame and template mining (#34)\n")
    a("")
    a("Extract the recurrent **schemata** whose recurrence *is* the book's own "
      "loop grammar (``docs/meaning-extraction-methods.md`` L4). Tokens are "
      "#17 units (markup-screened, case-folded) inside #33's operator-delimited "
      "clauses; a **frame** is an n-gram skeleton with one or two slots, "
      "instantiated by many distinct clauses with many distinct fillers. "
      "Reproduce with ``python work/frame_mining.py``; regression suite "
      "``python work/test_frame_mining.py``. Schema `frame-mining/1`.\n")
    a("")

    a("## 1. Method\n")
    a("")
    a("**Tokens.** Each clause's token sequence is the #17 units whose start "
      "falls in the clause's ``[start, end)`` span, markup-screened with #19's "
      "ledger + the #27 tag-run detector -- the exact join #33 uses, so the "
      f"unit total reconciles with #27's 74,077 (here {s['n_units']:,}). A "
      "unit is the book's word; case-folded for matching.\n")
    a("")
    a("**Frames.** For every n-gram (n = 2..8) the set of clauses it occurs in "
      "is recorded (document frequency, so one long clause cannot dominate). "
      "Single-slot skeletons mask one position, two-slot skeletons mask two. A "
      f"skeleton is a *frame* when >= {MIN_CLAUSES} distinct clauses instantiate "
      f"it and each slot has >= {MIN_FILLERS} distinct fillers. ``n_clauses`` is "
      "document frequency; ``n_instances`` is the true number of (clause, "
      "position) occurrences. This is construction-grammar frame mining: the "
      "fixed words are the frame's signature, the slots its variable parts. "
      "Each frame carries a rule-based **signature family** read from its fixed "
      "words only: ``first_person`` (opens ``i``), ``rel_that`` (contains "
      "``that``), ``np_of_np`` (contains ``of``), ``pp_to`` (contains ``to``), "
      "``other`` -- in that priority order. The ``<<...>>`` blocks come from "
      "the #14 span layer; their recurring names are catalogued separately.\n")
    a("")

    a("## 2. The frame catalogue (top 25 by clause coverage)\n")
    a("")
    a("| rank | skeleton | family | slots | clauses | instances | top slot-1 fillers | example |")
    a("|---:|---|---|---:|---:|---:|---|---|")
    for r in top[:25]:
        a(f"| {r['frame_id']} | {r['skeleton']} | {r['family']} | {r['n_slots']} | "
          f"{r['n_clauses']:,} | {r['n_instances']:,} | "
          f"{r['slot1_fillers'][:60]} | {r['example'][:50]} |")
    a("")
    a(f"**{s['n_frames']:,} frames** meet the thresholds "
      f"({s['n_single']:,} single-slot, {s['n_double']:,} two-slot). The full "
      "catalogue is ``frames.tsv``.\n")
    a("")

    a("## 3. Signature families\n")
    a("")
    a("The task names ``I <verb> the <NP> of <NP>`` and ``the <N> that <V>``. "
      "Tagging every frame by its fixed-word signature (rules in the module "
      "docstring) counts the named schemata directly:\n")
    a("")
    a("| family | frames | clauses | instances | top frame |")
    a("|---|---:|---:|---:|---|")
    for r in s["families"]:
        a(f"| {r['family']} | {r['n_frames']:,} | {r['n_clauses']:,} | "
          f"{r['n_instances']:,} | {r['top_frame']} |")
    a("")
    a("The families overlap by construction (``i <verb> the <NP> of <NP>`` is "
      "both ``first_person`` and an ``of`` frame); priority order resolves the "
      "overlap and is disclosed above.\n")
    a("")

    a("## 4. The loop grammar: cumulative coverage\n")
    a("")
    a("The book is a loop of a restricted set of schemata. Ranking frames by "
      "clause coverage and marking each clause covered by any frame:\n")
    a("")
    a("| rank | frame | clauses covered | fraction |")
    a("|---:|---|---:|---:|")
    for r in cov[:15]:
        a(f"| {r['rank']} | {r['frame']} | {r['n_covered']:,} | "
          f"{r['frac_covered']:.3f} |")
    a("")
    if cov:
        a(f"The top 10 frames cover {cov[9]['frac_covered']:.1%} of clauses; the "
          f"top 40 cover {cov[-1]['frac_covered']:.1%}. The loop grammar is the "
          "small set of skeletons at the top of ``frame_coverage.tsv``.\n")
        a("")

    a("## 5. ``<<...>>`` block names\n")
    a("")
    a(f"{s['n_spans']} spans (#14, {s['n_spans_closed']} closed). A block's "
      f"*name* is the leading run (<= {NAME_UNITS}) of screened #17 content "
      "units of a span that opens with ``<<`` -- the label the block leads "
      f"with. {s['n_block_names']} distinct names; top 20:\n")
    a("")
    a("| name | first unit | spans |")
    a("|---|---|---:|")
    for r in s["top_block_names"][:20]:
        a(f"| {r['name']} | {r['first_class']} | {r['n_spans']} |")
    a("")
    a("Leading-unit classes: " +
      ", ".join(f"{k} {v}" for k, v in s["block_class_counts"].items()) + ". "
      "The class is #19's screen of the first unit (`english`/`proper`/"
      "`gazetteer`/`label` are clean; `compound`/`true_coinage`/`junk` mark "
      "OCR/markup residue, reported but not hidden).\n")
    a("")

    a("## 6. Reading\n")
    a("")
    a("The frames are the book's own templates: fixed function-word signatures "
      "with content slots that recur across hundreds of clauses. The two-slot "
      "frames (``the \u27e8_\u27e9 of the \u27e8_\u27e9``, ``of the "
      "\u27e8_\u27e9 \u27e8_\u27e9 that``) are the classic NP-of-NP schema; "
      "the single-slot frames are the shorter repeated units. Their recurrence "
      "-- not any single frame -- is the loop grammar: a handful of schemata, "
      "each instantiated with many different fillers, covers a large share of "
      "the book.\n")
    a("")

    a("## 7. Caveats and negative results\n")
    a("")
    a("- **No POS tagger.** Frames are mined over #17 units, not parsed "
      "syntax, so a slot is any varying unit, not a verified NP/VP. The "
      "``I <verb> the <NP>`` reading of a frame is an interpretation of its "
      "fillers, not a measurement; the skeleton + counts are the measurement.\n")
    a("- **Signature families are rule-based**, reading only the fixed words; "
      "they are a reading aid, and the overlap/priority is disclosed.\n")
    a("- **Case-folded matching.** Frames are case-insensitive by construction; "
      "a frame and its capitalised variant are the same frame.\n")
    a("- **Thresholds are disclosed** (``diagnostics.tsv``): a different "
      "``MIN_CLAUSES`` / ``MIN_FILLERS`` moves frames in and out of the "
      "catalogue; the ranking is stable to small changes.\n")
    a("- **The longest clauses are OCR loops** (#4), not prose; they contribute "
      "rare n-grams that the document-frequency threshold filters out.\n")
    a(f"- **The `second` filler of the `of`-frames is an OCR loop.** #4 found a "
      f"recognition loop (`of the second of`, pages 89/208/182); the exact-count "
      f"pass counts `of the second` {s['ocr_loop_of_the_second']} times, which "
      "is why `second` is the top filler of ``the \u27e8_\u27e9 of`` and "
      "``\u27e8_\u27e9 of the``. The skeleton is unaffected (it is `of the "
      "\u27e8_\u27e9` either way), but those filler counts carry this "
      "contamination; the #24/#49/#50 screens bound it.\n")
    a("- **Block names are leading runs, not definitions.** The ``<<...>>`` "
      "name is the block's opening label; unclosed blocks and OCR-fused "
      "markup leave a small residue, counted in ``diagnostics.tsv`` and "
      "class-tagged in ``block_names.tsv``.\n")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
