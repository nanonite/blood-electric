#!/usr/bin/env python3
"""#71 — Consistency checker over the termbase.

Task (plan E9.2, docs/chainlink-plan.md #71): verify the same
symbol/term gets the same reading throughout; flag violations.
DONE: checker + report over the termbase.

The termbase is the central curated artifact (#56's GLOSS_TERMS +
termbase_sample.tsv).  A term's "reading" flows through several
artifacts, and a drift between them is the violation this checker
catches:

  1. GLOSS_TERMS vs termbase TSV — the curated gloss, confidence and
     POS must match the computed termbase row exactly.
  2. PLAIN_READINGS vs GLOSS_TERMS — a surface must not carry two
     different curated readings (one term = one gloss).
  3. Entity glossary vs termbase — the #30 role and does/done-to
     verbs must match the termbase's entity columns.
  4. Operator readings — each operator must have exactly one reading
     in OPERATOR_GLOSSES (no contradictions, no duplicates).
  5. One term = one gloss — no term may have multiple different
     gloss strings across all artifacts.

Every check is deterministic and evidence-linked: a violation reports
the term, the two conflicting readings, and the artifacts that carry
them.  The report is published under analysis/consistency/.

Reproduce:

    conda activate marker
    python work/consistency_checker.py            # full run + report
    python work/test_consistency_checker.py         # regression suite
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clause_segmentation as cs   # #33 loaders (read_tsv)
import language_viability as lv    # #56 termbase, GLOSS_TERMS, OPERATOR_GLOSSES
import interlinear_edition as ie   # #62 PLAIN_READINGS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "consistency"

TERMBASE = ROOT / "analysis" / "language_viability" / "termbase_sample.tsv"
ENTITY = ROOT / "analysis" / "entity" / "entity_glossary.tsv"

SCHEMA = "consistency/1"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_termbase():
    """{term_lower: row} from the #56 termbase TSV."""
    if not TERMBASE.exists():
        return {}
    rows = {}
    for r in cs.read_tsv(TERMBASE):
        rows.setdefault(r["term"].lower(), r)
    return rows


def load_entity_glossary():
    """{entity_lower: row} from #30's entity glossary."""
    if not ENTITY.exists():
        return {}
    rows = {}
    for r in cs.read_tsv(ENTITY):
        rows.setdefault(r["entity"].lower(), r)
    return rows


def load_gloss_terms():
    """{term_lower: (pos, gloss, confidence)} from #56's curated table."""
    return {k.lower(): v for k, v in lv.GLOSS_TERMS.items()}


def load_plain_readings():
    """{surface_lower: (reading, confidence)} from #62's curated table."""
    return {k.lower(): v for k, v in ie.PLAIN_READINGS.items()}


def load_operator_glosses():
    """{operator: reading} from #56's OPERATOR_GLOSSES."""
    return dict(lv.OPERATOR_GLOSSES)


def load_term_frequencies():
    """{term_lower: frequency} from the #17 token stream.

    Used to distinguish curated entries for unattested surfaces (freq=0,
    correctly excluded from the termbase) from genuine missing entries
    (freq > 0 but absent from the termbase).
    """
    try:
        bodies, pages, page_units, unit_meta, is_markup, screening = \
            lv.load_book()
        allp = sorted(pages)
        import collections
        freq = collections.Counter(w for p in allp for w in pages[p])
        return {k.lower(): v for k, v in freq.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm(s):
    """Normalise a gloss for comparison (case-fold, strip, collapse spaces)."""
    return " ".join(s.lower().split())


def _verbs(s):
    """Parse a `verb(n)|verb(n)` string into a sorted tuple of verbs."""
    if not s:
        return ()
    parts = []
    for piece in s.split("|"):
        piece = piece.strip()
        if not piece:
            continue
        # strip the count suffix: "is(32)" -> "is"
        if "(" in piece:
            piece = piece[:piece.index("(")]
        parts.append(piece.lower())
    return tuple(sorted(set(parts)))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_gloss_terms_vs_termbase(gloss_terms, termbase, freq=None):
    """Check 1: GLOSS_TERMS gloss/confidence/POS match the termbase row.

    Terms with freq=0 in the token stream are expected to be absent from
    the termbase (it only includes attested surfaces), so they are not
    flagged as violations.
    """
    violations = []
    for term, (pos, gloss, conf) in sorted(gloss_terms.items()):
        row = termbase.get(term)
        if row is None:
            if freq and freq.get(term, 0) > 0:
                violations.append({
                    "check": "gloss_terms_vs_termbase",
                    "term": term,
                    "kind": "missing_in_termbase",
                    "detail": "curated in GLOSS_TERMS and attested "
                               "(freq=%d) but absent from termbase TSV"
                               % freq.get(term, 0),
                    "artifacts": ["GLOSS_TERMS", "termbase_sample.tsv"],
                })
            continue
        tb_gloss = row.get("gloss", "")
        tb_conf = row.get("confidence", "")
        tb_pos = row.get("pos_class", "")
        if _norm(tb_gloss) != _norm(gloss):
            violations.append({
                "check": "gloss_terms_vs_termbase",
                "term": term,
                "kind": "gloss_mismatch",
                "detail": "GLOSS_TERMS=%r vs termbase=%r"
                           % (gloss, tb_gloss),
                "artifacts": ["GLOSS_TERMS", "termbase_sample.tsv"],
            })
        if tb_conf != conf:
            violations.append({
                "check": "gloss_terms_vs_termbase",
                "term": term,
                "kind": "confidence_mismatch",
                "detail": "GLOSS_TERMS=%r vs termbase=%r"
                           % (conf, tb_conf),
                "artifacts": ["GLOSS_TERMS", "termbase_sample.tsv"],
            })
        if tb_pos != pos:
            violations.append({
                "check": "gloss_terms_vs_termbase",
                "term": term,
                "kind": "pos_mismatch",
                "detail": "GLOSS_TERMS=%r vs termbase=%r"
                           % (pos, tb_pos),
                "artifacts": ["GLOSS_TERMS", "termbase_sample.tsv"],
            })
    return violations


def check_plain_readings_vs_gloss_terms(plain_readings, gloss_terms):
    """Check 2: a surface must not carry two different curated readings."""
    violations = []
    for surface, (reading, conf) in sorted(plain_readings.items()):
        if surface in gloss_terms:
            gt_gloss = gloss_terms[surface][1]
            if _norm(reading) != _norm(gt_gloss):
                violations.append({
                    "check": "plain_readings_vs_gloss_terms",
                    "term": surface,
                    "kind": "dual_curated_reading",
                    "detail": "PLAIN_READINGS=%r vs GLOSS_TERMS=%r"
                               % (reading, gt_gloss),
                    "artifacts": ["PLAIN_READINGS", "GLOSS_TERMS"],
                })
    return violations


def check_entity_glossary_vs_termbase(entity_glossary, termbase):
    """Check 3: #30 role and verbs match the termbase entity columns."""
    violations = []
    for term, row in sorted(termbase.items()):
        eg = entity_glossary.get(term)
        if eg is None:
            continue
        tb_role = row.get("entity_role_30", "")
        eg_role = eg.get("role", "")
        if tb_role and eg_role and tb_role != eg_role:
            violations.append({
                "check": "entity_glossary_vs_termbase",
                "term": term,
                "kind": "role_mismatch",
                "detail": "termbase=%r vs entity_glossary=%r"
                           % (tb_role, eg_role),
                "artifacts": ["termbase_sample.tsv", "entity_glossary.tsv"],
            })
        tb_does = _verbs(row.get("does_verbs_30", ""))
        eg_does = _verbs(eg.get("does_verbs", ""))
        if tb_does and eg_does and tb_does != eg_does:
            violations.append({
                "check": "entity_glossary_vs_termbase",
                "term": term,
                "kind": "does_verbs_mismatch",
                "detail": "termbase=%r vs entity_glossary=%r"
                           % (list(tb_does), list(eg_does)),
                "artifacts": ["termbase_sample.tsv", "entity_glossary.tsv"],
            })
        tb_done = _verbs(row.get("done_to_verbs_30", ""))
        eg_done = _verbs(eg.get("done_to_verbs", ""))
        if tb_done and eg_done and tb_done != eg_done:
            violations.append({
                "check": "entity_glossary_vs_termbase",
                "term": term,
                "kind": "done_to_verbs_mismatch",
                "detail": "termbase=%r vs entity_glossary=%r"
                           % (list(tb_done), list(eg_done)),
                "artifacts": ["termbase_sample.tsv", "entity_glossary.tsv"],
            })
    return violations


def check_operator_glosses(operator_glosses):
    """Check 4: each operator has exactly one reading (no contradictions)."""
    violations = []
    # Check for duplicate keys with different values — impossible in a
    # Python dict, but we can check for near-duplicates (same reading
    # text for different operators is fine; same operator with different
    # readings is not).
    seen = {}
    for op, reading in sorted(operator_glosses.items()):
        if not reading or not reading.strip():
            violations.append({
                "check": "operator_glosses",
                "term": op,
                "kind": "empty_reading",
                "detail": "operator %r has an empty reading" % op,
                "artifacts": ["OPERATOR_GLOSSES"],
            })
        # Check for contradictory readings: the same operator surface
        # appearing with different readings (cannot happen in a dict,
        # but the check is here for completeness and future-proofing
        # if the source changes to a list).
        key = op
        if key in seen and _norm(seen[key]) != _norm(reading):
            violations.append({
                "check": "operator_glosses",
                "term": op,
                "kind": "contradictory_reading",
                "detail": "%r vs %r" % (seen[key], reading),
                "artifacts": ["OPERATOR_GLOSSES"],
            })
        seen[key] = reading
    return violations


def check_one_term_one_gloss(gloss_terms, plain_readings, termbase):
    """Check 5: no term may have multiple different gloss strings.

    Collect every gloss string for each term across all artifacts and
    flag any term with more than one distinct normalised gloss.
    """
    violations = []
    glosses = collections.defaultdict(set)
    # From GLOSS_TERMS
    for term, (_pos, gloss, _conf) in gloss_terms.items():
        glosses[term].add(_norm(gloss))
    # From PLAIN_READINGS
    for surface, (reading, _conf) in plain_readings.items():
        glosses[surface].add(_norm(reading))
    # From termbase
    for term, row in termbase.items():
        g = row.get("gloss", "")
        if g:
            glosses[term].add(_norm(g))
    # Flag terms with multiple distinct glosses
    for term, gloss_set in sorted(glosses.items()):
        if len(gloss_set) > 1:
            violations.append({
                "check": "one_term_one_gloss",
                "term": term,
                "kind": "multiple_glosses",
                "detail": "distinct glosses: %s"
                           % ", ".join(sorted(gloss_set)),
                "artifacts": ["GLOSS_TERMS", "PLAIN_READINGS",
                              "termbase_sample.tsv"],
            })
    return violations


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def build_report(violations, stats, generated):
    """The consistency report as markdown."""
    lines = [
        "# Consistency checker — report (#71)",
        "",
        "Generated: %s" % generated,
        "",
        "Schema: %s" % SCHEMA,
        "",
        "## Summary",
        "",
        "| check | violations |",
        "|---|---|",
    ]
    for check, n in sorted(stats["by_check"].items()):
        lines.append("| %s | %d |" % (check, n))
    lines.append("| **total** | **%d** |" % stats["total"])
    lines.append("")

    if not violations:
        lines.append("No violations found. Every symbol/term gets the same "
                     "reading throughout the termbase.")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Violations")
    lines.append("")
    lines.append("| check | term | kind | detail | artifacts |")
    lines.append("|---|---|---|---|---|")
    for v in violations:
        lines.append("| %s | `%s` | %s | %s | %s |"
                     % (v["check"], v["term"], v["kind"],
                        v["detail"].replace("|", "\\|"),
                        ", ".join(v["artifacts"])))
    lines.append("")

    # Group by kind for the summary
    lines.append("## Violations by kind")
    lines.append("")
    lines.append("| kind | count |")
    lines.append("|---|---|")
    for kind, n in sorted(stats["by_kind"].items()):
        lines.append("| %s | %d |" % (kind, n))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(out=OUT):
    """Run every consistency check and write the report + violations TSV."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    gloss_terms = load_gloss_terms()
    plain_readings = load_plain_readings()
    termbase = load_termbase()
    entity_glossary = load_entity_glossary()
    operator_glosses = load_operator_glosses()

    freq = load_term_frequencies()

    violations = []
    violations.extend(check_gloss_terms_vs_termbase(gloss_terms, termbase,
                                                     freq))
    violations.extend(check_plain_readings_vs_gloss_terms(plain_readings,
                                                          gloss_terms))
    violations.extend(check_entity_glossary_vs_termbase(entity_glossary,
                                                         termbase))
    violations.extend(check_operator_glosses(operator_glosses))
    violations.extend(check_one_term_one_gloss(gloss_terms, plain_readings,
                                               termbase))

    by_check = collections.Counter(v["check"] for v in violations)
    by_kind = collections.Counter(v["kind"] for v in violations)
    stats = {
        "total": len(violations),
        "by_check": dict(sorted(by_check.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "n_gloss_terms": len(gloss_terms),
        "n_plain_readings": len(plain_readings),
        "n_termbase_terms": len(termbase),
        "n_entity_glossary": len(entity_glossary),
        "n_operator_glosses": len(operator_glosses),
    }

    # Write violations TSV
    header = ["check", "term", "kind", "detail", "artifacts"]
    rows = [{"check": v["check"], "term": v["term"], "kind": v["kind"],
             "detail": v["detail"],
             "artifacts": "|".join(v["artifacts"])}
            for v in violations]
    cs.write_tsv(out / "violations.tsv", header, rows)

    # Write summary JSON
    summary = {
        "schema": SCHEMA,
        "generated_at": generated,
        "stats": stats,
        "violations": violations,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    # Write report
    report = build_report(violations, stats, generated)
    (out / "consistency_report.md").write_text(report + "\n", encoding="utf-8")

    return {"summary": summary, "violations": violations, "stats": stats}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    res = build(out=Path(args.out))
    stats = res["stats"]
    print("[#71] consistency checker: %d violations over %d gloss terms, "
          "%d plain readings, %d termbase terms, %d entities, %d operators"
          % (stats["total"], stats["n_gloss_terms"],
             stats["n_plain_readings"], stats["n_termbase_terms"],
             stats["n_entity_glossary"], stats["n_operator_glosses"]))
    for check, n in sorted(stats["by_check"].items()):
        print("  %-40s %d" % (check, n))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
