#!/usr/bin/env python3
"""#76 — Anti-pattern guardrails audit -> analysis/guardrails/

Task (plan E9.6, docs/chainlink-plan.md #76): audit for hallucinated
coherence, LLM circularity (same model translating and judging),
over-correction, single-method certainty, and residue-dropping.
DONE: guardrail checklist + audit results, wired into CI-style checks.

The checklist itself lives in docs/guardrails.md, written for a human
reader.  This module is the mechanical half: one check function per
anti-pattern, each reading the *published* artifact of the workspace
module that is supposed to guard against it (never re-deriving the
module's own numbers, so a drift between this audit and the module it
audits is a build error here, not a silent disagreement):

  hallucinated_coherence   -> analysis/triangulation/summary.json (#73)
                              fails if the convergence report ever
                              stops disclosing divergence/partial rows
                              alongside its convergent ones.
  llm_circularity           -> work/glossing_pipeline.py (#63) and
                              work/translation_committee.py (#64)
                              source: the functions that compute a
                              verdict must not themselves call a model
                              backend, parsed via `ast` rather than a
                              plain substring scan of the whole file so
                              a docstring mentioning "opencode" in an
                              unrelated function cannot trip it.
  over_correction           -> analysis/lexical_correction/{corrections,
                              unchanged_review}.tsv (#6): fails if the
                              lexical corrector ever stops holding back
                              a REVIEW class and "corrects" every OOV
                              token it sees.
  single_method_certainty   -> analysis/triangulation/summary.json (#73)
                              `methods`: fails if a convergence claim is
                              ever backed by fewer than 3 independent
                              methods.
  residue_dropping          -> analysis/coverage_ledger/summary.json
                              (#70): fails if residue is ever claimed
                              away (zero rows) instead of itemized, or
                              if any level reports unexplained residue
                              without it being surfaced here.

Each check takes the artifact root(s) as a parameter rather than
reaching for a module-level constant, so the regression suite can point
every check at a hand-built fixture and prove the failure path has
teeth, not just that today's repo happens to pass.

Reproduce:

    conda activate marker
    python work/guardrail_audit.py            # full run + report
    python work/test_guardrail_audit.py        # regression suite
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
ANALYSIS = ROOT / "analysis"
OUT = ANALYSIS / "guardrails"

SCHEMA = "guardrails/1"

# Tokens that mean "this function reaches out to a model backend" —
# checked against the *source of the verdict-computing function only*,
# never the whole file, so a loader elsewhere that legitimately talks
# to opencode does not make the verdict function circular.
BACKEND_TOKENS = (
    "opencode", "subprocess", "requests.", "urllib", "http://", "https://",
    "resolve_glosses(", "backend(", "call_model", "generator(",
)


class GuardrailResult(NamedTuple):
    id: str
    name: str
    anti_pattern: str
    passed: bool
    evidence: dict
    detail: str


# ---------------------------------------------------------------------------
# guardrail 1 — hallucinated coherence
# ---------------------------------------------------------------------------

def check_hallucinated_coherence(analysis_root: Path = ANALYSIS) -> GuardrailResult:
    """A convergence report that only ever reports convergence has not
    been checked against disagreeing evidence — it has asserted
    coherence rather than found it.  #73's triangulation report is the
    workspace's one convergence claim across independent methods; this
    passes only while it keeps disclosing divergent/partial rows next
    to its convergent ones."""
    gid, name = "hallucinated_coherence", "#73 triangulation discloses divergence"
    path = analysis_root / "triangulation" / "summary.json"
    if not path.exists():
        return GuardrailResult(gid, name, "hallucinated coherence", False, {},
                                "missing %s" % path)
    summary = json.loads(path.read_text())
    counts = summary.get("verdict_counts", {}).get("all", {})
    convergent = counts.get("convergent", 0)
    divergent = counts.get("divergent", 0)
    partial = counts.get("partial", 0)
    untestable = counts.get("untestable", 0)
    total = convergent + divergent + partial + untestable
    disagreement = divergent + partial
    passed = total > 0 and disagreement > 0 and convergent < total
    detail = ("%d/%d rows disagree (divergent+partial); %d/%d fully convergent"
               % (disagreement, total, convergent, total))
    return GuardrailResult(gid, name, "hallucinated coherence", passed,
                            {"verdict_counts": counts, "total_rows": total}, detail)


# ---------------------------------------------------------------------------
# guardrail 2 — LLM circularity
# ---------------------------------------------------------------------------

def _function_source(module_source: str, func_name: str) -> str | None:
    """Return the source text of the top-level function `func_name` in
    `module_source`, or None if it is not defined there."""
    tree = ast.parse(module_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(module_source, node)
    return None


# Which functions compute a verdict/judgement in each audited module —
# these are exactly the functions a circular design would have call
# back into the model that produced the thing they are judging.
VERDICT_FUNCTIONS = {
    "glossing_pipeline.py": (
        "check_operator_fidelity", "check_entity_fidelity",
        "check_residue_discipline", "check_frame_preservation", "check_gloss",
    ),
    "translation_committee.py": (
        "reading_facts", "operator_compliance", "channel_compliance",
        "classify_disagreement", "compare_readings",
    ),
}


def check_llm_circularity(work_root: Path = WORK,
                           targets: dict[str, tuple] = None) -> GuardrailResult:
    """The same model must not translate and then judge its own
    translation.  #63's four deterministic checks and #64's critic/
    comparer are supposed to be pure functions of frozen data; this
    parses their source with `ast` and fails if any of them contains a
    token that would reach a model backend or re-invoke generation."""
    gid, name = "llm_circularity", "#63/#64 verdict functions call no model backend"
    targets = targets if targets is not None else VERDICT_FUNCTIONS
    violations = []
    checked = []
    for fname, funcs in targets.items():
        path = work_root / fname
        if not path.exists():
            violations.append("%s: file missing" % fname)
            continue
        source = path.read_text()
        for fn in funcs:
            seg = _function_source(source, fn)
            if seg is None:
                violations.append("%s:%s not found" % (fname, fn))
                continue
            checked.append("%s:%s" % (fname, fn))
            lowered = seg.lower()
            for token in BACKEND_TOKENS:
                if token.lower() in lowered:
                    violations.append("%s:%s contains forbidden token %r"
                                       % (fname, fn, token))
    passed = len(checked) > 0 and not violations
    detail = ("%d verdict function(s) checked, %d violation(s)"
               % (len(checked), len(violations)))
    return GuardrailResult(gid, name, "LLM circularity", passed,
                            {"checked": checked, "violations": violations}, detail)


# ---------------------------------------------------------------------------
# guardrail 3 — over-correction
# ---------------------------------------------------------------------------

def _count_data_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return max(sum(1 for _ in fh) - 1, 0)  # minus header


def check_over_correction(analysis_root: Path = ANALYSIS) -> GuardrailResult:
    """Forcing a confident reading onto data that is genuinely
    ambiguous is over-correction.  #6's lexical corrector is supposed
    to hold uncertain OOV tokens in a REVIEW class rather than
    "fixing" them all; this fails the moment that class goes empty,
    i.e. the corrector starts correcting everything it sees."""
    gid, name = "over_correction", "#6 lexical corrector holds back a non-empty REVIEW class"
    corrections = analysis_root / "lexical_correction" / "corrections.tsv"
    review = analysis_root / "lexical_correction" / "unchanged_review.tsv"
    if not corrections.exists() or not review.exists():
        return GuardrailResult(gid, name, "over-correction", False, {},
                                "missing corrections.tsv or unchanged_review.tsv")
    n_corrections = _count_data_rows(corrections)
    n_review = _count_data_rows(review)
    passed = n_review > 0
    detail = "%d corrected, %d held back for review" % (n_corrections, n_review)
    return GuardrailResult(gid, name, "over-correction", passed,
                            {"n_corrections": n_corrections, "n_review": n_review}, detail)


# ---------------------------------------------------------------------------
# guardrail 4 — single-method certainty
# ---------------------------------------------------------------------------

def check_single_method_certainty(analysis_root: Path = ANALYSIS,
                                   minimum_methods: int = 3) -> GuardrailResult:
    """A claim backed by one method is a method's opinion, not a
    finding.  #73's triangulation report is the workspace's convergence
    claim; this fails if it is ever backed by fewer than three
    independent methods."""
    gid, name = "single_method_certainty", "#73 triangulation uses >= %d independent methods" % minimum_methods
    path = analysis_root / "triangulation" / "summary.json"
    if not path.exists():
        return GuardrailResult(gid, name, "single-method certainty", False, {},
                                "missing %s" % path)
    summary = json.loads(path.read_text())
    methods = summary.get("methods", [])
    n = len(methods)
    passed = n >= minimum_methods
    detail = "%d method(s): %s" % (n, ", ".join(m.get("key", "?") for m in methods))
    return GuardrailResult(gid, name, "single-method certainty", passed,
                            {"n_methods": n, "methods": [m.get("key") for m in methods]},
                            detail)


# ---------------------------------------------------------------------------
# guardrail 5 — residue-dropping
# ---------------------------------------------------------------------------

def check_residue_dropping(analysis_root: Path = ANALYSIS) -> GuardrailResult:
    """Residue that is dropped silently never shows up as a number
    anyone can question.  #70's coverage ledger is the workspace's
    closure test; this fails if residue is ever claimed away (zero
    itemized rows) instead of disclosed, or if any level reports
    unexplained residue that is not itself surfaced here."""
    gid, name = "residue_dropping", "#70 coverage ledger itemizes residue and reports zero unexplained"
    path = analysis_root / "coverage_ledger" / "summary.json"
    if not path.exists():
        return GuardrailResult(gid, name, "residue-dropping", False, {},
                                "missing %s" % path)
    summary = json.loads(path.read_text())
    residue = summary.get("residue", {})
    unexplained = summary.get("unexplained", {})
    n_rows = residue.get("n_rows", 0)
    all_tracked = bool(unexplained) and all(v == 0 for v in unexplained.values())
    passed = n_rows > 0 and all_tracked
    detail = ("%d residue row(s) itemized; unexplained by level: %s"
               % (n_rows, unexplained))
    return GuardrailResult(gid, name, "residue-dropping", passed,
                            {"n_rows": n_rows, "unexplained": unexplained}, detail)


CHECKS = (
    check_hallucinated_coherence,
    check_llm_circularity,
    check_over_correction,
    check_single_method_certainty,
    check_residue_dropping,
)


def run_audit(analysis_root: Path = ANALYSIS, work_root: Path = WORK):
    results = []
    for fn in CHECKS:
        if fn is check_llm_circularity:
            results.append(fn(work_root=work_root))
        else:
            results.append(fn(analysis_root=analysis_root))
    return results


def build_report(results, generated: str) -> str:
    lines = [
        "# #76 Anti-pattern guardrails audit",
        "",
        "Generated: %s" % generated,
        "",
        "Checklist in `docs/guardrails.md`; this report is its mechanical",
        "verdict, computed from the published artifact of the workspace",
        "module each guardrail audits — never re-derived.",
        "",
        "| guardrail | anti-pattern | verdict | detail |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append("| %s | %s | %s | %s |" % (
            r.name, r.anti_pattern, "PASS" if r.passed else "FAIL", r.detail))
    lines.append("")
    n_pass = sum(1 for r in results if r.passed)
    lines.append("**%d/%d guardrails pass.**" % (n_pass, len(results)))
    lines.append("")
    for r in results:
        lines.append("## %s" % r.name)
        lines.append("")
        lines.append("Anti-pattern: *%s*. Verdict: **%s**." % (
            r.anti_pattern, "PASS" if r.passed else "FAIL"))
        lines.append("")
        lines.append(r.detail)
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r.evidence, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def build(out: Path = OUT, analysis_root: Path = ANALYSIS, work_root: Path = WORK):
    out.mkdir(parents=True, exist_ok=True)
    results = run_audit(analysis_root=analysis_root, work_root=work_root)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = ["id", "name", "anti_pattern", "passed", "detail"]
    with (out / "findings.tsv").open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in results:
            fh.write("\t".join([
                r.id, r.name, r.anti_pattern, "pass" if r.passed else "fail", r.detail,
            ]) + "\n")

    summary = {
        "schema": SCHEMA,
        "generated_at": generated,
        "all_passed": all(r.passed for r in results),
        "n_passed": sum(1 for r in results if r.passed),
        "n_total": len(results),
        "guardrails": [
            {
                "id": r.id,
                "name": r.name,
                "anti_pattern": r.anti_pattern,
                "passed": r.passed,
                "evidence": r.evidence,
                "detail": r.detail,
            }
            for r in results
        ],
    }
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    report = build_report(results, generated)
    (out / "audit_report.md").write_text(report + "\n", encoding="utf-8")

    return {"summary": summary, "results": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    res = build(out=Path(args.out))
    summary = res["summary"]
    print("[#76] guardrail audit: %d/%d guardrails pass"
          % (summary["n_passed"], summary["n_total"]))
    for g in summary["guardrails"]:
        print("  %-28s %-6s %s" % (g["id"], "PASS" if g["passed"] else "FAIL", g["detail"]))
    print("wrote %s" % args.out)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
