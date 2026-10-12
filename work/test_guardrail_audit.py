#!/usr/bin/env python3
"""Regression tests for the anti-pattern guardrails audit (#76).

Run from the workspace root (inside the `marker` conda env):

    python work/test_guardrail_audit.py
    python -m unittest discover -s work -p 'test_*.py'

Two tiers, matching the workspace's other suites:

* **Synthetic tests** build a fixture `analysis/`/`work/` tree by hand
  for each of the five guardrails and prove the failure path has
  teeth: an all-convergent triangulation summary must fail
  hallucinated_coherence, a verdict function that calls out to a model
  backend must fail llm_circularity, an empty REVIEW class must fail
  over_correction, two methods must fail single_method_certainty, and
  unexplained residue (or a residue-free ledger) must fail
  residue_dropping.
* **Artifact tests** run the real audit against this workspace's own
  published artifacts and pin today's result: all five guardrails
  pass. If a future change to #63/#64/#70/#71/#73 regresses one of
  them, this is the test that is supposed to fail.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import guardrail_audit as ga

HAVE_ARTIFACTS = all(p.exists() for p in (
    Path(ROOT) / "analysis" / "triangulation" / "summary.json",
    Path(ROOT) / "analysis" / "lexical_correction" / "corrections.tsv",
    Path(ROOT) / "analysis" / "coverage_ledger" / "summary.json",
    Path(HERE) / "glossing_pipeline.py",
    Path(HERE) / "translation_committee.py",
))


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(c) for c in row) + "\n")


def _triangulation_summary(verdict_counts, methods):
    return {
        "verdict_counts": {"all": verdict_counts},
        "methods": [{"key": m} for m in methods],
    }


# ---------------------------------------------------------------------------
# synthetic tests: hallucinated coherence
# ---------------------------------------------------------------------------

class TestHallucinatedCoherence(unittest.TestCase):

    def test_all_convergent_fails(self):
        """A report claiming every motif converges, with no disagreeing
        row anywhere, is the anti-pattern itself."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "triangulation" / "summary.json", _triangulation_summary(
                {"convergent": 10, "divergent": 0, "partial": 0, "untestable": 0},
                ["net", "lda", "nmf"]))
            result = ga.check_hallucinated_coherence(analysis_root=root)
        self.assertFalse(result.passed)

    def test_mixed_verdicts_passes(self):
        """Disclosing some divergence alongside convergence is honest
        reporting, not hallucinated coherence."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "triangulation" / "summary.json", _triangulation_summary(
                {"convergent": 3, "divergent": 5, "partial": 2, "untestable": 0},
                ["net", "lda", "nmf"]))
            result = ga.check_hallucinated_coherence(analysis_root=root)
        self.assertTrue(result.passed)

    def test_missing_artifact_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ga.check_hallucinated_coherence(analysis_root=Path(tmp))
        self.assertFalse(result.passed)

    def test_all_divergent_with_no_convergence_still_passes(self):
        """A report that finds zero convergence anywhere is not
        hallucinating coherence either — it is this check's job only to
        catch the opposite failure."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "triangulation" / "summary.json", _triangulation_summary(
                {"convergent": 0, "divergent": 10, "partial": 0, "untestable": 0},
                ["net", "lda", "nmf"]))
            result = ga.check_hallucinated_coherence(analysis_root=root)
        self.assertTrue(result.passed)


# ---------------------------------------------------------------------------
# synthetic tests: LLM circularity
# ---------------------------------------------------------------------------

class TestLLMCircularity(unittest.TestCase):

    def test_clean_function_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mod.py").write_text(
                "def judge(a, b):\n"
                "    return a == b\n",
                encoding="utf-8")
            result = ga.check_llm_circularity(
                work_root=root, targets={"mod.py": ("judge",)})
        self.assertTrue(result.passed)

    def test_model_call_in_verdict_function_fails(self):
        """A 'deterministic' check that quietly shells out to the model
        it is supposed to be grading is exactly the circularity this
        guardrail exists to catch."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mod.py").write_text(
                "def judge(a, b):\n"
                "    verdict = opencode.run(a)\n"
                "    return verdict == b\n",
                encoding="utf-8")
            result = ga.check_llm_circularity(
                work_root=root, targets={"mod.py": ("judge",)})
        self.assertFalse(result.passed)
        self.assertIn("opencode", result.evidence["violations"][0].lower())

    def test_model_token_outside_target_function_does_not_trip_check(self):
        """A loader elsewhere in the same file that legitimately talks
        to opencode must not fail the audit of an unrelated, clean
        verdict function — the scan is per-function, not per-file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mod.py").write_text(
                "def fetch():\n"
                "    return opencode.run('x')\n"
                "\n"
                "def judge(a, b):\n"
                "    return a == b\n",
                encoding="utf-8")
            result = ga.check_llm_circularity(
                work_root=root, targets={"mod.py": ("judge",)})
        self.assertTrue(result.passed)

    def test_missing_function_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mod.py").write_text("def other():\n    pass\n", encoding="utf-8")
            result = ga.check_llm_circularity(
                work_root=root, targets={"mod.py": ("judge",)})
        self.assertFalse(result.passed)

    def test_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ga.check_llm_circularity(
                work_root=Path(tmp), targets={"mod.py": ("judge",)})
        self.assertFalse(result.passed)


# ---------------------------------------------------------------------------
# synthetic tests: over-correction
# ---------------------------------------------------------------------------

class TestOverCorrection(unittest.TestCase):

    def test_empty_review_class_fails(self):
        """If every OOV token gets 'corrected' and none are held back as
        genuinely ambiguous, the corrector has started forcing readings
        onto uncertain data."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tsv(root / "lexical_correction" / "corrections.tsv",
                       ["token", "corrected"], [["foo", "food"]] * 20)
            _write_tsv(root / "lexical_correction" / "unchanged_review.tsv",
                       ["token", "count", "reason"], [])
            result = ga.check_over_correction(analysis_root=root)
        self.assertFalse(result.passed)

    def test_nonempty_review_class_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tsv(root / "lexical_correction" / "corrections.tsv",
                       ["token", "corrected"], [["foo", "food"]])
            _write_tsv(root / "lexical_correction" / "unchanged_review.tsv",
                       ["token", "count", "reason"], [["zog", 3, "no candidate"]])
            result = ga.check_over_correction(analysis_root=root)
        self.assertTrue(result.passed)

    def test_missing_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ga.check_over_correction(analysis_root=Path(tmp))
        self.assertFalse(result.passed)


# ---------------------------------------------------------------------------
# synthetic tests: single-method certainty
# ---------------------------------------------------------------------------

class TestSingleMethodCertainty(unittest.TestCase):

    def test_two_methods_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "triangulation" / "summary.json",
                        _triangulation_summary({}, ["net", "lda"]))
            result = ga.check_single_method_certainty(analysis_root=root)
        self.assertFalse(result.passed)

    def test_three_methods_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "triangulation" / "summary.json",
                        _triangulation_summary({}, ["net", "lda", "nmf"]))
            result = ga.check_single_method_certainty(analysis_root=root)
        self.assertTrue(result.passed)

    def test_threshold_is_a_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "triangulation" / "summary.json",
                        _triangulation_summary({}, ["net", "lda", "nmf"]))
            result = ga.check_single_method_certainty(analysis_root=root, minimum_methods=5)
        self.assertFalse(result.passed)


# ---------------------------------------------------------------------------
# synthetic tests: residue-dropping
# ---------------------------------------------------------------------------

class TestResidueDropping(unittest.TestCase):

    def test_zero_residue_rows_fails(self):
        """A ledger that itemizes nothing has not proven there is no
        residue — it has just stopped looking."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "coverage_ledger" / "summary.json", {
                "residue": {"n_rows": 0},
                "unexplained": {"characters": 0},
            })
            result = ga.check_residue_dropping(analysis_root=root)
        self.assertFalse(result.passed)

    def test_nonzero_unexplained_fails(self):
        """Residue that is tracked but left genuinely unexplained is a
        guardrail failure, not a passing disclosure."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "coverage_ledger" / "summary.json", {
                "residue": {"n_rows": 5},
                "unexplained": {"characters": 12, "tokens": 0},
            })
            result = ga.check_residue_dropping(analysis_root=root)
        self.assertFalse(result.passed)

    def test_itemized_and_fully_explained_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "coverage_ledger" / "summary.json", {
                "residue": {"n_rows": 5},
                "unexplained": {"characters": 0, "tokens": 0},
            })
            result = ga.check_residue_dropping(analysis_root=root)
        self.assertTrue(result.passed)


# ---------------------------------------------------------------------------
# build() plumbing
# ---------------------------------------------------------------------------

class TestBuild(unittest.TestCase):

    def test_build_writes_three_files_with_consistent_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis_root = Path(tmp) / "analysis"
            work_root = Path(tmp) / "work"
            _write_json(analysis_root / "triangulation" / "summary.json",
                        _triangulation_summary(
                            {"convergent": 1, "divergent": 2, "partial": 0, "untestable": 0},
                            ["net", "lda", "nmf"]))
            _write_tsv(analysis_root / "lexical_correction" / "corrections.tsv",
                       ["token", "corrected"], [["foo", "food"]])
            _write_tsv(analysis_root / "lexical_correction" / "unchanged_review.tsv",
                       ["token", "count", "reason"], [["zog", 3, "no candidate"]])
            _write_json(analysis_root / "coverage_ledger" / "summary.json", {
                "residue": {"n_rows": 2}, "unexplained": {"characters": 0},
            })
            work_root.mkdir(parents=True, exist_ok=True)
            for fname, funcs in ga.VERDICT_FUNCTIONS.items():
                body = "\n".join("def %s(*a, **k):\n    return True\n" % f for f in funcs)
                (work_root / fname).write_text(body, encoding="utf-8")

            out = Path(tmp) / "out"
            res = ga.build(out=out, analysis_root=analysis_root, work_root=work_root)

            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "audit_report.md").exists())
            self.assertTrue((out / "findings.tsv").exists())

            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["schema"], ga.SCHEMA)
            self.assertEqual(summary["n_total"], len(ga.CHECKS))
            self.assertEqual(len(summary["guardrails"]), len(ga.CHECKS))
            self.assertEqual(summary["n_passed"],
                              sum(1 for g in summary["guardrails"] if g["passed"]))
            self.assertEqual(summary["all_passed"], res["summary"]["all_passed"])

            findings_lines = (out / "findings.tsv").read_text().splitlines()
            self.assertEqual(len(findings_lines), len(ga.CHECKS) + 1)  # + header

    def test_main_exit_code_reflects_all_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            rc = ga.main(["--out", str(out)])
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(rc, 0 if summary["all_passed"] else 1)


# ---------------------------------------------------------------------------
# artifact tests: today's real audit, pinned
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_ARTIFACTS, "requires the real workspace artifacts")
class TestRealWorkspaceAudit(unittest.TestCase):
    """This is the CI-style check #76 asks for: the current state of
    the workspace's own guardrailed modules is audited, and the result
    is pinned here. A future change to #63, #64, #70, #71 or #73 that
    regresses a guardrail is caught by one of these assertions, not by
    this docstring."""

    @classmethod
    def setUpClass(cls):
        cls.results = {r.id: r for r in ga.run_audit()}

    def test_all_five_guardrails_currently_pass(self):
        failing = [gid for gid, r in self.results.items() if not r.passed]
        self.assertEqual(failing, [], "guardrails failing: %r" % failing)

    def test_hallucinated_coherence_sees_real_disagreement(self):
        r = self.results["hallucinated_coherence"]
        self.assertGreater(r.evidence["verdict_counts"].get("divergent", 0)
                            + r.evidence["verdict_counts"].get("partial", 0), 0)

    def test_llm_circularity_checked_all_ten_functions(self):
        r = self.results["llm_circularity"]
        expected = sum(len(fns) for fns in ga.VERDICT_FUNCTIONS.values())
        self.assertEqual(len(r.evidence["checked"]), expected)
        self.assertEqual(r.evidence["violations"], [])

    def test_over_correction_review_class_outnumbers_corrections(self):
        """Today's real split (15 corrected vs 200 reviewed) — a sanity
        anchor, not just a nonzero check."""
        r = self.results["over_correction"]
        self.assertGreater(r.evidence["n_review"], r.evidence["n_corrections"])

    def test_single_method_certainty_uses_five_methods(self):
        r = self.results["single_method_certainty"]
        self.assertEqual(r.evidence["n_methods"], 5)

    def test_residue_dropping_zero_unexplained_at_every_level(self):
        r = self.results["residue_dropping"]
        self.assertTrue(all(v == 0 for v in r.evidence["unexplained"].values()))
        self.assertGreater(r.evidence["n_rows"], 0)

    def test_build_end_to_end_against_real_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = ga.build(out=Path(tmp) / "out")
        self.assertTrue(res["summary"]["all_passed"])


if __name__ == "__main__":
    unittest.main()
