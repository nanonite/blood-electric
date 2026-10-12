#!/usr/bin/env python3
"""Regression tests for the consistency checker (#71).

Run from the workspace root (inside the `marker` conda env):

    python work/test_consistency_checker.py
    python -m unittest discover -s work -p 'test_*.py'

Two tiers, matching the workspace's other suites:

* **Synthetic tests** on hand-built term/readings dicts, which pin each
  check's logic and prove the failure paths have teeth (a gloss
  mismatch, a dual curated reading, a role mismatch, contradictory
  operator readings and a term with two different glosses all have to
  be caught).
* **Artifact tests** (guarded by skipUnless) which load the real
  #56/#62/#30 artifacts and verify the checker runs end-to-end,
  produces a well-formed report, and that the curated tables are
  internally consistent on this workspace.
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

import consistency_checker as cc

HAVE_ARTIFACTS = all(p.exists() for p in (
    cc.TERMBASE, cc.ENTITY))


# ---------------------------------------------------------------------------
# synthetic helpers
# ---------------------------------------------------------------------------

def _gt():
    """A minimal GLOSS_TERMS-style dict."""
    return {
        "foo": ("noun", "foo: a test term", "high"),
        "bar": ("compound", "bar: a compound", "medium"),
    }


def _tb():
    """A minimal termbase-style dict."""
    return {
        "foo": {"term": "foo", "pos_class": "noun",
                "gloss": "foo: a test term", "confidence": "high",
                "entity_role_30": "", "does_verbs_30": "",
                "done_to_verbs_30": ""},
        "bar": {"term": "bar", "pos_class": "compound",
                "gloss": "bar: a compound", "confidence": "medium",
                "entity_role_30": "", "does_verbs_30": "",
                "done_to_verbs_30": ""},
    }


# ---------------------------------------------------------------------------
# check 1: GLOSS_TERMS vs termbase
# ---------------------------------------------------------------------------

class TestGlossTermsVsTermbase(unittest.TestCase):
    def test_matching(self):
        v = cc.check_gloss_terms_vs_termbase(_gt(), _tb())
        self.assertEqual(v, [])

    def test_gloss_mismatch(self):
        tb = _tb()
        tb["foo"]["gloss"] = "foo: a different gloss"
        v = cc.check_gloss_terms_vs_termbase(_gt(), tb)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "gloss_mismatch")
        self.assertEqual(v[0]["term"], "foo")

    def test_confidence_mismatch(self):
        tb = _tb()
        tb["foo"]["confidence"] = "low"
        v = cc.check_gloss_terms_vs_termbase(_gt(), tb)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "confidence_mismatch")

    def test_pos_mismatch(self):
        tb = _tb()
        tb["foo"]["pos_class"] = "verb"
        v = cc.check_gloss_terms_vs_termbase(_gt(), tb)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "pos_mismatch")

    def test_missing_in_termbase_with_freq(self):
        tb = _tb()
        del tb["foo"]
        # foo is attested (freq > 0) but missing from termbase -> violation
        v = cc.check_gloss_terms_vs_termbase(_gt(), tb, freq={"foo": 5})
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "missing_in_termbase")

    def test_missing_in_termbase_no_freq(self):
        tb = _tb()
        del tb["foo"]
        # no freq data -> not flagged (conservative)
        v = cc.check_gloss_terms_vs_termbase(_gt(), tb)
        self.assertEqual(v, [])

    def test_missing_in_termbase_unattested(self):
        tb = _tb()
        del tb["foo"]
        # foo has freq=0 -> expected to be missing, not a violation
        v = cc.check_gloss_terms_vs_termbase(_gt(), tb, freq={"foo": 0})
        self.assertEqual(v, [])

    def test_multiple_violations(self):
        tb = _tb()
        tb["foo"]["gloss"] = "wrong"
        tb["foo"]["confidence"] = "low"
        tb["bar"]["pos_class"] = "noun"
        v = cc.check_gloss_terms_vs_termbase(_gt(), tb)
        self.assertEqual(len(v), 3)


# ---------------------------------------------------------------------------
# check 2: PLAIN_READINGS vs GLOSS_TERMS
# ---------------------------------------------------------------------------

class TestPlainReadingsVsGlossTerms(unittest.TestCase):
    def test_no_overlap(self):
        pr = {"feti": ("fetuses", "high")}
        v = cc.check_plain_readings_vs_gloss_terms(pr, _gt())
        self.assertEqual(v, [])

    def test_same_reading_no_violation(self):
        pr = {"foo": ("foo: a test term", "high")}
        v = cc.check_plain_readings_vs_gloss_terms(pr, _gt())
        self.assertEqual(v, [])

    def test_dual_curated_reading(self):
        pr = {"foo": ("foo: something else", "high")}
        v = cc.check_plain_readings_vs_gloss_terms(pr, _gt())
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "dual_curated_reading")
        self.assertEqual(v[0]["term"], "foo")


# ---------------------------------------------------------------------------
# check 3: entity glossary vs termbase
# ---------------------------------------------------------------------------

class TestEntityGlossaryVsTermbase(unittest.TestCase):
    def test_matching(self):
        eg = {"foo": {"role": "persona", "does_verbs": "is(2)|was(1)",
                      "done_to_verbs": "love(1)"}}
        tb = _tb()
        tb["foo"]["entity_role_30"] = "persona"
        tb["foo"]["does_verbs_30"] = "is(2)|was(1)"
        tb["foo"]["done_to_verbs_30"] = "love(1)"
        v = cc.check_entity_glossary_vs_termbase(eg, tb)
        self.assertEqual(v, [])

    def test_role_mismatch(self):
        eg = {"foo": {"role": "persona", "does_verbs": "",
                      "done_to_verbs": ""}}
        tb = _tb()
        tb["foo"]["entity_role_30"] = "concept"
        v = cc.check_entity_glossary_vs_termbase(eg, tb)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "role_mismatch")

    def test_does_verbs_mismatch(self):
        eg = {"foo": {"role": "persona", "does_verbs": "is(2)",
                      "done_to_verbs": ""}}
        tb = _tb()
        tb["foo"]["entity_role_30"] = "persona"
        tb["foo"]["does_verbs_30"] = "was(3)"
        v = cc.check_entity_glossary_vs_termbase(eg, tb)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "does_verbs_mismatch")

    def test_done_to_verbs_mismatch(self):
        eg = {"foo": {"role": "persona", "does_verbs": "",
                      "done_to_verbs": "love(1)|speed(2)"}}
        tb = _tb()
        tb["foo"]["entity_role_30"] = "persona"
        tb["foo"]["done_to_verbs_30"] = "hate(1)"
        v = cc.check_entity_glossary_vs_termbase(eg, tb)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "done_to_verbs_mismatch")

    def test_no_entity_entry_is_skip(self):
        # termbar has no entity_glossary entry -> no violation
        v = cc.check_entity_glossary_vs_termbase({}, _tb())
        self.assertEqual(v, [])


# ---------------------------------------------------------------------------
# check 4: operator glosses
# ---------------------------------------------------------------------------

class TestOperatorGlosses(unittest.TestCase):
    def test_no_violations(self):
        op = {"//": "sentence boundary", "=": "naming seam"}
        v = cc.check_operator_glosses(op)
        self.assertEqual(v, [])

    def test_empty_reading(self):
        op = {"//": "sentence boundary", "=": ""}
        v = cc.check_operator_glosses(op)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "empty_reading")

    def test_whitespace_only_reading(self):
        op = {"//": "sentence boundary", "=": "   "}
        v = cc.check_operator_glosses(op)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "empty_reading")


# ---------------------------------------------------------------------------
# check 5: one term = one gloss
# ---------------------------------------------------------------------------

class TestOneTermOneGloss(unittest.TestCase):
    def test_no_violations(self):
        gt = _gt()
        pr = {}
        tb = _tb()
        v = cc.check_one_term_one_gloss(gt, pr, tb)
        self.assertEqual(v, [])

    def test_multiple_glosses(self):
        gt = {"foo": ("noun", "foo: a test term", "high")}
        pr = {}
        tb = {"foo": {"gloss": "foo: a different gloss"}}
        v = cc.check_one_term_one_gloss(gt, pr, tb)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "multiple_glosses")
        self.assertEqual(v[0]["term"], "foo")

    def test_same_gloss_different_artifact_no_violation(self):
        gt = {"foo": ("noun", "foo: a test term", "high")}
        pr = {"bar": ("bar reading", "medium")}
        tb = {"foo": {"gloss": "foo: a test term"},
               "bar": {"gloss": "bar reading"}}
        v = cc.check_one_term_one_gloss(gt, pr, tb)
        self.assertEqual(v, [])

    def test_three_way_mismatch(self):
        gt = {"foo": ("noun", "foo: a test term", "high")}
        pr = {"foo": ("foo: plain reading", "medium")}
        tb = {"foo": {"gloss": "foo: termbase gloss"}}
        v = cc.check_one_term_one_gloss(gt, pr, tb)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["kind"], "multiple_glosses")
        # 3 distinct glosses
        self.assertIn("foo: a test term", v[0]["detail"])
        self.assertIn("foo: plain reading", v[0]["detail"])
        self.assertIn("foo: termbase gloss", v[0]["detail"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_norm_case_and_space(self):
        self.assertEqual(cc._norm("  Foo   Bar "), "foo bar")
        self.assertEqual(cc._norm("foo bar"), cc._norm("Foo  Bar"))

    def test_verbs_parsing(self):
        self.assertEqual(cc._verbs("is(32)|was(26)"), ("is", "was"))
        self.assertEqual(cc._verbs(""), ())
        self.assertEqual(cc._verbs("love(7)|speed(6)"), ("love", "speed"))
        # order doesn't matter
        self.assertEqual(cc._verbs("was(1)|is(2)"), ("is", "was"))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):
    def test_no_violations_report(self):
        stats = {"total": 0, "by_check": {}, "by_kind": {},
                 "n_gloss_terms": 10, "n_plain_readings": 5,
                 "n_termbases": 8, "n_entity_glossary": 3,
                 "n_operator_glosses": 4}
        r = cc.build_report([], stats, "2026-01-01 00:00 UTC")
        self.assertIn("No violations found", r)
        self.assertIn("total", r)

    def test_violations_report(self):
        violations = [{
            "check": "test_check", "term": "foo", "kind": "test_kind",
            "detail": "a | b", "artifacts": ["A", "B"],
        }]
        stats = {"total": 1, "by_check": {"test_check": 1},
                 "by_kind": {"test_kind": 1},
                 "n_gloss_terms": 10, "n_plain_readings": 5,
                 "n_termbases": 8, "n_entity_glossary": 3,
                 "n_operator_glosses": 4}
        r = cc.build_report(violations, stats, "2026-01-01 00:00 UTC")
        self.assertIn("test_check", r)
        self.assertIn("foo", r)
        self.assertIn("test_kind", r)
        # pipe in detail is escaped
        self.assertIn("a \\| b", r)


# ---------------------------------------------------------------------------
# artifact tests (the real book)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts not present")
class ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="cc-test-")
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build_produces_artifacts(self):
        res = cc.build(out=self.out)
        self.assertTrue((self.out / "violations.tsv").exists())
        self.assertTrue((self.out / "summary.json").exists())
        self.assertTrue((self.out / "consistency_report.md").exists())

    def test_summary_json_structure(self):
        res = cc.build(out=self.out)
        s = res["summary"]
        self.assertEqual(s["schema"], "consistency/1")
        self.assertIn("stats", s)
        self.assertIn("violations", s)
        for key in ("total", "by_check", "by_kind"):
            self.assertIn(key, s["stats"])

    def test_violations_tsv_header(self):
        cc.build(out=self.out)
        rows = cc.cs.read_tsv(self.out / "violations.tsv")
        # header must be present and consistent
        with (self.out / "violations.tsv").open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
        self.assertEqual(header, ["check", "term", "kind", "detail",
                                  "artifacts"])

    def test_curated_tables_are_self_consistent(self):
        """On this workspace the curated tables must not contradict
        each other — the whole point of the checker."""
        res = cc.build(out=self.out)
        # We assert zero violations: the curated tables are authored
        # to be consistent, and if they drift the checker must catch it.
        # If this fails, the violation TSV has the detail.
        self.assertEqual(res["violations"], [],
                         "curated tables contradict each other: %s"
                         % json.dumps(res["violations"][:3], indent=2))

    def test_gloss_terms_loaded(self):
        gt = cc.load_gloss_terms()
        self.assertTrue(gt, "GLOSS_TERMS is empty")
        # every entry is (pos, gloss, confidence)
        for term, (pos, gloss, conf) in gt.items():
            self.assertTrue(pos, "%s has empty pos" % term)
            self.assertTrue(gloss, "%s has empty gloss" % term)
            self.assertIn(conf, ("high", "medium", "low"),
                          "%s has unexpected confidence %r" % (term, conf))

    def test_termbase_loaded(self):
        tb = cc.load_termbase()
        self.assertTrue(tb, "termbase is empty")
        for term, row in tb.items():
            self.assertTrue(row.get("gloss", ""),
                            "%s has empty gloss" % term)
            self.assertIn(row.get("confidence", ""),
                          ("high", "medium", "low"),
                          "%s has unexpected confidence" % term)

    def test_plain_readings_loaded(self):
        pr = cc.load_plain_readings()
        self.assertTrue(pr, "PLAIN_READINGS is empty")
        for surface, (reading, conf) in pr.items():
            self.assertTrue(reading, "%s has empty reading" % surface)
            self.assertIn(conf, ("high", "medium", "low"),
                          "%s has unexpected confidence" % surface)

    def test_operator_glosses_loaded(self):
        op = cc.load_operator_glosses()
        self.assertTrue(op, "OPERATOR_GLOSSES is empty")
        for operator, reading in op.items():
            self.assertTrue(reading and reading.strip(),
                            "%r has empty reading" % operator)

    def test_entity_glossary_loaded(self):
        eg = cc.load_entity_glossary()
        self.assertTrue(eg, "entity glossary is empty")
        for entity, row in eg.items():
            self.assertTrue(row.get("role", ""),
                            "%s has empty role" % entity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
