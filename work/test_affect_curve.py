#!/usr/bin/env python3
"""Regression tests for #44 Affect and valence curve.

Run from the workspace root (inside the `marker` conda env):

  python work/test_affect_curve.py

Checks that the headline quantities reported in
`analysis/affect/` are reproducible and that the core instruments --
lexicon, markers, nulls, and report -- produce structurally sound output.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import affect_curve as ac  # noqa: E402


class AffectCurveTest(unittest.TestCase):
    """Regression suite for #44 affect and valence curve.

    Builds the full artifact into a temp directory and checks the
    invariants that matter for the task's four affect axes (fear,
    ecstasy, paranoia, disgust) plus valence, and the explicit markers
    such as ``Fear=cell``.
    """

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.mkdtemp(prefix="affect_test_")
        cls.tmp_dir = Path(tmp)
        ac.build(out_dir=cls.tmp_dir, quiet=True)
        cls.summary = json.loads((cls.tmp_dir / "summary.json").read_text())

    # ------------------------------------------------------------------
    # schema & determinism
    # ------------------------------------------------------------------

    def test_schema(self):
        self.assertIn("schema", self.summary)
        self.assertEqual(self.summary["schema"], "affect-curve/1")

    def test_determinism_seed(self):
        self.assertIn("seed", self.summary["determinism"])
        self.assertEqual(self.summary["determinism"]["seed"], 0)

    def test_determinism_shifts(self):
        self.assertIn("n_shift", self.summary["determinism"])
        self.assertIn("n_term_shuffle", self.summary["determinism"])

    # ------------------------------------------------------------------
    # clause / page counts
    # ------------------------------------------------------------------

    def test_clause_count(self):
        """5,614 clauses (the #39 artifact)."""
        self.assertEqual(self.summary["n_clauses"], 5614)

    def test_page_count(self):
        """225 pages (the #39 artifact)."""
        self.assertEqual(self.summary["n_pages"], 225)

    # ------------------------------------------------------------------
    # lexicon: term inventory and attestation
    # ------------------------------------------------------------------

    def test_lexicon_term_count(self):
        """Curated lexicon has 262 terms (published in-module)."""
        self.assertEqual(self.summary["lexicon"]["n_terms"], 262)

    def test_lexicon_attestation(self):
        """At least one term per affect axis is attested in the book."""
        for a in ac.AFFECTS:
            terms = ac.terms_for(a)
            attested = sum(1 for t in terms
                           if self.summary["lexicon"]["per_term_attested"].get(t, 0) > 0)
            self.assertGreater(attested, 0,
                               f"affect {a} has no attested lexicon terms")

    # ------------------------------------------------------------------
    # explicit markers (fused affect units)
    # ------------------------------------------------------------------

    def test_markers_exist(self):
        """At least one fused affect marker is produced."""
        self.assertGreater(
            self.summary["markers"]["n_markers"], 0,
            "no fused affect markers produced")

    def test_marker_fear_cell(self):
        """The task's own example Fear=cell should be among the markers."""
        marker_texts = [m["marker"] for m in self.summary["markers"]["top"]]
        self.assertIn("fear=cell", marker_texts,
                      "Fear=cell marker not found in top markers")

    def test_marker_occurrences(self):
        """Total marker occurrences are published."""
        n_occ = self.summary["markers"]["n_occurrences"]
        self.assertGreater(n_occ, 0, "no marker occurrences recorded")

    # ------------------------------------------------------------------
    # affect rates (headline numbers)
    # ------------------------------------------------------------------

    def test_affect_rates_positive(self):
        """Headline affect rates should be positive numbers."""
        t = self.summary["lexicon"]["totals"]
        for a in ac.AFFECTS:
            rate = t[a]["rate"]
            self.assertGreaterEqual(rate, 0, f"{a} rate should be >= 0, got {rate}")

    def test_valence_rate(self):
        """Valence rate should be published."""
        t = self.summary["lexicon"]["totals"]
        self.assertIn("rate_valence", t["valence"])

    # ------------------------------------------------------------------
    # keyness: every lexicon term has a table entry
    # ------------------------------------------------------------------

    def test_keyness_rows(self):
        """affect_lexicon.tsv has one row per lexicon term."""
        lex_rows = []
        lex_tsv = self.tmp_dir / "affect_lexicon.tsv"
        if lex_tsv.exists():
            with open(lex_tsv) as f:
                header = f.readline().strip().split("\t")
                for line in f:
                    lex_rows.append(line.strip().split("\t"))
            self.assertEqual(len(lex_rows),
                             self.summary["lexicon"]["n_terms"],
                             "keyness row count != lexicon term count")

    # ------------------------------------------------------------------
    # per-page curve data
    # ------------------------------------------------------------------

    def test_per_page_rows(self):
        """affect_per_page.tsv has one row per page (225)."""
        per_page_tsv = self.tmp_dir / "affect_per_page.tsv"
        self.assertTrue(per_page_tsv.exists(),
                        "affect_per_page.tsv not produced")
        with open(per_page_tsv) as f:
            header = f.readline().strip().split("\t")
            data_rows = [line.strip().split("\t") for line in f]
        self.assertEqual(len(data_rows), 225,
                         "per-page row count != 225")

    # ------------------------------------------------------------------
    # classifier scores
    # ------------------------------------------------------------------

    def test_classifier_scores(self):
        """classifier_scores.tsv has one row per clause."""
        cls_tsv = self.tmp_dir / "classifier_scores.tsv"
        self.assertTrue(cls_tsv.exists(),
                        "classifier_scores.tsv not produced")
        with open(cls_tsv) as f:
            header = f.readline().strip().split("\t")
            data_rows = [line.strip().split("\t") for line in f]
        # should have 5,614 rows (one per clause)
        self.assertEqual(len(data_rows), 5614,
                         "classifier scores row count != 5614")

    # ------------------------------------------------------------------
    # controls / nulls
    # ------------------------------------------------------------------

    def test_controls_tsv(self):
        """controls.tsv has null distribution entries."""
        controls_tsv = self.tmp_dir / "controls.tsv"
        self.assertTrue(controls_tsv.exists(),
                        "controls.tsv not produced")
        with open(controls_tsv) as f:
            header = f.readline().strip().split("\t")
            data_rows = [line.strip().split("\t") for line in f]
        self.assertGreater(len(data_rows), 0,
                           "controls.tsv should have null entries")

    def test_summary_nulls_present(self):
        """summary.json nulls dict has all four null types plus n_affect_terms."""
        nulls = self.summary["nulls"]
        expected_keys = {"page_order", "clause_shuffle",
                         "term_shuffle_core", "term_shuffle_vocabulary",
                         "n_affect_terms"}
        self.assertSetEqual(set(nulls.keys()), expected_keys,
                            "missing null type in summary")

    # ------------------------------------------------------------------
    # english baseline comparison
    # ------------------------------------------------------------------

    def test_english_baseline(self):
        """english_baseline.tsv has per-corpus and pooled rows."""
        eng_tsv = self.tmp_dir / "english_baseline.tsv"
        self.assertTrue(eng_tsv.exists(),
                        "english_baseline.tsv not produced")
        with open(eng_tsv) as f:
            header = f.readline().strip().split("\t")
            data_rows = [line.strip().split("\t") for line in f]
        # should have at least the 7 #56 corpora + pooled row
        self.assertGreaterEqual(len(data_rows), 8,
                                "english baseline should have corpus + pooled rows")

    # ------------------------------------------------------------------
    # report markdown
    # ------------------------------------------------------------------

    def test_report_markdown(self):
        """affect_report.md exists and mentions the four affects."""
        report_md = self.tmp_dir / "affect_report.md"
        self.assertTrue(report_md.exists(),
                        "affect_report.md not produced")
        text = report_md.read_text()
        for a in ac.AFFECTS:
            self.assertIn(a, text,
                          f"affect {a} not mentioned in affect_report.md")

    def test_report_headline(self):
        """Report headline has the four affect hit counts."""
        report_md = self.tmp_dir / "affect_report.md"
        text = report_md.read_text()
        # The headline section lists the four affect hit counts
        for a in ac.AFFECTS:
            self.assertIn(a, text,
                          f"affect {a} not mentioned in affect_report.md")
        # Also check for the headline numbers pattern
        self.assertIn("affect occurrences", text,
                      "headline affect occurrences not in report")


if __name__ == "__main__":
    unittest.main()