#!/usr/bin/env python3
"""Regression tests for operator function classification (Chainlink #12).

Run from the workspace root (inside the `marker` conda env):

  python work/test_operator_functions.py
  python -m unittest discover -s work -p 'test_*.py'

Checks that the headline quantities reported in
`analysis/operator_functions/` are reproducible from the layout corpus, that
the operator tokenization reproduces the #10 inventory exactly, and that the
per-operator verdicts follow from the measured distributions.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import operator_functions as of

SRC = os.path.join(ROOT, "analysis", "layout_recovery", "blood-electric.layout.md")
OUT = os.path.join(ROOT, "analysis", "operator_functions")

EXPECTED_COUNTS = {
    "::": 3083, "//": 2149, "=": 2047, "/": 1484, ">": 777,
    "<<": 412, ">>": 386, "<": 381, "++": 198, "|": 17, "+": 15,
}


@unittest.skipUnless(os.path.exists(SRC), "layout corpus not present")
class OperatorFunctionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = of.load_pages(SRC)
        cls.occ = of.build_occurrences(cls.pages)
        cls.ops, cls.prof = of.per_operator(cls.occ)
        cls.free = of.free_hyphen_occurrences(cls.pages)
        _fo, cls.fprof = of.per_operator(cls.free)

    def test_tokenization_reproduces_inventory(self):
        self.assertEqual(len(self.occ), 10949)
        self.assertEqual(dict(self.ops), EXPECTED_COUNTS)
        self.assertEqual(len(self.ops), 11)
        # free hyphens are additional to the #10 operator inventory
        self.assertEqual(len(self.free), 177)

    def test_free_hyphen_profile(self):
        p = self.fprof["-"]
        self.assertEqual(p["n"], 177)
        self.assertAlmostEqual(p["interior"], 0.994, delta=0.01)
        self.assertAlmostEqual(p["upper_next"], 0.113, delta=0.01)
        # right neighbours are dominated by clause-openers
        top = dict(p["top_rwords"])
        self.assertEqual(top.get("the"), 47)
        self.assertEqual(top.get("I"), 14)
        # the free dash behaves like :: (phrase separator), not // (sentence)
        self.assertLess(p["upper_next"], self.prof["//"]["upper_next"] / 3)
        # whitespace on both sides by construction
        self.assertEqual(p["glue_l"], 0.0)
        self.assertEqual(p["glue_r"], 0.0)

    def test_absent_operators(self):
        raw = "".join(self.pages.values())
        for ch in ("`", "~", "^"):
            self.assertEqual(raw.count(ch), 0, ch)
        for op in ("--", "==", "->", "<-"):
            self.assertEqual(raw.count(op), 0, op)

    def test_absent_operator_audit_is_measured_both_ways(self):
        """`--`/`->` are 0 in page bodies but non-zero in the raw layout file.

        A naive grep of the source markdown finds them, because they live in
        `<!-- ... -->` scaffolding comments. The spec must state both numbers
        so the "0 occurrences" claim is auditable rather than a bare assertion.
        """
        with open(SRC, encoding="utf-8", errors="ignore") as f:
            raw_file = f.read()
        audit = of.absent_operator_audit(raw_file, self.pages)
        self.assertEqual(set(audit), set(of.ABSENT_OPS))
        for op in ("`", "~", "^", "==", "<-"):
            self.assertEqual(audit[op]["page_body"], 0, op)
            self.assertEqual(audit[op]["raw_file"], 0, op)
        # `--` and `->` appear ONLY in scaffolding, never in the page bodies
        self.assertEqual(audit["--"]["page_body"], 0)
        self.assertEqual(audit["->"]["page_body"], 0)
        self.assertGreater(audit["--"]["raw_file"], 0)
        self.assertGreater(audit["->"]["raw_file"], 0)
        # every raw occurrence sits inside a `<!-- ... -->` comment
        stripped = re.sub(r"<!--.*?-->", "", raw_file, flags=re.S)
        for op in ("--", "->"):
            self.assertEqual(stripped.count(op), 0, op)
        self.assertEqual(audit["--"]["raw_file"], 474)
        self.assertEqual(audit["->"]["raw_file"], 237)

    def test_report_scopes_absent_operator_claim(self):
        path = os.path.join(OUT, "classification_spec.md")
        if not os.path.exists(path):
            self.skipTest("classification_spec.md not generated")
        with open(path) as f:
            text = f.read()
        # the spec must not make a bare, unscoped "0 times in the corpus" claim
        self.assertNotIn("occur 0 times in the corpus", text)
        self.assertIn("page-body corpus", text)
        self.assertIn("474", text)   # raw-file `--` count, disclosed
        self.assertIn("237", text)   # raw-file `->` count, disclosed
        self.assertIn("scaffolding", text)
        # summary.json carries both numbers
        spath = os.path.join(OUT, "summary.json")
        if os.path.exists(spath):
            with open(spath) as f:
                summary = json.load(f)
            audit = summary["absent_operator_audit"]
            self.assertEqual(audit["--"], {"raw_file": 474, "page_body": 0})
            self.assertEqual(audit["->"], {"raw_file": 237, "page_body": 0})

    def test_signature_profiles(self):
        # // starts sentences; :: does not.
        self.assertAlmostEqual(self.prof["//"]["upper_next"], 0.527, delta=0.01)
        self.assertAlmostEqual(self.prof["::"]["upper_next"], 0.130, delta=0.01)
        self.assertGreater(self.prof["//"]["upper_next"],
                           4 * self.prof["::"]["upper_next"])
        # = is a tight word-word linker.
        self.assertGreater(self.prof["="]["both_glued"], 0.90)
        self.assertGreater(self.prof["="]["interior"], 0.90)
        # << opens at line start / after an operator.
        self.assertGreater(self.prof["<<"]["sol"], 0.15)
        self.assertGreater(self.prof["<<"]["lclass"].get("OP", 0) / self.prof["<<"]["n"], 0.60)
        # / has a sentence-final use.
        self.assertAlmostEqual(self.prof["/"]["eol"], 0.29, delta=0.02)

    def test_two_proportion_test(self):
        n_col = self.prof["::"]["n"]
        k_col = sum(o["upper_next"] for o in self.occ if o["op"] == "::")
        n_sl = self.prof["//"]["n"]
        k_sl = sum(o["upper_next"] for o in self.occ if o["op"] == "//")
        z, p = of.two_proportion_z(k_sl, n_sl, k_col, n_col)
        self.assertGreater(z, 20)
        self.assertLess(p, 1e-100)

    def test_context_independence(self):
        # Every frequent operator has a context distribution distinct from rest.
        for op in ("::", "//", "=", "/", "<<", ">>", "<", ">"):
            G, df, p = of.chi2_context(self.occ, op)
            self.assertIsNotNone(p, op)
            self.assertLess(p, 1e-3, op)

    def test_linker_evidence(self):
        link = of.linker_evidence(self.pages)
        self.assertEqual(link["equals"]["n"], 1747)
        self.assertEqual(link["equals"]["unique"], 646)
        self.assertEqual(link["slash"]["n"], 767)
        self.assertEqual(link["plus"]["n"], 13)
        self.assertEqual(link["plusplus"]["n"], 127)
        self.assertGreater(link["hyphen"]["n"], 2800)
        top = dict(link["equals"]["top"])
        self.assertEqual(top.get("gene=TV"), 147)

    def test_delimiter_pairing(self):
        block = of.pairing_stats(self.pages, "<<", ">>")
        block_clean = of.pairing_stats(self.pages, "<<", ">>",
                                       of.HTML_ARTIFACT_PAGES)
        angle = of.pairing_stats(self.pages, "<", ">")
        angle_clean = of.pairing_stats(self.pages, "<", ">",
                                       of.HTML_ARTIFACT_PAGES)
        self.assertEqual(block["matched"], 230)
        self.assertEqual(block["median_span_words"], 5)
        # <<>> pairing is robust to removing HTML-artifact pages...
        self.assertEqual(block_clean["matched"], 229)
        # ...whereas single-angle pairing is not.
        self.assertEqual(angle["matched"], 245)
        self.assertEqual(angle_clean["matched"], 145)

    def test_summary_artifact(self):
        path = os.path.join(OUT, "summary.json")
        if not os.path.exists(path):
            self.skipTest("summary.json not generated; run work/operator_functions.py")
        with open(path) as f:
            summary = json.load(f)
        self.assertEqual(summary["n_operator_tokens"], 10949)
        self.assertEqual(summary["counts"], EXPECTED_COUNTS)
        self.assertEqual(summary["n_free_hyphen"], 177)
        self.assertEqual(summary["free_hyphen"]["n"], 177)
        self.assertEqual(summary["free_hyphen"]["top_rwords"][0], ["the", 47])
        self.assertEqual(summary["verdict_short"]["="], "SUPPORTED naming/identity linker")
        self.assertEqual(
            summary["verdict_short"]["-"],
            "SUPPORTED word-internal compound linker + free-standing dash separator")
        self.assertEqual(summary["verdict_short"]["`"], "REJECTED absent")
        self.assertEqual(summary["delimiter_pairing"]["block"]["matched"], 230)

    def test_operator_profiles_has_free_hyphen(self):
        path = os.path.join(OUT, "operator_profiles.tsv")
        if not os.path.exists(path):
            self.skipTest("operator_profiles.tsv not generated")
        with open(path) as f:
            rows = {line.split("\t")[0]: line.rstrip("\n").split("\t")
                    for line in f}
        self.assertIn("-", rows)
        self.assertEqual(rows["-"][1], "177")
        self.assertAlmostEqual(float(rows["-"][5]), 0.9944, delta=0.001)  # interior

    def test_report_present_and_verdicts(self):
        path = os.path.join(OUT, "classification_spec.md")
        if not os.path.exists(path):
            self.skipTest("classification_spec.md not generated")
        with open(path) as f:
            text = f.read()
        for op in ("`::`", "`//`", "`=`", "`/`", "`<<`", "`>>`", "`++`",
                   "`-`", "`` ` ``"):
            self.assertIn(op, text)
        self.assertIn("REJECTED", text)
        # the corrected `-` verdict reports the measured free-standing role
        self.assertIn("177 free-standing whitespace-bounded", text)
        self.assertIn("free-standing role (measured)", text)
        self.assertIn("free-dash top right words", text)
        # the old definitional inference must be gone
        self.assertNotIn("not a free operator", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
