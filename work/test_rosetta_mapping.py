#!/usr/bin/env python3
"""Regression tests for the Rosetta mapping (Chainlink #15).

Run from the workspace root (inside the `marker` conda env):

  python work/test_rosetta_mapping.py
  python -m unittest discover -s work -p 'test_*.py'

Checks that the headline quantities reported in `analysis/rosetta/` are
reproducible from the layout corpus, that the scoring tiers behave as
specified, that the book side reconciles to #10/#12, and that the winning
notation is stable under the published sensitivity variants.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import operator_functions as of  # reuse #12 corpus reader
import rosetta_mapping as rm

SRC = os.path.join(ROOT, "analysis", "layout_recovery",
                   "blood-electric.layout.md")
OUT = os.path.join(ROOT, "analysis", "rosetta")

EXPECTED_COUNTS = {
    "::": 3083, "//": 2149, "=": 2047, "/": 1484, ">": 777,
    "<<": 412, ">>": 386, "<": 381, "++": 198, "|": 17, "+": 15,
}


@unittest.skipUnless(os.path.exists(SRC), "layout corpus not present")
class RosettaMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = of.load_pages(SRC)
        cls.occ = of.build_occurrences(cls.pages)
        cls.ops, cls.prof = of.per_operator(cls.occ)
        for n in rm.NOTATIONS.values():
            rm.index_notation(n)
        cls.rows = rm.scorecard(rm.BOOK_CHANNELS, rm.BOOK_FEATURES)
        cls.by_name = {r["notation"]: r for r in cls.rows}
        cls.sens = rm.sensitivity(rm.BOOK_CHANNELS, rm.BOOK_FEATURES)
        cls.perm = rm.permutation_test(rm.BOOK_CHANNELS, rm.BOOK_FEATURES,
                                       n=400)

    # -- book side / reconciliation ----------------------------------------
    def test_reconciles_operator_inventory(self):
        self.assertEqual(len(self.occ), 10949)
        self.assertEqual(dict(self.ops), EXPECTED_COUNTS)
        self.assertEqual(sum(c["count"] for c in rm.BOOK_CHANNELS), 10949)

    def test_hyphen_channels_excluded_from_headline(self):
        ops = {c["op"] for c in rm.BOOK_CHANNELS}
        self.assertNotIn("-", ops)
        extra = {c["op"] for c in rm.HYPHEN_CHANNELS}
        self.assertIn("- (word-internal)", extra)
        self.assertIn("- (free dash)", extra)
        self.assertEqual(sum(c["count"] for c in rm.HYPHEN_CHANNELS), 3011)

    def test_book_features_are_measured(self):
        # #12 rejected // as a comment and the backtick escape; no quantifiers.
        self.assertFalse(rm.BOOK_FEATURES["has_comment"])
        self.assertFalse(rm.BOOK_FEATURES["has_escape"])
        self.assertFalse(rm.BOOK_FEATURES["has_quantifier"])
        # 230 matched <<>> spans with max depth 30 -> nested pairs exist.
        self.assertTrue(rm.BOOK_FEATURES["nested_pairs"])
        self.assertTrue(rm.BOOK_FEATURES["angle_brackets"])
        self.assertTrue(rm.BOOK_FEATURES["multi_char_ops"])
        self.assertTrue(rm.BOOK_FEATURES["free_form"])

    # -- taxonomy / signatures ---------------------------------------------
    def test_all_roles_are_in_taxonomy(self):
        for name, n in rm.NOTATIONS.items():
            for role in n["roles"]:
                self.assertIn(role, rm.ROLE_FAMILY, f"{name}:{role}")
        for c in rm.BOOK_CHANNELS + rm.HYPHEN_CHANNELS:
            for role in c["roles"]:
                self.assertIn(role, rm.ROLE_FAMILY, c["op"])
                self.assertGreater(c["roles"][role], 0.0)

    def test_role_weights_sum_to_one(self):
        for c in rm.BOOK_CHANNELS + rm.HYPHEN_CHANNELS:
            self.assertAlmostEqual(sum(c["roles"].values()), 1.0, places=6,
                                   msg=c["op"])

    def test_notation_indexing(self):
        c = rm.NOTATIONS["C / C++"]
        self.assertIn("::", c["symbols"])
        self.assertIn("field_sep", c["role_symbols"])
        # every symbol has at least one role and vice versa
        for sym, rs in c["symbol_roles"].items():
            self.assertTrue(rs, sym)
            for role, _pos in rs:
                self.assertIn(role, rm.ROLE_FAMILY)

    # -- match tiers -------------------------------------------------------
    def test_exact_match(self):
        c = rm.NOTATIONS["C / C++"]
        score, tier, msym, mrole, _ = rm.match_role("::", "field_sep", c)
        self.assertEqual((score, tier), (rm.TIER_EXACT, "exact"))
        self.assertEqual((msym, mrole), ("::", "field_sep"))

    def test_double_slash_not_credited_as_comment(self):
        # // is a comment symbol in C, but the book's // is an utterance
        # boundary; #12 rejected the comment reading.
        c = rm.NOTATIONS["C / C++"]
        self.assertIn(("comment", "line"), c["symbol_roles"]["//"])
        score, tier, msym, mrole, _ = rm.match_role("//", "clause_boundary", c)
        self.assertNotEqual(tier, "exact")
        self.assertNotEqual(mrole, "comment")
        # C expresses the utterance boundary with `;`, not with its `//` comment
        self.assertEqual(tier, "role_only")
        self.assertEqual((msym, mrole), (";", "clause_boundary"))
        self.assertEqual(score, rm.TIER_ROLE)
        # comment and clause_boundary are deliberately different families
        self.assertNotEqual(rm.ROLE_FAMILY["comment"],
                            rm.ROLE_FAMILY["clause_boundary"])

    def test_role_only_when_symbol_absent(self):
        # The book's :: has no symbol in Lisp, and Lisp has no field_sep role.
        lisp = rm.NOTATIONS["Lisp / S-expressions"]
        score, tier, *_ = rm.match_role("::", "field_sep", lisp)
        self.assertEqual((score, tier), (rm.TIER_NONE, "none"))
        # English punctuation has a field_sep role (:) but not the :: symbol.
        eng = rm.NOTATIONS["English punctuation"]
        score, tier, msym, _mrole, _ = rm.match_role("::", "field_sep", eng)
        self.assertEqual(tier, "role_only")
        self.assertEqual(msym, ":")
        self.assertEqual(score, rm.TIER_ROLE)

    def test_family_match(self):
        # Molecular `=` is a base-pair linker (compound_link), family-compatible
        # with the book's naming `=`.
        mol = rm.NOTATIONS["Molecular / genetic"]
        score, tier, _msym, mrole, _ = rm.match_role("=", "naming", mol)
        self.assertEqual(tier, "family")
        self.assertEqual(score, rm.TIER_FAMILY)
        self.assertEqual(mrole, "compound_link")

    def test_plusplus_is_c_increment_collision(self):
        c = rm.NOTATIONS["C / C++"]
        score, tier, _msym, mrole, _ = rm.match_role("++", "coupling", c)
        self.assertEqual(tier, "symbol_only")
        self.assertEqual(score, rm.TIER_SYMBOL)
        self.assertEqual(mrole, "arith")
        # Perl can express coupling (string concat `.`), so it scores higher.
        perl = rm.NOTATIONS["Perl"]
        pscore, ptier, _s, _r, _ = rm.match_role("++", "coupling", perl)
        self.assertEqual(ptier, "role_only")
        self.assertGreater(pscore, score)

    def test_double_angle_block_exact_in_c(self):
        c = rm.NOTATIONS["C / C++"]
        self.assertEqual(rm.match_role("<<", "block_open", c)[1], "exact")
        self.assertEqual(rm.match_role(">>", "block_close", c)[1], "exact")

    def test_molecular_record_terminator(self):
        mol = rm.NOTATIONS["Molecular / genetic"]
        _score, tier, _msym, mrole, mpos = rm.match_role(
            "//", "clause_boundary", mol)
        self.assertEqual(tier, "exact")
        self.assertEqual(mrole, "clause_boundary")
        self.assertEqual(mpos, "line")   # GenBank // is line-anchored
        # the book's // is inline -> positional mismatch
        self.assertEqual(rm.position_agree("inline", mpos), 0.25)

    def test_position_agree(self):
        self.assertEqual(rm.position_agree("inline", "inline"), 1.0)
        self.assertEqual(rm.position_agree("any", "inline"), 0.75)
        self.assertEqual(rm.position_agree("inline", "line"), 0.25)

    # -- metric helpers ----------------------------------------------------
    def test_jaccard(self):
        self.assertEqual(rm.jaccard({"a", "b"}, {"a", "b"}), 1.0)
        self.assertEqual(rm.jaccard({"a"}, {"b"}), 0.0)
        self.assertAlmostEqual(rm.jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_feature_agree(self):
        book = {"x": True, "y": False, "z": "partial"}
        self.assertEqual(rm.feature_agree(book, book), 1.0)
        self.assertEqual(rm.feature_agree(book, {"x": False, "y": True,
                                                 "z": "partial"}), 1 / 3)
        # partial counts as half
        self.assertEqual(rm.feature_agree({"z": True}, {"z": "partial"}), 0.5)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(rm.WEIGHTS.values()), 1.0)

    # -- scorecard ---------------------------------------------------------
    def test_scorecard_ranked_and_complete(self):
        self.assertEqual(len(self.rows), len(rm.NOTATIONS))
        self.assertEqual([r["rank"] for r in self.rows], list(range(1, 9)))
        totals = [r["total"] for r in self.rows]
        self.assertEqual(totals, sorted(totals, reverse=True))
        for r in self.rows:
            for k in ("role_cov", "exact_cov", "symbol_recog", "alphabet_cov",
                      "role_expressible", "jaccard", "feature", "position"):
                self.assertGreaterEqual(r[k], 0.0)
                self.assertLessEqual(r[k], 1.0)

    def test_c_and_perl_lead(self):
        self.assertEqual(self.rows[0]["notation"], "C / C++")
        self.assertEqual(self.rows[1]["notation"], "Perl")
        self.assertEqual(self.rows[-1]["notation"], "Regular expressions")
        self.assertGreater(self.rows[0]["role_cov"], 0.70)
        self.assertGreater(self.rows[0]["exact_cov"], 0.55)

    def test_c_contains_every_book_symbol(self):
        # C/C++'s punctuation set contains every book operator symbol.
        self.assertEqual(self.by_name["C / C++"]["alphabet_cov"], 1.0)
        # but not every occurrence is best explained by that symbol (e.g. //)
        self.assertLess(self.by_name["C / C++"]["symbol_recog"], 1.0)

    def test_alphabet_role_split(self):
        # C/Perl win on the symbol axis; English punctuation has zero symbol
        # overlap but high role expressibility -- the core finding.
        self.assertEqual(self.by_name["English punctuation"]["symbol_recog"], 0.0)
        self.assertEqual(self.by_name["English punctuation"]["exact_cov"], 0.0)
        self.assertGreater(
            self.by_name["English punctuation"]["role_expressible"], 0.70)

    def test_lisp_and_regex_low(self):
        self.assertLess(self.by_name["Lisp / S-expressions"]["role_cov"], 0.30)
        self.assertLess(self.by_name["Regular expressions"]["role_cov"], 0.30)

    def test_per_operator_best_matches(self):
        best = {c["op"]: max(self.rows,
                             key=lambda r: r["per_op"][c["op"]]["score"])
                for c in rm.BOOK_CHANNELS}
        self.assertEqual(best["::"]["notation"], "C / C++")
        self.assertEqual(best["="]["notation"], "C / C++")
        self.assertEqual(best["<<"]["notation"], "C / C++")
        self.assertEqual(best["//"]["notation"], "Molecular / genetic")
        # every operator has a non-zero best match
        for op, r in best.items():
            self.assertGreater(r["per_op"][op]["score"], 0.0, op)

    # -- null model + sensitivity -----------------------------------------
    def test_permutation_beats_chance(self):
        self.assertLess(self.perm["p_value"], 0.01)
        self.assertGreater(self.perm["z"], 2.0)
        self.assertGreater(self.perm["observed"], self.perm["null_mean"])

    def test_sensitivity_winner_stable(self):
        for key in ("baseline", "with_hyphens", "no_html_pages",
                    "drop_rare_ops"):
            self.assertEqual(self.sens[key], "C / C++", key)

    def test_no_html_pages_recount(self):
        counts = self.sens["no_html_pages_counts"]
        # removing the three raw-HTML OCR pages can only shrink the inventory
        self.assertLess(sum(counts.values()), 10949)
        self.assertGreater(sum(counts.values()), 10000)
        for op, n in counts.items():
            self.assertIn(op, EXPECTED_COUNTS)
            self.assertLessEqual(n, EXPECTED_COUNTS[op])

    def test_determinism(self):
        again = rm.scorecard(rm.BOOK_CHANNELS, rm.BOOK_FEATURES)
        self.assertEqual([(r["notation"], r["total"]) for r in self.rows],
                         [(r["notation"], r["total"]) for r in again])

    # -- artifacts ---------------------------------------------------------
    def test_artifacts_exist(self):
        for name in ("rosetta_report.md", "notation_signatures.tsv",
                     "operator_role_map.tsv", "scores.tsv",
                     "per_operator_best_match.tsv",
                     "positional_diagnostics.tsv", "structural_metrics.tsv",
                     "permutation.tsv", "reconciliation.tsv",
                     "diagnostics.tsv", "summary.json"):
            self.assertTrue(os.path.exists(os.path.join(OUT, name)), name)

    def test_summary_json(self):
        path = os.path.join(OUT, "summary.json")
        if not os.path.exists(path):
            self.skipTest("summary.json not generated")
        with open(path) as f:
            s = json.load(f)
        self.assertEqual(s["schema"], "rosetta-mapping/1")
        self.assertEqual(s["n_operator_tokens"], 10949)
        self.assertEqual(s["counts"], EXPECTED_COUNTS)
        self.assertEqual(s["winner"], "C / C++")
        self.assertEqual(s["runner_up"], "Perl")
        self.assertEqual(s["last"], "Regular expressions")
        self.assertEqual(len(s["scores"]), 8)
        self.assertEqual(s["scores"][0]["rank"], 1)
        self.assertIn("permutation", s)
        self.assertLess(s["permutation"]["p_value"], 0.01)
        self.assertEqual(s["delimiter_pairing"]["block"]["matched"], 230)
        self.assertEqual(s["delimiter_pairing"]["block"]["max_depth"], 30)
        self.assertEqual(s["weights"], rm.WEIGHTS)

    def test_reconciliation_tsv(self):
        path = os.path.join(OUT, "reconciliation.tsv")
        if not os.path.exists(path):
            self.skipTest("reconciliation.tsv not generated")
        with open(path) as f:
            rows = [line.rstrip("\n").split("\t") for line in f][1:]
        table = {r[0]: r[2] for r in rows}
        self.assertEqual(table["operator tokens (#10)"], "10949")
        self.assertEqual(table["`<<` (#10)"], "412")
        self.assertEqual(table["`<<`...`>>` matched spans (#12)"], "230")
        self.assertEqual(table["`<<`...`>>` max depth (#12)"], "30")
        self.assertEqual(table["free-standing ` - ` (#12)"], "177")

    def test_scores_tsv_matches_scorecard(self):
        path = os.path.join(OUT, "scores.tsv")
        if not os.path.exists(path):
            self.skipTest("scores.tsv not generated")
        with open(path) as f:
            rows = [line.rstrip("\n").split("\t") for line in f][1:]
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0][1], "C / C++")
        self.assertAlmostEqual(float(rows[0][2]), self.rows[0]["role_cov"],
                               places=4)

    def test_report_contains_all_notations_and_claim(self):
        path = os.path.join(OUT, "rosetta_report.md")
        if not os.path.exists(path):
            self.skipTest("rosetta_report.md not generated")
        with open(path) as f:
            text = f.read()
        for name in ("C / C++", "Perl", "Lisp / S-expressions", "HTML / XML",
                     "Markdown", "Regular expressions", "Molecular / genetic",
                     "English punctuation"):
            self.assertIn(name, text)
        self.assertIn("symbolic, not functional", text)
        self.assertIn("false friend", text)
        self.assertIn("C / C++", text)
        # the report must disclose that notation signatures are spec-based
        self.assertIn("spec-based", text)
        self.assertIn("permutation", text.lower())

    def test_diagnostics_flags_scope(self):
        path = os.path.join(OUT, "diagnostics.tsv")
        if not os.path.exists(path):
            self.skipTest("diagnostics.tsv not generated")
        with open(path) as f:
            text = f.read()
        self.assertIn("spec-based", text)
        self.assertIn("no ground truth", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
