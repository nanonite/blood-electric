#!/usr/bin/env python3
"""Regression tests for the operator n-gram / Markov model (Chainlink #11).

Run from the workspace root (inside the `marker` conda env):

  python work/test_operator_ngrams.py
  python -m unittest discover -s work -p 'test_*.py'

Checks that the headline quantities reported in `analysis/operator_ngrams/`
are reproducible from the layout corpus, that the transition matrices are
proper probability distributions, and that the reported randomness verdict
matches the computed evidence.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import operator_ngrams as on

SRC = os.path.join(ROOT, "analysis", "layout_recovery", "blood-electric.layout.md")
OUT = os.path.join(ROOT, "analysis", "operator_ngrams")


@unittest.skipUnless(os.path.exists(SRC), "layout corpus not present")
class OperatorNgramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = on.page_runs(SRC)
        cls.page_ops = {p: [op for run in runs for op in run]
                        for p, runs in cls.pages.items()}
        (cls.stream, cls.page_of, cls.same_run,
         cls.run_lengths) = on.build_stream(cls.pages)

    def test_token_and_run_counts(self):
        self.assertEqual(len(self.stream), 10949)
        self.assertEqual(len(set(self.stream)), 11)
        self.assertEqual(len(self.run_lengths), 9739)
        self.assertEqual(sum(self.run_lengths), 10949)
        # length 2/3/4 runs: 1024 + 87 + 4
        self.assertEqual(sum(1 for L in self.run_lengths if L >= 2), 1115)

    def test_transition_counts(self):
        c1 = on.build_counts(self.stream, 1)
        c2 = on.build_counts(self.stream, 2)
        self.assertEqual(sum(sum(c.values()) for c in c1.values()),
                         len(self.stream) - 1)
        self.assertEqual(sum(sum(c.values()) for c in c2.values()),
                         len(self.stream) - 2)

    def test_probabilities_normalise(self):
        for order in (1, 2):
            counts = on.build_counts(self.stream, order)
            self.assertTrue(counts)
            for counter in counts.values():
                total = sum(counter.values())
                self.assertGreater(total, 0)
                self.assertAlmostEqual(
                    sum(v / total for v in counter.values()), 1.0, places=9)

    def test_headline_numbers(self):
        H_uni, _ = on.unigram_entropy(self.stream)
        H1, _ = on.weighted_cond_entropy(on.build_counts(self.stream, 1))
        H2, _ = on.weighted_cond_entropy(on.build_counts(self.stream, 2))
        mi, G, _M = on.mutual_information_1st(self.stream)
        self.assertAlmostEqual(2 ** H_uni, 6.675, delta=0.01)
        self.assertAlmostEqual(2 ** H1, 6.097, delta=0.01)
        self.assertAlmostEqual(2 ** H2, 5.310, delta=0.01)
        self.assertAlmostEqual(mi, 0.1305, delta=0.001)
        self.assertAlmostEqual(G, 1980.5, delta=5.0)

    def test_heldout_and_chrono_generalise(self):
        uni, _ = on.held_out_perplexity(self.page_ops, 0)
        o1, n1 = on.held_out_perplexity(self.page_ops, 1)
        o2, _ = on.held_out_perplexity(self.page_ops, 2)
        self.assertGreater(n1, 0)
        self.assertLess(o1, uni)
        self.assertLess(o2, o1)
        cu, _ = on.held_out_perplexity_chrono(self.page_ops, 0)
        c1, _ = on.held_out_perplexity_chrono(self.page_ops, 1)
        self.assertLess(c1, cu)

    def test_summary_artifact(self):
        path = os.path.join(OUT, "summary.json")
        if not os.path.exists(path):
            self.skipTest("summary.json not generated; run work/operator_ngrams.py")
        with open(path) as f:
            summary = json.load(f)
        self.assertEqual(summary["n_operator_tokens"], 10949)
        self.assertEqual(summary["vocab"], 11)
        self.assertTrue(summary["verdict"].startswith("PARTIALLY IDIOMATIC"))
        for key in ("heldout_perplexity_chrono", "heldout_gain_vs_unigram_chrono",
                    "sensitivity_leave_out"):
            self.assertIn(key, summary)

    def test_leave_out_sensitivity(self):
        artifact = on.html_artifact_pages(SRC)
        self.assertEqual(artifact, {92, 158, 219})
        full = on.stream_stats(self.pages)
        drop = on.stream_stats(self.pages, {92})
        self.assertAlmostEqual(full["intra"]["gain_fraction"], 0.198, delta=0.01)
        # page 92 (HTML `</a>`) inflates the intra-run gain; removing it ~halves it
        self.assertLess(drop["intra"]["gain_fraction"], 0.12)
        self.assertGreater(drop["intra"]["gain_fraction"], 0.05)
        # sequential dependence survives the removal
        self.assertGreater(drop["mi_bits"], 0.08)

    def test_first_order_matrix_sums(self):
        path = os.path.join(OUT, "transition_1st_order.tsv")
        if not os.path.exists(path):
            self.skipTest("matrix not generated; run work/operator_ngrams.py")
        sums = {}
        with open(path) as f:
            next(f)  # header
            for line in f:
                cur, _nxt, _cnt, p, _lp = line.rstrip("\n").split("\t")
                sums[cur] = sums.get(cur, 0.0) + float(p)
        self.assertTrue(sums)
        for cur, total in sums.items():
            self.assertAlmostEqual(total, 1.0, places=3,
                                   msg=f"P(.|{cur}) sums to {total}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
