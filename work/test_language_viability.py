#!/usr/bin/env python3
"""Regression tests for the within-text language viability assessment
(Chainlink #56, revised scope).

Run from the workspace root (inside the `marker` conda env):

  python work/test_language_viability.py
  python -m unittest discover -s work -p 'test_*.py'

Two tiers:

* **Fixture tests** (fast, synthetic): the statistics are checked
  against hand-computed values, the nulls are checked to preserve
  exactly the marginals they claim to preserve, and the harness is
  checked to *detect* a synthetic language and *fail* a synthetic
  non-language (the V0 negative-control property).
* **Artifact tests** (guarded by skipUnless): the frozen workspace
  numbers reconcile (#10 operators, #17 units, #33 clauses), the
  six checks run end-to-end on the real book in the pre-registered
  direction, and the gloss passages/termbase are grounded in the
  artifacts.

The end-to-end artifact tests assert *direction* (book value beats
the null in the pre-registered direction), not the p-gate: with
n_nulls=8 the add-one p floor is 1/9 > 0.05, so the gate cannot
fire at that draw count by construction.
"""
from __future__ import annotations

import collections
import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import language_viability as lv
import frame_mining as fm

LAYOUT = os.path.join(ROOT, "analysis", "layout_recovery",
                      "blood-electric.layout.md")
UNITS = os.path.join(ROOT, "analysis", "tokenizer", "units.tsv")
TOKENS = os.path.join(ROOT, "analysis", "tokenizer", "tokens.tsv")
CLAUSES = os.path.join(ROOT, "analysis", "clauses", "clauses.tsv")


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

class StatisticsTest(unittest.TestCase):
    def test_spearman_perfect(self):
        self.assertAlmostEqual(lv.spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(lv.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_spearman_ties_and_empty(self):
        # constant vectors: no ranks to correlate
        self.assertEqual(lv.spearman([1, 1, 1], [2, 2, 2]), 0.0)
        self.assertEqual(lv.spearman([], []), 0.0)
        # monotone but with ties
        self.assertAlmostEqual(
            lv.spearman([1, 2, 2, 3], [10, 20, 20, 30]), 1.0)

    def test_add_one_p_lower_better(self):
        # book better than every null: p = 1/(B+1)
        self.assertAlmostEqual(lv.add_one_p(1.0, [2.0, 3.0, 4.0]), 0.25)
        # book worse than every null: p = 1.0
        self.assertAlmostEqual(lv.add_one_p(5.0, [2.0, 3.0, 4.0]), 1.0)
        # one null ties the book
        self.assertAlmostEqual(lv.add_one_p(2.0, [2.0, 3.0, 4.0]), 0.5)

    def test_add_one_p_higher_better(self):
        self.assertAlmostEqual(lv.add_one_p(5.0, [2.0, 3.0, 4.0],
                                             lower_better=False), 0.25)
        self.assertAlmostEqual(lv.add_one_p(1.0, [2.0, 3.0, 4.0],
                                             lower_better=False), 1.0)

    def test_fmt_p(self):
        # 4 decimals, but never a useless "0.0000" for a tiny p
        self.assertEqual(lv._fmt_p(0.0244), "0.0244")
        self.assertEqual(lv._fmt_p(0.123456), "0.1235")
        self.assertEqual(lv._fmt_p(0.000084), "8.4e-05")
        self.assertEqual(lv._fmt_p(0.0), "0")
        self.assertEqual(lv._fmt_p(None), "-")

    def test_sign_test_p(self):
        # 40 of 40: two-sided p ~ 2^-39
        p = lv.sign_test_p(40, 40)
        self.assertLess(p, 1e-10)
        # balanced: p = 1.0
        self.assertAlmostEqual(lv.sign_test_p(10, 20), 1.0)
        self.assertAlmostEqual(lv.sign_test_p(0, 0), 1.0)

    def test_dice_topk(self):
        a = collections.Counter({"x": 10, "y": 9, "z": 8})
        b = collections.Counter({"x": 5, "y": 4, "z": 3})
        self.assertAlmostEqual(lv.dice_topk(a, b, 2), 1.0)
        c = collections.Counter({"u": 5, "v": 4, "w": 3})
        self.assertAlmostEqual(lv.dice_topk(a, c, 2), 0.0)
        # partial overlap: {x,y} vs {x,u} -> 2*1/(2+2) = 0.5
        d = collections.Counter({"x": 5, "u": 4})
        self.assertAlmostEqual(lv.dice_topk(a, d, 2), 0.5)


# ---------------------------------------------------------------------------
# cross-entropy machinery (hand-computed fixtures)
# ---------------------------------------------------------------------------

class CrossEntropyTest(unittest.TestCase):
    def test_heldout_ce_hand_computed(self):
        # train [a a b]: bigram (a,a)=1, (a,b)=1.
        # order-1, lam=1.0 (pure ML): P(b|a) = P(a|a) = 1/2.
        # test [a b]: CE = -log2(1/2) = 1.0 bit exactly.
        seqmap = {"tr": [["a", "a", "b"]], "te": [["a", "b"]]}
        ce = lv.heldout_ce(seqmap, ["tr"], ["te"], 1, 1.0)
        self.assertAlmostEqual(ce, 1.0, places=6)

    def test_heldout_ce_order0_is_unigram(self):
        # order-0 with lam=1.0: P(w) = ML unigram.
        # train [a a b] -> P(a)=2/3, P(b)=1/3.
        # test [a b]: CE = -(log2(2/3) + log2(1/3)) / 2
        seqmap = {"tr": [["a", "a", "b"]], "te": [["a", "b"]]}
        ce = lv.heldout_ce(seqmap, ["tr"], ["te"], 0, 1.0)
        expected = -(math.log2(2.0 / 3.0) + math.log2(1.0 / 3.0)) / 2.0
        self.assertAlmostEqual(ce, expected, places=6)

    def test_heldout_ce_smoothing_helps_oov(self):
        # a test transition whose context is unseen in train:
        # the ML term is 0, so the interpolated probability is
        # the smoothed unigram mass -- finite, not inf.
        # train [a a b] -> p0(a) = 2/3; lam=0.5 -> P(a|q) = 1/3.
        seqmap = {"tr": [["a", "a", "b"]], "te": [["q", "a"]]}
        ce = lv.heldout_ce(seqmap, ["tr"], ["te"], 1, 0.5)
        self.assertAlmostEqual(ce, -math.log2(1.0 / 3.0), places=6)

    def test_tune_lambda_picks_grid_point(self):
        seqmap = {"tr": [["a", "a", "b"]], "va": [["a", "b"]]}
        lam, ce = lv.tune_lambda(seqmap, ["tr"], ["va"], 1)
        self.assertIn(lam, lv.LAM_GRID)
        self.assertTrue(ce > 0.0)


# ---------------------------------------------------------------------------
# null constructors preserve the marginals they claim
# ---------------------------------------------------------------------------

class NullsTest(unittest.TestCase):
    def test_within_page_shuffle_preserves_marginals(self):
        pages = {1: ["a", "b", "c", "a"], 2: ["d", "d", "e"]}
        rng = random.Random(0)
        sh = lv.within_page_shuffle(pages, rng)
        for p in pages:
            self.assertEqual(collections.Counter(pages[p]),
                             collections.Counter(sh[p]))
            self.assertEqual(sorted(pages[p]), sorted(sh[p]))

    def test_type_permutation_is_bijection(self):
        types = ["a", "b", "c", "d"]
        rng = random.Random(1)
        pmap = lv.type_permutation(types, rng)
        self.assertEqual(sorted(pmap.values()), sorted(types))
        self.assertEqual(len(set(pmap.values())), len(types))

    def test_within_run_shuffle_preserves_run_lengths(self):
        pages_runs = {1: [["::", "//"], ["="]], 2: [["<<", ">>", "::"]]}
        rng = random.Random(2)
        sh = lv.within_run_shuffle(pages_runs, rng)
        for p in pages_runs:
            flat = [op for run in pages_runs[p] for op in run]
            self.assertEqual(sorted(flat), sorted(sh[p]))
            self.assertEqual(len(flat), len(sh[p]))

    def test_within_clause_shuffle_preserves_tokens(self):
        clauses = [{"clause_id": 0, "page": 1, "opener": "//",
                    "container": "block", "n_words_clean": 4,
                    "tokens": ["of", "the", "artificial", "sun"]}]
        rng = random.Random(3)
        sh = lv.within_clause_shuffle(clauses, rng)
        self.assertEqual(collections.Counter(clauses[0]["tokens"]),
                         collections.Counter(sh[0]["tokens"]))

    def test_within_unit_part_shuffle_preserves_parts(self):
        seqmap = {1: [["suck", "=", "blood"], ["boy", "-", "roid"]]}
        rng = random.Random(4)
        sh = lv.within_unit_part_shuffle(seqmap, rng)
        for i, seq in enumerate(seqmap[1]):
            self.assertEqual(collections.Counter(seq),
                             collections.Counter(sh[1][i]))

    def test_unit_part_seq(self):
        self.assertEqual(lv.unit_part_seq("suck=blood"),
                         ["suck", "=", "blood"])
        self.assertEqual(lv.unit_part_seq("boy-roid"),
                         ["boy", "-", "roid"])
        self.assertEqual(lv.unit_part_seq("gene=TV/spasm"),
                         ["gene", "=", "TV", "/", "spasm"])
        self.assertEqual(lv.unit_part_seq("plain"), ["plain"])


# ---------------------------------------------------------------------------
# the harness detects language and fails non-language (V0 property)
# ---------------------------------------------------------------------------

class HarnessTest(unittest.TestCase):
    @staticmethod
    def _pages(stream, chunk=50):
        return {i: stream[i * chunk:(i + 1) * chunk]
                for i in range((len(stream) + chunk - 1) // chunk)}

    def test_detects_synthetic_language(self):
        # A deterministic 3-cycle: a->b->c->a.  A bigram learner
        # must read held-out pages at ~0 bits; shuffled nulls at
        # ~log2(3).
        stream = ["a", "b", "c"] * 120
        pages = self._pages(stream)
        allp = sorted(pages)
        train = allp[::2]
        test = allp[1::2]
        flat = {p: [pages[p]] for p in allp}
        lam, _ = lv.tune_lambda(flat, train[::2], train[1::2], 1)
        book = lv.heldout_ce(flat, train, test, 1, lam)
        rng = random.Random(lv.SEED)
        nulls = []
        for _ in range(40):
            sh = lv.within_page_shuffle(pages, rng)
            nulls.append(lv.heldout_ce({p: [sh[p]] for p in allp},
                                          train, test, 1, lam))
        self.assertLess(book, 0.5)                 # near-perfect prediction
        self.assertGreater(min(nulls), 1.0)        # nulls near log2(3)
        self.assertLess(book, min(nulls))
        self.assertLess(lv.add_one_p(book, nulls), 0.05)

    def test_fails_synthetic_non_language(self):
        # IID uniform over {a,b,c}: no learnable order.  The V0
        # decision (book < null min AND p < 0.05) must be False.
        rng = random.Random(1234)
        stream = rng.choices(["a", "b", "c"], k=360)
        pages = self._pages(stream)
        allp = sorted(pages)
        train = allp[::2]
        test = allp[1::2]
        flat = {p: [pages[p]] for p in allp}
        lam, _ = lv.tune_lambda(flat, train[::2], train[1::2], 1)
        book = lv.heldout_ce(flat, train, test, 1, lam)
        nulls = []
        for _ in range(40):
            sh = lv.within_page_shuffle(pages, rng)
            nulls.append(lv.heldout_ce({p: [sh[p]] for p in allp},
                                          train, test, 1, lam))
        decision = book < min(nulls) and lv.add_one_p(book, nulls) < 0.05
        self.assertFalse(decision,
                         "harness found 'language' in IID noise")

    def test_frame_transfer_detects_constructions(self):
        # Train: 'of the <filler> sun/embryo/city' x6 each (3 distinct
        # fillers, 18 clauses -> the 'of the <_>' frame qualifies).
        # Test: the same skeleton with a novel filler.
        fillers = ["artificial", "drug", "cadaver"]
        train = []
        for f in fillers:
            for _ in range(6):
                train.append({"tokens": ["of", "the", f, "sun"]})
        test = [{"tokens": ["of", "the", "ice", "nebula"]}
                for _ in range(4)]
        frames = fm.mine_frames(train)
        skeletons = [" ".join(f["skeleton"]) for f in frames]
        self.assertIn("of the \u27e8_\u27e9", skeletons)
        book = lv.frame_coverage_frac(test, frames)
        self.assertEqual(book, 1.0)
        rng = random.Random(lv.SEED)
        nulls = []
        for _ in range(20):
            sh = lv.within_clause_shuffle(train, rng)
            fsh = fm.mine_frames(sh)
            nulls.append(lv.frame_coverage_frac(test, fsh))
        self.assertGreater(book, max(nulls))


# ---------------------------------------------------------------------------
# verdict logic
# ---------------------------------------------------------------------------

class VerdictTest(unittest.TestCase):
    @staticmethod
    def _row(check, passed):
        return {"check": check, "passed": passed, "book": 0.0,
                "null_mean": 0.0, "null_min": 0.0, "null_max": 0.0,
                "p_value": 0.0, "statistic": "", "split": "",
                "criterion": ""}

    def test_all_pass_is_viable(self):
        rows = [self._row(c, True) for c in
                ("V1", "V2", "V3", "V4a", "V4b", "V5", "V6")]
        v, n = lv.verdict(rows)
        self.assertEqual(v, "VIABLE (BOUNDED)")
        self.assertEqual(n, 6)

    def test_v4_requires_both_statistics(self):
        # V4a passes, V4b fails -> V4 counts as a failed check.
        rows = [self._row("V1", True), self._row("V2", True),
                self._row("V3", True), self._row("V4a", True),
                self._row("V4b", False), self._row("V5", True),
                self._row("V6", True)]
        v, n = lv.verdict(rows)
        self.assertEqual(n, 5)          # V1,V2,V3,V5,V6
        self.assertEqual(v, "VIABLE (BOUNDED)")

    def test_partial(self):
        rows = [self._row("V1", True), self._row("V2", False),
                self._row("V3", False), self._row("V4a", False),
                self._row("V4b", False), self._row("V5", True),
                self._row("V6", True)]
        v, n = lv.verdict(rows)
        self.assertEqual(n, 3)
        self.assertEqual(v, "PARTIALLY VIABLE")

    def test_not_viable(self):
        rows = [self._row(c, c == "V5") for c in
                ("V1", "V2", "V3", "V4a", "V4b", "V5", "V6")]
        v, n = lv.verdict(rows)
        self.assertEqual(n, 1)
        self.assertEqual(v, "NOT VIABLE")


# ---------------------------------------------------------------------------
# passage glossing
# ---------------------------------------------------------------------------

class PassageTest(unittest.TestCase):
    def test_passage_span(self):
        body = "xx // foo bar // yy // baz qux //"
        self.assertEqual(lv.passage_span(body, "foo bar", "baz qux"),
                         (6, 30))
        self.assertIsNone(lv.passage_span(body, "missing", "baz qux"))
        self.assertIsNone(lv.passage_span(body, "foo bar", "also missing"))

    def test_render_variant(self):
        toks = [
            {"gap_json": json.dumps(""), "text_json": json.dumps("The"),
             "kind": "WORD", "start": "0"},
            {"gap_json": json.dumps(" "), "text_json": json.dumps("//"),
             "kind": "OP", "start": "3"},
            {"gap_json": json.dumps(""), "text_json": json.dumps("soul"),
             "kind": "WORD", "start": "5"},
            {"gap_json": json.dumps(""), "text_json": json.dumps("/"),
             "kind": "OP", "start": "10"},
            {"gap_json": json.dumps(""), "text_json": json.dumps("gram"),
             "kind": "WORD", "start": "11"},
        ]
        v3 = lv.render_variant(toks, "v3")
        v4 = lv.render_variant(toks, "v4")
        self.assertEqual(v3, "The soul gram")
        self.assertEqual(v4, "The. soul. gram")


# ---------------------------------------------------------------------------
# artifact tests (the real book)
# ---------------------------------------------------------------------------

@unittest.skipUnless(os.path.exists(UNITS), "tokenizer units not present")
class ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.bodies, cls.pages, cls.page_units, cls.unit_meta,
         is_markup_fn, cls.screening) = lv.load_book()
        # a plain function stored on the class would bind as a
        # method on self access; pin it as a staticmethod
        cls.is_markup = staticmethod(is_markup_fn)
        cls.allp = sorted(cls.pages)
        cls.train = [p for p in cls.allp if p % 2 == 0]
        cls.test = [p for p in cls.allp if p % 2 == 1]

    def setUp(self):
        # a scratch directory for the tests that exercise the writers
        self.tmp = tempfile.TemporaryDirectory(prefix="lv-test-")
        self.addCleanup(self.tmp.cleanup)

    def test_reconciliation_operators(self):
        pages_runs, pages_ops = lv.load_operator_pages()
        stream, _page_of, _same, _runs = ng_build_stream(pages_runs)
        self.assertEqual(len(stream), 10949)      # frozen #10 count
        self.assertEqual(len(pages_ops), 225)

    def test_reconciliation_units(self):
        n = sum(len(v) for v in self.pages.values())
        self.assertEqual(n, 74077)                # frozen #17 screened count
        self.assertEqual(len(self.allp), 225)
        # train/test halves are disjoint and non-empty
        self.assertEqual(len(self.train) + len(self.test), 225)
        self.assertEqual(len(set(self.train) & set(self.test)), 0)
        self.assertTrue(min(self.train) >= 1 and max(self.test) <= 225)

    def test_reconciliation_clauses(self):
        clauses = lv.load_clauses()
        self.assertEqual(len(clauses), 5614)      # frozen #33 count
        self.assertTrue(all(c["tokens"] is not None for c in clauses))

    def test_checks_run_in_preregistered_direction(self):
        """End-to-end on the real book: every check's book value beats
        its null in the pre-registered direction (direction, not the
        p-gate -- n_nulls=8 cannot fire the gate by construction)."""
        rng = random.Random(lv.SEED)
        flat = {p: [self.pages[p]] for p in self.allp}

        # V1
        lam, _ = lv.tune_lambda(flat, self.train[::2], self.train[1::2], 1)
        v1 = lv.heldout_ce(flat, self.train, self.test, 1, lam)
        sh = lv.within_page_shuffle(self.pages, rng)
        n1a = lv.heldout_ce({p: [sh[p]] for p in self.allp},
                             self.train, self.test, 1, lam)
        types = sorted({w for p in self.allp for w in self.pages[p]})
        pmap = lv.type_permutation(types, rng)
        tm = {p: [[pmap[w] for w in self.pages[p]]] for p in self.test}
        full = dict(flat)
        full.update(tm)
        n1b = lv.heldout_ce(full, self.train, self.test, 1, lam)
        self.assertLess(v1, n1a)
        self.assertLess(v1, n1b)

        # V2
        pages_runs, pages_ops = lv.load_operator_pages()
        ls, nt = lv.ng._held_out_perplexity_split(
            pages_ops, 1, self.train, self.test, lv.OPERATOR_LAM)
        v2 = -ls / nt
        shr = lv.within_run_shuffle(pages_runs, rng)
        ls, nt = lv.ng._held_out_perplexity_split(
            shr, 1, self.train, self.test, lv.OPERATOR_LAM)
        self.assertLess(v2, -ls / nt)

        # V3
        clauses = lv.load_clauses()
        tr = [c for c in clauses if c["page"] % 2 == 0]
        te = [c for c in clauses if c["page"] % 2 == 1]
        frames = fm.mine_frames(tr)
        v3 = lv.frame_coverage_frac(te, frames)
        shc = lv.within_clause_shuffle(tr, rng)
        v3_null = lv.frame_coverage_frac(te, fm.mine_frames(shc))
        self.assertGreater(v3, v3_null)

        # V4
        tr_types = {w for p in self.train for w in self.pages[p]}
        te_toks = [w for p in self.test for w in self.pages[p]]
        v4a = sum(1 for w in te_toks if w not in tr_types) / len(te_toks)
        pmap = lv.type_permutation(types, rng)
        v4a_null = sum(1 for p in self.test for w in self.pages[p]
                       if pmap[w] not in tr_types) / len(te_toks)
        self.assertLess(v4a, v4a_null)
        ca = collections.Counter(w for p in self.train for w in self.pages[p])
        cb = collections.Counter(w for p in self.test for w in self.pages[p])
        shared = sorted(set(ca) & set(cb))
        v4b = lv.spearman([ca[w] for w in shared], [cb[w] for w in shared])
        cb2 = collections.Counter(pmap[w] for p in self.test
                                  for w in self.pages[p])
        sh_shared = sorted(set(ca) & set(cb2))
        v4b_null = lv.spearman([ca[w] for w in sh_shared],
                               [cb2[w] for w in sh_shared])
        self.assertGreater(v4b, v4b_null)

        # V5
        freq = collections.Counter(w for p in self.allp
                                   for w in self.pages[p])
        terms = [w for w, _c in freq.most_common()
                 if w not in lv.FUNCTION_WORDS and len(w) > 2][:12]
        pa_all = lv.collocate_profiles(self.pages, self.train, terms)
        pb_all = lv.collocate_profiles(self.pages, self.test, terms)
        same, mism = [], []
        for idx, t in enumerate(terms):
            pa, pb = pa_all[t], pb_all[t]
            if sum(pa.values()) < 20:
                continue
            same.append(lv.dice_topk(pa, pb, lv.TOP_K))
            pb2 = pb_all[terms[(idx + 5) % len(terms)]]
            if sum(pb2.values()) < 20:
                continue
            mism.append(lv.dice_topk(pa, pb2, lv.TOP_K))
        med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
        self.assertGreater(med(same), med(mism))
        self.assertGreater(sum(1 for a, b in zip(same, mism) if a > b),
                           len(same) // 2)

        # V6
        seqmap = {p: [lv.unit_part_seq(u) for u in self.pages[p]]
                  for p in self.allp}
        v6 = lv.heldout_ce(seqmap, self.train, self.test, 1, 0.5)
        shp = lv.within_unit_part_shuffle(seqmap, rng)
        v6_null = lv.heldout_ce(shp, self.train, self.test, 1, 0.5)
        self.assertLess(v6, v6_null)

    def test_negative_control_fails_on_real_book(self):
        rng = random.Random(lv.SEED)
        n = 20
        sh = lv.within_page_shuffle(self.pages, rng)
        flat_sh = {p: [sh[p]] for p in self.allp}
        lam, _ = lv.tune_lambda(flat_sh, self.train[::2],
                                self.train[1::2], 1)
        book = lv.heldout_ce(flat_sh, self.train, self.test, 1, lam)
        nulls = []
        for _ in range(n):
            sh2 = lv.within_page_shuffle(sh, rng)
            nulls.append(lv.heldout_ce({p: [sh2[p]] for p in self.allp},
                                          self.train, self.test, 1, lam))
        # the shuffled book must NOT beat its own null
        self.assertFalse(book < min(nulls)
                         and lv.add_one_p(book, nulls) < 0.05)

    def test_checks_are_deterministic(self):
        rng1 = random.Random(lv.SEED)
        rng2 = random.Random(lv.SEED)
        flat = {p: [self.pages[p]] for p in self.allp}
        lam, _ = lv.tune_lambda(flat, self.train[::2],
                                self.train[1::2], 1)
        book = lv.heldout_ce(flat, self.train, self.test, 1, lam)
        n1 = [lv.heldout_ce({p: [lv.within_page_shuffle(self.pages, rng1)[p]]
                              for p in self.allp},
                             self.train, self.test, 1, lam)
              for _ in range(10)]
        n2 = [lv.heldout_ce({p: [lv.within_page_shuffle(self.pages, rng2)[p]]
                              for p in self.allp},
                             self.train, self.test, 1, lam)
              for _ in range(10)]
        self.assertEqual(n1, n2)
        self.assertAlmostEqual(book, 8.3379, places=2)

    def test_gloss_passages_are_located_and_grounded(self):
        entity_glossary = lv.load_entity_glossary()
        clauses = lv.load_clauses()
        termbase, _det = lv.build_termbase(self.pages, self.allp,
                                             entity_glossary,
                                             self.screening, clauses)
        gloss_of = {r["term"]: r for r in termbase}
        passages, token_rows = lv.gloss_passages(
            self.bodies, self.page_units, self.unit_meta, self.is_markup,
            self.screening, entity_glossary, termbase, [])
        # three passages, on the pages the evidence says
        self.assertEqual([ps["page"] for ps in passages], [4, 4, 6])
        for ps in passages:
            self.assertGreater(len(ps["rows"]), 5)
            self.assertIn("//", ps["original"] + " ")
            self.assertTrue(ps["v3_space"])
            self.assertTrue(ps["v4_period"])
        # every WORD token in the gloss resolves to a #17 uid
        word_rows = [r for r in token_rows if r["kind"] == "WORD"]
        self.assertTrue(word_rows)
        self.assertTrue(all(r["uid"] for r in word_rows))
        # the curated terms are present with computed evidence
        for t in ("soul/gram", "boy-roid", "tokage", "ndro",
                  "artificial-sun", "fecundates"):
            self.assertIn(t, gloss_of, t)
            self.assertGreater(int(gloss_of[t]["n_occurrences"]), 0)
            self.assertIn(gloss_of[t]["confidence"],
                          ("high", "medium", "low"))
            self.assertTrue(gloss_of[t]["gloss"])
        # confidence levels are only the pre-registered four
        self.assertTrue(all(r["confidence"] in
                            ("high", "medium", "low", "unresolved")
                            for r in token_rows))

    def test_termbase_reconciles_with_unit_counts(self):
        entity_glossary = lv.load_entity_glossary()
        clauses = lv.load_clauses()
        termbase, _det = lv.build_termbase(self.pages, self.allp,
                                             entity_glossary,
                                             self.screening, clauses)
        freq = collections.Counter(w for p in self.allp
                                   for w in self.pages[p])
        for r in termbase:
            self.assertEqual(int(r["n_occurrences"]), freq[r["term"]])
            self.assertGreater(int(r["n_pages"]), 0)
            self.assertLessEqual(int(r["first_page"]),
                                 max(self.allp))

    def test_residue_ledger_reconciles(self):
        ledger, oov_types, _oov_pages = lv.residue_ledger(
            self.pages, self.allp, self.train, self.test, self.screening)
        n_test_tokens = sum(len(self.pages[p]) for p in self.test)
        self.assertEqual(sum(oov_types.values()),
                         sum(1 for p in self.test
                             for w in self.pages[p]
                             if w not in {x for q in self.train
                                          for x in self.pages[q]}))
        self.assertLess(sum(oov_types.values()), n_test_tokens)
        self.assertTrue(ledger)
        # every ledger row cites its class
        self.assertTrue(all(r["class"] for r in ledger))

    def test_slash_recount_matches_frozen_node_counts(self):
        """The `/` dual-role split is measured, and reproduces #14.

        Regression: the `/` gloss once carried hard-coded literals
        (1,033 / 459) that matched neither this recount nor #14's frozen
        node-level 952/540, while the residue ledger quoted the correct
        numbers -- two contradictory claims in one deliverable.  Both
        now read the same computed recount.
        """
        link, slash = lv.slash_role_counts()
        # #14's numbers, read from #14's own artifact (literals only as a
        # fallback for a workspace without it)
        g = lv.grammar_counts()
        self.assertEqual(link, g["n_link_slash"])       # #14 Link nodes
        self.assertEqual(slash, g["n_slash"])            # #14 Slash nodes
        # every `/` OP token is classified exactly once
        self.assertEqual(link + slash,
                         sum(1 for r in cs_rows(lv.TOKENS)
                             if r["text"] == "/" and r["kind"] == "OP"))

    def test_grammar_counts_come_from_the_artifact(self):
        """The frozen `/` split is read from #14's summary.json, not
        restated here.

        Regression: literals in this module are a second copy of #14's
        counts, so a rebuild of #14 would leave the deliverable quietly
        stale.  The literals remain only as a fallback and must equal the
        artifact while the artifact is present.
        """
        if not lv.GRAMMAR.exists():
            self.skipTest("#14 summary.json not present")
        with lv.GRAMMAR.open(encoding="utf-8") as fh:
            d = json.load(fh)
        self.assertEqual(lv.grammar_counts()["n_link_slash"],
                         d["n_link_slash"])
        self.assertEqual(lv.grammar_counts()["n_slash"], d["n_slash"])
        # and the fallback literals agree with it on this workspace
        self.assertEqual((lv.FROZEN_SLASH_LINK, lv.FROZEN_SLASH_BOUND),
                         (d["n_link_slash"], d["n_slash"]))

    def test_loop_pages_cite_their_own_repeat(self):
        """Each recognition-loop page is cited with its own evidence.

        Regression: every OCR-loop row carried page 89's repeat ('of the
        second of' x145), so pages 182 and 208 were credited with a loop
        they do not have.  The ledger now quotes #4's per-page figures.
        """
        loops = lv.loop_page_evidence()
        for p in lv.OCR_LOOP_PAGES:
            self.assertIn(p, loops, "no #4 evidence for page %d" % p)
        reps = {p: loops[p][0] for p in lv.OCR_LOOP_PAGES}
        # the three pages have *different* loops, so a single repeated
        # string in the ledger would be a copy-paste error
        self.assertEqual(len(set(reps.values())), len(reps))
        ledger, _o, _p = lv.residue_ledger(
            self.pages, self.allp, self.train, self.test, self.screening)
        rows = {int(r["pages"]): r["detail"] for r in ledger
                if r["class"] == "OCR recognition loop"}
        for p, (n_rep, top, _over) in ((p, loops[p]) for p in
                                       lv.OCR_LOOP_PAGES):
            self.assertIn("runs %s times" % n_rep, rows[p])
            self.assertIn(top, rows[p])

    def test_kwic_examples_are_cited_and_page_numbered(self):
        """#26's contextual examples reach the termbase, page-cited."""
        ex = lv.load_kwic_examples(max_per_term=2)
        self.assertTrue(ex, "#26 concordance is empty")
        for term in ("soul/gram", "boy-roid", "ndro"):
            self.assertIn(term, ex)
            self.assertLessEqual(len(ex[term]), 2)
            for ctx in ex[term]:
                self.assertRegex(ctx, r"^p\d+: ")
        entity_glossary = lv.load_entity_glossary()
        clauses = lv.load_clauses()
        termbase, det = lv.build_termbase(self.pages, self.allp,
                                          entity_glossary,
                                          self.screening, clauses,
                                          kwic_examples=ex)
        by = {r["term"]: r for r in termbase}
        self.assertTrue(by["soul/gram"]["kwic_examples_26"])
        self.assertIn("p", by["soul/gram"]["kwic_examples_26"])
        self.assertEqual(det["n_terms_with_kwic"],
                         sum(1 for r in termbase
                             if r["kwic_examples_26"]))
        self.assertGreater(det["n_terms_with_kwic"], 0)

    def test_true_coinage_residue_is_filtered_not_assumed(self):
        """The 'attested once' residue row filters, and says both counts.

        Regression: the row filtered only on #19 category, so its "a
        single attestation" wording was an unverified claim -- it happens
        to hold on this book, which is exactly why it must be checked in
        code rather than in the reader's trust.
        """
        ledger, oov, _p = lv.residue_ledger(
            self.pages, self.allp, self.train, self.test, self.screening)
        row = next(r for r in ledger if r["class"] == "unexplained residue")
        coins = [w for w in oov
                 if (self.screening.get(w) or {}).get("category")
                 == "true_coinage"]
        n_once = sum(1 for w in coins if oov[w] == 1)
        self.assertIn("%d of %d" % (n_once, len(coins)), row["item"])
        self.assertLessEqual(n_once, len(coins))

    def test_sensitivity_uses_the_scored_draw_count(self):
        """D3 must not report a marginal p because of a small B.

        Regression: the sensitivity run hard-coded 20 draws while the
        scored checks used 40, so a clean result could only ever reach
        p = 1/21 = 0.0476 -- just under the 0.05 gate -- purely because
        of the draw count.
        """
        n = 6
        sens = lv.sensitivity(self.pages, self.allp, [92, 158, 219],
                              n_nulls=n)
        self.assertEqual(sens["n_nulls"], n)
        # add-one empirical p over B draws: 1/(B+1) is the floor, so a
        # small B can only ever report a *larger* p for the same result
        self.assertGreaterEqual(sens["p_value"], 1.0 / (n + 1))
        self.assertLessEqual(sens["p_value"], 1.0)
        self.assertLess(sens["test_ce"], sens["null_min"])
        # the default is the same draw count the scored checks use
        self.assertEqual(lv.sensitivity.__defaults__[0], lv.N_NULLS)
        # and the artifact records it, so the add-one floor is auditable
        lv.build(out=Path(self.tmp.name), n_nulls=2)
        row = lv.cs.read_tsv(Path(self.tmp.name) / "sensitivity.tsv")[0]
        self.assertEqual(row["n_nulls"], "2")

    def test_slash_gloss_uses_the_measured_counts(self):
        """The `/` gloss must carry the recount, not a stale literal."""
        link, slash = lv.slash_role_counts()
        templ = lv.OPERATOR_GLOSSES["/"]
        self.assertIn("{link}", templ)
        self.assertIn("{slash}", templ)
        filled = templ.format(link=link, slash=slash)
        self.assertIn("(%d tokens)" % link, filled)
        self.assertIn("(%d tokens)" % slash, filled)
        self.assertNotIn("{", filled)

    def test_residue_ledger_quotes_the_slash_recount(self):
        ledger, _o, _p = lv.residue_ledger(
            self.pages, self.allp, self.train, self.test, self.screening)
        link, slash = lv.slash_role_counts()
        row = next(r for r in ledger if r["item"] == "/ dual role")
        self.assertIn("%d slash tokens glue" % link, row["detail"])
        self.assertIn("vs %d boundary-like" % slash, row["detail"])

    def test_v5_same_and_mismatched_are_paired(self):
        """`same` and `mism` are built as pairs, so the sign test's zip
        cannot silently misalign.

        Regression: the two lists were appended under *independent*
        `continue` guards, so a term passing the profile floor but
        failing the partner floor shifted every later pair.  Here it
        does not fire on the real book (40/40 both ways), so the check
        is structural: lengths must agree and each mismatch pair must be
        recomputable from the same term index.
        """
        rng = random.Random(lv.SEED)
        _row, det = lv.check_v5_sense_stability(
            self.pages, self.allp, self.train, self.test, 4, rng)
        self.assertEqual(len(det["same"]), len(det["mism"]))
        self.assertEqual(det["n_terms"], len(det["same"]))
        # per_term rows must line up 1:1 with the scored pairs
        scored = [r for r in det["per_term"] if r["dice_mismatched"] != ""]
        self.assertEqual(len(scored), len(det["same"]))
        # and the per-term mismatched values are the ones zipped
        self.assertEqual([r["dice_mismatched"] for r in scored],
                         list(det["mism"]))

    def test_v5_dropped_partner_drops_the_pair(self):
        """A term whose partner fails the floor contributes to neither
        list -- the synthetic case the real book does not exercise."""
        terms = ["t%02d" % i for i in range(8)]
        pages = {}
        for p in (1, 2):
            seq = []
            for i, t in enumerate(terms):
                # "t00" is well attested, but its partner "t05" is not
                if t == "t05":
                    continue
                seq.extend([t] * 40)
            pages[p] = seq
        pa = lv.collocate_profiles(pages, [1], terms)
        pb = lv.collocate_profiles(pages, [2], terms)
        same, mism = [], []
        for idx, t in enumerate(terms):
            a, b = pa[t], pb[t]
            t2 = terms[(idx + 17) % len(terms)]
            b2 = pb[t2]
            if sum(a.values()) < 20 or sum(b2.values()) < 20:
                continue
            same.append(lv.dice_topk(a, b, lv.TOP_K))
            mism.append(lv.dice_topk(a, b2, lv.TOP_K))
        # "t00"'s partner is "t01", which is attested, so nothing drops;
        # the point is that the two lists are always the same length.
        self.assertEqual(len(same), len(mism))

    def test_writers_emit_every_declared_column(self):
        """No TSV column may be silently empty because of a key/header
        mismatch.

        Regression: `sense_stability.tsv` declared `dice_mismatched`
        while the rows carried `dice_mism`, so `write_tsv`'s
        `row.get(header)` lookup wrote an all-empty column -- and the
        V5 null comparison vanished from the artifact while the report
        still quoted its median.  Every writer is now checked by
        intercepting `write_tsv` and comparing header names to row keys.
        """
        seen = {}
        real = lv.cs.write_tsv

        def spy(path, header, rows):
            keys = set()
            for r in rows:
                keys |= set(r)
            missing = [h for h in header if h not in keys]
            extra = sorted(k for k in keys if k not in header)
            seen[str(path)] = (missing, extra)
            return real(path, header, rows)

        lv.cs.write_tsv = spy
        try:
            lv.build(out=Path(self.tmp.name), n_nulls=2)
        finally:
            lv.cs.write_tsv = real
        self.assertTrue(seen, "no writer was intercepted")
        for name, (missing, extra) in seen.items():
            self.assertEqual(missing, [], "%s: header names with no row key"
                             % name)
            self.assertEqual(extra, [], "%s: row keys not in the header "
                             "(dropped on write)" % name)

    def test_v0_null_draws_are_persisted(self):
        """V0's nulls are returned so they reach null_distributions.tsv.

        Regression: the negative control computed 40 null draws and
        discarded them (`det` carried only `lam`), so the file the
        module documents as "every null draw" held nothing for the one
        check whose soundness the whole verdict rests on.
        """
        rng = random.Random(lv.SEED)
        _row, det = lv.check_v0_negative_control(
            self.pages, self.allp, self.train, self.test, 6, rng)
        self.assertIn("nulls", det)
        self.assertEqual(len(det["nulls"]), 6)
        lv.build(out=Path(self.tmp.name), n_nulls=2)
        out = Path(self.tmp.name) / "null_distributions.tsv"
        rows = lv.cs.read_tsv(out)
        v0 = [r for r in rows if r["check"] == "V0"]
        self.assertTrue(v0, "no V0 rows in null_distributions.tsv")
        self.assertEqual(len(v0), 2)     # the --nulls 2 build
        # every scored check must have its draws recorded (V1 files two null
        # families, so a check matches by prefix)
        scored = [r["check"] for r in lv.cs.read_tsv(
            out.parent / "checks.tsv")]
        recorded = [r["check"] for r in rows]
        missing = [c for c in scored
                   if not any(d == c or d.startswith(c + "_")
                              for d in recorded)]
        # V5's null is the mismatched-term set (deterministic, not a
        # random draw), so it is reported per-term in sense_stability.tsv
        self.assertEqual(missing, ["V5"])
        # the two V1 families must be distinguishable, and must not
        # collide with the V4a/V4b convention (a leading letter means a
        # sub-statistic of one check)
        v1 = sorted({r["check"] for r in rows if r["check"].startswith("V1")})
        self.assertEqual(v1, ["V1_shuffle", "V1_typeperm"])


def ng_build_stream(pages_runs):
    import operator_ngrams as ng
    return ng.build_stream(pages_runs)


def cs_rows(path):
    """The raw #17 token rows, for recounting operators in the tests."""
    import clause_segmentation as cs
    return cs.read_tsv(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
