#!/usr/bin/env python3
"""Regression suite for #72 predictive validity on held-out text.

Runs the analysis once into a temp directory, then checks the invariants
that matter:

  * the operator stream is #10/#11's (10,949 tokens, V=11, 225 pages);
  * the splits are page-level and disjoint (180 train / 45 test), the
    chronological split is first-80%-of-sorted-pages, and the seeded
    random split is repeatable;
  * the chronological operator model beats the unigram baseline at every
    order (perplexity 6.7923 -> 6.3076 -> 6.1004) while argmax accuracy
    does not improve and the random split's order 2 overfits;
  * the run inventory covers >= 99% of test run tokens in both splits;
  * collocation logDice ranks are stable across the split (unit 0.8984,
    operator-word 0.7080) with no systematic level shift;
  * the reconciliation matches #11 / #27 exactly;
  * the verdicts are non-null, appear in the report, and the deliverable
    heldout_scores.tsv carries the headline numbers;
  * the n-gram interpolation, scoring and rank statistics are exact on
    synthetic fixtures.

    python work/test_heldout_validity.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import heldout_validity as hv

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="heldout_validity_test_")
    hv.main(["--out", tmp, "--quiet"])
    for name in ("operator_heldout", "run_coverage", "collocation_heldout",
                 "heldout_scores", "diagnostics", "reconciliation"):
        with open(Path(tmp) / f"{name}.tsv", encoding="utf-8") as fh:
            _ARTIFACTS[name] = list(csv.DictReader(fh, delimiter="\t"))
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    with open(Path(tmp) / "heldout_validity_report.md", encoding="utf-8") as fh:
        _ARTIFACTS["report"] = fh.read()
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


def _diag():
    return {r["diag"]: r["value"] for r in rows("diagnostics")}


def _op_row(split, order):
    return next(r for r in rows("operator_heldout")
                if r["split"] == split and int(r["order"]) == order)


def _coll_row(split, layer):
    return next(r for r in rows("collocation_heldout")
                if r["split"] == split and r["layer"] == layer)


class TestDiagnostics(unittest.TestCase):
    """The build ledger must pin every knob of the evaluation."""

    def test_schema(self):
        self.assertEqual(_diag()["schema"], "heldout_validity/1")

    def test_seed_and_frac(self):
        self.assertEqual(_diag()["seed"], "42")
        self.assertEqual(_diag()["train_frac"], "0.8")

    def test_model_knobs(self):
        self.assertEqual(_diag()["lam"], "0.8")
        self.assertEqual(_diag()["orders"], "[0, 1, 2]")
        self.assertEqual(_diag()["top_k"], "100")
        self.assertEqual(_diag()["min_bigram"], "2")

    def test_corpus_size(self):
        self.assertEqual(_diag()["n_pages"], "225")
        self.assertEqual(_diag()["n_operator_tokens"], "10949")
        self.assertEqual(_diag()["operator_vocab"], "11")

    def test_split_sizes(self):
        self.assertEqual(_diag()["chrono_train_pages"], "180")
        self.assertEqual(_diag()["chrono_test_pages"], "45")
        self.assertEqual(_diag()["random_train_pages"], "180")
        self.assertEqual(_diag()["random_test_pages"], "45")

    def test_collocation_item_counts_reassemble_the_book(self):
        # train + test items must add back to #27's full-book streams
        self.assertEqual(
            int(_diag()["unit_items_train_chrono"])
            + int(_diag()["unit_items_test_chrono"]), 74077)
        self.assertEqual(
            int(_diag()["token_items_train_chrono"])
            + int(_diag()["token_items_test_chrono"]), 88669)

    def test_markup_units_removed(self):
        self.assertEqual(_diag()["markup_units_removed"], "588")


class TestSplits(unittest.TestCase):
    """Splits are page-level, disjoint, and the seeded one is repeatable."""

    @classmethod
    def setUpClass(cls):
        import operator_ngrams as on
        cls.pages = sorted(on.page_runs(str(hv.SRC)))

    def test_chrono_split_sizes(self):
        train, test = hv.chrono_split(self.pages)
        self.assertEqual(len(train), 180)
        self.assertEqual(len(test), 45)

    def test_chrono_split_is_first_80_percent_of_sorted_pages(self):
        train, test = hv.chrono_split(self.pages)
        self.assertEqual(train, self.pages[:180])
        self.assertEqual(test, self.pages[180:])

    def test_chrono_split_is_disjoint_and_exhaustive(self):
        train, test = hv.chrono_split(self.pages)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(self.pages))

    def test_random_split_sizes(self):
        train, test = hv.random_split(self.pages)
        self.assertEqual(len(train), 180)
        self.assertEqual(len(test), 45)

    def test_random_split_is_disjoint_and_exhaustive(self):
        train, test = hv.random_split(self.pages)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(self.pages))

    def test_random_split_is_seeded_and_repeatable(self):
        a = hv.random_split(self.pages, seed=42)
        b = hv.random_split(self.pages, seed=42)
        self.assertEqual(a, b)

    def test_random_split_seed_changes_the_draw(self):
        a = hv.random_split(self.pages, seed=42)
        b = hv.random_split(self.pages, seed=43)
        self.assertNotEqual(a, b)


class TestOperatorStream(unittest.TestCase):
    """The scored stream must be #10/#11's."""

    def test_operator_token_count(self):
        self.assertEqual(_diag()["n_operator_tokens"], "10949")

    def test_operator_vocab(self):
        self.assertEqual(_diag()["operator_vocab"], "11")

    def test_page_count(self):
        self.assertEqual(_ARTIFACTS["summary"]["split"]["n_pages"], 225)


class TestOperatorHeldout(unittest.TestCase):
    """The grammar predicts held-out operator sequences in probability."""

    def test_chrono_perplexity_by_order(self):
        self.assertAlmostEqual(float(_op_row("chrono", 0)["perplexity"]),
                               6.7923, places=4)
        self.assertAlmostEqual(float(_op_row("chrono", 1)["perplexity"]),
                               6.3076, places=4)
        self.assertAlmostEqual(float(_op_row("chrono", 2)["perplexity"]),
                               6.1004, places=4)

    def test_chrono_perplexity_strictly_decreases_with_order(self):
        p = [float(_op_row("chrono", o)["perplexity"]) for o in (0, 1, 2)]
        self.assertLess(p[1], p[0])
        self.assertLess(p[2], p[1])

    def test_chrono_order2_gain_vs_unigram(self):
        self.assertAlmostEqual(
            float(_op_row("chrono", 2)["gain_vs_unigram"]), 0.1019, places=4)

    def test_chrono_order1_gain_positive(self):
        self.assertGreater(float(_op_row("chrono", 1)["gain_vs_unigram"]), 0.0)

    def test_chrono_argmax_accuracy_does_not_improve(self):
        # better probability mass, not a better single pick: order 2's
        # argmax accuracy is below the unigram-majority baseline
        self.assertLess(float(_op_row("chrono", 2)["argmax_accuracy"]),
                        float(_op_row("chrono", 0)["argmax_accuracy"]))

    def test_chrono_top2_accuracy_improves_with_order(self):
        t = [float(_op_row("chrono", o)["top2_accuracy"]) for o in (0, 1, 2)]
        self.assertGreater(t[1], t[0])
        self.assertGreater(t[2], t[1])

    def test_chrono_test_token_counts(self):
        self.assertEqual(int(_op_row("chrono", 0)["n_test_tokens"]), 2006)
        self.assertEqual(int(_op_row("chrono", 2)["n_test_tokens"]), 1918)

    def test_chrono_train_token_count(self):
        self.assertEqual(int(_op_row("chrono", 0)["n_train_tokens"]), 8943)

    def test_random_perplexity_by_order(self):
        self.assertAlmostEqual(float(_op_row("random", 0)["perplexity"]),
                               6.4268, places=4)
        self.assertAlmostEqual(float(_op_row("random", 1)["perplexity"]),
                               6.2280, places=4)
        self.assertAlmostEqual(float(_op_row("random", 2)["perplexity"]),
                               6.3226, places=4)

    def test_random_order2_overfits_but_beats_unigram(self):
        # the disclosed random-split pathology: order 2 is worse than
        # order 1 yet still better than the unigram baseline
        self.assertGreater(float(_op_row("random", 2)["perplexity"]),
                           float(_op_row("random", 1)["perplexity"]))
        self.assertLess(float(_op_row("random", 2)["perplexity"]),
                        float(_op_row("random", 0)["perplexity"]))

    def test_random_order2_gain_is_small(self):
        self.assertAlmostEqual(
            float(_op_row("random", 2)["gain_vs_unigram"]), 0.0162, places=4)

    def test_chrono_gain_far_exceeds_random_gain(self):
        # the conservative cross-book split is where the grammar transfers
        self.assertGreater(
            float(_op_row("chrono", 2)["gain_vs_unigram"]),
            5 * float(_op_row("random", 2)["gain_vs_unigram"]))

    def test_chrono_gain_sits_under_mi_ceiling(self):
        # the context gain in bits must stay under #11's lag-1 MI ceiling
        # (0.1305 bits, ~5% of the 2.7639-bit unigram entropy) at order 1
        h0 = -float(_op_row("chrono", 0)["mean_log2_prob"])
        h1 = -float(_op_row("chrono", 1)["mean_log2_prob"])
        self.assertLess(h0 - h1, 0.1305)
        self.assertLess((h0 - h1) / h0, 0.05)

    def test_summary_matches_operator_tsv(self):
        s = _ARTIFACTS["summary"]["operator"]["chrono"]["perplexity"]
        self.assertAlmostEqual(s["0"], 6.792302, places=6)
        self.assertAlmostEqual(s["1"], 6.307556, places=6)
        self.assertAlmostEqual(s["2"], 6.100375, places=6)


class TestRunCoverage(unittest.TestCase):
    """The training pages' run inventory must cover the test pages' runs."""

    def test_chrono_type_coverage(self):
        r = next(r for r in rows("run_coverage") if r["split"] == "chrono")
        self.assertEqual(r["n_test_run_types"], "1744")
        self.assertEqual(r["n_test_run_types_seen_in_train"], "1738")
        self.assertAlmostEqual(float(r["run_type_coverage"]), 0.9966,
                               places=4)

    def test_chrono_token_coverage(self):
        r = next(r for r in rows("run_coverage") if r["split"] == "chrono")
        self.assertEqual(r["n_test_run_tokens"], "2006")
        self.assertEqual(r["n_test_run_tokens_in_train_types"], "1991")
        self.assertAlmostEqual(float(r["run_token_coverage"]), 0.9925,
                               places=4)

    def test_coverage_is_above_99_percent_in_both_splits(self):
        for r in rows("run_coverage"):
            self.assertGreaterEqual(float(r["run_token_coverage"]), 0.99)
            self.assertGreaterEqual(float(r["run_type_coverage"]), 0.99)

    def test_coverage_is_a_ceiling_not_a_score(self):
        # unseen run types are unpredictable by construction; the residual
        # must be disclosed, i.e. coverage strictly below 1.0
        for r in rows("run_coverage"):
            self.assertLess(float(r["run_token_coverage"]), 1.0)
            self.assertLess(float(r["run_type_coverage"]), 1.0)


class TestCollocationStability(unittest.TestCase):
    """Association ranks computed on train must predict the held-out 20%."""

    def test_chrono_unit_logdice_spearman(self):
        r = _coll_row("chrono", "unit")
        self.assertEqual(r["n_shared_types"], "1193")
        self.assertAlmostEqual(float(r["spearman_logdice"]), 0.8984,
                               places=4)

    def test_chrono_unit_pearson(self):
        r = _coll_row("chrono", "unit")
        self.assertAlmostEqual(float(r["pearson_logdice"]), 0.9206,
                               places=4)

    def test_chrono_token_op_logdice_spearman(self):
        r = _coll_row("chrono", "token_op")
        self.assertEqual(r["n_shared_types"], "316")
        self.assertAlmostEqual(float(r["spearman_logdice"]), 0.7080,
                               places=4)

    def test_chrono_token_all_logdice_spearman(self):
        r = _coll_row("chrono", "token_all")
        self.assertEqual(r["n_shared_types"], "1452")
        self.assertAlmostEqual(float(r["spearman_logdice"]), 0.8717,
                               places=4)

    def test_random_unit_logdice_spearman(self):
        r = _coll_row("random", "unit")
        self.assertEqual(r["n_shared_types"], "1212")
        self.assertAlmostEqual(float(r["spearman_logdice"]), 0.9166,
                               places=4)

    def test_random_token_op_logdice_spearman(self):
        r = _coll_row("random", "token_op")
        self.assertEqual(r["n_shared_types"], "338")
        self.assertAlmostEqual(float(r["spearman_logdice"]), 0.6513,
                               places=4)

    def test_spearman_exceeds_pearson_is_not_required_but_both_high(self):
        for r in rows("collocation_heldout"):
            self.assertGreaterEqual(float(r["spearman_logdice"]), 0.6)
            self.assertGreaterEqual(float(r["pearson_logdice"]), 0.6)

    def test_topk_overlap_and_survival_chrono(self):
        cu = _coll_row("chrono", "unit")
        ct = _coll_row("chrono", "token_op")
        self.assertEqual(cu["top_k"], "100")
        self.assertAlmostEqual(float(cu["topk_overlap"]), 0.47, places=2)
        self.assertAlmostEqual(float(ct["topk_overlap"]), 0.59, places=2)
        self.assertAlmostEqual(
            float(cu["topk_train_pairs_survive_in_test"]), 0.48, places=2)
        self.assertAlmostEqual(
            float(ct["topk_train_pairs_survive_in_test"]), 0.66, places=2)

    def test_survival_is_asymmetric_by_direction(self):
        # test->train survival is high (the train table is larger); the
        # train->test direction is the predictive one and is weaker
        for r in rows("collocation_heldout"):
            self.assertGreaterEqual(
                float(r["topk_test_pairs_survive_in_train"]),
                float(r["topk_train_pairs_survive_in_test"]))

    def test_no_systematic_level_shift(self):
        # a high rank correlation cannot hide a constant offset: mean
        # |delta logDice| must stay small on the 14-point logDice scale
        for r in rows("collocation_heldout"):
            self.assertLess(float(r["mean_abs_delta_logdice"]), 1.5)

    def test_pmi_rank_correlation_reported_separately(self):
        # PMI carries N, so only its rank correlation is comparable; it
        # must be present and high wherever logDice is
        for r in rows("collocation_heldout"):
            self.assertGreaterEqual(float(r["spearman_pmi"]), 0.6)


class TestVerdicts(unittest.TestCase):
    """The verdicts must be computed, non-null, and internally consistent."""

    def test_chrono_operator_verdict(self):
        v = _ARTIFACTS["summary"]["operator"]["chrono"]["verdict"]
        self.assertEqual(v, "PREDICTS IN PROBABILITY")

    def test_random_operator_verdict(self):
        v = _ARTIFACTS["summary"]["operator"]["random"]["verdict"]
        self.assertEqual(v, "PREDICTS AT ORDER 1 (ORDER 2 OVERFITS)")

    def test_combined_operator_verdict(self):
        v = _ARTIFACTS["summary"]["operator_verdict"]
        self.assertIn("PREDICTS IN PROBABILITY (chronological)", v)
        self.assertIn("PREDICTS AT ORDER 1 (ORDER 2 OVERFITS)", v)
        self.assertIn("10.2%", v)

    def test_collocation_verdict(self):
        v = _ARTIFACTS["summary"]["collocation_verdict"]
        self.assertIn("STABLE (unit layer", v)
        self.assertIn("MODERATELY STABLE (operator-word layer", v)
        self.assertIn("0.8984", v)
        self.assertIn("0.7080", v)

    def test_headline_is_non_empty_and_carries_the_numbers(self):
        h = _ARTIFACTS["summary"]["headline"]
        self.assertIn("6.100", h)
        self.assertIn("6.792", h)
        self.assertIn("99.3%", h)
        self.assertIn("0.8984", h)

    def test_verdicts_follow_from_the_scores(self):
        # re-derive the operator classification from the TSV alone
        for split in ("chrono", "random"):
            p = [float(_op_row(split, o)["perplexity"]) for o in (0, 1, 2)]
            if p[1] < p[0] and p[2] < p[1]:
                expect = "PREDICTS IN PROBABILITY"
            elif p[1] < p[0]:
                expect = "PREDICTS AT ORDER 1 (ORDER 2 OVERFITS)"
            else:
                expect = "DOES NOT PREDICT"
            self.assertEqual(
                _ARTIFACTS["summary"]["operator"][split]["verdict"], expect)


class TestReconciliation(unittest.TestCase):
    """This build must reconcile with #11 / #27 exactly."""

    def test_operator_tokens_reconciled(self):
        r = next(r for r in rows("reconciliation")
                 if r["quantity"] == "operator tokens")
        self.assertEqual(r["earlier"], "10,949 (#10/#11)")
        self.assertEqual(r["here"], "10,949")

    def test_chrono_split_pages_reconciled(self):
        r = next(r for r in rows("reconciliation")
                 if r["quantity"] == "chronological split pages")
        self.assertEqual(r["earlier"], "180/45 (#11)")
        self.assertEqual(r["here"], "180/45")

    def test_order2_chrono_perplexity_matches_11(self):
        r = next(r for r in rows("reconciliation")
                 if r["quantity"].startswith("operator order-2"))
        self.assertEqual(r["earlier"], "6.1004 (#11)")
        self.assertEqual(r["here"], "6.1004")

    def test_mi_lag1_reconciled(self):
        r = next(r for r in rows("reconciliation")
                 if r["quantity"] == "operator MI lag-1 (bits)")
        self.assertEqual(r["earlier"], "0.1305 (#11)")
        self.assertEqual(r["here"], "0.1305")

    def test_unit_items_reconciled(self):
        r = next(r for r in rows("reconciliation")
                 if r["quantity"] == "unit collocation items (full book)")
        self.assertEqual(r["earlier"], "74,077 (#27)")
        self.assertEqual(r["here"], "74,077")

    def test_token_items_reconciled(self):
        r = next(r for r in rows("reconciliation")
                 if r["quantity"] == "token collocation items (full book)")
        self.assertEqual(r["earlier"], "88,669 (#27)")
        self.assertEqual(r["here"], "88,669")

    def test_reconciliation_covers_every_earlier_task(self):
        earlier = " ".join(r["earlier"] for r in rows("reconciliation"))
        for marker in ("#10/#11", "#11", "#27"):
            self.assertIn(marker, earlier)


class TestDeliverable(unittest.TestCase):
    """heldout_scores.tsv is THE DELIVERABLE: one row per headline number."""

    def test_twelve_score_rows(self):
        self.assertEqual(len(rows("heldout_scores")), 12)

    def test_claims_and_splits_present(self):
        claims = {r["claim"] for r in rows("heldout_scores")}
        self.assertEqual(claims, {"operator_sequences", "collocations"})

    def test_chrono_operator_headline_values(self):
        d = {(r["split_or_layer"], r["metric"]): r["value"]
             for r in rows("heldout_scores")}
        self.assertEqual(d[("chrono", "perplexity_order2")], "6.1004")
        self.assertEqual(d[("chrono", "perplexity_gain_order2_vs_unigram")],
                         "0.1019")
        self.assertEqual(d[("chrono", "run_token_coverage")], "0.9925")

    def test_random_operator_headline_values(self):
        d = {(r["split_or_layer"], r["metric"]): r["value"]
             for r in rows("heldout_scores")}
        self.assertEqual(d[("random", "perplexity_order2")], "6.3226")
        self.assertEqual(d[("random", "perplexity_gain_order2_vs_unigram")],
                         "0.0162")

    def test_collocation_headline_values(self):
        d = {(r["split_or_layer"], r["metric"]): r["value"]
             for r in rows("heldout_scores")}
        self.assertEqual(
            d[("unit_chrono", "spearman_logdice_train_vs_test")], "0.8984")
        self.assertEqual(
            d[("token_op_chrono", "spearman_logdice_train_vs_test")],
            "0.708")
        self.assertEqual(
            d[("unit_random", "spearman_logdice_train_vs_test")], "0.9166")

    def test_values_agree_with_summary(self):
        d = {(r["split_or_layer"], r["metric"]): r["value"]
             for r in rows("heldout_scores")}
        s = _ARTIFACTS["summary"]
        self.assertEqual(
            d[("chrono", "perplexity_order2")],
            f"{s['operator']['chrono']['perplexity']['2']:.4f}")
        self.assertEqual(
            d[("unit_chrono", "spearman_logdice_train_vs_test")],
            f"{s['collocation']['chrono']['unit']['spearman_logdice']:.4f}")


class TestReport(unittest.TestCase):
    """The report must exist, carry the verdicts, and document the method."""

    def test_report_has_all_sections(self):
        for section in ("Headline result",
                        "Operator sequences — held-out scores by order",
                        "Run inventory coverage",
                        "Collocations — held-out stability",
                        "Method and limits",
                        "Artifacts"):
            self.assertIn(section, _ARTIFACTS["report"])

    def test_verdicts_in_report(self):
        self.assertIn("PREDICTS IN PROBABILITY", _ARTIFACTS["report"])
        self.assertIn("STABLE (unit layer", _ARTIFACTS["report"])
        self.assertIn("MODERATELY STABLE", _ARTIFACTS["report"])

    def test_headline_numbers_in_report(self):
        for number in ("6.100", "6.792", "10.2%", "0.8984", "0.7080",
                       "99.3%", "1738/1744"):
            self.assertIn(number, _ARTIFACTS["report"])

    def test_method_documents_the_split_and_the_baseline(self):
        report = _ARTIFACTS["report"]
        self.assertIn("first 80% of pages train, last 20% test", report)
        self.assertIn("seeded random", report)
        self.assertIn("unigram baseline", report)
        self.assertIn("train-only", report)

    def test_report_lists_every_artifact(self):
        for artifact in ("heldout_scores.tsv", "operator_heldout.tsv",
                         "run_coverage.tsv", "collocation_heldout.tsv",
                         "summary.json", "diagnostics.tsv",
                         "reconciliation.tsv"):
            self.assertIn(artifact, _ARTIFACTS["report"])

    def test_report_cites_reproducibility(self):
        self.assertIn("work/heldout_validity.py", _ARTIFACTS["report"])
        self.assertIn("work/test_heldout_validity.py", _ARTIFACTS["report"])

    def test_argmax_non_improvement_stated_honestly(self):
        # the report must not claim argmax accuracy rises when it falls
        self.assertIn("does not improve", _ARTIFACTS["report"])
        self.assertNotIn("Argmax accuracy rises", _ARTIFACTS["report"])

    def test_random_split_overfitting_disclosed_in_report(self):
        self.assertIn("order 2 overfits", _ARTIFACTS["report"])


class TestModelMath(unittest.TestCase):
    """The n-gram interpolation, scoring and rank statistics, exactly."""

    PAGES = {"p1": ["a", "b", "a"], "p2": ["a", "b", "b"]}

    def _train(self, order):
        return hv.train_ngram({"p1": self.PAGES["p1"]}, order)

    def test_p0_is_train_unigram(self):
        m = self._train(1)
        self.assertAlmostEqual(m["p0"]("a"), 2 / 3, places=6)
        self.assertAlmostEqual(m["p0"]("b"), 1 / 3, places=6)

    def test_p1_interpolation_exact(self):
        m = self._train(1)
        # p1(b|a) = lam * c(a,b)/n(a) + (1-lam) * p0(b) = 0.8*1 + 0.2/3
        self.assertAlmostEqual(m["p1"]("b", "a"), 0.8 + 0.2 / 3, places=9)
        # unseen continuation: pure backoff
        self.assertAlmostEqual(m["p1"]("a", "a"), 0.2 * 2 / 3, places=9)

    def test_p2_interpolation_exact(self):
        m = self._train(2)
        # p1(a|b) = lam * c(b,a)/n(b) + (1-lam) * p0(a) = 0.8 + 0.2*2/3
        p1_a_given_b = 0.8 + 0.2 * 2 / 3
        # p2(a|a,b) = lam * c(a,b,a)/n(a,b) + (1-lam) * p1(a|b)
        self.assertAlmostEqual(m["p2"]("a", "a", "b"),
                               0.8 + 0.2 * p1_a_given_b, places=9)
        # unseen: backoff to p1(b|b) = 0.2 * p0(b)
        self.assertAlmostEqual(m["p2"]("b", "a", "b"),
                               0.2 * (0.2 * 1 / 3), places=9)

    def test_score_order0(self):
        m = self._train(0)
        s = hv.score_ngram(m, {"p2": self.PAGES["p2"]})
        self.assertEqual(s["n_test_tokens"], 3)
        self.assertAlmostEqual(s["argmax_accuracy"], 1 / 3, places=9)
        self.assertAlmostEqual(s["top2_accuracy"], 1.0, places=9)
        expect = (math.log2(2 / 3) + 2 * math.log2(1 / 3)) / 3
        self.assertAlmostEqual(s["mean_log2_prob"], expect, places=9)

    def test_score_order1(self):
        m = self._train(1)
        s = hv.score_ngram(m, {"p2": self.PAGES["p2"]})
        self.assertEqual(s["n_test_tokens"], 2)
        self.assertAlmostEqual(s["argmax_accuracy"], 0.5, places=9)
        self.assertAlmostEqual(s["top2_accuracy"], 1.0, places=9)
        expect = (math.log2(0.8 + 0.2 / 3)
                  + math.log2(0.2 * 1 / 3)) / 2
        self.assertAlmostEqual(s["mean_log2_prob"], expect, places=9)
        self.assertAlmostEqual(s["perplexity"], 2 ** (-expect), places=6)

    def test_score_order2(self):
        m = self._train(2)
        s = hv.score_ngram(m, {"p2": self.PAGES["p2"]})
        # only one scorable position per 3-token page at order 2
        self.assertEqual(s["n_test_tokens"], 1)
        self.assertAlmostEqual(s["argmax_accuracy"], 0.0, places=9)
        self.assertAlmostEqual(s["top2_accuracy"], 1.0, places=9)

    def test_higher_order_scores_fewer_tokens(self):
        # the first `order` tokens of each page are context, not targets
        n = [hv.score_ngram(self._train(o),
                            {"p2": self.PAGES["p2"]})["n_test_tokens"]
             for o in (0, 1, 2)]
        self.assertEqual(n, [3, 2, 1])

    def test_rankdata_ties_share_mean_rank(self):
        self.assertEqual(hv.rankdata([1, 1, 2]), [1.5, 1.5, 3.0])
        self.assertEqual(hv.rankdata([3, 1, 2]), [3.0, 1.0, 2.0])

    def test_spearman_and_perfect_correlations(self):
        self.assertAlmostEqual(hv.spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(hv.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(hv.pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_spearman_of_constant_input_is_nan(self):
        # a constant series has undefined rank correlation, not zero
        self.assertTrue(math.isnan(hv.spearman([1, 1, 1], [1, 2, 3])))

    def test_collocation_heldout_topk_and_survival(self):
        train_rows = {(i, i + 1): {"logdice": 14.0 - i, "f_xy": 100 - i,
                                   "pmi": 5.0, "n": 1000}
                      for i in range(10)}
        test_rows = {(i, i + 1): {"logdice": 14.0 - i + 0.5, "f_xy": 50,
                                  "pmi": 4.0, "n": 250}
                     for i in range(10)}
        d = hv.collocation_heldout(train_rows, test_rows, top_k=3)
        self.assertEqual(d["n_shared_types"], 10)
        # top-3 train pairs by logdice are (0,1),(1,2),(2,3); all survive
        self.assertAlmostEqual(d["topk_overlap"], 1.0)
        self.assertAlmostEqual(d["topk_train_pairs_survive_in_test"], 1.0)
        self.assertAlmostEqual(d["topk_test_pairs_survive_in_train"], 1.0)
        self.assertAlmostEqual(d["spearman_logdice"], 1.0)

    def test_collocation_heldout_survives_the_f2_floor(self):
        train_rows = {(i, i + 1): {"logdice": 14.0 - i, "f_xy": 10,
                                   "pmi": 5.0, "n": 1000}
                      for i in range(5)}
        # test side: pairs (0,1)/(1,2) at f>=2, pair (2,3) below the floor
        test_rows = {(0, 1): {"logdice": 14.5, "f_xy": 3, "pmi": 5.0,
                              "n": 250},
                     (1, 2): {"logdice": 13.5, "f_xy": 2, "pmi": 4.0,
                              "n": 250},
                     (2, 3): {"logdice": 12.9, "f_xy": 1, "pmi": 3.0,
                              "n": 250}}
        d = hv.collocation_heldout(train_rows, test_rows, top_k=5)
        # shared = types present in both row dicts, whatever their test f
        self.assertEqual(d["n_shared_types"], 3)
        # train top-5 = all five; only the two at f>=2 in test survive
        self.assertAlmostEqual(d["topk_train_pairs_survive_in_test"], 0.4)
        # test top-5 holds the three present pairs, all surviving in train;
        # the denominator is top_k, so 3/5
        self.assertAlmostEqual(d["topk_test_pairs_survive_in_train"], 0.6)

    def test_run_coverage_exact(self):
        page_runs = {"p1": [["a", "b"], ["a"]], "p2": [["a", "b"], ["c"]]}
        d = hv.run_coverage(page_runs, ["p1"], ["p2"])
        self.assertEqual(d["n_test_run_types"], 2)
        self.assertEqual(d["n_test_run_types_seen_in_train"], 1)
        self.assertEqual(d["n_test_run_tokens"], 3)
        self.assertEqual(d["n_test_run_tokens_in_train_types"], 2)
        self.assertAlmostEqual(d["run_type_coverage"], 0.5)
        self.assertAlmostEqual(d["run_token_coverage"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
