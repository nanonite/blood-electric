#!/usr/bin/env python3
"""Regression suite for #36 formal grammar induction and complexity.

Runs the analysis once into a temp directory, then checks the invariants
that matter:

  * the operator stream is #10/#11's (10,949 tokens, V=11);
  * the delimiter pairing reconciles with #12/#14 (230/182/156);
  * the entropy rates match #47's leave-one-page-out values;
  * the state-count estimators are present and ordered;
  * the verdict is non-null and the report contains it;
  * the reconciliation and diagnostics artifacts exist and are consistent.

    python work/test_grammar_induction.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grammar_induction as gi

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="grammar_induction_test_")
    gi.main(["--out", tmp, "--quiet"])
    for name in ("subword_complexity", "entropy_rate", "mi_by_lag", "lz78",
                 "state_counts", "delimiter_pairing", "opener_closer_dependence",
                 "reconciliation", "diagnostics"):
        with open(Path(tmp) / f"{name}.tsv", encoding="utf-8") as fh:
            _ARTIFACTS[name] = list(csv.DictReader(fh, delimiter="\t"))
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    with open(Path(tmp) / "grammar_induction_report.md", encoding="utf-8") as fh:
        _ARTIFACTS["report"] = fh.read()
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


class TestStreamIdentity(unittest.TestCase):
    """The operator stream must be #10/#11's."""

    def test_operator_token_count(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_operator_tokens"], 10949)

    def test_operator_vocab(self):
        self.assertEqual(_ARTIFACTS["summary"]["vocab_op"], 11)

    def test_token_stream_count(self):
        s = _ARTIFACTS["summary"]
        # token stream is loaded via ngram_surprisal; just check it ran
        self.assertIn("nerode_state_count", s)


class TestDelimiterReconciliation(unittest.TestCase):
    """The stack simulation must reproduce #12/#14 exactly."""

    def test_matched_pairs(self):
        self.assertEqual(_ARTIFACTS["summary"]["pairing_matched"], 230)

    def test_unclosed_opens(self):
        self.assertEqual(_ARTIFACTS["summary"]["pairing_unclosed"], 182)

    def test_stray_closers(self):
        self.assertEqual(_ARTIFACTS["summary"]["pairing_stray"], 156)

    def test_depth_histogram_sums_to_matched(self):
        hist = _ARTIFACTS["summary"]["pairing_depth_hist"]
        self.assertEqual(sum(hist.values()), 230)


class TestEntropyRates(unittest.TestCase):
    """Held-out entropy rates must match #47."""

    def test_operator_heldout_H0(self):
        self.assertAlmostEqual(
            _ARTIFACTS["summary"]["op_heldout_H"]["0"], 2.759702, places=4)

    def test_operator_heldout_H2(self):
        self.assertAlmostEqual(
            _ARTIFACTS["summary"]["op_heldout_H"]["2"], 2.708941, places=4)

    def test_operator_plugin_H0(self):
        self.assertAlmostEqual(
            _ARTIFACTS["summary"]["op_plugin_H"]["0"], 2.738828, places=4)

    def test_best_order_is_2(self):
        self.assertEqual(_ARTIFACTS["summary"]["best_order"], 2)

    def test_parsimony_order_is_1(self):
        self.assertEqual(_ARTIFACTS["summary"]["parsimony_order"], 1)


class TestStateCounts(unittest.TestCase):
    """State-count estimators must be present and ordered."""

    def test_predictive_state_count(self):
        self.assertEqual(_ARTIFACTS["summary"]["predictive_state_count"], 98)

    def test_predictive_state_count_parsimony(self):
        self.assertEqual(
            _ARTIFACTS["summary"]["predictive_state_count_parsimony"], 11)

    def test_nerode_state_count(self):
        self.assertEqual(_ARTIFACTS["summary"]["nerode_state_count"], 325)

    def test_nerode_per_length(self):
        pl = _ARTIFACTS["summary"]["nerode_per_length"]
        self.assertEqual(pl["0"], 1)
        self.assertEqual(pl["1"], 7)
        self.assertEqual(pl["2"], 54)

    def test_state_counts_tsv_has_three_estimators(self):
        estimators = {r["estimator"] for r in rows("state_counts")}
        self.assertIn("predictive_order_contexts", estimators)
        self.assertIn("nerode_continuation_sets", estimators)
        self.assertIn("lz78_grammar_phrases", estimators)


class TestLZ78(unittest.TestCase):
    """LZ78 compression stats must be present."""

    def test_phrase_count(self):
        self.assertEqual(_ARTIFACTS["summary"]["lz78_phrases"], 2433)

    def test_reused_types(self):
        self.assertEqual(_ARTIFACTS["summary"]["lz78_reused"], 1)

    def test_bits_per_symbol(self):
        self.assertAlmostEqual(
            _ARTIFACTS["summary"]["lz78_bits_per_symbol"], 2.948335, places=4)


class TestMIByLag(unittest.TestCase):
    """MI lag-1 must match #11."""

    def test_mi_lag1(self):
        self.assertAlmostEqual(
            _ARTIFACTS["summary"]["mi_lag1"], 0.13049, places=4)

    def test_mi_lag1_null(self):
        self.assertAlmostEqual(
            _ARTIFACTS["summary"]["mi_lag1_null"], 0.006494, places=4)

    def test_mi_exceeds_null(self):
        s = _ARTIFACTS["summary"]
        self.assertGreater(s["mi_lag1"], 10 * s["mi_lag1_null"])


class TestSubwordComplexity(unittest.TestCase):
    """p(L) must not saturate within the measured range."""

    def test_saturation_L_is_none(self):
        self.assertIsNone(
            _ARTIFACTS["summary"]["subword_complexity_saturation_L"])

    def test_pL_grows_with_L(self):
        op_rows = [r for r in rows("subword_complexity")
                   if r["stream"] == "operator"]
        p_values = [int(r["p_L"]) for r in op_rows]
        self.assertEqual(p_values, sorted(p_values))
        self.assertGreater(p_values[-1], p_values[0])

    def test_pL_below_iid_null(self):
        op_rows = {int(r["L"]): int(r["p_L"])
                   for r in rows("subword_complexity")
                   if r["stream"] == "operator"}
        null_rows = {int(r["L"]): int(r["p_L"])
                     for r in rows("subword_complexity")
                     if r["stream"] == "operator_iid_null"}
        for L, p_val in op_rows.items():
            # L=1 is the vocabulary size (11 for both); structure shows from L=2
            if L == 1:
                self.assertEqual(p_val, null_rows[L])
            else:
                self.assertLess(p_val, null_rows[L])


class TestReconciliation(unittest.TestCase):
    """The reconciliation artifact must match earlier tasks."""

    def test_reconciliation_has_rows(self):
        self.assertGreater(len(rows("reconciliation")), 0)

    def test_operator_tokens_reconciled(self):
        r = next(r for r in rows("reconciliation")
                 if r["quantity"] == "operator tokens")
        self.assertEqual(r["here"], "10,949")

    def test_delimiter_matched_reconciled(self):
        r = next(r for r in rows("reconciliation")
                 if r["quantity"] == "delimiter matched pairs")
        self.assertEqual(r["here"], "230")

    def test_diagnostics_has_seed(self):
        d = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(d["seed"], "42")


class TestReport(unittest.TestCase):
    """The report must exist and contain the verdict."""

    def test_report_exists(self):
        self.assertIn("Verdict", _ARTIFACTS["report"])

    def test_verdict_in_summary(self):
        v = _ARTIFACTS["summary"]["verdict"]
        self.assertIsNotNone(v)
        self.assertIn("FINITE-MEMORY", v)

    def test_verdict_in_report(self):
        self.assertIn("FINITE-MEMORY", _ARTIFACTS["report"])

    def test_report_has_all_sections(self):
        for section in ("Subword complexity", "Entropy-rate",
                        "Mutual information", "Lempel-Ziv",
                        "state count", "Delimiter pairing",
                        "Verdict", "Caveats"):
            self.assertIn(section, _ARTIFACTS["report"])


class TestPageOrderNull(unittest.TestCase):
    """The page-order null must reproduce p(L)."""

    def test_page_order_null_pL_close_to_observed(self):
        s = _ARTIFACTS["summary"]
        null_mean = s["page_order_null_pL_mean"]
        op_rows = {int(r["L"]): int(r["p_L"])
                   for r in rows("subword_complexity")
                   if r["stream"] == "operator"}
        for L_str, null_val in null_mean.items():
            L = int(L_str)
            if L in op_rows:
                # within 5% of observed
                self.assertLess(
                    abs(null_val - op_rows[L]) / op_rows[L], 0.05,
                    f"L={L}: null {null_val} vs observed {op_rows[L]}")


if __name__ == "__main__":
    unittest.main()
