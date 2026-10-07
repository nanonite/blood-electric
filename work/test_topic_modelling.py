#!/usr/bin/env python3
"""Regression suite for #41 topic modelling (LDA / NMF / BERTopic).

Builds the artifact into a temp directory (the full build over the real #39
clause vectors) and checks the invariants that matter:

  * the document set reconciles with #39 / #33 / #17 / #29 exactly;
  * the coherence scan is complete and selects the catalogue k per model;
  * LDA/NMF/BERTopic topics are well-formed (top terms, descending weights);
  * clause topic assignments cover every clause and join to #33;
  * the community comparison produces bounded adjusted Rand values;
  * two builds produce identical tables (determinism).

    python work/test_topic_modelling.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clause_segmentation as cs
import topic_modelling as tm

_A = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="topics_test_")
    tm.build(Path(tmp), quiet=True)
    _A["dir"] = Path(tmp)
    _A["coherence"] = cs.read_tsv(Path(tmp) / "coherence_scan.tsv")
    _A["lda_topics"] = cs.read_tsv(Path(tmp) / "lda_topics.tsv")
    _A["nmf_topics"] = cs.read_tsv(Path(tmp) / "nmf_topics.tsv")
    _A["bert_topics"] = cs.read_tsv(Path(tmp) / "bertopic_topics.tsv")
    _A["lda_sum"] = cs.read_tsv(Path(tmp) / "lda_topic_summary.tsv")
    _A["nmf_sum"] = cs.read_tsv(Path(tmp) / "nmf_topic_summary.tsv")
    _A["bert_sum"] = cs.read_tsv(Path(tmp) / "bertopic_topic_summary.tsv")
    _A["bert_scan"] = cs.read_tsv(Path(tmp) / "bertopic_scan.tsv")
    _A["clause_topics"] = cs.read_tsv(Path(tmp) / "clause_topics.tsv")
    _A["overlap"] = cs.read_tsv(Path(tmp) / "community_overlap.tsv")
    _A["ari"] = cs.read_tsv(Path(tmp) / "community_ari.tsv")
    _A["recon"] = cs.read_tsv(Path(tmp) / "reconciliation.tsv")
    _A["diag"] = {r["metric"]: r["value"]
                  for r in cs.read_tsv(Path(tmp) / "diagnostics.tsv")}
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _A["summary"] = json.load(fh)


def recon(quantity):
    for r in _A["recon"]:
        if r["quantity"] == quantity:
            return r
    raise KeyError(quantity)


def ari(model):
    for r in _A["ari"]:
        if r["model"] == model:
            return r
    raise KeyError(model)


class TestReconciliation(unittest.TestCase):
    def test_clause_total_matches_39(self):
        self.assertEqual(_A["summary"]["n_clauses"], 5614)
        self.assertEqual(int(_A["diag"]["n_clauses"]), 5614)
        self.assertEqual(recon("clauses")["earlier"], "5,614 (#39)")

    def test_pages_match_39(self):
        self.assertEqual(_A["summary"]["n_pages"], 225)

    def test_content_tokens_stable(self):
        self.assertEqual(_A["summary"]["content_tokens"], 43413)
        self.assertEqual(int(_A["diag"]["content_tokens"]), 43413)

    def test_vocabulary_stable(self):
        self.assertEqual(_A["summary"]["vocab"], 1046)
        self.assertEqual(_A["summary"]["raw_types"], 5369)

    def test_all_network_nodes_in_vocab(self):
        self.assertEqual(_A["summary"]["community"]["n_network_nodes"], 391)
        self.assertEqual(_A["summary"]["community"]["n_shared_terms"], 391)
        self.assertEqual(recon("network nodes in vocab")["here"], "391")
        self.assertEqual(recon("network nodes in vocab")["earlier"],
                         "391 (#29)")

    def test_lda_nmf_sizes_sum_to_clauses(self):
        for name, rows in (("lda", _A["lda_sum"]), ("nmf", _A["nmf_sum"])):
            total = sum(int(r["size"]) for r in rows)
            self.assertEqual(total, 5614, name)
            self.assertEqual(int(recon(f"{name} cluster sizes sum")["here"]),
                             total)

    def test_bertopic_sizes_plus_noise_sum(self):
        total = sum(int(r["size"]) for r in _A["bert_sum"])
        total += _A["summary"]["bertopic"]["noise"]
        self.assertEqual(total, 5614)
        self.assertEqual(int(recon("bertopic sizes + noise")["here"]), total)


class TestCoherence(unittest.TestCase):
    def test_scan_grid_complete(self):
        for model in ("lda", "nmf"):
            ks = [int(r["k"]) for r in _A["coherence"]
                  if r["model"] == model]
            self.assertEqual(ks, sorted(tm.K_GRID), model)

    def test_coherence_values_finite(self):
        for r in _A["coherence"]:
            self.assertTrue(np.isfinite(float(r["coherence_c_v"])), r)
            self.assertTrue(np.isfinite(float(r["coherence_u_mass"])), r)

    def test_selected_k_is_best_coherence(self):
        for model in ("lda", "nmf"):
            rows = [r for r in _A["coherence"] if r["model"] == model]
            best = max(rows, key=lambda r: (float(r["coherence_c_v"]),
                                            -int(r["k"])))
            self.assertEqual(_A["summary"][model]["k"], int(best["k"]), model)

    def test_no_interior_peak_for_lda(self):
        rows = sorted((r for r in _A["coherence"] if r["model"] == "lda"),
                      key=lambda r: int(r["k"]))
        self.assertEqual(int(rows[0]["k"]), tm.K_GRID[0])
        self.assertGreater(float(rows[0]["coherence_c_v"]),
                           float(rows[-1]["coherence_c_v"]))

    def test_lda_selected_k_expected(self):
        self.assertEqual(_A["summary"]["lda"]["k"], 5)
        self.assertEqual(_A["summary"]["nmf"]["k"], 6)


class TestLDANMF(unittest.TestCase):
    def _check_topics(self, topics, k):
        by_topic = {}
        for r in topics:
            by_topic.setdefault(int(r["topic_id"]), []).append(r)
        self.assertEqual(len(by_topic), k)
        for tid, rows in by_topic.items():
            self.assertEqual(len(rows), tm.TOP_WORDS, tid)
            self.assertEqual([int(r["rank"]) for r in rows],
                             list(range(1, tm.TOP_WORDS + 1)))
            weights = [float(r["weight"]) for r in rows]
            self.assertEqual(weights, sorted(weights, reverse=True), tid)
            self.assertTrue(all(r["term"] for r in rows))

    def test_lda_topics_wellformed(self):
        self._check_topics(_A["lda_topics"], _A["summary"]["lda"]["k"])

    def test_nmf_topics_wellformed(self):
        self._check_topics(_A["nmf_topics"], _A["summary"]["nmf"]["k"])

    def test_lda_nmf_summaries_labelled(self):
        for rows in (_A["lda_sum"], _A["nmf_sum"]):
            for r in rows:
                self.assertTrue(r["label"])
                self.assertTrue(r["top_terms"])
                self.assertGreaterEqual(int(r["size"]), 0)

    def test_shared_vocabulary(self):
        """LDA and NMF must draw from one vocabulary (same terms possible)."""
        lda_terms = {r["term"] for r in _A["lda_topics"]}
        nmf_terms = {r["term"] for r in _A["nmf_topics"]}
        self.assertTrue(lda_terms & nmf_terms)


class TestBERTopic(unittest.TestCase):
    def test_scan_grid_complete(self):
        mts = [int(r["min_topic_size"]) for r in _A["bert_scan"]]
        self.assertEqual(mts, sorted(tm.BERT_MTS_GRID))

    def test_topics_and_noise_sum(self):
        bt = _A["summary"]["bertopic"]
        self.assertGreaterEqual(bt["n_topics"], 1)
        self.assertGreater(bt["noise"], 0)
        self.assertLess(bt["noise"], 5614)
        self.assertGreater(bt["noise_frac"], 0.0)
        self.assertLess(bt["noise_frac"], 1.0)
        self.assertEqual(int(_A["diag"]["bertopic_topics"]), bt["n_topics"])

    def test_reducer_is_umap(self):
        # umap-learn is installed in this environment; the build uses it.
        self.assertEqual(_A["summary"]["bertopic"]["reducer"], "umap")

    def test_topics_wellformed(self):
        rows = _A["bert_topics"]
        by_topic = {}
        for r in rows:
            by_topic.setdefault(int(r["topic_id"]), []).append(r)
        self.assertEqual(len(by_topic), _A["summary"]["bertopic"]["n_topics"])
        for tid, rr in by_topic.items():
            self.assertEqual(len(rr), tm.TOP_WORDS, tid)
            weights = [float(r["weight"]) for r in rr]
            self.assertEqual(weights, sorted(weights, reverse=True), tid)

    def test_ctfidf_terms_positive(self):
        for r in _A["bert_topics"]:
            self.assertGreater(float(r["weight"]), 0.0, r["term"])


class TestCommunity(unittest.TestCase):
    def test_ari_rows_present_and_bounded(self):
        models = {r["model"] for r in _A["ari"]}
        self.assertEqual(models, {"lda", "nmf", "bertopic"})
        for r in _A["ari"]:
            a = float(r["adjusted_rand"])
            self.assertGreaterEqual(a, -0.5)
            self.assertLessEqual(a, 1.0)
            self.assertEqual(int(r["n_terms"]), 391)
            self.assertEqual(int(r["n_communities"]), 6)

    def test_lda_converges_best(self):
        """LDA's term partition agrees most with #29 (still modest)."""
        lda = float(ari("lda")["adjusted_rand"])
        nmf = float(ari("nmf")["adjusted_rand"])
        bert = float(ari("bertopic")["adjusted_rand"])
        self.assertGreater(lda, nmf)
        self.assertGreater(nmf, bert)
        self.assertGreater(lda, 0.1)

    def test_overlap_rows_bounded(self):
        self.assertTrue(_A["overlap"])
        for r in _A["overlap"]:
            self.assertGreaterEqual(int(r["best_community"]), 0)
            self.assertLessEqual(int(r["best_community"]), 5)
            f = float(r["frac_top_in_community"])
            m = float(r["weighted_mass_frac"])
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)
            self.assertGreaterEqual(m, 0.0)
            self.assertLessEqual(m, 1.0)

    def test_overlap_covers_all_topics(self):
        by_model = {}
        for r in _A["overlap"]:
            by_model.setdefault(r["model"], set()).add(int(r["topic_id"]))
        for name, k in (("lda", _A["summary"]["lda"]["k"]),
                        ("nmf", _A["summary"]["nmf"]["k"]),
                        ("bertopic", _A["summary"]["bertopic"]["n_topics"])):
            self.assertEqual(by_model[name], set(range(k)), name)


class TestClauseTopics(unittest.TestCase):
    def test_rows_and_ids(self):
        rows = _A["clause_topics"]
        self.assertEqual(len(rows), 5614)
        all_ids = {int(r["clause_id"]) for r in cs.read_tsv(tm.CLAUSE_INDEX)}
        for r in rows:
            self.assertIn(int(r["clause_id"]), all_ids)

    def test_topics_in_range(self):
        k_lda = _A["summary"]["lda"]["k"]
        k_nmf = _A["summary"]["nmf"]["k"]
        for r in _A["clause_topics"]:
            self.assertGreaterEqual(int(r["lda_topic"]), 0)
            self.assertLess(int(r["lda_topic"]), k_lda)
            self.assertGreaterEqual(int(r["nmf_topic"]), 0)
            self.assertLess(int(r["nmf_topic"]), k_nmf)

    def test_bertopic_noise_flag_consistent(self):
        n_noise = 0
        for r in _A["clause_topics"]:
            t = int(r["bertopic_topic"])
            flag = int(r["bertopic_noise"])
            if t == -1:
                self.assertEqual(flag, 1)
                n_noise += 1
            else:
                self.assertEqual(flag, 0)
        self.assertEqual(n_noise, _A["summary"]["bertopic"]["noise"])


class TestDeterminism(unittest.TestCase):
    def test_two_builds_identical(self):
        a = _A["dir"]
        b = Path(tempfile.mkdtemp(prefix="topics_det_"))
        tm.build(b, quiet=True)
        for name in ("coherence_scan.tsv", "lda_topics.tsv", "nmf_topics.tsv",
                     "bertopic_topics.tsv", "lda_topic_summary.tsv",
                     "nmf_topic_summary.tsv", "bertopic_topic_summary.tsv",
                     "bertopic_scan.tsv", "community_ari.tsv",
                     "community_overlap.tsv", "clause_topics.tsv",
                     "reconciliation.tsv", "diagnostics.tsv", "summary.json"):
            pa = a / name
            pb = b / name
            self.assertTrue(pb.exists(), name)
            self.assertEqual(pa.read_text(), pb.read_text(), name)


class TestHelpers(unittest.TestCase):
    def test_unescape(self):
        self.assertEqual(tm.unescape("a\\nb\\tc\\\\d"), "a b c d")

    def test_tokenize_stopwords_removed(self):
        toks = tm.tokenize("The internal organ of the body")
        self.assertNotIn("the", toks)
        self.assertNotIn("of", toks)
        self.assertIn("internal", toks)
        self.assertIn("organ", toks)

    def test_tokenize_keeps_fused_units(self):
        toks = tm.tokenize("gene=tv soul/gram boy-roid")
        self.assertEqual(toks, ["gene=tv", "soul/gram", "boy-roid"])

    def test_topic_words_coercion(self):
        self.assertEqual(tm._topic_words([[("a", 1.0), ("b", 2.0)]]),
                         [["a", "b"]])
        self.assertEqual(tm._topic_words([["a", "b"]]), [["a", "b"]])

    def test_top_terms_tie_break_by_term(self):
        vocab = ["b", "a", "c"]
        comp = np.array([[1.0, 1.0, 0.5]])
        rows = tm.top_terms(comp, vocab, n=2)
        self.assertEqual([t for t, _w in rows[0]], ["a", "b"])

    def test_ctfidf_matrix(self):
        docs = [["a", "b"], ["a"], ["c"]]
        vocab_index = {"a": 0, "b": 1, "c": 2}
        counts, ctfidf, cids = tm.ctfidf_matrix([0, 0, 1], docs, vocab_index)
        self.assertEqual(cids, [0, 1])
        self.assertEqual(counts.shape, (2, 3))
        self.assertAlmostEqual(counts[0, 0], 2.0)   # a in cluster 0
        self.assertAlmostEqual(counts[0, 1], 1.0)   # b in cluster 0
        self.assertAlmostEqual(counts[1, 2], 1.0)   # c in cluster 1
        # within cluster 0, term a has higher c-TF-IDF than b
        self.assertGreater(ctfidf[0, 0], ctfidf[0, 1])

    def test_ctfidf_excludes_noise(self):
        docs = [["a"], ["a"], ["b"]]
        vocab_index = {"a": 0, "b": 1}
        counts, _c, cids = tm.ctfidf_matrix([0, 0, -1], docs, vocab_index)
        self.assertEqual(cids, [0])
        self.assertAlmostEqual(counts[0, 1], 0.0)

    def test_term_partition_ari(self):
        comm_of = {"a": 0, "b": 0, "c": 1, "d": 1}
        same = {"a": 0, "b": 0, "c": 1, "d": 1}
        r = tm.term_partition_ari(same, comm_of, ["a", "b", "c", "d"])
        self.assertEqual(r["adjusted_rand"], 1.0)
        self.assertEqual(r["n_terms"], 4)
        self.assertEqual(r["n_communities"], 2)

    def test_community_overlap_picks_mass_community(self):
        topics = [[("a", 3.0), ("b", 1.0), ("x", 1.0)]]
        comm_of = {"a": 0, "b": 0, "x": 1}
        rows = tm.community_overlap("t", topics, comm_of, {0: "d0", 1: "d1"})
        self.assertEqual(rows[0]["best_community"], 0)
        self.assertAlmostEqual(rows[0]["weighted_mass_frac"], 4.0 / 5.0, 4)
        self.assertEqual(rows[0]["n_shared_top"], 2)


if __name__ == "__main__":
    unittest.main()
