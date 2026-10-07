#!/usr/bin/env python3
"""Regression suite for #40 clustering to motifs and scenes.

Builds the artifact into a temp directory (full 5,614-clause build over the
real #39 vectors) and checks the invariants that matter:

  * the clause set reconciles with #39 / #33 / #17 / #27 exactly;
  * the KMeans scan runs and the catalogue k is the disclosed fixed k;
  * cluster sizes sum to the clause count; every cluster has a label,
    distinctive terms and valid exemplar clauses (ids join to #33);
  * HDBSCAN clusters + noise sum to the clause count; noise is disclosed;
  * the task's named motifs are located with the exact clause counts;
  * the raw-view robustness check produces an adjusted Rand in [0, 1];
  * two builds produce identical tables (determinism).

    python work/test_clustering.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clustering as cl  # noqa: E402
import clause_segmentation as cs  # noqa: E402

_A = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="clustering_test_")
    cl.build(Path(tmp), quiet=True)
    _A["dir"] = Path(tmp)
    _A["clusters"] = cs.read_tsv(Path(tmp) / "clusters.tsv")
    _A["exemplars"] = cs.read_tsv(Path(tmp) / "cluster_exemplars.tsv")
    _A["terms"] = cs.read_tsv(Path(tmp) / "cluster_terms.tsv")
    _A["kscan"] = cs.read_tsv(Path(tmp) / "kmeans_scan.tsv")
    _A["hscan"] = cs.read_tsv(Path(tmp) / "hdbscan_scan.tsv")
    _A["hclusters"] = cs.read_tsv(Path(tmp) / "hdbscan_clusters.tsv")
    _A["motifs"] = cs.read_tsv(Path(tmp) / "motif_checks.tsv")
    _A["robust"] = cs.read_tsv(Path(tmp) / "robustness.tsv")
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


def motif(name):
    for r in _A["motifs"]:
        if r["motif"] == name:
            return r
    raise KeyError(name)


class TestReconciliation(unittest.TestCase):
    def test_clause_total_matches_39(self):
        self.assertEqual(_A["summary"]["n_clauses"], 5614)
        self.assertEqual(recon("clauses")["earlier"], "5,614 (#39)")

    def test_pages_match_39(self):
        self.assertEqual(_A["summary"]["n_pages"], 225)

    def test_screened_units_match_27(self):
        self.assertEqual(recon("screened units")["here"], "74,077")
        self.assertEqual(recon("screened units")["earlier"], "74,077 (#27)")

    def test_kmeans_sizes_sum_to_clauses(self):
        total = sum(int(r["size"]) for r in _A["clusters"])
        self.assertEqual(total, 5614)
        self.assertEqual(int(recon("kmeans cluster sizes sum")["here"]), total)

    def test_hdbscan_clusters_plus_noise_sum_to_clauses(self):
        if not _A["summary"]["hdbscan"]["available"]:
            self.skipTest("hdbscan not available")
        total = sum(int(r["size"]) for r in _A["hclusters"])
        total += _A["summary"]["hdbscan"]["noise"]
        self.assertEqual(total, 5614)

    def test_empty_text_clauses_disclosed(self):
        self.assertEqual(_A["summary"]["n_empty_text_clauses"], 36)
        self.assertEqual(int(_A["diag"]["n_empty_text_clauses"]), 36)


class TestKMeans(unittest.TestCase):
    def test_scan_grid_complete(self):
        ks = [int(r["k"]) for r in _A["kscan"]]
        self.assertEqual(ks, sorted(cl.DEFAULT_K_GRID))

    def test_catalogue_k_is_fixed_default(self):
        self.assertEqual(_A["summary"]["kmeans"]["k"], cl.DEFAULT_K)
        self.assertEqual(_A["summary"]["kmeans"]["k_fixed_default"],
                         cl.DEFAULT_K)
        cat = [r for r in _A["kscan"] if r["is_catalogue_k"] == "True"]
        self.assertEqual(len(cat), 1)
        self.assertEqual(int(cat[0]["k"]), cl.DEFAULT_K)

    def test_silhouette_low_and_rising(self):
        """The continuum signature: no interior peak, silhouette rises."""
        sils = [float(r["silhouette"]) for r in _A["kscan"]]
        self.assertTrue(all(0 < s < 0.2 for s in sils),
                        f"silhouettes not low: {sils}")
        self.assertGreater(sils[-1], sils[0],
                           "silhouette does not rise with k")

    def test_cluster_count_and_size_range(self):
        self.assertEqual(len(_A["clusters"]), cl.DEFAULT_K)
        sizes = [int(r["size"]) for r in _A["clusters"]]
        self.assertGreaterEqual(min(sizes), 100)
        self.assertLessEqual(max(sizes), 1000)


class TestCatalogue(unittest.TestCase):
    def test_every_cluster_labelled(self):
        for r in _A["clusters"]:
            self.assertTrue(r["label"], f"cluster {r['cluster_id']} unlabelled")
            self.assertTrue(r["top_terms"], f"cluster {r['cluster_id']} "
                                            "has no terms")

    def test_distinctive_terms_present_and_positive(self):
        by_cluster = {}
        for r in _A["terms"]:
            by_cluster.setdefault(r["cluster_id"], []).append(r)
        self.assertEqual(len(by_cluster), len(_A["clusters"]))
        for cid, rows in by_cluster.items():
            self.assertGreaterEqual(len(rows), 1)
            for r in rows:
                self.assertGreater(float(r["z"]), 0.0)
                self.assertGreaterEqual(int(r["f_in"]), cl.MIN_TERM_F)

    def test_exemplars_valid_and_ranked(self):
        all_ids = {int(r["clause_id"]) for r in cs.read_tsv(cl.CLAUSES)}
        by_cluster = {}
        for r in _A["exemplars"]:
            by_cluster.setdefault(r["cluster_id"], []).append(r)
        self.assertEqual(len(by_cluster), len(_A["clusters"]))
        for cid, rows in by_cluster.items():
            self.assertEqual(len(rows), cl.TOP_EXEMPLARS)
            self.assertEqual([int(r["rank"]) for r in rows],
                             list(range(1, cl.TOP_EXEMPLARS + 1)))
            for r in rows:
                self.assertIn(int(r["clause_id"]), all_ids)
                cos = float(r["cos_to_centroid"])
                self.assertGreater(cos, 0.0)
                self.assertLessEqual(cos, 1.0)

    def test_medoid_and_centroid_clauses_are_members(self):
        """medoid/centroid clause ids are valid #33 clause ids."""
        all_ids = {int(r["clause_id"]) for r in cs.read_tsv(cl.CLAUSES)}
        for r in _A["clusters"]:
            self.assertIn(int(r["medoid_clause"]), all_ids)
            self.assertIn(int(r["centroid_clause"]), all_ids)

    def test_spread_classes_valid(self):
        valid = {"scene", "motif", "mixed"}
        for r in _A["clusters"]:
            self.assertIn(r["spread_class"], valid)
            self.assertGreaterEqual(int(r["n_pages"]), 1)
            self.assertLessEqual(int(r["n_pages"]), 225)
            share = float(r["top5_page_share"])
            self.assertGreater(share, 0.0)
            self.assertLessEqual(share, 1.0)

    def test_mean_within_cos_bounded(self):
        for r in _A["clusters"]:
            if r["mean_within_cos"] == "":
                continue  # singleton
            v = float(r["mean_within_cos"])
            self.assertGreater(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestHDBSCAN(unittest.TestCase):
    def test_scan_grid_complete(self):
        if not _A["summary"]["hdbscan"]["available"]:
            self.skipTest("hdbscan not available")
        mcs = [int(r["min_cluster_size"]) for r in _A["hscan"]]
        self.assertEqual(mcs, sorted(cl.DEFAULT_MCS_GRID))

    def test_noise_disclosed_and_majority(self):
        if not _A["summary"]["hdbscan"]["available"]:
            self.skipTest("hdbscan not available")
        noise = _A["summary"]["hdbscan"]["noise"]
        self.assertGreater(noise, 0)
        self.assertGreater(noise / 5614, 0.5,
                       "expected a majority of clauses to be density-noise")
        self.assertEqual(int(_A["diag"]["hdbscan_noise"]), noise)

    def test_clusters_have_labels(self):
        if not _A["summary"]["hdbscan"]["available"]:
            self.skipTest("hdbscan not available")
        for r in _A["hclusters"]:
            self.assertTrue(r["label"])

    def test_noise_sample_written(self):
        if not _A["summary"]["hdbscan"]["available"]:
            self.skipTest("hdbscan not available")
        path = _A["dir"] / "hdbscan_noise_sample.tsv"
        self.assertTrue(path.exists())
        rows = cs.read_tsv(path)
        self.assertGreater(len(rows), 0)
        self.assertLessEqual(len(rows), 50)


class TestMotifs(unittest.TestCase):
    def test_named_motif_dog_murder_gene(self):
        m = motif("dog+murder+gene")
        self.assertEqual(int(m["n_clauses"]), 15)
        self.assertGreater(int(m["n_in_top_cluster"]), 0)
        self.assertTrue(m["example_text"])

    def test_named_motif_brain_load_ecstasy(self):
        m = motif("brain+load+ecstasy")
        self.assertEqual(int(m["n_clauses"]), 3)
        self.assertGreater(int(m["n_in_top_cluster"]), 0)

    def test_seed_motifs_located(self):
        # the #27/#39 motif set concentrates in single clusters
        for name in ("artificial sun", "drug embryo", "ice nebula", "adam"):
            m = motif(name)
            self.assertGreater(int(m["n_clauses"]), 0, name)
            self.assertGreater(float(m["frac_in_top_cluster"]), 0.0, name)

    def test_unattested_motif_reported(self):
        m = motif("necropolis")
        self.assertEqual(int(m["n_clauses"]), 0)


class TestRobustness(unittest.TestCase):
    def test_adjusted_rand_present_and_bounded(self):
        row = _A["robust"][0]
        self.assertEqual(row["check"], "adjusted_rand_units_vs_raw")
        ari = float(row["value"])
        self.assertGreater(ari, 0.0)
        self.assertLessEqual(ari, 1.0)

    def test_raw_view_agreement_moderate(self):
        """The operator strings move clauses but not the motif structure."""
        ari = float(_A["robust"][0]["value"])
        self.assertGreater(ari, 0.3)


class TestDeterminism(unittest.TestCase):
    def test_two_builds_identical(self):
        a = tempfile.mkdtemp(prefix="cl_det_a_")
        b = tempfile.mkdtemp(prefix="cl_det_b_")
        cl.build(Path(a), quiet=True)
        cl.build(Path(b), quiet=True)
        for name in ("clusters.tsv", "cluster_exemplars.tsv",
                     "cluster_terms.tsv", "kmeans_scan.tsv",
                     "hdbscan_scan.tsv", "hdbscan_clusters.tsv",
                     "motif_checks.tsv", "robustness.tsv"):
            pa = Path(a) / name
            pb = Path(b) / name
            if not pa.exists():
                self.assertFalse(pb.exists(), name)
                continue
            self.assertEqual(pa.read_text(), pb.read_text(), name)


class TestHelpers(unittest.TestCase):
    def test_spread_class_thresholds(self):
        self.assertEqual(cl.spread_class(10, 0.6), "scene")
        self.assertEqual(cl.spread_class(25, 0.5), "scene")
        self.assertEqual(cl.spread_class(26, 0.9), "mixed")
        self.assertEqual(cl.spread_class(100, 0.1), "motif")
        self.assertEqual(cl.spread_class(50, 0.3), "mixed")

    def test_distinctive_terms_prior(self):
        in_c = __import__("collections").Counter(
            "dog dog dog murder gene gene tv".split())
        out_c = __import__("collections").Counter(
            "dog murder sun sun sun brain brain brain brain".split())
        rows = cl.distinctive_terms(in_c, 7, out_c, 11, top_n=5)
        self.assertTrue(rows)
        terms = [r[0] for r in rows]
        self.assertIn("dog", terms)
        # z-scores are positive and sorted desc
        zs = [r[1] for r in rows]
        self.assertEqual(zs, sorted(zs, reverse=True))
        self.assertTrue(all(z > 0 for z in zs))

    def test_mean_within_cos_singleton(self):
        import numpy as np
        X = np.eye(4)
        self.assertIsNone(cl.mean_within_cos(X, [2]))
        v = cl.mean_within_cos(X, [0, 1])
        self.assertAlmostEqual(v, 0.0)

    def test_medoid_index_ties_break_low(self):
        import numpy as np
        X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(cl.medoid_index(X, [0, 1, 2]), 0)


if __name__ == "__main__":
    unittest.main()
