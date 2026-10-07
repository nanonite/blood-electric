#!/usr/bin/env python3
"""Regression suite for #29 co-occurrence network and community detection.

Runs the builder once into a temp directory, then checks the invariants that
matter: the markup screen and function-word removal, the PMI formula, the
graph construction, the Louvain communities, the centrality measures, the
reconciliation with #17/#19/#27, and build determinism.

    python work/test_cooccurrence_network.py
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cooccurrence_network as cn

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="network_test_")
    cn.main(["--out", tmp, "--quiet"])
    for name in ("graph_nodes", "graph_edges", "communities", "centrality",
                 "diagnostics", "reconciliation"):
        _ARTIFACTS[name] = cn.read_tsv(Path(tmp) / f"{name}.tsv")
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    _ARTIFACTS["_dir"] = tmp
    _ARTIFACTS["report"] = (Path(tmp) / "network_report.md").read_text(
        encoding="utf-8")


def rows(name):
    return _ARTIFACTS[name]


def diag():
    return {r["diag"]: r["value"] for r in rows("diagnostics")}


def summary():
    return _ARTIFACTS["summary"]


# ---------------------------------------------------------------------------
class TestStream(unittest.TestCase):
    def test_screened_items_match_27(self):
        d = diag()
        self.assertEqual(d["screened_items"], "74077")      # #27
        self.assertEqual(d["markup_units_removed"], "588")  # 347 #19 + 241

    def test_function_words_removed(self):
        d = diag()
        # the closed class is removed, not silently dropped
        self.assertGreater(int(d["function_words_removed"]), 0)
        self.assertEqual(
            int(d["content_items"]),
            int(d["screened_items"]) - int(d["function_words_removed"]))

    def test_content_items_are_screened_minus_stopwords(self):
        d = diag()
        self.assertEqual(d["content_items"], "43321")

    def test_markup_absent_from_nodes(self):
        bad = {"href", "mailto", "colspan", "th", "p", "pre", "br", "li",
               "hr", "td", "tr"}
        for r in rows("graph_nodes"):
            self.assertNotIn(r["term"], bad)

    def test_stopwords_absent_from_nodes(self):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop = set(ENGLISH_STOP_WORDS) | cn.EXTRA_STOP
        for r in rows("graph_nodes"):
            self.assertNotIn(r["term"], stop)


# ---------------------------------------------------------------------------
class TestPMI(unittest.TestCase):
    def test_pmi_formula(self):
        # f_xy=10, f_x=100, f_y=200, N=1000 -> log2(10*1000/(100*200))
        self.assertAlmostEqual(cn.pmi(10, 100, 200, 1000),
                               math.log2(10 * 1000 / (100 * 200)))

    def test_pmi_zero_when_any_count_zero(self):
        self.assertEqual(cn.pmi(0, 100, 200, 1000), 0.0)
        self.assertEqual(cn.pmi(10, 0, 200, 1000), 0.0)
        self.assertEqual(cn.pmi(10, 100, 0, 1000), 0.0)
        self.assertEqual(cn.pmi(10, 100, 200, 0), 0.0)

    def test_pmi_positive_for_associated_pair(self):
        # co-occur more than chance -> positive PMI
        self.assertGreater(cn.pmi(50, 100, 100, 1000), 0.0)

    def test_pmi_negative_for_anti_associated_pair(self):
        # co-occur less than chance -> negative PMI
        self.assertLess(cn.pmi(1, 500, 500, 1000), 0.0)


# ---------------------------------------------------------------------------
class TestGraph(unittest.TestCase):
    def test_graph_sizes(self):
        d = diag()
        self.assertEqual(d["nodes"], str(len(rows("graph_nodes"))))
        self.assertEqual(d["edges"], str(len(rows("graph_edges"))))
        self.assertGreater(int(d["nodes"]), 300)
        self.assertGreater(int(d["edges"]), 5000)

    def test_nodes_have_frequency_at_or_above_floor(self):
        d = diag()
        floor = int(d["min_freq"])
        for r in rows("graph_nodes"):
            self.assertGreaterEqual(int(r["freq"]), floor)

    def test_edges_meet_pmi_and_cooccur_floors(self):
        d = diag()
        min_pmi = float(d["min_pmi"])
        min_cooccur = int(d["min_cooccur"])
        for r in rows("graph_edges"):
            self.assertGreaterEqual(float(r["pmi"]), min_pmi)
            self.assertGreaterEqual(int(r["f_xy"]), min_cooccur)

    def test_edges_reference_known_nodes(self):
        nodes = {r["term"] for r in rows("graph_nodes")}
        for r in rows("graph_edges"):
            self.assertIn(r["x"], nodes)
            self.assertIn(r["y"], nodes)

    def test_edge_pmi_matches_formula(self):
        d = diag()
        n = int(d["content_items"])
        for r in rows("graph_edges")[:50]:
            expect = math.log2(
                int(r["f_xy"]) * n / (int(r["f_x"]) * int(r["f_y"])))
            self.assertAlmostEqual(float(r["pmi"]), round(expect, 4),
                                   places=3)

    def test_graph_gexf_exists(self):
        self.assertTrue((Path(_ARTIFACTS["_dir"]) / "graph.gexf").exists())


# ---------------------------------------------------------------------------
class TestCommunities(unittest.TestCase):
    def test_communities_detected(self):
        d = diag()
        self.assertGreaterEqual(int(d["communities"]), 2)

    def test_every_node_assigned_exactly_once(self):
        nodes = {r["term"] for r in rows("graph_nodes")}
        assigned = [r["term"] for r in rows("communities")]
        self.assertEqual(len(assigned), len(nodes))
        self.assertEqual(set(assigned), nodes)

    def test_community_ids_are_contiguous(self):
        ids = sorted({int(r["community"]) for r in rows("communities")})
        self.assertEqual(ids, list(range(len(ids))))

    def test_community_sizes_match_summary(self):
        sizes = {}
        for r in rows("communities"):
            sizes[int(r["community"])] = sizes.get(int(r["community"]), 0) + 1
        self.assertEqual(sorted(sizes.values(), reverse=True),
                         summary()["community_sizes"])

    def test_report_names_the_domains(self):
        report = _ARTIFACTS["report"]
        for dom in ("body", "machine", "violence", "sex", "genetics",
                    "corporate"):
            self.assertIn(dom, report)


# ---------------------------------------------------------------------------
class TestCentrality(unittest.TestCase):
    def test_centrality_covers_every_node(self):
        nodes = {r["term"] for r in rows("graph_nodes")}
        cent = {r["term"] for r in rows("centrality")}
        self.assertEqual(nodes, cent)

    def test_centrality_measures_present(self):
        for r in rows("centrality"):
            for m in ("degree", "strength", "betweenness", "eigenvector",
                      "pagerank"):
                self.assertIn(m, r)
                self.assertGreaterEqual(float(r[m]), 0.0)

    def test_centrality_values_bounded(self):
        for r in rows("centrality"):
            self.assertLessEqual(float(r["degree"]), 1.0)
            self.assertLessEqual(float(r["betweenness"]), 1.0)
            self.assertLessEqual(float(r["eigenvector"]), 1.0)
            self.assertLessEqual(float(r["pagerank"]), 1.0)

    def test_summary_top_betweenness_matches_table(self):
        top = summary()["top_betweenness"]
        table = sorted(rows("centrality"),
                       key=lambda r: -float(r["betweenness"]))[:len(top)]
        self.assertEqual([t["term"] for t in top],
                         [r["term"] for r in table])


# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_build_is_reproducible(self):
        tmp = tempfile.mkdtemp(prefix="network_det_")
        cn.main(["--out", tmp, "--quiet"])
        d = {r["diag"]: r["value"] for r in cn.read_tsv(
            Path(tmp) / "diagnostics.tsv")}
        self.assertEqual(d["communities"], diag()["communities"])
        self.assertEqual(d["nodes"], diag()["nodes"])
        self.assertEqual(d["edges"], diag()["edges"])
        # the community assignment itself must be identical
        a = {(r["community"], r["term"]) for r in rows("communities")}
        b = {(r["community"], r["term"]) for r in cn.read_tsv(
            Path(tmp) / "communities.tsv")}
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
class TestArtifacts(unittest.TestCase):
    def test_all_artifacts_exist_and_are_non_empty(self):
        for name in ("graph_nodes", "graph_edges", "communities",
                     "centrality", "diagnostics", "reconciliation"):
            self.assertGreater(len(rows(name)), 0)

    def test_tsv_columns_have_no_embedded_tabs_or_newlines(self):
        for name in ("graph_nodes", "graph_edges", "communities",
                     "centrality"):
            for r in rows(name):
                for v in r.values():
                    self.assertNotIn("\t", v)
                    self.assertNotIn("\n", v)

    def test_summary_json_round_trips(self):
        s = summary()
        self.assertEqual(s["schema"], "network/1")
        self.assertIn("counts", s)
        self.assertIn("top_betweenness", s)
        self.assertIn("community_sizes", s)

    def test_reconciliation_agrees_with_17_19_27(self):
        recon = {r["quantity"]: r for r in rows("reconciliation")}
        self.assertEqual(recon["screened occurrences"]["earlier"], "74,318 (#19/#24)")
        self.assertEqual(recon["screened occurrences"]["here"], "74077")
        self.assertEqual(recon["collocation items"]["earlier"], "74,077 (#27)")


if __name__ == "__main__":
    unittest.main()
