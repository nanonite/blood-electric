#!/usr/bin/env python3
"""Regression suite for #43 contrastive semantic axes.

Builds the artifact into a temp directory (full 5,614-clause build over the
real #39 vectors, seed words encoded with the pinned #39 encoder) and checks
the invariants that matter:

  * the clause set reconciles with #39 / #33 / #27 exactly;
  * the cluster partition is #40's (sizes, medoids, cross-check note);
  * all four named axes are present, with the disclosed pole terms, and every
    axis beats its randomized-axis control and its random-direction null;
  * the ordering finding is pinned: organic<->machine is the strongest
    separator, self<->other the weakest (with its thin other-pole disclosed);
  * every seed pair's cosine is in range and the exemplar/projection tables
    are internally consistent;
  * all 66 cluster-pair dimensions are present, sorted, interpreted;
  * two builds produce byte-identical tables (determinism).

    python work/test_semantic_axes.py
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

import clause_segmentation as cs  # noqa: E402
import semantic_axes as sa  # noqa: E402

_A = {}
AXIS_NAMES = ("organic<->machine", "self<->other", "body<->information",
              "sacred<->profane")


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="semaxes_test_")
    sa.build(Path(tmp), quiet=True)
    _A["dir"] = Path(tmp)
    _A["axes"] = {r["axis"]: r for r in cs.read_tsv(Path(tmp) / "axes.tsv")}
    _A["pairs"] = cs.read_tsv(Path(tmp) / "seed_pairs.tsv")
    _A["proj"] = cs.read_tsv(Path(tmp) / "axis_projections.tsv")
    _A["extreme"] = cs.read_tsv(Path(tmp) / "extreme_clauses.tsv")
    _A["pairdims"] = cs.read_tsv(Path(tmp) / "cluster_pair_axes.tsv")
    _A["controls"] = cs.read_tsv(Path(tmp) / "controls.tsv")
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


class TestReconciliation(unittest.TestCase):
    def test_clause_total_matches_39(self):
        self.assertEqual(_A["summary"]["n_clauses"], 5614)
        self.assertEqual(recon("clauses")["earlier"], "5,614 (#39)")

    def test_pages_match_39(self):
        self.assertEqual(_A["summary"]["n_pages"], 225)

    def test_screened_units_match_27(self):
        self.assertEqual(recon("screened units")["here"], "74,077")
        self.assertEqual(recon("screened units")["earlier"], "74,077 (#27)")

    def test_view_and_encoder_pinned(self):
        self.assertEqual(_A["summary"]["view"], "units")
        self.assertEqual(_A["summary"]["encoder"], "sbert")

    def test_partition_is_40s(self):
        km = _A["summary"]["kmeans"]
        self.assertEqual(km["k"], 12)
        self.assertEqual(km["n_init"], 10)
        self.assertEqual(km["seed"], 0)
        self.assertIn("reproduced exactly", km["catalogue_crosscheck"])
        self.assertEqual(sum(km["sizes"].values()), 5614)

    def test_partition_matches_40_catalogue(self):
        """Cross-check against #40's persisted catalogue when present."""
        cat_path = sa.CLUSTERING_DIR / "clusters.tsv"
        if not cat_path.exists():
            self.skipTest("#40 catalogue not present")
        cat = {int(r["cluster_id"]): r for r in cs.read_tsv(cat_path)}
        sizes = _A["summary"]["kmeans"]["sizes"]
        for cid, r in cat.items():
            self.assertEqual(int(r["size"]), sizes[str(cid)],
                             f"cluster {cid} size differs from #40")

    def test_empty_text_clauses_disclosed(self):
        self.assertEqual(_A["summary"]["n_empty_text_clauses"], 36)
        self.assertEqual(_A["summary"]["empty_clusters"], {"9": 36})

    def test_axis_count(self):
        self.assertEqual(len(_A["axes"]), 4)
        self.assertEqual(recon("axis count")["earlier"], "4 (#43 task text)")


class TestAxes(unittest.TestCase):
    def test_four_named_axes(self):
        self.assertEqual(set(_A["axes"]), set(AXIS_NAMES))

    def test_poles_are_the_disclosed_seed_terms(self):
        for name, spec in sa.AXES.items():
            self.assertEqual(len(spec["plus"]), len(spec["minus"]), name)
            self.assertGreaterEqual(len(spec["plus"]), 5, name)
            row = _A["axes"][name]
            self.assertEqual(row["plus_pole"].split(), list(spec["plus"]))
            self.assertEqual(row["minus_pole"].split(), list(spec["minus"]))

    def test_every_seed_term_occurs(self):
        for term, n in _A["summary"]["seed_terms"].items():
            self.assertGreaterEqual(n, 1, f"seed term {term!r} unattested")

    def test_all_axes_beat_randomized_axis_control(self):
        for name, r in _A["axes"].items():
            self.assertGreater(float(r["control_z"]), 2.0, name)
            self.assertGreater(float(r["mean_pair_cos"]),
                               float(r["control_mean"]), name)

    def test_all_axes_beat_random_direction_null(self):
        for name, r in _A["axes"].items():
            self.assertGreater(float(r["spread_z"]), 2.0, name)
            self.assertGreater(float(r["cluster_spread"]),
                               float(r["spread_null_mean"]), name)

    def test_clause_contrast_positive_when_both_poles_present(self):
        for name, r in _A["axes"].items():
            if int(r["n_minus_clauses"]) > 0:
                self.assertGreater(float(r["contrast"]), 0.0, name)
                self.assertGreater(float(r["contrast_d"]), 0.0, name)

    def test_verdicts(self):
        for name, r in _A["axes"].items():
            self.assertEqual(r["verdict"], "supported", name)

    def test_organic_machine_strongest_separator(self):
        self.assertEqual(
            max(_A["axes"], key=lambda n: float(_A["axes"][n]["cluster_spread"])),
            "organic<->machine")

    def test_self_other_weakest_separator(self):
        """The reported ordering: self<->other separates least (thin other
        pole, pronoun-heavy self pole)."""
        self.assertEqual(
            min(_A["axes"], key=lambda n: float(_A["axes"][n]["cluster_spread"])),
            "self<->other")
        self.assertLess(int(_A["axes"]["self<->other"]["n_minus_clauses"]), 50)


class TestTables(unittest.TestCase):
    def test_seed_pairs_rows_consistent(self):
        rows = _A["pairs"]
        self.assertEqual(len(rows), sum(int(r["n_pairs"])
                                        for r in _A["axes"].values()))
        for r in rows:
            self.assertIn(r["axis"], AXIS_NAMES)
            self.assertGreaterEqual(float(r["cos_to_axis"]), -1.0)
            self.assertLessEqual(float(r["cos_to_axis"]), 1.0)
            self.assertGreaterEqual(int(r["plus_clauses"]), 1)
            self.assertGreaterEqual(int(r["minus_clauses"]), 1)

    def test_axis_projections_table(self):
        rows = _A["proj"]
        self.assertEqual(len(rows), 12)
        self.assertEqual([int(r["cluster_id"]) for r in rows],
                         list(range(12)))
        self.assertEqual(sum(int(r["size"]) for r in rows), 5614)
        for r in rows:
            for name in AXIS_NAMES:
                self.assertTrue(-1.0 <= float(r[name]) <= 1.0, (r, name))
            self.assertTrue(r["label"])

    def test_extreme_clauses_table(self):
        rows = _A["extreme"]
        self.assertEqual(len(rows), 4 * 2 * sa.TOP_EXTREME)
        for name in AXIS_NAMES:
            for pole in ("plus", "minus"):
                sub = [r for r in rows if r["axis"] == name
                       and r["pole"] == pole]
                self.assertEqual(len(sub), sa.TOP_EXTREME)
                self.assertEqual([int(r["rank"]) for r in sub],
                                 list(range(1, sa.TOP_EXTREME + 1)))
                vals = [float(r["projection"]) for r in sub]
                if pole == "plus":
                    self.assertEqual(vals, sorted(vals, reverse=True))
                else:
                    self.assertEqual(vals, sorted(vals))
        all_ids = {int(r["clause_id"]) for r in
                   cs.read_tsv(sa.STORE_DIR / "clause_index.tsv")}
        for r in rows:
            self.assertIn(int(r["clause_id"]), all_ids)

    def test_cluster_pair_dimensions(self):
        rows = _A["pairdims"]
        self.assertEqual(len(rows), 66)          # 12 choose 2
        seps = [float(r["separation"]) for r in rows]
        self.assertEqual(seps, sorted(seps, reverse=True))
        for i, r in enumerate(rows, start=1):
            self.assertEqual(int(r["pair_id"]), i)
            self.assertLess(int(r["cluster_a"]), int(r["cluster_b"]))
            self.assertEqual(len(r["top_seed_loadings"].split(", ")),
                             sa.TOP_PAIR_TERMS)
            for name in AXIS_NAMES:
                self.assertTrue(-1.0 <= float(r[f"align_{name}"]) <= 1.0)

    def test_controls_table_matches_axes(self):
        rows = _A["controls"]
        self.assertEqual(len(rows), 8)           # 2 controls x 4 axes
        by = {(r["axis"], r["control"].split(" ")[0]): r for r in rows}
        for name, r in _A["axes"].items():
            self.assertAlmostEqual(float(by[(name, "randomized_axis")]["z"]),
                                   float(r["control_z"]), places=2)
            self.assertAlmostEqual(float(by[(name, "random_direction")]["z"]),
                                   float(r["spread_z"]), places=2)


class TestArtifacts(unittest.TestCase):
    def test_report_exists_with_sections(self):
        path = _A["dir"] / "semantic_axes_report.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        for section in ("# Contrastive semantic axes (#43)",
                        "## 1. Method",
                        "## 3. The four named axes",
                        "## 4. Seed-pair contrasts",
                        "## 6. Cluster-pair dimensions",
                        "## 8. Negative results and limitations"):
            self.assertIn(section, text)

    def test_all_files_present(self):
        for name in ("axes.tsv", "seed_pairs.tsv", "axis_projections.tsv",
                     "extreme_clauses.tsv", "cluster_pair_axes.tsv",
                     "controls.tsv", "reconciliation.tsv",
                     "diagnostics.tsv", "summary.json",
                     "semantic_axes_report.md"):
            self.assertTrue((_A["dir"] / name).exists(), name)

    def test_diagnostics(self):
        self.assertEqual(int(_A["diag"]["n_clauses"]), 5614)
        self.assertEqual(int(_A["diag"]["dim"]), 384)
        self.assertEqual(int(_A["diag"]["kmeans_k"]), 12)
        self.assertEqual(int(_A["diag"]["n_axes"]), 4)
        self.assertEqual(int(_A["diag"]["n_cluster_pairs"]), 66)


class TestDeterminism(unittest.TestCase):
    def test_two_builds_identical(self):
        b = Path(tempfile.mkdtemp(prefix="semaxes_det_"))
        sa.build(b, quiet=True)
        for name in ("axes.tsv", "seed_pairs.tsv", "axis_projections.tsv",
                     "extreme_clauses.tsv", "cluster_pair_axes.tsv",
                     "controls.tsv", "reconciliation.tsv",
                     "diagnostics.tsv", "summary.json"):
            pa, pb = _A["dir"] / name, b / name
            self.assertTrue(pb.exists(), name)
            self.assertEqual(pa.read_text(), pb.read_text(), name)


class TestHelpers(unittest.TestCase):
    def test_unit(self):
        v = sa.unit(np.array([3.0, 4.0]))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)
        self.assertEqual(list(sa.unit(np.zeros(3))), [0.0, 0.0, 0.0])

    def test_term_re_word_boundary(self):
        r = sa.term_re("data")
        self.assertTrue(r.search("the data stream"))
        self.assertFalse(r.search("database"))     # no match inside a word
        self.assertFalse(r.search("metadata"))

    def test_seed_terms_sorted_and_complete(self):
        terms = sa.seed_terms()
        self.assertEqual(terms, sorted(terms))
        self.assertEqual(len(terms), len(set(terms)))
        for spec in sa.AXES.values():
            for t in list(spec["plus"]) + list(spec["minus"]):
                self.assertIn(t, terms)

    def test_axis_direction_and_pairs_on_synthetic_vectors(self):
        V = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
        vidx = {"a": 0, "ap": 1, "b": 2, "bp": 3}
        axis = sa.axis_direction(V, vidx, ["a", "ap"], ["b", "bp"])
        self.assertAlmostEqual(axis[0], 1.0, places=6)
        self.assertAlmostEqual(axis[1], 0.0, places=6)
        pairs = sa.pair_cosines(V, vidx, axis, ["a", "ap"], ["b", "bp"])
        for _p, _m, c in pairs:
            self.assertAlmostEqual(c, 1.0, places=6)
        # a slightly tilted pair still has positive cosine to the axis
        V2 = np.array([[0.9, 0.1], [1.0, 0.0]])
        V2 = V2 / np.linalg.norm(V2, axis=1, keepdims=True)
        c = sa.pair_cosines(V2, {0: 0, 1: 1}, np.array([1.0, 0.0]),
                             [1], [0])[0][2]
        self.assertGreater(c, 0.0)
        self.assertLess(c, 1.0)

    def test_projection_contrast_synthetic(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(scale=0.1, size=200)
        proj = np.concatenate([np.full(100, 2.0) + noise[:100],
                               np.full(100, 0.0) + noise[100:]])
        p_hit = np.zeros(200, dtype=bool)
        p_hit[:100] = True
        m_hit = np.zeros(200, dtype=bool)
        m_hit[100:] = True
        r = sa.projection_contrast(proj, p_hit, m_hit)
        self.assertAlmostEqual(r["contrast"], 2.0, delta=0.2)
        self.assertGreater(r["d"], 1.0)
        self.assertGreater(r["z"], 5.0)
        self.assertEqual(r["n_both"], 0)

    def test_projection_contrast_empty_pole(self):
        proj = np.arange(10.0)
        r = sa.projection_contrast(proj, np.ones(10, dtype=bool),
                                   np.zeros(10, dtype=bool))
        self.assertEqual(r["contrast"], "")
        self.assertEqual(r["n_minus"], 0)

    def test_random_direction_control_nullish(self):
        rng = np.random.default_rng(0)
        cents = rng.normal(size=(12, 16))
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        d = rng.normal(size=16)
        d /= np.linalg.norm(d)
        value = float((cents @ d).std())
        ctrl = sa.random_direction_control(cents, value, rng, n_iter=50)
        self.assertLess(abs(ctrl["z"]), 3.0)       # a random direction is typical

    def test_random_axis_control_beats_named_pairing(self):
        rng = np.random.default_rng(1)
        # three +e0 terms, three -e0 terms, six off-axis noise terms: the
        # named pairs must cohere better than random pole assignments.
        dim = 32
        V = rng.normal(size=(12, dim))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        V[0:3] = 0.0
        V[0:3, 0] = 1.0
        V[3:6] = 0.0
        V[3:6, 0] = -1.0
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        terms = [f"t{i}" for i in range(12)]
        vidx = {t: i for i, t in enumerate(terms)}
        plus, minus = terms[:3], terms[3:6]
        axis = sa.axis_direction(V, vidx, plus, minus)
        named = float(np.mean([c for _p, _m, c
                               in sa.pair_cosines(V, vidx, axis, plus, minus)]))
        ctrl = sa.random_axis_control(V, vidx, plus, minus, rng, n_iter=60)
        self.assertAlmostEqual(named, 1.0, places=6)
        self.assertGreater(named, ctrl["mean"])
        self.assertLess(ctrl["mean"], 0.9)
        self.assertGreater(ctrl["sd"], 0.0)


if __name__ == "__main__":
    unittest.main()
