#!/usr/bin/env python3
"""Regression suite for #42 UMAP theme map of the book.

Builds the artifact into a temp directory (full 5,614-clause build over the
real #39 vectors) and checks the invariants that matter:

  * the clause set reconciles with #39 / #33 / #17 / #27 exactly;
  * the map coordinates are finite, row-aligned with the clause index, and
    cover every clause and page;
  * the theme partition is #40's (sizes sum to the clause count; matches
    #40's persisted catalogue sizes and medoids);
  * the drift statistics are bounded and have the expected null relations
    (step/random near 1, neighbour page delta below the permutation null,
    between-page variance small, lag dip below its null);
  * the interactive HTML and the static/trajectory/drift figures exist and
    are well-formed;
  * two builds produce identical tables (determinism).

    python work/test_umap_theme_map.py
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
import umap_theme_map as um

_A = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="umap_map_test_")
    um.build(Path(tmp), quiet=True)
    _A["dir"] = Path(tmp)
    _A["coords"] = cs.read_tsv(Path(tmp) / "umap_coords.tsv")
    _A["drift"] = {r["metric"]: r["value"]
                   for r in cs.read_tsv(Path(tmp) / "drift_stats.tsv")}
    _A["lag"] = cs.read_tsv(Path(tmp) / "lag_return.tsv")
    _A["pages_tbl"] = cs.read_tsv(Path(tmp) / "page_nn_spread.tsv")
    _A["crossref"] = cs.read_tsv(Path(tmp) / "cluster_crossref.tsv")
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


def drift(metric):
    v = _A["drift"][metric]
    return None if v in ("", None) else float(v)


def lag_row(k):
    for r in _A["lag"]:
        if int(r["lag"]) == k:
            return r
    raise KeyError(k)


class TestReconciliation(unittest.TestCase):
    def test_clause_total_matches_39(self):
        self.assertEqual(_A["summary"]["n_clauses"], 5614)
        self.assertEqual(recon("clauses")["earlier"], "5,614 (#39)")

    def test_pages_match_39(self):
        self.assertEqual(_A["summary"]["n_pages"], 225)

    def test_screened_units_match_27(self):
        self.assertEqual(recon("screened units")["here"], "74,077")
        self.assertEqual(recon("screened units")["earlier"], "74,077 (#27)")

    def test_cluster_sizes_sum_to_clauses(self):
        total = sum(int(r["size"]) for r in _A["crossref"])
        self.assertEqual(total, 5614)
        self.assertEqual(int(recon("cluster sizes sum")["here"]), total)

    def test_empty_text_clauses_disclosed(self):
        self.assertEqual(_A["summary"]["n_empty_text_clauses"], 36)

    def test_view_and_encoder_pinned(self):
        self.assertEqual(_A["summary"]["view"], "units")
        self.assertEqual(_A["summary"]["encoder"], "sbert")


class TestCoordinates(unittest.TestCase):
    def test_row_count_and_alignment(self):
        all_ids = {int(r["clause_id"]) for r in cs.read_tsv(um.STORE_DIR / "clause_index.tsv")}
        rows = _A["coords"]
        self.assertEqual(len(rows), 5614)
        for r in rows:
            self.assertIn(int(r["clause_id"]), all_ids)

    def test_all_pages_covered(self):
        pages = {int(r["page"]) for r in _A["coords"]}
        self.assertEqual(len(pages), 225)
        self.assertEqual(min(pages), 1)
        self.assertEqual(max(pages), 225)

    def test_coords_finite(self):
        for col in ("umap_1", "umap_2", "tsne_1", "tsne_2"):
            vals = np.array([float(r[col]) for r in _A["coords"]])
            self.assertTrue(np.all(np.isfinite(vals)), col)

    def test_umap_extent_bounded(self):
        u1 = np.array([float(r["umap_1"]) for r in _A["coords"]])
        u2 = np.array([float(r["umap_2"]) for r in _A["coords"]])
        self.assertLess(float(np.abs(u1).max()), 100.0)
        self.assertLess(float(np.abs(u2).max()), 100.0)

    def test_cluster_ids_in_range(self):
        cids = {int(r["cluster_id"]) for r in _A["coords"]}
        self.assertEqual(cids, set(range(um.KMEANS_K)))

    def test_coords_consistent_with_store(self):
        """Pages and clause ids match the #39 store row-for-row."""
        store = cs.read_tsv(um.STORE_DIR / "clause_index.tsv")
        self.assertEqual(len(store), len(_A["coords"]))
        for a, b in zip(store, _A["coords"]):
            self.assertEqual(int(a["clause_id"]), int(b["clause_id"]))
            self.assertEqual(int(a["page"]), int(b["page"]))


class TestThemePartition(unittest.TestCase):
    def test_crossref_has_12_clusters(self):
        self.assertEqual(len(_A["crossref"]), um.KMEANS_K)
        self.assertEqual(_A["summary"]["theme_kmeans"]["k"], um.KMEANS_K)

    def test_partition_matches_40_catalogue(self):
        """Same algorithm/settings/seed as #40 -> identical sizes + medoids."""
        cat_path = um.CLUSTERING_DIR / "clusters.tsv"
        if not cat_path.exists():
            self.skipTest("#40 catalogue not present")
        cat = {int(r["cluster_id"]): r for r in cs.read_tsv(cat_path)}
        for r in _A["crossref"]:
            cid = int(r["cluster_id"])
            self.assertTrue(r["in_40_catalogue"] == "True")
            self.assertEqual(int(cat[cid]["size"]), int(r["size"]),
                             f"cluster {cid} size differs from #40")
            self.assertEqual(r["size_matches_40"], "True", cid)
            self.assertEqual(r["medoid_in_cluster"], "True", cid)


class TestDriftStats(unittest.TestCase):
    def test_metrics_present(self):
        for m in ("mean_step_cos_dist", "mean_random_cos_dist",
                  "step_over_random", "mean_nn_page_delta",
                  "random_pair_page_delta", "perm_null_mean",
                  "z_vs_perm_null", "between_page_frac",
                  "half1_half2_cos", "page_vs_umap_x_r2"):
            self.assertIn(m, _A["drift"], m)

    def test_step_ratio_near_one(self):
        """Stationary random walk: consecutive ~ random distance."""
        ratio = drift("step_over_random")
        self.assertGreater(ratio, 0.9)
        self.assertLess(ratio, 1.1)

    def test_nn_page_delta_below_null(self):
        """Local page coherence: neighbours closer in page than null."""
        self.assertLess(drift("mean_nn_page_delta"),
                        drift("perm_null_mean"))
        self.assertLess(drift("mean_nn_page_delta"),
                        drift("random_pair_page_delta"))
        self.assertLess(drift("z_vs_perm_null"), -3.0)

    def test_between_page_variance_small(self):
        frac = drift("between_page_frac")
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 0.2)

    def test_halves_identical(self):
        cos = drift("half1_half2_cos")
        self.assertGreater(cos, 0.9)
        self.assertLessEqual(cos, 1.0)

    def test_page_axis_correlation_tiny(self):
        rx2 = drift("page_vs_umap_x_r2")
        ry2 = drift("page_vs_umap_y_r2")
        self.assertLess(rx2, 0.05)
        self.assertLess(ry2, 0.05)

    def test_all_values_finite(self):
        for m, v in _A["drift"].items():
            if v in ("", None):
                continue
            self.assertTrue(np.isfinite(float(v)), m)


class TestLagReturn(unittest.TestCase):
    def test_grid_complete(self):
        lags = [int(r["lag"]) for r in _A["lag"]]
        self.assertEqual(lags, sorted(um.LAG_GRID))

    def test_dip_below_null(self):
        """Short-range recurrence: mid-grid lag below the shuffle null."""
        for k in (5, 7, 10):
            r = lag_row(k)
            self.assertLess(float(r["mean_cos_dist"]),
                            float(r["null_mean"]), k)
            self.assertLess(float(r["z_vs_null"]), -2.0, k)

    def test_min_lag_is_small(self):
        self.assertLessEqual(_A["summary"]["lag_return"]["min_lag"], 10)

    def test_lag800_at_null(self):
        r = lag_row(800)
        z = abs(float(r["z_vs_null"]))
        self.assertLess(z, 2.0)


class TestPerPageTable(unittest.TestCase):
    def test_all_pages_present(self):
        self.assertEqual(len(_A["pages_tbl"]), 225)

    def test_clause_counts_match_coords(self):
        counts = {}
        for r in _A["coords"]:
            counts[int(r["page"])] = counts.get(int(r["page"]), 0) + 1
        for r in _A["pages_tbl"]:
            self.assertEqual(int(r["n_clauses"]), counts[int(r["page"])])

    def test_within_page_cos_bounded(self):
        vals = [float(r["within_page_nn_cos"]) for r in _A["pages_tbl"]
                if r["within_page_nn_cos"] != ""]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))

    def test_nn10_delta_bounded(self):
        vals = [float(r["nn10_page_delta"]) for r in _A["pages_tbl"]]
        self.assertTrue(all(0.0 <= v <= 224.0 for v in vals))

    def test_summary_texture_extreme_filter(self):
        """The summary's page-texture extremes filter tiny pages."""
        self.assertGreaterEqual(
            _A["summary"]["page_cos_texture"]["min_clauses"], 10)
        self.assertIn("max_page", _A["summary"]["page_cos_texture"])


class TestArtifacts(unittest.TestCase):
    def test_interactive_html(self):
        path = _A["dir"] / "umap_page_map.html"
        self.assertTrue(path.exists())
        h = path.read_text(encoding="utf-8")
        self.assertIn("plotly", h)
        # plotly write_html JSON-escapes "/" as "￿" in button labels,
        # so match on the un-escaped fragments only.
        for frag in ("page (UMAP)", "page (t-SNE)", "cluster (UMAP)",
                     "cluster (t-SNE)", "trajectory"):
            self.assertIn(frag, h)
        self.assertIn('"visible":[true', h)
        self.assertGreater(h.count("scattergl"), 4)  # 5 map traces + template

    def test_static_png_wellformed(self):
        path = _A["dir"] / "umap_map_static.png"
        self.assertTrue(path.exists())
        magic = path.open("rb").read(8)
        self.assertEqual(magic, b"\x89PNG\r\n\x1a\n")
        self.assertGreater(path.stat().st_size, 100_000)

    def test_trajectory_png_wellformed(self):
        path = _A["dir"] / "trajectory_map.png"
        self.assertTrue(path.exists())
        self.assertEqual(path.open("rb").read(8), b"\x89PNG\r\n\x1a\n")

    def test_drift_curve_png_wellformed(self):
        path = _A["dir"] / "drift_curve.png"
        self.assertTrue(path.exists())
        self.assertEqual(path.open("rb").read(8), b"\x89PNG\r\n\x1a\n")

    def test_report_exists_with_sections(self):
        path = _A["dir"] / "umap_theme_map_report.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        for section in ("# UMAP theme map of the book (#42)",
                        "## 1. Method",
                        "## 3. Reading the page-coloured map",
                        "## 4. Where the loop is",
                        "## 8. Negative results and limitations"):
            self.assertIn(section, text)

    def test_helper_tables_exist(self):
        for name in ("lag_return.tsv", "page_nn_spread.tsv",
                     "cluster_crossref.tsv", "drift_stats.tsv",
                     "reconciliation.tsv", "diagnostics.tsv",
                     "summary.json"):
            self.assertTrue((_A["dir"] / name).exists(), name)


class TestDeterminism(unittest.TestCase):
    def test_two_builds_identical(self):
        a = _A["dir"]
        b = Path(tempfile.mkdtemp(prefix="umap_map_det_"))
        um.build(b, quiet=True)
        for name in ("umap_coords.tsv", "drift_stats.tsv", "lag_return.tsv",
                     "page_nn_spread.tsv", "cluster_crossref.tsv",
                     "reconciliation.tsv", "diagnostics.tsv", "summary.json"):
            pa = a / name
            pb = b / name
            self.assertTrue(pb.exists(), name)
            self.assertEqual(pa.read_text(), pb.read_text(), name)


class TestHelpers(unittest.TestCase):
    def test_lag_return_grid_constant(self):
        self.assertIn(1, um.LAG_GRID)
        self.assertIn(800, um.LAG_GRID)

    def test_projection_params_disclosed(self):
        self.assertEqual(um.UMAP_NEIGHBORS, 30)
        self.assertEqual(um.UMAP_MIN_DIST, 0.1)
        self.assertEqual(um.TSNE_PERPLEXITY, 30)
        self.assertEqual(um.KMEANS_K, 12)
        s = _A["summary"]["projections"]["umap"]
        self.assertEqual(s["n_jobs"], 1)
        self.assertEqual(s["seed"], 0)


if __name__ == "__main__":
    unittest.main()
