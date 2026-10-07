#!/usr/bin/env python3
"""Regression suite for #39 clause / sentence embeddings.

Builds the artifact into a temp directory (200-clause deterministic sample,
primary SBERT only) and checks the invariants that matter:

  * the clause/unit join reconciles with #33 / #17 / #27 exactly;
  * every persisted vector is L2-normalised and row-aligned with the index;
  * the loader returns vectors / texts / neighbours by ``clause_id`` and page,
    and a clause's own vector is its top-1 neighbour;
  * the text views reconstruct the markup-screened #17 unit sequence exactly;
  * the book-vs-English geometry reproduces the repetition finding;
  * two builds produce identical vectors (determinism).

    python work/test_clause_embeddings.py
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

import clause_embeddings as ce  # noqa: E402
import clause_segmentation as cs  # noqa: E402

_A = {}
SAMPLE = 200


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="clause_emb_test_")
    ce.build(Path(tmp), encoder_key=ce.PRIMARY, sample=SAMPLE, secondary="",
             english=True, quiet=True)
    _A["dir"] = Path(tmp)
    _A["store"] = ce.ClauseEmbeddingStore(tmp)
    _A["index"] = cs.read_tsv(Path(tmp) / "clause_index.tsv")
    _A["stats"] = cs.read_tsv(Path(tmp) / "embedding_stats.tsv")
    _A["agreement"] = cs.read_tsv(Path(tmp) / "view_agreement.tsv")
    _A["recon"] = cs.read_tsv(Path(tmp) / "reconciliation.tsv")
    _A["english"] = cs.read_tsv(Path(tmp) / "english_baseline.tsv")
    _A["diag"] = {r["metric"]: r["value"]
                  for r in cs.read_tsv(Path(tmp) / "diagnostics.tsv")}
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _A["summary"] = json.load(fh)


def stat(set_name):
    for r in _A["stats"]:
        if r["set"] == set_name:
            return r
    raise KeyError(set_name)


def recon(quantity):
    for r in _A["recon"]:
        if r["quantity"] == quantity:
            return r
    raise KeyError(quantity)


class TestReconciliation(unittest.TestCase):
    def test_clause_total_matches_33(self):
        self.assertEqual(_A["summary"]["n_clauses_total"], 5614)

    def test_embedded_count(self):
        self.assertEqual(_A["summary"]["n_clauses_embedded"], SAMPLE)
        self.assertEqual(len(_A["index"]), SAMPLE)

    def test_screened_units_match_27(self):
        self.assertEqual(_A["summary"]["screened_units"], 74077)
        self.assertEqual(recon("screened units")["earlier"], "74,077 (#27)")

    def test_raw_units_match_17(self):
        self.assertEqual(_A["summary"]["raw_units"], 74665)
        self.assertEqual(_A["summary"]["markup_units"], 588)

    def test_pages_bounded(self):
        self.assertTrue(1 <= _A["summary"]["n_pages"] <= 225)
        self.assertEqual({int(r["page"]) for r in _A["index"]},
                         set(int(p) for p in _A["store"].pages))


class TestVectors(unittest.TestCase):
    def test_shapes(self):
        st = _A["store"]
        self.assertEqual(st.matrix("units").shape, (SAMPLE, 384))
        self.assertEqual(st.matrix("raw").shape, (SAMPLE, 384))
        self.assertEqual(st.clause_ids.shape, (SAMPLE,))

    def test_normalised(self):
        for view in ce.VIEWS:
            norms = np.linalg.norm(_A["store"].matrix(view), axis=1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-4),
                            f"{view} not L2-normalised")

    def test_ids_are_clause_ids_from_33(self):
        all_ids = {int(r["clause_id"]) for r in
                   cs.read_tsv(ce.CLAUSES)}
        self.assertTrue(set(int(c) for c in _A["store"].clause_ids) <= all_ids)

    def test_index_row_alignment(self):
        st = _A["store"]
        for i, cid in enumerate(st.clause_ids):
            self.assertEqual(int(_A["index"][i]["clause_id"]), int(cid))
            self.assertEqual(st.vector(int(cid), "units")[0],
                             st.matrix("units")[i][0])


class TestLoader(unittest.TestCase):
    def test_self_excluded_and_high_similarity(self):
        st = _A["store"]
        for cid in (int(st.clause_ids[0]), int(st.clause_ids[SAMPLE // 2]),
                    int(st.clause_ids[-1])):
            hits = st.nearest(cid, k=5)
            self.assertNotIn(cid, [c for c, _ in hits])
            self.assertGreater(hits[0][1], 0.4)

    def test_search_recovers_self(self):
        st = _A["store"]
        cid = int(st.clause_ids[3])
        hits = st.search(st.vector(cid, "units"), view="units", k=3)
        self.assertEqual(hits[0][0], cid)

    def test_nearest_include_self_flag(self):
        st = _A["store"]
        cid = int(st.clause_ids[0])
        hits = ce.nearest(st.matrix("units"), 0, k=3, exclude_self=False)
        self.assertEqual(int(st.clause_ids[hits[0][0]]), cid)

    def test_text_round_trip(self):
        st = _A["store"]
        cid = int(st.clause_ids[5])
        self.assertEqual(st.text(cid, "units"),
                         st.meta(cid)["text_units"])
        self.assertEqual(st.text(cid, "raw"), st.meta(cid)["text_raw"])

    def test_page_matrix(self):
        st = _A["store"]
        page = int(st.pages[0])
        rows = st.page_rows(page)
        self.assertGreater(len(rows), 0)
        self.assertEqual(st.page_matrix(page, "units").shape,
                         (len(rows), 384))

    def test_same_page_neighbours_stay_on_page(self):
        st = _A["store"]
        # find a page with >= 2 clauses in the sample
        counts = {}
        for p in st.pages:
            counts[int(p)] = counts.get(int(p), 0) + 1
        page = max(counts, key=lambda k: counts[k])
        if counts[page] < 2:
            self.skipTest("no page with 2+ sampled clauses")
        cid = int(st.clause_ids[st.page_rows(page)[0]])
        hits = st.nearest(cid, k=1, same_page_only=True)
        self.assertTrue(hits)
        for c, _ in hits:
            self.assertEqual(int(st.meta(c)["page"]), page)

    def test_unknown_view_raises(self):
        with self.assertRaises(KeyError):
            _A["store"].matrix("nope")


class TestTextViews(unittest.TestCase):
    def test_units_reconstructs_screened_unit_sequence(self):
        """Independent recompute of ``text_units`` for sampled clauses."""
        bodies, page_units, unit_meta, is_markup = ce.load_context()
        for r in _A["index"][:25]:
            page, s, e = int(r["page"]), int(r["start"]), int(r["end"])
            toks = [unit_meta[u][3]
                    for (us, _ue, u) in page_units.get(page, ())
                    if s <= us < e and not is_markup(u)]
            self.assertEqual(r["text_units"], " ".join(toks))

    def test_no_markup_tag_names_in_units_text(self):
        bad = {"<a", "</a>", "<th", "</th>", "mailto:", "colspan"}
        for r in _A["index"]:
            low = r["text_units"].lower()
            for b in bad:
                self.assertNotIn(b, low, f"markup residue in clause "
                                         f"{r['clause_id']}")


class TestGeometryAndRobustness(unittest.TestCase):
    def test_book_more_self_similar_than_english(self):
        b = float(stat("book_units")["mean_nn"])
        e = float(stat("english")["mean_nn"])
        self.assertGreater(b, e)

    def test_english_control_count(self):
        self.assertEqual(len(_A["english"]), SAMPLE)

    def test_view_overlap_present_and_high(self):
        rows = [r for r in _A["agreement"]
                if r["view_a"] == "units" and r["view_b"] == "raw"]
        self.assertEqual(len(rows), 1)
        self.assertGreater(float(rows[0]["mean_overlap"]), 0.5)

    def test_no_secondary_file_when_disabled(self):
        self.assertEqual(_A["summary"]["secondary"], None)
        self.assertFalse(
            (_A["dir"] / "clause_embeddings_bge.npz").exists())

    def test_centroid_identity(self):
        """mean centroid cosine == ||centroid|| for unit vectors."""
        for name in ("book_units", "book_raw", "english"):
            s = stat(name)
            self.assertAlmostEqual(float(s["mean_centroid_cos"]),
                                   float(s["centroid_norm"]), places=4)


class TestDeterminism(unittest.TestCase):
    def test_two_builds_identical(self):
        a = tempfile.mkdtemp(prefix="ce_det_a_")
        b = tempfile.mkdtemp(prefix="ce_det_b_")
        for d in (a, b):
            ce.build(Path(d), sample=24, secondary="", english=False,
                     quiet=True)
        za = np.load(Path(a) / ce.model_filename(ce.PRIMARY))
        zb = np.load(Path(b) / ce.model_filename(ce.PRIMARY))
        for view in ce.VIEWS:
            self.assertTrue(np.allclose(za[f"emb_{view}"], zb[f"emb_{view}"],
                                        atol=1e-6))
        self.assertTrue(np.array_equal(za["clause_ids"], zb["clause_ids"]))


class TestHelpers(unittest.TestCase):
    def test_sample_indices(self):
        idx = ce.sample_indices(1000, 10)
        self.assertEqual(len(idx), 10)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 999)
        self.assertEqual(idx, sorted(idx))
        self.assertEqual(ce.sample_indices(5, 10), list(range(5)))

    def test_registry(self):
        self.assertIn(ce.PRIMARY, ce.MODELS)
        self.assertEqual(ce.MODELS["sbert"]["pooling"], "mean")
        self.assertEqual(ce.MODELS["e5"]["prefix"], "passage: ")
        self.assertEqual(ce.MODELS["bge"]["pooling"], "cls")

    def test_model_filename(self):
        self.assertEqual(ce.model_filename(ce.PRIMARY), "clause_embeddings.npz")
        self.assertEqual(ce.model_filename("bge"), "clause_embeddings_bge.npz")

    def test_diagnostics(self):
        self.assertEqual(int(_A["diag"]["n_clauses_total"]), 5614)
        self.assertEqual(int(_A["diag"]["screened_units"]), 74077)
        self.assertEqual(int(_A["diag"]["dim"]), 384)


if __name__ == "__main__":
    unittest.main()
