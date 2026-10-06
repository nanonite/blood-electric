#!/usr/bin/env python3
"""Regression suite for #28 static embeddings and analogies.

Runs the builder once in ``--quick`` mode (tiny book-only models) into a temp
directory, then checks the invariants that matter: the #19 markup screen and
the asserted screening join, the utterance sequence construction, the GloVe
co-occurrence / duplicate-safe scatter, the unified cosine search, the analogy
ranking, and the artifact schema.  If the full artifacts are present under
``analysis/embeddings/`` they are validated too.

    python work/test_embedding_models.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import embedding_models as em

_ART = {}
FULL = em.OUT


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="emb_test_")
    em.main(["--out", tmp, "--quick", "--quiet"])
    _ART["_dir"] = tmp
    for name in ("diagnostics", "reconciliation", "model_summary",
                 "vocabulary", "analogies", "analogy_summary",
                 "analogies_english", "neighbours_selected",
                 "neighbours_book"):
        _ART[name] = em.read_tsv(Path(tmp) / f"{name}.tsv")
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["report"] = (Path(tmp) / "embedding_report.md").read_text(
        encoding="utf-8")


def diag():
    return {r["diag"]: r["value"] for r in _ART["diagnostics"]}


# ---------------------------------------------------------------------------
class TestScreenAndJoin(unittest.TestCase):
    def test_screened_counts(self):
        d = diag()
        self.assertEqual(d["units"], "74318")            # #19/#24
        self.assertEqual(d["types_folded"], "5502")      # #24
        self.assertEqual(d["markup_removed"], "347")     # #19
        self.assertEqual(d["sequences"], "2345")

    def test_join_to_19_is_asserted(self):
        # load_book raises SystemExit on any mismatch; this simply exercises it
        kept, seqs, cats, meta, pages = em.load_book()
        self.assertEqual(len(kept), 74318)
        self.assertEqual(len(meta), 5502)
        self.assertEqual(pages, 225)

    def test_sequences_cover_every_unit(self):
        _kept, seqs, _c, _m, _p = em.load_book()
        self.assertEqual(sum(len(s) for s in seqs), 74318)

    def test_markup_types_absent(self):
        lows = {r["lower"] for r in _ART["vocabulary"]}
        for tag in ("th", "p", "pre", "br", "li", "hr", "td", "tr"):
            self.assertNotIn(tag, lows, tag)

    def test_reconciliation(self):
        rec = {r["quantity"]: r for r in _ART["reconciliation"]}
        self.assertEqual(rec["screened occurrences"]["here"], "74318")
        self.assertEqual(rec["case-folded types"]["here"], "5502")
        self.assertEqual(rec["markup occurrences removed"]["here"], "347")


# ---------------------------------------------------------------------------
class TestGlovePrimitives(unittest.TestCase):
    def test_scatter_sum_matches_add_at(self):
        rng = np.random.default_rng(0)
        idx = rng.integers(0, 7, size=500)
        vals = rng.normal(size=(500, 4))
        dest = np.zeros((7, 4))
        np.add.at(dest, idx, vals)
        uniq, sums = em._scatter_sum(idx, vals, 7)
        got = np.zeros((7, 4))
        got[uniq] = sums
        self.assertTrue(np.allclose(dest, got))

    def test_cooccurrence_symmetric(self):
        seqs = [["a", "b", "a"]]
        vi = {"a": 0, "b": 1}
        pairs = em.build_cooccurrence(seqs, vi, window=5)
        # a-b at distance 1 from either direction -> 1.0 + 1.0; a-a at
        # distance 2 -> 0.5
        self.assertAlmostEqual(pairs[(0, 1)], 2.0)
        self.assertAlmostEqual(pairs[(0, 0)], 0.5)
        self.assertEqual(set(pairs), {(0, 0), (0, 1)})

    def test_train_glove_deterministic_and_shaped(self):
        seqs = [["a", "b", "c"], ["b", "a", "c"]] * 5
        k1, m1 = em.train_glove(seqs, dim=8, window=3, min_count=1, epochs=2,
                                seed=1)
        k2, m2 = em.train_glove(seqs, dim=8, window=3, min_count=1, epochs=2,
                                seed=1)
        self.assertEqual(k1, k2)
        self.assertEqual(m1.shape, (3, 8))
        self.assertTrue(np.allclose(m1, m2))

    def test_glove_respects_min_count(self):
        seqs = [["a", "b", "a", "b"], ["c"]]
        keys, _ = em.train_glove(seqs, dim=4, window=3, min_count=2, epochs=1,
                                 seed=1)
        self.assertIn("a", keys)
        self.assertIn("b", keys)
        self.assertNotIn("c", keys)


# ---------------------------------------------------------------------------
class TestVecModel(unittest.TestCase):
    def _model(self):
        keys = ["king", "queen", "man", "woman", "other"]
        mat = np.array([
            [3.0, 3.0],   # king
            [2.0, 4.0],   # queen  (= king - man + woman)
            [1.0, 0.0],   # man
            [0.0, 1.0],   # woman
            [0.0, 3.0],   # other
        ])
        return em.VecModel("toy", "toy", "toy", keys, mat)

    def test_most_similar_orders_and_excludes(self):
        m = self._model()
        res = m.most_similar(["king"], topn=4)
        self.assertNotIn("king", [k for k, _ in res])
        sims = [s for _, s in res]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertEqual(res[0][0], "queen")

    def test_analogy_rank_hit(self):
        m = self._model()
        rank, _res = em.analogy_rank(m, "man", "king", "woman", "queen")
        self.assertEqual(rank, 1)

    def test_analogy_oov_returns_none(self):
        m = self._model()
        rank, _res = em.analogy_rank(m, "zzz", "king", "woman", "queen")
        self.assertIsNone(rank)

    def test_subword_model_has_everything(self):
        m = em.VecModel("ft", "book", "fastText", ["a"], np.ones((1, 3)),
                        subwords=True, vector_fn=lambda t: np.ones(3))
        self.assertTrue(m.has("never-seen-unit"))


# ---------------------------------------------------------------------------
class TestMining(unittest.TestCase):
    def test_derivation_probes_deterministic(self):
        a = em.mine_derivation_probes(max_probes=10)
        b = em.mine_derivation_probes(max_probes=10)
        self.assertEqual(a, b)
        self.assertTrue(a)
        self.assertLessEqual(len(a), 10)
        for p in a:
            self.assertNotEqual(p["b"], p["expected"])   # non-trivial head

    def test_collocation_probes_deterministic(self):
        a = em.mine_collocation_probes(max_probes=10, min_f=40)
        b = em.mine_collocation_probes(max_probes=10, min_f=40)
        self.assertEqual(a, b)
        self.assertTrue(a)
        self.assertLessEqual(len(a), 10)


# ---------------------------------------------------------------------------
class TestArtifacts(unittest.TestCase):
    def test_model_summary(self):
        rows = _ART["model_summary"]
        self.assertEqual({r["model"] for r in rows},
                         {"w2v_book", "ft_book", "glove_book"})
        self.assertEqual({r["subwords"] for r in rows}, {"True", "False"})

    def test_coverage_fasttext_is_complete(self):
        d = diag()
        self.assertEqual(d["coverage_ft_book"], "5502")
        self.assertLessEqual(int(d["coverage_w2v_book"]), 5502)
        self.assertLessEqual(int(d["coverage_glove_book"]), 5502)

    def test_vocabulary_columns(self):
        row = _ART["vocabulary"][0]
        for col in ("lower", "n", "category", "in_w2v_book", "in_ft_book",
                    "in_glove_book"):
            self.assertIn(col, row)

    def test_analogy_families_present(self):
        fams = {r["family"] for r in _ART["analogy_summary"]}
        self.assertEqual(fams, {"naming", "derivation", "collocation",
                                "mined_derivation", "mined_collocation"})
        # every summary row has a model and valid rates
        for r in _ART["analogy_summary"]:
            self.assertGreaterEqual(float(r["hit10"]), 0.0)
            self.assertLessEqual(float(r["hit10"]), 1.0)

    def test_analogy_rows_unique_and_ranked(self):
        seen = set()
        for r in _ART["analogies"]:
            key = (r["model"], r["family"], r["probe"])
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertGreaterEqual(int(r["rank"]), 0)

    def test_selected_neighbours_cover_models(self):
        models = {r["model"] for r in _ART["neighbours_selected"]}
        self.assertTrue(models.issubset(
            {"w2v_book", "ft_book", "glove_book",
             "w2v_merged", "ft_merged", "glove_merged"}))
        self.assertIn("ft_book", models)

    def test_report_sections(self):
        for needle in ("# Static embeddings and analogies (#28)",
                       "## 4. Nearest neighbours",
                       "## 5. Analogies",
                       "## 6. Interpretation and limits"):
            self.assertIn(needle, _ART["report"])

    def test_tsv_quote_safe(self):
        for name in ("vocabulary", "analogies", "neighbours_selected"):
            path = Path(_ART["_dir"]) / f"{name}.tsv"
            with path.open(encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                for line in fh:
                    if not line.strip():
                        continue
                    self.assertEqual(len(line.rstrip("\n").split("\t")),
                                     len(header), name)


# ---------------------------------------------------------------------------
class TestFullArtifacts(unittest.TestCase):
    """Validate the committed full build when it exists (skipped otherwise)."""

    @unittest.skipUnless((FULL / "reconciliation.tsv").exists(),
                         "full artifacts not built")
    def test_full_reconciliation(self):
        rec = {r["quantity"]: r for r in em.read_tsv(
            FULL / "reconciliation.tsv")}
        self.assertEqual(rec["screened occurrences"]["here"], "74318")
        self.assertEqual(rec["case-folded types"]["here"], "5502")

    @unittest.skipUnless((FULL / "diagnostics.tsv").exists(),
                         "full artifacts not built")
    def test_full_models_and_coverage(self):
        d = {r["diag"]: r["value"] for r in em.read_tsv(
            FULL / "diagnostics.tsv")}
        self.assertEqual(d["coverage_ft_book"], "5502")
        self.assertEqual(d["coverage_ft_merged"], "5502")
        self.assertEqual(d["models"], "6")

    @unittest.skipUnless((FULL / "analogy_summary.tsv").exists(),
                         "full artifacts not built")
    def test_full_english_analogies(self):
        rows = em.read_tsv(FULL / "analogies_english.tsv")
        cols = [c for c in rows[0] if c.endswith("_merged")]
        self.assertTrue(cols)
        # the merged fastText model should get several English analogies
        hits = sum(1 for r in rows if r["ft_merged"] == "1")
        self.assertGreaterEqual(hits, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
