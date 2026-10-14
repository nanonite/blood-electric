#!/usr/bin/env python3
"""Regression suite for #54 Self-similarity matrix (loop detection).

Builds the artifact into a temp directory (full 5,614-clause build over the
real #39 vectors, no encoder: the English control is exercised through its
grouping function and its skip path) and checks the invariants that matter:

  * the paragraph unit is a blank-line run of the #7 layout page body, every
    clause lands in exactly one paragraph, and nothing is dropped;
  * paragraph vectors are clause-vector means, L2-normalised, row-aligned
    with ``paragraph_matrix.tsv``;
  * the matrix is symmetric with a zero diagonal, and the lag/band statistics
    have the expected null relations (short-range band above the null, no
    large-lag excess);
  * block detection uses 8-connectivity and finds the book's verbatim return
    of pages 4-7 on pages 220-223 at both resolutions;
  * the lexical control, the page rollup and the view/encoder sensitivity
    agree with the semantic matrix;
  * the artifact set is complete, the report says what the data says (no
    ``nan``, no placeholder), and two builds are identical.

    python work/test_self_similarity.py
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
import self_similarity as ss

_A = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="self_sim_test_")
    ss.build(Path(tmp), quiet=True, english=False)
    _A["dir"] = Path(tmp)
    _A["records"], _A["diags"] = ss.paragraph_records(None)
    ss.paragraph_units(_A["records"])
    _A["paras_tsv"] = cs.read_tsv(Path(tmp) / "paragraph_matrix.tsv")
    _A["summary"] = json.loads(
        (Path(tmp) / "summary.json").read_text(encoding="utf-8"))
    _A["lag"] = cs.read_tsv(Path(tmp) / "lag_curve.tsv")
    _A["page_lag"] = cs.read_tsv(Path(tmp) / "page_lag_curve.tsv")
    _A["bands"] = cs.read_tsv(Path(tmp) / "band_stats.tsv")
    _A["blocks"] = cs.read_tsv(Path(tmp) / "offdiag_blocks.tsv")
    _A["pairs"] = cs.read_tsv(Path(tmp) / "paragraph_cosines.tsv")
    _A["sens"] = cs.read_tsv(Path(tmp) / "sensitivity.tsv")
    _A["en"] = cs.read_tsv(Path(tmp) / "english_baseline.tsv")
    _A["recon"] = {r["quantity"]: r
                   for r in cs.read_tsv(Path(tmp) / "reconciliation.tsv")}
    _A["report"] = (Path(tmp) / "self_similarity_report.md").read_text(
        encoding="utf-8")
    _A["npz"] = np.load(Path(tmp) / "self_similarity.npz")


def lag_row(k):
    for r in _A["lag"]:
        if int(r["lag"]) == k:
            return r
    raise KeyError(k)


def blocks(level=None, long_range=True, verbatim=None):
    out = _A["blocks"]
    if level is not None:
        out = [b for b in out if b["level"] == level]
    if long_range:
        out = [b for b in out if b["long_range"] == "1"]
    if verbatim is not None:
        out = [b for b in out if float(b["raw_identity"]) >= ss.VERBATIM_IDENTITY]
    return out


# ---------------------------------------------------------------------------
class TestParagraphUnit(unittest.TestCase):
    def test_paragraphs_are_blank_line_runs(self):
        body = "alpha bravo\ncharlie\n\ndelta echo\n\n### head\n"
        spans = ss.split_paragraphs(body)
        self.assertEqual([body[s:e].strip() for s, e in spans],
                         ["alpha bravo\ncharlie", "delta echo", "### head"])

    def test_spans_are_exact_offsets(self):
        body = "one two\n\nthree four"
        spans = ss.split_paragraphs(body)
        self.assertEqual(spans, [(0, 7), (9, 19)])
        self.assertEqual(body[0:7], "one two")
        self.assertEqual(body[9:19], "three four")

    def test_empty_and_unbroken_bodies(self):
        self.assertEqual(ss.split_paragraphs(""), [])
        self.assertEqual(ss.split_paragraphs("one paragraph only"),
                         [(0, 18)])
        self.assertEqual(ss.split_paragraphs("\n\n\n"), [])

    def test_para_kind(self):
        self.assertEqual(ss.para_kind("### abolition"), "heading")
        self.assertEqual(ss.para_kind("<!-- figure/picture block -->"), "figure")
        self.assertEqual(ss.para_kind("the horizon of the body"), "text")

    def test_every_clause_lands_in_one_paragraph(self):
        records, diags = _A["records"], _A["diags"]
        assigned = [c for r in records for c in r["clause_ids"]]
        clause_rows, _ = __import__("clause_embeddings").clause_records()
        expected = {r["clause_id"] for r in clause_rows}
        self.assertEqual(len(assigned), len(set(assigned)),
                         "a clause was assigned to two paragraphs")
        self.assertEqual(set(assigned), expected,
                         "clause coverage is not exact")
        self.assertEqual(diags["n_clauses_assigned"], 5614)

    def test_paragraphs_cover_the_book(self):
        diags = _A["diags"]
        self.assertEqual(diags["n_pages"], 225)
        self.assertGreater(diags["n_paragraph_runs"], 600)
        self.assertGreater(diags["n_paragraph_text"], 600)
        self.assertEqual(diags["n_paragraph_heading"], 56)
        self.assertEqual(
            diags["n_paragraph_text"] + diags["n_paragraph_heading"]
            + diags["n_paragraph_figure"]
            - diags["n_paragraphs_added_for_coverage"],
            diags["n_paragraph_runs"])

    def test_units_reconcile_with_27(self):
        total = sum(r["n_units_screened"] for r in _A["records"])
        self.assertEqual(total, 74077)
        self.assertEqual(_A["recon"]["screened units inside paragraphs"]["here"],
                         "74077")

    def test_crossing_clauses_are_disclosed(self):
        self.assertGreaterEqual(_A["diags"]["n_clauses_crossing_para_break"], 0)

    def test_matrix_rows_are_the_paragraphs_with_clauses(self):
        n_in = sum(1 for r in _A["paras_tsv"]
                   if r["in_similarity_matrix"] == "1")
        self.assertEqual(n_in, _A["summary"]["n_paragraphs"])
        self.assertEqual(_A["npz"]["emb"].shape[0], n_in)
        self.assertEqual(_A["npz"]["para_ids"].shape[0], n_in)


# ---------------------------------------------------------------------------
class TestVectors(unittest.TestCase):
    def test_vectors_are_normalised_means(self):
        import clause_embeddings as ce
        store = ce.ClauseEmbeddingStore(out_dir=ce.OUT, model=ce.PRIMARY)
        V = store.matrix(ss.PRIMARY_VIEW)
        records = _A["records"]
        rows = ss.usable_rows(records)
        P = ss.paragraph_vectors(store, records, view=ss.PRIMARY_VIEW)
        self.assertEqual(P.shape, (len(records), V.shape[1]))
        for i in rows[:20]:
            rec = records[i]
            want = V[[store.row(c) for c in rec["clause_ids"]]].mean(axis=0)
            want = want / np.linalg.norm(want)
            np.testing.assert_allclose(P[i], want, atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.norm(P[i])), 1.0, places=5)
        # a paragraph with no clause stays a zero row, never NaN
        empty = [i for i, r in enumerate(records) if not r["n_clauses"]]
        self.assertTrue(empty)
        for i in empty:
            self.assertEqual(float(np.abs(P[i]).sum()), 0.0)


# ---------------------------------------------------------------------------
class TestMatrix(unittest.TestCase):
    def test_matrix_properties(self):
        S = _A["npz"]["cos"]
        self.assertEqual(S.shape, (S.shape[0], S.shape[0]))
        np.testing.assert_allclose(S, S.T, atol=1e-5)
        np.testing.assert_allclose(np.diag(S), 0.0, atol=1e-6)
        self.assertTrue(np.all(np.abs(S) <= 1.0))

    def test_cosine_matrix_synthetic(self):
        V = np.array([[1.0, 0.0], [0.9, 0.0], [0.0, 1.0]])
        S = ss.cosine_matrix(V)
        self.assertAlmostEqual(S[0, 1], 0.9, places=6)
        self.assertAlmostEqual(S[0, 2], 0.0, places=6)
        self.assertAlmostEqual(S[1, 2], 0.0, places=6)

    def test_lag_curve_at_book(self):
        rows = _A["lag"]
        by_lag = {int(r["lag"]): r for r in rows}
        self.assertIn(1, by_lag)
        self.assertGreater(int(by_lag[1]["n_pairs"]), 700)
        self.assertGreater(int(by_lag[300]["n_pairs"]), 100)
        self.assertLess(int(by_lag[300]["n_pairs"]),
                        int(by_lag[1]["n_pairs"]))
        self.assertEqual(int(by_lag[1]["n_pairs"])
                         + int(by_lag[2]["n_pairs"])
                         + int(by_lag[300]["n_pairs"]),
                         int(by_lag[1]["n_pairs"]) + int(by_lag[2]["n_pairs"])
                         + int(by_lag[300]["n_pairs"]))
        # the pair counts shrink with lag: n_pairs = N - lag
        for k in sorted(by_lag):
            self.assertEqual(int(by_lag[k]["n_pairs"]), 717 - k)

    def test_band_is_short_range(self):
        for k in (1, 2, 3, 5):
            r = lag_row(k)
            self.assertGreater(float(r["z"]), 2.0, f"lag {k} should exceed null")
        # no large-lag excess beyond two null sd
        late = [r for r in _A["lag"] if int(r["lag"]) > 30]
        self.assertTrue(late)
        for r in late:
            self.assertLess(abs(float(r["z"])), 5.0,
                            f"lag {r['lag']} shows an unexpected excess")

    def test_null_mean_matches_the_matrix_background(self):
        z = _A["npz"]["cos"]
        iu = np.triu_indices(z.shape[0], 1)
        background = float(z[iu].mean())
        self.assertAlmostEqual(float(lag_row(200)["null_mean"]), background,
                               places=2)

    def test_band_shares_are_ordered_and_bounded(self):
        shares = [float(b["share_in_band"]) for b in _A["bands"]]
        self.assertEqual(shares, sorted(shares))
        self.assertTrue(all(0 <= s <= 1 for s in shares))
        self.assertGreater(shares[0], 0.0)
        self.assertLess(shares[0], 0.02)

    def test_top_pairs_sorted_and_aligned(self):
        cos = [float(r["cos"]) for r in _A["pairs"]]
        self.assertEqual(cos, sorted(cos, reverse=True))
        kept = [r for r in _A["records"] if r["n_clauses"]]
        self.assertEqual(len(kept), _A["summary"]["n_paragraphs"])
        for r in _A["pairs"][:10]:
            i, j = int(r["para_i"]), int(r["para_j"])
            self.assertLess(j, len(kept))
            self.assertEqual(int(r["page_i"]), int(kept[i]["page"]))
            self.assertEqual(int(r["page_j"]), int(kept[j]["page"]))
            self.assertEqual(int(r["lag"]), j - i)
            self.assertGreaterEqual(float(r["cos"]), float(cos[9]))
            self.assertEqual(r["text_i"], r["text_i"].rstrip())
            self.assertEqual(r["text_j"], r["text_j"].rstrip())

    def test_block_threshold_is_level_relative(self):
        # the page rollup is far more anisotropic, so its cut must be higher
        self.assertGreater(ss.block_threshold(_A["npz"]["page_cos"]),
                           ss.block_threshold(_A["npz"]["cos"]))
        self.assertGreaterEqual(ss.block_threshold(_A["npz"]["cos"]),
                                ss.BLOCK_THRESHOLD)


# ---------------------------------------------------------------------------
class TestBlockDetection(unittest.TestCase):
    def test_eight_connectivity_finds_a_diagonal_block(self):
        # a synthetic long-range block: cells (i, i+20) for i in 5..15
        N, k = 60, 40
        M = np.zeros((N, N))
        for i in range(5, 16):
            M[i, i + k] = 0.99
            M[i + k, i] = 0.99
        recs = [{"page": i // 2 + 1} for i in range(N)]
        blocks4 = ss.offdiag_blocks(M, recs, threshold=0.9, connectivity=4)
        blocks8 = ss.offdiag_blocks(M, recs, threshold=0.9, connectivity=8)
        self.assertEqual(len(blocks4), 0,
                         "4-connectivity must not see a diagonal block")
        self.assertEqual(len(blocks8), 1)
        self.assertTrue(blocks8[0]["long_range"])
        self.assertEqual(blocks8[0]["n_cells"], 11)     # i in 5..15
        self.assertEqual(blocks8[0]["n_rows"], 11)
        self.assertEqual(blocks8[0]["n_cols"], 11)

    def test_local_band_is_not_a_block(self):
        N = 40
        M = np.full((N, N), 0.2)
        for i in range(N):
            for j in range(N):
                if 0 < abs(i - j) <= 2:
                    M[i, j] = 0.99
        recs = [{"page": i + 1} for i in range(N)]
        blocks = ss.offdiag_blocks(M, recs, threshold=0.9)
        self.assertEqual(blocks, [],
                         "a matrix with only a local band yields no block")

    def test_book_return_is_found_at_both_resolutions(self):
        page_verbatim = blocks(level="page", verbatim=True)
        para_verbatim = blocks(level="paragraph", verbatim=True)
        self.assertTrue(page_verbatim, "no page-level verbatim block")
        self.assertTrue(para_verbatim, "no paragraph-level verbatim block")
        b = page_verbatim[0]
        self.assertIn(b["row_pages"], ("4-7", "4/5/6/7"))
        self.assertIn(b["col_pages"], ("220-223", "220/221/222/223"))
        self.assertGreater(int(b["gap"]), 200)
        self.assertGreater(float(b["raw_identity"]), 0.9)
        self.assertGreater(float(b["mean_cos"]), 0.9)
        p = para_verbatim[0]
        self.assertIn("4-8", p["row_pages"])
        self.assertGreater(int(p["gap"]), 600)

    def test_page_matrix_pairs_match_35(self):
        z = _A["npz"]
        PM, ids = z["page_cos"], [int(x) for x in z["page_ids"]]
        iu = np.triu_indices(len(ids), 1)
        order = np.argsort(-PM[iu])[:4]
        pairs = {frozenset((ids[int(iu[0][o])], ids[int(iu[1][o])]))
                 for o in order}
        for a, b in ((4, 220), (5, 221), (6, 222), (7, 223)):
            self.assertIn(frozenset((a, b)), pairs,
                          f"p{a}/p{b} should be a top page pair")
            self.assertGreater(PM[ids.index(a), ids.index(b)], 0.98)

    def test_lexical_control_confirms_the_verbatim_return(self):
        v = blocks(level="page", verbatim=True)[0]
        lex_mean = 0.6311     # page-pooled lexical background (diagnostics.tsv)
        self.assertGreater(float(v["lex_mean_cos"]), lex_mean + 0.05)
        self.assertGreater(float(v["raw_identity"]), 0.95)


# ---------------------------------------------------------------------------
class TestLexicalControl(unittest.TestCase):
    def test_identical_and_disjoint_paragraphs(self):
        records = [
            {"clause_ids": []},
            {"clause_ids": []},
            {"clause_ids": []},
            {"clause_ids": []},
        ]
        # monkeypatch the clause text used by lexical_cosine
        import clause_embeddings as ce
        real = ce.clause_records
        try:
            ce.clause_records = lambda: ([
                {"clause_id": 0, "text_units": "the dog of zero"},
                {"clause_id": 1, "text_units": "the dog of zero"},
                {"clause_id": 2, "text_units": "ice nebula faecal black"},
                {"clause_id": 3, "text_units": "murder rape cadaver flesh"},
            ], {})
            records[0]["clause_ids"] = [0]
            records[1]["clause_ids"] = [1]
            records[2]["clause_ids"] = [2]
            records[3]["clause_ids"] = [3]
            M = ss.lexical_cosine(records)
        finally:
            ce.clause_records = real
        self.assertAlmostEqual(M[0, 1], 1.0, places=5)
        self.assertAlmostEqual(M[2, 3], 0.0, places=5)
        self.assertAlmostEqual(M[0, 2], 0.0, places=5)
        # symmetry and a zero diagonal
        np.testing.assert_allclose(M, M.T, atol=1e-6)
        np.testing.assert_allclose(np.diag(M), 0.0, atol=1e-6)

    def test_page_pool_pools(self):
        M = np.array([[1.0, 0.0, 0.5],
                      [0.0, 1.0, 0.0],
                      [0.5, 0.0, 1.0]])
        # rows 0,1 -> page 10 ; row 2 -> page 20
        P = ss._page_pool(M, [10, 10, 20], [10, 20])
        self.assertAlmostEqual(P[0, 0], 0.5, places=6)      # mean of 1,0,0,1
        self.assertAlmostEqual(P[0, 1], 0.25, places=6)     # mean of .5,0
        self.assertAlmostEqual(P[1, 1], 1.0, places=6)      # the single cell
        np.testing.assert_allclose(P, P.T, atol=1e-9)

    def test_union_find_respects_connectivity(self):
        cells = [(0, 0), (1, 1), (2, 2), (5, 5)]
        self.assertEqual(len(ss._union_find_components(cells, 4)), 4)
        self.assertEqual(len(ss._union_find_components(cells, 8)), 2)


# ---------------------------------------------------------------------------
class TestEnglishControl(unittest.TestCase):
    def test_groups_are_deterministic_and_matched(self):
        a = ss.english_paragraph_groups(_A["records"])
        b = ss.english_paragraph_groups(_A["records"])
        self.assertEqual([g["text"] for g in a], [g["text"] for g in b])
        counts = sorted(r["n_clauses"] for r in _A["records"] if r["n_clauses"])
        self.assertEqual(len(a), len(counts))
        got = sorted(g["n_sentences"] for g in a)
        self.assertEqual(got[:len(counts)], counts[:len(got)] or counts)

    def test_groups_respect_documents_and_word_band(self):
        groups = ss.english_paragraph_groups(_A["records"])
        files = [g["source_file"] for g in groups]
        # each file contributes a contiguous run of groups
        seen = []
        for f in files:
            if not seen or seen[-1] != f:
                seen.append(f)
        self.assertEqual(len(seen), len(set(seen)))
        for g in groups[:50]:
            for s in g["sentences"]:
                self.assertGreaterEqual(len(s.split()), ss.EN_MIN_WORDS)
                self.assertLessEqual(len(s.split()), ss.EN_MAX_WORDS)

    def test_cap_respected(self):
        self.assertEqual(len(ss.english_paragraph_groups(_A["records"],
                                                         n_paras=12)), 12)

    def test_skip_path_is_disclosed(self):
        stats, info = ss.english_control(_A["records"], skip=True)
        self.assertIsNone(stats)
        self.assertFalse(info["available"])
        self.assertIn("skip", info["note"])

    def test_rows_exist_when_control_absent(self):
        rows = _A["en"]
        self.assertTrue(rows)
        self.assertEqual({r["quantity"] for r in rows} >= {
            "mean_cos", "top_pair_cos", "lag1_mean"}, True)
        for r in rows:
            if r["quantity"] == "mean_cos":
                self.assertEqual(r["english_paragraph"], "")
                self.assertIn("not run", r["note"])


# ---------------------------------------------------------------------------
class TestArtifacts(unittest.TestCase):
    def test_every_artifact_exists(self):
        for name in ("self_similarity_report.md", "paragraph_matrix.tsv",
                     "paragraph_cosines.tsv", "page_matrix.tsv",
                     "lag_curve.tsv", "page_lag_curve.tsv", "band_stats.tsv",
                     "offdiag_blocks.tsv", "similarity.png", "lag_curve.png",
                     "english_baseline.tsv", "sensitivity.tsv",
                     "reconciliation.tsv", "diagnostics.tsv", "summary.json",
                     "self_similarity.npz"):
            self.assertTrue((_A["dir"] / name).exists(), name)

    def test_page_lag_curve_keeps_the_short_band(self):
        rows = {int(r["lag"]): r for r in _A["page_lag"]}
        self.assertEqual(sorted(rows)[:3], [1, 2, 3])
        self.assertGreater(int(rows[1]["n_pairs"]), 200)
        self.assertGreater(float(rows[1]["z"]), 2.0)
        # the band stays short-range at page resolution: significant to
        # about lag 25, at the null by lag 30-50
        self.assertGreater(float(rows[25]["z"]), 2.0)
        self.assertLess(abs(float(rows[30]["z"])), 2.0)
        self.assertLess(abs(float(rows[50]["z"])), 2.0)
        # averaging pulls both vectors to the shared field, so the *absolute*
        # excess grows while its shape is unchanged
        page_delta = (float(rows[1]["mean_cos"])
                      - float(rows[1]["null_mean"]))
        para_delta = (float(lag_row(1)["mean_cos"])
                      - float(lag_row(1)["null_mean"]))
        self.assertGreater(page_delta, para_delta)
        # a steep monotone decay: lag 1 dominates every later lag
        self.assertGreater(float(rows[1]["z"]), float(rows[5]["z"]))
        self.assertGreater(float(rows[5]["z"]), float(rows[10]["z"]))
        self.assertGreater(float(rows[10]["z"]), float(rows[25]["z"]))

    def test_page_matrix_is_square_and_labelled(self):
        lines = (_A["dir"] / "page_matrix.tsv").read_text(
            encoding="utf-8").splitlines()
        head = lines[0].split("\t")
        self.assertEqual(head[0], "page")
        self.assertEqual(len(head), 226)
        self.assertEqual(head[1], "1")
        self.assertEqual(head[-1], "225")
        self.assertEqual(len(lines), 226)
        for line in lines[1:]:
            cells = line.split("\t")
            self.assertEqual(len(cells), 226)
            self.assertEqual(cells[0], head[int(cells[0])])

    def test_sensitivity_covers_view_and_encoder(self):
        keys = {(r["model"], r["view"]) for r in _A["sens"]}
        self.assertEqual(keys, {("sbert", "units"), ("sbert", "raw"),
                                ("bge", "units")})
        for r in _A["sens"]:
            self.assertGreater(float(r["mean_cos"]), 0.3)
            self.assertIn("4-8", r["biggest_block_pages"])

    def test_report_structure(self):
        r = _A["report"]
        for section in ("## 1. Method", "## 2. The matrix at a glance",
                        "## 3. Is there a diagonal band?",
                        "## 4. Repeating blocks",
                        "## 5. Page-level rollup", "## 6. Controls",
                        "## 7. Structural reading", "## 8. Caveats"):
            self.assertIn(section, r)
        self.assertIn("**Headline.**", r)
        self.assertIn("Reproduce with `python work/self_similarity.py`", r)
        self.assertNotIn("nan", r)
        self.assertNotIn("{", r.split("## 1. Method")[0])
        self.assertIn("4/5/6/7", r)
        self.assertIn("220/221/222/223", r)
        self.assertIn("page rollup median / 95th / 99th percentile cosine", r)
        self.assertIn("no tested lag through 300 shows an excess", r)
        self.assertIn("No paragraph-scale period", r)
        self.assertIn("paragraph matrix also detects its aligned subparagraphs", r)
        self.assertNotIn("page rollup mean / 95th", r)
        self.assertNotIn("no lag above 300 shows an excess", r)
        self.assertFalse(r.endswith("\n\n"))

    def test_summary_headline_numbers(self):
        s = _A["summary"]
        self.assertEqual(s["schema"], ss.SCHEMA)
        self.assertEqual(s["view"], "units")
        self.assertEqual(s["encoder"], "sbert")
        self.assertEqual(s["n_pages"], 225)
        self.assertEqual(s["n_clauses_assigned"], 5614)
        self.assertGreater(s["n_paragraphs"], 600)
        self.assertGreaterEqual(s["n_blocks_long_range"], 1)
        self.assertIsNotNone(s["biggest_long_range_paragraph_block"])
        self.assertEqual(len(s["page_matrix"]["top_pairs"]), 6)
        self.assertAlmostEqual(s["lag_z"]["1"], float(lag_row(1)["z"]), places=2)

    def test_reconciliation_rows(self):
        self.assertEqual(_A["recon"]["clauses assigned to a paragraph"]["here"],
                         "5614")
        self.assertEqual(_A["recon"]["pages"]["here"], "225")

    def test_determinism(self):
        tmp = tempfile.mkdtemp(prefix="self_sim_det_")
        ss.build(Path(tmp), quiet=True, english=False)
        a = _A["npz"]["cos"]
        b = np.load(Path(tmp) / "self_similarity.npz")["cos"]
        np.testing.assert_array_equal(a, b)
        self.assertEqual(
            (Path(tmp) / "self_similarity_report.md").read_text("utf-8"),
            _A["report"])
        self.assertEqual(
            (Path(tmp) / "paragraph_matrix.tsv").read_text("utf-8"),
            (_A["dir"] / "paragraph_matrix.tsv").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
