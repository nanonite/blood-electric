#!/usr/bin/env python3
"""Regression tests for the ensemble OCR / glyph vote (#4).

Run from the workspace root (inside the `marker` conda env):

    python work/test_ensemble_ocr.py
    python -m unittest discover -s work -p 'test_*.py'   # whole suite

Two kinds of test:

  * **pure-function tests** - whitespace normalisation, pairwise character
    alignment, multi-alignment of insertions, column building, the vote and the
    category function, all on synthetic strings.  These need no GPU and no OCR;
  * **artifact reconciliation** - if `analysis/ensemble_ocr/summary.json` exists
    (i.e. a build has been run), the headline numbers are checked against the
    tables: every event is in the atlas, every override is in the change list,
    the per-page column counts sum to the total, and the consensus has one row
    per page.  These assert the build is self-consistent, not that the corpus
    is a particular size.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.normpath(os.path.join(HERE, ".."))

import ensemble_ocr as E  # noqa: E402

OUT = E.OUT


def _read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh if line.strip()]


class NormalizeTest(unittest.TestCase):
    def test_collapses_runs(self):
        n, m = E.normalize_ws("a  b\n\n\tc ")
        self.assertEqual(n, "a b c")
        self.assertEqual(m[0], 0)          # 'a'
        self.assertEqual(n[n.index("b")], "b")
        self.assertEqual(n[n.index("c")], "c")

    def test_no_leading_or_double_space(self):
        n, _m = E.normalize_ws("   x")
        self.assertEqual(n, "x")

    def test_trailing_space_stripped(self):
        n, m = E.normalize_ws("x   ")
        self.assertEqual(n, "x")
        self.assertEqual(len(n), len(m))

    def test_plain_text_identity(self):
        n, m = E.normalize_ws("abc")
        self.assertEqual(n, "abc")
        self.assertEqual(m, [0, 1, 2])


class AlignTest(unittest.TestCase):
    def test_equal(self):
        mp, ins = E.align_pair("abc", "abc")
        self.assertEqual(mp, [0, 1, 2])
        self.assertEqual(dict(ins), {})

    def test_deletion(self):
        mp, ins = E.align_pair("a::b", "a:b")
        # second ':' is unmatched -> deletion
        self.assertEqual(mp[0], 0)
        self.assertEqual(mp[1], 1)
        self.assertIsNone(mp[2])
        self.assertEqual(mp[3], 2)
        self.assertEqual(dict(ins), {})

    def test_insertion(self):
        mp, ins = E.align_pair("ab", "axb")
        self.assertEqual(mp, [0, 2])
        self.assertEqual(ins[1], [1])

    def test_substitution(self):
        mp, ins = E.align_pair("abc", "axc")
        self.assertEqual(mp, [0, 1, 2])
        self.assertEqual(dict(ins), {})

    def test_empty(self):
        mp, ins = E.align_pair("", "")
        self.assertEqual(mp, [])
        self.assertEqual(dict(ins), {})


class MultiAlignTest(unittest.TestCase):
    def test_identical_insertions_share_columns(self):
        cols = E.align_multi(["x", "x", ""])
        self.assertEqual(len(cols), 1)
        self.assertEqual(cols[0], ["x", "x", None])

    def test_different_insertions_align(self):
        cols = E.align_multi(["ab", "ab", "a"])
        self.assertEqual([c[2] for c in cols], ["a", None])

    def test_all_empty(self):
        self.assertEqual(E.align_multi(["", "", ""]), [])


class ColumnTest(unittest.TestCase):
    def test_substitution_column(self):
        cols = E.build_columns("a:b", {"t": "a=b", "d": "a=b"})
        sub = [c for c in cols if c["values"]["marker"] == ":"][0]
        self.assertEqual(sub["values"]["t"], "=")
        self.assertEqual(sub["values"]["d"], "=")

    def test_deletion_column(self):
        cols = E.build_columns("a::b", {"t": "a:b", "d": "a:b"})
        deleted = [c for c in cols if c["ref_index"] == 2][0]
        self.assertIsNone(deleted["values"]["t"])

    def test_insertion_column(self):
        cols = E.build_columns("a<b", {"t": "a<<b", "d": "a<<b"})
        inserted = [c for c in cols if c["ref_index"] is None]
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0]["values"]["t"], "<")
        self.assertIsNone(inserted[0]["values"]["marker"])

    def test_marker_is_always_present_in_ref_columns(self):
        cols = E.build_columns("hello", {"t": "hxllo", "d": "hello"})
        for c in cols:
            if c["ref_index"] is not None:
                self.assertIsNotNone(c["values"]["marker"])


class VoteTest(unittest.TestCase):
    def test_unanimous(self):
        c, _n, v = E.vote({"marker": "a", "t": "a", "d": "a", "e": "a"},
                         ["marker", "t", "d", "e"])
        self.assertEqual((c, v), ("a", "unanimous"))

    def test_majority_override(self):
        c, _n, v = E.vote({"marker": ":", "t": "=", "d": "=", "e": "="},
                         ["marker", "t", "d", "e"])
        self.assertEqual((c, v), ("=", "majority"))

    def test_deletion_majority(self):
        c, _n, v = E.vote({"marker": ":", "t": None, "d": None, "e": None},
                         ["marker", "t", "d", "e"])
        self.assertEqual((c, v), (None, "majority"))

    def test_insertion_majority(self):
        c, _n, v = E.vote({"marker": None, "t": "<", "d": "<", "e": "<"},
                         ["marker", "t", "d", "e"])
        self.assertEqual((c, v), ("<", "majority"))

    def test_ambiguous_tie_keeps_none(self):
        c, _n, v = E.vote({"marker": ":", "t": None, "d": ":", "e": None},
                         ["marker", "t", "d", "e"])
        self.assertEqual(v, "ambiguous")
        self.assertIsNone(c)

    def test_three_engine_threshold(self):
        # with only marker + 2 engines a 2-of-3 majority is enough
        c, _n, v = E.vote({"marker": "x", "t": "y", "d": "y"}, ["marker", "t", "d"])
        self.assertEqual((c, v), ("y", "majority"))


class CategoryTest(unittest.TestCase):
    def test_operator(self):
        self.assertEqual(E.categorize([":", "=", "=", "="], ":"), "operator")

    def test_deletion(self):
        self.assertEqual(E.categorize([":", None, None, None], ":"), "deletion")

    def test_insertion(self):
        self.assertEqual(E.categorize([None, "<", "<", "<"], None), "insertion")

    def test_whitespace_insertion(self):
        self.assertEqual(E.categorize([None, " ", " ", " "], None), "whitespace")

    def test_case(self):
        self.assertEqual(E.categorize(["A", "a", "a", "a"], "A"), "case")
        # I/l are homoglyphs, not a case pair: they lower to different letters
        self.assertEqual(E.categorize(["I", "l", "l", "l"], "I"), "digit_letter")

    def test_digit_letter(self):
        self.assertEqual(E.categorize(["1", "l", "l", "l"], "1"), "digit_letter")

    def test_pair_label(self):
        self.assertEqual(E.pair_label([":", None, None, None]), ":|∅")


class BuildPageTest(unittest.TestCase):
    def test_synthetic_page(self):
        ref_lines = [{"block_id": "/page/0/Text/0", "block_index": 0,
                      "line_index": 0, "bbox": "0,0,1,1", "text": "gene=TV"}]
        engine_pages = {
            "tesseract": {"text": "gene=TV", "words": [["gene=TV", 0, 0, 10, 10, 99.0]]},
            "doctr": {"text": "gene-TV", "words": [["gene-TV", 0, 0, 10, 10, 99.0]]},
            "easyocr": {"text": "gene-TV", "words": [["gene-TV", 0, 0, 10, 10, 99.0]]},
        }
        engines = ["marker", "tesseract", "doctr", "easyocr"]
        words = {e: {0: E.words_to_spans(engine_pages[e]["text"], engine_pages[e]["words"])}
                 for e in engine_pages}
        _cols, consensus, events, _ref, _norm = E.build_page(
            0, ref_lines, engine_pages, engines, words, 300)
        self.assertEqual(consensus, "gene=TV")            # marker + tesseract majority
        self.assertTrue(any(e["category"] == "operator" for e in events))
        # the '=' column is 3-of-4 against doctr/easyocr -> marker kept
        eq = [e for e in events if e["marker_char"] == "="]
        self.assertEqual(len(eq), 1)
        self.assertEqual(eq[0]["consensus_char"], "=")

    def test_majority_deletes_char(self):
        ref_lines = [{"block_id": "/page/0/Text/0", "block_index": 0,
                      "line_index": 0, "bbox": "0,0,1,1", "text": "a::b"}]
        engine_pages = {
            "tesseract": {"text": "a:b", "words": []},
            "doctr": {"text": "a:b", "words": []},
            "easyocr": {"text": "a:b", "words": []},
        }
        engines = ["marker", "tesseract", "doctr", "easyocr"]
        _cols, consensus, events, _ref, _norm = E.build_page(
            0, ref_lines, engine_pages, engines, {}, 300)
        self.assertEqual(consensus, "a:b")
        self.assertTrue(any(e["category"] == "deletion" for e in events))


class ArtifactReconciliationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary_path = os.path.join(OUT, "summary.json")
        if not os.path.exists(cls.summary_path):
            raise unittest.SkipTest("no build yet: run work/ensemble_ocr.py build")
        with open(cls.summary_path, encoding="utf-8") as fh:
            cls.summary = json.load(fh)
        cls.atlas = _read_tsv(os.path.join(OUT, "disagreement_atlas.tsv"))
        cls.changes = _read_tsv(os.path.join(OUT, "consensus_changes.tsv"))
        cls.pages = _read_tsv(os.path.join(OUT, "page_disagreement.tsv"))
        cls.agreement = _read_tsv(os.path.join(OUT, "engine_agreement.tsv"))
        cls.consensus = _read_tsv(os.path.join(OUT, "consensus.tsv"))

    def test_events_match_atlas(self):
        self.assertEqual(self.summary["total_events"], len(self.atlas))

    def test_changes_match_file(self):
        self.assertEqual(self.summary["total_changed"], len(self.changes))

    def test_page_columns_sum(self):
        self.assertEqual(self.summary["total_columns"],
                         sum(int(r["n_cols"]) for r in self.pages))

    def test_page_events_sum(self):
        self.assertEqual(self.summary["total_events"],
                         sum(int(r["n_events"]) for r in self.pages))

    def test_category_counts_sum(self):
        self.assertEqual(self.summary["total_events"],
                         sum(self.summary["category_counts"].values()))

    def test_engine_agreement_sums(self):
        for r in self.agreement:
            self.assertEqual(int(r["present"]),
                             int(r["agree_with_marker"]) + int(r["disagree_with_marker"]))

    def test_consensus_one_row_per_page(self):
        self.assertEqual(len(self.consensus), self.summary["pages"])

    def test_changed_rows_are_exactly_overrides(self):
        for r in self.changes:
            self.assertNotEqual(r["consensus_char"], r["marker_char"])

    def test_atlas_rows_are_non_unanimous(self):
        for r in self.atlas:
            self.assertIn(r["verdict"], ("majority", "ambiguous"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
