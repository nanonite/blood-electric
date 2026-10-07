#!/usr/bin/env python3
"""Regression suite for #33 clause segmentation via operators.

Runs the segmentation once into a temp directory, then checks the invariants
that matter:

  * the three boundary operators are exactly #14's counts, recomputed from a
    fresh AST parse (not read back from the artifact);
  * the clause partition reconstitutes the 90,220-token #14 stream
    (clause tokens + boundary tokens == total);
  * no #17 unit crosses a clause boundary and no clause crosses a page;
  * the length distributions aggregate to the clause inventory and the markup
    screen only ever removes words;
  * the persisted artifacts agree with `summary.json`.

    python work/test_clause_segmentation.py
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

import clause_segmentation as cs
import operator_grammar as og

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="clause_segmentation_test_")
    cs.main(["--out", tmp, "--quiet"])
    for name in ("clauses", "length_by_opener", "length_by_container",
                 "length_histogram", "composition_by_opener",
                 "opener_transitions", "per_page", "diagnostics",
                 "reconciliation"):
        with open(Path(tmp) / f"{name}.tsv", encoding="utf-8") as fh:
            _ARTIFACTS[name] = list(csv.DictReader(fh, delimiter="\t"))
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


class TestBoundaryReconciliation(unittest.TestCase):
    """The boundaries must be #14's, recomputed independently."""

    @classmethod
    def setUpClass(cls):
        book, _ = og.parse_book()
        nodes = list(book.walk())
        utt = [n for n in nodes if n.k == "Utterance"
               and n.x.get("open_op") == "//"]
        fld = [n for n in nodes if n.k == "Field"
               and n.x.get("open_op") == "::"]
        orph = [n for n in nodes if n.k == "Op"
                and n.x.get("role") == "orphan-boundary"]
        slash = [n for n in nodes if n.k == "Slash"]
        cls.ref = {
            "//": len(utt) + sum(1 for n in orph if n.v == "//"),
            "::": len(fld) + sum(1 for n in orph if n.v == "::"),
            "/": len(slash),
        }
        cls.total_tokens = sum(len(p.token_list()) for p in book.kids)

    def test_slash_boundary_is_540(self):
        self.assertEqual(self.ref["/"], 540)

    def test_utterance_boundary_is_2149(self):
        self.assertEqual(self.ref["//"], 2149)

    def test_field_boundary_is_3084(self):
        self.assertEqual(self.ref["::"], 3084)

    def test_summary_matches_recomputation(self):
        s = _ARTIFACTS["summary"]
        self.assertEqual(s["boundary_by_op"], self.ref)
        self.assertEqual(s["boundary_total"], sum(self.ref.values()))

    def test_token_stream_is_90220(self):
        self.assertEqual(self.total_tokens, 90220)
        self.assertEqual(_ARTIFACTS["summary"]["total_tokens"], 90220)

    def test_partition_is_lossless(self):
        s = _ARTIFACTS["summary"]
        self.assertTrue(s["partition_ok"])
        self.assertEqual(s["clause_tokens"] + s["boundary_total"],
                         self.total_tokens)


class TestInventory(unittest.TestCase):
    def test_clause_count(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_clauses"], 5614)

    def test_rows_match_summary(self):
        self.assertEqual(len(rows("clauses")),
                         _ARTIFACTS["summary"]["n_clauses"])

    def test_every_clause_has_a_boundary_or_page_edge_opener(self):
        for r in rows("clauses"):
            self.assertIn(r["opener"], cs.OPENER_TYPES)

    def test_every_clause_has_a_closer(self):
        valid = {"//", "::", "/", "HDR", "EOS"}
        for r in rows("clauses"):
            self.assertIn(r["closer"], valid)

    def test_opener_counts(self):
        by = _ARTIFACTS["summary"]["by_opener"]
        self.assertEqual(by["BOS"], 223)
        self.assertEqual(by["HDR"], 69)
        self.assertEqual(by["//"], 1993)
        self.assertEqual(by["::"], 2898)
        self.assertEqual(by["/"], 431)
        self.assertEqual(sum(by.values()), 5614)

    def test_container_counts(self):
        by = _ARTIFACTS["summary"]["by_container"]
        self.assertEqual(by["block"], 3406)
        self.assertEqual(by["span"], 1329)
        self.assertEqual(by["inline"], 401)
        self.assertEqual(by["header"], 56)
        self.assertEqual(by["multi"], 422)

    def test_no_unit_crosses_a_boundary(self):
        self.assertEqual(_ARTIFACTS["summary"]["unit_crossings"], 0)

    def test_clauses_do_not_cross_pages(self):
        pages = {}
        for r in rows("clauses"):
            pages.setdefault(int(r["page"]), []).append(r)
        # every clause's offsets are inside its own page body
        bodies = cs.load_pages()
        for p, grp in pages.items():
            body = bodies[p]
            for r in grp:
                s, e = int(r["start"]), int(r["end"])
                self.assertGreaterEqual(s, 0)
                self.assertLessEqual(e, len(body))
                self.assertLessEqual(s, e)

    def test_words_never_exceed_tokens(self):
        for r in rows("clauses"):
            self.assertLessEqual(int(r["n_words"]), int(r["n_tokens"]))
            self.assertLessEqual(int(r["n_words_clean"]), int(r["n_words"]))

    def test_ids_are_unique_and_ordered(self):
        ids = [int(r["clause_id"]) for r in rows("clauses")]
        self.assertEqual(ids, list(range(len(ids))))

    def test_first_clause_per_page_is_earliest(self):
        first = {}
        for r in rows("clauses"):
            p = int(r["page"])
            if p not in first or int(r["clause_id"]) < int(first[p]["clause_id"]):
                first[p] = r
        self.assertEqual(len(first), 225)
        for p, r in first.items():
            self.assertGreaterEqual(int(r["start"]), 0)
            for other in rows("clauses"):
                if int(other["page"]) == p:
                    self.assertLessEqual(int(r["start"]), int(other["start"]))


class TestMarkupScreen(unittest.TestCase):
    def test_raw_words_match_17(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_words_raw"], 77485)

    def test_screened_words(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_words_screened"], 76897)
        self.assertEqual(_ARTIFACTS["summary"]["markup_words_removed"], 588)

    def test_raw_units_match_17(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_units_raw"], 74665)

    def test_screened_units_match_27(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_units"], 74077)
        self.assertEqual(_ARTIFACTS["summary"]["markup_units_removed"], 588)

    def test_clean_never_exceeds_raw(self):
        for r in rows("clauses"):
            self.assertLessEqual(int(r["n_words_clean"]), int(r["n_words"]))


class TestDistributions(unittest.TestCase):
    def test_opener_groups_cover_inventory(self):
        s = _ARTIFACTS["summary"]
        by_opener = [r for r in rows("length_by_opener") if r["scope"] == "all"
                     and r["opener"] != "ALL"]
        self.assertEqual(sum(int(r["n"]) for r in by_opener),
                         s["n_clauses"])
        self.assertEqual(sum(int(r["total"]) for r in by_opener),
                         s["n_words_screened"])

    def test_all_row_matches_inventory(self):
        allrow = next(r for r in rows("length_by_opener")
                      if r["opener"] == "ALL" and r["scope"] == "all")
        self.assertEqual(int(allrow["n"]), 5614)
        self.assertEqual(int(allrow["total"]), 76897)

    def test_median_and_mean(self):
        s = _ARTIFACTS["summary"]
        self.assertEqual(s["words_per_clause_median"], 11.0)
        self.assertAlmostEqual(s["words_per_clause_mean"], 13.697, places=3)

    def test_container_groups_cover_inventory(self):
        s = _ARTIFACTS["summary"]
        self.assertEqual(sum(int(r["n"]) for r in rows("length_by_container")),
                         s["n_clauses"])
        self.assertEqual(sum(int(r["total"])
                             for r in rows("length_by_container")),
                         s["n_words_screened"])

    def test_header_clauses_are_short(self):
        hdr = next(r for r in rows("length_by_container")
                   if r["container"] == "header")
        self.assertEqual(int(hdr["n"]), 56)
        self.assertLessEqual(float(hdr["median"]), 2.0)

    def test_slash_opener_is_longest_group(self):
        grp = {r["opener"]: float(r["mean"])
               for r in rows("length_by_opener")
               if r["scope"] == "all" and r["opener"] != "ALL"}
        body = {k: v for k, v in grp.items() if k != "HDR"}
        self.assertEqual(max(body, key=body.get), "/")

    def test_histogram_totals_match_groups(self):
        hist = {}
        for r in rows("length_histogram"):
            hist.setdefault(r["opener"], 0)
            hist[r["opener"]] += int(r["n"])
        by = _ARTIFACTS["summary"]["by_opener"]
        for op, n in by.items():
            self.assertEqual(hist.get(op, 0), n, op)

    def test_composition_groups_cover_inventory(self):
        by = _ARTIFACTS["summary"]["by_opener"]
        comp = {r["opener"]: r for r in rows("composition_by_opener")}
        self.assertEqual(set(comp), set(by))
        for op, n in by.items():
            self.assertEqual(int(comp[op]["n_clauses"]), n)
        # a clause cannot have more units than words
        for r in rows("composition_by_opener"):
            self.assertLessEqual(float(r["mean_units"]), float(r["mean_words"]))

    def test_transitions_are_same_page(self):
        total = sum(int(r["n"]) for r in rows("opener_transitions"))
        # one transition per adjacent same-page pair
        n = _ARTIFACTS["summary"]["n_clauses"]
        pages = {}
        for r in rows("clauses"):
            pages.setdefault(int(r["page"]), 0)
            pages[int(r["page"])] += 1
        self.assertEqual(total, n - len(pages))

    def test_per_page_covers_all_pages(self):
        self.assertEqual(len(rows("per_page")), 225)
        self.assertEqual(sum(int(r["n_clauses"]) for r in rows("per_page")),
                         5614)


class TestReconciliationArtifact(unittest.TestCase):
    def test_reconciliation_rows(self):
        rec = {r["quantity"]: r for r in rows("reconciliation")}
        self.assertEqual(rec["boundary //"]["here"], "2149")
        self.assertEqual(rec["boundary ::"]["here"], "3084")
        self.assertEqual(rec["boundary / (sentence-final)"]["here"], "540")
        self.assertEqual(rec["total tokens"]["here"], "90220")

    def test_diagnostics_present(self):
        diag = {r["metric"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(diag["clauses"], "5614")
        self.assertEqual(diag["partition_ok"], "1")
        self.assertEqual(diag["unit_crossings"], "0")
        self.assertEqual(diag["header_edges"], "75")


if __name__ == "__main__":
    unittest.main(verbosity=2)
