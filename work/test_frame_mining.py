#!/usr/bin/env python3
"""Regression suite for #34 frame and template mining.

Runs the miner once into a temp directory, then checks the invariants that
matter:

  * the clause/unit/span inputs reconcile with #33 / #27 / #14 exactly;
  * every frame is a well-formed skeleton (>= 1 fixed word, exactly ``n_slots``
    slots), attested by >= ``MIN_CLAUSES`` clauses and >= ``MIN_FILLERS``
    fillers per slot, and its reported ``n_instances`` equals an independent
    recount from the clause tokens;
  * the catalogue is deterministically ranked and the task's named schemata
    (``of the <_>``, ``the <_> that``, ``i <_> the``, two-slot ``of`` frames)
    are present;
  * the signature-family tag matches the disclosed rule and the family table
    aggregates the catalogue;
  * the ``<<...>>`` block-name catalogue is drawn only from spans that open
    with ``<<`` and never starts a name with punctuation or an HTML tag;
  * coverage is monotonic and its top row matches an independent recompute;
  * two builds produce byte-identical TSVs (determinism).

    python work/test_frame_mining.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clause_segmentation as cs
import frame_mining as fm

_ARTIFACTS = {}
_FILLER_RE = re.compile(r"\((\d+)\)")


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="frame_mining_test_")
    fm.main(["--out", tmp, "--quiet"])
    for name in ("frames", "family_counts", "block_names", "frame_coverage",
                 "diagnostics", "reconciliation"):
        _ARTIFACTS[name] = cs.read_tsv(Path(tmp) / f"{name}.tsv")
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


def diag():
    return {r["diag"]: r["value"] for r in rows("diagnostics")}


def frame(skeleton):
    for r in rows("frames"):
        if r["skeleton"] == skeleton:
            return r
    return None


def norm(records):
    """Stringify a list of dicts so JSON and TSV reads compare equal."""
    return [{k: ("" if v is None else str(v)) for k, v in rec.items()}
            for rec in records]


class TestReconciliation(unittest.TestCase):
    def test_clauses_match_33(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_clauses"], 5614)

    def test_screened_units_match_27(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_units"], 74077)

    def test_spans_match_14(self):
        self.assertEqual(_ARTIFACTS["summary"]["n_spans"], 432)
        self.assertEqual(_ARTIFACTS["summary"]["n_spans_closed"], 250)

    def test_reconciliation_rows_agree(self):
        rec = {r["quantity"]: r["here"] for r in rows("reconciliation")}
        self.assertEqual(rec["clauses"], "5614")
        self.assertEqual(rec["screened units"], "74077")
        self.assertEqual(rec["spans (<<...>>)"], "432")
        self.assertEqual(rec["closed spans"], "250")

    def test_every_span_opens_with_marker(self):
        d = diag()
        self.assertEqual(int(d["spans"]), int(d["spans_open_marker"]))
        self.assertEqual(int(d["spans"]), 432)


class TestFrameCatalogue(unittest.TestCase):
    def test_ids_are_ordered(self):
        ids = [int(r["frame_id"]) for r in rows("frames")]
        self.assertEqual(ids, list(range(1, len(ids) + 1)))

    def test_slot_counts_agree_with_summary(self):
        n_single = sum(1 for r in rows("frames") if r["n_slots"] == "1")
        n_double = sum(1 for r in rows("frames") if r["n_slots"] == "2")
        s = _ARTIFACTS["summary"]
        self.assertEqual(n_single, s["n_single"])
        self.assertEqual(n_double, s["n_double"])
        self.assertEqual(n_single + n_double, s["n_frames"])
        self.assertEqual(len(rows("frames")), s["n_frames"])

    def test_every_skeleton_is_well_formed(self):
        for r in rows("frames"):
            parts = r["skeleton"].split(" ")
            n_slots = int(r["n_slots"])
            self.assertIn(n_slots, (1, 2))
            self.assertEqual(parts.count(fm.SLOT), n_slots)
            self.assertGreater(len(parts) - n_slots, 0)   # >= 1 fixed word

    def test_every_frame_is_attested(self):
        for r in rows("frames"):
            self.assertGreaterEqual(int(r["n_clauses"]), fm.MIN_CLAUSES)
            self.assertGreaterEqual(int(r["n_instances"]),
                                    int(r["n_clauses"]))

    def test_every_slot_has_min_fillers(self):
        for r in rows("frames"):
            n1 = len(_FILLER_RE.findall(r["slot1_fillers"]))
            self.assertGreaterEqual(n1, fm.MIN_FILLERS)
            if r["n_slots"] == "2":
                n2 = len(_FILLER_RE.findall(r["slot2_fillers"]))
                self.assertGreaterEqual(n2, fm.MIN_FILLERS)
            else:
                self.assertEqual(r["slot2_fillers"], "")

    def test_family_matches_rule(self):
        for r in rows("frames"):
            skel = tuple(r["skeleton"].split(" "))
            self.assertEqual(r["family"], fm.family_of(skel))
            self.assertIn(r["family"], fm.FAMILY_ORDER)

    def test_ranked_by_coverage_then_instances(self):
        prev = None
        for r in rows("frames"):
            key = (int(r["n_clauses"]), int(r["n_instances"]))
            if prev is not None:
                self.assertLessEqual(key[0], prev[0])
                if key[0] == prev[0]:
                    self.assertLessEqual(key[1], prev[1])
            prev = key

    def test_named_schemata_present(self):
        for skel in ("of the " + fm.SLOT,
                     "the " + fm.SLOT + " that",
                     "i " + fm.SLOT + " the",
                     "the " + fm.SLOT + " of the " + fm.SLOT,
                     "of the " + fm.SLOT + " of " + fm.SLOT):
            self.assertIsNotNone(frame(skel), skel)

    def test_first_person_is_the_i_template(self):
        fam = {r["family"]: r for r in rows("family_counts")}
        self.assertEqual(fam["first_person"]["n_frames"], "1")
        self.assertEqual(fam["first_person"]["top_frame"], "i " + fm.SLOT + " the")


class TestFamilyRules(unittest.TestCase):
    def test_rules(self):
        S = fm.SLOT
        self.assertEqual(fm.family_of(("i", S, "the")), "first_person")
        self.assertEqual(fm.family_of(("the", S, "that")), "rel_that")
        self.assertEqual(fm.family_of(("of", "the", S)), "np_of_np")
        self.assertEqual(fm.family_of(("to", "the", S)), "pp_to")
        self.assertEqual(fm.family_of(("x", "y")), "other")

    def test_priority_first_person_beats_of(self):
        S = fm.SLOT
        self.assertEqual(fm.family_of(("i", S, "the", S, "of", S)),
                         "first_person")

    def test_family_table_aggregates(self):
        fams = rows("family_counts")
        self.assertEqual([r["family"] for r in fams], list(fm.FAMILY_ORDER))
        self.assertEqual(sum(int(r["n_frames"]) for r in fams),
                         len(rows("frames")))
        for r in fams:
            if r["top_frame"]:
                f = frame(r["top_frame"])
                self.assertIsNotNone(f)
                self.assertEqual(f["family"], r["family"])


class TestInstanceCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clauses = fm.load_clause_tokens(fm.load_context())

    def _recount(self, skeleton):
        k = len(skeleton)
        n = 0
        for cl in self.clauses:
            toks = cl["tokens"]
            for i in range(len(toks) - k + 1):
                if all(skeleton[t] == fm.SLOT or skeleton[t] == toks[i + t]
                       for t in range(k)):
                    n += 1
        return n

    def test_single_slot_recount(self):
        skel = ("of", "the", fm.SLOT)
        self.assertEqual(int(frame(" ".join(skel))["n_instances"]),
                         self._recount(skel))

    def test_two_slot_recount(self):
        skel = ("of", "the", fm.SLOT, fm.SLOT, "that")
        self.assertEqual(int(frame(" ".join(skel))["n_instances"]),
                         self._recount(skel))

    def test_every_frame_recounts(self):
        for r in rows("frames"):
            skel = tuple(r["skeleton"].split(" "))
            self.assertEqual(int(r["n_instances"]), self._recount(skel),
                             r["skeleton"])


class TestBlockNames(unittest.TestCase):
    def test_named_spans(self):
        d = diag()
        self.assertEqual(int(d["block_names_named_spans"]), 432)
        self.assertEqual(int(d["block_names_residue_leading"]), 9)
        self.assertEqual(int(d["block_names_distinct"]), len(rows("block_names")))

    def test_top_name(self):
        top = rows("block_names")[0]
        self.assertEqual(top["name"], "entrails that radiate")
        self.assertEqual(int(top["n_spans"]), 30)

    def test_names_never_start_with_punctuation_or_tag(self):
        for r in rows("block_names"):
            first = r["name"].split(" ")[0]
            self.assertTrue(any(c.isalnum() for c in first), r["name"])
            self.assertNotIn(first.lower(), cs.HTML_TAGS, r["name"])

    def test_first_class_is_known(self):
        allowed = {"english", "proper", "gazetteer", "label", "compound",
                   "true_coinage", "junk", "ocr_error", "rare_english",
                   "numeric", "unknown"}
        for r in rows("block_names"):
            self.assertIn(r["first_class"], allowed)

    def test_sorted_by_recurrence(self):
        counts = [int(r["n_spans"]) for r in rows("block_names")]
        self.assertEqual(counts, sorted(counts, reverse=True))


class TestCoverage(unittest.TestCase):
    def test_rows(self):
        cov = rows("frame_coverage")
        self.assertEqual(len(cov), 40)
        self.assertEqual([int(r["rank"]) for r in cov], list(range(1, 41)))

    def test_monotonic(self):
        prev = 0
        for r in rows("frame_coverage"):
            n = int(r["n_covered"])
            self.assertGreaterEqual(n, prev)
            prev = n

    def test_rank1_matches_top_frame(self):
        cov = rows("frame_coverage")
        self.assertEqual(cov[0]["frame"], rows("frames")[0]["skeleton"])

    def test_fraction_matches_counts(self):
        for r in rows("frame_coverage"):
            self.assertAlmostEqual(float(r["frac_covered"]),
                                   int(r["n_covered"]) / 5614, places=5)

    def test_top_row_recomputes(self):
        top = rows("frames")[0]
        skel = tuple(top["skeleton"].split(" "))
        clauses = fm.load_clause_tokens(fm.load_context())
        recomputed = fm.coverage(clauses, [{"skeleton": skel}])[0]
        self.assertEqual(int(rows("frame_coverage")[0]["n_covered"]),
                         recomputed["n_covered"])

    def test_summary_matches_rows(self):
        self.assertEqual(norm(_ARTIFACTS["summary"]["coverage"]),
                         norm(rows("frame_coverage")))


class TestSummaryConsistency(unittest.TestCase):
    def test_top_frames_matches_catalogue(self):
        self.assertEqual(norm(_ARTIFACTS["summary"]["top_frames"]),
                         norm(rows("frames")[:40]))

    def test_families_match(self):
        self.assertEqual(norm(_ARTIFACTS["summary"]["families"]),
                         norm(rows("family_counts")))

    def test_block_class_counts_sum(self):
        # one class per name-bearing span (432), not per distinct name
        self.assertEqual(sum(_ARTIFACTS["summary"]["block_class_counts"].values()),
                         432)


class TestDeterminism(unittest.TestCase):
    def test_second_build_is_byte_identical(self):
        tmp2 = tempfile.mkdtemp(prefix="frame_mining_test2_")
        fm.main(["--out", tmp2, "--quiet"])
        for name in ("frames", "family_counts", "block_names",
                     "frame_coverage"):
            a = (Path(_ARTIFACTS["_dir"]) / f"{name}.tsv").read_bytes()
            b = (Path(tmp2) / f"{name}.tsv").read_bytes()
            self.assertEqual(a, b, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
