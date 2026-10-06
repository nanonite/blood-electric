#!/usr/bin/env python3
"""Regression suite for #22 case and emphasis inventory.

Runs the inventory once into a temp directory, then checks the invariants that
matter: the token-layer screen and its join to #17's unit layer, the four-class
decision (header / label / scream / function), the section-header report, the
markup-residue exclusion, and reconciliation.

    python work/test_case_emphasis.py
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import case_emphasis as ce

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="case_emphasis_test_")
    ce.main(["--out", tmp, "--quiet"])
    for name in ("caps_occurrences", "caps_types", "section_headers",
                 "per_page", "all_caps_units", "diagnostics",
                 "reconciliation"):
        with open(Path(tmp) / f"{name}.tsv", encoding="utf-8") as fh:
            _ARTIFACTS[name] = list(csv.DictReader(fh, delimiter="\t"))
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


def types_by_name():
    return {r["type"]: r for r in rows("caps_types")}


class TestLayer(unittest.TestCase):
    def test_word_token_layer(self):
        toks = ce.read_tsv(ce.TOKENS)
        words = [t for t in toks if t["kind"] == "WORD"]
        self.assertEqual(len(words), 77485)

    def test_joins_unit_layer(self):
        units = {r["uid"]: r for r in ce.read_tsv(ce.UNITS)}
        self.assertEqual(len(units), 74665)
        for o in rows("caps_occurrences"):
            self.assertIn(o["unit_uid"], units)

    def test_all_225_pages(self):
        self.assertEqual(len(ce.load_pages()), 225)


class TestInventory(unittest.TestCase):
    def test_caps_occurrences(self):
        self.assertEqual(_ARTIFACTS["summary"]["caps_occurrences"], 1890)

    def test_caps_types(self):
        self.assertEqual(_ARTIFACTS["summary"]["caps_types"], 141)

    def test_caps_share(self):
        self.assertAlmostEqual(_ARTIFACTS["summary"]["caps_share_of_words"],
                               0.0244, places=4)

    def test_occurrence_rows_match_summary(self):
        self.assertEqual(len(rows("caps_occurrences")),
                         _ARTIFACTS["summary"]["caps_occurrences"])

    def test_every_occurrence_is_all_caps(self):
        for o in rows("caps_occurrences"):
            self.assertTrue(ce.is_caps(o["text"]), o["text"])


class TestClasses(unittest.TestCase):
    def test_class_counts(self):
        by = _ARTIFACTS["summary"]["by_class_occurrences"]
        self.assertEqual(by["function"], 453)
        self.assertEqual(by["label"], 1138)
        self.assertEqual(by["scream"], 263)
        self.assertEqual(by["header"], 36)
        self.assertEqual(sum(by.values()), 1890)

    def test_adam_is_label(self):
        t = types_by_name()["ADAM"]
        self.assertEqual(t["class"], "label")
        self.assertEqual(t["subtype"], "proper_name")

    def test_load_is_command_scream(self):
        t = types_by_name()["LOAD"]
        self.assertEqual(t["class"], "scream")
        self.assertEqual(t["subtype"], "command")

    def test_tv_is_acronym_label(self):
        t = types_by_name()["TV"]
        self.assertEqual(t["class"], "label")
        self.assertEqual(t["subtype"], "acronym")

    def test_function_letters(self):
        t = types_by_name()
        self.assertEqual(t["I"]["class"], "function")
        self.assertEqual(t["A"]["class"], "function")
        self.assertEqual(t["I"]["n"], "407")
        self.assertEqual(t["A"]["n"], "46")

    def test_single_letter_designator(self):
        self.assertEqual(types_by_name()["K"]["subtype"], "single_letter")

    def test_scream_subtypes(self):
        by = _ARTIFACTS["summary"]["by_subtype_occurrences"]
        self.assertEqual(by["command"], 208)
        self.assertEqual(by["emphasis"], 55)

    def test_every_type_has_a_class(self):
        for r in rows("caps_types"):
            self.assertIn(r["class"], {"header", "label", "scream", "function"})

    def test_every_occurrence_has_a_reason(self):
        for o in rows("caps_occurrences"):
            self.assertTrue(o["reason"])


class TestHeaders(unittest.TestCase):
    def test_sections(self):
        self.assertEqual(len(rows("section_headers")), 56)

    def test_all_caps_sections(self):
        self.assertEqual(
            sum(1 for r in rows("section_headers") if r["all_caps_title"] == "y"),
            24)

    def test_header_caps_count(self):
        self.assertEqual(_ARTIFACTS["summary"]["header_caps_occurrences"], 36)

    def test_warning_title_is_header(self):
        t = types_by_name()["WARNING"]
        self.assertEqual(t["class"], "header")
        self.assertEqual(t["header_n"], "1")

    def test_mixed_type_carries_both_classes(self):
        t = types_by_name()["ADAM"]
        self.assertIn("header:1", t["classes"])
        self.assertIn("label:327", t["classes"])

    def test_blood_electric_title(self):
        sec = {r["index"]: r for r in rows("section_headers")}
        self.assertEqual(sec["1"]["caps_tokens"], "BLOOD|ELECTRIC")
        self.assertEqual(sec["1"]["all_caps_title"], "y")


class TestEdgeCases(unittest.TestCase):
    def test_no_markup_residue(self):
        self.assertEqual(
            _ARTIFACTS["summary"]["markup_residue_caps_occurrences"], 0)

    def test_naming_units(self):
        self.assertEqual(
            _ARTIFACTS["summary"]["naming_unit_caps_occurrences"], 91)

    def test_all_caps_units(self):
        s = _ARTIFACTS["summary"]
        self.assertEqual(s["all_caps_unit_types"], 134)
        self.assertEqual(s["all_caps_unit_occurrences"], 1512)

    def test_ambiguous_types_disclose_lowercase(self):
        t = types_by_name()
        self.assertEqual(t["ANDROID"]["lower_exact_n"], "47")
        self.assertEqual(t["ANDROID"]["class"], "label")

    def test_first_attestation_recorded(self):
        for r in rows("caps_types"):
            self.assertTrue(r["first_page"])
            self.assertTrue(r["first_start"])


class TestReconciliation(unittest.TestCase):
    def test_reconciliation_rows(self):
        rec = {r["quantity"]: r for r in rows("reconciliation")}
        self.assertEqual(rec["word tokens"]["here"], "77485")
        self.assertEqual(rec["section headers"]["here"], "56")

    def test_per_page_covers_book(self):
        self.assertEqual(len(rows("per_page")), 225)


if __name__ == "__main__":
    unittest.main(verbosity=2)
