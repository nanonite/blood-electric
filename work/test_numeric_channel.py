#!/usr/bin/env python3
"""Regression suite for #21 numeric and symbol channel catalogue.

Runs the cataloguer once into a temp directory, then checks the invariants
that matter: the unit-layer screen, the channel classification, the
sub-inventories (entities, measurements, levels, zero, roman), the loop
evidence, the negative results, and reconciliation with #19.

    python work/test_numeric_channel.py
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numeric_channel as nc

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="numeric_channel_test_")
    nc.main(["--out", tmp, "--quiet"])
    for name in ("numeric_channel", "entity_numbers", "measurements", "levels",
                 "zero_channel", "roman_markers", "loop_evidence",
                 "diagnostics", "reconciliation"):
        with open(Path(tmp) / f"{name}.tsv", encoding="utf-8") as fh:
            _ARTIFACTS[name] = list(csv.DictReader(fh, delimiter="\t"))
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


def by_unit(name="numeric_channel"):
    return {r["unit"]: r for r in rows(name)}


class TestUnitLayer(unittest.TestCase):
    def test_screens_unit_layer(self):
        units = nc.load_units()
        self.assertEqual(len(units), 74665)

    def test_all_225_pages(self):
        pages = nc.load_pages()
        self.assertEqual(len(pages), 225)


class TestChannelClassification(unittest.TestCase):
    def test_channel_occurrences(self):
        # 300 channel occurrences out of 74,665 units
        self.assertEqual(_ARTIFACTS["summary"]["channel_occurrences"], 300)

    def test_channel_types(self):
        self.assertEqual(_ARTIFACTS["summary"]["channel_types"], 32)

    def test_digit_bearing_match_19(self):
        self.assertTrue(_ARTIFACTS["summary"]["digit_types_match_19"])

    def test_zero_is_spelled_number(self):
        t = by_unit()
        self.assertEqual(t["zero"]["channel"], "spelled_number")
        self.assertEqual(t["zero"]["kind"], "null_point")

    def test_xx_is_roman_marker(self):
        t = by_unit()
        self.assertEqual(t["XX"]["channel"], "roman_marker")
        self.assertEqual(t["XX"]["kind"], "roman")

    def test_666_is_entity_id(self):
        t = by_unit()
        self.assertEqual(t["666"]["channel"], "digit_bearing")
        self.assertEqual(t["666"]["kind"], "entity_id")

    def test_isbn_is_paratext(self):
        t = by_unit()
        self.assertEqual(t["1-84068-060-1"]["channel"], "paratext")
        self.assertEqual(t["1-84068-060-1"]["kind"], "isbn")

    def test_year_is_paratext(self):
        t = by_unit()
        self.assertEqual(t["2002"]["channel"], "paratext")
        self.assertEqual(t["2002"]["kind"], "year")

    def test_colspan_is_markup_residue(self):
        t = by_unit()
        self.assertEqual(t["2"]["channel"], "markup_residue")
        self.assertEqual(t["2"]["kind"], "markup")

    def test_4d_is_dimension(self):
        t = by_unit()
        self.assertEqual(t["4-D"]["channel"], "digit_bearing")
        self.assertEqual(t["4-D"]["kind"], "dimension")

    def test_1mg_is_measurement_dose(self):
        t = by_unit()
        self.assertEqual(t["1mg"]["channel"], "digit_bearing")
        self.assertEqual(t["1mg"]["kind"], "measurement_dose")

    def test_creature_13_entity_id(self):
        t = by_unit()
        self.assertEqual(t["13"]["kind"], "entity_id")

    def test_level_6(self):
        t = by_unit()
        self.assertEqual(t["6"]["kind"], "level")


class TestSubInventories(unittest.TestCase):
    def test_entity_numbers(self):
        ents = rows("entity_numbers")
        # Creature 13/9/8, Super Cherry 666/17/9, Necrodrome 13/9, NDRO 4
        self.assertGreaterEqual(len(ents), 10)
        units = {r["unit"] for r in ents}
        self.assertIn("13", units)
        self.assertIn("666", units)

    def test_measurements(self):
        meas = rows("measurements")
        # 8 seconds x9, 8 second x1, 1 milligram x2, 1mg x2
        self.assertEqual(len(meas), 14)

    def test_levels(self):
        lvls = rows("levels")
        # Level zero x46 standalone (2 more are fused: zero++her, zero/air),
        # Level 6 x1, Level 9 x1 — occurrence-level
        self.assertEqual(len(lvls), 48)

    def test_zero_channel(self):
        z = rows("zero_channel")
        # 180 standalone + fused zero units
        self.assertGreaterEqual(len(z), 180)

    def test_roman_markers(self):
        r = rows("roman_markers")
        # XX x26, XXX x5, XXXXXX x1, X x2, V x2
        self.assertEqual(len(r), 36)

    def test_loop_evidence(self):
        loop = rows("loop_evidence")
        # p4=p220, p5=p221, p6=p222, p7=p223, p41=p113, p209 markup x12
        self.assertGreaterEqual(len(loop), 6)
        pages = {r["pages"] for r in loop}
        self.assertIn("4,220", pages)
        self.assertIn("5,221", pages)
        self.assertIn("6,222", pages)
        self.assertIn("7,223", pages)


class TestNegativeResults(unittest.TestCase):
    def test_no_version_strings(self):
        self.assertEqual(_ARTIFACTS["summary"]["negatives"]["version_strings"], 0)

    def test_no_ordinals(self):
        self.assertEqual(_ARTIFACTS["summary"]["negatives"]["ordinals"], 0)

    def test_no_dates(self):
        self.assertEqual(_ARTIFACTS["summary"]["negatives"]["dates"], 0)

    def test_no_times(self):
        self.assertEqual(_ARTIFACTS["summary"]["negatives"]["times"], 0)

    def test_no_pure_symbol_units(self):
        self.assertEqual(_ARTIFACTS["summary"]["negatives"]["pure_symbol_units"], 0)


class TestReconciliation(unittest.TestCase):
    def test_reconciliation_table(self):
        recon = rows("reconciliation")
        quantities = {r["quantity"] for r in recon}
        self.assertIn("unit occurrences", quantities)
        self.assertIn("digit-bearing types", quantities)

    def test_digit_types_match(self):
        recon = rows("reconciliation")
        for r in recon:
            if r["quantity"] == "digit-bearing types":
                self.assertEqual(r["note"], "exact match")


class TestFirstAttestation(unittest.TestCase):
    def test_zero_first_attestation(self):
        t = by_unit()
        self.assertEqual(t["zero"]["first_page"], "5")

    def test_666_first_attestation(self):
        t = by_unit()
        self.assertEqual(t["666"]["first_page"], "56")

    def test_xx_first_attestation(self):
        t = by_unit()
        self.assertEqual(t["XX"]["first_page"], "4")


class TestQuoteSafety(unittest.TestCase):
    def test_tsv_quote_safe(self):
        for name in ("numeric_channel", "zero_channel", "roman_markers"):
            for r in rows(name):
                for v in r.values():
                    self.assertNotIn("\t", v)
                    self.assertNotIn("\n", v)


if __name__ == "__main__":
    unittest.main()
