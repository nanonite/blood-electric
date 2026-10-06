#!/usr/bin/env python3
"""
Regression tests for work/validated_segmentation.py (Chainlink #20).

Tests the gazetteer validation logic, the Morfessor/UniMorph segmenters,
and the precision spot-check sampling.
"""
import collections
import csv
import json
import os
import sys
import unittest

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK_DIR)

from validated_segmentation import (
    load_gazetteer,
    load_closed_juncture_types,
    load_unimorph_english,
    run_morfessor,
    run_unimorph,
    validate_against_gazetteer,
    precision_spot_check,
    OUTDIR,
)


class TestGazetteer(unittest.TestCase):
    def test_loads(self):
        gaz = load_gazetteer()
        self.assertGreater(len(gaz), 1000)
        self.assertIn("tokage", gaz)
        self.assertEqual(gaz["tokage"], "entity")
        self.assertIn("boy-roid", gaz)
        self.assertEqual(gaz["boy-roid"], "neologism")

    def test_validate_unprotected(self):
        gaz = {"tokage": "entity"}
        v = validate_against_gazetteer("unknownword", ["unknown", "word"], gaz)
        self.assertFalse(v["protected"])
        self.assertFalse(v["violation"])

    def test_validate_protected_not_segmented(self):
        gaz = {"tokage": "entity"}
        v = validate_against_gazetteer("tokage", ["tokage"], gaz)
        self.assertTrue(v["protected"])
        self.assertFalse(v["violation"])

    def test_validate_protected_segmented_violation(self):
        gaz = {"tokage": "entity"}
        v = validate_against_gazetteer("tokage", ["tok", "age"], gaz)
        self.assertTrue(v["protected"])
        self.assertTrue(v["violation"])
        self.assertEqual(v["category"], "entity")

    def test_validate_compound_segmented_ok(self):
        gaz = {"boy-roid": "compound"}
        v = validate_against_gazetteer("boy-roid", ["boy", "roid"], gaz)
        self.assertTrue(v["protected"])
        self.assertFalse(v["violation"])


class TestDataLoading(unittest.TestCase):
    def test_closed_juncture_types(self):
        types = load_closed_juncture_types()
        self.assertGreater(len(types), 3000)
        # All should be alphabetic
        for t in types:
            self.assertTrue(t["surface"].isalpha())
        # Should include known coinage
        surfaces = {t["surface"] for t in types}
        self.assertIn("boyroid", surfaces)
        self.assertIn("hydromania", surfaces)
        self.assertIn("tokage", surfaces)

    def test_unimorph_loads(self):
        lemma_to_forms, form_to_lemma = load_unimorph_english()
        self.assertGreater(len(form_to_lemma), 100000)
        # Check a known inflection
        self.assertIn("eats", form_to_lemma)
        self.assertEqual(form_to_lemma["eats"], "eat")


class TestMorfessor(unittest.TestCase):
    def test_segments_words(self):
        types = load_closed_juncture_types()[:100]
        result = run_morfessor(types)
        self.assertEqual(len(result), 100)
        # All segmentations should be non-empty lists of strings
        for word, seg in result.items():
            self.assertIsInstance(seg, list)
            self.assertGreater(len(seg), 0)
            for m in seg:
                self.assertIsInstance(m, str)

    def test_coinage_protection(self):
        """Morfessor should not segment protected coinages."""
        types = load_closed_juncture_types()
        gaz = load_gazetteer()
        result = run_morfessor(types)
        violations = []
        for word, seg in result.items():
            v = validate_against_gazetteer(word, seg, gaz)
            if v["violation"]:
                violations.append((word, seg, v["category"]))
        # Morfessor should have very few violations (ideally 0-2)
        self.assertLessEqual(len(violations), 5)


class TestUniMorph(unittest.TestCase):
    def test_segments_inflections(self):
        types = load_closed_juncture_types()[:100]
        _, form_to_lemma = load_unimorph_english()
        result = run_unimorph(types, form_to_lemma)
        self.assertEqual(len(result), 100)
        # Check that inflected forms are segmented
        # e.g., "eats" -> ["eat", "s"]
        if "eats" in result:
            self.assertEqual(result["eats"], ["eat", "s"])

    def test_no_false_positives(self):
        """UniMorph should not segment words that are their own lemma."""
        _, form_to_lemma = load_unimorph_english()
        # "cadaver" is its own lemma - should not be segmented
        result = run_unimorph([{"surface": "cadaver"}], form_to_lemma)
        self.assertEqual(result["cadaver"], ["cadaver"])


class TestSpotCheck(unittest.TestCase):
    def test_sampling(self):
        types = load_closed_juncture_types()[:200]
        gaz = load_gazetteer()
        morfessor_result = run_morfessor(types)
        _, form_to_lemma = load_unimorph_english()
        unimorph_result = run_unimorph(types, form_to_lemma)
        byt5_result = {t["surface"]: [t["surface"]] for t in types}
        spot_checks, violations = precision_spot_check(
            types, morfessor_result, unimorph_result, byt5_result, gaz, n_samples=20
        )
        self.assertGreater(len(spot_checks), 0)
        # All spot checks should have required fields
        for row in spot_checks:
            self.assertIn("word", row)
            self.assertIn("morfessor", row)
            self.assertIn("unimorph", row)
            self.assertIn("byt5", row)


class TestArtifacts(unittest.TestCase):
    def test_validation_tsv_exists(self):
        path = os.path.join(OUTDIR, "validation.tsv")
        if not os.path.exists(path):
            self.skipTest("validation.tsv not yet generated")
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        self.assertGreater(len(rows), 0)
        # Check columns
        expected = {"word", "n", "segmenter", "segmentation", "n_morphs",
                    "protected", "category", "violation", "reason"}
        self.assertEqual(set(rows[0].keys()), expected)

    def test_summary_json_exists(self):
        path = os.path.join(OUTDIR, "summary.json")
        if not os.path.exists(path):
            self.skipTest("summary.json not yet generated")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("n_word_types", data)
        self.assertIn("n_gazetteer", data)
        self.assertIn("violations", data)


if __name__ == "__main__":
    unittest.main()
