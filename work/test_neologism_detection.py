#!/usr/bin/env python3
"""Regression suite for #19 neologism detection.

Runs the detector once into a temp directory, then checks the invariants that
matter: the unit-layer join, the markup screen, the OOV categories, OCR
evidence, first attestation, and quote-safety.

    python work/test_neologism_detection.py
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import neologism_detection as nd

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="neologism_test_")
    nd.main(["--out", tmp, "--quiet"])
    for name in ("screening", "neologisms", "neologisms_lower", "ocr_evidence",
                 "numeric_channel", "markup_residue", "diagnostics",
                 "reconciliation"):
        with open(Path(tmp) / f"{name}.tsv", encoding="utf-8") as fh:
            _ARTIFACTS[name] = list(csv.DictReader(fh, delimiter="\t"))
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


def by_unit(name="screening"):
    return {r["unit"]: r for r in rows(name)}


class TestJoin(unittest.TestCase):
    def test_join_1to1_span_exact(self):
        units = nd.read_tsv(nd.UNITS)
        decs = nd.read_tsv(nd.DECOMP)
        joined = nd.join_units(units, decs)
        self.assertEqual(len(joined), 74665)

    def test_join_rejects_span_mismatch(self):
        units = nd.read_tsv(nd.UNITS)
        decs = nd.read_tsv(nd.DECOMP)
        bad = [dict(d) for d in decs]
        bad[0]["start"] = "999999"
        with self.assertRaises(SystemExit):
            nd.join_units(units, bad)


class TestMarkupScreen(unittest.TestCase):
    def test_markup_counts_reconcile(self):
        diag = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(int(diag["markup_base_occurrences"]), 231)
        self.assertEqual(int(diag["tokenizer_html_tag_residue"]), 229)
        self.assertEqual(int(diag["markup_extended_new_occurrences"]), 116)
        self.assertEqual(int(diag["markup_total_occurrences"]), 347)

    def test_screened_occurrences_reconcile(self):
        diag = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(int(diag["units_joined"]), 74665)
        self.assertEqual(int(diag["screened_occurrences"]), 74665 - 347)
        # #18 screened 74,434 by removing only the 231-occurrence base set.
        self.assertEqual(int(diag["screened_occurrences"]), 74434 - 116)

    def test_th_is_markup_only_and_not_screened(self):
        self.assertNotIn("th", by_unit())
        ledger = {r["unit"] for r in rows("markup_residue")}
        self.assertIn("th", ledger)
        th = [r for r in rows("markup_residue") if r["unit"] == "th"]
        self.assertTrue(all(r["base"] == "y" or r["extended"] == "y" for r in th))

    def test_tag_names_do_not_enter_as_words(self):
        # every all-markup tag name is dropped from the lexical screen
        for tag in ("p", "pre", "hr", "br", "li"):
            self.assertNotIn(tag, by_unit(), tag)
        diag = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(int(diag["markup_only_types_dropped"]), 8)

    def test_a_and_body_keep_their_real_occurrences(self):
        # `a`/`body` are real English words too; only their tag occurrences go
        u = by_unit()
        self.assertEqual(u["a"]["category"], "english")
        self.assertEqual(u["body"]["category"], "english")


class TestScreening(unittest.TestCase):
    def test_screened_type_count(self):
        self.assertEqual(len(rows("screening")), 5868)

    def test_categories_sum_to_types(self):
        cats = {}
        for r in rows("screening"):
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        self.assertEqual(sum(cats.values()), 5868)
        self.assertEqual(cats["compound"], 2216)
        self.assertEqual(cats["true_coinage"], 51)
        self.assertEqual(cats["proper_noun"], 23)
        self.assertEqual(cats["ocr_error"], 15)
        self.assertEqual(cats["numeric"], 18)
        self.assertEqual(cats["junk"], 2)

    def test_candidate_count(self):
        cands = [r for r in rows("screening") if r["is_candidate"] == "y"]
        self.assertEqual(len(cands), 2305)
        self.assertEqual(len(rows("neologisms")), 2305)

    def test_no_core_english_word_is_a_candidate(self):
        for r in rows("neologisms"):
            if r["en_status"] == "dict":
                self.assertEqual(r["category"], "ocr_error",
                                 f"{r['unit']} is dict but candidate as {r['category']}")

    def test_seam_and_open_counts(self):
        diag = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(int(diag["seam_unit_types_excl_markup"]), 1292)
        self.assertEqual(int(diag["open_juncture_types_excl_markup"]), 2046)

    def test_every_row_has_first_attestation(self):
        for r in rows("screening"):
            self.assertRegex(r["first_page"], r"^\d+$")
            self.assertRegex(r["first_start"], r"^\d+$")
            self.assertGreater(int(r["n"]), 0)
            self.assertGreaterEqual(int(r["n_pages"]), 1)


class TestExamples(unittest.TestCase):
    def test_boy_roid_is_all_english_compound(self):
        r = by_unit()["boy-roid"]
        self.assertEqual(r["category"], "compound")
        self.assertEqual(r["rule"], "R1")
        self.assertEqual(r["parts"], "boy+roid")
        self.assertEqual(r["parts_english"], "y")
        self.assertEqual((r["first_page"], r["first_start"]), ("4", "387"))

    def test_gene_tv_identity_compound(self):
        r = by_unit()["gene=TV"]
        self.assertEqual(r["category"], "compound")
        self.assertEqual(r["rule"], "R2")
        self.assertEqual(r["morph"], "gene+TV")

    def test_soul_gram_slash_compound(self):
        r = by_unit()["soul/gram"]
        self.assertEqual(r["category"], "compound")
        self.assertEqual(r["rule"], "R3")

    def test_tokage_is_proper_noun_not_split(self):
        r = by_unit()["ToKAGE"]
        self.assertEqual(r["category"], "proper_noun")
        self.assertEqual(r["n_morph"], "1")
        self.assertEqual(r["gazetteer"], "entity")

    def test_feti_honours_gazetteer(self):
        r = by_unit()["feti"]
        self.assertEqual(r["category"], "true_coinage")
        self.assertEqual(r["gazetteer"], "neologism")

    def test_hydromachine_is_a_coinage_not_ocr(self):
        # closed form dominates its rare open variant: not an OCR error
        r = by_unit()["hydromachine"]
        self.assertEqual(r["category"], "compound")
        self.assertEqual(r["ocr_rule"], "")

    def test_numeric_channel(self):
        u = by_unit()
        self.assertEqual(u["666"]["category"], "numeric")
        self.assertEqual(u["1-84068-060-1"]["category"], "numeric")
        self.assertEqual(len(rows("numeric_channel")), 18)


class TestOCR(unittest.TestCase):
    def test_ocr_error_count_matches_6(self):
        ocr = [r for r in rows("screening") if r["category"] == "ocr_error"]
        self.assertEqual(len(ocr), 15)

    def test_lt_is_a_confirmed_seed(self):
        r = by_unit()["lt"]
        self.assertEqual(r["category"], "ocr_error")
        self.assertEqual(r["ocr_correct"], "it")
        self.assertEqual(r["ocr_strength"], "confirmed")

    def test_boyroid_is_dropped_linker(self):
        r = by_unit()["boyroid"]
        self.assertEqual(r["category"], "ocr_error")
        self.assertEqual(r["ocr_correct"], "boy-roid")

    def test_cadaverfeti_is_structural_dropped_linker(self):
        r = by_unit()["cadaverfeti"]
        self.assertEqual(r["category"], "ocr_error")
        self.assertEqual(r["ocr_rule"], "dropped_linker")
        self.assertEqual(r["ocr_correct"], "cadaver-feti")

    def test_ocr_evidence_is_auditable(self):
        for r in rows("ocr_evidence"):
            self.assertTrue(r["ocr_rule"] or r["ocr_suspect"],
                            f"{r['unit']} in ocr_evidence without evidence")


class TestTSVSafety(unittest.TestCase):
    def test_column_counts_match_header(self):
        for name in ("screening", "neologisms", "neologisms_lower",
                     "ocr_evidence", "numeric_channel", "markup_residue",
                     "diagnostics", "reconciliation"):
            path = Path(_ARTIFACTS["_dir"]) / f"{name}.tsv"
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                for i, line in enumerate(fh, 2):
                    self.assertEqual(len(line.rstrip("\n").split("\t")), len(header),
                                     f"{name} row {i}")

    def test_no_tabs_in_cells(self):
        path = Path(_ARTIFACTS["_dir"]) / "screening.tsv"
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                self.assertEqual(len(line.rstrip("\n").split("\t")), len(header))


class TestPureFunctions(unittest.TestCase):
    def test_delink(self):
        self.assertEqual(nd.delink("boy-roid"), "boyroid")
        self.assertEqual(nd.delink("soul/gram"), "soulgram")
        self.assertEqual(nd.delink("bio=less=soul/gram"), "biolesssoulgram")

    def test_cell_escapes(self):
        self.assertEqual(nd.cell("a\tb\nc"), "a\\tb\\nc")

    def test_classify_dict_is_english(self):
        self.assertEqual(nd.classify("the", "dict", "R5", "atomic_seen", False, "")[0], "english")

    def test_classify_opaque_is_coinage(self):
        self.assertEqual(nd.classify("zzz", "oov", "R5", "opaque_coinage", False, "")[0], "true_coinage")

    def test_classify_open_is_compound(self):
        self.assertEqual(nd.classify("x=y", "oov", "R2", "", False, "")[0], "compound")


if __name__ == "__main__":
    unittest.main(verbosity=2)
