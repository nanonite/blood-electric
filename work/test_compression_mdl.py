#!/usr/bin/env python3
"""Regression suite for #48 compression / MDL vs English baseline.

Three layers:

* **Primitive tests** verify the compressors are deterministic, that
  ``bits_per_symbol`` and ``ratio`` are the arithmetic they claim to be, and
  that the stream builders return the expected shapes.
* **Corpus tests** verify the book streams (74,077 tokens; char totals) and the
  #13 variant reader strips layout page markers.
* **Quick-run tests** run a deterministic ``--quick`` end-to-end build into a
  temp dir and check the artifacts, the headline direction (book bits/symbol
  below English for every pair), the MDL split, and TSV safety.

    python work/test_compression_mdl.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compression_mdl as cm

_ART = {}
_STR = b"the quick brown fox jumps over the lazy dog " * 40


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="compression_test_")
    rc = cm.main(["--quick", "--out", tmp, "--no-plot", "--quiet"])
    assert rc == 0
    for name in ("compression_by_stream", "english_baseline", "mdl_split",
                 "silence_variants", "per_page", "reconciliation", "diagnostics"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = cm.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["_dir"] = tmp


def rows(name):
    return _ART[name]


def diag(name):
    return {r["diag"]: r["value"] for r in rows("diagnostics")}[name]


class TestPrimitives(unittest.TestCase):
    def test_compressors_deterministic(self):
        for name, level in cm.COMPRESSORS:
            a = cm.compress(name, level, _STR)
            b = cm.compress(name, level, _STR)
            if a is None:
                continue
            self.assertEqual(a, b, f"{name}-{level} not deterministic")

    def test_compressors_shrink(self):
        for name, level in cm.COMPRESSORS:
            c = cm.compress(name, level, _STR)
            if c is None:
                continue
            self.assertLess(len(c), len(_STR), name)

    def test_bpc_formula(self):
        text = "abcabcabc"
        comp = cm.compress("gzip", 9, text.encode())
        self.assertAlmostEqual(cm.bpc(text, comp, len(text)),
                               8.0 * len(comp) / len(text), places=9)

    def test_ratio_formula(self):
        text = "abcabcabc"
        comp = cm.compress("gzip", 9, text.encode())
        self.assertAlmostEqual(cm.ratio(text, comp),
                               len(comp) / len(text.encode()), places=9)

    def test_cname(self):
        self.assertEqual(cm.cname("zstd", 19), "zstd-19")

    def test_available_compressors_nonempty(self):
        self.assertTrue(cm.available_compressors())


class TestCorpus(unittest.TestCase):
    def test_book_streams(self):
        book = cm.book_streams()
        self.assertEqual(book["token_unit"][2], 74077)
        self.assertGreater(book["char_raw"][2], 400000)
        self.assertEqual(book["char_raw"][1], "char")
        # letters are a subset of raw
        self.assertLess(book["char_letters"][2], book["char_raw"][2])
        self.assertTrue(set(book["char_letters"][0]) <= set("abcdefghijklmnopqrstuvwxyz"))

    def test_silence_variant_reader_strips_markers(self):
        text = cm.silence_variant_text("v0_original")
        self.assertIsNotNone(text)
        self.assertNotIn("<!--", text)
        self.assertNotIn("#", text[:50])

    def test_zstd_backend_reported(self):
        self.assertIn(diag("zstd_backend"), ("package", "cli", "none"))


class TestQuickRun(unittest.TestCase):
    def test_artifacts_present(self):
        for name in ("compression_by_stream", "english_baseline", "mdl_split",
                     "silence_variants", "per_page", "reconciliation",
                     "diagnostics"):
            self.assertTrue(rows(name), name)

    def test_book_more_compressible_every_pair(self):
        pairs = {}
        for r in rows("compression_by_stream"):
            pairs.setdefault((r["stream"], r["compressor"]), {})[r["corpus"]] = \
                float(r["bits_per_symbol"])
        n = 0
        for key, d in pairs.items():
            if "book" in d and "english_pooled" in d:
                self.assertLess(d["book"], d["english_pooled"], key)
                n += 1
        self.assertGreater(n, 0)

    def test_mdl_split_book_lower(self):
        m = {}
        for r in rows("mdl_split"):
            m.setdefault((r["stream"], r["compressor"]), {})[r["corpus"]] = \
                float(r["incremental_bits_per_symbol"])
        checked = 0
        for key, d in m.items():
            if "book" in d and "english_pooled" in d:
                self.assertLess(d["book"], d["english_pooled"], key)
                checked += 1
        self.assertGreater(checked, 0)

    def test_silence_variants_include_baseline(self):
        variants = {r["variant"] for r in rows("silence_variants")}
        self.assertIn("v0_original", variants)
        self.assertIn("v1_delete_all_operators", variants)

    def test_per_page_rows(self):
        pp = rows("per_page")
        self.assertEqual(len(pp), 225)
        self.assertIn("gzip-9", pp[0])
        self.assertTrue(all(float(r["gzip-9"]) > 0 for r in pp))

    def test_english_baseline_eight_refs(self):
        refs = {r["reference"] for r in rows("english_baseline")}
        self.assertEqual(len(refs), 8)

    def test_summary_contract(self):
        s = _ART["summary"]
        self.assertEqual(s["schema"], cm.SCHEMA)
        self.assertTrue(s["verdict_headline"])
        for key, d in s["verdict_headline"].items():
            self.assertTrue(d["book_lower"], key)

    def test_reconciliation_counts(self):
        recon = {r["quantity"]: r["here"] for r in rows("reconciliation")}
        self.assertEqual(recon["English references"], "8")


class TestTSVSafety(unittest.TestCase):
    def test_column_counts_match_header(self):
        for name in ("compression_by_stream", "english_baseline", "mdl_split",
                     "silence_variants", "per_page", "reconciliation",
                     "diagnostics"):
            path = Path(_ART["_dir"]) / f"{name}.tsv"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                for i, line in enumerate(fh, 2):
                    self.assertEqual(len(line.rstrip("\n").split("\t")),
                                     len(header), f"{name} row {i}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
