#!/usr/bin/env python3
"""Regression suite for #24 frequency / Zipf / type-token analysis.

Runs the analysis once into a temp directory (using the pinned reference
cache if present, otherwise the book-only path), then checks the invariants
that matter: the unit-layer join and markup screen, the type/token and hapax
reconciliation, the Zipf/Heaps estimators on synthetic data, the English
baseline table, and TSV safety.

    python work/test_frequency_zipf.py
"""
import csv
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import frequency_zipf as fz

_ART = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="frequency_test_")
    fz.main(["--out", tmp, "--reference-dir", str(fz.REF_CACHE), "--quiet"])
    for name in ("fits", "book_zipf", "category_breakdown", "hapax_ledger",
                 "english_baseline", "reference_manifest", "reconciliation",
                 "diagnostics"):
        path = Path(tmp) / f"{name}.tsv"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                _ART[name] = list(csv.DictReader(fh, delimiter="\t"))
        else:
            _ART[name] = []
    _ART["_dir"] = tmp


def rows(name):
    return _ART[name]


def fits():
    return {r["corpus"]: r for r in rows("fits")}


class TestEstimators(unittest.TestCase):
    def test_ols(self):
        slope, intercept = fz.ols([1, 2, 3], [2, 4, 6])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 0.0)

    def test_counter_stats(self):
        c = Counter({"a": 4, "b": 2, "c": 1, "d": 1})
        s = fz.counter_stats(c)
        self.assertEqual(s["N"], 8)
        self.assertEqual(s["V"], 4)
        self.assertEqual(s["V1"], 2)
        self.assertAlmostEqual(s["ttr"], 0.5)
        self.assertAlmostEqual(s["hapax_ratio"], 0.5)
        self.assertAlmostEqual(s["top1_share"], 0.5)

    def test_mattr_bounds(self):
        toks = list("abcabcabc")
        self.assertAlmostEqual(fz.mattr(toks, 3), 1.0)  # window 3 has 3 types
        self.assertGreater(fz.mattr(toks, 100), 0.0)

    def test_zipf_slope_on_exact_power_law(self):
        freqs = [round(100000 / r) for r in range(1, 2000)]
        fit = fz.fit_zipf(freqs)
        self.assertAlmostEqual(fit["zipf_slope_all"], -1.0, delta=0.15)

    def test_mle_recovers_pareto_alpha(self):
        # inverse-transform sample of a continuous Pareto with alpha = 2.5
        import random
        rng = random.Random(7)
        alpha, xmin = 2.5, 10.0
        xs = [xmin * (1 - rng.random()) ** (-1 / (alpha - 1)) for _ in range(4000)]
        freqs = [max(1, round(x)) for x in xs]
        fit = fz.fit_zipf(freqs)
        self.assertAlmostEqual(fit["mle_alpha"], alpha, delta=0.3)

    def test_heaps_monotone(self):
        toks = [f"t{i % 500}" for i in range(5000)]
        h = fz.heaps_fit(toks)
        self.assertGreater(h["heaps_beta"], 0.0)
        vs = [v for _, v in h["heaps_points"]]
        self.assertEqual(vs, sorted(vs))


class TestText(unittest.TestCase):
    def test_tokenize_english(self):
        self.assertEqual(fz.tokenize_english("The cat's mat, Don't!"),
                         ["the", "cat's", "mat", "don't"])

    def test_strip_gutenberg(self):
        t = "header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\ntail"
        self.assertEqual(fz.strip_gutenberg(t).strip(), "body")

    def test_cell_escapes(self):
        self.assertEqual(fz.cell("a\tb\nc"), "a\\tb\\nc")

    def test_unescape_cell_roundtrips_19_convention(self):
        # #19 wrote a literal backslash-n as \\n; unescaping once recovers
        # #17's convention (a literal backslash + n)
        self.assertEqual(fz.unescape_cell("device++\\\\nmass"), "device++\\nmass")
        self.assertEqual(fz.unescape_cell("a\\tb"), "a\tb")


class TestBookReconciliation(unittest.TestCase):
    def test_join_contract_holds(self):
        # load_book asserts the uid join reproduces #19's screen; if the
        # invariant were broken it would SystemExit, so reaching here is the
        # test.  Also check the counts it produces.
        units, parts, cats = fz.load_book()
        self.assertEqual(len(units), 74318)
        self.assertEqual(len(parts), 80015)
        self.assertEqual(len(cats), 5868)

    def test_unit_counts(self):
        f = fits()["book_units"]
        self.assertEqual(int(f["N"]), 74318)
        self.assertEqual(int(f["V"]), 5868)
        self.assertEqual(int(f["V1"]), 3543)

    def test_zipf_rows_match_types(self):
        zr = rows("book_zipf")
        self.assertEqual(len(zr), 5868)
        self.assertEqual(sum(int(r["n"]) for r in zr), 74318)
        self.assertEqual([int(r["rank"]) for r in zr], list(range(1, 5869)))
        # ranks are sorted by descending frequency
        ns = [int(r["n"]) for r in zr]
        self.assertEqual(ns, sorted(ns, reverse=True))

    def test_markup_types_absent(self):
        present = {r["unit"] for r in rows("book_zipf")}
        for tag in ("th", "p", "pre", "br", "li", "hr", "Pre", "fluid=streaming"):
            self.assertNotIn(tag, present, tag)

    def test_category_breakdown_sums(self):
        cb = rows("category_breakdown")
        self.assertEqual(sum(int(r["types"]) for r in cb), 5868)
        self.assertEqual(sum(int(r["occ"]) for r in cb), 74318)
        cats = {r["category"]: r for r in cb}
        self.assertEqual(int(cats["english"]["occ"]), 66506)
        self.assertEqual(int(cats["compound"]["types"]), 2216)
        self.assertEqual(int(cats["ocr_error"]["types"]), 15)

    def test_hapax_ledger(self):
        hl = rows("hapax_ledger")
        self.assertEqual(len(hl), 3543)
        units = [r["unit"] for r in hl]
        self.assertEqual(len(set(units)), len(units))  # one row per type

    def test_diagnostics_contract(self):
        d = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(int(d["join_mismatches"]), 0)
        self.assertEqual(int(d["markup_uids_removed"]), 347)
        self.assertEqual(int(d["screened_occurrences"]), 74318)
        self.assertGreater(float(d["coinage_repeat_token_share"]), 0.5)

    def test_subsets_partition(self):
        f = fits()
        self.assertEqual(int(f["book_english"]["N"]) + int(f["book_nonenglish"]["N"]),
                         int(f["book_units"]["N"]))
        self.assertLessEqual(int(f["book_coinage"]["N"]), int(f["book_nonenglish"]["N"]))

    def test_english_subset_is_more_concentrated(self):
        f = fits()
        self.assertGreater(float(f["book_english"]["mattr_5000"]),
                           0.0)
        self.assertLess(float(f["book_english"]["mattr_5000"]),
                        float(f["book_nonenglish"]["mattr_5000"]))


class TestEnglishBaseline(unittest.TestCase):
    def test_baseline_present(self):
        if not rows("english_baseline"):
            self.skipTest("reference cache unavailable")
        eb = rows("english_baseline")
        modes = {(r["text"], r["mode"]) for r in eb}
        self.assertEqual(len({t for t, _ in modes}), 8)
        self.assertEqual({m for _, m in modes}, {"full", "matched"})

    def test_manifest_pinned(self):
        if not rows("reference_manifest"):
            self.skipTest("reference cache unavailable")
        self.assertEqual(len(rows("reference_manifest")), 8)
        for r in rows("reference_manifest"):
            self.assertEqual(len(r["sha256"]), 64)
            self.assertGreater(int(r["tokens_full"]), 30000)

    def test_english_matched_n(self):
        if not rows("english_baseline"):
            self.skipTest("reference cache unavailable")
        for r in rows("english_baseline"):
            if r["mode"] == "matched" and r["text"] != "heart_of_darkness":
                self.assertEqual(int(r["N"]), fz.BOOK_N)


class TestTSVSafety(unittest.TestCase):
    def test_column_counts_match_header(self):
        for name in ("fits", "book_zipf", "category_breakdown", "hapax_ledger",
                     "english_baseline", "reference_manifest", "reconciliation",
                     "diagnostics"):
            path = Path(_ART["_dir"]) / f"{name}.tsv"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                for i, line in enumerate(fh, 2):
                    self.assertEqual(len(line.rstrip("\n").split("\t")), len(header),
                                     f"{name} row {i}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
