#!/usr/bin/env python3
"""Regression suite for #51 periodicity / spectral analysis.

Four layers:

* **Primitive tests** on synthetic series pin the spectral primitives: the
  autocorrelation of a pure sine peaks at the period, white noise has near-zero
  autocorrelation and the expected spectral flatness ``exp(-gamma)``, the
  periodogram of a sine peaks at its frequency, the interval statistics recover
  the geometric (memoryless) moments, and the Fano factor of a Bernoulli process
  is flat at large windows.
* **Corpus tests** pin the book inputs: 90,220 tokens, 11,000 operators
  (10,949 body), the lossless character reconstruction, 732 markup operators,
  and the operator stream reproduced operator-for-operator from #11.
* **Quick-run tests** run a deterministic ``--quick`` end-to-end build into a
  temp dir and check the artifacts and the headline directions: near-white
  flatness, near-Poisson intervals, no significant periodicity, and English
  punctuation more regular than the book.
* **Determinism / TSV-safety tests** re-run ``--quick`` and require every TSV to
  be byte-identical (the only per-run field, ``generated``, is excluded from the
  JSON comparison).

    python work/test_periodicity.py
"""
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import periodicity as per

_ART = {}
_ART2 = {}


def _run(out):
    rc = per.main(["--quick", "--out", str(out), "--no-plot", "--quiet"])
    assert rc == 0
    data = {}
    for name in ("series_specs", "autocorrelation", "spectrum_peaks", "spectrum",
                 "periodicity_tests", "distance_histogram", "fano",
                 "english_baseline", "reconciliation", "diagnostics"):
        path = Path(out) / f"{name}.tsv"
        data[name] = per.read_tsv(path) if path.exists() else []
    with open(Path(out) / "summary.json", encoding="utf-8") as fh:
        data["summary"] = json.load(fh)
    data["_dir"] = str(out)
    return data


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="periodicity_test_")
    _ART.update(_run(Path(tmp) / "a"))


def rows(name):
    return _ART[name]


def diag(name):
    return {r["diag"]: r["value"] for r in rows("diagnostics")}[name]


def test_row(series):
    for r in rows("periodicity_tests"):
        if r["series"] == series:
            return r
    raise KeyError(series)


class TestPrimitives(unittest.TestCase):
    def test_autocorr_sine_peaks_at_period(self):
        x = np.sin(2 * np.pi * np.arange(2000) / 10.0)
        ac = per.autocorr(x, 40)
        self.assertAlmostEqual(ac[10], 1.0, delta=0.02)
        self.assertAlmostEqual(ac[5], -1.0, delta=0.02)
        self.assertAlmostEqual(ac[0], 1.0, places=9)

    def test_autocorr_white_noise_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(50000)
        ac = per.autocorr(x, 50)
        self.assertLess(float(np.max(np.abs(ac[1:]))), 0.05)

    def test_periodogram_sine_peak(self):
        x = np.sin(2 * np.pi * np.arange(4096) / 8.0)
        f, P = per.periodogram(x)
        k = int(np.argmax(P[1:])) + 1
        self.assertAlmostEqual(float(f[k]), 1.0 / 8.0, places=4)

    def test_spectral_flatness_white_noise_anchor(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(200000)
        _f, P = per.periodogram(x)
        sf = per.spectral_flatness(P[1:])
        self.assertAlmostEqual(sf, per.WHITE_FLATNESS, delta=0.05)

    def test_spectral_flatness_sine_is_tonal(self):
        x = np.sin(2 * np.pi * np.arange(4096) / 8.0)
        _f, P = per.periodogram(x)
        self.assertLess(per.spectral_flatness(P[1:]), 0.05)

    def test_intervals(self):
        x = np.zeros(20)
        x[[2, 5, 9]] = 1.0
        np.testing.assert_array_equal(per.intervals(x), [3.0, 4.0])

    def test_renewal_stats_geometric(self):
        rng = np.random.default_rng(2)
        g = rng.geometric(0.2, size=200000).astype(float)
        st = per.renewal_stats(g)
        # mean 5; memoryless CV = sqrt((mean-1)/mean) ~ 0.894; geo ratio ~ 1
        self.assertAlmostEqual(st["interval_mean"], 5.0, delta=0.05)
        self.assertAlmostEqual(st["geometric_var_ratio"], 1.0, delta=0.05)
        self.assertAlmostEqual(st["interval_lag1"], 0.0, delta=0.02)

    def test_renewal_stats_regular_is_sub_poisson(self):
        # a perfectly regular gap series has CV 0 and geo ratio 0
        g = np.full(1000, 4.0)
        st = per.renewal_stats(g)
        self.assertAlmostEqual(st["interval_cv"], 0.0, places=9)
        self.assertAlmostEqual(st["geometric_var_ratio"], 0.0, places=9)

    def test_fano_bernoulli_flattens(self):
        rng = np.random.default_rng(3)
        x = (rng.random(200000) < 0.1).astype(float)
        f1 = per.fano_curve(x, windows=(1, 1000))
        self.assertAlmostEqual(f1[0][1], 0.9, delta=0.02)     # 1 - p
        self.assertAlmostEqual(f1[1][1], 1.0, delta=0.1)

    def test_peak_stats_synthetic_period(self):
        # a square wave of period 17 has its fundamental at 1/17
        x = (np.sin(2 * np.pi * np.arange(4000) / 17.0) > 0).astype(float)
        f, P = per.periodogram(x)
        rel, period, _med = per.peak_stats(f, P, (5.0, 1000.0))
        self.assertGreater(rel, 20.0)
        self.assertAlmostEqual(period, 17.0, delta=2.0)

    def test_block_shuffle_preserves_multiset(self):
        x = np.arange(4096, dtype=float)
        rng = np.random.default_rng(4)
        y = per.block_shuffle(x, 512, rng)
        np.testing.assert_array_equal(np.sort(x), np.sort(y))
        self.assertFalse(np.array_equal(x, y))

    def test_global_shuffle_preserves_multiset(self):
        x = np.array([0.0] * 90 + [1.0] * 10)
        rng = np.random.default_rng(5)
        y = per.global_shuffle(x, rng)
        np.testing.assert_array_equal(np.sort(x), np.sort(y))

    def test_p_value_bounds(self):
        null = np.arange(100, dtype=float)
        self.assertLessEqual(per.p_value(-1.0, null), 1.0)
        self.assertGreater(per.p_value(1e9, null), 0.0)
        self.assertAlmostEqual(per.p_value(50.0, null), (1 + 50) / 101, places=9)

    def test_detrend_removes_linear_trend(self):
        x = np.linspace(0, 10, 4000)
        d = per.detrend(x, 201)
        self.assertLess(float(np.abs(d[500:-500]).max()), 0.2)

    def test_distance_histogram_expected_totals(self):
        rng = np.random.default_rng(6)
        g = rng.geometric(0.25, size=10000).astype(float)
        hist = per.distance_histogram(g, max_d=30)
        obs = sum(r["observed"] for r in hist)
        exp = sum(r["expected_geometric"] for r in hist)
        self.assertEqual(obs, len(g))
        self.assertAlmostEqual(exp, len(g), delta=5.0)

    def test_band_mask(self):
        f = np.fft.rfftfreq(1000)
        m = per.band_mask(f, (10.0, 100.0))
        self.assertTrue(m.any())
        self.assertFalse(m[0])   # DC excluded


class TestCorpus(unittest.TestCase):
    def test_token_counts(self):
        rows_ = per.load_token_rows()
        self.assertEqual(len(rows_), per.BOOK_TOKENS)
        self.assertEqual(sum(1 for r in rows_ if per.is_op(r)), per.BOOK_OPS_ALL)
        self.assertEqual(per._body_op_count(rows_), per.BOOK_OPS_BODY)

    def test_operator_stream_matches_11(self):
        rows_ = per.load_token_rows()
        rebuilt = per._op_stream_from_tokens(rows_)
        from_file = defaultdict(list)
        for r in per.read_tsv(per.OP_STREAM):
            from_file[int(r["page"])].append(r["operator"])
        self.assertEqual(set(from_file), set(rebuilt))
        for page in from_file:
            self.assertEqual(from_file[page], rebuilt[page])

    def test_char_reconstruction(self):
        rows_ = per.load_token_rows()
        char_op = per.reconstruct_char_binary(rows_, per.is_op)
        self.assertEqual(len(char_op), 480267)
        self.assertEqual(int(char_op.sum()), 17269)

    def test_markup_ops(self):
        bodies = per.coll.load_bodies()
        rows_ = per.load_token_rows()
        flags = per.markup_op_flags(rows_, bodies)
        self.assertEqual(int(flags.sum()), 732)
        heavy = per._heavy_markup_pages(rows_, flags, threshold=20)
        self.assertEqual(heavy, {92, 183, 209})

    def test_clause_lengths(self):
        cl = per.clause_lengths()
        self.assertEqual(len(cl), per.BOOK_CLAUSES)
        self.assertAlmostEqual(float(cl.mean()), 13.80, delta=0.1)


class TestQuickRun(unittest.TestCase):
    def test_artifacts_present(self):
        for name in ("series_specs", "autocorrelation", "spectrum_peaks",
                     "spectrum", "periodicity_tests", "distance_histogram",
                     "fano", "english_baseline", "reconciliation", "diagnostics"):
            self.assertTrue(rows(name), name)

    def test_pages_used(self):
        self.assertEqual(int(diag("pages_used")), 40)

    def test_seed(self):
        self.assertEqual(int(diag("seed")), per.SEED)

    def test_flatness_near_white_anchor(self):
        for s in ("tok_op", "tok_op_screened", "tok_op_nomarkuppages"):
            flat = float(test_row(s)["spectral_flatness"])
            self.assertAlmostEqual(flat, per.WHITE_FLATNESS, delta=0.06, msg=s)

    def test_intervals_near_poisson(self):
        r = test_row("tok_op")
        cv = float(r["interval_cv"])
        # memoryless CV is sqrt((mean-1)/mean) ~ 0.93 for mean ~ 8; allow burstiness
        self.assertGreater(cv, 0.6)
        self.assertLess(cv, 1.6)
        self.assertGreater(float(r["geometric_var_ratio"]), 0.4)
        self.assertLess(float(r["geometric_var_ratio"]), 2.5)

    def test_no_significant_periodicity(self):
        # the headline: the operator stream has no cycle surviving the block null
        for s in ("tok_op", "tok_op_screened", "tok_op_nomarkuppages"):
            p_block = float(test_row(s)["p_block_peak"])
            self.assertGreater(p_block, 0.05, s)

    def test_english_more_regular_than_book(self):
        book_cv = float(test_row("tok_op")["interval_cv"])
        en_cv = float(test_row("en_tok_punct")["interval_cv"])
        self.assertLess(en_cv, book_cv)

    def test_spectrum_peaks_present(self):
        peaks = defaultdict(int)
        for r in rows("spectrum_peaks"):
            peaks[r["series"]] += 1
        self.assertGreaterEqual(peaks["tok_op"], 5)
        for r in rows("spectrum_peaks"):
            self.assertGreater(float(r["period"]), 0.0)

    def test_distance_histogram_present(self):
        series = {r["series"] for r in rows("distance_histogram")}
        self.assertIn("gap_tok_op", series)
        self.assertIn("en_gap_punct", series)

    def test_fano_growth(self):
        f = {int(r["window"]): float(r["fano"])
             for r in rows("fano") if r["series"] == "tok_op"}
        self.assertLess(f[1], 1.0)          # 1 - p for a binary series
        self.assertLess(f[16], f[256])      # growth = clustering/drift
        self.assertGreater(f[1024], 1.0)

    def test_reconciliation_present(self):
        recon = {r["quantity"]: r["here"] for r in rows("reconciliation")}
        self.assertIn("operator tokens (#17 OP)", recon)
        self.assertIn("white-noise flatness anchor", recon)
        self.assertIn("markup OP tokens screened", recon)

    def test_summary_contract(self):
        s = _ART["summary"]
        self.assertEqual(s["schema"], per.SCHEMA)
        self.assertEqual(s["seed"], per.SEED)
        self.assertIn("tok_op", s["book"])
        self.assertAlmostEqual(s["white_flatness_anchor"], per.WHITE_FLATNESS,
                               places=6)
        self.assertIn("gap_tok_op", s["gaps"])

    def test_series_specs_cover_primary(self):
        series = {r["series"] for r in rows("series_specs")}
        for s in ("tok_op", "tok_op_screened", "tok_op_nomarkuppages",
                  "tok_punct", "char_op", "char_punct", "gap_tok_op",
                  "gap_char_op", "en_tok_punct", "en_gap_punct"):
            self.assertIn(s, series)


class TestDeterminismAndSafety(unittest.TestCase):
    def test_tsvs_byte_identical_on_rerun(self):
        tmp = tempfile.mkdtemp(prefix="periodicity_det_")
        second = _run(Path(tmp) / "b")
        for name in ("series_specs", "autocorrelation", "spectrum_peaks",
                     "spectrum", "periodicity_tests", "distance_histogram",
                     "fano", "english_baseline", "reconciliation", "diagnostics"):
            a = Path(_ART["_dir"]) / f"{name}.tsv"
            b = Path(second["_dir"]) / f"{name}.tsv"
            self.assertEqual(a.read_bytes(), b.read_bytes(), name)

    def test_summary_deterministic_except_generated(self):
        tmp = tempfile.mkdtemp(prefix="periodicity_det2_")
        second = _run(Path(tmp) / "c")
        a = dict(_ART["summary"])
        b = dict(second["summary"])
        a.pop("generated", None)
        b.pop("generated", None)
        self.assertEqual(a, b)

    def test_column_counts_match_header(self):
        for name in ("series_specs", "autocorrelation", "spectrum_peaks",
                     "spectrum", "periodicity_tests", "distance_histogram",
                     "fano", "english_baseline", "reconciliation", "diagnostics"):
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
