#!/usr/bin/env python3
"""Regression suite for #35 n-gram LM / per-token surprisal.

Two layers:

* **Primitive tests** on synthetic streams verify the probability model itself:
  the leave-one-page-out count subtraction, that the smoothed conditional
  distribution sums to 1, that a deterministic bigram costs ~0 bits, and the
  statistics helpers.
* **Corpus tests** verify the markup screen (74,077 units = #17's 74,665 minus
  #19's 347 and #27's 241 tag-furniture occurrences), the #14 clause join, and a
  deterministic ``--quick`` end-to-end run whose artifacts are structurally
  sound and TSV-safe.

    python work/test_ngram_surprisal.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ngram_surprisal as ns

_ART = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="ngram_surprisal_test_")
    rc = ns.main(["--quick", "--out", tmp, "--no-plot", "--quiet"])
    assert rc == 0
    for name in ("per_token", "per_page", "window_curve", "zones",
                 "type_surprisal", "hot_occurrences", "cold_occurrences",
                 "category_surprisal", "freq_band_surprisal",
                 "clause_surprisal", "perplexity_by_order", "reconciliation",
                 "diagnostics"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = ns.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["_dir"] = tmp


def rows(name):
    return _ART[name]


class TestPrimitives(unittest.TestCase):
    def test_percentile(self):
        self.assertAlmostEqual(ns.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(ns.percentile([5], 0.9), 5.0)
        self.assertEqual(ns.percentile([], 0.5), 0.0)

    def test_mean_sd(self):
        m, sd = ns.mean_sd([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(m, 5.0)
        self.assertAlmostEqual(sd, 2.0)

    def test_pearson_and_spearman(self):
        xs = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(ns.pearson(xs, [2, 4, 6, 8, 10]), 1.0)
        self.assertAlmostEqual(ns.pearson(xs, [10, 8, 6, 4, 2]), -1.0)
        # monotone but non-linear -> Spearman is exactly 1
        self.assertAlmostEqual(ns.spearman(xs, [1, 4, 9, 16, 25]), 1.0)

    def test_rankdata_ties(self):
        self.assertEqual(ns.rankdata([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_freq_band(self):
        self.assertEqual(ns.freq_band(1), "1 (hapax)")
        self.assertEqual(ns.freq_band(3), "2-3")
        self.assertEqual(ns.freq_band(5), "4-5")
        self.assertEqual(ns.freq_band(1000), "201+")

    def test_zone_runs(self):
        items = [{"k": i, "f": i in (1, 2, 4)} for i in range(6)]
        runs = ns.zone_runs(items, lambda r: r["f"], lambda r: r["k"])
        self.assertEqual(runs, [(1, 2), (4, 4)])


class TestStoreFold(unittest.TestCase):
    def _stores(self):
        stream = ["a", "b", "c", "a", "b", "c", "a", "b"]
        g = ns.Store(2)
        g.add(stream, 2)
        p = ns.Store(2)
        p.add(["a", "b"], 2)
        return g, p, ns.Fold(g, p)

    def test_subtraction(self):
        g, _p, fold = self._stores()
        # b occurs 3x globally, 1x on the held-out page
        self.assertEqual(g.uni["b"], 3)
        self.assertEqual(fold.uni_count("b"), 2)
        # (a,b) bigram occurs 3x globally, 1x on the page
        self.assertEqual(g.counts[1][("a",)]["b"], 3)
        self.assertEqual(fold.counter(1, ("a",))["b"], 2)
        self.assertEqual(fold.total(1, ("a",)), 2)
        self.assertEqual(fold.n1p(1, ("a",)), 1)

    def test_conditional_sums_to_one(self):
        g, p, fold = self._stores()
        V = len(g.uni)
        ntrain = g.n - p.n
        total = 0.0
        for w in g.uni:
            total += 2 ** (-ns.token_surprisal(fold, ["a"], w, 1, V, ntrain))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_deterministic_bigram_is_cheap(self):
        # a stream where b always follows a: bigram surprisal << unigram
        pages = {"1": ["a", "b"] * 60, "2": ["a", "b"] * 60}
        res = ns.analyze_stream(pages, contexts=(0, 1))
        mean_uni = res["sums"][0] / res["ns"][0]
        mean_bi = res["sums"][1] / res["ns"][1]
        self.assertLess(mean_bi, 1.0)
        self.assertLess(mean_bi, mean_uni)

    def test_analyze_stream_is_deterministic(self):
        pages = {"1": ["x", "y", "z", "x", "y"], "2": ["z", "x", "y", "z"]}
        a = ns.analyze_stream(pages, contexts=(0, 1, 2))
        b = ns.analyze_stream(pages, contexts=(0, 1, 2))
        self.assertEqual(a["sums"], b["sums"])
        self.assertEqual(a["ns"], b["ns"])

    def test_higher_order_helps_on_structured_stream(self):
        # a periodic stream has strong higher-order structure
        pages = {str(i): ["p", "q", "r", "s"] * 20 for i in range(1, 9)}
        res = ns.analyze_stream(pages, contexts=(0, 1, 2, 3))
        perpl = {k: 2 ** (res["sums"][k] / res["ns"][k]) for k in res["sums"]}
        self.assertLess(perpl[3], perpl[0])


class TestCorpus(unittest.TestCase):
    def test_full_screen_counts(self):
        page_tokens, page_rows, _cats, pages = ns.load_page_streams(True, True)
        self.assertEqual(len(pages), 225)
        self.assertEqual(sum(len(v) for v in page_tokens.values()), 74077)
        self.assertEqual(sum(len(v) for v in page_rows.values()), 74077)

    def test_ledger_only_screen_counts(self):
        page_tokens, _rows, _cats, _pages = ns.load_page_streams(True, False)
        self.assertEqual(sum(len(v) for v in page_tokens.values()), 74318)

    def test_tag_furniture_screened(self):
        page_tokens, _rows, _cats, _pages = ns.load_page_streams(True, True)
        present = {t for toks in page_tokens.values() for t in toks}
        for tag in ("href", "mailto", "colspan"):
            self.assertNotIn(tag, present, tag)

    def test_case_sensitivity(self):
        folded, _r1, _c1, _p1 = ns.load_page_streams(True, True)
        cased, _r2, _c2, _p2 = ns.load_page_streams(False, True)
        f = {t for toks in folded.values() for t in toks}
        c = {t for toks in cased.values() for t in toks}
        self.assertTrue(any(t.isupper() for t in c))
        self.assertGreaterEqual(len(c), len(f))


class TestQuickRun(unittest.TestCase):
    def test_artifacts_present(self):
        for name in ("per_token", "per_page", "window_curve", "zones",
                     "type_surprisal", "hot_occurrences", "cold_occurrences",
                     "category_surprisal", "freq_band_surprisal",
                     "clause_surprisal", "perplexity_by_order", "reconciliation",
                     "diagnostics"):
            self.assertTrue(rows(name), name)

    def test_per_token_rows_match_diagnostics(self):
        d = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(len(rows("per_token")), int(d["units_scored"]))

    def test_per_token_columns_and_join(self):
        pt = rows("per_token")
        self.assertEqual(list(pt[0].keys()),
                         ["index", "uid", "page", "start", "token", "category",
                          "context", "surprisal_bits", "unigram_bits"])
        # every row carries a uid so it can be joined back to #17
        self.assertTrue(all(r["uid"] for r in pt))
        # surprisal is finite and non-negative
        for r in pt[:200]:
            self.assertGreaterEqual(float(r["surprisal_bits"]), 0.0)

    def test_per_page_zones(self):
        pp = rows("per_page")
        self.assertEqual([int(r["page"]) for r in pp],
                         sorted(int(r["page"]) for r in pp))
        for r in pp:
            self.assertIn(r["zone"], ("", "hot", "cold"))
            if r["zone"]:
                self.assertGreaterEqual(int(r["n_tokens"]), ns.MIN_PAGE_TOKENS)
        zones = {r["zone"] for r in rows("zones")}
        self.assertTrue(zones)
        self.assertTrue(zones <= {"hot", "cold"})

    def test_freq_band_monotone(self):
        bands = rows("freq_band_surprisal")
        means = [float(r["mean_surprisal"]) for r in bands]
        # the rarest band must be the most surprising and the commonest cheapest
        self.assertGreater(means[0], means[-1])

    def test_english_categories_cheaper_than_coinage(self):
        cats = {r["category"]: r for r in rows("category_surprisal")}
        if "english" in cats and "compound" in cats:
            self.assertLess(float(cats["english"]["mean_surprisal"]),
                            float(cats["compound"]["mean_surprisal"]))

    def test_summary_contract(self):
        s = _ART["summary"]
        self.assertEqual(s["schema"], ns.SCHEMA)
        self.assertEqual(s["n_units"], int(
            {r["diag"]: r["value"] for r in rows("diagnostics")}["units_scored"]))
        self.assertIn("hapax_share", s["page_driver_correlation"])
        self.assertIn("perplexity_by_order", s)

    def test_clause_join_has_matches(self):
        cl = rows("clause_surprisal")
        matched = [r for r in cl if int(r["n_units_matched"]) > 0]
        self.assertGreater(len(matched), 0)
        # matched units are a subset of the scored stream
        d = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertLessEqual(sum(int(r["n_units_matched"]) for r in matched),
                             int(d["units_scored"]))

    def test_perplexity_by_order_includes_sensitivity(self):
        models = {r["model"] for r in rows("perplexity_by_order")}
        self.assertTrue(any(m.startswith("order ") for m in models))
        self.assertTrue(any(m.startswith("case-preserved") for m in models))


class TestTSVSafety(unittest.TestCase):
    def test_column_counts_match_header(self):
        names = ("per_token", "per_page", "window_curve", "zones",
                 "type_surprisal", "hot_occurrences", "cold_occurrences",
                 "category_surprisal", "freq_band_surprisal",
                 "clause_surprisal", "perplexity_by_order", "reconciliation",
                 "diagnostics")
        for name in names:
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
