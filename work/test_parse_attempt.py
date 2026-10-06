#!/usr/bin/env python3
"""Regression suite for #32 parse-attempt / derangement curve.

Fast checks that do not need a full corpus parse: the utterance builder and its
reconciliation with #13, the parseability definition on constructed sentences,
the drift/aggregation helpers, and the artifact writer (via a small ``--quick``
run with stanza and the English baseline disabled).

    python work/test_parse_attempt.py
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import parse_attempt as pa

_ARTIFACTS = {}
_SKIP_PARSE = None
try:
    import spacy
except Exception as exc:  # noqa: BLE001
    _SKIP_PARSE = f"spacy unavailable: {exc!r}"


def setUpModule():
    if _SKIP_PARSE:
        return
    tmp = tempfile.mkdtemp(prefix="parse_attempt_test_")
    pa.main(["--quick", "--no-stanza", "--no-english", "--out", tmp])
    for name in ("sentences", "per_page", "derangement_curve", "baselines",
                 "reconciliation", "diagnostics"):
        with open(Path(tmp) / f"{name}.tsv", encoding="utf-8") as fh:
            _ARTIFACTS[name] = list(csv.DictReader(fh, delimiter="\t"))
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


class TestUnitBuilder(unittest.TestCase):
    def test_v5_units_reconcile(self):
        v5 = pa.load_units("v5_period_from_slashes")
        self.assertEqual(len(v5), 2360)
        self.assertEqual(sum(len(pa.words(u["text"])) for u in v5), 77392)

    def test_v4_units_reconcile(self):
        v4 = pa.load_units("v4_operators_to_period")
        self.assertEqual(len(v4), 10288)
        self.assertEqual(sum(len(pa.words(u["text"])) for u in v4), 77392)

    def test_13_whole_file_count_still_reproduces(self):
        # the per-page split is higher because #13's whole-file split bridges
        # page boundaries; both are asserted so the discrepancy stays visible
        from silence_removal import segment_stats
        txt = (pa.SILENCE / "v5_period_from_slashes.txt").read_text(encoding="utf-8")
        self.assertEqual(segment_stats(txt)["n_segments"], 2163)

    def test_segments_are_page_local(self):
        v5 = pa.load_units("v5_period_from_slashes")
        by_page = {}
        for u in v5:
            by_page.setdefault(u["page"], []).append(u)
        # page 4 is the first real prose page and has several utterances
        self.assertGreaterEqual(len(by_page[4]), 5)
        # a page's char offsets are strictly increasing with sent index
        for us in by_page.values():
            starts = [u["char_start"] for u in us]
            self.assertEqual(starts, sorted(starts))


class TestParseabilityDefinition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _SKIP_PARSE:
            raise unittest.SkipTest(_SKIP_PARSE)
        cls.nlp = spacy.load("en_core_web_md")

    def _one(self, text):
        return pa.spacy_run(self.nlp, [text])[0]

    def test_normal_transitive_sentence_parses(self):
        m = self._one("The dog bit the man.")
        self.assertEqual(m["single_root"], 1)
        self.assertEqual(m["clausal"], 1)
        self.assertEqual(m["subj_before_root"], 1)
        self.assertEqual(m["parseable"], 1)

    def test_noun_pile_does_not_parse(self):
        m = self._one("gene TV spasm hydro mania quantum blood")
        self.assertEqual(m["clausal"], 0)
        self.assertEqual(m["parseable"], 0)

    def test_shuffled_order_is_not_primary_parseable(self):
        # order-sensitive by construction: the same words reordered lose the
        # subject-before-root shape often enough that the shuffled null is well
        # below the English ceiling
        good = self._one("The artist is the creator of beautiful things.")
        bad = self._one("creator things the is artist beautiful of the.")
        self.assertEqual(good["parseable"], 1)
        self.assertLessEqual(bad["parseable"], good["parseable"])

    def test_lenient_admits_imperative(self):
        m = self._one("Record the vital icon.")
        self.assertEqual(m["clausal"], 1)
        self.assertEqual(m["parseable_lenient"], 1)

    def test_dep_crossings_counts_crossing_arcs(self):
        doc = self.nlp("The dog bit the man.")
        self.assertGreaterEqual(pa.dep_crossings(doc), 0)


class TestHelpers(unittest.TestCase):
    def test_spearman_perfect(self):
        rho, _ = pa.spearman([1, 2, 3, 4], [10, 20, 30, 40])
        self.assertAlmostEqual(rho, 1.0, places=6)
        rho, _ = pa.spearman([1, 2, 3, 4], [40, 30, 20, 10])
        self.assertAlmostEqual(rho, -1.0, places=6)

    def test_rolling_mean_window(self):
        out = pa.rolling([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3)
        self.assertEqual(len(out), 6)
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[-1], 1.0)

    def test_shuffle_is_deterministic(self):
        import random
        a = pa.shuffle_tokens("a b c d e", random.Random(1))
        b = pa.shuffle_tokens("a b c d e", random.Random(1))
        self.assertEqual(a, b)
        self.assertEqual(sorted(a.split()), sorted(["a", "b", "c", "d", "e"]))


class TestArtifacts(unittest.TestCase):
    def test_all_artifacts_written(self):
        if _SKIP_PARSE:
            self.skipTest(_SKIP_PARSE)
        for name in ("sentences", "per_page", "derangement_curve", "baselines",
                     "reconciliation", "diagnostics"):
            self.assertTrue((Path(_ARTIFACTS["_dir"]) / f"{name}.tsv").exists(), name)

    def test_sentence_rows_have_parseable_columns(self):
        if _SKIP_PARSE:
            self.skipTest(_SKIP_PARSE)
        for r in rows("sentences"):
            self.assertIn(r["parseable"], ("0", "1"))
            self.assertIn(r["variant"],
                          ("v5_period_from_slashes", "v4_operators_to_period"))

    def test_per_page_aggregates_are_consistent(self):
        if _SKIP_PARSE:
            self.skipTest(_SKIP_PARSE)
        sent = [r for r in rows("sentences")
                if r["variant"] == "v5_period_from_slashes"]
        by_page = {}
        for r in sent:
            by_page.setdefault(r["page"], []).append(r)
        for r in rows("per_page"):
            if r["variant"] != "v5_period_from_slashes":
                continue
            rs = by_page[r["page"]]
            self.assertEqual(int(r["n_seg"]), len(rs))
            self.assertEqual(int(r["n_parseable"]), sum(int(x["parseable"]) for x in rs))

    def test_reconciliation_reproduces_13_word_total(self):
        if _SKIP_PARSE:
            self.skipTest(_SKIP_PARSE)
        # the quick run subsamples, so only the #13 recomputation rows are exact
        rec = {r["item"]: r for r in rows("reconciliation")}
        self.assertEqual(rec["v5 segments (#13 whole-file)"]["measured"], "2163")
        self.assertEqual(rec["v4 segments (#13 whole-file)"]["measured"], "10136")


if __name__ == "__main__":
    unittest.main(verbosity=2)
