#!/usr/bin/env python3
"""Regression suite for #47 multi-level entropy analysis.

Three layers:

* **Primitive tests** on synthetic symbol streams verify the entropy estimators
  themselves: the known unigram value, a deterministic bigram costing 0 bits, the
  context weighting, and that the held-out cross-entropy agrees with the plug-in
  estimate at order 0.
* **Corpus tests** verify the book inputs: 74,077 screened units, 10,949
  operators, and that the rebuilt operator stream equals #11's
  ``operator_stream.tsv`` operator-for-operator.
* **Quick-run tests** run a deterministic ``--quick`` end-to-end build into a
  temp dir and check the artifacts, the plugin <= held-out ordering, the
  English anchor (9-10 bits/word), the direction of every book-vs-English
  comparison, and TSV safety.

    python work/test_entropy_analysis.py
"""
import json
import math
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import entropy_analysis as ea

_ART = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="entropy_test_")
    rc = ea.main(["--quick", "--out", tmp, "--no-plot", "--quiet"])
    assert rc == 0
    for name in ("entropy_by_level", "char_entropy", "token_entropy",
                 "operator_entropy", "english_baseline", "type_entropy",
                 "reconciliation", "diagnostics"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = ea.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["_dir"] = tmp


def rows(name):
    return _ART[name]


def diag(name):
    return {r["diag"]: r["value"] for r in rows("diagnostics")}[name]


class TestPrimitives(unittest.TestCase):
    def test_unigram_known_value(self):
        seqs = {"1": ["a", "a", "a", "b"]}
        r = ea.plugin_conditional_entropy(seqs.values(), 0)
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        self.assertAlmostEqual(r["H"], expected, places=9)
        self.assertEqual(r["n"], 4)
        self.assertEqual(r["n_ctx"], 1)

    def test_order0_is_unigram(self):
        seqs = {"1": list("abracadabra"), "2": list("cadabra")}
        r = ea.plugin_conditional_entropy(seqs.values(), 0)
        combined = list("abracadabra") + list("cadabra")
        from collections import Counter
        c = Counter(combined)
        n = len(combined)
        exp = -sum((v / n) * math.log2(v / n) for v in c.values())
        self.assertAlmostEqual(r["H"], exp, places=9)

    def test_deterministic_bigram_is_zero(self):
        seqs = {"1": ["a", "b"] * 60}
        r = ea.plugin_conditional_entropy(seqs.values(), 1)
        self.assertAlmostEqual(r["H"], 0.0, places=9)

    def test_context_weighting(self):
        # pairs (x,a)x2, (a,x), (a,y)x2, (y,a), (y,b): contexts x(2,H=0),
        # a(3,H=0.918), y(2,H=1); weighted H = (2*0 + 3*0.918 + 2*1)/7
        seq = ["x", "a", "x", "a", "y", "a", "y", "b"]
        r = ea.plugin_conditional_entropy([seq], 1)
        self.assertAlmostEqual(r["H"], 0.6792696431662097, places=9)
        self.assertEqual(r["n"], 7)
        self.assertEqual(r["n_ctx"], 3)

    def test_empty(self):
        r = ea.plugin_conditional_entropy([], 2)
        self.assertEqual(r["H"], 0.0)
        self.assertEqual(r["n"], 0)
        self.assertEqual(r["n_ctx"], 0)

    def test_measure_heldout_agrees_at_order0(self):
        seqs = {"1": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "d"],
                "2": ["a", "b", "c", "d", "a", "b", "c", "d"]}
        m = ea.measure(seqs, orders=(0, 1, 2))
        self.assertEqual(m["n"], 18)
        self.assertAlmostEqual(m["plugin"][0], m["heldout"][0], delta=0.25)

    def test_measure_is_deterministic(self):
        seqs = {"1": list("abcabc"), "2": list("abcabd")}
        a = ea.measure(seqs, orders=(0, 1, 2, 3))
        b = ea.measure(seqs, orders=(0, 1, 2, 3))
        self.assertEqual(a["plugin"], b["plugin"])
        self.assertEqual(a["heldout"], b["heldout"])

    def test_split_pages(self):
        pages = ea.split_pages(list(range(10)), 3)
        self.assertEqual(sum(len(v) for v in pages.values()), 10)
        self.assertTrue(all(pages.values()))


class TestCorpus(unittest.TestCase):
    def test_book_counts(self):
        toks, pages = ea.load_book_tokens(True, True)
        self.assertEqual(len(pages), 225)
        self.assertEqual(sum(len(v) for v in toks.values()), ea.BOOK_UNITS_FULL)

    def test_operator_count(self):
        ops = ea.load_operator_pages()
        self.assertEqual(sum(len(v) for v in ops.values()), ea.BOOK_OPS_FULL)
        self.assertEqual(len({o for v in ops.values() for o in v}), 11)

    def test_operator_stream_matches_11(self):
        from_file = defaultdict(list)
        for r in ea.read_tsv(ea.OP_STREAM):
            from_file[int(r["page"])].append(r["operator"])
        rebuilt = ea.load_operator_pages()
        self.assertEqual(set(from_file), set(rebuilt))
        for page in from_file:
            self.assertEqual(from_file[page], rebuilt[page])

    def test_clean_page_strips_layout(self):
        self.assertEqual(ea.clean_page("<!-- x -->\n# Title\nbody"),
                         " \nTitle\nbody")

    def test_char_sequences_letters(self):
        seqs = ea.char_sequences({1: "Ab3 -c!"}, letters_only=True)
        self.assertEqual(seqs[1], ["a", "b", "c"])


class TestQuickRun(unittest.TestCase):
    def test_artifacts_present(self):
        for name in ("entropy_by_level", "char_entropy", "token_entropy",
                     "operator_entropy", "english_baseline", "type_entropy",
                     "reconciliation", "diagnostics"):
            self.assertTrue(rows(name), name)

    def test_pages_used(self):
        self.assertEqual(int(diag("pages_used")), 40)

    def test_plugin_not_above_heldout(self):
        # the MLE plug-in estimate is the in-sample minimum; the leakage-free
        # cross-entropy cannot beat it on the same stream
        for r in rows("entropy_by_level"):
            self.assertLessEqual(float(r["H_plugin_bits"]),
                                 float(r["H_heldout_bits"]) + 1e-6,
                                 f"{r['level']}/{r['variant']} order {r['order']}")

    def test_all_orders_present(self):
        book = {(r["level"], r["variant"], int(r["order"]))
                for r in rows("entropy_by_level") if r["corpus"] == "book"}
        for level, variant in (("char", "raw"), ("char", "letters"),
                               ("char", "screened"), ("token", "unit"),
                               ("token", "unit_cased"),
                               ("operator", "operator")):
            for k in ea.ORDERS:
                self.assertIn((level, variant, k), book)

    def test_english_word_anchor(self):
        # the 9-10 bits/word raw anchor must be reproduced by the baseline
        tok = {(r["corpus"], int(r["order"])): r
               for r in rows("entropy_by_level")
               if r["level"] == "token" and r["variant"] == "unit"}
        eng0 = float(tok[("english_pooled", 0)]["H_plugin_bits"])
        self.assertGreaterEqual(eng0, 8.0)
        self.assertLessEqual(eng0, 11.0)

    def test_book_more_predictable_than_english(self):
        rows_by = {(r["corpus"], r["level"], r["variant"], int(r["order"])): r
                   for r in rows("entropy_by_level")}
        # token unigram
        self.assertLess(float(rows_by[("book", "token", "unit", 0)]["H_plugin_bits"]),
                        float(rows_by[("english_pooled", "token", "unit", 0)]["H_plugin_bits"]))
        # char letters order 3 (held-out)
        self.assertLess(float(rows_by[("book", "char", "letters", 3)]["H_heldout_bits"]),
                        float(rows_by[("english_pooled", "char", "letters", 3)]["H_heldout_bits"]))

    def test_operator_channel_does_not_gain_from_context(self):
        op = {int(r["order"]): r for r in rows("entropy_by_level")
              if r["corpus"] == "book" and r["level"] == "operator"}
        self.assertGreaterEqual(float(op[3]["H_heldout_bits"]),
                                float(op[1]["H_heldout_bits"]))

    def test_operator_stream_match_reported(self):
        recon = {r["quantity"]: r["here"] for r in rows("reconciliation")}
        self.assertEqual(recon["operator stream identity"], "match")
        self.assertEqual(diag("operator_stream_match"), "1")

    def test_english_baseline_has_eight_references(self):
        per = defaultdict(set)
        for r in rows("english_baseline"):
            per[(r["level"], r["variant"])].add(r["reference"])
        self.assertTrue(per)
        for key, refs in per.items():
            self.assertEqual(len(refs), 8, key)

    def test_type_entropy_top_symbols(self):
        tok = {(r["variant"], r["symbol"]) for r in rows("type_entropy")
               if r["level"] == "token"}
        self.assertIn(("unit", "the"), tok)
        ops = {r["symbol"] for r in rows("type_entropy")
               if r["level"] == "operator"}
        self.assertIn("::", ops)

    def test_summary_contract(self):
        s = _ART["summary"]
        self.assertEqual(s["schema"], ea.SCHEMA)
        self.assertEqual(s["orders"], list(ea.ORDERS))
        self.assertIn("book", s)
        self.assertIn("english_pooled", s)
        self.assertTrue(s["operator_stream_match"])


class TestTSVSafety(unittest.TestCase):
    def test_column_counts_match_header(self):
        for name in ("entropy_by_level", "char_entropy", "token_entropy",
                     "operator_entropy", "english_baseline", "type_entropy",
                     "reconciliation", "diagnostics"):
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
