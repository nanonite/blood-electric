#!/usr/bin/env python3
"""Regression suite for #49 LM perplexity per page.

Four layers:

* **Primitives** use a stub LM (uniform logits) to verify the perplexity
  arithmetic and the sliding-window scorer: correct NLL values, exactly one
  score per token, correct streaming attribution, and determinism.
* **Corpus** verifies the 225 page bodies, the #13 variant reader, and that the
  #35 surrogate returns finite per-page NLLs in nats.
* **Surrogate quick-run** runs a deterministic, offline end-to-end build into a
  temp dir and checks the artifacts, the per-page curve, and the book-vs-English
  direction.
* **GPT-2** (skipped when the cached weights are absent) exercises the neural
  path end to end on a small deterministic sample.

    python work/test_lm_perplexity.py
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lm_perplexity as lp

try:  # torch backs the neural path and the stub-model primitive tests
    import torch
except Exception:  # noqa: BLE001
    torch = None

_ART = {}
_GPT = {}
_HAS_GPT2 = lp.gpt2_local_available()


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="lmppl_test_")
    rc = lp.main(["--instrument", "surrogate", "--quick", "--out", tmp,
                  "--no-plot", "--quiet"])
    assert rc == 0
    for name in ("per_page", "perplexity_by_instrument", "reconciliation",
                 "diagnostics"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = lp.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["report"] = (Path(tmp) / "lm_perplexity_report.md").read_text(
        encoding="utf-8")
    _ART["_dir"] = tmp

    if _HAS_GPT2:
        gtmp = tempfile.mkdtemp(prefix="lmppl_gpt_")
        rc = lp.main(["--instrument", "gpt2", "--sample", "4", "--no-english",
                      "--screened", "--out", gtmp, "--no-plot", "--quiet"])
        assert rc == 0
        _GPT["per_page"] = lp.read_tsv(Path(gtmp) / "per_page.tsv")
        _GPT["diag"] = lp.read_tsv(Path(gtmp) / "diagnostics.tsv")
        _GPT["_dir"] = gtmp


def rows(name):
    return _ART[name]


def diag(name):
    return {r["diag"]: r["value"] for r in rows("diagnostics")}[name]


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Stub:
    """A fake causal LM with uniform logits over ``V`` classes."""

    def __init__(self, V):
        self.V = V

    def __call__(self, ids):
        B, T = ids.shape
        return _Out(torch.zeros(B, T, self.V))


@unittest.skipUnless(torch is not None, "torch not installed")
class TestPrimitives(unittest.TestCase):
    V = 64

    def test_nll_uniform_is_ln_v(self):
        m = _Stub(self.V)
        nlls = lp.score_token_ids(m, list(range(1, 51)), max_context=1024)
        self.assertEqual(len(nlls), 49)
        for x in nlls:
            self.assertAlmostEqual(x, math.log(self.V), places=5)

    def test_windowing_scores_every_token_once(self):
        m = _Stub(128)
        ids = list(range(1, 101))
        nlls = lp.score_token_ids(m, ids, max_context=16)
        self.assertEqual(len(nlls), len(ids) - 1)
        # No token was skipped (None replaced by 0.0 would break this) and none
        # was counted twice in a way that changes a uniform-logit value.
        for x in nlls:
            self.assertAlmostEqual(x, math.log(128), places=5)

    def test_short_sequence(self):
        m = _Stub(self.V)
        self.assertEqual(lp.score_token_ids(m, [1], max_context=16), [])
        self.assertEqual(len(lp.score_token_ids(m, [1, 2], max_context=16)), 1)

    def test_streaming_attribution(self):
        m = _Stub(self.V)
        pages = {1: [1, 2, 3, 4, 5], 2: [6, 7, 8, 9, 10], 3: [11, 12, 13]}
        res = lp.score_pages_streaming(m, pages, [1, 2, 3], max_context=1024)
        self.assertEqual(res[1]["n_scored"], 4)   # first stream token has no ctx
        self.assertEqual(res[2]["n_scored"], 5)
        self.assertEqual(res[3]["n_scored"], 3)
        total = sum(res[p]["n_scored"] for p in res)
        self.assertEqual(total, sum(len(v) for v in pages.values()) - 1)

    def test_isolated_resets_context(self):
        m = _Stub(self.V)
        pages = {1: [1, 2, 3, 4, 5], 2: [6, 7, 8, 9, 10]}
        res = lp.score_pages_isolated(m, pages, [1, 2], max_context=1024)
        self.assertEqual(res[1]["n_scored"], 4)
        self.assertEqual(res[2]["n_scored"], 4)

    def test_perplexity_arithmetic(self):
        nlls = [math.log(2), math.log(2)]
        self.assertAlmostEqual(lp.perplexity_from_nll(nlls), 2.0, places=9)
        self.assertAlmostEqual(lp.bits_per_token(nlls), 1.0, places=9)
        self.assertEqual(lp.perplexity_from_nll([]), 0.0)

    def test_summarize_pages(self):
        res = {1: {"n_tokens": 3, "n_scored": 2, "nlls": [0.0, math.log(4)]},
               2: {"n_tokens": 2, "n_scored": 2, "nlls": [0.0, 0.0]}}
        s = lp.summarize_pages(res)
        self.assertEqual(s["n_scored"], 4)
        self.assertAlmostEqual(s["mean_nll_nats"], math.log(4) / 4, places=9)
        self.assertAlmostEqual(s["perplexity"], math.exp(math.log(4) / 4),
                               places=9)

    def test_deterministic(self):
        m = _Stub(self.V)
        a = lp.score_token_ids(m, list(range(1, 60)), max_context=16)
        b = lp.score_token_ids(m, list(range(1, 60)), max_context=16)
        self.assertEqual(a, b)

    def test_zones_flag(self):
        page_rows = [{"page": i, "n_scored": 50, "mean_nll_nats": float(i)}
                     for i in range(1, 11)]
        lp._zones(page_rows)
        self.assertTrue(any(r["zone"] == "hot" for r in page_rows))
        self.assertTrue(any(r["zone"] == "cold" for r in page_rows))
        # small pages never flagged
        small = [{"page": 1, "n_scored": 3, "mean_nll_nats": 99.0}]
        lp._zones(small)
        self.assertEqual(small[0]["zone"], "")


class TestCorpus(unittest.TestCase):
    def test_page_texts_225(self):
        texts = lp.page_texts()
        self.assertEqual(len(texts), 225)
        self.assertGreater(sum(1 for t in texts.values() if t.strip()), 200)

    def test_silence_variant_reader(self):
        text = lp.silence_variant_text("v0_original")
        self.assertIsNotNone(text)
        self.assertNotIn("<!--", text)
        self.assertGreater(len(text), 1000)

    def test_surrogate_pages_finite(self):
        pages = [4, 5, 6]
        per_page, meta = lp.surrogate_pages(pages)
        self.assertTrue(per_page)
        self.assertIn(meta["best_context_len"], (0, 1, 2, 3))
        for r in per_page.values():
            self.assertGreater(r["n_scored"], 0)
            self.assertTrue(all(x > 0 for x in r["nlls"]))

    def test_join_tables_keys(self):
        joins = lp.join_tables()
        self.assertIn(4, joins)
        self.assertIn("ng_surprisal_bits", joins[4])


class TestSurrogateQuickRun(unittest.TestCase):
    def test_artifacts_present(self):
        for name in ("per_page", "perplexity_by_instrument", "reconciliation",
                     "diagnostics"):
            self.assertTrue(rows(name), name)
        self.assertTrue(_ART["summary"])
        self.assertIn("LM perplexity per page", _ART["report"])

    def test_per_page_rows_and_values(self):
        pp = rows("per_page")
        self.assertEqual(len(pp), 40)  # --quick samples 40 evenly spaced pages
        scored = [r for r in pp if int(r["n_scored"]) >= lp.MIN_PAGE_TOKENS]
        self.assertTrue(scored)
        for r in scored:
            self.assertGreater(float(r["perplexity"]), 1.0)
            self.assertGreater(float(r["bits_per_token"]), 0.0)
            self.assertIn(r["zone"], ("hot", "cold", ""))

    def test_book_more_predictable_than_english_surrogate(self):
        by = {}
        for r in rows("perplexity_by_instrument"):
            by[(r["scope"], r["mode"])] = float(r["perplexity"])
        self.assertIn(("book", "streaming"), by)
        self.assertIn(("english", "streaming"), by)
        # the #35/#47/#48 direction: the book's repetitive loop is cheaper
        self.assertLess(by[("book", "streaming")], by[("english", "streaming")])

    def test_instrument_and_counts(self):
        self.assertEqual(diag("instrument"), "surrogate")
        self.assertEqual(diag("pages_scored"), "40")
        recon = {r["quantity"]: r["here"] for r in rows("reconciliation")}
        self.assertEqual(recon["English references"], "8")
        self.assertEqual(recon["pages scored"], "40")

    def test_summary_contract(self):
        s = _ART["summary"]
        self.assertEqual(s["schema"], lp.SCHEMA)
        self.assertEqual(s["instrument"], "surrogate")
        self.assertGreater(s["book"]["streaming"]["perplexity"], 1.0)
        self.assertIn("hot_pages", s)
        self.assertIn("cold_pages", s)


@unittest.skipUnless(_HAS_GPT2, "cached GPT-2 weights unavailable")
class TestGPT2Path(unittest.TestCase):
    def test_gpt2_quick_run(self):
        pp = _GPT["per_page"]
        self.assertEqual(len(pp), 4)
        scored = [r for r in pp if int(r["n_scored"]) >= lp.MIN_PAGE_TOKENS]
        self.assertTrue(scored)
        for r in scored:
            self.assertGreater(float(r["perplexity"]), 1.0)
        d = {r["diag"]: r["value"] for r in _GPT["diag"]}
        self.assertEqual(d["instrument"], "gpt2")
        self.assertGreater(int(d["book_tokens"]), 0)
        # the markup-screened sensitivity ran and produced a finite value
        self.assertIn("screened_perplexity", pp[0])
        self.assertNotEqual(d["book_perplexity_screened"], "n/a")


class TestTSVSafety(unittest.TestCase):
    def test_column_counts_match_header(self):
        for name in ("per_page", "perplexity_by_instrument", "reconciliation",
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
