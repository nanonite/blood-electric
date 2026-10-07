#!/usr/bin/env python3
"""Regression suite for #50 quantified silence-removal experiment.

Four layers:

* **Primitives** verify the sign-test arithmetic and the per-page splitter.
* **Entropy/compression legs** run the builder with ``--instrument skip`` into a
  temp dir and check the entropy blind spot (v3-v6 token entropy == v0), the
  compression deltas (all negative -- stripping operators lowers bits/char),
  and the reconciliation against the frozen #13/#48 tables.
* **Perplexity arithmetic** uses a stub LM (uniform logits) to verify the
  scoring path and the paired-delta aggregation on synthetic data.
* **GPT-2** (skipped when the cached weights are absent) exercises the neural
  leg end to end on a small deterministic sample and checks the verdict.

    python work/test_silence_experiment.py
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import silence_experiment as se

try:  # torch backs the stub-model primitive tests
    import torch
except Exception:  # noqa: BLE001
    torch = None

_ART = {}
_HAS_GPT2 = se.lp.gpt2_local_available()


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="se_test_")
    rc = se.main(["--instrument", "skip", "--out", tmp, "--no-plot", "--quiet"])
    assert rc == 0
    for name in ("deltas", "entropy_by_variant", "compression_by_variant",
                 "reconciliation", "diagnostics"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = se.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["report"] = (Path(tmp) / "silence_experiment_report.md").read_text(
        encoding="utf-8")
    _ART["_dir"] = tmp


def rows(name):
    return _ART[name]


def diag(name):
    return {r["diag"]: r["value"] for r in rows("diagnostics")}[name]


class TestPrimitives(unittest.TestCase):
    def test_sign_test_extremes(self):
        # all improved -> tiny two-sided p
        self.assertAlmostEqual(se._sign_test_p(10, 0), 2 / 1024, places=8)
        # balanced -> p = 1
        self.assertAlmostEqual(se._sign_test_p(5, 5), 1.0, places=8)
        # empty -> 1
        self.assertEqual(se._sign_test_p(0, 0), 1.0)

    def test_sign_test_known(self):
        # n=8, k=3: tail = (C(8,0)+C(8,1)+C(8,2)+C(8,3))/256 = (1+8+28+56)/256
        self.assertAlmostEqual(se._sign_test_p(3, 5), 2 * 93 / 256, places=8)

    def test_variant_pages_shape(self):
        pages = se.variant_pages("v0_original")
        self.assertEqual(len(pages), 225)
        # page 1 is the front-matter line
        self.assertIn("japanese cyberpunk", pages[1].lower())
        # every variant has the same page set
        for name in se.VARIANTS:
            self.assertEqual(set(se.variant_pages(name)), set(pages), name)

    def test_char_entropy_finite(self):
        for name in se.VARIANTS:
            self.assertGreater(se.char_entropy_bits(se.variant_text(name)), 0.0)


class TestEntropyLeg(unittest.TestCase):
    def test_seven_variants(self):
        self.assertEqual(len(rows("entropy_by_variant")), 7)

    def test_token_entropy_blind_spot(self):
        # v3-v6 keep the word sequence identical to v0 -> identical token entropy
        ent = {r["variant"]: r for r in rows("entropy_by_variant")}
        base = ent["v0_original"]["token_entropy_bits"]
        for name in ("v3_operators_to_space", "v4_operators_to_period",
                     "v5_period_from_slashes", "v6_period_from_colons"):
            self.assertEqual(ent[name]["token_entropy_bits"], base, name)
            self.assertEqual(float(ent[name]["delta_token_entropy_vs_v0"]), 0.0, name)

    def test_delete_variants_change_token_entropy(self):
        ent = {r["variant"]: r for r in rows("entropy_by_variant")}
        base = ent["v0_original"]["token_entropy_bits"]
        # v1 fuses the most -> largest token-entropy jump; v2 is partial
        self.assertGreater(ent["v1_delete_all_operators"]["token_entropy_bits"], base)
        self.assertGreater(ent["v2_delete_colon_slash"]["token_entropy_bits"], base)

    def test_char_entropy_moves(self):
        ent = {r["variant"]: r for r in rows("entropy_by_variant")}
        base = ent["v0_original"]["char_entropy_bits"]
        for name in se.VARIANTS[1:]:
            self.assertNotEqual(ent[name]["char_entropy_bits"], base, name)


class TestCompressionLeg(unittest.TestCase):
    def test_shape(self):
        # 7 variants x 6 compressors
        self.assertEqual(len(rows("compression_by_variant")), 42)

    def test_gzip_deltas_negative(self):
        # the headline compressor: stripping/replacing operators lowers bits/char
        comp = [r for r in rows("compression_by_variant") if r["compressor"] == "gzip-9"]
        self.assertEqual(len(comp), 7)
        for r in comp:
            if r["variant"] == "v0_original":
                self.assertEqual(float(r["delta_bits_per_char_vs_v0"]), 0.0)
            else:
                self.assertLess(float(r["delta_bits_per_char_vs_v0"]), 0.0,
                                f"{r['variant']}/{r['compressor']}")

    def test_bz2_delete_variants_exception_disclosed(self):
        # honest negative result: under bz2, the two DELETE variants (which fuse
        # words into portmanteaux) are slightly LESS compressible than v0 --
        # the only positive deltas.  Every other variant/compressor pair is
        # negative (stripping operators lowers bits/char).
        comp = {(r["variant"], r["compressor"]): r for r in rows("compression_by_variant")}
        exceptions = sorted((v, c) for (v, c), r in comp.items()
                            if v != "v0_original" and float(r["delta_bits_per_char_vs_v0"]) > 0)
        self.assertEqual(exceptions, [("v1_delete_all_operators", "bz2-9"),
                                      ("v2_delete_colon_slash", "bz2-9")])

    def test_reconciles_with_48(self):
        # my bits/char must match #48's frozen silence_variants.tsv exactly
        v48 = {(r["variant"], r["compressor"]): r for r in se._frozen(
            se.COMPRESSION_DIR / "silence_variants.tsv")}
        checked = 0
        for r in rows("compression_by_variant"):
            old = v48.get((r["variant"], r["compressor"]))
            if old:
                self.assertAlmostEqual(float(r["bits_per_char"]),
                                       float(old["bits_per_symbol"]), places=6)
                checked += 1
        self.assertGreater(checked, 0)


class TestMasterTable(unittest.TestCase):
    def test_deltas_shape(self):
        d = rows("deltas")
        self.assertEqual(len(d), 7)
        self.assertEqual(d[0]["variant"], "v0_original")
        # v0 deltas are all zero
        for k in ("delta_token_entropy_vs_v0", "delta_char_entropy_vs_v0",
                  "delta_gzip9_bits_per_char_vs_v0"):
            self.assertEqual(float(d[0][k]), 0.0, k)

    def test_perplexity_columns_empty_when_skipped(self):
        d = rows("deltas")
        for r in d:
            self.assertEqual(r["perplexity_streaming"], "")
            self.assertEqual(r["delta_perplexity_vs_v0"], "")


class TestPairedLeg(unittest.TestCase):
    def test_paired_aggregation(self):
        # synthetic: v1 improves 8 pages, worsens 2
        base = {p: {"n_scored": 10, "nlls": [1.0] * 10} for p in range(1, 11)}
        var = {p: {"n_scored": 10,
                   "nlls": [0.5] * 10 if p <= 8 else [1.5] * 10} for p in range(1, 11)}
        rows = se.paired_leg({"v0_original": base, "v1_delete_all_operators": var})
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["n_pages_paired"], 10)
        self.assertEqual(r["n_improved"], 8)
        self.assertEqual(r["n_worsened"], 2)
        self.assertAlmostEqual(r["mean_delta_nll_nats"], -0.3, places=6)
        self.assertAlmostEqual(r["frac_pages_improved"], 0.8, places=6)

    def test_paired_excludes_low_token_pages(self):
        base = {1: {"n_scored": 10, "nlls": [1.0] * 10},
                2: {"n_scored": 2, "nlls": [1.0, 1.0]}}      # below floor
        var = {1: {"n_scored": 10, "nlls": [0.5] * 10},
               2: {"n_scored": 2, "nlls": [0.5, 0.5]}}
        rows = se.paired_leg({"v0_original": base, "v1_delete_all_operators": var})
        self.assertEqual(rows[0]["n_pages_paired"], 1)


@unittest.skipUnless(torch is not None, "torch not installed")
class TestPerplexityArithmetic(unittest.TestCase):
    V = 32

    def test_score_token_ids_uniform(self):
        class _Out:
            def __init__(self, logits):
                self.logits = logits

        class _Stub:
            def __init__(self, V):
                self.V = V

            def __call__(self, ids):
                B, T = ids.shape
                return _Out(torch.zeros(B, T, self.V))

        ids = list(range(1, self.V))          # 31 ids, all < V=32
        nlls = se.lp.score_token_ids(_Stub(self.V), ids, max_context=1024)
        self.assertEqual(len(nlls), len(ids) - 1)
        for x in nlls:
            self.assertAlmostEqual(x, math.log(self.V), places=5)

    def test_perplexity_and_bits(self):
        nlls = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(se.lp.perplexity_from_nll(nlls), math.exp(2.0), places=9)
        self.assertAlmostEqual(se.lp.bits_per_token(nlls), 2.0 / math.log(2), places=9)


@unittest.skipUnless(_HAS_GPT2, "GPT-2 weights not cached")
class TestGPT2EndToEnd(unittest.TestCase):
    def test_quick_build_with_verdict(self):
        tmp = tempfile.mkdtemp(prefix="se_gpt_")
        rc = se.main(["--instrument", "gpt2", "--sample", "6",
                      "--out", tmp, "--no-plot", "--quiet"])
        assert rc == 0
        ppl = se.read_tsv(Path(tmp) / "perplexity_by_variant.tsv")
        self.assertEqual(len(ppl), 7)
        # v0 is the baseline; every variant has a delta column populated
        v0 = next(r for r in ppl if r["variant"] == "v0_original")
        self.assertGreater(float(v0["perplexity"]), 0.0)
        for r in ppl:
            self.assertNotEqual(r["delta_perplexity_vs_v0"], "")
        paired = se.read_tsv(Path(tmp) / "paired_deltas.tsv")
        self.assertEqual(len(paired), 6)
        with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertIn("winner", summary["verdict"])
        self.assertNotEqual(summary["verdict"]["winner"], "v0_original")
        report = (Path(tmp) / "silence_experiment_report.md").read_text(encoding="utf-8")
        self.assertIn("Verdict", report)


if __name__ == "__main__":
    unittest.main()
