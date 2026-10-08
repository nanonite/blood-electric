#!/usr/bin/env python3
"""Regression suite for #56 keyness harness.

Checks the invariants that matter: the log-likelihood and log-ratio formulas on
known values, the book-side markup screen (word parts), the reference-side
tokenisation, the keyness row structure, and that the full run writes every
expected artifact.  Reads only frozen artifacts and the cached reference
corpora - no network, no OCR.

    python work/test_keyness.py
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import keyness as kn

ROOT = kn.ROOT
CORPORA = kn.CORPORA


class TestLogLikelihood(unittest.TestCase):
    """Signed G^2 on known 2x2 tables."""

    def test_over_represented_positive(self):
        # 100 in book vs 10 in ref at equal N -> clearly over-represented.
        g2 = kn.log_likelihood(100, 10, 100_000, 100_000)
        self.assertGreater(g2, 0)
        self.assertAlmostEqual(g2, 85.472, places=2)

    def test_equal_frequency_zero(self):
        self.assertAlmostEqual(kn.log_likelihood(50, 50, 100_000, 100_000), 0.0)

    def test_absent_from_ref_positive(self):
        g2 = kn.log_likelihood(5, 0, 100_000, 100_000)
        self.assertGreater(g2, 0)

    def test_absent_from_book_negative(self):
        g2 = kn.log_likelihood(0, 500, 100_000, 100_000)
        self.assertLess(g2, 0)

    def test_both_zero_zero(self):
        self.assertEqual(kn.log_likelihood(0, 0, 100, 100), 0.0)


class TestLogRatio(unittest.TestCase):
    """0.5-smoothed log-ratio effect size."""

    def test_over_represented_positive(self):
        lr = kn.log_ratio(100, 10, 100_000, 100_000)
        self.assertGreater(lr, 0)
        # 0.5-smoothed: ln((100.5)/(10.5)) rather than ln(100/10).
        self.assertAlmostEqual(lr, math.log(100.5 / 10.5), places=4)

    def test_under_represented_negative(self):
        lr = kn.log_ratio(10, 100, 100_000, 100_000)
        self.assertLess(lr, 0)

    def test_zero_counts_finite(self):
        # 0.5 smoothing keeps the ratio finite when a count is zero.
        self.assertTrue(math.isfinite(kn.log_ratio(5, 0, 1000, 1000)))
        self.assertTrue(math.isfinite(kn.log_ratio(0, 5, 1000, 1000)))


class TestBookSide(unittest.TestCase):
    """The book side is the markup-screened #17 word-part layer."""

    @classmethod
    def setUpClass(cls):
        cls.book, cls.stats = kn.load_book_words(log=lambda m: None)

    def test_counts_reconcile(self):
        s = self.stats
        self.assertEqual(s["parts"], 80_363, "#17 word-part total")
        self.assertEqual(s["kept"] + s["dropped"], s["parts"])
        self.assertEqual(sum(self.book.values()), s["kept"])
        self.assertEqual(len(self.book), s["types"])

    def test_markup_screen_removes_some(self):
        self.assertGreater(self.stats["dropped"], 0)
        self.assertLess(self.stats["dropped"], self.stats["parts"])

    def test_no_markup_types(self):
        # #19 junk / markup types must not survive the screen.
        _spans, _uids, types = kn.load_markup()
        for t in types:
            self.assertNotIn(t, self.book,
                             f"markup type {t!r} survived the screen")

    def test_common_words_present(self):
        for w in ("the", "of", "and"):
            self.assertIn(w, self.book)


class TestReferenceSide(unittest.TestCase):
    """Each reference corpus tokenises with the #24 English word regex."""

    def test_general_english(self):
        ref = kn.load_reference_words("general_english", log=lambda m: None)
        self.assertGreater(sum(ref.values()), 1_000_000)
        self.assertIn("the", ref)

    def test_excerpt_corpora_strip_comments(self):
        # The # header and -- attribution lines must not be counted.
        ref = kn.load_reference_words("cyberpunk", log=lambda m: None)
        for w in ("fair", "use", "excerpts", "copyright", "neuromancer"):
            self.assertNotIn(w, ref,
                             f"excerpt header/attribution word {w!r} counted")

    def test_medical_and_cs(self):
        med = kn.load_reference_words("medical", log=lambda m: None)
        cs = kn.load_reference_words("cs", log=lambda m: None)
        self.assertGreater(sum(med.values()), 100_000)
        self.assertGreater(sum(cs.values()), 50_000)


class TestBenjaminiHochberg(unittest.TestCase):
    """BH (1995) step-up adjustment on known inputs."""

    def test_empty(self):
        self.assertEqual(kn.benjamini_hochberg([]), [])

    def test_all_ones_is_one(self):
        self.assertEqual(kn.benjamini_hochberg([1.0, 1.0, 1.0]), [1.0, 1.0, 1.0])

    def test_single_test_unchanged(self):
        # BH cannot correct a single test.
        self.assertAlmostEqual(kn.benjamini_hochberg([0.01])[0], 0.01)

    def test_known_vector(self):
        # p = [.01, .02, .03, .04, .05]; n=5.  (5/j)*p_j is exactly 0.05 for
        # every rank j, so the step-up minimum is 0.05 throughout.
        q = kn.benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
        for v in q:
            self.assertAlmostEqual(v, 0.05)

    def test_known_vector_min_takes_over(self):
        # p = [.001, .5, .5, .5, .5, .5, .5, .5]; n=8.  The smallest p is
        # 0.001*8/1 = 0.008; every larger p gives (8/j)*0.5 >= 0.5, so the
        # step-up minimum for the smallest term is 0.008 and the rest stay 0.5.
        q = kn.benjamini_hochberg([0.001] + [0.5] * 7)
        self.assertAlmostEqual(q[0], 0.008)
        for v in q[1:]:
            self.assertAlmostEqual(v, 0.5)

    def test_q_is_monotone_in_p(self):
        ps = [0.001, 0.02, 0.2, 0.5, 0.9]
        q = kn.benjamini_hochberg(ps)
        self.assertEqual(q, sorted(q))

    def test_q_ge_p(self):
        # BH is a conservative adjustment: q never falls below p.
        ps = [0.001, 0.02, 0.2, 0.5, 0.9, 0.03]
        for p, qq in zip(ps, kn.benjamini_hochberg(ps)):
            self.assertGreaterEqual(qq, p - 1e-12)

    def test_q_bounded(self):
        q = kn.benjamini_hochberg([0.5, 0.9, 0.99])
        for v in q:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestKeynessRows(unittest.TestCase):
    """The keyness table has the right shape and ranking."""

    @classmethod
    def setUpClass(cls):
        cls.book, _ = kn.load_book_words(log=lambda m: None)
        cls.ref = kn.load_reference_words("general_english", log=lambda m: None)
        cls.rows, cls.n1, cls.n2 = kn.keyness_rows(cls.book, cls.ref)

    def test_row_columns(self):
        for col in ("word", "book_freq", "ref_freq", "book_rel", "ref_rel",
                    "log_likelihood", "p_value", "q_value", "log_ratio"):
            self.assertIn(col, self.rows[0])

    def test_ranked_by_g2_descending(self):
        g2s = [r["log_likelihood"] for r in self.rows]
        self.assertEqual(g2s, sorted(g2s, reverse=True))

    def test_union_of_types(self):
        n_types = len(set(self.book) | set(self.ref))
        self.assertEqual(len(self.rows), n_types)

    def test_top_key_word_over_represented(self):
        top = self.rows[0]
        self.assertGreater(top["log_likelihood"], 0)
        self.assertGreater(top["book_rel"], top["ref_rel"])

    def test_p_value_range(self):
        for r in self.rows:
            self.assertGreaterEqual(r["p_value"], 0.0)
            self.assertLessEqual(r["p_value"], 1.0)

    def test_q_value_range_and_ordering(self):
        for r in self.rows:
            self.assertGreaterEqual(r["q_value"], 0.0)
            self.assertLessEqual(r["q_value"], 1.0)
            self.assertGreaterEqual(r["q_value"], r["p_value"] - 1e-12)

    def test_fdr_gate_is_stricter_than_p_gate(self):
        """The q-gate must admit no more words than the raw p-gate.

        This is the whole reason the gate is q: with tens of thousands of
        types tested, uncorrected p < 0.05 is far too permissive.
        """
        by_p = [r for r in self.rows if r["p_value"] < kn.P_THRESHOLD]
        by_q = [r for r in self.rows if r["q_value"] < kn.Q_THRESHOLD]
        self.assertLessEqual(len(by_q), len(by_p))
        self.assertGreater(len(by_p), len(by_q),
                           "the correction should actually remove words")


class TestCorpusFileCount(unittest.TestCase):
    """The reported per-corpus file count must match what was counted."""

    def test_counts_match_manifest(self):
        manifest = kn.CORPORA / "manifest.tsv"
        self.assertTrue(manifest.exists(), "run reference_corpora.py first")
        with manifest.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            rows = [dict(zip(header, ln.rstrip("\n").split("\t")))
                    for ln in fh if ln.strip()]
        for corpus in kn.CORPUS_DIRS:
            expected = sum(1 for r in rows if r["corpus"] == corpus)
            self.assertEqual(len(kn.corpus_files(corpus)), expected,
                             f"{corpus} file count disagrees with the manifest")

    def test_staging_files_excluded(self):
        for corpus in kn.CORPUS_DIRS:
            for f in kn.corpus_files(corpus):
                self.assertFalse(f.name.startswith("_"),
                                 f"{f.name} is a staging file")


class TestKeynessManifestReconciliation(unittest.TestCase):
    """Keyness token counts must match the reference-corpus manifest."""

    def test_tokens_reconcile(self):
        manifest = kn.CORPORA / "manifest.tsv"
        self.assertTrue(manifest.exists())
        with manifest.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            rows = [dict(zip(header, ln.rstrip("\n").split("\t")))
                    for ln in fh if ln.strip()]
        # load_reference_words strips # / -- header lines from excerpt files,
        # so only the prose corpora must reconcile exactly.
        for corpus in ("general_english", "medical", "cs", "legal"):
            ref = kn.load_reference_words(corpus, log=lambda m: None)
            expected = sum(int(r["tokens"]) for r in rows
                           if r["corpus"] == corpus)
            self.assertEqual(sum(ref.values()), expected,
                             f"{corpus} token count disagrees with manifest")


class TestReferenceCores(unittest.TestCase):
    """The reference-independent core is computed over inferential corpora only.

    A ~50-token excerpt corpus admits almost nothing through the q-gate, so
    folding it into the intersection would turn corpus size into the finding
    ("nothing is stable").  That is what the first implementation did, and it
    is the reason ``inferential`` exists.
    """

    def _lists(self, shared, small=()):
        out = {}
        for c in kn.CORPUS_DIRS:
            out[c] = ([] if c in small else
                      [{"word": f"{c}_only"}] + [{"word": w} for w in shared])
        return out

    def _per_corpus(self, **overrides):
        per = {c: {"tokens": kn.MIN_TOKENS_INFERENTIAL + 1}
               for c in kn.CORPUS_DIRS}
        per.update(overrides)
        return per

    def test_core_ignores_corpora_below_the_token_floor(self):
        lists = self._lists(["cadaver", "murder"])
        core, inferential, _ge = kn.reference_cores(
            self._per_corpus(cyberpunk={"tokens": 105},
                             finnegans_wake={"tokens": 49}), lists)
        self.assertEqual(core, sorted(["cadaver", "murder"]))
        self.assertNotIn("cyberpunk", inferential)
        self.assertNotIn("finnegans_wake", inferential)
        self.assertEqual(len(inferential), len(kn.CORPUS_DIRS) - 2)

    def test_empty_core_when_nothing_is_shared(self):
        lists = {c: [{"word": c}] for c in kn.CORPUS_DIRS}
        core, _infer, _ge = kn.reference_cores(self._per_corpus(), lists)
        self.assertEqual(core, [])

    def test_general_english_only_residue(self):
        lists = {
            "general_english": [{"word": "codes"}, {"word": "cadaver"}],
            "medical": [{"word": "cadaver"}],
            "cs": [{"word": "cadaver"}],
            "legal": [{"word": "cadaver"}],
            "cyberpunk": [],
            "cutup_surrealist": [],
            "finnegans_wake": [],
        }
        per = self._per_corpus(cyberpunk={"tokens": 105},
                               cutup_surrealist={"tokens": 56},
                               finnegans_wake={"tokens": 49})
        core, _inferential, ge_only = kn.reference_cores(per, lists)
        self.assertEqual(core, ["cadaver"])
        self.assertEqual(ge_only, ["codes"])

    def test_section_flags_small_corpora_and_still_reports_a_core(self):
        per = self._per_corpus(cyberpunk={"tokens": 105},
                               finnegans_wake={"tokens": 49})
        lists = self._lists(["cadaver"], small=("cyberpunk", "finnegans_wake"))
        core, _inferential, _ge = kn.reference_cores(per, lists)
        text = "\n".join(kn.cross_corpus_section(per, lists,
                                                (core, _inferential, _ge),
                                                log=lambda m: None))
        self.assertIn("| cyberpunk | 105 | no (0 keys) | 0 | - | - |", text)
        self.assertIn("Reference-independent core.", text)
        self.assertNotIn("reference-dependent throughout", text)


class TestWriteReport(unittest.TestCase):
    """write_report must not reach for a module-level ``log`` (#56 regression).

    The first implementation called ``cross_corpus_section(..., log)`` with no
    ``log`` in scope, so every full run raised after writing the TSVs and the
    report was never produced.
    """

    @staticmethod
    def _row(word, book_freq, ref_freq, g2, q, lr):
        n1, n2 = 10_000, 10_000
        return {"word": word, "book_freq": book_freq, "ref_freq": ref_freq,
                "book_rel": 1e6 * book_freq / n1,
                "ref_rel": 1e6 * ref_freq / n2,
                "log_likelihood": g2, "p_value": 1e-8, "q_value": q,
                "log_ratio": lr}

    def test_report_written_without_a_log_argument(self):
        key_lists = {c: [self._row("cadaver", 600, 3, 900.0, 1e-198, 7.4)]
                     for c in kn.CORPUS_DIRS}
        antikey_lists = {c: [self._row("the", 100, 900, -800.0, 1e-188, -2.2)]
                         for c in kn.CORPUS_DIRS}
        per_corpus = {c: {"files": 1, "tokens": 500_000, "types": 20_000}
                      for c in kn.CORPUS_DIRS}
        book_stats = {"parts": 80_363, "dropped": 593, "kept": 79_770,
                      "types": 3_683}
        cores = kn.reference_cores(per_corpus, key_lists)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "keyness_report.md"
            kn.write_report(out, book_stats, per_corpus, key_lists,
                            antikey_lists, cores, log=lambda m: None)
            text = out.read_text()
        for section in ("## Sizes", "## Key words", "## Anti-key words",
                        "## Cross-corpus stability"):
            self.assertIn(section, text)
        self.assertIn("key words | anti-key words", text)
        self.assertIn("cadaver", text)
        self.assertIn("Reference-independent core.", text)
        for c in kn.CORPUS_DIRS:
            self.assertIn(c, text)


class TestFullRun(unittest.TestCase):
    """The full run writes every expected artifact."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="keyness_test_")
        kn.main(["--out", cls.tmp])
        cls.out = Path(cls.tmp)

    def test_output_files(self):
        for corpus in kn.CORPUS_DIRS:
            self.assertTrue((self.out / f"keyness_{corpus}.tsv").exists(),
                            f"missing keyness_{corpus}.tsv")
        self.assertTrue((self.out / "keyness_report.md").exists())
        self.assertTrue((self.out / "summary.json").exists())

    def test_summary_json(self):
        s = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(s["schema"], kn.SCHEMA)
        self.assertIn("book", s)
        self.assertIn("corpora", s)
        for c in kn.CORPUS_DIRS:
            self.assertIn(c, s["corpora"])

    def test_report_has_key_and_antikey(self):
        text = (self.out / "keyness_report.md").read_text()
        self.assertIn("Key words", text)
        self.assertIn("Anti-key words", text)
        for c in kn.CORPUS_DIRS:
            self.assertIn(c, text)

    def test_report_reconciles_with_summary(self):
        """The Sizes table must agree with summary.json's token counts."""
        s = json.loads((self.out / "summary.json").read_text())
        text = (self.out / "keyness_report.md").read_text()
        for corpus, st in s["corpora"].items():
            self.assertIn(f"{st['tokens']:,}", text,
                          f"{corpus} token count missing from the report")

    def test_cross_corpus_section_present_and_data_driven(self):
        """The stability section is written and carries a real core.

        The corpus-size caveat must name the excerpt corpora and the
        cross-corpus table must flag them as non-inferential, so a reader
        cannot mistake a 50-token reference for a tested one.
        """
        text = (self.out / "keyness_report.md").read_text()
        self.assertIn("## Cross-corpus stability", text)
        self.assertIn("Reference-independent core.", text)
        self.assertIn("| cyberpunk | 105 | no (", text)
        for small in ("cyberpunk", "cutup_surrealist", "finnegans_wake"):
            self.assertIn(small, text)
        # Every inferential corpus contributes a full top-N list.
        for corpus in ("general_english", "medical", "cs", "legal"):
            self.assertIn(f"| {corpus} |", text)


if __name__ == "__main__":
    unittest.main()
