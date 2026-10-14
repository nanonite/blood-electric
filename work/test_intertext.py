#!/usr/bin/env python3
"""Regression suite for #57 intertextual n-gram matching.

Checks the invariants that matter:

  * the n-gram / overlap / shuffle primitives on known inputs,
  * the book-side screen reproduces #56's word-part counts exactly,
  * the acquisition ledger contains all five named works and ingests **none**
    of them (the no-silent-substitution invariant), and every matcher arm
    resolves to a real, lawfully available source,
  * the provenance of every ingested text is verified, not asserted: the
    embedded excerpt matches its pinned token count and sha256, every
    reference-corpus file matches #56's manifest sha256, and the author arm
    contains only the author's own interview turns,
  * the formula classifier assigns exactly one class per book formula and
    prefers the author over the influence and both over the registers,
  * a full (quick) build writes every artifact and reports the expected
    structure: idiolect formulas above the shuffle null, a thin genre arm,
    no named work ingested, and a deterministic rerun.

Reads only frozen artifacts and the cached reference corpora - no network, no
OCR.

    python work/test_intertext.py
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import intertext as it

ROOT = it.ROOT


class TestNgrams(unittest.TestCase):
    def test_counts_and_boundaries(self):
        c = it.ngrams(["a", "b", "c", "a", "b"], 2)
        self.assertEqual(c[("a", "b")], 2)
        self.assertEqual(c[("b", "c")], 1)
        self.assertEqual(c[("c", "a")], 1)
        self.assertEqual(sum(c.values()), 4)

    def test_unigram(self):
        c = it.ngrams(["a", "a", "b"], 1)
        self.assertEqual(c[("a",)], 2)
        self.assertEqual(c[("b",)], 1)

    def test_whole_sequence(self):
        self.assertEqual(it.ngrams(["a", "b", "c"], 3)[("a", "b", "c")], 1)

    def test_n_longer_than_seq_is_empty(self):
        self.assertEqual(len(it.ngrams(["a"], 3)), 0)

    def test_rejects_bad_n(self):
        with self.assertRaises(ValueError):
            it.ngrams(["a"], 0)


class TestShuffled(unittest.TestCase):
    def test_multiset_preserved(self):
        import random
        seq = list("mississippi")
        out = it.shuffled(seq, random.Random(1))
        self.assertCountEqual(out, seq)

    def test_deterministic(self):
        import random
        a = it.shuffled(list("abcdefghij"), random.Random(7))
        b = it.shuffled(list("abcdefghij"), random.Random(7))
        self.assertEqual(a, b)

    def test_long_sequence_actually_moves(self):
        import random
        seq = list(range(200))
        out = it.shuffled(seq, random.Random(3))
        self.assertNotEqual(out, seq)


class TestCountBookNgrams(unittest.TestCase):
    def test_counts_only_book_ngrams(self):
        book = {2: {("a", "b")}, 3: {("a", "b", "c")}}
        seqs = [["a", "b", "c", "a", "b"], ["x", "y"]]
        present, tokens = it.count_book_ngrams_in(book, seqs, ns=(2, 3))
        self.assertEqual(tokens, 7)
        self.assertEqual(present[2][("a", "b")], 2)
        self.assertEqual(present[3][("a", "b", "c")], 1)
        self.assertNotIn(("x", "y"), present[2])

    def test_boundaries_are_respected(self):
        # an n-gram cannot span two sequences/files
        book = {2: {("b", "c")}}
        present, _ = it.count_book_ngrams_in(book, [["a", "b"], ["c", "d"]],
                                             ns=(2,))
        self.assertEqual(sum(present[2].values()), 0)


class TestTokenise(unittest.TestCase):
    def test_operators_are_separators(self):
        toks = it._tokenise("gene=TV soul/gram <<corpse city>>")
        self.assertEqual(toks, ["gene", "tv", "soul", "gram", "corpse", "city"])

    def test_apostrophe_kept(self):
        self.assertEqual(it._tokenise("technojunkies' ill-treatment"),
                         ["technojunkies", "ill", "treatment"])

    def test_lowercased(self):
        self.assertEqual(it._tokenise("FUCKNAMLOAD"), ["fucknamload"])


class TestAcquisitionLedger(unittest.TestCase):
    def test_all_named_works_present(self):
        keys = {s.key for s in it.NAMED_WORKS}
        self.assertEqual(keys, {
            "siratori_headcode", "siratori_human_worms", "siratori_smart_d",
            "siratori_genedub", "siratori_nonexistence"})

    def test_no_named_work_is_ingestible(self):
        for s in it.NAMED_WORKS:
            self.assertIn(s.authorization, it.NON_INGESTIBLE,
                          f"named work {s.key} must not be ingestible")
            self.assertTrue(s.license, f"{s.key} lacks a licence/provenance note")

    def test_ingested_sources_are_lawful_and_attributed(self):
        ingestible = [s for s in it.SOURCES
                      if s.authorization not in it.NON_INGESTIBLE]
        self.assertTrue(ingestible)
        for s in ingestible:
            self.assertIn(s.authorization, (it.AUTH_OPEN, it.AUTH_EXCERPT),
                          f"{s.key} ingested under a non-open class")
            self.assertTrue(s.license and s.note,
                            f"{s.key} ingested without licence/provenance")

    def test_every_arm_resolves_to_a_source(self):
        for arm in it.ARMS:
            self.assertIn(arm.source_key, it.SOURCES_BY_KEY,
                          f"arm {arm.key} has no registered source")

    def test_same_author_probe_is_not_a_named_work(self):
        # the availability probe must not be mistaken for a name substitution
        arm = it.ARMS_BY_KEY["siratori_other"]
        src = it.SOURCES_BY_KEY[arm.source_key]
        self.assertNotEqual(src.kind, "named_work")
        self.assertEqual(src.authorization, it.AUTH_OPEN)


class TestBookSide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seqs, cls.stats = it.build_book(log=lambda m: None)

    def test_reconciles_with_keyness(self):
        import keyness
        _book, kstats = keyness.load_book_words(log=lambda m: None)
        self.assertEqual(self.stats["parts"], kstats["parts"])
        self.assertEqual(self.stats["tokens"], kstats["kept"])
        self.assertEqual(self.stats["types"], kstats["types"])

    def test_pages(self):
        self.assertEqual(self.stats["pages"], 225)

    def test_screen_removes_markup(self):
        self.assertGreater(self.stats["dropped"], 0)
        self.assertLess(self.stats["dropped"], self.stats["parts"])

    def test_token_total_equals_unigram_sum(self):
        total = sum(len(s) for s in self.seqs)
        self.assertEqual(total, self.stats["tokens"])


class TestGreatWorksExcerpt(unittest.TestCase):
    def test_short_and_attributed(self):
        toks = it._tokenise(it.SIRATORI_GREATWORKS_EXCERPT)
        self.assertGreater(len(toks), 100)
        self.assertLess(len(toks), 500, "excerpt must stay short (fair use)")

    def test_shares_signature_formulas_with_the_book(self):
        seqs, _stats = it.build_book(log=lambda m: None)
        book = {n: it.ngrams([t for s in seqs for t in s], n)
                for n in (2, 3)}
        present, _ = it.count_book_ngrams_in(
            {n: set(c) for n, c in book.items()},
            [it._tokenise(it.SIRATORI_GREATWORKS_EXCERPT)], ns=(2, 3))
        # the probe text is formulaically close to the book, not random
        self.assertGreater(len(present[2]), 20)
        self.assertGreater(len(present[3]), 5)


class TestParatextProvenance(unittest.TestCase):
    def test_interviewer_turn_is_excluded(self):
        author = it.paratext_rows()
        excluded = it.paratext_excluded_rows()
        self.assertTrue(author, "no author paratext rows")
        self.assertEqual([r["id"] for r in excluded], ["PT04"])
        ids = {r["id"] for r in author}
        self.assertNotIn("PT04", ids)
        self.assertEqual(len(author) + len(excluded), 6)

    def test_loaded_sequences_are_author_only(self):
        for r in it.paratext_rows():
            self.assertFalse(
                r.get("claim", "").strip().lower().startswith("an interviewer"),
                f"{r['id']} is the interviewer's turn")
        self.assertEqual(len(it.load_paratext_sequences()),
                         len(it.paratext_rows()))

    def test_both_interviews_are_registered(self):
        for key in ("interview_3am_nude_brain", "interview_rhizome_14416_wb"):
            self.assertIn(key, it.SOURCES_BY_KEY)
        # the live Rhizome site refused this agent; the ledger says so
        self.assertEqual(it.SOURCES_BY_KEY["interview_rhizome_14416"].verdict,
                         it.V_BLOCKED)


class TestCorpusProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = it.verify_corpus_provenance()

    def test_manifest_is_checked_file_by_file(self):
        self.assertTrue(it.CORPUS_MANIFEST.exists())
        manifest = it.kn.read_tsv(it.CORPUS_MANIFEST)
        self.assertEqual(len(self.rows), len(manifest))
        self.assertGreater(len(self.rows), 0)

    def test_every_file_matches_its_pinned_hash(self):
        bad = [r for r in self.rows if r["ok"] != "y"]
        self.assertEqual(bad, [], f"provenance mismatch: {bad}")

    def test_every_arm_corpus_is_covered_by_the_manifest(self):
        manifest_files = {r["item"].split(":", 1)[1] for r in self.rows}
        for arm_key, corpus in it._REFERENCE_ARM_CORPUS.items():
            files = it.kn.corpus_files(corpus)
            self.assertTrue(files, f"{corpus} has no files")
            for f in files:
                rel = f"{corpus}/{f.name}"
                self.assertIn(rel, manifest_files,
                              f"{arm_key}: {rel} not in the #56 manifest")

    def test_source_labels_match_the_corpus_contents(self):
        # a hard-coded file count in a source label must agree with reality
        n = len(it.kn.corpus_files("general_english"))
        title = it.SOURCES_BY_KEY["register_general_english"].title
        self.assertIn(f"{n} public-domain texts", title)


class TestGreatWorksProvenance(unittest.TestCase):
    def test_pinned_token_count(self):
        self.assertEqual(len(it._tokenise(it.SIRATORI_GREATWORKS_EXCERPT)),
                         it.GREATWORKS_EXCERPT_TOKENS)

    def test_pinned_sha256(self):
        self.assertEqual(it.sha256_text(it.SIRATORI_GREATWORKS_EXCERPT),
                         it.GREATWORKS_EXCERPT_SHA256)

    def test_url_matches_the_registered_source(self):
        self.assertEqual(it.SOURCES_BY_KEY["siratori_greatworks_ks"].url,
                         it.GREATWORKS_URL)

    def test_extractor_reproduces_the_embedded_text(self):
        # the re-extraction --probe performs against the live page, tested
        # offline on a page built from the embedded paragraphs themselves
        page = "".join(f'<p class="prose">"{p}"</p>'
                       for p in it.SIRATORI_GREATWORKS_EXCERPT.split("\n\n"))
        self.assertEqual(it.extract_greatworks_excerpt(page),
                         it.SIRATORI_GREATWORKS_EXCERPT)

    def test_extractor_unescapes_entities_and_strips_tags(self):
        page = ('<p class="prose">"a&amp;b&nbsp;c <b>d</b>"</p>'
                '<p class="prose">"second"</p>')
        self.assertEqual(it.extract_greatworks_excerpt(page),
                         "a&b c d\n\nsecond")


class TestFormulaClassifier(unittest.TestCase):
    def test_precedence(self):
        cases = {
            (True, False, False, False): "author_idiolect",
            (True, False, True, False): "author_idiolect",
            (False, True, False, False): "author_idiolect",
            (True, False, False, True): "author_common",
            (False, True, False, True): "author_common",
            (False, False, True, False): "genre_influence",
            (False, False, True, True): "influence_common",
            (False, False, False, True): "register_common",
            (False, False, False, False): "book_unique",
        }
        for args, want in cases.items():
            self.assertEqual(it.formula_class(*args), want, args)

    def test_class_list_matches_the_classifier_output(self):
        seen = {it.formula_class(*args) for args in
                [(a, b, c, d) for a in (0, 1) for b in (0, 1)
                 for c in (0, 1) for d in (0, 1)]}
        self.assertEqual(seen, set(it.FORMULA_CLASSES))


class TestBuildArtifacts(unittest.TestCase):
    """A real (quick) build into a scratch directory."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.orig = it.OUT
        it.OUT = Path(cls.tmp.name)
        it._SEQ_CACHE.clear()
        cls.summary = it.build(log=lambda m: None, quick=True)

    @classmethod
    def tearDownClass(cls):
        it.OUT = cls.orig
        it._SEQ_CACHE.clear()
        cls.tmp.cleanup()

    def test_all_artifacts_written(self):
        for name in ("sources.tsv", "availability_report.md",
                     "corpus_summary.tsv", "overlap_by_n.tsv",
                     "null_control.tsv", "formula_classes.tsv",
                     "shared_formulas.tsv", "reconciliation.tsv",
                     "diagnostics.tsv", "summary.json", "overlap_report.md"):
            self.assertTrue((it.OUT / name).exists(), f"missing {name}")

    def test_schema_and_no_named_work_ingested(self):
        self.assertEqual(self.summary["schema"], it.SCHEMA)
        self.assertEqual(self.summary["named_works_ingested"], 0)
        self.assertEqual(len(self.summary["named_works"]), 5)

    def test_idiolect_formulas_exist(self):
        self.assertGreater(self.summary["formula_classes"].get(
            "author_idiolect", 0), 0)

    def test_formula_table_row_count_matches_summary(self):
        rows = it.kn.read_tsv(it.OUT / "formula_classes.tsv")
        self.assertEqual(len(rows), sum(self.summary["formula_classes"].values()))
        for r in rows:
            self.assertGreaterEqual(int(r["n"]), 2)
            self.assertGreaterEqual(int(r["book_count"]), it.MIN_BOOK_FREQ)

    def test_same_author_above_null(self):
        rows = it.kn.read_tsv(it.OUT / "null_control.tsv")
        by = {(r["arm"], int(r["n"])): r for r in rows}
        r = by[("siratori_other", 3)]
        self.assertGreater(int(r["observed_instances"]), 0)
        self.assertGreaterEqual(int(r["observed_instances"]),
                                float(r["expected_instances"]))

    def test_influence_arms_are_thin(self):
        # at excerpt size the influence overlap must be small: no long shared
        # n-gram and few shared bigrams - a lower bound, not a genre finding.
        rows = it.kn.read_tsv(it.OUT / "shared_formulas.tsv")
        infl = [r for r in rows if r["kind"] == "influence"]
        self.assertTrue(infl, "influence arms must be exercised")
        self.assertEqual([r for r in infl if int(r["n"]) >= 3], [])

    def test_sources_ledger_marks_named_works_not_ingested(self):
        rows = it.kn.read_tsv(it.OUT / "sources.tsv")
        named = [r for r in rows if r["kind"] == "named_work"]
        self.assertEqual(len(named), 5)
        for r in named:
            self.assertEqual(r["ingested"], "n")

    def test_excerpt_matches_its_pin_in_summary(self):
        exc = self.summary["siratori_excerpt"]
        self.assertTrue(exc["matches_pin"])
        self.assertEqual(exc["sha256"], exc["pinned_sha256"])
        self.assertEqual(exc["tokens"], exc["pinned_tokens"])

    def test_summary_reports_all_corpus_files_verified(self):
        prov = self.summary["provenance"]
        self.assertGreater(prov["manifest_files_checked"], 0)
        self.assertEqual(prov["manifest_files_checked"],
                         prov["manifest_files_ok"])
        self.assertTrue(prov["all_corpus_files_match_manifest"])

    def test_reconciliation_has_no_failures(self):
        rows = it.kn.read_tsv(it.OUT / "reconciliation.tsv")
        self.assertTrue(rows)
        bad = [r for r in rows if r["ok"] == "n"]
        self.assertEqual(bad, [], f"reconciliation failures: {bad}")
        # every manifest file is present as a provenance row
        corpus_rows = [r for r in rows
                       if r["item"].startswith("corpus_file:")]
        self.assertEqual(len(corpus_rows),
                         self.summary["provenance"]["manifest_files_checked"])

    def test_diagnostics_record_provenance_and_paratext_filter(self):
        rows = {r["check"]: r for r in
                it.kn.read_tsv(it.OUT / "diagnostics.tsv")}
        self.assertEqual(rows["named_works_ingested"]["value"], "0")
        self.assertEqual(float(rows["corpus_files_provenance_ok"]["value"]),
                         self.summary["provenance"]["manifest_files_checked"])
        self.assertEqual(
            float(rows["paratext_interviewer_turns_excluded"]["value"]),
            self.summary["paratext"]["interviewer_turns_excluded"])

    def test_formulas_partition_into_the_declared_classes(self):
        rows = it.kn.read_tsv(it.OUT / "formula_classes.tsv")
        self.assertTrue(rows)
        for r in rows:
            self.assertIn(r["class"], it.FORMULA_CLASSES)
        # each class agrees with the flags recorded on the row
        for r in rows:
            want = it.formula_class(
                r["in_siratori_other"] == "y",
                r["in_author_paratext"] == "y",
                bool(r["influences"]), bool(r["registers"]))
            self.assertEqual(r["class"], want, r["phrase"])
            if r["class"] != "book_unique":
                self.assertTrue(r["shared_arms"],
                                f"{r['phrase']} classified {r['class']} "
                                "but names no shared arm")
            else:
                self.assertEqual(r["shared_arms"], "")


class TestDeterminism(unittest.TestCase):
    """A quick build is byte-identical on a rerun (no wall-clock, fixed seed)."""

    ARTIFACTS = ("sources.tsv", "corpus_summary.tsv", "overlap_by_n.tsv",
                 "null_control.tsv", "formula_classes.tsv",
                 "shared_formulas.tsv", "reconciliation.tsv",
                 "diagnostics.tsv", "summary.json", "availability_report.md",
                 "overlap_report.md")

    def test_two_quick_builds_are_identical(self):
        orig = it.OUT
        digests = []
        try:
            for _ in range(2):
                with tempfile.TemporaryDirectory() as tmp:
                    it.OUT = Path(tmp)
                    it._SEQ_CACHE.clear()
                    it.build(log=lambda m: None, quick=True)
                    digests.append(
                        {name: hashlib.sha256(
                            (Path(tmp) / name).read_bytes()).hexdigest()
                         for name in self.ARTIFACTS})
        finally:
            it.OUT = orig
            it._SEQ_CACHE.clear()
        for name in self.ARTIFACTS:
            self.assertEqual(digests[0][name], digests[1][name],
                             f"{name} is not deterministic")


class TestPublishedArtifacts(unittest.TestCase):
    """If the published report exists, its summary must be self-consistent."""

    def test_published_summary(self):
        p = ROOT / "analysis" / "intertext" / "summary.json"
        if not p.exists():
            self.skipTest("analysis/intertext not built yet")
        s = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(s["schema"], it.SCHEMA)
        self.assertEqual(s["named_works_ingested"], 0)
        rows = it.kn.read_tsv(ROOT / "analysis" / "intertext"
                              / "formula_classes.tsv")
        self.assertEqual(len(rows), sum(s["formula_classes"].values()))
        # the idiolect signal must be present, not a null report
        self.assertGreater(s["formula_classes"].get("author_idiolect", 0), 0)

    def test_published_provenance_and_pin(self):
        p = ROOT / "analysis" / "intertext" / "summary.json"
        if not p.exists():
            self.skipTest("analysis/intertext not built yet")
        s = json.loads(p.read_text(encoding="utf-8"))
        self.assertTrue(s["siratori_excerpt"]["matches_pin"])
        self.assertEqual(s["siratori_excerpt"]["sha256"],
                         it.GREATWORKS_EXCERPT_SHA256)
        self.assertTrue(s["provenance"]["all_corpus_files_match_manifest"])
        recon = it.kn.read_tsv(ROOT / "analysis" / "intertext"
                               / "reconciliation.tsv")
        self.assertEqual([r for r in recon if r["ok"] == "n"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
