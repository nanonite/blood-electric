#!/usr/bin/env python3
"""Regression suite for the interlinear edition generator (#62).

Run the whole work/ suite with:
    python -m unittest discover -s work -p 'test_*.py'

The suite is two-layered, like the workspace's other test suites:

  * **synthetic tests** on hand-built tokens and contexts, which pin the
    rendering, the reading rule order, the closure arithmetic and the
    curated-table invariants -- and which prove the *failure* paths have
    teeth: a token bucket that does not close, a normalization whose two
    paths disagree, a reversed page range and a mixed-case reading key
    all have to fail the build loudly;
  * **corpus tests** (guarded by `skipUnless`), which build the edition
    twice into temporary directories (determinism, byte-identical) and
    re-derive every number it publishes against the artifacts that own
    it (#7/#13/#17/#19/#26/#30/#33/#56), including the two-clause
    continuity check against #56's glossed passages.

`build()` is exercised into temporary directories only, never into
`analysis/interlinear/`.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))
import interlinear_edition as ie


HAVE_ARTIFACTS = all(p.exists() for p in (
    ie.CLAUSES, ie.TOKENS, ie.ENTITY, ie.TERMBASE, ie.KWIC))


# ---------------------------------------------------------------------------
# synthetic helpers
# ---------------------------------------------------------------------------

def tok(kind, text, start, end, gap="", page=1, glue_L="", glue_R=""):
    """A #17 token row, the shape `tokens.tsv` carries."""
    return {"page": str(page), "region": "body", "item": "-1", "hlevel": "0",
            "kind": kind, "start": str(start), "end": str(end),
            "text": text, "text_json": json.dumps(text),
            "gap_json": json.dumps(gap), "glue_L": glue_L, "glue_R": glue_R,
            "upper": "", "num": "", "has_digit": "", "caps": "",
            "left_op": "", "pclass": "", "unit": "", "upos": "",
            "ast_path": "", "container": ""}


def fake_context(screening=None, entity=None, kwic=None, page_units=None,
                 unit_meta=None):
    """The `load_context()` tuple from hand-built pieces."""
    return ({"1": "the new Japanese cyberpunk classic\n\nkenji siratori"},
            {1: [(0, 100, 1)]} if page_units is None else page_units,
            {1: (1, 0, 100, "soul/gram")} if unit_meta is None else unit_meta,
            lambda uid: False,                        # is_markup
            screening or {},                          # #19 screen
            {1: []},                                  # markup spans
            entity or {},                             # #30 glossary
            {},                                       # #56 termbase
            kwic or {})                               # #26 KWIC


# ---------------------------------------------------------------------------
# CLI / helpers
# ---------------------------------------------------------------------------

class TestParsePages(unittest.TestCase):
    def test_single_page(self):
        self.assertEqual(ie.parse_pages("4"), (4, 4))

    def test_range(self):
        self.assertEqual(ie.parse_pages("4-6"), (4, 6))

    def test_none_is_the_whole_book(self):
        self.assertIsNone(ie.parse_pages(None))
        self.assertIsNone(ie.parse_pages(""))

    def test_reversed_range_fails(self):
        with self.assertRaises(SystemExit):
            ie.parse_pages("6-4")

    def test_garbage_fails(self):
        with self.assertRaises(SystemExit):
            ie.parse_pages("four")


class TestRendering(unittest.TestCase):
    """The #13 v3/v4 semantics, with the #56 token shape."""

    def test_render_clause_matches_the_56_semantics(self):
        toks = [
            tok("WORD", "The", 0, 3, gap=""),
            tok("OP", "//", 3, 5, gap=" "),
            tok("WORD", "soul", 5, 9, gap=""),
            tok("OP", "/", 9, 10, gap=""),
            tok("WORD", "gram", 10, 14, gap=""),
        ]
        self.assertEqual(ie.render_clause(toks, "v3"), "The soul gram")
        self.assertEqual(ie.render_clause(toks, "v4"), "The. soul. gram")

    def test_punctuation_is_kept_on_both_lines(self):
        toks = [
            tok("WORD", "www", 0, 3, gap=""),
            tok("PUNCT", ".", 3, 4, gap=""),
            tok("WORD", "creationbooks", 4, 17, gap=""),
            tok("PUNCT", ".", 17, 18, gap=""),
            tok("WORD", "com", 18, 21, gap=""),
        ]
        # the URL must not be split, on either line
        self.assertEqual(ie.render_clause(toks, "v4"),
                         "www.creationbooks.com")

    def test_layout_comments_are_scaffolding_not_text(self):
        toks = [
            tok("WORD", "word", 40, 44,
                gap="\n\n<!-- figure/picture block (no text) -->\n\n"),
        ]
        rendered = ie.render_clause(toks, "v4")
        self.assertEqual(rendered, "word")
        self.assertNotIn("figure", rendered)

    def test_the_two_normalization_paths_agree(self):
        # the build-time invariant, on a clause with operators, glued
        # compounds, punctuation and a markup span
        original = ("the suck=blood of boy-roid ++ <<th> the [ice nebula] "
                    "larvae")
        kept = [
            tok("WORD", "the", 0, 3, gap=""),
            tok("WORD", "suck", 4, 8, gap=" "),
            tok("OP", "=", 8, 9, gap="", glue_L="x", glue_R="x"),
            tok("WORD", "blood", 9, 14, gap=""),
            tok("WORD", "of", 15, 17, gap=" "),
            tok("WORD", "boy-roid", 18, 26, gap=" "),
            tok("OP", "++", 27, 29, gap=" ", glue_L="x", glue_R="x"),
            tok("OP", "<<", 30, 32, gap=" ", glue_L="x", glue_R="x"),
            tok("WORD", "th", 32, 34, gap=""),
            tok("OP", ">", 34, 35, gap=""),
            tok("WORD", "the", 36, 39, gap=" "),
            tok("PUNCT", "[", 40, 41, gap=" "),
            tok("WORD", "ice", 41, 44, gap=""),
            tok("WORD", "nebula", 45, 51, gap=" "),
            tok("PUNCT", "]", 51, 52, gap=""),
            tok("WORD", "larvae", 53, 59, gap=" "),
        ]
        excluded = [kept[7], kept[8], kept[9]]      # the `<<th>` tag run
        clean = [t for t in kept if t not in excluded]
        silenced = [t for t in kept if t["kind"] == "OP"
                    or t in excluded]

        def spans(mode):
            r = ". " if mode == "v4" else " "
            out = []
            for t in silenced:
                repl = "" if t in excluded else (
                    r if t["kind"] == "OP" else None)
                if repl is None:
                    continue
                out.append((int(t["start"]), int(t["end"]), repl))
            return sorted(out)

        for mode in ("v3", "v4"):
            self.assertEqual(
                ie.render_clause(clean, mode),
                ie.normalized_from_original(original, spans(mode), 0, mode),
                "paths disagree in %s" % mode)


class TestTokenReading(unittest.TestCase):
    """The published rule order, one branch per surface class."""

    def reading(self, t, ctx=None):
        return ie.token_reading(t, ctx or fake_context())

    def test_operator_is_silence_with_a_12_14_reading(self):
        g, c, s, note = self.reading(tok("OP", "//", 0, 2))
        self.assertEqual((g, c, s), ("", "high", "silence"))
        self.assertIn("sentence boundary", note)

    def test_slash_reading_quotes_the_measured_roles(self):
        _g, _c, _s, note = self.reading(tok("OP", "/", 0, 1))
        self.assertIn("952", note)
        self.assertIn("540", note)

    def test_punctuation_survives_on_the_gloss_line(self):
        g, c, s, note = self.reading(tok("PUNCT", ",", 0, 1))
        self.assertEqual((g, c, s), (",", "high", "silence"))
        self.assertIn("ordinary English", note)

    def test_curated_reading_substitutes_and_carries_case(self):
        ctx = fake_context(screening={"feti": {"category": "true_coinage"}})
        g, c, s, _note = self.reading(tok("WORD", "feti", 0, 4), ctx)
        self.assertEqual((g, c, s), ("fetuses", "high", "plain_reading"))
        g, _c, _s, _n = self.reading(tok("WORD", "Feti", 0, 4), ctx)
        self.assertEqual(g, "Fetuses")
        g, _c, _s, _n = self.reading(tok("WORD", "FETI", 0, 4), ctx)
        self.assertEqual(g, "FETUSES")

    def test_function_word_is_its_own_gloss(self):
        ctx = fake_context(screening={"the": {"category": "english"}})
        g, c, s, note = self.reading(tok("WORD", "the", 0, 3), ctx)
        self.assertEqual((g, c, s), ("the", "high", "identity_19"))
        self.assertIn("function word", note)

    def test_english_content_word_is_its_own_gloss(self):
        ctx = fake_context(screening={"murder": {"category": "english"}})
        g, c, s, note = self.reading(tok("WORD", "murder", 0, 6), ctx)
        self.assertEqual((g, c, s), ("murder", "high", "identity_19"))
        self.assertIn("content word", note)

    def test_rare_english_and_numeric_are_identity(self):
        ctx = fake_context(screening={
            "flensing": {"category": "rare_english"},
            "666": {"category": "numeric"}})
        self.assertEqual(self.reading(tok("WORD", "flensing", 0, 8), ctx)[:3],
                         ("flensing", "high", "identity_19"))
        self.assertEqual(self.reading(tok("WORD", "666", 0, 3), ctx)[:3],
                         ("666", "high", "identity_numeric"))

    def test_compound_is_kept_with_its_parts(self):
        ctx = fake_context(screening={
            "boy-roid": {"category": "compound", "morph": "boy+roid"}})
        g, c, s, note = self.reading(tok("WORD", "boy-roid", 0, 8), ctx)
        self.assertEqual((g, c, s), ("boy-roid", "medium", "compound_19"))
        self.assertIn("boy+roid", note)

    def test_proper_noun_is_not_translated_but_carries_the_30_role(self):
        ctx = fake_context(screening={
            "tokage": {"category": "proper_noun"}},
            entity={"tokage": {"role": "persona", "does_verbs": "is|was",
                               "done_to_verbs": "love|speed"}})
        g, c, s, note = self.reading(tok("WORD", "ToKAGE", 0, 6), ctx)
        self.assertEqual((g, c, s), ("ToKAGE", "medium", "entity_30"))
        self.assertIn("#30 role: persona", note)
        self.assertIn("does: is|was", note)

    def test_ocr_error_is_kept_not_corrected(self):
        ctx = fake_context(screening={"lt": {"category": "ocr_error"}})
        g, c, s, note = self.reading(tok("WORD", "lt", 0, 2), ctx)
        self.assertEqual((g, c, s), ("lt", "low", "ocr_error_19"))
        self.assertIn("never", note)

    def test_coinage_without_a_reading_is_marked_and_unresolved(self):
        ctx = fake_context(screening={"omanko": {"category": "true_coinage"}})
        g, c, s, note = self.reading(tok("WORD", "omanko", 0, 6), ctx)
        self.assertEqual((g, c, s), ("\u2039omanko\u203a", "unresolved",
                                    "unresolved"))
        self.assertIn("residue", note)

    def test_junk_is_unresolved(self):
        ctx = fake_context(screening={"href": {"category": "junk"}})
        g, c, s, _note = self.reading(tok("WORD", "href", 0, 4), ctx)
        self.assertEqual((g, c, s), ("", "unresolved", "junk_19"))

    def test_word_part_of_a_unit_reads_the_containing_unit(self):
        ctx = fake_context(screening={
            "soul/gram": {"category": "compound", "morph": "soul+gram"}})
        g, c, s, note = self.reading(tok("WORD", "gram", 10, 14), ctx)
        self.assertEqual((g, c, s), ("gram", "medium", "unit_context_17"))
        self.assertIn("word-part of the compound unit `soul/gram`", note)

    def test_unknown_surface_with_no_unit_is_unresolved(self):
        ctx = fake_context(page_units={1: []}, unit_meta={})
        g, c, s, _note = self.reading(tok("WORD", "qwerty", 0, 6), ctx)
        self.assertEqual((g, c, s),
                         ("\u2039qwerty\u203a", "unresolved", "unresolved"))

    def test_every_branch_returns_one_confidence(self):
        surfaces = [
            ("OP", "//"), ("PUNCT", "."),
            ("WORD", "feti"), ("WORD", "the"), ("WORD", "murder"),
            ("WORD", "flensing"), ("WORD", "666"), ("WORD", "boy-roid"),
            ("WORD", "ToKAGE"), ("WORD", "lt"), ("WORD", "omanko"),
            ("WORD", "href"), ("WORD", "gram"), ("WORD", "qwerty"),
        ]
        ctx = fake_context(screening={
            "feti": {"category": "true_coinage"},
            "the": {"category": "english"},
            "murder": {"category": "english"},
            "flensing": {"category": "rare_english"},
            "666": {"category": "numeric"},
            "boy-roid": {"category": "compound", "morph": "boy+roid"},
            "tokage": {"category": "proper_noun"},
            "lt": {"category": "ocr_error"},
            "omanko": {"category": "true_coinage"},
            "href": {"category": "junk"},
            "soul/gram": {"category": "compound", "morph": "soul+gram"}})
        for kind, text in surfaces:
            g, c, s, note = ie.token_reading(tok(kind, text, 0, len(text)),
                                             ctx)
            self.assertIn(c, ie.CONFIDENCES, (kind, text, c))
            self.assertIn(s, ie.SOURCE_LABEL, (kind, text, s))
            self.assertIsInstance(note, str)


class TestCuratedTable(unittest.TestCase):
    def test_keys_are_lower_case(self):
        bad = [k for k in ie.PLAIN_READINGS if k != k.lower()]
        self.assertEqual(bad, [], "the lookup is case-folded")

    def test_confidences_are_from_the_56_scheme(self):
        bad = {k: v[1] for k, v in ie.PLAIN_READINGS.items()
               if v[1] not in ie.CONFIDENCES}
        self.assertEqual(bad, {})

    def test_readings_are_non_empty(self):
        bad = [k for k, v in ie.PLAIN_READINGS.items() if not v[0].strip()]
        self.assertEqual(bad, [])

    def test_unresolved_marker_is_tsv_safe(self):
        marked = ie.mark_unresolved("omanko")
        self.assertEqual(marked, "\u2039omanko\u203a")
        for ch in ("\t", "\n", "\r"):
            self.assertNotIn(ch, marked)


class TestDerivedTables(unittest.TestCase):
    def ctx(self):
        return fake_context(
            screening={"feti": {"unit": "feti", "category": "true_coinage",
                                "n": "88", "n_pages": "38",
                                "first_page": "18", "morph": "feti"},
                       "tokage": {"unit": "ToKAGE", "category": "proper_noun",
                                  "n": "127", "n_pages": "84",
                                  "first_page": "4", "morph": "ToKAGE"}},
            entity={"tokage": {"role": "persona", "does_verbs": "is",
                               "done_to_verbs": "love"}},
            kwic={"feti": ["of the city-mimic//:: [feti] //accelerates"]})

    def test_lexicon_counts_come_from_the_screen(self):
        rows = ie.build_lexicon(self.ctx())
        feti = next(r for r in rows if r["surface"] == "feti")
        self.assertEqual(feti["reading"], "fetuses")
        self.assertEqual(feti["confidence"], "high")
        self.assertEqual(feti["n_occurrences"], "88")
        self.assertEqual(feti["n_pages"], "38")
        self.assertEqual(feti["first_page"], "18")
        self.assertIn("[feti]", feti["kwic_examples_26"])
        tokage = next(r for r in rows if r["surface"] == "ToKAGE")
        self.assertEqual(tokage["source"], "entity_30")
        self.assertEqual(tokage["entity_role_30"], "persona")

    def test_lexicon_is_sorted_by_frequency_then_surface(self):
        rows = ie.build_lexicon(self.ctx())
        keys = [(-int(r["n_occurrences"] or 0), r["surface"].lower())
                for r in rows]
        self.assertEqual(keys, sorted(keys))

    def test_residue_itemizes_unresolved_and_ocr(self):
        token_rows = [
            {"surface": "omanko", "kind": "WORD", "page": 108,
             "source": "unresolved", "confidence": "unresolved"},
            {"surface": "omanko", "kind": "WORD", "page": 129,
             "source": "unresolved", "confidence": "unresolved"},
            {"surface": "lt", "kind": "WORD", "page": 5,
             "source": "ocr_error_19", "confidence": "low"},
        ]
        rows = ie.build_residue(self.ctx(), token_rows)
        unres = next(r for r in rows if r["class"] == "unresolved_unresolved")
        self.assertEqual((unres["item"], unres["n"]), ("omanko", 2))
        self.assertEqual(unres["pages"], [108, 129])
        self.assertIn("not repaired", unres["verdict"])
        ocr = next(r for r in rows if r["class"] == "ocr_error_kept")
        self.assertEqual(ocr["item"], "lt")
        self.assertIn("never auto-corrected", ocr["verdict"])

    def test_operator_table_counts_and_reads(self):
        toks = {1: [tok("OP", "//", 0, 2, glue_L="x", glue_R="x"),
                    tok("OP", "//", 2, 4),
                    tok("WORD", "a", 5, 6)]}
        rows, total = ie.build_operator_rows(toks)
        self.assertEqual(total, 2)
        self.assertEqual(rows[0]["operator"], "//")
        self.assertEqual(rows[0]["n_tokens"], 2)
        self.assertEqual(rows[0]["n_glued_both_sides"], 1)
        self.assertIn("sentence boundary", rows[0]["reading_12_14"])

    def test_operator_table_has_a_reading_for_every_frozen_operator(self):
        toks = {1: [tok("OP", s, 0, len(s)) for s in
                    ("::", "//", "=", "/", ">", "<<", ">>", "<", "++", "|",
                     "+")]}
        rows, _total = ie.build_operator_rows(toks)
        for r in rows:
            self.assertNotEqual(r["reading_12_14"], "operator (silence)",
                                "%s has no reading" % r["operator"])


class TestPickSamples(unittest.TestCase):
    def records(self, flags):
        return [{"page": p, "clause_id": str(i), "n_readings": r,
                 "n_unresolved": u}
                for i, (p, r, u) in enumerate(flags)]

    def test_prefers_content_bearing_clauses(self):
        recs = self.records([(1, 0, 0), (1, 1, 0), (2, 0, 0), (5, 0, 1),
                             (9, 1, 0), (9, 0, 0)])
        picks = ie.pick_samples(recs)
        self.assertEqual([p["n_readings"] + p["n_unresolved"] for p in picks],
                         [1, 1, 1])

    def test_falls_back_when_nothing_is_content_bearing(self):
        recs = self.records([(1, 0, 0), (5, 0, 0), (9, 0, 0)])
        picks = ie.pick_samples(recs)
        self.assertEqual(len(picks), 3)

    def test_empty(self):
        self.assertEqual(ie.pick_samples([]), [])


class TestBuildFailurePaths(unittest.TestCase):
    """The closure checks are the point of the suite: a bucket that does
    not close and a normalization whose two paths disagree must fail the
    build loudly rather than publish a wrong edition."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="interlinear-fail-")
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    @unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts not present")
    def test_an_unclosed_token_bucket_fails_the_build(self):
        real = ie.clause_record

        def doctored(row, toks_by_page, ctx):
            rec, toks = real(row, toks_by_page, ctx)
            rec["n_high"] += 1          # one token classified twice
            return rec, toks

        with mock.patch.object(ie, "clause_record", doctored):
            with self.assertRaises(ie.InterlinearError):
                ie.build(self.out, pages=(4, 4), quiet=True)

    @unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts not present")
    def test_a_disagreeing_normalization_fails_the_build(self):
        real = ie.clause_record

        def doctored(row, toks_by_page, ctx):
            rec, toks = real(row, toks_by_page, ctx)
            rec["norm_match"] = False
            return rec, toks

        with mock.patch.object(ie, "clause_record", doctored):
            with self.assertRaises(ie.InterlinearError):
                ie.build(self.out, pages=(4, 4), quiet=True)


# ---------------------------------------------------------------------------
# corpus tests (the real book)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts not present")
class CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="interlinear-test-")
        cls.summary = ie.build(Path(cls._tmp.name) / "full", quiet=True)
        cls.out = Path(cls._tmp.name) / "full"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def rows(self, name):
        with (self.out / name).open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            return [dict(zip(header, line.rstrip("\n").split("\t")))
                    for line in fh if line.strip()]

    # -- artifact set ------------------------------------------------------

    def test_every_artifact_is_written(self):
        for name in ("interlinear.md", "interlinear.tsv", "gloss_tokens.tsv",
                     "gloss_lexicon.tsv", "residue.tsv", "per_page.tsv",
                     "operator_roles.tsv", "reconciliation.tsv",
                     "diagnostics.tsv", "summary.json",
                     "interlinear_report.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_summary_contract(self):
        s = self.summary
        self.assertEqual(s["schema"], "interlinear/1")
        self.assertEqual(s["clauses"], 5614)
        self.assertEqual(s["pages_rendered"], 225)
        self.assertEqual(s["tokens_total"], 90220)          # #17
        self.assertEqual(s["operator_tokens"], 11000)       # #17
        self.assertEqual(s["word_tokens_in_clauses"], 76897)  # #33
        self.assertEqual(
            s["tokens_in_clauses"] + s["tokens_outside_clauses"],
            s["tokens_total"])
        # the confidence buckets cover the kept tokens; the markup-
        # excluded remainder is the difference to the contained total
        self.assertEqual(
            sum(s["confidence"].values()) + s["markup_excluded_tokens"],
            s["tokens_in_clauses"])

    def test_reconciliation_matches_everywhere_but_the_disclosed_row(self):
        rows = self.rows("reconciliation.tsv")
        self.assertTrue(rows)
        for r in rows:
            if r["match"] == "disclosed":
                continue
            self.assertEqual(r["match"], "true",
                             "%s: %s vs %s" % (r["quantity"], r["earlier"],
                                               r["here"]))
        quantities = {r["quantity"] for r in rows}
        self.assertIn("word tokens inside clauses (#33 n_words_clean)",
                      quantities)

    # -- determinism -------------------------------------------------------

    def test_two_builds_are_byte_identical(self):
        second = Path(self._tmp.name) / "second"
        ie.build(second, quiet=True)
        for name in ("interlinear.tsv", "gloss_tokens.tsv",
                     "gloss_lexicon.tsv", "residue.tsv", "per_page.tsv",
                     "operator_roles.tsv", "reconciliation.tsv"):
            self.assertEqual((self.out / name).read_bytes(),
                             (second / name).read_bytes(), name)
        # the report and summary differ only in the generated stamp
        rep_a = (self.out / "interlinear_report.md").read_text()
        rep_b = (second / "interlinear_report.md").read_text()
        self.assertEqual(rep_a.split("Generated: ")[1::2],
                         rep_b.split("Generated: ")[1::2])

    def test_slice_build_reproduces_the_full_build_slice(self):
        sl = Path(self._tmp.name) / "slice"
        ie.build(sl, pages=(4, 6), quiet=True)
        with (sl / "interlinear.tsv").open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            sliced = [line.rstrip("\n") for line in fh if line.strip()]
        full = [ "\t".join(r[h] for h in header)
                 for r in self.rows("interlinear.tsv")
                 if 4 <= int(r["page"]) <= 6 ]
        self.assertEqual(sliced, full)
        self.assertTrue(sliced)

    # -- the three lines on known clauses ----------------------------------

    def test_the_56_passage_renders_identically(self):
        """#56's glossed passage 1 (p. 4) is two #33 clauses; the interlinear
        must reproduce #56's v3/v4 lines for them exactly."""
        rows = {r["clause_id"]: r for r in self.rows("interlinear.tsv")}
        c10, c11 = rows["10"], rows["11"]
        self.assertEqual(c10["page"], "4")
        self.assertEqual(c10["v3_space"],
                         "I suck the porno nerve gas of boy-roid")
        self.assertEqual(c10["gloss"],
                         "I suck the porno nerve gas of boy-roid")
        self.assertEqual(c11["original"],
                         "The soul/gram of self crushes to the quantum "
                         "tragedy of NDRO")
        self.assertEqual(c11["v4_period"],
                         "The soul. gram of self crushes to the quantum "
                         "tragedy of NDRO")

    def test_a_reading_substitution_lands_on_the_gloss_line_only(self):
        rows = {r["clause_id"]: r for r in self.rows("interlinear.tsv")}
        c = rows["505"]
        self.assertEqual((c["original"], c["v3_space"], c["v4_period"]),
                         ("feti", "feti", "feti"))
        self.assertEqual(c["gloss"], "fetuses")

    def test_an_unread_surface_is_marked_and_itemized(self):
        rows = {r["clause_id"]: r for r in self.rows("interlinear.tsv")}
        c = rows["2789"]
        self.assertIn("\u2039omanko\u203a", c["gloss"])
        self.assertEqual(c["unresolved"], "omanko x1")
        residue = {r["item"] for r in self.rows("residue.tsv")}
        self.assertIn("omanko", residue)
        self.assertIn("lt", residue)          # the ocr_error class, kept

    def test_every_clause_has_three_lines(self):
        for r in self.rows("interlinear.tsv"):
            for field in ("original", "v4_period", "gloss"):
                self.assertNotIn("\t", r[field])
                self.assertNotIn("\n", r[field])
            self.assertTrue(r["gloss"], r["clause_id"])

    def test_per_page_rows_close_to_the_clause_table(self):
        per_page = self.rows("per_page.tsv")
        self.assertEqual(len(per_page), 225)
        self.assertEqual(sum(int(r["n_clauses"]) for r in per_page), 5614)
        self.assertEqual(sum(int(r["n_words"]) for r in per_page), 76897)

    def test_token_rows_close_over_the_kept_stream(self):
        toks = self.rows("gloss_tokens.tsv")
        conf = {}
        for t in toks:
            conf[t["confidence"]] = conf.get(t["confidence"], 0) + 1
        self.assertEqual(sum(conf.values()), len(toks))
        markup = sum(int(r["n_markup"]) for r in
                     self.rows("per_page.tsv"))
        self.assertEqual(len(toks) + markup, 84306)


if __name__ == "__main__":
    unittest.main()
