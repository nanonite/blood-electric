#!/usr/bin/env python3
"""Regression suite for the coverage accounting / residue ledger (#70).

Run the whole work/ suite with:
    python -m unittest discover -s work -p 'test_*.py'

The suite is two-layered, like #17's:

  * **synthetic tests** on hand-built corpora, which pin the ledger's rules
    and -- most importantly -- prove that the residue detector has teeth: a
    planted unexplained character, a planted unscreened unit surface and a
    planted consumer split all have to be caught.  A closure metric that
    cannot fail is worthless, so these tests are the point of the suite;
  * **corpus tests** (guarded by `skipUnless`), which re-derive every number
    the ledger publishes and pin it against the issue that owns it (#7, #14,
    #17, #19, #27, #47).

`build()` is exercised twice into temporary directories to check
determinism, so the artifact set is verified end to end without touching
`analysis/coverage_ledger/`.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))
import coverage_ledger as cl
import operator_tokenizer as T

ART = ROOT / "analysis" / "coverage_ledger"
HAVE_ARTIFACTS = all(p.exists() for p in (
    cl.LAYOUT, cl.TOKENS, cl.TAILS, cl.UNITS, cl.SCREENING, cl.MARKUP))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ledger_for(body, page=1):
    """(char Counter, token list) for a bare page body."""
    toks, _diags, _tail = T.tokenize_page(page, body)
    return cl.char_map(body, toks, T.comment_spans(body)), toks


def buckets_of(body, page=1):
    counts = {}
    for label in cl.char_map(body, *T.tokenize_page(page, body)[:1],
                             T.comment_spans(body)):
        counts[label] = counts.get(label, 0) + 1
    return counts


def screening_row(unit, category="english"):
    return {"unit": unit, "category": category, "n": "1", "first_page": "1"}


# ---------------------------------------------------------------------------
class TestEscaping(unittest.TestCase):
    """#19 stores two surfaces with one more backslash than #17 does."""

    def test_unescape_reverses_one_pass(self):
        self.assertEqual(cl.unescape("plain"), "plain")
        self.assertEqual(cl.unescape(r"a\tb"), "a\tb")
        self.assertEqual(cl.unescape(r"eruption=\njunk"), "eruption=\njunk")
        self.assertEqual(cl.unescape(r"a\\nb"), "a\\nb")
        self.assertEqual(cl.unescape(r"a\\\\nb"), "a\\\\nb")

    def test_surface_join_realigns_the_two_conventions(self):
        # #17 writes the newline-bearing surface once-escaped; #19 writes it
        # with the backslash escaped as well.
        units_surface = "eruption=" + "\\" + "njunk"
        screening_surface = "eruption=" + "\\\\" + "njunk"
        self.assertNotEqual(units_surface, screening_surface)
        index, mismatched = cl.screening_index(
            [screening_row(screening_surface, "compound")])
        self.assertEqual(mismatched, [screening_surface])
        self.assertIs(index[units_surface]["category"],
                      index[screening_surface]["category"])
        self.assertEqual(index[units_surface]["category"], "compound")

    def test_plain_surfaces_are_untouched(self):
        index, mismatched = cl.screening_index(
            [screening_row("boy-roid"), screening_row("gene=TV")])
        self.assertEqual(mismatched, [])
        self.assertEqual(set(index), {"boy-roid", "gene=TV"})


# ---------------------------------------------------------------------------
class TestCharLedger(unittest.TestCase):
    """L1: every character of a page body lands in a named bucket."""

    def test_buckets_close_on_a_synthetic_page(self):
        body = "the suck=blood chromosome::of the dog//\n<<a block>> here"
        counts = buckets_of(body)
        self.assertEqual(sum(counts.values()), len(body))
        self.assertEqual(counts.get("gap_other", 0), 0)
        self.assertGreater(counts["token_word"], 0)
        self.assertGreater(counts["gap_whitespace"], 0)

    def test_tokens_are_bucketed_by_kind(self):
        counts = buckets_of("gene=TV//soul/gram: x")
        self.assertEqual(counts["token_word"], len("gene") + len("TV")
                         + len("soul") + len("gram") + len("x"))
        self.assertEqual(counts["token_op"], len("=") + len("//") + len("/"))
        self.assertEqual(counts["token_punct_colon"], 1)

    def test_layout_comments_win_over_tokens(self):
        body = "<!-- figure/picture block (no text) -->word"
        counts = buckets_of(body)
        self.assertEqual(counts["layout_comment"],
                         len("<!-- figure/picture block (no text) -->"))
        self.assertEqual(counts["token_word"], 4)
        self.assertEqual(sum(counts.values()), len(body))

    def test_planted_residue_lands_in_gap_other(self):
        # The teeth of the coverage metric: a character that is neither token
        # text nor whitespace nor a comment has to be reported, not absorbed.
        body = "aa bb"
        toks, _d, _t = T.tokenize_page(1, body)
        lost = [t for t in toks if t.text != "bb"]   # a token the stream drops
        labels = cl.char_map(body, lost, T.comment_spans(body))
        self.assertEqual(labels.count("gap_other"), 2)
        self.assertEqual(labels.count("token_word"), 2)
        self.assertEqual(len(labels), len(body))
        self.assertEqual(cl.coverage(len(body), 2), 1.0 - 2.0 / len(body))

    def test_unclassified_punctuation_is_a_token_not_residue(self):
        # the tokenizer has a class for everything it emits, so a stray glyph
        # is counted (`punct_other`) rather than lost
        counts = buckets_of("text\u00a7more")
        self.assertEqual(counts.get("gap_other", 0), 0)
        self.assertEqual(counts.get("token_punct_other", 0), 1)

    def test_coverage_is_one_when_nothing_is_unexplained(self):
        self.assertEqual(cl.coverage(480962, 0), 1.0)
        self.assertEqual(cl.coverage(100, 4), 0.96)
        self.assertEqual(cl.coverage(0, 0), 0.0)

    def test_char_map_agrees_with_the_page_body_length(self):
        body = "I am disillusioned with the volume inoculation::\n\n<<adAM>>"
        labels = cl.char_map(body, *T.tokenize_page(7, body)[:1],
                             T.comment_spans(body))
        self.assertEqual(len(labels), len(body))
        self.assertNotIn("gap_other", labels)


# ---------------------------------------------------------------------------
class TestOperatorPartition(unittest.TestCase):
    """L3: every operator token carries exactly one #17 role."""

    def test_every_op_token_gets_exactly_one_role(self):
        body = ("gene=TV/spasm::soul/gram//boy-roid ++ more=stuff\n"
                "<<block>> <> <a> | /")
        toks, _d, _t = T.tokenize_page(3, body)
        roles = [cl.operator_role(toks, i) for i, t in enumerate(toks)
                 if t.kind == "OP"]
        self.assertTrue(roles)
        for role in roles:
            self.assertIn(role, [cl.ROLE_SEGMENT, cl.ROLE_SEAM]
                          + list(cl.ROLE_UNBOUND))
        self.assertEqual(len(roles), len(set(range(len(roles)))))  # one each

    def test_seam_and_segment_roles_are_measured(self):
        toks, _d, _t = T.tokenize_page(3, "gene=TV::soul/gram//x=y")
        roles = [cl.operator_role(toks, i) for i, t in enumerate(toks)
                 if t.kind == "OP"]
        # `=` in gene=TV, `/` in soul/gram (glued to the right word), `=` in x=y
        self.assertEqual(roles.count(cl.ROLE_SEAM), 3)
        self.assertEqual(roles.count(cl.ROLE_SEGMENT), 2)   # `::` and `//`

    def test_unbound_operators_are_named_not_repaired(self):
        toks, _d, _t = T.tokenize_page(3, "word = \n\n word / end")
        roles = [cl.operator_role(toks, i) for i, t in enumerate(toks)
                 if t.kind == "OP"]
        self.assertIn("seam_orphan_side", roles)     # `=` over a blank line
        self.assertIn("slash_free", roles)           # unglued `/`


# ---------------------------------------------------------------------------
class TestUnitLedger(unittest.TestCase):
    """L4: the unit layer closes against #19's screen and ledger."""

    UNITS = [{"uid": "1", "page": "4", "start": "0", "unit": "gene=TV"},
             {"uid": "2", "page": "9", "start": "0", "unit": "th"},
             {"uid": "3", "page": "11", "start": "0", "unit": "li"},
             {"uid": "4", "page": "4", "start": "1", "unit": "a"},
             {"uid": "5", "page": "4", "start": "2", "unit": "a"}]

    def test_occurrences_close_against_the_screen(self):
        screening = [screening_row("gene=TV", "compound"),
                    screening_row("a", "english")]
        markup_uids = {2, 3}
        cats, residue, _detail, escape = cl.unit_ledger(
            self.UNITS, screening, markup_uids)
        self.assertEqual(cats["compound"], 1)
        self.assertEqual(cats["english"], 2)
        self.assertEqual(residue["markup_ledger_occurrence"], 2)
        self.assertEqual(residue["unscreened_surface"], 0)
        self.assertEqual(sum(cats.values()) + sum(residue.values()),
                         len(self.UNITS))

    def test_uid_level_screening_beats_a_surface_join(self):
        # `a` occurs twice as the article and once inside </a>: only the uid
        # join removes the markup one, so a surface join over-counts `a`.
        screening = [screening_row("a", "english")]
        markup_uids = {4}
        cats, residue, _d, _e = cl.unit_ledger(self.UNITS, screening,
                                              markup_uids)
        self.assertEqual(cats["english"], 1)
        self.assertEqual(residue["markup_ledger_occurrence"], 1)

    def test_an_unscreened_surface_is_counted_as_residue(self):
        screening = [screening_row("gene=TV", "compound")]
        _cats, residue, detail, _e = cl.unit_ledger(
            self.UNITS, screening, markup_uids=set())
        self.assertEqual(residue["unscreened_surface"], 4)
        self.assertEqual(len(detail["unscreened_surface"]), 4)
        self.assertEqual(cl.coverage(len(self.UNITS), 4),
                         1.0 - 4.0 / len(self.UNITS))

    def test_markup_only_and_mixed_types_are_separated(self):
        screening = [screening_row("gene=TV", "compound"),
                    screening_row("a", "english")]
        markup_uids = {2, 3, 4}
        markup_only, mixed = cl.type_ledger(self.UNITS, screening, markup_uids)
        self.assertEqual([r["unit"] for r in markup_only], ["li", "th"])
        self.assertEqual([r["unit"] for r in mixed], ["a"])
        self.assertEqual(mixed[0]["markup_n"], 1)

    def test_type_level_and_full_screen_arithmetic_are_checked(self):
        units = self.UNITS + [{"uid": "6", "page": "5", "start": "0",
                               "unit": "gene=TV"}]
        screening = [screening_row("gene=TV", "compound"),
                    screening_row("a", "english")]
        markup_only, _mixed = cl.type_ledger(units, screening, {2, 3})
        types = cl.check_types(units, screening, markup_only, tag_run_total=1,
                              expected_types=4, expected_screened=2,
                              expected_full_screen=3, expected_markup=2,
                              against_artifact=False)
        self.assertEqual(types["markup_only_types"], 2)
        self.assertEqual(types["screened_types"], 2)
        self.assertEqual(types["full_screen_occurrences"], 3)

    def test_a_type_level_that_does_not_close_fails(self):
        screening = [screening_row("gene=TV", "compound")]
        with self.assertRaises(cl.LedgerError):
            cl.check_types(self.UNITS, screening, [], tag_run_total=1,
                           expected_types=4, expected_screened=2,
                           expected_full_screen=3, expected_markup=2,
                           against_artifact=False)

    def test_check_closed_fails_on_planted_residue(self):
        chars = {"token_word": 10, "gap_whitespace": 2, "gap_other": 1}
        tokens = {"token_WORD": 1}
        ops = {cl.ROLE_SEGMENT: 1}
        with self.assertRaises(cl.LedgerError):
            cl.check_closed(chars, tokens, ops, {}, {"unscreened_surface": 1},
                            [])

    def test_check_closed_fails_on_a_char_total_that_does_not_add_up(self):
        chars = {"token_word": 10}
        with self.assertRaises(cl.LedgerError):
            cl.check_closed(chars, {"token_WORD": 1}, {cl.ROLE_SEGMENT: 1},
                            {"english": 1}, {"markup_ledger_occurrence": 1},
                            [])


class TestConsumers(unittest.TestCase):
    """Every downstream layer's kept/dropped split must close."""

    def test_split_arithmetic_is_checked(self):
        rows = cl.consumer_ledger()
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(int(row["kept"]) + int(row["dropped"]),
                             row["level_total"], row["issue"])
            self.assertEqual(row["closes"], "y")

    def test_a_drifted_split_fails_the_build(self):
        original = cl.CONSUMERS
        drifted = list(original) + [("#99 test", "unit occurrence",
                                    "analysis/tokenizer/summary.json",
                                    ("units", "total"), 5,
                                    "planted drift for the test")]
        try:
            cl.CONSUMERS = drifted
            with self.assertRaises(cl.LedgerError):
                cl.consumer_ledger()
        finally:
            cl.CONSUMERS = original

    def test_a_missing_key_fails_loudly(self):
        original = cl.CONSUMERS
        broken = list(original) + [("#99 test", "unit occurrence",
                                    "analysis/tokenizer/summary.json",
                                    ("units", "not_a_key"), 0, "planted")]
        try:
            cl.CONSUMERS = broken
            with self.assertRaises(cl.LedgerError):
                cl.consumer_ledger()
        finally:
            cl.CONSUMERS = original


class TestWriters(unittest.TestCase):
    def test_cell_escapes_tabs_and_newlines(self):
        self.assertEqual(cl._cell("a\tb"), "a\\tb")
        self.assertEqual(cl._cell("a\nb"), "a\\nb")
        self.assertEqual(cl._cell("a\\b"), "a\\\\b")
        self.assertEqual(cl._cell(None), "")

    def test_an_escaped_field_cannot_break_the_row_structure(self):
        # read_rows is a plain line splitter (like every reader in this
        # workspace), so the escaping is what keeps a tab or a newline inside
        # a field from splitting the row in two.
        with tempfile.TemporaryDirectory() as tmp:
            cl.OUT = Path(tmp)
            cl.write_tsv("t.tsv", ["a", "b"],
                         [{"a": "x\ty", "b": 1}, {"a": "z\nw", "b": 2}])
            text = (Path(tmp) / "t.tsv").read_text()
            rows = cl.read_rows(Path(tmp) / "t.tsv")
        self.assertEqual(text.count("\n"), 3)          # header + 2 rows
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["a"], "x\\ty")        # stored escaped
        self.assertEqual(rows[1]["b"], "2")


# ---------------------------------------------------------------------------
# corpus tests: the frozen workspace artifacts
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "frozen analysis artifacts not present")
class TestCorpus(unittest.TestCase):
    """Every number the ledger publishes, against the issue that owns it."""

    @classmethod
    def setUpClass(cls):
        cls.bodies = cl.page_bodies()
        cls.book = cl.check_stream(cls.bodies)
        cls.chars, _ex = cl.char_ledger(cls.book, cls.bodies)
        cls.tokens = cl.token_ledger(cls.book)
        cls.ops, cls.ops_by_op = cl.operator_ledger(cls.book)
        cls.units = cl.read_rows(cl.UNITS)
        cls.screening = cl.read_rows(cl.SCREENING)
        cls.markup_rows = cl.read_rows(cl.MARKUP)
        cls.markup_uids = {int(r["uid"]) for r in cls.markup_rows}
        cls.cats, cls.residue, _d, cls.escape = cl.unit_ledger(
            cls.units, cls.screening, cls.markup_uids)
        cls.markup_only, cls.mixed = cl.type_ledger(cls.units, cls.screening,
                                                    cls.markup_uids)
        cls.furniture, cls.book_text, cls.tag_pages = cl.tag_run_detail()

    def test_stream_rebuilds_and_agrees_with_the_code(self):
        # check_stream raises on any disagreement; reaching here is the test
        self.assertEqual(len(self.book), cl.EXPECT_PAGES)

    def test_char_level_closes_with_no_unexplained_residue(self):
        self.assertEqual(sum(self.chars.values()), cl.EXPECT_BODY_CHARS)
        self.assertEqual(self.chars.get("gap_other", 0), 0)
        self.assertEqual(self.chars["layout_comment"],
                         cl.EXPECT_SCAFFOLD_CHARS)

    def test_token_partition_reproduces_14_and_17(self):
        self.assertEqual(self.tokens["token_WORD"], cl.EXPECT_WORDS)
        self.assertEqual(self.tokens["token_OP"], cl.EXPECT_OPS)
        self.assertEqual(self.tokens["token_PUNCT"], cl.EXPECT_PUNCT)
        self.assertEqual(sum(self.tokens.values()), cl.EXPECT_TOKENS)

    def test_operator_partition_reproduces_17s_own_summary(self):
        summary = cl.read_json(ROOT / "analysis" / "tokenizer" / "summary.json")
        self.assertEqual(dict(self.ops), summary["op_roles"])
        self.assertEqual(self.ops[cl.ROLE_SEAM], cl.EXPECT_SEAMS)
        self.assertEqual(self.ops[cl.ROLE_SEGMENT], cl.EXPECT_SEGMENTS)
        self.assertEqual(sum(self.ops.get(r, 0) for r in cl.ROLE_UNBOUND),
                         cl.EXPECT_UNBOUND)
        self.assertEqual(sum(self.ops.values()), cl.EXPECT_OPS)

    def test_unit_occurrences_close_against_19(self):
        self.assertEqual(len(self.units), cl.EXPECT_UNITS)
        self.assertEqual(sum(self.cats.values()), cl.EXPECT_SCREENED)
        self.assertEqual(self.residue["markup_ledger_occurrence"],
                         cl.EXPECT_MARKUP)
        self.assertEqual(self.residue["unscreened_surface"], 0)
        self.assertEqual(len(self.markup_rows), cl.EXPECT_MARKUP)

    def test_type_level_closes_against_19(self):
        self.assertEqual(len(self.markup_only), cl.EXPECT_MARKUP_TYPES)
        self.assertEqual(sum(r["n"] for r in self.markup_only), 145)
        self.assertEqual([r["unit"] for r in self.mixed],
                         ["a", "body"])

    def test_the_escape_mismatch_is_exactly_two_occurrences(self):
        self.assertEqual(len(self.escape), 2)
        self.assertEqual({int(r["page"]) for r in self.escape}, {102, 209})

    def test_tag_run_removals_decompose_into_241(self):
        self.assertEqual(sum(self.furniture.values()) + len(self.book_text),
                         cl.EXPECT_TAG_RUN)
        self.assertEqual(sum(self.furniture.values()), 228)
        self.assertEqual(len(self.book_text), cl.EXPECT_BOOK_TEXT_IN_SPANS)
        self.assertEqual({r["page"] for r in self.book_text}, {158, 219})
        self.assertIn("konganic-eye", {r["unit"] for r in self.book_text})
        self.assertIn("retro-ADAM", {r["unit"] for r in self.book_text})

    def test_published_type_inventory_matches_the_unit_stream(self):
        published = {r["unit_type"] for r in cl.read_rows(cl.UNIT_TYPES)}
        surfaces = {u["unit"] for u in self.units}
        self.assertEqual(published, surfaces)
        self.assertEqual(len(surfaces), cl.EXPECT_UNIT_TYPES)

    def test_entropy_char_total_reconciles(self):
        total, markers, spans = cl.entropy_char_total(self.bodies, self.chars)
        self.assertEqual(total, cl.EXPECT_ENTROPY_CHARS)
        self.assertEqual(markers, cl.EXPECT_HEADER_MARKER_CHARS)
        self.assertEqual(spans, cl.EXPECT_SCAFFOLD_SPANS)

    def test_scaffold_spans_split_between_tails_and_gaps(self):
        scaffold = cl.scaffold_spans(self.book, self.bodies)
        self.assertEqual(len(scaffold), cl.EXPECT_SCAFFOLD_SPANS)
        self.assertEqual({s["page"] for s in scaffold}, {1, 224, 225})
        self.assertEqual(sum(1 for s in scaffold if s["in_tail"]), 6)


@unittest.skipUnless(HAVE_ARTIFACTS, "frozen analysis artifacts not present")
class TestBuild(unittest.TestCase):
    """The artifact set, end to end, and deterministically."""

    def test_build_writes_every_artifact_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = cl.build(Path(tmp) / "a", quiet=True)
            cl.build(Path(tmp) / "b", quiet=True)
            out = Path(tmp) / "a"
            for name in ("coverage_ledger_report.md", "coverage.tsv",
                         "residue_ledger.tsv", "per_page.tsv",
                         "consumer_reconciliation.tsv", "reconciliation.tsv",
                         "diagnostics.tsv", "summary.json"):
                self.assertTrue((out / name).exists(), name)
        self.assertEqual(first["unexplained"],
                         {"characters": 0, "tokens": 0, "operator_tokens": 0,
                          "unit_occurrences": 0})
        self.assertEqual(first["residue"]["n_unexplained"], 0)
        self.assertEqual(first["levels"]["characters"], cl.EXPECT_BODY_CHARS)
        self.assertEqual(first["levels"]["unit_occurrences"], cl.EXPECT_UNITS)

    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            cl.build(Path(tmp) / "a", quiet=True)
            cl.build(Path(tmp) / "b", quiet=True)
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            for name in ("coverage.tsv", "residue_ledger.tsv", "per_page.tsv",
                         "consumer_reconciliation.tsv", "reconciliation.tsv",
                         "diagnostics.tsv", "summary.json",
                         "coverage_ledger_report.md"):
                self.assertEqual((a / name).read_bytes(),
                                 (b / name).read_bytes(), name)

    def test_residue_ledger_rows_are_auditable(self):
        summary = json.loads((ART / "summary.json").read_text()) \
            if (ART / "summary.json").exists() else None
        if summary is None:
            self.skipTest("analysis/coverage_ledger/ not built yet")
        for row in cl.read_rows(ART / "residue_ledger.tsv"):
            self.assertIn(row["verdict"], (cl.VERDICT_EXPLAINED,
                                           cl.VERDICT_DISCLOSED,
                                           cl.VERDICT_UNEXPLAINED))
            self.assertNotEqual(row["verdict"], cl.VERDICT_UNEXPLAINED)
            self.assertTrue(row["owner"], row["class"])
            self.assertTrue(row["example"], row["class"])

    def test_the_published_report_is_what_the_code_builds(self):
        if not (ART / "coverage_ledger_report.md").exists():
            self.skipTest("analysis/coverage_ledger/ not built yet")
        published = (ART / "coverage_ledger_report.md").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            cl.build(Path(tmp), quiet=True)
            rebuilt = (Path(tmp) / "coverage_ledger_report.md").read_text()
        self.assertEqual(published, rebuilt)

    def test_per_page_rows_sum_to_the_level_totals(self):
        rows = cl.read_rows(ART / "per_page.tsv") \
            if (ART / "per_page.tsv").exists() else []
        if not rows:
            self.skipTest("analysis/coverage_ledger/ not built yet")
        self.assertEqual(len(rows), cl.EXPECT_PAGES)
        self.assertEqual(sum(int(r["n_chars"]) for r in rows),
                         cl.EXPECT_BODY_CHARS)
        self.assertEqual(sum(int(r["n_tokens"]) for r in rows),
                         cl.EXPECT_TOKENS)
        self.assertEqual(sum(int(r["n_units"]) for r in rows),
                         cl.EXPECT_UNITS)
        self.assertEqual(sum(int(r["unexplained_chars"]) for r in rows), 0)
        self.assertEqual(sum(int(r["book_text_removed"]) for r in rows),
                         cl.EXPECT_BOOK_TEXT_IN_SPANS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
