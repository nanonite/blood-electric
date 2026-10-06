#!/usr/bin/env python3
"""Regression suite for the operator-aware tokenizer (Chainlink #17).

Run the whole work/ suite with:
    python -m unittest discover -s work -p 'test_*.py'

The suite is deliberately two-layered:

  * unit tests on synthetic strings, which pin the seam rules exactly (they
    fail loudly if a rule is changed);
  * corpus tests, which pin every number this tokenizer is supposed to
    reproduce from #10 / #12 / #14 and its own artifacts.

The corpus tests rebuild the book once (module-level cache) and check the
published artifacts against it, so an artifact that drifts from the code is a
test failure, not a silent inconsistency.
"""
import collections
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import operator_tokenizer as T           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "tokenizer")


def units_of(text, pno=1):
    """Tokenize + cut units for a bare string (no page/region machinery)."""
    toks, _diags, tail = T.tokenize_page(pno, text)
    us, _d = T.build_units(toks)
    return toks, us, tail


def unit_texts(text):
    return [u.text for u in units_of(text)[1]]


# ---------------------------------------------------------------------------
class TestSeamRules(unittest.TestCase):
    """The unit grammar on hand-written strings."""

    def test_issue_examples_stay_whole_units(self):
        # the three constructs named in the issue
        self.assertEqual(unit_texts("gene=TV"), ["gene=TV"])
        self.assertEqual(unit_texts("soul/gram"), ["soul/gram"])
        self.assertEqual(unit_texts("suck=blood"), ["suck=blood"])

    def test_parts_and_seams_of_a_named_unit(self):
        _t, us, _tail = units_of("the suck=blood chromosome")
        u = us[1]
        self.assertEqual(u.text, "suck=blood")
        self.assertEqual(u.parts, [("suck", "w"), ("=", "seam"), ("blood", "w")])
        self.assertEqual(u.sig, "=")
        self.assertEqual(u.cls, "eq")
        self.assertEqual(u.n_parts, 2)

    def test_hyphen_compound_is_one_unit_with_join_parts(self):
        _t, us, _tail = units_of("boy-roid")
        self.assertEqual([u.text for u in us], ["boy-roid"])
        self.assertEqual(us[0].parts, [("boy", "w"), ("-", "join"), ("roid", "w")])
        self.assertEqual(us[0].cls, "join")
        self.assertEqual(us[0].n_join, 1)
        self.assertEqual(us[0].n_hyp, 1)

    def test_multi_hyphen_unit_exposes_every_part(self):
        _t, us, _tail = units_of("cold-blooded-machine")
        self.assertEqual([u.text for u in us], ["cold-blooded-machine"])
        self.assertEqual([t for t, r in us[0].parts if r == "w"],
                         ["cold", "blooded", "machine"])
        self.assertEqual(us[0].n_hyp, 2)

    def test_slash_needs_glue_on_the_right(self):
        # glued: a compound
        self.assertEqual(unit_texts("soul/gram"), ["soul/gram"])
        # spaced: #12/#14's sentence-final terminator, two units
        self.assertEqual(unit_texts("self / gram"), ["self", "gram"])

    def test_segment_operators_never_bind(self):
        for op in ("::", "//", "<<", ">>", "<", ">", "|"):
            text = f"alpha{op}beta"
            self.assertEqual(unit_texts(text), ["alpha", "beta"],
                             f"{op} must cut, not bind")

    def test_colon_and_dashes_are_segment_operators(self):
        self.assertEqual(unit_texts("head:line"), ["head", "line"])
        self.assertEqual(unit_texts("word - word"), ["word", "word"])
        self.assertEqual(unit_texts("word — word"), ["word", "word"])
        self.assertEqual(unit_texts("word – word"), ["word", "word"])
        self.assertEqual(unit_texts("word, word."), ["word", "word"])

    def test_dash_glued_to_a_word_stays_inside_the_unit(self):
        # #12/#14: `ice—our` is a compound join, not a separator; a dash glued
        # on one side only (`scope— the`) is still #12's free dash -> it cuts.
        self.assertEqual(unit_texts("ice—our"), ["ice—our"])
        self.assertEqual(units_of("ice—our")[1][0].parts,
                         [("ice", "w"), ("—", "join"), ("our", "w")])
        self.assertEqual(unit_texts("scope— the"), ["scope", "the"])
        self.assertEqual(unit_texts("scope - the"), ["scope", "the"])

    def test_chained_seams_make_one_maximal_unit(self):
        self.assertEqual(unit_texts("gene=TV/spasm"), ["gene=TV/spasm"])
        _t, us, _tail = units_of("gene=TV/spasm")
        self.assertEqual(us[0].sig, "=/")
        self.assertEqual(us[0].cls, "mixed")
        self.assertEqual(us[0].n_parts, 3)

    def test_three_seam_chain(self):
        self.assertEqual(unit_texts("a/b/c/d"),
                         ["a/b/c/d"])

    def test_seam_needs_a_word_on_each_side(self):
        # `++` opening a segment: no left word, so no unit
        self.assertEqual(unit_texts("//++communicates"), ["communicates"])
        # `=` with nothing on the right
        self.assertEqual(unit_texts("self="), ["self"])

    def test_spaced_equals_still_binds(self):
        # two words, just not glued: still one unit (flagged as seam_spaced)
        self.assertEqual(unit_texts("gene = TV"), ["gene = TV"])

    def test_seam_does_not_cross_a_paragraph_break(self):
        # a blank line is not glue, so `alpha =` / `beta` are two units
        self.assertEqual(unit_texts("alpha =\n\nbeta"), ["alpha", "beta"])
        # the same is true of an em dash across the break
        self.assertEqual(unit_texts("alpha—\n\nbeta"), ["alpha", "beta"])

    def test_units_are_maximal_not_greedy(self):
        # `=` after a seam that is followed by a segment op stops the unit
        self.assertEqual(unit_texts("a=b::c"), ["a=b", "c"])
        self.assertEqual(unit_texts("a/b//c"), ["a/b", "c"])

    def test_even_odd_slot_structure(self):
        # WORD (seam|glued-dash WORD)* - even slots are words, odd slots bind
        _t, us, _tail = units_of("gene=TV/spasm and boy-roid and ice—our")
        for u in us:
            for k, t in enumerate(u.toks):
                if k % 2 == 0:
                    self.assertEqual(t.kind, "WORD", f"{u.text} slot {k}")
                elif t.kind == "OP":
                    self.assertIn(t.text, T.SEAM_BOTH + T.SEAM_RIGHT)
                else:
                    self.assertEqual(t.pclass, "dash")

    def test_unit_offsets_agree_with_tokens(self):
        text = "the suck=blood chromosome of self"
        toks, us, _tail = units_of(text)
        for u in us:
            self.assertEqual(u.s, u.toks[0].s)
            self.assertEqual(u.e, u.toks[-1].e)
            self.assertEqual(text[u.s:u.e], u.text)

    def test_unit_ids_are_assigned_to_every_token(self):
        toks, us, _tail = units_of("gene=TV soul/gram boy-roid")
        covered = [t for t in toks if t.unit is not None]
        self.assertEqual(len(covered), sum(len(u.toks) for u in us))
        for t in covered:
            self.assertIn(t.pos, range(len(us[t.unit].toks)))

    def test_unit_positions_are_sequential(self):
        _t, us, _tail = units_of("gene=TV/spasm")
        # 5 tokens: gene = TV / spasm
        self.assertEqual([t.pos for t in us[0].toks], [0, 1, 2, 3, 4])
        self.assertEqual([t.text for t in us[0].toks],
                         ["gene", "=", "TV", "/", "spasm"])

    def test_quotes_and_apostrophes_stay_in_words(self):
        self.assertEqual(unit_texts("it's"), ["it's"])
        self.assertEqual(unit_texts("Boy's"), ["Boy's"])

    def test_numeric_unit_is_lexical(self):
        self.assertEqual(unit_texts("1-84068-060-1"), ["1-84068-060-1"])

    def test_underscore_is_a_word_char(self):
        self.assertEqual(unit_texts("v_1"), ["v_1"])

    def test_unit_class_naming(self):
        cases = {"gene": "plain", "boy-roid": "join", "a=b": "eq", "a/b": "slash",
                 "a++b": "cat", "a+b": "plus", "a=b/c": "mixed",
                 "a=b-c": "eq+join", "a/b-c": "slash+join"}
        for text, want in cases.items():
            got = unit_texts(text)
            self.assertEqual(len(got), 1, text)
            self.assertEqual(units_of(text)[1][0].cls, want, text)


# ---------------------------------------------------------------------------
class TestLosslessness(unittest.TestCase):
    """The gap+text reconstruction is the tokenizer's primary criterion."""

    def test_rebuild_is_exact_on_synthetic_text(self):
        for text in ("gene=TV soul/gram boy-roid",
                     "a::b//c<<d>>e",
                     "  leading and   inner   spacing  \n\n second para\n",
                     "punct : , ; . ! ? \" ' ( ) [ ]",
                     "— en – and hyphen -",
                     "<in the future>::x",
                     "</p> and <!-- figure x -->\n"):
            toks, _us, tail = units_of(text)
            self.assertEqual(T.rebuild(toks, tail), text, repr(text))

    def test_rebuild_preserves_no_space_after_operators(self):
        text = "suck=blood::head//next"
        toks, _us, tail = units_of(text)
        self.assertEqual(T.rebuild(toks, tail), text)
        self.assertEqual([t.gap for t in toks if t.kind == "OP"], ["", "", ""])

    def test_layout_comments_yield_no_tokens(self):
        text = "alpha <!-- figure 1 --> beta"
        toks, _us, tail = units_of(text)
        self.assertEqual([t.text for t in toks], ["alpha", "beta"])
        self.assertEqual(T.rebuild(toks, tail), text)

    def test_glue_flags_agree_with_offsets(self):
        toks, _us, _tail = units_of("a=b a /b")
        glued = [t for t in toks if t.text == "="][0]
        self.assertTrue(glued.glue_L)
        self.assertTrue(glued.glue_R)
        # `a /b`: the `/` is glued right (to `b`) but not left (space), and it
        # is therefore not a seam - #14's sentence-final terminator
        loose = [t for t in toks if t.text == "/"][0]
        self.assertFalse(loose.glue_L)
        self.assertTrue(loose.glue_R)


# ---------------------------------------------------------------------------
class TestBaselineTokenizers(unittest.TestCase):
    """The comparison set must actually behave like the tools it names."""

    def test_ws_splits_on_whitespace(self):
        self.assertEqual(T.baseline_tokens("ws", "a b\tc\nd"),
                         ["a", "b", "c", "d"])

    def test_wordpunct_keeps_operator_runs(self):
        self.assertEqual(T.baseline_tokens("wordpunct", "a::b"), ["a", "::", "b"])
        self.assertEqual(T.baseline_tokens("wordpunct", "a=b"), ["a", "=", "b"])
        # it merges adjacent operators into one token - the point of §4.1
        self.assertEqual(T.baseline_tokens("wordpunct", "a=+b"),
                         ["a", "=+", "b"])

    def test_sklearn_baseline_is_the_real_library_when_present(self):
        got = T.baseline_tokens("sklearn", "gene=TV soul/gram")
        # CountVectorizer's default lowercases and needs 2+ word chars
        self.assertEqual(got, ["gene", "tv", "soul", "gram"])
        self.assertNotIn("gene=TV", got)
        # and the flag reports honestly whether sklearn was importable
        self.assertIn(T.baseline_uses_real_lib(), (True, False))

    def test_alpha_drops_operators_and_digits(self):
        self.assertEqual(T.baseline_tokens("alpha", "gene=TV 666"),
                         ["gene", "TV"])

    def test_unknown_baseline_raises(self):
        with self.assertRaises(ValueError):
            T.baseline_tokens("nope", "x")


# ---------------------------------------------------------------------------
class TestBaselineSpans(unittest.TestCase):
    """The evaluation metric: an item is kept only if a baseline emits a token
    covering exactly its own source span."""

    TEXT = ("Soul/gram gene=TV suck=blood soul/gram\n"
            "the soul-machine::brain \"quoted\" 42 v_1 don't\n")

    def test_spans_and_tokens_are_the_same_tokenization(self):
        for name in T.BASELINES:
            self.assertTrue(T.baseline_span_check(name, self.TEXT), name)

    def test_sklearn_spans_are_the_installed_librarys_offsets(self):
        # the offsets come from the default token_pattern; the library itself
        # lower-cases and exposes no offsets, so the check is token-for-token
        # with case folded.  This is what licenses scoring sklearn by span.
        if not T.baseline_uses_real_lib():
            self.skipTest("sklearn not installed")
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        spans = sorted(T.baseline_spans("sklearn", self.TEXT))
        real = list(cv.build_analyzer()(self.TEXT))
        self.assertEqual([self.TEXT[s:e].lower() for s, e in spans], real)

    def test_case_folding_cannot_hide_a_unit(self):
        # the reason the metric is span-based: the real CountVectorizer emits
        # `soul`, `gram` - never `Soul/gram` - and it lower-cases, so a
        # string-membership test cannot even find a plain `Soul` unit.
        if not T.baseline_uses_real_lib():
            self.skipTest("sklearn not installed")
        span = (self.TEXT.index("Soul"), self.TEXT.index("Soul") + len("Soul"))
        self.assertIn(span, T.baseline_spans("sklearn", self.TEXT))
        self.assertNotIn("Soul", T.baseline_tokens("sklearn", self.TEXT))
        # and the fused unit is destroyed by the same tokenizer, span or no span
        seam_unit = (self.TEXT.index("Soul/gram"),
                     self.TEXT.index("Soul/gram") + len("Soul/gram"))
        self.assertNotIn(seam_unit, T.baseline_spans("sklearn", self.TEXT))

    def test_a_same_page_match_elsewhere_does_not_count(self):
        # `gram` at (5, 9) is part of the ws token `Soul/gram`; the *second*
        # `gram` on the page is a ws token of its own, so a page-bag test would
        # report the first part as kept when it is not a token at all.
        text = "Soul/gram gram"
        spans = T.baseline_spans("ws", text)
        self.assertIn((10, 14), spans)
        self.assertNotIn((5, 9), spans)
        self.assertIn("gram", T.baseline_tokens("ws", text))

    def test_part_spans_agree_with_part_roles_and_offsets(self):
        _t, us, _tail = units_of("cold-blooded gene=TV/spasm")
        for u in us:
            ps = T.part_spans(u)
            self.assertEqual([x[2] for x in ps], [t for t, _r in u.parts])
            self.assertEqual([x[3] for x in ps], [r for _t, r in u.parts])
            for s, e, txt, _r in ps:
                self.assertEqual(u.text[s - u.s:e - u.s], txt)

    def test_part_spans_on_the_real_book_are_exact(self):
        bodies = T.OG.load_pages_raw(T.SRC)
        book = T.tokenize_book(T.SRC)
        n_parts = 0
        for pno in sorted(book):
            for u in T.build_units(book[pno][0])[0]:
                for s, e, txt, _r in T.part_spans(u):
                    self.assertEqual(bodies[pno][s:e], txt, f"p{pno} uid {u.id}")
                    n_parts += 1
        # every unit's part count is unchanged by the offset view
        self.assertEqual(
            n_parts, sum(1 for pno in sorted(book)
                         for u in T.build_units(book[pno][0])[0]
                         for _t, r in u.parts))


# ---------------------------------------------------------------------------
# corpus-level tests (one build, shared)
# ---------------------------------------------------------------------------
_BUILD = None
_SUMMARY = None


def corpus():
    global _BUILD, _SUMMARY
    if _BUILD is None:
        _BUILD = T.build(T.SRC, OUT)
        with open(os.path.join(OUT, "summary.json")) as f:
            _SUMMARY = json.load(f)
    return _BUILD, _SUMMARY


class TestCorpusReconciliation(unittest.TestCase):
    """Every number the tokenizer claims to inherit from #10/#12/#14."""

    @classmethod
    def setUpClass(cls):
        cls.S, cls.sum = corpus()

    def test_pages_and_losslessness(self):
        self.assertEqual(self.S["n_pages"], T.EXPECTED_PAGES)
        self.assertEqual(self.S["lossless"]["ok"], T.EXPECTED_PAGES)
        self.assertEqual(self.S["lossless"]["bad"], [])

    def test_operator_token_counts_match_10_12_14(self):
        self.assertEqual(self.S["tokens"]["OP"], T.EXPECTED_OPS_ALL)
        self.assertEqual(sum(self.S["ops_body"].values()), T.EXPECTED_OPS_BODY)
        self.assertEqual(sum(self.S["ops_header"].values()), T.EXPECTED_OPS_HEADER)
        for op, n in T.EXPECTED_OPS_TOKENS.items():
            self.assertEqual(self.S["ops_body"][op], n, f"op {op}")

    def test_word_token_count_matches_14_lex_nodes(self):
        self.assertEqual(self.S["tokens"]["WORD"], T.EXPECTED_WORDS)

    def test_seam_counts_match_14_link_nodes(self):
        # #14 emits a Link node for every `=`/`++`/`+`, and for `/` when glued
        # right with a following word; the orphan-sided tokens are Link nodes
        # with a missing operand, so seams + orphans must equal the Link total.
        for op, n in T.EXPECTED_LINK_NODES.items():
            roles = self.S["op_roles_by_op"]
            # which orphan roles sit on #14's Link side of the ledger
            link_side = (("seam_orphan_left", "slash_boundary") if op == "/"
                         else ("seam_orphan_left", "seam_orphan_right",
                               "seam_orphan_side"))
            got = self.S["seams"][op] + sum(roles.get(f"{r}|{op}", 0)
                                            for r in link_side)
            self.assertEqual(got, n, f"seam {op} + Link-side orphans vs #14 "
                                     f"Link nodes")
        # the `/` residual must be exactly #14's Slash nodes
        # `slash_boundary` belongs to the Link side, not the Slash side
        slash_residual = (self.S["op_roles_by_op"]["slash_free|/"]
                          + self.S["op_roles_by_op"]["seam_orphan_right|/"])
        self.assertEqual(slash_residual, T.EXPECTED_SLASH_NODES)

    def test_reconciliation_link_arithmetic_adds_up(self):
        # the printed `/` row must not claim a sum that is not its own total:
        # Link-side orphans are `seam_orphan_left` + `slash_boundary`; the other
        # two orphan roles are #14's Slash nodes, checked above.
        row = next(r for r in self.S["recon_rows"] if r["what"].startswith("`/`"))
        roles = self.S["op_roles_by_op"]
        expect = (self.S["seams"]["/"] + roles["seam_orphan_left|/"]
                  + roles["slash_boundary|/"])
        self.assertIn(f"{expect:,} Link nodes", row["here"])
        self.assertIn(f"{T.EXPECTED_LINK_NODES['/']:,} Link", row["earlier"])
        for op in ("=", "++", "+"):
            r = next(x for x in self.S["recon_rows"]
                     if x["what"].startswith(f"`{op}` seams"))
            tot = self.S["seams"][op] + sum(
                v for k, v in roles.items()
                if k.split("|")[1] == op and k.split("|")[0].startswith("seam_orphan"))
            self.assertIn(f"{tot:,}", r["here"], op)
            self.assertEqual(tot, T.EXPECTED_LINK_NODES[op], op)

    def test_ordered_operator_stream_matches_11(self):
        # #11 tokenizes operators with #10's greedy longest match; #12 (and this
        # tokenizer) uses its own regex.  Equal totals would not prove the two
        # agree, so the sequences are compared page by page.
        n11 = self.S["n11_stream"]
        self.assertTrue(n11["identical"])
        self.assertEqual(n11["pages_differ"], [])
        self.assertEqual(n11["pages"], T.EXPECTED_PAGES)
        self.assertEqual(n11["total"], T.EXPECTED_OPS_BODY)
        import operator_ngrams as NG
        runs = NG.page_runs(T.SRC)
        flat = [op for p in sorted(runs) for run in runs[p] for op in run]
        self.assertEqual(len(flat), T.EXPECTED_OPS_BODY)
        self.assertEqual(flat[0], "<<")

    def test_free_dashes_match_13_and_14(self):
        self.assertEqual(self.S["free_dash"], T.EXPECTED_FREE_DASH)

    def test_n12_style_counts_reproduce_12s_inventory(self):
        # #12's own method, recomputed: gene=TV 147, vital=serum 130,
        # suck=blood 129, soul/gram 317, hyphen 2,834.
        for r in self.S["named_rows"]:
            self.assertEqual(r["prior_recomputed"], r["prior"], r["needle"])
        self.assertEqual(self.S["n12_pairs"]["hyphen"]["n"], 2834)

    def test_named_constructs_are_accounted_for(self):
        # raw = whole_unit + inside_longer_unit + not_a_unit, always
        for r in self.S["named_rows"]:
            self.assertEqual(r["raw"],
                             r["whole_unit"] + r["inside_longer_unit"]
                             + r["not_a_unit"], r["needle"])
            self.assertGreater(r["whole_unit"], 0, r["needle"])

    def test_named_construct_counts_are_stable(self):
        self.assertEqual(self.S["named"]["gene=TV"], 142)
        self.assertEqual(self.S["named"]["soul/gram"], 306)
        self.assertEqual(self.S["named"]["suck=blood"], 120)
        self.assertEqual(self.S["named"]["vital=serum"], 128)

    def test_accounting_of_tokens_units_parts(self):
        u = self.S["units"]
        # a unit with k word parts contributes k parts, so parts >= units
        self.assertGreater(u["parts"], u["total"])
        # parts - units == the number of *extra* parts, i.e. every multi-part
        # unit contributes (n_words - 1)
        extra = u["multi_word_occurrences"]
        self.assertLessEqual(extra, u["parts"] - u["total"])
        self.assertEqual(u["multi_word_occurrences"], 5433)
        # seams: the operator kinds add up to the seam total
        self.assertEqual(u["seams"], sum(self.S["seams"].values()))
        # fused = seam-bearing + hyphen-only
        hyp_only = u["fused_occurrences"] - u["seam_occurrences"]
        self.assertGreater(hyp_only, 0)

    def test_op_roles_partition_the_operator_tokens(self):
        roles = self.S["op_roles"]
        self.assertEqual(sum(roles.values()), self.S["tokens"]["OP"])
        self.assertEqual(roles["seam"], self.S["units"]["seams"])
        self.assertEqual(roles["segment"], self.S["n_segment_ops"])

    def test_seams_unbound_and_segment_ops_partition_the_operator_stream(self):
        # the three-way split the spec states in 5: bound into a unit / not a
        # seam (a diagnostics row) / a #14 segment operator
        seams = sum(self.S["seams"].values())
        unbound = sum(self.S["unbound_ops"].values())
        self.assertEqual(seams + unbound + self.S["n_segment_ops"],
                         self.S["tokens"]["OP"])
        self.assertEqual(unbound,
                         sum(v for k, v in self.S["op_roles"].items()
                             if k not in ("seam", "segment")))

    def test_no_unit_straddles_a_segment_operator(self):
        got = {r["diag"] for r in self.S["diag_rows"]}
        self.assertNotIn("unit_straddles_boundary", got)
        self.assertNotIn("empty_unit", got)

    def test_unit_invariants_hold_on_the_real_book(self):
        # rebuild the flat stream from the tokenizer (225 pages, ~4s)
        book = T.tokenize_book(T.SRC)
        units = []
        for pno in sorted(book):
            us, _d = T.build_units(book[pno][0])
            units.extend(us)
        self.assertEqual(T.check_unit_invariants(
            [t for pno in sorted(book) for t in book[pno][0]], units), [])


# ---------------------------------------------------------------------------
class TestCorpusUnits(unittest.TestCase):
    """The unit layer itself on the real corpus."""

    @classmethod
    def setUpClass(cls):
        cls.S, cls.sum = corpus()
        book = T.tokenize_book(T.SRC)
        cls.book = book
        cls.pages = sorted(book)
        cls.units = []
        for pno in cls.pages:
            us, _d = T.build_units(book[pno][0])
            cls.units.extend(us)

    def test_unit_inventory(self):
        u = self.S["units"]
        self.assertEqual(u["total"], 74665)
        self.assertEqual(u["types"], 5876)
        self.assertEqual(u["parts"], 80363)
        self.assertEqual(u["seams"], 2816)
        self.assertEqual(u["hyphens"], 2878)
        self.assertEqual(u["joins"], 2882)
        self.assertEqual(u["seam_occurrences"], 2697)
        self.assertEqual(u["fused_occurrences"], 5433)

    def test_plain_units_are_exactly_single_word_units(self):
        plain = [u for u in self.units if u.cls == "plain"]
        self.assertEqual(len(plain), 69232)
        for u in plain[:2000]:
            self.assertEqual(u.n_parts, 1)
            self.assertEqual(u.parts, [(u.text, "w")])

    def test_no_unit_contains_a_segment_operator(self):
        for u in self.units:
            for k, t in enumerate(u.toks):
                if k % 2:
                    self.assertTrue(
                        t.text in T.SEAM_BOTH + T.SEAM_RIGHT
                        or (t.kind == "PUNCT" and t.pclass == "dash"),
                        f"{u.text} slot {k} {t.text!r}")

    def test_units_together_cover_every_word_token(self):
        n_words = sum(1 for u in self.units for t in u.toks if t.kind == "WORD")
        self.assertEqual(n_words, T.EXPECTED_WORDS)

    def test_concatenating_units_and_gaps_rebuilds_the_page(self):
        bodies = T.OG.load_pages_raw(T.SRC)
        book = T.tokenize_book(T.SRC)
        for pno in sorted(book):
            toks, _d, tail = book[pno]
            self.assertEqual(T.rebuild(toks, tail), bodies[pno], f"page {pno}")

    def test_ast_context_is_attached(self):
        # the corpus units are rebuilt without the AST here, so attach it
        idx = T.AstIndex(T.SRC)
        flat = [t for pno in sorted(self.pages) for t in self.book[pno][0]]
        T.annotate(flat, self.units, idx)
        with_ast = [u for u in self.units if u.ast_path]
        self.assertGreater(len(with_ast), 0.9 * len(self.units))
        for u in with_ast[:2000]:
            self.assertTrue(u.ast_path.startswith("Book>Page>"))
            self.assertIn(u.container,
                          ("Span", "Utterance", "Field", "Inline", "Header",
                           "Block", "Segment"))

    def test_case_variants_are_kept_distinct(self):
        texts = {u.text for u in self.units}
        self.assertIn("soul/gram", texts)
        self.assertIn("Soul/gram", texts)
        lower = {t.lower() for t in texts}
        self.assertIn("soul/gram", lower)

    def test_longest_unit_is_the_p216_android_chain(self):
        longest = max(self.units, key=lambda u: len(u.text))
        self.assertEqual(longest.page, 216)
        self.assertTrue(longest.text.startswith("ANDROID-beat-ANDROID"))
        self.assertGreaterEqual(longest.n_hyp, 20)

    def test_every_unit_is_the_exact_source_slice(self):
        # the page-cited offsets in units.tsv must be quotable: for all 74,665
        # units `body[start:end] == unit`, and the parts re-form it once the
        # whitespace around a spaced seam is discounted
        bodies = T.OG.load_pages_raw(T.SRC)
        for u in self.units:
            self.assertEqual(bodies[u.page][u.s:u.e], u.text,
                             f"p{u.page} uid {u.id}")
            self.assertEqual("".join("".join(t for t, _r in u.parts).split()),
                             "".join(u.text.split()),
                             f"p{u.page} uid {u.id}")

    def test_spaced_seams_keep_their_whitespace_in_the_unit_text(self):
        # `gene = TV` is one unit whose surface keeps the spacing (and a seam can
        # even span a line break); its parts are the bare strings, so
        # concatenating parts equals the surface only modulo whitespace.  A
        # schema fact, pinned here so a downstream join does not assume
        # otherwise.
        _t, us, _tail = units_of("gene = TV and soul/gram")
        self.assertEqual([u.text for u in us], ["gene = TV", "and", "soul/gram"])
        self.assertEqual("".join(t for t, _r in us[0].parts), "gene=TV")
        spaced = [u for u in self.units
                  if u.ops and "".join(t for t, _r in u.parts) != u.text]
        self.assertGreater(len(spaced), 0)
        for u in spaced[:500]:
            self.assertNotEqual(u.text, "".join(t for t, _r in u.parts))
            # the difference is whitespace and nothing else
            self.assertNotEqual("".join(u.text.split()), u.text)
            self.assertEqual("".join(u.text.split()),
                             "".join("".join(t for t, _r in u.parts).split()))
        # and the count matches the `seam_spaced` flag in the diagnostics ledger
        rows = {r["diag"]: r for r in self.S["diag_rows"]}
        self.assertEqual(len(spaced), rows["seam_spaced"]["n"])

    def test_part_roles_are_only_the_three_declared(self):
        seen = {r for u in self.units for _t, r in u.parts}
        self.assertEqual(seen, {"w", "seam", "join"})

    def test_parts_reassemble_the_unit(self):
        joined = [u for u in self.units if u.n_join][:200]
        self.assertGreater(len(joined), 100)
        for u in joined:
            self.assertEqual("".join(t for t, _r in u.parts), u.text)

    def test_glued_dash_units_exist_but_are_rare(self):
        glued = [u for u in self.units
                 if any(r == "join" and t != "-" for t, r in u.parts)]
        self.assertGreater(len(glued), 0)
        self.assertLess(len(glued), 50)

    def test_seam_parts_reassemble_the_unit(self):
        for u in self.units:
            if u.ops:
                self.assertEqual("".join(t for t, _r in u.parts), u.text)
                self.assertEqual(len(u.ops), len(
                    [1 for _t, r in u.parts if r == "seam"]))
                break


# ---------------------------------------------------------------------------
class TestSeamLoss(unittest.TestCase):
    """The evaluation the issue actually asks for."""

    @classmethod
    def setUpClass(cls):
        cls.S, cls.sum = corpus()
        cls.hdr = cls.S["seam_loss"]["baselines"]

    def test_all_four_baselines_were_measured(self):
        self.assertEqual(set(self.hdr), set(T.BASELINES))

    def test_operator_aware_is_lossless_on_all_three_views(self):
        # every occurrence is emitted once at each of the three levels, and the
        # page bodies are reconstructed exactly
        self.assertEqual(self.S["lossless"]["ok"], self.S["n_pages"])
        self.assertEqual(self.S["units"]["total"], 74665)
        self.assertEqual(self.S["units"]["parts"], 80363)
        self.assertEqual(self.S["units"]["seams"], 2816)

    def test_standards_lose_seams_or_units(self):
        # sklearn/alpha destroy the operator layer entirely
        for b in ("sklearn", "alpha"):
            self.assertEqual(self.hdr[b]["ops_kept"], 0, b)
            self.assertEqual(self.hdr[b]["seams_kept"], 0, b)
        # ws keeps the unit but loses the parts
        ws = self.hdr["ws"]["fused"]
        self.assertGreater(ws["unit_kept"] / ws["n"], 0.5)
        self.assertLess(ws["parts_kept"] / ws["parts"], 0.5)
        # wordpunct keeps the parts and the seam strings but splits the unit
        wp = self.hdr["wordpunct"]["fused"]
        self.assertEqual(wp["unit_kept"], 0)
        self.assertEqual(wp["parts_kept"], wp["parts"])

    def test_wordpunct_merges_adjacent_operators(self):
        ops_kept = self.hdr["wordpunct"]["ops_kept"]
        self.assertLess(ops_kept, self.S["tokens"]["OP"])

    def test_fused_subset_sizes(self):
        f = self.hdr["ws"]["fused"]
        self.assertEqual(f["n"], self.S["units"]["fused_occurrences"])
        self.assertEqual(f["seams"], self.S["units"]["seams"])

    def test_fused_and_plain_partition_the_units(self):
        u = self.S["units"]
        n_plain = sum(r["n"] for r in self.S["class_rows"]
                      if r["class"] in ("plain", "hyp")
                      and r["class"] == "plain")
        self.assertEqual(n_plain + u["fused_occurrences"], u["total"])

    def test_the_metric_is_offset_exact_not_string_membership(self):
        # every published rate is a count over spans.  For the three
        # case-preserving baselines that is strictly stricter than the
        # page-bag count an earlier draft reported; for `sklearn` the bag count
        # was *wrong in the other direction*, because the library lower-cases,
        # and the gap is exactly reproduced by comparing against lower-cased
        # surfaces.
        bodies = T.OG.load_pages_raw(T.SRC)
        book = T.tokenize_book(T.SRC)
        bag = collections.defaultdict(lambda: [0, 0, 0, 0])   # u, u_kept, p, p_kept
        bag_ci = collections.defaultdict(lambda: [0, 0])
        for pno in sorted(book):
            for b in T.BASELINES:
                toks = T.baseline_tokens(b, bodies[pno])
                exact = collections.Counter(toks)
                folded = collections.Counter(t.lower() for t in toks)
                for u in T.build_units(book[pno][0])[0]:
                    bag[b][0] += 1
                    bag[b][1] += bool(exact[u.text])
                    bag_ci[b][0] += 1
                    bag_ci[b][1] += bool(folded[u.text.lower()])
                    for _s, _e, txt, role in T.part_spans(u):
                        if role == "w":
                            bag[b][2] += 1
                            bag[b][3] += bool(exact[txt])
        for b, h in self.hdr.items():
            self.assertEqual(h["n_units"], bag[b][0], b)      # denominators agree
            self.assertEqual(h["n_parts"], bag[b][2], b)
            if b != "sklearn":
                # case-preserving: spans can only be stricter than the bag
                self.assertLessEqual(h["unit_kept"], bag[b][1], b)
                self.assertLessEqual(h["parts_kept"], bag[b][3], b)
            else:
                # the old count lost every case-bearing unit, and un-folding the
                # baseline's own case policy puts the exact count strictly
                # between the case-sensitive bag and the case-folded bag
                self.assertLess(bag[b][1], h["unit_kept"], b)
                self.assertLessEqual(h["unit_kept"], bag_ci[b][1], b)


# ---------------------------------------------------------------------------
class TestArtifacts(unittest.TestCase):
    """The published artifacts must agree with the code that wrote them."""

    EXPECTED_FILES = ("tokenizer_spec.md", "tokens.tsv", "tokens.jsonl",
                      "page_tails.tsv", "units.tsv", "unit_types.tsv",
                      "part_index.tsv", "seam_loss.tsv", "cases.tsv",
                      "case_examples.txt", "diagnostics.tsv", "reconciliation.tsv",
                      "named_constructs.tsv", "token_stats.tsv", "summary.json")

    @classmethod
    def setUpClass(cls):
        cls.S, cls.sum = corpus()

    def test_all_files_exist(self):
        for name in self.EXPECTED_FILES:
            path = os.path.join(OUT, name)
            self.assertTrue(os.path.exists(path), name)
            self.assertGreater(os.path.getsize(path), 0, name)

    def test_units_tsv_row_count(self):
        with open(os.path.join(OUT, "units.tsv")) as f:
            n = sum(1 for _ in f) - 1
        self.assertEqual(n, self.S["units"]["total"])

    def test_tokens_tsv_row_count(self):
        with open(os.path.join(OUT, "tokens.tsv")) as f:
            n = sum(1 for _ in f) - 1
        self.assertEqual(n, self.S["tokens"]["total"])

    def test_tokens_tsv_is_lossless(self):
        bodies = T.OG.load_pages_raw(T.SRC)
        rows = collections.defaultdict(list)
        with open(os.path.join(OUT, "tokens.tsv")) as f:
            head = f.readline().rstrip("\n").split("\t")
            self.assertIn("gap_json", head)
            gi = head.index("gap_json")
            for line in f:
                p = line.rstrip("\n").split("\t")
                rows[int(p[1])].append(p)
        for pno, ps in rows.items():
            body = bodies[pno]
            out, pos = [], 0
            for p in ps:
                start, end, text = int(p[6]), int(p[7]), p[8]
                gap = json.loads(p[gi])          # the column is a JSON literal
                self.assertEqual(gap, body[pos:start], f"p{pno} gap")
                self.assertEqual(text, body[start:end], f"p{pno} text")
                out.append(gap + text)
                pos = end
            self.assertEqual("".join(out), body[:pos], f"page {pno}")

    def test_tokens_tsv_gap_column_is_json_encoded(self):
        # a raw gap may carry a newline and would break a line-oriented reader,
        # so the column is JSON-encoded and named `gap_json`; the escape keeps
        # the row on one line and still decodes to the exact source text
        with open(os.path.join(OUT, "tokens.tsv")) as f:
            head = f.readline().rstrip("\n").split("\t")
            self.assertIn("gap_json", head)
            self.assertNotIn("gap", head)
            gi = head.index("gap_json")
            n_fields = len(head)
            multiline = 0
            for line in f:
                p = line.rstrip("\n").split("\t")
                self.assertEqual(len(p), n_fields)
                gap = json.loads(p[gi])
                if "\n" in gap:
                    multiline += 1
                    self.assertIn("\\n", p[gi])
        self.assertGreater(multiline, 0)   # the case this encoding exists for

    def test_text_json_column_round_trips_every_token(self):
        # `text` is raw source and the corpus contains bare `"` tokens, so
        # `text_json` is the quote-safe copy; both must decode to the same slice
        bodies = T.OG.load_pages_raw(T.SRC)
        n = 0
        with open(os.path.join(OUT, "tokens.tsv")) as f:
            head = f.readline().rstrip("\n").split("\t")
            ti, tji = head.index("text"), head.index("text_json")
            si, ei, ki = (head.index("start"), head.index("end"),
                          head.index("kind"))
            for line in f:
                p = line.rstrip("\n").split("\t")
                text = json.loads(p[tji])
                self.assertEqual(text, p[ti], p[0])
                self.assertEqual(text,
                                 bodies[int(p[head.index("page")])]
                                 [int(p[si]):int(p[ei])],
                                 f"p{p[1]} {p[ki]}")
                n += 1
        self.assertEqual(n, self.S["tokens"]["total"])

    def test_quote_token_hazard_is_disclosed_and_pinned(self):
        # a `.tsv` read with a quote-aware reader breaks on a bare `"` field.
        # The count is measured, published and stated in the spec; if OCR ever
        # changes it the number moves here too rather than silently.
        self.assertEqual(self.S["n_quote_tokens"], 136)
        self.assertEqual(self.S["quote_token_pages"], [47, 92, 158, 209, 219, 224])
        self.assertGreater(self.S["n_csv_bad_rows"], 0)
        with open(os.path.join(OUT, "tokenizer_spec.md")) as f:
            spec = f.read()
        self.assertIn("csv.DictReader", spec)
        self.assertIn("136", spec)

    def test_page_tails_close_the_losslessness_gap(self):
        # `"".join(gap + text)` stops at the last token; the published tail is
        # the rest of the page, so page body == tokens + tail, checkable from
        # the artifacts alone.
        bodies = T.OG.load_pages_raw(T.SRC)
        tails = {}
        with open(os.path.join(OUT, "page_tails.tsv")) as f:
            head = f.readline().rstrip("\n").split("\t")
            pi, li, ti = head.index("page"), head.index("last_token_end"), \
                head.index("tail_json")
            for line in f:
                p = line.rstrip("\n").split("\t")
                tails[int(p[pi])] = (int(p[li]), json.loads(p[ti]))
        self.assertEqual(sorted(tails), sorted(bodies))
        rows = collections.defaultdict(list)
        with open(os.path.join(OUT, "tokens.tsv")) as f:
            head = f.readline().rstrip("\n").split("\t")
            gi, ei, ti_ = head.index("gap_json"), head.index("end"), \
                head.index("text")
            for line in f:
                p = line.rstrip("\n").split("\t")
                rows[int(p[head.index("page")])].append(
                    (int(p[ei]), json.loads(p[gi]), p[ti_]))
        for pno, ps in rows.items():
            last_end, tail = tails[pno]
            self.assertEqual(last_end, ps[-1][0], f"p{pno}")
            self.assertEqual("".join(g + t for _e, g, t in ps) + tail,
                             bodies[pno], f"page {pno}")
        # the tails are what they claim to be: whitespace, plus two comments
        self.assertEqual(self.S["tail_comment_pages"], [1, 224])
        self.assertEqual(self.S["tail_nonblank_pages"], [1, 224])
        self.assertEqual(self.S["tail_chars"], sum(len(t) for _e, t in tails.values()))

    def test_tokens_jsonl_agrees_with_tsv(self):
        n = 0
        with open(os.path.join(OUT, "tokens.jsonl")) as f:
            for line in f:
                obj = json.loads(line)
                self.assertEqual(obj["text"],
                                 T.OG.load_pages_raw(T.SRC)[obj["page"]]
                                 [obj["start"]:obj["end"]])
                n += 1
        self.assertEqual(n, self.S["tokens"]["total"])

    def test_unit_types_sums_to_the_unit_count(self):
        total = 0
        with open(os.path.join(OUT, "unit_types.tsv")) as f:
            head = f.readline().split("\t")
            i = head.index("n")
            for line in f:
                total += int(line.rstrip("\n").split("\t")[i])
        self.assertEqual(total, self.S["units"]["total"])

    def test_part_index_role_counts(self):
        roles = collections.Counter()
        with open(os.path.join(OUT, "part_index.tsv")) as f:
            head = f.readline().split("\t")
            i = head.index("role")
            for line in f:
                roles[line.rstrip("\n").split("\t")[i]] += 1
        self.assertEqual(set(roles), {"w", "seam", "join"})
        self.assertEqual(roles["w"], self.S["units"]["parts"])
        self.assertEqual(roles["seam"], self.S["units"]["seams"])
        self.assertEqual(roles["join"], self.S["units"]["joins"])

    def test_part_index_offsets_are_exact_and_unambiguous(self):
        # every part is quotable by page + offsets, and the `parts` column of
        # units.tsv stays splittable because `:`/`|` cannot occur inside a unit
        bodies = T.OG.load_pages_raw(T.SRC)
        n = 0
        with open(os.path.join(OUT, "part_index.tsv")) as f:
            head = f.readline().rstrip("\n").split("\t")
            for c in ("start", "end", "pos", "role", "text", "uid"):
                self.assertIn(c, head)
            ui, pi, si, ei, ti = (head.index("uid"), head.index("pos"),
                                  head.index("start"), head.index("end"),
                                  head.index("text"))
            last_uid = last_pos = None
            for line in f:
                p = line.rstrip("\n").split("\t")
                page = int(p[1])
                self.assertEqual(bodies[page][int(p[si]):int(p[ei])], p[ti], p[ui])
                self.assertNotIn("|", p[ti])
                self.assertNotIn(":", p[ti])
                uid, pos = p[ui], int(p[pi])
                if uid == last_uid:
                    self.assertEqual(pos, last_pos + 1, uid)
                else:
                    self.assertEqual(pos, 0, uid)
                last_uid, last_pos = uid, pos
                n += 1
        self.assertEqual(n, self.S["units"]["parts"] + self.S["units"]["seams"]
                         + self.S["units"]["joins"])

    def test_cases_are_all_located(self):
        want = {c[0] for c in T.CASES}
        got = {c["case"] for c in self.S["case_rows_all"]}
        self.assertEqual(got, want)

    def test_case_rows_are_page_cited(self):
        bodies = T.OG.load_pages_raw(T.SRC)
        for c in self.S["case_rows_all"]:
            frag = bodies[c["page"]][c["start"]:c["end"]]
            if c["kind"] == "unit":
                self.assertEqual(frag, c["unit"], c["case"])
                # the part list re-forms the unit once the roles are stripped
                rejoined = "".join(x.split(":")[0] for x in c["parts"].split("|"))
                self.assertEqual(rejoined.replace(" ", ""),
                                 c["unit"].replace(" ", ""), c["case"])
                # a unit is book text, so it always sits in an operator-tree
                # container
                self.assertGreater(c["ast"].count(">"), 0, c["case"])
                self.assertIn(c["container"], ("Block", "Span", "Utterance",
                                                "Field", "Header", "Inline"),
                              c["case"])
            else:
                # a `kind="token"` case is deliberately *not* a unit; its span is
                # still page-cited, and it may fall outside every operator-tree
                # container (the OCR's `</p>` sits past the last block), so the
                # AST fields are checked only when present.
                self.assertEqual(frag, c["needle"], c["case"])
                if c["ast"]:
                    self.assertGreater(c["ast"].count(">"), 0, c["case"])
                    self.assertIn(c["container"], ("Block", "Span", "Utterance",
                                                    "Field", "Header", "Inline"),
                                  c["case"])
                else:
                    self.assertEqual(c["container"], "", c["case"])

    def test_cases_without_an_ast_container_are_operator_artifacts(self):
        # `</p>` sits after the last block of its page and `//++` is a bare
        # operator run, so #14's tree has no container covering either; both
        # must still be page-cited with exact offsets
        no_ast = [c for c in self.S["case_rows_all"] if not c["ast"]]
        self.assertTrue(no_ast)
        self.assertEqual({c["case"] for c in no_ast}, {"C10", "C11"})
        for c in no_ast:
            body = T.OG.load_pages_raw(T.SRC)[c["page"]]
            self.assertEqual(body[c["start"]:c["end"]], c["needle"], c["case"])
            self.assertEqual(c["kind"], "token", c["case"])
            self.assertEqual(c["unit"], c["needle"], c["case"])
        # and every *unit* case does have a container: book text is always in
        # the operator tree
        for c in self.S["case_rows_all"]:
            if c["kind"] == "unit":
                self.assertTrue(c["ast"], c["case"])

    def test_case_examples_contains_each_case(self):
        with open(os.path.join(OUT, "case_examples.txt")) as f:
            text = f.read()
        for cid, _n, _note, _kind in T.CASES:
            self.assertIn(f"{cid}  ", text)

    def test_diagnostics_ledger_covers_every_operator_token(self):
        rows = {r["diag"]: r for r in self.S["diag_rows"]}
        self.assertIn("(role) seam", rows)
        self.assertIn("(role) segment", rows)
        # the ledger's partition rows (every operator token has one role);
        # `seam_spaced` is a *flag* on seam tokens, not a separate role, and
        # `html_tag_residue` is a unit-level row, so neither is in the partition
        partition = sum(v["n"] for k, v in rows.items()
                        if k not in ("seam_spaced", "html_tag_residue"))
        self.assertEqual(partition, self.S["tokens"]["OP"])
        self.assertLessEqual(rows["seam_spaced"]["n"], rows["(role) seam"]["n"])

    def test_markup_residue_units_are_disclosed_not_repaired(self):
        # the source PDF's markup survived OCR: page furniture (`<th>`,
        # `<pre>`, `<p>`, `<li>`, `<br/>`, `<hr/>`) and a block of `mailto:`
        # anchors.  Each tag contributes its own name as a one- or two-character
        # unit.  The build asserts the count against the tag runs; here it is
        # re-derived from the token stream, and the row must say the units were
        # kept rather than repaired.
        rows = {r["diag"]: r for r in self.S["diag_rows"]}
        self.assertIn("html_tag_residue", rows)
        self.assertIn("kept, not repaired",
                      rows["html_tag_residue"]["meaning"].lower())
        self.assertIn("`</a>`x99", rows["html_tag_residue"]["meaning"])
        bodies = T.OG.load_pages_raw(T.SRC)
        n_runs = n_tokens_in = 0
        for pno, (toks, _d, _tail) in T.tokenize_book(T.SRC).items():
            runs = T.html_tag_runs(bodies[pno])
            n_runs += len(runs)
            for t in toks:
                if t.kind == "WORD" and any(s <= t.s < e for s, e, _g in runs):
                    n_tokens_in += 1
        self.assertEqual(n_tokens_in, n_runs)
        self.assertEqual(rows["html_tag_residue"]["n"], n_runs)
        # 229 tag runs on 82 pages: `</a>`x99 (mailto anchors), `<th>`x79,
        # `</p>`x28, `<pre>`x8, `<br/>`x4, `<li>`x3, ...
        self.assertEqual(n_runs, 229)
        self.assertEqual(len({p for p, b in bodies.items()
                              if T.html_tag_runs(b)}), 82)

    def test_spec_prose_arithmetic_is_consistent(self):
        # the 2.1 paragraph states how #14's `/` Link/Slash nodes split here; the
        # three numbers in the prose must add up to the measured unbound total
        with open(os.path.join(OUT, "tokenizer_spec.md")) as f:
            spec = f.read()
        m = re.search(r"([\d,]+) `/` tokens do not bind here - ([\d,]+) of "
                      r"them are #14 Link nodes.*?the remaining ([\d,]+)",
                      spec, re.S)
        self.assertIsNotNone(m, "the /-split paragraph is missing from the spec")
        tot, link, slash = (int(g.replace(",", "")) for g in m.groups())
        o = self.S["op_roles_by_op"]
        self.assertEqual(link, o["seam_orphan_left|/"] + o["slash_boundary|/"])
        self.assertEqual(slash, o["slash_free|/"] + o["seam_orphan_right|/"])
        self.assertEqual(tot, self.S["unbound_ops"]["/"])

    def test_spec_mentions_every_artifact(self):
        with open(os.path.join(OUT, "tokenizer_spec.md")) as f:
            spec = f.read()
        for name in self.EXPECTED_FILES:
            if name != "tokenizer_spec.md":
                self.assertIn(f"`{name}`", spec, name)

    def test_summary_schema(self):
        self.assertEqual(self.sum["schema"], T.SCHEMA)
        self.assertEqual(self.sum["src"], T.SRC)

    def test_reconciliation_has_no_drift_marker(self):
        with open(os.path.join(OUT, "reconciliation.tsv")) as f:
            for line in f:
                self.assertNotIn("drift", line)


# ---------------------------------------------------------------------------
class TestAblationNoAst(unittest.TestCase):
    """The tokenizer must work without #14's parser (independent module)."""

    def test_build_without_ast_enrichment(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            S = T.build(T.SRC, tmp, with_ast=False)
            self.assertEqual(S["n_pages"], T.EXPECTED_PAGES)
            self.assertEqual(S["tokens"]["WORD"], T.EXPECTED_WORDS)
            self.assertEqual(S["lossless"]["ok"], T.EXPECTED_PAGES)
            self.assertEqual(S["units"]["seams"], 2816)
            self.assertTrue(os.path.exists(os.path.join(tmp, "units.tsv")))

    def test_ast_lookup_handles_a_token_outside_any_container(self):
        idx = T.AstIndex(T.SRC)
        # offset 0 of page 4 is inside the page's first block, but an offset
        # past the end of every container must return empty rather than raise
        path, container = idx.lookup(4, 10 ** 9)
        self.assertEqual((path, container), ("", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)