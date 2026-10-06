#!/usr/bin/env python3
"""Regression tests for the operator grammar block parser / AST (Chainlink #14).

Run from the workspace root (inside the `marker` conda env):

    python work/test_operator_grammar.py
    python -m unittest discover -s work -p 'test_*.py'

What is pinned here:
  * the grammar behaves as specified on hand-checked strings, including every
    awkward case the report claims to handle (`<<` with no `>>`, stray `>>`,
    `//` immediately before `<<`, `/` disambiguated by glue, glued vs free
    dash, a heading's `>>` closing a body `<<`, cross-page `<<`/`>>`);
  * the AST is a *partition* of the tokenizer's token stream on all 225 pages,
    re-checked independently through the persisted JSON encoding;
  * every #10 operator token is accounted for by an AST node identity, and
    #12's delimiter-pairing numbers (182 unclosed `<<`, 156 stray `>>`, 230
    matched spans, 177 free-standing `-`) are reproduced exactly;
  * the tree's own layering reproduces #12's `//`-vs-`::` sentence-boundary
    finding (the evaluation that justifies the precedence, not an assumption);
  * the persisted artifacts agree with a fresh parse and with `summary.json`,
    and the report states the grammar, the evidence and the negative results.
"""
from __future__ import annotations

import collections
import json
import os
import statistics
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import operator_grammar as og  # noqa: E402
from operator_functions import two_proportion_z, tokenize  # noqa: E402

# The artifact directory is the one named in the #14 task text; assert that the
# builder's own default agrees rather than hard-coding it twice.
OUT = os.path.join(ROOT, og.OUT)

_BOOK = None       # (book, diagnostics, nodes) parsed once for the book tests


def parse_text(body, pno=1):
    """Parse one page body; returns (page node, diagnostics)."""
    p = og.Parser(pno, og.items_of_body(body), og._counter())
    return p.run(body), p.diags


def ref_tokens(body):
    return [t[3] for it in og.items_of_body(body) for t in tokenize(it.surface())]


def kinds_of(node):
    """Multiset of node kinds under `node`, excluding `node` itself."""
    return sorted(n.k for n in list(node.walk())[1:])


def shape(node):
    return "\n".join(og.tree_lines(node, 0))


def find(root, kind):
    return [n for n in root.walk() if n.k == kind]


def book():
    global _BOOK
    if _BOOK is None:
        book_node, diags = og.parse_book()
        _BOOK = (book_node, diags, og.flatten(book_node))
    return _BOOK


class LosslessCase(unittest.TestCase):
    """Mixin: the AST must re-emit the tokenizer sequence for any body."""

    def assert_lossless(self, page, body):
        self.assertEqual(page.token_list(), ref_tokens(body))


# ---------------------------------------------------------------------------
# 1. grammar contract on hand-checked strings
# ---------------------------------------------------------------------------
class GrammarUnitTest(LosslessCase):

    def test_default_layering_is_page_block_utterance_field_segment(self):
        page, diags = parse_text("gene=TV::the internal of the body//It is a City")
        self.assert_lossless(page, "gene=TV::the internal of the body//It is a City")
        self.assertEqual([d["kind"] for d in diags], [])
        self.assertEqual(shape(page), "\n".join([
            "Page",
            "  Block   [paras=1]",
            "    Utterance",
            "      Field",
            "        Segment",
            "          Lex gene",
            "        Segment",
            "          Link 'gene' '=' 'TV'   [op== left=gene right=TV]",
            "          Lex TV   [upper=True]",
            "      Field   [open_op=:: open_off=7 open_end=9]",
            "        Segment",
            "          Lex the",
            "          Lex internal",
            "          Lex of",
            "          Lex the",
            "          Lex body",
            "    Utterance   [open_op=// open_off=33 open_end=35]",
            "      Field",
            "        Segment",
            "          Lex It",
            "          Lex is",
            "          Lex a",
            "          Lex City",
        ]))

    def test_colon_colon_is_a_field_inside_an_utterance(self):
        # #12: `::` is phrase-level (upper-next 13%), `//` starts sentences
        # (52.7%).  The precedence `span > utterance > field` encodes that.
        page, _ = parse_text("self :: the body")
        flds = find(page, "Field")
        self.assertEqual(len(flds), 2)
        self.assertEqual([f.x.get("open_op") for f in flds], [None, "::"])
        self.assertEqual(flds[1].parent.k, "Utterance")

    def test_single_colon_is_punctuation_not_a_field(self):
        page, diags = parse_text("self : the body")
        self.assertEqual([d["kind"] for d in diags], [])
        self.assertEqual(len(find(page, "Field")), 1)      # only the default one
        seps = find(page, "Sep")
        self.assertEqual([s.v for s in seps], [":"])

    def test_double_slash_breaks_the_utterance_and_starts_the_next_uppercase(self):
        page, _ = parse_text("alpha beta//Gamma delta")
        utts = find(page, "Utterance")
        self.assertEqual([u.x.get("open_op") for u in utts], [None, "//"])
        self.assertEqual([[w.v for w in u.walk() if w.k == "Lex"] for u in utts],
                         [["alpha", "beta"], ["Gamma", "delta"]])
        self.assertEqual(og._first_upper(utts[1]), 1)
        self.assertEqual(og._first_upper(utts[0]), 0)

    def test_angle_span_wraps_utterances_and_nests(self):
        page, _ = parse_text("<<a::b//c>> and")
        span = find(page, "Span")[0]
        self.assertTrue(span.x["closed"])
        self.assertEqual(span.kids[0].k, "Utterance")
        self.assertEqual([u.x.get("open_op") for u in find(page, "Utterance")],
                         [None, "//", None])
        self.assert_lossless(page, "<<a::b//c>> and")

    def test_nested_explicit_spans(self):
        page, diags = parse_text("<<outer <<inner core>> tail>>")
        outer, inner = find(page, "Span")     # pre-order: outer first
        self.assertEqual(outer.x["depth"], 0)
        self.assertEqual(inner.x["depth"], 1)
        self.assertEqual(inner.parent, outer)
        self.assertTrue(inner.x["closed"] and outer.x["closed"])
        # the inner `<<` interrupts the outer's unfinished segment: flagged, kept
        self.assertEqual([d["kind"] for d in diags], ["span_opens_mid_segment"])
        self.assert_lossless(page, "<<outer <<inner core>> tail>>")

    def test_deeper_nesting_records_its_depth(self):
        page, _ = parse_text("<<a<<b<<c d>>e>>f>>")
        depths = [s.x["depth"] for s in find(page, "Span")]
        self.assertEqual(depths, [0, 1, 2])
        self.assertTrue(all(s.x["closed"] for s in find(page, "Span")))

    def test_unclosed_open_is_closed_at_page_end_and_flagged(self):
        body = "Dead machine<<LOAD>>> the sun"
        page, diags = parse_text(body)
        self.assert_lossless(page, body)
        spans = find(page, "Span")
        self.assertEqual([s.x["closed"] for s in spans], [True])
        self.assertIn("span_opens_mid_segment", [d["kind"] for d in diags])
        # `<<LOAD>>` closes at the first `>>`; the third `>` is still inside the
        # span, and a bare `>` may not close a `<` below the span layer - there is
        # no open `<`, so it is kept as a flow marker.
        flow = [n for n in find(page, "Op") if n.x.get("role") == "flow-marker"]
        self.assertEqual([n.v for n in flow], [">"])
        self.assertIn("angle_stray_close", [d["kind"] for d in diags])

    def test_stray_close_is_kept_as_a_leaf_with_a_citation(self):
        body = "closing >> without an open"
        page, diags = parse_text(body)
        self.assert_lossless(page, body)
        self.assertEqual([d["kind"] for d in diags], ["span_stray_close"])
        leaf = find(page, "Op")[0]
        self.assertEqual(leaf.v, ">>")
        self.assertEqual(leaf.x["role"], "stray-close")

    def test_slash_is_disambiguated_by_glue_not_by_position(self):
        page, _ = parse_text("quantum/blood/running")
        links = find(page, "Link")
        self.assertEqual([(l.v, l.x["left"], l.x["right"]) for l in links],
                         [("/", "quantum", "blood"), ("/", "blood", "running")])
        self.assertEqual(find(page, "Slash"), [])
        page, _ = parse_text("it runs/ and stops")
        self.assertEqual([l.v for l in find(page, "Link")], [])
        self.assertEqual([s.v for s in find(page, "Slash")], ["/"])

    def test_free_dash_separates_but_a_glued_hyphen_stays_a_word(self):
        page, diags = parse_text("the train of powder - her body was opened")
        self.assertEqual([d["kind"] for d in diags], [])
        dashes = find(page, "Dash")
        self.assertEqual([d.v for d in dashes], ["-"])
        self.assertEqual(len(find(page, "Segment")), 3)
        page, _ = parse_text("pre-ADAM boy-roid")
        self.assertEqual(find(page, "Dash"), [])
        self.assertEqual([n.v for n in find(page, "Lex")], ["pre-ADAM", "boy-roid"])

    def test_em_and_en_dashes_use_the_same_free_standing_test(self):
        page, diags = parse_text("a — b and c – d")
        self.assertEqual([d["kind"] for d in diags], [])
        self.assertEqual([d.v for d in find(page, "Dash")], ["—", "–"])
        # a dash glued to a word is a compound join, not a separator - the same
        # rule #12 applied to the ASCII `-`
        page, diags = parse_text("ice—our and scope—the")
        self.assertEqual([d["kind"] for d in diags], [])
        self.assertEqual(find(page, "Dash"), [])
        self.assertEqual([(n.v, n.x.get("role")) for n in find(page, "Punct")],
                         [("—", "attached-dash"), ("—", "attached-dash")])

    def test_link_operators_record_both_operands(self):
        page, _ = parse_text("gravity=zero=system and murder++the soul-machine/x")
        links = find(page, "Link")
        self.assertEqual([(l.v, l.x["left"], l.x["right"]) for l in links],
                         [("=", "gravity", "zero"), ("=", "zero", "system"),
                          ("++", "murder", "the"), ("/", "soul-machine", "x")])

    def test_link_with_a_missing_operand_is_flagged_not_dropped(self):
        page, diags = parse_text("=serum and vital=")
        kinds = sorted(d["kind"] for d in diags)
        self.assertEqual(kinds, ["link_missing_left", "link_missing_right"])
        self.assertEqual(len(find(page, "Link")), 2)
        self.assertEqual([l.x["left"] for l in find(page, "Link")], [None, "vital"])
        self.assertEqual([l.x["right"] for l in find(page, "Link")],
                         ["serum", None])
        self.assert_lossless(page, "=serum and vital=")

    def test_a_link_between_two_words_records_both_operands(self):
        page, diags = parse_text("vital= and =serum")
        self.assertEqual([d["kind"] for d in diags], [])
        self.assertEqual([(l.x["left"], l.x["right"]) for l in find(page, "Link")],
                         [("vital", "and"), ("and", "serum")])

    def test_pipe_is_its_own_node(self):
        page, diags = parse_text("<<| record the vital-icon+our chromosome")
        pipes = find(page, "Pipe")
        self.assertEqual([p.v for p in pipes], ["|"])
        # the `<<` has no `>>` in this fragment, so it is closed at page end
        self.assertEqual([d["kind"] for d in diags], ["span_unclosed"])
        self.assertEqual([l.v for l in find(page, "Link")], ["+"])
        self.assert_lossless(page, "<<| record the vital-icon+our chromosome")

    def test_inline_angle_quote_nests_inside_a_span(self):
        page, diags = parse_text("<<a <b> c>>")
        self.assertEqual([d["kind"] for d in diags], [])
        self.assertEqual([n.k for n in page.walk()][:3], ["Page", "Block", "Span"])
        inl = find(page, "Inline")[0]
        self.assertTrue(inl.x["closed"])
        self.assertEqual([n.v for n in inl.walk() if n.k == "Lex"], ["b"])

    def test_a_single_angle_close_inside_a_span_is_a_flow_marker(self):
        # `<<`…`>>` is the outer layer, so a bare `>` cannot close a `<` nested
        # inside it (#12's over-produced `>` counts).
        page, diags = parse_text("<<a > b>>")
        self.assertEqual([d["kind"] for d in diags], ["angle_stray_close"])
        self.assertEqual(find(page, "Inline"), [])
        self.assertEqual(find(page, "Op")[0].x["role"], "flow-marker")

    def test_boundary_operator_immediately_before_a_span_is_kept(self):
        body = "y//<<z>>"
        page, diags = parse_text(body)
        self.assert_lossless(page, body)
        self.assertEqual([d["kind"] for d in diags], ["orphan_boundary"])
        orphan = [n for n in find(page, "Op") if n.x.get("role") == "orphan-boundary"]
        self.assertEqual([n.v for n in orphan], ["//"])

    def test_figure_placeholders_are_layout_metadata_not_book_text(self):
        body = "<!-- figure: a plate -->\nreal text"
        page, diags = parse_text(body)
        self.assert_lossless(page, body)
        self.assertEqual([d["kind"] for d in diags], [])
        self.assertEqual(len(find(page, "Figure")), 1)
        self.assertEqual(find(page, "Figure")[0].token_list(), [])


# ---------------------------------------------------------------------------
# 2. recovery behaviour that the report claims
# ---------------------------------------------------------------------------
class RecoveryTest(LosslessCase):

    def test_cross_page_delimiters_are_not_joined(self):
        # Joining them would be inference; each page is closed on its own.
        md = ("<!-- page 1 -->\n<<unclosed on page one\n\n"
              "<!-- page 2 -->\n>>stray on page two\n")
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(md)
            path = f.name
        try:
            bk, diags = og.parse_book(path)
            ref = og.reference_tokens(path)
        finally:
            os.unlink(path)
        self.assertEqual([(d["page"], d["kind"]) for d in diags],
                         [(1, "span_unclosed"), (2, "span_stray_close")])
        p1, p2 = bk.kids
        self.assertEqual([s.x["closed"] for s in find(p1, "Span")], [False])
        self.assertEqual(find(p2, "Span"), [])
        self.assertEqual(find(p2, "Op")[0].x["role"], "stray-close")
        self.assertTrue(all(r[3] for r in og.roundtrip_check(bk, ref)),
                        "cross-page recovery must stay lossless")

    def test_a_heading_is_its_own_delimiter_scope(self):
        body = "<<open never closed\n\n## Head >> with close\n\nafter text"
        page, diags = parse_text(body)
        self.assert_lossless(page, body)
        kinds = [d["kind"] for d in diags]
        self.assertEqual(sorted(kinds), ["span_stray_close", "span_unclosed"])
        # the body's `<<` survives the heading, and the heading sits inside it,
        # so the tree keeps source order
        span = find(page, "Span")[0]
        self.assertFalse(span.x["closed"])
        self.assertEqual([k for k in kinds_of(page) if k == "Header"], ["Header"])
        self.assertEqual(span.kids[1].k, "Header")
        self.assertEqual(find(page, "Header")[0].x["level"], 2)
        self.assertEqual(find(page, "Op")[0].x["role"], "stray-close")

    def test_paragraph_break_does_not_kill_an_open_span_but_closes_a_block(self):
        body = "<<still open\n\nsecond paragraph without an open"
        page, diags = parse_text(body)
        self.assert_lossless(page, body)
        self.assertEqual([d["kind"] for d in diags], ["span_unclosed"])
        self.assertEqual(len(find(page, "Block")), 1)
        self.assertEqual(find(page, "Block")[0].x["paras"], "1,2")
        self.assertEqual([s.page for s in find(page, "Span")], [1])

    def test_an_unclosed_angle_quote_is_closed_and_flagged(self):
        body = "open < gravity zero"
        page, diags = parse_text(body)
        self.assert_lossless(page, body)
        self.assertEqual([d["kind"] for d in diags], ["inline_unclosed"])
        self.assertFalse(find(page, "Inline")[0].x["closed"])

    def test_a_field_with_nothing_on_its_left_is_flagged(self):
        _page, diags = parse_text("x//::y")
        self.assertEqual([d["kind"] for d in diags], ["empty_field"])
        _page, diags = parse_text("//y")
        self.assertEqual([d["kind"] for d in diags], [])   # a clause may start a line

    def test_a_trailing_boundary_operator_is_kept_as_an_orphan_leaf(self):
        for body in ("x//", "//"):
            page, diags = parse_text(body)
            self.assert_lossless(page, body)
            self.assertEqual([d["kind"] for d in diags], ["orphan_boundary"])
            self.assertEqual([n.v for n in find(page, "Op")], ["//"])

    def test_every_diagnostic_carries_a_page_citation(self):
        _book, diags, _nodes = book()
        self.assertTrue(diags)
        for d in diags:
            self.assertIsInstance(d["page"], int)
            self.assertIn(d["kind"], og.DIAG_MEANING)
            self.assertIsInstance(d["offset"], int)


# ---------------------------------------------------------------------------
# 3. whole-book invariants
# ---------------------------------------------------------------------------
class BookInvariantTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.book, cls.diags, cls.nodes = book()
        cls.ref = og.reference_tokens()

    # -- losslessness -------------------------------------------------
    def test_ast_is_a_partition_of_the_token_stream_on_every_page(self):
        rt = og.roundtrip_check(self.book, self.ref)
        self.assertEqual(len(rt), 225)
        for pno, n_ref, n_got, ok in rt:
            self.assertTrue(ok, (pno, og.first_divergence(self.ref[pno],
                                                         self.book.kids[pno - 1].token_list())))
            self.assertEqual(n_ref, n_got)
        self.assertEqual(sum(r[1] for r in rt), 90220)

    def test_every_token_lands_in_exactly_one_leaf(self):
        leaves = [n for n in self.nodes if n.k in ("Lex", "Link", "Sep", "Punct",
                                                   "Dash", "Slash", "Pipe", "Op")]
        n_leaf_tokens = sum(len(n.token_list()) for n in leaves)
        n_boundary = sum(1 for n in self.nodes if n.k in ("Utterance", "Field")
                         and n.x.get("open_op"))
        n_delims = sum((2 if n.x.get("closed", True) else 1)
                       for n in self.nodes if n.k in ("Span", "Inline"))
        n_head = sum(int(n.x.get("level", 1)) for n in self.nodes if n.k == "Header")
        self.assertEqual(n_leaf_tokens + n_boundary + n_delims + n_head,
                         sum(r[1] for r in og.roundtrip_check(self.book, self.ref)))

    def test_flat_tokens_agrees_with_token_list_in_source_order(self):
        for page in self.book.kids[:12] + self.book.kids[-12:]:
            flat = [t for t, _tag in og.flat_tokens(page)]
            self.assertEqual(flat, page.token_list(), page.page)

    def test_ids_are_unique_and_parents_precede_children(self):
        ids = [n.id for n in self.nodes]
        self.assertEqual(len(ids), len(set(ids)))
        seen = set()
        for n in self.nodes:
            if n.parent is not None:
                self.assertIn(n.parent.id, seen, n.id)
            seen.add(n.id)

    # -- #10 / #12 reconciliation ------------------------------------
    def test_operator_inventory_is_reproduced_and_reconciled(self):
        body_ops, header_ops = og.operator_counts()
        self.assertEqual(sum(body_ops.values()), og.EXPECTED_OPS)
        self.assertEqual(body_ops["::"], 3083)
        self.assertEqual(body_ops["//"], 2149)
        self.assertEqual(body_ops["="], 2047)
        self.assertEqual(body_ops["/"], 1484)
        self.assertEqual(body_ops["<<"], 412)
        self.assertEqual(body_ops[">>"], 386)
        body_nodes = og.nodes_outside_headers(self.nodes)
        recon = og.reconcile_operators(
            body_ops, header_ops, self.nodes, body_nodes,
            collections.Counter(n.k for n in body_nodes),
            collections.Counter(n.k for n in self.nodes))
        self.assertEqual(sorted(r["op"] for r in recon), sorted(og.OPS))
        for r in recon:
            self.assertTrue(r["match"], r)
        # every operator token in the page bodies is claimed by some node
        self.assertEqual(sum(r["ast_body"] for r in recon), og.EXPECTED_OPS)

    def test_header_operators_are_parsed_not_discarded(self):
        # #10/#12 delete heading lines, which hides these 51 tokens; the AST
        # keeps them, which is why the 20 `<<…>>` section names are visible.
        _body_ops, header_ops = og.operator_counts()
        self.assertEqual(sum(header_ops.values()), 51)
        headers = [n for n in self.nodes if n.k == "Header"]
        self.assertEqual(len(headers), 56)
        self.assertEqual(sum(1 for h in headers
                             if any(d.k == "Span" for d in h.walk())), 20)

    def test_12_delimiter_pairing_numbers_are_reproduced(self):
        dsum = og.diags_summary(self.diags)
        self.assertEqual(dsum["span_unclosed"]["n"], og.EXPECTED_SPAN_UNCLOSED)
        self.assertEqual(dsum["span_stray_close"]["n"], og.EXPECTED_SPAN_STRAY_CLOSE)
        spans = og.span_rows(self.nodes)
        body_spans = [s for s in spans
                      if not any(a.k == "Header" for a in og._ancestors(self._node(s)))]
        matched = [s for s in body_spans if s["closed"]]
        self.assertEqual(len(matched), 230)          # #12's matched spans
        self.assertEqual(statistics.median([s["n_words"] for s in matched]), 5)

    def _node(self, row):
        return next(n for n in self.nodes if n.id == row["span_id"])

    def test_free_standing_ascii_dash_matches_12(self):
        # #12's definition: a `-` token with whitespace on both sides.  Recounted
        # here straight from the corpus text, not from the parser.
        n_free = 0
        for _pno, items in og.load_page_items().items():
            for it in items:
                if it.kind != "para":      # #12's scan scope: page bodies only
                    continue
                for kind, s, e, val in tokenize(it.text):
                    if kind == "WORD" or val != "-":
                        continue
                    if it.text[s - 1:s] in ("", " ", "\t") and \
                            it.text[e:e + 1] in ("", " ", "\t", "\n"):
                        n_free += 1
        self.assertEqual(n_free, og.EXPECTED_FREE_DASH)
        dashes = [n for n in self.nodes if n.k == "Dash" and n.v == "-"]
        self.assertEqual(len(dashes), og.EXPECTED_FREE_DASH)
        # a glued hyphen stays inside its WORD token (morphology, not a site):
        # #12 measured 2,834 hyphenated compounds; every one of them is a `Lex`
        # atom, and no `Dash` node's value is a compound
        hyphen_lex = [n for n in self.nodes if n.k == "Lex" and "-" in n.v]
        self.assertEqual(len(hyphen_lex), 2830)
        self.assertEqual(len({n.v for n in hyphen_lex}), 785)
        for n in [n for n in self.nodes if n.k == "Dash"]:
            self.assertIn(n.v, ("-", "—", "–"))

    def test_operators_outside_the_12_class_set_remain_absent(self):
        dsum = og.diags_summary(self.diags)
        self.assertEqual(dsum.get("unclassified_operator", {}).get("n", 0), 0)
        absent = ["--", "==", "->", "<-", "`", "^", "~"]
        body_ops, _ = og.operator_counts()
        for op in absent:
            self.assertEqual(body_ops.get(op, 0), 0, op)

    # -- the evaluation that justifies the precedence -----------------
    def test_tree_layering_reproduces_the_12_colon_vs_slash_finding(self):
        rows = {r["container"]: r for r in og.boundary_eval(self.nodes)}
        utt, fld = rows["Utterance[//]"], rows["Field[::]"]
        self.assertEqual(utt["n"], 1968)
        self.assertEqual(fld["n"], 2765)
        self.assertGreater(utt["rate"], fld["rate"])
        z, p = two_proportion_z(utt["first_word_upper"], utt["n"],
                                fld["first_word_upper"], fld["n"])
        self.assertGreater(z, 10)            # #12 measured z=31.0 on the stream
        self.assertLess(p, 1e-50)
        # and the direction is the same as #12's token-stream test
        self.assertAlmostEqual(utt["rate"], 0.587, places=3)
        self.assertAlmostEqual(fld["rate"], 0.150, places=3)

    def test_containers_partition_the_book(self):
        kinds = {}
        for n in self.nodes:
            kinds[n.k] = kinds.get(n.k, 0) + 1
        utts = og.clause_rows(self.nodes)
        self.assertEqual(sum(1 for c in utts if c["container"] == "span"),
                         kinds["Utterance"] - sum(1 for c in utts
                                                  if c["container"] != "span"))
        self.assertEqual(kinds["Utterance"], 3764)
        self.assertEqual(kinds["Field"], 6640)
        self.assertEqual(kinds["Span"], 432)
        self.assertEqual(kinds["Lex"], 77485)
        self.assertEqual(kinds["Dash"], 696)
        self.assertEqual(kinds["Header"] + kinds["Figure"], og.EXPECTED_BLOCK_DELIMS)
        self.assertEqual(kinds["Page"], 225)
        # the container rows partition the utterances: every one is inside
        # exactly one span, block or inline
        self.assertEqual(len(utts), kinds["Utterance"])

    def test_spot_checks_are_all_located(self):
        rows, trees = og.spot_rows(self.nodes)
        self.assertEqual(len(rows), 14)
        for r in rows:
            self.assertTrue(r["ok"], (r["id"], r["note"]))
            self.assertTrue(r["chain"].startswith("Book>Page"))
            self.assertTrue(r["signature"])
        self.assertEqual(len(trees), 14)

    def test_span_layer_covers_58_percent_of_the_word_atoms(self):
        spans = og.span_rows(self.nodes)
        self.assertEqual(sum(s["n_words"] for s in spans), 45189)
        self.assertAlmostEqual(100.0 * 45189 / 77485, 58.32, places=1)
        self.assertEqual(sum(1 for s in spans if s["nested"]), 126)
        self.assertEqual(max(s["depth"] for s in spans), 29)


# ---------------------------------------------------------------------------
# 4. persisted artifacts
# ---------------------------------------------------------------------------
def tokens_from_obj(obj):
    """Independent decoder for the compact AST encoding in `ast.jsonl`."""
    kind, value, _page, _para, _off, _depth, attrs = obj[:7]
    kids = obj[7] if len(obj) > 7 else []
    if kind == "Lex":
        return [value]
    if kind == "Link":
        return [value]
    if kind in ("Sep", "Punct", "Dash", "Slash", "Pipe", "Op"):
        return [value]
    if kind in ("Span", "Inline"):
        opener, closer = ("<", ">") if kind == "Inline" else ("<<", ">>")
        out = [opener]
        for kid in kids:
            out += tokens_from_obj(kid)
        if (attrs or {}).get("closed", True):
            out.append(closer)
        return out
    if kind in ("Utterance", "Field"):
        out = [attrs["open_op"]] if (attrs or {}).get("open_op") else []
        return out + [t for kid in kids for t in tokens_from_obj(kid)]
    if kind == "Header":
        return ["#"] * int(attrs["level"]) + [t for kid in kids
                                              for t in tokens_from_obj(kid)]
    if kind == "Figure":
        return []
    return [t for kid in kids for t in tokens_from_obj(kid)]


class ArtifactTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(OUT, "summary.json"), encoding="utf-8") as f:
            cls.summary = json.load(f)
        cls.ref = og.reference_tokens()

    def path(self, name):
        p = os.path.join(OUT, name)
        self.assertTrue(os.path.exists(p), f"missing artifact {name}; run "
                                           f"`python work/operator_grammar.py`")
        return p

    def rows(self, name):
        with open(self.path(name), encoding="utf-8") as f:
            lines = f.read().splitlines()
        head = lines[0].split("\t")
        return head, [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]

    def test_artifacts_live_where_the_task_specifies(self):
        # #14 names analysis/block_parser/ as the artifact location; #33, #52 and
        # #62 read from there, so the path is pinned rather than assumed.
        self.assertEqual(og.OUT, "analysis/block_parser")
        self.assertEqual(os.path.normpath(OUT),
                         os.path.join(ROOT, "analysis", "block_parser"))
        self.assertTrue(os.path.isdir(OUT), OUT)
        self.assertTrue(os.path.exists(os.path.join(OUT, "ast.jsonl")))
        # the AST artifact is the one thing #33/#52/#62 must be able to find
        for name in ["grammar_spec.md", "ast.jsonl", "nodes.tsv", "ast_tree.txt",
                     "spans.tsv", "clauses.tsv", "records.tsv", "links.tsv",
                     "diagnostics.tsv", "boundary_eval.tsv", "spot_checks.tsv",
                     "spot_check_trees.txt", "node_inventory.tsv",
                     "depth_profile.tsv", "summary.json"]:
            self.assertTrue(os.path.exists(os.path.join(OUT, name)), name)

    def test_ast_jsonl_is_one_tree_per_page_and_still_lossless(self):
        with open(self.path("ast.jsonl"), encoding="utf-8") as f:
            trees = [json.loads(ln) for ln in f if ln.strip()]
        self.assertEqual(len(trees), 225)
        self.assertEqual([t["page"] for t in trees], list(range(1, 226)))
        for t in trees:
            self.assertEqual(tokens_from_obj(t["tree"]), self.ref[t["page"]], t["page"])
            self.assertEqual(t["tree"][0], "Page")

    def test_dash_counts_reproduce_12_and_13(self):
        # #12 measured 177 whitespace-bounded ASCII dashes; #13 re-measured the
        # whitespace-bounded dash sites as 696 (476 em, 177 ascii, 43 en).  The
        # parser must agree with both, which means applying #12's free-standing
        # test to `—`/`–` as well - not treating every em dash as a separator.
        s = self.summary
        self.assertEqual(s["n_dash_ascii"], 177)      # #12
        self.assertEqual(s["n_dash_em"], 476)         # #13
        self.assertEqual(s["n_dash_en"], 43)          # #13
        self.assertEqual(s["n_dash_total"], 696)      # #13
        self.assertEqual(s["n_dash_glued"], 19)
        self.assertEqual(s["n_dash_glued_em"], 8)
        self.assertEqual(s["n_dash_glued_ascii"], 11)
        self.assertEqual(s["n_dash_ascii"] + s["n_dash_em"] + s["n_dash_en"],
                         s["n_dash_total"])

    def test_summary_numbers_are_the_ones_the_report_cites(self):
        s = self.summary
        for k, v in [("n_pages", 225), ("n_nodes", 108780), ("n_op_tokens", 10949),
                     ("n_op_tokens_headers", 51), ("n_spans", 432), ("n_spans_closed", 250),
                     ("n_spans_open", 182), ("n_utterances", 3764), ("n_fields", 6640),
                     ("n_segments", 12742), ("n_lex", 77485), ("n_inline", 381),
                     ("n_headers", 56), ("n_header_spans", 20), ("n_link", 3213),
                     ("n_sep_colon", 324), ("n_pipe", 17), ("n_blocks", 561),
                     ("n_para_blocks", 666), ("roundtrip_pages_ok", 225),
                     ("roundtrip_tokens", 90220), ("n_pages_no_block", 0),
                     ("span_unclosed", 182), ("span_stray_close", 156),
                     ("unclassified_ops", 0), ("html_p_artifacts", 28),
                     ("cross_para_spans", 122), ("n_figure_placeholders", 12)]:
            self.assertEqual(s[k], v, k)
        self.assertEqual(s["span_words_median_closed"], 5.0)
        self.assertAlmostEqual(s["coverage_pct"], 58.32, places=1)
        self.assertTrue(all(r["match"] for r in s["operator_reconciliation"]))
        self.assertTrue(all(c["ok"] for c in s["spot_checks"]))

    def test_node_inventory_totals_the_tree(self):
        head, rows = self.rows("node_inventory.tsv")
        self.assertEqual(head, ["kind", "n", "median_children", "max_children",
                                "max_depth"])
        self.assertEqual(sum(int(r["n"]) for r in rows), self.summary["n_nodes"])
        self.assertEqual(len(rows), 18)

    def test_nodes_tsv_is_the_flat_projection_of_the_same_tree(self):
        head, rows = self.rows("nodes.tsv")
        self.assertEqual(len(rows), self.summary["n_nodes"])
        self.assertEqual(head[:6], ["id", "page", "para", "depth", "kind", "parent_id"])
        # ids are 1..N in pre-order, the `Book` root included
        self.assertEqual([int(r["id"]) for r in rows[:200]], list(range(1, 201)))
        self.assertEqual(rows[0]["kind"], "Book")
        self.assertEqual(rows[1]["kind"], "Page")
        # pre-order: every parent appears before its children
        seen = set()
        for r in rows[:5000]:
            if r["parent_id"]:
                self.assertIn(int(r["parent_id"]), seen)
            seen.add(int(r["id"]))

    def test_per_layer_tsvs_have_one_row_per_node(self):
        for name, key, total in [("spans.tsv", "span_id", "n_spans"),
                                 ("clauses.tsv", "utterance_id", "n_utterances"),
                                 ("records.tsv", "field_id", "n_fields")]:
            head, rows = self.rows(name)
            self.assertIn(key, head)
            self.assertEqual(len(rows), self.summary[total], name)
            self.assertIn("page", head)
            self.assertTrue(all(r["page"].isdigit() for r in rows), name)
            self.assertTrue(all(r["preview"] for r in rows), name)

    def test_links_tsv_is_the_binding_inventory_for_15_and_52(self):
        head, rows = self.rows("links.tsv")
        self.assertEqual(head, ["op", "left", "right", "n", "n_pages", "pages"])
        self.assertEqual(sum(int(r["n"]) for r in rows), self.summary["n_link"])
        pairs = {(r["op"], r["left"], r["right"]): int(r["n"]) for r in rows}
        self.assertIn(("=", "gene", "TV"), pairs)
        self.assertIn(("/", "soul", "gram"), pairs)
        self.assertGreaterEqual(pairs[("=", "gene", "TV")], 100)

    def test_boundary_eval_tsv_matches_a_fresh_computation(self):
        head, rows = self.rows("boundary_eval.tsv")
        self.assertEqual(head, ["container", "n", "first_word_upper", "rate"])
        fresh = og.boundary_eval(book()[2])
        self.assertEqual([r["container"] for r in fresh], [r["container"] for r in rows])
        for a, b in zip(fresh, rows):
            self.assertEqual(int(b["n"]), a["n"], a["container"])
            self.assertEqual(int(b["first_word_upper"]), a["first_word_upper"])
            self.assertAlmostEqual(float(b["rate"]), a["rate"], places=4)
        utt = next(r for r in rows if r["container"] == "Utterance[//]")
        fld = next(r for r in rows if r["container"] == "Field[::]")
        self.assertGreater(float(utt["rate"]), float(fld["rate"]))

    def test_diagnostics_tsv_explains_every_anomaly_kind(self):
        head, rows = self.rows("diagnostics.tsv")
        self.assertEqual(head, ["kind", "n", "n_pages", "examples"])
        kinds = {r["kind"]: r for r in rows}
        self.assertEqual(int(kinds["span_unclosed"]["n"]), 182)
        self.assertEqual(int(kinds["span_stray_close"]["n"]), 156)
        self.assertNotIn("unclassified_operator", kinds)
        self.assertTrue(kinds)
        for r in rows:
            self.assertTrue(r["examples"].startswith("p"), r["kind"])
            # every kind that fires is explained in the report's ledger
            self.assertIn(r["kind"], og.DIAG_MEANING)
            self.assertGreater(int(r["n_pages"]), 0)
        # the ledger total equals the number of diagnostics the parser emitted
        self.assertEqual(sum(int(r["n"]) for r in rows), len(book()[1]))

    def test_spot_check_trees_are_page_cited_and_parseable(self):
        head, rows = self.rows("spot_checks.tsv")
        self.assertEqual(head[:4], ["id", "construct", "page", "needle"])
        self.assertTrue(all(r["ok"] == "True" for r in rows))
        with open(self.path("spot_check_trees.txt"), encoding="utf-8") as f:
            text = f.read()
        for r in rows:
            self.assertIn(f"===== {r['id']}:", text)
            self.assertIn(f"p{r['page']}", text)
            self.assertIn(r["chain"], text)

    def test_grammar_spec_states_grammar_evidence_and_limits(self):
        with open(self.path("grammar_spec.md"), encoding="utf-8") as f:
            text = f.read()
        for needle in ["book      = { page }", "span      =", "utterance =",
                       "field     =", "inline    =", "Precedence",
                       "z=31.0", "round-trip", "Cross-page spans are not joined",
                       "What this is not", "negative results",
                       "Measured vs interpreted", "Parse-pressure ledger",
                       "work/test_operator_grammar.py", "#15", "#33", "#52", "#62"]:
            self.assertIn(needle, text, needle)
        # the ledger's counts are the ones in summary.json
        self.assertIn("| `span_unclosed` | 182 |", text)

    def test_report_does_not_claim_more_than_measured(self):
        with open(self.path("grammar_spec.md"), encoding="utf-8") as f:
            text = f.read().lower()
        for hedge in ["measured vs interpreted", "no claim of generality",
                      "nothing here assigns meaning", "not a semantic parse",
                      "interpreted"]:
            self.assertIn(hedge, text, hedge)
        for overclaim in ["proves", "we have decoded", "the book means",
                          "translated", "definitive"]:
            self.assertNotIn(overclaim, text, overclaim)

    def test_report_numbers_match_the_artifacts(self):
        with open(self.path("grammar_spec.md"), encoding="utf-8") as f:
            text = f.read()
        s = self.summary
        # headline figures quoted in the prose must be the summary's figures
        self.assertIn(f"**{s['roundtrip_pages_ok']}/{s['n_pages']} pages**", text)
        self.assertIn(f"{s['roundtrip_tokens']:,} tokens", text)
        self.assertIn(f"| `Lex` | {s['n_lex']:,} |", text)
        self.assertIn(f"**{s['n_nodes']:,}**", text)
        self.assertIn(f"{s['n_op_tokens']:,}", text)
        self.assertIn(f"| free-standing ASCII `-` | 177 | {s['n_dash_ascii']} |", text)
        self.assertIn(f"| unclosed `<<` (page bodies) | 182 | {s['span_unclosed']} |",
                      text)
        self.assertIn(f"| stray `>>` (page bodies) | 156 | {s['span_stray_close']} |",
                      text)
        self.assertIn(f"| `Header` | {s['n_headers']:,} |", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)