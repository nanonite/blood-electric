#!/usr/bin/env python3
"""Regression tests for the silence-removal transform variants (Chainlink #13).

Run from the workspace root (inside the `marker` conda env):

    python work/test_silence_removal.py
    python -m unittest discover -s work -p 'test_*.py'

What is pinned here:
  * the silence-site inventory reproduces #10's 10,949 operator tokens and adds
    the measured 696 whitespace-bounded dash sites;
  * every variant's word-token count is fully explained by the transform
    (deletes join lexical items, separators cannot change the token sequence at
    all), so a silently broken transform fails;
  * the four issue-named transforms behave as specified on hand-checked strings,
    including the awkward cases (operator runs, hyphen bridges, pre-existing
    periods, word-internal hyphens);
  * the persisted artifacts agree with summary.json and with #12's numbers.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import silence_removal as sr  # noqa: E402
from operator_functions import tokenize  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "silence_removal")
STREAMS = [k for k, *_ in sr.VARIANTS]
ISSUE_VARIANTS = ["v1_delete_all_operators", "v2_delete_colon_slash",
                  "v3_operators_to_space", "v4_operators_to_period"]


def apply_variant(line, key):
    treatment, only, sep = next((t, o, s) for k, t, o, s in sr.VARIANTS if k == key)
    new, _n, _f = sr.transform_line(line, sr.site_spans(line), treatment, only, sep)
    return sr.normalize_line(new)


class TransformUnitTest(unittest.TestCase):
    """Hand-checked strings: the transform contract."""

    def test_delete_all_fuses_glued_neighbours(self):
        self.assertEqual(apply_variant("soul/gram", "v1_delete_all_operators"), "soulgram")
        self.assertEqual(apply_variant("gene=TV", "v1_delete_all_operators"), "geneTV")

    def test_delete_all_keeps_whitespace_separated_items_separate(self):
        # deleting `::` between spaces must not weld the two words together
        self.assertEqual(apply_variant("the horizon :: of the body", "v1_delete_all_operators"),
                         "the horizon of the body")

    def test_delete_colon_slash_only(self):
        # only :: and // go; = / << >> survive verbatim
        self.assertEqual(apply_variant("gene=TV::x//y", "v2_delete_colon_slash"),
                         "gene=TVxy")
        self.assertEqual(apply_variant("a=b/c", "v2_delete_colon_slash"), "a=b/c")

    def test_map_to_space(self):
        self.assertEqual(apply_variant("soul/gram", "v3_operators_to_space"), "soul gram")
        self.assertEqual(apply_variant("a::b", "v3_operators_to_space"), "a b")

    def test_map_to_period(self):
        self.assertEqual(apply_variant("a::b//c", "v4_operators_to_period"), "a. b. c")
        # no space before the period, exactly one after
        self.assertEqual(apply_variant("here::the", "v4_operators_to_period"), "here. the")

    def test_word_internal_hyphens_are_never_touched(self):
        for key in ISSUE_VARIANTS:
            self.assertEqual(apply_variant("boy-roid and cold-blooded", key),
                             "boy-roid and cold-blooded", key)

    def test_free_dash_is_removed_but_glued_hyphen_is_not(self):
        self.assertEqual(apply_variant("mimic - neural", "v1_delete_all_operators"),
                         "mimic neural")
        # a hyphen glued to a word is morphological, not a site
        self.assertIn("-", apply_variant("pre-ADAM", "v1_delete_all_operators"))

    def test_contiguous_operator_run_is_one_junction(self):
        # `//::` between two words must be counted once, not twice
        line = "alpha//::beta"
        _new, _n, fused = sr.transform_line(line, sr.site_spans(line), "delete", None, None,
                                            collect_fused=True)
        self.assertEqual(len(fused), 1)
        self.assertEqual((fused[0][2], fused[0][3]), ("alpha", "beta"))

    def test_hyphen_bridge_joins_two_tokens(self):
        # regression for the rare second join mechanism: a trailing hyphen that
        # was NOT word-internal becomes word-internal once the operator goes
        line = "hydro-</p>"
        self.assertIn("-", line)
        out, _n, fused = sr.transform_line(line, sr.site_spans(line), "delete", None, None,
                                           collect_fused=True)
        self.assertEqual(out, "hydro-p")
        self.assertEqual(len(fused), 1)
        w = [t[3] for t in tokenize(out) if t[0] == "WORD"]
        w0 = [t[3] for t in tokenize(line) if t[0] == "WORD"]
        self.assertEqual(len(w0) - len(w), 1)

    def test_preexisting_periods_are_not_mangled(self):
        # normalisation only touches *inserted* periods
        self.assertEqual(apply_variant("www.creationbooks.com", "v4_operators_to_period"),
                         "www.creationbooks.com")
        self.assertEqual(apply_variant("ISBN 0-9675493-0-9", "v4_operators_to_period"),
                         "ISBN 0-9675493-0-9")

    def test_absent_operators_remain_absent(self):
        for key in STREAMS:
            self.assertNotIn("`", apply_variant("a::b//c", key))

    def test_transform_is_deterministic(self):
        line = "the vital-icon+our chromosome form escape of the suck=blood chromosome::the"
        for key in STREAMS:
            self.assertEqual(apply_variant(line, key), apply_variant(line, key), key)


@unittest.skipUnless(os.path.exists(sr.SRC), "layout corpus not present")
class CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = sr.load_pages(sr.SRC)
        cls.text = "".join(cls.pages.values())

    def test_operator_inventory_reproduces_10949(self):
        counts = {}
        for _s, _e, op, kind in sr.site_spans(self.text, dashes=False):
            counts[op] = counts.get(op, 0) + 1
        self.assertEqual(sum(counts.values()), 10949)
        self.assertEqual(counts["::"], 3083)
        self.assertEqual(counts["//"], 2149)
        self.assertEqual(counts["="], 2047)
        self.assertEqual(counts["/"], 1484)
        self.assertEqual(len(counts), 11)   # #10/#12: 11 operator types occur

    def test_dash_sites_are_measured_not_assumed(self):
        by_char = {}
        for _s, _e, op, kind in sr.site_spans(self.text, dashes=True):
            if kind == "dash":
                by_char[op] = by_char.get(op, 0) + 1
        self.assertEqual(by_char["-"], 177)      # #12 measured 177 free-standing `-`
        self.assertEqual(by_char["\u2014"], 476)  # em dash
        self.assertEqual(by_char["\u2013"], 43)   # en dash
        self.assertEqual(sum(by_char.values()), 696)

    def test_dash_regex_only_matches_whitespace_bounded_dashes(self):
        self.assertEqual(len(sr.dash_spans("word- word")), 0)
        self.assertEqual(len(sr.dash_spans("word - word")), 1)
        self.assertEqual(len(sr.dash_spans("word \u2014 word")), 1)

    def test_site_spans_are_disjoint_and_sorted(self):
        for body in list(self.pages.values())[:20]:
            spans = sr.site_spans(body)
            self.assertEqual(spans, sorted(spans))
            for a, b in zip(spans, spans[1:]):
                self.assertLessEqual(a[1], b[0])

    def test_baseline_lexeme_count(self):
        n = sum(1 for t in tokenize(self.text) if t[0] == "WORD")
        self.assertEqual(n, 77392)

    def test_no_double_counting_of_operators_and_dashes(self):
        # operator_spans + dash_spans must not overlap
        ops = sr.operator_spans(self.text)
        dashes = sr.dash_spans(self.text)
        opset = {(s, e) for s, e, _o, _k in ops}
        self.assertFalse([d for d in dashes if (d[0], d[1]) in opset])


@unittest.skipUnless(os.path.exists(os.path.join(OUT, "summary.json")),
                     "run work/silence_removal.py first")
class ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(OUT, "summary.json"), encoding="utf-8") as f:
            cls.summary = json.load(f)
        with open(os.path.join(OUT, "variants.tsv"), encoding="utf-8") as f:
            cls.rows = {r.split("\t")[0]: r.rstrip("\n").split("\t")
                        for r in f.read().splitlines()[1:]}

    def test_all_seven_streams_exist(self):
        for k in STREAMS:
            path = os.path.join(OUT, k + ".txt")
            self.assertTrue(os.path.exists(path), path)
            self.assertGreater(os.path.getsize(path), 100_000, path)

    def test_the_four_issue_named_variants_exist(self):
        # the issue asks for: delete all operators / delete only ::,// /
        # map to space / map to period
        for k in ISSUE_VARIANTS:
            self.assertIn(k, self.rows, k)
            self.assertTrue(os.path.exists(os.path.join(OUT, k + ".txt")), k)

    def test_streams_are_page_tagged_in_order(self):
        for k in STREAMS:
            with open(os.path.join(OUT, k + ".txt"), encoding="utf-8") as f:
                pages = [int(m) for m in re.findall(r"<!-- page (\d+) -->", f.read())]
            self.assertEqual(pages, list(range(1, 226)), k)

    def test_headline_counts(self):
        s = self.summary
        self.assertEqual(s["operator_tokens"], 10949)
        self.assertEqual(s["dash_sites"], 696)
        self.assertEqual(s["silence_sites_total"], 11645)
        self.assertEqual(s["lexemes_baseline"], 77392)
        self.assertTrue(s["site_counts_reproduce_10949"])
        self.assertEqual(s["n_pages"], 225)

    def test_every_token_count_is_explained(self):
        # the builder asserts this; the artifact must record it too
        for k in STREAMS:
            r = self.summary["reconciliation"][k]
            self.assertEqual(r["word_tokens"], r["expected"], k)
            self.assertEqual(r["delta"], 0, k)

    def test_separator_variants_keep_the_baseline_token_sequence(self):
        for k in ("v0_original", "v3_operators_to_space", "v4_operators_to_period",
                  "v5_period_from_slashes", "v6_period_from_colons"):
            self.assertTrue(self.summary["reconciliation"][k]["same_token_sequence_as_baseline"], k)
            self.assertEqual(self.rows[k][4], "77392", k)   # word_tokens column
        for k in ("v1_delete_all_operators", "v2_delete_colon_slash"):
            self.assertFalse(self.summary["reconciliation"][k]["same_token_sequence_as_baseline"], k)

    def test_delete_variants_join_tokens(self):
        self.assertEqual(self.rows["v1_delete_all_operators"][4], "69346")
        self.assertEqual(self.rows["v2_delete_colon_slash"][4], "73340")
        self.assertEqual(self.summary["lexical_joins"]["v1_delete_all_operators"], 8046)
        self.assertEqual(self.summary["lexical_joins"]["v2_delete_colon_slash"], 4052)

    def test_v2_touches_only_colon_colon_and_slashes(self):
        # v2 must keep every other operator; assert on a real corpus line
        line = None
        for pno in sorted(sr.load_pages(sr.SRC)):
            for ln in sr.load_pages(sr.SRC)[pno].split("\n"):
                if "gene=TV" in ln and "::" in ln and "//" in ln:
                    line = ln
                    break
            if line:
                break
        self.assertIsNotNone(line)
        out = apply_variant(line, "v2_delete_colon_slash")
        self.assertNotIn("::", out)
        self.assertNotIn("//", out)
        self.assertIn("gene=TV", out)

    def test_period_mapping_adds_periods_only_where_operators_were(self):
        s = self.summary["variants"]
        self.assertEqual(s["v0_original"]["periods"], 22)
        self.assertGreater(s["v4_operators_to_period"]["periods"], 10000)
        self.assertEqual(s["v0_original"]["periods"], s["v3_operators_to_space"]["periods"])

    def test_sentence_shape_replicates_the_12_colon_vs_slash_finding(self):
        ss = self.summary["sentence_shape"]
        self.assertGreater(ss["v5_period_from_slashes"]["frac_upper_initial"],
                           ss["v6_period_from_colons"]["frac_upper_initial"])
        # and the length-controlled version must agree in sign
        self.assertGreater(ss["v5_period_from_slashes"]["frac_upper_initial_ge5"],
                           ss["v6_period_from_colons"]["frac_upper_initial_ge5"])

    def test_fused_residue_is_a_decomposition_dictionary(self):
        path = os.path.join(OUT, "fused_residue.tsv")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0].split("\t")[0], "left_operand")
        rows = [r.split("\t") for r in lines[1:]]
        pairs = {(r[0], r[1]): int(r[3]) for r in rows}
        # these three must agree with #12's linker evidence exactly
        self.assertEqual(pairs[("soul", "gram")], 310)
        self.assertEqual(pairs[("gene", "TV")], 147)
        self.assertEqual(pairs[("suck", "blood")], 130)
        self.assertEqual(pairs[("vital", "serum")], 129)
        for r in rows:
            self.assertEqual(r[0] + r[1], r[2])   # joined_form is the concatenation
            self.assertTrue(r[5])                 # every row is citable

    def test_per_page_metrics_cover_every_page_and_variant(self):
        path = os.path.join(OUT, "per_page_metrics.tsv")
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 1 + 225 * len(STREAMS))
        self.assertEqual(lines[0].split("\t")[:3], ["page", "folio", "variant"])

    def test_line_index_is_line_exact(self):
        path = os.path.join(OUT, "line_index.tsv")
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()[1:]
        by_key = {}
        for line in lines:
            page, variant, out_i, src_i, _folio = line.split("\t")
            by_key.setdefault((int(page), variant), []).append((int(out_i), int(src_i)))
        self.assertTrue(by_key)
        for (page, variant), pairs in by_key.items():
            self.assertEqual([o for o, _ in pairs], list(range(1, len(pairs) + 1)),
                             (page, variant))
            self.assertEqual([s for _, s in pairs], sorted(s for _, s in pairs),
                             (page, variant))

    def test_silence_sites_tsv_reports_the_dash_policy(self):
        with open(os.path.join(OUT, "silence_sites.tsv"), encoding="utf-8") as f:
            lines = f.read().splitlines()
        head = lines[0].split("\t")
        self.assertIn("v2_delete_colon_slash", head)
        body = {r.split("\t")[0]: r.split("\t") for r in lines[1:]}
        self.assertEqual(body["::"][2], "3083")
        self.assertEqual(body["//"][2], "2149")
        self.assertEqual(body["="][head.index("v2_delete_colon_slash")], "kept")
        self.assertEqual(body["::"][head.index("v2_delete_colon_slash")], "deleted")
        self.assertTrue(any(k.startswith("dash:") for k in body))

    def test_report_states_rationale_and_caveats(self):
        with open(os.path.join(OUT, "silence_removal_report.md"), encoding="utf-8") as f:
            text = f.read()
        for k in ISSUE_VARIANTS:
            self.assertIn(k, text)
        self.assertIn("Begu", text)
        self.assertIn("proxy", text)              # not the #50 perplexity verdict
        self.assertIn("Interpretation, not measurement", text)
        self.assertIn("dash policy", text)
        self.assertIn("#50", text)
        self.assertIn("#67", text)
        self.assertIn("#62", text)
        self.assertIn("negative result", text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)