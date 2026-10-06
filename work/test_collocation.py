#!/usr/bin/env python3
"""Regression suite for #27 collocation / association.

Runs the builder once into a temp directory, then checks the invariants that
matter: the markup screen (units and tokens), the two sequences (unit vs
token), the association formulas, the operator-word pairs, the entity-verb
direction split, and the verb-candidate lexicon.

    python work/test_collocation.py
"""
import collections
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collocation as cl

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="colloc_test_")
    cl.main(["--out", tmp, "--quiet"])
    for name in ("collocations_bigram", "collocations_trigram",
                 "collocations_bigram_token", "collocations_trigram_token",
                 "entity_collocations", "entity_verb_pairs",
                 "entity_inventory", "verb_lexicon", "diagnostics",
                 "reconciliation"):
        _ARTIFACTS[name] = cl.read_tsv(Path(tmp) / f"{name}.tsv")
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    _ARTIFACTS["_dir"] = tmp
    _ARTIFACTS["report"] = (Path(tmp) / "collocation_report.md").read_text(
        encoding="utf-8")


def rows(name):
    return _ARTIFACTS[name]


def diag():
    return {r["diag"]: r["value"] for r in rows("diagnostics")}


def bigram_index():
    return {(r["x"], r["y"]): r for r in rows("collocations_bigram")}


def token_bigram_index():
    return {(r["x"], r["y"]): r for r in rows("collocations_bigram_token")}


def verb_index():
    return {r["verb"]: r for r in rows("verb_lexicon")}


# ---------------------------------------------------------------------------
class TestScreening(unittest.TestCase):
    def test_screened_items_match_19(self):
        d = diag()
        self.assertEqual(d["screened_items"], "74318")     # #19/#24
        self.assertEqual(d["markup_n19_residue"], "347")

    def test_unit_items_are_screened_minus_extra(self):
        d = diag()
        self.assertEqual(
            int(d["unit_items"]),
            int(d["screened_items"]) - int(d["markup_extra_removed"]))

    def test_markup_absent_from_unit_bigrams(self):
        bad = {"href", "mailto", "colspan", "th", "p", "pre", "br", "li",
               "hr", "td", "tr"}
        for r in rows("collocations_bigram"):
            self.assertNotIn(r["x"], bad)
            self.assertNotIn(r["y"], bad)

    def test_markup_absent_from_token_bigrams(self):
        bad = {"href", "mailto", "colspan", "th", "pre", "br", "li", "hr"}
        for r in rows("collocations_bigram_token"):
            self.assertNotIn(r["x"], bad)
            self.assertNotIn(r["y"], bad)

    def test_no_tag_name_as_hapax(self):
        # the #19/operator direction: `a`/`th`/`p`/`pre` must not enter as
        # words.  `a` is a real article, so only the tag-only names are checked.
        idx = bigram_index()
        for tag in ("th", "pre", "br", "li", "hr", "colspan", "href", "mailto"):
            self.assertNotIn((tag, tag), idx)
            self.assertFalse(any(tag in pair for pair in idx), tag)


# ---------------------------------------------------------------------------
class TestSequences(unittest.TestCase):
    def test_bigram_positions_reconcile(self):
        d = diag()
        page_units, _ = cl.load_units()
        bodies = cl.load_bodies()
        spans, markup_uids, markup_types = cl.load_markup(bodies)
        pos = 0
        for page, units in page_units.items():
            seq = [u for u in units
                   if not cl.unit_is_markup(u[3], u[1], u[0], page, spans,
                                            markup_uids, markup_types)]
            pos += max(0, len(seq) - 1)
        self.assertEqual(int(d["unit_bigram_positions"]), pos)

    def test_token_positions_reconcile(self):
        d = diag()
        page_tokens = cl.load_tokens()
        bodies = cl.load_bodies()
        spans, _, _ = cl.load_markup(bodies)
        pos = 0
        for page, toks in page_tokens.items():
            s = spans.get(page, [])
            seq = [t for t in toks if not any(a <= t[0] < b for a, b in s)]
            pos += max(0, len(seq) - 1)
        self.assertEqual(int(d["token_bigram_positions"]), pos)

    def test_unit_and_token_denominators_differ(self):
        d = diag()
        # the token layer keeps operators; the unit layer is the lexical view
        self.assertGreater(int(d["token_items"]), int(d["unit_items"]))


# ---------------------------------------------------------------------------
class TestAssociation(unittest.TestCase):
    def test_formula_matches_artifact(self):
        r = bigram_index()[("artificial", "sun")]
        f_xy, f_x, f_y, n = (int(r["f_xy"]), int(r["f_x"]),
                             int(r["f_y"]), int(r["n"]))
        pmi = math.log2((f_xy * n) / (f_x * f_y))
        logdice = cl.LOG_DICE_BASE + math.log2(2 * f_xy / (f_x + f_y))
        dice = 2 * f_xy / (f_x + f_y)
        tscore = (f_xy - f_x * f_y / n) / math.sqrt(f_xy)
        self.assertAlmostEqual(float(r["pmi"]), round(pmi, 4), places=3)
        self.assertAlmostEqual(float(r["logdice"]), round(logdice, 4), places=3)
        self.assertAlmostEqual(float(r["dice"]), round(dice, 5), places=4)
        self.assertAlmostEqual(float(r["tscore"]), round(tscore, 4), places=3)

    def test_known_collocations(self):
        idx = bigram_index()
        self.assertEqual(int(idx[("of", "the")]["f_xy"]), 3854)
        self.assertEqual(int(idx[("artificial", "sun")]["f_xy"]), 367)
        self.assertEqual(int(idx[("internal", "organ")]["f_xy"]), 250)
        self.assertEqual(int(idx[("cadaver", "city")]["f_xy"]), 192)

    def test_marginals_consistent(self):
        for r in rows("collocations_bigram"):
            self.assertLessEqual(int(r["f_xy"]), int(r["f_x"]))
            self.assertLessEqual(int(r["f_xy"]), int(r["f_y"]))

    def test_ranked_by_logdice(self):
        vals = [float(r["logdice"]) for r in rows("collocations_bigram")]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_trigrams(self):
        idx = {(r["x"], r["y"], r["z"]): r
               for r in rows("collocations_trigram")}
        self.assertEqual(int(idx[("of", "the", "artificial")]["f_xyz"]), 258)
        self.assertEqual(int(idx[("the", "artificial", "sun")]["f_xyz"]), 296)


# ---------------------------------------------------------------------------
class TestOperatorWord(unittest.TestCase):
    def test_known_operator_bigrams(self):
        idx = token_bigram_index()
        self.assertEqual(int(idx[("::", "the")]["f_xy"]), 928)
        self.assertEqual(idx[("::", "the")]["x_kind"], "OP")
        self.assertEqual(idx[("::", "the")]["y_kind"], "WORD")
        self.assertEqual(int(idx[("=", "tv")]["f_xy"]), 166)
        self.assertEqual(int(idx[("soul", "/")]["f_xy"]), 323)

    def test_has_op_flag(self):
        for r in rows("collocations_bigram_token"):
            expect = "y" if (r["x_kind"] == "OP" or r["y_kind"] == "OP") else "n"
            self.assertEqual(r["has_op"], expect)

    def test_operator_count_matches_17(self):
        # #17's operator stream: `::` = 3,083 body + 1 header = 3,084
        idx = token_bigram_index()
        self.assertEqual(int(idx[("::", "the")]["f_x"]), 3084)
        self.assertEqual(int(idx[("=", "tv")]["f_x"]), 1933)  # 2,047 minus markup


# ---------------------------------------------------------------------------
class TestEntities(unittest.TestCase):
    def test_entity_inventory(self):
        idx = {r["lower"]: r for r in rows("entity_inventory")}
        self.assertIn("tokage", idx)
        self.assertIn("adam", idx)
        self.assertIn("sato", idx)
        # entities have a source
        self.assertTrue(idx["tokage"]["sources"])

    def test_entity_collocations_present(self):
        by_ent = collections.defaultdict(list)
        for r in rows("entity_collocations"):
            by_ent[r["entity_lower"]].append(r)
        self.assertTrue(by_ent["tokage"])
        self.assertTrue(by_ent["adam"])
        # every row is a window partner with a direction split
        for r in rows("entity_collocations")[:200]:
            self.assertEqual(int(r["n_pair"]),
                             int(r["n_left"]) + int(r["n_right"]))

    def test_entity_verb_direction_tokage(self):
        idx = {(r["entity_lower"], r["partner"]): r
               for r in rows("entity_verb_pairs")}
        r = idx[("tokage", "fecundates")]
        self.assertEqual(r["verb_source"], "book")
        self.assertEqual(r["verb_confidence"], "high")
        self.assertGreater(int(r["n_right"]), 0)   # ToKAGE does
        self.assertGreater(int(r["n_left"]), 0)    # done to ToKAGE
        self.assertEqual(int(r["n_adjacent_right"]), 5)

    def test_entity_verb_partition(self):
        # every entity-verb row is also an entity-collocation row
        all_pairs = {(r["entity_lower"], r["partner"])
                     for r in rows("entity_collocations")}
        for r in rows("entity_verb_pairs"):
            self.assertIn((r["entity_lower"], r["partner"]), all_pairs)


# ---------------------------------------------------------------------------
class TestVerbLexicon(unittest.TestCase):
    def test_brown_verbs(self):
        v = verb_index()
        self.assertIn("is", v)
        self.assertEqual(v["is"]["source"], "brown")

    def test_book_verbs(self):
        v = verb_index()
        self.assertIn("fecundates", v)
        self.assertEqual(v["fecundates"]["source"], "book")
        self.assertIn("defleshed", v)
        self.assertEqual(v["defleshed"]["source"], "book")

    def test_noun_coinage_not_a_verb(self):
        # `apoptosis` ends in -s but is not swept in by the tightened rule
        self.assertNotIn("apoptosis", verb_index())

    def test_confidence(self):
        self.assertEqual(cl.verb_confidence("brown"), "high")
        self.assertEqual(cl.verb_confidence("book"), "high")
        self.assertEqual(cl.verb_confidence("morph_s"), "low")
        self.assertEqual(cl.verb_confidence(""), "")


# ---------------------------------------------------------------------------
class TestArtifacts(unittest.TestCase):
    def test_report_exists(self):
        self.assertIn("# Collocation and association (#27)", _ARTIFACTS["report"])
        self.assertIn("entity-verb", _ARTIFACTS["report"])

    def test_summary_counts(self):
        s = _ARTIFACTS["summary"]
        self.assertEqual(s["schema"], cl.SCHEMA)
        self.assertEqual(s["counts"]["unit_items"],
                         int(diag()["unit_items"]))

    def test_unique_pairs(self):
        pairs = [(r["x"], r["y"]) for r in rows("collocations_bigram")]
        self.assertEqual(len(pairs), len(set(pairs)))
        ep = [(r["entity_lower"], r["partner"])
              for r in rows("entity_collocations")]
        self.assertEqual(len(ep), len(set(ep)))

    def test_tsv_read_with_split(self):
        # quote-unsafe artifacts must survive a plain split reader
        for name in ("collocations_bigram", "entity_collocations"):
            path = Path(_ARTIFACTS["_dir"]) / f"{name}.tsv"
            with path.open(encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                for line in fh:
                    if not line.strip():
                        continue
                    self.assertEqual(len(line.rstrip("\n").split("\t")),
                                     len(header), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
