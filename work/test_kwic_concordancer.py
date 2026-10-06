#!/usr/bin/env python3
"""Regression suite for #26 KWIC concordancer.

Runs the concordancer once into a temp directory, then checks the invariants
that matter: the term list (entities + suspicious terms from #19/#22/#6/#21),
the case-insensitive unit join, the +/-5-token context, the uid-based (not
token-index) join, and reconciliation with #22's all-caps token counts.

    python work/test_kwic_concordancer.py
"""
import bisect
import collections
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kwic_concordancer as kc

_ARTIFACTS = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="kwic_test_")
    kc.main(["--out", tmp, "--quiet"])
    for name in ("concordance", "term_summaries", "diagnostics",
                 "reconciliation"):
        _ARTIFACTS[name] = kc.read_tsv(Path(tmp) / f"{name}.tsv")
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    _ARTIFACTS["_dir"] = tmp


def rows(name):
    return _ARTIFACTS[name]


def summaries_by_lower():
    return {r["term_lower"]: r for r in rows("term_summaries")}


def concordance_by_lower():
    m = collections.defaultdict(list)
    for r in rows("concordance"):
        m[r["term_lower"]].append(r)
    return m


# ---------------------------------------------------------------------------
class TestTermList(unittest.TestCase):
    def test_neologism_candidates_present(self):
        # every #19 candidate surface is a term (case-insensitively)
        neo = [r for r in kc.read_tsv(kc.NEOLOGISMS) if r["is_candidate"] == "y"]
        have = set(summaries_by_lower())
        missing = [r["unit"] for r in neo
                   if kc.normalize_surface(r["unit"]).lower() not in have]
        self.assertEqual(missing, [])

    def test_caps_labels_and_command_present(self):
        caps = kc.read_tsv(kc.CAPS)
        have = set(summaries_by_lower())
        for r in caps:
            if r["class"] == "label" or (r["class"] == "scream"
                                        and r["subtype"] == "command"):
                self.assertIn(r["type"].lower(), have)

    def test_caps_screams_headers_function_excluded(self):
        caps = kc.read_tsv(kc.CAPS)
        have = set(summaries_by_lower())
        # types legitimately in the term list via a NON-scream/header source
        other = set()
        for r in kc.read_tsv(kc.NEOLOGISMS):
            if r["is_candidate"] == "y":
                other.add(r["unit"].lower())
        for r in kc.read_tsv(kc.GAZETTEER):
            if r["category"] == "entity":
                other.add(r["token"].lower())
        for r in kc.read_tsv(kc.NUMERIC):
            other.add(r["unit"].lower())
        for r in caps:
            if r["class"] == "label" or (r["class"] == "scream"
                                        and r["subtype"] == "command"):
                other.add(r["type"].lower())
        for r in caps:
            if r["class"] == "function":
                self.assertNotIn(r["type"].lower(), have)
            if r["class"] == "scream" and r["subtype"] == "emphasis":
                # excluded unless present via another source (e.g. `zero`
                # is also a #21 numeric term)
                if r["type"].lower() not in other:
                    self.assertNotIn(r["type"].lower(), have)
            if r["class"] == "header":
                if r["type"].lower() not in other:
                    self.assertNotIn(r["type"].lower(), have)

    def test_gazetteer_entities_present(self):
        have = set(summaries_by_lower())
        for r in kc.read_tsv(kc.GAZETTEER):
            if r["category"] == "entity":
                self.assertIn(r["token"].lower(), have)

    def test_numeric_channel_present(self):
        have = set(summaries_by_lower())
        for r in kc.read_tsv(kc.NUMERIC):
            self.assertIn(r["unit"].lower(), have)

    def test_case_insensitive_dedup(self):
        # ADAM / adam / Adam are one term; the summary lists all variants
        s = summaries_by_lower().get("adam")
        self.assertIsNotNone(s)
        self.assertIn("ADAM", s["surface_variants"])

    def test_no_markup_tag_terms(self):
        # tag names must never enter as words (#19 removed them)
        have = set(summaries_by_lower())
        for tag in ("th", "pre", "br", "li", "hr", "p"):
            self.assertNotIn(tag, have)


# ---------------------------------------------------------------------------
class TestConcordance(unittest.TestCase):
    def test_every_exact_match_unit_appears_once(self):
        units = {int(r["uid"]): r for r in kc.read_tsv(kc.UNITS)}
        # rebuild the exact-match map the way the builder does
        unit_lower = collections.defaultdict(list)
        for uid, r in units.items():
            unit_lower[r["unit"].lower()].append(uid)
        conc = concordance_by_lower()
        for key, uids in unit_lower.items():
            if key in summaries_by_lower():
                self.assertEqual(len(conc[key]), len(uids), key)

    def test_occ_n_sequential_per_term(self):
        for key, rs in concordance_by_lower().items():
            self.assertEqual([int(r["occ_n"]) for r in rs],
                             list(range(1, len(rs) + 1)), key)

    def test_key_matches_term_case_insensitively(self):
        for r in rows("concordance"):
            self.assertEqual(r["unit_surface"].lower(), r["term_lower"])

    def test_uid_is_valid_unit(self):
        units = {int(r["uid"]) for r in kc.read_tsv(kc.UNITS)}
        for r in rows("concordance"):
            self.assertIn(int(r["unit_uid"]), units)

    def test_context_nonempty_for_interior_terms(self):
        # a term with occurrences deep in the book has non-empty context
        for r in rows("concordance"):
            if r["term_lower"] == "adam" and r["page"] == "15":
                self.assertTrue(r["left_context"])
                self.assertTrue(r["right_context"])


# ---------------------------------------------------------------------------
class TestContextWindow(unittest.TestCase):
    """The +/-5-token context is the raw slice over the 5 tokens around the
    unit, located by containment on (page, start) -- never by token index."""

    def _unit_tokens(self, page, s, e):
        toks = []
        for r in kc.read_tsv(kc.TOKENS):
            if r["page"] == str(page):
                toks.append((int(r["start"]), int(r["end"]), r["text"],
                             r["kind"]))
        toks.sort()
        starts = [t[0] for t in toks]
        a = bisect.bisect_left(starts, s)
        b = bisect.bisect_left(starts, e) - 1
        return toks, a, b

    def test_adam_p15_context_exact(self):
        bodies = kc.load_bodies()
        # ADAM uid 3823 on page 15, span 912-916
        toks, a, b = self._unit_tokens(15, 912, 916)
        self.assertEqual([t[2] for t in toks[a:b + 1]], ["ADAM"])
        left = toks[max(0, a - 5):a]
        right = toks[b + 1:b + 6]
        lctx = kc.normalize_ws(bodies[15][left[0][0]:left[-1][1]])
        rctx = kc.normalize_ws(bodies[15][right[0][0]:right[-1][1]])
        self.assertEqual(lctx, "the bondage= mode of")
        self.assertEqual(rctx, "to the masses of flesh")
        # and the concordance row matches
        conc = [r for r in rows("concordance")
                if r["term_lower"] == "adam" and r["page"] == "15"
                and r["unit_uid"] == "3823"]
        self.assertEqual(len(conc), 1)
        self.assertEqual(conc[0]["left_context"], lctx)
        self.assertEqual(conc[0]["right_context"], rctx)

    def test_context_is_five_tokens_each_side(self):
        bodies = kc.load_bodies()
        units = {int(r["uid"]): r for r in kc.read_tsv(kc.UNITS)}
        # pick an interior ADAM occurrence and count context tokens
        for r in rows("concordance"):
            if r["term_lower"] != "adam":
                continue
            uid = int(r["unit_uid"])
            page, s, e = int(units[uid]["page"]), int(units[uid]["start"]), \
                int(units[uid]["end"])
            toks, a, b = self._unit_tokens(page, s, e)
            left = toks[max(0, a - 5):a]
            right = toks[b + 1:b + 6]
            self.assertEqual(len(left), 5)
            self.assertEqual(len(right), 5)
            return
        self.fail("no ADAM occurrence found")


# ---------------------------------------------------------------------------
class TestJoin(unittest.TestCase):
    def test_containment_not_token_index(self):
        # the tokens.tsv `unit` column is a per-page index, not the global
        # uid; the concordancer must locate a unit's tokens by containment.
        # Verify: for a sample unit, the tokens found by containment are
        # exactly the unit's own tokens.
        units = {int(r["uid"]): r for r in kc.read_tsv(kc.UNITS)}
        for uid in (106, 108, 3823):
            r = units[uid]
            toks = []
            for t in kc.read_tsv(kc.TOKENS):
                if t["page"] == r["page"]:
                    toks.append((int(t["start"]), int(t["end"]), t["text"]))
            toks.sort()
            inside = [t[2] for t in toks
                      if int(r["start"]) <= t[0] < int(r["end"])]
            self.assertEqual("".join(inside).replace(" ", ""),
                             r["unit"].replace(" ", ""), uid)


# ---------------------------------------------------------------------------
class TestReconciliation(unittest.TestCase):
    def test_adam_reconciles_with_22(self):
        s = summaries_by_lower()["adam"]
        # 303 standalone + 25 token-embedded = 328 (#22 all-caps ADAM tokens)
        self.assertEqual(int(s["n_occurrences"]) + int(s["embedded_tok"]), 328)

    def test_dna_reconciles_with_22(self):
        s = summaries_by_lower()["dna"]
        self.assertEqual(int(s["n_occurrences"]) + int(s["embedded_tok"]), 138)

    def test_tv_reconciles_with_22(self):
        s = summaries_by_lower()["tv"]
        self.assertEqual(int(s["n_occurrences"]) + int(s["embedded_tok"]), 173)

    def test_total_occurrences(self):
        total = sum(int(r["n_occurrences"]) for r in rows("term_summaries"))
        self.assertEqual(total, len(rows("concordance")))
        self.assertEqual(total, _ARTIFACTS["summary"]["occurrences"])

    def test_summary_json(self):
        s = _ARTIFACTS["summary"]
        self.assertEqual(s["context_tokens"], 5)
        self.assertEqual(s["terms"], len(rows("term_summaries")))
        self.assertGreater(s["terms_with_occurrences"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
