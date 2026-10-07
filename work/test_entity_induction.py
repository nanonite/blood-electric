#!/usr/bin/env python3
"""Regression suite for #30 entity / proper-noun induction (dramatis personae).

Builds once into a temp directory, then checks the invariants that matter:
the markup screen, the reconciliation against #19/#22/#27, the induction
rules (seeds accepted, English false positives rejected), role assignment,
and determinism of the emitted summary.

    python work/test_entity_induction.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import entity_induction as EI

cl = EI.cl
_A = {}


def setUpModule():
    tmp = tempfile.mkdtemp(prefix="entity_test_")
    summary = EI.main(["--out", tmp, "--quiet"])
    _A["summary"] = summary
    _A["glossary"] = cl.read_tsv(Path(tmp) / "entity_glossary.tsv")
    _A["candidates"] = cl.read_tsv(Path(tmp) / "entity_candidates.tsv")
    _A["recon"] = cl.read_tsv(Path(tmp) / "reconciliation.tsv")
    _A["diag"] = {r["diag"]: r["value"]
                  for r in cl.read_tsv(Path(tmp) / "diagnostics.tsv")}
    _A["report"] = (Path(tmp) / "entity_report.md").read_text(encoding="utf-8")
    _A["_dir"] = tmp


def summary():
    return _A["summary"]


def by_entity():
    return {r["entity"]: r for r in _A["glossary"]}


# ---------------------------------------------------------------------------
class TestScreening(unittest.TestCase):
    def test_screened_items_match_27(self):
        self.assertEqual(int(summary()["unit_items_screened"]), 74077)

    def test_markup_removed(self):
        self.assertEqual(int(_A["diag"]["markup_units_removed"]), 588)

    def test_no_markup_tag_names_admitted(self):
        # HTML tag residue must never become an entity.
        banned = {"a", "th", "p", "pre", "li", "br", "hr", "td", "tr",
                  "href", "mailto", "colspan"}
        self.assertEqual(banned & set(by_entity()), set())


# ---------------------------------------------------------------------------
class TestReconciliation(unittest.TestCase):
    def test_all_recon_checks_ok(self):
        bad = [r["check"] for r in _A["recon"] if r["status"] != "ok"]
        self.assertEqual(bad, [])

    def test_recon_ties_to_upstream_tasks(self):
        checks = {r["check"] for r in _A["recon"]}
        for needle in ("#27", "#22", "#19"):
            self.assertTrue(any(needle in c for c in checks),
                            f"no reconciliation against {needle}")

    def test_seed_types_is_85(self):
        self.assertEqual(int(summary()["seed_types"]), 85)


# ---------------------------------------------------------------------------
class TestInduction(unittest.TestCase):
    def test_counts_consistent(self):
        s = summary()
        self.assertEqual(int(s["accepted_entities"]), len(_A["glossary"]))
        self.assertEqual(sum(int(v) for v in s["roles"].values()),
                         int(s["accepted_entities"]))

    def test_seed_entities_present(self):
        ents = set(by_entity())
        for name in ("ADAM", "ANTI-ADAM", "retro-ADAM", "ToKAGE",
                     "Sato Corporation", "Super Cherry 666"):
            self.assertIn(name, ents, f"seed entity {name} missing")

    def test_function_class_words_not_admitted(self):
        # #22 function class (I, A) must never be an entity.
        self.assertNotIn("I", by_entity())
        self.assertNotIn("A", by_entity())

    def test_english_common_nouns_not_admitted(self):
        # The zipf >= 2.5 false-positive control.
        ents = set(by_entity())
        for common in ("Doll", "Corporation", "God", "Heaven", "Project"):
            self.assertNotIn(common, ents)

    def test_every_row_has_evidence_and_span(self):
        for r in _A["glossary"]:
            self.assertTrue(r["role_evidence"], r["entity"])
            n = int(r["n_occurrences"])
            self.assertGreaterEqual(n, 1)
            self.assertLessEqual(int(r["first_page"]), int(r["last_page"]))

    def test_rejected_candidates_are_recorded(self):
        # Every unit-level decision is disclosed, accepted or rejected.
        rejected = [r for r in _A["candidates"] if r["accepted"] == "n"]
        accepted = [r for r in _A["candidates"] if r["accepted"] == "y"]
        self.assertEqual(len(rejected), int(summary()["rejected_candidates"]))
        self.assertEqual(len(rejected) + len(accepted), len(_A["candidates"]))

    def test_report_is_substantive(self):
        self.assertIn("dramatis personae", _A["report"].lower())
        self.assertIn("entity/1", _A["report"])


# ---------------------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_is_capitalized(self):
        self.assertTrue(EI.is_capitalized("ADAM"))
        self.assertTrue(EI.is_capitalized("Sato"))
        self.assertFalse(EI.is_capitalized("adam"))
        self.assertFalse(EI.is_capitalized(""))

    def test_is_numeric(self):
        self.assertTrue(EI.is_numeric("666"))
        self.assertTrue(EI.is_numeric("13"))
        self.assertFalse(EI.is_numeric("4-D"))   # has a letter
        self.assertFalse(EI.is_numeric("1mg"))
        self.assertFalse(EI.is_numeric("ADAM"))
        self.assertFalse(EI.is_numeric(""))


# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_summary_is_deterministic_modulo_timestamp(self):
        a = dict(summary())
        a.pop("generated", None)
        tmp = tempfile.mkdtemp(prefix="entity_test2_")
        b = dict(EI.build(Path(tmp)))
        b.pop("generated", None)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=1)
