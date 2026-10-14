#!/usr/bin/env python3
"""Regression suite for #59 entity roles from the jacket synopsis.

Run the whole work/ suite with:
    python -m unittest discover -s work -p 'test_*.py'

Two layers, like the workspace's other suites:

  * synthetic tests pin the probe dispatcher (every kind's pass/fail
    boundary), the verdict rule (evidenced / uncertain / refuted, with
    presence as a precondition), the seed validator and the evidence
    container -- on hand-built corpora, with no artifacts needed;
  * corpus tests (guarded by skipUnless) build the role table into a
    temporary directory, check determinism and reconciliation, and pin
    the published verdicts against the frozen #17/#27/#30/#33/#66 layers
    -- including the known divergences (ANTi-ADAM never clashes, the dog
    never wages war, Super Cherry 666 is refuted).

build() is exercised into temporary directories only, never into
analysis/entity_roles/.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))
import entity_roles as er

HAVE_ARTIFACTS = all(p.exists() for p in (
    er.nr.UNITS, er.nr.CLAUSES, er.nr.GLOSSARY, er.nr.VERBS,
    er.nr.INTERLINEAR))


# ---------------------------------------------------------------------------
# synthetic helpers
# ---------------------------------------------------------------------------
def ev(mentions=None, pages=None, roles=None, events=None, keys=None,
       clauses=None):
    page_counts = {k: set(range(1, v + 1)) for k, v in (pages or {}).items()}
    return er.Evidence(
        dict(mentions or {}), page_counts, dict(roles or {}),
        list(events or []), dict(keys or {}), list(clauses or []))


def event(actor, verb, target, page=1, cite=None):
    return {"actor": actor, "verb": verb, "target": target, "page": page,
            "cite": cite or f"p{page}:1"}


def clause(cid, page=1, start=1, keys=()):
    return {"clause_id": cid, "page": str(page), "start": str(start),
            "keys": frozenset(keys)}


def cooccur_ev(a, b, page=1, start=5):
    c = clause("c1", page, start)
    return ev(keys={c["clause_id"]: frozenset({a, b})}, clauses=[c])


def probe(kind, params, defining=True):
    return (kind, "label", params, defining)


# ---------------------------------------------------------------------------
# evidence container
# ---------------------------------------------------------------------------
class TestEvidence(unittest.TestCase):
    def test_mentions_is_the_max_over_the_family(self):
        e = ev(mentions={"A": 3, "B": 7})
        self.assertEqual(e.mentions(("A", "B")), 7)
        self.assertEqual(e.mentions(("A",)), 3)
        self.assertEqual(e.mentions(("Z",)), 0)

    def test_pages_is_the_max_over_the_family(self):
        e = ev(pages={"A": 3, "B": 7})
        self.assertEqual(e.pages(("A", "B")), 7)
        self.assertEqual(e.pages(("Z",)), 0)

    def test_role_set_is_sorted_and_deduped(self):
        e = ev(roles={"A": "persona", "B": "persona"})
        self.assertEqual(e.role_set(("A", "B")), ("persona",))
        self.assertEqual(e.role_set(("Z",)), ())

    def test_events_for_family_counts_either_side(self):
        e = ev(events=[event("A", "murder", "B"), event("C", "burn", "A"),
                       event("A", "burn", "A"), event("B", "murder", "B")])
        self.assertEqual(len(e.events_for(("A",))), 3)

    def test_cooccur_reads_clause_keys(self):
        c1, c2 = clause("c1"), clause("c2")
        e = ev(keys={"c1": frozenset({"A", "B"}), "c2": frozenset({"A"})},
               clauses=[c1, c2])
        self.assertEqual(len(e.cooccur("A", "B")), 1)
        self.assertEqual(len(e.cooccur("A", "A")), 2)
        self.assertEqual(len(e.cooccur("B", "Z")), 0)

    def test_verb_events_filters_by_verb(self):
        e = ev(events=[event("A", "hunt", "-"), event("A", "is", "-")])
        self.assertEqual(len(e.verb_events(("A",), ("hunt",))), 1)
        self.assertEqual(len(e.verb_events(("A",), ("wage",))), 0)

    def test_pair_events_is_orientation_agnostic(self):
        e = ev(events=[event("ADAM", "clashed", "ANTI-ADAM"),
                       event("ANTI-ADAM", "murder", "ADAM")])
        self.assertEqual(
            len(e.pair_events(("ADAM",), ("ANTI-ADAM",))), 2)
        self.assertEqual(
            len(e.pair_events(("ADAM",), ("ANTI-ADAM",), ("clashed",))), 1)

    def test_hostile_events_subset(self):
        e = ev(events=[event("A", "murder", "-"), event("A", "is", "-")])
        self.assertEqual(len(e.hostile_events(("A",))), 1)


# ---------------------------------------------------------------------------
# probe dispatcher
# ---------------------------------------------------------------------------
class TestProbes(unittest.TestCase):
    def run_one(self, spec, e):
        kind, _label, res = er.run_probe(spec, e)
        self.assertEqual(kind, spec[0])
        return res

    def test_present_passes_and_fails(self):
        e = ev(mentions={"A": 2})
        self.assertEqual(self.run_one(probe("present", {"entities": ("A",)}),
                                      e)["status"], "pass")
        self.assertEqual(self.run_one(probe("present", {"entities": ("Z",)}),
                                      e)["status"], "fail")

    def test_recurrence_threshold(self):
        e = ev(pages={"A": 5})
        self.assertEqual(self.run_one(
            probe("recurrence", {"entities": ("A",), "min": 5}), e)["status"],
            "pass")
        self.assertEqual(self.run_one(
            probe("recurrence", {"entities": ("A",), "min": 6}), e)["status"],
            "fail")

    def test_centrality_counts_events(self):
        e = ev(events=[event("A", "burn", "-")] * 3)
        self.assertEqual(self.run_one(
            probe("centrality", {"entities": ("A",), "min": 3}), e)["status"],
            "pass")
        self.assertEqual(self.run_one(
            probe("centrality", {"entities": ("A",), "min": 4}), e)["status"],
            "fail")

    def test_category_allows_any_member_role(self):
        e = ev(roles={"A": "persona", "B": "creature"})
        self.assertEqual(self.run_one(probe(
            "category", {"entities": ("A", "B"), "allowed": ("creature",)}),
            e)["status"], "pass")
        self.assertEqual(self.run_one(probe(
            "category", {"entities": ("A", "B"), "allowed": ("place",)}),
            e)["status"], "fail")

    def test_cooccur_carries_citations(self):
        res = self.run_one(probe("cooccur", {"a": "A", "b": "B"}),
                           cooccur_ev("A", "B", page=7, start=90))
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["citations"], "p7:90")
        self.assertEqual(res["observed"], 1)

    def test_any_verb_requires_a_matching_event(self):
        e = ev(events=[event("A", "hunting", "-", cite="p9:3")])
        res = self.run_one(
            probe("any_verb", {"entities": ("A",), "verbs": ("hunt",
                                                             "hunting")}), e)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["citations"], "p9:3")
        self.assertEqual(self.run_one(probe(
            "any_verb", {"entities": ("A",), "verbs": ("wage",)}),
            e)["status"], "fail")

    def test_pair_verb_requires_both_sides_and_verb(self):
        e = ev(events=[event("ADAM", "clashed", "ANTI-ADAM")])
        self.assertEqual(self.run_one(probe(
            "pair_verb", {"a": ("ADAM",), "b": ("ANTI-ADAM",),
                          "verbs": ("clashed",)}), e)["status"], "pass")
        self.assertEqual(self.run_one(probe(
            "pair_verb", {"a": ("ADAM",), "b": ("ANTI-ADAM",),
                          "verbs": ("burn",)}), e)["status"], "fail")

    def test_hostile_probe(self):
        e = ev(events=[event("A", "murder", "-")])
        self.assertEqual(self.run_one(
            probe("hostile", {"entities": ("A",)}), e)["status"], "pass")
        self.assertEqual(self.run_one(
            probe("hostile", {"entities": ("Z",)}), e)["status"], "fail")

    def test_unknown_kind_fails_loudly(self):
        with self.assertRaises(er.RoleError):
            er.run_probe(("bogus", "x", {}, True), ev())


# ---------------------------------------------------------------------------
# verdict rule
# ---------------------------------------------------------------------------
class TestVerdict(unittest.TestCase):
    def rows(self, statuses):
        out = []
        for i, (kind, defining, status) in enumerate(statuses):
            out.append({"probe": f"p{i}", "kind": kind,
                        "defining": "y" if defining else "n", "status": status,
                        "observed": 1, "threshold": ">=1",
                        "citations": "p1:1"})
        return out

    def test_all_defining_pass_is_evidenced(self):
        rows = self.rows([("present", True, "pass"),
                          ("category", True, "pass"),
                          ("hostile", False, "fail")])
        self.assertEqual(er.role_verdict(rows), "evidenced")

    def test_some_defining_fail_is_uncertain(self):
        rows = self.rows([("present", True, "pass"),
                          ("category", True, "pass"),
                          ("pair_verb", True, "fail")])
        self.assertEqual(er.role_verdict(rows), "uncertain")

    def test_no_substantive_pass_is_refuted_even_if_present(self):
        rows = self.rows([("present", True, "pass"),
                          ("category", True, "fail"),
                          ("any_verb", True, "fail")])
        self.assertEqual(er.role_verdict(rows), "refuted")

    def test_presence_only_seed_falls_back_to_all_defining(self):
        rows = self.rows([("present", True, "pass")])
        self.assertEqual(er.role_verdict(rows), "evidenced")

    def test_no_defining_probe_fails_loudly(self):
        with self.assertRaises(er.RoleError):
            er.role_verdict(self.rows([("present", False, "pass")]))


# ---------------------------------------------------------------------------
# seed validation
# ---------------------------------------------------------------------------
class TestSeeds(unittest.TestCase):
    def test_shipped_seeds_validate(self):
        self.assertTrue(er.validate_seeds())

    def test_duplicate_role_id_rejected(self):
        seed = {"role_id": "X", "entities": ("A",), "claims": ("S1",),
                "probes": (probe("present", {"entities": ("A",)}),)}
        with self.assertRaises(er.RoleError):
            er.validate_seeds((seed, dict(seed)))

    def test_unknown_claim_rejected(self):
        seed = {"role_id": "X", "entities": ("A",), "claims": ("S9",),
                "probes": (probe("present", {"entities": ("A",)}),)}
        with self.assertRaises(er.RoleError):
            er.validate_seeds((seed,))

    def test_empty_entities_rejected(self):
        seed = {"role_id": "X", "entities": (), "claims": ("S1",),
                "probes": (probe("present", {"entities": ("A",)}),)}
        with self.assertRaises(er.RoleError):
            er.validate_seeds((seed,))

    def test_no_defining_probe_rejected(self):
        seed = {"role_id": "X", "entities": ("A",), "claims": ("S1",),
                "probes": (probe("present", {"entities": ("A",)},
                                 defining=False),)}
        with self.assertRaises(er.RoleError):
            er.validate_seeds((seed,))

    def test_unknown_probe_kind_rejected(self):
        seed = {"role_id": "X", "entities": ("A",), "claims": ("S1",),
                "probes": (("bogus", "x", {}, True),)}
        with self.assertRaises(er.RoleError):
            er.validate_seeds((seed,))

    def test_every_synopsis_sentence_is_used_by_some_seed(self):
        used = set()
        for seed in er.ROLE_SEEDS:
            used.update(seed["claims"])
        self.assertEqual(used, set(er.CLAIM_IDS))

    def test_entity_families_are_glossary_or_referent_names(self):
        # every seeded name is a #30 glossary entity or a disclosed #66
        # referent ('the dog'); nothing is invented.
        _, entities = er.nr.load_glossary() if HAVE_ARTIFACTS else ({}, {})
        allowed_extra = {"the dog"}
        for seed in er.ROLE_SEEDS:
            for name in seed["entities"]:
                if HAVE_ARTIFACTS:
                    self.assertIn(name, set(entities) | allowed_extra, name)


# ---------------------------------------------------------------------------
# evaluate_role
# ---------------------------------------------------------------------------
class TestEvaluateRole(unittest.TestCase):
    def seed(self):
        return {"role_id": "X", "entity": "A", "entities": ("A",),
                "synopsis_role": "r", "claims": ("S1",),
                "probes": (probe("present", {"entities": ("A",)}),
                           probe("cooccur", {"a": "A", "b": "B"}))}

    def test_row_fields_and_citations(self):
        e = cooccur_ev("A", "B", page=4, start=9)
        e.mention_counts["A"] = 2
        e.mention_pages["A"] = {4, 5}
        e.roles["A"] = "persona"
        row, probes = er.evaluate_role(self.seed(), e)
        self.assertEqual(row["status"], "evidenced")
        self.assertEqual(row["entity_role"], "persona")
        self.assertEqual(row["citations"], "p4:9")
        self.assertEqual(row["mentions"], 2)
        self.assertEqual(row["pages"], 2)
        self.assertEqual(len(probes), 2)
        self.assertEqual(probes[0]["defining"], "y")

    def test_uncertain_row_records_the_failed_defining_probe(self):
        seed = {"role_id": "X", "entity": "A", "entities": ("A",),
                "synopsis_role": "r", "claims": ("S1",),
                "probes": (probe("present", {"entities": ("A",)}),
                           probe("category", {"entities": ("A",),
                                              "allowed": ("persona",)}),
                           probe("cooccur", {"a": "A", "b": "B"}))}
        e = ev(mentions={"A": 1}, roles={"A": "persona"})
        row, _ = er.evaluate_role(seed, e)
        # presence and category pass, cooccur fails -> uncertain
        self.assertEqual(row["status"], "uncertain")
        self.assertIn("cooccur", row["refutation"])

    def test_run_probe_error_propagates(self):
        seed = {"role_id": "X", "entity": "A", "entities": ("A",),
                "synopsis_role": "r", "claims": ("S1",),
                "probes": (("bogus", "x", {}, True),)}
        with self.assertRaises(er.RoleError):
            er.evaluate_role(seed, ev())


# ---------------------------------------------------------------------------
# corpus tests (the real book)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts not present")
class CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="entity-roles-test-")
        cls.out = Path(cls._tmp.name) / "full"
        cls.summary = er.build(cls.out, quiet=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def rows(self, name):
        with (self.out / name).open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            return [dict(zip(header, line.rstrip("\n").split("\t")))
                    for line in fh if line.strip()]

    def test_every_artifact_is_written(self):
        for name in ("roles.tsv", "role_tests.tsv", "refutations.tsv",
                     "reconciliation.tsv", "diagnostics.tsv", "summary.json",
                     "roles_report.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_summary_contract(self):
        s = self.summary
        self.assertEqual(s["schema"], "entity_roles/1")
        self.assertEqual(s["roles"], 9)
        self.assertEqual(sum(s["role_status"].values()), 9)
        self.assertEqual(s["role_status"],
                         {"evidenced": 6, "uncertain": 2, "refuted": 1})
        self.assertEqual(len(s["verdicts"]), 9)
        self.assertEqual(s["evidence_rows"], len(self.rows("role_tests.tsv")))
        self.assertEqual(s["synopsis_page_excluded"], 224)

    def test_role_verdicts_are_pinned(self):
        want = {"R1": "evidenced", "R2": "uncertain", "R3": "evidenced",
                "R4": "evidenced", "R5": "evidenced", "R6": "uncertain",
                "R7": "refuted", "R8": "evidenced", "R9": "evidenced"}
        got = {r["role_id"]: r["status"] for r in self.rows("roles.tsv")}
        self.assertEqual(got, want)

    def test_every_evidenced_role_has_citations(self):
        for r in self.rows("roles.tsv"):
            if r["status"] == "evidenced":
                self.assertTrue(r["citations"], r["role_id"])
                self.assertEqual(r["n_defining_pass"], r["n_defining"])
            else:
                self.assertTrue(r["refutation"], r["role_id"])

    def test_refutations_table_lists_the_non_evidenced_roles(self):
        ref = self.rows("refutations.tsv")
        self.assertEqual({r["role_id"] for r in ref}, {"R2", "R6", "R7"})
        for r in ref:
            self.assertIn(r["status"], ("uncertain", "refuted"))

    def test_p224_is_never_cited(self):
        for r in self.rows("roles.tsv"):
            for cite in r["citations"].split("; "):
                if cite:
                    self.assertNotEqual(cite.split(":")[0], "p224")
        for p in self.rows("role_tests.tsv"):
            for cite in p["citations"].split("; "):
                if cite:
                    self.assertNotEqual(cite.split(":")[0], "p224")

    def test_reconciliation_is_all_ok(self):
        rows = self.rows("reconciliation.tsv")
        bad = [r["quantity"] for r in rows if r["match"] != "ok"]
        self.assertEqual(bad, [])
        quantities = " ".join(r["quantity"] for r in rows)
        for needle in ("#30", "#66", "p224", "screened units"):
            self.assertIn(needle, quantities)

    def test_two_builds_are_byte_identical(self):
        second = Path(self._tmp.name) / "second"
        er.build(second, quiet=True)
        for name in ("roles.tsv", "role_tests.tsv", "refutations.tsv",
                     "reconciliation.tsv", "diagnostics.tsv"):
            self.assertEqual((self.out / name).read_bytes(),
                             (second / name).read_bytes(), name)

    def test_build_is_hash_seed_independent(self):
        # frozenset iteration order must never reach a published measure:
        # build the CLI under two hash seeds and compare (a plain
        # same-process rebuild would reuse one seed and miss the bug).
        blobs = []
        for seed in ("0", "1"):
            dest = Path(self._tmp.name) / f"seed{seed}"
            env = dict(os.environ, PYTHONHASHSEED=seed)
            subprocess.run(
                [sys.executable, str(ROOT / "work" / "entity_roles.py"),
                 "--out", str(dest), "--quiet"],
                check=True, env=env, capture_output=True)
            blobs.append((dest / "role_tests.tsv").read_bytes())
        self.assertEqual(blobs[0], blobs[1])

    # -- the known divergences --------------------------------------------

    def test_anti_adam_clash_is_not_attested(self):
        tests = {r["probe"]: r for r in self.rows("role_tests.tsv")
                 if r["role_id"] == "R2"}
        clash = next(r for r in tests.values() if r["kind"] == "pair_verb")
        self.assertEqual(clash["status"], "fail")
        self.assertEqual(clash["observed"], "0")
        # but the pair does co-occur and ANTi-ADAM is the antagonist
        co = next(r for r in tests.values() if r["kind"] == "cooccur")
        self.assertEqual(co["status"], "pass")
        row = next(r for r in self.rows("roles.tsv") if r["role_id"] == "R2")
        self.assertEqual(row["entity_role"], "antagonist")

    def test_the_dog_never_wages_war(self):
        tests = [r for r in self.rows("role_tests.tsv")
                 if r["role_id"] == "R6" and r["kind"] == "any_verb"]
        self.assertTrue(tests)
        self.assertEqual(tests[0]["status"], "fail")
        # the zero co-occurrence that the phrase does support is a pass
        zero = next(r for r in self.rows("role_tests.tsv")
                    if r["role_id"] == "R6"
                    and "zero family" in r["probe"])
        self.assertEqual(zero["status"], "pass")

    def test_super_cherry_666_is_refuted(self):
        row = next(r for r in self.rows("roles.tsv") if r["role_id"] == "R7")
        self.assertEqual(row["status"], "refuted")
        self.assertEqual(row["entity_role"], "concept")
        self.assertIn("hunt", row["refutation"])
        self.assertIn("succubus", row["refutation"])
        # presence alone passed; nothing else did
        self.assertEqual(row["n_defining_pass"], "1")
        self.assertEqual(row["n_defining"], "4")

    def test_evol_pink_ash_planet_is_evidenced(self):
        tests = [r for r in self.rows("role_tests.tsv")
                 if r["role_id"] == "R8" and r["status"] == "pass"]
        labels = " ".join(r["probe"] for r in tests)
        self.assertIn("planet", labels)
        self.assertIn("pink", labels)
        self.assertIn("ash", labels)

    def test_probe_rows_are_well_formed(self):
        roles = {r["role_id"] for r in self.rows("roles.tsv")}
        seen = set()
        for p in self.rows("role_tests.tsv"):
            self.assertIn(p["role_id"], roles)
            self.assertIn(p["status"], ("pass", "fail"))
            self.assertIn(p["defining"], ("y", "n"))
            self.assertIn(p["kind"], er.PROBE_KINDS)
            key = (p["role_id"], p["probe"])
            self.assertNotIn(key, seen)
            seen.add(key)

    def test_claims_come_only_from_the_synopsis(self):
        for r in self.rows("roles.tsv"):
            for claim in r["claims"].split("|"):
                self.assertIn(claim, er.CLAIM_IDS)


if __name__ == "__main__":
    unittest.main()
