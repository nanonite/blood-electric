#!/usr/bin/env python3
"""Regression suite for the translation committee (#64).

Run the whole work/ suite with:
    python -m unittest discover -s work -p 'test_*.py'

Two layers, like the workspace's other suites:

  * **synthetic tests** on hand-built readings, which pin the classifier
    (all four classes and the order the rule tests them in), the
    per-surface operator compliance test, and the residue mirror.  The
    classifier is the load-bearing output of the task -- a disagreement
    that is misclassified localises the wrong passage -- so each class
    has a planted case that must land in it, including the two that are
    easy to get wrong:

      - a keep-strategy that kept `<<` as told and a drop-strategy that
        dropped it as told are `contract_divergence`, even when the
        keep-strategy failed `operator_fidelity` on a *different* surface
        (it added `/`): a row is classified by what the readings did on
        that surface, never by a reading's verdict elsewhere;
      - two readings that both breached the governing surface are
        `rule_violation` naming *both*, which is not the same statement
        as one of them being responsible.

  * **corpus tests** (guarded by `skipUnless`), which assert the
    committee's re-derivation agrees with #63's published record (the
    sample, every verdict, every check row, the operator columns and the
    response hashes), that the log's invariants hold over all 136 rows,
    and that the build is byte-identical on rebuild.

`build()` is exercised into temporary directories only, never into
`analysis/committee/`.
"""
import collections
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))
import glossing_pipeline as gp
import translation_committee as tc


HAVE_ARTIFACTS = all(p.exists() for p in (
    tc.PUBLISHED_GLOSSES, tc.PUBLISHED_CHECKS, tc.PUBLISHED_AGREE,
    tc.PUBLISHED_SAMPLE, gp.CLAUSES, gp.TOKENS, gp.ENTITY,
    gp.GLOSS_TOKENS))


def clause(cid="1", page=4, original="", gloss=""):
    """A #33/#62 clause row, the shape `check_gloss` reads."""
    return {"clause_id": cid, "page": str(page), "region": "body",
            "opener": "-", "start": "0", "end": "100",
            "original": original, "gloss": gloss}


def check_row(name, applies=True, passed=True):
    """A `check_gloss` row, as `reading_facts` consumes it."""
    return {"check": name, "applies": applies, "pass": passed,
            "verdict": "held" if passed else "violated", "detail": "",
            "contract": ""}


def facts(cid, sid, gloss, ops_in, entities=(), residue=(), verdict="pass",
          check_rows=(), page="4", stratum="plain_control"):
    """A `reading_facts` row, built the way `run_committee` builds it."""
    rows = list(check_rows) or [check_row(c) for c in tc.CHECKS]
    return tc.reading_facts(cid, page, stratum, sid, list(ops_in),
                            list(entities), list(residue), gloss, verdict,
                            rows)


def ents(*names):
    return [{"entity": n, "role": "persona", "n_occurrences": "3",
             "does": "", "done_to": ""} for n in names]


# ---------------------------------------------------------------------------
# the residue mirror
# ---------------------------------------------------------------------------

class TestResidueMirror(unittest.TestCase):
    """`residue_kept` must be the check's own predicate, per surface."""

    def test_a_surface_inside_the_gloss_is_kept(self):
        self.assertTrue(tc.residue_kept("omanko", "her omanko hydromachine"))

    def test_a_surface_inside_a_compound_is_kept(self):
        # the check's compound clause: the token the surface sits inside
        self.assertTrue(tc.residue_kept("laple", "the laple-machine runs"))

    def test_an_absent_surface_is_smoothed(self):
        self.assertFalse(tc.residue_kept("omanko", "her hydromachine"))

    def test_case_insensitive(self):
        self.assertTrue(tc.residue_kept("OMANKO", "her Omanko machine"))

    def test_the_mirror_agrees_with_the_check_it_mirrors(self):
        """For every residue/gloss pair, the check passes iff all kept.

        The check returns one verdict for the whole clause; the comparer
        needs each surface separately.  This pins the two together so a
        future edit to either shows up as a disagreement here rather than
        as a silently different residue semantics.
        """
        cases = [
            (["omanko"], "her omanko hydromachine"),
            (["omanko"], "her hydromachine"),
            (["omanko", "laple"], "omanko and laple together"),
            (["omanko", "laple"], "omanko alone, the other smoothed"),
            (["laple"], "the laple-machine runs"),
            ([], "no residue at all"),
        ]
        for residue, gloss in cases:
            want, _detail = gp.check_residue_discipline(residue, gloss)
            mirrored = all(tc.residue_kept(s, gloss) for s in residue)
            self.assertEqual(
                want, mirrored,
                "residue mirror disagrees with the check on %r / %r"
                % (residue, gloss))


# ---------------------------------------------------------------------------
# operator states and per-surface compliance
# ---------------------------------------------------------------------------

class TestOpState(unittest.TestCase):
    def test_absent(self):
        self.assertEqual(tc.op_state(0, 0, 0), "(absent)")

    def test_single_counts_read_as_words(self):
        self.assertEqual(tc.op_state(1, 0, 0), "kept 1")
        self.assertEqual(tc.op_state(0, 1, 0), "dropped 1")
        self.assertEqual(tc.op_state(0, 0, 1), "added 1")

    def test_multiplicities_are_counted(self):
        self.assertEqual(tc.op_state(2, 0, 0), "kept 2")
        self.assertEqual(tc.op_state(0, 0, 13), "added 13")

    def test_partial_keep_is_not_agreement(self):
        # a clause with two `=` and a reading that keeps one: set
        # membership would call this agreement, and the log would
        # understate the divergence.
        self.assertEqual(tc.op_state(1, 1, 0), "kept 1 dropped 1")


class TestOperatorCompliance(unittest.TestCase):
    """Per-surface compliance: what the reading's own contract owes."""

    def _reading(self, sid, ops_in, ops_out):
        f = facts("1", sid, "x " + " ".join(ops_out) + " y", ops_in)
        return f

    def test_a_keep_strategy_owes_the_channel_exactly(self):
        f = self._reading("literal_operator", ["=", ">"], ["=", ">"])
        self.assertTrue(tc.operator_compliance(f, "="))
        self.assertTrue(tc.operator_compliance(f, ">"))

    def test_a_keep_strategy_that_added_is_in_breach(self):
        f = self._reading("literal_operator", ["="], ["=", "|"])
        self.assertTrue(tc.operator_compliance(f, "="))
        self.assertFalse(tc.operator_compliance(f, "|"))

    def test_a_keep_strategy_that_dropped_is_in_breach(self):
        f = self._reading("literal_operator", ["=", ">"], ["="])
        self.assertFalse(tc.operator_compliance(f, ">"))

    def test_a_drop_strategy_owes_an_empty_channel(self):
        f = self._reading("scene_summary", ["="], [])
        self.assertTrue(tc.operator_compliance(f, "="))

    def test_a_drop_strategy_that_kept_or_added_is_in_breach(self):
        kept = self._reading("scene_summary", ["="], ["="])
        self.assertFalse(tc.operator_compliance(kept, "="))
        added = self._reading("plain_english", ["="], ["|"])
        self.assertFalse(tc.operator_compliance(added, "|"))

    def test_the_channel_test_sees_a_reorder_the_symbol_test_cannot(self):
        f = self._reading("literal_operator", ["=", ">"], [">", "="])
        self.assertTrue(tc.operator_compliance(f, "="))
        self.assertTrue(tc.operator_compliance(f, ">"))
        self.assertFalse(tc.channel_compliance(f))

    def test_the_channel_test_passes_an_exact_channel(self):
        f = self._reading("literal_operator", ["=", ">"], ["=", ">"])
        self.assertTrue(tc.channel_compliance(f))

    def test_the_channel_test_passes_an_empty_drop_channel(self):
        f = self._reading("scene_summary", ["="], [])
        self.assertTrue(tc.channel_compliance(f))


# ---------------------------------------------------------------------------
# the classifier
# ---------------------------------------------------------------------------

class TestClassifyDisagreement(unittest.TestCase):
    """Each class has a planted case that must land in it."""

    def _pair(self, sid_a="literal_operator", sid_b="scene_summary",
              verdict_a="pass", verdict_b="pass", failed_a=(),
              failed_b=(), checks_a=None, checks_b=None):
        checks_a = checks_a or {c: check_row(c) for c in tc.CHECKS}
        checks_b = checks_b or {c: check_row(c) for c in tc.CHECKS}
        for c in failed_a:
            checks_a[c] = check_row(c, passed=False)
        for c in failed_b:
            checks_b[c] = check_row(c, passed=False)
        a = facts("1", sid_a, "gloss a", [], verdict=verdict_a,
                  check_rows=list(checks_a.values()))
        b = facts("1", sid_b, "gloss b", [], verdict=verdict_b,
                  check_rows=list(checks_b.values()))
        return a, b

    def test_a_breach_by_one_is_a_rule_violation_naming_it(self):
        a, b = self._pair()
        cls, gov, resp = tc.classify_disagreement(
            "operator", a, b, compliant_a=True, compliant_b=False)
        self.assertEqual(cls, tc.CLASS_RULE_VIOLATION)
        self.assertEqual(gov, "operator_fidelity")
        self.assertEqual(resp, "scene_summary")

    def test_a_breach_by_both_names_both(self):
        # two defects are still not evidence about the text, and the row
        # says so by naming both rather than picking one.
        a, b = self._pair()
        cls, gov, resp = tc.classify_disagreement(
            "operator", a, b, compliant_a=False, compliant_b=False)
        self.assertEqual(cls, tc.CLASS_RULE_VIOLATION)
        self.assertEqual(resp, "literal_operator+scene_summary")

    def test_instructed_divergence_is_a_contract_not_a_violation(self):
        # the keep-strategy kept `<<` as told and the drop-strategy
        # dropped it as told; the keep-strategy's `/` failure is another
        # row's business.
        a, b = self._pair(failed_a=("operator_fidelity",))
        cls, gov, resp = tc.classify_disagreement(
            "operator", a, b, compliant_a=True, compliant_b=True)
        self.assertEqual(cls, tc.CLASS_CONTRACT)
        self.assertEqual(gov, "operator_fidelity")
        self.assertEqual(resp, "")

    def test_two_compliant_readings_differing_without_instruction_are_genuine(
            self):
        a, b = self._pair()
        cls, gov, resp = tc.classify_disagreement(
            "content", a, b)
        self.assertEqual(cls, tc.CLASS_GENUINE)
        self.assertEqual(gov, "none")
        self.assertEqual(resp, "")

    def test_a_defective_reading_diverging_on_an_ungoverned_surface(self):
        a, b = self._pair(verdict_a="fail", failed_a=("operator_fidelity",))
        cls, gov, resp = tc.classify_disagreement("content", a, b)
        self.assertEqual(cls, tc.CLASS_DEFECT)
        self.assertEqual(resp, "literal_operator")

    def test_a_defect_on_a_governed_surface_is_still_a_violation(self):
        # the reading failed the governing check *on this surface*, so
        # the row is a rule violation even though the other reading is
        # the one whose gloss differs in content.
        a, b = self._pair(verdict_a="fail", failed_a=("entity_fidelity",))
        cls, gov, resp = tc.classify_disagreement(
            "entity", a, b, compliant_a=False, compliant_b=True)
        self.assertEqual(cls, tc.CLASS_RULE_VIOLATION)
        self.assertEqual(resp, "literal_operator")

    def test_entity_and_residue_surfaces_have_the_same_rule(self):
        a, b = self._pair()
        for kind, state in (("entity", True), ("residue", False)):
            cls, gov, resp = tc.classify_disagreement(
                kind, a, b, compliant_a=state, compliant_b=not state)
            self.assertEqual(cls, tc.CLASS_RULE_VIOLATION)
            # the responsible agent is the one whose own treatment of the
            # surface breached the rule: `b` when `a` complied, else `a`.
            self.assertEqual(resp, "scene_summary" if state
                             else "literal_operator")


# ---------------------------------------------------------------------------
# the comparer
# ---------------------------------------------------------------------------

class TestCompareReadings(unittest.TestCase):
    """The log rows a pair of synthetic readings produces."""

    def test_identical_readings_produce_no_rows(self):
        # two agents pinned to the same text (clause 1005's
        # register_paraphrase / entity_consistent in the corpus): the log
        # records divergence, and there is none.
        a = facts("1", "register_paraphrase", "while devouring body-OMOTYA "
                  "of control deficiency to bio=less= intermediate area",
                  ["=", "="], entities=ents("OMOTYA"))
        b = facts("1", "entity_consistent", a["gloss"], ["=", "="],
                  entities=ents("OMOTYA"))
        self.assertEqual(tc.compare_readings(a, b), [])
        m = tc.pair_metrics(a, b)
        self.assertEqual(m["n_rows"], 0)
        self.assertEqual(m["content_jaccard"], 1.0)
        self.assertTrue(m["both_compliant"])

    def test_a_dropped_entity_is_logged_with_its_governing_check(self):
        a = facts("1", "register_paraphrase", "Put out BABEL=TV> noise>",
                  ["=", ">"], entities=ents("BABEL=TV"))
        b = facts("1", "scene_summary", "BABEL is put out", [],
                  entities=ents("BABEL=TV"))
        rows = tc.compare_readings(a, b)
        entity_rows = [r for r in rows if r["surface_kind"] == "entity"]
        self.assertEqual(len(entity_rows), 1)
        r = entity_rows[0]
        self.assertEqual(r["surface"], "BABEL=TV")
        self.assertEqual(r["state_a"], "carried")
        self.assertEqual(r["state_b"], "dropped")
        self.assertEqual(r["classification"], tc.CLASS_RULE_VIOLATION)
        self.assertEqual(r["governing_check"], "entity_fidelity")
        self.assertEqual(r["responsible"], "scene_summary")

    def test_a_smoothed_residue_is_logged_against_the_kept_reading(self):
        a = facts("1", "register_paraphrase",
                  "her omanko hydromachine of ToKAGE", [],
                  entities=ents("ToKAGE"), residue=["omanko"])
        b = facts("1", "plain_english",
                  "her machine of ToKAGE", [], entities=ents("ToKAGE"),
                  residue=["omanko"])
        rows = tc.compare_readings(a, b)
        residue_rows = [r for r in rows if r["surface_kind"] == "residue"]
        self.assertEqual(len(residue_rows), 1)
        r = residue_rows[0]
        self.assertEqual(r["state_a"], "kept")
        self.assertEqual(r["state_b"], "smoothed")
        self.assertEqual(r["classification"], tc.CLASS_RULE_VIOLATION)
        self.assertEqual(r["governing_check"], "residue_discipline")
        self.assertEqual(r["responsible"], "plain_english")

    def test_a_verdict_row_is_the_pairs_bottom_line(self):
        a = facts("1", "register_paraphrase", "I surrender to the soul/gram "
                  "of self", ["/"], verdict="pass",
                  check_rows=[check_row(c) for c in tc.CHECKS])
        b = facts("1", "literal_operator", "I surrender [l] / to the "
                  "soul/gram of self", ["/"], verdict="fail",
                  check_rows=[check_row(c) for c in tc.CHECKS]
                  + [check_row("operator_fidelity", passed=False)])
        # rebuild b's check rows properly: operator_fidelity failed
        b = facts("1", "literal_operator", "I surrender [l] / to the "
                  "soul/gram of self", ["/"], verdict="fail",
                  check_rows=[check_row(c, passed=(c != "operator_fidelity"))
                              for c in tc.CHECKS])
        rows = tc.compare_readings(a, b)
        verdict_rows = [r for r in rows if r["surface_kind"] == "verdict"]
        self.assertEqual(len(verdict_rows), 1)
        r = verdict_rows[0]
        self.assertEqual(r["state_a"], "pass")
        self.assertEqual(r["state_b"], "fail")
        self.assertEqual(r["classification"], tc.CLASS_RULE_VIOLATION)
        self.assertEqual(r["governing_check"], "operator_fidelity")
        self.assertEqual(r["responsible"], "literal_operator")

    def test_content_divergence_between_compliant_readings_is_genuine(self):
        a = facts("1", "register_paraphrase",
                  "<< I draw in the porno nerve gas of boy-roid", ["<<"])
        b = facts("1", "scene_summary",
                  "The speaker sucks the porno nerve gas of boy-roid.", [])
        rows = tc.compare_readings(a, b)
        content_rows = [r for r in rows if r["surface_kind"] == "content"]
        self.assertEqual(len(content_rows), 1)
        r = content_rows[0]
        self.assertEqual(r["classification"], tc.CLASS_GENUINE)
        self.assertEqual(r["governing_check"], "none")
        self.assertIn("draw", r["state_a"].split(","))
        self.assertIn("sucks", r["state_b"].split(","))

    def test_a_reorder_row_is_logged_when_no_symbol_state_differs(self):
        # Both readings share the clause's channel as `ops_in` -- it is
        # #17's stream for the clause, not a per-reading choice -- so a
        # keep-strategy that reordered it is in breach of its own contract
        # and fails `operator_fidelity`, as the planted check row records.
        a = facts("1", "literal_operator", "A > B = C", ["=", ">"],
                  verdict="fail",
                  check_rows=[check_row(c, passed=(c != "operator_fidelity"))
                              for c in tc.CHECKS])
        b = facts("1", "entity_consistent", "A = B > C", ["=", ">"])
        rows = tc.compare_readings(a, b)
        order_rows = [r for r in rows if r["surface"] == "channel order"]
        self.assertEqual(len(order_rows), 1)
        self.assertEqual(order_rows[0]["state_a"], ">,=")
        self.assertEqual(order_rows[0]["state_b"], "=,>")
        self.assertEqual(order_rows[0]["classification"],
                         tc.CLASS_RULE_VIOLATION)
        self.assertEqual(order_rows[0]["responsible"], "literal_operator")


# ---------------------------------------------------------------------------
# the corpus: the committee's re-derivation must agree with #63
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts absent")
class TestPublishedAgrees(unittest.TestCase):
    """Every cross-check `run_committee` performs, as a test.

    The committee's independence is architectural (it re-derives from the
    pins and the frozen artifacts) and these tests are what make a drift
    between its judgement and #63's published record a suite failure
    rather than a silent rebuild.
    """

    @classmethod
    def setUpClass(cls):
        cls.sample, cls.facts = tc.run_committee()
        cls.pub_glosses, cls.pub_checks, _agree, cls.pub_sample = \
            tc.load_published()

    def test_the_redrawn_sample_is_63s_published_sample(self):
        published = {r["clause_id"]: r["stratum"] for r in self.pub_sample}
        self.assertEqual(self.sample, published)

    def test_every_rederived_gloss_is_what_63_published(self):
        pub = {(r["clause_id"], r["strategy"]): r for r in self.pub_glosses}
        for (cid, sid), f in self.facts.items():
            self.assertEqual(f["gloss"], pub[(cid, sid)]["gloss"])
            self.assertEqual(f["verdict"], pub[(cid, sid)]["verdict"])

    def test_every_critic_check_row_matches_the_published_row(self):
        pub = {(r["clause_id"], r["strategy"], r["check"]): r
               for r in self.pub_checks}
        for (cid, sid), f in self.facts.items():
            for name, c in f["checks"].items():
                p = pub[(cid, sid, name)]
                self.assertEqual(c["applies"], p["applies"] == "True")
                self.assertEqual(c["pass"], p["pass"] == "True")

    def test_the_operator_columns_rederive_from_the_gloss_text(self):
        pub = {(r["clause_id"], r["strategy"]): r for r in self.pub_glosses}
        for (cid, sid), f in self.facts.items():
            self.assertEqual(",".join(f["ops_out"]) or "(none)",
                             pub[(cid, sid)]["operators_out"] or "(none)")

    def test_the_residue_mirror_agrees_with_the_check_on_the_corpus(self):
        """The mirror is pinned on synthetic cases; this pins it on the
        book's own residue readings (clause 2789's `omanko`)."""
        residue = gp.load_clause_residue()
        checked = 0
        for cid, rows in residue.items():
            surfaces = gp.residue_surfaces(rows)
            if not surfaces:
                continue
            for (c, sid), f in self.facts.items():
                if c != cid:
                    continue
                want, _ = gp.check_residue_discipline(surfaces, f["gloss"])
                mirrored = all(f["residue_kept"][s] for s in surfaces)
                self.assertEqual(want, mirrored,
                                 "residue mirror drifted on %s/%s" % (c, sid))
                checked += 1
        self.assertGreater(checked, 0, "no residue clause was checked")


@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts absent")
class TestLogInvariants(unittest.TestCase):
    """Suite-wide invariants over every row of the disagreement log."""

    @classmethod
    def setUpClass(cls):
        cls.sample, cls.facts = tc.run_committee()
        cls.pairs, cls.log = tc.build_log(cls.facts, cls.sample)

    def test_every_clause_has_all_ten_pairs(self):
        per_clause = collections.Counter(p["clause_id"] for p in self.pairs)
        self.assertEqual(per_clause, {cid: 10 for cid in self.sample})
        self.assertEqual(len(self.pairs), 50)

    def test_every_genuine_row_has_two_fully_compliant_readings(self):
        """The definition of the class, asserted over the whole log."""
        for r in self.log:
            if r["classification"] != tc.CLASS_GENUINE:
                continue
            fa = self.facts[(r["clause_id"], r["agent_a"])]
            fb = self.facts[(r["clause_id"], r["agent_b"])]
            self.assertEqual(fa["verdict"], "pass")
            self.assertEqual(fb["verdict"], "pass")

    def test_no_governing_check_separates_a_genuine_row(self):
        """A genuine row's pair passed the governing check (or none
        governs it) -- re-derived from the facts, not from the row."""
        for r in self.log:
            if r["classification"] != tc.CLASS_GENUINE:
                continue
            gov = tc.KIND_GOVERNING[r["surface_kind"]]
            if gov is None:
                self.assertEqual(r["governing_check"], "none")
                continue
            fa = self.facts[(r["clause_id"], r["agent_a"])]
            fb = self.facts[(r["clause_id"], r["agent_b"])]
            self.assertEqual(fa["checks"][gov]["pass"], True)
            self.assertEqual(fb["checks"][gov]["pass"], True)

    def test_every_rule_violation_names_a_reader_that_broke_the_rule(self):
        """Per-surface compliance implies the aggregate check failure:
        a dropped operator, a dropped entity and a smoothed residue each
        fail the check that governs the surface."""
        for r in self.log:
            if r["classification"] != tc.CLASS_RULE_VIOLATION:
                continue
            gov = r["governing_check"]
            for agent in r["responsible"].split("+"):
                f = self.facts[(r["clause_id"], agent)]
                self.assertFalse(f["checks"][gov]["pass"],
                                 "%s named for %s but passed it"
                                 % (agent, gov))

    def test_every_defect_row_has_a_defective_participant(self):
        for r in self.log:
            if r["classification"] != tc.CLASS_DEFECT:
                continue
            named = r["responsible"].split("+")
            self.assertTrue(named)
            for agent in named:
                f = self.facts[(r["clause_id"], agent)]
                self.assertNotEqual(f["verdict"], "pass")

    def test_every_contract_row_is_an_instructed_operator_divergence(self):
        for r in self.log:
            if r["classification"] != tc.CLASS_CONTRACT:
                continue
            fa = self.facts[(r["clause_id"], r["agent_a"])]
            fb = self.facts[(r["clause_id"], r["agent_b"])]
            self.assertEqual(r["surface_kind"], "operator")
            self.assertNotEqual(
                gp.STRATEGY_OWES[fa["strategy"]]["operators"],
                gp.STRATEGY_OWES[fb["strategy"]]["operators"])
            # both readings treated this surface as their own contract
            # requires -- re-derived, not read off the row.
            sym = r["surface"]
            if sym == "channel order":
                self.assertTrue(tc.channel_compliance(fa))
                self.assertTrue(tc.channel_compliance(fb))
            else:
                self.assertTrue(tc.operator_compliance(fa, sym))
                self.assertTrue(tc.operator_compliance(fb, sym))

    def test_every_content_row_states_the_exact_symmetric_difference(self):
        for r in self.log:
            if r["surface_kind"] != "content":
                continue
            fa = self.facts[(r["clause_id"], r["agent_a"])]
            fb = self.facts[(r["clause_id"], r["agent_b"])]
            want_a = ",".join(sorted(fa["content_words"]
                                     - fb["content_words"])) or "(none)"
            want_b = ",".join(sorted(fb["content_words"]
                                     - fa["content_words"])) or "(none)"
            self.assertEqual(r["state_a"], want_a)
            self.assertEqual(r["state_b"], want_b)

    def test_the_classification_of_every_row_rederives(self):
        """The whole log, re-classified from the facts by the published
        rule, equals the published classification -- the strongest form of
        the determinism claim, applied to the load-bearing output."""
        for r in self.log:
            fa = self.facts[(r["clause_id"], r["agent_a"])]
            fb = self.facts[(r["clause_id"], r["agent_b"])]
            if r["surface_kind"] == "verdict":
                continue  # the pair-level row is built, not classified
            if r["surface_kind"] == "operator" and \
                    r["surface"] == "channel order":
                ca, cb = tc.channel_compliance(fa), tc.channel_compliance(fb)
            elif r["surface_kind"] == "operator":
                ca = tc.operator_compliance(fa, r["surface"])
                cb = tc.operator_compliance(fb, r["surface"])
            elif r["surface_kind"] == "entity":
                ca = fa["entity_carried"][r["surface"]]
                cb = fb["entity_carried"][r["surface"]]
            elif r["surface_kind"] == "residue":
                ca = fa["residue_kept"][r["surface"]]
                cb = fb["residue_kept"][r["surface"]]
            else:
                ca = cb = None
            cls, gov, _resp = tc.classify_disagreement(
                r["surface_kind"], fa, fb, ca, cb)
            self.assertEqual(cls, r["classification"],
                             "row re-classified differently: %r" % r)
            self.assertEqual(gov, r["governing_check"])


@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts absent")
class TestPinnedFindings(unittest.TestCase):
    """The corpus's own findings, pinned.

    These are the observations the taxonomy exists to make; if the pins
    ever stop holding, the committee has drifted from the book (or the
    book from the committee) and a human should look.
    """

    @classmethod
    def setUpClass(cls):
        cls.sample, cls.facts = tc.run_committee()
        cls.pairs, cls.log = tc.build_log(cls.facts, cls.sample)

    def _rows(self, cid, cls=None):
        return [r for r in self.log
                if r["clause_id"] == cid
                and (cls is None or r["classification"] == cls)]

    def test_clause_1003_has_no_genuine_ambiguity(self):
        """The control: only one of its five readings passes, so every
        divergence is a defect or a contract."""
        self.assertEqual(self._rows("1003", tc.CLASS_GENUINE), [])
        self.assertTrue(self._rows("1003", tc.CLASS_RULE_VIOLATION))

    def test_clause_10s_verb_divergence_is_genuine(self):
        """`sucks` versus `draw in`: two compliant readings of the same
        verb, no instruction between them."""
        rows = [r for r in self._rows("10", tc.CLASS_GENUINE)
                if r["agent_a"] == "register_paraphrase"
                and r["agent_b"] == "scene_summary"]
        self.assertEqual(len(rows), 1)
        self.assertIn("sucks", rows[0]["state_b"].split(","))
        self.assertIn("draw", rows[0]["state_a"].split(","))

    def test_clause_2789s_ocr_first_word_divergence_is_genuine(self):
        """`l surrender` versus `I surrender`: the OCR-misread first word
        read two ways by two compliant readings."""
        rows = [r for r in self._rows("2789", tc.CLASS_GENUINE)
                if {r["agent_a"], r["agent_b"]}
                == {"register_paraphrase", "plain_english"}]
        self.assertEqual(len(rows), 1)
        states = rows[0]["state_a"] + "," + rows[0]["state_b"]
        self.assertTrue("l" in states.split(",")
                        or "myself" in states.split(","))

    def test_clause_101_is_fully_compliant_yet_still_divergent(self):
        """All five readings pass every check, and all ten pairs still
        diverge on content: compliance does not imply agreement."""
        for sid in gp.STRATEGY_IDS:
            self.assertEqual(self.facts[("101", sid)]["verdict"], "pass")
        self.assertEqual(len(self._rows("101", tc.CLASS_GENUINE)), 10)

    def test_two_agents_pinned_to_the_same_text_agree_exactly(self):
        """Clause 1005's register_paraphrase and entity_consistent
        returned byte-identical glosses: the committee's answer for that
        pair is no divergence at all."""
        pair = [p for p in self.pairs if p["clause_id"] == "1005"
                and {p["agent_a"], p["agent_b"]}
                == {"register_paraphrase", "entity_consistent"}]
        self.assertEqual(len(pair), 1)
        self.assertEqual(pair[0]["n_rows"], 0)
        self.assertEqual(pair[0]["content_jaccard"], 1.0)

    def test_the_headline_counts(self):
        summary = tc.build(out=Path(tempfile.mkdtemp()), quiet=True)
        self.assertEqual(summary["clauses"], 5)
        self.assertEqual(summary["readings"], 25)
        self.assertEqual(summary["pairs"], 50)
        self.assertEqual(summary["fully_compliant_readings"], 15)
        self.assertEqual(summary["compliant_pairs"], 20)
        self.assertEqual(summary["clauses_with_genuine_ambiguity"],
                         ["10", "101", "1005", "2789"])
        self.assertEqual(summary["clauses_without_genuine_ambiguity"],
                         ["1003"])
        self.assertEqual(sum(summary["rows_by_class"].values()),
                         summary["rows_total"])


@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts absent")
class TestBuild(unittest.TestCase):
    def test_the_pinned_build_is_byte_identical_on_rebuild(self):
        """Determinism is what makes the disagreement log reviewable."""
        a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        tc.build(out=a, quiet=True)
        tc.build(out=b, quiet=True)
        for name in ("readings.tsv", "pairs.tsv", "disagreement_log.tsv",
                     "diagnostics.tsv"):
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(),
                             name)

    def test_the_summary_reproduces_identically_apart_from_the_stamp(self):
        a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        sa = tc.build(out=a, quiet=True)
        sb = tc.build(out=b, quiet=True)
        sa.pop("generated"), sb.pop("generated")
        self.assertEqual(sa, sb)

    def test_the_map_publishes_the_headline_and_the_audit(self):
        out = Path(tempfile.mkdtemp())
        tc.build(out=out, quiet=True)
        text = (out / "disagreement_map.md").read_text(encoding="utf-8")
        self.assertIn("## Headline", text)
        self.assertIn("## The map", text)
        self.assertIn("## Guardrails (L9.8 audit)", text)
        self.assertIn("genuine_ambiguity", text)

    def test_a_missing_63_artifact_is_a_loud_error(self):
        """The committee judges #63's readings; without them it must say
        so rather than generate readings of its own."""
        with unittest.mock.patch.object(tc, "PUBLISHED_GLOSSES",
                                         Path("/nonexistent/glosses.tsv")):
            with self.assertRaises(tc.CommitteeError):
                tc.load_published()


if __name__ == "__main__":
    unittest.main()
