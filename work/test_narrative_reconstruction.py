#!/usr/bin/env python3
"""Regression suite for #66 narrative reconstruction (entity/event tracking).

Run the whole work/ suite with:
    python -m unittest discover -s work -p 'test_*.py'

Two layers, like the workspace's other suites:

  * synthetic tests on hand-built units/clauses pin the mention matcher
    (phrase precedence, referents), the verb classification, the
    direction rule and its windows, the confidence pricing, the clause
    partition, the actor aggregates, the synopsis element evaluator and
    the failure paths (an event off the clause table, a confidence
    outside the scheme, a reconciliation mismatch must all fail loudly);
  * corpus tests (guarded by skipUnless) build the skeleton twice into
    temporary directories, check determinism and the slice build, and
    re-derive every published number against the frozen #17/#27/#30/#33/
    #62 layers, including the known-content findings (ADAM murders, the
    dog and self as referents, the Super Cherry 666 divergence).

build() is exercised into temporary directories only, never into
analysis/narrative/.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))
import narrative_reconstruction as nr

HAVE_ARTIFACTS = all(p.exists() for p in (
    nr.UNITS, nr.CLAUSES, nr.GLOSSARY, nr.VERBS, nr.INTERLINEAR))


# ---------------------------------------------------------------------------
# synthetic helpers
# ---------------------------------------------------------------------------
def unit(surface, start):
    return {"uid": start, "start": start, "end": start + len(surface),
            "surface": surface, "lower": surface.lower()}


def units_of(surfaces):
    out, pos = [], 0
    for s in surfaces:
        out.append(unit(s, pos))
        pos += len(s) + 1
    return out


def vocab(word, entity, role="persona"):
    return {word: {"entity": entity, "role": role}}


def clause(cid, page, start, end, preview="", region="body",
           container="block"):
    return {"clause_id": cid, "page": str(page), "region": region,
            "container": container, "start": str(start), "end": str(end),
            "preview": preview}


def run_clause(cl, surfaces, verbs, surface_vocab):
    """The full per-clause pipeline on synthetic input.

    surface_vocab maps surface_lower -> {"entity", "role"}, exactly the
    shape load_glossary() emits.
    """
    us = units_of(surfaces)
    single, phrases = nr.build_surface_index(surface_vocab)
    mentions = nr.match_mentions(us, single, phrases)
    vrecs = nr.verb_records(us, verbs)
    return nr.extract_events(cl, us, mentions, vrecs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestParsePages(unittest.TestCase):
    def test_single_page(self):
        self.assertEqual(nr.parse_pages("4"), (4, 4))

    def test_range(self):
        self.assertEqual(nr.parse_pages("4-6"), (4, 6))

    def test_none_is_the_whole_book(self):
        self.assertIsNone(nr.parse_pages(None))
        self.assertIsNone(nr.parse_pages(""))

    def test_reversed_range_fails(self):
        with self.assertRaises(SystemExit):
            nr.parse_pages("6-4")

    def test_garbage_fails(self):
        with self.assertRaises(SystemExit):
            nr.parse_pages("four")


# ---------------------------------------------------------------------------
# mention layer
# ---------------------------------------------------------------------------
class TestMentions(unittest.TestCase):
    def test_phrase_wins_over_its_constituent(self):
        single = vocab("adam", "ADAM", "persona")
        phrases = {"adam": [(["adam", "doll"],
                             {"entity": "ADAM Doll", "role": "creature"})]}
        us = units_of(["the", "ADAM", "Doll", "murders"])
        mentions = nr.match_mentions(us, single, phrases)
        self.assertEqual([m["entity"] for m in mentions], ["ADAM Doll"])

    def test_single_unit_entity(self):
        single = vocab("adam", "ADAM", "persona")
        us = units_of(["ADAM", "was", "here"])
        mentions = nr.match_mentions(us, single, {})
        self.assertEqual([m["entity"] for m in mentions], ["ADAM"])
        self.assertEqual(mentions[0]["kind"], "glossary")

    def test_referents_are_tracked_but_not_glossary(self):
        us = units_of(["the", "dog", "and", "self"])
        mentions = nr.match_mentions(us, {}, {})
        self.assertEqual([m["entity"] for m in mentions],
                         ["the dog", "self"])
        for m in mentions:
            self.assertEqual(m["kind"], "referent")
            self.assertEqual(m["role"], "referent")

    def test_dogs_plural_maps_to_the_same_referent(self):
        us = units_of(["dogs", "of", "zero"])
        mentions = nr.match_mentions(us, {}, {})
        self.assertEqual([m["entity"] for m in mentions], ["the dog"])

    def test_glued_compounds_are_not_referent_mentions(self):
        # dog-strand / self-ruin stay their own surfaces (disclosed rule)
        us = units_of(["dog-strand", "self-ruin"])
        self.assertEqual(nr.match_mentions(us, {}, {}), [])

    def test_case_insensitive_surface_match(self):
        single = vocab("sato corporation", "Sato Corporation",
                             "organization")
        phrases = {"sato": [(["sato", "corporation"],
                             {"entity": "Sato Corporation",
                              "role": "organization"})]}
        us = units_of(["SATO", "Corporation", "hunts"])
        mentions = nr.match_mentions(us, single, phrases)
        self.assertEqual([m["entity"] for m in mentions],
                         ["Sato Corporation"])
        self.assertEqual(mentions[0]["surface"], "SATO Corporation")

    def test_mentions_never_overlap(self):
        single = {**vocab("adam", "ADAM"), **vocab("doll", "Doll")}
        us = units_of(["ADAM", "Doll", "ADAM"])
        mentions = nr.match_mentions(us, single, {})
        self.assertEqual([(m["first_idx"], m["last_idx"]) for m in mentions],
                         [(0, 0), (1, 1), (2, 2)])


# ---------------------------------------------------------------------------
# verb layer + the direction rule
# ---------------------------------------------------------------------------
class TestVerbLayer(unittest.TestCase):
    def test_source_prices_confidence(self):
        verbs = {"murder": "brown", "is": "brown", "fecundates": "book",
                 "dogs": "morph_s"}
        recs = nr.verb_records(units_of(["ADAM", "fecundates", "is",
                                         "dogs"]), verbs)
        got = [(r["verb"], r["conf"], r["class"]) for r in recs]
        self.assertEqual(got, [("fecundates", "high", "action"),
                               ("is", "high", "copula"),
                               ("dogs", "low", "action")])

    def test_morph_s_is_never_high(self):
        self.assertEqual(nr.event_confidence("low", 1), "medium")
        self.assertEqual(nr.event_confidence("low", 3), "medium")
        self.assertEqual(nr.event_confidence("low", 6), "low")

    def test_high_verb_gap_pricing(self):
        self.assertEqual(nr.event_confidence("high", 0), "high")
        self.assertEqual(nr.event_confidence("high", 4), "high")
        self.assertEqual(nr.event_confidence("high", 5), "medium")
        self.assertEqual(nr.event_confidence("high", 8), "medium")
        self.assertEqual(nr.event_confidence("high", 9), "low")

    def test_non_verb_units_are_not_verbs(self):
        self.assertEqual(nr.verb_records(units_of(["the", "ADAM", "Doll"]),
                                         {"murder": "brown"}), [])


class TestEventRule(unittest.TestCase):
    CL = clause("c1", 15, 0, 100, preview="ADAM murders the dog")

    def test_left_is_actor_right_is_target(self):
        events = run_clause(
            self.CL, ["ADAM", "murders", "the", "dog"],
            {"murders": "brown"},
            vocab("adam", "ADAM"))
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual((e["actor"], e["verb"], e["target"]),
                         ("ADAM", "murders", "the dog"))
        self.assertEqual(e["pattern"], "AVT")
        self.assertEqual(e["confidence"], "high")
        self.assertEqual(e["verb_class"], "action")

    def test_genitive_chain_reads_the_referent(self):
        # the disclosed convention: nearest entity left of the verb, even
        # inside a genitive NP
        events = run_clause(
            self.CL, ["the", "brain", "of", "ADAM", "murders", "nothing"],
            {"murders": "brown"}, vocab("adam", "ADAM"))
        self.assertEqual(events[0]["actor"], "ADAM")
        self.assertEqual(events[0]["target"], "")
        self.assertEqual(events[0]["pattern"], "AV")
        self.assertEqual(events[0]["actor_gap"], 1)

    def test_verb_first_target_only(self):
        events = run_clause(
            self.CL, ["murders", "the", "dog"], {"murders": "brown"}, {})
        self.assertEqual(events[0]["pattern"], "VT")
        self.assertEqual(events[0]["actor"], "")
        self.assertEqual(events[0]["target"], "the dog")

    def test_window_limits_are_enforced(self):
        far = ["ADAM"] + ["pad"] * 13 + ["murders"]
        events = run_clause(self.CL, far, {"murders": "brown"},
                            vocab("adam", "ADAM"))
        self.assertEqual(events, [])      # actor beyond ACTOR_MAX_GAP
        near = ["ADAM"] + ["pad"] * 11 + ["murders"]
        events = run_clause(self.CL, near, {"murders": "brown"},
                            vocab("adam", "ADAM"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["actor_gap"], 12)

    def test_target_window_is_tighter(self):
        near = ["murders"] + ["pad"] * 8 + ["dog"]
        events = run_clause(self.CL, near, {"murders": "brown"}, {})
        self.assertEqual(events, [])      # target beyond TARGET_MAX_GAP
        ok = ["murders"] + ["pad"] * 7 + ["dog"]
        events = run_clause(self.CL, ok, {"murders": "brown"}, {})
        self.assertEqual(events[0]["target_gap"], 8)

    def test_mention_spanning_the_verb_stands_on_neither_side(self):
        # a glossary surface that contains the verb unit: the mention
        # spans the verb, so it can stand on neither side of it (pinned
        # directly -- in the real stream units are consumed by the run)
        cl = clause("c2", 1, 0, 50, preview="x")
        us = units_of(["murders", "Doll", "was"])
        single = vocab("murders doll", "ADAM Doll", "creature")
        s, p = nr.build_surface_index(single)
        mentions = nr.match_mentions(us, s, p)
        self.assertEqual([m["entity"] for m in mentions], ["ADAM Doll"])
        self.assertEqual((mentions[0]["first_idx"], mentions[0]["last_idx"]),
                         (0, 1))
        vrecs = nr.verb_records(us, {"murders": "brown", "was": "brown"})
        events = nr.extract_events(cl, us, mentions, vrecs)
        self.assertEqual([e for e in events if e["verb"] == "murders"], [])
        for e in events:
            self.assertNotEqual(e["target"], "ADAM Doll")

    def test_event_carries_its_citation_fields(self):
        events = run_clause(
            self.CL, ["ADAM", "murders", "the", "dog"],
            {"murders": "brown"}, vocab("adam", "ADAM"))
        e = events[0]
        self.assertEqual(e["page"], 15)
        self.assertEqual(e["clause_id"], "c1")
        self.assertEqual(e["region"], "body")
        self.assertEqual(e["ast_container"], "block")
        self.assertEqual(e["verb_start"], 5)
        self.assertEqual(e["evidence"], "ADAM murders the dog")


# ---------------------------------------------------------------------------
# clause partition
# ---------------------------------------------------------------------------
class TestClausePartition(unittest.TestCase):
    def test_units_land_in_their_clause(self):
        rows = [clause("c1", 1, 0, 20), clause("c2", 1, 25, 50)]
        units = {1: [unit("a", 2), unit("b", 10), unit("c", 30)]}
        got, n_assigned, n_unassigned = nr.assign_units_to_clauses(rows,
                                                                   units)
        self.assertEqual(n_assigned, 3)
        self.assertEqual(n_unassigned, 0)
        self.assertEqual([u["surface"] for u in got["c1"]], ["a", "b"])
        self.assertEqual([u["surface"] for u in got["c2"]], ["c"])

    def test_gap_between_clauses_is_disclosed_not_dropped(self):
        rows = [clause("c1", 1, 0, 10)]
        units = {1: [unit("a", 2), unit("b", 40)]}
        got, n_assigned, n_unassigned = nr.assign_units_to_clauses(rows,
                                                                   units)
        self.assertEqual((n_assigned, n_unassigned), (1, 1))
        self.assertEqual(got["c1"][0]["surface"], "a")

    def test_clauses_on_a_page_with_no_units(self):
        rows = [clause("c1", 1, 0, 10), clause("c2", 2, 0, 10)]
        units = {1: [unit("a", 2)]}
        got, n_assigned, n_unassigned = nr.assign_units_to_clauses(rows,
                                                                   units)
        self.assertEqual((n_assigned, n_unassigned), (1, 0))
        self.assertNotIn("c2", got)


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------
def synth_events():
    """A miniature event table: ADAM murders the dog twice, the dog kills
    ADAM once, self copulates with nothing (AV pattern)."""
    base = {"page": 10, "clause_id": "c1", "region": "body",
            "ast_container": "block", "verb_source": "brown",
            "verb_conf": "high", "verb_class": "action",
            "actor_role": "persona", "actor_kind": "glossary",
            "target_role": "referent", "target_kind": "referent",
            "actor_gap": 1, "target_gap": 1, "confidence": "high",
            "evidence": "x"}
    events = []
    for i, (a, v, t) in enumerate([("ADAM", "murders", "the dog"),
                                   ("ADAM", "murders", "the dog"),
                                   ("the dog", "kills", "ADAM"),
                                   ("self", "is", "")]):
        e = dict(base, event_id="e%d" % i, verb=v, verb_start=i * 10,
                 actor=a, target=t, pattern="AVT" if t else "AV")
        e["cite"] = "p10:%d" % (i * 10)
        events.append(e)
    return events


class TestAggregates(unittest.TestCase):
    def test_actor_profile_does_done_to_targets_actors(self):
        events = synth_events()
        entities = {"ADAM": {"role": "persona", "n": 303, "kind": "glossary"}}
        mention_counts = {"ADAM": 3, "the dog": 4, "self": 1}
        mention_pages = {"ADAM": {10}, "the dog": {10, 11}, "self": {10}}
        rows = {r["entity"]: r for r in nr.build_actor_profiles(
            events, entities, mention_counts, mention_pages)}
        adam = rows["ADAM"]
        self.assertEqual(adam["n_does"], 2)
        self.assertEqual(adam["n_done_to"], 1)
        self.assertEqual(adam["top_does"], "murders(2)")
        self.assertEqual(adam["top_targets"], "the dog(2)")
        self.assertEqual(adam["top_actors"], "the dog(1)")
        self.assertEqual((adam["first_page"], adam["last_page"]), (10, 10))
        self.assertIn("the dog", rows)
        self.assertEqual(rows["the dog"]["role"], "referent")
        self.assertEqual(rows["the dog"]["kind"], "referent")

    def test_pairs_are_directed_and_counted(self):
        rows = {(r["actor"], r["target"]): r
                for r in nr.build_pairs(synth_events())}
        self.assertEqual(rows[("ADAM", "the dog")]["n_events"], 2)
        self.assertEqual(rows[("ADAM", "the dog")]["top_verbs"],
                         "murders(2)")
        self.assertEqual(rows[("the dog", "ADAM")]["n_events"], 1)
        self.assertNotIn(("self", ""), rows)     # AV rows are not pairs

    def test_arcs_bucket_pages(self):
        events = [{"page": 5}, {"page": 25}, {"page": 45}]
        mention_pages = {"ADAM": {5, 25}, "the dog": {45}}
        rows = nr.build_arcs(events, mention_pages, 60,
                             {5: 2, 25: 1, 45: 4})
        by_arc = {r["arc"]: r for r in rows}
        self.assertEqual(sorted(by_arc), ["p1-20", "p21-40", "p41-60"])
        self.assertEqual(by_arc["p1-20"]["n_events"], 1)
        self.assertEqual(by_arc["p1-20"]["n_mentions"], 2)
        self.assertEqual(by_arc["p1-20"]["n_adam"], 1)
        self.assertEqual(by_arc["p21-40"]["n_events"], 1)
        self.assertEqual(by_arc["p21-40"]["n_adam"], 1)
        self.assertEqual(by_arc["p41-60"]["n_events"], 1)
        self.assertEqual(by_arc["p41-60"]["n_dog"], 1)


# ---------------------------------------------------------------------------
# synopsis element evaluator
# ---------------------------------------------------------------------------
class TestSynopsisEvaluator(unittest.TestCase):
    def _ctx(self):
        rows = [clause("c1", 10, 0, 20, preview="ADAM murders the dog"),
                clause("c2", 200, 0, 20, preview="Super Cherry 666 hunts")]
        keys = {"c1": frozenset({"ADAM", "the dog", "fam:dog"}),
                "c2": frozenset({"Super Cherry 666", "fam:skull"})}
        events = synth_events()
        mpc = {("ADAM", 10): 3, ("the dog", 10): 2}
        fpc = {("dog", 10): 2, ("skull", 200): 1}
        return rows, keys, events, mpc, fpc

    def test_presence_cooccur_and_event_kinds(self):
        rows, keys, events, mpc, fpc = self._ctx()
        with mock.patch.object(nr, "SYNOPSIS_CLAIMS",
                               [("T1", "test claim", [
                                   ("presence", "ADAM"),
                                   ("presence", "fam:skull"),
                                   ("cooccur", "ADAM", "the dog"),
                                   ("cooccur", "ADAM", "fam:skull"),
                                   ("event", ("ADAM",), ("murders",), None),
                                   ("event", ("ADAM",), ("clash",), None),
                               ])]):
            check, elems, drops = nr.evaluate_synopsis(
                rows, keys, events, mpc, fpc, {"murders": "brown"})
        by_label = {e["element"]: e for e in elems}
        self.assertEqual(check[0]["verdict"], "partial")
        self.assertEqual(check[0]["n_found"], 4)
        self.assertEqual(by_label["present: ADAM"]["n_found"], 3)
        self.assertEqual(by_label["present: skull"]["n_found"], 1)
        self.assertEqual(by_label["cooccur: ADAM + the dog"]["n_found"], 1)
        self.assertEqual(by_label["cooccur: ADAM + skull"]["n_found"], 0)
        self.assertEqual(by_label["event: {ADAM} x {murders}"]["n_found"], 2)
        self.assertEqual(by_label["event: {ADAM} x {clash}"]["n_found"], 0)
        # 'clash' is not in the lexicon: disclosed, not silently dropped
        self.assertEqual(drops["clash"], 1)
        self.assertEqual(
            by_label["event: {ADAM} x {clash}"]["measure"].find("{}") >= 0,
            True)

    def test_supported_when_every_element_fires(self):
        rows, keys, events, mpc, fpc = self._ctx()
        with mock.patch.object(nr, "SYNOPSIS_CLAIMS",
                               [("T2", "t", [("presence", "ADAM")])]):
            check, elems, _ = nr.evaluate_synopsis(
                rows, keys, events, mpc, fpc, {})
        self.assertEqual(check[0]["verdict"], "supported")
        self.assertEqual(elems[0]["found"], "y")

    def test_unsupported_when_nothing_fires(self):
        rows, keys, events, mpc, fpc = self._ctx()
        with mock.patch.object(nr, "SYNOPSIS_CLAIMS",
                               [("T3", "t", [("presence", "Nobody"),
                                             ("cooccur", "Nobody",
                                              "fam:skull")])]):
            check, _elems, _ = nr.evaluate_synopsis(
                rows, keys, events, mpc, fpc, {})
        self.assertEqual(check[0]["verdict"], "unsupported")
        self.assertEqual(check[0]["n_found"], 0)
        self.assertEqual(check[0]["missing_elements"].count(";"), 1)

    def test_spec_labels_are_stable(self):
        self.assertEqual(nr.spec_label(("presence", "ADAM")),
                         "present: ADAM")
        self.assertEqual(nr.spec_label(("cooccur", "fam:gene", "fam:war")),
                         "cooccur: gene + war")
        self.assertEqual(
            nr.spec_label(("event", ("ADAM", "ANTI-ADAM"),
                           ("clashed", "clashing"), None)),
            "event: {ADAM|ANTI-ADAM} x {clashed|clashing}")


# ---------------------------------------------------------------------------
# failure paths: a table that does not close must fail the build
# ---------------------------------------------------------------------------
class TestClosureFailures(unittest.TestCase):
    def _base(self):
        events = synth_events()
        events[0]["confidence"] = "certain"          # outside the scheme
        recon = [{"quantity": "q", "earlier": "1", "here": "1",
                  "match": "ok"}]
        syn = [{"claim": "S1", "claim_text": "t", "n_elements": 1,
                "n_found": 1, "verdict": "supported", "missing": ""}]
        return events, recon, syn

    def test_unknown_confidence_fails(self):
        events, recon, syn = self._base()
        with self.assertRaises(nr.NarrativeError):
            nr.check_closures([clause("c1", 10, 0, 20)], events, {}, {},
                              recon, syn)

    def test_event_off_the_clause_table_fails(self):
        events, recon, syn = self._base()
        events[0]["confidence"] = "high"
        events[1]["clause_id"] = "nope"
        with self.assertRaises(nr.NarrativeError):
            nr.check_closures([clause("c1", 10, 0, 20)], events, {}, {},
                              recon, syn)

    def test_entity_outside_the_registry_fails(self):
        events, recon, syn = self._base()
        events[0]["confidence"] = "high"
        events[0]["actor"] = "Ghost"
        with self.assertRaises(nr.NarrativeError):
            nr.check_closures([clause("c1", 10, 0, 20)], events,
                              {"Ghost": {"role": "x", "kind": "glossary"}},
                              {"Ghost": {"role": "x", "kind": "glossary"}},
                              recon, syn)

    def test_reconciliation_mismatch_fails(self):
        events, recon, syn = self._base()
        events[0]["confidence"] = "high"
        recon[0]["match"] = "MISMATCH"
        with self.assertRaises(nr.NarrativeError):
            nr.check_closures([clause("c1", 10, 0, 20)], events, {}, {},
                              recon, syn)


# ---------------------------------------------------------------------------
# corpus tests (the real book)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts not present")
class CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="narrative-test-")
        cls.out = Path(cls._tmp.name) / "full"
        cls.summary = nr.build(cls.out, quiet=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def rows(self, name):
        with (self.out / name).open(encoding="utf-8") as fh:
            header = fh.readline().rstrip(chr(10)).split(chr(9))
            return [dict(zip(header, line.rstrip(chr(10)).split(chr(9))))
                    for line in fh if line.strip()]

    # -- artifact set ------------------------------------------------------

    def test_every_artifact_is_written(self):
        for name in ("events.tsv", "actors.tsv", "pairs.tsv", "per_arc.tsv",
                     "synopsis_check.tsv", "synopsis_elements.tsv",
                     "reconciliation.tsv", "diagnostics.tsv", "summary.json",
                     "narrative_report.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_summary_contract(self):
        s = self.summary
        self.assertEqual(s["schema"], "narrative/1")
        self.assertEqual(s["clauses"], 5613)          # #33 minus p224
        self.assertEqual(s["screened_units"],
                         s["units_assigned_to_clauses"] + s["units_unassigned"])
        self.assertEqual(s["units_unassigned"], 0)
        self.assertEqual(s["events"], len(self.rows("events.tsv")))
        self.assertEqual(sum(s["events_by_pattern"].values()), s["events"])
        self.assertEqual(sum(s["events_by_confidence"].values()),
                         s["events"])
        self.assertEqual(sum(s["events_by_verb_class"].values()),
                         s["events"])
        self.assertGreater(s["events"], 4000)
        self.assertEqual(s["referent_mentions"],
                         {"the dog": 459, "self": 364})

    def test_reconciliation_is_all_ok(self):
        rows = self.rows("reconciliation.tsv")
        self.assertTrue(rows)
        bad = [r["quantity"] for r in rows if r["match"] != "ok"]
        self.assertEqual(bad, [])
        quantities = " ".join(r["quantity"] for r in rows)
        for needle in ("#30", "#33", "#27", "#62"):
            self.assertIn(needle, quantities)

    # -- determinism -------------------------------------------------------

    def test_two_builds_are_byte_identical(self):
        second = Path(self._tmp.name) / "second"
        nr.build(second, quiet=True)
        for name in ("events.tsv", "actors.tsv", "pairs.tsv", "per_arc.tsv",
                     "synopsis_check.tsv", "synopsis_elements.tsv",
                     "reconciliation.tsv", "diagnostics.tsv"):
            self.assertEqual((self.out / name).read_bytes(),
                             (second / name).read_bytes(), name)

    def test_slice_build_reproduces_the_full_build_slice(self):
        sl = Path(self._tmp.name) / "slice"
        nr.build(sl, pages=(4, 6), quiet=True)
        sliced = self._rows_from(sl / "events.tsv")
        full = [r for r in self.rows("events.tsv")
                if 4 <= int(r["page"]) <= 6]
        # event_id is a per-build sequence number; every other field must
        # reproduce exactly, in the same order
        strip = lambda rows: [{k: v for k, v in r.items() if k != "event_id"}
                              for r in rows]
        self.assertEqual(strip(sliced), strip(full))
        self.assertTrue(sliced)
        # the synopsis check is skipped on a slice, not silently run
        with (sl / "synopsis_check.tsv").open(encoding="utf-8") as fh:
            next(fh)
            verdicts = {line.split(chr(9))[4] for line in fh if line.strip()}
        self.assertEqual(verdicts, {"n/a (slice build)"})

    def _rows_from(self, path):
        with path.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip(chr(10)).split(chr(9))
            return [dict(zip(header, line.rstrip(chr(10)).split(chr(9))))
                    for line in fh if line.strip()]

    # -- event table integrity ---------------------------------------------

    def test_event_rows_are_well_formed(self):
        events = self.rows("events.tsv")
        interlinear = nr.load_interlinear_clause_ids()
        cite_re = nr.re.compile(r"^p(\d+):(\d+)$")
        seen = set()
        for e in events:
            self.assertNotIn(e["event_id"], seen)
            seen.add(e["event_id"])
            self.assertTrue(cite_re.match(e["cite"]), e["cite"])
            self.assertIn(e["pattern"], nr.PATTERNS)
            self.assertIn(e["confidence"], nr.CONFIDENCES)
            self.assertIn(e["verb_conf"], ("high", "low"))
            self.assertIn(e["verb_class"], ("action", "copula"))
            self.assertTrue(e["actor"] or e["target"])
            self.assertTrue(1 <= int(e["page"]) <= 223)
            if interlinear is not None:
                self.assertIn(e["clause_id"], interlinear)
        ids = [e["event_id"] for e in events]
        self.assertEqual(ids, sorted(ids))
        cites = [(int(e["page"]), int(e["cite"].split(":")[1]))
                 for e in events]
        self.assertEqual(cites, sorted(cites))

    def test_every_named_verb_is_in_the_frozen_lexicon(self):
        verbs = nr.load_verbs()
        for e in self.rows("events.tsv"):
            self.assertIn(e["verb"], verbs, e["verb"])

    # -- known content ------------------------------------------------------

    def test_adam_murders(self):
        # the direction convention, on the book's central persona
        hits = [e for e in self.rows("events.tsv")
                if e["actor"] == "ADAM"
                and e["verb"] in ("murder", "murders", "murdered",
                                  "murdering")]
        self.assertTrue(hits)
        self.assertTrue(any(e["confidence"] == "high" for e in hits))

    def test_referents_act_and_are_acted_on(self):
        events = self.rows("events.tsv")
        for name in ("the dog", "self"):
            self.assertTrue(any(e["actor"] == name for e in events),
                            "%s never acts" % name)
            self.assertTrue(any(e["target"] == name for e in events),
                            "%s is never a target" % name)

    def test_the_dog_and_self_are_not_glossary_entities(self):
        _surfaces, entities = nr.load_glossary()
        self.assertNotIn("the dog", entities)
        self.assertNotIn("self", entities)

    def test_super_cherry_666_divergence_is_published(self):
        elements = {(r["claim"], r["element"]): r
                    for r in self.rows("synopsis_elements.tsv")}
        # the book's 3 mentions never stand with hunt/succubus/the corp
        self.assertEqual(
            elements[("S5", "event: {Super Cherry 666} x "
                            "{hunt|hunts|hunting|hunted}")]["n_found"], "0")
        self.assertEqual(
            elements[("S5", "cooccur: Super Cherry 666 + succubus")]
            ["n_found"], "0")
        # while the hunting formula itself is attested (ToKAGE's)
        self.assertGreater(
            int(elements[("S5", "cooccur: torture + victim")]
                ["n_found"]), 0)

    def test_synopsis_verdicts_match_the_findings(self):
        rows = {r["claim"]: r for r in self.rows("synopsis_check.tsv")}
        self.assertEqual(set(rows), {"S1", "S2", "S3", "S4", "S5"})
        self.assertEqual(rows["S1"]["verdict"], "supported")
        for claim in ("S2", "S3", "S4", "S5"):
            self.assertEqual(rows[claim]["verdict"], "partial", claim)
            self.assertTrue(rows[claim]["missing_elements"], claim)
        for r in rows.values():
            self.assertIn(r["verdict"], nr.VERDICTS)

    def test_report_states_the_direction_convention(self):
        text = (self.out / "narrative_report.md").read_text(encoding="utf-8")
        self.assertIn("LEFT is the actor", text)
        self.assertIn("sanity check", text)
        self.assertIn("ToKAGE", text)

    # -- failure paths on the real build ------------------------------------

    def test_a_bad_confidence_fails_the_real_build(self):
        real = nr.extract_events

        def doctored(clause_row, units, mentions, vrecs):
            rows = real(clause_row, units, mentions, vrecs)
            if rows:
                rows[0]["confidence"] = "certain"
            return rows

        out = Path(self._tmp.name) / "fail"
        with (mock.patch.object(nr, "extract_events", doctored),
              self.assertRaises(nr.NarrativeError)):
            nr.build(out, quiet=True)


if __name__ == "__main__":
    unittest.main(verbosity=1)
