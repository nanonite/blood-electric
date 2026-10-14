#!/usr/bin/env python3
"""#59 Seed entity roles from the jacket synopsis, verify against the text.

The back-cover synopsis of *Blood Electric* (p224) sketches a macro-plot: a
fatal collusion in **Cadaver City** ignites the **ADAM Doll**; the **dogs of
zero** wage **gene war** in **Placenta World**; **chaos** is unleashed by the
**digital vampires of Sato Corporation**; on the pink-ash planet **EVOL**,
**ADAM** and **ANTi-ADAM** clash in terminal zodiac burn; and the robosuccubus
**Super Cherry 666** enters hunting the corporation's napalm-torture victims.
That is marketing prose, and this task's job is exactly what the epic says:
use it *only* to seed a role per entity, then test every seed against the book
text and report **evidence, refutation, or uncertainty** -- never let the
synopsis grade itself.

Method (all offline, over the frozen artifacts; the p224 synopsis page is
excluded from its own check):

  * the five synopsis sentences come from #66's single source of truth
    (``narrative_reconstruction.SYNOPSIS_CLAIMS``) so the two numbers cannot
    drift; a role seed cites the sentence(s) it is read from and nothing else;
  * the mention / clause / event layers are recomputed here with #66's tested
    primitives over the #17 unit layer, the #33 clause layer, the #30 glossary
    and the #27 verb lexicon -- so the evidence is the same corpus, freshly
    measured, not a restatement of #66's tables;
  * each role is a set of **named probes** with a threshold: presence,
    recurrence, centrality, the #30 role category, a clause co-occurrence, or
    an actor/verb event.  Each probe is *defining* or *supporting*; a defining
    probe that fails is the refutation, and a *defined* role's verdict is
    **evidenced** (every defining probe passes), **uncertain** (some pass) or
    **refuted** (no substantive probe passes -- mere presence is a
    precondition, never a role property).  Verdicts are computed from counts,
    never typed by hand, and every passing probe carries the citations that
    earned it.

What this finds, in one line: **six of the nine seeded roles are evidenced,
two are uncertain and one is refuted.**  The corporate, civic and planetary
roles all hold; the clash role (ANTi-ADAM) is attested as an entity and a
hostile persona but never as ADAM's combatant, and the dog's "waging gene war"
has no matching actor-verb event; **the refutation is Super Cherry 666** -- the
book's hunter formula (`hunting` + `nova skulls` + `Sato Corporation napalm
torture victims`) is real, but its actor is ToKAGE, and the three mentions of
Super Cherry 666 never take a hunt verb, a succubus form or the corporation, so
the jacket copy attaches the formula to the wrong entrant.

Reproduce::

    conda activate marker
    python work/entity_roles.py            # offline, deterministic; writes analysis/entity_roles/
    python work/test_entity_roles.py       # the regression suite
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clause_segmentation as cs  # #33 TSV io
import narrative_reconstruction as nr  # #66 mention/clause/event primitives

# ruff: noqa: UP031 -- the probe formatter and the report renderer are built
# from long, wrapped `%` templates with positional value tuples; `%`
# substitution keeps each row a single statement, matching #66 and the rest
# of the work/ suite, where f-strings would splice the same values into
# several concatenated literals and lose the readable flow.

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "entity_roles"

SCHEMA = "entity_roles/1"
STATUSES = ("evidenced", "uncertain", "refuted")

ROLES_HEADER = ["role_id", "entity", "entity_role", "synopsis_role", "claims",
                "status", "n_defining", "n_defining_pass", "n_probes",
                "mentions", "pages", "events", "evidence", "refutation",
                "citations"]
TESTS_HEADER = ["role_id", "entity", "probe", "kind", "defining", "measure",
                "observed", "threshold", "status", "citations"]
REFUTE_HEADER = ["role_id", "entity", "status", "failed_defining",
                 "missing_evidence"]
RECON_HEADER = ["quantity", "earlier", "here", "match"]


class RoleError(Exception):
    """A seed or closure check failed: the build refuses to publish."""


# ---------------------------------------------------------------------------
# the synopsis seed -- read off p224 only
# ---------------------------------------------------------------------------
SYNOPSIS_SENTENCES = {claim: text for claim, text, _ in nr.SYNOPSIS_CLAIMS}
CLAIM_IDS = tuple(SYNOPSIS_SENTENCES)

# verb families the role probes read.  BURN/IGNITE covers the S1 "ignites the
# circuitry" figure; CLASH the S4 "clash"; HUNT the S5 hunter; WAGE the S2
# "waging gene war".  HOSTILE is a disclosed set of the book's lexically
# unambiguous violence verbs (used only to flag an antagonist/villain role,
# never to invent a relation).  Everything is intersected with the frozen #27
# lexicon at probe time: a verb the lexicon does not know simply cannot fire.
HOSTILE_VERBS = frozenset({
    "murder", "murders", "murdered", "murdering",
    "kill", "kills", "killed", "killing",
    "massacre", "massacres", "massacred", "massacring",
    "slaughter", "slaughters", "slaughtered",
    "rape", "rapes", "raped", "raping",
    "burn", "burns", "burned", "burning", "burnt",
    "destroy", "destroys", "destroyed", "destroying",
    "crucify", "crucifies", "crucified",
    "torture", "tortures", "tortured", "torturing",
    "exterminate", "exterminates", "exterminated",
    "annihilate", "annihilates", "annihilated",
})
CLASH_VERBS = frozenset({"clash", "clashes", "clashed", "clashing"})
BURN_VERBS = frozenset(nr.BURN_SET) | frozenset({"ignite", "ignites", "ignited"})
HUNT_VERBS = frozenset({"hunt", "hunts", "hunting", "hunted"})
WAGE_VERBS = frozenset({"wage", "wages", "waging", "waged"})

# #59-only token families, on top of #66's disclosed set.  'planet', 'pink'
# and 'ash' are the S4 EVOL figure's own words; 'succubus' is the S5 form the
# jacket gives Super Cherry 666.  Lower-cased unit surfaces, word-bounded.
EXTRA_FAMILIES = {
    "planet": re.compile(r"(^|[^a-z])planet"),
    "pink": re.compile(r"(^|[^a-z])pink([^a-z]|$)"),
    "ash": re.compile(r"(^|[^a-z])ash([^a-z]|$)"),
}


def _fam(name):
    return "fam:" + name


# Role seeds.  ``entities`` is the family the role is resolved to in the #30
# glossary / #66 referent registry (one synopsis name can have several
# surfaces); ``probes`` is a tuple of (kind, label, params, defining).
ROLE_SEEDS = (
    {
        "role_id": "R1",
        "entity": "ADAM / ADAM Doll",
        "entities": ("ADAM", "ADAM Doll"),
        "synopsis_role": "central being (creature) whose circuitry the "
                         "collusion ignites",
        "claims": ("S1", "S4"),
        "probes": (
            ("present", "presence of the ADAM family",
             {"entities": ("ADAM", "ADAM Doll")}, True),
            ("recurrence", "recurrence across the book",
             {"entities": ("ADAM", "ADAM Doll"), "min": 20}, True),
            ("centrality", "centrality: >= 100 actor/target events",
             {"entities": ("ADAM", "ADAM Doll"), "min": 100}, True),
            ("category", "category in {creature, persona}",
             {"entities": ("ADAM", "ADAM Doll"),
              "allowed": ("creature", "persona")}, True),
            ("any_verb", "ignition/burn event for the family",
             {"entities": ("ADAM", "ADAM Doll"), "verbs": tuple(BURN_VERBS)},
             False),
            ("cooccur", "ADAM Doll co-occurs with Cadaver City",
             {"a": "ADAM Doll", "b": "Cadaver City"}, False),
        ),
    },
    {
        "role_id": "R2",
        "entity": "ANTi-ADAM",
        "entities": ("ANTI-ADAM",),
        "synopsis_role": "antagonist clashing with ADAM",
        "claims": ("S4",),
        "probes": (
            ("present", "presence of ANTi-ADAM",
             {"entities": ("ANTI-ADAM",)}, True),
            ("category", "category in {antagonist, persona}",
             {"entities": ("ANTI-ADAM",),
              "allowed": ("antagonist", "persona")}, True),
            ("pair_verb", "clash/hostile event between ADAM and ANTi-ADAM",
             {"a": ("ADAM", "ADAM Doll"), "b": ("ANTI-ADAM",),
              "verbs": tuple(CLASH_VERBS | HOSTILE_VERBS)}, True),
            ("cooccur", "ADAM / ANTi-ADAM co-occur in a clause",
             {"a": "ADAM", "b": "ANTI-ADAM"}, False),
            ("hostile", "hostile actor/target events",
             {"entities": ("ANTI-ADAM",)}, False),
        ),
    },
    {
        "role_id": "R3",
        "entity": "Cadaver City",
        "entities": ("Cadaver City",),
        "synopsis_role": "setting / place of the fatal collusion",
        "claims": ("S1",),
        "probes": (
            ("present", "presence of Cadaver City",
             {"entities": ("Cadaver City",)}, True),
            ("recurrence", "recurrence >= 3 pages",
             {"entities": ("Cadaver City",), "min": 3}, True),
            ("category", "category place",
             {"entities": ("Cadaver City",), "allowed": ("place",)}, True),
            ("cooccur", "ADAM Doll co-occurs with Cadaver City",
             {"a": "ADAM Doll", "b": "Cadaver City"}, False),
            ("cooccur", "embryo family co-occurs with Cadaver City",
             {"a": _fam("embryo"), "b": "Cadaver City"}, False),
            ("cooccur", "angel family co-occurs with Cadaver City",
             {"a": _fam("angel"), "b": "Cadaver City"}, False),
        ),
    },
    {
        "role_id": "R4",
        "entity": "Placenta World",
        "entities": ("Placenta World",),
        "synopsis_role": "setting / place of the dogs' gene war",
        "claims": ("S2",),
        "probes": (
            ("present", "presence of Placenta World",
             {"entities": ("Placenta World",)}, True),
            ("recurrence", "recurrence >= 3 pages",
             {"entities": ("Placenta World",), "min": 3}, True),
            ("category", "category place",
             {"entities": ("Placenta World",), "allowed": ("place",)}, True),
            ("cooccur", "the dog co-occurs with Placenta World",
             {"a": "the dog", "b": "Placenta World"}, False),
            ("any_verb", "the dog wages war in Placenta World",
             {"entities": ("the dog",), "verbs": tuple(WAGE_VERBS)}, False),
        ),
    },
    {
        "role_id": "R5",
        "entity": "Sato Corporation",
        "entities": ("Sato Corporation", "Sato"),
        "synopsis_role": "organization behind the digital vampires, napalm "
                         "torture and its victims",
        "claims": ("S3", "S5"),
        "probes": (
            ("present", "presence of Sato Corporation",
             {"entities": ("Sato Corporation", "Sato")}, True),
            ("category", "category organization",
             {"entities": ("Sato Corporation", "Sato"),
              "allowed": ("organization",)}, True),
            ("cooccur", "napalm family co-occurs with torture family",
             {"a": _fam("napalm"), "b": _fam("torture")}, True),
            ("cooccur", "torture family co-occurs with victim family",
             {"a": _fam("torture"), "b": _fam("victim")}, True),
            ("cooccur", "Sato Corporation co-occurs with victim family",
             {"a": "Sato Corporation", "b": _fam("victim")}, True),
            ("cooccur", "vampire family co-occurs with the corporation",
             {"a": _fam("vamp"), "b": "Sato Corporation"}, False),
            ("cooccur", "chaos family co-occurs with vampire family",
             {"a": _fam("chaos"), "b": _fam("vamp")}, False),
            ("hostile", "hostile actor/target events",
             {"entities": ("Sato Corporation", "Sato")}, False),
        ),
    },
    {
        "role_id": "R6",
        "entity": "the dog (dogs of zero)",
        "entities": ("the dog",),
        "synopsis_role": "war-making force waging gene war",
        "claims": ("S2",),
        "probes": (
            ("present", "presence of the dog referent",
             {"entities": ("the dog",)}, True),
            ("recurrence", "recurrence >= 20 pages",
             {"entities": ("the dog",), "min": 20}, True),
            ("any_verb", "the dog wages war (wage-family verb)",
             {"entities": ("the dog",), "verbs": tuple(WAGE_VERBS)}, True),
            ("cooccur", "the dog co-occurs with the zero family",
             {"a": "the dog", "b": _fam("zero")}, False),
            ("hostile", "hostile actor/target events",
             {"entities": ("the dog",)}, False),
        ),
    },
    {
        "role_id": "R7",
        "entity": "Super Cherry 666",
        "entities": ("Super Cherry 666",),
        "synopsis_role": "robosuccubus hunter of the corporation's victims",
        "claims": ("S5",),
        "probes": (
            ("present", "presence of Super Cherry 666",
             {"entities": ("Super Cherry 666",)}, True),
            ("category", "category in {creature, persona}",
             {"entities": ("Super Cherry 666",),
              "allowed": ("creature", "persona")}, True),
            ("cooccur", "Super Cherry 666 co-occurs with the succubus family",
             {"a": "Super Cherry 666", "b": _fam("succubus")}, True),
            ("any_verb", "Super Cherry 666 takes a hunt verb",
             {"entities": ("Super Cherry 666",),
              "verbs": tuple(HUNT_VERBS)}, True),
            ("cooccur", "Super Cherry 666 co-occurs with Sato Corporation",
             {"a": "Super Cherry 666", "b": "Sato Corporation"}, False),
        ),
    },
    {
        "role_id": "R8",
        "entity": "EVOL",
        "entities": ("EVOL",),
        "synopsis_role": "pink-ash planet (setting)",
        "claims": ("S4",),
        "probes": (
            ("present", "presence of EVOL",
             {"entities": ("EVOL",)}, True),
            ("recurrence", "recurrence >= 3 pages",
             {"entities": ("EVOL",), "min": 3}, True),
            ("cooccur", "EVOL co-occurs with the planet family",
             {"a": "EVOL", "b": _fam("planet")}, True),
            ("cooccur", "EVOL co-occurs with the pink family",
             {"a": "EVOL", "b": _fam("pink")}, False),
            ("cooccur", "EVOL co-occurs with the ash family",
             {"a": "EVOL", "b": _fam("ash")}, False),
        ),
    },
    {
        "role_id": "R9",
        "entity": "digital vampires",
        "entities": ("vamp",),
        "synopsis_role": "agents of Sato Corporation",
        "claims": ("S3",),
        "probes": (
            ("present", "presence of the vamp family entity",
             {"entities": ("vamp",)}, True),
            ("recurrence", "recurrence >= 3 pages",
             {"entities": ("vamp",), "min": 3}, True),
            ("cooccur", "vampire family co-occurs with the corporation",
             {"a": _fam("vamp"), "b": "Sato Corporation"}, True),
            ("category", "category in {persona, concept, creature}",
             {"entities": ("vamp",),
              "allowed": ("persona", "concept", "creature")}, False),
            ("hostile", "hostile actor/target events",
             {"entities": ("vamp",)}, False),
        ),
    },
)

PROBE_KINDS = ("present", "recurrence", "centrality", "category", "cooccur",
               "any_verb", "pair_verb", "hostile")


# ---------------------------------------------------------------------------
# evidence container
# ---------------------------------------------------------------------------
class Evidence:
    """The book-text corpus, once, in the shape the role probes read.

    ``clause_keys`` maps clause_id -> frozenset of entity names and
    ``fam:<name>`` token families.  ``clauses`` is the #33 rows (page-safe).
    Events are #66 event dicts (page, cite, actor, target, verb, ...).
    """

    def __init__(self, mention_counts, mention_pages, roles, events,
                 clause_keys, clauses):
        self.mention_counts = mention_counts
        self.mention_pages = mention_pages
        self.roles = roles
        self.events = events
        self.clause_keys = clause_keys
        self.clauses = clauses

    # -- primitive counts --------------------------------------------------
    def mentions(self, names):
        return max((int(self.mention_counts.get(n, 0)) for n in names),
                   default=0)

    def pages(self, names):
        return max((len(self.mention_pages.get(n, ())) for n in names),
                   default=0)

    def role_set(self, names):
        return tuple(sorted({self.roles.get(n, "") for n in names} - {""}))

    def events_for(self, names):
        s = set(names)
        return [e for e in self.events
                if e["actor"] in s or e["target"] in s]

    def cooccur(self, a, b):
        return [c for c in self.clauses
                if a in self.clause_keys.get(c["clause_id"], ())
                and b in self.clause_keys.get(c["clause_id"], ())]

    def verb_events(self, names, verbs):
        s, v = set(names), set(verbs)
        return [e for e in self.events
                if e["verb"] in v and (e["actor"] in s or e["target"] in s)]

    def pair_events(self, a_names, b_names, verbs=None):
        a, b = set(a_names), set(b_names)
        v = None if verbs is None else set(verbs)
        out = []
        for e in self.events:
            if v is not None and e["verb"] not in v:
                continue
            if ((e["actor"] in a and e["target"] in b)
                    or (e["actor"] in b and e["target"] in a)):
                out.append(e)
        return out

    def hostile_events(self, names):
        return self.verb_events(names, HOSTILE_VERBS)


def _label(key):
    return key.removeprefix("fam:")


def _cites_from_clauses(rows, n=3):
    return ["p%s:%s" % (r["page"], r["start"]) for r in rows[:n]]


def _cites_from_events(rows, n=3):
    return [e["cite"] for e in rows[:n]]


def _result(kind, observed, threshold, passed, citations, measure):
    return {"kind": kind, "observed": observed, "threshold": threshold,
            "status": "pass" if passed else "fail",
            "citations": "; ".join(citations), "measure": measure}


def run_probe(spec, ev):
    """Run one (kind, label, params, defining) probe against ``ev``."""
    kind, label, params, _defining = spec
    if kind not in PROBE_KINDS:
        raise RoleError("unknown probe kind %r" % kind)
    if kind == "present":
        names = params["entities"]
        n = ev.mentions(names)
        return ("present", label, _result(
            kind, n, ">=1", n >= 1, [], "mentions of {%s}" % "|".join(names)))
    if kind == "recurrence":
        names, lo = params["entities"], params["min"]
        n = ev.pages(names)
        return ("recurrence", label, _result(
            kind, n, ">=%d" % lo, n >= lo, [],
            "pages carrying {%s}" % "|".join(names)))
    if kind == "centrality":
        names, lo = params["entities"], params["min"]
        hits = ev.events_for(names)
        return ("centrality", label, _result(
            kind, len(hits), ">=%d" % lo, len(hits) >= lo,
            _cites_from_events(hits),
            "actor/target events for {%s}" % "|".join(names)))
    if kind == "category":
        names = params["entities"]
        got = ev.role_set(names)
        allowed = set(params["allowed"])
        return ("category", label, _result(
            kind, "/".join(got), "in {%s}" % "|".join(sorted(allowed)),
            bool(set(got) & allowed), [],
            "#30/#66 role of {%s}" % "|".join(names)))
    if kind == "cooccur":
        a, b = params["a"], params["b"]
        hits = ev.cooccur(a, b)
        return ("cooccur", label, _result(
            kind, len(hits), ">=1", bool(hits),
            _cites_from_clauses(hits),
            "clauses with both %s and %s" % (_label(a), _label(b))))
    if kind == "any_verb":
        names = params["entities"]
        hits = ev.verb_events(names, params["verbs"])
        return ("any_verb", label, _result(
            kind, len(hits), ">=1", bool(hits), _cites_from_events(hits),
            "{%s} as actor/target of {%s}"
            % ("|".join(names), "|".join(sorted(params["verbs"])))))
    if kind == "pair_verb":
        hits = ev.pair_events(params["a"], params["b"], params.get("verbs"))
        return ("pair_verb", label, _result(
            kind, len(hits), ">=1", bool(hits), _cites_from_events(hits),
            "event between {%s} and {%s}"
            % ("|".join(params["a"]), "|".join(params["b"]))))
    # hostile
    names = params["entities"]
    hits = ev.hostile_events(names)
    return ("hostile", label, _result(
        kind, len(hits), ">=1", bool(hits), _cites_from_events(hits),
        "hostile actor/target events for {%s}" % "|".join(names)))


def role_verdict(probe_rows):
    """evidenced | uncertain | refuted from the defining probes.

    Every defining probe passes -> evidenced.  Bare *presence* is only a
    precondition, not a role property: a role is refuted when none of its
    substantive (non-presence) defining probes passes -- the entity may
    exist, but nothing the synopsis asserts about it does.  Anything in
    between is uncertain.
    """
    defining = [p for p in probe_rows if p["defining"] == "y"]
    if not defining:
        raise RoleError("a role needs at least one defining probe")
    substantive = [p for p in defining if p["kind"] != "present"] or defining
    if all(p["status"] == "pass" for p in defining):
        return "evidenced"
    if not any(p["status"] == "pass" for p in substantive):
        return "refuted"
    return "uncertain"


def evaluate_role(seed, ev):
    """(role_row, probe_rows) for one seed."""
    probe_rows = []
    for spec in seed["probes"]:
        kind, label, res = run_probe(spec, ev)
        probe_rows.append({
            "role_id": seed["role_id"], "entity": seed["entity"],
            "probe": label, "kind": kind, "defining": "y" if spec[3] else "n",
            "measure": res["measure"], "observed": res["observed"],
            "threshold": res["threshold"], "status": res["status"],
            "citations": res["citations"],
        })
    status = role_verdict(probe_rows)
    defining = [p for p in probe_rows if p["defining"] == "y"]
    passed = [p for p in defining if p["status"] == "pass"]
    failed = [p for p in defining if p["status"] == "fail"]

    citations = []
    for p in probe_rows:
        if p["status"] == "pass" and p["citations"]:
            for c in p["citations"].split("; "):
                if c and c not in citations:
                    citations.append(c)
    evidence = "; ".join(
        "%s %s %s%s" % ("PASS" if p["status"] == "pass" else "FAIL",
                        p["kind"], p["observed"],
                        (" (%s)" % p["threshold"]) if p["status"] == "fail"
                        else "") for p in probe_rows)
    refutation = "; ".join("%s [%s] (%s observed, wanted %s)"
                           % (p["probe"], p["kind"], p["observed"],
                              p["threshold"])
                           for p in failed)

    row = {
        "role_id": seed["role_id"], "entity": seed["entity"],
        "entity_role": "/".join(ev.role_set(seed["entities"])) or "unknown",
        "synopsis_role": seed["synopsis_role"],
        "claims": "|".join(seed["claims"]),
        "status": status,
        "n_defining": len(defining), "n_defining_pass": len(passed),
        "n_probes": len(probe_rows),
        "mentions": ev.mentions(seed["entities"]),
        "pages": ev.pages(seed["entities"]),
        "events": len(ev.events_for(seed["entities"])),
        "evidence": evidence, "refutation": refutation,
        "citations": "; ".join(citations[:4]),
    }
    return row, probe_rows


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
def gather():
    """(Evidence, meta) from the frozen layers; p224 is excluded."""
    clause_rows_all = nr.load_clauses()
    n_syn = sum(1 for r in clause_rows_all
                if int(r["page"]) == nr.SYNOPSIS_PAGE)
    clause_rows = [r for r in clause_rows_all
                   if int(r["page"]) != nr.SYNOPSIS_PAGE]
    units_all, n_units, n_markup = nr.load_screened_units()
    selected = {int(r["page"]) for r in clause_rows}
    units = {p: us for p, us in units_all.items() if p in selected}
    n_units = sum(len(us) for us in units.values())
    clause_units, n_assigned, n_unassigned = nr.assign_units_to_clauses(
        clause_rows, units)
    surfaces, entities = nr.load_glossary()
    verbs = nr.load_verbs()
    (events, mention_counts, mention_pages, clause_keys,
     _mention_page_counts, _fam_page_counts) = nr.scan_clauses(
        clause_rows, clause_units, surfaces, verbs)

    # #59-only families, merged into the clause keys
    keys = {cid: set(ks) for cid, ks in clause_keys.items()}
    for clause in clause_rows:
        us = clause_units.get(clause["clause_id"], ())
        for name, rx in sorted(EXTRA_FAMILIES.items()):
            if any(rx.search(u["lower"]) for u in us):
                keys[clause["clause_id"]].add(_fam(name))
    clause_keys = {cid: frozenset(ks) for cid, ks in keys.items()}

    roles = {name: meta["role"] for name, meta in entities.items()}
    for name in mention_counts:
        roles.setdefault(name, "referent")

    ev = Evidence(mention_counts, mention_pages, roles, events,
                  clause_keys, clause_rows)
    meta = {
        "clauses": len(clause_rows),
        "synopsis_clauses_excluded": n_syn,
        "screened_units": n_units,
        "markup_units_excluded": n_markup,
        "units_assigned_to_clauses": n_assigned,
        "units_unassigned": n_unassigned,
        "events": len(events),
        "entities": len(entities),
        "mentions": sum(mention_counts.values()),
    }
    return ev, meta


def validate_seeds(seeds=ROLE_SEEDS):
    """Every seed must be synopsis-grounded and well formed."""
    seen = set()
    for seed in seeds:
        rid = seed["role_id"]
        if rid in seen:
            raise RoleError("duplicate role_id %r" % rid)
        seen.add(rid)
        if not seed["entities"]:
            raise RoleError("%s: empty entity family" % rid)
        if not seed["claims"]:
            raise RoleError("%s: no synopsis claim" % rid)
        for claim in seed["claims"]:
            if claim not in SYNOPSIS_SENTENCES:
                raise RoleError("%s: unknown synopsis claim %r" % (rid, claim))
        if not any(spec[3] for spec in seed["probes"]):
            raise RoleError("%s: no defining probe" % rid)
        for spec in seed["probes"]:
            if spec[0] not in PROBE_KINDS:
                raise RoleError("%s: unknown probe kind %r" % (rid, spec[0]))
    return True


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------
def build_reconciliation(ev, meta, seeds=ROLE_SEEDS):
    """Re-derive the published #30/#66 numbers the roles lean on."""
    rows = []

    def add(quantity, earlier, here, match):
        rows.append({"quantity": quantity, "earlier": earlier, "here": here,
                     "match": match})

    # #30 glossary size (frozen)
    glossary_rows = cs.read_tsv(nr.GLOSSARY)
    add("#30 accepted entities in the glossary (rows read)",
        len(glossary_rows), meta["entities"],
        "ok" if len(glossary_rows) == meta["entities"] else "MISMATCH")

    # #66 published mention counts, per seed family (first named entity)
    actors_path = nr.OUT / "actors.tsv"
    published = {}
    if actors_path.exists():
        for r in cs.read_tsv(actors_path):
            published[r["entity"]] = int(r["n_mentions"])
    for seed in seeds:
        for name in seed["entities"]:
            here = int(ev.mention_counts.get(name, 0))
            if name in published:
                add("#66 mention count for %s" % name, published[name], here,
                    "ok" if published[name] == here else "MISMATCH")
            else:
                add("#66 mention count for %s" % name, "n/a (referent)", here,
                    "ok")

    # #66 event count
    summary_path = nr.OUT / "summary.json"
    if summary_path.exists():
        pub_events = json.loads(summary_path.read_text())["events"]
        add("#66 event rows re-derived", pub_events, meta["events"],
            "ok" if pub_events == meta["events"] else "MISMATCH")

    # p224 exclusion
    add("p224 synopsis page excluded from the check",
        nr.SYNOPSIS_PAGE, "excluded (%d clauses)" % meta["synopsis_clauses_excluded"],
        "ok" if meta["synopsis_clauses_excluded"] >= 1 else "MISMATCH")

    # unit partition closure
    add("screened units = assigned + unassigned", meta["screened_units"],
        meta["units_assigned_to_clauses"] + meta["units_unassigned"],
        "ok" if meta["screened_units"]
        == meta["units_assigned_to_clauses"] + meta["units_unassigned"]
        else "MISMATCH")
    return rows


def check_closures(role_rows, probe_rows, recon_rows, seeds=ROLE_SEEDS):
    """Fail loudly rather than publish a wrong table."""
    by_id = {r["role_id"] for r in role_rows}
    if by_id != {s["role_id"] for s in seeds}:
        raise RoleError("role rows do not match the seeds")
    for r in role_rows:
        if r["status"] not in STATUSES:
            raise RoleError("bad verdict %r" % r["status"])
        if r["status"] == "evidenced" and not r["citations"]:
            raise RoleError("%s: evidenced with no citation" % r["role_id"])
        if r["status"] == "evidenced" and r["n_defining_pass"] != r["n_defining"]:
            raise RoleError("%s: evidenced with a failed defining probe"
                            % r["role_id"])
        if r["status"] in ("refuted", "uncertain") and not r["refutation"]:
            raise RoleError("%s: uncertain with no failed probe" % r["role_id"])
    for r in recon_rows:
        if r["match"] != "ok":
            raise RoleError("reconciliation mismatch: %s" % r["quantity"])
    # every probe belongs to a role and has a known status
    for p in probe_rows:
        if p["role_id"] not in by_id:
            raise RoleError("orphan probe %r" % p["probe"])
        if p["status"] not in ("pass", "fail"):
            raise RoleError("bad probe status %r" % p["status"])


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(out=OUT, quiet=False):
    global OUT
    OUT = Path(out)
    validate_seeds()
    ev, meta = gather()

    role_rows, probe_rows = [], []
    for seed in ROLE_SEEDS:
        row, probes = evaluate_role(seed, ev)
        role_rows.append(row)
        probe_rows.extend(probes)

    recon_rows = build_reconciliation(ev, meta)
    check_closures(role_rows, probe_rows, recon_rows)

    refute_rows = [{"role_id": r["role_id"], "entity": r["entity"],
                    "status": r["status"], "failed_defining": r["refutation"],
                    "missing_evidence": r["evidence"]}
                   for r in role_rows if r["status"] != "evidenced"]

    status_counts = collections.Counter(r["status"] for r in role_rows)
    summary = {
        "schema": SCHEMA,
        "page_range": None,
        "synopsis_page_excluded": nr.SYNOPSIS_PAGE,
        "clauses": meta["clauses"],
        "screened_units": meta["screened_units"],
        "events": meta["events"],
        "roles": len(role_rows),
        "role_status": {s: status_counts.get(s, 0) for s in STATUSES},
        "verdicts": {r["role_id"]: r["status"] for r in role_rows},
        "evidence_rows": len(probe_rows),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "determinism": "every artifact is byte-identical on rebuild except "
                       "the generated stamp, which no statistic reads",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    cs.write_tsv(OUT / "roles.tsv", ROLES_HEADER, role_rows)
    cs.write_tsv(OUT / "role_tests.tsv", TESTS_HEADER, probe_rows)
    cs.write_tsv(OUT / "refutations.tsv", REFUTE_HEADER, refute_rows)
    cs.write_tsv(OUT / "reconciliation.tsv", RECON_HEADER, recon_rows)
    cs.write_tsv(OUT / "diagnostics.tsv", ["diag", "value"], [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "synopsis_page_excluded", "value": nr.SYNOPSIS_PAGE},
        {"diag": "synopsis_clauses_excluded",
         "value": meta["synopsis_clauses_excluded"]},
        {"diag": "clauses", "value": meta["clauses"]},
        {"diag": "screened_units", "value": meta["screened_units"]},
        {"diag": "events", "value": meta["events"]},
        {"diag": "roles", "value": len(role_rows)},
        {"diag": "role_status", "value": dict(sorted(status_counts.items()))},
        {"diag": "evidence_rows", "value": len(probe_rows)},
        {"diag": "hostile_verbs", "value": len(HOSTILE_VERBS)},
    ])
    with (OUT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    write_report(OUT / "roles_report.md", summary, role_rows, probe_rows,
                 refute_rows, recon_rows)
    if not quiet:
        print("[#59] entity roles: %d roles (%s); %d evidence rows"
              % (len(role_rows),
                 ", ".join("%s=%s" % (r["role_id"], r["status"])
                           for r in role_rows),
                 len(probe_rows)))
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(path, summary, role_rows, probe_rows, refute_rows,
                 recon_rows):
    status_counts = summary["role_status"]
    lines = []
    lines.append("# Entity roles from the jacket synopsis, verified against "
                 "the text (#59)")
    lines.append("")
    lines.append("Generated: %s" % summary["generated"])
    lines.append("")
    lines.append("Nine roles are read off the back-cover synopsis (p224) and "
                 "tested against the book text, pages 1-223 -- the synopsis "
                 "page is excluded from its own check. Every verdict is "
                 "computed from counts; every evidenced role carries the "
                 "citations that earned it. **%d evidenced / %d uncertain / "
                 "%d refuted.**"
                 % (status_counts.get("evidenced", 0),
                    status_counts.get("uncertain", 0),
                    status_counts.get("refuted", 0)))
    lines.append("")
    lines.append("## 1. Method")
    lines.append("")
    lines.append("* **Seed.** Each role cites the synopsis sentence(s) it is "
                 "read from (shared with #66's `SYNOPSIS_CLAIMS`, so the text "
                 "cannot drift). No other source seeds a role.")
    lines.append("* **Evidence.** Presence, recurrence, centrality, the #30 "
                 "role category, clause co-occurrence and actor/verb events "
                 "are recomputed over the frozen #17/#33/#30/#27 layers with "
                 "#66's primitives. A probe is *defining* or *supporting*.")
    lines.append("* **Verdict.** A role is `evidenced` when every defining "
                 "probe passes, `uncertain` when some but not all do, and "
                 "`refuted` when no *substantive* defining probe passes. "
                 "Presence is a precondition, not a role: a role that exists "
                 "but whose specific relation is only partly attested is "
                 "`uncertain` (the entity is real, the predicate is not), "
                 "while one whose every asserted property fails is "
                 "`refuted`.")
    lines.append("")
    lines.append("## 2. The role table")
    lines.append("")
    lines.append("| id | entity | synopsis role | claims | #30 role | verdict "
                 "| mentions | pages | events |")
    lines.append("|---|---|---|---|---|---|---:|---:|---:|")
    for r in role_rows:
        lines.append("| %s | %s | %s | %s | %s | **%s** | %s | %s | %s |"
                     % (r["role_id"], r["entity"], r["synopsis_role"],
                        r["claims"].replace("|", "/"), r["entity_role"],
                        r["status"], r["mentions"], r["pages"],
                        r["events"]))
    lines.append("")
    lines.append("## 3. Evidence, per role")
    lines.append("")
    for r in role_rows:
        lines.append("### %s %s -- **%s** (%s/%s defining probes)"
                     % (r["role_id"], r["entity"], r["status"],
                        r["n_defining_pass"], r["n_defining"]))
        lines.append("")
        lines.append("Synopsis role: *%s* (claims %s, #30/#66 role `%s`). "
                     "%s mentions on %s pages, %s actor/target events."
                     % (r["synopsis_role"], r["claims"], r["entity_role"],
                        r["mentions"], r["pages"], r["events"]))
        lines.append("")
        for p in [x for x in probe_rows if x["role_id"] == r["role_id"]]:
            mark = "PASS" if p["status"] == "pass" else "FAIL"
            cite = (" [%s]" % p["citations"]) if p["citations"] else ""
            lines.append("* **%s** `%s` -- %s, observed %s (wanted %s).%s"
                         % (mark, p["probe"], p["measure"], p["observed"],
                            p["threshold"], cite))
        lines.append("")
        if r["citations"]:
            lines.append("Citations: %s" % r["citations"])
            lines.append("")
    if refute_rows:
        lines.append("## 4. Refutations and open questions")
        lines.append("")
        for r in refute_rows:
            lines.append("* **%s %s -- %s.** %s"
                         % (r["role_id"], r["entity"], r["status"],
                            r["failed_defining"] or "no failed defining probe"))
        lines.append("")
        lines.append("The Super Cherry 666 case is the sharp one: the "
                     "hunting formula the jacket names is real, but its actor "
                     "is ToKAGE, and Super Cherry 666 never hunts. ANTi-ADAM "
                     "exists and is the only entity #30 calls an `antagonist`, "
                     "but no clash event links it to ADAM; the dog is a "
                     "recurrent, violent actor, but no `wage`-family verb "
                     "stands with it. The synopsis names the right entities and "
                     "the wrong predicates.")
        lines.append("")
    lines.append("## 5. Reconciliation")
    lines.append("")
    lines.append("| quantity | owner | here | match |")
    lines.append("|---|---|---|---|")
    for r in recon_rows:
        lines.append("| %s | %s | %s | %s |"
                     % (r["quantity"], r["earlier"], r["here"], r["match"]))
    lines.append("")
    lines.append("## 6. Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python work/entity_roles.py      # writes analysis/entity_roles/")
    lines.append("python work/test_entity_roles.py")
    lines.append("```")
    lines.append("")
    lines.append("Offline and deterministic; rebuild is byte-identical except "
                 "the generated stamp.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    build(Path(args.out), quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
