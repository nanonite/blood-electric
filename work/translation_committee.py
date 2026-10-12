#!/usr/bin/env python3
"""#64 Multi-agent translation committee with disagreement log -> analysis/committee/

Task (plan E8.3, docs/chainlink-plan.md #64): run a **committee** over the
#63 glosses -- a generator / critic / comparer loop -- and log the
disagreement, so that disagreement localises the passages that are
*genuinely ambiguous* rather than the passages where a reading is merely
defective.

The five prompt strategies of #63 are the committee's **agents**.  They are
not interchangeable samples of one process: each was given a different
contract (keep the operator channel verbatim / drop it / verbalise it / hold
the #30 names), so two agents can differ because they were told to differ,
because one of them broke a rule, or because the passage itself admits more
than one compliant reading.  Only the third kind is ambiguity, and telling
the three apart is what this module exists to do.

The loop, run over every clause of #63's sampled set:

    1  GENERATOR   replay #63's pinned generation (`resolve_glosses`) over
                   the same stratified sample.  The glosses are the agents'
                   candidate readings; the replay is cross-checked against
                   the published `glosses.tsv`, so a pin that no longer
                   produces what #63 published fails the build rather than
                   shipping a reading the artifacts do not support.
    2  CRITIC      re-judge every reading with the four deterministic checks,
                   recomputed here from the frozen artifacts -- not read off
                   #63's report.  The committee's verdicts are then checked
                   against #63's published `checks.tsv` (and the replayed
                   verdict against `glosses.tsv`): a drift between what the
                   pins produce and what #63 published is a loud build error,
                   because the two were produced by the same deterministic
                   functions from the same frozen inputs and cannot honestly
                   disagree.
    3  COMPARER    for each clause, compare every pair of agents' readings
                   and localise the divergence to surfaces: each operator
                   (kept / dropped / added), each #30 entity (carried /
                   dropped), each unresolved residue surface (kept marked /
                   smoothed), and the content words (present in one reading
                   only).  A pair-level verdict row is logged too.

What each localised disagreement is
----------------------------------
Every row of the disagreement log is classified, and the classification is
the load-bearing output -- a raw divergence count cannot distinguish a
defective reading from an ambiguous passage.  A row is classified by what
the two readings did *on that surface*, never by a reading's verdict
elsewhere: a keep-strategy that kept `<<` as told and a drop-strategy that
dropped it as told have diverged by instruction, and charging the
keep-strategy with a `/` it added on another row would name a rule this
surface did not break.

  * `rule_violation`      the rule that governs this surface (operator /
                          entity / residue) is breached by exactly one
                          reading's own treatment of it, or by both.  The
                          divergence involves a defective reading; the
                          ambiguity is *apparent*, and the responsible agent
                          is named.
  * `contract_divergence` both readings treated this surface as their own
                          instructions required, and the difference is
                          exactly what the differing instructions require --
                          a `verbatim, in order` strategy prints the operator
                          channel, a `dropped by instruction` strategy
                          empties it.  The divergence is by instruction, not
                          by text.
  * `genuine_ambiguity`   both readings pass every check that applies to
                          them, and no instruction explains the difference.
                          The passage admits more than one compliant reading
                          and no rule chooses between them.  This is the
                          signal the task asks the disagreement to localise.
  * `defect_unattributed` a participant failed some check, but no check
                          governs this surface, so the divergence cannot be
                          counted as evidence about the passage.  Recorded
                          rather than dropped: silently collapsing it into
                          `genuine_ambiguity` would overstate the ambiguity,
                          and collapsing it into `rule_violation` would name
                          a check that does not govern the surface.

`genuine_ambiguity` is therefore *earned* twice: both readings must be
fully compliant, and the difference must not be one their instructions
required.  In this corpus the genuine surfaces are content-word
divergences among compliant readings -- `suck` versus `draw in` on clause
10, the added `crucified` on 1005, the `l`/`I` of the OCR-misread first
word on 2789 -- and they are what the map points at.  Clause 1003 is the
control that shows the taxonomy works: only one of its five readings
passes, so every one of its divergences is a defect or a contract, and the
map reports **no** genuine ambiguity for it.

What the module actually establishes
------------------------------------
Not the glosses (one dated capture, published by #63) and not the agreement
score (Jaccard, published by #63 as `agreement.tsv`).  It is the
**classification of every localised disagreement**: which surfaces the
committee's agents disagree on, which of those disagreements survive the
removal of defective readings and instructed divergence, and therefore
which sampled passages are genuinely ambiguous *by the corpus's own
rules* -- the four checks are the workspace's codified standard of a
compliant reading, so "no rule or instruction explains the difference" is
a measurement, not an impression.

Guardrails this module answers for (#76's audit)
------------------------------------------------
* **Circularity**: the critic grades a pinned replay and the frozen
  artifacts; it does not read #63's *verdicts* as inputs, it checks them
  as outputs.  The comparer never prefers a reading -- it records each
  agent's measured state on each surface.
* **Hallucinated coherence**: the map reports states, not resolutions; no
  reading is edited, ranked or preferred, and a genuine-ambiguity row
  carries both readings' evidence verbatim.
* **Single-method certainty**: five agents, all ten pairs per clause
  published, every row classified by a published rule.
* **Determinism**: no model is called.  The generator phase replays #63's
  pins through #63's own `resolve_glosses`; everything else is a pure
  function of the frozen artifacts.  The build is byte-identical on
  rebuild, apart from the report's generation stamp (disclosed in
  `summary.json`, as in #63).

Not done, and said plainly: an *invented* proper noun (the second failure
mode of `entity_fidelity`) is not re-derived here -- it surfaces as a
content-word divergence classified `defect_unattributed`, which is
conservative (a defective reading's divergence is never counted as
ambiguity) but less precise than an entity-surface row would be; and the
committee judges #63's five-clause sample, not the whole book, because
those are the readings that exist.

Reproduce::

    python work/translation_committee.py          # pinned, offline
    python work/test_translation_committee.py

Inputs: #63's published tables (`analysis/glossing/`) and, through #63's
loaders, the same frozen artifacts #63 read (#17 tokens, #33 clauses,
#30 glossary, #62 gloss tables).  No external corpus is consulted or
acquired.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clause_segmentation as cs            # TSV io
import glossing_pipeline as gp             # #63: the agents, the checks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "committee"

GLOSSING = ROOT / "analysis" / "glossing"
PUBLISHED_GLOSSES = GLOSSING / "glosses.tsv"
PUBLISHED_CHECKS = GLOSSING / "checks.tsv"
PUBLISHED_AGREE = GLOSSING / "agreement.tsv"
PUBLISHED_SAMPLE = GLOSSING / "sample.tsv"

SCHEMA = "committee/1"

#: The four checks, named once so the summary and the log cannot drift
#: from the list `check_gloss` actually runs.
CHECKS = ("operator_fidelity", "entity_fidelity", "residue_discipline",
          "frame_preservation")

#: The disagreement classes, in the order the classification rule tests
#: them (see `classify_disagreement`).  Published so a reader can tell
#: which kind of row they are looking at without re-deriving the rule.
CLASS_RULE_VIOLATION = "rule_violation"
CLASS_CONTRACT = "contract_divergence"
CLASS_GENUINE = "genuine_ambiguity"
CLASS_DEFECT = "defect_unattributed"
CLASS_ORDER = (CLASS_RULE_VIOLATION, CLASS_CONTRACT, CLASS_GENUINE,
               CLASS_DEFECT)

#: Surface kind -> the check that governs it.  A disagreement on a governed
#: surface is a rule matter before it is anything else; `content` is
#: governed by no check (content-word choice is what the strategies are
#: *for*), which is why content divergence between two compliant readings
#: is the signature of genuine ambiguity.
KIND_GOVERNING = {
    "operator": "operator_fidelity",
    "entity": "entity_fidelity",
    "residue": "residue_discipline",
    "content": None,
}

#: Log row order within a clause: the governed surfaces first (a rule
#: violation outranks an ambiguity claim), then content, then the
#: pair-level verdict row.
KIND_ORDER = ("operator", "entity", "residue", "content", "verdict")


class CommitteeError(Exception):
    """The committee's inputs are missing, or a cross-check failed."""


# ---------------------------------------------------------------------------
# 1. the published #63 artifacts the committee consumes
# ---------------------------------------------------------------------------

def load_published():
    """The #63 tables, as the committee's inputs.

    Read with #33's `read_tsv` rather than `csv.DictReader` for the same
    reason #62 reads its own tables that way: gloss texts exceed the
    `csv` module's default field limit.
    """
    missing = [str(p) for p in (PUBLISHED_GLOSSES, PUBLISHED_CHECKS,
                                PUBLISHED_AGREE, PUBLISHED_SAMPLE)
               if not p.exists()]
    if missing:
        raise CommitteeError(
            "#63's published artifacts are missing (%s); build "
            "work/glossing_pipeline.py first -- the committee judges "
            "#63's readings, it does not generate its own"
            % ", ".join(missing))
    return (cs.read_tsv(PUBLISHED_GLOSSES), cs.read_tsv(PUBLISHED_CHECKS),
            cs.read_tsv(PUBLISHED_AGREE), cs.read_tsv(PUBLISHED_SAMPLE))


# ---------------------------------------------------------------------------
# 2. what each reading says: the facts the comparer measures
# ---------------------------------------------------------------------------

def residue_kept(surface, gloss):
    """Per-surface mirror of `check_residue_discipline`'s kept-test.

    The check returns one verdict for the whole clause; the comparer needs
    each surface's state separately, so the predicate is stated once more
    here and a suite test pins the two together: for every reading, the
    check passes if and only if every surface is kept.
    """
    low = surface.lower()
    if low in gloss.lower():
        return True
    return any(low in w.lower() for w in re.findall(r"[A-Za-z\-]+", gloss))


def op_state(kept, dropped, added):
    """A reading's measured state on one operator surface.

    Counted, never set-membership: a clause with two `=` and a reading that
    keeps one has *partially* kept the channel, and set membership would
    report that as agreement.  The count is always shown, so a state reads
    the same whether it is the only component or one of several -- `kept
    dropped` next to `kept 1 dropped 1` is a distinction the log should
    not have to draw.
    """
    if not (kept or dropped or added):
        return "(absent)"
    bits = []
    if kept:
        bits.append("kept %d" % kept)
    if dropped:
        bits.append("dropped %d" % dropped)
    if added:
        bits.append("added %d" % added)
    return " ".join(bits)


def reading_facts(clause_id, page, stratum, strategy, ops_in, entities,
                  residue, gloss, verdict, check_rows):
    """Everything the comparer measures about one reading.

    A pure function of its arguments: the operator channel is read off #17's
    stream (`ops_in`, as #63's checks read it), the entity constraints are
    the #30 rows for the entities the clause names (as #63's prompts
    carried them), and `check_rows` are the critic's own rows for this
    reading, so a reading's verdict and its per-check results are one
    structure the classifier can read rather than two that can disagree.
    """
    ops_out = gp.operator_subsequence(gloss)
    in_c, out_c = collections.Counter(ops_in), collections.Counter(ops_out)
    symbols = {}
    for sym in set(in_c) | set(out_c):
        kept = min(in_c[sym], out_c[sym])
        state = (kept, in_c[sym] - kept, out_c[sym] - kept)
        if any(state):
            symbols[sym] = state
    checks = {c["check"]: {"pass": c["pass"], "applies": c["applies"]}
              for c in check_rows}
    failed = sorted(c["check"] for c in check_rows
                    if c["applies"] and not c["pass"])
    content = gp.gloss_tokens(gloss)
    return {
        "clause_id": clause_id,
        "page": page,
        "stratum": stratum,
        "strategy": strategy,
        "gloss": gloss,
        "verdict": verdict,
        "failed": failed,
        "checks": checks,
        "ops_in": list(ops_in),
        "ops_out": ops_out,
        "op_symbols": symbols,
        "entities_in": [e["entity"] for e in entities],
        "entity_carried": {e["entity"]: gp.entity_present(e["entity"], gloss)
                           for e in entities},
        "residue_in": list(residue),
        "residue_kept": {s: residue_kept(s, gloss) for s in residue},
        "content_words": content,
        "n_content_words": len(content),
    }


# ---------------------------------------------------------------------------
# 3. the classifier
# ---------------------------------------------------------------------------

def operator_compliance(reading, sym):
    """Is this reading's treatment of operator `sym` what its contract owes?

    A `verbatim, in order` strategy owes the clause's channel exactly: for
    each symbol, kept-as-many-as-the-clause-has and nothing added.  A
    `dropped`/`verbalised` strategy owes an empty channel: nothing kept,
    nothing added.  Evaluated per symbol, so a reading that kept the
    channel and added a persona bracket's `|` is compliant on `<<` and in
    breach on `|` -- the two surfaces are logged separately, and each row
    is classified by what happened on it.
    """
    _kept, dropped, added = reading["op_symbols"].get(sym, (0, 0, 0))
    if gp.STRATEGY_OWES[reading["strategy"]]["operators"] == \
            "verbatim, in order":
        return dropped == 0 and added == 0
    return _kept == 0 and added == 0


def channel_compliance(reading):
    """Is the reading's whole operator channel what its contract owes?

    The per-symbol test cannot see a reorder: two readings can keep every
    symbol in a different order and agree on every per-symbol count.  This
    is the whole-sequence form of `operator_compliance`, for the reorder
    row.
    """
    if gp.STRATEGY_OWES[reading["strategy"]]["operators"] == \
            "verbatim, in order":
        return reading["ops_out"] == reading["ops_in"]
    return not reading["ops_out"]


def classify_disagreement(kind, a, b, compliant_a=None, compliant_b=None):
    """What a localised disagreement on `kind` between readings `a`/`b` is.

    Returns ``(classification, governing_check, responsible)``.  The rule,
    in order, and the reason for the order:

      1. A governed surface (operator / entity / residue) is classified by
         each reading's own treatment of it, passed in as `compliant_a` /
         `compliant_b`: a breach by exactly one is a `rule_violation`
         naming it, a breach by both names both.  A reading that broke the
         rule *governing this surface* cannot testify about the passage,
         whatever its verdict on other surfaces was.
      2. Two readings that both treated the surface as instructed and were
         told to differ (`verbatim, in order` versus `dropped by
         instruction`) diverge by instruction: `contract_divergence`.
         Two readings that both treated the surface as instructed, both
         pass every check that applies to them, and differ on a surface
         no instruction explains are the case the task asks for:
         `genuine_ambiguity`.
      3. Otherwise a participant is defective and no check governs this
         surface: `defect_unattributed`.  The row is kept rather than
         dropped or quietly promoted, because it is a real measured
         divergence that the classification rule refuses to call ambiguity.
    """
    gov = KIND_GOVERNING[kind]
    if gov is not None:
        if compliant_a is not compliant_b:
            return (CLASS_RULE_VIOLATION, gov,
                    b["strategy"] if compliant_a else a["strategy"])
        if compliant_a is False:
            return (CLASS_RULE_VIOLATION, gov,
                    a["strategy"] + "+" + b["strategy"])
        if kind == "operator" and \
                gp.STRATEGY_OWES[a["strategy"]]["operators"] != \
                gp.STRATEGY_OWES[b["strategy"]]["operators"]:
            return CLASS_CONTRACT, gov, ""
        if a["verdict"] == "pass" and b["verdict"] == "pass":
            return CLASS_GENUINE, gov, ""
        responsible = "+".join(
            s for s, f in ((a["strategy"], a), (b["strategy"], b))
            if f["verdict"] != "pass")
        return CLASS_DEFECT, gov, responsible
    if a["verdict"] == "pass" and b["verdict"] == "pass":
        return CLASS_GENUINE, "none", ""
    responsible = "+".join(
        s for s, f in ((a["strategy"], a), (b["strategy"], b))
        if f["verdict"] != "pass")
    return CLASS_DEFECT, "none", responsible


# ---------------------------------------------------------------------------
# 4. the comparer
# ---------------------------------------------------------------------------

def compare_readings(a, b):
    """Every localised disagreement between two readings of one clause.

    One row per surface the two readings treat differently, classified by
    `classify_disagreement`.  A surface both readings treat the same way
    produces no row: the log records divergence, not agreement -- agreement
    is #63's `agreement.tsv`, published there for every pair including the
    unanimous ones.
    """
    rows = []

    def add(kind, surface, state_a, state_b, force=None,
            compliant_a=None, compliant_b=None):
        if force is not None:
            cls, gov, responsible = force
        else:
            cls, gov, responsible = classify_disagreement(
                kind, a, b, compliant_a, compliant_b)
        rows.append({
            "clause_id": a["clause_id"],
            "surface_kind": kind,
            "surface": surface,
            "agent_a": a["strategy"],
            "state_a": state_a,
            "agent_b": b["strategy"],
            "state_b": state_b,
            "classification": cls,
            "governing_check": gov,
            "responsible": responsible,
        })

    # operator surfaces: each symbol the clause carries or either reading
    # adds, with each reading's kept/dropped/added counts.
    for sym in sorted(set(a["op_symbols"]) | set(b["op_symbols"])):
        sa = op_state(*a["op_symbols"].get(sym, (0, 0, 0)))
        sb = op_state(*b["op_symbols"].get(sym, (0, 0, 0)))
        if sa != sb:
            add("operator", sym, sa, sb,
                compliant_a=operator_compliance(a, sym),
                compliant_b=operator_compliance(b, sym))
    # Same multiset in a different order: no per-symbol state differs, so
    # without this row the reorder would be invisible in the log.
    if a["ops_out"] != b["ops_out"] and \
            collections.Counter(a["ops_out"]) == \
            collections.Counter(b["ops_out"]):
        add("operator", "channel order",
            ",".join(a["ops_out"]) or "(empty)",
            ",".join(b["ops_out"]) or "(empty)",
            compliant_a=channel_compliance(a),
            compliant_b=channel_compliance(b))

    # entity surfaces: the #30 names this clause's prompts actually carried.
    for e in a["entities_in"]:
        ca, cb = a["entity_carried"][e], b["entity_carried"][e]
        if ca != cb:
            add("entity", e, "carried" if ca else "dropped",
                "carried" if cb else "dropped",
                compliant_a=ca, compliant_b=cb)

    # residue surfaces: the #62 unresolved tokens, kept marked or smoothed.
    for s in a["residue_in"]:
        ka, kb = a["residue_kept"][s], b["residue_kept"][s]
        if ka != kb:
            add("residue", s, "kept" if ka else "smoothed",
                "kept" if kb else "smoothed",
                compliant_a=ka, compliant_b=kb)

    # content words: the divergence no check governs.
    only_a = sorted(a["content_words"] - b["content_words"])
    only_b = sorted(b["content_words"] - a["content_words"])
    if only_a or only_b:
        add("content", "content words",
            ",".join(only_a) or "(none)",
            ",".join(only_b) or "(none)")

    # the pair-level row: one reading acceptable, the other not.  This is
    # the committee's bottom line for the pair, and it is what a raw
    # divergence count would otherwise have to stand in for.
    if a["verdict"] != b["verdict"]:
        governing = ",".join(sorted(set(a["failed"]) ^ set(b["failed"])))
        add("verdict", "verdict", a["verdict"], b["verdict"], force=(
            CLASS_RULE_VIOLATION, governing or "verdict",
            b["strategy"] if a["verdict"] == "pass" else a["strategy"]))
    return rows


def pair_metrics(a, b):
    """The comparer's per-pair summary row, for `pairs.tsv`."""
    rows = compare_readings(a, b)
    by_class = collections.Counter(r["classification"] for r in rows)
    union = a["content_words"] | b["content_words"]
    inter = a["content_words"] & b["content_words"]
    return {
        "clause_id": a["clause_id"],
        "agent_a": a["strategy"],
        "agent_b": b["strategy"],
        "content_jaccard": round(len(inter) / len(union), 4) if union else 1.0,
        "n_rows": len(rows),
        "n_rule_violation": by_class[CLASS_RULE_VIOLATION],
        "n_contract_divergence": by_class[CLASS_CONTRACT],
        "n_genuine_ambiguity": by_class[CLASS_GENUINE],
        "n_defect_unattributed": by_class[CLASS_DEFECT],
        "both_compliant": a["verdict"] == "pass" and b["verdict"] == "pass",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 5. the loop: generator -> critic -> comparer
# ---------------------------------------------------------------------------

def run_committee():
    """Run the three phases and return ``(sample, facts)``.

    The generator phase replays #63's pins over #63's sample; the critic
    re-judges each reading from the frozen artifacts; the comparer
    localises and classifies every pairwise divergence.  Every phase's
    output is cross-checked against #63's published tables, so the
    committee stands on the published record and fails loudly when it
    stops agreeing with it.
    """
    pub_glosses, pub_checks, _pub_agree, pub_sample = load_published()
    clauses = gp.load_clauses()
    glossary = gp.load_glossary()
    opmap = gp.load_operator_token_map()
    residue_by_clause = gp.load_clause_residue()
    ocr_by_clause = gp.load_clause_ocr(gp.load_ocr_class())

    sample = gp.pick_sample(clauses, opmap, glossary, residue_by_clause)
    published_strata = {r["clause_id"]: r["stratum"] for r in pub_sample}
    if sample != published_strata:
        raise CommitteeError(
            "the re-drawn sample is not #63's published sample (%s vs %s); "
            "the committee judges the readings that exist, so #63 must be "
            "rebuilt -- or its sample pinned -- first"
            % (sorted(sample.items()), sorted(published_strata.items())))

    # -- 1. generator: replay the pins -----------------------------------
    replayed, _prompts, _replayed_checks = gp.resolve_glosses(
        sample, opmap, glossary, residue_by_clause, ocr_by_clause,
        gp.STRATEGY_IDS, clauses=clauses)
    pub_by_key = {(r["clause_id"], r["strategy"]): r for r in pub_glosses}
    for r in replayed:
        key = (r["clause_id"], r["strategy"])
        pub = pub_by_key.get(key)
        if pub is None:
            raise CommitteeError(
                "no published #63 row for %s/%s; the replay produced a "
                "reading #63 did not publish" % key)
        for field in ("gloss", "verdict", "operators_in", "operators_out"):
            if r[field] != pub[field]:
                raise CommitteeError(
                    "the replay of %s/%s disagrees with #63's published "
                    "%s (%r vs %r); the pins no longer produce the "
                    "published artifact" % (key[0], key[1], field,
                                            r[field], pub[field]))
        sha = hashlib.sha256(r["gloss"].encode("utf-8")).hexdigest()
        if sha != pub["response_sha256"]:
            raise CommitteeError(
                "the replayed gloss for %s/%s does not hash to #63's "
                "published response_sha256" % key)

    # -- 2. critic: re-judge every reading from the frozen artifacts ----
    # The checks are deterministic pure functions of the frozen artifacts,
    # so the committee's critic is #63's checks re-run here; the judgement
    # is then checked against #63's published rows, which is what makes
    # this a re-derivation rather than a restatement.
    pub_check_rows = {(r["clause_id"], r["strategy"], r["check"]): r
                      for r in pub_checks}
    facts = {}
    for cid, stratum in sorted(sample.items(), key=lambda kv: int(kv[0])):
        clause = clauses[cid]
        ops = opmap.get(cid, [])
        names = {n for n in glossary if n in clause["original"].lower()}
        ents = gp.entity_constraints(glossary, names)
        res = gp.residue_surfaces(residue_by_clause.get(cid, []))
        for sid in gp.STRATEGY_IDS:
            gloss = pub_by_key[(cid, sid)]["gloss"]
            verdict, check_rows = gp.check_gloss(clause, ops, ents, res,
                                                 sid, gloss)
            for c in check_rows:
                pub = pub_check_rows.get((cid, sid, c["check"]))
                if pub is None:
                    raise CommitteeError(
                        "#63 published no row for %s/%s/%s"
                        % (cid, sid, c["check"]))
                if (c["applies"], c["pass"]) != \
                        (pub["applies"] == "True", pub["pass"] == "True"):
                    raise CommitteeError(
                        "the critic's %s for %s/%s disagrees with #63's "
                        "published row (applies %s/%s, pass %s/%s)"
                        % (c["check"], cid, sid, c["applies"],
                           pub["applies"], c["pass"], pub["pass"]))
            if verdict != pub_by_key[(cid, sid)]["verdict"]:
                raise CommitteeError(
                    "the critic's verdict for %s/%s disagrees with #63's "
                    "published verdict" % (cid, sid))
            facts[(cid, sid)] = reading_facts(
                cid, clause["page"], stratum, sid, ops, ents, res, gloss,
                verdict, check_rows)
    return sample, facts


# ---------------------------------------------------------------------------
# 6. the disagreement log and the pairs table
# ---------------------------------------------------------------------------

READINGS_HEADER = [
    "clause_id", "page", "stratum", "agent", "strategy_title", "verdict",
    "n_applicable", "n_passed", "failed_checks", "operators_in",
    "operators_out", "operators_dropped", "operators_added", "entities_in",
    "entities_carried", "residue_in", "residue_kept", "n_content_words",
    "gloss",
]
PAIRS_HEADER = [
    "clause_id", "agent_a", "agent_b", "content_jaccard", "n_rows",
    "n_rule_violation", "n_contract_divergence", "n_genuine_ambiguity",
    "n_defect_unattributed", "both_compliant",
]
LOG_HEADER = [
    "clause_id", "surface_kind", "surface", "agent_a", "state_a",
    "agent_b", "state_b", "classification", "governing_check",
    "responsible",
]


def _join(seq):
    return ",".join(seq)


def _multiset_diff(in_c, out_c):
    """The symbols one reading dropped and the ones it added, as lists."""
    dropped, added = [], []
    for sym in sorted(set(in_c) | set(out_c)):
        d = in_c[sym] - min(in_c[sym], out_c[sym])
        a = out_c[sym] - min(in_c[sym], out_c[sym])
        dropped.extend([sym] * d)
        added.extend([sym] * a)
    return dropped, added


def build_log(facts, sample):
    """Every pair, every clause -> (pairs rows, log rows).

    Pairs are enumerated over the agents that actually produced a reading
    for the clause, in #63's strategy order, so the pair list is a pure
    function of the published record.
    """
    by_clause = collections.defaultdict(dict)
    for (cid, sid), f in facts.items():
        by_clause[cid][sid] = f
    pairs, log = [], []
    for cid in sorted(by_clause, key=int):
        agents = [s for s in gp.STRATEGY_IDS if s in by_clause[cid]]
        for sa, sb in combinations(agents, 2):
            m = pair_metrics(by_clause[cid][sa], by_clause[cid][sb])
            pairs.append({k: v for k, v in m.items() if k != "rows"})
            log.extend(m["rows"])
    log.sort(key=lambda r: (int(r["clause_id"]),
                            KIND_ORDER.index(r["surface_kind"]),
                            r["surface"], r["agent_a"], r["agent_b"]))
    return pairs, log


# ---------------------------------------------------------------------------
# 7. the disagreement map
# ---------------------------------------------------------------------------

#: Cap on a state quoted inside the map's compact table.  The full state is
#: always in `disagreement_log.tsv`; the cap is a rendering choice, and it
#: is disclosed here rather than left as a silent truncation.
MAP_STATE_CAP = 12


def _cap(words):
    """The first `MAP_STATE_CAP` words of a comma-joined state, marked."""
    bits = words.split(",")
    if len(bits) <= MAP_STATE_CAP:
        return words
    return ",".join(bits[:MAP_STATE_CAP]) + ", ... (+%d more)" \
        % (len(bits) - MAP_STATE_CAP)


def write_map(path, sample, facts, pairs, log, summary, generated):
    """The deliverable: the disagreement map, with its method and audit."""
    by_clause = collections.defaultdict(list)
    for r in log:
        by_clause[r["clause_id"]].append(r)
    genuine = [r for r in log if r["classification"] == CLASS_GENUINE]
    genuine_clauses = sorted({r["clause_id"] for r in genuine}, key=int)
    clauses = gp.load_clauses()

    L = []
    a = L.append
    a("# Multi-agent translation committee -- disagreement map (#64)")
    a("")
    a("Generated: %s" % generated)
    a("")
    a("## What this is")
    a("")
    a("Five agents -- #63's prompt strategies -- each produced one reading")
    a("of each of the %d sampled clauses (%d readings).  The committee ran"
      % (summary["clauses"], summary["readings"]))
    a("the generator / critic / comparer loop over them: it replayed the")
    a("pinned generation, re-judged every reading with the four")
    a("deterministic checks recomputed from the frozen artifacts, and")
    a("compared every pair of readings surface by surface.  This map is")
    a("the comparer's output: every localised disagreement, classified.")
    a("")
    a("**The load-bearing output is the classification, not the raw")
    a("divergence count.**  Two agents can differ because one of them")
    a("broke a rule, because they were told to differ, or because the")
    a("passage itself admits more than one compliant reading.  Only the")
    a("third kind is ambiguity.")
    a("")
    a("## The four kinds of disagreement")
    a("")
    a("| class | rows | what it means |")
    a("|---|---|---|")
    a("| `%s` | %d | the rule that governs this surface is breached by"
      % (CLASS_RULE_VIOLATION,
         summary["rows_by_class"][CLASS_RULE_VIOLATION]))
    a("  exactly one reading's own treatment of it, or by both: the")
    a("  divergence involves a defective reading, and the responsible")
    a("  agent is named |")
    a("| `%s` | %d | both readings treated the surface as their own"
      % (CLASS_CONTRACT, summary["rows_by_class"][CLASS_CONTRACT]))
    a("  instructions required, and the difference is exactly what the")
    a("  differing instructions require (a strategy told to keep the")
    a("  operator channel versus one told to drop it): divergence by")
    a("  instruction, not by text |")
    a("| `%s` | %d | both readings pass every check that applies to them,"
      % (CLASS_GENUINE, summary["rows_by_class"][CLASS_GENUINE]))
    a("  and no instruction explains the difference: the passage admits")
    a("  more than one compliant reading. **This is the signal the task")
    a("  asks the disagreement to localise.** |")
    a("| `%s` | %d | a participant failed some check, but no check governs"
      % (CLASS_DEFECT, summary["rows_by_class"][CLASS_DEFECT]))
    a("  this surface, so the divergence is not usable as evidence about")
    a("  the passage.  Recorded rather than dropped or quietly promoted:")
    a("  it is a real measured divergence the rule refuses to call")
    a("  ambiguity |")
    a("")
    a("## Headline")
    a("")
    a("**%d of the %d sampled passages** contain at least one surface on"
      % (len(genuine_clauses), summary["clauses"]))
    a("which two fully compliant readings disagree -- the passages that are")
    a("genuinely ambiguous by the corpus's own rules:")
    a("")
    for cid in genuine_clauses:
        rows = [r for r in genuine if r["clause_id"] == cid]
        kinds = collections.Counter(r["surface_kind"] for r in rows)
        a("* clause **%s** -- %d genuine surface(s): %s"
          % (cid, len(rows),
             ", ".join("%s (%d)" % (k, kinds[k]) for k in sorted(kinds))))
    a("")
    other = [cid for cid in sorted(sample, key=int)
             if cid not in genuine_clauses]
    if other:
        a("Clauses whose every divergence is a defect or a contract:")
        a("")
        for cid in other:
            n = sum(1 for r in by_clause[cid])
            a("* clause **%s** -- %d localised disagreement(s), none of"
              " them genuine" % (cid, n))
        a("")
    a("## The map")
    a("")
    a("Every genuine-ambiguity surface in the sample, one row each.")
    a("States are quoted at %d words per side here; the complete states"
      % MAP_STATE_CAP)
    a("are in `disagreement_log.tsv`.")
    a("")
    if genuine:
        a("| clause | kind | surface | `%s` state | `%s` state | governing "
          "check |" % ("agent_a", "agent_b"))
        a("|---|---|---|---|---|---|")
        for r in genuine:
            a("| %s | %s | `%s` | %s: %s | %s: %s | %s |"
              % (r["clause_id"], r["surface_kind"], r["surface"],
                 r["agent_a"], _cap(r["state_a"]),
                 r["agent_b"], _cap(r["state_b"]), r["governing_check"]))
        a("")
    else:
        a("(none -- no two compliant readings disagree anywhere in the")
        a("sample; the committee found no genuine ambiguity to localise)")
        a("")
    a("## The passages, in full")
    a("")
    for cid in sorted(sample, key=int):
        clause = clauses[cid]
        rows = by_clause[cid]
        a("### clause %s (p. %s, `%s`)"
          % (cid, clause["page"], sample[cid]))
        a("")
        a("```text")
        a("src | %s" % clause["original"].replace("\n", " "))
        a("```")
        a("")
        for sid in gp.STRATEGY_IDS:
            if (cid, sid) not in facts:
                continue
            f = facts[(cid, sid)]
            a("* **`%s`** -- %s -- failed: %s"
              % (sid, f["verdict"], ",".join(f["failed"]) or "(none)"))
            a("  > %s" % f["gloss"])
        a("")
        if rows:
            cls_count = collections.Counter(r["classification"] for r in rows)
            a("Localised disagreements: %d -- %s."
              % (len(rows), ", ".join(
                  "%s %d" % (k, cls_count[k]) for k in CLASS_ORDER
                  if cls_count[k])))
            a("")
        genuine_rows = [r for r in rows
                        if r["classification"] == CLASS_GENUINE]
        if genuine_rows:
            a("Genuine ambiguity (%d surface(s)):" % len(genuine_rows))
            a("")
            for r in genuine_rows:
                a("* `%s` `%s` -- `%s`: %s ; `%s`: %s -- governing check:"
                  " %s"
                  % (r["surface_kind"], r["surface"], r["agent_a"],
                     r["state_a"], r["agent_b"], r["state_b"],
                     r["governing_check"]))
            a("")
    a("## What the disagreement is not")
    a("")
    a("The other three classes, with the corpus's own examples:")
    a("")
    for cls, why in (
        (CLASS_RULE_VIOLATION,
         "a defective reading.  Example: clause 1003's `scene_summary` "
         "drops `BABEL=TV` and fails `entity_fidelity`, so its "
         "disagreement with `register_paraphrase` on that entity is "
         "apparent ambiguity, not a property of the passage."),
        (CLASS_CONTRACT,
         "instructed divergence.  Example: clause 101's `<< >>` is kept by "
         "the three `verbatim, in order` strategies and dropped by the two "
         "drop-strategies, all five compliant -- the channel differs "
         "because the prompts differ."),
        (CLASS_DEFECT,
         "a defective reading diverging on a surface no failed check "
         "governs.  Example: clause 1003's `literal_operator` fails "
         "`operator_fidelity` (its bracketed persona constraints add "
         "thirteen `|`), and its content words diverge from the passing "
         "`register_paraphrase` for reasons no check on that surface "
         "explains -- the divergence is real but it is not evidence about "
         "the text."),
    ):
        n = summary["rows_by_class"][cls]
        a("* `%s` (%d row(s)) -- %s" % (cls, n, why))
    a("")
    a("## Method")
    a("")
    a("Three phases over each sampled clause, no model called:")
    a("")
    a("1. **Generator** -- #63's pinned replay (`resolve_glosses` over the")
    a("   same stratified sample), cross-checked against the published")
    a("   `glosses.tsv` (gloss text, verdict, operator columns and the")
    a("   response sha256 all re-derived).")
    a("2. **Critic** -- the four deterministic checks recomputed here from")
    a("   the frozen artifacts, then checked against #63's published")
    a("   `checks.tsv`: same functions, same frozen inputs, so a")
    a("   disagreement between the critic and the published record is a")
    a("   build error, not a finding.")
    a("3. **Comparer** -- every pair of a clause's readings, compared on")
    a("   each operator symbol (kept/dropped/added, counted), each #30")
    a("   entity (carried/dropped), each #62 residue surface")
    a("   (kept/smoothed) and the content-word sets, plus a pair-level")
    a("   verdict row.")
    a("")
    a("A row is `genuine_ambiguity` only if both readings pass every check")
    a("that applies to them *and* no instruction explains the difference")
    a("(operator surfaces on strategies with differing operator")
    a("obligations are `contract_divergence` instead).  That is what makes")
    a("the headline a measurement: the checks are the workspace's codified")
    a("standard of a compliant reading.")
    a("")
    a("## Guardrails (L9.8 audit)")
    a("")
    a("* **Circularity**: the critic grades a pinned replay and the frozen")
    a("  artifacts; it checks #63's verdicts as outputs rather than")
    a("  reading them as inputs.  The comparer never grades a reading --")
    a("  it records each agent's measured state on each surface, and the")
    a("  classifier never prefers one.")
    a("* **Hallucinated coherence**: the map reports states, not")
    a("  resolutions; no reading is edited, ranked or preferred, and a")
    a("  genuine-ambiguity row carries both readings' evidence.")
    a("* **Single-method certainty**: five agents, all ten pairs per")
    a("  clause published, every row classified by a published rule.")
    a("* **Determinism**: the build is byte-identical on rebuild apart from")
    a("  the generation stamp, for the same reason #63's is -- nothing")
    a("  moves with the sampler.")
    a("")
    a("## Reproducing")
    a("")
    a("```bash")
    a("python work/translation_committee.py      # pinned, offline")
    a("python work/test_translation_committee.py")
    a("```")
    a("")
    a("Prerequisites are #63's frozen artifacts (`analysis/glossing/`) and,")
    a("through #63's loaders, #17/#33/#30/#62.  Inputs are read-only; the")
    a("committee writes only `analysis/committee/`.")
    a("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 8. build
# ---------------------------------------------------------------------------

DIAG_HEADER = ["diag", "value"]


def build(out=OUT, quiet=False):
    """The committee build: run the loop, publish the log, the map, summary."""
    OUT = Path(out)
    sample, facts = run_committee()

    pairs, log = build_log(facts, sample)
    by_class = collections.Counter(r["classification"] for r in log)
    by_kind = collections.Counter(r["surface_kind"] for r in log)
    genuine_clauses = sorted({r["clause_id"] for r in log
                              if r["classification"] == CLASS_GENUINE},
                             key=int)
    fully_compliant = sum(1 for f in facts.values() if f["verdict"] == "pass")
    compliant_pairs = sum(1 for p in pairs if p["both_compliant"])

    readings_rows = []
    for (cid, sid) in sorted(facts, key=lambda k: (int(k[0]), k[1])):
        f = facts[(cid, sid)]
        dropped, added = _multiset_diff(collections.Counter(f["ops_in"]),
                                        collections.Counter(f["ops_out"]))
        readings_rows.append({
            "clause_id": cid,
            "page": f["page"],
            "stratum": f["stratum"],
            "agent": sid,
            "strategy_title": gp._STRATEGY_BY_ID[sid]["title"],
            "verdict": f["verdict"],
            "n_applicable": sum(1 for c in f["checks"].values()
                                 if c["applies"]),
            "n_passed": sum(1 for c in f["checks"].values()
                            if c["applies"] and c["pass"]),
            "failed_checks": ",".join(f["failed"]) or "(none)",
            "operators_in": ",".join(f["ops_in"]) or "(none)",
            "operators_out": ",".join(f["ops_out"]) or "(none)",
            "operators_dropped": ",".join(dropped) or "(none)",
            "operators_added": ",".join(added) or "(none)",
            "entities_in": ",".join(f["entities_in"]) or "(none)",
            "entities_carried": ",".join(
                e for e in f["entities_in"] if f["entity_carried"][e])
            or "(none)",
            "residue_in": ",".join(f["residue_in"]) or "(none)",
            "residue_kept": ",".join(
                s for s in f["residue_in"] if f["residue_kept"][s])
            or "(none)",
            "n_content_words": f["n_content_words"],
            "gloss": f["gloss"],
        })

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = {
        "schema": SCHEMA,
        "mode": "pinned replay of #63's published artifacts; no model call",
        "clauses": len(sample),
        "agents": len(gp.STRATEGY_IDS),
        "readings": len(facts),
        "pairs": len(pairs),
        "rows_total": len(log),
        "rows_by_class": {c: by_class[c] for c in CLASS_ORDER},
        "rows_by_kind": {k: by_kind[k] for k in KIND_ORDER if by_kind[k]},
        "clauses_with_disagreement": sum(
            1 for cid in sample
            if any(r["clause_id"] == cid for r in log)),
        "clauses_with_genuine_ambiguity": genuine_clauses,
        "clauses_without_genuine_ambiguity": [
            cid for cid in sorted(sample, key=int)
            if cid not in genuine_clauses],
        "genuine_surfaces": sum(1 for r in log
                                if r["classification"] == CLASS_GENUINE),
        "fully_compliant_readings": fully_compliant,
        "compliant_pairs": compliant_pairs,
        "generated": generated,
        "determinism": "the build replays #63's pins through #63's own "
                       "resolve_glosses and re-derives every number from "
                       "the frozen artifacts; byte-identical on rebuild "
                       "(every data file; only the generation stamp moves "
                       "with the clock)",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    cs.write_tsv(OUT / "readings.tsv", READINGS_HEADER, readings_rows)
    cs.write_tsv(OUT / "pairs.tsv", PAIRS_HEADER, pairs)
    cs.write_tsv(OUT / "disagreement_log.tsv", LOG_HEADER, log)
    with (OUT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    write_map(OUT / "disagreement_map.md", sample, facts, pairs, log,
              summary, generated)
    cs.write_tsv(OUT / "diagnostics.tsv", DIAG_HEADER, [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "mode", "value": summary["mode"]},
        {"diag": "clauses", "value": len(sample)},
        {"diag": "agents", "value": len(gp.STRATEGY_IDS)},
        {"diag": "readings", "value": len(facts)},
        {"diag": "pairs", "value": len(pairs)},
        {"diag": "rows_total", "value": len(log)},
    ] + [{"diag": "rows_%s" % c, "value": by_class[c]} for c in CLASS_ORDER] + [
        {"diag": "rows_%s" % k, "value": by_kind[k]}
        for k in KIND_ORDER if by_kind[k]] + [
        {"diag": "clauses_with_genuine_ambiguity",
         "value": ",".join(genuine_clauses) or "(none)"},
        {"diag": "clauses_without_genuine_ambiguity",
         "value": ",".join(summary["clauses_without_genuine_ambiguity"])
                   or "(none)"},
        {"diag": "fully_compliant_readings", "value": fully_compliant},
        {"diag": "compliant_pairs", "value": compliant_pairs},
    ])
    if not quiet:
        print("[#64] committee: %d clauses x %d agents = %d readings, "
              "%d pairs, %d disagreement rows (%d genuine)"
              % (len(sample), len(gp.STRATEGY_IDS), len(facts), len(pairs),
                 len(log), summary["genuine_surfaces"]))
    return summary


# ---------------------------------------------------------------------------
# 9. main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        build(Path(args.out), quiet=args.quiet)
    except CommitteeError as exc:
        print("translation committee failed: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
