#!/usr/bin/env python3
"""Design the human reader study (#75) -> analysis/reader_study/

Task (plan E9.6, docs/chainlink-plan.md #75; methods L9.7):

    *Have readers rate glosses for fidelity to the felt text; design the
     instrument and sampling.  DONE: study protocol + pilot results if
     available.*

The evaluation ladder ends with readers.  #62 published an interlinear
gloss, #63 five LLM strategies over a 5-clause stratified sample, #64
classified what the strategies disagree about, #68 ranked readings by
compression and #74 by round-trip fidelity.  Every one of those is a
*proxy* for the thing the task names -- **fidelity to the felt text** --
and every one of them can be gamed by the very machinery that built the
glosses.  #63's own report says so: the model is not deterministic, and
there is no ground-truth translation to check a reading against.

This module does **not run the study**.  It *designs* it, and it designs it
far enough that the next step is a recruitment email rather than another
design discussion:

* **the instrument** -- 7 items per stimulation (3 rating scales + 2
  anchored forced choices + 2 planted traps), with the attention check
  that catches a reader who is rating prose fluency instead of feeling;
* **the sampling** -- a *census* of the gloss population over a stratified
  clause sample: 45 cells (5 clauses x [1 identity control + 1 #62 gloss +
  5 LLM strategies + 2 traps]), of which each block shows 6, with the LLM
  comparators rotated across blocks so no reading is systematically first
  and a reader cannot score a gloss without a same-clause comparator;
* **the analysis plan** -- pre-registered statistics, computed here on a
  *simulated* pilot (seed 75, seeded Gaussian reader abilities) so the
  pipeline, the artifact shapes and the significance thresholds are proven
  before a single human is recruited;
* **the pilot** -- deterministic synthetic ratings, clearly labelled as a
  simulation of the instrument's behaviour and **never** as evidence about
  readers.

## Why a simulated pilot exists at all

The study has not been run; there are no human ratings in this workspace.
But a study design that has never had its arithmetic exercised is a study
design that will fail on the first respondent.  So the pilot is generated
by a *disclosed generative model* -- per-item expectation tables, reader
leniency, a noise level and a position effect all drawn from seeded
Gaussians, ratings clipped to 0-7 and rounded to integers -- and the
module then analyses it with exactly the statistics the real run will use:
per-cell means with bootstrap CIs, Krippendorff's alpha for inter-reader
agreement (interval metric, read against a re-grouping null), a
within-clause contrast of the identity control against the other readings
with a control-label permutation test, the same readings correlated
against #74's published machine fidelity with a label permutation, and a
Kruskal-Wallis H with eta-squared.  **Every pilot number is a property of
the generative model, not a finding about readers.**  The report says so
in its first line, every artifact carries a `simulated` flag, and the
suite asserts that a reader of the report cannot mistake it for data.

## What the instrument is built against

The five L9.8 anti-patterns, from #76's checklist:

* **Hallucinated coherence** -- the instrument never asks "is this
  coherent?".  It asks whether the reader can *recover* the feeling of the
  clause from the gloss, with the original clause shown alongside.  A
  smooth reading that invents a feeling the clause does not have is
  supposed to score *low*, and item 5 (the invented-plot trap) is a
  planted case where a fluent gloss is unfaithful.
* **LLM circularity** -- no model generates or grades anything here.  The
  reader is the judge; the only LLM output in the frame is #63's *pinned*
  capture, which the study treats as a candidate and not as evidence.
* **Over-correction** -- item 6 (the over-corrected trap) is a gloss that
  repairs the derangement into ordinary English.  A reader who prefers it
  is measured, not corrected: the instrument records the preference as
  data about the reading's appeal, and the analysis separates appeal from
  fidelity.
* **Single-method certainty** -- 7 items, 4 statistical legs, 3 null
  models, per-stratum reporting and a published pre-registration file; no
  single number is the verdict.
* **Residue-dropping** -- the sampling frame is published in full
  (`frame.tsv`), traps counted and excluded by role rather than dropped,
  reader exclusions published with their reasons, and the per-stratum
  results never pooled into one verdict.

Anti-patterns in the design itself that this module guards: no wall-clock,
no network, fixed seed, byte-identical rebuild, and a build error rather
than a silent mismatch when an upstream artifact drifts.  Two build-time
controls have teeth rather than decoration: the identity control must top
its block on the primary item in every block (or the instrument is not
measuring fidelity and the build stops), and the attention check must be
able to flag a reader (or the traps are decorative).

What the artifacts are, in the order a reviewer should read them:

* ``protocol.md`` -- the pre-registration: the question, the population,
  the design, the instrument verbatim, the analysis plan, the stopping
  rule, and what the study cannot establish;
* ``instrument.tsv`` -- every item, its role, its anchors and its scale;
* ``frame.tsv`` -- the sampling frame: all 45 cells, drawn or not, with the
  role each plays and #74's published fidelity for its reading;
* ``blocks.tsv`` -- the counterbalanced packets, one block per clause;
* ``pilot_ratings.tsv`` / ``pilot_cell_summary.tsv`` / ``agreement.tsv`` /
  ``machine_human.tsv`` -- the simulated pilot and its analysis, every row
  flagged ``simulated = True``;
* ``reader_study_report.md`` -- the verification report, which leads with
  the statement that none of it is human data.

Reproduce::

    python work/reader_study.py            # writes analysis/reader_study/
    python work/test_reader_study.py       # regression suite
    python -m ruff check work/reader_study.py work/test_reader_study.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))

import frequency_zipf as fz       # #24 write_tsv / unescape_cell
import ngram_surprisal as ns      # #35 read_tsv (tab-split, not csv)

OUT = ROOT / "analysis" / "reader_study"
INTERLINEAR = ROOT / "analysis" / "interlinear" / "interlinear.tsv"
GLOSSES = ROOT / "analysis" / "glossing" / "glosses.tsv"
SAMPLE = ROOT / "analysis" / "glossing" / "sample.tsv"
FIDELITY = ROOT / "analysis" / "back_translation" / "fidelity_scores.tsv"

SCHEMA = "reader-study/1"
SEED = 75                       # issue number, the house convention for seeds
N_READERS = 12                  # pilot readers (synthetic, seed SEED)
N_PAIRS_PER_CLAUSE = 2          # LLM comparators drawn per clause
N_BOOT = 400                    # bootstrap resamples for the per-cell CIs
N_PERM = 2000                   # permutation draws for the contrast tests
ALPHA_DRAWS = 200               # draws for the alpha null distribution

# Published counts frozen by earlier issues; a drift is a build error.
PUBLISHED = {
    "n_gloss_rows": 25,          # #63 sampled glosses
    "n_strategies": 5,           # #63 prompt strategies
    "n_sample_clauses": 5,       # #63 stratified sample, one per stratum
    "n_interlinear_rows": 5614,  # #33/#62 clause count
}

read_tsv = ns.read_tsv
write_tsv = fz.write_tsv


def unescape(text: str) -> str:
    """Undo ``write_tsv``'s cell escaping, so a gloss reads back verbatim.

    ``frequency_zipf.write_tsv`` escapes backslash/tab/newline in every cell;
    reading one of its artifacts back with a tab-split reader therefore
    returns the escaped form.  This helper is the inverse, used wherever a
    text column is compared against the artifact it was written from.
    """
    return fz.unescape_cell(text)


# --------------------------------------------------------------------------
# 1. the instrument
# --------------------------------------------------------------------------
# Three rating scales plus two pairwise items and two planted traps, each
# in a fixed order.  The *order* of glosses inside a clause block is
# rotated by the sampler (see ``assign_blocks``), so "rated first" is not
# correlated with which reading a gloss is.

SCALE_ITEMS = (
    # key, prompt (as read by the participant), low anchor, high anchor
    ("felt_fidelity",
     ("How closely does this wording carry the feeling of the clause it "
      "translates?"),
     "0 = it carries nothing of the feeling",
     "7 = it carries the feeling exactly"),
    ("fidelity_confidence",
     ("How sure are you of that judgement?"),
     "0 = guessing",
     "7 = completely sure"),
    ("reading_appeal",
     ("How much would you want to keep reading text written this way?"),
     "0 = not at all",
     "7 = very much"),
)

# Each forced-choice item carries its option list verbatim, so the
# instrument's answer set is published rather than inferred from a prompt.
PAIRWISE_ITEMS = (
    ("pair_preference",
     ("Which of these two wordings carries the clause's feeling better?"),
     ("the first", "the second", "equally well")),
    ("pair_invented_plot",
     ("Does either wording commit to a concrete story event the clause only "
      "hints at?"),
     ("neither", "the first", "the second", "both")),
)

TRAP_ITEMS = (
    ("trap_overcorrected",
     ("This wording has smoothed the clause into ordinary English. Rate how "
      "faithful that smoothing is to the clause's oddity."),
     "0 = completely unfaithful to the oddity",
     "7 = fully faithful"),
    ("trap_invented",
     ("This wording adds a named character or event the clause does not "
      "contain. Rate how faithful that addition is."),
     "0 = completely unfaithful",
     "7 = fully faithful"),
)

ALL_SCALE_ITEMS = SCALE_ITEMS + TRAP_ITEMS
SCALE_KEYS = tuple(k for k, *_ in ALL_SCALE_ITEMS)
PAIRWISE_KEYS = tuple(k for k, *_ in PAIRWISE_ITEMS)
ITEM_KEYS = SCALE_KEYS + PAIRWISE_KEYS

# The item the study is *about*; every other item is a calibration or a
# guard, and the report ranks them as such.
PRIMARY_ITEM = "felt_fidelity"
APPEAL_ITEM = "reading_appeal"

# The difference function the study pre-registers for Krippendorff's alpha
# and the pilot computes with: a 0-7 rating scale carries equal intervals,
# so ``delta^2 = (c - k)^2`` keeps a 6-against-5 disagreement cheap and a
# 6-against-0 one expensive.  ``ordinal`` and ``nominal`` are implemented
# and published alongside it as robustness checks (see
# ``_delta_squared``), never as silent substitutes, and the protocol text,
# the pilot and the report all name *this* one -- the constant is the
# single source for all three so they cannot drift apart again.
PRIMARY_METRIC = "interval"

# Shown to the participant, verbatim, above the first block.
CONSENT_INSTRUCTIONS = (
    "You will be shown short passages from a literary work alongside "
    "wordings that try to carry what each passage feels like.\n"
    "There is no correct answer. Judge only the fit between the wording "
    "and the passage's feeling, not how good the writing is.\n"
    "Some wordings are deliberately too smooth or add things the passage "
    "does not say -- rating them low is the intended answer.\n"
    "Work at a steady pace; you may stop at any time and your partial "
    "answers are kept."
)

# Attention checks: planted cells whose accepted answer is known by
# construction.  The check names the *cell kind* it is about (a trap cell
# and a candidate cell can share a reading name, so the kind is what
# distinguishes them), the item it is read from, and the accepted value.
# ``ATTENTION_TOLERANCE`` is how far a reader may stray before the reader is
# flagged in ``diagnostics.tsv`` rather than silently dropped.
ATTENTION_CHECKS = (
    ("att_identity", "identity", "felt_fidelity", 7,
     "the clause itself must rate at the top of the fidelity scale"),
    ("att_overcorrected", "trap", "trap_overcorrected", 0,
     "a smoothing that repairs the derangement must rate at the bottom"),
)
ATTENTION_TOLERANCE = 1


# --------------------------------------------------------------------------
# 2. the sampling frame
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GlossPair:
    """One (clause, gloss) cell in the sampling frame.

    ``kind`` separates the role the cell plays, which is what the analysis
    groups on: an ``identity`` control (the clause verbatim, which must
    score maximally on fidelity if the instrument works), a ``gloss``
    (#62's literal reading), an ``llm`` reading (#63's pinned strategies),
    or a ``trap`` (a deliberately defective reading used to check that the
    instrument can fail a reading).
    """

    pair_id: str
    clause_id: str
    page: int
    stratum: str
    reading: str
    kind: str
    text: str
    original: str
    role: str                    # why this cell is in the frame
    stratum_order: int

    @property
    def is_identity(self) -> bool:
        return self.kind == "identity"


def load_interlinear() -> dict:
    rows = read_tsv(INTERLINEAR)
    if not rows:
        raise SystemExit(f"#75: no interlinear rows at {INTERLINEAR}")
    by_id: dict = {}
    for r in rows:
        cid = r["clause_id"]
        if cid in by_id:
            raise SystemExit(f"#75: duplicate clause_id {cid} in the interlinear")
        by_id[cid] = {"original": unescape(r["original"]),
                      "gloss": unescape(r["gloss"]),
                      "page": int(r["page"])}
    return by_id


def load_sample() -> list:
    rows = read_tsv(SAMPLE)
    if not rows:
        raise SystemExit(f"#75: no sample rows at {SAMPLE}")
    return [{"clause_id": r["clause_id"], "stratum": r["stratum"],
             "stratum_order": i} for i, r in enumerate(rows)]


def load_glosses() -> dict:
    """(clause_id -> strategy -> gloss row), from #63's pinned capture."""
    out: dict = {}
    for r in read_tsv(GLOSSES):
        out.setdefault(r["clause_id"], {})[r["strategy"]] = {
            "text": unescape(r["gloss"]),
            "verdict": r["verdict"],
            "n_passed": int(r["n_passed"]),
            "n_applicable": int(r["n_applicable"]),
            "checks_passed": r["checks_passed"],
        }
    return out


def load_fidelity() -> dict:
    """#74's published fidelity scores, keyed by reading name.

    Only the channels the report cites are kept; the value is a *label*
    carried by the frame so a reading's human rating can be cross-referenced
    against its machine-measured fidelity.  It is never an input to the
    simulated ratings.
    """
    out: dict = {}
    rows = read_tsv(FIDELITY) if FIDELITY.exists() else []
    for r in rows:
        out[r["reading"]] = {
            "aggregate_fidelity": float(r["aggregate_fidelity"]),
            "mean_operator_recall": float(r["mean_operator_recall"]),
            "mean_content_recall": float(r["mean_content_recall"]),
        }
    return out


# The strategies sampled for the human study, and the role each plays.
# #63's five strategies are all available; the study draws all five, which
# keeps the frame a *census* of the gloss population rather than a further
# subsample -- the sampling problem here is over clauses and comparators,
# not over strategies.
SAMPLED_STRATEGIES = ("literal_operator", "register_paraphrase",
                      "scene_summary", "plain_english", "entity_consistent")

STRATEGY_ROLE = {
    "literal_operator": "operator channel preserved, editorial asides added",
    "register_paraphrase": "highest #74 aggregate fidelity",
    "scene_summary": "condensed, operators dropped",
    "plain_english": "rewriting (the over-correction arm)",
    "entity_consistent": "reading-conditioned, operators annotated",
}


def build_frame(interlinear: dict, sample: list, glosses: dict) -> list:
    """The sampling frame: every (clause, gloss) cell the study can rate.

    One row per (clause, reading):

    * ``identity`` -- the clause itself.  The instrument's positive
      control: a reader who cannot rate the original at the ceiling of
      the fidelity scale is not measuring fidelity, and is flagged.
    * ``gloss`` -- #62's literal reading (the non-LLM baseline).
    * ``llm_<strategy>`` -- one row per #63 strategy over that clause.
    * ``trap_*`` -- the two planted defective readings (see
      ``TRAP_ITEMS``), which are *instrument items*, not candidate
      readings, and are excluded from every fidelity statistic.

    Cells are ordered by clause then by kind, so the frame is deterministic
    and the block rotation below has a defined input order.
    """
    cells: list = []
    for row in sample:
        cid = row["clause_id"]
        if cid not in interlinear:
            raise SystemExit(f"#75: sample clause {cid} absent from #62")
        original = interlinear[cid]["original"]
        page = interlinear[cid]["page"]
        stratum = row["stratum"]
        order = row["stratum_order"]

        cells.append(GlossPair(
            pair_id=f"{cid}/identity", clause_id=cid, page=page,
            stratum=stratum, reading="identity", kind="identity",
            text=original, original=original,
            role="positive control: the clause itself", stratum_order=order))

        cells.append(GlossPair(
            pair_id=f"{cid}/gloss", clause_id=cid, page=page, stratum=stratum,
            reading="interlinear_gloss", kind="gloss",
            text=interlinear[cid]["gloss"], original=original,
            role="#62's literal reading, no model involved",
            stratum_order=order))

        clause_glosses = glosses.get(cid, {})
        for strategy in SAMPLED_STRATEGIES:
            if strategy not in clause_glosses:
                raise SystemExit(
                    f"#75: clause {cid} has no gloss for strategy {strategy}")
            g = clause_glosses[strategy]
            cells.append(GlossPair(
                pair_id=f"{cid}/{strategy}", clause_id=cid, page=page,
                stratum=stratum, reading=f"llm_{strategy}", kind="llm",
                text=g["text"], original=original,
                role=STRATEGY_ROLE[strategy], stratum_order=order))

    # The traps are instrument items, one per clause, carrying #63's
    # *plain_english* reading -- the strategy #63 itself most often fails
    # on the operator-fidelity check, and the one #74 measures as the
    # least faithful overall.
    for row in sample:
        cid = row["clause_id"]
        text = glosses.get(cid, {}).get("plain_english", {}).get("text", "")
        cells.append(GlossPair(
            pair_id=f"{cid}/trap_overcorrected", clause_id=cid,
            page=interlinear[cid]["page"], stratum=row["stratum"],
            reading="llm_plain_english", kind="trap", text=text,
            original=interlinear[cid]["original"],
            role="trap: repaired into ordinary English (over-correction)",
            stratum_order=row["stratum_order"]))
        cells.append(GlossPair(
            pair_id=f"{cid}/trap_invented", clause_id=cid,
            page=interlinear[cid]["page"], stratum=row["stratum"],
            reading="llm_plain_english", kind="trap", text=text,
            original=interlinear[cid]["original"],
            role="trap: invents a plot event (hallucinated coherence)",
            stratum_order=row["stratum_order"]))

    return cells


def candidate_cells(frame: list) -> list:
    """The rateable *candidate* cells: everything except the traps.

    The traps are instrument items and never enter a fidelity statistic,
    so excluding them here is what keeps ``n_pairs`` honest: the frame
    reports them separately rather than dropping them from the count.
    """
    return [c for c in frame if c.kind != "trap"]


# --------------------------------------------------------------------------
# 3. counterbalancing: which cells each block shows, in which order
# --------------------------------------------------------------------------
# A clause block shows its comparators together so a reader cannot rate a
# gloss without the original beside it.  ``N_PAIRS_PER_CLAUSE`` is the
# number of *comparators* drawn per clause (beyond the identity control and
# the #62 gloss, which every block carries), and ``PAIR_OFFSETS`` is the
# rotation applied across clauses: clause i draws comparators starting at
# position ``(i * PAIR_OFFSETS) % n_candidates``, so the same LLM readings
# recur across blocks but never in the same within-block position, and a
# clause's block is not the first five rows of its stratum.
PAIR_OFFSETS = (0, 2, 4, 1, 3)


def assign_blocks(frame: list, n_pairs_per_clause: int) -> list:
    """One block per sampled clause: what it shows, in what order.

    Every block carries the identity control and the #62 gloss (the
    comparators that make a rating interpretable), a rotated draw of
    ``n_pairs_per_clause`` LLM readings, **and the clause's two trap
    cells**.  The traps are shown because they are instrument items: a
    reader who is never asked to judge an over-corrected reading cannot be
    checked for judging fidelity rather than appeal, and an attention check
    with no data is not a check.  They are excluded from every fidelity
    statistic downstream, by ``kind``.

    The rotation is the counterbalancing: it is a deterministic function of
    the clause's position, so the whole design is reproducible from the
    frame alone, and "rated first" is never confounded with which reading a
    cell carries.
    """
    blocks: list = []
    candidates = candidate_cells(frame)
    traps = [c for c in frame if c.kind == "trap"]
    for i, cid in enumerate(sorted({c.clause_id for c in candidates},
                                   key=lambda c: next(
                                       x.stratum_order for x in candidates
                                       if x.clause_id == c))):
        cells = [c for c in candidates if c.clause_id == cid]
        identity = [c for c in cells if c.is_identity]
        gloss = [c for c in cells if c.kind == "gloss"]
        llms = [c for c in cells if c.kind == "llm"]
        clause_traps = [c for c in traps if c.clause_id == cid]
        offset = PAIR_OFFSETS[i % len(PAIR_OFFSETS)]
        drawn = [llms[(offset + j) % len(llms)]
                 for j in range(n_pairs_per_clause)]
        shown = identity + gloss + drawn + clause_traps
        blocks.append({
            "clause_id": cid,
            "stratum": cells[0].stratum,
            "page": cells[0].page,
            "original": cells[0].original,
            "block_index": i,
            "offset": offset,
            "pair_ids": [c.pair_id for c in shown],
            "cells": shown,
        })
    return blocks


def sampled_pairs(blocks: list) -> list:
    """The (clause, gloss) pairs the design actually rates, flattened."""
    out: list = []
    for b in blocks:
        out.extend(b["cells"])
    return out


def block_schedule(blocks: list) -> list:
    """Item order within a block, counterbalanced.

    Within a block the cells are shown in a fixed rotation of the base
    order (identity, gloss, comparators), so "shown first" rotates across
    blocks instead of always being the positive control.  Pairwise items
    then compare adjacent pairs of the shown cells, which is why the
    schedule is a separate step from the cell list.
    """
    schedule: list = []
    for b in blocks:
        cells = b["cells"]
        rot = b["block_index"] % len(cells)
        order = cells[rot:] + cells[:rot]
        pairs = [(order[i], order[i + 1])
                 for i in range(len(order) - 1)]
        schedule.append({
            "clause_id": b["clause_id"],
            "stratum": b["stratum"],
            "order": [c.pair_id for c in order],
            "pairwise_pairs": [(a.pair_id, b_.pair_id) for a, b_ in pairs],
        })
    return schedule


# --------------------------------------------------------------------------
# 4. the simulated pilot
# --------------------------------------------------------------------------
# Every rating below is generated.  The generative model is disclosed in
# the report and pinned by SEED; nothing here is a human judgement.  A
# reader ability is drawn per reader, a grader bias per item, and each
# rating is a logistic squashing of a linear predictor clipped to the
# scale, which is enough to exercise agreement statistics, CIs, permutation
# tests and effect sizes without pretending to be people.

READER_NAMES = ("reader_a", "reader_b", "reader_c", "reader_d", "reader_e",
                "reader_f", "reader_g", "reader_h", "reader_i", "reader_j",
                "reader_k", "reader_l")

# Item "expectation" tables.  Each is a *design assumption of the
# simulation* on a 0-7 scale, not a measurement.  The felt-fidelity table
# is anchored to #74's published channels -- readings that round-trip the
# operator channel sit above readings that destroy it -- but it is **not**
# a monotone function of them, because the whole point of the study is that
# readers are not machine measures: the two operator-destroying English
# readings (`interlinear_gloss`, `plain_english`) sit *above* the machine's
# ranking of them, since they are the ones a reader finds easiest to read.
# A test asserts that the rank order differs from #74's, so the pilot's
# machine/human correlation is a real (simulated) correlation and not a
# restatement of the machine score.
FEELING_OFFSET = {
    "identity": 6.6,                 # positive control, near the ceiling
    "llm_register_paraphrase": 5.6,  # near-verbatim: carries the surface
    "llm_literal_operator": 5.0,     # operators kept, buried in asides
    "llm_entity_consistent": 4.4,    # reading-conditioned, annotated
    "llm_plain_english": 3.9,        # easiest to read, least faithful
    "interlinear_gloss": 3.4,        # #62's literal reading
    "llm_scene_summary": 3.5,        # condensed and fluent, lexis lost
}

# The diagnostic item, with its own table.  It measures something genuinely
# different from fidelity: the over-corrected reading is the *most*
# appealing and the *least* faithful, which is the confound the instrument
# exists to expose rather than hide.
APPEAL_OFFSET = {
    "identity": 5.0,                 # the raw clause: not appealing prose
    "llm_register_paraphrase": 5.6,
    "llm_literal_operator": 3.4,     # editorial apparatus slows it down
    "llm_entity_consistent": 3.8,
    "llm_plain_english": 6.2,        # readers love the repaired English
    "interlinear_gloss": 5.2,
    "llm_scene_summary": 4.6,
}

# The traps: the accepted answer is the bottom of the scale.
TRAP_OFFSET = {
    "trap_overcorrected": 0.6,       # smoothing the derangement is unfaithful
    "trap_invented": 0.8,            # adding a plot event is unfaithful
}

# The calibration item is derived, not tabulated: a reader is confident in
# proportion to how far the rating sits from the midpoint of the scale.
CONFIDENCE_MIDPOINT = 3.5
CONFIDENCE_GAIN = 1.9


def item_expectation(cell: GlossPair, item: str) -> float:
    """The simulation's expected score for one (cell, item), 0-7."""
    if item in TRAP_OFFSET:
        return TRAP_OFFSET[item]
    if item == APPEAL_ITEM:
        return APPEAL_OFFSET.get(cell.reading, 3.5)
    if item == "fidelity_confidence":
        felt = FEELING_OFFSET.get(cell.reading, 3.5)
        return CONFIDENCE_MIDPOINT + CONFIDENCE_GAIN * (abs(felt - CONFIDENCE_MIDPOINT)
                                                        / CONFIDENCE_MIDPOINT)
    return FEELING_OFFSET.get(cell.reading, 3.5)


# How strongly a reader's ability (leniency/harshness) moves each item.
ITEM_ABILITY_WEIGHT = {
    "felt_fidelity": 0.55,
    "fidelity_confidence": 0.30,
    "reading_appeal": 0.45,
    "trap_overcorrected": 0.50,
    "trap_invented": 0.50,
}


def reader_abilities(n_readers: int, seed: int) -> dict:
    """Per-reader ability, noise and order effect, drawn once from the seed."""
    rng = random.Random(seed)
    out: dict = {}
    for i in range(n_readers):
        out[READER_NAMES[i % len(READER_NAMES)]] = {
            "ability": rng.gauss(0.0, 0.45),     # leniency / harshness
            "noise": abs(rng.gauss(0.30, 0.12)) + 0.05,
            "order_effect": rng.choice((-1, 1)) * abs(rng.gauss(0.10, 0.05)),
        }
    return out


def simulate_ratings(blocks: list, schedule: list, n_readers: int,
                     seed: int) -> list:
    """Generate the pilot: one row per (reader, cell, item).

    Deterministic given (blocks, n_readers, seed): the reader abilities
    are drawn first, then the ratings in a fixed order, so a rebuild is
    byte-identical.  Every rating is the item's expectation, shifted by the
    reader's ability, wobbled by that reader's noise, and displaced by the
    order effect when the cell was shown first -- then clipped to 0-7 and
    **rounded to an integer**, because the instrument presents a 7-point
    scale and a real response is an integer.  The order effect is a *seeded
    design feature*, not a bug: a real run has position effects too, and
    recording presentation order is what lets the analysis measure them.
    """
    abilities = reader_abilities(n_readers, seed)
    rng = random.Random(seed + 1)
    rows: list = []
    for b in blocks:
        order = next(s["order"] for s in schedule
                     if s["clause_id"] == b["clause_id"])
        by_id = {c.pair_id: c for c in b["cells"]}
        for pos, pair_id in enumerate(order):
            cell = by_id[pair_id]
            for name in list(abilities)[:n_readers]:
                ab = abilities[name]
                for item in SCALE_KEYS:
                    mu = item_expectation(cell, item)
                    value = mu + ab["ability"] * ITEM_ABILITY_WEIGHT.get(item, 0.4)
                    if pos == 0:
                        value += ab["order_effect"]
                    value = min(7.0, max(0.0, value + rng.gauss(0.0, ab["noise"])))
                    rows.append({
                        "reader": name,
                        "clause_id": b["clause_id"],
                        "stratum": b["stratum"],
                        "pair_id": pair_id,
                        "reading": cell.reading,
                        "kind": cell.kind,
                        "item": item,
                        "score": round(value),
                        "position": pos,
                        "simulated": True,
                    })
    return rows


def simulate_pairwise(ratings: list, schedule: list, n_readers: int,
                      seed: int) -> list:
    """Generate the pairwise items from the same ratings.

    A pairwise choice is *derived* from the simulated scale ratings of the
    two cells being compared (plus a small seeded jitter), which keeps the
    two halves of the instrument consistent: a reader who rates ``identity``
    far above ``llm_scene_summary`` on the primary item also chooses
    ``identity`` in the pairwise comparison of that pair.  A real run
    collects both halves independently; the simulation derives one from the
    other so that a disagreement between the two halves cannot be an
    artefact of generating them separately.
    """
    primary: dict = {}
    for r in ratings:
        if r["item"] == PRIMARY_ITEM:
            primary[(r["reader"], r["pair_id"])] = r["score"]
    rng = random.Random(seed + 2)
    out: list = []
    for s in schedule:
        for a, b in s["pairwise_pairs"]:
            for name in READER_NAMES[:n_readers]:
                sa = primary.get((name, a))
                sb = primary.get((name, b))
                if sa is None or sb is None:
                    continue
                jitter = rng.gauss(0.0, 0.25)
                if abs((sa + jitter) - sb) < 0.35:
                    choice = "tie"
                elif (sa + jitter) > sb:
                    choice = "a"
                else:
                    choice = "b"
                out.append({
                    "reader": name, "clause_id": s["clause_id"],
                    "stratum": s["stratum"], "item": "pair_preference",
                    "pair_a": a, "pair_b": b, "choice": choice,
                    "score_a": sa, "score_b": sb, "simulated": True,
                })
    return out


def attention_flags(ratings: list, readers: list | None = None) -> list:
    """Readers who fail a planted attention check.

    The check is not a filter that silently removes readers -- the flags
    are published and every excluded row is counted, so reader exclusions
    are auditable instead of a quiet denominator change (L9.8
    residue-dropping).

    ``readers`` defaults to every reader present in the ratings, so a
    planted or partially-collected reader is judged on their own answers
    rather than skipped for not being in a fixed roster.
    """
    by_reader: dict = {}
    for r in ratings:
        by_reader.setdefault(r["reader"], {})[(r["kind"], r["clause_id"],
                                              r["item"])] = r["score"]
    roster = list(readers) if readers is not None else sorted(by_reader)
    flags: list = []
    for name in roster:
        got = by_reader.get(name, {})
        failed = []
        for check_id, kind, item, expected, why in ATTENTION_CHECKS:
            values = [v for (k, _c, it), v in got.items()
                      if k == kind and it == item]
            if values and abs(sum(values) / len(values) - expected) > ATTENTION_TOLERANCE:
                failed.append(check_id)
        flags.append({"reader": name, "failed_checks": ",".join(failed) or "none",
                      "excluded": bool(failed), "reason": why if failed else ""})
    return flags


# --------------------------------------------------------------------------
# 5. statistics the real run will use
# --------------------------------------------------------------------------
# All of them are pure functions of a rating matrix, so this module's pilot
# and a real run's data go through the *same* code.  Nothing here calls a
# model, reads a clock, or touches the network.


def mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def sd(values) -> float:
    """Sample standard deviation (Bessel-corrected); 0.0 for n < 2."""
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def quantile(sorted_vals: list, q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(values, n_boot: int, seed: int, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI for the mean, seeded so it is reproducible."""
    vals = list(values)
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    draws = []
    for _ in range(n_boot):
        draws.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return (quantile(draws, alpha / 2), quantile(draws, 1 - alpha / 2))


def _categories(values) -> tuple:
    cats = sorted({v for v in values if v is not None})
    return tuple(cats)


def _delta_squared(categories: tuple, marginals: dict, metric: str) -> dict:
    """The squared difference function between two categories.

    **Interval** (the default, and the study's pre-registered metric):
    ``delta^2(c,k) = (c - k)^2``.  A 0-7 rating scale carries equal
    intervals, so a 6 against a 5 must cost far less than a 6 against a 0
    -- which is what keeps this statistic meaningful on a Likert scale.
    The **nominal** metric counts every non-identical pair as equally
    wrong, which on a finely-graded scale drives alpha toward zero no
    matter how well the readers agree.

    **Ordinal**: Krippendorff's rank metric

        delta^2(c,k) = ( sum_{g in [c..k]} n_g - (n_c + n_k)/2 )^2

    which is invariant to any monotone re-scaling of the scale.  That
    invariance is a property and a disclosed trap: with only two categories
    present, "6 against 5" and "6 against 0" are the same rank distance, so
    the ordinal metric cannot tell them apart.  It is published as a
    robustness check, and the regression suite pins the limitation rather
    than silently relying on the metric.

    **Nominal**: ``delta^2 = 1`` for different categories, ``0`` for the
    same.  This is the form Krippendorff's own worked examples use (binary
    data, where all three metrics coincide), so it is the path the
    regression suite verifies against the published number.

    A non-numeric category set falls back to nominal, since no interval or
    rank order is defined for it.
    """
    if metric not in ("interval", "ordinal", "nominal"):
        raise ValueError(f"unknown metric: {metric}")
    numeric = all(isinstance(c, (int, float)) and not isinstance(c, bool)
                  for c in categories)
    if not numeric:
        metric = "nominal"
    delta: dict = {}
    for c in categories:
        for k in categories:
            if c == k:
                delta[(c, k)] = 0.0
            elif metric == "nominal":
                delta[(c, k)] = 1.0
            elif metric == "interval":
                delta[(c, k)] = float((c - k) ** 2)
            else:
                lo, hi = (c, k) if c <= k else (k, c)
                inner = sum(marginals.get(g, 0.0) for g in categories
                            if lo <= g <= hi)
                delta[(c, k)] = (inner - (marginals.get(c, 0.0)
                                          + marginals.get(k, 0.0)) / 2.0) ** 2
    return delta


def krippendorff_alpha(units: list, metric: str = PRIMARY_METRIC) -> float:
    """Krippendorff's alpha over units of unequal size.

    ``units`` is a list of rating vectors, one per *unit* (a (clause, cell)
    rated by several readers, or a reader's several cells).  The
    coincidence matrix counts each unit's ordered pairs of *distinct*
    values, weighted by ``1/(m_u - 1)`` so units with different numbers of
    raters contribute in proportion to their size, and ``alpha = 1 -
    D_o/D_e`` where ``D_o`` is the mean weighted disagreement inside units
    and ``D_e`` is the disagreement expected from the pooled marginals
    without replacement.

    The metric is chosen by ``_delta_squared`` and defaults to the
    study's pre-registered one (``PRIMARY_METRIC`` = ``interval``, a 0-7
    Likert scale carries equal intervals).  Pinned by the regression suite
    against two published worked examples from Krippendorff's 2011 note:

    * binary data, two observers, no missing values, nominal metric
      (``o_00=10, o_01=o_10=4, o_11=2``) -> alpha = 2/21 = 0.095238 --
      for binary data all three metrics coincide, so this pins the whole
      machinery;
    * perfect agreement -> alpha = 1.0 exactly, under every metric.

    Perfect agreement gives 1.0; a single category gives 1.0 (no chance
    disagreement to correct for); a unit with one value contributes
    nothing and is skipped; ``nan`` is returned when fewer than two
    pairable values exist.
    """
    coincidence: dict = {}
    n = 0
    for values in units:
        vals = [v for v in values if v is not None]
        m = len(vals)
        if m < 2:
            continue
        weight = 1.0 / (m - 1)
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i == j:
                    continue
                key = (a, b)
                coincidence[key] = coincidence.get(key, 0.0) + weight
        n += m
    if n < 2 or not coincidence:
        return float("nan")
    categories = _categories({v for (a, b) in coincidence for v in (a, b)})
    marginals: dict = {}
    for (a, _b), v in coincidence.items():
        marginals[a] = marginals.get(a, 0.0) + v
    delta = _delta_squared(categories, marginals, metric)
    observed = sum(v * delta[(a, b)] for (a, b), v in coincidence.items()) / n
    expected = sum(marginals[c] * marginals[k] * delta[(c, k)]
                   for c in categories for k in categories) / (n * (n - 1))
    if expected <= 0:
        return 1.0 if observed <= 0 else 0.0
    return 1.0 - observed / expected


def permutation_test(values_a, values_b, n_perm: int, seed: int) -> dict:
    """Two-sided permutation test for a difference of means.

    The statistic is ``mean(a) - mean(b)``; the null distribution pools both
    samples and redraws the same sizes ``n_perm`` times with a seeded RNG.
    The p-value is the share of permutations at least as extreme as the
    observed difference, with the conventional ``+1`` correction so a
    perfectly separated sample never reports p = 0.
    """
    a = list(values_a)
    b = list(values_b)
    if not a or not b:
        return {"observed": float("nan"), "p": float("nan"), "n_perm": 0}
    observed = mean(a) - mean(b)
    pool = a + b
    na = len(a)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        stat = mean(pool[:na]) - mean(pool[na:])
        if abs(stat) >= abs(observed) - 1e-12:
            hits += 1
    return {"observed": observed, "p": (hits + 1) / (n_perm + 1),
            "n_perm": n_perm}


def rankdata(values) -> list:
    """Ranks with ties averaged (1-based), no scipy."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys) -> float:
    """Spearman rank correlation; 0.0 when either side is constant."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = rankdata(list(xs))
    ry = rankdata(list(ys))
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def kruskal_h(groups) -> float:
    """Kruskal-Wallis H over the supplied groups (tie-uncorrected)."""
    groups = [list(g) for g in groups if len(g)]
    if len(groups) < 2:
        return float("nan")
    pooled = [v for g in groups for v in g]
    ranks = rankdata(pooled)
    n = len(pooled)
    idx = 0
    total = 0.0
    for g in groups:
        m = len(g)
        total += (sum(ranks[idx:idx + m]) ** 2) / m
        idx += m
    h = (12.0 / (n * (n + 1))) * total - 3.0 * (n + 1)
    return h


def eta_squared(groups) -> float:
    """Eta-squared (the between-group share of rank variance)."""
    groups = [list(g) for g in groups if len(g)]
    if not any(len(g) > 1 for g in groups):
        return float("nan")
    pooled = [v for g in groups for v in g]
    overall = mean(pooled)
    ss_between = sum(len(g) * (mean(g) - overall) ** 2 for g in groups)
    ss_total = sum((v - overall) ** 2 for v in pooled)
    return ss_between / ss_total if ss_total > 0 else float("nan")


# --------------------------------------------------------------------------
# 6. reconciliation against the artifacts the frame is built from
# --------------------------------------------------------------------------
def reconciliation(frame: list, blocks: list, sample: list,
                   interlinear: dict, glosses: dict, ratings: list,
                   pairs: list) -> list:
    """This build vs #62 / #63 / #74.

    Exact-match rows are build errors when they drift; disclosed rows carry
    a reason, following #68's and #74's convention.
    """
    out = []
    candidates = candidate_cells(frame)
    traps = [c for c in frame if c.kind == "trap"]

    exact = [
        ("n_interlinear_rows", len(interlinear), PUBLISHED["n_interlinear_rows"],
         "#33/#62 clause count (exact match required)"),
        ("n_sample_clauses", len(sample), PUBLISHED["n_sample_clauses"],
         "#63 stratified sample size (exact match required)"),
        ("n_strategies", len({r["strategy"] for r in read_tsv(GLOSSES)}),
         PUBLISHED["n_strategies"], "#63 strategies (exact match required)"),
        ("n_gloss_rows", len(read_tsv(GLOSSES)), PUBLISHED["n_gloss_rows"],
         "#63 sampled glosses (exact match required)"),
        ("n_frame_cells", len(frame), len(frame),
         "identity + gloss + 5 strategies + 2 traps per clause"),
        ("n_candidate_cells", len(candidates), len(candidates),
         "rateable candidate cells (traps excluded)"),
        ("n_trap_cells", len(traps), len(traps),
         "planted trap cells, excluded from fidelity statistics"),
        ("n_blocks", len(blocks), len(sample),
         "one clause block per sampled clause"),
    ]
    for qty, here, earlier, note in exact:
        out.append({"quantity": qty, "earlier": earlier, "here": here,
                    "note": note})
        if here != earlier and qty not in ("n_frame_cells",
                                           "n_candidate_cells",
                                           "n_trap_cells"):
            raise SystemExit(f"#75: {qty} {here} != published {earlier} -- "
                             "upstream artifact drifted; refusing to build")

    # Disclosed rows: the pairs the design draws, the ratings generated,
    # and the analytic population they are drawn from.  A block draws
    # 2 identity+gloss plus a rotated LLM draw, so the sampled-pair count is
    # a function of the counterbalancing, published rather than asserted.
    n_sampled = len(sampled_pairs(blocks))
    out.append({"quantity": "n_sampled_pairs", "earlier": n_sampled,
                "here": n_sampled,
                "note": f"(clause, gloss) cells shown to a reader: "
                        f"{len(load_sample())} blocks x "
                        f"{2 + N_PAIRS_PER_CLAUSE + 2} cells "
                        f"(identity + #62 gloss + {N_PAIRS_PER_CLAUSE} "
                        f"rotated LLM comparators + 2 trap cells)"})
    out.append({"quantity": "n_ratings", "earlier": len(ratings),
                "here": len(ratings),
                "note": "simulated pilot rows (reader x cell x scale item); "
                        "a human run replaces this row, it does not "
                        "reinterpret it"})
    out.append({"quantity": "n_pairwise_judgements", "earlier": len(pairs),
                "here": len(pairs),
                "note": "simulated pairwise judgements over adjacent cells"})
    return out


# --------------------------------------------------------------------------
# 7. the pilot's analysis
# --------------------------------------------------------------------------
def cell_meta(frame: list) -> dict:
    """pair_id -> (clause_id, stratum, reading, kind), from the frame itself.

    Deriving the metadata from the frame rather than from the ``pair_id``
    string keeps the names identical to #74's reading names (a ``pair_id``
    suffix ``gloss`` is #74's ``interlinear_gloss``) and keeps a trap
    identifiable without re-parsing roles out of an identifier.
    """
    return {c.pair_id: (c.clause_id, c.stratum, c.reading, c.kind)
            for c in frame}


def primary_by_cell(ratings: list) -> dict:
    """reader-averaged primary score per pair_id (the unique cell key).

    The key is the ``pair_id``, never (clause, reading): a clause can carry
    the same reading under two roles -- #63's ``plain_english`` is both a
    candidate reading and the text of both trap cells -- and a key that
    merged them would average a trap into the fidelity statistic it is
    supposed to be excluded from.  Traps are excluded here and analysed
    separately, by ``agreement_by_item``.
    """
    acc: dict = {}
    for r in ratings:
        if r["item"] != PRIMARY_ITEM or r["kind"] == "trap":
            continue
        acc.setdefault(r["pair_id"], []).append(r["score"])
    return {k: mean(v) for k, v in acc.items()}


def per_cell_summary(frame: list, ratings: list) -> list:
    """Per-cell primary-item statistics, in the frame's own order.

    Includes the identity control and the #62 gloss (the comparators every
    block carries), excludes the traps, and reports the within-clause
    contrast of each reading against the identity control -- the number the
    study exists to produce.
    """
    meta = cell_meta(frame)
    acc: dict = {}
    for r in ratings:
        if r["item"] != PRIMARY_ITEM or r["kind"] == "trap":
            continue
        acc.setdefault(r["pair_id"], []).append(r["score"])
    identity = {k.split("/", 1)[0]: v
                for k, v in primary_by_cell(ratings).items()
                if meta[k][2] == "identity"}
    rows = []
    for pair_id, values in sorted(acc.items(), key=lambda kv: kv[0]):
        lo, hi = bootstrap_ci(values, N_BOOT, SEED + len(values))
        clause_id, _stratum, reading, kind = meta[pair_id]
        base = identity.get(clause_id, float("nan"))
        rows.append({
            "clause_id": clause_id,
            "pair_id": pair_id,
            "reading": reading,
            "kind": kind,
            "n_readers": len(values),
            "mean": round(mean(values), 4),
            "sd": round(sd(values), 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "identity_mean": round(base, 4),
            "gap_vs_identity": round(mean(values) - base, 4),
        })
    return rows


def per_stratum_summary(rows: list, strata: dict) -> list:
    """The pre-registered between-stratum contrast.

    #63's sample is stratified *so that each check has at least one case
    where it can fail*; the study keeps that property and asks whether the
    reader ratings separate the strata.  If a human run finds they do not,
    that is a result about readers' tolerance for deranged syntax, not a
    failure of the instrument -- which is why it is reported per stratum
    and never pooled into one verdict.
    """
    acc: dict = {}
    for r in rows:
        acc.setdefault(r["stratum"], []).append(r["mean"])
    out = []
    for stratum, values in sorted(acc.items()):
        out.append({"stratum": stratum, "n_cells": len(values),
                    "mean_felt_fidelity": round(mean(values), 4),
                    "sd": round(sd(values), 4)})
    return out


def contrast_identity_vs_glosses(frame: list, ratings: list) -> dict:
    """Within-clause paired contrast: identity control vs every other reading.

    For each clause the primary-item means of the identity control and of
    the other readings are differenced, and the clause-level differences are
    pooled (so a clause that is hard for everyone does not masquerade as a
    reading that is bad).  The null distribution **permutes the label**: in
    each clause a reading is drawn at random to play the control role and
    the contrast is recomputed.  That destroys exactly the claim being
    tested -- that the ordering follows which reading is the clause itself
    -- while keeping every clause's values, its size and its difficulty.
    """
    meta = cell_meta(frame)
    per_clause: dict = {}
    for pair_id, value in primary_by_cell(ratings).items():
        clause_id, _stratum, reading, _kind = meta[pair_id]
        per_clause.setdefault(clause_id, {})[reading] = value
    if not per_clause:
        return {"observed": float("nan"), "p": float("nan"), "n_clauses": 0,
                "per_clause": []}
    diffs = []
    for clause_id, readings in sorted(per_clause.items()):
        if "identity" not in readings:
            continue
        others = [v for r, v in sorted(readings.items()) if r != "identity"]
        if not others:
            continue
        diffs.append(readings["identity"] - mean(others))
    if not diffs:
        return {"observed": float("nan"), "p": float("nan"), "n_clauses": 0,
                "per_clause": []}
    observed = mean(diffs)
    rng = random.Random(SEED)
    hits = 0
    for _ in range(N_PERM):
        null_diffs = []
        for readings in per_clause.values():
            keys = sorted(readings)
            control = keys[rng.randrange(len(keys))]
            others = [v for k, v in readings.items() if k != control]
            if not others:
                continue
            null_diffs.append(readings[control] - mean(others))
        if null_diffs and abs(mean(null_diffs)) >= abs(observed) - 1e-12:
            hits += 1
    return {"observed": round(observed, 4),
            "p": (hits + 1) / (N_PERM + 1),
            "n_clauses": len(diffs),
            "per_clause": [round(d, 4) for d in diffs]}


def machine_human_agreement(rows: list, fidelity: dict) -> dict:
    """Does the human primary score track #74's published fidelity?

    The human mean per reading is correlated (Spearman) with #74's
    ``aggregate_fidelity`` for the same readings.  This is the
    study's *external* leg: the judgments are human, the machine score is
    #74's, and the two are independent measurements of the same readings.
    A positive rank correlation is the evidence that the instrument is
    measuring fidelity rather than fluency; the null distribution is built
    by permuting the machine labels, so a correlation that survives its own
    permutation is not an artefact of the reading order.
    """
    acc: dict = {}
    for r in rows:
        acc.setdefault(r["reading"], []).append(r["mean"])
    readings = sorted(acc)
    human = [mean(acc[r]) for r in readings]
    machine = [fidelity.get(r, {}).get("aggregate_fidelity", float("nan"))
               for r in readings]
    keep = [i for i, m in enumerate(machine) if not math.isnan(m)]
    if len(keep) < 3:
        return {"spearman": float("nan"), "p": float("nan"), "n": len(keep)}
    hs = [human[i] for i in keep]
    ms = [machine[i] for i in keep]
    observed = spearman(hs, ms)
    rng = random.Random(SEED + 3)
    hits = 0
    for _ in range(N_PERM):
        shuffled = list(ms)
        rng.shuffle(shuffled)
        if abs(spearman(hs, shuffled)) >= abs(observed) - 1e-12:
            hits += 1
    return {"spearman": round(observed, 4), "p": (hits + 1) / (N_PERM + 1),
            "n": len(keep),
            "pairs": [{"reading": readings[i], "human": round(human[i], 4),
                       "machine_fidelity": round(machine[i], 4)}
                      for i in keep]}


def reading_effect(rows: list) -> dict:
    """Between-reading effect size, pre-registered leg 6 of the protocol.

    The cell means are grouped by reading and compared with
    Kruskal-Wallis H and eta-squared.  This is the *non-parametric*
    companion to the per-cell CIs: it says whether the readings differ at
    all without assuming the per-cell spread is normal, which is the honest
    way to report a 7-item instrument over 5 clauses.
    """
    acc: dict = {}
    for r in rows:
        acc.setdefault(r["reading"], []).append(r["mean"])
    groups = [acc[k] for k in sorted(acc)]
    if len(groups) < 2:
        return {"kruskal_h": float("nan"), "eta_squared": float("nan"),
                "n_readings": len(groups)}
    return {"kruskal_h": round(kruskal_h(groups), 4),
            "eta_squared": round(eta_squared(groups), 4),
            "n_readings": len(groups),
            "groups": {k: round(mean(v), 4) for k, v in sorted(acc.items())}}


def _units_from_cells(grid: dict, readers: list, cells: list) -> list:
    """Group a ``(reader, cell)`` -> score grid into cell-shaped units."""
    units = []
    for cid in cells:
        vals = [grid[(rd, cid)] for rd in readers if (rd, cid) in grid]
        if len(vals) > 1:
            units.append(vals)
    return units


def alpha_null_draws(grid: dict, readers: list, cells: list, n_draws: int,
                     seed: int) -> list:
    """The null distribution for a cell-shaped coincidence matrix.

    Each reader's scores are shuffled **across the cells**, so a cell ends
    up holding values that were earned on other cells.  Each reader stays
    themselves -- their own response distribution is untouched -- but no
    longer rates the same cells high, which is exactly the agreement being
    destroyed.

    Shuffling *inside* a unit would be a no-op: the unit's value multiset
    does not change, so the coincidence matrix, ``D_o`` and ``alpha`` all
    come out identical, and a null that cannot move is worse than no null
    at all.  This is a module-level entry point rather than a closure
    precisely so the regression suite can assert that the null *does* move.
    """
    rng = random.Random(seed)
    nulls = []
    for _ in range(n_draws):
        null_grid = {}
        for rd in readers:
            cells_rd = [cid for cid in cells if (rd, cid) in grid]
            vals = [grid[(rd, cid)] for cid in cells_rd]
            rng.shuffle(vals)
            for cid, v in zip(cells_rd, vals):
                null_grid[(rd, cid)] = v
        a = krippendorff_alpha(_units_from_cells(null_grid, readers, cells))
        if not math.isnan(a):
            nulls.append(a)
    nulls.sort()
    return nulls


def agreement_by_item(ratings: list) -> list:
    """Krippendorff's alpha per item, over the cells rated by the readers.

    A unit is one **cell** (``pair_id``), and its values are the readers'
    scores for that item -- so the pairs alpha compares are two *readers*
    rating the same stimulation, which is what "the readers agree" means.
    A unit is never (clause, reading): a clause can carry the same reading
    under two roles (a candidate and both traps), and merging them would
    average a trap into the statistic meant to test it.

    The null is ``alpha_null_draws``: it re-groups the values rather than
    reshuffling inside a unit, so every reported alpha is read against what
    chance produces on the same data.
    """
    readers = sorted({r["reader"] for r in ratings})
    cells = sorted({r["pair_id"] for r in ratings})
    out = []
    for item in SCALE_KEYS:
        grid: dict = {}
        for r in ratings:
            if r["item"] == item:
                grid[(r["reader"], r["pair_id"])] = r["score"]
        units = _units_from_cells(grid, readers, cells)
        alpha = krippendorff_alpha(units)
        nulls = alpha_null_draws(grid, readers, cells, ALPHA_DRAWS, SEED + 4)
        p = ((sum(1 for v in nulls if v >= alpha) + 1) / (len(nulls) + 1)
             if nulls else float("nan"))
        out.append({"item": item, "alpha": round(alpha, 4),
                    "null_q05": round(quantile(nulls, 0.05), 4) if nulls else "",
                    "null_q95": round(quantile(nulls, 0.95), 4) if nulls else "",
                    "null_p": round(p, 4), "n_units": len(units)})
    return out


def pairwise_summary(pairs: list) -> list:
    """Choice shares for the pairwise items, per clause.

    Reported as the share choosing the higher-fidelity cell, which is the
    form a human run's data will take: the two pairwise items are
    *forced-choice*, so they are published as shares and never averaged
    into a scale score.
    """
    by_pair: dict = {}
    for r in pairs:
        key = (r["clause_id"], r["pair_a"], r["pair_b"])
        by_pair.setdefault(key, []).append(r["choice"])
    out = []
    for (cid, a, b), choices in sorted(by_pair.items()):
        n = len(choices)
        out.append({"clause_id": cid, "pair_a": a, "pair_b": b,
                    "n": n,
                    "share_a": round(choices.count("a") / n, 4) if n else "",
                    "share_tie": round(choices.count("tie") / n, 4) if n else "",
                    "share_b": round(choices.count("b") / n, 4) if n else ""})
    return out


# --------------------------------------------------------------------------
# 8. artifacts
# --------------------------------------------------------------------------
def write_frame_tsv(path: Path, frame: list, blocks: list, fidelity: dict) -> None:
    """The sampling frame, published in full -- traps included.

    Every cell the study can rate is a row, whether a block draws it or not,
    with the role it plays and the published machine-measured fidelity of
    its reading.  A frame that published only the drawn cells would hide
    what was left out, which is the residue-dropping failure mode.
    """
    drawn = {p for b in blocks for p in b["pair_ids"]}
    rows = []
    for c in frame:
        rows.append({
            "pair_id": c.pair_id,
            "clause_id": c.clause_id,
            "page": c.page,
            "stratum": c.stratum,
            "reading": c.reading,
            "kind": c.kind,
            "drawn_in_block": c.pair_id in drawn,
            "n_words": len(c.text.split()),
            "n_chars": len(c.text),
            "role": c.role,
            "machine_fidelity_74": round(
                fidelity.get(c.reading, {}).get("aggregate_fidelity", float("nan")), 4),
            "gloss_text": c.text,
            "original_clause": c.original,
        })
    write_tsv(path, ["pair_id", "clause_id", "page", "stratum", "reading",
                     "kind", "drawn_in_block", "n_words", "n_chars", "role",
                     "machine_fidelity_74", "gloss_text",
                     "original_clause"], rows)


def write_instrument_tsv(path: Path) -> None:
    """The instrument itself, one row per item, so the design is readable.

    The anchors are published verbatim: a study whose scale endpoints are
    not written down cannot be run twice the same way.
    """
    rows = []
    for key, prompt, low, high in SCALE_ITEMS:
        rows.append({"item": key, "type": "scale", "role": "primary" if key == PRIMARY_ITEM
                     else "diagnostic" if key == APPEAL_ITEM else "calibration",
                     "prompt": prompt, "low_anchor": low, "high_anchor": high,
                     "n_scale": 8, "analysed_as": "mean with bootstrap CI"})
    for key, prompt, low, high in TRAP_ITEMS:
        rows.append({"item": key, "type": "scale", "role": "trap",
                     "prompt": prompt, "low_anchor": low, "high_anchor": high,
                     "n_scale": 8, "analysed_as": "pass/fail per reader"})
    for key, prompt, options in PAIRWISE_ITEMS:
        rows.append({"item": key, "type": "forced_choice",
                     "role": "calibration", "prompt": prompt,
                     "low_anchor": f"a = {options[0]}",
                     "high_anchor": f"b = {options[1]}",
                     "n_scale": "", "analysed_as": "choice shares"})
    write_tsv(path, ["item", "type", "role", "prompt", "low_anchor",
                     "high_anchor", "n_scale", "analysed_as"], rows)


def write_blocks_tsv(path: Path, blocks: list, schedule: list) -> None:
    """The counterbalanced packets, in the order a reader actually sees.

    The rows are the *presentation* order from ``block_schedule``, not the
    base order: the within-block rotation is part of the counterbalancing,
    so an artifact that published the base order would contradict the
    report's claim that no reading is systematically first.
    """
    rows = []
    by_clause = {s["clause_id"]: s for s in schedule}
    for b in blocks:
        for pos, pid in enumerate(by_clause[b["clause_id"]]["order"]):
            rows.append({"clause_id": b["clause_id"], "page": b["page"],
                         "stratum": b["stratum"], "position": pos,
                         "pair_id": pid, "block_offset": b["offset"]})
    write_tsv(path, ["clause_id", "page", "stratum", "position", "pair_id",
                     "block_offset"], rows)


def write_ratings_tsv(path: Path, ratings: list) -> None:
    write_tsv(path, ["reader", "clause_id", "stratum", "pair_id", "reading",
                     "kind", "item", "score", "position", "simulated"],
              ratings)


def write_pairwise_tsv(path: Path, pairs: list) -> None:
    write_tsv(path, ["reader", "clause_id", "stratum", "item", "pair_a",
                     "pair_b", "choice", "score_a", "score_b", "simulated"],
              pairs)


def _emit_protocol(L: list, ctx: dict) -> None:
    """Append the pre-registration lines to ``L``.

    Published as a separate file so a human run can be started from it
    without reading the pilot, and so an auditor can check the claims the
    analysis is allowed to make.
    """
    blocks = ctx["blocks"]
    frame = ctx["frame"]
    a = L.append
    a("# Reader study protocol (#75)\n")
    a("")
    a("Pre-registration for the human jury. **No human rating exists yet** "
      "-- this protocol is written before any reader is recruited, and the "
      "pilot published beside it in `reader_study_report.md` is a "
      "*simulation* of this protocol run on synthetic readers (seed 75), "
      "not a result from it.  Nothing in the pilot is evidence about "
      "readers; its job is to prove the instrument and the analysis "
      "pipeline work before a person is asked.\n")
    a("## 1. Question\n")
    a("Do the published readings of *Blood Electric* carry the **felt "
      "quality** of the clauses they translate, as judged by readers who "
      "are not the system that produced them?\n")
    a("## 2. Population and sampling\n")
    a("**Stimuli.** The 5 clauses of #63's stratified sample "
      "(`analysis/glossing/sample.tsv`), one per stratum "
      "(`plain_control`, `quoting_frame`, `operator_dense`, `entity_named`, "
      "`residue`), each shown with:\n")
    a("")
    a("1. the clause itself (identity control),")
    a("2. #62's interlinear gloss (the non-LLM baseline),")
    a("3. a counterbalanced draw of "
      f"{N_PAIRS_PER_CLAUSE} of #63's five LLM readings.\n")
    a("")
    a(f"That is {len(candidate_cells(frame))} rateable candidate cells "
      f"across {len(blocks)} blocks, plus "
      f"{sum(1 for c in frame if c.kind == 'trap')} trap cells used to check "
      "the instrument. The sampling frame is published in full "
      "(`frame.tsv`) including the cells a block does *not* draw and why.\n")
    a("## 3. Design\n")
    a("* **Within-clause, within-reader**: every reader rates every cell of "
      "every block, so each reading is compared against the identity "
      "control *and* the other readings of the same clause.")
    a("* **What a block shows**: the identity control, #62's gloss, a "
      f"rotated draw of {N_PAIRS_PER_CLAUSE} of #63's five LLM readings, "
      "and the clause's two trap cells (instrument items, excluded from "
      "every fidelity statistic by ``kind``).")
    a("* **Counterbalancing**: within a block, cells are rotated by the "
      "block index; across blocks, the LLM comparator draw is rotated by "
      "`PAIR_OFFSETS`. No reading is systematically first or last.")
    a("* **Order-arm alternation**: the first position of each block "
      "receives a seeded order effect in the simulation, and a real run "
      "records presentation order per response so the same effect can be "
      "measured rather than assumed away.")
    a("* **Attention checks**: two planted cells with known answers (the "
      "identity control at the ceiling, the over-corrected trap at the "
      f"floor, tolerance {ATTENTION_TOLERANCE}). Failures are published, "
      "not silently dropped.\n")
    a("## 4. Instrument\n")
    a("Seven items per cell, published verbatim in `instrument.tsv`. The "
      "simulated pilot exercises the five scale items and "
      "`pair_preference`; `pair_invented_plot` is deliberately **not** "
      "simulated, because inventing a reader's judgement about an invented "
      "plot event is the one thing the simulation has no business "
      "assuming.\n")
    a("")
    for key, prompt, low, high in ALL_SCALE_ITEMS:
        a(f"* **{key}** -- {prompt} ({low}; {high})")
    for key, prompt, options in PAIRWISE_ITEMS:
        a(f"* **{key}** -- {prompt} (options: {' / '.join(options)})")
    a("")
    a(f"**{PRIMARY_ITEM}** is the study's outcome. "
      f"**{APPEAL_ITEM}** is a diagnostic: a reading that is appealing but "
      "unfaithful is a *finding*, and the instrument is built so the two "
      "cannot be conflated.\n")
    a("## 5. Analysis plan (pre-registered)\n")
    a("1. **Per-cell means** with seeded percentile bootstrap CIs "
      f"({N_BOOT} resamples).")
    a("2. **Within-clause contrast**: identity control minus the mean of "
      "the other readings, pooled per clause, tested by **permuting the "
      "control label** within each clause "
      f"({N_PERM} draws).")
    a("3. **Between-reading agreement with the machine**: Spearman "
      "correlation of the human per-reading mean against #74's published "
      f"`aggregate_fidelity`, with a {N_PERM}-draw permutation test on "
      "the machine labels.")
    a("4. **Inter-reader agreement**: Krippendorff's alpha with the "
      f"**{PRIMARY_METRIC}** difference function "
      "(`delta^2 = (c - k)^2`): a 0-7 rating scale carries equal "
      "intervals, so a 6-against-5 disagreement must cost far less than a "
      "6-against-0 one. Per item, over cells as units, each read against a "
      "null that re-groups the values (each reader's scores shuffled "
      f"across cells, {ALPHA_DRAWS} draws). The **ordinal** and "
      "**nominal** difference functions are implemented and published "
      "alongside it as robustness checks, not substitutes: with only two "
      "categories observed the ordinal metric cannot tell a 0-vs-6 "
      "disagreement from a 5-vs-6 one, and the nominal metric prices "
      "every non-identical pair the same, which on a finely-graded scale "
      "drives alpha toward zero however well the readers agree. The "
      "traps are analysed pass/fail against their known answer, not as a "
      "scale.")
    a("5. **Per-stratum descriptives**: reported separately, never pooled "
      "into a single verdict.")
    a("6. **Effect size**: Kruskal-Wallis H and eta-squared across "
      "readings, reported beside the raw means.")
    a("7. **Appeal against fidelity**: the diagnostic `reading_appeal` "
      "item is reported beside the primary item per reading, so a reading "
      "that is appealing but unfaithful is visible as such rather than "
      "averaged into the fidelity score.\n")
    a("## 6. Stopping rule and exclusions\n")
    a("* Recruit until every cell has at least 5 independent readers, or "
      "until a maximum of 12; both numbers are recorded either way.")
    a("* A reader who fails either attention check is flagged and their "
      "rows are itemised as excluded, with the count published.")
    a("* Partial responses are kept; the number of rated cells per reader "
      "is part of the dataset, not a filter.\n")
    a("## 7. What this study cannot establish\n")
    a("* It measures *felt fidelity*, not correctness. There is no "
      "ground-truth translation of *Blood Electric* (#74's finding), so a "
      "high score means readers recognise the feeling, not that the "
      "reading is the translation.")
    a("* Readers are not the text's original audience; the felt quality "
      "measured is the felt quality for these readers.")
    a("* A stratified 5-clause sample is a sample of #63's strata, not of "
      "the book.\n")
    a("## 8. Determinism\n")
    a("No clock, no network, no model call, fixed seed "
      f"({SEED}); every artifact is byte-identical on rebuild.")


def build_report(ctx: dict) -> str:
    L = []
    a = L.append
    frame = ctx["frame"]
    blocks = ctx["blocks"]
    ratings = ctx["ratings"]
    candidates = candidate_cells(frame)
    traps = [c for c in frame if c.kind == "trap"]

    a("# Human jury / reader study design (#75)\n")
    a("")
    a("Generated: deterministic (no clock read by any statistic)\n")
    a("")
    a("**This report contains no human data.** The reader judgments in it "
      "are *simulated* by a disclosed generative model, seeded "
      f"({SEED}), and exist to prove the instrument and the analysis "
      "pipeline work before a reader is recruited. They are not evidence "
      "about readers, and no claim in this report rests on them.\n")
    a("## 1. What was designed\n")
    a("The plan's **L9.7** item -- *have readers rate glosses for fidelity "
      "to the felt text* -- needs two things that did not exist: an "
      "**instrument** a reader can answer, and a **sampling** rule that says "
      "which glosses get rated and against what comparator. Both are here, "
      "along with the pre-registered analysis (`protocol.md`) and a "
      "simulated pilot that exercises every statistic the real run will "
      "use.\n")
    a(f"The frame is a **census** of the gloss population over a stratified "
      f"clause sample: {len(candidates)} candidate cells "
      f"({len(blocks)} clauses x "
      f"[1 identity control + 1 #62 gloss + {len(SAMPLED_STRATEGIES)} LLM "
      f"strategies]) plus {len(traps)} trap cells. Because the instrument "
      "compares a gloss against the clause it translates and against the "
      "other readings of the same clause, a reader cannot score a gloss "
      "without a comparator; that is the design property #63 and #64 both "
      "needed and neither had.\n")
    a("## 2. The instrument\n")
    a("| item | type | role | prompt | anchors |")
    a("|---|---|---|---|---|")
    for key, prompt, low, high in SCALE_ITEMS:
        role = ("primary" if key == PRIMARY_ITEM
                else "diagnostic" if key == APPEAL_ITEM else "calibration")
        a(f"| `{key}` | scale 0-7 | {role} | {prompt} | {low} / {high} |")
    for key, prompt, low, high in TRAP_ITEMS:
        a(f"| `{key}` | scale 0-7 | trap | {prompt} | {low} / {high} |")
    for key, prompt, options in PAIRWISE_ITEMS:
        a(f"| `{key}` | forced choice | calibration | {prompt} | "
          f"{' / '.join(options)} |")
    a("")
    a("The traps are the instrument's teeth. `trap_overcorrected` carries "
      "#63's `plain_english` reading -- the strategy #63's own checks most "
      "often fail and #74 measures as least faithful overall -- and asks "
      "how faithful its smoothing is to the clause's oddity; "
      "`trap_invented` asks the same about an addition the clause does not "
      "make. A reader who rates those high is not measuring fidelity, and "
      "the attention check says so rather than averaging over it.\n")
    a("## 3. Sampling and counterbalancing\n")
    a("| block | clause | stratum | page | cells shown | LLM draw offset |")
    a("|---|---|---|---|---|---|")
    for b in blocks:
        a(f"| {b['block_index']} | {b['clause_id']} | {b['stratum']} | "
          f"{b['page']} | {len(b['cells'])} | {b['offset']} |")
    a("")
    a("Every block shows the identity control and #62's gloss (the "
      "comparators without which a rating is uninterpretable), a rotated "
      f"draw of {N_PAIRS_PER_CLAUSE} LLM readings, and the clause's "
      f"{len([c for c in frame if c.kind == 'trap']) // len(blocks)} trap "
      "cells. The traps are instrument items: a reader who is never asked "
      "to judge an over-corrected reading cannot be checked for judging "
      "fidelity rather than appeal. The LLM rotation "
      "(`PAIR_OFFSETS`) and the within-block rotation are both functions "
      "of the block index, so the design is reproducible from the frame "
      "alone and no reading is systematically first.\n")
    a("## 4. The simulated pilot\n")
    a("**No human was involved in these numbers.** For each reader the "
      "model draws a leniency (``ability``), a noise level and an order "
      "effect from seeded Gaussians; each rating is that item's expectation "
      "for that cell, shifted by the reader's ability, displaced by the "
      "order effect when the cell was shown first, wobbled by the reader's "
      "noise, clipped to 0-7 and **rounded to an integer**, because the "
      "instrument presents a 7-point scale and a real response is an "
      "integer. Every expectation is a *design assumption*, published in "
      "the source as ``FEELING_OFFSET`` / ``APPEAL_OFFSET`` / "
      "``TRAP_OFFSET``: the felt-fidelity table is anchored to #74's "
      "channels but is deliberately **not** a monotone function of them, "
      "because the study's premise is that readers are not machine "
      "measures.\n")
    a(f"{len(ratings)} integer ratings over {len(blocks)} blocks x "
      f"{N_READERS} readers x {len(SCALE_KEYS)} items, plus "
      f"{len(ctx['pairs'])} forced-choice judgements.\n")
    a("**What the pilot does not generate:** the second forced-choice item "
      "(`pair_invented_plot`) is part of the instrument a human run must "
      "administer, and part of what this design commits to, but the "
      "simulation derives `pair_preference` only. Simulating the other one "
      "would mean inventing a reader's judgement about whether a wording "
      "commits to a plot event the clause merely hints at -- which is the "
      "one judgement the study exists to collect -- so it is left blank "
      "rather than filled with an assumption.\n")
    a("### 4.1 Per-cell primary scores\n")
    a("| clause | reading | kind | n | mean | sd | 95% CI | gap vs identity |")
    a("|---|---|---|---:|---:|---:|---|---:|")
    for r in ctx["cell_rows"]:
        a(f"| {r['clause_id']} | `{r['reading']}` | {r['kind']} | "
          f"{r['n_readers']} | {r['mean']:.3f} | {r['sd']:.3f} | "
          f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
          f"{r['gap_vs_identity']:+.3f} |")
    a("")
    a("### 4.2 The contrast the study exists for\n")
    c = ctx["contrast"]
    a("The identity control minus the mean of the other readings, pooled "
      f"over {c['n_clauses']} clauses (per clause: "
      f"{', '.join(f'{v:+.2f}' for v in c['per_clause'])}):\n")
    a("")
    a(f"**observed contrast {c['observed']:+.4f}, permutation p = "
      f"{c['p']:.4f}** ({N_PERM} draws). The null **permutes the control "
      "label**: within each clause a reading is drawn at random to play "
      "the role of the identity control and the contrast is recomputed. "
      "That destroys exactly the claim being tested -- that the ordering "
      "follows which reading *is* the clause -- while keeping every "
      "clause's values, size and difficulty. A contrast that survives its "
      "own label permutation is following the reading, not the clause.\n")
    a("### 4.3 Human vs machine\n")
    m = ctx["machine"]
    a("Spearman correlation of the human per-reading mean against #74's "
      f"published `aggregate_fidelity`: **rho = {m['spearman']:+.4f}, "
      f"permutation p = {m['p']:.4f}** over {m['n']} readings.\n")
    a("")
    a("| reading | human mean | #74 aggregate fidelity |")
    a("|---|---:|---:|")
    for p in m["pairs"]:
        a(f"| `{p['reading']}` | {p['human']:.3f} | {p['machine_fidelity']:.3f} |")
    a("")
    a("### 4.4 Inter-reader agreement\n")
    a("| item | alpha | null 5% | null 95% | p |")
    a("|---|---:|---:|---:|---:|")
    for r in ctx["agreement"]:
        a(f"| `{r['item']}` | {r['alpha']:.4f} | {r['null_q05']} | "
          f"{r['null_q95']} | {r['null_p']:.4f} |")
    a("")
    a(f"Krippendorff's alpha with the **{PRIMARY_METRIC}** difference "
      "function (`delta^2 = (c - k)^2`, the pre-registered metric): a 0-7 "
      "rating scale carries equal intervals, so a 6 against a 5 must cost "
      "far less than a 6 against a 0. The **ordinal** and **nominal** "
      "metrics are implemented and disclosed alongside it, because a case "
      "has to be made rather than assumed: with only two categories "
      "observed, the ordinal metric cannot tell a 0-vs-6 disagreement "
      "from a 5-vs-6 one, and the nominal metric prices every "
      "non-identical pair the same -- on a finely-graded scale that "
      "drives alpha toward zero however well the readers agree. Each "
      "alpha is read against a null that re-groups the values (every "
      "reader's scores shuffled across the cells), so a number is only "
      "impressive against what chance produces on the same data. The "
      "implementation is pinned against the worked example in "
      "Krippendorff's 2011 note (binary data, two observers, no missing "
      "values, `o_00=10, o_01=o_10=4, o_11=2` -> alpha = 2/21) by the "
      "regression suite.\n")
    a("**The two trap items sit at the floor of the scale and show no "
      "agreement, and that is a property of the instrument, not of the "
      "readers.** Two reasons, both disclosed: every trap cell carries the "
      "same expectation, so a scale statistic has no between-cell signal to "
      "track; and an expectation of 0.6-0.8 on a 0-7 scale leaves the "
      "responses crowded into 0, 1 and 2, where a seven-point scale has "
      "almost no resolution left to distinguish opinions.  It is exactly "
      "why the traps are analysed as **pass/fail against a known answer** "
      "(the attention check) rather than as a scale -- the design never "
      "asks the traps to carry a scale statistic, and the report does not "
      "pretend otherwise.\n")
    a("### 4.5 Between-reading effect size\n")
    a(f"Kruskal-Wallis H = {ctx['effect']['kruskal_h']:.4f}, "
      f"eta-squared = {ctx['effect']['eta_squared']:.4f} over "
      f"{ctx['effect']['n_readings']} readings -- the non-parametric leg "
      "pre-registered as analysis 6.\n")
    a("### 4.6 Appeal against fidelity\n")
    a("| reading | mean felt fidelity | mean appeal | direction |")
    a("|---|---:|---:|---|")
    appeal_acc: dict = {}
    fidelity_acc: dict = {}
    for r in ctx["ratings"]:
        if r["kind"] == "trap":
            continue
        if r["item"] == PRIMARY_ITEM:
            fidelity_acc.setdefault(r["reading"], []).append(r["score"])
        elif r["item"] == APPEAL_ITEM:
            appeal_acc.setdefault(r["reading"], []).append(r["score"])
    for reading in sorted(fidelity_acc):
        f_m = mean(fidelity_acc[reading])
        a_m = mean(appeal_acc.get(reading, []))
        direction = ("appeal > fidelity" if a_m > f_m + 0.25
                     else "fidelity > appeal" if f_m > a_m + 0.25
                     else "about equal")
        a(f"| `{reading}` | {f_m:.3f} | {a_m:.3f} | {direction} |")
    a("")
    a("This is the table the diagnostic item exists for. A reading that is "
      "more appealing than it is faithful is not a better reading -- it is "
      "the over-correction arm, and the instrument is built so that "
      "preference cannot be mistaken for fidelity.\n")
    a("### 4.7 Per stratum\n")
    a("| stratum | cells | mean felt fidelity | sd |")
    a("|---|---:|---:|---:|")
    for r in ctx["stratum_rows"]:
        a(f"| {r['stratum']} | {r['n_cells']} | "
          f"{r['mean_felt_fidelity']:.3f} | {r['sd']:.3f} |")
    a("")
    a("Reported per stratum and never pooled: #63 stratified the sample so "
      "each check would have a case that can fail, and the study keeps "
      "that property rather than averaging it away.\n")
    a("### 4.8 Reader exclusions\n")
    a("| reader | failed checks | excluded |")
    a("|---|---|---|")
    for r in ctx["flags"]:
        a(f"| {r['reader']} | {r['failed_checks']} | {r['excluded']} |")
    a("")
    a("## 5. What the pilot shows and what it cannot\n")
    a("**What it shows.** The instrument produces interpretable orderings: "
      "the identity control sits above every reading, the ranking tracks "
      "#74's machine-measured fidelity, and the traps and attention checks "
      "separate a reader who is judging fidelity from one who is not (one "
      f"of {N_READERS} simulated readers is flagged by "
      "`att_overcorrected`, which is the check working). The analysis "
      "pipeline -- bootstrap CIs, label-permuted contrast, "
      "label-permuted machine correlation, alpha against a re-grouping null "
      "-- runs end to end on the same code a human run would "
      "use.\n")
    a("**What it cannot show.** Anything about readers. The generative "
      "model is an assumption of this module, not a measurement. In "
      "particular the pilot says nothing about whether real readers agree "
      "with each other, whether they use the whole scale, or whether they "
      "prefer the over-corrected readings -- which is exactly the question "
      "the traps exist to ask. Those need people.\n")
    a("## 6. Anti-pattern audit (L9.8)\n")
    a("* **Hallucinated coherence**: the instrument never asks whether a "
      "reading is coherent; it asks whether the reader can recover the "
      "clause's feeling, with the original clause on screen, and plants a "
      "fluent-but-invented case (`trap_invented`) that must score low.")
    a("* **LLM circularity**: no model generates or grades anything here. "
      "The only LLM output in the frame is #63's pinned capture, treated "
      "as a candidate; the judge is a reader.")
    a("* **Over-correction**: `trap_overcorrected` measures a reader's "
      "appetite for repaired English as data about the reading's appeal, "
      "kept separate from the primary item rather than corrected away.")
    a("* **Single-method certainty**: 7 items, seven pre-registered "
      "analysis legs, three null models, per-stratum reporting and a "
      "published protocol; no single number is the verdict, and the traps "
      "are never asked to carry one.")
    a("* **Residue-dropping**: the frame itemises the cells a block does "
      "not draw, the traps are counted and excluded rather than dropped, "
      "and reader exclusions are published with their reasons.\n")
    a("## 7. Reconciliation\n")
    a("| quantity | earlier | here | note |")
    a("|---|---|---:|---|")
    for r in ctx["reconciliation"]:
        a(f"| {r['quantity']} | {r['earlier']} | {r['here']} | {r['note']} |")
    a("")
    a("## 8. Verification\n")
    a("```bash")
    a("python work/reader_study.py            # writes analysis/reader_study/")
    a("python work/test_reader_study.py       # regression suite")
    a("python -m ruff check work/reader_study.py work/test_reader_study.py")
    a("```\n")
    a("Deterministic: no clock, no network, no model backend, seed "
      f"{SEED}, byte-identical rebuild (only the report's own text moves, "
      "and no statistic reads it).\n")
    a("## 9. Handoff\n")
    a("The next step is recruitment, not design: `protocol.md` is the "
      "pre-registration, `instrument.tsv` the verbatim items, `frame.tsv` "
      "the sampling frame and `blocks.tsv` the counterbalanced packets. A "
      "human run replaces `pilot_ratings.tsv` with its own rows and runs "
      "the same functions; the pilot's numbers are then retired, not "
      "reinterpreted.\n")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="#75 reader study design")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--readers", type=int, default=N_READERS)
    ap.add_argument("--pairs-per-clause", type=int, default=N_PAIRS_PER_CLAUSE)
    ap.add_argument("--quick", action="store_true",
                    help="accepted for CLI symmetry; the design is the whole "
                         "frame, so subsampling it would change the finding "
                         "rather than the cost")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if args.quiet else print

    interlinear = load_interlinear()
    sample = load_sample()
    glosses = load_glosses()
    fidelity = load_fidelity()

    frame = build_frame(interlinear, sample, glosses)
    blocks = assign_blocks(frame, args.pairs_per_clause)
    schedule = block_schedule(blocks)
    ratings = simulate_ratings(blocks, schedule, args.readers, SEED)
    pairs = simulate_pairwise(ratings, schedule, args.readers, SEED)
    flags = attention_flags(
        ratings, list(reader_abilities(args.readers, SEED)))

    check_identity_control(frame, blocks, ratings)

    ctx = analyse(frame, blocks, schedule, ratings, pairs, flags, fidelity,
                   sample, interlinear, glosses, args.readers)
    write_artifacts(out_dir, ctx)
    summary = build_summary(ctx, args.readers)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    (out_dir / "protocol.md").write_text(write_protocol_text(ctx),
                                         encoding="utf-8")
    (out_dir / "reader_study_report.md").write_text(build_report(ctx),
                                                    encoding="utf-8")

    contrast = ctx["contrast"]
    machine = ctx["machine"]
    log(f"[#75] {len(frame)} frame cells ({len(candidate_cells(frame))} "
        f"candidates + {sum(1 for c in frame if c.kind == 'trap')} traps), "
        f"{len(blocks)} blocks, {len(ratings)} simulated ratings")
    log(f"[#75] wrote {out_dir} (schema {SCHEMA}; NO human data)")
    log(f"[#75] identity-vs-glosses contrast {contrast['observed']:+.4f} "
        f"(p={contrast['p']:.4f}); machine/human rho "
        f"{machine['spearman']:+.4f} (p={machine['p']:.4f})")
    return 0


def check_identity_control(frame: list, blocks: list, ratings: list) -> None:
    """The build's own control: the original must top every block.

    If the identity control does not outscore every other reading of its
    clause on the primary item, the instrument cannot separate the original
    from a gloss -- which means it is not measuring fidelity, and the study
    cannot proceed.  A build error, not a footnote.
    """
    cell_rows = per_cell_summary(frame, ratings)
    for b in blocks:
        cid = b["clause_id"]
        rows = [r for r in cell_rows if r["clause_id"] == cid]
        base = [r for r in rows if r["kind"] == "identity"]
        others = [r for r in rows if r["kind"] != "identity"]
        if not base:
            raise SystemExit(f"#75: block {cid} has no identity control")
        if not others:
            raise SystemExit(f"#75: block {cid} has no comparator cells")
        if base[0]["mean"] <= max(r["mean"] for r in others):
            raise SystemExit(
                f"#75: the identity control does not top its block ({cid}: "
                f"{base[0]['mean']:.3f} vs "
                f"{max(r['mean'] for r in others):.3f}) -- the instrument "
                "cannot separate the original from a gloss, so it is not "
                "measuring fidelity")


def analyse(frame: list, blocks: list, schedule: list, ratings: list,
            pairs: list, flags: list, fidelity: dict, sample: list,
            interlinear: dict, glosses: dict, n_readers: int) -> dict:
    """Every statistic the study pre-registers, in one context dict."""
    cell_rows = per_cell_summary(frame, ratings)
    strata = {r["clause_id"]: r["stratum"] for r in sample}
    return {
        "frame": frame, "blocks": blocks, "schedule": schedule,
        "ratings": ratings, "pairs": pairs, "flags": flags,
        "cell_rows": cell_rows,
        "stratum_rows": per_stratum_summary(
            [dict(r, stratum=strata.get(r["clause_id"], ""))
             for r in cell_rows], strata),
        "contrast": contrast_identity_vs_glosses(frame, ratings),
        "machine": machine_human_agreement(cell_rows, fidelity),
        "agreement": agreement_by_item(ratings),
        "effect": reading_effect(cell_rows),
        "pairwise_rows": pairwise_summary(pairs),
        "reconciliation": reconciliation(frame, blocks, sample,
                                         interlinear, glosses, ratings,
                                         pairs),
        "fidelity": fidelity, "interlinear": interlinear, "sample": sample,
        "n_readers": n_readers,
    }


def write_artifacts(out_dir: Path, ctx: dict) -> None:
    """Every artifact except ``summary.json`` and the two markdown files."""
    frame, blocks = ctx["frame"], ctx["blocks"]
    ratings, pairs = ctx["ratings"], ctx["pairs"]
    write_frame_tsv(out_dir / "frame.tsv", frame, blocks, ctx["fidelity"])
    write_instrument_tsv(out_dir / "instrument.tsv")
    write_blocks_tsv(out_dir / "blocks.tsv", blocks, ctx["schedule"])
    write_ratings_tsv(out_dir / "pilot_ratings.tsv", ratings)
    write_pairwise_tsv(out_dir / "pilot_pairwise.tsv", pairs)
    write_tsv(out_dir / "pilot_cell_summary.tsv",
              ["clause_id", "reading", "kind", "n_readers", "mean", "sd",
               "ci_lo", "ci_hi", "identity_mean", "gap_vs_identity"],
              ctx["cell_rows"])
    write_tsv(out_dir / "pilot_stratum_summary.tsv",
              ["stratum", "n_cells", "mean_felt_fidelity", "sd"],
              ctx["stratum_rows"])
    write_tsv(out_dir / "agreement.tsv",
              ["item", "alpha", "null_q05", "null_q95", "null_p", "n_units"],
              ctx["agreement"])
    write_tsv(out_dir / "machine_human.tsv",
              ["reading", "human", "machine_fidelity"], ctx["machine"]["pairs"])
    write_tsv(out_dir / "reading_effect.tsv", ["statistic", "value"],
              [{"statistic": "kruskal_h", "value": ctx["effect"]["kruskal_h"]},
               {"statistic": "eta_squared",
                "value": ctx["effect"]["eta_squared"]},
               {"statistic": "n_readings",
                "value": ctx["effect"]["n_readings"]}])
    write_tsv(out_dir / "pairwise_summary.tsv",
              ["clause_id", "pair_a", "pair_b", "n", "share_a", "share_tie",
               "share_b"], ctx["pairwise_rows"])
    write_tsv(out_dir / "reader_flags.tsv",
              ["reader", "failed_checks", "excluded", "reason"], ctx["flags"])
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], ctx["reconciliation"])
    write_tsv(out_dir / "diagnostics.tsv", ["diag", "value"],
              diagnostics_rows(ctx))


def diagnostics_rows(ctx: dict) -> list:
    """The ``diagnostics.tsv`` rows, including the no-human-data flags."""
    frame, blocks = ctx["frame"], ctx["blocks"]
    ratings, pairs = ctx["ratings"], ctx["pairs"]
    return [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "seed", "value": SEED},
        {"diag": "n_frame_cells", "value": len(frame)},
        {"diag": "n_candidate_cells", "value": len(candidate_cells(frame))},
        {"diag": "n_trap_cells",
         "value": sum(1 for c in frame if c.kind == "trap")},
        {"diag": "n_blocks", "value": len(blocks)},
        {"diag": "n_sampled_pairs", "value": len(sampled_pairs(blocks))},
        {"diag": "n_readers_simulated", "value": ctx["n_readers"]},
        {"diag": "n_scale_ratings", "value": len(ratings)},
        {"diag": "n_pairwise_judgements", "value": len(pairs)},
        {"diag": "primary_item", "value": PRIMARY_ITEM},
        {"diag": "human_data_collected", "value": False},
        {"diag": "pilot_is_simulation", "value": True},
        {"diag": "identity_tops_every_block", "value": True},
        {"diag": "n_readers_flagged",
         "value": sum(1 for f in ctx["flags"] if f["excluded"])},
        {"diag": "n_boot_resamples", "value": N_BOOT},
        {"diag": "n_permutations", "value": N_PERM},
        {"diag": "n_alpha_draws", "value": ALPHA_DRAWS},
    ]


def build_summary(ctx: dict, n_readers: int) -> dict:
    """The machine-readable summary, built from the same context."""
    frame = ctx["frame"]
    return {
        "schema": SCHEMA,
        "seed": SEED,
        "human_data_collected": False,
        "pilot_is_simulation": True,
        "n_frame_cells": len(frame),
        "n_candidate_cells": len(candidate_cells(frame)),
        "n_trap_cells": sum(1 for c in frame if c.kind == "trap"),
        "n_blocks": len(ctx["blocks"]),
        "n_sampled_pairs": len(sampled_pairs(ctx["blocks"])),
        "n_readers_simulated": n_readers,
        "n_scale_ratings": len(ctx["ratings"]),
        "n_pairwise_judgements": len(ctx["pairs"]),
        "primary_item": PRIMARY_ITEM,
        "contrast_identity_vs_glosses": ctx["contrast"],
        "machine_human_agreement": ctx["machine"],
        "agreement_by_item": ctx["agreement"],
        "reading_effect": ctx["effect"],
        "per_cell_primary": ctx["cell_rows"],
        "per_stratum": ctx["stratum_rows"],
        "reader_flags": ctx["flags"],
        "reconciliation": ctx["reconciliation"],
    }


def write_protocol_text(ctx: dict) -> str:
    """The protocol text, generated from the same context as the report."""
    return "\n".join(_protocol_lines(ctx)) + "\n"


def _protocol_lines(ctx: dict) -> list:
    """The protocol body, as a list of lines (``write_protocol`` joins it)."""
    lines: list = []
    _emit_protocol(lines, ctx)
    return lines


if __name__ == "__main__":
    raise SystemExit(main())






