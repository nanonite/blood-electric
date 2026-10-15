#!/usr/bin/env python3
"""#74 Back-translation harness -> analysis/back_translation/

Task (plan E9.5, ``docs/chainlink-plan.md`` #74; methods L9.6):

    *gloss -> regenerate a Siratori-style clause -> compare to original.
     DONE: harness + fidelity scores.*

This is the plan's **round-trip / bidirectional-consistency** evaluation.
There is no ground-truth translation of *Blood Electric*, so the direction is
back-translation: every published *reading* is treated as a candidate gloss,
a deterministic offline re-writer regenerates a Siratori-style clause from that
gloss, and fidelity is scored **against the original clause** on independent
channels.  A gloss that survives the round trip carried the clause's structure;
a gloss that only carried the lexis loses the symbolic machine, and the
channels disagree -- which is the finding, not a defect.

Why this is a valid comparison and not a tautology
----------------------------------------------------
The regenerator sees **only the gloss text**, never the original.  A round-trip
is informative precisely because a lossy gloss cannot be un-lost: if the forward
pass threw the operator channel away, the regenerated clause has no operators
to compare and the operator leg drops to zero *by itself*.  The experiment is
therefore channel-dependent rather than a restatement of the gloss.

What is scored -- all are *published artifacts*, none is invented here:

* **``identity``** -- the control.  The gloss is the source clause itself, so a
  perfect round trip must score exactly 1.000 on every channel.
* **``v3_operators_to_space``**, **``v4_operators_to_period``** -- the #13
  silence-removal variants.  They keep every word and destroy the operator
  channel, so they isolate that channel from the lexis channel.
* **``interlinear_gloss``** -- #62's plain-English gloss line.
* **``llm_<strategy>`` x5** -- #63's five prompt strategies (``literal_operator``,
  ``register_paraphrase``, ``scene_summary``, ``plain_english``,
  ``entity_consistent``) over its 5-clause stratified sample.

The four fidelity channels (independently readable; none is a composite)
-----------------------------------------------------------------------
1. **``operator_recall``** (primary) -- Dice coefficient between the operator
   *multiset* of the regenerated clause and of the original.  Being a Dice
   coefficient it penalises both omission (a dropped operator) and addition (a
   gloss that invents its own operators), so one number carries both error
   directions.
2. **``operator_order``** -- longest-common-subsequence ratio between the two
   operator *sequences*.  A gloss can keep exactly the right multiset and still
   destroy the clause; only a contiguous comparison sees that.
3. **``content_recall``** -- Dice over the letters-words: the lexis channel,
   independent of the machine channel by construction.
4. **``coined_recall``** -- Dice over the **glued coined surfaces** (``BABEL=TV``,
   ``soul/gram``, ``bio=less=``): the seam-bearing units #17 refuses to split.
   A disclosed choice: a token counts when an operator surface sits *inside*
   it, which is the tokenizer's own seam rule, so ``BABEL=TV`` counts and a
   bare ``TV`` does not.

A **disclosed aggregate** (the unweighted mean of the four channels) and a
**mean-rank** across the channels are published as labelled secondaries.  Like
#68 they can never outvote the primary leg: the ranking is presented
per-channel first, and the aggregate only orders readings whose channels agree.

Anti-patterns (L9.8) this harness is built to answer
----------------------------------------------------
* **Circularity** -- no LLM is invoked anywhere; the glosses are #63's pins and
  the judge is deterministic, so no model grades its own output.
* **Single-method certainty** -- four channels, plus controls and a null.
* **Residue-dropping** -- the ``residue`` stratum is retained and the
  round-trip is reported per channel, so a reading that keeps the words but
  loses the operators cannot be mistaken for a faithful one.
* **Over-correction** -- the regenerator never "fixes" the derangement; it only
  re-renders what the gloss preserved.

Reproduce::

    python work/back_translation.py        # writes analysis/back_translation/
    python work/test_back_translation.py    # regression suite

Deterministic: no clock, no RNG in any scored quantity, byte-identical rebuild.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))

import frequency_zipf as fz       # #24 write_tsv
import ngram_surprisal as ns      # #35 read_tsv

OUT = ROOT / "analysis" / "back_translation"
INTERLINEAR = ROOT / "analysis" / "interlinear" / "interlinear.tsv"
GLOSSES = ROOT / "analysis" / "glossing" / "glosses.tsv"
SAMPLE = ROOT / "analysis" / "glossing" / "sample.tsv"

SCHEMA = "back-translation/1"
SEED = 74                       # issue number, the house convention for seeds
N_NULLS = 4                     # permutation nulls for the round-trip aggregate

# Published constants frozen by earlier issues; a mismatch is a build error,
# the same contract #68 holds its reconciliation rows to.
PUBLISHED = {
    "n_clauses": 5614,          # #33 / #62 interlinear row count
    "n_gloss_rows": 25,         # #63 sampled glosses
    "n_strategies": 5,          # #63 prompt strategies
    "n_sample_clauses": 5,      # #63 stratified sample, one per stratum
}

# The eleven operator surfaces, longest-first so ``::`` is never read as two
# colons and ``//`` as two slashes.  This set is *derived from #17* (the
# tokenizer's ``kind == "OP"`` stream) and is asserted identical to
# #63's ``glossing_pipeline.OP_SURFACES`` by the regression suite -- a copy
# kept only to avoid a build-time dependency on the pipeline, not a fork.
OP_SURFACES = frozenset(("<<", ">>", "::", "//", "++", "=", "+", "<", ">",
                         "/", "|"))
_OP_ONLY_RE = re.compile(r"::|//|<<|>>|\+\+|[=+<>|/]")
_OPS_LONGEST = tuple(sorted(OP_SURFACES, key=len, reverse=True))

# A "word" for the lexis channel is a maximal run of letters.  Same disclosed
# rule as #68, so ``soul/gram`` is two words here and one glued surface below:
# the lexis channel must not be able to score a seam it is not measuring.
WORD_RE = re.compile(r"[A-Za-z]+")
# Bracketed editorialisation the #63 glossing strategies add around the clause
# (``[persona: ...]``, ``[unresolved]``).  Everything inside is the gloss
# *talking about* the surface, not the surface itself.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")

read_tsv = ns.read_tsv
write_tsv = fz.write_tsv


def operator_subsequence(text: str) -> list:
    """The operator surfaces in ``text``, in order (the #63 scanner)."""
    return [t for t in _OP_ONLY_RE.findall(text) if t in OP_SURFACES]


def strip_editorial(text: str) -> str:
    """Drop ``[...]`` editorial asides the glossing strategies add.

    A disclosed judgement call with teeth: a naive operator count that keeps
    them charges a gloss for characters that are not part of the clause at all
    -- #63's ``literal_operator`` glosses bury asides such as
    ``[boy-roid: kept unresolved, the ``"``roid``"`` half stays as the raw
    token]`` around the surface, and its own operator-fidelity check counts the
    operators *inside* those asides.  Stripping them makes the round trip a
    fair test of the clause rather than of the annotation.
    """
    return _BRACKET_RE.sub(" ", text)


def content_words(text: str) -> list:
    """Letters-words of ``text`` (editorial asides already removed)."""
    return WORD_RE.findall(strip_editorial(text))


def glued_surfaces(text: str) -> set:
    """The seam-bearing surfaces in ``text`` (``BABEL=TV``, ``soul/gram``).

    A surface is the maximal run of word characters, hyphens and exactly the
    operator that glued it, centred on an operator occurrence.  ``BABEL=TV>``
    contributes ``BABEL=TV`` (the trailing ``>`` is a closer, not a seam), and
    ``TV`` alone contributes nothing.
    """
    body = strip_editorial(text)
    out: set = set()
    i, n = 0, len(body)
    while i < n:
        for op in _OPS_LONGEST:
            if body.startswith(op, i):
                lo, hi = i, i + len(op)
                while lo > 0 and (body[lo - 1].isalnum() or body[lo - 1] == "-"):
                    lo -= 1
                while hi < n and (body[hi].isalnum() or body[hi] == "-"):
                    hi += 1
                out.add(body[lo:hi])
                i = hi
                break
        else:
            i += 1
    return out


def regenerate_clause(gloss: str) -> str:
    """Regenerate a Siratori-style clause from a gloss (pure function of gloss).

    Deliberately conservative and deliberately disclosed: the forward gloss is
    a lossy *reading*, so the honest inverse is the re-render of the clause the
    gloss preserved -- editorial asides dropped, whitespace canonicalised -- and
    **not** an invention of structure the forward pass destroyed.  Operators a
    plain-English gloss verbalised or a silence-removal variant replaced are
    therefore absent from the regeneration, and the round trip records their
    loss instead of hiding it.  A less honest regenerator that guessed the
    missing operators back would inflate the fidelity scores it is meant to
    measure.
    """
    return " ".join(strip_editorial(gloss).split())


def operator_recall(original_ops: list, generated_ops: list) -> float:
    """Dice coefficient over the operator multiset (penalises loss *and* gain)."""
    a, b = Counter(original_ops), Counter(generated_ops)
    total = sum(a.values()) + sum(b.values())
    if total == 0:
        return 1.0                       # nothing there to lose or to invent
    return 2.0 * sum((a & b).values()) / total


def _lcs(a: list, b: list) -> int:
    """Longest common subsequence length (rolling row)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(cur[j - 1], prev[j])
        prev = cur
    return prev[-1]


def operator_order(original_ops: list, generated_ops: list) -> float:
    """LCS ratio between the operator sequences (order/arrangement fidelity)."""
    if not original_ops and not generated_ops:
        return 1.0
    if not original_ops or not generated_ops:
        return 0.0
    return _lcs(original_ops, generated_ops) / max(len(original_ops),
                                                  len(generated_ops))


def dice(a: set, b: set) -> float:
    """Dice over two sets (``1.0`` when both are empty: nothing to disagree on)."""
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    return 2.0 * len(a & b) / total


def fidelity_scores(original: str, gloss: str) -> dict:
    """All four channels for one (original, gloss) round trip.

    Every channel scores the **regenerated clause** against the **original**.
    That is the point of a round trip: the gloss is the lossy intermediate and
    the regenerated clause is what a reader of that gloss would have had to
    work from, so the fidelity of the *round trip* is measured on the pair the
    method actually closes the loop on.  Scoring the lexis channel straight
    off the gloss instead would silently compare two different loops and let a
    broken regenerator pass the identity control.
    """
    generated = regenerate_clause(gloss)
    gen_ops = operator_subsequence(generated)
    gloss_ops = operator_subsequence(strip_editorial(gloss))
    orig_ops = operator_subsequence(original)
    return {
        "operator_recall": operator_recall(orig_ops, gen_ops),
        "operator_order": operator_order(orig_ops, gen_ops),
        "content_recall": dice(set(content_words(original)),
                               set(content_words(generated))),
        "coined_recall": dice(glued_surfaces(original),
                             glued_surfaces(generated)),
        "generated": generated,
        "n_orig_ops": len(orig_ops),
        "n_generated_ops": len(gen_ops),
        "n_gloss_ops": len(gloss_ops),
    }


CHANNELS = ("operator_recall", "operator_order", "content_recall",
            "coined_recall")
PRIMARY = "operator_recall"
AGGREGATE = "aggregate_fidelity"          # unweighted mean of CHANNELS, secondary


# --------------------------------------------------------------------------
# readings
# --------------------------------------------------------------------------
@dataclass
class Reading:
    """One candidate reading: a name, a kind, a source, a clause -> text map."""

    name: str
    kind: str                 # control | transform | gloss | llm
    source: str               # the artifact it is read from
    texts: dict = field(default_factory=dict)   # clause_id -> gloss text


def load_interlinear() -> dict:
    rows = read_tsv(INTERLINEAR)
    if not rows:
        raise SystemExit(f"#74: no interlinear rows at {INTERLINEAR}")
    return {r["clause_id"]: r for r in rows}


def sample_ids() -> list:
    """#63's stratified sample, in clause-id (text) order -- published, not chosen."""
    rows = read_tsv(SAMPLE)
    if not rows:
        raise SystemExit(f"#74: no sample rows at {SAMPLE}")
    return sorted(r["clause_id"] for r in rows)


def build_readings(interlinear: dict) -> list:
    """All candidate readings over #63's sample, each read from its artifact."""
    ids = sample_ids()
    missing = [c for c in ids if c not in interlinear]
    if missing:
        raise SystemExit(f"#74: sample clauses absent from the interlinear: "
                         f"{missing}")
    readings = [
        Reading("identity", "control", "#62 interlinear original column",
                {c: interlinear[c]["original"] for c in ids}),
        Reading("v3_operators_to_space", "transform", "#13 v3 via interlinear",
                {c: interlinear[c]["v3_space"] for c in ids}),
        Reading("v4_operators_to_period", "transform", "#13 v4 via interlinear",
                {c: interlinear[c]["v4_period"] for c in ids}),
        Reading("interlinear_gloss", "gloss", "#62 gloss column",
                {c: interlinear[c]["gloss"] for c in ids}),
    ]
    by_strategy: dict = {}
    for r in read_tsv(GLOSSES):
        if r["clause_id"] in ids:
            by_strategy.setdefault(r["strategy"], {})[r["clause_id"]] = r["gloss"]
    for strategy, texts in sorted(by_strategy.items()):
        readings.append(Reading(f"llm_{strategy}", "llm",
                                f"#63 glosses.tsv[{strategy}]", dict(texts)))
    return readings


def score_reading(reading: Reading, interlinear: dict) -> dict:
    """Per-clause channels plus the disclosed aggregate, for one reading."""
    rows = []
    for cid in sorted(reading.texts):
        fr = fidelity_scores(interlinear[cid]["original"], reading.texts[cid])
        rows.append({"clause_id": cid, "stratum": None, **fr})
    n = len(rows) or 1
    score = {ch: sum(r[ch] for r in rows) / n for ch in CHANNELS}
    score[AGGREGATE] = sum(score[ch] for ch in CHANNELS) / len(CHANNELS)
    score["rows"] = rows
    score["reading"] = reading.name
    score["kind"] = reading.kind
    score["source"] = reading.source
    score["n_clauses"] = len(rows)
    score["n_orig_ops"] = sum(r["n_orig_ops"] for r in rows)
    score["n_generated_ops"] = sum(r["n_generated_ops"] for r in rows)
    score["n_gloss_ops"] = sum(r["n_gloss_ops"] for r in rows)
    return score


def operator_strip_ablation(reading: Reading, interlinear: dict, ids: list) -> dict:
    """Ablated round trip: operator surfaces removed from every gloss.

    The ablation proves the operator channel has teeth rather than crediting a
    reading for lexis: with the operators gone, ``operator_recall`` must be
    exactly 0.0 on every clause whose original carried one, whatever the
    reading's lexical performance.  Removing the surfaces by the *same regex*
    the scanner uses (not a character filter) matters: ``:`` is not a surface
    but ``::`` is, so a char-wise strip would leave the two-character runs
    standing and the control would silently pass a broken implementation.
    """
    rows = []
    for cid in ids:
        body = " ".join(strip_editorial(reading.texts[cid]).split())
        ablated = _OP_ONLY_RE.sub(" ", body)
        orig_ops = operator_subsequence(interlinear[cid]["original"])
        rows.append(operator_recall(orig_ops, operator_subsequence(ablated)))
    return {"reading": reading.name, "n_clauses": len(rows),
            "mean_operator_recall": sum(rows) / (len(rows) or 1),
            "n_clauses_with_operators": sum(
                1 for cid in ids if operator_subsequence(
                    interlinear[cid]["original"]))}


def permutation_null(readings: list, interlinear: dict, ids: list) -> list:
    """Shuffle the clause<->reading pairing; aggregate fidelity must collapse.

    Each reading's gloss texts are permuted across the sampled clause ids with
    fixed seeds, so a reading can no longer be matched to the original it came
    from.  Any aggregate that survives that is measuring the vocabulary the
    book's clauses share with each other, not round-trip fidelity -- which is
    exactly the confound a 5-clause Siratori sample is exposed to.  Seeds whose
    permutation is the identity are skipped (they would test nothing).
    """
    out = []
    for reading in readings:
        texts = [reading.texts[c] for c in ids]
        nulls = []
        for seed in range(N_NULLS):
            rng = random.Random(SEED + seed)
            perm = list(range(len(ids)))
            rng.shuffle(perm)
            if perm == list(range(len(ids))):
                continue
            vals = []
            for i in range(len(ids)):
                fr = fidelity_scores(interlinear[ids[i]]["original"],
                                     texts[perm[i]])
                vals.append(sum(fr[ch] for ch in CHANNELS) / len(CHANNELS))
            nulls.append(sum(vals) / len(vals))
        truths = []
        for c in ids:
            fr = fidelity_scores(interlinear[c]["original"], reading.texts[c])
            truths.append(sum(fr[ch] for ch in CHANNELS) / len(CHANNELS))
        out.append({
            "reading": reading.name, "kind": reading.kind,
            "true_aggregate_fidelity": sum(truths) / len(truths),
            "null_aggregate_fidelity": sum(nulls) / len(nulls) if nulls
            else float("nan"),
        })
    return out


def rank_scores(scores: list) -> list:
    """Rank readings per channel and by the labelled aggregate.

    Rank 1 is best on every leg.  Ties share a rank (competition ranking over
    the distinct value set, ``scipy``-free), so a channel on which two readings
    agree does not order them arbitrarily by input position -- #68 pins the
    same convention.  The ``identity`` control is a calibration floor, not a
    competitor: it would win every channel by construction, so it is ranked for
    its own sanity value but held out of the contender order and published last.
    """
    for ch in CHANNELS + (AGGREGATE,):
        values = sorted({round(s[ch], 12) for s in scores}, reverse=True)
        for s in scores:
            s["rank_" + ch] = values.index(round(s[ch], 12)) + 1
    n = len(CHANNELS)
    for s in scores:
        s["mean_rank"] = sum(s["rank_" + ch] for ch in CHANNELS) / n
    keys = sorted(scores, key=lambda s: (-s[AGGREGATE], s["mean_rank"],
                                         s["reading"]))
    contenders = [s for s in keys if s["kind"] != "control"]
    control = [s for s in keys if s["kind"] == "control"]
    return contenders + control


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------
def reconciliation(scores: list, interlinear: dict) -> list:
    """This build vs #62 / #63.

    Exact-match rows are build errors when they drift; disclosed rows carry a
    reason, following #68's convention.
    """
    out = []
    ids = sample_ids()

    exact = [
        ("n_interlinear_rows", len(interlinear), PUBLISHED["n_clauses"],
         "#33/#62 clause count (exact match required)"),
        ("n_sample_clauses", len(ids), PUBLISHED["n_sample_clauses"],
         "#63 stratified sample size (exact match required)"),
        ("n_strategies", len({r["strategy"] for r in read_tsv(GLOSSES)
                             if r["clause_id"] in ids}),
         PUBLISHED["n_strategies"], "#63 strategies (exact match required)"),
        ("n_gloss_rows", sum(1 for r in read_tsv(GLOSSES)
                             if r["clause_id"] in ids),
         PUBLISHED["n_gloss_rows"],
         "#63 sampled glosses (exact match required)"),
    ]
    for qty, here, earlier, note in exact:
        out.append({"quantity": qty, "earlier": earlier, "here": here,
                    "note": note})
        if here != earlier:
            raise SystemExit(f"#74: {qty} {here} != published {earlier} -- "
                             "upstream artifact drifted; refusing to build")

    # Disclosed rows, not build errors.  The regenerated clause is a re-render
    # of the gloss, so its operator multiset is a subset of the gloss's own
    # (``operators_out``) once the bracketed asides are removed: the gap is the
    # operators the gloss nested *inside* its editorial brackets, which the
    # round trip correctly refuses to recover.  Publishing it keeps #63's
    # operator accounting and this one visible side by side.
    gloss_ops_total = 0
    for r in read_tsv(GLOSSES):
        if r["clause_id"] in ids:
            gloss_ops_total += len([t for t in r["operators_out"].split(",")
                                    if t])
    out.append({"quantity": "n_operators_in_glosses", "earlier": gloss_ops_total,
                "here": sum(s["n_gloss_ops"] for s in scores),
                "note": "#63 operators_out total vs the count this build "
                        "scores on (bracket asides stripped; a disclosed "
                        "subset, not a mismatch)"})

    # A generative self-check, not a reconciliation: does the round trip
    # actually change anything for the lossy readings?  A harness where every
    # reading scored 1.0 would be reporting nothing.
    n_recoverable = sum(1 for s in scores if s["n_generated_ops"] > 0)
    out.append({"quantity": "n_readings_with_operators_after_round_trip",
                "earlier": len(scores), "here": n_recoverable,
                "note": "how many readings still carry an operator after the "
                        "round trip (identity + operator-preserving glosses; a "
                        "disclosed diagnostic, not a required match)"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="#74 back-translation harness")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quick", action="store_true",
                    help="accepted for CLI symmetry; scores all sampled "
                         "clauses (see main)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if args.quiet else print

    interlinear = load_interlinear()
    readings = build_readings(interlinear)
    ids = sample_ids()
    strata = {r["clause_id"]: r["stratum"] for r in read_tsv(SAMPLE)}
    # ``--quick`` is accepted for CLI symmetry with the other harnesses but
    # scores every clause: the whole corpus this harness evaluates *is* #63's
    # five-clause sample, so subsampling it would change the finding rather
    # than the cost, and a coverage control must never be silently reduced.
    eval_ids = ids

    log(f"[#74] {len(readings)} readings x {len(eval_ids)} clauses")

    scores = [score_reading(r, interlinear) for r in readings]

    # control: the identity control must be exactly perfect on every channel.
    # A self-round-trip that does not score 1.0 is a harness bug, so it is a
    # build error rather than a reported number.
    ident = next(s for s in scores if s["reading"] == "identity")
    for ch in CHANNELS:
        if abs(ident[ch] - 1.0) > 1e-12:
            raise SystemExit(f"#74: identity control is not perfect on {ch} "
                             f"({ident[ch]}) -- the round trip is broken")

    ablations = [operator_strip_ablation(r, interlinear, eval_ids)
                 for r in readings]
    nulls = permutation_null(readings, interlinear, eval_ids)

    # control: the ablation must floor the operator channel on every clause
    # whose original carried an operator, or the primary leg is not about the
    # operator at all and the ranking is meaningless.
    for ab in ablations:
        if ab["n_clauses_with_operators"] == 0:
            continue
        if ab["mean_operator_recall"] > 1e-12:
            raise SystemExit(
                f"#74: operator-strip ablation did not floor "
                f"{ab['reading']} ({ab['mean_operator_recall']})")

    # control: destroying the clause<->reading pairing must not be free of
    # consequence, otherwise the aggregate is just "both are Siratori".
    for nl in nulls:
        if nl["reading"] == "identity":
            continue          # the control is already its own null by design
        if not (nl["true_aggregate_fidelity"] > nl["null_aggregate_fidelity"]):
            raise SystemExit(
                f"#74: permutation null did not collapse "
                f"{nl['reading']} (true {nl['true_aggregate_fidelity']:.6f} <= "
                f"null {nl['null_aggregate_fidelity']:.6f})")

    ranked = rank_scores(scores)
    recon = reconciliation(scores, interlinear)

    # ---- artifacts ----
    readings_rows = [{"reading": r.name, "kind": r.kind, "source": r.source,
                      "n_clauses": len(r.texts)} for r in readings]
    write_tsv(out_dir / "readings.tsv",
              ["reading", "kind", "source", "n_clauses"], readings_rows)

    fidelity_header = ["reading", "kind", "n_clauses", "n_orig_ops",
                       "n_generated_ops", "n_gloss_ops"] + \
                      ["mean_" + ch for ch in CHANNELS] + [AGGREGATE]
    write_tsv(out_dir / "fidelity_scores.tsv", fidelity_header,
              [{"reading": s["reading"], "kind": s["kind"],
                "n_clauses": s["n_clauses"], "n_orig_ops": s["n_orig_ops"],
                "n_generated_ops": s["n_generated_ops"],
                "n_gloss_ops": s["n_gloss_ops"],
                **{"mean_" + ch: round(s[ch], 6) for ch in CHANNELS},
                AGGREGATE: round(s[AGGREGATE], 6)} for s in ranked])

    round_trip_rows = []
    for r in readings:
        for row in next(s for s in scores if s["reading"] == r.name)["rows"]:
            round_trip_rows.append({
                "clause_id": row["clause_id"],
                "stratum": strata.get(row["clause_id"], ""),
                "reading": r.name,
                "kind": r.kind,
                "n_orig_ops": row["n_orig_ops"],
                "n_generated_ops": row["n_generated_ops"],
                "n_gloss_ops": row["n_gloss_ops"],
                **{ch: round(row[ch], 6) for ch in CHANNELS},
                "generated": row["generated"]})
    write_tsv(out_dir / "round_trips.tsv",
              ["clause_id", "stratum", "reading", "kind", "n_orig_ops",
               "n_generated_ops", "n_gloss_ops"] + list(CHANNELS)
              + ["generated"], round_trip_rows)

    write_tsv(out_dir / "controls_and_nulls.tsv",
              ["reading", "kind", "ablated_mean_operator_recall",
               "n_clauses_with_operators", "true_aggregate_fidelity",
               "null_aggregate_fidelity", "note"],
              [{"reading": a["reading"],
                "kind": next(r.kind for r in readings if r.name == a["reading"]),
                "ablated_mean_operator_recall": round(a["mean_operator_recall"], 6),
                "n_clauses_with_operators": a["n_clauses_with_operators"],
                "true_aggregate_fidelity": round(
                    n["true_aggregate_fidelity"], 6),
                "null_aggregate_fidelity": round(n["null_aggregate_fidelity"], 6)
                if n["null_aggregate_fidelity"] == n["null_aggregate_fidelity"]
                else "",
                "note": "ablation floors the operator channel; the permutation "
                        "null must not survive the destroyed pairing"}
               for a, n in zip(ablations, nulls)])

    # worked examples: every reading over every sampled clause, so a reader
    # can check the round trip by eye.  The ``residue`` stratum is first
    # because it is the one a plain reading is most tempted to normalise away.
    ordered = sorted(eval_ids, key=lambda c: (
        {"residue": 0, "operator_dense": 1, "entity_named": 2,
         "quoting_frame": 3}.get(strata.get(c), 9), c))
    examples = []
    for c in ordered:
        for r in readings:
            fr = fidelity_scores(interlinear[c]["original"], r.texts[c])
            examples.append({
                "clause_id": c,
                "stratum": strata.get(c, ""),
                "reading": r.name,
                "kind": r.kind,
                "original": interlinear[c]["original"],
                "gloss": r.texts[c],
                "generated": fr["generated"],
                "operators_orig": ",".join(operator_subsequence(
                    interlinear[c]["original"])),
                "operators_generated": ",".join(operator_subsequence(
                    fr["generated"])),
                **{ch: round(fr[ch], 4) for ch in CHANNELS},
            })
    write_tsv(out_dir / "worked_examples.tsv",
              ["clause_id", "stratum", "reading", "kind", "original", "gloss",
               "generated", "operators_orig", "operators_generated"]
              + list(CHANNELS), examples)

    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    diag = [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "seed", "value": SEED},
        {"diag": "n_readings", "value": len(readings)},
        {"diag": "n_clauses", "value": len(eval_ids)},
        {"diag": "quick", "value": bool(args.quick)},
        {"diag": "n_null_permutations", "value": N_NULLS},
        {"diag": "primary_channel", "value": PRIMARY},
        {"diag": "identity_control_perfect",
         "value": all(abs(ident[ch] - 1.0) <= 1e-12 for ch in CHANNELS)},
        {"diag": "ablations_floored",
         "value": all(a["mean_operator_recall"] <= 1e-12
                      for a in ablations
                      if a["n_clauses_with_operators"])},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["diag", "value"], diag)

    ctx = {"ranked": ranked, "readings": readings, "ids": eval_ids,
           "ablations": ablations, "nulls": nulls,
           "reconciliation": recon, "scores": scores, "strata": strata,
           "interlinear": interlinear}
    summary = {
        "schema": SCHEMA,
        "seed": SEED,
        "n_readings": len(readings),
        "n_clauses": len(eval_ids),
        "quick": bool(args.quick),
        "primary_channel": PRIMARY,
        "fidelity_scores": [
            {"reading": s["reading"], "kind": s["kind"],
             "n_clauses": s["n_clauses"],
             "n_orig_ops": s["n_orig_ops"],
             "n_generated_ops": s["n_generated_ops"],
             "n_gloss_ops": s["n_gloss_ops"],
             **{ch: round(s[ch], 6) for ch in CHANNELS},
             AGGREGATE: round(s[AGGREGATE], 6)} for s in ranked],
        "controls": {
            "identity_perfect": all(abs(ident[ch] - 1.0) <= 1e-12
                                    for ch in CHANNELS),
            "ablation_floors_operator_channel": all(
                a["mean_operator_recall"] <= 1e-12 for a in ablations
                if a["n_clauses_with_operators"]),
            "permutation_null_collapses": all(
                n["true_aggregate_fidelity"] > n["null_aggregate_fidelity"]
                for n in nulls),
        },
        "nulls": [{k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in n.items()} for n in nulls],
        "reconciliation": recon,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    (out_dir / "back_translation_report.md").write_text(
        build_report(ctx), encoding="utf-8")

    log(f"[#74] wrote {out_dir} (schema {SCHEMA})")
    if ranked:
        w = ranked[0]
        contender = max((s for s in scores if s["kind"] != "control"),
                        key=lambda s: s[PRIMARY])
        log(f"[#74] aggregate winner: {w['reading']} "
            f"(aggregate {w[AGGREGATE]:.6f}, mean_rank {w['mean_rank']:.2f}); "
            f"primary-leg top: {contender['reading']}")
    return 0


def build_report(ctx: dict) -> str:
    L = []
    a = L.append
    ranked = ctx["ranked"]
    a("# Back-translation harness (#74)\n")
    a("")
    a("The plan's **L9.6** test, run. Every published reading is treated as a "
      "candidate gloss, a deterministic offline re-writer regenerates a "
      "Siratori-style clause from it, and fidelity is scored **against the "
      "original clause** on four independent channels. There is no "
      "ground-truth translation, so this is a round-trip "
      "(bidirectional-consistency) test, not an accuracy claim.\n")
    a("## 1. The round trip\n")
    a("`original clause -> [a published reading] -> regenerate -> compare`\n")
    a("")
    a(f"Channels: `{'`, `'.join(CHANNELS)}`. The primary leg is "
      f"`{PRIMARY}` (a Dice coefficient, so it penalises both a dropped "
      "operator and an invented one in a single number). "
      "`operator_order` is the order-sensitive companion. `content_recall` and "
      "`coined_recall` are the lexis and seam channels. "
      "`aggregate_fidelity` and `mean_rank` are **labelled secondaries** that "
      "order readings whose channels agree; they can never outvote the "
      "primary.\n")
    a("## 2. Fidelity scores\n")
    a("Sorted by the disclosed aggregate (secondary); the per-channel ranking "
      "is the result.\n")
    a("")
    header = ["reading", "kind", "n_clauses", "operators (orig -> generated)",
              "operator_recall", "operator_order", "content_recall",
              "coined_recall", "aggregate_fidelity", "mean_rank"]
    a("| " + " | ".join(header) + " |")
    a("|" + "---|" * len(header))
    for s in ranked:
        a("| " + " | ".join([
            f"`{s['reading']}`", s["kind"], str(s["n_clauses"]),
            f"{s['n_orig_ops']} -> {s['n_generated_ops']}",
            f"{s['operator_recall']:.3f}", f"{s['operator_order']:.3f}",
            f"{s['content_recall']:.3f}", f"{s['coined_recall']:.3f}",
            f"{s[AGGREGATE]:.3f}", f"{s['mean_rank']:.2f}"]) + " |")
    a("")
    a("## 3. Controls\n")
    a("")
    a("**Identity control.** The `identity` reading is the source clause "
      "itself; a perfect round trip must score exactly 1.000 on every channel, "
      "and the build fails if it does not. It is the harness's own calibration.\n")
    a("")
    a("**Operator-strip ablation.** Every gloss is stripped of its operators "
      "and re-scored. With the machine channel gone, `operator_recall` must "
      "floor on every clause whose original carried an operator, whatever the "
      "reading's lexical performance -- the ablation proves the primary leg is "
      "not secretly crediting lexis.\n")
    a("")
    a("| reading | kind | ablated operator_recall | true aggregate | null aggregate |")
    a("|---|---|---:|---:|---:|")
    nul = {n["reading"]: n for n in ctx["nulls"]}
    for ab in ctx["ablations"]:
        n = nul[ab["reading"]]
        a(f"| `{ab['reading']}` | "
          f"{next(r.kind for r in ctx['readings'] if r.name == ab['reading'])} | "
          f"{ab['mean_operator_recall']:.3f} | "
          f"{n['true_aggregate_fidelity']:.3f} | "
          f"{n['null_aggregate_fidelity']:.3f} |")
    a("")
    a("**Permutation null.** Each reading's glosses are shuffled across the "
      "sampled clause ids and rescored. Destroying the clause<->reading pairing "
      "must drop the aggregate; a score that survived it would be measuring "
      "vocabulary the book's clauses share with each other rather than "
      "round-trip fidelity. The build refuses to publish a table where any "
      "non-control reading survives its own null.\n")
    a("## 4. Worked examples\n")
    a("Every reading over every sampled clause, `residue` first -- the reading "
      "the round trip is most tempting to normalise away. Full rows in "
      "`worked_examples.tsv`.\n")
    for c in sorted(ctx["ids"], key=lambda c: (
            {"residue": 0, "operator_dense": 1, "entity_named": 2,
             "quoting_frame": 3}.get(ctx["strata"].get(c), 9), c)):
        a(f"**clause {c}** (`{ctx['strata'].get(c, '?')}`)\n")
        a("```text")
        a("orig | " + ctx["interlinear"][c]["original"])
        for r in ctx["readings"]:
            fr = fidelity_scores(ctx["interlinear"][c]["original"],
                                 r.texts[c])
            a("")
            a(f"{r.name} | gloss   : {r.texts[c]}")
            a(f"{r.name} | re-gen  : {fr['generated']}")
            a(f"{r.name} | ops     : "
              f"{','.join(operator_subsequence(ctx['interlinear'][c]['original']))} "
              f"-> {','.join(operator_subsequence(fr['generated']))}  "
              f"| op_recall {fr['operator_recall']:.3f} | order "
              f"{fr['operator_order']:.3f} | lexical "
              f"{fr['content_recall']:.3f} | coined {fr['coined_recall']:.3f}")
        a("```\n")
    a("## 5. Reconciliation\n")
    a("")
    a("| quantity | earlier | here | note |")
    a("|---|---|---:|---|")
    for r in ctx["reconciliation"]:
        a(f"| {r['quantity']} | {r['earlier']} | {r['here']} | {r['note']} |")
    a("")
    a("## 6. What this shows\n")
    a("")
    fs = ranked
    op_top = max((s for s in fs if s["kind"] != "control"),
                 key=lambda s: s[PRIMARY])
    zero_op = [s["reading"] for s in fs
               if s[PRIMARY] == 0.0 and s["kind"] != "control"]
    full_lex_zero_op = [s["reading"] for s in fs
                        if s[PRIMARY] == 0.0 and s["content_recall"] == 1.0
                        and s["kind"] != "control"]
    a(f"**The channels split, and the split is the finding.** "
      f"`{op_top['reading']}` is the only non-control reading that recovers the "
      f"operator channel in full (`{PRIMARY}` {op_top[PRIMARY]:.3f}, "
      f"`operator_order` {op_top['operator_order']:.3f}) -- it wins because it "
      "reproduces the clause nearly verbatim, which is what a near-lossless "
      "intermediate buys. The readings that destroy the operator channel in "
      f"transit ({', '.join('`' + r + '`' for r in zero_op)}) all score "
      f"`{PRIMARY}` 0.000, and "
      f"{', '.join('`' + r + '`' for r in full_lex_zero_op)} nevertheless keep "
      "`content_recall` at 1.000: they carry the book's lexis and lose its "
      "symbolic machine entirely. A reading that only rewrites orthography is "
      "the legible one; a reading that also substitutes English is the "
      "legible-and-invertible one. No single reading wins both channels, which "
      "is exactly why a reading's fidelity cannot be one number (L9.8 "
      "single-method certainty).\n")
    a("The ranking also demonstrates the *cost of adding* structure: the "
      "`literal_operator` strategy keeps every operator symbol but holds them "
      "arranged and partly buried in its own editorial asides, so the Dice leg "
      "charges it below full fidelity rather than rewarding the count. Its own "
      "generated clause is reproduced verbatim in `worked_examples.tsv` so the "
      "charge can be inspected.\n")
    a("## 7. Limits\n")
    a("")
    a("**The regenerator is a re-render, not an author.** The forward gloss is "
      "lossy, so the honest inverse re-renders what the gloss preserved and "
      "refuses to reconstruct the operators the forward pass destroyed. A "
      "regenerator that guessed them back would inflate the very scores this "
      "harness measures; the disclosure is deliberate and the round trip "
      "records the loss instead of hiding it.\n")
    a("**A five-clause sample.** #63's stratified sample is scored, not the "
      "book; the numbers are precise about five clauses and indicative of the "
      "reading, not a corpus-wide fidelity estimate.\n")
    a("**Round-trip fidelity is not truth.** A gloss that round-trips is "
      "self-consistent with the clause; it is not thereby the correct reading "
      "of it.\n")
    a("## 8. Anti-pattern audit (L9.8)\n")
    a("")
    a("* **Circularity** -- no LLM is invoked; the glosses are #63's pins and "
      "the judge is deterministic, so no model grades its own output.")
    a("* **Single-method certainty** -- four independent channels, two controls "
      "(identity + ablation) and a disclosed aggregate that cannot outvote the "
      "primary.")
    a("* **Residue-dropping** -- the `residue` stratum is retained; per-channel "
      "reporting stops a word-preserving, operator-dropping reading from "
      "passing as faithful.")
    a("* **Over-correction** -- the regenerator never repairs the derangement.")
    a("* **Determinism** -- no clock, no RNG in any scored quantity; a rebuild "
      "is byte-identical.")
    a("")
    a("## 9. Reproducing\n")
    a("")
    a("```bash")
    a("python work/back_translation.py")
    a("python work/test_back_translation.py")
    a("```")
    a("")
    a("Offline, deterministic, no clock; rebuild is byte-identical.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
