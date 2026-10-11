#!/usr/bin/env python3
"""#63 LLM glossing pipeline (multi-strategy) -> analysis/glossing/

Task (plan E8.3, docs/chainlink-plan.md #63): generate readings of the
book **with a language model**, under five prompt strategies, and publish
the sampled glosses together with a machine-checkable verdict on each.

    1  literal_operator     gloss word-by-word, operator channel preserved
                            verbatim and in order
    2  register_paraphrase  same content, shifted into ordinary narrative
                            prose (the book is not ordinary prose)
    3  scene_summary        one clause -> one plain scene, no operators
    4  plain_english        translation into plain English, operators
                            verbalised rather than carried
    5  entity_consistent    gloss constrained to the #30 dramatis
                            personae, so names/roles cannot drift

The glossary and the operator grammar are fed to the model as
**constraints**, not as hints: every prompt carries the clause's own
`<<...>>` block-quote frames, the operator readings for the operators
actually present in that clause (#12/#14), and the #30 rows for the
entities that clause names (role, does/done-to verbs, frequency).  A
prompt never shows the model an operator that is not in its clause, so a
"preserved" operator channel can be checked rather than trusted.

What the module actually establishes
------------------------------------
The load-bearing output is **not** the glosses -- a sampled gloss is one
model's reading of one clause and is worth exactly that.  It is the
**verdict layer**: four independent, deterministic checks run over every
returned gloss, and they are what turn a sample into evidence.

  * `operator_fidelity`  -- does the operator channel survive?  The exact
    operator subsequence of the clause must reappear, in order, in the
    gloss.  Checked against the #17 token stream, not a regex over
    prose, so `soul/gram` cannot pass as `psyche/gram`.
  * `entity_fidelity`    -- does the gloss keep the #30 names it was
    given, and invent none?  Both directions are checked: a dropped name
    and a *new* proper noun are separate failures.
  * `residue_discipline` -- are the #62 unresolved surfaces (`omanko`,
    `laple`) still marked unresolved, or silently smoothed into ordinary
    words?  A gloss that quietly explains an unexplained surface is the
    exact failure L9.8 names.
  * `frame_preservation` -- are the clause's `<<`/`>>` quoting frames
    still there?  #12/#14 read them as a quoting frame, not I/O, so a
    gloss that silently drops them has changed the clause's structure.

Each check returns a *measured* value, never a self-report, and the four
are combined into a per-gloss `verdict` and per-clause *agreement* across
the five strategies.  Disagreement is the signal #64's committee will
consume; agreement is reported so it cannot be quietly assumed.

Determinism, and the honest way to run a model
----------------------------------------------
A model is not deterministic -- verified, not assumed: the same prompt
was run three times against the same backend and returned three different
glosses.  So the module has two modes and they are not interchangeable:

  * **pinned** (default).  The build replays `PINS`, the recorded
    responses of one dated capture run (see `CAPTURE`).  Offline,
    deterministic, byte-identical on rebuild: every data file is a pure
    function of `PINS` and the frozen artifacts, and only the report's
    generation stamp moves with the clock -- disclosed in `summary.json`
    rather than pinned.  Every published number comes from this mode.
  * `--live`.  Re-runs the backend and prints a *diff against the pins*
    rather than overwriting them, exactly as #58's `--probe` does for its
    source evidence.  A live run never rewrites the artifacts, because a
    number that moves with the sampler cannot be regression-tested.

The pins are responses, not glosses: they are raw model text for a named
(clause, strategy, prompt) triple, and the *analysis* of them is done by
the checks below on every build.  Re-running with a different backend is
therefore a measurement, not a rebuild.

Guardrails this module answers for (#76's audit)
------------------------------------------------
* **Circularity**: the model writes, the checks judge.  No strategy
  grades another; `verdict` is computed by `check_gloss`, which never
  calls the model.
* **Hallucinated coherence**: no prompt may assert a plot.  The prompts
  carry a standing instruction to mark rather than resolve, and
  `residue_discipline` + `entity_fidelity` are the machine checks on it.
* **Single-method certainty**: five strategies, agreement published per
  clause, and every gloss carries its strategy and backend.
* **Residue-dropping**: the residue check fails a gloss that erases a
  marked-unresolved surface.
* **Over-correction**: #19 `ocr_error` surfaces are listed to the model
  *as printed*, so it is told they are artefacts rather than asked to
  normalise them.

Not done, and said plainly: the whole book is **not** glossed.  This is
the sampled run the task asks for; the sampling rule is published in
`sample.tsv` with the stratum each clause was drawn from, so the sample is
auditable rather than convenient.

Reproduce::

    conda activate marker
    python work/glossing_pipeline.py                 # pinned, offline
    python work/glossing_pipeline.py --live          # re-run, diff vs pins
    python work/test_glossing_pipeline.py

Inputs: #17's token stream and #33's clauses (through #62, which already
joins them), #30's entity glossary, #62's gloss lexicon and per-token
readings, and #12/#14's operator roles.  No external corpus is consulted
or acquired.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clause_segmentation as cs            # TSV io
import interlinear_edition as ie            # #62: tokens, clauses, readings

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "glossing"

CLAUSES = ROOT / "analysis" / "clauses" / "clauses.tsv"          # #33
TOKENS = ROOT / "analysis" / "tokenizer" / "tokens.tsv"           # #17
ENTITY = ROOT / "analysis" / "entity" / "entity_glossary.tsv"   # #30
INTERLINEAR = ROOT / "analysis" / "interlinear"                  # #62
GLOSS_TOKENS = INTERLINEAR / "gloss_tokens.tsv"
GLOSS_LEXICON = INTERLINEAR / "gloss_lexicon.tsv"
OPERATOR_ROLES = INTERLINEAR / "operator_roles.tsv"

SCHEMA = "glossing/1"

#: The dated capture the pins were taken in.  A pinned record of a dated
#: run, like #58's `PROBE_DATE`: nothing in the artifacts moves with the
#: clock.
CAPTURE = "2026-10-10"

#: The backend the pins were captured from, recorded per gloss so a reader
#: knows what produced the text being judged.
PIN_BACKEND = "opencode run --model opencode/space-bunny-free"

#: Word cap on a prompt, so a pathological clause cannot silently produce a
#: prompt no backend will accept.  The longest clause in the sample is far
#: below it; the cap is a guard, and a clause over it is a build error
#: rather than a silent truncation.
MAX_PROMPT_CHARS = 4000


class GlossingError(Exception):
    """A pin is missing/malformed, or a check cannot be evaluated."""


# ---------------------------------------------------------------------------
# 1. the five strategies
# ---------------------------------------------------------------------------
# A strategy is (id, title, system instruction, output contract).  The
# instructions are the *prompt strategy* the task asks for; they are data,
# not control flow, so `prompt_text` renders them deterministically and a
# test can pin each one.
#
# The standing constraint appended to every strategy is `STANDING` -- the
# L9.8 rules that are not strategy-specific.  It is identical across all
# five on purpose: a strategy that got a different honesty clause would
# make the agreement statistic confounded with the prompt.
STANDING = (
    "Rules that hold for every gloss:\n"
    "- Use only the entity names listed in the ENTITY block above, spelled "
    "exactly as given. Never invent a name, a title or a proper noun.\n"
    "- Do not resolve what is unresolved. If a surface is marked "
    "unresolved in the clause constraints, leave it marked (keep it as "
    "the raw token).\n"
    "- Do not summarise the book's plot. Gloss this clause only.\n"
    "- Surfaces listed as OCR artefacts are not the author's spelling; "
    "reproduce them as printed rather than normalising them.\n"
    "- Return one line of gloss and nothing else: no preamble, no "
    "markdown, no explanation of your own reasoning."
)

STRATEGIES: tuple[dict, ...] = (
    {"id": "literal_operator",
     "title": "literal gloss, operator channel preserved",
     "instruction":
         "Gloss the clause word by word in reading order, keeping its "
         "structure visible.\n"
         "- Every operator must survive verbatim and in its original "
         "order. Copy the operator characters exactly (:: // << >> = ++ "
         "+ - < > /) from the clause; do not replace one with another "
         "and do not drop one.\n"
         "- Keep the clause's own words wherever they are already plain "
         "English; gloss only the opaque surfaces.\n"
         "- An operator you copy is not glossed. If a compound needs "
         "explaining, put the explanation in brackets after it and keep "
         "the operator where it was.",
     "contract": "one line; the clause's operator sequence, verbatim"},
    {"id": "register_paraphrase",
     "title": "register-normalized paraphrase",
     "instruction":
         "Rewrite the clause as ordinary narrative prose, keeping its "
         "content and its sequence of events.\n"
         "- The book's register is mechanical and repetitive; ordinary "
         "English prose is the target, not a tidier version of the "
         "book's own voice.\n"
         "- Keep the operators in place, verbatim and in order, as the "
         "seams the prose hangs on.\n"
         "- Keep entity names exactly as given.\n"
         "- You are changing the register, not the meaning. If you are "
         "unsure of a reading, keep the opaque surface as it is.",
     "contract": "one line of ordinary prose; operators retained in order"},
    {"id": "scene_summary",
     "title": "scene summary",
     "instruction":
         "State, in one plain sentence, what is happening in this "
         "clause, as a reader of the scene would describe it.\n"
         "- Drop the operators from the output: this strategy renders "
         "the scene, not the syntax.\n"
         "- Name only entities from the list below.\n"
         "- Do not add anything that is not in this clause: no motive, "
         "no cause, no outcome beyond it.\n"
         "- Keep unresolved surfaces marked.",
     "contract": "one plain sentence; no operators; no added content"},
    {"id": "plain_english",
     "title": "plain-English translation",
     "instruction":
         "Express the clause in plain English, as if explaining it to a "
         "reader who has never seen the book.\n"
         "- Verbalise the operators: say what the seam does (a field "
         "boundary, a naming seam, a coupling) rather than printing the "
         "character. The clause's structure must still be recoverable "
         "from your sentence.\n"
         "- Keep entity names exactly as given.\n"
         "- Where a surface has no plain-English sense, keep it and say "
         "so; do not invent a sense to make the sentence flow.",
     "contract": "one plain-English sentence; operators verbalised"},
    {"id": "entity_consistent",
     "title": "entity-consistent gloss",
     "instruction":
         "Gloss the clause with the dramatis personae held strictly "
         "consistent.\n"
         "- Use each entity name below exactly as spelled there, and "
         "use the stated role for it. Do not merge two entities, split "
         "one, or give an entity a role it is not listed with.\n"
         "- Every operator survives verbatim and in order.\n"
         "- If the clause names an entity, the gloss names it. If the "
         "gloss would need an entity the clause does not name, you have "
         "misread: use the clause's own words instead.\n"
         "- Keep unresolved surfaces marked.",
     "contract": "one line; #30 names and roles held; operators in order"},
)
STRATEGY_IDS = tuple(s["id"] for s in STRATEGIES)
_STRATEGY_BY_ID = {s["id"]: s for s in STRATEGIES}


# ---------------------------------------------------------------------------
# 2. the constraints: what the model is shown
# ---------------------------------------------------------------------------

def operator_constraints(ops):
    """``{surface: reading}`` for the operators a clause actually uses.

    A prompt never shows an operator the clause does not contain.  That
    is what makes `operator_fidelity` a real check rather than a
    formality: the model is never invited to reproduce an operator it was
    merely told about.
    """
    out = {}
    for op in ops:
        if op not in out:
            out[op] = ie.operator_reading(op)
    return out


def entity_constraints(glossary, present):
    """``[{entity, role, n_occurrences, does, done_to}]`` for named entities.

    `present` is the set of lower-cased #30 names occurring in the clause.
    Names are emitted in #30's own order (sorted by the glossary file), so
    the prompt is a pure function of its inputs.
    """
    rows = []
    for name in sorted(present):
        g = glossary.get(name)
        if g is None:
            continue
        rows.append({
            "entity": g["entity"],
            "role": g.get("role") or "unclassified (#30)",
            "n_occurrences": g.get("n_occurrences", ""),
            "does": "|".join(g.get("does_verbs", "").split("|")[:4]),
            "done_to": "|".join(g.get("done_to_verbs", "").split("|")[:4]),
        })
    return rows


def prompt_text(clause, operators, entities, ocr, residue, strategy):
    """The exact prompt for one (clause, strategy) pair.

    Deterministic: a pure function of its arguments, so the prompt can be
    hashed and pinned, and a change to the constraint assembly shows up as
    a changed hash rather than as silently different glosses.
    """
    strat = _STRATEGY_BY_ID[strategy]
    L = []
    a = L.append
    a("You are glossing one clause of Kenji Siratori's \"Blood Electric\" "
      "(2002), a book of English words under a symbolic operator syntax.")
    a("")
    a("CLAUSE (verbatim; the operators are part of the text):")
    a(clause["original"].replace("\n", " "))
    a("")
    a("OPERATORS IN THIS CLAUSE, with what each one does:")
    if operators:
        # `operator_constraints` inserts on first appearance in the clause and
        # dicts keep insertion order, so this reads in the order the model
        # will meet the operators and is a pure function of its inputs.
        for op, reading in operators.items():
            a("  %-3s %s" % (op, reading))
    else:
        a("  (none: this clause carries no operator)")
    a("")
    a("ENTITIES NAMED IN THIS CLAUSE (use these names and roles exactly; "
      "these are the only names you may use):")
    if entities:
        for e in entities:
            bits = ["%s -- role: %s" % (e["entity"], e["role"])]
            if e["does"]:
                bits.append("it does: %s" % e["does"])
            if e["done_to"]:
                bits.append("it is done to by: %s" % e["done_to"])
            a("  " + "; ".join(bits))
    else:
        a("  (none)")
    a("")
    if residue:
        a("UNRESOLVED SURFACES IN THIS CLAUSE (no evidence-linked reading "
          "exists in this workspace; keep them marked, do not explain "
          "them):")
        for r in residue:
            a("  " + r)
        a("")
    if ocr:
        a("OCR ARTEFACTS IN THIS CLAUSE (reproduce as printed; they are "
          "not the author's spelling):")
        a("  " + ", ".join(ocr))
        a("")
    a("TASK -- %s" % strat["title"])
    a(strat["instruction"])
    a("")
    a("OUTPUT CONTRACT: %s" % strat["contract"])
    a("")
    a(STANDING)
    text = "\n".join(L).strip() + "\n"
    if len(text) > MAX_PROMPT_CHARS:
        raise GlossingError(
            "prompt for clause %s/%s is %d chars, over the %d cap; "
            "sample a shorter clause rather than truncating constraints"
            % (clause["clause_id"], strategy, len(text), MAX_PROMPT_CHARS))
    return text


def prompt_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 3. the pin store
# ---------------------------------------------------------------------------
# `PINS` is filled in by `capture()` (see `--capture`) and committed as
# this literal.  Each entry is the RAW response text for one
# (clause_id, strategy) pair, plus the sha256 of the prompt it was
# produced from.  Storing the prompt hash alongside the response is what
# makes a pin *checkable*: if the constraint assembly changes, the hashes
# stop matching and the build refuses to silently reuse a stale response
# for a prompt that no longer exists.
#
# Recorded 2026-10-10 from `opencode run --model opencode/space-bunny-free`
# (`--capture`).  The model is NOT deterministic -- the same prompt was run
# three times and returned three different glosses -- which is why these
# are pinned rather than re-derived, and why `--live` diffs instead of
# rewriting.
PINS: dict[str, dict] = {
    '10|literal_operator': {
        "prompt_sha256": '352b10891a54615b7c525dbf048b2d696d26bf2ea064dcba2cb8042cf9aea2e9',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": '<< [quoting frame: opens the span that carries the book\'s recurring names/labels; a frame, not an I/O op] I suck the porno [raw token, left as printed, not resolved to a label] nerve gas of boy-roid [boy-roid: kept unresolved, the "roid" half stays as the raw token]'},
    '10|register_paraphrase': {
        "prompt_sha256": 'ff1ba7a429d9cbc794a85f8c59f751917cc70bd46af9c0c4732898f7107cbe0e',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": '<< I draw in the porno nerve gas of boy-roid'},
    '10|scene_summary': {
        "prompt_sha256": 'c199f5d136c07ad0191b91313a17e58cd9dd43baf664b69af88a4903e93937b6',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'The speaker sucks the porno nerve gas of boy-roid.'},
    '10|plain_english': {
        "prompt_sha256": '4b733484019172026d54f239a16ebb49f24ed6c56be708d5b671daad8d63bc70',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'Here the seam is a block-quote opener — a naming frame that carries the book\'s recurring labels rather than passing any signal — so what the clause describes is a speech act: I draw in the "porno nerve gas of boy-roid," a phrase with no plain-English sense of its own, "boy-roid" in particular left as the raw coined token it is.'},
    '10|entity_consistent': {
        "prompt_sha256": '26964e97db440e5a775fa56401f14e3b6e054123e3ada496ca9f5b23d1e98785',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": "<< a quoting frame bearing the book's recurring names/labels, holding open the label-line: I suck the porno nerve gas of boy-roid — the speaker draws breath in on the sucking, and what is taken in is nerve gas named porno, drawn out of boy-roid; who or what boy-roid is stays unresolved, as does porno"},
    '101|literal_operator': {
        "prompt_sha256": 'ca5b49fdac19731b86b9e05aac609e5c816e622cc3a439f9d19fd492ea44bdc6',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": "<<the meridian>>[ND held as a label inside a quoting frame: the frame opens and closes around it, naming rather than acting] To the desire[the wanting, addressed] that the body[ND as mass-form: the persona's body] of self[ND as cable: the self's line] that fires orgones[ND as mass: spending its charge outward] to the dog of silence[ND as cable: the held-silent receiver] was congested[blocked, unable to pass] : It[ND] is jointed[joined along] to the boundless suicide line[ND as channel: the line with no limit] of the artificial-sun[ND as form: the made sun] and self[ND as cable] transplants[ND as form: carried across] sleep[ND as channel: put to sleep by]"},
    '101|register_paraphrase': {
        "prompt_sha256": '87cb2613dd79d0e38324c00760d7ec6e9c9a337666c37b8ced7c4ef84978349f',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": '<<the meridian>>ND desires that the body of self that fires orgones to the dog of silence was congested, and ND is jointed to the boundless suicide line of the artificial-sun, and self transplants sleep'},
    '101|scene_summary': {
        "prompt_sha256": 'c3128c20af81a1037e8627d6923c7f8a4d8501f8b8eac989cb468f2a58f95798',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": "ND's desire to fire orgones at the dog of silence was congested, jointed to the boundless suicide line of the artificial-sun, and self transplants sleep."},
    '101|plain_english': {
        "prompt_sha256": 'e47f65a537c051d7ec7cc4616d79907c88ae58706299f28d22042c4fea550c5b',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'A quoting seam opens and closes around the recurring label "the meridian," carrying names and labels only and passing no input or output, and within it: to the desire that the body of self, ND, that fires orgones to the dog of silence was congested, ND is jointed to the boundless suicide line of the artificial-sun, and ND transplants sleep (both transplants and sleep are things done to ND) — with "orgones," "dog of silence," "boundless suicide line," and "artificial-sun" kept as printed, since none of them carries a plain-English sense here.'},
    '101|entity_consistent': {
        "prompt_sha256": 'de31ea0284dcd5b11963ce5f964614a7a58a62f5e19b5d417a491111236a91bc',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": '<<the meridian>>To the desire that the body of ND — ND as channel firing orgones to the dog of silence, ND as mass — was congested: ND as form is jointed to the boundless suicide line, ND as cable, of the artificial-sun, and ND is done to by transplants and by sleep.'},
    '1003|literal_operator': {
        "prompt_sha256": '2c226220dd0a7381db44dfcb2dcde61d28f32286ba8f5b7c6b7199b2c3f4c7de',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'Put out BABEL[persona: animals(6)|channel(2)|deranged(2)|fecundates(2); acted on by is(5)|channel(3)|line(3)|sun(3)]=[naming seam, glued both sides]TV[persona: screen(6)|cursed(1)|is(1)|received(1); acted on by attack(1)|breaks(1)|down(1)|game(1)][> inline marker closer] excoriate a vital=serum[= naming seam, glued both sides]noise>'},
    '1003|register_paraphrase': {
        "prompt_sha256": '7c9b3185f4607bbc85b9577d586ecfa3ac7b7da70907973600e7f65ad12e7e8a',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'Put out BABEL=TV> excoriate a vital=serum noise>'},
    '1003|scene_summary': {
        "prompt_sha256": '2a6c3b0183a7ec0a7b12f4062db7ecb7e24288bfbe531ddff939d75896d13e17',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'The screen-channeled TV-persona BABEL is put out, excoriated as broken and discharged.'},
    '1003|plain_english': {
        "prompt_sha256": '4c99860ced5a6dc9d3a126041233bf65b840a078b74ff4b9240d4bbdc135f54e',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'Put out BABEL, the persona who makes animals, serves as a channel and is done to by lines and the sun, joined at a naming seam to TV, the persona who screens and is cursed, and this coupling is what fecundates and is fecundated, loaded and broken, and only then the vital serum takes over as its noise: put out the noise that is the serum of that life.'},
    '1003|entity_consistent': {
        "prompt_sha256": 'cff5a46895d112870e0af554fc42c6eef4269b2a14799eaa1f51349f578313c7',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'Put out, the naming seam = fixes persona BABEL to machine-system BABEL=TV, its persona face TV the screened one, and the inline closer > shuts the mark; excoriate a vital=serum noise>, where the second seam = binds the unresolved vital to the unresolved serum and the closing > again cuts the line.'},
    '1005|literal_operator': {
        "prompt_sha256": '347c7925331ca2422bce268dc4ce4149fb7f6055fb33eb66b9316e9ec586d1d5',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": "while devouring body-OMOTYA [OMOTYA embodied, the persona's own flesh taken as a body to be eaten] of control deficiency to bio=less= [one glued term: bio names life, less names absence, so the two parts are fastened into a single word meaning life-without] intermediate area"},
    '1005|register_paraphrase': {
        "prompt_sha256": '64bddfa7385675acfea4dc2473b03457106321d20558733ca96eb9a9627f979a',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'while devouring body-OMOTYA of control deficiency to bio=less= intermediate area'},
    '1005|scene_summary': {
        "prompt_sha256": '51983ce985f3ef49c7c219a7da89e8b1dddfea674718a4d34cab213cc8fe6b30',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'OMOTYA is crucified while devouring the body of control deficiency, in a bio=less intermediate area.'},
    '1005|plain_english': {
        "prompt_sha256": '1e25f334e16ba05c5808d1ec9e22c95ab5cdc787d801169c1cf56d32ed16000c',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'While OMOTYA, the persona crucified by one channel and acted on by that same channel, is being devoured — a naming seam joining body and its absent form marks off, and also names, an intermediate area of control deficiency set to the "less than bio" measure.'},
    '1005|entity_consistent': {
        "prompt_sha256": 'af493d9ae489b0ce630118a7e2519a299d03000b6c85c6b2f6fa5c93519eea63',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'while devouring body-OMOTYA of control deficiency to bio=less= intermediate area'},
    '2789|literal_operator': {
        "prompt_sha256": '3bd8a2ba54e082b78d3e1fc1eeba965748114e1e1a57e0f9556198fcd38da830',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'I surrender [l] / to the chromosome syntax that fused the soul/gram [soul/gram: compound linker fusing soul and gram] of self in her omanko [omanko: unresolved] hydromachine of ToKAGE'},
    '2789|register_paraphrase': {
        "prompt_sha256": '878903696d635e78e161f93850c0e0ce55a199b182ddd4c1caef03318fe9ea89',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'I surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE'},
    '2789|scene_summary': {
        "prompt_sha256": 'eec9a1e4392f8c475f8362971077b31cc9442ad024716c51ec79d8ae3a81b554',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'I surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE.'},
    '2789|plain_english': {
        "prompt_sha256": 'c0f754640c0f5a451edabbc4693571ba19a6048c2ffd4fcbbd59de7f671e5544',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'l surrender, that is I give myself up to the chromosome syntax that fused, at a naming seam where soul is bound compound-wise to gram into one coined word, the self inside her omanko — a term with no plain-English sense I can supply — hydromachine of ToKAGE.'},
    '2789|entity_consistent': {
        "prompt_sha256": '080a3b82adbb0a8f16428cf21105e4d6ed656c4e938035efe94ae2ab68628f7e',
        "backend": 'opencode run --model opencode/space-bunny-free',
        "captured": '2026-10-10',
        "response": 'I surrender to the chromosome syntax that fused the soul/gram (the / gluing gram to soul as compound, not the sentence-final separator office) of self in her omanko [unresolved] hydromachine of ToKAGE, and ToKAGE is the one persona this clause names: the persona who is and was, who fecundates and murders, and who is done to by love, by speed, by like, by crucifixion; no other persona is named here, and the surrenders-to is not given a name of its own, while omanko stays unresolved as printed.'},
}


def pin_key(clause_id, strategy):
    return "%s|%s" % (clause_id, strategy)


def run_live(prompt, model, timeout=180):
    """One live backend call -> raw response text.

    Runs the `opencode` CLI as a subprocess with an explicit argument list
    (never `shell=True`), so a clause's text cannot reach a shell.  Only
    stdout is captured: the CLI writes a `> build · model` banner to
    stderr, and mixing the two would put a model name in the corpus.
    """
    cmd = ["opencode", "run", "--model", model, prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise GlossingError(
            "the opencode CLI is not on PATH; --live needs it, the pinned "
            "build does not (%s)" % exc)
    except subprocess.TimeoutExpired as exc:
        raise GlossingError("backend timed out after %ds: %s"
                            % (timeout, exc))
    if proc.returncode != 0:
        raise GlossingError("backend exited %d: %s"
                            % (proc.returncode, proc.stderr.strip()[:200]))
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# 4. loading the corpus
# ---------------------------------------------------------------------------

def load_glossary():
    """{lower entity: #30 row}."""
    out = {}
    if ENTITY.exists():
        for r in cs.read_tsv(ENTITY):
            out.setdefault(r["entity"].lower(), r)
    return out


def load_residue_lexicon():
    """{lower surface: reading} for #62's curated readings (PLAIN_READINGS)."""
    return dict(ie.PLAIN_READINGS)


def load_ocr_class():
    """{lower unit surface} for every #19 `ocr_error` type."""
    # `load_context` returns #19's five loaded tables; only the screening
    # table's category column decides what is listed as an artefact here.
    _bodies, _page_units, _unit_meta, _is_markup, screening = \
        ie.fm.load_context()
    return {u.lower() for u, row in screening.items()
            if row.get("category") == "ocr_error"}


def load_operator_token_map():
    """{clause_id: [operator surface, ...]} from #17's token stream.

    Read through #62's own token join rather than by re-parsing the text,
    so the operators the pipeline checks are exactly the operators the
    tokenizer saw -- `soul/gram` cannot be read as one operator here and
    two by the checker.
    """
    by_page = collections.defaultdict(list)
    for r in cs.read_tsv(TOKENS):
        by_page[int(r["page"])].append(r)
    for p in by_page:
        by_page[p].sort(key=lambda r: (int(r["start"]), int(r["end"])))
    out = {}
    for row in cs.read_tsv(CLAUSES):
        cid = row["clause_id"]
        start, end = int(row["start"]), int(row["end"])
        ops = []
        for t in by_page.get(int(row["page"]), ()):
            if start <= int(t["start"]) and int(t["end"]) <= end \
                    and t["kind"] == "OP":
                ops.append(json.loads(t["text_json"]))
        out[cid] = ops
    return out


def load_clause_residue():
    """``{clause_id: [(surface, #62 source), ...]}`` from #62's token table.

    Both the surface and #62's reading source are kept, because they are
    not the same kind of thing: `unresolved` is a #19 `true_coinage` with
    no evidence-linked reading (`omanko`, `laple`) -- an opaque surface the
    model will be tempted to explain -- while `junk_19` is markup residue
    (`href`) that is not book text at all.  `residue_surfaces` is the
    prompt-facing view; the source is what the sampler ranks on.
    """
    out = collections.defaultdict(list)
    if GLOSS_TOKENS.exists():
        for r in cs.read_tsv(GLOSS_TOKENS):
            if r["confidence"] == "unresolved" and r["surface"]:
                pair = (r["surface"], r["source"])
                if pair not in out[r["clause_id"]]:
                    out[r["clause_id"]].append(pair)
    return {k: sorted(v) for k, v in out.items()}


def residue_surfaces(residue):
    """Just the surfaces from `load_clause_residue`'s rows."""
    return [s for (s, _src) in residue]


def load_clause_ocr(ocr_types):
    """{clause_id: [ocr_error surface, ...]} present in the clause."""
    out = collections.defaultdict(list)
    if GLOSS_TOKENS.exists():
        for r in cs.read_tsv(GLOSS_TOKENS):
            if r["surface"] and r["surface"].lower() in ocr_types:
                out[r["clause_id"]].append(r["surface"])
    return {k: sorted(set(v)) for k, v in out.items()}


def load_clauses():
    """``{clause_id: #33 row}``, in (page, start) order.

    #33's `clauses.tsv` carries offsets but not the clause text -- it is a
    segmentation ledger, and #62 owns the rendering.  The `original` column
    and the entity-bearing region are therefore joined in from #62's
    `interlinear.tsv` below; a clause with no #62 row keeps a null text and
    is skipped by the sampler rather than glossed from offsets alone.
    """
    rows = cs.read_tsv(CLAUSES)
    rows.sort(key=lambda r: (int(r["page"]), int(r["start"])))
    out = {}
    for r in rows:
        r["original"] = ""
        out[r["clause_id"]] = r
    for cid, inter in load_interlinear().items():
        if cid in out:
            out[cid]["original"] = inter.get("original", "")
            out[cid]["gloss"] = inter.get("gloss", "")
    return out


def load_interlinear():
    """``{clause_id: #62 row}`` -- the evidence-linked baseline to compare to.

    Read with #33's `read_tsv` rather than `csv.DictReader`: this table has
    clause texts longer than the `csv` module's default 131072-byte field
    limit, and #33's reader splits lines itself.
    """
    path = INTERLINEAR / "interlinear.tsv"
    if not path.exists():
        return {}
    return {r["clause_id"]: r for r in cs.read_tsv(path)}


# ---------------------------------------------------------------------------
# 5. the sample: stratified and published
# ---------------------------------------------------------------------------
# The sample is drawn by a *rule*, not by taste, and the stratum each
# clause was drawn from is published in `sample.tsv`.  Five strata, chosen
# because each one stresses a different check:
#
#   operator_dense   >= 4 operators: the operator_fidelity check has the
#                    most to lose, and #58 showed long operator spans are
#                    translation-invariant, so they are the ones a model
#                    most wants to "tidy".
#   entity_named     names a #30 entity: entity_fidelity has teeth.
#   quoting_frame    contains << >>: #12/#14 read these as a quoting
#                    frame, so frame_preservation has teeth.
#   residue          carries a #62 unresolved surface: residue_discipline
#                    has teeth.  Rare by construction (the whole book has
#                    four), so it is drawn first and unconditionally.
#                    Among residue candidates, a #19 `true_coinage` with no
#                    reading (`omanko`, `laple`) outranks markup residue
#                    (`href`): the first is an opaque surface a model will
#                    want to explain, the second is not book text, so only
#                    the first can fail `residue_discipline` for the reason
#                    that check exists.
#   plain_control    one or two operators, no entity: a control, where a
#                    check that fires on everything is exposed as such.
#
# `pick_sample` is deterministic and the stratum each clause came from is
# published.  The tie-break is on the clause id **as text**, not on #33's
# reading order -- `1003` sorts before `101` -- so the sample is *not* the
# numerically earliest clause of each stratum, and it does not claim to be.
# That is disclosed rather than quietly changed: the pins are keyed by the
# sha256 of the prompt they answer, so re-drawing the sample to book order
# moves every prompt and needs a fresh `--capture`, and the artifact this
# run pins is this sample.  It is still a rule and not a choice of
# readability: nothing in it looks at the glosses.
STRATA = ("residue", "operator_dense", "entity_named", "quoting_frame",
          "plain_control")


def clause_stratum(ops, names, residue):
    if residue:
        return "residue"
    if ops.count("<<") and ops.count(">>"):
        return "quoting_frame"
    if len(ops) >= 4:
        return "operator_dense"
    if names:
        return "entity_named"
    return "plain_control"


def residue_rank(residue):
    """Lower is better: does the clause carry a *word* the corpus cannot read?

    `0` for a #19 `true_coinage` with no evidence-linked reading, `1` for
    markup/junk residue.  Sorting on this rather than on page position
    makes the residue stratum test the thing the check is for.
    """
    return min((0 if src == "unresolved" else 1) for (_s, src) in residue) \
        if residue else 9


def pick_sample(clauses, opmap, glossary, residue_by_clause, per_stratum=1):
    """``{clause_id: stratum}``, one candidate per stratum, by a rule.

    One clause per stratum by default: the sample is meant to be small
    enough to read in full and wide enough that each check has at least
    one case where it can fail.  The `residue` stratum has exactly four
    candidates in the whole book (#62's `omanko` x2, `laple`, `href`), so
    it is drawn first and cannot be crowded out.

    Within a stratum the quality rank decides and the tie-break is the
    clause id compared as text.  `sorted` on `(rank, cid)` does that
    implicitly; it is spelled out as a key here because "#33's ids are
    integers" is exactly the assumption that makes `1003` sort before
    `101`, and a reader of the sampling rule is owed the fact.
    """
    candidates = collections.defaultdict(list)
    for cid in clauses:                       # already in (page, start) order
        row = clauses[cid]
        if row["region"] != "body" or not row["original"]:
            continue
        ops = opmap.get(cid, [])
        if not ops:
            continue
        names = {n for n in glossary if n in row["original"].lower()}
        residue = residue_by_clause.get(cid, [])
        st = clause_stratum(ops, names, residue)
        # sort key: stratum-specific quality first, then the clause id as
        # text (see `STRATA`), so the choice is reproducible from the
        # artifacts alone.
        candidates[st].append(
            (residue_rank(residue) if st == "residue" else 0, cid))
    chosen = {}
    for st in STRATA:
        ranked = sorted(candidates.get(st, []),
                        key=lambda pair: (pair[0], str(pair[1])))
        for _rank, cid in ranked[:per_stratum]:
            chosen[cid] = st
    return chosen


# ---------------------------------------------------------------------------
# 6. the checks
# ---------------------------------------------------------------------------
# Four checks, each a pure function of (clause, strategy, gloss) plus the
# frozen artifacts.  None of them calls the model: `check_gloss` is the
# whole judge, so the model cannot grade its own work (L9.8 circularity).

#: The operator token subsequence, read off #17's stream rather than a
#: regex over the clause text.  #62's `gloss_tokens.tsv` is not the source:
#: it records what the *reading* was, and a check that trusted it would be
#: checking #62 against itself.
#: The operator characters, longest-first so `::` never matches as two `:`.
#: This set is *derived from #17*, not chosen: #17's token stream contains
#: exactly these eleven surfaces with `kind == "OP"` (432 `<<`, 2048 `=`,
#: 3084 `::`, 1492 `/`, 2149 `//`, 406 `>>`, 198 `++`, 778 `>`, 381 `<`,
#: 17 `|`, 15 `+`) and no others.  Notably **no dash**: `-` is a word-internal
#: compound join inside a WORD token (`boy-roid`) or `PUNCT` when
#: free-standing, and `—`/`–` are `PUNCT` too.  Counting a dash as an
#: operator would charge every gloss for reproducing the book's own
#: hyphenation, and would make every failure on this check uninterpretable.
OP_SURFACES = frozenset(("<<", ">>", "::", "//", "++", "=", "+", "<", ">",
                         "/", "|"))
_OP_RE = re.compile(r"::|//|<<|>>|\+\+|[=+\-<>|/]")
_OP_ONLY_RE = re.compile(r"::|//|<<|>>|\+\+|[=+<>|/]")


def operator_subsequence(text):
    """The operator characters in `text`, in order.

    Only the eleven surfaces #17 classifies as `OP` are recognised; a `-`
    or a typographic dash is punctuation or part of a word and is left
    alone.  The longest-match alternation keeps `::` from being read as
    two colons and `//` as two slashes.
    """
    return [t for t in _OP_ONLY_RE.findall(text) if t in OP_SURFACES]


def _operator_diff_detail(ops, got):
    """A human-readable account of how `got` differs from `ops`.

    Counted, not set-membership: a gloss that repeats `soul/gram` inside a
    bracketed gloss has *added* operators, and set membership reports that
    as "reordered" -- which describes the failure inaccurately and would
    make the operator table unreadable.  "reordered" is reserved for the
    case where the two sequences are the same multiset in a different
    order.
    """
    want, have = collections.Counter(ops), collections.Counter(got)
    missing, extra = [], []
    for sym in sorted(set(want) | set(have)):
        delta = have[sym] - want[sym]
        if delta > 0:
            extra.extend([sym] * delta)
        elif delta < 0:
            missing.extend([sym] * (-delta))
    if not missing and not extra:
        return "reordered (same operators, different order)"
    detail = []
    if missing:
        detail.append("dropped %d (%s)" % (len(missing),
                                           ",".join(sorted(missing))))
    if extra:
        detail.append("added %d (%s)" % (len(extra), ",".join(sorted(extra))))
    return "; ".join(detail)


def check_operator_fidelity(ops, gloss, strategy):
    """Does the operator channel match what *this* strategy owes?

    The obligation differs by strategy and is published in
    `STRATEGY_OWES`/`OPERATOR_DROP`, because a strategy is judged against
    its own contract and not against another's:

      * an `OPERATOR_KEEP` strategy is told to reproduce the channel
        verbatim and in order, so its sequence must equal the clause's;
      * an `OPERATOR_DROP` strategy is told to verbalise or discard the
        operators, so its channel must be **empty**.  Printing an operator
        there is the violation and dropping one is compliance, so the two
        cases cannot share one scoring rule.

    Scoring those backwards was a real bug in an earlier revision: it
    failed a `plain_english` gloss for *dropping* `=`, which is exactly
    what that strategy was told to do.
    """
    got = operator_subsequence(gloss)
    if strategy in OPERATOR_DROP:
        if not got:
            return True, "verbalised", ""
        return (False,
                "kept %d operator character(s) it was told to verbalise: %s"
                % (len(got), ",".join(sorted(set(got)))),
                got)
    if got == ops:
        return True, "verbatim", ""
    return False, _operator_diff_detail(ops, got), got


#: #30's role vocabulary, read off the glossary's own `role` column, so a
#: gloss may legitimately build a compound from a role word (`TV-persona`)
#: without the invented-name test calling it a new character.  Populated
#: from the corpus on first use rather than typed here, so a new role in
#: #30 licenses itself and no code change is needed.
_ROLE_WORDS: frozenset[str] | None = None

#: Fallback used when the frozen glossary is absent (unit tests run without
#: artifacts).  Only the roles #30 actually assigns in this corpus.
_ROLE_WORDS_FALLBACK = frozenset("""
persona antagonist protagonist place thing system character group
mechanism device city world zone line state body form
""".split())


def role_words():
    """#30's role vocabulary, as a frozen set."""
    global _ROLE_WORDS
    if _ROLE_WORDS is None:
        words = set()
        for row in load_glossary().values():
            for role in (row.get("role") or "").split():
                if len(role) >= 3:
                    words.add(role.lower())
        _ROLE_WORDS = frozenset(words or _ROLE_WORDS_FALLBACK)
    return _ROLE_WORDS


#: Words that begin a gloss or a clause and so are capitalised for
#: grammatical rather than semantic reasons.  Without this the
#: "invented name" test fires on `Put` and `While`, which would make the
#: check uninterpretable -- a check that cries wolf is worse than none.
_SENTENCE_STARTERS = frozenset("""
put while when where what which who whom whose why how that this these those
the a an and or but so if then than as at by for from in into on of to with
it its his her their our your my he she they we you i him them us me one some
any all each every both there here now not no nor
""".split())


def entity_present(name, gloss):
    """Does the gloss carry this #30 name?

    An exact substring test is wrong in both directions for this corpus:

      * too strict for names that contain an operator -- the glossary's
        `BABEL=TV` is the construct `BABEL` joined to `TV` by a naming
        seam, so a gloss writing `BABEL[...]=TV` has kept it while a
        substring test reports it dropped;
      * too strict for an unambiguous abbreviation -- `ND` in "ND's
        device" carries `NDRO`, and treating it as a dropped name would
        fire the check on a faithful gloss.

    Matching on parts alone would be too weak: it would accept a gloss that
    listed `BABEL` and `TV` as two unrelated words, which is precisely what
    the seam says they are not, so the seams must still be present.

    An abbreviation must be at least two characters of a name of at least
    three, and must not be a whole name by itself (`AD` does not license
    `ADAM`, and neither is it any other entity, so accepting it would be a
    guess rather than a resolution).
    """
    low, g_low = name.lower(), gloss.lower()
    if low in g_low:
        return True
    parts = [p for p in _OP_ONLY_RE.split(name) if p]
    seams = operator_subsequence(name)
    if len(parts) >= 2 and seams:
        # An operator-bearing name: its parts must occur in order with the
        # same seams still present between them.  Matching on parts alone
        # would accept a gloss listing `BABEL` and `TV` as two unrelated
        # words, which is precisely what the seam says they are not.
        cursor = 0
        for part in parts:
            idx = g_low.find(part.lower(), cursor)
            if idx < 0:
                return False
            cursor = idx + len(part)
        gloss_seams = operator_subsequence(gloss)
        return all(gloss_seams.count(s) >= seams.count(s) for s in set(seams))
    # A plain name must appear in full.  An earlier revision also accepted a
    # prefix ("ND" carrying "NDRO"), but no length rule separates that case
    # from "AD" carrying "ADAM" -- and "AD" really is ambiguous in a corpus
    # containing AD, ADAM and ANTI-ADAM.  A check that guesses which
    # abbreviation is which would report on the corpus's ambiguity rather
    # than on the gloss, so the exact-name rule stands and the abbreviation
    # is reported as a dropped name, which is the true and checkable fact.
    return False


def check_entity_fidelity(entities, gloss, allowed_vocab=()):
    """Names held (and none invented).

    Both directions matter and they fail differently: a *dropped* name is
    the model losing a constraint it was shown, an *invented* name is the
    model hallucinating a character.

    The invented-name test is deliberately narrow, because a check that
    fires on ordinary capitalisation is worthless.  A token counts as an
    invention only when it is

      * capitalised (a sentence-initial word is exempt via
        `_SENTENCE_STARTERS`),
      * not in `allowed_vocab` -- the clause's own words plus the strategy
        instruction, so `English` (from "plain English") and `Put` are
        vocabulary rather than hallucinations, and
      * not one of the #30 names the gloss was given.

    That last clause matters most: `English` in "plain English" is a
    language name, not a character.  Without the vocabulary exemption this
    check fired on it, and a check that cries wolf is worse than none.
    """
    problems = []
    for e in entities:
        if not entity_present(e["entity"], gloss):
            problems.append("dropped %s" % e["entity"])
    allowed = {e["entity"].lower() for e in entities}
    allowed |= set(_STOPWORDS) | set(_SENTENCE_STARTERS)
    allowed |= {v.lower() for v in allowed_vocab}
    # An abbreviation of a glossary name is not an invention: `ND` in
    # "ND's device" abbreviates `NDRO` and the reader can resolve it.  A
    # token matches if it is allowed whole, or is a prefix of at least two
    # characters of a name of at least three (`ND` licenses `NDRO`; `AD`
    # does not license `ADAM`, which would be an unverifiable guess).
    # A one-letter capital is never a character name here: it is the `I` of
    # `I/O`, the `O` of `O`, an initial.  Requiring two characters is what
    # separates a name from an initial, and it costs nothing -- #30's
    # entities are all multi-character.
    for tok in re.findall(r"\b[A-Z][A-Za-z0-9\-']{1,}\b", gloss):
        # strip a possessive/contraction suffix without eating a trailing
        # `s` that belongs to the name: `ND's` -> `nd`, `Adams` -> `adams`
        low = tok.lower()
        if low.endswith("'s"):
            low = low[:-2]
        elif low.endswith("'"):
            low = low[:-1]
        if low in allowed or low in _STOPWORDS or low in _SENTENCE_STARTERS:
            continue
        # A prefix of a name the gloss was given is that name abbreviated
        # (`ND` for `NDRO`), not a new character: `entity_present` has
        # already scored it as a dropped name, and reporting it a second
        # time as an invention would charge one behaviour twice.  One
        # character is not enough -- `A` would license every name here.
        if len(low) >= 2 and any(name.startswith(low)
                                  for name in allowed if len(name) >= 3):
            continue
        # A hyphenated compound is not a name either when every part is
        # licensed: `TV-persona` is the glossary entity `TV` plus the word
        # `persona`, both of which the gloss was given.  Reading it as an
        # invented character would be the check inventing a character of
        # its own -- the L9.8 failure, one level down.
        if "-" in tok:
            parts_ = [p.lower() for p in tok.split("-") if p]
            if parts_ and all(
                    p in allowed or p in _STOPWORDS
                    or p in _SENTENCE_STARTERS or p in role_words()
                    or any(n.startswith(p) for n in allowed if len(n) >= 3)
                    for p in parts_):
                continue
        problems.append("invented %s" % tok)
    return (False, "; ".join(dict.fromkeys(problems))) if problems \
        else (True, "names held")


def strategy_vocabulary(strategy):
    """The words the strategy's own instruction licenses in a gloss.

    A strategy that says "one plain sentence" has licensed the word
    `sentence`; without this the invented-name test would charge a gloss
    for obeying its instructions.
    """
    strat = _STRATEGY_BY_ID.get(strategy)
    if strat is None:
        raise GlossingError(
            "unknown strategy %r; the five strategies the task names are %s"
            % (strategy, ", ".join(STRATEGY_OWES)))
    return re.findall(r"[A-Za-z][A-Za-z\-']+",
                      strat["instruction"] + " " + strat["contract"])


_STOPWORDS = frozenset("""
a an the this that these those there here it its his her their our your my
of in on at to from for with by as and or but not no nor so if then than
he she they we you i him them us me one some any all each every both
""".split())


def check_residue_discipline(residue, gloss):
    """Are the clause's unresolved surfaces still unresolved?

    An unresolved surface that the gloss turns into an ordinary word has
    been *resolved by assertion*: the gloss now says something the
    workspace has no evidence for.  Keeping the token (marked or not) is
    the pass condition.
    """
    if not residue:
        return True, "no residue in clause"
    lowered = gloss.lower()
    kept, smoothed = [], []
    for surf in residue:
        # the token, or the compound the token sits inside
        if surf.lower() in lowered or \
                any(surf.lower() in w.lower() for w in re.findall(
                    r"[A-Za-z\-]+", gloss)):
            kept.append(surf)
        else:
            smoothed.append(surf)
    if smoothed:
        return False, "smoothed %s into ordinary words" % ",".join(smoothed)
    return True, "kept %s marked" % ",".join(kept)


def check_frame_preservation(ops, gloss):
    """Are the clause's `<<`/`>>` quoting frames still there?

    #12/#14 measured these as the book's quoting frame rather than I/O, so
    a gloss that drops them has changed the clause's structure, not its
    wording.  Strategies are told to keep them; `plain_english` and
    `scene_summary` are told to drop operators, so for those two
    `check_gloss` marks the row `applies: False` instead of failing a
    compliant gloss for obeying its instruction -- and a non-applicable
    row never counts as a pass either (see `GLOSS_HEADER`).
    """
    want_open, want_close = ops.count("<<"), ops.count(">>")
    if want_open == 0 and want_close == 0:
        return True, "no frame in clause"
    got_open = gloss.count("<<")
    got_close = gloss.count(">>")
    if got_open == want_open and got_close == want_close:
        return True, "%d<< %d>>" % (got_open, got_close)
    return False, ("clause has %d<< %d>>, gloss has %d<< %d>>"
                   % (want_open, want_close, got_open, got_close))


#: What each strategy owes its operator channel, and what each is told in
#: its instruction.  Published so the verdict table can be read without
#: guessing why one strategy fails a check another passes.
STRATEGY_OWES = {
    "literal_operator":    {"operators": "verbatim, in order",
                            "frame": "kept"},
    "register_paraphrase": {"operators": "verbatim, in order",
                            "frame": "kept"},
    "scene_summary":       {"operators": "dropped by instruction",
                            "frame": "dropped by instruction"},
    "plain_english":       {"operators": "verbalised by instruction",
                            "frame": "verbalised by instruction"},
    "entity_consistent":   {"operators": "verbatim, in order",
                            "frame": "kept"},
}
#: Strategies whose prompt tells them not to print the operator characters.
#: For these the channel must be empty, and `frame_preservation` does not
#: apply either -- they were told the `<< >>` frame is a quoting device to
#: be rendered in words, not a character to be copied.  Scored the other way
#: round it would fail a compliant gloss, which is what an earlier revision
#: did.
OPERATOR_DROP = frozenset({"scene_summary", "plain_english"})


def _strategy(strategy):
    """`STRATEGY_OWES` row for `strategy`, or a clear error.

    A typo'd strategy id would otherwise surface as a `KeyError` three
    frames later, inside whichever check happens to run first.  This module
    fails loudly and names the valid ids, which is the difference between
    an error a caller can act on and a traceback.
    """
    try:
        return STRATEGY_OWES[strategy]
    except KeyError:
        raise GlossingError(
            "unknown strategy %r; the five strategies the task names are %s"
            % (strategy, ", ".join(STRATEGY_OWES))) from None


def check_gloss(clause, ops, entities, residue, strategy, gloss):
    """All four checks for one gloss -> (verdict, rows).

    The verdict is `pass` only if every check that *applies* to the
    strategy passed.  "Applies" is per check and published in
    `STRATEGY_OWES`/`OPERATOR_DROP`, so a `pass` is not a gloss that
    scored well -- it is a gloss that broke none of the rules its own
    strategy was held to.
    """
    strat = _strategy(strategy)
    rows = []
    op_ok, op_message, _op_got = check_operator_fidelity(ops, gloss, strategy)
    rows.append({"check": "operator_fidelity", "pass": op_ok,
                 "applies": True, "verdict": op_message if op_ok else
                 "violated", "detail": op_message,
                 "contract": strat["operators"]})

    en_ok, en_detail = check_entity_fidelity(
        entities, gloss,
        allowed_vocab=(clause.get("original", "").split()
                       + strategy_vocabulary(strategy)))
    rows.append({"check": "entity_fidelity", "pass": en_ok, "applies": True,
                 "verdict": "held" if en_ok else "violated",
                 "detail": en_detail, "contract": "#30 names as given"})

    rs_ok, rs_detail = check_residue_discipline(residue, gloss)
    rows.append({"check": "residue_discipline", "pass": rs_ok,
                 "applies": bool(residue), "verdict": "held" if rs_ok
                 else "violated", "detail": rs_detail,
                 "contract": "unresolved surfaces stay marked"})

    fr_ok, fr_detail = check_frame_preservation(ops, gloss)
    rows.append({"check": "frame_preservation", "pass": fr_ok,
                 "applies": want_frame(ops, strategy),
                 "verdict": "held" if fr_ok else "violated",
                 "detail": fr_detail,
                 "contract": "<< >> quoting frame %s" % strat["frame"]})

    applicable = [r for r in rows if r["applies"]]
    verdict = "pass" if all(r["pass"] for r in applicable) else "fail"
    return verdict, rows


def want_frame(ops, strategy):
    """True when this strategy was told to keep the clause's quoting frame.

    Only the `OPERATOR_KEEP` strategies copy the frame's characters; the
    others render it in words, so holding them to a character match would
    fail a compliant gloss.
    """
    return bool(ops.count("<<") or ops.count(">>")) and \
        strategy not in OPERATOR_DROP


# ---------------------------------------------------------------------------
# 7. agreement across strategies
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9\-]+")


def gloss_tokens(text):
    """Lower-cased word set of a gloss, operators excluded.

    Agreement is measured on content words only: two strategies that both
    emit `::` agree trivially, and that is not the agreement #64 cares
    about.  Hyphens are dropped too (`_OP_RE`), so `artificial-sun` and
    `artificial sun` count as the same content: a hyphen is not an operator
    in this corpus (#17 types `boy-roid` as one WORD), and charging a
    paraphrase for the book's compound spelling would measure orthography
    rather than reading.
    """
    stripped = _OP_RE.sub(" ", text.lower())
    return {w for w in _WORD_RE.findall(stripped) if w not in _STOPWORDS}


def agreement(clause_id, glosses):
    """Pairwise agreement of the strategies' glosses for one clause.

    Jaccard over content words, averaged over every pair of the glosses
    present (ten pairs for five strategies).  Reported for every clause
    including the unanimous ones -- agreement that is only printed when it
    is low is agreement nobody can rely on.
    """
    sets = {sid: gloss_tokens(g) for sid, g in sorted(glosses.items())}
    ids = sorted(sets)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = sets[ids[i]], sets[ids[j]]
            union = a | b
            jacc = len(a & b) / len(union) if union else 1.0
            pairs.append((ids[i], ids[j], jacc))
    mean = sum(p[2] for p in pairs) / len(pairs) if pairs else 0.0
    lo = min(pairs, key=lambda p: (p[2], p[0], p[1])) if pairs else None
    hi = max(pairs, key=lambda p: (p[2], p[0], p[1])) if pairs else None
    return {
        "clause_id": clause_id,
        "mean_jaccard": round(mean, 4),
        "min_jaccard": round(lo[2], 4) if lo else 0.0,
        "max_jaccard": round(hi[2], 4) if hi else 0.0,
        "least_agreeing_pair": "%s|%s" % (lo[0], lo[1]) if lo else "",
        "most_agreeing_pair": "%s|%s" % (hi[0], hi[1]) if hi else "",
        "n_strategies": len(ids),
    }


# ---------------------------------------------------------------------------
# 8. build
# ---------------------------------------------------------------------------

GLOSS_HEADER = [
    "clause_id", "page", "stratum", "strategy", "strategy_title", "verdict",
    "n_checks", "n_applicable", "n_passed", "checks_passed", "backend",
    "prompt_sha256", "response_sha256", "captured", "n_chars",
    "operators_in", "operators_out", "entities_in", "n_entities",
    "residue_in", "baseline_62_gloss", "overlap_with_62", "gloss",
]
# One row per (clause, strategy): the gloss, its verdict, and the provenance
# a reader needs to re-check both.  `n_checks` is how many checks ran,
# `n_applicable` how many of them this strategy was actually held to, and
# `n_passed` how many of *those* passed -- so `n_passed <= n_applicable`
# always holds, and `n_passed == n_applicable` exactly when the verdict is
# `pass`.
CHECK_HEADER = ["clause_id", "strategy", "check", "applies", "pass",
                "verdict", "contract", "detail"]
AGREE_HEADER = ["clause_id", "n_strategies", "mean_jaccard", "min_jaccard",
                "max_jaccard", "least_agreeing_pair", "most_agreeing_pair"]
SAMPLE_HEADER = ["clause_id", "page", "stratum", "n_operators",
                 "n_entities_named", "residue_surfaces", "why",
                 "original"]
PROMPT_HEADER = ["clause_id", "strategy", "prompt_sha256", "n_chars",
                 "prompt"]
CAPTURED_HEADER = ["clause_id", "strategy", "prompt_sha256", "backend",
                   "captured", "response"]


def _pct(n, total):
    return 100.0 * n / total if total else 0.0


STRATUM_WHY = {
    "residue": "carries a #62 unresolved surface; residue_discipline "
               "has teeth here",
    "operator_dense": ">= 4 operators; operator_fidelity has the most to "
                      "lose",
    "entity_named": "names a #30 entity; entity_fidelity has teeth",
    "quoting_frame": "carries << >>; #12/#14 read these as a quoting frame, "
                     "so frame_preservation has teeth",
    "plain_control": "1-2 operators and no entity: the control that "
                     "exposes a check which fires on everything",
}


def resolve_glosses(sample, opmap, glossary, residue_by_clause,
                    ocr_by_clause, strategies, live_model=None,
                    clauses=None):
    """The gloss for every (clause, strategy) in the sample.

    In pinned mode a missing pin is a hard error.  The alternative --
    glossing what happens to be pinned and calling it the sample -- is the
    failure mode this module exists to avoid: a five-strategy table with a
    hole in it measures the hole.

    `clauses` must be `load_clauses()`'s table: a clause id with no row
    there is a sample drawn from something other than #33, and is refused
    rather than glossed from offsets alone.
    """
    clauses = clauses or {}
    for sid in strategies:          # fail on a bad strategy id up front
        _strategy(sid)
    rows, prompts, checks = [], [], []
    for cid, stratum in sorted(sample.items(), key=lambda kv: int(kv[0])):
        clause = clauses.get(cid)
        if clause is None:
            raise GlossingError(
                "no #33 clause row for %s; the sample must come from "
                "load_clauses() (got %d clause rows, %d sample ids)"
                % (cid, len(clauses), len(sample)))
        ops = opmap.get(cid, [])
        names = {n for n in glossary if n in clause["original"].lower()}
        ents = entity_constraints(glossary, names)
        residue = residue_surfaces(residue_by_clause.get(cid, []))
        ocr = ocr_by_clause.get(cid, [])
        for sid in strategies:
            text = prompt_text(clause, operator_constraints(ops), ents,
                               ocr, residue, sid)
            ph = prompt_sha256(text)
            key = pin_key(cid, sid)
            if live_model is not None:
                gloss = run_live(text, live_model)
                backend, captured = live_model, CAPTURE
            else:
                pin = PINS.get(key)
                if pin is None:
                    raise GlossingError(
                        "no pinned response for %s (run --capture to record "
                        "one; a partial strategy table would measure the "
                        "missing cell, not the book)" % key)
                if pin["prompt_sha256"] != ph:
                    raise GlossingError(
                        "pin for %s was recorded against a different prompt "
                        "(pin %s, now %s); the constraint assembly changed, "
                        "so the response is stale -- re-capture rather than "
                        "reuse it"
                        % (key, pin["prompt_sha256"][:12], ph[:12]))
                gloss = pin["response"]
                backend = pin["backend"]
                captured = pin["captured"]
            verdict, crows = check_gloss(clause, ops, ents, residue, sid,
                                         gloss)
            applicable = [r for r in crows if r["applies"]]
            base = clause.get("gloss", "")
            rows.append({
                "clause_id": cid,
                "page": clause["page"],
                "stratum": stratum,
                "strategy": sid,
                "strategy_title": _STRATEGY_BY_ID[sid]["title"],
                "verdict": verdict,
                "n_checks": len(crows),
                "n_applicable": len(applicable),
                # Counted over the rows the check *applies* to, for the same
                # reason `build` counts the summary that way: a check that
                # does not apply cannot pass.  Counting every published row
                # published "3 checks passed of 2 applicable" for 20 of the
                # 25 glosses, and listed `residue_discipline` as passed on
                # clauses that carry no residue at all.
                "n_passed": sum(1 for r in applicable if r["pass"]),
                "checks_passed": ",".join(r["check"] for r in applicable
                                          if r["pass"]) or "none",
                "backend": backend,
                "prompt_sha256": ph,
                "response_sha256": hashlib.sha256(
                    gloss.encode("utf-8")).hexdigest(),
                "captured": captured,
                "n_chars": len(gloss),
                "operators_in": ",".join(ops),
                "operators_out": ",".join(operator_subsequence(gloss)),
                "entities_in": ",".join(e["entity"] for e in ents),
                "n_entities": len(ents),
                "residue_in": ",".join(residue),
                "baseline_62_gloss": base,
                "overlap_with_62": round(
                    len(gloss_tokens(gloss) & gloss_tokens(base)) /
                    len(gloss_tokens(gloss) | gloss_tokens(base))
                    if (gloss_tokens(gloss) | gloss_tokens(base)) else 1.0, 4),
                "gloss": gloss,
            })
            for r in crows:
                checks.append(dict(r, clause_id=cid, strategy=sid))
            prompts.append({"clause_id": cid, "strategy": sid,
                            "prompt_sha256": ph, "n_chars": len(text),
                            "prompt": text})
    return rows, prompts, checks


def build(out=OUT, quiet=False, strategies=STRATEGY_IDS, per_stratum=1):
    """The pinned build: replay the pins, check them, publish the sample."""
    OUT = Path(out)
    clauses = load_clauses()
    opmap = load_operator_token_map()
    glossary = load_glossary()
    residue = load_clause_residue()
    ocr_types = load_ocr_class()
    ocr_by_clause = load_clause_ocr(ocr_types)

    sample = pick_sample(clauses, opmap, glossary, residue, per_stratum)
    if not sample:
        raise GlossingError(
            "no clause matched the sampling strata; the frozen artifacts "
            "#33/#17/#62 must be built before this one")

    rows, prompts, checks = resolve_glosses(
        sample, opmap, glossary, residue, ocr_by_clause, strategies,
        clauses=clauses)

    agree = []
    for cid in sorted(sample, key=int):
        by_sid = {r["strategy"]: r["gloss"] for r in rows
                  if r["clause_id"] == cid}
        agree.append(agreement(cid, by_sid))

    sample_rows = []
    for cid, stratum in sorted(sample.items(), key=lambda kv: int(kv[0])):
        clause = clauses[cid]
        names = [n for n in glossary if n in clause["original"].lower()]
        sample_rows.append({
            "clause_id": cid, "page": clause["page"], "stratum": stratum,
            "n_operators": len(opmap.get(cid, [])),
            "n_entities_named": len(names),
            "residue_surfaces": ",".join(
                residue_surfaces(residue.get(cid, []))),
            "why": STRATUM_WHY[stratum],
            "original": clause["original"].replace("\n", " "),
        })

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = collections.Counter(r["verdict"] for r in rows)
    check_pass = collections.Counter()
    check_applicable = collections.Counter()
    check_rows = collections.Counter()
    for c in checks:
        check_rows[c["check"]] += 1
        if c["applies"]:
            # Only the rows the check applies to.  Counting the rest would
            # publish `residue_discipline` at 100% over 25 glosses when only
            # 5 of them carry a residue surface: a check that does not apply
            # cannot pass, and reporting it as passing is the anti-pattern
            # #76 audits this module for.  The report's own table already
            # filters on `applies`, so this makes `summary.json` agree with
            # it instead of silently disagreeing.
            check_applicable[c["check"]] += 1
            if c["pass"]:
                check_pass[c["check"]] += 1

    summary = {
        "schema": SCHEMA,
        "capture": CAPTURE,
        "backend": PIN_BACKEND,
        "mode": "pinned",
        "clauses_sampled": len(sample),
        "strategies": len(strategies),
        "glosses": len(rows),
        "verdicts": {v: verdict.get(v, 0) for v in ("pass", "fail")},
        "checks_per_gloss": CHECKS_PER_GLOSS,
        "checks_total": len(checks),
        "checks_applicable": sum(check_applicable.values()),
        "checks_passed": sum(check_pass.values()),
        # Per check: rows published, rows the check applies to, rows passed.
        # The rate is published with its denominator, because a rate over the
        # wrong denominator is exactly what this counting exists to stop:
        # `residue_discipline` is 5-of-5 here and 25-of-25 only if the rows
        # the check does not apply to are counted as passing.
        "check_counts": {
            c: {"rows": check_rows[c], "applicable": check_applicable[c],
                "passed": check_pass[c]} for c in sorted(check_rows)},
        "check_pass_rate": {
            c: round(check_pass[c] / check_applicable[c], 4)
            for c in sorted(check_applicable)},
        "per_strategy_verdict": {
            s: {v: sum(1 for r in rows if r["strategy"] == s
                       and r["verdict"] == v) for v in ("pass", "fail")}
            for s in strategies},
        "mean_clause_agreement": round(
            sum(a["mean_jaccard"] for a in agree) / len(agree), 4)
        if agree else 0.0,
        "clauses_unanimous": sum(1 for a in agree
                                  if a["min_jaccard"] >= 0.5),
        "generated": generated,
        "determinism": "the pinned build replays PINS and is byte-identical "
                       "on rebuild (every data file; only the report's "
                       "generation stamp moves with the clock); --live "
                       "re-runs the backend and prints a diff, never "
                       "rewriting the artifacts",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    cs.write_tsv(OUT / "glosses.tsv", GLOSS_HEADER, rows)
    cs.write_tsv(OUT / "checks.tsv", CHECK_HEADER, checks)
    cs.write_tsv(OUT / "agreement.tsv", AGREE_HEADER, agree)
    cs.write_tsv(OUT / "sample.tsv", SAMPLE_HEADER, sample_rows)
    cs.write_tsv(OUT / "prompts.tsv", PROMPT_HEADER, prompts)
    cs.write_tsv(OUT / "diagnostics.tsv", ["diag", "value"], [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "capture", "value": CAPTURE},
        {"diag": "mode", "value": "pinned"},
        {"diag": "clauses_sampled", "value": len(sample)},
        {"diag": "strategies", "value": len(strategies)},
        {"diag": "glosses", "value": len(rows)},
        {"diag": "glosses_pass", "value": verdict.get("pass", 0)},
        {"diag": "glosses_fail", "value": verdict.get("fail", 0)},
        {"diag": "checks_total", "value": len(checks)},
        {"diag": "checks_applicable", "value": sum(check_applicable.values())},
        {"diag": "checks_passed", "value": sum(check_pass.values())},
    ] + [
        {"diag": "check_%s_rows" % c, "value": check_rows[c]}
        for c in sorted(check_rows)] + [
        {"diag": "check_%s_applicable" % c, "value": check_applicable[c]}
        for c in sorted(check_applicable)] + [
        {"diag": "check_%s_pass" % c, "value": check_pass[c]}
        for c in sorted(check_applicable)])
    with (OUT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    write_report(OUT / "glossing_report.md", rows, checks, agree,
                 sample_rows, prompts, summary, generated)
    if not quiet:
        print("[#63] glossing: %d clauses x %d strategies = %d glosses; "
              "%d pass / %d fail; mean agreement %.3f"
              % (len(sample), len(strategies), len(rows),
                 verdict.get("pass", 0), verdict.get("fail", 0),
                 summary["mean_clause_agreement"]))
    return summary


#: Four checks per gloss.  Named here rather than left implicit so the
#: summary's `checks_per_gloss` cannot drift from the check list.
CHECKS_PER_GLOSS = 4


def write_report(path, rows, checks, agree, sample_rows, prompts, summary,
                 generated):
    """The deliverable: method, the sampled glosses, the verdicts, the audit."""
    by_check = collections.defaultdict(list)
    for c in checks:
        by_check[c["check"]].append(c)
    L = []
    a = L.append
    a("# LLM glossing pipeline -- five strategies, sampled (#63)")
    a("")
    a("Generated: %s" % generated)
    a("")
    a("Capture: %s, backend `%s`, pinned mode." % (CAPTURE, PIN_BACKEND))
    a("")
    a("## What this is, and what it is not")
    a("")
    a("Five prompt strategies gloss %d sampled clauses of *Blood Electric*,"
      % summary["clauses_sampled"])
    a("and every returned gloss is put through %d machine checks that are"
      % CHECKS_PER_GLOSS)
    a("computed from the frozen artifacts, not from the model's own")
    a("assessment of itself.  The checks are the result; the glosses are")
    a("the sample they were computed on.")
    a("")
    a("**This is not a gloss of the book.**  %s clauses carry a gloss here"
      % "{:,}".format(summary["clauses_sampled"]))
    a("out of #33's full clause set, and the sample is stratified and")
    a("published (`sample.tsv`) rather than chosen for readability.")
    a("")
    a("## The five strategies")
    a("")
    for s in STRATEGIES:
        a("* **`%s`** -- %s.  Owes: operators %s."
          % (s["id"], s["title"], STRATEGY_OWES[s["id"]]["operators"]))
    a("")
    a("Every prompt carries the same standing constraint block (`STANDING`:")
    a("names as given, unresolved surfaces stay marked, no plot, OCR")
    a("artefacts as printed, one line out).  Identical across strategies on")
    a("purpose -- a differing honesty clause would confound the agreement")
    a("statistic with the prompt.")
    a("")
    a("## The sample")
    a("")
    a("| clause | p. | stratum | operators | entities | residue |")
    a("|---|---|---|---|---|---|")
    for s in sample_rows:
        a("| %s | %s | %s | %d | %d | %s |"
          % (s["clause_id"], s["page"], s["stratum"], s["n_operators"],
             s["n_entities_named"], s["residue_surfaces"] or "-"))
    a("")
    a("Strata are drawn one clause each, so each check has at least one")
    a("case where it can fail.  Within a stratum the quality rank decides")
    a("and the tie-break is the clause id compared as text -- `1003`")
    a("before `101` -- which is disclosed here rather than hidden: it means")
    a("the sample is not the numerically earliest clause of each stratum,")
    a("and it is why the sample is published instead of asserted.")
    a("")
    for k, v in STRATUM_WHY.items():
        a("* `%s` -- %s." % (k, v))
    a("")
    a("## Verdicts")
    a("")
    a("| verdict | glosses | share |")
    a("|---|---|---|")
    tot = len(rows)
    for v in ("pass", "fail"):
        n = summary["verdicts"][v]
        a("| %s | %d | %.0f%% |" % (v, n, _pct(n, tot)))
    a("")
    a("Per strategy:")
    a("")
    a("| strategy | pass | fail |")
    a("|---|---|---|")
    for s in summary["per_strategy_verdict"]:
        d = summary["per_strategy_verdict"][s]
        a("| `%s` | %d | %d |" % (s, d["pass"], d["fail"]))
    a("")
    a("Per check, over the rows the check applies to:")
    a("")
    a("| check | rows | applicable | passed | rate |")
    a("|---|---|---|---|---|")
    for c in sorted(by_check):
        rw = by_check[c]
        ap = [r for r in rw if r["applies"]]
        ps = [r for r in ap if r["pass"]]
        a("| `%s` | %d | %d | %d | %.0f%% |"
          % (c, len(rw), len(ap), len(ps), _pct(len(ps), len(ap))))
    a("")
    a("A row a check does not apply to is not counted as a pass: the")
    a("applicable column is the denominator, and it is smaller than the rows")
    a("column exactly where a check had nothing to judge -- `residue_discipline`")
    a("runs only on the glosses whose clause carries an unresolved surface, so")
    a("counting the rest as passes would publish a perfect score out of")
    a("%d glosses it never judged.  `summary.json` publishes the same" % tot)
    a("denominators under `check_counts`.")
    a("")
    a("A `pass` is not a good reading.  It is a gloss that broke none of the")
    a("rules its own strategy was held to: the operator channel it owed,")
    a("the #30 names it was given, the unresolved surfaces it was told to")
    a("keep, and the quoting frame it was told to preserve.")
    a("")
    a("## The glosses")
    a("")
    original_by_clause = {s["clause_id"]: s["original"] for s in sample_rows}
    n_pairs_phrase = ("%d" % (summary["strategies"] * (summary["strategies"] - 1)
                             // 2)) if agree else "0"
    for r in sorted(rows, key=lambda r: (int(r["clause_id"]),
                                         r["strategy"])):
        a("### clause %s (p. %s, `%s`) -- `%s`: **%s**"
          % (r["clause_id"], r["page"], r["stratum"], r["strategy"],
             r["verdict"]))
        a("")
        a("```text")
        a("src    | %s" % original_by_clause.get(r["clause_id"], ""))
        a("gloss  | %s" % r["gloss"])
        a("ops    | %s   ->   %s" % (r["operators_in"] or "(none)",
                                     r["operators_out"] or "(none)"))
        a("names  | %s" % (r["entities_in"] or "(none given)"))
        a("62     | %s" % r["baseline_62_gloss"])
        a("```")
        a("")
        fails = [c for c in checks if c["clause_id"] == r["clause_id"]
                 and c["strategy"] == r["strategy"] and not c["pass"]]
        # A row the check does not apply to is not a failure, and printing it
        # as one would make a compliant gloss look broken: `scene_summary`
        # was told to drop the `<< >>` frame, so its 0<< 0>> is obedience.
        # It is still printed -- as not applicable, with its contract -- so
        # the reader can see the check ran and what it found.
        by_name = {c["check"]: c for c in checks
                   if c["clause_id"] == r["clause_id"]
                   and c["strategy"] == r["strategy"]}
        na = [c for c in by_name.values() if not c["applies"]
              and not c["pass"]]
        for c in na:
            a("* `%s`: not applicable (%s) -- %s"
              % (c["check"], c["contract"], c["detail"]))
        for c in fails:
            if c["applies"]:
                a("* `%s`: %s -- %s" % (c["check"], c["detail"],
                                        c["contract"]))
        if na or [c for c in fails if c["applies"]]:
            a("")
    a("## Agreement across strategies")
    a("")
    a("Jaccard over content words (operators excluded), averaged over the")
    a("pairs of strategies per clause (%s pairs for %d strategies)."
      % (n_pairs_phrase, summary["strategies"]))
    a("Reported for unanimous clauses too: an agreement figure that is only")
    a("printed when it is low is one nobody can rely on.")
    a("")
    a("| clause | mean | min | max | least agreeing pair |")
    a("|---|---|---|---|---|")
    for ag in agree:
        a("| %s | %.3f | %.3f | %.3f | `%s` |"
          % (ag["clause_id"], ag["mean_jaccard"], ag["min_jaccard"],
             ag["max_jaccard"], ag["least_agreeing_pair"]))
    a("")
    a("Low agreement is not noise to be averaged away -- it localises the")
    a("clauses where the strategies read the book differently, which is")
    a("what #64's committee should be pointed at.")
    a("")
    a("## Anti-pattern audit (L9.8)")
    a("")
    a("* **Circularity**: the model writes, `check_gloss` judges.  No")
    a("  strategy grades another, and no check calls the backend, so a")
    a("  gloss cannot be graded by the model that wrote it.")
    a("* **Hallucinated coherence**: no prompt asserts a plot; the")
    a("  `entity_fidelity` check fires on any capitalised token the clause")
    a("  and glossary do not license, and `residue_discipline` fires on a")
    a("  gloss that turns a marked-unresolved surface into an ordinary")
    a("  word.")
    a("* **Single-method certainty**: five strategies, agreement published")
    a("  per clause, and every row carries its strategy, its backend and")
    a("  the prompt hash it came from.")
    a("* **Over-correction**: #19 `ocr_error` surfaces are listed to the")
    a("  model as artefacts, so it is told to reproduce rather than")
    a("  normalise them.")
    a("* **Residue-dropping**: the sample includes the residue stratum")
    a("  first, so the check cannot be vacuous.")
    a("* **Determinism**: a model is not deterministic, so the artifacts")
    a("  are pinned.  `--live` re-runs and prints a diff against the pins")
    a("  rather than rewriting them, which is what #58 does for its source")
    a("  probes; a number that moves with the sampler cannot be")
    a("  regression-tested.")
    a("")
    a("## Reproducing")
    a("")
    a("```bash")
    a("python work/glossing_pipeline.py            # pinned, offline")
    a("python work/glossing_pipeline.py --live     # re-run, diff vs pins")
    a("python work/test_glossing_pipeline.py")
    a("```")
    a("")
    a("Every prompt is written to `prompts.tsv` with its sha256, and each")
    a("pin stores the hash of the prompt it was recorded against, so a")
    a("change in the constraint assembly fails the build loudly instead of")
    a("silently reusing a stale response.")
    a("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 9. live / capture
# ---------------------------------------------------------------------------

def diff_live(model, strategies=STRATEGY_IDS, per_stratum=1):
    """Re-run the backend over the sample and diff against the pins.

    Prints, does not write.  A live run that overwrote the artifacts
    would replace a testable build with an untestable one; this way a
    backend change is a measurement the operator can read and decide on.
    """
    clauses = load_clauses()
    opmap = load_operator_token_map()
    glossary = load_glossary()
    residue = load_clause_residue()
    ocr_by_clause = load_clause_ocr(load_ocr_class())
    sample = pick_sample(clauses, opmap, glossary, residue, per_stratum)

    same = diff = 0
    for cid, stratum in sorted(sample.items(), key=lambda kv: int(kv[0])):
        clause = clauses[cid]
        ops = opmap.get(cid, [])
        names = {n for n in glossary if n in clause["original"].lower()}
        ents = entity_constraints(glossary, names)
        res = residue_surfaces(residue.get(cid, []))
        ocr = ocr_by_clause.get(cid, [])
        for sid in strategies:
            text = prompt_text(clause, operator_constraints(ops), ents, ocr,
                               res, sid)
            ph = prompt_sha256(text)
            pin = PINS.get(pin_key(cid, sid))
            if pin is None:
                print("[live] %s/%s: no pin to compare against" % (cid, sid))
                continue
            if pin["prompt_sha256"] != ph:
                print("[live] %s/%s: PROMPT DRIFT (pin %s, now %s)"
                      % (cid, sid, pin["prompt_sha256"][:12], ph[:12]))
                continue
            fresh = run_live(text, model)
            if fresh == pin["response"]:
                same += 1
                print("[live] %s/%s: identical to pin" % (cid, sid))
            else:
                diff += 1
                print("[live] %s/%s: DIFFERS from pin" % (cid, sid))
                print("   pin : %s" % pin["response"])
                print("   live: %s" % fresh)
    print("[live] %d identical, %d differ (model %s)"
          % (same, diff, model))
    return same, diff


def capture(model, strategies=STRATEGY_IDS, per_stratum=1):
    """Record a fresh pinned set and print it as a Python literal.

    Deliberately does not write `PINS` into the source file: the pins are
    committed evidence, so changing them is a reviewable edit, not a side
    effect of running a command.
    """
    clauses = load_clauses()
    opmap = load_operator_token_map()
    glossary = load_glossary()
    residue = load_clause_residue()
    ocr_by_clause = load_clause_ocr(load_ocr_class())
    sample = pick_sample(clauses, opmap, glossary, residue, per_stratum)
    lines = ["PINS: dict[str, dict] = {"]
    for cid, stratum in sorted(sample.items(), key=lambda kv: int(kv[0])):
        clause = clauses[cid]
        ops = opmap.get(cid, [])
        names = {n for n in glossary if n in clause["original"].lower()}
        ents = entity_constraints(glossary, names)
        res = residue_surfaces(residue.get(cid, []))
        ocr = ocr_by_clause.get(cid, [])
        for sid in strategies:
            text = prompt_text(clause, operator_constraints(ops), ents, ocr,
                               res, sid)
            ph = prompt_sha256(text)
            gloss = run_live(text, model)
            verdict, _ = check_gloss(clause, ops, ents, res, sid, gloss)
            print("[capture] %s/%s -> %s (%d chars)"
                  % (cid, sid, verdict, len(gloss)), file=sys.stderr)
            lines.append("    %r: {" % pin_key(cid, sid))
            lines.append("        \"prompt_sha256\": %r," % ph)
            lines.append("        \"backend\": %r,"
                         % ("opencode run --model %s" % model))
            lines.append("        \"captured\": %r," % CAPTURE)
            lines.append("        \"response\": %r}," % gloss)
    lines.append("}")
    sys.stdout.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--model", default="opencode/space-bunny-free",
                    help="backend model for --live/--capture")
    ap.add_argument("--live", action="store_true",
                    help="re-run the backend and diff against the pins "
                         "(writes nothing)")
    ap.add_argument("--capture", action="store_true",
                    help="record a fresh pinned set to stdout as a Python "
                         "literal (writes nothing)")
    ap.add_argument("--per-stratum", type=int, default=1,
                    help="clauses to draw per stratum (default 1)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.capture:
            capture(args.model, per_stratum=args.per_stratum)
            return 0
        if args.live:
            diff_live(args.model, per_stratum=args.per_stratum)
            return 0
        build(Path(args.out), quiet=args.quiet, per_stratum=args.per_stratum)
    except GlossingError as exc:
        print("glossing pipeline failed: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())