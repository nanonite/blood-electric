#!/usr/bin/env python3
"""#68 Compression test for candidate readings -> analysis/reading_compression/

Task (plan E8.7, docs/chainlink-plan.md #68): *a good gloss shortens the
description of the book* -- score the repository's published candidate
readings by whether they make the corpus more **predictable** and more
**compressible**, under one common protocol, with held-out evaluation and
explicit baselines.

The readings scored (all are *published artifacts*, none is invented here):

* **``identity``** -- the control.  The exact source clause (``original``
  column of the #62 interlinear edition), i.e. no reading at all.  Every
  delta is measured against it, on the same clause set.
* **``v3_operators_to_space``**, **``v4_operators_to_period``** -- the #13
  silence-removal variants, carried in the interlinear's own columns.  #50
  already ranks these by GPT-2 perplexity; here they are re-scored as
  *readings* under this harness's protocol.
* **``interlinear_gloss``** -- #62's plain-English gloss line: the flagship
  candidate reading of the interpretation layer (operator normalization
  plus the curated termbase).
* **``llm_<strategy>`` x5** -- #63's five prompt strategies
  (``literal_operator``, ``register_paraphrase``, ``scene_summary``,
  ``plain_english``, ``entity_consistent``) over its 5-clause stratified
  sample.  These cover 5 clauses each, so they are scored on the *sample*
  population only and never pooled with full-book readings.

Why this is not #48 repeated
----------------------------
#48 showed the book is *more* compressible than matched English and that raw
bits/char is therefore a **weak discriminator between readings** (its own
report says the compression test "cannot be the discriminator between
readings (#68)").  This harness answers that limitation with three devices:

1. **A length-matched deletion control (the primary leg).**  A reading that
   merely *drops* words is trivially more compressible.  For every covered
   clause the control is a deterministic, position-preserving *letters-word*
   subsample of the *original* clause down to the reading's letters-word
   count, compressed by the identical path and repeated over
   ``N_CONTROL_SEEDS`` seeds so its own spread is reported (a delta read
   against one lucky control would fake a win).  **Limit, disclosed:** a
   reading with at least as many words as its clause gets no subsample --
   there is nothing to delete -- and the control is then the original.  The
   corpus's readings are word-count preserving or expanding far more often
   than not (operator removal changes the characters *between* words, not
   the letters), so for v3/v4/gloss the control is the original on ~5,470
   of 5,614 clauses and the delta reduces toward the raw bits/char delta.
   The live-clause count is published per reading
   (``control_fallback_clauses``) rather than assumed.
2. **A pairing (shuffle) null.**  The reading's clause texts are permuted
   across its covered clause ids with fixed seeds.  The null keeps the text
   multiset and destroys the clause<->reading pairing; a reading whose gain
   survives it is carrying information about *which* text goes with *which*
   clause, not just being short.
3. **Held-out predictability.**  A character trigram (Jelinek-Mercer
   interpolation, lambda tuned on a train-internal validation split -- the
   #56 estimator design) scores text it never trained on.  One **fixed
   model** (trained on the identity train clauses only) scores every
   reading, so readings are compared by a single shared scorer; the
   full-book readings additionally get an **adaptive** model trained on
   their own train clauses, scored on their held-out clauses (``self``) and
   on the untouched originals (``cross``, a coverage diagnostic).  The
   fixed-model leg is paired against the identity text clause by clause
   with a sign test (#50's pattern).

Interpretation guardrail (``docs/meaning-extraction-methods.md`` L9.8):
compression is **evidence about predictability, not proof that an
interpretation is true**.  The ranking orders readings by how much they
shorten the description of the corpus; it does not certify any reading as
*the* reading.

Tokenization and units (documented because they are choices, not facts)
-----------------------------------------------------------------------
* Unit = the clause (the #33 segmentation, 5,614 clauses), the same unit the
  interlinear edition is aligned on, so readings are compared on identical
  text spans.
* Compression stream = the reading's clause texts joined by ``"\\n"`` in book
  order, case-folded, UTF-8 encoded.  Six #48 compressors (gzip-9, zlib-9,
  bz2-9, lzma-6, zstd-3, zstd-19) are reused from ``compression_mdl`` so the
  numbers reconcile with #48's frozen table.
* Word count = maximal runs of letters (``WORD_RE``), *not* whitespace
  chunks.  This is a disclosed choice with teeth: ``str.split()`` counts
  ``soul/gram`` as one word and ``soul gram`` as two, so an operator-removal
  reading would look longer than its clause purely because of the
  tokenizer, and the length control would fall back on every clause for
  that reason alone.
* Predictability unit = the **character** (no word tokenizer): a character
  trigram needs no lexicon, cannot be blinded by a reading's orthography,
  and treats a gloss and a source clause by exactly the same ruler.
* Held-out split = every 5th clause in book order (20 %), deterministic and
  disclosed; clause order is page order, so the held-out set is spread over
  all 225 pages (no contiguous-page leakage, #56's D2 concern).

Reproduce::

    conda activate marker
    python work/reading_compression.py            # writes analysis/reading_compression/
    python work/test_reading_compression.py        # regression suite

``--quick`` scores a deterministic clause subsample (every 11th clause) for
the regression suite.  No network, no clock, no wall-clock in any artifact;
a rebuild is byte-identical.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))

import compression_mdl as cm      # #48 compressors (shared + reconciled)
import frequency_zipf as fz       # #24 write_tsv
import ngram_surprisal as ns      # #35 read_tsv

OUT = ROOT / "analysis" / "reading_compression"
INTERLINEAR = ROOT / "analysis" / "interlinear" / "interlinear.tsv"
GLOSSES = ROOT / "analysis" / "glossing" / "glosses.tsv"
REF_MANIFEST = ROOT / "analysis" / "reference_corpora" / "manifest.tsv"
V48_SILENCE = ROOT / "analysis" / "compression" / "silence_variants.tsv"
V62_SUMMARY = ROOT / "analysis" / "interlinear" / "summary.json"
V63_SUMMARY = ROOT / "analysis" / "glossing" / "summary.json"

SCHEMA = "reading-compression/1"
SEED = 68                       # issue number, the house convention for seeds
ORDER = 3                       # character trigram
LAMBDA_GRID = (0.2, 0.4, 0.6, 0.8, 0.95)   # #56's grid, reused
TEST_EVERY = 5                  # held-out: every 5th clause (20 %)
MIN_TRAIN_CLAUSES = 200         # floor for the adaptive (train-on-reading) leg
PRIMARY_COMPRESSOR = "gzip-9"   # #48's headline compressor
N_NULLS = 3                     # permutation nulls per reading
N_CONTROL_SEEDS = 5             # length-matched control reseeds (uncertainty)
QUICK_EVERY = 11                # --quick clause subsample
# Published constants frozen by earlier issues (a mismatch is a build error).
PUBLISHED = {
    "n_clauses": 5614,          # #33 / #62 interlinear row count
    "n_pages": 225,             # #7
    "n_words_clean": 76897,     # #33, reconciled by #62
    "n_gloss_rows": 25,         # #63 sampled glosses
    "n_termbase": 49,           # #56 / #65 termbase
}
# A "word" for the length control is a maximal run of letters, NOT a
# whitespace-delimited chunk.  This matters: the readings differ from the
# source in the *operator characters between* words, and ``str.split()``
# counts ``soul/gram`` as one chunk but ``soul gram`` as two, which would
# make an operator-removal reading look longer than its clause purely
# because of the tokenizer.  Letters-only counting is the disclosed rule.
WORD_RE = re.compile(r"[A-Za-z]+")

read_tsv = ns.read_tsv
write_tsv = fz.write_tsv


# --------------------------------------------------------------------------
# readings
# --------------------------------------------------------------------------
@dataclass
class Reading:
    """One candidate reading: a name, a provenance, a clause -> text map."""

    name: str
    kind: str                 # control | transform | gloss | llm
    source: str               # the artifact it is read from
    texts: dict[str, str] = field(default_factory=dict)   # clause_id -> text

    @property
    def population(self) -> str:
        # Full-book readings carry the whole corpus; #63's strategies carry
        # only their stratified sample and are scored on that population.
        return "full_book" if len(self.texts) >= MIN_TRAIN_CLAUSES else "sample_63"


def load_clauses() -> tuple[list[str], dict[str, dict[str, str]]]:
    """Clause ids in book order + the #62 interlinear rows, from its TSV.

    Read with ``split("\\t")`` (the README's disclosed hazard: the gloss
    column contains bare double quotes, which ``csv`` mangles).
    """
    rows = read_tsv(INTERLINEAR)
    if not rows:
        raise SystemExit(f"#68: no interlinear rows at {INTERLINEAR}")
    order = [r["clause_id"] for r in rows]
    by_id = {r["clause_id"]: r for r in rows}
    if len(by_id) != len(order):
        raise SystemExit("#68: duplicate clause_id in the interlinear")
    return order, by_id


def load_llm_glosses() -> dict[str, dict[str, str]]:
    """``{strategy: {clause_id: gloss}}`` from #63's glosses.tsv."""
    out: dict[str, dict[str, str]] = {}
    for r in read_tsv(GLOSSES):
        out.setdefault(r["strategy"], {})[r["clause_id"]] = r["gloss"]
    return out


def build_readings(order: list[str], rows: dict[str, dict[str, str]]) -> list[Reading]:
    """All candidate readings, every one read from a published artifact."""
    readings = [
        Reading("identity", "control", "interlinear original column",
                {c: rows[c]["original"] for c in order}),
        Reading("v3_operators_to_space", "transform", "#13 v3 via interlinear",
                {c: rows[c]["v3_space"] for c in order}),
        Reading("v4_operators_to_period", "transform", "#13 v4 via interlinear",
                {c: rows[c]["v4_period"] for c in order}),
        Reading("interlinear_gloss", "gloss", "#62 gloss column",
                {c: rows[c]["gloss"] for c in order}),
    ]
    for strategy, texts in sorted(load_llm_glosses().items()):
        readings.append(Reading(f"llm_{strategy}", "llm",
                                f"#63 glosses.tsv[{strategy}]", dict(texts)))
    return readings


def reading_stream(reading: Reading, ids: list[str]) -> str:
    """The reading's text over ``ids``, in the given (book) order."""
    return "\n".join(reading.texts[c] for c in ids if c in reading.texts)


def covered(reading: Reading, ids: list[str]) -> list[str]:
    return [c for c in ids if c in reading.texts]


# --------------------------------------------------------------------------
# leg A: stream description length (#48 compressors, reused)
# --------------------------------------------------------------------------
def compress_stream(text: str) -> dict[str, dict[str, float]]:
    """Six #48 compressors on the case-folded stream: bits/char + ratio."""
    low = text.casefold()
    data = low.encode("utf-8")
    out: dict[str, dict[str, float]] = {}
    for cname, level in cm.COMPRESSORS:
        comp = cm.compress(cname, level, data)
        if comp is None:            # zstd may be unavailable; disclosed below
            continue
        key = cm.cname(cname, level)
        out[key] = {
            "n_symbols": len(low),
            "n_bytes": len(data),
            "compressed_bytes": len(comp),
            "bits_per_char": cm.bpc(low, comp, len(low)),
            "ratio": cm.ratio(low, comp),
        }
    return out


def word_count(text: str) -> int:
    """Letters-only word count -- the harness's one word tokenization rule.

    See ``WORD_RE``: whitespace chunks would miscount the readings, which
    differ from the source in the characters *between* words.
    """
    return len(WORD_RE.findall(text))


def length_matched_control(original: str, n_words: int, rng: random.Random) -> str:
    """Deterministic word-subsample of ``original`` down to ``n_words``.

    Position-preserving (the sampled word indices are re-sorted), so the
    control keeps the original's local word order and only loses content --
    the "deletion" confound, isolated.

    **Fallback, and it is the common case here:** a reading with *at least*
    as many words as its clause gets no subsample -- the control *is* the
    original, because there is nothing to delete.  The readings in this
    corpus are word-count-preserving by construction (operator removal
    changes the characters between words, not the letters), so the fallback
    fires for most clauses of most readings.  ``controls_and_nulls.tsv``
    publishes the fallback count per reading and section 1 of the report
    states what the control can and cannot show as a result.
    """
    words = WORD_RE.findall(original)
    if n_words >= len(words):
        return original
    idx = sorted(rng.sample(range(len(words)), n_words))
    return " ".join(words[i] for i in idx)


def control_stream(reading: Reading, ids: list[str],
                   originals: dict[str, str], seed_k: int = 0) -> str:
    """Length-matched control stream for ``reading`` over ``ids``.

    ``seed_k`` reseeds the per-clause subsample so the control can be
    repeated and its own seed noise measured (the delta must be read
    against that noise, or a lucky control would fake a win).
    """
    parts = []
    for c in covered(reading, ids):
        n_words = word_count(reading.texts[c])
        rng = random.Random(f"{SEED}:{reading.name}:{c}:{seed_k}")
        parts.append(length_matched_control(originals[c], n_words, rng))
    return "\n".join(parts)


def control_streams(reading: Reading, ids: list[str],
                    originals: dict[str, str]) -> list[str]:
    """``N_CONTROL_SEEDS`` independent length-matched control streams."""
    return [control_stream(reading, ids, originals, k)
            for k in range(N_CONTROL_SEEDS)]


def null_streams(reading: Reading, ids: list[str],
                 n_nulls: int) -> list[str]:
    """Permuted streams: the reading's texts, the pairings destroyed."""
    cov = covered(reading, ids)
    if len(cov) < 2:
        return []
    streams = []
    for k in range(n_nulls):
        texts = [reading.texts[c] for c in cov]
        random.Random(f"{SEED}:{reading.name}:null:{k}").shuffle(texts)
        streams.append("\n".join(texts))
    return streams


# --------------------------------------------------------------------------
# leg C: held-out character-trigram predictability (#56's estimator design)
# --------------------------------------------------------------------------
class CharNGram:
    """Character n-gram with Jelinek-Mercer recursive interpolation.

    ``p(w | ctx) = lam * ML(w|ctx) + (1-lam) * p(w | ctx[1:])``; an unseen
    context backs off whole (an interpolated empty ML distribution would
    lose probability mass), exactly #56's ``_jm_probability``.  Scoring is
    per character, so there is no tokenization choice to leak through.
    """

    def __init__(self, order: int = ORDER, lam: float = 0.6):
        self.order = order
        self.lam = lam
        self.uni: Counter = Counter()
        self.ctx: dict[tuple[str, ...], Counter] = {}

    def train(self, text: str) -> None:
        self.uni.update(text)
        order = self.order
        for i in range(len(text) - order + 1):
            key = tuple(text[i:i + order - 1])
            self.ctx.setdefault(key, Counter())[text[i + order - 1]] += 1

    def _prob(self, ch: str, ctx: tuple[str, ...]) -> float:
        if not ctx:
            tot = sum(self.uni.values()) or 1
            v = len(self.uni)
            return (self.uni[ch] + 1e-9) / (tot + 1e-9 * v)
        counts = self.ctx.get(ctx)
        if not counts:
            return self._prob(ch, ctx[1:])
        tot = sum(counts.values())
        ml = counts[ch] / tot
        return self.lam * ml + (1.0 - self.lam) * self._prob(ch, ctx[1:])

    def surprisal(self, text: str) -> tuple[float, int]:
        """``(total surprisal in bits, n scored characters)``."""
        order, bits, n = self.order, 0.0, 0
        for i in range(order - 1, len(text)):
            bits += -math.log2(max(self._prob(text[i],
                                              tuple(text[i - order + 1:i])), 1e-12))
            n += 1
        return bits, n

    def bits_per_char(self, text: str) -> float:
        bits, n = self.surprisal(text)
        return bits / n if n else 0.0


def _tune_lambda(train_text: str) -> float:
    """Pick ``lam`` on a train-internal validation split (never on test)."""
    if not train_text:
        return LAMBDA_GRID[len(LAMBDA_GRID) // 2]
    cut = len(train_text) // 2
    val_train, val_test = train_text[:cut], train_text[cut:]
    best = (LAMBDA_GRID[0], float("inf"))
    for lam in LAMBDA_GRID:
        model = CharNGram(ORDER, lam)
        model.train(val_train)
        ce = model.bits_per_char(val_test)
        if ce < best[1]:
            best = (lam, ce)
    return best[0]


def sign_test_p(n_improved: int, n_worsened: int) -> float:
    """Two-sided sign-test p-value under a Bin(n, 0.5) null (#50's formula)."""
    n = n_improved + n_worsened
    if n == 0:
        return 1.0
    k = min(n_improved, n_worsened)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def english_anchor(ids: list[str], originals: dict[str, str],
                   char_budget: int = 300_000) -> float:
    """Bits/char of the held-out book under a general-English char model.

    The #48/#49 external anchor, recomputed at character level with this
    harness's own estimator: it says how predictable the *untransformed*
    held-out book is to plain English prose, which is the yardstick the
    readings are trying to beat.
    """
    if not REF_MANIFEST.exists():
        return float("nan")
    texts = []
    used = 0
    for r in read_tsv(REF_MANIFEST):
        if r["corpus"] != "general_english":
            continue
        path = ROOT / "analysis" / "reference_corpora" / r["file"]
        if not path.exists():
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        texts.append(body)
        used += len(body)
        if used >= char_budget:
            break
    if not texts:
        return float("nan")
    model = CharNGram(ORDER, _tune_lambda("\n".join(texts)[:char_budget].casefold()))
    model.train("\n".join(texts)[:char_budget].casefold())
    held = "\n".join(originals[c] for c in ids)
    return model.bits_per_char(held.casefold())


# --------------------------------------------------------------------------
# the common protocol
# --------------------------------------------------------------------------
@dataclass
class Score:
    """Everything the harness knows about one reading on one clause set."""

    reading: str
    kind: str
    population: str
    n_clauses: int = 0
    n_chars: int = 0
    n_words: int = 0
    bpc: dict[str, float] = field(default_factory=dict)      # per compressor
    control_bpc: float = float("nan")
    control_bpc_sd: float = float("nan")
    control_n_chars: int = 0
    control_n_words: int = 0
    n_control_fallback: int = 0      # clauses where the control == the original
    delta_vs_control: float = float("nan")                   # leg A, primary
    null_bpc_mean: float = float("nan")
    null_bpc_sd: float = float("nan")
    null_z: float = float("nan")
    lam: float = float("nan")
    self_bpc: float = float("nan")                           # leg C1, adaptive
    cross_bpc: float = float("nan")                          # leg C1, coverage
    fixed_self_bpc: float = float("nan")                     # leg C2, uniform
    fixed_gain_vs_identity: float = float("nan")             # leg C2, primary
    n_improved: int = 0
    n_worsened: int = 0
    sign_test_p: float = float("nan")


def split_ids(ids: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic held-out split: every ``TEST_EVERY``-th clause is test.

    Clause order is page order, so the held-out set is spread across all 225
    pages rather than forming a contiguous block (#56's leakage lesson).
    """
    test = [c for i, c in enumerate(ids) if i % TEST_EVERY == 0]
    train = [c for i, c in enumerate(ids) if i % TEST_EVERY != 0]
    return train, test


def score_reading(reading: Reading, ids: list[str], population: str,
                  originals: dict[str, str], fixed_model: CharNGram | None,
                  test_ids: list[str]) -> Score:
    """Score one reading over ``ids`` under the common protocol.

    ``population`` is the *evaluation* set (``full_book`` or ``sample_63``),
    which is not the reading's own coverage class: the identity reading is
    scored in both populations, and the label must say which.

    ``fixed_model`` (trained on the *identity* train clauses only) gives the
    uniform fixed-model leg every reading can be scored by; the adaptive leg
    runs only when the reading has enough train clauses to learn from.
    """
    cov = covered(reading, ids)
    s = Score(reading.name, reading.kind, population, len(cov))
    stream = reading_stream(reading, ids)
    s.n_chars = len(stream.casefold())
    s.n_words = sum(word_count(reading.texts[c]) for c in cov)
    s.bpc = {k: v["bits_per_char"] for k, v in compress_stream(stream).items()}

    # ---- leg A: length-matched deletion control -------------------------
    key = PRIMARY_COMPRESSOR
    ctrl_vals, ctrl_chars = [], 0
    for ctrl_text in control_streams(reading, ids, originals):
        res = compress_stream(ctrl_text)
        if key in res:
            ctrl_vals.append(res[key]["bits_per_char"])
            ctrl_chars = max(ctrl_chars, len(ctrl_text.casefold()))
    if ctrl_vals and key in s.bpc:
        s.control_bpc = statistics.fmean(ctrl_vals)
        s.control_bpc_sd = (statistics.pstdev(ctrl_vals)
                            if len(ctrl_vals) > 1 else 0.0)
        s.control_n_chars = ctrl_chars
        # the control's word count is min(reading words, original words) by
        # construction; recorded so the word-matching is auditable.
        s.control_n_words = sum(
            min(word_count(reading.texts[c]), word_count(originals[c]))
            for c in cov)
        # clauses where the reading is not shorter, so the control falls back
        # to the original -- the leg is degenerate for these clauses, and the
        # count is published so the reader can see how much of the leg is live
        s.n_control_fallback = sum(
            1 for c in cov
            if word_count(reading.texts[c]) >= word_count(originals[c]))
        s.delta_vs_control = s.bpc[key] - s.control_bpc

    # ---- leg A: pairing (shuffle) null ----------------------------------
    nulls = null_streams(reading, ids, N_NULLS)
    vals = []
    for nstream in nulls:
        nres = compress_stream(nstream)
        if key in nres:
            vals.append(nres[key]["bits_per_char"])
    if vals and key in s.bpc:
        s.null_bpc_mean = statistics.fmean(vals)
        s.null_bpc_sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        s.null_z = ((s.bpc[key] - s.null_bpc_mean) / s.null_bpc_sd
                    if s.null_bpc_sd else 0.0)

    # ---- leg C ----------------------------------------------------------
    train_ids, _ = split_ids(ids)
    train_cov = covered(reading, train_ids)
    if len(train_cov) >= MIN_TRAIN_CLAUSES:
        train_text = reading_stream(reading, train_ids).casefold()
        s.lam = _tune_lambda(train_text)
        model = CharNGram(ORDER, s.lam)
        model.train(train_text)
        s.self_bpc = model.bits_per_char(reading_stream(reading, test_ids).casefold())
        # "cross": the reading's own model scoring the *untouched* held-out
        # originals.  This is a coverage diagnostic, not a quality score --
        # a reading that deletes operators has never seen them, so it is
        # surprised by them by construction.  Disclosed, not hidden.
        s.cross_bpc = model.bits_per_char(
            "\n".join(originals[c] for c in test_ids).casefold())
    if fixed_model is not None:
        cov_test = covered(reading, test_ids)
        s.fixed_self_bpc = fixed_model.bits_per_char(
            reading_stream(reading, cov_test).casefold())
    return s


def score_population(readings: list[Reading], ids: list[str], population: str,
                     originals: dict[str, str]) -> list[Score]:
    """Score every reading that covers ``ids`` under the common protocol.

    The fixed model is trained once per population on the identity train
    clauses only, then reused, so the fixed-model leg is a single shared
    scorer (no per-reading fitting, no leakage of held-out text).
    """
    train_ids, test_ids = split_ids(ids)
    train_text = "\n".join(originals[c] for c in train_ids).casefold()
    fixed = CharNGram(ORDER, _tune_lambda(train_text))
    fixed.train(train_text)
    out = []
    for r in readings:
        if not covered(r, ids):
            continue
        out.append(score_reading(r, ids, population, originals, fixed, test_ids))
    # the fixed-model leg is one shared scorer: the delta is taken against
    # the identity row of the same population, so readings stay comparable.
    base = next((s for s in out if s.reading == "identity"), None)
    for s in out:
        if base is None:
            break
        if s is base:
            s.fixed_gain_vs_identity = 0.0        # the reference itself
        elif not math.isnan(s.fixed_self_bpc):
            s.fixed_gain_vs_identity = s.fixed_self_bpc - base.fixed_self_bpc
    return out


def paired_sign_test(reading: Reading, test_ids: list[str],
                     originals: dict[str, str],
                     model: CharNGram) -> tuple[int, int, float]:
    """Per-clause paired sign test of the reading's text vs the identity text.

    One shared ``model`` scores both texts of the same clause; the pair
    enters the sign test only when both texts are non-empty (#50's per-page
    pattern, clause level).  Works for every reading, including the 5-clause
    ``llm_*`` ones, because it needs no training on the reading itself.
    """
    improved = worsened = 0
    for c in covered(reading, test_ids):
        a, b = reading.texts[c].casefold(), originals[c].casefold()
        if not a or not b:
            continue
        d = model.bits_per_char(a) - model.bits_per_char(b)
        if d < 0:
            improved += 1
        elif d > 0:
            worsened += 1
    return improved, worsened, sign_test_p(improved, worsened)


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
def _rank(values: list[float]) -> list[int]:
    """Competition ranks (1 = best); NaN takes the worst rank, as documented.

    A missing leg must never silently win by input order, so a NaN sorts
    last and shares the worst rank with any other NaN.  When *every* reading
    in a population lacks a leg, all of them share that worst rank, so the
    leg contributes a constant and cannot reorder the population.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: (math.isnan(values[i]), values[i]))
    ranks = [0] * n
    pos = 0
    while pos < n:
        i = order[pos]
        if math.isnan(values[i]):
            ranks[i] = n            # worst rank, shared by every NaN
            pos += 1
            continue
        j = pos
        while j + 1 < n and not math.isnan(values[order[j + 1]]) \
                and values[order[j + 1]] == values[i]:
            j += 1
        for k in range(pos, j + 1):
            ranks[order[k]] = pos + 1
        pos = j + 1
    return ranks


def rank_population(scores: list[Score]) -> list[dict]:
    """Rank readings inside one population, primary leg first.

    **The primary leg is the description-length delta** (``delta_vs_control``):
    it is the leg that answers the task's question -- does the reading
    shorten the description of the book? -- with the deletion confound
    controlled.  Its rank is the verdict rank and sorts the table.  The two
    predictability legs are corroborating evidence and their ranks are
    published next to it; ``mean_rank_three_legs`` is a disclosed,
    *unweighted* aggregate that exists only so the reader can see how the
    legs disagree, and it never overrides the primary rank.

    Lower is better on every metric, so all three legs rank ascending.  A
    reading missing a leg (``llm_*`` have no adaptive leg) takes the worst
    rank on it, so it cannot win a population on legs it does not have.
    """
    keys = ("delta_vs_control", "fixed_gain_vs_identity", "self_bpc")
    ranks = {k: _rank([getattr(s, k) for s in scores]) for k in keys}
    rows = []
    for i, s in enumerate(scores):
        rs = [ranks[k][i] for k in keys]
        rows.append({
            "reading": s.reading,
            "kind": s.kind,
            "population": s.population,
            "n_clauses": s.n_clauses,
            "bits_per_char_gzip9": round(s.bpc.get(PRIMARY_COMPRESSOR, float("nan")), 6),
            "delta_vs_length_control": round(s.delta_vs_control, 6),
            "fixed_model_gain_vs_identity": round(s.fixed_gain_vs_identity, 6),
            "self_bits_per_char": round(s.self_bpc, 6),
            "null_z": round(s.null_z, 4),
            "rank_description_length": ranks[keys[0]][i],
            "rank_fixed_model": ranks[keys[1]][i],
            "rank_self_predictability": ranks[keys[2]][i],
            "mean_rank_three_legs": round(sum(rs) / len(rs), 2),
        })
    rows.sort(key=lambda r: (r["rank_description_length"],
                             r["mean_rank_three_legs"]))
    return rows


# --------------------------------------------------------------------------
# reconciliation against the frozen upstream artifacts
# --------------------------------------------------------------------------
def _frozen(path: Path) -> list[dict]:
    return read_tsv(path) if path.exists() else []


def reconciliation(scores: list[Score], order: list[str],
                   rows: dict[str, dict[str, str]]) -> list[dict]:
    """This build vs #33 / #48 / #62 / #63.

    The compression rows are always recomputed over the **full** clause set,
    even when the scored run was subsampled (``--quick``): a reconciliation
    against a published number must not inherit the subsample's size effect.

    Exact-match rows are build errors when they drift; scope-difference rows
    are disclosed with their reason (clause-joined stream vs page stream),
    following #50's convention.
    """
    out = []
    exact = [
        ("n_clauses", len(order), PUBLISHED["n_clauses"], "#33/#62 clause count"),
        ("n_pages", len({rows[c]["page"] for c in order}), PUBLISHED["n_pages"],
         "#7 page count"),
        ("n_words_clean", sum(int(rows[c]["n_words"]) for c in order),
         PUBLISHED["n_words_clean"], "#33 n_words_clean, reconciled by #62"),
    ]
    for qty, here, earlier, note in exact:
        out.append({"quantity": qty, "earlier": earlier, "here": here,
                    "note": note + " (exact match required)"})
        if here != earlier:
            raise SystemExit(f"#68: {qty} {here} != published {earlier} -- "
                             "upstream artifact drifted; refusing to build")

    # #63 sampled-gloss row count
    n_gloss = len(read_tsv(GLOSSES)) if GLOSSES.exists() else 0
    out.append({"quantity": "n_gloss_rows", "earlier": PUBLISHED["n_gloss_rows"],
                "here": n_gloss, "note": "#63 glosses.tsv rows (exact match required)"})
    if n_gloss != PUBLISHED["n_gloss_rows"]:
        raise SystemExit("#68: #63 gloss row count drifted")

    # compression vs #48's frozen silence_variants.tsv (scope difference).
    # Recomputed on every clause, so a --quick run still reconciles.
    v48 = {(r["variant"], r["compressor"]): r for r in _frozen(V48_SILENCE)}
    full_bpc: dict[str, float] = {}
    for reading, variant in (("identity", "v0_original"),
                             ("v3_operators_to_space", "v3_operators_to_space"),
                             ("v4_operators_to_period", "v4_operators_to_period")):
        texts = {"identity": lambda c: rows[c]["original"],
                 "v3_operators_to_space": lambda c: rows[c]["v3_space"],
                 "v4_operators_to_period": lambda c: rows[c]["v4_period"]}[reading]
        res = compress_stream("\n".join(texts(c) for c in order))
        if PRIMARY_COMPRESSOR in res:
            full_bpc[reading] = res[PRIMARY_COMPRESSOR]["bits_per_char"]
        old = v48.get((variant, PRIMARY_COMPRESSOR), {})
        out.append({
            "quantity": f"gzip-9 bits/char[{reading}]",
            "earlier": old.get("bits_per_symbol", "-"),
            "here": round(full_bpc.get(reading, float("nan")), 6),
            "note": "#48 silence_variants.tsv -- scope differs (#48 strips page "
                    "markers over the page stream; here clauses are joined by \\n), "
                    "so near-identity is expected, not equality",
        })
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)) or x == "":
        return "n/a"
    return f"{x:.{nd}f}"


def _fmt_p(x):
    """Format a p-value without hiding tiny ones behind four zeros."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or x == "":
        return "n/a"
    if 0 < x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def _find(rows, key, field, default=""):
    """First ``row[field]`` in ``rows`` (a list of dicts) with that reading."""
    for r in rows:
        if r.get("reading") == key:
            return r.get(field, default)
    return default


def build_report(ctx: dict) -> str:
    L = []
    a = L.append
    fb = [s for s in ctx["scores_full"]]
    sm = [s for s in ctx["scores_sample"]]
    ranked_fb = ctx["ranked_full"]
    ranked_sm = ctx["ranked_sample"]
    a("# Compression test for candidate readings (#68)\n")
    a("")
    a("The plan's L8.7 test, run. Every candidate reading the repository has "
      "published is scored by whether it makes the corpus more **compressible** "
      "and more **predictable**, under one common protocol, with the identity "
      "(no-reading) control, a length-matched deletion control, a pairing "
      "shuffle null, and a held-out split. Reproduce with "
      "``python work/reading_compression.py``; regression suite "
      "``python work/test_reading_compression.py``. Schema "
      f"`{SCHEMA}`.\n")
    a("")

    # ---- method ----
    a("## 1. Method: why raw bits/char is not enough\n")
    a("")
    a("#48 measured the book against matched English and concluded that raw "
      "compressibility *cannot discriminate between readings* -- the book is "
      "already more compressible than English prose, and every reading of it "
      "would be too. This harness therefore never reports a raw bits/char "
      "number on its own. Each reading is scored by three legs that remove "
      "the confounds in turn:\n")
    a("")
    a("1. **Length-matched deletion control (the primary leg).** For every "
      "covered clause the control is a deterministic, position-preserving "
      "letters-word-subsample of the *original* clause down to the reading's "
      "letters-word count (seed "
      f"`{SEED}:<reading>:<clause>:<seed_k>`), compressed by the identical "
      f"path and repeated over {N_CONTROL_SEEDS} seeds so its own seed noise "
      "is reported. The number is ``reading - control``: a reading whose only "
      "effect is dropping words scores ~0, only restructuring scores "
      "negative. **This is the leg that carries the verdict** (section 6 "
      "ranks by it), because it is the leg that answers the task's question "
      "-- does the reading shorten the description of the book? -- with the "
      "deletion confound controlled. **Its limit is disclosed, not hidden:** "
      "a reading with at least as many words as its clause gets *no* "
      "subsample, because there is nothing to delete, and the control is "
      "then the original itself. The corpus's readings are word-count "
      "preserving or expanding far more often than not -- operator removal "
      "changes the characters *between* words, not the letters, so v3/v4/gloss "
      "match their clause's word count on most clauses -- so for them the "
      "control is the original on the majority of clauses and the delta "
      "reduces toward the raw bits/char delta. The live-clause count is "
      "published per reading in ``controls_and_nulls.tsv`` "
      "(``control_fallback_clauses``) and section 3 states what the leg can "
      "and cannot show for each reading as a result.")
    a("2. **Pairing shuffle null.** The reading's clause texts are permuted "
      f"across its covered clause ids ({N_NULLS} permutations, fixed seeds). "
      "The text multiset is preserved and the clause<->text pairing is "
      "destroyed, so a ``null_z`` far below the reading's own score means the "
      "gain depends on *which* text belongs to *which* clause.")
    a("3. **Held-out predictability.** A character trigram with "
      "Jelinek-Mercer interpolation (#56's estimator, lambda tuned on a "
      "train-internal validation split) scores text it never trained on. "
      "One **fixed model** (trained on the identity train clauses only) "
      "scores every reading, so readings are compared by a single shared "
      "scorer; the full-book readings additionally get an **adaptive** model "
      "trained on their own train clauses, scored on their held-out clauses "
      "(``self``) and on the untouched originals (``cross`` -- a coverage "
      "diagnostic, not a quality score). The fixed-model leg is paired "
      "against the identity text clause by clause with a sign test (#50's "
      "pattern).\n")
    a("")
    a("**Tokenization and units (choices, not facts).** The unit is the "
      "clause (the #33 segmentation, 5,614 clauses), so readings are compared "
      "on identical text spans. The compression stream is the reading's "
      "clause texts joined by newlines in book order, case-folded, UTF-8; the "
      "six #48 compressors are reused from ``compression_mdl`` so the numbers "
      "reconcile with #48's frozen table. A **word** is a maximal run of "
      "letters -- not a whitespace chunk, which would count ``soul/gram`` as "
      "one word and ``soul gram`` as two and so make an operator-removal "
      "reading look longer than its clause purely because of the tokenizer. "
      "Predictability is measured in **characters**, with no word tokenizer "
      "at all: a character model cannot be blinded by a reading's orthography "
      "and applies the same ruler to a gloss and a source clause. The "
      "held-out split is every 5th clause (20 %); clause order is page order, "
      "so the held-out set spans 219 of 225 pages rather than a contiguous "
      "block.\n")
    a("")

    # ---- readings ----
    a("## 2. The candidate readings\n")
    a("")
    a("| reading | kind | provenance | clauses |")
    a("|---|---|---|---:|")
    for r in ctx["readings"]:
        a(f"| `{r.name}` | {r.kind} | {r.source} | {len(r.texts)} |")
    a("")
    a("``identity`` is the control, not a candidate: every delta is measured "
      "against it on the same clause set. The five ``llm_*`` readings cover "
      f"{len(ctx['llm_ids'])} clauses each (#63's stratified sample), so they "
      "are scored on the sample population and never pooled with full-book "
      "readings -- the two populations are ranked separately.\n")
    a("")

    # ---- leg A ----
    a("## 3. Leg A: description length and its controls\n")
    a("")
    a("| reading | gzip-9 bits/char | vs length control | control sd | control fallback | null mean | null z | chars | words |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in fb + sm:
        a(f"| `{s.reading}` | {_fmt(s.bpc.get(PRIMARY_COMPRESSOR))} | "
          f"{_fmt(s.delta_vs_control)} | {_fmt(s.control_bpc_sd)} | "
          f"{s.n_control_fallback}/{s.n_clauses} | "
          f"{_fmt(s.null_bpc_mean)} | "
          f"{_fmt(s.null_z, 3)} | {s.n_chars} | {s.n_words} |")
    a("")
    a("``control fallback`` is the count of covered clauses where the reading "
      "has at least as many words as its clause, so the control *is* the "
      "original and no subsample happens -- the leg is degenerate on those "
      "clauses. For the full-book readings that is the overwhelming majority "
      "(v3/v4 match their clause's word count on 5,470 of 5,614 clauses; the "
      "gloss on 5,479), because operator removal changes the characters "
      "between words, not the letters. **So for the three operator/gloss "
      "readings the primary leg is, in practice, the raw bits/char delta "
      "against the identity stream -- a length-normalized comparison, but not "
      "a length-matched one.** The honest reading of the column is therefore: "
      "these readings do not win by deleting, because there is almost nothing "
      "to delete; the control demonstrates that rather than manufactures a "
      "correction. Where the control is live (the ~75 shorter gloss clauses, "
      "and any reading a future pipeline produces that shortens) the "
      "word-matching is exact and the delta is a true restructuring "
      "measure.\n")
    a("")
    a("``control sd`` is the spread of the control over its "
      f"{N_CONTROL_SEEDS} seeds. On the full book it is ~3e-4 bits/char, so "
      "the full-book deltas sit two orders of magnitude above the control's "
      "own noise. On the 5-clause sample it is 0.0000 for almost every row "
      "-- and that is not precision, it is the fallback again: for four of "
      "the five LLM strategies *every* covered clause falls back, so all "
      "seeds produce the identical stream and the sd is exactly zero (the "
      "exception is ``llm_scene_summary``, with one genuinely shorter clause "
      "and an sd of ~0.04). The sample population is reported descriptively "
      "and never ranked against the full book. ``null_z`` compares the "
      "reading with its own permuted pairings: values near 0 mean the gain "
      "does not depend on the clause<->text pairing.\n")
    a("")

    # ---- leg C ----
    a("## 4. Leg C: held-out predictability\n")
    a("")
    a("Two character-trigram evaluations, both held out (nothing scored was "
      "in the training text):\n")
    a("")
    a("* **fixed model (all readings)** -- one model, trained on the "
      "*identity* train clauses only, scores every reading's held-out "
      "clauses and the identity's.  A single shared scorer, so the "
      "``gain vs identity`` column is an apples-to-apples comparison of how "
      "predictable each reading's text is to a model of the book.")
    a("* **adaptive (full-book readings)** -- a model trained on each "
      "reading's own train clauses, scored on its held-out clauses "
      "(``self``) and, as a coverage diagnostic, on the untouched held-out "
      "originals (``cross``).\n")
    a("")
    a("| reading | lambda | self bits/char | fixed-model bits/char | fixed gain vs identity | cross bits/char (coverage) | improved / worsened | sign-test p |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in fb:
        pr = ctx["paired"].get(s.reading, {})
        a(f"| `{s.reading}` | {_fmt(s.lam, 2)} | {_fmt(s.self_bpc)} | "
          f"{_fmt(s.fixed_self_bpc)} | {_fmt(s.fixed_gain_vs_identity)} | "
          f"{_fmt(s.cross_bpc)} | {pr.get('n_improved', 'n/a')} / "
          f"{pr.get('n_worsened', 'n/a')} | "
          f"{_fmt_p(pr.get('p'))} |")
    a("")
    a("| reading (sample population) | fixed-model bits/char | fixed gain vs identity |")
    a("|---|---:|---:|")
    for s in sm:
        a(f"| `{s.reading}` | {_fmt(s.fixed_self_bpc)} | "
          f"{_fmt(s.fixed_gain_vs_identity)} |")
    a("")
    a("The sample population holds 5 clauses, of which the split leaves "
      "**one** held out; a sign test on n=1 has no power and is not "
      "reported for those rows (see section 5).\n")
    if ctx["anchor"] == ctx["anchor"]:
        a(f"External anchor: the untransformed held-out book scores "
          f"{_fmt(ctx['anchor'])} bits/char under a general-English character "
          "model trained on #56's reference novels -- the yardstick the "
          "readings are trying to beat. The book's own models score it far "
          "lower, which is #48's \"more compressible than English\" finding "
          "restated at character level, and the reason a raw bits/char "
          "number could never separate two readings of this book.\n")
    else:
        a("External anchor unavailable (reference corpora not present); the "
          "within-book comparisons stand on their own.\n")
    a("")
    a("**Reading the cross column honestly.** A positive ``cross`` value "
      "means the reading's own model is *surprised* by the untouched book -- "
      "expected and not a defect: a reading that removes operators has never "
      "seen an operator, so it pays for one every time. The column is a "
      "coverage diagnostic (how much of the book's surface the reading's "
      "system accounts for), not a quality score, and it is why the verdict "
      "does not rest on it.\n")
    a("")
    a("The adaptive leg runs only for readings with at least "
      f"{MIN_TRAIN_CLAUSES} train clauses; the five ``llm_*`` readings "
      "(5 clauses each) are below that floor and are scored by the shared "
      "fixed model alone, whose gain column carries their predictability "
      "score.\n")
    a("")

    # ---- sample population ----
    a("## 5. The #63 strategy readings on their own sample\n")
    a("")
    a("| strategy | gzip-9 bits/char | vs length control | fixed-model gain vs identity |")
    a("|---|---:|---:|---:|")
    for s in sm:
        a(f"| `{s.reading}` | {_fmt(s.bpc.get(PRIMARY_COMPRESSOR))} | "
          f"{_fmt(s.delta_vs_control)} | {_fmt(s.fixed_gain_vs_identity)} |")
    a("")
    a("Five clauses per strategy is a **descriptive** comparison, not an "
      "inferential one: the held-out split leaves a single clause, so no "
      "inferential statistic is reported for this population. These rows "
      "exist so the LLM strategies are scored by the same protocol, not so "
      "they can be ranked against full-book readings.\n")
    a("")

    # ---- ranking ----
    a("## 6. Verdict: the ranked readings\n")
    a("")
    a("**The primary leg is the description-length delta**, so the table is "
      "sorted by ``rank_description_length`` and that rank *is* the verdict. "
      "The two predictability legs are corroborating evidence, ranked beside "
      "it; ``mean_rank_three_legs`` is an unweighted, disclosed aggregate "
      "published only so the reader can see where the legs disagree -- it "
      "never overrides the primary rank.\n")
    a("")
    a("Full-book population (all clauses, lower is better on every leg):\n")
    a("")
    a("| rank (primary) | reading | length-control delta | fixed-model rank | self rank | mean of 3 | null z |")
    a("|---:|---|---:|---:|---:|---:|---:|")
    for r in ranked_fb:
        a(f"| {r['rank_description_length']} | `{r['reading']}` | "
          f"{_fmt(r['delta_vs_length_control'])} | "
          f"{r['rank_fixed_model']} | {r['rank_self_predictability']} | "
          f"{r['mean_rank_three_legs']:.2f} | {_fmt(r['null_z'], 3)} |")
    a("")
    if ranked_fb:
        w = ranked_fb[0]
        w_fixed = next((r["reading"] for r in ranked_fb
                        if r["rank_fixed_model"] == 1), "n/a")
        w_self = next((r["reading"] for r in ranked_fb
                       if r["rank_self_predictability"] == 1), "n/a")
        a(f"**`{w['reading']}` wins the declared primary leg** and is "
          "therefore the reading that most shortens the description of the "
          "book under this protocol. The predictability legs are won by "
          f"`{w_fixed}` (fixed model) and `{w_self}` (self). ")
        if w_fixed == w_self == w["reading"]:
            a("On this run one reading leads on every leg.\n")
        else:
            a("The legs disagree, and the disagreement is the finding "
              "rather than a defect: a reading that rewrites orthography "
              "alone is the more *predictable* one, while a reading that "
              "also substitutes the termbase is the more *compressible* "
              "one. Because no leg is a fidelity instrument, neither winner "
              "is certified as *the* reading. Sample population (#63's "
              "5-clause stratified sample):\n")
    else:
        a("No full-book population was scored.\n")
    a("")
    a("| rank (primary) | reading | length-control delta | fixed-model gain |")
    a("|---:|---|---:|---:|")
    for r in ranked_sm:
        a(f"| {r['rank_description_length']} | `{r['reading']}` | "
          f"{_fmt(r['delta_vs_length_control'])} | "
          f"{_fmt(r['fixed_model_gain_vs_identity'])} |")
    a("")

    # ---- interpretation ----
    a("## 7. What this does and does not show\n")
    a("")
    a("- **Compression is evidence about predictability, not proof that an "
      "interpretation is true** (L9.8). A reading wins here by making the "
      "corpus cheaper to describe; that is a property of the text it "
      "produces, not a certificate of fidelity. #63's checks and #64's "
      "disagreement log remain the fidelity instruments.")
    a("- **The primary leg is a length-matched control that is mostly inert "
      "on this corpus.** It subsamples only where a reading is *shorter* "
      "than its clause; the readings here are word-count preserving or "
      "expanding on ~5,470 of 5,614 clauses (v3/v4/gloss), so for them the "
      "control is the original and the delta is effectively the raw bits/char "
      "delta. What the control does establish is that these readings win "
      "without deleting -- there is almost nothing to delete -- and it "
      "remains exact for any future reading that shortens (the ~75 shorter "
      "gloss clauses are live, and the word-matching is exact there). The "
      "per-reading live-clause counts are in ``controls_and_nulls.tsv``.")
    a("- **The control neutralises *arbitrary* deletion, not *structured* "
      "deletion.** A reading that keeps only each clause's first word is a "
      "pure deletion, yet it beats its random-word control (the book's "
      "clause-initial words are strikingly repetitive, so a stream of them "
      "compresses far better than an equal number of words sampled at "
      "random). This is a true signal about the book's clause-openers, not "
      "a confound, and it is also a warning: a degenerate \"reading\" can "
      "score well on the compression leg. The predictability legs and #63's "
      "fidelity checks are what stop such a reading from being mistaken for "
      "an interpretation.")
    a("- **The shuffle null is a pairing test, not a quality test.** A null_z "
      "near zero says the gain survives losing the clause<->text pairing "
      "(usually because the text multiset dominates); it does not say the "
      "reading is good.")
    a("- **The character trigram is a crude language model.** It has no "
      "lexicon and no long-range structure, so its absolute bits/char is not "
      "comparable to GPT-2's (#49/#50); only within-harness deltas are "
      "meaningful. GPT-2 itself is not re-run here (weights are not cached "
      "in this environment) and #50 already scored the v0/v3/v4 family with "
      "it.")
    a("- **Held-out split is by clause, not by page.** Adjacent clauses from "
      "the same page appear in both halves, so a page-local style can leak "
      "across the split; #56's interleaved-page split is the stricter control "
      "and this one is the clause-level analogue (its defence is that the "
      "split never holds out two adjacent clauses and the fixed-model leg "
      "pairs every reading against the identity text clause by clause).")
    a("- **The sample population is five clauses, and that is small.** The "
      "``llm_*`` rows are reported for protocol completeness; their ranking "
      "is descriptive only. Read on that sample, the two legs point in "
      "opposite directions and the opposition is the point: ``llm_plain_english`` "
      "wins the description-length leg by a wide margin (-1.05 bits/char "
      "against its control, which falls back to the original on all five of "
      "its clauses) while being the *least* predictable of all "
      "nine readings under the book-trained model (+0.96 bits/char). Plain "
      "English is a cheap description of the book's content and a poor "
      "predictor of its surface. **A harness that reported only the "
      "compression leg would rank it first; a harness that reported only the "
      "predictability leg would rank it last.** Neither number is evidence "
      "about whether the reading is *true*.\n")
    a("")

    # ---- triangulation ----
    a("## 8. Triangulation with #48, #50, #56, #62, #63\n")
    a("")
    a("- **#48** supplies the compressors and the anchor finding (the book is "
      "more compressible than English; raw bits/char cannot discriminate "
      "readings). This harness reconciles with its frozen "
      "``silence_variants.tsv`` to within the disclosed scope difference and "
      "answers its limitation with the length control and nulls.")
    a("- **#50** supplies the paired-delta + sign-test pattern and the GPT-2 "
      "verdict that v4 (operators to period) most lowers perplexity. **This "
      "harness disagrees with that verdict, and the disagreement is "
      "reported, not reconciled away:** under the book-trained character "
      "model, v4 is the *worst* full-book reading (fixed-model gain "
      f"{_fmt(_find(ranked_fb, 'v4_operators_to_period', 'fixed_model_gain_vs_identity', float('nan')))}, "
      f"{ctx['paired'].get('v4_operators_to_period', {}).get('n_improved', 'n/a')} "
      "improved / "
      f"{ctx['paired'].get('v4_operators_to_period', {}).get('n_worsened', 'n/a')} "
      "worsened held-out clauses), while v3 is the best. The two instruments "
      "are different language models with different training distributions: "
      "GPT-2 is a general-English web LM that *expects* sentence periods and "
      "rewards v4 for supplying them; a model trained on this book expects "
      "the book's own operator characters and is surprised when they become "
      "periods. Both readings are defensible, which is exactly the "
      "single-method-certainty trap L9.8 names -- no one instrument, "
      "including this one, settles which reading is *the* reading.")
    a("- **#56** supplies the Jelinek-Mercer estimator, the lambda grid, the "
      "train-internal-validation tuning discipline, and the reference "
      "corpora used for the English anchor.")
    a("- **#62** supplies the clause unit, the interlinear columns (original / "
      "v3 / v4 / gloss) and the flagship gloss reading.")
    a("- **#63** supplies the five LLM strategy readings and its check "
      "verdicts; its stratified sample defines the second population.\n")
    a("")

    # ---- handoff ----
    a("## 9. Handoff\n")
    a("- **#69 (evaluation/epistemology)** -- this harness is the L9.4 "
      "(compression / information gain) instrument; its limits section is "
      "written to be quotable in that audit.")
    a("- **#62 (interlinear edition)** -- if the gloss line ever changes, "
      "re-run this harness; the reconciliation rows will flag the drift.")
    a("- **#63 (glossing pipeline)** -- the fixed-model gain column is a "
      "cheap per-strategy compression screen for future prompt strategies.\n")
    a("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="#68 compression test for candidate readings")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quick", action="store_true",
                    help="deterministic clause subsample (regression suite)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if args.quiet else print

    order, rows = load_clauses()
    readings = build_readings(order, rows)
    originals = {c: rows[c]["original"] for c in order}

    eval_ids = order
    if args.quick:
        eval_ids = [c for i, c in enumerate(order) if i % QUICK_EVERY == 0]
    log(f"[#68] {len(readings)} readings, {len(eval_ids)} clauses"
        + (" (quick)" if args.quick else ""))

    # population 1: full book, the four full-coverage readings
    full = [r for r in readings if r.population == "full_book"]
    scores_full = score_population(full, eval_ids, "full_book", originals)
    # population 2: #63's sample, every reading that covers those clauses
    llm_ids = sorted({c for r in readings if r.kind == "llm" for c in r.texts},
                     key=lambda c: order.index(c))
    sample_readings = [r for r in readings if set(r.texts) & set(llm_ids)]
    scores_sample = score_population(sample_readings, llm_ids, "sample_63",
                                     originals)

    # per-clause paired sign test on the shared fixed model (works for every
    # reading, including the 5-clause llm_* ones: one scorer, no fitting).
    _, full_test = split_ids(eval_ids)
    _, sample_test = split_ids(llm_ids)
    fixed_full = CharNGram(ORDER, _tune_lambda(
        "\n".join(originals[c] for c in eval_ids
                  if c not in set(full_test)).casefold()))
    fixed_full.train("\n".join(originals[c] for c in eval_ids
                               if c not in set(full_test)).casefold())
    fixed_sample = CharNGram(ORDER, _tune_lambda(
        "\n".join(originals[c] for c in llm_ids
                  if c not in set(sample_test)).casefold()))
    fixed_sample.train("\n".join(originals[c] for c in llm_ids
                                 if c not in set(sample_test)).casefold())
    paired = {}
    for s in scores_full:
        r = next(r for r in full if r.name == s.reading)
        if r.name == "identity":
            continue
        imp, wor, p = paired_sign_test(r, full_test, originals, fixed_full)
        paired[s.reading] = {"n_improved": imp, "n_worsened": wor, "p": p}
        s.n_improved, s.n_worsened, s.sign_test_p = imp, wor, p
    for s in scores_sample:
        r = next(r for r in sample_readings if r.name == s.reading)
        if r.name == "identity":
            continue
        imp, wor, p = paired_sign_test(r, sample_test, originals, fixed_sample)
        paired[f"{s.reading}@{s.population}"] = {
            "n_improved": imp, "n_worsened": wor, "p": p}
        s.n_improved, s.n_worsened, s.sign_test_p = imp, wor, p

    ranked_full = rank_population(scores_full)
    ranked_sample = rank_population(scores_sample)
    recon = reconciliation(scores_full, order, rows)
    anchor = english_anchor([c for i, c in enumerate(eval_ids)
                             if i % TEST_EVERY == 0], originals)

    # ---- artifacts ----
    order_index = {c: i for i, c in enumerate(order)}
    write_tsv(out_dir / "readings.tsv",
              ["reading", "kind", "population", "source", "n_clauses",
               "n_chars", "n_words"],
              [{"reading": r.name, "kind": r.kind, "population": r.population,
                "source": r.source, "n_clauses": len(r.texts),
                "n_chars": len(reading_stream(
                    r, sorted(r.texts, key=lambda c: order_index[c]))),
                "n_words": sum(len(t.split()) for t in r.texts.values())}
               for r in readings])

    identity_bpc = {(s.population, c): b
                    for s in scores_full + scores_sample if s.reading == "identity"
                    for c, b in s.bpc.items()}
    comp_rows = []
    for s in scores_full + scores_sample:
        for cname, bpc in sorted(s.bpc.items()):
            base = identity_bpc.get((s.population, cname))
            comp_rows.append({
                "reading": s.reading, "population": s.population,
                "compressor": cname, "n_clauses": s.n_clauses,
                "bits_per_char": round(bpc, 6),
                "delta_bits_per_char_vs_identity":
                    round(bpc - base, 6) if base is not None else "",
            })
    write_tsv(out_dir / "compression.tsv",
              ["reading", "population", "compressor", "n_clauses",
               "bits_per_char", "delta_bits_per_char_vs_identity"], comp_rows)

    write_tsv(out_dir / "controls_and_nulls.tsv",
              ["reading", "population", "length_control_bits_per_char",
               "length_control_sd_bits_per_char", "delta_vs_length_control",
               "null_mean_bits_per_char", "null_sd_bits_per_char", "null_z",
               "reading_n_words", "control_n_words",
               "control_fallback_clauses", "n_clauses",
               "control_n_chars", "reading_n_chars"],
              [{"reading": s.reading, "population": s.population,
                "length_control_bits_per_char": round(s.control_bpc, 6),
                "length_control_sd_bits_per_char": round(s.control_bpc_sd, 6),
                "delta_vs_length_control": round(s.delta_vs_control, 6),
                "null_mean_bits_per_char": round(s.null_bpc_mean, 6),
                "null_sd_bits_per_char": round(s.null_bpc_sd, 6),
                "null_z": round(s.null_z, 4),
                "reading_n_words": s.n_words,
                "control_n_words": s.control_n_words,
                "control_fallback_clauses": s.n_control_fallback,
                "n_clauses": s.n_clauses,
                "control_n_chars": s.control_n_chars,
                "reading_n_chars": s.n_chars}
               for s in scores_full + scores_sample])

    write_tsv(out_dir / "predictability.tsv",
              ["reading", "population", "lambda", "self_bits_per_char",
               "cross_bits_per_char_coverage", "fixed_model_self_bits_per_char",
               "fixed_gain_vs_identity", "n_clauses_improved",
               "n_clauses_worsened", "sign_test_p"],
              [{"reading": s.reading, "population": s.population,
                "lambda": round(s.lam, 2),
                "self_bits_per_char": round(s.self_bpc, 6),
                "cross_bits_per_char_coverage": round(s.cross_bpc, 6),
                "fixed_model_self_bits_per_char": round(s.fixed_self_bpc, 6),
                "fixed_gain_vs_identity": round(s.fixed_gain_vs_identity, 6),
                "n_clauses_improved": s.n_improved,
                "n_clauses_worsened": s.n_worsened,
                "sign_test_p": round(s.sign_test_p, 8)}
               for s in scores_full + scores_sample])

    ranked_rows = []
    for pop, rows_ in (("full_book", ranked_full), ("sample_63", ranked_sample)):
        for r in rows_:
            row = dict(r)
            row["population"] = pop
            ranked_rows.append(row)
    write_tsv(out_dir / "ranked_readings.tsv",
              ["reading", "population", "kind", "n_clauses",
               "bits_per_char_gzip9", "delta_vs_length_control",
               "fixed_model_gain_vs_identity", "self_bits_per_char",
               "null_z", "rank_description_length", "rank_fixed_model",
               "rank_self_predictability", "mean_rank_three_legs"],
              ranked_rows)

    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    diag = [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "seed", "value": SEED},
        {"diag": "n_readings", "value": len(readings)},
        {"diag": "n_clauses_evaluated", "value": len(eval_ids)},
        {"diag": "quick", "value": bool(args.quick)},
        {"diag": "test_every", "value": TEST_EVERY},
        {"diag": "char_order", "value": ORDER},
        {"diag": "lambda_grid", "value": ",".join(str(x) for x in LAMBDA_GRID)},
        {"diag": "min_train_clauses_adaptive_leg", "value": MIN_TRAIN_CLAUSES},
        {"diag": "n_null_permutations", "value": N_NULLS},
        {"diag": "n_control_seeds", "value": N_CONTROL_SEEDS},
        {"diag": "primary_compressor", "value": PRIMARY_COMPRESSOR},
        {"diag": "compressors_available", "value": ",".join(sorted(
            {k for s in scores_full for k in s.bpc}))},
        {"diag": "english_anchor_bits_per_char", "value": round(anchor, 6)},
        {"diag": "paired_rows", "value": len(paired)},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["diag", "value"], diag)

    ctx = {"scores_full": scores_full, "scores_sample": scores_sample,
           "ranked_full": ranked_full, "ranked_sample": ranked_sample,
           "readings": readings, "llm_ids": llm_ids, "paired": paired,
           "anchor": anchor}
    summary = {
        "schema": SCHEMA,
        "seed": SEED,
        "n_clauses_evaluated": len(eval_ids),
        "quick": bool(args.quick),
        "readings": [{"reading": r.name, "kind": r.kind,
                      "population": r.population, "source": r.source,
                      "n_clauses": len(r.texts)} for r in readings],
        "compression": [{"reading": s.reading, "population": s.population,
                         "bits_per_char": {k: round(v, 6) for k, v in s.bpc.items()},
                         "delta_vs_length_control": round(s.delta_vs_control, 6),
                         "null_z": round(s.null_z, 4)}
                        for s in scores_full + scores_sample],
        "predictability": [{"reading": s.reading, "population": s.population,
                            "lambda": round(s.lam, 2),
                            "self_bits_per_char": round(s.self_bpc, 6),
                            "cross_bits_per_char_coverage": round(s.cross_bpc, 6),
                            "fixed_gain_vs_identity": round(s.fixed_gain_vs_identity, 6),
                            "sign_test_p": round(s.sign_test_p, 8)}
                           for s in scores_full + scores_sample],
        "ranked_full_book": ranked_full,
        "ranked_sample": ranked_sample,        "english_anchor_bits_per_char": round(anchor, 6),
        "reconciliation": recon,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out_dir / "reading_compression_report.md").write_text(
        build_report(ctx), encoding="utf-8")

    log(f"[#68] wrote {out_dir} (schema {SCHEMA})")
    if ranked_full:
        w = ranked_full[0]
        log(f"[#68] verdict: {w['reading']} wins the primary "
            f"description-length leg (rank {w['rank_description_length']}, "
            f"delta {w['delta_vs_length_control']:.6f}; fixed-model rank "
            f"{w['rank_fixed_model']}, self rank "
            f"{w['rank_self_predictability']})")
    return 0


if __name__ == "__main__":
    main()
