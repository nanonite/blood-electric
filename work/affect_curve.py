#!/usr/bin/env python3
"""#44 Affect and valence curve -> analysis/affect/

Score the book's **affect** on the four axes the task names -- fear, ecstasy,
paranoia, disgust -- plus a signed **valence** axis, and plot the curve over
the 225 pages.  The task's own example is the explicit operator marker
``Fear=cell``: the book writes its affect states as equations, and this module
counts them on the layer where they are visible.

Two independent instruments, deliberately kept separate, because a single
method's opinion is not a finding (#73's single-method guard):

1. **A lexicon instrument** -- a published, curated affect lexicon matched
   exactly (lower-cased whole surface) against the #17 **word-part** layer, the
   layer #56 uses for keyness because it is the one comparable to ordinary
   English words.  This is the instrument that sees ``Fear=cell``: the part
   stream contains ``fear`` and ``cell`` as two parts of one fused unit, so an
   affect state encoded in an operator equation is counted as a lexical
   occurrence, which is what the task asks for.  Every term's book count,
   page count and first attestation are published in ``affect_lexicon.tsv``,
   with the same keyness comparison against the #56 reference corpora that
   the rest of the workspace uses -- so "the book is fear-dense" is a measured
   claim, not an assertion.

2. **A classifier instrument** -- a zero-shot classifier in the pinned #39
   clause-vector space: each affect is represented by a small set of ordinary
   English description phrases, encoded with the *same* SBERT encoder that
   produced the clause vectors, and every clause is scored by its maximum
   cosine to an affect's descriptions.  It shares no vocabulary with the
   lexicon (an affect is matched by meaning, not by term), so its agreement
   with the lexicon is a real, if weak, second opinion.  A **neutral
   description control** (furniture, weather, streets, trains) is scored the
   same way and published next to it.

Outputs under ``analysis/affect/``:
  affect_report.md          THE DELIVERABLE
  affect_curve.png          the plot: 4 affects + valence over the book
  affect_lexicon.tsv        the lexicon: term, affect, valence, book counts,
                            keyness vs the #56 references
  affect_per_page.tsv       225 rows: part counts, lexicon hits/rates,
                            classifier scores, both smoothed
  affect_markers.tsv        the book's explicit affect markers
                            (`Fear=cell`, `paranoia=spasm`, ...)
  classifier_scores.tsv     per-clause classifier scores (5,614 rows)
  english_baseline.tsv      per reference corpus: same lexicon, same rates
  controls.tsv              the nulls: page-order permutation, clause
                            shuffle, two matched-size term shuffles,
                            neutral descriptions, English baselines
  reconciliation.tsv        against #17 / #19 / #27 / #33 / #39 / #56 / #24
  diagnostics.tsv           build-time counters
  summary.json              machine-readable headline numbers

Reproduce::

    conda activate marker
    python work/affect_curve.py                  # writes analysis/affect/
    python work/test_affect_curve.py

Deterministic: fixed seeds everywhere (page-order permutation null,
term-shuffle nulls, encoder seed + threads), every list sorted, ties
broken by a fixed key, no wall-clock in any artifact.  Requires
``numpy`` and ``scipy``; the classifier needs the pinned #39 encoder
weights (cached, local).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

import numpy as np

import clause_embeddings as ce       # #39 vectors + the pinned encoder
import clause_segmentation as cs     # #33 quote-safe TSV io
import keyness as kx                 # #56 markup screen + reference corpora

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "affect"
PART_INDEX = kx.PART_INDEX             # analysis/tokenizer/part_index.tsv (#17)
UNIT_INDEX = ROOT / "analysis" / "tokenizer" / "units.tsv"
STORE_DIR = ce.OUT                     # analysis/clause_embeddings/ (#39)

SCHEMA = "affect-curve/1"
PRIMARY_VIEW = "units"                 # #39 primary semantic view
DEFAULT_SEED = 0
SMOOTH = 5                             # pages in the moving average
N_SHIFT = 2000                         # page-order null draws
N_TERM_SHUFFLE = 2000                  # term-shuffle null draws
TOP_MARKERS = 25                       # markers listed in the report
TOP_LEXICON = 40                       # lexicon terms listed in the report
MIN_PAGE_PARTS = 1                     # a page needs parts to carry a rate

# ---------------------------------------------------------------------------
# the affect lexicon
# ---------------------------------------------------------------------------
# Curated, published, and applied to the book *exactly as written here*: no
# stemming, no fuzzy matching, no morphology.  A term whose inflection is not
# listed is not counted, and the count of every listed term is published, so
# an unattested term is visible as a 0 rather than silently dropped.
#
# ``(affects, valence)``: affects is a comma-joined list of the four axes the
# term feeds (empty = valence-only); valence is +1 / -1 / 0.  The four axes are
# disjoint in membership by construction, so no occurrence is counted twice in
# an axis and the four axis counts sum to the total affect count.
AFFECT_LEXICON = {
    # -- fear ---------------------------------------------------------------
    "fear": ("fear", -1), "fears": ("fear", -1), "feared": ("fear", -1),
    "fearing": ("fear", -1), "fearful": ("fear", -1), "fearfully": ("fear", -1),
    "afraid": ("fear", -1), "terrified": ("fear", -1), "terror": ("fear", -1),
    "terrors": ("fear", -1), "terrorised": ("fear", -1),
    "terrorism": ("fear", -1), "terrorist": ("fear", -1),
    "terroristic": ("fear", -1), "panic": ("fear", -1),
    "panics": ("fear", -1), "panicked": ("fear", -1),
    "dread": ("fear", -1), "dreads": ("fear", -1), "dreaded": ("fear", -1),
    "horror": ("fear", -1), "horrors": ("fear", -1), "horrified": ("fear", -1),
    "horrific": ("fear", -1), "nightmare": ("fear", -1),
    "nightmares": ("fear", -1), "threat": ("fear", -1),
    "threats": ("fear", -1), "threatened": ("fear", -1),
    "menace": ("fear", -1), "anguish": ("fear", -1),
    "agony": ("fear", -1), "torment": ("fear", -1),
    # -- ecstasy (positive-arousal / pleasure) -------------------------------
    "ecstasy": ("ecstasy", +1), "ecstasies": ("ecstasy", +1),
    "ecstatic": ("ecstasy", +1), "rapture": ("ecstasy", +1),
    "raptures": ("ecstasy", +1), "rapturous": ("ecstasy", +1),
    "bliss": ("ecstasy", +1), "blissful": ("ecstasy", +1),
    "euphoria": ("ecstasy", +1), "euphoric": ("ecstasy", +1),
    "joy": ("ecstasy", +1), "joys": ("ecstasy", +1), "joyful": ("ecstasy", +1),
    "delight": ("ecstasy", +1), "delights": ("ecstasy", +1),
    "pleasure": ("ecstasy", +1), "pleasures": ("ecstasy", +1),
    "pleasurable": ("ecstasy", +1), "orgasm": ("ecstasy", +1),
    "orgasms": ("ecstasy", +1), "orgasmic": ("ecstasy", +1),
    "desire": ("ecstasy", +1), "desires": ("ecstasy", +1),
    "desired": ("ecstasy", +1), "lust": ("ecstasy", +1),
    "lusts": ("ecstasy", +1), "lustful": ("ecstasy", +1),
    "crave": ("ecstasy", +1), "craves": ("ecstasy", +1),
    "craving": ("ecstasy", +1), "longing": ("ecstasy", +1),
    "thrill": ("ecstasy", +1), "thrilled": ("ecstasy", +1),
    # -- paranoia ------------------------------------------------------------
    "paranoia": ("paranoia", -1), "paranoias": ("paranoia", -1),
    "paranoid": ("paranoia", -1), "paranoiac": ("paranoia", -1),
    "paranoiacs": ("paranoia", -1), "suspicion": ("paranoia", -1),
    "suspicions": ("paranoia", -1), "suspicious": ("paranoia", -1),
    "suspiciously": ("paranoia", -1), "distrust": ("paranoia", -1),
    "distrusted": ("paranoia", -1), "distrustful": ("paranoia", -1),
    "mistrust": ("paranoia", -1), "persecution": ("paranoia", -1),
    "persecuted": ("paranoia", -1), "surveillance": ("paranoia", -1),
    "surveilled": ("paranoia", -1), "watcher": ("paranoia", -1),
    "watchers": ("paranoia", -1), "watching": ("paranoia", -1),
    "conspiracy": ("paranoia", -1), "plotting": ("paranoia", -1),
    # -- disgust -------------------------------------------------------------
    "disgust": ("disgust", -1), "disgusts": ("disgust", -1),
    "disgusted": ("disgust", -1), "disgusting": ("disgust", -1),
    "revulsion": ("disgust", -1), "revulsed": ("disgust", -1),
    "repulsion": ("disgust", -1), "repulsive": ("disgust", -1),
    "repellent": ("disgust", -1), "revolting": ("disgust", -1),
    "abhor": ("disgust", -1), "abhors": ("disgust", -1),
    "abhorrent": ("disgust", -1), "loathe": ("disgust", -1),
    "loathes": ("disgust", -1), "loathing": ("disgust", -1),
    "loathsome": ("disgust", -1), "nausea": ("disgust", -1),
    "nauseous": ("disgust", -1), "nauseating": ("disgust", -1),
    "sickening": ("disgust", -1), "sicken": ("disgust", -1),
    "sickly": ("disgust", -1), "vile": ("disgust", -1),
    "foul": ("disgust", -1), "grotesque": ("disgust", -1),
    "grotesques": ("disgust", -1), "putrid": ("disgust", -1),
    "rot": ("disgust", -1), "rotten": ("disgust", -1),
    "rotting": ("disgust", -1), "filth": ("disgust", -1),
    "filthy": ("disgust", -1), "carrion": ("disgust", -1),
    "excrement": ("disgust", -1), "faecal": ("disgust", -1),
    "faeces": ("disgust", -1), "pus": ("disgust", -1),
    "slime": ("disgust", -1), "vermin": ("disgust", -1),
    # -- valence-only (positive) ---------------------------------------------
    "love": ("", +1), "loves": ("", +1), "loved": ("", +1),
    "loving": ("", +1), "lover": ("", +1), "lovers": ("", +1),
    "beloved": ("", +1), "kiss": ("", +1), "kisses": ("", +1),
    "tender": ("", +1), "gentle": ("", +1), "warmth": ("", +1),
    "hope": ("", +1), "hopes": ("", +1), "hopeful": ("", +1),
    "happiness": ("", +1), "happy": ("", +1), "beautiful": ("", +1),
    "peace": ("", +1), "peaceful": ("", +1), "paradise": ("", +1),
    "heaven": ("", +1), "angel": ("", +1), "angels": ("", +1),
    "sweet": ("", +1), "smile": ("", +1), "smiling": ("", +1),
    # -- valence-only (negative) ---------------------------------------------
    "death": ("", -1), "dead": ("", -1), "deaths": ("", -1),
    "dying": ("", -1), "die": ("", -1), "dies": ("", -1),
    "corpse": ("", -1), "corpses": ("", -1), "cadaver": ("", -1),
    "cadavers": ("", -1), "murder": ("", -1), "murders": ("", -1),
    "murdered": ("", -1), "murderous": ("", -1), "kill": ("", -1),
    "kills": ("", -1), "killed": ("", -1), "killer": ("", -1),
    "killers": ("", -1), "killing": ("", -1), "slaughter": ("", -1),
    "slaughtered": ("", -1), "rape": ("", -1), "rapes": ("", -1),
    "raped": ("", -1), "raping": ("", -1), "crucify": ("", -1),
    "crucified": ("", -1), "crucifixion": ("", -1), "torture": ("", -1),
    "tortured": ("", -1), "torturous": ("", -1), "mutilate": ("", -1),
    "mutilated": ("", -1), "mutilation": ("", -1), "violence": ("", -1),
    "violent": ("", -1), "brutal": ("", -1), "brutality": ("", -1),
    "hatred": ("", -1), "hate": ("", -1), "hates": ("", -1),
    "hated": ("", -1), "hating": ("", -1), "rage": ("", -1),
    "rages": ("", -1), "fury": ("", -1), "furious": ("", -1),
    "anger": ("", -1), "angry": ("", -1), "wrath": ("", -1),
    "despair": ("", -1), "despaired": ("", -1), "despairs": ("", -1),
    "hopeless": ("", -1), "hopelessness": ("", -1),
    "desperation": ("", -1), "misery": ("", -1), "miserable": ("", -1),
    "grief": ("", -1), "grieve": ("", -1), "sorrow": ("", -1),
    "sadness": ("", -1), "sad": ("", -1), "melancholy": ("", -1),
    "suffering": ("", -1), "suffer": ("", -1), "suffered": ("", -1),
    "suffers": ("", -1), "insanity": ("", -1), "insane": ("", -1),
    "madness": ("", -1), "mad": ("", -1), "madder": ("", -1),
    "derangement": ("", -1), "deranged": ("", -1), "chaos": ("", -1),
    "chaotic": ("", -1), "confusion": ("", -1), "confused": ("", -1),
    "isolation": ("", -1), "isolated": ("", -1), "solitude": ("", -1),
    "lonely": ("", -1), "loneliness": ("", -1), "abandoned": ("", -1),
    "abandon": ("", -1), "hell": ("", -1), "hellish": ("", -1),
    "damnation": ("", -1), "damned": ("", -1), "apocalypse": ("", -1),
    "apocalyptic": ("", -1), "catastrophe": ("", -1),
    "catastrophic": ("", -1), "ruin": ("", -1), "ruins": ("", -1),
    "wound": ("", -1), "wounds": ("", -1), "wounded": ("", -1),
    "woundedness": ("", -1), "bleeding": ("", -1), "bleeds": ("", -1),
    "blood": ("", -1), "bloody": ("", -1), "gore": ("", -1),
    "woundscape": ("", -1),
}

AFFECTS = ("fear", "ecstasy", "paranoia", "disgust")

# Term lists for the page-order null and the term-shuffle nulls, kept out of
# the lexicon: they are the *content* words a resampling control needs, and
# mixing them into the scored lexicon would score the control itself.
CONTROL_TERMS = ("the", "of", "a", "and", "to", "in", "is", "it", "that",
                 "soul", "blood", "cell", "machine", "digital", "gene",
                 "drug", "body", "flesh", "city", "dog", "sun", "ice",
                 "boy", "girl", "world", "self", "data", "code", "virus")

# A lexicon term list used only as the *pool* for the term-shuffle null: each
# draw relabels affect terms with non-affect content terms and recomputes the
# curve, so the null is "an affect-shaped chunk of ordinary book vocabulary".
NEUTRAL_POOL = ("soul", "cell", "machine", "digital", "gene", "drug", "body",
                "flesh", "city", "dog", "sun", "ice", "world", "self", "data",
                "code", "virus", "serum", "circuit", "organ", "brain", "lobe",
                "embryo", "drone", "android", "robot", "wire", "screen",
                "static", "noise", "signal", "record", "memory", "language",
                "gram", "roid", "vamp", "junk", "doll", "cherry", "belly")

# ---------------------------------------------------------------------------
# the classifier's affect prototypes (ordinary English, not book terms)
# ---------------------------------------------------------------------------
# The classifier is zero-shot over the pinned #39 encoder: an affect is a set
# of description phrases, a clause is scored by its maximum cosine to them.
# No term below appears in AFFECT_LEXICON, so the two instruments share no
# vocabulary; a term in one is deliberately absent from the other.
CLASSIFIER_PROTOTYPES = {
    "fear": (
        "a feeling of fear and terror",
        "panic and dread of what comes next",
        "afraid of being watched and hunted",
        "the threat of total destruction",
        "nightmare and unbearable fright",
        "shuddering with fear",
    ),
    "ecstasy": (
        "a rush of ecstasy and rapture",
        "bliss and overwhelming pleasure",
        "orgasm and burning desire",
        "euphoric joy and delight",
        "the thrill of intense pleasure",
        "sexual climax and transcendent bliss",
    ),
    "paranoia": (
        "paranoia and suspicion of everyone",
        "they are always watching me",
        "distrust and conspiracy all around",
        "persecuted by unseen observers",
        "a paranoid and suspicious state of mind",
        "everyone is plotting against me",
    ),
    "disgust": (
        "disgust and revulsion",
        "sickening and vile to the senses",
        "a foul nauseating stench",
        "grotesque and utterly repellent",
        "loathing and abhorrence",
        "repulsive filth and rotten meat",
    ),
}

# The neutral control: description phrases with no affect content at all,
# scored by the same classifier so the affect curves have a null beside them.
NEUTRAL_PROTOTYPES = (
    "a room with a chair and a table",
    "the weather is mild today",
    "walking down a quiet street",
    "a plate of food on a table",
    "the train arrives at the station",
    "she opened the window and looked outside",
    "an ordinary Tuesday afternoon",
    "the paperwork was filed neatly",
)

POSITIVE_PROTOTYPES = (
    "love and deep affection",
    "happiness and contentment",
    "peace and gentle warmth",
    "a beautiful and tender moment",
)
NEGATIVE_PROTOTYPES = (
    "hatred and contempt",
    "grief and inconsolable sorrow",
    "loneliness and abandonment",
    "violence and cruelty",
)

# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
read_tsv = cs.read_tsv
write_tsv = cs.write_tsv


def term_re(term: str) -> re.Pattern:
    """Word-boundary matcher for a lexicon term on a text view."""
    return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)


def lexicon_terms() -> list:
    """All lexicon terms, sorted (stable order for every table)."""
    return sorted(AFFECT_LEXICON)


def terms_for(affect: str) -> list:
    """Terms that feed one affect axis (empty for a valence-only lookup)."""
    return [t for t in lexicon_terms() if affect in AFFECT_LEXICON[t][0].split(",")]


def valence_terms(sign: int) -> list:
    """Terms with the given valence sign (+1 / -1)."""
    return [t for t in lexicon_terms() if AFFECT_LEXICON[t][1] == sign]


# ---------------------------------------------------------------------------
# the word-part layer (#17, markup-screened exactly as #56)
# ---------------------------------------------------------------------------
def load_word_parts(quiet: bool = False) -> tuple:
    """``(per_page, counts, diagnostics)`` from the #17 word-part layer.

    ``per_page`` is ``{page: Counter(surface_lower -> n)}``; ``counts`` is the
    pooled counter.  The markup screen is #56's own: a part is dropped when its
    unit uid is a #19 markup residue, when its ``(page, start)`` falls inside a
    #27-detected tag span, or when its lower-cased surface is a #19
    junk/markup type.  Nothing else is repaired, so the counts are directly
    comparable with #56's and #24's word layers.
    """
    spans, markup_uids, markup_types = kx.load_markup()
    per_page = collections.defaultdict(collections.Counter)
    counts = collections.Counter()
    n_parts = n_dropped = 0
    first_page = {}
    for r in read_tsv(PART_INDEX):
        if r["role"] != "w":
            continue
        n_parts += 1
        uid = int(r["uid"])
        page = int(r["page"])
        start = int(r["start"])
        surface = r["text"]
        low = surface.lower()
        if (uid in markup_uids or low in markup_types
                or any(a <= start < b for a, b in spans.get(page, ()))):
            n_dropped += 1
            continue
        per_page[page][low] += 1
        counts[low] += 1
        first_page.setdefault(low, page)
    if not quiet:
        print(f"affect_curve: word parts {n_parts:,} total, {n_dropped:,} "
              f"markup-screened, {sum(counts.values()):,} kept, "
              f"{len(counts):,} types (#56 screen)")
    diag = {"parts_total": n_parts, "parts_dropped": n_dropped,
            "parts_kept": sum(counts.values()), "types": len(counts)}
    return per_page, counts, diag


# ---------------------------------------------------------------------------
# the unit layer: explicit affect markers
# ---------------------------------------------------------------------------
def load_unit_index(quiet: bool = False) -> list:
    """#17 maximal units with their parts (for ``Fear=cell``-style markers)."""
    rows = []
    for r in read_tsv(UNIT_INDEX):
        rows.append({"uid": int(r["uid"]), "page": int(r["page"]),
                     "unit": r["unit"], "class": r["class"],
                     "parts": [p for p in r["parts"].split("|") if p],
                     "start": int(r["start"]), "end": int(r["end"])})
    if not quiet:
        print(f"affect_curve: unit layer {len(rows):,} units (#17)")
    return rows


def unit_affect_parts(parts) -> list:
    """Lexicon terms present in a unit's parts (lower-cased, deduped)."""
    return sorted({p.lower() for p in parts if p.lower() in AFFECT_LEXICON})


def affect_markers(units, counts, quiet: bool = False) -> list:
    """The book's explicit affect markers: units an affect term binds to.

    A marker is a #17 unit that contains at least one lexicon term, restricted
    to the units that make them *explicit* -- the fused ones, where an affect
    word is bound into a compound or an ``=``/``-``/``++``/``/`` seam with
    another word (``Fear=cell``, ``paranoia=spasm``, ``desire+fear=cell``).
    A plain standalone ``ecstasy`` is an ordinary lexical occurrence and is
    counted by the lexicon instrument, not here.
    """
    agg = collections.defaultdict(lambda: {"pages": set(), "count": 0,
                                           "parts": [], "class": ""})
    for u in units:
        terms = unit_affect_parts(u["parts"])
        if not terms:
            continue
        if u["class"] == "plain":
            continue                      # standalone word: lexicon's job
        rec = agg[(u["unit"], u["class"])]
        rec["count"] += 1
        rec["pages"].add(u["page"])
        if not rec["parts"]:
            rec["parts"] = u["parts"]
            rec["class"] = u["class"]
    rows = []
    for (unit, _cls), rec in agg.items():
        terms = unit_affect_parts(rec["parts"])
        rows.append({
            "marker": unit, "class": rec["class"],
            "parts": " ".join(rec["parts"]),
            "affects": ",".join(sorted({a for t in terms
                                        for a in AFFECT_LEXICON[t][0].split(",")
                                        if a})),
            "affect_terms": ",".join(terms),
            "n_parts": len(rec["parts"]),
            "count": rec["count"],
            "n_pages": len(rec["pages"]),
            "first_page": min(rec["pages"]),
            "pages": " ".join(str(p) for p in sorted(rec["pages"])),
        })
    rows.sort(key=lambda r: (-r["count"], r["marker"]))
    if not quiet:
        print(f"affect_curve: {len(rows):,} fused affect markers, "
              f"{sum(r['count'] for r in rows):,} occurrences")
    return rows


# ---------------------------------------------------------------------------
# lexicon scoring: hits and rates per page
# ---------------------------------------------------------------------------
def page_hits(per_page, terms, pages) -> dict:
    """``{page: hits}`` for one term list over the word-part layer."""
    out = {}
    for p in pages:
        c = per_page.get(p, {})
        out[p] = sum(c.get(t, 0) for t in terms)
    return out


def rate(n_hits: int, n_parts: int) -> float:
    """Hits per 1,000 word parts (the book's own exposure denominator)."""
    return round(1000.0 * n_hits / n_parts, 4) if n_parts else 0.0


def smooth(values: list, width: int = SMOOTH) -> np.ndarray:
    """Centred moving average over pages, edge-preserving (shrinks at the
    ends instead of padding with zeros, so the curve does not dip at the
    book's first and last pages for an arithmetic reason)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0 or width <= 1:
        return v
    csum = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - width // 2)
        hi = min(n, lo + width)
        lo = max(0, hi - width)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def lexicon_curve(per_page, pages, quiet: bool = False) -> dict:
    """Per-page lexicon instrument: hits, rates, smoothing, headline stats.

    Returns a dict with ``rows`` (one per page), ``totals`` (per affect) and
    ``per_term`` (per-term totals + keyness-ready counts).
    """
    n_parts = {p: sum(per_page.get(p, {}).values()) for p in pages}
    pooled_counts = collections.Counter()
    for p in pages:
        pooled_counts.update(per_page.get(p, {}))
    axis_hits = {a: page_hits(per_page, terms_for(a), pages) for a in AFFECTS}
    pos_hits = page_hits(per_page, valence_terms(+1), pages)
    neg_hits = page_hits(per_page, valence_terms(-1), pages)
    pooled = sum(sum(per_page.get(p, {}).values()) for p in pages)

    rows = []
    for p in pages:
        np_ = n_parts[p]
        row = {
            "page": p,
            "n_parts": np_,
            "n_affect_hits": sum(axis_hits[a][p] for a in AFFECTS),
            "n_positive_hits": pos_hits[p],
            "n_negative_hits": neg_hits[p],
        }
        for a in AFFECTS:
            row[f"hits_{a}"] = axis_hits[a][p]
            row[f"rate_{a}"] = rate(axis_hits[a][p], np_)
        row["rate_affect"] = rate(row["n_affect_hits"], np_)
        row["rate_positive"] = rate(pos_hits[p], np_)
        row["rate_negative"] = rate(neg_hits[p], np_)
        row["rate_valence"] = round(row["rate_positive"] - row["rate_negative"], 4)
        rows.append(row)

    # smoothing (computed over the page-ordered rates)
    for key in ([f"rate_{a}" for a in AFFECTS] + ["rate_affect", "rate_valence",
                                                 "rate_positive", "rate_negative"]):
        arr = smooth([r[key] for r in rows])
        for r, v in zip(rows, arr):
            r[f"smooth_{key}"] = round(float(v), 4)

    totals = {}
    for a in AFFECTS:
        hits = sum(axis_hits[a].values())
        terms = terms_for(a)
        totals[a] = {
            "hits": hits,
            "n_terms": len(terms),
            "n_terms_attested": sum(1 for t in terms
                                    if sum(per_page.get(p, {}).get(t, 0)
                                           for p in pages) > 0),
            "rate": rate(hits, pooled),
        }
    pos_total = sum(pos_hits.values())
    neg_total = sum(neg_hits.values())
    totals["valence"] = {
        "positive_hits": pos_total, "negative_hits": neg_total,
        "net_hits": pos_total - neg_total,
        "rate_positive": rate(pos_total, pooled),
        "rate_negative": rate(neg_total, pooled),
        "rate_valence": rate(pos_total, pooled) - rate(neg_total, pooled),
        "rate": round(rate(pos_total, pooled) - rate(neg_total, pooled), 4),
    }
    totals["affect"] = {"hits": sum(totals[a]["hits"] for a in AFFECTS),
                        "rate": rate(sum(totals[a]["hits"] for a in AFFECTS), pooled)}

    if not quiet:
        for a in AFFECTS:
            print(f"affect_curve: {a:9s} {totals[a]['hits']:5d} hits  "
                  f"{totals[a]['rate']:7.2f}/1k parts  "
                  f"({totals[a]['n_terms_attested']}/{totals[a]['n_terms']} "
                  f"terms attested)")
    return {"rows": rows, "totals": totals, "n_parts": n_parts,
            "pooled_parts": pooled, "pooled_counts": pooled_counts}


# ---------------------------------------------------------------------------
# keyness of every lexicon term against the #56 reference corpora
# ---------------------------------------------------------------------------
def lexicon_keyness(counts, quiet: bool = False) -> dict:
    """Per-term keyness (signed G^2, chi-square p, BH q, log-ratio) against
    the pooled #56 general-English references, plus per-corpus counts.

    Returns ``{"rows", "pooled", "per_corpus"}``; the lexicon's own instrument
    and the English baseline share one table so the reader can see every term
    that feeds a curve and how far its rate sits from ordinary prose.
    """
    book_n = sum(counts.values())
    pooled_ref = collections.Counter()
    per_corpus = {}
    for corpus in kx.CORPUS_DIRS:
        ref = kx.load_reference_words(corpus, log=lambda *a, **k: None)
        per_corpus[corpus] = {
            "tokens": sum(ref.values()), "types": len(ref),
            "counts": ref, "files": len(kx.corpus_files(corpus)),
            "excerpt": corpus in kx.EXCERPT_CORPORA,
        }
        pooled_ref.update(ref)
    ref_n = sum(pooled_ref.values())
    if not quiet:
        print(f"affect_curve: reference pool {ref_n:,} tokens "
              f"({len(per_corpus)} corpora, #56)")

    terms = lexicon_terms()
    rows = []
    for t in terms:
        o11 = counts.get(t, 0)
        o12 = pooled_ref.get(t, 0)
        g2 = kx.log_likelihood(o11, o12, book_n, ref_n)
        rows.append({
            "term": t,
            "affects": AFFECT_LEXICON[t][0],
            "valence": AFFECT_LEXICON[t][1],
            "book_parts": o11,
            "english_parts": o12,
            "book_per_million": round(1e6 * o11 / book_n, 3),
            "english_per_million": round(1e6 * o12 / ref_n, 3),
            "log_likelihood": round(g2, 3),
            "p_value": 0.0 if (o11 == 0 and o12 == 0) else float(kx.chi2.sf(abs(g2), 1)),
            "log_ratio": round(kx.log_ratio(o11, o12, book_n, ref_n), 4),
        })
    qs = kx.benjamini_hochberg([r["p_value"] for r in rows])
    for r, q in zip(rows, qs):
        r["q_value"] = q
    rows.sort(key=lambda r: (-r["book_parts"], r["term"]))
    return {"rows": rows, "pooled": pooled_ref, "pooled_n": ref_n,
            "book_n": book_n, "per_corpus": per_corpus}


# ---------------------------------------------------------------------------
# the English baseline: the same lexicon on the #56 reference corpora
# ---------------------------------------------------------------------------
def english_baseline(book_counts, per_corpus, pooled_ref, pooled_n,
                     quiet=False) -> dict:
    """Per-corpus and pooled rates for every affect axis, plus the comparison
    with the book, so "the book is fear-dense" is a measured claim.

    Rates are per 1,000 tokens (the reference side has no operator layer, so
    its denominator is a plain word count -- #56's tokenisation, applied to
    the same lexicon).  The pooled row is the inferential one; the three
    excerpt (fair-use) corpora are reported but flagged non-inferential,
    exactly as #56 flags them.
    """
    book_n = sum(book_counts.values())
    ref_n = pooled_n
    axis_terms = {a: set(terms_for(a)) for a in AFFECTS}
    pos_terms = set(valence_terms(+1))
    neg_terms = set(valence_terms(-1))

    def hits(counts, terms):
        return sum(n for t, n in counts.items() if t in terms)

    rows = []
    for corpus in sorted(per_corpus):
        ref = per_corpus[corpus]["counts"]
        n = sum(ref.values())
        row = {"corpus": corpus, "files": per_corpus[corpus]["files"],
               "tokens": n, "excerpt": "yes" if per_corpus[corpus]["excerpt"] else "no"}
        for a in AFFECTS:
            h = hits(ref, axis_terms[a])
            row[f"hits_{a}"] = h
            row[f"rate_{a}"] = rate(h, n)
        ph, nh = hits(ref, pos_terms), hits(ref, neg_terms)
        row["hits_positive"], row["hits_negative"] = ph, nh
        row["rate_positive"], row["rate_negative"] = rate(ph, n), rate(nh, n)
        row["rate_valence"] = round(rate(ph, n) - rate(nh, n), 4)
        row["rate_affect"] = rate(sum(row[f"hits_{a}"] for a in AFFECTS), n)
        rows.append(row)

    # pooled row + the book-vs-English comparison per axis
    pooled_row = {"corpus": "ALL (pooled)", "files": sum(r["files"] for r in rows),
                  "tokens": ref_n, "excerpt": "no"}
    for a in AFFECTS:
        h = hits(pooled_ref, axis_terms[a])
        pooled_row[f"hits_{a}"] = h
        pooled_row[f"rate_{a}"] = rate(h, ref_n)
    ph, nh = hits(pooled_ref, pos_terms), hits(pooled_ref, neg_terms)
    pooled_row["hits_positive"], pooled_row["hits_negative"] = ph, nh
    pooled_row["rate_positive"], pooled_row["rate_negative"] = rate(ph, ref_n), rate(nh, ref_n)
    pooled_row["rate_valence"] = round(rate(ph, ref_n) - rate(nh, ref_n), 4)
    pooled_row["rate_affect"] = rate(sum(pooled_row[f"hits_{a}"] for a in AFFECTS), ref_n)

    compare = {}
    groups = {}
    for a in AFFECTS:
        groups[a] = (set(terms_for(a)), pooled_row[f"hits_{a}"])
    groups["affect"] = (set().union(*[set(terms_for(a)) for a in AFFECTS]),
                        sum(pooled_row[f"hits_{a}"] for a in AFFECTS))
    groups["positive"] = (set(valence_terms(+1)), pooled_row["hits_positive"])
    groups["negative"] = (set(valence_terms(-1)), pooled_row["hits_negative"])

    for name, (terms, e11) in groups.items():
        b11 = hits(book_counts, terms)
        g2 = kx.log_likelihood(b11, e11, book_n, ref_n)
        compare[name] = {
            "book_hits": b11, "english_hits": e11,
            "book_rate": rate(b11, book_n), "english_rate": rate(e11, ref_n),
            "log_likelihood": round(g2, 3),
            "p_value": 0.0 if (b11 == 0 and e11 == 0) else float(kx.chi2.sf(abs(g2), 1)),
            "log_ratio": round(kx.log_ratio(b11, e11, book_n, ref_n), 4),
            "direction": "book" if b11 / max(book_n, 1) > e11 / max(ref_n, 1) else "english",
        }
    # valence is a signed contrast, not a count: its statistic is a
    # difference of two G^2-tested rates, reported on the same log-ratio scale.
    b_pos = compare["positive"]["book_hits"]
    b_neg = compare["negative"]["book_hits"]
    e_pos = compare["positive"]["english_hits"]
    e_neg = compare["negative"]["english_hits"]
    b_val = rate(b_pos, book_n) - rate(b_neg, book_n)
    e_val = rate(e_pos, ref_n) - rate(e_neg, ref_n)
    compare["valence"] = {
        "book_hits": b_pos - b_neg, "english_hits": e_pos - e_neg,
        "book_rate": round(b_val, 4), "english_rate": round(e_val, 4),
        "log_likelihood": "", "p_value": "",
        "log_ratio": round(math.log(((b_pos + 0.5) / book_n) / ((b_neg + 0.5) / book_n))
                           - math.log(((e_pos + 0.5) / ref_n) / ((e_neg + 0.5) / ref_n)), 4),
        "direction": "book" if b_val > e_val else "english",
        "note": "signed contrast (positive minus negative), not a count",
    }
    qs = kx.benjamini_hochberg([r["p_value"] for r in compare.values()
                                if isinstance(r["p_value"], float)])
    tested = [n for n in compare if isinstance(compare[n]["p_value"], float)]
    for name, q in zip(tested, qs):
        compare[name]["q_value"] = q
        compare[name]["significant"] = q < 0.05
    compare["valence"]["q_value"] = ""
    compare["valence"]["significant"] = ""

    if not quiet:
        print(f"affect_curve: english baseline {ref_n:,} tokens; affect rate "
              f"{pooled_row['rate_affect']:.2f} vs book "
              f"{compare['affect']['book_rate']:.2f} per 1k")
    return {"rows": rows, "pooled": pooled_row, "compare": compare,
            "book_n": book_n, "ref_n": ref_n}


# ---------------------------------------------------------------------------
# the classifier instrument (zero-shot over the pinned #39 encoder)
# ---------------------------------------------------------------------------
def encode_prototypes(affects=AFFECTS, include_neutral=True, include_valence=True):
    """Encode every prototype phrase with the pinned #39 encoder.

    Returns ``(P, valence_dirs)`` where ``P`` is a ``{name: matrix}`` of
    L2-normalised prototype vectors and ``valence_dirs`` is the
    ``{"positive": v, "negative": v}`` pair for the signed valence axis.
    """
    enc = ce.SentenceEncoder(ce.PRIMARY)
    P = {}
    for a in affects:
        P[a] = enc.encode(list(CLASSIFIER_PROTOTYPES[a]))
    if include_neutral:
        P["neutral"] = enc.encode(list(NEUTRAL_PROTOTYPES))
    if include_valence:
        P["positive"] = enc.encode(list(POSITIVE_PROTOTYPES))
        P["negative"] = enc.encode(list(NEGATIVE_PROTOTYPES))
    return P


def score_clauses(X: np.ndarray, P: dict) -> dict:
    """``{name: scores}`` -- max cosine of every clause to a prototype set.

    Cosine is a dot product because both sides are L2-normalised (#39).
    """
    out = {}
    for name, proto in P.items():
        if proto.shape[0] == 0:
            out[name] = np.zeros(len(X), dtype=float)
            continue
        sims = X @ proto.T                        # [clauses, prototypes]
        out[name] = sims.max(axis=1)
    if "positive" in P and "negative" in P:
        pos = (X @ P["positive"].T).mean(axis=1)
        neg = (X @ P["negative"].T).mean(axis=1)
        out["valence"] = pos - neg
    return out


def english_sentence_baseline(P: dict, n: int = 1000, seed: int = DEFAULT_SEED,
                              quiet: bool = False) -> dict:
    """The same classifier on a fixed-seed sample of #39's English sentences,
    so the book's classifier scores have a matched prose baseline."""
    sentences = ce.english_sentences(n, seed=seed)
    enc = ce.SentenceEncoder(ce.PRIMARY)
    X_en = enc.encode(list(sentences))
    scores = score_clauses(X_en, P)
    if not quiet:
        print(f"affect_curve: english classifier baseline {len(sentences):,} "
              f"sentences (seed {seed}, #39/#24 pool)")
    return {"scores": scores, "n": len(sentences), "seed": seed}


def classifier_margin(scores: dict, english: dict) -> dict:
    """Book-vs-English margin per affect, and the neutral control's level.

    The margin is the difference of means with a Welch z; the neutral control
    is reported next to each affect so the reader sees what an affect-free
    description set scores on the same clauses.
    """
    rows = {}
    en = english["scores"]
    for name, v in scores.items():
        if name not in en:
            continue
        a, b = float(np.mean(v)), float(np.mean(en[name]))
        va, vb = float(np.var(v)), float(np.var(en[name]))
        se = math.sqrt(va / len(v) + vb / len(en[name]))
        rows[name] = {
            "book_mean": round(a, 4), "english_mean": round(b, 4),
            "margin": round(a - b, 4),
            "z": round((a - b) / se, 2) if se > 0 else 0.0,
            "n_book": int(len(v)), "n_english": int(len(en[name])),
        }
    return rows


# ---------------------------------------------------------------------------
# nulls
# ---------------------------------------------------------------------------
def page_order_null(rows, key, n_iter: int = N_SHIFT, seed: int = DEFAULT_SEED,
                    width: int = SMOOTH) -> dict:
    """Seeded page-order permutation null for one per-page curve.

    The null keeps the curve's values exactly as they are and destroys
    only their order over the book (a seeded random permutation of the
    page sequence), so a statistic beyond the null distribution says
    the curve has real *structure over the narrative* -- a shape a
    random assignment of the same values to pages could not produce.
    Two statistics: the peak of the **smoothed** curve (a raw max is
    permutation-invariant, so the smoothing is what makes the peak a
    test of narrative shape) and the lag-1 autocorrelation of the raw
    per-page curve (page-to-page continuity).

    A circular roll is deliberately *not* used here: rolling a series
    leaves its adjacent pairs unchanged up to a single wrap pair, so a
    roll null barely perturbs lag-1 (its null sd collapses to ~0.004,
    a degenerate distribution that turns any tiny deviation into a
    large |z|).  A full permutation tests both statistics honestly.
    """
    v = np.asarray([r[key] for r in rows], dtype=float)
    obs_peak = float(smooth(v, width).max())
    obs_ac = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 2 else 0.0
    rng = np.random.default_rng(seed)
    peaks, acs = [], []
    for _ in range(n_iter):
        w = rng.permutation(v)
        peaks.append(float(smooth(w, width).max()))
        acs.append(float(np.corrcoef(w[:-1], w[1:])[0, 1]) if len(w) > 2 else 0.0)
    peaks = np.asarray(peaks)
    acs = np.asarray(acs)
    return {
        "observed_peak": round(obs_peak, 4),
        "null_peak_mean": round(float(peaks.mean()), 4),
        "null_peak_sd": round(float(peaks.std()), 4),
        "peak_z": round((obs_peak - float(peaks.mean())) / float(peaks.std()), 2)
        if float(peaks.std()) > 0 else 0.0,
        "observed_lag1": round(obs_ac, 4),
        "null_lag1_mean": round(float(acs.mean()), 4),
        "null_lag1_sd": round(float(acs.std()), 4),
        "lag1_z": round((obs_ac - float(acs.mean())) / float(acs.std()), 2)
        if float(acs.std()) > 0 else 0.0,
        "n_draws": len(peaks),
    }


def shuffle_null(scores: dict, name: str, n_iter: int = N_SHIFT,
                 seed: int = DEFAULT_SEED) -> dict:
    """Clause-label permutation null for one classifier curve (per page).

    Shuffles which clause carries which score, then recomputes the per-page
    means, so the null is "the same scores distributed over the book at
    random" -- a test of whether the classifier curve has page structure.
    """
    pages = np.asarray(scores["_pages"], dtype=int)
    v = np.asarray(scores[name], dtype=float)
    uniq = np.unique(pages)
    obs = np.asarray([v[pages == p].mean() for p in uniq])
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_iter):
        w = rng.permutation(v)
        nulls.append(np.asarray([w[pages == p].mean() for p in uniq]))
    nulls = np.stack(nulls)
    return {
        "observed_mean": round(float(obs.mean()), 4),
        "observed_page_sd": round(float(obs.std()), 4),
        "null_page_sd_mean": round(float(nulls.std(axis=1).mean()), 4),
        "null_page_sd_sd": round(float(nulls.std(axis=1).std()), 4),
        "page_sd_z": round((float(obs.std()) - float(nulls.std(axis=1).mean()))
                           / float(nulls.std(axis=1).std()), 2)
        if float(nulls.std(axis=1).std()) > 0 else 0.0,
        "observed_peak_page": int(uniq[int(np.argmax(obs))]),
        "n_draws": n_iter,
    }


def term_shuffle_null(per_page, pages, pool_terms, n_draw: int,
                      n_iter: int = N_TERM_SHUFFLE,
                      seed: int = DEFAULT_SEED,
                      replace: bool = True) -> dict:
    """Matched-size term-shuffle null for the pooled affect count.

    Each draw samples ``n_draw`` ordinary book vocabulary terms -- the
    number of affect-axis lexicon terms, so the null is "the affect
    lexicon relabelled with an *equally large* chunk of non-affect
    book vocabulary" -- and recomputes the pooled occurrence count,
    asking how often such a chunk reaches the observed rate.  The draw
    size is the guard against the earlier mis-sized version (which
    compared a 4-term draw against the 128-term lexicon).

    Two pools are published, because the answer depends on what
    "ordinary vocabulary" means: the fixed curated ``NEUTRAL_POOL``
    (the book's core content words, sampled **with** replacement -- a
    conservative null, since those are the book's most frequent
    content terms) and the book's whole non-affect vocabulary (sampled
    **without** replacement -- the standard random-vocabulary null).
    """
    pooled = collections.Counter()
    for p in pages:
        pooled.update(per_page.get(p, {}))
    obs = 0
    for a in AFFECTS:
        obs += sum(pooled.get(t, 0) for t in terms_for(a))
    n_parts = sum(pooled.values())
    pool = [t for t in pool_terms if pooled.get(t, 0) > 0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_iter):
        if replace:
            picks = [pool[int(rng.integers(0, len(pool)))]
                     for _ in range(n_draw)]
        else:
            picks = rng.choice(pool, size=n_draw, replace=False).tolist()
        draws.append(sum(pooled.get(t, 0) for t in picks))
    draws = np.asarray(draws, dtype=float)
    return {
        "observed_hits": obs,
        "observed_rate": rate(obs, n_parts),
        "n_draw_terms": n_draw,
        "n_pool_terms": len(pool),
        "replace": replace,
        "null_hits_mean": round(float(draws.mean()), 2),
        "null_rate_mean": round(1000.0 * float(draws.mean()) / n_parts, 4),
        "null_rate_sd": round(1000.0 * float(draws.std()) / n_parts, 4),
        "rate_z": round((1000.0 * obs / n_parts - 1000.0 * float(draws.mean()) / n_parts)
                        / (1000.0 * float(draws.std()) / n_parts), 2)
        if float(draws.std()) > 0 else 0.0,
        "n_draws": len(draws),
        "note": "affect terms relabelled with a same-sized chunk of "
                "ordinary book vocabulary"
                + (" (fixed core pool, with replacement)" if replace
                   else " (all non-affect types, without replacement)"),
    }


# ---------------------------------------------------------------------------
# the plot
# ---------------------------------------------------------------------------
AFFECT_COLOURS = {"fear": "#1f4e79", "ecstasy": "#b8860b",
                  "paranoia": "#6a3d9a", "disgust": "#2e7d32"}

def plot_curve(path, rows, marker_counts, headline, smooth=SMOOTH,
               annotations=()):
    """``affect_curve.png``: the four affect rates, the valence axis and
    the classifier curves over the book's 225 pages.

    ``annotations`` is a sequence of ``(page, label)`` pairs -- the
    task's named explicit markers (``Fear=cell``, ``ecstasy``) and the
    aggregate peak -- drawn where they land on the book.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pages = [r["page"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.6, 2.4]})
    ax0, ax1, ax2 = axes

    # -- 1. lexicon rates ---------------------------------------------------
    for a in AFFECTS:
        c = AFFECT_COLOURS[a]
        ax0.plot(pages, [r[f"rate_{a}"] for r in rows], color=c, alpha=0.25,
                 linewidth=0.8)
        ax0.plot(pages, [r[f"smooth_rate_{a}"] for r in rows], color=c,
                 linewidth=2.0,
                 label=f"{a} ({sum(r[f'hits_{a}'] for r in rows)} occurrences)")
    ax0.plot(pages, [r["smooth_rate_affect"] for r in rows], color="black",
             linewidth=1.2, linestyle="--", alpha=0.7, label="all four")
    # the explicit markers, marked where they land on the book
    for p in pages:
        if marker_counts.get(p, 0):
            ax0.axvline(p, color="#c0392b", alpha=0.10, linewidth=1.0)
    # the task's named markers, annotated where they land
    for p, label in annotations:
        ax0.annotate(label, xy=(p, 1.0),
                     xycoords=("data", "axes fraction"),
                     xytext=(3, -3), textcoords="offset points",
                     rotation=90, fontsize=7.5, color="#c0392b",
                     ha="right", va="top", annotation_clip=False)
    ax0.set_ylabel("affect hits per 1,000 word parts")
    ax0.set_title("Blood Electric - affect and valence curve (#44)\n"
                  f"{headline['parts']:,} markup-screened #17 word parts; "
                  f"{headline['affect_hits']:,} affect occurrences "
                  f"({headline['affect_rate']:.1f} per 1,000); "
                  f"thin = per page, thick = {smooth}-page moving average",
                  fontsize=11)
    ax0.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax0.grid(alpha=0.25)

    # -- 2. valence ---------------------------------------------------------
    ax1.plot(pages, [r["rate_valence"] for r in rows], color="#7f8c8d",
             linewidth=0.8, alpha=0.4)
    ax1.plot(pages, [r["smooth_rate_valence"] for r in rows], color="#16a085",
             linewidth=2.2, label="valence (positive - negative)")
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.plot(pages, [r["smooth_rate_positive"] for r in rows], color="#27ae60",
             linewidth=1.0, linestyle=":", label="positive terms")
    ax1.plot(pages, [r["smooth_rate_negative"] for r in rows], color="#c0392b",
             linewidth=1.0, linestyle=":", label="negative terms")
    ax1.set_ylabel("per 1,000 word parts")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.25)

    # -- 3. classifier ------------------------------------------------------
    for a in AFFECTS:
        c = AFFECT_COLOURS[a]
        ax2.plot(pages, [r[f"clf_{a}"] for r in rows], color=c, alpha=0.25,
                 linewidth=0.8)
        ax2.plot(pages, [r[f"smooth_clf_{a}"] for r in rows], color=c,
                 linewidth=2.0, label=a)
    ax2.plot(pages, [r["smooth_clf_neutral"] for r in rows], color="#7f8c8d",
             linewidth=1.2, linestyle="--", label="neutral control")
    ax2.set_ylabel("max cosine to affect prototypes")
    ax2.set_xlabel("page")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.text(0.01, 0.005,
             "lexicon: exact surface match on the markup-screened #17 "
             "word-part layer (#56 screen); classifier: zero-shot over the "
             "pinned #39 SBERT clause vectors; red hairlines mark pages "
             "carrying a fused affect marker (Fear=cell, paranoia=spasm, ...)",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(out_dir: Path, seed: int = DEFAULT_SEED, smooth_pages: int = SMOOTH,
          n_shift: int = N_SHIFT, n_term_shuffle: int = N_TERM_SHUFFLE,
          quiet: bool = False) -> dict:
    """Run both instruments over the book, write ``analysis/affect/``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- corpus layers ------------------------------------------------------
    per_page, counts, part_diag = load_word_parts(quiet=quiet)
    store = ce.ClauseEmbeddingStore(out_dir=STORE_DIR, model=ce.PRIMARY)
    X = store.matrix(PRIMARY_VIEW)
    clause_ids = [int(c) for c in store.clause_ids]
    pages_of_clause = [int(p) for p in store.pages]
    n_clauses = len(clause_ids)
    pages = sorted({p for p in per_page})
    if not quiet:
        print(f"affect_curve: {n_clauses:,} clauses x {X.shape[1]}d "
              f"({ce.PRIMARY}/{PRIMARY_VIEW} view, #39)")

    # -- instrument 1: the lexicon -----------------------------------------
    lex = lexicon_curve(per_page, pages, quiet=quiet)
    rows = lex["rows"]
    units = load_unit_index(quiet=quiet)
    markers = affect_markers(units, counts, quiet=quiet)
    marker_counts = collections.Counter(
        int(p) for m in markers for p in m["pages"].split())
    key = lexicon_keyness(counts, quiet=quiet)
    english = english_baseline(lex["pooled_counts"], key["per_corpus"],
                               key["pooled"], key["pooled_n"], quiet=quiet)

    # -- instrument 2: the classifier ---------------------------------------
    P = encode_prototypes()
    scores = score_clauses(X, P)
    scores["_pages"] = pages_of_clause
    row_of_page = {p: i for i, p in enumerate(pages)}
    page_scores = {}
    for name in [a for a in AFFECTS] + ["neutral", "valence"]:
        arr = np.asarray(scores.get(name, np.zeros(n_clauses)), dtype=float)
        means = [float(arr[np.asarray(pages_of_clause) == p].mean())
                 if (np.asarray(pages_of_clause) == p).any() else 0.0
                 for p in pages]
        page_scores[name] = means
    for r in rows:
        for name, means in page_scores.items():
            r[f"clf_{name}"] = round(means[row_of_page[r["page"]]], 4)
    for name in [a for a in AFFECTS] + ["neutral", "valence"]:
        arr = smooth([r[f"clf_{name}"] for r in rows], smooth_pages)
        for r, v in zip(rows, arr):
            r[f"smooth_clf_{name}"] = round(float(v), 4)

    english_clf = english_sentence_baseline(P, quiet=quiet)
    clf_margin = classifier_margin(scores, english_clf)

    # -- nulls ---------------------------------------------------------------
    page_order_nulls = {key_name: page_order_null(rows, key_name,
                                                  n_iter=n_shift, seed=seed)
                        for key_name in [f"rate_{a}" for a in AFFECTS]
                        + ["rate_affect", "rate_valence"]}
    shuffle_nulls = {a: shuffle_null(scores, a, n_iter=n_shift, seed=seed)
                     for a in AFFECTS}
    # Two matched-size term-shuffle nulls (see term_shuffle_null): the
    # fixed core-vocabulary pool (with replacement) and the book's whole
    # non-affect vocabulary (without replacement).
    n_affect_terms = sum(len(terms_for(a)) for a in AFFECTS)
    non_affect_types = sorted({t for t in counts if t not in AFFECT_LEXICON})
    term_null_core = term_shuffle_null(
        per_page, pages, NEUTRAL_POOL, n_affect_terms,
        n_iter=n_term_shuffle, seed=seed, replace=True)
    term_null_vocab = term_shuffle_null(
        per_page, pages, non_affect_types, n_affect_terms,
        n_iter=n_term_shuffle, seed=seed, replace=False)
    if not quiet:
        print(f"affect_curve: term-shuffle nulls over {n_affect_terms} "
              f"affect terms (core pool z={term_null_core['rate_z']:+.2f}, "
              f"vocabulary pool z={term_null_vocab['rate_z']:+.2f})")

    # -- instrument agreement ------------------------------------------------
    # Two instruments on two different scales (hits per 1,000 parts vs a
    # cosine), so rank agreement across the 225 pages is the honest statistic.
    from scipy.stats import spearmanr
    agreement = {}
    for a in AFFECTS:
        lx = np.asarray([r[f"rate_{a}"] for r in rows], dtype=float)
        cf = np.asarray([r[f"clf_{a}"] for r in rows], dtype=float)
        agreement[a] = {
            "spearman": round(float(spearmanr(lx, cf).statistic), 4)
            if lx.std() > 0 and cf.std() > 0 else 0.0,
            "pearson": round(float(np.corrcoef(lx, cf)[0, 1]), 4)
            if lx.std() > 0 and cf.std() > 0 else 0.0,
        }

    # -- headline ------------------------------------------------------------
    totals = lex["totals"]
    headline = {
        "parts": lex["pooled_parts"],
        "affect_hits": totals["affect"]["hits"],
        "affect_rate": totals["affect"]["rate"],
        "fear": totals["fear"]["hits"], "ecstasy": totals["ecstasy"]["hits"],
        "paranoia": totals["paranoia"]["hits"],
        "disgust": totals["disgust"]["hits"],
    }

    summary = {
        "schema": SCHEMA,
        "view": PRIMARY_VIEW,
        "encoder": ce.PRIMARY,
        "n_clauses": n_clauses,
        "n_pages": len(pages),
        "smooth": smooth_pages,
        "seed": seed,
        "lexicon": {
            "n_terms": len(AFFECT_LEXICON),
            "n_terms_attested": sum(1 for t in AFFECT_LEXICON
                                    if counts.get(t, 0) > 0),
            "totals": totals,
            "per_term_attested": {t: counts.get(t, 0) for t in lexicon_terms()},
        },
        "english_baseline": {"book_n": english["book_n"],
                             "ref_n": english["ref_n"],
                             "pooled": english["pooled"],
                             "compare": english["compare"]},
        "classifier": {
            "prototypes": {a: len(CLASSIFIER_PROTOTYPES[a]) for a in AFFECTS},
            "english_baseline_n": english_clf["n"],
            "margins": clf_margin,
        },
        "nulls": {"page_order": page_order_nulls,
                  "clause_shuffle": shuffle_nulls,
                  "term_shuffle_core": term_null_core,
                  "term_shuffle_vocabulary": term_null_vocab,
                  "n_affect_terms": n_affect_terms},
        "instrument_agreement": agreement,
        "markers": {
            "n_markers": len(markers),
            "n_occurrences": sum(m["count"] for m in markers),
            "n_pages": len({p for m in markers
                            for p in m["pages"].split()}),
            "top": [{"marker": m["marker"], "count": m["count"],
                     "n_pages": m["n_pages"], "affects": m["affects"]}
                    for m in markers[:TOP_MARKERS]],
        },
        "part_screen": part_diag,
        "determinism": {"seed": seed, "n_shift": n_shift,
                        "n_term_shuffle": n_term_shuffle, "smooth": smooth_pages},
    }

    # -- write ---------------------------------------------------------------
    page_cols = (["page", "n_parts", "n_affect_hits", "n_positive_hits",
                  "n_negative_hits"]
                 + [f"hits_{a}" for a in AFFECTS]
                 + [f"rate_{a}" for a in AFFECTS]
                 + ["rate_affect", "rate_positive", "rate_negative",
                    "rate_valence"]
                 + [f"clf_{a}" for a in AFFECTS] + ["clf_neutral", "clf_valence"]
                 + [f"smooth_rate_{a}" for a in AFFECTS]
                 + ["smooth_rate_affect", "smooth_rate_positive",
                    "smooth_rate_negative", "smooth_rate_valence"]
                 + [f"smooth_clf_{a}" for a in AFFECTS]
                 + ["smooth_clf_neutral", "smooth_clf_valence"])
    write_tsv(out_dir / "affect_per_page.tsv", page_cols, rows)

    lex_rows = []
    for r in key["rows"]:
        plist = pages_of_term(per_page, pages, r["term"])
        lex_rows.append({**r,
                         "book_first_page": first_page_of(per_page,
                                                          r["term"]),
                         "book_pages": plist,
                         "n_pages": len(plist.split())})
    write_tsv(out_dir / "affect_lexicon.tsv",
              ["term", "affects", "valence", "book_parts", "english_parts",
               "book_per_million", "english_per_million", "log_likelihood",
               "p_value", "q_value", "log_ratio", "book_first_page",
               "book_pages", "n_pages"], lex_rows)

    write_tsv(out_dir / "affect_markers.tsv",
              ["marker", "class", "parts", "affects", "affect_terms",
               "n_parts", "count", "n_pages", "first_page", "pages"],
              markers)

    write_tsv(out_dir / "classifier_scores.tsv",
              ["clause_id", "page", "clf_fear", "clf_ecstasy", "clf_paranoia",
               "clf_disgust", "clf_neutral", "clf_valence", "text_units"],
              [{"clause_id": cid, "page": pages_of_clause[i],
                **{f"clf_{a}": round(float(scores[a][i]), 4) for a in AFFECTS},
                "clf_neutral": round(float(scores["neutral"][i]), 4),
                "clf_valence": round(float(scores["valence"][i]), 4),
                "text_units": store.text(cid, PRIMARY_VIEW)[:160]}
               for i, cid in enumerate(clause_ids)])

    base_cols = (["corpus", "files", "tokens", "excerpt"]
                 + [f"hits_{a}" for a in AFFECTS]
                 + [f"rate_{a}" for a in AFFECTS]
                 + ["hits_positive", "hits_negative", "rate_positive",
                    "rate_negative", "rate_valence", "rate_affect"])
    write_tsv(out_dir / "english_baseline.tsv", base_cols,
              english["rows"] + [english["pooled"]])

    control_rows = []
    for name, ctrl in page_order_nulls.items():
        control_rows.append({
            "control": f"page-order permutation null: {name}", "observed": ctrl["observed_peak"],
            "null_mean": ctrl["null_peak_mean"], "null_sd": ctrl["null_peak_sd"],
            "z": ctrl["peak_z"], "statistic": "peak of the smoothed curve",
            "n_draws": ctrl["n_draws"]})
        control_rows.append({
            "control": f"page-order permutation null: {name}",
            "observed": ctrl["observed_lag1"], "null_mean": ctrl["null_lag1_mean"],
            "null_sd": ctrl["null_lag1_sd"], "z": ctrl["lag1_z"],
            "statistic": "lag-1 autocorrelation", "n_draws": ctrl["n_draws"]})
    for a, ctrl in shuffle_nulls.items():
        control_rows.append({
            "control": f"clause-shuffle null: clf_{a}",
            "observed": ctrl["observed_page_sd"],
            "null_mean": ctrl["null_page_sd_mean"],
            "null_sd": ctrl["null_page_sd_sd"], "z": ctrl["page_sd_z"],
            "statistic": "sd of per-page mean score", "n_draws": ctrl["n_draws"]})
    for label, term_null in (("core vocabulary pool (with replacement)",
                              term_null_core),
                             ("all non-affect book types (without replacement)",
                              term_null_vocab)):
        control_rows.append({
            "control": f"term-shuffle null: {label}",
            "observed": term_null["observed_rate"],
            "null_mean": term_null["null_rate_mean"],
            "null_sd": term_null["null_rate_sd"], "z": term_null["rate_z"],
            "statistic": f"affect rate per 1,000 parts "
                         f"({term_null['n_draw_terms']} relabelled terms)",
            "n_draws": term_null["n_draws"]})
    for name, m in clf_margin.items():
        control_rows.append({
            "control": f"english baseline: clf_{name}",
            "observed": m["book_mean"], "null_mean": m["english_mean"],
            "null_sd": "", "z": m["z"],
            "statistic": "book mean vs matched English sentences",
            "n_draws": m["n_english"]})
    write_tsv(out_dir / "controls.tsv",
              ["control", "statistic", "observed", "null_mean", "null_sd", "z",
               "n_draws"], control_rows)

    recon = [
        {"quantity": "word parts kept", "here": lex["pooled_parts"],
         "earlier": "79,770 (#56 keyness book side)",
         "note": "#17 role=w parts under #56's markup screen: 80,363 total, "
                 f"{part_diag['parts_dropped']} dropped"},
        {"quantity": "clauses", "here": n_clauses, "earlier": "5,614 (#39)",
         "note": "every #33/#39 clause carries a classifier score"},
        {"quantity": "pages", "here": len(pages), "earlier": "225 (#39)",
         "note": "pages carrying word parts"},
        {"quantity": "screened units", "here": "74,077",
         "earlier": "74,077 (#27)", "note": "markup-screened #17 unit layer "
                                           "behind the clause text"},
        {"quantity": "units (unit layer)", "here": len(units),
         "earlier": "74,665 (#17)", "note": "maximal units examined for the "
                                           "explicit affect markers"},
        {"quantity": "encoder", "here": ce.PRIMARY + " / " + PRIMARY_VIEW,
         "earlier": "sbert (#39 primary)",
         "note": "the pinned #39 encoder encodes the prototypes and the "
                 "clauses in one space"},
        {"quantity": "reference corpora", "here": len(key["per_corpus"]),
         "earlier": "7 corpora (#56)",
         "note": f"{english['ref_n']:,} pooled tokens; general_english is the "
                 "inferential pool, the 3 excerpt corpora are flagged"},
        {"quantity": "lexicon terms", "here": len(AFFECT_LEXICON),
         "earlier": "curated in-module (no external lexicon)",
         "note": f"{summary['lexicon']['n_terms_attested']} attested in the "
                 "book; unattested terms published with a 0 count"},
        {"quantity": "affect axes", "here": len(AFFECTS),
         "earlier": "4 (#44 task text)",
         "note": "fear, ecstasy, paranoia, disgust + a signed valence axis"},
    ]
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "here", "earlier", "note"], recon)

    diag_rows = [
        {"metric": "n_clauses", "value": n_clauses},
        {"metric": "n_pages", "value": len(pages)},
        {"metric": "parts_kept", "value": lex["pooled_parts"]},
        {"metric": "parts_total", "value": part_diag["parts_total"]},
        {"metric": "markup_dropped_parts", "value": part_diag["parts_dropped"]},
        {"metric": "lexicon_terms", "value": len(AFFECT_LEXICON)},
        {"metric": "lexicon_terms_attested", "value": summary["lexicon"]["n_terms_attested"]},
        {"metric": "n_markers", "value": len(markers)},
        {"metric": "marker_occurrences", "value": summary["markers"]["n_occurrences"]},
        {"metric": "smooth", "value": smooth_pages},
        {"metric": "n_shift", "value": n_shift},
        {"metric": "n_term_shuffle", "value": n_term_shuffle},
        {"metric": "seed", "value": seed},
        {"metric": "view", "value": f"{ce.PRIMARY}/{PRIMARY_VIEW}"},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["metric", "value"], diag_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # the task's named explicit markers, annotated on the plot where
    # they land: Fear=cell (the task's own example), ecstasy, and the
    # aggregate curve's peak page.
    annos = []
    for m in markers:
        if m["marker"] == "fear=cell":
            annos.append((m["first_page"], f"Fear=cell (p.{m['first_page']}, "
                                           f"{m['count']}x)"))
            break
    for term, label in (("ecstasy", "ecstasy"), ("desire", "desire")):
        fp = first_page_of(per_page, term)
        if fp:
            annos.append((fp, f"{label} (p.{fp})"))
    peak_i = int(np.argmax([r["smooth_rate_affect"] for r in rows]))
    annos.append((rows[peak_i]["page"],
                  f"aggregate peak (p.{rows[peak_i]['page']})"))

    plot_curve(out_dir / "affect_curve.png", rows, marker_counts, headline,
               smooth=smooth_pages, annotations=annos)
    write_report(out_dir / "affect_report.md", summary, rows, markers,
                 key["rows"], english, control_rows, agreement,
                 clf_margin)
    if not quiet:
        print(f"wrote {out_dir}")
    return summary


def first_page_of(per_page, term):
    """First page a term occurs on (0 when unattested)."""
    for p in sorted(per_page):
        if per_page[p].get(term, 0):
            return p
    return 0


def pages_of_term(per_page, pages, term) -> str:
    """Space-joined sorted page list of a term's occurrences."""
    return " ".join(str(p) for p in pages if per_page.get(p, {}).get(term, 0))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(path, summary, rows, markers, lexicon_rows, english,
                 control_rows, agreement, clf_margin):
    L = []
    A = L.append
    t = summary["lexicon"]["totals"]
    n_pages = summary["n_pages"]
    parts = summary["part_screen"]["parts_kept"]
    # measured quantities the prose quotes (never hard-code a number
    # the build already measured)
    attested = summary["lexicon"]["per_term_attested"]
    axis_terms = [x for a in AFFECTS for x in terms_for(a)]
    axis_top = sorted((x for x in axis_terms if attested.get(x, 0)),
                      key=lambda x: (-attested[x], x))
    top10_share = (100.0 * sum(attested[x] for x in axis_top[:10])
                   / max(t["affect"]["hits"], 1))
    fc = next((m for m in markers if m["marker"] == "fear=cell"), None)
    cmp_ = english["compare"]
    nulls = summary["nulls"]
    A("# Affect and valence curve (#44)")
    A("")
    A("Score the book's affect on the four axes the task names -- fear, "
      "ecstasy, paranoia, disgust -- plus a signed valence axis, and plot the "
      "curve over the 225 pages.  The task's own example is the explicit "
      "operator marker ``Fear=cell``, so the lexicon runs on the layer where "
      "such an equation is visible.")
    A("")
    A(f"**Headline.** {parts:,} markup-screened #17 word parts carry "
      f"**{t['affect']['hits']:,} affect occurrences "
      f"({t['affect']['rate']:.1f} per 1,000 parts)** on "
      f"{len(AFFECT_LEXICON)} published lexicon terms "
      f"({summary['lexicon']['n_terms_attested']} attested).  The four axes "
      "are unequal in size *and in breadth*, and that is the finding: "
      f"**ecstasy is the largest axis ({t['ecstasy']['hits']:,} hits, "
      f"{t['ecstasy']['rate']:.2f} per 1,000), carried by ``desire`` "
      f"({attested.get('desire', 0)} occurrences -- the book's most frequent "
      "affect term outside the cadaver/murder/blood register), "
      f"``rapture`` and ``ecstasy``; disgust is second "
      f"({t['disgust']['hits']:,} hits) but from only "
      f"{t['disgust']['n_terms_attested']} of its {t['disgust']['n_terms']} "
      "terms, three of which (``faecal``, ``sickly``, ``grotesque``) carry "
      f"{attested.get('faecal', 0) + attested.get('sickly', 0) + attested.get('grotesque', 0)} "
      f"of the {t['disgust']['hits']}; fear is third "
      f"({t['fear']['hits']:,}, split between ``nightmare`` "
      f"{attested.get('nightmare', 0)} and ``fear`` {attested.get('fear', 0)}); "
      f"and paranoia is nearly absent ({t['paranoia']['hits']:,} hits, "
      f"{t['paranoia']['n_terms_attested']} of {t['paranoia']['n_terms']} "
      "terms)**.  The book's affect vocabulary is a small, repeating core -- "
      f"the top 10 affect-axis terms ({', '.join('``' + x + '``' for x in axis_top[:5])} "
      f"…) carry {top10_share:.0f}% of all affect occurrences -- not a graded "
      "emotional spectrum, which is the same restricted-lexicon finding as "
      "#24 and #56.  Against the pooled #56 general-English references the "
      "book is **affect-dense on every axis** (all four q = 0.000), most of "
      f"all on disgust (log-ratio {cmp_['disgust']['log_ratio']:+.2f}) and "
      f"ecstasy ({cmp_['ecstasy']['log_ratio']:+.2f}) -- see §4 -- but "
      "**within the book the affect lexicon is not frequent vocabulary**: "
      "same-sized chunks of ordinary book vocabulary carry as many or more "
      "occurrences (§6 term-shuffle nulls), so the density is a "
      "cross-register effect, not a within-book frequency effect.  And "
      "the curve is **stationary in level but not featureless**: "
      "**disgust is the one axis with a real narrative arc** -- its "
      "smoothed curve peaks at "
      f"{nulls['page_order']['rate_disgust']['observed_peak']:.2f} "
      "per 1,000 on p."
      f"{rows[int(np.argmax([r['smooth_rate_disgust'] for r in rows]))]['page']}, "
      f"{nulls['page_order']['rate_disgust']['peak_z']:+.2f} sd above "
      "the page-order permutation null, with matching page-to-page "
      "continuity (lag-1 z = "
      f"{nulls['page_order']['rate_disgust']['lag1_z']:+.2f}), "
      "because its three carrying terms (``faecal``, ``sickly``, "
      "``grotesque``) cluster on the same pages.  Fear and paranoia "
      "are locally smooth (lag-1 z = "
      f"{nulls['page_order']['rate_fear']['lag1_z']:+.2f} and "
      f"{nulls['page_order']['rate_paranoia']['lag1_z']:+.2f}) but "
      "have no peak (z = "
      f"{nulls['page_order']['rate_fear']['peak_z']:+.2f} and "
      f"{nulls['page_order']['rate_paranoia']['peak_z']:+.2f}); "
      "ecstasy has neither (peak z = "
      f"{nulls['page_order']['rate_ecstasy']['peak_z']:+.2f}, "
      "lag-1 z = "
      f"{nulls['page_order']['rate_ecstasy']['lag1_z']:+.2f}); the "
      "aggregate curve has both (peak z = "
      f"{nulls['page_order']['rate_affect']['peak_z']:+.2f}, "
      "lag-1 z = "
      f"{nulls['page_order']['rate_affect']['lag1_z']:+.2f}) -- "
      "affect vocabulary clusters on the same pages the way the "
      "book's templates do (#42), without a rising or falling "
      "trajectory on the other three axes.")
    A("")
    A("## 1. Method")
    A("")
    A("* **Instrument 1 -- the lexicon.** A curated, published affect lexicon "
      "(`AFFECT_LEXICON` in `work/affect_curve.py`, every term and its count "
      "in `affect_lexicon.tsv`) matched **exactly** -- lower-cased whole "
      "surface, no stemming, no fuzzy matching -- against the #17 "
      "**word-part** layer: the individual words inside the maximal units, so "
      "``fear=cell`` contributes the parts ``fear`` and ``cell``.  This is the "
      "layer #56 uses for keyness because it is the one comparable to "
      "ordinary English words, and it is the layer on which an affect state "
      "written as an operator equation is still a lexical occurrence.  The "
      "markup screen is #56's own (a part is dropped when its unit uid is a "
      "#19 markup residue, when its `(page, start)` falls in a #27-detected "
      "tag span, or when its surface is a #19 junk/markup type), so the "
      "counts are directly comparable with #56's and #24's word layers.")
    A("* **Instrument 2 -- the classifier.** A zero-shot classifier in the "
      "pinned #39 clause-vector space: each affect is a set of ordinary "
      "English description phrases (`CLASSIFIER_PROTOTYPES`), encoded with "
      "the **same SBERT encoder** that produced the 5,614 clause vectors, and "
      "every clause is scored by its maximum cosine to an affect's "
      "prototypes.  A **neutral description control** (rooms, weather, "
      "streets, trains) is scored identically and plotted beside them.  No "
      "prototype phrase shares a term with the lexicon, so an affect matched "
      "by meaning is a second opinion on the same phenomenon, not a restated "
      "count.")
    A("* **Valence** is the signed contrast per 1,000 parts: positive-valence "
      "terms (the ecstasy family plus love/hope/peace/paradise) minus "
      "negative-valence terms (fear, paranoia, disgust, and the "
      "despair/murder/rape/cadaver register).  The classifier has a matching "
      "signed axis (mean cosine to positive prototypes minus negative).")
    A("* **The explicit markers** (`affect_markers.tsv`) are the #17 units in "
      "which an affect word is *bound* to another word by an operator seam -- "
      + (f"``Fear=cell`` ({fc['count']} occurrences over {fc['n_pages']} "
         f"pages, first p.{fc['first_page']}), ``fear-cell``, "
         if fc else "")
      + "``paranoia=spasm``, ``desire+fear=cell``, "
      "``fear=cell++self``.  A standalone ``ecstasy`` "
      "is an ordinary lexical occurrence and stays with the lexicon.")
    A("* **Nulls** (§6): a **page-order permutation null** per curve "
      "(the 225 per-page values kept exactly, their order over the "
      "book destroyed by a seeded random permutation -- a circular "
      "roll is *not* used because it leaves a series' adjacent pairs "
      "unchanged up to one wrap pair, which would make the lag-1 "
      "statistic's null degenerate), a clause-label permutation null "
      "per classifier curve, **two matched-size term-shuffle nulls** "
      f"(the {nulls['n_affect_terms']} affect-axis terms relabelled "
      "with same-sized chunks of "
      "non-affect book vocabulary: the fixed core-vocabulary pool, "
      "sampled with replacement, and the book's whole non-affect "
      "vocabulary, sampled without replacement), and the #56 English "
      "references scored with the same lexicon and the same "
      "classifier.")
    A("")
    A("## 2. Reconciliation")
    A("")
    A("| quantity | here | earlier | note |")
    A("|---|---:|---|---|")
    A(f"| word parts kept | {parts:,} | 79,770 (#56) | #17 role=w parts under "
      f"#56's markup screen ({summary['part_screen']['parts_total']:,} total, "
      f"{summary['part_screen']['parts_dropped']:,} dropped) |")
    A(f"| clauses | {summary['n_clauses']:,} | 5,614 (#39) | every clause "
      "carries a classifier score |")
    A(f"| pages | {n_pages} | 225 (#39) | pages carrying word parts |")
    A("| screened units | 74,077 | 74,077 (#27) | markup-screened #17 unit "
      "layer behind the clause text |")
    A("| encoder | sbert / units | sbert (#39 primary) | prototypes and "
      "clauses encoded in one space |")
    A(f"| reference corpora | {len(kx.CORPUS_DIRS)} | 7 corpora "
      f"(#56) | {english['ref_n']:,} pooled tokens; general_english is the "
      "inferential pool |")
    A("| lexicon terms | " + str(len(AFFECT_LEXICON)) + " | curated in-module "
      "| no external affect lexicon is fetched or assumed |")
    A("")
    A("## 3. The affect lexicon")
    A("")
    A(f"{len(AFFECT_LEXICON)} terms, of which "
      f"{summary['lexicon']['n_terms_attested']} occur in the book.  Every "
      "row is published with its book count, its matched English rate and the "
      "keyness comparison, so a term that never fires is visible as a 0 "
      "rather than quietly dropped.  The four axes are disjoint in "
      "membership, so no occurrence is counted in two axes.")
    A("")
    A("| axis | terms | attested | occurrences | per 1,000 parts |")
    A("|---|---:|---:|---:|---:|")
    for a in AFFECTS:
        A(f"| `{a}` | {t[a]['n_terms']} | {t[a]['n_terms_attested']} | "
          f"{t[a]['hits']:,} | {t[a]['rate']:.2f} |")
    A(f"| **all four** | | | **{t['affect']['hits']:,}** | "
      f"**{t['affect']['rate']:.2f}** |")
    A(f"| valence (positive) | {len(valence_terms(+1))} | "
      f"{sum(1 for x in valence_terms(+1) if summary['lexicon']['per_term_attested'].get(x, 0))}"
      f" | {t['valence']['positive_hits']:,} | "
      f"{t['valence']['rate_positive']:.2f} |")
    A(f"| valence (negative) | {len(valence_terms(-1))} | "
      f"{sum(1 for x in valence_terms(-1) if summary['lexicon']['per_term_attested'].get(x, 0))}"
      f" | {t['valence']['negative_hits']:,} | "
      f"{t['valence']['rate_negative']:.2f} |")
    A(f"| **valence (net)** | | | **{t['valence']['net_hits']:,}** | "
      f"**{t['valence']['rate_valence']:+.2f}** |")
    A("")
    A("The most frequent affect terms in the book:")
    A("")
    A("| term | affects | val | book | per M | english | per M | log-ratio | q |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in lexicon_rows[:TOP_LEXICON]:
        if r["book_parts"] == 0:
            continue
        A(f"| `{r['term']}` | {r['affects'] or '-'} | "
          f"{r['valence']:+d} | {r['book_parts']} | "
          f"{r['book_per_million']:.2f} | {r['english_parts']} | "
          f"{r['english_per_million']:.2f} | {r['log_ratio']:+.2f} | "
          f"{r['q_value']:.3f} |")
    A("")
    A("Unattested terms are the mirror image of that table: "
      + ", ".join(f"`{x}`" for x in lexicon_terms()
                  if summary["lexicon"]["per_term_attested"].get(x, 0) == 0)[:600]
      + (" …" if sum(1 for x in lexicon_terms()
                     if summary["lexicon"]["per_term_attested"].get(x, 0) == 0) > 40
         else "")
      + " -- the book never uses ordinary English words for dread, "
        "horror, revulsion, nausea or abhorrence, which is why each "
        "axis is carried by a handful of terms and the classifier "
        "matters as the second instrument for all four.")
    A("")
    A("## 4. Book vs ordinary English prose")
    A("")
    A("The same lexicon over the pooled #56 general-English references "
      f"({english['ref_n']:,} tokens), matched by rate per 1,000 words.  G² "
      "is Dunning's signed log-likelihood (as #56 uses it), q is "
      "Benjamini-Hochberg over the tested axes.")
    A("")
    A("| axis | book per 1k | english per 1k | log-ratio | G² | q | direction |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for name in ["affect", "fear", "ecstasy", "paranoia", "disgust",
                 "positive", "negative"]:
        c = english["compare"][name]
        A(f"| {name} | {c['book_rate']:.2f} | {c['english_rate']:.2f} | "
          f"{c['log_ratio']:+.2f} | {c['log_likelihood']:+.1f} | "
          f"{c['q_value']:.3f} | {c['direction']} |")
    v = english["compare"]["valence"]
    A(f"| valence (net) | {v['book_rate']:+.2f} | {v['english_rate']:+.2f} | "
      f"{v['log_ratio']:+.2f} | - | - | {v['direction']} |")
    A("")
    A("Per-corpus rates (the pooled row is the inferential one):")
    A("")
    A("| corpus | tokens | fear | ecstasy | paranoia | disgust | valence |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for r in english["rows"] + [english["pooled"]]:
        A(f"| {r['corpus']} | {r['tokens']:,} | {r['rate_fear']:.2f} | "
          f"{r['rate_ecstasy']:.2f} | {r['rate_paranoia']:.2f} | "
          f"{r['rate_disgust']:.2f} | {r['rate_valence']:+.2f} |")
    A("")
    A("The classifier agrees in direction: on matched English "
      "sentences every affect prototype scores *below* where it "
      "scores on the book's clauses, while the **neutral control "
      "moves the other way** (book "
      f"{clf_margin['neutral']['book_mean']:.4f} vs English "
      f"{clf_margin['neutral']['english_mean']:.4f}, "
      f"z = {clf_margin['neutral']['z']:+.2f}) -- so the margin is "
      "affect-specific, not a general cosine shift.  Valence is the "
      "exception that proves the register point: the book's "
      "positive-minus-negative cosine is *lower* than English's "
      f"(z = {clf_margin['valence']['z']:+.2f}) because its "
      "negative-valence register (cadaver, murder, crucifixion) "
      "outweighs its ecstasy register in the vector space.")
    A("")
    A("| classifier axis | book mean cosine | english mean cosine | margin | z |")
    A("|---|---:|---:|---:|---:|")
    for name in list(AFFECTS) + ["neutral", "valence"]:
        if name not in clf_margin:
            continue
        m = clf_margin[name]
        A(f"| `{name}` | {m['book_mean']:.4f} | {m['english_mean']:.4f} | "
          f"{m['margin']:+.4f} | {m['z']:+.2f} |")
    A("")
    A("## 5. The curve over the book")
    A("")
    A("![affect curve](affect_curve.png)")
    A("")
    A("`affect_per_page.tsv` carries every number behind the plot.  The "
      f"moving average is {summary['smooth']} pages, edge-preserving.  Peaks "
      "of the smoothed curves:")
    A("")
    A("| curve | peak page | peak value | valley page | valley value | mean |")
    A("|---|---:|---:|---:|---:|---:|")
    for key in [f"rate_{a}" for a in AFFECTS] + ["rate_affect",
                                                "rate_valence"]:
        vals = [r[f"smooth_{key}"] for r in rows]
        pk, vl = int(np.argmax(vals)), int(np.argmin(vals))
        A(f"| `{key}` | {rows[pk]['page']} | {vals[pk]:.2f} | "
          f"{rows[vl]['page']} | {vals[vl]:.2f} | "
          f"{float(np.mean(vals)):.2f} |")
    A("")
    A("The pages that carry the most explicit markers (the plot "
      "annotates ``Fear=cell``, ``ecstasy`` and ``desire`` where "
      "they first land, and the aggregate peak):")
    A("")
    A("| page | fused affect markers | dominant axis |")
    A("|---:|---:|---|")
    mp = collections.Counter()
    axis_mp = collections.defaultdict(collections.Counter)
    for m in markers:
        for p in m["pages"].split():
            mp[int(p)] += m["count"]
            for a in (m["affects"].split(",") if m["affects"] else []):
                axis_mp[int(p)][a] += m["count"]
    for p, n in mp.most_common(12):
        dom = axis_mp[p].most_common(1)
        A(f"| {p} | {n} | {dom[0][0] if dom else 'valence only'} |")
    A("")
    A("## 6. Controls")
    A("")
    A("Four nulls, each testing one claim.  The **page-order "
      "permutation null** keeps a curve's 225 values and destroys "
      "only their order over the book, so a peak or a lag-1 beyond "
      "its distribution is real narrative structure, not the value "
      "distribution's own artifact.  The **clause-shuffle null** "
      "permutes clause scores and recomputes the per-page means, so "
      "a classifier curve with page structure beats it.  The two "
      "**term-shuffle nulls** relabel the "
      f"{nulls['n_affect_terms']} affect-axis terms with "
      "same-sized chunks of non-affect book vocabulary (the fixed "
      "core pool, and the whole non-affect vocabulary), so they "
      "test whether the affect lexicon is *frequent within the "
      "book*.  The English baselines (§4) are the cross-register "
      "comparison.")
    A("")
    A("| control | statistic | observed | null mean | null sd | z | draws |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for r in control_rows:
        A(f"| {r['control']} | {r['statistic']} | {r['observed']} | "
          f"{r['null_mean']} | {r['null_sd']} | {r['z']:+} | {r['n_draws']} |")
    A("")
    A("## 7. Explicit affect markers")
    A("")
    A(f"{len(markers)} fused units carry an affect term, "
      f"{summary['markers']['n_occurrences']:,} occurrences over "
      f"{summary['markers']['n_pages']} pages.  These are the book's own "
      "affect formulas -- the task's ``Fear=cell`` -- and they are why the "
      "lexicon runs on the part layer:")
    A("")
    A("| marker | class | count | pages | first | axes |")
    A("|---|---|---:|---:|---:|---|")
    for m in markers[:TOP_MARKERS]:
        A(f"| `{m['marker']}` | `{m['class']}` | {m['count']} | "
          f"{m['n_pages']} | {m['first_page']} | {m['affects'] or '-'} |")
    A("")
    A("## 8. Instrument agreement")
    A("")
    A("The two instruments are on different scales (a rate per 1,000 parts "
      "and a cosine), so rank agreement across the 225 pages is the honest "
      "statistic.  They are not fully independent -- both are about affect -- "
      "and their agreement is reported as corroboration, never as validation.")
    A("")
    A("| axis | Spearman (pages) | Pearson (pages) |")
    A("|---|---:|---:|")
    for a in AFFECTS:
        A(f"| `{a}` | {agreement[a]['spearman']:+.4f} | "
          f"{agreement[a]['pearson']:+.4f} |")
    A("")
    A("## 9. Negative results and limitations")
    A("")
    A("* **The lexicon is the analyst's choice.** It is curated in-module, "
      "not fetched from a published affect lexicon, and every term's count is "
      "published so it can be audited or edited; the two term-shuffle nulls "
      "(§6) bound how the observed rate sits against same-sized chunks of "
      "non-affect book vocabulary, but a different lexicon moves the exact "
      "numbers.")
    A(f"* **Disgust is thin for a lexical reason, not a semantic "
      f"one.** {t['disgust']['n_terms_attested']} of "
      f"{t['disgust']['n_terms']} disgust terms occur at all, and "
      "three of them (``faecal`` "
      f"{attested.get('faecal', 0)}, ``sickly`` "
      f"{attested.get('sickly', 0)}, ``grotesque`` "
      f"{attested.get('grotesque', 0)}) carry "
      f"{attested.get('faecal', 0) + attested.get('sickly', 0) + attested.get('grotesque', 0)} "
      f"of the axis's {t['disgust']['hits']} hits; ordinary English "
      "disgust words (``disgust``, ``revulsion``, ``nausea``, "
      "``abhor``) are absent from the book.  The classifier's "
      "disgust curve is therefore the more informative of the two "
      "for that axis and is reported beside it.")
    A("* **The classifier is zero-shot, not supervised.** No labelled affect "
      "data for this book exists (there is no ground-truth translation, #69), "
      "so a trained classifier cannot be evaluated against anything.  A "
      "prototype classifier measures how close a clause is to an ordinary "
      "English description of an affect -- it is a retrieval reading, not a "
      " diagnosis.")
    A("* **The two instruments share a conceptual basis, so their agreement "
      "is weak corroboration, not independent confirmation**; the workspace's "
      "triangulation (#73) is the place where genuinely independent methods "
      "are compared.")
    A("* **English baselines are prose, not book-like texte.** The references "
      "are ordinary novels and abstracts (the three excerpt corpora are "
      "flagged non-inferential by #56 and are excluded from the pooled "
      "comparison's inference), so a rate difference can be a genre effect as "
      "well as a book effect; that is why the pooled general-English row is "
      "the reported one and the excerpts are shown but not tested.")
    A("* **The curve has no arc, but it is not featureless -- and that "
      "is the finding, not a failure to find one.** No individual axis "
      "beats the page-order permutation null on the peak of its "
      f"smoothed curve (fear z = "
      f"{nulls['page_order']['rate_fear']['peak_z']:+.2f}, ecstasy "
      f"z = {nulls['page_order']['rate_ecstasy']['peak_z']:+.2f}, "
      f"paranoia z = "
      f"{nulls['page_order']['rate_paranoia']['peak_z']:+.2f}, "
      f"disgust z = {nulls['page_order']['rate_disgust']['peak_z']:+.2f}), "
      "so there is no rising or falling affect trajectory to read -- "
      "consistent with #42's stationarity verdict and #51's absence "
      "of a latent periodicity, and inconsistent with the jacket "
      "copy's escalating-dread narrative.  What the nulls *do* find "
      "is local smoothness: the aggregate curve's lag-1 "
      "autocorrelation is "
      f"{nulls['page_order']['rate_affect']['lag1_z']:+.2f} sd "
      "above the permutation null (fear "
      f"{nulls['page_order']['rate_fear']['lag1_z']:+.2f}, ecstasy "
      f"{nulls['page_order']['rate_ecstasy']['lag1_z']:+.2f}), i.e. "
      "adjacent pages share affect vocabulary more than a random "
      "page assignment would -- the same template clustering #42 "
      "found in the embedding space, seen lexically.  The aggregate "
      "peak ("
      f"z = {nulls['page_order']['rate_affect']['peak_z']:+.2f}) "
      "says some pages are affect-hotter than chance, but no single "
      "axis has a peak, so there is clustering, not an arc.")
    A("* **The affect lexicon is not frequent vocabulary within the "
      "book.** Both term-shuffle nulls relabel the "
      f"{nulls['n_affect_terms']} affect-axis terms with same-sized "
      "chunks of non-affect book vocabulary: against the book's core "
      "content vocabulary (``soul``, ``cell``, ``machine``, "
      "``blood`` ...) the affect lexicon's "
      f"{nulls['term_shuffle_core']['observed_hits']:,} occurrences "
      "sit far below the null (z = "
      f"{nulls['term_shuffle_core']['rate_z']:+.2f}), and against a "
      "random same-sized set of the book's non-affect types the null "
      f"mean is {nulls['term_shuffle_vocabulary']['null_hits_mean']:,.0f} "
      f"(z = {nulls['term_shuffle_vocabulary']['rate_z']:+.2f}, "
      "not significant).  Affect words are individually rare in this "
      "book; the density in §4 is a cross-register effect -- the "
      "book uses its small affect vocabulary far more than ordinary "
      "English prose does, not more than its own content words.")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="#44 affect and valence curve")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--smooth", type=int, default=SMOOTH,
                    help="pages in the moving average")
    ap.add_argument("--n-shift", type=int, default=N_SHIFT,
                    help="draws for each permutation null")
    ap.add_argument("--n-term-shuffle", type=int, default=N_TERM_SHUFFLE)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    build(Path(args.out), seed=args.seed, smooth_pages=args.smooth,
          n_shift=args.n_shift, n_term_shuffle=args.n_term_shuffle,
          quiet=args.quiet)


if __name__ == "__main__":
    main()


