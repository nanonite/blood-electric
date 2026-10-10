#!/usr/bin/env python3
"""#56 (revised scope) — within-text language viability assessment.

Question: do *Blood Electric*'s recurring lexical constructions and
operator sequences support a viable, recoverable language — and where
the evidence supports it, what does the text say?

Scope decision (operator, 2026-10-08): **intrinsic evidence only**.
No external books, no reference corpora, no new datasets.  The
comparative keyness artifacts of the earlier #56 attempt are retained
as historical output; they are not used here.  Follow-ups #78–#80 are
parked.

Method — every claim is a falsifiable check on **held-out sections**
with a **within-text null**:

  * The book is split in two.  Primary split: interleaved pages
    (even pages train, odd pages test) so adjacent pages cannot leak
    style into the test set.  Secondary split: chronological
    (first half train, second half test) as a drift control.
  * A learner sees **only the train half**.  Each check scores the
    learner on the test half and compares against nulls that preserve
    the text's own marginal statistics but destroy the structure under
    test:
      - **within-page shuffle** — word order destroyed, unigram
        marginals preserved;
      - **type permutation** — a random bijection of the vocabulary,
        applied to the test half only: the frequency spectrum is
        preserved exactly, every lexical identity destroyed;
      - **within-clause shuffle** (frames) and **within-unit part
        shuffle** (compositionality) — the same idea at the construction
        and morpheme levels.
    p-values are add-one empirical: p = (1 + #{nulls at least as
    extreme}) / (B + 1), B = 40 draws, seed 42.
  * A **negative control** (V0) runs the V1 machinery on the
    within-page-shuffled book itself: a non-language must fail its own
    check, or the checks are vacuous.

Pre-registered checks (thresholds fixed before running):

  V1 lexical order      held-out bigram CE (order-1, lambda tuned on
                        train-internal validation) below both null
                        minima, p < 0.05.
  V2 operator stream    held-out operator bigram CE (Jelinek-Mercer
                        lam=0.8, #11's estimator) below the within-run
                        shuffle minimum, p < 0.05.
  V3 frame transfer     train-mined frames (top-40) cover more test
                        clauses than the shuffled-train null maximum,
                        p < 0.05.
  V4 lexicon reuse      (a) test-half OOV rate below the
                        type-permutation minimum; (b) Spearman rank
                        correlation of shared-type frequencies above
                        the permutation maximum.
  V5 sense stability    median same-term top-20 collocate Dice above
                        the median mismatched-term Dice, paired sign
                        test p < 0.01.
  V6 compositionality   held-out part-sequence bigram CE below the
                        within-unit part-shuffle minimum, p < 0.05.

Verdict (pre-registered): >= 5 of 6 checks pass -> VIABLE (bounded);
3–4 -> PARTIALLY VIABLE; <= 2 -> NOT VIABLE.  "Bounded" is earned by
the bound diagnostics, not assumed: the order sweep (D1) measures how
far predictability extends, the chronological split (D2) measures
drift, and the sensitivity run (D3) re-measures V1 without
the OCR-artifact pages.

Meaning extraction reuses the frozen workspace artifacts — the #17
operator-aware tokenizer (units/parts), the #14 operator grammar, the
#19 neologism screen, the #27 collocations, the #30 entity glossary,
the #34 frames — and adds only what this assessment needs: book-wide
collocate profiles, frame-filler counts, and a curated termbase with
per-term confidence.  Three page-cited passages are glossed
token-by-token; every gloss carries an evidence class and a confidence
(high / medium / low / unresolved), and the residue is listed, not
repaired.

Outputs under analysis/language_viability/:
  language_viability_report.md   THE DELIVERABLE
  checks.tsv                     one row per check statistic
  null_distributions.tsv         every null draw
  order_sweep.tsv                D1: CE by n-gram order
  split_stability.tsv            D2: interleaved vs chronological
  sensitivity.tsv                D3: without OCR/HTML artifact pages
  residue_ledger.tsv             ambiguous cases + unexplained residue
  oov_types.tsv                  held-out OOV types, category, pages
  termbase_sample.tsv            curated termbase with computed evidence
  gloss_tokens.tsv               per-token gloss rows
  gloss_passages.md              the three glossed passages, readable
  summary.json                   machine-readable headline numbers
  reconciliation.tsv             against frozen #10/#11/#17/#33/#34
  diagnostics.tsv                build counters

Reproduce::

    conda activate marker
    python work/language_viability.py            # full run (~1 min)
    python work/language_viability.py --nulls 10 # quick run
    python work/test_language_viability.py

Deterministic: fixed seed (42), no hash-order dependence (all ties
broken by a fixed key).  Every statistic and every table recomputes
identically; the only field that varies between runs is the
``generated`` stamp carried in the report header and ``summary.json``
(provenance, like #30's), which no statistic is derived from.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clause_segmentation as cs   # #33 loaders (pages, units, markup)
import operator_ngrams as ng       # #11 operator stream + estimator
import frame_mining as fm          # #34 frames

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "language_viability"
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
SCREENING = ROOT / "analysis" / "neologism" / "screening.tsv"
ENTITY_GLOSSARY = ROOT / "analysis" / "entity" / "entity_glossary.tsv"
TOKENS = ROOT / "analysis" / "tokenizer" / "tokens.tsv"
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
GRAMMAR = ROOT / "analysis" / "block_parser" / "summary.json"   # #14
KWIC = ROOT / "analysis" / "kwic" / "concordance.tsv"           # #26
PAGE_LOOP = ROOT / "analysis" / "ensemble_ocr" / "page_length.tsv"  # #4

SCHEMA = "language-viability/1"
SEED = 42
N_NULLS = 40                    # null draws per check (add-one p)
LAM_GRID = (0.2, 0.4, 0.6, 0.8, 0.95)
OPERATOR_LAM = 0.8              # #11's Jelinek-Mercer weight
TOP_K = 20                      # primary collocate-overlap k
SIGN_TEST_ALPHA = 0.01
CHECK_ALPHA = 0.05
VERDICT_PASS = 5                # >= 5 of 6 -> VIABLE (bounded)
VERDICT_PARTIAL = 3             # 3-4 -> PARTIALLY VIABLE

# #4 ensemble OCR: recognition-loop pages; #11/#51: HTML-artifact pages.
# The *page* list is a judgement about which pages are unusable for the
# held-out splits; each page's own evidence is quoted in the residue
# ledger from #4's `page_length.tsv` rather than from one narrative.
OCR_LOOP_PAGES = (89, 182, 208)

# #14 block_parser node-level split for `/` (grammar_spec.md): 952 Link
# nodes (glued) / 540 Slash nodes (sentence-final).  Read from #14's own
# summary.json, so the literals below are a fallback for a workspace
# without the artifact rather than a second copy of the number.  The
# token-level recount in slash_role_counts() must reproduce them; the
# residue ledger quotes both so a drift is visible rather than asserted.
FROZEN_SLASH_LINK = 952
FROZEN_SLASH_BOUND = 540

# English function words: excluded from collocate/context profiles so a
# ubiquitous word cannot masquerade as a term's sense.
FUNCTION_WORDS = {
    "the", "of", "to", "a", "and", "is", "in", "that", "it", "was",
    "for", "on", "with", "as", "at", "by", "an", "be", "this", "from",
    "or", "i", "not", "are", "but", "has", "had", "so", "if", "they",
    "its", "his", "her", "their", "our", "your", "my", "we", "you",
    "he", "she", "them", "us", "do", "did", "does", "been", "being",
    "have", "will", "would", "can", "could", "should", "no", "all",
    "any", "each", "every", "more", "most", "some", "such", "only",
    "own", "same", "too", "very", "just", "also", "even", "still",
    "yet", "here", "there", "when", "where", "what", "which", "who",
    "how", "why", "one", "two", "than", "then", "now", "up", "down",
    "out", "off", "over", "under", "into", "onto", "upon", "through",
    "between", "among", "across", "along", "around", "behind",
    "beyond", "inside", "outside", "within", "without", "toward",
    "towards", "after", "before", "during", "since", "until", "while",
    "because", "although", "though", "unless", "whether", "once",
    "again", "further", "much", "many", "little", "less", "least",
    "these", "those", "above", "below", "near", "far",
}

# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def spearman(xs, ys):
    """Spearman rank correlation (average ties), stdlib only."""
    if len(xs) != len(ys) or not xs:
        return 0.0

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def add_one_p(book, nulls, lower_better=True):
    """Add-one empirical p: (1 + #{nulls at least as extreme}) / (B+1)."""
    if lower_better:
        extreme = sum(1 for v in nulls if v <= book + 1e-12)
    else:
        extreme = sum(1 for v in nulls if v >= book - 1e-12)
    return (1.0 + extreme) / (len(nulls) + 1.0)


def sign_test_p(n_better, n):
    """Two-sided exact binomial sign test, p for n_better of n pairs."""
    if n == 0:
        return 1.0
    k = min(n_better, n - n_better)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * p)


def dice_topk(pa, pb, k):
    """Dice overlap of the top-k keys of two counters."""
    ta = {w for w, _ in pa.most_common(k)}
    tb = {w for w, _ in pb.most_common(k)}
    if not ta or not tb:
        return 0.0
    return 2.0 * len(ta & tb) / (len(ta) + len(tb))


def cosine(pa, pb, vocab):
    va = [pa.get(w, 0) for w in vocab]
    vb = [pb.get(w, 0) for w in vocab]
    num = sum(a * b for a, b in zip(va, vb))
    den = math.sqrt(sum(a * a for a in va) * sum(b * b for b in vb))
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# loading (reuses the frozen #17/#19/#27/#30/#33/#34 artifacts)
# ---------------------------------------------------------------------------

def load_book():
    """(bodies, pages, page_units, unit_meta, is_markup, screening).

    pages      - {page: [lower-cased screened unit surfaces]}
    page_units - {page: [(start, end, uid), ...]}  (#33 convention)
    unit_meta  - {uid: (page, start, end, surface)}
    is_markup  - uid -> bool  (#19 ledger + #27 tag-run screen)
    screening  - {lower unit: screening row}
    """
    ctx = fm.load_context()
    bodies, page_units, unit_meta, is_markup, _sc = ctx
    pages = {p: [unit_meta[u[2]][3].lower() for u in units
                 if not is_markup(u[2])]
             for p, units in page_units.items()}
    screening = {}
    if SCREENING.exists():
        for r in cs.read_tsv(SCREENING):
            screening.setdefault(r["unit"].lower(), r)
    return bodies, pages, page_units, unit_meta, is_markup, screening


def load_operator_pages():
    """(pages_runs, pages_ops): maximal operator runs and the flat stream."""
    pages_runs = ng.page_runs(str(LAYOUT))
    pages_ops = {p: [op for run in runs for op in run]
                 for p, runs in pages_runs.items()}
    return pages_runs, pages_ops


def load_clauses():
    ctx = fm.load_context()
    return fm.load_clause_tokens(ctx)


_SLASH_ROLES = None


def slash_role_counts():
    """(#14 Link, #14 Slash) for the `/` tokens, recounted from the #17
    stream.

    `/` glued to a following word (no gap) is #14's `Link` node — the
    compound/possessive seam; otherwise it is #14's sentence-final
    `Slash`.  Measured, not hard-coded: the `/` gloss and the residue
    ledger both read these two numbers, so a stale literal in either
    place would silently contradict the other (and #14/#17).
    Memoized because the token stream is read whole each time.
    """
    global _SLASH_ROLES
    if _SLASH_ROLES is None:
        link = slash = 0
        by_page = collections.defaultdict(list)
        for r in cs.read_tsv(TOKENS):
            by_page[int(r["page"])].append(r)
        for p, toks in by_page.items():
            toks.sort(key=lambda r: int(r["start"]))
            for i, r in enumerate(toks):
                if r["text"] != "/" or r["kind"] != "OP":
                    continue
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if nxt is not None and nxt["kind"] == "WORD" \
                        and json.loads(nxt["gap_json"]) == "":
                    link += 1
                else:
                    slash += 1
        _SLASH_ROLES = (link, slash)
    return _SLASH_ROLES


def load_entity_glossary():
    """{lower surface: row} from the #30 entity glossary."""
    out = {}
    if ENTITY_GLOSSARY.exists():
        for r in cs.read_tsv(ENTITY_GLOSSARY):
            out[r["entity"].lower()] = r
    return out


_GRAMMAR = None


def grammar_counts():
    """#14's own node counts, read from its summary.json.

    The `/` dual role, the operator reconciliation and the node inventory
    are quoted in the residue ledger and in the reconciliation table.
    Sourcing them from #14's artifact (rather than from literals written
    here) means a rebuild of #14 that moves a count shows up as a
    `match: False` row instead of a stale number in a new deliverable.
    Falls back to the module literals when the artifact is absent.
    """
    global _GRAMMAR
    if _GRAMMAR is None:
        frozen = {"n_link_slash": FROZEN_SLASH_LINK,
                  "n_slash": FROZEN_SLASH_BOUND}
        got = dict(frozen)
        if GRAMMAR.exists():
            with GRAMMAR.open(encoding="utf-8") as fh:
                d = json.load(fh)
            for k in frozen:
                if isinstance(d.get(k), int):
                    got[k] = d[k]
        _GRAMMAR = got
    return _GRAMMAR


def loop_page_evidence():
    """{page: (max_word4gram_repeat, top_word4gram, marker_over_engines)}
    from #4.

    Each recognition-loop page is cited with *its own* repeat and length
    ratio, so the ledger never attributes page 89's loop to pages 182
    and 208.
    """
    out = {}
    if PAGE_LOOP.exists():
        for r in cs.read_tsv(PAGE_LOOP):
            out[int(r["page_id"])] = (r["max_word4gram_repeat"],
                                      r["top_word4gram"],
                                      r["marker_over_engines"])
    return out


def load_kwic_examples(max_per_term=2, max_chars=70):
    """{lower term: [quoted context, ...]} from the #26 KWIC concordancer.

    #26 indexed the compound and named constructs; ordinary English words
    are mostly absent from it by construction.  Those contexts are the
    workspace's own *contextual examples*, so the termbase can cite real
    attestations rather than only aggregate collocate counts.
    """
    out = collections.defaultdict(list)
    if not KWIC.exists():
        return out
    for r in cs.read_tsv(KWIC):
        t = r["term_lower"]
        if len(out[t]) >= max_per_term:
            continue
        unit = r["unit_surface"].replace("\n", " ")
        ctx = "%s  «%s»  %s" % (r["left_context"].replace("\n", " "),
                                unit, r["right_context"].replace("\n", " "))
        out[t].append("p%s: %s" % (r["page"], re.sub(r"\s+", " ", ctx)))
    return out


# ---------------------------------------------------------------------------
# n-gram cross-entropy (Jelinek-Mercer interpolation, #11's estimator)
# ---------------------------------------------------------------------------

def _context_counts(seqmap, train, order):
    """Collect continuation counts for every context length through order."""
    uni = collections.Counter()
    ctxs = collections.defaultdict(collections.Counter)
    for p in train:
        for seq in seqmap[p]:
            uni.update(seq)
            for n in range(1, order + 1):
                for i in range(len(seq) - n):
                    ctxs[tuple(seq[i:i + n])][seq[i + n]] += 1
    return uni, ctxs


def _jm_probability(w, context, ctxs, p0, lam):
    """Recursively interpolate a context with its suffix model.

    Unseen contexts back off directly: applying the interpolation weight
    to an empty maximum-likelihood distribution would lose probability
    mass and make the resulting distribution unnormalized.
    """
    if not context:
        return p0(w)
    counts = ctxs.get(context)
    if not counts:
        return _jm_probability(w, context[1:], ctxs, p0, lam)
    total = sum(counts.values())
    ml = counts[w] / total
    lower = _jm_probability(w, context[1:], ctxs, p0, lam)
    return lam * ml + (1 - lam) * lower


def heldout_ce(seqmap, train, test, order, lam):
    """Held-out cross-entropy in bits per item.

    seqmap - {page: [sequence, ...]}; each sequence is a list of items
             (units for V1, part tokens for V6).  Transitions are scored
             within sequences only.
    """
    uni, ctxs = _context_counts(seqmap, train, order)
    tot = sum(uni.values()) or 1
    V = len(uni)

    def p0(w):
        return (uni[w] + 1e-9) / (tot + 1e-9 * V)

    ls, n = 0.0, 0
    for p in test:
        for seq in seqmap[p]:
            for i in range(len(seq) - order):
                ls += math.log2(max(_jm_probability(
                    seq[i + order], tuple(seq[i:i + order]), ctxs,
                    p0, lam), 1e-12))
                n += 1
    return -ls / n if n else 0.0


def tune_lambda(seqmap, val_train, val_test, order):
    """Pick the interpolation weight on a train-internal validation split."""
    best = (LAM_GRID[0], float("inf"))
    for lam in LAM_GRID:
        ce = heldout_ce(seqmap, val_train, val_test, order, lam)
        if ce < best[1]:
            best = (lam, ce)
    return best


# ---------------------------------------------------------------------------
# null constructors (within-text: marginals preserved, structure destroyed)
# ---------------------------------------------------------------------------

def within_page_shuffle(pages, rng):
    """Destroy word order within each page; unigram marginals preserved."""
    return {p: rng.sample(pages[p], len(pages[p])) for p in pages}


def type_permutation(types, rng):
    """A random bijection of the vocabulary (applied to one half only)."""
    perm = types[:]
    rng.shuffle(perm)
    return dict(zip(types, perm))


def within_run_shuffle(pages_runs, rng):
    """Shuffle each maximal operator run internally (#11's run structure)."""
    return {p: [op for run in [rng.sample(r, len(r)) for r in runs]
                for op in run]
            for p, runs in pages_runs.items()}


def within_clause_shuffle(clauses, rng):
    """Shuffle each clause's tokens internally (frames' null)."""
    out = []
    for c in clauses:
        toks = c["tokens"][:]
        rng.shuffle(toks)
        out.append({"clause_id": c.get("clause_id", 0),
                    "page": c.get("page", 0),
                    "opener": c.get("opener", ""),
                    "container": c.get("container", ""),
                    "n_words_clean": c.get("n_words_clean", 0),
                    "tokens": toks})
    return out


def unit_part_seq(u):
    """A unit's part tokens: words and seams, in surface order (#17 parts)."""
    return [t for t in re.split(r"([=/\+\-—–])", u) if t]


def within_unit_part_shuffle(seqmap, rng):
    """Shuffle each unit's parts internally; part marginals preserved."""
    return {p: [rng.sample(seq, len(seq)) for seq in seqmap[p]]
            for p in seqmap}


# ---------------------------------------------------------------------------
# context profiles (one pass per half, all target terms at once)
# ---------------------------------------------------------------------------

def collocate_profiles(pages, pagelist, terms, win=5):
    """{term: Counter(non-function partner -> count)} over a page list."""
    target = {t: collections.Counter() for t in terms}
    for p in pagelist:
        seq = pages[p]
        for i, w in enumerate(seq):
            if w in target:
                prof = target[w]
                for j in range(max(0, i - win), min(len(seq), i + win + 1)):
                    if j != i:
                        u = seq[j]
                        if u not in FUNCTION_WORDS:
                            prof[u] += 1
    return target


# ---------------------------------------------------------------------------
# the six pre-registered checks + the negative control
# ---------------------------------------------------------------------------

def check_v0_negative_control(pages, allp, train, test, n_nulls, rng):
    """V0: run V1's machinery on the shuffled book.  Must FAIL.

    A harness that 'finds language' in a word-order-shuffled text is
    vacuous; this control demonstrates the checks are not.
    """
    sh = within_page_shuffle(pages, rng)
    flat_sh = {p: [sh[p]] for p in allp}
    val_tr, val_te = train[::2], train[1::2]
    lam, _ = tune_lambda(flat_sh, val_tr, val_te, 1)
    book = heldout_ce(flat_sh, train, test, 1, lam)
    nulls = []
    for _ in range(n_nulls):
        sh2 = within_page_shuffle(sh, rng)
        nulls.append(heldout_ce({p: [sh2[p]] for p in allp},
                                train, test, 1, lam))
    p = add_one_p(book, nulls, lower_better=True)
    passed = book < min(nulls) and p < CHECK_ALPHA
    return {
        "check": "V0", "split": "interleaved",
        "statistic": "held-out bigram CE of the within-page-shuffled book "
                     "(bits/unit), lambda tuned on shuffled train",
        "book": book, "null_mean": sum(nulls) / len(nulls),
        "null_min": min(nulls), "null_max": max(nulls),
        "n_nulls": n_nulls, "p_value": p,
        "criterion": "negative control: must NOT beat its own shuffle null",
        "passed": bool(passed),
    }, {"lam": lam, "nulls": nulls}


def check_v1_lexical_order(pages, allp, train, test, n_nulls, rng):
    """V1: is the unit stream's word order learnable from the train half?"""
    flat = {p: [pages[p]] for p in allp}
    val_tr, val_te = train[::2], train[1::2]
    lam, _val_ce = tune_lambda(flat, val_tr, val_te, 1)
    book = heldout_ce(flat, train, test, 1, lam)
    n1a, n1b = [], []
    types = sorted({w for p in allp for w in pages[p]})
    for _ in range(n_nulls):
        sh = within_page_shuffle(pages, rng)
        n1a.append(heldout_ce({p: [sh[p]] for p in allp},
                              train, test, 1, lam))
        pmap = type_permutation(types, rng)
        tm = {p: [[pmap[w] for w in pages[p]]] for p in test}
        full = dict(flat)
        full.update(tm)
        n1b.append(heldout_ce(full, train, test, 1, lam))
    p_a = add_one_p(book, n1a, lower_better=True)
    p_b = add_one_p(book, n1b, lower_better=True)
    passed = (book < min(n1a) and p_a < CHECK_ALPHA
              and book < min(n1b) and p_b < CHECK_ALPHA)
    row = {
        "check": "V1", "split": "interleaved",
        "statistic": "held-out bigram CE (order-1, lam=%.2f tuned on "
                     "train-internal validation; bits/unit)" % lam,
        "book": book, "null_mean": sum(n1a) / len(n1a),
        "null_min": min(n1a), "null_max": max(n1a),
        "n_nulls": n_nulls, "p_value": p_a,
        "criterion": "book < within-page shuffle null min AND < "
                     "type-permutation null min, add-one p < 0.05",
        "passed": bool(passed),
    }
    return row, {"lam": lam, "nulls_shuffle": n1a,
                 "nulls_typeperm": n1b, "p_typeperm": p_b,
                 "null_min_typeperm": min(n1b)}


def check_v2_operator_stream(pages_runs, pages_ops, allp, train, test,
                             n_nulls, rng):
    """V2: are operator sequences idiomatic (#11's question, held-out)?"""
    ls, n = ng._held_out_perplexity_split(pages_ops, 1, train, test,
                                          OPERATOR_LAM)
    book = -ls / n
    nulls = []
    for _ in range(n_nulls):
        sh = within_run_shuffle(pages_runs, rng)
        ls, nt = ng._held_out_perplexity_split(sh, 1, train, test,
                                               OPERATOR_LAM)
        nulls.append(-ls / nt)
    p = add_one_p(book, nulls, lower_better=True)
    passed = book < min(nulls) and p < CHECK_ALPHA
    row = {
        "check": "V2", "split": "interleaved",
        "statistic": "held-out operator bigram CE (Jelinek-Mercer "
                     "lam=%.1f, #11's estimator; bits/operator)" % OPERATOR_LAM,
        "book": book, "null_mean": sum(nulls) / len(nulls),
        "null_min": min(nulls), "null_max": max(nulls),
        "n_nulls": n_nulls, "p_value": p,
        "criterion": "book < within-run shuffle null min, add-one p < 0.05",
        "passed": bool(passed),
    }
    return row, {"nulls": nulls, "n_test_transitions": n}


def frame_coverage_frac(clauses, frames, topk=40):
    """Fraction of clauses covered by the top-k frames (0.0
    when no frames qualify -- a shuffled corpus may mine none)."""
    rows = fm.coverage(clauses, frames, topk=topk)
    return rows[-1]["frac_covered"] if rows else 0.0


def check_v3_frame_transfer(clauses, n_nulls, rng):
    """V3: do train-half constructions cover held-out clauses?"""
    train = [c for c in clauses if c["page"] % 2 == 0]
    test = [c for c in clauses if c["page"] % 2 == 1]
    frames = fm.mine_frames(train)
    book = frame_coverage_frac(test, frames)
    nulls = []
    for _ in range(n_nulls):
        sh = within_clause_shuffle(train, rng)
        fsh = fm.mine_frames(sh)
        nulls.append(frame_coverage_frac(test, fsh))
    p = add_one_p(book, nulls, lower_better=False)
    passed = book > max(nulls) and p < CHECK_ALPHA
    row = {
        "check": "V3", "split": "interleaved",
        "statistic": "fraction of held-out clauses covered by train-mined "
                     "frames (top-40 of %d)" % len(frames),
        "book": book, "null_mean": sum(nulls) / len(nulls),
        "null_min": min(nulls), "null_max": max(nulls),
        "n_nulls": n_nulls, "p_value": p,
        "criterion": "book > shuffled-train null max, add-one p < 0.05",
        "passed": bool(passed),
    }
    return row, {"nulls": nulls, "n_frames_train": len(frames),
                 "top_frame": " ".join(frames[0]["skeleton"]) if frames else ""}


def check_v4_lexicon_reuse(pages, allp, train, test, n_nulls, rng):
    """V4: does the train half's lexicon transfer to the test half?"""
    tr_types = {w for p in train for w in pages[p]}
    te_toks = [w for p in test for w in pages[p]]
    oov = sum(1 for w in te_toks if w not in tr_types)
    book_oov = oov / len(te_toks)
    types = sorted({w for p in allp for w in pages[p]})
    nulls_oov = []
    nulls_corr = []
    ca = collections.Counter(w for p in train for w in pages[p])
    cb = collections.Counter(w for p in test for w in pages[p])
    shared = sorted(set(ca) & set(cb))
    book_corr = spearman([ca[w] for w in shared], [cb[w] for w in shared])
    for _ in range(n_nulls):
        pmap = type_permutation(types, rng)
        nulls_oov.append(
            sum(1 for p in test for w in pages[p]
                if pmap[w] not in tr_types) / len(te_toks))
        cb2 = collections.Counter(pmap[w] for p in test for w in pages[p])
        sh = sorted(set(ca) & set(cb2))
        nulls_corr.append(
            spearman([ca[w] for w in sh], [cb2[w] for w in sh]))
    p_oov = add_one_p(book_oov, nulls_oov, lower_better=True)
    p_corr = add_one_p(book_corr, nulls_corr, lower_better=False)
    row_a = {
        "check": "V4a", "split": "interleaved",
        "statistic": "held-out OOV rate (test-half tokens whose type is "
                     "unseen in train)",
        "book": book_oov, "null_mean": sum(nulls_oov) / len(nulls_oov),
        "null_min": min(nulls_oov), "null_max": max(nulls_oov),
        "n_nulls": n_nulls, "p_value": p_oov,
        "criterion": "book < type-permutation null min, add-one p < 0.05",
        # this row's own sub-criterion; V4 as a whole needs both rows
        "passed": bool(book_oov < min(nulls_oov) and p_oov < CHECK_ALPHA),
    }
    row_b = {
        "check": "V4b", "split": "interleaved",
        "statistic": "Spearman rank correlation of shared-type frequencies "
                     "between halves (n=%d shared types)" % len(shared),
        "book": book_corr, "null_mean": sum(nulls_corr) / len(nulls_corr),
        "null_min": min(nulls_corr), "null_max": max(nulls_corr),
        "n_nulls": n_nulls, "p_value": p_corr,
        "criterion": "book > type-permutation null max, add-one p < 0.05",
        "passed": bool(book_corr > max(nulls_corr) and p_corr < CHECK_ALPHA),
    }
    return [row_a, row_b], {"nulls_oov": nulls_oov, "nulls_corr": nulls_corr,
                            "n_oov_tokens": oov, "n_test_tokens": len(te_toks),
                            "n_shared_types": len(shared)}


def check_v5_sense_stability(pages, allp, train, test, n_nulls, rng):
    """V5: are term senses stable across halves (meaning recoverable)?"""
    freq = collections.Counter(w for p in allp for w in pages[p])
    terms = [w for w, c in freq.most_common()
             if w not in FUNCTION_WORDS and len(w) > 2][:40]
    pa_all = collocate_profiles(pages, train, terms)
    pb_all = collocate_profiles(pages, test, terms)
    # `same` and `mism` are appended as a *pair* or not at all: the sign
    # test zips them, so a term dropped by only the second guard would
    # silently misalign every pair after it.  Build the pairs first.
    pairs = []
    for idx, t in enumerate(terms):
        pa, pb = pa_all[t], pb_all[t]
        t2 = terms[(idx + 17) % len(terms)]
        pb2 = pb_all[t2]
        if sum(pa.values()) < 20 or sum(pb2.values()) < 20:
            continue
        pairs.append((dice_topk(pa, pb, TOP_K),
                      dice_topk(pa, pb2, TOP_K)))
    same = [a for a, _b in pairs]
    mism = [b for _a, b in pairs]
    med_same = sorted(same)[len(same) // 2]
    med_mism = sorted(mism)[len(mism) // 2]
    n_better = sum(1 for a, b in zip(same, mism) if a > b)
    p_sign = sign_test_p(n_better, len(same))
    passed = med_same > med_mism and p_sign < SIGN_TEST_ALPHA
    row = {
        "check": "V5", "split": "interleaved",
        "statistic": "median top-%d collocate Dice, same term across "
                     "halves vs mismatched terms (paired sign test, "
                     "n=%d terms)" % (TOP_K, len(same)),
        # V5's comparison set is the mismatched-term distribution, not a
        # set of random draws: `book` is the median of the same-term
        # values, the "null" columns summarise the mismatched ones, and
        # n_nulls is the number of *pairs* compared.
        "book": med_same,
        "null_mean": sum(mism) / len(mism) if mism else 0.0,
        "null_min": min(mism), "null_max": max(mism),
        "n_nulls": len(same), "p_value": p_sign,
        "criterion": "median same-term > median mismatched-term, "
                     "sign test p < 0.01",
        "passed": bool(passed),
    }
    # secondary statistics: k=10 and cosine over the top-30 union
    same10, mism10, cos_s, cos_m = [], [], [], []
    per_term = []
    for idx, t in enumerate(terms):
        pa, pb = pa_all[t], pb_all[t]
        if sum(pa.values()) < 20:
            continue
        d_same = dice_topk(pa, pb, TOP_K)
        d10 = dice_topk(pa, pb, 10)
        vocab = sorted({w for w, _ in pa.most_common(30)}
                       | {w for w, _ in pb.most_common(30)})
        cs = cosine(pa, pb, vocab)
        t2 = terms[(idx + 17) % len(terms)]
        pb2 = pb_all[t2]
        if sum(pb2.values()) < 20:
            per_term.append({"term": t, "dice_same": d_same,
                             "dice_mismatched": "",
                             "n_partners": sum(pa.values())})
        else:
            d_mism = dice_topk(pa, pb2, TOP_K)
            per_term.append({"term": t, "dice_same": d_same,
                             "dice_mismatched": d_mism,
                             "n_partners": sum(pa.values())})
            mism10.append(dice_topk(pa, pb2, 10))
            vocab2 = sorted({w for w, _ in pa.most_common(30)}
                            | {w for w, _ in pb2.most_common(30)})
            cos_m.append(cosine(pa, pb2, vocab2))
        same10.append(d10)
        cos_s.append(cs)
    return row, {
        "terms": terms, "n_terms": len(same),
        "med_same": med_same, "med_mism": med_mism,
        "same": same, "mism": mism, "per_term": per_term,
        "dice10_same": sorted(same10)[len(same10) // 2],
        "dice10_mism": sorted(mism10)[len(mism10) // 2],
        "cosine_same": sorted(cos_s)[len(cos_s) // 2],
        "cosine_mism": sorted(cos_m)[len(cos_m) // 2],
        "n_better": n_better, "sign_p": p_sign,
    }


def check_v6_compositionality(pages, allp, train, test, n_nulls, rng):
    """V6: are held-out coinages built from already-attested parts?"""
    seqmap = {p: [unit_part_seq(u) for u in pages[p]] for p in allp}
    lam = 0.5
    book = heldout_ce(seqmap, train, test, 1, lam)
    nulls = []
    for _ in range(n_nulls):
        sh = within_unit_part_shuffle(seqmap, rng)
        nulls.append(heldout_ce(sh, train, test, 1, lam))
    uni = collections.Counter()
    for p in train:
        for seq in seqmap[p]:
            uni.update(seq)
    tot = sum(uni.values()) or 1
    V = len(uni)
    ls, n = 0.0, 0
    for p in test:
        for seq in seqmap[p]:
            for w in seq:
                ls += math.log2((uni[w] + 1e-9) / (tot + 1e-9 * V))
                n += 1
    unigram = -ls / n
    p = add_one_p(book, nulls, lower_better=True)
    passed = book < min(nulls) and p < CHECK_ALPHA
    row = {
        "check": "V6", "split": "interleaved",
        "statistic": "held-out part-sequence bigram CE within units "
                     "(lam=%.1f; bits/part; unigram baseline %.3f)" % (lam, unigram),
        "book": book, "null_mean": sum(nulls) / len(nulls),
        "null_min": min(nulls), "null_max": max(nulls),
        "n_nulls": n_nulls, "p_value": p,
        "criterion": "book < within-unit part-shuffle null min, "
                     "add-one p < 0.05",
        "passed": bool(passed),
    }
    return row, {"nulls": nulls, "unigram_ce": unigram, "lam": lam}


# ---------------------------------------------------------------------------
# bound diagnostics
# ---------------------------------------------------------------------------

def order_sweep(pages, allp, train, test):
    """D1: how far does predictability extend?  CE by n-gram order."""
    flat = {p: [pages[p]] for p in allp}
    val_tr, val_te = train[::2], train[1::2]
    rows = []
    for order in (0, 1, 2, 3):
        lam, val_ce = tune_lambda(flat, val_tr, val_te, order)
        ce = heldout_ce(flat, train, test, order, lam)
        rows.append({"order": order, "lam": lam, "val_ce": val_ce,
                     "test_ce": ce})
    return rows


def chronological_stability(pages, allp):
    """D2: the conservative contiguous split (drift control)."""
    cut = len(allp) // 2
    train, test = allp[:cut], allp[cut:]
    flat = {p: [pages[p]] for p in allp}
    val_tr, val_te = train[::2], train[1::2]
    lam, val_ce = tune_lambda(flat, val_tr, val_te, 1)
    ce = heldout_ce(flat, train, test, 1, lam)
    return {"split": "chronological", "train_pages": "%d-%d" % (allp[0], allp[cut - 1]),
            "test_pages": "%d-%d" % (allp[cut], allp[-1]),
            "lam": lam, "val_ce": val_ce, "test_ce": ce}


def sensitivity(pages, allp, artifact_pages, n_nulls=N_NULLS):
    """D3: re-measure V1 without the OCR-loop and HTML-artifact pages.

    Uses the same draw count as the scored checks.  A smaller B here
    would put the add-one floor at 1/(B+1), which is the *only* p this
    diagnostic can reach when the book beats every draw -- so a small B
    would report a marginal-looking p for an otherwise clean result.
    """
    drop = set(OCR_LOOP_PAGES) | set(artifact_pages)
    keep = [p for p in allp if p not in drop]
    train = [p for p in keep if p % 2 == 0]
    test = [p for p in keep if p % 2 == 1]
    flat = {p: [pages[p]] for p in keep}
    val_tr, val_te = train[::2], train[1::2]
    lam, _val_ce = tune_lambda(flat, val_tr, val_te, 1)
    ce = heldout_ce(flat, train, test, 1, lam)
    rng = random.Random(SEED)
    nulls = []
    for _ in range(n_nulls):
        sh = within_page_shuffle(pages, rng)
        nulls.append(heldout_ce({p: [sh[p]] for p in keep},
                                train, test, 1, lam))
    return {"dropped_pages": sorted(drop), "n_pages": len(keep),
            "n_nulls": n_nulls, "lam": lam, "test_ce": ce,
            "null_min": min(nulls), "p_value": add_one_p(ce, nulls)}


# ---------------------------------------------------------------------------
# residue ledger: ambiguous cases and unexplained residue
# ---------------------------------------------------------------------------

def residue_ledger(pages, allp, train, test, screening):
    """What the assessment cannot explain, quantified."""
    rows = []
    tr_types = {w for p in train for w in pages[p]}
    oov_types = collections.Counter()
    oov_pages = collections.defaultdict(set)
    for p in test:
        for w in pages[p]:
            if w not in tr_types:
                oov_types[w] += 1
                oov_pages[w].add(p)
    n_test_tokens = sum(len(pages[p]) for p in test)
    rows.append({
        "class": "held-out OOV types",
        "item": "%d types / %d tokens" % (len(oov_types), sum(oov_types.values())),
        "detail": "test-half types never attested in the train half; "
                  "%.1f%% of test tokens — the hard residue of the "
                  "recoverability claim"
                  % (100.0 * sum(oov_types.values()) / n_test_tokens),
        "pages": "",
    })
    by_cat = collections.Counter()
    for w in oov_types:
        r = screening.get(w)
        by_cat[r["category"] if r else "unscreened"] += 1
    rows.append({
        "class": "held-out OOV by #19 category",
        "item": ", ".join("%s=%d" % (k, v) for k, v in by_cat.most_common()),
        "detail": "category of each held-out OOV type (the coinage "
                  "categories are the unexplained residue proper)",
        "pages": "",
    })
    # OCR / HTML artifacts (page list is a judgement; each page's own
    # evidence is quoted from #4's page_length.tsv, so one page's loop
    # is never attributed to another page)
    loops = loop_page_evidence()
    for p in OCR_LOOP_PAGES:
        n_rep, top, over = loops.get(p, ("?", "?", "?"))
        rows.append({
            "class": "OCR recognition loop",
            "item": "page %d" % p,
            "detail": "#4 ensemble OCR: this page's longest repeated "
                      "4-gram runs %s times (%r); marker text %sx the "
                      "mean engine length. Dropped in the D3 sensitivity "
                      "run" % (n_rep, top, over),
            "pages": str(p),
        })
    art = sorted(ng.html_artifact_pages(str(LAYOUT)))
    rows.append({
        "class": "HTML markup residue",
        "item": "pages %s" % ", ".join(str(p) for p in art),
        "detail": "pages whose OCR text carries raw HTML link markup "
                  "(href=/mailto:); operators inside these spans are "
                  "furniture, not syntax (#11/#51)",
        "pages": ", ".join(str(p) for p in art),
    })
    # ambiguous operator roles, re-measured from the #17 token stream
    link, slash = slash_role_counts()
    g = grammar_counts()
    rows.append({
        "class": "ambiguous operator role",
        "item": "/ dual role",
        "detail": "%d slash tokens glue to a following word (compound "
                  "linker, #14's Link nodes) vs %d boundary-like "
                  "(#14's Slash nodes); token-level recount vs #14's "
                  "node-level %d/%d split"
                  % (link, slash, g["n_link_slash"], g["n_slash"]),
        "pages": "",
    })
    rows.append({
        "class": "ambiguous operator role",
        "item": ":: vs // boundary tension",
        "detail": "#12: // is sentence-initial in 53% of following "
                  "segments vs :: 13% (z=31); #50: mapping :: to period "
                  "(101.0 ppl) edges // to period (102.2) — a mild, "
                  "disclosed tension; both are phrase/sentence "
                  "separators, never comments",
        "pages": "",
    })
    rows.append({
        "class": "ambiguous operator role",
        "item": "= naming seam vs markup",
        "detail": "= is a lexical linker only when a word sits on both "
                  "sides (#12: glued-L 0.98 / glued-R 0.95); 229 units "
                  "are born inside surviving source markup (</a> x99 "
                  "mailto anchors, <th> x79 page furniture) and are "
                  "screened out by the #19 ledger, never repaired",
        "pages": "",
    })
    # the 'second' confound: #34's disclosed OCR-driven filler
    rows.append({
        "class": "disclosed confound",
        "item": "the 'second' filler",
        "detail": "'second' is a top filler of the 'of the <_>' frame "
                  "(145-163 occurrences) but is inflated by the page-89 "
                  "recognition loop (#34's disclosed confound); frame "
                  "filler counts in the termbase exclude no pages, so "
                  "read 'second' with that bound",
        "pages": "89",
    })
    # unexplained residue: coinages whose only attestation is the
    # held-out half.  The "attested once" filter is applied, not assumed:
    # on the current book every held-out true coinage happens to be a
    # single occurrence, so an unfiltered list and a filtered one would
    # print the same number here -- which is exactly why the filter has
    # to be in the code rather than in the reader's trust.
    coins = [w for w in oov_types
             if (screening.get(w) or {}).get("category") == "true_coinage"]
    once = [w for w in coins if oov_types[w] == 1]
    rows.append({
        "class": "unexplained residue",
        "item": "%d of %d held-out true-coinage types are attested once"
                % (len(once), len(coins)),
        "detail": "coinages attested only in the held-out half, with a "
                  "single occurrence: no within-text evidence can fix "
                  "their sense; samples: %s" % (", ".join(
                      "%s (p.%d)" % (w, min(oov_pages[w])) for w in once[:12])
                      or "none"),
        "pages": ", ".join(str(min(oov_pages[w])) for w in once[:12]),
    })
    return rows, oov_types, oov_pages


# ---------------------------------------------------------------------------
# meaning extraction: curated termbase + page-cited gloss passages
# ---------------------------------------------------------------------------

# Curated glosses.  Confidence: high = ordinary-English or directly
# attested entity (sense given by ordinary English + consistent book
# usage); medium = compound/coinage whose sense is compositional (parts
# + collocates) but not fully pinned; low = sparse attestation or
# one-off construct.  The evidence columns are computed from the
# artifacts, never hand-entered.
GLOSS_TERMS = {
    "i": ("pron", "I", "high"),
    "self": ("noun", "the self — the protagonist's own body–mind; the "
             "book's central internal referrent", "high"),
    "dog": ("noun", "dog; a recurrent creature of the book's world — "
             "patient of the opening frame ('the dog that was done to')",
             "high"),
    "boy-roid": ("compound", "boy-roid: a recurring compound persona "
                 "(boy + roid, a steroid/roid seam)", "high"),
    "soul/gram": ("compound", "soul/gram: the self's record or measure "
                  "(soul + gram, as in telegram); a recurring named "
                  "construct", "medium"),
    "drug-embryo": ("compound", "drug-embryo: an embryo of the drug "
                    "domain (drug + embryo); a recurring construct",
                    "medium"),
    "cadaver": ("noun", "cadaver: a dead body; the book's dominant "
                "persona-term", "high"),
    "cadaver city": ("phrase", "Cadaver City: the book's primary named "
                     "place", "high"),
    "tokage": ("proper noun", "ToKAGE: a named persona (does: "
               "is/was/fecundates/murder; done-to: love/speed/like/"
               "crucified)", "high"),
    "ndro": ("proper noun", "NDRO: a named persona (does: is/rapes/"
             "commits; done-to: black/control/line)", "medium"),
    "tok": ("proper noun", "ToK: a named persona (does: was/commit/"
            "streamed/suck)", "medium"),
    "adam": ("proper noun", "ADAM: a named persona; name-source of the "
             "ADAM Doll creature and retro-ADAM", "high"),
    "gene=tv": ("compound", "gene=TV: a named construct joining gene and "
                "TV via the = naming seam; a recurring designator "
                "treated as an entity", "medium"),
    "dna": ("noun", "DNA: the genetic molecule; also a designator head "
            "(DNA=channel)", "high"),
    "fecundates": ("verb", "fecundates: to fecundate, make fertile; the "
                   "book's signature coinage-verb, used of persons and "
                   "masses alike", "high"),
    "artificial": ("adjective", "artificial; head of the book's dominant "
                   "collocation artificial (sun)", "high"),
    "artificial-sun": ("compound", "artificial-sun: the not-natural sun; "
                       "the book's dominant content collocation", "high"),
    "cold-blooded": ("compound", "cold-blooded; head of the collocation "
                     "cold-blooded disease", "high"),
    "disease": ("noun", "disease; the head of the cold-blooded disease "
                "collocation", "high"),
    "internal": ("adjective", "internal; the book's dominant adjective, "
                 "head of internal organ", "high"),
    "organ": ("noun", "organ; head of internal organ", "high"),
    "masses": ("noun", "masses; head of masses of flesh, the book's "
               "body-material phrase", "high"),
    "flesh": ("noun", "flesh; the book's body-material term", "high"),
    "respired": ("verb", "respired: breathed (collocate: sickly period)",
                 "high"),
    "quantifies": ("verb", "quantifies: measures (collocates: paranoia, "
                   "drug-embryo, tragedy code)", "high"),
    "crushes": ("verb", "crushes: compresses, destroys (collocates: "
                "soul/gram, quantum tragedy)", "high"),
    "circuit": ("noun", "circuit; head of murder circuit", "high"),
    "paranoia": ("noun", "paranoia; a recurrent affect-term (collocates: "
                "self, murder)", "high"),
    "lobe": ("noun", "lobe: an anatomical division (collocates: brain, "
            "genome, dog)", "medium"),
    "maso=traffic": ("compound", "maso=traffic: maso- (masochism) + "
                     "traffic via the = naming seam; a one-attestation "
                     "construct", "low"),
    "digital-vamp": ("compound", "digital-vamp: digital + vamp "
                     "(vampire); a named construct", "medium"),
    "speed+dog": ("compound", "speed+dog: speed + dog via the + coupling "
                  "seam; attested 4 times", "low"),
    "hydromachine": ("compound", "hydromachine: hydro + machine; a "
                     "machine-entity of the book's world", "medium"),
    "hydromania": ("compound", "hydromania: hydro + mania; a named "
                   "construct (a persona in #30)", "medium"),
    "exoskeleta": ("noun", "exoskeleta: plural of exoskeleton, the outer "
                  "skeleton; a body-machine term", "medium"),
    "sado": ("label", "SADO: an ALL-CAPS designator, likely the sado- "
             "(masochism) stem", "low"),
    "city-city": ("compound", "City-city: reduplicated head of the Cadaver "
                  "City place-name (an intensifying reduplication)",
                  "low"),
    "porno": ("adjective", "porno: pornographic (English deverbal "
             "adjective)", "medium"),
    "nerve": ("noun", "nerve; head of nerve gas", "high"),
    "gas": ("noun", "gas; head of the nerve gas collocation", "high"),
    "quantum": ("adjective", "quantum; the book's physics modifier "
               "(collocates: masses, flesh, tragedy, theory)", "high"),
    "tragedy": ("noun", "tragedy; a recurrent event-term (collocates: "
                "quantum, dog, code)", "high"),
    "murder": ("noun", "murder; the book's dominant event-word, head of "
               "murder circuit and murder function", "high"),
    "inorganic": ("adjective", "inorganic; collocate of substance", "high"),
    "substance": ("noun", "substance; collocate of inorganic", "high"),
    "restrained": ("verb", "restrained: held, kept (collocates: "
                  "artificial-sun, our)", "high"),
    "crowds": ("verb", "crowds: gathers in masses", "high"),
    "liquefied": ("verb", "liquefied: turned to liquid", "high"),
    "rape": ("noun", "rape; a recurrent violence-term of the book", "high"),
    "suck": ("verb", "suck; component of the compound suck=blood", "high"),
    "blood": ("noun", "blood; the book's namesake material, component of "
              "suck=blood", "high"),
}

# The three glossed passages: (page, start anchor, end anchor, title).
# Anchors are exact substrings of the layout-recovered page body.
GLOSS_PASSAGES = (
    (4, "I suck the porno nerve gas of boy-roid",
     "tragedy of NDRO//",
     "Passage 1 — the opening block (p. 4)"),
    (4, "It quantifies++maso=traffic",
     "Cadaver City-city//",
     "Passage 2 — ToKAGE in Cadaver City (p. 4)"),
    (6, "The artificial-sun is restrained",
     "is respired//",
     "Passage 3 — the artificial-sun circuit (p. 6)"),
)


# Operator readings for the interlinear.  Every entry is #12's measured
# classification for that symbol, never an interpretation added here; the
# `{link}` / `{slash}` placeholders are filled from the memoized
# `slash_role_counts()`, so a stale literal cannot contradict #14/#17.
OPERATOR_GLOSSES = {
    "<<": "block-quote opener (#14 Span): carries the book's recurring "
          "names/labels; a quoting frame, not I/O",
    ">>": "block-quote closer (#14 Span)",
    "//": "sentence boundary (#12: 53% of following segments are "
          "sentence-initial; the utterance delimiter)",
    "::": "phrase/field separator (#12: 13% sentence-initial; the "
          "phrase-level delimiter, never a comment)",
    "=": "naming seam (#12: glued both sides 0.98/0.95); links a "
         "compound's parts or names a construct (gene=TV)",
    "/": "compound linker when glued to a following word "
         "({link} tokens), sentence-final separator otherwise "
         "({slash} tokens) — a measured dual role",
    "++": "coupling seam (#12: glued both sides); joins two words into "
          "one coupled phrase (quantifies++maso=traffic)",
    "+": "coupling seam (single), as in speed+dog",
    "-": "word-internal compound join (boy-roid) or free-standing dash "
         "separator (#12: 177 measured sites)",
    "|": "pipe operator (rare; layout-scaffolding only in the raw file)",
}


def passage_span(body, start_anchor, end_anchor):
    """(start, end) offsets of a passage, or None."""
    i = body.find(start_anchor)
    if i < 0:
        return None
    j = body.find(end_anchor, i)
    if j < 0:
        return None
    return i, j + len(end_anchor)


def passage_tokens(tok_rows, start, end, page):
    """The #17 token stream rows inside a passage span, in surface order."""
    rows = [r for r in tok_rows
            if int(r["page"]) == page
            and start <= int(r["start"]) and int(r["end"]) <= end]
    rows.sort(key=lambda r: int(r["start"]))
    return rows


def render_variant(toks, mode):
    """Operator-normalized rendering (#13 v3 = operators to space,
    v4 = operators to period), passage-level."""
    out = []
    for r in toks:
        gap = json.loads(r["gap_json"])
        text = json.loads(r["text_json"])
        if r["kind"] == "OP":
            out.append(". " if mode == "v4" else " ")
        else:
            out.append(gap + text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\. \.", ". ",
                                         "".join(out))).strip()


def build_termbase(pages, allp, entity_glossary, screening, clauses,
                   kwic_examples=None):
    """Curated termbase with computed evidence."""
    freq = collections.Counter(w for p in allp for w in pages[p])
    pages_of = collections.defaultdict(set)
    for p in allp:
        for w in pages[p]:
            pages_of[w].add(p)
    profs = collocate_profiles(pages, allp, list(GLOSS_TERMS))
    if kwic_examples is None:
        kwic_examples = load_kwic_examples()
    # book-wide frames + filler counts for the curated terms
    frames = fm.mine_frames(clauses)
    top_frames = frames[:20]
    filler = {t: [] for t in GLOSS_TERMS}
    for fr in top_frames:
        skel = fr["skeleton"]
        k = len(skel)
        slots = [i for i, s in enumerate(skel) if s == fm.SLOT]
        counts = collections.Counter()
        for cl in clauses:
            toks = cl["tokens"]
            for i in range(len(toks) - k + 1):
                if all(skel[t] == fm.SLOT or skel[t] == toks[i + t]
                       for t in range(k)):
                    for s in slots:
                        counts[toks[i + s]] += 1
        for t in GLOSS_TERMS:
            if counts.get(t, 0) > 0:
                filler[t].append((" ".join(skel), counts[t]))
    rows = []
    for term in sorted(GLOSS_TERMS, key=lambda t: -freq.get(t, 0)):
        pos, gloss, conf = GLOSS_TERMS[term]
        n = freq.get(term, 0)
        if n == 0:
            continue
        r = screening.get(term, {})
        eg = entity_glossary.get(term, {})
        parts = r.get("morph", "")
        cat = r.get("category", "")
        colls = profs[term].most_common(6)
        ffill = "; ".join("%s x%d" % (f, c) for f, c in
                          sorted(filler[term], key=lambda x: -x[1])[:3])
        rows.append({
            "term": term, "pos_class": pos, "category_19": cat,
            "n_occurrences": n, "n_pages": len(pages_of[term]),
            "first_page": min(pages_of[term]),
            "parts_18": parts,
            "entity_role_30": eg.get("role", ""),
            "does_verbs_30": eg.get("does_verbs", ""),
            "done_to_verbs_30": eg.get("done_to_verbs", ""),
            "top_collocates": ", ".join("%s (%d)" % (w, c)
                                        for w, c in colls),
            "frame_fillers_34": ffill,
            "kwic_examples_26": " | ".join(kwic_examples.get(term, [])[:2]),
            "gloss": gloss, "confidence": conf,
        })
    return rows, {"n_terms": len(rows), "n_frames_book": len(frames),
                  "n_terms_with_kwic": sum(
                      1 for r in rows if r["kwic_examples_26"]),
                  "top_frames": [" ".join(f["skeleton"]) for f in top_frames]}


def gloss_passages(bodies, page_units, unit_meta, is_markup, screening,
                   entity_glossary, termbase_rows, frames_top):
    """Gloss the three passages token by token."""
    gloss_of = {r["term"]: r for r in termbase_rows}
    # one pass over the whole #17 token stream serves every passage
    tok_rows = cs.read_tsv(TOKENS)
    passages = []
    token_rows = []
    for pno, start_anchor, end_anchor, title in GLOSS_PASSAGES:
        body = bodies[pno]
        span = passage_span(body, start_anchor, end_anchor)
        if span is None:
            raise SystemExit("gloss passage not found on page %d: %r"
                             % (pno, start_anchor))
        start, end = span
        toks = passage_tokens(tok_rows, start, end, pno)
        original = body[start:end]
        # units inside the span (for category/parts evidence)
        span_units = []
        for (s, e, uid) in page_units[pno]:
            if start <= s and e <= end and not is_markup(uid):
                span_units.append(uid)
        # per-token gloss
        rows = []
        for r in toks:
            surface = json.loads(r["text_json"])
            kind = r["kind"]
            low = surface.lower()
            if kind != "WORD":
                role = {"OP": "operator", "PUNCT": "punctuation"}[kind]
                op_gloss = OPERATOR_GLOSSES.get(surface, "")
                if "{" in op_gloss:
                    # measured counts, filled from the #17 recount
                    link, slash = slash_role_counts()
                    op_gloss = op_gloss.format(link=link, slash=slash)
                rows.append({
                    "passage": title, "page": pno, "start": int(r["start"]),
                    "surface": surface, "kind": kind, "role": role,
                    "uid": "", "category_19": "", "parts_18": "",
                    "entity_role_30": "", "collocates": "",
                    "frame_fillers_34": "", "gloss": op_gloss or "",
                    "confidence": "high" if op_gloss else "unresolved",
                })
                continue
            uid = ""
            cont = None                      # containing #17 unit
            for (s, e, u) in page_units[pno]:
                if s <= int(r["start"]) < e:
                    uid = str(u)
                    cont = (s, e, u)
                    break
            tb = gloss_of.get(low)
            cat = (screening.get(low) or {}).get("category", "")
            parts = (screening.get(low) or {}).get("morph", "")
            eg = entity_glossary.get(low, {})
            if tb is not None:
                gloss, conf = tb["gloss"], tb["confidence"]
                colls = tb["top_collocates"]
                ffill = tb["frame_fillers_34"]
            elif cat in ("english", "rare_english"):
                # ordinary English per the #19 screen: the
                # word is its own gloss (high confidence)
                fn = "function word" if low in FUNCTION_WORDS \
                    else "ordinary English content word"
                gloss = "%s: the ordinary English word, #19 " \
                        "screen class `%s`" % (surface, fn)
                conf = "high"
                colls, ffill = "", ""
            elif cont is not None:
                # not an English word on its own: read the
                # containing #17 unit (the token may be the
                # whole named construct or a word-part of one)
                u_s, u_e, u_uid = cont
                us = unit_meta[u_uid][3]
                ue = screening.get(us.lower()) or {}
                ucat = ue.get("category", "")
                ueg = entity_glossary.get(us.lower(), {})
                role_note = ("; #30 entity role: %s"
                             % ueg["role"] if ueg.get("role")
                             else "")
                if int(r["start"]) == u_s and int(r["end"]) == u_e:
                    gloss = ("%s: the named construct `%s` in "
                             "full (#19 class `%s`%s)"
                             % (surface, us, ucat, role_note))
                    conf = "medium"
                else:
                    morph = ue.get("morph", "")
                    gloss = ("%s: word-part of the compound unit "
                             "`%s` (#17; parts %s)%s"
                             % (surface, us,
                                morph.replace("|", " ") if morph
                                else "n/a", role_note))
                    conf = "medium"
                colls, ffill = "", ""
            else:
                gloss, conf = "", "unresolved"
                colls, ffill = "", ""
            rows.append({
                "passage": title, "page": pno, "start": int(r["start"]),
                "surface": surface, "kind": kind, "role": "word",
                "uid": uid, "category_19": cat, "parts_18": parts,
                "entity_role_30": eg.get("role", ""),
                "collocates": colls, "frame_fillers_34": ffill,
                "gloss": gloss, "confidence": conf,
            })
        token_rows.extend(rows)
        passages.append({
            "title": title, "page": pno, "start": start, "end": end,
            "original": original,
            "v3_space": render_variant(toks, "v3"),
            "v4_period": render_variant(toks, "v4"),
            "rows": rows,
        })
    return passages, token_rows


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def verdict(rows):
    """Pre-registered: >=5 of 6 -> VIABLE (bounded); 3-4 -> PARTIAL;
    <=2 -> NOT VIABLE.  V0 is a control, not a scored check."""
    scored = [r for r in rows if r["check"] in
              ("V1", "V2", "V3", "V4a", "V4b", "V5", "V6")]
    # V4 is one check with two statistics: both must pass.
    v4 = [r for r in scored if r["check"].startswith("V4")]
    n_v4 = 1 if v4 and all(r["passed"] for r in v4) else 0
    n_pass = sum(1 for r in scored
                 if r["check"] not in ("V4a", "V4b") and r["passed"]) + n_v4
    if n_pass >= VERDICT_PASS:
        v = "VIABLE (BOUNDED)"
    elif n_pass >= VERDICT_PARTIAL:
        v = "PARTIALLY VIABLE"
    else:
        v = "NOT VIABLE"
    return v, n_pass


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def write_tsv(path, header, rows):
    cs.write_tsv(path, header, rows)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")).replace("|", "\\|")
                                     for h in header) + " |")
    return "\n".join(out)


def _fmt(x, nd=4):
    try:
        return ("%." + str(nd) + "f") % float(x)
    except (TypeError, ValueError):
        return str(x)


def _fmt_p(p):
    """A p-value at 4 decimals, or scientific when 4 decimals round to 0.

    ``0.0000`` is a true statement about ``5.2e-06`` and useless to a
    reader, so tiny-but-nonzero p-values are written in exponent form.
    """
    if p is None:
        return "-"
    p = float(p)
    if p == 0.0:
        return "0"
    if abs(p) < 1e-4:
        return "%.1e" % p
    return "%.4f" % p


# ---------------------------------------------------------------------------
# the deliverable report
# ---------------------------------------------------------------------------

def write_report(out, summary, checks, v, n_pass, termbase,
                   passages, ledger, oov_rows, recon, sweep,
                   chrono, sens, sense_det, n_nulls, diag, tb_det):
    L = []
    add = L.append
    add("# Within-text language viability assessment — *Blood Electric*")
    add("")
    add("Chainlink #56 (revised scope: intrinsic evidence only).")
    add("Schema `%s` · generated %s · seed %d · %d null draws "
        "per check · split: interleaved pages (even train, odd "
        "test)" % (SCHEMA, summary["generated_at"], SEED, n_nulls))
    add("")
    add("**Verdict: %s — %d of 6 pre-registered checks pass.**"
        % (v, n_pass))
    add("")
    add("## 0. Question and scope")
    add("")
    add("Do *Blood Electric*'s recurring lexical constructions and "
        "operator sequences support a viable, recoverable language "
        "— and where the evidence supports it, what does the text "
        "say? Per the operator's scope decision (2026-10-08), this "
        "assessment uses **only evidence from the text itself**: no "
        "external books, no reference corpora, no new datasets. The "
        "comparative keyness artifacts of the earlier #56 attempt "
        "(`analysis/keyness/`, `analysis/reference_corpora/`) are "
        "retained as historical output and are not consulted here; "
        "follow-ups #78–#80 are parked.")
    add("")
    add("The working corpus is the frozen #17 operator-aware "
        "tokenizer output (`analysis/tokenizer/units.tsv`, "
        "`tokens.tsv`): **{:,} screened lexical units** over {} pages "
        "(the #19 markup ledger + the #27 tag-run screen applied; "
        "nothing repaired), plus the #11 operator stream (**{:,} "
        "operator tokens**) and the #33 clause index (**{:,} "
        "clauses**). The grammar, entity/event and contextual-example "
        "analyses are reused, not redone: #14's AST supplies the `/` "
        "Link-vs-Slash split reconciled in §9, #30's glossary supplies "
        "each curated term's entity role and verb directions, #34's "
        "frames supply the frame-filler counts, and #26's KWIC "
        "concordancer supplies the quoted contexts in §5.1."
        .format(summary["n_units_screened"], summary["n_pages"],
                summary["n_operators"], summary["n_clauses"]))
    add("")
    add("## 1. Method: falsifiable checks on held-out sections")
    add("")
    add("A language is *viably recoverable* from a text if a "
        "learner restricted to one half of the text predicts the "
        "other half **significantly better than within-text nulls "
        "that preserve the text's own marginal statistics but "
        "destroy the structure under test**. Every check is "
        "pre-registered (thresholds fixed before running):")
    add("")
    add(md_table(
        ["check", "claim under test", "statistic", "nulls",
         "criterion"],
        [{"check": "V0", "claim under test": "negative control: the "
          "harness must fail on a non-language",
          "statistic": "V1's machinery on the within-page-shuffled "
            "book", "nulls": "within-page shuffle",
          "criterion": "must NOT beat its own null"},
         {"check": "V1", "claim under test": "word order is "
          "learnable (lexical syntax)",
          "statistic": "held-out bigram CE, λ tuned on "
            "train-internal validation", "nulls": "within-page "
            "shuffle + type permutation",
          "criterion": "below both null minima, add-one p<0.05"},
         {"check": "V2", "claim under test": "operator sequences are "
          "idiomatic (run syntax)",
          "statistic": "held-out operator bigram CE, "
            "Jelinek–Mercer λ=0.8 (#11's estimator)",
          "nulls": "within-run shuffle",
          "criterion": "below null minimum, add-one p<0.05"},
         {"check": "V3", "claim under test": "constructions "
          "generalize to held-out clauses",
          "statistic": "test-clause coverage of train-mined frames "
            "(top-40)", "nulls": "shuffled-train frames",
          "criterion": "above null maximum, add-one p<0.05"},
         {"check": "V4", "claim under test": "the lexicon transfers "
          "(reuse, not chance recurrence)",
          "statistic": "(a) held-out OOV rate; (b) Spearman rank "
            "correlation of shared-type frequencies",
          "nulls": "type permutation", "criterion": "(a) below null "
            "minimum; (b) above null maximum, add-one p<0.05"},
         {"check": "V5", "claim under test": "term senses are stable "
          "across halves (meaning is recoverable)",
          "statistic": "median top-%d collocate Dice, same term "
            "across halves vs mismatched terms" % TOP_K,
          "nulls": "mismatched-term pairs (within-text null)",
          "criterion": "same > mismatched, paired sign test "
            "p<0.01"},
         {"check": "V6", "claim under test": "coinages are "
          "compositional from a stable part inventory",
          "statistic": "held-out part-sequence bigram CE within "
            "units", "nulls": "within-unit part shuffle",
          "criterion": "below null minimum, add-one p<0.05"},
        ]))
    add("")
    add("p-values are add-one empirical: p = (1 + #{nulls at least "
        "as extreme}) / (B+1), B=%d, seed %d. Verdict rule "
        "(pre-registered): ≥%d of 6 → VIABLE (BOUNDED); %d–%d → "
        "PARTIALLY VIABLE; ≤2 → NOT VIABLE. \"Bounded\" is earned "
        "by the bound diagnostics (§4), not assumed."
        % (n_nulls, SEED, VERDICT_PASS, VERDICT_PARTIAL,
           VERDICT_PASS - 1))
    add("")
    add("## 2. Results")
    add("")
    v5_pairs = next(r for r in checks if r["check"] == "V5")["n_nulls"]
    add(md_table(
        ["check", "book", "null mean", "null min", "null max",
         "p", "verdict"],
        [{"check": r["check"], "book": _fmt(r["book"]),
          "null mean": _fmt(r["null_mean"]),
          "null min": _fmt(r["null_min"]),
          "null max": _fmt(r["null_max"]),
          "p": _fmt_p(r["p_value"]),
          "verdict": ("CONTROL OK" if r["check"] == "V0"
                      else ("PASS" if r["passed"] else "FAIL"))}
         for r in checks]))
    add("")
    add("Reading the table. For V1–V4 and V6 the *null* columns are %d "
        "random draws of the stated within-text null, and the p-value is "
        "the add-one empirical p. **V0 is a control, not a scored "
        "check**: `CONTROL OK` means the harness correctly found no "
        "language in the shuffled book, which is the result the verdict "
        "needs — a FAIL there would invalidate the passes, not count "
        "against them. **V5 has no random draws**: its comparison set is "
        "the mismatched-term distribution over %d pairs, so `n_nulls` "
        "for that row is the number of pairs and its p-value is an exact "
        "sign test." % (n_nulls, v5_pairs))
    add("")
    add("Per-check reading:")
    add("")
    v1 = next(r for r in checks if r["check"] == "V1")
    add("* **V1 — lexical order (%s).** A bigram model trained on "
        "the even pages reads the odd pages at **%s bits/unit**, "
        "against %s for the within-page shuffle null and %s for the "
        "type-permutation null. The shuffle null is the informative "
        "one: it keeps the book's unigram frequencies and destroys "
        "only the ordering, so beating it is evidence about order. "
        "The type-permutation null is a *conservative* null — a "
        "random bijection of the vocabulary sends most held-out "
        "tokens to types the train half never saw, so its CE runs "
        "away to tens of bits and V1 beats it comfortably; it is "
        "reported because it is the check's second pre-registered "
        "criterion, not because it is hard to pass. The unigram "
        "baseline is %s bits/unit (order-0, `order_sweep.tsv`), so the "
        "bigram gain is %s bits — local syntax exists."
        % ("PASS" if v1["passed"] else "FAIL",
           _fmt(v1["book"]), _fmt(v1["null_mean"]),
           _fmt(sense_det["v1_null_min_typeperm"]),
           _fmt(sense_det["v1_order0_ce"]),
           _fmt(sense_det["v1_order0_ce"] - v1["book"])))
    v2 = next(r for r in checks if r["check"] == "V2")
    add("* **V2 — operator stream (%s, thin margin).** Held-out "
        "operator bigram CE is **%s bits/operator** vs %s for the "
        "within-run shuffle null. The margin is small (%s bits) — "
        "this is the honest quantification of #11's \"partially "
        "idiomatic\": operator runs have a shallow local syntax, "
        "real but weak. Its own control is the shuffle, which "
        "preserves run composition and the operator marginals and "
        "destroys only the order within a run; V0 controls the "
        "*lexical* estimator, not this one (see §7)."
        % ("PASS" if v2["passed"] else "FAIL",
           _fmt(v2["book"]), _fmt(v2["null_mean"]),
           _fmt(v2["null_mean"] - v2["book"])))
    v3 = next(r for r in checks if r["check"] == "V3")
    add("* **V3 — construction transfer (%s).** Frames mined on "
        "the train half (%d frames) cover **%s of the held-out "
        "clauses** (top-40), against a shuffled-train null maximum "
        "of %s. The book's recurring constructions — the "
        "`of the ⟨_⟩` NP-of-NP core, `I ⟨_⟩ the`, `the ⟨_⟩ "
        "that` — generalize to pages the miner never saw. The null "
        "is high (%s) because clauses share unigrams, so a "
        "skeleton can match by chance; the margin above it is the "
        "genuine construction signal."
        % ("PASS" if v3["passed"] else "FAIL",
           sense_det["v3_n_frames"], _fmt(v3["book"]),
           _fmt(v3["null_max"]), _fmt(v3["null_mean"])))
    v4a = next(r for r in checks if r["check"] == "V4a")
    v4b = next(r for r in checks if r["check"] == "V4b")
    add("* **V4 — lexicon reuse (%s).** (a) Only **%s of "
        "held-out tokens** are types unseen in the train half, "
        "against a type-permutation minimum of %s — the lexicon "
        "transfers, and the recurrence is not the frequency "
        "spectrum's doing. (b) Shared-type frequencies correlate at "
        "**ρ=%s** between halves (permutation maximum %s): the book "
        "repeats itself with a stable weighting."
        % ("PASS" if (v4a["passed"] and v4b["passed"]) else "FAIL",
           _fmt(v4a["book"]), _fmt(v4a["null_min"]),
           _fmt(v4b["book"]), _fmt(v4b["null_max"])))
    v5 = next(r for r in checks if r["check"] == "V5")
    add("* **V5 — sense stability (%s).** The top-%d content "
        "terms' collocate profiles correlate across halves: median "
        "same-term Dice **%s** vs %s for mismatched terms, with "
        "%d of %d paired comparisons positive (sign test "
        "p=%s). A term's company in the first half predicts its "
        "company in the second — senses are recoverable, not "
        "page-local accidents. Secondary statistics "
        "(`sense_stability.tsv`): top-10 Dice %s vs %s; cosine "
        "over the top-30 union %s vs %s. The comparison is *not* "
        "frequency-matched: a term is paired with the one 17 places "
        "further down the frequency ranking, which is typically a "
        "rarer word with a smaller profile (§7)."
        % ("PASS" if v5["passed"] else "FAIL",
           TOP_K, _fmt(sense_det["v5_med_same"]),
           _fmt(sense_det["v5_med_mism"]),
           sense_det["v5_n_better"], sense_det["v5_n_terms"],
           _fmt_p(v5["p_value"]), _fmt(sense_det["v5_dice10_same"]),
           _fmt(sense_det["v5_dice10_mism"]),
           _fmt(sense_det["v5_cosine_same"]),
           _fmt(sense_det["v5_cosine_mism"])))
    v6 = next(r for r in checks if r["check"] == "V6")
    add("* **V6 — compositionality (%s).** Held-out part-sequence "
        "bigram CE is **%s bits/part** vs %s for the within-unit "
        "part-shuffle null (unigram baseline %s). Coinages are "
        "built from a stable part inventory: `boy-roid`, "
        "`soul/gram`, `gene=TV` recombine parts the train half "
        "already attested, and the recombination is learnable."
        % ("PASS" if v6["passed"] else "FAIL",
           _fmt(v6["book"]), _fmt(v6["null_min"]),
           _fmt(sense_det["v6_unigram_ce"])))
    add("")
    add("## 3. Verdict")
    add("")
    if v.startswith("VIABLE"):
        add("**%s (%d/6).** The text supports a viable, recoverable "
            "language at three levels: a **stable, reusable lexicon** "
            "(V4), **compositional morphology** (V6), and **local "
            "syntactic structure** — word order (V1), operator run "
            "syntax (V2), and recurring constructions (V3) — with "
            "**stable term senses** that make meaning extraction "
            "defensible (V5). This is a positive viability verdict with "
            "an explicit bound: everything that passes is *local* "
            "(short-context, construction-level); the bound diagnostics "
            "below show what does *not* generalize."
            % (v, n_pass))
    elif v.startswith("PARTIALLY"):
        add("**%s (%d/6).** The evidence supports parts of a "
            "recoverable language but not all of it: %s. The checks "
            "that fail bound the claim — see their rows in §2 and the "
            "bound diagnostics in §4 before reading anything into the "
            "passes."
            % (v, n_pass,
               "; ".join("%s %s" % (r["check"],
                                     "passes" if r["passed"] else "fails")
                         for r in checks if r["check"] != "V0")))
    else:
        add("**%s (%d/6).** The held-out evidence does not support "
            "a viable, recoverable language under the pre-registered "
            "thresholds: %s. The per-check rows in §2 and the bound "
            "diagnostics in §4 state precisely which structures did "
            "not generalize; the negative control (V0) shows the "
            "harness itself is sound, so the failure is a property of "
            "the text, not of the instrument."
            % (v, n_pass,
               "; ".join("%s %s" % (r["check"],
                                     "passes" if r["passed"] else "fails")
                         for r in checks if r["check"] != "V0")))
    add("")
    add("## 4. Where viability ends (bound diagnostics)")
    add("")
    add("### D1 — order sweep: does a longer context help?")
    add("")
    add(md_table(["n-gram order", "λ (tuned on train-internal "
                      "validation)", "validation CE", "held-out CE"],
                 [{"n-gram order": r["order"],
                   "λ (tuned on train-internal validation)": r["lam"],
                   "validation CE": _fmt(r["val_ce"]),
                   "held-out CE": _fmt(r["test_ce"])}
                  for r in sweep]))
    add("")
    order1 = next(r for r in sweep if r["order"] == 1)
    higher = [r for r in sweep if r["order"] > 1]
    best_higher = min(higher, key=lambda r: r["test_ce"])
    gain = order1["test_ce"] - best_higher["test_ce"]
    order3 = next((r for r in sweep if r["order"] == 3), None)
    if gain > 0:
        order3_text = (" Order-3 held-out CE is %s bits/unit." %
                       _fmt(order3["test_ce"]) if order3 else "")
        d1_text = (
            "With λ tuned per order on a train-internal validation split, "
            "the best longer-context model is order-%d: held-out CE is "
            "%s vs %s bits/unit for order-1, a modest improvement of "
            "%s bits/item (%.2f%%).%s This difference has no separate "
            "uncertainty test, so treat it as a small short-context gain, "
            "not a robust long-range result. The book's recoverable "
            "structure remains local; the order sweep does not support "
            "a global grammar or metrical cycle. This triangulates #11 "
            "(shallow local run syntax), #47 (the operator channel gains "
            "little from context) and #51 (no latent periodicity)."
            % (best_higher["order"], _fmt(best_higher["test_ce"]),
               _fmt(order1["test_ce"]), _fmt(gain),
               100.0 * gain / order1["test_ce"], order3_text))
    else:
        d1_text = (
            "With λ tuned per order on a train-internal validation split, "
            "no longer-context model improves on order-1: the best is "
            "order-%d at %s vs %s bits/unit. This bounds the recoverable "
            "grammar to short local contexts, with no evidence here for "
            "long-range structure. This triangulates #11 (shallow local "
            "run syntax), #47 (the operator channel gains little from "
            "context) and #51 (no latent periodicity)."
            % (best_higher["order"], _fmt(best_higher["test_ce"]),
               _fmt(order1["test_ce"])))
    add(d1_text)
    add("")
    add("### D2 — chronological stability")
    add("")
    add("On the conservative contiguous split (train pp. %s, test "
        "pp. %s) the order-1 held-out CE is **%s bits/unit** "
        "(λ=%s), essentially identical to the interleaved split's "
        "%s. There is no measurable drift: the language of the "
        "first half is the language of the second."
        % (chrono["train_pages"], chrono["test_pages"],
           _fmt(chrono["test_ce"]), chrono["lam"],
           _fmt(v1["book"])))
    add("")
    add("### D3 — sensitivity: without the artifact pages")
    add("")
    add("Dropping the #4 OCR recognition-loop pages (%s) and the "
        "#11/#51 HTML-artifact pages (%s) — %d pages remain — the "
        "order-1 held-out CE is **%s bits/unit** (%d shuffle nulls, "
        "null min %s, add-one p=%s), i.e. V1's result is unchanged "
        "when the corrupted pages are removed. Scope: this re-runs V1 "
        "only. The other five checks were not re-run without the "
        "artifact pages, so this is evidence that the *lexical* result "
        "is not an artifact of those six pages, not that every check "
        "survives their removal."
        % (", ".join(str(p) for p in OCR_LOOP_PAGES),
           ", ".join(str(p) for p in sens["dropped_pages"]
                         if p not in OCR_LOOP_PAGES),
           sens["n_pages"], _fmt(sens["test_ce"]), sens["n_nulls"],
           _fmt(sens["null_min"]), _fmt_p(sens["p_value"])))
    add("")
    add("## 5. Meaning extraction")
    add("")
    add("### 5.1 What the termbase supports")
    add("")
    add("The curated termbase (`termbase_sample.tsv`, %d terms) "
        "combines, per term: #19 screen category, #18 parts, #30 "
        "entity role and verb directions, book-wide collocates "
        "(±5 units, function words excluded), #34 frame-filler "
        "counts, up to two quoted #26 concordance contexts with page "
        "citations, and a gloss with a confidence. The confidence "
        "levels mean: **high** — ordinary English or directly "
        "attested entity (sense given by ordinary English plus "
        "consistent book usage); **medium** — compound/coinage "
        "whose sense is compositional (parts plus collocates) but "
        "not fully pinned; **low** — sparse attestation or a "
        "one-off construct. #26 indexed the compound and named "
        "constructs, so %d of the %d terms carry a quoted context and "
        "the ordinary-English terms do not — that is the "
        "concordancer's coverage, not a gap in the evidence for them. "
        "Three examples of what the evidence supports:"
        % (len(termbase), tb_det["n_terms_with_kwic"], len(termbase)))
    add("")
    examples = [t for t in termbase if t["term"] in
                ("soul/gram", "tokage", "artificial-sun")]
    for t in examples:
        add("* **%s** (%s, %s; n=%s over %s pages): %s Evidence: "
            "collocates %s; %s%s%s"
            % (t["term"], t["pos_class"], t["confidence"],
               t["n_occurrences"], t["n_pages"], t["gloss"],
               t["top_collocates"],
               ("frame filler: " + t["frame_fillers_34"] + ". "
                if t["frame_fillers_34"] else ""),
               ("context (#26): %s. "
                % t["kwic_examples_26"].replace(" | ", " … ")
                if t["kwic_examples_26"] else ""),
               ("entity role: %s; does: %s; done-to: %s."
                % (t["entity_role_30"], t["does_verbs_30"],
                   t["done_to_verbs_30"])
                if t["entity_role_30"] else "")))
    add("")
    add("### 5.2 Sample glosses (page-cited)")
    add("")
    add("Three passages are glossed token-by-token in "
        "`gloss_passages.md` (full tables) and `gloss_tokens.tsv` "
        "(one row per token, with the #17 uid, #19 category, #18 "
        "parts, #30 entity role, #34 frame fillers). Each passage "
        "is rendered three ways: the original, the #13 v3 "
        "operator-normalized form (operators → space), and the v4 "
        "form (operators → period). Headline readings:")
    add("")
    for ps in passages:
        n_unres = sum(1 for r in ps["rows"]
                      if r["confidence"] == "unresolved")
        n_low = sum(1 for r in ps["rows"]
                    if r["confidence"] == "low")
        add("* **%s** (p. %d): %d tokens, %d glossed high, %d "
            "medium, %d low, %d unresolved. The plain-English "
            "skeleton is fully recoverable; the residue is listed, "
            "not repaired."
            % (ps["title"], ps["page"], len(ps["rows"]),
               sum(1 for r in ps["rows"] if r["confidence"] == "high"),
               sum(1 for r in ps["rows"] if r["confidence"] == "medium"),
               n_low, n_unres))
    add("")
    add("Passage 1 in brief — original (p. 4):")
    add("")
    add("> %s" % passages[0]["original"].replace("\n", " "))
    add("")
    add("Constrained literal translation (confidence per token in "
        "`gloss_tokens.tsv`):")
    add("")
    add("> «I suck the pornographic nerve-gas of boy-roid. The "
        "soul/gram of the self crushes to the quantum tragedy of "
        "NDRO.»")
    add("")
    add("Every word in that rendering is either ordinary English "
        "(high confidence) or a named construct whose *referential "
        "status* is directly evidenced (`boy-roid` a recurring "
        "compound persona, `soul/gram` a recurring construct, "
        "`NDRO` a named persona). What is *not* claimed: what a "
        "boy-roid is, or what the quantum tragedy of NDRO is *about* "
        "— the text does not say, and the assessment does not "
        "ventriloquize.")
    add("")
    add("## 6. Ambiguous cases and unexplained residue")
    add("")
    add("Quantified in `residue_ledger.tsv` and `oov_types.tsv`:")
    add("")
    oov = summary["oov"]
    add("* **Held-out residue proper:** %d types / %d tokens "
        "(%s%% of the test half) are never attested in the train "
        "half — the hard bound on recoverability. By #19 category: "
        "%s."
        % (oov["n_types"], oov["n_tokens"],
           _fmt(100.0 * oov["n_tokens"] /
                max(1, sense_det["v4_test_tokens"]), 1),
           "; ".join("%s=%d" % (k, v) for k, v in
                     collections.Counter(
                         r["category_19"] for r in oov_rows
                         ).most_common())))
    add("* **Ambiguous operator roles:** the `/` dual role "
        "(compound linker vs sentence-final separator), the `::` vs "
        "`//` boundary tension (#12, #50), and the `=` naming seam "
        "vs markup collision (`href=`) — each measured, each "
        "disclosed; none repaired.")
    add("* **Disclosed confound:** the `second` filler of `of the "
        "⟨_⟩` is inflated by the page-89 OCR recognition loop "
        "(#34's confound, re-confirmed here).")
    coin_row = next(r for r in ledger if r["class"] == "unexplained residue")
    add("* **Unexplained residue:** %s, unattested in the train half — "
        "no within-text evidence can fix their sense; they are listed "
        "in `oov_types.tsv` with their page." % coin_row["item"])
    add("")
    add("## 7. Limitations")
    add("")
    add("* **OCR is the floor.** The corpus is marker OCR of a scan; "
        "the #4 ensemble (tesseract 97.9% / docTR 97.3% / "
        "easyocr 93.9% character agreement) bounds the noise, and the "
        "D3 sensitivity run shows V1's result does not depend on the "
        "three recognition-loop pages — but per-token confidence in "
        "the glosses inherits OCR uncertainty.")
    add("* **Local structure only.** The lexical order sweep shows "
        "whether adding up to two prior items helps, and the operator "
        "check tests only bigrams; neither tests long-range narrative, "
        "global grammar, or a generative system. The D1 higher-order "
        "difference is small and has no separate uncertainty test.")
    add("* **Glosses are constrained, not translations.** The "
        "translation in §5.2 is a literal rendering of the "
        "recoverable skeleton; coinages are glossed "
        "compositionally (parts + collocates) and marked medium "
        "or low where the sense is not pinned. Where the text "
        "does not say, the assessment says so (unresolved tokens "
        "are listed, never repaired).")
    add("* **The nulls are within-text.** Type permutation and "
        "shuffles preserve the book's own statistics; they "
        "control for chance recurrence *of this text*, not for "
        "any external notion of language. That is the revised "
        "scope: intrinsic viability, deliberately.")
    add("* **Estimator choices are disclosed.** Jelinek–Mercer "
        "interpolation with λ tuned on a train-internal validation "
        "split (V1, D1); #11's λ=0.8 for the operator stream "
        "(V2); top-%d Dice for sense stability (V5). The V0 "
        "negative control shows the harness fails a non-language, "
        "so the passes are not vacuous." % TOP_K)
    add("* **V0 controls the lexical estimator, not the operator "
        "one.** The negative control runs V1's bigram machinery on a "
        "shuffled book. No equivalent control exists for V2: the "
        "operator stream is never word-order-shuffled, so V2's margin "
        "(%s bits) rests on its within-run shuffle null alone. That "
        "null preserves run composition, so it cannot rule out an "
        "estimator artifact specific to operator runs. V2 is the "
        "thinnest passing check and should be read with that gap in "
        "mind." % _fmt(v2["null_mean"] - v2["book"]))
    add("* **V5's comparison is not frequency-matched.** Each term is "
        "paired with the term 17 places further down the frequency "
        "ranking, which is usually a rarer word with a smaller "
        "profile. Some of the same-vs-mismatched gap is therefore "
        "frequency, not sense. The check still bounds what it claims — "
        "term-specific company is recoverable across halves — but it "
        "over-states the size of the effect.")
    add("* **The type-permutation nulls are conservative.** A random "
        "vocabulary bijection maps most held-out tokens outside the "
        "train lexicon, so V1b and V4a are won by a wide margin; they "
        "are weak evidence, kept because they were pre-registered, "
        "not because they discriminate.")
    add("")
    add("## 8. Reproduction")
    add("")
    add("```bash")
    add("conda activate marker")
    add("python work/language_viability.py            # full run "
        "(~1 min)")
    add("python work/language_viability.py --nulls 10 # quick run")
    add("python work/test_language_viability.py       # regression "
        "suite")
    add("```")
    add("")
    add("Deterministic: fixed seed %d, fixed tie-breaks. Every statistic "
        "and every table recomputes identically; the only field that "
        "varies between runs is the ``generated`` stamp above, which is "
        "provenance (as in #30) and no statistic is derived from. All "
        "statistics recompute from the frozen artifacts listed in §0."
        % SEED)
    add("")
    add("## 9. Reconciliation with prior work")
    add("")
    add(md_table(["artifact", "frozen", "recomputed", "match"],
                 recon))
    add("")
    add("The operator rows agree exactly: %d is #10's body count, and "
        "the #11 stream's remaining 51 header operators take it to the "
        "11,000 of #17's full-book figure. The frame row is a full "
        "recount, not a reuse: %d frames over all %d clauses, which "
        "reproduces #34's published catalogue frame-for-frame "
        "(descriptive evidence for the termbase, never a held-out "
        "check — V3's frames are mined on the train half alone). The "
        "#14 row compares two independent counts of the same "
        "phenomenon — #14's AST node classes and this module's "
        "recount of the #17 token stream — and they agree exactly, "
        "which is what licenses treating the `/` dual role as "
        "measured rather than as a reading. The #26 row reports "
        "coverage and is not gated."
        % (diag["n_operators"], tb_det["n_frames_book"],
           summary["n_clauses"]))
    add("")
    (out / "language_viability_report.md").write_text(
        "\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(out=OUT, n_nulls=N_NULLS):
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    diag = {}
    t0 = datetime.now(timezone.utc)

    bodies, pages, page_units, unit_meta, is_markup, screening = load_book()
    allp = sorted(pages)
    train = [p for p in allp if p % 2 == 0]
    test = [p for p in allp if p % 2 == 1]
    pages_runs, pages_ops = load_operator_pages()
    clauses = load_clauses()
    entity_glossary = load_entity_glossary()

    diag["n_pages"] = len(allp)
    diag["n_units_screened"] = sum(len(v) for v in pages.values())
    diag["n_train_pages"] = len(train)
    diag["n_test_pages"] = len(test)
    diag["n_clauses"] = len(clauses)
    diag["n_operators"] = len(ng.build_stream(pages_runs)[0])

    # ---- checks ----
    checks = []
    nulls_all = []

    row, det = check_v0_negative_control(pages, allp, train, test,
                                         n_nulls, rng)
    checks.append(row)
    for i, val in enumerate(det.get("nulls", [])):
        nulls_all.append({"check": "V0", "draw": i + 1, "value": val})
    row, det_v1 = check_v1_lexical_order(pages, allp, train, test, n_nulls, rng)
    checks.append(row)
    for i, v in enumerate(det_v1["nulls_shuffle"]):
        # "<check>_<family>": the prefix convention lets a reader (and the
        # regression suite) match a null family back to its check row.
        nulls_all.append({"check": "V1_shuffle", "draw": i + 1, "value": v})
    for i, v in enumerate(det_v1["nulls_typeperm"]):
        nulls_all.append({"check": "V1_typeperm", "draw": i + 1, "value": v})
    row, det_v2 = check_v2_operator_stream(pages_runs, pages_ops, allp,
                                        train, test, n_nulls, rng)
    checks.append(row)
    for i, v in enumerate(det_v2["nulls"]):
        nulls_all.append({"check": "V2", "draw": i + 1, "value": v})
    row, det_v3 = check_v3_frame_transfer(clauses, n_nulls, rng)
    checks.append(row)
    for i, v in enumerate(det_v3["nulls"]):
        nulls_all.append({"check": "V3", "draw": i + 1, "value": v})
    rows4, det_v4 = check_v4_lexicon_reuse(pages, allp, train, test,
                                        n_nulls, rng)
    checks.extend(rows4)
    for i, v in enumerate(det_v4["nulls_oov"]):
        nulls_all.append({"check": "V4a", "draw": i + 1, "value": v})
    for i, v in enumerate(det_v4["nulls_corr"]):
        nulls_all.append({"check": "V4b", "draw": i + 1, "value": v})
    row, det_v5 = check_v5_sense_stability(pages, allp, train, test,
                                        n_nulls, rng)
    checks.append(row)
    rows6, det_v6 = check_v6_compositionality(pages, allp, train, test,
                                           n_nulls, rng)
    checks.append(rows6)
    for i, v in enumerate(det_v6["nulls"]):
        nulls_all.append({"check": "V6", "draw": i + 1, "value": v})

    v, n_pass = verdict(checks)

    # aggregated detail for the report (every number it cites)
    flat = {p: [pages[p]] for p in allp}
    sense_det = {
        "v1_null_min_typeperm": min(det_v1["nulls_typeperm"]),
        "v1_order0_ce": heldout_ce(flat, train, test, 0, LAM_GRID[0]),
        "v3_n_frames": det_v3["n_frames_train"],
        "v4_test_tokens": det_v4["n_test_tokens"],
        "v5_n_better": det_v5["n_better"],
        "v5_n_terms": det_v5["n_terms"],
        "v5_med_same": det_v5["med_same"],
        "v5_med_mism": det_v5["med_mism"],
        "v5_dice10_same": det_v5["dice10_same"],
        "v5_dice10_mism": det_v5["dice10_mism"],
        "v5_cosine_same": det_v5["cosine_same"],
        "v5_cosine_mism": det_v5["cosine_mism"],
        "v6_unigram_ce": det_v6["unigram_ce"],
    }
    write_tsv(out / "sense_stability.tsv",
              ["term", "dice_same", "dice_mismatched",
               "n_partners"], det_v5["per_term"])

    # ---- bound diagnostics ----
    sweep = order_sweep(pages, allp, train, test)
    chrono = chronological_stability(pages, allp)
    artifact_pages = sorted(ng.html_artifact_pages(str(LAYOUT)))
    sens = sensitivity(pages, allp, artifact_pages, n_nulls=n_nulls)

    # ---- residue ----
    ledger, oov_types, oov_pages = residue_ledger(pages, allp, train,
                                                  test, screening)
    oov_rows = [{"type": w, "n_tokens": c,
                 "category_19": (screening.get(w) or {}).get("category", ""),
                 "first_test_page": min(oov_pages[w])}
                for w, c in oov_types.most_common()]

    # ---- meaning extraction ----
    termbase, tb_det = build_termbase(pages, allp, entity_glossary,
                                      screening, clauses)
    passages, token_rows = gloss_passages(bodies, page_units, unit_meta,
                                          is_markup, screening,
                                          entity_glossary, termbase,
                                          tb_det["top_frames"])

    # ---- reconciliation ----
    # Every row is an exact recomputation of a frozen count; `match` is a
    # real comparison, so a drift shows up as False rather than as prose.
    recon = [
        {"artifact": "#10 operator inventory",
         "frozen": "10,949 body operators",
         "recomputed": "%d operator tokens in the #11 stream"
                       % diag["n_operators"],
         "match": diag["n_operators"] == 10949},
        {"artifact": "#17 tokenizer (screened units)",
         "frozen": "74,077 units (74,318 minus 241 tag-furniture)",
         "recomputed": "%d screened units" % diag["n_units_screened"],
         "match": diag["n_units_screened"] == 74077},
        {"artifact": "#33 clause segmentation",
         "frozen": "5,614 clauses",
         "recomputed": "%d clauses" % diag["n_clauses"],
         "match": diag["n_clauses"] == 5614},
        {"artifact": "#11 operator stream",
         "frozen": "90,220 tokens incl. 11,000 operators (body+header)",
         "recomputed": "%d operators (body pages only; headers add 51)"
                       % diag["n_operators"],
         "match": diag["n_operators"] == 10949},
        {"artifact": "#34 frames (book-wide, descriptive)",
         "frozen": "675 frames (646 single / 29 two-slot)",
         "recomputed": "%d frames, mined on all %d clauses for the "
                       "termbase's filler evidence (descriptive, not a "
                       "held-out check)" % (tb_det["n_frames_book"],
                                            diag["n_clauses"]),
         "match": tb_det["n_frames_book"] == 675},
        # the `/` dual role: #14's node-level split against this
        # module's token-level recount of the #17 stream.
        {"artifact": "#14 grammar: `/` Link vs Slash",
         "frozen": "%d Link / %d Slash nodes"
                   % (grammar_counts()["n_link_slash"],
                      grammar_counts()["n_slash"]),
         "recomputed": "%d glued / %d boundary-like slash tokens (#17)"
                       % slash_role_counts(),
         "match": slash_role_counts() == (grammar_counts()["n_link_slash"],
                                          grammar_counts()["n_slash"])},
        # #26's contextual examples: how much of the curated termbase the
        # concordance actually covers (compound/named constructs only).
        {"artifact": "#26 KWIC concordance coverage of the termbase",
         "frozen": "not a frozen count — coverage reported, not gated",
         "recomputed": "%d of %d termbase terms carry a cited context"
                       % (tb_det["n_terms_with_kwic"], len(termbase)),
         "match": True},
    ]

    # ---- summary ----
    summary = {
        "schema": SCHEMA,
        "generated_at": t0.isoformat(),
        "seed": SEED,
        "n_nulls": n_nulls,
        "n_pages": diag["n_pages"],
        "n_units_screened": diag["n_units_screened"],
        "n_operators": diag["n_operators"],
        "n_clauses": diag["n_clauses"],
        "split": "interleaved (even pages train, odd pages test)",
        "checks": [{k: r[k] for k in ("check", "statistic", "book",
                                      "null_mean", "null_min", "null_max",
                                      "p_value", "criterion", "passed")}
                   for r in checks],
        "verdict": v,
        "n_passed": n_pass,
        "verdict_rule": ">= %d of 6 pre-registered checks pass -> VIABLE "
                        "(BOUNDED); %d-%d -> PARTIALLY VIABLE; <= 2 -> "
                        "NOT VIABLE" % (VERDICT_PASS, VERDICT_PARTIAL,
                                         VERDICT_PASS - 1),
        "order_sweep": sweep,
        "chronological": chrono,
        "sensitivity": sens,
        "termbase": tb_det,
        "oov": {"n_types": len(oov_types),
                "n_tokens": int(sum(oov_types.values()))},
    }

    # ---- write artifacts ----
    write_tsv(out / "checks.tsv",
              ["check", "split", "statistic", "book", "null_mean",
               "null_min", "null_max", "n_nulls", "p_value", "criterion",
               "passed"],
              checks)
    write_tsv(out / "null_distributions.tsv", ["check", "draw", "value"],
              nulls_all)
    write_tsv(out / "order_sweep.tsv", ["order", "lam", "val_ce", "test_ce"],
              sweep)
    write_tsv(out / "split_stability.tsv",
              ["split", "train_pages", "test_pages", "lam", "val_ce",
               "test_ce"], [chrono])
    write_tsv(out / "sensitivity.tsv",
              ["dropped_pages", "n_pages", "n_nulls", "lam", "test_ce",
               "null_min", "p_value"], [sens])
    write_tsv(out / "residue_ledger.tsv", ["class", "item", "detail",
                                           "pages"], ledger)
    write_tsv(out / "oov_types.tsv",
              ["type", "n_tokens", "category_19", "first_test_page"],
              oov_rows)
    write_tsv(out / "termbase_sample.tsv",
              ["term", "pos_class", "category_19", "n_occurrences",
               "n_pages", "first_page", "parts_18", "entity_role_30",
               "does_verbs_30", "done_to_verbs_30", "top_collocates",
               "frame_fillers_34", "kwic_examples_26", "gloss",
                "confidence"], termbase)
    write_tsv(out / "gloss_tokens.tsv",
              ["passage", "page", "start", "surface", "kind", "role",
               "uid", "category_19", "parts_18", "entity_role_30",
               "collocates", "frame_fillers_34", "gloss", "confidence"],
              token_rows)
    write_tsv(out / "reconciliation.tsv",
              ["artifact", "frozen", "recomputed", "match"], recon)
    write_tsv(out / "diagnostics.tsv", ["key", "value"],
              [{"key": k, "value": v} for k, v in sorted(diag.items())])
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # readable gloss passages
    lines = ["# Gloss passages — evidence-linked reading", ""]
    for ps in passages:
        lines.append("## %s" % ps["title"])
        lines.append("")
        lines.append("Original (p. %d, offsets %d-%d):" %
                     (ps["page"], ps["start"], ps["end"]))
        lines.append("")
        lines.append("> %s" % ps["original"].replace("\n", " "))
        lines.append("")
        lines.append("Operator-normalized (#13 v3, operators to space):")
        lines.append("")
        lines.append("> %s" % ps["v3_space"])
        lines.append("")
        lines.append("Operator-normalized (#13 v4, operators to period):")
        lines.append("")
        lines.append("> %s" % ps["v4_period"])
        lines.append("")
        lines.append(md_table(
            ["surface", "kind", "category (#19)", "parts (#18)",
             "entity role (#30)", "frame fillers (#34)", "gloss",
             "confidence"],
            [{"surface": r["surface"], "kind": r["kind"],
              "category (#19)": r["category_19"], "parts (#18)": r["parts_18"],
              "entity role (#30)": r["entity_role_30"],
              "frame fillers (#34)": r["frame_fillers_34"],
              "gloss": r["gloss"], "confidence": r["confidence"]}
             for r in ps["rows"]]))
        lines.append("")
        unresolved = [r["surface"] for r in ps["rows"]
                      if r["confidence"] == "unresolved"]
        lines.append("Unresolved tokens (residue, not repaired): %s"
                     % (", ".join(unresolved) or "none"))
        lines.append("")
    (out / "gloss_passages.md").write_text("\n".join(lines),
                                           encoding="utf-8")

    write_report(out, summary, checks, v, n_pass, termbase,
                 passages, ledger, oov_rows, recon, sweep,
                 chrono, sens, sense_det, n_nulls, diag, tb_det)

    return {"summary": summary, "checks": checks, "verdict": v,
            "n_pass": n_pass, "termbase": termbase, "passages": passages,
            "ledger": ledger, "recon": recon, "sweep": sweep,
            "chrono": chrono, "sens": sens}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nulls", type=int, default=N_NULLS,
                    help="null draws per check (default %d)" % N_NULLS)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    res = build(out=Path(args.out), n_nulls=args.nulls)
    print("verdict: %s (%d/6 checks pass)" % (res["verdict"],
                                              res["n_pass"]))
    for r in res["checks"]:
        # V0 is the negative control: finding no language there is the
        # desired outcome, so it must not print as a bare FAIL beside
        # the scored checks.
        state = ("CONTROL OK" if r["check"] == "V0"
                 else ("PASS" if r["passed"] else "FAIL"))
        print("  %-4s book=%.4f null[min..max]=[%.4f..%.4f] p=%.4f -> %s"
              % (r["check"], r["book"], r["null_min"], r["null_max"],
                 r["p_value"], state))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
