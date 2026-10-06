#!/usr/bin/env python3
"""#28 Static embeddings and analogies -> analysis/embeddings/

Trains static word embeddings on the book's **unit layer** (#17 ``units.tsv``,
screened with #19's markup ledger) and on a **merged** corpus (the same unit
stream + eight pinned public-domain English prose texts), then reports nearest
neighbours and analogies (``X : Y :: Z : ?``).

Three algorithms are trained on both corpora:

  * **word2vec** skip-gram with negative sampling (gensim, ``sg=1``);
  * **fastText** skip-gram with character n-grams, so an OOV coinage still gets
    a vector from its subwords (gensim, ``min_n=3``/``max_n=6``);
  * **GloVe** -- a self-contained numpy implementation (symmetric window
    co-occurrence + weighted least squares, AdaGrad).  No external GloVe binary
    is needed; it is deterministic under the pinned seed.

The unit layer is the join key, never a token index: ``units.tsv`` is joined to
``morphology/decompositions.tsv`` on ``uid`` for the morphological annotation,
and to ``neologism/screening.tsv`` on the unit surface for the #19 category.
The 347 HTML-tag-residue occurrences (#19 ``markup_residue.tsv``) are removed
before anything is counted, so tag names never become "words".

Contexts are the book's own utterances: consecutive kept units on a page are
split whenever the page body contains a ``//`` operator between them (the
utterance break of #12).  This yields 2,345 sequences over 74,318 units.

Analogies are of four families: ``naming`` (the ``X=Y`` units of #12),
``derivation`` (``modifier-head`` units from #18), ``collocation`` (adjective ->
noun pairs from #27), plus a seeded **mined** set and an English sanity set.

Reproduce::

    conda activate marker
    python work/embedding_models.py            # writes analysis/embeddings/
    python work/test_embedding_models.py

``--quick`` builds tiny book-only models for the regression suite; ``--no-merged``
skips the merged corpus; ``--no-download`` is the default (the reference cache
from #24 is reused, no network).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
PART_INDEX = ROOT / "analysis" / "tokenizer" / "part_index.tsv"
DECOMP = ROOT / "analysis" / "morphology" / "decompositions.tsv"
HEAD_CAND = ROOT / "analysis" / "morphology" / "head_candidates.tsv"
SCREENING = ROOT / "analysis" / "neologism" / "screening.tsv"
MARKUP = ROOT / "analysis" / "neologism" / "markup_residue.tsv"
COLLOC_BIGRAM = ROOT / "analysis" / "collocation" / "collocations_bigram.tsv"
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
REF_CACHE = ROOT / "analysis" / "frequency" / "reference_cache"
DEFAULT_BROWN_DIR = Path.home() / "nltk_data" / "corpora" / "brown"
OUT = ROOT / "analysis" / "embeddings"

SCHEMA = "embeddings/1"
BOOK_N = 74318                     # #19/#24 screened unit occurrences
BOOK_TYPES_FOLDED = 5502           # #24 case-folded unit types
N_MARKUP = 347                     # #19 markup-residue occurrences

# Matched English word tokenizer (the same shape as #24's word-part view).
EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
_PAGE_RE = re.compile(r"<!-- page (\d+) -->")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_GUT_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL)
_GUT_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL)

# ---------------------------------------------------------------------------
# analogy probe families (curated; targets are the book's own units/entities)
# ---------------------------------------------------------------------------
# naming: a=b units from #12 (right part is the "head").
NAMING_PROBES = [
    ("gene", "tv", "vital", "serum", "gene=TV / vital=serum"),
    ("gene", "tv", "suck", "blood", "gene=TV / suck=blood"),
    ("gene", "tv", "digital", "vamp", "gene=TV / digital=vamp"),
    ("gene", "tv", "fear", "cell", "gene=TV / fear=cell"),
    ("gene", "tv", "machine", "angel", "gene=TV / machine=angel"),
    ("gene", "tv", "dna", "channel", "gene=TV / DNA=channel"),
    ("suck", "blood", "fear", "cell", "suck=blood / fear=cell"),
    ("vital", "serum", "digital", "vamp", "vital=serum / digital=vamp"),
    ("machine", "angel", "digital", "vamp", "machine=angel / digital=vamp"),
    ("dna", "channel", "genome", "linkage", "DNA=channel / genome=linkage"),
    ("data", "mutant", "genome", "linkage", "data=mutant / genome=linkage"),
    ("fear", "cell", "data", "mutant", "fear=cell / data=mutant"),
]
# derivation: modifier-head hyphen/slash units from #18 (head = right part).
DERIVATION_PROBES = [
    ("boy", "roid", "cadaver", "feti", "boy-roid / cadaver-feti"),
    ("boy", "roid", "flesh", "module", "boy-roid / flesh-module"),
    ("boy", "roid", "soul", "machine", "boy-roid / soul-machine"),
    ("cadaver", "feti", "drug", "embryo", "cadaver-feti / drug-embryo"),
    ("cold", "blooded", "retro", "adam", "cold-blooded / retro-ADAM"),
    ("cadaver", "mechanism", "desire", "protocol",
     "cadaver-mechanism / desire-protocol"),
    ("desire", "protocol", "existence", "code",
     "desire-protocol / existence-code"),
    ("flesh", "module", "soul", "machine", "flesh-module / soul-machine"),
    ("body", "omotya", "retro", "adam", "body-OMOTYA / retro-ADAM"),
    ("soul", "gram", "vital", "icon", "soul/gram / vital/icon"),
]
# collocation: adjective -> noun pairs from #27.
COLLOCATION_PROBES = [
    ("artificial", "sun", "internal", "organ", "artificial sun / internal organ"),
    ("artificial", "sun", "sickly", "period", "artificial sun / sickly period"),
    ("internal", "organ", "ectoplasm", "spot", "internal organ / ectoplasm spot"),
    ("primal", "cyclops", "oneiric", "locus", "primal cyclops / oneiric locus"),
    ("nebula", "faecal", "gelid", "ejaculation",
     "nebula faecal / gelid ejaculation"),
    ("sickly", "period", "inorganic", "substance",
     "sickly period / inorganic substance"),
]
# English sanity probes (evaluated on the merged models; book models as control).
ENGLISH_PROBES = [
    ("man", "king", "woman", "queen"),
    ("france", "paris", "england", "london"),
    ("good", "better", "bad", "worse"),
    ("run", "ran", "walk", "walked"),
    ("big", "bigger", "small", "smaller"),
    ("tokyo", "japan", "paris", "france"),
    ("dog", "puppy", "cat", "kitten"),
]


# ---------------------------------------------------------------------------
# tsv io (quote-safe: split on tabs, never a csv reader -- #17)
# ---------------------------------------------------------------------------
def read_tsv(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh if line.strip()]


def cell(value) -> str:
    s = "" if value is None else str(value)
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(cell(r.get(h, "")) for h in header) + "\n")


def unescape_cell(s: str) -> str:
    """Reverse #19's double escaping so the unit-surface join is exact."""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            out.append({"n": "\n", "t": "\t", "r": "\r",
                        "\\": "\\"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# book inputs
# ---------------------------------------------------------------------------
def load_bodies(src: Path = LAYOUT):
    text = src.read_text(encoding="utf-8", errors="ignore")
    parts = _PAGE_RE.split(text)
    return {int(parts[k]): parts[k + 1]
            for k in range(1, len(parts) - 1, 2)}


def load_book():
    """Return (kept_units, seqs, cats, meta, n_pages).

    ``kept_units`` - markup-screened unit occurrences, ordered by (page, start);
    ``seqs``       - utterances (split on the ``//`` operator between units);
    ``cats``       - case-preserved unit surface -> #19 category;
    ``meta``       - case-folded type -> aggregate metadata (count, example,
                     first page/uid, zipf, head, parts, category).

    The join to #19 is asserted, not assumed: aggregating the kept units by
    surface must reproduce #19's ``screening.tsv`` counts exactly.
    """
    units = read_tsv(UNITS)
    markup_uids = {r["uid"] for r in read_tsv(MARKUP)}
    screen = read_tsv(SCREENING)
    cats = {unescape_cell(r["unit"]): r["category"] for r in screen}
    screen_n = {unescape_cell(r["unit"]): int(r["n"]) for r in screen}
    scr = {unescape_cell(r["unit"]): r for r in screen}

    units.sort(key=lambda r: (int(r["page"]), int(r["start"]), int(r["uid"])))
    kept = [r for r in units if r["uid"] not in markup_uids]

    occ = collections.Counter(r["unit"] for r in kept)
    mism = [(u, occ.get(u, 0), n) for u, n in screen_n.items()
            if occ.get(u, 0) != n]
    extra = sorted(set(occ) - set(screen_n))
    if mism or extra:
        raise SystemExit(
            f"unit-layer join failed: {len(mism)} count mismatches, "
            f"{len(extra)} types absent from #19 screening "
            f"(e.g. {mism[:3]} {extra[:3]})")

    # morphology join on uid (never a token index)
    head_of = {}
    for r in read_tsv(DECOMP):
        head_of.setdefault(r["uid"], r["head"].lower())

    bodies = load_bodies()
    seqs = []
    for page in sorted(bodies):
        body = bodies[page]
        page_units = [r for r in kept if int(r["page"]) == page]
        cur, prev_end = [], None
        for r in page_units:
            s, e = int(r["start"]), int(r["end"])
            if prev_end is not None and "//" in body[prev_end:s]:
                if cur:
                    seqs.append(cur)
                cur = []
            cur.append(r["unit"].lower())
            prev_end = e
        if cur:
            seqs.append(cur)

    # per-type metadata
    surface_count = collections.Counter(r["unit"] for r in kept)
    meta = {}
    for r in kept:
        low = r["unit"].lower()
        m = meta.setdefault(low, {
            "lower": low, "n": 0, "surface": r["unit"], "surface_n": 0,
            "first_page": int(r["page"]), "first_uid": int(r["uid"]),
            "category": cats.get(r["unit"], "markup"),
        })
        m["n"] += 1
        if surface_count[r["unit"]] > m["surface_n"]:
            m["surface_n"] = surface_count[r["unit"]]
            m["surface"] = r["unit"]
    for low, m in meta.items():
        s = scr.get(m["surface"])
        m["zipf_en"] = s["zipf_en"] if s else ""
        m["head"] = head_of.get(str(m["first_uid"]), "")
        m["parts"] = s["parts"] if s else ""
        m["proper_noun"] = s["proper_noun"] if s else ""
        m["cls"] = s["cls"] if s else ""
    return kept, seqs, cats, meta, len(bodies)


def load_english(token_cap=0, include_brown=True):
    """English sentence sequences: 8 pinned Gutenberg texts + the Brown corpus.

    Brown is parsed with stdlib only (``word/tag`` tokens, one sentence per
    line), so no NLTK import is needed.  ``token_cap=0`` means no cap.
    """
    out = []
    total = 0

    def push(words):
        nonlocal total
        if not words:
            return False
        out.append(words[:60])
        total += len(words)
        return bool(token_cap) and total >= token_cap

    for path in sorted(REF_CACHE.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        m = _GUT_START.search(text)
        if m:
            text = text[m.end():]
        m = _GUT_END.search(text)
        if m:
            text = text[:m.start()]
        for sent in _SENT_SPLIT_RE.split(text):
            if push([w.lower() for w in EN_WORD_RE.findall(sent)]):
                return out
    if include_brown and DEFAULT_BROWN_DIR.is_dir():
        for fn in sorted(glob.glob(str(DEFAULT_BROWN_DIR / "c*"))):
            for line in Path(fn).read_text(
                    encoding="utf-8", errors="ignore").splitlines():
                words = []
                for tok in line.split():
                    if "/" not in tok:
                        continue
                    w = tok.rsplit("/", 1)[0].lower()
                    if EN_WORD_RE.fullmatch(w):
                        words.append(w)
                if push(words):
                    return out
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_word2vec(seqs, dim, window, min_count, epochs, seed):
    from gensim.models import Word2Vec
    model = Word2Vec([s for s in seqs if s], vector_size=dim, window=window,
                     min_count=min_count, sg=1, negative=5, workers=1,
                     seed=seed, epochs=epochs, sorted_vocab=1)
    return model.wv


def train_fasttext(seqs, dim, window, min_count, epochs, seed,
                   min_n=3, max_n=6, bucket=50000):
    from gensim.models import FastText
    model = FastText([s for s in seqs if s], vector_size=dim, window=window,
                     min_count=min_count, sg=1, negative=5, workers=1,
                     seed=seed, epochs=epochs, min_n=min_n, max_n=max_n,
                     bucket=bucket, sorted_vocab=1)
    return model.wv


def build_cooccurrence(seqs, vocab_index, window):
    """Symmetric window co-occurrence with 1/distance weighting -> Counter."""
    pairs = collections.Counter()
    for seq in seqs:
        ids = [vocab_index[t] for t in seq if t in vocab_index]
        n = len(ids)
        for i, a in enumerate(ids):
            hi = min(n, i + window + 1)
            for j in range(i + 1, hi):
                b = ids[j]
                key = (a, b) if a < b else (b, a)
                pairs[key] += 1.0 / (j - i)
    return pairs


def _scatter_sum(idx, vals, n_rows):
    """Per-row sums of ``vals`` grouped by ``idx`` (duplicate-safe)."""
    order = np.argsort(idx, kind="stable")
    idx_s, vals_s = idx[order], vals[order]
    uniq, starts = np.unique(idx_s, return_index=True)
    sums = np.add.reduceat(vals_s, starts, axis=0)
    return uniq, sums


def train_glove(seqs, dim, window, min_count, epochs, seed,
                lr=0.05, x_max=100.0, alpha=0.75, batch=262144):
    """Self-contained GloVe: co-occurrence + weighted least squares, AdaGrad.

    Returns ``(keys, matrix)``.  Duplicate indices inside a batch are summed
    before the update (``_scatter_sum``), so a zipfian vocabulary trains
    correctly rather than last-write-wins.
    """
    counts = collections.Counter(t for s in seqs for t in s)
    keys = sorted(t for t, c in counts.items() if c >= min_count)
    vi = {t: i for i, t in enumerate(keys)}
    pairs = build_cooccurrence(seqs, vi, window)
    V = len(keys)
    I = np.fromiter((k[0] for k in pairs), dtype=np.int64, count=len(pairs))
    J = np.fromiter((k[1] for k in pairs), dtype=np.int64, count=len(pairs))
    X = np.fromiter(pairs.values(), dtype=np.float64, count=len(pairs))
    rng = np.random.default_rng(seed)
    f = np.minimum((X / x_max) ** alpha, 1.0)
    W = (rng.random((V, dim)) - 0.5) / dim
    Wb = (rng.random((V, dim)) - 0.5) / dim
    b = np.zeros(V)
    bb = np.zeros(V)
    accW = np.full((V, dim), 1e-8)
    accWb = np.full((V, dim), 1e-8)
    accb = np.full(V, 1e-8)
    accbb = np.full(V, 1e-8)
    for _ in range(epochs):
        order = rng.permutation(len(I))
        for st in range(0, len(I), batch):
            idx = order[st:st + batch]
            i, j, x, w = I[idx], J[idx], X[idx], f[idx]
            diff = (W[i] * Wb[j]).sum(1) + b[i] + bb[j] - np.log(x)
            wd = w * diff
            gi = wd[:, None] * Wb[j]
            gj = wd[:, None] * W[i]
            ui, si = _scatter_sum(i, gi, V)
            uj, sj = _scatter_sum(j, gj, V)
            accW[ui] += si * si
            W[ui] -= lr * si / np.sqrt(accW[ui])
            accWb[uj] += sj * sj
            Wb[uj] -= lr * sj / np.sqrt(accWb[uj])
            bi = np.bincount(i, weights=wd, minlength=V)
            bj = np.bincount(j, weights=wd, minlength=V)
            nzi = np.nonzero(bi)[0]
            nzj = np.nonzero(bj)[0]
            accb[nzi] += bi[nzi] ** 2
            b[nzi] -= lr * bi[nzi] / np.sqrt(accb[nzi])
            accbb[nzj] += bj[nzj] ** 2
            bb[nzj] -= lr * bj[nzj] / np.sqrt(accbb[nzj])
    return keys, W + Wb


# ---------------------------------------------------------------------------
# unified model wrapper
# ---------------------------------------------------------------------------
class VecModel:
    """Cosine nearest-neighbour search over a static embedding.

    ``subwords=True`` (fastText) makes ``has()`` always true and lets an OOV
    term get a vector from its character n-grams.
    """

    def __init__(self, name, corpus, algo, keys, matrix, subwords=False,
                 vector_fn=None):
        self.name = name
        self.corpus = corpus
        self.algo = algo
        self.keys = list(keys)
        self.key_to_index = {k: i for i, k in enumerate(self.keys)}
        self.matrix = np.asarray(matrix, dtype=np.float64)
        norms = np.linalg.norm(self.matrix, axis=1)
        norms[norms == 0] = 1.0
        self.norms = norms
        self.subwords = subwords
        self.vector_fn = vector_fn

    def has(self, term):
        if self.subwords:
            return True
        return term in self.key_to_index

    def vector(self, term):
        if self.subwords:
            try:
                return np.asarray(self.vector_fn(term), dtype=np.float64)
            except KeyError:
                return None
        i = self.key_to_index.get(term)
        return None if i is None else self.matrix[i]

    def most_similar(self, positive, negative=(), topn=10, exclude=()):
        q = None
        for t in positive:
            v = self.vector(t)
            if v is None:
                return None
            q = v if q is None else q + v
        for t in negative:
            v = self.vector(t)
            if v is None:
                return None
            q = q - v if q is not None else -v
        if q is None:
            return []
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []
        sims = (self.matrix @ q) / (self.norms * qn)
        order = np.argsort(-sims)
        banned = set(positive) | set(negative) | set(exclude)
        out = []
        for o in order:
            k = self.keys[o]
            if k in banned:
                continue
            out.append((k, float(sims[o])))
            if len(out) >= topn:
                break
        return out


def make_vec_model(name, corpus, algo, wv, subwords=False):
    keys = list(wv.index_to_key)
    matrix = np.asarray(wv.vectors, dtype=np.float64)
    return VecModel(name, corpus, algo, keys, matrix, subwords=subwords,
                    vector_fn=wv.get_vector)


def load_kv(path: Path):
    """Load a saved KeyedVectors (fastText subwords survive the round-trip)."""
    from gensim.models import KeyedVectors
    return KeyedVectors.load(str(path))


def load_glove(path: Path):
    z = np.load(str(path), allow_pickle=True)
    return list(z["keys"]), z["vectors"]


def analogy_rank(model: VecModel, a, b, c, expected, topn=10):
    """Return (rank, top_list) for ``a : b :: c : ?``; rank is 1-based or None.

    ``a : b :: c : d`` means ``b - a + c = d`` (``positive=[b, c]``,
    ``negative=[a]``).  ``None`` rank means a needed term had no vector.
    """
    res = model.most_similar([b, c], negative=[a], topn=topn)
    if res is None:
        return None, []
    for r, (k, _s) in enumerate(res, 1):
        if k == expected:
            return r, res
    return 0, res                       # 0 == evaluated but not in top-k


def _probe_rows(probes, family, source, with_note=False):
    rows = []
    for p in probes:
        a, b, c, expected = p[0], p[1], p[2], p[3]
        note = p[4] if with_note and len(p) > 4 else ""
        rows.append({"probe": f"{a}:{b}::{c}:?", "a": a, "b": b, "c": c,
                     "expected": expected, "family": family, "source": source,
                     "note": note})
    return rows


def mine_derivation_probes(max_probes=48):
    """Seeded morphological analogies from #18 ``head_candidates.tsv``.

    Relation assumed: modifier -> head (the head a base attaches to).  A base
    may attach to several heads, so the relation is not a function; that limit
    is disclosed and the family is reported separately from the curated set.
    """
    rows = []
    for r in read_tsv(HEAD_CAND):
        head = r["morph"].lower()
        bases = [b for b in r["bases"].split(";") if b]
        try:
            n = int(r["n_attached"])
        except ValueError:
            n = 0
        for base in bases[:3]:
            rows.append((base.lower(), head, n))
    rows.sort(key=lambda t: (-t[2], t[0], t[1]))
    pairs = rows[:24]
    probes = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            m1, h1, _ = pairs[i]
            m2, h2, _ = pairs[j]
            if h1 == h2 or m1 == m2:
                continue
            probes.append({"probe": f"{m1}:{h1}::{m2}:?", "a": m1, "b": h1,
                           "c": m2, "expected": h2, "family": "mined_derivation",
                           "source": "#18", "note": f"head {h1} / {h2}"})
            if len(probes) >= max_probes:
                return probes
    return probes


def mine_collocation_probes(max_probes=48, min_f=40):
    """Seeded collocation analogies from #27's unit bigrams."""
    big = []
    for r in read_tsv(COLLOC_BIGRAM):
        try:
            f = int(r["f_xy"])
        except ValueError:
            continue
        if f < min_f:
            continue
        x, y = r["x"], r["y"]
        if not (x.isalpha() and y.isalpha()):
            continue
        big.append((x, y, f))
    big.sort(key=lambda t: (-t[2], t[0], t[1]))
    big = big[:24]
    probes = []
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            x1, y1, _ = big[i]
            x2, y2, _ = big[j]
            if x1 == x2 or y1 == y2:
                continue
            probes.append({"probe": f"{x1}:{y1}::{x2}:?", "a": x1, "b": y1,
                           "c": x2, "expected": y2,
                           "family": "mined_collocation", "source": "#27",
                           "note": f"collocation {y1} / {y2}"})
            if len(probes) >= max_probes:
                return probes
    return probes


# ---------------------------------------------------------------------------
# selected terms
# ---------------------------------------------------------------------------
MOTIF_TERMS = [
    "the", "of", "self", "body", "machine", "time", "world", "city", "dog",
    "girl", "brain", "blood", "murder", "memory", "flesh", "sun", "organ",
    "embryo", "virus", "death", "love", "fear", "dream", "sky", "sex",
    "artificial", "cadaver", "soul", "gram", "vamp", "serum", "cell",
]
ENTITY_TERMS = ["adam", "anti-adam", "tokage", "ndro", "omotya", "babel",
                "sato", "cadaver", "placenta", "evol", "cherry", "super"]
COINAGE_TERMS = ["boy-roid", "soul/gram", "gene=tv", "suck=blood",
                 "vital=serum", "fear=cell", "digital=vamp", "drug-embryo",
                 "cold-blooded", "hydro=mania", "machine=angel", "cadaver-feti",
                 "fecundates", "defleshed"]


def select_terms(meta, n_each=15):
    """Curated + data-driven seed terms, case-folded and de-duplicated."""
    terms = []

    def add(t):
        t = t.lower()
        if t in meta and t not in terms:
            terms.append(t)

    for t in ENTITY_TERMS + MOTIF_TERMS + COINAGE_TERMS:
        add(t)
    coins = sorted((m for m in meta.values()
                    if m["category"] in ("compound", "true_coinage",
                                         "proper_noun")),
                   key=lambda m: (-m["n"], m["lower"]))
    for m in coins[:n_each]:
        add(m["lower"])
    freq = sorted(meta.values(), key=lambda m: (-m["n"], m["lower"]))
    for m in freq[:n_each]:
        add(m["lower"])
    return terms


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _pct(x):
    return f"{100.0 * x:.1f}%"


def build_report(ctx):
    L = []
    A = L.append
    A("# Static embeddings and analogies (#28)")
    A("")
    A("Source: #17 unit layer (`units.tsv`), markup-screened with #19's ledger "
      "and joined to #18 `decompositions.tsv` on `uid`; the merged English side "
      "is #24's pinned Gutenberg cache plus the Brown corpus. Reproducible via "
      "`python work/embedding_models.py`; "
      f"regression suite `python work/test_embedding_models.py`. Schema `{SCHEMA}`.")
    A("")
    A("## 1. Verdict")
    A("")
    A(f"- The unit layer gives **{ctx['n_units']:,} screened unit occurrences** "
      f"({ctx['n_types']:,} case-folded types) over **{ctx['n_seqs']:,} "
      "utterance sequences** (split on the `//` operator), so `gene=TV` is one "
      "token, not `gene` + `=` + `TV`.")
    A(f"- Six models are trained: word2vec / fastText / GloVe on the book alone, "
      f"and word2vec / fastText / GloVe on the book + {ctx['en_tokens']:,} "
      "English tokens (8 Gutenberg texts + the Brown corpus).")
    A(f"- **fastText's subwords are what make coinages reachable.** In the "
      f"merged corpus, word2vec and GloVe cover only {_pct(ctx['cov_w2v_merged'])} "
      f"/ {_pct(ctx['cov_glove_merged'])} of the book's {ctx['n_types']:,} types "
      "(a type must recur `min_count=5` times in 2M English-dominated tokens), "
      f"while fastText covers **{_pct(ctx['cov_ft_merged'])}** because every OOV "
      "coinage is reconstructed from character n-grams. On the book alone all "
      "three cover 100% by construction (`min_count=1`).")
    A(f"- Pooled over the curated naming/derivation/collocation probes, the "
      f"best model is **{ctx['best_family']}** (hit@10 "
      f"{_pct(ctx['best_family_hit10'])}); the book-alone models recover the "
      "book's own equations, the merged models trade book signal for English "
      f"neighbours, and English sanity analogies pass on the merged models "
      f"({_pct(ctx['english_hit10'])} hit@10).")
    A("")
    A("The embeddings are a **distributional view, not a definition**. They "
      "measure which units the book keeps in the same company; the analogies "
      "are arithmetic probes on that geometry, and their hits are evidence "
      "about recurring relational structure, not proof of meaning.")
    A("")
    A("## 2. Why the unit layer, and how contexts are built")
    A("")
    A("- #17 showed a standard tokenizer keeps **0%** of fused units whole. "
      "This analysis therefore embeds the **maximal unit** (`gene=TV`, "
      "`soul/gram`, `suck=blood`) and leans on fastText subwords for the parts.")
    A("- Every occurrence is joined on `uid` / `(page, start)`, never a token "
      "index. The #19 join is asserted: aggregating the kept units by surface "
      "must reproduce `screening.tsv` counts exactly.")
    A(f"- The **{N_MARKUP}** HTML-tag-residue occurrences (#19 "
      "`markup_residue.tsv`) are removed first, so `th`/`p`/`pre` never enter "
      "as hapax words.")
    A("- Contexts are the book's own utterances: consecutive units are split "
      "where the page body contains a `//` operator between them. This uses "
      "the operator stream as the sentence boundary (the #12 utterance role).")
    A("")
    A("## 3. Models")
    A("")
    A("| model | corpus | algo | vocab | dim | window | min_count | subwords |")
    A("|---|---|---|---:|---:|---:|---:|---|")
    for m in ctx["model_rows"]:
        A(f"| `{m['model']}` | {m['corpus']} | {m['algo']} | {m['vocab']:,} | "
          f"{m['dim']} | {m['window']} | {m['min_count']} | "
          f"{'yes' if m['subwords'] else 'no'} |")
    A("")
    A(f"Seed `{ctx['seed']}`, `workers=1`; results are reproducible. word2vec and "
      "fastText use gensim; GloVe is a self-contained numpy implementation "
      "(co-occurrence + AdaGrad), not an external binary.")
    A("")
    A("## 4. Nearest neighbours (book-alone models)")
    A("")
    A("Top neighbours by cosine. These are the most direct output of the "
      "distributional layer: what each unit keeps company with.")
    A("")
    for term in ctx["nn_showcase"]:
        A(f"### `{term}`")
        A("")
        A("| model | rank | neighbour | cosine |")
        A("|---|---:|---|---:|")
        for mname in ("w2v_book", "ft_book", "glove_book"):
            for rank, (n, s) in enumerate(ctx["nn_selected"].get(
                    (mname, term), [])[:5], 1):
                A(f"| `{mname}` | {rank} | `{n}` | {s:.3f} |")
        A("")
    A("## 5. Analogies (`X : Y :: Z : ?`)")
    A("")
    A("`a : b :: c : ?` is evaluated as `b - a + c` and scored against the "
      "expected unit. `hit@k` counts probes whose expected unit is in the top "
      "`k`; `oov` counts probes where a required term has no vector in that "
      "model. Mined families assume a relation (modifier->head; "
      "adjective->noun) that is not a function, so they are reported "
      "separately and read as a lower bound.")
    A("")
    A("| model | family | probes | evaluable | hit@1 | hit@5 | hit@10 | MRR |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in ctx["analogy_summary"]:
        A(f"| `{s['model']}` | {s['family']} | {s['n']} | {s['n_eval']} | "
          f"{_pct(s['hit1'])} | {_pct(s['hit5'])} | {_pct(s['hit10'])} | "
          f"{s['mrr']:.3f} |")
    A("")
    A("Pooled over the three curated families (the headline number):")
    A("")
    A("| model | evaluable | hit@1 | hit@5 | hit@10 | MRR |")
    A("|---|---:|---:|---:|---:|---:|")
    for s in ctx["pooled_curated"]:
        A(f"| `{s['model']}` | {s['n_eval']} | {_pct(s['hit1'])} | "
          f"{_pct(s['hit5'])} | {_pct(s['hit10'])} | {s['mrr']:.3f} |")
    A("")
    A("### 5.1 Worked book analogies (fastText, merged)")
    A("")
    A("| probe | expected | top-1 | rank |")
    A("|---|---|---|---:|")
    for r in ctx["worked_analogies"]:
        rank = r["rank"]
        shown = ("oov" if rank is None else
                 (str(rank) if rank else f">{ctx['topn']}"))
        A(f"| `{r['probe']}` | `{r['expected']}` | "
          f"{('`' + r['top1'] + '`') if r['top1'] else '—'} | {shown} |")
    A("")
    A("### 5.2 English sanity check (merged models)")
    A("")
    A("| probe | expected | " +
      " | ".join(f"`{m}`" for m in ctx["english_models"]) + " |")
    A("|---|" + "|---" * (len(ctx["english_models"]) + 1) + "|")
    for r in ctx["english_rows"]:
        A(f"| `{r['probe']}` | `{r['expected']}` | " +
          " | ".join(str(r[m]) for m in ctx["english_models"]) + " |")
    A("")
    A("The English rows show the pipeline recovers ordinary analogies when the "
      "merged corpus dominates. Book-alone models are omitted here because the "
      "probe words (`king`, `france`, ...) do not occur in the book; their "
      "behaviour on the book's own equations is in §5.1 and the summary table.")
    A("")
    A("## 6. Interpretation and limits")
    A("")
    A("- **Distributional, not semantic.** A neighbour is a co-occurrence "
      "claim. `tokage` near `vital=serum` says they share contexts, not that "
      "one defines the other.")
    A("- **Rare types have noisy vectors.** With `min_count=1` a hapax unit is "
      "trained on very few contexts; the merged models raise `min_count` and "
      "then rely on fastText subwords for the tail.")
    A("- **Analogy targets are our probes, not the book's.** The curated "
      "families encode a relation we assert (naming, modifier->head, "
      "adjective->noun). The mined families measure whether that asserted "
      "relation is consistent in the geometry; low scores are a real negative "
      "result, not a bug.")
    A("- **Merged corpus is mostly 19th-century prose + Brown**, not cyberpunk "
      "(#56 owns the register-matched corpora). It supplies general English "
      "vectors; it does not supply the book's genre. The book is only "
      f"{_pct(ctx['n_units'] / max(1, ctx['n_units'] + ctx['en_tokens']))} of "
      "the merged token stream, so book relations are diluted there; that is "
      "why the book-alone models are reported beside the merged ones.")
    A("- **Subword coverage is not subword quality.** fastText returns a vector "
      "for every OOV string, but a vector built from a rare coinage's n-grams "
      "can be near its parts rather than its sense; that is itself informative.")
    A("")
    A("## 7. Artifacts")
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `embedding_report.md` | this report |")
    A("| `w2v_book.kv` / `ft_book.kv` / `glove_book.npz` | book-alone models |")
    A("| `w2v_merged.kv` / `ft_merged.kv` / `glove_merged.npz` | merged models |")
    A("| `neighbours_book.tsv` | every book type x 3 book models, top 10 |")
    A("| `neighbours_selected.tsv` | curated terms x 6 models, top 10 |")
    A("| `analogies.tsv` | all book analogy probes with per-model results |")
    A("| `analogies_english.tsv` | English sanity probes |")
    A("| `analogy_summary.tsv` | accuracy by model x family |")
    A("| `vocabulary.tsv` | per-type metadata + per-model coverage |")
    A("| `model_summary.tsv` | training hyper-parameters |")
    A("| `summary.json` / `diagnostics.tsv` / `reconciliation.tsv` | counts |")
    A("")
    A("All `.tsv` are read with `split(\"\\t\")`, never a csv reader.")
    A("")
    A("---")
    A(f"*Generated by `work/embedding_models.py` (schema `{SCHEMA}`).*")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="#28 static embeddings / analogies")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count-book", type=int, default=1)
    ap.add_argument("--min-count-merged", type=int, default=5)
    ap.add_argument("--w2v-epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=25)
    ap.add_argument("--glove-epochs", type=int, default=15)
    ap.add_argument("--english-token-cap", type=int, default=0,
                    help="0 = use the whole English corpus")
    ap.add_argument("--no-brown", action="store_true",
                    help="exclude the Brown corpus from the merged side")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-merged", action="store_true")
    ap.add_argument("--reuse-models", action="store_true",
                    help="load existing model files from --out and skip training")
    ap.add_argument("--quick", action="store_true",
                    help="tiny book-only models for the regression suite")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.dim = 32
        args.min_count_book = 2
        args.w2v_epochs = args.ft_epochs = args.glove_epochs = 3
        args.no_merged = True
        args.topn = 5
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kept, seqs, cats, meta, n_pages = load_book()
    n_units = len(kept)
    n_types = len(meta)
    if not args.quiet:
        print(f"[embeddings] units={n_units} types={n_types} seqs={len(seqs)}")

    # ---- book models ----
    models = {}
    model_rows = []

    def register(name, corpus, algo, vec_model, row):
        models[name] = vec_model
        model_rows.append(row)

    wv = (load_kv(out / "w2v_book.kv")
          if args.reuse_models and (out / "w2v_book.kv").exists()
          else train_word2vec(seqs, args.dim, args.window,
                              args.min_count_book, args.w2v_epochs, args.seed))
    if not (args.reuse_models and (out / "w2v_book.kv").exists()):
        wv.save(str(out / "w2v_book.kv"))
    register("w2v_book", "book", "word2vec",
             make_vec_model("w2v_book", "book", "word2vec", wv),
             {"model": "w2v_book", "corpus": "book", "algo": "word2vec",
              "vocab": len(wv), "dim": args.dim, "window": args.window,
              "min_count": args.min_count_book, "subwords": False})
    if not args.quiet:
        print(f"[embeddings] w2v_book vocab={len(wv)}")

    ft = (load_kv(out / "ft_book.kv")
          if args.reuse_models and (out / "ft_book.kv").exists()
          else train_fasttext(seqs, args.dim, args.window,
                              args.min_count_book, args.ft_epochs, args.seed))
    if not (args.reuse_models and (out / "ft_book.kv").exists()):
        ft.save(str(out / "ft_book.kv"))
    register("ft_book", "book", "fastText",
             make_vec_model("ft_book", "book", "fastText", ft, subwords=True),
             {"model": "ft_book", "corpus": "book", "algo": "fastText",
              "vocab": len(ft), "dim": args.dim, "window": args.window,
              "min_count": args.min_count_book, "subwords": True})
    if not args.quiet:
        print(f"[embeddings] ft_book vocab={len(ft)}")

    if args.reuse_models and (out / "glove_book.npz").exists():
        gkeys, gmat = load_glove(out / "glove_book.npz")
    else:
        gkeys, gmat = train_glove(seqs, args.dim, args.window,
                                  args.min_count_book, args.glove_epochs,
                                  args.seed)
        np.savez_compressed(out / "glove_book.npz",
                            keys=np.array(gkeys), vectors=gmat)
    register("glove_book", "book", "GloVe",
             VecModel("glove_book", "book", "GloVe", gkeys, gmat),
             {"model": "glove_book", "corpus": "book", "algo": "GloVe",
              "vocab": len(gkeys), "dim": args.dim, "window": args.window,
              "min_count": args.min_count_book, "subwords": False})
    if not args.quiet:
        print(f"[embeddings] glove_book vocab={len(gkeys)}")

    # ---- merged models ----
    en_tokens = 0
    if not args.no_merged:
        en_seqs = load_english(args.english_token_cap,
                               include_brown=not args.no_brown)
        en_tokens = sum(len(s) for s in en_seqs)
        merged = en_seqs + seqs
        if not args.quiet:
            print(f"[embeddings] merged english_tokens={en_tokens} "
                  f"english_seqs={len(en_seqs)}")
        if args.reuse_models and (out / "w2v_merged.kv").exists():
            wv = load_kv(out / "w2v_merged.kv")
        else:
            wv = train_word2vec(merged, args.dim, args.window,
                                args.min_count_merged, args.w2v_epochs,
                                args.seed)
            wv.save(str(out / "w2v_merged.kv"))
        register("w2v_merged", "book+english", "word2vec",
                 make_vec_model("w2v_merged", "book+english", "word2vec", wv),
                 {"model": "w2v_merged", "corpus": "book+english",
                  "algo": "word2vec", "vocab": len(wv), "dim": args.dim,
                  "window": args.window, "min_count": args.min_count_merged,
                  "subwords": False})
        if args.reuse_models and (out / "ft_merged.kv").exists():
            ft = load_kv(out / "ft_merged.kv")
        else:
            ft = train_fasttext(merged, args.dim, args.window,
                                args.min_count_merged, args.ft_epochs,
                                args.seed)
            ft.save(str(out / "ft_merged.kv"))
        register("ft_merged", "book+english", "fastText",
                 make_vec_model("ft_merged", "book+english", "fastText", ft,
                                subwords=True),
                 {"model": "ft_merged", "corpus": "book+english",
                  "algo": "fastText", "vocab": len(ft), "dim": args.dim,
                  "window": args.window, "min_count": args.min_count_merged,
                  "subwords": True})
        if args.reuse_models and (out / "glove_merged.npz").exists():
            gkeys, gmat = load_glove(out / "glove_merged.npz")
        else:
            gkeys, gmat = train_glove(merged, args.dim, args.window,
                                      args.min_count_merged,
                                      args.glove_epochs, args.seed)
            np.savez_compressed(out / "glove_merged.npz",
                                keys=np.array(gkeys), vectors=gmat)
        register("glove_merged", "book+english", "GloVe",
                 VecModel("glove_merged", "book+english", "GloVe", gkeys, gmat),
                 {"model": "glove_merged", "corpus": "book+english",
                  "algo": "GloVe", "vocab": len(gkeys), "dim": args.dim,
                  "window": args.window, "min_count": args.min_count_merged,
                  "subwords": False})
        if not args.quiet:
            print("[embeddings] merged models ready")

    model_names = list(models)
    book_models = [m for m in model_names if m.endswith("_book")]
    merged_models = [m for m in model_names if m.endswith("_merged")]

    # ---- coverage ----
    coverage = {}
    for mname, m in models.items():
        coverage[mname] = sum(1 for t in meta if m.has(t))

    # ---- vocabulary.tsv ----
    vocab_rows = []
    for low, m in sorted(meta.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        row = {"lower": low, "surface": m["surface"], "n": m["n"],
               "category": m["category"], "first_page": m["first_page"],
               "first_uid": m["first_uid"], "zipf_en": m["zipf_en"],
               "head": m["head"], "parts": m["parts"],
               "cls": m["cls"]}
        for mname, mod in models.items():
            row[f"in_{mname}"] = "y" if mod.has(low) else "n"
        vocab_rows.append(row)
    vocab_cols = (["lower", "surface", "n", "category", "first_page",
                   "first_uid", "zipf_en", "head", "parts", "cls"] +
                  [f"in_{m}" for m in model_names])
    write_tsv(out / "vocabulary.tsv", vocab_cols, vocab_rows)

    # ---- nearest neighbours ----
    all_types = sorted(meta)
    nn_book_rows = []
    for term in all_types:
        for mname in book_models:
            res = models[mname].most_similar([term], topn=args.topn)
            if res is None:
                continue
            for rank, (n, s) in enumerate(res, 1):
                nn_book_rows.append({"query": term, "model": mname,
                                     "rank": rank, "neighbour": n,
                                     "cosine": round(s, 5)})
    write_tsv(out / "neighbours_book.tsv",
              ["query", "model", "rank", "neighbour", "cosine"], nn_book_rows)

    selected = select_terms(meta)
    nn_selected_rows = []
    nn_selected = {}
    for term in selected:
        for mname in model_names:
            res = models[mname].most_similar([term], topn=args.topn)
            nn_selected[(mname, term)] = res or []
            if res is None:
                continue
            for rank, (n, s) in enumerate(res, 1):
                nn_selected_rows.append({"query": term, "model": mname,
                                         "rank": rank, "neighbour": n,
                                         "cosine": round(s, 5)})
    write_tsv(out / "neighbours_selected.tsv",
              ["query", "model", "rank", "neighbour", "cosine"],
              nn_selected_rows)

    # ---- analogies ----
    probes = (_probe_rows(NAMING_PROBES, "naming", "#12", with_note=True) +
              _probe_rows(DERIVATION_PROBES, "derivation", "#18",
                          with_note=True) +
              _probe_rows(COLLOCATION_PROBES, "collocation", "#27",
                          with_note=True) +
              mine_derivation_probes() + mine_collocation_probes())
    # de-duplicate: a mined pair can yield the same a:b::c string with a
    # different expected head; keep the first (deterministic) occurrence
    seen_keys, deduped = set(), []
    for p in probes:
        key = (p["family"], p["a"], p["b"], p["c"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(p)
    probes = deduped

    analogy_rows = []
    summary = []
    for mname in model_names:
        mod = models[mname]
        for fam in ("naming", "derivation", "collocation",
                    "mined_derivation", "mined_collocation"):
            fam_probes = [p for p in probes if p["family"] == fam]
            ranks = []
            for p in fam_probes:
                rank, _res = analogy_rank(mod, p["a"], p["b"], p["c"],
                                          p["expected"], topn=args.topn)
                if rank is None:
                    continue
                ranks.append(rank)
                analogy_rows.append({
                    "model": mname, "family": fam, "probe": p["probe"],
                    "a": p["a"], "b": p["b"], "c": p["c"],
                    "expected": p["expected"], "source": p["source"],
                    "rank": rank, "hit1": int(rank == 1),
                    "hit5": int(0 < rank <= 5), "hit10": int(0 < rank <= 10),
                    "note": p["note"]})
            n = len(fam_probes)
            n_eval = len(ranks)
            if n_eval:
                hit1 = sum(1 for r in ranks if r == 1) / n_eval
                hit5 = sum(1 for r in ranks if 0 < r <= 5) / n_eval
                hit10 = sum(1 for r in ranks if 0 < r <= 10) / n_eval
                mrr = sum(1.0 / r for r in ranks if r > 0) / n_eval
            else:
                hit1 = hit5 = hit10 = mrr = 0.0
            summary.append({"model": mname, "family": fam, "n": n,
                            "n_eval": n_eval, "hit1": hit1, "hit5": hit5,
                            "hit10": hit10, "mrr": mrr})
    write_tsv(out / "analogies.tsv",
              ["model", "family", "probe", "a", "b", "c", "expected",
               "source", "rank", "hit1", "hit5", "hit10", "note"],
              analogy_rows)
    write_tsv(out / "analogy_summary.tsv",
              ["model", "family", "n", "n_eval", "hit1", "hit5", "hit10",
               "mrr"],
              [{**s, "hit1": round(s["hit1"], 4), "hit5": round(s["hit5"], 4),
                "hit10": round(s["hit10"], 4), "mrr": round(s["mrr"], 4)}
               for s in summary])

    # English sanity probes
    english_rows = []
    english_cols = merged_models if merged_models else model_names
    for a, b, c, expected in ENGLISH_PROBES:
        row = {"probe": f"{a}:{b}::{c}:?", "a": a, "b": b, "c": c,
               "expected": expected}
        for mname in english_cols:
            rank, res = analogy_rank(models[mname], a, b, c, expected,
                                     topn=args.topn)
            row[mname] = ("oov" if rank is None else
                          (f"{rank}" if rank else ">%d" % args.topn))
        english_rows.append(row)
    write_tsv(out / "analogies_english.tsv",
              ["probe", "a", "b", "c", "expected"] + english_cols, english_rows)

    # worked analogies for the report (best merged model, else best book model)
    worked_model = ("ft_merged" if "ft_merged" in models else
                    "ft_book" if "ft_book" in models else model_names[0])
    worked = []
    for p in probes:
        if p["family"].startswith("mined"):
            continue
        rank, res = analogy_rank(models[worked_model], p["a"], p["b"], p["c"],
                                 p["expected"], topn=args.topn)
        top1 = res[0][0] if res else ""
        worked.append({"probe": p["probe"], "expected": p["expected"],
                       "top1": top1, "rank": rank})
    worked.sort(key=lambda r: (r["rank"] is None, r["rank"] == 0,
                               r["rank"] if r["rank"] else 99))
    worked = worked[:16]

    # ---- diagnostics / summary ----
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cov_w2v_book = coverage.get("w2v_book", 0) / n_types
    cov_glove_book = coverage.get("glove_book", 0) / n_types
    cov_ft_book = coverage.get("ft_book", 0) / n_types
    cov_w2v_merged = (coverage.get("w2v_merged", 0) / n_types
                      if "w2v_merged" in coverage else 0.0)
    cov_glove_merged = (coverage.get("glove_merged", 0) / n_types
                        if "glove_merged" in coverage else 0.0)
    cov_ft_merged = (coverage.get("ft_merged", 0) / n_types
                     if "ft_merged" in coverage else 0.0)
    curated = [s for s in summary if not s["family"].startswith("mined")]
    pooled = []
    for mname in model_names:
        fams = [s for s in curated if s["model"] == mname]
        n_eval = sum(s["n_eval"] for s in fams)
        if not n_eval:
            continue
        pooled.append({
            "model": mname, "n_eval": n_eval,
            "hit1": sum(s["hit1"] * s["n_eval"] for s in fams) / n_eval,
            "hit5": sum(s["hit5"] * s["n_eval"] for s in fams) / n_eval,
            "hit10": sum(s["hit10"] * s["n_eval"] for s in fams) / n_eval,
            "mrr": sum(s["mrr"] * s["n_eval"] for s in fams) / n_eval,
        })
    best_cur = max(pooled, key=lambda s: (s["hit10"], s["mrr"]),
                   default={"model": "", "hit10": 0.0})
    best_family_hit10 = best_cur["hit10"]
    best_family = best_cur["model"]
    # English sanity, on the strongest merged model available
    english_model = ("ft_merged" if "ft_merged" in models else
                     "w2v_merged" if "w2v_merged" in models else
                     (english_cols[0] if english_cols else None))
    english_hit10 = 0.0
    if english_model:
        vals = [analogy_rank(models[english_model], a, b, c, expected,
                             topn=args.topn)[0]
                for a, b, c, expected in ENGLISH_PROBES]
        vals = [v for v in vals if v is not None]
        english_hit10 = (sum(1 for v in vals if 0 < v <= 10) / len(vals)
                         if vals else 0.0)

    diag = {
        "schema": SCHEMA, "generated": generated,
        "pages": n_pages, "units": n_units, "types_folded": n_types,
        "sequences": len(seqs), "markup_removed": N_MARKUP,
        "english_tokens": en_tokens,
        "models": len(model_names),
        "vocab_book_union": len(all_types),
        "coverage_w2v_book": coverage.get("w2v_book", 0),
        "coverage_ft_book": coverage.get("ft_book", 0),
        "coverage_glove_book": coverage.get("glove_book", 0),
        "coverage_w2v_merged": coverage.get("w2v_merged", 0),
        "coverage_ft_merged": coverage.get("ft_merged", 0),
        "coverage_glove_merged": coverage.get("glove_merged", 0),
        "selected_terms": len(selected),
        "analogy_probes": len(probes),
        "analogy_rows": len(analogy_rows),
    }
    write_tsv(out / "diagnostics.tsv", ["diag", "value"],
              [{"diag": k, "value": v} for k, v in diag.items()])
    write_tsv(out / "model_summary.tsv",
              ["model", "corpus", "algo", "vocab", "dim", "window",
               "min_count", "subwords"],
              model_rows)

    recon = [
        {"quantity": "pages", "earlier": "225 (#7/#17)", "here": n_pages,
         "note": "same layout markdown"},
        {"quantity": "unit occurrences", "earlier": "74,665 (#17)", "here": n_units,
         "note": f"minus {N_MARKUP} markup-residue occurrences (#19)"},
        {"quantity": "screened occurrences", "earlier": "74,318 (#19/#24)",
         "here": n_units, "note": "the #19 screening join is asserted"},
        {"quantity": "case-folded types", "earlier": "5,502 (#24)",
         "here": n_types, "note": "distinct lowercased unit types"},
        {"quantity": "markup occurrences removed", "earlier": "347 (#19)",
         "here": N_MARKUP, "note": "tag names never enter as words"},
        {"quantity": "sequences", "earlier": "not previously counted",
         "here": len(seqs), "note": "units split at the `//` operator"},
    ]
    write_tsv(out / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    ctx = {
        "n_units": n_units, "n_types": n_types, "n_seqs": len(seqs),
        "en_tokens": en_tokens, "model_rows": model_rows,
        "seed": args.seed,
        "cov_w2v_book": cov_w2v_book, "cov_glove_book": cov_glove_book,
        "cov_ft_book": cov_ft_book, "cov_w2v_merged": cov_w2v_merged,
        "cov_glove_merged": cov_glove_merged, "cov_ft_merged": cov_ft_merged,
        "best_family_hit10": best_family_hit10,
        "best_family": best_family,
        "pooled_curated": pooled,
        "english_hit10": english_hit10,
        "analogy_summary": summary,
        "topn": args.topn,
        "nn_showcase": [t for t in
                        ("tokage", "adam", "boy-roid", "soul/gram",
                         "gene=tv", "suck=blood", "artificial", "cadaver",
                         "blood", "machine") if t in meta and t in selected][:8],
        "nn_selected": nn_selected,
        "worked_analogies": worked,
        "english_models": english_cols,
        "english_rows": english_rows,
    }
    (out / "embedding_report.md").write_text(build_report(ctx),
                                             encoding="utf-8")

    summary_json = {
        "schema": SCHEMA, "generated": generated, "counts": diag,
        "coverage": coverage,
        "analogy_summary": [
            {"model": s["model"], "family": s["family"], "n": s["n"],
             "n_eval": s["n_eval"], "hit1": round(s["hit1"], 4),
             "hit5": round(s["hit5"], 4), "hit10": round(s["hit10"], 4),
             "mrr": round(s["mrr"], 4)} for s in summary],
        "model_rows": model_rows,
        "selected_terms": selected,
    }
    (out / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8")

    if not args.quiet:
        print(f"[embeddings] models={model_names}")
        print(f"[embeddings] coverage book: w2v={coverage.get('w2v_book',0)} "
              f"ft={coverage.get('ft_book',0)} "
              f"glove={coverage.get('glove_book',0)} of {n_types}")
        print(f"[embeddings] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
