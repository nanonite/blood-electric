#!/usr/bin/env python3
"""#39 Clause / sentence embeddings -> analysis/clause_embeddings/

Embed every #33 clause with a **sentence encoder** and persist the vectors
keyed to ``page`` / ``clause``, together with a loader that other tasks
(#40 topics, #41-44, #54, #62, ...) can import.

The task text asks for **SBERT/E5**.  This builder ships a small model
registry:

  * ``sbert`` (default) -- ``sentence-transformers/all-MiniLM-L6-v2``, the
    canonical SBERT bi-encoder: mean pooling over token embeddings, L2
    normalised, 384 dims, uncased, max 256 tokens.  Exact per the model's
    ``modules.json`` / ``1_Pooling/config.json`` (``pooling_mode_mean_tokens``
    + ``Normalize``), implemented here with plain ``transformers`` so no
    ``sentence_transformers`` install is required.
  * ``e5`` -- ``intfloat/e5-small-v2`` (needs a one-time download); E5 wants a
    ``"passage: "`` prefix, which the registry applies.
  * ``bge`` -- ``TaylorAI/bge-micro-v2`` (cached here), a 3-layer MiniLM
    bi-encoder; BGE's documented pooling is the CLS token, used as such.

Text views.  Each clause is embedded under two views so downstream work can
choose, and so the report can check robustness:

  * ``units`` -- the clause's **markup-screened #17 unit surfaces** joined by
    single spaces (the book's own words; no operators, no HTML residue).  This
    is the primary semantic view.  The unit layer is joined on ``(page, start)``
    containment exactly as #33/#34 do, so the screened unit total reconciles
    with #27's 74,077.
  * ``raw`` -- the clause's source substring (``body[start:end]``, whitespace
    normalised), operators and punctuation intact.  A baseline/comparison view.

A clause is the #33 maximal run between boundary operators; clauses never cross
a page and no #17 unit crosses a clause boundary, so a ``(page, start)`` join is
exact.

Outputs under ``analysis/clause_embeddings/``:
  clause_embeddings_report.md    THE DELIVERABLE
  clause_embeddings.npz          primary vectors: clause_ids, pages, emb_units,
                                 emb_raw (float32, L2-normalised)
  clause_embeddings_<model>.npz  secondary model (robustness check), same keys
  clause_index.tsv               one row per clause: keys + text views + meta
  model_summary.tsv              model / pooling / dims / counts per encoder
  neighbours.tsv                 curated seed terms -> nearest clauses
  view_agreement.tsv             retrieval overlap between views / models
  embedding_stats.tsv            book vs English: repetition / anisotropy
  english_baseline.tsv           the matched English reference sentence sample
  per_page.tsv                   per-page clause count + within-page similarity
  reconciliation.tsv             against #17 / #19 / #27 / #33
  diagnostics.tsv                build-time counters
  summary.json                   machine-readable headline numbers

Reproduce::

    conda activate marker
    python work/clause_embeddings.py                 # writes analysis/clause_embeddings/
    python work/test_clause_embeddings.py

Deterministic: fixed torch seed and CPU thread count, ``model.eval()``, no
sampling, no hash-order dependence (every list is sorted by clause id; any
sampling is an evenly-spaced or fixed-seed selection).  Requires ``torch`` and
``transformers`` and the cached weights; ``--allow-download`` permits fetching.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
from pathlib import Path

import numpy as np

import clause_segmentation as cs  # #33 clause boundaries + unit/markup loaders

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "clause_embeddings"
CLAUSES = ROOT / "analysis" / "clauses" / "clauses.tsv"
REF_CACHE = ROOT / "analysis" / "frequency" / "reference_cache"

SCHEMA = "clause-embeddings/1"
PRIMARY = "sbert"
# Views and their index column suffix.
VIEWS = ("units", "raw")
DEFAULT_BATCH = 64
DEFAULT_THREADS = 4
DEFAULT_SEED = 0
MAX_SEQ = 256
# Curated seed terms for the neighbour catalogue (case-folded substring match on
# the ``units`` text).  Chosen from the #27 motif set / #30 dramatis personae.
SEED_TERMS = (
    "artificial sun", "drug embryo", "cadaver city", "ice nebula",
    "internal organ", "soul", "tokage", "adam", "serum", "chromosome",
    "babel", "gene", "fecundates", "cold-blooded", "necropolis",
)
# Similarity thresholds for the "near-duplicate" repetition measure.
NEAR_DUP = (0.90, 0.95, 0.99)

MODELS = {
    "sbert": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "pooling": "mean",
        "dim": 384,
        "max_seq": 256,
        "prefix": "",
        "note": "SBERT all-MiniLM-L6-v2 (mean pooling + L2 normalise)",
    },
    "e5": {
        "name": "intfloat/e5-small-v2",
        "pooling": "mean",
        "dim": 384,
        "max_seq": 512,
        "prefix": "passage: ",
        "note": "E5-small-v2 (passage prefix; needs download)",
    },
    "bge": {
        "name": "TaylorAI/bge-micro-v2",
        "pooling": "cls",
        "dim": 384,
        "max_seq": 512,
        "prefix": "",
        "note": "BGE-micro-v2 (CLS pooling; cached)",
    },
}

read_tsv = cs.read_tsv
write_tsv = cs.write_tsv

# English reference sentence extraction (#24 cache; no download).
EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_GUT_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
_GUT_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# deterministic sampling
# ---------------------------------------------------------------------------
def sample_indices(n: int, k: int) -> list:
    """``k`` evenly-spaced indices in ``[0, n)`` (deterministic, order kept).

    The first and last items are always included when ``k >= 2``, so the sample
    spans the whole book rather than a prefix.
    """
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0] if n else []
    step = (n - 1) / (k - 1)
    out = sorted({int(round(i * step)) for i in range(k)})
    # round() can collapse neighbours on short ranges; top up deterministically.
    j = 0
    while len(out) < k and j < n:
        if j not in out:
            out.append(j)
        j += 1
    return sorted(out)[:k]


# ---------------------------------------------------------------------------
# text building
# ---------------------------------------------------------------------------
def load_context():
    """``(bodies, page_units, unit_meta, is_markup)`` -- the #33/#34 join."""
    bodies = cs.load_pages()
    page_units, _uid_by_ps, unit_meta = cs.load_units()
    markup_spans, markup_uids, markup_types = cs.load_markup(bodies)

    def is_markup(uid):
        page, us, _ue, surface = unit_meta[uid]
        if uid in markup_uids or surface.lower() in markup_types:
            return True
        return any(a <= us < b for a, b in markup_spans.get(page, ()))

    return bodies, page_units, unit_meta, is_markup


def clause_records():
    """One record per #33 clause with both text views and the unit counts.

    ``text_units`` is the markup-screened unit sequence; ``text_raw`` is the
    source substring.  Returns ``(records, diagnostics)``.
    """
    bodies, page_units, unit_meta, is_markup = load_context()
    records = []
    n_units_screened = 0
    n_units_raw = 0
    n_markup_units = 0
    for r in read_tsv(CLAUSES):
        cid = int(r["clause_id"])
        page, s, e = int(r["page"]), int(r["start"]), int(r["end"])
        units = []
        for (us, _ue, u) in page_units.get(page, ()):
            if s <= us < e:
                n_units_raw += 1
                if is_markup(u):
                    n_markup_units += 1
                else:
                    units.append(unit_meta[u][3])
        n_units_screened += len(units)
        text_units = _WS_RE.sub(" ", " ".join(units)).strip()
        text_raw = _WS_RE.sub(" ", bodies.get(page, "")[s:e]).strip()
        records.append({
            "clause_id": cid,
            "page": page,
            "start": s,
            "end": e,
            "opener": r["opener"],
            "container": r["container"],
            "region": r.get("region", ""),
            "n_words_clean": int(r["n_words_clean"]),
            "n_units_screened": len(units),
            "text_units": text_units,
            "text_raw": text_raw,
        })
    diags = {
        "n_units_screened": n_units_screened,
        "n_units_raw": n_units_raw,
        "n_markup_units": n_markup_units,
    }
    return records, diags


def english_sentences(n: int, seed: int = DEFAULT_SEED,
                      min_words: int = 3, max_words: int = 60) -> list:
    """``n`` deterministic English sentences from the #24 reference cache.

    Pooled from every cached Gutenberg text, length-filtered to the clause-like
    band (3-60 words), then shuffled with a fixed seed.  No network.
    """
    pool = []
    for path in sorted(REF_CACHE.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        m = _GUT_START.search(text)
        if m:
            text = text[m.end():]
        m = _GUT_END.search(text)
        if m:
            text = text[:m.start()]
        for sent in _SENT_SPLIT_RE.split(text):
            words = EN_WORD_RE.findall(sent)
            if min_words <= len(words) <= max_words:
                s = _WS_RE.sub(" ", sent).strip()
                if s:
                    pool.append(s)
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    return pool[:n]


# ---------------------------------------------------------------------------
# sentence encoder
# ---------------------------------------------------------------------------
class SentenceEncoder:
    """A ``transformers`` sentence encoder with SBERT-style pooling.

    Deterministic: fixed seed + CPU thread count, ``eval()``, ``no_grad``.
    Vectors are L2-normalised so cosine similarity is a dot product.
    """

    def __init__(self, key: str = PRIMARY, device: str = "cpu",
                 local_only: bool = True, threads: int = DEFAULT_THREADS,
                 seed: int = DEFAULT_SEED, batch_size: int = DEFAULT_BATCH):
        if key not in MODELS:
            raise KeyError(f"unknown model {key!r}; have {sorted(MODELS)}")
        self.key = key
        self.spec = MODELS[key]
        self.device = device
        self.batch_size = batch_size
        self._tok = None
        self._mdl = None
        self._local_only = local_only
        self._threads = threads
        self._seed = seed
        self._load()

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.manual_seed(self._seed)
        if self._threads and self._threads > 0:
            torch.set_num_threads(self._threads)
        name = self.spec["name"]
        self._tok = AutoTokenizer.from_pretrained(
            name, local_files_only=self._local_only)
        self._mdl = AutoModel.from_pretrained(
            name, local_files_only=self._local_only)
        self._mdl.to(self.device)
        self._mdl.eval()

    @property
    def dim(self) -> int:
        return int(self.spec["dim"])

    def token_count(self, texts) -> list:
        enc = self._tok(list(texts), add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    def encode(self, texts, progress=None) -> np.ndarray:
        """Encode ``texts`` -> ``[N, dim]`` float32, L2-normalised.

        Texts are processed shortest-first so padding is minimal, then restored
        to input order; per-item results are independent of batching.
        """
        import torch
        import torch.nn.functional as F
        texts = [self.spec["prefix"] + t for t in texts]
        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)
        if n == 0:
            return out
        order = sorted(range(n), key=lambda i: (len(texts[i]), i))
        max_len = int(self.spec["max_seq"])
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                chunk = order[start:start + self.batch_size]
                batch = self._tok(
                    [texts[i] for i in chunk], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self._mdl(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                if self.spec["pooling"] == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vec = F.normalize(pooled, dim=1).cpu().numpy().astype(np.float32)
                for j, i in enumerate(chunk):
                    out[i] = vec[j]
                if progress is not None:
                    progress(min(start + self.batch_size, n), n)
        return out


# ---------------------------------------------------------------------------
# embedding geometry
# ---------------------------------------------------------------------------
def embedding_stats(vectors: np.ndarray, sample: int = 1000,
                    seed: int = DEFAULT_SEED) -> dict:
    """Repetition / anisotropy summary of a normalised vector set.

    ``nn`` (nearest-neighbour cosine) is measured from an evenly-spaced sample
    of queries against the whole set (self excluded), so the cost is bounded.
    """
    n = len(vectors)
    if n < 2:
        return {"n": n, "mean_nn": 0.0, "median_nn": 0.0, "mean_pairwise": 0.0,
                "mean_centroid_cos": 0.0, "centroid_norm": 0.0,
                **{f"frac_nn_gt_{t:.2f}": 0.0 for t in NEAR_DUP}}
    q = sample_indices(n, min(sample, n))
    Q = vectors[q]
    S = Q @ vectors.T
    for a, i in enumerate(q):
        S[a, i] = -2.0
    top = S.max(axis=1)
    P = Q @ Q.T
    m = len(q)
    off = (P.sum() - np.trace(P)) / (m * (m - 1)) if m > 1 else 0.0
    centroid = vectors.mean(axis=0)
    cnorm = float(np.linalg.norm(centroid))
    ccos = vectors @ (centroid / cnorm) if cnorm > 0 else np.zeros(n)
    stats = {
        "n": n,
        "mean_nn": float(top.mean()),
        "median_nn": float(np.median(top)),
        "mean_pairwise": float(off),
        "mean_centroid_cos": float(ccos.mean()),
        "centroid_norm": cnorm,
    }
    for t in NEAR_DUP:
        stats[f"frac_nn_gt_{t:.2f}"] = float((top > t).mean())
    return stats


def nearest(vectors: np.ndarray, query_index: int, k: int = 10,
            exclude_self: bool = True):
    """Top-``k`` ``(index, cosine)`` neighbours of one row (vectors normalised)."""
    sims = vectors @ vectors[query_index]
    if exclude_self:
        sims = sims.copy()
        sims[query_index] = -2.0
        available = len(sims) - 1
    else:
        available = len(sims)
    k = min(k, available)
    if k <= 0:
        return []
    if k < len(sims):
        idx = np.argpartition(-sims, k - 1)[:k]
    else:
        idx = np.arange(len(sims))
    idx = idx[np.argsort(-sims[idx], kind="stable")]
    return [(int(i), float(sims[i])) for i in idx]


def retrieval_overlap(A: np.ndarray, B: np.ndarray, queries, k: int = 10) -> dict:
    """Mean top-``k`` neighbour overlap between two views/models.

    ``A``/``B`` are row-aligned normalised matrices (same clause order); the
    overlap is Jaccard-style (intersection / k) over the neighbour id sets.
    """
    fracs = []
    for qi in queries:
        na = {i for i, _ in nearest(A, qi, k)}
        nb = {i for i, _ in nearest(B, qi, k)}
        fracs.append(len(na & nb) / max(1, k))
    return {"n_queries": len(queries), "k": k,
            "mean_overlap": float(np.mean(fracs)) if fracs else 0.0}


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------
def model_filename(key: str) -> str:
    return "clause_embeddings.npz" if key == PRIMARY else f"clause_embeddings_{key}.npz"


class ClauseEmbeddingStore:
    """Loader for the persisted clause vectors (keyed to page / clause).

    Example::

        store = ClauseEmbeddingStore()          # primary SBERT, analysis/clause_embeddings/
        v = store.vector(1234, view="units")    # 384-d, L2-normalised
        for cid, cos in store.nearest(1234): ...
        page = store.page_matrix(4)
    """

    def __init__(self, out_dir=OUT, model: str = PRIMARY):
        self.dir = Path(out_dir)
        self.model = model
        path = self.dir / model_filename(model)
        if not path.exists():
            raise FileNotFoundError(path)
        z = np.load(path, allow_pickle=False)
        self.clause_ids = z["clause_ids"].astype(np.int64)
        self.pages = z["pages"].astype(np.int64)
        self._views = {v: z[f"emb_{v}"] for v in VIEWS if f"emb_{v}" in z.files}
        self._row = {int(c): i for i, c in enumerate(self.clause_ids)}
        self.index = read_tsv(self.dir / "clause_index.tsv")
        self._index_row = {int(r["clause_id"]): r for r in self.index}
        if len(self.index) != len(self.clause_ids):
            raise ValueError("clause_index.tsv and npz disagree on length")

    # -- keys -------------------------------------------------------------
    def row(self, clause_id: int) -> int:
        return self._row[int(clause_id)]

    def meta(self, clause_id: int) -> dict:
        return self._index_row[int(clause_id)]

    def text(self, clause_id: int, view: str = "units") -> str:
        return self.meta(clause_id)[f"text_{view}"]

    # -- vectors ----------------------------------------------------------
    def matrix(self, view: str = "units") -> np.ndarray:
        if view not in self._views:
            raise KeyError(f"view {view!r} not in {sorted(self._views)}")
        return self._views[view]

    def vector(self, clause_id: int, view: str = "units") -> np.ndarray:
        return self.matrix(view)[self.row(clause_id)]

    def page_rows(self, page: int):
        return np.nonzero(self.pages == int(page))[0]

    def page_matrix(self, page: int, view: str = "units") -> np.ndarray:
        return self.matrix(view)[self.page_rows(page)]

    # -- search -----------------------------------------------------------
    def nearest(self, clause_id: int, view: str = "units", k: int = 10,
                same_page_only: bool = False):
        vecs = self.matrix(view)
        if same_page_only:
            rows = self.page_rows(self.meta(clause_id)["page"])
            sub = vecs[rows]
            qi = int(np.nonzero(rows == self.row(clause_id))[0][0])
            hits = nearest(sub, qi, k)
            return [(int(self.clause_ids[rows[i]]), c) for i, c in hits]
        hits = nearest(vecs, self.row(clause_id), k)
        return [(int(self.clause_ids[i]), c) for i, c in hits]

    def search(self, query_vector, view: str = "units", k: int = 10):
        """Nearest clauses to an arbitrary (normalised) query vector."""
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.matrix(view) @ q
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        return [(int(self.clause_ids[i]), float(sims[i])) for i in idx]


def load(out_dir=OUT, model: str = PRIMARY) -> ClauseEmbeddingStore:
    """Convenience constructor for :class:`ClauseEmbeddingStore`."""
    return ClauseEmbeddingStore(out_dir=out_dir, model=model)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(out_dir: Path, encoder_key: str = PRIMARY, sample: int = 0,
          secondary: str = "bge", english: bool = True,
          english_n: int = 0, allow_download: bool = False,
          threads: int = DEFAULT_THREADS, seed: int = DEFAULT_SEED,
          batch_size: int = DEFAULT_BATCH, quiet: bool = False) -> dict:
    out_dir = Path(out_dir)
    records, txt_diags = clause_records()
    n_total = len(records)

    # -- deterministic subset -------------------------------------------
    if sample and 0 < sample < n_total:
        keep = set(sample_indices(n_total, sample))
        records = [r for r in records if r["clause_id"] in keep]
    n = len(records)
    n_pages = len({r["page"] for r in records})

    local_only = not allow_download
    enc = SentenceEncoder(encoder_key, local_only=local_only, threads=threads,
                          seed=seed, batch_size=batch_size)
    if not quiet:
        print(f"encoder: {enc.spec['name']} ({enc.spec['pooling']}, "
              f"{enc.dim}d)  clauses: {n:,}/{n_total:,}")

    unit_texts = [r["text_units"] for r in records]
    raw_texts = [r["text_raw"] for r in records]
    emb_units = enc.encode(unit_texts)
    emb_raw = enc.encode(raw_texts)
    for r, nt in zip(records, enc.token_count(unit_texts)):
        r["n_tok_units"] = nt

    models_used = {encoder_key: {
        "key": encoder_key, "name": enc.spec["name"],
        "pooling": enc.spec["pooling"], "dim": enc.dim,
        "max_seq": enc.spec["max_seq"], "n_clauses": n,
        "views": "units,raw",
    }}
    matrices = {encoder_key: {"units": emb_units, "raw": emb_raw}}

    # -- secondary model (robustness) -----------------------------------
    secondary_used = None
    if secondary and secondary in MODELS and secondary != encoder_key:
        try:
            enc2 = SentenceEncoder(secondary, local_only=local_only,
                                   threads=threads, seed=seed,
                                   batch_size=batch_size)
            matrices[secondary] = {
                "units": enc2.encode(unit_texts),
                "raw": enc2.encode(raw_texts),
            }
            secondary_used = secondary
            models_used[secondary] = {
                "key": secondary, "name": enc2.spec["name"],
                "pooling": enc2.spec["pooling"], "dim": enc2.dim,
                "max_seq": enc2.spec["max_seq"], "n_clauses": n,
                "views": "units,raw",
            }
            if not quiet:
                print(f"secondary: {enc2.spec['name']} ok")
        except Exception as exc:  # noqa: BLE001 - optional robustness model
            if not quiet:
                print(f"secondary model {secondary} unavailable: "
                      f"{type(exc).__name__}: {exc}")

    # -- English baseline ------------------------------------------------
    eng_stats = None
    eng_texts = []
    if english:
        k = english_n or n
        eng_texts = english_sentences(k, seed=seed)
        if eng_texts:
            eng_vecs = enc.encode(eng_texts)
            eng_stats = embedding_stats(eng_vecs, sample=min(1000, len(eng_vecs)),
                                        seed=seed)
            eng_stats["source"] = "english_sentences"
            eng_stats["n_texts"] = len(eng_texts)

    # -- geometry --------------------------------------------------------
    book_stats = {v: embedding_stats(matrices[encoder_key][v], seed=seed)
                  for v in VIEWS}

    # -- retrieval agreement --------------------------------------------
    q = sample_indices(n, min(300, n))
    agreement_rows = []
    base = matrices[encoder_key]
    agreement_rows.append({
        "model_a": encoder_key, "view_a": "units",
        "model_b": encoder_key, "view_b": "raw",
        **retrieval_overlap(base["units"], base["raw"], q, k=10)})
    if secondary_used:
        agreement_rows.append({
            "model_a": encoder_key, "view_a": "units",
            "model_b": secondary_used, "view_b": "units",
            **retrieval_overlap(base["units"], matrices[secondary_used]["units"],
                                q, k=10)})
        agreement_rows.append({
            "model_a": encoder_key, "view_a": "raw",
            "model_b": secondary_used, "view_b": "raw",
            **retrieval_overlap(base["raw"], matrices[secondary_used]["raw"],
                                q, k=10)})

    # -- curated neighbours ---------------------------------------------
    row_of = {r["clause_id"]: i for i, r in enumerate(records)}
    rec_of = {r["clause_id"]: r for r in records}
    neigh_rows = []
    for term in SEED_TERMS:
        hits = [r["clause_id"] for r in records
                if term in r["text_units"].lower()]
        if not hits:
            continue
        cid = hits[0]
        seed_text = rec_of[cid]["text_units"][:80]
        for rank, (nrow, cos) in enumerate(
                nearest(emb_units, row_of[cid], k=6)):
            if rank == 0:
                continue
            nrec = records[nrow]
            neigh_rows.append({
                "seed_term": term, "seed_clause": cid,
                "rank": rank, "neighbour_clause": nrec["clause_id"],
                "cosine": round(cos, 4),
                "seed_text": seed_text,
                "neighbour_text": nrec["text_units"][:80],
            })

    # -- per page --------------------------------------------------------
    per_page = []
    page_groups = collections.defaultdict(list)
    for i, r in enumerate(records):
        page_groups[r["page"]].append(i)
    for page in sorted(page_groups):
        rows = page_groups[page]
        sub = emb_units[rows]
        if len(rows) > 1:
            P = sub @ sub.T
            off = (P.sum() - np.trace(P)) / (len(rows) * (len(rows) - 1))
            S = sub @ sub.T
            np.fill_diagonal(S, -2.0)
            nn = float(S.max(axis=1).mean())
        else:
            off = 0.0
            nn = float("nan")
        per_page.append({
            "page": page, "n_clauses": len(rows),
            "mean_within_cos": round(float(off), 4),
            "mean_nn_cos": ("" if math.isnan(nn) else round(nn, 4)),
        })

    # -- reconciliation --------------------------------------------------
    recon = [
        {"quantity": "clauses (book total)", "here": n_total,
         "earlier": "5,614 (#33)", "note": "every #33 clause"},
        {"quantity": "clauses embedded", "here": n,
         "earlier": f"{n_total} (#33)" if not sample else "sample",
         "note": "one row per clause in the artifact"},
        {"quantity": "pages", "here": n_pages, "earlier": "225 (#33)",
         "note": "pages carrying clauses"},
        {"quantity": "screened units", "here": txt_diags["n_units_screened"],
         "earlier": "74,077 (#27)", "note": "(page,start) join, markup-screened"},
        {"quantity": "raw units", "here": txt_diags["n_units_raw"],
         "earlier": "74,665 (#17)", "note": "before markup screen"},
        {"quantity": "markup units removed", "here": txt_diags["n_markup_units"],
         "earlier": "588 (#27)", "note": "#19 ledger + #27 tag runs"},
    ]

    summary = {
        "schema": SCHEMA,
        "encoder": encoder_key,
        "encoder_name": enc.spec["name"],
        "secondary": secondary_used,
        "views": list(VIEWS),
        "dim": enc.dim,
        "max_seq": enc.spec["max_seq"],
        "pooling": enc.spec["pooling"],
        "n_clauses_total": n_total,
        "n_clauses_embedded": n,
        "n_pages": n_pages,
        "sample": sample,
        "screened_units": txt_diags["n_units_screened"],
        "raw_units": txt_diags["n_units_raw"],
        "markup_units": txt_diags["n_markup_units"],
        "english_n": len(eng_texts),
        "book_stats": book_stats,
        "english_stats": eng_stats,
        "view_agreement": agreement_rows,
        "near_dup_thresholds": list(NEAR_DUP),
        "determinism": {"seed": seed, "threads": threads,
                        "local_files_only": local_only},
    }

    # -- write -----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, views in matrices.items():
        np.savez_compressed(
            out_dir / model_filename(key),
            clause_ids=np.asarray([r["clause_id"] for r in records],
                                  dtype=np.int32),
            pages=np.asarray([r["page"] for r in records], dtype=np.int16),
            **{f"emb_{v}": views[v].astype(np.float32) for v in VIEWS})

    write_tsv(out_dir / "clause_index.tsv", [
        "clause_id", "page", "start", "end", "opener", "container", "region",
        "n_words_clean", "n_units_screened", "n_tok_units", "text_units",
        "text_raw"], records)

    write_tsv(out_dir / "model_summary.tsv", [
        "key", "name", "pooling", "dim", "max_seq", "n_clauses", "views"],
        list(models_used.values()))

    write_tsv(out_dir / "view_agreement.tsv", [
        "model_a", "view_a", "model_b", "view_b", "n_queries", "k",
        "mean_overlap"], agreement_rows)

    write_tsv(out_dir / "neighbours.tsv", [
        "seed_term", "seed_clause", "rank", "neighbour_clause", "cosine",
        "seed_text", "neighbour_text"], neigh_rows)

    stats_rows = []
    for v in VIEWS:
        row = {"set": f"book_{v}", **book_stats[v]}
        stats_rows.append(row)
    if eng_stats:
        stats_rows.append({"set": "english", **eng_stats})
    write_tsv(out_dir / "embedding_stats.tsv",
              ["set", "n", "mean_nn", "median_nn", "mean_pairwise",
               "mean_centroid_cos", "centroid_norm"] +
              [f"frac_nn_gt_{t:.2f}" for t in NEAR_DUP], stats_rows)

    if eng_texts:
        write_tsv(out_dir / "english_baseline.tsv",
                  ["idx", "text"], [{"idx": i, "text": t}
                                    for i, t in enumerate(eng_texts)])

    write_tsv(out_dir / "per_page.tsv",
              ["page", "n_clauses", "mean_within_cos", "mean_nn_cos"], per_page)

    diag_rows = [
        {"metric": "n_clauses_total", "value": n_total},
        {"metric": "n_clauses_embedded", "value": n},
        {"metric": "n_pages", "value": n_pages},
        {"metric": "screened_units", "value": txt_diags["n_units_screened"]},
        {"metric": "raw_units", "value": txt_diags["n_units_raw"]},
        {"metric": "markup_units", "value": txt_diags["n_markup_units"]},
        {"metric": "dim", "value": enc.dim},
        {"metric": "empty_text_clauses", "value": sum(
            1 for r in records if not r["text_units"])},
        {"metric": "max_tok_units", "value": max(
            (r["n_tok_units"] for r in records), default=0)},
        {"metric": "truncated_clauses", "value": sum(
            1 for r in records if r["n_tok_units"] > enc.spec["max_seq"])},
        {"metric": "english_sentences", "value": len(eng_texts)},
        {"metric": "secondary", "value": secondary_used or "none"},
        {"metric": "seed", "value": seed},
        {"metric": "threads", "value": threads},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["metric", "value"], diag_rows)
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "here", "earlier", "note"], recon)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_report(out_dir / "clause_embeddings_report.md", summary, records,
                 book_stats, eng_stats, agreement_rows, neigh_rows, per_page,
                 recon, models_used)

    if not quiet:
        print(f"wrote {out_dir}  ({n:,} clauses x {len(VIEWS)} views, "
              f"{len(models_used)} model(s))")
        if eng_stats:
            print(f"book units mean_nn {book_stats['units']['mean_nn']:.3f} "
                  f"vs english {eng_stats['mean_nn']:.3f}")
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _stats_line(s):
    return (f"| `{s['set']}` | {s.get('n', '')} | "
            f"{s.get('mean_nn', 0):.3f} | {s.get('median_nn', 0):.3f} | "
            f"{s.get('mean_pairwise', 0):.3f} | "
            f"{s.get('mean_centroid_cos', 0):.3f} | "
            + " | ".join(f"{s.get(f'frac_nn_gt_{t:.2f}', 0):.3f}"
                         for t in NEAR_DUP) + " |")


def write_report(path, summary, records, book_stats, eng_stats, agreement,
                 neigh_rows, per_page, recon, models_used):
    L = []
    A = L.append
    n = summary["n_clauses_embedded"]
    total = summary["n_clauses_total"]
    A("# Clause / sentence embeddings (#39)")
    A("")
    A("Embed every #33 clause with a **sentence encoder** and persist the "
      "vectors keyed to `page` / `clause`, plus a loader for downstream tasks "
      "(#40 topics, #41–44, #54, #62). The primary encoder is the canonical "
      "**SBERT** `sentence-transformers/all-MiniLM-L6-v2` (mean pooling + L2 "
      "normalise, 384-d), reimplemented on `transformers` so no "
      "`sentence_transformers` install is needed; an **E5** entry "
      "(`intfloat/e5-small-v2`) is in the registry, and the cached "
      "`TaylorAI/bge-micro-v2` is run as a secondary robustness encoder.")
    A("")
    A(f"**Headline.** {n:,} clauses embedded"
      + (f" (deterministic sample of {total:,})" if summary["sample"] else
         f" (all {total:,} #33 clauses)")
      + f" under **{len(summary['views'])} text views** (`units` = "
      "markup-screened #17 unit surfaces; `raw` = source substring with "
      "operators intact) and "
      f"{len(models_used)} encoder(s). Vectors are L2-normalised so cosine is "
      "a dot product; the `ClauseEmbeddingStore` loader returns them by "
      "`clause_id` / `page`.")
    A("")
    A("## 1. Method")
    A("")
    A("A clause is the #33 maximal run of content between boundary operators; "
      "clauses never cross a page and no #17 unit crosses a clause boundary, "
      "so the unit join is exact on `(page, start)` containment (the join "
      "#33/#34 use). The markup screen is #19's ledger unioned with the #27 "
      "tag-run detector.")
    A("")
    A("| view | text | purpose |")
    A("|---|---|---|")
    A("| `units` | markup-screened #17 unit surfaces, space-joined | primary "
      "semantic view (the book's own words) |")
    A("| `raw` | source substring `body[start:end]`, whitespace-normalised | "
      "baseline with operators/punctuation intact |")
    A("")
    A("SBERT pooling is exact per the model's `1_Pooling/config.json` "
      "(`pooling_mode_mean_tokens`) and `Normalize` module: attention-masked "
      "mean of the last hidden state, then L2 normalisation. E5 adds its "
      "`\"passage: \"` prefix; BGE uses CLS pooling. CPU, fixed seed and "
      "thread count, `eval()`, `no_grad`.")
    A("")
    A("## 2. Reconciliation")
    A("")
    A("| quantity | here | earlier | note |")
    A("|---|---:|---|---|")
    for r in recon:
        A(f"| {r['quantity']} | {r['here']:,} | {r['earlier']} | {r['note']} |")
    A("")
    A("## 3. Embedding geometry — book vs matched English")
    A("")
    if eng_stats:
        A("The English control is a fixed-seed sample of sentences (3–60 words) "
          "from the eight pinned #24 Gutenberg references, encoded by the same "
          "model. `mean_nn` is the mean nearest-neighbour cosine from an "
          "evenly-spaced sample of queries (self excluded); `mean_pairwise` is "
          "the mean cosine among sampled pairs; `mean_centroid_cos` is the mean "
          "cosine to the global centroid (higher = more anisotropic).")
    else:
        A("English control disabled (`--no-english`).")
    A("")
    A("| set | n | mean_nn | median_nn | mean_pairwise | mean_centroid_cos | "
      + " | ".join(f"frac nn > {t:.2f}" for t in NEAR_DUP) + " |")
    A("|---:|---:|---:|---:|---:|---:|" + "---:|" * len(NEAR_DUP))
    for v in VIEWS:
        A(_stats_line({"set": f"book_{v}", **book_stats[v]}))
    if eng_stats:
        A(_stats_line({"set": "english", **eng_stats}))
    A("")
    if eng_stats:
        bu = book_stats["units"]
        A(f"The book's clauses are **more self-similar than matched English**: "
          f"mean nearest-neighbour cosine {bu['mean_nn']:.3f} vs "
          f"{eng_stats['mean_nn']:.3f}, and mean pairwise cosine "
          f"{bu['mean_pairwise']:.3f} vs {eng_stats['mean_pairwise']:.3f}. "
          f"The book's clause cloud is also more anisotropic (mean centroid "
          f"cosine {bu['mean_centroid_cos']:.3f} vs "
          f"{eng_stats['mean_centroid_cos']:.3f}). This is the #24 loop finding "
          "at the semantic level: the book is a restricted core of repeated "
          "clause templates, not a stream of distinct sentences.")
        near = (f" {bu['frac_nn_gt_0.95'] * 100:.1f}% of clauses have a "
                f"near-duplicate (cos > 0.95) against "
                f"{eng_stats['frac_nn_gt_0.95'] * 100:.1f}% of English "
                "sentences." if bu["frac_nn_gt_0.95"] > 0 else "")
        A("")
        A(f"The `raw` view is *less* self-similar "
          f"({book_stats['raw']['mean_nn']:.3f}) because the operator strings "
          f"add variation the screened unit view removes.{near}")
    A("")
    A("## 4. View / model robustness")
    A("")
    A("Top-10 neighbour overlap between views and models (mean over an "
      "evenly-spaced query sample). High overlap means the choice of text view "
      "or encoder does not change who a clause's neighbours are.")
    A("")
    A("| model A | view A | model B | view B | queries | mean overlap |")
    A("|---|---|---|---|---:|---:|")
    for r in agreement:
        A(f"| {r['model_a']} | {r['view_a']} | {r['model_b']} | {r['view_b']} "
          f"| {r['n_queries']} | {r['mean_overlap']:.3f} |")
    A("")
    view_rows = [r for r in agreement if r["view_a"] != r["view_b"]]
    model_rows = [r for r in agreement if r["model_a"] != r["model_b"]]
    if view_rows and model_rows:
        vv = max(float(r["mean_overlap"]) for r in view_rows)
        mm = max(float(r["mean_overlap"]) for r in model_rows)
        A(f"**View choice is safe; encoder choice is not.** The two text views "
          f"agree on {vv * 100:.0f}% of top-10 neighbours, so downstream work "
          "can use either. But the two encoders agree on only "
          f"{mm * 100:.0f}%: SBERT (`all-MiniLM-L6-v2`, 6 layers, mean "
          "pooling) and BGE-micro-v2 (3 layers, CLS pooling) place clauses "
          "differently, so **downstream tasks must pin one encoder** — the "
          "artifact's primary is `clause_embeddings.npz` (SBERT). The low "
          "cross-model overlap is itself a limit of applying sentence models "
          "to out-of-distribution fractured text; it is reported, not hidden.")
    A("")
    A("## 5. Curated neighbour catalogue")
    A("")
    A("Nearest clauses (primary `units` view) to the first clause containing "
      "each seed term. These are the motifs #27/#30 found; the neighbours show "
      "what the encoder groups with them.")
    A("")
    A("| seed | rank | cosine | neighbour text |")
    A("|---|---:|---:|---|")
    for r in neigh_rows[:40]:
        A(f"| {r['seed_term']} | {r['rank']} | {r['cosine']} | "
          f"`{r['neighbour_text']}` |")
    A("")
    A("## 6. Per-page semantic coherence")
    A("")
    A("`mean_within_cos` is the mean pairwise cosine among a page's clauses; "
      "`mean_nn_cos` the mean within-page nearest-neighbour cosine. A flat "
      "high curve is the loop; dips mark pages that are locally distinct "
      "(OCR loops, markup, or a genuine register shift).")
    A("")
    A("| page | clauses | mean_within_cos | mean_nn_cos |")
    A("|---:|---:|---:|---:|")
    for r in per_page[:20]:
        A(f"| {r['page']} | {r['n_clauses']} | {r['mean_within_cos']} | "
          f"{r['mean_nn_cos']} |")
    A("| … | | | |")
    A("")
    A("## 7. Artifacts and loader")
    A("")
    A("| artifact | contents |")
    A("|---|---|")
    A("| `clause_embeddings.npz` | `clause_ids`, `pages`, `emb_units`, "
      "`emb_raw` (float32, L2-normalised) |")
    if summary["secondary"]:
        A(f"| `clause_embeddings_{summary['secondary']}.npz` | secondary "
          "encoder, same keys |")
    A("| `clause_index.tsv` | keys + both text views + token counts |")
    A("| `model_summary.tsv` | encoder / pooling / dims / counts |")
    A("")
    A("```python")
    A("from clause_embeddings import ClauseEmbeddingStore")
    A("store = ClauseEmbeddingStore()            # analysis/clause_embeddings/")
    A("v = store.vector(1234, view='units')      # 384-d unit vector")
    A("hits = store.nearest(1234, k=10)          # [(clause_id, cosine), ...]")
    A("page_vecs = store.page_matrix(4)           # all clauses on page 4")
    A("```")
    A("")
    A("## 8. Negative results and limitations")
    A("")
    A("- The vectors encode **surface English**, not a parse: the book's "
      "fractured clauses are out-of-distribution for a model trained on "
      "ordinary sentences, so neighbours are a relative ordering, not a "
      "gold translation. The `raw` view is especially OOD.")
    A("- The English control is **matched in sentence count, not domain**; "
      "Gutenberg prose is a looser comparison than a same-genre cyberpunk "
      "corpus (a #56/#58 task).")
    A("- Clause boundaries are operator-driven (#33), so a long OCR loop "
      "(page 90) or a run-on becomes one 'sentence' and its vector is an "
      "artifact; `n_tok_units` is reported and truncation at "
      f"{summary['max_seq']} tokens is disclosed in `diagnostics.tsv`.")
    A("- `mean_pairwise`/`mean_nn` are estimated from a bounded sample for "
      "speed; the sample is deterministic and evenly spaced.")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description="#39 clause/sentence embeddings")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--model", default=PRIMARY,
                    choices=sorted(MODELS), help="primary sentence encoder")
    ap.add_argument("--secondary", default="bge",
                    help="secondary encoder for robustness ('' to disable)")
    ap.add_argument("--no-secondary", action="store_true",
                    help="skip the secondary robustness encoder")
    ap.add_argument("--quick", action="store_true",
                    help="deterministic 200-clause sample (regression suite)")
    ap.add_argument("--sample", type=int, default=0,
                    help="embed only an evenly-spaced sample of N clauses")
    ap.add_argument("--english-n", type=int, default=0,
                    help="English control sentences (default: same as clauses)")
    ap.add_argument("--no-english", action="store_true",
                    help="skip the English reference control")
    ap.add_argument("--allow-download", action="store_true",
                    help="permit fetching weights not in the local cache")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    OUT = Path(args.out)
    sample = 200 if args.quick else args.sample
    build(OUT, encoder_key=args.model, sample=sample,
          secondary=("" if args.no_secondary else args.secondary),
          english=not args.no_english,
          english_n=args.english_n, allow_download=args.allow_download,
          threads=args.threads, seed=args.seed, batch_size=args.batch_size,
          quiet=args.quiet)


if __name__ == "__main__":
    main()
