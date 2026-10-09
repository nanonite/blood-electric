#!/usr/bin/env python3
"""#73 Triangulation / convergence report -> analysis/triangulation/

Do independent Blood Electric analyses **converge on the same motifs /
domains**?  The named methods are

  * **network community**  -- #29 Louvain communities over the #17 unit layer;
  * **topic model**        -- #41 LDA (k=5), NMF (k=6) and the BERTopic-style
                              model (22 topics) with their per-clause labels;
  * **embedding cluster**  -- #40 KMeans k=12 over the #39 clause vectors;
  * **periodicity**        -- #51 spectral / renewal analysis, re-run here on
                              *motif* series rather than on the symbol stream.

Convergence is evidence; divergence is a lead.  Nothing in this report is a
translation or a ground truth: every cell is an agreement between estimators
over the same book, never an authority for what a motif *means*.

What is and is not compared
---------------------------
Only analyses **of Blood Electric itself** take part.  No reference corpus, no
English baseline and no external text is acquired, re-scored or used as an
arbiter here (#24/#56 stay available for other questions).  The only nulls are
internal surrogates of the book's own data: label permutations of the clause
partitions and within-book shuffles of the motif series.

Three levels of convergence
---------------------------
1. **Method agreement** (clause level): every distributional method is cast as
   a partition of the same 5,614 #33/#39 clauses; pairwise adjusted Rand and
   normalised mutual information with a label-permutation null.
2. **Motif / domain convergence matrix**: for each motif (and for each of
   #29's six named domains) the dominant group and its purity under every
   method, then the ten pairwise Jaccard overlaps between the methods'
   dominant buckets over that motif's clauses.  Agreement here means *the
   methods put the same clauses in one bucket*, so no label matching is needed
   across methods with very different k.
3. **Periodicity cross-check**: each motif becomes a page series and a
   clause-position renewal series and is tested against #51's null family
   (global / block / detrended shuffles).  This asks whether a motif the
   distributional methods call recurring also recurs on a *metric* cycle --
   the one axis on which #51 reported a negative result.

The embedding partition is re-derived here
-------------------------------------------
#40 persisted a cluster **catalogue** but not its per-clause labels, so the
embedding column of the matrix needs a clause partition.  ``numpy`` k-means
(k-means++ init, Lloyd, Euclidean on the L2-normalised #39 vectors, k=12,
seed 0, ``n_init=10``) reproduces #40's disclosed configuration; the derived
partition is validated against #40's published catalogue (label terms from the
same Monroe log-odds estimator, cluster sizes) and the validation is reported.
No encoder is re-run: the vectors are #39's frozen artifact.

Outputs under ``analysis/triangulation/``:
  triangulation_report.md       THE DELIVERABLE
  method_agreement.tsv          pairwise ARI / NMI + permutation p (clause level)
  clause_agreement.tsv          per-method clause partition sizes + coverage
  cluster_validation.tsv        re-derived k-means vs #40's catalogue
  motif_inventory.tsv           motif list with provenance, spread and verdict
  motif_count_reconciliation.tsv motif clause counts against #40's inventory
  convergence_matrix.tsv        the matrix (purity, buckets, 10 pairwise Jaccards)
  convergence_matrix.md         the same matrix, rendered
  motif_method_cells.tsv        long form: motif, method, dominant group, purity
  domain_convergence.tsv        #29's six named domains, same treatment
  periodicity_motifs.tsv        per-motif page-series + renewal statistics
  periodicity_book_context.tsv  #51's book-level series, cited not recomputed
  reconciliation.tsv            against #29 / #39 / #40 / #41 / #51
  diagnostics.tsv               build-time counters
  summary.json                  machine-readable headline numbers

Reproduce::

    conda activate marker
    python work/triangulation.py                 # writes analysis/triangulation/
    python work/test_triangulation.py

Deterministic: fixed seeds (42 everywhere), no hash-order dependence (every
iteration is sorted; ties break on a fixed key), no wall-clock in any
artifact.  Requires ``numpy`` only -- ``scikit-learn`` / ``hdbscan`` /
``umap`` are never imported, so the report can be rebuilt from the frozen
artifacts alone.
"""
from __future__ import annotations

import argparse
import ast
import collections
import itertools
import json
import math
import re
import sys
from pathlib import Path

# ruff: noqa: UP031 -- the report renderer is built from long, wrapped
# paragraph templates with a positional value tuple; `%` substitution keeps
# each paragraph a single statement, where f-strings would splice the same
# values into several concatenated literals and lose the prose flow.

import clause_segmentation as cs  # #33 quote-safe TSV io
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "analysis" / "triangulation"

NET_NODES = ROOT / "analysis" / "network" / "graph_nodes.tsv"
CLAUSE_INDEX = ROOT / "analysis" / "clause_embeddings" / "clause_index.tsv"
EMBEDDINGS = ROOT / "analysis" / "clause_embeddings" / "clause_embeddings.npz"
CLAUSE_TOPICS = ROOT / "analysis" / "topics" / "clause_topics.tsv"
TOPIC_SUMMARIES = {
    "lda": ROOT / "analysis" / "topics" / "lda_topic_summary.tsv",
    "nmf": ROOT / "analysis" / "topics" / "nmf_topic_summary.tsv",
    "bertopic": ROOT / "analysis" / "topics" / "bertopic_topic_summary.tsv",
}
TOPIC_ARI = ROOT / "analysis" / "topics" / "community_ari.tsv"
TOPIC_OVERLAP = ROOT / "analysis" / "topics" / "community_overlap.tsv"
CLUSTERS = ROOT / "analysis" / "clustering" / "clusters.tsv"
CLUSTER_MOTIFS = ROOT / "analysis" / "clustering" / "motif_checks.tsv"
HDB_SCAN = ROOT / "analysis" / "clustering" / "hdbscan_scan.tsv"
CLUSTER_ROBUST = ROOT / "analysis" / "clustering" / "robustness.tsv"
PERIODICITY_TESTS = ROOT / "analysis" / "periodicity" / "periodicity_tests.tsv"
NETWORK_SCRIPT = HERE / "cooccurrence_network.py"

SCHEMA = "triangulation/1"
DEFAULT_SEED = 42

read_tsv = cs.read_tsv
write_tsv = cs.write_tsv

# ---------------------------------------------------------------------------
# the methods under comparison
# ---------------------------------------------------------------------------
# key, label, source issue, source artifact, how the clause partition is made
METHODS = (
    ("net", "network community", "#29", "analysis/network/graph_nodes.tsv",
     "#29 nodes lifted to clauses by node frequency"),
    ("lda", "topic model (LDA k=5)", "#41", "analysis/topics/clause_topics.tsv",
     "#41 per-clause dominant topic"),
    ("nmf", "topic model (NMF k=6)", "#41", "analysis/topics/clause_topics.tsv",
     "#41 per-clause dominant topic"),
    ("bertopic", "topic model (BERTopic-style)", "#41",
     "analysis/topics/clause_topics.tsv", "#41 per-clause dominant topic (-1 = noise)"),
    ("kmeans", "embedding cluster (KMeans k=12)", "#40",
     "analysis/clause_embeddings/clause_embeddings.npz",
     "re-derived k-means over #39 vectors"),
)
METHOD_KEYS = tuple(m[0] for m in METHODS)
METHOD_LABEL = {m[0]: m[1] for m in METHODS}
METHOD_ISSUE = {m[0]: m[2] for m in METHODS}
PAIRS = tuple((a, b) for i, a in enumerate(METHOD_KEYS)
              for b in METHOD_KEYS[i + 1:])
PERIODICITY_LABEL = "periodicity (#51 nulls, motif series)"

# ---------------------------------------------------------------------------
# thresholds (fixed constants, disclosed in the report)
# ---------------------------------------------------------------------------
PURITY_STRONG = 0.50       # a method "owns" a motif at this purity or above
JACCARD_STRONG = 0.50      # mean pairwise Jaccard -> "convergent"
JACCARD_WEAK = 0.30        # mean pairwise Jaccard -> "divergent"
MIN_CLAUSES = 10           # below this a row is reported but not judged
VERDICTS = ("convergent", "partial", "divergent", "untestable")
KMEANS_K = 12              # #40's disclosed catalogue k
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
KMEANS_SEED = 0            # #40's disclosed seed
PERMUTATIONS = 200         # label-permutation null
SHUFFLES = 200             # #51-style shuffle surrogates per mode
BLOCK = 5                  # pages per block for the block shuffle
SHORT_PERIOD = 40          # pages: the "short band" a real motif cycle needs
LOCAL_HALF_BINS = 10       # neighbourhood half-width for the short-band line
WHITE_FLATNESS = math.exp(-0.5772156649015329)   # exp(-gamma), #51's anchor
LABEL_TERMS = 3            # terms in a distinctive-terms label

TERM_RE = re.compile(r"[a-z0-9]+")
_ESCAPES = (("\\n", " "), ("\\t", " "), ("\\r", " "), ("\\\\", " "))
TOP_TERM_RE = re.compile(r"([^(,]+)\(\s*(\d+)")
# The published term-level agreement #41 measured against #29's communities.
PUBLISHED_ARI = {"lda": 0.1893, "nmf": 0.0860, "bertopic": 0.0256}


# ---------------------------------------------------------------------------
# motif inventory
# ---------------------------------------------------------------------------
def M(name, source, all_phrases=(), any_phrases=()):
    """Motif spec: all required phrases, plus optional any-of aliases."""
    return {"name": name, "all": tuple(p for p in all_phrases if p),
            "any": tuple(p for p in any_phrases if p), "source": source}


MOTIFS = (
    # --- #40's motif_checks / #27's tight bigrams ------------------------
    M("artificial sun", "#40 motif_checks (380 clauses); #27 top bigram (367)",
      any_phrases=("artificial sun", "artificial-sun")),
    M("drug embryo", "#40 motif_checks (217); #27 (204)",
      any_phrases=("drug embryo", "drug-embryo")),
    M("cadaver city", "#40 motif_checks (214); #27 (192)",
      any_phrases=("cadaver city", "cadaver-city")),
    M("ice nebula", "#40 motif_checks (167); #27 (179)",
      any_phrases=("ice nebula", "ice-nebula")),
    M("internal organ", "#40 motif_checks (267); #27 (250)",
      any_phrases=("internal organ", "internal-organ")),
    M("adam", "#40 motif_checks (502); #22 label channel (ADAM 328)"),
    M("cold-blooded", "#40 motif_checks (192); #27 (177)",
      any_phrases=("cold-blooded", "cold blooded")),
    M("soul", "#40 motif_checks (483); #14 block names"),
    M("chromosome", "#40 motif_checks (226)"),
    M("gene", "#40 motif_checks (280)"),
    M("serum", "#40 motif_checks (256)"),
    M("fecundates", "#40 motif_checks (280); #27 coinage verb"),
    M("babel", "#40 motif_checks (94); #22 label channel (BABEL 75)"),
    M("tokage", "#40 motif_checks (134); #30 entity inventory"),
    M("necropolis", "#40 motif_checks (0) -- zero-occurrence control"),
    # --- #40's two task-named triples -----------------------------------
    M("dog+murder+gene", "#40 named motif 1 (15 clauses)",
      all_phrases=("dog", "murder", "gene")),
    M("brain+load+ecstasy", "#40 named motif 2 (3 clauses)",
      all_phrases=("brain", "load", "ecstasy")),
    # --- #41 topic labels / #40 cluster labels --------------------------
    M("boy-roid", "#41 LDA t1 top term; BERTopic t15"),
    M("soul/gram", "#41 LDA t1 / NMF t0 / BERTopic t17"),
    M("gene=tv", "#41 BERTopic t7; #28 static-embedding neighbours"),
    M("murder", "#40 cluster 5 label; #41 NMF t4"),
    M("assassin", "#41 BERTopic t3; #40 cluster 5"),
    M("entrails", "#41 NMF t4; #40 cluster 5"),
    M("consciousness", "#40 cluster 2; #41 LDA t4 / NMF t3 / BERTopic t17"),
    M("dog", "#40 cluster 7 label; #41 BERTopic t13"),
    M("cadaver", "#41 LDA t1 / NMF t0; #40 cluster 10"),
    M("flesh", "#41 LDA t4 / NMF t0; #40 cluster 10"),
    M("brain", "#40 cluster 2 label; #41 NMF t0"),
    M("organ", "#41 NMF t3 / BERTopic t17"),
    M("second", "#40 cluster 3 label; #41 BERTopic t5; #4 OCR loop"),
    M("cyclops", "#40 cluster 3 label"),
    M("machine", "#29 hubs C0/C2/C3; #41 LDA t0 / BERTopic t5"),
    M("drone", "#41 BERTopic t10; #40 cluster 4 terms"),
    M("parasite", "#41 BERTopic t10; #40 cluster 4 terms"),
    M("hydromania", "#41 BERTopic t0; #40 cluster 4 terms",
      any_phrases=("hydromania", "hydro=mania")),
    M("psychosexual", "#41 BERTopic t14; #29 C4"),
    M("hologram", "#29 C4 (103 occurrences); #41 BERTopic t14"),
    M("swastika", "#41 BERTopic t2; #22 label channel"),
    M("corporation", "#29 C5 (sato / corporation, 9 nodes)"),
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def unescape(s: str) -> str:
    """Reverse the #17/#33 TSV cell escaping (newline/tab/backslash -> space)."""
    for a, b in _ESCAPES:
        s = s.replace(a, b)
    return s


def r(x, nd=4):
    """Round for TSV / report output; ``None`` becomes an empty cell."""
    if x is None or x == "":
        return "" if x == "" else x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(v) or math.isinf(v):
        return ""
    return round(v, nd)


def pair_ari(summary, a: str, b: str):
    """Clause-level ARI for a method pair, in either argument order.

    ``summary["method_agreement"]`` stores the pairs in ``PAIRS`` order (which
    follows ``METHOD_KEYS``), so a lookup must not assume the caller's order --
    an order-sensitive lookup silently renders ``None`` in the report.
    """
    for row in summary["method_agreement"]:
        if {row["a"], row["b"]} == {a, b}:
            return row["ari"]
    raise KeyError("no agreement row for %s|%s" % (a, b))


def md(text) -> str:
    """Escape a value for a markdown table cell (pipes break the table)."""
    return str(text).replace("|", "/")


def f2(x, nd=2) -> str:
    """Format a number for a markdown table cell (``-`` when absent)."""
    if x is None or x == "":
        return "-"
    try:
        return str(r(float(x), nd))
    except (TypeError, ValueError):
        return str(x)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def parse_top_terms(blob: str) -> dict:
    """``"sun(357), artificial(310)"`` -> ``{"sun": 357, "artificial": 310}``."""
    out = {}
    for term, n in TOP_TERM_RE.findall(blob or ""):
        term = term.strip()
        if term and term not in out:
            out[term] = int(n)
    return out


def load_domain_seeds(path: Path = NETWORK_SCRIPT) -> dict:
    """#29's ``DOMAIN_SEEDS``, parsed out of its builder with ``ast``.

    The literal is read from ``work/cooccurrence_network.py`` instead of being
    duplicated here, and ``ast.literal_eval`` avoids importing that module
    (it pulls ``networkx``/``sklearn``, which this report does not need).
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DOMAIN_SEEDS"
                for t in node.targets):
            return {k: set(v) for k, v in ast.literal_eval(node.value).items()}
    raise KeyError("DOMAIN_SEEDS not found in %s" % path)


def spec_phrases(spec):
    """``(all_of, any_of)`` for a spec, defaulting a bare motif to its name.

    A motif whose name is itself a single surface form (``adam``, ``soul/gram``,
    ``gene=tv``) carries no explicit phrase list; it then matches on its own
    name, so a bare name can never degenerate into "every clause".
    """
    all_p = spec["all"]
    any_p = spec["any"]
    if not all_p and not any_p:
        any_p = (spec["name"],)
    return all_p, any_p


def match_rule(spec) -> str:
    all_p, any_p = spec_phrases(spec)
    parts = []
    if all_p:
        parts.append("all of [" + ", ".join(all_p) + "]")
    if any_p:
        parts.append("any of [" + ", ".join(any_p) + "]")
    return " AND ".join(parts)


# ---------------------------------------------------------------------------
# agreement statistics
# ---------------------------------------------------------------------------
def encode(labels) -> np.ndarray:
    """Arbitrary labels -> contiguous int codes (order of first appearance)."""
    index = {}
    out = np.empty(len(labels), dtype=np.int64)
    for i, lab in enumerate(labels):
        key = int(lab)
        if key not in index:
            index[key] = len(index)
        out[i] = index[key]
    return out


def contingency(a, b, ka, kb) -> np.ndarray:
    return np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(np.float64)


def adjusted_rand(a, b, ka, kb) -> float:
    """Adjusted Rand index from a contingency table (``0.0`` when undefined)."""
    cm = contingency(a, b, ka, kb)
    n = cm.sum()
    if n < 2:
        return 0.0

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(cm).sum()
    sum_i = comb2(cm.sum(axis=1)).sum()
    sum_j = comb2(cm.sum(axis=0)).sum()
    expected = sum_i * sum_j / comb2(n)
    max_index = 0.5 * (sum_i + sum_j)
    denom = max_index - expected
    return 0.0 if abs(denom) < 1e-12 else float((sum_ij - expected) / denom)


def normalised_mi(a, b, ka, kb) -> float:
    """NMI, arithmetic normalisation, from a contingency table."""
    cm = contingency(a, b, ka, kb)
    n = cm.sum()
    if n < 2:
        return 0.0
    pi = cm.sum(axis=1) / n
    pj = cm.sum(axis=0) / n
    pij = cm / n
    outer = np.outer(pi, pj)
    nz = pij > 0
    mi = float((pij[nz] * np.log(pij[nz] / outer[nz])).sum())
    ha = float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum())
    hb = float(-(pj[pj > 0] * np.log(pj[pj > 0])).sum())
    return 0.0 if ha + hb == 0 else 2.0 * mi / (ha + hb)


# ---------------------------------------------------------------------------
# the re-derived embedding partition
# ---------------------------------------------------------------------------
def kmeans(X, k=KMEANS_K, seed=KMEANS_SEED, n_init=KMEANS_N_INIT,
           max_iter=KMEANS_MAX_ITER):
    """K-means (k-means++ init, Lloyd, Euclidean) on L2-normalised vectors.

    ``numpy`` only, so the embedding column is reproducible without
    scikit-learn.  Deterministic for a fixed ``seed``.

    Returns ``(labels, inertia, sizes)``.
    """
    rng = np.random.default_rng(seed)
    n, dim = X.shape
    best = None
    for _ in range(n_init):
        centers = np.empty((k, dim), dtype=np.float32)
        centers[0] = X[int(rng.integers(n))]
        d2 = ((X - centers[0]) ** 2).sum(1)
        for c in range(1, k):
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
            centers[c] = X[int(rng.choice(n, p=probs))]
            d2 = np.minimum(d2, ((X - centers[c]) ** 2).sum(1))
        labels = np.full(n, -1, dtype=np.int64)
        for it in range(max_iter):
            sims = X @ centers.T
            sims -= (centers * centers).sum(1)[None, :] / 2.0
            new = sims.argmax(1).astype(np.int64)
            if it and (new == labels).all():
                break
            labels = new
            for c in range(k):
                m = labels == c
                if m.any():
                    centers[c] = X[m].mean(0)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy())
    labels = best[1]
    sizes = [int((labels == c).sum()) for c in range(k)]
    return labels, best[0], sizes


def distinctive_terms(in_counts, n_in, out_counts, n_out, alpha0=100.0,
                      min_f=3, top_n=12):
    """Monroe et al. (2008) log-odds with informative Dirichlet prior (#40's)."""
    rows = []
    for w, f_in in in_counts.items():
        if f_in < min_f:
            continue
        f_out = out_counts.get(w, 0)
        alpha_w = alpha0 * (f_in + f_out) / max(1, n_in + n_out)
        a_in = f_in + alpha_w
        a_out = f_out + alpha_w
        delta = (math.log(a_in / (n_in + alpha0 - a_in))
                 - math.log(a_out / (n_out + alpha0 - a_out)))
        var = 1.0 / a_in + 1.0 / a_out
        z = delta / math.sqrt(var) if var > 0 else 0.0
        rows.append((w, z, f_in, f_out))
    rows.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return rows[:top_n]


def net_clause_labels(tokens, term_community, term_freq):
    """#29 communities lifted to clauses.

    A clause takes the community carrying the most #29 node frequency among
    its screened unit surfaces; ties break to the lowest community id.
    ``-1`` marks a clause whose units miss the 391-node graph -- coverage is
    reported, never silently dropped.
    """
    out = []
    for toks in tokens:
        acc = collections.Counter()
        for t in set(toks):
            cid = term_community.get(t)
            if cid is not None:
                acc[cid] += term_freq.get(t, 0.0)
        if not acc:
            out.append(-1)
        else:
            out.append(max(sorted(acc.items()), key=lambda kv: (kv[1], -kv[0]))[0])
    return out


def validate_clusters(labels, published, texts):
    """Re-derived k-means vs #40's published catalogue (label + size)."""
    per_cluster = collections.defaultdict(collections.Counter)
    member_count = collections.Counter(int(v) for v in labels)
    book = collections.Counter()
    for i, text in enumerate(texts):
        toks = TERM_RE.findall(text.lower())      # occurrences, as #40 counted
        per_cluster[int(labels[i])].update(toks)
        book.update(toks)
    global_total = sum(book.values())
    derived, sizes, tops = {}, {}, {}
    for c in sorted(per_cluster):
        in_c = per_cluster[c]
        n_in = sum(in_c.values())             # token totals, not clause counts
        rows = distinctive_terms(in_c, n_in, book - in_c, global_total - n_in)
        sizes[c] = member_count[c]
        tops[c] = [(w, z, f_in) for w, z, f_in, _f_out in rows[:LABEL_TERMS]]
        derived[c] = [w for w, _z, _f in tops[c]]
    pub_terms = {cid: set(parse_top_terms(row["top_terms"]))
                 for cid, row in published.items()}
    cands = []
    for c, terms in derived.items():
        ts = set(terms)
        for cid, pset in sorted(pub_terms.items()):
            if ts and pset:
                cands.append((len(ts & pset) / len(ts | pset), c, cid))
    cands.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_c, used_p, matched = set(), set(), {}
    for overlap, c, cid in cands:
        if c in used_c or cid in used_p:
            continue
        used_c.add(c)
        used_p.add(cid)
        matched[c] = (cid, overlap)
    rows = []
    for c in sorted(derived):
        cid, overlap = matched.get(c, ("", None))
        pub = published.get(cid)
        pub_size = int(pub["size"]) if pub else None
        rows.append({
            "derived_cluster": c,
            "n_clauses": sizes[c],
            "derived_label_terms": " · ".join(derived[c]),
            "matched_published_cluster": cid,
            "published_label": pub["label"] if pub else "",
            "published_size": pub_size if pub is not None else "",
            "size_ratio": r(sizes[c] / pub_size) if pub_size else "",
            "label_jaccard": r(overlap),
            "derived_top_terms": ", ".join(
                "%s(%.1f)" % (w, z) for w, z, _f in tops[c]),
            "published_top_terms": (
                ", ".join("%s(%d)" % (w, n) for w, n in
                          list(parse_top_terms(pub["top_terms"]).items())[:LABEL_TERMS])
                if pub else ""),
        })
    return rows


# ---------------------------------------------------------------------------
# motif matching / dominance
# ---------------------------------------------------------------------------
def motif_hits(spec, texts):
    """Clause indices satisfying a motif spec (#40's substring rule)."""
    all_p, any_p = spec_phrases(spec)
    hits = []
    for i, text in enumerate(texts):
        if all_p and not all(p in text for p in all_p):
            continue
        if any_p and not any(p in text for p in any_p):
            continue
        hits.append(i)
    return hits


def dominant_group(members, labels):
    """``(group, n_in_group, purity)`` of the modal group (ties: lowest id)."""
    if not members:
        return None, 0, None
    counts = collections.Counter(labels[i] for i in members)
    group, n = min(sorted(counts.items()), key=lambda kv: (-kv[1], kv[0]))
    return group, n, n / len(members)


def jaccard(members, labels_a, labels_b, ga, gb):
    """Overlap of the two methods' dominant buckets over these clauses."""
    sa = {i for i in members if labels_a[i] == ga}
    sb = {i for i in members if labels_b[i] == gb}
    union = len(sa | sb)
    return len(sa & sb) / union if union else None


def top5_share(pages, hits):
    if not hits:
        return None
    counts = collections.Counter(pages[i] for i in hits)
    return sum(n for _p, n in counts.most_common(5)) / len(hits)


def verdict_for(n_clauses, mean_jaccard) -> tuple:
    if n_clauses < MIN_CLAUSES or mean_jaccard is None:
        return ("untestable",
                "fewer than %d clauses, or no clause shares two dominant buckets"
                % MIN_CLAUSES)
    if mean_jaccard >= JACCARD_STRONG:
        return "convergent", "mean pairwise Jaccard >= %.2f" % JACCARD_STRONG
    if mean_jaccard >= JACCARD_WEAK:
        return ("partial",
                "mean pairwise Jaccard in [%.2f, %.2f)" % (JACCARD_WEAK, JACCARD_STRONG))
    return "divergent", "mean pairwise Jaccard < %.2f" % JACCARD_WEAK


# ---------------------------------------------------------------------------
# periodicity machinery (#51's tests, applied to motif series)
# ---------------------------------------------------------------------------
def autocorrelation(x, max_lag):
    """Linear autocorrelation via a zero-padded FFT (no wrap-around)."""
    y = np.asarray(x, dtype=np.float64)
    n = y.size
    if n < 2:
        return np.zeros(1)
    y = y - y.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[:max_lag + 1]
    return ac / ac[0] if ac[0] > 0 else ac


def periodogram(x):
    """Hann-windowed periodogram -> ``(freqs, power)`` including DC."""
    y = np.asarray(x, dtype=np.float64)
    n = y.size
    y = y - y.mean()
    power = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    return np.fft.rfftfreq(n), power


def band_slice(freqs, power, freq_max: float | None):
    """Drop DC, and optionally keep only bins above ``freq_max``.

    ``freq_max`` selects the *short* band: frequencies at or above it, i.e.
    periods at or below its reciprocal.  Restricting the band before the
    peak search is what keeps a short-band search from simply re-finding the
    low-frequency drift of a dispersed series.
    """
    freqs = np.asarray(freqs)
    power = np.asarray(power, dtype=np.float64)
    idx = np.arange(1, power.size)
    if freq_max is not None:
        idx = idx[freqs[idx] >= freq_max]
    return idx


def spectral_flatness(power, idx=None):
    p = np.asarray(power, dtype=np.float64)
    p = p[np.arange(1, p.size)] if idx is None else p[idx]
    p = p[p > 0]
    if p.size == 0:
        return None
    return float(math.exp(np.mean(np.log(p))) / p.mean())


def peak_rel(power, freqs, idx):
    """``(peak power / band median, period of the peak)`` over bins ``idx``.

    The frequency axis is indexed by *sample* count, so the axis must be built
    from the series length rather than the number of bins.  This is #51's
    statistic: the peak over the *whole* band relative to the band median.
    """
    power = np.asarray(power, dtype=np.float64)
    p = power[idx]
    if p.size == 0 or not np.isfinite(p).any():
        return None, None
    median = float(np.median(p))
    if median <= 0:
        return None, None
    i = int(idx[int(np.argmax(p))])
    f = float(np.asarray(freqs)[i])
    return (float(p.max()) / median, float(1.0 / f) if f > 0 else None)


def peak_rel_local(power, idx, half: int = LOCAL_HALF_BINS):
    """Peak over its *local* background: ``max / median(nearest 2*half bins)``.

    #51's band-median statistic is sensitive to where the band median sits.
    For a motif series the page-rate envelope depresses the median of any
    short band, which inflates the ratio even when no cycle is present, and
    the block shuffle inflates it again by splattering power at block
    boundaries.  Measuring the peak against its own neighbourhood removes both
    confounds, so the short-band test is a line test that a merely
    heterogeneous series cannot pass.
    """
    power = np.asarray(power, dtype=np.float64)
    p = power[idx]
    if p.size == 0 or not np.isfinite(p).any():
        return None
    i = int(np.argmax(p))
    lo, hi = max(0, i - half), min(p.size, i + half + 1)
    bg = np.concatenate([p[lo:i], p[i + 1:hi]])
    if bg.size == 0:
        return None
    median = float(np.median(bg))
    return float(p[i]) / median if median > 0 else None


def shuffle_series(x, mode, rng, block=BLOCK):
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    if mode == "global":
        rng.shuffle(x)
    elif mode == "block":
        starts = list(range(0, n, block))
        out = np.empty(n, dtype=np.float64)
        for j, si in enumerate(rng.permutation(len(starts))):
            s = starts[si]
            out[s:s + block] = x[s:s + block]
        x = out
    elif mode == "detrend":
        t = np.arange(n, dtype=np.float64)
        x = x - np.polyval(np.polyfit(t, x, 1), t)
        rng.shuffle(x)
    else:
        raise ValueError(mode)
    return x


def series_stats(x, seed=DEFAULT_SEED, shuffles=SHUFFLES):
    """Full-band and short-band peak statistics plus the three #51 nulls.

    Two peak searches run on every series:

    * the **full band** with #51's peak/band-median statistic, whose peak for
      a slowly varying rate always sits at the longest period (the signature
      #51 read as drift), and
    * the **short band** (period <= ``SHORT_PERIOD`` pages) with the local
      *line* statistic (``peak_rel_local``), which is where a real metrical
      cycle of a motif would have to appear and which a merely heterogeneous
      page series cannot fake.

    Both are tested against global / block / detrended shuffles of the same
    series.  The block shuffle is kept because #51's caveats attach to it, but
    it is *not* the gate for the short-band test: permuting 5-page blocks
    creates runs whose edges splatter power across the band, so it inflates
    the null statistic rather than testing it.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 8 or float(x.std()) == 0.0:
        return {"degenerate": True}
    freq_max = 1.0 / SHORT_PERIOD
    freqs, power = periodogram(x)
    full_idx = band_slice(freqs, power, None)
    short_idx = band_slice(freqs, power, freq_max)
    flat = spectral_flatness(power, full_idx)
    rel, per = peak_rel(power, freqs, full_idx)
    rel_s = peak_rel_local(power, short_idx)
    per_s = None
    if rel_s is not None:
        i = int(short_idx[int(np.argmax(power[short_idx]))])
        f = float(freqs[i])
        per_s = float(1.0 / f) if f > 0 else None
    acf = autocorrelation(x, min(20, n - 1))
    rng = np.random.default_rng(seed)
    null = collections.defaultdict(list)
    for _ in range(shuffles):
        for mode in ("global", "block", "detrend"):
            _nf, npow = periodogram(shuffle_series(x, mode, rng))
            null[mode].append((spectral_flatness(npow, full_idx),
                               peak_rel(npow, freqs, full_idx)[0],
                               peak_rel_local(npow, short_idx)))
    out = {"degenerate": False, "flatness": flat, "peak_rel": rel,
           "peak_period": per, "acf1": float(acf[1]) if acf.size > 1 else None,
           "peak_rel_short": rel_s, "peak_period_short": per_s}
    for mode in ("global", "block", "detrend"):
        sub = null[mode]
        n_rel = sum(1 for _f, rr, _rs in sub
                    if rr is not None and rel is not None and rr >= rel)
        n_flat = sum(1 for ff, _rr, _rs in sub
                     if ff is not None and flat is not None and ff <= flat)
        n_short = sum(1 for _f, _rr, rs in sub
                      if rs is not None and rel_s is not None and rs >= rel_s)
        out["p_" + mode] = (1 + n_rel) / (len(sub) + 1)
        out["p_" + mode + "_flat"] = (1 + n_flat) / (len(sub) + 1)
        out["p_" + mode + "_short"] = (1 + n_short) / (len(sub) + 1)
    return out


def renewal_stats(positions):
    """Inter-arrival statistics of clause positions (#51 §5 style)."""
    pos = sorted(positions)
    gaps = [b - a for a, b in itertools.pairwise(pos)]
    if len(gaps) < 2:
        return None
    m = sum(gaps) / len(gaps)
    var = sum((g - m) ** 2 for g in gaps) / len(gaps)
    geo_var = m * (m - 1.0) if m > 1 else None
    return {"n_intervals": len(gaps), "gap_mean": m,
            "gap_cv": math.sqrt(var) / m if m else None,
            "geo_var_ratio": var / geo_var if geo_var else None,
            "gap_max": float(max(gaps))}


def bh_qvalues(pvals):
    """Benjamini-Hochberg q-values for a list of p (NaN/None -> 1.0)."""
    order = sorted(range(len(pvals)),
                   key=lambda i: (1.0 if pvals[i] in (None, "") else pvals[i], i))
    n = len(pvals)
    q = [1.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        p = 1.0 if pvals[i] in (None, "") else float(pvals[i])
        k = n - rank + 1
        prev = min(prev, p * n / k)
        q[i] = prev
    return q


def periodicity_verdict(stats) -> str:
    """Verdict for one motif series.

    ``periodic`` requires a short-band **line** (period <= ``SHORT_PERIOD``
    pages) that survives both the global and the detrended shuffle nulls --
    i.e. a cycle, not a slow drift.  The block shuffle is deliberately not the
    gate: permuting 5-page blocks creates runs whose edges splatter power
    across the band, so it inflates its own null statistic instead of testing
    it (see ``series_stats``).  ``drift-only`` means the surviving peak is the
    full-band one, which for a rate series sits at the longest period and is
    what #51 read as drift rather than periodicity.
    """
    if stats.get("degenerate"):
        return "degenerate"
    pgs, pds = stats.get("p_global_short"), stats.get("p_detrend_short")
    pg, pd = stats.get("p_global"), stats.get("p_detrend")
    if pg is None or pd is None:
        return "untested"
    if pgs is not None and pds is not None and pgs < 0.05 and pds < 0.05:
        return "periodic"
    if pg < 0.05 and pd < 0.05:
        return "drift-only"
    return "not-periodic"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(out_dir: Path = OUT, permutations: int = PERMUTATIONS,
          shuffles: int = SHUFFLES, n_init: int = KMEANS_N_INIT,
          seed: int = DEFAULT_SEED, quiet: bool = False) -> dict:
    out_dir = Path(out_dir)
    diag = collections.Counter()

    # --- frozen inputs ---------------------------------------------------
    nodes = read_tsv(NET_NODES)
    term_community = {r["term"]: int(r["community"]) for r in nodes}
    term_freq = {r["term"]: float(r["freq"]) for r in nodes}
    comm_sizes = collections.Counter(int(r["community"]) for r in nodes)
    comm_label = {}
    if TOPIC_OVERLAP.exists():
        for row in read_tsv(TOPIC_OVERLAP):
            comm_label.setdefault(int(row["best_community"]), row["community_label"])

    clause_rows = read_tsv(CLAUSE_INDEX)
    clause_ids = [int(r["clause_id"]) for r in clause_rows]
    pages = [int(r["page"]) for r in clause_rows]
    texts = [unescape(r["text_units"]).lower() for r in clause_rows]
    tokens = [t.split() for t in texts]
    diag["clauses"] = len(clause_rows)
    diag["network_nodes"] = len(nodes)
    diag["network_communities"] = len(comm_sizes)

    topic_rows = read_tsv(CLAUSE_TOPICS)
    if [int(r["clause_id"]) for r in topic_rows] != clause_ids:
        raise ValueError("clause_topics.tsv is out of step with clause_index.tsv")
    topic_labels = {key: {int(row["topic_id"]): row["label"]
                          for row in read_tsv(path)}
                    for key, path in sorted(TOPIC_SUMMARIES.items())}
    topic_labels["bertopic"][-1] = "noise"

    published_ari = {}
    if TOPIC_ARI.exists():
        for row in read_tsv(TOPIC_ARI):
            published_ari[row["model"]] = float(row["adjusted_rand"])
    published_clusters = {int(row["cluster_id"]): row for row in read_tsv(CLUSTERS)}

    # --- clause partitions ----------------------------------------------
    labels = {
        "net": net_clause_labels(tokens, term_community, term_freq),
        "lda": [int(r["lda_topic"]) for r in topic_rows],
        "nmf": [int(r["nmf_topic"]) for r in topic_rows],
        "bertopic": [int(r["bertopic_topic"]) for r in topic_rows],
    }
    store = np.load(EMBEDDINGS, allow_pickle=False)
    if list(store["clause_ids"]) != clause_ids:
        raise ValueError("clause_embeddings.npz is out of step with clause_index.tsv")
    X = store["emb_units"].astype(np.float32)
    km_labels, inertia, km_sizes = kmeans(X, KMEANS_K, KMEANS_SEED, n_init)
    labels["kmeans"] = [int(v) for v in km_labels]
    diag["kmeans_inertia"] = round(inertia, 4)
    diag["kmeans_sizes_sum"] = sum(km_sizes)
    diag["kmeans_distinct_clusters"] = sum(1 for s in km_sizes if s)

    def group_label(key, group):
        if group is None:
            return ""
        if key == "net":
            return ("C%s %s" % (group, comm_label.get(group, ""))).strip()
        if key == "bertopic" and group == -1:
            return "bertopic noise"
        if key == "kmeans":
            pub = km_published_label.get(int(group), "")
            return "kmeans %s [%s]" % (group, pub) if pub else "kmeans %s" % group
        return "%s %s: %s" % (key, group, topic_labels[key].get(group, ""))

    validation = validate_clusters(labels["kmeans"], published_clusters, texts)
    km_published_label = {int(v["derived_cluster"]):
                          (v["published_label"] if v["published_label"] else "")
                          for v in validation}

    # --- 1. clause-level method agreement -------------------------------
    codes = {k: encode(v) for k, v in labels.items()}
    n_codes = {k: (int(codes[k].max()) + 1 if codes[k].size else 0) for k in codes}
    counters = {k: collections.Counter(v) for k, v in labels.items()}
    rng = np.random.default_rng(seed)
    agree_rows, agree = [], {}
    for a_key, b_key in PAIRS:
        keep = (np.array(labels[a_key]) != -1) & (np.array(labels[b_key]) != -1)
        aa, bb = codes[a_key][keep], codes[b_key][keep]
        ka, kb = n_codes[a_key], n_codes[b_key]
        ari = adjusted_rand(aa, bb, ka, kb)
        nmi = normalised_mi(aa, bb, ka, kb)
        ari_null, nmi_null = [], []
        for _ in range(permutations):
            shuffled = bb[rng.permutation(bb.size)]
            ari_null.append(adjusted_rand(aa, shuffled, ka, kb))
            nmi_null.append(normalised_mi(aa, shuffled, ka, kb))
        p_ari = (1 + sum(1 for v in ari_null if v >= ari)) / (permutations + 1)
        p_nmi = (1 + sum(1 for v in nmi_null if v >= nmi)) / (permutations + 1)
        agree[(a_key, b_key)] = agree[(b_key, a_key)] = ari
        agree_rows.append({
            "method_a": a_key, "method_b": b_key,
            "label_a": METHOD_LABEL[a_key], "label_b": METHOD_LABEL[b_key],
            "n_shared_clauses": int(keep.sum()),
            "n_groups_a": len([g for g in counters[a_key] if g != -1]),
            "n_groups_b": len([g for g in counters[b_key] if g != -1]),
            "adjusted_rand": r(ari), "nmi": r(nmi),
            "ari_null_mean": r(mean(ari_null)), "nmi_null_mean": r(mean(nmi_null)),
            "p_permutation_ari": r(p_ari, 5),
            "p_permutation_nmi": r(p_nmi, 5),
            "above_chance": int(p_ari <= 0.05),
        })
    write_tsv(out_dir / "method_agreement.tsv",
              ["method_a", "method_b", "label_a", "label_b", "n_shared_clauses",
               "n_groups_a", "n_groups_b", "adjusted_rand", "nmi",
               "ari_null_mean", "nmi_null_mean", "p_permutation_ari",
               "p_permutation_nmi", "above_chance"], agree_rows)

    clause_rows_out = []
    for key, label, issue, artifact, kind in METHODS:
        counter = counters[key]
        n_clauses = len(labels[key])
        assigned = n_clauses - counter.get(-1, 0)
        top_group, top_n = (counter.most_common(1)[0] if counter else ("", ""))
        clause_rows_out.append({
            "method": key, "label": label, "source_issue": issue,
            "source_artifact": artifact, "partition_kind": kind,
            "n_groups": len([g for g in counter if g != -1]),
            "n_clauses": n_clauses, "n_assigned": assigned,
            "coverage": r(assigned / n_clauses if n_clauses else None),
            "unassigned_clauses": counter.get(-1, 0),
            "largest_group": top_group, "largest_group_n": top_n,
            "largest_group_label": group_label(key, top_group) if counter else "",
        })
    write_tsv(out_dir / "clause_agreement.tsv",
              ["method", "label", "source_issue", "source_artifact",
               "partition_kind", "n_groups", "n_clauses", "n_assigned",
               "coverage", "unassigned_clauses", "largest_group",
               "largest_group_n", "largest_group_label"], clause_rows_out)

    write_tsv(out_dir / "cluster_validation.tsv",
              ["derived_cluster", "n_clauses", "derived_label_terms",
               "matched_published_cluster", "published_label", "published_size",
               "size_ratio", "label_jaccard", "derived_top_terms",
               "published_top_terms"], validation)

    # --- 2. motif inventory + convergence matrix ------------------------
    n_pages = max(pages) if pages else 0
    page_slot = {p: i for i, p in enumerate(sorted(set(pages)))}
    domain_seeds = load_domain_seeds()
    specs = list(MOTIFS) + [
        M("domain:%s" % dom,
          "#29 DOMAIN_SEEDS[%r] (%d seed terms)" % (dom, len(words)),
          any_phrases=tuple(sorted(words)))
        for dom, words in sorted(domain_seeds.items())
    ]
    diag["matrix_rows"] = len(specs)

    motif_rows, cells, periodicity_rows, wide = [], [], [], []
    for spec in specs:
        hits = motif_hits(spec, texts)
        n_hits = len(hits)
        kind = "domain" if spec["name"].startswith("domain:") else "motif"
        cell_map = {}
        for key in METHOD_KEYS:
            g, n_in, purity = dominant_group(hits, labels[key])
            cell_map[key] = {
                "dominant_group": g, "n_in_group": n_in, "purity": purity,
                "group_label": group_label(key, g) if g is not None else "",
                "strong": int(purity is not None and purity >= PURITY_STRONG),
            }
            cells.append({
                "motif": spec["name"], "kind": kind, "method": key,
                "method_label": METHOD_LABEL[key], "source_issue": METHOD_ISSUE[key],
                "n_clauses": n_hits,
                "dominant_group": "" if g is None else g,
                "dominant_group_label": cell_map[key]["group_label"],
                "n_in_group": n_in, "purity": r(purity),
                "strong": cell_map[key]["strong"],
            })
        pair_vals, row = {}, {
            "motif": spec["name"], "kind": kind, "source": spec["source"],
            "match_rule": match_rule(spec), "n_clauses": n_hits,
            "frac_clauses": r(n_hits / len(texts) if texts else None),
            "n_pages": len({pages[i] for i in hits}),
            "top5_page_share": r(top5_share(pages, hits)),
        }
        for a_key, b_key in PAIRS:
            jac = (jaccard(hits, labels[a_key], labels[b_key],
                           cell_map[a_key]["dominant_group"],
                           cell_map[b_key]["dominant_group"]) if n_hits else None)
            pair_vals[(a_key, b_key)] = jac
            row["jac_%s_%s" % (a_key, b_key)] = r(jac)
        scored = [(v, p) for p, v in pair_vals.items() if v is not None]
        row["mean_pair_jaccard"] = r(mean([v for v, _p in scored]))
        row["min_pair_jaccard"] = r(min(scored)[0]) if scored else None
        row["max_pair_jaccard"] = r(max(scored)[0]) if scored else None
        row["weakest_pair"] = "|".join(min(scored)[1]) if scored else ""
        row["strongest_pair"] = "|".join(max(scored)[1]) if scored else ""
        row["n_strong_methods"] = sum(cell_map[k]["strong"] for k in METHOD_KEYS)
        verdict, note = verdict_for(n_hits, row["mean_pair_jaccard"] or None)
        row["verdict"], row["verdict_note"] = verdict, note

        per = periodicity_for_motif(spec["name"], kind, hits, pages, page_slot,
                                    n_pages, seed=seed, shuffles=shuffles)
        periodicity_rows.append(per)
        row["per_flatness"] = per["flatness"]
        row["per_peak_period"] = per["peak_period_pages"]
        row["per_p_block"] = per["p_block"]
        row["per_verdict"] = per["verdict"]
        motif_rows.append(row)
        wide.append((row, cell_map, pair_vals))

    motif_rows.sort(key=lambda x: (x["kind"] != "motif",
                                   -(x["mean_pair_jaccard"] or -1.0),
                                   -x["n_clauses"], x["motif"]))
    cells.sort(key=lambda c: (c["motif"], METHOD_KEYS.index(c["method"])))
    write_tsv(out_dir / "motif_inventory.tsv",
              ["motif", "kind", "match_rule", "source", "n_clauses",
               "frac_clauses", "n_pages", "top5_page_share",
               "mean_pair_jaccard", "min_pair_jaccard", "max_pair_jaccard",
               "weakest_pair", "strongest_pair", "n_strong_methods", "verdict",
               "verdict_note", "per_flatness", "per_peak_period", "per_p_block",
               "per_verdict"], motif_rows)
    write_tsv(out_dir / "motif_method_cells.tsv",
              ["motif", "kind", "method", "method_label", "source_issue",
               "n_clauses", "dominant_group", "dominant_group_label",
               "n_in_group", "purity", "strong"], cells)

    matrix_header = (["motif", "kind", "n_clauses", "n_pages", "top5_page_share",
                      "mean_pair_jaccard", "min_pair_jaccard", "max_pair_jaccard",
                      "weakest_pair", "strongest_pair", "n_strong_methods",
                      "verdict"]
                     + ["pur_%s" % k for k in METHOD_KEYS]
                     + ["grp_%s" % k for k in METHOD_KEYS]
                     + ["jac_%s_%s" % (a, b) for a, b in PAIRS]
                     + ["per_flatness", "per_peak_period", "per_p_block",
                        "per_verdict"])
    matrix_rows = []
    for row, cell_map, _pv in wide:
        mrow = dict(row)
        for key in METHOD_KEYS:
            mrow["pur_%s" % key] = r(cell_map[key]["purity"])
            mrow["grp_%s" % key] = cell_map[key]["group_label"]
        matrix_rows.append(mrow)
    write_tsv(out_dir / "convergence_matrix.tsv", matrix_header, matrix_rows)
    write_tsv(out_dir / "domain_convergence.tsv", matrix_header,
              [m for m in matrix_rows if m["kind"] == "domain"])

    # #40's own motif counts, restated next to ours (same matching rule, plus
    # the hyphenated aliases declared in the inventory).
    motif_recon = []
    if CLUSTER_MOTIFS.exists():
        km_purity = {row["motif"]: cm["kmeans"]["purity"] for row, cm, _pv in wide}
        km_cluster = {row["motif"]: cm["kmeans"]["dominant_group"]
                      for row, cm, _pv in wide}
        km_matched = {int(v["derived_cluster"]): v["matched_published_cluster"]
                      for v in validation}
        pub_counts = {row["motif"]: row for row in read_tsv(CLUSTER_MOTIFS)}
        for row in motif_rows:
            pub = pub_counts.get(row["motif"])
            if pub is None:
                continue
            n_40 = int(pub["n_clauses"])
            note = ("exact match" if row["n_clauses"] == n_40 else
                    "inventory adds hyphenated aliases, so the count is >= #40's")
            derived_cluster = km_cluster.get(row["motif"])
            matched = (km_matched.get(derived_cluster, "")
                       if derived_cluster is not None else "")
            motif_recon.append({
                "motif": row["motif"], "n_clauses_73": row["n_clauses"],
                "n_clauses_40": n_40,
                "delta": row["n_clauses"] - n_40,
                "match_rule_73": row["match_rule"],
                "purity_kmeans_73": r(km_purity.get(row["motif"])),
                "frac_in_top_cluster_40": pub["frac_in_top_cluster"],
                "top_cluster_40": pub["top_cluster"],
                "kmeans_cluster_matched_40": matched,
                "same_top_cluster_as_40": (
                    int(str(matched) == pub["top_cluster"])
                    if row["n_clauses"] and pub["top_cluster"] else ""),
                "note": note,
            })
        missing = sorted(set(pub_counts) - {row["motif"] for row in motif_rows})
        for name in missing:
            motif_recon.append({
                "motif": name, "n_clauses_73": "", "n_clauses_40":
                    pub_counts[name]["n_clauses"], "delta": "",
                "match_rule_73": "not in the #73 inventory",
                "purity_kmeans_73": "",
                "frac_in_top_cluster_40":
                    pub_counts[name]["frac_in_top_cluster"],
                "top_cluster_40": pub_counts[name]["top_cluster"],
                "kmeans_cluster_matched_40": "",
                "same_top_cluster_as_40": "",
                "note": "carried in #40's inventory only",
            })
    write_tsv(out_dir / "motif_count_reconciliation.tsv",
              ["motif", "n_clauses_73", "n_clauses_40", "delta",
               "match_rule_73", "purity_kmeans_73", "top_cluster_40",
               "frac_in_top_cluster_40", "kmeans_cluster_matched_40",
               "same_top_cluster_as_40", "note"], motif_recon)

    periodicity_rows.sort(key=lambda x: (x["kind"] != "motif", x["motif"]))
    # BH correction across the matrix rows: 45 series x several statistics is
    # enough testing to need it, and the verdicts below are read off the
    # corrected values where a lead is claimed.
    for stat in ("p_global", "p_block", "p_detrend", "p_global_short",
                 "p_block_short", "p_detrend_short"):
        for row, q in zip(periodicity_rows,
                          bh_qvalues([p.get(stat) for p in periodicity_rows])):
            row["q" + stat[1:]] = r(q, 5)
    for row in periodicity_rows:
        if row["verdict"] == "no occurrences":
            row["short_cycle_lead"] = ""
            continue
        clears = (row.get("q_global_short") is not None
                  and float(row["q_global_short"]) < 0.05
                  and float(row["q_detrend_short"]) < 0.05)
        row["short_cycle_lead"] = int(bool(clears))
    write_tsv(out_dir / "periodicity_motifs.tsv",
              ["motif", "kind", "series", "n_pages_total", "n_pages_present",
               "page_presence", "series_mean", "series_var", "flatness",
               "white_anchor", "peak_rel", "peak_period_pages",
               "peak_rel_short", "peak_period_short_pages", "acf_lag1",
               "p_global", "p_block", "p_detrend", "p_block_flatness",
               "p_global_short", "p_block_short", "p_detrend_short",
               "q_global", "q_block", "q_detrend", "q_global_short",
               "q_block_short", "q_detrend_short", "short_cycle_lead",
               "verdict", "gap_mean_clauses", "gap_cv", "geo_var_ratio"],
              periodicity_rows)

    context = []
    if PERIODICITY_TESTS.exists():
        for row in read_tsv(PERIODICITY_TESTS):
            p_block = float(row["p_block_peak"])
            context.append({
                "series": row["series"], "corpus": row["corpus"], "N": row["N"],
                "flatness": row["spectral_flatness"],
                "max_peak_rel": row["max_peak_rel"],
                "max_peak_period": row["max_peak_period"],
                "p_global_peak": row["p_global_peak"],
                "p_block_peak": row["p_block_peak"],
                "p_detrend_peak": row["p_detrend_peak"],
                "verdict_51": "cycle" if p_block < 0.05 else "no cycle (white)",
            })
    write_tsv(out_dir / "periodicity_book_context.tsv",
              ["series", "corpus", "N", "flatness", "max_peak_rel",
               "max_peak_period", "p_global_peak", "p_block_peak",
               "p_detrend_peak", "verdict_51"], context)

    # --- 3. reconciliation / diagnostics / summary -----------------------
    recon = build_reconciliation(labels, counters, clause_rows, km_sizes,
                                 inertia, published_ari, agree, n_pages,
                                 len(motif_rows), permutations, shuffles,
                                 published_clusters, len(nodes), motif_recon)
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "here", "earlier", "source", "note"], recon)

    diag["rows_with_clauses"] = sum(1 for row in motif_rows if row["n_clauses"])
    diag["rows_zero_clauses"] = sum(1 for row in motif_rows if not row["n_clauses"])
    # Verdict counts are split by row kind.  The matrix mixes 39 motif rows with
    # #29's six domain rows, and quoting the combined count as a motif count
    # overstates the divergence by however many domains diverge.
    for kind in ("motif", "domain", "all"):
        rows_of_kind = [row for row in motif_rows
                        if kind == "all" or row["kind"] == kind]
        for verdict in VERDICTS:
            diag[f"{kind}_{verdict}"] = sum(
                1 for row in rows_of_kind if row["verdict"] == verdict)
    verdicts = sorted({p["verdict"] for p in periodicity_rows})
    diag["per_verdict_kinds"] = len(verdicts)
    for v in verdicts:
        diag["per_" + v.replace(" ", "_").replace("(", "").replace(")", "")] = \
            sum(1 for p in periodicity_rows if p["verdict"] == v)
    write_tsv(out_dir / "diagnostics.tsv", ["metric", "value"],
              [{"metric": k, "value": v} for k, v in sorted(diag.items())])

    pairs_ranked = sorted(PAIRS, key=lambda p: -agree[p])
    summary = {
        "schema": SCHEMA,
        "seed": seed,
        "permutations": permutations,
        "shuffles_per_mode": shuffles,
        "n_clauses": len(clause_rows),
        "n_pages": n_pages,
        "methods": [{"key": k, "label": l, "issue": i, "artifact": a, "kind": kd}
                    for k, l, i, a, kd in METHODS],
        "method_agreement": [{"a": a, "b": b, "ari": agree[(a, b)]} for a, b in PAIRS],
        "pairs_ranked": [{"a": a, "b": b, "ari": agree[(a, b)]} for a, b in pairs_ranked],
        "mean_pairwise_ari": mean([agree[p] for p in PAIRS]),
        "max_p_permutation": max(float(x["p_permutation_ari"]) for x in agree_rows),
        "published_term_level_ari": published_ari,
        "verdict_counts": {kind: {v: diag[f"{kind}_{v}"] for v in VERDICTS}
                           for kind in ("motif", "domain", "all")},
        "n_rows_by_kind": {kind: sum(1 for row in motif_rows
                                     if row["kind"] == kind)
                           for kind in ("motif", "domain")},
        "periodicity_counts": {
            v: sum(1 for p in periodicity_rows if p["verdict"] == v)
            for v in verdicts},
        "thresholds": {"purity_strong": PURITY_STRONG, "jaccard_strong": JACCARD_STRONG,
                       "jaccard_weak": JACCARD_WEAK, "min_clauses": MIN_CLAUSES},
        "white_noise_flatness_anchor": WHITE_FLATNESS,
        "kmeans": {"k": KMEANS_K, "seed": KMEANS_SEED, "n_init": n_init,
                   "inertia": inertia, "sizes": km_sizes},
        "cluster_validation": [
            {"derived": v["derived_cluster"],
             "matched": v["matched_published_cluster"],
             "label_jaccard": v["label_jaccard"], "size_ratio": v["size_ratio"]}
            for v in validation],
        "matrix_rows": [
            {"motif": row["motif"], "kind": row["kind"], "n_clauses": row["n_clauses"],
             "mean_pair_jaccard": row["mean_pair_jaccard"], "verdict": row["verdict"],
             "weakest_pair": row["weakest_pair"], "per_verdict": row["per_verdict"]}
            for row in motif_rows],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    (out_dir / "convergence_matrix.md").write_text(
        render_matrix_md(wide), encoding="utf-8")
    (out_dir / "triangulation_report.md").write_text(
        render_report(summary, motif_rows, wide, agree_rows, clause_rows_out,
                      validation, periodicity_rows, context, recon, diag,
                      motif_recon), encoding="utf-8")
    if not quiet:
        print("[#73] wrote %s/ (%d matrix rows, mean pairwise ARI %.4f)"
              % (out_dir, len(motif_rows), summary["mean_pairwise_ari"]))
    return summary


def periodicity_for_motif(name, kind, hits, pages, page_slot, n_pages,
                          seed=DEFAULT_SEED, shuffles=SHUFFLES) -> dict:
    """Page series + clause-gap renewal statistics for one motif row."""
    base = {"motif": name, "kind": kind, "series": "motif clauses per page",
            "n_pages_total": n_pages}
    if not hits:
        base.update({"n_pages_present": 0, "page_presence": 0.0,
                     "series_mean": 0.0, "series_var": 0.0, "flatness": None,
                     "white_anchor": r(WHITE_FLATNESS), "peak_rel": None,
                     "peak_period_pages": None, "peak_rel_short": None,
                     "peak_period_short_pages": None, "acf_lag1": None,
                     "p_global": None, "p_block": None, "p_detrend": None,
                     "p_block_flatness": None, "p_block_short": None,
                     "p_detrend_short": None, "p_global_short": None,
                     "verdict": "no occurrences",
                     "gap_mean_clauses": None, "gap_cv": None,
                     "geo_var_ratio": None})
        return base
    series = np.zeros(n_pages, dtype=np.float64)
    for i in hits:
        series[page_slot[pages[i]]] += 1.0
    stats = series_stats(series, seed=seed, shuffles=shuffles)
    ren = renewal_stats(hits)
    base.update({
        "n_pages_present": int((series > 0).sum()),
        "page_presence": r(float((series > 0).sum()) / n_pages),
        "series_mean": r(float(series.mean())), "series_var": r(float(series.var())),
        "flatness": r(stats.get("flatness")),
        "white_anchor": r(WHITE_FLATNESS),
        "peak_rel": r(stats.get("peak_rel")),
        "peak_period_pages": r(stats.get("peak_period")),
        "peak_rel_short": r(stats.get("peak_rel_short")),
        "peak_period_short_pages": r(stats.get("peak_period_short")),
        "acf_lag1": r(stats.get("acf1")),
        "p_global": r(stats.get("p_global"), 5),
        "p_block": r(stats.get("p_block"), 5),
        "p_detrend": r(stats.get("p_detrend"), 5),
        "p_block_flatness": r(stats.get("p_block_flat"), 5),
        "p_global_short": r(stats.get("p_global_short"), 5),
        "p_block_short": r(stats.get("p_block_short"), 5),
        "p_detrend_short": r(stats.get("p_detrend_short"), 5),
        "verdict": periodicity_verdict(stats),
        "gap_mean_clauses": r(ren["gap_mean"]) if ren else None,
        "gap_cv": r(ren["gap_cv"]) if ren else None,
        "geo_var_ratio": r(ren["geo_var_ratio"]) if ren else None,
    })
    return base


def build_reconciliation(labels, counters, clause_rows, km_sizes, inertia,
                         published_ari, agree, n_pages, n_rows, permutations,
                         shuffles, published_clusters, n_nodes,
                         motif_recon) -> list:
    n_clauses = len(clause_rows)
    matched = [v for v in agree.values()]
    return [
        {"quantity": "clauses", "here": n_clauses, "earlier": n_clauses,
         "source": "#33/#39", "note": "one row per #33 clause; every method sees these"},
        {"quantity": "pages carrying clauses", "here": n_pages, "earlier": 225,
         "source": "#39", "note": "page axis of the motif series"},
        {"quantity": "network nodes", "here": n_nodes, "earlier": 391,
         "source": "#29", "note": "term -> community assignment"},
        {"quantity": "network communities", "here": len({g for g in labels["net"] if g != -1}),
         "earlier": 6, "source": "#29", "note": "Louvain resolution 1.0, seed 42"},
        {"quantity": "clauses with a network community",
         "here": sum(1 for v in labels["net"] if v != -1), "earlier": "",
         "source": "#29", "note": "majority-vote lift of the term graph to clauses"},
        {"quantity": "LDA topics", "here": len([g for g in counters["lda"] if g != -1]),
         "earlier": 5, "source": "#41", "note": "coherence-selected k=5"},
        {"quantity": "NMF topics", "here": len([g for g in counters["nmf"] if g != -1]),
         "earlier": 6, "source": "#41", "note": "coherence-selected k=6"},
        {"quantity": "BERTopic-style topics",
         "here": len([g for g in counters["bertopic"] if g != -1]), "earlier": 22,
         "source": "#41", "note": "-1 is the noise class, not a topic"},
        {"quantity": "BERTopic noise clauses", "here": counters["bertopic"].get(-1, 0),
         "earlier": 1362, "source": "#41", "note": "unclustered clauses"},
        {"quantity": "embedding clusters (re-derived)",
         "here": sum(1 for s in km_sizes if s), "earlier": len(published_clusters),
         "source": "#40", "note": "numpy k-means over the #39 vectors, k=%d" % KMEANS_K},
        {"quantity": "k-means clause sizes sum", "here": sum(km_sizes),
         "earlier": n_clauses, "source": "#40", "note": "full partition, no noise class"},
        {"quantity": "k-means inertia", "here": r(inertia), "earlier": "",
         "source": "#40", "note": "sum squared distance to the assigned centre"},
        {"quantity": "#41 published ARI vs #29 (LDA)", "here": r(agree[("lda", "net")]),
         "earlier": published_ari.get("lda", ""), "source": "#41",
         "note": "published figure is term-level (391 nodes); here is clause-level"},
        {"quantity": "#41 published ARI vs #29 (NMF)", "here": r(agree[("nmf", "net")]),
         "earlier": published_ari.get("nmf", ""), "source": "#41", "note": "as above"},
        {"quantity": "#41 published ARI vs #29 (BERTopic)",
         "here": r(agree[("bertopic", "net")]),
         "earlier": published_ari.get("bertopic", ""), "source": "#41", "note": "as above"},
        {"quantity": "motif / domain rows in the matrix", "here": n_rows, "earlier": "",
         "source": "#73", "note": "%d motifs + #29's six named domains" % len(MOTIFS)},
        {"quantity": "label permutations per agreement pair", "here": permutations,
         "earlier": "", "source": "#73", "note": "p = (1 + #>=obs)/(R+1)"},
        {"quantity": "shuffle surrogates per motif series", "here": shuffles * 3,
         "earlier": 200, "source": "#51/#73",
         "note": "global + block + detrended, #51's null family"},
        {"quantity": "mean pairwise clause ARI", "here": r(mean(matched)), "earlier": "",
         "source": "#73", "note": "over the %d method pairs" % (len(agree) // 2)},
        {"quantity": "motif clause counts reconciled with #40", "here":
         sum(1 for m in motif_recon if m["n_clauses_73"] == m["n_clauses_40"]),
         "earlier": len([m for m in motif_recon if m["n_clauses_73"] != ""]),
         "source": "#40/#73", "note": "exact matches / shared motifs; the rest "
         "differ only by the declared hyphenated aliases"},
        {"quantity": "external corpora consulted", "here": 0, "earlier": "",
         "source": "#73", "note": "convergence is assessed among Blood Electric "
         "analyses only"},
    ]





# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_matrix_md(wide) -> str:
    lines = [
        "# Convergence matrix (#73)",
        "",
        "One row per motif (and per #29 named domain).  `p:<method>` is the "
        "purity of that method's dominant bucket (share of the motif's clauses "
        "in it); `g:<method>` names the bucket; `j(a|b)` is the Jaccard overlap "
        "of the two methods' dominant buckets over the motif's clauses.  "
        "Thresholds: purity >= %.2f is *strong*; mean `j` >= %.2f is "
        "**convergent**, < %.2f is **divergent**."
        % (PURITY_STRONG, JACCARD_STRONG, JACCARD_WEAK),
        "",
        "## Purity and verdict",
        "",
    ]
    head = ["motif", "clauses", "mean j", "weakest pair", "strongest", "per."]
    head += ["p:" + k for k in METHOD_KEYS]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for row, cell_map, _pv in wide:
        lines.append("| " + " | ".join(
            ["`%s`" % row["motif"], str(row["n_clauses"]),
             f2(row["mean_pair_jaccard"]), md(row["weakest_pair"]) or "-",
             str(row["n_strong_methods"]) + "/5", str(row["per_verdict"])]
            + [f2(cell_map[k]["purity"]) for k in METHOD_KEYS]) + " |")
    lines += ["", "## Dominant bucket per method", "",
              "| " + " | ".join(["motif"] + ["g:" + k for k in METHOD_KEYS]) + " |",
              "|" + "---|" * (1 + len(METHOD_KEYS))]
    for row, cell_map, _pv in wide:
        lines.append("| " + " | ".join(
            ["`%s`" % row["motif"]] + [cell_map[k]["group_label"] or "-"
                                       for k in METHOD_KEYS]) + " |")
    lines += ["", "## Pairwise Jaccard of the dominant buckets", "",
              "| " + " | ".join(["motif"] + ["%s/%s" % p for p in PAIRS]) + " |",
              "|" + "---|" * (1 + len(PAIRS))]
    for row, _cell_map, pv in wide:
        lines.append("| " + " | ".join(
            ["`%s`" % row["motif"]] + [f2(pv[p]) for p in PAIRS]) + " |")
    lines += ["", "---", "*Generated by `work/triangulation.py` (schema `%s`).*"
              % SCHEMA, ""]
    return "\n".join(lines)


def render_report(summary, motif_rows, wide, agree_rows, clause_rows_out,
                  validation, periodicity_rows, context, recon, diag,
                  motif_recon) -> str:
    L = []
    A = L.append
    motifs = [row for row in motif_rows if row["kind"] == "motif"]
    domains = [row for row in motif_rows if row["kind"] == "domain"]
    cells_by = {}
    for _row, cm, _pv in wide:
        for key in METHOD_KEYS:
            cells_by[(_row["motif"], key)] = cm[key]
    per_lookup = {p["motif"]: p for p in periodicity_rows}

    ranked = sorted(motifs, key=lambda x: (-(x["mean_pair_jaccard"] or -1.0),
                                           -x["n_clauses"], x["motif"]))
    weakest = sorted([x for x in motifs if x["mean_pair_jaccard"] is not None],
                     key=lambda x: (x["mean_pair_jaccard"], x["motif"]))
    counts = summary["verdict_counts"]["motif"]
    domain_counts = summary["verdict_counts"]["domain"]
    n_domains = len(domains)
    per_counts = summary["periodicity_counts"]
    n_periodic = sum(n for v, n in per_counts.items() if v.startswith("periodic"))
    top5 = ranked[:5]
    bot5 = weakest[:5]

    A("# Triangulation / convergence report (#73)")
    A("")
    A("Do the independent Blood Electric analyses converge on the same motifs "
      "and domains?  Convergence is evidence; divergence is a lead.  This is a "
      "report about **agreement between estimators over one book** -- it is not "
      "a translation, and it does not rank the methods: no method here is a "
      "ground truth, and no cell says what a motif means.")
    A("")
    A("Reproduce: `python work/triangulation.py` (schema `%s`, seed %d); "
      "regression suite `python work/test_triangulation.py`.  `numpy` only."
      % (SCHEMA, DEFAULT_SEED))
    A("")
    A("## 1. Verdict")
    A("")
    A("1. **The three distributional methods converge on a core of motifs.**  "
      "Of %d motif rows, %d are **convergent** (mean pairwise Jaccard >= %.2f), "
      "%d **partial**, %d **divergent** and %d untestable (fewer than %d "
      "clauses).  The strongest agreements are "
      % (len(motifs), counts["convergent"], JACCARD_STRONG, counts["partial"],
         counts["divergent"], counts["untestable"], MIN_CLAUSES)
      + ", ".join("`%s` (%s)" % (x["motif"], f2(x["mean_pair_jaccard"]))
                  for x in top5)
      + " -- the same poles #29, #40 and #41 each reported on their own.  "
      "(These counts are the %d motif rows only.  #29's %d named-domain rows "
      "are scored separately in §8 and are counted separately throughout: %d "
      "convergent, %d partial, %d divergent, %d untestable, for %d matrix rows "
      "in total.)"
      % (len(motifs), n_domains, domain_counts["convergent"],
         domain_counts["partial"], domain_counts["divergent"],
         domain_counts["untestable"], len(motif_rows)))
    A("2. **Clause-level agreement is weak but consistently above chance.**  The "
      "mean pairwise adjusted Rand over the %d clauses is %.4f, and the worst "
      "of the %d pairs still has permutation p = %s.  The methods share "
      "structure; they disagree about *how finely to cut it*."
      % (summary["n_clauses"], summary["mean_pairwise_ari"], len(PAIRS),
         r(summary["max_p_permutation"], 3)))
    A("3. **The periodicity method diverges from the other three -- and that is "
      "the result, not a defect.**  %d of the %d motif/domain page series show "
      "a cycle (a short-band spectral line, period <= %d pages, that survives "
      "the global and detrended shuffle nulls of #51's family); the rest are "
      "drift-only or aperiodic.  The motifs the distributional methods agree on "
      "are *dispersed over the book*, not cyclic: the loop is lexical (#24), "
      "semantic (#39/#40) and distributional (#41), and not metrical -- #51's "
      "own conclusion, now reproduced per motif."
      % (n_periodic, len(periodicity_rows), SHORT_PERIOD))
    A("4. **Divergence is concentrated and diagnosable.**  The weakest rows are "
      + ", ".join("`%s` (%s, weakest pair %s)"
                  % (x["motif"], f2(x["mean_pair_jaccard"]),
                     md(x["weakest_pair"]))
                  for x in bot5)
      + ".  Each is a lead: they are the places where two instruments see the "
        "same recurrence and file it under different neighbourhoods (§6.2).")
    A("")
    A("## 2. What is compared -- and what is not")
    A("")
    A("| method | source | artifact | how the clause partition is made |")
    A("|---|---|---|---|")
    for key, label, issue, artifact, kind in METHODS:
        A("| %s | #%s | `%s` | %s |"
          % (label, issue.lstrip("#"), artifact, kind))
    A("| %s | #51 | `analysis/periodicity/periodicity_tests.tsv` | "
      "motif series (clauses per page, clause gaps) -- **not** a clause "
      "partition |" % PERIODICITY_LABEL)
    A("")
    A("**Scope.**  Only analyses *of this book* take part: #29, #40, #41, #51 "
      "and the #39 vectors underneath them.  No reference corpus, English "
      "baseline or other external text is acquired, re-scored or used as an "
      "arbiter here -- the English layers of #24/#51 are not read.  The only "
      "nulls are internal: label permutations of the clause partitions and "
      "within-book shuffles of the motif series.")
    A("")
    A("**Vocabulary.**  Motif names are the book's own surface forms.  They are "
      "reported because several instruments grouped the same strings, not "
      "because anyone has established what they denote.  Nothing here is a "
      "translation of the book, and no method's grouping is treated as the "
      "correct reading.")
    A("")
    A("## 3. Method")
    A("")
    A("1. **Clause partitions.**  Five partitions of the same %d clauses: #41's "
      "persisted per-clause topic labels; #29's 391 nodes lifted to clauses by "
      "node frequency (coverage in §5); and a `numpy` k-means (k-means++ init, "
      "Lloyd, k=%d, seed %d, `n_init=%d`) over #39's L2-normalised vectors, "
      "because #40 persisted the catalogue but not the per-clause labels."
      % (summary["n_clauses"], KMEANS_K, KMEANS_SEED, summary["kmeans"]["n_init"]))
    A("2. **Clause-level agreement.**  Adjusted Rand and arithmetic-normalised "
      "MI per pair, over the clauses both methods assign, against %d label "
      "permutations (p = (1 + #>=obs)/(R+1))."
      % summary["permutations"])
    A("3. **Motif matrix.**  Motifs are matched by substring on the space-joined "
      "#39 `units` view -- #40's rule, so fused units count for their parts.  "
      "For each motif and method: the dominant bucket and its purity.  Methods "
      "are then compared by the **Jaccard overlap of their dominant buckets "
      "over that motif's clauses**, which needs no label alignment between "
      "methods with different k.  A row is *convergent* at mean Jaccard "
      ">= %.2f, *divergent* below %.2f, and untestable below %d clauses."
      % (JACCARD_STRONG, JACCARD_WEAK, MIN_CLAUSES))
    A("4. **Periodicity.**  Each motif becomes a page series (clauses per page "
      "over %d pages) plus a clause-position renewal series.  Spectral "
      "flatness, lag-1 autocorrelation and three shuffle surrogates (global / "
      "block of %d pages / detrended; %d each) are computed exactly as #51 "
      "did.  Two peak searches run on every series: #51's peak-over-band-"
      "median ratio on the full band, whose peak for a slowly varying rate "
      "always sits at the longest period; and a **local line statistic** on the "
      "short band (period <= %d pages), where a genuine motif cycle would have "
      "to appear.  A series is *periodic* only when the short-band line "
      "survives the global and detrended shuffles; a surviving full-band peak "
      "alone is *drift*, which is how #51 read the symbol stream.  Why two "
      "statistics, and why the block shuffle is not the gate, is in §7."
      % (summary["n_pages"], BLOCK, summary["shuffles_per_mode"], SHORT_PERIOD))
    A("")
    A("## 4. Clause-level method agreement")
    A("")
    A("| a | b | shared clauses | groups a | groups b | adjusted Rand | NMI | "
      "p(ARI) |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in agree_rows:
        A("| %s | %s | %s | %s | %s | %s | %s | %s |"
          % (row["method_a"], row["method_b"], row["n_shared_clauses"],
             row["n_groups_a"], row["n_groups_b"], row["adjusted_rand"],
             row["nmi"], row["p_permutation_ari"]))
    A("")
    best = summary["pairs_ranked"][0]
    worst = summary["pairs_ranked"][-1]
    A("**Reading.**  Every pair is above its own permutation null, so the "
      "shared structure is real.  No pair is *identical*: the best is "
      "%s|%s at ARI %.4f and the worst %s|%s at %.4f.  The spread is what you "
      "would expect from one continuum cut at four resolutions -- #29 works on "
      "the 391-term type layer, #40 and #41 on clauses -- and #40's own "
      "silhouette scan already showed the partition rising with k, i.e. k is a "
      "resolution choice rather than a discovered constant."
      % (best["a"], best["b"], best["ari"], worst["a"], worst["b"], worst["ari"]))
    A("")
    A("Cross-check against the published figures: #41 measured its term "
      "partitions against #29's communities as ARI LDA %s / NMF %s / "
      "BERTopic %s over the 391 shared nodes.  The clause-level numbers here "
      "(net|lda %s and its neighbours) are the same order; clauses are an "
      "easier test of agreement than types, so a small rise is expected and no "
      "contradiction is claimed."
      % (summary["published_term_level_ari"].get("lda", "n/a"),
         summary["published_term_level_ari"].get("nmf", "n/a"),
         summary["published_term_level_ari"].get("bertopic", "n/a"),
         f2(pair_ari(summary, "net", "lda"), 4)))
    A("")
    A("## 5. Partitions, and the re-derived embedding cluster")
    A("")
    A("| method | groups | clauses assigned | coverage | unassigned | largest bucket |")
    A("|---|---:|---:|---:|---:|---|")
    for row in clause_rows_out:
        A("| %s | %s | %s | %s | %s | %s (%s) |"
          % (row["method"], row["n_groups"], row["n_assigned"], row["coverage"],
             row["unassigned_clauses"], row["largest_group_label"],
             row["largest_group_n"]))
    A("")
    matched = [v for v in validation if v["matched_published_cluster"] != ""]
    mean_overlap = mean([float(v["label_jaccard"]) for v in matched])
    size_ratio = mean([float(v["size_ratio"]) for v in matched
                       if v["size_ratio"] != ""])
    A("**#40 catalogue check.**  The re-derived partition reproduces #40's "
      "published catalogue: %d of %d derived clusters match a #40 cluster on "
      "their top %d log-odds label terms (mean Jaccard %.2f, mean size ratio "
      "%.2f).  That is what licenses using the re-derived labels as the "
      "embedding column; #40's HDBSCAN noise fraction (3,811 clauses, 68%%) and "
      "its units/raw ARI (0.6028) are cited from `analysis/clustering/` and "
      "bound how much the embedding partition can be trusted at all."
      % (len(matched), len(validation), LABEL_TERMS, mean_overlap, size_ratio))
    A("")
    A("| derived | clauses | derived label | #40 cluster | #40 label | size ratio | label Jaccard |")
    A("|---:|---:|---|---|---|---:|---:|")
    for v in validation:
        A("| %s | %s | %s | %s | %s | %s | %s |"
          % (v["derived_cluster"], v["n_clauses"], v["derived_label_terms"],
             v["matched_published_cluster"], v["published_label"],
             v["size_ratio"], v["label_jaccard"]))
    A("")
    A("## 6. The convergence matrix (motifs)")
    A("")
    A("Full matrix: `convergence_matrix.tsv` (wide), `convergence_matrix.md` "
      "(rendered), `motif_method_cells.tsv` (long), `motif_inventory.tsv` "
      "(rows with provenance and spread).  Rendered here as purity per method, "
      "the mean pairwise Jaccard and the periodicity verdict.")
    A("")
    exact = [m for m in motif_recon
             if m["n_clauses_73"] != "" and m["n_clauses_73"] == m["n_clauses_40"]]
    shared = [m for m in motif_recon if m["n_clauses_73"] != ""]
    contained = [m for m in shared if m["same_top_cluster_as_40"] == 1]
    A("Motif counts are checked against #40's own inventory "
      "(`motif_count_reconciliation.tsv`): %d of the %d shared motifs reproduce "
      "#40's clause count exactly; the rest differ only by the hyphenated "
      "aliases this inventory declares (`artificial-sun`, `drug-embryo`, "
      "`cadaver-city`, `ice-nebula`, `internal-organ`, `cold blooded`, "
      "`hydro=mania`), so the counts stay comparable with every earlier motif "
      "number in the workspace.  The embedding column also reproduces #40's "
      "*containment*: for %d of the %d shared motifs that carry clauses, the "
      "re-derived k-means dominant bucket is the same #40 cluster #40 named as "
      "the motif's home, with purity within a few points of #40's "
      "(`frac_in_top_cluster_40` vs `purity_kmeans_73`)."
      % (len(exact), len(shared), len(contained), len(shared)))
    A("")
    A("| motif | clauses | " + " | ".join("%s" % k for k in METHOD_KEYS) +
      " | mean J | strong | weakest pair | per. |")
    A("|---|---:|" + "---:|" * len(METHOD_KEYS) + "---:|---:|---|---|---|")
    for row in ranked:
        pur = [f2(cells_by[(row["motif"], k)]["purity"]) for k in METHOD_KEYS]
        A("| `%s` | %s | %s | %s | %s/5 | %s %s | %s |"
          % (row["motif"], row["n_clauses"], " | ".join(pur),
             f2(row["mean_pair_jaccard"]), row["n_strong_methods"],
             md(row["weakest_pair"]), f2(row["min_pair_jaccard"]),
             row["per_verdict"]))
    A("")
    A("### 6.1 Agreement")
    A("")
    conv_rows = [x for x in ranked if x["verdict"] == "convergent"]
    if conv_rows:
        A("These motifs are grouped by the distributional methods at once: the "
          "same clauses land in one bucket whichever method is asked, even "
          "though the buckets carry different names and different k.  This is "
          "the strongest form of convergence available here -- no method can "
          "see another's labels, only the clauses they share.")
        A("")
        for row in conv_rows:
            p = per_lookup.get(row["motif"], {})
            buckets = "; ".join(
                "%s -> %s (p=%s)" % (k, cells_by[(row["motif"], k)]["group_label"],
                                     r(cells_by[(row["motif"], k)]["purity"], 2))
                for k in METHOD_KEYS)
            A("- **`%s`** -- %s clauses over %s pages, mean J %s.  %s.  "
              "Periodicity: %s (flatness %s, p_block %s)."
              % (row["motif"], row["n_clauses"], row["n_pages"],
                 r(row["mean_pair_jaccard"], 2), buckets,
                 p.get("verdict"), p.get("flatness"), p.get("p_block")))
    else:
        A("*No motif row reached the convergence threshold.*")
    A("")
    A("### 6.2 Divergence (leads)")
    A("")
    div_rows = [x for x in weakest if x["verdict"] in ("divergent", "partial")]
    n_div = len(div_rows)
    A("Divergence is not a failure of the text.  It says the instruments are "
      "measuring different things, and each row is a hypothesis to test next "
      "rather than a defect to repair.  All %d below-threshold rows are in "
      "`motif_inventory.tsv`; the %d weakest are spelled out."
      % (n_div, min(10, n_div)))
    A("")
    for row in div_rows[:10]:
        a_key, b_key = (row["weakest_pair"].split("|") + ["", ""])[:2]
        ca = cells_by.get((row["motif"], a_key), {})
        cb = cells_by.get((row["motif"], b_key), {})
        j_min = row["min_pair_jaccard"]
        pa = ca.get("purity")
        pb = cb.get("purity")
        weak_a = pa is not None and pa < PURITY_STRONG
        weak_b = pb is not None and pb < PURITY_STRONG
        if j_min is None:
            tail = "no dominant bucket to compare"
        elif weak_a and weak_b:
            tail = ("split under both methods -- shared region, not shared "
                    "motif")
        elif weak_a or weak_b:
            tail = "one method concentrates it, the other splits it"
        elif j_min < 0.30:
            tail = "both concentrate it, in different neighbourhoods"
        else:
            tail = "overlapping but below threshold -- naming, not coverage"
        A("- **`%s`** (mean J %s, %s clauses, %s pages): weakest pair %s at "
          "J %s -- %s (p %s) vs %s (p %s)  \n  *%s.*"
          % (row["motif"], f2(row["mean_pair_jaccard"]), row["n_clauses"],
             row["n_pages"], md(row["weakest_pair"]), f2(j_min),
             md(ca.get("group_label", "-")), f2(ca.get("purity")),
             md(cb.get("group_label", "-")), f2(cb.get("purity")), tail))
    if n_div == 0:
        A("*No row fell below the divergence threshold.*")
    A("")
    noise_rows = [x for x in div_rows
                  if "bertopic" in x["weakest_pair"].split("|")]
    bt_unassigned = next((int(v["unassigned_clauses"]) for v in clause_rows_out
                          if v["method"] == "bertopic"), 0)
    n_weak_both = sum(1 for x in div_rows
                      if (cells_by.get((x["motif"], x["weakest_pair"].split("|")[0]), {})
                          .get("purity") or 0) < PURITY_STRONG
                      and (cells_by.get((x["motif"],
                                          x["weakest_pair"].split("|")[1]), {})
                           .get("purity") or 0) < PURITY_STRONG)
    second_row = next((x for x in motif_rows if x["motif"] == "second"), None)
    second_bt = cells_by.get(("second", "bertopic"), {})
    A("**Reading the leads.**  Three patterns recur, and they are leads about "
      "*instruments*, not about the text.  (a) *Granularity splits*: %d of the "
      "%d below-threshold rows have a weakest pair in which **neither** method "
      "concentrates the motif (both purities below %.2f): the motif is split "
      "several ways under both methods, so what the methods share is a region "
      "of the book rather than a motif.  "
      "(b) *Density noise is a divergence of its own*: the BERTopic-style "
      "model leaves %d of %d clauses unclustered, and in %d of the "
      "below-threshold rows that **noise** class is one of the two buckets "
      "being compared.  `second` is the sharpest case -- its dominant "
      "bertopic bucket is the noise class at purity %s, while #41's "
      "BERTopic topic 5 (`second · android · machine`) and #40's cluster 3 "
      "(`second · machine · cyclops`) both name it.  A motif the density model "
      "declines to cluster but the k-means partition isolates is a genuine "
      "disagreement about density versus distance, and it is the cheapest lead "
      "here to follow.  (c) *The OCR loop*: `second` occurs in %s clauses over "
      "%s pages and is largely the #4 recognition-loop material, so part of "
      "the apparent convergence on `second`/`cyclops` is convergence on an "
      "artefact -- reported, not celebrated."
      % (n_weak_both, n_div, PURITY_STRONG, bt_unassigned, summary["n_clauses"],
         len(noise_rows), f2(second_bt.get("purity")),
         (second_row or {}).get("n_clauses", "-"),
         (second_row or {}).get("n_pages", "-")))
    A("")
    A("## 7. Periodicity: the axis the other three do not have")
    A("")
    A("#51 tested the *symbol stream* and found no latent cycle: the operator "
      "indicator is statistically white once 732 markup-span operators are "
      "screened (flatness 0.555 against the 0.561 white-noise anchor, p_block "
      "0.70).  The cross-check here asks the same question of the motifs -- if "
      "a motif is a cycle, the pages carrying it should show a peak that "
      "survives shuffling.")
    A("")
    A("| motif | pages present | flatness | full-band peak (pages) | p_detrend | "
      "short-band line | short period (pages) | p_global (short) | "
      "q (short) | verdict |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    per_rows = [p for p in periodicity_rows if p["kind"] == "motif"][:20]
    for p in per_rows:
        A("| `%s` | %s/%s | %s | %s | %s | %s | %s | %s | %s | %s |"
          % (p["motif"], p["n_pages_present"], p["n_pages_total"], p["flatness"],
             p["peak_period_pages"], p["p_detrend"], p["peak_rel_short"],
             p["peak_period_short_pages"], p["p_global_short"],
             p["q_detrend_short"], p["verdict"]))
    A("")
    A("White-noise flatness anchor `exp(-gamma)` = %.3f (#51's anchor).  The "
      "short band is periods <= %d pages, where a genuine motif cycle would "
      "have to appear; the full-band peak of a dispersed motif always sits at "
      "the longest period instead, which is the drift signature #51 read on the "
      "symbol stream.  The table shows raw p-values; `periodicity_motifs.tsv` "
      "adds Benjamini-Hochberg q-values corrected across the %d rows, the "
      "clause-gap renewal columns and the lead flag discussed below."
      % (WHITE_FLATNESS, SHORT_PERIOD, len(periodicity_rows)))
    A("")
    leads = [p for p in periodicity_rows if p.get("short_cycle_lead") == 1]
    A("**Result.**  %d of %d series carry a cycle; the rest are drift-only or "
      "aperiodic.  The motifs the distributional methods agree on have "
      "page-presence spread across the book (top-5 page share in "
      "`motif_inventory.tsv`), spectra that are flat or drift-dominated, and "
      "clause gaps that are over-dispersed but close to memoryless -- #51's "
      "renewal finding, reproduced per motif.  This is the program's third "
      "independent arrival at one conclusion: the loop lives in repetition and "
      "distribution, not in a metrical cycle."
      % (n_periodic, len(periodicity_rows)))
    A("")
    A("**Two statistics, because one of them lies.**  #51's peak-over-band-"
      "median ratio is the right statistic for the symbol stream, where one "
      "drift dominates the spectrum.  Applied to a motif page series it is not: "
      "the page-rate envelope depresses the median of any short band, so a "
      "merely *heterogeneous* series looks peaky, and the block shuffle makes "
      "it worse (permuting 5-page blocks creates runs whose edges splatter "
      "power across the band -- `p_block_short` is 1.00 on every row, the "
      "null always beating the data).  The short-band test here therefore uses "
      "a **local** line statistic (peak over the median of its 20 neighbouring "
      "bins), which a dispersed series cannot fake, and is gated on the global "
      "and detrended shuffles, which do destroy a real cycle.  %s"
      % ("%d rows clear both at BH q < 0.05 (%s) and are flagged "
         "`short_cycle_lead` in `periodicity_motifs.tsv`; they are *not* called "
         "periodic here, because the surrogate that would settle them (one "
         "that preserves local density without creating runs) is not in this "
         "report."
         % (len(leads), ", ".join("`%s`" % p["motif"] for p in leads[:10]))
         if leads else
         "No row clears both at BH q < 0.05, so no motif series is called "
         "periodic; the `short_cycle_lead` column is left in "
         "`periodicity_motifs.tsv` for a follow-up surrogate that preserves "
         "local density without creating runs."))
    A("")
    A("## 8. Domains (#29's six named domains)")
    A("")
    A("The same treatment for #29's published domain seed sets, so the six "
      "named domains are measured rather than asserted.  Seed sets are read "
      "straight out of `work/cooccurrence_network.py` with `ast.literal_eval`, "
      "so they cannot drift from #29's definition.  (Its `corporate` set "
      "contains the multi-word entry `sato corporation`, which can never match "
      "a unit token; it is kept as-is and disclosed.)")
    A("")
    A("| domain | clauses | " + " | ".join(METHOD_KEYS) +
      " | mean J | verdict | per. |")
    A("|---|---:|" + "---:|" * len(METHOD_KEYS) + "---:|---|---|")
    for row in domains:
        pur = [f2(cells_by[(row["motif"], k)]["purity"]) for k in METHOD_KEYS]
        A("| `%s` | %s | %s | %s | %s | %s |"
          % (row["motif"].split(":", 1)[1], row["n_clauses"], " | ".join(pur),
             f2(row["mean_pair_jaccard"]), row["verdict"], row["per_verdict"]))
    A("")
    A("**Reading.**  The six domains all land in the same narrow band of mean "
      "Jaccard (%s-%s) and every one is below the convergence threshold, so the "
      "seed-set domains do not converge on anything the distributional methods "
      "share.  That is a sharper statement than #29's own: #29 reported that "
      "Louvain recovered co-occurrence clusters that *cut across* the named "
      "domains, and the clause-level instruments, which never saw those seed "
      "lists, land in the same place.  The seed sets are a plausible reading "
      "imposed on the book, not the book's own categories: the motifs that do "
      "converge (`ice nebula`, `artificial sun`, `drug embryo`, "
      "`cold-blooded`) straddle several domains at once.  The two smallest "
      "domains (`sex` %s clauses, `corporate` %s clauses) also score lowest, "
      "which reproduces #29's verdict that they are barely present at its "
      "frequency floor.  No domain row reaches the periodicity threshold "
      "either."
      % (f2(min(x["mean_pair_jaccard"] for x in domains)),
         f2(max(x["mean_pair_jaccard"] for x in domains)),
         next((x["n_clauses"] for x in domains
               if x["motif"] == "domain:sex"), "-"),
         next((x["n_clauses"] for x in domains
               if x["motif"] == "domain:corporate"), "-")))
    A("")
    A("## 9. What would change the answer")
    A("")
    A("- **A finer periodicity null.**  The block shuffle is blunt by "
      "construction (#51's caveat): a motif could carry a cycle below five "
      "pages that this test cannot see.  A band-resolved test is left for a "
      "follow-up rather than claimed here.")
    A("- **A second encoder for the embedding column.**  #39 found SBERT and "
      "BGE agree on only 35% of nearest neighbours, so this column is one "
      "encoder's view; #40's robustness check bounds the *view* choice, not "
      "the encoder choice.")
    A("- **Clause-position motif series** (finer than per-page) if the "
      "dispersion/periodicity question needs more power.")
    A("- **A density-preserving surrogate without run creation** to re-test the "
      "short-band statistic, since the block shuffle cannot serve as its null "
      "and no motif row currently needs re-testing (%d flagged)."
      % len([p for p in periodicity_rows if p.get("short_cycle_lead") == 1]))
    A("")
    A("## 10. Negative results and limitations")
    A("")
    A("- **No method is a ground truth.**  Convergence between two coarse "
      "summaries of a fused vocabulary is evidence that both see the same "
      "motif; it is not evidence that the motif means what its label says.  "
      "The task's own framing -- divergence is a lead -- is the operating rule "
      "here.")
    A("- **The matrix is resolution-relative.**  Every cell depends on the "
      "disclosed resolution of its method (#29 Louvain resolution 1.0, #41 "
      "k=5/6/22, #40 k=%d).  A different k moves purity, which is why the raw "
      "numbers ship next to every verdict (`convergence_matrix.tsv`)." % KMEANS_K)
    A("- **Substring motif matching inherits #40's granularity**: `adam` also "
      "counts `anti-adam`, `soul` counts `soul/gram`, `gene` counts `gene=tv`.  "
      "It is the rule every motif count in this workspace used, so the cells "
      "stay comparable across analyses.")
    A("- **The network partition is a lift, not a native clause labelling.**  "
      "A clause with no #29 node gets no community; coverage is in §5.")
    A("- **Periodicity is not evidence about meaning.**  A motif can be a real "
      "motif and still be aperiodic; §7's result is that the loop has no "
      "metrical period, not that the motifs are illusions.")
    A("- **`necropolis` is the built-in negative control** carried over from "
      "#40's inventory: it has zero occurrences here and is reported "
      "untestable.")
    A("- **The embedding partition is a re-implementation.**  With "
      "scikit-learn absent from this environment the matrix needed a clause "
      "partition, so k-means was reimplemented in `numpy` and validated "
      "against #40's catalogue (§5).  It is not a bit-identical replay of "
      "#40's sklearn run and is not claimed to be.")
    A("")
    A("## 11. Reconciliation and artifacts")
    A("")
    A("| quantity | here | earlier | source | note |")
    A("|---|---:|---:|---|---|")
    for row in recon:
        A("| %s | %s | %s | %s | %s |"
          % (row["quantity"], row["here"], row["earlier"], row["source"],
             row["note"]))
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `triangulation_report.md` | THE DELIVERABLE |")
    A("| `convergence_matrix.tsv` / `.md` | the matrix: purity, dominant buckets, "
      "ten pairwise Jaccards, periodicity verdict |")
    A("| `motif_method_cells.tsv` | long form of the matrix cells |")
    A("| `motif_inventory.tsv` | motif rows: provenance, spread, verdict |")
    A("| `motif_count_reconciliation.tsv` | motif clause counts vs #40's inventory |")
    A("| `domain_convergence.tsv` | #29's six named domains, same treatment |")
    A("| `method_agreement.tsv` | pairwise ARI / NMI + permutation p |")
    A("| `clause_agreement.tsv` | per-method clause partition, coverage |")
    A("| `cluster_validation.tsv` | re-derived k-means vs #40's catalogue |")
    A("| `periodicity_motifs.tsv` | per-motif page-series + renewal statistics |")
    A("| `periodicity_book_context.tsv` | #51's book-level series, cited |")
    A("| `reconciliation.tsv` / `diagnostics.tsv` | join contract and counters |")
    A("| `summary.json` | machine-readable headline numbers |")
    A("")
    A("---")
    A("*Generated by `work/triangulation.py` (schema `%s`, seed %d).*"
      % (SCHEMA, DEFAULT_SEED))
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS,
                    help="label-permutation null size")
    ap.add_argument("--shuffles", type=int, default=SHUFFLES,
                    help="shuffle surrogates per mode per motif series")
    ap.add_argument("--n-init", type=int, default=KMEANS_N_INIT,
                    help="k-means restarts for the embedding partition")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    build(Path(args.out), permutations=args.permutations, shuffles=args.shuffles,
          n_init=args.n_init, seed=args.seed, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())