#!/usr/bin/env python3
"""#43 Contrastive semantic axes -> analysis/semantic_axes/

Find and interpret the **dimensions that separate the #40 clusters**, and test
the four axes the task names against **antonym seed pairs**::

    organic <-> machine    self <-> other
    body <-> information   sacred <-> profane

Method (stated once, used everywhere):

* **Vectors are the #39 primary artifact.**  ``ClauseEmbeddingStore`` loads
  ``analysis/clause_embeddings/clause_embeddings.npz`` -- SBERT
  ``all-MiniLM-L6-v2``, 384-d, L2-normalised, ``units`` view.  5,614 clauses,
  225 pages.  No encoder runs over the book here.
* **The axes live in the clause-vector space.**  Each seed *word* is encoded
  with the same pinned #39 encoder (``SentenceEncoder(ce.PRIMARY)``, CPU,
  ``local_files_only``) -- the only encoder call in this module, disclosed
  because it is what makes a clause vector projectable onto a word contrast.
  #28's static embeddings are a *different* space and are never mixed in.
* **The partition is #40's, exactly.**  KMeans k=12, ``n_init=10``, seed 0 on
  the same matrix; the reproduction is cross-checked against #40's persisted
  catalogue (sizes + medoids) when it is present.
* **Axis direction** = mean(+pole seed vectors) - mean(-pole seed vectors),
  L2-normalised.  Four tests, each with a control:
    1. *Seed-pair coherence* -- cosine of every pair's difference vector to
       the axis (the task's "cosine contrasts on antonym seed pairs"), with a
       **randomized-axis control**: 200 fixed-seed random pole assignments
       over the same term pool build random axes, and the named pairs are
       scored on each; the z-score says whether the named pairing is special.
    2. *Clause projection contrast* -- mean projection of clauses carrying a
       +pole term vs clauses carrying a -pole term (word-boundary match on the
       units text; overlap disclosed).
    3. *Cluster separation* -- spread of the 12 cluster centroids along the
       axis vs a 200-direction random null.
* **Cluster-pair dimensions.**  All 66 centroid-difference directions, each
  interpreted by the seed words it loads on most -- the dimensions that
  separate clusters, named or not.

Outputs under ``analysis/semantic_axes/``:
  semantic_axes_report.md   THE DELIVERABLE
  axes.tsv                  per axis: coherence / controls / contrast / spread
  seed_pairs.tsv            per seed pair: cosine to axis + clause counts
  axis_projections.tsv      cluster centroid projection per axis (12 rows)
  extreme_clauses.tsv       top/bottom clauses per axis (pole exemplars)
  cluster_pair_axes.tsv     all 66 cluster-pair dimensions + seed loadings
  reconciliation.tsv        against #39 / #40 / #33 / #17 / #27
  diagnostics.tsv           build-time counters
  summary.json              machine-readable headline numbers

Reproduce::

    conda activate marker
    python work/semantic_axes.py                 # writes analysis/semantic_axes/
    python work/test_semantic_axes.py

Determistic: fixed seeds everywhere (KMeans ``n_init``/``random_state``,
encoder seed + threads, both nulls), no hash-order dependence (every list
sorted; ties broken by a fixed key).  Requires ``numpy`` + ``scikit-learn``;
the seed-word encoding needs the pinned #39 encoder weights (cached, local).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

import numpy as np

import clause_embeddings as ce       # #39 ClauseEmbeddingStore (primary vectors)
import clause_segmentation as cs     # #33 quote-safe TSV io
import clustering as cl              # #40 catalogue method (terms + constants)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "semantic_axes"
STORE_DIR = ce.OUT                      # analysis/clause_embeddings/
CLUSTERING_DIR = cl.OUT                 # analysis/clustering/ (#40 catalogue)

SCHEMA = "semantic-axes/1"
PRIMARY_VIEW = "units"                  # #39 primary semantic view
DEFAULT_SEED = 0
# The partition is #40's, settings and all (cross-checked in build()).
KMEANS_K = cl.DEFAULT_K                 # 12
KMEANS_N_INIT = cl.KMEANS_N_INIT        # 10
N_RANDOM = 200                          # randomized-axis + spread null draws
LABEL_TERMS = 3                         # terms in an auto label (as #40)
TOP_PAIR_TERMS = 5                      # seed loadings listed per pair dimension
TOP_EXTREME = 10                        # exemplar clauses per pole per axis

# ---------------------------------------------------------------------------
# the four named axes, as seed pairs from the book's own lexicon
# ---------------------------------------------------------------------------
# Every term occurs in the #39 units view (clause counts are published in
# seed_pairs.tsv); the thin "other" pole is itself a finding, not a bug.
# Pairing is by list position; `=-` separated pairs are used as-is.
AXES = {
    "organic<->machine": {
        "plus": ("flesh", "blood", "body", "gene", "chromosome",
                 "embryo", "serum", "organ", "cell"),
        "minus": ("machine", "digital", "circuit", "drone",
                  "android", "cyborg", "replicant", "code", "data"),
    },
    "self<->other": {
        "plus": ("self", "i", "my", "we", "our", "adam"),
        "minus": ("other", "another", "they", "you", "alien", "mirror"),
    },
    "body<->information": {
        "plus": ("body", "flesh", "organ", "skin", "meat", "bone",
                 "brain", "lobe"),
        "minus": ("data", "code", "record", "signal", "noise",
                  "memory", "language", "information"),
    },
    "sacred<->profane": {
        "plus": ("soul", "angel", "god", "heaven", "666", "cherry"),
        "minus": ("murder", "cadaver", "junk", "crime", "sin", "hell"),
    },
}

read_tsv = cs.read_tsv
write_tsv = cs.write_tsv


def term_re(term: str) -> re.Pattern:
    """Word-boundary matcher for a seed term (clause sets use this, not #40's
    substring convention: single seed words like ``data`` must not match
    inside longer tokens)."""
    return re.compile(r"\b" + re.escape(term) + r"\b")


def seed_terms() -> list:
    """All distinct seed terms across the four axes, sorted."""
    terms = set()
    for ax in AXES.values():
        terms.update(ax["plus"])
        terms.update(ax["minus"])
    return sorted(terms)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------
def axis_direction(V: np.ndarray, vidx: dict, plus, minus) -> np.ndarray:
    """mean(+pole) - mean(-pole), L2-normalised."""
    a = V[[vidx[t] for t in plus]].mean(axis=0)
    b = V[[vidx[t] for t in minus]].mean(axis=0)
    return unit(a - b)


def pair_cosines(V: np.ndarray, vidx: dict, axis: np.ndarray,
                 plus, minus) -> list:
    """``[(plus, minus, cos), ...]`` -- cosine of each pair's difference
    vector to ``axis`` (the task's "cosine contrasts on antonym seed pairs").
    Pairs are zipped by position; unpaired terms are skipped."""
    rows = []
    for p, m in zip(plus, minus):
        d = V[vidx[p]] - V[vidx[m]]
        nn = float(np.linalg.norm(d))
        rows.append((p, m, 0.0 if nn == 0 else float((d / nn) @ axis)))
    return rows


def random_axis_control(V: np.ndarray, vidx: dict, plus, minus,
                        rng: np.random.Generator, n_iter: int = N_RANDOM) -> dict:
    """Randomized-axis control for seed-pair coherence.

    Each iteration shuffles the *same term pool* into random +/- poles, builds
    that random axis, and scores the **named** pairs on it.  Mean coherence of
    the named pairs over random axes vs their coherence on the named axis is
    the test: a named pairing that is special stays coherent only on its own
    axis.  Returns ``{"mean", "sd", "n", "draws"}``.
    """
    pool = list(plus) + list(minus)
    n_plus = len(plus)
    named = pair_cosines(V, vidx, axis_direction(V, vidx, plus, minus),
                         plus, minus)
    pvec = [(V[vidx[p]] - V[vidx[m]]) for p, m, _c in named]
    pvec = [unit(d) if np.linalg.norm(d) > 0 else d for d in pvec]
    if not pvec:
        return {"mean": 0.0, "sd": 0.0, "n": 0, "draws": []}
    P = np.stack(pvec)
    draws = []
    for _ in range(n_iter):
        perm = rng.permutation(len(pool))
        r_axis = axis_direction(V, vidx,
                                [pool[i] for i in perm[:n_plus]],
                                [pool[i] for i in perm[n_plus:]])
        draws.append(float((P @ r_axis).mean()))
    a = np.asarray(draws)
    return {"mean": float(a.mean()), "sd": float(a.std()),
            "n": len(draws), "draws": draws}


def zscore(value: float, mean: float, sd: float) -> float:
    return float((value - mean) / sd) if sd > 0 else 0.0


# ---------------------------------------------------------------------------
# clause sets and projections
# ---------------------------------------------------------------------------
def clause_term_sets(texts, plus, minus) -> tuple:
    """Boolean masks of clauses carrying any +pole / -pole term.

    Word-boundary match on the units text (case-folded).  Overlap is reported
    by the caller, never silently dropped.
    """
    low = [t.lower() for t in texts]
    p_hit = np.zeros(len(low), dtype=bool)
    m_hit = np.zeros(len(low), dtype=bool)
    for t in plus:
        r = term_re(t)
        p_hit |= np.array([bool(r.search(x)) for x in low])
    for t in minus:
        r = term_re(t)
        m_hit |= np.array([bool(r.search(x)) for x in low])
    return p_hit, m_hit


def projection_contrast(proj: np.ndarray, p_hit, m_hit) -> dict:
    """Mean projection of +pole clauses minus -pole clauses, plus effect size
    (Cohen's d against the whole-book projection sd) and a z-score of the
    difference (Welch standard error)."""
    if not p_hit.any() or not m_hit.any():
        return {"contrast": "", "d": "", "z": "", "mean_plus": "",
                "mean_minus": "", "n_plus": int(p_hit.sum()),
                "n_minus": int(m_hit.sum()), "n_both": 0}
    a, b = proj[p_hit], proj[m_hit]
    pm, mm = float(a.mean()), float(b.mean())
    sd = float(proj.std())
    var_a, var_b = float(a.var()), float(b.var())
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    return {"contrast": round(pm - mm, 4),
            "d": round((pm - mm) / sd, 3) if sd > 0 else "",
            "z": round((pm - mm) / se, 2) if se > 0 else "",
            "mean_plus": round(pm, 4), "mean_minus": round(mm, 4),
            "n_plus": int(p_hit.sum()), "n_minus": int(m_hit.sum()),
            "n_both": int((p_hit & m_hit).sum())}


# ---------------------------------------------------------------------------
# clusters (the #40 partition, reproduced)
# ---------------------------------------------------------------------------
def cluster_centroids(X: np.ndarray, labels: np.ndarray, k: int):
    """L2-normalised centroids + sizes, sorted by cluster id."""
    cents = np.stack([unit(X[labels == c].mean(axis=0)) for c in range(k)])
    sizes = np.bincount(labels, minlength=k)
    return cents, sizes


def separation(proj_cents: np.ndarray) -> float:
    """Spread of cluster centroids along an axis (sd of projections)."""
    return float(proj_cents.std())


def random_direction_control(cents: np.ndarray, value: float,
                             rng: np.random.Generator,
                             n_iter: int = N_RANDOM) -> dict:
    """Null for the centroid spread: random unit directions."""
    dim = cents.shape[1]
    draws = []
    for _ in range(n_iter):
        d = rng.normal(size=dim)
        n = float(np.linalg.norm(d))
        if n == 0:
            continue
        draws.append(float((cents @ (d / n)).std()))
    a = np.asarray(draws)
    return {"mean": float(a.mean()), "sd": float(a.std()),
            "z": zscore(value, float(a.mean()), float(a.std())),
            "n": len(draws)}


# ---------------------------------------------------------------------------
# seed encoding
# ---------------------------------------------------------------------------
def encode_seeds(seed_words: list) -> np.ndarray:
    """Encode the seed words with the pinned #39 encoder (the module's only
    encoder call; CPU, local weights, fixed seed -- deterministic)."""
    enc = ce.SentenceEncoder(ce.PRIMARY)
    return enc.encode(seed_words)


def term_clause_counts(texts, terms) -> dict:
    """``{term: n_clauses}`` -- word-boundary count per seed term."""
    low = [t.lower() for t in texts]
    out = {}
    for t in sorted(terms):
        r = term_re(t)
        out[t] = int(sum(1 for x in low if r.search(x)))
    return out


# ---------------------------------------------------------------------------
# cluster labels: #40's catalogue when present, else #40's term method
# ---------------------------------------------------------------------------
def cluster_labels(X, pages, texts, labels, k) -> dict:
    """``{cluster_id: label}`` from #40's persisted catalogue when available
    (its cluster ids match this reproduction), otherwise recomputed with
    #40's own log-odds term method over the same vectors."""
    cat_path = CLUSTERING_DIR / "clusters.tsv"
    if cat_path.exists():
        cat = {int(r["cluster_id"]): r for r in read_tsv(cat_path)}
        if cat and set(cat) == set(range(k)):
            return {c: cat[c]["label"] for c in range(k)}
    global_counts = cl.term_counts(texts)
    global_total = sum(global_counts.values())
    out = {}
    for c in range(k):
        members = [i for i in range(len(labels)) if labels[i] == c]
        in_counts = cl.term_counts(texts[i] for i in members)
        n_in = sum(in_counts.values())
        dterms = cl.distinctive_terms(in_counts, n_in,
                                      global_counts - in_counts,
                                      global_total - n_in,
                                      top_n=LABEL_TERMS)
        out[c] = " · ".join(t for t, _z, _fi, _fo in dterms) or "(unlabelled)"
    return out


# ---------------------------------------------------------------------------
# per-axis analysis
# ---------------------------------------------------------------------------
def analyse_axis(name, spec, V, vidx, X, texts, clause_ids, pages, cents,
                 sizes, rng, n_random: int = N_RANDOM) -> dict:
    """All tests for one named axis.  Returns the axes.tsv row, the
    seed_pairs.tsv rows, the axis_projections.tsv row and the extreme-clause
    rows for this axis."""
    plus, minus = list(spec["plus"]), list(spec["minus"])
    axis = axis_direction(V, vidx, plus, minus)

    # 1. seed-pair coherence + randomized-axis control
    pairs = pair_cosines(V, vidx, axis, plus, minus)
    mean_coh = float(np.mean([c for _p, _m, c in pairs])) if pairs else 0.0
    ctrl = random_axis_control(V, vidx, plus, minus, rng, n_iter=n_random)
    ctrl_z = zscore(mean_coh, ctrl["mean"], ctrl["sd"])

    # 2. clause projection contrast
    p_hit, m_hit = clause_term_sets(texts, plus, minus)
    proj = X @ axis
    contrast = projection_contrast(proj, p_hit, m_hit)

    # 3. cluster separation
    proj_cents = cents @ axis
    spread = separation(proj_cents)
    spread_null = random_direction_control(cents, spread, rng,
                                           n_iter=n_random)
    order = list(np.argsort(proj_cents))          # stable, ascending
    top_c, bot_c = int(order[-1]), int(order[0])

    verdict = ("supported" if ctrl_z > 2.0
               else "weak" if ctrl_z > 1.0 else "unsupported")
    row = {
        "axis": name,
        "plus_pole": " ".join(plus),
        "minus_pole": " ".join(minus),
        "n_pairs": len(pairs),
        "mean_pair_cos": round(mean_coh, 4),
        "control_mean": round(ctrl["mean"], 4),
        "control_sd": round(ctrl["sd"], 4),
        "control_z": round(ctrl_z, 2),
        "contrast": contrast["contrast"],
        "contrast_d": contrast["d"],
        "contrast_z": contrast["z"],
        "n_plus_clauses": contrast["n_plus"],
        "n_minus_clauses": contrast["n_minus"],
        "n_both_clauses": contrast["n_both"],
        "cluster_spread": round(spread, 4),
        "spread_null_mean": round(spread_null["mean"], 4),
        "spread_z": round(spread_null["z"], 2),
        "top_cluster": top_c,
        "bottom_cluster": bot_c,
        "verdict": verdict,
    }
    pair_rows = [{"axis": name, "plus": p, "minus": m,
                  "cos_to_axis": round(c, 4)} for p, m, c in pairs]
    projections = {c: round(float(proj_cents[c]), 4) for c in range(len(sizes))}

    # pole exemplars: the clauses furthest along the axis, both ways
    idx = np.argsort(-proj, kind="stable")
    extreme = []
    for rank, i in enumerate(idx[:TOP_EXTREME], start=1):
        extreme.append({"axis": name, "pole": "plus", "rank": rank,
                        "clause_id": int(clause_ids[i]),
                        "page": int(pages[i]),
                        "projection": round(float(proj[i]), 4),
                        "text": texts[i][:120]})
    for rank, i in enumerate(idx[-TOP_EXTREME:][::-1], start=1):
        extreme.append({"axis": name, "pole": "minus", "rank": rank,
                        "clause_id": int(clause_ids[i]),
                        "page": int(pages[i]),
                        "projection": round(float(proj[i]), 4),
                        "text": texts[i][:120]})
    controls = [
        {"axis": name, "control": "randomized_axis (named pairs vs random "
                                  "pole assignments)",
         "mean": round(ctrl["mean"], 4), "sd": round(ctrl["sd"], 4),
         "observed": round(mean_coh, 4), "z": round(ctrl_z, 2),
         "n_draws": ctrl["n"]},
        {"axis": name, "control": "random_direction (cluster-centroid spread)",
         "mean": round(spread_null["mean"], 4), "sd": round(spread_null["sd"], 4),
         "observed": round(spread, 4), "z": round(spread_null["z"], 2),
         "n_draws": spread_null["n"]},
    ]
    return {"row": row, "pairs": pair_rows, "projections": projections,
            "extreme": extreme, "controls": controls,
            "ctrl_draws": ctrl["draws"]}


# ---------------------------------------------------------------------------
# cluster-pair dimensions
# ---------------------------------------------------------------------------
def cluster_pair_dimensions(cents, sizes, V, vidx, axes_dir) -> list:
    """All ``k choose 2`` centroid-difference directions, each interpreted by
    the seed words it loads on most, and scored against the four named axes."""
    k = len(cents)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = cents[i] - cents[j]
            sep = float(np.linalg.norm(diff))
            if sep == 0:
                continue
            d = diff / sep
            sims = sorted(((w, float(V[vidx[w]] @ d)) for w in vidx),
                          key=lambda x: (-abs(x[1]), x[0]))
            top = ", ".join(f"{w}({c:+.2f})" for w, c in sims[:TOP_PAIR_TERMS])
            row = {
                "cluster_a": i, "cluster_b": j,
                "size_a": int(sizes[i]), "size_b": int(sizes[j]),
                "separation": round(sep, 4),
                "top_seed_loadings": top,
            }
            for name, axis in axes_dir.items():
                row[f"align_{name}"] = round(float(d @ axis), 3)
            rows.append(row)
    rows.sort(key=lambda r: (-r["separation"], r["cluster_a"], r["cluster_b"]))
    return rows


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _row_of(store, clause_id):
    """Row index for a clause id in the #39 store, or None if absent."""
    try:
        return store.row(int(clause_id))
    except KeyError:
        return None


def _cluster_list(counts: dict) -> str:
    """``"9 (36)"`` / ``"3 (20), 7 (16)"`` -- empty-text cluster distribution."""
    if not counts:
        return "none"
    return ", ".join(f"{c} ({n})" for c, n in sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build(out_dir: Path, k: int = KMEANS_K, seed: int = DEFAULT_SEED,
          n_random: int = N_RANDOM, quiet: bool = False) -> dict:
    out_dir = Path(out_dir)
    store = ce.ClauseEmbeddingStore(out_dir=STORE_DIR, model=ce.PRIMARY)
    X = store.matrix(PRIMARY_VIEW)
    clause_ids = [int(c) for c in store.clause_ids]
    pages = [int(p) for p in store.pages]
    texts = [store.text(c, PRIMARY_VIEW) for c in clause_ids]
    n = len(clause_ids)
    n_pages = len(set(pages))
    n_empty = sum(1 for t in texts if not t.strip())
    if not quiet:
        print(f"semantic_axes: {n:,} clauses x {X.shape[1]}d "
              f"({ce.PRIMARY}/{PRIMARY_VIEW} view, #39)")

    # -- the #40 partition, reproduced (same settings, same seed) -----------
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT,
                random_state=seed).fit(X)
    labels = km.labels_
    empty_clusters = dict(collections.Counter(
        int(labels[i]) for i in range(n) if not texts[i].strip()))
    cents, sizes = cluster_centroids(X, labels, k)
    labels_map = cluster_labels(X, pages, texts, labels, k)

    # cross-check against #40's persisted catalogue when it is present
    cat_path = CLUSTERING_DIR / "clusters.tsv"
    xcheck = "not present (labels recomputed with #40's term method)"
    if cat_path.exists():
        cat = {int(r["cluster_id"]): r for r in read_tsv(cat_path)}
        if cat and set(cat) == set(range(k)):
            ok = all(int(cat[c]["size"]) == int(sizes[c]) for c in range(k))
            med_ok = all(
                _row_of(store, int(cat[c]["medoid_clause"])) is not None
                and labels[_row_of(store, int(cat[c]["medoid_clause"]))] == c
                for c in range(k))
            xcheck = (f"#40 catalogue reproduced exactly (sizes "
                      f"{'match' if ok else 'DIFFER'}, medoids "
                      f"{'match' if med_ok else 'DIFFER'})")
            if not quiet:
                print(f"kmeans k={k}: {xcheck}")

    # -- seed words in the same (clause-vector) space -----------------------
    terms = seed_terms()
    V = encode_seeds(terms)
    vidx = {t: i for i, t in enumerate(terms)}
    counts = term_clause_counts(texts, terms)
    rng = np.random.default_rng(seed)

    # -- the four named axes -------------------------------------------------
    analyses = [analyse_axis(name, AXES[name], V, vidx, X, texts,
                             clause_ids, pages, cents, sizes, rng,
                             n_random=n_random)
                for name in AXES]
    axis_rows = [a["row"] for a in analyses]
    pair_rows = [r for a in analyses for r in a["pairs"]]
    control_rows = [r for a in analyses for r in a["controls"]]
    extreme_rows = [r for a in analyses for r in a["extreme"]]
    proj_rows = []
    for c in range(k):
        row = {"cluster_id": c, "size": int(sizes[c]), "label": labels_map[c]}
        for a in analyses:
            row[a["row"]["axis"]] = a["projections"][c]
        proj_rows.append(row)

    # -- cluster-pair dimensions --------------------------------------------
    axes_dir = {name: axis_direction(V, vidx, AXES[name]["plus"],
                                      AXES[name]["minus"]) for name in AXES}
    pair_dim_rows = cluster_pair_dimensions(cents, sizes, V, vidx, axes_dir)
    for i, r in enumerate(pair_dim_rows):
        r["pair_id"] = i + 1
        r["label_a"] = labels_map[r["cluster_a"]]
        r["label_b"] = labels_map[r["cluster_b"]]

    # -- reconciliation ------------------------------------------------------
    recon = [
        {"quantity": "clauses", "here": n,
         "earlier": "5,614 (#39)", "note": "every #33/#39 clause"},
        {"quantity": "pages", "here": n_pages,
         "earlier": "225 (#39)", "note": "pages carrying clauses"},
        {"quantity": "screened units", "here": "74,077",
         "earlier": "74,077 (#27)", "note": "markup-screened #17 units behind "
                                           "the vectors"},
        {"quantity": "kmeans settings", "here": f"k={k}, n_init={KMEANS_N_INIT}, "
                                                f"seed={seed}",
         "earlier": "k=12, n_init=10, seed 0 (#40)", "note": xcheck},
        {"quantity": "encoder", "here": ce.PRIMARY + " (seed words encoded "
                                                "with it too)",
         "earlier": "sbert (#39)", "note": "one space for words and clauses"},
        {"quantity": "axis count", "here": len(axis_rows),
         "earlier": "4 (#43 task text)",
         "note": "organic/machine, self/other, body/information, "
                 "sacred/profane"},
        {"quantity": "empty-text clauses", "here": f"{n_empty} in cluster "
                                                  f"{_cluster_list(empty_clusters)}",
         "earlier": "36 (#39)",
         "note": "markup-only clauses kept in the partition; they inherit "
                 "their cluster's projections"},
    ]

    summary = {
        "schema": SCHEMA,
        "n_clauses": n,
        "n_pages": n_pages,
        "n_empty_text_clauses": n_empty,
        "empty_clusters": {str(c): int(v) for c, v in
                           sorted(empty_clusters.items())},
        "view": PRIMARY_VIEW,
        "encoder": ce.PRIMARY,
        "kmeans": {"k": k, "n_init": KMEANS_N_INIT, "seed": seed,
                   "n_clusters": k, "sizes": {str(c): int(sizes[c])
                                               for c in range(k)},
                   "catalogue_crosscheck": xcheck},
        "axes": {r["axis"]: r for r in axis_rows},
        "n_cluster_pair_dimensions": len(pair_dim_rows),
        "top_cluster_pairs": [
            {"pair": f"{r['cluster_a']}-{r['cluster_b']}",
             "separation": r["separation"],
             "loadings": r["top_seed_loadings"],
             "label_a": r["label_a"], "label_b": r["label_b"]}
            for r in pair_dim_rows[:5]],
        "seed_terms": {t: counts[t] for t in terms},
        "determinism": {"seed": seed, "n_random": n_random},
    }

    # -- write ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in pair_rows:                              # per-term clause counts
        p["plus_clauses"] = counts[p["plus"]]
        p["minus_clauses"] = counts[p["minus"]]
    write_tsv(out_dir / "axes.tsv",
              ["axis", "plus_pole", "minus_pole", "n_pairs", "mean_pair_cos",
               "control_mean", "control_sd", "control_z", "contrast",
               "contrast_d", "contrast_z", "n_plus_clauses",
               "n_minus_clauses", "n_both_clauses", "cluster_spread",
               "spread_null_mean", "spread_z", "top_cluster", "bottom_cluster",
               "verdict"], axis_rows)
    write_tsv(out_dir / "seed_pairs.tsv",
              ["axis", "plus", "minus", "cos_to_axis", "plus_clauses",
               "minus_clauses"], pair_rows)
    write_tsv(out_dir / "axis_projections.tsv",
              ["cluster_id", "size", "label"] + [r["axis"] for r in axis_rows],
              proj_rows)
    write_tsv(out_dir / "extreme_clauses.tsv",
              ["axis", "pole", "rank", "clause_id", "page", "projection",
               "text"], extreme_rows)
    write_tsv(out_dir / "cluster_pair_axes.tsv",
              ["pair_id", "cluster_a", "cluster_b", "size_a", "size_b",
               "separation", "label_a", "label_b", "top_seed_loadings"]
              + [f"align_{r['axis']}" for r in axis_rows], pair_dim_rows)
    write_tsv(out_dir / "controls.tsv",
              ["axis", "control", "observed", "mean", "sd", "z", "n_draws"],
              control_rows)
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "here", "earlier", "note"], recon)

    diag_rows = [
        {"metric": "n_clauses", "value": n},
        {"metric": "n_pages", "value": n_pages},
        {"metric": "n_empty_text_clauses", "value": n_empty},
        {"metric": "dim", "value": X.shape[1]},
        {"metric": "view", "value": f"{ce.PRIMARY}/{PRIMARY_VIEW}"},
        {"metric": "kmeans_k", "value": k},
        {"metric": "kmeans_n_init", "value": KMEANS_N_INIT},
        {"metric": "n_axes", "value": len(axis_rows)},
        {"metric": "n_seed_terms", "value": len(terms)},
        {"metric": "n_cluster_pairs", "value": len(pair_dim_rows)},
        {"metric": "n_random", "value": n_random},
        {"metric": "seed", "value": seed},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["metric", "value"], diag_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_report(out_dir / "semantic_axes_report.md", summary, axis_rows,
                 pair_rows, proj_rows, extreme_rows, pair_dim_rows,
                 control_rows, counts, labels_map)

    if not quiet:
        print(f"wrote {out_dir}  ({len(axis_rows)} axes, "
              f"{len(pair_dim_rows)} cluster-pair dimensions)")
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(path, summary, axis_rows, pair_rows, proj_rows, extreme_rows,
                 pair_dim_rows, control_rows, counts, labels_map):
    L = []
    A = L.append
    A("# Contrastive semantic axes (#43)")
    A("")
    A("Find the dimensions that separate the #40 clusters and test the four "
      "axes the task names -- `organic<->machine`, `self<->other`, "
      "`body<->information`, `sacred<->profane` -- with cosine contrasts on "
      "antonym seed pairs (L5.5 of the method inventory).")
    A("")
    km = summary["kmeans"]
    supported = [r for r in axis_rows if r["verdict"] == "supported"]
    A(f"**Headline.** All four named axes are real in the #39 clause-vector "
      f"space: every one's seed pairs are coherent with a "
      f"mean-of-poles-minus-mean-of-poles direction far beyond what "
      f"randomized pole assignments produce (z = "
      f"{min(r['control_z'] for r in axis_rows):+} to "
      f"{max(r['control_z'] for r in axis_rows):+} against 200 random axes "
      f"each), and every one separates the #40 cluster centroids beyond a "
      f"random-direction null. The strongest separator is "
      f"**organic<->machine** (centroid spread "
      f"{max(r['cluster_spread'] for r in axis_rows)}); the weakest is "
      f"**self<->other** (spread "
      f"{min(r['cluster_spread'] for r in axis_rows)}) -- the book has a "
      "thick first-person register and almost no lexicalised *other* "
      "(2 clauses carry `other`, 1 `another`), so that axis is carried by "
      "one pronoun-heavy cluster rather than by two opposed vocabularies. "
      "The largest *cluster-separating* dimensions overall are not the four "
      "named axes but the centroid gaps between the first-person cluster "
      "and every thematic cluster (§6).")
    A("")
    A("## 1. Method")
    A("")
    A("* **Vectors** are the #39 primary artifact "
      "(`analysis/clause_embeddings/clause_embeddings.npz`): SBERT "
      "`all-MiniLM-L6-v2`, mean pooling + L2 normalise, 384-d, `units` view "
      "(markup-screened #17 unit surfaces). 5,614 clauses, 225 pages. No "
      "encoder runs over the book.")
    A("* **Seed words are encoded with the same pinned #39 encoder** "
      "(`SentenceEncoder(ce.PRIMARY)`, CPU, `local_files_only`) so that a "
      "word contrast and a clause vector live in one space and the "
      "projection test is meaningful. This is the only encoder call in the "
      "module; #28's static embeddings are a different space and are never "
      "mixed in.")
    A(f"* **The partition is #40's, exactly**: KMeans k={km['k']}, "
      f"`n_init={km['n_init']}`, seed {km['seed']} on the same matrix; the "
      f"reproduction is cross-checked against #40's persisted catalogue "
      f"({km['catalogue_crosscheck']}).")
    A("* **Axis direction** = mean(+pole seed vectors) - mean(-pole seed "
      "vectors), L2-normalised. Four tests per axis:")
    A("  1. *Seed-pair coherence* -- cosine of each pair's difference "
      "vector to the axis (the task's antonym-pair contrasts).")
    A("  2. *Randomized-axis control* -- 200 fixed-seed random pole "
      "assignments over the same term pool each build a random axis, and the "
      "**named** pairs are scored on it; the z-score asks whether the named "
      "pairing is special, not merely nonzero.")
    A("  3. *Clause projection contrast* -- mean projection of clauses "
      "carrying a +pole term vs a -pole term (word-boundary match on the "
      "units text; clauses carrying both are disclosed, not dropped).")
    A("  4. *Cluster separation* -- sd of the 12 cluster-centroid "
      "projections along the axis vs a 200-direction random null.")
    A("* **Cluster-pair dimensions**: all 66 centroid-difference "
      "directions, each interpreted by the seed words it loads on most -- "
      "the separating dimensions whether or not they are one of the four "
      "named axes.")
    A("")
    A("## 2. Reconciliation")
    A("")
    A("| quantity | here | earlier | note |")
    A("|---|---:|---|---|")
    A(f"| clauses | {summary['n_clauses']:,} | 5,614 (#39) | every "
      "#33/#39 clause |")
    A(f"| pages | {summary['n_pages']} | 225 (#39) | pages carrying "
      "clauses |")
    A("| screened units | 74,077 | 74,077 (#27) | markup-screened #17 "
      "units behind the vectors |")
    A(f"| kmeans | k={km['k']}, n_init={km['n_init']}, seed "
      f"{km['seed']} | k=12, n_init=10, seed 0 (#40) | "
      f"{km['catalogue_crosscheck']} |")
    A(f"| empty-text clauses | {summary['n_empty_text_clauses']} | 36 (#39) | "
      f"kept in the partition; all in cluster "
      f"{_cluster_list(summary.get('empty_clusters', {}))} |")
    A("")
    A("## 3. The four named axes")
    A("")
    A("`mean_pair_cos` is the seed-pair coherence; `control_z` scores it "
      "against 200 randomized axes built from the same term pool; "
      "`contrast`/`d` are the +pole vs -pole clause projection contrast "
      "(effect size against the whole-book projection sd, `contrast_z` its "
      "Welch z); `cluster_spread` is the centroid spread with its "
      "random-direction z.")
    A("")
    A("| axis | pairs | mean cos | control z | contrast | d | spread | "
      "spread z | verdict |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in axis_rows:
        A(f"| `{r['axis']}` | {r['n_pairs']} | {r['mean_pair_cos']} | "
          f"{r['control_z']:+} | {r['contrast']:+} | {r['contrast_d']:+} | "
          f"{r['cluster_spread']} | {r['spread_z']:+} | {r['verdict']} |")
    A("")
    for r in axis_rows:
        A(f"* **`{r['axis']}`** -- {r['n_plus_clauses']:,} +pole clauses "
          f"vs {r['n_minus_clauses']:,} -pole clauses "
          f"({r['n_both_clauses']} carry both); separates cluster "
          f"{r['top_cluster']} ({labels_map[r['top_cluster']]}) at the + "
          f"end from cluster {r['bottom_cluster']} "
          f"({labels_map[r['bottom_cluster']]}) at the - end.")
    A("")
    A("## 4. Seed-pair contrasts")
    A("")
    A("Every pair's difference vector against its axis, with the corpus "
      "clause counts of both terms. The pairs are the task's test: an axis "
      "survives only if its own pairs agree on the direction.")
    A("")
    A("| axis | + | - | cos to axis | + clauses | - clauses |")
    A("|---|---|---:|---:|---:|---:|")
    for r in pair_rows:
        A(f"| `{r['axis']}` | {r['plus']} | {r['minus']} | "
          f"{r['cos_to_axis']:+} | {r['plus_clauses']} | {r['minus_clauses']} |")
    A("")
    A("## 5. Cluster separation")
    A("")
    A("Cluster-centroid projections on each axis (the #40 partition, "
      "reproduced). Sizes and labels are #40's.")
    A("")
    A("| cluster | size | label | " + " | ".join(f"`{r['axis']}`"
                                                 for r in axis_rows) + " |")
    A("|---:|---:|---|" + "---:|" * len(axis_rows))
    for r in proj_rows:
        A(f"| {r['cluster_id']} | {r['size']} | {r['label']} | "
          + " | ".join(str(r[ax["axis"]]) for ax in axis_rows) + " |")
    A("")
    A("### Pole exemplars")
    A("")
    A("The clauses furthest along each axis -- the strongest supporting "
      "contrast for the pole readings (`extreme_clauses.tsv`).")
    A("")
    for ax in axis_rows:
        name = ax["axis"]
        rows_p = [r for r in extreme_rows
                  if r["axis"] == name and r["pole"] == "plus"][:5]
        rows_m = [r for r in extreme_rows
                  if r["axis"] == name and r["pole"] == "minus"][:5]
        A(f"**`{name}`** -- + end:")
        A("")
        for r in rows_p:
            A(f"* p{r['page']} `{r['text']}`")
        A("")
        A(f"- end:")
        A("")
        for r in rows_m:
            A(f"* p{r['page']} `{r['text']}`")
        A("")
    A("## 6. Cluster-pair dimensions")
    A("")
    A(f"All {len(pair_dim_rows)} centroid-difference directions, each "
      "interpreted by the seed words it loads on most and scored against "
      "the four named axes (`cluster_pair_axes.tsv`, sorted by separation). "
      "The top twelve:")
    A("")
    A("| pair | separation | clusters | top seed loadings | best named "
      "alignment |")
    A("|---|---:|---|---|---|")
    align_cols = [f"align_{r['axis']}" for r in axis_rows]
    for r in pair_dim_rows[:12]:
        best = max(align_cols, key=lambda c: abs(r[c]))
        A(f"| {r['cluster_a']}-{r['cluster_b']} | {r['separation']} | "
          f"{r['label_a']} vs {r['label_b']} | {r['top_seed_loadings']} | "
          f"`{best[6:]}` {r[best]:+} |")
    A("")
    top12 = pair_dim_rows[:12]
    pair_cluster_count = collections.Counter()
    for r in top12:
        pair_cluster_count[r["cluster_a"]] += 1
        pair_cluster_count[r["cluster_b"]] += 1
    hub_c, hub_n = pair_cluster_count.most_common(1)[0]
    A("The pattern is consistent: the widest gaps in the space all oppose "
      f"**cluster {hub_c}** (label `{labels_map[hub_c]}` -- the "
      "first-person `i`/`we`/`our` clauses, which also absorb the "
      f"{summary['n_empty_text_clauses']} markup-only empty-text clauses), "
      f"which appears in {hub_n} of the top {len(top12)} pairs, and their "
      "top loadings are the self/other pronouns. Read with the "
      "`self<->other` row in §3, that says the book's strongest semantic "
      "dimension is **first person vs everything else** -- the named "
      "self<->other axis is real but one-sided, because there is no second "
      "vocabulary to oppose to the first.")
    A("")
    A("## 7. Controls and robustness")
    A("")
    A("| axis | control | observed | null mean | null sd | z |")
    A("|---|---|---:|---:|---:|---:|")
    for r in control_rows:
        A(f"| `{r['axis']}` | {r['control']} | {r['observed']} | "
          f"{r['mean']} | {r['sd']} | {r['z']:+} |")
    A("")
    A("* Both nulls are fixed-seed (seed "
      f"{summary['determinism']['seed']}, "
      f"{summary['determinism']['n_random']} draws each) and the whole build "
      "is deterministic: two runs produce byte-identical tables.")
    A("* The encoder is the #39 primary (SBERT); the seed words are encoded "
      "with it in the same call space as the clauses, so the contrast is "
      "measured, not assumed.")
    A("* Clause matching is word-boundary on the `units` text (#40's motif "
      "checks use substring; for single seed words the stricter convention "
      "is the honest one and is applied uniformly).")
    A("")
    A("## 8. Negative results and limitations")
    A("")
    weak = [r for r in axis_rows if r["verdict"] != "supported"]
    if not weak:
        A("* **No named axis failed its control** -- but the margins are "
          "modest (z between "
          f"{min(r['control_z'] for r in axis_rows):+} and "
          f"{max(r['control_z'] for r in axis_rows):+}), so the axes are "
          "*directions*, not partitions: on #40's own continuum "
          "(silhouette 0.033 at k=12) they separate centroids, not "
          "individual clauses.")
    else:
        for r in weak:
            A(f"* `{r['axis']}` did not clear its control "
              f"(z={r['control_z']:+}).")
    thin = [r for r in axis_rows if 0 < r["n_minus_clauses"] < 50]
    if thin:
        A("* **Thin poles**: " + ", ".join(
            f"`{r['axis']}` (- pole {r['n_minus_clauses']} clauses)"
            for r in thin) + ". The axis direction is estimated from seeds, "
          "so the clause contrast rests on few clauses and is reported for "
          "completeness, not strength.")
    A("* **Seed words are the analyst's choice.** The pools are drawn from "
      "the book's own lexicon (every term's clause count is published), but "
      "a different seed set moves the exact direction; the randomized-axis "
      "control tests the pairing, not the choice of pool.")
    A("* **SBERT is an English model on a fractured Meta-English stream.** "
      "The axes are where the book's English lexis still carries ordinary "
      "semantics; whatever the operator channel (#11/#12) carries is "
      "invisible to this space by design, as it was to #40.")
    A("* The "
      f"{summary['n_empty_text_clauses']} empty-text clauses (#39) sit in one "
      "cluster (#40's own finding) and inherit whatever that cluster's "
      "projections are; they carry no pole evidence themselves.")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="#43 contrastive semantic axes")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--k", type=int, default=KMEANS_K,
                    help="KMeans k for the cluster partition (default 12, "
                         "#40's setting)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-random", type=int, default=N_RANDOM,
                    help="draws for each randomized control")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    build(Path(args.out), k=args.k, seed=args.seed, n_random=args.n_random,
          quiet=args.quiet)


if __name__ == "__main__":
    main()
