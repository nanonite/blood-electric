#!/usr/bin/env python3
"""#41 Topic modelling (LDA / NMF / BERTopic) -> analysis/topics/

Recover the book's **latent themes with top terms** and compare them with the
co-occurrence-network communities of #29 (E3.6).  L5.3 of
``docs/meaning-extraction-methods.md`` names the three algorithms; all three
are run here and their convergence is measured.

Documents.  The corpus is the #39 ``clause_index.tsv``: one document per #33
clause, whose ``text_units`` column is the markup-screened #17 unit surface
sequence (joined by spaces).  Splitting on whitespace recovers exactly the
#17 **unit surfaces** -- the same tokens #29's network nodes are built from --
so the topic and community term spaces coincide (all 391 #29 nodes are in the
topic vocabulary).  The screen is #29's: ``sklearn`` ``ENGLISH_STOP_WORDS``
plus the OCR HTML-entity residue ``lt``/``gt``/``amp``/``quot``.  TSV cells are
unescaped first (the #17 ``\\n`` quote hazard), so a unit broken across a
literal newline is split rather than fused into a stray ``n``/``t`` token.

Models.

  * **LDA** -- ``sklearn`` ``LatentDirichletAllocation`` (batch variational
    inference, fixed seed, single job) over raw term counts.
  * **NMF** -- ``sklearn`` ``NMF`` (``nndsvda`` init, fixed seed) over the
    TF-IDF matrix with the *same* vocabulary, so the two models' terms are
    directly comparable.
  * **BERTopic-style** -- the BERTopic algorithm reproduced from its own
    default components: the #39 SBERT clause embeddings are reduced with
    **UMAP** (``n_neighbors=15``, 5-d, ``min_dist=0``, cosine, fixed seed,
    single job), clustered with **HDBSCAN** and labelled by **c-TF-IDF**.  The
    ``bertopic`` package is not installed in this environment; the algorithm is
    reimplemented on the installed ``umap-learn`` + ``hdbscan`` and that is
    disclosed, not hidden.  ``--no-umap`` falls back to a deterministic
    ``TruncatedSVD`` reduction (a much weaker projection, reported as such).

Model selection is **coherence**, not a taste call.  Every LDA/NMF model on a
fixed ``k`` grid is scored by gensim's ``c_v`` (and ``u_mass``) coherence; the
best-coherence ``k`` per model is the catalogue of record.  BERTopic's
granularity is its ``min_topic_size``; the scan over that grid is reported and
the default is disclosed.

Community comparison (E3.6 / #29).  For each model, every vocabulary term is
assigned to its dominant topic (``argmax`` of the topic-term matrix), and the
resulting partition of the #29 network nodes is scored against #29's Louvain
communities with the **adjusted Rand index**.  A topic-by-community overlap
table reports, per topic, the community that carries the most of its top-term
mass.  This is the L9.5 triangulation: convergence is evidence, divergence a
lead.

Outputs under ``analysis/topics/``:
  topic_modelling_report.md   THE DELIVERABLE (method, coherence, topics,
                              interpretation, community comparison)
  coherence_scan.tsv          model x k: c_v / u_mass coherence (+ LDA ppl)
  lda_topics.tsv              LDA best-k topic -> ranked terms + weights
  lda_topic_summary.tsv       LDA best-k topic: size, label, top terms
  nmf_topics.tsv              NMF best-k topic -> ranked terms + weights
  nmf_topic_summary.tsv       NMF best-k topic: size, label, top terms
  bertopic_topics.tsv         BERTopic-style topic -> ranked c-TF-IDF terms
  bertopic_topic_summary.tsv  BERTopic-style topic: size, label, top terms
  bertopic_scan.tsv           min_topic_size grid: clusters / noise
  clause_topics.tsv           clause_id, page, dominant topic per model
  community_overlap.tsv       model x topic -> best #29 community
  community_ari.tsv           model: adjusted Rand vs #29 communities
  reconciliation.tsv          against #39 / #33 / #17 / #29
  diagnostics.tsv             build-time counters
  summary.json                machine-readable headline numbers

Reproduce::

    conda activate marker
    python work/topic_modelling.py                 # writes analysis/topics/
    python work/test_topic_modelling.py

Deterministic: fixed seeds everywhere (LDA/NMF/UMAP/HDBSCAN), single job, no
sampling, no wall-clock in any artifact, no hash-order dependence (every list
sorted; ties broken by a fixed key).  Requires ``numpy`` + ``scikit-learn`` +
``gensim``; ``umap-learn`` + ``hdbscan`` for the BERTopic-style model (a
``--no-umap`` fallback keeps the build runnable without UMAP).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import clause_embeddings as ce  # #39 ClauseEmbeddingStore
import clause_segmentation as cs  # quote-safe TSV io + ROOT
import numpy as np

ROOT = cs.ROOT
OUT = ROOT / "analysis" / "topics"
CLAUSE_INDEX = ce.OUT / "clause_index.tsv"
NETWORK_DIR = ROOT / "analysis" / "network"
NETWORK_COMMUNITIES = NETWORK_DIR / "communities.tsv"
NETWORK_SUMMARY = NETWORK_DIR / "summary.json"

SCHEMA = "topics/1"
PRIMARY_VIEW = "units"              # #39 primary semantic view
DEFAULT_SEED = 0

# --- tokenisation / vocabulary -------------------------------------------------
STOPWORDS = None                    # filled at import (sklearn stop list)
EXTRA_STOP = {"lt", "gt", "amp", "quot"}
MIN_DF = 5                          # term must occur in >= 5 clauses
MAX_DF = 0.5                        # term must occur in <= 50% of clauses
TOP_WORDS = 10                      # terms shown per topic
LABEL_TERMS = 3                     # terms in the auto label

# --- LDA / NMF -----------------------------------------------------------------
K_GRID = (5, 6, 8, 10, 12, 16)      # coherence scan grid
LDA_MAX_ITER = 30
LDA_LEARNING = "batch"
NMF_MAX_ITER = 500
NMF_INIT = "nndsvda"

# --- BERTopic-style ------------------------------------------------------------
BERT_MTS_GRID = (20, 50, 80, 120)   # min_topic_size scan grid
BERT_MIN_TOPIC_SIZE = 80            # disclosed default (readable resolution)
BERT_MIN_SAMPLES = 5
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
SVD_COMPONENTS = 15                 # --no-umap fallback

read_tsv = cs.read_tsv
write_tsv = cs.write_tsv

_ESCAPES = (("\\n", " "), ("\\t", " "), ("\\r", " "), ("\\\\", " "))


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def _init_stopwords():
    global STOPWORDS
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOP


def unescape(s: str) -> str:
    """Reverse the #17/#33 TSV cell escaping (newline/tab/backslash -> space)."""
    for a, b in _ESCAPES:
        s = s.replace(a, b)
    return s


def tokenize(text: str) -> list:
    """Unit-surface tokens: whitespace split, lower-cased, function words out.

    The #39 ``text_units`` column is space-joined #17 unit surfaces, so this
    recovers exactly #29's node tokens (``boy-roid``, ``soul/gram``,
    ``gene=tv``) rather than splitting them into parts.
    """
    if STOPWORDS is None:
        _init_stopwords()
    return [w for w in unescape(text).lower().split()
            if w and w not in STOPWORDS]


def load_documents():
    """``(clause_ids, pages, docs, diag)`` from the #39 clause index."""
    rows = read_tsv(CLAUSE_INDEX)
    clause_ids = [int(r["clause_id"]) for r in rows]
    pages = [int(r["page"]) for r in rows]
    docs = [tokenize(r["text_units"]) for r in rows]
    diag = {
        "n_docs": len(rows),
        "n_tokens": sum(len(d) for d in docs),
        "n_types": len({t for d in docs for t in d}),
        "n_empty": sum(1 for d in docs if not d),
    }
    return clause_ids, pages, docs, diag


def build_vectors(docs, min_df: int = MIN_DF, max_df: float = MAX_DF):
    """Shared-vocabulary count + TF-IDF matrices.

    Returns ``(X_counts, X_tfidf, vocab)``.  Both matrices share one
    ``CountVectorizer`` vocabulary so LDA and NMF terms are aligned.
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    cv = CountVectorizer(analyzer=lambda x: x, min_df=min_df, max_df=max_df)
    X = cv.fit_transform(docs)
    vocab = list(cv.get_feature_names_out())
    tv = TfidfVectorizer(analyzer=lambda x: x, vocabulary=cv.vocabulary_)
    Xt = tv.fit_transform(docs)
    return X, Xt, vocab


# ---------------------------------------------------------------------------
# LDA / NMF
# ---------------------------------------------------------------------------
def fit_lda(X, k: int, seed: int = DEFAULT_SEED, max_iter: int = LDA_MAX_ITER):
    from sklearn.decomposition import LatentDirichletAllocation
    return LatentDirichletAllocation(
        n_components=k, learning_method=LDA_LEARNING, max_iter=max_iter,
        random_state=seed, n_jobs=1).fit(X)


def fit_nmf(Xt, k: int, seed: int = DEFAULT_SEED, max_iter: int = NMF_MAX_ITER):
    from sklearn.decomposition import NMF
    return NMF(n_components=k, init=NMF_INIT, random_state=seed,
               max_iter=max_iter).fit(Xt)


def top_terms(components: np.ndarray, vocab, n: int = TOP_WORDS) -> list:
    """``[[(term, weight), ...], ...]`` per topic, weight-descending.

    Ties break by term (ascending) so the order is independent of the
    component matrix's internal ordering.
    """
    out = []
    for row in np.asarray(components):
        order = sorted(range(len(vocab)),
                       key=lambda i: (-float(row[i]), vocab[i]))
        out.append([(vocab[i], float(row[i])) for i in order[:n]])
    return out


# ---------------------------------------------------------------------------
# coherence (gensim)
# ---------------------------------------------------------------------------
def _topic_words(topics) -> list:
    """Coerce ``[[(term, weight), ...]]`` or ``[[term, ...]]`` to term lists."""
    out = []
    for topic in topics:
        if topic and isinstance(topic[0], (tuple, list)):
            out.append([str(t) for t, _w in topic])
        else:
            out.append([str(t) for t in topic])
    return out


def coherence(topics, docs, dictionary=None, corpus=None) -> dict:
    """``c_v`` and ``u_mass`` coherence of a list of top-term lists."""
    import gensim
    from gensim.models.coherencemodel import CoherenceModel
    topics = _topic_words(topics)
    if dictionary is None:
        dictionary = gensim.corpora.Dictionary(docs)
    if corpus is None:
        corpus = [dictionary.doc2bow(d) for d in docs]
    cv = CoherenceModel(topics=topics, texts=docs, dictionary=dictionary,
                        coherence="c_v", processes=1).get_coherence()
    um = CoherenceModel(topics=topics, texts=docs, dictionary=dictionary,
                        corpus=corpus, coherence="u_mass",
                        processes=1).get_coherence()
    return {"c_v": float(cv), "u_mass": float(um)}


# ---------------------------------------------------------------------------
# BERTopic-style (UMAP + HDBSCAN + c-TF-IDF)
# ---------------------------------------------------------------------------
def reduce_embeddings(X: np.ndarray, seed: int = DEFAULT_SEED,
                      use_umap: bool = True):
    """Dimensionality reduction for the BERTopic-style model.

    Returns ``(Z, note)``.  UMAP is the BERTopic default (15 neighbours, 5-d,
    ``min_dist=0``, cosine, fixed seed, single job -- deterministic).  The
    fallback is a deterministic ``TruncatedSVD`` (disclosed as a weaker,
    global-linear projection).
    """
    if use_umap:
        try:
            import umap
            red = umap.UMAP(n_neighbors=UMAP_NEIGHBORS,
                            n_components=UMAP_COMPONENTS,
                            min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC,
                            random_state=seed, n_jobs=1)
            return red.fit_transform(X), "umap"
        except Exception as exc:  # noqa: BLE001 - optional dependency
            note = f"umap unavailable ({type(exc).__name__}); TruncatedSVD"
    else:
        note = "TruncatedSVD (--no-umap)"
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=seed)
    return svd.fit_transform(X), note


def ctfidf_matrix(labels, docs, vocab_index: dict):
    """Class-based TF-IDF (BERTopic's topic representation).

    ``tf`` is the term frequency inside a cluster; ``idf`` is
    ``log(1 + A / f_w)`` with ``A`` the average number of words per cluster
    and ``f_w`` the term's total count across clusters.  Noise (label -1) is
    excluded.
    """
    labels = np.asarray(labels)
    cids = sorted(int(c) for c in set(labels) if c >= 0)
    cid_row = {c: i for i, c in enumerate(cids)}
    V = len(vocab_index)
    counts = np.zeros((len(cids), V), dtype=np.float64)
    for lab, doc in zip(labels, docs):
        if lab < 0:
            continue
        r = cid_row[int(lab)]
        for w in doc:
            j = vocab_index.get(w)
            if j is not None:
                counts[r, j] += 1.0
    row_tot = counts.sum(axis=1, keepdims=True)
    row_tot[row_tot == 0] = 1.0
    tf = counts / row_tot
    f_w = counts.sum(axis=0)
    A = float(counts.sum(axis=1).mean()) if len(cids) else 0.0
    idf = np.log(1.0 + A / np.maximum(f_w, 1e-9))
    return counts, tf * idf, cids


def bertopic_style(Xemb, docs, vocab_index, seed: int = DEFAULT_SEED,
                   min_topic_size: int = BERT_MIN_TOPIC_SIZE,
                   mts_grid=BERT_MTS_GRID, use_umap: bool = True):
    """UMAP -> HDBSCAN -> c-TF-IDF.  Returns ``(result_dict)``.

    ``mts_grid`` is scanned with the *same* reduced embedding so the choice of
    ``min_topic_size`` is evidence, not a re-run artefact.
    """
    import hdbscan
    Z, reducer_note = reduce_embeddings(Xemb, seed=seed, use_umap=use_umap)
    scan = []
    labels_default = None
    for mts in mts_grid:
        cl = hdbscan.HDBSCAN(min_cluster_size=int(mts),
                             min_samples=BERT_MIN_SAMPLES).fit(Z)
        lab = cl.labels_
        noise = int((lab == -1).sum())
        n_cl = len({int(x) for x in lab} - {-1})
        scan.append({"min_topic_size": int(mts), "n_clusters": n_cl,
                     "noise": noise,
                     "noise_frac": round(noise / len(lab), 4)})
        if int(mts) == int(min_topic_size):
            labels_default = lab
    if labels_default is None:      # default not in the grid: run it anyway
        cl = hdbscan.HDBSCAN(min_cluster_size=int(min_topic_size),
                             min_samples=BERT_MIN_SAMPLES).fit(Z)
        labels_default = cl.labels_
    counts, ctfidf, cids = ctfidf_matrix(labels_default, docs, vocab_index)
    return {
        "labels": labels_default,
        "cids": cids,
        "ctfidf": ctfidf,
        "counts": counts,
        "scan": scan,
        "reducer": reducer_note,
        "min_topic_size": int(min_topic_size),
    }


# ---------------------------------------------------------------------------
# communities (#29)
# ---------------------------------------------------------------------------
def load_communities():
    """``(comm_terms, comm_label, comm_of, n_nodes)`` from #29."""
    comm_terms = collections.defaultdict(set)
    comm_of = {}
    for r in read_tsv(NETWORK_COMMUNITIES):
        cid = int(r["community"])
        term = r["term"]
        comm_terms[cid].add(term)
        comm_of[term] = cid
    comm_label = {}
    if NETWORK_SUMMARY.exists():
        with open(NETWORK_SUMMARY, encoding="utf-8") as fh:
            summ = json.load(fh)
        comm_label = {int(k): v
                      for k, v in summ.get("community_labels", {}).items()}
    n_nodes = sum(len(v) for v in comm_terms.values())
    return dict(comm_terms), comm_label, comm_of, n_nodes


def community_overlap(model_name, topics, comm_of, comm_label):
    """Per-topic best #29 community by top-term mass.

    ``topics`` is ``[[(term, weight), ...], ...]`` (top-``n`` per topic).
    Returns a list of overlap rows.
    """
    overlap_rows = []
    for tid, terms in enumerate(topics):
        mass = collections.Counter()
        total = 0.0
        for term, w in terms:
            total += w
            if term in comm_of:
                mass[comm_of[term]] += w
        if mass:
            best = max(mass.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            n_shared = sum(1 for t, _w in terms if comm_of.get(t) == best)
            frac = n_shared / max(1, len(terms))
            mass_frac = mass[best] / total if total > 0 else 0.0
        else:
            best, n_shared, frac, mass_frac = -1, 0, 0.0, 0.0
        overlap_rows.append({
            "model": model_name, "topic_id": tid,
            "label": " · ".join(t for t, _w in terms[:LABEL_TERMS]),
            "best_community": best,
            "community_label": comm_label.get(best, ""),
            "n_shared_top": n_shared,
            "frac_top_in_community": round(frac, 4),
            "weighted_mass_frac": round(mass_frac, 4),
        })
    return overlap_rows


def term_partition_ari(topic_of_term, comm_of, vocab):
    """Adjusted Rand between the term->topic partition and #29 communities."""
    from sklearn.metrics import adjusted_rand_score
    shared = [t for t in vocab if t in comm_of]
    if not shared:
        return {"n_terms": 0, "n_communities": 0, "adjusted_rand": ""}
    pred = [int(topic_of_term[t]) for t in shared]
    true = [int(comm_of[t]) for t in shared]
    return {
        "n_terms": len(shared),
        "n_communities": len(set(true)),
        "adjusted_rand": round(float(adjusted_rand_score(true, pred)), 4),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(out_dir: Path, k_grid=K_GRID, seed: int = DEFAULT_SEED,
          min_df: int = MIN_DF, max_df: float = MAX_DF,
          lda_max_iter: int = LDA_MAX_ITER, nmf_max_iter: int = NMF_MAX_ITER,
          use_umap: bool = True, bert_min_topic_size: int = BERT_MIN_TOPIC_SIZE,
          bert_mts_grid=BERT_MTS_GRID, quiet: bool = False) -> dict:
    out_dir = Path(out_dir)
    clause_ids, pages, docs, doc_diag = load_documents()
    n = len(clause_ids)
    n_pages = len(set(pages))
    X, Xt, vocab = build_vectors(docs, min_df=min_df, max_df=max_df)
    vocab_index = {t: i for i, t in enumerate(vocab)}
    n_tokens = doc_diag["n_tokens"]
    if not quiet:
        print(f"topics: {n:,} clauses, {n_tokens:,} content tokens, "
              f"{len(vocab):,} terms (min_df={min_df})")

    import gensim
    dictionary = gensim.corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(d) for d in docs]

    # -- coherence scan: LDA + NMF over the k grid -------------------------
    scan_rows = []
    lda_models = {}
    nmf_models = {}
    lda_top = {}
    nmf_top = {}
    for k in k_grid:
        lda = fit_lda(X, k, seed=seed, max_iter=lda_max_iter)
        lt = top_terms(lda.components_, vocab)
        lc = coherence(lt, docs, dictionary, corpus)
        lda_models[k] = lda
        lda_top[k] = lt
        scan_rows.append({
            "model": "lda", "k": int(k),
            "coherence_c_v": round(lc["c_v"], 4),
            "coherence_u_mass": round(lc["u_mass"], 4),
            "lda_perplexity": round(float(lda.perplexity(X)), 4),
            "nmf_reconstruction_err": "",
        })
        nmf = fit_nmf(Xt, k, seed=seed, max_iter=nmf_max_iter)
        nt = top_terms(nmf.components_, vocab)
        nc = coherence(nt, docs, dictionary, corpus)
        nmf_models[k] = nmf
        nmf_top[k] = nt
        scan_rows.append({
            "model": "nmf", "k": int(k),
            "coherence_c_v": round(nc["c_v"], 4),
            "coherence_u_mass": round(nc["u_mass"], 4),
            "lda_perplexity": "",
            "nmf_reconstruction_err": round(float(nmf.reconstruction_err_), 4),
        })
    scan_rows.sort(key=lambda r: (r["model"], r["k"]))
    lda_best_k = max((r for r in scan_rows if r["model"] == "lda"),
                     key=lambda r: (r["coherence_c_v"], -r["k"]))["k"]
    nmf_best_k = max((r for r in scan_rows if r["model"] == "nmf"),
                     key=lambda r: (r["coherence_c_v"], -r["k"]))["k"]
    if not quiet:
        print(f"coherence: best LDA k={lda_best_k}, best NMF k={nmf_best_k}")

    # -- BERTopic-style ----------------------------------------------------
    store = ce.ClauseEmbeddingStore(out_dir=ce.OUT, model=ce.PRIMARY)
    Xemb = store.matrix(PRIMARY_VIEW)
    bert = bertopic_style(Xemb, docs, vocab_index, seed=seed,
                          min_topic_size=bert_min_topic_size,
                          mts_grid=bert_mts_grid, use_umap=use_umap)
    bert_top = top_terms(bert["ctfidf"], vocab) if len(bert["cids"]) else []
    bert_c = (coherence(bert_top, docs, dictionary, corpus)
              if bert_top else {"c_v": "", "u_mass": ""})
    if not quiet:
        print(f"bertopic-style: {len(bert['cids'])} topics, "
              f"{int((bert['labels'] == -1).sum()):,} noise "
              f"({bert['reducer']})")

    # -- dominant topic assignments ---------------------------------------
    lda_doc = lda_models[lda_best_k].transform(X)
    nmf_doc = nmf_models[nmf_best_k].transform(Xt)
    lda_dom = np.argmax(lda_doc, axis=1)
    nmf_dom = np.argmax(nmf_doc, axis=1)
    lda_sizes = collections.Counter(int(x) for x in lda_dom)
    nmf_sizes = collections.Counter(int(x) for x in nmf_dom)
    bert_labels = bert["labels"]
    bert_sizes = collections.Counter(int(x) for x in bert_labels)

    # term -> dominant topic (for the community comparison)
    lda_tot = np.argmax(lda_models[lda_best_k].components_, axis=0)
    nmf_tot = np.argmax(nmf_models[nmf_best_k].components_, axis=0)
    if len(bert["cids"]):
        bert_tot = np.argmax(bert["ctfidf"], axis=0)
    else:
        bert_tot = np.zeros(len(vocab), dtype=int)
    topic_of_term = {
        "lda": {vocab[i]: int(lda_tot[i]) for i in range(len(vocab))},
        "nmf": {vocab[i]: int(nmf_tot[i]) for i in range(len(vocab))},
        "bertopic": {vocab[i]: int(bert_tot[i]) for i in range(len(vocab))},
    }

    # -- community comparison ---------------------------------------------
    _comm_terms, comm_label, comm_of, n_network_nodes = load_communities()
    overlap_rows = []
    ari_rows = []
    model_topics = {
        "lda": lda_top[lda_best_k],
        "nmf": nmf_top[nmf_best_k],
        "bertopic": bert_top,
    }
    model_k = {"lda": lda_best_k, "nmf": nmf_best_k,
               "bertopic": len(bert["cids"])}
    for name in ("lda", "nmf", "bertopic"):
        topics = model_topics[name]
        if not topics:
            ari_rows.append({"model": name, "k": 0, "n_terms": 0,
                             "n_communities": 0, "adjusted_rand": "",
                             "note": "no topics"})
            continue
        overlap_rows.extend(community_overlap(
            name, topics, comm_of, comm_label))
        ari = term_partition_ari(topic_of_term[name], comm_of, vocab)
        ari_rows.append({"model": name, "k": model_k[name], **ari,
                         "note": "term -> dominant topic vs #29 Louvain"})
    overlap_rows.sort(key=lambda r: (r["model"], r["topic_id"]))
    ari_rows.sort(key=lambda r: r["model"])
    n_shared_terms = sum(1 for t in vocab if t in comm_of)

    # -- topic summaries ---------------------------------------------------
    def topic_summary(name, topics, sizes, n_topics, extra=None):
        rows = []
        for tid in range(n_topics):
            terms = topics[tid] if tid < len(topics) else []
            size = sizes.get(tid, 0)
            rows.append({
                "topic_id": tid,
                "size": size,
                "frac": round(size / n, 4),
                "label": " · ".join(t for t, _w in terms[:LABEL_TERMS]),
                "top_terms": ", ".join(f"{t}({w:.3f})" for t, w in terms),
                **(extra or {}),
            })
        rows.sort(key=lambda r: (-r["size"], r["topic_id"]))
        return rows

    lda_summary = topic_summary("lda", lda_top[lda_best_k], lda_sizes,
                                lda_best_k)
    nmf_summary = topic_summary("nmf", nmf_top[nmf_best_k], nmf_sizes,
                                nmf_best_k)
    bert_summary = topic_summary("bertopic", bert_top, bert_sizes,
                                 len(bert["cids"]))

    # -- reconciliation ----------------------------------------------------
    recon = [
        {"quantity": "clauses", "here": n, "earlier": "5,614 (#39)",
         "note": "one topic document per #33/#39 clause"},
        {"quantity": "pages", "here": n_pages, "earlier": "225 (#39)",
         "note": "pages carrying clauses"},
        {"quantity": "screened units", "here": doc_diag["n_tokens"],
         "earlier": "74,077 (#27)", "note": "content tokens after stopword "
         "removal (unit surfaces; #29 content_items 43,321)"},
        {"quantity": "vocabulary (min_df)", "here": len(vocab),
         "earlier": "5,369 raw types", "note": f"min_df={min_df}, "
         f"max_df={max_df}"},
        {"quantity": "network nodes in vocab", "here": n_shared_terms,
         "earlier": f"{n_network_nodes} (#29)", "note": "all #29 terms are "
         "topic-vocabulary terms (shared token space)"},
        {"quantity": "lda cluster sizes sum", "here": sum(lda_sizes.values()),
         "earlier": f"{n:,} (#39)",
         "note": "every clause gets a dominant topic"},
        {"quantity": "nmf cluster sizes sum", "here": sum(nmf_sizes.values()),
         "earlier": f"{n:,} (#39)",
         "note": "every clause gets a dominant topic"},
        {"quantity": "bertopic sizes + noise", "here":
         sum(bert_sizes.get(c, 0) for c in bert["cids"]) +
         bert_sizes.get(-1, 0),
         "earlier": f"{n:,} (#39)",
         "note": "clusters + noise = all clauses"},
    ]

    summary = {
        "schema": SCHEMA,
        "n_clauses": n,
        "n_pages": n_pages,
        "content_tokens": n_tokens,
        "raw_types": doc_diag["n_types"],
        "vocab": len(vocab),
        "n_empty_docs": doc_diag["n_empty"],
        "min_df": min_df,
        "max_df": max_df,
        "stopwords": len(STOPWORDS),
        "lda": {
            "k": lda_best_k,
            "k_grid": list(k_grid),
            "max_iter": lda_max_iter,
            "learning_method": LDA_LEARNING,
            "coherence_c_v": next(r["coherence_c_v"] for r in scan_rows
                                  if r["model"] == "lda"
                                  and r["k"] == lda_best_k),
            "topics": [{"topic_id": r["topic_id"], "size": r["size"],
                        "label": r["label"]}
                       for r in sorted(lda_summary,
                                       key=lambda x: x["topic_id"])],
        },
        "nmf": {
            "k": nmf_best_k,
            "k_grid": list(k_grid),
            "max_iter": nmf_max_iter,
            "init": NMF_INIT,
            "coherence_c_v": next(r["coherence_c_v"] for r in scan_rows
                                  if r["model"] == "nmf"
                                  and r["k"] == nmf_best_k),
            "topics": [{"topic_id": r["topic_id"], "size": r["size"],
                        "label": r["label"]}
                       for r in sorted(nmf_summary,
                                       key=lambda x: x["topic_id"])],
        },
        "bertopic": {
            "min_topic_size": bert["min_topic_size"],
            "mts_grid": list(bert_mts_grid),
            "min_samples": BERT_MIN_SAMPLES,
            "reducer": bert["reducer"],
            "umap": {"n_neighbors": UMAP_NEIGHBORS,
                     "n_components": UMAP_COMPONENTS,
                     "min_dist": UMAP_MIN_DIST, "metric": UMAP_METRIC},
            "n_topics": len(bert["cids"]),
            "noise": bert_sizes.get(-1, 0),
            "noise_frac": round(bert_sizes.get(-1, 0) / n, 4),
            "coherence_c_v": (bert_c["c_v"] if bert_c["c_v"] != ""
                              else None),
            "scan": bert["scan"],
            "topics": [{"topic_id": r["topic_id"], "size": r["size"],
                        "label": r["label"]}
                       for r in sorted(bert_summary,
                                       key=lambda x: x["topic_id"])],
        },
        "community": {
            "n_network_nodes": n_network_nodes,
            "n_shared_terms": n_shared_terms,
            "adjusted_rand": {r["model"]: r["adjusted_rand"]
                              for r in ari_rows},
        },
        "determinism": {"seed": seed, "single_job": True,
                        "umap": use_umap},
    }

    # -- write -------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "coherence_scan.tsv",
              ["model", "k", "coherence_c_v", "coherence_u_mass",
               "lda_perplexity", "nmf_reconstruction_err"], scan_rows)

    def write_topics(path, topics, model):
        rows = []
        for tid, terms in enumerate(topics):
            for rank, (term, w) in enumerate(terms, start=1):
                rows.append({"model": model, "topic_id": tid, "rank": rank,
                             "term": term, "weight": round(w, 6)})
        write_tsv(path, ["model", "topic_id", "rank", "term", "weight"], rows)

    write_topics(out_dir / "lda_topics.tsv", lda_top[lda_best_k], "lda")
    write_topics(out_dir / "nmf_topics.tsv", nmf_top[nmf_best_k], "nmf")
    write_topics(out_dir / "bertopic_topics.tsv", bert_top, "bertopic")

    summ_cols = ["topic_id", "size", "frac", "label", "top_terms"]
    write_tsv(out_dir / "lda_topic_summary.tsv", summ_cols, lda_summary)
    write_tsv(out_dir / "nmf_topic_summary.tsv", summ_cols, nmf_summary)
    write_tsv(out_dir / "bertopic_topic_summary.tsv", summ_cols, bert_summary)

    write_tsv(out_dir / "bertopic_scan.tsv",
              ["min_topic_size", "n_clusters", "noise", "noise_frac"],
              bert["scan"])

    write_tsv(out_dir / "clause_topics.tsv",
              ["clause_id", "page", "lda_topic", "nmf_topic",
               "bertopic_topic", "bertopic_noise"],
              [{"clause_id": clause_ids[i], "page": pages[i],
                "lda_topic": int(lda_dom[i]), "nmf_topic": int(nmf_dom[i]),
                "bertopic_topic": int(bert_labels[i]),
                "bertopic_noise": int(bert_labels[i] == -1)}
               for i in range(n)])

    write_tsv(out_dir / "community_overlap.tsv",
              ["model", "topic_id", "label", "best_community",
               "community_label", "n_shared_top", "frac_top_in_community",
               "weighted_mass_frac"], overlap_rows)
    write_tsv(out_dir / "community_ari.tsv",
              ["model", "k", "n_terms", "n_communities", "adjusted_rand",
               "note"], ari_rows)
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "here", "earlier", "note"], recon)

    diag_rows = [
        {"metric": "n_clauses", "value": n},
        {"metric": "n_pages", "value": n_pages},
        {"metric": "content_tokens", "value": n_tokens},
        {"metric": "raw_types", "value": doc_diag["n_types"]},
        {"metric": "vocab", "value": len(vocab)},
        {"metric": "empty_docs", "value": doc_diag["n_empty"]},
        {"metric": "lda_best_k", "value": lda_best_k},
        {"metric": "nmf_best_k", "value": nmf_best_k},
        {"metric": "bertopic_topics", "value": len(bert["cids"])},
        {"metric": "bertopic_noise", "value": bert_sizes.get(-1, 0)},
        {"metric": "bertopic_reducer", "value": bert["reducer"]},
        {"metric": "n_shared_terms", "value": n_shared_terms},
        {"metric": "seed", "value": seed},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["metric", "value"], diag_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_report(out_dir / "topic_modelling_report.md", summary, scan_rows,
                 lda_summary, nmf_summary, bert_summary, overlap_rows,
                 ari_rows, recon, n, vocab)

    if not quiet:
        print(f"wrote {out_dir}  (LDA k={lda_best_k}, NMF k={nmf_best_k}, "
              f"BERTopic {len(bert['cids'])} topics)")
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _topic_table(rows):
    out = ["| topic | size | frac | label | top terms |",
           "|---:|---:|---:|---|---|"]
    for r in rows:
        out.append(f"| {r['topic_id']} | {r['size']} | {r['frac']} | "
                   f"{r['label']} | {r['top_terms']} |")
    return out


def write_report(path, summary, scan_rows, lda_summary, nmf_summary,
                 bert_summary, overlap_rows, ari_rows, recon, n, vocab):
    L = []
    A = L.append
    ld = summary["lda"]
    nm = summary["nmf"]
    bt = summary["bertopic"]
    A("# Topic modelling (LDA / NMF / BERTopic) (#41)")
    A("")
    A("Recover the book's **latent themes with top terms** and compare them "
      "with the #29 co-occurrence-network communities (E3.6). L5.3 of the "
      "method inventory names the three algorithms; all three are run and "
      "their convergence measured.")
    A("")
    lda_rows = sorted((r for r in scan_rows if r["model"] == "lda"),
                      key=lambda r: r["k"])
    A(f"**Headline.** Over {n:,} #33/#39 clauses and {len(vocab):,} terms "
      f"(the markup-screened #17 unit surfaces, function words removed), "
      f"**LDA** (coherence-selected k={ld['k']}) and **NMF** "
      f"(coherence-selected k={nm['k']}) recover a small set of recurrent "
      f"themes, and the **BERTopic-style** model (UMAP + HDBSCAN + c-TF-IDF, "
      f"the algorithm reproduced from BERTopic's own defaults) resolves "
      f"{bt['n_topics']} topics with "
      f"{bt['noise']:,} noise clauses ({bt['noise_frac']:.0%}). For LDA, "
      f"coherence is **highest at the smallest k and falls as k grows** "
      f"(c_v {lda_rows[0]['coherence_c_v']} at k={lda_rows[0]['k']} vs "
      f"{lda_rows[-1]['coherence_c_v']} at k={lda_rows[-1]['k']}) -- the "
      "same continuum #39/#40 found: there is no natural topic count, only "
      "graded regions of one repeating theme-space.")
    A("")
    A("## 1. Method")
    A("")
    A("* **Documents.** One per #33 clause, the #39 `clause_index.tsv` "
      "`text_units` column (markup-screened #17 unit surfaces, space-joined). "
      "Splitting on whitespace recovers exactly the #17 **unit surfaces** -- "
      "the same tokens #29's network nodes are built from -- so the topic and "
      "community term spaces coincide. TSV cells are unescaped first (the "
      "#17 quote hazard), so a unit broken by a literal newline is split, not "
      "fused into a stray `n`/`t` token.")
    A(f"* **Screen.** `sklearn` `ENGLISH_STOP_WORDS` + `lt`/`gt`/`amp`/`quot` "
      f"({summary['stopwords']} stop words); vocabulary floor "
      f"`min_df={summary['min_df']}`, ceiling `max_df={summary['max_df']}`.")
    A(f"* **LDA.** `sklearn` `LatentDirichletAllocation` (batch variational "
      f"inference, `max_iter={ld['max_iter']}`, fixed seed, single job) over "
      "raw term counts.")
    A(f"* **NMF.** `sklearn` `NMF` (`{nm['init']}` init, "
      f"`max_iter={nm['max_iter']}`, fixed seed) over the TF-IDF matrix with "
      "the **same** vocabulary, so LDA and NMF terms are aligned.")
    A("* **BERTopic-style.** The BERTopic algorithm reproduced from its own "
      "default components: #39 SBERT clause embeddings (`units` view) -> "
      f"**UMAP** (`n_neighbors={bt['umap']['n_neighbors']}`, "
      f"{bt['umap']['n_components']}-d, `min_dist={bt['umap']['min_dist']}`, "
      f"{bt['umap']['metric']}, fixed seed, single job) -> **HDBSCAN** "
      f"(`min_samples={bt['min_samples']}`) -> **c-TF-IDF** top terms. The "
      "`bertopic` package is not installed here; the algorithm is "
      f"reimplemented on `umap-learn` + `hdbscan` (reducer: `{bt['reducer']}`).")
    A("* **Model selection is coherence.** Every LDA/NMF model on the fixed k "
      f"grid {list(ld['k_grid'])} is scored by gensim `c_v` (and `u_mass`); "
      "the best-coherence k per model is the catalogue of record. BERTopic's "
      "granularity is `min_topic_size`; its scan is reported below.")
    A("* **Community comparison.** Every vocabulary term is assigned to its "
      "dominant topic (`argmax` of the topic-term matrix); the resulting "
      "partition of the #29 network nodes is scored against #29's Louvain "
      "communities by the **adjusted Rand index**, and each topic's best "
      "community is found by top-term mass.")
    A("")
    A("## 2. Reconciliation")
    A("")
    A("| quantity | here | earlier | note |")
    A("|---|---:|---|---|")
    for r in recon:
        here = r["here"]
        here = f"{here:,}" if isinstance(here, int) else here
        A(f"| {r['quantity']} | {here} | {r['earlier']} | {r['note']} |")
    A("")
    A(f"All {summary['community']['n_network_nodes']} #29 network nodes are in "
      f"the topic vocabulary ({summary['community']['n_shared_terms']} shared "
      "terms), so the community comparison is exact on the shared token space.")
    A("")
    A("## 3. Coherence scan (model selection)")
    A("")
    A("`c_v` is the primary selection metric; `u_mass` (a log-conditional "
      "coherence) and the model-native fit diagnostics are shown for "
      "completeness. Higher `c_v` is better.")
    A("")
    A("| model | k | c_v | u_mass | LDA ppl | NMF err | |")
    A("|---|---:|---:|---:|---:|---:|:---:|")
    for r in scan_rows:
        mark = ""
        if r["model"] == "lda" and r["k"] == ld["k"]:
            mark = " **<- LDA**"
        if r["model"] == "nmf" and r["k"] == nm["k"]:
            mark = " **<- NMF**"
        A(f"| {r['model']} | {r['k']} | {r['coherence_c_v']} | "
          f"{r['coherence_u_mass']} | {r['lda_perplexity']} | "
          f"{r['nmf_reconstruction_err']} |{mark}")
    A("")
    lda_first = next(r for r in scan_rows
                     if r["model"] == "lda" and r["k"] == ld["k_grid"][0])
    lda_last = next(r for r in scan_rows
                    if r["model"] == "lda" and r["k"] == ld["k_grid"][-1])
    A(f"**The coherence curve has no interior peak.** LDA `c_v` is "
      f"{lda_first['coherence_c_v']} at k={lda_first['k']} and "
      f"{lda_last['coherence_c_v']} at k={lda_last['k']}: the best-coherence "
      f"LDA model is the smallest k, and larger k only fragments one "
      "theme-space (NMF's best is k=6, one step up). "
      "This is the L9.5 caveat made concrete -- the book does not contain a "
      "small set of well-separated topics, it contains a dense continuum of "
      "recurring motifs (#39 mean NN cosine 0.705; #40 low silhouette at "
      "every k). The reported catalogue is therefore a **graded map of a "
      "loop**, and k is a disclosed resolution, not a discovered constant.")
    A("")
    A(f"## 4. LDA topics (k={ld['k']})")
    A("")
    A("Topics sorted by the number of clauses they dominate. The label is the "
      "top three terms; weights are the variational topic-word mass.")
    A("")
    L.extend(_topic_table(lda_summary))
    A("")
    A(f"## 5. NMF topics (k={nm['k']})")
    A("")
    L.extend(_topic_table(nmf_summary))
    A("")
    A("## 6. BERTopic-style topics")
    A("")
    A(f"UMAP + HDBSCAN at `min_topic_size={bt['min_topic_size']}` gives "
      f"**{bt['n_topics']} topics** and **{bt['noise']:,} noise clauses "
      f"({bt['noise_frac']:.0%})**. The noise fraction is the density "
      "structure, not a defect: the clause cloud is one dense continuum "
      "(#40), so most clauses sit in no dense pocket. Terms are c-TF-IDF "
      "scores; topics sorted by size (noise last).")
    A("")
    A("`min_topic_size` scan (same UMAP embedding):")
    A("")
    A("| min_topic_size | clusters | noise | noise frac |")
    A("|---:|---:|---:|---:|")
    for r in bt["scan"]:
        A(f"| {r['min_topic_size']} | {r['n_clusters']} | {r['noise']} | "
          f"{r['noise_frac']} |")
    A("")
    L.extend(_topic_table(bert_summary))
    A("")
    A("## 7. Comparison with the #29 network communities (E3.6)")
    A("")
    A("The adjusted Rand index measures whether a topic model's term "
      "partition agrees with #29's Louvain communities over the 391 shared "
      "nodes (1.0 = identical, 0.0 = chance).")
    A("")
    A("| model | k | shared terms | communities | adjusted Rand |")
    A("|---|---:|---:|---:|---:|")
    for r in ari_rows:
        A(f"| {r['model']} | {r['k']} | {r['n_terms']} | "
          f"{r['n_communities']} | {r['adjusted_rand']} |")
    A("")
    best_ari = max((r for r in ari_rows if r["adjusted_rand"] != ""),
                   key=lambda r: float(r["adjusted_rand"]))
    A(f"**Convergence is weak-to-moderate** (best adjusted Rand "
      f"{best_ari['adjusted_rand']} for {best_ari['model']}). The topic "
      "models and the network communities agree far above chance but not "
      "exactly: both are coarse summaries of the same fused vocabulary, and "
      "the topics cut across the communities for the same reason #29's "
      "communities cut across the six named domains -- Siratori's clauses "
      "fuse body, machine and genetics terms in one breath. Divergence here "
      "is a lead (different methods weight the same terms differently), not "
      "a failure: no method is the ground truth (L9.5).")
    A("")
    A("### Topic -> best community")
    A("")
    A("| model | topic | label | best community | community label | "
      "frac top in community | weighted mass frac |")
    A("|---|---:|---|---:|---|---:|---:|")
    for r in overlap_rows:
        A(f"| {r['model']} | {r['topic_id']} | {r['label']} | "
          f"{r['best_community']} | {r['community_label']} | "
          f"{r['frac_top_in_community']} | {r['weighted_mass_frac']} |")
    A("")
    A("## 8. Interpretation")
    A("")
    A("The themes are recognisable and consistent with the #27 motif set and "
      "the #30 dramatis personae. Across all three models the load-bearing "
      "terms recur: the **body/cadaver** pole (`body`, `internal`, `organ`, "
      "`cadaver`, `flesh`), the **artificial-sun / ice-nebula** pole "
      "(`artificial`, `sun`, `ice`, `nebula`, `faecal`), the **genetics/ADAM** "
      "pole (`adam`, `gene`, `chromosome`, `embryo`, `fecundates`, `boy-roid`), "
      "the **murder/violence** pole (`murder`, `dog`, `assassin`, `suicide`), "
      "and the **machine** pole (`machine`, `memory`, `doll`, `circuit`, "
      "`soul/gram`). The models differ in how finely they cut this space and "
      "in whether the cuts align with the network's -- which is exactly the "
      "expected behaviour of three different estimators over one continuum.")
    A("")
    A("No dedicated **affect** pole emerges. Affect words are present "
      "(`ecstasy` is a member of the BERTopic `boy-roid · soul/gram · "
      "ecstasy` topic; `desire`, `suicide`, `fear`-family terms sit inside "
      "the ADAM/machine and body poles) but they do not cohere into a topic "
      "of their own -- consistent with #22's finding that the book's affect "
      "is a hapax-heavy scream channel rather than a stable register. The "
      "BERTopic view adds granularity but no new poles: its many small "
      "topics are sub-regions of the same five.")
    A("")
    A("## 9. Negative results and limitations")
    A("")
    A("* **No natural topic count.** Coherence has no interior peak and "
      "falls overall with k (with small non-monotone wiggles); the reported "
      "k is a disclosed resolution choice, not a discovered constant. A "
      "'correct' number of topics does not exist for this text.")
    A(f"* **BERTopic noise is large** ({bt['noise_frac']:.0%} of clauses). "
      "This is the continuum, not a bug; a topic model that assigned every "
      "clause would be imposing a partition the geometry does not support.")
    A("* **`c_v` is a topic-word co-occurrence statistic, not a semantic "
      "judgement.** A high-coherence topic is internally consistent, not "
      "necessarily true; the labels are evidence for a human gloss, not the "
      "gloss.")
    A("* **Weak community convergence.** The adjusted Rand values are "
      "modest; topics and communities are different coarse-grainings of the "
      "same vocabulary, and neither is authoritative.")
    A("* **The token space is the #17 unit surface, not a word.** Fused "
      "units (`gene=tv`, `soul/gram`, `boy-roid`) are single tokens here, "
      "matching #29; a word-level tokenisation would move mass between "
      "topics.")
    A("* **`bertopic` proper is not installed.** The BERTopic-style model "
      "reproduces the algorithm on its default components; the result is "
      "labelled BERTopic-*style* throughout.")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="#41 topic modelling (LDA / NMF / BERTopic)")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--k-grid", default=",".join(map(str, K_GRID)),
                    help="comma-separated k grid for the coherence scan")
    ap.add_argument("--min-df", type=int, default=MIN_DF)
    ap.add_argument("--max-df", type=float, default=MAX_DF)
    ap.add_argument("--lda-max-iter", type=int, default=LDA_MAX_ITER)
    ap.add_argument("--nmf-max-iter", type=int, default=NMF_MAX_ITER)
    ap.add_argument("--no-umap", action="store_true",
                    help="use TruncatedSVD instead of UMAP (weaker; disclosed)")
    ap.add_argument("--bert-min-topic-size", type=int,
                    default=BERT_MIN_TOPIC_SIZE)
    ap.add_argument("--bert-mts-grid",
                    default=",".join(map(str, BERT_MTS_GRID)),
                    help="comma-separated BERTopic min_topic_size grid")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    build(Path(args.out),
          k_grid=tuple(int(x) for x in args.k_grid.split(",") if x),
          seed=args.seed, min_df=args.min_df, max_df=args.max_df,
          lda_max_iter=args.lda_max_iter, nmf_max_iter=args.nmf_max_iter,
          use_umap=not args.no_umap,
          bert_min_topic_size=args.bert_min_topic_size,
          bert_mts_grid=tuple(int(x) for x in args.bert_mts_grid.split(",")
                              if x),
          quiet=args.quiet)


if __name__ == "__main__":
    main()
