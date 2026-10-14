#!/usr/bin/env python3
"""#54 Self-similarity matrix (loop detection) -> analysis/self_similarity/

Build the **paragraph x paragraph cosine self-similarity matrix** of the book
over #39's clause vectors, then read it structurally for the two shapes the
task names: a **diagonal band** (a paragraph is similar to the paragraphs near
it -- local recurrence, the "loop" experienced while reading) and
**repeating blocks** (a whole stretch of paragraphs recurs far away in the
book -- the loop as an editorial structure).

Unit (stated once, used everywhere):

* a **paragraph** is one blank-line-separated run of the #7 layout page body
  (``analysis/layout_recovery/blood-electric.layout.md``) -- i.e. marker's
  ``Text`` block as #7 recovered it, with #7's headings and figure
  placeholders kept and flagged.  Character spans are in page-body
  coordinates, exactly the coordinates #33's clauses use.
* a **clause** (#33) is joined to a paragraph by an offset-overlap join on
  ``(page, start, end)`` -- the same containment style #33/#34/#39 use -- so
  every clause belongs to exactly one paragraph and no clause is dropped.
* the **paragraph vector** is the mean of its clause vectors from #39's
  primary artifact (SBERT ``all-MiniLM-L6-v2``, 384-d, L2-normalised,
  ``units`` view), renormalised.  No encoder is run for the book: the encoder
  is pinned by #39.  (The encoder *is* run, once, for the English control.)

Statistics (all deterministic, fixed seed):

* the full cosine matrix, plus a **page rollup** (mean paragraph vector per
  page, 225 x 225) -- the book-scale readable view;
* a **lag curve** -- mean cosine of paragraph pairs ``k`` apart -- against a
  fixed-seed paragraph-order permutation null (the null destroys serial
  position while preserving the marginal similarity distribution);
* **band widths** -- the share of above-threshold similarity mass that sits
  inside a band of |i-j| <= w, with the same null;
* **off-diagonal blocks** -- connected components of above-threshold cells
  with the local band removed, reported at paragraph and page level, with
  row/col spans, density and a ``difflib`` identity ratio on the raw text;
* controls: the **English paragraph baseline** (same encoder, real English
  paragraphs from #24's pinned reference cache at matched paragraph and
  sentence counts, document order preserved), a **lexical TF cosine**
  contrast (no encoder: answers "is the repetition lexical or semantic?"),
  and a **view/encoder sensitivity** check (#39's ``raw`` view and its
  secondary BGE encoder).

Outputs under ``analysis/self_similarity/``:
  self_similarity_report.md   THE DELIVERABLE
  paragraph_matrix.tsv        one row per paragraph: keys + unit/word counts
  paragraph_cosines.tsv       long-form pair list above the screen threshold
  page_matrix.tsv             the 225 x 225 page rollup, page ids in row 1
  lag_curve.tsv               lag -> observed cosine, null mean/sd, z
  page_lag_curve.tsv          the same analysis on the page rollup
  band_stats.tsv              band width -> in-band share, null, z
  offdiag_blocks.tsv          block catalogue (paragraph + page level)
  similarity.png              the matrix + page rollup, page rulers
  lag_curve.png               lag curve, band curve, nulls
  english_baseline.tsv        English paragraph control statistics
  sensitivity.tsv             view/encoder sensitivity of the headline stats
  reconciliation.tsv          against #39 / #33 / #17 / #27 / #7 / #35
  diagnostics.tsv             build-time counters
  summary.json                machine-readable headline numbers

Reproduce::

    conda activate marker
    python work/self_similarity.py             # writes analysis/self_similarity/
    python work/test_self_similarity.py

Deterministic: fixed seed for the permutation null and every control, no
hash-order dependence (every list sorted), no wall-clock in any artifact.
English control needs ``transformers``/``torch`` and the cached MiniLM
weights; if they are missing the build still succeeds and the report
discloses the absence (``--skip-english`` forces it, for speed in tests).
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
from pathlib import Path

import clause_embeddings as ce  # #39 ClauseEmbeddingStore + SentenceEncoder
import clause_segmentation as cs  # #33 quote-safe TSV io + page bodies
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "self_similarity"
STORE_DIR = ce.OUT                      # analysis/clause_embeddings/
REF_CACHE = ce.REF_CACHE                # #24's pinned English text cache

SCHEMA = "self-similarity/1"
PRIMARY_VIEW = "units"                  # #39 primary semantic view
SECONDARY_VIEW = "raw"                  # #39 comparison view
SECONDARY_MODEL = "bge"                 # #39 secondary encoder
DEFAULT_SEED = 0
DEFAULT_THREADS = 4

# The paragraph layer.
MIN_PARA_CHARS = 12                     # shorter runs are not paragraphs
# Similarity screens.
PAIR_SCREEN = 0.80                      # pairs listed in paragraph_cosines.tsv
BLOCK_THRESHOLD = 0.85                  # floor for block detection
BLOCK_QUANTILE = 0.995                  # and the relative cut at each level:
                                        # the page rollup is far more
                                        # anisotropic than the paragraph
                                        # matrix (mean 0.76 vs 0.54), so an
                                        # absolute cosine would screen the
                                        # page matrix at its 90th
                                        # percentile and dissolve every
                                        # block into noise blobs.
BAND_EXCLUDE = 2                        # local band removed before blocks
MIN_BLOCK_CELLS = 4                     # ignore 1-3 cell speckle
LONG_RANGE_GAP = 25                     # a block this far from its twin is
                                        # a repetition, not local cohesion
VERBATIM_IDENTITY = 0.50                # difflib ratio above which a block
                                        # is the same text, not the same theme
# Lag curve.
LAG_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50,
            75, 100, 150, 200, 300)
N_PERM = 200                            # paragraph-order permutation null
# Bands.
BAND_WIDTHS = (1, 2, 3, 5, 10, 20, 30, 50, 100)
# English control.
EN_MIN_WORDS = 3                        # matches #39's sentence filter
EN_MAX_WORDS = 60
EN_SENT_TARGET = 8                      # ~ the book's clauses-per-paragraph
# Lexical contrast.
LEX_MIN_DF = 2                          # terms appearing in >= this many paras

read_tsv = cs.read_tsv
write_tsv = cs.write_tsv

PARA_BREAK = re.compile(r"\n\s*\n")
HEADING_RE = re.compile(r"^#")
PLACEHOLDER_RE = re.compile(r"^<!--")
_GUT_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.DOTALL)
_GUT_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.DOTALL)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def _f(x, nd=4):
    """Format a number for a TSV/report cell ('' for None)."""
    if x is None or x == "":
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def _round(x, nd=4):
    if x is None:
        return None
    return round(float(x), nd)


# ---------------------------------------------------------------------------
# paragraphs
# ---------------------------------------------------------------------------
def split_paragraphs(body: str) -> list:
    """``[(start, end)]`` of blank-line-separated runs in ``body``.

    Spans are in the page-body string's own coordinates, so they compose
    directly with #33's clause ``(page, start, end)`` join.
    """
    spans = []
    pos = 0
    n = len(body)
    while pos < n:
        m = PARA_BREAK.search(body, pos)
        if m is None:
            end, nxt = n, n
        else:
            end, nxt = m.start(), m.end()
        if end > pos:
            spans.append((pos, end))
        if m is None:
            break
        pos = nxt
    return spans


def para_kind(text: str) -> str:
    """``heading`` / ``figure`` / ``text`` -- #7's block taxonomy."""
    s = text.strip()
    if HEADING_RE.match(s):
        return "heading"
    if PLACEHOLDER_RE.match(s):
        return "figure"
    return "text"


def paragraph_records(store) -> tuple:
    """One record per paragraph of the layout page bodies.

    Returns ``(records, diagnostics)``.  ``records`` are ordered by
    ``(page, start)``; every clause is joined to exactly one paragraph by
    maximum character-overlap (ties -> earliest paragraph), and the number of
    clauses whose span crosses a paragraph break is disclosed.
    """
    bodies = cs.load_pages()
    clause_rows, _ = ce.clause_records()

    paras = []
    for page in sorted(bodies):
        body = bodies[page]
        for (s, e) in split_paragraphs(body):
            text = body[s:e].strip()
            if len(text) < MIN_PARA_CHARS and para_kind(text) == "text":
                continue
            paras.append({
                "page": page,
                "start": s,
                "end": e,
                "kind": para_kind(text),
                "text": text,
            })    # index paragraphs by page for the join
    by_page = collections.defaultdict(list)
    for i, p in enumerate(paras):
        by_page[p["page"]].append(i)
    for page in by_page:
        by_page[page].sort(key=lambda i: paras[i]["start"])

    assign = collections.defaultdict(list)
    placed = set()
    n_cross = 0
    n_words = {}
    for r in clause_rows:
        page, s, e = r["page"], r["start"], r["end"]
        cands = by_page.get(page, ())
        if not cands:
            continue
        best_i, best_ov = None, -1
        for i in cands:
            ov = min(e, paras[i]["end"]) - max(s, paras[i]["start"])
            if ov > best_ov:
                best_i, best_ov = i, ov
        if best_i is None:
            continue
        assign[best_i].append(r["clause_id"])
        placed.add(r["clause_id"])
        if len(cands) > 1 and best_ov < (e - s):
            n_cross += 1
        n_words[r["clause_id"]] = r["n_words_clean"]
    records = []
    for i, p in enumerate(paras):
        ids = sorted(assign.get(i, ()))
        records.append({
            "para_id": i,
            "page": p["page"],
            "start": p["start"],
            "end": p["end"],
            "kind": p["kind"],
            "n_clauses": len(ids),
            "n_words_clean": sum(n_words.get(c, 0) for c in ids),
            "n_units_screened": 0,
            "text": p["text"],
            "clause_ids": ids,
        })
    # Coverage close: a clause that landed in no paragraph (a run shorter
    # than MIN_PARA_CHARS -- the book's last word is one) gets its own
    # paragraph record, so the join is total and nothing is silently lost.
    n_extra = 0
    for r in clause_rows:
        if r["clause_id"] in placed:
            continue
        records.append({
            "para_id": len(records),
            "page": r["page"], "start": r["start"], "end": r["end"],
            "kind": "text",
            "n_clauses": 1,
            "n_words_clean": r["n_words_clean"],
            "n_units_screened": 0,
            "text": (bodies.get(r["page"], "")[r["start"]:r["end"]]).strip(),
            "clause_ids": [r["clause_id"]],
        })
        n_extra += 1
    diags = {
        "n_paragraph_runs": len(paras),
        "n_paragraph_text": sum(1 for r in records if r["kind"] == "text"),
        "n_paragraph_heading": sum(1 for r in records if r["kind"] == "heading"),
        "n_paragraph_figure": sum(1 for r in records if r["kind"] == "figure"),
        "n_clauses_assigned": sum(len(r["clause_ids"]) for r in records),
        "n_clauses_crossing_para_break": n_cross,
        "n_paragraphs_without_clauses": sum(1 for r in records if not r["n_clauses"]),
        "n_paragraphs_added_for_coverage": n_extra,
        "n_pages": len(bodies),
    }
    return records, diags


def paragraph_units(records) -> dict:
    """``clause_id -> screened unit surface count`` from #39's records."""
    clause_rows, _ = ce.clause_records()
    counts = {}
    for r in clause_rows:
        counts[r["clause_id"]] = r["n_units_screened"]
    for rec in records:
        rec["n_units_screened"] = sum(counts.get(c, 0)
                                      for c in rec["clause_ids"])
    return counts


def paragraph_vectors(store, records, view=PRIMARY_VIEW) -> np.ndarray:
    """Mean-of-clause-embedding per paragraph, L2-normalised.

    Paragraphs with no clause contribute a zero row (kept, never dropped: the
    matrix index must stay aligned with ``paragraphs.tsv``); the caller
    screens them out of the similarity statistics.
    """
    V = store.matrix(view)
    n, d = len(records), V.shape[1]
    out = np.zeros((n, d), dtype=np.float64)
    for i, rec in enumerate(records):
        if not rec["clause_ids"]:
            continue
        rows = [store.row(c) for c in rec["clause_ids"]]
        v = V[rows].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        out[i] = v / nrm if nrm > 0 else v
    return out


def usable_rows(records) -> list:
    """Indices of paragraphs with at least one clause (finite vectors)."""
    return [i for i, r in enumerate(records) if r["n_clauses"]]


# ---------------------------------------------------------------------------
# the matrix and its structural statistics
# ---------------------------------------------------------------------------
def cosine_matrix(V: np.ndarray) -> np.ndarray:
    """Symmetric cosine matrix with a zeroed diagonal."""
    S = V @ V.T
    S = np.clip(S, -1.0, 1.0)
    np.fill_diagonal(S, 0.0)
    return S


def lag_curve(S: np.ndarray, grid=LAG_GRID) -> list:
    """For each lag ``d``: the rows of pairs ``(i, i+d)`` and their mean."""
    N = S.shape[0]
    rows = []
    for d in grid:
        if d >= N:
            continue
        iu = np.arange(N - d)
        vals = S[iu, iu + d]
        rows.append({
            "lag": d,
            "n_pairs": int(N - d),
            "mean_cos": float(vals.mean()),
            "median_cos": float(np.median(vals)),
            "max_cos": float(vals.max()),
            "frac_gt_screen": float((vals > PAIR_SCREEN).mean()),
        })
    return rows


def permutation_null(V: np.ndarray, grid=LAG_GRID, n_perm=N_PERM,
                     seed=DEFAULT_SEED, band_widths=BAND_WIDTHS) -> tuple:
    """Shuffle paragraph order ``n_perm`` times; collect lag + band nulls.

    The permutation destroys serial position and keeps the marginal
    similarity distribution, so a positive ``z`` at lag ``d`` means paragraphs
    ``d`` apart are *serially* closer than chance, not merely similar.
    """
    rng = np.random.default_rng(seed)
    N = V.shape[0]
    band_by_w = {w: [] for w in band_widths}
    lag_vals = {d: [] for d in grid if d < N}
    for _ in range(n_perm):
        perm = rng.permutation(N)
        S = V[perm] @ V[perm].T
        np.fill_diagonal(S, 0.0)
        for d, acc in lag_vals.items():
            iu = np.arange(N - d)
            acc.append(float(S[iu, iu + d].mean()))
        # band share: mass inside |i-j| <= w as a fraction of all
        # above-screen mass
        above = S > PAIR_SCREEN
        # S is symmetric with a zeroed diagonal, so .sum() counts every
        # unordered pair twice; halve it to match the upper-triangle
        # in-band counts below, so the share is a true pair share.
        total = float(above.sum()) / 2.0
        for w in band_widths:
            inband = 0
            for d in range(1, w + 1):
                idx = np.arange(N - d)
                inband += int(above[idx, idx + d].sum())
            band_by_w[w].append(inband / total if total else 0.0)

    lag_null = {}
    for d, vals in lag_vals.items():
        v = np.asarray(vals)
        lag_null[d] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    band_null = {}
    for w, vals in band_by_w.items():
        v = np.asarray(vals)
        band_null[w] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    return lag_null, band_null


def lag_rows_with_null(rows, lag_null) -> list:
    out = []
    for r in rows:
        nl = lag_null.get(r["lag"], {"mean": 0.0, "sd": 1e-12})
        sd = nl["sd"] or 1e-12
        obs = r["mean_cos"]
        # the null sd already *is* the sampling sd of this mean over pairs
        # (each permutation re-draws all N-d pairs), so no further scaling.
        z = (obs - nl["mean"]) / sd
        out.append({**r,
                    "null_mean": nl["mean"],
                    "null_sd": nl["sd"],
                    "z": float(z),
                    "delta": float(obs - nl["mean"])})
    return out


def band_rows(S: np.ndarray, band_null, band_widths=BAND_WIDTHS) -> list:
    N = S.shape[0]
    above = S > PAIR_SCREEN
    # symmetric matrix, zero diagonal: the true number of unordered pairs
    total = int(above.sum()) // 2
    rows = []
    for w in band_widths:
        inband = 0
        for d in range(1, w + 1):
            idx = np.arange(N - d)
            inband += int(above[idx, idx + d].sum())
        share = inband / total if total else 0.0
        nl = band_null.get(w, {"mean": 0.0, "sd": 0.0})
        sd = nl["sd"] or 1e-12
        rows.append({
            "band": w,
            "in_band_cells": inband,
            "total_cells_above_screen": total,
            "share_in_band": share,
            "null_mean": nl["mean"],
            "null_sd": nl["sd"],
            "z": float((share - nl["mean"]) / sd),
        })
    return rows


def top_pairs(S: np.ndarray, records, k=50, min_lag=1) -> list:
    """The ``k`` most similar paragraph pairs, in cosine order."""
    N = S.shape[0]
    rows_idx, cols_idx = np.triu_indices(N, min_lag)
    vals = S[rows_idx, cols_idx]
    order = np.argsort(-vals)[:k]
    out = []
    for o in order:
        i, j = int(rows_idx[o]), int(cols_idx[o])
        a, b = records[i], records[j]
        out.append({
            "para_i": i, "para_j": j,
            "page_i": a["page"], "page_j": b["page"],
            "lag": j - i,
            "cos": float(vals[o]),
            "kind_i": a["kind"], "kind_j": b["kind"],
            "text_i": a["text"][:400].rstrip(),
            "text_j": b["text"][:400].rstrip(),
        })
    return out


def _union_find_components(cells, connectivity=8):
    """Union-find over above-threshold cells.

    **8-connectivity is required, not cosmetic.**  A block in a similarity
    matrix is a *diagonal* rectangle -- its cells sit at (i, i+k), (i+1,
    i+k+1), ... -- and those cells are diagonal neighbours, so a 4-neighbour
    rule shreds every long-range block into scattered singletons and finds
    nothing.  With the king-move rule a block is a component.
    """
    parent = {c: c for c in cells}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if connectivity == 8:
        steps = tuple((di, dj) for di in (-1, 0, 1)
                      for dj in (-1, 0, 1) if di or dj)
    else:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for (i, j) in cells:
        for di, dj in steps:
            nb = (i + di, j + dj)
            if nb in parent:
                union((i, j), nb)
    groups = collections.defaultdict(list)
    for c in cells:
        groups[find(c)].append(c)
    return list(groups.values())


def block_threshold(S: np.ndarray, quantile=BLOCK_QUANTILE,
                    floor=BLOCK_THRESHOLD) -> float:
    """The cosine cut for block detection at this resolution.

    ``max(quantile cut, floor)``: relative so the paragraph and page levels
    are held to the same standard, absolute so the number stays
    interpretable ("a cosine of 0.93" rather than "the 99.5th percentile").
    """
    N = S.shape[0]
    if N < 2:
        return floor
    iu = np.triu_indices(N, 1)
    return float(max(np.quantile(S[iu], quantile), floor))


def offdiag_blocks(S: np.ndarray, records, threshold=None,
                   band_exclude=BAND_EXCLUDE, connectivity=8,
                   min_cells=MIN_BLOCK_CELLS) -> list:
    """Connected components of above-threshold cells, local band removed.

    Returns one row per block with its row/col paragraph spans, the pages they
    cover, cell count, density (cells / rows x cols) and the mean cosine.
    The two mirror halves of a block are merged: the component read the other
    way up is the same block.  ``connectivity=4`` is exposed so the regression
    suite can demonstrate that a 4-neighbour rule misses diagonal blocks.
    """
    if threshold is None:
        threshold = block_threshold(S)
    N = S.shape[0]
    mask = S > threshold
    for d in range(band_exclude + 1):
        idx = np.arange(N - d)
        mask[idx, idx + d] = False
        mask[idx + d, idx] = False
    cells = [(int(i), int(j)) for i, j in np.argwhere(mask)]
    comps = _union_find_components(cells, connectivity=connectivity)
    out = []
    seen = set()
    for comp in comps:
        if len(comp) < min_cells:
            continue
        rset = sorted({i for i, _ in comp})
        cset = sorted({j for _, j in comp})
        key = (tuple(min(rset, cset)), tuple(max(rset, cset)))
        if key in seen:
            continue
        seen.add(key)
        vals = np.array([S[i, j] for i, j in comp])
        pages_r = sorted({records[i]["page"] for i in rset})
        pages_c = sorted({records[j]["page"] for j in cset})
        gap = cset[0] - rset[-1]
        out.append({
            "rows": rset, "cols": cset,
            "n_cells": len(comp),
            "n_rows": len(rset), "n_cols": len(cset),
            "row_para_first": rset[0], "row_para_last": rset[-1],
            "col_para_first": cset[0], "col_para_last": cset[-1],
            "row_pages": pages_r, "col_pages": pages_c,
            "gap": gap,
            "long_range": bool(gap > LONG_RANGE_GAP),
            "density": len(comp) / max(len(rset) * len(cset), 1),
            "mean_cos": float(vals.mean()),
            "min_cos": float(vals.min()),
            "max_cos": float(vals.max()),
        })
    out.sort(key=lambda b: (-int(b["long_range"]), -b["n_cells"],
                            b["row_para_first"], b["col_para_first"]))
    return out


def long_range_blocks(blocks) -> list:
    """Only the blocks that do not touch the local band."""
    return [b for b in blocks if b["long_range"]]


def verbatim_blocks(blocks) -> list:
    """Long-range blocks whose two halves are the same *text*."""
    return [b for b in blocks if b["long_range"] and b.get("verbatim")]


def _annotate_block(block, records, S, lexS):
    """Add the lexical cosine and the raw-text identity of a block.

    ``records`` is whatever the block's indices refer to (paragraph records
    for the paragraph matrix, page-title records for the page rollup); the
    lex matrix must be at the same resolution.  ``autojunk=False`` is not
    optional: SequenceMatcher's default junk heuristic silently discards the
    most common characters on inputs over ~200 chars, which would zero the
    ratio exactly where the paragraphs are longest.
    """
    cells = np.array([[i, j] for i in block["rows"] for j in block["cols"]])
    block["lex_mean_cos"] = float(
        lexS[cells[:, 0], cells[:, 1]].mean())
    block["raw_identity"] = float(difflib.SequenceMatcher(
        None,
        " ".join(records[i]["text"] for i in block["rows"]).lower(),
        " ".join(records[j]["text"] for j in block["cols"]).lower(),
        autojunk=False).ratio())
    block["verbatim"] = bool(block["raw_identity"] >= VERBATIM_IDENTITY)
    return block


def _page_pool(M, page_of_row, page_ids) -> np.ndarray:
    """Pool a row-index matrix to page level by mean over the page pair.

    Deliberately a *mean of the pairwise cosines*, not a cosine of means:
    the question is how similar the paragraphs of one page are to the
    paragraphs of another, which a centroid would hide.
    """
    n = len(page_of_row)
    pidx = {p: i for i, p in enumerate(page_ids)}
    G = np.zeros((n, len(page_ids)))
    for i, p in enumerate(page_of_row):
        G[i, pidx[p]] = 1.0
    counts = G.sum(axis=0)
    denom = np.outer(counts, counts)
    denom[denom == 0] = 1.0
    return (G.T @ M @ G) / denom


def page_rollup(records, V: np.ndarray) -> tuple:
    """Mean paragraph vector per page -> ``(page_ids, page_vectors, matrix)``."""
    pages = sorted({r["page"] for r in records})
    idx = {p: [] for p in pages}
    for i, r in enumerate(records):
        if r["n_clauses"]:
            idx[r["page"]].append(i)
    out = []
    keep = []
    for p in pages:
        rows = idx[p]
        if not rows:
            continue
        v = V[rows].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        out.append(v / nrm if nrm else v)
        keep.append(p)
    M = np.vstack(out) if out else np.zeros((0, V.shape[1]))
    return keep, M, cosine_matrix(M)


def lexical_cosine(records) -> np.ndarray:
    """Unsupervised lexical control: TF cosine on screened unit surfaces.

    No encoder, no semantics: this asks whether the repeated blocks are
    *verbatim* (they must be, cosine ~ 1) while the semantic matrix answers
    the different question of near-but-not-identical recurrence.
    """
    clause_rows, _ = ce.clause_records()
    text_of = {r["clause_id"]: r["text_units"] for r in clause_rows}
    docs = []
    for r in records:
        docs.append(" ".join(text_of.get(c, "")
                             for c in r["clause_ids"]).strip())
    df = collections.Counter()
    for d in docs:
        for t in set(d.split()):
            df[t] += 1
    vocab = {t: i for i, t in enumerate(sorted(
        t for t, c in df.items() if c >= LEX_MIN_DF))}
    N, D = len(docs), len(vocab)
    M = np.zeros((N, D), dtype=np.float64)
    for i, d in enumerate(docs):
        for t in d.split():
            j = vocab.get(t)
            if j is not None:
                M[i, j] += 1.0
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return cosine_matrix(M / norms)


# ---------------------------------------------------------------------------
# English paragraph control (same encoder, real English paragraphs)
# ---------------------------------------------------------------------------
def english_paragraph_groups(records, n_paras=None) -> list:
    """Paragraph-sized groups of consecutive English sentences (#24 cache).

    Sentence counts per group cycle through the book's *observed*
    clauses-per-paragraph multiset (sorted, so the match is exact and
    deterministic). Groups are drawn from each reference text in turn, with
    an even quota per file, and respect document order, so no group ever
    mixes two books. Returns a list of dicts with ``text``, ``sentences`` and
    ``n_sentences``.
    """
    counts = sorted(r["n_clauses"] for r in records if r["n_clauses"])
    if not counts:
        return []
    if n_paras is None:
        n_paras = len(counts)

    paths = sorted(REF_CACHE.glob("*.txt"))
    if not paths:
        return []
    # even quota per reference text, so the control is a sample of the
    # eight pinned Gutenberg files rather than the first one exhausted
    quota = max(1, n_paras // len(paths))
    extra = n_paras - quota * len(paths)
    groups = []
    g = 0
    for file_no, path in enumerate(paths):
        want = quota + (1 if file_no < extra else 0)
        if want <= 0 or len(groups) >= n_paras:
            break
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = _GUT_START.sub(" ", raw)
        raw = _GUT_END.sub(" ", raw)
        sents = []
        for chunk in raw.replace("\r", " ").split("\n"):
            for s in _SENT_SPLIT_RE.split(chunk):
                s = _WS_RE.sub(" ", s).strip()
                words = s.split()
                if EN_MIN_WORDS <= len(words) <= EN_MAX_WORDS:
                    sents.append(s)
        k = 0
        taken = 0
        while k < len(sents) and taken < want:
            target = counts[g % len(counts)] or 1
            take = sents[k:k + target]
            k += target
            g += 1
            if not take:
                break
            groups.append({"text": " ".join(take), "sentences": take,
                           "n_sentences": len(take),
                           "source_file": path.name})
            taken += 1
    return groups


def english_control(records, n_paras=None, encoder=None, skip=False,
                    threads=DEFAULT_THREADS) -> tuple:
    """English paragraph matrix + its statistics, or a disclosed absence.

    The control is built the *same* way as the book: encode the sentences,
    average per paragraph, renormalise, take the cosine matrix.  Any encoder
    difference would make the comparison meaningless.
    """
    groups = english_paragraph_groups(records, n_paras=n_paras)
    info = {"n_groups": len(groups),
            "n_sentences": sum(g["n_sentences"] for g in groups),
            "available": False, "note": ""}
    if skip:
        info["note"] = "skipped on the request of the caller (--skip-english)"
        return None, info
    if not groups:
        info["note"] = "no English paragraph groups could be built"
        return None, info
    try:
        enc = encoder or ce.SentenceEncoder(key=ce.PRIMARY, device="cpu",
                                           local_only=True, threads=threads)
    except Exception as exc:                       # pragma: no cover
        info["note"] = f"encoder unavailable: {type(exc).__name__}: {exc}"
        return None, info
    flat = [s for g in groups for s in g["sentences"]]
    # encode per *sentence*, not per group: the book's vector is an average
    # of clause vectors, so the control must average sentence vectors.
    sents = flat
    del flat
    V = enc.encode(sents)
    n = V.shape[1] if V.ndim == 2 and V.shape[1] else 0
    if not n:
        info["note"] = "encoder returned no vectors"
        return None, info
    groups_vec = []
    k = 0
    for g in groups:
        take = V[k:k + g["n_sentences"]]
        k += g["n_sentences"]
        if take.shape[0] == 0:
            groups_vec.append(np.zeros(n, dtype=np.float32))
            continue
        v = take.mean(axis=0)
        nrm = float(np.linalg.norm(v))
        groups_vec.append(v / nrm if nrm else v)
    M = np.vstack(groups_vec)
    S = cosine_matrix(M)
    N = S.shape[0]
    iu = np.triu_indices(N, 1)
    flat = S[iu]
    # A repeated *sentence* inside one source text can make two adjacent
    # chunks identical by accident (Dracula's journal headers).  It is a
    # chunking artefact, not a property of English prose, so the headline
    # top-pair excludes it -- and the artefact itself is disclosed, not
    # silently repaired.
    dup = np.zeros(flat.shape, dtype=bool)
    n_identical = 0
    for pos, (gi, gj) in enumerate(zip(iu[0], iu[1])):
        if groups[gi]["text"] == groups[gj]["text"]:
            dup[pos] = True
            n_identical += 1
    top_pair_cos = float(flat.max()) if flat.size else 0.0
    keep = ~dup
    top_unique = float(flat[keep].max()) if keep.any() else top_pair_cos
    stats = {
        "mean_cos": float(flat.mean()),
        "median_cos": float(np.median(flat)),
        "frac_gt_screen": float((flat > PAIR_SCREEN).mean()),
        "lag1_mean": float(np.diag(S, 1).mean()) if N > 1 else 0.0,
        "lag2_mean": float(np.diag(S, 2).mean()) if N > 2 else 0.0,
        "lag5_mean": float(np.diag(S, 5).mean()) if N > 5 else 0.0,
        "top_pair_cos": top_pair_cos,
        "top_pair_cos_excluding_identical_chunks": top_unique,
        "n_identical_chunk_pairs": n_identical,
    }
    info["available"] = True
    info["n_identical_chunk_pairs"] = n_identical
    return stats, info


# ---------------------------------------------------------------------------
# sensitivity (view / encoder)
# ---------------------------------------------------------------------------
def sensitivity(records, store_dir=STORE_DIR) -> list:
    """Headline stats under #39's other view and its secondary encoder."""
    out = []
    models = [(ce.PRIMARY, PRIMARY_VIEW), (ce.PRIMARY, SECONDARY_VIEW),
              (SECONDARY_MODEL, PRIMARY_VIEW)]
    for model, view in models:
        try:
            st = ce.ClauseEmbeddingStore(out_dir=store_dir, model=model)
        except Exception as exc:                   # pragma: no cover
            out.append({"model": model, "view": view,
                        "note": f"unavailable: {type(exc).__name__}"})
            continue
        if view not in st._views:
            continue
        V = paragraph_vectors(st, records, view=view)
        rows = usable_rows(records)
        sub = cosine_matrix(V[rows])
        iu = np.triu_indices(sub.shape[0], 1)
        blocks = offdiag_blocks(sub, [records[i] for i in rows])
        out.append({
            "model": model,
            "view": view,
            "mean_cos": float(sub[iu].mean()),
            "frac_gt_screen": float((sub[iu] > PAIR_SCREEN).mean()),
            "n_blocks": len(blocks),
            "biggest_block_cells": blocks[0]["n_cells"] if blocks else 0,
            "biggest_block_pages": (
                f"{blocks[0]['row_pages'][0]}-{blocks[0]['row_pages'][-1]}"
                f" / {blocks[0]['col_pages'][0]}-{blocks[0]['col_pages'][-1]}"
            ) if blocks else "",
            "biggest_block_mean_cos": blocks[0]["mean_cos"] if blocks else "",
            "lag1_mean": float(np.diag(sub, 1).mean()),
        })
    return out


# ---------------------------------------------------------------------------
# headline helpers shared by the report and the tests
# ---------------------------------------------------------------------------
def matrix_stats(S: np.ndarray) -> dict:
    N = S.shape[0]
    iu = np.triu_indices(N, 1)
    flat = S[iu]
    def _mm(x):
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "sd": float(x.std()), "max": float(x.max())}
    lags = {}
    for d in (1, 2, 3, 5, 10, 25, 50, 100):
        if d < N:
            lags[d] = float(np.diag(S, d).mean())
    return {
        "n_paragraphs": N,
        "mean_cos": _mm(flat),
        "frac_gt_screen": float((flat > PAIR_SCREEN).mean()),
        "frac_gt_0_90": float((flat > 0.90).mean()),
        "frac_gt_0_95": float((flat > 0.95).mean()),
        "n_gt_0_95": int((flat > 0.95).sum()),
        "top_pair_cos": float(flat.max()),
        "lag_means": lags,
    }


def summary_of(records, V, S, page_ids, PM, lag_rows, bands, blocks,
               diags, seed=DEFAULT_SEED, english=None) -> dict:
    st = matrix_stats(S)
    lblocks = long_range_blocks(blocks)
    return {
        "schema": SCHEMA,
        "n_paragraphs": st["n_paragraphs"],
        "n_paragraphs_dropped_no_clause": len(records) - st["n_paragraphs"],
        "n_pages": len(page_ids),
        "view": PRIMARY_VIEW,
        "encoder": ce.PRIMARY,
        "paragraph_unit": "blank-line run of the #7 layout page body",
        "n_clauses_assigned": diags["n_clauses_assigned"],
        "n_clauses_crossing_para_break": diags["n_clauses_crossing_para_break"],
        "screen": PAIR_SCREEN,
        "block_threshold": BLOCK_THRESHOLD,
        "long_range_gap": LONG_RANGE_GAP,
        "mean_cos": _round(st["mean_cos"]["mean"]),
        "median_cos": _round(st["mean_cos"]["median"]),
        "max_cos": _round(st["mean_cos"]["max"]),
        "frac_pairs_gt_screen": _round(st["frac_gt_screen"], 6),
        "n_pairs_gt_0_95": st["n_gt_0_95"],
        "top_pair_cos": _round(st["top_pair_cos"]),
        "lag_means": {str(k): _round(v) for k, v in st["lag_means"].items()},
        "lag_null_deltas": {str(r["lag"]): _round(r["delta"])
                            for r in lag_rows if r["lag"] <= 30},
        "lag_z": {str(r["lag"]): _round(r["z"], 2) for r in lag_rows},
        "band_shares": {str(b["band"]): _round(b["share_in_band"], 4)
                        for b in bands},
        "band_z": {str(b["band"]): _round(b["z"], 2) for b in bands},
        "n_blocks": len(blocks),
        "n_blocks_long_range": len(lblocks),
        "biggest_long_range_paragraph_block": _block_dict(lblocks[0])
        if lblocks else None,
        "biggest_block": _block_dict(blocks[0]) if blocks else None,
        "page_matrix": {
            "n_pages": len(page_ids),
            "mean_cos": _round(PM[np.triu_indices(len(page_ids), 1)].mean()),
            "top_pairs": [
                {"pages": [page_ids[int(i)], page_ids[int(j)]],
                 "cos": _round(PM[int(i), int(j)])}
                for i, j in _top_pairs_idx(PM, 6)],
        },
        "english_control": english,
        "determinism": {"seed": seed, "n_permutations": N_PERM},
    }


def _block_dict(b):
    if not b:
        return None
    return {
        "n_cells": b["n_cells"],
        "n_rows": b["n_rows"], "n_cols": b["n_cols"],
        "row_pages": b["row_pages"], "col_pages": b["col_pages"],
        "row_para_span": [b["row_para_first"], b["row_para_last"]],
        "col_para_span": [b["col_para_first"], b["col_para_last"]],
        "gap": b["gap"],
        "mean_cos": _round(b["mean_cos"]),
        "density": _round(b["density"], 4),
        "lex_mean_cos": _round(b.get("lex_mean_cos")),
        "raw_identity": _round(b.get("raw_identity")),
    }


def _top_pairs_idx(S: np.ndarray, k: int) -> list:
    N = S.shape[0]
    ri, ci = np.triu_indices(N, 1)
    vals = S[ri, ci]
    order = np.argsort(-vals)[:k]
    return [(int(ri[o]), int(ci[o])) for o in order if vals[o] > 0]


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def make_plots(out_dir: Path, S, PM, page_ids, para_pages, lag_rows, bands,
               para_blocks=(), page_blocks=()):
    """Two figures.

    ``similarity.png`` -- the paragraph x paragraph matrix and the page
    rollup with the same colour scale, page rulers, and the long-range
    blocks boxed in.  ``lag_curve.png`` -- the lag curve against its
    permutation null, and the in-band share against its null.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_dir = Path(out_dir)

    def _box(ax, rows, cols, colour):
        ax.add_patch(Rectangle(
            (min(cols) - 0.5, min(rows) - 0.5),
            max(cols) - min(cols) + 1, max(rows) - min(rows) + 1,
            fill=False, edgecolor=colour, lw=1.4))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6), layout="constrained")
    for ax, M, ids, title in (
            (axes[0], S, None, "paragraph x paragraph"),
            (axes[1], PM, page_ids, "page x page rollup")):
        unit = "paragraph" if ids is None else "page"
        im = ax.imshow(M, cmap="magma", vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"{unit} index (left -> right = book order)")
        ax.set_ylabel(f"{unit} index")
        n = M.shape[0]
        ticks = list(range(0, n, 25))
        if ticks and ticks[-1] != n - 1:
            ticks.append(n - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(ids[t]) if ids else str(t) for t in ticks],
                           fontsize=7, rotation=90)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(ids[t]) if ids else str(t) for t in ticks],
                           fontsize=7)
        if ids is None:
            bounds = np.cumsum([0] + [para_pages.count(p)
                                     for p in sorted(set(para_pages))])[:-1]
            for b in bounds[1:]:
                ax.axhline(b, color="#8ac", lw=0.25, alpha=0.5)
                ax.axvline(b, color="#8ac", lw=0.25, alpha=0.5)
    # box the long-range blocks: green = verbatim return, red = thematic
    for b in [b for b in para_blocks if b.get("long_range")]:
        _box(axes[0], b["rows"], b["cols"],
             "#00e08a" if b.get("verbatim") else "#ff6a4d")
    pidx = {p: i for i, p in enumerate(page_ids)}
    for b in [b for b in page_blocks if b.get("long_range")]:
        _box(axes[1], [pidx[p] for p in b["row_pages"]],
             [pidx[p] for p in b["col_pages"]],
             "#00e08a" if b.get("verbatim") else "#ff6a4d")
    axes[0].set_xlim(-0.5, S.shape[1] - 0.5)
    axes[0].set_ylim(S.shape[0] - 0.5, -0.5)
    axes[1].set_xlim(-0.5, PM.shape[1] - 0.5)
    axes[1].set_ylim(PM.shape[0] - 0.5, -0.5)
    fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02, label="cosine")
    fig.suptitle("#54 self-similarity: a short diagonal band, and the "
                 "book's opening section returning at its end "
                 "(boxed; green = verbatim, orange = thematic)",
                 fontsize=12)
    fig.savefig(out_dir / "similarity.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             layout="constrained")
    lags = [r["lag"] for r in lag_rows]
    obs = [r["mean_cos"] for r in lag_rows]
    nul = [r["null_mean"] for r in lag_rows]
    sd = [r["null_sd"] for r in lag_rows]
    axes[0].plot(lags, obs, color="#b33", lw=1.8, label="observed")
    axes[0].plot(lags, nul, color="#333", lw=1.4, ls="--",
                 label="order-permutation null")
    axes[0].fill_between(lags, [m - s for m, s in zip(nul, sd)],
                         [m + s for m, s in zip(nul, sd)],
                         color="#333", alpha=0.18, linewidth=0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("lag (paragraphs apart, log scale)")
    axes[0].set_ylabel("mean cosine")
    axes[0].set_title("lag curve: a band only at short range")
    axes[0].legend(fontsize=8, loc="upper right")

    w = [b["band"] for b in bands]
    share = [b["share_in_band"] for b in bands]
    nshare = [b["null_mean"] for b in bands]
    axes[1].plot(w, share, "o-", color="#26a", lw=1.8,
                 label="observed in-band share")
    axes[1].plot(w, nshare, "s--", color="#333", lw=1.4,
                 label="null in-band share")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("band width |i-j| <= w (log scale)")
    axes[1].set_ylabel(f"share of cells > {PAIR_SCREEN}")
    axes[1].set_title("where the similar cells actually sit")
    axes[1].legend(fontsize=8, loc="center right")
    fig.savefig(out_dir / "lag_curve.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(out_dir: Path, seed: int = DEFAULT_SEED, quiet: bool = False,
          english: bool = True, english_cap: int = 0,
          encoder=None) -> dict:
    """Build the analysis into ``out_dir``; returns the summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, diags = paragraph_records(None)
    paragraph_units(records)
    store = ce.ClauseEmbeddingStore(out_dir=STORE_DIR, model=ce.PRIMARY)
    V = paragraph_vectors(store, records, view=PRIMARY_VIEW)
    rows = usable_rows(records)
    kept = [records[i] for i in rows]
    S = cosine_matrix(V[rows])
    if not quiet:
        print(f"self-similarity: {len(records)} paragraphs "
              f"({diags['n_paragraph_text']} text, "
              f"{diags['n_paragraph_heading']} headings, "
              f"{diags['n_paragraph_figure']} figure placeholders), "
              f"{S.shape[0]} with clauses", flush=True)

    page_ids, PM_v, PM = page_rollup(records, V)
    para_thr = block_threshold(S)
    page_thr = block_threshold(PM)
    page_lag_null, _ = permutation_null(PM_v, seed=seed, band_widths=())
    page_lag = lag_rows_with_null(lag_curve(PM), page_lag_null)
    lag_rows = lag_curve(S)
    lag_null, band_null = permutation_null(V[rows], seed=seed)
    lag_rows = lag_rows_with_null(lag_rows, lag_null)
    bands = band_rows(S, band_null)
    blocks = offdiag_blocks(S, kept, threshold=para_thr)
    page_blocks = offdiag_blocks(
        PM, [{"page": p, "kind": "page"} for p in page_ids],
        threshold=page_thr)
    # lexical control on the same paragraphs (no encoder at all)
    lex = lexical_cosine(records)
    lexS = lex[rows][:, rows] if len(rows) else np.zeros((0, 0))
    lex_stats = matrix_stats(lexS) if lexS.size else {}
    for b in blocks:
        _annotate_block(b, kept, S, lexS)
        b["_text_rows"] = " ".join(kept[i]["text"] for i in b["rows"])
    # page-level equivalents: pooled paragraph lexical cosines + page text
    page_text = [" ".join(r["text"] for r in kept if r["page"] == p)
                 for p in page_ids]
    lexP = _page_pool(lexS, [r["page"] for r in kept], page_ids)
    for b in page_blocks:
        _annotate_block(
            b, [{"text": page_text[i]} for i in range(len(page_ids))], PM,
            lexP)
        b["_text_rows"] = " ".join(page_text[i] for i in b["rows"])
    pairs = top_pairs(S, kept, k=60)

    en_stats, en_info = english_control(
        records, n_paras=(english_cap or None),
        encoder=encoder, skip=not english)
    english_summary = None
    if en_stats is not None:
        english_summary = {**en_stats, **en_info,
                           "n_paragraphs": en_info["n_groups"]}

    sens = sensitivity(records)
    lex_page_stats = matrix_stats(lexP) if lexP.size else {}
    summary = summary_of(records, V[rows], S, page_ids, PM, lag_rows, bands,
                         blocks, diags, seed=seed,
                         english=english_summary)

    # -- TSVs ------------------------------------------------------------
    write_tsv(out_dir / "paragraph_matrix.tsv",
              ["para_id", "page", "start", "end", "kind", "n_clauses",
               "n_units_screened", "n_words_clean", "in_similarity_matrix",
               "text"],
              [{"para_id": r["para_id"], "page": r["page"],
                "start": r["start"], "end": r["end"], "kind": r["kind"],
                "n_clauses": r["n_clauses"],
                "n_units_screened": r["n_units_screened"],
                "n_words_clean": r["n_words_clean"],
                "in_similarity_matrix": int(i in rows),
                "text": r["text"]}
               for i, r in enumerate(records)])

    write_tsv(out_dir / "paragraph_cosines.tsv",
              ["para_i", "para_j", "page_i", "page_j", "lag", "cos",
               "kind_i", "kind_j", "text_i", "text_j"],
              [{"para_i": r["para_i"], "para_j": r["para_j"],
                "page_i": r["page_i"], "page_j": r["page_j"],
                "lag": r["lag"], "cos": _round(r["cos"], 5),
                "kind_i": r["kind_i"], "kind_j": r["kind_j"],
                "text_i": r["text_i"], "text_j": r["text_j"]}
               for r in pairs])

    with (out_dir / "page_matrix.tsv").open("w", encoding="utf-8") as fh:
        fh.write("\t".join(["page"] + [str(p) for p in page_ids]) + "\n")
        for i in range(PM.shape[0]):
            fh.write("\t".join(
                [str(page_ids[i])] +
                [_f(PM[i, j], 4) for j in range(PM.shape[1])]) + "\n")

    write_tsv(out_dir / "lag_curve.tsv",
              ["lag", "n_pairs", "mean_cos", "median_cos", "max_cos",
               "frac_gt_screen", "null_mean", "null_sd", "delta", "z"],
              [{"lag": r["lag"], "n_pairs": r["n_pairs"],
                "mean_cos": _round(r["mean_cos"], 5),
                "median_cos": _round(r["median_cos"], 5),
                "max_cos": _round(r["max_cos"], 5),
                "frac_gt_screen": _round(r["frac_gt_screen"], 5),
                "null_mean": _round(r["null_mean"], 5),
                "null_sd": _round(r["null_sd"], 6),
                "delta": _round(r["delta"], 5), "z": _round(r["z"], 2)}
               for r in lag_rows])

    write_tsv(out_dir / "page_lag_curve.tsv",
              ["lag", "n_pairs", "mean_cos", "null_mean", "null_sd", "z"],
              [{"lag": r["lag"], "n_pairs": r["n_pairs"],
                "mean_cos": _round(r["mean_cos"], 5),
                "null_mean": _round(r["null_mean"], 5),
                "null_sd": _round(r["null_sd"], 6), "z": _round(r["z"], 2)}
               for r in page_lag])

    write_tsv(out_dir / "band_stats.tsv",
              ["band", "in_band_cells", "total_cells_above_screen",
               "share_in_band", "null_mean", "null_sd", "z"],
              [{"band": b["band"], "in_band_cells": b["in_band_cells"],
                "total_cells_above_screen": b["total_cells_above_screen"],
                "share_in_band": _round(b["share_in_band"], 5),
                "null_mean": _round(b["null_mean"], 5),
                "null_sd": _round(b["null_sd"], 6),
                "z": _round(b["z"], 2)}
               for b in bands])

    block_rows = []
    for level, bl in (("paragraph", blocks), ("page", page_blocks)):
        for b in bl:
            block_rows.append({
                "level": level,
                "n_cells": b["n_cells"],
                "n_rows": b["n_rows"], "n_cols": b["n_cols"],
                "row_first": b["row_para_first"], "row_last": b["row_para_last"],
                "col_first": b["col_para_first"], "col_last": b["col_para_last"],
                "gap": b["gap"],
                "long_range": int(b["long_range"]),
                "row_pages": f"{b['row_pages'][0]}-{b['row_pages'][-1]}"
                             if len(b["row_pages"]) > 1 else str(b["row_pages"][0]),
                "col_pages": f"{b['col_pages'][0]}-{b['col_pages'][-1]}"
                             if len(b["col_pages"]) > 1 else str(b["col_pages"][0]),
                "row_idx": ", ".join(str(x) for x in b["rows"][:24]),
                "col_idx": ", ".join(str(x) for x in b["cols"][:24]),
                "density": _round(b["density"], 4),
                "mean_cos": _round(b["mean_cos"], 5),
                "min_cos": _round(b["min_cos"], 5),
                "max_cos": _round(b["max_cos"], 5),
                "lex_mean_cos": _round(b.get("lex_mean_cos"), 5),
                "raw_identity": _round(b.get("raw_identity"), 4),
            })
    write_tsv(out_dir / "offdiag_blocks.tsv",
              ["level", "n_cells", "n_rows", "n_cols", "row_first", "row_last",
               "col_first", "col_last", "gap", "long_range", "row_pages",
               "col_pages", "row_idx", "col_idx", "density", "mean_cos",
               "min_cos", "max_cos", "lex_mean_cos", "raw_identity"],
              block_rows)

    lex_stats = lex_stats or {}
    write_tsv(out_dir / "sensitivity.tsv",
              ["model", "view", "note", "mean_cos", "frac_gt_screen", "lag1_mean",
               "n_blocks", "biggest_block_cells", "biggest_block_pages",
               "biggest_block_mean_cos"],
              [{"model": r["model"], "view": r["view"],
                "mean_cos": _round(r.get("mean_cos"), 5),
                "frac_gt_screen": _round(r.get("frac_gt_screen"), 5),
                "lag1_mean": _round(r.get("lag1_mean"), 5),
                "n_blocks": r.get("n_blocks", ""),
                "biggest_block_cells": r.get("biggest_block_cells", ""),
                "biggest_block_pages": r.get("biggest_block_pages", ""),
                "biggest_block_mean_cos":
                    _round(r.get("biggest_block_mean_cos"), 5),
                "note": r.get("note", "")} for r in sens])

    write_tsv(out_dir / "english_baseline.tsv",
              ["quantity", "book_paragraph", "note", "english_paragraph"],
              _english_rows(S, english_summary,
                            n_book_sentences=diags["n_clauses_assigned"]))

    write_tsv(out_dir / "reconciliation.tsv", ["quantity", "here", "earlier", "note"],
              _reconciliation(records, diags, lex_stats, S))

    write_tsv(out_dir / "diagnostics.tsv", ["metric", "value"],
              [{"metric": k, "value": v} for k, v in sorted(diags.items())] +
              [{"metric": "n_paragraphs_in_matrix", "value": S.shape[0]},
               {"metric": "n_pages_in_rollup", "value": len(page_ids)},
               {"metric": "block_threshold_paragraph",
                "value": _f(para_thr, 4)},
               {"metric": "block_threshold_page", "value": _f(page_thr, 4)},
               {"metric": "lexical_mean_cos",
                "value": _round(
                    lex_stats.get("mean_cos", {}).get("mean")
                    if lex_stats else None, 5)},
               {"metric": "lexical_top_pair_cos", "value": _round(
                   lex_stats.get("top_pair_cos") if lex_stats else None, 5)},
               {"metric": "lexical_page_mean_cos", "value": _round(
                   lex_page_stats.get("mean_cos", {}).get("mean")
                   if lex_page_stats else None, 5)},
               {"metric": "english_control_available",
                "value": int(bool(english_summary and
                                  english_summary.get("available")))}])

    with (out_dir / "self_similarity.npz").open("wb") as fh:
        pass
    np.savez(out_dir / "self_similarity.npz",
             para_ids=np.array([r["para_id"] for r in kept]),
             pages=np.array([r["page"] for r in kept]),
             page_ids=np.array(page_ids),
             emb=V[rows].astype(np.float32),
             page_emb=PM_v.astype(np.float32),
             cos=S.astype(np.float32),
             page_cos=PM.astype(np.float32),
             lex_cos=lexS.astype(np.float32) if lexS.size else
             np.zeros((0, 0), np.float32))

    write_report(out_dir / "self_similarity_report.md", summary, records,
                 kept, page_ids, lag_rows, bands, blocks, page_blocks, pairs,
                 english_summary, sens, lex_stats, lex_page_stats, S, PM,
                 page_lag, diags, seed, para_thr, page_thr)
    make_plots(out_dir, S, PM, page_ids,
               [r["page"] for r in kept], lag_rows, bands,
               para_blocks=blocks, page_blocks=page_blocks)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    if not quiet:
        print(f"wrote {out_dir / 'self_similarity_report.md'}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# reconciliation / control rows
# ---------------------------------------------------------------------------
def _reconciliation(records, diags, lex_stats, S) -> list:
    n_clauses = sum(r["n_clauses"] for r in records)
    rows = [
        {"quantity": "clauses assigned to a paragraph", "here": f"{n_clauses}",
         "earlier": "5,614 (#33/#39)",
         "note": "every clause is joined to exactly one paragraph by offset overlap"},
        {"quantity": "screened units inside paragraphs",
         "here": f"{sum(r['n_units_screened'] for r in records)}",
         "earlier": "74,077 (#27)",
         "note": "sum of #39's per-clause screened unit counts"},
        {"quantity": "pages", "here": f"{diags['n_pages']}",
         "earlier": "225 (#7)", "note": "layout page bodies"},
        {"quantity": "paragraph runs", "here": f"{diags['n_paragraph_runs']}",
         "earlier": "678 text blocks (#7)",
         "note": "blank-line runs incl. headings and figure placeholders"},
        {"quantity": "paragraphs in the similarity matrix",
         "here": f"{S.shape[0]}",
         "earlier": "",
         "note": "paragraphs with at least one #33 clause"},
    ]
    if lex_stats:
        rows.append({"quantity": "lexical (TF) mean cosine",
                     "here": _f(lex_stats.get("mean_cos", {}).get("mean"), 5),
                     "earlier": "",
                     "note": "unsupervised control, same paragraphs"})
    return rows


def _english_rows(S, english, n_book_sentences=None) -> list:
    st = matrix_stats(S)
    if english is None:
        return [{"quantity": k, "book_paragraph": _f(v, 5),
                 "english_paragraph": "", "note": "control not run"}
                for k, v in (
                    ("mean_cos", st["mean_cos"]["mean"]),
                    ("median_cos", st["mean_cos"]["median"]),
                    ("frac_gt_screen", st["frac_gt_screen"]),
                    ("frac_gt_0_95", st["frac_gt_0_95"]),
                    ("top_pair_cos", st["top_pair_cos"]),
                    ("lag1_mean", st["lag_means"].get(1, 0.0)),
                    ("lag2_mean", st["lag_means"].get(2, 0.0)),
                    ("lag5_mean", st["lag_means"].get(5, 0.0)))]
    return [
        {"quantity": "n_paragraphs", "book_paragraph": st["n_paragraphs"],
         "english_paragraph": english.get("n_paragraphs"),
         "note": "matched where the English cache allows"},
        {"quantity": "n_sentences",
         "book_paragraph": n_book_sentences or st["n_paragraphs"],
         "english_paragraph": english.get("n_sentences"),
         "note": "the book's sentences are #33 clauses"},
        {"quantity": "mean_cos", "book_paragraph": _f(st["mean_cos"]["mean"], 5),
         "english_paragraph": _f(english.get("mean_cos"), 5),
         "note": ("book > English = more self-similar than prose "
                  "(#39's verdict at clause level)")},
        {"quantity": "median_cos", "book_paragraph": _f(st["mean_cos"]["median"], 5),
         "english_paragraph": _f(english.get("median_cos"), 5), "note": ""},
        {"quantity": "frac_gt_screen",
         "book_paragraph": _f(st["frac_gt_screen"], 5),
         "english_paragraph": _f(english.get("frac_gt_screen"), 5),
         "note": f"pairs above cosine {PAIR_SCREEN}"},
        {"quantity": "frac_gt_0_95", "book_paragraph": _f(st["frac_gt_0_95"], 5),
         "english_paragraph": "not computed",
         "note": "near-duplicate paragraph pairs; English reports the 0.8 screen"},
        {"quantity": "top_pair_cos", "book_paragraph": _f(st["top_pair_cos"], 5),
         "english_paragraph": _f(
             english.get("top_pair_cos_excluding_identical_chunks"), 5),
         "note": ("most similar pair; English excludes "
                  f"{english.get('n_identical_chunk_pairs', 0)} "
                  "accidentally identical chunks (repeated source sentence)")},
        {"quantity": "lag1_mean", "book_paragraph": _f(st["lag_means"].get(1, 0.0), 5),
         "english_paragraph": _f(english.get("lag1_mean"), 5),
         "note": "consecutive paragraphs"},
        {"quantity": "lag2_mean", "book_paragraph": _f(st["lag_means"].get(2, 0.0), 5),
         "english_paragraph": _f(english.get("lag2_mean"), 5), "note": ""},
        {"quantity": "lag5_mean", "book_paragraph": _f(st["lag_means"].get(5, 0.0), 5),
         "english_paragraph": _f(english.get("lag5_mean"), 5), "note": ""},
    ]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(path, summary, records, kept, page_ids, lag_rows, bands,
                 blocks, page_blocks, pairs, english, sens, lex_stats,
                 lex_page_stats, S, PM, page_lag=(), diags=None,
                 seed=DEFAULT_SEED, para_thr=BLOCK_THRESHOLD,
                 page_thr=BLOCK_THRESHOLD):
    L = []
    A = L.append

    n = S.shape[0]
    st = matrix_stats(S)
    lr = {r["lag"]: r for r in lag_rows}
    lblocks = long_range_blocks(blocks)
    plblocks = long_range_blocks(page_blocks)
    vblocks = verbatim_blocks(blocks)
    pvblocks = verbatim_blocks(page_blocks)
    pb = plblocks[0] if plblocks else None
    vb = pvblocks[0] if pvblocks else (vblocks[0] if vblocks else None)
    n_long = len(lblocks)
    thematic_noun = _plural(
        n_long - len(vblocks), "paragraph-level blocks", "paragraph-level block")
    n_pl = len(plblocks)
    if n_pl == 0:
        page_cut = "no long-range page block clears the cut"
    elif n_pl == 1:
        page_cut = "the long-range page block is verbatim"
    else:
        page_cut = f"all {n_pl} long-range page blocks are verbatim"
    near_dup = st["n_gt_0_95"]
    null_lag = _first_null_lag(lag_rows)

    A("# Self-similarity matrix and loop detection (#54)")
    A("")
    A("Build the paragraph x paragraph cosine self-similarity matrix of the "
      "book and read it for the two structures the task names: a "
      "**diagonal band** (each paragraph close to its neighbours -- the loop "
      "you feel while reading) and **repeating blocks** (a whole stretch of "
      "paragraphs recurring later in the book -- the loop as structure).")
    A("")
    A(f"**Headline.** Both are present, at different scales, and neither is a "
      f"period. The matrix is globally high (mean off-diagonal cosine "
      f"{st['mean_cos']['mean']:.4f} -- the whole book sits in one semantic "
      f"field, so 'similar' is the default), but the similar cells are "
      f"**not** organised as a running diagonal band: of the "
      f"{bands[0]['total_cells_above_screen']:,} paragraph pairs above cosine "
      f"{PAIR_SCREEN}, only {bands[0]['share_in_band'] * 100:.1f}% sit within "
      f"one paragraph of the diagonal and {bands[3]['share_in_band'] * 100:.1f}% "
      f"within {bands[3]['band']}. The diagonal is real but **short-range**: "
      f"mean cosine at lag 1 is {st['lag_means'].get(1, 0):.4f} against an "
      f"order-permutation null of {lr[1]['null_mean']:.4f} "
      f"(z = {lr[1]['z']:+.1f}), significant to about lag 30 and at the null "
      f"from lag {null_lag}.")
    A("")
    A(f"**The loop is elsewhere, and it is block-shaped.** "
      f"{_plural(n_long, 'paragraph blocks', 'paragraph block')} and "
      f"{_plural(len(plblocks), 'page blocks', 'page block')} sit far off the "
      f"diagonal. The one that matters is a **verbatim return**: pages "
      f"{_pages_str(vb['row_pages'])} recur on pages "
      f"{_pages_str(vb['col_pages'])} -- at page resolution "
      f"{vb['n_rows']} x {vb['n_cols']} pages, mean cosine "
      f"{vb['mean_cos']:.4f}, and at paragraph resolution "
      f"{vblocks[0]['n_rows']} x {vblocks[0]['n_cols']} paragraphs "
      f"({vblocks[0]['n_cells']} cells) -- while a `difflib` ratio on the raw "
      f"text is {vb['raw_identity']:.3f} and the encoder-free lexical control "
      f"agrees the words are the same. The book's opening section returns "
      f"near its end: **the 'devastating loop of language' is, at least in "
      f"part, literally a reprint.** The rest of the long-range block "
      f"population is small thematic repetition "
      f"({len(plblocks) - len(pvblocks)} page-level and "
      f"{thematic_noun} are *not* verbatim, "
      f"4-5 cells each, cosine ~0.87 against a text identity of ~0.2): the "
      f"book reuses its motifs far apart, but it does not rewrite whole "
      f"passages at page scale.")
    A("")
    A("Read with the neighbouring tasks, the three scales separate cleanly: "
      "the short-range band is #42's lag-2-30 template recurrence one level "
      "up and #34's frames; the absence of any large-lag periodicity "
      "restates #51's verdict in semantic space; and the verbatim return is "
      "#35's page-level near-duplicate finding "
      f"({_page_pairs_str(page_ids, _top_pairs_idx(PM, 4))}) "
      "recovered here from the embedding side without any n-grams.")
    A("")

    A("## 1. Method")
    A("")
    A("* **The paragraph unit** is a blank-line-separated run of the #7 layout "
      "page body (`analysis/layout_recovery/blood-electric.layout.md`), i.e. "
      "marker's `Text` block as #7 recovered it; #7's section headings and "
      "figure placeholders are kept and flagged in `paragraph_matrix.tsv`, "
      "not dropped. Character spans are in page-body coordinates, the same "
      "coordinates #33's clauses use.")
    A(f"* **Clauses are joined to paragraphs by offset overlap** on "
      f"`(page, start, end)`, so every clause lands in exactly one paragraph "
      f"and none is lost: {diags['n_clauses_assigned']:,} of 5,614 clauses "
      f"assigned, {diags['n_clauses_crossing_para_break']} of them spanning "
      f"a paragraph break (disclosed, not repaired).")
    A("* **The paragraph vector** is the mean of its clause vectors from "
      "#39's primary artifact -- SBERT `all-MiniLM-L6-v2`, 384-d, "
      "L2-normalised, `units` view -- renormalised. No encoder runs for the "
      "book; the encoder is pinned by #39.")
    A(f"* **Statistics.** The full {n} x {n} cosine matrix "
      "(`self_similarity.npz`), a page rollup (mean paragraph vector per "
      "page, 225 x 225), a **lag curve** at "
      f"{len(lag_rows)} lags against a {N_PERM}-draw paragraph-order "
      "permutation null (seed "
      f"{seed}), **band statistics** for widths "
      f"{', '.join(str(b) for b in BAND_WIDTHS)} with the same null, and "
      "**off-diagonal blocks** = 8-connected components of cells above the "
      f"{BLOCK_QUANTILE:.1%} off-diagonal quantile of their own resolution "
      f"(cosine {para_thr:.4f} for paragraphs, {page_thr:.4f} for pages, "
      f"never below the {BLOCK_THRESHOLD} floor) with the local band "
      f"(|i-j| <= {BAND_EXCLUDE}) removed, run at both paragraph and page "
      "level. 8-connectivity is required because a block is a *diagonal* "
      "rectangle: with a 4-neighbour rule the cells (i, i+k) and (i+1, i+k+1) "
      "are not adjacent and every long-range block is shredded.")
    A("* **Controls.** (a) The **English paragraph baseline**: real English "
      "paragraphs from #24's pinned reference cache, grouped so the "
      "sentences-per-paragraph multiset matches the book's "
      "clauses-per-paragraph, encoded with the *same* MiniLM encoder and "
      "pooled the same way -- `english_baseline.tsv`. (b) A **lexical TF "
      "cosine** over #39's screened unit surfaces, no encoder at all, asking "
      "whether a block is *verbatim* or merely thematic. (c) A "
      "**view/encoder sensitivity** pass over #39's `raw` view and its "
      "secondary BGE encoder -- `sensitivity.tsv`.")
    A("* **Determinism.** Fixed seed for the permutation null and every "
      "control; no hash-order dependence; no wall-clock in any artifact.")
    A("")
    A("Reproduce with `python work/self_similarity.py`; regression suite "
      "`python work/test_self_similarity.py`. "
      f"Schema `{SCHEMA}`.")
    A("")

    A("## 2. The matrix at a glance")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| paragraphs in the matrix | {n:,} (of {len(records):,} runs; "
      f"{len(records) - n} carry no clause) |")
    A(f"| pages in the rollup | {len(page_ids)} |")
    A(f"| mean off-diagonal cosine | {st['mean_cos']['mean']:.4f} |")
    A(f"| median off-diagonal cosine | {st['mean_cos']['median']:.4f} |")
    A(f"| shares above {PAIR_SCREEN} / 0.90 / 0.95 | "
      f"{st['frac_gt_screen'] * 100:.2f}% / {st['frac_gt_0_90'] * 100:.2f}% / "
      f"{st['frac_gt_0_95'] * 100:.2f}% |")
    A(f"| pairs above cosine 0.95 | {near_dup:,} of "
      f"{n * (n - 1) // 2:,} |")
    A(f"| single most similar pair | cosine {st['top_pair_cos']:.4f} |")
    A(f"| lag 1 / 5 / 30 mean cosine | {lr[1]['mean_cos']:.4f} / "
      f"{lr[5]['mean_cos']:.4f} / {lr[30]['mean_cos']:.4f} |")
    A(f"| lag 1 null mean (z) | {lr[1]['null_mean']:.4f} ({lr[1]['z']:+.1f}) |")
    if lex_stats:
        A(f"| lexical TF mean cosine (no encoder) | "
          f"{lex_stats['mean_cos']['mean']:.4f}, top pair "
          f"{lex_stats['top_pair_cos']:.4f} |")
    A(f"| page rollup median / 95th / 99th percentile cosine | "
      f"{_page_pct(PM, 50):.4f} / {_page_pct(PM, 95):.4f} / "
      f"{_page_pct(PM, 99):.4f} |")
    A("")
    A("The mean is high by construction of the corpus, not of the method: "
      "#39 measured the same thing one level down (mean clause-to-clause "
      "cosine far above matched English), and the paragraph average only "
      "smooths towards the corpus centroid. Every structural claim below is "
      "therefore made **against the permutation null**, which keeps that "
      "background fixed and asks only where the mass sits.")
    A("")

    A("## 3. Is there a diagonal band?")
    A("")
    A("| lag | pairs | mean cosine | null | delta | z |")
    A("|---|---|---|---|---|---|")
    for r in lag_rows:
        if r["lag"] in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 50, 100, 200):
            A(f"| {r['lag']} | {r['n_pairs']:,} | {r['mean_cos']:.4f} | "
              f"{r['null_mean']:.4f} | {r['delta']:+.4f} | {r['z']:+.1f} |")
    A("")
    A(f"A band exists but it is **narrow and shallow**: it is significant "
      f"(z up to {max(r['z'] for r in lag_rows if r['lag'] <= 30):+.1f}) for "
      f"the first few lags and is gone by lag {null_lag} "
      f"(no later lag exceeds its null by two sd). In cosine terms the band adds "
      f"{lr[1]['delta']:+.4f} at lag 1 and {lr[3]['delta']:+.4f} at lag 3 "
      f"over a background of {lr[3]['null_mean']:.4f}. Reading it "
      f"structurally: a paragraph is closer to the paragraphs immediately "
      f"around it than chance allows, at the scale of a few paragraphs -- "
      f"#42's 'template recurrence at the scale of a few clauses', one "
      f"level up. There is no evidence of a band that *runs*: no tested lag "
      f"through {max(r['lag'] for r in lag_rows)} shows an excess, which is the "
      f"semantic-space restatement of #51's no-fixed-periodicity result.")
    A("")
    A("The band statistics say the same thing from the other side:")
    A("")
    A("| band width | cells in band | share of cells above screen | null share | z |")
    A("|---|---|---|---|---|")
    for b in bands:
        A(f"| {b['band']} | {b['in_band_cells']:,} | "
          f"{b['share_in_band'] * 100:.2f}% | {b['null_mean'] * 100:.2f}% | "
          f"{b['z']:+.1f} |")
    A("")
    A(f"Even at width 100 the band holds only "
      f"{bands[-1]['share_in_band'] * 100:.1f}% of the similar cells against "
      f"a {bands[-1]['null_mean'] * 100:.1f}% null: the similar cells are "
      f"spread across the book, not concentrated near the diagonal. "
      f"`lag_curve.png` plots both views.")
    A("")

    A("## 4. Repeating blocks")
    A("")
    A(f"Removing the local band and taking 8-connected components of the cells "
      f"above cosine {para_thr:.4f} (paragraph level) and {page_thr:.4f} "
      f"(page level) leaves {len(blocks)} paragraph-level blocks and "
      f"{len(page_blocks)} page-level ones. Most are "
      f"near-diagonal blobs -- the same local cohesion the lag curve "
      f"measures -- so the catalogue is split by **range**: a block whose "
      f"twin starts more than {LONG_RANGE_GAP} positions later is a "
      f"repetition, everything else is the band. "
      f"{_plural(n_long, 'paragraph blocks', 'paragraph block')} and "
      f"{_plural(len(plblocks), 'page blocks', 'page block')} are "
      f"long-range, and they split "
      f"again on the raw-text `difflib` identity into **verbatim returns** "
      f"(identity >= {VERBATIM_IDENTITY}: {len(vblocks)} paragraph, "
      f"{len(pvblocks)} page) and **template blocks** (the rest: same "
      f"template, different words).")
    A("")
    A("The verbatim long-range blocks -- the same words, far apart:")
    A("")
    A("| level | cells | rows x cols | row pages | col pages | gap | cosine "
      "| identity | first words |")
    A("|---|---|---|---|---|---|---|---|---|")
    for level, bl in (("page", pvblocks), ("paragraph", vblocks)):
        for b in bl[:8]:
            A(f"| {level} | {b['n_cells']} | {b['n_rows']} x {b['n_cols']} | "
              f"{_pages_str(b['row_pages'])} | {_pages_str(b['col_pages'])} | "
              f"{b['gap']} | {b['mean_cos']:.4f} | {b['raw_identity']:.3f} | "
              f"{_block_first_words(b, level)} |")
    A("")
    A("The largest template blocks -- the same template, different words:")
    A("")
    A("| level | cells | rows x cols | row pages | col pages | gap | cosine "
      "| lexical cosine | identity |")
    A("|---|---|---|---|---|---|---|---|---|")
    for level, bl in (("page", [b for b in plblocks if not b.get("verbatim")]),
                      ("paragraph", [b for b in lblocks
                                     if not b.get("verbatim")])):
        for b in bl[:8]:
            A(f"| {level} | {b['n_cells']} | {b['n_rows']} x {b['n_cols']} | "
              f"{_pages_str(b['row_pages'])} | {_pages_str(b['col_pages'])} | "
              f"{b['gap']} | {b['mean_cos']:.4f} | "
              f"{b['lex_mean_cos']:.4f} | {b['raw_identity']:.3f} |")
    A("")
    if vb is not None:
        topm = _top_pairs_idx(PM, 4)
        A(f"**The verbatim return is the structural loop, and it is the "
          f"load-bearing finding.** Three independent instruments agree on "
          f"what it is. The cosine itself: page pairs "
          f"{_page_pairs_str(page_ids, topm)} at "
          f"{', '.join(f'{PM[int(i), int(j)]:.4f}' for i, j in topm)} -- "
          f"#35's page-level near-duplicate finding, recovered here from "
          f"the embedding side with no n-grams anywhere. The encoder-free "
          f"lexical control: {vb['lex_mean_cos']:.4f} against "
          f"{lex_page_stats['mean_cos']['mean']:.4f} for two pages drawn at "
          f"random. And the raw text: **{vb['raw_identity'] * 100:.1f}% "
          f"identical** by `difflib`. Two adjacent mechanisms of the same "
          f"loop, separated by instrument: *sentences* recur elsewhere in "
          f"the book at long range (#35, #39), and the *section* recurs "
          f"verbatim near the end (here).")
        A("")
    tpage = [b for b in plblocks if not b.get("verbatim")]
    tpara = [b for b in lblocks if not b.get("verbatim")]
    if tpage:
        A(f"**Template blocks at page scale.** {len(tpage)} page-level runs "
          f"at cosine {tpage[0]['mean_cos']:.4f} whose text is only "
          f"{tpage[0]['raw_identity'] * 100:.1f}% identical: the same "
          f"passage *shape* rewritten with different nouns -- #34's frames "
          f"seen from the paragraph side.")
        A("")
    else:
        A(f"**No template blocks at page scale -- a negative result worth "
          f"recording.** Above the {page_thr:.4f} page-level cut, "
          f"{page_cut}. The large "
          f"diffuse long-range regions that *do* appear at a looser absolute "
          f"cut (cosine 0.85 is only this matrix's 90th percentile, against "
          f"a median of {_page_pct(PM, 50):.4f}) are background anisotropy "
          f"of the embedding space, not structure. That is why the block "
          f"screen is a **per-level quantile** and why every claim here is "
          f"stated against a null rather than an absolute cosine.")
        A("")
    if tpara:
        A(f"**Thematic near-duplicates at paragraph scale.** {len(tpara)} "
          f"paragraph blocks of 4-5 cells sit far from their twin without "
          f"being the same words: cosine "
          f"~{tpara[0]['mean_cos']:.2f}, text identity "
          f"~{tpara[0]['raw_identity']:.2f}, for example pages "
          f"{_pages_str(tpara[0]['row_pages'])} against "
          f"{_pages_str(tpara[0]['col_pages'])} (gap "
          f"{tpara[0]['gap']} paragraphs). This is the loop at the scale "
          f"this task named -- a paragraph's *shape* returning hundreds of "
          f"paragraphs later -- and it is #34's frames measured through the "
          f"embedding rather than through the token stream. The lexical "
          f"control does **not** separate these from the verbatim return "
          f"(both sit near {pb['lex_mean_cos']:.2f} page-pooled cosine): "
          f"the semantic matrix is already saturated at this resolution, "
          f"which is why this report carries an encoder-free `difflib` "
          f"identity as the discriminator between a return and a rewrite.")
        A("")
    if pairs:
        A("The ten most similar paragraph pairs, in cosine order. The row "
          "numbers are matrix rows (clause-less runs excluded); "
          "`paragraph_matrix.tsv` maps them to `para_id` and page.")
        A("")
        A("| matrix rows | pages | lag | cosine | first words |")
        A("|---|---|---|---|---|")
        for pp in pairs[:10]:
            A(f"| {pp['para_i']} / {pp['para_j']} | p{pp['page_i']} / "
              f"p{pp['page_j']} | {pp['lag']} | {pp['cos']:.4f} | "
              f"{_clip(pp['text_i'], 60)} |")
        A("")
    A(f"Structurally, the loop is **not** a diagonal band and **not** a "
      f"tiled repetition of the whole book: it is a short diagonal band, "
      f"one verbatim return of the opening section near the end, and a "
      f"handful of small thematic repeats at paragraph scale. The two "
      f"resolutions also disagree in *sharpness*: the return reads "
      f"{vb['mean_cos'] if vb else 0:.4f} at page resolution and "
      f"{vblocks[0]['mean_cos'] if vblocks else 0:.4f} at paragraph "
      f"resolution, because the rollup averages each page's repeated and "
      f"unrepeated paragraphs together while the paragraph block covers "
      f"only the paragraphs that recur.")
    A("")

    A("## 5. Page-level rollup")
    A("")
    A("Averaging each page's paragraph vectors gives the 225 x 225 matrix "
      "`page_matrix.tsv` -- the readable view, and the one that matches "
      "#35's page-resolution finding directly.")
    A("")
    A("| pages | cosine |")
    A("|---|---|")
    for i, j in _top_pairs_idx(PM, 8):
        A(f"| p{page_ids[int(i)]} / p{page_ids[int(j)]} | {PM[int(i), int(j)]:.4f} |")
    A("")
    iu = np.triu_indices(len(page_ids), 1)
    A(f"The rollup's own mean off-diagonal cosine is {PM[iu].mean():.4f} "
      f"-- *higher* than the paragraph matrix's "
      f"{st['mean_cos']['mean']:.4f}, because averaging paragraphs into "
      f"pages pools out the variation that made paragraph pairs differ. "
      f"Running the same lag analysis on the rollup "
      f"(`page_lag_curve.tsv`, same {N_PERM}-draw page-order null):")
    A("")
    A("| lag (pages apart) | pairs | mean cosine | null | z |")
    A("|---|---|---|---|---|")
    for r in page_lag:
        if r["lag"] in (1, 2, 3, 5, 10, 25, 50, 100, 200):
            A(f"| {r['lag']} | {r['n_pairs']:,} | {r['mean_cos']:.4f} | "
              f"{r['null_mean']:.4f} | {r['z']:+.1f} |")
    A("")
    if page_lag:
        p1 = next(r for r in page_lag if r["lag"] == 1)
        plast = max(r["lag"] for r in page_lag if r["z"] > 2.0)
        A(f"Pooling **keeps the local band and amplifies it in absolute "
          f"terms**: adjacent pages sit {p1['delta']:+.4f} above their own "
          f"null at lag 1 (z = {p1['z']:+.1f}), against the paragraph "
          f"matrix's {lr[1]['delta']:+.4f} -- averaging three paragraphs per "
          f"page pulls both vectors toward the shared semantic field, so the "
          f"background rises *and* the diagonal rises faster. The band is "
          f"significant to about lag {plast} at page resolution and at the "
          f"null by lag 50 -- exactly the short-range shape the paragraph "
          f"matrix showed: one phenomenon at two scales, not two "
          f"phenomena. The page rollup makes the repeated section clear at "
          f"page scale; the paragraph matrix also detects its aligned "
          f"subparagraphs.")
    A("")

    A("## 6. Controls")
    A("")
    if english:
        A("| quantity | book paragraphs | English paragraphs |")
        A("|---|---|---|")
        for r in _english_rows(S, english,
                               n_book_sentences=diags["n_clauses_assigned"]):
            if r["quantity"] in ("n_paragraphs", "n_sentences", "mean_cos",
                                 "median_cos", "frac_gt_screen",
                                 "top_pair_cos", "lag1_mean", "lag2_mean",
                                 "lag5_mean"):
                A(f"| {r['quantity']} | {r['book_paragraph']} | "
                  f"{r['english_paragraph']} |")
        A("")
        A(f"The book is more self-similar than matched English prose at "
      f"paragraph resolution exactly as #39 found at clause resolution "
      f"(mean cosine {english.get('mean_cos', 0):.4f} English vs "
      f"{st['mean_cos']['mean']:.4f} book; pairs above {PAIR_SCREEN}: "
      f"{st['frac_gt_screen'] * 100:.2f}% vs "
      f"{english.get('frac_gt_screen', 0) * 100:.3f}%). The instructive "
      f"difference is the tail. The book's most similar pair is "
      f"{st['top_pair_cos']:.4f} -- a *repeated passage*, the opening "
      f"section returning near the end -- while English's is "
      f"{english.get('top_pair_cos_excluding_identical_chunks', 0):.4f} even "
      f"after excluding "
      f"{english.get('n_identical_chunk_pairs', 0)} accidentally identical "
      f"chunk(s) (a sentence repeated in the source text, so two chunks come "
      f"out byte-identical -- disclosed rather than repaired, because that is "
      f"the only way English reaches 1.0). English's own lag-1 mean "
      f"({english.get('lag1_mean', 0):.4f}) is *below* the book's "
      f"({st['lag_means'].get(1, 0):.4f}), and the {english.get('n_paragraphs')} "
      f"English groups are drawn from the eight pinned #24 texts with the "
      f"book's sentences-per-paragraph distribution, so the two matrices are "
      f"built the same way: the excess the book shows over its own null "
      f"(z = {lr[1]['z']:+.1f} at lag 1) is a property of the book, not of "
      f"paragraph pooling.")
    else:
        A("The English paragraph control was **not run** for this build "
          "(no encoder available, or `--skip-english`); the row set is "
          "disclosed in `english_baseline.tsv` with empty control columns. "
          "The permutation null inside the book needs no encoder and every "
          "claim above stands on it.")
    A("")
    A("| view / encoder | mean cosine | lag 1 | blocks | biggest block |")
    A("|---|---|---|---|---|")
    for r in sens:
        A(f"| {r['model']} / {r['view']} | {_f(r.get('mean_cos'), 5)} | "
          f"{_f(r.get('lag1_mean'), 5)} | {r.get('n_blocks', '')} | "
          f"{r.get('biggest_block_pages', '')} "
          f"({r.get('biggest_block_cells', '')} cells) |")
    A("")
    A("The headline numbers and the block survive the change of view and of "
      "encoder, so the finding is not an artifact of the pinned model. "
      "(#39 measured the two views as 86% neighbour-agreement and the two "
      "encoders as 35%, so the *encoder* change is the sharper test here.)")
    A("")

    A("## 7. Structural reading")
    A("")
    A(f"1. **A diagonal band, but a shallow one.** Paragraphs are closer to "
      f"their immediate neighbours than the permutation null allows "
      f"(z = {lr[1]['z']:+.1f} at lag 1, decaying by lag {null_lag}). The "
      f"band is a local cohesion effect -- the same template recurrence #42 "
      f"found at clause scale and #34 catalogued as frames -- not a metrical "
      f"structure.")
    late = [r for r in lag_rows if r['lag'] > 30]
    big = [r for r in late if abs(r['z']) > 2.0]
    A(f"2. **No paragraph-scale period.** Beyond lag 30 no tested lag up to "
      f"{max(r['lag'] for r in lag_rows)} differs from its null by more than "
      f"two sd ({_plural(len(big), 'exceptions', 'exception')} out of "
      f"{len(late)}), so there is no "
      f"repeating interval at paragraph resolution: the jacket-copy 'loop' is "
      f"not a cycle. This is #51's verdict recovered in semantic space.")
    A(f"3. **The long-range structure is a block, and it is verbatim.** "
      f"{_plural(n_long, 'paragraph blocks', 'paragraph block')} and "
      f"{_plural(len(plblocks), 'page blocks', 'page block')} sit far off "
      f"the diagonal, the largest page block covering "
      f"{pb['n_rows']} x {pb['n_cols']} pages: {_pages_str(pb['row_pages'])} "
      f"against {_pages_str(pb['col_pages'])} at mean cosine "
      f"{pb['mean_cos']:.4f}. The book's opening section returns near its "
      f"end, and the lexical control plus the `difflib` ratio show it is the "
      f"same *text*, not the same theme.")
    A(f"4. **The background is one field, and that is itself the finding.** "
      f"Every paragraph is similar to every other (median "
      f"{st['mean_cos']['median']:.4f}), so the matrix cannot be read as "
      f"clusters; the structure is in the *excess over that background* -- "
      f"which is exactly what the permutation null isolates, and exactly why "
      f"this report quotes deltas and z-scores rather than raw cosines.")
    A("5. **What the task asked, answered directly:** the 'loop' shows as "
      "**repeating blocks at long range on top of a short diagonal band**, "
      "not as a band alone and not as a cycle. In one sentence: the loop is "
      "local cohesion plus one verbatim return.")
    A("")
    A("## 8. Caveats")
    A("")
    A(f"* {diags['n_paragraphs_without_clauses']} of the {len(records)} "
      f"paragraph records ({diags['n_paragraph_runs']} blank-line runs plus "
      f"{diags['n_paragraphs_added_for_coverage']} added so that no clause is "
      f"dropped) carry no #33 clause -- figure placeholders and bare page "
      f"furniture -- and are excluded from the matrix; they are listed in "
      f"`paragraph_matrix.tsv` with `in_similarity_matrix = 0`.")
    A(f"* {diags['n_clauses_crossing_para_break']} clauses straddle a "
      f"paragraph break and are assigned by maximum overlap; the overlap is "
      f"large in every case, so the assignment is not a coin flip, but the "
      f"boundary is soft.")
    A("* The paragraph vector is an average of clause vectors, so a "
      "paragraph's position in the matrix depends on #33's segmentation as "
      "much as on its content. A different segmentation would shift "
      "individual cells, not the two structural results.")
    A("* Cosine under MiniLM is anisotropic: all pairs sit well above zero "
      "(see the English control), which is why every claim here is made "
      "against the null and never against an absolute threshold.")
    A("")
    path.write_text("\n".join(L).rstrip() + "\n", encoding="utf-8")


def _pages_str(pages):
    if len(pages) <= 4:
        return "/".join(str(p) for p in pages)
    return f"{pages[0]}-{pages[-1]} ({len(pages)} pages)"


def _page_pairs_str(page_ids, pairs) -> str:
    """``p4=p220, p5=p221, ...`` for a list of matrix-index pairs."""
    return ", ".join(
        f"p{page_ids[int(i)]}=p{page_ids[int(j)]}" for i, j in pairs)


def _page_pct(PM, q):
    iu = np.triu_indices(PM.shape[0], 1)
    return float(np.percentile(PM[iu], q))


def _block_first_words(block, level, n=34):
    """First words of the block's row half (page number + text snippet)."""
    pages = block["row_pages"]
    span = (str(pages[0]) if len(pages) == 1
            else f"{pages[0]}-{pages[-1]}")
    txt = block.get("_text_rows", "")
    return f"[{level} {span}] {_clip(txt, n)}"


def _plural(n, many, one):
    return f"{n} {many}" if n != 1 else f"{n} {one}"


def _clip(text, n):
    s = " ".join(text.split())
    return s[:n] + ("..." if len(s) > n else "")


def _first_null_lag(lag_rows):
    """First lag at or above which every later lag is within two null sd."""
    zs = [(r["lag"], r["z"]) for r in sorted(lag_rows, key=lambda r: r["lag"])]
    for k in range(len(zs)):
        if all(z <= 2.0 for _, z in zs[k:]):
            return zs[k][0]
    return zs[-1][0]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--skip-english", action="store_true",
                    help="do not run the English paragraph control")
    ap.add_argument("--english-cap", type=int, default=0,
                    help="cap the number of English paragraphs (0 = all)")
    args = ap.parse_args(argv)
    build(Path(args.out), seed=args.seed, quiet=args.quiet,
          english=not args.skip_english, english_cap=args.english_cap)


if __name__ == "__main__":
    main()
