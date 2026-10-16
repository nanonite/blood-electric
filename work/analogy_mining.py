#!/usr/bin/env python3
"""#45 Mine analogical equations among recurring book entities.

The search is deliberately restricted to #30's accepted single-unit entity
inventory and #28's three book-only embedding models.  It compares pair
offsets, ``b - a`` and ``d - c``, and ranks equations by cross-model
agreement.  The output is a shortlist for human reading, not a claim that an
embedding relation defines the entities' meanings.

Reproduce with::

    python work/analogy_mining.py
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_models as em  # noqa: E402

ENTITY_CANDIDATES = ROOT / "analysis" / "entity" / "entity_candidates.tsv"
KWIC = ROOT / "analysis" / "kwic" / "term_summaries.tsv"
EMBEDDINGS = ROOT / "analysis" / "embeddings"
OUT = ROOT / "analysis" / "analogy_mining"
MIN_OCCURRENCES = 8
MAX_ROWS = 100
MODELS = (
    ("w2v_book", "word2vec", EMBEDDINGS / "w2v_book.kv"),
    ("ft_book", "fastText", EMBEDDINGS / "ft_book.kv"),
    ("glove_book", "GloVe", EMBEDDINGS / "glove_book.npz"),
)


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_models():
    models = {}
    for name, algo, path in MODELS:
        if not path.exists():
            raise SystemExit(f"Missing #28 model artifact: {path}")
        if path.suffix == ".kv":
            wv = em.load_kv(path)
            models[name] = em.make_vec_model(
                name, "book", algo, wv, subwords=(name == "ft_book"))
        else:
            keys, matrix = em.load_glove(path)
            models[name] = em.VecModel(name, "book", algo, keys, matrix)
    return models


def candidate_terms(models):
    """Join accepted #30 unit entities to all three #28 book vocabularies."""
    vocab = {r["lower"]: r for r in read_tsv(EMBEDDINGS / "vocabulary.tsv")}
    accepted = []
    for r in read_tsv(ENTITY_CANDIDATES):
        term = r["candidate"].lower()
        n = int(r["n"])
        if r["accepted"] != "y" or n < MIN_OCCURRENCES or term not in vocab:
            continue
        if not all(model.has(term) for model in models.values()):
            continue
        accepted.append({
            "term": term,
            "surface": r["candidate"],
            "n": n,
            "n_pages": int(r["n_pages"]),
            "first_page": int(r["first_page"]),
            "role": r["role"],
        })
    accepted.sort(key=lambda x: x["term"])
    if len(accepted) < 4:
        raise SystemExit(f"Only {len(accepted)} entities meet the cutoff")
    return accepted


def kwic_examples():
    examples = {}
    for r in read_tsv(KWIC):
        term = r["term_lower"]
        examples[term] = {
            "page": r["first_page"],
            "example": r.get("example_1", ""),
        }
    return examples


def generate_equations(terms, models):
    """Build 3 pair partitions per four-entity set, then orient by consensus."""
    model_vectors = {}
    for name, model in models.items():
        model_vectors[name] = np.asarray([model.vector(t["term"]) for t in terms],
                                         dtype=np.float64)
    n = len(terms)
    pair_ids = {}
    pairs = list(itertools.combinations(range(n), 2))
    for pi, pair in enumerate(pairs):
        pair_ids[pair] = pi

    # One row for every four-entity set and its three possible pairings.
    eq_ids = []
    for i, j, k, l in itertools.combinations(range(n), 4):
        eq_ids.extend(((i, j, k, l), (i, k, j, l), (i, l, j, k)))
    eq_ids = np.asarray(eq_ids, dtype=np.int32)
    left_pair = np.fromiter((pair_ids[(int(a), int(b))]
                             for a, b in eq_ids[:, :2]), dtype=np.int32)
    right_pair = np.fromiter((pair_ids[(int(c), int(d))]
                              for c, d in eq_ids[:, 2:]), dtype=np.int32)

    raw_by_model = {}
    for name, vectors in model_vectors.items():
        offsets = np.asarray([vectors[b] - vectors[a] for a, b in pairs])
        norms = np.linalg.norm(offsets, axis=1)
        norms[norms == 0] = 1.0
        offsets /= norms[:, None]
        pair_cosines = offsets @ offsets.T
        raw_by_model[name] = pair_cosines[left_pair, right_pair]

    raw_matrix = np.column_stack([raw_by_model[name] for name in models])
    # Choose a single orientation by the median sign across models. Reversing
    # the second pair flips the sign while preserving the equation's relation.
    flips = np.where(np.median(raw_matrix, axis=1) < 0, -1.0, 1.0)
    oriented = raw_matrix * flips[:, None]
    percentiles = np.empty_like(oriented)
    for col in range(oriented.shape[1]):
        ordered = np.sort(oriented[:, col])
        percentiles[:, col] = np.searchsorted(
            ordered, oriented[:, col], side="right") / len(ordered)

    median_percentile = np.median(percentiles, axis=1)
    min_percentile = np.min(percentiles, axis=1)
    mean_cosine = np.mean(oriented, axis=1)
    sign_agreement = np.sum(oriented >= 0, axis=1)
    # If consensus chose the negative orientation, reverse c:d in the output.
    eq_ids[flips < 0, 2], eq_ids[flips < 0, 3] = (
        eq_ids[flips < 0, 3].copy(), eq_ids[flips < 0, 2].copy())

    order = np.lexsort((np.arange(len(eq_ids)), -mean_cosine,
                        -min_percentile, -median_percentile))
    ranked = []
    for ix in order[:MAX_ROWS]:
        ranked.append({
            "ids": tuple(int(i) for i in eq_ids[ix]),
            "cos": {name: float(oriented[ix, col])
                    for col, name in enumerate(models)},
            "percentile": {name: float(percentiles[ix, col])
                           for col, name in enumerate(models)},
            "median_percentile": float(median_percentile[ix]),
            "min_percentile": float(min_percentile[ix]),
            "mean_cosine": float(mean_cosine[ix]),
            "sign_agreement": int(sign_agreement[ix]),
        })
    return ranked


def full_vocab_rank(model, a, b, c, expected):
    """Exact analogy completion rank against the model's full vocabulary."""
    va, vb, vc = (model.vector(t) for t in (a, b, c))
    q = vb - va + vc
    qnorm = float(np.linalg.norm(q))
    if qnorm == 0:
        return "", ""
    sims = (model.matrix @ q) / (model.norms * qnorm)
    target_index = model.key_to_index.get(expected)
    if target_index is None:
        return "", ""
    target_score = float(sims[target_index])
    excluded = {a, b, c}
    better = sum(1 for idx, term in enumerate(model.keys)
                 if term not in excluded and float(sims[idx]) > target_score)
    return str(better + 1), f"{target_score:.4f}"


def clean_cell(value):
    return " ".join(str(value).replace("\t", " ").replace("\n", " ").split())


def main():
    models = load_models()
    terms = candidate_terms(models)
    evidence = kwic_examples()
    candidates = generate_equations(terms, models)[:MAX_ROWS]
    by_term = {t["term"]: t for t in terms}
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        a, b, c, d = (by_term[terms[i]["term"]] for i in candidate["ids"])
        row = {
            "rank": rank,
            "equation": f"{a['surface']} : {b['surface']} :: "
                        f"{c['surface']} : {d['surface']}",
            "a": a["surface"], "b": b["surface"],
            "c": c["surface"], "d": d["surface"],
            "consensus_percentile": f"{candidate['median_percentile']:.4f}",
            "weakest_model_percentile": f"{candidate['min_percentile']:.4f}",
            "mean_offset_cosine": f"{candidate['mean_cosine']:.4f}",
            "model_sign_agreement": f"{candidate['sign_agreement']}/3",
        }
        for label, ent in zip("abcd", (a, b, c, d)):
            ex = evidence.get(ent["term"], {})
            row[f"{label}_n"] = ent["n"]
            row[f"{label}_pages"] = ent["n_pages"]
            row[f"{label}_role"] = ent["role"]
            row[f"{label}_evidence"] = clean_cell(ex.get("example", ""))
            row[f"{label}_evidence_page"] = ex.get("page", "")
        for name, model in models.items():
            r, s = full_vocab_rank(model, a["term"], b["term"],
                                   c["term"], d["term"])
            row[f"{name}_completion_rank"] = r
            row[f"{name}_completion_cosine"] = s
            row[f"{name}_offset_cosine"] = f"{candidate['cos'][name]:.4f}"
            row[f"{name}_percentile"] = f"{candidate['percentile'][name]:.4f}"
        row["completion_models_top10"] = sum(
            int(row[f"{name}_completion_rank"]) <= 10 for name in models) if all(
                row[f"{name}_completion_rank"] for name in models) else 0
        rows.append(row)

    header = ["rank", "equation", "a", "b", "c", "d",
              "consensus_percentile", "weakest_model_percentile",
              "mean_offset_cosine", "model_sign_agreement"]
    for label in "abcd":
        header.extend([f"{label}_n", f"{label}_pages", f"{label}_role",
                       f"{label}_evidence_page", f"{label}_evidence"])
    for name in models:
        header.extend([f"{name}_offset_cosine", f"{name}_percentile",
                       f"{name}_completion_rank", f"{name}_completion_cosine"])
    header.append("completion_models_top10")
    write_tsv(OUT / "equations.tsv", header, rows)

    top10_all = sum(r["completion_models_top10"] == 3 for r in rows)
    top10_two = sum(r["completion_models_top10"] >= 2 for r in rows)
    best_worst_rank = min(max(int(r[f"{name}_completion_rank"])
                              for name in models) for r in rows)
    best_median_rank = min(statistics.median(
        int(r[f"{name}_completion_rank"]) for name in models) for r in rows)
    summary = {
        "schema": "analogy-mining/1",
        "source": "#30 accepted single-unit entities + #28 book-only embeddings",
        "entity_recurrence_floor": MIN_OCCURRENCES,
        "candidate_entities": len(terms),
        "four_entity_sets": math.comb(len(terms), 4),
        "pairings_scored": math.comb(len(terms), 4) * 3,
        "models": list(models),
        "ranked_equations_written": len(rows),
        "top10_completions_all_models": top10_all,
        "top10_completions_at_least_two_models": top10_two,
        "best_worst_model_full_vocabulary_rank": best_worst_rank,
        "best_median_full_vocabulary_rank": best_median_rank,
        "ranking": "median within-model empirical percentile of oriented pair-offset cosine; ties by weakest-model percentile and mean cosine",
        "meaning": "distributional candidates for human interpretation, not semantic definitions",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    report = [
        "# Analogy and metaphor mining (#45)", "",
        "## Result", "",
        f"Scored {summary['pairings_scored']:,} four-entity pairings over "
        f"{len(terms)} recurring entities from #30. The inventory is restricted "
        f"to accepted single-unit entities occurring at least {MIN_OCCURRENCES} "
        "times and present in all three #28 book-only models. Multiword glossary "
        "entities are excluded because averaging their component vectors would "
        "build the shared word into the result and produce trivial compositional "
        "offsets. The first 100 ranked candidates are in `equations.tsv`.", "",
        "The ranking uses the median within-model percentile of cosine similarity "
        "between the two entity-pair offsets (`b - a` and `d - c`). A shared "
        "orientation is chosen by majority agreement across word2vec, fastText, "
        "and GloVe. The table also reports each model's offset cosine and the "
        "exact rank of `d` for `b - a + c` among the model's full vocabulary. "
        "Higher ranks and percentiles are descriptive; they are not significance "
        "tests.", "",
        "## Finding", "",
        f"The top-100 offset matches do not produce a stable analogy: "
        f"{top10_all}/100 predict `d` in the full-vocabulary top 10 in all three "
        f"models, and {top10_two}/100 do so in at least two. The best candidate's "
        f"worst model rank is {best_worst_rank} (best median rank {best_median_rank}). "
        "Thus the strongest offset similarities are not enough to establish a "
        "viable cross-model equation. The table is a ranked set of leads, with a "
        "negative result on stable completion.", "",
        "## Leading candidates", "",
        "Completion ranks are against each model's full vocabulary after excluding "
        "the three input terms; lower is better.", "",
        "| Rank | Candidate equation | Offset percentile | word2vec rank | fastText rank | GloVe rank |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows[:10]:
        report.append(
            f"| {row['rank']} | `{row['equation']}` | "
            f"{float(row['consensus_percentile']):.4f} | "
            f"{row['w2v_book_completion_rank']} | "
            f"{row['ft_book_completion_rank']} | "
            f"{row['glove_book_completion_rank']} |")
    report.extend([
        "",
        "## Reading the candidates", "",
        "Each row includes the entity-induction role, occurrence/page counts, and "
        "one KWIC context for each term. These contexts attest the terms; they "
        "are not direct evidence that the text asserts the relation between "
        "them. Read them before interpreting an equation. Entity role and vector "
        "proximity do not by themselves establish a metaphor or a translation. "
        "The entity inventory contains "
        "acronyms and coined labels as well as named characters, so apparent "
        "letter-pattern relations can be artifacts of spelling or shared "
        "contexts.", "",
        "All three models use only the book's screened unit stream; no external "
        "reference corpus is used. The models share a corpus and are not three "
        "independent textual sources. The results therefore form a ranked "
        "shortlist for close reading, not an automatic semantic glossary.", "",
        "Reproduce with `python work/analogy_mining.py`.",
    ])
    (OUT / "analogy_mining_report.md").write_text("\n".join(report) + "\n",
                                                   encoding="utf-8")
    print(f"Wrote {len(rows)} equations for {len(terms)} entities to {OUT}")


if __name__ == "__main__":
    main()
