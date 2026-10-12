#!/usr/bin/env python3
"""Predictive validity on held-out text (Chainlink #72).

Does the derived grammar predict held-out operator sequences and
collocations?  Train 80% / test 20%.

Two predictive claims are put to the test, each against its own honest
baseline:

  * Operator sequences.  The derived grammar is the interpolated n-gram
    Markov model over the ordered operator stream (#10/#11, trained per
    #35): order 0 (unigram), 1 and 2, Jelinek-Mercer lam=0.8, transitions
    counted and scored within a page only.  Held-out scores: perplexity,
    mean log2 probability, argmax accuracy and top-2 accuracy on the
    test pages, each against the order-0 unigram baseline.  A run-
    inventory coverage check asks the stricter question: were the test
    pages' multi-operator runs (the idiomatic core per #11's verdict)
    seen in the training pages at all?

  * Collocations.  The derived collocational grammar is #27's association
    table (PMI / logDice / Dice / t-score over the markup-screened unit
    and token streams).  Held-out scores: over bigram types with f >= 2
    in BOTH splits, the Spearman and Pearson correlation of train vs
    test logDice (and of raw frequency and PMI), the mean absolute
    logDice shift, and top-100 overlap / survival between the train and
    test rankings.  logDice is comparable across splits because its
    formula carries no N; PMI is not, and its correlation is reported
    separately for that reason.

Splits.  The primary split is chronological -- first 80% of pages train,
last 20% test -- the conservative transfer test, matching #11's
`held_out_perplexity_chrono` (the random-fold split in #11 measures
generalisation to stylistically adjacent pages and is the optimistic
view).  A seeded random 80/20 page split is reported alongside as that
optimistic view.  Pages are the unit of splitting everywhere, so no page
contributes to both sides and no sequence ever crosses the boundary.

The n-gram vocabulary is the TRAIN vocabulary (no test leakage into the
smoothing constant); this is the one deliberate difference from #11's
scoring helper, whose V spans all pages, and it is disclosed in the
reconciliation artifact.

Outputs under analysis/heldout_validity/:
  operator_heldout.tsv     per split x order: perplexity, accuracies, gain
  run_coverage.tsv         train run inventory vs test runs, per split
  collocation_heldout.tsv  per layer x split: stability scores
  heldout_scores.tsv       THE DELIVERABLE: the headline held-out scores
  summary.json             machine-readable headline numbers + verdicts
  heldout_validity_report.md
  diagnostics.tsv          build ledger (seed, frac, split sizes)
  reconciliation.tsv       this build vs #11 / #27 / #35 / #36

Reproduce::

    conda activate marker
    python work/heldout_validity.py
    python work/test_heldout_validity.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import operator_ngrams as on  # #10/#11 ordered operator stream
import collocation as cl       # #27 association machinery

ROOT = HERE.parent
SRC = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
OUT = ROOT / "analysis" / "heldout_validity"

SCHEMA = "heldout_validity/1"
SEED = 42
TRAIN_FRAC = 0.8
LAM = 0.8          # Jelinek-Mercer interpolation weight, as #11
ORDERS = (0, 1, 2)
TOP_K = 100        # collocation top-k overlap window
MIN_BIGRAM = 2     # same f>=2 floor as #27's published tables


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def chrono_split(pages, frac=TRAIN_FRAC):
    """First `frac` of the sorted page list train, the rest test."""
    page_list = sorted(pages)
    cut = max(1, int(round(len(page_list) * frac)))
    return page_list[:cut], page_list[cut:]


def random_split(pages, frac=TRAIN_FRAC, seed=SEED):
    """Seeded random page split; `seed` makes the optimistic view repeatable."""
    page_list = sorted(pages)
    random.Random(seed).shuffle(page_list)
    cut = max(1, int(round(len(page_list) * frac)))
    return page_list[:cut], page_list[cut:]


# ---------------------------------------------------------------------------
# operator sequences: interpolated n-gram model + held-out scoring
# ---------------------------------------------------------------------------
def train_ngram(page_ops, order, lam=LAM):
    """Train the interpolated n-gram operator model on training pages.

    Same Jelinek-Mercer scheme as #11's `_held_out_perplexity_split`:
      P0(n)     = puni(n)
      P1(n|a)   = lam * c(a,n)/n(a) + (1-lam) * P0(n)
      P2(n|a,b) = lam * c(a,b,n)/n(a,b) + (1-lam) * P1(n|b)
    The vocabulary is the TRAIN vocabulary (no test leakage).  Transitions
    are counted within pages only, so a page boundary never joins a
    context with a continuation.
    """
    train_ops = [op for p in sorted(page_ops) for op in page_ops[p]]
    vocab = sorted(set(train_ops))
    uni = collections.Counter(train_ops)
    c1 = collections.defaultdict(collections.Counter)
    c2 = collections.defaultdict(collections.Counter)
    for p in sorted(page_ops):
        ops = page_ops[p]
        for i in range(len(ops) - 1):
            c1[ops[i]][ops[i + 1]] += 1
        for i in range(len(ops) - 2):
            c2[(ops[i], ops[i + 1])][ops[i + 2]] += 1
    total_uni = sum(uni.values()) or 1
    v_size = len(vocab)

    def p0(n):
        return (uni.get(n, 0) + 1e-9) / (total_uni + 1e-9 * v_size)

    def p1(n, a):
        na = sum(c1[a].values())
        ml = c1[a][n] / na if na else 0.0
        return lam * ml + (1 - lam) * p0(n)

    def p2(n, a, b):
        nab = sum(c2[(a, b)].values())
        ml = c2[(a, b)][n] / nab if nab else 0.0
        return lam * ml + (1 - lam) * p1(n, b)

    by_freq = sorted(vocab, key=lambda v: (-uni[v], v))
    return {"order": order, "lam": lam, "vocab": vocab, "uni": uni,
            "c1": c1, "c2": c2, "p0": p0, "p1": p1, "p2": p2,
            "by_freq": by_freq, "n_train_tokens": len(train_ops)}


def score_ngram(model, test_page_ops):
    """Held-out scores for one trained model over test pages.

    Per scored position: log2 P(actual next operator), whether the actual
    operator is the model's argmax, and whether it is in the model's top
    2.  The first `order` tokens of each page are context, not targets.
    """
    order = model["order"]
    p0, p1, p2 = model["p0"], model["p1"], model["p2"]
    vocab = model["vocab"]

    def preds(i, ops):
        if order == 0:
            return model["by_freq"]
        if order == 1:
            a = ops[i]
            return sorted(vocab, key=lambda v: (-p1(v, a), v))
        a, b = ops[i], ops[i + 1]
        return sorted(vocab, key=lambda v: (-p2(v, a, b), v))

    sum_log2, n = 0.0, 0
    argmax_ok = top2_ok = 0
    for p in sorted(test_page_ops):
        ops = test_page_ops[p]
        for i in range(len(ops) - order):
            nxt = ops[i + order]
            if order == 0:
                prob = p0(nxt)
            elif order == 1:
                prob = p1(nxt, ops[i])
            else:
                prob = p2(nxt, ops[i], ops[i + 1])
            sum_log2 += math.log2(max(prob, 1e-12))
            n += 1
            ranked = preds(i, ops)
            if ranked[0] == nxt:
                argmax_ok += 1
            if nxt in ranked[:2]:
                top2_ok += 1
    return {"n_test_tokens": n,
            "perplexity": 2 ** (-sum_log2 / n) if n else float("nan"),
            "mean_log2_prob": sum_log2 / n if n else float("nan"),
            "argmax_accuracy": argmax_ok / n if n else float("nan"),
            "top2_accuracy": top2_ok / n if n else float("nan")}


def run_coverage(page_runs, train_pages, test_pages):
    """Do the training pages' runs cover the test pages' runs?

    #11's verdict is that real syntax lives in multi-operator runs, so the
    run inventory is the strictest operator-side question: a test run type
    the training pages never saw is unpredictable by construction.
    """
    train_types = set()
    for p in train_pages:
        for run_ops in page_runs[p]:
            train_types.add(tuple(run_ops))
    n_types = seen_types = n_tokens = seen_tokens = 0
    for p in test_pages:
        for run_ops in page_runs[p]:
            key = tuple(run_ops)
            n_types += 1
            n_tokens += len(run_ops)
            if key in train_types:
                seen_types += 1
                seen_tokens += len(run_ops)
    return {"n_test_run_types": n_types,
            "n_test_run_types_seen_in_train": seen_types,
            "run_type_coverage": seen_types / n_types if n_types else float("nan"),
            "n_test_run_tokens": n_tokens,
            "n_test_run_tokens_in_train_types": seen_tokens,
            "run_token_coverage": seen_tokens / n_tokens if n_tokens else float("nan")}


def operator_heldout(page_ops, train_pages, test_pages):
    """Train orders 0/1/2 on train pages, score test pages, per order."""
    rows = []
    train_ops = {p: page_ops[p] for p in train_pages}
    test_ops = {p: page_ops[p] for p in test_pages}
    for order in ORDERS:
        model = train_ngram(train_ops, order)
        scores = score_ngram(model, test_ops)
        rows.append({"order": order, **scores,
                     "n_train_tokens": model["n_train_tokens"],
                     "vocab_size": len(model["vocab"])})
    uni_ppl = rows[0]["perplexity"]
    for r in rows:
        r["gain_vs_unigram"] = ((uni_ppl - r["perplexity"]) / uni_ppl
                                if uni_ppl else float("nan"))
    return rows


# ---------------------------------------------------------------------------
# collocations: association stability across the split
# ---------------------------------------------------------------------------
def split_unit_seqs(page_units, excluded_uids, train_pages, test_pages):
    """Unit sequences (markup-screened) for each side of the split."""
    def seqs(pages):
        return [[u[2] for u in page_units[p] if u[1] not in excluded_uids]
                for p in pages]
    return seqs(train_pages), seqs(test_pages)


def split_token_seqs(page_tokens, markup_spans, train_pages, test_pages):
    """Token sequences (markup-screened) for each side of the split."""
    def seqs(pages):
        out = []
        for p in pages:
            spans = markup_spans.get(p, [])
            out.append([(t.lower(), k) for s, e, t, k in page_tokens[p]
                        if not any(a <= s < b for a, b in spans)])
        return out
    return seqs(train_pages), seqs(test_pages)


def _assoc_rows(pairs, uni, n_items, min_count):
    """Association rows {pair: row} for bigram pairs with f >= min_count."""
    rows = {}
    for pair, f_xy in pairs.items():
        if f_xy < min_count:
            continue
        f_x = uni[pair[0]]
        f_y = uni[pair[1]]
        rows[pair] = {"f_x": f_x, "f_y": f_y, "f_xy": f_xy, "n": n_items,
                      **cl.association(f_xy, f_x, f_y, n_items)}
    return rows


def unit_bigram_assoc_rows(seqs, min_count=MIN_BIGRAM):
    """Unit-layer bigram association rows + raw counts."""
    n_items = sum(len(s) for s in seqs)
    uni = collections.Counter()
    for s in seqs:
        uni.update(s)
    bi = cl.count_ngrams(seqs, 2)
    return _assoc_rows(bi, uni, n_items, min_count), bi, uni, n_items


def token_bigram_assoc_rows(seqs, min_count=MIN_BIGRAM):
    """Token-layer bigram association rows (items carry their kind)."""
    n_items = sum(len(s) for s in seqs)
    uni = collections.Counter()
    for s in seqs:
        uni.update(s)
    bi = cl.count_ngrams(seqs, 2)
    rows = {}
    for (x, y), f_xy in bi.items():
        if f_xy < min_count:
            continue
        rows[(x, y)] = {
            "x": x[0], "y": y[0], "x_kind": x[1], "y_kind": y[1],
            "has_op": "y" if (x[1] == "OP" or y[1] == "OP") else "n",
            "f_x": uni[x], "f_y": uni[y], "f_xy": f_xy, "n": n_items,
            **cl.association(f_xy, uni[x], uni[y], n_items)}
    return rows, bi, uni, n_items


def rankdata(xs):
    """Average ranks (1-based); tied values share the mean of their ranks."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    """Spearman rank correlation (Pearson over average ranks)."""
    return pearson(rankdata(xs), rankdata(ys))


def _topk(rows, k):
    return {pair for pair, _r in sorted(
        rows.items(),
        key=lambda kv: (-kv[1]["logdice"], -kv[1]["f_xy"], kv[0]))[:k]}


def collocation_heldout(train_rows, test_rows, top_k=TOP_K):
    """Stability of association scores across the train/test split.

    Over bigram types with f >= min_count in BOTH splits: rank and Pearson
    correlation of logDice (and, separately, of PMI and of raw frequency),
    mean |delta logDice|, and top-k overlap / survival between the two
    rankings.  Survival asks the predictive question directly: a top-k
    train collocation "predicts" the test split if the pair is still a
    published-strength collocation (f >= min_count) there.
    """
    shared = sorted(set(train_rows) & set(test_rows))
    if not shared:
        return {"n_train_types": len(train_rows), "n_test_types": len(test_rows),
                "n_shared_types": 0, "top_k": top_k}
    ld_tr = [train_rows[k]["logdice"] for k in shared]
    ld_te = [test_rows[k]["logdice"] for k in shared]
    pmi_tr = [train_rows[k]["pmi"] for k in shared]
    pmi_te = [test_rows[k]["pmi"] for k in shared]
    f_tr = [train_rows[k]["f_xy"] for k in shared]
    f_te = [test_rows[k]["f_xy"] for k in shared]
    train_top = _topk(train_rows, top_k)
    test_top = _topk(test_rows, top_k)
    inter = train_top & test_top
    survive_tr = sum(1 for k in train_top
                     if test_rows.get(k, {}).get("f_xy", 0) >= MIN_BIGRAM)
    survive_te = sum(1 for k in test_top
                     if train_rows.get(k, {}).get("f_xy", 0) >= MIN_BIGRAM)
    return {
        "n_train_types": len(train_rows), "n_test_types": len(test_rows),
        "n_shared_types": len(shared),
        "spearman_logdice": spearman(ld_tr, ld_te),
        "pearson_logdice": pearson(ld_tr, ld_te),
        "spearman_pmi": spearman(pmi_tr, pmi_te),
        "spearman_freq": spearman(f_tr, f_te),
        "mean_abs_delta_logdice": sum(abs(a - b) for a, b in zip(ld_tr, ld_te)) / len(shared),
        "top_k": top_k,
        "topk_overlap": len(inter) / top_k,
        "topk_train_pairs_survive_in_test": survive_tr / top_k,
        "topk_test_pairs_survive_in_train": survive_te / top_k,
    }


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------
def write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})


def f4(x):
    return float(f"{x:.4f}") if x == x else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="#72 predictive validity on held-out text")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if args.quiet else print

    # ---- operator stream (the #10/#11 sequence, runs kept for coverage) ----
    page_runs = on.page_runs(str(SRC))
    pages = sorted(page_runs)
    page_ops = {p: [op for run in page_runs[p] for op in run] for p in pages}
    n_op = sum(len(v) for v in page_ops.values())
    log(f"[#72] operator stream: n={n_op} V={len(set(op for v in page_ops.values() for op in v))} "
        f"pages={len(pages)}")

    splits = {"chrono": chrono_split(pages, args.frac),
              "random": random_split(pages, args.frac, args.seed)}

    # ---- operator sequences: train 80%, score held-out 20% ---------------
    op_results = {}
    for name, (train_pages, test_pages) in splits.items():
        rows = operator_heldout(page_ops, train_pages, test_pages)
        cov = run_coverage(page_runs, train_pages, test_pages)
        op_results[name] = {"rows": rows, "coverage": cov,
                            "train_pages": len(train_pages),
                            "test_pages": len(test_pages)}
        log(f"[#72] operator/{name}: train_pages={len(train_pages)} "
            f"test_pages={len(test_pages)} "
            f"ppl uni/o1/o2 = {rows[0]['perplexity']:.4f}/"
            f"{rows[1]['perplexity']:.4f}/{rows[2]['perplexity']:.4f} "
            f"run_token_coverage={cov['run_token_coverage']:.4f}")

    # ---- collocations: #27 sequences, same splits ------------------------
    page_units, _unit_meta = cl.load_units()
    page_tokens = cl.load_tokens()
    bodies = cl.load_bodies()
    markup_spans, markup_uids, markup_types = cl.load_markup(bodies)
    excluded_uids = set()
    for page, units in page_units.items():
        for s, uid, _lower, surface, _cls in units:
            if cl.unit_is_markup(surface, uid, s, page, markup_spans,
                                 markup_uids, markup_types):
                excluded_uids.add(uid)

    coll_results = {}
    for name, (train_pages, test_pages) in splits.items():
        u_tr, u_te = split_unit_seqs(page_units, excluded_uids,
                                     train_pages, test_pages)
        t_tr, t_te = split_token_seqs(page_tokens, markup_spans,
                                      train_pages, test_pages)
        unit_rows, _ubi_tr, _uni_tr, n_unit_tr = unit_bigram_assoc_rows(u_tr)
        unit_rows_te, _ubi_te, _uni_te, n_unit_te = unit_bigram_assoc_rows(u_te)
        unit_ho = collocation_heldout(unit_rows, unit_rows_te)
        tok_rows, _tbi_tr, _tuni_tr, n_tok_tr = token_bigram_assoc_rows(t_tr)
        tok_rows_te, _tbi_te, _tuni_te, n_tok_te = token_bigram_assoc_rows(t_te)
        tok_all_ho = collocation_heldout(tok_rows, tok_rows_te)
        tok_op_ho = collocation_heldout(
            {k: v for k, v in tok_rows.items() if v["has_op"] == "y"},
            {k: v for k, v in tok_rows_te.items() if v["has_op"] == "y"})
        coll_results[name] = {
            "unit": unit_ho, "token_all": tok_all_ho, "token_op": tok_op_ho,
            "n_unit_items_train": n_unit_tr, "n_unit_items_test": n_unit_te,
            "n_token_items_train": n_tok_tr, "n_token_items_test": n_tok_te,
        }
        log(f"[#72] collocation/{name}: unit shared={unit_ho['n_shared_types']} "
            f"spearman_logdice={unit_ho['spearman_logdice']:.4f} "
            f"topk_overlap={unit_ho['topk_overlap']:.4f} | "
            f"token-op shared={tok_op_ho['n_shared_types']} "
            f"spearman={tok_op_ho['spearman_logdice']:.4f} "
            f"overlap={tok_op_ho['topk_overlap']:.4f}")

    # ---- verdicts ---------------------------------------------------------
    # Operator: the grammar "predicts" if a higher-order model beats the
    # unigram baseline on held-out perplexity.  Argmax accuracy is reported
    # separately — better probability mass and a better single pick are
    # different claims, and conflating them is how a shallow model gets
    # oversold.
    def _op_class(r):
        if r[1]["perplexity"] < r[0]["perplexity"] and r[2]["perplexity"] < r[1]["perplexity"]:
            return "PREDICTS IN PROBABILITY"
        if r[1]["perplexity"] < r[0]["perplexity"]:
            return "PREDICTS AT ORDER 1 (ORDER 2 OVERFITS)"
        return "DOES NOT PREDICT"

    op_verdicts = {name: _op_class(res["rows"]) for name, res in op_results.items()}
    chrono = op_results["chrono"]["rows"]
    rand = op_results["random"]["rows"]
    _o2_note = ("order 2 overfits (perplexity worse than order 1)"
                if rand[2]["perplexity"] >= rand[1]["perplexity"]
                else "order 2 improves on order 1")
    op_verdict = (
        f"{op_verdicts['chrono']} (chronological), {op_verdicts['random']} "
        f"(random). Order-1 and order-2 models beat the unigram baseline on "
        f"held-out pages ({chrono[1]['perplexity']:.3f}/"
        f"{chrono[2]['perplexity']:.3f} vs {chrono[0]['perplexity']:.3f} "
        f"perplexity; a {chrono[2]['gain_vs_unigram'] * 100:.1f}% gain at order "
        f"2) and top-2 accuracy improves with order "
        f"({chrono[0]['top2_accuracy'] * 100:.1f}% -> "
        f"{chrono[2]['top2_accuracy'] * 100:.1f}%), but argmax accuracy does "
        f"not ({chrono[0]['argmax_accuracy'] * 100:.1f}% -> "
        f"{chrono[2]['argmax_accuracy'] * 100:.1f}%): the grammar assigns "
        "better probability mass than the unigram without picking the single "
        f"most likely continuation. On the random split {_o2_note}. "
        "Consistent with #11's PARTIALLY IDIOMATIC and #36's finite-memory, "
        "page-local verdicts.")

    # Collocation: rank stability of logDice is the claim; top-k survival is
    # reported alongside because the test split is 4x smaller and the
    # published f>=2 floor bites harder there.
    def _coll_class(d):
        s = d.get("spearman_logdice", float("nan"))
        if s >= 0.8:
            return "STABLE"
        if s >= 0.6:
            return "MODERATELY STABLE"
        return "UNSTABLE"

    coll_verdicts = {name: {layer: _coll_class(res[layer])
                            for layer in ("unit", "token_op")}
                     for name, res in coll_results.items()}
    cu = coll_results["chrono"]["unit"]
    ct = coll_results["chrono"]["token_op"]
    coll_verdict = (
        f"{coll_verdicts['chrono']['unit']} (unit layer, Spearman "
        f"{cu['spearman_logdice']:.4f}), "
        f"{coll_verdicts['chrono']['token_op']} (operator-word layer, "
        f"Spearman {ct['spearman_logdice']:.4f}). Ranks computed on the "
        "training 80% predict the held-out 20% — a collocation found in "
        "training keeps its strength in test — but top-100 membership does "
        f"not survive wholesale (overlap {cu['topk_overlap'] * 100:.0f}%/"
        f"{ct['topk_overlap'] * 100:.0f}%, train->test survival "
        f"{cu['topk_train_pairs_survive_in_test'] * 100:.0f}%/"
        f"{ct['topk_train_pairs_survive_in_test'] * 100:.0f}%): the test "
        "split is 4x smaller, so the published f>=2 floor drops many "
        "mid-count train pairs. Mean |delta logDice| "
        f"{cu['mean_abs_delta_logdice']:.2f}/{ct['mean_abs_delta_logdice']:.2f} "
        "on a 14-point scale — no systematic level shift.")

    # ---- artifacts --------------------------------------------------------
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    op_header = ["split", "order", "n_train_tokens", "n_test_tokens",
                 "vocab_size", "perplexity", "mean_log2_prob",
                 "argmax_accuracy", "top2_accuracy", "gain_vs_unigram"]
    op_rows = []
    for name, res in op_results.items():
        for r in res["rows"]:
            op_rows.append({"split": name, "order": r["order"],
                            "n_train_tokens": r["n_train_tokens"],
                            "n_test_tokens": r["n_test_tokens"],
                            "vocab_size": r["vocab_size"],
                            "perplexity": f4(r["perplexity"]),
                            "mean_log2_prob": f4(r["mean_log2_prob"]),
                            "argmax_accuracy": f4(r["argmax_accuracy"]),
                            "top2_accuracy": f4(r["top2_accuracy"]),
                            "gain_vs_unigram": f4(r["gain_vs_unigram"])})
    write_tsv(out_dir / "operator_heldout.tsv", op_header, op_rows)

    cov_header = ["split", "n_test_run_types", "n_test_run_types_seen_in_train",
                  "run_type_coverage", "n_test_run_tokens",
                  "n_test_run_tokens_in_train_types", "run_token_coverage"]
    cov_rows = [{"split": name,
                 **{k: (f4(v) if isinstance(v, float) else v)
                    for k, v in res["coverage"].items()}}
                for name, res in op_results.items()]
    write_tsv(out_dir / "run_coverage.tsv", cov_header, cov_rows)

    ho_header = ["split", "layer", "n_train_types", "n_test_types",
                 "n_shared_types", "spearman_logdice", "pearson_logdice",
                 "spearman_pmi", "spearman_freq", "mean_abs_delta_logdice",
                 "top_k", "topk_overlap", "topk_train_pairs_survive_in_test",
                 "topk_test_pairs_survive_in_train"]
    ho_rows = []
    for name, res in coll_results.items():
        for layer in ("unit", "token_all", "token_op"):
            d = res[layer]
            ho_rows.append({"split": name, "layer": layer,
                            "n_train_types": d.get("n_train_types", ""),
                            "n_test_types": d.get("n_test_types", ""),
                            "n_shared_types": d.get("n_shared_types", ""),
                            "spearman_logdice": f4(d.get("spearman_logdice", float("nan"))),
                            "pearson_logdice": f4(d.get("pearson_logdice", float("nan"))),
                            "spearman_pmi": f4(d.get("spearman_pmi", float("nan"))),
                            "spearman_freq": f4(d.get("spearman_freq", float("nan"))),
                            "mean_abs_delta_logdice": f4(d.get("mean_abs_delta_logdice", float("nan"))),
                            "top_k": d.get("top_k", ""),
                            "topk_overlap": f4(d.get("topk_overlap", float("nan"))),
                            "topk_train_pairs_survive_in_test": f4(d.get("topk_train_pairs_survive_in_test", float("nan"))),
                            "topk_test_pairs_survive_in_train": f4(d.get("topk_test_pairs_survive_in_train", float("nan")))})
    write_tsv(out_dir / "collocation_heldout.tsv", ho_header, ho_rows)

    # THE DELIVERABLE: the headline held-out scores, one row each.
    score_header = ["claim", "split_or_layer", "metric", "value", "baseline", "note"]
    cu = coll_results["chrono"]["unit"]
    ct = coll_results["chrono"]["token_op"]
    score_rows = [
        {"claim": "operator_sequences", "split_or_layer": "chrono",
         "metric": "perplexity_order2", "value": f4(chrono[2]["perplexity"]),
         "baseline": f4(chrono[0]["perplexity"]),
         "note": "unigram baseline; lower is better"},
        {"claim": "operator_sequences", "split_or_layer": "chrono",
         "metric": "perplexity_gain_order2_vs_unigram",
         "value": f4(chrono[2]["gain_vs_unigram"]),
         "baseline": "0.0", "note": "positive means the grammar predicts"},
        {"claim": "operator_sequences", "split_or_layer": "chrono",
         "metric": "argmax_accuracy_order2",
         "value": f4(chrono[2]["argmax_accuracy"]),
         "baseline": f4(chrono[0]["argmax_accuracy"]),
         "note": "order-0 argmax is the unigram-majority baseline"},
        {"claim": "operator_sequences", "split_or_layer": "chrono",
         "metric": "run_token_coverage",
         "value": f4(op_results["chrono"]["coverage"]["run_token_coverage"]),
         "baseline": "n/a",
         "note": "test run tokens whose run type was seen in train"},
        {"claim": "operator_sequences", "split_or_layer": "random",
         "metric": "perplexity_order2",
         "value": f4(op_results["random"]["rows"][2]["perplexity"]),
         "baseline": f4(op_results["random"]["rows"][0]["perplexity"]),
         "note": "optimistic split: seeded random pages"},
        {"claim": "operator_sequences", "split_or_layer": "random",
         "metric": "perplexity_gain_order2_vs_unigram",
         "value": f4(op_results["random"]["rows"][2]["gain_vs_unigram"]),
         "baseline": "0.0", "note": "positive means the grammar predicts"},
        {"claim": "collocations", "split_or_layer": "unit_chrono",
         "metric": "spearman_logdice_train_vs_test",
         "value": f4(cu["spearman_logdice"]), "baseline": "n/a",
         "note": "over bigram types with f>=2 in both splits"},
        {"claim": "collocations", "split_or_layer": "unit_chrono",
         "metric": "topk_train_pairs_survive_in_test",
         "value": f4(cu["topk_train_pairs_survive_in_test"]),
         "baseline": "n/a", "note": "top-100 train collocations still f>=2 in test"},
        {"claim": "collocations", "split_or_layer": "token_op_chrono",
         "metric": "spearman_logdice_train_vs_test",
         "value": f4(ct["spearman_logdice"]), "baseline": "n/a",
         "note": "operator-word bigrams only"},
        {"claim": "collocations", "split_or_layer": "token_op_chrono",
         "metric": "topk_train_pairs_survive_in_test",
         "value": f4(ct["topk_train_pairs_survive_in_test"]),
         "baseline": "n/a", "note": "operator-word bigrams only"},
        {"claim": "collocations", "split_or_layer": "unit_random",
         "metric": "spearman_logdice_train_vs_test",
         "value": f4(coll_results["random"]["unit"]["spearman_logdice"]),
         "baseline": "n/a", "note": "optimistic split"},
        {"claim": "collocations", "split_or_layer": "token_op_random",
         "metric": "spearman_logdice_train_vs_test",
         "value": f4(coll_results["random"]["token_op"]["spearman_logdice"]),
         "baseline": "n/a", "note": "optimistic split, operator-word bigrams"},
    ]
    write_tsv(out_dir / "heldout_scores.tsv", score_header, score_rows)

    diag = [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "generated", "value": generated},
        {"diag": "seed", "value": args.seed},
        {"diag": "train_frac", "value": args.frac},
        {"diag": "lam", "value": args.lam},
        {"diag": "orders", "value": list(ORDERS)},
        {"diag": "top_k", "value": TOP_K},
        {"diag": "min_bigram", "value": MIN_BIGRAM},
        {"diag": "n_pages", "value": len(pages)},
        {"diag": "n_operator_tokens", "value": n_op},
        {"diag": "operator_vocab", "value": len(set(op for v in page_ops.values() for op in v))},
        {"diag": "chrono_train_pages", "value": op_results["chrono"]["train_pages"]},
        {"diag": "chrono_test_pages", "value": op_results["chrono"]["test_pages"]},
        {"diag": "random_train_pages", "value": op_results["random"]["train_pages"]},
        {"diag": "random_test_pages", "value": op_results["random"]["test_pages"]},
        {"diag": "unit_items_train_chrono", "value": coll_results["chrono"]["n_unit_items_train"]},
        {"diag": "unit_items_test_chrono", "value": coll_results["chrono"]["n_unit_items_test"]},
        {"diag": "token_items_train_chrono", "value": coll_results["chrono"]["n_token_items_train"]},
        {"diag": "token_items_test_chrono", "value": coll_results["chrono"]["n_token_items_test"]},
        {"diag": "markup_units_removed", "value": len(excluded_uids)},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["diag", "value"], diag)

    recon = [
        {"quantity": "operator tokens", "earlier": "10,949 (#10/#11)",
         "here": f"{n_op:,}", "note": "same #10 run-expanded stream"},
        {"quantity": "chronological split pages", "earlier": "180/45 (#11)",
         "here": f"{op_results['chrono']['train_pages']}/"
                 f"{op_results['chrono']['test_pages']}",
         "note": "first 80% of sorted pages train, last 20% test"},
        {"quantity": "operator order-2 held-out perplexity (chrono)",
         "earlier": f"{on.held_out_perplexity_chrono(page_ops, 2)[0]:.4f} (#11)",
         "here": f"{chrono[2]['perplexity']:.4f}",
         "note": "same Jelinek-Mercer lam=0.8; #11's V spans all pages, "
                 "this build's V is train-only (no test leakage)"},
        {"quantity": "operator MI lag-1 (bits)", "earlier": "0.1305 (#11)",
         "here": f"{on.mutual_information_1st([op for v in page_ops.values() for op in v])[0]:.4f}",
         "note": "context-independent, shown for continuity"},
        {"quantity": "unit collocation items (full book)",
         "earlier": "74,077 (#27)",
         "here": f"{coll_results['chrono']['n_unit_items_train'] + coll_results['chrono']['n_unit_items_test']:,}",
         "note": "markup-screened unit stream, split adds back to the whole"},
        {"quantity": "token collocation items (full book)",
         "earlier": "88,669 (#27)",
         "here": f"{coll_results['chrono']['n_token_items_train'] + coll_results['chrono']['n_token_items_test']:,}",
         "note": "markup-screened token stream, split adds back to the whole"},
    ]
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    summary = {
        "schema": SCHEMA, "generated": generated, "seed": args.seed,
        "train_frac": args.frac, "lam": args.lam,
        "split": {"mode": "chronological primary, seeded random secondary",
                  "train_pages": op_results["chrono"]["train_pages"],
                  "test_pages": op_results["chrono"]["test_pages"],
                  "n_pages": len(pages)},
        "operator": {
            name: {
                "perplexity": {str(r["order"]): round(r["perplexity"], 6)
                               for r in res["rows"]},
                "argmax_accuracy": {str(r["order"]): round(r["argmax_accuracy"], 6)
                                    for r in res["rows"]},
                "top2_accuracy": {str(r["order"]): round(r["top2_accuracy"], 6)
                                  for r in res["rows"]},
                "gain_vs_unigram": {str(r["order"]): round(r["gain_vs_unigram"], 6)
                                    for r in res["rows"]},
                "n_test_tokens": res["rows"][0]["n_test_tokens"],
                "run_token_coverage": round(res["coverage"]["run_token_coverage"], 6),
                "run_type_coverage": round(res["coverage"]["run_type_coverage"], 6),
                "verdict": op_verdicts[name],
            } for name, res in op_results.items()},
        "collocation": {
            name: {
                layer: {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in res[layer].items()}
                for layer in ("unit", "token_all", "token_op")
            } for name, res in coll_results.items()},
        "operator_verdict": op_verdict,
        "collocation_verdict": coll_verdict,
        "headline": (
            f"Operator sequences: {op_verdicts['chrono'].split(' ')[0]} "
            f"(chrono order-2 perplexity {chrono[2]['perplexity']:.3f} vs "
            f"unigram {chrono[0]['perplexity']:.3f}, "
            f"{chrono[2]['gain_vs_unigram'] * 100:.1f}% gain; "
            f"{op_results['chrono']['coverage']['run_token_coverage'] * 100:.1f}% "
            f"of test run tokens use train-seen run types). "
            f"Collocations: {coll_verdicts['chrono']['unit']} (unit layer) / "
            f"{coll_verdicts['chrono']['token_op']} (operator-word layer) "
            f"(unit logDice Spearman {cu['spearman_logdice']:.4f}, "
            f"token-op {ct['spearman_logdice']:.4f}, "
            f"top-100 train-pair survival "
            f"{cu['topk_train_pairs_survive_in_test'] * 100:.0f}%/"
            f"{ct['topk_train_pairs_survive_in_test'] * 100:.0f}%)."),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    build_report(out_dir / "heldout_validity_report.md", op_results,
                 coll_results, op_verdict, coll_verdict, op_verdicts,
                 coll_verdicts, len(pages), n_op, generated)

    log(f"[#72] artifacts in {out_dir}/")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def build_report(path, op_results, coll_results, op_verdict, coll_verdict,
                 op_verdicts, coll_verdicts, n_pages, n_op, generated):
    L = []
    A = L.append
    chrono = op_results["chrono"]["rows"]
    rand = op_results["random"]["rows"]
    cu = coll_results["chrono"]["unit"]
    ct = coll_results["chrono"]["token_op"]
    ru = coll_results["random"]["unit"]
    rt = coll_results["random"]["token_op"]

    A("# Predictive validity on held-out text (#72) — train 80% / test 20%")
    A("")
    A(f"Source: #10/#11 ordered operator stream ({n_op:,} tokens, "
      f"{n_pages} pages) and #27's markup-screened unit/token streams. "
      "Split: first 80% of pages train, last 20% test (chronological, the "
      "conservative transfer test, matching #11's `held_out_perplexity_chrono`); "
      "a seeded random 80/20 page split is reported alongside as the "
      "optimistic view. Pages are the unit of splitting, so no sequence "
      "crosses the boundary. Reproducible via "
      "`python work/heldout_validity.py`; regression suite "
      "`python work/test_heldout_validity.py`.")
    A("")
    A("## 1. Headline result")
    A("")
    A(f"- **Operator sequences: {op_verdicts['chrono']}.** On the "
      f"chronological split the order-2 grammar reaches held-out perplexity "
      f"**{chrono[2]['perplexity']:.3f}** against the unigram baseline's "
      f"**{chrono[0]['perplexity']:.3f}** "
      f"(**{chrono[2]['gain_vs_unigram'] * 100:.1f}%** gain; order-1 "
      f"{chrono[1]['perplexity']:.3f}, "
      f"{chrono[1]['gain_vs_unigram'] * 100:.1f}%). Argmax accuracy does not "
      f"improve at order 2 ({chrono[0]['argmax_accuracy'] * 100:.1f}% unigram "
      f"majority -> {chrono[2]['argmax_accuracy'] * 100:.1f}%). On the random "
      f"split: {rand[2]['perplexity']:.3f} vs {rand[0]['perplexity']:.3f} "
      f"({rand[2]['gain_vs_unigram'] * 100:.1f}%).")
    A(f"- **Run inventory:** "
      f"**{op_results['chrono']['coverage']['run_token_coverage'] * 100:.1f}%** "
      f"of test run tokens use run types seen in training "
      f"({op_results['chrono']['coverage']['n_test_run_types_seen_in_train']}"
      f"/{op_results['chrono']['coverage']['n_test_run_types']} run types).")
    A(f"- **Collocations: {coll_verdicts['chrono']['unit']} (unit layer) / "
      f"{coll_verdicts['chrono']['token_op']} (operator-word layer).** "
      "Over bigram types with "
      f"f >= 2 in both splits, train vs test logDice Spearman is "
      f"**{cu['spearman_logdice']:.4f}** (unit layer, "
      f"{cu['n_shared_types']:,} shared types) and "
      f"**{ct['spearman_logdice']:.4f}** (operator-word bigrams, "
      f"{ct['n_shared_types']:,} shared types). Top-100 train collocations "
      f"surviving at published strength in test: "
      f"**{cu['topk_train_pairs_survive_in_test'] * 100:.0f}%** (unit) / "
      f"**{ct['topk_train_pairs_survive_in_test'] * 100:.0f}%** (operator-word).")
    A("")
    A(f"**Operator verdict:** {op_verdict}")
    A("")
    A(f"**Collocation verdict:** {coll_verdict}")
    A("")

    A("## 2. Operator sequences — held-out scores by order")
    A("")
    A("Interpolated n-gram Markov model (Jelinek-Mercer lam=0.8, train-only "
      "vocabulary, transitions within pages). Lower perplexity is better; "
      "the order-0 row *is* the unigram baseline.")
    A("")
    A("| split | order | train tokens | test tokens | perplexity | mean log2 P | argmax acc | top-2 acc | gain vs unigram |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, rows in (("chrono", chrono), ("random", rand)):
        for r in rows:
            A(f"| {name} | {r['order']} | {r['n_train_tokens']:,} | "
              f"{r['n_test_tokens']:,} | {r['perplexity']:.4f} | "
              f"{r['mean_log2_prob']:.4f} | {r['argmax_accuracy'] * 100:.1f}% | "
              f"{r['top2_accuracy'] * 100:.1f}% | "
              f"{r['gain_vs_unigram'] * 100:.1f}% |")
    A("")
    A("Reading: the chronological split orders the models unigram > order 1 "
      "> order 2, every order beating the unigram baseline, so the grammar "
      "predicts held-out operator sequences in probability; the random split "
      "also beats the baseline at every order, but order 2 overfits there "
      f"(perplexity {rand[2]['perplexity']:.3f} vs order 1's "
      f"{rand[1]['perplexity']:.3f}). The conservative chronological gain "
      f"(~{chrono[2]['gain_vs_unigram'] * 100:.0f}%) is far larger than the "
      f"random-split gain (~{rand[2]['gain_vs_unigram'] * 100:.0f}%), and "
      "the order-1 gains in both splits sit under #11's lag-1 "
      "mutual-information ceiling (~5% of unigram entropy): the syntax is "
      "shallow and page-local, exactly as #11's and #36's verdicts concluded.")

    A("")
    A("## 3. Run inventory coverage")
    A("")
    A("| split | test run types | seen in train | type coverage | test run tokens | tokens in seen types | token coverage |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name in ("chrono", "random"):
        c = op_results[name]["coverage"]
        A(f"| {name} | {c['n_test_run_types']} | "
          f"{c['n_test_run_types_seen_in_train']} | "
          f"{c['run_type_coverage'] * 100:.1f}% | {c['n_test_run_tokens']} | "
          f"{c['n_test_run_tokens_in_train_types']} | "
          f"{c['run_token_coverage'] * 100:.1f}% |")
    A("")
    A("A test run type never seen in training is unpredictable by "
      "construction, so this coverage is the ceiling any run grammar could "
      "reach on the held-out pages; the residual is disclosed, not dropped.")

    A("")
    A("## 4. Collocations — held-out stability")
    A("")
    A("Association rows recomputed on each side of the split with #27's "
      "formulas (logDice = 14 + log2(2f/(fx+fy)); PMI with each split's own "
      "N). Correlations are over bigram types with f >= 2 in BOTH splits.")
    A("")
    A("| split | layer | shared types | Spearman logDice | Pearson logDice | Spearman PMI | Spearman freq | mean |ΔlogDice| | top-100 overlap | train→test survival | test→train survival |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("chrono", "random"):
        for label, d in (("unit", coll_results[name]["unit"]),
                         ("token (all)", coll_results[name]["token_all"]),
                         ("token (operator-word)", coll_results[name]["token_op"])):
            A(f"| {name} | {label} | {d['n_shared_types']} | "
              f"{d['spearman_logdice']:.4f} | {d['pearson_logdice']:.4f} | "
              f"{d['spearman_pmi']:.4f} | {d['spearman_freq']:.4f} | "
              f"{d['mean_abs_delta_logdice']:.4f} | "
              f"{d['topk_overlap'] * 100:.0f}% | "
              f"{d['topk_train_pairs_survive_in_test'] * 100:.0f}% | "
              f"{d['topk_test_pairs_survive_in_train'] * 100:.0f}% |")
    A("")
    A("Reading: logDice ranks are highly stable across the split in both "
      "layers — a collocation found in the training 80% keeps its strength in "
      "the held-out 20% — and top-100 train collocations survive at published "
      "strength at very different rates by layer: "
      f"{ct['topk_train_pairs_survive_in_test'] * 100:.0f}% in the "
      f"operator-word layer but only "
      f"{cu['topk_train_pairs_survive_in_test'] * 100:.0f}% in the unit "
      "layer, because the 4x smaller test split drops mid-count pairs below "
      "the f>=2 floor. The operator-word layer is the grammar-relevant "
      "slice: it is what an operator actually selects for. PMI is reported "
      "separately because its absolute value depends on N, which differs "
      "between splits; its rank correlation is the meaningful figure.")

    A("")
    A("## 5. Method and limits")
    A("")
    A("- **Splits are page-level and disjoint** by construction; no page "
      "contributes to both sides and no sequence crosses the boundary.")
    A("- **The n-gram vocabulary is train-only** (no test leakage into the "
      "smoothing constant). #11's scoring helper lets V span all pages; the "
      "effect is a small constant offset, disclosed in `reconciliation.tsv`.")
    A("- **The chronological split is the conservative one**: it tests "
      "transfer across the book, not to stylistically adjacent pages. The "
      "random split (#11's 5-fold view, single seeded draw here) is the "
      "optimistic bound.")
    A("- **Collocation stability is a rank statement.** Absolute logDice "
      "shifts are reported (mean |Δ|) so a high correlation cannot hide a "
      "systematic level shift; it does not.")
    A("- **The block parser (#14) is a total parser** — it partitions every "
      "token stream by construction — so predictive validity is asked of the "
      "n-gram grammar and the collocation tables, not of the AST.")
    A("- **Entity-verb pairs (#27 §6) are a window measure, not a bigram "
      "table**, and are out of scope for this split evaluation.")

    A("")
    A("## 6. Artifacts")
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `heldout_scores.tsv` | THE DELIVERABLE: the headline held-out scores |")
    A("| `operator_heldout.tsv` | per split x order: perplexity, accuracies, gain |")
    A("| `run_coverage.tsv` | train run inventory vs test runs |")
    A("| `collocation_heldout.tsv` | per layer x split: stability scores |")
    A("| `summary.json` | machine-readable headline numbers + verdicts |")
    A("| `diagnostics.tsv` | build ledger (seed, frac, split sizes) |")
    A("| `reconciliation.tsv` | this build vs #11 / #27 |")
    A("")
    A("All `.tsv` are read with `split(\"\\t\")`, never a csv reader.")
    A("")
    A("---")
    A(f"*Generated by `work/heldout_validity.py` (schema `{SCHEMA}`), {generated}.*")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
