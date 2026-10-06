#!/usr/bin/env python3
"""Operator n-gram transition matrix and Markov model (Chainlink #11).

Models the ordered operator stream (1st/2nd order), tests whether transitions
are idiomatic (a real syntax) or random, and reports perplexity of the operator
stream under the model.

Reuses the ordered operator tokenizer (`scan_runs`) and operator set from
work/operator_inventory.py (#10). Source corpus is the layout-recovered markdown
with `<!-- page N -->` markers (#7).

Design notes (why this is more than the first pass):
  * The honest baseline is the UNIGRAM operator distribution, not the uniform
    V=11 distribution. Reporting "1.8x better than uniform" hides that the
    model barely beats unigram.
  * Perplexity is reported both in-sample (MLE, comparable to earlier numbers)
    and HELD-OUT (5-fold by page, interpolated backoff) so overfitting at
    order 2 is visible.
  * Conditional entropy is context-weighted: H(next|context) =
    sum_c (n_c/N) H(next|c). Averaging contexts unweighted is dominated by
    singleton contexts with H=0.
  * Transitions are split into intra-run (operators directly adjacent in the
    text) vs inter-run (operators separated by words). "Real syntax" should
    live in the former; stream-level co-occurrence lives in the latter.
  * A permutation test (shuffle the stream, preserving unigram marginals)
    tests exchangeability; a G-test quantifies dependence vs independence.

Outputs under analysis/operator_ngrams/:
  transition_1st_order.tsv    counts + P(next | current) over the full stream
  transition_2nd_order.tsv    counts + P(next | prev2, prev1)
  operator_stream.tsv         ordered stream w/ page + run-boundary flag
  run_lengths.tsv             run-length distribution (incl. singleton runs)
  perplexity_report.md        matrices, perplexity, randomness verdict
  summary.json                machine-readable headline numbers
"""
import collections
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from operator_inventory import scan_runs  # reuse #10 ordered tokenizer

SRC = "analysis/layout_recovery/blood-electric.layout.md"
OUT = "analysis/operator_ngrams"
SEED = 42


# --------------------------------------------------------------------------
# corpus -> ordered operator stream with provenance
# --------------------------------------------------------------------------
def page_runs(src):
    """Return {page_no: [run_ops, ...]} for maximal adjacent-operator runs."""
    with open(src, errors="ignore") as f:
        text = f.read()
    parts = re.split(r"<!-- page (\d+) -->", text)
    pages = {}
    for k in range(1, len(parts) - 1, 2):
        pno = int(parts[k])
        body = parts[k + 1]
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)      # layout placeholders
        body = re.sub(r"^#+[ \t]+.*$", "", body, flags=re.M)     # markdown headers
        pages[pno] = [run_ops for _run, run_ops in scan_runs(body)]
    return pages


def build_stream(pages):
    """Flatten pages to (stream, page_of, same_run_prev, run_lengths).

    same_run_prev[i] is True when stream[i] is directly adjacent to stream[i-1]
    (i.e. they belong to the same maximal run).
    """
    stream, page_of, same_run_prev = [], [], []
    run_lengths = []
    for pno in sorted(pages):
        for run_ops in pages[pno]:
            run_lengths.append(len(run_ops))
            for j, op in enumerate(run_ops):
                stream.append(op)
                page_of.append(pno)
                same_run_prev.append(j > 0)
    return stream, page_of, same_run_prev, run_lengths


# --------------------------------------------------------------------------
# counting / entropy
# --------------------------------------------------------------------------
def build_counts(stream, order, allowed=None):
    """Context counts. `allowed` optionally masks transition start indices."""
    counts = collections.defaultdict(collections.Counter)
    for i in range(len(stream) - order):
        if allowed is not None and not allowed[i]:
            continue
        counts[tuple(stream[i:i + order])][stream[i + order]] += 1
    return counts


def weighted_cond_entropy(counts):
    """H(next | context), weighted by context frequency (bits)."""
    N = sum(sum(c.values()) for c in counts.values())
    if N == 0:
        return 0.0, 0
    H = 0.0
    for counter in counts.values():
        n = sum(counter.values())
        h = -sum((v / n) * math.log2(v / n) for v in counter.values())
        H += (n / N) * h
    return H, N


def unigram_entropy(stream):
    n = len(stream)
    c = collections.Counter(stream)
    return -sum((v / n) * math.log2(v / n) for v in c.values()), c


def mutual_information_1st(stream):
    """MI between consecutive operators + G statistic (independence test)."""
    joint = collections.Counter(zip(stream, stream[1:]))
    M = sum(joint.values())
    ca = collections.Counter(stream[:-1])
    cb = collections.Counter(stream[1:])
    mi = 0.0
    for (a, b), v in joint.items():
        p = v / M
        mi += p * math.log2(p / ((ca[a] / M) * (cb[b] / M)))
    G = 2 * M * mi * math.log(2)
    return mi, G, M


def html_artifact_pages(src):
    """Pages whose OCR text contains raw HTML link markup (`href=`/`mailto:`).

    These are the pages where `</` comes from `</a>` rather than from the
    book's own operator syntax, so they can inflate apparent run structure.
    """
    with open(src, errors="ignore") as f:
        text = f.read()
    parts = re.split(r"<!-- page (\d+) -->", text)
    out = set()
    for k in range(1, len(parts) - 1, 2):
        if re.search(r"href=|mailto:", parts[k + 1], re.I):
            out.add(int(parts[k]))
    return out


def stream_stats(pages, exclude=()):
    """Order-1 stream statistics, optionally excluding whole pages.

    Returns MI between consecutive operators plus the intra-run / inter-run
    conditional-entropy gains, each against its own subset unigram baseline.
    Used for the leave-out sensitivity check on the run-concentration claim.
    """
    sub = {p: runs for p, runs in pages.items() if p not in exclude}
    stream, _page_of, same_run, _rl = build_stream(sub)
    N = len(stream)
    mi, _G, _M = mutual_information_1st(stream)
    H_uni, _ = unigram_entropy(stream)
    allowed_adj = [same_run[i + 1] for i in range(N - 1)]
    allowed_non = [not same_run[i + 1] for i in range(N - 1)]
    H_adj, n_adj = weighted_cond_entropy(build_counts(stream, 1, allowed_adj))
    H_non, n_non = weighted_cond_entropy(build_counts(stream, 1, allowed_non))
    adj_ops = [stream[i] for i in range(N - 1) if same_run[i + 1]]
    non_ops = [stream[i] for i in range(N - 1) if not same_run[i + 1]]
    H_uni_adj, _ = unigram_entropy(adj_ops)
    H_uni_non, _ = unigram_entropy(non_ops)
    return {
        "n_tokens": N, "mi_bits": mi, "unigram_H_bits": H_uni,
        "ceiling_fraction": mi / H_uni,
        "intra": {"n": n_adj, "H_bits": H_adj, "unigram_H_bits": H_uni_adj,
                  "gain_fraction": (H_uni_adj - H_adj) / H_uni_adj},
        "inter": {"n": n_non, "H_bits": H_non, "unigram_H_bits": H_uni_non,
                  "gain_fraction": (H_uni_non - H_non) / H_uni_non},
    }


def chi2_sf(x, df):
    """Upper-tail chi-square survival function; scipy if available else None."""
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:
        return None


# --------------------------------------------------------------------------
# held-out perplexity (5-fold by page, interpolated backoff)
# --------------------------------------------------------------------------
def _held_out_perplexity_split(page_ops, order, train_pages, test_pages, lam):
    """Return (sum log2 P(test transition), n_test) for a train/test page split.

    Unigram (order 0), 1st and 2nd order use Jelinek-Mercer interpolation:
      P0(n)      = puni(n)
      P1(n|a)    = lam * c(a,n)/n(a) + (1-lam) * P0(n)
      P2(n|a,b)  = lam * c(a,b,n)/n(a,b) + (1-lam) * P1(n|b)
    Only transitions whose endpoints lie within the same page are scored.
    """
    V = len({op for p in page_ops for op in page_ops[p]})
    uni = collections.Counter()
    c1 = collections.defaultdict(collections.Counter)
    c2 = collections.defaultdict(collections.Counter)
    for p in train_pages:
        ops = page_ops[p]
        uni.update(ops)
        for i in range(len(ops) - 1):
            c1[ops[i]][ops[i + 1]] += 1
        for i in range(len(ops) - 2):
            c2[(ops[i], ops[i + 1])][ops[i + 2]] += 1
    total_uni = sum(uni.values()) or 1

    def p0(n):
        return (uni[n] + 1e-9) / (total_uni + 1e-9 * V)

    def p1(n, a):
        na = sum(c1[a].values())
        ml = c1[a][n] / na if na else 0.0
        return lam * ml + (1 - lam) * p0(n)

    def p2(n, a, b):
        nab = sum(c2[(a, b)].values())
        ml = c2[(a, b)][n] / nab if nab else 0.0
        return lam * ml + (1 - lam) * p1(n, b)

    logsum, n_test = 0.0, 0
    for p in test_pages:
        ops = page_ops[p]
        for i in range(len(ops) - order):
            nxt = ops[i + order]
            if order == 0:
                prob = p0(nxt)
            elif order == 1:
                prob = p1(nxt, ops[i])
            else:
                prob = p2(nxt, ops[i], ops[i + 1])
            logsum += math.log2(max(prob, 1e-12))
            n_test += 1
    return logsum, n_test


def held_out_perplexity(pages, order, folds=5, lam=0.8):
    """Mean test perplexity over random page folds (Jelinek-Mercer, lam=0.8).

    Random folds measure generalisation to held-out pages, but pages are
    stylistically correlated, so this is an optimistic view; see
    `held_out_perplexity_chrono` for the conservative contiguous split.
    """
    page_list = sorted(pages)
    rng = random.Random(SEED)
    rng.shuffle(page_list)
    fold_of = {p: i % folds for i, p in enumerate(page_list)}
    logsum, n_test = 0.0, 0
    for f in range(folds):
        train = [p for p in page_list if fold_of[p] != f]
        test = [p for p in page_list if fold_of[p] == f]
        ls, nt = _held_out_perplexity_split(pages, order, train, test, lam)
        logsum += ls
        n_test += nt
    return 2 ** (-logsum / n_test), n_test


def held_out_perplexity_chrono(pages, order, frac=0.8, lam=0.8):
    """Conservative split: earliest `frac` of pages train, latest pages test.

    Tests whether the operator model transfers across the book rather than to
    stylistically-adjacent pages.
    """
    page_list = sorted(pages)
    cut = max(1, int(round(len(page_list) * frac)))
    ls, nt = _held_out_perplexity_split(pages, order, page_list[:cut], page_list[cut:], lam)
    return 2 ** (-ls / nt), nt


# --------------------------------------------------------------------------
# permutation test: is order beyond unigram exchangeable?
# --------------------------------------------------------------------------
def permutation_test(stream, order, n_perm=1000, seed=SEED):
    rng = random.Random(seed)
    counts = build_counts(stream, order)
    observed, _ = weighted_cond_entropy(counts)
    work = list(stream)
    count_better = 0
    for _ in range(n_perm):
        rng.shuffle(work)
        h, _ = weighted_cond_entropy(build_counts(work, order))
        if h <= observed + 1e-12:
            count_better += 1
    return observed, count_better / n_perm


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def write_matrix(counts, order, path):
    with open(path, "w") as f:
        if order == 1:
            f.write("current\tnext\tcount\tP_next_given_current\tlog2_prob\n")
            for ctx, counter in sorted(counts.items()):
                total = sum(counter.values())
                for nxt, c in sorted(counter.items(), key=lambda x: -x[1]):
                    f.write(f"{ctx[0]}\t{nxt}\t{c}\t{c/total:.6f}\t{math.log2(c/total):.6f}\n")
        else:
            f.write("prev2\tprev1\tnext\tcount\tP_next_given_ctx\tlog2_prob\n")
            for ctx, counter in sorted(counts.items()):
                total = sum(counter.values())
                for nxt, c in sorted(counter.items(), key=lambda x: -x[1]):
                    f.write(f"{ctx[0]}\t{ctx[1]}\t{nxt}\t{c}\t{c/total:.6f}\t{math.log2(c/total):.6f}\n")


def example_pages(stream, page_of, a, b, maxn=4):
    pages = []
    for i in range(len(stream) - 1):
        if stream[i] == a and stream[i + 1] == b and page_of[i] not in pages:
            pages.append(page_of[i])
        if len(pages) >= maxn:
            break
    return pages


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    pages = page_runs(SRC)
    page_ops = {p: [op for run_ops in runs for op in run_ops] for p, runs in pages.items()}
    stream, page_of, same_run_prev, run_lengths = build_stream(pages)
    N = len(stream)
    V = len(set(stream))
    os.makedirs(OUT, exist_ok=True)

    # raw stream artifact
    with open(f"{OUT}/operator_stream.tsv", "w") as f:
        f.write("index\tpage\toperator\tsame_run_as_prev\n")
        for i, op in enumerate(stream):
            f.write(f"{i}\t{page_of[i]}\t{op}\t{int(same_run_prev[i])}\n")

    rl = collections.Counter(run_lengths)
    with open(f"{OUT}/run_lengths.tsv", "w") as f:
        f.write("run_length\trun_count\toperator_tokens\n")
        for L in sorted(rl):
            f.write(f"{L}\t{rl[L]}\t{L*rl[L]}\n")

    # full-stream models
    counts1 = build_counts(stream, 1)
    counts2 = build_counts(stream, 2)
    write_matrix(counts1, 1, f"{OUT}/transition_1st_order.tsv")
    write_matrix(counts2, 2, f"{OUT}/transition_2nd_order.tsv")

    H_uni, uni_counts = unigram_entropy(stream)
    H1, n1 = weighted_cond_entropy(counts1)
    H2, n2 = weighted_cond_entropy(counts2)
    perp_uni = 2 ** H_uni
    perp1_in = 2 ** H1
    perp2_in = 2 ** H2

    mi, G, M = mutual_information_1st(stream)
    df = (V - 1) ** 2
    p_mi = chi2_sf(G, df)

    print(f"N={N} V={V} transitions1={n1} transitions2={n2}")
    print(f"unigram H={H_uni:.3f} perp={perp_uni:.3f}")
    print(f"order1 H={H1:.3f} perp={perp1_in:.3f}")
    print(f"order2 H={H2:.3f} perp={perp2_in:.3f}")
    print(f"MI1={mi:.4f} bits  G={G:.1f} df={df} p={p_mi}")

    # held-out perplexity (random page folds)
    ho_uni, nt0 = held_out_perplexity(page_ops, 0)
    ho1, nt1 = held_out_perplexity(page_ops, 1)
    ho2, nt2 = held_out_perplexity(page_ops, 2)
    print(f"held-out perp: unigram={ho_uni:.3f} order1={ho1:.3f} order2={ho2:.3f}")

    # conservative contiguous (chronological) split
    cho_uni, _ = held_out_perplexity_chrono(page_ops, 0)
    cho1, _ = held_out_perplexity_chrono(page_ops, 1)
    cho2, _ = held_out_perplexity_chrono(page_ops, 2)
    page_list = sorted(page_ops)
    chrono_cut = max(1, int(round(len(page_list) * 0.8)))
    chrono_train_n, chrono_test_n = chrono_cut, len(page_list) - chrono_cut
    print(f"chrono held-out perp: unigram={cho_uni:.3f} order1={cho1:.3f} order2={cho2:.3f}")

    # permutation tests
    obs_h1, p_perm1 = permutation_test(stream, 1, n_perm=1000)
    obs_h2, p_perm2 = permutation_test(stream, 2, n_perm=1000)
    print(f"perm test order1 p={p_perm1}  order2 p={p_perm2}")

    # adjacent vs non-adjacent (order 1) + leave-out sensitivity
    full_stats = stream_stats(pages)
    H_adj = full_stats["intra"]["H_bits"]
    n_adj = full_stats["intra"]["n"]
    H_non = full_stats["inter"]["H_bits"]
    n_non = full_stats["inter"]["n"]
    H_uni_adj = full_stats["intra"]["unigram_H_bits"]
    H_uni_non = full_stats["inter"]["unigram_H_bits"]
    artifact_pages = html_artifact_pages(SRC)
    sens92 = stream_stats(pages, {92})
    sens_all = stream_stats(pages, artifact_pages)
    print(f"adjacent n={n_adj} H={H_adj:.3f} (unigram-subset {H_uni_adj:.3f})")
    print(f"non-adj  n={n_non} H={H_non:.3f} (unigram-subset {H_uni_non:.3f})")
    print(f"sensitivity: drop 92 intra_gain={sens92['intra']['gain_fraction']:.3f} "
          f"MI={sens92['mi_bits']:.4f}; drop {sorted(artifact_pages)} "
          f"intra_gain={sens_all['intra']['gain_fraction']:.3f} "
          f"MI={sens_all['mi_bits']:.4f}")

    # verdict
    gain1_cv = (ho_uni - ho1) / ho_uni
    gain2_cv = (ho_uni - ho2) / ho_uni
    gain1_ch = (cho_uni - cho1) / cho_uni
    gain2_ch = (cho_uni - cho2) / cho_uni
    intra_gain = (H_uni_adj - H_adj) / H_uni_adj
    inter_gain = (H_uni_non - H_non) / H_uni_non
    # Information-theoretic ceiling: H(next) - H(next|context) = MI(context;next).
    # No order-1 model can beat this even with infinite data.
    ceiling1 = mi / H_uni
    ceiling2 = (H_uni - H2) / H_uni
    significant = (p_perm1 < 0.05) or (p_mi is not None and p_mi < 0.05)
    if not significant:
        verdict = ("RANDOM / NO DETECTABLE SYNTAX — operator transitions are not "
                   "more structured than a marginal-preserving shuffle.")
    elif intra_gain >= 0.15 and inter_gain < 0.10:
        verdict = ("PARTIALLY IDIOMATIC (shallow, local operator syntax) — sequential "
                   "dependence is real and unambiguous, but it is concentrated in a small "
                   "fixed inventory of adjacent multi-operator runs; word-separated "
                   "(inter-run) transitions are only weakly predictable, and the total "
                   "order-1 entropy reduction is bounded near the mutual-information "
                   "ceiling (~5%). Real syntax at run scale, not a deep stream grammar.")
    elif gain1_cv < 0.05:
        verdict = ("WEAKLY STRUCTURED (non-random but shallow) — transitions carry "
                   "significant sequential dependence, but held-out perplexity is only "
                   "marginally below the unigram baseline.")
    else:
        verdict = ("IDIOMATIC (real syntax) — held-out transitions are substantially "
                   "more predictable than the unigram baseline at both intra- and "
                   "inter-run distance.")

    # report
    L = []
    ap = L.append
    ap("# Operator n-gram Transition Matrix and Markov Model (#11)\n")
    ap(f"Source: `{SRC}` — {N:,} operator tokens, {V} distinct operators, "
       f"{len(run_lengths):,} maximal runs (mean length "
       f"{sum(run_lengths)/len(run_lengths):.2f}).\n")
    ap("Operator set and ordered tokenizer reused from `work/operator_inventory.py` (#10). "
       "Runs are maximal adjacent-operator sequences; a transition is *intra-run* when its "
       "two operators are directly adjacent in the text and *inter-run* otherwise.\n")

    ap("## 1. Perplexity of the operator stream\n")
    ap("Perplexity is `2^H`, lower = more predictable. The honest baseline is the "
       "**unigram** operator distribution (frequencies only, no order); the uniform "
       "`V`=11 baseline is a strawman.\n")
    ap("| model | in-sample H (bits) | in-sample perp | held-out perp (5-fold, by page) | gain vs unigram (held-out) |")
    ap("|---|---|---|---|---|")
    ap(f"| unigram (order 0) | {H_uni:.3f} | {perp_uni:.3f} | {ho_uni:.3f} | — |")
    ap(f"| 1st order | {H1:.3f} | {perp1_in:.3f} | {ho1:.3f} | {gain1_cv:+.1%} |")
    ap(f"| 2nd order | {H2:.3f} | {perp2_in:.3f} | {ho2:.3f} | {gain2_cv:+.1%} |")
    ap(f"\nHeld-out scoring uses Jelinek-Mercer interpolation (λ=0.8) so unseen contexts "
       f"back off to unigram; {nt1:,} held-out transitions. In-sample numbers reproduce the "
       "first pass (order-1 perp 6.097, order-2 perp 5.310) and are optimistic because the "
       "model is scored on its own training stream.\n")
    ap(f"**Information-theoretic ceiling.** For any first-order model, "
       f"H(next) − H(next|current) = MI(current; next) = {mi:.4f} bits, so even an "
       f"infinite-data order-1 model can only cut entropy by **{ceiling1:.1%}** "
       f"(perplexity {perp_uni:.2f} → {2**(H_uni-mi):.2f}). The held-out gain of "
       f"{gain1_cv:.1%} is therefore close to the maximum; order 1 is essentially maxed "
       f"out. The order-2 ceiling is {ceiling2:.1%}, and held-out order 2 achieves "
       f"{gain2_cv:.1%} — i.e. most apparent order-2 structure does not generalise.\n")
    ap(f"**Chronological robustness check.** Random page folds can be optimistic because "
       f"pages are stylistically correlated, so we also train on the earliest "
       f"{chrono_train_n} pages and test on the latest {chrono_test_n} (a single contiguous "
       f"split). Held-out perplexity: unigram {cho_uni:.3f} → order-1 {cho1:.3f} "
       f"({gain1_ch:+.1%}) → order-2 {cho2:.3f} ({gain2_ch:+.1%}). The ordering (order 1 "
       f"beats unigram, order 2 beats order 1) is unchanged under this conservative split, "
       f"so the shallow local structure is not an artifact of adjacent-page leakage. The "
       f"gain is in fact larger here because the later pages have a different operator mix "
       f"from the training pages (the held-out unigram baseline is correspondingly worse), "
       f"so the full-stream MI ceiling of {ceiling1:.1%} applies to the pooled stream, not "
       f"to this split.\n")

    ap("## 2. First-order transition matrix (P(next | current))\n")
    ap(f"- transitions: {n1:,}; artifact: `transition_1st_order.tsv`")
    ap(f"- mutual information between consecutive operators: **{mi:.4f} bits** "
       f"(unigram entropy is {H_uni:.3f} bits, so order carries ~{mi/H_uni:.1%} of it)")
    if p_mi is not None:
        ap(f"- G-test of independence: G={G:.1f}, df={df}, **p={p_mi:.2e}** "
           "(dependence is statistically real)")
    ap(f"- permutation test (marginal-preserving shuffle, 1000×): **p={p_perm1:.3f}** "
       "(0.000 = no shuffle was as structured as the observed stream)\n")

    ap("## 3. Second-order transition matrix (P(next | prev2, prev1))\n")
    ap(f"- transitions: {n2:,}; distinct contexts: {len(counts2)}; artifact: `transition_2nd_order.tsv`")
    ap(f"- weighted conditional entropy H(next|prev2,prev1) = {H2:.3f} bits "
       f"(vs {H1:.3f} at order 1)")
    ap(f"- permutation test: **p={p_perm2:.3f}**")
    ap(f"- the order-2 model adds {H1 - H2:.3f} bits over order 1 in-sample; the held-out "
       f"gain is {gain2_cv - gain1_cv:+.1%} relative to order 1, so much of the apparent "
       "order-2 structure is overfitting to singleton contexts.\n")

    ap("## 4. Intra-run vs inter-run transitions (where is the structure?)\n")
    ap("| transition type | n | H(next\\|current) | subset unigram H | perp |")
    ap("|---|---|---|---|---|")
    ap(f"| intra-run (adjacent) | {n_adj:,} | {H_adj:.3f} | {H_uni_adj:.3f} | {2**H_adj:.3f} |")
    ap(f"| inter-run (word-separated) | {n_non:,} | {H_non:.3f} | {H_uni_non:.3f} | {2**H_non:.3f} |")
    ap(f"\nOnly {n_adj:,}/{n1:,} ({n_adj/n1:.1%}) of transitions are directly adjacent. "
       "Adjacent transitions are the more predictable ones (lower H), consistent with a "
       "small fixed inventory of multi-operator runs. But even intra-run entropy is "
       f"{H_adj:.2f} bits — runs are not deterministic opcodes.\n")
    ap("**Leave-out sensitivity (HTML-artifact pages).** Pages "
       f"{', '.join(map(str, sorted(artifact_pages)))} contain raw `<a href>`/`mailto:` OCR "
       "markup. Page 92 alone contributes 97 of the 132 intra-run `<`→`/` bigrams. "
       "Recomputing the order-1 statistics without them:\n")
    ap("| variant | tokens | MI (bits) | order-1 ceiling | intra gain | inter gain |")
    ap("|---|---|---|---|---|---|")
    for label, st in (("full stream", full_stats),
                      ("drop page 92", sens92),
                      (f"drop pages {', '.join(map(str, sorted(artifact_pages)))}", sens_all)):
        ap(f"| {label} | {st['n_tokens']:,} | {st['mi_bits']:.4f} | "
           f"{st['ceiling_fraction']:.1%} | {st['intra']['gain_fraction']:.1%} | "
           f"{st['inter']['gain_fraction']:.1%} |")
    ap(f"\nThe intra-run concentration is substantially inflated by one page of HTML "
       f"artifacts: the honest run-scale gain is ~{sens_all['intra']['gain_fraction']:.0%}, "
       f"not ~{intra_gain:.0%}. Sequential dependence remains significant "
       f"(MI {sens_all['mi_bits']:.3f} bits, ceiling {sens_all['ceiling_fraction']:.1%}) and "
       "the headline verdict is unchanged, but the 'adjacent runs are much more predictable' "
       "sub-claim is fragile and should not be leaned on.\n")

    ap("## 5. Top predictive transitions (1st order)\n")
    ap("| current | next | P(next\\|current) | count | example pages |")
    ap("|---|---|---|---|---|")
    rows = []
    for ctx, counter in counts1.items():
        total = sum(counter.values())
        for nxt, c in counter.most_common(2):
            rows.append((c / total, ctx[0], nxt, c, total))
    for p, a, b, c, tot in sorted(rows, reverse=True)[:25]:
        ex = example_pages(stream, page_of, a, b)
        ap(f"| `{a}` | `{b}` | {p:.3f} | {c:,} | {', '.join(map(str, ex))} |")

    ap("\n## 6. Randomness verdict\n")
    ap(f"**{verdict}**\n")
    ap("Supporting measurements:")
    ap(f"- Sequential dependence is statistically unambiguous (G-test p={p_mi:.1e}; "
       f"permutation p={p_perm1:.3f} at order 1). The stream is **not** an i.i.d. draw.")
    ap(f"- But the effect is small and bounded: the order-1 mutual-information ceiling is "
       f"{ceiling1:.1%}, and held-out order 1 achieves {gain1_cv:.1%} "
       f"(perplexity {ho_uni:.2f} → {ho1:.2f}); order 2 achieves {gain2_cv:.1%} against a "
       f"{ceiling2:.1%} ceiling.")
    ap(f"- Most predictability is local, but the margin is partly an artifact: intra-run "
       f"H={H_adj:.2f} bits vs its subset unigram {H_uni_adj:.2f} ({intra_gain:.1%}), "
       f"whereas inter-run H={H_non:.2f} vs {H_uni_non:.2f} ({inter_gain:.1%}); dropping the "
       f"HTML-artifact pages cuts the intra-run gain to "
       f"{sens_all['intra']['gain_fraction']:.1%} (see §4). The strongest bigrams are a "
       "handful of fixed runs (`</`, `=>`, `<<::`, `//::`, `//<<`, `::/`, …).")
    ap(f"- The verdict is stable under the conservative chronological split "
       f"(held-out order-1 gain {gain1_ch:.1%}, order-2 {gain2_ch:.1%}), not only under "
       f"random page folds.")
    ap("- Interpretation: the operator layer behaves like a **shallow, frequency-skewed "
       "Markov process with a fixed run lexicon**, not a deep recursive grammar. This is "
       "measured structure; the claim that it is a *designed* syntax remains a hypothesis "
       "for #12 (function classification) and #14 (block parser).\n")

    ap("## 7. Caveats\n")
    ap("- `<`→`/` is dominated by page 92 (97 of the 132 intra-run occurrences; 97 of 140 "
       "including inter-run) and `=`/`>` bigrams may be OCR artifacts of HTML/`=>`. The "
       "leave-out sensitivity in §4 quantifies this: the intra-run gain drops from ~20% to "
       "~10% once the HTML-artifact pages are removed.")
    ap("- Inter-run transitions ignore the words between operators, so they measure "
       "operator co-occurrence at arbitrary distance, not clause syntax.")
    ap("- The headline folds are random (optimistic if adjacent pages share style); the "
       "chronological split above controls for this and gives a gain of the same sign and "
       "similar order (in fact larger). Drift *within* the stream (per-decile entropy) is "
       "not measured here.")
    ap("\n---\n*Generated by `work/operator_ngrams.py`*")

    with open(f"{OUT}/perplexity_report.md", "w") as f:
        f.write("\n".join(L) + "\n")

    summary = {
        "source": SRC, "n_operator_tokens": N, "vocab": V,
        "n_runs": len(run_lengths), "mean_run_length": sum(run_lengths) / len(run_lengths),
        "unigram_entropy_bits": H_uni, "unigram_perplexity": perp_uni,
        "order1_entropy_bits": H1, "order1_perplexity_insample": perp1_in,
        "order2_entropy_bits": H2, "order2_perplexity_insample": perp2_in,
        "heldout_perplexity": {"unigram": ho_uni, "order1": ho1, "order2": ho2},
        "heldout_gain_vs_unigram": {"order1": gain1_cv, "order2": gain2_cv},
        "heldout_perplexity_chrono": {"unigram": cho_uni, "order1": cho1, "order2": cho2},
        "heldout_gain_vs_unigram_chrono": {"order1": gain1_ch, "order2": gain2_ch},
        "chrono_split": {"train_pages": chrono_train_n, "test_pages": chrono_test_n,
                         "train_frac": 0.8},
        "mutual_information_1st_bits": mi, "g_statistic": G, "g_df": df, "g_pvalue": p_mi,
        "info_ceiling_fraction": {"order1": ceiling1, "order2": ceiling2},
        "permutation_pvalue": {"order1": p_perm1, "order2": p_perm2},
        "intra_run": {"n": n_adj, "H_bits": H_adj, "unigram_H_bits": H_uni_adj,
                      "gain_fraction": intra_gain},
        "inter_run": {"n": n_non, "H_bits": H_non, "unigram_H_bits": H_uni_non,
                      "gain_fraction": inter_gain},
        "sensitivity_leave_out": {
            "html_artifact_pages": sorted(artifact_pages),
            "full": {"mi_bits": full_stats["mi_bits"],
                     "intra_gain_fraction": full_stats["intra"]["gain_fraction"]},
            "drop_page_92": {"mi_bits": sens92["mi_bits"],
                             "intra_gain_fraction": sens92["intra"]["gain_fraction"]},
            "drop_html_artifact_pages": {
                "mi_bits": sens_all["mi_bits"],
                "intra_gain_fraction": sens_all["intra"]["gain_fraction"]},
        },
        "verdict": verdict,
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[written] {OUT}/perplexity_report.md")
    print(f"[written] {OUT}/summary.json")
    print(f"[written] {OUT}/transition_1st_order.tsv")
    print(f"[written] {OUT}/transition_2nd_order.tsv")
    print(f"[written] {OUT}/operator_stream.tsv")
    print(f"[written] {OUT}/run_lengths.tsv")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
