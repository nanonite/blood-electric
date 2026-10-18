# Predictive validity on held-out text (#72) — train 80% / test 20%

Source: #10/#11 ordered operator stream (10,949 tokens, 225 pages) and #27's markup-screened unit/token streams. Split: first 80% of pages train, last 20% test (chronological, the conservative transfer test, matching #11's `held_out_perplexity_chrono`); a seeded random 80/20 page split is reported alongside as the optimistic view. Pages are the unit of splitting, so no sequence crosses the boundary. Reproducible via `python work/heldout_validity.py`; regression suite `python work/test_heldout_validity.py`.

## 1. Headline result

- **Operator sequences: PREDICTS IN PROBABILITY.** On the chronological split the order-2 grammar reaches held-out perplexity **6.100** against the unigram baseline's **6.792** (**10.2%** gain; order-1 6.308, 7.1%). Argmax accuracy does not improve at order 2 (30.4% unigram majority -> 29.9%). On the random split: 6.323 vs 6.427 (1.6%).
- **Run inventory:** **99.3%** of test run tokens use run types seen in training (1738/1744 run types).
- **Collocations: STABLE (unit layer) / MODERATELY STABLE (operator-word layer).** Over bigram types with f >= 2 in both splits, train vs test logDice Spearman is **0.8984** (unit layer, 1,193 shared types) and **0.7080** (operator-word bigrams, 316 shared types). Top-100 train collocations surviving at published strength in test: **48%** (unit) / **66%** (operator-word).

**Operator verdict:** PREDICTS IN PROBABILITY (chronological), PREDICTS AT ORDER 1 (ORDER 2 OVERFITS) (random). Order-1 and order-2 models beat the unigram baseline on held-out pages (6.308/6.100 vs 6.792 perplexity; a 10.2% gain at order 2) and top-2 accuracy improves with order (44.9% -> 50.8%), but argmax accuracy does not (30.4% -> 29.9%): the grammar assigns better probability mass than the unigram without picking the single most likely continuation. On the random split order 2 overfits (perplexity worse than order 1). Consistent with #11's PARTIALLY IDIOMATIC and #36's finite-memory, page-local verdicts.

**Collocation verdict:** STABLE (unit layer, Spearman 0.8984), MODERATELY STABLE (operator-word layer, Spearman 0.7080). Ranks computed on the training 80% predict the held-out 20% — a collocation found in training keeps its strength in test — but top-100 membership does not survive wholesale (overlap 47%/59%, train->test survival 48%/66%): the test split is 4x smaller, so the published f>=2 floor drops many mid-count train pairs. Mean |delta logDice| 0.86/0.94 on a 14-point scale — no systematic level shift.

## 2. Operator sequences — held-out scores by order

Interpolated n-gram Markov model (Jelinek-Mercer lam=0.8, train-only vocabulary, transitions within pages). Lower perplexity is better; the order-0 row *is* the unigram baseline.

| split | order | train tokens | test tokens | perplexity | mean log2 P | argmax acc | top-2 acc | gain vs unigram |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chrono | 0 | 8,943 | 2,006 | 6.7923 | -2.7639 | 30.4% | 44.9% | 0.0% |
| chrono | 1 | 8,943 | 1,962 | 6.3076 | -2.6571 | 31.3% | 49.0% | 7.1% |
| chrono | 2 | 8,943 | 1,918 | 6.1004 | -2.6089 | 29.9% | 50.8% | 10.2% |
| random | 0 | 9,041 | 1,908 | 6.4268 | -2.6841 | 30.0% | 51.1% | 0.0% |
| random | 1 | 9,041 | 1,864 | 6.2280 | -2.6388 | 29.9% | 51.1% | 3.1% |
| random | 2 | 9,041 | 1,820 | 6.3226 | -2.6605 | 28.0% | 51.4% | 1.6% |

Reading: the chronological split orders the models unigram > order 1 > order 2, every order beating the unigram baseline, so the grammar predicts held-out operator sequences in probability; the random split also beats the baseline at every order, but order 2 overfits there (perplexity 6.323 vs order 1's 6.228). The conservative chronological gain (~10%) is far larger than the random-split gain (~2%), and the order-1 gains in both splits sit under #11's lag-1 mutual-information ceiling (~5% of unigram entropy): the syntax is shallow and page-local, exactly as #11's and #36's verdicts concluded.

## 3. Run inventory coverage

| split | test run types | seen in train | type coverage | test run tokens | tokens in seen types | token coverage |
|---|---:|---:|---:|---:|---:|---:|
| chrono | 1744 | 1738 | 99.7% | 2006 | 1991 | 99.3% |
| random | 1699 | 1692 | 99.6% | 1908 | 1890 | 99.1% |

A test run type never seen in training is unpredictable by construction, so this coverage is the ceiling any run grammar could reach on the held-out pages; the residual is disclosed, not dropped.

## 4. Collocations — held-out stability

Association rows recomputed on each side of the split with #27's formulas (logDice = 14 + log2(2f/(fx+fy)); PMI with each split's own N). Correlations are over bigram types with f >= 2 in BOTH splits.

| split | layer | shared types | Spearman logDice | Pearson logDice | Spearman PMI | Spearman freq | mean |ΔlogDice| | top-100 overlap | train→test survival | test→train survival |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chrono | unit | 1193 | 0.8984 | 0.9206 | 0.9015 | 0.6925 | 0.8631 | 47% | 48% | 83% |
| chrono | token (all) | 1452 | 0.8717 | 0.9059 | 0.8853 | 0.7028 | 0.8784 | 50% | 51% | 85% |
| chrono | token (operator-word) | 316 | 0.7080 | 0.7986 | 0.8067 | 0.6891 | 0.9439 | 59% | 66% | 84% |
| random | unit | 1212 | 0.9166 | 0.9361 | 0.9142 | 0.7499 | 0.8210 | 60% | 62% | 75% |
| random | token (all) | 1517 | 0.8840 | 0.9206 | 0.8927 | 0.7401 | 0.8189 | 64% | 64% | 80% |
| random | token (operator-word) | 338 | 0.6513 | 0.7749 | 0.7909 | 0.6375 | 0.8927 | 59% | 76% | 77% |

Reading: logDice ranks are highly stable across the split in both layers — a collocation found in the training 80% keeps its strength in the held-out 20% — and top-100 train collocations survive at published strength at very different rates by layer: 66% in the operator-word layer but only 48% in the unit layer, because the 4x smaller test split drops mid-count pairs below the f>=2 floor. The operator-word layer is the grammar-relevant slice: it is what an operator actually selects for. PMI is reported separately because its absolute value depends on N, which differs between splits; its rank correlation is the meaningful figure.

## 5. Method and limits

- **Splits are page-level and disjoint** by construction; no page contributes to both sides and no sequence crosses the boundary.
- **The n-gram vocabulary is train-only** (no test leakage into the smoothing constant). #11's scoring helper lets V span all pages; the effect is a small constant offset, disclosed in `reconciliation.tsv`.
- **The chronological split is the conservative one**: it tests transfer across the book, not to stylistically adjacent pages. The random split (#11's 5-fold view, single seeded draw here) is the optimistic bound.
- **Collocation stability is a rank statement.** Absolute logDice shifts are reported (mean |Δ|) so a high correlation cannot hide a systematic level shift; it does not.
- **The block parser (#14) is a total parser** — it partitions every token stream by construction — so predictive validity is asked of the n-gram grammar and the collocation tables, not of the AST.
- **Entity-verb pairs (#27 §6) are a window measure, not a bigram table**, and are out of scope for this split evaluation.

## 6. Artifacts

| file | contents |
|---|---|
| `heldout_scores.tsv` | THE DELIVERABLE: the headline held-out scores |
| `operator_heldout.tsv` | per split x order: perplexity, accuracies, gain |
| `run_coverage.tsv` | train run inventory vs test runs |
| `collocation_heldout.tsv` | per layer x split: stability scores |
| `summary.json` | machine-readable headline numbers + verdicts |
| `diagnostics.tsv` | build ledger (seed, frac, split sizes) |
| `reconciliation.tsv` | this build vs #11 / #27 |

All `.tsv` are read with `split("\t")`, never a csv reader.

---
*Generated by `work/heldout_validity.py` (schema `heldout_validity/1`), 2026-10-12T22:55:02Z.*
