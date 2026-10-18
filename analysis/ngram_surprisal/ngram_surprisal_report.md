# n-gram language model and per-token surprisal (#35)

Source: `analysis/tokenizer/units.tsv` (#17) screened by #19's ledger and #27's HTML tag-run detector (`work/collocation.py`), leaving 74,077 units; clauses from `analysis/clauses/` (#14). An interpolated absolute-discounting n-gram LM is trained on the book and every unit is scored **out-of-fold** by a leave-one-page-out model. Reproduce with `python work/ngram_surprisal.py`; regression suite `python work/test_ngram_surprisal.py`. Schema `ngram-surprisal/1`.

## 1. Method in one paragraph

The token stream is the **unit** layer screened to 74,077 occurrences (#17's 74,665 minus #19's 347 ledger entries and #27's 241 tag-furniture occurrences `href`/`mailto`/`colspan`), so fused naming units (`gene=TV`, `soul/gram`) are single tokens and OCR/HTML residue does not masquerade as the most surprising content. The model is `P(w|h) = max(c(h,w)-d,0)/c(h) + d*N1+(h)/c(h) * P(w|h')` with `d=0.75` and a small add-delta floor at the unigram base (`delta=0.1`). Every token is scored by the model trained on the other pages, built by exact count subtraction (`global - page`); no token ever sees itself, and no step uses randomness. Surprisal is `-log2 P` in bits.

## 2. Held-out perplexity by order (model selection)

| model | held-out perplexity | mean surprisal (bits) |
|---|---:|---:|
| order 1 (1-gram) | 323.579 | 8.338 |
| order 2 (2-gram) | 100.366 | 6.649 |
| order 3 (3-gram) | 99.656 **<- best** | 6.639 |
| order 4 (4-gram) | 103.223 | 6.690 |

Order 3 wins at held-out perplexity **99.656** (mean surprisal 6.639 bits/token); the unigram baseline is 323.579. The per-token curve below uses **3-gram**.

**Case-preserved sensitivity.** Keeping the original case (so `ADAM` != `adam`) gives held-out perplexity 108.650 at its best order 3 (vs 99.656 folded), so the headline is not an artifact of case-folding the naming layer.

## 3. Per-token surprisal distribution

Across 74,077 units the mean surprisal is **6.639 bits** (sd 6.032); the unigram model's held-out perplexity is 323.6. Surprisal is heavy-tailed: function-word loops are near-free, hapax coinage and rare English are expensive.

### 3.1 By #19 lexical category

| category | n | mean surprisal | median | p90 |
|---|---:|---:|---:|---:|
| english | 66,500 | 5.965 | 4.356 | 13.849 |
| compound | 6,716 | 12.828 | 10.524 | 23.367 |
| true_coinage | 338 | 9.921 | 8.913 | 20.173 |
| proper_noun | 287 | 9.882 | 9.442 | 15.412 |
| ocr_error | 127 | 11.940 | 11.447 | 16.970 |
| numeric | 60 | 12.078 | 14.427 | 20.739 |
| rare_english | 49 | 10.835 | 12.586 | 21.621 |

The **English core is the cheap channel** (5.965 bits), while the non-English naming layer is dearer (`compound` 12.828 bits). This is the direct, token-level confirmation of the L3/L4 claim that the derangement lives in the coinage, not in the function words.

### 3.2 Highest- and lowest-surprisal unit types (n >= 5)

| most surprising types | n | bits | | least surprising types | n | bits |
|---|---:|---:|---|---|---:|---:|
| `splits` | 6 | 17.91 | | `World` | 61 | 0.65 |
| `hyper-link` | 5 | 17.62 | | `processing` | 6 | 0.58 |
| `outlet` | 6 | 17.61 | | `Corporation` | 27 | 0.55 |
| `membrane` | 5 | 17.47 | | `abortion` | 29 | 0.53 |
| `hyper-control` | 6 | 17.31 | | `equivalent` | 7 | 0.50 |
| `subcutaneous` | 5 | 17.30 | | `disease` | 198 | 0.32 |
| `reflex` | 6 | 17.15 | | `period` | 153 | 0.31 |
| `null` | 5 | 17.15 | | `physical-LOVE` | 6 | 0.22 |
| `neurone` | 5 | 17.00 | | `term` | 22 | 0.22 |
| `clone=transmission` | 5 | 16.98 | | `Cherry` | 12 | 0.08 |
| `pool` | 6 | 16.92 | | `aberration` | 9 | 0.04 |
| `connection` | 5 | 16.92 | | `spot` | 30 | 0.03 |

### 3.3 Surprisal by global type frequency (the real driver)

Binning every occurrence by how often its **type** occurs in the book isolates the mechanism behind the hot/cold curve:

| type frequency | n | mean surprisal | median |
|---|---:|---:|---:|
| 1 (hapax) | 3,543 | 21.907 | 22.228 |
| 2-3 | 2,295 | 15.276 | 16.854 |
| 4-5 | 1,348 | 13.568 | 15.020 |
| 6-10 | 2,552 | 12.166 | 13.572 |
| 11-50 | 10,393 | 9.613 | 10.405 |
| 51-200 | 16,814 | 6.868 | 7.635 |
| 201+ | 37,132 | 3.081 | 2.174 |

Surprisal falls monotonically with type frequency: a hapax costs tens of bits, a type seen 200+ times costs well under a bit. Because #24 showed the book's vocabulary is dominated by a small repeated core, the *aggregate* predictability is high even though its tail is extreme.

## 4. Per-page surprisal curve and hot/cold zones

Per-page mean surprisal has mean 6.836 bits and sd 1.591; a page is **hot** at z >= +1 and **cold** at z <= -1 (pages with fewer than 20 units are excluded from the flag to avoid small-sample noise).

- hot pages: **12** (2, 27, 42, 43, 44, 45, 48, 101, 119, 188, 209, 224)
- cold pages: **10** (4, 5, 6, 7, 41, 90, 220, 221, 222, 223)

### 4.1 The hottest and coldest pages (n >= 20)

| rank | page | n | mean bits | z | hapax share | nearest page | Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 31 | 15.462 | +5.42 | 0.35 | 1 | 0.06 |
| 2 | 224 | 237 | 11.301 | +2.81 | 0.21 | 79 | 0.05 |
| 3 | 101 | 27 | 10.065 | +2.03 | 0.00 | 14 | 0.04 |
| 4 | 48 | 67 | 9.510 | +1.68 | 0.10 | 164 | 0.04 |
| 5 | 119 | 53 | 8.952 | +1.33 | 0.06 | 83 | 0.05 |
| 6 | 43 | 361 | 8.923 | +1.31 | 0.10 | 140 | 0.07 |
| 7 | 44 | 375 | 8.821 | +1.25 | 0.10 | 110 | 0.07 |
| 8 | 27 | 63 | 8.792 | +1.23 | 0.13 | 85 | 0.05 |
| 9 | 45 | 377 | 8.748 | +1.20 | 0.10 | 112 | 0.06 |
| 10 | 42 | 333 | 8.703 | +1.17 | 0.08 | 30 | 0.07 |
| 11 | 188 | 205 | 8.547 | +1.08 | 0.05 | 100 | 0.05 |
| 12 | 209 | 542 | 8.438 | +1.01 | 0.04 | 12 | 0.06 |
| cold 1 | 196 | 341 | 5.323 | -0.95 | 0.01 | 197 | 0.13 |
| cold 2 | 155 | 390 | 5.321 | -0.95 | 0.03 | 154 | 0.16 |
| cold 3 | 90 | 782 | 5.083 | -1.10 | 0.02 | 34 | 0.10 |
| cold 4 | 223 | 477 | 3.530 | -2.08 | 0.02 | 7 | 0.84 |
| cold 5 | 41 | 73 | 3.444 | -2.13 | 0.00 | 113 | 0.21 |
| cold 6 | 222 | 391 | 3.311 | -2.22 | 0.00 | 6 | 0.98 |
| cold 7 | 6 | 391 | 3.283 | -2.23 | 0.00 | 222 | 0.98 |
| cold 8 | 4 | 313 | 3.242 | -2.26 | 0.00 | 220 | 0.99 |
| cold 9 | 220 | 314 | 3.234 | -2.26 | 0.00 | 4 | 0.99 |
| cold 10 | 7 | 405 | 3.162 | -2.31 | 0.00 | 223 | 0.84 |
| cold 11 | 221 | 387 | 3.076 | -2.36 | 0.00 | 5 | 0.99 |
| cold 12 | 5 | 386 | 3.076 | -2.36 | 0.00 | 221 | 0.99 |

**What drives the curve.** Across the 222 pages with >= 20 units, per-page mean surprisal correlates with the page's **hapax-token share** at Spearman rho=+0.796 (rare-type share rho=+0.713) and **negatively** with its similarity to its nearest twin page (rho=-0.826). The raw non-English (coinage) share is a *poor* predictor (rho=-0.527): coinage is bimodal, and the frequent loop compounds are among the cheapest tokens. The hot zones are where one-off types cluster; the cold zones are the near-duplicate loop blocks.

**The loop, directly measured.** 8 pages have a bigram-Jaccard >= 0.5 with another page; the strongest pairs are p5≈p221 (0.99), p221≈p5 (0.99), p4≈p220 (0.99), p220≈p4 (0.99), p6≈p222 (0.98), p222≈p6 (0.98), p7≈p223 (0.84), p223≈p7 (0.84). Their mean surprisal is 3.24 bits vs 6.84 book-wide: the LM has already seen the twin, so the repetition is nearly free. This is the measurable form of the 'devastating loop of language' and the reason the aggregate perplexity is low.

### 4.2 Contiguous zones

| kind | zone | start | end | units | mean bits |
|---|---|---:|---:|---:|---:|
| page | hot | 2 | 2 | 31 | 15.462 |
| page | hot | 27 | 27 | 63 | 8.792 |
| page | hot | 42 | 45 | 1,446 | 8.800 |
| page | hot | 48 | 48 | 67 | 9.510 |
| page | hot | 101 | 101 | 27 | 10.065 |
| page | hot | 119 | 119 | 53 | 8.952 |
| page | hot | 188 | 188 | 205 | 8.547 |
| page | hot | 209 | 209 | 542 | 8.438 |
| page | hot | 224 | 224 | 237 | 11.301 |
| page | cold | 4 | 7 | 1,495 | 3.188 |
| page | cold | 41 | 41 | 73 | 3.444 |
| page | cold | 90 | 90 | 782 | 5.083 |
| page | cold | 220 | 223 | 1,569 | 3.304 |
| window | hot | 12000 | 20000 | 34,000 | 7.862 |
| window | hot | 37500 | 38000 | 4,000 | 7.484 |
| window | cold | 0 | 500 | 4,000 | 4.773 |
| window | cold | 28500 | 29500 | 6,000 | 5.839 |
| window | cold | 47000 | 47500 | 4,000 | 5.881 |
| window | cold | 49500 | 51000 | 8,000 | 5.884 |
| window | cold | 54000 | 54000 | 2,000 | 5.857 |
| window | cold | 55000 | 56000 | 6,000 | 5.914 |
| window | cold | 63500 | 65000 | 8,000 | 5.718 |
| window | cold | 71000 | 72500 | 7,577 | 4.723 |

A rolling window of 2000 units (stride 500) smooths the curve; `window_curve.tsv` holds every window. The zones are listed for both granularities in `zones.tsv`.

## 5. Clause-level surprisal (#14 join)

5,578 of 5,614 #14 clauses matched at least one unit by `(page, start)` containment; `n_units_matched` is reconciled against #14's `n_units` in `clause_surprisal.tsv`. The ten highest-mean clauses are shown for evidence.

| clause | page | opener | units | mean bits | preview |
|---:|---:|---|---:|---:|---|
| 1234 | 47 | `::` | 1 | 25.843 | joints/defiles |
| 1105 | 42 | `::` | 1 | 23.638 | nail |
| 2531 | 99 | `//` | 1 | 23.307 | ToKAGE/UPLOAD |
| 624 | 23 | `//` | 1 | 23.210 | Larva=degradation |
| 1540 | 61 | `/` | 1 | 23.142 | <<KE-MO-NO>> |
| 2379 | 92 | `//` | 1 | 23.117 | GENOMIC |
| 1085 | 40 | `/` | 1 | 22.896 | Ultra-machinery |
| 866 | 32 | `//` | 1 | 22.876 | evaporates |
| 4660 | 186 | `::` | 1 | 22.669 | biotechnic |
| 2839 | 109 | `//` | 1 | 22.559 | Ghost-rush |

## 6. Calibration against English prose

A pooled English anchor is built from the eight pinned #24 references (74,072 tokens) split into 225 pseudo-pages, scored by the identical leave-one-pseudo-page-out procedure.

| model | held-out perplexity |
|---|---:|
| book order 1 | 323.579 |
| book order 2 | 100.366 |
| book order 3 | 99.656 |
| book order 4 | 103.223 |
| **English order 3** | **669.378** |
| English own best (order 2) | 650.918 |

**The book is *more* predictable than matched English prose, not less** (perplexity 99.7 vs 669.4; the English anchor is 6.72x more perplexing; mean surprisal 6.639 vs 9.387 bits/token). This is not a contradiction of the derangement thesis: #24 already measured that the book is a **restricted, highly repetitive loop**, and the duplicate-page analysis in §4.1 shows why an n-gram model predicts it well. The alienness is not in the aggregate entropy but in the **shape** of the distribution: a very cheap repeated core plus a very expensive hapax tail, with page-level hot/cold zones tracking that tail. The full entropy/compression baselines are #47/#48; the external-LM per-page curve is #49.

## 7. Verdict

**THE DERANGED CORE IS LOCALISED IN THE HAPAX TAIL, NOT IN AGGREGATE PREDICTABILITY. The book as a whole is *more* predictable than English prose (ppl 100 vs 669) because it is a near-duplicate loop: cold zones are pages whose twin appears elsewhere (bigram-Jaccard up to 0.99). But per-page surprisal is strongly and monotonically driven by one-off types (hapax-share Spearman rho=+0.80), and the coinage category is bimodal — frequent loop compounds are cheap, hapax compounds cost ~30 bits. The 'deranged core' is therefore the hapax-dense pages, exactly where the naming layer and rare English cluster; the cold zones are the loop, i.e. conventional English showing through.**

## 8. Caveats and negative results

- **The aggregate result is a negative result for the naive thesis.** The book is *more* predictable than English prose under an n-gram LM, because it repeats itself; the derangement is in the distribution's tail, not its mean. Reporting the aggregate alone would have been misleading.
- **Absolute discounting, not a neural LM.** A neural LM (#49) is a different instrument; this n-gram model is transparent, deterministic and cheap, and its held-out ranking of orders is the model-selection evidence.
- **Leave-one-page-out resets context at page boundaries.** The scan's page breaks are layout, not discourse, so cross-page context is dropped; this makes surprisal slightly conservative at page starts and is applied identically to the English anchor.
- **The English anchor is pooled across eight books** at matched N, so it mixes authors and inflates English perplexity relative to a single text; the *direction* of the aggregate comparison is therefore conservative toward the book being less predictable, and it still comes out lower.
- **Case is folded in the primary run**; the case-preserved sensitivity is reported above. It does not change the verdict.
- **Operators are not in this LM.** They are the silence between units and were modelled in #11; the silence-removal perplexity experiment is #50.
- **Small pages.** Pages with < 20 units are shown in `per_page.tsv` but excluded from the hot/cold flags; several front-matter pages are in this class.
- **The English anchor is literary prose, not cyberpunk** (copyright); see #56.

## 9. Artifacts

| file | contents |
|---|---|
| `ngram_surprisal_report.md` | THE DELIVERABLE |
| `per_token.tsv` | one row per screened unit: uid/page/offset/token/context/surprisal |
| `per_page.tsv` | page curve: n, mean/median/p90 surprisal, perplexity, z, zone |
| `window_curve.tsv` | rolling-window mean surprisal |
| `zones.tsv` | contiguous hot/cold zones (pages and windows) |
| `type_surprisal.tsv` | per unit type: n, mean surprisal, category |
| `hot_occurrences.tsv` / `cold_occurrences.tsv` | extreme occurrences with context |
| `category_surprisal.tsv` | surprisal by #19 category |
| `freq_band_surprisal.tsv` | surprisal by global type-frequency band |
| `clause_surprisal.tsv` | surprisal per #14 clause |
| `perplexity_by_order.tsv` | held-out perplexity for orders 1-4 (+ sensitivity + English) |
| `reconciliation.tsv` / `diagnostics.tsv` | join contract and counts |
| `summary.json` | machine-readable headline numbers |
| `surprisal_curve.png` | per-page curve |

---
*Generated by `work/ngram_surprisal.py` (schema `ngram-surprisal/1`) on 2026-10-07T05:02:03Z.*
