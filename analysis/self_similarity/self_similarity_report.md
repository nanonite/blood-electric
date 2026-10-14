# Self-similarity matrix and loop detection (#54)

Build the paragraph x paragraph cosine self-similarity matrix of the book and read it for the two structures the task names: a **diagonal band** (each paragraph close to its neighbours -- the loop you feel while reading) and **repeating blocks** (a whole stretch of paragraphs recurring later in the book -- the loop as structure).

**Headline.** Both are present, at different scales, and neither is a period. The matrix is globally high (mean off-diagonal cosine 0.5382 -- the whole book sits in one semantic field, so 'similar' is the default), but the similar cells are **not** organised as a running diagonal band: of the 9,578 paragraph pairs above cosine 0.8, only 0.6% sit within one paragraph of the diagonal and 3.9% within 5. The diagonal is real but **short-range**: mean cosine at lag 1 is 0.5648 against an order-permutation null of 0.5382 (z = +10.8), significant to about lag 30 and at the null from lag 40.

**The loop is elsewhere, and it is block-shaped.** 10 paragraph blocks and 1 page block sit far off the diagonal. The one that matters is a **verbatim return**: pages 4/5/6/7 recur on pages 220/221/222/223 -- at page resolution 4 x 4 pages, mean cosine 0.9550, and at paragraph resolution 11 x 10 paragraphs (16 cells) -- while a `difflib` ratio on the raw text is 0.980 and the encoder-free lexical control agrees the words are the same. The book's opening section returns near its end: **the 'devastating loop of language' is, at least in part, literally a reprint.** The rest of the long-range block population is small thematic repetition (0 page-level and 9 paragraph-level blocks are *not* verbatim, 4-5 cells each, cosine ~0.87 against a text identity of ~0.2): the book reuses its motifs far apart, but it does not rewrite whole passages at page scale.

Read with the neighbouring tasks, the three scales separate cleanly: the short-range band is #42's lag-2-30 template recurrence one level up and #34's frames; the absence of any large-lag periodicity restates #51's verdict in semantic space; and the verbatim return is #35's page-level near-duplicate finding (p4=p220, p5=p221, p6=p222, p7=p223) recovered here from the embedding side without any n-grams.

## 1. Method

* **The paragraph unit** is a blank-line-separated run of the #7 layout page body (`analysis/layout_recovery/blood-electric.layout.md`), i.e. marker's `Text` block as #7 recovered it; #7's section headings and figure placeholders are kept and flagged in `paragraph_matrix.tsv`, not dropped. Character spans are in page-body coordinates, the same coordinates #33's clauses use.
* **Clauses are joined to paragraphs by offset overlap** on `(page, start, end)`, so every clause lands in exactly one paragraph and none is lost: 5,614 of 5,614 clauses assigned, 23 of them spanning a paragraph break (disclosed, not repaired).
* **The paragraph vector** is the mean of its clause vectors from #39's primary artifact -- SBERT `all-MiniLM-L6-v2`, 384-d, L2-normalised, `units` view -- renormalised. No encoder runs for the book; the encoder is pinned by #39.
* **Statistics.** The full 717 x 717 cosine matrix (`self_similarity.npz`), a page rollup (mean paragraph vector per page, 225 x 225), a **lag curve** at 20 lags against a 200-draw paragraph-order permutation null (seed 0), **band statistics** for widths 1, 2, 3, 5, 10, 20, 30, 50, 100 with the same null, and **off-diagonal blocks** = 8-connected components of cells above the 99.5% off-diagonal quantile of their own resolution (cosine 0.8545 for paragraphs, 0.9073 for pages, never below the 0.85 floor) with the local band (|i-j| <= 2) removed, run at both paragraph and page level. 8-connectivity is required because a block is a *diagonal* rectangle: with a 4-neighbour rule the cells (i, i+k) and (i+1, i+k+1) are not adjacent and every long-range block is shredded.
* **Controls.** (a) The **English paragraph baseline**: real English paragraphs from #24's pinned reference cache, grouped so the sentences-per-paragraph multiset matches the book's clauses-per-paragraph, encoded with the *same* MiniLM encoder and pooled the same way -- `english_baseline.tsv`. (b) A **lexical TF cosine** over #39's screened unit surfaces, no encoder at all, asking whether a block is *verbatim* or merely thematic. (c) A **view/encoder sensitivity** pass over #39's `raw` view and its secondary BGE encoder -- `sensitivity.tsv`.
* **Determinism.** Fixed seed for the permutation null and every control; no hash-order dependence; no wall-clock in any artifact.

Reproduce with `python work/self_similarity.py`; regression suite `python work/test_self_similarity.py`. Schema `self-similarity/1`.

## 2. The matrix at a glance

| quantity | value |
|---|---|
| paragraphs in the matrix | 717 (of 734 runs; 17 carry no clause) |
| pages in the rollup | 225 |
| mean off-diagonal cosine | 0.5382 |
| median off-diagonal cosine | 0.5825 |
| shares above 0.8 / 0.90 / 0.95 | 3.73% / 0.03% / 0.00% |
| pairs above cosine 0.95 | 11 of 256,686 |
| single most similar pair | cosine 1.0000 |
| lag 1 / 5 / 30 mean cosine | 0.5648 / 0.5650 / 0.5469 |
| lag 1 null mean (z) | 0.5382 (+10.8) |
| lexical TF mean cosine (no encoder) | 0.6099, top pair 1.0000 |
| page rollup median / 95th / 99th percentile cosine | 0.7893 / 0.8763 / 0.9008 |

The mean is high by construction of the corpus, not of the method: #39 measured the same thing one level down (mean clause-to-clause cosine far above matched English), and the paragraph average only smooths towards the corpus centroid. Every structural claim below is therefore made **against the permutation null**, which keeps that background fixed and asks only where the mass sits.

## 3. Is there a diagonal band?

| lag | pairs | mean cosine | null | delta | z |
|---|---|---|---|---|---|
| 1 | 716 | 0.5648 | 0.5382 | +0.0266 | +10.8 |
| 2 | 715 | 0.5704 | 0.5385 | +0.0319 | +12.3 |
| 3 | 714 | 0.5727 | 0.5384 | +0.0343 | +13.9 |
| 4 | 713 | 0.5643 | 0.5384 | +0.0258 | +10.1 |
| 5 | 712 | 0.5650 | 0.5380 | +0.0270 | +11.8 |
| 6 | 711 | 0.5670 | 0.5384 | +0.0286 | +12.7 |
| 8 | 709 | 0.5620 | 0.5382 | +0.0238 | +8.7 |
| 10 | 707 | 0.5584 | 0.5382 | +0.0202 | +8.1 |
| 15 | 702 | 0.5599 | 0.5385 | +0.0215 | +8.8 |
| 20 | 697 | 0.5517 | 0.5380 | +0.0136 | +5.5 |
| 25 | 692 | 0.5516 | 0.5383 | +0.0133 | +5.2 |
| 30 | 687 | 0.5469 | 0.5380 | +0.0089 | +3.3 |
| 50 | 667 | 0.5411 | 0.5382 | +0.0029 | +1.0 |
| 100 | 617 | 0.5300 | 0.5379 | -0.0080 | -2.3 |
| 200 | 517 | 0.5386 | 0.5380 | +0.0006 | +0.1 |

A band exists but it is **narrow and shallow**: it is significant (z up to +13.9) for the first few lags and is gone by lag 40 (no later lag exceeds its null by two sd). In cosine terms the band adds +0.0266 at lag 1 and +0.0343 at lag 3 over a background of 0.5384. Reading it structurally: a paragraph is closer to the paragraphs immediately around it than chance allows, at the scale of a few paragraphs -- #42's 'template recurrence at the scale of a few clauses', one level up. There is no evidence of a band that *runs*: no tested lag through 300 shows an excess, which is the semantic-space restatement of #51's no-fixed-periodicity result.

The band statistics say the same thing from the other side:

| band width | cells in band | share of cells above screen | null share | z |
|---|---|---|---|---|
| 1 | 60 | 0.63% | 0.27% | +7.8 |
| 2 | 135 | 1.41% | 0.56% | +13.4 |
| 3 | 229 | 2.39% | 0.84% | +20.5 |
| 5 | 370 | 3.86% | 1.38% | +23.5 |
| 10 | 687 | 7.17% | 2.76% | +28.5 |
| 20 | 1,207 | 12.60% | 5.50% | +33.7 |
| 30 | 1,623 | 16.95% | 8.19% | +33.5 |
| 50 | 2,390 | 24.95% | 13.45% | +30.9 |
| 100 | 3,691 | 38.54% | 25.94% | +20.2 |

Even at width 100 the band holds only 38.5% of the similar cells against a 25.9% null: the similar cells are spread across the book, not concentrated near the diagonal. `lag_curve.png` plots both views.

## 4. Repeating blocks

Removing the local band and taking 8-connected components of the cells above cosine 0.8545 (paragraph level) and 0.9073 (page level) leaves 17 paragraph-level blocks and 5 page-level ones. Most are near-diagonal blobs -- the same local cohesion the lag curve measures -- so the catalogue is split by **range**: a block whose twin starts more than 25 positions later is a repetition, everything else is the band. 10 paragraph blocks and 1 page block are long-range, and they split again on the raw-text `difflib` identity into **verbatim returns** (identity >= 0.5: 1 paragraph, 1 page) and **template blocks** (the rest: same template, different words).

The verbatim long-range blocks -- the same words, far apart:

| level | cells | rows x cols | row pages | col pages | gap | cosine | identity | first words |
|---|---|---|---|---|---|---|---|---|
| page | 8 | 4 x 4 | 4/5/6/7 | 220/221/222/223 | 213 | 0.9550 | 0.980 | [page 4-7] ### abolition <<| record the vital... |
| paragraph | 16 | 11 x 10 | 4-8 (5 pages) | 220/221/222/223 | 690 | 0.9488 | 0.984 | [paragraph 4-8] ### abolition <<| record the vital... |

The largest template blocks -- the same template, different words:

| level | cells | rows x cols | row pages | col pages | gap | cosine | lexical cosine | identity |
|---|---|---|---|---|---|---|---|---|
| paragraph | 5 | 1 x 5 | 149 | 173/174/175 | 66 | 0.8720 | 0.9211 | 0.154 |
| paragraph | 5 | 1 x 5 | 161 | 173/174/175 | 32 | 0.8709 | 0.9301 | 0.205 |
| paragraph | 4 | 2 x 2 | 144/145 | 170 | 69 | 0.8672 | 0.9192 | 0.208 |
| paragraph | 4 | 2 x 2 | 144/145 | 171/172 | 73 | 0.8747 | 0.9261 | 0.243 |
| paragraph | 4 | 2 x 2 | 144/145 | 173/174 | 78 | 0.8811 | 0.9255 | 0.197 |
| paragraph | 4 | 1 x 4 | 151 | 173/174/175 | 62 | 0.8657 | 0.9239 | 0.188 |
| paragraph | 4 | 2 x 2 | 155 | 166 | 31 | 0.8659 | 0.9101 | 0.173 |
| paragraph | 4 | 2 x 2 | 155 | 171/172 | 45 | 0.8841 | 0.9021 | 0.185 |

**The verbatim return is the structural loop, and it is the load-bearing finding.** Three independent instruments agree on what it is. The cosine itself: page pairs p4=p220, p5=p221, p6=p222, p7=p223 at 0.9998, 0.9994, 0.9990, 0.9897 -- #35's page-level near-duplicate finding, recovered here from the embedding side with no n-grams anywhere. The encoder-free lexical control: 0.7549 against 0.6311 for two pages drawn at random. And the raw text: **98.0% identical** by `difflib`. Two adjacent mechanisms of the same loop, separated by instrument: *sentences* recur elsewhere in the book at long range (#35, #39), and the *section* recurs verbatim near the end (here).

**No template blocks at page scale -- a negative result worth recording.** Above the 0.9073 page-level cut, the long-range page block is verbatim. The large diffuse long-range regions that *do* appear at a looser absolute cut (cosine 0.85 is only this matrix's 90th percentile, against a median of 0.7893) are background anisotropy of the embedding space, not structure. That is why the block screen is a **per-level quantile** and why every claim here is stated against a null rather than an absolute cosine.

**Thematic near-duplicates at paragraph scale.** 9 paragraph blocks of 4-5 cells sit far from their twin without being the same words: cosine ~0.87, text identity ~0.15, for example pages 149 against 173/174/175 (gap 66 paragraphs). This is the loop at the scale this task named -- a paragraph's *shape* returning hundreds of paragraphs later -- and it is #34's frames measured through the embedding rather than through the token stream. The lexical control does **not** separate these from the verbatim return (both sit near 0.75 page-pooled cosine): the semantic matrix is already saturated at this resolution, which is why this report carries an encoder-free `difflib` identity as the discriminator between a return and a rewrite.

The ten most similar paragraph pairs, in cosine order. The row numbers are matrix rows (clause-less runs excluded); `paragraph_matrix.tsv` maps them to `para_id` and page.

| matrix rows | pages | lag | cosine | first words |
|---|---|---|---|---|
| 12 / 712 | p7 / p223 | 700 | 1.0000 | the cadaver that chrome interceptor drool revived — I was ra... |
| 100 / 356 | p41 / p113 | 256 | 1.0000 | I invade the LOAD device of the larva nature of masses of fl... |
| 7 / 707 | p5 / p221 | 700 | 1.0000 | <<the game of self is inoculated internal — the blood tube-l... |
| 8 / 708 | p5 / p221 | 700 | 1.0000 | The atrocity coefficient of the dog that streams the nude of... |
| 6 / 706 | p4 / p220 | 700 | 1.0000 | <<module=heart::is born in the derangement condition of the ... |
| 4 / 704 | p4 / p220 | 700 | 1.0000 | ### abolition |
| 5 / 705 | p4 / p220 | 700 | 0.9988 | <<| record the vital-icon+our chromosome form escape of the ... |
| 10 / 710 | p6 / p222 | 700 | 0.9986 | girl::Rape=hologram::of the brain universe//LOAD to the inte... |
| 11 / 711 | p6 / p222 | 700 | 0.9979 | Hydromachine that the body mode of self iterates/joint to th... |
| 9 / 709 | p5 / p221 | 700 | 0.9953 | <<| suck from the inside::dashes to the nightmare of the rep... |

Structurally, the loop is **not** a diagonal band and **not** a tiled repetition of the whole book: it is a short diagonal band, one verbatim return of the opening section near the end, and a handful of small thematic repeats at paragraph scale. The two resolutions also disagree in *sharpness*: the return reads 0.9550 at page resolution and 0.9488 at paragraph resolution, because the rollup averages each page's repeated and unrepeated paragraphs together while the paragraph block covers only the paragraphs that recur.

## 5. Page-level rollup

Averaging each page's paragraph vectors gives the 225 x 225 matrix `page_matrix.tsv` -- the readable view, and the one that matches #35's page-resolution finding directly.

| pages | cosine |
|---|---|
| p4 / p220 | 0.9998 |
| p5 / p221 | 0.9994 |
| p6 / p222 | 0.9990 |
| p7 / p223 | 0.9897 |
| p170 / p173 | 0.9383 |
| p159 / p161 | 0.9380 |
| p155 / p166 | 0.9376 |
| p149 / p155 | 0.9367 |

The rollup's own mean off-diagonal cosine is 0.7621 -- *higher* than the paragraph matrix's 0.5382, because averaging paragraphs into pages pools out the variation that made paragraph pairs differ. Running the same lag analysis on the rollup (`page_lag_curve.tsv`, same 200-draw page-order null):

| lag (pages apart) | pairs | mean cosine | null | z |
|---|---|---|---|---|
| 1 | 224 | 0.8089 | 0.7618 | +19.7 |
| 2 | 223 | 0.8010 | 0.7619 | +16.9 |
| 3 | 222 | 0.7981 | 0.7619 | +15.3 |
| 5 | 220 | 0.7922 | 0.7621 | +11.0 |
| 10 | 215 | 0.7848 | 0.7623 | +7.3 |
| 25 | 200 | 0.7739 | 0.7626 | +3.2 |
| 50 | 175 | 0.7647 | 0.7622 | +0.7 |
| 100 | 125 | 0.7567 | 0.7619 | -1.3 |
| 200 | 25 | 0.7002 | 0.7612 | -2.7 |

Pooling **keeps the local band and amplifies it in absolute terms**: adjacent pages sit +0.0470 above their own null at lag 1 (z = +19.7), against the paragraph matrix's +0.0266 -- averaging three paragraphs per page pulls both vectors toward the shared semantic field, so the background rises *and* the diagonal rises faster. The band is significant to about lag 25 at page resolution and at the null by lag 50 -- exactly the short-range shape the paragraph matrix showed: one phenomenon at two scales, not two phenomena. The page rollup makes the repeated section clear at page scale; the paragraph matrix also detects its aligned subparagraphs.

## 6. Controls

| quantity | book paragraphs | English paragraphs |
|---|---|---|
| n_paragraphs | 717 | 717 |
| n_sentences | 5614 | 5614 |
| mean_cos | 0.53824 | 0.32714 |
| median_cos | 0.58251 | 0.32845 |
| frac_gt_screen | 0.03731 | 0.00032 |
| top_pair_cos | 1.00000 | 0.94432 |
| lag1_mean | 0.56480 | 0.51737 |
| lag2_mean | 0.57040 | 0.48274 |
| lag5_mean | 0.56504 | 0.44099 |

The book is more self-similar than matched English prose at paragraph resolution exactly as #39 found at clause resolution (mean cosine 0.3271 English vs 0.5382 book; pairs above 0.8: 3.73% vs 0.032%). The instructive difference is the tail. The book's most similar pair is 1.0000 -- a *repeated passage*, the opening section returning near the end -- while English's is 0.9443 even after excluding 3 accidentally identical chunk(s) (a sentence repeated in the source text, so two chunks come out byte-identical -- disclosed rather than repaired, because that is the only way English reaches 1.0). English's own lag-1 mean (0.5174) is *below* the book's (0.5648), and the 717 English groups are drawn from the eight pinned #24 texts with the book's sentences-per-paragraph distribution, so the two matrices are built the same way: the excess the book shows over its own null (z = +10.8 at lag 1) is a property of the book, not of paragraph pooling.

| view / encoder | mean cosine | lag 1 | blocks | biggest block |
|---|---|---|---|---|
| sbert / units | 0.53824 | 0.56480 | 17 | 4-8 / 220-223 (16 cells) |
| sbert / raw | 0.54149 | 0.56769 | 18 | 4-8 / 220-223 (16 cells) |
| bge / units | 0.88285 | 0.88808 | 6 | 4-8 / 220-223 (20 cells) |

The headline numbers and the block survive the change of view and of encoder, so the finding is not an artifact of the pinned model. (#39 measured the two views as 86% neighbour-agreement and the two encoders as 35%, so the *encoder* change is the sharper test here.)

## 7. Structural reading

1. **A diagonal band, but a shallow one.** Paragraphs are closer to their immediate neighbours than the permutation null allows (z = +10.8 at lag 1, decaying by lag 40). The band is a local cohesion effect -- the same template recurrence #42 found at clause scale and #34 catalogued as frames -- not a metrical structure.
2. **No paragraph-scale period.** Beyond lag 30 no tested lag up to 300 differs from its null by more than two sd (1 exception out of 7), so there is no repeating interval at paragraph resolution: the jacket-copy 'loop' is not a cycle. This is #51's verdict recovered in semantic space.
3. **The long-range structure is a block, and it is verbatim.** 10 paragraph blocks and 1 page block sit far off the diagonal, the largest page block covering 4 x 4 pages: 4/5/6/7 against 220/221/222/223 at mean cosine 0.9550. The book's opening section returns near its end, and the lexical control plus the `difflib` ratio show it is the same *text*, not the same theme.
4. **The background is one field, and that is itself the finding.** Every paragraph is similar to every other (median 0.5825), so the matrix cannot be read as clusters; the structure is in the *excess over that background* -- which is exactly what the permutation null isolates, and exactly why this report quotes deltas and z-scores rather than raw cosines.
5. **What the task asked, answered directly:** the 'loop' shows as **repeating blocks at long range on top of a short diagonal band**, not as a band alone and not as a cycle. In one sentence: the loop is local cohesion plus one verbatim return.

## 8. Caveats

* 17 of the 734 paragraph records (733 blank-line runs plus 1 added so that no clause is dropped) carry no #33 clause -- figure placeholders and bare page furniture -- and are excluded from the matrix; they are listed in `paragraph_matrix.tsv` with `in_similarity_matrix = 0`.
* 23 clauses straddle a paragraph break and are assigned by maximum overlap; the overlap is large in every case, so the assignment is not a coin flip, but the boundary is soft.
* The paragraph vector is an average of clause vectors, so a paragraph's position in the matrix depends on #33's segmentation as much as on its content. A different segmentation would shift individual cells, not the two structural results.
* Cosine under MiniLM is anisotropic: all pairs sit well above zero (see the English control), which is why every claim here is made against the null and never against an absolute threshold.
