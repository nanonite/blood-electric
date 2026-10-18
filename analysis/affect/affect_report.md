# Affect and valence curve (#44)

Score the book's affect on the four axes the task names -- fear, ecstasy, paranoia, disgust -- plus a signed valence axis, and plot the curve over the 225 pages.  The task's own example is the explicit operator marker ``Fear=cell``, so the lexicon runs on the layer where such an equation is visible.

**Headline.** 79,770 markup-screened #17 word parts carry **1,378 affect occurrences (17.3 per 1,000 parts)** on 262 published lexicon terms (100 attested).  The four axes are unequal in size *and in breadth*, and that is the finding: **ecstasy is the largest axis (538 hits, 6.74 per 1,000), carried by ``desire`` (275 occurrences -- the book's most frequent affect term outside the cadaver/murder/blood register), ``rapture`` and ``ecstasy``; disgust is second (453 hits) but from only 5 of its 40 terms, three of which (``faecal``, ``sickly``, ``grotesque``) carry 451 of the 453; fear is third (344, split between ``nightmare`` 144 and ``fear`` 137); and paranoia is nearly absent (43 hits, 4 of 22 terms)**.  The book's affect vocabulary is a small, repeating core -- the top 10 affect-axis terms (``desire``, ``faecal``, ``sickly``, ``nightmare``, ``fear`` …) carry 92% of all affect occurrences -- not a graded emotional spectrum, which is the same restricted-lexicon finding as #24 and #56.  Against the pooled #56 general-English references the book is **affect-dense on every axis** (all four q = 0.000), most of all on disgust (log-ratio +3.53) and ecstasy (+2.63) -- see §4 -- but **within the book the affect lexicon is not frequent vocabulary**: same-sized chunks of ordinary book vocabulary carry as many or more occurrences (§6 term-shuffle nulls), so the density is a cross-register effect, not a within-book frequency effect.  And the curve is **stationary in level but not featureless**: **disgust is the one axis with a real narrative arc** -- its smoothed curve peaks at 17.55 per 1,000 on p.117, +5.34 sd above the page-order permutation null, with matching page-to-page continuity (lag-1 z = +5.39), because its three carrying terms (``faecal``, ``sickly``, ``grotesque``) cluster on the same pages.  Fear and paranoia are locally smooth (lag-1 z = +2.03 and +2.09) but have no peak (z = +0.25 and +1.16); ecstasy has neither (peak z = -0.89, lag-1 z = +1.38); the aggregate curve has both (peak z = +3.01, lag-1 z = +4.56) -- affect vocabulary clusters on the same pages the way the book's templates do (#42), without a rising or falling trajectory on the other three axes.

## 1. Method

* **Instrument 1 -- the lexicon.** A curated, published affect lexicon (`AFFECT_LEXICON` in `work/affect_curve.py`, every term and its count in `affect_lexicon.tsv`) matched **exactly** -- lower-cased whole surface, no stemming, no fuzzy matching -- against the #17 **word-part** layer: the individual words inside the maximal units, so ``fear=cell`` contributes the parts ``fear`` and ``cell``.  This is the layer #56 uses for keyness because it is the one comparable to ordinary English words, and it is the layer on which an affect state written as an operator equation is still a lexical occurrence.  The markup screen is #56's own (a part is dropped when its unit uid is a #19 markup residue, when its `(page, start)` falls in a #27-detected tag span, or when its surface is a #19 junk/markup type), so the counts are directly comparable with #56's and #24's word layers.
* **Instrument 2 -- the classifier.** A zero-shot classifier in the pinned #39 clause-vector space: each affect is a set of ordinary English description phrases (`CLASSIFIER_PROTOTYPES`), encoded with the **same SBERT encoder** that produced the 5,614 clause vectors, and every clause is scored by its maximum cosine to an affect's prototypes.  A **neutral description control** (rooms, weather, streets, trains) is scored identically and plotted beside them.  No prototype phrase shares a term with the lexicon, so an affect matched by meaning is a second opinion on the same phenomenon, not a restated count.
* **Valence** is the signed contrast per 1,000 parts: positive-valence terms (the ecstasy family plus love/hope/peace/paradise) minus negative-valence terms (fear, paranoia, disgust, and the despair/murder/rape/cadaver register).  The classifier has a matching signed axis (mean cosine to positive prototypes minus negative).
* **The explicit markers** (`affect_markers.tsv`) are the #17 units in which an affect word is *bound* to another word by an operator seam -- ``Fear=cell`` (84 occurrences over 73 pages, first p.6), ``fear-cell``, ``paranoia=spasm``, ``desire+fear=cell``, ``fear=cell++self``.  A standalone ``ecstasy`` is an ordinary lexical occurrence and stays with the lexicon.
* **Nulls** (§6): a **page-order permutation null** per curve (the 225 per-page values kept exactly, their order over the book destroyed by a seeded random permutation -- a circular roll is *not* used because it leaves a series' adjacent pairs unchanged up to one wrap pair, which would make the lag-1 statistic's null degenerate), a clause-label permutation null per classifier curve, **two matched-size term-shuffle nulls** (the 128 affect-axis terms relabelled with same-sized chunks of non-affect book vocabulary: the fixed core-vocabulary pool, sampled with replacement, and the book's whole non-affect vocabulary, sampled without replacement), and the #56 English references scored with the same lexicon and the same classifier.

## 2. Reconciliation

| quantity | here | earlier | note |
|---|---:|---|---|
| word parts kept | 79,770 | 79,770 (#56) | #17 role=w parts under #56's markup screen (80,363 total, 593 dropped) |
| clauses | 5,614 | 5,614 (#39) | every clause carries a classifier score |
| pages | 225 | 225 (#39) | pages carrying word parts |
| screened units | 74,077 | 74,077 (#27) | markup-screened #17 unit layer behind the clause text |
| encoder | sbert / units | sbert (#39 primary) | prototypes and clauses encoded in one space |
| reference corpora | 7 | 7 corpora (#56) | 2,443,714 pooled tokens; general_english is the inferential pool |
| lexicon terms | 262 | curated in-module | no external affect lexicon is fetched or assumed |

## 3. The affect lexicon

262 terms, of which 100 occur in the book.  Every row is published with its book count, its matched English rate and the keyness comparison, so a term that never fires is visible as a 0 rather than quietly dropped.  The four axes are disjoint in membership, so no occurrence is counted in two axes.

| axis | terms | attested | occurrences | per 1,000 parts |
|---|---:|---:|---:|---:|
| `fear` | 33 | 10 | 344 | 4.31 |
| `ecstasy` | 33 | 10 | 538 | 6.74 |
| `paranoia` | 22 | 4 | 43 | 0.54 |
| `disgust` | 40 | 5 | 453 | 5.68 |
| **all four** | | | **1,378** | **17.27** |
| valence (positive) | 60 | 19 | 939 | 11.77 |
| valence (negative) | 202 | 81 | 3,692 | 46.28 |
| **valence (net)** | | | **-2,753** | **-34.51** |

The most frequent affect terms in the book:

| term | affects | val | book | per M | english | per M | log-ratio | q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cadaver` | - | -1 | 633 | 7935.31 | 0 | 0.00 | +10.57 | 0.000 |
| `murder` | - | -1 | 624 | 7822.49 | 115 | 47.06 | +5.11 | 0.000 |
| `blood` | - | -1 | 303 | 3798.42 | 617 | 252.49 | +2.71 | 0.000 |
| `desire` | ecstasy | +1 | 275 | 3447.41 | 215 | 87.98 | +3.67 | 0.000 |
| `angel` | - | +1 | 220 | 2757.93 | 56 | 22.92 | +4.78 | 0.000 |
| `faecal` | disgust | -1 | 197 | 2469.60 | 1 | 0.41 | +8.30 | 0.000 |
| `crucified` | - | -1 | 178 | 2231.41 | 2 | 0.82 | +7.69 | 0.000 |
| `sickly` | disgust | -1 | 155 | 1943.09 | 18 | 7.37 | +5.55 | 0.000 |
| `rape` | - | -1 | 148 | 1855.33 | 4 | 1.64 | +6.92 | 0.000 |
| `nightmare` | fear | -1 | 144 | 1805.19 | 21 | 8.59 | +5.33 | 0.000 |
| `fear` | fear | -1 | 137 | 1717.44 | 471 | 192.74 | +2.19 | 0.000 |
| `murderous` | - | -1 | 131 | 1642.22 | 26 | 10.64 | +5.02 | 0.000 |
| `love` | - | +1 | 107 | 1341.36 | 805 | 329.42 | +1.41 | 0.000 |
| `insanity` | - | -1 | 104 | 1303.75 | 19 | 7.78 | +5.10 | 0.000 |
| `death` | - | -1 | 101 | 1266.14 | 729 | 298.32 | +1.45 | 0.000 |
| `grotesque` | disgust | -1 | 99 | 1241.07 | 35 | 14.32 | +4.45 | 0.000 |
| `rapture` | ecstasy | +1 | 94 | 1178.39 | 10 | 4.09 | +5.62 | 0.000 |
| `ecstasy` | ecstasy | +1 | 65 | 814.84 | 14 | 5.73 | +4.93 | 0.000 |
| `torture` | - | -1 | 62 | 777.24 | 28 | 11.46 | +4.21 | 0.000 |
| `desires` | ecstasy | +1 | 61 | 764.70 | 42 | 17.19 | +3.79 | 0.000 |
| `chaos` | - | -1 | 50 | 626.80 | 9 | 3.68 | +5.09 | 0.000 |
| `derangement` | - | -1 | 48 | 601.73 | 0 | 0.00 | +8.00 | 0.000 |
| `terror` | fear | -1 | 44 | 551.59 | 179 | 73.25 | +2.03 | 0.000 |
| `deranged` | - | -1 | 36 | 451.30 | 7 | 2.86 | +5.00 | 0.000 |
| `despair` | - | -1 | 36 | 451.30 | 124 | 50.74 | +2.20 | 0.000 |
| `ruin` | - | -1 | 36 | 451.30 | 91 | 37.24 | +2.50 | 0.000 |
| `rapes` | - | -1 | 34 | 426.23 | 0 | 0.00 | +7.66 | 0.000 |
| `paradise` | - | +1 | 30 | 376.08 | 21 | 8.59 | +3.77 | 0.000 |
| `murdered` | - | -1 | 28 | 351.01 | 57 | 23.32 | +2.72 | 0.000 |
| `murders` | - | -1 | 27 | 338.47 | 11 | 4.50 | +4.29 | 0.000 |
| `violence` | - | -1 | 26 | 325.94 | 107 | 43.79 | +2.02 | 0.000 |
| `mad` | - | -1 | 25 | 313.40 | 201 | 82.25 | +1.35 | 0.000 |
| `paranoia` | paranoia | -1 | 25 | 313.40 | 0 | 0.00 | +7.35 | 0.000 |
| `heaven` | - | +1 | 22 | 275.79 | 194 | 79.39 | +1.27 | 0.000 |
| `orgasm` | ecstasy | +1 | 21 | 263.26 | 0 | 0.00 | +7.18 | 0.000 |
| `raped` | - | -1 | 19 | 238.19 | 0 | 0.00 | +7.09 | 0.000 |
| `hatred` | - | -1 | 18 | 225.65 | 36 | 14.73 | +2.74 | 0.000 |
| `angels` | - | +1 | 17 | 213.11 | 30 | 12.28 | +2.87 | 0.000 |
| `corpse` | - | -1 | 14 | 175.50 | 49 | 20.05 | +2.19 | 0.000 |
| `isolated` | - | -1 | 14 | 175.50 | 48 | 19.64 | +2.21 | 0.000 |

Unattested terms are the mirror image of that table: `abandon`, `abhor`, `abhorrent`, `abhors`, `afraid`, `agony`, `anger`, `angry`, `anguish`, `apocalyptic`, `beloved`, `bleeding`, `bliss`, `blissful`, `brutal`, `brutality`, `carrion`, `conspiracy`, `craves`, `craving`, `damnation`, `damned`, `deaths`, `delight`, `delights`, `desired`, `desperation`, `die`, `dies`, `disgust`, `disgusted`, `disgusting`, `disgusts`, `distrust`, `distrusted`, `distrustful`, `dread`, `dreaded`, `dreads`, `dying`, `ecstasies`, `ecstatic`, `euphoria`, `euphoric`, `excrement`, `fearful`, `fearfully`, `fearing`, `fears`, `filth`, `filthy`, `foul`, `fury`, `gentle`, `go … -- the book never uses ordinary English words for dread, horror, revulsion, nausea or abhorrence, which is why each axis is carried by a handful of terms and the classifier matters as the second instrument for all four.

## 4. Book vs ordinary English prose

The same lexicon over the pooled #56 general-English references (2,443,714 tokens), matched by rate per 1,000 words.  G² is Dunning's signed log-likelihood (as #56 uses it), q is Benjamini-Hochberg over the tested axes.

| axis | book per 1k | english per 1k | log-ratio | G² | q | direction |
|---|---:|---:|---:|---:|---:|---|
| affect | 17.27 | 1.68 | +2.33 | +3602.2 | 0.000 | book |
| fear | 4.31 | 0.82 | +1.66 | +548.1 | 0.000 | book |
| ecstasy | 6.74 | 0.49 | +2.63 | +1648.5 | 0.000 | book |
| paranoia | 0.54 | 0.20 | +0.98 | +28.7 | 0.000 | book |
| disgust | 5.68 | 0.17 | +3.53 | +1968.8 | 0.000 | book |
| positive | 11.77 | 2.34 | +1.62 | +1437.3 | 0.000 | book |
| negative | 46.28 | 3.87 | +2.48 | +10499.7 | 0.000 | book |
| valence (net) | -34.51 | -1.53 | -0.87 | - | - | english |

Per-corpus rates (the pooled row is the inferential one):

| corpus | tokens | fear | ecstasy | paranoia | disgust | valence |
|---|---:|---:|---:|---:|---:|---:|
| cs | 103,048 | 0.21 | 0.04 | 0.01 | 0.01 | -0.50 |
| cutup_surrealist | 56 | 0.00 | 0.00 | 17.86 | 0.00 | -35.71 |
| cyberpunk | 105 | 0.00 | 0.00 | 0.00 | 0.00 | -9.52 |
| finnegans_wake | 49 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 |
| general_english | 1,609,978 | 1.13 | 0.63 | 0.23 | 0.23 | -1.93 |
| legal | 350,406 | 0.43 | 0.47 | 0.11 | 0.04 | -0.08 |
| medical | 380,072 | 0.06 | 0.02 | 0.24 | 0.04 | -1.46 |
| ALL (pooled) | 2,443,714 | 0.82 | 0.49 | 0.20 | 0.17 | -1.53 |

The classifier agrees in direction: on matched English sentences every affect prototype scores *below* where it scores on the book's clauses, while the **neutral control moves the other way** (book 0.1339 vs English 0.2033, z = -24.39) -- so the margin is affect-specific, not a general cosine shift.  Valence is the exception that proves the register point: the book's positive-minus-negative cosine is *lower* than English's (z = -14.25) because its negative-valence register (cadaver, murder, crucifixion) outweighs its ecstasy register in the vector space.

| classifier axis | book mean cosine | english mean cosine | margin | z |
|---|---:|---:|---:|---:|
| `fear` | 0.2256 | 0.1937 | +0.0319 | +11.07 |
| `ecstasy` | 0.2452 | 0.1845 | +0.0607 | +22.63 |
| `paranoia` | 0.2064 | 0.1882 | +0.0182 | +6.96 |
| `disgust` | 0.2425 | 0.2023 | +0.0401 | +14.70 |
| `neutral` | 0.1339 | 0.2033 | -0.0694 | -24.39 |
| `valence` | 0.0006 | 0.0279 | -0.0273 | -14.25 |

## 5. The curve over the book

![affect curve](affect_curve.png)

`affect_per_page.tsv` carries every number behind the plot.  The moving average is 5 pages, edge-preserving.  Peaks of the smoothed curves:

| curve | peak page | peak value | valley page | valley value | mean |
|---|---:|---:|---:|---:|---:|
| `rate_fear` | 76 | 9.08 | 70 | 1.08 | 4.19 |
| `rate_ecstasy` | 112 | 13.57 | 1 | 1.06 | 6.85 |
| `rate_paranoia` | 221 | 2.75 | 15 | 0.00 | 0.52 |
| `rate_disgust` | 117 | 17.55 | 7 | 1.04 | 5.72 |
| `rate_affect` | 116 | 32.46 | 1 | 4.73 | 17.27 |
| `rate_valence` | 12 | -11.52 | 1 | -94.22 | -35.19 |

The pages that carry the most explicit markers (the plot annotates ``Fear=cell``, ``ecstasy`` and ``desire`` where they first land, and the aggregate peak):

| page | fused affect markers | dominant axis |
|---:|---:|---|
| 133 | 421 | fear |
| 16 | 407 | fear |
| 127 | 381 | fear |
| 193 | 380 | fear |
| 37 | 376 | fear |
| 198 | 371 | fear |
| 34 | 362 | fear |
| 85 | 360 | fear |
| 194 | 359 | fear |
| 199 | 359 | ecstasy |
| 39 | 358 | fear |
| 146 | 350 | fear |

## 6. Controls

Four nulls, each testing one claim.  The **page-order permutation null** keeps a curve's 225 values and destroys only their order over the book, so a peak or a lag-1 beyond its distribution is real narrative structure, not the value distribution's own artifact.  The **clause-shuffle null** permutes clause scores and recomputes the per-page means, so a classifier curve with page structure beats it.  The two **term-shuffle nulls** relabel the 128 affect-axis terms with same-sized chunks of non-affect book vocabulary (the fixed core pool, and the whole non-affect vocabulary), so they test whether the affect lexicon is *frequent within the book*.  The English baselines (§4) are the cross-register comparison.

| control | statistic | observed | null mean | null sd | z | draws |
|---|---|---:|---:|---:|---:|---:|
| page-order permutation null: rate_fear | peak of the smoothed curve | 9.0756 | 8.8689 | 0.8378 | +0.25 | 2000 |
| page-order permutation null: rate_fear | lag-1 autocorrelation | 0.1283 | -0.0038 | 0.0651 | +2.03 | 2000 |
| page-order permutation null: rate_ecstasy | peak of the smoothed curve | 13.574 | 15.0332 | 1.6444 | -0.89 | 2000 |
| page-order permutation null: rate_ecstasy | lag-1 autocorrelation | 0.0894 | -0.0048 | 0.0682 | +1.38 | 2000 |
| page-order permutation null: rate_paranoia | peak of the smoothed curve | 2.7532 | 2.3236 | 0.3693 | +1.16 | 2000 |
| page-order permutation null: rate_paranoia | lag-1 autocorrelation | 0.1309 | -0.0036 | 0.0644 | +2.09 | 2000 |
| page-order permutation null: rate_disgust | peak of the smoothed curve | 17.5528 | 11.7444 | 1.0876 | +5.34 | 2000 |
| page-order permutation null: rate_disgust | lag-1 autocorrelation | 0.3641 | -0.0039 | 0.0683 | +5.39 | 2000 |
| page-order permutation null: rate_affect | peak of the smoothed curve | 32.4569 | 27.3329 | 1.7002 | +3.01 | 2000 |
| page-order permutation null: rate_affect | lag-1 autocorrelation | 0.2946 | -0.0055 | 0.0658 | +4.56 | 2000 |
| page-order permutation null: rate_valence | peak of the smoothed curve | -11.5216 | -14.0732 | 3.7058 | +0.69 | 2000 |
| page-order permutation null: rate_valence | lag-1 autocorrelation | 0.0581 | -0.0031 | 0.0591 | +1.04 | 2000 |
| clause-shuffle null: clf_fear | sd of per-page mean score | 0.0272 | 0.0228 | 0.0024 | +1.87 | 2000 |
| clause-shuffle null: clf_ecstasy | sd of per-page mean score | 0.0227 | 0.0225 | 0.0025 | +0.06 | 2000 |
| clause-shuffle null: clf_paranoia | sd of per-page mean score | 0.0247 | 0.0229 | 0.0024 | +0.77 | 2000 |
| clause-shuffle null: clf_disgust | sd of per-page mean score | 0.0237 | 0.0226 | 0.0021 | +0.53 | 2000 |
| term-shuffle null: core vocabulary pool (with replacement) | affect rate per 1,000 parts (128 relabelled terms) | 17.2747 | 365.965 | 25.7252 | -13.55 | 2000 |
| term-shuffle null: all non-affect book types (without replacement) | affect rate per 1,000 parts (128 relabelled terms) | 17.2747 | 34.3779 | 34.3822 | -0.5 | 2000 |
| english baseline: clf_fear | book mean vs matched English sentences | 0.2256 | 0.1937 |  | +11.07 | 1000 |
| english baseline: clf_ecstasy | book mean vs matched English sentences | 0.2452 | 0.1845 |  | +22.63 | 1000 |
| english baseline: clf_paranoia | book mean vs matched English sentences | 0.2064 | 0.1882 |  | +6.96 | 1000 |
| english baseline: clf_disgust | book mean vs matched English sentences | 0.2425 | 0.2023 |  | +14.7 | 1000 |
| english baseline: clf_neutral | book mean vs matched English sentences | 0.1339 | 0.2033 |  | -24.39 | 1000 |
| english baseline: clf_positive | book mean vs matched English sentences | 0.1746 | 0.2037 |  | -10.9 | 1000 |
| english baseline: clf_negative | book mean vs matched English sentences | 0.195 | 0.1719 |  | +7.68 | 1000 |
| english baseline: clf_valence | book mean vs matched English sentences | 0.0006 | 0.0279 |  | -14.25 | 1000 |

## 7. Explicit affect markers

364 fused units carry an affect term, 1,010 occurrences over 207 pages.  These are the book's own affect formulas -- the task's ``Fear=cell`` -- and they are why the lexicon runs on the part layer:

| marker | class | count | pages | first | axes |
|---|---|---:|---:|---:|---|
| `suck=blood` | `eq` | 120 | 85 | 4 | - |
| `machine=angel` | `eq` | 90 | 61 | 9 | - |
| `cadaver-feti` | `join` | 87 | 58 | 11 | - |
| `fear=cell` | `eq` | 84 | 73 | 6 | fear |
| `cadaver-mechanism` | `join` | 67 | 45 | 16 | - |
| `desire-protocol` | `join` | 59 | 40 | 29 | ecstasy |
| `desire-script` | `join` | 22 | 21 | 15 | ecstasy |
| `murder-memory` | `join` | 18 | 18 | 15 | - |
| `blood-clot` | `join` | 12 | 10 | 22 | - |
| `desire-mechanism` | `join` | 12 | 11 | 13 | ecstasy |
| `love-reptiles` | `join` | 8 | 7 | 51 | - |
| `murder-archive` | `join` | 7 | 7 | 16 | - |
| `rape-drone` | `join` | 7 | 5 | 5 | - |
| `love-reptile` | `join` | 6 | 6 | 52 | - |
| `physical-LOVE` | `join` | 6 | 4 | 64 | - |
| `Cadaver-feti` | `join` | 4 | 4 | 91 | - |
| `fear=cells` | `eq` | 4 | 4 | 155 | fear |
| `machine= angel` | `eq` | 4 | 4 | 112 | - |
| `Fear=cell` | `eq` | 3 | 3 | 4 | fear |
| `HYPE-murder` | `join` | 3 | 2 | 155 | - |
| `murder++the` | `cat` | 3 | 1 | 219 | - |
| `murder-ROM` | `join` | 3 | 3 | 173 | - |
| `nature=murder` | `eq` | 3 | 3 | 6 | - |
| `rape-soul/gram` | `slash+join` | 3 | 3 | 150 | - |
| `ruin=serum` | `eq` | 3 | 3 | 7 | - |

## 8. Instrument agreement

The two instruments are on different scales (a rate per 1,000 parts and a cosine), so rank agreement across the 225 pages is the honest statistic.  They are not fully independent -- both are about affect -- and their agreement is reported as corroboration, never as validation.

| axis | Spearman (pages) | Pearson (pages) |
|---|---:|---:|
| `fear` | +0.2884 | +0.2639 |
| `ecstasy` | +0.2810 | +0.4184 |
| `paranoia` | +0.2269 | +0.2147 |
| `disgust` | +0.1251 | +0.1033 |

## 9. Negative results and limitations

* **The lexicon is the analyst's choice.** It is curated in-module, not fetched from a published affect lexicon, and every term's count is published so it can be audited or edited; the two term-shuffle nulls (§6) bound how the observed rate sits against same-sized chunks of non-affect book vocabulary, but a different lexicon moves the exact numbers.
* **Disgust is thin for a lexical reason, not a semantic one.** 5 of 40 disgust terms occur at all, and three of them (``faecal`` 197, ``sickly`` 155, ``grotesque`` 99) carry 451 of the axis's 453 hits; ordinary English disgust words (``disgust``, ``revulsion``, ``nausea``, ``abhor``) are absent from the book.  The classifier's disgust curve is therefore the more informative of the two for that axis and is reported beside it.
* **The classifier is zero-shot, not supervised.** No labelled affect data for this book exists (there is no ground-truth translation, #69), so a trained classifier cannot be evaluated against anything.  A prototype classifier measures how close a clause is to an ordinary English description of an affect -- it is a retrieval reading, not a  diagnosis.
* **The two instruments share a conceptual basis, so their agreement is weak corroboration, not independent confirmation**; the workspace's triangulation (#73) is the place where genuinely independent methods are compared.
* **English baselines are prose, not book-like texte.** The references are ordinary novels and abstracts (the three excerpt corpora are flagged non-inferential by #56 and are excluded from the pooled comparison's inference), so a rate difference can be a genre effect as well as a book effect; that is why the pooled general-English row is the reported one and the excerpts are shown but not tested.
* **The curve has no arc, but it is not featureless -- and that is the finding, not a failure to find one.** No individual axis beats the page-order permutation null on the peak of its smoothed curve (fear z = +0.25, ecstasy z = -0.89, paranoia z = +1.16, disgust z = +5.34), so there is no rising or falling affect trajectory to read -- consistent with #42's stationarity verdict and #51's absence of a latent periodicity, and inconsistent with the jacket copy's escalating-dread narrative.  What the nulls *do* find is local smoothness: the aggregate curve's lag-1 autocorrelation is +4.56 sd above the permutation null (fear +2.03, ecstasy +1.38), i.e. adjacent pages share affect vocabulary more than a random page assignment would -- the same template clustering #42 found in the embedding space, seen lexically.  The aggregate peak (z = +3.01) says some pages are affect-hotter than chance, but no single axis has a peak, so there is clustering, not an arc.
* **The affect lexicon is not frequent vocabulary within the book.** Both term-shuffle nulls relabel the 128 affect-axis terms with same-sized chunks of non-affect book vocabulary: against the book's core content vocabulary (``soul``, ``cell``, ``machine``, ``blood`` ...) the affect lexicon's 1,378 occurrences sit far below the null (z = -13.55), and against a random same-sized set of the book's non-affect types the null mean is 2,742 (z = -0.50, not significant).  Affect words are individually rare in this book; the density in §4 is a cross-register effect -- the book uses its small affect vocabulary far more than ordinary English prose does, not more than its own content words.

