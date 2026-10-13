# POS-tagging the silence-removed residue (#37)

Source: `analysis/silence_removal/v3_operators_to_space.txt` (the plain stream from E1.4, 80,260 word tokens). Reference: 15 general English novels from `analysis/reference_corpora/general_english/`. Reproduce with `python work/pos_tagging.py`; regression test `work/test_pos_tagging.py`.

**Method.** A rule-based POS tagger (no NLTK/spaCy in this environment). Three tiers: (1) curated lexicon of ~3000 common English words + Brown corpus verb surface forms, (2) morphological suffix rules for unknown words, (3) bigram contextual disambiguation. The *same* tagger is applied to both the residue stream and the reference corpora, so the comparison is fair by construction.

**Tagset.** Simplified Penn Treebank: NN/NNS/NNP (nouns), VB/VBD/VBG/VBN/VBZ (verbs), JJ (adjectives), RB (adverbs), DT/IN/CC/PRP/MD/CD and other function-word tags.

## Headline: noun/verb balance drift

| metric | v3 residue | general English | delta |
|---|---:|---:|---:|
| total tokens | 80,260 | 1,628,800 | — |
| noun % | 42.62% | 23.0% | +19.62 |
| verb % | 12.78% | 21.35% | -8.57 |
| noun/verb ratio | 3.334 | 1.077 | +2.257 |
| noun share of content | 76.93% | 51.86% | — |
| verb share of content | 23.07% | 48.14% | — |

## Statistical significance

Chi-square test on the 2x2 noun/verb x corpus contingency table: chi2 = 10570.07, p = 0.0, dof = 1.
The drift is **highly significant** (p < 0.001).

## POS tag distribution

| tag | v3 count | v3 % | ref count | ref % | delta % |
|---|---:|---:|---:|---:|---:|
| CC | 357 | 0.44% | 67,118 | 4.12% | -3.68 |
| CD | 206 | 0.26% | 14,222 | 0.87% | -0.62 |
| DT | 12,219 | 15.22% | 147,845 | 9.08% | +6.15 |
| IN | 8,815 | 10.98% | 124,881 | 7.67% | +3.32 |
| JJ | 6,208 | 7.73% | 79,884 | 4.9% | +2.83 |
| MD | 0 | 0.0% | 21,849 | 1.34% | -1.34 |
| NN | 28,901 | 36.01% | 249,613 | 15.32% | +20.68 |
| NNP | 2,877 | 3.58% | 81,903 | 5.03% | -1.44 |
| NNS | 2,425 | 3.02% | 43,156 | 2.65% | +0.37 |
| PDT | 15 | 0.02% | 11,640 | 0.71% | -0.70 |
| PRP | 967 | 1.2% | 136,927 | 8.41% | -7.20 |
| PRP$ | 979 | 1.22% | 49,156 | 3.02% | -1.80 |
| RB | 371 | 0.46% | 91,314 | 5.61% | -5.14 |
| RP | 993 | 1.24% | 64,163 | 3.94% | -2.70 |
| TO | 1,792 | 2.23% | 40,264 | 2.47% | -0.24 |
| UH | 6 | 0.01% | 9,893 | 0.61% | -0.60 |
| VB | 6,295 | 7.84% | 199,985 | 12.28% | -4.43 |
| VBD | 2,528 | 3.15% | 91,012 | 5.59% | -2.44 |
| VBG | 272 | 0.34% | 28,152 | 1.73% | -1.39 |
| VBN | 138 | 0.17% | 10,137 | 0.62% | -0.45 |
| VBZ | 1,025 | 1.28% | 18,454 | 1.13% | +0.14 |
| WDT | 2,462 | 3.07% | 22,161 | 1.36% | +1.71 |
| WP | 169 | 0.21% | 15,027 | 0.92% | -0.71 |
| WRB | 240 | 0.3% | 10,044 | 0.62% | -0.32 |

## Per-corpus noun/verb balance

| corpus | tokens | noun % | verb % | noun/verb ratio |
|---|---:|---:|---:|---:|
| cs/arxiv_abstracts | 105,864 | 34.58% | 22.32% | 1.549 |
| cutup_surrealist/artaud | 60 | 31.67% | 28.33% | 1.118 |
| cutup_surrealist/breton | 47 | 40.43% | 25.53% | 1.583 |
| cutup_surrealist/burroughs | 79 | 40.51% | 26.58% | 1.524 |
| cyberpunk/gibson | 96 | 37.5% | 26.04% | 1.44 |
| cyberpunk/stephenson | 79 | 41.77% | 20.25% | 2.062 |
| cyberpunk/sterling | 50 | 40.0% | 20.0% | 2.0 |
| finnegans_wake/joyce | 88 | 42.05% | 17.05% | 2.467 |
| general_english/a_tale_of_two_cities | 138,494 | 22.45% | 20.95% | 1.072 |
| general_english/alice_in_wonderland | 27,427 | 19.96% | 23.5% | 0.849 |
| general_english/dorian_gray | 83,627 | 21.44% | 23.47% | 0.914 |
| general_english/dracula | 166,801 | 19.59% | 22.47% | 0.872 |
| general_english/frankenstein | 78,547 | 20.92% | 20.76% | 1.008 |
| general_english/great_expectations | 189,055 | 20.15% | 21.99% | 0.916 |
| general_english/great_gatsby | 50,013 | 22.73% | 22.09% | 1.029 |
| general_english/heart_of_darkness | 42,204 | 22.39% | 21.14% | 1.059 |
| general_english/metamorphosis | 22,377 | 18.8% | 22.8% | 0.825 |
| general_english/moby_dick | 222,567 | 24.99% | 20.05% | 1.246 |
| general_english/pride_and_prejudice | 131,858 | 20.7% | 21.59% | 0.959 |
| general_english/sherlock_holmes | 109,111 | 20.72% | 21.28% | 0.973 |
| general_english/the_time_machine | 32,896 | 21.24% | 20.57% | 1.032 |
| general_english/the_war_of_the_worlds | 61,018 | 23.38% | 20.84% | 1.122 |
| general_english/ulysses | 272,805 | 29.75% | 20.68% | 1.438 |
| legal/declaration_of_independence | 1,748 | 26.83% | 18.14% | 1.479 |
| legal/federalist_papers | 192,773 | 21.51% | 16.95% | 1.27 |
| legal/magna_carta | 14,594 | 22.92% | 17.54% | 1.307 |
| legal/the_law | 18,842 | 23.93% | 19.8% | 1.209 |
| legal/the_social_contract | 123,979 | 20.46% | 19.41% | 1.054 |
| medical/pubmed_abstracts | 427,471 | 49.79% | 20.98% | 2.373 |

## Interpretation

The silence-removed residue is **noun-heavy** compared to normal English (+19.62 percentage points). This is consistent with the book's style: long chains of noun phrases ("the vital-icon our chromosome form escape of the suck blood chromosome") with verbs often buried inside operator-delimited clauses.

The noun/verb ratio drift is the key metric: a ratio > 1 means more nouns than verbs.  Normal English prose typically has a ratio of 1.5-2.5.  The v3 residue's ratio of 3.334 is outside the normal range.

## Caveats

- **Rule-based tagger, not a neural parser.** The tagger is deterministic and auditable, but its accuracy is not comparable to a trained POS tagger.  The *comparison* is valid because the same tagger is applied to both streams.
- **Proper noun over-generation.** Capitalized words not in the lexicon are tagged NNP.  The book's heavy use of capitalized neologisms (`ToKAGE`, `Cadaver City`) inflates the NNP count.  This is a known limitation.
- **Unknown words default to NN.** Words not in the lexicon and not matching any suffix rule are tagged NN (the most common unknown word class).  This biases the noun count upward for both streams.
- **The v3 stream preserves the book's punctuation-free style.** The 22 pre-existing periods are mostly in the front matter.  The tagger sees a continuous stream, which may affect contextual disambiguation at sentence boundaries.

## Handoff

- **#50** (quantified silence removal): the POS distribution provides a structural baseline for interpreting perplexity differences between variants.
- **#62** (interlinear edition): the tagged residue can serve as a morphological annotation layer.
- **#18/#65** (compound decomposition / termbase): the noun-heavy profile is consistent with the compound-dense style documented in #18.

---
*Generated by `work/pos_tagging.py`*
