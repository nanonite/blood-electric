# Burstiness and Heaps' law (#53)

Vocabulary growth vs text length, and the burstiness of entity use. Source: the #17 unit layer screened two ways (both disclosed in Method), #30's entity glossary, and the eight pinned #24 English references at matched N. Reproduce with `python work/burstiness_heaps.py`; regression suite `python work/test_burstiness_heaps.py`. Schema `burstiness-heaps/1`.

## Verdict

1. **Vocabulary growth is ordinary, not alien.** The book's unit layer grows at Heaps' beta **0.603** (log K 1.970, R^2 0.995, leave-one-page-out jackknife se 0.011), inside the English prose range 0.540-0.747 (English mean 0.637), below the English mean. The growth curve asks for no exotic technique: the book mints vocabulary like a novel. The only fast-growing layer remains #24's coinage subset (beta 0.835), already explained there as rule-resolved compounds with a shallow Zipf slope.

2. **The cast is introduced early.** Over the entity-mention stream the named layer grows at beta **0.367** - well below the word layer. On the **mention** partition (of 2,452 mentions) 64 / 20 / 14 of the 98 entities are first attested in the first / second / last third, so 64 of 98 are already on stage by the first third; on the **token** partition (of 74,077 tokens) the same split is 64 / 17 / 17. The two partitions are different denominators and are not interchangeable (section 2). Either way the named layer is a fixed dramatis personae put to work, not an open-ended naming system.

3. **Entity use is bursty - and so is the coinage layer.** Of 39 entities with n >= 8, **20 are bursty** (over-dispersed against a placement null at their own rate, with B > 0.0), and 6 sit under the descriptive F(1000) < 0.9 cut - but the low-tail test certifies **none** of them as more evenly than chance (0 rows carry an `even` verdict). The entity median is F(1000) 1.42. The merged entity channel (1,988 distinct positions, 2.7% of tokens) has F(1000) 3.02 against a null median 0.95 - over-dispersed, but *not* the most clustered repetition in the book: the merged coinage-type channel reaches F(1000) 3.54 and the merged English-word channel 2.81 (`channels.tsv` compares merged channel against merged channel, each against its own rate-matched null; the previous build stacked the entity channel against per-term control medians, which are not the same unit). The finding that survives is the *split* inside the entity layer: a bursty foreground that arrives in arcs, against a background that is at best weakly under-dispersed (section 3).

## Method

- **Two screens, disclosed and never mixed.** The growth analysis runs on #24's screen (347 markup removals, 74,318 units) so every fit reconciles with `analysis/frequency/fits.tsv` row for row; the entity analysis runs on the #27/#30 screen (the general tag-run detector on top, 588 removals, 74,077 units) so every entity count reconciles with #30's `entity_glossary.tsv` row for row. The 241-occurrence difference is the `href`/`mailto`/`colspan` tag furniture the general detector removes and #19's ledger does not list.
- **Heaps' law** is fitted exactly as #24 fits it (OLS on the log-log cumulative type curve at 400 checkpoints), so the point estimates reproduce `fits.tsv`. The uncertainty is a leave-one-page-out jackknife for the book and leave-one-1000-token-block-out for the references - a resampling that never duplicates a group, which a page bootstrap would (a duplicated page contributes no new types, biasing beta down). The reported uncertainty is the **jackknife standard error** `sd * sqrt(n - 1)`, not the plain sd across the n refits: the book's 225 page refits have sd 0.0008 but se 0.011, and the matched-N English references span se 0.008-0.020 (`heaps_fits.tsv` publishes the sd, the se, the refit mean and the refit mean minus the point estimate for every row).
- **Burstiness** reports three legs per entity: the Fano factor Var(count)/Mean(count) over non-overlapping windows of 100 / 500 / 1000 tokens (1.0 is a chance placement); the Goh-Barabasi burstiness B = (sigma - mu)/(sigma + mu) of the inter-arrival gaps (0 is geometric gaps); and **both tails** of a placement null - 2,000 draws per term, each a uniform random *subset* of the stream with exactly the term's own rate (no token used twice), seeded per term by `SEED + crc32(name)`. The high tail (ties counted, add-one corrected) asks *more clustered than chance*; the low tail asks *more evenly than chance*. A `bursty` verdict needs p <= 0.05 **and** B > 0.0. An `even` verdict needs the low-tail p <= 0.05; F(1000) < 0.9 is a descriptive cut and certifies nothing on its own.
- **Entity positions** are located on the screened stream with #30's own phrase rule: single-unit surfaces match exactly; multi-word surfaces require every word capitalized and whitespace-adjacent in the page body, with a trailing numeric token read as a designator. The located counts equal #30's `n_occurrences` for 98 of 98 entities (98/98 in this build).

## 1. Heaps' law: vocabulary growth vs text length

| corpus | N | V | beta | log K | R^2 | beta LOO se | sd | mean-bias | beta LOO range | V@10k | V@25k | V@50k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| book_units | 74,318 | 5,868 | 0.603 | 1.970 | 0.995 | 0.0114 | 0.0008 | -2.17e-06 | 0.600-0.609 | 1832 | 3372 | 4853 |
| book_units_lower | 74,318 | 5,502 | 0.598 | 1.959 | 0.995 | 0.0108 | 0.0007 | +4.02e-06 | 0.595-0.604 | 1734 | 3172 | 4554 |
| book_parts | 80,015 | 3,687 | 0.499 | 2.653 | 0.982 | 0.0132 | 0.0009 | +2.25e-04 | 0.494-0.506 | 1404 | 2450 | 3149 |
| book_english | 66,506 | 3,209 | 0.502 | 2.585 | 0.978 | 0.0133 | 0.0009 | +3.43e-04 | 0.496-0.509 | 1347 | 2313 | 2884 |
| book_nonenglish | 7,812 | 2,293 | 0.818 | 0.472 | 0.996 | 0.0166 | 0.0011 | -6.50e-06 | 0.814-0.829 | - | - | - |
| book_coinage | 7,361 | 2,245 | 0.835 | 0.354 | 0.995 | 0.0134 | 0.0009 | -1.61e-04 | 0.831-0.839 | - | - | - |
| book_ocr | 127 | 15 | 0.570 | 0.127 | 0.943 |  |  |  | - | - | - | - |
| en_pride_and_prejudice | 74,318 | 5,579 | 0.540 | 2.629 | 0.985 | 0.0109 | 0.0013 | +1.30e-03 | 0.539-0.551 | 2136 | 3413 | 4739 |
| en_moby_dick | 74,318 | 10,042 | 0.681 | 1.603 | 0.990 | 0.0127 | 0.0015 | +7.75e-04 | 0.672-0.684 | 2895 | 5092 | 7728 |
| en_frankenstein | 74,318 | 6,975 | 0.598 | 2.232 | 0.984 | 0.0079 | 0.0009 | +1.06e-03 | 0.595-0.601 | 2532 | 4159 | 5853 |
| en_dracula | 74,318 | 6,597 | 0.622 | 1.885 | 0.993 | 0.0094 | 0.0011 | +5.65e-04 | 0.618-0.625 | 2111 | 3557 | 5458 |
| en_sherlock_holmes | 74,318 | 6,788 | 0.605 | 2.072 | 0.996 | 0.0147 | 0.0017 | +1.07e-03 | 0.605-0.619 | 2184 | 3691 | 5548 |
| en_heart_of_darkness | 38,774 | 5,455 | 0.672 | 1.576 | 0.993 | 0.0151 | 0.0024 | +1.92e-03 | 0.667-0.686 | 2520 | 4336 | - |
| en_dorian_gray | 74,318 | 6,719 | 0.628 | 1.777 | 0.996 | 0.0203 | 0.0024 | +9.53e-04 | 0.622-0.645 | 1905 | 3480 | 5304 |
| en_ulysses | 74,318 | 12,108 | 0.747 | 1.041 | 0.997 | 0.0117 | 0.0014 | +4.68e-04 | 0.742-0.749 | 2638 | 5781 | 8895 |
| **English mean** (8 refs) | | | 0.637 | | | | | | | | | |

The jackknife columns are the corrected uncertainty, published separately because they are not interchangeable: **se** is `sd * sqrt(n - 1)` across the 225 book-page refits, **sd** is their plain spread, and **mean-bias** is the refit mean minus the point estimate. The bias is small but not zero - book_units -2.17e-06, and the largest in this table is 0.0019 - because dropping a group shortens the curve and moves the checkpoint grid. The refit mean is therefore *not* the point estimate and is not used as one; only the spread is, and the se is what an uncertainty statement on a single beta means. The earlier build put the plain sd in this column and called it the jackknife uncertainty, which understates the book's se by a factor of `sqrt(225 - 1)` = 15.0.

The full curve is in `growth_curve.tsv` (200 checkpoints per corpus; the English references are averaged on their shared matched-N grid). The segment table (`segment_growth.tsv`) reads the growth question from the other side - how much of each fifth of the text is vocabulary the book has not used before:

| corpus | seg 1 | seg 2 | seg 3 | seg 4 | seg 5 |
|---|---:|---:|---:|---:|---:|
| book_units | 2,403 new / 2,403 types / beta 0.66 | 1,306 new / 2,288 types / beta 0.65 | 873 new / 2,087 types / beta 0.91 | 631 new / 1,714 types / beta 0.58 | 655 new / 2,088 types / beta 0.62 |
| book_units_lower | 2,281 new / 2,281 types / beta 0.65 | 1,207 new / 2,166 types / beta 0.63 | 812 new / 1,985 types / beta 0.90 | 597 new / 1,647 types / beta 0.57 | 605 new / 1,984 types / beta 0.62 |
| book_parts | 1,878 new / 1,878 types / beta 0.58 | 804 new / 1,796 types / beta 0.58 | 423 new / 1,589 types / beta 0.88 | 245 new / 1,294 types / beta 0.52 | 337 new / 1,669 types / beta 0.57 |
| book_english | 1,647 new / 1,647 types / beta 0.61 | 716 new / 1,586 types / beta 0.59 | 360 new / 1,384 types / beta 0.68 | 201 new / 1,134 types / beta 0.54 | 285 new / 1,451 types / beta 0.58 |
| book_coinage | 620 new / 620 types / beta 0.84 | 563 new / 654 types / beta 0.81 | 471 new / 601 types / beta 0.83 | 242 new / 347 types / beta 0.71 | 349 new / 556 types / beta 0.76 |

Novelty falls steeply for the core layers (2,403 -> 1,306 -> 873 -> 631 -> 655 new unit types per fifth), the shape of a finite vocabulary being recombined; it is not strictly monotone (the last fifth adds 655 against 631 in the fourth), and the segment betas stay in the same band (0.58-0.91) rather than decaying, with the seg-3 high (0.91) an artifact: the #4/#34 OCR recognition loop on page 90 mints one-off types (`apoptosis=second`), not coinage.

## 2. The entity cast's growth

The mention stream (2,452 mentions over 74,077 tokens) has its own type curve (`entity_growth.tsv`): beta 0.367, log K 1.718.

| cast measure (partition, denominator) | first third | second third | last third |
|---|---:|---:|---:|
| entities **first attested** here - mention stream, 2,452 mentions | 64 | 20 | 14 |
| entities **first attested** here - token stream, 74,077 tokens | 64 | 17 | 17 |
| cumulative **share of the cast** seen by the end of the third - mention stream | 65% | 86% | 100% |

The first two rows are different partitions of the same book and their totals are not comparable: the mention stream splits on the order mentions arrive in, the token stream splits on stream position, so an entity introduced by a single early mention and one introduced by a concentration of early mentions land in different thirds. The third row uses the mention stream's own denominator. Each row is therefore reported only against itself, and the two first-attestation rows are shown side by side precisely because the earlier build mixed them in one table.

## 3. Burstiness of entity use

Per-entity table: `entity_burstiness.tsv` (39 rows, n >= 8, sorted by F(1000)). The control population is in `controls_burstiness.tsv` (20 rows, entity surfaces removed) and the merged-channel comparison in `channels.tsv` (3 channels). `p high` is the over-dispersion tail, `p low` the under-dispersion tail, and `ties` the share of null draws landing exactly on the observed value.

| entity | role | n | pages | B | F(100) | F(1000) | null F | p high | p low | ties | verdict | top page |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| retro-ADAM | persona | 165 | 43 | 0.667 | 1.37 | 9.82 | 0.96 | <0.001 | 1.000 | 0.00 | bursty | 149 (7) |
| ADAM Doll | creature | 166 | 62 | 0.551 | 1.34 | 6.32 | 0.97 | <0.001 | 1.000 | 0.00 | bursty | 217 (7) |
| ADAM | persona | 303 | 123 | 0.374 | 1.14 | 4.25 | 0.98 | <0.001 | 1.000 | 0.00 | bursty | 57 (8) |
| NDROID | persona | 20 | 13 | 0.380 | 0.97 | 3.73 | 0.93 | <0.001 | 1.000 | 0.00 | bursty | 116 (4) |
| HIV | persona | 13 | 9 | -0.262 | 0.98 | 3.29 | 0.98 | <0.001 | 1.000 | 0.00 | neutral | 171 (3) |
| DNA=channel | machine-system | 75 | 55 | 0.370 | 0.95 | 2.31 | 0.97 | <0.001 | 1.000 | 0.00 | bursty | 128 (3) |
| cadaver | persona | 441 | 182 | 0.037 | 0.81 | 2.14 | 0.97 | <0.001 | 1.000 | 0.00 | bursty | 161 (7) |
| ToKAGE | persona | 127 | 84 | 0.240 | 1.06 | 2.01 | 0.97 | <0.001 | 1.000 | 0.00 | bursty | 98 (4) |
| dustNirverna | persona | 15 | 12 | 0.438 | 1.11 | 2.00 | 0.93 | <0.001 | 1.000 | 0.00 | bursty | 70 (2) |
| Cadaver City | place | 191 | 118 | 0.207 | 0.87 | 1.81 | 0.97 | <0.001 | 1.000 | 0.00 | bursty | 80 (4) |
| BABEL=TV | machine-system | 16 | 14 | 0.269 | 1.23 | 1.78 | 0.91 | <0.001 | 1.000 | 0.00 | bursty | 28 (2) |
| Placenta | persona | 71 | 50 | 0.281 | 1.07 | 1.76 | 0.97 | <0.001 | 1.000 | 0.00 | bursty | 216 (5) |
| ToK | persona | 12 | 8 | 0.295 | 0.98 | 1.67 | 1.00 | 0.003 | 0.999 | 0.00 | bursty | 4 (2) |
| cyberBuddha | persona | 14 | 11 | 0.009 | 1.41 | 1.67 | 0.95 | 0.004 | 0.998 | 0.00 | bursty | 219 (4) |
| NDRO | persona | 23 | 20 | 0.050 | 1.06 | 1.65 | 0.95 | 0.003 | 0.998 | 0.00 | bursty | 117 (2) |

The burst profile splits the cast. The **foreground** arrives in arcs: `retro-ADAM` (F(1000) 9.82, B 0.667) is confined to pages 87-176 - a late-book sequence, not a texture spread over the novel.

The **background is not the mirror image of that.** The rows under the cut, in full:

| entity | n | B | F(1000) | p high | p low | ties | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `SM` | 23 | 0.079 | 0.78 | 0.978 | 0.128 | 0.10 | neutral |
| `Sato` | 29 | -0.175 | 0.88 | 0.815 | 0.337 | 0.15 | neutral |
| `Sato Corporation` | 26 | -0.152 | 0.88 | 0.827 | 0.352 | 0.18 | neutral |
| `City-city` | 12 | -0.299 | 0.84 | 1.000 | 0.399 | 0.40 | neutral |
| `Cadaver City-city` | 12 | -0.299 | 0.84 | 1.000 | 0.402 | 0.40 | neutral |
| `ND` | 8 | -0.170 | 0.89 | 1.000 | 0.682 | 0.68 | neutral |

None of them is certified even by the low tail. Those 6 rows are the ones the earlier build called "spread more evenly than chance", but at these rates the Fano factor over 74 windows is a coarse, lumpy statistic and the null's own spread puts plenty of chance placements under the cut. The most extreme row is `SM` (F(1000) 0.78, low-tail p 0.128) - the only row anywhere near significance, and still short of it, on 23 occurrences. Every other low row is ordinary: `Sato` p 0.337, `Sato Corporation` p 0.352, `City-city` p 0.399, `Cadaver City-city` p 0.402, `ND` p 0.682. So **no** tested entity is certified more evenly than chance, and the `even` count in this build is 0. The claim removed here is the aggregate one; the per-entity low-tail p-values are published so the reading can be re-checked. The `ties` column is the share of null draws landing on exactly the observed value, and it is why the reading is coarse: `ND`, `Cadaver City-city`, `City-city` sit on the null's *mode* - 0.40-0.68 of the draws tie exactly at their observed index - so a strict-inequality count (only draws strictly below) returns the floor for them. That convention is what gives `SM` its 0.02-0.03, and it gives the rows sitting on the null's mode the 1/2001 floor for being ordinary; ties are counted toward the p instead.

`HIV` is the case that shows why the rule needs two legs. Its occurrences are over-dispersed against the placement null (F(1000) 3.29, high-tail p <0.001) but its inter-arrival gaps are *more regular* than geometric (B -0.262 < 0.0), an even-spacing signature a clustered process does not have, so it stays `neutral` rather than `bursty`.

The **foreground is not as concentrated as the headline says**, because the cast nests: `ADAM Doll` occurrences sit *inside* `ADAM` (`nested_entities.tsv`), 166 of ADAM's 303 occurrences are also ADAM Doll occurrences. Counting them only under the long surface lifts ADAM's F(1000) from 2.87 to 4.25 - a 48% inflation of the clustering statistic on no new evidence, since both surfaces mark the same token. The nesting runs the other way too: `cadaver` contains 191 `Cadaver City` occurrences, and removing them takes cadaver's F(1000) from 2.14 to 4.57 - its modest index *dilutes* a strongly clustered one. Every nested pair is published, with both the unadjusted and the nested-excluded index, so no headline here stands without its denominator. The high-frequency textures sit between the two regimes - `cadaver` (441 occurrences over 182 pages) F(1000) 2.14 unadjusted - over-dispersed in the strict test but a fraction of the arcs' concentration.

What the bursts *are* is checkable by eye. These are the densest single-page entity clusters in the glossary, shown in the unit stream:

- **ADAM** - 8 occurrences on p57 (of 303 total): `...of city body gland like the ADAM Doll that makes the body gland...`
- **ADAM** - 7 occurrences on p72 (of 303 total): `...the swastika contraction of the universe ADAM Doll assassination speed to the body...`
- **ADAM** - 7 occurrences on p217 (of 303 total): `...like the machine nature of the ADAM Doll that quiesced Quantum suicide machine...`
- **ADAM Doll** - 7 occurrences on p217 (of 166 total): `...like the machine nature of the ADAM Doll that quiesced Quantum suicide machine...`

A measured burst is not always narrative. On the corpus's disclosed OCR recognition loop pages (page 90: `cadaver` x4, `Cadaver City` x3; page 219: `cyberBuddha` x4, `ADAM` x3, `ADAM Doll` x3, `VTR` x3, `konganic-eye` x3) the entity clusters are machine artifacts - page 219 repeats `...the technology of murder href mailto konganic-eye of cyberBuddha konganic-eye of cyberBuddha and was sutured to...`, and the #4/#34 page-90 loop mints the one-off types behind the seg-3 novelty spike. They are reported as bursts because they are, with the cause named, rather than filtered out.

## 4. Controls

The same estimator on non-entity repetitions: the 10 most frequent English unit types and the 10 most frequent non-English (coinage) types (`controls_burstiness.tsv`). Every surface the glossary can match is excluded before the selection, so the population being controlled for no longer contains its own controls - the earlier build let #30's `retro-ADAM` and `ToKAGE` appear as coinage-type controls while also appearing as entities in the table above, which padded the control with the very effect it was meant to measure. Even this baseline is over-dispersed, because the corpus's own density varies by page (spine pages, OCR loops) - the placement null controls rate, not the envelope - so the entity claim is carried by *magnitude*, not by significance alone.

- **English word control** (10 types): 4 bursty, median F(1000) 1.81, max 3.46, median B -0.011.
- **Coinage-type control** (10 types): 9 bursty, median F(1000) 1.57, max 2.34, median B +0.213.

| kind | term | n | B | F(100) | F(1000) | null F | p high | p low | ties | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| english_word | the | 9740 | -0.128 | 0.68 | 1.62 | 0.85 | <0.001 | 1.000 | 0.00 | neutral |
| english_word | of | 8096 | -0.145 | 0.77 | 1.95 | 0.87 | <0.001 | 1.000 | 0.00 | neutral |
| english_word | that | 2456 | -0.027 | 0.96 | 1.93 | 0.95 | <0.001 | 1.000 | 0.00 | neutral |
| english_word | to | 1744 | 0.003 | 1.16 | 3.46 | 0.95 | <0.001 | 1.000 | 0.00 | bursty |
| english_word | is | 944 | -0.019 | 1.22 | 2.62 | 0.97 | <0.001 | 1.000 | 0.00 | neutral |
| english_word | a | 925 | 0.018 | 0.97 | 1.68 | 0.97 | <0.001 | 1.000 | 0.00 | bursty |
| english_word | The | 847 | 0.040 | 0.89 | 1.63 | 0.97 | <0.001 | 1.000 | 0.00 | bursty |
| english_word | was | 713 | 0.132 | 1.07 | 1.94 | 0.97 | <0.001 | 1.000 | 0.00 | bursty |
| english_word | in | 568 | -0.046 | 0.85 | 1.12 | 0.97 | 0.179 | 0.822 | 0.00 | neutral |
| english_word | internal | 554 | -0.002 | 0.80 | 0.91 | 0.96 | 0.642 | 0.360 | 0.00 | neutral |
| coinage_type | boy-roid | 315 | 0.399 | 0.95 | 2.08 | 0.98 | <0.001 | 1.000 | 0.00 | bursty |
| coinage_type | soul/gram | 306 | 0.494 | 0.87 | 2.34 | 0.98 | <0.001 | 1.000 | 0.00 | bursty |
| coinage_type | fecundates | 289 | 0.124 | 1.14 | 2.10 | 0.97 | <0.001 | 1.000 | 0.00 | bursty |
| coinage_type | cold-blooded | 188 | 0.083 | 0.87 | 1.43 | 0.98 | 0.008 | 0.993 | 0.00 | bursty |
| coinage_type | defleshed | 155 | 0.094 | 0.97 | 1.28 | 0.97 | 0.042 | 0.963 | 0.00 | bursty |
| coinage_type | gene=TV | 142 | 0.119 | 0.89 | 1.50 | 0.97 | 0.002 | 0.999 | 0.00 | bursty |
| coinage_type | vital=serum | 128 | 0.440 | 0.98 | 2.02 | 0.97 | <0.001 | 1.000 | 0.00 | bursty |
| coinage_type | respires | 124 | 0.065 | 1.01 | 1.24 | 0.98 | 0.073 | 0.939 | 0.01 | neutral |
| coinage_type | body-OMOTYA | 121 | 0.302 | 0.97 | 1.63 | 0.98 | <0.001 | 1.000 | 0.00 | bursty |
| coinage_type | suck=blood | 120 | 0.322 | 0.95 | 1.31 | 0.98 | 0.039 | 0.968 | 0.01 | bursty |

Per-term medians describe the per-term table and nothing more. A median over 10 single-term Fano factors and one merged channel's Fano factor are different units, and comparing them is what produced the earlier "the entity channel sits well above both control populations" line. The like-for-like comparison merges each control population the same way the entity channel is merged and scores every channel against its own rate-matched null (`channels.tsv`):

| channel | kind | positions | rate | F(1000) | null F | p high | p low | B | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| entity_channel | entity | 1,988 | 2.7% | 3.02 | 0.95 | <0.001 | 1.000 | 0.022 | bursty |
| english_word_channel | english_word | 26,587 | 35.9% | 2.81 | 0.63 | <0.001 | 1.000 | -0.178 | neutral |
| coinage_type_channel | coinage_type | 1,888 | 2.5% | 3.54 | 0.95 | <0.001 | 1.000 | 0.078 | bursty |

Read on that footing the superlative does not survive. The merged entity channel (F(1000) 3.02) is *below* the merged coinage-type channel (F(1000) 3.54, 1,888 positions at a comparable 2.5% rate) and above the merged English-word channel (F(1000) 2.81, 26,587 positions at 35.9%). Both control channels are over-dispersed against their own nulls at p < 0.001, so *named* vocabulary in general clusters in this book - the coinages are its motif vocabulary (`boy-roid`, `soul/gram`, `vital=serum`) - and the entity layer is a member of that population rather than its most extreme case. What separates them is the gap statistic: the entity channel's B is 0.022 against -0.178 for the English-word channel, whose over-dispersion comes with *more regular* gaps - the texture of running prose, not of arcs.

## 5. Limits

- The placement null controls rate, not the corpus's page-length envelope: a page-rate-preserving null would be stricter, so the p-values read as *over-dispersion* while the F(1000) magnitude - not the p alone - carries the claim.
- Entity occurrences overlap where one glossary entity contains another (`Sato` inside `Sato Corporation`, `ADAM` inside `ADAM Doll`, `cadaver` inside `Cadaver City`); the merged channel counts distinct positions, the per-entity rows do not de-duplicate, and the nested-excluded index for every overlapping pair is in `nested_entities.tsv`.
- The `even` verdict rests on a low-tail p from the same null, which at these rates is a coarse statistic: for some terms the observed Fano value is a *common* null value rather than an unusual one. Ties are counted toward the p, which is the conservative reading; a strict-inequality count would score those terms as maximally even because they sit on the null's mode, so it is not used.
- The matched-N comparison is #24's convention: English references are truncated to the book's token count, so their full-length growth (Ulysses to N = 270k) is not compared here.
- The growth curves are in-sample; the book has no held-out text. The jackknife is an uncertainty statement, not a prediction interval, and the refit mean is not the point estimate - see the `mean-bias` column in section 1.

Reproduce with `python work/burstiness_heaps.py`; suite `python work/test_burstiness_heaps.py`. Offline on the pinned cache; deterministic (fixed seeds, no clock), rebuild byte-identical.
