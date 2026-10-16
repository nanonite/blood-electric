# Burstiness and Heaps' law (#53)

Vocabulary growth vs text length, and the burstiness of entity use. Source: the #17 unit layer screened two ways (both disclosed in Method), #30's entity glossary, and the eight pinned #24 English references at matched N. Reproduce with `python work/burstiness_heaps.py`; regression suite `python work/test_burstiness_heaps.py`. Schema `burstiness-heaps/1`.

## Verdict

1. **Vocabulary growth is ordinary, not alien.** The book's unit layer grows at Heaps' beta **0.603** (log K 1.970, R^2 0.995, leave-one-page-out sd 0.0008), inside the English prose range 0.540-0.747 (English mean 0.637), below the English mean. The growth curve asks for no exotic technique: the book mints vocabulary like a novel. The only fast-growing layer remains #24's coinage subset (beta 0.835), already explained there as rule-resolved compounds with a shallow Zipf slope.

2. **The cast is introduced early.** Over the entity-mention stream the named layer grows at beta **0.367** - well below the word layer - and 64 / 17 / 17 of the 98 entities are first attested in the first / second / last third of the stream. By the first third of the mentions 64 of 98 entities are already on stage: the named layer is a fixed dramatis personae put to work, not an open-ended naming system.

3. **Entity use is bursty, and the named layer is the most clustered repetition in the book.** Of 39 entities with n >= 8, **20 are bursty** (over-dispersed against a Bernoulli placement null at their own rate) and 6 are spread *more evenly than chance*. The entity median is F(1000) 1.42. The aggregate entity channel (1,988 distinct positions, 2.7% of tokens) has F(1000) 3.02 against a null median 0.98 - above both control populations (median F(1000) 1.81 for the ten most frequent English words, 2.01 for the ten most frequent coinage types). The finding is the *split* inside the entity layer: a bursty foreground that arrives in arcs, against an ambient background flatter than chance (section 3).

## Method

- **Two screens, disclosed and never mixed.** The growth analysis runs on #24's screen (347 markup removals, 74,318 units) so every fit reconciles with `analysis/frequency/fits.tsv` row for row; the entity analysis runs on the #27/#30 screen (the general tag-run detector on top, 588 removals, 74,077 units) so every entity count reconciles with #30's `entity_glossary.tsv` row for row. The 241-occurrence difference is the `href`/`mailto`/`colspan` tag furniture the general detector removes and #19's ledger does not list.
- **Heaps' law** is fitted exactly as #24 fits it (OLS on the log-log cumulative type curve at 400 checkpoints), so the point estimates reproduce `fits.tsv`. The uncertainty is a leave-one-page-out jackknife for the book and leave-one-1000-token-block-out for the references - a resampling that never duplicates a group, which a page bootstrap would (a duplicated page contributes no new types, biasing beta down); the jackknife refit mean reproduces the point estimate.
- **Burstiness** reports three legs per entity: the Fano factor Var(count)/Mean(count) over non-overlapping windows of 100 / 500 / 1000 tokens (1.0 is a Bernoulli placement); the Goh-Barabasi burstiness B = (sigma - mu)/(sigma + mu) of the inter-arrival gaps (0 is geometric gaps); and a one-sided add-one p-value from 200 relocation draws that place the entity's occurrences uniformly at random at its own rate (seeded per term by `SEED + crc32(name)`). A `bursty` verdict needs p <= 0.05 and B > 0.0; `even` is F(1000) < 0.9.
- **Entity positions** are located on the screened stream with #30's own phrase rule: single-unit surfaces match exactly; multi-word surfaces require every word capitalized and whitespace-adjacent in the page body, with a trailing numeric token read as a designator. The located counts equal #30's `n_occurrences` for 98 of 98 entities (98/98 in this build).

## 1. Heaps' law: vocabulary growth vs text length

| corpus | N | V | beta | log K | R^2 | beta LOO sd | beta LOO range | V@10k | V@25k | V@50k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| book_units | 74,318 | 5,868 | 0.603 | 1.970 | 0.995 | 0.0008 | 0.600-0.609 | 1832 | 3372 | 4853 |
| book_units_lower | 74,318 | 5,502 | 0.598 | 1.959 | 0.995 | 0.0007 | 0.595-0.604 | 1734 | 3172 | 4554 |
| book_parts | 80,015 | 3,687 | 0.499 | 2.653 | 0.982 | 0.0009 | 0.494-0.506 | 1404 | 2450 | 3149 |
| book_english | 66,506 | 3,209 | 0.502 | 2.585 | 0.978 | 0.0009 | 0.496-0.509 | 1347 | 2313 | 2884 |
| book_nonenglish | 7,812 | 2,293 | 0.818 | 0.472 | 0.996 | 0.0011 | 0.814-0.829 | - | - | - |
| book_coinage | 7,361 | 2,245 | 0.835 | 0.354 | 0.995 | 0.0009 | 0.831-0.839 | - | - | - |
| book_ocr | 127 | 15 | 0.570 | 0.127 | 0.943 |  | - | - | - | - |
| en_pride_and_prejudice | 74,318 | 5,579 | 0.540 | 2.629 | 0.985 | 0.0013 | 0.539-0.551 | 2136 | 3413 | 4739 |
| en_moby_dick | 74,318 | 10,042 | 0.681 | 1.603 | 0.990 | 0.0015 | 0.672-0.684 | 2895 | 5092 | 7728 |
| en_frankenstein | 74,318 | 6,975 | 0.598 | 2.232 | 0.984 | 0.0009 | 0.595-0.601 | 2532 | 4159 | 5853 |
| en_dracula | 74,318 | 6,597 | 0.622 | 1.885 | 0.993 | 0.0011 | 0.618-0.625 | 2111 | 3557 | 5458 |
| en_sherlock_holmes | 74,318 | 6,788 | 0.605 | 2.072 | 0.996 | 0.0017 | 0.605-0.619 | 2184 | 3691 | 5548 |
| en_heart_of_darkness | 38,774 | 5,455 | 0.672 | 1.576 | 0.993 | 0.0024 | 0.667-0.686 | 2520 | 4336 | - |
| en_dorian_gray | 74,318 | 6,719 | 0.628 | 1.777 | 0.996 | 0.0024 | 0.622-0.645 | 1905 | 3480 | 5304 |
| en_ulysses | 74,318 | 12,108 | 0.747 | 1.041 | 0.997 | 0.0014 | 0.742-0.749 | 2638 | 5781 | 8895 |
| **English mean** (8 refs) | | | 0.637 | | | | | | | |

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

| cast measure | first third | second third | last third |
|---|---:|---:|---:|
| entities first attested here | 64 | 17 | 17 |
| cumulative share of the cast seen by the end of the third | 65% | 86% | 100% |

## 3. Burstiness of entity use

Per-entity table: `entity_burstiness.tsv` (39 rows, n >= 8, sorted by F(1000)).

| entity | role | n | pages | B | F(100) | F(1000) | null F | p | verdict | top page |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| retro-ADAM | persona | 165 | 43 | 0.667 | 1.37 | 9.82 | 0.97 | 0.005 | bursty | 149 (7) |
| ADAM Doll | creature | 166 | 62 | 0.551 | 1.34 | 6.32 | 0.99 | 0.005 | bursty | 217 (7) |
| ADAM | persona | 303 | 123 | 0.374 | 1.14 | 4.25 | 0.98 | 0.005 | bursty | 57 (8) |
| NDROID | persona | 20 | 13 | 0.380 | 0.97 | 3.73 | 0.93 | 0.005 | bursty | 116 (4) |
| HIV | persona | 13 | 9 | -0.262 | 0.98 | 3.29 | 0.98 | 0.005 | neutral | 171 (3) |
| DNA=channel | machine-system | 75 | 55 | 0.370 | 0.95 | 2.31 | 0.97 | 0.005 | bursty | 128 (3) |
| cadaver | persona | 441 | 182 | 0.037 | 0.81 | 2.14 | 0.97 | 0.005 | bursty | 161 (7) |
| ToKAGE | persona | 127 | 84 | 0.240 | 1.06 | 2.01 | 0.97 | 0.005 | bursty | 98 (4) |
| dustNirverna | persona | 15 | 12 | 0.438 | 1.11 | 2.00 | 0.93 | 0.005 | bursty | 70 (2) |
| Cadaver City | place | 191 | 118 | 0.207 | 0.87 | 1.81 | 0.99 | 0.005 | bursty | 80 (4) |
| BABEL=TV | machine-system | 16 | 14 | 0.269 | 1.23 | 1.78 | 0.91 | 0.005 | bursty | 28 (2) |
| Placenta | persona | 71 | 50 | 0.281 | 1.07 | 1.76 | 0.97 | 0.005 | bursty | 216 (5) |
| ToK | persona | 12 | 8 | 0.295 | 0.98 | 1.67 | 1.00 | 0.015 | bursty | 4 (2) |
| cyberBuddha | persona | 14 | 11 | 0.009 | 1.41 | 1.67 | 0.95 | 0.010 | bursty | 219 (4) |
| NDRO | persona | 23 | 20 | 0.050 | 1.06 | 1.65 | 0.95 | 0.005 | bursty | 117 (2) |

The burst profile splits the cast. The **foreground** arrives in arcs: `retro-ADAM` (F(1000) 9.82, B 0.667) is confined to pages 87-176 - a late-book sequence, not a texture spread over the novel. The **background** is the opposite of bursty: 6 ambient entities are spread *more evenly than chance* (`SM` F(1000) 0.78, `Cadaver City-city` F(1000) 0.84, `City-city` F(1000) 0.84, `Sato Corporation` F(1000) 0.88, ...). The high-frequency textures sit between the two regimes - `cadaver` (441 occurrences over 182 pages) F(1000) 2.14, `ADAM` F(1000) 4.25 - over-dispersed in the strict test but a fraction of the arcs' concentration. The ambient layer is the book's climate; the bursty layer is its weather.

What the bursts *are* is checkable by eye. These are the densest single-page entity clusters in the glossary, shown in the unit stream:

- **ADAM** - 8 occurrences on p57 (of 303 total): `...of city body gland like the ADAM Doll that makes the body gland...`
- **ADAM** - 7 occurrences on p72 (of 303 total): `...the swastika contraction of the universe ADAM Doll assassination speed to the body...`
- **ADAM** - 7 occurrences on p217 (of 303 total): `...like the machine nature of the ADAM Doll that quiesced Quantum suicide machine...`
- **ADAM Doll** - 7 occurrences on p217 (of 166 total): `...like the machine nature of the ADAM Doll that quiesced Quantum suicide machine...`

A measured burst is not always narrative. On the corpus's disclosed OCR recognition loop pages (page 90: `cadaver` x4, `Cadaver City` x3; page 219: `cyberBuddha` x4, `ADAM` x3, `ADAM Doll` x3, `VTR` x3, `konganic-eye` x3) the entity clusters are machine artifacts - page 219 repeats `...the technology of murder href mailto konganic-eye of cyberBuddha konganic-eye of cyberBuddha and was sutured to...`, and the #4/#34 page-90 loop mints the one-off types behind the seg-3 novelty spike. They are reported as bursts because they are, with the cause named, rather than filtered out.

## 4. Controls

The same estimator on non-entity repetitions: the ten most frequent English unit types and the ten most frequent non-English (coinage) types (`controls_burstiness.tsv`). Even this baseline is over-dispersed, because the corpus's own density varies by page (spine pages, OCR loops) - the Bernoulli null controls rate, not the envelope - so the entity claim is carried by *magnitude*, not by significance alone.

- **English word control** (10 types): 4 bursty, median F(1000) 1.81, max 3.46, median B -0.011.
- **Coinage-type control** (10 types): 8 bursty, median F(1000) 2.01, max 9.82, median B +0.182.

| kind | term | n | B | F(100) | F(1000) | null F | p | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| english_word | the | 9740 | -0.128 | 0.68 | 1.62 | 0.99 | 0.005 | neutral |
| english_word | of | 8096 | -0.145 | 0.77 | 1.95 | 0.98 | 0.005 | neutral |
| english_word | that | 2456 | -0.027 | 0.96 | 1.93 | 0.99 | 0.005 | neutral |
| english_word | to | 1744 | 0.003 | 1.16 | 3.46 | 0.99 | 0.005 | bursty |
| english_word | is | 944 | -0.019 | 1.22 | 2.62 | 1.00 | 0.005 | neutral |
| english_word | a | 925 | 0.018 | 0.97 | 1.68 | 0.97 | 0.005 | bursty |
| english_word | The | 847 | 0.040 | 0.89 | 1.63 | 1.00 | 0.010 | bursty |
| english_word | was | 713 | 0.132 | 1.07 | 1.94 | 0.96 | 0.005 | bursty |
| english_word | in | 568 | -0.046 | 0.85 | 1.12 | 0.96 | 0.214 | neutral |
| english_word | internal | 554 | -0.002 | 0.80 | 0.91 | 0.97 | 0.637 | neutral |
| coinage_type | boy-roid | 315 | 0.399 | 0.95 | 2.08 | 0.98 | 0.005 | bursty |
| coinage_type | soul/gram | 306 | 0.494 | 0.87 | 2.34 | 0.96 | 0.005 | bursty |
| coinage_type | fecundates | 289 | 0.124 | 1.14 | 2.10 | 0.98 | 0.005 | bursty |
| coinage_type | cold-blooded | 188 | 0.083 | 0.87 | 1.43 | 0.97 | 0.010 | bursty |
| coinage_type | retro-ADAM | 165 | 0.667 | 1.37 | 9.82 | 0.97 | 0.005 | bursty |
| coinage_type | defleshed | 155 | 0.094 | 0.97 | 1.28 | 1.00 | 0.075 | neutral |
| coinage_type | gene=TV | 142 | 0.119 | 0.89 | 1.50 | 0.97 | 0.005 | bursty |
| coinage_type | vital=serum | 128 | 0.440 | 0.98 | 2.02 | 0.94 | 0.005 | bursty |
| coinage_type | ToKAGE | 127 | 0.240 | 1.06 | 2.01 | 0.97 | 0.005 | bursty |
| coinage_type | respires | 124 | 0.065 | 1.01 | 1.24 | 0.97 | 0.080 | neutral |

The coinage controls are as bursty as the entities (8/10) - the coinages are the book's motif vocabulary (`boy-roid`, `soul/gram`, `vital=serum`), so *named* vocabulary in general clusters; the English-word control is the least bursty population and the right baseline for "ordinary language", and the aggregate entity channel (3.02) sits well above both.

## 5. Limits

- The Bernoulli null controls for rate, not for the corpus's page-length envelope: a page-rate-preserving null would be stricter, so the p-values read as *over-dispersion* while the F(1000) magnitude - not the p alone - carries the claim.
- Entity occurrences overlap where one glossary entity contains another (`Sato` inside `Sato Corporation`); the aggregate channel counts distinct positions, the per-entity rows do not de-duplicate.
- The matched-N comparison is #24's convention: English references are truncated to the book's token count, so their full-length growth (Ulysses to N = 270k) is not compared here.
- The growth curves are in-sample; the book has no held-out text. The jackknife is an uncertainty statement, not a prediction interval.

Reproduce with `python work/burstiness_heaps.py`; suite `python work/test_burstiness_heaps.py`. Offline on the pinned cache; deterministic (fixed seeds, no clock), rebuild byte-identical.
