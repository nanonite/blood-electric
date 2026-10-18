# Compression test for candidate readings (#68)


The plan's L8.7 test, run. Every candidate reading the repository has published is scored by whether it makes the corpus more **compressible** and more **predictable**, under one common protocol, with the identity (no-reading) control, a length-matched deletion control, a pairing shuffle null, and a held-out split. Reproduce with ``python work/reading_compression.py``; regression suite ``python work/test_reading_compression.py``. Schema `reading-compression/1`.


## 1. Method: why raw bits/char is not enough


#48 measured the book against matched English and concluded that raw compressibility *cannot discriminate between readings* -- the book is already more compressible than English prose, and every reading of it would be too. This harness therefore never reports a raw bits/char number on its own. Each reading is scored by three legs that remove the confounds in turn:


1. **Length-matched deletion control (the primary leg).** For every covered clause the control is a deterministic, position-preserving letters-word-subsample of the *original* clause down to the reading's letters-word count (seed `68:<reading>:<clause>:<seed_k>`), compressed by the identical path and repeated over 5 seeds so its own seed noise is reported. The number is ``reading - control``: a reading whose only effect is dropping words scores ~0, only restructuring scores negative. **This is the leg that carries the verdict** (section 6 ranks by it), because it is the leg that answers the task's question -- does the reading shorten the description of the book? -- with the deletion confound controlled. **Its limit is disclosed, not hidden:** a reading with at least as many words as its clause gets *no* subsample, because there is nothing to delete, and the control is then the original itself. The corpus's readings are word-count preserving or expanding far more often than not -- operator removal changes the characters *between* words, not the letters, so v3/v4/gloss match their clause's word count on most clauses -- so for them the control is the original on the majority of clauses and the delta reduces toward the raw bits/char delta. The live-clause count is published per reading in ``controls_and_nulls.tsv`` (``control_fallback_clauses``) and section 3 states what the leg can and cannot show for each reading as a result.
2. **Pairing shuffle null.** The reading's clause texts are permuted across its covered clause ids (3 permutations, fixed seeds). The text multiset is preserved and the clause<->text pairing is destroyed, so a ``null_z`` far below the reading's own score means the gain depends on *which* text belongs to *which* clause.
3. **Held-out predictability.** A character trigram with Jelinek-Mercer interpolation (#56's estimator, lambda tuned on a train-internal validation split) scores text it never trained on. One **fixed model** (trained on the identity train clauses only) scores every reading, so readings are compared by a single shared scorer; the full-book readings additionally get an **adaptive** model trained on their own train clauses, scored on their held-out clauses (``self``) and on the untouched originals (``cross`` -- a coverage diagnostic, not a quality score). The fixed-model leg is paired against the identity text clause by clause with a sign test (#50's pattern).


**Tokenization and units (choices, not facts).** The unit is the clause (the #33 segmentation, 5,614 clauses), so readings are compared on identical text spans. The compression stream is the reading's clause texts joined by newlines in book order, case-folded, UTF-8; the six #48 compressors are reused from ``compression_mdl`` so the numbers reconcile with #48's frozen table. A **word** is a maximal run of letters -- not a whitespace chunk, which would count ``soul/gram`` as one word and ``soul gram`` as two and so make an operator-removal reading look longer than its clause purely because of the tokenizer. Predictability is measured in **characters**, with no word tokenizer at all: a character model cannot be blinded by a reading's orthography and applies the same ruler to a gloss and a source clause. The held-out split is every 5th clause (20 %); clause order is page order, so the held-out set spans 219 of 225 pages rather than a contiguous block.


## 2. The candidate readings


| reading | kind | provenance | clauses |
|---|---|---|---:|
| `identity` | control | interlinear original column | 5614 |
| `v3_operators_to_space` | transform | #13 v3 via interlinear | 5614 |
| `v4_operators_to_period` | transform | #13 v4 via interlinear | 5614 |
| `interlinear_gloss` | gloss | #62 gloss column | 5614 |
| `llm_entity_consistent` | llm | #63 glosses.tsv[entity_consistent] | 5 |
| `llm_literal_operator` | llm | #63 glosses.tsv[literal_operator] | 5 |
| `llm_plain_english` | llm | #63 glosses.tsv[plain_english] | 5 |
| `llm_register_paraphrase` | llm | #63 glosses.tsv[register_paraphrase] | 5 |
| `llm_scene_summary` | llm | #63 glosses.tsv[scene_summary] | 5 |

``identity`` is the control, not a candidate: every delta is measured against it on the same clause set. The five ``llm_*`` readings cover 5 clauses each (#63's stratified sample), so they are scored on the sample population and never pooled with full-book readings -- the two populations are ranked separately.


## 3. Leg A: description length and its controls


| reading | gzip-9 bits/char | vs length control | control sd | control fallback | null mean | null z | chars | words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `identity` | 2.1066 | 0.0000 | 0.0000 | 5614/5614 | 2.1753 | -30.596 | 472919 | 80334 |
| `v3_operators_to_space` | 2.0533 | -0.0647 | 0.0004 | 5470/5614 | 2.1214 | -22.315 | 467680 | 79716 |
| `v4_operators_to_period` | 2.0766 | -0.0413 | 0.0004 | 5470/5614 | 2.1442 | -16.047 | 472399 | 79716 |
| `interlinear_gloss` | 2.0508 | -0.0671 | 0.0004 | 5479/5614 | 2.1150 | -35.769 | 470948 | 80279 |
| `identity` | 5.0042 | 0.0000 | 0.0000 | 5/5 | 5.0432 | -2.475 | 478 | 85 |
| `v3_operators_to_space` | 4.8851 | -0.1191 | 0.0000 | 5/5 | 4.9248 | -4.950 | 470 | 85 |
| `v4_operators_to_period` | 4.9129 | -0.0913 | 0.0000 | 5/5 | 4.9295 | -0.612 | 482 | 85 |
| `interlinear_gloss` | 5.0678 | 0.0636 | 0.0000 | 5/5 | 5.0904 | -1.069 | 472 | 85 |
| `llm_entity_consistent` | 4.2018 | -0.8024 | 0.0000 | 5/5 | 4.1815 | 7.778 | 1447 | 262 |
| `llm_literal_operator` | 4.1624 | -0.8418 | 0.0000 | 5/5 | 4.1686 | -0.784 | 1724 | 299 |
| `llm_plain_english` | 3.9588 | -1.0454 | 0.0000 | 5/5 | 3.9436 | 0.981 | 1748 | 320 |
| `llm_register_paraphrase` | 5.0083 | 0.0041 | 0.0000 | 5/5 | 5.0468 | -2.475 | 484 | 86 |
| `llm_scene_summary` | 5.1360 | -0.0395 | 0.0350 | 4/5 | 5.1147 | 1.069 | 500 | 86 |

``control fallback`` is the count of covered clauses where the reading has at least as many words as its clause, so the control *is* the original and no subsample happens -- the leg is degenerate on those clauses. For the full-book readings that is the overwhelming majority: the control is live on only 144 of 5614 clauses for v3/v4 (fallback 5470) and 135 of 5614 for the gloss (fallback 5479), because operator removal changes the characters between words, not the letters. **So for the three operator/gloss readings the primary leg is, in practice, the raw bits/char delta against the identity stream -- a length-normalized comparison, but not a length-matched one.** The honest reading of the column is therefore: these readings do not win by deleting, because there is almost nothing to delete; the control demonstrates that rather than manufactures a correction. Where the control is live (those 135 and 144 clauses, and any reading a future pipeline produces that shortens) the word-matching is exact and the delta is a true restructuring measure.


``control sd`` is the spread of the control over its 5 seeds. On the full-book population it is at most 4.5e-04 bits/char, so the full-book deltas sit two orders of magnitude above the control's own noise. On the 5-clause sample it is 0.0000 for almost every row -- and that is not precision, it is the fallback again: for four of the five LLM strategies *every* covered clause falls back, so all seeds produce the identical stream and the sd is exactly zero (the exception is ``llm_scene_summary``, with one genuinely shorter clause and an sd of ~0.04). The sample population is reported descriptively and never ranked against the full book. ``null_z`` compares the reading with its own permuted pairings: values near 0 mean the gain does not depend on the clause<->text pairing.


## 4. Leg C: held-out predictability


Two character-trigram evaluations, both held out (nothing scored was in the training text):


* **fixed model (all readings)** -- one model, trained on the *identity* train clauses only, scores every reading's held-out clauses and the identity's.  A single shared scorer, so the ``gain vs identity`` column is an apples-to-apples comparison of how predictable each reading's text is to a model of the book.
* **adaptive (full-book readings)** -- a model trained on each reading's own train clauses, scored on its held-out clauses (``self``) and, as a coverage diagnostic, on the untouched held-out originals (``cross``).


| reading | lambda | self bits/char | fixed-model bits/char | fixed gain vs identity | cross bits/char (coverage) | improved / worsened | sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `identity` | 0.95 | 2.5380 | 2.5380 | 0.0000 | 2.5380 | n/a / n/a | n/a |
| `v3_operators_to_space` | 0.95 | 2.4989 | 2.5245 | -0.0135 | 2.9782 | 304 / 208 | 2.55e-05 |
| `v4_operators_to_period` | 0.95 | 2.5140 | 2.6938 | 0.1558 | 2.9889 | 27 / 485 | 1.02e-109 |
| `interlinear_gloss` | 0.95 | 2.5029 | 2.5398 | 0.0018 | 2.9819 | 300 / 247 | 0.0261 |

| reading (sample population) | fixed-model bits/char | fixed gain vs identity |
|---|---:|---:|
| `identity` | 4.2380 | 0.0000 |
| `v3_operators_to_space` | 4.2635 | 0.0255 |
| `v4_operators_to_period` | 4.2380 | 0.0000 |
| `interlinear_gloss` | 4.2635 | 0.0255 |
| `llm_entity_consistent` | 5.0583 | 0.8202 |
| `llm_literal_operator` | 6.0786 | 1.8406 |
| `llm_plain_english` | 5.2000 | 0.9620 |
| `llm_register_paraphrase` | 4.1990 | -0.0390 |
| `llm_scene_summary` | 4.9550 | 0.7170 |

The sample population holds 5 clauses, of which the split leaves **one** held out; a sign test on n=1 has no power and is not reported for those rows (see section 5).

External anchor: the untransformed held-out book scores 4.1342 bits/char under a general-English character model trained on #56's reference novels -- the yardstick the readings are trying to beat. The book's own models score it far lower, which is #48's "more compressible than English" finding restated at character level, and the reason a raw bits/char number could never separate two readings of this book.


**Reading the cross column honestly.** A positive ``cross`` value means the reading's own model is *surprised* by the untouched book -- expected and not a defect: a reading that removes operators has never seen an operator, so it pays for one every time. The column is a coverage diagnostic (how much of the book's surface the reading's system accounts for), not a quality score, and it is why the verdict does not rest on it.


The adaptive leg runs only for readings with at least 200 train clauses; the five ``llm_*`` readings (5 clauses each) are below that floor and are scored by the shared fixed model alone, whose gain column carries their predictability score.


## 5. The #63 strategy readings on their own sample


| strategy | gzip-9 bits/char | vs length control | fixed-model gain vs identity |
|---|---:|---:|---:|
| `identity` | 5.0042 | 0.0000 | 0.0000 |
| `v3_operators_to_space` | 4.8851 | -0.1191 | 0.0255 |
| `v4_operators_to_period` | 4.9129 | -0.0913 | 0.0000 |
| `interlinear_gloss` | 5.0678 | 0.0636 | 0.0255 |
| `llm_entity_consistent` | 4.2018 | -0.8024 | 0.8202 |
| `llm_literal_operator` | 4.1624 | -0.8418 | 1.8406 |
| `llm_plain_english` | 3.9588 | -1.0454 | 0.9620 |
| `llm_register_paraphrase` | 5.0083 | 0.0041 | -0.0390 |
| `llm_scene_summary` | 5.1360 | -0.0395 | 0.7170 |

Five clauses per strategy is a **descriptive** comparison, not an inferential one: the held-out split leaves a single clause, so no inferential statistic is reported for this population. These rows exist so the LLM strategies are scored by the same protocol, not so they can be ranked against full-book readings.


## 6. Verdict: the ranked readings


**The primary leg is the description-length delta**, so the table is sorted by ``rank_description_length`` and that rank *is* the verdict. The two predictability legs are corroborating evidence, ranked beside it; ``mean_rank_three_legs`` is an unweighted, disclosed aggregate published only so the reader can see where the legs disagree -- it never overrides the primary rank.


Full-book population (all clauses, lower is better on every leg):


| rank (primary) | reading | length-control delta | fixed-model rank | self rank | mean of 3 | null z |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `interlinear_gloss` | -0.0671 | 3 | 2 | 2.00 | -35.769 |
| 2 | `v3_operators_to_space` | -0.0647 | 1 | 1 | 1.33 | -22.315 |
| 3 | `v4_operators_to_period` | -0.0413 | 4 | 3 | 3.33 | -16.047 |
| 4 | `identity` | 0.0000 | 2 | 4 | 3.33 | -30.596 |

**`interlinear_gloss` wins the declared primary leg** and is therefore the reading that most shortens the description of the book under this protocol. The predictability legs are won by `v3_operators_to_space` (fixed model) and `v3_operators_to_space` (self). 
The legs disagree, and the disagreement is the finding rather than a defect: a reading that rewrites orthography alone is the more *predictable* one, while a reading that also substitutes the termbase is the more *compressible* one. Because no leg is a fidelity instrument, neither winner is certified as *the* reading. Sample population (#63's 5-clause stratified sample):


| rank (primary) | reading | length-control delta | fixed-model gain |
|---:|---|---:|---:|
| 1 | `llm_plain_english` | -1.0454 | 0.9620 |
| 2 | `llm_literal_operator` | -0.8418 | 1.8406 |
| 3 | `llm_entity_consistent` | -0.8024 | 0.8202 |
| 4 | `v3_operators_to_space` | -0.1191 | 0.0255 |
| 5 | `v4_operators_to_period` | -0.0913 | 0.0000 |
| 6 | `llm_scene_summary` | -0.0395 | 0.7170 |
| 7 | `identity` | 0.0000 | 0.0000 |
| 8 | `llm_register_paraphrase` | 0.0041 | -0.0390 |
| 9 | `interlinear_gloss` | 0.0636 | 0.0255 |

## 7. What this does and does not show


- **Compression is evidence about predictability, not proof that an interpretation is true** (L9.8). A reading wins here by making the corpus cheaper to describe; that is a property of the text it produces, not a certificate of fidelity. #63's checks and #64's disagreement log remain the fidelity instruments.
- **The primary leg is a length-matched control that is mostly inert on this corpus.** It subsamples only where a reading is *shorter* than its clause; the readings here are word-count preserving or expanding on 5479 of 5614 clauses (v3/v4/gloss), so for them the control is the original and the delta is effectively the raw bits/char delta. What the control does establish is that these readings win without deleting -- there is almost nothing to delete -- and it remains exact for any future reading that shortens (the 135 shorter gloss clauses are live, 144 for v3/v4, and the word-matching is exact there). The per-reading live-clause counts are in ``controls_and_nulls.tsv``.
- **The control neutralises *arbitrary* deletion, not *structured* deletion.** A reading that keeps only each clause's first word is a pure deletion, yet it beats its random-word control (the book's clause-initial words are strikingly repetitive, so a stream of them compresses far better than an equal number of words sampled at random). This is a true signal about the book's clause-openers, not a confound, and it is also a warning: a degenerate "reading" can score well on the compression leg. The predictability legs and #63's fidelity checks are what stop such a reading from being mistaken for an interpretation.
- **The shuffle null is a pairing test, not a quality test.** A null_z near zero says the gain survives losing the clause<->text pairing (usually because the text multiset dominates); it does not say the reading is good.
- **The character trigram is a crude language model.** It has no lexicon and no long-range structure, so its absolute bits/char is not comparable to GPT-2's (#49/#50); only within-harness deltas are meaningful. GPT-2 itself is not re-run here (weights are not cached in this environment) and #50 already scored the v0/v3/v4 family with it.
- **Held-out split is by clause, not by page.** Adjacent clauses from the same page appear in both halves, so a page-local style can leak across the split; #56's interleaved-page split is the stricter control and this one is the clause-level analogue (its defence is that the split never holds out two adjacent clauses and the fixed-model leg pairs every reading against the identity text clause by clause).
- **The sample population is five clauses, and that is small.** The ``llm_*`` rows are reported for protocol completeness; their ranking is descriptive only. Read on that sample, the two legs point in opposite directions and the opposition is the point: ``llm_plain_english`` wins the description-length leg by a wide margin (-1.05 bits/char against its control, which falls back to the original on all five of its clauses) while being the *least* predictable of all nine readings under the book-trained model (+0.96 bits/char). Plain English is a cheap description of the book's content and a poor predictor of its surface. **A harness that reported only the compression leg would rank it first; a harness that reported only the predictability leg would rank it last.** Neither number is evidence about whether the reading is *true*.


## 8. Triangulation with #48, #50, #56, #62, #63


- **#48** supplies the compressors and the anchor finding (the book is more compressible than English; raw bits/char cannot discriminate readings). This harness reconciles with its frozen ``silence_variants.tsv`` to within the disclosed scope difference and answers its limitation with the length control and nulls.
- **#50** supplies the paired-delta + sign-test pattern and the GPT-2 verdict that v4 (operators to period) most lowers perplexity. **This harness disagrees with that verdict, and the disagreement is reported, not reconciled away:** under the book-trained character model, v4 is the *worst* full-book reading (fixed-model gain 0.1558, 27 improved / 485 worsened held-out clauses), while v3 is the best. The two instruments are different language models with different training distributions: GPT-2 is a general-English web LM that *expects* sentence periods and rewards v4 for supplying them; a model trained on this book expects the book's own operator characters and is surprised when they become periods. Both readings are defensible, which is exactly the single-method-certainty trap L9.8 names -- no one instrument, including this one, settles which reading is *the* reading.
- **#56** supplies the Jelinek-Mercer estimator, the lambda grid, the train-internal-validation tuning discipline, and the reference corpora used for the English anchor.
- **#62** supplies the clause unit, the interlinear columns (original / v3 / v4 / gloss) and the flagship gloss reading.
- **#63** supplies the five LLM strategy readings and its check verdicts; its stratified sample defines the second population.


## 9. Handoff

- **#69 (evaluation/epistemology)** -- this harness is the L9.4 (compression / information gain) instrument; its limits section is written to be quotable in that audit.
- **#62 (interlinear edition)** -- if the gloss line ever changes, re-run this harness; the reconciliation rows will flag the drift.
- **#63 (glossing pipeline)** -- the fixed-model gain column is a cheap per-strategy compression screen for future prompt strategies.

