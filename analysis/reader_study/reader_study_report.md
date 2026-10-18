# Human jury / reader study design (#75)


Generated: deterministic (no clock read by any statistic)


**This report contains no human data.** The reader judgments in it are *simulated* by a disclosed generative model, seeded (75), and exist to prove the instrument and the analysis pipeline work before a reader is recruited. They are not evidence about readers, and no claim in this report rests on them.

## 1. What was designed

The plan's **L9.7** item -- *have readers rate glosses for fidelity to the felt text* -- needs two things that did not exist: an **instrument** a reader can answer, and a **sampling** rule that says which glosses get rated and against what comparator. Both are here, along with the pre-registered analysis (`protocol.md`) and a simulated pilot that exercises every statistic the real run will use.

The frame is a **census** of the gloss population over a stratified clause sample: 35 candidate cells (5 clauses x [1 identity control + 1 #62 gloss + 5 LLM strategies]) plus 10 trap cells. Because the instrument compares a gloss against the clause it translates and against the other readings of the same clause, a reader cannot score a gloss without a comparator; that is the design property #63 and #64 both needed and neither had.

## 2. The instrument

| item | type | role | prompt | anchors |
|---|---|---|---|---|
| `felt_fidelity` | scale 0-7 | primary | How closely does this wording carry the feeling of the clause it translates? | 0 = it carries nothing of the feeling / 7 = it carries the feeling exactly |
| `fidelity_confidence` | scale 0-7 | calibration | How sure are you of that judgement? | 0 = guessing / 7 = completely sure |
| `reading_appeal` | scale 0-7 | diagnostic | How much would you want to keep reading text written this way? | 0 = not at all / 7 = very much |
| `trap_overcorrected` | scale 0-7 | trap | This wording has smoothed the clause into ordinary English. Rate how faithful that smoothing is to the clause's oddity. | 0 = completely unfaithful to the oddity / 7 = fully faithful |
| `trap_invented` | scale 0-7 | trap | This wording adds a named character or event the clause does not contain. Rate how faithful that addition is. | 0 = completely unfaithful / 7 = fully faithful |
| `pair_preference` | forced choice | calibration | Which of these two wordings carries the clause's feeling better? | the first / the second / equally well |
| `pair_invented_plot` | forced choice | calibration | Does either wording commit to a concrete story event the clause only hints at? | neither / the first / the second / both |

The traps are the instrument's teeth. `trap_overcorrected` carries #63's `plain_english` reading -- the strategy #63's own checks most often fail and #74 measures as least faithful overall -- and asks how faithful its smoothing is to the clause's oddity; `trap_invented` asks the same about an addition the clause does not make. A reader who rates those high is not measuring fidelity, and the attention check says so rather than averaging over it.

## 3. Sampling and counterbalancing

| block | clause | stratum | page | cells shown | LLM draw offset |
|---|---|---|---|---|---|
| 0 | 10 | plain_control | 4 | 6 | 0 |
| 1 | 101 | quoting_frame | 6 | 6 | 2 |
| 2 | 1003 | operator_dense | 37 | 6 | 4 |
| 3 | 1005 | entity_named | 37 | 6 | 1 |
| 4 | 2789 | residue | 108 | 6 | 3 |

Every block shows the identity control and #62's gloss (the comparators without which a rating is uninterpretable), a rotated draw of 2 LLM readings, and the clause's 2 trap cells. The traps are instrument items: a reader who is never asked to judge an over-corrected reading cannot be checked for judging fidelity rather than appeal. The LLM rotation (`PAIR_OFFSETS`) and the within-block rotation are both functions of the block index, so the design is reproducible from the frame alone and no reading is systematically first.

## 4. The simulated pilot

**No human was involved in these numbers.** For each reader the model draws a leniency (``ability``), a noise level and an order effect from seeded Gaussians; each rating is that item's expectation for that cell, shifted by the reader's ability, displaced by the order effect when the cell was shown first, wobbled by the reader's noise, clipped to 0-7 and **rounded to an integer**, because the instrument presents a 7-point scale and a real response is an integer. Every expectation is a *design assumption*, published in the source as ``FEELING_OFFSET`` / ``APPEAL_OFFSET`` / ``TRAP_OFFSET``: the felt-fidelity table is anchored to #74's channels but is deliberately **not** a monotone function of them, because the study's premise is that readers are not machine measures.

1800 integer ratings over 5 blocks x 12 readers x 5 items, plus 300 forced-choice judgements.

**What the pilot does not generate:** the second forced-choice item (`pair_invented_plot`) is part of the instrument a human run must administer, and part of what this design commits to, but the simulation derives `pair_preference` only. Simulating the other one would mean inventing a reader's judgement about whether a wording commits to a plot event the clause merely hints at -- which is the one judgement the study exists to collect -- so it is left blank rather than filled with an assumption.

### 4.1 Per-cell primary scores

| clause | reading | kind | n | mean | sd | 95% CI | gap vs identity |
|---|---|---|---:|---:|---:|---|---:|
| 10 | `interlinear_gloss` | gloss | 12 | 3.500 | 0.522 | [3.250, 3.750] | -2.917 |
| 10 | `identity` | identity | 12 | 6.417 | 0.515 | [6.167, 6.667] | +0.000 |
| 10 | `llm_literal_operator` | llm | 12 | 4.917 | 0.289 | [4.750, 5.000] | -1.500 |
| 10 | `llm_register_paraphrase` | llm | 12 | 5.417 | 0.515 | [5.167, 5.667] | -1.000 |
| 1003 | `llm_entity_consistent` | llm | 12 | 4.333 | 0.651 | [3.917, 4.667] | -2.250 |
| 1003 | `interlinear_gloss` | gloss | 12 | 3.167 | 0.389 | [3.000, 3.417] | -3.417 |
| 1003 | `identity` | identity | 12 | 6.583 | 0.515 | [6.333, 6.833] | +0.000 |
| 1003 | `llm_literal_operator` | llm | 12 | 4.917 | 0.289 | [4.750, 5.000] | -1.667 |
| 1005 | `interlinear_gloss` | gloss | 12 | 3.583 | 0.515 | [3.333, 3.833] | -3.083 |
| 1005 | `identity` | identity | 12 | 6.667 | 0.492 | [6.417, 6.917] | +0.000 |
| 1005 | `llm_register_paraphrase` | llm | 12 | 5.500 | 0.522 | [5.250, 5.750] | -1.167 |
| 1005 | `llm_scene_summary` | llm | 12 | 3.500 | 0.674 | [3.167, 3.917] | -3.167 |
| 101 | `interlinear_gloss` | gloss | 12 | 3.250 | 0.452 | [3.000, 3.500] | -3.167 |
| 101 | `identity` | identity | 12 | 6.417 | 0.515 | [6.167, 6.667] | +0.000 |
| 101 | `llm_plain_english` | llm | 12 | 3.917 | 0.289 | [3.750, 4.000] | -2.500 |
| 101 | `llm_scene_summary` | llm | 12 | 3.750 | 0.622 | [3.417, 4.083] | -2.667 |
| 2789 | `llm_entity_consistent` | llm | 12 | 4.417 | 0.515 | [4.167, 4.667] | -2.083 |
| 2789 | `interlinear_gloss` | gloss | 12 | 3.250 | 0.622 | [2.917, 3.583] | -3.250 |
| 2789 | `identity` | identity | 12 | 6.500 | 0.522 | [6.250, 6.750] | +0.000 |
| 2789 | `llm_plain_english` | llm | 12 | 3.917 | 0.515 | [3.667, 4.167] | -2.583 |

### 4.2 The contrast the study exists for

The identity control minus the mean of the other readings, pooled over 5 clauses (per clause: +1.81, +2.44, +2.47, +2.78, +2.64):


**observed contrast +2.4278, permutation p = 0.0015** (2000 draws). The null **permutes the control label**: within each clause a reading is drawn at random to play the role of the identity control and the contrast is recomputed. That destroys exactly the claim being tested -- that the ordering follows which reading *is* the clause -- while keeping every clause's values, size and difficulty. A contrast that survives its own label permutation is following the reading, not the clause.

### 4.3 Human vs machine

Spearman correlation of the human per-reading mean against #74's published `aggregate_fidelity`: **rho = +0.8929, permutation p = 0.0135** over 7 readings.


| reading | human mean | #74 aggregate fidelity |
|---|---:|---:|
| `identity` | 6.517 | 1.000 |
| `interlinear_gloss` | 3.350 | 0.250 |
| `llm_entity_consistent` | 4.375 | 0.699 |
| `llm_literal_operator` | 4.917 | 0.823 |
| `llm_plain_english` | 3.917 | 0.102 |
| `llm_register_paraphrase` | 5.458 | 0.909 |
| `llm_scene_summary` | 3.625 | 0.417 |

### 4.4 Inter-reader agreement

| item | alpha | null 5% | null 95% | p |
|---|---:|---:|---:|---:|
| `felt_fidelity` | 0.7898 | -0.0295 | 0.035 | 0.0050 |
| `fidelity_confidence` | 0.4935 | -0.0322 | 0.0386 | 0.0050 |
| `reading_appeal` | 0.7653 | -0.0302 | 0.0342 | 0.0050 |
| `trap_overcorrected` | 0.0073 | -0.038 | 0.0231 | 0.1542 |
| `trap_invented` | -0.0501 | -0.0425 | 0.0266 | 0.9950 |

Krippendorff's alpha with the **interval** difference function (`delta^2 = (c - k)^2`, the pre-registered metric): a 0-7 rating scale carries equal intervals, so a 6 against a 5 must cost far less than a 6 against a 0. The **ordinal** and **nominal** metrics are implemented and disclosed alongside it, because a case has to be made rather than assumed: with only two categories observed, the ordinal metric cannot tell a 0-vs-6 disagreement from a 5-vs-6 one, and the nominal metric prices every non-identical pair the same -- on a finely-graded scale that drives alpha toward zero however well the readers agree. Each alpha is read against a null that re-groups the values (every reader's scores shuffled across the cells), so a number is only impressive against what chance produces on the same data. The implementation is pinned against the worked example in Krippendorff's 2011 note (binary data, two observers, no missing values, `o_00=10, o_01=o_10=4, o_11=2` -> alpha = 2/21) by the regression suite.

**The two trap items sit at the floor of the scale and show no agreement, and that is a property of the instrument, not of the readers.** Two reasons, both disclosed: every trap cell carries the same expectation, so a scale statistic has no between-cell signal to track; and an expectation of 0.6-0.8 on a 0-7 scale leaves the responses crowded into 0, 1 and 2, where a seven-point scale has almost no resolution left to distinguish opinions.  It is exactly why the traps are analysed as **pass/fail against a known answer** (the attention check) rather than as a scale -- the design never asks the traps to carry a scale statistic, and the report does not pretend otherwise.

### 4.5 Between-reading effect size

Kruskal-Wallis H = 18.1021, eta-squared = 0.9930 over 7 readings -- the non-parametric leg pre-registered as analysis 6.

### 4.6 Appeal against fidelity

| reading | mean felt fidelity | mean appeal | direction |
|---|---:|---:|---|
| `identity` | 6.517 | 4.983 | fidelity > appeal |
| `interlinear_gloss` | 3.350 | 5.100 | appeal > fidelity |
| `llm_entity_consistent` | 4.375 | 3.792 | fidelity > appeal |
| `llm_literal_operator` | 4.917 | 3.292 | fidelity > appeal |
| `llm_plain_english` | 3.917 | 6.125 | appeal > fidelity |
| `llm_register_paraphrase` | 5.458 | 5.667 | about equal |
| `llm_scene_summary` | 3.625 | 4.708 | appeal > fidelity |

This is the table the diagnostic item exists for. A reading that is more appealing than it is faithful is not a better reading -- it is the over-correction arm, and the instrument is built so that preference cannot be mistaken for fidelity.

### 4.7 Per stratum

| stratum | cells | mean felt fidelity | sd |
|---|---:|---:|---:|
| entity_named | 4 | 4.812 | 1.543 |
| operator_dense | 4 | 4.750 | 1.422 |
| plain_control | 4 | 5.062 | 1.214 |
| quoting_frame | 4 | 4.333 | 1.417 |
| residue | 4 | 4.521 | 1.403 |

Reported per stratum and never pooled: #63 stratified the sample so each check would have a case that can fail, and the study keeps that property rather than averaging it away.

### 4.8 Reader exclusions

| reader | failed checks | excluded |
|---|---|---|
| reader_a | none | False |
| reader_b | none | False |
| reader_c | att_overcorrected | True |
| reader_d | none | False |
| reader_e | none | False |
| reader_f | none | False |
| reader_g | none | False |
| reader_h | none | False |
| reader_i | none | False |
| reader_j | none | False |
| reader_k | none | False |
| reader_l | none | False |

## 5. What the pilot shows and what it cannot

**What it shows.** The instrument produces interpretable orderings: the identity control sits above every reading, the ranking tracks #74's machine-measured fidelity, and the traps and attention checks separate a reader who is judging fidelity from one who is not (one of 12 simulated readers is flagged by `att_overcorrected`, which is the check working). The analysis pipeline -- bootstrap CIs, label-permuted contrast, label-permuted machine correlation, alpha against a re-grouping null -- runs end to end on the same code a human run would use.

**What it cannot show.** Anything about readers. The generative model is an assumption of this module, not a measurement. In particular the pilot says nothing about whether real readers agree with each other, whether they use the whole scale, or whether they prefer the over-corrected readings -- which is exactly the question the traps exist to ask. Those need people.

## 6. Anti-pattern audit (L9.8)

* **Hallucinated coherence**: the instrument never asks whether a reading is coherent; it asks whether the reader can recover the clause's feeling, with the original clause on screen, and plants a fluent-but-invented case (`trap_invented`) that must score low.
* **LLM circularity**: no model generates or grades anything here. The only LLM output in the frame is #63's pinned capture, treated as a candidate; the judge is a reader.
* **Over-correction**: `trap_overcorrected` measures a reader's appetite for repaired English as data about the reading's appeal, kept separate from the primary item rather than corrected away.
* **Single-method certainty**: 7 items, seven pre-registered analysis legs, three null models, per-stratum reporting and a published protocol; no single number is the verdict, and the traps are never asked to carry one.
* **Residue-dropping**: the frame itemises the cells a block does not draw, the traps are counted and excluded rather than dropped, and reader exclusions are published with their reasons.

## 7. Reconciliation

| quantity | earlier | here | note |
|---|---|---:|---|
| n_interlinear_rows | 5614 | 5614 | #33/#62 clause count (exact match required) |
| n_sample_clauses | 5 | 5 | #63 stratified sample size (exact match required) |
| n_strategies | 5 | 5 | #63 strategies (exact match required) |
| n_gloss_rows | 25 | 25 | #63 sampled glosses (exact match required) |
| n_frame_cells | 45 | 45 | identity + gloss + 5 strategies + 2 traps per clause |
| n_candidate_cells | 35 | 35 | rateable candidate cells (traps excluded) |
| n_trap_cells | 10 | 10 | planted trap cells, excluded from fidelity statistics |
| n_blocks | 5 | 5 | one clause block per sampled clause |
| n_sampled_pairs | 30 | 30 | (clause, gloss) cells shown to a reader: 5 blocks x 6 cells (identity + #62 gloss + 2 rotated LLM comparators + 2 trap cells) |
| n_ratings | 1800 | 1800 | simulated pilot rows (reader x cell x scale item); a human run replaces this row, it does not reinterpret it |
| n_pairwise_judgements | 300 | 300 | simulated pairwise judgements over adjacent cells |

## 8. Verification

```bash
python work/reader_study.py            # writes analysis/reader_study/
python work/test_reader_study.py       # regression suite
python -m ruff check work/reader_study.py work/test_reader_study.py
```

Deterministic: no clock, no network, no model backend, seed 75, byte-identical rebuild (only the report's own text moves, and no statistic reads it).

## 9. Handoff

The next step is recruitment, not design: `protocol.md` is the pre-registration, `instrument.tsv` the verbatim items, `frame.tsv` the sampling frame and `blocks.tsv` the counterbalanced packets. A human run replaces `pilot_ratings.tsv` with its own rows and runs the same functions; the pilot's numbers are then retired, not reinterpreted.
