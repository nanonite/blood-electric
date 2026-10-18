# LLM glossing pipeline -- five strategies, sampled (#63)

Generated: 2026-10-12 02:23 UTC

Capture: 2026-10-10, backend `opencode run --model opencode/space-bunny-free`, pinned mode.

## What this is, and what it is not

Five prompt strategies gloss 5 sampled clauses of *Blood Electric*,
and every returned gloss is put through 4 machine checks that are
computed from the frozen artifacts, not from the model's own
assessment of itself.  The checks are the result; the glosses are
the sample they were computed on.

**This is not a gloss of the book.**  5 clauses carry a gloss here
out of #33's full clause set, and the sample is stratified and
published (`sample.tsv`) rather than chosen for readability.

## The five strategies

* **`literal_operator`** -- literal gloss, operator channel preserved.  Owes: operators verbatim, in order.
* **`register_paraphrase`** -- register-normalized paraphrase.  Owes: operators verbatim, in order.
* **`scene_summary`** -- scene summary.  Owes: operators dropped by instruction.
* **`plain_english`** -- plain-English translation.  Owes: operators verbalised by instruction.
* **`entity_consistent`** -- entity-consistent gloss.  Owes: operators verbatim, in order.

Every prompt carries the same standing constraint block (`STANDING`:
names as given, unresolved surfaces stay marked, no plot, OCR
artefacts as printed, one line out).  Identical across strategies on
purpose -- a differing honesty clause would confound the agreement
statistic with the prompt.

## The sample

| clause | p. | stratum | operators | entities | residue |
|---|---|---|---|---|---|
| 10 | 4 | plain_control | 1 | 0 | - |
| 101 | 6 | quoting_frame | 2 | 1 | - |
| 1003 | 37 | operator_dense | 4 | 3 | - |
| 1005 | 37 | entity_named | 2 | 1 | - |
| 2789 | 108 | residue | 1 | 3 | omanko |

Strata are drawn one clause each, so each check has at least one
case where it can fail.  Within a stratum the quality rank decides
and the tie-break is the clause id compared as text -- `1003`
before `101` -- which is disclosed here rather than hidden: it means
the sample is not the numerically earliest clause of each stratum,
and it is why the sample is published instead of asserted.

* `residue` -- carries a #62 unresolved surface; residue_discipline has teeth here.
* `operator_dense` -- >= 4 operators; operator_fidelity has the most to lose.
* `entity_named` -- names a #30 entity; entity_fidelity has teeth.
* `quoting_frame` -- carries << >>; #12/#14 read these as a quoting frame, so frame_preservation has teeth.
* `plain_control` -- 1-2 operators and no entity: the control that exposes a check which fires on everything.

## Verdicts

| verdict | glosses | share |
|---|---|---|
| pass | 15 | 60% |
| fail | 10 | 40% |

Per strategy:

| strategy | pass | fail |
|---|---|---|
| `literal_operator` | 2 | 3 |
| `register_paraphrase` | 5 | 0 |
| `scene_summary` | 2 | 3 |
| `plain_english` | 4 | 1 |
| `entity_consistent` | 2 | 3 |

Per check, over the rows the check applies to:

| check | rows | applicable | passed | rate |
|---|---|---|---|---|
| `entity_fidelity` | 25 | 25 | 23 | 92% |
| `frame_preservation` | 25 | 6 | 6 | 100% |
| `operator_fidelity` | 25 | 25 | 17 | 68% |
| `residue_discipline` | 25 | 5 | 5 | 100% |

A row a check does not apply to is not counted as a pass: the
applicable column is the denominator, and it is smaller than the rows
column exactly where a check had nothing to judge -- `residue_discipline`
runs only on the glosses whose clause carries an unresolved surface, so
counting the rest as passes would publish a perfect score out of
25 glosses it never judged.  `summary.json` publishes the same
denominators under `check_counts`.

A `pass` is not a good reading.  It is a gloss that broke none of the
rules its own strategy was held to: the operator channel it owed,
the #30 names it was given, the unresolved surfaces it was told to
keep, and the quoting frame it was told to preserve.

## The glosses

### clause 10 (p. 4, `plain_control`) -- `entity_consistent`: **fail**

```text
src    | <<I suck the porno nerve gas of boy-roid
gloss  | << a quoting frame bearing the book's recurring names/labels, holding open the label-line: I suck the porno nerve gas of boy-roid — the speaker draws breath in on the sucking, and what is taken in is nerve gas named porno, drawn out of boy-roid; who or what boy-roid is stays unresolved, as does porno
ops    | <<   ->   <<,/
names  | (none given)
62     | I suck the porno nerve gas of boy-roid
```

* `operator_fidelity`: added 1 (/) -- verbatim, in order

### clause 10 (p. 4, `plain_control`) -- `literal_operator`: **fail**

```text
src    | <<I suck the porno nerve gas of boy-roid
gloss  | << [quoting frame: opens the span that carries the book's recurring names/labels; a frame, not an I/O op] I suck the porno [raw token, left as printed, not resolved to a label] nerve gas of boy-roid [boy-roid: kept unresolved, the "roid" half stays as the raw token]
ops    | <<   ->   <<,/,/
names  | (none given)
62     | I suck the porno nerve gas of boy-roid
```

* `operator_fidelity`: added 2 (/,/) -- verbatim, in order

### clause 10 (p. 4, `plain_control`) -- `plain_english`: **pass**

```text
src    | <<I suck the porno nerve gas of boy-roid
gloss  | Here the seam is a block-quote opener — a naming frame that carries the book's recurring labels rather than passing any signal — so what the clause describes is a speech act: I draw in the "porno nerve gas of boy-roid," a phrase with no plain-English sense of its own, "boy-roid" in particular left as the raw coined token it is.
ops    | <<   ->   (none)
names  | (none given)
62     | I suck the porno nerve gas of boy-roid
```

* `frame_preservation`: not applicable (<< >> quoting frame verbalised by instruction) -- clause has 1<< 0>>, gloss has 0<< 0>>

### clause 10 (p. 4, `plain_control`) -- `register_paraphrase`: **pass**

```text
src    | <<I suck the porno nerve gas of boy-roid
gloss  | << I draw in the porno nerve gas of boy-roid
ops    | <<   ->   <<
names  | (none given)
62     | I suck the porno nerve gas of boy-roid
```

### clause 10 (p. 4, `plain_control`) -- `scene_summary`: **pass**

```text
src    | <<I suck the porno nerve gas of boy-roid
gloss  | The speaker sucks the porno nerve gas of boy-roid.
ops    | <<   ->   (none)
names  | (none given)
62     | I suck the porno nerve gas of boy-roid
```

* `frame_preservation`: not applicable (<< >> quoting frame dropped by instruction) -- clause has 1<< 0>>, gloss has 0<< 0>>

### clause 101 (p. 6, `quoting_frame`) -- `entity_consistent`: **pass**

```text
src    | <<the meridian>>To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
gloss  | <<the meridian>>To the desire that the body of ND — ND as channel firing orgones to the dog of silence, ND as mass — was congested: ND as form is jointed to the boundless suicide line, ND as cable, of the artificial-sun, and ND is done to by transplants and by sleep.
ops    | <<,>>   ->   <<,>>
names  | ND
62     | the meridian To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
```

### clause 101 (p. 6, `quoting_frame`) -- `literal_operator`: **pass**

```text
src    | <<the meridian>>To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
gloss  | <<the meridian>>[ND held as a label inside a quoting frame: the frame opens and closes around it, naming rather than acting] To the desire[the wanting, addressed] that the body[ND as mass-form: the persona's body] of self[ND as cable: the self's line] that fires orgones[ND as mass: spending its charge outward] to the dog of silence[ND as cable: the held-silent receiver] was congested[blocked, unable to pass] : It[ND] is jointed[joined along] to the boundless suicide line[ND as channel: the line with no limit] of the artificial-sun[ND as form: the made sun] and self[ND as cable] transplants[ND as form: carried across] sleep[ND as channel: put to sleep by]
ops    | <<,>>   ->   <<,>>
names  | ND
62     | the meridian To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
```

### clause 101 (p. 6, `quoting_frame`) -- `plain_english`: **pass**

```text
src    | <<the meridian>>To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
gloss  | A quoting seam opens and closes around the recurring label "the meridian," carrying names and labels only and passing no input or output, and within it: to the desire that the body of self, ND, that fires orgones to the dog of silence was congested, ND is jointed to the boundless suicide line of the artificial-sun, and ND transplants sleep (both transplants and sleep are things done to ND) — with "orgones," "dog of silence," "boundless suicide line," and "artificial-sun" kept as printed, since none of them carries a plain-English sense here.
ops    | <<,>>   ->   (none)
names  | ND
62     | the meridian To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
```

* `frame_preservation`: not applicable (<< >> quoting frame verbalised by instruction) -- clause has 1<< 1>>, gloss has 0<< 0>>

### clause 101 (p. 6, `quoting_frame`) -- `register_paraphrase`: **pass**

```text
src    | <<the meridian>>To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
gloss  | <<the meridian>>ND desires that the body of self that fires orgones to the dog of silence was congested, and ND is jointed to the boundless suicide line of the artificial-sun, and self transplants sleep
ops    | <<,>>   ->   <<,>>
names  | ND
62     | the meridian To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
```

### clause 101 (p. 6, `quoting_frame`) -- `scene_summary`: **pass**

```text
src    | <<the meridian>>To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
gloss  | ND's desire to fire orgones at the dog of silence was congested, jointed to the boundless suicide line of the artificial-sun, and self transplants sleep.
ops    | <<,>>   ->   (none)
names  | ND
62     | the meridian To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
```

* `frame_preservation`: not applicable (<< >> quoting frame dropped by instruction) -- clause has 1<< 1>>, gloss has 0<< 0>>

### clause 1003 (p. 37, `operator_dense`) -- `entity_consistent`: **fail**

```text
src    | Put out BABEL=TV> excoriate a vital=serum noise>
gloss  | Put out, the naming seam = fixes persona BABEL to machine-system BABEL=TV, its persona face TV the screened one, and the inline closer > shuts the mark; excoriate a vital=serum noise>, where the second seam = binds the unresolved vital to the unresolved serum and the closing > again cuts the line.
ops    | =,>,=,>   ->   =,=,>,=,>,=,>
names  | BABEL,BABEL=TV,TV
62     | Put out BABEL TV excoriate a vital serum noise
```

* `operator_fidelity`: added 3 (=,=,>) -- verbatim, in order

### clause 1003 (p. 37, `operator_dense`) -- `literal_operator`: **fail**

```text
src    | Put out BABEL=TV> excoriate a vital=serum noise>
gloss  | Put out BABEL[persona: animals(6)|channel(2)|deranged(2)|fecundates(2); acted on by is(5)|channel(3)|line(3)|sun(3)]=[naming seam, glued both sides]TV[persona: screen(6)|cursed(1)|is(1)|received(1); acted on by attack(1)|breaks(1)|down(1)|game(1)][> inline marker closer] excoriate a vital=serum[= naming seam, glued both sides]noise>
ops    | =,>,=,>   ->   |,|,|,|,|,|,=,|,|,|,|,|,|,>,=,=,>
names  | BABEL,BABEL=TV,TV
62     | Put out BABEL TV excoriate a vital serum noise
```

* `operator_fidelity`: added 13 (=,|,|,|,|,|,|,|,|,|,|,|,|) -- verbatim, in order

### clause 1003 (p. 37, `operator_dense`) -- `plain_english`: **fail**

```text
src    | Put out BABEL=TV> excoriate a vital=serum noise>
gloss  | Put out BABEL, the persona who makes animals, serves as a channel and is done to by lines and the sun, joined at a naming seam to TV, the persona who screens and is cursed, and this coupling is what fecundates and is fecundated, loaded and broken, and only then the vital serum takes over as its noise: put out the noise that is the serum of that life.
ops    | =,>,=,>   ->   (none)
names  | BABEL,BABEL=TV,TV
62     | Put out BABEL TV excoriate a vital serum noise
```

* `entity_fidelity`: dropped BABEL=TV -- #30 names as given

### clause 1003 (p. 37, `operator_dense`) -- `register_paraphrase`: **pass**

```text
src    | Put out BABEL=TV> excoriate a vital=serum noise>
gloss  | Put out BABEL=TV> excoriate a vital=serum noise>
ops    | =,>,=,>   ->   =,>,=,>
names  | BABEL,BABEL=TV,TV
62     | Put out BABEL TV excoriate a vital serum noise
```

### clause 1003 (p. 37, `operator_dense`) -- `scene_summary`: **fail**

```text
src    | Put out BABEL=TV> excoriate a vital=serum noise>
gloss  | The screen-channeled TV-persona BABEL is put out, excoriated as broken and discharged.
ops    | =,>,=,>   ->   (none)
names  | BABEL,BABEL=TV,TV
62     | Put out BABEL TV excoriate a vital serum noise
```

* `entity_fidelity`: dropped BABEL=TV -- #30 names as given

### clause 1005 (p. 37, `entity_named`) -- `entity_consistent`: **pass**

```text
src    | while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
gloss  | while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
ops    | =,=   ->   =,=
names  | OMOTYA
62     | while devouring body-OMOTYA of control deficiency to bio less intermediate area
```

### clause 1005 (p. 37, `entity_named`) -- `literal_operator`: **pass**

```text
src    | while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
gloss  | while devouring body-OMOTYA [OMOTYA embodied, the persona's own flesh taken as a body to be eaten] of control deficiency to bio=less= [one glued term: bio names life, less names absence, so the two parts are fastened into a single word meaning life-without] intermediate area
ops    | =,=   ->   =,=
names  | OMOTYA
62     | while devouring body-OMOTYA of control deficiency to bio less intermediate area
```

### clause 1005 (p. 37, `entity_named`) -- `plain_english`: **pass**

```text
src    | while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
gloss  | While OMOTYA, the persona crucified by one channel and acted on by that same channel, is being devoured — a naming seam joining body and its absent form marks off, and also names, an intermediate area of control deficiency set to the "less than bio" measure.
ops    | =,=   ->   (none)
names  | OMOTYA
62     | while devouring body-OMOTYA of control deficiency to bio less intermediate area
```

### clause 1005 (p. 37, `entity_named`) -- `register_paraphrase`: **pass**

```text
src    | while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
gloss  | while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
ops    | =,=   ->   =,=
names  | OMOTYA
62     | while devouring body-OMOTYA of control deficiency to bio less intermediate area
```

### clause 1005 (p. 37, `entity_named`) -- `scene_summary`: **fail**

```text
src    | while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
gloss  | OMOTYA is crucified while devouring the body of control deficiency, in a bio=less intermediate area.
ops    | =,=   ->   =
names  | OMOTYA
62     | while devouring body-OMOTYA of control deficiency to bio less intermediate area
```

* `operator_fidelity`: kept 1 operator character(s) it was told to verbalise: = -- dropped by instruction

### clause 2789 (p. 108, `residue`) -- `entity_consistent`: **fail**

```text
src    | l surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE
gloss  | I surrender to the chromosome syntax that fused the soul/gram (the / gluing gram to soul as compound, not the sentence-final separator office) of self in her omanko [unresolved] hydromachine of ToKAGE, and ToKAGE is the one persona this clause names: the persona who is and was, who fecundates and murders, and who is done to by love, by speed, by like, by crucifixion; no other persona is named here, and the surrenders-to is not given a name of its own, while omanko stays unresolved as printed.
ops    | /   ->   /,/
names  | ND,ToK,ToKAGE
62     | l surrender to the chromosome syntax that fused the soul gram of self in her ‹omanko› hydromachine of ToKAGE
```

* `operator_fidelity`: added 1 (/) -- verbatim, in order

### clause 2789 (p. 108, `residue`) -- `literal_operator`: **fail**

```text
src    | l surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE
gloss  | I surrender [l] / to the chromosome syntax that fused the soul/gram [soul/gram: compound linker fusing soul and gram] of self in her omanko [omanko: unresolved] hydromachine of ToKAGE
ops    | /   ->   /,/,/
names  | ND,ToK,ToKAGE
62     | l surrender to the chromosome syntax that fused the soul gram of self in her ‹omanko› hydromachine of ToKAGE
```

* `operator_fidelity`: added 2 (/,/) -- verbatim, in order

### clause 2789 (p. 108, `residue`) -- `plain_english`: **pass**

```text
src    | l surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE
gloss  | l surrender, that is I give myself up to the chromosome syntax that fused, at a naming seam where soul is bound compound-wise to gram into one coined word, the self inside her omanko — a term with no plain-English sense I can supply — hydromachine of ToKAGE.
ops    | /   ->   (none)
names  | ND,ToK,ToKAGE
62     | l surrender to the chromosome syntax that fused the soul gram of self in her ‹omanko› hydromachine of ToKAGE
```

### clause 2789 (p. 108, `residue`) -- `register_paraphrase`: **pass**

```text
src    | l surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE
gloss  | I surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE
ops    | /   ->   /
names  | ND,ToK,ToKAGE
62     | l surrender to the chromosome syntax that fused the soul gram of self in her ‹omanko› hydromachine of ToKAGE
```

### clause 2789 (p. 108, `residue`) -- `scene_summary`: **fail**

```text
src    | l surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE
gloss  | I surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE.
ops    | /   ->   /
names  | ND,ToK,ToKAGE
62     | l surrender to the chromosome syntax that fused the soul gram of self in her ‹omanko› hydromachine of ToKAGE
```

* `operator_fidelity`: kept 1 operator character(s) it was told to verbalise: / -- dropped by instruction

## Agreement across strategies

Jaccard over content words (operators excluded), averaged over the
pairs of strategies per clause (10 pairs for 5 strategies).
Reported for unanimous clauses too: an agreement figure that is only
printed when it is low is one nobody can rely on.

| clause | mean | min | max | least agreeing pair |
|---|---|---|---|---|
| 10 | 0.245 | 0.132 | 0.625 | `plain_english|scene_summary` |
| 101 | 0.433 | 0.309 | 0.652 | `literal_operator|scene_summary` |
| 1003 | 0.229 | 0.147 | 0.308 | `entity_consistent|scene_summary` |
| 1005 | 0.481 | 0.225 | 1.000 | `literal_operator|plain_english` |
| 2789 | 0.434 | 0.207 | 1.000 | `entity_consistent|plain_english` |

Low agreement is not noise to be averaged away -- it localises the
clauses where the strategies read the book differently, which is
what #64's committee should be pointed at.

## Anti-pattern audit (L9.8)

* **Circularity**: the model writes, `check_gloss` judges.  No
  strategy grades another, and no check calls the backend, so a
  gloss cannot be graded by the model that wrote it.
* **Hallucinated coherence**: no prompt asserts a plot; the
  `entity_fidelity` check fires on any capitalised token the clause
  and glossary do not license, and `residue_discipline` fires on a
  gloss that turns a marked-unresolved surface into an ordinary
  word.
* **Single-method certainty**: five strategies, agreement published
  per clause, and every row carries its strategy, its backend and
  the prompt hash it came from.
* **Over-correction**: #19 `ocr_error` surfaces are listed to the
  model as artefacts, so it is told to reproduce rather than
  normalise them.
* **Residue-dropping**: the sample includes the residue stratum
  first, so the check cannot be vacuous.
* **Determinism**: a model is not deterministic, so the artifacts
  are pinned.  `--live` re-runs and prints a diff against the pins
  rather than rewriting them, which is what #58 does for its source
  probes; a number that moves with the sampler cannot be
  regression-tested.

## Reproducing

```bash
python work/glossing_pipeline.py            # pinned, offline
python work/glossing_pipeline.py --live     # re-run, diff vs pins
python work/test_glossing_pipeline.py
```

Every prompt is written to `prompts.tsv` with its sha256, and each
pin stores the hash of the prompt it was recorded against, so a
change in the constraint assembly fails the build loudly instead of
silently reusing a stale response.

