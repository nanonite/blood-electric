# Multi-agent translation committee -- disagreement map (#64)

Generated: 2026-10-12 15:40 UTC

## What this is

Five agents -- #63's prompt strategies -- each produced one reading
of each of the 5 sampled clauses (25 readings).  The committee ran
the generator / critic / comparer loop over them: it replayed the
pinned generation, re-judged every reading with the four
deterministic checks recomputed from the frozen artifacts, and
compared every pair of readings surface by surface.  This map is
the comparer's output: every localised disagreement, classified.

**The load-bearing output is the classification, not the raw
divergence count.**  Two agents can differ because one of them
broke a rule, because they were told to differ, or because the
passage itself admits more than one compliant reading.  Only the
third kind is ambiguity.

## The four kinds of disagreement

| class | rows | what it means |
|---|---|---|
| `rule_violation` | 60 | the rule that governs this surface is breached by
  exactly one reading's own treatment of it, or by both: the
  divergence involves a defective reading, and the responsible
  agent is named |
| `contract_divergence` | 28 | both readings treated the surface as their own
  instructions required, and the difference is exactly what the
  differing instructions require (a strategy told to keep the
  operator channel versus one told to drop it): divergence by
  instruction, not by text |
| `genuine_ambiguity` | 19 | both readings pass every check that applies to them,
  and no instruction explains the difference: the passage admits
  more than one compliant reading. **This is the signal the task
  asks the disagreement to localise.** |
| `defect_unattributed` | 29 | a participant failed some check, but no check governs
  this surface, so the divergence is not usable as evidence about
  the passage.  Recorded rather than dropped or quietly promoted:
  it is a real measured divergence the rule refuses to call
  ambiguity |

## Headline

**4 of the 5 sampled passages** contain at least one surface on
which two fully compliant readings disagree -- the passages that are
genuinely ambiguous by the corpus's own rules:

* clause **10** -- 3 genuine surface(s): content (3)
* clause **101** -- 10 genuine surface(s): content (10)
* clause **1005** -- 5 genuine surface(s): content (5)
* clause **2789** -- 1 genuine surface(s): content (1)

Clauses whose every divergence is a defect or a contract:

* clause **1003** -- 41 localised disagreement(s), none of them genuine

## The map

Every genuine-ambiguity surface in the sample, one row each.
States are quoted at 12 words per side here; the complete states
are in `disagreement_log.tsv`.

| clause | kind | surface | `agent_a` state | `agent_b` state | governing check |
|---|---|---|---|---|---|
| 10 | content | `content words` | register_paraphrase: (none) | plain_english: act,block,book,carries,clause,coined,describes,english,frame,is,labels,left, ... (+18 more) | none |
| 10 | content | `content words` | register_paraphrase: draw | scene_summary: speaker,sucks | none |
| 10 | content | `content words` | scene_summary: speaker,sucks | plain_english: act,block,book,carries,clause,coined,describes,draw,english,frame,is,labels, ... (+19 more) | none |
| 101 | content | `content words` | literal_operator: across,acting,addressed,along,around,blocked,carried,charge,closes,fires,frame,held, ... (+20 more) | entity_consistent: done,firing | none |
| 101 | content | `content words` | literal_operator: across,acting,addressed,along,blocked,cable,carried,channel,charge,form,frame,held, ... (+17 more) | plain_english: are,carries,carrying,done,english,input,kept,labels,names,none,only,output, ... (+9 more) | none |
| 101 | content | `content words` | literal_operator: across,acting,addressed,along,around,blocked,cable,carried,channel,charge,closes,desire, ... (+23 more) | register_paraphrase: desires | none |
| 101 | content | `content words` | literal_operator: across,acting,addressed,along,around,blocked,body,cable,carried,channel,charge,closes, ... (+25 more) | scene_summary: fire | none |
| 101 | content | `content words` | plain_english: are,around,carries,carrying,closes,english,fires,input,kept,label,labels,names, ... (+15 more) | entity_consistent: cable,channel,firing,form,mass | none |
| 101 | content | `content words` | register_paraphrase: desires,fires,self | entity_consistent: cable,channel,desire,done,firing,form,mass | none |
| 101 | content | `content words` | register_paraphrase: desires | plain_english: are,around,carries,carrying,closes,desire,done,english,input,kept,label,labels, ... (+15 more) | none |
| 101 | content | `content words` | register_paraphrase: body,desires,fires,is,meridian | scene_summary: desire,fire,s | none |
| 101 | content | `content words` | scene_summary: fire,s,self | entity_consistent: body,cable,channel,done,firing,form,is,mass,meridian | none |
| 101 | content | `content words` | scene_summary: fire,s | plain_english: are,around,body,carries,carrying,closes,done,english,fires,input,is,kept, ... (+18 more) | none |
| 1005 | content | `content words` | literal_operator: absence,are,be,eaten,embodied,fastened,flesh,glued,into,life,meaning,names, ... (+10 more) | entity_consistent: (none) | none |
| 1005 | content | `content words` | literal_operator: absence,are,be,devouring,eaten,embodied,fastened,flesh,glued,into,life,meaning, ... (+9 more) | plain_english: absent,acted,also,being,channel,crucified,devoured,form,is,joining,marks,measure, ... (+5 more) | none |
| 1005 | content | `content words` | literal_operator: absence,are,be,eaten,embodied,fastened,flesh,glued,into,life,meaning,names, ... (+10 more) | register_paraphrase: (none) | none |
| 1005 | content | `content words` | plain_english: absent,acted,also,being,channel,crucified,devoured,form,is,joining,marks,measure, ... (+7 more) | entity_consistent: devouring | none |
| 1005 | content | `content words` | register_paraphrase: devouring | plain_english: absent,acted,also,being,channel,crucified,devoured,form,is,joining,marks,measure, ... (+7 more) | none |
| 2789 | content | `content words` | register_paraphrase: (none) | plain_english: bound,can,coined,compound,english,give,inside,into,is,l,myself,naming, ... (+9 more) | none |

## The passages, in full

### clause 10 (p. 4, `plain_control`)

```text
src | <<I suck the porno nerve gas of boy-roid
```

* **`literal_operator`** -- fail -- failed: operator_fidelity
  > << [quoting frame: opens the span that carries the book's recurring names/labels; a frame, not an I/O op] I suck the porno [raw token, left as printed, not resolved to a label] nerve gas of boy-roid [boy-roid: kept unresolved, the "roid" half stays as the raw token]
* **`register_paraphrase`** -- pass -- failed: (none)
  > << I draw in the porno nerve gas of boy-roid
* **`scene_summary`** -- pass -- failed: (none)
  > The speaker sucks the porno nerve gas of boy-roid.
* **`plain_english`** -- pass -- failed: (none)
  > Here the seam is a block-quote opener — a naming frame that carries the book's recurring labels rather than passing any signal — so what the clause describes is a speech act: I draw in the "porno nerve gas of boy-roid," a phrase with no plain-English sense of its own, "boy-roid" in particular left as the raw coined token it is.
* **`entity_consistent`** -- fail -- failed: operator_fidelity
  > << a quoting frame bearing the book's recurring names/labels, holding open the label-line: I suck the porno nerve gas of boy-roid — the speaker draws breath in on the sucking, and what is taken in is nerve gas named porno, drawn out of boy-roid; who or what boy-roid is stays unresolved, as does porno

Localised disagreements: 29 -- rule_violation 13, contract_divergence 6, genuine_ambiguity 3, defect_unattributed 7.

Genuine ambiguity (3 surface(s)):

* `content` `content words` -- `register_paraphrase`: (none) ; `plain_english`: act,block,book,carries,clause,coined,describes,english,frame,is,labels,left,naming,opener,own,particular,passing,phrase,plain,quote,rather,raw,recurring,s,seam,sense,signal,speech,token,what -- governing check: none
* `content` `content words` -- `register_paraphrase`: draw ; `scene_summary`: speaker,sucks -- governing check: none
* `content` `content words` -- `scene_summary`: speaker,sucks ; `plain_english`: act,block,book,carries,clause,coined,describes,draw,english,frame,is,labels,left,naming,opener,own,particular,passing,phrase,plain,quote,rather,raw,recurring,s,seam,sense,signal,speech,token,what -- governing check: none

### clause 101 (p. 6, `quoting_frame`)

```text
src | <<the meridian>>To the desire that the body of self that fires orgones to the dog of silence was congested: It is jointed to the boundless suicide line of the artificial-sun and self transplants sleep
```

* **`literal_operator`** -- pass -- failed: (none)
  > <<the meridian>>[ND held as a label inside a quoting frame: the frame opens and closes around it, naming rather than acting] To the desire[the wanting, addressed] that the body[ND as mass-form: the persona's body] of self[ND as cable: the self's line] that fires orgones[ND as mass: spending its charge outward] to the dog of silence[ND as cable: the held-silent receiver] was congested[blocked, unable to pass] : It[ND] is jointed[joined along] to the boundless suicide line[ND as channel: the line with no limit] of the artificial-sun[ND as form: the made sun] and self[ND as cable] transplants[ND as form: carried across] sleep[ND as channel: put to sleep by]
* **`register_paraphrase`** -- pass -- failed: (none)
  > <<the meridian>>ND desires that the body of self that fires orgones to the dog of silence was congested, and ND is jointed to the boundless suicide line of the artificial-sun, and self transplants sleep
* **`scene_summary`** -- pass -- failed: (none)
  > ND's desire to fire orgones at the dog of silence was congested, jointed to the boundless suicide line of the artificial-sun, and self transplants sleep.
* **`plain_english`** -- pass -- failed: (none)
  > A quoting seam opens and closes around the recurring label "the meridian," carrying names and labels only and passing no input or output, and within it: to the desire that the body of self, ND, that fires orgones to the dog of silence was congested, ND is jointed to the boundless suicide line of the artificial-sun, and ND transplants sleep (both transplants and sleep are things done to ND) — with "orgones," "dog of silence," "boundless suicide line," and "artificial-sun" kept as printed, since none of them carries a plain-English sense here.
* **`entity_consistent`** -- pass -- failed: (none)
  > <<the meridian>>To the desire that the body of ND — ND as channel firing orgones to the dog of silence, ND as mass — was congested: ND as form is jointed to the boundless suicide line, ND as cable, of the artificial-sun, and ND is done to by transplants and by sleep.

Localised disagreements: 22 -- contract_divergence 12, genuine_ambiguity 10.

Genuine ambiguity (10 surface(s)):

* `content` `content words` -- `literal_operator`: across,acting,addressed,along,around,blocked,carried,charge,closes,fires,frame,held,inside,joined,label,limit,made,naming,opens,outward,pass,persona,put,quoting,rather,receiver,s,self,silent,spending,unable,wanting ; `entity_consistent`: done,firing -- governing check: none
* `content` `content words` -- `literal_operator`: across,acting,addressed,along,blocked,cable,carried,channel,charge,form,frame,held,inside,joined,limit,made,mass,naming,outward,pass,persona,put,rather,receiver,s,silent,spending,unable,wanting ; `plain_english`: are,carries,carrying,done,english,input,kept,labels,names,none,only,output,passing,plain,printed,recurring,seam,sense,since,things,within -- governing check: none
* `content` `content words` -- `literal_operator`: across,acting,addressed,along,around,blocked,cable,carried,channel,charge,closes,desire,form,frame,held,inside,joined,label,limit,made,mass,naming,opens,outward,pass,persona,put,quoting,rather,receiver,s,silent,spending,unable,wanting ; `register_paraphrase`: desires -- governing check: none
* `content` `content words` -- `literal_operator`: across,acting,addressed,along,around,blocked,body,cable,carried,channel,charge,closes,fires,form,frame,held,inside,is,joined,label,limit,made,mass,meridian,naming,opens,outward,pass,persona,put,quoting,rather,receiver,silent,spending,unable,wanting ; `scene_summary`: fire -- governing check: none
* `content` `content words` -- `plain_english`: are,around,carries,carrying,closes,english,fires,input,kept,label,labels,names,none,only,opens,output,passing,plain,printed,quoting,recurring,seam,self,sense,since,things,within ; `entity_consistent`: cable,channel,firing,form,mass -- governing check: none
* `content` `content words` -- `register_paraphrase`: desires,fires,self ; `entity_consistent`: cable,channel,desire,done,firing,form,mass -- governing check: none
* `content` `content words` -- `register_paraphrase`: desires ; `plain_english`: are,around,carries,carrying,closes,desire,done,english,input,kept,label,labels,names,none,only,opens,output,passing,plain,printed,quoting,recurring,seam,sense,since,things,within -- governing check: none
* `content` `content words` -- `register_paraphrase`: body,desires,fires,is,meridian ; `scene_summary`: desire,fire,s -- governing check: none
* `content` `content words` -- `scene_summary`: fire,s,self ; `entity_consistent`: body,cable,channel,done,firing,form,is,mass,meridian -- governing check: none
* `content` `content words` -- `scene_summary`: fire,s ; `plain_english`: are,around,body,carries,carrying,closes,done,english,fires,input,is,kept,label,labels,meridian,names,none,only,opens,output,passing,plain,printed,quoting,recurring,seam,sense,since,things,within -- governing check: none

### clause 1003 (p. 37, `operator_dense`)

```text
src | Put out BABEL=TV> excoriate a vital=serum noise>
```

* **`literal_operator`** -- fail -- failed: operator_fidelity
  > Put out BABEL[persona: animals(6)|channel(2)|deranged(2)|fecundates(2); acted on by is(5)|channel(3)|line(3)|sun(3)]=[naming seam, glued both sides]TV[persona: screen(6)|cursed(1)|is(1)|received(1); acted on by attack(1)|breaks(1)|down(1)|game(1)][> inline marker closer] excoriate a vital=serum[= naming seam, glued both sides]noise>
* **`register_paraphrase`** -- pass -- failed: (none)
  > Put out BABEL=TV> excoriate a vital=serum noise>
* **`scene_summary`** -- fail -- failed: entity_fidelity
  > The screen-channeled TV-persona BABEL is put out, excoriated as broken and discharged.
* **`plain_english`** -- fail -- failed: entity_fidelity
  > Put out BABEL, the persona who makes animals, serves as a channel and is done to by lines and the sun, joined at a naming seam to TV, the persona who screens and is cursed, and this coupling is what fecundates and is fecundated, loaded and broken, and only then the vital serum takes over as its noise: put out the noise that is the serum of that life.
* **`entity_consistent`** -- fail -- failed: operator_fidelity
  > Put out, the naming seam = fixes persona BABEL to machine-system BABEL=TV, its persona face TV the screened one, and the inline closer > shuts the mark; excoriate a vital=serum noise>, where the second seam = binds the unresolved vital to the unresolved serum and the closing > again cuts the line.

Localised disagreements: 41 -- rule_violation 25, contract_divergence 6, defect_unattributed 10.

### clause 1005 (p. 37, `entity_named`)

```text
src | while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
```

* **`literal_operator`** -- pass -- failed: (none)
  > while devouring body-OMOTYA [OMOTYA embodied, the persona's own flesh taken as a body to be eaten] of control deficiency to bio=less= [one glued term: bio names life, less names absence, so the two parts are fastened into a single word meaning life-without] intermediate area
* **`register_paraphrase`** -- pass -- failed: (none)
  > while devouring body-OMOTYA of control deficiency to bio=less= intermediate area
* **`scene_summary`** -- fail -- failed: operator_fidelity
  > OMOTYA is crucified while devouring the body of control deficiency, in a bio=less intermediate area.
* **`plain_english`** -- pass -- failed: (none)
  > While OMOTYA, the persona crucified by one channel and acted on by that same channel, is being devoured — a naming seam joining body and its absent form marks off, and also names, an intermediate area of control deficiency set to the "less than bio" measure.
* **`entity_consistent`** -- pass -- failed: (none)
  > while devouring body-OMOTYA of control deficiency to bio=less= intermediate area

Localised disagreements: 20 -- rule_violation 8, contract_divergence 3, genuine_ambiguity 5, defect_unattributed 4.

Genuine ambiguity (5 surface(s)):

* `content` `content words` -- `literal_operator`: absence,are,be,eaten,embodied,fastened,flesh,glued,into,life,meaning,names,own,parts,persona,s,single,taken,term,two,without,word ; `entity_consistent`: (none) -- governing check: none
* `content` `content words` -- `literal_operator`: absence,are,be,devouring,eaten,embodied,fastened,flesh,glued,into,life,meaning,own,parts,s,single,taken,term,two,without,word ; `plain_english`: absent,acted,also,being,channel,crucified,devoured,form,is,joining,marks,measure,naming,off,same,seam,set -- governing check: none
* `content` `content words` -- `literal_operator`: absence,are,be,eaten,embodied,fastened,flesh,glued,into,life,meaning,names,own,parts,persona,s,single,taken,term,two,without,word ; `register_paraphrase`: (none) -- governing check: none
* `content` `content words` -- `plain_english`: absent,acted,also,being,channel,crucified,devoured,form,is,joining,marks,measure,names,naming,off,persona,same,seam,set ; `entity_consistent`: devouring -- governing check: none
* `content` `content words` -- `register_paraphrase`: devouring ; `plain_english`: absent,acted,also,being,channel,crucified,devoured,form,is,joining,marks,measure,names,naming,off,persona,same,seam,set -- governing check: none

### clause 2789 (p. 108, `residue`)

```text
src | l surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE
```

* **`literal_operator`** -- fail -- failed: operator_fidelity
  > I surrender [l] / to the chromosome syntax that fused the soul/gram [soul/gram: compound linker fusing soul and gram] of self in her omanko [omanko: unresolved] hydromachine of ToKAGE
* **`register_paraphrase`** -- pass -- failed: (none)
  > I surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE
* **`scene_summary`** -- fail -- failed: operator_fidelity
  > I surrender to the chromosome syntax that fused the soul/gram of self in her omanko hydromachine of ToKAGE.
* **`plain_english`** -- pass -- failed: (none)
  > l surrender, that is I give myself up to the chromosome syntax that fused, at a naming seam where soul is bound compound-wise to gram into one coined word, the self inside her omanko — a term with no plain-English sense I can supply — hydromachine of ToKAGE.
* **`entity_consistent`** -- fail -- failed: operator_fidelity
  > I surrender to the chromosome syntax that fused the soul/gram (the / gluing gram to soul as compound, not the sentence-final separator office) of self in her omanko [unresolved] hydromachine of ToKAGE, and ToKAGE is the one persona this clause names: the persona who is and was, who fecundates and murders, and who is done to by love, by speed, by like, by crucifixion; no other persona is named here, and the surrenders-to is not given a name of its own, while omanko stays unresolved as printed.

Localised disagreements: 24 -- rule_violation 14, contract_divergence 1, genuine_ambiguity 1, defect_unattributed 8.

Genuine ambiguity (1 surface(s)):

* `content` `content words` -- `register_paraphrase`: (none) ; `plain_english`: bound,can,coined,compound,english,give,inside,into,is,l,myself,naming,plain,seam,sense,supply,term,up,where,wise,word -- governing check: none

## What the disagreement is not

The other three classes, with the corpus's own examples:

* `rule_violation` (60 row(s)) -- a defective reading.  Example: clause 1003's `scene_summary` drops `BABEL=TV` and fails `entity_fidelity`, so its disagreement with `register_paraphrase` on that entity is apparent ambiguity, not a property of the passage.
* `contract_divergence` (28 row(s)) -- instructed divergence.  Example: clause 101's `<< >>` is kept by the three `verbatim, in order` strategies and dropped by the two drop-strategies, all five compliant -- the channel differs because the prompts differ.
* `defect_unattributed` (29 row(s)) -- a defective reading diverging on a surface no failed check governs.  Example: clause 1003's `literal_operator` fails `operator_fidelity` (its bracketed persona constraints add thirteen `|`), and its content words diverge from the passing `register_paraphrase` for reasons no check on that surface explains -- the divergence is real but it is not evidence about the text.

## Method

Three phases over each sampled clause, no model called:

1. **Generator** -- #63's pinned replay (`resolve_glosses` over the
   same stratified sample), cross-checked against the published
   `glosses.tsv` (gloss text, verdict, operator columns and the
   response sha256 all re-derived).
2. **Critic** -- the four deterministic checks recomputed here from
   the frozen artifacts, then checked against #63's published
   `checks.tsv`: same functions, same frozen inputs, so a
   disagreement between the critic and the published record is a
   build error, not a finding.
3. **Comparer** -- every pair of a clause's readings, compared on
   each operator symbol (kept/dropped/added, counted), each #30
   entity (carried/dropped), each #62 residue surface
   (kept/smoothed) and the content-word sets, plus a pair-level
   verdict row.

A row is `genuine_ambiguity` only if both readings pass every check
that applies to them *and* no instruction explains the difference
(operator surfaces on strategies with differing operator
obligations are `contract_divergence` instead).  That is what makes
the headline a measurement: the checks are the workspace's codified
standard of a compliant reading.

## Guardrails (L9.8 audit)

* **Circularity**: the critic grades a pinned replay and the frozen
  artifacts; it checks #63's verdicts as outputs rather than
  reading them as inputs.  The comparer never grades a reading --
  it records each agent's measured state on each surface, and the
  classifier never prefers one.
* **Hallucinated coherence**: the map reports states, not
  resolutions; no reading is edited, ranked or preferred, and a
  genuine-ambiguity row carries both readings' evidence.
* **Single-method certainty**: five agents, all ten pairs per
  clause published, every row classified by a published rule.
* **Determinism**: the build is byte-identical on rebuild apart from
  the generation stamp, for the same reason #63's is -- nothing
  moves with the sampler.

## Reproducing

```bash
python work/translation_committee.py      # pinned, offline
python work/test_translation_committee.py
```

Prerequisites are #63's frozen artifacts (`analysis/glossing/`) and,
through #63's loaders, #17/#33/#30/#62.  Inputs are read-only; the
committee writes only `analysis/committee/`.

