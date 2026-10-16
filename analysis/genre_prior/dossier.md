# Genre/prior dossier: what the priors expect, and what the corpus measures (#60)

Generated: 2026-10-16 16:43 UTC

L7.5 of `docs/meaning-extraction-methods.md` names the priors this dossier is built from -- cyberpunk, *Tetsuo: The Iron Man*, Ballard, Aase Berg, Blake Butler, text-virus/codework poetics -- and says what they are for: *they set expectations for the kind of meaning present*. L7's governing rule sits above that line: external material is a **prior, not the object of study**; it is used to disambiguate, never to ventriloquize.

This dossier states every expectation one by one, names the probe that would settle it, runs that probe on the book's own frozen layers, and annotates the verdict. No tradition text is ingested (ledger below); no prior quotation is used as evidence about the book.

**Verdicts: 18 present, 5 thin, 0 absent, 2 underdetermined, of 25 expectations.**

## 1. The prior ledger -- what the dossier refuses to ingest

Every prior work is in copyright. The cyberpunk tradition is already represented lawfully by #56's fair-use excerpt corpus (105 tokens, non-inferential by #56's own 10,000-token rule); the rest are priors only. #57's survey pinned the L7.5 priors as not-acquired on 2026-10-14, and this module inherits that status rather than re-probing.

| work | tradition | year | status here |
|---|---|---|---|
| Neuromancer | cyberpunk (Gibson / Stephenson / Sterling) | 1984 | excerpt corpus via #56 |
| Snow Crash | cyberpunk (Gibson / Stephenson / Sterling) | 1992 | excerpt corpus via #56 |
| Mirrorshades (ed.) | cyberpunk (Gibson / Stephenson / Sterling) | 1986 | excerpt corpus via #56 |
| Tetsuo: The Iron Man (film) | Tetsuo: The Iron Man (Tsukamoto 1989) | 1989 | prior only, not ingested |
| Crash | J. G. Ballard | 1973 | prior only, not ingested |
| The Atrocity Exhibition | J. G. Ballard | 1970 | prior only, not ingested |
| Forsla fett (Transfer Fat) | Aase Berg | 2002 | prior only, not ingested |
| There Is No Year | Blake Butler | 2011 | prior only, not ingested |
| 300,000,000 | Blake Butler | 2014 | prior only, not ingested |
| codework / net.wurk | text-virus / codework poetics | 1990s- | prior only, not ingested |

## 2. Method: the probes and the gates

* **Book side.** The #17 *word-part* layer, markup-screened exactly as #56 (79,770 kept of 80,363 parts, 3,683 types) -- the only layer comparable to ordinary English words.  The screen is #56's exactly, jacket pages included (p1-3 and the p224 synopsis): no page is excluded, so every count reconciles with #56 -- and a cited exemplar can be front matter.
* **Lexicon probes.** Each family is a set of word-part patterns declared before measurement (full match on the surface). Density is reported per 1,000 book parts against the same family's density in each #56 reference corpus, with a Poisson contrast against the general-English rate. The excerpt corpora are context only.
* **Co-occurrence probes.** A #33 clause *has* a family when any of its parts full-matches it; the observed count of clauses holding both families is read against random placement (`n_a x n_b / 5,614` clauses, Poisson z). "Fused" means more often than chance allows, not merely adjacent. The clause partition is #33's operator-delimited utterance, so every kept part lands in exactly one clause (79,770 assigned, 0 unassigned).
* **Artifact probes.** The number is read from the frozen artifact of the analysis that owns it; the pass rule is declared here before the number is read.
* **Gates (declared conventions, not fitted to results).** Lexicon: *absent* if never occurring, *present* at >= 100 occurrences and >= 15x its general-English density, else *thin*. Co-occurrence: *present* at >= 5 clauses and z >= 3, else *thin*; *absent* below 5. *Underdetermined* means no probe in this workspace can settle the expectation -- it is written down with the evidence that would, not dropped.

Why the ratio and not significance is the lexicon gate: every family here clears a Poisson z against generic prose by orders of magnitude -- the book is not generic prose -- so significance cannot discriminate. Distinctiveness can. Every number is published below so a reader can re-gate.

## 3. The expectations table

| id | tradition | expectation | probe | verdict |
|---|---|---|---|---|
| CP1 | cyberpunk (Gibson / Stephenson / Sterling) | corporate power is a live lexical register, not a name | `lexicon:corporate` | **thin** |
| CP2 | cyberpunk (Gibson / Stephenson / Sterling) | a dense digital/technological register carries the texture | `lexicon:digital` | **present** |
| CP3 | cyberpunk (Gibson / Stephenson / Sterling) | body and machine fuse inside single clauses, not across them | `cooccur:machine+organ` | **present** |
| CP4 | cyberpunk (Gibson / Stephenson / Sterling) | the dystopia is named in a proper-noun layer | `artifact:roles_named_world` | **present** |
| TS1 | Tetsuo: The Iron Man (Tsukamoto 1989) | metal is a live material register of the body | `lexicon:metal` | **thin** |
| TS2 | Tetsuo: The Iron Man (Tsukamoto 1989) | transformation is processual: mutation and proliferation verbs | `lexicon:transformation` | **present** |
| TS3 | Tetsuo: The Iron Man (Tsukamoto 1989) | fusion happens in contact: metal and organ in the same clause | `cooccur:metal+organ` | **thin** |
| TS4 | Tetsuo: The Iron Man (Tsukamoto 1989) | the horror is located in the body's interior | `lexicon:organ` | **present** |
| BL1 | J. G. Ballard | a clinical register is over-dense against ordinary prose | `lexicon:clinical` | **present** |
| BL2 | J. G. Ballard | the affect profile is the new psychopathology | `artifact:affect_margins` | **present** |
| BL3 | J. G. Ballard | violence is eroticized: sex and violence share clauses | `cooccur:sexual+violence` | **thin** |
| BL4 | J. G. Ballard | the pathology is organized as a case with testable claims | `artifact:roles_predicates` | **thin** |
| BG1 | Aase Berg | reproduction is a live and distinctive register | `lexicon:reproduction` | **present** |
| BG2 | Aase Berg | the grotesque is a named register, not an accident | `lexicon:grotesque` | **present** |
| BG3 | Aase Berg | birth and grotesque fuse in single clauses | `cooccur:reproduction+grotesque` | **present** |
| BG4 | Aase Berg | tenderness persists inside the grotesque | `declared` | **underdetermined** |
| BT1 | Blake Butler | syntax is deranged but not noise | `artifact:parse_derangement` | **present** |
| BT2 | Blake Butler | phrases recur densely enough to be formulas | `artifact:formula_density` | **present** |
| BT3 | Blake Butler | the book loops back on itself | `artifact:self_similarity` | **present** |
| BT4 | Blake Butler | narrative time is elliptical | `declared` | **underdetermined** |
| TV1 | text-virus / codework poetics | the surface is code-like: operators as a structural share | `artifact:operator_surface` | **present** |
| TV2 | text-virus / codework poetics | infection is a live and distinctive register | `lexicon:virus` | **present** |
| TV3 | text-virus / codework poetics | the virus is digital: infection and technology share clauses | `cooccur:virus+digital` | **present** |
| TV4 | text-virus / codework poetics | an asemic layer: invented surfaces outside the dictionary | `artifact:neologism_coinage` | **present** |
| TV5 | text-virus / codework poetics | a command/scream layer in capitals | `artifact:case_emphasis` | **present** |

## 4. The lexicon families, measured

Densities per 1,000 book parts; ratio against general English. Excerpt corpora (105/56/49 tokens) are context only and cannot carry an inference.

| family | serves | occ | types | book /1k | gen. English /1k | ratio | z | medical /1k | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| corporate | CP1 | 59 | 4 | 0.740 | 0.170 | x4.35 | 12.3 | 0.550 | **thin** |
| digital | CP2 | 1,507 | 33 | 18.892 | 0.311 | x60.71 | 297.5 | 6.891 | **present** |
| machine | CP3,TS3 | 1,688 | 22 | 21.161 | 0.264 | x80.16 | 363.3 | 2.594 | **present** |
| metal | TS1,TS3 | 114 | 7 | 1.429 | 0.330 | x4.32 | 17.1 | 0.221 | **thin** |
| organ | TS4,CP3 | 1,490 | 18 | 18.679 | 0.945 | x19.77 | 163.0 | 4.168 | **present** |
| clinical | BL1 | 560 | 13 | 7.020 | 0.155 | x45.21 | 155.6 | 12.353 | **present** |
| reproduction | BG1,BG3 | 2,616 | 28 | 32.794 | 0.607 | x53.99 | 368.8 | 1.484 | **present** |
| grotesque | BG2,BG3 | 951 | 22 | 11.922 | 0.248 | x48.1 | 209.4 | 0.087 | **present** |
| virus | TV2,TV3 | 650 | 15 | 8.148 | 0.058 | x139.56 | 299.0 | 9.867 | **present** |
| transformation | TS2 | 523 | 37 | 6.556 | 0.182 | x36.03 | 133.5 | 0.353 | **present** |
| sexual | BL3 | 778 | 24 | 9.753 | 0.993 | x9.82 | 78.5 | 0.426 | **thin** |
| violence | (context) | 1,759 | 39 | 22.051 | 0.554 | x39.8 | 257.9 | 0.021 | **present** |
| city | (context) | 284 | 2 | 3.560 | 0.665 | x5.36 | 31.7 | 1.013 | **thin** |

Top types per family: **corporate**: sato:29, corporation:27, syndicate:2, corp:1; **digital**: circuit:220, digital:197, channel:180, code:148, cyber:146, hologram:144; **machine**: machine:604, drone:271, mechanism:263, device:187, android:78, exoskeleta:59; **metal**: metal:93, copper:8, chrome:6, scrapped:3, steel:2, wire:1; **organ**: brain:511, flesh:409, organ:290, heart:87, entrails:69, tissue:41; **clinical**: serum:239, nerve:132, sutured:104, fracture:31, syndrome:26, suture:16; **reproduction**: boy:372, embryo:335, fecundates:293, gene:229, chromosome:224, genome:207; **grotesque**: cadaver:633, mutant:141, grotesque:99, mutants:21, decay:13, aberration:9; **virus**: disease:210, parasite:185, parasitic:79, virus:71, infected:42, infectious:24; **transformation**: sutured:104, streaming:54, mutated:37, excoriate:36, mutation:35, excoriates:31; **sexual**: desire:275, love:107, vamp:91, ecstasy:65, sperm:43, sexual:38; **violence**: murder:624, assassin:179, crucified:178, rape:148, murderous:131, torture:62; **city**: city:283, population:1

## 5. The fusion probes (clause-level co-occurrence)

| pair | clauses with both | expected | z | example | verdict |
|---|---:|---:|---:|---|---|
| machine+organ | 403 | 323.5 | 4.42 | p4:1018 | **present** |
| metal+organ | 39 | 26.6 | 2.41 | p5:711 | **thin** |
| sexual+violence | 231 | 190.4 | 2.95 | p4:100 | **thin** |
| reproduction+grotesque | 491 | 305.6 | 10.61 | p4:1634 | **present** |
| virus+digital | 186 | 133.9 | 4.5 | p16:587 | **present** |

## 6. The artifact probes

| key | owner | observed | rule | verdict |
|---|---|---|---|---|
| roles_named_world | #59 | the named dystopia exists as a proper-noun layer with text evidence | >=2 of Cadaver City / Placenta World / Sato Corporation evidenced against the text | **present** |
| roles_predicates | #59 + #66 | the case-history scaffolding is dense (4,829 actor->verb->target events) but the synopsis predicates are not merely incomplete: 1 role(s) refuted, 2 uncertain (#59) | events >= 1000 and refuted >= 1 -> thin; events >= 1000 and refuted == 0 -> present | **thin** |
| affect_margins | #44 | fear z=11.07; ecstasy z=22.63; paranoia z=6.96; disgust z=14.7 | all four z >= 3.0 against the #44 English baseline | **present** |
| parse_derangement | #32 | parseable 0.521 vs English 0.785 and shuffle null 0.417: deranged yet above noise | book < english - 0.10 and book > null + 0.05 | **present** |
| formula_density | #57 | 25,969 recurring phrase formulas (n>=2, freq>=2) | >= 20000 book formulas (n>=2, freq>=2) on the #17 word-part layer | **present** |
| self_similarity | #54 | band z: lag10 28.52, lag100 20.15, max 33.69 | z at lag 10 or 100 paragraphs >= 10 | **present** |
| operator_surface | #13 + #33 | 10,949 operator tokens over 77,392 words (14.1%); 5,773 clause-boundary operators | operator tokens / word tokens >= 0.10 | **present** |
| neologism_coinage | #19 | 74 true coinages over 338 occurrences; 2,216 compound types | coinage types >= 40 and occurrences >= 100 | **present** |
| case_emphasis | #22 | caps layer: scream 263, label 1138, function 453 (share of words 2.44%) | scream >= 100 and scream+label+function >= 500 | **present** |
| cyberpunk_arm | #56 + #57 | the cyberpunk arm is 105 tokens (< 10,000); it shares 13 book n-gram types (n>=2) and is a bound, not a test -- #56's own rule | (not an expectation) reported as context; inferential only at >= 10,000 tokens | **underdetermined** |

The machine-readable `observed` values are in `artifact_probes.tsv` and `summary.json`.

## 7. The annotated dossier

### cyberpunk (Gibson / Stephenson / Sterling)

**CP1 -- corporate power is a live lexical register, not a name** -- **THIN**

*Why the prior expects it.* The genre's founding triangle (Gibson's zaibatsu worlds, Stephenson's franchulate, Sterling's Movement) makes the firm the ambient institution, so corporate vocabulary should be ordinary texture, not a single proper noun.

*Probe `lexicon:corporate`.* corporate: 59 occurrences / 4 types, 0.74/1k vs general English 0.17/1k (x4.35, z 12.3); e.g. corporation p2:218

*Annotation.* The firm exists, but as a name and not as a register. `Sato Corporation` is #59-evidenced as an `organization` with napalm/torture/vampire clusters, yet the generic corporate lexicon stays below the distinctiveness gate while the digital register clears it by a wide margin. The prior is met at the proper-noun layer and missed at the lexical layer: this is a cyberpunk of bodies, not of boardrooms. The verdict licenses 'corporate power is named and violent', not 'the corporation is rendered as infrastructure'.

*What would falsify it.* corporate density at generic-prose level would strip the prediction: the corporation is named but not rendered.

**CP2 -- a dense digital/technological register carries the texture** -- **PRESENT**

*Why the prior expects it.* Cyberpunk's surface is made of interfaces, data flows and electronics; the register should be dense and it should be neologism-prone, since the genre coins its own hardware words.

*Probe `lexicon:digital`.* digital: 1507 occurrences / 33 types, 18.892/1k vs general English 0.311/1k (x60.71, z 297.5); e.g. cyberpunk p1:160

*Annotation.* The technological register is among the book's densest families and it is neologism-prone (`cyber*` matches coined surfaces such as cyberBuddha), which is exactly cyberpunk's coinage habit. This verdict licenses 'the texture is technological'; it does not identify the register with any particular cyberpunk text -- the excerpt arm is 105 tokens and cannot.

*What would falsify it.* a digital register no denser than prose would mean the technology is named, not inhabited.

**CP3 -- body and machine fuse inside single clauses, not across them** -- **PRESENT**

*Why the prior expects it.* The cyborg is the genre's body: interface as tissue, so machine and body vocabulary should share clauses more than chance allows.

*Probe `cooccur:machine+organ`.* machine+organ: 403 clauses hold both (expected 323.5 under random placement, z 4.42); e.g. clause p4:1018

*Annotation.* Machine and organ vocabulary share clauses far above random placement, so the fusion is staged inside the utterance, the codework/cyborg unit of this book. The annotation is about units: the clause is #33's operator-delimited utterance, so 'one clause' means one breath of the machine voice.

*What would falsify it.* machine and organ vocabulary in separate clauses would be adjacency, not fusion.

**CP4 -- the dystopia is named in a proper-noun layer** -- **PRESENT**

*Why the prior expects it.* Genre worlds are named (the Sprawl, Neo-Tokyo, the zaibatsu); the book's own jacket names Sato Corporation, Cadaver City and Placenta World, and #59 tested those names against the text.

*Probe `artifact:roles_named_world`.* the named dystopia exists as a proper-noun layer with text evidence

*Annotation.* #59 verified the named layer against the text: Cadaver City and Placenta World are `place`s, Sato Corporation an `organization`, EVOL the pink-ash planet the jacket names. The proper-noun world is real in the text, not only on the jacket. This is also the honest limit of the cyberpunk prior here: the named world carries the genre, while the lexical registers that would render it are thin.

*What would falsify it.* if the named place/organization entities were refuted against the text, the proper-noun layer would be marketing only.

### Tetsuo: The Iron Man (Tsukamoto 1989)

**TS1 -- metal is a live material register of the body** -- **THIN**

*Why the prior expects it.* Tsukamoto's Metal Fetishist assimilates iron and scrap into flesh; the prior expects metal words to be ordinary material vocabulary, not a rare image.

*Probe `lexicon:metal`.* metal: 114 occurrences / 7 types, 1.429/1k vs general English 0.33/1k (x4.32, z 17.1); e.g. copper p5:733

*Annotation.* Metal is present -- chrome, copper, steel, wire, scrap -- but it stays below the distinctiveness gate, an order of magnitude sparser than the reproductive register. The prior's metal-flesh fusion is not this book's dominant fusion image; what fuses here is organ with machine (CP3) and birth with grotesque (BG3). The verdict licenses 'metal is material', not 'the body assimilates metal'.

*What would falsify it.* metal at generic-prose density would mean the fusion is organic-only: the machine-horror is metaphoric.

**TS2 -- transformation is processual: mutation and proliferation verbs** -- **PRESENT**

*Why the prior expects it.* The film's horror is *becoming* -- liquefaction, fusion, excrescence -- so the process verbs should be a live class.

*Probe `lexicon:transformation`.* transformation: 523 occurrences / 37 types, 6.556/1k vs general English 0.182/1k (x36.03, z 133.5); e.g. liquefied p4:601

*Annotation.* Transformation is processual: mutation, proliferation, liquefaction, excoriation, fusion are a live verb class far above prose. The book's horror is *becoming*, Tetsuo's central verb, even though the material that becomes is flesh rather than steel.

*What would falsify it.* a transformation register at prose density would reduce the body-horror to static description.

**TS3 -- fusion happens in contact: metal and organ in the same clause** -- **THIN**

*Why the prior expects it.* The prior image is metal *inside* the body, so the two registers should co-occur within clauses beyond chance.

*Probe `cooccur:metal+organ`.* metal+organ: 39 clauses hold both (expected 26.6 under random placement, z 2.41); e.g. clause p5:711

*Annotation.* Metal and organ vocabulary do share clauses, but not decisively above chance -- the fusion is staged rarely. Read beside TS1: the Tetsuo paradigm is accounted for by the transformation verbs and the body-interior register, not by metallic fusion as a dominant image.

*What would falsify it.* metal and organ vocabulary never sharing a clause would mean no fusion is ever staged.

**TS4 -- the horror is located in the body's interior** -- **PRESENT**

*Why the prior expects it.* Body horror's camera looks inward: organs, entrails, flesh as the stage of the event.

*Probe `lexicon:organ`.* organ: 1490 occurrences / 18 types, 18.679/1k vs general English 0.945/1k (x19.77, z 163.0); e.g. flesh p4:292

*Annotation.* The organ register is distinctive and among the densest families; #56's keyness independently names `organ`, `embryo` and `flesh` in its 32-word cross-corpus core. The interior camera is this book's camera -- the body-horror prior is met on its strongest leg.

*What would falsify it.* an organ register at prose density would locate the horror outside the body.

### J. G. Ballard

**BL1 -- a clinical register is over-dense against ordinary prose** -- **PRESENT**

*Why the prior expects it.* Ballard's diction is medicine-show: syndromes, sera, inoculations, pathologies as ordinary words, so the clinical register should stand far above prose.

*Probe `lexicon:clinical`.* clinical: 560 occurrences / 13 types, 7.02/1k vs general English 0.155/1k (x45.21, z 155.6); e.g. inoculation p4:222

*Annotation.* The clinical register is far above prose yet below a medical-abstract density: a novel with a medical accent, not a medical text. That middle position is itself the Ballardian finding -- clinical diction as literary method, measured between the two poles.

*What would falsify it.* clinical at prose density would remove Ballard's method rather than his themes.

**BL2 -- the affect profile is the new psychopathology** -- **PRESENT**

*Why the prior expects it.* 'Inner space' Ballard expects fear, ecstasy, paranoia and disgust to be *elevated together* -- the pathology is euphoric, not merely dreadful.

*Probe `artifact:affect_margins`.* fear z=11.07; ecstasy z=22.63; paranoia z=6.96; disgust z=14.7

*Annotation.* All four psychopathological affects are elevated against #44's English baseline, with ecstasy strongest. The elevation of ecstasy *with* fear, paranoia and disgust is the Ballardian shape: the pathology is euphoric, not merely dreadful. This is #44's classifier on #44's lexicon, read as a profile, not as ground truth about feeling.

*What would falsify it.* an affect profile flat against the English baseline would make the book's affect ordinary.

**BL3 -- violence is eroticized: sex and violence share clauses** -- **THIN**

*Why the prior expects it.* Crash's scandal is the erotic charge of the collision; the prior expects sexual and violent vocabulary fused at clause level.

*Probe `cooccur:sexual+violence`.* sexual+violence: 231 clauses hold both (expected 190.4 under random placement, z 2.95); e.g. clause p4:100

*Annotation.* Sexual and violent vocabulary co-occur in clauses about a quarter more often than random placement, but not decisively above the gate. Ballard's fusion is *suggested* here, not certified: the erotic register is dense (it clears z generically), the fusion specifically is thin. The verdict licenses 'eroticism and violence share the field', not 'the crash is eroticized'.

*What would falsify it.* sexual and violent registers in separate clauses would be Ballard's subject matter without Ballard's fusion.

**BL4 -- the pathology is organized as a case with testable claims** -- **THIN**

*Why the prior expects it.* Even at their most deranged, Ballard's texts keep a clinical scaffolding: named subjects doing definable things to definable targets, whose claims can be checked and can fail.

*Probe `artifact:roles_predicates`.* the case-history scaffolding is dense (4,829 actor->verb->target events) but the synopsis predicates are not merely incomplete: 1 role(s) refuted, 2 uncertain (#59)

*Annotation.* The clinical-case prior is supported and refuted in the same breath: the event layer is dense (thousands of actor->verb->target events, #66), yet #59 refutes one synopsis role outright (Super Cherry 666 never hunts, never co-occurs with the succubus family) and leaves two predicates uncertain (no ADAM/ANTi-ADAM clash event; no wage-family verb for the dog). A Ballardian case-history has checkable claims, and this text has exactly that: claims that mostly check, and some that fail.

*What would falsify it.* if the event layer were dense while every named predicate were refuted, there would be no case, only texture.

### Aase Berg

**BG1 -- reproduction is a live and distinctive register** -- **PRESENT**

*Why the prior expects it.* Berg's 'mommy trilogy' makes pregnancy, birth and the genetic a central grotesque; the prior expects the reproductive lexicon to be unusually dense.

*Probe `lexicon:reproduction`.* reproduction: 2616 occurrences / 28 types, 32.794/1k vs general English 0.607/1k (x53.99, z 368.8); e.g. chromosome p4:47

*Annotation.* Reproduction is the densest family in the book -- embryo, fetus, placenta, gene, chromosome, clone, fecundate, spawn. Berg's pregnancy grotesque and the biotechnological register share one lexicon here, and it dominates. The verdict licenses 'the reproductive is the book's central material'; it does not sort how much of that density is womb and how much is gene-war -- that split is not probed here.

*What would falsify it.* reproduction at prose density would make the fertility imagery incidental.

**BG2 -- the grotesque is a named register, not an accident** -- **PRESENT**

*Why the prior expects it.* The criticism of Berg describes a violently grotesque register; the prior expects a deformation vocabulary standing well above prose.

*Probe `lexicon:grotesque`.* grotesque: 951 occurrences / 22 types, 11.922/1k vs general English 0.248/1k (x48.1, z 209.4); e.g. cadaver p4:563

*Annotation.* The grotesque is a named, dense register: cadaver, deformity, corruption, decay, aberration. It is systemic, as the tradition's criticism describes it, not rhetorical.

*What would falsify it.* grotesque at prose density would make the deformation rhetorical rather than systemic.

**BG3 -- birth and grotesque fuse in single clauses** -- **PRESENT**

*Why the prior expects it.* In Berg the birth event *is* the deformation event, so the two registers should share clauses beyond chance.

*Probe `cooccur:reproduction+grotesque`.* reproduction+grotesque: 491 clauses hold both (expected 305.6 under random placement, z 10.61); e.g. clause p4:1634

*Annotation.* Birth and grotesque share clauses an order of magnitude above the placement noise: the birth event *is* the deformation event in this book, which is the Berg prior's central image. This is the strongest fusion result in the dossier.

*What would falsify it.* the two registers in separate clauses would separate birth from horror, which is not the prior's image.

**BG4 -- tenderness persists inside the grotesque** -- **UNDERDETERMINED**

*Why the prior expects it.* Berg's signature is horror blended with empathy -- the grotesque body is also loved -- so the prior expects tender wording inside the deformation register.

*Probe `declared`.* unmeasurable here: no workspace lexicon separates tenderness from ecstasy (see the annotation).

*Annotation.* Underdetermined, and stated as such. #44's affect lexicon measures intensity and valence classes (ecstasy above all) but no artifact distinguishes *tenderness* from *ecstasy*, so the prior's tender register cannot be probed without inventing a lexicon -- which would be a new measurement, not a dossier check. What would settle it: a tenderness lexicon scored with #44's own pipeline, with the same English-baseline contrast, declared before running.

*What would falsify it.* unmeasurable here: no workspace lexicon separates tenderness from ecstasy (see the annotation).

### Blake Butler

**BT1 -- syntax is deranged but not noise** -- **PRESENT**

*Why the prior expects it.* Butler's prose deranges syntax while remaining predictable-ish above chance; the prior expects parseability below ordinary English yet above a shuffle null.

*Probe `artifact:parse_derangement`.* parseable 0.521 vs English 0.785 and shuffle null 0.417: deranged yet above noise

*Annotation.* Parseable well below ordinary English yet clearly above the shuffle null: the syntax is deranged but not noise. Butler's prior expects exactly this posture -- a system that breaks grammar without dissolving into chance. Note the book sits *above* the null, so the derangement is structured derangement.

*What would falsify it.* parseability at prose level would be ordinary syntax; at null level would be noise rather than a system.

**BT2 -- phrases recur densely enough to be formulas** -- **PRESENT**

*Why the prior expects it.* Butler's roiling repetition: the same phrases churn, so a large share of the text should sit in repeated n-grams.

*Probe `artifact:formula_density`.* 25,969 recurring phrase formulas (n>=2, freq>=2)

*Annotation.* The recurring-phrase skeleton is large: tens of thousands of book formulas (n>=2, freq>=2) on the #17 word-part layer, #57's count, with 116 classified `author_idiolect`. Repetition at formula scale is the Butler prior's pressure: the same phrases churn.

*What would falsify it.* a text with no recurring phrase skeleton would be all novelty, no pressure.

**BT3 -- the book loops back on itself** -- **PRESENT**

*Why the prior expects it.* Butler's copycat structures and obsessive returns expect long-range self-similarity, not local coherence.

*Probe `artifact:self_similarity`.* band z: lag10 28.52, lag100 20.15, max 33.69

*Annotation.* Paragraph-level self-similarity rises with lag far above its shuffle null (#54): the book loops back on itself. Butler's copycat-structure prior is met at the macro scale, not the sentence scale.

*What would falsify it.* self-similarity at chance level would remove the loop structure the prior expects.

**BT4 -- narrative time is elliptical** -- **UNDERDETERMINED**

*Why the prior expects it.* Butler's narratives refuse chronology; the prior expects no recoverable temporal order of events.

*Probe `declared`.* unmeasurable here: no workspace artifact measures temporal order (see the annotation).

*Annotation.* Underdetermined: no workspace artifact measures the *temporal order* of events (#66's arcs are event-density profiles, not chronology), so 'elliptical time' cannot be checked without a new instrument. What would settle it: a chronology instrument over #66's event cites (tense, sequence markers, first/last page per referent), declared before running. The expectation is recorded rather than dropped.

*What would falsify it.* unmeasurable here: no workspace artifact measures temporal order (see the annotation).

### text-virus / codework poetics

**TV1 -- the surface is code-like: operators as a structural share** -- **PRESENT**

*Why the prior expects it.* Codework's poetics is the executable surface -- punctuation as syntax, operators as grammar -- so operator density should be a structural fraction, not ornament.

*Probe `artifact:operator_surface`.* 10,949 operator tokens over 77,392 words (14.1%); 5,773 clause-boundary operators

*Annotation.* Operators are a structural fraction of the stream -- over a tenth of the word tokens' count in operator tokens, with thousands of clause-boundary operators. The surface is code-like by measurement, not by impression: #14 parses the operator stream as a grammar and #13 removes it as silence.

*What would falsify it.* operators at ornament level would make the surface look code-like without being code-like.

**TV2 -- infection is a live and distinctive register** -- **PRESENT**

*Why the prior expects it.* The text-virus prior takes infection literally: virus, infection, parasite as the genre's governing metaphor made of words.

*Probe `lexicon:virus`.* virus: 650 occurrences / 15 types, 8.148/1k vs general English 0.058/1k (x139.56, z 299.0); e.g. infectious p4:1030

*Annotation.* Infection is a distinctive register -- virus, infection, parasite, disease, plague -- with a density near a medical-abstract corpus and two orders above prose. #57 found the one non-generic influence-excerpt overlap in the book is `a virus` (Burroughs, *The Ticket That Exploded*): the virus prior has a documented intertextual toehold.

*What would falsify it.* infection at prose density would leave the virus as a title only.

**TV3 -- the virus is digital: infection and technology share clauses** -- **PRESENT**

*Why the prior expects it.* In codework the virus is *code* -- the biological and the computational are one figure -- so the registers should be fused at clause level.

*Probe `cooccur:virus+digital`.* virus+digital: 186 clauses hold both (expected 133.9 under random placement, z 4.5); e.g. clause p16:587

*Annotation.* Infection and digital vocabulary share clauses above chance: the virus is digital here, the codework figure literalized. The verdict licenses 'infection and computation are one image field'; it does not identify the figure with any codework text.

*What would falsify it.* the two registers in separate clauses would keep the virus biological and the data merely technical.

**TV4 -- an asemic layer: invented surfaces outside the dictionary** -- **PRESENT**

*Why the prior expects it.* Codework and asemic writing expect machine-coined, non-dictionary surfaces as a visible layer of the text.

*Probe `artifact:neologism_coinage`.* 74 true coinages over 338 occurrences; 2,216 compound types

*Annotation.* A true-coinage layer exists outside the dictionary, with thousands of compound types beside it (#19). The asemic surface is a measured layer, not a reading impression.

*What would falsify it.* no true-coinage layer would make the invention decorative rather than systemic.

**TV5 -- a command/scream layer in capitals** -- **PRESENT**

*Why the prior expects it.* Codework's machine voice shouts in caps -- commands, labels, LOAD -- so an ALL-CAPS functional layer should be measurable.

*Probe `artifact:case_emphasis`.* caps layer: scream 263, label 1138, function 453 (share of words 2.44%)

*Annotation.* The ALL-CAPS layer is large and mostly functional -- labels, commands, function letters, screams (#22). The machine voice shouts in caps and the shouting is classifiable.

*What would falsify it.* no caps layer would leave the machine voice unwritten.

## 8. The honest shape of the book, as the priors found it

Ranked by density, the book's live registers are **reproduction** (32.794/1k), **violence** (22.051/1k), **machine** (21.161/1k), **digital** (18.892/1k), **organ** (18.679/1k). The thin families are **corporate** (x4.35 above prose), **metal** (x4.32 above prose), **sexual** (x9.82 above prose), **city** (x5.36 above prose).

The split is the finding. Where the priors predict *infrastructure* -- the corporate register of cyberpunk, the metal register of Tetsuo -- the corpus is thin: every thin family sits between x4.3 and x9.8 above generic prose, while every present family clears x15. Where the priors predict *interiority* -- the organ register of body horror, the reproductive register of Berg, the infection register of the text-virus -- the corpus is dense and distinctive. The fusions that are certified at clause level are machine x organ and reproduction x grotesque; the fusions the priors would most expect (metal x organ, sex x violence) are the two that fall just under the gate. A cyberpunk of the boardroom this is not; a cyberpunk of the body, clinically inflected, reproductively driven, deranged above noise -- that is what the priors find here.

**What this dossier does not license.** A verdict of *present* says the meaning-material a tradition is made of is measurably in the book; it does not say the author read that tradition. The one documented intertextual toehold in either direction is #57's: the book's only non-generic shared phrase with an influence excerpt is `a virus` (Burroughs). Conversely, a *thin* verdict does not exonerate the tradition -- the excerpt arms are too small to refute anything, which is why they are reported as bounds. And the two *underdetermined* expectations (BG4 tenderness, BT4 elliptical time) are open questions with named instruments, not negatives.

## 9. Reconciliation

| item | expected | observed | source | ok |
|---|---|---|---|---|
| book_parts_total | 80363 | 80363 | #56 keyness/summary.json | y |
| book_kept_tokens | 79770 | 79770 | #56 keyness/summary.json | y |
| book_types | 3683 | 3683 | #56 keyness/summary.json | y |
| clause_count | 5614 | 5614 | #33 clauses/summary.json | y |
| parts_assigned_to_clauses | 79770 | 79770 | #33 clause partition (char-span containment) | y |
| parts_unassigned | 0 | 0 | every kept part lands in exactly one clause | y |
| intertext_ledger_marks_priors_not_acquired | y | y | #57 analysis/intertext/sources.tsv | y |
| quick_build | n | n | mode flag | skip |

## 10. Reproduce

```bash
python work/genre_prior.py      # writes analysis/genre_prior/
python work/genre_prior.py --quick  # capped technical registers
python work/test_genre_prior.py
```

Offline and deterministic; no network, no clock in any statistic; rebuild is byte-identical except the generated stamp.  The prior ledger's probe date is #57's pinned survey date, inherited rather than re-derived.
