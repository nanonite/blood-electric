# Changelog

All notable changes in this workspace. The project decodes Kenji Siratori's *Blood Electric*
(2002) into an evidence-linked English reading. Chainlink issue numbers are given as `#N`.

## [Unreleased]

### Added
- Genre/prior dossier, `analysis/genre_prior/` (#60) — the plan's **L7.5** item (*cyberpunk, Tetsuo: The Iron Man, Ballard, Aase Berg, Blake Butler, text-virus/codework poetics set expectations for the kind of meaning present*) built as an **annotated dossier** (`dossier.md` — the deliverable; `expectations.tsv`, `lexicon_probes.tsv`, `cooccurrence.tsv`, `artifact_probes.tsv`, `verdicts.tsv`, `sources.tsv` — the prior ledger, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/genre_prior.py` + 48-test regression suite `work/test_genre_prior.py`. L7's rule governs everything: external material is **a prior, not the object of study** — so each of the 25 expectations is stated with its rationale, probed on the book's own frozen layers or read from the owning analysis's frozen artifact, verdicted (**present / thin / absent / underdetermined**), and annotated with what the verdict does and does not license. **No tradition text is ingested:** every prior work is in copyright, #57 already pinned the L7.5 priors as not-acquired (survey date 2026-10-14, inherited rather than re-probed), and the only tradition representation is #56's 105-token cyberpunk excerpt corpus — below #56's own 10,000-token rule, so non-inferential and reported as a bound, never as a genre test. Three probe kinds, all corpus-side, all declared before measurement: **(a) lexicon families** — 13 pattern sets full-matched on the #17 word-part layer screened exactly as #56 (80,363 parts / 79,770 kept / 3,683 types), measured as density per 1,000 parts beside the same family in each #56 register, with a Poisson contrast against general English — the gate is *distinctiveness* (≥100 occurrences and ≥15x the general-English rate) rather than significance, because every family clears a z against generic prose by orders of magnitude (the book is not generic prose), and the table publishes every number so a reader can re-gate; **(b) clause-level fusion** — a #33 clause *has* a family when any of its parts full-matches it, and the observed co-occurrence is read against random placement `n_a xn_b / 5,614` (every one of the 79,770 kept parts lands in exactly one clause, 0 unassigned — asserted and reconciled); **(c) artifact reads** whose pass rule is declared before the number is read (#19 true coinages 51 types / 338 occurrences, #32 parse 0.521 vs English 0.785 and null 0.417, #13/#33 operators 10,949 over 77,392 words = 14.1%, #44 affect z ecstasy 22.63 / disgust 14.70 / fear 11.07 / paranoia 6.96, #54 self-similarity z lag-10 28.52 / lag-100 20.15, #57 formulas 25,969 with 116 `author_idiolect`, #59 roles). **Verdicts: 18 present / 5 thin / 0 absent / 2 underdetermined, and the split is the finding — the book is thin exactly where the priors predict infrastructure and dense where they predict interiority.** The thin families are `corporate` (59 occurrences / 4 types, x4.35 above prose: Sato Corporation is #59-evidenced as a named `organization` with napalm/torture/vampire clusters but the generic corporate register is never rendered — a cyberpunk of bodies, not boardrooms), `metal` (114 occurrences / 7 types, x4.32 — Tetsuo's metal-flesh fusion is not this book's fusion image), `sexual` (778, x9.82) and `city` (284 / 2 types, x5.36). The present ones are led by `reproduction` (2,616 occurrences, 32.8/1k, x54 — embryo 335, fecundates 293, gene 229, chromosome 224, genome 207), `machine` (1,688, x80), `organ` (1,490, x20 — brain 511, flesh 409, organ 290), `digital` (1,507, x61 — and neologism-prone: `cyber*` matches cyberBuddha), `virus` (650, x140, at near medical-corpus density 9.9/1k), `grotesque` (951, x48 — cadaver 633), `clinical` (560, x45 — above prose, below a medical abstract: a novel with a medical accent, the Ballardian middle position), `violence` (1,759, x40) and `transformation` (523 / 37 types, x36 — mutation, liquefaction, excoriation are a live verb class). The fusions certified at clause level are **machine x organ** (403 clauses, expected 323.5, z 4.42 — the cyborg interface staged inside the utterance) and **reproduction x grotesque** (491, expected 305.6, z 10.61 — birth *is* the deformation event, the Berg prior's central image), while the two the priors would most expect, **metal x organ** (39, expected 26.6, z 2.41) and **sex x violence** (231, expected 190.4, z 2.95), fall just under the z = 3 gate: Ballard's eroticized crash is *suggested, not certified*, and his clinical-case prior is supported and refuted in the same breath (4,829 actor->verb->target events, yet #59 refutes Super Cherry 666 outright and leaves the ADAM/ANTi-ADAM clash and the dog's war unwaged). The two underdetermined expectations are written down with the instrument that would settle each rather than dropped — **BG4** tenderness inside the grotesque (a tenderness lexicon scored with #44's own pipeline) and **BT4** elliptical narrative time (a chronology instrument over #66's event cites) — because an unmeasurable expectation silently dropped is the residue-dropping anti-pattern #76 audits. The dossier's closing guardrail is mechanical, not decorative: *present* licenses only that a tradition's meaning-material is measurably here, not that the author read it; the one documented intertextual toehold in either direction is #57's (`a virus`, Burroughs). The book side reconciles against #56's word-part counts (80,363 / 79,770 / 3,683) and #33's clause count (5,614); the 48-test suite pins the gates on synthetic families (below the ratio gate must be thin, below the z gate must be thin, a never-occurring family must be absent), the probe primitives on known sets, every artifact probe's declared rule in both directions via a planted-summary registry whose coverage test fails the build if a probe is added without planted cases, mode-invariance (quick caps only the medical/cs/legal registers, so general English and every verdict are unchanged), determinism on rerun, and report-vs-table numeric agreement — and it caught a real bug in development: a `viruses?` pattern matching "viruses" but not "virus", which undercounted the infection register by 71 occurrences until an exemplar test failed. Review fixes: the #19 artifact probe had read `coinage_types` (74 = true_coinage 51 + proper_noun 23) while pairing it with the true_coinage-only 338 occurrences, so the type and occurrence figures described different populations -- it now reads `true_coinage_types` (51) on both legs and discloses the 74 union beside it; and the artifact-probe rule coverage the previous entry claimed was four probes deep, so the planted-summary registry now pins all ten rows (nine registered probes plus the cyberpunk_arm context row) in both directions under a coverage guard. Offline, deterministic, no clock in any statistic, rebuild byte-identical except the generated stamp.
- Burstiness and Heaps' law, `analysis/burstiness/` (#53) — vocabulary growth vs text length, and the burstiness of entity use. **Growth is ordinary, not alien:** the unit layer's Heaps β is **0.603** (log K 1.970, R² 0.995) with a leave-one-page-out jackknife **se of 0.011** — `sd × sqrt(n−1)` across the 225 page refits, whose plain sd is 0.0008 — inside the eight pinned #24 references' range 0.540–0.747 and below their mean 0.637 (the references themselves span se 0.008–0.020). The book mints vocabulary like a novel, with per-fifth novelty falling 2,403 → 1,306 → 873 → 631 → 655 new types (the seg-3 β spike 0.91 is the disclosed p90 OCR loop minting one-off types, not coinage) and #24's coinage subset (β 0.835) still the only fast-growing layer. The jackknife refit **mean is not the point estimate** (−2.2e−06 for the book, up to +1.9e−03 for the matched-N references, because dropping a group shortens the curve and moves the checkpoint grid), so only the spread is used and the bias is published as its own column. **The cast is introduced early:** over the entity-mention stream the named layer grows at β 0.367, 64/98 entities are first attested in the token stream's first third (the mention partition gives 64/20/14 — different denominators, labelled as such), and 65% of the cast is on stage by the first mention-third — a fixed dramatis personae, not an open-ended naming system. **Entity use is bursty, and so is the coinage layer:** of the 39 entities with n ≥ 8, 20 are over-dispersed against a placement null at their own rate (2,000 seeded draws per term, each a uniform random subset of the stream so no token is used twice), while **none is certified more evenly than chance**: six rows fall under the descriptive F(1000) < 0.9 cut, but the low tail of the same null certifies none of them (SM is the only one close, p 0.128; the statistic is coarse at these rates and the tie mass is published). `retro-ADAM` (F(1000) 9.82, B 0.667) is confined to pages 87–176 — a late-book arc — and `ADAM`'s F(1000) 4.25 counts the 166 occurrences that are also `ADAM Doll`, falling to 2.87 without them (the opposite nesting, `cadaver` inside `Cadaver City`, runs 2.14 → 4.57). The aggregate entity channel (1,988 distinct positions, 2.7% of tokens) has F(1000) 3.02 against a null median 0.95 — over-dispersed, but *not* the book's most clustered repetition, because the merged coinage-type channel reaches 3.54 and the merged English-word channel 2.81 when each is compared as a merged channel against its own rate-matched null: *named* vocabulary in general clusters in this book. `HIV` stays neutral despite p < 0.001 because its inter-arrival gaps are more regular than geometric (B −0.262), the negative-B leg of the rule; a missing B no longer vetoes a significant high tail. The coinage controls drop the glossary's own entity surfaces (`retro-ADAM`, `ToKAGE`) before selection, so the control population no longer contains the effect it is meant to measure. The densest clusters are the corpus's disclosed OCR recognition loops (p219 `konganic-eye of cyberBuddha` ×3–4, p90's type-minter), reported as bursts with the cause named rather than narratively. Three legs per entity (Fano factor at 100/500/1000 tokens, Goh–Barabási B, and both tails of the placement null) with the thresholds disclosed. Screens are disclosed and never mixed — growth on #24's screen (74,318 units) so every β reproduces #24's `fits.tsv` to 1e-9, entities on #27/#30's (74,077) so every occurrence count reproduces #30's glossary 98/98 — and the β uncertainty is a leave-one-group-out jackknife, chosen over a page bootstrap because duplicating a page contributes no new types and biases β down (both properties pinned by tests). Deliverables: `burstiness_report.md` (the deliverable), `heaps_fits.tsv`, `growth_curve.tsv`, `segment_growth.tsv`, `entity_growth.tsv`, `entity_burstiness.tsv`, `controls_burstiness.tsv`, `channels.tsv`, `nested_entities.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`; `work/burstiness_heaps.py` + 59-test regression suite `work/test_burstiness_heaps.py`. Offline on the pinned cache; deterministic (fixed seeds, no clock), rebuild byte-identical.
- Human jury / reader study design, `analysis/reader_study/` (#75) — the plan's **L9.7** item (*have readers rate glosses for fidelity to the felt text*) **designed rather than run**: `protocol.md` is the pre-registration (question, population, design, instrument verbatim, analysis plan, stopping rule, and what the study cannot establish), and the artifact set beside it is a *simulated* pilot, labelled `simulated = True` in every row with `human_data_collected: False` pinned by the suite — **no human rating exists in this workspace, and no claim rests on the simulated numbers.** #62 published an interlinear gloss, #63 five LLM strategies over a 5-clause stratified sample, #64 classified what those strategies disagree about, #68 ranked readings by compression and #74 by round-trip fidelity; each is a proxy for *fidelity to the felt text*, and each can be gamed by the machinery that built the glosses (#63's own report notes the model is not deterministic and there is no ground-truth translation). **Instrument: 7 items per stimulation** — `felt_fidelity` (0–7, the primary outcome), `fidelity_confidence`, `reading_appeal` (a *diagnostic* item, deliberately not the primary in disguise), two anchored forced choices, and **two planted traps** carrying #63's `plain_english` reading (the strategy #63's checks most often fail and #74 measures least faithful, aggregate 0.102): `trap_overcorrected` asks how faithful a *smoothing* of the derangement is and `trap_invented` how faithful an *added plot event* is — both read pass/fail against a known answer rather than as a scale, because they sit at the floor where a 7-point scale has no resolution left. **Sampling: a *census* of the gloss population over #63's stratified sample** — 45 cells (5 clauses × [identity control + #62 gloss + 5 LLM strategies + 2 traps]), each block showing 6 with the LLM comparators rotated by `PAIR_OFFSETS` and the within-block order rotated by the block index, so a reader never rates a gloss without the clause it translates beside it and no reading is systematically first. **Analysis, pre-registered before any number exists:** per-cell means with seeded bootstrap CIs; a within-clause contrast (identity control minus the other readings, pooled per clause) tested by *permuting the control label* — each clause's values, size and difficulty survive while the claim being tested does not; Krippendorff's alpha with the **interval** metric over cells as units (a 0–7 scale carries equal intervals; the ordinal and nominal alternatives are implemented and disclosed, since with two observed categories the ordinal metric cannot tell a 0-vs-6 disagreement from a 5-vs-6 one and the nominal metric drives alpha to zero however well readers agree), each alpha read against a null that *re-groups* the values — shuffling inside a unit is a no-op and so is not a null; Spearman of the human per-reading mean against #74's published `aggregate_fidelity` with a label permutation; Kruskal–Wallis H with eta²; per-stratum descriptives never pooled. **Pilot results (properties of the disclosed generative model, none of them evidence about readers):** 1,800 integer ratings over 5 blocks × 12 synthetic readers × 5 items; the identity control tops every block (contrast **+2.43**, label-permutation p = 0.0015); `felt_fidelity` alpha **0.790** against a null centred on zero (q05 −0.030, q95 0.035, p = 0.005); machine/human Spearman **ρ = +0.893** (p = 0.014); H = 18.10, η² = 0.993; and the appeal table shows the split the instrument exists to expose — `plain_english` is the *most appealing* reading (6.13) and among the least faithful (3.92). The generative model's felt-fidelity expectations are anchored to #74's channels but deliberately **not** a monotone function of them (the two operator-destroying English readings sit above the machine's ranking, because they are the ones a reader finds easiest to read), and a test fails if the ranking ever becomes an exact copy of #74's, which would make the machine/human leg a tautology. **Two controls are build errors rather than footnotes:** the identity control must top its block or the build stops, and the attention check must be able to flag a reader (1 of the 12 synthetic readers is). The alpha implementation itself is pinned against the published worked example in Krippendorff's 2011 note (binary data, two observers, no missing values -> 2/21). **L9.8 audit:** circularity is structurally impossible (no model generates or grades anything; the only LLM output in the frame is #63's pinned capture, treated as a candidate); single-method certainty is answered by 7 items, 4 statistical legs, 3 nulls and per-stratum reporting; residue-dropping by publishing the cells a block does *not* draw, counting the traps and excluding them by role, and publishing reader exclusions with their reasons; over-correction is *measured* as appeal rather than corrected away. Disclosed limits: no ground-truth translation exists (#74), so a high score means readers recognise the feeling rather than that a reading is *the* translation; readers are not the text's original audience; and a 5-clause stratified sample is a sample of #63's strata, not of the book. Reconciliation: 5,614 = 5,614 interlinear clauses, 5 = 5 sampled clauses, 25 = 25 gloss rows, 5 = 5 strategies — a mismatch is a build error. Offline, deterministic, no clock, seed 75, byte-identical rebuild; 62-test regression suite `work/test_reader_study.py`.
- Back-translation harness, `analysis/back_translation/` (#74) — the plan's **L9.6** test (*gloss -> regenerate a Siratori-style clause -> compare to original*) run as a round-trip fidelity harness. There is no ground-truth translation of *Blood Electric*, so this measures **bidirectional consistency, not accuracy**: each published reading is the candidate gloss, a deterministic offline re-writer regenerates a Siratori-style clause **without seeing the original**, and four independent channels score the pair — **`operator_recall`** (primary; a *Dice* coefficient over the operator multiset, so one number charges both a dropped operator and an invented one), **`operator_order`** (LCS ratio — catches a right-multiset/wrong-order gloss), **`content_recall`** (lexis) and **`coined_recall`** (seam surfaces `BABEL=TV`, `soul/gram`). **The channels split, and that is the finding:** v3/v4 and #62's `interlinear_gloss` keep every word and lose every operator (`content_recall` 1.000, `operator_recall` 0.000); `plain_english` loses both, because it rewords as well as de-operatorises (0.408 / 0.000); and only the operator-preserving strategies round-trip the machine — **`register_paraphrase` is the sole reading that recovers the operator channel in full** (1.000 / 1.000 / 0.937 / 0.700, aggregate 0.909, first on the primary leg), exactly because it reproduces the clause near-verbatim, while `literal_operator` is charged for the operators it *adds* (Dice 0.905, the failure #63's own `operator_fidelity` check already found). Three controls, each a build error rather than a footnote: the **identity control** (self-round-trip must score exactly 1.000 on every channel), the **operator-strip ablation** (must floor `operator_recall` to 0.000 while `content_recall` survives — proving the primary leg is about the operator, not crediting shared lexis) and the **permutation null** (shuffling the clause<->reading pairing must collapse the aggregate, so the score is not the vocabulary Siratori's clauses share with each other). **Circularity is structurally impossible** (L9.8): no LLM is invoked; the glosses are #63's pins, the operator scanner is #63's own eleven-surface scanner (asserted equal to `glossing_pipeline.py` over 400 corpus clauses), the judge is this module. The regenerator's **limit is disclosed, not dressed up**: a bounded re-render that deliberately refuses to invent the structure a lossy forward pass destroyed, so the round trip records what a reading *lost*; a `[...]`-keeping operator count is shown to overscore, and the `residue` stratum is retained. #63's `operators_out` total (54) is published beside this build's scored count (47), the 7-operator gap being exactly the operators nested inside the glosses' editorial asides. Reconciliation: **5,614 = 5,614** interlinear clauses, **5 = 5** sampled clauses, **25 = 25** gloss rows — a mismatch is a build error. Deliverables: `back_translation_report.md` (the deliverable), `fidelity_scores.tsv` (the result), `round_trips.tsv`, `worked_examples.tsv`, `controls_and_nulls.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`; `work/back_translation.py` + 36-test regression suite `work/test_back_translation.py` (primitives against #63's scanner, controls, round-trip inventory, end-to-end build with determinism, and planted failures — a sabotaged regenerator and a bracket-keeping operator count must each be caught). Offline, deterministic, no clock; rebuild is byte-identical.
- Compression test for candidate readings, `analysis/reading_compression/` (#68) — the plan's L8.7 test run as a harness: **a good gloss shortens the description of the book**, so every candidate reading the repository has published is scored by whether it makes the corpus more compressible *and* more predictable. Nine readings, all read from frozen artifacts and none invented here: the **identity control** (the untouched source clause, 5,614 clauses), #13's `v3_operators_to_space` and `v4_operators_to_period`, #62's `interlinear_gloss`, and #63's five LLM prompt strategies over its 5-clause stratified sample — the five strategies therefore form a second, smaller population that is ranked separately and never pooled with the full book. **Why three legs instead of raw bits/char:** #48 measured the book against matched English and concluded that raw compressibility *cannot discriminate between readings* (the book is already more compressible than English prose). The harness answers that with (1) a **length-matched deletion control** (the declared primary leg) — a position-preserving *letters-word* subsample of the original down to the reading's letters-word count (a word is a maximal run of letters, not a whitespace chunk, which would miscount the readings because operator removal changes the characters *between* words), repeated over five independent seeds with its spread reported (sd ≈ 4e-4 bits/char on the full book, so the full-book deltas sit two orders of magnitude above the control's own noise) — so a reading whose entire gain is deletion is pinned near zero and only restructuring scores negative. **Its limit is disclosed rather than hidden:** a reading with at least as many words as its clause gets no subsample, because there is nothing to delete, and the control is then the original; the corpus's readings are word-count preserving or expanding on ~5,470 of 5,614 clauses (v3/v4/gloss), so for them the control is the original on most clauses and the delta reduces toward the raw bits/char delta. The live-clause count is published per reading (`control_fallback_clauses`: 135 live clauses for the gloss, 144 for v3/v4) instead of assumed, and the report states in §1, §3 and §7 that the leg demonstrates these readings win *without deleting* rather than manufacturing a correction; (2) a **pairing shuffle null** — the reading's clause texts permuted across its clause ids, separating "needs the right text on the right clause" from "just needs short text"; (3) **held-out predictability** — a character trigram with Jelinek-Mercer interpolation (#56's estimator, λ tuned on a train-internal validation split) scoring clauses it never trained on, with one **fixed** model trained on the identity train clauses only so all readings share a single scorer, plus an **adaptive** per-reading model for the full-book readings. The held-out split is every 5th clause (1,123 test / 4,491 train), spread over 219 of 225 pages with no two adjacent clauses both held out (#56's leakage lesson). Reconciliation is exact where it can be and disclosed where it cannot: **5,614 = 5,614** clauses, **225 = 225** pages, **76,897 = 76,897** word tokens against #33/#62 (a mismatch is a build error), and the #48 compression rows agree to within the disclosed scope difference (identity 2.107 vs #48's 2.135 bits/char; clause-joined stream vs page stream). **Verdict — `interlinear_gloss` wins the declared primary leg** (length-control delta −0.0671 vs v3's −0.0647 and identity's 0.0000) and is therefore the reading that most shortens the description of the book; the ranking is sorted by that rank, and the unweighted mean of the three ranks is published beside it as a labelled secondary aggregate that can never outvote the primary. **The legs then split, which is the finding:** `v3_operators_to_space` wins *both* predictability legs (fixed-model gain −0.0135 bits/char across the 1,123 held-out clauses — 304 improved / 208 worsened with the remaining 611 tied where the reading equals the original — sign-test p = 2.6e-05; own held-out self-surprisal 2.4989 vs identity's 2.5380) while `interlinear_gloss` wins only the description-length leg. A reading that only rewrites orthography is the more predictable one; a reading that also substitutes the termbase is the more compressible one. **The instrument disagreement is reported, not reconciled away:** v4 — #50's GPT-2 winner — is the *worst* full-book reading here (fixed-model gain +0.1558; 27 improved and 485 worsened among the 512 non-tied pairs; the remaining 611 of 1,123 held-out clauses tie (p ≈ 0)), because a model trained on this book expects the book's own operator characters and is surprised when they become periods, while GPT-2 is a general-English LM that *expects* periods — two instruments, two defensible answers, the single-method-certainty trap L9.8 names. External anchor: the untouched held-out book scores **4.134 bits/char under a general-English character model** (#56's 15 reference novels) versus ~2.5 under its own models. The guardrail is carried into every artifact: **compression is evidence about predictability, not proof that an interpretation is true**. A reading missing a leg takes the *worst* rank on it (never an arbitrary order by input position — pinned by a unit test), and every sample-population reading lacks the adaptive leg, so that leg contributes a constant there instead of reordering the population. The limits section also discloses the control's honest boundary — it neutralises *arbitrary* deletion but not *structured* deletion, so a degenerate "keep each clause's first word" reading wins the compression leg (the book's clause-openers are strikingly repetitive) and only the predictability legs and #63's fidelity checks stop it being mistaken for a reading. Deliverables: `reading_compression_report.md` (the deliverable), `ranked_readings.tsv` (the ranking, both populations), `compression.tsv`, `controls_and_nulls.tsv`, `predictability.tsv`, `readings.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`; `work/reading_compression.py` + 58-test regression suite `work/test_reading_compression.py` (primitives with planted failures, reading inventory, end-to-end build, protocol/leakage properties, determinism, cross-artifact word-rule agreement, and report-vs-artifact numeric consistency). Offline, deterministic, no clock; rebuild is byte-identical.
- Spectral reading artifact, `analysis/spectral_reading/` (#67) — the silence-removed stream published as a **distinct under-text reading, explicitly not 'the real text'**. The v4 transform (`v4_operators_to_period`, the variant #50 found most lowers LM perplexity: 84.05 vs the 106.22 baseline) is presented as a continuous, page-tagged reading with a framing note that documents the derivation, the provenance and the interpretive limits; the reading section and the colophon each repeat the disclaimer, and the suite fails if the artifact ever claims to be the real text. 11,645 silence sites (10,949 operators + 696 free-standing dashes, #13's disclosed judgement call) become 10,435 inserted periods — with the corpus's 22 pre-existing periods, 10,457 periods and 10,136 word-bearing segments (median 6 words). **Every headline number is computed from #13's own primitives or read from its published artifact and reconciled row by row (14/14 match: 225 pages; 77,392 word tokens with the baseline's token sequence preserved exactly; segments 10,136; uppercase-initial 25.5% / 27.6% length-controlled vs the baseline's 52.9% / 56.2% on 17 segments and 22 periods; #50's perplexities) — a mismatch is a build error, not a footnote.** The limits are stated with numbers: 12.7% of the segments are a single word and a quarter start uppercase, so this is a **sentence skeleton, not a reconstruction**, and the transform erases the operators' distinct functions (`::` phrase separator, `//` utterance boundary, `=` naming seam all become periods). A rule-selected worked example (p18 source line 3, folio 20) shows one line before and after so the transform can be checked by eye, and the counting-rule residue is itemized rather than dropped (5 token-less period segments, each a lone `:` between two operator-derived periods). Deliverables: `spectral_reading.md` (the publishable artifact), `spectral_reading_report.md` (the verification report), `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`; `work/spectral_reading.py` + 57-test regression suite `work/test_spectral_reading.py`. Offline, deterministic, no clock; rebuild is byte-identical.
- Entity roles seeded from the jacket synopsis and verified against the text, `analysis/entity_roles/`
  (#59) — the role table (the deliverable), the per-probe evidence, the refutations and the
  reconciliation. Nine roles are read **only** off the back-cover synopsis (p224; its five
  sentences shared with #66 so they cannot drift) and tested against the book text pages 1–223,
  the synopsis page excluded from its own check. Each role is a set of named probes (presence,
  recurrence, centrality, the #30 role category, a clause co-occurrence, an actor/verb event),
  each defining or supporting; the evidence is recomputed with #66's primitives over the frozen
  #17/#33/#30/#27 artifacts. A role is `evidenced` when every defining probe passes, `uncertain`
  when some do, and `refuted` when no *substantive* probe does — presence is a precondition, never
  a role. **Verdict: 6 evidenced / 2 uncertain / 1 refuted.** Sato Corporation is an
  `organization` whose napalm+torture / torture+victim / vampire+chaos clusters fire; Cadaver City
  and Placenta World are `place`s; and **EVOL is the pink-ash planet the jacket names** (`The
  planet of the angel mechanism of the ADAM Doll is EVOL`, `p77:111`). The two uncertainties are
  predicates, not entities: ANTi-ADAM is the only entity #30 calls an `antagonist` and it
  co-occurs with ADAM once (`p216:1132`) but **no clash event links them**, and the dog is a
  recurrent, violent actor (459 mentions, 72 hostile events, 32 dog+zero clauses) but **no
  `wage`-family verb stands with it** anywhere. **The refutation is Super Cherry 666**: it is a
  `concept`, never a succubus form, never takes a hunt verb and never stands with the corporation,
  while the hunting formula itself is real and belongs to ToKAGE (`p147:70`) — the jacket copy
  attaches the formula to the wrong entrant. Offline, deterministic, byte-identical rerun;
  47-test regression suite `work/test_entity_roles.py`.
- Intertextual n-gram matching against the Siratori corpus, `analysis/intertext/` (#57) — the
  overlap report (the deliverable), the source survey, and the acquisition ledger. Every book
  n-gram (n = 1–6) recovered on the #17 word-part layer, markup-screened exactly as #56, is matched
  against a *different*, freely published Siratori text (the *Great Works* piece, first three
  paragraphs reused and attributed), the author's own interview turns (via #58), the #56 cited-
  influence excerpts (Gibson / Stephenson / Sterling; Burroughs / Artaud / Breton; Joyce), and the
  #56 general/technical registers — with a shuffle null (5 permutations, seed 57) so every count is
  read against chance. **Verdict — an honest split.** The named-work half is a documented negative:
  all five works the task names (*Headcode*, *Human_Worms*, *Smart-d*, *GENEDUB*, *Nonexistence*)
  are in-copyright print editions (the only reachable copy is a borrow-only Internet Archive scan of
  *Headcode*, `access-restricted-item=true`), so **0 of 5 named works is ingested** and no other
  text silently stands in for them. The idiolect half is a positive: of 25,969 book formulas
  (n ≥ 2, freq ≥ 2), **116 are `author_idiolect`** — present in the author's own text and in *no*
  general/technical register (104 in the different Siratori work, 12 in the interview paratext
  alone) — and the signal is far above chance (same-author probe n = 3: 97 observed vs 7.2 expected,
  13.5×; n = 4: 36 vs 0.4). The signature formulas are `soul gram`, `cold blooded disease`,
  `retro adam`, `brain universe`, `dna channel`, `data mutant`. The genre side is underpowered by
  construction (the influence excerpts are 49–105 tokens): no influence arm shares a trigram and the
  only non-generic shared bigram is `a virus` (Burroughs, *The Ticket That Exploded*), so
  genre-vs-English cannot be separated at that size — reported as a bound, not a finding.
  **Provenance is verified, not asserted:** the embedded excerpt is pinned by token count (330) and
  sha256, and `--probe` re-fetches the live page to check it; every reference-corpus file is
  verified against #56's manifest sha256 (a mismatch is a hard failure in `reconciliation.tsv`); and
  the one interviewer turn in #58's Q/A table is excluded from the author arm. Offline,
  deterministic, byte-identical rerun; 52-test regression suite `work/test_intertext.py`.
- Self-similarity matrix / loop detection, `analysis/self_similarity/` (#54) — the paragraph ×
  paragraph cosine self-similarity matrix built from #39's clause vectors, and its structural
  reading. The unit is a blank-line run of the #7 layout page body (733 runs plus 1 added so that
  no clause is dropped = 734 records; 717 carry a #33 clause); every #33 clause is joined to exactly
  one paragraph by offset overlap on `(page, start, end)` (5,614 assigned, none dropped, 23
  disclosed as spanning a break), and a paragraph's vector is the mean of its clause vectors
  (`units` view, SBERT, renormalised — no encoder runs for the book). **Verdict: the "loop" is a
  short diagonal band plus repeating blocks, and neither is a period.** The similar cells are not
  organised as a running diagonal — of the 9,578 paragraph pairs above cosine 0.8, 0.6% sit within
  one paragraph of the diagonal (null 0.3%) — and the band is real but short-range (lag-1 mean
  cosine 0.5648 against a 200-draw order-permutation null of 0.5382, z = +10.8; significant to
  ~lag 30, at the null from lag 40). The load-bearing long-range structure is a **verbatim return**:
  the opening section pages 4/5/6/7 recurs on 220/221/222/223 (page cosine 0.9550–0.9998,
  paragraph-level block mean 0.9488 over 16 cells, `difflib` raw-text identity 0.98, encoder-free
  lexical control 0.75 against a 0.63 page background) — #35's page-level near-duplicate recovered
  from the embedding side without any n-grams. Nine further paragraph-scale blocks are thematic
  repeats (cosine ~0.87, text identity ~0.15), and there are **no template blocks at page scale**
  above the per-level 99.5th-percentile cut. Controls: a matched English paragraph baseline built
  the same way from #24's pinned cache (mean cosine 0.327 vs the book's 0.538; 0.03% vs 3.73% of
  pairs above 0.8), the encoder-free lexical cosine, and a view/encoder sensitivity pass (the block
  and the headline survive #39's `raw` view and its BGE encoder). Deterministic (fixed seed, 200
  permutations, no wall-clock); rebuild byte-identical. One defect found and fixed on the way: the
  band-share denominator counted both triangles of the symmetric matrix while the numerator counted
  one, halving every reported in-band share (and doubling the headline pair count); the denominator
  is now the true unordered-pair count, which leaves every z-score unchanged and corrects the
  shares.
- Operator-word mutual information, `analysis/operator_word_mi/` (#52) — the operator grammar as
  statistics. Every one of the 10,268 screened operator occurrences (#17's 11,000 minus the 732
  raw-HTML `<`/`>` operator tokens, so #10's 10,949 reconciles exactly) is resolved on the unit
  layer to the term it stands next to on each side: a free operator takes the nearest surviving
  unit surface, an in-unit operator its own operand substrings (`=` in `suck=blood` → `suck`
  left, `blood` right). Terms are matched against #29's six seeded domains, the literal parsed
  from `work/cooccurrence_network.py` with `ast` rather than duplicated. The matrix (operator ×
  domain pointwise MI over the incidence table) is backed by per-(operator, domain) 1-df G²
  tests with Benjamini–Hochberg q, three MI totals per side (term-level I(O;T), domain-coarsened
  I(O;D), incidence multi-information) each against a 200-replicate label-permutation null with
  both marginals preserved, plus strict-seed and book-half robustness. **Verdicts: the operator
  predicts the adjacent term (I(O;T) = 1.1938 bits right, 1.2013 left; null 0.5488 / 0.695,
  z = 124.8 / 90.1) — 9× #11's operator-to-operator lag-1 MI. `::` before body terms is
  rejected (lift 0.39, q < 1e-4 — it prefers machine/genetics terms); `=` before tech terms is
  confirmed (machine lift 2.49, body lift 2.94, genetics on its left lift 1.76 — `=` is the
  book's cross-domain naming binder); `<<` opens blocks of flesh (body lift 1.72, `entrails`
  30× at PMI 3.48).** Disclosed limits: #29's seeds cover 16% of right-hand neighbours (the
  matrix is about the domain-bearing minority), multi-domain terms are incidences in each
  domain, and `+` (n=15) / `|` (n=17) are reported but not verdicted. One upstream defect found
  on the way: #29's source comment says `blood` is body + violence, but its literal
  `DOMAIN_SEEDS['violence']` has no `blood` — this build follows the literal and says so.
- LLM glossing pipeline (five prompt strategies), `analysis/glossing/` (#63) — the first LLM
  output in the workspace, and the first thing built so it can be audited rather than believed.
  Five strategies — `literal_operator` (operator channel preserved verbatim), `register_paraphrase`,
  `scene_summary` (operators dropped by instruction), `plain_english` (operators verbalised),
  `entity_consistent` (#30 names and roles held) — gloss a **stratified sample of 5 of #33's 5,614
  clauses** (residue → operator-dense → entity-named → quoting-frame → plain-control, one per
  stratum, drawn by a published rule and published in `sample.tsv`, not chosen for readability;
  the tie-break is on the clause id **as text**, so `1003` is drawn before `101` and the sample is
  *not* the numerically earliest clause of each stratum — disclosed in the report and pinned in the
  suite, because re-drawing it would move every prompt and need a fresh capture), producing 25
  glosses. Every prompt carries the clause's own operators with their #12/#14 readings
  ("a prompt never shows an operator the clause does not contain", so a "preserved" channel can be
  checked rather than trusted), the #30 rows for the entities it names, its #62 unresolved surfaces
  and its #19 `ocr_error` artefacts, plus one standing constraint block identical across all five
  strategies. **The load-bearing output is the verdict layer, not the glosses:** four deterministic
  checks — `operator_fidelity` (the operator subsequence, read off #17's stream rather than a regex
  over prose), `entity_fidelity` (both a dropped #30 name and an invented proper noun), `residue_discipline`
  (a marked-unresolved surface smoothed into an ordinary word), `frame_preservation` (the `<< >>`
  quoting frame) — score each gloss against its own strategy's contract, and no check calls the model,
  so the model cannot grade itself. **Verdicts: 15 pass / 10 fail; per check, `operator_fidelity` 17/25
  (68%), `entity_fidelity` 23/25 (92%), `frame_preservation` 6/6, `residue_discipline` 5/5; mean
  cross-strategy Jaccard agreement 0.364 (min 0.229, no unanimous clause).** The mode is pinned and
  offline: `PINS` is one dated capture run (2026-10-10, `opencode run --model
  opencode/space-bunny-free`), each pin keyed by the sha256 of the prompt it answered, so a change in
  the constraint assembly fails the build instead of silently reusing a stale response; `--live`
  re-runs the backend and prints a diff against the pins rather than rewriting them, because the same
  prompt was run three times and returned three different glosses — a number that moves with the
  sampler cannot be regression-tested. Every prompt is published in `prompts.tsv`. A build-time defect
  found and fixed here: the summary counted a check's non-applicable rows as passes, so
  `residue_discipline` read 100% over 25 glosses when it judged 5 (and disagreed with the report's own
  table); the denominators are now published per check in `check_counts` — and, in the same class, the
  per-gloss columns counted non-applicable rows as passes too, publishing "3 checks passed of 2
  applicable" on 20 of the 25 rows and naming `residue_discipline` as passed for clauses carrying no
  residue, so `n_passed <= n_applicable` is now a suite-tested invariant. New:
  `work/glossing_pipeline.py` + `work/test_glossing_pipeline.py` (95 tests, including a
  suite-wide assertion that the operator scanner agrees with #17 on every clause, that it re-derives
  #17's 5,227 in-clause / 11,000 total operator tokens, and that the pinned build is byte-identical
  on rebuild).
- Multi-agent translation committee with disagreement log, `analysis/committee/` (#64) — #63's five
  prompt strategies as the committee's agents, run through a **generator / critic / comparer loop**
  over their 25 pinned readings. The generator replays #63's pins through #63's own
  `resolve_glosses`, cross-checked against the published `glosses.tsv` (gloss text, verdict, operator
  columns, response sha256); the critic re-judges every reading with the four deterministic checks
  recomputed from the frozen artifacts, cross-checked against the published `checks.tsv`, so a drift
  between the two is a build error rather than a finding; the comparer localises every pairwise
  divergence to surfaces — each operator symbol (kept/dropped/added, counted), each #30 entity
  (carried/dropped), each #62 residue surface (kept/smoothed), the content-word symmetric difference —
  and classifies each row. Four classes: `rule_violation` (the rule governing the surface is
  breached by one reading's own treatment of it, or by both — the responsible agent is named),
  `contract_divergence` (both readings treated the surface as their differing instructions required —
  a keep-the-channel strategy vs a drop-it strategy), `genuine_ambiguity` (both readings pass every
  check that applies to them and no instruction explains the difference), `defect_unattributed`
  (a participant failed some other check — recorded rather than silently counted as ambiguity).
  **136 rows over 50 pairs: 60 rule violations, 28 contract divergences, 19 genuine ambiguities, 29
  unattributed defects. Headline: 4 of the 5 sampled passages are genuinely ambiguous** — clause 10's
  `sucks` vs `draw in`, clause 101 (all five readings compliant, all ten pairs still divergent on
  content), clause 1005's added `crucified`, clause 2789's OCR `l`/`I` — while clause 1003, with only
  one passing reading, has none: every divergence there is a defect or a contract. A row is classified
  by what the readings did *on that surface*, never by a verdict elsewhere: a keep-strategy that kept
  `<<` as told and failed `/` on another row is a contract divergence on `<<`, not a violation. No
  model is called; the build is byte-identical on rebuild. New: `work/translation_committee.py` +
  `work/test_translation_committee.py` (53 tests, including a suite-wide re-derivation of every row's
  classification from the facts).
- Consistency checker over the termbase, `analysis/consistency/` (#71) — five deterministic checks
  that a term's reading does not drift between the artifacts that carry it: #56's `GLOSS_TERMS`
  vs its own computed termbase row, #62's `PLAIN_READINGS` vs `GLOSS_TERMS` (one surface, one
  curated gloss, never two), the #30 entity glossary's role / does-done-to verbs vs the termbase's
  entity columns, `OPERATOR_GLOSSES` checked against itself for contradiction or duplication, and
  a global one-term-one-gloss sweep across every artifact. **0 violations** over 51 gloss terms,
  49 plain readings, 49 termbase terms, 98 entity-glossary rows and 10 operator readings. Found and
  fixed one real inconsistency while building the checker: `work/interlinear_edition.py` glossed
  `exoskeleta` at `high` confidence as a plain plural while every other body-machine compound in
  the termbase is `medium` — corrected. New: `work/consistency_checker.py` +
  `work/test_consistency_checker.py` (36 tests).
- Anti-pattern guardrails audit, `docs/guardrails.md` + `analysis/guardrails/` (#76) — a checklist
  pairing five named anti-patterns with the mechanism in the codebase that guards against each one,
  and a mechanical check per guardrail that reads the *published* artifact of the module it audits
  rather than re-deriving its numbers (a drift is a build error in the audit, not a silent
  disagreement). **Hallucinated coherence:** #73's triangulation report must keep disclosing
  divergent/partial rows alongside convergent ones — currently 37/45 motifs disagree, 6/45 fully
  convergent. **LLM circularity:** the verdict-computing functions in #63's glossing pipeline and
  #64's translation committee are parsed with `ast` (function-scoped, so a loader elsewhere in the
  same file that legitimately talks to a model backend cannot trip it) and must contain no token
  that would reach a model or re-invoke generation — 0 violations across the 10 functions checked.
  **Over-correction:** #6's lexical corrector's `REVIEW` class must stay non-empty, i.e. it must
  keep holding back genuinely ambiguous OOV tokens instead of "fixing" every one it sees —
  currently 15 corrected vs 200 held back. **Single-method certainty:** a convergence claim must be
  backed by at least three independent methods — #73 uses five (network community, LDA, NMF,
  BERTopic-style, KMeans). **Residue-dropping:** #70's coverage ledger must keep itemizing residue
  rather than claiming none exists, and report zero unexplained residue at every level it closes —
  11 residue classes itemized, 0 unexplained at all four levels. **All five guardrails currently
  pass.** New: `work/guardrail_audit.py` + `work/test_guardrail_audit.py` (27 tests: synthetic
  fixtures construct an all-convergent triangulation summary, a verdict function calling
  `opencode`, an emptied `REVIEW` class, a two-method convergence claim and an unexplained-residue
  ledger, and assert each one *fails* its guardrail before the artifact-backed suite pins today's
  passing result as the regression baseline).
- Predictive validity on held-out text, `analysis/heldout_validity/` (#72) — does the derived
  grammar predict held-out operator sequences and collocations, trained 80% / tested 20%? Two
  claims, each against its own honest baseline, pages as the split unit so no sequence crosses the
  boundary and the n-gram vocabulary is train-only (no test leakage into smoothing). *Operator
  sequences* — the #11/#35 interpolated n-gram Markov model (order 0/1/2, Jelinek-Mercer, per-page):
  on the conservative chronological split, order 2 **PREDICTS IN PROBABILITY** — perplexity 6.100
  vs the unigram baseline's 6.792 (a 10.2% gain) and top-2 accuracy rises 44.9%→50.8%, though argmax
  accuracy does not (30.4%→29.9%: better probability mass, not a better single guess) — and 99.3%
  of held-out run tokens use a run type seen in training. On the optimistic random split, order 2
  **overfits** (perplexity 6.323 worse than order 1's 6.228), so the random split only supports
  order 1. *Collocations* — the #27 PMI/logDice/Dice/t-score association table: rank is **STABLE**
  at the unit layer (Spearman logDice 0.8984) and **MODERATELY STABLE** at the operator-word layer
  (0.7080) — a collocation's strength in training predicts its strength in test — but top-100
  membership does not survive wholesale (overlap 47%/59%, train→test survival 48%/66%, the 4×
  smaller test split dropping many mid-count train pairs) with no systematic logDice level shift
  (mean |Δ| 0.86/0.94 on a 14-point scale). Consistent with #11's PARTIALLY IDIOMATIC verdict and
  #36's finite-memory, page-local verdict: the grammar generalises in probability and in rank, at
  short range, not as a system that names the one right continuation. New: `work/heldout_validity.py`
  + `work/test_heldout_validity.py` (86 tests).
- POS-tagging the silence-removed residue, `analysis/pos_tagging/` (#37) — tags the #13
  `v3_operators_to_space` plain stream (80,260 word tokens) and the #56 reference corpora's 15
  general-English novels (1,628,800 tokens) with a rule-based, auditable tagger (no NLTK/spaCy in
  this environment: a ~3,000-word lexicon + Brown-corpus verb forms, morphological suffix rules,
  bigram contextual disambiguation), the same tagger on both streams so the comparison is fair by
  construction. **Verdict: the residue is noun-heavy relative to normal English** — noun 42.62% vs
  23.0% (+19.62pp), verb 12.78% vs 21.35% (−8.57pp), noun/verb ratio 3.334 vs 1.077 (English prose
  runs 1.5–2.5), chi-square highly significant (χ²=10570.07, p<0.001). Three caveats disclosed: the
  tagger's absolute accuracy is unbenchmarked (only the cross-stream comparison is load-bearing),
  capitalized unknowns default to NNP (inflating proper nouns on neologisms like `ToKAGE`), and
  other unknowns default to NN (biasing both streams' noun counts upward together). New:
  `work/pos_tagging.py` + `work/test_pos_tagging.py` (29 tests).
- UMAP theme map of the book, `analysis/umap_theme_map/` (#42) — projects the #39 SBERT clause
  embeddings (5,614 clauses, 384-d) with UMAP (primary, cosine) and t-SNE (comparison), coloured by
  page and by #40's twelve theme clusters (exact partition match). **Verdict: no narrative drift, a
  real but local loop.** The page-coloured map is one intermixed cloud (consecutive-step/random
  ratio 0.995; page explains 5.4% of variance; first-/second-half centroid cosine 0.9879) — the book
  is stationary. The lag-return curve dips significantly below a shuffle null at lags 2–30 (minimum
  at lag 5, z=−8.2), gone by lag ~400: the "devastating loop" is template recurrence at the scale of
  a few clauses, not a page-scale cycle — restates #51's no-fixed-periodicity verdict in semantic
  space. The theme-coloured map shows #40's twelve motif regions as neighbourhoods of the same
  cloud, not separate islands (silhouette 0.033, #40's own number). Needs `umap-learn` (the `marker`
  conda env). New: `work/umap_theme_map.py` + `work/test_umap_theme_map.py` (39 tests).
- Contrastive semantic axes, `analysis/semantic_axes/` (#43) — the four axes the task names
  (`organic<->machine`, `self<->other`, `body<->information`, `sacred<->profane`) are built as
  directions **in #39's own clause-vector space** and tested four ways each: seed-pair coherence
  against a 200-draw **randomized-axis control** (random pole assignments over the same term
  pool), the +pole vs −pole clause projection contrast (word-boundary match on the `units` text;
  overlap disclosed), and the spread of #40's twelve centroids along the axis against a
  200-direction random null. Seed words are encoded with the pinned #39 SBERT encoder — the
  module's only encoder call, disclosed, because a word contrast and a clause vector must share
  one space for the projection test to mean anything (#28's static embeddings are a different
  space and are never mixed in). The partition is #40's exactly (k=12, `n_init=10`, seed 0;
  catalogue sizes *and* medoids reproduced). **Verdict: all four axes are real, in a stable
  strength order — `organic<->machine` strongest, `self<->other` weakest.** Every axis beats both
  controls (coherence z=+2.73…+3.19; centroid-spread z=+3.97…+17.43) and every clause contrast is
  positive (d=+0.88…+1.60). `organic<->machine` spread 0.149 (cluster 10 `cadaver·flesh·city`
  vs cluster 9 `i·lt·l`), `body<->information` 0.104, `sacred<->profane` 0.081 (cluster 1
  `adam·doll·retro` vs cluster 5 `murder·assassin·entrails`), `self<->other` 0.058 — the last
  because the book has a thick first-person register and almost no lexicalised *other* (`other`
  2 clauses, `another` 1), so the axis is carried by one pronoun-heavy cluster instead of two
  opposed vocabularies. The all-pairs **66 centroid-difference dimensions** show the book's
  widest gaps all oppose the first-person cluster 9 (10 of the top 12 pairs, self/other pronoun
  loadings): the strongest semantic dimension is **first person vs everything else** — the
  self<->other axis measured from the other side. All 29 seed pairs cohere positively (min
  +0.30 `cherry−hell`, max +0.69 `flesh−code`). Consistent with #40's continuum (silhouette
  0.033): the axes are directions that separate centroids, not partitions of clauses. New:
  `work/semantic_axes.py` + `work/test_semantic_axes.py` (35 tests).
- Formal grammar induction and complexity, `analysis/grammar_induction/` (#36) — is the
  operator/token stream regular, context-free, or does it need context? Four independent
  instruments, each isolating one axis of the Chomsky hierarchy, over the #10/#11 ordered operator
  stream (10,949 tokens, V=11), with the #17/#35 screened unit stream (74,077, V=5,498) as a
  token-level contrast and the #47 English `punct_core` channel (matched N) as the baseline.
  **Verdict: FINITE-MEMORY, PAGE-APERIODIC CODE CHANNEL** — regular-like in memory, non-regular
  (drifting) in state space; no context-free and no context-sensitive evidence; neither i.i.d.
  noise nor a copy loop. Subword complexity p(L) sits below an i.i.d. null at every L but is
  reproduced to within ~1% by a page-order-shuffle null (the structure is page-local, not a global
  grammar) and does not saturate (0.92 of ceiling at L=12, still rising — not a small fixed
  automaton). The held-out entropy-rate curve plateaus at order 1–2 (2.7597→2.7089 bits) — finite
  memory. MI(1)=0.1305 bits is 20× the i.i.d. null (permutation p=0.000) — real structure, not
  noise. LZ78 reuses only 1 of 2,433 phrases — not a copy loop. The `<<`/`>>` pairing is local and
  shallow (depth ≤ 4 for 225/230 pairs) — no stack, no context-free signature. **The spread between
  three state-count estimators is the finding:** ~98 predictive states, ≥325 Nerode classes (a
  growing lower bound), 2,433 LZ78 phrases — small enough to be predictable, too large and
  drifting to be a small fixed grammar. Reproduces #12/#14's delimiter pairing exactly (230
  matched / 182 unclosed / 156 stray). New: `work/grammar_induction.py` +
  `work/test_grammar_induction.py` (35 tests).
- Translations as an interpretive oracle, `analysis/translation_oracle/` (#58) — L7's
  "interpretive oracle", run as both a **documented negative result** and a working alignment.
  *Negative:* the A. Skobin Russian edition could not be obtained lawfully, and that is the
  finding rather than a failure to finish — 21 candidate sources probed and pinned
  (`sources.tsv`), the edition being a commercial 2006 АСТ/Адаптек printing (ISBN 5-17-037280-9)
  that FantLab catalogues but serves no text of, while the only full texts on the open web are
  eight unauthorized mirrors that the ingestion gate refuses after a **HEAD-only** probe (no
  mirror body is ever requested); the nettime thread named in the methods inventory is recorded
  as searched-not-found, not dropped. *Positive:* the signal was obtainable from material that
  is lawfully quotable — twelve short attributed quotations of the translation that Russian
  readers publish inside their FantLab reviews — and **the Russian edition preserves the book's
  operator channel verbatim**, which makes the symbol stream a translation-invariant key: two long
  quotations are located by **operator skeleton alone, with no lexical anchor**, at **100%
  positional operator retention** (RUQ01 16/16, its sequence unique among the corpus's 10,145
  operator tokens; RUQ02 12/12, matching twice at byte-identical sites pp. 5 and 221 — a finding
  about the book's loop rather than the aligner). 11 of 12 quotations resolve to a single English
  text; the bilingual lexicon publishes **17 attested** EN↔RU pairs (`tomb-device` →
  «могильное устройство», `boyroid` → «бойроида», `Cadaver City` → «Город Трупов») graded against
  the anchor priors that produced them, and one **divergence** is flagged rather than resolved
  (the reader's «ЛИКВИДАЦИЯ» against the English `abolition`). The aligner is a monotone DP with a
  bounded n-m diagonal, scored by standard AER (Gale & Church / Levow: deletions + insertions
  of null-collapsed links over the gold links of the same split — an error rate per gold link, so
  it is not bounded above by 1) against a built-in 24-pair EN–RU gold with planted
  1-n / n-1 operations — weights fitted on a **tuning half**, reported on a **held-out half**
  (**0.5385 against a length-only proportional baseline's 1.8205 on that same held-out half**;
  0.4051 vs 1.9114 over all twelve documents). The author's own lexicon is tested
  against the book: the two nouns the interviewer names (`ADAM` 509, `dogs` 495) are
  corroborated, the essayistic vocabulary around them (`hardweb`, `creature intensity`,
  `echo archive`) is **absent from the book**, and the two that do occur (`gene-dub`,
  `Biocapture`) occur once each, both as chapter headings — a claim about the interviews, not
  about the book. New: `work/translation_oracle.py` + `work/test_translation_oracle.py` (80
  tests). Offline and deterministic: the acquisition evidence is a pinned record of a dated probe
  run, and `--probe` re-runs the HTTP requests and diffs them against the pins.
- Coverage accounting / residue ledger, `analysis/coverage_ledger/` (#70) — one ledger over the
  whole corpus at four levels, each forced to close: **480,962 page-body characters** (390,469 word /
  17,269 operator / 1,735 punctuation / 71,021 whitespace / 468 layout scaffolding, **0 unexplained**),
  **90,220 tokens** (#14's 77,485 Lex nodes, #17's 11,000 operators), **11,000 operator tokens**
  each with exactly one #17 role (2,816 seams, 7,247 segment operators, 937 unbound), and
  **74,665 unit occurrences** (74,318 with a #19 category + 347 in #19's markup ledger; `5,868 + 8 =
  5,876` types). The **coverage metric** is the share of a level with no unexplained residue — 1.000000
  at all four levels, and a build failure if it ever drops. The residue ledger publishes 11 classes of
  awkward residue with owners, pages and page-cited examples, two of which no artifact itemized: the
  **13 book-text occurrences deleted by #27's tag-run screen** (the OCR wrapped printed text in
  `<a href="...">` anchors on pp. 158 and 219, so the 241 "tag-furniture" occurrences #27 publishes
  are 13 text occurrences larger than that description) and the **2 unit surfaces #17 and #19 store
  with a different number of backslashes** (`eruption=\njunk` p102, `device++\nmass` p209), which a
  surface-level join across the two artifacts loses. 14 downstream layers' kept/dropped splits are
  checked in one place, and #47's 480,309-char stream is reconciled to the raw 480,962 (comments →
  space, 197 heading-marker chars stripped). New: `work/coverage_ledger.py` +
  `work/test_coverage_ledger.py` (42 tests, including planted-residue tests that prove the coverage
  detector fires).
- Within-text language viability assessment, `analysis/language_viability/` (#56, revised scope) —
  six pre-registered held-out checks plus a negative control, scored against within-text nulls
  (within-page / within-run / within-clause / within-unit shuffle and a vocabulary bijection) with
  add-one empirical p-values. Verdict **VIABLE (BOUNDED)**, 6 of 6, with an explicit bound: the
  order sweep shows predictability stops at the bigram, so the recoverable language is local
  (triangulating #11, #47 and #51). Intrinsic evidence only — no reference corpus consulted or
  acquired; the earlier keyness output is retained as history and #78–#80 stay parked.
- Interlinear gloss + termbase (`termbase_sample.tsv`, `gloss_passages.md`, `gloss_tokens.tsv`) —
  49 terms with computed evidence and a confidence level, and three page-cited passages glossed
  token by token with the #13 v3/v4 operator-normalized renderings.
- Interlinear edition generator (#62) — `analysis/interlinear/` + `work/interlinear_edition.py` +
  `work/test_interlinear_edition.py` (49 tests). The plan's E8.2 three-line rendering — original
  clause | operator-normalized (silence removed) | plain-English gloss — for all 5,614 #33 clauses
  on all 225 pages. Line 2 is #13's v4 transform (operators → `.`, the variant #50 scored closest
  to the underlying system) rendered with #56's `render_variant` semantics, pinned by two build-time
  invariants: the token-assembled rendering equals the substring-deleted rendering for every clause,
  and every kept token's span reads back as its surface. Line 3 substitutes a plain reading only
  where the surface is not ordinary English, so the book's English layer glosses to itself (99.5%
  identity) and the interpretation concentrates at the non-English sites: 49 of the 51 #19
  `true_coinage` types are read from the #26 KWIC contexts, #19 morphology and #30 roles under #56's
  confidence scheme; named constructs keep their surface (names are not translated) with the #30
  role quoted; #19's `ocr_error` class is kept as printed, never auto-corrected; the two opaque
  coinages (`omanko`, `laple`) are marked `‹…›` on the line and itemized in `residue.tsv as reading
  leads for #63. No LLM and no external corpus. Reconciles with #33 exactly on the word layer
  (76,897 = 76,897 `n_words_clean`), with #17 on the token stream (90,220 / 11,000 operators) and
  with #14 on the slash roles (952/540); the 141-token gap to #33's event-counted `n_tokens` is
  disclosed as #14's synthesized marker events, and the 5,914 tokens in no #33 clause are counted
  rather than dropped. Deterministic: a rebuild is byte-identical except the report's generated
  stamp.

### Fixed
- Legal abstract reference corpus evaluated and curated; the historical legal register labelled exactly (#79, follow-up from #56) — the technical-register slot the plan names as "medical/CS/legal **abstracts**" (`docs/meaning-extraction-methods.md` L7.1) had been filled by five public-domain **full texts** written 1215-1850 (the Federalist Papers, the Declaration of Independence, Magna Carta, Bastiat's *The Law*, Rousseau's *The Social Contract*), and `analysis/keyness/keyness_report.md` then described that register as "modern ... legal writing" — a claim exceeding the reference set. Both registers are kept and both are labelled exactly: `legal` is the **historical** legal/political pole (a pinned `LEGAL_REGISTER_LABEL` in `work/reference_corpora.py`, the manifest's `public domain` licence cells, a register note in the report, and tests that fail if any of them drifts), and the new `legal_abstracts` register is the **modern** legal one. The value question was answered by measurement before the decision: scoring a 1,000-record DOAJ sample on the same harness leaves the 32-word reference-independent core **unchanged** and moves the reference-dependent residue 13 → **8**, because `discharge`, `exceeded`, `receives`, `reverse`, `rotation` are ordinary modern legal-register words (discharge of a contract, appellate reversal, a model code, exceeding an authority) that 18th-19th-century legal/political prose does not use — so the residue verdict now rests on three *modern* technical registers (`medical`, `cs`, `legal_abstracts`) instead of two modern ones plus a historical text, and the decision, the numbers and the alternatives (court opinions as the case-law candidate, subscription services, SSRN, the broader DOAJ subject query, a CC-BY-only subset) are recorded in `docs/legal-corpus.md`. The register is a pinned snapshot of `bibjson.journal.title:(law) AND bibjson.journal.language:EN` from DOAJ — 922 records / 920 distinct DOIs, 60 journals, 2020-2026, **249,872 tokens / 17,402 types / 969 key words at q < 0.05**, well above the 10,000-token inferential floor — with the same snapshot pinning and drift reporting as #78's medical corpus: the query is recency-sorted, so the stable unit is the committed bytes and their observed checksum. DOAJ editorial stubs and non-English abstracts are excluded by a 30-word abstract gate (part of the selection rule, so they appear in neither file), and the bilingual-journal leak the gate cannot see — Indonesian function words from EN/ID journals, ~3% of the register's tokens, visible in the anti-key table — is disclosed in the memo rather than filtered by an undisclosed screen. Rights are resolved per record and never asserted: every record's journal licence is looked up in DOAJ's journal API (batched 50 ISSNs per request through the module's retrying `download` helper) and written to `analysis/reference_corpora/legal_abstracts/rights.tsv`, one row per DOI with journal, publisher, year, language, ISSNs, the verbatim licence, a derived class (`open_access` / `free_to_read` / `unverified`), the verification route and date; the snapshot verifies **920 open access** (525 `cc by`, 171 `cc by-nc`, 156 `cc by-nc-nd`, 54 `cc by-sa`, 14 `cc by-nc-sa`, 2 `cc by-nd`), 0 unverified. The corpus file's per-record headers carry DOI + ISSNs, so a cached corpus re-resolves its licences from its own bytes (the medical pattern): a ledger covering the corpus's DOI set is kept, a stale or missing one is re-resolved in download mode and warned about, network-free, under `--no-download`. The manifest's licence cell is the policy statement plus the pointer to the ledger, never a blanket claim. Scope is bounded deliberately: the register is wired into the keyness harness where the plan placed legal abstracts, while #57's intertext arm set, #60's genre dossier and #37's POS comparison are unchanged and their published numbers stand (only #37's corpus-file inventory moves 29 → 30, from re-running its offline build over the new directory). Regenerated artifacts: `analysis/keyness/` (the 8th table, the register note, the residue verdict) and #57's `sources.tsv` / `availability_report.md` (a note on the new register, no new matcher arm). Tests 57 → **86** (`work/test_reference_corpora.py`: the historical label, DOAJ record parsing and the stub gate, header round-trip, licence-class derivation and batching, non-JSON and missing-journal handling, ledger round-trip and DOI-set coverage, the four build-wiring modes (covering ledger needs no network; a stale ledger warns network-free under `--no-download` and re-resolves in download mode, from the builder and from the cache-only orchestrator), and tree-level checks that the built ledger covers the snapshot and names only CC licences) and 43 → **45** (`work/test_keyness.py`: the modern-register-decides-residue behaviour, and the report's register labels). All offline; the live corpora are read from the committed cache.
- Medical abstract reference corpus curated as a rights-verified snapshot (#78, follow-up from
  #56's review) — the `medical` register was labelled **"open access (PubMed abstracts)"** in
  `analysis/reference_corpora/manifest.tsv`, in #57's acquisition ledger and in the README. The
  label is false for most of the corpus: PubMed *indexes* publisher-copyrighted abstracts (the
  efetch text itself carries `Copyright © 2026 Elsevier Inc. All rights reserved.` lines), and open
  access is a **per-record** fact. The corpus is **retained, and the decision is recorded with its
  evidence** (`docs/medical-corpus.md`): it is one of the four inferential registers in #56's
  keyness (380,072 tokens / 25,536 types, so its agreement enters the 32-word
  reference-independent core and it is one of the corpora that decide the 13 reference-dependent
  residue words), and in #60's genre dossier it is the pole that makes two verdicts readable —
  `clinical` 7.02/1k in the book against 12.35/1k in medical abstracts and 0.155/1k in prose,
  `virus` 8.15 against 9.87 and 0.058. **The correction is verification, not removal or
  restriction.** `work/reference_corpora.py` now extracts the snapshot's PMIDs (`parse_pmids`) and
  verifies each one against Europe PMC's core record (`license`, `isOpenAccess`, `inEPMC`,
  `inPMC`, `hasPDF`; 200 ids per request through the module's retrying `download` helper), writing
  `analysis/reference_corpora/medical/rights.tsv` — one row per record with a derived class
  (`open_access` when a CC licence is recorded, `free_full_text` for a PMC deposit without one,
  `publisher_copyright`, and `unverified` for an id Europe PMC does not return, which is never
  upgraded into a right). The pinned snapshot verifies **602 open access** (296 `cc by`, 183
  `cc by-nc`, 122 `cc by-nc-nd`, 1 `cc0`), **21 free full text**, **376 publisher copyright** and
  **1 unverified** — the old label claimed 1,000. The one unverified id is disclosed rather than
  dropped: `PMID: 42832289181` is a corrupt id string in the snapshot (PubMed's text output glued
  record 181's citation header onto record 180's PMID line); the underlying record is DOI
  `10.1097/MEG.0000000000003269` = PMID 42832289, which is publisher copyright anyway, so the
  conservative treatment costs nothing analytically. **Snapshot stability is documented, not
  assumed.** The query (`disease[Title]`, retmax=1000, sort=pub_date) is recency-sorted, so
  re-running it returns a *different* sample; the stable unit is the pinned snapshot — #80's
  manifest pin and drift mechanism are unchanged, and a changed snapshot re-verifies its rights
  automatically (a fresh fetch re-verifies every record; a cached corpus keeps its ledger when it
  covers exactly its PMIDs; a stale or missing ledger is re-verified in download mode and warned
  about, network-free, under `--no-download`). The manifest's `medical` licence cell, #57's
  `register_medical` row (new `rights_verified_per_record` authorization class) and the README
  table now state the per-record position; `docs/medical-corpus.md` records the decision, the
  record selection, the reuse terms (internal comparison statistics, quotations attributed to
  journal + PMID, no redistribution of the abstract files) and the alternatives considered (a
  PMC-OA-only subset — cleaner rights by construction, but biased toward CC-licensed,
  NIH-funded, PMC-deposited records — rejected as the main corpus and available by filtering
  `rights.tsv`). Tests 42 → **57**, all offline: PMID parsing, class derivation, 200-id batching,
  non-JSON and missing-record handling, ledger round-trip and exact-coverage checks, the three
  build-wiring modes (covering ledger needs no network; stale ledger warns under `--no-download`
  and re-verifies in download mode), and tree-level checks that the built ledger covers the
  snapshot exactly, contradicts a blanket open-access label, and no longer appears in the manifest.
  Rebuilt cache-only: every corpus hash unchanged; only the manifest's `medical` licence cell and
  the #57 `sources.tsv` `register_medical` row differ. Downstream manifest consumers
  (`test_keyness`, `test_intertext`, `test_genre_prior`, `test_reading_compression`,
  `test_frequency_zipf`) stay green.
- Seam-loss evaluation reconciled to one source of truth (#77, follow-up to #17's review) — the
  result comment posted on #17 scored the fused scope with a **page-level bag of baseline token
  strings** instead of the offset-exact span rule the spec, `seam_loss.tsv` and the build all use,
  so the record contradicted itself (`ws` 83.5%/22.2%/2.0% vs 77.6%/0.6%/0.1%; `sklearn` parts
  90.3% vs 99.6%; `wordpunct` operator tokens 97.7% vs 76.6%). Regenerating from the current code
  reproduces the published artifacts byte-for-byte, which settles which side was right: the bag
  rule is *looser* for the case-preserving baselines (a same-page string elsewhere stands in for
  the item, so `ws` earns 323 fused units and 2,400 parts it never emitted as tokens) and *too
  harsh* for the real `CountVectorizer`, which lower-cases, so `Soul/gram` is simply absent from
  its token set — 90.3% of fused parts against the span rule's 99.6%. The ranking, and therefore
  the finding (each standard tokenizer keeps ≤ 2 of the three views; this one keeps all three from
  one pass), is unchanged; only the magnitudes were wrong, and they flattered `ws` and `sklearn`.
  `analysis/tokenizer/seam_loss.tsv` now carries the scopes the spec advertised — 56 rows, one
  per `(scope, baseline)`, with a `scope` column (`overall` / `fused` / `class`) and a `class`
  column: `fused` is the missing headline scope (5,433 units / 11,131 parts / 2,816 seams) and
  `class` adds the per-class word-part and seam counts beside the unit counts. Book-level columns
  (`n_tokens`, `ops_*`) are filled on the `overall` rows only, and the build asserts that the
  `class` rows partition the book and `fused` + `plain` equals `overall` on all three numerators
  and denominators. The corrected table is reposted on #17, §4.1 of `tokenizer_spec.md` now names
  the file as the machine-readable home of both tables, §2.2 documents the `tokens.jsonl` `gap`
  sparsity, and the suite (97 → **103** tests) pins the three scopes, the reconciliation, and a
  cell-by-cell comparison of both spec tables against the file — the exact failure mode of this
  follow-up.
- Corpus-cache reproducibility and normalization in `work/reference_corpora.py` (#80, follow-up from
  #56's review) — three fixes. **(1) `--no-download` is now genuinely cache-only for every source.**
  It used to skip only the live PubMed / arXiv queries while `build_general_english` and
  `build_legal` still fetched any missing Gutenberg text over the network; both builders now take
  `allow_download` and, cache-only, *skip a missing text with a warning instead of fetching it*, so
  the flag can never be surprised by a network call (the eight pinned baselines remain a local file
  copy from #24's cache; a missing #24 cache is still a hard error naming the fix). **(2) The
  manifest's SHA-256 semantics are declared and enforced.** The docstring used to claim the live
  corpora were "pinned by content hash" while the manifest was rewritten with whatever bytes a build
  happened to produce — a claim with no mechanism behind it. The manifest now carries a `hash_role`
  column: `enforced_pin` for the eight #24 baselines (verified against the constants in the script; a
  mismatch aborts the build, never a silent substitution) and `observed_checksum` for everything
  else. For the observed rows, the previously published manifest's value is treated as the file's
  expected pin, and a rebuild that yields different bytes is **reported as drift** — logged and
  persisted to `manifest_drift.tsv` beside the manifest, expected value preserved — *before* the
  manifest is rewritten; a file that vanishes from a build is warned about rather than dropped
  silently. A clean rebuild clears a stale drift report, and a fresh clone (no prior manifest)
  records observations without crying drift. **(3) arXiv Atom summaries are entity-decoded before
  tokenization.** `parse_arxiv` now runs each `<summary>` body through `html.unescape` *after*
  extraction (so a decoded `&lt;` cannot re-enter the markup the regex matches): left raw, `&amp;`,
  `&#39;`, `&rsquo;` & co. reached the #24 word tokenizer as fake word tokens (`amp`, `rsquo`,
  `quot`) and inflated the counts. Rebuilt the corpora cache-only — no network touched, every corpus
  byte-identical, only the manifest gains the `hash_role` column. Tests 27 → **42** in
  `work/test_reference_corpora.py`, all offline: entity decoding (named + numeric refs, no token
  leakage, decode-after-extract), enforced pins (a tampered or missing #24 cache aborts), cache-only
  mode (a planted `download` that raises on any call proves the whole `ensure_corpora` build is
  network-free, missing texts are skipped with warnings, download mode still fetches, and a second
  rebuild is byte-identical), and drift (reported + persisted, clean rebuild clears the stale
  report, new files are not drift). Downstream manifest consumers (`test_keyness`, `test_intertext`,
  `test_reading_compression`, `test_frequency_zipf`) read named columns and stay green.
- Pre-registered agreement metric in the reader study (#75) — the generated `protocol.md` registered
  Krippendorff's alpha with the **ordinal** difference function while the pilot computed, and the
  report disclosed, the **interval** metric, so every published alpha was a violation of its own
  pre-registration. The choice is load-bearing, not cosmetic: recomputing the pilot's `felt_fidelity`
  alpha from the published ratings gives **0.7898** (interval — the published number) against
  **0.7117** (ordinal) and **0.3449** (nominal). `PRIMARY_METRIC` is now the single source for the
  protocol text, the report text and `krippendorff_alpha`'s default parameter, so the three cannot
  drift apart again, and two new tests pin the protocol to the metric the pilot uses — one checks
  the default, the other recomputes the published alpha from `pilot_ratings.tsv` and fails if the
  registered metric is not the one that produced it. Suite 60 → **62** tests; rebuild byte-identical.
- Quotation provenance in the translation oracle (#58) — **six of the twelve Russian
  quotations were credited to a man who did not write them.** RUQ05, RUQ06, RUQ08, RUQ09, RUQ10 and
  RUQ12 were attributed to "Reza Negarestani (FantLab review, 2011)". His name does appear on the
  FantLab work page, but in the «Рецензии» bibliography block, as the author of an *external review
  of the book* («Пишущие машины из далёких мест», work1915318) — a page that carries no review body
  and none of the six strings. The quotations were in fact transcribed from three reader reviews:
  **Stirliz77, 9 Oct 2020** (`#response409177`) for RUQ05/08/09/10, and **ekidonaa, 12 Feb 2025**
  (`#response498807`) for RUQ06/12. This was not cosmetic: these rows are short quotations of a
  *copyrighted* translation, so the attribution is precisely what makes them lawfully quotable, and
  the module's whole ingestion position rests on it. Corrected by removing the possibility rather
  than the strings — each quotation now names a `review` block and its attribution is **derived**
  from a pinned review table (`fantlab_reviews.tsv`: author, account, `datePublished`, the
  `#responseNNNNNN` permalink and a SHA-256 over the block's tag-stripped body), so a quotation
  cannot claim an author its block does not have. New `fantlab_reviews.tsv`; `ru_quotations.tsv`
  gains a `review` column; the alignment report prints the reviewer *and* the permalink and now
  explains the attribution rule. The three block fingerprints were verified identical across a
  cached fetch and an independent live re-fetch, and `--verify-reviews` re-checks all twelve
  quotations against the live page (all 12 OK). Tests 80 → **100**, including a
  planted-misattribution test that proves the provenance scan fires; re-injecting the original
  defect fails the suite in six independent places. Build remains byte-identical on rebuild.

## [0.1.0] — 2026-10-06

### Changed
- Evaluate and curate an optional legal abstract reference corpus (#79)
- Evaluate and curate an optional medical abstract reference corpus (#78)
- Reconcile #17 tokenizer evaluation numbers (single source of truth) (#77)
- Genre/prior dossier (#60)
- Burstiness and Heaps' law (#53)
- Analogy and metaphor mining (#45)
- Harden corpus cache reproducibility and text normalization (#80)
- Human jury / reader study design (#75)
- Back-translation harness (#74)
- Compression test for candidate readings (#68)
- Spectral reading artifact (#67)
- Seed entity roles from jacket synopsis, verify against text (#59)
- Intertextual n-gram matching vs Siratori corpus (#57)
- Self-similarity matrix and loop detection (#54)
- Operator-word mutual information (#52)
- Affect and valence curve (#44)
- UMAP theme map of the book (#42)
- POS-tagging the silence-removed residue (#37)
- Predictive validity on held-out text (#72)
- Formal grammar induction and complexity (#36)
- Anti-pattern guardrails audit (#76)
- Consistency checker (#71)
- Multi-agent translation committee with disagreement log (#64)
- LLM glossing pipeline (multi-strategy) (#63)
- E3 — Distributional semantics & semantic network (#23)
- E1 — Operator grammar & the symbolic machine (#9)
- E0 — Corpus ingestion & OCR stabilization (#2)
- Obtain and align translations (interpretive oracle) (#58)
- Narrative reconstruction (entity/event tracking) (#66)
- Interlinear edition generator (#62)
- Termbase and consistency enforcement (#65)
- Coverage accounting / residue ledger (#70)
- Narrative reconstruction / entity-event tracking (#66)
- Triangulation / convergence report (#73)
- Reference corpora + keyness harness (#56)
- Periodicity and spectral analysis (#51)
- Topic modelling (LDA/NMF/BERTopic) (#41)
- Topic modelling (LDA / NMF / BERTopic-style) + #29 community comparison (#41)
- Clustering to motifs and scenes (#40)
- Clause/sentence embeddings (#39)
- Clause/sentence embeddings: SBERT clause vectors + loader (#39)
- Frame and template mining (#34)
- Frame and template mining (#34)
- Quantified silence-removal experiment (#50)
- LM perplexity per page (#49)
- LM perplexity per page with GPT-2, matched English baseline (#49)
- Compression / MDL vs English baseline (gzip/zstd/bz2/lzma) (#48)
- Multi-level entropy analysis: char / token / operator + conditional orders 1-3 (#47)
- n-gram LM and per-token surprisal (#35)
- Clause segmentation via operators (#33)
- Entity and proper-noun induction (dramatis personae) (#30)
- Co-occurrence network and community detection (#29)
- Compound/portmanteau decomposition and affix lexicon (#18)

Initial commit. Full-book marker OCR, the L0–L9 extraction-method inventory, and the
E0–E4 pipeline (22 Chainlink tasks closed).

### Added

**Foundations**
- `docs/meaning-extraction-methods.md` — the L0–L9 method inventory.
- `docs/chainlink-plan.md` + `work/build_chainlink_plan.py` — the 77-issue program plan
  (1 program epic, 10 phase epics, 66 tasks).
- Full-book marker OCR (`docs/markdown/blood-electric/blood-electric.md`) and the first-pass
  analysis (`analysis/first_pass/`, `work/analyze.py`).

**E0 — Corpus ingestion & OCR stabilization**
- Map OCR confidence for the marker corpus (#3).
- Ensemble OCR with glyph voting (Tesseract / docTR / easyOCR vs marker) (#4).
- Inventory glyphs and freeze unicode normalization (#5).
- Build gazetteer-protected lexical correction (#6).
- Recover layout, paragraphs and section headers (#7).
- Build a page/folio map for citation (#8).

**E1 — Operator grammar & the symbolic machine**
- Operator inventory & frequency, incl. multi-operator runs (#10).
- Operator n-gram transition matrix and Markov model (#11).
- Operator function classification (#12).
- Silence-removal transform variants (#13).
- Operator grammar block parser / AST (#14).
- Rosetta mapping against real notations (#15).

**E2 — Lexicon & morphology**
- Operator-aware tokenizer (#17).
- Neologism detection → `neologisms.tsv` (#19).
- Validated morphological segmentation (#20).
- Numeric and symbol channel catalogue (#21).
- Case and emphasis inventory (#22).

**E3 — Distributional semantics & semantic network**
- Frequency, Zipf and type/token analysis (#24).
- KWIC concordancer for entities and terms (#26).
- Collocation and association (PMI/logDice) (#27).
- Static embeddings and analogies (#28).

**E4 — Syntax and structure without a parser**
- Parse-attempt and derangement curve per page (#32).

### Known issues
- **#18** Compound/portmanteau decomposition — closed, then reopened: `work/compound_decomposition.py:1817`
  raises `TypeError: 'int' object is not subscriptable`, so `build()` and its test suite fail.
  The artifact (`analysis/morphology/decompositions.tsv`) exists from an earlier build;
  reproducibility is pending.
