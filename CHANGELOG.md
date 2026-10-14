# Changelog

All notable changes in this workspace. The project decodes Kenji Siratori's *Blood Electric*
(2002) into an evidence-linked English reading. Chainlink issue numbers are given as `#N`.

## [Unreleased]

### Added
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
