# Changelog

All notable changes in this workspace. The project decodes Kenji Siratori's *Blood Electric*
(2002) into an evidence-linked English reading. Chainlink issue numbers are given as `#N`.

## [Unreleased]

### Added
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
