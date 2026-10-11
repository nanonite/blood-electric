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
