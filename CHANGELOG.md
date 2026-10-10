# Changelog

All notable changes in this workspace. The project decodes Kenji Siratori's *Blood Electric*
(2002) into an evidence-linked English reading. Chainlink issue numbers are given as `#N`.

## [Unreleased]

### Added
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

## [0.1.0] — 2026-10-06

### Changed
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
