# Changelog

All notable changes in this workspace. The project decodes Kenji Siratori's *Blood Electric*
(2002) into an evidence-linked English reading. Chainlink issue numbers are given as `#N`.

## [0.1.0] — 2026-10-06

### Changed
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
