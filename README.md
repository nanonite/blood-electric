# blood-electric-workspace

A research workspace for **decoding Kenji Siratori's *Blood Electric* (2002)** — a 224-page
"asemic cyberpunk" novel written in a fractured Meta-English of ordinary words, invented
neologisms, and code-like punctuation. The goal is not to force the book into ordinary prose
but to build a *transparent, evidence-linked translation into understandability*: to find out
where it is coming from.

## The idea in one paragraph

The book is not gibberish and not a foreign language — it is **English lexis under a symbolic
transform**. Where scholars of alien signals (Beguš, *The Next Theory of Evolution*, included in
`docs/pdfs/`) decoded sperm-whale codas by reframing them as language and removing the silences
to reveal latent structure, this workspace does the same to Siratori: characterize the transform,
strip it, and re-embed the residue in meaning-space.

## Layout

```
docs/
  pdfs/
    blood-electric.pdf                 # source scan, 225 pages
    eScholarship UC item 37t8k62b.pdf  # Beguš, "The Next Theory of Evolution" (methodological key)
  markdown/
    blood-electric/blood-electric.md   # marker OCR output (the working corpus)
  meaning-extraction-methods.md        # THE DELIVERABLE: every way to extract meaning (L0–L9)
analysis/                              # generated reports
  charset/                            # frozen unicode charset + normalization freeze (#5)
work/
  analyze.py                           # first-pass lexical/symbolic analysis (stdlib+numpy)
  run_analysis_when_ready.sh           # waits for OCR, then runs analyze.py
  normalize.py                         # unicode normalizer: rules pipeline + CLI (normalize/verify/freeze) (#5)
  test_normalize.py                    # 36-test freeze-consistency suite (#5)
  build_charset.py                     # regenerates analysis/charset/ artifacts (#5)
  silence_removal.py                   # builds analysis/silence_removal/ variant streams (#13)
  test_silence_removal.py              # 33-test regression suite (#13)
  operator_grammar.py                  # operator-grammar block parser + AST (#14)
  test_operator_grammar.py             # 52-test regression suite (#14)
  operator_tokenizer.py                # operator-aware tokenizer: units + parts + seams (#17)
  test_operator_tokenizer.py           # 97-test regression suite (#17)
  operator_word_mi.py                  # operator-word mutual information: the MI matrix over
                                       #   #29's lexical domains + per-domain G^2 tests + permutation
                                       #   nulls (#52)
  test_operator_word_mi.py             # 60-test regression suite (#52)
  compound_decomposition.py            # compound/portmanteau decomposition + affix lexicon (#18)
  test_compound_decomposition.py       # regression suite (#18)
  neologism_detection.py               # OOV unit screen: coinages / compounds / OCR errors (#19)
  test_neologism_detection.py          # 32-test regression suite (#19)
  collocation.py                       # PMI/logDice collocations + entity-verb pairs (#27)
  test_collocation.py                  # 28-test regression suite (#27)
  parse_attempt.py                     # dependency-parse attempt + derangement curve (#32)
  test_parse_attempt.py                # regression suite (#32)
  frame_mining.py                      # frame / template mining: n-gram skeletons + <<...>> names (#34)
  test_frame_mining.py                 # 35-test regression suite (#34)
  clause_embeddings.py                 # clause/sentence embeddings: SBERT vectors + loader (#39)
  test_clause_embeddings.py            # 28-test regression suite (#39)
  topic_modelling.py                   # LDA/NMF/BERTopic topics + coherence + #29 compare (#41)
  test_topic_modelling.py              # 38-test regression suite (#41)
  triangulation.py                     # convergence matrix over #29/#40/#41/#51 + report (#73)
  test_triangulation.py                # 63-test regression suite (#73)
  narrative_reconstruction.py          # entity/event tracking: who does what to whom,
                                       #   over the #14 AST + the jacket-synopsis check (#66)
  test_narrative_reconstruction.py     # 52-test regression suite (#66)
  embedding_models.py                  # static embeddings: w2v/fastText/GloVe (#28)
  test_embedding_models.py             # regression suite (#28)
  case_emphasis.py                     # ALL-CAPS inventory: headers / labels / screams (#22)
  test_case_emphasis.py                # 30-test regression suite (#22)
  frequency_zipf.py                    # type/token, hapax, Zipf vs English prose (#24)
  test_frequency_zipf.py               # regression suite (#24)
  reference_corpora.py                 # reference corpora for keyness: general English, cyberpunk,
                                       #   cut-up/Surrealist, Finnegans Wake, medical/CS/legal (#56)
  test_reference_corpora.py            # 26-test regression suite (#56)
  keyness.py                           # keyness vs each reference corpus: signed G^2 + BH q + log-ratio (#56)
  test_keyness.py                      # 43-test regression suite (#56)
  language_viability.py                # within-text viability: held-out checks + within-text nulls (#56)
  test_language_viability.py           # 50-test regression suite (#56)
  rosetta_mapping.py                   # Rosetta scorecard vs real notations (#15)
  test_rosetta_mapping.py              # regression suite (#15)
  translation_oracle.py                # translations as an interpretive oracle: acquisition ledger,
                                       #   EN-RU segment aligner + AER harness, skeleton alignment of
                                       #   Russian paratext quotations to the book (#58)
  test_translation_oracle.py           # 80-test regression suite (#58)
  glossing_pipeline.py                 # LLM glossing, five prompt strategies + four deterministic
                                       #   checks over the returned glosses (#63)
  test_glossing_pipeline.py            # 95-test regression suite (#63)
  translation_committee.py             # multi-agent committee over #63's glosses: generator /
                                       #   critic / comparer loop + disagreement map (#64)
  test_translation_committee.py        # 53-test regression suite (#64)
  consistency_checker.py               # same symbol/term, same reading: 5 checks over the
                                       #   termbase + glossary + operator readings (#71)
  test_consistency_checker.py          # 36-test regression suite (#71)
  spectral_reading.py                 # the v4 silence-removed stream published as a
                                       #   distinct under-text reading with a framing note
                                       #   + reconciliation against #13/#50 (#67)
  test_spectral_reading.py            # 57-test regression suite (#67)
  guardrail_audit.py                   # anti-pattern guardrails: hallucinated coherence, LLM
                                       #   circularity, over-correction, single-method certainty,
                                       #   residue-dropping — audited against #63/#64/#70/#73 (#76)
  test_guardrail_audit.py              # 27-test regression suite (#76)
  reader_study.py                      # human jury / reader study DESIGN: the instrument,
                                       #   the sampling frame, the counterbalanced packets,
                                       #   the pre-registered analysis and a simulated
                                       #   pilot that proves the pipeline works (#75)
  test_reader_study.py                 # 62-test regression suite for the study design (#75)
  grammar_induction.py                 # Chomsky-hierarchy classification of the operator stream:
                                       #   subword complexity, entropy-rate, MI vs lag, LZ78,
                                       #   three state-count estimators, stack/pairing test (#36)
  test_grammar_induction.py            # 35-test regression suite (#36)
  heldout_validity.py                  # predictive validity: #11/#35 n-gram model + #27 collocation
                                       #   table scored on a held-out 20% of pages, chrono + random
                                       #   splits (#72)
  test_heldout_validity.py             # 86-test regression suite (#72)
  pos_tagging.py                       # rule-based POS tagger (3 tiers: lexicon, suffix rules,
                                       #   bigram disambiguation); noun/verb balance, v3 residue vs
                                       #   15 general-English novels (#37)
  test_pos_tagging.py                  # 29-test regression suite (#37)
  umap_theme_map.py                    # UMAP/t-SNE projection of #39's clause embeddings, coloured
                                       #   by page + #40's theme clusters; drift/loop statistics
                                       #   (step ratio, neighbour page delta, lag-return) (#42)
  test_umap_theme_map.py               # 39-test regression suite (#42)
  semantic_axes.py                     # #43 contrastive semantic axes: the four named
                                       #   axes (organic/machine, self/other,
                                       #   body/information, sacred/profane) as seed-pair
                                       #   directions in #39's clause-vector space, with
                                       #   randomized-axis + random-direction controls
  test_semantic_axes.py                # 35-test regression suite (#43)
  self_similarity.py                   # #54 paragraph x paragraph cosine self-similarity
                                       #   matrix: lag curve + permutation null, band
                                       #   statistics, off-diagonal blocks, page rollup;
                                       #   English / lexical / view-encoder controls
  test_self_similarity.py              # 39-test regression suite (#54)
  entity_roles.py                      # #59 entity roles seeded from the jacket synopsis,
                                       #   each verified against the book text (role table
                                       #   with evidence / refutation / uncertainty)
  test_entity_roles.py                 # 47-test regression suite (#59)
  ensemble_ocr.py                      # multi-engine OCR: consensus + glyph-vote atlas (#4)
  test_ensemble_ocr.py                 # regression suite (#4)
  marker_full.log                      # OCR run log
```

## Status

- [x] Source acquired; corpus characterized (English lexis + operator syntax + neologisms).
- [x] Method inventory written: `docs/meaning-extraction-methods.md`.
- [x] **OCR conversion of all 225 pages** via `marker` (conda env `marker`) — 22 min, 0.17 pages/s.
- [x] First-pass statistical report: `analysis/first_pass/first_pass_report.md`.
- [x] **Chainlink plan built**: 1 program epic + 10 phase epics + 65 tasks.
- [x] Chainlink #5 — glyph inventory + unicode normalization freeze: `analysis/charset/` (charset.tsv, 148 codepoints) + `work/normalize.py`; measured no-op on this corpus.
- [x] Chainlink #7 — layout recovery: `analysis/layout_recovery/` (sections.tsv, layout md with `<!-- page N -->`).
- [x] Chainlink #8 — page/folio map: `analysis/page_folio_map/page_folio_map.tsv` (225 rows; detected 171 / inferred 40 / none 14 folios) + `work/build_page_folio_map.py`.
- [x] Chainlink #11 — operator n-gram/Markov model: `analysis/operator_ngrams/` (1st/2nd-order matrices, held-out perplexity 6.79→6.52 random-fold and 6.79→6.31 chronological, MI 0.13 bits; leave-out sensitivity: intra-run gain 19.8%→9.7% excluding HTML-artifact pages; verdict: partially idiomatic — shallow local run syntax, not a deep stream grammar) + `work/operator_ngrams.py` + regression test `work/test_operator_ngrams.py`.
- [x] Chainlink #12 — operator function classification: `analysis/operator_functions/` (classification_spec.md with per-operator evidence; `//` utterance break not line comment — 53% sentence-initial vs `::` 13%, z=31; `=` naming linker 1,747 `word=word` pairs; `<<`/`>>` paired blocks, 230 spans, median 5 words; `/` dual compound + sentence-final; `-` word-internal compound linker + 177 measured free-standing ` - ` dash separators; backtick escape rejected on absence, with `--`×474 / `->`×237 disclosed as layout-scaffolding-only in the raw file vs 0 in the page bodies) + `work/operator_functions.py` + regression test `work/test_operator_functions.py` (13 tests; full `work/` suite 57).
- [x] Chainlink #13 — silence-removal transform variants: `analysis/silence_removal/` — 7 page-tagged residue streams (`v0` baseline, `v1` delete-all, `v2` delete-only-`::,//`, `v3` map-to-space, `v4` map-to-period, plus `v5`/`v6` sentence-boundary diagnostics) + `variants.tsv`, `per_page_metrics.tsv`, `silence_sites.tsv`, `fused_residue.tsv` (operator-supplied decomposition dictionary), `sentence_shape.tsv`, `line_index.tsv`, `samples.tsv`, `silence_removal_report.md` + `work/silence_removal.py` + regression test `work/test_silence_removal.py` (33 tests; full `work/` suite 90). 11,645 silence sites (10,949 operators + 696 whitespace-bounded dashes); every variant's token count is arithmetically reconciled; `//`→period beats `::`→period on uppercase-initial segments 55.8% vs 17.3% (55.7% vs 16.5% length-controlled), independently replicating #12.
- [x] Chainlink #14 — operator grammar block parser + AST: `analysis/block_parser/` (`grammar_spec.md`, `ast.jsonl` — one JSON tree per page, 108,780 nodes, `nodes.tsv`, `ast_tree.txt`, `spans.tsv`, `clauses.tsv`, `records.tsv`, `links.tsv`, `diagnostics.tsv`, `boundary_eval.tsv`, `spot_checks.tsv` + trees, `summary.json`) + `work/operator_grammar.py` + regression test `work/test_operator_grammar.py` (54 tests; full `work/` suite 144). Grammar `span(<< >>) > utterance(//) > field(::) > segment(- / : . , ;) > atom`, single pass with one delimiter stack; losslessness asserted at build time (225/225 pages, 90,220 tokens). 432 spans (250 closed / 182 unclosed / 126 nested), 3,764 utterances, 6,640 fields; `<<…>>` covers 58.3% of word atoms. Reproduces #10's 10,949 operators, #12's 182/156/230/177 delimiter numbers and #13's 696 whitespace-bounded dash sites. New: 519 free-standing em/en dash separators, 20 headers carrying `<<…>>` names.
- [x] Chainlink #17 — operator-aware tokenizer (gateway to the whole E2 lexicon tier): `analysis/tokenizer/` (`tokenizer_spec.md` = grammar + evaluation, `tokens.tsv`/`tokens.jsonl` lossless stream, `page_tails.tsv` the page tail that closes the reconstruction, `units.tsv`/`unit_types.tsv`/`part_index.tsv` the unit-and-parts layer, `seam_loss.tsv`, `cases.tsv`/`case_examples.txt`, `named_constructs.tsv`, `diagnostics.tsv`, `reconciliation.tsv`, `token_stats.tsv`, `summary.json`) + `work/operator_tokenizer.py` + regression test `work/test_operator_tokenizer.py` (97 tests; full `work/` suite 241). Emits all three views in one pass — unit (`gene=TV`, `soul/gram`), parts (`gene`|`=`|`TV`, each part with its own offsets) and seam tokens: 90,220 tokens, 74,665 maximal units (5,876 types), 80,363 word parts, 2,816 seams, 2,882 dash joins; losslessness 225/225 pages (page body = `"".join(gap+text)` + `page_tails.tsv`); reproduces #10's 10,949 body operators + 51 header operators, #11's ordered operator stream operator-for-operator on all 225 pages, #14's 77,485 Lex nodes and 2,048/198/15/952 Link nodes + 540 Slash nodes, #13's 696 free dashes, and #12's four named-pair counts recomputed exactly (147/130/129/317). Seam loss, scored **offset-exact** (an item counts only if the baseline emits a token covering exactly its own span): `ws` keeps 77.6% of fused units but 0.6% of their parts; `wordpunct` keeps every part and every seam string but 0% of fused units; sklearn/`[A-Za-z]+` keep 0% of the operator layer. Also disclosed, not repaired: 229 units born inside surviving source markup (`</a>`×99 mailto anchors, `<th>`×79 page furniture, `</p>`×28 …), and the `.tsv` quote hazard (136 bare `"` tokens → read with `split("\t")`, not `csv.DictReader`).
- [x] Chainlink #19 — neologism detection: `analysis/neologism/` (`neologism_report.md`, `neologisms.tsv` — 2,305 candidate unit types, `screening.tsv` — all 5,868 screened types, `neologisms_lower.tsv` case-folded, `ocr_evidence.tsv`, `numeric_channel.tsv`, `markup_residue.tsv`, `diagnostics.tsv`, `reconciliation.tsv`, `summary.json`) + `work/neologism_detection.py` + regression test `work/test_neologism_detection.py` (32 tests). Screens the unit layer (join on `uid`), never words: 2,046 open-juncture types are invisible to a word-frequency list (1,292 carry a binding seam — #17's 1,293 less the markup unit `fluid=streaming`). Removes 347 HTML-tag-residue occurrences (229 strict #17 / 231 #18, plus 116 opening tag names `<a href=…>`, `<th colspan=…>`), so `th`/`p`/`pre` never enter as hapax words. Of 74,318 screened occurrences, 2,305 types are non-English: 2,216 compounds of English parts, 51 true coinages + 23 named coinages, 15 OCR errors (14 #6-confirmed + dropped-linker `cadaverfeti`→`cadaver-feti`), 18 numeric (handoff to #21); first attestation recorded per type.
- [x] Chainlink #22 — case and emphasis inventory: `analysis/case_emphasis/` (`case_emphasis_report.md`, `caps_occurrences.tsv`, `caps_types.tsv`, `section_headers.tsv`, `per_page.tsv`, `all_caps_units.tsv`, `diagnostics.tsv`, `reconciliation.tsv`, `summary.json`) + `work/case_emphasis.py` + regression test `work/test_case_emphasis.py` (30 tests). Screens the #17 *token* layer (case is a token property) and joins every occurrence to the unit layer on `uid`. **1,890 ALL-CAPS tokens / 141 types = 2.44 % of 77,485 words.** Class split: `header` 36 (in a #7 heading), `function` 453 (`I`/`A`, excluded), `label` 1,138, `scream` 263. **Verdict:** a designator channel, not a scream channel — the top 16 labels (`ADAM` 328, `TV` 173, `DNA` 138, `BABEL` 75, `HDD` 39, `VTR` 37, `SADO` 31 …) carry 88 % of label occurrences and recur on 100+ pages; a distinct machine-command register (`LOAD` 140, `PLAY` 38, `REC` 7, `OUTPUT` 5, `UPLOAD`/`DOWNLOAD` 4 …) sits beside hapax affective screams; 24 of 56 section titles are fully upper-case; the 229 HTML-tag-residue units contribute **0** all-caps tokens.
- [x] Chainlink #4 — ensemble OCR with glyph voting: `analysis/ensemble_ocr/` (`ensemble_ocr_report.md`, `consensus.tsv`/`consensus.md`, `disagreement_atlas.tsv`, `consensus_changes.tsv`, `confusion_pairs.tsv`, `category_counts.tsv`, `verdict_counts.tsv`, `page_disagreement.tsv`, `page_length.tsv`, `engine_agreement.tsv`, `focus_glyphs.tsv`, `doubled_operators.tsv`, `marker_reference.tsv`, `summary.json`, `engines/*.jsonl`) + `work/ensemble_ocr.py` + regression test `work/test_ensemble_ocr.py` (40 tests). Marker/surya (#3) is the reference; three independent engines re-OCR all 225 pages at 300 dpi — tesseract 5.3.4, docTR 1.0.1 (db_resnet50 + crnn_vgg16_bn) and easyocr 1.7.2 (CRAFT + CRNN; PaddleOCR is not installable here). 497,354 aligned columns; character-level vote with **absence as a vote**, so `::`→`:` is a deletion and an engine-only glyph an insertion; strict-majority consensus, ambiguous columns keep marker. Atlas localizes every non-unanimous column to page + block + line + char offset + word bbox; markup columns flagged. Verdict: marker is strongly confirmed (tesseract 97.9% char agreement, docTR 97.3%, easyocr 93.9%); the atlas exposes the ambiguous glyph families (`l`/`I`/`1`/`|`, `o`/`O`, `-`/`=` — 1,493 exact, easyocr reading `=` as `-`, `::`/`:` — 3,340 dropped-copy events, `<<`/`<`, backtick/apostrophe). **The ensemble also catches marker's own failure modes**: recognition loops on pages 89/208/182 (page 89 repeats *"of the second of"* ×145, ~2,000 surplus chars) and surviving HTML markup (3,160 events) are voted out of the consensus.
- [x] Chainlink #24 — frequency, Zipf and type/token analysis: `analysis/frequency/` (`frequency_report.md`, `fits.tsv`, `book_zipf.tsv`, `category_breakdown.tsv`, `hapax_ledger.tsv`, `english_baseline.tsv`, `reference_manifest.tsv`, `diagnostics.tsv`, `reconciliation.tsv`, `summary.json`, `zipf.png`, `type_token_curve.png`) + `work/frequency_zipf.py` + regression test `work/test_frequency_zipf.py` (23 tests). Screens the unit layer (74,318 occurrences / 5,868 types after the 347 markup removals; the uid join is asserted to reproduce #19's type-level screen exactly) against eight pinned public-domain English prose baselines at matched N=74,318. **Verdict:** the book is *more* repetitive than English prose (MATTR-5000 0.203 vs 0.285; top-10 share 36.6% vs 24.5%; `the` 14.2% vs 3.8–6.0%), i.e. a loop of a restricted English core — and its coinage is a separate minority channel (10.5% of tokens, 39% of types) with a shallow Zipf slope (−0.66 vs English −1.11) and Heaps β 0.835 (vs 0.637), but 73.8% of coinage tokens come from repeating types and only 15 types are OCR errors, so the heavy tail is structured coinage, not noise.
- [x] Chainlink #47 — multi-level entropy: `analysis/entropy/` (`entropy_report.md` — the deliverable, `entropy_by_level.tsv`, `char_entropy.tsv`, `token_entropy.tsv`, `operator_entropy.tsv`, `english_baseline.tsv`, `type_entropy.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`, `entropy_curve.png`) + `work/entropy_analysis.py` + regression test `work/test_entropy_analysis.py` (25 tests). Shannon entropy at the **char / token / operator** levels and the context-weighted conditional entropy `H(next|prev k)` for k=1–3, each reported as an in-sample plug-in estimate **and** a leakage-free leave-one-page-out cross-entropy (the #35 model); the book is compared to the eight pinned #24 English references at matched symbol N, and the operator stream to English punctuation. **Verdict:** the book is *more* predictable than matched English at the char and token levels (token unigram 8.12 vs 9.13 bits/word; letter order-3 held-out 1.97 vs 2.55 bits/char) but its **operator channel is the exception and, uniquely, does not gain from context** (2.76→2.80 bits, vs English `punct_core` 1.92→1.87) — the entropy-level confirmation of #11's "shallow local run syntax". Reproduces #11's operator stream operator-for-operator, #17's 74,077 screened units and #10's 10,949 operators.
- [x] Chainlink #48 — compression / MDL vs English baseline: `analysis/compression/` (`compression_report.md` — the deliverable incl. the method note, `compression_by_stream.tsv`, `english_baseline.tsv`, `mdl_split.tsv`, `silence_variants.tsv`, `per_page.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`, `compression_curve.png`) + `work/compression_mdl.py` + regression test `work/test_compression_mdl.py` (20 tests). Six compressors across four families (gzip/zlib, bz2, lzma, zstd-3/19), matched N, raw and letters-only alphabets, plus a length-robust MDL split and the #13 silence-removal variants. **Verdict — a negative result for the task's stated prior:** the book is *more* compressible than matched English prose (char gzip-9 2.14 vs 2.87 bits/char; letters 2.24 vs 3.07; token 12.7 vs 14.6 bits/word), stable across all six compressors and both alphabets, so "low compressibility = high alienness" is false here. Stripping the #13 operators makes the residue *even* more compressible, so the alienness is local (operator grammar #11, hapax tail #35), not aggregate; per-page compressibility is flat on body pages (sd 0.50 on mean 3.82) and only moderately correlated with #35 surprisal (r=0.68).
- [x] Chainlink #49 — LM perplexity per page: `analysis/lm_perplexity/` (`lm_perplexity_report.md` — the deliverable, `per_page.tsv` — per-page curve + joined #35/#48 columns, `english_baseline.tsv` — per-reference at matched N, `perplexity_by_instrument.tsv` — pooled book vs English across modes, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`, `perplexity_curve.png`) + `work/lm_perplexity.py` + regression test `work/test_lm_perplexity.py` (20 tests). **GPT-2** (124M, CPU, fixed seed/threads, `eval()`, no sampling) scores every page with a sliding 1024-token window (50 % stride, each token scored once), **streaming** (context across page breaks) as primary and **isolated** as a robustness check; the eight pinned #24 references are truncated to the same GPT-2 token N (116,411) and split into 225 pseudo-pages scored identically; the #35 n-gram LM is the disclosed fallback ("measured surrogate"). **Verdict — the neural external LM reverses the self-trained/compression direction:** the book is *less* predictable than matched English (streaming perplexity **104.2 vs 33.7**, 3.10×; isolated 191.9 vs 41.3; 6.70 vs 5.07 bits/token), exactly the L6.3 prior — while #35/#47/#48 (book-trained n-gram / entropy / compression) found it *more* predictable. The instruments do not conflict: a book-trained model can memorise the loop, GPT-2 encodes natural-English expectations the book violates; and they still **agree on which pages are hard** (Spearman rho 0.52). Per-page perplexity tracks the #35 hapax share (rho 0.43) and the #48 gzip bits/char (rho 0.71); the coldest pages are OCR recognition loops / surviving markup (`of the second of …`, `<a href="mailto:">`), disclosed and bounded by a `--screened` sensitivity. Reproduces #35's held-out perplexity exactly (99.66) on the surrogate path and #17's 74,077 screened units.
- [x] Chainlink #26 — KWIC concordancer for entities and terms: `analysis/kwic/` (`kwic_report.md`, `concordance.tsv` — 9,329 occurrences in ±5-token context, `term_summaries.tsv` — 2,341 terms with counts / attestation / surface variants / embedded-in-larger-unit, `summary.json`, `diagnostics.tsv`, `reconciliation.tsv`) + `work/kwic_concordancer.py` + regression test `work/test_kwic_concordancer.py` (20 tests). Term list = entities (#19 `proper_noun` + #22 `label` + #6 gazetteer `entity`) + suspicious terms (#19 `compound`/`true_coinage`/`ocr_error` + #21 numeric + #22 command register); #22 screams/headers/function excluded as ordinary English. Joins the #17 **unit** layer case-insensitively (never words, never token index — the `tokens.tsv` `unit` column is a per-page index, so context is located by a (page, start) containment join). **Verdict:** the concordance is the primary route to a term's stable sense — e.g. `TV` is only 7 standalone units but 175 word-parts inside compounds (`gene=TV` 142, `BABEL=TV` 16), and `ADAM` 303 + 25 token-embedded = 328 reconciles #22 exactly; two embedded views (part-level vs token-level) are both reported.
- [x] Chainlink #27 — collocation and association (PMI/logDice): `analysis/collocation/` (`collocation_report.md`, `collocations_bigram.tsv` — unit-layer bigrams with f_x/f_y/f_xy/N + PMI/PPMI/logDice/Dice/t-score, `collocations_trigram.tsv`, `collocations_bigram_token.tsv`/`collocations_trigram_token.tsv` — the token layer incl. operators, `entity_collocations.tsv`, `entity_verb_pairs.tsv`, `verb_lexicon.tsv`, `entity_inventory.tsv`, `summary.json`, `diagnostics.tsv`, `reconciliation.tsv`) + `work/collocation.py` + regression test `work/test_collocation.py` (28 tests; full `work/` suite 532, one disclosed pre-existing #18 error). Two sequences from #17 — the maximal **unit** layer (74,077 items after #19's 347 markup removals plus 241 more tag-furniture occurrences `href`/`mailto`/`colspan` the general tag-run detector catches) and the **token** layer (88,669 tokens incl. operators) — never token indices. **Verdict:** the content collocations are a tight motif set (`artificial sun` 367, `internal organ` 250, `drug embryo` 204, `cadaver city` 192, `ice nebula`/`nebula faecal` 179, `cold-blooded disease` 177, `sickly period` 149) over a function-word loop (`of the` 3,854); operator–word pairs are sharply selective (`::`→`the` 928, `//`→`the` 672, `/`→`gram` 326, `=`→`tv` 166, `=`→`serum` 160). Entity–verb: 85 entities / 1,869 occurrences / 1,848 verb pairs; `ToKAGE` does `is`/`was`/`fecundates` and is done to by `love`/`speed`/`like`/`crucified` — the same coinage verb `fecundates` on both sides. Verb flag is a disclosed *candidate filter* (10,242 Brown surface forms + 56 verb-diagnostic book forms + 275 ambiguous `-s` forms), not a parser.
- [x] Chainlink #28 — static embeddings and analogies: `analysis/embeddings/` (`embedding_report.md` — the deliverable, `neighbours_book.tsv` — every book type × 3 book models, `neighbours_selected.tsv` — 64 curated terms × 6 models, `analogies.tsv` — 103 probes × 6 models, `analogy_summary.tsv`, `analogies_english.tsv`, `vocabulary.tsv`, `model_summary.tsv`, `summary.json`, `diagnostics.tsv`, `reconciliation.tsv`, plus `w2v_book.kv`/`ft_book.kv`/`glove_book.npz` and the merged trio) + `work/embedding_models.py` + regression test `work/test_embedding_models.py` (26 tests). Six models (word2vec skip-gram, fastText subwords, and a self-contained numpy GloVe) trained on the #17 **unit** layer alone and on the unit layer + 1,997,635 English tokens (8 pinned Gutenberg texts + Brown). Contexts are the book's own utterances: 74,318 screened units over 2,345 `//`-delimited sequences; the #19 screen join is asserted (347 markup occurrences removed, no tag names as words). **Verdict:** fastText's subwords are the only way a coinage survives the merged corpus — w2v/GloVe cover 48.7% of the 5,502 book types at `min_count=5`, fastText 100.0%; on the book alone all three cover 100%. Pooled curated analogies (`X:Y::Z:?`) hit@10: `w2v_book` 40.7%, `glove_book` 37.0%, `ft_book` 25.0%, `glove_merged` 34.8% — the book-alone models recover the book's own naming/derivation equations, the merged models trade book signal for English neighbours, and English sanity analogies pass at 42.9% hit@10 (a small-corpus limit, reported plainly). Neighbours show `gene=tv`→`fecundates`/`boy-roid`, `tokage`→`vital=serum`, `adam`→`doll`/`anti-adam`, and fastText recovers the parts (`boy-roid`→`boy-roid=chaos`, `suck=blood`→`mobile=suck=blood`).
- [x] Chainlink #15 — Rosetta mapping against real notations: `analysis/rosetta/` (`rosetta_report.md` — the deliverable, `notation_signatures.tsv`, `operator_role_map.tsv`, `scores.tsv`, `per_operator_best_match.tsv`, `positional_diagnostics.tsv`, `structural_metrics.tsv`, `permutation.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/rosetta_mapping.py` + regression test `work/test_rosetta_mapping.py` (33 tests; full `work/` suite 408, one disclosed pre-existing #18 error). Scores the #10/#12 operator set against eight real notations in four frequency-weighted dimensions (role-symbol coverage 0.55, alphabet Jaccard 0.15, capability features 0.15, positional fidelity 0.15). **Verdict:** the operator *alphabet* is C-family — **C/C++** wins (total 0.75, role-symbol coverage 82.4%, exact symbol+role 64.9% of occurrences; its punctuation set contains 100% of the book's operator occurrences), then **Perl** (0.72) and **HTML/XML** (0.58); lowest **Regular expressions** (0.35). But the resemblance is **symbolic, not functional**: `//` is not a comment anywhere in the book (its closest analogue is the GenBank/EMBL `//` record terminator, role-compatible but line-anchored there vs inline here), `++` collides with C increment (coupling, never numerals), `<<`/`>>` are block quotes not I/O, and ordinary English punctuation — with **zero** symbol overlap — still expresses the *role* of 79.3% of operator occurrences. The book is **English punctuation roles encoded in C-family symbols**, i.e. a measured false friend of code. Permutation test (shuffled symbol→role, 2,000 shuffles) p=0.0005, z=3.1; winner stable under hyphens included, raw-HTML pages excluded, and rare operators dropped.
- [x] Chainlink #34 — frame and template mining: `analysis/frame_mining/` (`frame_mining_report.md` — the deliverable, `frames.tsv` — the frame catalogue, `family_counts.tsv`, `block_names.tsv` — recurring `<<...>>` names, `frame_coverage.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/frame_mining.py` + regression test `work/test_frame_mining.py` (35 tests; full `work/` suite 813). Clause token sequences are the #17 units (markup-screened) inside #33's 5,614 clauses, joined on `(page, start)`; a **frame** is an n-gram skeleton (n = 2..8) with one or two slots, qualified at ≥ 4 distinct clauses and ≥ 3 attested fillers/slot, with `n_instances` and filler counts recomputed **exactly** over every occurrence. **Verdict:** the loop grammar is a small NP-of-NP core — `of the ⟨_⟩` covers 2,184 clauses / 3,842 instances (top fillers `artificial` 258, `cadaver` 175, `drug` 161) — and the task's named schemata are present: `i ⟨_⟩ the` (140 clauses; `murder`/`invade`/`rape`/`eviscerate`), `the ⟨_⟩ that` (401), and the two-slot `the ⟨_⟩ ⟨_⟩ of the` (276) / `of the ⟨_⟩ of ⟨_⟩` (118); 675 frames (646 single / 29 two-slot), top 10 covering 61.0 % of clauses and top 40 67.9 %. `<<...>>` block names are the leading run of screened units (355 distinct; `entrails that radiate` 30 spans). Disclosed confound: the `second` filler (145–163) is the #4 OCR recognition loop, surfaced by exact counting.
- [x] Chainlink #39 — clause / sentence embeddings: `analysis/clause_embeddings/` (`clause_embeddings_report.md` — the deliverable, `clause_embeddings.npz` — primary vectors `clause_ids`/`pages`/`emb_units`/`emb_raw` (float32, L2-normalised), `clause_embeddings_bge.npz` — secondary encoder, `clause_index.tsv` — keys + both text views + token counts, `model_summary.tsv`, `neighbours.tsv` — curated motifs → nearest clauses, `view_agreement.tsv`, `embedding_stats.tsv`, `english_baseline.tsv`, `per_page.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/clause_embeddings.py` + regression test `work/test_clause_embeddings.py` (28 tests; full `work/` suite 841). Embeds every #33 clause with **SBERT** `sentence-transformers/all-MiniLM-L6-v2` (mean pooling + L2 normalise, 384-d; exact per the model's `1_Pooling/config.json`, reimplemented on `transformers` so no `sentence_transformers` install is needed); E5 (`intfloat/e5-small-v2`) is in the registry and the cached `TaylorAI/bge-micro-v2` is the secondary robustness encoder. Two text views: `units` (markup-screened #17 unit surfaces, joined on `(page,start)`; reconciles with #27's 74,077) and `raw` (source substring, operators intact). A `ClauseEmbeddingStore` loader returns vectors/texts/neighbours by `clause_id`/`page`. **Verdict:** the book's clauses are *more self-similar than matched English* — mean nearest-neighbour cosine **0.705 vs 0.533**, **10.6%** of clauses have a near-duplicate (cos>0.95) vs 0.2% of English sentences, and the clause cloud is more anisotropic (centroid cosine 0.449 vs 0.359) — the #24 loop finding at the semantic level. The `units`/`raw` views agree on 86% of top-10 neighbours (view choice is safe); the SBERT/BGE encoders agree on only 35% (encoder choice must be pinned, the primary being SBERT). Curated neighbours are coherent (`artificial sun`→`The artificial sun`, `drug embryo`→`Drug-embryo`, `cadaver city`→`ToKAGE … Cadaver City`). Disclosed: 36 empty-text clauses and 2 clauses truncated at 256 tokens (the page-90 OCR loop), and the English control is matched in count, not domain.
- [x] Chainlink #41 — topic modelling (LDA / NMF / BERTopic-style): `analysis/topics/` (`topic_modelling_report.md` — the deliverable, `coherence_scan.tsv` — model × k gensim `c_v`/`u_mass`, `lda_topics.tsv`/`lda_topic_summary.tsv`, `nmf_topics.tsv`/`nmf_topic_summary.tsv`, `bertopic_topics.tsv`/`bertopic_topic_summary.tsv`, `bertopic_scan.tsv`, `clause_topics.tsv` — dominant topic per clause, `community_overlap.tsv`, `community_ari.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/topic_modelling.py` + regression test `work/test_topic_modelling.py` (38 tests). Documents are the #39 clause index `text_units` (markup-screened #17 unit surfaces, function words removed), so the token space is **exactly #29's**: all 391 network nodes are topic-vocabulary terms. **Verdict:** coherence has **no interior peak** (LDA `c_v` 0.586 at k=5 falling to 0.492 at k=16), so the book has no natural topic count — only graded regions of one repeating theme-space (the #39/#40 continuum seen from the topic side). The coherence-selected catalogues are LDA k=5 (`adam·doll·world`, `drug·boy-roid·cadaver`, `sun·artificial·murder`, `internal·flesh·organ`, `ice·faecal·black`) and NMF k=6; the **BERTopic-style** model (BERTopic's algorithm reproduced on `umap-learn` + `hdbscan` + c-TF-IDF — the `bertopic` package is not installed) resolves 22 topics with 24% noise. **Triangulation vs #29 is weak-to-moderate**: adjusted Rand over the 391 shared nodes is 0.189 (LDA), 0.086 (NMF), 0.026 (BERTopic) — above chance but far from identical, because topics and communities are different coarse-grainings of the same fused vocabulary (divergence is a lead, not a failure). A dedicated affect pole does **not** emerge (`ecstasy`/`desire` are absorbed into the body/genetics poles), consistent with #22's hapax scream channel. Deterministic: fixed seeds, single job, no wall-clock in any artifact.
- [ ] Embeddings / topics / motif map (clusters #40 ✓, topics #41 ✓; UMAP map #42 ✓, axes #43 ✓; affect #44, analogies #45 open).
- [x] Chainlink #73 — triangulation / convergence report: `analysis/triangulation/` (`triangulation_report.md` — the deliverable, `convergence_matrix.tsv` + `.md` — motif × method purity, dominant buckets, ten pairwise Jaccards, periodicity verdict, `motif_method_cells.tsv` — long form, `motif_inventory.tsv` — provenance / page spread / verdict, `domain_convergence.tsv` — #29's six named domains, `method_agreement.tsv` — clause-level pairwise ARI + NMI with permutation p, `clause_agreement.tsv`, `cluster_validation.tsv` — re-derived k-means vs #40's catalogue, `motif_count_reconciliation.tsv` — motif counts vs #40's inventory, `periodicity_motifs.tsv` — per-motif page-series + renewal stats with BH q, `periodicity_book_context.tsv` — #51's series cited, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/triangulation.py` + regression test `work/test_triangulation.py` (63 tests). Four independent analyses are cast as partitions of the same 5,614 clauses — #29 communities (lifted to clauses by node frequency, coverage 93.6%), #41 LDA k=5 / NMF k=6 / BERTopic-style (their persisted per-clause labels) and #40's embedding clusters (re-derived in `numpy` because #40 persisted the catalogue, not the labels; validated against it: 12/12 clusters matched, mean size ratio 1.00, and the same #40 cluster is the motif's home for all 16 clause-bearing shared motifs) — then compared pairwise by **adjusted Rand over clauses** and by **Jaccard overlap of each method's dominant bucket per motif** (label-free, so k=5 and k=22 are comparable). **Verdict:** convergence is real but partial — mean pairwise ARI 0.1028 with all ten pairs above their permutation null, and 6 of 39 motifs convergent (`ice nebula` 0.86, `entrails` 0.71, `artificial sun` 0.64, `cold-blooded` 0.58, `cyclops` 0.56, `drug embryo` 0.56) — the same poles #29/#40/#41 each reported alone; of the 39 motifs 19 partial, 12 divergent, 2 untestable (the six #29 domain rows are counted separately and all six diverge), with the motif divergence concentrated on (a) motifs neither method concentrates (24 rows where both weakest-pair purities < 0.5), (b) the BERTopic noise class (1,362 unclustered clauses; 15 divergent rows compare against it — `second` is the sharpest case, since the density model declines to cluster what #40's and #41's catalogues both name), and (c) the thin `sex`/`corporate` domains. **The periodicity method diverges from the other three, and that is the result:** none of the 45 motif/domain page series carries a cycle (#51's null family, with a short-band local line statistic because #51's band-median ratio is inflated both by the page-rate envelope and by block-shuffling runs) — the motifs are dispersed over the book, not metrical. #29's six named domains converge on *nothing* (mean Jaccard 0.16–0.22 for all six), a sharper form of #29's own verdict that the seed lists cut across the vocabulary. Scope is deliberately internal: **no external corpus is acquired or used as an arbiter**, and no method is treated as ground truth.
- [x] Chainlink #50 — quantified silence-removal experiment (the signature Beguš test): `analysis/silence_experiment/` (`silence_experiment_report.md` — the deliverable, `deltas.tsv` — the master variant × 3-delta table, `entropy_by_variant.tsv`, `compression_by_variant.tsv`, `perplexity_by_variant.tsv`, `perplexity_per_page.tsv`, `paired_deltas.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/silence_experiment.py` + regression test `work/test_silence_experiment.py` (19 tests). Each frozen #13 variant (v0 baseline .. v6) is scored by three instruments and the **delta vs v0** reported: token + char entropy, six #48 compressors, and **GPT-2 streaming perplexity** (the #49 instrument, one pass per variant → pooled book ppl + per-page paired sign test). Token entropy reconciles with #13 exactly and compression with #48 exactly; v0 ppl 106.2 vs #49's 104.2 (near-identical scope). **Verdict — `v4_operators_to_period` most lowers perplexity (84.1 vs 106.2, −22.2; 214/223 pages improved, sign-test p≈0), so it is closest to the underlying "vowel" system.** The clean controlled contrast is v3 (map-to-space, neutral: 106.7, p=0.59) vs v4 (map-to-period, big win) — same operator positions, different replacement character — so the active ingredient is **making the implicit punctuation explicit**, not operator removal per se (#12 showed the operators *are* the book's punctuation: 22 periods in 224 pages). **Negative result:** the Beguš-faithful strict deletions v1/v2 *raise* perplexity (128.4 / 124.9) because fusing neighbours into portmanteau non-words (`soulgram`) destroys readable structure faster than removing separators creates it. The v5/v6 diagnostic shows `::`->period (101.0) edging `//`->period (102.2) — a mild tension with #12's `//`-is-the-boundary claim, disclosed with the density confound (`::` 3,083 vs `//` 2,149).
- [x] Chainlink #51 — periodicity / spectral analysis (L6.5): `analysis/periodicity/` (`periodicity_report.md` — the deliverable, `series_specs.tsv`, `autocorrelation.tsv` — book ACF + shuffle envelopes, `spectrum_peaks.tsv`, `spectrum.tsv`, `periodicity_tests.tsv`, `distance_histogram.tsv`, `fano.tsv`, `english_baseline.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`, `periodicity_spectrum.png`, `periodicity_autocorr.png`) + `work/periodicity.py` + regression test `work/test_periodicity.py` (37 tests). The #17 token stream is mapped to a **delimiter indicator** (1 at an operator token; 90,220 tokens, 11,000 operators) and its **inter-delimiter distances**, plus char-level and #33 clause-length series; each is autocorrelated and Hann-windowed-FFT'd and tested against **global / block / detrended shuffle** nulls (200 each, seed 42). **Verdict — a negative result for the jacket-copy "loop": there is no latent fixed periodicity.** The operator indicator is close to white (spectral flatness 0.532 vs the white-noise anchor `exp(-γ)=0.561`); its strongest periodogram peak is at the *longest* period (4,511 tokens) and is an **OCR/HTML artifact** — 732 operator tokens lie inside #27 markup spans (486 on page 92) and screening them gives flatness 0.555 with **no significant peak** (p_global 0.33, p_block 0.70), as does dropping the three markup-heavy pages (flatness 0.567, p_block 0.72). The inter-operator distances are close to geometric but over-dispersed (mean 8.18, CV 1.056, observed/geometric variance 1.271, lag-1 correlation 0.046) — memoryless plus operator runs and page drift, not a cycle; English punctuation is *more regular* than Poisson (CV 0.823) and the #33 clause lengths show no periodicity (p_block 0.71). Consistent with #11's "shallow local run syntax" and #47's context-free operator channel: the "devastating loop" lives in lexical repetition (#24/#39), not in a metrical cycle of the symbol stream.
- [x] Chainlink #36 — formal grammar induction and complexity: `analysis/grammar_induction/` (`grammar_induction_report.md` — the deliverable, `subword_complexity.tsv` — p(L) by stream × L with ceilings and i.i.d./page-order nulls, `entropy_rate.tsv` — plug-in + leave-one-page-out H(next|prev k), `mi_by_lag.tsv` — MI vs lag with a shuffle null, `lz78.tsv`, `state_counts.tsv` — three estimators, `delimiter_pairing.tsv`, `opener_closer_dependence.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/grammar_induction.py` + `work/test_grammar_induction.py` (35 tests). Four independent instruments, each isolating one axis of the Chomsky hierarchy, over the #10/#11 ordered operator stream (10,949 tokens, V=11), with the #17/#35 screened unit stream (74,077, V=5,498) as a token-level contrast and the #47 English `punct_core` channel (matched N) as the baseline. **Verdict: FINITE-MEMORY, PAGE-APERIODIC CODE CHANNEL** — regular-like in memory, non-regular (drifting) in state space; no context-free and no context-sensitive evidence; neither i.i.d. noise nor a copy loop. Subword complexity p(L) sits below the i.i.d. null at every L (24% reduction at L=5) but is reproduced to within ~1% by a page-order-shuffle null, so the structure is page-local, not a global grammar, and p(L) does not saturate (0.92 of ceiling at L=12, still rising) — not a small fixed automaton. The held-out entropy-rate curve plateaus at order 1–2 (2.7597→2.7089 bits, a 0.0508-bit context gain vs English's 0.0881) — finite memory, no unbounded context. MI(1)=0.1305 bits is 20× the i.i.d. null (permutation p=0.000) — real sequential structure, not noise. LZ78 reuses only 1 of 2,433 phrases — not a copy loop. The `<<`/`>>` pairing is local (median 33 chars) and shallow (depth ≤ 4 for 225/230 pairs) — no stack, no CFG signature. **State count spread is the finding:** ~98 predictive states (order-2), ≥325 Nerode classes (lower bound, growing), 2,433 LZ78 phrases — small enough to be predictable, too large and drifting to be a small fixed grammar. Reproduces #12/#14's delimiter pairing exactly (230 matched / 182 unclosed / 156 stray) and is consistent with #11's "shallow local run syntax" and #51's page-local-not-metrical periodicity finding.
- [x] Chainlink #72 — predictive validity on held-out text: `analysis/heldout_validity/` (`heldout_validity_report.md` — the deliverable, `operator_heldout.tsv` — per split × order perplexity/accuracy/gain, `run_coverage.tsv`, `collocation_heldout.tsv` — per layer × split stability scores, `heldout_scores.tsv` — the headline numbers, `reconciliation.tsv` against #11/#27/#35/#36, `diagnostics.tsv`, `summary.json`) + `work/heldout_validity.py` + `work/test_heldout_validity.py` (86 tests). Does the derived grammar predict held-out operator sequences and collocations, train 80% / test 20%, pages as the split unit (no sequence crosses the boundary) and a train-only n-gram vocabulary (no test leakage into smoothing)? Two claims, two baselines. **Operator sequences** (the #11/#35 interpolated n-gram model, order 0/1/2, Jelinek-Mercer, per-page): on the conservative chronological split, order 2 **PREDICTS IN PROBABILITY** — perplexity 6.100 vs the order-0 baseline's 6.792 (a 10.2% gain), top-2 accuracy 44.9%→50.8%, 99.3% of held-out run tokens using a train-seen run type — but argmax accuracy is flat (30.4%→29.9%: better probability mass, not a better single guess). On the optimistic random split, order 2 **overfits** (6.323 worse than order 1's 6.228), so only order 1 generalises there. **Collocations** (the #27 PMI/logDice/Dice/t-score table): rank is **STABLE** at the unit layer (Spearman logDice 0.8984) and **MODERATELY STABLE** at the operator-word layer (0.7080) — a train-ranked collocation keeps its relative strength in test — but top-100 membership does not survive wholesale (overlap 47%/59%, train→test survival 48%/66%, the 4×-smaller test split dropping mid-count train pairs below the f≥2 floor), with no systematic logDice level shift (mean |Δ| 0.86/0.94 on a 14-point scale). Consistent with #11's PARTIALLY IDIOMATIC verdict and #36's finite-memory, page-local verdict: the grammar generalises in probability and in rank, at short range, not as a system that names the one right continuation.
- [x] Chainlink #37 — POS-tagging the silence-removed residue: `analysis/pos_tagging/` (`pos_comparison_report.md` — the deliverable, `pos_distribution.tsv` — full tag counts v3 vs reference, `noun_verb_balance.tsv`, `per_corpus_balance.tsv` — per-reference-file noun/verb ratios, `tagged_v3_sample.tsv` + `tagged_ref_sample.tsv` — spot-checkable tagged excerpts, `summary.json`) + `work/pos_tagging.py` + `work/test_pos_tagging.py` (29 tests). Tags the #13 `v3_operators_to_space` plain stream (80,260 word tokens, E1.4's silence removal) and the #56 reference corpora's 15 general-English novels (1,628,800 tokens) with a **rule-based, auditable tagger** (no NLTK/spaCy in this environment): a ~3,000-word curated lexicon + Brown-corpus verb forms, morphological suffix rules for unknown words, and bigram contextual disambiguation — the *same* tagger on both streams, so the comparison is fair by construction. **Verdict: the residue is noun-heavy relative to normal English** — noun 42.62% vs 23.0% (+19.62pp), verb 12.78% vs 21.35% (−8.57pp), noun/verb ratio 3.334 vs 1.077 (English prose runs 1.5–2.5), chi-square on the noun/verb × corpus table highly significant (χ²=10570.07, p<0.001 across all 23 reference files individually). Three caveats disclosed in the report rather than hidden: the deterministic tagger's absolute accuracy is not benchmarked against a neural tagger (only the *cross-stream* comparison is load-bearing), capitalized unknowns default to NNP (inflating proper-noun counts on the book's neologisms like `ToKAGE`), and other unknowns default to NN (biasing both streams' noun counts upward together, not differentially). Consistent with #18's compound-dense style and handed off to #50 (silence-removal perplexity) and #62 (interlinear edition) as a POS annotation layer.
- [x] Chainlink #42 — UMAP theme map of the book: `analysis/umap_theme_map/` (`umap_theme_map_report.md` — the deliverable, `umap_page_map.html` — the interactive map, plotly, UMAP/t-SNE and page/cluster toggles, trajectory overlay, hover text, `umap_map_static.png` — 2×2 static map, `trajectory_map.png`, `drift_curve.png`, `umap_coords.tsv`, `drift_stats.tsv`, `lag_return.tsv`, `page_nn_spread.tsv`, `cluster_crossref.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/umap_theme_map.py` + `work/test_umap_theme_map.py` (39 tests; environment needs `umap-learn`, run inside the `marker` conda env). Projects the #39 SBERT clause embeddings (5,614 clauses, 384-d, `units` view) with UMAP (primary, `n_neighbors=30`, cosine) and t-SNE (comparison), coloured by page and by #40's twelve theme clusters (same algorithm/settings/seed — exact partition match, confirmed by `cluster_crossref.tsv`). **Verdict: no narrative drift, a real but local loop.** The page-coloured map is one intermixed cloud — every page's clauses sit on every other page's (consecutive-step/random-pair ratio 0.995; 10-NN mean neighbour-page delta ≈ a 75-page random-pair null; page explains only 5.4% of embedding variance; first-half/second-half centroid cosine 0.9879) — the book is **stationary**, not drifting. The lag-return curve dips significantly below a 10-shuffle null at lags 2–30 (minimum at lag 5, z=−8.2) and is gone by lag ~400: the jacket's "devastating loop" is **template recurrence at the scale of a few clauses**, not a page-scale cycle — consistent with #51's no-fixed-periodicity verdict restated in semantic space. The theme-coloured map shows what *does* separate: #40's twelve motif regions occupy distinct neighbourhoods of the same cloud (not separate islands — silhouette 0.033 at k=12, #40's own number). Per-page texture (`page_nn_spread.tsv`) locates the loopiest pages (p83, p8, p188) and the most register-distinct pages (p183, p219, p209), with the #39/#40/#51 page-90/92 OCR-markup stretch sitting mid-table rather than as an outlier.
- [x] Chainlink #43 — contrastive semantic axes: `analysis/semantic_axes/` (`semantic_axes_report.md` — the deliverable, `axes.tsv` — the four named axes with all four tests, `seed_pairs.tsv` — every antonym pair's cosine to its axis + corpus counts, `axis_projections.tsv` — the 12 #40 centroids on each axis, `extreme_clauses.tsv` — pole exemplars, `cluster_pair_axes.tsv` — all 66 centroid-difference dimensions with seed loadings, `controls.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/semantic_axes.py` + `work/test_semantic_axes.py` (35 tests). The task's four axes — `organic<->machine`, `self<->other`, `body<->information`, `sacred<->profane` — are tested as **directions in #39's own clause-vector space**: seed words are encoded with the pinned #39 SBERT encoder (the only encoder call; #28's static embeddings are a different space and are never mixed in), an axis is mean(+pole) − mean(−pole), and each axis faces four tests — seed-pair coherence vs a **200-draw randomized-axis control** (random pole assignments over the same term pool), the +pole vs −pole clause projection contrast, and the spread of #40's twelve centroids along the axis vs a **200-direction random null**. The partition is #40's exactly (k=12, `n_init=10`, seed 0; catalogue sizes and medoids reproduced). **Verdict: all four axes are real, with a stable strength order.** Every axis beats both controls (coherence z = +2.73 … +3.19; centroid-spread z = +3.97 … +17.43) and every +/− clause contrast is positive (d = +0.88 … +1.60). **`organic<->machine` is the strongest separator** (centroid spread 0.149, z=+17.4; cluster 10 `cadaver·flesh·city` at the organic pole vs cluster 9 `i·lt·l` at the machine pole), then `body<->information` (0.104), `sacred<->profane` (0.081, cluster 1 `adam·doll·retro` vs cluster 5 `murder·assassin·entrails`), and **`self<->other` the weakest** (0.058, z=+4.0) — the book has a thick first-person register but almost no lexicalised *other* (`other` 2 clauses, `another` 1), so that axis is carried by one pronoun-heavy cluster rather than two opposed vocabularies. The **66 cluster-pair dimensions** show the book's largest separating gaps are all against the first-person cluster 9 (in 10 of the top 12 pairs) with self/other pronoun loadings — the strongest semantic dimension is **first person vs everything else**, which is the self<->other axis measured from the other side. All 29 seed pairs cohere positively with their axis (min +0.30, `cherry−hell`; max +0.69, `flesh−code`). Consistent with #40's continuum (silhouette 0.033): the axes are directions that separate centroids, not partitions of clauses.
- [x] Chainlink #56 — reference corpora + keyness harness (feeds #25): `analysis/reference_corpora/` (7 registers, `manifest.tsv` with source/licence/SHA-256/tokens per file) + `analysis/keyness/` (`keyness_report.md` — the deliverable, `keyness_<corpus>.tsv` — the 7 full tables, `summary.json`) + `work/reference_corpora.py` + `work/keyness.py` + regression tests `work/test_reference_corpora.py` (26 tests) and `work/test_keyness.py` (43 tests). Book side is the #17 *word-part* layer (79,770 kept of 80,363 parts, markup-screened with #19's ledger + #27's tag-run detector) — the only layer comparable to ordinary English words. Statistic: signed log-likelihood G² (Dunning 1993; Rayson & Garside 2000) with its chi-square p, a Benjamini–Hochberg **q** and a 0.5-smoothed log-ratio, gated at **q < 0.05** (tens of thousands of tested types per corpus make an uncorrected p-gate a false-positive factory). **Verdict:** 32 words are in the top 50 against *every* large reference (`cadaver`, `fecundates`, `flesh`, `organ`, `embryo`, `roid`, `gram`, `adam`, `doll`, `drone`, `vital`, plus the #24 function-word loop `of`/`the`/`that`), so they are findings about the book and not about a reference; 13 of the 944 general-English keys survive no technical register and are reported as residue of the historical-prose reference; the three copyright-cleared excerpt corpora (cyberpunk 105, cut-up 56, Finnegans Wake 49 tokens) are written but flagged as non-inferential and excluded from every stability claim.
- [x] Chainlink #56 (revised scope) — within-text language viability assessment: `analysis/language_viability/` (`language_viability_report.md` — the deliverable, `checks.tsv`, `null_distributions.tsv`, `order_sweep.tsv`, `split_stability.tsv`, `sensitivity.tsv`, `residue_ledger.tsv`, `oov_types.tsv`, `sense_stability.tsv`, `termbase_sample.tsv`, `gloss_passages.md`, `gloss_tokens.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/language_viability.py` + regression test `work/test_language_viability.py` (50 tests). Intrinsic evidence only: the book is split (interleaved pages: even train, odd test) and six pre-registered checks score the held-out half against **within-text nulls** (within-page shuffle, within-run shuffle, within-clause shuffle, within-unit part shuffle, type permutation) with add-one empirical p (B=40, seed 42), plus a **negative control** (V0) that runs the same machinery on the shuffled book and must fail. **Verdict: VIABLE (BOUNDED), 6 of 6 pass** — held-out bigram CE 8.2619 vs 10.0986 bits/unit (V1), operator bigram CE 2.6595 vs 2.6927 bits/op (V2), train-mined frames cover 0.6766 of held-out clauses vs a 0.6410 null maximum (V3), held-out OOV 0.0615 vs a 0.2006 null minimum and ρ=0.8340 shared-type frequency correlation (V4), same-term collocate Dice 0.5000 vs 0.3000 for mismatched terms (V5, sign-test p=8.4e-06), part-sequence bigram CE 5.0014 vs 6.0408 bits/part (V6). **Bound:** order-2 improves on order-1 by only 0.0253 bits/item (0.3%); order-3 does not improve, and this small gap has no separate uncertainty test. The evidence supports short-context structure, not a long-range grammar, alongside #11/#47/#51. Reuses #17, #18, #19, #27, #30, #33 and #34 artifacts only — no reference corpus is consulted or acquired (#78–#80 parked).
- [x] Silence-removal (Beguš) transform variants (#13) — residue streams persisted; the quantified perplexity version is done (#50).
- [x] Chainlink #70 — coverage accounting / residue ledger: `analysis/coverage_ledger/` (`coverage_ledger_report.md` — the deliverable, `coverage.tsv` — every bucket at four levels, `residue_ledger.tsv` — the awkward residue itemized and verdicted, `per_page.tsv`, `consumer_reconciliation.tsv` — 14 downstream layers' kept/dropped splits, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/coverage_ledger.py` + regression test `work/test_coverage_ledger.py` (42 tests). One ledger over the whole corpus at four levels, each forced to close: **480,962 page-body characters** (390,469 word / 17,269 operator / 1,735 punctuation / 71,021 whitespace / 468 layout scaffolding / **0 unexplained**), **90,220 tokens** (reproduces #14's 77,485 Lex nodes and #17's 11,000 operators), **11,000 operator tokens** each carrying exactly one #17 role (2,816 seams + 7,247 segment operators + 937 that bind nothing), and **74,665 unit occurrences** (74,318 with a #19 category + 347 in #19's markup ledger, `5,868 + 8 = 5,876` types). The **coverage metric** is 1.000000 at every level and the build fails if it ever drops, so it is a closure test rather than a score. The residue ledger publishes 11 classes of awkward residue, each with its owner, pages and page-cited examples: 468 scaffolding characters on the three figure-only pages (6 of the 12 spans sit between tokens, published nowhere until now), 155 characters of punctuation in no #12/#14 class, 937 unbound operators, 347 markup occurrences (201 of them on the surface `a`, so only a uid join removes them — a surface-level reader of #17's `unit_types.tsv` over-counts the article by 21.7%), 202 `href`/`mailto` occurrences #19 keeps as English `junk` and #27 screens out, and 127 OCR-error occurrences kept rather than corrected. Two classes are itemized by no artifact until this ledger: **13 occurrences of real book text deleted by #27's tag-run screen** (the OCR wrapped printed text in `<a href="atrocity genomics of retro-ADAM">`, p158, and `<a href="mailto:konganic-eye of cyberBuddha">` ×3, p219, and the span rule deletes the href value with the tag — the 241 "tag-furniture" occurrences #27 publishes are therefore 13 larger than that description), and **2 unit surfaces (`eruption=\njunk` p102, `device++\nmass` p209) that #17 and #19 store with a different number of backslashes**, so a surface-level join across the two artifacts loses exactly those two occurrences. #47's char total (480,309) is reconciled, not merely different: 480,962 − 468 scaffolding + 12 comment spaces − 197 heading-marker characters.
- [x] Chainlink #71 — consistency checker over the termbase: `analysis/consistency/` (`consistency_report.md` — the deliverable, `violations.tsv`, `summary.json`) + `work/consistency_checker.py` + `work/test_consistency_checker.py` (36 tests). Five deterministic checks that a term's reading does not drift between the artifacts that carry it: `#56`'s `GLOSS_TERMS` vs its own computed termbase row, `#62`'s `PLAIN_READINGS` vs `GLOSS_TERMS` (one surface, one curated gloss), the `#30` entity glossary's role/does-done-to verbs vs the termbase's entity columns, `OPERATOR_GLOSSES` for internal contradiction, and a global one-term-one-gloss sweep across every artifact. **Verdict: 0 violations** over 51 gloss terms, 49 plain readings, 49 termbase terms, 98 entity-glossary rows and 10 operator readings — the curated layers this project's glosses depend on agree with each other everywhere they overlap. Found and fixed one real inconsistency while auditing `#62`'s termbase: `exoskeleta` was glossed at `high` confidence as a plain plural while every other body-machine compound in the termbase is `medium` — corrected to `medium` (`work/interlinear_edition.py`).
- [x] Chainlink #76 — anti-pattern guardrails audit: `docs/guardrails.md` — the checklist, one anti-pattern per section, each paired with the mechanism that guards against it and the check that enforces the mechanism — + `analysis/guardrails/` (`audit_report.md` — the deliverable, `findings.tsv`, `summary.json`) + `work/guardrail_audit.py` + `work/test_guardrail_audit.py` (27 tests). Five checks, each reading the *published* artifact of the module it audits rather than re-deriving its numbers: **hallucinated coherence** — `#73`'s triangulation report must keep disclosing divergent/partial rows alongside convergent ones (currently 37/45 motifs disagree, 6/45 fully convergent); **LLM circularity** — the verdict-computing functions in `#63`'s glossing pipeline and `#64`'s translation committee are parsed with `ast` (function-scoped, so an unrelated loader elsewhere in the same file can't trip it) and must contain no token that would call a model backend or re-invoke generation (0 violations across the 10 functions checked); **over-correction** — `#6`'s lexical corrector's `REVIEW` class must stay non-empty, i.e. it must keep holding back genuinely ambiguous OOV tokens instead of "fixing" everything (currently 15 corrected vs 200 held back); **single-method certainty** — a convergence claim must be backed by ≥ 3 independent methods (`#73` uses 5: network community, LDA, NMF, BERTopic-style, KMeans); **residue-dropping** — `#70`'s coverage ledger must keep itemizing residue rather than claiming none exists, and report zero unexplained residue at every level it closes (11 residue classes itemized, 0 unexplained at all four levels). **All five guardrails currently pass.** The regression suite is adversarial as well as confirmatory: synthetic fixtures (an all-convergent triangulation summary, a verdict function that calls out to `opencode`, an emptied `REVIEW` class, two methods instead of three, an unexplained-residue ledger) are constructed and asserted to *fail* each check, so the guardrail is proven to have teeth before the artifact-backed suite pins today's passing result as the regression baseline.
- [x] Chainlink #62 — interlinear edition generator: `analysis/interlinear/` (`interlinear_report.md` — the deliverable, `interlinear.md` — the edition itself, three aligned lines for every clause, `interlinear.tsv` — one row per clause with both #13 normalizations + the gloss + confidence counts, `gloss_tokens.tsv` — the per-token reading/evidence table, `gloss_lexicon.tsv` — the readings with computed counts and #26 KWIC examples, `residue.tsv`, `per_page.tsv`, `operator_roles.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/interlinear_edition.py` + regression test `work/test_interlinear_edition.py` (49 tests). The plan's E8.2 rendering — **original clause | operator-normalized (silence removed) | plain-English gloss** — for all **5,614 #33 clauses / 225 pages**: line 2 is #13's v4 transform (operators → `.`, the variant #50 scored closest to the underlying system) rendered with #56's `render_variant` semantics and pinned by two build-time invariants (the token-assembled rendering equals the substring-deleted rendering for every clause; every token span reads back as its surface); line 3 substitutes a plain reading **only where the surface is not ordinary English**, so the book's English layer glosses to itself and the interpretation concentrates at the non-English sites — **99.5% identity**, 49 of 51 #19 `true_coinage` types read from the #26 KWIC contexts + #19 morphology + #30 roles (#56's confidence scheme), names not translated, #19's `ocr_error` class kept as printed, and the two opaque coinages (`omanko`, `laple`) marked `‹…›` and itemized for #63 rather than guessed. No LLM and no external corpus. Reconciliation: **76,897 = 76,897** word tokens against #33's `n_words_clean`, 90,220 tokens / 11,000 operators against #17, 952/540 slash roles against #14, and the 141-token gap to #33's event-counted `n_tokens` disclosed as #14's synthesized marker events. Reads only frozen artifacts (#7/#13/#17/#19/#26/#30/#33/#56); deterministic, byte-identical on rebuild.
- [x] Chainlink #67 — spectral reading artifact: `analysis/spectral_reading/` (`spectral_reading.md` — the publishable artifact: the full v4 silence-removed stream as a continuous, page-tagged reading, preceded by a framing note; `spectral_reading_report.md` — THE DELIVERABLE, the verification report; `reconciliation.tsv` — this build vs the published #13/#50 numbers; `diagnostics.tsv`, `summary.json`) + `work/spectral_reading.py` + regression test `work/test_spectral_reading.py` (57 tests). The silence-removed stream (v4, `v4_operators_to_period`, the transform #50 found most lowers LM perplexity: 84.05 vs 106.22) is published as a **distinct under-text reading, explicitly NOT 'the real text'** — the framing note says so in bold, the reading section repeats it, the colophon repeats it, and a test fails if the phrase "this is the real text" ever appears. 11,645 silence sites (10,949 operators + 696 free-standing dashes, #13's disclosed judgement call) become 10,435 inserted periods; with the corpus's 22 pre-existing periods the reading has 10,457 periods and 10,136 word-bearing segments (median 6 words). **Every headline number is computed or read from its published source and reconciled row by row** (14/14 rows match: pages 225, word tokens 77,392 identical to the baseline's token sequence, segments 10,136, median 6, uppercase-initial 25.5% / 27.6% length-controlled vs the baseline's 52.9% / 56.2% on 17 segments and 22 periods) — a mismatch is a build error, not a footnote. The interpretive limits are stated with numbers: 25.5% uppercase-initial segments and 12.7% single-word segments make this a **skeleton, not a reconstruction**, and the transform erases the operators' distinct functions (`::` phrase separator / `//` utterance boundary / `=` naming seam all become periods). A rule-selected worked example (page 18, source line 3, folio 20) shows one line before and after so the transform can be checked by eye. The residue is itemized, not dropped: 5 token-less period segments, each a lone `:` where a literal colon stands between two operator-derived periods. Offline, deterministic, no clock; rebuild is byte-identical.
- [x] Chainlink #68 — compression test for candidate readings: `analysis/reading_compression/` (`reading_compression_report.md` — the deliverable, `readings.tsv` — the reading inventory with provenance, `compression.tsv` — per reading x six compressors, `controls_and_nulls.tsv` — the length-matched control and the shuffle null, `predictability.tsv` — the held-out character-model legs with the paired sign tests, `ranked_readings.tsv` — THE RANKING, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/reading_compression.py` + regression test `work/test_reading_compression.py` (58 tests). The plan's L8.7 test: **a good gloss shortens the description of the book** — score every candidate reading the repository has published by whether it makes the corpus more compressible *and* more predictable. Nine readings, all read from frozen artifacts, none invented: the **identity control** (the untouched source clause), #13's `v3_operators_to_space` and `v4_operators_to_period`, #62's `interlinear_gloss`, and #63's five LLM prompt strategies (`literal_operator`, `register_paraphrase`, `scene_summary`, `plain_english`, `entity_consistent`) over its 5-clause stratified sample. Because #48 proved raw bits/char cannot discriminate readings (the book is *more* compressible than matched English), the harness scores each reading by three legs instead: **(1) a length-matched deletion control (the declared primary leg)** — a position-preserving *letters-word* subsample of the original down to the reading's letters-word count (a word is a maximal run of letters, not a whitespace chunk, which would miscount the readings because operator removal changes the characters *between* words), repeated over 5 independent seeds with its spread reported (sd ~4e-4 bits/char on the full book, so the full-book deltas are two orders of magnitude above the control's own noise); **(2) a pairing shuffle null** — the reading's clause texts permuted across its clause ids, so a gain that needs the *right* text on the *right* clause is separated from a gain that only needs short text; **(3) held-out predictability** — a character trigram with Jelinek-Mercer interpolation (#56's estimator, lambda tuned on a train-internal validation split) scored on clauses it never trained on: one **fixed** model (trained on the identity train clauses only) scores every reading, so the readings are compared by a single shared scorer, and the full-book readings additionally get an **adaptive** model of their own. The held-out split is every 5th clause (1,123 test / 4,491 train), spanning 219 of 225 pages with **no two adjacent clauses both held out** (#56's leakage lesson). Reconciliation: **5,614 = 5,614** clauses, **225 = 225** pages, **76,897 = 76,897** word tokens against #33/#62 — a mismatch is a build error; the #48 compression rows match to within the disclosed scope difference (clause-joined stream vs page stream: identity 2.107 vs #48's 2.135 bits/char). **Verdict — `interlinear_gloss` wins the declared primary leg** (length-control delta −0.0671 vs v3's −0.0647, identity's 0.0000), so it is the reading that most shortens the description of the book under this protocol; the ranking is sorted by that rank and the mean-of-three aggregate never overrides it. **The legs then split, which is the finding:** `v3_operators_to_space` wins *both* predictability legs (fixed-model gain −0.0135 bits/char, 304 improved / 208 worsened held-out clauses, sign-test p = 2.6e-05; own held-out self-surprisal 2.4989 vs identity's 2.5380) while `interlinear_gloss` wins only the description-length leg. A reading that only rewrites orthography is the more predictable one; a reading that also substitutes the termbase is the more compressible one. **The disclosed instrument disagreement: v4 — #50's GPT-2 winner — is the *worst* full-book reading here** (fixed-model gain +0.1558; only 27 of 512 held-out clauses improved, p ≈ 0), because a model trained on this book expects the book's own operator characters and is surprised when they become periods, while GPT-2 is a general-English LM that *expects* periods. Two instruments, two defensible answers — the single-method-certainty trap L9.8 names. External anchor: the untouched held-out book scores **4.134 bits/char under a general-English character model** (trained on #56's 15 reference novels) versus ~2.5 under its own models — #48's "more compressible than English" restated at character level, and the reason a raw compression number could never separate two readings of this book. Guardrail carried into every artifact: **compression is evidence about predictability, not proof that an interpretation is true**. The limits section discloses the primary leg's honest boundary — a reading with at least as many words as its clause gets *no* subsample (nothing to delete), so for v3/v4/gloss the control is the original on 5,470–5,479 of 5,614 clauses and the delta reduces toward the raw bits/char delta; the per-reading live-clause count is published in `controls_and_nulls.tsv` (`control_fallback_clauses`: 135 live clauses for the gloss, 144 for v3/v4) rather than assumed. It also discloses that the control neutralises *arbitrary* deletion but not *structured* deletion (a degenerate "keep each clause's first word" reading wins the compression leg, and only the predictability legs and #63's fidelity checks stop it being mistaken for a reading). Offline, deterministic, no clock; rebuild is byte-identical.
- [x] Chainlink #74 — back-translation harness: `analysis/back_translation/` (`back_translation_report.md` — the deliverable, `fidelity_scores.tsv` — THE RESULT, one row per reading with all four channels plus the labelled aggregate and mean-rank, `round_trips.tsv` — one row per (clause × reading) round trip with the regenerated clause, `worked_examples.tsv` — every reading over every sampled clause so the loop can be checked by eye, `controls_and_nulls.tsv` — the operator-strip ablation and the permutation null, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/back_translation.py` + regression test `work/test_back_translation.py` (36 tests). The plan's **L9.6** test: *gloss -> regenerate a Siratori-style clause -> compare to original*. There is no ground-truth translation, so this is a **round-trip (bidirectional-consistency)** measure, not an accuracy claim: every published reading is treated as the candidate gloss, an offline deterministic re-writer regenerates a Siratori-style clause from it **without seeing the original**, and four independent channels score the pair — **`operator_recall`** (the primary leg; a *Dice* coefficient over the operator multiset, so one number charges both a dropped operator and an invented one), **`operator_order`** (LCS ratio over the operator *sequence*, which catches a gloss that kept the right multiset in the wrong order), **`content_recall`** (the lexis channel) and **`coined_recall`** (the seam-bearing surfaces `BABEL=TV`, `soul/gram`). **The channels split, and the split is the finding:** the silence-removal variants (v3/v4) and #62's own `interlinear_gloss` keep every word and lose every operator, so they score `content_recall` 1.000 and `operator_recall` 0.000; `plain_english` loses *both* (0.408 / 0.000) because it rewords as well as de-operatorises; and only the operator-preserving strategies round-trip the machine — **`register_paraphrase` is the only reading that recovers the operator channel in full** (1.000/1.000/0.937/0.700, aggregate 0.909, first on the primary leg), exactly because it reproduces the clause nearly verbatim. `literal_operator` is charged for the operators it *adds* (its own #63 `operator_fidelity` check already failed on that): Dice 0.905, not 1.000. Three controls keep the table honest and each is a build error rather than a footnote: the **identity control** (a gloss that is the source clause itself must score exactly 1.000 on every channel — the harness's own calibration), the **operator-strip ablation** (removing the operator surfaces must floor `operator_recall` to 0.000 while `content_recall` survives, which is what proves the primary leg is about the *operator* and not crediting shared lexis) and the **permutation null** (shuffling the clause<->reading pairing must collapse the aggregate, so the score cannot be the vocabulary Siratori's clauses share with each other). **Circularity is structurally impossible:** no LLM is invoked anywhere; the glosses are #63's pins, the operator scanner is #63's own eleven-surface scanner (asserted equal to `glossing_pipeline.py` over 400 corpus clauses), and the judge is this module — the L9.8 answer to "same model translating and judging". The regenerator's **limit is disclosed rather than dressed up**: it is a bounded re-render (editorial asides dropped, whitespace canonicalised) that deliberately refuses to invent the structure a lossy forward pass destroyed, so the round trip records what a reading *lost* instead of manufacturing a fuller clause than the gloss supports — a `[...]`-keeping operator count is shown to overscore, and the `residue` stratum is retained. #63's `operators_out` total (54) and this build's scored count (47) are published side by side, the 7-operator gap being exactly the operators nested inside the glosses' editorial asides. Reconciliation: **5,614 = 5,614** interlinear clauses, **5 = 5** sampled clauses, **25 = 25** gloss rows, **5 = 5** strategies — a mismatch is a build error. Offline, deterministic, no clock; rebuild is byte-identical.
- [x] Chainlink #75 — human jury / reader study design: `analysis/reader_study/` (`protocol.md` — the pre-registration (question, population, design, instrument verbatim, analysis plan, stopping rule, what the study cannot establish), `reader_study_report.md` — the verification report, `instrument.tsv` — the 7 items with their anchors and roles, `frame.tsv` — the 45-cell sampling frame with every cell's role and #74's published fidelity, `blocks.tsv` — the counterbalanced packets, `pilot_ratings.tsv` / `pilot_pairwise.tsv` / `pilot_cell_summary.tsv` / `pilot_stratum_summary.tsv` / `agreement.tsv` / `machine_human.tsv` / `pairwise_summary.tsv` / `reader_flags.tsv` / `reading_effect.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/reader_study.py` + regression test `work/test_reader_study.py` (62 tests). The plan's **L9.7** item — *have readers rate glosses for fidelity to the felt text* — designed rather than run, because the next expensive step is asking people and every earlier step was cheap to get wrong. **Instrument:** 7 items per stimulation — `felt_fidelity` (0–7, the primary outcome), `fidelity_confidence`, `reading_appeal` (a *diagnostic* item, deliberately not the primary in disguise), two anchored forced choices, and **two planted traps** carrying #63's `plain_english` reading (the strategy #63's own checks most often fail and #74 measures least faithful overall, aggregate 0.102): `trap_overcorrected` asks how faithful a *smoothing* of the derangement is, and `trap_invented` how faithful an *added plot event* is — both are read pass/fail against a known answer rather than as a scale, because they sit at the floor where a 7-point scale has no resolution left. **Sampling:** a *census* of the gloss population over #63's stratified 5-clause sample — 45 cells (5 × [identity control + #62 gloss + 5 LLM strategies + 2 traps]), each block showing 6 with the LLM comparators rotated by `PAIR_OFFSETS` and the within-block order rotated by the block index, so a reader never rates a gloss without the clause it translates beside it and no reading is systematically first. **Analysis, pre-registered in `protocol.md` before any number exists:** per-cell means with seeded bootstrap CIs; a within-clause contrast (identity control minus the other readings, pooled per clause) tested by *permuting the control label*, which keeps each clause's values, size and difficulty while destroying exactly the claim being tested; Krippendorff's alpha with the **interval** metric (a 0–7 scale carries equal intervals) over cells as units, each alpha read against a null that re-groups the values — shuffling *inside* a unit is a no-op, so it is not a null; Spearman of the human per-reading mean against #74's published `aggregate_fidelity` with a label permutation; Kruskal–Wallis H and eta²; per-stratum descriptives never pooled. **The pilot is a *simulation* and says so in its first line:** 1,800 integer ratings over 5 blocks × 12 synthetic readers × 5 items from a disclosed generative model, every artifact carrying `simulated = True`, with `human_data_collected: False` pinned by the suite. The generative model's felt-fidelity expectations are anchored to #74's channels but deliberately *not* a monotone function of them — the two operator-destroying English readings (`interlinear_gloss`, `plain_english`) sit above the machine's ranking of them, because they are the ones a reader finds easiest to read — and a test fails if the ranking ever becomes an exact copy of #74's, which would make the machine/human leg a tautology. Pilot results (all properties of that model, none of them evidence about readers): the identity control tops every block (contrast **+2.43**, label-permutation p = 0.0015), `felt_fidelity` alpha **0.790** against a null centred on 0 (q05 −0.030, q95 0.035, p = 0.005), machine/human Spearman **ρ = +0.893** (p = 0.014), H = 18.10, η² = 0.993, and the appeal table shows the intended split — `plain_english` is the most appealing reading (6.13) and among the least faithful (3.92), which is the confound the instrument exists to expose rather than average away. Two build-time controls have teeth: the identity control must top its block or the build stops, and the attention check must be able to flag a reader (1 of the 12 synthetic readers is flagged). **Anti-pattern audit (L9.8):** circularity is structurally impossible — no model generates or grades anything, and the only LLM output in the frame is #63's pinned capture treated as a candidate; single-method certainty is answered by 7 items, 4 statistical legs, 3 nulls and per-stratum reporting; residue-dropping is answered by publishing the cells a block does *not* draw, counting the traps and excluding them by role, and publishing reader exclusions with their reasons; over-correction is measured rather than corrected away. The disclosed limits: no ground-truth translation exists (#74), so a high score means readers recognise the feeling, not that a reading is *the* translation; readers are not the text's original audience; and a 5-clause stratified sample is a sample of #63's strata, not of the book. Reconciliation: **5,614 = 5,614** interlinear clauses, **5 = 5** sampled clauses, **25 = 25** gloss rows, **5 = 5** strategies — a mismatch is a build error. Offline, deterministic, no clock; rebuild is byte-identical.
- [x] Interlinear gloss + termbase (partial, from #56's viability assessment): `analysis/language_viability/termbase_sample.tsv` (49 terms with #18 parts, #19 category, #30 entity roles/verbs, computed collocates and #34 frame fillers, plus a gloss and a confidence level) and `gloss_passages.md` / `gloss_tokens.tsv` (three page-cited passages, token by token, with the #13 v3/v4 operator-normalized renderings and every unresolved token listed rather than repaired).
- [x] Chainlink #58 — translations as an interpretive oracle: `analysis/translation_oracle/` (`availability_report.md` — the deliverable, `alignment_report.md` — the alignment itself, `sources.tsv` — the acquisition ledger, `ru_quotations.tsv`, `fantlab_reviews.tsv` — the source reviews, `author_paratext.tsv`, `alignment.tsv`, `operator_invariance.tsv`, `bilingual_lexicon.tsv`, `oracle_checks.tsv`, `aligner_eval.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/translation_oracle.py` + regression test `work/test_translation_oracle.py` (100 tests). **Half a negative result, half a real one.** (a) *The Skobin parallel text could not be obtained lawfully, and that is the finding, not a failure.* 21 candidate sources probed and pinned: the Russian edition is a commercial 2006 АСТ/Адаптек printing (ISBN 5-17-037280-9), FantLab catalogues it and serves no text, Litres and the NEL refuse the agent — and the only full texts on the open web are eight unauthorized mirrors, which the ingestion gate refuses after a **HEAD-only** probe (no mirror body was ever requested). The nettime thread named in the methods inventory is recorded as searched-not-found rather than dropped. (b) *The disambiguation signal was obtained anyway, from lawfully quotable material:* twelve short attributed quotations of the Skobin translation that Russian readers publish inside their FantLab reviews, aligned to the book. **The load-bearing result: the Russian edition preserves the operator channel verbatim**, so the symbol stream is a translation-invariant key — two long quotations are located by **operator skeleton alone, with no lexical anchor at all**, at **100% positional operator retention** (RUQ01 16/16, its 16-operator sequence occurring exactly once in the corpus's 10,145 operator tokens; RUQ02 12/12, occurring twice at byte-identical sites pp. 5 and 221 — a finding about the book's loop, not about the aligner). 11 of 12 quotations resolve to a single English text; the bilingual lexicon carries **17 attested** EN↔RU pairs (`tomb-device`→`могильное устройство`, `boyroid`→`бойроида`, `Cadaver City`→`Город Трупов`) graded against the anchor priors. One **divergence** is flagged, not resolved: the reader reports the chapter subtitle as «ЛИКВИДАЦИЯ» where the English reads `abolition`. The aligner itself is a monotone DP with a bounded n-m diagonal, scored by **standard AER** (Gale & Church / Levow: deletions + insertions of null-collapsed links over the gold links of the same split, so the score is an error *rate* per gold link and is not bounded above by 1) on a built-in 24-pair EN–RU gold with planted 1-n / n-1 operations, weights fitted on a **tuning half** and reported on a **held-out half**: **0.539 against a length-only proportional baseline's 1.821 on the same held-out half** (0.405 vs 1.911 over all twelve documents). The author's own lexicon is tested against the book: the two nouns the interviewer names (`ADAM` 509, `dogs` 495) are corroborated, the essayistic vocabulary around them (`hardweb`, `creature intensity`, `echo archive`) is **absent**, and the two terms that do occur (`gene-dub`, `Biocapture`) occur once each — both as chapter headings. **Quotation provenance is pinned, not asserted:** each of the twelve names the reader review it was transcribed from (`fantlab_reviews.tsv` — author, account, `datePublished`, the `#responseNNNNNN` permalink, and a SHA-256 over the block's tag-stripped body), and the attribution is *derived* from that table rather than typed per row. The three source reviews are josser (2011), Stirliz77 (2020) and ekidonaa (2025). An earlier revision typed the attribution in and six rows were wrong, crediting Reza Negarestani — a name that appears on the page only in the «Рецензии» bibliography, as the author of an external review *of* the book; the derivation makes that class of error unrepresentable, and `--verify-reviews` re-checks all twelve against the live page.
- [x] Chainlink #66 — narrative reconstruction (entity/event tracking over the AST): `analysis/narrative/` (`narrative_report.md` — the deliverable, `events.tsv` — one row per actor->verb->target event with a page citation and the #62 clause link, `actors.tsv`, `pairs.tsv`, `per_arc.tsv`, `synopsis_check.tsv` / `synopsis_elements.tsv` — the jacket-synopsis sanity check, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/narrative_reconstruction.py` + `work/test_narrative_reconstruction.py` (52 tests). 4,829 events over 5,613 clauses (the back-cover synopsis page excluded from the narrative layers); mentions are #30 glossary surfaces plus the disclosed 'the dog' / 'self' referents; verbs are the frozen #27 lexicon; the direction convention is #27/#30's does/done-to asymmetry with confidence priced by the mention-verb gap. Synopsis check: S1 supported, S2-S5 partial, every divergence disclosed — the book's hunting-for-the-grotesque-skulls-of-Sato-Corporation formula belongs to ToKAGE, not to Super Cherry 666 as the jacket copy has it.
- [x] Chainlink #63 — LLM glossing pipeline (five prompt strategies): `analysis/glossing/` (`glossing_report.md` — the deliverable, `glosses.tsv` — one row per (clause, strategy) with the verdict, the checks passed, the backend, the prompt and response hashes and the #62 baseline, `checks.tsv` — the four check rows per gloss with contract and detail, `agreement.tsv` — cross-strategy Jaccard per clause, `sample.tsv` — the stratified sample and why each clause was drawn, `prompts.tsv` — every prompt with its sha256, `diagnostics.tsv`, `summary.json`) + `work/glossing_pipeline.py` + `work/test_glossing_pipeline.py` (95 tests). Five strategies — `literal_operator`, `register_paraphrase`, `scene_summary`, `plain_english`, `entity_consistent` — gloss a **stratified sample of 5 of #33's 5,614 clauses** drawn by a published rule (residue → operator-dense → entity-named → quoting-frame → plain-control, one per stratum, so each check has at least one case where it can fail, tie-broken on the clause id as text so `1003` precedes `101` — disclosed, and *not* the numerically earliest clause of each stratum, because re-drawing the sample to book order would move every prompt and need a fresh capture), **25 glosses**. The glossary and the operator grammar are **constraints, not hints**: every prompt carries only the operators that clause actually contains, with their #12/#14 readings, the #30 rows for the entities it names (role, does/done-to, frequency), its #62 unresolved surfaces and its #19 `ocr_error` artefacts, plus one standing constraint block identical across all five strategies so a differing honesty clause cannot confound the agreement statistic. **The load-bearing output is the verdict layer, not the glosses:** four deterministic checks — `operator_fidelity` (the operator subsequence, read off #17's token stream rather than a regex over prose, so `soul/gram` cannot pass as `psyche/gram`), `entity_fidelity` (a dropped #30 name *and* an invented proper noun are separate failures), `residue_discipline` (a marked-unresolved surface smoothed into an ordinary word), `frame_preservation` (the `<< >>` quoting frame) — each judged against its own strategy's contract (`scene_summary`/`plain_english` are told to drop or verbalise operators, so for them a *printed* operator is the violation and an empty channel is compliance), and none of them calls the model. **Verdicts: 15 pass / 10 fail; `operator_fidelity` 17/25 (68%), `entity_fidelity` 23/25 (92%), `frame_preservation` 6/6, `residue_discipline` 5/5; mean cross-strategy Jaccard 0.364, no unanimous clause — the disagreement localises the clauses #64's committee should be pointed at.** The dominant failure is instructive: the two strategies told to reproduce the
operator channel verbatim (`literal_operator`, `entity_consistent`) fail it on
3 of 5 clauses each, and 6 of the 8 `operator_fidelity` failures are *added*
operators — a model asked to copy a channel verbatim annotates around it
instead, inserting bracketed readings between the operator characters it was
meant to preserve. Determinism is handled honestly: the model is not deterministic (the same prompt was run three times and returned three different glosses), so the artifacts are **pinned** — one dated capture run (2026-10-10, `opencode run --model opencode/space-bunny-free`), each pin keyed by the sha256 of the prompt it answered so a changed constraint assembly fails the build rather than silently reusing a stale response — while `--live` re-runs the backend and prints a *diff* against the pins instead of rewriting them. No number comes from a live run.
- [x] Chainlink #64 — multi-agent translation committee with disagreement log: `analysis/committee/` (`disagreement_map.md` — the deliverable, `disagreement_log.tsv` — one row per localised disagreement with its classification, `readings.tsv` — the critic's per-reading facts, `pairs.tsv` — per-pair metrics, `diagnostics.tsv`, `summary.json`) + `work/translation_committee.py` + `work/test_translation_committee.py` (53 tests). #63's five prompt strategies are the committee's **agents**; the harness runs a **generator / critic / comparer loop** over their 25 pinned readings: the generator replays #63's pins (cross-checked against the published `glosses.tsv`), the critic re-judges every reading with the four deterministic checks recomputed from the frozen artifacts (cross-checked against the published `checks.tsv` — a drift is a build error), and the comparer localises every pairwise divergence to surfaces (each operator kept/dropped/added, each #30 entity carried/dropped, each #62 residue surface kept/smoothed, the content-word symmetric difference) and **classifies** it. Four classes: `rule_violation` (the rule governing the surface is breached — the responsible agent is named), `contract_divergence` (both readings did what their differing instructions required — keep vs drop the operator channel), `genuine_ambiguity` (both readings pass every check that applies to them and no instruction explains the difference), `defect_unattributed` (a participant failed some other check — recorded, not silently counted as ambiguity). **136 rows over 50 pairs: 60 rule violations, 28 contract divergences, 19 genuine ambiguities, 29 unattributed defects. Headline: 4 of the 5 sampled passages are genuinely ambiguous** (clause 10 `sucks` vs `draw in`; clause 101 all five compliant yet all ten pairs divergent; clause 1005 the added `crucified`; clause 2789 the OCR `l`/`I`) — while clause 1003, with only one passing reading, has **none**: every divergence there is a defect or a contract. A row is classified by what the readings did *on that surface*, never by a verdict elsewhere — a keep-strategy that kept `<<` as told and failed `/` on another row is a contract divergence on `<<`, not a violation.

- [x] Chainlink #52 — operator-word mutual information: `analysis/operator_word_mi/` (`operator_domain_mi_report.md` — the deliverable, `mi_matrix.tsv` — the MI matrix, `operator_profile.tsv`, `operator_seed_terms.tsv`, `mi_summary.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/operator_word_mi.py` + regression test `work/test_operator_word_mi.py` (60 tests). Converts the operator grammar into statistics: every one of **10,268 screened operator occurrences** (#17's 11,000 minus 732 raw-HTML `<`/`>` tokens, reconciling #10's 10,949) is resolved on the unit layer to the term it stands next to on each side (a free operator takes the nearest surviving unit surface; an in-unit operator takes its own operand substrings, so `=` in `suck=blood` has `suck` left and `blood` right), and those terms are matched against #29's six seeded lexical domains. **Verdicts: the operator predicts the adjacent term strongly (I(O;T) = 1.1938 bits right / 1.2013 left against a 200-replicate label-permutation null of 0.5488 / 0.695, z = 124.8 / 90.1; H(T) 7.39 → H(T|O) 6.20 bits)** — 9x #11's operator-to-operator lag-1 MI (0.1305 bits), which is what #12's role classification asserted and #11's entropy curve could not see. **The issue's own two hypotheses split: `::` before body terms is *rejected*** (P(body|`::`) 2.2% vs 5.7% base, lift 0.39, q < 1e-4 — `::` prefers machine and genetics terms), while **`=` before tech terms is confirmed** (P(machine|`=`) 14.2%, lift 2.49, q < 1e-4) with an even stronger body preference (lift 2.94) and a *genetics* left side (lift 1.76): `=` is the book's cross-domain naming binder (`suck=blood`, `fear=cell`, `gene=TV`). `<<` opens blocks of flesh (body lift 1.72, q = 0.0043, driven by `entrails` 30x at PMI 3.48) while `>>` shows no closing preference. Robustness: the domain totals are insensitive to the compound-part matching rule (0.0776 vs 0.0777 bits) and stable across book halves. Disclosed limits: #29's seed sets cover only 16% of right-hand neighbours, so the matrix is about the domain-bearing minority; multi-domain terms count as incidences in each domain; rare operators (`+` n=15, `|` n=17) are reported but not verdicted.

- [x] Chainlink #54 — self-similarity matrix (loop detection): `analysis/self_similarity/` (`self_similarity_report.md` — the deliverable, `paragraph_matrix.tsv` — one row per paragraph with its page and clause counts, `paragraph_cosines.tsv` — the most similar pairs, `page_matrix.tsv` — the 225 × 225 page rollup, `lag_curve.tsv` / `page_lag_curve.tsv` — the lag curves against a 200-draw order-permutation null, `band_stats.tsv` — in-band share vs the same null, `offdiag_blocks.tsv` — the block catalogue at both resolutions, `english_baseline.tsv`, `sensitivity.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`, `similarity.png`, `lag_curve.png`, `self_similarity.npz`) + `work/self_similarity.py` + regression test `work/test_self_similarity.py` (39 tests). The paragraph unit is a blank-line run of the #7 layout page body (733 runs plus 1 added so no clause is dropped = 734; 717 carry a #33 clause), every #33 clause is joined to exactly one paragraph by offset overlap on `(page, start, end)` (5,614 assigned, none dropped, 23 disclosed as spanning a break), and a paragraph's vector is the mean of its #39 clause vectors (`units` view, SBERT, renormalised) — no encoder runs for the book. **Verdict: the "loop" is a short diagonal band plus repeating blocks, and neither is a period.** The similar cells are not organised as a running diagonal — of the 9,578 paragraph pairs above cosine 0.8, 0.6% sit within one paragraph of the diagonal (null 0.3%) — and the band is real but short-range (lag-1 mean cosine 0.5648 against an order-permutation null of 0.5382, z = +10.8, significant to ~lag 30 and at the null from lag 40). The load-bearing long-range structure is a **verbatim return**: the opening section pages 4/5/6/7 recurs on 220/221/222/223 (page cosine 0.9550–0.9998, paragraph block mean 0.9488 over 16 cells, `difflib` raw-text identity 0.98, encoder-free lexical control 0.75 vs a 0.63 page background) — #35's page near-duplicate recovered from the embedding side with no n-grams. Nine further paragraph-scale blocks are thematic repeats (cosine ~0.87, text identity ~0.15) and there are **no template blocks at page scale** above the per-level 99.5th-percentile cut. Controls: a matched English paragraph baseline built the same way from #24's pinned cache (mean cosine 0.327 vs the book's 0.538; 0.03% vs 3.73% of pairs above 0.8), the encoder-free lexical cosine, and a view/encoder sensitivity pass (the block and the headline survive #39's `raw` view and its BGE encoder). Deterministic (fixed seed, 200 permutations, no wall-clock); rebuild byte-identical.

- [x] Chainlink #57 — intertextual n-gram matching vs the Siratori corpus: `analysis/intertext/` (`overlap_report.md` — the deliverable, `availability_report.md` — the source survey, `sources.tsv` — the acquisition ledger with provenance/reuse terms per source, `corpus_summary.tsv`, `overlap_by_n.tsv`, `null_control.tsv`, `formula_classes.tsv`, `shared_formulas.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/intertext.py` + regression test `work/test_intertext.py` (52 tests). **Honest split: the named-work half is a documented negative, the idiolect half is a positive.** All five works the task names (*Headcode*, *Human_Worms*, *Smart-d*, *GENEDUB*, *Nonexistence*) are in-copyright print editions — the only reachable copy is a borrow-only Internet Archive scan of *Headcode* (`access-restricted-item=true`) — so **0 of 5 named works is ingested** and no other text silently stands in for them. The one genuine second Siratori text available under the project's research-quotation convention is a different, freely published piece (*Great Works*, first three paragraphs reused, attributed), used as an idiolect probe. The matcher is n-gram overlap (n=1–6) over the #17 word-part layer screened exactly as #56, with a **shuffle null** (5 permutations, seed 57) so every count is read against chance. **Verdict:** of 25,969 book formulas (n≥2, freq≥2), **116 are `author_idiolect`** — present in the author's own other text and in *no* general/technical register — and the signal is far above chance (same-author probe n=3: 97 observed vs 7.2 expected, 13.5×; n=4: 36 vs 0.4). The signature formulas are `soul gram`, `cold blooded disease`, `retro adam`, `brain universe`, `dna channel`, `data mutant`. The **genre side is underpowered by construction** (the #56 influence excerpts are 49–105 tokens): no influence arm shares a trigram with the book and the only non-generic shared bigram is `a virus`, so genre-vs-English cannot be separated at that size — reported as a bound, not a finding. **Provenance is verified, not asserted:** the embedded *Great Works* excerpt is pinned by token count and sha256 and `--probe` re-fetches the page to check it, every reference-corpus file is checked against #56's manifest sha256, and the one interviewer turn in #58's Q/A table is excluded from the author arm — so no comparison text is invented or silently substituted.

- [x] Chainlink #59 — entity roles from the jacket synopsis, verified against the text: `analysis/entity_roles/` (`roles_report.md` — the deliverable, `roles.tsv` — the role table with a verdict and citations per entity, `role_tests.tsv` — one row per named probe with its observed count and threshold, `refutations.tsv` — the non-evidenced roles and the probes that failed them, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/entity_roles.py` + regression test `work/test_entity_roles.py` (47 tests). Nine roles are read **only** off the back-cover synopsis (p224; the five sentences come from #66's single source of truth) and tested against the book text pages 1–223 — the synopsis page is excluded from its own check. Each role is a set of **named probes** (presence, recurrence, centrality, the #30 role category, a clause co-occurrence, or an actor/verb event), each *defining* or *supporting*; the mention/clause/event layer is recomputed with #66's primitives over the frozen #17/#33/#30/#27 artifacts. A role is `evidenced` when every defining probe passes, `uncertain` when some do, and `refuted` when no substantive probe does — presence alone is a precondition, never a role. **Verdict: 6 evidenced / 2 uncertain / 1 refuted.** The corporate, civic and planetary roles all hold — Sato Corporation is an `organization` whose napalm+torture / torture+victim / vampire+chaos clusters fire (`p83:415`, `p79:1396`), Cadaver City and Placenta World are `place`s, and **EVOL is the pink-ash planet the jacket says** (`The planet of the angel mechanism of the ADAM Doll is EVOL`, `p77:111`; `pink ash planet EVOL`, `p153:2045`). The two uncertainties are the synopsis's *predicates*, not its entities: ANTi-ADAM is the only entity #30 calls an `antagonist` and it co-occurs with ADAM once (`p216:1132`), but **no clash event links them**; and the dog is a recurrent, violent actor (459 mentions, 72 hostile events, 32 dog+zero clauses) but **no `wage`-family verb stands with it** anywhere in the book. **The refutation is Super Cherry 666**: every substantive property fails — it is a `concept`, never a succubus form, never takes a hunt verb, and never stands with the corporation — while the hunting formula itself is real and belongs to ToKAGE (`p147:70`), so the jacket copy attaches the formula to the wrong entrant.

### First-pass headline numbers (full book)
| Metric | Value |
|---|---|
| word tokens | 74,760 |
| unique types | 5,289 (TTR 0.071; 58.5% hapax) |
| token entropy | 8.10 bits |
| gzip ratio | 0.271 (English prose ≈ 0.35–0.45) |
| operators | 10,949 (`::` 3,083 · `//` 2,149 · `=` 2,047 · `/` 1,484) |
| silence-removal effect | entropy 8.10→8.02, gzip 0.271→0.257 |

## The Chainlink plan

The methodology is decomposed into a trackable issue graph:

```bash
chainlink issue tree          # 1 program epic -> 10 phase epics -> 65 tasks
chainlink issue ready         # next unblocked work
chainlink issue next          # suggested next task
```

- `docs/chainlink-plan.md` — human-readable export (all 76 issues with descriptions)
- `docs/chainlink-issues.json` — machine-readable export
- `docs/chainlink-plan.json` — logical key → issue ID map (E1.1 = #10, etc.)
- `work/build_chainlink_plan.py` — reproducible builder (idempotent via `be-plan` label)

Phase epics: **E0** ingest · **E1** operator grammar · **E2** lexicon · **E3** distribution ·
**E4** syntax · **E5** themes · **E6** signal/Beguš · **E7** intertext · **E8** gloss ·
**E9** evaluation. Blockers chain the phases and key tasks (e.g. silence-removal #50 waits on
the operator inventory + perplexity baselines).

## Reproduce the OCR

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate marker
cd /mnt/sharedOs/blood-electric-workspace
TORCH_DEVICE=cuda marker work/input --output_dir docs/markdown \
  --disable_image_extraction --workers 1
```

## Reproduce the first-pass analysis

```bash
conda activate marker
python work/analyze.py docs/markdown/blood-electric/blood-electric.md analysis/first_pass
```

## Reproduce the silence-removal variants (#13)

```bash
conda activate marker
python work/silence_removal.py                    # writes analysis/silence_removal/
python -m unittest discover -s work -p 'test_*.py'
```

## Reproduce the operator grammar AST (#14)

```bash
conda activate marker
python work/operator_grammar.py                   # writes analysis/block_parser/
python work/test_operator_grammar.py
```

## Reproduce the operator-aware tokenizer (#17)

```bash
conda activate marker
python work/operator_tokenizer.py                   # writes analysis/tokenizer/
python work/test_operator_tokenizer.py
```

## Reproduce the neologism screen (#19)

```bash
conda activate marker
python work/neologism_detection.py                   # writes analysis/neologism/
python work/test_neologism_detection.py
```

## Reproduce the case / emphasis inventory (#22)

```bash
conda activate marker
python work/case_emphasis.py                        # writes analysis/case_emphasis/
python work/test_case_emphasis.py
```

## Reproduce the frequency / Zipf analysis (#24)

```bash
conda activate marker
python work/frequency_zipf.py                        # writes analysis/frequency/
python work/test_frequency_zipf.py
```

The eight English baselines are downloaded once into
`analysis/frequency/reference_cache/` and verified against pinned SHA-256
hashes (`reference_manifest.tsv`). Use `--no-download` to require the cache and
`--no-reference` to run the book-only path.

## Reproduce the reference corpora and keyness (#56)

```bash
conda activate marker
python work/reference_corpora.py                        # writes analysis/reference_corpora/
python work/keyness.py                                  # writes analysis/keyness/
python work/test_reference_corpora.py
python work/test_keyness.py
```

The corpora are seven reference registers, each with a provenance row in
`analysis/reference_corpora/manifest.tsv` (source, licence, SHA-256, token
count — no wall-clock field, so the build is byte-for-byte reproducible given
the same downloads):

| corpus | contents | tokens |
|---|---|---:|
| `general_english` | the eight pinned #24 public-domain baselines + 7 more novels (Austen to Kafka) | 1.61 M |
| `medical` | 1,000 PubMed abstracts (`disease[Title]`, live E-utilities query) | 380 k |
| `legal` | Federalist Papers, Declaration of Independence, Magna Carta, *The Law*, *The Social Contract* | 350 k |
| `cs` | 500 arXiv abstracts (cs.AI / cs.LG / cs.CL) | 103 k |
| `cyberpunk` | Gibson / Stephenson / Sterling — **fair-use excerpts** | 105 |
| `cutup_surrealist` | Burroughs / Artaud / Breton — fair-use excerpts | 56 |
| `finnegans_wake` | Joyce — fair-use excerpts (*Finnegans Wake* is in US copyright to 2034) | 49 |

The three excerpt corpora are short **by necessity**: those authors are in
copyright, so only attributed quotations are used. They are far too small to
carry a keyness test, and the harness says so rather than hiding it — every
table is written, the report carries a corpus-size caveat, and the
cross-corpus stability test is computed over the four large corpora only.

`--no-download` rebuilds from cache (the live PubMed / arXiv queries are
skipped and reported, never silently dropped); every write goes through a token
gate, so a 404 page or an empty efetch body raises instead of replacing a good
corpus.

**Statistics.** Book side = the #17 *word-part* layer, markup-screened with
#19's ledger + #27's tag-run detector (the unit layer fuses compounds and the
token layer adds operators, so neither is comparable to ordinary English
words). Reference side = the #24 English word regex. Per type in the union:
signed log-likelihood G² (Dunning 1993; Rayson & Garside 2000), chi-square p,
a **Benjamini–Hochberg q** and a 0.5-smoothed log-ratio. The gate is
**q < 0.05**, not p < 0.05 — the union runs to tens of thousands of types, and
uncorrected p admits hundreds of false positives by construction. `p` is still
written per word so the correction is visible.

**Verdict.** 32 words are in the top 50 against *every* large reference —
the book's own lexicon (`cadaver`, `fecundates`, `flesh`, `organ`, `embryo`,
`roid`, `gram`) and its designator channel (`adam`, `doll`, `drone`, `vital`),
plus the function-word loop `of`/`the`/`that` that #24 measured. 13 of the 944
general-English keys survive no technical register and are reported as
reference residue: they are artifacts of comparing against 19th-century prose.
Reproduce (`python work/keyness.py`) and read
`analysis/keyness/keyness_report.md` for the per-corpus key / anti-key tables.

## Reproduce the intertextual n-gram matching (#57)

```bash
conda activate marker
python work/intertext.py            # offline, deterministic; writes analysis/intertext/
python work/intertext.py --quick    # sampled register corpora + 1 shuffle (regression suite)
python work/intertext.py --probe    # re-run the live HTTP source probes (writes nothing)
python work/test_intertext.py
```

The build is offline and byte-identical on a rerun: it reads only frozen
artifacts (the #17 word-part layer screened as #56, the #56 reference corpora,
#58's pinned author-paratext quotations) and the one embedded, attributed
excerpt of a freely published Siratori text. `--probe` re-requests each source
URL and prints a diff against the pinned statuses in `sources.tsv`; it never
writes. The acquisition ledger records the five named works as print-only /
searched-not-found with their evidence — none is ingested, and no other text
stands in for them.

## Reproduce the entity-role table (#59)

```bash
conda activate marker
python work/entity_roles.py         # offline, deterministic; writes analysis/entity_roles/
python work/test_entity_roles.py
```

The nine roles are seeded from the back-cover synopsis only, then measured
against the book text (pages 1–223; p224 is excluded from its own check) with
#66's mention/clause/event primitives over the frozen #17/#33/#30/#27 layers.
Every verdict is computed from counts, every evidenced role carries the
citations that earned it, and the build reconciles its re-derived counts
against #66's published tables before it writes. Rebuild is byte-identical
except the generated stamp.

## Reproduce the multi-level entropy analysis (#47)

```bash
conda activate marker
python work/entropy_analysis.py                      # writes analysis/entropy/
python work/test_entropy_analysis.py
```

The builder reads only frozen artifacts (`analysis/tokenizer/units.tsv` via the
#19/#27 screen, `analysis/operator_ngrams/operator_stream.tsv`, the #7 layout
bodies) and the #24 English reference cache — no downloads, no OCR. `--quick`
runs a deterministic page sample for the regression suite; `--sample N` controls
the book page count.

## Reproduce the compression / MDL analysis (#48)

```bash
conda activate marker
python work/compression_mdl.py                       # writes analysis/compression/
python work/test_compression_mdl.py
```

Reads the #7 layout bodies, the #17 screened unit layer, the #13 frozen
silence-removal variants and the #24 English reference cache — no downloads, no
OCR. `zstd` uses the `zstandard` package when present and falls back to the
`zstd` CLI (disclosed in `diagnostics.tsv`); `--quick` runs a small matched-N
subset for the test suite.

## Reproduce the periodicity / spectral analysis (#51)

```bash
conda activate marker
python work/periodicity.py                           # writes analysis/periodicity/
python work/test_periodicity.py
```

Reads only frozen artifacts (the #17 `tokens.tsv` stream and its lossless
character reconstruction, the #33 `clauses.tsv` lengths, the #11
`operator_stream.tsv` for the identity check, #19/#27 for the markup screen, and
the #24 English reference cache) — no downloads, no OCR. Needs `numpy`; plotting
needs `matplotlib`. `--quick` runs a deterministic 40-page sample with 40
shuffles for the regression suite; `--sample N` sets the page count, `--shuffles
R` the surrogate count, `--no-english` skips the baseline, and `--out` redirects
the output directory. Shuffles use a fixed seed (42), so every TSV is
reproducible byte-for-byte.

## Reproduce the within-text language viability assessment (#56)

```bash
conda activate marker
python work/language_viability.py            # writes analysis/language_viability/
python work/test_language_viability.py       # regression suite
```

Reads **only frozen artifacts** — the #17 token stream (`units.tsv`, `tokens.tsv`), the #19 markup screen, the #30 entity glossary, the #33 clause index, the #34 frame miner and the #11 operator stream — so there are no downloads, no OCR and no reference corpus. This is the revised #56 scope: intrinsic viability, not external keyness (the earlier `analysis/keyness/` + `analysis/reference_corpora/` output is retained as history and is not consulted).

**Design.** The book is split and every claim is a falsifiable check on the held-out half. The primary split is *interleaved* (even pages train, odd pages test) so neighbouring pages cannot leak style; a chronological split is reported as a drift control. Each check is scored against **within-text nulls** that preserve the book's own marginals but destroy the structure under test (within-page / within-run / within-clause / within-unit shuffle, and a random vocabulary bijection), with add-one empirical p-values over 40 draws and a fixed seed. A **negative control** runs the same machinery on a word-order-shuffled copy of the book and must *fail*, which is what makes the passes non-vacuous.

**Verdict.** VIABLE (BOUNDED), 6 of 6 pre-registered checks pass (`checks.tsv` for every statistic, its nulls and its p-value). The order sweep gives a small held-out gain from order-1 to order-2 (8.2619 to 8.2366 bits/item, a 0.3% reduction); order-3 does not improve (8.2799). This 0.0253-bit edge has no separate uncertainty test, so the evidence supports short-context structure, not a long-range grammar — alongside #11, #47 and #51.

## Reproduce the coverage accounting / residue ledger (#70)

```bash
conda activate marker
python work/coverage_ledger.py            # writes analysis/coverage_ledger/
python work/test_coverage_ledger.py       # regression suite
```

Reads **only frozen artifacts** — #7's layout markdown, #17's token/unit streams, #19's screen and markup ledger, #27's tag-run detector and the published `summary.json` of every downstream layer — so there is no OCR, no download and no reference corpus (#78–#80 stay parked).

**Question.** Every analysis in this workspace reads *some* layer and every screen removes occurrences, each justified in the removing issue's own reconciliation table. Nobody had asked whether the whole corpus is accounted for. This ledger places every character, token, operator token and unit occurrence in a named bucket and forces each of the four levels to close.

**Method.** The page body is rebuilt from #17's artifacts alone (`"".join(gap + text)` + `page_tails.tsv`) and both directions are asserted before the ledger is built — the reconstruction equals #7's layout markdown on all 225 pages, and #17's `tokens.tsv` equals what `operator_tokenizer.tokenize_page` emits. Operator roles are re-derived with #17's own `_is_seam` rule rather than read from its summary, so the published partition is tested, not assumed; unit categories are joined per-occurrence through #19's uid-level markup ledger, never by surface.

**Coverage metric.** The share of a level carrying no unexplained residue, which is 1.000000 at all four levels here and a build failure if it ever drops. A metric that cannot fail needs teeth, so the tests plant residue — a token the stream drops, an unscreened unit surface, a consumer split that no longer closes — and require the detector to fire.

**Verdict.** The analysis accounts for everything, and the awkward residue is named rather than hidden: 11 classes in `residue_ledger.tsv`, each with its owner, pages and page-cited examples, verdicted `explained` (a published artifact names the items), `disclosed` (counted, itemized nowhere) or `unexplained` (nothing accounts for it — there are none). The two classes this ledger itemizes for the first time are the 13 book-text occurrences deleted by #27's tag-run screen and the 2 unit surfaces #17 and #19 escape differently. The ledger names and counts; it repairs nothing and re-opens no decision.

## Reproduce the KWIC concordancer (#26)

```bash
conda activate marker
python work/kwic_concordancer.py                        # writes analysis/kwic/
python work/test_kwic_concordancer.py
```

The concordancer reads only frozen artifacts (`analysis/tokenizer/`,
`analysis/neologism/`, `analysis/case_emphasis/`,
`analysis/lexical_correction/gazetteer.tsv`,
`analysis/numeric_channel/`) and the layout markdown — no downloads, no OCR.

## Reproduce the collocation / association analysis (#27)

```bash
conda activate marker
python work/collocation.py                          # writes analysis/collocation/
python work/test_collocation.py
```

## Reproduce the operator-word mutual information (#52)

```bash
conda activate marker
python work/operator_word_mi.py                     # writes analysis/operator_word_mi/
python work/test_operator_word_mi.py
```

## Reproduce the self-similarity matrix (#54)

```bash
conda activate marker
python work/self_similarity.py                      # writes analysis/self_similarity/
python work/test_self_similarity.py
```

## Reproduce the parse-attempt / derangement curve (#32)

```bash
conda activate marker
python work/parse_attempt.py                        # writes analysis/parse_attempt/
python work/test_parse_attempt.py
```

The parse-attempt builder reads the #13 silence-removal streams and the #24
English reference cache. It needs `spacy` (`en_core_web_md`) and, for the
independent replication, `stanza` (English `tokenize,pos,lemma,depparse`).
`--quick` runs a small deterministic subset for the regression suite;
`--no-stanza` / `--no-english` skip the replication and the ceiling baseline.

The builder reads the #17 unit and token layers plus the #19 markup ledger —
no downloads. The verb-candidate lexicon is parsed from the Brown corpus
(`~/nltk_data/corpora/brown`) with stdlib only when that directory exists; the
frozen `analysis/collocation/verb_lexicon.tsv` is the fallback otherwise.

## Reproduce the frame / template mining (#34)

```bash
conda activate marker
python work/frame_mining.py                         # writes analysis/frame_mining/
python work/test_frame_mining.py
```

The frame miner reads only frozen artifacts — the #33 `clauses.tsv` boundaries,
the #17 `units.tsv` layer (markup-screened with #19's ledger + #27's tag-run
detector, joined on `(page, start)`) and the #14 `spans.tsv` block layer — no
downloads, no OCR. `--out` redirects the output directory; the suite runs the
miner into a temp directory.

## Reproduce the clause / sentence embeddings (#39)

```bash
conda activate marker
python work/clause_embeddings.py                     # writes analysis/clause_embeddings/
python work/test_clause_embeddings.py
```

The builder reads only frozen artifacts (the #33 `clauses.tsv` boundaries, the
#17 `units.tsv`/`pages` layer markup-screened with #19's ledger + the #27
tag-run detector, and the #24 English reference cache) — no downloads, no OCR.
It needs `torch` + `transformers` and the cached SBERT weights
(`sentence-transformers/all-MiniLM-L6-v2`; E5 via `--model e5` would download).
`--quick` embeds a deterministic 200-clause sample for the regression suite;
`--sample N` sets the sample size, `--no-secondary` skips the BGE robustness
encoder, `--no-english` skips the English control, and `--out` redirects the
output directory. `ClauseEmbeddingStore` loads the persisted vectors:

```python
from clause_embeddings import ClauseEmbeddingStore
store = ClauseEmbeddingStore()            # analysis/clause_embeddings/
v = store.vector(1234, view="units")      # 384-d, L2-normalised
hits = store.nearest(1234, k=10)          # [(clause_id, cosine), ...]
page_vecs = store.page_matrix(4)          # all clauses on page 4
```

## Reproduce the topic modelling (#41)

```bash
conda activate marker
python work/topic_modelling.py                       # writes analysis/topics/
python work/test_topic_modelling.py
```

The builder reads only frozen artifacts (the #39 `clause_index.tsv` document
view, the #39 `clause_embeddings.npz` vectors for the BERTopic-style model, and
the #29 `communities.tsv`/`summary.json` for the comparison) — no downloads, no
OCR. It needs `scikit-learn` + `gensim`; the BERTopic-style model needs
`umap-learn` + `hdbscan` (the `bertopic` package itself is not required, and
`--no-umap` falls back to a deterministic `TruncatedSVD` reduction). `--k-grid`
sets the coherence scan grid, `--bert-min-topic-size` / `--bert-mts-grid` set
the BERTopic-style granularity, and `--out` redirects the output directory.

## Reproduce the triangulation / convergence report (#73)

```bash
conda activate marker
python work/triangulation.py                        # writes analysis/triangulation/
python work/test_triangulation.py
```

Reads only frozen artifacts — #29 `graph_nodes.tsv`, #39 `clause_index.tsv` +
`clause_embeddings.npz`, #41 `clause_topics.tsv` + the topic summaries +
`community_ari.tsv`, #40 `clusters.tsv` + `motif_checks.tsv`, #51
`periodicity_tests.tsv`, and #29's `DOMAIN_SEEDS` (read out of
`work/cooccurrence_network.py` with `ast.literal_eval`, so they cannot drift) —
no downloads, no OCR, and **no reference corpus**: convergence is assessed only
among analyses of this book, against internal nulls (label permutations and
within-book shuffles).  Needs `numpy` only — `scikit-learn` / `hdbscan` /
`umap` are never imported, so the report rebuilds even where #40 and #41 cannot.
`--out` redirects the output directory, `--permutations` / `--shuffles` resize
the nulls, `--n-init` the k-means restarts.  Fixed seed (42) everywhere and no
wall-clock in any artifact, so every TSV is byte-reproducible.

## Reproduce the static embeddings (#28)

```bash
conda activate marker
python work/embedding_models.py                      # writes analysis/embeddings/
python work/test_embedding_models.py
```

The merged corpus is the eight pinned Gutenberg texts from #24 plus the Brown
corpus (both local, no download). `--reuse-models` reloads the saved vectors and
regenerates only the tables/report; `--no-merged` builds the book-only models;
`--quick` builds tiny book-only models for the regression suite. Requires
`gensim` for word2vec/fastText (GloVe is a self-contained numpy implementation).

## Reproduce the Rosetta mapping (#15)

```bash
conda activate marker
python work/rosetta_mapping.py                       # writes analysis/rosetta/
python work/test_rosetta_mapping.py
```

The notation signatures are spec-based and live in `work/rosetta_mapping.py`;
the permutation test defaults to 2,000 shuffles (pass an integer to override).
The book side is measured from the #10 operator inventory and #12
classification, and the scorecard is reproduced against both.

## Reproduce the translation oracle (#58)

```bash
conda activate marker
python work/translation_oracle.py                   # writes analysis/translation_oracle/
python work/translation_oracle.py --probe           # re-run the HTTP source probes live
python work/translation_oracle.py --verify-reviews  # re-check every quotation's attribution live
python work/test_translation_oracle.py
```

The build is offline and deterministic: the acquisition evidence is a *pinned
record* of a dated probe run (`PROBE_DATE`), so nothing in the artifacts moves
with the clock, and `--probe` re-runs the HTTP requests and prints a diff
against the pins. `--verify-reviews` re-fetches the FantLab work page, re-reads
each review's author and `datePublished` from the page's own
`schema.org/Review` markup, and confirms each quotation's text still occurs in
the review it is attributed to — so the attribution is checkable against the
source rather than trusted. The only prerequisites are #7's layout markdown
(`analysis/layout_recovery/`) and #19/#30/#62's gloss tables; the aligner, the
gold and the AER harness are entirely self-contained.

The ingestion gate is the point of the negative half: unauthorized mirrors of
the Skobin translation are probed with `HEAD` only, to establish that they
exist, and are refused. What *is* ingested is short attributed research
quotations of the translation published inside third-party reader reviews, the
same convention `work/reference_corpora.py` applies to Gibson and Burroughs.
Because the attribution is what makes a quotation of a copyrighted work
publishable, it is pinned to the source review rather than typed: every
quotation names a block in `fantlab_reviews.tsv` (author, account,
`datePublished`, permalink, body SHA-256) and its attribution is derived from
that table. If a licensed copy of the Russian edition is ever dropped into the
repository, rerunning the module consumes it unchanged: the skeleton localiser
that aligns eleven quotations would then align all 225 pages.

## Reproduce the ensemble OCR (#4)

```bash
conda activate marker
# heavy: ~25 min over 225 pages at 300 dpi; run in the background and poll
nohup python work/ensemble_ocr.py ocr --engines tesseract,doctr,easyocr \
    > analysis/ensemble_ocr/ocr_run.log 2>&1 &
python work/ensemble_ocr.py build                   # writes analysis/ensemble_ocr/
python work/test_ensemble_ocr.py
```

The per-page engine outputs are cached in `analysis/ensemble_ocr/engines/*.jsonl`
and the `ocr` step is resumable (it skips cached pages; `--force` re-runs).
`--page-range LO-HI` limits either step for a quick check. docTR and easyocr
weights download once on first use.

## Reproduce the LLM glossing pipeline (#63)

```bash
python work/glossing_pipeline.py            # pinned, offline; writes analysis/glossing/
python work/glossing_pipeline.py --live     # re-run the backend, diff vs the pins (writes nothing)
python work/glossing_pipeline.py --capture  # print a fresh PINS literal to stdout (writes nothing)
python work/test_glossing_pipeline.py
```

The default build is a **pinned replay**: it answers each sample clause with the
recorded response of one dated capture run (`CAPTURE`, `PIN_BACKEND`), and the
pins are keyed by the sha256 of the prompt they answered — so if the
constraint assembly changes, the hash stops matching and the build fails
loudly instead of quietly reusing a response to a prompt that no longer
exists. `--live` and `--capture` are the two override paths and neither of
them writes to `analysis/glossing/`: a live run **prints a diff against the
pins** (the model is not deterministic — the same prompt, three runs, three
different glosses), and `--capture` prints the new pin block as a Python
literal so adopting it is a reviewable edit rather than a side effect of
running a command. Prerequisites are the frozen artifacts only (#17 tokens,
#33 clauses, #62's interlinear join and gloss tables, #30's glossary).
`--per-stratum N` widens the sample if more glosses are wanted.

## Reproduce the translation committee (#64)

```bash
python work/translation_committee.py      # pinned, offline; writes analysis/committee/
python work/test_translation_committee.py
```

The committee calls no model: the generator phase replays #63's pins
through #63's own `resolve_glosses` (so it inherits #63's prompt-hash
pinning), and the critic and comparer are pure functions of the frozen
artifacts. Every phase cross-checks #63's published tables — the replayed
glosses against `glosses.tsv` (text, verdict, operator columns, response
sha256), the recomputed checks against `checks.tsv` — so the committee
stands on the published record and fails loudly when it stops agreeing
with it. Prerequisites are #63's artifacts (`analysis/glossing/`) and,
through #63's loaders, #17/#33/#30/#62.

## Reproduce the spectral reading (#67)

```bash
python work/spectral_reading.py          # writes analysis/spectral_reading/
python work/test_spectral_reading.py
```

Prerequisites are #13's published streams (`analysis/silence_removal/`,
regenerate with `python work/silence_removal.py`) and #50's
`perplexity_by_variant.tsv`; the build reconciles against both and fails
loudly on a mismatch. The suite pins the derivation, the reconciliation,
the framing-note contract, the worked example and byte-identical
determinism (57 tests).

## Reproduce the human jury / reader study design (#75)

```bash
python work/reader_study.py            # writes analysis/reader_study/
python work/test_reader_study.py       # regression suite, 62 tests
```

The study is designed here, not run: no human rating exists in this
workspace, and every pilot number is a disclosed simulation (seed 75)
labelled `simulated = True` in every artifact. `protocol.md` is the
pre-registration — the question, the population, the design, the
instrument verbatim, the analysis plan, the stopping rule and what the
study cannot establish. `frame.tsv` is the sampling frame: all 45 cells
over #63's stratified 5-clause sample, published whether a block draws
them or not. `blocks.tsv` is the counterbalanced packets.

The instrument is 7 items, of which two are planted traps carrying #63's
`plain_english` reading, and the design point is that a reader never
rates a gloss without the clause it translates beside it. The analysis
pipeline is statistics the real run will use unchanged: bootstrap CIs,
Krippendorff's alpha with the interval metric against a re-grouping null,
a control-label permutation test for the within-clause contrast, a
label-permutation Spearman against #74's published fidelity, and
Kruskal-Wallis H with eta-squared. Two controls are build errors rather
than footnotes: the identity control must top its block, and the
attention check must be able to flag a reader. Prerequisites are #62's
interlinear, #63's pinned glosses and #74's fidelity scores — no model
backend is called.

## Key references

- Wenaus, A. *The Twilight of Information Illiteracy: Kenji Siratori, Blood Electric, and
  Asemic Cyberpunk.* — names the mode and the problem.
- Beguš, G. *The Next Theory of Evolution* (eScholarship, 2026) — the alien-signal method:
  reframe, remove silences, find hidden structure.
- Publisher synopsis / jacket copy — canonical macro-plot and entity names.
