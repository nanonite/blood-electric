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
  compound_decomposition.py            # compound/portmanteau decomposition + affix lexicon (#18)
  test_compound_decomposition.py       # regression suite (#18)
  neologism_detection.py               # OOV unit screen: coinages / compounds / OCR errors (#19)
  test_neologism_detection.py          # 32-test regression suite (#19)
  collocation.py                       # PMI/logDice collocations + entity-verb pairs (#27)
  test_collocation.py                  # 28-test regression suite (#27)
  parse_attempt.py                     # dependency-parse attempt + derangement curve (#32)
  test_parse_attempt.py                # regression suite (#32)
  embedding_models.py                  # static embeddings: w2v/fastText/GloVe (#28)
  test_embedding_models.py             # regression suite (#28)
  case_emphasis.py                     # ALL-CAPS inventory: headers / labels / screams (#22)
  test_case_emphasis.py                # 30-test regression suite (#22)
  frequency_zipf.py                    # type/token, hapax, Zipf vs English prose (#24)
  test_frequency_zipf.py               # regression suite (#24)
  rosetta_mapping.py                   # Rosetta scorecard vs real notations (#15)
  test_rosetta_mapping.py              # regression suite (#15)
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
- [x] Chainlink #48 — compression / MDL vs English baseline: `analysis/compression/` (`compression_report.md` — the deliverable incl. the method note, `compression_by_stream.tsv`, `english_baseline.tsv`, `mdl_split.tsv`, `silence_variants.tsv`, `per_page.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`, `compression_curve.png`) + `work/compression_mdl.py` + regression test `work/test_compression_mdl.py` (18 tests). Six compressors across four families (gzip/zlib, bz2, lzma, zstd-3/19), matched N, raw and letters-only alphabets, plus a length-robust MDL split and the #13 silence-removal variants. **Verdict — a negative result for the task's stated prior:** the book is *more* compressible than matched English prose (char gzip-9 2.14 vs 2.91 bits/char; letters 2.24 vs 3.11; token 12.7 vs 14.6 bits/word), stable across all six compressors and both alphabets, so "low compressibility = high alienness" is false here. Stripping the #13 operators makes the residue *even* more compressible, so the alienness is local (operator grammar #11, hapax tail #35), not aggregate; per-page compressibility is flat on body pages (sd 0.50 on mean 3.82) and only moderately correlated with #35 surprisal (r=0.68).
- [x] Chainlink #26 — KWIC concordancer for entities and terms: `analysis/kwic/` (`kwic_report.md`, `concordance.tsv` — 9,329 occurrences in ±5-token context, `term_summaries.tsv` — 2,341 terms with counts / attestation / surface variants / embedded-in-larger-unit, `summary.json`, `diagnostics.tsv`, `reconciliation.tsv`) + `work/kwic_concordancer.py` + regression test `work/test_kwic_concordancer.py` (20 tests). Term list = entities (#19 `proper_noun` + #22 `label` + #6 gazetteer `entity`) + suspicious terms (#19 `compound`/`true_coinage`/`ocr_error` + #21 numeric + #22 command register); #22 screams/headers/function excluded as ordinary English. Joins the #17 **unit** layer case-insensitively (never words, never token index — the `tokens.tsv` `unit` column is a per-page index, so context is located by a (page, start) containment join). **Verdict:** the concordance is the primary route to a term's stable sense — e.g. `TV` is only 7 standalone units but 175 word-parts inside compounds (`gene=TV` 142, `BABEL=TV` 16), and `ADAM` 303 + 25 token-embedded = 328 reconciles #22 exactly; two embedded views (part-level vs token-level) are both reported.
- [x] Chainlink #27 — collocation and association (PMI/logDice): `analysis/collocation/` (`collocation_report.md`, `collocations_bigram.tsv` — unit-layer bigrams with f_x/f_y/f_xy/N + PMI/PPMI/logDice/Dice/t-score, `collocations_trigram.tsv`, `collocations_bigram_token.tsv`/`collocations_trigram_token.tsv` — the token layer incl. operators, `entity_collocations.tsv`, `entity_verb_pairs.tsv`, `verb_lexicon.tsv`, `entity_inventory.tsv`, `summary.json`, `diagnostics.tsv`, `reconciliation.tsv`) + `work/collocation.py` + regression test `work/test_collocation.py` (28 tests; full `work/` suite 532, one disclosed pre-existing #18 error). Two sequences from #17 — the maximal **unit** layer (74,077 items after #19's 347 markup removals plus 241 more tag-furniture occurrences `href`/`mailto`/`colspan` the general tag-run detector catches) and the **token** layer (88,669 tokens incl. operators) — never token indices. **Verdict:** the content collocations are a tight motif set (`artificial sun` 367, `internal organ` 250, `drug embryo` 204, `cadaver city` 192, `ice nebula`/`nebula faecal` 179, `cold-blooded disease` 177, `sickly period` 149) over a function-word loop (`of the` 3,854); operator–word pairs are sharply selective (`::`→`the` 928, `//`→`the` 672, `/`→`gram` 326, `=`→`tv` 166, `=`→`serum` 160). Entity–verb: 85 entities / 1,869 occurrences / 1,848 verb pairs; `ToKAGE` does `is`/`was`/`fecundates` and is done to by `love`/`speed`/`like`/`crucified` — the same coinage verb `fecundates` on both sides. Verb flag is a disclosed *candidate filter* (10,242 Brown surface forms + 56 verb-diagnostic book forms + 275 ambiguous `-s` forms), not a parser.
- [x] Chainlink #28 — static embeddings and analogies: `analysis/embeddings/` (`embedding_report.md` — the deliverable, `neighbours_book.tsv` — every book type × 3 book models, `neighbours_selected.tsv` — 64 curated terms × 6 models, `analogies.tsv` — 103 probes × 6 models, `analogy_summary.tsv`, `analogies_english.tsv`, `vocabulary.tsv`, `model_summary.tsv`, `summary.json`, `diagnostics.tsv`, `reconciliation.tsv`, plus `w2v_book.kv`/`ft_book.kv`/`glove_book.npz` and the merged trio) + `work/embedding_models.py` + regression test `work/test_embedding_models.py` (26 tests). Six models (word2vec skip-gram, fastText subwords, and a self-contained numpy GloVe) trained on the #17 **unit** layer alone and on the unit layer + 1,997,635 English tokens (8 pinned Gutenberg texts + Brown). Contexts are the book's own utterances: 74,318 screened units over 2,345 `//`-delimited sequences; the #19 screen join is asserted (347 markup occurrences removed, no tag names as words). **Verdict:** fastText's subwords are the only way a coinage survives the merged corpus — w2v/GloVe cover 48.7% of the 5,502 book types at `min_count=5`, fastText 100.0%; on the book alone all three cover 100%. Pooled curated analogies (`X:Y::Z:?`) hit@10: `w2v_book` 40.7%, `glove_book` 37.0%, `ft_book` 25.0%, `glove_merged` 34.8% — the book-alone models recover the book's own naming/derivation equations, the merged models trade book signal for English neighbours, and English sanity analogies pass at 42.9% hit@10 (a small-corpus limit, reported plainly). Neighbours show `gene=tv`→`fecundates`/`boy-roid`, `tokage`→`vital=serum`, `adam`→`doll`/`anti-adam`, and fastText recovers the parts (`boy-roid`→`boy-roid=chaos`, `suck=blood`→`mobile=suck=blood`).
- [x] Chainlink #15 — Rosetta mapping against real notations: `analysis/rosetta/` (`rosetta_report.md` — the deliverable, `notation_signatures.tsv`, `operator_role_map.tsv`, `scores.tsv`, `per_operator_best_match.tsv`, `positional_diagnostics.tsv`, `structural_metrics.tsv`, `permutation.tsv`, `reconciliation.tsv`, `diagnostics.tsv`, `summary.json`) + `work/rosetta_mapping.py` + regression test `work/test_rosetta_mapping.py` (33 tests; full `work/` suite 408, one disclosed pre-existing #18 error). Scores the #10/#12 operator set against eight real notations in four frequency-weighted dimensions (role-symbol coverage 0.55, alphabet Jaccard 0.15, capability features 0.15, positional fidelity 0.15). **Verdict:** the operator *alphabet* is C-family — **C/C++** wins (total 0.75, role-symbol coverage 82.4%, exact symbol+role 64.9% of occurrences; its punctuation set contains 100% of the book's operator occurrences), then **Perl** (0.72) and **HTML/XML** (0.58); lowest **Regular expressions** (0.35). But the resemblance is **symbolic, not functional**: `//` is not a comment anywhere in the book (its closest analogue is the GenBank/EMBL `//` record terminator, role-compatible but line-anchored there vs inline here), `++` collides with C increment (coupling, never numerals), `<<`/`>>` are block quotes not I/O, and ordinary English punctuation — with **zero** symbol overlap — still expresses the *role* of 79.3% of operator occurrences. The book is **English punctuation roles encoded in C-family symbols**, i.e. a measured false friend of code. Permutation test (shuffled symbol→role, 2,000 shuffles) p=0.0005, z=3.1; winner stable under hyphens included, raw-HTML pages excluded, and rare operators dropped.
- [ ] Embeddings / topics / motif map.
- [x] Silence-removal (Beguš) transform variants (#13) — residue streams persisted; the perplexity version is still open (#50).
- [ ] Interlinear gloss + termbase + narrative skeleton.

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

## Key references

- Wenaus, A. *The Twilight of Information Illiteracy: Kenji Siratori, Blood Electric, and
  Asemic Cyberpunk.* — names the mode and the problem.
- Beguš, G. *The Next Theory of Evolution* (eScholarship, 2026) — the alien-signal method:
  reframe, remove silences, find hidden structure.
- Publisher synopsis / jacket copy — canonical macro-plot and entity names.
