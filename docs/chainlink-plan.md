# Chainlink Issues Export

Exported: 2026-10-02 22:18:50 UTC

## Open Issues

### [ ] #76: Anti-pattern guardrails audit

- **Priority:** high
- **Status:** open
- **Parent:** #69
- **Labels:** be-plan, eval, research
- **Created:** 2026-10-02

Audit for hallucinated coherence, LLM circularity (same model translating and judging), over-correction, single-method certainty, and residue-dropping. DONE: guardrail checklist + audit results, wired into CI-style checks.

---

### [ ] #75: Human jury / reader study design

- **Priority:** medium
- **Status:** open
- **Parent:** #69
- **Labels:** be-plan, eval, research
- **Created:** 2026-10-02

Have readers rate glosses for fidelity to the felt text; design the instrument and sampling. DONE: study protocol + pilot results if available.

---

### [ ] #74: Back-translation harness

- **Priority:** medium
- **Status:** open
- **Parent:** #69
- **Labels:** be-plan, eval, research
- **Created:** 2026-10-02

Gloss -> regenerate a Siratori-style clause -> compare to original. DONE: harness + fidelity scores.

---

### [ ] #73: Triangulation / convergence report

- **Priority:** critical
- **Status:** open
- **Parent:** #69
- **Labels:** be-plan, eval, research
- **Created:** 2026-10-02

Do independent methods (network community, topic model, embedding cluster, periodicity) converge on the same motifs/domains? Convergence is evidence; divergence is a lead. DONE: convergence matrix + report.

---

### [ ] #72: Predictive validity on held-out text

- **Priority:** high
- **Status:** open
- **Parent:** #69
- **Labels:** be-plan, eval, research
- **Created:** 2026-10-02

Does the derived grammar predict held-out operator sequences and collocations? Train 80% / test 20%. DONE: held-out scores.

---

### [ ] #71: Consistency checker

- **Priority:** high
- **Status:** open
- **Parent:** #69
- **Labels:** be-plan, eval, research
- **Created:** 2026-10-02

Verify the same symbol/term gets the same reading throughout; flag violations. DONE: checker + report over the termbase.

---

### [ ] #70: Coverage accounting / residue ledger

- **Priority:** high
- **Status:** open
- **Parent:** #69
- **Labels:** be-plan, eval, research
- **Created:** 2026-10-02

Does the analysis account for every token/operator, or ignore the awkward residue? Unexplained residue is the test's teeth. DONE: residue ledger + coverage metric.

---

### [ ] #69: E9 — Evaluation and epistemology

- **Priority:** high
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, eval
- **Created:** 2026-10-02

L9. No ground-truth translation exists; state what would count as evidence and audit the anti-patterns that would fool us.

---

### [ ] #68: Compression test for candidate readings

- **Priority:** medium
- **Status:** open
- **Parent:** #61
- **Labels:** be-plan, gloss, research
- **Created:** 2026-10-02

A good gloss shortens the description of the book; score candidate readings by whether they make the corpus more predictable and compressible. DONE: scoring harness + ranked readings.

---

### [ ] #67: Spectral reading artifact

- **Priority:** medium
- **Status:** open
- **Parent:** #61
- **Labels:** be-plan, gloss, research
- **Created:** 2026-10-02

Present the silence-removed stream as a distinct under-text reading, explicitly not 'the real text'. DONE: publishable spectral reading + framing note.

---

### [ ] #66: Narrative reconstruction (entity/event tracking)

- **Priority:** high
- **Status:** open
- **Parent:** #61
- **Labels:** be-plan, gloss, research
- **Created:** 2026-10-02

Track entities/events over the AST to recover who does what to whom (ADAM, ANTi-ADAM, Super Cherry 666, Sato Corp, the dog, 'self'); jacket synopsis is a sanity check, not a script. DONE: narrative skeleton + evidence links.

---

### [ ] #65: Termbase and consistency enforcement

- **Priority:** high
- **Status:** open
- **Parent:** #61
- **Labels:** be-plan, gloss, research
- **Created:** 2026-10-02

One term = one gloss (except where discrepancy is the point); a termbase prevents drift across 224 pages. DONE: termbase + consistency checker input.

---

### [ ] #64: Multi-agent translation committee with disagreement log

- **Priority:** medium
- **Status:** open
- **Parent:** #61
- **Labels:** be-plan, gloss, research
- **Created:** 2026-10-02

Generator/critic/comparer loop; disagreement localizes genuinely ambiguous passages. DONE: committee harness + disagreement map.

---

### [ ] #63: LLM glossing pipeline (multi-strategy)

- **Priority:** high
- **Status:** open
- **Parent:** #61
- **Labels:** be-plan, gloss, research
- **Created:** 2026-10-02

Prompt strategies: literal gloss preserving operators; register-normalized paraphrase; scene summary; plain-English translation; entity-consistent gloss. Feed the glossary and operator grammar as constraints. DONE: pipeline + sampled glosses.

---

### [ ] #62: Interlinear edition generator

- **Priority:** high
- **Status:** open
- **Parent:** #61
- **Labels:** be-plan, gloss, research
- **Created:** 2026-10-02

Three-line aligned rendering: original clause | operator-normalized (silence removed) | plain-English gloss. DONE: generator + first full-book rendering.

---

### [ ] #61: E8 — Interpretation, gloss and interlinear edition

- **Priority:** high
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, gloss
- **Created:** 2026-10-02

L8. Only after the machinery above generate readings - rigorously, with every interpretive move visible and overridable.

---

### [ ] #60: Genre/prior dossier

- **Priority:** low
- **Status:** open
- **Parent:** #55
- **Labels:** be-plan, intertext, research
- **Created:** 2026-10-02

Cyberpunk, Tetsuo: The Iron Man, Ballard, Aase Berg, Blake Butler, text-virus/codework poetics - expectations for the kind of meaning present. DONE: annotated dossier.

---

### [ ] #59: Seed entity roles from jacket synopsis, verify against text

- **Priority:** medium
- **Status:** open
- **Parent:** #55
- **Labels:** be-plan, intertext, research
- **Created:** 2026-10-02

Use the canonical macro-plot (ADAM/ANTi-ADAM clash, Sato Corp, Cadaver City, Placenta World) to seed entity roles, then verify each against the corpus. DONE: role table with text evidence or refutation.

---

### [ ] #58: Obtain and align translations (interpretive oracle)

- **Priority:** high
- **Status:** open
- **Parent:** #55
- **Labels:** be-plan, intertext, research
- **Created:** 2026-10-02

A Russian translation exists (trans. A. Skobin) plus author interviews glossing the method. Obtain available parallel text, sentence-align, and treat it as an independent disambiguation signal - the closest thing to a Rosetta stone. DONE: alignment artifact or a documented negative result.

---

### [ ] #57: Intertextual n-gram matching vs Siratori corpus

- **Priority:** medium
- **Status:** open
- **Parent:** #55
- **Labels:** be-plan, intertext, research
- **Created:** 2026-10-02

Match phrases against Siratori's other works (Headcode, Human_Worms, Smart-d, GENEDUB, Nonexistence) and cited influences; shared formulas separate idiolect from genre. DONE: overlap report.

---

### [ ] #56: Build reference corpora and keyness harness

- **Priority:** high
- **Status:** open
- **Parent:** #55
- **Labels:** be-plan, intertext, research
- **Created:** 2026-10-02

Assemble general English, cyberpunk (Gibson/Stephenson/Sterling), cut-up/Surrealist (Burroughs/Artaud/Breton), Finnegans Wake, and medical/CS/legal abstracts; wire the keyness harness. DONE: corpora + harness.

---

### [ ] #55: E7 — Cross-corpus, translation and intertext

- **Priority:** low
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, intertext
- **Created:** 2026-10-02

L7. External material is a prior, not the object. Use it to disambiguate, never to ventriloquize. Highest-value item: translations as an interpretive oracle.

---

### [ ] #54: Self-similarity matrix (loop detection)

- **Priority:** medium
- **Status:** open
- **Parent:** #46
- **Labels:** be-plan, research, signal
- **Created:** 2026-10-02

Paragraph x paragraph cosine self-similarity; does the 'loop' show as a diagonal band or repeating blocks? DONE: matrix + structural reading.

---

### [ ] #53: Burstiness and Heaps' law

- **Priority:** low
- **Status:** open
- **Parent:** #46
- **Labels:** be-plan, research, signal
- **Created:** 2026-10-02

Vocabulary growth vs text length; burstiness of entity use. DONE: growth/burstiness report.

---

### [ ] #52: Operator-word mutual information

- **Priority:** medium
- **Status:** open
- **Parent:** #46
- **Labels:** be-plan, research, signal
- **Created:** 2026-10-02

Which operators predict which lexical domains (:: before body terms? = before tech terms?); converts the operator grammar into statistics. DONE: MI matrix + findings.

---

### [ ] #51: Periodicity and spectral analysis

- **Priority:** high
- **Status:** open
- **Parent:** #46
- **Labels:** be-plan, research, signal
- **Created:** 2026-10-02

Map the symbol stream to numeric series (1 for delimiter else 0; inter-delimiter distances); run autocorrelation and FFT for latent periodicity - the measurable form of 'a devastating loop of language'. DONE: spectra + periodicity report.

---

### [ ] #50: Quantified silence-removal experiment

- **Priority:** critical
- **Status:** open
- **Parent:** #46
- **Labels:** be-plan, research, signal
- **Created:** 2026-10-02

For each E1.4 variant measure delta in entropy, compressibility, and LM perplexity. The transform that most lowers perplexity is closest to the underlying 'vowel' system - a direct methodological replication of Begus on text. DONE: results table + verdict.

---

### [ ] #49: LM perplexity per page

- **Priority:** high
- **Status:** open
- **Parent:** #46
- **Labels:** be-plan, research, signal
- **Created:** 2026-10-02

Score the book with GPT-2 / a modern LM; plot perplexity per page. DONE: perplexity curve + baseline comparison.

---

### [ ] #48: Compression / MDL vs English baseline

- **Priority:** high
- **Status:** open
- **Parent:** #46
- **Labels:** be-plan, research, signal
- **Created:** 2026-10-02

gzip/zstd compression ratio vs English prose; low compressibility = high alienness. DONE: compressibility comparison + method note.

---

### [ ] #47: Multi-level entropy analysis

- **Priority:** high
- **Status:** open
- **Parent:** #46
- **Labels:** be-plan, research, signal
- **Created:** 2026-10-02

Shannon entropy at char/token/operator level and conditional entropy H(next|prev) at several orders; compare to English (~1 bit/char, ~9-10 bits/word raw). DONE: entropy report.

---

### [ ] #46: E6 — Information-theoretic signal layer (Begus programme)

- **Priority:** high
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, signal
- **Created:** 2026-10-02

L6. Treat the token/operator stream as an acoustic-like signal, directly transferring the sperm-whale methodology. This is the signature method of the program.

---

### [ ] #45: Analogy and metaphor mining

- **Priority:** low
- **Status:** open
- **Parent:** #38
- **Labels:** be-plan, research, semantics
- **Created:** 2026-10-02

Vector arithmetic on entity terms to surface the book's own equations (gene=TV : digital=vamp :: ? : ?). DONE: candidate equations, ranked and annotated.

---

### [ ] #44: Affect and valence curve

- **Priority:** medium
- **Status:** open
- **Parent:** #38
- **Labels:** be-plan, research, semantics
- **Created:** 2026-10-02

Lexicon + classifier scores for fear/ecstasy/paranoia/disgust; plot over the book (explicit markers like Fear=cell, ecstasy). DONE: affect curve + report.

---

### [ ] #43: Contrastive semantic axes

- **Priority:** medium
- **Status:** open
- **Parent:** #38
- **Labels:** be-plan, research, semantics
- **Created:** 2026-10-02

Find and interpret dimensions separating clusters (organic<->machine, self<->other, body<->information, sacred<->profane); test with antonym seed pairs. DONE: named axes + supporting contrasts.

---

### [ ] #42: UMAP theme map of the book

- **Priority:** medium
- **Status:** open
- **Parent:** #38
- **Labels:** be-plan, research, semantics
- **Created:** 2026-10-02

Project embeddings with UMAP/t-SNE colored by page to reveal narrative drift or loops. DONE: interactive/static map artifact.

---

### [ ] #41: Topic modelling (LDA/NMF/BERTopic)

- **Priority:** high
- **Status:** open
- **Parent:** #38
- **Labels:** be-plan, research, semantics
- **Created:** 2026-10-02

Latent themes with top terms; compare against network communities from E3.6. DONE: topic model + coherence/interpretation report.

---

### [ ] #40: Clustering to motifs and scenes

- **Priority:** high
- **Status:** open
- **Parent:** #38
- **Labels:** be-plan, research, semantics
- **Created:** 2026-10-02

k-means/HDBSCAN over clause embeddings to recover recurring motifs (dog+murder+gene; brain+Load+ecstasy). DONE: cluster catalogue with exemplar clauses.

---

### [ ] #39: Clause/sentence embeddings

- **Priority:** high
- **Status:** open
- **Parent:** #38
- **Labels:** be-plan, research, semantics
- **Created:** 2026-10-02

Embed clauses/sentences with SBERT/E5; persist vectors keyed to page/clause. DONE: embedding artifact + loader.

---

### [ ] #38: E5 — Semantic themes, motifs and affect

- **Priority:** medium
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, semantics
- **Created:** 2026-10-02

L5. Recurring scenes and motifs via embeddings, clustering, topics, axes, and affect.

---

### [ ] #37: POS-tagging the silence-removed residue

- **Priority:** medium
- **Status:** open
- **Parent:** #31
- **Labels:** be-plan, research, syntax
- **Created:** 2026-10-02

Tag the plain stream from E1.4; measure drift in noun/verb balance vs normal English. DONE: POS comparison report.

---

### [ ] #36: Formal grammar induction and complexity

- **Priority:** medium
- **Status:** open
- **Parent:** #31
- **Labels:** be-plan, research, syntax
- **Created:** 2026-10-02

Ask whether the operator/token stream is regular, context-free, or needs context; estimate automaton state count via compression/entropy. DONE: a measurable 'language or noise' verdict.

---

### [ ] #35: n-gram LM and per-token surprisal

- **Priority:** high
- **Status:** open
- **Parent:** #31
- **Labels:** be-plan, research, syntax
- **Created:** 2026-10-02

Train an n-gram or neural LM on the book; compute per-token surprisal; high-surprisal zones are the deranged core. DONE: surprisal curve + hot/cold zones.

---

### [ ] #34: Frame and template mining

- **Priority:** high
- **Status:** open
- **Parent:** #31
- **Labels:** be-plan, research, syntax
- **Created:** 2026-10-02

Extract recurrent schemata ('I <verb> the <NP> of <NP>', 'the <N> that <V>', <<...>>'); their recurrence is the book's own loop grammar. DONE: frame catalogue + counts.

---

### [ ] #33: Clause segmentation via operators

- **Priority:** high
- **Status:** open
- **Parent:** #31
- **Labels:** be-plan, research, syntax
- **Created:** 2026-10-02

Split at ::, //, sentence-final /; treat each operator-delimited span as a clause; measure length/type distributions. DONE: clause inventory + distributions.

---

### [ ] #32: Parse-attempt and derangement curve per page

- **Priority:** high
- **Status:** open
- **Parent:** #31
- **Labels:** be-plan, research, syntax
- **Created:** 2026-10-02

Run spaCy/stanza; record the fraction of sentences that parse and their confidence. Percent-parseable-per-page quantifies the observed drift toward coherence. DONE: derangement curve + report.

---

### [ ] #31: E4 — Syntax and structure without a parser

- **Priority:** medium
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, syntax
- **Created:** 2026-10-02

L4. A constituency/dependency parser will mostly fail; that failure is data. Recover structure from operators, repeated frames, and surprisal.

---

### [ ] #30: Entity and proper-noun induction (dramatis personae)

- **Priority:** high
- **Status:** open
- **Parent:** #23
- **Labels:** be-plan, distribution, research
- **Created:** 2026-10-02

Capitalization + recurrence + collocation to build the personae: ADAM, ANTi-ADAM, Super Cherry 666, Sato Corporation, ToK/ToKAGE, NDRO, OMOTYA, etc. DONE: entity glossary with evidence and roles.

---

### [ ] #29: Co-occurrence network and community detection

- **Priority:** high
- **Status:** open
- **Parent:** #23
- **Labels:** be-plan, distribution, research
- **Created:** 2026-10-02

Nodes = terms, edges = PMI; Louvain community detection should recover the semantic domains (body/machine/violence/sex/genetics/corporate); centrality identifies load-bearing concepts. DONE: graph + community report + centrality.

---

### [ ] #28: Static embeddings and analogies

- **Priority:** high
- **Status:** open
- **Parent:** #23
- **Labels:** be-plan, distribution, research
- **Created:** 2026-10-02

Train word2vec/GloVe/fastText on this text alone and on a merged English corpus; fastText handles OOV coinages via subwords. Produce nearest neighbours and analogies (X:Y :: Z:?). DONE: embedding artifacts + neighbours/analogies.

---

### [ ] #27: Collocation and association (PMI/logDice)

- **Priority:** high
- **Status:** open
- **Parent:** #23
- **Labels:** be-plan, distribution, research
- **Created:** 2026-10-02

Bigram/trigram association, especially operator-word and entity-verb pairs: what does ToKAGE do, what is done to it. DONE: ranked collocation tables.

---

### [ ] #26: KWIC concordancer for entities and terms

- **Priority:** high
- **Status:** open
- **Parent:** #23
- **Labels:** be-plan, distribution, research
- **Created:** 2026-10-02

For every entity and suspicious term print every occurrence in +/-5-token context; this is the primary route to a term's stable sense. DONE: concordance artifact + per-term summaries.

---

### [ ] #25: Keyness against reference corpora

- **Priority:** high
- **Status:** open
- **Parent:** #23
- **Labels:** be-plan, distribution, research
- **Created:** 2026-10-02

Log-likelihood / TF-IDF keyness vs general English and the technical register; identify words diagnostic of this text. DONE: keyness tables.

---

### [ ] #24: Frequency, Zipf and type/token analysis

- **Priority:** high
- **Status:** open
- **Parent:** #23
- **Labels:** be-plan, distribution, research
- **Created:** 2026-10-02

Type/token ratio, hapax ratio, Zipf slope; compare to English prose to distinguish heavy coinage from noise. DONE: statistics report.

---

### [ ] #23: E3 — Distributional semantics & semantic network

- **Priority:** high
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, distribution, epic
- **Created:** 2026-10-02

L3. Meaning from usage, not definition: frequency, keyness, concordance, collocation, embeddings, and a co-occurrence network over the whole book.

---

### [ ] #22: Case and emphasis inventory

- **Priority:** low
- **Status:** open
- **Parent:** #16
- **Labels:** be-plan, lexicon, research
- **Created:** 2026-10-02

Extract ALL-CAPS tokens and decide whether they are headers, screams, or labels; report against section headers. DONE: inventory + classification.

---

### [ ] #21: Numeric and symbol channel catalogue

- **Priority:** medium
- **Status:** open
- **Parent:** #16
- **Labels:** be-plan, lexicon, research
- **Created:** 2026-10-02

Catalogue in-text numbers (666, XX, zero, v1.0) and treat them as a separate channel that may encode counts, versions, or pseudo-code. DONE: numeric/symbol inventory + interpretation hypotheses.

---

### [ ] #20: Validated morphological segmentation

- **Priority:** medium
- **Status:** open
- **Parent:** #16
- **Labels:** be-plan, lexicon, research
- **Created:** 2026-10-02

Apply Morhpessor/UniMorph/ByT5 segmentation but validate against the gazetteer so it does not segment away real coinages. DONE: segmentation with precision spot-check.

---

### [ ] #19: Neologism detection -> neologisms.tsv

- **Priority:** high
- **Status:** open
- **Parent:** #16
- **Labels:** be-plan, lexicon, research
- **Created:** 2026-10-02

OOV against an English wordlist + Zipf; separate true coinages, compounds of English parts, and OCR errors; record first attestation and counts. DONE: neologisms.tsv + method notes.

---

### [ ] #18: Compound/portmanteau decomposition and affix lexicon

- **Priority:** high
- **Status:** open
- **Parent:** #16
- **Labels:** be-plan, lexicon, research
- **Created:** 2026-10-02

Handle hyphen (=), concat (++), identity (=), slash (/). Detect productive classifier/head morphemes: -oid, -machine, -vamp, -gram, -cell, -embryo, -drone, -junkie, -mania. DONE: decomposition rules + productive-affix mini-lexicon.

---

### [ ] #17: Operator-aware tokenizer

- **Priority:** high
- **Status:** open
- **Parent:** #16
- **Labels:** be-plan, lexicon, research
- **Created:** 2026-10-02

Tokenize so gene=TV, soul/gram, suck=blood remain analyzable units plus their parts. Standard tokenizers destroy the meaningful seams. DONE: tokenizer + spec + tests.

---

### [ ] #16: E2 — Lexicon & morphology

- **Priority:** high
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, lexicon
- **Created:** 2026-10-02

L2. Words, compounds, inventions. Because English lexis survives, this level builds the vocabulary the rest of the program depends on.

---

### [ ] #15: Rosetta mapping against real notations

- **Priority:** medium
- **Status:** open
- **Parent:** #9
- **Labels:** be-plan, operator, research
- **Created:** 2026-10-02

Place the operator set beside C/Perl/Lisp/S-expressions, HTML/Markdown, regex, and molecular/genetic notation; score which real syntax the book's code most resembles. DONE: scored comparison = a testable 'where the book is coming from' claim.

---

### [ ] #14: Operator grammar block parser (AST)

- **Priority:** high
- **Status:** open
- **Parent:** #9
- **Labels:** be-plan, operator, research
- **Created:** 2026-10-02

Implement a parser for the operator grammar: << ... >> as quoted/blocked speech, :: fields as records, // clauses as lines; emit an AST/tree over the whole book, not flat text. DONE: parser + AST artifact + spot-check evaluation.

---

### [ ] #13: Silence-removal transform variants

- **Priority:** high
- **Status:** open
- **Parent:** #9
- **Labels:** be-plan, operator, research
- **Created:** 2026-10-02

Begus transform: produce variants that delete all operators, delete only ::,//, map operators to space, and map operators to period. Persist each residue stream. DONE: variant files + rationale; feeds E6.4 and E8.6.

---

### [ ] #12: Operator function classification

- **Priority:** high
- **Status:** open
- **Parent:** #9
- **Labels:** be-plan, operator, research
- **Created:** 2026-10-02

Test the working hypothesis for each operator role: :: separator/field, // utterance/comment, ++ concatenate/increment, = naming/identity (gene=TV), << >> stream/block delimiters, / alternation/possessive, ` escape, - compound linker. Test each against distribution; refine or reject. DONE: classification spec with evidence per operator.

---

### [ ] #11: Operator n-gram transition matrix and Markov model

- **Priority:** high
- **Status:** open
- **Parent:** #9
- **Labels:** be-plan, operator, research
- **Created:** 2026-10-02

Model operator sequences (1st/2nd order); test whether transitions are idiomatic (a real syntax) or random; report perplexity of the operator stream under the model. DONE: transition matrix + randomness verdict.

---

### [ ] #10: Operator inventory & frequency (incl. multi-operator runs)

- **Priority:** high
- **Status:** open
- **Parent:** #9
- **Labels:** be-plan, operator, research
- **Created:** 2026-10-02

Count each operator and multi-operator run (//++, ::<<, >::, ::>>++ all occur); emit operators.tsv with counts and per-page distribution. DONE: operators.tsv.

---

### [ ] #9: E1 — Operator grammar & the symbolic machine

- **Priority:** critical
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, operator
- **Created:** 2026-10-02

L1. The operators are the book's opcodes. If the text is 'a fictional code', its grammar is the highest-leverage symbolic-meaning target. Deliverables: operator grammar spec, block parser, silence-removal variants, Rosetta mapping.

---

### [ ] #8: Build a page/folio map for citation

- **Priority:** high
- **Status:** open
- **Parent:** #2
- **Labels:** be-plan, ocr, research
- **Created:** 2026-10-02

Align markdown spans to PDF page indices and printed folio numbers so every downstream claim is citable to a page. DONE: folio map + coverage validation.

---

### [ ] #7: Recover layout, paragraphs and section headers

- **Priority:** high
- **Status:** open
- **Parent:** #2
- **Labels:** be-plan, ocr, research
- **Created:** 2026-10-02

Use marker block structure to recover paragraph/section breaks; extract ALL section headers (only 'abolition' is known at present). DONE: sections.tsv + layout-corrected markdown.

---

### [ ] #6: Build gazetteer-protected lexical correction

- **Priority:** high
- **Status:** open
- **Parent:** #2
- **Labels:** be-plan, ocr, research
- **Created:** 2026-10-02

Full observed token list; correct only tokens within edit distance 1-2 of a known English word or a confirmed entity (boyroid->boy-roid); never correct true coinages - protect them with an explicit gazetteer. DONE: corrections log + vetted corpus + gazetteer.

---

### [ ] #5: Inventory glyphs and freeze unicode normalization

- **Priority:** high
- **Status:** open
- **Parent:** #2
- **Labels:** be-plan, ocr, research
- **Created:** 2026-10-02

Enumerate every codepoint across the corpus; decide canonical forms; normalize ligatures and quote variants; freeze charset.tsv and a reusable normalizer with a test. DONE: charset.tsv + normalizer + passing test.

---

### [ ] #4: Ensemble OCR (Tesseract/PaddleOCR/docTR) with glyph voting

- **Priority:** medium
- **Status:** open
- **Parent:** #2
- **Labels:** be-plan, ocr, research
- **Created:** 2026-10-02

OCR the same pages with >=2 other engines, align outputs, take a character-level vote, and produce a disagreement atlas localizing ambiguous glyphs (:: vs :, << , backtick vs apostrophe). DONE: consensus transcript + disagreement atlas.

---

### [ ] #3: Map OCR confidence for the marker corpus

- **Priority:** high
- **Status:** open
- **Parent:** #2
- **Labels:** be-plan, ocr, research
- **Created:** 2026-10-02

Re-run marker with --output_format json to obtain per-line/word confidences; extract scores, flag low-confidence tokens, and write analysis/ocr_confidence/. DONE: confidence table + flagged-token list + short report.

---

### [ ] #2: E0 — Corpus ingestion & OCR stabilization

- **Priority:** critical
- **Status:** open
- **Parent:** #1
- **Labels:** be-plan, epic, ocr
- **Created:** 2026-10-02

L0. Stabilize the signal before any analysis. Source is a 225-page scanned PDF; marker OCR already preserves operators but drifts (It->lt, boyroid->boy-roid, spacing around >>). Deliver a vetted, citable token stream.

---

### [ ] #1: Decode Blood Electric — end-to-end reading program

- **Priority:** high
- **Status:** open
- **Labels:** be-plan, epic, program
- **Created:** 2026-10-02

Umbrella epic for turning Kenji Siratori's *Blood Electric* (2002) into an evidence-linked, understandable translation into English. Thesis: the book is English lexis under a symbolic transform (operator punctuation + neologisms); measure the transform, strip it (Begus-style silence removal), re-embed the residue in meaning-space.

Authoritative method doc: docs/meaning-extraction-methods.md (levels L0-L9). Orientation: README.md. Corpus: docs/markdown/blood-electric/blood-electric.md.

DONE = every token/operator accounted for; operator grammar specified and predictive; motif/theme map converges across independent methods; an interlinear edition + glossed termbase + narrative skeleton exist; evaluation criteria (L9) are met and anti-patterns audited.

---

