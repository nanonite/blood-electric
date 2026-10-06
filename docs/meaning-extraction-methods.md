# Extracting Lexical & Symbolic Meaning from *Blood Electric*

*Kenji Siratori, Blood Electric (Creation Books, 2002) — 224 pp.*
*Workspace: `blood-electric-workspace`*

---

## 0. The problem, stated precisely

*Blood Electric* is not written in a foreign language and it is not, strictly, nonsense. It is
**English lexis running through a non-English symbolic machine**. A normal reader fails because
the surface keeps almost-working while never resolving into conventional syntax or reference.
A reviewer for the original release put it exactly right:

> "This is not so much a book written in English but a book written in a new language, a
> Meta-English where words no longer carry meaning but become almost symbolic in use."

Scholarship has named the mode **asemic cyberpunk** (Wenaus, *The Twilight of Information
Illiteracy*): English words interleaved with "non-grammatical punctuation symbols, typographical
eccentricities, and numbers," producing "something more akin to a fictional code than a
conventional language."

The companion document in this repo, Gašper Beguš's *The Next Theory of Evolution* (eScholarship,
2026), supplies the methodological key. Beguš decoded sperm-whale codas — an "alien," supposedly
Morse-like signal — not by finding a translation key but by **re-framing the signal**: treating it
as language, removing the silences, and discovering latent spectral structure hidden in plain
sight. The proposal here is to do the same to Siratori.

**Premise:** the text is the only ground truth. Every extraction method below must justify itself
by what it pulls out of the token stream itself, not by what we wish the book said.

---

## 1. What we actually have (corpus characterization)

Before any interpretation, fix the material:

| Layer | Content |
|---|---|
| Source | `docs/pdfs/blood-electric.pdf` — 225 pages, scanned image pages (img2pdf), 99 MB |
| OCR | `docs/markdown/blood-electric/blood-electric.md` — produced by **marker** (surya models), symbolic operators preserved |
| Companion | `docs/pdfs/eScholarship UC item 37t8k62b.pdf` — Beguš, *The Next Theory of Evolution* |

**Observed orthographic inventory** (from pilot pages 4–6):
`::` `//` `++` `=` `/` `<<` `>>` `` ` `` `-` `+` `.` `,` `<` `>` and ALL-CAPS emphasis.

**Observed operator frequencies** (pilot, 976 words): `::` 64, `//` 53, `=` 26, `/` 25,
`++` 15, `<<` 10, `>>` 8.

**Observed lexical skeleton:** ordinary English function words survive (`the` 156, `of` 118,
`that` 40, `to` 38, `is` 21), while content is dominated by recurrent quasi-technical
neologisms and named entities:

- **Entities:** `ToK`, `ToKAGE`, `NDRO`, `OMOTYA`, `Sato Corporation`, `Cadaver City`,
  `Placenta World`, `gene=TV`, `digital=vamp`, `ADAM` / `ANTi-ADAM`, `EVOL`, `Super Cherry 666`,
  `drug-embryo`, `boy-roid`, `soul/gram`, `fear=cell`, `suck=blood`, `hydro-machine`, `exoskeleta`.
- **Semantic domains:** body/anatomy · machine/computation · murder/violence · sex/pornography ·
  genetics/virology · corporate/military · quantum/physics · affect (fear, ecstasy, paranoia).

**Section headers** exist as isolated bold words (`abolition` on p. 4). Locating all of them is
itself a task (see §2.1). Numbers recur (pagination plus in-text forms like `666`, `XX`, `v1.0`).

**Important consequence:** because the English lexis survives, the problem is *not* decipherment
from zero. It is **de-noising + structure induction + glossing** over a partly-legible signal.
That is a far more tractable and more honest framing.

---

## 2. The extraction ladder

Meaning can be pulled from the text at concentric levels. Each level has a signal unit, methods,
tools, and a failure mode. Work outward; later levels depend on earlier ones.

```
L0  Bytes / glyphs          →  what did OCR actually capture?
L1  Typography / operators  →  the punctuation-as-code
L2  Tokens / morphology     →  words, compounds, neologisms
L3  Lexis / distribution    →  frequency, collocation, vectors
L4  Syntax / operator grammar →  clause structure without a parser
L5  Semantics / topics      →  recurring scenes, motifs, domains
L6  Information / signal    →  entropy, surprisal, periodicity (the Beguš layer)
L7  Cross-corpus            →  comparanda, translations, intertext
L8  Interpretation / gloss  →  interlinear translation, narrative reconstruction
L9  Evaluation              →  how would we know a reading is right?
```

---

### L0 — Glyph & OCR layer (stabilize the signal before analyzing it)

The pilot already shows OCR drift: `It` → `lt`, `boy-roid` → `boyroid`, spacing around `>>`.
Uncorrected noise will poison every downstream count.

1. **OCR confidence mapping** — marker/surya emits per-line/per-word confidence in its JSON
   output. Re-run with `--output_format json` to get scores; flag low-confidence tokens.
2. **Multi-engine ensemble OCR** — re-OCR the same pages with Tesseract, PaddleOCR, and
   docTR; take a character-level vote. Disagreements localize genuinely ambiguous glyphs
   (the `::` vs `:`, `<<` vs `<<`, backtick vs apostrophe collisions).
3. **Glyph inventory / unicode normalization** — enumerate every codepoint, decide canonical
   forms, normalize ligatures and quote variants. Freeze a `charset.tsv`.
4. **Lexicon-based correction** — build the full observed token list, then correct only tokens
   that are within edit distance 1–2 of a known English word or a confirmed entity
   (`boyroid→boy-roid`). Never "correct" true neologisms — protect them with a gazetteer.
5. **Layout recovery** — recover paragraph and section breaks from marker's block structure,
   since line/paragraph boundaries may carry meaning (the "loop" uses them).
6. **Page/folio map** — align markdown to PDF page and printed folio, so every claim is citable.

> Output: a vetted, stable token stream + a corrections log. This is the foundation stone.

---

### L1 — Typographic / operator layer (the symbolic machine)

If the text is "a fictional code," the operators are its opcodes. Extracting their grammar is
the single most important symbolic-meaning task.

1. **Operator inventory & frequency** — count each operator and multi-operator run
   (`//++`, `::<<`, `>::`, `::>>++` all occur). Build an `operators.tsv`.
2. **Operator n-grams & transition matrix** — what follows what? A first-order Markov model of
   operators reveals whether sequences are idiomatic (a real syntax) or random.
3. **Function classification** — hypothesis: each operator has a systematic role, e.g.
   - `::` → clause/field separator (like `:` or a struct field access)
   - `//` → utterance/line break, or "comment" boundary (C/JS comment)
   - `++` → concatenation / increment / coupling (C increment, string concat)
   - `=` → identity, assignment, *naming* (`gene=TV` = "gene called TV"; `suck=blood`)
   - `<<` `>>` → stream/block delimiters (C++ I/O, heredoc, quote)
   - `/` → alternation ("or"), possessive, or comment-slash (`soul/gram`)
   - `` ` `` → escape / literalization marker
   - `-` → compound linker (`drug-embryo`, `boy-roid`)
   Test each against the distribution; refine.
   *("brute force" cost only if we assert human limits in what is fundamentally an agentic-engineering task.)*
4. **The "silence removal" transform (Beguš)** — systematically **delete** the operators and
   re-join the residue, then read the resulting plain-English stream. Beguš removed inter-click
   silences in whale codas and vowels appeared. Here, removing operator "silence" may expose an
   underlying narrative/argument. Produce multiple variants (delete all; delete only `:://`;
   replace each operator with a whitespace/period/comma) and compare.
5. **Block parser** — implement a small parser for the operator grammar:
   `<< ... >>` as quoted/blocked speech, `::`-separated fields as records, `//`-separated
   clauses as lines. Emit an AST/tree, not just flat text.
6. **Rosetta mapping** — place the operator set beside real notations (C/Perl/Lisp/S-expressions,
   HTML/Markdown, regex, molecular/genetic notation) and score which real syntax the book's code
   most resembles. This is a concrete, testable "where the book is coming from" claim.

> Output: an operator grammar spec + a parse tree for the whole book.

---

### L2 — Token & morphological layer (words, compounds, inventions)

1. **Operator-aware tokenization** — tokenize so `gene=TV`, `soul/gram`, `suck=blood` remain
   analyzable units plus their parts. Standard tokenizers destroy exactly the meaningful seams.
2. **Compound/portmanteau decomposition** — handle hyphen (`drug-embryo`), concat (`++`),
   identity (`=`), slash (`/`). Detect productive **classifier/head morphemes**:
   `-oid`, `-machine`, `-vamp`, `-gram`, `-cell`, `-embryo`, `-drone`, `-junkie`, `-mania`.
   If these recur as heads, they form a productive derivational system worth a mini-lexicon.
3. **Neologism detection** — OOV against an English wordlist (`wordfreq`, `wordninja`,
   WordNet) + Zipf frequency. Separate: (a) true coinages, (b) compounds of English parts,
   (c) OCR errors. Build `neologisms.tsv` with first attestation and counts.
4. **Morphological segmentation** — apply Morhpessor / UniMorph / a ByT5 segmenter, but
   validate against the gazetteer so it does not "segment away" real coinages.
5. **Numeric & symbol tokens** — catalogue in-text numbers (`666`, `XX`, `zero`, `v1.0`) and
   treat them as a separate channel; they may encode counts, versions, or pseudo-code.
6. **Case & emphasis** — extract ALL-CAPS tokens; are they headers, screams, or labels?

> Output: a lexicon (English + coinages + entities) with morphological structure.

---

### L3 — Lexical & distributional layer (meaning from usage, not definition)

1. **Frequency & Zipf analysis** — type/token ratio, hapax ratio, Zipf slope; compare to
   English prose. A shallow Zipf tail signals either heavy coinage or noise.
2. **Keyness (log-likelihood / TF-IDF)** against reference corpora (§L7) —
   which words are *diagnostic* of this text rather than general English.
3. **KWIC concordances** — for every entity and suspicious term, print every occurrence in
   ±5-token context. This is the philologist's core tool and the most direct route to a
   term's stable sense.
4. **Collocation & association** — PPMI / logDice for bigrams/trigrams, especially
   operator–word and entity–verb pairs. What does `ToKAGE` *do*? What is done *to* it?
5. **Static embeddings** — train word2vec / GloVe / fastText **on this text alone** (and a
   merged corpus with English). fastText handles OOV coinages via subwords. Produce nearest
   neighbours and analogies (`X : Y :: Z : ?`).
6. **Co-occurrence network** — nodes = terms, edges = PMI. Community detection (Louvain)
   should recover the semantic domains (body / machine / violence / sex / genetics / corporate).
   Centrality identifies the load-bearing concepts.
7. **Entity & proper-noun induction** — capitalize pattern + recurrence + collocation to build
   the dramatis personae: `ADAM`, `ANTi-ADAM`, `Super Cherry 666`, `Sato Corporation`, etc.

> Output: a weighted semantic network + entity glossary with evidence.

---

### L4 — Syntactic / structural layer (structure without a working parser)

A constituency/dependency parser will mostly fail on this text. That failure is **data**.

1. **Parse-attempt & derangement metric** — run spaCy/stanza; record the fraction of sentences
   that parse, and the confidence. "Percent parseable per page" is a quantitative trace of the
   reviewer's observation that the prose "becomes more coherent as the story unfolds."
2. **Clause segmentation via operators** — split at `::`, `//`, sentence-final `/`; treat each
   operator-delimited span as a clause/utterance. Measure clause length distributions.
3. **Frame / template mining** — many clauses share frames: `I <verb> the <NP> of <NP>`,
   `the <N> that <V>`, `<<...>>`. Extract and count these schemata. The recurrence is a
   grammar of its own — the "loop."
4. **n-gram language modelling & surprisal** — train an n-gram or neural LM on the book; compute
   per-token surprisal. High-surprisal zones are the deranged core; low-surprisal zones are
   where conventional English shows through.
5. **Formal grammar induction** — can the operator/token stream be captured by a regular
   grammar, a CFG, or does it need context? Estimate the automaton's state count via
   compression/entropy. This answers "is this a language or noise?" with a measurable claim.
6. **POS-tagging the residue** — after "silence removal," tag the plain stream; how does the
   noun/verb balance drift from normal English?

> Output: structural statistics + clause inventory + coherence curve over the book.

---

### L5 — Semantic / topical layer (recurring scenes and motifs)

1. **Sentence/clause embeddings** — SBERT / E5 / other sentence transformers.
2. **Clustering** — k-means / HDBSCAN over clause embeddings → recurring **motifs/scenes**
   (e.g. "dog + murder + gene", "brain + Load + ecstasy"). Cluster counts are testable.
3. **Topic modelling** — LDA / NMF / BERTopic / Top2Vec → latent themes with top terms.
4. **Dimensional reduction & map** — UMAP/t-SNE projection → a navigable "map of the book,"
   coloured by page, revealing narrative drift or loops.
5. **Contrastive semantic axes** — find dimensions that separate clusters and interpret them,
   e.g. organic↔machine, self↔other, body↔information, sacred↔profane, creation↔destruction.
   Test with cosine contrasts on antonym seed pairs.
6. **Affect / valence** — lexicon + classifier scores for fear, ecstasy, paranoia, disgust;
   plot over the book. Siratori's affect words are explicit (`Fear=cell`, `ecstasy`).
7. **Analogy & metaphor mining** — vector arithmetic on entity terms (e.g.
   `gene=TV : digital=vamp :: ? : ?`) to surface the book's own equations.

> Output: motif catalogue, topic list, thematic map, affect curve.

---

### L6 — Information-theoretic & signal layer (the Beguš programme, made literal)

This is where the whale-coda method transfers most directly. Treat the token/operator stream as
an acoustic-like signal.

1. **Entropy** — Shannon entropy at character, token, and operator level; conditional entropy
   H(next | prev) at several orders. Compare to English (≈ 1 bit/char, 9–10 bits/word raw).
2. **Compression / MDL** — gzip / zstd compression ratio vs English prose. "How much less
   predictable is this text?" Low compressibility = high alienness.
3. **Perplexity under an external LM** — score the book with GPT-2 / a modern LLM. Plot
   perplexity per page. Beguš's "remove silences" intuition predicts that stripping operators
   will *lower* perplexity, i.e. reveal human-like structure underneath.
4. **"Silence removal" experiment, quantified** — for each operator-removal variant, measure
   delta in entropy, compressibility, and LM perplexity. Whichever transform most lowers
   perplexity is the one closest to the underlying "vowel" system. This is a *direct
   methodological replication* of Beguš's coda work on a text corpus.
5. **Periodicity / spectral analysis** — map the symbol stream to a numeric series (e.g.
   1 for a delimiter, 0 otherwise; or inter-delimiter distances). Run autocorrelation and FFT
   to find latent periodicity. The jacket copy literally calls the novel "a devastating loop
   of language" — periodicity is the measurable form of that claim.
6. **Operator–word mutual information** — which operators predict which lexical domains?
   (`::` before body terms? `=` before tech terms?) This converts the operator grammar into
   statistics.
7. **Burstiness & Heaps' law** — vocabulary growth vs text length; burstiness of entity use.
8. **Recurrence/skip-gram self-similarity** — a self-similarity matrix of the book
   (paragraph × paragraph cosine); does the "loop" structure show as a diagonal band?

> Output: alienness profile, periodicity report, and a quantified silence-removal result.

---

### L7 — Cross-corpus & intertextual layer (triangulating from outside, carefully)

External material is a *prior*, not the object of study. Use it to disambiguate, never to
ventriloquize.

1. **Reference corpora & keyness** — compare against (a) general English, (b) cyberpunk
   fiction (Gibson, Stephenson, Sterling), (c) cut-up/Surrealist prose (Burroughs, Artaud,
   Breton), (d) *Finnegans Wake*, (e) medical/CS/legal abstracts for the technical register.
2. **Intertextual n-gram matching** — match phrases against Siratori's other works
   (*Headcode*, *Human_Worms*, *Smart-d*, *GENEDUB*, *Nonexistence*…) and the cited influences.
   Shared formulas reveal his idiolect vs. the genre's.
3. **Translations as an interpretive oracle** — a Russian translation of *Blood Electric*
   exists (trans. A. Skobin), and the author gives interviews glossing his method
   (3AM Magazine "The Nude Brain"; nettime; Rhizome "Loop Writing"). If a translated
   parallel text can be obtained, **sent-alignment against it gives an independent
   disambiguation signal** — the closest thing to a Rosetta Stone this project can get.
4. **Author's own commented description** — the jacket synopsis (§0) supplies the canonical
   macro-plot (ADAM / ANTi-ADAM clash, Sato Corp, Cadaver City, Placenta World). Use it to
   seed entity roles, then verify against text.
5. **Genre/priors** — cyberpunk, *Tetsuo: The Iron Man*, Ballard, Aase Berg, Blake Butler,
   "text-virus" / codework poetics. These set expectations for the *kind* of meaning present.

> Output: a keyness report, an intertext map, and (if obtainable) a parallel-text alignment.

---

### L8 — Interpretation & gloss layer (turning structure into English)

Only after the machinery above should we generate readings. Then, do it rigorously.

1. **Interlinear edition** — produce a three-line aligned rendering:
   `original clause | operator-normalized (silence removed) | plain-English gloss`.
   This makes every interpretive move visible and overridable.
2. **LLM glossing pipeline** — multiple prompt strategies:
   (a) literal gloss preserving operators; (b) register-normalized paraphrase;
   (c) scene-level summary; (d) "translate into plain English"; (e) entity-consistent gloss
   using the extracted glossary. Feed the glossary + operator grammar as constraints.
3. **Multi-agent translation committee** — generator / critic / comparer loop; keep a
   disagreement log. Disagreement localizes genuinely ambiguous passages.
4. **Consistency enforcement** — one term = one gloss (except where discrepancy is the point).
   A termbase prevents drift across 224 pages.
5. **Narrative reconstruction** — entity/event tracking over the parsed AST to recover
   *who does what to whom* (ADAM, ANTi-ADAM, Super Cherry 666, Sato Corp, the dog, "self").
   The jacket synopsis is a sanity check on the skeleton, not a script to copy.
6. **The "spectral reading"** — the silence-removed stream is a distinct reading of the book,
   arguably the under-text. Present it as such, not as the "real" text.
7. **Compression test for a reading** — a good gloss should *shorten* the description of the
   book (explain more with fewer symbols). Score candidate readings by whether they make the
   corpus more predictable (lower perplexity) and more compressible.

> Output: an annotated interlinear edition + a narrative skeleton + glossed termbase.

---

### L9 — Evaluation & epistemology (the part that keeps us honest)

There is no ground-truth translation. So we must state *what would count as evidence*:

1. **Coverage** — does the analysis account for every token/operator, or does it ignore the
   awkward residue? Unexplained residue is the test's teeth.
2. **Consistency** — does the same symbol/term get the same reading throughout?
3. **Predictive validity** — does the derived grammar predict held-out operator sequences and
   collocations? (Train on 80%, test on 20%.)
4. **Compression / information gain** — does the interpretation reduce description length?
5. **Triangulation** — do independent methods (network community, topic model, embedding
   cluster, periodicity) converge on the same motifs/domains? Convergence is evidence;
   divergence is a lead.
6. **Back-translation** — gloss → regenerate a Siratori-style clause → compare to original.
7. **Human jury / reader study** — have readers rate glosses for fidelity to the felt text.
8. **Explicit anti-patterns to avoid:**
   - **Hallucinated coherence** — LLMs will happily invent a smooth plot the text does not support.
   - **Circularity** — using an LLM as both translator and judge.
   - **Over-correction** — "fixing" deliberate derangement into normal English and losing the point.
   - **Single-method certainty** — any one method's output is a hypothesis, not a finding.
   - **Ignoring the residue** — the unparseable parts are the phenomenon, not noise to discard.

---

## 3. Tooling (what to actually run)

| Purpose | Candidate tools |
|---|---|
| OCR | `marker` (surya) — already set up in conda env `marker`; Tesseract, PaddleOCR, docTR for ensembles |
| Normalization | Python + `regex`, `unicodedata`, `wordfreq`, `wordninja` |
| NLP | spaCy, stanza, NLTK, textacy |
| Embeddings | Gensim (word2vec/GloVe), fastText, sentence-transformers |
| Topics | scikit-learn (NMF/LDA), BERTopic, Top2Vec |
| Networks | networkx, igraph |
| Info-theory | `gzip`/zstd, `zlib`, HuggingFace `transformers` perplexity, numpy/scipy (FFT) |
| Viz | matplotlib, seaborn, plotly, UMAP |
| Glossing | local/API LLMs, constrained prompting, retrieval over glossary |
| Environment | conda env `marker` (exists); add an analysis env as needed |

---

## 4. Proposed roadmap

- **P0 — Ingest.** marker OCR (running) → OCR confidence → normalize/correct → stable corpus with page map. *(L0)*
- **P1 — Code layer.** Operator inventory, grammar spec, block parser, silence-removal variants. *(L1)*
- **P2 — Lexicon.** Tokenizer, compounds, neologism detection, practiced gazetteer of entities. *(L2)*
- **P3 — Statistics.** Frequencies, keyness, collocations, co-occurrence network, embeddings. *(L3)*
- **P4 — Structure.** Clause inventory, frame mining, derangement curve. *(L4)*
- **P5 — Themes.** Embedding clusters, topics, motif map, affect curve. *(L5)*
- **P6 — Signal.** Entropy, perplexity, silence-removal deltas, periodicity. *(L6)*
- **P7 — Triangulation.** Comparanda, translations, intertext. *(L7)*
- **P8 — Reading.** Interlinear edition, glossed termbase, narrative skeleton. *(L8)*
- **P9 — Verification.** Coverage/consistency/prediction tests, convergence report. *(L9)*

---

## 5. The one-sentence thesis of this program

> *Blood Electric* is legible as **English under a symbolic transform**; by measuring the
> transform (operator grammar), stripping it (silence removal), and re-embedding the residue
> in meaning-space, we can produce a transparent, evidence-linked translation without pretending
> the book is ordinary prose — and, following Beguš, we may find that its alienness was
> "hidden in plain sight."
