#!/usr/bin/env python3
"""
Build the Chainlink epic/issue plan for decoding Blood Electric.

Reads nothing; defines the plan inline (mirrors docs/meaning-extraction-methods.md)
and drives the `chainlink` CLI: program epic -> phase epics -> tasks, then blockers.

Idempotency: every issue gets the label `be-plan`; if the program epic already
exists, the script aborts rather than duplicating.
"""
import subprocess, json, sys, os, datetime

WS = "/mnt/sharedOs/blood-electric-workspace"
MARK = "be-plan"

def cl(args, quiet=True):
    base = ["chainlink"] + (["-q"] if quiet else []) + args
    p = subprocess.run(base, cwd=WS, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"chainlink failed: {base}\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()

def create(title, desc, priority, labels):
    args = ["issue", "create", title, "-p", priority, "-d", desc]
    for l in labels: args += ["-l", l]
    return int(cl(args))

def subissue(parent, title, desc, priority, labels):
    args = ["issue", "subissue", str(parent), title, "-p", priority, "-d", desc]
    for l in labels: args += ["-l", l]
    return int(cl(args))

def block(blocked, blocker):
    cl(["issue", "block", str(blocked), str(blocker)])

# ---------------------------------------------------------------------------
# Plan definition
# ---------------------------------------------------------------------------
PROGRAM = dict(
    title="Decode Blood Electric — end-to-end reading program",
    priority="high",
    desc=(
        "Umbrella epic for turning Kenji Siratori's *Blood Electric* (2002) into an "
        "evidence-linked, understandable translation into English. Thesis: the book is "
        "English lexis under a symbolic transform (operator punctuation + neologisms); "
        "measure the transform, strip it (Begus-style silence removal), re-embed the "
        "residue in meaning-space.\n\n"
        "Authoritative method doc: docs/meaning-extraction-methods.md (levels L0-L9). "
        "Orientation: README.md. Corpus: docs/markdown/blood-electric/blood-electric.md.\n\n"
        "DONE = every token/operator accounted for; operator grammar specified and "
        "predictive; motif/theme map converges across independent methods; an interlinear "
        "edition + glossed termbase + narrative skeleton exist; evaluation criteria (L9) "
        "are met and anti-patterns audited."
    ),
)

PHASES = [
 dict(key="E0", title="E0 — Corpus ingestion & OCR stabilization",
   priority="critical", label="ocr",
   desc=("L0. Stabilize the signal before any analysis. Source is a 225-page scanned PDF; "
         "marker OCR already preserves operators but drifts (It->lt, boyroid->boy-roid, "
         "spacing around >>). Deliver a vetted, citable token stream."),
   deps=[],
   tasks=[
    dict(key="E0.1", title="Map OCR confidence for the marker corpus", priority="high",
      desc=("Re-run marker with --output_format json to obtain per-line/word confidences; "
            "extract scores, flag low-confidence tokens, and write analysis/ocr_confidence/. "
            "DONE: confidence table + flagged-token list + short report."),
      deps=[]),
    dict(key="E0.2", title="Ensemble OCR (Tesseract/PaddleOCR/docTR) with glyph voting", priority="medium",
      desc=("OCR the same pages with >=2 other engines, align outputs, take a character-level "
            "vote, and produce a disagreement atlas localizing ambiguous glyphs (:: vs :, "
            "<< , backtick vs apostrophe). DONE: consensus transcript + disagreement atlas."),
      deps=["E0.1"]),
    dict(key="E0.3", title="Inventory glyphs and freeze unicode normalization", priority="high",
      desc=("Enumerate every codepoint across the corpus; decide canonical forms; normalize "
            "ligatures and quote variants; freeze charset.tsv and a reusable normalizer with "
            "a test. DONE: charset.tsv + normalizer + passing test."),
      deps=[]),
    dict(key="E0.4", title="Build gazetteer-protected lexical correction", priority="high",
      desc=("Full observed token list; correct only tokens within edit distance 1-2 of a known "
            "English word or a confirmed entity (boyroid->boy-roid); never correct true "
            "coinages - protect them with an explicit gazetteer. DONE: corrections log + "
            "vetted corpus + gazetteer."),
      deps=["E0.1", "E0.3"]),
    dict(key="E0.5", title="Recover layout, paragraphs and section headers", priority="high",
      desc=("Use marker block structure to recover paragraph/section breaks; extract ALL "
            "section headers (only 'abolition' is known at present). DONE: sections.tsv + "
            "layout-corrected markdown."),
      deps=["E0.3"]),
    dict(key="E0.6", title="Build a page/folio map for citation", priority="high",
      desc=("Align markdown spans to PDF page indices and printed folio numbers so every "
            "downstream claim is citable to a page. DONE: folio map + coverage validation."),
      deps=["E0.5"]),
   ]),

 dict(key="E1", title="E1 — Operator grammar & the symbolic machine",
   priority="critical", label="operator",
   desc=("L1. The operators are the book's opcodes. If the text is 'a fictional code', its "
         "grammar is the highest-leverage symbolic-meaning target. Deliverables: operator "
         "grammar spec, block parser, silence-removal variants, Rosetta mapping."),
   deps=["E0"],
   tasks=[
    dict(key="E1.1", title="Operator inventory & frequency (incl. multi-operator runs)", priority="high",
      desc=("Count each operator and multi-operator run (//++, ::<<, >::, ::>>++ all occur); "
            "emit operators.tsv with counts and per-page distribution. DONE: operators.tsv."),
      deps=[]),
    dict(key="E1.2", title="Operator n-gram transition matrix and Markov model", priority="high",
      desc=("Model operator sequences (1st/2nd order); test whether transitions are idiomatic "
            "(a real syntax) or random; report perplexity of the operator stream under the "
            "model. DONE: transition matrix + randomness verdict."),
      deps=["E1.1"]),
    dict(key="E1.3", title="Operator function classification", priority="high",
      desc=("Test the working hypothesis for each operator role: :: separator/field, // "
            "utterance/comment, ++ concatenate/increment, = naming/identity (gene=TV), "
            "<< >> stream/block delimiters, / alternation/possessive, ` escape, - compound "
            "linker. Test each against distribution; refine or reject. DONE: classification "
            "spec with evidence per operator."),
      deps=["E1.2"]),
    dict(key="E1.4", title="Silence-removal transform variants", priority="high",
      desc=("Begus transform: produce variants that delete all operators, delete only ::,//, "
            "map operators to space, and map operators to period. Persist each residue stream. "
            "DONE: variant files + rationale; feeds E6.4 and E8.6."),
      deps=["E1.1"]),
    dict(key="E1.5", title="Operator grammar block parser (AST)", priority="high",
      desc=("Implement a parser for the operator grammar: << ... >> as quoted/blocked speech, "
            ":: fields as records, // clauses as lines; emit an AST/tree over the whole book, "
            "not flat text. DONE: parser + AST artifact + spot-check evaluation."),
      deps=["E1.3"]),
    dict(key="E1.6", title="Rosetta mapping against real notations", priority="medium",
      desc=("Place the operator set beside C/Perl/Lisp/S-expressions, HTML/Markdown, regex, "
            "and molecular/genetic notation; score which real syntax the book's code most "
            "resembles. DONE: scored comparison = a testable 'where the book is coming from' "
            "claim."),
      deps=["E1.3"]),
   ]),

 dict(key="E2", title="E2 — Lexicon & morphology",
   priority="high", label="lexicon",
   desc=("L2. Words, compounds, inventions. Because English lexis survives, this level builds "
         "the vocabulary the rest of the program depends on."),
   deps=["E1"],
   tasks=[
    dict(key="E2.1", title="Operator-aware tokenizer", priority="high",
      desc=("Tokenize so gene=TV, soul/gram, suck=blood remain analyzable units plus their "
            "parts. Standard tokenizers destroy the meaningful seams. DONE: tokenizer + spec "
            "+ tests."),
      deps=["E1.1"]),
    dict(key="E2.2", title="Compound/portmanteau decomposition and affix lexicon", priority="high",
      desc=("Handle hyphen (=), concat (++), identity (=), slash (/). Detect productive "
            "classifier/head morphemes: -oid, -machine, -vamp, -gram, -cell, -embryo, -drone, "
            "-junkie, -mania. DONE: decomposition rules + productive-affix mini-lexicon."),
      deps=["E2.1"]),
    dict(key="E2.3", title="Neologism detection -> neologisms.tsv", priority="high",
      desc=("OOV against an English wordlist + Zipf; separate true coinages, compounds of "
            "English parts, and OCR errors; record first attestation and counts. DONE: "
            "neologisms.tsv + method notes."),
      deps=["E2.1"]),
    dict(key="E2.4", title="Validated morphological segmentation", priority="medium",
      desc=("Apply Morhpessor/UniMorph/ByT5 segmentation but validate against the gazetteer so "
            "it does not segment away real coinages. DONE: segmentation with precision "
            "spot-check."),
      deps=["E2.2"]),
    dict(key="E2.5", title="Numeric and symbol channel catalogue", priority="medium",
      desc=("Catalogue in-text numbers (666, XX, zero, v1.0) and treat them as a separate "
            "channel that may encode counts, versions, or pseudo-code. DONE: numeric/symbol "
            "inventory + interpretation hypotheses."),
      deps=["E2.1"]),
    dict(key="E2.6", title="Case and emphasis inventory", priority="low",
      desc=("Extract ALL-CAPS tokens and decide whether they are headers, screams, or labels; "
            "report against section headers. DONE: inventory + classification."),
      deps=["E2.1"]),
   ]),

 dict(key="E3", title="E3 — Distributional semantics & semantic network",
   priority="high", label="distribution",
   desc=("L3. Meaning from usage, not definition: frequency, keyness, concordance, "
         "collocation, embeddings, and a co-occurrence network over the whole book."),
   deps=["E2"],
   tasks=[
    dict(key="E3.1", title="Frequency, Zipf and type/token analysis", priority="high",
      desc=("Type/token ratio, hapax ratio, Zipf slope; compare to English prose to distinguish "
            "heavy coinage from noise. DONE: statistics report."),
      deps=["E2.1"]),
    dict(key="E3.2", title="Keyness against reference corpora", priority="high",
      desc=("Log-likelihood / TF-IDF keyness vs general English and the technical register; "
            "identify words diagnostic of this text. DONE: keyness tables."),
      deps=["E2.1", "E7.1"]),
    dict(key="E3.3", title="KWIC concordancer for entities and terms", priority="high",
      desc=("For every entity and suspicious term print every occurrence in +/-5-token "
            "context; this is the primary route to a term's stable sense. DONE: concordance "
            "artifact + per-term summaries."),
      deps=["E2.1"]),
    dict(key="E3.4", title="Collocation and association (PMI/logDice)", priority="high",
      desc=("Bigram/trigram association, especially operator-word and entity-verb pairs: what "
            "does ToKAGE do, what is done to it. DONE: ranked collocation tables."),
      deps=["E2.1"]),
    dict(key="E3.5", title="Static embeddings and analogies", priority="high",
      desc=("Train word2vec/GloVe/fastText on this text alone and on a merged English corpus; "
            "fastText handles OOV coinages via subwords. Produce nearest neighbours and "
            "analogies (X:Y :: Z:?). DONE: embedding artifacts + neighbours/analogies."),
      deps=["E2.3"]),
    dict(key="E3.6", title="Co-occurrence network and community detection", priority="high",
      desc=("Nodes = terms, edges = PMI; Louvain community detection should recover the "
            "semantic domains (body/machine/violence/sex/genetics/corporate); centrality "
            "identifies load-bearing concepts. DONE: graph + community report + centrality."),
      deps=["E3.4"]),
    dict(key="E3.7", title="Entity and proper-noun induction (dramatis personae)", priority="high",
      desc=("Capitalization + recurrence + collocation to build the personae: ADAM, ANTi-ADAM, "
            "Super Cherry 666, Sato Corporation, ToK/ToKAGE, NDRO, OMOTYA, etc. DONE: entity "
            "glossary with evidence and roles."),
      deps=["E2.3", "E3.3"]),
   ]),

 dict(key="E4", title="E4 — Syntax and structure without a parser",
   priority="medium", label="syntax",
   desc=("L4. A constituency/dependency parser will mostly fail; that failure is data. Recover "
         "structure from operators, repeated frames, and surprisal."),
   deps=["E1", "E2"],
   tasks=[
    dict(key="E4.1", title="Parse-attempt and derangement curve per page", priority="high",
      desc=("Run spaCy/stanza; record the fraction of sentences that parse and their confidence. "
            "Percent-parseable-per-page quantifies the observed drift toward coherence. DONE: "
            "derangement curve + report."),
      deps=["E2.1"]),
    dict(key="E4.2", title="Clause segmentation via operators", priority="high",
      desc=("Split at ::, //, sentence-final /; treat each operator-delimited span as a clause; "
            "measure length/type distributions. DONE: clause inventory + distributions."),
      deps=["E1.5"]),
    dict(key="E4.3", title="Frame and template mining", priority="high",
      desc=("Extract recurrent schemata ('I <verb> the <NP> of <NP>', 'the <N> that <V>', "
            "<<...>>'); their recurrence is the book's own loop grammar. DONE: frame catalogue "
            "+ counts."),
      deps=["E4.2"]),
    dict(key="E4.4", title="n-gram LM and per-token surprisal", priority="high",
      desc=("Train an n-gram or neural LM on the book; compute per-token surprisal; high-"
            "surprisal zones are the deranged core. DONE: surprisal curve + hot/cold zones."),
      deps=["E2.1"]),
    dict(key="E4.5", title="Formal grammar induction and complexity", priority="medium",
      desc=("Ask whether the operator/token stream is regular, context-free, or needs context; "
            "estimate automaton state count via compression/entropy. DONE: a measurable "
            "'language or noise' verdict."),
      deps=["E4.2"]),
    dict(key="E4.6", title="POS-tagging the silence-removed residue", priority="medium",
      desc=("Tag the plain stream from E1.4; measure drift in noun/verb balance vs normal "
            "English. DONE: POS comparison report."),
      deps=["E1.4", "E2.1"]),
   ]),

 dict(key="E5", title="E5 — Semantic themes, motifs and affect",
   priority="medium", label="semantics",
   desc=("L5. Recurring scenes and motifs via embeddings, clustering, topics, axes, and affect."),
   deps=["E3", "E4"],
   tasks=[
    dict(key="E5.1", title="Clause/sentence embeddings", priority="high",
      desc=("Embed clauses/sentences with SBERT/E5; persist vectors keyed to page/clause. DONE: "
            "embedding artifact + loader."),
      deps=["E4.2"]),
    dict(key="E5.2", title="Clustering to motifs and scenes", priority="high",
      desc=("k-means/HDBSCAN over clause embeddings to recover recurring motifs (dog+murder+"
            "gene; brain+Load+ecstasy). DONE: cluster catalogue with exemplar clauses."),
      deps=["E5.1"]),
    dict(key="E5.3", title="Topic modelling (LDA/NMF/BERTopic)", priority="high",
      desc=("Latent themes with top terms; compare against network communities from E3.6. DONE: "
            "topic model + coherence/interpretation report."),
      deps=["E5.1"]),
    dict(key="E5.4", title="UMAP theme map of the book", priority="medium",
      desc=("Project embeddings with UMAP/t-SNE colored by page to reveal narrative drift or "
            "loops. DONE: interactive/static map artifact."),
      deps=["E5.1"]),
    dict(key="E5.5", title="Contrastive semantic axes", priority="medium",
      desc=("Find and interpret dimensions separating clusters (organic<->machine, self<->other, "
            "body<->information, sacred<->profane); test with antonym seed pairs. DONE: named "
            "axes + supporting contrasts."),
      deps=["E5.1"]),
    dict(key="E5.6", title="Affect and valence curve", priority="medium",
      desc=("Lexicon + classifier scores for fear/ecstasy/paranoia/disgust; plot over the book "
            "(explicit markers like Fear=cell, ecstasy). DONE: affect curve + report."),
      deps=["E5.1"]),
    dict(key="E5.7", title="Analogy and metaphor mining", priority="low",
      desc=("Vector arithmetic on entity terms to surface the book's own equations (gene=TV : "
            "digital=vamp :: ? : ?). DONE: candidate equations, ranked and annotated."),
      deps=["E3.5"]),
   ]),

 dict(key="E6", title="E6 — Information-theoretic signal layer (Begus programme)",
   priority="high", label="signal",
   desc=("L6. Treat the token/operator stream as an acoustic-like signal, directly transferring "
         "the sperm-whale methodology. This is the signature method of the program."),
   deps=["E0", "E1"],
   tasks=[
    dict(key="E6.1", title="Multi-level entropy analysis", priority="high",
      desc=("Shannon entropy at char/token/operator level and conditional entropy H(next|prev) "
            "at several orders; compare to English (~1 bit/char, ~9-10 bits/word raw). DONE: "
            "entropy report."),
      deps=["E1.1"]),
    dict(key="E6.2", title="Compression / MDL vs English baseline", priority="high",
      desc=("gzip/zstd compression ratio vs English prose; low compressibility = high alienness. "
            "DONE: compressibility comparison + method note."),
      deps=["E0.4"]),
    dict(key="E6.3", title="LM perplexity per page", priority="high",
      desc=("Score the book with GPT-2 / a modern LM; plot perplexity per page. DONE: perplexity "
            "curve + baseline comparison."),
      deps=["E0.4"]),
    dict(key="E6.4", title="Quantified silence-removal experiment", priority="critical",
      desc=("For each E1.4 variant measure delta in entropy, compressibility, and LM perplexity. "
            "The transform that most lowers perplexity is closest to the underlying 'vowel' "
            "system - a direct methodological replication of Begus on text. DONE: results table "
            "+ verdict."),
      deps=["E1.4", "E6.2", "E6.3"]),
    dict(key="E6.5", title="Periodicity and spectral analysis", priority="high",
      desc=("Map the symbol stream to numeric series (1 for delimiter else 0; inter-delimiter "
            "distances); run autocorrelation and FFT for latent periodicity - the measurable "
            "form of 'a devastating loop of language'. DONE: spectra + periodicity report."),
      deps=["E1.4"]),
    dict(key="E6.6", title="Operator-word mutual information", priority="medium",
      desc=("Which operators predict which lexical domains (:: before body terms? = before tech "
            "terms?); converts the operator grammar into statistics. DONE: MI matrix + findings."),
      deps=["E1.3", "E3.4"]),
    dict(key="E6.7", title="Burstiness and Heaps' law", priority="low",
      desc=("Vocabulary growth vs text length; burstiness of entity use. DONE: growth/burstiness "
            "report."),
      deps=["E2.1"]),
    dict(key="E6.8", title="Self-similarity matrix (loop detection)", priority="medium",
      desc=("Paragraph x paragraph cosine self-similarity; does the 'loop' show as a diagonal "
            "band or repeating blocks? DONE: matrix + structural reading."),
      deps=["E5.1"]),
   ]),

 dict(key="E7", title="E7 — Cross-corpus, translation and intertext",
   priority="low", label="intertext",
   desc=("L7. External material is a prior, not the object. Use it to disambiguate, never to "
         "ventriloquize. Highest-value item: translations as an interpretive oracle."),
   deps=["E0"],
   tasks=[
    dict(key="E7.1", title="Build reference corpora and keyness harness", priority="high",
      desc=("Assemble general English, cyberpunk (Gibson/Stephenson/Sterling), cut-up/"
            "Surrealist (Burroughs/Artaud/Breton), Finnegans Wake, and medical/CS/legal "
            "abstracts; wire the keyness harness. DONE: corpora + harness."),
      deps=["E0.4"]),
    dict(key="E7.2", title="Intertextual n-gram matching vs Siratori corpus", priority="medium",
      desc=("Match phrases against Siratori's other works (Headcode, Human_Worms, Smart-d, "
            "GENEDUB, Nonexistence) and cited influences; shared formulas separate idiolect "
            "from genre. DONE: overlap report."),
      deps=["E0.4"]),
    dict(key="E7.3", title="Obtain and align translations (interpretive oracle)", priority="high",
      desc=("A Russian translation exists (trans. A. Skobin) plus author interviews glossing "
            "the method. Obtain available parallel text, sentence-align, and treat it as an "
            "independent disambiguation signal - the closest thing to a Rosetta stone. DONE: "
            "alignment artifact or a documented negative result."),
      deps=["E0.6"]),
    dict(key="E7.4", title="Seed entity roles from jacket synopsis, verify against text", priority="medium",
      desc=("Use the canonical macro-plot (ADAM/ANTi-ADAM clash, Sato Corp, Cadaver City, "
            "Placenta World) to seed entity roles, then verify each against the corpus. DONE: "
            "role table with text evidence or refutation."),
      deps=["E3.7"]),
    dict(key="E7.5", title="Genre/prior dossier", priority="low",
      desc=("Cyberpunk, Tetsuo: The Iron Man, Ballard, Aase Berg, Blake Butler, text-virus/"
            "codework poetics - expectations for the kind of meaning present. DONE: annotated "
            "dossier."),
      deps=[]),
   ]),

 dict(key="E8", title="E8 — Interpretation, gloss and interlinear edition",
   priority="high", label="gloss",
   desc=("L8. Only after the machinery above generate readings - rigorously, with every "
         "interpretive move visible and overridable."),
   deps=["E2", "E3", "E5"],
   tasks=[
    dict(key="E8.1", title="Interlinear edition generator", priority="high",
      desc=("Three-line aligned rendering: original clause | operator-normalized (silence "
            "removed) | plain-English gloss. DONE: generator + first full-book rendering."),
      deps=["E1.5", "E2.1"]),
    dict(key="E8.2", title="LLM glossing pipeline (multi-strategy)", priority="high",
      desc=("Prompt strategies: literal gloss preserving operators; register-normalized "
            "paraphrase; scene summary; plain-English translation; entity-consistent gloss. "
            "Feed the glossary and operator grammar as constraints. DONE: pipeline + sampled "
            "glosses."),
      deps=["E8.1", "E3.7"]),
    dict(key="E8.3", title="Multi-agent translation committee with disagreement log", priority="medium",
      desc=("Generator/critic/comparer loop; disagreement localizes genuinely ambiguous "
            "passages. DONE: committee harness + disagreement map."),
      deps=["E8.2"]),
    dict(key="E8.4", title="Termbase and consistency enforcement", priority="high",
      desc=("One term = one gloss (except where discrepancy is the point); a termbase prevents "
            "drift across 224 pages. DONE: termbase + consistency checker input."),
      deps=["E3.7", "E2.3"]),
    dict(key="E8.5", title="Narrative reconstruction (entity/event tracking)", priority="high",
      desc=("Track entities/events over the AST to recover who does what to whom (ADAM, "
            "ANTi-ADAM, Super Cherry 666, Sato Corp, the dog, 'self'); jacket synopsis is a "
            "sanity check, not a script. DONE: narrative skeleton + evidence links."),
      deps=["E4.3", "E3.7"]),
    dict(key="E8.6", title="Spectral reading artifact", priority="medium",
      desc=("Present the silence-removed stream as a distinct under-text reading, explicitly "
            "not 'the real text'. DONE: publishable spectral reading + framing note."),
      deps=["E1.4", "E6.4"]),
    dict(key="E8.7", title="Compression test for candidate readings", priority="medium",
      desc=("A good gloss shortens the description of the book; score candidate readings by "
            "whether they make the corpus more predictable and compressible. DONE: scoring "
            "harness + ranked readings."),
      deps=["E8.2", "E6.2"]),
   ]),

 dict(key="E9", title="E9 — Evaluation and epistemology",
   priority="high", label="eval",
   desc=("L9. No ground-truth translation exists; state what would count as evidence and audit "
         "the anti-patterns that would fool us."),
   deps=["E8"],
   tasks=[
    dict(key="E9.1", title="Coverage accounting / residue ledger", priority="high",
      desc=("Does the analysis account for every token/operator, or ignore the awkward residue? "
            "Unexplained residue is the test's teeth. DONE: residue ledger + coverage metric."),
      deps=["E2.1"]),
    dict(key="E9.2", title="Consistency checker", priority="high",
      desc=("Verify the same symbol/term gets the same reading throughout; flag violations. "
            "DONE: checker + report over the termbase."),
      deps=["E8.4"]),
    dict(key="E9.3", title="Predictive validity on held-out text", priority="high",
      desc=("Does the derived grammar predict held-out operator sequences and collocations? "
            "Train 80% / test 20%. DONE: held-out scores."),
      deps=["E4.4", "E4.5"]),
    dict(key="E9.4", title="Triangulation / convergence report", priority="critical",
      desc=("Do independent methods (network community, topic model, embedding cluster, "
            "periodicity) converge on the same motifs/domains? Convergence is evidence; "
            "divergence is a lead. DONE: convergence matrix + report."),
      deps=["E5.2", "E5.3", "E3.6", "E6.5"]),
    dict(key="E9.5", title="Back-translation harness", priority="medium",
      desc=("Gloss -> regenerate a Siratori-style clause -> compare to original. DONE: harness "
            "+ fidelity scores."),
      deps=["E8.2"]),
    dict(key="E9.6", title="Human jury / reader study design", priority="medium",
      desc=("Have readers rate glosses for fidelity to the felt text; design the instrument and "
            "sampling. DONE: study protocol + pilot results if available."),
      deps=["E8.1"]),
    dict(key="E9.7", title="Anti-pattern guardrails audit", priority="high",
      desc=("Audit for hallucinated coherence, LLM circularity (same model translating and "
            "judging), over-correction, single-method certainty, and residue-dropping. DONE: "
            "guardrail checklist + audit results, wired into CI-style checks."),
      deps=["E8.2"]),
   ]),
]

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main():
    existing = subprocess.run(["chainlink","issue","list","--json"], cwd=WS,
                              capture_output=True, text=True).stdout
    try:
        arr = json.loads(existing) if existing.strip() else []
    except Exception:
        arr = []
    if any(MARK in (i.get("labels") or []) for i in arr):
        print("be-plan issues already exist; aborting to avoid duplicates.")
        return

    ids = {}
    program = create(PROGRAM["title"], PROGRAM["desc"], PROGRAM["priority"],
                     ["epic", "program", MARK])
    ids["PROGRAM"] = program
    print(f"program epic #{program}")

    phase_ids = {}
    for ph in PHASES:
        pid = subissue(program, ph["title"], ph["desc"], ph["priority"],
                       ["epic", ph["label"], MARK])
        ids[ph["key"]] = pid
        phase_ids[ph["key"]] = pid
        print(f"  {ph['key']} epic #{pid}: {ph['title']}")
        for t in ph["tasks"]:
            tid = subissue(pid, t["title"], t["desc"], t["priority"],
                           [ph["label"], "research", MARK])
            ids[t["key"]] = tid
            print(f"    {t['key']} #{tid}: {t['title']}")

    # blockers: task-level deps
    for ph in PHASES:
        for t in ph["tasks"]:
            for dep in t.get("deps", []):
                block(ids[t["key"]], ids[dep])
    # blockers: phase-level deps
    for ph in PHASES:
        for dep in ph.get("deps", []):
            block(phase_ids[ph["key"]], ids[dep])

    # persist the mapping
    with open(os.path.join(WS, "docs", "chainlink-plan.json"), "w") as f:
        json.dump({"built_at": datetime.datetime.now().isoformat(),
                   "program": program, "ids": ids}, f, indent=2)
    print(f"\ncreated {len(ids)} issues (1 program + {len(PHASES)} epics + "
          f"{sum(len(p['tasks']) for p in PHASES)} tasks)")
    print("mapping -> docs/chainlink-plan.json")

if __name__ == "__main__":
    main()
