# The legal abstract reference corpus — evaluation and curation (#79)

Optional domain-specific comparison corpus for the #56 keyness harness,
evaluated and curated 2026-10-16. This note records the decision, what each
legal register is, the provenance of the new one and its rights/access terms,
so the corpus's position is auditable rather than remembered.

## 1. Decision: keep the historical register, label it, and add the modern one

The question was whether *legal abstracts* — modern legal writing — add value
for the intended analyses. They do, and the defect they fix was already
published. The plan's own words for the technical-register slot are
"medical/CS/legal **abstracts**" (`docs/meaning-extraction-methods.md` L7.1).
What #56 built into that slot was five public-domain *full texts* written
between 1215 and 1850, and the keyness report then described the register as
"modern medical, CS or legal writing" — a claim that exceeded the reference
set, because the legal half of it is two centuries old.

The effect of that gap is measurable, so it was measured before deciding.
Scoring a 1,000-record modern legal-abstract sample (DOAJ, selection rule in
§3) against the book on the same harness reproduces the published core
unchanged and moves the residue:

| quantity | with the historical `legal` register only | with the modern `legal_abstracts` register added |
|---|---:|---:|
| reference-independent core (top-50 of every inferential corpus) | 32 | **32** (no word lost) |
| reference-dependent residue (key vs prose, key against no technical register) | 13 | **8** |
| words the modern legal register decides | — | `discharge`, `exceeded`, `receives`, `reverse`, `rotation` |

The five decided words are ordinary modern legal-register words — discharge of
a contract, appellate reversal, a model code, exceeding an authority, a court
receiving a filing — that 18th-19th-century legal/political prose simply does
not use. While the legal register in the technical union was historical, those
five words were reported as residue of "how old the general-English reference
is" when the deciding fact was the *legal* reference's age. The modern
register is also the more discriminating technical pole: its top-50 key list
overlaps general English at Jaccard 0.695 against the historical register's
0.786.

So the corpus stays and the register is added. Both are kept, because they
measure different things: `legal` is the book against the prose of past
legal/political argument, `legal_abstracts` the book against modern legal
scholarship. **Neither stands in for the other, and no claim may cite one as
the other** — the keyness report now carries a register note saying exactly
this, and the test suite pins the labels.

What changes for the published numbers: the residue verdict 13 → 8, decided
by three *modern* technical registers (`medical`, `cs`, `legal_abstracts`)
rather than two modern ones plus a historical text. The 32-word core is
unchanged, which is the robustness check on the addition.

## 2. What the historical `legal` register is (and is not)

Five Project Gutenberg full texts, all public domain, already in the tree:

| file | text | written |
|---|---|---|
| `legal/federalist_papers.txt` | The Federalist Papers (Hamilton, Madison, Jay) | 1787-88 |
| `legal/declaration_of_independence.txt` | The Declaration of Independence (Jefferson) | 1776 |
| `legal/magna_carta.txt` | Magna Carta (Anonymous) | 1215 |
| `legal/the_law.txt` | Bastiat, *The Law* | 1850 |
| `legal/the_social_contract.txt` | Rousseau, *The Social Contract* | 1762 |

These are **historical legal/political full texts**: treatises, essays and
charters. They are **not legal abstracts**, and they are **not modern legal
writing**. The claims they support are the ones a 350k-token historical
legal/political register supports — a keyness contrast against the prose of
past legal argument (its top-50 overlap with general English is the highest of
the technical registers, 0.786, which is what "close to prose" looks like in
that table). The claims they do not support include any statement about
modern legal diction; for that, use `legal_abstracts`. The register label is
a module constant (`LEGAL_REGISTER_LABEL` in `work/reference_corpora.py`),
pinned by tests, and the manifest's licence cell for the five files stays the
plain truth: `public domain`.

## 3. The modern `legal_abstracts` register: source, selection, provenance

**Source.** The Directory of Open Access Journals (DOAJ) public API,
`search/articles/{query}`:

```
bibjson.journal.title:(law) AND bibjson.journal.language:EN
```

1,000 records, recency-sorted (`sort=created_date:desc`, 10 pages of 100).
The selection rule mirrors medical's `disease[Title]`: journals whose *title*
contains "law", publishing in English. It is arbitrary but explicit, and what
makes it usable is that the **sample is pinned**: the corpus bytes live in
`analysis/reference_corpora/legal_abstracts/legal_abstracts.txt` and their
SHA-256 is recorded in `manifest.tsv` as an `observed_checksum`, which #80's
drift mechanism treats as the file's expected pin.

**Snapshot stability — read this before re-running the query.** A
recency-sorted live query returns a *different* sample each time, so the
stable unit is the pinned snapshot, not the query (exactly the medical
corpus's position). A fresh fetch that produces different bytes is reported
as `manifest_drift.tsv`, and the new snapshot's licences are re-resolved
automatically (§4).

**Record selection inside the fetch.** Each DOAJ record becomes one corpus
entry: a provenance header naming journal, year, language, DOI and ISSNs,
then the abstract. Two rules apply:

- **Markup and entities.** Tags are stripped before entities are decoded
  (decoding first would let a `&lt;i&gt;` become strippable markup — #80's
  arXiv lesson), and whitespace is collapsed.
- **The stub gate.** An abstract under 30 English words is not register text:
  DOAJ's editorial stubs ("Comment.", "Introduction.", "Abstract not
  available.") and non-English abstracts sit under the floor, so they appear
  in neither the corpus nor the ledger.

**Size of the pinned snapshot** (verified 2026-10-16): 922 records, 920
distinct DOIs, 60 journals, 53 publishers, years 2020-2026 (582 records
2026); **249,872 tokens / 17,402 types** by the #24 word regex, well above
the 10,000-token inferential floor (the module gate is 100,000). The corpus
carries 969 key and 2,314 anti-key words at q < 0.05. Two DOIs appear twice
in the DOAJ index itself (both University of Vienna Law Review); both copies
stay in the register text and the ledger collapses them to one row per DOI —
the licence is a property of the record, not of the copy.

**A disclosed limit of the selection rule.** `bibjson.journal.language` is a
*journal*-level field, so a bilingual journal's non-English abstracts pass
the filter. 283 of the 920 records come from EN/ID journals, and their
Indonesian function words are visible in the anti-key table (`hukum` 1,093,
`penelitian` 1,078, `yang` 1,046, `dan` 1,486 — a few per cent of the
register's tokens). This is a property of the rule, disclosed rather than
filtered: there is no article-level language field in the search API, a
language classifier would be an undisclosed screen, and the leak is visible
in the published table where a reader can see it.

## 4. Rights and access terms — per record, not blanket

DOAJ's inclusion criteria make every listed journal open access, but the
*licence* is a per-journal fact, so it is resolved, never asserted. For each
record's journal ISSN(s), `work/reference_corpora.py` queries DOAJ's
`search/journals/issn:...` (batched 50 ISSNs per request through the
module's retrying `download` helper), reads the journal's `license` type, and
writes `analysis/reference_corpora/legal_abstracts/rights.tsv` — one row per
DOI with journal, publisher, year, language, ISSNs, the verbatim licence, a
derived class, the verification route and the verification date (the
module's one wall-clock value):

| rights_class | meaning |
|---|---|
| `open_access` | the journal's DOAJ record carries a licence (the CC variants DOAJ lists) |
| `free_to_read` | DOAJ-listed but no licence recorded: free to read, redistribution not licensed |
| `unverified` | the journal lookup returned nothing, or the record has no ISSN — **never** treated as open |

**Result for the pinned snapshot** (verified 2026-10-16, all 920 rows dated):

| class | records | of which |
|---|---:|---|
| open_access | 920 | 525 `cc by`, 171 `cc by-nc`, 156 `cc by-nc-nd`, 54 `cc by-sa`, 14 `cc by-nc-sa`, 2 `cc by-nd` |
| free_to_read | 0 | — |
| unverified | 0 | — |

The ledger is bound to the corpus, not to the query: the corpus file's
headers carry the DOIs and ISSNs, so a cached corpus re-resolves its licences
from its own bytes. A cached corpus keeps its ledger when the ledger covers
exactly the corpus's DOI set; a stale or missing ledger is re-resolved in
download mode and reported as a warning (network-free) under `--no-download`
— a rights claim is never silently assumed. The manifest's licence cell for
the register is "open access (DOAJ); per-record licence in
legal_abstracts/rights.tsv" — the policy statement plus the pointer, never a
blanket claim in place of the ledger.

**Access terms.** DOAJ's public API is used for the corpus and for the
licence lookups (no API key; the build runs 12 requests, far under any
polite-use limit). The licence strings are reproduced verbatim from the
journal records.

## 5. Intended reuse

- Internal comparison for the #56 keyness harness: frequency and keyness
  statistics derived from the abstracts, with the corpus identified by
  journal + DOI (the corpus file's headers and the ledger both carry them).
- Quotations, where any are published, are attributed to journal and DOI.
- The abstract files are **not** redistributable as a corpus: 343 of the 920
  records sit under `cc by-nc*` / `cc by-nd*` licences that forbid commercial
  reuse or derivatives, and the remaining `cc by` / `cc by-sa` records still
  require attribution. A fresh clone rebuilds the snapshot from the query
  above rather than from a copy.
- A fully redistributable alternative, if one is ever needed, is the CC-BY
  subset obtainable by filtering `rights.tsv` on `license = CC BY` (525 of
  920 records). It was considered and rejected as the main corpus: breadth is
  the point, and a licence-filtered sample is a different register population.
- **Scope.** The register is wired into the keyness harness, where the plan
  placed legal abstracts. `genre_prior` (#60), `intertext` (#57) and
  `pos_tagging` (#37) were deliberately **not** re-armed: each has an
  explicit register list, their published numbers stand unchanged, and
  `pos_tagging`'s only movement is its corpus-file inventory (29 → 30) from
  re-running its offline build over the new directory.

## 6. Alternatives considered

- **Keep the historical register only and shrink the residue claim.** The
  honest fallback: drop "modern legal writing" from the report and let the
  residue verdict rest on two modern registers. Rejected because the measured
  effect (§1) is real — five words would stay misclassified — and because the
  plan asked for the register; the fix costs one pinned snapshot.
- **Court opinions** (CourtListener / the Caselaw Access Project). A
  different register — full texts of judgments, not abstracts — and the plan
  asked for abstracts. It remains the identified candidate if a case-law
  register is ever wanted; it is not what the technical-register slot asked
  for.
- **Subscription services** (HeinOnline, Westlaw, Lexis): no bulk access,
  no redistributability. Not viable for a committed corpus.
- **SSRN** abstracts: no open API for abstracts. Not viable.
- **The broader DOAJ subject query** (`bibjson.subject.term:(law)`,
  ~217k articles). Bigger, but the journal-title rule mirrors medical's
  `disease[Title]` and states plainly what the sample is; the broader query
  mixes in journals whose subject merely classifies as law.
- **Restrict to CC-BY records.** Clean rights by construction, but the
  sample shrinks by ~43% and skews by licence — a different register,
  not a cleaner one. Rejected as the main corpus; available as a subset (§5).
- **Rename the historical register instead of adding a modern one.** Done
  anyway (§2): the labels are corrected regardless, and the tests pin them.

## 7. Regenerating / re-verifying

```bash
python work/reference_corpora.py            # rebuild + resolve licences on a fresh fetch
python work/reference_corpora.py --no-download   # cache-only; skips the DOAJ fetch,
                                               #   warns if the ledger is stale
python work/keyness.py                      # re-score with the new register
python work/test_reference_corpora.py       # 86 tests, offline
python work/test_keyness.py                 # 45 tests, offline
```

A fresh fetch re-resolves every journal it fetched and rewrites
`legal_abstracts/rights.tsv`. A cached corpus keeps its ledger when the
ledger covers exactly its DOIs; a changed snapshot (drift) or a missing
ledger triggers re-resolution in download mode and a loud, network-free
warning under `--no-download`.
