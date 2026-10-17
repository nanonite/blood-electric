# The medical abstract reference corpus — evaluation and curation (#78)

Optional domain-specific comparison corpus for the #56 keyness harness,
evaluated and curated 2026-10-16. This note records the decision, the record
selection, the provenance and the rights/access terms, so the corpus's
position is auditable rather than remembered.

## 1. Decision: retained, with the rights claim corrected

The question was whether the medical abstract register improves the intended
analyses enough to keep. It does, and it is load-bearing in two places:

- **#56 keyness.** The corpus is one of the four *inferential* references
  (380,072 tokens / 25,536 types, well above the 10,000-token floor). The
  32-word reference-independent core is computed as agreement across every
  inferential corpus, so it includes this one; and the 13 words whose keyness
  survives no technical register — the "reference-dependent residue" — are
  decided by `medical` and `cs`, not by the (much older) general-English
  reference. Dropping the medical corpus would remove one of the three
  technical registers and leave the residue verdict resting on two.

  **(#79 update.)** Since this memo was written, a modern legal-abstract
  register has joined the technical union (`legal_abstracts`, DOAJ — see
  `docs/legal-corpus.md`). The register set is now five inferential corpora
  with three *modern* technical ones; the reference-independent core is still
  32 words, and the residue this memo computed as 13 words is now **8**
  (`discharge`, `exceeded`, `receives`, `reverse` and `rotation` were decided
  by the modern legal register, which the historical legal texts could not
  decide). The ledger and every other number in this memo are unchanged.
- **#60 genre/prior dossier.** The medical register is the pole that makes
  two verdicts readable rather than merely significant: `clinical` runs at
  7.02/1k in the book against 12.35/1k in medical abstracts and 0.155/1k in
  prose, and `virus` at 8.15 against 9.87 and 0.058. The dossier's
  "a novel with a medical accent, not a medical text" is a measurement
  *between* those poles.

So the corpus stays. What changes is the claim attached to it. Until this
pass the manifest labelled the whole corpus **"open access (PubMed
abstracts)"** — the same claim appeared in #57's acquisition ledger and the
README. It is false for most of the records.

## 2. Source, record selection and provenance

**Source.** PubMed, via NCBI E-utilities: `esearch` with
`term=disease[Title]`, `retmax=1000`, `sort=pub_date`, then `efetch`
(`rettype=abstract`, `retmode=text`) in batches of 200 ids.

**Record selection.** The 1,000 records are the newest PubMed records returned
for `disease[Title]` at fetch time. This query does not add PubMed's
`medline[sb]` subset filter, so the sample is not described as exclusively
MEDLINE-indexed. The selection rule is arbitrary but explicit; what makes it
usable is that the *sample itself* is pinned:
the corpus bytes live in `analysis/reference_corpora/medical/` and their
SHA-256 is recorded in `manifest.tsv` as an `observed_checksum`, which #80's
drift mechanism treats as the file's expected pin — a rebuild that produces
different bytes is reported (`manifest_drift.tsv`) rather than substituted.

**Snapshot stability — read this before re-running the query.** The query
is recency-sorted, so re-running it returns a *different* sample (a newer
1,000 records). The stable unit is therefore the pinned snapshot, not the
query. Any future rebuild that should reproduce a known sample must narrow
the query to a publication-date window (`datetype=pdat` with `mindate` /
`maxdate`) and record the window; the drift report will otherwise fire, and
the new snapshot's rights are re-verified automatically (see §3).

## 3. Rights and access terms — per record, not blanket

PubMed *indexes* abstracts whose copyright stays with the publisher (the
efetch text itself carries `Copyright © 2026 Elsevier Inc. All rights
reserved.` lines). "Open access" is a per-record fact, so the build now
verifies every record instead of asserting it.

**Method.** `work/reference_corpora.py` extracts the snapshot's PMIDs
(`parse_pmids`) and queries Europe PMC's REST core record for each one (200
ids per request, through the module's retrying `download` helper), reading
`license`, `isOpenAccess`, `inEPMC`, `inPMC` and `hasPDF`. The answers are
written to `analysis/reference_corpora/medical/rights.tsv`, one row per
record, with a derived class:

| rights_class | meaning |
|---|---|
| `open_access` | a licence string is recorded (`cc by`, `cc by-nc`, `cc by-nc-nd`, `cc0`) |
| `free_full_text` | deposited in Europe PMC / PMC, no licence recorded — free to read, redistribution not licensed |
| `publisher_copyright` | no Europe PMC / PMC deposit recorded |
| `unverified` | Europe PMC returned no record for the id — **never** treated as open |

The ledger carries a verification date (the module's one wall-clock value):
licences move, and an undated rights claim cannot be re-checked.

**Result for the pinned snapshot** (verified 2026-10-16):

| class | records | of which |
|---|---:|---|
| open_access | 602 | 296 `cc by`, 183 `cc by-nc`, 122 `cc by-nc-nd`, 1 `cc0` |
| free_full_text | 21 | in PMC / Europe PMC, licence unrecorded |
| publisher_copyright | 376 | the old label's false positive |
| unverified | 1 | see below |

The old label claimed 1,000 open-access records; the verified number is 602.

**The one unverified record, disclosed rather than dropped.** Its id string
is `PMID: 42832289181`, which is corrupt in the snapshot itself: PubMed's
text output glued record 181's citation header onto record 180's PMID line
(`PMID: 42832289181. JCEM Case Rep. ...`). The underlying record is
identifiable by its DOI, `10.1097/MEG.0000000000003269`, i.e. PMID 42832289,
which Europe PMC records as publisher copyright — so treating the id as
unverified changes no analytical outcome. The ledger keeps it `unverified`
because an unknown id is an unknown id.

**What the correction does not do.** It does not restrict the corpus to the
CC-licensed subset. The register is a *measuring* instrument for the
book's vocabulary, and a `rights.tsv` of full-text-deposited, licence-bearing,
largely NIH-funded records would be a different (smaller, biased) sample.

**Access terms.** NCBI E-utilities are used for the corpus itself (no API
key: this build runs well under the 3-requests/second limit). Europe PMC is
used only as rights metadata; its own terms of use cover the ledger's
metadata, and the per-record licence strings are reproduced verbatim.

## 4. Intended reuse

- Internal comparison for the #56 keyness, #60 genre-prior, #37 POS-balance,
  #44 affect and #57 intertext analyses: frequency and keyness statistics
  derived from the abstracts, with the corpus identified by journal + PMID.
- Quotations, where any are published, are attributed to journal and PMID.
- The abstract files are **not** redistributable as a corpus: records under
  `publisher_copyright` / `free_full_text` may not be republished, and
  `cc by-nc*` records forbid commercial reuse. A fresh clone rebuilds the
  snapshot from the query above rather than from a copy.
- A fully redistributable alternative, if one is ever needed, is the
  CC-licensed subset obtainable by filtering `rights.tsv` on
  `rights_class = open_access` — or by re-querying Europe PMC with
  `OPEN_ACCESS:Y`. It was considered and rejected as the main corpus (§1:
  breadth is the point; §5: the subset is biased).

## 5. Alternatives considered

- **Restrict to open-access records only** (PubMed `open access[filter]` /
  Europe PMC `OPEN_ACCESS:Y`). Clean rights by construction, but the sample
  shrinks (~60% here) and skews toward CC-licensed, PMC-deposited,
  disproportionately NIH-funded journals — a different register population,
  not a cleaner one. Rejected as the main corpus; available as a subset.
- **Keep the blanket label.** Rejected: it is a false statement about ~40%
  of the records, in a workspace whose audit (#76) exists to catch exactly
  this kind of confident over-claim.
- **Drop the corpus.** Rejected (§1): it is load-bearing for two published
  analyses and its loss would weaken both verdicts.
- **The arXiv (`cs`) label is untouched**: arXiv's abstracts are open by
  author-granted licence as a matter of policy, so "open access (arXiv
  abstracts)" is a policy-level claim, verified per record differently (by
  the archive's terms) than PubMed's.

## 6. Regenerating / re-verifying

```bash
python work/reference_corpora.py            # rebuild + re-verify rights on a fresh fetch
python work/reference_corpora.py --no-download   # cache-only; skips verification,
                                              # warns if the ledger is stale
python work/test_reference_corpora.py       # 57 tests, offline
```

A fresh fetch re-verifies every record it fetched and rewrites
`medical/rights.tsv`. A cached corpus keeps its ledger when the ledger
covers exactly its PMIDs; a changed snapshot (drift) or a missing ledger
triggers re-verification in download mode and a loud, network-free warning
under `--no-download` — a rights claim is never silently assumed, and the
manifest, #57's `sources.tsv` and the README all now point here.
