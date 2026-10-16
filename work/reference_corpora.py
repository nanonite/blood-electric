#!/usr/bin/env python3
"""#56 Reference corpora for keyness -> analysis/reference_corpora/

Assembles the eight reference registers that the keyness harness
(``work/keyness.py``) compares *Blood Electric* against:

  general_english    public-domain English prose.  The eight pinned #24 baselines
                    (``analysis/frequency/reference_cache/``) plus seven more
                    public-domain novels/ novellas, so the register spans
                    Austen to Kafka and is not just "the #24 eight".
  cyberpunk         Gibson / Stephenson / Sterling.  All three are in copyright,
                    so this is a small **fair-use excerpt** corpus: a handful of
                    short, attributed quotations from each signature work.
  cutup_surrealist  Burroughs / Artaud / Breton.  Same fair-use excerpt
                    treatment (Burroughs died 1997, Artaud's English
                    translations and Breton's *Nadja* are still under copyright
                    in the US).
  finnegans_wake    Joyce.  *Finnegans Wake* (1939) is still in US copyright
                    (95 years from publication -> 2034), so this is a fair-use
                    excerpt corpus, not the full text.
  medical           PubMed abstracts, kept as a pinned snapshot.  The
                    E-utilities query (``disease[Title]``, retmax=1000,
                    sort=pub_date) is recency-sorted, so re-running it
                    returns a *different* sample: the stable unit is the
                    committed corpus and its observed checksum, and a rebuild
                    that changes it is reported as drift.  Rights are verified
                    per record against Europe PMC and written to
                    ``medical/rights.tsv`` -- PubMed abstracts are the
                    publishers' copyright unless a licence is recorded, so the
                    corpus is never labelled blanket open access
                    (docs/medical-corpus.md, #78).
  cs                arXiv abstracts (open access), categories cs.AI / cs.LG /
                    cs.CL.  Live API query; same observed-checksum +
                    drift-reporting treatment as medical.
  legal             **historical** legal/political *full texts* (public domain,
                    Gutenberg): the Federalist Papers (1787-88), the
                    Declaration of Independence (1776), Magna Carta (1215),
                    Bastiat's *The Law* (1850) and Rousseau's *The Social
                    Contract* (1762).  This is the historical legal register:
                    it is **not** legal abstracts and **not** modern legal
                    writing, and no claim in this workspace may describe it
                    as either (docs/legal-corpus.md, #79).
  legal_abstracts   modern law-journal abstracts from the Directory of Open
                    Access Journals (DOAJ): journals whose title contains
                    "law", English-language, recency-sorted, 1,000 records --
                    a pinned snapshot with the same observed-checksum +
                    drift-reporting treatment as medical.  The register is
                    open access as a matter of DOAJ policy, but the licence
                    is a per-journal fact, so every journal's DOAJ record is
                    looked up and the answer is written to
                    ``legal_abstracts/rights.tsv``: the per-record ledger is
                    the claim, never a blanket label (#79).

Every corpus is written under ``analysis/reference_corpora/<corpus>/`` and the
provenance of every file is recorded in ``manifest.tsv`` (source, licence,
SHA-256, token count).  The manifest carries **no wall-clock date**, so it is
byte-for-byte reproducible given the same downloads.

**What the manifest SHA-256 values are.**  Every value is a checksum of the
exact bytes in the tree, and the ``hash_role`` column records what each one
*means*, because the two roles are enforced differently:

  enforced_pin       the eight #24 baselines (``PINNED_GENERAL``).  Their
                     hashes are constants in this script; a copy whose hash
                     does not match **aborts the build** -- the bytes are
                     never silently substituted.
  observed_checksum  everything else (the extra Gutenberg novels, the legal
                     register, the excerpt files, the live PubMed / arXiv
                     corpora).  The value records what this build produced;
                     the previously published manifest's value for the same
                     file is treated as its **expected pin**, and a rebuild
                     that yields different bytes is reported as drift --
                     logged and written to ``manifest_drift.tsv`` beside the
                     manifest -- instead of silently replacing the row.  A
                     fresh clone has no prior manifest, so the first build
                     records observations only.

**``--no-download`` is cache-only for every source.**  The live PubMed /
arXiv / DOAJ queries are skipped, and so is any general-English or legal
Gutenberg text that is not already cached: those are skipped with a warning
rather than fetched, so the flag can never be surprised by a network call.
(The eight pinned baselines are copied from #24's local cache, a file copy
rather than a network fetch; a missing #24 cache is a hard error telling you
to run #24.)
A cached medical corpus keeps its per-record rights ledger when that ledger
still covers exactly the corpus's PMIDs; a stale or missing ledger is
re-verified in download mode and reported as a warning under ``--no-download``,
so a rights claim is never silently assumed.  The same holds for
``legal_abstracts``, whose ledger is keyed by the DOIs (and ISSNs) the corpus
file itself carries, so a cached corpus re-resolves its licences from its own
bytes.

The fair-use excerpt files are embedded below as string constants and written
verbatim, so a fresh clone regenerates them from the committed script alone.

Reproduce::

    conda activate marker
    python work/reference_corpora.py            # downloads + assembles
    python work/reference_corpora.py --no-download   # cache only
    python work/test_reference_corpora.py
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "reference_corpora"
FREQ_CACHE = ROOT / "analysis" / "frequency" / "reference_cache"

SCHEMA = "reference-corpora/1"

USER_AGENT = ("blood-electric-workspace/1.0 "
              "(https://github.com/; research; contact via repo)")
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# ---------------------------------------------------------------------------
# source tables
# ---------------------------------------------------------------------------

# The eight pinned #24 public-domain prose baselines.  They already live in
# analysis/frequency/reference_cache/ with pinned SHA-256 hashes; we copy them
# into the general_english corpus so that corpus is self-contained.
PINNED_GENERAL = [
    ("pride_and_prejudice", "Jane Austen", 1342,
     "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"),
    ("moby_dick", "Herman Melville", 2701,
     "907420db6c4b68c70e2988cd2ad9c8cf79138667a01b63376d18dd17fef1a18b"),
    ("frankenstein", "Mary Shelley", 84,
     "7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3"),
    ("dracula", "Bram Stoker", 345,
     "96cd16eacdbfebae8fdda5591f66e0cc8ee76be18e0cd1aca02bc00615782d28"),
    ("sherlock_holmes", "Arthur Conan Doyle", 1661,
     "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0"),
    ("heart_of_darkness", "Joseph Conrad", 219,
     "c0b0bc91c7695f9d01aacb240e82a9b559f57558f98ad0b1f167eba21f6be6f7"),
    ("dorian_gray", "Oscar Wilde", 174,
     "38f36b510417177aa87a6a24c968e3ec63a447b7df36a9c1a7c5a3f2d9e51547"),
    ("ulysses", "James Joyce", 4300,
     "e03094626f9528cf3fc287a49d6edbdbf47cd40483cb13ceb301e74e15ebbf9e"),
]

# Seven further public-domain English prose texts, downloaded fresh.
EXTRA_GENERAL = [
    ("great_gatsby", "F. Scott Fitzgerald", 64317),
    ("alice_in_wonderland", "Lewis Carroll", 11),
    ("the_time_machine", "H. G. Wells", 35),
    ("the_war_of_the_worlds", "H. G. Wells", 36),
    ("a_tale_of_two_cities", "Charles Dickens", 98),
    ("great_expectations", "Charles Dickens", 1400),
    ("metamorphosis", "Franz Kafka", 5200),
]

# Public-domain **historical** legal / political register.  These are full
# texts, not abstracts, and they are centuries old: the register measures the
# book against the prose of past legal/political argument, not against modern
# legal writing (docs/legal-corpus.md, #79).  The modern legal-abstract
# register is built separately, from DOAJ (see DOAJ_QUERY below).
LEGAL_REGISTER_LABEL = ("historical legal/political full texts (public "
                        "domain); not legal abstracts, not modern legal "
                        "writing")
LEGAL = [
    ("federalist_papers", "Hamilton, Madison, Jay", 1404),
    ("declaration_of_independence", "Thomas Jefferson", 1),
    ("magna_carta", "Anonymous", 10000),
    ("the_law", "Frederic Bastiat", 44800),
    ("the_social_contract", "Jean-Jacques Rousseau", 46333),
]

# Live-query sources.
PUBMED_QUERY = "disease[Title]"
PUBMED_RETMAX = 1000
ARXIV_QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
ARXIV_MAX = 500

# #79: the modern legal-abstract register, from the Directory of Open Access
# Journals.  The selection rule mirrors medical's ``disease[Title]``: journals
# whose *title* contains "law", English-language, recency-sorted
# (``created_date``), 1,000 records.  A recency-sorted live query is not a
# stable sample, so -- exactly as for medical -- the stable unit is the pinned
# snapshot and its observed checksum, and a rebuild that fetches different
# bytes is reported as drift.  The broader alternative
# (``bibjson.subject.term:(law)``, ~217k articles) was considered and left
# documented in docs/legal-corpus.md.
DOAJ_SEARCH = "https://doaj.org/api/v2/search/articles/{query}"
DOAJ_JOURNAL = "https://doaj.org/api/v2/search/journals/{query}"
DOAJ_QUERY = "bibjson.journal.title:(law) AND bibjson.journal.language:EN"
DOAJ_MAX = 1000
DOAJ_PAGE = 100
# A real law abstract runs to hundreds of words.  DOAJ editorial stubs
# ("Comment.", "Introduction.", "Abstract not available.") and non-English
# abstracts sit under this floor, so the selection rule skips them: they are
# in neither the corpus nor the ledger.
MIN_ABSTRACT_WORDS = 30
# Licence lookups are per journal (by ISSN), batched through one OR query.
DOAJ_RIGHTS_CHUNK = 50

# ---------------------------------------------------------------------------
# fair-use excerpt corpora (copyrighted works; short attributed quotations)
# ---------------------------------------------------------------------------
# Each entry is a plain-text file: a header block naming the author, the
# works quoted, the year, and the fair-use rationale, followed by the
# quotations.  These are embedded so the files are reproducible from the
# committed script alone.

EXCERPTS = {
    "cyberpunk/gibson.txt": """\
# William Gibson - fair-use excerpts for linguistic research
# Quoted from: Neuromancer (1984), Burning Chrome (1982).
# Copyright held by the author / publisher.  Short quotations reproduced for
# register comparison only; not a substitute for the works.

"The sky above the port was the color of television, tuned to a dead channel."
-- Neuromancer (1984)

"Cyberspace. A consensual hallucination experienced daily by billions of legitimate operators, in every nation, by children being taught mathematical concepts."
-- Neuromancer (1984)

"The matrix has its roots in primitive arcade games."
-- Neuromancer (1984)

"The future is already here - it's just not evenly distributed."
-- William Gibson
""",
    "cyberpunk/stephenson.txt": """\
# Neal Stephenson - fair-use excerpts for linguistic research
# Quoted from: Snow Crash (1992), The Diamond Age (1995).
# Copyright held by the author / publisher.  Short quotations reproduced for
# register comparison only; not a substitute for the works.

"The Deliverator belongs to an elite order, a hallowed subcategory. He's got esprit up to here."
-- Snow Crash (1992)

"The Deliverator's car has enough potential energy packed into its batteries to fire a pound of bacon into the Asteroid Belt."
-- Snow Crash (1992)
""",
    "cyberpunk/sterling.txt": """\
# Bruce Sterling - fair-use excerpts for linguistic research
# Quoted from: Schismatrix (1985), Holy Fire (1996).
# Copyright held by the author / publisher.  Short quotations reproduced for
# register comparison only; not a substitute for the works.

"Now, for the first time, the President began to look genuinely powerful, even dangerous."
-- Schismatrix (1985)
""",
    "cutup_surrealist/burroughs.txt": """\
# William S. Burroughs - fair-use excerpts for linguistic research
# Quoted from: The Ticket That Exploded (1962), Naked Lunch (1959).
# Copyright held by the author's estate / publisher.  Short quotations
# reproduced for register comparison only; not a substitute for the works.

"Language is a virus from outer space."
-- The Ticket That Exploded (1962)

"A paranoid is someone who knows a little of what's going on."
-- William S. Burroughs

"Silence is only frightening to people who are compulsively verbalizing."
-- William S. Burroughs
""",
    "cutup_surrealist/artaud.txt": """\
# Antonin Artaud - fair-use excerpts for linguistic research
# Quoted from: The Theatre and Its Double (1938).
# Copyright held by the author's estate / publisher.  Short quotations
# reproduced for register comparison only; not a substitute for the works.

"No one has ever written, painted, sculpted, modeled, built, or invented except literally to get out of hell."
-- The Theatre and Its Double (1938)
""",
    "cutup_surrealist/breton.txt": """\
# Andre Breton - fair-use excerpts for linguistic research
# Quoted from: Nadja (1928), Manifestoes of Surrealism (1929).
# Copyright held by the author's estate / publisher.  Short quotations
# reproduced for register comparison only; not a substitute for the works.

"Beauty will be convulsive or will not be all."
-- Nadja (1928)
""",
    "finnegans_wake/joyce.txt": """\
# James Joyce - fair-use excerpts for linguistic research
# Quoted from: Finnegans Wake (1939).
# Copyright held by the author's estate / publisher.  Short quotations
# reproduced for register comparison only; not a substitute for the works.

"riverrun, past Eve and Adam's, from swerve of shore to bend of bay, brings us by a commodius vicus of recirculation back to Howth Castle and Environs."
-- Finnegans Wake (1939)

"The fall (bababadalgharaghtakamminarronnkonnbronntonnerronntuonnthunntrovarrhounawnskawntoohoohoordenenthurnuk!) of a once wallstrait oldparr is retaled early in bed and later on life down through all christian minstrelsy."
-- Finnegans Wake (1939)
""",
}

# ---------------------------------------------------------------------------
# io helpers (quote-unsafe TSVs: split on tab, never a csv reader -- #17)
# ---------------------------------------------------------------------------
def cell(value) -> str:
    s = "" if value is None else str(value)
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(cell(r.get(h, "")) for h in header) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# download with retry / backoff
# ---------------------------------------------------------------------------
def download(url: str, path: Path, log=print, retries: int = 4,
             timeout: int = 90) -> bytes:
    """GET *url* to *path* with exponential backoff; returns the bytes."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return data
        except Exception as exc:  # noqa: BLE001 - any transient failure retries
            last = exc
            if attempt < retries - 1:
                wait = 2 ** attempt
                log(f"[reference_corpora] download failed ({exc}); "
                    f"retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"download failed after {retries} tries: {url} ({last})")


def strip_gutenberg(text: str) -> str:
    """Drop the Project Gutenberg header / footer (same rule as #24)."""
    import frequency_zipf as fz  # #24 helper, local import
    return fz.strip_gutenberg(text)


# ---------------------------------------------------------------------------
# validated corpus writes
# ---------------------------------------------------------------------------
# Every corpus write goes through ``write_corpus``.  Without this gate a
# degenerate fetch (an arXiv error page, an empty efetch body, a Gutenberg 404
# page) silently replaces a good cached corpus with content that then flows
# into the keyness tables as if it were real reference text.  The gate is what
# makes the build idempotent: a second run over intact caches is a no-op, and a
# run whose fetch fails leaves the previous corpus untouched and reports.
MIN_TOKENS = {
    "cs": 50_000, "medical": 100_000, "legal_abstracts": 100_000,
}
# Floor distinguishing a real work from a fetch artefact (Gutenberg 404 page,
# redirect stub).  Set below the smallest genuine document in these corpora
# (the Declaration of Independence, ~1.7k tokens) and far above the ~100 words
# of boilerplate an error page carries.
MIN_TOKENS_GUTENBERG = 1_000


def corpus_tokens(data: bytes) -> int:
    """English-word token count of *data* (#24 word regex)."""
    import frequency_zipf as fz  # #24 helper, local import
    return len(fz.EN_WORD_RE.findall(data.decode("utf-8", errors="replace")))


def corpus_ok(path: Path, min_tokens: int) -> bool:
    """True when *path* already holds a corpus clearing the token gate."""
    return (path.exists() and path.stat().st_size > 0
            and corpus_tokens(path.read_bytes()) >= min_tokens)


def write_corpus(path: Path, data: bytes, min_tokens: int, log=print) -> int:
    """Validate *data* against *min_tokens*, then atomically replace *path*.

    Returns the token count.  Raises ValueError when the gate fails, leaving
    any existing corpus byte-for-byte intact: a bad fetch is reported, never
    absorbed.  The write goes to a sibling temp file and is renamed into place,
    so an interrupted run cannot leave a half-written corpus behind.
    """
    n_tok = corpus_tokens(data)
    if n_tok < min_tokens:
        raise ValueError(
            f"{path.name}: {n_tok:,} tokens < {min_tokens:,} gate; refusing to "
            f"overwrite {path} (a bad fetch is reported, never absorbed)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    log(f"[reference_corpora] wrote {path.name}: {len(data):,} bytes, "
        f"{n_tok:,} tokens")
    return n_tok


# ---------------------------------------------------------------------------
# provenance rows for the two live-query corpora
# ---------------------------------------------------------------------------
# Shared by the download and the --no-download path so the manifest is
# byte-identical whichever mode built it.
LIVE_SOURCES = {
    "medical": (
        (f"pubmed eutils esearch efetch term={PUBMED_QUERY} "
         f"retmax={PUBMED_RETMAX} sort=pub_date"),
        # #78: not "open access" -- the publishers hold the copyright in most
        # of these abstracts, and only the per-record ledger can say which.
        "publisher copyright; per-record rights in medical/rights.tsv"),
    "cs": (f"arxiv.org api query={ARXIV_QUERY} max_results={ARXIV_MAX}",
           "open access (arXiv abstracts)"),
    # #79: DOAJ-listed journals are open access as a matter of policy, but
    # the licence is a per-journal fact -- resolve it per record in
    # legal_abstracts/rights.tsv, never assert it blanket.
    "legal_abstracts": (
        (f"doaj.org api search articles query={DOAJ_QUERY} max={DOAJ_MAX} "
         f"sort=created_date:desc"),
        ("open access (DOAJ); per-record licence in "
         "legal_abstracts/rights.tsv")),
}
LIVE_FILES = {
    "medical": "medical/pubmed_abstracts.txt",
    "cs": "cs/arxiv_abstracts.txt",
    "legal_abstracts": "legal_abstracts/legal_abstracts.txt",
}


def live_row(corpus: str, data: bytes) -> dict:
    """The manifest row for a live-query corpus, from its bytes."""
    source, license_ = LIVE_SOURCES[corpus]
    return {
        "corpus": corpus, "file": LIVE_FILES[corpus], "source": source,
        "license": license_, "author": "various",
        "sha256": sha256_bytes(data), "bytes": len(data),
    }


# ---------------------------------------------------------------------------
# per-record rights verification for the medical corpus (#78)
# ---------------------------------------------------------------------------
# PubMed *indexes* publisher-copyrighted abstracts; the efetch text itself
# carries "Copyright (c) ... All rights reserved" lines, so "open access" is a
# per-record fact and not a property of the corpus.  Every record in the
# snapshot is therefore looked up in Europe PMC, which records a licence string
# and an open-access flag per record, and the answers are written to
# ``medical/rights.tsv`` beside the corpus.  A record Europe PMC does not
# return is ``unverified`` and treated as *not* open: the ledger never upgrades
# an unknown into a right, and it covers exactly the PMIDs of the snapshot it
# was built from.  Unlike the manifest, the ledger *does* carry a verification
# date: licences move, and a rights claim without a date cannot be re-checked.

EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
RIGHTS_FILE = "medical/rights.tsv"
RIGHTS_HEADER = ["pmid", "license", "is_open_access", "in_epmc", "in_pmc",
                 "has_pdf", "rights_class", "verified_via", "verified_at"]
RIGHTS_CHUNK = 200

PMID_RE = re.compile(r"PMID:\s*(\d+)")


def parse_pmids(data: bytes) -> list:
    """The PMIDs of an efetch ``rettype=abstract`` body, in file order.

    Decorations (``PMID: 123 [Indexed for MEDLINE]``) are ignored, so the list
    is the snapshot's record set rather than only its clean lines.
    """
    text = data.decode("utf-8", errors="replace")
    return list(dict.fromkeys(PMID_RE.findall(text)))


def rights_class(license_: str, in_epmc: str) -> str:
    """The reuse class of one verified record.

    open_access          a licence string is recorded (``cc by``, ``cc by-nc``,
                         ...): reuse under that licence is verified per record.
    free_full_text       deposited in Europe PMC / PMC with no licence recorded:
                         free to read, redistribution not licensed.
    publisher_copyright  no Europe PMC / PMC deposit recorded: the abstract
                         stays the publisher's copyright.
    """
    if license_:
        return "open_access"
    if in_epmc == "Y":
        return "free_full_text"
    return "publisher_copyright"


def today_iso() -> str:
    """Today's date, UTC -- the one wall-clock value in this module.

    The manifest carries no date on purpose (byte-for-byte reproducibility),
    but a rights claim without a verification date cannot be re-checked and
    licences move, so ``medical/rights.tsv`` records when each row was
    verified (see ``fetch_rights``).
    """
    return datetime.now(timezone.utc).date().isoformat()


def fetch_rights(pmids, dest: Path, log=print, fetch=download) -> list:
    """Verify each PMID in *pmids* against Europe PMC; returns rights rows.

    Batched (200 ids per request: the URI-length limit that splits efetch
    applies here too) through the module's ``download`` helper, so a fetch
    failure retries and a test-level ``no_network`` patch covers these calls
    like any other fetch.  Responses are staged beside the corpus and unlinked,
    exactly like the efetch chunks.
    """
    rows = []
    n_chunks = (len(pmids) + RIGHTS_CHUNK - 1) // RIGHTS_CHUNK
    for n in range(0, len(pmids), RIGHTS_CHUNK):
        chunk = pmids[n:n + RIGHTS_CHUNK]
        query = "(" + " OR ".join(f"ext_id:{p}" for p in chunk) + ") AND src:med"
        url = EUROPE_PMC_SEARCH + "?" + urllib.parse.urlencode(
            {"query": query, "format": "json", "resultType": "core",
             "pageSize": len(chunk)})
        stage = dest / f"_rights_{n // RIGHTS_CHUNK}.json"
        log(f"[reference_corpora] verifying rights for {len(chunk)} PubMed "
            f"records against Europe PMC ({n // RIGHTS_CHUNK + 1}/{n_chunks})")
        seen = {}
        try:
            meta = json.loads(fetch(url, stage, log))
            for r in meta.get("resultList", {}).get("result", []):
                seen[str(r.get("pmid") or r.get("id"))] = r
        except ValueError:
            log("[reference_corpora] WARNING: Europe PMC response was not "
                "JSON; this batch's records are recorded unverified")
        for p in chunk:
            r = seen.get(p)
            if r is None:
                rows.append({
                    "pmid": p, "license": "", "is_open_access": "",
                    "in_epmc": "", "in_pmc": "", "has_pdf": "",
                    "rights_class": "unverified",
                    "verified_via": "Europe PMC core (no record returned)",
                    "verified_at": today_iso(),
                })
                continue
            license_ = r.get("license") or ""
            rows.append({
                "pmid": p, "license": license_,
                "is_open_access": r.get("isOpenAccess", ""),
                "in_epmc": r.get("inEPMC", ""),
                "in_pmc": r.get("inPMC", ""),
                "has_pdf": r.get("hasPDF", ""),
                "rights_class": rights_class(license_, r.get("inEPMC", "")),
                "verified_via": "Europe PMC core",
                "verified_at": today_iso(),
            })
        stage.unlink(missing_ok=True)
    return rows


def write_rights(path: Path, rows) -> str:
    """Write the per-record rights ledger (sorted by PMID); returns a summary.

    The summary ("23 open access, 1 free_full_text, ...") is returned so the
    caller logs the distribution the ledger records, not just its row count.
    """
    ordered = sorted(rows, key=lambda r: r["pmid"])
    write_tsv(path, RIGHTS_HEADER, ordered)
    counts = {}
    for r in ordered:
        counts[r["rights_class"]] = counts.get(r["rights_class"], 0) + 1
    return ", ".join(f"{n} {c}" for c, n in sorted(counts.items()))


def read_rights(path: Path) -> dict:
    """Ledger rows keyed by PMID; {} when absent or unreadable."""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row.get("pmid"):
                out[row["pmid"]] = row
    return out


def rights_ledger_covers(path: Path, pmids) -> bool:
    """True when the ledger holds exactly one row for every corpus PMID."""
    ledger = read_rights(path)
    return bool(pmids) and len(ledger) == len(pmids) and all(
        p in ledger for p in pmids)


# ---------------------------------------------------------------------------
# the modern legal-abstract register: DOAJ records + per-journal licences
# (#79)
# ---------------------------------------------------------------------------
# The register is open access as a matter of DOAJ policy, but the licence is
# a per-journal fact, so it is resolved, not asserted.  Each corpus record
# is a law-journal article whose abstract is the register text; the
# provenance (DOI, journal, publisher, year, language, ISSN) and the licence
# are written to ``legal_abstracts/rights.tsv`` beside the corpus, one row
# per record, with a verification date -- the medical ledger's structure
# (#78), applied to a different registry.  A record whose journal lookup
# returns nothing is ``unverified`` and is never treated as open.

DOAJ_RIGHTS_FILE = "legal_abstracts/rights.tsv"
DOAJ_RIGHTS_HEADER = ["doi", "journal", "publisher", "year", "language",
                      "issn", "license", "license_class", "verified_via",
                      "verified_at"]

# The per-record header the corpus file carries, e.g.
# "RUDN Journal of Law (2025; EN). doi: 10.22363/...; ISSN 2313-2337/2408-9001".
# The header is provenance (what the record is), while ``rights.tsv`` is the
# reuse position (what may be done with it).  It carries the DOI *and* the
# ISSNs so a cached corpus can re-resolve its licences from its own bytes --
# the medical pattern, where PMIDs in the efetch text bind a cached corpus
# to its ledger.
RECORD_HEADER_RE = re.compile(
    r"^(?P<journal>.+?) \((?P<year>[^;]*); (?P<language>[^)]*)\)\. "
    r"doi: (?P<doi>\S+?); ISSN (?P<issn>\S+)\s*$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")


def clean_abstract(raw: str) -> str:
    """Abstract text with markup stripped, entities decoded, space collapsed.

    Markup is stripped *before* entities are decoded (decoding first would
    turn a ``&lt;i&gt;`` into a tag that a second pass would then strip --
    the same re-entrancy lesson #80 recorded for the arXiv parser).  A
    decoded ``&lt;`` therefore stays a literal character, which is harmless
    to the #24 word regex.
    """
    text = HTML_TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def doi_of(record: dict) -> str:
    """A record's stable key: its DOI, or ``doaj:<id>`` when it has none."""
    for ident in record.get("bibjson", {}).get("identifier", []):
        if ident.get("type") == "doi" and ident.get("id"):
            return ident["id"]
    return f"doaj:{record.get('id', '')}"


def parse_doaj_records(pages) -> list:
    """Record dicts from DOAJ article-search JSON, stubs skipped.

    Each item is {doi, journal, publisher, year, language, issns, abstract}.
    The abstract word gate (``MIN_ABSTRACT_WORDS``) is the selection rule
    that skips DOAJ's editorial stubs and non-English abstracts: it is part
    of what the corpus *is*, so the skipped records appear in neither the
    corpus file nor the ledger.
    """
    records = []
    for page in pages:
        try:
            meta = json.loads(page)
        except ValueError:
            continue
        for r in meta.get("results", []):
            bib = r.get("bibjson", {})
            abstract = clean_abstract(bib.get("abstract", "") or "")
            if len(EN_WORD_RE.findall(abstract)) < MIN_ABSTRACT_WORDS:
                continue
            journal = bib.get("journal", {})
            records.append({
                "doi": doi_of(r),
                "journal": journal.get("title", ""),
                "publisher": journal.get("publisher", ""),
                "year": bib.get("year", "") or "",
                "language": "/".join(journal.get("language", []) or []),
                "issns": [i for i in (journal.get("issns", []) or []) if i],
                "abstract": abstract,
            })
    return records


def doaj_corpus_bytes(records) -> bytes:
    """The corpus file: a provenance header, then the abstract, per record.

    The header names journal, year, language, DOI and ISSNs, so every
    counted line of register text has its record beside it -- the medical
    corpus's shape (its efetch text carries citation headers and PMIDs),
    kept deliberately.  Because the ISSNs travel in the file, a cached
    corpus can re-resolve its licences from its own bytes (the medical
    pattern, where PMIDs in the text bind a cached corpus to its ledger).
    """
    parts = []
    for r in records:
        # "-" for a record with no ISSN: the header must stay parseable so
        # the corpus can always be read back against its ledger.
        issns = "/".join(r["issns"]) if r["issns"] else "-"
        head = (f"{r['journal']} ({r['year']}; {r['language']}). "
                f"doi: {r['doi']}; ISSN {issns}")
        parts.append(f"{head}\n\n{r['abstract']}")
    return "\n\n".join(parts).encode("utf-8")


def doaj_rights_class(license_: str) -> str:
    """The reuse class of one record, from its journal's licence.

    open_access          the journal's DOAJ record carries a licence (the CC
                         variants DOAJ lists): reuse under that licence is
                         verified per record.
    free_to_read         DOAJ-listed but no licence recorded: free to read,
                         redistribution not licensed.
    """
    return "open_access" if license_ else "free_to_read"


def resolve_doaj_licences(records, log=print, fetch=download) -> list:
    """Resolve each record's journal licence; returns the ledger rows.

    The licence lives on the *journal*, so the lookup is batched by ISSN
    (``DOAJ_RIGHTS_CHUNK`` ISSNs per request through one OR query, like the
    medical batching) through the module's retrying ``download`` helper.
    Responses are staged beside the destination and unlinked.  A journal the
    API does not return is ``unverified``: an unknown licence is never read
    as an open one, and a record with no ISSN at all cannot be looked up and
    is ``unverified`` for that stated reason.

    Used both from a fresh fetch (records parsed from the API pages) and
    from a cached corpus (records parsed back out of the corpus file), so
    the licence position is re-verifiable without refetching the corpus.
    """
    rows = []
    issn_rows = {}
    for r in records:
        row = {
            "doi": r["doi"], "journal": r["journal"],
            "publisher": r.get("publisher", ""), "year": r["year"],
            "language": r["language"], "issn": "/".join(r["issns"]),
            "license": "", "license_class": "unverified",
            "verified_via": "DOAJ journal record (no ISSN on the record)"
                            if not r["issns"] else "",
            "verified_at": ""}
        rows.append(row)
        for issn in r["issns"]:
            issn_rows.setdefault(issn, []).append(row)
    issns = sorted(issn_rows)
    n_chunks = (len(issns) + DOAJ_RIGHTS_CHUNK - 1) // DOAJ_RIGHTS_CHUNK
    seen = {}
    for n in range(0, len(issns), DOAJ_RIGHTS_CHUNK):
        chunk = issns[n:n + DOAJ_RIGHTS_CHUNK]
        query = " OR ".join(f"issn:{i}" for i in chunk)
        url = DOAJ_JOURNAL.format(query=urllib.parse.quote(query)) + \
              f"?pageSize={len(chunk)}"
        stage = OUT / Path(DOAJ_RIGHTS_FILE).parent / \
                f"_licences_{n // DOAJ_RIGHTS_CHUNK}.json"
        log(f"[reference_corpora] resolving licences for {len(chunk)} "
            f"journals against DOAJ ({n // DOAJ_RIGHTS_CHUNK + 1}/{n_chunks})")
        try:
            meta = json.loads(fetch(url, stage, log))
            for j in meta.get("results", []):
                jbib = j.get("bibjson", {})
                lic = (jbib.get("license") or [{}])[0].get("type", "")
                # A DOAJ journal record carries its ISSNs as ``pissn`` /
                # ``eissn`` (there is no ``issns`` array), so both fields
                # together are the record's ISSN set.
                j_issns = [i for i in (jbib.get("pissn"), jbib.get("eissn"))
                           if i]
                # A DOAJ journal's ``publisher`` is a {country, name} object.
                pub = jbib.get("publisher")
                if isinstance(pub, dict):
                    pub = pub.get("name", "")
                for issn in j_issns:
                    seen[issn] = (lic, pub or "")
        except ValueError:
            log("[reference_corpora] WARNING: DOAJ journal response was not "
                "JSON; this batch's journals are recorded unverified")
        stage.unlink(missing_ok=True)
    for issn, chunk_rows in issn_rows.items():
        hit = seen.get(issn)
        for row in chunk_rows:
            if hit is None:
                row["license"] = ""
                row["license_class"] = "unverified"
                row["verified_via"] = ("DOAJ journal record (no record "
                                       "returned)")
            else:
                lic, publisher = hit
                row["publisher"] = publisher or row["publisher"]
                row["license"] = lic
                row["license_class"] = doaj_rights_class(lic)
                row["verified_via"] = "DOAJ journal record"
            row["verified_at"] = today_iso()
    for row in rows:
        if row["verified_via"] == "DOAJ journal record (no ISSN on the record)":
            row["verified_at"] = today_iso()
    return rows


def write_doaj_rights(path: Path, rows) -> str:
    """Write the per-record licence ledger (sorted by DOI); returns a summary.

    The ledger is keyed by DOI, so a DOI the DOAJ index repeats (the snapshot
    contains two) collapses to one row: the licence is a property of the
    record, not of the copy.
    """
    by_doi = {}
    for r in rows:
        by_doi.setdefault(r["doi"], r)
    ordered = sorted(by_doi.values(), key=lambda r: r["doi"])
    write_tsv(path, DOAJ_RIGHTS_HEADER, ordered)
    counts = {}
    for r in ordered:
        counts[r["license_class"]] = counts.get(r["license_class"], 0) + 1
    return ", ".join(f"{n} {c}" for c, n in sorted(counts.items()))


def read_doaj_rights(path: Path) -> list:
    """Ledger rows in file order; [] when absent or unreadable."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row.get("doi"):
                out.append(row)
    return out


def parse_records_from_corpus(data: bytes) -> list:
    """Record dicts read back out of a corpus file, in file order.

    The header lines carry journal, year, language, DOI and ISSNs, so a
    cached corpus re-resolves its licences from its own bytes: no re-fetch
    of the corpus is needed to re-verify the rights position.  (The abstract
    itself is not needed for a licence lookup.)
    """
    text = data.decode("utf-8", errors="replace")
    records = []
    for line in text.splitlines():
        m = RECORD_HEADER_RE.match(line)
        if not m:
            continue
        issn = m.group("issn")
        records.append({
            "doi": m.group("doi"),
            "journal": m.group("journal"),
            "publisher": "",
            "year": m.group("year"),
            "language": m.group("language"),
            "issns": [] if issn == "-" else [i for i in issn.split("/") if i],
        })
    return records


def parse_dois(data: bytes) -> list:
    """The DOIs a corpus file carries, in file order.

    The corpus file's per-record header names the record's DOI, so the key
    set is recoverable from the cached bytes -- which is what binds a cached
    corpus to the ledger that covers it.
    """
    return list(dict.fromkeys(r["doi"] for r in parse_records_from_corpus(data)))


def doaj_rights_ledger_covers(path: Path, dois) -> bool:
    """True when the ledger holds exactly the corpus's DOI set.

    The DOI set, not the row count: the corpus file is record-per-abstract
    (a DOI the DOAJ index repeats appears twice in the register text), while
    the ledger is keyed by DOI.
    """
    ledger = read_doaj_rights(path)
    covered = {r["doi"] for r in ledger}
    return bool(dois) and covered == set(dois)


# ---------------------------------------------------------------------------
# corpus builders
# ---------------------------------------------------------------------------
def build_general_english(log=print, allow_download: bool = True) -> list:
    """Copy the 8 pinned #24 texts + download 7 more public-domain novels.

    With ``allow_download=False`` the builder is cache-only: a novel that is
    not already under ``general_english/`` is skipped with a warning instead
    of being fetched, and the network is never touched.
    """
    dest = OUT / "general_english"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, author, gid, pinned_hash in PINNED_GENERAL:
        src = FREQ_CACHE / f"{name}.txt"
        dst = dest / f"{name}.txt"
        if not src.exists():
            raise FileNotFoundError(
                f"pinned #24 reference missing: {src} (run #24 first)")
        data = src.read_bytes()
        got = sha256_bytes(data)
        if got != pinned_hash:
            raise ValueError(
                f"pinned #24 reference {name} hash mismatch: {got} != {pinned_hash}")
        if not dst.exists() or sha256_file(dst) != got:
            shutil.copyfile(src, dst)
            log(f"[reference_corpora] copied pinned {name}")
        rows.append({
            "corpus": "general_english", "file": f"general_english/{name}.txt",
            "source": f"gutenberg.org {gid} (pinned #24 cache)",
            "license": "public domain", "author": author,
            "sha256": got, "bytes": len(data),
        })
    for name, author, gid in EXTRA_GENERAL:
        dst = dest / f"{name}.txt"
        url = GUTENBERG_URL.format(id=gid)
        if dst.exists():
            data = dst.read_bytes()
        elif not allow_download:
            # Cache-only means cache-only: report the gap, never fetch.
            log(f"[reference_corpora] WARNING: general_english/{name}.txt not "
                f"cached and --no-download set; skipping (re-run without "
                f"--no-download to fetch gutenberg {gid})")
            continue
        else:
            log(f"[reference_corpora] downloading {name} <- gutenberg {gid}")
            data = download(url, dst, log)
        text = strip_gutenberg(data.decode("utf-8", errors="replace"))
        clean = text.encode("utf-8")
        write_corpus(dst, clean, MIN_TOKENS_GUTENBERG, log)
        rows.append({
            "corpus": "general_english", "file": f"general_english/{name}.txt",
            "source": url, "license": "public domain", "author": author,
            "sha256": sha256_bytes(clean), "bytes": len(clean),
        })
    return rows


def build_excerpts(log=print) -> list:
    """Write the fair-use excerpt files (embedded constants)."""
    rows = []
    for rel, text in EXCERPTS.items():
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode("utf-8")
        dst.write_bytes(data)
        corpus = rel.split("/", 1)[0]
        rows.append({
            "corpus": corpus, "file": rel,
            "source": "fair-use excerpts (see file header)",
            "license": "copyrighted; short quotations for research",
            "author": rel.split("/", 1)[1].replace(".txt", ""),
            "sha256": sha256_bytes(data), "bytes": len(data),
        })
        log(f"[reference_corpora] wrote excerpt {rel}")
    return rows


def build_medical(log=print, allow_download: bool = True) -> list:
    """Fetch PubMed abstracts via E-utilities and verify their rights.

    The efetch step batches the PMIDs (a single GET with 1,000 ids exceeds
    the URI length limit and returns HTTP 414), so abstracts are pulled in
    chunks of 200 and concatenated.

    The corpus is only usable once its rights are per-record facts, so this
    builder maintains ``medical/rights.tsv`` alongside it (#78): a fresh fetch
    re-verifies every record it fetched, a cached corpus keeps its ledger when
    the ledger still covers exactly its PMIDs, and a stale or missing ledger is
    re-verified in download mode or reported as a warning under
    ``--no-download`` -- never assumed open access.  See docs/medical-corpus.md.
    """
    dest = OUT / "medical"
    dest.mkdir(parents=True, exist_ok=True)
    dst = dest / "pubmed_abstracts.txt"
    ledger = OUT / RIGHTS_FILE
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = urllib.parse.quote(PUBMED_QUERY)
    esearch = (f"{base}/esearch.fcgi?db=pubmed&term={term}"
               f"&retmax={PUBMED_RETMAX}&retmode=json&sort=pub_date")
    fresh = False
    if corpus_ok(dst, MIN_TOKENS["medical"]):
        data = dst.read_bytes()
        log("[reference_corpora] medical corpus cached")
    else:
        log(f"[reference_corpora] querying PubMed: {PUBMED_QUERY} "
            f"(retmax={PUBMED_RETMAX})")
        meta = json.loads(download(esearch, dest / "_esearch.json", log))
        ids = meta["esearchresult"]["idlist"]
        chunks = [ids[i:i + 200] for i in range(0, len(ids), 200)]
        parts = []
        for n, chunk in enumerate(chunks):
            efetch = (f"{base}/efetch.fcgi?db=pubmed&id={','.join(chunk)}"
                      f"&rettype=abstract&retmode=text")
            log(f"[reference_corpora] efetch chunk {n + 1}/{len(chunks)} "
                f"({len(chunk)} ids)")
            parts.append(download(efetch, dest / f"_efetch_{n}.txt", log))
        data = b"\n".join(parts)
        for n in range(len(chunks)):
            (dest / f"_efetch_{n}.txt").unlink(missing_ok=True)
        (dest / "_esearch.json").unlink(missing_ok=True)
        # Raises (leaving any earlier corpus intact) if PubMed returned an
        # empty or truncated body.
        write_corpus(dst, data, MIN_TOKENS["medical"], log)
        fresh = True
    # Rights are a property of the *snapshot*, so the ledger is checked against
    # the PMIDs actually present, whatever mode produced them.
    pmids = parse_pmids(data)
    if not rights_ledger_covers(ledger, pmids):
        if fresh or allow_download:
            summary = write_rights(ledger, fetch_rights(pmids, dest, log))
            log(f"[reference_corpora] medical rights ledger: {len(pmids)} "
                f"records verified ({summary}) -> {ledger}")
        else:
            log(f"[reference_corpora] WARNING: {RIGHTS_FILE} is missing or "
                f"stale for the cached medical corpus and --no-download is "
                f"set; the corpus is kept, but its rights stay unverified per "
                f"record (re-run without --no-download to verify them)")
    else:
        log(f"[reference_corpora] medical rights ledger covers "
            f"{len(pmids)} records ({ledger})")
    return [live_row("medical", data)]


ARXIV_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


def parse_arxiv(data: bytes) -> bytes:
    """The ``<summary>`` bodies of a raw arXiv Atom response, joined.

    Atom escapes abstract text, so each body is run through ``html.unescape``
    before it becomes corpus text: left raw, ``&amp;`` / ``&rsquo;`` /
    ``&#39;`` would reach the tokenizer as fake word tokens (``amp``,
    ``rsquo``, ``quot``) and inflate the counts.  The decode happens *after*
    the summaries are extracted, so a decoded ``&lt;`` cannot re-enter the
    markup the regex searches.

    Returns b"" when *data* is not an Atom feed (an error page, an empty
    body).  Callers must gate on the token count before writing the result:
    re-running the parse over an already-clean corpus also yields b"", which
    is what made the build non-idempotent.
    """
    text = data.decode("utf-8", errors="replace")
    abstracts = [html.unescape(a.strip())
                 for a in ARXIV_SUMMARY_RE.findall(text)]
    return "\n\n".join(abstracts).encode("utf-8")


def build_cs(log=print) -> list:
    """Fetch arXiv abstracts (open access), categories cs.AI/cs.LG/cs.CL."""
    dest = OUT / "cs"
    dest.mkdir(parents=True, exist_ok=True)
    dst = dest / "arxiv_abstracts.txt"
    query = urllib.parse.quote(ARXIV_QUERY)
    url = (f"https://export.arxiv.org/api/query?search_query={query}"
           f"&start=0&max_results={ARXIV_MAX}&sortBy=submittedDate"
           f"&sortOrder=descending")
    if corpus_ok(dst, MIN_TOKENS["cs"]):
        data = dst.read_bytes()
        log("[reference_corpora] cs corpus cached")
    else:
        # Stage the raw Atom response beside the corpus rather than on top of
        # it: an interrupted earlier run used to leave raw XML where the
        # corpus belongs, and the next run re-parsed that into an empty file
        # and overwrote good data with nothing.
        raw = dest / "_arxiv_raw.xml"
        clean = b""
        for attempt in range(4):
            log(f"[reference_corpora] querying arXiv: {ARXIV_QUERY} "
                f"(max_results={ARXIV_MAX}) attempt {attempt + 1}/4")
            clean = parse_arxiv(download(url, raw, log))
            if corpus_tokens(clean) >= MIN_TOKENS["cs"]:
                break
            log(f"[reference_corpora] arXiv attempt {attempt + 1} yielded "
                f"{corpus_tokens(clean):,} tokens "
                f"(< {MIN_TOKENS['cs']:,}); retrying")
            clean = b""
            time.sleep(2 ** attempt)
        raw.unlink(missing_ok=True)
        # Raises, leaving any earlier corpus byte-for-byte intact, when all
        # four attempts failed.
        write_corpus(dst, clean, MIN_TOKENS["cs"], log)
        data = clean
    return [live_row("cs", data)]


def build_legal(log=print, allow_download: bool = True) -> list:
    """Download the public-domain **historical** legal / political register.

    These are 1215-1850 full texts, kept as the historical legal pole
    (#79); the modern legal-abstract register is built separately from DOAJ
    (``build_legal_abstracts``).  With ``allow_download=False`` the builder
    is cache-only: a text that is not already under ``legal/`` is skipped
    with a warning instead of being fetched, and the network is never
    touched.
    """
    dest = OUT / "legal"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, author, gid in LEGAL:
        dst = dest / f"{name}.txt"
        url = GUTENBERG_URL.format(id=gid)
        if dst.exists():
            data = dst.read_bytes()
        elif not allow_download:
            log(f"[reference_corpora] WARNING: legal/{name}.txt not cached "
                f"and --no-download set; skipping (re-run without "
                f"--no-download to fetch gutenberg {gid})")
            continue
        else:
            log(f"[reference_corpora] downloading {name} <- gutenberg {gid}")
            data = download(url, dst, log)
        text = strip_gutenberg(data.decode("utf-8", errors="replace"))
        clean = text.encode("utf-8")
        write_corpus(dst, clean, MIN_TOKENS_GUTENBERG, log)
        rows.append({
            "corpus": "legal", "file": f"legal/{name}.txt",
            "source": url, "license": "public domain", "author": author,
            "sha256": sha256_bytes(clean), "bytes": len(clean),
        })
    return rows


def fetch_doaj_pages(log=print) -> list:
    """The DOAJ article-search pages for ``DOAJ_QUERY``, recency-sorted.

    Staged beside the destination and unlinked by the caller, like the
    efetch chunks: an interrupted run must never leave raw JSON where a
    corpus belongs.
    """
    dest = OUT / "legal_abstracts"
    dest.mkdir(parents=True, exist_ok=True)
    pages = []
    n_pages = DOAJ_MAX // DOAJ_PAGE
    for page in range(1, n_pages + 1):
        url = (DOAJ_SEARCH.format(query=urllib.parse.quote(DOAJ_QUERY))
               + f"?pageSize={DOAJ_PAGE}&page={page}"
                 f"&sort=created_date:desc")
        log(f"[reference_corpora] querying DOAJ: {DOAJ_QUERY} "
            f"(page {page}/{n_pages})")
        pages.append(download(url, dest / f"_doaj_{page}.json", log))
    return pages


def build_legal_abstracts(log=print, allow_download: bool = True) -> list:
    """Fetch DOAJ law-journal abstracts; resolve their licences per journal.

    The query is recency-sorted, so the stable unit is the pinned snapshot:
    a rebuild that fetches different bytes is reported as drift by the
    orchestrator, not silently substituted (#80's mechanism).  DOAJ
    editorial stubs and non-English abstracts are skipped by the selection
    rule (``MIN_ABSTRACT_WORDS``), so they appear in neither file.

    The register is open access as a matter of DOAJ policy, but the licence
    is a per-journal fact, so this builder maintains
    ``legal_abstracts/rights.tsv`` alongside the corpus (#79): a fresh fetch
    resolves every journal it fetched; a cached corpus keeps its ledger when
    the ledger still covers exactly the corpus's DOIs, and a stale or
    missing ledger is re-resolved in download mode or reported as a warning
    under ``--no-download`` -- never assumed open.  See
    docs/legal-corpus.md.
    """
    dest = OUT / "legal_abstracts"
    dest.mkdir(parents=True, exist_ok=True)
    dst = dest / "legal_abstracts.txt"
    ledger = OUT / DOAJ_RIGHTS_FILE
    fresh_rows = None
    if corpus_ok(dst, MIN_TOKENS["legal_abstracts"]):
        data = dst.read_bytes()
        log("[reference_corpora] legal_abstracts corpus cached")
    elif not allow_download:
        log("[reference_corpora] WARNING: legal_abstracts/"
            "legal_abstracts.txt not cached and --no-download set; "
            "skipping (re-run without --no-download to fetch DOAJ)")
        return []
    else:
        pages = fetch_doaj_pages(log)
        for f in dest.glob("_doaj_*.json"):
            f.unlink(missing_ok=True)
        records = parse_doaj_records(pages)
        # ``fetch=download`` is passed through rather than left as the default
        # argument: the default is bound at definition time, so it would
        # bypass a test-level patch of the module's download helper.
        fresh_rows = resolve_doaj_licences(records, log, fetch=download)
        # Raises (leaving any earlier corpus intact) if the pages yielded too
        # little register text to be a corpus.
        write_corpus(dst, doaj_corpus_bytes(records),
                     MIN_TOKENS["legal_abstracts"], log)
        data = dst.read_bytes()
    # Rights are a property of the *snapshot*, so the ledger is checked
    # against the records actually present, whatever mode produced them.
    # The corpus is record-per-abstract while the ledger is keyed by DOI, so
    # the counts reported below are distinct DOIs.
    records = parse_records_from_corpus(data)
    dois = list(dict.fromkeys(r["doi"] for r in records))
    if fresh_rows is not None:
        summary = write_doaj_rights(ledger, fresh_rows)
        log(f"[reference_corpora] legal_abstracts rights ledger: "
            f"{len(dois)} records verified ({summary}) -> {ledger}")
    elif not doaj_rights_ledger_covers(ledger, dois):
        if allow_download:
            summary = write_doaj_rights(
                ledger, resolve_doaj_licences(records, log, fetch=download))
            log(f"[reference_corpora] legal_abstracts rights ledger: "
                f"{len(dois)} records verified ({summary}) -> {ledger}")
        else:
            log(f"[reference_corpora] WARNING: {DOAJ_RIGHTS_FILE} is missing "
                f"or stale for the cached legal_abstracts corpus and "
                f"--no-download is set; the corpus is kept, but its licences "
                f"stay unverified per record (re-run without --no-download "
                f"to verify them)")
    else:
        log(f"[reference_corpora] legal_abstracts rights ledger covers "
            f"{len(dois)} records ({ledger})")
    return [live_row("legal_abstracts", data)]


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
MANIFEST_HEADER = ["corpus", "file", "source", "license", "author",
                   "sha256", "bytes", "tokens", "types", "hash_role"]
DRIFT_FILE = "manifest_drift.tsv"

# Files whose sha256 is an *enforced pin* (checked against PINNED_GENERAL);
# every other manifest row carries an *observed checksum*.  See the module
# docstring: the two roles are enforced differently, and the manifest says
# which is which per row.  Derived from PINNED_GENERAL so the two can never
# drift apart.
def pinned_files():
    return {f"general_english/{name}.txt" for name, *_ in PINNED_GENERAL}


def read_manifest(path: Path) -> dict:
    """Prior manifest rows keyed by ``file``; {} when absent or unreadable."""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row.get("file") and row.get("sha256"):
                out[row["file"]] = row
    return out


def manifest_drift(rows, prior: dict, log=print) -> list:
    """Drift between *rows* and the expected pins recorded in *prior*.

    The previously published manifest's sha256 for a file is that file's
    expected pin.  A rebuild that yields different bytes for it is drift:
    every instance is returned *and* logged here, before the manifest is
    rewritten, so a changed corpus can never replace its row silently.
    """
    drift = []
    for row in rows:
        exp = prior.get(row["file"])
        if exp is None or exp["sha256"] == row["sha256"]:
            continue
        drift.append({
            "corpus": row["corpus"], "file": row["file"],
            "expected_sha256": exp["sha256"],
            "observed_sha256": row["sha256"],
            "expected_bytes": exp.get("bytes", ""),
            "observed_bytes": row["bytes"],
        })
    for d in drift:
        log(f"[reference_corpora] DRIFT: {d['file']} "
            f"{d['expected_sha256'][:12]}... -> {d['observed_sha256'][:12]}... "
            f"({d['expected_bytes']} -> {d['observed_bytes']} bytes): the "
            f"expected pin no longer matches the built corpus")
    return drift


def write_drift_report(manifest_path: Path, drift, log=print):
    """Persist *drift* beside the manifest; clear a stale report when clean."""
    report = manifest_path.parent / DRIFT_FILE
    if drift:
        write_tsv(report,
                  ["corpus", "file", "expected_sha256", "observed_sha256",
                   "expected_bytes", "observed_bytes"], drift)
        log(f"[reference_corpora] {len(drift)} drifted file(s) reported in "
            f"{report} (the expected pins are preserved there)")
    else:
        report.unlink(missing_ok=True)


def ensure_corpora(allow_download: bool = True, log=print) -> list:
    """Assemble every corpus; returns the manifest rows.

    ``allow_download=False`` is cache-only for every source: nothing is
    fetched, and anything not already cached is reported and skipped.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += build_general_english(log, allow_download)
    rows += build_excerpts(log)
    rows += build_legal(log, allow_download)
    if allow_download:
        rows += build_medical(log, allow_download)
        rows += build_cs(log)
        rows += build_legal_abstracts(log, allow_download)
    else:
        for corpus, rel in LIVE_FILES.items():
            p = OUT / rel
            if corpus_ok(p, MIN_TOKENS[corpus]):
                rows.append(live_row(corpus, p.read_bytes()))
                if corpus == "legal_abstracts":
                    # Cache-only still checks the rights position: the DOI
                    # set comes from the corpus file itself, so the check
                    # needs no network, and a stale ledger is reported
                    # rather than silently assumed current.
                    dois = parse_dois(p.read_bytes())
                    if not doaj_rights_ledger_covers(
                            OUT / DOAJ_RIGHTS_FILE, dois):
                        log(f"[reference_corpora] WARNING: "
                            f"{DOAJ_RIGHTS_FILE} is missing or stale for the "
                            f"cached legal_abstracts corpus and "
                            f"--no-download is set; the corpus is kept, but "
                            f"its licences stay unverified per record "
                            f"(re-run without --no-download to verify them)")
            else:
                # A cached file that fails the gate is worse than no file: it
                # would be counted as reference text.  Report and skip.
                log(f"[reference_corpora] WARNING: {rel} missing or below the "
                    f"{MIN_TOKENS[corpus]:,}-token gate and --no-download set; "
                    f"skipping (re-run without --no-download to refetch)")
    # token counts (English word regex, same as #24)
    import frequency_zipf as fz
    for row in rows:
        p = OUT / row["file"]
        text = p.read_text(encoding="utf-8", errors="replace")
        toks = fz.EN_WORD_RE.findall(text)
        row["tokens"] = len(toks)
        row["types"] = len({t.lower() for t in toks})
        row["hash_role"] = ("enforced_pin" if row["file"] in pinned_files()
                            else "observed_checksum")
    # Pin semantics, enforced: the prior manifest's checksums are the expected
    # pins for every file it already recorded.  Drift is reported (logged and
    # written beside the manifest) *before* the manifest is rewritten, and a
    # file that vanishes from the build is reported too -- neither is silent.
    manifest_path = OUT / "manifest.tsv"
    prior = read_manifest(manifest_path)
    missing = sorted(set(prior) - {r["file"] for r in rows})
    for rel in missing:
        log(f"[reference_corpora] WARNING: {rel} was in the previous manifest "
            f"but is not in this build; its row is dropped from the manifest "
            f"(expected pin preserved in the prior manifest / drift report)")
    drift = manifest_drift(rows, prior, log)
    write_drift_report(manifest_path, drift, log)
    write_tsv(manifest_path, MANIFEST_HEADER, rows)
    log(f"[reference_corpora] manifest: {len(rows)} files -> {manifest_path}")
    return rows


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-download", action="store_true",
                    help="cache-only build: never touch the network; skips the "
                         "live PubMed/arXiv queries and any general-English or "
                         "legal Gutenberg text that is not already cached")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    OUT = Path(args.out)
    ensure_corpora(allow_download=not args.no_download)
    return 0


if __name__ == "__main__":
    sys.exit(main())
