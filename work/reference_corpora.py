#!/usr/bin/env python3
"""#56 Reference corpora for keyness -> analysis/reference_corpora/

Assembles the six reference corpora that the keyness harness (``work/keyness.py``)
compares *Blood Electric* against:

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
  medical           PubMed abstracts (open access).  A live E-utilities query;
                    the manifest records the observed checksum, and a rebuild
                    that changes it is reported as drift.
  cs                arXiv abstracts (open access), categories cs.AI / cs.LG /
                    cs.CL.  Live API query; same observed-checksum +
                    drift-reporting treatment as medical.
  legal             public-domain legal / political texts (Gutenberg): the
                    Federalist Papers, the Declaration of Independence, Magna
                    Carta, Bastiat's *The Law*, Rousseau's *The Social Contract*.

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
arXiv queries are skipped, and so is any general-English or legal Gutenberg
text that is not already cached: those are skipped with a warning rather than
fetched, so the flag can never be surprised by a network call.  (The eight
pinned baselines are copied from #24's local cache, a file copy rather than a
network fetch; a missing #24 cache is a hard error telling you to run #24.)

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

# Public-domain legal / political register.
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
    "cs": 50_000, "medical": 100_000,
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
        "open access (PubMed abstracts)"),
    "cs": (f"arxiv.org api query={ARXIV_QUERY} max_results={ARXIV_MAX}",
           "open access (arXiv abstracts)"),
}
LIVE_FILES = {
    "medical": "medical/pubmed_abstracts.txt",
    "cs": "cs/arxiv_abstracts.txt",
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


def build_medical(log=print) -> list:
    """Fetch PubMed abstracts via E-utilities (open access).

    The efetch step batches the PMIDs (a single GET with 1,000 ids exceeds
    the URI length limit and returns HTTP 414), so abstracts are pulled in
    chunks of 200 and concatenated.
    """
    dest = OUT / "medical"
    dest.mkdir(parents=True, exist_ok=True)
    dst = dest / "pubmed_abstracts.txt"
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = urllib.parse.quote(PUBMED_QUERY)
    esearch = (f"{base}/esearch.fcgi?db=pubmed&term={term}"
               f"&retmax={PUBMED_RETMAX}&retmode=json&sort=pub_date")
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
    """Download the public-domain legal / political register texts.

    With ``allow_download=False`` the builder is cache-only: a text that is
    not already under ``legal/`` is skipped with a warning instead of being
    fetched, and the network is never touched.
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
        rows += build_medical(log)
        rows += build_cs(log)
    else:
        for corpus, rel in LIVE_FILES.items():
            p = OUT / rel
            if corpus_ok(p, MIN_TOKENS[corpus]):
                rows.append(live_row(corpus, p.read_bytes()))
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
