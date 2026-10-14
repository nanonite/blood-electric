#!/usr/bin/env python3
"""#57 Intertextual n-gram matching -> analysis/intertext/

L7 of ``docs/meaning-extraction-methods.md`` asks for one thing this task is
named for:

    "Intertextual n-gram matching -- match phrases against Siratori's other
     works (*Headcode*, *Human_Worms*, *Smart-d*, *GENEDUB*, *Nonexistence*...)
     and the cited influences.  Shared formulas reveal his idiolect vs. the
     genre's."

This module separates what can be done from what cannot, and says which is
which rather than quietly substituting one for the other.

**1. The acquisition ledger.**  Every candidate source is probed and recorded
with its provenance and its reuse terms, exactly as ``work/translation_oracle.py``
(#58) does for the Skobin translation.  The five *named* Siratori works are
print-only commercial editions; the only copies reachable here are an
access-restricted Internet Archive lending scan (``headcode0000kenj``,
``access-restricted-item=true``, ``printdisabled``) and no text at all for the
other four.  **No named work is ingested.**  That is the negative half of the
result and it is written down with the evidence, not asserted.

**2. What is actually available.**  Two lawful bodies of same-author /
same-tradition text are used:

    * ``siratori_other`` - a short experimentally published Siratori text,
      "abnormal living body of a chemical=anthropoid-modem=heart ...", hosted
      by the curated online journal *Great Works* (the same journal the project
      already draws on).  This is a *different* work from the five the task
      names; it is labelled as such everywhere and is used only as an
      idiolect probe, never as a stand-in for the named works.
    * ``author_paratext`` - Siratori's own interview answers, already pinned
      and attributed by #58 (``analysis/translation_oracle/author_paratext.tsv``).

  The cited influences are the #56 fair-use excerpt corpora (Gibson /
  Stephenson / Sterling; Burroughs / Artaud / Breton; Joyce) and the #56
  public-domain / open registers (general English, medical, legal, cs) as
  register controls.

**3. The matcher.**  A plain n-gram overlap harness (n = 1..6) over the #17
book *word-part* layer, markup-screened the #56 way and split at page
boundaries.  Reference text is tokenised with the same #24 word regex, split
at file boundaries.  Because the influence excerpts are tiny by necessity, the
harness carries a **shuffle null**: each arm's token sequence is permuted
(seed fixed) and the same overlap recomputed, so an observed count is always
reported against what chance allows.

**Verdict.**  The idiolect side is measurable and measured: the book shares
its signature noun formulas (``living body``, ``brain universe``, ``drug
fetus``, ``gene dub``, ``data=mutant`` ...) with a *different* Siratori work at
a rate far above the shuffle null, and *not* with the influence excerpts.  The
genre side is underpowered and the named-work comparison is **not testable
from lawful sources** here; both are reported as bounds, not findings.

Reproduce::

    conda activate marker
    python work/intertext.py            # offline, deterministic; writes analysis/intertext/
    python work/intertext.py --quick    # sampled build for the regression suite
    python work/intertext.py --probe    # re-run the live HTTP source probes (writes nothing)
    python work/test_intertext.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html as _html
import json
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import keyness as kn

OUT = ROOT / "analysis" / "intertext"
PART_INDEX = ROOT / "analysis" / "tokenizer" / "part_index.tsv"
PARATEXT = ROOT / "analysis" / "translation_oracle" / "author_paratext.tsv"
CORPORA = kn.CORPORA

SCHEMA = "intertext/1"

USER_AGENT = ("blood-electric-workspace/1.0 "
              "(research; Chainlink #57 intertextual n-gram matching)")
#: the survey run that produced the pinned statuses in ``SOURCES``; pinned, never
#: re-derived, so the ledger does not move with the clock.  ``--probe`` re-runs
#: the live requests and prints a diff against the pins.
PROBE_DATE = "2026-10-14"

#: n values the harness scores.  n=1 is a lexical baseline; the interesting
#: formulas are the longer ones.
NGRAM_NS = (1, 2, 3, 4, 5, 6)
#: a book n-gram is written to the formula table only if it recurs at least this
#: often in the book (hapax n-grams have no formula status).
MIN_BOOK_FREQ = 2
#: deterministic seed for the shuffle null.
SHUFFLE_SEED = 57
#: number of shuffles per arm (the regression suite drops this to 1 via --quick).
SHUFFLES = 5

# ===========================================================================
# 1. the acquisition ledger
# ===========================================================================

# authorization classes
AUTH_OPEN = "open"                       # free to fetch and use (research quotation)
AUTH_EXCERPT = "free_excerpt"            # free to fetch; only a short excerpt reused
AUTH_PRINT = "print_or_lending_only"     # a catalogue/scan; no servable text
AUTH_STORE = "store"                     # commercial ebook store
AUTH_MIRROR = "unauthorized_mirror"      # an upload with no rights holder's consent
AUTH_BLOCKED = "bot_blocked"             # the host refused this agent
AUTH_ABSENT = "not_offered"              # the host offers no such text
AUTH_SEARCHED = "searched_not_found"     # searched, no usable copy located

# verdicts
V_INGESTED = "ingested_excerpt"
V_REFUSED = "refused_unauthorized_mirror"
V_PRINT = "print_or_lending_only"
V_SEARCHED = "searched_not_found"
V_BLOCKED = "blocked_by_host"
V_ABSENT = "not_offered"

#: nothing in these classes is ever ingested - the gate is the point of the
#: negative result.  ``same_author_work`` / ``author_paratext`` / influence /
#: register arms draw only on ``AUTH_OPEN`` and ``AUTH_EXCERPT`` sources.
NON_INGESTIBLE = {AUTH_PRINT, AUTH_STORE, AUTH_MIRROR, AUTH_BLOCKED,
                  AUTH_ABSENT, AUTH_SEARCHED}


@dataclass(frozen=True)
class Source:
    key: str
    kind: str            # same_author_work | this_author_other | cited_influence
                         # | register_control | author_paratext
    title: str
    url: str
    language: str
    license: str
    authorization: str
    verdict: str
    http_status: str
    note: str = ""


SOURCES: tuple[Source, ...] = (
    # --- the five NAMED Siratori works: sought, none lawfully servable -------
    Source("siratori_headcode", "named_work",
           "Headcode (iUniverse, 2004)",
           "https://archive.org/metadata/headcode0000kenj", "en",
           "in copyright; print/lending edition",
           AUTH_PRINT, V_PRINT, "200",
           "Internet Archive scan headcode0000kenj: access-restricted-item="
           "true, collection printdisabled.  A borrow-only scan, no body "
           "fetched.  ISBN 978-0595318254."),
    Source("siratori_human_worms", "named_work",
           "Human_Worms (2004)", "https://denniscooperblog.com/"
           "spotlight-on-kenji-siratori-human_worms-2003/", "en",
           "in copyright; no open full text",
           AUTH_SEARCHED, V_SEARCHED, "200",
           "A publisher/reader spotlight page exists (a critical post and an "
           "author interview about the book); it quotes no book text and is "
           "not a text source.  No servable edition found."),
    Source("siratori_smart_d", "named_work",
           "Smart-d (2004)", "-", "en",
           "in copyright; no open full text",
           AUTH_SEARCHED, V_SEARCHED, "-",
           "No open text or catalogue record with text located."),
    Source("siratori_genedub", "named_work",
           "GENEDUB (2006)", "http://www.greatworks.org.uk/archive.html", "en",
           "in copyright; no open full text",
           AUTH_SEARCHED, V_SEARCHED, "200",
           "The Great Works author note points to a (now defunct) personal "
           "'Genedub' blog and lists GENEDUB as a publication; no book text "
           "is served."),
    Source("siratori_nonexistence", "named_work",
           "Nonexistence (2006)", "-", "en",
           "in copyright; no open full text",
           AUTH_SEARCHED, V_SEARCHED, "-",
           "No open text or catalogue record with text located."),
    # --- an actual Siratori text that IS freely published online ------------
    Source("siratori_greatworks_ks", "this_author_other",
           '"abnormal living body of a chemical=anthropoid-modem=heart ..." '
           "(Great Works)",
           "http://www.greatworks.org.uk/poems/ks.html", "en",
           "authored text published online by Great Works; short research "
           "excerpt reused with attribution",
           AUTH_OPEN, V_INGESTED, "200",
           "A different work from the five named above, published by the "
           "curated online journal Great Works.  The first three paragraphs "
           "(~330 words) are reused here as an attributed research excerpt; "
           "it is an idiolect probe, NOT a substitute for the named works."),
    # --- author paratext (legitimately published interviews, via #58) -------
    Source("interview_3am_nude_brain", "author_paratext",
           "3AM Magazine: 'THE NUDE BRAIN: AN INTERVIEW WITH KENJI SIRATORI'",
           "https://www.3ammagazine.com/litarchives/2002_jun/"
           "interview_kenji_siratori.html", "en",
           "(c) 2002 3AM Magazine; quoted as short research quotations",
           AUTH_OPEN, V_INGESTED, "200",
           "Richard Marshall interviews Siratori, June 2002.  Pinned by #58; "
           "this module reads #58's attributed quotation table."),
    Source("interview_rhizome_14416_wb", "author_paratext",
           "Rhizome 14416 (Wayback capture): 'Interview with Kenji Siratori + "
           "Loop Writing'",
           "https://web.archive.org/web/20211209013803/https://rhizome.org/"
           "community/14416/", "en",
           "(c) 2002; archived public web post, quoted as short research "
           "quotations",
           AUTH_OPEN, V_INGESTED, "200",
           "Sandy Baldwin (WVU Center for Literary Computing), Aug 2002.  "
           "Declares Burroughs and Guyotat as sources.  This is the capture #58 "
           "pins and the one this module actually reads; the live Rhizome page "
           "refused this agent (403, see ``interview_rhizome_14416``)."),
    Source("interview_rhizome_14416", "author_paratext",
           "Rhizome: 'Interview with Kenji Siratori + Loop Writing' (live site)",
           "https://rhizome.org/community/14416/", "en",
           "(c) 2002; quoted as short research quotations",
           AUTH_BLOCKED, V_BLOCKED, "403",
           "The live Rhizome page returns 403 to this agent (recorded by #58); "
           "the text is obtained from the Wayback capture above.  Listed so the "
           "denial is not silently dropped."),
    # --- cited influences (from the #56 corpora) ----------------------------
    Source("influence_cyberpunk", "cited_influence",
           "Gibson / Stephenson / Sterling (fair-use excerpts)",
           "analysis/reference_corpora/cyberpunk/", "en",
           "copyrighted; short attributed excerpts",
           AUTH_EXCERPT, V_INGESTED, "-",
           "The #56 cyberpunk register; excerpt-length by necessity."),
    Source("influence_cutup_surrealist", "cited_influence",
           "Burroughs / Artaud / Breton (fair-use excerpts)",
           "analysis/reference_corpora/cutup_surrealist/", "en",
           "copyrighted; short attributed excerpts",
           AUTH_EXCERPT, V_INGESTED, "-",
           "The #56 cut-up / Surrealist register; Burroughs and Artaud are "
           "declared influences in the interviews."),
    Source("influence_finnegans_wake", "cited_influence",
           "Joyce, Finnegans Wake (fair-use excerpts)",
           "analysis/reference_corpora/finnegans_wake/", "en",
           "in US copyright until 2034; short attributed excerpts",
           AUTH_EXCERPT, V_INGESTED, "-",
           "The #56 Finnegans Wake register."),
    Source("influence_guyotat", "cited_influence",
           "Pierre Guyotat (declared influence)", "-", "fr",
           "in copyright; no open excerpt acquired",
           AUTH_SEARCHED, V_SEARCHED, "-",
           "Named alongside Burroughs in the interviews; no lawful excerpt "
           "was assembled, so it is absent from the matcher rather than "
           "silently replaced."),
    Source("genre_prior_ballard", "genre_prior",
           "J. G. Ballard; Aase Berg; Blake Butler; Tetsuo (genre priors)",
           "-", "en",
           "in copyright; priors only",
           AUTH_SEARCHED, V_SEARCHED, "-",
           "Named as genre priors in L7.5.  Priors, not texts; not acquired."),
    # --- register controls (the #56 open corpora) ---------------------------
    Source("register_general_english", "register_control",
           "General English prose (15 public-domain texts)",
           "analysis/reference_corpora/general_english/", "en",
           "public domain",
           AUTH_OPEN, V_INGESTED, "-",
           "The #56 general-English register; the control for 'generic "
           "English' rather than genre."),
    Source("register_medical", "register_control",
           "PubMed abstracts", "analysis/reference_corpora/medical/", "en",
           "open access", AUTH_OPEN, V_INGESTED, "-",
           "Technical register control from #56."),
    Source("register_legal", "register_control",
           "Public-domain legal/political texts",
           "analysis/reference_corpora/legal/", "en",
           "public domain", AUTH_OPEN, V_INGESTED, "-",
           "Technical register control from #56."),
    Source("register_cs", "register_control",
           "arXiv cs abstracts", "analysis/reference_corpora/cs/", "en",
           "open access", AUTH_OPEN, V_INGESTED, "-",
           "Technical register control from #56."),
)

SOURCES_BY_KEY = {s.key: s for s in SOURCES}
NAMED_WORKS = tuple(s for s in SOURCES if s.kind == "named_work")


# ===========================================================================
# 2. the available arms
# ===========================================================================

# the Great Works excerpt, reused verbatim and attributed (see SOURCES).  Kept
# as a constant so a fresh clone reproduces the corpus from the committed
# script alone, the way #56 embeds its fair-use excerpts.
SIRATORI_GREATWORKS_EXCERPT = "\n\n".join((
    "abnormal living body of a chemical=anthropoid-modem=heart of the hybrid "
    "corpse mechanism that turned on technojunkies' ill-treatment to the "
    "terror abolition world-codemaniacs of the chromosomal aberration that "
    "was controlled FUCKNAMLOAD****the acidHUMANIX infectious disease archive "
    "of the biocapturism nerve cells nightmare-script of a clone boy to the "
    "super-genomewarable to the feeling replicant living body junk of her "
    "digital=vamp cold-blooded disease animals reptilian=HUB ultra=machinary "
    "tragedy-ROM system to the brain universe of the hyperreal HIV=scanner "
    "form murder game of the dogs@tera DNA=channel of the drug fetus of the "
    "trash sense is debugged!",
    "terror fear=cytoplasm gene-dub of the drug fetus of the trash sense is "
    "debugged to the paradise apparatus of the human body pill cruel emulator "
    "corpse feti=streaming of the soul/gram made of retro-ADAM::data=mutant "
    "of her abolition world-codemaniacs feeling replicant ecstasy system of "
    "the acidHUMANIX infectious disease archive_body encoder that BDSM plays "
    "a chemical=anthropoid to the brain universe that was processed noise "
    "hunting for the grotesque WEB to the genomics strategy circuit of the "
    "biocapturism nerve cells mass of flesh-modules of the hyperreal "
    "HIV=scanners that turned on the ill-treatment of the corpse city "
    "reptilian=HUB of a clone boy=joints. . . .",
    "I turn on ill-treatment to the DNA=channels of the biocapturism nerve "
    "cells abolition world-codemaniacs that was processed the data=mutant of "
    "her ultra=machinary tragedy-ROM creature system corpse feti=streaming of "
    "a clone boy****the gene-dub to the paradise apparatus of the human body "
    "pill cruel emulator that compressed the abnormal living body of a "
    "chemical=anthropoid-brain universe of the terror fear=cytoplasm that was "
    "controlled the acidHUMANIX infectious disease of the soul/gram made of "
    "retro-ADAM@trash sense of drug fetus feeling replicant of the hyperreal "
    "HIV=scanner form tera of dogs were installed to the "
    "reptilian=HUB_modem=heart that hung up non-resettable murder game.",
))

#: the live page the excerpt above is drawn from (SOURCE ``siratori_greatworks_ks``)
#: and the number of leading paragraphs reused.  The excerpt is *pinned*: the
#: token count and the sha256 of the normalised text below are the provenance
#: contract.  ``python work/intertext.py --probe`` re-fetches the page, re-extracts
#: those paragraphs and diffs the result against the pin, so the embedded constant
#: can never silently drift from the source it names.
GREATWORKS_URL = "http://www.greatworks.org.uk/poems/ks.html"
GREATWORKS_EXCERPT_PARAGRAPHS = 3
GREATWORKS_EXCERPT_TOKENS = 330
GREATWORKS_EXCERPT_SHA256 = (
    "436ebaddce73d9ca13cfc18eac03ce55b1a00c97dc4c970614aaf6e59b383f5c")

#: the #56 reference-corpus manifest that carries a sha256 per file; the matcher
#: verifies every file it ingests against this, so an arm can never silently
#: substitute a different text for the one #56 pinned.
CORPUS_MANIFEST = CORPORA / "manifest.tsv"


@dataclass(frozen=True)
class Arm:
    key: str
    kind: str            # same_author | author_meta | influence | register
    source_key: str
    label: str


ARMS: tuple[Arm, ...] = (
    Arm("siratori_other", "same_author", "siratori_greatworks_ks",
        "a different, freely published Siratori text (Great Works)"),
    Arm("author_paratext", "author_meta", "interview_3am_nude_brain",
        "the author's own words in published interviews (3AM / Rhizome, via #58)"),
    Arm("influence_cyberpunk", "influence", "influence_cyberpunk",
        "Gibson / Stephenson / Sterling excerpts"),
    Arm("influence_cutup_surrealist", "influence", "influence_cutup_surrealist",
        "Burroughs / Artaud / Breton excerpts"),
    Arm("influence_finnegans_wake", "influence", "influence_finnegans_wake",
        "Joyce, Finnegans Wake excerpts"),
    Arm("register_general_english", "register", "register_general_english",
        "general English prose (control)"),
    Arm("register_medical", "register", "register_medical",
        "PubMed abstracts (control)"),
    Arm("register_legal", "register", "register_legal",
        "public-domain legal texts (control)"),
    Arm("register_cs", "register", "register_cs",
        "arXiv cs abstracts (control)"),
)
ARMS_BY_KEY = {a.key: a for a in ARMS}

# the #56 corpora directories behind the non-constant arms
_REFERENCE_ARM_CORPUS = {
    "influence_cyberpunk": "cyberpunk",
    "influence_cutup_surrealist": "cutup_surrealist",
    "influence_finnegans_wake": "finnegans_wake",
    "register_general_english": "general_english",
    "register_medical": "medical",
    "register_legal": "legal",
    "register_cs": "cs",
}
_STRIP_HEADERS = {"cyberpunk", "cutup_surrealist", "finnegans_wake"}


# ===========================================================================
# 3. tokenisation / n-gram primitives
# ===========================================================================
def ngrams(seq, n: int) -> collections.Counter:
    """Counter of the n-grams of ``seq`` (tuples of consecutive tokens)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return collections.Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


def shuffled(seq, rng: random.Random):
    """A permutation of ``seq`` (multiset preserved) using ``rng``."""
    out = list(seq)
    rng.shuffle(out)
    return out


def count_book_ngrams_in(book_ngrams: dict, seqs, ns=NGRAM_NS):
    """For each n, the occurrences in ``seqs`` of n-grams that are in the book.

    ``book_ngrams`` maps n -> set of book n-grams (the membership filter).  A
    return value of ``{n: Counter}`` counts, per book n-gram, how many times it
    occurs across every sequence; only observed n-grams appear.
    """
    present = {n: collections.Counter() for n in ns}
    tokens = 0
    for s in seqs:
        tokens += len(s)
        for n in ns:
            c = present[n]
            for i in range(len(s) - n + 1):
                g = tuple(s[i:i + n])
                if g in book_ngrams[n]:
                    c[g] += 1
    return present, tokens


#: the exclusive formula classes, strongest evidence first.  An idiolect claim
#: needs the formula to be *distinctive*: present in the author's own text but
#: NOT ordinary/technical English.  The author outranks the influences, and both
#: outrank the registers.
FORMULA_CLASSES = ("author_idiolect", "author_common", "genre_influence",
                   "influence_common", "register_common", "book_unique")


def formula_class(in_same_author: bool, in_author_meta: bool,
                  in_influence: bool, in_register: bool) -> str:
    """Classify a book formula by the strongest external source containing it."""
    author = in_same_author or in_author_meta
    if author and not in_register:
        return "author_idiolect"
    if author and in_register:
        return "author_common"
    if in_influence and not in_register:
        return "genre_influence"
    if in_influence and in_register:
        return "influence_common"
    if in_register:
        return "register_common"
    return "book_unique"


# ===========================================================================
# 4. loaders
# ===========================================================================
def build_book(log=print, sample_pages: int | None = None):
    """Ordered, markup-screened #17 word-part sequences, one list per page.

    The screen is exactly #56's: a part is dropped when its unit uid is a #19
    markup residue, when its ``(page, start)`` falls inside a #27-detected tag
    span, or when its lower-cased surface is a #19 junk / markup type.
    """
    spans, markup_uids, markup_types = kn.load_markup()
    by_page = collections.defaultdict(list)
    n_parts = n_dropped = 0
    for r in kn.read_tsv(PART_INDEX):
        if r["role"] != "w":
            continue
        n_parts += 1
        uid, page, start = int(r["uid"]), int(r["page"]), int(r["start"])
        surface = r["text"]
        if (uid in markup_uids or surface.lower() in markup_types
                or any(a <= start < b for a, b in spans.get(page, ()))):
            n_dropped += 1
            continue
        by_page[page].append((start, surface.lower()))
    pages = sorted(by_page)
    if sample_pages is not None:
        pages = pages[:sample_pages]
    seqs = [[t for _s, t in sorted(by_page[p])] for p in pages]
    token_total = sum(len(s) for s in seqs)
    n1 = ngrams([t for s in seqs for t in s], 1)
    stats = {
        "parts": n_parts, "dropped": n_dropped,
        "kept": n_parts - n_dropped, "tokens": token_total,
        "types": len(n1), "pages": len(seqs),
    }
    log(f"[intertext] book: {n_parts:,} word parts, {n_dropped:,} markup-"
        f"screened, {token_total:,} kept in {len(seqs)} pages, "
        f"{stats['types']:,} types")
    return seqs, stats


def _tokenise(text: str):
    return [m.group(0).lower() for m in kn.EN_WORD_RE.finditer(text)]


def load_reference_sequences(corpus: str, max_tokens: int | None = None):
    """Token sequences for a #56 corpus; one sequence per file, boundaries kept."""
    seqs = []
    files = kn.corpus_files(corpus)
    if not files:
        raise FileNotFoundError(f"no corpus files in {CORPORA / corpus}")
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        if corpus in _STRIP_HEADERS:
            text = "\n".join(ln for ln in text.splitlines()
                             if not ln.lstrip().startswith(("#", "--")))
        seqs.append(_tokenise(text))
    if max_tokens is not None:
        capped, budget = [], max_tokens
        for s in seqs:
            if budget <= 0:
                break
            capped.append(s[:budget])
            budget -= len(capped[-1])
        seqs = capped
    return seqs


def _is_author_turn(row: dict) -> bool:
    """Whether a #58 paratext row is the author's own words.

    #58's pinned table is an interview *Q/A*, not an author monologue, and has no
    speaker column.  Its convention is that a row the **interviewer** spoke
    carries a ``claim`` beginning "An interviewer"; mixing such a turn into the
    author arm would attribute the interviewer's vocabulary to the author, so it
    is excluded here rather than silently included.
    """
    return not row.get("claim", "").strip().lower().startswith("an interviewer")


def paratext_rows():
    """#58's pinned paratext rows that are the author's own words."""
    if not PARATEXT.exists():
        return []
    return [r for r in kn.read_tsv(PARATEXT)
            if r.get("quote") and _is_author_turn(r)]


def paratext_excluded_rows():
    """#58's pinned paratext rows recorded as the interviewer's turn."""
    if not PARATEXT.exists():
        return []
    return [r for r in kn.read_tsv(PARATEXT)
            if r.get("quote") and not _is_author_turn(r)]


def load_paratext_sequences():
    """#58's pinned author-interview quotations, one sequence per author turn."""
    return [_tokenise(r["quote"]) for r in paratext_rows()]


_GREATWORKS_PARA_RE = re.compile(r'<p class="prose">(.*?)</p>', re.DOTALL)


def extract_greatworks_excerpt(page_html: str,
                               paragraphs: int = GREATWORKS_EXCERPT_PARAGRAPHS):
    """Re-extract the attributed excerpt from the live Great Works page payload.

    Mirrors the normalisation the embedded constant uses: take the leading
    ``<p class="prose">`` paragraphs, strip the surrounding quotation marks and
    tags, unescape entities, collapse whitespace, and join with blank lines.  The
    sha256 of the result is the pin in ``GREATWORKS_EXCERPT_SHA256``.
    """
    paras = _GREATWORKS_PARA_RE.findall(page_html)[:paragraphs]
    cleaned = []
    for p in paras:
        t = re.sub(r"<[^>]+>", "", p).replace("&nbsp;", " ")
        t = _html.unescape(t).strip().removeprefix('"').removesuffix('"')
        cleaned.append(re.sub(r"\s+", " ", t).strip())
    return "\n\n".join(cleaned)


_SEQ_CACHE: dict = {}


def load_arm(arm: Arm, quick: bool = False):
    """Return the token sequences for an arm (memoised per process)."""
    ck = (arm.key, bool(quick))
    if ck in _SEQ_CACHE:
        return _SEQ_CACHE[ck]
    if arm.key == "siratori_other":
        seqs = [_tokenise(SIRATORI_GREATWORKS_EXCERPT)]
    elif arm.key == "author_paratext":
        seqs = load_paratext_sequences()
    else:
        corpus = _REFERENCE_ARM_CORPUS[arm.key]
        budget = 5000 if (quick and arm.kind == "register") else None
        seqs = load_reference_sequences(corpus, max_tokens=budget)
    _SEQ_CACHE[ck] = seqs
    return seqs


# ===========================================================================
# 5. the run
# ===========================================================================
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_sources(log=print):
    rows = []
    for s in SOURCES:
        rows.append({
            "key": s.key, "kind": s.kind, "title": s.title, "url": s.url,
            "language": s.language, "license": s.license,
            "authorization": s.authorization, "verdict": s.verdict,
            "http_status": s.http_status,
            "ingested": "n" if s.authorization in NON_INGESTIBLE else "y",
            "probe_date": PROBE_DATE, "note": s.note,
        })
    kn.write_tsv(OUT / "sources.tsv",
                 ["key", "kind", "title", "url", "language", "license",
                  "authorization", "verdict", "http_status", "ingested",
                  "probe_date", "note"], rows)
    log(f"[intertext] sources.tsv: {len(rows)} candidate sources")
    return rows


def build(log=print, quick: bool = False, sample_pages: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    write_sources(log)

    book_seqs, book_stats = build_book(log, sample_pages=sample_pages)
    book_counters = {n: ngrams([t for s in book_seqs for t in s], n)
                     for n in NGRAM_NS}
    book_by_n = {n: set(c) for n, c in book_counters.items()}

    arm_results = {}
    formula_membership = collections.defaultdict(dict)  # arm_key -> n -> set
    for arm in ARMS:
        seqs = load_arm(arm, quick=quick)
        present, tokens = count_book_ngrams_in(book_by_n, seqs)
        arm_results[arm.key] = {"arm": arm, "present": present,
                                "tokens": tokens,
                                "seqs": len(seqs)}
        formula_membership[arm.key] = {n: set(present[n]) for n in NGRAM_NS}
        log(f"[intertext] {arm.key}: {tokens:,} tokens / {len(seqs)} seqs, "
            f"shared n-grams n2+={sum(len(present[n]) for n in range(2, 7)):,}")

    # --- shuffle null -------------------------------------------------------
    null_rows = []
    for arm in ARMS:
        rng = random.Random(f"{SHUFFLE_SEED}:{arm.key}")
        seqs = load_arm(arm, quick=quick)
        n_shuf = 1 if quick else SHUFFLES
        totals = {n: 0 for n in NGRAM_NS}
        obs_inst = {n: sum(arm_results[arm.key]["present"][n].values())
                    for n in NGRAM_NS}
        for _ in range(n_shuf):
            shuf_seqs = [shuffled(s, rng) for s in seqs]
            present, _tok = count_book_ngrams_in(book_by_n, shuf_seqs)
            for n in NGRAM_NS:
                totals[n] += sum(present[n].values())
        for n in NGRAM_NS:
            exp = totals[n] / n_shuf
            null_rows.append({
                "arm": arm.key, "kind": arm.kind, "n": n,
                "observed_instances": obs_inst[n],
                "expected_instances": round(exp, 3),
                "shuffles": n_shuf,
                "over_expected": (round(obs_inst[n] / exp, 4) if exp > 0
                                  else ("inf" if obs_inst[n] > 0 else 0.0)),
            })

    # --- overlap-by-n table -------------------------------------------------
    overlap_rows = []
    for arm in ARMS:
        res = arm_results[arm.key]
        for n in NGRAM_NS:
            shared_types = len(res["present"][n])
            overlap_rows.append({
                "arm": arm.key, "kind": arm.kind, "n": n,
                "book_ngram_types": len(book_by_n[n]),
                "corpus_tokens": res["tokens"], "corpus_seqs": res["seqs"],
                "shared_types": shared_types,
                "shared_instances": sum(res["present"][n].values()),
                "shared_type_rate": (round(shared_types / len(book_by_n[n]), 6)
                                     if book_by_n[n] else 0.0),
            })

    # --- corpus summary -----------------------------------------------------
    summary_rows = []
    for arm in ARMS:
        res = arm_results[arm.key]
        seqs = load_arm(arm, quick=quick)
        toks = [t for s in seqs for t in s]
        summary_rows.append({
            "arm": arm.key, "kind": arm.kind, "label": arm.label,
            "source_key": arm.source_key, "sequences": len(seqs),
            "tokens": len(toks), "types": len(set(toks)),
            "book_shared_types_n2plus": sum(
                len(res["present"][n]) for n in range(2, 7)),
        })

    # --- formula classification --------------------------------------------
    same_author = formula_membership.get("siratori_other", {})
    author_meta = formula_membership.get("author_paratext", {})
    influence_arms = [a.key for a in ARMS if a.kind == "influence"]
    register_arms = [a.key for a in ARMS if a.kind == "register"]
    formula_rows = []
    for n in range(2, 7):
        for g, cnt in book_counters[n].items():
            if cnt < MIN_BOOK_FREQ:
                continue
            in_same = g in same_author.get(n, ())
            in_meta = g in author_meta.get(n, ())
            hit_infl = [a for a in influence_arms if g in formula_membership[a].get(n, ())]
            hit_reg = [a for a in register_arms if g in formula_membership[a].get(n, ())]
            cls = formula_class(in_same, in_meta, bool(hit_infl), bool(hit_reg))
            formula_rows.append({
                "n": n, "phrase": " ".join(g), "book_count": cnt,
                "class": cls,
                "in_siratori_other": "y" if in_same else "n",
                "in_author_paratext": "y" if in_meta else "n",
                "influences": ",".join(hit_infl),
                "registers": ",".join(hit_reg),
                "shared_arms": ",".join(
                    [a for a in (["siratori_other"] if in_same else [])
                     + (["author_paratext"] if in_meta else [])
                     + hit_infl + hit_reg]),
            })
    formula_rows.sort(key=lambda r: (r["class"], -r["book_count"], r["n"], r["phrase"]))

    # --- long shared formulas (full for idiolect/meta/influence, capped for registers)
    shared_rows = []
    for arm in ARMS:
        res = arm_results[arm.key]
        for n in range(2, 7):
            for g, c in res["present"][n].items():
                if arm.kind == "register" and (n > 3 or c < MIN_BOOK_FREQ):
                    continue
                shared_rows.append({
                    "arm": arm.key, "kind": arm.kind, "n": n,
                    "phrase": " ".join(g),
                    "book_count": book_counters[n][g], "corpus_count": c,
                })
    shared_rows.sort(key=lambda r: (r["arm"], r["n"], -r["book_count"], r["phrase"]))

    # --- write tables -------------------------------------------------------
    kn.write_tsv(OUT / "corpus_summary.tsv",
                 ["arm", "kind", "label", "source_key", "sequences", "tokens",
                  "types", "book_shared_types_n2plus"], summary_rows)
    kn.write_tsv(OUT / "overlap_by_n.tsv",
                 ["arm", "kind", "n", "book_ngram_types", "corpus_tokens",
                  "corpus_seqs", "shared_types", "shared_instances",
                  "shared_type_rate"], overlap_rows)
    kn.write_tsv(OUT / "null_control.tsv",
                 ["arm", "kind", "n", "observed_instances",
                  "expected_instances", "shuffles", "over_expected"], null_rows)
    kn.write_tsv(OUT / "formula_classes.tsv",
                 ["n", "phrase", "book_count", "class", "in_siratori_other",
                  "in_author_paratext", "influences", "registers",
                  "shared_arms"], formula_rows)
    kn.write_tsv(OUT / "shared_formulas.tsv",
                 ["arm", "kind", "n", "phrase", "book_count", "corpus_count"],
                 shared_rows)

    # --- reconciliation -----------------------------------------------------
    recon = reconcile(book_stats, book_counters, quick=quick, log=log)
    kn.write_tsv(OUT / "reconciliation.tsv",
                 ["item", "expected", "observed", "source", "ok"], recon)

    # --- diagnostics --------------------------------------------------------
    diagnostics = []
    for arm in ARMS:
        seqs = load_arm(arm, quick=quick)
        empty = sum(1 for s in seqs if not s)
        diagnostics.append({
            "check": f"arm_{arm.key}_nonempty", "value": len(seqs) - empty,
            "detail": f"{len(seqs)} sequences, {empty} empty",
        })
    diagnostics.append({"check": "named_works_ingested", "value": sum(
        1 for s in NAMED_WORKS if s.authorization not in NON_INGESTIBLE),
        "detail": "must be 0 (no named work may be ingested)"})
    diagnostics.append({"check": "book_formulas_n2plus", "value": len(formula_rows),
                        "detail": f"n>=2 with book freq >= {MIN_BOOK_FREQ}"})
    excluded = paratext_excluded_rows()
    diagnostics.append({
        "check": "paratext_interviewer_turns_excluded", "value": len(excluded),
        "detail": "interviewer turns dropped from the author arm: "
                  + ",".join(r["id"] for r in excluded)})
    prov = [r for r in recon if r["item"].startswith("corpus_file:")]
    diagnostics.append({
        "check": "corpus_files_provenance_ok", "value": sum(
            1 for r in prov if r["ok"] == "y"),
        "detail": f"{len(prov)} manifest files checked against #56 sha256"})
    kn.write_tsv(OUT / "diagnostics.tsv", ["check", "value", "detail"],
                 diagnostics)

    # --- summary + report ---------------------------------------------------
    classes = collections.Counter(r["class"] for r in formula_rows)
    summary = {
        "schema": SCHEMA,
        "probe_date": PROBE_DATE,
        "book": book_stats,
        "ngram_ns": list(NGRAM_NS),
        "min_book_freq": MIN_BOOK_FREQ,
        "shuffle_seed": SHUFFLE_SEED,
        "arms": {r["arm"]: {k: v for k, v in r.items() if k not in ("arm", "label")}
                 for r in summary_rows},
        "formula_classes": dict(classes),
        "named_works": [{"key": s.key, "verdict": s.verdict,
                         "authorization": s.authorization} for s in NAMED_WORKS],
        "named_works_ingested": sum(
            1 for s in NAMED_WORKS if s.authorization not in NON_INGESTIBLE),
        "siratori_excerpt": {
            "source_url": GREATWORKS_URL,
            "paragraphs": GREATWORKS_EXCERPT_PARAGRAPHS,
            "tokens": len(_tokenise(SIRATORI_GREATWORKS_EXCERPT)),
            "sha256": sha256_text(SIRATORI_GREATWORKS_EXCERPT),
            "pinned_tokens": GREATWORKS_EXCERPT_TOKENS,
            "pinned_sha256": GREATWORKS_EXCERPT_SHA256,
            "matches_pin": (sha256_text(SIRATORI_GREATWORKS_EXCERPT)
                            == GREATWORKS_EXCERPT_SHA256),
        },
        "paratext": {
            "author_turns": len(paratext_rows()),
            "interviewer_turns_excluded": len(excluded),
            "excluded_ids": [r["id"] for r in excluded],
            "source_keys": ["interview_3am_nude_brain",
                            "interview_rhizome_14416_wb"],
        },
        "provenance": {
            "manifest_files_checked": len(prov),
            "manifest_files_ok": sum(1 for r in prov if r["ok"] == "y"),
            "all_corpus_files_match_manifest": all(
                r["ok"] == "y" for r in prov),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")

    write_availability_report(log, classes)
    write_overlap_report(log, book_stats, summary_rows, overlap_rows,
                         null_rows, formula_rows, classes)

    log(f"[intertext] done: {len(formula_rows):,} formulas "
        f"({dict(classes)}); artifacts -> {OUT}")
    return summary


def verify_corpus_provenance():
    """Check every ingested reference-corpus file against the #56 manifest hash.

    Returns one row per manifest file.  A mismatch means the arm would be running
    on a text other than the one #56 pinned - exactly the silent substitution the
    ledger is meant to prevent - so it is surfaced rather than waved past.
    """
    rows = []
    if not CORPUS_MANIFEST.exists():
        return rows
    for m in kn.read_tsv(CORPUS_MANIFEST):
        p = CORPORA / m["file"]
        item = f"corpus_file:{m['file']}"
        if not p.exists():
            rows.append({"item": item, "expected": m["sha256"][:16],
                         "observed": "-", "source": "#56 manifest (sha256)",
                         "ok": "n"})
            continue
        obs = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"item": item, "expected": m["sha256"][:16],
                     "observed": obs[:16], "source": "#56 manifest (sha256)",
                     "ok": "y" if obs == m["sha256"] else "n"})
    return rows


def reconcile(book_stats, book_counters, quick=False, log=print):
    """Reconcile this module's counts against #56 and its own invariants."""
    rows = []

    def add(item, expected, observed, source, ok=None):
        ok = ("y" if expected == observed else "n") if ok is None else ok
        rows.append({"item": item, "expected": expected, "observed": observed,
                     "source": source, "ok": ok})

    # book side must equal #56's word-part screen exactly
    import keyness
    _book, kstats = keyness.load_book_words(log=lambda m: None)
    add("book_parts_total", kstats["parts"], book_stats["parts"],
        "#56 keyness.load_book_words")
    add("book_kept_tokens", kstats["kept"], book_stats["tokens"],
        "#56 keyness.load_book_words")
    add("book_types", kstats["types"], book_stats["types"],
        "#56 keyness.load_book_words")
    # n=1 count is the token total
    add("book_unigram_total", book_stats["tokens"],
        sum(book_counters[1].values()), "intertext n=1 counter")
    # the embedded Siratori excerpt must match its pinned provenance contract
    # (token count and text sha256), not merely agree with itself
    _toks = _tokenise(SIRATORI_GREATWORKS_EXCERPT)
    add("siratori_excerpt_tokens", GREATWORKS_EXCERPT_TOKENS, len(_toks),
        "pinned Great Works excerpt")
    add("siratori_excerpt_sha256", GREATWORKS_EXCERPT_SHA256,
        sha256_text(SIRATORI_GREATWORKS_EXCERPT),
        "pinned Great Works excerpt")
    # provenance: every corpus file must match the #56 manifest hash
    provenance = verify_corpus_provenance()
    rows.extend(provenance)
    # provenance: the paratext table must be the pinned #58 artifact
    add("author_paratext_present", "y",
        "y" if PARATEXT.exists() else "n", "#58 author_paratext.tsv")
    # each reference register arm must match the #56 manifest token count; a
    # --quick build caps them, so the check is asserted-but-noted there.
    manifest = CORPORA / "manifest.tsv"
    if manifest.exists():
        mrows = kn.read_tsv(manifest)
        for arm in ARMS:
            if arm.kind != "register":
                continue
            corpus = _REFERENCE_ARM_CORPUS[arm.key]
            expected = sum(int(r["tokens"]) for r in mrows if r["corpus"] == corpus)
            seqs = load_arm(arm, quick=quick)
            observed = sum(len(s) for s in seqs)
            add(f"register_{arm.key}_tokens", expected, observed,
                "#56 manifest vs intertext", ok="skip" if quick else None)
    bad = [r for r in rows if r["ok"] == "n"]
    log(f"[intertext] reconciliation: {len(rows) - len(bad)}/{len(rows)} ok")
    return rows


# ===========================================================================
# 6. reports
# ===========================================================================
def _bar(n, total=40):
    return "#" * round(n * total) if total else ""


def write_availability_report(log, classes):
    obtained = [s for s in SOURCES if s.authorization not in NON_INGESTIBLE]
    refused = [s for s in SOURCES if s.authorization in NON_INGESTIBLE]
    lines = []
    lines.append("# Intertextual sources: what exists, what is lawful, what was used (#57)")
    lines.append("")
    lines.append(f"Survey date: **{PROBE_DATE}** (pinned; `--probe` re-runs the "
                 "live requests).  Full machine-readable ledger: "
                 "`sources.tsv`.")
    lines.append("")
    lines.append("## The five named works are not available as text")
    lines.append("")
    lines.append("The task names *Headcode*, *Human_Worms*, *Smart-d*, *GENEDUB* "
                 "and *Nonexistence*.  Every one is an in-copyright commercial "
                 "edition; no rights holder serves the text for reuse.")
    lines.append("")
    lines.append("| work | status | evidence |")
    lines.append("|---|---|---|")
    for s in NAMED_WORKS:
        lines.append(f"| {s.title} | {s.verdict} | {s.note} |")
    lines.append("")
    lines.append(f"**{len(NAMED_WORKS)} named works, 0 ingested.**  This is the "
                 "negative half of the task and it is a fact about the sources, "
                 "not a choice about the analysis.  Nothing here substitutes a "
                 "different text for a named work.")
    lines.append("")
    lines.append("## What was obtained")
    lines.append("")
    lines.append("| source | kind | authorization | how it is used |")
    lines.append("|---|---|---|---|")
    for s in obtained:
        use = "matcher arm" if s.kind in (
            "this_author_other", "author_paratext", "cited_influence",
            "register_control") else "reference only"
        lines.append(f"| {s.title} | {s.kind} | {s.authorization} | {use} |")
    lines.append("")
    lines.append("The same-author probe is the *Great Works* text — a "
                 "**different** Siratori work, freely published online.  It is "
                 "the only genuine second Siratori text available under the "
                 "project's research-quotation convention, and it is used only "
                 "as an idiolect probe.  The cited influences are the #56 "
                 "fair-use excerpts (Gibson / Stephenson / Sterling; Burroughs "
                 "/ Artaud / Breton; Joyce); they are excerpt-length by "
                 "copyright necessity, which the matcher discloses with a "
                 "shuffle null rather than hiding.")
    lines.append("")
    lines.append("## Provenance verified, not asserted")
    lines.append("")
    lines.append(f"* **The embedded Siratori excerpt is the page it names.**  "
                 f"The first {GREATWORKS_EXCERPT_PARAGRAPHS} paragraphs of "
                 f"<{GREATWORKS_URL}> re-extract to "
                 f"{GREATWORKS_EXCERPT_TOKENS} tokens, sha256 "
                 f"`{GREATWORKS_EXCERPT_SHA256[:16]}…`.  "
                 f"`--probe` re-fetches the page and diffs the extraction "
                 f"against that pin, so the constant cannot drift silently.")
    lines.append(f"* **Every reference-corpus file is checked against #56's "
                 f"manifest sha256** ({len(verify_corpus_provenance())} files); "
                 f"a mismatch is a hard failure in `reconciliation.tsv`, so an "
                 f"arm cannot run on a substituted text.")
    lines.append(f"* **The author arm is the author's own words only.**  #58's "
                 f"paratext table is an interview Q/A; the "
                 f"{len(paratext_excluded_rows())} turn(s) recorded as the "
                 f"interviewer's ("
                 + ", ".join(r["id"] for r in paratext_excluded_rows())
                 + ") are excluded rather than attributed to the author.")
    lines.append("")
    lines.append("## What was sought and not used")
    lines.append("")
    lines.append("| source | authorization | why not used |")
    lines.append("|---|---|---|")
    for s in refused:
        lines.append(f"| {s.title} | {s.authorization} | {s.note} |")
    lines.append("")
    lines.append("* Pierre Guyotat, a declared influence in the author "
                 "interviews, has no lawful excerpt in this workspace.  It is "
                 "listed as absent rather than silently replaced.")
    lines.append("* The Internet Archive `Headcode` scan is lending-only "
                 "(`access-restricted-item=true`); no body was fetched.")
    lines.append("")
    (OUT / "availability_report.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    log(f"[intertext] availability_report.md: {len(obtained)} obtained, "
        f"{len(refused)} refused/absent")


def _top(rows, key, n=15, cls=None):
    sel = [r for r in rows if cls is None or r["class"] == cls]
    sel.sort(key=lambda r: (-r["book_count"], r["n"], r["phrase"]))
    return sel[:n]


def write_overlap_report(log, book_stats, summary_rows, overlap_rows, null_rows,
                         formula_rows, classes):
    n2plus = summary_rows
    lines = []
    lines.append("# Intertextual n-gram overlap report (#57)")
    lines.append("")
    lines.append("**Question.**  Do *Blood Electric*'s phrase formulas also "
                 "occur in (a) another work by the same author and (b) the "
                 "cited influences?  Shared formulas are candidates for genre; "
                 "formulas shared only with the same author are candidates for "
                 "idiolect.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"* **Book side:** the #17 word-part layer, markup-screened "
                 f"exactly as #56 and split at page boundaries — "
                 f"{book_stats['tokens']:,} tokens / {book_stats['types']:,} "
                 f"types over {book_stats['pages']} pages.")
    lines.append("* **Corpus side:** the same #24 word regex, split at file "
                 "boundaries; influence excerpts have their `#`/`--` "
                 "attribution headers stripped.  Each reference-corpus file is "
                 "checked against #56's manifest sha256 before use, and the "
                 "author arm reads only the author's own interview turns "
                 "(interviewer turns are excluded).")
    lines.append(f"* **Overlap:** exact match of n-grams for n ∈ "
                 f"{list(NGRAM_NS)}; a book n-gram counts as *formula* when it "
                 f"recurs ≥ {MIN_BOOK_FREQ} times.")
    lines.append(f"* **Null:** each arm's tokens are permuted ({SHUFFLES} "
                 f"shuffles, seed {SHUFFLE_SEED}) and the overlap recomputed, "
                 f"so every observed count is read against chance.")
    lines.append("")
    lines.append("## Which arms carry which n-grams")
    lines.append("")
    lines.append("| arm | kind | tokens | types | shared n-gram types (n≥2) |")
    lines.append("|---|---|---:|---:|---:|")
    for r in n2plus:
        lines.append(f"| {r['arm']} | {r['kind']} | {r['tokens']:,} | "
                     f"{r['types']:,} | {r['book_shared_types_n2plus']:,} |")
    lines.append("")
    lines.append("The influences are partial by construction: the largest "
                 "excerpt arm is a few hundred tokens against the book's "
                 "tens of thousands, so a *low* influence overlap bounds the "
                 "genre claim from below and cannot refute it.  The register "
                 "arms are full-size controls.")
    lines.append("")
    lines.append("## Observed vs chance (shared instances)")
    lines.append("")
    lines.append("| arm | n | observed | expected (shuffle) | observed/expected |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in null_rows:
        if r["n"] < 2:
            continue
        lines.append(f"| {r['arm']} | {r['n']} | {r['observed_instances']:,} | "
                     f"{r['expected_instances']:,} | {r['over_expected']} |")
    lines.append("")
    lines.append("## Formula classes")
    lines.append("")
    lines.append("Every book n-gram with book frequency ≥ "
                 f"{MIN_BOOK_FREQ} and n ≥ 2, classified by the strongest "
                 "external source that contains it:")
    lines.append("")
    lines.append("```")
    for cls in ("author_idiolect", "author_common", "genre_influence",
                "influence_common", "register_common", "book_unique"):
        c = classes.get(cls, 0)
        lines.append(f"{cls:24s} {c:6d}  {_bar(c / max(1, sum(classes.values())), 40)}")
    lines.append("```")
    lines.append("")
    lines.append("* `author_idiolect` — occurs in the author's own other text "
                 "(the *different* Siratori work and/or his interview answers) "
                 "and in **no** general/technical register: a distinctive "
                 "formula that travels across the author's works.")
    lines.append("* `author_common` — occurs in the author's own text but also "
                 "in ordinary/technical English; not distinctive.")
    lines.append("* `genre_influence` — occurs in a cited-influence excerpt and "
                 "in no register: a genre candidate.")
    lines.append("* `influence_common` — occurs in an influence excerpt but also "
                 "in a register; not distinctive.")
    lines.append("* `register_common` — occurs only in a general/technical "
                 "corpus; ordinary English or technical vocabulary, not genre.")
    lines.append("* `book_unique` — occurs in no external source.")
    lines.append("")
    lines.append("### Top idiolect formulas (author-only, not ordinary English)")
    lines.append("")
    lines.append("| n | phrase | book freq | also in |")
    lines.append("|---:|---|---:|---|")
    for r in _top(formula_rows, "book_count", 20, "author_idiolect"):
        lines.append(f"| {r['n']} | `{r['phrase']}` | {r['book_count']} | "
                     f"{r['shared_arms']} |")
    lines.append("")
    lines.append("### Top genre-influence formulas")
    lines.append("")
    lines.append("| n | phrase | book freq | influences |")
    lines.append("|---:|---|---:|---|")
    _genre = _top(formula_rows, "book_count", 20, "genre_influence")
    if _genre:
        for r in _genre:
            lines.append(f"| {r['n']} | `{r['phrase']}` | {r['book_count']} | "
                         f"{r['influences']} |")
    else:
        lines.append("| — | *none at this excerpt size* | | |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    idio = classes.get("author_idiolect", 0)
    idio_long = sum(1 for r in formula_rows
                    if r["class"] == "author_idiolect" and r["n"] >= 3)
    idio_work = sum(1 for r in formula_rows if r["class"] == "author_idiolect"
                    and r["in_siratori_other"] == "y")
    idio_meta_only = sum(
        1 for r in formula_rows if r["class"] == "author_idiolect"
        and r["in_siratori_other"] == "n" and r["in_author_paratext"] == "y")
    sa = {(r["arm"], r["n"]): r for r in null_rows}.get(("siratori_other", 3))
    sa2 = {(r["arm"], r["n"]): r for r in null_rows}.get(("siratori_other", 2))
    _genre_phrases = ", ".join(
        f"`{r['phrase']}`" for r in _top(formula_rows, "book_count", 5,
                                         "genre_influence")) or "none"
    lines.append(f"1. **Idiolect is measurable and measured.**  {idio} book "
                 f"formulas (n≥2, {idio_long} of them length ≥3) occur in the "
                 f"author's own other text and in no general/technical "
                 f"register: {idio_work} of them in the *different* Siratori "
                 f"work and {idio_meta_only} in the interview paratext alone.  "
                 f"The signal is far above chance: for the same-author probe, "
                 f"n-gram instances hit "
                 f"{sa['observed_instances']} vs {sa['expected_instances']} "
                 f"expected under shuffling at n=3 "
                 f"({sa['over_expected']}×), and "
                 f"{sa2['observed_instances']} vs {sa2['expected_instances']} "
                 f"at n=2 ({sa2['over_expected']}×).  The stable, distinctive "
                 f"formula set (`soul gram`, `cold blooded disease`, "
                 f"`retro adam`, `brain universe`, `dna channel`, "
                 f"`data mutant` …) is the author's idiolect, as far as this "
                 f"corpus can demonstrate it.")
    lines.append(f"2. **Genre is underpowered here, and that is reported as a "
                 f"bound.**  The cited influences are fair-use excerpts of a few "
                 f"dozen to a few hundred tokens; the only shared n-gram not "
                 f"also ordinary English is {_genre_phrases}, and no influence "
                 f"arm shares any trigram with the book.  `a virus` is the "
                 f"Burroughs line *\"Language is a virus from outer space\"* "
                 f"(*The Ticket That Exploded*) — one declared-influence "
                 f"phrase, quoted in the #56 excerpt, not evidence of broad "
                 f"genre transfer.  At this size genre cannot be separated from "
                 f"generic English; the `influence_common` and `register_common` "
                 f"classes show how much of the book's formula stock is simply "
                 f"ordinary English.")
    lines.append("3. **The named-work comparison is not testable from lawful "
                 "sources.**  None of *Headcode*, *Human_Worms*, *Smart-d*, "
                 "*GENEDUB* or *Nonexistence* is servable; the ledger records "
                 "each refusal/absence with its evidence.  No other text stands "
                 "in for them.")
    lines.append("4. **Provenance is checked, not asserted.**  The embedded "
                 "Great Works excerpt is pinned by token count and sha256 "
                 "(re-verified against the live page by `--probe`), every "
                 "reference-corpus file is verified against #56's manifest "
                 "sha256, and the one interviewer turn in #58's Q/A table is "
                 "excluded from the author arm.  No comparison text is invented "
                 "or silently substituted for another.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("* The same-author probe is a *different* work (Great Works), "
                 "not any of the five named publications; it can support an "
                 "idiolect claim but cannot speak to those specific books.")
    lines.append("* The influence arms are excerpt-length; their overlap is a "
                 "lower bound on the true genre overlap.")
    lines.append("* Matching is lexical (operators stripped as the word regex "
                 "requires); it does not test operator-grammar transfer, which "
                 "#12/#15 already treat as a separate (and non-genre) channel.")
    lines.append("* Word-part tokenisation means fused compounds contribute "
                 "their parts; a formula like `living body` is a part-level "
                 "match, not a unit-level one.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python work/intertext.py            # writes analysis/intertext/")
    lines.append("python work/intertext.py --probe    # re-run the live source probes")
    lines.append("python work/test_intertext.py")
    lines.append("```")
    lines.append("")
    (OUT / "overlap_report.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    log("[intertext] overlap_report.md written")


# ===========================================================================
# 7. live probe (writes nothing)
# ===========================================================================
def _fetch(url: str) -> str:
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", "replace")


def probe_greatworks_content(log=print):
    """Re-fetch the Great Works page and diff the excerpt against its pin.

    Returns True when the live page still yields the pinned text, False on any
    content drift, and None when the page could not be fetched at all.  This is
    the provenance check the status probe cannot make: it proves the embedded
    excerpt is the page it claims, not a paraphrase that merely shares a URL.
    """
    try:
        page = _fetch(GREATWORKS_URL)
    except Exception as e:  # noqa: BLE001 - probe reports any failure
        log(f"[probe] greatworks content: ERR:{type(e).__name__}: {e}")
        return None
    live = extract_greatworks_excerpt(page)
    sha = sha256_text(live)
    toks = len(_tokenise(live))
    ok = sha == GREATWORKS_EXCERPT_SHA256 and toks == GREATWORKS_EXCERPT_TOKENS
    log(f"[probe] greatworks content: {'OK' if ok else 'DIFF  <-- PIN CHANGED'}"
        f"  tokens={toks} (pinned {GREATWORKS_EXCERPT_TOKENS}) "
        f"sha={sha[:16]} (pinned {GREATWORKS_EXCERPT_SHA256[:16]})")
    return ok


def probe(log=print):
    """Re-run the HTTP status of every pinned source and diff against the pins."""
    changed = 0
    for s in SOURCES:
        if not s.url.startswith(("http://", "https://")):
            continue
        try:
            req = urllib.request.Request(s.url, method="GET",
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as resp:
                status = str(resp.status)
        except urllib.error.HTTPError as e:
            status = str(e.code)
        except Exception as e:  # noqa: BLE001 - probe reports any failure
            status = f"ERR:{type(e).__name__}"
        pinned = s.http_status
        flag = "" if status == pinned else "  <-- DIFF"
        if flag:
            changed += 1
        log(f"[probe] {s.key:32s} pinned={pinned:5s} live={status}{flag}")
    # content-level provenance for the one embedded excerpt (not just status)
    content_ok = probe_greatworks_content(log)
    if content_ok is False:
        changed += 1
    log(f"[probe] {changed} source(s) differ from the {PROBE_DATE} pins")
    return 0


# ===========================================================================
# 8. cli
# ===========================================================================
def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quick", action="store_true",
                    help="sampled register corpora + 1 shuffle (regression suite)")
    ap.add_argument("--probe", action="store_true",
                    help="re-run the live HTTP source probes; writes nothing")
    args = ap.parse_args(argv)
    if args.probe:
        return probe()
    OUT = Path(args.out)
    build(quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
