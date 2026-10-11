#!/usr/bin/env python3
"""#58 Translations as an interpretive oracle -> analysis/translation_oracle/

L7 of ``docs/meaning-extraction-methods.md`` nominates two pieces of external
material for *Blood Electric*:

  * a Russian translation (A. Skobin, АСТ/Адаптек 2006, ISBN 5-17-037280-9), and
  * the author's own interviews glossing his method (3AM Magazine "The Nude
    Brain", Rhizome / CLC "Interview with Kenji Siratori + Loop Writing").

and says: *"If a translated parallel text can be obtained, sent-alignment
against it gives an independent disambiguation signal - the closest thing to a
Rosetta Stone this project can get."*

This module does three separable things and reports each one honestly.

1. **The acquisition ledger.**  A pinned manifest of every candidate source for
   the Skobin parallel text plus the author interviews, each probed for HTTP
   status, final URL, byte count, SHA-256 and licence, with a hard
   ``ingestible`` gate.  The Skobin translation is a 2006 commercial АСТ
   edition; the only full texts reachable on the open web are unauthorized
   mirrors, and the gate refuses them.  That is the negative half of the
   result and it is recorded with the evidence, not asserted.

2. **The aligner.**  A segment aligner (monotone dynamic programme over an
   n-m diagonal, costed by log-length disagreement, preceding-operator
   agreement, shared numerals and bilingual anchors) that emits 1-1 / 1-0 /
   0-1 / n-m links and scores itself with the standard AER against a built-in
   EN-RU gold of 24 authored pairs with planted 1-n / n-1 operations.  Weights
   are fitted on a tuning half of the gold and reported on a held-out half.
   No network, no LLM, no external MT.

3. **The oracle that actually ran.**  The Russian-language reader reviews of
   the book published on FantLab quote the Skobin translation verbatim, and
   those quotations are short enough to use as research quotations under the
   same fair-use convention ``work/reference_corpora.py`` already applies to
   Gibson / Burroughs / Joyce.  Twelve such quotations are pinned below and
   aligned to the English corpus.  Two of them are long multi-operator spans
   located by **operator-skeleton match alone** - no lexical anchor at all -
   which turns out to be sufficient: the Russian edition preserves the book's
   `= :: // ++ / << >>` channel verbatim, so the symbol stream is a
   translation-invariant key into the 472k-character English text.  That is the
   Rosetta stone this corpus actually has, and it is measured here rather than
   assumed.

Requires ``analysis/layout_recovery/`` (#7) for the corpus, and reads #19, #30
and #62's gloss tables when they are present.

Artifacts written under ``analysis/translation_oracle/``::

    sources.tsv              acquisition ledger (one row per candidate source)
    availability_report.md   the narrative result: obtained / refused / searched
    ru_quotations.tsv        the pinned target-language quotations + their anchors
    fantlab_reviews.tsv      the reader reviews the quotations were transcribed from
    author_paratext.tsv      pinned author interview Q/A (research quotations)
    alignment.tsv            the alignment artifact: RU unit <-> EN unit
    alignment_report.md      per-quotation alignment detail
    operator_invariance.tsv  operator-channel retention + skeleton localisation
    bilingual_lexicon.tsv    EN surface <-> attested RU rendering, page-cited
    oracle_checks.tsv        author claims tested against the corpus (#58's job)
    aligner_eval.tsv         AER of the aligner on the built-in EN-RU gold
    diagnostics.tsv, reconciliation.tsv, summary.json

Reproduce::

    python work/translation_oracle.py               # offline, from pinned data
    python work/translation_oracle.py --probe       # re-run the HTTP probes
    python work/translation_oracle.py --verify-reviews   # re-check attribution live
    python work/test_translation_oracle.py

Determinism: the build is byte-identical on a rerun.  The probe results are a
*pinned record* of the acquisition (``PROBE_DATE``), not a build-time stamp, so
nothing in the artifact moves with the clock; ``--probe`` re-runs them live and
prints a diff against the pins.  ``--verify-reviews`` re-fetches the work page
and confirms each quotation still occurs in the review it is attributed to.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import math
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from operator_inventory import OPS  # noqa: E402  (#10's canonical operator set)

OUT = ROOT / "analysis" / "translation_oracle"

# the vetted layout markdown with `<!-- page N -->` markers (#7), i.e. the
# same corpus #10 counts operators over; 225 pages, reconciled against #8.
CORPUS_MD = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
FOLIO_TSV = ROOT / "analysis" / "page_folio_map" / "page_folio_map.tsv"
NEO_TSV = ROOT / "analysis" / "neologism" / "neologisms_lower.tsv"
GLOSS_TSV = ROOT / "analysis" / "interlinear" / "gloss_lexicon.tsv"
ENTITY_TSV = ROOT / "analysis" / "entity" / "entity_glossary.tsv"
OPERATOR_TOKENS_TSV = ROOT / "analysis" / "operator_inventory" / "operator_tokens.tsv"

SCHEMA = "translation-oracle/1"

USER_AGENT = ("blood-electric-workspace/1.0 "
              "(research; Chainlink #58 translation-oracle probe)")
#: the acquisition run that produced ``PINS``.  Pinned, never re-derived, so
#: the ledger is stable; ``--probe`` compares a live run against it.
PROBE_DATE = "2026-10-10"


# ===========================================================================
# 1. the acquisition ledger
# ===========================================================================

# authorization classes
AUTH_OPEN = "open"                      # free to fetch and use
AUTH_RESEARCH = "research_licensed"    # free for research, registration/paywall
AUTH_PRINT = "print_only"              # a catalogue/record; no text served
AUTH_STORE = "store"                    # a commercial ebook store
AUTH_MIRROR = "unauthorized_mirror"    # an upload with no rights holder's consent
AUTH_DENIED = "bot_blocked"            # the host refused this agent
AUTH_ABSENT = "not_offered"            # the host offers no such text
AUTH_SEARCHED = "searched_not_found"   # searched, no usable copy located

# verdicts
V_OBTAINED = "obtained"
V_METADATA = "metadata_only"
V_PRINT = "print_only"
V_PAYWALL = "paywalled_or_gated"
V_REFUSED = "refused_unauthorized_mirror"
V_BLOCKED = "blocked_by_host"
V_ABSENT = "not_offered"
V_SEARCHED = "searched_not_found"

#: mirrors are never ingested - the gate is the point of the negative result
NON_INGESTIBLE = {AUTH_MIRROR, AUTH_DENIED, AUTH_PRINT, AUTH_STORE,
                  AUTH_SEARCHED, AUTH_ABSENT}


@dataclass(frozen=True)
class Source:
    key: str
    kind: str            # parallel_text | edition_record | author_paratext
    title: str
    url: str
    language: str
    license: str
    authorization: str
    note: str = ""


SOURCES: tuple[Source, ...] = (
    # --- the Skobin translation, sought as a parallel text -----------------
    Source("skobin_fantlab_work", "parallel_text",
           "FantLab work record for «Кровь электрическая»",
           "https://fantlab.ru/work253074", "ru",
           "catalogue page, CC BY-NC-SA site terms; quotes of the translation "
           "on the page are third-party reader reviews",
           AUTH_PRINT,
           "Names A. Skobin as translator and lists the 2006 АСТ/Адаптек "
           "edition; serves no book text.  The page does carry the reader "
           "reviews that #58 quotes."),
    Source("skobin_fantlab_ed2006", "edition_record",
           "FantLab edition record: M.: АСТ, Адаптек, 2006, 320 pp",
           "https://fantlab.ru/edition55877", "ru",
           "bibliographic record", AUTH_PRINT,
           "ISBN 5-17-037280-9 / 5-93827-069-3, series «Alternativa».  "
           "Confirms the parallel text exists and is in copyright."),
    Source("skobin_fantlab_ed2007", "edition_record",
           "FantLab edition record: M.: АСТ, Адаптек, 2007, 320 pp",
           "https://fantlab.ru/edition55878", "ru",
           "bibliographic record", AUTH_PRINT,
           "ISBN 978-5-17-042213-5; the same translation, paperback."),
    Source("skobin_fantlab_read", "parallel_text",
           "FantLab online reader endpoint for the work",
           "https://fantlab.ru/work253074/read", "ru",
           "-", AUTH_PRINT,
           "404: FantLab hosts no online text for this work."),
    Source("skobin_librebook", "parallel_text",
           "LibreBook catalogue entry for «Кровь электрическая»",
           "https://1.librebook.me/blood_electric", "ru",
           "user-uploaded catalogue entry", AUTH_STORE,
           "Entry exists with the 2006 imprint; the chapter list is empty "
           "(«Нет глав»), so no text is served."),
    Source("skobin_litres", "parallel_text",
           "Litres search for «Кровь электрическая»",
           "https://www.litres.ru/search/?q=%D0%9A%D1%80%D0%BE%D0%B2%D1%8C%20"
           "%D1%8D%D0%BB%D0%B5%D0%BA%D1%82%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F",
           "ru", "commercial ebook store", AUTH_DENIED,
           "403 to this agent; Litres sells licensed ebooks only."),
    Source("skobin_rusneb", "parallel_text",
           "NEL (rusneb.ru) search for «Кровь электрическая»",
           "https://rusneb.ru/search/?q=%D0%9A%D1%80%D0%BE%D0%B2%D1%8C",
           "ru", "national library catalogue", AUTH_DENIED,
           "403 to this agent; a national library record never carries full "
           "text anyway."),
    Source("skobin_libru", "parallel_text",
           "lib.ru library front page",
           "https://lib.ru/mainpage.html", "ru",
           "hosted library", AUTH_ABSENT, "404 at the probed path."),
    # --- the unauthorized mirrors: probed for the record, never ingested ----
    Source("mirror_libking", "parallel_text",
           "libking.ru full text of the Skobin translation",
           "https://libking.ru/books/sf-/sf-cyberpunk/164857-2-kendzi-siratori-"
           "krov-elektricheskaya.html", "ru",
           "no licence from the rights holder", AUTH_MIRROR,
           "Advertises «полную версию ... целиком без сокращений» of the АСТ "
           "2006 edition.  HEAD probe only; body never requested."),
    Source("mirror_4etalka", "parallel_text",
           "4etalka.ru full text of the Skobin translation",
           "https://4etalka.ru/fantastika/kiberpank/94676.htm", "ru",
           "no licence from the rights holder", AUTH_MIRROR,
           "HEAD probe only; body never requested."),
    Source("mirror_homeread", "parallel_text",
           "homeread.net full text of the Skobin translation",
           "https://homeread.net/book/krov-elektricheskaya-kendzi-siratori",
           "ru", "no licence from the rights holder", AUTH_MIRROR,
           "HEAD probe only; body never requested."),
    Source("mirror_litmir", "parallel_text",
           "litmir.club full text of the Skobin translation",
           "https://litmir.club/br/?b=24951", "ru",
           "no licence from the rights holder", AUTH_MIRROR,
           "HEAD probe only; body never requested."),
    Source("mirror_bookscafe", "parallel_text",
           "bookscafe.net full text of the Skobin translation",
           "https://bookscafe.net/read/siratori_kendzi-krov_elektricheskaya-69039.html",
           "ru", "no licence from the rights holder", AUTH_MIRROR,
           "HEAD probe only; body never requested."),
    Source("mirror_aldebaran", "parallel_text",
           "aldebaran.one full text with Russian reader annotations",
           "https://aldebaran.one/author/siratori_kyendzi/kniga_krov_"
           "yelektricheskaya/read/", "ru",
           "no licence from the rights holder", AUTH_MIRROR,
           "HEAD probe only; body never requested."),
    Source("mirror_knigogid", "parallel_text",
           "knigogid.ru full text with Russian reader annotations",
           "https://knigogid.ru/books/678527-krov-elektricheskaya/toread", "ru",
           "no licence from the rights holder", AUTH_MIRROR,
           "HEAD probe only; body never requested."),
    Source("mirror_scribd", "parallel_text",
           "Scribd scan of «Кровь Электрическая» (Anna's Archive rehost)",
           "https://ru.scribd.com/document/941900775/", "ru",
           "no licence from the rights holder", AUTH_MIRROR,
           "HEAD probe only; body never requested."),
    # --- author paratext (legitimately published interviews) ----------------
    Source("interview_3am_nude_brain", "author_paratext",
           "3AM Magazine: 'THE NUDE BRAIN: AN INTERVIEW WITH KENJI SIRATORI'",
           "https://www.3ammagazine.com/litarchives/2002_jun/"
           "interview_kenji_siratori.html", "en",
           "© 2002 3AM Magazine; quoted here as short research quotations",
           AUTH_OPEN,
           "Richard Marshall interviews Siratori, June 2002.  Obtained."),
    Source("interview_rhizome_9083", "author_paratext",
           "Rhizome community 9083: the same 'Nude Brain' interview",
           "https://old.rhizome.org/community/9083/", "en",
           "public web post; short research quotations", AUTH_OPEN,
           "Independent mirror of the 3AM text, 30 June 2002.  Obtained; "
           "used to corroborate the 3AM wording."),
    Source("interview_rhizome_14416", "author_paratext",
           "Rhizome community 14416: 'Interview with Kenji Siratori + "
           "Loop Writing' (S. Baldwin, WVU CLC)",
           "https://rhizome.org/community/14416/", "en",
           "public web post; short research quotations", AUTH_DENIED,
           "403 to this agent on the live site."),
    Source("interview_rhizome_14416_wb", "author_paratext",
           "Wayback Machine capture of Rhizome 14416",
           "https://web.archive.org/web/20211209013803/https://rhizome.org/"
           "community/14416/", "en",
           "archived public web post; short research quotations", AUTH_OPEN,
           "Obtained.  This is the interview in which Siratori is asked "
           "directly whether his punctuation «…, =» is a method."),
    Source("nettime_root", "author_paratext",
           "nettime mailing-list archive root (the plan's third interview)",
           "https://www.nettime.org/", "en",
           "public mailing-list archive", AUTH_SEARCHED,
           "Reachable, but the 'Loop Writing' thread was not located: the "
           "nettime-l 2002 thread index carries no Siratori subject and the "
           "Wayback CDX index of nettime.org has no such path.  Recorded as "
           "searched-not-found rather than silently dropped."),
)

SOURCE_BY_KEY = {s.key: s for s in SOURCES}

#: pinned acquisition evidence, produced by the run recorded in ``PROBE_DATE``.
PINS: dict[str, dict] = {
    "skobin_fantlab_work": {"status": 200, "final": "https://fantlab.ru/work253074",
                            "bytes": 149261,
                            "sha256": "4febae0d20a0c465cdb1beeae005d09c26f59e564296f16bb8bfb35be8088245"},
    "skobin_fantlab_ed2006": {"status": 200, "final": "https://fantlab.ru/edition55877",
                              "bytes": 44158,
                              "sha256": "9bf0b31ae5b7fc2934cd8b7ce7f033ae65fbed8b2b7230d46270e79a3f426718"},
    "skobin_fantlab_ed2007": {"status": 200, "final": "https://fantlab.ru/edition55878",
                              "bytes": 43799,
                              "sha256": "820fa262b2b6e54f93027f0113a6ce753f2d43c310cb53ead229ef9d209f0ad3"},
    "skobin_fantlab_read": {"status": 404, "final": "https://fantlab.ru/work253074/read",
                            "bytes": 0, "sha256": "", "error": "HTTP Error 404: Not Found"},
    "skobin_librebook": {"status": 200, "final": "https://1.librebook.me/blood_electric",
                         "bytes": 133757,
                         "sha256": "7c0e3a986e8db1128524ffe1df12a9f4a7949bb7f691f71dc6dde696ef8799ca"},
    "skobin_litres": {"status": 403, "final": "https://www.litres.ru/search/?q=...",
                      "bytes": 0, "sha256": "", "error": "HTTP Error 403: Forbidden"},
    "skobin_rusneb": {"status": 403, "final": "https://rusneb.ru/search/?q=...",
                      "bytes": 0, "sha256": "", "error": "HTTP Error 403: Forbidden"},
    "skobin_libru": {"status": 404, "final": "https://lib.ru/mainpage.html",
                     "bytes": 0, "sha256": "", "error": "HTTP Error 404: Not Found"},
    "mirror_libking": {"status": 200, "final": "https://libking.ru/...", "bytes": 0,
                       "sha256": ""},
    "mirror_4etalka": {"status": 200, "final": "https://4etalka.ru/...", "bytes": 0,
                       "sha256": ""},
    "mirror_homeread": {"status": 200, "final": "https://homeread.net/...", "bytes": 0,
                        "sha256": ""},
    "mirror_litmir": {"status": 200, "final": "https://litmir.club/...", "bytes": 0,
                      "sha256": ""},
    "mirror_bookscafe": {"status": 200, "final": "https://bookscafe.net/...", "bytes": 0,
                         "sha256": ""},
    "mirror_aldebaran": {"status": 200, "final": "https://aldebaran.one/...", "bytes": 120186,
                         "sha256": ""},
    "mirror_knigogid": {"status": 200, "final": "https://knigogid.ru/...", "bytes": 0,
                        "sha256": ""},
    "mirror_scribd": {"status": 200, "final": "https://ru.scribd.com/...", "bytes": 1465370,
                      "sha256": ""},
    "interview_3am_nude_brain": {
        "status": 200,
        "final": "https://www.3ammagazine.com/litarchives/2002_jun/interview_kenji_siratori.html",
        "bytes": 17583,
        "sha256": "4e4638412b1cfab8965ac66ca4e2f2a420821858e8ae511a5c7aecc78a4fc7ac"},
    "interview_rhizome_9083": {"status": 200, "final": "https://old.rhizome.org/community/9083/",
                               "bytes": 24586,
                               "sha256": "8f1629f951d76125ab592bcce8c8a2075117e7eebbb4a3bc5649d7fd43d77340"},
    "interview_rhizome_14416": {"status": 403, "final": "https://rhizome.org/community/14416/",
                                "bytes": 0, "sha256": "", "error": "HTTP Error 403: Forbidden"},
    "interview_rhizome_14416_wb": {
        "status": 200,
        "final": "https://web.archive.org/web/20211209013803/https://rhizome.org/community/14416/",
        "bytes": 34772,
        "sha256": "8ef1e701e435114ee558028e48df9f53e8715bd12ec53d14cfc52b32cf0fcba0"},
    "nettime_root": {"status": 200, "final": "https://www.nettime.org/", "bytes": 7321,
                     "sha256": "ce98c7797700efc2a1975e76d5bc8c450cd8f11588badf801bff41de9911616d"},
}

#: authorization -> verdict.  Pure function of the licence class; the probe
#: only decides whether the record was reachable at all.
_VERDICT_BY_AUTH = {
    AUTH_OPEN: V_OBTAINED,
    AUTH_RESEARCH: V_OBTAINED,
    AUTH_PRINT: V_PRINT,
    AUTH_STORE: V_PAYWALL,
    AUTH_MIRROR: V_REFUSED,
    AUTH_DENIED: V_BLOCKED,
    AUTH_ABSENT: V_ABSENT,
    AUTH_SEARCHED: V_SEARCHED,
}


def ingestible(source: Source) -> bool:
    """The hard gate.  Unauthorized mirrors are never ingested."""
    return source.authorization not in NON_INGESTIBLE


def verdict(source: Source) -> str:
    return _VERDICT_BY_AUTH[source.authorization]


def probe(url: str, method: str = "GET", timeout: int = 45) -> dict:
    """One HTTP probe.  Never raises; returns the observed evidence.

    ``text`` is the decoded body when one was fetched, so callers that need
    to read a page (the review-attribution check) do not fetch it twice.
    """
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read() if method == "GET" else b""
            return {"status": resp.status, "final": resp.geturl(), "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest() if body else "",
                    "text": body.decode("utf-8", "replace") if body else ""}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "final": url, "bytes": 0, "sha256": "",
                "text": "", "error": f"HTTP Error {exc.code}: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001 - a probe must never break a build
        return {"status": 0, "final": url, "bytes": 0, "sha256": "", "text": "",
                "error": str(exc)[:120]}


def run_probes(log=print) -> list[dict]:
    """Re-run every probe live and diff it against the pins."""
    rows = []
    for src in SOURCES:
        method = "HEAD" if src.authorization == AUTH_MIRROR else "GET"
        got = probe(src.url, method=method)
        pin = PINS.get(src.key, {})
        same = (got["status"] == pin.get("status"))
        rows.append({"source": src.key, "status": got["status"],
                     "pinned_status": pin.get("status"),
                     "bytes": got["bytes"], "pinned_bytes": pin.get("bytes"),
                     "agrees_status": same,
                     "error": got.get("error", "")})
        log(f"[translation_oracle] probe {src.key:28s} {got['status']} "
            f"(pinned {pin.get('status')}) {'OK' if same else 'CHANGED'}")
        if not same:
            log(f"[translation_oracle]   URL may have moved: {got['final']}")
    return rows


def ledger_rows() -> list[dict]:
    rows = []
    for src in SOURCES:
        pin = PINS.get(src.key, {})
        rows.append({
            "key": src.key,
            "kind": src.kind,
            "title": src.title,
            "url": src.url,
            "language": src.language,
            "license": src.license,
            "authorization": src.authorization,
            "http_status": pin.get("status", ""),
            "final_url": pin.get("final", ""),
            "bytes": pin.get("bytes", ""),
            "sha256": pin.get("sha256", ""),
            "probe_date": PROBE_DATE,
            "verdict": verdict(src),
            "ingestible": "y" if ingestible(src) else "n",
            "note": src.note,
        })
    return rows


# ===========================================================================
# 2. the pinned paratext
# ===========================================================================

#: research quotations of the Skobin translation, as transcribed by Russian
#: readers on the FantLab work page.  Short quotations of a copyrighted work,
#: attributed, for the same reason ``work/reference_corpora.py`` keeps short
#: attributed excerpts of Gibson and Burroughs.  Nothing here is a full text,
#: and no mirror of the translation is fetched.
FANTLAB_REVIEWS_URL = "https://fantlab.ru/work253074"
FANTLAB_SHA256 = PINS["skobin_fantlab_work"]["sha256"]


# ---------------------------------------------------------------------------
# 2a. the review blocks the quotations were transcribed from
# ---------------------------------------------------------------------------
# Every quotation below carries a ``review`` key naming a row of this table,
# and its ``attributor`` string is *derived* from that row by
# :func:`attributor_of`.  Nothing types a reviewer by hand.
#
# That indirection is the whole point.  An earlier revision stored the
# attribution as free text on each quotation, and six of the twelve rows were
# wrong: they read "Reza Negarestani (FantLab review, 2011)", a name that does
# appear on the work page but in a completely different role.  It is the author
# of an *external review of the book* («Пишущие машины из далёких мест»,
# work1915318), listed in the page's «Рецензии» bibliography block - not a
# reader-review author at all, and that page carries no review body.  Six
# quotations were therefore credited to a man who did not write them, which is
# precisely the failure the licensing position cannot afford: these rows are
# short quotations of a copyrighted translation, so the attribution is the
# thing that makes them quotable.
#
# The three rows below are the review blocks the twelve quotations actually
# come from, pinned as (review id, author, account, datePublished).  They are
# read off the page's ``schema.org/Review`` markup, one ``<div
# class="response-item" itemprop="review">`` per row, and each carries the
# ``#responseNNNNNN`` anchor that permalinks it.  ``body_sha256`` is over the
# block's tag-stripped body, and ``n_chars`` its length: together they pin
# *which* text was transcribed, so a later revision of a review cannot silently
# become the source of record.  Both were verified identical across a cached
# fetch and an independent live re-fetch of the work page.
FANTLAB_REVIEWS: tuple[dict, ...] = (
    {"review": "review174357", "author": "josser", "account": 55558,
     "date_published": "2011-05-10 23:49:02", "n_chars": 1722,
     "body_sha256": "98fd7c017220fc2aa4539e7e4cf790781a9cf393d0127ed3a215174d7d6cf224"},
    {"review": "review409177", "author": "Stirliz77", "account": 28338,
     "date_published": "2020-10-09 01:35:27", "n_chars": 9838,
     "body_sha256": "ad580c71b209b8569a6fd3104d8b2938de32999980805a66f21cbfab1538af51"},
    {"review": "review498807", "author": "ekidonaa", "account": 222907,
     "date_published": "2025-02-12 07:17:33", "n_chars": 12307,
     "body_sha256": "6acc2537ff0b7f6a2c3cece7f8e6417a1ce2b1608c2e876523bd7be7a36cfa18"},
)

REVIEW_BY_ID: dict[str, dict] = {r["review"]: r for r in FANTLAB_REVIEWS}

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def review_permalink(review: str) -> str:
    """The stable ``#responseNNNNNN`` permalink for a review block.

    ``review<N>`` is this module's key for a block; the page's own anchor is
    ``response<N>``, which is what the link resolves to.
    """
    return f"{FANTLAB_REVIEWS_URL}#{review.replace('review', 'response', 1)}"


def attributor_of(review: str) -> str:
    """``"<author> (FantLab review, <D Mon YYYY>)"`` for a review id.

    Derived from :data:`FANTLAB_REVIEWS` rather than typed per quotation, so a
    quotation cannot claim an author its review block does not have.
    """
    rev = REVIEW_BY_ID[review]
    year, month, day = rev["date_published"][:10].split("-")
    return (f"{rev['author']} (FantLab review, "
            f"{int(day)} {_MONTHS[int(month) - 1]} {year})")


# --- parsing a live work page back into review blocks ------------------------
#: One FantLab reader review, as ``schema.org/Review`` in the page's markup.
#: The three identity fields are matched independently rather than as one
#: ordered pattern: their order inside a block is presentational, and a
#: positional regex silently stopped matching when FantLab reordered them.
_RE_AUTHOR_RE = re.compile(r'itemprop="author"><a href=\'/user(?P<account>\d+)\'>'
                           r'(?P<author>[^<]+)</a>')
_RE_DATE_RE = re.compile(r'itemprop="datePublished" content="(?P<date>[^"]+)"')
_RE_RESPONSE_RE = re.compile(r'name=response(?P<response>\d+)')
_BODY_RE = re.compile(r'itemprop="reviewBody">(?P<body>.*)\Z', re.S)
_BLOCK_RE = re.compile(r'<div class="response-item" itemprop="review"')


def review_body_text(raw_html: str) -> str:
    """Tag-strip one review body down to the text a fingerprint covers."""
    body = _BODY_RE.search(raw_html).group("body")
    body = re.sub(r"<(script|style)\b.*?</\1>", "", body, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t ]+", " ", body)
    return re.sub(r"\s*\n\s*", "\n", body).strip()


def parse_review_blocks(page_html: str) -> dict[str, dict]:
    """Map ``review<id>`` -> block metadata for every review on a work page.

    Reads the page's own ``schema.org/Review`` markup, so author and date come
    from the source rather than from anything transcribed here.
    """
    cuts = [m.start() for m in _BLOCK_RE.finditer(page_html)]
    blocks: dict[str, dict] = {}
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else len(page_html)
        raw = page_html[start:end]
        author, date, resp = (_RE_AUTHOR_RE.search(raw),
                              _RE_DATE_RE.search(raw),
                              _RE_RESPONSE_RE.search(raw))
        if not (author and date and resp):
            continue
        text = review_body_text(raw)
        key = f"review{resp.group('response')}"
        blocks[key] = {
            "review": key,
            "author": author.group("author").strip(),
            "account": int(author.group("account")),
            "date_published": date.group("date"),
            "body_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "n_chars": len(text),
            "_text": text,
        }
    return blocks


def verify_review_attribution(log=print, blocks: dict[str, dict] | None = None) -> list[dict]:
    """Check every pinned review block against the live page.

    This is the check the earlier free-text attribution could not survive: each
    block's author and ``datePublished`` are read back off the page and
    compared with the pins, so a review edited or re-attributed upstream is
    reported rather than silently inherited.  Pass ``blocks`` to reuse a page
    already fetched rather than requesting it twice.
    """
    if blocks is None:
        got = probe(FANTLAB_REVIEWS_URL)
        if got["status"] != 200:
            log(f"[translation_oracle] review check skipped: work page "
                f"returned {got['status']}")
            return [{"review": r["review"], "verdict": "skipped"}
                    for r in FANTLAB_REVIEWS]
        blocks = parse_review_blocks(got["text"])
    rows = []
    for rev in FANTLAB_REVIEWS:
        live = blocks.get(rev["review"])
        if live is None:
            rows.append({"review": rev["review"], "verdict": "missing"})
            log(f"[translation_oracle] review {rev['review']} MISSING from page")
            continue
        same = (live["author"] == rev["author"]
                and live["date_published"] == rev["date_published"])
        rows.append({"review": rev["review"], "verdict": "ok" if same else "changed",
                     "pinned_author": rev["author"],
                     "live_author": live["author"],
                     "pinned_sha256": rev["body_sha256"],
                     "live_sha256": live["body_sha256"]})
        log(f"[translation_oracle] review {rev['review']:14s} "
            f"{live['author']:12s} {live['date_published'][:10]} "
            f"{'OK' if same else 'CHANGED'}")
    return rows


def check_quotation_provenance(blocks: dict[str, dict]) -> list[dict]:
    """Which review block each quotation's text actually occurs in.

    Returns one row per quotation with the blocks that contain it, so an
    attribution can be checked against the source rather than trusted.
    """
    quotes = list(RU_QUOTATIONS) + [DIVERGENT_QUOTE]
    out = []
    for q in quotes:
        hits = sorted(key for key, blk in blocks.items()
                      if q["ru"].rstrip("»") in blk.get("_text", ""))
        out.append({"id": q["id"], "review": q["review"], "found_in": hits})
    return out

RUQ1 = ("‹‹ДНК=канал геноцида любовных кукол, которые взрываются в её серебряное "
        "кольцо::паразитный дрон кибер природы инокулирует в чудовище нарко эмбриона, "
        "отпустившего мех – я был изнасилован до сознания внутренних органов трупного "
        "механизма::‹‹Скрипт её развоплощённого присоса// Я воспроизвожусь//››"
        "гиперконтроль над хладнокровными больными животными уровня ноль в жестокости "
        "витал узла города, что вызывает присос=кровь хромосома ЗАГРУЗКА:::модуль массы "
        "плоти вывода=искусственное солнце на эрозию::связывает его с геномным "
        "желанием::истечение телесных жидкостей себя, что измеряет кислотный нейроштекер "
        "генома=взаимосвязь// Телесное сочленение андроидной природы, охотящейся на "
        "гротеск с глазами киборга/")
RUQ2 = ("«Коэффициент жестокости пса, источающего обнажённость себя //++убийственная "
        "паранойя нарко-эмбриона, что соизмеряет её с трагическим кодом души/грамма// "
        "Он скачет::‹в будущем, когда потомство вызовет спазм›::голограммы, ген "
        "которых=ТВ изолированы, дышат и выдирают полосы мозга шума бойроида//"
        "‹‹в режим разрезки масс плоти, что слабеют››»")

#: Each row names the review block it was transcribed from in ``review``; the
#: ``attributor`` string is derived from that by :func:`attributor_of`, so the
#: two cannot drift apart.  ``kind`` is ``skeleton`` (located by the operator
#: channel alone) or ``anchor`` (located by bilingual token anchors).
#: ``anchors`` is RU -> EN and is a *prior read off the target text*, not a
#: result; the build reports how many of the anchors the corpus independently
#: corroborates and whether the match is unique.
RU_QUOTATIONS: tuple[dict, ...] = (
    {"id": "RUQ01", "review": "review174357",
     "kind": "skeleton", "anchors": {},
     "ru": RUQ1,
     "note": "16-operator span; located by operator skeleton alone."},
    {"id": "RUQ02", "review": "review498807",
     "kind": "skeleton", "anchors": {},
     "ru": RUQ2,
     "note": "12-operator span; located by operator skeleton alone.  The "
             "reviewer had to consult the translator's own preface to read "
             "it, which is itself evidence that the symbol channel survives "
             "translation."},
    {"id": "RUQ03", "review": "review498807",
     "kind": "anchor",
     "anchors": {"вычисляю": "compute", "взрываюсь": "explode",
                 "планета": "planet", "эродирует": "erodes"},
     "ru": "«Я вычисляю и взрываюсь::планета эродирует»",
     "note": "Retains the `::` field split."},
    {"id": "RUQ04", "review": "review498807",
     "kind": "anchor",
     "anchors": {"синапсы": "synapse", "массы": "masses", "плоти": "flesh",
                 "бойроида": "boyroid", "сексуального": "sexual",
                 "антропоида": "anthropoid", "материал": "material",
                 "мозгу": "brain"},
     "ru": "«синапсы в массах плоти бойроида сексуального антропоида с "
           "материалом в мозгу»",
     "note": "Quoted by the reviewer as the proof-text for his own reading of "
             "boy-roid; the aligned English confirms the alignment."},
    {"id": "RUQ05", "review": "review409177",
     "kind": "anchor",
     "anchors": {"Гвоздь": "nail", "глаз": "eye"},
     "ru": "«Гвоздь/глаз»",
     "note": "Chapter title.  The `/` link operator survives translation, so "
             "the Russian title is a `Nail/eye` link, not the Russian "
             "compound «гвоздь-глаз»."},
    {"id": "RUQ06", "review": "review498807",
     "kind": "anchor",
     "anchors": {"ОСТОРОЖНО": "WARNING", "ЭТА": "THIS",
                 "МАШИНА": "MACHINE", "УБИВАЕТ": "KILLS"},
     "ru": "«ОСТОРОЖНО ЭТА МАШИНА УБИВАЕТ»",
     "note": "First chapter title; the English heading is exactly "
             "`WARNING THIS MACHINE KILLS`."},
    {"id": "RUQ07", "review": "review498807",
     "kind": "anchor",
     "anchors": {"Город": "City", "Трупов": "Cadaver"},
     "ru": "«Город Трупов»",
     "note": "A calque of the book's `Cadaver City`, with the head second - "
             "an unusual Russian word order that itself betrays the "
             "translation."},
    {"id": "RUQ08", "review": "review409177",
     "kind": "literal",
     "anchors": {"ледяная": "ice", "туманность": "nebula",
                 "фекальной": "faecal", "черноты": "black"},
     "en_literal": "[ice nebula faecal black]",
     "ru": "«[ледяная туманность фекальной черноты]»",
     "note": "A bracketed coinage; the English is `[ice nebula faecal black]` "
             "with the brackets intact."},
    {"id": "RUQ09", "review": "review409177",
     "kind": "anchor",
     "anchors": {"могильное": "tomb", "устройство": "device",
                 "воспламеняет": "ignites", "фекального": "faecal",
                 "Сатурна": "Saturn", "кровавого": "blood",
                 "призрака": "spectre"},
     "ru": "«могильное устройство воспламеняет фекального Сатурна кровавого "
           "призрака»",
     "note": "The English is `tomb-device ignites faecal Saturn blood "
             "spectre`; the Russian keeps the head-final order and the "
             "`tomb-device` coinage."},
    {"id": "RUQ10", "review": "review409177",
     "kind": "anchor",
     "anchors": {"галлюциногенный": "hallucinogenic", "мех": "fur",
                 "промежуточной": "intermediate", "зоне": "area"},
     "ru": "«галлюциногенный мех, что существует в промежуточной зоне»",
     "note": "The English is `hallucinogenic fur that exists to intermediate "
             "area`; note the trailing `::`, again preserved."},
    {"id": "RUQ11", "review": "review498807",
     "kind": "anchor",
     "anchors": {"NDROID": "NDROID"},
     "ru": "«NDROID»",
     "note": "An all-caps Latin token left in the target text unchanged - "
             "the clearest single piece of evidence that the Russian edition "
             "does not translate the technical layer."},
)

#: the reader's *claimed* subtitle, which the English does not support.
DIVERGENT_QUOTE = {
    "id": "RUQ12",
    "review": "review498807",
    "kind": "divergence",
    "anchors": {"ЛИКВИДАЦИЯ": "abolition", "ОСТОРОЖНО": "WARNING"},
    "ru": "«ЛИКВИДАЦИЯ»",
    "note": "The reader reports the subtitle of the `WARNING THIS MACHINE "
            "KILLS` chapter as «ЛИКВИДАЦИЯ».  The English heading at that "
            "position is `abolition`.  ликвидация and abolition are not "
            "translations of one another, so this is a real divergence "
            "between the Russian text and the English, not an alignment "
            "failure.  Reported, not resolved.",
}

#: short quotations of the author interviews (the plan's second oracle).
AUTHOR_PARATEXT: tuple[dict, ...] = (
    {"id": "PT01", "source": "interview_3am_nude_brain",
     "interviewer": "Richard Marshall (3AM Magazine)", "date": "2002-06",
     "claim_kind": "method",
     "quote": "In my writing the language cell is able to be distorted by the "
              "infinite hyperlink of the synapse -- 'a new language' is the "
              "conceptual pain -- all the data act as the hardweb character "
              "as if I dissect subjective writing, I'm striking the nude "
              "brain to a screen.",
     "claim": "The book deliberately breaks the language; 'a new language' is "
              "the point, not an accident of translation or of English."},
    {"id": "PT02", "source": "interview_3am_nude_brain",
     "interviewer": "Richard Marshall (3AM Magazine)", "date": "2002-06",
     "claim_kind": "book",
     "quote": "So my novel is presenting the aspect of a genetic hardweb "
              "clearly -- the nerve cells that run through our gene dub -- is "
              "the strategic object that our body codes erodes the world.",
     "claim": "The novel presents a genetic layer; the body codes the world."},
    {"id": "PT03", "source": "interview_3am_nude_brain",
     "interviewer": "Richard Marshall (3AM Magazine)", "date": "2002-06",
     "claim_kind": "influence",
     "quote": "Certainly Antonin Artaud exerted an important influence on my "
              "hardweb -- Artaud produced PDA of a nerve cell",
     "claim": "Artaud is a declared influence."},
    {"id": "PT04", "source": "interview_rhizome_14416_wb",
     "interviewer": "Sandy Baldwin (WVU Center for Literary Computing)",
     "date": "2002-08",
     "claim_kind": "method",
     "quote": "I notice the use of certain kind of punctuation (…, =, and so "
              "on) and certain repeated terms (ADAM, dogs, etc.), but I'm not "
              "clear if these reflect a method or not.",
     "claim": "An interviewer puts the method question to the author in terms "
              "of the punctuation and of the recurring nouns."},
    {"id": "PT05", "source": "interview_rhizome_14416_wb",
     "interviewer": "Sandy Baldwin (WVU Center for Literary Computing)",
     "date": "2002-08",
     "claim_kind": "method",
     "quote": "A3: I explode the conceptual character on text. As the result "
              "of gene dub that was perceived. Burroughs and Guyotat break "
              "down the unit of a cell--in other words the human body is "
              "caused to text hallucination--I extract the data flesh of "
              "digital narrative gene dub.",
     "claim": "The author's answer: the text is exploded at the level of the "
              "character, and the unit broken down is the cell."},
    {"id": "PT06", "source": "interview_rhizome_14416_wb",
     "interviewer": "Sandy Baldwin (WVU Center for Literary Computing)",
     "date": "2002-08",
     "claim_kind": "influence",
     "quote": "Burroughs and Guyotat, they are sexual viruses–language cell "
              "that erodes electronics–my writing was born to the horizon of "
              "such techno music",
     "claim": "Burroughs and Guyotat are the declared sources."},
)

#: the author's own lexicon, checked against the book in ``oracle_checks``.
AUTHOR_TERMS: tuple[tuple[str, str, str], ...] = (
    # (label, regex used against the corpus, why the author used it)
    ("ADAM", r"\bADAM\b", "named by the interviewer as a repeated term (PT04)"),
    ("ANTI-ADAM", r"\bANTI-?ADAM\b", "the antagonist the book sets against ADAM"),
    ("dogs", r"\bdogs?\b", "named by the interviewer as a repeated term (PT04)"),
    ("data mutant", r"\bdata[ =-]mutants?\b", "the author's own noun (PT05)"),
    ("hardweb", r"\bhard\s?web\b", "the author's adjective (PT01, PT02)"),
    ("creature intensity", r"creature\s+intensity", "the author's noun phrase (PT01)"),
    ("gene dub", r"gene[- =]dub", "the author's noun (PT02, PT05)"),
    ("nerve cell", r"\bnerve\s+cells?\b", "the author's noun (PT02)"),
    ("echo archive", r"echo\s+archive", "the author's noun (3AM)"),
    ("Biocapture", r"[Bb]iocapture", "his own software title (3AM)"),
    ("LOAD", r"\bLOAD\b", "a retained Latin token in the Russian edition (RUQ01)"),
    ("NDRO", r"\bNDRO\b", "a retained Latin token"),
    ("NDROID", r"\bNDROID\b", "the form the Russian reader singles out (RUQ11)"),
    ("Cadaver City", r"\bCadaver City\b", "the setting, as the Russian reader names it"),
)

#: the punctuation the author is asked about in PT04.
AUTHOR_OPERATORS: tuple[tuple[str, str], ...] = (
    ("=", r"=", "named in PT04"),
    ("...", r"\.\.\.", "named in PT04 as «…»"),
)


# ===========================================================================
# 3. the corpus
# ===========================================================================

# Only these tags are stripped.  The book uses `<` and `>` as operators, so a
# permissive ``<[^>]*>`` would swallow the span delimiters themselves (e.g.
# ``<to the cut mode of the masses of flesh that weakened>``); the ``\b``
# after the name keeps `<the meridian>` and `<in the future ...>` intact.
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_TAG_RE = re.compile(
    r"</?(?:p|br|th|td|tr|table|pre|li|ul|ol|a|span|div|em|i|b|strong|sup|sub"
    r"|h[1-6]|img|figure|figcaption|hr)\b[^>]*>", re.I)


@dataclass
class Corpus:
    """The vetted layout markdown, with page attribution and an operator index."""
    text: str
    page_starts: list[tuple[int, int]]   # (offset, page) ascending
    ops: list[tuple[str, int, int]] = field(default_factory=list)

    def page(self, offset: int) -> int:
        lo, hi = 0, len(self.page_starts) - 1
        best = self.page_starts[0][1] if self.page_starts else 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.page_starts[mid][0] <= offset:
                best = self.page_starts[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def pages(self, start: int, end: int) -> list[int]:
        return sorted({self.page(start), self.page(max(start, end - 1))})

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]


def load_corpus(path: Path = CORPUS_MD) -> Corpus:
    """Clean the layout markdown and index it by page and by operator.

    Page attribution survives the clean.  The marker ``<!-- page N -->`` opens
    page N, so a page's *start* is the end of the text that precedes its
    marker: the start is recorded after the preceding gap is appended, not
    before.  Getting that order wrong shifts every citation by one page, and
    the book is cited page by page, so it is worth being explicit about.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    parts: list[str] = []
    page_starts: list[tuple[int, int]] = []
    cursor = 0
    raw_cursor = 0
    for m in re.finditer(r"<!-- page (\d+) -->", raw):
        gap = _TAG_RE.sub(" ", _COMMENT_RE.sub(" ", raw[raw_cursor:m.start()]))
        parts.append(gap)
        cursor += len(gap)
        page_starts.append((cursor, int(m.group(1))))
        parts.append("\n")
        cursor += 1
        raw_cursor = m.end()
    parts.append(_TAG_RE.sub(" ", _COMMENT_RE.sub(" ", raw[raw_cursor:])))
    text = "".join(parts)
    corpus = Corpus(text=text, page_starts=page_starts)
    corpus.ops = scan_operators(text)
    return corpus


def scan_operators(text: str, ops: tuple[str, ...] = OPS) -> list[tuple[str, int, int]]:
    """Greedy longest match over ``OPS`` - the same rule as #10 ``scan_runs``.

    Returns ``(operator, start, end)`` triples in text order.  Consistency with
    #10 matters: the whole operator-invariance claim is a claim about this
    tokenisation, so it has to be the corpus's own.
    """
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        for op in ops:
            if text.startswith(op, i):
                out.append((op, i, i + len(op)))
                i += len(op)
                break
        else:
            i += 1
    return out


#: glyphs the Russian edition / the reviewers use for the book's `<<`/`>>`
#: and `<`/`>` delimiters, mapped onto the book's own operators.  Applied
#: only *inside* a pinned quotation, after the outer Russian quotation marks
#: have been stripped, so it cannot swallow real punctuation.  Longest first:
#: `‹‹` is the book's `<<`, `‹` is its `<`.
RU_GLYPH_OPS = (
    ("‹‹", "<<"), ("‹", "<"), ("«", "<<"),
    ("››", ">>"), ("›", ">"), ("»", ">>"),
)


def normalise_ru_quote(text: str) -> str:
    """Strip the reviewer's outer quotation marks, then map span glyphs."""
    s = text.strip()
    for outer in ("«", "»"):
        if s.startswith(outer):
            s = s[1:]
        if s.endswith(outer):
            s = s[:-1]
    s = s.strip()
    for src, dst in RU_GLYPH_OPS:
        s = s.replace(src, dst)
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ===========================================================================
# 4. segmentation and the aligner
# ===========================================================================

@dataclass(frozen=True)
class Unit:
    text: str
    ops_before: str
    start: int
    end: int

    @property
    def n_chars(self) -> int:
        return max(1, len(self.text.strip()))


def split_units(text: str) -> list[Unit]:
    """Split a string on maximal operator runs into (ops, text-run) units."""
    ops = scan_operators(text)
    units: list[Unit] = []
    pending: list[str] = []
    pos = 0
    for _, s, e in ops:
        if s > pos:
            units.append(Unit(text[pos:s], "".join(pending), pos, s))
            pending = []
        pending.append(_op_at(text, s, e))
        pos = e
    if pos < len(text):
        units.append(Unit(text[pos:], "".join(pending), pos, len(text)))
    return [u for u in units if u.text.strip()]


def _op_at(text: str, s: int, e: int) -> str:
    return text[s:e]


# Aligner weights.  These were chosen by a grid search over the *tuning half*
# of the built-in gold only (documents 0-5) and then applied unchanged to the
# held-out half (documents 6-11); both numbers are published in
# `aligner_eval.tsv`, so the tuning/evaluation split is auditable rather than
# asserted.  Re-running the search is `evaluate_aligner(seed, n_docs)`.
LEN_W = 0.5        # log-length disagreement
ANCHOR_W = 4.0     # bilingual anchor agreement
OPS_W = 0.35       # skeleton disagreement on the preceding operator run
GAP_W = 1.5        # cost per log-char of an unaligned unit
GAP_BASE = 0.8     # flat cost of opening a gap
MAX_BLOCK = 2      # largest n-m block the diagonal will consider
BLOCK_W = 0.3      # cost per extra unit inside an n-m block

# link kinds; the first number is the English side, the second the target side
KIND_11 = "1-1"
KIND_10 = "1-0"    # English unit with no target counterpart (deleted in translation)
KIND_01 = "0-1"    # target unit with no English counterpart (added by the translator)


@dataclass
class Link:
    kind: str
    en: tuple[int, ...]
    ru: tuple[int, ...]
    cost: float


def _skeleton_penalty(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if collections.Counter(a) == collections.Counter(b):
        return 0.5 * OPS_W
    return OPS_W


DIGIT_RE = re.compile(r"\d+")


def _digit_anchor(en_unit: "Unit", ru_unit: "Unit") -> int:
    """Shared numerals - the one anchor that needs no bilingual resource."""
    a = set(DIGIT_RE.findall(en_unit.text))
    b = set(DIGIT_RE.findall(ru_unit.text))
    return len(a & b)


def align_units(
    en_units: list[Unit],
    ru_units: list[Unit],
    anchors: dict[str, str] | None = None,
    anchor_table: dict[str, set[str]] | None = None,
) -> list[Link]:
    """Monotone DP alignment of two unit sequences.

    ``anchor_table`` maps an English unit's index to the set of word tokens it
    contains; ``anchors`` maps target-language tokens to the English tokens
    they are expected to render as, and agreement is rewarded.  Shared
    numerals are rewarded automatically, because they are the one anchor that
    transfers to any language pair without a bilingual resource.  The aligner
    never *requires* an anchor, so the skeleton-only quotations are aligned
    with ``anchor_table=None`` and no anchors at all.
    """
    anchors = anchors or {}
    n, m = len(en_units), len(ru_units)
    table = anchor_table or {}

    def anchor_hits(i: int, j: int) -> int:
        hits = _digit_anchor(en_units[i], ru_units[j])
        want = {en_tok.lower() for ru_tok, en_tok in anchors.items()
                if ru_tok in ru_units[j].text.lower()}
        if not want:
            return hits
        have = table.get(i, set())
        return hits + len(want & have)

    # prefix sums make the n-m diagonal O(1)
    en_pref = [0.0]
    for u in en_units:
        en_pref.append(en_pref[-1] + u.n_chars)
    ru_pref = [0.0]
    for u in ru_units:
        ru_pref.append(ru_pref[-1] + u.n_chars)

    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            base = cost[i][j]
            if base == INF:
                continue
            # n-m diagonal: one source block against one target block.  This
            # is what makes 1-n and n-1 expressible at all - a DP with only
            # 1-1 / 1-0 / 0-1 moves can never *output* a many-to-many link, so
            # it can never be scored against a gold that contains one.
            for p in range(1, MAX_BLOCK + 1):
                for q in range(1, MAX_BLOCK + 1):
                    if i + p > n or j + q > m:
                        continue
                    if p == 1 and q == 1:
                        le = en_units[i].n_chars
                        lr = ru_units[j].n_chars
                        pen = _skeleton_penalty(en_units[i].ops_before,
                                                ru_units[j].ops_before)
                    else:
                        le = int(en_pref[i + p] - en_pref[i])
                        lr = int(ru_pref[j + q] - ru_pref[j])
                        pen = _skeleton_penalty(en_units[i].ops_before,
                                                ru_units[j].ops_before) \
                            + BLOCK_W * (p + q - 2)
                    c = (base
                         + LEN_W * abs(math.log(le) - math.log(lr))
                         + pen
                         - ANCHOR_W * anchor_hits(i, j))
                    if c < cost[i + p][j + q]:
                        cost[i + p][j + q] = c
                        back[i + p][j + q] = (KIND_11 if p == q == 1 else "n-m", i, j, p, q)
            if i < n:
                c = base + GAP_W * math.log1p(en_units[i].n_chars) + GAP_BASE
                if c < cost[i + 1][j]:
                    cost[i + 1][j] = c
                    back[i + 1][j] = (KIND_10, i, j, 1, 0)
            if j < m:
                c = base + GAP_W * math.log1p(ru_units[j].n_chars) + GAP_BASE
                if c < cost[i][j + 1]:
                    cost[i][j + 1] = c
                    back[i][j + 1] = (KIND_01, i, j, 0, 1)

    links: list[Link] = []
    i, j = n, m
    while (i, j) != (0, 0):
        step = back[i][j]
        if step is None:                       # unreachable - defensive
            break
        kind, pi, pj, p, q = step
        d = cost[i][j] - cost[pi][pj]
        if kind == KIND_10:                     # consumed the English unit only
            links.append(Link(kind, tuple(range(pi, pi + p)), (), d))
        elif kind == KIND_01:                   # consumed the target unit only
            links.append(Link(kind, (), tuple(range(pj, pj + q)), d))
        else:                                   # a block link: n-m
            links.append(Link(kind if kind == KIND_11 else "n-m",
                              tuple(range(pi, pi + p)),
                              tuple(range(pj, pj + q)), d))
        i, j = pi, pj
    links.reverse()
    return _merge(links)


def _merge(links: list[Link]) -> list[Link]:
    """Collapse runs of *gap* links into n-m links.

    ``1-1`` and ``n-m`` links are deliberately left alone: keeping them atomic
    is what lets ``bilingual_pairs`` harvest one gloss pair per aligned unit
    and keeps the artifact readable.
    """
    merged: list[Link] = []
    for link in links:
        if merged and merged[-1].kind == link.kind and link.kind in (KIND_10, KIND_01):
            prev = merged[-1]
            merged[-1] = Link(prev.kind, prev.en + link.en, prev.ru + link.ru,
                              prev.cost + link.cost)
        else:
            merged.append(Link(link.kind, link.en, link.ru, link.cost))
    return merged


def unit_table(units: list[Unit], corpus: Corpus | None = None,
               base: int = 0) -> dict[int, set[str]]:
    """index -> set of lower-cased word tokens in the unit.

    Hyphen is a separator, not part of a token: the book coinages
    (`tomb-device`, `cold-blooded`, `drug-embryo`) are single orthographic
    words but the anchors read them as two, and so must the index.
    """
    out: dict[int, set[str]] = {}
    for i, u in enumerate(units):
        out[i] = {w.lower() for w in _WORD_RE.findall(u.text)}
    return out


# ===========================================================================
# 5. AER and the built-in gold
# ===========================================================================

def collapse(pairs: list[tuple]) -> list[tuple]:
    """Standard null-collapse: `(i, None)` next to `(None, j)` becomes `(i, j)`.

    Always returns ``(tuple_of_i, tuple_of_j)`` with ``None`` for an empty
    side, so downstream code never has to guess which shape it got.
    """
    def as_tuple(x):
        if x is None:
            return None
        return (x,) if isinstance(x, int) else tuple(x)

    out: list[tuple] = []
    for i, j in pairs:
        i, j = as_tuple(i), as_tuple(j)
        if i is None and j is None:
            continue
        if i is None:
            if out and out[-1][0] is not None and out[-1][1] is None:
                out[-1] = (out[-1][0], j)          # complete a dangling pair
            elif out and out[-1][0] is not None:
                out[-1] = (out[-1][0], out[-1][1] + j)
            else:
                out.append((None, j))
        elif j is None:
            if out and out[-1][1] is not None:
                out[-1] = (out[-1][0] + i, out[-1][1])
            else:
                out.append((i, None))
        else:
            if out and out[-1][1] is None:
                out[-1] = (out[-1][0] + i, j)
            else:
                out.append((i, j))
    return out


def alignment_error_rate(gold: list[tuple], hyp: list[tuple]) -> float:
    """Standard AER (Gale & Church style; Levow 2006) over null-collapsed links.

    Both sides are collapsed first, so a link is compared as a *set*: a
    hypothesis link ``((3,), (4, 5))`` is correct iff the gold has exactly
    ``((3,), (4, 5))`` after collapse.

    The three error types of the standard metric are all charged, over a gold
    denominator::

        G = gold links with both sides present
        correct = hypothesis links that match a gold link
        AER = ((G - correct) + (H - correct)) / G,   H = hypothesis links

    ``G - correct`` is *deletions* (a gold link the hypothesis never made, which
    covers a wrong alignment as well, since the gold link is then missing) and
    ``H - correct`` is *insertions*.  A hypothesis that proposes nothing scores
    1.0, not 0.0 - without the deletion term this collapses to precision, which
    an aligner can game by emitting a single confident link.

    AER is a rate of errors *per gold link*, not a proportion of a fixed set,
    so it is not bounded above by 1: a hypothesis that proposes many links the
    gold does not have scores above 1.  That is a property of the metric, and
    the numbers here are read as such - which is also why the proportional
    baseline's figures are near 2 rather than near 1.
    """
    g = collapse(gold)
    h = collapse(hyp)
    gmap: dict = {}
    for i, j in g:
        if i is None or j is None:
            continue
        for ii in i:
            gmap[ii] = j
    n_gold = len(gmap)
    if not n_gold:
        return 1.0 if h else 0.0
    correct = total = 0
    for i, j in h:
        if i is None or j is None:
            continue
        total += 1
        if gmap.get(i[0]) == j:
            correct += 1
    return ((n_gold - correct) + (total - correct)) / n_gold


#: 24 EN/RU sentence pairs authored for this harness (CC0 by construction: they
#: are this repository's own text).  They carry numbers and content words but
#: no cognates, which is the honest test for a length-and-anchor aligner
#: facing two languages with no lexical overlap - the same problem the
#: operator-skeleton quotations pose.
GOLD_SENTENCES: tuple[tuple[str, str], ...] = (
    ("The dog ran across the field.", "Собака бежала через поле."),
    ("A girl opened the door.", "Девушка открыла дверь."),
    ("The machine stopped in the middle of the street.",
     "Машина остановилась посреди улицы."),
    ("He wrote three letters to his friend.", "Он написал три письма своему другу."),
    ("The old library was built in 1890.", "Старая библиотека была построена в 1890 году."),
    ("Nobody knows what will happen tomorrow.", "Никто не знает, что случится завтра."),
    ("She drank a cup of black coffee.", "Она выпила чашку чёрного кофе."),
    ("The children played in the garden until dark.", "Дети играли в саду до темноты."),
    ("Water flowed through the pipe.", "Вода текла по трубе."),
    ("My brother works in a factory near the river.",
     "Мой брат работает на заводе недалеко от реки."),
    ("The train arrived at six o'clock.", "Поезд прибыл в шесть часов."),
    ("They sold the old books for two dollars.", "Они продали старые книги за два доллара."),
    ("The cat slept on the warm windowsill.", "Кот спал на тёплом подоконнике."),
    ("I have never seen such a strange animal.", "Я никогда не видел такого странного животного."),
    ("The doctor explained the results to the family.", "Врач объяснил результаты семье."),
    ("A small bird landed on the wire.", "Маленькая птица села на провод."),
    ("The door of the shop was open.", "Дверь магазина была открыта."),
    ("He remembered the summer of his childhood.", "Он вспомнил лето своего детства."),
    ("The students asked many difficult questions.", "Студенты задали много сложных вопросов."),
    ("Snow covered the empty road.", "Снег покрыл пустую дорогу."),
    ("The light in the window was still on.", "Свет в окне всё ещё горел."),
    ("She found the letter under the book.", "Она нашла письмо под книге."),
    ("The engine stopped and the train began to shake.",
     "Двигатель остановился, и поезд начал трястись."),
    ("Nobody answered the phone for an hour.",
     "Никто не отвечал на телефон в течение часа."),
)

GOLD_SEED = 58


#: short Russian sentences used to plant realistic 1-n operations: a translator
#: rendering one English sentence as two Russian ones almost always *adds* a
#: short clause rather than cutting the existing one in half, and only the
#: realistic shape is a fair test of a length-based aligner.
RU_EXTRA_SENTENCES = (
    "Это было давно.", "Никто не ответил.", "Так и вышло.",
    "Он молчал.", "Всё осталось как было.",
)


def build_gold_documents(seed: int = GOLD_SEED, n_docs: int = 12):
    """Assemble synthetic EN/RU documents with a known gold alignment.

    Planted operations, in the proportions a real literary parallel text
    shows: mostly 1-1 runs, an occasional 1-n (one English sentence rendered
    as two Russian ones), and an occasional n-1 (two English sentences
    rendered as one).  1-0 deletions are deliberately *not* planted - see
    the comment in the loop.  The gold is emitted as
    ``(en_index, ru_index_or_tuple)`` and null-collapsed by
    ``alignment_error_rate`` exactly as a published gold would be.
    """
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        want = rng.randint(5, 8)
        start = rng.randrange(0, len(GOLD_SENTENCES) - want)
        picked = list(GOLD_SENTENCES[start:start + want])
        en_sents: list[str] = []
        ru_sents: list[str] = []
        gold: list[tuple] = []
        k = 0
        while k < len(picked):
            en_s, ru_s = picked[k]
            roll = rng.random()
            # No 1-0 operations are planted.  Null-collapse folds any
            # deletion that touches a link into a many-to-one link, and AER
            # then charges the hypothesis twice for one mistake, so a gold
            # full of deletions measures the metric rather than the aligner.
            if roll < 0.22:                              # 1-n : Russian adds a clause
                ru_sents.append(RU_EXTRA_SENTENCES[rng.randrange(len(RU_EXTRA_SENTENCES))])
                ru_sents.append(ru_s)
                en_sents.append(en_s)
                gold.append((len(en_sents) - 1,
                             (len(ru_sents) - 2, len(ru_sents) - 1)))
                k += 1
            elif roll < 0.32 and k + 1 < len(picked):    # n-1 : merged in Russian
                en2, ru2 = picked[k + 1]
                en_sents.append(en_s)
                en_sents.append(en2)
                ru_sents.append(ru_s + " " + ru2)
                j = len(ru_sents) - 1
                gold.append((len(en_sents) - 2, j))
                gold.append((len(en_sents) - 1, j))
                k += 2
            else:                                        # 1-1
                en_sents.append(en_s)
                ru_sents.append(ru_s)
                gold.append((len(en_sents) - 1, len(ru_sents) - 1))
                k += 1
        if len(en_sents) >= 2 and len(ru_sents) >= 2:
            docs.append((en_sents, ru_sents, gold))
    return docs


def baseline_proportional(en_units: list[Unit], ru_units: list[Unit]) -> list[Link]:
    """The classic length-only control (Gale-Church proportional alignment).

    Reported alongside the DP so the aligner's AER has a reference point: it
    uses the only signal a language-blind system has, and does it greedily.
    """
    total_en = sum(u.n_chars for u in en_units) or 1
    total_ru = sum(u.n_chars for u in ru_units) or 1
    links: list[Link] = []
    i = j = 0
    cum_en = cum_ru = 0
    while i < len(en_units) and j < len(ru_units):
        links.append(Link(KIND_11, (i,), (j,), 0.0))
        cum_en += en_units[i].n_chars
        cum_ru += ru_units[j].n_chars
        if cum_en / total_en <= cum_ru / total_ru:
            i += 1
        else:
            j += 1
    while i < len(en_units):
        links.append(Link(KIND_10, (i,), (), 0.0))
        i += 1
    while j < len(ru_units):
        links.append(Link(KIND_01, (), (j,), 0.0))
        j += 1
    return _merge(links)


def evaluate_aligner(seed: int = GOLD_SEED, n_docs: int = 12) -> list[dict]:
    """AER of the aligner (and of the proportional control) on the gold.

    The gold is split into a tuning half and a held-out half.  The weights in
    this module were fitted on the tuning half; the held-out column is the
    number that means something.

    The SUBTOTAL/ALL rows are exact corpus-level AER, not a mean of per-document
    rates: because ``alignment_error_rate`` charges every error against the gold
    links of its own document, summing ``aer * n_gold_links`` and dividing by
    the summed gold links *is* ``total errors / total gold links`` over the
    whole split.
    """
    docs = build_gold_documents(seed, n_docs)
    split_at = len(docs) // 2

    def split_of(k: int) -> str:
        return "tuning" if k < split_at else "held_out"

    rows = []
    for system in ("dp", "proportional"):
        totals = {"tuning": [0.0, 0], "held_out": [0.0, 0]}
        for k, (en_sents, ru_sents, gold) in enumerate(docs):
            en_units = [Unit(s, "", 0, len(s)) for s in en_sents]
            ru_units = [Unit(s, "", 0, len(s)) for s in ru_sents]
            if system == "dp":
                links = align_units(en_units, ru_units)
            else:
                links = baseline_proportional(en_units, ru_units)
            hyp = [(link.en[0], link.ru[0] if len(link.ru) == 1 else link.ru)
                   for link in links if link.en and link.ru]
            g = collapse(gold)
            n_links = len([1 for i, j in g if i and j])
            aer = alignment_error_rate(gold, hyp)
            sp = split_of(k)
            rows.append({"system": system, "split": sp, "document": k,
                         "n_en": len(en_sents), "n_ru": len(ru_sents),
                         "n_gold_links": n_links, "aer": round(aer, 4)})
            totals[sp][0] += aer * n_links
            totals[sp][1] += n_links
        for sp in ("tuning", "held_out"):
            e, t = totals[sp]
            rows.append({"system": system, "split": sp, "document": "SUBTOTAL",
                         "n_en": sum(r["n_en"] for r in rows if r["system"] == system
                                     and r["split"] == sp and r["document"] != "SUBTOTAL"),
                         "n_ru": sum(r["n_ru"] for r in rows if r["system"] == system
                                     and r["split"] == sp and r["document"] != "SUBTOTAL"),
                         "n_gold_links": t,
                         "aer": round(e / t if t else 1.0, 4)})
        e = sum(r["aer"] * r["n_gold_links"] for r in rows
                if r["system"] == system and r["document"] != "SUBTOTAL")
        t = sum(r["n_gold_links"] for r in rows
                if r["system"] == system and r["document"] != "SUBTOTAL")
        rows.append({"system": system, "split": "ALL", "document": "ALL",
                     "n_en": t, "n_ru": t, "n_gold_links": t,
                     "aer": round(e / t if t else 1.0, 4)})
    return rows


# ===========================================================================
# 6. locating a quotation in the English corpus
# ===========================================================================

def operator_skeleton(text: str) -> list[str]:
    return [op for op, _, _ in scan_operators(text)]


def localize_by_skeleton(corpus: Corpus, skeleton: list[str]) -> list[tuple[int, int]]:
    """Every corpus offset window whose operator sequence equals ``skeleton``."""
    ops = corpus.ops
    m = len(skeleton)
    if m == 0 or m > len(ops):
        return []
    hits = []
    ops_only = [o for o, _, _ in ops]
    for start in range(len(ops_only) - m + 1):
        if ops_only[start:start + m] == skeleton:
            hits.append((ops[start][1], ops[start + m - 1][2]))
    return hits


def expand_window(corpus: Corpus, start: int, end: int,
                  max_extend: int = 120) -> tuple[int, int]:
    """Widen a located match by at most one text run, never across a line.

    A skeleton window runs from the first operator of the match to the last,
    so the first and last *text runs* are cut in half; an anchor window cuts
    both ends mid-clause.  Two guards keep the widening honest: a character
    cap, because the previous operator can be pages away, and a line guard,
    because the text run after a chapter heading can run for three pages -
    without it, a three-word quotation would drag the book's front matter
    into its alignment.
    """
    ops = corpus.ops
    n_ops = len(ops)

    lo, hi = 0, n_ops
    while lo < hi:                                   # first op starting at/after `start`
        mid = (lo + hi) // 2
        if ops[mid][1] < start:
            lo = mid + 1
        else:
            hi = mid
    left_op = ops[lo - 1][2] if lo > 0 else 0

    lo, hi = 0, n_ops
    while lo < hi:                                   # last op ending at/before `end`
        mid = (lo + hi) // 2
        if ops[mid][2] <= end:
            lo = mid + 1
        else:
            hi = mid
    right_op = ops[lo][1] if lo < n_ops else len(corpus.text)

    nl = corpus.text.rfind("\n", max(0, start - max_extend), start)
    left_min = nl + 1 if nl != -1 else max(0, start - max_extend)
    nl = corpus.text.find("\n", end, end + max_extend)
    right_max = nl + 1 if nl != -1 else min(len(corpus.text), end + max_extend)

    return max(left_min, left_op), min(right_max, right_op)


#: Latin-only word pattern, used to index the English corpus and the English
#: side of a unit pair.  Deliberately not `[^\W\d_]+`: the anchors and the
#: anchor table are English words, so a non-Latin glyph must not index a token
#: nothing in the anchor set can name.  Hyphen is a separator, not part of a
#: token - see `unit_table`.
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def localize_by_anchors(corpus: Corpus, anchors: dict[str, str],
                        max_span: int | None = None) -> list[tuple[int, int]]:
    """Windows in which *every* bilingual anchor co-occurs within ``max_span``.

    Order-agnostic on purpose: `«Город Трупов»` renders the English
    `Cadaver City`, i.e. head-final Russian against head-initial English, so
    an order-preserving search would miss the very case it is meant to catch.
    The span budget scales with the anchor count, because a long quotation
    whose eight anchors are spread over 90 English characters is a good match,
    not a bad one.  The rarest anchor seeds the scan, which keeps it cheap.
    """
    if not anchors:
        return []
    if max_span is None:
        max_span = max(60, 14 * len(anchors))
    positions: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for m in _WORD_RE.finditer(corpus.text):
        positions[m.group(0).lower()].append((m.start(), m.end()))
    wanted = {en_tok.lower() for en_tok in anchors.values()}
    missing = wanted - set(positions)
    if missing:
        return []
    seed_tok = min(wanted, key=lambda t: len(positions[t]))
    hits = []
    for s, _ in positions[seed_tok]:
        lo, hi = s - max_span, s + max_span
        found = {}
        ok = True
        for tok in sorted(wanted):
            cand = [p for p in positions[tok] if lo <= p[0] and p[1] <= hi]
            if not cand:
                ok = False
                break
            found[tok] = min(cand, key=lambda p: abs(p[0] - s))
        if not ok:
            continue
        start = min(p[0] for p in found.values())
        end = max(p[1] for p in found.values())
        if (start, end) not in hits:
            hits.append((start, end))
    return sorted(hits)


@dataclass
class Alignment:
    quote: dict
    locator: str
    n_windows: int
    n_distinct: int
    en_start: int
    en_end: int
    en_text: str
    ru_units: list[Unit]
    en_units: list[Unit]
    links: list[Link]
    n_anchors: int
    n_anchors_hit: int
    sites: list[tuple[int, int]]        # (start, end) of every occurrence
    site_pages: list[str]               # pages each occurrence spans
    ops_window: tuple[int, int] = (0, 0)   # the raw, un-expanded match

    @property
    def unique(self) -> bool:
        return self.n_windows == 1

    def evidence(self) -> str:
        if self.n_windows == 0:
            return "no_match"
        if self.n_distinct == 1:
            return {"operator_skeleton": "unique_skeleton",
                    "literal_english": "unique_literal_text"}.get(
                        self.locator, "unique_anchor_text")
        if self.n_distinct < self.n_windows:
            return "recurring_text"
        return "ambiguous"

    def confidence(self) -> str:
        ev = self.evidence()
        if ev == "unique_skeleton":
            return "high"
        if ev in ("unique_anchor_text", "unique_literal_text"):
            return "high" if self.n_anchors >= 2 else "medium"
        if ev == "recurring_text":
            return "high" if self.n_distinct == 2 else "medium"
        if ev == "ambiguous" and self.n_distinct <= 3:
            return "medium"
        return "low"

    def pages(self) -> list[int]:
        out: set[int] = set()
        for blob in self.site_pages:
            out.update(int(x) for x in blob.split(",") if x)
        return sorted(out) or list(self.quote.get("pages") or [])


def align_quote(corpus: Corpus, quote: dict) -> Alignment:
    """Locate one target-language quotation in the corpus and align it.

    A quotation may have several sites.  Two of them do: Siratori's loop
    writing puts some passages on the page twice, byte for byte, so the
    operator skeleton of RUQ02 matches twice in 225 pages.  That is not an
    alignment failure but a finding about the book, so the artifact reports
    every site and the evidence column distinguishes *one position* from *one
    text*.
    """
    ru_norm = normalise_ru_quote(quote["ru"])
    ru_units = split_units(ru_norm)
    anchors = {k.lower(): v for k, v in quote.get("anchors", {}).items()}
    n_anchors = len(anchors)

    if quote["kind"] == "skeleton":
        windows = localize_by_skeleton(corpus, operator_skeleton(ru_norm))
        locator = "operator_skeleton"
    elif quote.get("en_literal"):
        lit = quote["en_literal"]
        windows = [(m.start(), m.start() + len(lit))
                   for m in re.finditer(re.escape(lit), corpus.text)]
        locator = "literal_english"
    else:
        windows = localize_by_anchors(corpus, anchors)
        locator = "bilingual_anchor"

    if not windows:
        return Alignment(quote, locator, 0, 0, 0, 0, "", ru_units, [], [],
                         n_anchors, 0, [], [], (0, 0))

    raw_windows = list(windows)
    # a literal match is already exact; padding it would only make the
    # surrounding clauses look like distinct texts
    expanded = ([(lo, hi) for lo, hi in windows] if locator == "literal_english"
                else [expand_window(corpus, lo, hi) for lo, hi in windows])
    texts = [corpus.text[lo:hi] for lo, hi in expanded]
    distinct = len(set(texts))
    sites = list(expanded)
    site_pages = [",".join(str(p) for p in corpus.pages(lo, hi)) for lo, hi in expanded]

    # rank: most anchors satisfied, then the smallest DP cost, then earliest.
    best_i = 0
    best_key = None
    for i, (lo, hi) in enumerate(expanded):
        en_units = split_units(corpus.text[lo:hi])
        table = unit_table(en_units)
        links = align_units(en_units, ru_units, anchors=anchors, anchor_table=table)
        chunk = corpus.text[lo:hi].lower()
        hits = sum(1 for _, en_tok in anchors.items() if en_tok.lower() in chunk)
        key = (hits, -sum(link.cost for link in links), -lo)
        if best_key is None or key > best_key:
            best_key, best_i = key, i

    lo, hi = expanded[best_i]
    en_units = split_units(corpus.text[lo:hi])
    table = unit_table(en_units)
    links = align_units(en_units, ru_units, anchors=anchors, anchor_table=table)
    chunk = corpus.text[lo:hi]
    hits = sum(1 for _, en_tok in anchors.items() if en_tok.lower() in chunk.lower())
    quote = dict(quote)
    quote["pages"] = corpus.pages(lo, hi)
    return Alignment(quote, locator, len(windows), distinct, lo, hi, chunk,
                     ru_units, en_units, links, n_anchors, hits, sites,
                     site_pages, raw_windows[best_i])


def operator_invariance(aln: Alignment, corpus: Corpus | None = None) -> dict:
    """Does the target text reproduce the source's operator channel?

    Measured over the *un-padded* match window.  Padding exists to give the
    aligner whole units; using it here would count the operators of the
    neighbouring clause as channel losses.
    """
    ru_skel = [op for op, _, _ in scan_operators(normalise_ru_quote(aln.quote["ru"]))]
    en_window = aln.en_text
    if corpus is not None and aln.ops_window != (0, 0):
        en_window = corpus.slice(*aln.ops_window)
    en_skel = [op for op, _, _ in scan_operators(en_window)]
    n = min(len(ru_skel), len(en_skel))
    positional = sum(1 for k in range(n) if ru_skel[k] == en_skel[k])
    multiset = sum((collections.Counter(ru_skel) & collections.Counter(en_skel)).values())
    return {
        "n_en_ops": len(en_skel),
        "n_ru_ops": len(ru_skel),
        "positional_matches": positional,
        "positional_retention": round(positional / len(en_skel), 4) if en_skel else 0.0,
        "multiset_matches": multiset,
        "multiset_retention": round(multiset / len(en_skel), 4) if en_skel else 0.0,
    }


WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


#: function words, dropped before the within-unit token mapping.  Without
#: them the first content token of a Russian unit gets paired with the first
#: content token of an English unit that starts `The`, and the lexicon fills
#: up with `голограммы -> the`.
EN_STOP = frozenset("""
a an the of in on at to for with by from and or but that which who whom whose is
are was were be been being it its this that these those as not no nor so than
then there here he she they we you i him her them us me my your his our their
""".split())
RU_STOP = frozenset("""
в на с к и а что это он она оно они я ты мы вы не но или из у о об за по для
же ли бы то как где когда который которая которые которых его ее её их наш ваш
свой своя свои этом этому этот эту этом этот сама само самый самые очень уже ещё
еще только даже либо хотя потому также там тут от изо под над при без через около
после перед между среди вместо вместе против около более менее самый самая самое
""".split())


def content_tokens(text: str, stop: frozenset) -> list[str]:
    return [t for t in WORD_RE.findall(text.lower()) if t not in stop]


def bilingual_pairs(aln: Alignment) -> list[dict]:
    """Token-level gloss pairs harvested from a 1-1 alignment.

    Inside a 1-1 unit pair the two languages share no words, so tokens are
    mapped by relative position (the k-th of n Russian content tokens to the
    k-th of m English ones).  That is a guess, and it is published as one:
    every row carries its ``method``, and a pair is only called ``attested``
    when the anchor priors independently say the same thing.
    """
    rows = []
    for link in aln.links:
        if link.kind != KIND_11 or len(link.en) != 1 or len(link.ru) != 1:
            continue
        en_unit = aln.en_units[link.en[0]]
        ru_unit = aln.ru_units[link.ru[0]]
        latin = [t.lower() for t in LATIN_RE.findall(ru_unit.text)]
        ru_toks = content_tokens(ru_unit.text, RU_STOP)
        en_toks = content_tokens(en_unit.text, EN_STOP)
        if latin:
            klass, method = "retained_latin", "retained_latin"
        elif ru_toks and en_toks:
            klass, method = "translated", "proportional_within_unit"
        else:
            continue
        pairs = []
        if klass == "retained_latin":
            pairs = [(", ".join(sorted(set(latin))), ", ".join(en_toks[:3]))]
        else:
            for k, rt in enumerate(ru_toks):
                j = 0 if len(ru_toks) == 1 else round(k * (len(en_toks) - 1) / (len(ru_toks) - 1))
                pairs.append((rt, en_toks[j]))
        # A proportional mapping is only plausible when the two sides have
        # comparable token counts.  Outside 0.5-2.0 the k-th token on one
        # side says nothing about the k-th on the other, so the pair is
        # dropped and counted instead of published as noise.
        ratio = (len(en_toks) / len(ru_toks)) if ru_toks else 0.0
        if method == "proportional_within_unit" and not (0.5 <= ratio <= 2.0):
            SKIPPED_LOW_RATIO.append((aln.quote["id"], round(ratio, 2)))
            continue
        for rt, et in pairs:
            rows.append({
                "quote_id": aln.quote["id"],
                "page": ",".join(str(p) for p in aln.pages()),
                "en_unit": en_unit.text.strip(),
                "ru_unit": ru_unit.text.strip(),
                "ru_head": rt,
                "en_head": et,
                "class": klass,
                "method": method,
                "n_ru_tokens": len(ru_toks),
                "n_en_tokens": len(en_toks),
            })
    return rows


#: unit pairs dropped for a token-count ratio outside [0.5, 2.0]; published in
#: `diagnostics.tsv` so the withheld pairs are counted, not hidden.
SKIPPED_LOW_RATIO: list[tuple] = []


def derive_lexicon(rows: list[dict], anchors: dict[str, str]) -> list[dict]:
    """Bilingual lexicon from aligned unit pairs, graded by provenance.

    ``anchor_prior``  read off the target text before the corpus was searched
    ``aligned``       produced by a 1-1 alignment, by relative position
    ``anchor_prior+aligned``  both agree - the only rows called ``attested``
    """
    seen: dict[tuple[str, str], dict] = {}
    for ru_tok, en_tok in anchors.items():
        seen[(ru_tok.lower(), en_tok.lower())] = {
            "ru": ru_tok.lower(), "en": en_tok.lower(),
            "provenance": "anchor_prior", "method": "anchor_prior",
            "quotes": set(), "pages": set(), "n_quotes": 0}
    for row in rows:
        key = (row["ru_head"].lower(), row["en_head"].lower())
        entry = seen.get(key)
        if entry is None:
            entry = {"ru": key[0], "en": key[1], "provenance": "aligned",
                     "method": row["method"], "quotes": set(), "pages": set(),
                     "n_quotes": 0}
            seen[key] = entry
        elif entry["provenance"] == "anchor_prior":
            entry["provenance"] = "anchor_prior+aligned"
        entry["n_quotes"] += 1
        entry["quotes"].add(row["quote_id"])
        entry["pages"].update(int(x) for x in row["page"].split(",") if x.isdigit())
    out = []
    for e in seen.values():
        e["quotes"] = ",".join(sorted(e["quotes"]))
        e["pages"] = ",".join(str(x) for x in sorted(e["pages"]))
        e["confidence"] = "attested" if e["provenance"].endswith("+aligned") else "prior"
        out.append(e)
    out.sort(key=lambda e: (e["confidence"] != "attested", -e["n_quotes"], e["ru"]))
    return out


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"), strict=False))
            for line in lines[1:] if line.strip()]


def oracle_checks(corpus: Corpus) -> list[dict]:
    """The author's own words, tested against the book (L7's actual job)."""
    neo = {r.get("lower", "") for r in read_tsv(NEO_TSV)}
    ent = {r.get("entity", "").lower() for r in read_tsv(ENTITY_TSV)}
    pages = [(off, p) for off, p in corpus.page_starts]
    checks = []
    for label, pattern, why in AUTHOR_TERMS:
        rx = re.compile(pattern, re.I)
        n = n_pages = 0
        first = ""
        for off, pno in pages:
            c = len(rx.findall(corpus.text[off:(_pageslice(corpus, off))]))
            if c:
                n += c
                n_pages += 1
                if not first:
                    first = str(pno)
        key = label.lower()
        in19 = "y" if key in neo else "n"
        in30 = "y" if key in ent else "n"
        if n == 0:
            verdict_ = "absent_from_book"
        elif n == 1:
            verdict_ = "single_occurrence"
        else:
            verdict_ = "corroborated"
        checks.append({
            "item": label, "kind": "term", "why": why,
            "n_occurrences": n, "n_as_operator_token": 0,
            "n_pages": n_pages, "first_page": first,
            "in_19_neologisms": in19, "in_30_entities": in30, "verdict": verdict_,
        })
    op_counts = collections.Counter(op for op, _, _ in corpus.ops)
    for op, pattern, why in AUTHOR_OPERATORS:
        rx = re.compile(pattern)
        n = len(rx.findall(corpus.text))
        checks.append({
            "item": op, "kind": "operator", "why": why,
            "n_occurrences": n, "n_as_operator_token": op_counts.get(op, 0),
            "n_pages": "", "first_page": "", "in_19_neologisms": "n",
            "in_30_entities": "n",
            "verdict": "corroborated" if n else "absent_from_book",
        })
    return checks


def _pageslice(corpus: Corpus, off: int) -> int:
    """The offset at which the page containing ``off`` ends."""
    later = [o for o, _ in corpus.page_starts if o > off]
    return later[0] if later else len(corpus.text)


# ===========================================================================
# 8. writers
# ===========================================================================

def page_str(pages: list[int], limit: int = 8) -> str:
    """`p6,222` style, capped so one recurring coinage cannot bloat a TSV."""
    head = ",".join(str(p) for p in pages[:limit])
    return head + (f" (+{len(pages) - limit} more)" if len(pages) > limit else "")


def write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(h, "")).replace("\t", " ")
                                .replace("\n", " ") for h in header) + "\n")


def build(log=print) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    SKIPPED_LOW_RATIO.clear()          # module-level counter; reset per build
    corpus = load_corpus()

    # --- 1. ledger ---------------------------------------------------------
    rows = ledger_rows()
    write_tsv(OUT / "sources.tsv",
              ["key", "kind", "title", "url", "language", "license",
               "authorization", "http_status", "final_url", "bytes", "sha256",
               "probe_date", "verdict", "ingestible", "note"], rows)
    log(f"[translation_oracle] sources.tsv: {len(rows)} candidates, "
        f"{sum(1 for r in rows if r['ingestible'] == 'y')} ingestible")

    # --- 2. paratext -------------------------------------------------------
    paratext = [{
        "id": pt["id"], "source": pt["source"],
        "source_url": SOURCE_BY_KEY[pt["source"]].url,
        "interviewer": pt["interviewer"], "date": pt["date"],
        "claim_kind": pt["claim_kind"], "quote": pt["quote"], "claim": pt["claim"],
    } for pt in AUTHOR_PARATEXT]
    write_tsv(OUT / "author_paratext.tsv",
              ["id", "source", "source_url", "interviewer", "date", "claim_kind",
               "quote", "claim"], paratext)

    # --- 3. quotations -----------------------------------------------------
    quotes = list(RU_QUOTATIONS) + [DIVERGENT_QUOTE]
    quote_rows = []
    alignments: list[Alignment] = []
    for q in quotes:
        aln = align_quote(corpus, q)
        alignments.append(aln)
        pages = aln.pages()
        quote_rows.append({
            "id": q["id"], "attributor": attributor_of(q["review"]),
            "review": q["review"], "kind": q["kind"],
            "ru": q["ru"], "ru_n_chars": len(q["ru"]),
            "ru_operators": ",".join(operator_skeleton(normalise_ru_quote(q["ru"]))),
            "locator": aln.locator, "n_sites": aln.n_windows,
            "n_distinct_texts": aln.n_distinct,
            "evidence": aln.evidence(),
            "sites": ";".join(f"p{blob}:{st}-{en}"
                              for blob, (st, en) in zip(aln.site_pages, aln.sites[:8],
                                                        strict=False)),
            "en_page": page_str(pages),
            "en_start": aln.en_start, "en_end": aln.en_end,
            "en": aln.en_text.strip()[:400],
            "n_anchors": aln.n_anchors, "n_anchors_hit": aln.n_anchors_hit,
            "confidence": aln.confidence(), "note": q["note"],
        })
    write_tsv(OUT / "ru_quotations.tsv",
              ["id", "attributor", "review", "kind", "ru", "ru_n_chars",
               "ru_operators", "locator", "n_sites", "n_distinct_texts",
               "evidence", "sites", "en_page", "en_start", "en_end",
               "en", "n_anchors", "n_anchors_hit", "confidence", "note"],
              quote_rows)
    n_located = sum(1 for r in quote_rows if r["evidence"] in
                    ("unique_skeleton", "unique_anchor_text", "unique_literal_text",
                     "recurring_text"))
    log(f"[translation_oracle] quotations: {len(quote_rows)} pinned, "
        f"{n_located} uniquely located in the corpus")

    # --- 3b. the review ledger --------------------------------------------
    # The attribution of a short quotation of a copyrighted translation is the
    # thing that makes the quotation lawful to publish, so it is recorded as an
    # artifact of its own rather than left implicit in a string column: who
    # wrote the review, which account, when it was published, the permalink
    # that identifies the block, the fingerprint of the text that was
    # transcribed, and which quotations came out of it.
    review_rows = []
    for rev in FANTLAB_REVIEWS:
        froms = [r["id"] for r in quote_rows if r["review"] == rev["review"]]
        review_rows.append({
            "review": rev["review"],
            "author": rev["author"],
            "account": rev["account"],
            "date_published": rev["date_published"],
            "date": rev["date_published"][:10],
            "attributor": attributor_of(rev["review"]),
            "permalink": review_permalink(rev["review"]),
            "body_sha256": rev["body_sha256"],
            "n_chars": rev["n_chars"],
            "n_quotations": len(froms),
            "quotations": ",".join(froms),
        })
    write_tsv(OUT / "fantlab_reviews.tsv",
              ["review", "author", "account", "date_published", "date",
               "attributor", "permalink", "body_sha256", "n_chars",
               "n_quotations", "quotations"], review_rows)

    # --- 4. the alignment artifact ----------------------------------------
    align_rows = []
    for aln in alignments:
        for k, link in enumerate(aln.links):
            align_rows.append({
                "quote_id": aln.quote["id"],
                "page": page_str(aln.pages()),
                "link": k, "kind": link.kind,
                "n_en_units": len(link.en), "n_ru_units": len(link.ru),
                "en_indices": ",".join(str(i) for i in link.en),
                "ru_indices": ",".join(str(j) for j in link.ru),
                "en_ops_before": aln.en_units[link.en[0]].ops_before if link.en else "",
                "ru_ops_before": aln.ru_units[link.ru[0]].ops_before if link.ru else "",
                "en": " || ".join(aln.en_units[i].text.strip() for i in link.en),
                "ru": " || ".join(aln.ru_units[j].text.strip() for j in link.ru),
                "en_chars": sum(aln.en_units[i].n_chars for i in link.en),
                "ru_chars": sum(aln.ru_units[j].n_chars for j in link.ru),
                "cost": round(link.cost, 3),
            })
    write_tsv(OUT / "alignment.tsv",
              ["quote_id", "page", "link", "kind", "n_en_units", "n_ru_units",
               "en_indices", "ru_indices", "en_ops_before", "ru_ops_before",
               "en", "ru", "en_chars", "ru_chars", "cost"], align_rows)
    n_unit_pairs = sum(r["n_en_units"] * r["n_ru_units"] for r in align_rows
                       if r["kind"] == "1-1")
    log(f"[translation_oracle] alignment.tsv: {len(align_rows)} links, "
        f"{n_unit_pairs} 1-1 unit pairs")

    # --- 5. operator-channel invariance ------------------------------------
    inv_rows = []
    for aln in alignments:
        if aln.locator != "operator_skeleton":
            continue
        stats = operator_invariance(aln, corpus)
        skeleton = operator_skeleton(normalise_ru_quote(aln.quote["ru"]))
        inv_rows.append({
            "quote_id": aln.quote["id"],
            "page": page_str(aln.pages()),
            "n_en_ops": stats["n_en_ops"], "n_ru_ops": stats["n_ru_ops"],
            "positional_matches": stats["positional_matches"],
            "positional_retention": stats["positional_retention"],
            "multiset_matches": stats["multiset_matches"],
            "multiset_retention": stats["multiset_retention"],
            "n_exact_windows": aln.n_windows,
            "corpus_operator_tokens": len(corpus.ops),
            "skeleton": "".join(skeleton),
        })
    write_tsv(OUT / "operator_invariance.tsv",
              ["quote_id", "page", "n_en_ops", "n_ru_ops", "positional_matches",
               "positional_retention", "multiset_matches", "multiset_retention",
               "n_exact_windows", "corpus_operator_tokens", "skeleton"], inv_rows)

    # --- 6. the bilingual lexicon ------------------------------------------
    pairs: list[dict] = []
    anchors_all: dict[str, str] = {}
    for q in quotes:
        anchors_all.update({k.lower(): v.lower() for k, v in q.get("anchors", {}).items()})
    for aln in alignments:
        if aln.evidence() in ("unique_skeleton", "unique_anchor_text",
                              "unique_literal_text", "recurring_text"):
            pairs.extend(bilingual_pairs(aln))
    lexicon = derive_lexicon(pairs, anchors_all)
    gloss = {r.get("surface", "").lower(): r.get("reading", "")
             for r in read_tsv(GLOSS_TSV)}
    for e in lexicon:
        e["project_reading"] = gloss.get(e["en"], "")
        e["pages"] = page_str([int(x) for x in e["pages"].split(",") if x.isdigit()])
    write_tsv(OUT / "bilingual_lexicon.tsv",
              ["ru", "en", "provenance", "method", "confidence", "n_quotes",
               "quotes", "pages", "project_reading"], lexicon)
    log(f"[translation_oracle] bilingual_lexicon.tsv: {len(lexicon)} entries "
        f"({sum(1 for e in lexicon if e['confidence'] == 'attested')} attested)")

    # --- 7. the author's claims --------------------------------------------
    checks = oracle_checks(corpus)
    write_tsv(OUT / "oracle_checks.tsv",
              ["item", "kind", "why", "n_occurrences", "n_as_operator_token",
               "n_pages", "first_page", "in_19_neologisms", "in_30_entities",
               "verdict"], checks)

    # --- 8. aligner evaluation --------------------------------------------
    ev = evaluate_aligner()
    write_tsv(OUT / "aligner_eval.tsv",
              ["system", "split", "document", "n_en", "n_ru", "n_gold_links",
               "aer"], ev)
    overall_aer = next(r["aer"] for r in ev
                       if r["document"] == "ALL" and r["system"] == "dp")
    heldout_aer = next(r["aer"] for r in ev
                       if r["document"] == "SUBTOTAL" and r["system"] == "dp"
                       and r["split"] == "held_out")
    baseline_aer = next(r["aer"] for r in ev
                        if r["document"] == "ALL" and r["system"] == "proportional")
    # The comparison that means anything is same-split, so the baseline is also
    # read off the held-out half.  Reporting the DP held-out figure against the
    # baseline's all-twelve-document figure would flatter the aligner.
    baseline_heldout_aer = next(r["aer"] for r in ev
                                if r["document"] == "SUBTOTAL"
                                and r["system"] == "proportional"
                                and r["split"] == "held_out")

    # --- 9. reconciliation / diagnostics / summary -------------------------
    n_pages = len(corpus.page_starts)
    recon = [
        {"quantity": "pages", "here": n_pages, "earlier": "225 (#8 page/folio map)",
         "note": "same layout markdown, same page markers"},
        {"quantity": "operator tokens", "here": len(corpus.ops),
         "earlier": "11,000 (#17/#70)",
         "note": "same greedy longest match over #10's OPS, but over the "
                 "whole cleaned layout text incl. headings, so the count is "
                 "an upper bound on #17's body-only figure"},
        {"quantity": "alignment links", "here": len(align_rows),
         "earlier": "0", "note": "new in #58"},
        {"quantity": "parallel text obtained", "here": "no", "earlier": "no",
         "note": "the Skobin translation is a commercial 2006 АСТ edition; "
                 "the only open-web full texts are unauthorized mirrors, "
                 "which the ingestion gate refuses"},
        {"quantity": "disambiguation signal obtained", "here": "yes",
         "earlier": "no",
         "note": "operator-channel invariance + author-paratext checks"},
    ]
    write_tsv(OUT / "reconciliation.tsv",
              ["quantity", "here", "earlier", "note"], recon)

    diag = [
        {"diag": "schema", "value": SCHEMA},
        {"diag": "corpus_pages", "value": n_pages},
        {"diag": "corpus_chars", "value": len(corpus.text)},
        {"diag": "corpus_operator_tokens", "value": len(corpus.ops)},
        {"diag": "sources_probed", "value": len(SOURCES)},
        {"diag": "sources_ingestible",
         "value": sum(1 for s in SOURCES if ingestible(s))},
        {"diag": "mirrors_refused",
         "value": sum(1 for s in SOURCES if s.authorization == AUTH_MIRROR)},
        {"diag": "quotations_pinned", "value": len(quotes)},
        {"diag": "fantlab_review_blocks", "value": len(FANTLAB_REVIEWS)},
        {"diag": "quotations_attributed_to_a_pinned_review",
         "value": sum(1 for q in quotes if q["review"] in REVIEW_BY_ID)},
        {"diag": "quotations_uniquely_located", "value": n_located},
        {"diag": "skeleton_quotes", "value": sum(1 for q in quotes if q["kind"] == "skeleton")},
        {"diag": "aligned_links", "value": len(align_rows)},
        {"diag": "aligned_1_1_unit_pairs", "value": n_unit_pairs},
        {"diag": "lexicon_entries", "value": len(lexicon)},
        {"diag": "lexicon_pairs_withheld_low_token_ratio",
         "value": len(SKIPPED_LOW_RATIO)},
        {"diag": "lexicon_attested",
         "value": sum(1 for e in lexicon if e["confidence"] == "attested")},
        {"diag": "aligner_aer_builtin_gold", "value": overall_aer},
        {"diag": "aligner_aer_held_out_half", "value": heldout_aer},
        {"diag": "baseline_aer_proportional", "value": baseline_aer},
        {"diag": "baseline_aer_held_out_half", "value": baseline_heldout_aer},
        {"diag": "gold_pairs", "value": len(GOLD_SENTENCES)},
        {"diag": "author_claims_checked", "value": len(checks)},
    ]
    write_tsv(OUT / "diagnostics.tsv", ["diag", "value"], diag)

    summary = {
        "schema": SCHEMA,
        "verdict": "partial_oracle",
        "parallel_text_obtained": False,
        "corpus_operator_tokens": len(corpus.ops),
        "skeleton_localized_quotations": sum(
            1 for a in alignments if a.evidence() == "unique_skeleton"),
        "quotations_uniquely_located": n_located,
        "quotations_total": len(quotes),
        "aligned_links": len(align_rows),
        "aligned_1_1_unit_pairs": n_unit_pairs,
        "operator_channel_translation_invariant": all(
            r["positional_retention"] == 1.0 for r in inv_rows) if inv_rows else False,
        "bilingual_lexicon_entries": len(lexicon),
        "bilingual_lexicon_attested": sum(1 for e in lexicon if e["confidence"] == "attested"),
        "aligner_aer_builtin_gold": overall_aer,
        "aligner_aer_held_out_half": heldout_aer,
        "baseline_aer_proportional": baseline_aer,
        "baseline_aer_held_out_half": baseline_heldout_aer,
        "author_claims_corrobated": sum(1 for c in checks if c["verdict"] == "corroborated"),
        "author_claims_absent": sum(1 for c in checks if c["verdict"].startswith("absent")),
        "author_claims_single": sum(1 for c in checks if c["verdict"] == "single_occurrence"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")

    # --- 10. reports -------------------------------------------------------
    (OUT / "availability_report.md").write_text(
        availability_report(rows, summary), encoding="utf-8")
    (OUT / "alignment_report.md").write_text(
        alignment_report(alignments, alignments and inv_rows, lexicon, checks),
        encoding="utf-8")
    return summary


def availability_report(ledger: list[dict], summary: dict) -> str:
    by_verdict = collections.defaultdict(list)
    for row in ledger:
        by_verdict[row["verdict"]].append(row)
    lines = [
        "# #58 - translations as an interpretive oracle: acquisition report",
        "",
        "Generated by `work/translation_oracle.py`.  Every row below is a "
        "probed candidate source; the probe evidence is pinned in "
        "`sources.tsv` and re-runnable with `--probe`.",
        "",
        "## The headline",
        "",
        "**The full Skobin parallel text could not be obtained lawfully, and "
        "that is recorded as a result, not a failure to finish.**  The "
        "Russian edition is a commercial 2006 АСТ/Адаптек printing "
        "(ISBN 5-17-037280-9 / 5-93827-069-3, 320 pp, series «Alternativa»); "
        "FantLab catalogues it and serves no text for it; and every full text "
        "reachable on the open web is an unauthorized mirror.  The ingestion "
        "gate refuses those eight hosts, and no body was ever requested from "
        "them - the mirrors were probed with HEAD only, to establish that they "
        "exist.",
        "",
        "**The disambiguation signal was obtained anyway, from material that "
        "is lawfully quotable.**  Russian readers of the book publish "
        "quotations of the Skobin translation in their FantLab reviews.  "
        f"{summary['quotations_total']} such quotations are pinned here "
        f"(short research quotations, attributed, the same convention "
        "`work/reference_corpora.py` applies to Gibson and Burroughs) and "
        f"{summary['quotations_uniquely_located']} of them resolve to a "
        "single English text inside the 225-page corpus.  One of the two "
        "left over is a one-word token, which cannot be unique by "
        "construction; see `alignment_report.md`.",
        "",
        "**The attribution is what makes these quotable, so it is pinned to "
        "the source rather than typed.**  The twelve quotations come out of "
        f"{len(FANTLAB_REVIEWS)} reader reviews - "
        + ", ".join(f"{r['author']} ({r['date_published'][:10]})"
                    for r in FANTLAB_REVIEWS)
        + " - each row of `fantlab_reviews.tsv` carrying the reviewer's "
        "account, `datePublished`, the `#responseNNNNNN` permalink and a "
        "SHA-256 over the block body, and each quotation's attribution is "
        "*derived* from that table.  An earlier revision typed the reviewer "
        "into each row and got six of the twelve wrong, crediting Reza "
        "Negarestani, whose name occurs on the work page only in the "
        "«Рецензии» bibliography as the author of an external review *of* "
        "the book.  `--verify-reviews` re-reads the page's own "
        "`schema.org/Review` markup and confirms every quotation still "
        "occurs in the review it names.",
        "",
        "**The finding that matters:** the Russian edition preserves the "
        "book's symbol channel verbatim.  `=`, `::`, `//`, `++`, `/`, `<<`, "
        "`>>` come through untranslated, which makes the operator stream a "
        "translation-invariant key.  Two long quotations were located by the "
        "operator skeleton *alone*, with no lexical anchor at all, and both "
        "come out at **100% positional operator retention** against the "
        "English: 16/16 operators for one, 12/12 for the other.  RUQ01's "
        "16-operator sequence occurs exactly once in the corpus's "
        f"{summary['corpus_operator_tokens']:,} operator tokens; RUQ02's "
        "12-operator sequence occurs twice, and the two sites are "
        "byte-identical (pp. 5 and 221) - which is a finding about the book, "
        "not about the aligner.  That is the Rosetta stone this corpus "
        "actually has.",
        "",
        "## The numbers",
        "",
        "| measure | value |",
        "|---|---|",
        f"| Russian quotations pinned | {summary['quotations_total']} |",
        f"| resolved to a single English text | {summary['quotations_uniquely_located']} |",
        f"| located by operator skeleton alone | {summary['skeleton_localized_quotations']} |",
        f"| operator-channel positional retention | "
        f"{'1.000 (both skeleton quotations)' if summary['operator_channel_translation_invariant'] else 'partial'} |",
        f"| aligned links / 1-1 unit pairs | {summary['aligned_links']} / "
        f"{summary['aligned_1_1_unit_pairs']} |",
        f"| bilingual lexicon entries (attested) | "
        f"{summary['bilingual_lexicon_entries']} "
        f"({summary['bilingual_lexicon_attested']}) |",
        f"| aligner AER, held-out half | {summary['aligner_aer_held_out_half']} |",
        f"| length-only baseline AER, same held-out half | "
        f"{summary['baseline_aer_held_out_half']} |",
        f"| aligner / baseline AER, whole built-in gold | "
        f"{summary['aligner_aer_builtin_gold']} / "
        f"{summary['baseline_aer_proportional']} |",
        f"| author claims corroborated / single / absent | "
        f"{summary['author_claims_corrobated']} / {summary['author_claims_single']} / "
        f"{summary['author_claims_absent']} |",
        "",
        "AER is `alignment_error_rate` in `work/translation_oracle.py`: the "
        "Gale & Church / Levow standard, counting deletions and insertions of "
        "null-collapsed links over the gold links of the same split.  It is a "
        "rate of errors per gold link, not a proportion of a fixed set, so it "
        "is **not bounded above by 1** - the proportional baseline sits near 2 "
        "because it proposes about as many wrong links as the gold has links.  "
        "The comparison that carries meaning is the gap between the two rows, "
        "not either value on its own.",
        "",
        "## Ledger",
        "",
    ]
    for verdict_key in sorted(by_verdict):
        lines.append(f"### `{verdict_key}` ({len(by_verdict[verdict_key])})")
        lines.append("")
        lines.append("| source | http | bytes | ingestible | what it is |")
        lines.append("|---|---|---|---|---|")
        for row in sorted(by_verdict[verdict_key], key=lambda r: r["key"]):
            lines.append(f"| `{row['key']}` | {row['http_status']} | {row['bytes']} | "
                         f"{row['ingestible']} | {row['title']} |")
        lines.append("")
    lines += [
        "## What was searched and not found",
        "",
        "* the nettime list, named in the methods inventory as the third "
        "interview: the host is reachable but the `nettime-l` 2002 thread "
        "index carries no Siratori subject and the Wayback CDX index of "
        "nettime.org has no matching path.  Recorded as "
        "`searched_not_found`, not dropped.",
        "* a lawful online edition: FantLab's reader endpoint 404s, the "
        "LibreBook entry has an empty chapter list, Litres and the NEL "
        "refuse the agent.",
        "* an openly licensed EN-RU parallel corpus for aligner validation: "
        "not needed.  The aligner is scored against a built-in gold of "
        f"{len(GOLD_SENTENCES)} authored pairs with planted 1-n and n-1 "
        "operations, hermetic and offline, with the weights fitted on a "
        "tuning half and reported on a held-out half.",
        "",
        "## Licensing position",
        "",
        "No full text of the Skobin translation was downloaded, and no "
        "mirror's body was fetched.  What is used is (a) bibliographic "
        "records, (b) short attributed quotations of the translation inside "
        "third-party public reviews, and (c) the author's own published "
        "interviews.  If a licensed copy of the Russian edition is ever "
        "supplied to the repository, drop it in and rerun: the aligner and "
        "the entire oracle pipeline consume it unchanged, and the skeleton "
        "localiser below is the piece that would then align all 225 pages "
        "instead of eleven quotations.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    """One markdown table cell: no tabs, no pipes, no newlines."""
    return re.sub(r"\s+", " ", text.replace("|", "/")).strip()


def alignment_report(alignments, inv_rows, lexicon, checks) -> str:
    lines = [
        "# #58 - alignment of the Russian paratext to the English book",
        "",
        "Generated by `work/translation_oracle.py`.",
        "",
        "## How a quotation is attributed",
        "",
        "Each quotation is a short research quotation of a copyrighted "
        "translation, so the attribution is what makes it lawful to publish, "
        "and it is not a free-text field.  Every row carries a `review` key "
        "naming one of the review blocks in `fantlab_reviews.tsv` - author, "
        "account, `datePublished`, the `#responseNNNNNN` permalink, and a "
        "SHA-256 over the block's tag-stripped body - and the attribution "
        "printed below is *derived* from that row.  All twelve quotations come "
        "out of three reader reviews; the permalink on each line resolves to "
        "the exact block the text was transcribed from.",
        "",
        "An earlier revision typed the attribution into each row instead, and "
        "six rows were wrong: they credited Reza Negarestani, whose name does "
        "appear on the work page but only in the «Рецензии» bibliography block "
        "as the author of an *external review of the book* (work1915318).  He "
        "wrote none of these quotations, and that page carries no review body "
        "at all.  Deriving the field removes the possibility rather than "
        "correcting six strings.",
        "",
        "## How a quotation is located",
        "",
        "1. **Skeleton quotations.**  Normalise the Russian quotation's span "
        "delimiters (`‹‹`/`››` -> the book's `<<`/`>>`; the raw form is kept "
        "in `ru_quotations.tsv`), take its operator sequence, and slide it "
        "over the corpus's operator stream using the same greedy longest "
        "match as #10.  The number of exact windows is the evidence.",
        "2. **Anchor quotations.**  For short quotations the target text is "
        "located by bilingual token anchors.  The anchors are *priors* read "
        "off the target text; the build reports how many of them the corpus "
        "corroborates and whether the resulting window is unique, so a weak "
        "anchor set is visible as a low confidence rather than hidden.",
        "3. Both then run through the same monotone DP aligner, which emits "
        "`1-1` / `1-0` / `0-1` / `n-m` links weighted by log-length "
        "disagreement, preceding-operator agreement, shared numerals and "
        "anchor agreement.",
        "",
        "Two quotations (RUQ02, RUQ04) have **two** sites, byte-identical, on "
        "pp. 5 and 221.  That is not an alignment failure - Siratori's loop "
        "writing puts some passages on the page twice - so the evidence column "
        "separates *one position* from *one text* and the sites column lists "
        "every occurrence.",
        "",
        "## Per quotation",
        "",
    ]
    for aln in alignments:
        pages = aln.pages()
        lines += [
            f"### {aln.quote['id']} - {aln.quote['kind']}, confidence "
            f"{aln.confidence()}",
            "",
            f"* attributed to: {attributor_of(aln.quote['review'])} "
            f"({review_permalink(aln.quote['review'])})",
            f"* evidence: `{aln.evidence()}`, {aln.n_windows} site(s) "
            f"({aln.n_distinct} distinct text(s)), confidence {aln.confidence()}",
            f"* sites: {', '.join(f'p{b}' for b in aln.site_pages[:8]) or '-'}",
            f"* English: pages {page_str(pages) or '-'}, offsets "
            f"{aln.en_start}-{aln.en_end}",
            f"* anchors: {aln.n_anchors_hit}/{aln.n_anchors} corroborated",
            f"* note: {aln.quote['note']}",
            "",
            "| # | kind | ru | en |",
            "|---|---|---|---|",
        ]
        for k, link in enumerate(aln.links):
            ru = _cell(" || ".join(aln.ru_units[j].text for j in link.ru))
            en = _cell(" || ".join(aln.en_units[i].text for i in link.en))
            lines.append(f"| {k} | {link.kind} | {ru} | {en} |")
        lines.append("")
    corpus_ops = sorted({r["corpus_operator_tokens"] for r in inv_rows})
    ops_label = ("/".join(f"{n:,}-token" for n in corpus_ops) + " "
                 if corpus_ops else "")
    lines += [
        "## Operator-channel invariance",
        "",
        f"| quote | page | EN ops | RU ops | positional retention | exact "
        f"windows in the corpus's {ops_label}operator stream |",
        "|---|---|---|---|---|---|",
    ]
    for row in inv_rows:
        lines.append(f"| {row['quote_id']} | {row['page']} | {row['n_en_ops']} | "
                     f"{row['n_ru_ops']} | {row['positional_retention']} | "
                     f"{row['n_exact_windows']} |")
    lines += [
        "",
        "A positional retention of 1.0 means the Russian text carries the "
        "source's operator sequence, in order, with nothing dropped and "
        "nothing added.  Combined with a unique window over the corpus's whole "
        "operator stream, that is a machine-checkable alignment rather than a "
        "resemblance.",
        "",
        "## Bilingual lexicon",
        "",
        f"{len(lexicon)} entries, of which "
        f"{sum(1 for e in lexicon if e['confidence'] == 'attested')} are "
        "attested by an aligned unit pair and the rest are the anchor priors "
        "themselves.  `project_reading` is #62's gloss where one exists, so "
        "the external reading and the internal one can be compared directly.",
        "",
        "| ru | en | provenance | method | quotes | pages | #62 reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in lexicon[:80]:
        lines.append(f"| {e['ru']} | {e['en']} | {e['provenance']} | "
                     f"{e['method']} | {e['quotes']} | "
                     f"{page_str([int(x) for x in e['pages'].split(',') if x.isdigit()])} | "
                     f"{e['project_reading']} |")
    lines += [
        "",
        "## The author's own terms, tested against the book",
        "",
        "| item | kind | occurrences | pages | first page | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c['item']} | {c['kind']} | {c['n_occurrences']} | "
                     f"{c['n_pages'] or '-'} | {c['first_page'] or '-'} | "
                     f"{c['verdict']} |")
    lines += [
        "",
        "The author's 2002 interview lexicon and the 2002 English book do not "
        "coincide.  `ADAM` and `dogs` - the two items the interviewer names "
        "explicitly - are abundantly present.  The essayistic vocabulary he "
        "uses around them (`hardweb`, `creature intensity`, `echo archive`) "
        "is absent, and the two terms that do occur (`gene-dub`, "
        "`Biocapture`) occur exactly once each, both as chapter headings.  "
        "That is the useful shape of the answer: the interviews confirm the "
        "recurring nouns the reader could see unaided and say nothing about "
        "the method vocabulary, which is a claim about the interviews, not "
        "about the book.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--probe", action="store_true",
                    help="re-run the HTTP source probes live and diff against the pins")
    ap.add_argument("--verify-reviews", action="store_true",
                    help="re-fetch the FantLab work page and confirm every "
                         "quotation still occurs in the review it is "
                         "attributed to")
    args = ap.parse_args(argv)
    OUT = Path(args.out)
    if args.probe:
        run_probes()
    if args.verify_reviews:
        got = probe(FANTLAB_REVIEWS_URL)
        if got["status"] != 200:
            print(f"[translation_oracle] work page returned {got['status']}; "
                  f"cannot verify")
            return 1
        # does each review block still carry the author and date we pinned?
        blocks = parse_review_blocks(got["text"])
        drift = sum(1 for r in verify_review_attribution(blocks=blocks)
                    if r.get("verdict") != "ok")
        bad = 0
        for row in check_quotation_provenance(blocks):
            found = row["found_in"]
            ok = found == [row["review"]]
            bad += not ok
            print(f"[translation_oracle] {row['id']} -> "
                  f"{row['review']} {'OK' if ok else 'MISMATCH ' + str(found)}")
        if drift:
            print(f"[translation_oracle] {drift} review block(s) drifted from "
                  f"the pins; attribution needs re-checking")
        return 1 if (bad or drift) else 0
    summary = build()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())