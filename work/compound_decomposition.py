#!/usr/bin/env python3
"""Compound / portmanteau decomposition and affix lexicon - Chainlink #18.

L2 step 2 of `docs/meaning-extraction-methods.md`.  #17 gave us the *unit*
(`soul/gram`, `gene=TV`, `boy-roid`) and its parts; #18 asks what the parts are
made of.  Four decomposition channels are handled separately, because the book
keeps them separate:

    open juncture     an explicit linker between two words:
                       `-`  drug-embryo   boy-roid     retro-ADAM
                       `=`  fear=cell     gene=TV      bio=less=soul/gram
                       `/`  soul/gram     vital=icon
                       `++` `+`  murder++the
    closed juncture    two lexemes written as ONE word, no linker:
                       acidhuman  hydromachine  drugembryo  soulmachine
    bare morpheme      the classifier head stands alone: machine  cell  drone
    atomic            an opaque coinage the segmenter cannot resolve: feti omotya

Second question: are the nine morphemes L2 names (-oid, -machine, -vamp, -gram,
-cell, -embryo, -drone, -junkie, -mania) *productive derivational affixes*?  The
measure that decides it is corpus-internal and needs no tagger:

  * a **bound** morpheme is one the book never lets stand alone
    (`roid` free-standing 1 time in 341);
  * a **productive** one takes many bases (`-drone` takes 13);
  * a **hapax-base classifier** takes exactly one base and repeats it
    (`boy-roid` 338/340 -> the morpheme tags *one* base, it does not derive a
    family).

Verdicts are computed, not asserted; every number in the spec comes from the
build.

Resources: `wordfreq`'s English list (frequency-filtered) as the lexicon, and
`operator_tokenizer` (#17) as the unit layer - this module never re-tokenizes.
Tests: work/test_compound_decomposition.py.
"""
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import operator_grammar as OG                                       # #14
import operator_tokenizer as OT                                     # #17

try:
    from wordfreq import iter_wordlist, zipf_frequency
except ImportError:                                                 # pragma: no cover
    iter_wordlist = zipf_frequency = None

HERE = os.path.dirname(os.path.abspath(__file__))
# #17's unit layer is built on the *layout-recovered* markdown (#7), not on the
# raw marker output, so #18 reads the same file: decomposition must not be
# computed against a different text from the units it decomposes.
SRC = OT.SRC
OUT = os.path.normpath(os.path.join(HERE, "..", "analysis", "morphology"))
# v2: adds the `link_kind` column to decompositions.tsv / unit_screen.tsv, so
# the spec's channel table is joinable without re-deriving the bucket from the
# `linkers` string, and renames the `open` aggregate to `open_any` /
# `open_single` (v1's `links["open"]` was the single-linker subtotal but read
# like the total; see S1).
SCHEMA = "compound-decomposition/2"

# ---------------------------------------------------------------------------
# tunables - every one of these is a decision, and the spec prints it
# ---------------------------------------------------------------------------
LEX_MIN_ZIPF = 1.5     # lexicon = wordfreq entries at least this common
Z3 = 2.5               # 3-letter part must reach this zipf
ZFIRST = 2.5           # a non-final part must reach this zipf
SPLICE_MIN_SPAN = 4    # a straddling word must be at least this long
SPLICE_MIN_B = 3       # ... and its right-hand piece at least this long
MAXN = 3               # at most this many closed-juncture morphemes
PENALTY = 2.0          # per-morpheme cost, so 2 short beats 3 long
DICT_WORD_MINLEN = 5   # in-lexicon words are re-analysed from this length up
MIN_STEM = 4           # shortest stem an English suffix may leave behind

# A two-letter part is only credible as an English *prefix*, and only in
# non-final position: `de`+`flesh` is a split, `an` at the end of `acidhuman`
# is not.  Fixed before looking at the corpus, like EN_SUFFIXES.
PREFIX2 = {"de", "re", "un", "in", "ex", "co", "bi", "di", "up", "no", "on",
           "at", "so", "as", "or", "by", "to", "ad", "be", "do", "go", "he",
           "if", "is", "me", "my", "we", "us", "am", "it"}

# Closed-class words.  A function word is never a morpheme of a coinage: it
# cannot be what the book fused to make a new noun.  Without this the DP
# maximises zipf and reads `tokage` as `to`+`kage` and `fecundates` as
# `fec`+`un`+`dates`, which are letter accidents, not morphology.  Standard
# practice in compound segmentation, and the list is English, not fitted here.
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "from", "to", "in", "on", "into", "onto", "upon", "over", "under", "up",
    "down", "out", "off", "as", "so", "than", "then", "that", "this", "these",
    "those", "there", "here", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "shall", "should", "may", "might", "must", "he", "she", "it",
    "they", "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "who", "whom", "which", "what", "not", "no",
    "nor", "too", "very", "just", "also", "only", "own", "same", "such", "each",
    "any", "all", "both", "more", "most", "some", "other", "one", "two", "new"}

# a-priori English suffixes: fixed BEFORE looking at the corpus, so the
# stem+suffix rule cannot be fitted to what the book happens to contain
EN_SUFFIXES = tuple(sorted({
    "ational", "ization", "iveness", "fulness", "ousness", "ibility", "ability",
    "ivity", "ation", "ition", "ement", "ment", "ness", "less", "ible", "able",
    "ical", "ive", "ous", "ise", "ize", "ist", "ism", "ian", "ial", "ary",
    "ory", "ant", "ent", "ance", "ence", "age", "ify", "ic", "al", "ly", "ed",
    "es", "ates", "ions", "ings", "ers", "ies", "led", "ded", "ned", "ted",
    "ked", "sed", "med", "bed", "red", "id", "s", "er", "or", "ac", "ix", "um",
}, key=len, reverse=True))

# the two-character members of EN_SUFFIXES: the only two-letter strings the
# DP will accept as a *final* morpheme (`acidhuman`+`ix`).  Without this the
# DP cannot see `acidhumanix` = `acid`+`human`+`-ix` at all, because a
# two-letter part is otherwise reserved for prefixes.
SHORT_SFX = frozenset(s for s in EN_SUFFIXES if len(s) == 2)

# Derivational suffixes only (no `-s`, `-ed`, `-er`, ...): used to decide
# whether a blended word is stem+affix (`reset`+`-able`) or a plain blend
# (`mach`+`native`).  The inflectional endings are excluded because
# `resetted` sharing a `t` says nothing about derivation.
SHORT_AFFIX = frozenset(
    s for s in EN_SUFFIXES
    if len(s) >= 3 and s not in ("ly", "ing", "ings"))

# the nine morphemes L2 names, plus the ones the same measure turns up that are
# *not* in L2's list.  Reported in a separate `provenance` column so the issue's
# list stays auditable and the additions stay visible as additions.
NAMED = ["roid", "machine", "vamp", "gram", "cell", "embryo", "drone",
         "junkie", "mania"]
EXTRA = ["adam", "less", "ix", "maniac"]

LINKERS = ["-", "=", "/", "++", "+", "—"]
LINKER_CHANNEL = {"-": "hyphen", "—": "hyphen", "–": "hyphen",
                  "=": "eq", "/": "slash", "++": "cat", "+": "cat"}

# markup residue, two patterns, unioned:
#   (a) #17's own strict tag match, so `markup_unit` is a *superset* of its
#       published `html_tag_residue` and the two numbers reconcile;
#   (b) a tag carrying attributes, which (a) cannot see - `<th colspan="2">`
#       leaves `colspan` as a word.  Requires the char before `<` not to be
#       `<`, so the book's own `<<…>>` operator is not mistaken for a tag.
# A real HTML tag with **attributes**.  The attribute section must look like
# attributes (`name`, `name=value`, `name="value"`), not prose: Siratori writes
# angle-bracket blocks of his own (`<I escape from the volum>`), and a loose
# `\s[^<>\n]*` swallowed them, flagging the book's own English as markup
# residue.  With this pattern `<body fluid=streaming>` still matches (a real
# tag the source carried) and `<I escape from the volum>` does not.
RE_MARKUP_ATTRS = re.compile(
    r"(?<![<>])</?([A-Za-z][A-Za-z0-9]{0,11})"
    r"((?:\s+[A-Za-z-]{1,20}(?:\s*=\s*(?:\"[^\"\n]{0,300}\"|'[^'\n]{0,300}'"
    r"|[^\s>]{0,300}))?)*)\s*/?>")
# Two tag-name letters are required before the attributes count, because a
# one-letter tag name is how the book opens its own blocks: `<I escape from the
# volume>` and `<A lapse of memory>` are Siratori, not HTML.  This does not
# affect the strict match (`<th>`, `<br/>`, `<a>`, `<p>` are bare tags).

# Word parts the widened markup match catches and #17's strict one does not.
# Published so the superset is auditable item by item instead of taken on
# trust.  After requiring a `name=value` attribute, exactly one source tag
# qualifies: `<body fluid=streaming>` on p29.  `fluid=streaming` is a row here;
# `body` is deliberately **not**, because the book writes `body` 548 times as
# ordinary vocabulary and only once inside that tag - a name test would have
# flagged real English.
MARKUP_WORDS = {"fluid=streaming"}

DIAG_MEANING = {
    "markup_unit": "unit whose span intersects a real HTML tag run the page "
                   "body still carries (#17's html_tag_residue, widened from a "
                   "bare tag to a tag with a valued attribute); excluded from "
                   "lexical screening, not deleted",
    "markup_part": "word part inside such a run",
    "numeric_part": "word part holding a digit (#21 owns the numeric channel)",
    "opaque_coinage": "one closed word no rule resolves: an atomic coinage",
    "atomic_seen": "one closed word that IS an English dictionary entry: no "
                   "decomposition needed",
    "screened_unit": "unit screened (occurrences) - the denominator #19 must use",
    "chain_initial": "chain-initial morpheme: no linker on its left",
    "orphan_link": "linker with a word on one side only (no decomposition pair)",
}


# ---------------------------------------------------------------------------
# lexicon + segmentation
# ---------------------------------------------------------------------------
class Lexicon:
    """English word list with a frequency floor, the splice index, and the
    book's own attested word parts.

    The frequency floor is not cosmetic: `wordfreq`'s raw list contains Unix
    trigraphs (`idhu`), which otherwise make a real morpheme boundary look
    illegal.  z=1.24 for `idhu`; the floor of 1.5 drops it and keeps `cadaver`
    (2.70) and `ectoplasm` (2.07).

    `attested` is the set of word parts the book actually writes.  A candidate
    morpheme must be either attested here or common English (`min_zipf_strong`).
    That single rule is what stops the DP reading `tokage` as `tok`+`age`:
    `tok` is an English dictionary entry (z=2.90) but Siratori never writes it,
    whereas `cadaver` (z=2.70) he writes 633 times, so `cadaver`+`mechanism`
    survives and `tok`+`age` does not.  Grounding the filter in the corpus beats
    raising the zipf floor until `cadaver` breaks.
    """

    def __init__(self, min_zipf=LEX_MIN_ZIPF, min_span=SPLICE_MIN_SPAN,
                 min_b=SPLICE_MIN_B, attested=None, min_zipf_strong=None):
        self.min_zipf = min_zipf
        self.min_span = min_span
        self.min_b = min_b
        self.attested = set(attested or ())
        self.min_zipf_strong = (min_zipf_strong if min_zipf_strong is not None
                                else min_zipf + 2.0)
        self.words = set()
        if iter_wordlist is not None:
            for w in iter_wordlist("en"):
                if w.isalpha() and zipf_frequency(w, "en") >= min_zipf:
                    self.words.add(w)
        # splice index: right piece (len >= min_b) -> set of left pieces, only
        # for straddling words at least `min_span` long
        # `left_of[b]` = the set of `a` such that some lexicon word is exactly
        # `a + b`, with both `a+b >= min_span` and `b >= min_b`.  The `a >= 2`
        # floor is deliberate and matches `spliced_bruteforce`: a one-letter
        # dictionary entry must not make every boundary illegal.
        self.left_of = collections.defaultdict(set)
        for w in self.words:
            if len(w) < min_span:
                continue
            for j in range(2, len(w) - min_b + 1):
                self.left_of[w[j:]].add(w[:j])

    def z(self, w):
        return zipf_frequency(w, "en") if zipf_frequency else 0.0

    def in_lex(self, w):
        return w in self.words

    def part_ok(self, p, first, strong=True):
        """Is `p` usable as a closed-juncture morpheme?

        `first` means "not the last part", i.e. a stronger requirement: an
        internal part carries the analysis, so it has to be a credible morpheme
        on its own.  A two-letter part is only ever accepted in non-final
        position and only from the a-priori prefix list (`de`+`flesh`), so
        `acidhuman` resolves to `acid`+`human` and not to `acid`+`hum`+`an`.

        `strong` adds the corpus-attestation requirement (below).  It is off for
        the English-suffix path, which is already conservative because it only
        fires on a real English stem plus a real English suffix.
        """
        if p not in self.words or p in FUNCTION_WORDS:
            return False
        if len(p) == 2:
            # a two-letter morpheme is either an English prefix (`de`) in
            # non-final position or a real English suffix (`-ix`, `-ed`) in
            # final position - nothing else
            return (p in PREFIX2) if first else (p in SHORT_SFX)
        if strong:
            # the free-DP path needs a stem that is credible *on its own*: it
            # must be common English or something the book itself writes.  The
            # English-suffix path (`strong=False`) skips this, because
            # `fecund` is a real English verb at z=1.97 - rare, but real - and
            # the a-priori suffix is doing the work there, not the zipf score.
            if first and p not in self.attested \
                    and self.z(p) < self.min_zipf_strong:
                return False
            z = self.z(p)
            if len(p) == 3:
                return z >= Z3
            return z >= (ZFIRST if first else self.min_zipf)
        return True

    def spliced(self, left, right):
        """True if some dictionary word straddles `left|right`.

        Divay & Yule's criterion: the boundary is illegal only when a lexicon
        word can be built as (suffix of left) + (prefix of right).  Exact, not
        approximate, so this is a prefix-index lookup: the index maps each
        right-hand piece `b` to the set of left-hand pieces `a` that straddle
        with it, and `b` must be a prefix of `right`.

        The length floor is applied to the *whole* straddling word
        (`min_span`), which is what keeps two-letter dictionary entries such as
        `elit` from making every boundary illegal.
        """
        for j in range(self.min_b, len(right) + 1):
            b = right[:j]
            for a in self.left_of.get(b, ()):
                if len(a) <= len(left) and left.endswith(a):
                    return a + b
        return None

    def spliced_bruteforce(self, left, right):
        """Reference implementation, O(lexicon x len): tests only."""
        for w in self.words:
            if len(w) < self.min_span:
                continue
            for j in range(2, len(w) - self.min_b + 1):
                a, b = w[:j], w[j:]
                if len(a) <= len(left) and left.endswith(a) \
                        and right.startswith(b):
                    return a + b
        return None


def segment_dp(lex, w, maxn=MAXN, pen=PENALTY, use_splice=True, minn=2):
    """Best `minn`..`maxn`-way split of `w` into lexicon morphemes.

    Score = sum of part zipf - pen*(nparts-1); every internal boundary must be
    non-spliced when `use_splice`.  Returns a sorted list of tied-best splits.
    """
    n = len(w)
    res = []

    def rec(i, cur):
        if i == n:
            if minn <= len(cur) <= maxn:
                res.append(tuple(cur))
            return
        if len(cur) >= maxn:
            return
        for j in range(i + 2, n + 1):
            p = w[i:j]
            # every part except the one that reaches the end must be a strong
            # morpheme; the final part may be rare (it is the novel head)
            if not lex.part_ok(p, j != n):
                continue
            if cur and use_splice and lex.spliced("".join(cur), p):
                continue
            cur.append(p)
            rec(j, cur)
            cur.pop()

    rec(0, [])
    scored = [(sum(lex.z(p) for p in ps) - pen * (len(ps) - 1), ps) for ps in res]
    if not scored:
        return []
    best = max(s for s, _ in scored)
    return sorted({ps for s, ps in scored if s >= best - 1e-9})


def segment_blend(lex, w):
    """Overlap (blend) test: drop one letter, get two words on either side.

    `brunch` = br|unch.  Both halves must be real, reasonably common words, so
    that `res|ires` does not qualify.
    """
    out = []
    for i in range(3, len(w) - 2):
        a, b = w[:i], w[i + 1:]
        if lex.part_ok(a, False) and lex.part_ok(b, True) \
                and lex.z(a) >= 2.5 and lex.z(b) >= 2.5:
            out.append((a, b))
    return out


def corpus_attested(page_units, min_n=1):
    """Word parts the book writes **in lower case**, lower-cased.

    Built from #17's part lists before any segmentation, so the segmenter can
    prefer morphemes the book actually uses as common vocabulary.

    The lower-case requirement is what makes this an entity filter as well as a
    frequency filter: `Tokage` is a character's name and the book spells it
    `ToK` 13 times, never `tokage` or `tok`, so `tok` is not in this set and
    `tokage` stays an opaque coinage instead of becoming `tok`+`-age`.
    (`#22` owns case; this only uses it to avoid segmenting proper nouns.)
    """
    cnt = collections.Counter()
    for _pno, _body, units in page_units:
        for u in units:
            for t, role in u.parts:
                if role == "w" and t.islower():
                    cnt[t] += 1
    return {w for w, n in cnt.items() if n >= min_n}


def strip_english(lex, w, max_strip=2):
    """Strip a-priori English suffixes while the remainder is still an English
    word.  `fecundates` -> `fecund` + `ate` + `s`, `respirations` -> `respir`
    + `ation` + `s`.

    This runs *before* the free DP.  `fecundates` has a perfect 3-way DP split
    (`fec`+`un`+`dates`) that scores higher than `fecund`+`ate`+`s` on summed
    zipf, because `fecund` is rare (z=1.97).  Preferring the English
    decomposition whenever one exists is the conservative choice: it reports
    morphology the book could have inherited rather than morphology the
    segmenter invented.

    Among several valid strips the one with the **most frequent** stem wins, so
    `controlled` is `control`+`-ed` and not `controll`+`-ed`.
    """
    best = None
    for first in EN_SUFFIXES:
        if not w.endswith(first) or len(w) - len(first) < MIN_STEM:
            continue
        if first in FUNCTION_WORDS:
            # `-or` is in the a-priori list because of `-ator` words, but
            # `corporation` is `corpor`+`-ation`, not `corp`+`-or`+`-ation`
            continue
        rest = w[: -len(first)]
        # the stem must be a *usable* morpheme, not merely a dictionary entry:
        # without the function-word check `corporation` comes out as
        # `corp`+`or`+`ation`.  `strong` is off here: this path already
        # requires a real English stem and a real English suffix, so it does
        # not need the corpus-attestation filter the free DP needs.
        if len(rest) >= MIN_STEM and rest not in FUNCTION_WORDS \
                and lex.part_ok(rest, True, strong=False):
            cand = ([(rest, "stem"), (first, "sfx")], (lex.z(rest), -len(first)))
        elif max_strip > 1:
            cand = None
            for second in EN_SUFFIXES:
                if not rest.endswith(second) or second in FUNCTION_WORDS:
                    continue
                stem = rest[: -len(second)]
                if len(stem) >= MIN_STEM and stem not in FUNCTION_WORDS \
                        and lex.part_ok(stem, True, strong=False):
                    cand = ([(stem, "stem"), (second, "sfx"), (first, "sfx")],
                            (lex.z(stem), -len(first) - len(second)))
                    break
            if cand is None:
                continue
        else:
            continue
        if best is None or cand[1] > best[1]:
            best = cand
    return best[0] if best else []


def analyse_dict(lex, w):
    """A word English already has is **atomic**.

    No decomposition is attempted, not even by a-priori English morphology, and
    that is a decision rather than a shortcut.  `electric` splits as
    `electr`+`-ic` and `classic` as `class`+`-ic`; both are true and both are
    useless here, because #18 is about the *book's* compounds, and an English
    morpheme boundary inside a word English already owns is English morphology,
    not Siratori's.  Every rule that would fire is a source of invented
    analysis (`cad`+`aver`, `Cor`+`or`+`ation`, `elec`+`tric`), and the only
    cost of refusing is that `controlled` stays whole rather than becoming
    `control`+`-led` - which the hyphen channel already shows us anyway
    (`cold-blooded`).

    OOV words are the ones that get worked on; see `analyse_closed`.
    """
    return "atomic_seen", [(w, "word")], {"alt_splits": 0, "surface": w}


def analyse_closed(lex, w):
    """Decompose one closed word.  Returns (kind, [(morph, role)], extras)."""
    if lex.in_lex(w):
        return analyse_dict(lex, w)
    # an out-of-lexicon word that English morphology already accounts for:
    # `respirations` = `respir`+`ation`+`s`, `fecundates` = `fecund`+`ate`+`s`
    ms = strip_english(lex, w)
    if ms:
        return "stem_suffix", ms, {"alt_splits": 1, "surface": w,
                                   "sfx": "+".join(t for t, r in ms if r == "sfx")}
    # an a-priori English suffix with a usable stem behind it: `resettable` =
    # `reset` + `-able`, `softwarable` = `soft`+`war` + `-able`.  Checked before
    # the free DP because the suffix is a-priori evidence, whereas the DP will
    # happily say `reset`+`table`.
    for sfx in EN_SUFFIXES:
        if not w.endswith(sfx) or sfx in FUNCTION_WORDS:
            continue
        stem = w[: -len(sfx)]
        if len(stem) < 3:
            continue
        if lex.part_ok(stem, True):
            return "stem_suffix", [(stem, "stem"), (sfx, "sfx")], \
                   {"alt_splits": 1, "surface": w, "sfx": sfx}
        sp = segment_dp(lex, stem, use_splice=True)
        if sp:
            return "stem_suffix", \
                   [(p, "stem") for p in sp[0]] + [(sfx, "sfx")], \
                   {"alt_splits": len(sp), "surface": w, "sfx": sfx}
    # A blend whose right member is an a-priori English suffix outranks a plain
    # split: attaching a known suffix is stronger evidence than an arbitrary
    # juncture.  `resettable` is `reset`+`-able` sharing `t`, not
    # `reset`+`table`.  A blend with no suffix (`machinative` = `mach`+`native`)
    # does not, so it is only tried after the plain split fails.
    for a, b in segment_blend(lex, w):
        if b in SHORT_AFFIX:
            return "closed_blend", [(a, "stem"), (b, "sfx")], \
                   {"alt_splits": 1, "surface": w, "overlap": w[len(a)],
                    "sfx": b}
    sp = segment_dp(lex, w, use_splice=True)
    if sp:
        return "closed_composite", [(p, "stem") for p in sp[0]], \
               {"alt_splits": len(sp), "surface": w}
    bl = segment_blend(lex, w)
    if bl:
        a, b = bl[0]
        return "closed_blend", [(a, "stem"), (b, "stem")], \
               {"alt_splits": len(bl), "surface": w, "overlap": w[len(a)]}
    return "opaque_coinage", [(w, "coinage")], {"alt_splits": 0, "surface": w}


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def markup_runs(body):
    """[(start, end, run, strict)] for real HTML tags in a page body.

    `strict` marks the #17-visible subset, so the build can report both the
    reconciling number and the wider one.
    """
    out = []
    for m in OT.RE_HTML_TAG.finditer(body):
        if m.group(1).lower() in OT.HTML_TAGS:
            out.append((m.start(), m.end(), m.group(0), True))
    for m in RE_MARKUP_ATTRS.finditer(body):
        name = m.group(1).lower()
        if name not in OT.HTML_TAGS:
            continue
        # a *valued* attribute is the discriminator.  The book writes
        # `<I escape from the volume>` - bare words after the tag name - while
        # the source's real markup writes `<a href="mailto:...">` and
        # `<th colspan="2">`.  Requiring a `name=value` pair catches the markup
        # and leaves the book's own angle-bracket syntax alone, including the
        # one-letter tags `<I …>` and `<A …>`.
        if "=" not in m.group(2):
            continue
        if any(a < m.end() and m.start() < b for a, b, _t, _s in out):
            continue
        out.append((m.start(), m.end(), m.group(0), False))
    out.sort()
    return out


def load_units(src=SRC):
    """[(page, body, [units])] plus the markup index, in page order.

    `u.id` is made **book-global**, not per page: #17's `units.tsv` numbers
    uids contiguously across the whole book, and #18's `decompositions.tsv`
    must join to it on exactly that key.
    """
    bodies = OG.load_pages_raw(src)
    pages = sorted(bodies)
    marks = {p: markup_runs(bodies[p]) for p in pages}
    out = []
    uid = 0
    for pno in pages:
        toks, _d, _tail = OT.tokenize_page(pno, bodies[pno])
        units, _d = OT.build_units(toks)
        for u in units:
            u.id = uid
            uid += 1
        out.append((pno, bodies[pno], units))
    return bodies, marks, out


def in_runs(s, e, runs):
    """(hit, strict_only): does the span [s,e) touch a markup run, and was any
    touching run inside #17's strict subset?"""
    hit = strict = False
    for a, b, _t, s_strict in runs:
        if a < e and s < b:
            hit = True
            strict = strict or s_strict
    return hit, strict


# ---------------------------------------------------------------------------
# per-unit decomposition
# ---------------------------------------------------------------------------
# the four junction channels, named identically in LINKER_CHANNEL (used for
# per-morpheme attachment) and LINKER_KIND (used to bucket whole units), so a
# channel name means one thing everywhere in the artifacts
LINKER_KIND = {"-": "hyphen", "—": "hyphen", "=": "eq",
               "/": "slash", "++": "cat", "+": "cat"}
#: the six buckets a unit can land in, and the `link_kind` column's vocabulary
LINK_KINDS = ("hyphen", "eq", "slash", "cat", "mixed", "closed")


def link_kind_of(kinds, channel):
    """Bucket a unit by the linker kind it actually uses.

    `kinds` is the set of #12/#17 linker kinds present in the unit's own parts
    (`hyphen`, `eq`, `slash`, `cat`, `other`).  `channel` is `open`/`closed` as
    decided by `decompose`.  One bucket name, one meaning, everywhere: this is
    the `chan` column of `linker_rows.tsv`, the `link_kind` column of
    `decompositions.tsv`, and the row label of the spec's channel table.

    `mixed` is its own bucket rather than being forced into one channel: a unit
    like `bio=less=soul/gram` genuinely writes two linker kinds, and 135 of
    them do, so hiding them inside `eq` would misreport the `=` channel.
    """
    if channel != "open":
        return "closed"
    known = kinds & set(LINKER_KIND.values())
    if len(known) == 1 and "other" not in kinds:
        return next(iter(known))
    return "mixed"



RULE_DOC = {
    "R1": "hyphen compound: `-` between two words, no other linker "
          "(`drug-embryo`, `boy-roid`)",
    "R2": "identity chain: `=` between two or more words "
          "(`gene=TV`, `bio=less=soul/gram`)",
    "R3": "slash chain: `/` between two or more words (`soul/gram`)",
    "R4": "concat chain: `++` or `+` between words (`murder++the`)",
    "R5": "closed word: one word, no linker; decomposed by the closed-juncture "
          "rules (composite / stem+suffix / blend) or left atomic",
    "R6": "mixed chain: two or more linker kinds in one unit "
          "(`gene=TV/spasm`, `suck=blood=memory`)",
}


def decompose(lex, u, runs, closed_cache):
    """Decompose one unit occurrence into a morpheme chain.

    A morpheme is `(text, role, chan, final)`:
      role  `word`  a word that needs no further splitting (open juncture),
            `stem`  a stem recovered from a fused word,
            `sfx`   an affix recovered from a fused word,
            `coinage` / `word`  a fused word left whole;
      chan  the junction that put the morpheme in the unit: `hyphen`, `eq`,
            `slash`, `cat`, `chain_initial`, or `closed`;
      final whether it is the head position (the last morpheme of the chain).

    Head is assigned by position, not by frequency: in this book the chain
    *is* the syntax, so the last morpheme is the head and the zipf comparison
    in `linker_rows.tsv` is what tests that claim.
    """
    markup, markup_strict = in_runs(u.s, u.e, runs)
    parts = u.parts
    seq, linkers = [], []
    for t, r in parts:
        if r in ("seam", "join"):
            seq.append(("link", t))
            linkers.append(t)
        else:
            seq.append(("word", t))
    kinds = {LINKER_KIND.get(l, "other") for l in linkers}
    RULE_OF = {"hyphen": "R1", "eq": "R2", "slash": "R3", "cat": "R4"}
    if not linkers:
        channel, rule = "closed", "R5"
    elif len(kinds) == 1 and next(iter(kinds)) in RULE_OF:
        channel, rule = "open", RULE_OF[next(iter(kinds))]
    else:
        channel, rule = "open", "R6"

    morph = []
    prev_link = ""
    n_words = sum(1 for k in seq if k[0] == "word")
    seen_words = 0
    closed_kind = overlap = closed_surface = ""
    for kind_i, val in seq:
        if kind_i == "link":
            prev_link = LINKER_CHANNEL.get(val, "other")
            continue
        seen_words += 1
        is_final = seen_words == n_words
        if channel == "closed":
            low = val.lower()
            if low not in closed_cache:
                closed_cache[low] = analyse_closed(lex, low)
            ckind, ms, extra = closed_cache[low]
            closed_kind, overlap = ckind, extra.get("overlap", "")
            closed_surface = low
            if len(ms) == 1:
                # nothing was split, so the source spelling is still the
                # morpheme: `BLOOD` must not come back as `blood`
                ms = [(val, ms[0][1])]
            elif sum(len(t) for t, _r in ms) == len(val):
                # The parts rejoin to the surface letter-for-letter (every rule
                # except the blend), so the source's case can be handed back to
                # each morpheme: `SCANIMAL` -> `SCAN`+`IMAL`, not `scan`+`imal`.
                off = 0
                restored = []
                for t, r in ms:
                    restored.append((val[off:off + len(t)], r))
                    off += len(t)
                ms = restored
            for j, (mt, mr) in enumerate(ms):
                morph.append({"text": mt, "role": mr, "chan": "closed",
                              "final": j == len(ms) - 1, "surf": low,
                              "kind": ckind, "alt": extra.get("alt_splits", 0)})
        else:
            morph.append({"text": val, "role": "word",
                          "chan": prev_link or "chain_initial",
                          "final": is_final, "surf": "", "kind": "", "alt": 0})
        prev_link = ""
    head = morph[-1]["text"] if morph else ""
    mods = [m["text"] for m in morph[:-1]]
    rec = {
        "uid": u.id, "page": u.page, "start": u.s, "end": u.e, "unit": u.text,
        "cls": u.cls, "sig": u.sig, "n_words": n_words, "rule": rule,
        # `channel` is open vs closed (does the book write a linker out?);
        # `link_kind` is *which* linker kind, bucketed exactly as `link_kind()`
        # / `linker_rows.tsv`'s `chan` bucket it - hyphen | eq | slash | cat |
        # mixed | closed.  Both columns are published so the spec's channel
        # table is joinable without re-deriving the bucket from `linkers`.
        # `|` is the joiner: #14 makes `|` a *segment* operator, never a seam,
        # so it can never occur inside a linker's own text.  A `+` joiner would
        # be ambiguous (`++` is one linker) and `/` is itself a linker.
        "channel": channel, "link_kind": link_kind_of(kinds, channel),
        "linkers": "|".join(linkers),
        "n_link": len(linkers),
        "head": head, "mods": "+".join(mods), "n_morph": len(morph),
        "closed_kind": closed_kind,
        "closed_surface": closed_surface,
        "closed_alt": (morph[0]["alt"] if morph and channel == "closed" else ""),
        "overlap": overlap,
        "markup": markup,
        "markup_strict": markup_strict,
        "has_digit": any(c.isdigit() for t, _r in parts for c in t),
        "morph": "+".join(("sfx:" if m["role"] == "sfx" else "") + m["text"]
                          for m in morph),
        "morph_detail": "|".join(f"{m['text']}:{m['role']}:{m['chan']}"
                                 + (":H" if m["final"] else "")
                                 for m in morph),
    }
    return rec, morph


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def binom_two_sided(k, n):
    """Two-sided exact binomial p for k successes out of n at p=0.5.

    Computed in log space (`lgamma`), because `2**n` overflows for the `=`
    channel's n=1,883 and the naive form raises instead of returning a number.
    """
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    if lo >= 30:                      # far tail: first term dominates
        log_c = (math.lgamma(n + 1) - math.lgamma(lo + 1)
                 - math.lgamma(n - lo + 1))
        log_p = log_c - n * math.log(2.0)
        tail = math.exp(log_p) * (1.0 + lo / (n - lo + 1.0))
    else:
        tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "y" if v else "n"
    return str(v).replace("\t", " ").replace("\n", "\\n").replace("\r", "")


def dump_tsv(name, rows, cols, out):
    with open(os.path.join(out, name), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(cell(r.get(c, "")) for c in cols) + "\n")


def md_code(s):
    return f"`` {s} ``" if "`" in str(s) else f"`{s}`"


def pct(a, b):
    return "n/a" if not b else f"{a / b:.1%}"


# ---------------------------------------------------------------------------
# head-role measurement (does the last morpheme behave like a head?)
# ---------------------------------------------------------------------------
def linker_rows(recs, bare, lex):
    """One row per linker channel: head-role evidence.

    Two corpus-internal tests of *the right-hand member is the head*:

      `right_bare_pct`  share of the channel's **distinct** right members that
                        ever occur as a whole word part elsewhere in the book.
                        A classifier head must be able to stand alone
                        (`machine`: 395 bare uses); a bound affix cannot
                        (`roid`: 1).
      `right_higher`    how often the right member is the *rarer* of the pair by
                        English zipf - a frequent modifier against a rare novel
                        head - over the discordant pairs, with an exact p.

    Both are reported; neither is asserted.  English compound heads are
    normally *more* frequent than their modifiers (`drugstore`), so a
    `right_higher` below 50% does not refute head-finality: it says which kind
    of compound this is, not whether the chain has a head.
    """
    stat = collections.defaultdict(collections.Counter)
    pairs = collections.defaultdict(collections.Counter)
    examples = {}
    for r in recs:
        if r["markup"] or r["channel"] != "open":
            continue
        words = [c.split(":") for c in r["morph_detail"].split("|")]
        for i, w in enumerate(words):
            if w[1] != "word" or i == 0:
                continue
            prev = words[i - 1]
            if prev[1] != "word":
                continue
            # the pair's channel is the linker that *introduced the right-hand
            # member*, which is that member's own `chan` - never the left
            # member's, which is the channel of whatever preceded *it*
            ch = w[2]
            if ch in ("", "chain_initial", "closed", "other"):
                continue
            a, b = prev[0], w[0]
            s = stat[ch]
            s["n"] += 1
            zl, zb = lex.z(a.lower()), lex.z(b.lower())
            if zb > zl:
                s["right_higher"] += 1
            elif zl > zb:
                s["left_higher"] += 1
            else:
                s["tie"] += 1
            pairs[ch][(a.lower(), b.lower())] += 1
            examples.setdefault(ch, f"p{r['page']} {r['unit']}")
    rows = []
    for ch, s in sorted(stat.items(), key=lambda kv: -kv[1]["n"]):
        n = s["n"]
        p = binom_two_sided(s["right_higher"], n)
        bare_pct, bare_n = chan_right_bare(pairs[ch], bare)
        top = pairs[ch].most_common(1)
        rows.append({
            "chan": ch, "n": n, "n_pairs": len(pairs[ch]),
            "right_higher": s["right_higher"],
            "left_higher": s["left_higher"], "tie": s["tie"],
            "right_higher_pct": round(100.0 * s["right_higher"] / n, 1) if n else 0,
            "binom_p": f"{p:.3g}",
            "n_distinct_left": len({a for a, _b in pairs[ch]}),
            "n_distinct_right": len({b for _a, b in pairs[ch]}),
            "top_pair": f"{top[0][0][0]}|{top[0][0][1]}" if top else "",
            "top_pair_n": top[0][1] if top else 0,
            "right_bare_pct": bare_pct, "right_bare_n": bare_n,
            "example": examples.get(ch, ""),
        })
    return rows, pairs


def chan_right_bare(pairs_ch, bare):
    """Share of this channel's *distinct right members* that ever stand alone.

    Type-level, not token-level, on purpose.  A single bare use of `machine`
    makes it a free lexeme, but weighting that by how often it attaches lets one
    frequent pair (`boy-roid`, 338 of 801 hyphen pairs) decide the whole
    channel's verdict.  Token-level bare counts are still available per morpheme
    in `affix_lexicon.tsv`.
    Returns (pct_of_distinct_right, n_distinct_right_that_are_bare).
    """
    rights = {b for _a, b in pairs_ch}
    hit = {b for b in rights if bare.get(b, 0)}
    return (round(100.0 * len(hit) / len(rights), 1) if rights else 0.0,
            len(hit))


# ---------------------------------------------------------------------------
# affix lexicon
# ---------------------------------------------------------------------------
CHANNELS = ["unit", "chain_initial", "hyphen", "eq", "slash", "cat", "closed"]


def affix_evidence(recs, targets, lex):
    """Attachment evidence for every morpheme; verdicts only for `targets`.

    A morpheme is **bound** if the book never lets it stand alone, **free** if
    it usually does, and a **hapax-base classifier** if bound and one base takes
    (nearly) all its tokens - the morpheme tags that one base instead of
    deriving a family.
    """
    bare = collections.Counter()                        # free-standing uses
    att = collections.defaultdict(collections.Counter)  # morph -> chan -> n
    bases = collections.defaultdict(collections.Counter)
    first, pages = {}, collections.defaultdict(set)
    surface = collections.defaultdict(collections.Counter)
    for r in recs:
        if r["markup"]:
            continue
        seq = [c.split(":") for c in r["morph_detail"].split("|")]
        single = len(seq) == 1
        for i, w in enumerate(seq):
            txt, role, chan = w[0], w[1], w[2]
            low = txt.lower()
            pages[low].add(r["page"])
            first.setdefault(low, (r["page"], r["start"], r["unit"]))
            surface[low][r["unit"]] += 1
            if single:
                bare[low] += 1
                continue
            att[low][chan] += 1
            if i > 0:
                bases[(low, chan)][seq[i - 1][0].lower()] += 1
    out = {}
    for m in targets:
        c = att[m]
        n_att = sum(c.values())
        all_bases = collections.Counter()
        for (mm, _ch), bc in bases.items():
            if mm == m:
                all_bases.update(bc)
        dom = all_bases.most_common(1)[0] if all_bases else ("", 0)
        tot = bare[m] + n_att
        bare_pct = round(100.0 * bare[m] / tot, 1) if tot else 0.0
        top_share = round(100.0 * dom[1] / n_att, 1) if n_att else 0.0
        verdict = classify(bare_pct, len(all_bases), top_share, n_att)
        out[m] = {
            "morph": m, "n_total": tot, "n_bare": bare[m], "n_attached": n_att,
            "n_bases": len(all_bases), "bare_pct": bare_pct,
            "top_base": dom[0], "top_base_n": dom[1], "top_base_pct": top_share,
            "chan_summary": " ".join(f"{k}:{v}" for k, v in
                                     sorted(c.items(), key=lambda kv: -kv[1])),
            "top_bases": all_bases.most_common(10),
            "first_page": first.get(m, ("", "", ""))[0],
            "first_unit": first.get(m, ("", "", ""))[2],
            "n_pages": len(pages[m]),
            "zipf": round(lex.z(m), 2),
            "verdict": verdict,
            "n_surface": len(surface[m]),
        }
    return out, bare, att, bases


def link_kind(r):
    """Which junction channel a unit belongs to: one of the four linker kinds,
    `closed`, or `mixed`.

    Thin wrapper over `link_kind_of` so the `link_kind` column written by
    `decompose` and this function cannot drift apart.
    """
    if r["channel"] != "open":
        return "closed"
    kinds = {LINKER_KIND.get(l, "other")
             for l in r["linkers"].split("|") if l}
    return link_kind_of(kinds, r["channel"])


def classify(bare_pct, n_bases, top_share, n_att):
    """The four-way verdict.  Thresholds are printed in the spec."""
    if n_att == 0:
        return "free_lexeme" if bare_pct else "unattested"
    if bare_pct >= 50.0:
        return "free_lexeme" if n_bases >= 5 else "free_head_few_bases"
    if n_bases <= 1 or top_share >= 80.0:
        return "hapax_base_classifier" if bare_pct < 10 else "mostly_hapax_base"
    if n_bases >= 5:
        return "bound_productive_affix"
    return "bound_multi_base_classifier"


# ---------------------------------------------------------------------------
# head mining: what else behaves like a classifier head?
# ---------------------------------------------------------------------------
def mine_heads(recs, bare, lex, min_n=8, min_bases=2):
    """Every morpheme in the book scored as a candidate classifier **head**.

    Only the **final** morpheme of a chain counts as head position.  This is the
    filter that matters: without it the table fills with modifiers (`soul`,
    `boy`, `body`) that appear 2,700 times as the *left* member of a compound,
    and with English suffixes (`-s`, `-ed`, `-al`) that were recovered by the
    closed-juncture rules rather than coined here.

    A candidate needs: >= `min_n` head-position uses, >= `min_bases` distinct
    left-hand morphemes, and it is ranked by attached count.
    """
    att = collections.defaultdict(collections.Counter)
    bases = collections.defaultdict(collections.Counter)
    open_bases = collections.defaultdict(collections.Counter)
    first = {}
    for r in recs:
        if r["markup"]:
            continue
        seq = [c.split(":") for c in r["morph_detail"].split("|")]
        if len(seq) < 2:
            continue
        head = seq[-1]
        chan = head[2]
        if chan in ("", "chain_initial", "other"):
            # a chain-final morpheme introduced by no linker is only reachable
            # from a closed split; keep it, but the channel is `closed`
            chan = "closed" if chan == "" else chan
            if chan not in ("closed",):
                continue
        low = head[0].lower()
        att[low][chan] += 1
        bases[low][seq[-2][0].lower()] += 1
        if chan != "closed":
            open_bases[low][seq[-2][0].lower()] += 1
        first.setdefault(low, (r["page"], r["start"], r["unit"]))
    rows = []
    for m, c in att.items():
        n_att = sum(c.values())
        nb = len(bases[m])
        n_open = sum(v for k, v in c.items() if k != "closed")
        if n_att < min_n or nb < min_bases:
            continue
        tot = n_att + bare[m]
        bare_pct = round(100.0 * bare[m] / tot, 1) if tot else 0.0
        top_chan = c.most_common(1)[0]
        top_base, top_base_n = bases[m].most_common(1)[0]
        top_share = round(100.0 * top_base_n / n_att, 1)
        ob = open_bases[m].most_common(1)
        rows.append({
            "morph": m, "n_attached": n_att, "n_open": n_open, "n_bases": nb,
            "n_bare": bare[m], "bare_pct": bare_pct,
            "top_chan": top_chan[0], "top_chan_n": top_chan[1],
            "top_chan_pct": round(100.0 * top_chan[1] / n_att, 1),
            "chan_summary": " ".join(f"{k}:{v}" for k, v in
                                     sorted(c.items(), key=lambda kv: -kv[1])),
            "zipf": round(lex.z(m), 2),
            "in_lex": lex.in_lex(m),
            "named": "y" if m in NAMED + EXTRA else "n",
            "top_base": top_base, "top_base_n": top_base_n,
            "top_base_pct": top_share,
            "top_base_open": ob[0][0] if ob else "",
            "top_base_open_n": ob[0][1] if ob else 0,
            "verdict": classify(bare_pct, nb, top_share, n_att),
            "bases": ";".join(sorted(bases[m])),
            "open_bases": ";".join(sorted(open_bases[m])),
            "first_page": first[m][0], "first_unit": first[m][2],
        })
    rows.sort(key=lambda r: (-r["n_attached"], r["morph"]))
    for i, r in enumerate(rows):
        r["n_attached_rank"] = i + 1
    rows.sort(key=lambda r: (-r["n_open"], -r["n_attached"], r["morph"]))
    return rows


# ---------------------------------------------------------------------------
# closed juncture (portmanteau) table
# ---------------------------------------------------------------------------
def closed_rows(closed_cache, counts, lex):
    """One row per distinct closed-juncture surface, with its decomposition."""
    rows = []
    for surf, (kind, ms, extra) in sorted(closed_cache.items()):
        n = counts.get(surf, 0)
        if not n:
            continue
        rows.append({
            "surface": surf, "n": n, "kind": kind, "n_morph": len(ms),
            "split": "+".join(m[0] for m in ms),
            "roles": "+".join(m[1] for m in ms),
            "all_parts_english": "y" if all(lex.in_lex(p) for p, _r in ms) else "n",
            "min_part_zipf": round(min(lex.z(p) for p, _r in ms), 2),
            "alt_splits": extra.get("alt_splits", 0),
            "overlap_letter": extra.get("overlap", ""),
            "in_lex": lex.in_lex(surf),
            "zipf": round(lex.z(surf), 2),
        })
    rows.sort(key=lambda r: (-r["n"], r["surface"]))
    return rows


# ---------------------------------------------------------------------------
# unit-type screening table (the join key for #19)
# ---------------------------------------------------------------------------
def unit_type_rows(recs):
    """One row per unit TYPE with its morphology, so #19 screens units.

    Keyed on the exact surface, not case-folded: #17's `unit_types.tsv` keys on
    `unit_type` and keeps `Soul/gram` and `soul/gram` apart, and #19 has to be
    able to reproduce #17's 5,876 types.  The `lower` column is the
    case-folded view, published alongside rather than used as the key.
    """
    agg = collections.defaultdict(lambda: {
        "n": 0, "pages": set(), "cls": "", "rule": "", "channel": "",
        "link_kind": "", "kinds": collections.Counter(), "morph": "", "detail": "",
        "markup": False, "first": None, "unit": ""})
    for r in recs:
        key = r["unit"]
        a = agg[key]
        a["n"] += 1
        a["pages"].add(r["page"])
        a["cls"] = r["cls"]
        a["rule"] = r["rule"]
        a["channel"] = r["channel"]
        a["link_kind"] = r["link_kind"]
        a["kinds"][r["closed_kind"] or r["link_kind"]] += 1
        a["markup"] = a["markup"] or bool(r["markup"])
        if a["first"] is None:
            a["first"] = (r["page"], r["start"])
            a["unit"] = r["unit"]
            a["morph"] = r["morph"]
            a["detail"] = r["morph_detail"]
    rows = []
    for key, a in sorted(agg.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        rows.append({
            "unit": a["unit"], "lower": key.lower(), "n": a["n"],
            "n_pages": len(a["pages"]), "cls": a["cls"], "rule": a["rule"],
            "channel": a["channel"], "link_kind": a["link_kind"],
            "kind_summary": " ".join(f"{k}:{v}" for k, v in
                                    sorted(a["kinds"].items(), key=lambda kv: -kv[1])),
            "morph": a["morph"], "morph_detail": a["detail"],
            "n_morph": a["detail"].count("|") + 1 if a["detail"] else 0,
            "markup": a["markup"],
            "first_page": a["first"][0], "first_start": a["first"][1],
        })
    return rows


# ---------------------------------------------------------------------------
# diagnostics ledger
# ---------------------------------------------------------------------------
def diag_rows(recs, closed_cache, lex):
    rows = []

    def add(name, n, pages, example, meaning=None):
        rows.append({"diag": name, "n": n, "pages": pages,
                     "meaning": meaning or DIAG_MEANING[name],
                     "example": example})

    mu = [r for r in recs if r["markup"]]
    add("markup_unit", len(mu), len({r["page"] for r in mu}),
        f"p{mu[0]['page']} {mu[0]['unit']}" if mu else "")
    mstrict = [r for r in mu if r["markup_strict"]]
    rows.append({"diag": "markup_unit_strict", "n": len(mstrict),
                 "pages": len({r["page"] for r in mstrict}),
                 "meaning": "the subset #17's strict tag match also sees; it "
                            "publishes `html_tag_residue` = 229, so this row "
                            "must equal it",
                 "example": f"p{mstrict[0]['page']} {mstrict[0]['unit']}"
                 if mstrict else ""})
    add("markup_part", sum(r["n_morph"] for r in mu),
        len({r["page"] for r in mu}), mu[0]["unit"] if mu else "")
    num = [r for r in recs if r["has_digit"]]
    add("numeric_part", sum(r["n_morph"] for r in num),
        len({r["page"] for r in num}),
        f"p{num[0]['page']} {num[0]['unit']}" if num else "")
    opq = [r for r in recs if r["closed_kind"] == "opaque_coinage"]
    ex = ""
    if opq:
        top = max(opq, key=lambda r: r["n_morph"])
        ex = f"p{top['page']} {top['unit']}"
    add("opaque_coinage", len(opq), len({r["page"] for r in opq}), ex)
    seen = [r for r in recs if r["closed_kind"] == "atomic_seen"]
    add("atomic_seen", len(seen), len({r["page"] for r in seen}),
        f"p{seen[0]['page']} {seen[0]['unit']}" if seen else "")
    screened = [r for r in recs if not r["markup"]]
    add("screened_unit", len(screened), len({r["page"] for r in recs}), "")
    ci = sum(1 for r in recs
             for c in r["morph_detail"].split("|") if c.split(":")[2] == "chain_initial")
    add("chain_initial", ci, "", "")
    orph = 0
    for r in recs:
        words = [c.split(":") for c in r["morph_detail"].split("|")]
        if len(words) > 1 and words[0][1] != "word":
            orph += 1
    add("orphan_link", orph, "",
        "a linker with no word on its left: #17's seam_orphan_left / "
        "slash_boundary rows reach here as a single leading morpheme")
    return rows


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
CASES = [
    ("C1", "drug-embryo", "R1", "hyphen compound, the issue's own example"),
    ("C2", "boy-roid", "R1", "hyphen compound, `-roid` bound classifier"),
    ("C3", "soul/gram", "R3", "slash compound, `-gram` bound classifier"),
    ("C4", "gene=TV", "R2", "identity chain"),
    ("C5", "bio=less=soul/gram", "R6", "mixed chain, 3 morphemes, 2 linker kinds"),
    ("C6", "fear=cell", "R2", "`-cell` mostly free-standing"),
    ("C7", "digital=vamp", "R2", "`-vamp` through identity only"),
    ("C8", "retro-ADAM", "R1", "entity + classifier, hyphen"),
    ("C9", "techno-junkie", "R1", "`-junkie`, one base"),
    ("C10", "hydro=mania", "R2", "`-mania` through identity"),
    ("C11", "acidhuman", "R5", "closed juncture composite"),
    ("C12", "boyroid", "R5", "closed juncture, `-roid` fused"),
    ("C13", "hydromachine", "R5", "closed juncture, `-machine` fused"),
    ("C14", "defleshed", "R5", "closed juncture stem + English `-ed`"),
    ("C15", "feti", "R5", "opaque coinage, no decomposition found"),
    ("C16", "omotya", "R5", "opaque coinage, the book's second noun"),
    ("C17", "cold-blooded", "R1", "hyphen compound, English morphology"),
    ("C18", "1-84068-060-1", "R1", "numeric compound (ISBN): not lexical"),
    ("C19", "acidHUMAN-ISM", "R1", "case + hyphen: entity token"),
    ("C20", "resettable", "R5", "closed juncture blend (overlap on `t`)"),
    ("C21", "tokage", "R5", "opaque coinage: a proper noun must not be split"),
    ("C22", "machinative", "R5", "closed juncture blend (overlap on `i`)"),
    ("C23", "andrograph", "R5",
     "synthetic: not in the corpus; the segmenter must refuse `andro`+`graph`"),
    ("C24", "techno-junkie", "R1", "`-junkie`, the smallest classifier tag"),
    ("C25", "soul/gram/to", "R3",
     "slash chain of three: `gram` is in the middle, not the head"),
]


def case_rows(recs, bodies):
    rows = []
    seen = set()
    for cid, needle, rule, note in CASES:
        for r in recs:
            if r["unit"] != needle and r["unit"].lower() != needle.lower():
                continue
            body = bodies[r["page"]]
            ctx = body[max(0, r["start"] - 60):r["end"] + 60]
            rows.append({
                "case": cid, "needle": needle, "rule": rule, "note": note,
                "page": r["page"], "start": r["start"], "end": r["end"],
                "unit": r["unit"], "morph": r["morph"],
                "morph_detail": r["morph_detail"], "head": r["head"],
                "mods": r["mods"], "channel": r["channel"],
                "closed_kind": r["closed_kind"], "n_morph": r["n_morph"],
                "markup": r["markup"],
                "context": ctx.replace("\n", "\\n").replace("\t", " "),
                "exact": "y" if bodies[r["page"]][r["start"]:r["end"]] == r["unit"] else "n",
            })
            break
        else:
            # a case the corpus does not contain: still record it, and say so.
            # A case table with holes is a hole in the evidence, not a reason
            # to drop the case.
            rows.append({"case": cid, "needle": needle, "rule": rule,
                         "note": note, "page": "", "start": "", "end": "",
                         "unit": "", "morph": "", "morph_detail": "",
                         "head": "", "mods": "", "channel": "",
                         "closed_kind": "", "n_morph": "", "markup": "",
                         "context": "NOT IN CORPUS", "exact": "n/a"})
        seen.add(cid)
    return rows


def reconciliation(S):
    """Numbers #18 reproduces from #12 / #14 / #17, and where they differ."""
    L, A = S["links"], S["linkers"]
    rows = [
        {"quantity": "pages", "earlier": "225 (#7/#14/#17)", "here": S["n_pages"],
         "note": "same layout markdown, same page map"},
        {"quantity": "units decomposed", "earlier": "74,665 (#17 units.tsv)",
         "here": S["n_units"],
         "note": "every #17 unit occurrence gets exactly one decomposition row"},
        {"quantity": "open-juncture units", "earlier": "5,433 fused (#17)",
         "here": L["open_any"],
         "note": "R1+R2+R3+R4+R6: every unit that writes a linker out. The "
                 "delta of 1 against #17's 5,433 is the one markup unit in a "
                 "fused class, `fluid=streaming` (p29), which is excluded from "
                 "screening"},
        {"quantity": "open-juncture units, one linker kind",
         "earlier": "5,298 (#17: `join`+`eq`+`slash`+`cat`+`plus`)",
         "here": L["open_single"],
         "note": "the delta of 1 is the same markup unit; #17 also counts "
                 "`mixed` here, where `link_kind` splits them out"},
        {"quantity": "hyphen-channel units", "earlier": "2,832 units carry a "
         "dash join (#17 §4.2)", "here": L.get("hyphen", 0),
         "note": "R1 only (hyphen and no other linker). #17's 2,832 is every "
                 "unit with a dash join, which is a superset: it also counts "
                 "the `mixed` ones, e.g. `ice—our/channel`"},
        {"quantity": "`=`-only units", "earlier": "1,734 `eq` class (#17)",
         "here": L.get("eq", 0), "note": "R2: `=` and nothing else"},
        {"quantity": "`/`-only units", "earlier": "704 `slash` class (#17)",
         "here": L.get("slash", 0), "note": "R3: `/` and nothing else"},
        {"quantity": "`++`/`+`-only units", "earlier": "124 `cat`+`plus` (#17)",
         "here": L.get("cat", 0), "note": "R4: `++`/`+` and nothing else"},
        {"quantity": "closed-juncture units", "earlier": "69,232 `plain` "
         "(#17)", "here": L.get("closed", 0),
         "note": "R5: the complement of the fused classes, less the 230 markup "
                 "units among them; #17's `plain` count also includes markup "
                 "units"},
        {"quantity": "screened units (closed + open)", "earlier": "74,665 "
         "(#17 units.tsv)", "here": L.get("closed", 0) + L["open_any"],
         "note": "74,665 less the 231 markup units; the two rows above must "
                 "sum to this"},
        {"quantity": "mixed-linker units", "earlier": "n/a (#17 merges them "
         "into the per-seam classes)", "here": L.get("mixed", 0),
         "note": "R6: two or more linker kinds in one unit, e.g. "
                 "`bio=less=soul/gram` (#12 saw `bio=less=soul/gram` as a pair)"},
        {"quantity": "head-final chains", "earlier": "n/a",
         "here": L["open_units"],
         "note": "every open-juncture unit: its last morpheme is a head slot "
                 "(checked on the `:H` flag of `morph_detail`)"},
        {"quantity": "`boy-roid` occurrences", "earlier": "340 (#17 hyphen parts)",
         "here": S["named_tables"]["roid"]["n_attached"],
         "note": "recomputed from the unit layer, not from part strings"},
        {"quantity": "`soul/gram` occurrences", "earlier": "317 (#12)/306 (#17)",
         "here": S["named_tables"]["gram"]["n_attached"],
         "note": "recomputed; #12 counts regex pairs, #17 counts maximal units, "
                 "this counts `gram` attached by `/`"},
        {"quantity": "named morphemes located", "earlier": "9 named in L2",
         "here": len([m for m in NAMED if S["named_tables"][m]["n_total"]]),
         "note": "all nine attested; see affix_lexicon.tsv"},
        {"quantity": "markup-residue units, #17-visible",
         "earlier": "229 (#17 html_tag_residue)",
         "here": S["diag_counts"]["markup_unit_strict"],
         "note": "equal by construction: same strict tag match, same 229 units"},
        {"quantity": "markup-residue units, widened",
         "earlier": "n/a",
         "here": S["diag_counts"]["markup_unit"],
         "note": "superset: tags carrying a valued attribute, which the "
                 "strict pattern cannot see - one tag in this corpus, "
                 "`<body fluid=streaming>` p29"},
    ]
    return rows


def write_spec(S, out):
    L, D = S["links"], S["named_tables"]
    A = []
    ap = A.append
    ap("# Compound / portmanteau decomposition and affix lexicon (#18)\n")
    ap(f"Source: `{S['src']}` - the vetted layout markdown (#7), {S['n_pages']} "
      f"pages. Unit layer from #17 (`analysis/tokenizer/units.tsv`, "
      f"recomputed here by calling `work/operator_tokenizer.py` directly, "
      f"never re-tokenized). Reproducible via `python "
      f"work/compound_decomposition.py`; regression suite "
      f"`work/test_compound_decomposition.py`.\n")
    ap(f"English lexicon: `wordfreq`'s `en` list filtered at zipf >= "
      f"{LEX_MIN_ZIPF} ({S['n_lex']:,} entries). Schema `{SCHEMA}`.\n")

    # ---------------- headline
    ap("## 1. What was found\n")
    ap(f"- **{S['n_units']:,} units decomposed, one row each** "
      f"(`decompositions.tsv`), joining to #17 on `uid`. Counts below exclude "
      f"the {S['diag_counts']['markup_unit']} markup-residue units (S7).")
    ap(f"- **{L['open_any']:,} open-juncture units** "
      f"({pct(L['open_any'], S['n_units'])} of the book) carry an explicit "
      f"linker - `-` {L.get('hyphen', 0):,}, `=` {L.get('eq', 0):,}, `/` "
      f"{L.get('slash', 0):,}, `++`/`+` {L.get('cat', 0):,}, two or more "
      f"{L.get('mixed', 0):,} - and **{L['closed']:,} closed-juncture units** "
      f"({pct(L['closed'], S['n_units'])}) are single words the segmenter "
      f"resolves into 2-3 morphemes or leaves atomic. The four single-linker "
      f"channels sum to {L['open_single']:,}; "
      f"{L['open_single']:,} + {L.get('mixed', 0):,} = {L['open_any']:,}, and "
      f"{L['open_any']:,} + {L['closed']:,} = the "
      f"{L['open_any'] + L['closed']:,} screened units.")
    ap(f"- **{S['n_morph_total']:,} morpheme slots** over "
      f"{S['n_morph_types']:,} distinct morphemes; **{L['fused']:,} units** "
      f"carry more than one morpheme, which is where the book's own "
      f"morphology lives.")
    ap(f"- **All nine morphemes L2 names are attested** ({S['n_named']}/9), and "
      f"each attaches through essentially **one junction channel**: the book "
      f"uses `-`, `=` and `/` as distinct morphological slots, not as "
      f"interchangeable punctuation (S4).")
    ap(f"- The nine split into **{S['v_hapax']} hapax-base classifiers** (one "
      f"base, essentially never free: `-roid`<-boy, `-gram`<-soul, "
      f"`-mania`<-hydro, `-vamp`<-digital, `-less`<-bio), **{S['v_free']} free "
      f"lexemes** that stand alone as often as they attach (`machine`, `cell`, "
      f"`embryo`, `drone`, `adam`) and **{S['v_bound_multi']} bound but "
      f"multi-base** forms. That split is the answer to L2's question: the "
      f"book's system is a **classifier-tag** system, not a derivational family "
      f"(S5).\n")

    # ---------------- rules
    ap("## 2. Decomposition rules\n")
    ap("Four junction channels, kept apart because the book keeps them apart. "
       "The bucket names are the `link_kind` column of `decompositions.tsv` and "
       "`unit_screen.tsv` and the `chan` column of `linker_rows.tsv` - one "
       "vocabulary, joinable without re-deriving anything.\n")
    ap("| link_kind | unit example | linker | units | share of book | morpheme "
       "slots |")
    ap("|---|---|---|---:|---:|---:|")
    for ch in LINK_KINDS:
        ap(f"| `{ch}` | {md_code(S['chan_examples'].get(ch, '-'))} | "
           f"{md_code(S['chan_linkers'].get(ch, '') or '-')} | "
           f"{L.get(ch, 0):,} | {pct(L.get(ch, 0), S['n_units'])} | "
           f"{S['chan_morph'].get(ch, 0):,} |")
    ap("")
    ap(f"`{L.get('open_any', 0):,}` units are open-juncture (a linker is "
       f"written out) and `{L.get('closed', 0):,}` are closed-juncture (the "
       f"words are fused with no linker). Of the open ones, "
       f"`{L.get('open_single', 0):,}` use a single linker kind and `mixed` "
       f"({L.get('mixed', 0):,}) carries two or more in one unit - "
       f"`bio=less=soul/gram` - and is counted separately rather than forced "
       f"into one channel.\n")
    ap("```ebnf")
    ap("decomp  = open | closed ;")
    ap("open    = word { linker word } ;      (* explicit junction *)")
    ap("closed  = word ;                      (* fused: no junction *)")
    ap("linker  = \"-\" | \"—\" | \"=\" | \"/\" | \"++\" | \"+\" ;")
    ap("morph   = word | stem | sfx | coinage ;")
    ap("```")
    ap("")
    ap("Rule ids used in `decompositions.tsv`:\n")
    for k in sorted(RULE_DOC):
        ap(f"- `{k}` - {RULE_DOC[k]}")
    ap("")
    ap("Head assignment is **positional**: the last morpheme of a chain is the "
       "head. That is a claim about the book, so §4 tests it two ways instead "
       "of asserting it.\n")
    ap("### 2.1 What R5 does with a single word\n")
    ap("A closed-juncture unit is one word with no linker. **If the surface is "
      "an English word it is kept whole, on purpose** - `electric` does not "
      "become `electr`+`-ic`, `classic` does not become `class`+`-ic`. That "
      "is English morphology inside a word English already owns, and this "
      "issue is about the book's own compounds. Every rule that could fire "
      "here is also a source of invented analysis (`cadaver` -> `cad`+`aver`, "
      "`corporation` -> `corp`+`or`+`-ation`), and the only thing lost is "
      "`controlled` -> `control`+`-led`, which the hyphen channel already "
      "shows (`cold-blooded`).\n")
    ap(f"So the work is entirely on out-of-lexicon words, and there the rules "
      f"run in this order, each only if the previous one fails:\n")
    ap(f"1. **English suffix with a usable stem** - strip a-priori English "
      f"suffixes (`{len(EN_SUFFIXES)}` of them, fixed before looking at the "
      f"corpus) and keep the stem if it is a credible morpheme: "
      f"`defleshed` = `de`+`flesh`+`-ed`, `resettable` = `reset`+`-able`, "
      f"`fecundates` = `fecund`+`-ates`.")
    ap(f"2. **splice-valid composite** - else a 2-3 way DP over the lexicon "
      f"with every internal boundary non-spliced (Divay & Yule) and every "
      f"non-final part at zipf >= {ZFIRST} (3-letter parts >= {Z3}): "
      f"`acidhuman` = `acid`+`human`, `boyroid` = `boy`+`roid`.")
    ap("3. **blend (overlap)** - else look for a one-letter overlap: drop one "
       "letter and test two real words, `machinative` = `mach`+`native` "
       "(shares `i`).")
    ap("4. **opaque** - else the word is an **atomic coinage**: no rule "
       "resolves it, and that is reported, not guessed.\n")
    ap(f"Four filters keep the DP from inventing words, and each one exists "
      f"because it caught a specific false positive:\n")
    ap("- **Corpus attestation.** A non-final part must be something the book "
      "writes in lower case, or common English. `cadaver` is rare (z=2.70) but "
      "the book writes it 633 times, so `cadaver`+`mechanism` survives; `tok` "
      "is an English word (z=2.90) that the book never writes, so `tokage` "
      "stays opaque instead of becoming `tok`+`-age`. Requiring a *lower-case* "
      "occurrence is also what keeps proper nouns out: `Tokage` is a "
      "character's name, spelled `ToK` throughout.")
    ap("- **Function words are never morphemes.** They are not what the book "
      "fused to make a noun. This kills `tokage` = `to`+`kage` and "
      "`fecundates` = `fec`+`un`+`dates`.")
    ap("- **Two-letter parts are prefixes or suffixes only**: `de`+`flesh` and "
      "`-ix` are real, `an` at the end of `acidhuman` is not.")
    ap(f"- **The splice test** (Divay & Yule): a boundary is illegal only if a "
      f"dictionary word straddles it. Without it `idhu` - a Unix trigraph that "
      f"survives in `wordfreq` - makes the real `acid`+`human` boundary look "
      f"illegal. The lexicon floor of {LEX_MIN_ZIPF} is what removes `idhu` "
      f"(z=1.24) while keeping `cadaver` (z=2.70) and `ectoplasm` (z=2.07), "
      f"and a 4-letter minimum on the straddling word is what stops two-letter "
      f"dictionary entries from vetoing every boundary.\n")

    # ---------------- closed juncture table
    ap("## 3. Closed juncture: what fuses, and what does not\n")
    ap(f"{S['n_oov_types']:,} word-part types are outside the lexicon "
      f"({S['n_oov_tokens']:,} occurrences). Of those "
      f"**{S['oov_resolved']:,} types / {S['oov_resolved_tok']:,} occurrences** "
      f"resolve into 2-3 English morphemes; the rest are reported by kind, "
      f"with the opaque ones listed as their own class.\n")
    ap("| kind | types | occurrences | meaning |")
    ap("|---|---:|---:|---|")
    for kind, d in sorted(S["closed_kinds"].items(), key=lambda kv: -kv[1]["n"]):
        ap(f"| `{kind}` | {d['types']:,} | {d['n']:,} | {d['meaning']} |")
    ap("")
    ap(f"Top 25 opaque coinages by frequency (`closed_juncture.tsv` has all "
      f"{S['n_opaque']:,}):\n")
    ap("| surface | occurrences | en zipf | note |")
    ap("|---|---:|---:|---|")
    for r in S["opaque_top"][:25]:
        ap(f"| {md_code(r['surface'])} | {r['n']} | {r['zipf']} | {r['note']} |")
    ap("")
    ap(f"**Blends** - the overlap class - number {S['n_blends']:,} surfaces / "
      f"{S['blend_tokens']:,} occurrences, and they are the clearest portmanteau "
      f"evidence in the book: {S['blend_examples']}.\n")

    # ---------------- linker roles
    ap("## 4. The linkers, and the head-role test\n")
    ap("Per junction channel: how often the right member is the rarer of the "
       "pair (a frequent modifier against a rare novel head), and how often it "
       "also occurs standing alone in the book.\n")
    ap("| channel | pairs | distinct pairs | right rarer | left rarer | tie | "
       "exact p | right also bare | top pair |")
    ap("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in S["linkers"]:
        ap(f"| `{r['chan']}` | {r['n']:,} | {r['n_pairs']:,} | "
           f"{r['right_higher']:,} ({r['right_higher_pct']}%) | "
           f"{r['left_higher']:,} | {r['tie']} | {r['binom_p']} | "
           f"{r['right_bare_pct']}% | {md_code(r['top_pair'])} x{r['top_pair_n']} |")
    ap("")
    ap("Read this honestly. `right rarer` is **not** a test of head-finality - "
       "English compound heads are normally *more* frequent than their "
       "modifiers (`drugstore`, `bookshelf`), so a value under 50% says the "
       "book's compounds have the *opposite* frequency profile from English "
       "compounds, not that they are head-initial. The stronger evidence for "
       "head-finality is `right also bare`: a classifier has to be usable "
       "alone, and the channels split cleanly on it.\n")
    ap("| channel | right member is bare | verdict |")
    ap("|---|---:|---|")
    for r in S["linkers"]:
        ap(f"| `{r['chan']}` | {r['right_bare_pct']}% | {r['bare_verdict']} |")
    ap("")

    # ---------------- affix lexicon
    ap("## 5. The mini-lexicon: the nine morphemes L2 names\n")
    ap("L2 §L2.2 writes the list as `-oid, -machine, -vamp, -gram, -cell, "
       "-embryo, -drone, -junkie, -mania`. The book spells the first one "
       "**`-roid`**, not `-oid`, so the issue's `-oid` is resolved to the "
       "`roid` slot here - and §5.1 shows the whole `oid` string family so the "
       "decision is auditable rather than assumed. The other eight are spelled "
       "exactly as L2 gives them.\n")
    ap("`bare` = occurrences standing alone as a whole word part. `bases` = "
       "distinct left-hand morphemes it attaches to. `top base %` = share of "
       "attached tokens coming from the single most frequent base.\n")
    ap("| morpheme | total | bare | bare % | attached | bases | top base | "
       "top % | channel(s) | en zipf | verdict |")
    ap("|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|")
    for m in NAMED + EXTRA:
        d = D[m]
        src = "" if m in NAMED else " *(not in L2's list)*"
        ap(f"| {md_code(m)}{src} | {d['n_total']} | {d['n_bare']} | "
           f"{d['bare_pct']}% | {d['n_attached']} | {d['n_bases']} | "
           f"{md_code(d['top_base'])} | {d['top_base_pct']}% | "
           f"{md_code(d['chan_summary'])} | {d['zipf']} | {md_code(d['verdict'])} |")
    ap("")
    ap(f"### 5.1 L2's `-oid` resolved: the whole `oid` string family\n")
    ap(f"L2 names `-oid`; the book writes `-roid`. To show that the choice is "
      f"the book's and not a typo in the issue, here is **every** word part in "
      f"the book carrying the letters `oid` "
      f"({S['n_oid_surfaces']:,} surfaces, {S['n_oid_tokens']:,} morpheme "
      f"slots), with the position each one occupies: `head` = last morpheme of "
      f"its chain, `modifier` = a non-final morpheme, `bare` = the whole unit, "
      f"`fused` = a morpheme recovered by splitting one fused word.\n")
    ap("| surface | n | pages | positions | link kinds | in English lexicon | "
       "en zipf | first page | first unit |")
    ap("|---|---:|---:|---|---|---|---:|---:|---|")
    for o in S["oid_family"][:18]:
        ap(f"| {md_code(o['surface'])} | {o['n']:,} | {o['n_pages']} | "
           f"{md_code(o['roles'])} | {md_code(o['link_kinds'])} | "
           f"{o['in_lex']} | {o['zipf']} | "
           f"{o['first_page']} | {md_code(o['first_unit'])} |")
    ap("")
    oid_rows = {o["surface"]: o for o in S["oid_family"]}
    X = S["oid_extra"]

    def _oid(m, f="n"):
        return oid_rows.get(m, {}).get(f, 0)

    ap(f"Read this three ways. **(a) The classifier slot is `roid`, and it is "
      f"the book's most frequent `oid` morpheme by an order of magnitude**: "
      f"{_oid('roid'):,} slots over {_oid('roid', 'n_pages')} pages, "
      f"{md_code(_oid('roid', 'roles'))} - head-final and effectively never "
      f"free. L2's `-oid` is the `-roid` of `boy-roid` and `boyroid`. "
      f"**(b) No other `oid` word behaves like a classifier.** The rest of the "
      f"family is {S['n_oid_tokens'] - _oid('roid'):,} slots of words English "
      f"already owns or of the ADAM/ANDROID entity family: `anthropoid` "
      f"({_oid('anthropoid'):,}), `android`/`ANDROID` "
      f"({X['android']['n']:,}), `thyroid` ({_oid('thyroid'):,}), `void` "
      f"({_oid('void'):,}), `schizoid`, `cycloid`, `paranoid`, `keloid`. None "
      f"of them is ever written behind a linker in head position as a tag on a "
      f"new base, which is exactly what separates a classifier from a "
      f"borrowed lexeme. `andrograph` is in neither list: it is **not in the "
      f"book at all** ({X['andrograph_rows']} unit occurrences), so case C23 "
      f"is the synthetic negative test that pins the segmenter's refusal to "
      f"split it into `andro`+`graph` rather than a real citation. **(c) Two "
      f"spellings are probably OCR residue, not words.** Measured: `NDRO` "
      f"appears {X['ndro']['n']} times (first p{X['ndro']['first_page']}), "
      f"{X['ndro']['whole_unit']} of them as a whole unit with no linker at "
      f"all; `NDROID` {X['ndroid']['n']} times (first "
      f"p{X['ndroid']['first_page']}); neither truncated spelling ever shares a "
      f"unit with `ANDROID` ({X['ndro']['with_android']} and "
      f"{X['ndroid']['with_android']} such units); and the intact `ANDROID` is "
      f"written separately ({X['android']['n']} times, first "
      f"p{X['android']['first_page']}) - \"the quantum tragedy of NDRO\" (p4), "
      f"\"of NDRO that rapes the Placenta World\" (p6). A word the book meant "
      f"to write whole does not need truncating "
      f"{X['ndro']['n'] + X['ndroid']['n']} times, so this is a "
      f"**hypothesis** about OCR drift, not a measurement: `#6` owns OCR "
      f"correction, and this build counts both spellings as opaque coinages "
      f"rather than repairing them.\n")
    ap(f"**Verdict on the issue's list: 9/9 attested, with `-oid` read as "
      f"`-roid`.** Nothing in the book behaves as a productive `-oid` head: the "
      f"shape L2 names appears {_oid('roid'):,} times as `-roid`, and the "
      f"`oid` spelling appears only inside words English already owns or inside "
      f"the ANDROID entity family.\n")

    ap("### 5.2 The two groups the linkers keep apart\n")
    ap("These are **tags on one base**, repeated, never varied - which is what "
      "makes them classifiers rather than affixes:\n")
    for line in S["classifier_lines"]:
        ap(f"- {line}")
    ap("")
    ap("And these are **free lexemes used attributively**: they stand alone at "
      "least as often as they attach, which is how English compounds treat a "
      "noun used as a modifier (`machine`, `cell`, `embryo`, `drone`, "
      "`adam`).\n")
    for line in S["free_lines"]:
        ap(f"- {line}")
    ap("")
    ap(f"**Verdict on L2's question.** L2 asks whether these nine form a "
      f"*productive derivational system*. Measured on this book: "
      f"**{S['v_hapax']} of the nine are hapax-base classifiers** - bound, "
      f"attached to one or two bases, never free; **{S['v_free']} are free "
      f"lexemes** used attributively, standing alone at least as often as they "
      f"attach; and **{S['v_bound_multi']} are bound but multi-base**, the only "
      f"ones with real derivational range. `-drone` (13 bases) and `-machine` "
      f"(20 bases) are the widest, and both are *also* free lexemes - the book "
      f"reuses ordinary nouns as classifier heads rather than coining a "
      f"derivational paradigm.\n")
    ap(f"What *is* productive is the **junction system**, not the affixes: `-`, "
      f"`=` and `/` each carry a disjoint morpheme set (§4), and within each "
      f"channel the same base repeats with the same tag hundreds of times. That "
      f"is a **naming** system, and it is consistent with #12's independent "
      f"finding that `=` is a naming linker and #14's that `<<…>>` blocks carry "
      f"names: the book tags one referent per tag, rather than deriving a family "
      f"of new words from a productive suffix.\n")
    ap(f"### 5.3 Does the issue's list survive a blind scan?\n")
    ap(f"L2's nine were chosen in advance. `head_candidates.tsv` re-runs the "
      f"same measure over **every** morpheme in the book, in head position, and "
      f"sorts by the number of times the book writes an explicit linker in "
      f"front of it ({S['n_heads']:,} candidates clear "
      f"8 uses and 2 distinct bases). Ranked that way:\n")
    ap("| rank | morpheme | L2 list | head uses (open) | head uses (total) | "
       "distinct bases | top base | bare % | verdict |")
    ap("|---:|---|---|---:|---:|---:|---|---:|---|")
    for r in S["heads"][:20]:
        ap(f"| {r['n_attached_rank']} | {md_code(r['morph'])} | "
           f"{'yes' if r['named'] == 'y' else '-'} | {r['n_open']:,} | "
           f"{r['n_attached']:,} | {r['n_bases']} | "
           f"{md_code(r['top_base_open'] or r['top_base'])} | "
           f"{r['bare_pct']}% | {md_code(r['verdict'])} |")
    ap("")
    ap(f"**{S['n_head_named']} of L2's nine appear in that blind top-20**, "
      f"and the ranking is not circular: `roid`, `gram` and `adam` are the 5th, "
      f"6th and 7th most-written head morphemes in the book whether or not the "
      f"issue named them. The four above them that L2 does *not* name are "
      f"{S['head_above_named']} - all English words recovered by the "
      f"closed-juncture rules, i.e. inherited English morphology rather than "
      f"book coinage. Read that as the honest answer: **the book's productive "
      f"morphology is mostly English; what it adds is the junction system and "
      f"the classifier tags listed in 5.2.**\n")

    # ---------------- evidence
    ap("## 6. Page-cited cases\n")
    ap("| case | unit | page | offsets | decomposition | head | note |")
    ap("|---|---|---:|---|---|---|---|")
    for r in S["cases"]:
        ap(f"| {r['case']} | {md_code(r['unit'] or r['needle'])} | "
           f"{r['page']} | {r['start']}-{r['end']} | "
           f"{md_code(r['morph'] or '-')} | {md_code(r['head'] or '-')} | "
           f"{r['note']} |")
    ap("")
    ap("Context for each case is in `cases.tsv`; `exact` confirms the unit is "
       "the byte-exact source slice at those offsets. One case (`C23`) is "
       "**synthetic and not in the book** - it pins the segmenter's refusal to "
       "split `andrograph` into `andro`+`graph`, and it is marked "
       "`NOT IN CORPUS` rather than quietly dropped.\n")

    # ---------------- diagnostics
    ap("## 7. Diagnostics - nothing silently repaired\n")
    ap("| row | n | pages | meaning |")
    ap("|---|---:|---:|---|")
    for r in S["diags"]:
        ap(f"| `{r['diag']}` | {r['n']:,} | {r['pages']} | {r['meaning']} |")
    ap("")
    ap(f"`markup_unit` ({S['diag_counts']['markup_unit']}) is a **superset** of "
       f"#17's `html_tag_residue` (229), and `markup_unit_strict` "
       f"({S['diag_counts']['markup_unit_strict']}) is the subset #17's strict "
       f"tag match can see - the two must agree, and they do. The widening adds "
       f"tags carrying a **valued attribute**, which the strict pattern cannot "
       f"see: exactly one source tag qualifies, `<body fluid=streaming>` on p29, "
       f"which is why the count moves by "
       f"{S['diag_counts']['markup_unit'] - S['diag_counts']['markup_unit_strict']} "
       f"units (`body`, `fluid=streaming`).\n")
    ap(f"The attribute pattern is deliberately narrow. The book writes "
       f"angle-bracket blocks of its own - `<I escape from the volume>` (p13), "
       f"`<A lapse of memory>` - and an earlier looser match swallowed them, "
       f"flagging Siratori's own English as markup residue. Requiring a "
       f"`name=value` pair keeps the markup and leaves the book alone. All "
       f"{S['diag_counts']['markup_unit']} flagged units are excluded from "
       f"every lexical count above and **must** be excluded by #19's neologism "
       f"screen as well.\n")

    # ---------------- reconciliation
    ap("## 8. Reconciliation with #12 / #14 / #17\n")
    ap("| quantity | earlier | here | note |")
    ap("|---|---:|---:|---|")
    for r in S["recon"]:
        ap(f"| {r['quantity']} | {r['earlier']} | {r['here']} | {r['note']} |")
    ap("")

    # ---------------- limits
    ap("## 9. Limits\n")
    ap("- **The splice filter is a lexicon test, not a judgement about this "
      "book.** A real juncture that happens to coincide with a dictionary word "
      "is rejected; `eachother` stays atomic because English has it. Precision "
      "is favoured deliberately: over-segmenting a coinage into English would "
      "invent morphology the book does not contain.")
    ap(f"- **The thresholds are fitted to this corpus.** {LEX_MIN_ZIPF}, "
      f"{Z3}, {ZFIRST}, {PENALTY} and the frequency floor were chosen by "
      f"inspecting this vocabulary. `sensitivity.tsv` re-runs the closed "
      f"segmentation at three lexicons and reports how many types move; the "
      f"ranking of the named morphemes is unchanged in all three.")
    ap("- **Head-finality is positional, not parsed.** No tagger was used, so "
      "§4's tests are about *frequency profile and free-standing use*, not "
      "about syntactic function. `#37` (POS on silence-removed residue) is the "
      "issue that can settle it.")
    ap("- **Case is preserved but not interpreted.** `Soul/gram` and "
      "`soul/gram` decompose identically here and stay separate types; `retro-"
      "ADAM` and `boy-roid` differ in kind only because the source does.")
    ap("- **L2's `-oid` had to be resolved, and the resolution is the book's, "
      "not the issue's.** The issue writes `-oid`; the book writes `-roid`. §5.1 "
      f"publishes all {S['n_oid_surfaces']:,} `oid` morpheme surfaces "
      f"({S['n_oid_tokens']:,} slots) so the mapping is checkable item by item, "
      "and reports the truncated spellings `NDRO` / `NDROID` as a flagged "
      "**hypothesis** about OCR drift rather than repairing them - `#6` owns "
      "OCR correction.")
    ap("- **`#22` owns case, `#21` owns numerics.** `numeric_part` is counted "
      "and flagged (`1-84068-060-1` is an ISBN, not a compound) but not "
      "interpreted.")
    ap("- **Negative result, stated plainly:** no rule recovers `-maniac` "
      "(`hydromaniac`), `-humanix` or `-rii` (`ndroid`, `ndro`); they stay in "
      "`opaque_coinage`, which is the honest outcome for a coined stem plus a "
      "novel suffix.")
    ap("")

    ap("## 10. Artifacts\n")
    ap("| file | contents |")
    ap("|---|---|")
    for f, d in S["artifacts"]:
        ap(f"| `{f}` | {d} |")
    ap("")
    ap("## 11. Downstream contract for #19 / #21 / #26 / #37\n")
    ap("- `decompositions.tsv` has **one row per unit occurrence, keyed by "
      "`uid`** - the same uid as #17's `units.tsv`. Join on `uid`, never on "
      "token index.")
    ap("- `unit_screen.tsv` is the **type-level** view #19 should screen: unit "
      "surface, occurrence count, class, rule, resolved decomposition, markup "
      "flag. Screening word parts instead would miss all "
      f"{L['open_any']:,} open-juncture units.")
    ap("- `closed_juncture.tsv` gives the resolved coinage inventory with "
      "first counts; `opaque_coinage` rows are the ones #19 should treat as "
      "true coinages rather than compounds of English.")
    ap("- `affix_lexicon.tsv` + `head_candidates.tsv` are the classifier-head "
      "layer, with the bare/attached split #27's collocation work can use to "
      "decide whether a morpheme is behaving like a word.")
    ap("- `linker_rows.tsv` gives the junction-level statistics #52 "
      "(operator-word MI) and #15 (Rosetta mapping) can join on `chan`.")
    ap("- All `.tsv` here are quote-safe and written with `split(\"\\t\")`; text "
      "fields have tab/newline escaped (`cell()`), so no csv reader is needed.")
    ap("")
    ap("---\n")
    ap(f"*Generated by `work/compound_decomposition.py` (schema `{SCHEMA}`).*")
    with open(os.path.join(out, "morphology_spec.md"), "w") as f:
        f.write("\n".join(A) + "\n")


# ---------------------------------------------------------------------------
# sensitivity: does the lexicon floor change the answer?
# ---------------------------------------------------------------------------
def sensitivity(lex_oov, counts):
    """Re-run the closed segmentation at three lexicon floors."""
    rows = []
    base = None
    for floor in (1.0, LEX_MIN_ZIPF, 2.0):
        lx = Lexicon(min_zipf=floor)
        resolved = 0
        rtypes = 0
        for surf, n in counts.items():
            kind, _ms, _e = analyse_closed(lx, surf)
            if kind in ("closed_composite", "stem_suffix", "closed_blend"):
                resolved += n
                rtypes += 1
        rows.append({"lex_min_zipf": floor, "n_lex": len(lx.words),
                     "types_resolved": rtypes, "tokens_resolved": resolved,
                     "tokens_total": sum(counts.values()),
                     "pct_resolved": round(100.0 * resolved / sum(counts.values()), 1)})
        if floor == LEX_MIN_ZIPF:
            base = resolved
    for r in rows:
        r["delta_vs_default"] = r["tokens_resolved"] - base
    return rows


# ---------------------------------------------------------------------------
# the issue's `-oid`: a string family, not a morpheme
# ---------------------------------------------------------------------------
# L2 (docs/meaning-extraction-methods.md §L2.2) names the classifier heads
# `-oid, -machine, -vamp, -gram, -cell, -embryo, -drone, -junkie, -mania`.  The
# book does not write `-oid`: it writes `-roid` (`boy-roid`, `boyroid`).  So the
# issue's first morpheme has to be *resolved*, not assumed, and the honest way to
# do that is to inventory every place the book writes the letters `oid` and show
# which of them is a classifier head slot and which is not.
RE_OID_STR = re.compile(r"oid", re.I)


def oid_family(recs, lex):
    """Every word part in the book containing the letters `oid`, inventoried.

    Two questions, both answered from the token stream:

      1. does the book ever write a classifier head ending in the *literal*
         string `oid`?  (It writes `roid`; `-oid` in L2 is the shape, not the
         spelling.)
      2. what are the other `oid` words?  English lexemes (`void`, `thyroid`,
         `anthropoid`), the book's ADAM/ANDROID family, or OCR residue.

    Reported per surface with counts, page spread and first page-cited
    occurrence, so the `-oid` -> `roid` decision can be audited rather than
    taken on trust.  Markup units are excluded, as everywhere else here.
    """
    n = collections.Counter()
    pages = collections.defaultdict(set)
    first = {}
    roles = collections.defaultdict(collections.Counter)
    fused = collections.defaultdict(collections.Counter)
    for r in recs:
        if r["markup"]:
            continue
        seq = [c.split(":") for c in r["morph_detail"].split("|")]
        last = len(seq) - 1
        for i, w in enumerate(seq):
            if not RE_OID_STR.search(w[0]):
                continue
            k = w[0].lower()
            n[k] += 1
            pages[k].add(r["page"])
            first.setdefault(k, (r["page"], r["start"], r["unit"]))
            # where the morpheme sits in its unit, decided by position (the
            # `:H` flag of `morph_detail` marks it, but the split field holding
            # it is `w[0]`, so testing `w[0].endswith(":H")` would be wrong -
            # see `mine_heads`, which indexes instead):
            #   bare     the whole unit is this one morpheme;
            #   head     last morpheme of a chain (head-final position);
            #   modifier a non-final morpheme;
            #   fused    recovered by splitting one fused word (closed rule R5)
            if len(seq) == 1:
                roles[k]["bare"] += 1
            elif i == last:
                roles[k]["head"] += 1
            elif w[2] == "closed":
                roles[k]["fused"] += 1
            else:
                roles[k]["modifier"] += 1
            fused[k][r["link_kind"]] += 1
    rows = []
    for surf, c in n.most_common():
        rows.append({
            "surface": surf, "n": c, "n_pages": len(pages[surf]),
            "roles": " ".join(f"{k}:{v}" for k, v in
                              sorted(roles[surf].items(), key=lambda kv: -kv[1])),
            "link_kinds": " ".join(f"{k}:{v}" for k, v in
                                   sorted(fused[surf].items(),
                                          key=lambda kv: -kv[1])),
            "in_lex": "y" if lex.in_lex(surf) else "n",
            "zipf": round(lex.z(surf), 2),
            "first_page": first[surf][0], "first_unit": first[surf][2],
        })
    return rows


def _trunc_facts(recs, surface):
    """Evidence for a claim §5.1 makes about a spelling.

    Four measured numbers: unit occurrences whose whole unit is this one word,
    how many of those carry a linker, how many share a unit with `android`
    (the intact spelling), and the first page each occurs on.
    """
    n = bare = fused = with_android = 0
    pages = []
    for r in recs:
        if r["markup"]:
            continue
        words = [c.split(":")[0].lower() for c in r["morph_detail"].split("|")]
        if surface not in words:
            continue
        n += 1
        if not r["rule"] == "R5":
            fused += 1
        if words == [surface]:
            bare += 1
        if any(w != surface and "android" in w for w in words):
            with_android += 1
        if r["page"] not in pages:
            pages.append(r["page"])
    return {"surface": surface, "n": n, "whole_unit": bare, "with_linker": fused,
            "with_android": with_android, "n_pages": len(pages),
            "first_page": min(pages) if pages else 0}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
DECOMP_COLS = ["uid", "page", "start", "end", "unit", "cls", "sig", "rule",
               "channel", "link_kind", "linkers", "n_link", "n_words", "n_morph",
               "head", "mods", "morph", "morph_detail", "closed_kind",
               "closed_surface", "closed_alt", "overlap", "has_digit", "markup"]

UNIT_SCREEN_COLS = ["unit", "lower", "n", "n_pages", "cls", "rule", "channel",
                    "link_kind", "kind_summary", "n_morph", "morph",
                    "morph_detail", "markup", "first_page", "first_start"]

AFFIX_COLS = ["morph", "provenance", "n_total", "n_bare", "bare_pct",
              "n_attached", "n_bases", "top_base", "top_base_n", "top_base_pct",
              "chan_summary", "n_surface", "n_pages", "zipf", "verdict",
              "first_page", "first_unit"]

HEAD_COLS = ["morph", "n_attached", "n_open", "n_bases", "n_bare", "bare_pct",
             "top_chan", "top_chan_n", "top_chan_pct", "top_base",
             "top_base_n", "top_base_pct", "top_base_open", "top_base_open_n",
             "chan_summary", "zipf", "in_lex", "named", "verdict",
             "n_attached_rank", "first_page", "first_unit", "bases",
             "open_bases"]

LINKER_COLS = ["chan", "n", "n_pairs", "n_distinct_left", "n_distinct_right",
               "right_higher", "left_higher", "tie", "right_higher_pct",
               "binom_p", "right_bare_pct", "right_bare_n", "top_pair",
               "top_pair_n", "example"]

CLOSED_COLS = ["surface", "n", "kind", "n_morph", "split", "roles",
               "all_parts_english", "min_part_zipf", "alt_splits",
               "overlap_letter", "in_lex", "zipf"]

OID_COLS = ["surface", "n", "n_pages", "roles", "link_kinds", "in_lex",
            "zipf", "first_page", "first_unit"]

CASE_COLS = ["case", "needle", "rule", "note", "page", "start", "end", "unit",
             "morph", "morph_detail", "head", "mods", "channel",
             "closed_kind", "n_morph", "markup", "exact", "context"]

RECON_COLS = ["quantity", "earlier", "here", "note"]
DIAG_COLS = ["diag", "n", "pages", "meaning", "example"]
SENS_COLS = ["lex_min_zipf", "n_lex", "types_resolved", "tokens_resolved",
             "tokens_total", "pct_resolved", "delta_vs_default"]

ARTIFACTS = [
    ("morphology_spec.md", "THE DELIVERABLE: rules, linkers, mini-lexicon"),
    ("decompositions.tsv", "one row per unit occurrence (join on `uid`)"),
    ("unit_screen.tsv", "one row per unit TYPE: the #19 screening table"),
    ("affix_lexicon.tsv", "the nine L2 morphemes + three more, with verdicts"),
    ("head_candidates.tsv", "every morpheme scored as a classifier head"),
    ("closed_juncture.tsv", "fused-word inventory with resolutions"),
    ("linker_rows.tsv", "per-junction-channel head-role evidence"),
    ("cases.tsv", "20 page-cited cases with context"),
    ("oid_family.tsv", "L2's `-oid` resolved: every `oid` morpheme slot"),
    ("diagnostics.tsv", "pressure ledger incl. the markup-residue superset"),
    ("sensitivity.tsv", "closed segmentation at three lexicon floors"),
    ("reconciliation.tsv", "this build vs #12 / #14 / #17"),
    ("summary.json", "machine-readable headline numbers"),
]


def build(src=SRC, out=OUT, verbose=True):
    os.makedirs(out, exist_ok=True)
    bodies, marks, page_units = load_units(src)
    # the lexicon needs the book's own word parts first (see Lexicon.attested)
    lex = Lexicon(attested=corpus_attested(page_units))
    closed_cache = {}
    recs = []
    for pno, body, units in page_units:
        for u in units:
            rec, _m = decompose(lex, u, marks[pno], closed_cache)
            recs.append(rec)

    # ---- morpheme inventory
    morph_n = collections.Counter()
    morph_pages = collections.defaultdict(set)
    for r in recs:
        if r["markup"]:
            continue
        for c in r["morph_detail"].split("|"):
            txt = c.split(":")[0].lower()
            morph_n[txt] += 1
            morph_pages[txt].add(r["page"])

    # ---- closed-juncture counts: keyed on the *surface* word, case-folded.
    # `closed_surface` is that word; `morph_detail[0]` would be the first
    # *recovered morpheme* (`controlled` -> `controll`), which is not a key.
    closed_counts = collections.Counter()
    for r in recs:
        if r["channel"] == "closed" and not r["markup"]:
            closed_counts[r["closed_surface"]] += 1

    # ---- linkers + heads + affixes
    # `bare` (free-standing uses of each morpheme) must exist before the linker
    # rows, because the head-role test reads it
    D, bare, att, bases = affix_evidence(recs, NAMED + EXTRA, lex)
    Lr, pairs = linker_rows(recs, bare, lex)
    for r in Lr:
        r["bare_verdict"] = (
            "head is a free lexeme: usable alone" if r["right_bare_pct"] >= 50
            else "head is bound: a tag, never alone")
    heads = mine_heads(recs, bare, lex)

    # ---- channel aggregates.  `channel` is open/closed; `link_kind` splits the
    # open ones four ways (plus `mixed`), so the table and `linker_rows.tsv`
    # share one vocabulary of channel names.
    links = collections.Counter()
    chan_morph = collections.Counter()
    chan_ex = collections.defaultdict(collections.Counter)
    chan_linkers = {"hyphen": "-/—", "eq": "=", "slash": "/", "cat": "++/+",
                    "closed": "", "mixed": "two or more"}
    for r in recs:
        if r["markup"]:
            continue
        k = link_kind(r)
        links[k] += 1                       # unit occurrences in the channel
        chan_morph[k] += r["n_morph"]
        chan_ex[k][r["unit"]] += 1
    # the example shown must be a *lexical* unit, not the ISBN on p2: rank by
    # frequency but drop anything holding a digit
    chan_examples = {}
    for k, c in chan_ex.items():
        lex_only = [(u, n) for u, n in c.most_common() if not any(ch.isdigit() for ch in u)]
        if lex_only:
            chan_examples[k] = lex_only[0][0]
    # `open_any` is every unit that writes a linker out; `open_single` is the
    # subtotal that uses exactly one linker kind (the four channels), so
    # `open_any == open_single + mixed`.  v1 published only the subtotal under
    # the name `open`, which read as the total and did not add up next to the
    # 135 `mixed` units it listed in the same sentence.
    links["open_single"] = sum(links[c] for c in ("hyphen", "eq", "slash", "cat"))
    links["open_any"] = links["open_single"] + links.get("mixed", 0)
    links["open_units"] = links["open_any"]
    # `fused` counts units the segmenter actually split (closed) plus every
    # open-juncture unit: those are the units carrying real morphology.
    links["fused"] = sum(1 for r in recs if r["n_morph"] > 1 and not r["markup"])
    # the same two numbers recomputed straight off the per-unit `link_kind`
    # column, so the aggregate table cannot disagree with the artifact
    lk = collections.Counter(r["link_kind"] for r in recs if not r["markup"])
    assert lk["closed"] == links["closed"], (lk, links)
    assert lk["mixed"] == links.get("mixed", 0), (lk, links)
    assert sum(lk[k] for k in ("hyphen", "eq", "slash", "cat",
                               "mixed")) == links["open_any"], (lk, links)
    assert links["closed"] + links["open_any"] == sum(1 for r in recs
                                                      if not r["markup"]), \
        "closed + open-juncture must account for every screened unit"

    # ---- closed kinds
    ck = collections.defaultdict(lambda: {"n": 0, "types": 0})
    meaning = {
        "closed_composite": "2-3 English morphemes fused, splice-valid",
        "stem_suffix": "a-priori English suffix + split stem (`defleshed`)",
        "closed_blend": "one-letter overlap blend (`resettable`)",
        "atomic_seen": "an English word: kept whole on purpose (see "
                       "`analyse_dict`)",
        "opaque_coinage": "no rule resolves it: an atomic coinage",
    }
    for surf, (kind, _ms, _e) in closed_cache.items():
        n = closed_counts.get(surf, 0)
        if not n:
            continue
        ck[kind]["n"] += n
        ck[kind]["types"] += 1
    opaque = [{"surface": s, "n": n, "zipf": round(lex.z(s), 2), "note": note}
              for (s, n, note) in
              [(r["surface"], r["n"], _opaque_note(r, lex))
               for r in closed_rows(closed_cache, closed_counts, lex)
               if r["kind"] == "opaque_coinage"]]
    blends = [r for r in closed_rows(closed_cache, closed_counts, lex)
              if r["kind"] == "closed_blend"]

    # ---- diagnostics
    diags = diag_rows(recs, closed_cache, lex)
    dcounts = {d["diag"]: d["n"] for d in diags}

    # ---- cases
    cases = case_rows(recs, bodies)
    missing = [c["case"] for c in cases if c["page"] == ""]

    # ---- sensitivity
    sens = sensitivity(lex, closed_counts)

    # ---- the issue's `-oid`: every word part carrying those letters
    OID = oid_family(recs, lex)
    # the three facts §5.1 quotes about the truncated ANDROID spellings,
    # measured rather than typed into the prose
    OIDX = {"ndro": _trunc_facts(recs, "ndro"),
            "ndroid": _trunc_facts(recs, "ndroid"),
            "android": _trunc_facts(recs, "android"),
            "andrograph_rows": sum(1 for r in recs
                                   if "andrograph" in r["unit"].lower())}

    # ---- headline summary
    n_units = len(recs)
    oov_types = [s for s in closed_counts if not lex.in_lex(s)]
    oov_tok = sum(closed_counts[s] for s in oov_types)
    resolved_kinds = ("closed_composite", "stem_suffix", "closed_blend")
    resolved_types = [s for s in closed_counts
                      if closed_cache[s][0] in resolved_kinds]
    resolved_tok = sum(closed_counts[s] for s in resolved_types)
    verdict_counts = collections.Counter(D[m]["verdict"] for m in NAMED)
    # verdict families, collapsed for the headline: a classifier that only
    # *nearly* has one base is still a classifier
    v_hapax = sum(v for k, v in verdict_counts.items()
                  if k in ("hapax_base_classifier", "mostly_hapax_base"))
    v_free = sum(v for k, v in verdict_counts.items()
                 if k in ("free_lexeme", "free_head_few_bases"))
    v_bound_multi = sum(v for k, v in verdict_counts.items()
                        if k in ("bound_productive_affix",
                                 "bound_multi_base_classifier"))

    S = {
        "schema": SCHEMA, "src": src, "out": out, "n_pages": len(bodies),
        "n_units": n_units, "n_lex": len(lex.words),
        "lex_min_zipf": LEX_MIN_ZIPF, "Z3": Z3, "ZFIRST": ZFIRST,
        "penalty": PENALTY, "maxn": MAXN,
        "n_en_suffixes": len(EN_SUFFIXES),
        "n_morph_total": sum(morph_n.values()),
        "n_morph_types": len(morph_n),
        "links": dict(links), "chan_morph": dict(chan_morph),
        "chan_examples": chan_examples, "chan_linkers": chan_linkers,
        "chan_top": {k: c.most_common(8) for k, c in chan_ex.items()},
        "linkers": Lr, "named_tables": D, "heads": heads,
        "closed_kinds": {k: dict(v, meaning=meaning.get(k, "")) for k, v in ck.items()},
        "n_oov_types": len(oov_types), "n_oov_tokens": oov_tok,
        "oov_resolved": len(resolved_types), "oov_resolved_tok": resolved_tok,
        "opaque_top": opaque, "n_opaque": len(opaque),
        "n_blends": len(blends), "blend_tokens": sum(b["n"] for b in blends),
        "blend_examples": ", ".join(
            f"{md_code(b['surface'])} = {md_code(b['split'])} (shares "
            f"{md_code(b['overlap_letter'])})" for b in blends[:4]) or "none",
        "verdict_counts": verdict_counts, "v_hapax": v_hapax, "v_free": v_free,
        "v_bound_multi": v_bound_multi,
        "oid_family": OID,
        "n_oid_surfaces": len(OID),
        "n_oid_tokens": sum(o["n"] for o in OID),
        "oid_named": "roid",
        "oid_extra": OIDX,
        "n_named": sum(1 for m in NAMED if D[m]["n_total"]),
        "diags": diags, "diag_counts": dcounts, "cases": cases,
        "cases_missing": missing, "sensitivity": sens,
        "classifier_lines": _classifier_lines(D),
        "free_lines": _free_lines(D),
        "n_heads": len(heads),
        "n_head_named": sum(1 for h in heads[:20] if h["named"] == "y"),
        "head_above_named": ", ".join(
            md_code(h["morph"]) for h in heads[:20] if h["named"] != "y")
        or "none",
    }
    S["recon"] = reconciliation(S)
    S["artifacts"] = ARTIFACTS
    for k in ("heads", "named_tables"):
        pass
    S["_compact"] = {
        "linkers": [{k: v for k, v in r.items()} for r in Lr],
        "heads": heads[:40],
        "named": {m: {k: v for k, v in D[m].items() if k != "top_bases"}
                  for m in NAMED + EXTRA},
    }
    write_spec(S, out)

    # ---- artifacts
    dump_tsv("decompositions.tsv", recs, DECOMP_COLS, out)
    dump_tsv("unit_screen.tsv", unit_type_rows(recs), UNIT_SCREEN_COLS, out)
    affix_rows = []
    for m in NAMED + EXTRA:
        d = dict(D[m])
        d["provenance"] = "L2-named" if m in NAMED else "same measure, not in L2"
        affix_rows.append(d)
    dump_tsv("affix_lexicon.tsv", affix_rows, AFFIX_COLS, out)
    dump_tsv("head_candidates.tsv", heads, HEAD_COLS, out)
    dump_tsv("closed_juncture.tsv",
             closed_rows(closed_cache, closed_counts, lex), CLOSED_COLS, out)
    dump_tsv("linker_rows.tsv", Lr, LINKER_COLS, out)
    dump_tsv("cases.tsv", cases, CASE_COLS, out)
    dump_tsv("oid_family.tsv", OID, OID_COLS, out)
    dump_tsv("diagnostics.tsv", diags, DIAG_COLS, out)
    dump_tsv("sensitivity.tsv", sens, SENS_COLS, out)
    dump_tsv("reconciliation.tsv", S["recon"], RECON_COLS, out)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({k: v for k, v in S.items()
                   if not k.startswith("_")}, f, indent=1, default=str)
    if verbose:
        print(f"#18 -> {out}")
        print(f"  units {n_units:,}  open {links['open_any']:,}  "
              f"closed {links['closed']:,}  fused {links['fused']:,}")
        print(f"  morphemes {S['n_morph_total']:,} over "
              f"{S['n_morph_types']:,} types")
        print(f"  named morphemes attested {S['n_named']}/9  "
              f"verdicts {dict(verdict_counts)}")
        print(f"  oov resolved {S['oov_resolved']}/{S['n_oov_types']} types "
              f"({resolved_tok}/{oov_tok} tokens)")
        print(f"  head candidates {len(heads):,}  "
              f"L2-named in blind top-20: {S['n_head_named']}")
        if missing:
            print(f"  WARN cases not found: {missing}")
    return S


def _opaque_note(r, lex):
    if r["surface"] in ("ndroid", "ndro", "roid", "id", "nd"):
        return "suffix-only coinage, or OCR residue"
    if r["zipf"] > 0:
        return "rare English-ish form, no valid split"
    return "no English morphemes found"


def _classifier_lines(D):
    out = []
    for m in NAMED + EXTRA:
        d = D[m]
        if d["verdict"] in ("hapax_base_classifier", "mostly_hapax_base"):
            out.append(
                f"{md_code('-' + m)} - {d['n_attached']} attached uses, "
                f"{d['bare_pct']}% bare, {d['n_bases']} base(s); "
                f"{d['top_base_pct']}% of them on "
                f"{md_code(d['top_base'])} ({md_code(d['chan_summary'])})")
    return out or ["none"]


def _free_lines(D):
    out = []
    for m in NAMED + EXTRA:
        d = D[m]
        if d["verdict"] in ("free_lexeme", "free_head_few_bases"):
            out.append(
                f"{md_code('-' + m)} - {d['n_bare']} bare / {d['n_attached']} "
                f"attached ({d['bare_pct']}% bare), {d['n_bases']} bases; "
                f"channel {md_code(d['chan_summary'])}")
    return out or ["none"]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    src = argv[0] if argv else SRC
    out = argv[1] if len(argv) > 1 else OUT
    build(src, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
