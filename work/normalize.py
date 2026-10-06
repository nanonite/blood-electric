#!/usr/bin/env python3
"""
L0 / Chainlink #5 - glyph inventory and frozen unicode normalization
for the "Blood Electric" corpus.

Why this exists
---------------
The corpus (``docs/markdown/blood-electric/blood-electric.md``) is marker
OCR output that has already been through marker's markdown renderer and
ftfy.  Before any analysis runs on it, the L0 layer of
``docs/meaning-extraction-methods.md`` (§L0.3) requires:

  1. an enumeration of *every* codepoint in the corpus,
  2. a decision about the canonical form of each codepoint,
  3. normalization of ligatures and quote variants,
  4. a *frozen* ``charset.tsv`` plus a reusable normalizer with a test.

This module is the normalizer.  Its rules are deliberately conservative:

  * Only *typographic presentation* variants are canonicalized
    (ligatures, curly quotes, compatibility forms, stray whitespace
    controls).  None of these occur in this corpus - measured - so the
    frozen normalization is a no-op on the frozen corpus; that no-op is
    itself a recorded, checked result (see ``freeze``).
  * Case, diacritics, dashes and the book's own codework (e.g. literal
    ``\\n`` in the scan text) are NOT touched: they are evidence, not
    noise.  An optional ``flatten_dashes=True`` switch exists for
    downstream tasks that want a single dash class; it is off by
    default.

The rules live in exactly one place (the mapping tables and RULES
pipeline below); ``charset.tsv`` and the tests are generated/checked
against them, so the code, the frozen table and the test cannot
silently drift apart.  All mappings are written with ``\\u`` escapes so
the codepoints are unambiguous in source.

Subcommands
-----------
  normalize  normalize a text file and print a per-rule report
             (default when no subcommand is given)
  verify     check a text file against the frozen charset.tsv
  freeze     rebuild analysis/charset/{charset.tsv,freeze.json,
             charset_report.md} (delegates to work/build_charset.py)

Usage (from the workspace root, inside the `marker` conda env):

  python work/normalize.py docs/markdown/blood-electric/blood-electric.md
  python work/normalize.py verify docs/markdown/blood-electric/blood-electric.md
  python work/normalize.py freeze
  python work/test_normalize.py            # the test that must pass
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
import unicodedata

# --------------------------------------------------------------------------
# Locations (relative to the workspace root, which is the parent of work/)
# --------------------------------------------------------------------------
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(WORK_DIR)

DEFAULT_CORPUS = os.path.join(
    ROOT_DIR, "docs", "markdown", "blood-electric", "blood-electric.md"
)
DEFAULT_RAW_DUMP = os.path.join(
    ROOT_DIR, "analysis", "ocr_confidence", "raw_ocr_confidence.jsonl.gz"
)
DEFAULT_OUTDIR = os.path.join(ROOT_DIR, "analysis", "charset")
CHARSET_TSV = os.path.join(DEFAULT_OUTDIR, "charset.tsv")
FREEZE_JSON = os.path.join(DEFAULT_OUTDIR, "freeze.json")
REPORT_MD = os.path.join(DEFAULT_OUTDIR, "charset_report.md")
TEST_FILE = os.path.join(WORK_DIR, "test_normalize.py")

NORMALIZER_VERSION = "1.0.0"

# ==========================================================================
# The frozen rules.  Single source of truth for charset.tsv and the tests.
# ==========================================================================
# All mappings are written with \u escapes so that the exact codepoints
# are unambiguous in source.  Each mapping rule maps source codepoint ->
# replacement string ("" deletes).  Rules are applied in the order of
# the RULES pipeline below.

# Explicit Latin ligatures.  Every entry maps to EXACTLY what
# NFKC maps it to (asserted by the tests), so the table is
# pure defence in depth: the common ligatures stay caught even
# if a future Unicode version changes NFKC behaviour.  Note
# that ligature LETTERS Unicode does not decompose (OE oe IJ
# ij) are deliberately absent: they are legitimate letters in
# their own right, not typographic presentation forms, and
# decomposing them would be an interpretive choice the L0
# layer must not make.
LIGATURES = {
    "ﬀ": "ff",  # LATIN SMALL LIGATURE FF
    "ﬁ": "fi",  # LATIN SMALL LIGATURE FI
    "ﬂ": "fl",  # LATIN SMALL LIGATURE FL
    "ﬃ": "ffi",  # LATIN SMALL LIGATURE FFI
    "ﬄ": "ffl",  # LATIN SMALL LIGATURE FFL
    "ﬅ": "st",  # LATIN SMALL LIGATURE LONG S T (modern form: st)
    "ﬆ": "st",  # LATIN SMALL LIGATURE ST
    "Ĳ": "IJ",  # LATIN CAPITAL LIGATURE IJ
    "ĳ": "ij",  # LATIN SMALL LIGATURE IJ
    "Ǆ": "DŽ",  # LATIN CAPITAL LETTER DZ WITH CARON
    "ǅ": "Dž",  # LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON
    "ǆ": "dž",  # LATIN SMALL LETTER DZ WITH CARON
    "Ǉ": "LJ",  # LATIN CAPITAL LETTER LJ
    "ǈ": "Lj",  # LATIN CAPITAL LETTER L WITH SMALL LETTER J
    "ǉ": "lj",  # LATIN SMALL LETTER LJ
    "Ǌ": "NJ",  # LATIN CAPITAL LETTER NJ
    "ǋ": "Nj",  # LATIN CAPITAL LETTER N WITH SMALL LETTER J
    "ǌ": "nj",  # LATIN SMALL LETTER NJ
    "Ǳ": "DZ",  # LATIN CAPITAL LETTER DZ
    "ǲ": "Dz",  # LATIN CAPITAL LETTER D WITH SMALL LETTER Z
    "ǳ": "dz",  # LATIN SMALL LETTER DZ
}

# Typographic ("smart") quotes and prime marks -> ASCII straight quotes.
# NFKC does NOT map these, so this table is the actual workhorse of
# "normalize quote variants".  Prime marks are included because OCR of
# a scanned book frequently emits them as quotes.
QUOTES = {
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "‚": "'",  # SINGLE LOW-9 QUOTATION MARK
    "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "′": "'",  # PRIME
    "‵": "'",  # REVERSED PRIME
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
    "„": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "‟": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "″": '"',  # DOUBLE PRIME
    "‶": '"',  # REVERSED DOUBLE PRIME
}

# Unicode space characters -> plain space (U+0020).  The full set of
# characters Unicode treats as a space (category Zs, plus the
# Mongolian vowel separator, which behaves like one).
SPACES = {
    "\u00A0": " ",  # NO-BREAK SPACE
    "\u1680": " ",  # OGHAM SPACE MARK
    "\u2000": " ",  # EN QUAD
    "\u2001": " ",  # EM QUAD
    "\u2002": " ",  # EN SPACE
    "\u2003": " ",  # EM SPACE
    "\u2004": " ",  # THREE-PER-EM SPACE
    "\u2005": " ",  # FOUR-PER-EM SPACE
    "\u2006": " ",  # SIX-PER-EM SPACE
    "\u2007": " ",  # FIGURE SPACE
    "\u2008": " ",  # PUNCTUATION SPACE
    "\u2009": " ",  # THIN SPACE
    "\u200A": " ",  # HAIR SPACE
    "\u202F": " ",  # NARROW NO-BREAK SPACE
    "\u205F": " ",  # MEDIUM MATHEMATICAL SPACE
    "\u3000": " ",  # IDEOGRAPHIC SPACE
    "\u180E": " ",  # MONGOLIAN VOWEL SEPARATOR
}

# Zero-width and invisible format characters -> deleted.
ZERO_WIDTH = {
    "​": "",  # ZERO WIDTH SPACE
    "‌": "",  # ZERO WIDTH NON-JOINER
    "‍": "",  # ZERO WIDTH JOINER
    "⁠": "",  # WORD JOINER
    "⁡": "",  # FUNCTION APPLICATION
    "⁢": "",  # INVISIBLE TIMES
    "⁣": "",  # INVISIBLE SEPARATOR
    "⁤": "",  # INVISIBLE PLUS
    "­": "",  # SOFT HYPHEN
    "﻿": "",  # ZERO WIDTH NO-BREAK SPACE (BOM), if it survives rule 1
}

# Dash-class codepoints, mapped to the em dash U+2014 ONLY when
# flatten_dashes=True.  Off by default: the corpus uses both U+2013
# (en dash, 43x) and U+2014 (em dash, 484x) and the scan gives no
# evidence that they are the same glyph - the distinction may be
# marker's width quantization of one authorial dash.  Adjudicate with
# the ensemble OCR of Chainlink #4 before ever flattening by default.
#
# Membership rule: a codepoint is in this class only if it is
# NFKC-stable or NFKC-decomposes to another member of the class
# (e.g. U+2011 -> U+2010, U+208B -> U+2212, U+FE58 -> U+2014),
# so every member is reachable by the flatten step.  Hyphen-minus
# forms (U+FE63 small hyphen-minus, U+FF0D fullwidth hyphen-minus)
# are deliberately EXCLUDED: NFKC correctly sends them to U+002D,
# the compound linker (boy-roid, drug-embryo), which must never be
# flattened into an em dash.  U+2017 double low line is excluded
# too: it is a table character whose NFKC form is space + combining
# low line, not a prose dash.
DASH_CLASS = (
    "\u2010"  # HYPHEN
    "\u2011"  # NON-BREAKING HYPHEN (NFKC -> U+2010)
    "\u2012"  # FIGURE DASH
    "\u2013"  # EN DASH
    "\u2014"  # EM DASH
    "\u2015"  # HORIZONTAL BAR
    "\u2043"  # HYPHEN BULLET
    "\u2212"  # MINUS SIGN
    "\u208B"  # SUBSCRIPT MINUS (NFKC -> U+2212)
    "\uFE58"  # SMALL EM DASH (NFKC -> U+2014)
)

# Control characters (Unicode category Cc, except \t and \n) -> space.
# Remaining format characters (category Cf, not already mapped above)
# -> deleted.  The table is built per call in _control_table().

BOM = "﻿"  # ZERO WIDTH NO-BREAK SPACE / byte-order mark


def _translate_table(mapping):
    return {ord(k): v for k, v in mapping.items()}


LIGATURE_TABLE = _translate_table(LIGATURES)
QUOTE_TABLE = _translate_table(QUOTES)
SPACE_TABLE = _translate_table(SPACES)
ZERO_WIDTH_TABLE = _translate_table(ZERO_WIDTH)
FLATTEN_DASH_TABLE = {ord(c): "—" for c in DASH_CLASS}


def _control_table(text):
    """Build the translate table for stray control/format characters in
    ``text``: Cc controls (except tab/newline) become a space, any other
    format character (Cf) not already handled becomes deleted."""
    handled = set(ZERO_WIDTH) | {BOM}
    table = {}
    for ch in set(text):
        cat = unicodedata.category(ch)
        if cat == "Cc" and ch not in "\t\n":
            table[ord(ch)] = " "
        elif cat == "Cf" and ch not in handled:
            table[ord(ch)] = ""
    return table


# --------------------------------------------------------------------------
# Rule appliers: each takes text, returns (new_text, n_events)
# --------------------------------------------------------------------------
def _apply_translate(text, table):
    if not table:
        return text, 0
    counter = collections.Counter(text)
    events = sum(n for ch, n in counter.items() if ord(ch) in table)
    return text.translate(table), events


def _apply_nfkc(text):
    counter = collections.Counter(text)
    events = sum(
        n for ch, n in counter.items()
        if unicodedata.normalize("NFKC", ch) != ch
    )
    return unicodedata.normalize("NFKC", text), events


def _apply_crlf(text):
    events = text.count("\r\n") + text.count("\r")
    if not events:
        return text, 0
    return text.replace("\r\n", "\n").replace("\r", "\n"), events


def _apply_bom(text):
    events = text.count(BOM)
    if not events:
        return text, 0
    return text.replace(BOM, ""), events


# The ordered rule pipeline.  Names and descriptions appear in the
# freeze report and in freeze.json.  Order matters: the
# semantically named rules (quotes, spaces) run BEFORE the
# generic NFKC decomposition so that (a) typographic quotes and
# primes claim their characters before NFKC splits double primes
# (U+2033 -> U+2032 U+2032) into single primes, and (b) fire
# counts are attributed to the rule a human would name.  The
# explicit ligature table runs AFTER NFKC as pure defence in
# depth.
RULES = [
    ("bom", "strip byte-order mark (U+FEFF)", _apply_bom),
    ("crlf", "CRLF and CR line endings -> LF", _apply_crlf),
    ("quotes", "typographic/smart quotes and primes -> ASCII ' and \"",
     lambda t: _apply_translate(t, QUOTE_TABLE)),
    ("spaces", "unicode space characters (NBSP, thin space, ...) -> space",
     lambda t: _apply_translate(t, SPACE_TABLE)),
    ("compat", "NFKC: compatibility decomposition + canonical composition "
               "(ligatures, fullwidth forms, superscripts, ...)", _apply_nfkc),
    ("ligatures", "explicit Latin ligature map (redundant with NFKC, "
                  "defence in depth)",
     lambda t: _apply_translate(t, LIGATURE_TABLE)),
    ("zero_width", "zero-width/invisible format characters -> deleted",
     lambda t: _apply_translate(t, ZERO_WIDTH_TABLE)),
    ("control", "stray Cc controls -> space; other format chars -> deleted",
     lambda t: _apply_translate(t, _control_table(t))),
]

# Mapping tables that charset.tsv records as "normalize"/"strip" rows,
# keyed by the rule that owns them.  Used by the freeze builder (to
# write the table) and by the tests (to check code and table agree).
MAPPED_RULES = {
    "ligatures": LIGATURES,
    "quotes": QUOTES,
    "spaces": SPACES,
    "zero_width": ZERO_WIDTH,
}


# ==========================================================================
# The normalizer
# ==========================================================================
def normalize(text, flatten_dashes=False):
    """Return the canonically-normalized form of ``text``.

    Applies the frozen rule pipeline above.  With ``flatten_dashes=True``
    the dash class (hyphens, en/em dashes, minus signs, ...) is mapped to
    the em dash U+2014; default False keeps every dash codepoint as it
    appears, because the distinction is evidence in this corpus.
    """
    for _name, _desc, applier in RULES:
        text, _n = applier(text)
    if flatten_dashes:
        text = text.translate(FLATTEN_DASH_TABLE)
    return text


def normalize_report(text, flatten_dashes=False):
    """normalize() plus per-rule event counts, for reporting.

    Returns (normalized_text, [(rule_name, description, events), ...],
    total_events).
    """
    per_rule = []
    total = 0
    for name, desc, applier in RULES:
        text, events = applier(text)
        per_rule.append((name, desc, events))
        total += events
    if flatten_dashes:
        counter = collections.Counter(text)
        events = sum(n for ch, n in counter.items() if ch in DASH_CLASS)
        text = text.translate(FLATTEN_DASH_TABLE)
        per_rule.append(("flatten_dashes", "dash class -> em dash (opt-in)",
                         events))
        total += events
    return text, per_rule, total


def inventory(text):
    """Integer codepoint -> count counter for ``text``
    (the L0 glyph inventory)."""
    return collections.Counter(ord(ch) for ch in text)


# ==========================================================================
# charset.tsv I/O and freeze verification
# ==========================================================================
CHARSET_HEADER = [
    "codepoint", "char", "unicode_name", "category",
    "corpus_count", "raw_ocr_count", "decision", "canonical", "rule",
    "notes",
]


def parse_codepoint(s):
    """Parse a charset.tsv codepoint field ('U+0041' or '0041')
    into the integer codepoint."""
    s = s.strip()
    if s.upper().startswith("U+"):
        s = s[2:]
    return int(s, 16)


def load_charset(path=CHARSET_TSV):
    """Parse charset.tsv into a list of row dicts."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        cells += [""] * (len(CHARSET_HEADER) - len(cells))
        rows.append(dict(zip(CHARSET_HEADER, cells)))
    return rows


def verify_against_charset(text, rows):
    """Check that every codepoint of ``text`` is covered by charset rows.

    Returns (ok, violations) where violations is a list of
    (codepoint, char, count) for codepoints with no row.
    """
    known = set()
    for row in rows:
        try:
            known.add(parse_codepoint(row["codepoint"]))
        except (ValueError, KeyError):
            continue
    counts = inventory(text)
    violations = [
        (cp, chr(cp), n) for cp, n in sorted(counts.items()) if cp not in known
    ]
    return not violations, violations


# ==========================================================================
# CLI
# ==========================================================================
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_normalize(args):
    with open(args.file, encoding="utf-8") as fh:
        text = fh.read()
    out, per_rule, total = normalize_report(text, flatten_dashes=args.flatten_dashes)
    print(f"file: {args.file}")
    print(f"characters in:  {len(text):,}")
    print(f"characters out: {len(out):,}")
    print(f"normalization events: {total:,}")
    for name, desc, events in per_rule:
        if events or args.verbose:
            print(f"  {name:15} {events:8,}  {desc}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"wrote: {args.out}")
    if args.check_idempotent:
        twice = normalize(out, flatten_dashes=args.flatten_dashes)
        print(f"idempotent: {twice == out}")
    return 0


def cmd_verify(args):
    with open(args.file, encoding="utf-8") as fh:
        text = fh.read()
    rows = load_charset(args.charset)
    ok, violations = verify_against_charset(text, rows)
    counts = inventory(text)
    print(f"charset: {args.charset} ({len(rows)} rows)")
    print(f"corpus:  {args.file} ({len(counts)} distinct codepoints, "
          f"{len(text):,} chars)")
    if ok:
        print("OK: every corpus codepoint is covered by the frozen charset")
        return 0
    print("VIOLATIONS - codepoints in the corpus but not in charset.tsv:")
    for cp, ch, n in violations:
        print(f"  U+{cp:04X} {ch!r} x{n:,}")
    return 1


def cmd_freeze(args):
    # Delegated to work/build_charset.py so this module stays the
    # lean, reusable normalizer that downstream tasks import.
    if WORK_DIR not in sys.path:
        sys.path.insert(0, WORK_DIR)
    import build_charset

    return build_charset.build(
        corpus=args.corpus,
        raw_dump=args.raw_dump,
        outdir=args.outdir,
        run_tests=not args.skip_tests,
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    SUBCOMMANDS = {"normalize", "verify", "freeze"}
    if argv and argv[0] not in SUBCOMMANDS:
        argv = ["normalize"] + argv  # bare invocation = normalize

    parser = argparse.ArgumentParser(
        description="Frozen unicode normalization for the Blood Electric "
                    "corpus (Chainlink #5).")
    sub = parser.add_subparsers(dest="command")

    p_norm = sub.add_parser(
        "normalize", help="normalize a text file and print a per-rule report")
    p_norm.add_argument("file")
    p_norm.add_argument("-o", "--out", help="write normalized text here")
    p_norm.add_argument("--flatten-dashes", action="store_true",
                        help="map the dash class to em dash U+2014 "
                             "(off by default: dashes are evidence)")
    p_norm.add_argument("--check-idempotent", action="store_true")
    p_norm.add_argument("-v", "--verbose", action="store_true",
                        help="show zero-event rules too")
    p_norm.set_defaults(func=cmd_normalize)

    p_ver = sub.add_parser(
        "verify", help="check that a text file's codepoints are all in "
                       "charset.tsv")
    p_ver.add_argument("file")
    p_ver.add_argument("--charset", default=CHARSET_TSV)
    p_ver.set_defaults(func=cmd_verify)

    p_frz = sub.add_parser(
        "freeze", help="rebuild charset.tsv + freeze.json + report from "
                       "the corpus")
    p_frz.add_argument("--corpus", default=DEFAULT_CORPUS)
    p_frz.add_argument("--raw-dump", default=DEFAULT_RAW_DUMP)
    p_frz.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p_frz.add_argument("--skip-tests", action="store_true")
    p_frz.set_defaults(func=cmd_freeze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
