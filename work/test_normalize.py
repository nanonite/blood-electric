#!/usr/bin/env python3
"""
Tests for the frozen unicode normalizer (Chainlink #5).

Run from the workspace root (inside the `marker` conda env):

  python work/test_normalize.py
  python -m unittest discover -s work -p 'test_*.py'

The suite has three layers:

  * unit tests for every frozen rule (ligatures, quotes, spaces,
    zero-width, controls, NFKC, line endings, dash policy);
  * freeze-consistency tests: the frozen corpus must be a
    measured no-op, every corpus codepoint must be listed in
    charset.tsv, and charset.tsv must agree with the module's
    mapping tables (so code, table and test cannot drift);
  * CLI smoke tests.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(
    ROOT, "docs", "markdown", "blood-electric", "blood-electric.md"
)
CHARSET = os.path.join(ROOT, "analysis", "charset", "charset.tsv")


# ==========================================================================
# Unit tests: the frozen rules
# ==========================================================================
class LigatureTest(unittest.TestCase):
    def test_common_ligatures(self):
        self.assertEqual(normalize.normalize("ﬁne ﬂow ofﬁce"),
                         "fine flow office")
        self.assertEqual(normalize.normalize("ﬁ"), "fi")
        self.assertEqual(normalize.normalize("ﬂ"), "fl")
        self.assertEqual(normalize.normalize("ﬀ"), "ff")
        self.assertEqual(normalize.normalize("ﬃ"), "ffi")
        self.assertEqual(normalize.normalize("ﬄ"), "ffl")
        self.assertEqual(normalize.normalize("ﬅ"), "st")
        self.assertEqual(normalize.normalize("ﬆ"), "st")

    def test_latin_letter_ligatures(self):
        self.assertEqual(normalize.normalize("Ĳ"), "IJ")
        self.assertEqual(normalize.normalize("ĳ"), "ij")
        self.assertEqual(normalize.normalize("Ǆ"), "DŽ")
        self.assertEqual(normalize.normalize("ǆ"), "dž")
        self.assertEqual(normalize.normalize("Ǳ"), "DZ")
        self.assertEqual(normalize.normalize("ǳ"), "dz")

    def test_ligature_letters_are_preserved(self):
        # OE/oe are legitimate letters (French/Dutch), not
        # typographic presentation forms; Unicode does not
        # decompose them and neither does the freeze.
        self.assertEqual(normalize.normalize("Œuvre café"),
                         "Œuvre café")

    def test_ligature_map_matches_nfkc(self):
        # The explicit map is defence in depth for NFKC, so it
        # must never diverge from the standard.
        import unicodedata
        for src, dst in normalize.LIGATURES.items():
            self.assertEqual(
                unicodedata.normalize("NFKC", src), dst,
                msg=f"U+{ord(src):04X}")

    def test_dash_class_members_are_reachable(self):
        # Every DASH_CLASS member must be NFKC-stable or
        # NFKC-decompose to another member, so the opt-in
        # flatten step can always see it.
        import unicodedata
        for ch in normalize.DASH_CLASS:
            nfkc = unicodedata.normalize("NFKC", ch)
            self.assertTrue(
                nfkc == ch or nfkc in normalize.DASH_CLASS,
                msg=f"U+{ord(ch):04X} -> {nfkc!r}")


class QuoteTest(unittest.TestCase):
    def test_smart_single_quotes_become_apostrophe(self):
        self.assertEqual(normalize.normalize("‘don’t’"), "'don't'")
        self.assertEqual(normalize.normalize("‚low‛"), "'low'")

    def test_smart_double_quotes_become_straight(self):
        self.assertEqual(normalize.normalize("“quotes”"), '"quotes"')
        self.assertEqual(normalize.normalize("„low‟"), '"low"')

    def test_primes_become_quotes(self):
        # OCR of a scanned book frequently emits primes as
        # quotes.  Built from chr() so the codepoints are
        # unambiguous; the quotes rule runs before NFKC so the
        # double prime maps to one straight double quote (NFKC
        # alone would split it into two single primes).
        self.assertEqual(normalize.normalize("a" + chr(0x2032) + "b"),
                         "a'b")
        self.assertEqual(normalize.normalize("a" + chr(0x2033) + "b"),
                         'a"b')
        self.assertEqual(normalize.normalize("a" + chr(0x2035) + "b"),
                         "a'b")
        self.assertEqual(normalize.normalize("a" + chr(0x2036) + "b"),
                         'a"b')

    def test_straight_quotes_untouched(self):
        self.assertEqual(normalize.normalize("'a' \"b\""), "'a' \"b\"")


class SpaceTest(unittest.TestCase):
    def test_nbsp_and_friends_become_space(self):
        for cp in (
            " ", "᠎", " ", " ", " ",
            " ", " ", " ", " ",
            " ", " ", " ", " ",
            " ", " ", "　", "᠎",
        ):
            self.assertEqual(normalize.normalize(f"a{cp}b"), "a b",
                             msg=f"U+{ord(cp):04X}")

    def test_plain_space_untouched(self):
        self.assertEqual(normalize.normalize("a b"), "a b")


class ZeroWidthTest(unittest.TestCase):
    def test_invisible_characters_deleted(self):
        for cp in ("​", "‌", "‍", "⁠",
                   "⁡", "⁢", "⁣", "⁤", "­"):
            self.assertEqual(normalize.normalize(f"a{cp}b"), "ab",
                             msg=f"U+{ord(cp):04X}")

    def test_bom_stripped(self):
        self.assertEqual(normalize.normalize("﻿abc"), "abc")
        self.assertEqual(normalize.normalize("a﻿b"), "ab")


class LineEndingTest(unittest.TestCase):
    def test_crlf_and_cr_become_lf(self):
        self.assertEqual(normalize.normalize("a\r\nb\rc"), "a\nb\nc")

    def test_lf_untouched(self):
        self.assertEqual(normalize.normalize("a\nb"), "a\nb")


class CompatTest(unittest.TestCase):
    def test_nfkc_fullwidth_and_superscripts(self):
        self.assertEqual(normalize.normalize("ＡＢＣ"), "ABC")
        self.assertEqual(normalize.normalize("①"), "1")
        self.assertEqual(normalize.normalize("x²"), "x2")
        self.assertEqual(normalize.normalize("½"), "1⁄2")

    def test_corpus_glyphs_are_nfkc_stable(self):
        for ch in "©Ø–—•":
            self.assertEqual(
                unicodedata_normalize(ch), ch,
                msg=f"expected NFKC-stable: U+{ord(ch):04X}")


class ControlTest(unittest.TestCase):
    def test_stray_cc_controls_become_space(self):
        self.assertEqual(normalize.normalize("a\x00b\x0bb"), "a b b")

    def test_tab_and_newline_survive(self):
        self.assertEqual(normalize.normalize("a\tb\nc"), "a\tb\nc")


class DashPolicyTest(unittest.TestCase):
    def test_dashes_preserved_by_default(self):
        # The corpus uses both en dash and em dash; the distinction
        # is evidence and must not be flattened by default.
        self.assertEqual(normalize.normalize("a–b—c"), "a–b—c")

    def test_flatten_dashes_is_opt_in(self):
        self.assertEqual(
            normalize.normalize("a–b—c", flatten_dashes=True),
            "a—b—c")

    def test_flatten_dash_class(self):
        for ch in normalize.DASH_CLASS:
            self.assertEqual(
                normalize.normalize(f"a{ch}b", flatten_dashes=True),
                "a—b", msg=f"U+{ord(ch):04X}")


class PreservationTest(unittest.TestCase):
    def test_case_is_evidence(self):
        self.assertEqual(normalize.normalize("ToKAGE ADAM ToK"),
                         "ToKAGE ADAM ToK")

    def test_diacritics_are_evidence(self):
        # Ø is a genuine glyph of the section header MØNKEY/gene-dub.
        self.assertEqual(normalize.normalize("MØNKEY/gene-dub"),
                         "MØNKEY/gene-dub")

    def test_book_operators_survive(self):
        clause = "<<the suck=blood chromosome//Fear=cell::of ToKAGE>>"
        self.assertEqual(normalize.normalize(clause), clause)

    def test_ascii_passthrough(self):
        sample = "The quick brown fox jumps over the lazy dog 0123456789"
        self.assertEqual(normalize.normalize(sample), sample)

    def test_empty_string(self):
        self.assertEqual(normalize.normalize(""), "")

    def test_idempotent(self):
        nasty = ("﻿ﬁne ‘quotes’ – dashes —  NBSP​ZW"
                 "a\rb\r\nc\x00d ½ Ａ")
        once = normalize.normalize(nasty)
        self.assertEqual(normalize.normalize(once), once)

    def test_event_counts_are_counted(self):
        # The per-rule fire counts are themselves checked
        # output (the freeze report publishes them).  Built
        # from \u escapes so the look-alike characters (NBSP,
        # BOM) are unambiguous.
        text = (
            "ﬁne ‘quotes’ a" + chr(0x00A0) + "b"
            + chr(0xFEFF) + "c\rd"
        )
        out, per_rule, total = normalize.normalize_report(text)
        events = {name: n for name, _desc, n in per_rule}
        self.assertEqual(events["compat"], 1)      # the ligature ﬁ
        self.assertEqual(events["quotes"], 2)      # ‘ and ’
        self.assertEqual(events["spaces"], 1)      # the NBSP
        self.assertEqual(events["bom"], 1)         # the BOM
        self.assertEqual(events["crlf"], 1)        # the \r
        self.assertEqual(total, 6)
        self.assertEqual(out, "fine 'quotes' a bc\nd")


# ==========================================================================
# Freeze consistency: the frozen corpus and the frozen charset
# ==========================================================================
@unittest.skipUnless(os.path.exists(CORPUS), "corpus not present")
class CorpusFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CORPUS, encoding="utf-8") as fh:
            cls.text = fh.read()

    def test_corpus_normalization_is_a_measured_noop(self):
        # The frozen normalizer must not alter the frozen corpus:
        # marker + ftfy already emit clean Unicode, so the freeze
        # locks that in.  Any event here means a rule fires on real
        # text and the freeze needs re-adjudication.
        out, per_rule, total = normalize.normalize_report(self.text)
        self.assertEqual(out, self.text)
        self.assertEqual(total, 0)
        for name, _desc, events in per_rule:
            self.assertEqual(events, 0, msg=f"rule {name} fired")

    def test_corpus_covered_by_charset(self):
        rows = normalize.load_charset(CHARSET)
        ok, violations = normalize.verify_against_charset(self.text, rows)
        self.assertTrue(ok, msg=f"uncovered codepoints: {violations}")

    def test_charset_counts_match_corpus(self):
        rows = normalize.load_charset(CHARSET)
        counts = normalize.inventory(self.text)
        for row in rows:
            cp = normalize.parse_codepoint(row["codepoint"])
            if row["decision"] != "keep":
                continue
            listed = int(row["corpus_count"])
            self.assertEqual(
                listed, counts.get(cp, 0),
                msg=f"count mismatch for U+{cp:04X}: "
                    f"charset says {listed}, corpus has {counts.get(cp, 0)}")

    def test_charset_agrees_with_module_rules(self):
        # charset.tsv is generated from the module's tables; this
        # test fails if someone edits one but not the other.
        rows = normalize.load_charset(CHARSET)
        table_for_rule = {}
        for row in rows:
            if row["decision"] not in ("normalize", "strip"):
                continue
            rule = row["rule"]
            src = normalize.parse_codepoint(row["codepoint"])
            target = row["canonical"]
            table_for_rule.setdefault(rule, {})[src] = target
        for rule, mapping in normalize.MAPPED_RULES.items():
            expected = {}
            for src, dst in mapping.items():
                expected[ord(src)] = dst
            got = table_for_rule.get(rule, {})
            self.assertEqual(
                got, expected,
                msg=f"charset.tsv rule '{rule}' disagrees with "
                    f"normalize.py mapping")

    def test_raw_ocr_stream_alphabet_matches(self):
        # Freeze invariant: the raw OCR stream and the markdown
        # corpus share the SAME non-ASCII alphabet (no glyph
        # introduced or destroyed by post-processing).  Per-glyph
        # counts may differ by the documented, located rendering
        # deltas - pinned here because this is a freeze:
        # the em dash is the only count delta (raw 486 vs
        # corpus 484; marker's line-joiner drops a trailing em
        # dash at two line joins, 0-based PDF pages 41 and 53 -
        # see charset_report.md section 2.3).
        import gzip
        import json as _json
        raw_counts = {}
        dump = os.path.join(
            ROOT, "analysis", "ocr_confidence",
            "raw_ocr_confidence.jsonl.gz")
        if not os.path.exists(dump):
            self.skipTest("raw OCR dump not present")
        with gzip.open(dump, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = _json.loads(line)
                for ch in rec.get("surya_text", ""):
                    if ord(ch) > 0x7F:
                        raw_counts[ord(ch)] = raw_counts.get(ord(ch), 0) + 1
        corpus_non_ascii = {
            cp: n for cp, n in normalize.inventory(self.text).items()
            if cp > 0x7F
        }
        self.assertEqual(
            set(raw_counts), set(corpus_non_ascii),
            msg="non-ASCII alphabet differs between raw OCR "
                "stream and corpus")
        deltas = {
            cp: raw_counts.get(cp, 0) - corpus_non_ascii.get(cp, 0)
            for cp in set(raw_counts) | set(corpus_non_ascii)
        }
        deltas = {cp: d for cp, d in deltas.items() if d != 0}
        self.assertEqual(deltas, {0x2014: 2},
                         msg="unexpected per-glyph count deltas: "
                             f"{deltas}")


# ==========================================================================
# CLI smoke tests
# ==========================================================================
@unittest.skipUnless(os.path.exists(CORPUS), "corpus not present")
class CliTest(unittest.TestCase):
    def test_cli_normalize_reports_zero_events(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "normalize.py"),
             CORPUS],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("normalization events: 0", proc.stdout)

    def test_cli_verify_passes(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "normalize.py"),
             "verify", CORPUS],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("OK", proc.stdout)


def unicodedata_normalize(ch):
    import unicodedata
    return unicodedata.normalize("NFKC", ch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
