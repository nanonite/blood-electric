#!/usr/bin/env python3
"""Regression tests for the spectral reading artifact (Chainlink #67).

Run from the workspace root:

    python work/test_spectral_reading.py
    python -m unittest discover -s work -p 'test_*.py'

What is pinned here:
  * the spectral reading is correctly derived from the v4 silence-removed
    stream (every silence site -> `. `, word sequence preserved exactly,
    and the re-derivation matches #13's published v4 stream line for line);
  * every headline number is computed, arithmetically closed, and equal to
    the published #13/#50 number it leans on - and a planted mismatch is a
    build error, not a silent drift;
  * the worked example is selected by a published rule and is the genuine
    before/after of one citable source line;
  * the framing note contains the required elements (derivation, provenance,
    can/cannot support, interpretive status) and never claims to be the
    real text;
  * the artifact is publishable and the build is deterministic.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import spectral_reading as sra
from silence_removal import (
    SRC,
    VARIANTS,
    load_pages,
    normalize_line,
    segment_stats,
    site_spans,
    text_metrics,
    transform_line,
    words,
)

HAVE_SRC = os.path.exists(SRC)
HAVE_UPSTREAM = (os.path.exists(sra.SR_SUMMARY)
                 and os.path.exists(sra.PPL_TSV))


def apply_v4(line):
    """Apply the v4 transform to a single source line."""
    treatment, only, sep = next(
        (t, o, s) for k, t, o, s in VARIANTS if k == sra.V4_KEY
    )
    new, _n, _f = transform_line(line, site_spans(line), treatment, only, sep)
    return normalize_line(new)


def setUpModule():
    """Build the artifact once, into a scratch dir, for the whole suite."""
    global BUILD_DIR, NUMBERS, READING_PAGES, PAGE_NOS, PAGES
    BUILD_DIR = tempfile.mkdtemp(prefix="spectral_reading_test_")
    NUMBERS = sra.build(BUILD_DIR)
    READING_PAGES, PAGE_NOS = sra.build_reading()
    PAGES = load_pages(SRC)


# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "layout corpus not present")
class DerivationTest(unittest.TestCase):
    """The spectral reading is correctly derived from the v4 stream."""

    def test_all_225_pages_present(self):
        self.assertEqual(len(PAGE_NOS), 225)
        self.assertEqual(PAGE_NOS, list(range(1, 226)))

    def test_reading_matches_v4_transform(self):
        """Every line of the reading equals the v4 transform of the source."""
        for pno in PAGE_NOS:
            raw_lines = PAGES[pno].split("\n")
            expected = [apply_v4(line) for line in raw_lines]
            expected = [line for line in expected if line]
            self.assertEqual(
                READING_PAGES[pno], expected,
                f"page {pno}: reading does not match v4 transform"
            )

    def test_word_sequence_preserved(self):
        """The reading preserves the baseline word sequence exactly."""
        reading = sra.reading_text(READING_PAGES, PAGE_NOS)
        base = sra.baseline_text(PAGES, PAGE_NOS)
        self.assertEqual(words(reading), words(base))

    def test_no_operators_remain(self):
        """No operator tokens survive in the reading."""
        full_text_parts = []
        for pno in PAGE_NOS:
            full_text_parts.extend(READING_PAGES[pno])
        full_text = "\n\n".join(full_text_parts) + "\n"
        for op in ("::", "//", "=", "/", "<<", ">>", "<", ">", "++", "|"):
            self.assertNotRegex(
                full_text, r"(?<![A-Za-z0-9_])" + re.escape(op) + r"(?![A-Za-z0-9_])",
                f"operator {op!r} survives in the reading"
            )

    def test_line_count_matches_source(self):
        """The reading never has more lines than the source page."""
        for pno in PAGE_NOS:
            raw = [l for l in PAGES[pno].split("\n") if l.strip()]
            reading = [l for l in READING_PAGES[pno] if l.strip()]
            self.assertLessEqual(len(reading), len(raw),
                                 f"page {pno}: reading has more lines than source")

    @unittest.skipUnless(os.path.exists(sra.V4_PUBLISHED),
                         "published v4 stream not present")
    def test_matches_published_v4_stream(self):
        """The re-derived reading equals #13's published v4 stream."""
        published = sra.published_v4_stream()
        self.assertEqual(sorted(published), PAGE_NOS)
        for pno in PAGE_NOS:
            self.assertEqual(published[pno], READING_PAGES[pno],
                             f"page {pno}: diverges from the published stream")


# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "layout corpus not present")
class ComputationTest(unittest.TestCase):
    """Every headline number is computed, closed, and pinned."""

    def test_headline_numbers_pinned(self):
        n = NUMBERS
        self.assertEqual(n["n_pages"], 225)
        self.assertEqual(n["n_words"], 77392)
        self.assertEqual(n["n_operator_tokens"], 10949)
        self.assertEqual(n["n_dash_sites"], 696)
        self.assertEqual(n["n_sites"], 11645)
        self.assertEqual(n["n_site_runs"], 10435)
        self.assertEqual(n["n_periods"], 10457)
        self.assertEqual(n["baseline_periods"], 22)
        self.assertEqual(n["n_segments"], 10136)
        self.assertEqual(n["median_words"], 6.0)
        self.assertEqual(n["baseline_segments"], 17)

    def test_headline_rates_pinned(self):
        n = NUMBERS
        self.assertAlmostEqual(n["frac_upper"], 0.255327545382794, places=9)
        self.assertAlmostEqual(n["frac_upper_ge5"], 0.27554129911788294,
                               places=9)
        self.assertAlmostEqual(n["frac_single_word"], 0.12726913970007891,
                               places=9)
        self.assertAlmostEqual(n["baseline_frac_upper"], 0.5294117647058824,
                               places=9)

    def test_period_closure(self):
        """Inserted periods + pre-existing = total; runs never exceed sites."""
        n = NUMBERS
        self.assertEqual(n["n_periods"],
                         n["n_site_runs"] + n["baseline_periods"])
        self.assertLessEqual(n["n_site_runs"], n["n_sites"])
        # the difference is sites sharing a junction (multi-operator runs)
        self.assertGreater(n["n_sites"] - n["n_site_runs"], 0)

    def test_numbers_come_from_thirteen_primitives(self):
        """Recomputing with #13's own functions reproduces the numbers."""
        reading = sra.reading_text(READING_PAGES, PAGE_NOS)
        base = sra.baseline_text(PAGES, PAGE_NOS)
        rstats = segment_stats(reading)
        bstats = segment_stats(base)
        self.assertEqual(rstats["n_segments"], NUMBERS["n_segments"])
        self.assertEqual(rstats["median_words"], NUMBERS["median_words"])
        self.assertEqual(bstats["n_segments"], NUMBERS["baseline_segments"])
        self.assertEqual(text_metrics(base)["periods"],
                         NUMBERS["baseline_periods"])

    def test_baseline_skeleton_is_invisible(self):
        """The untouched corpus has almost no sentence punctuation."""
        n = NUMBERS
        self.assertEqual(n["baseline_periods"], 22)
        self.assertEqual(n["baseline_segments"], 17)
        self.assertGreater(n["n_segments"], 100 * n["baseline_segments"])

    def test_uppercase_rate_is_length_controlled(self):
        """The disclosed rates are the length-controlled companions too."""
        n = NUMBERS
        self.assertGreater(n["frac_upper_ge5"], n["frac_upper"])
        self.assertGreater(n["baseline_frac_upper_ge5"],
                           n["baseline_frac_upper"])

    def test_residue_is_itemized_not_dropped(self):
        """The token-less segments are counted and named, not discarded."""
        n = NUMBERS
        self.assertEqual(n["n_tokenless_segments"], 5)
        self.assertEqual(n["tokenless_examples"], [":"])


# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "layout corpus not present")
class WorkedExampleTest(unittest.TestCase):
    """The worked example is genuine, citable, and rule-selected."""

    @classmethod
    def setUpClass(cls):
        cls.example = sra.pick_worked_example(PAGES, PAGE_NOS)

    def test_selection_rule_is_what_the_note_claims(self):
        """The example shows all three operator classes the note names."""
        src = self.example["source"]
        for op in ("::", "//", "="):
            self.assertIn(op, src)

    def test_citation_line_exists_in_the_source(self):
        page = self.example["page"]
        line_no = self.example["line_no"]
        src_lines = PAGES[page].split("\n")
        self.assertLessEqual(line_no, len(src_lines))
        self.assertEqual(src_lines[line_no - 1].strip(),
                         self.example["source"])

    def test_reading_side_is_the_v4_transform(self):
        line = PAGES[self.example["page"]].split("\n")[self.example["line_no"] - 1]
        self.assertEqual(self.example["reading"], apply_v4(line))

    def test_word_tokens_survive_the_example(self):
        self.assertEqual(words(self.example["reading"]),
                         words(self.example["source"]))


# ---------------------------------------------------------------------------
class ReconciliationTest(unittest.TestCase):
    """The build reconciles against the published #13/#50 numbers."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(BUILD_DIR, "reconciliation.tsv")
        with open(path, encoding="utf-8", newline="") as f:
            cls.rows = list(csv.DictReader(f, delimiter="\t"))

    def test_every_row_matches(self):
        self.assertTrue(self.rows)
        for r in self.rows:
            self.assertNotEqual(r["match"], "MISMATCH", r["quantity"])

    def test_reconciliation_pins_the_load_bearing_numbers(self):
        by_quantity = {r["quantity"]: r for r in self.rows}
        pinned = {
            "pages": "225",
            "word tokens (lexemes)": "77392",
            "periods in the reading": "10457",
            "reading segments (word-bearing)": "10136",
            "baseline segments (v0)": "17",
        }
        for quantity, value in pinned.items():
            self.assertIn(quantity, by_quantity)
            self.assertEqual(by_quantity[quantity]["here"], value)
            self.assertEqual(by_quantity[quantity]["published"], value)

    @unittest.skipUnless(HAVE_UPSTREAM, "published upstream not present")
    def test_perplexity_verdict_is_read_from_fifty(self):
        by_quantity = {r["quantity"]: r for r in self.rows}
        row = by_quantity["GPT-2 perplexity, v4 reading"]
        self.assertAlmostEqual(float(row["here"]), 84.051671, places=6)
        self.assertAlmostEqual(float(row["published"]), 84.051671, places=6)
        v0 = float(by_quantity["GPT-2 perplexity, v0 baseline"]["here"])
        self.assertLess(float(row["here"]), v0)

    def test_mismatch_is_a_build_error(self):
        """A drifted number fails the check instead of being published."""
        rows = [{"quantity": "pages", "source": "test",
                 "published": 225, "here": 224, "match": "MISMATCH"}]
        with self.assertRaises(sra.SpectralError):
            sra.check_reconciliation(rows)

    def test_absent_upstream_is_not_a_claim(self):
        """With no published artifact every row is n/a, and nothing raises."""
        rows = sra.reconcile(NUMBERS, {"sr_summary": False, "ppl_tsv": False})
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["match"], "n/a (artifact absent)")
        sra.check_reconciliation(rows)  # must not raise


# ---------------------------------------------------------------------------
class FramingNoteTest(unittest.TestCase):
    """The framing note contains the required elements."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(BUILD_DIR, "spectral_reading.md"),
                  encoding="utf-8") as f:
            cls.text = f.read()

    def test_title_present(self):
        self.assertIn("# Spectral Reading: Blood Electric", self.text)

    def test_explicitly_not_the_real_text(self):
        """The reading is explicitly marked as NOT the real text."""
        self.assertIn("NOT the real text", self.text)
        self.assertIn("not the real text", self.text.lower())
        self.assertNotIn("this is the real text", self.text.lower())

    def test_framing_note_present(self):
        self.assertIn("## Framing Note", self.text)

    def test_derivation_documented(self):
        self.assertIn("### Derivation", self.text)
        self.assertIn("v4_operators_to_period", self.text)
        self.assertIn("blood-electric.layout.md", self.text)

    def test_can_support_documented(self):
        self.assertIn("### What the reading can support", self.text)
        self.assertIn("sentence-level analysis", self.text.lower())
        self.assertIn("rhythm", self.text.lower())

    def test_cannot_support_documented(self):
        self.assertIn("### What the reading cannot support", self.text)
        self.assertIn("word-level semantics", self.text.lower())
        self.assertIn("NOT a transcription", self.text)

    def test_interpretive_status_clear(self):
        self.assertIn("### Interpretive status", self.text)
        self.assertIn("interpretation", self.text.lower())
        self.assertIn("not a measurement", self.text.lower())

    def test_spectral_status_clear(self):
        """The reading's spectral status is clear."""
        self.assertIn("spectral", self.text.lower())
        self.assertIn("ghost", self.text.lower())
        self.assertIn("haunts", self.text.lower())

    def test_reading_section_present(self):
        self.assertIn("## The Reading", self.text)

    def test_colophon_present(self):
        self.assertIn("## Colophon", self.text)

    def test_page_tags_preserved(self):
        """All 225 page tags are present in the reading."""
        pages = [int(m) for m in re.findall(r"<!-- page (\d+) -->", self.text)]
        self.assertEqual(pages, list(range(1, 226)))

    def test_no_operators_in_reading_body(self):
        """No operator tokens appear in the reading body."""
        idx = self.text.find("## The Reading")
        self.assertGreater(idx, 0)
        body = self.text[idx:]
        for op in ("::", "//", "<<", ">>", "++"):
            self.assertNotIn(op, body, f"operator {op!r} in reading body")

    def test_provenance_section_present(self):
        """Provenance is a section of its own, naming the upstream files."""
        self.assertIn("### Provenance and evidence", self.text)
        self.assertIn("analysis/silence_removal/summary.json", self.text)
        self.assertIn("sentence_shape.tsv", self.text)

    def test_evidence_table_carries_the_computed_numbers(self):
        """The evidence table's values are the ones actually computed."""
        n = NUMBERS
        for needle in (f"{n['n_segments']:,}", f"{n['n_words']:,}",
                       f"{n['baseline_frac_upper']:.1%}",
                       f"{n['n_sites']:,}"):
            self.assertIn(needle, self.text)
        self.assertIn("perplexity_by_variant.tsv", self.text)

    def test_length_control_disclosed(self):
        """The uppercase rate is reported with its length-controlled rate."""
        self.assertIn(">= 5 words", self.text)
        self.assertIn("length-controlled", self.text)

    def test_residue_disclosed(self):
        """The token-less segments are disclosed, not dropped silently."""
        self.assertIn("Counting-rule residue", self.text)
        self.assertIn("lone `:`", self.text)

    def test_dash_class_disclosed(self):
        """The transform's dash scope is stated, not glossed over."""
        self.assertIn(f"{NUMBERS['n_dash_sites']:,}", self.text)
        self.assertIn("dash", self.text.lower())

    def test_no_fabricated_test_transcript(self):
        """No pasted test-run transcript or generated stamp is shipped."""
        self.assertNotRegex(self.text, r"Ran \d+ tests")
        self.assertNotRegex(self.text, r"Generated: 20\d\d-\d\d-\d\d")


# ---------------------------------------------------------------------------
class SummaryTest(unittest.TestCase):
    """The summary.json headline numbers are correct."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(BUILD_DIR, "summary.json"), encoding="utf-8") as f:
            cls.summary = json.load(f)

    def test_transform_is_v4(self):
        self.assertEqual(self.summary["transform"], sra.V4_KEY)

    def test_n_pages(self):
        self.assertEqual(self.summary["n_pages"], 225)

    def test_n_words(self):
        """Word count matches the baseline exactly."""
        self.assertEqual(self.summary["n_words"], 77392)

    def test_n_segments(self):
        """Segment count pins #13's word-bearing rule exactly."""
        self.assertEqual(self.summary["n_segments"], 10136)

    def test_interpretive_status(self):
        self.assertEqual(
            self.summary["interpretive_status"],
            "spectral reading, not the real text"
        )

    def test_can_support_list(self):
        self.assertIsInstance(self.summary["can_support"], list)
        self.assertGreater(len(self.summary["can_support"]), 0)

    def test_cannot_support_list(self):
        self.assertIsInstance(self.summary["cannot_support"], list)
        self.assertGreater(len(self.summary["cannot_support"]), 0)

    def test_period_closure_recorded(self):
        s = self.summary
        self.assertEqual(s["n_periods"],
                         s["n_site_runs"] + s["baseline_periods"])
        self.assertEqual(s["n_periods"], 10457)

    @unittest.skipUnless(HAVE_UPSTREAM, "published upstream not present")
    def test_perplexity_recorded_and_ordered(self):
        s = self.summary
        self.assertAlmostEqual(s["perplexity_v4"], 84.051671, places=6)
        self.assertAlmostEqual(s["perplexity_v0"], 106.21664, places=6)
        self.assertLess(s["perplexity_v4"], s["perplexity_v0"])


# ---------------------------------------------------------------------------
class PublishabilityTest(unittest.TestCase):
    """The artifact is publishable: well-formed, self-contained, honest."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(BUILD_DIR, "spectral_reading.md"),
                  encoding="utf-8") as f:
            cls.text = f.read()

    def test_markdown_structure(self):
        """The artifact has proper markdown structure."""
        self.assertIn("# ", self.text)   # H1 title
        self.assertIn("## ", self.text)  # H2 sections
        self.assertIn("### ", self.text)  # H3 subsections

    def test_self_contained(self):
        """The artifact explains itself without external references."""
        self.assertIn("What this is", self.text)
        self.assertIn("What this is not", self.text)
        self.assertIn("Derivation", self.text)

    def test_honest_about_nature(self):
        """The artifact is honest about its interpretive nature."""
        self.assertNotIn("this is the real text", self.text.lower())
        self.assertIn("transform", self.text.lower())
        self.assertIn("interpretation", self.text.lower())

    def test_citable(self):
        """The artifact is citable (page markers present)."""
        self.assertIn("<!-- page 1 -->", self.text)
        self.assertIn("<!-- page 225 -->", self.text)

    def test_no_placeholder_text(self):
        """No placeholder or TODO text remains."""
        self.assertNotIn("TODO", self.text)
        self.assertNotIn("PLACEHOLDER", self.text)
        self.assertNotIn("FIXME", self.text)

    def test_framing_precedes_reading(self):
        """The framing note comes before the reading (so no one mistakes it)."""
        framing_idx = self.text.find("## Framing Note")
        reading_idx = self.text.find("## The Reading")
        self.assertGreater(framing_idx, 0)
        self.assertGreater(reading_idx, 0)
        self.assertLess(framing_idx, reading_idx,
                        "framing note must come before the reading")


# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "layout corpus not present")
class DeterminismTest(unittest.TestCase):
    """The build is deterministic and byte-identical on rebuild."""

    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as second:
            sra.build(second)
            for name in ("spectral_reading.md", "summary.json",
                         "reconciliation.tsv", "diagnostics.tsv",
                         "spectral_reading_report.md"):
                with open(os.path.join(BUILD_DIR, name), "rb") as f:
                    first = f.read()
                with open(os.path.join(second, name), "rb") as f:
                    again = f.read()
                self.assertEqual(first, again, f"{name} is not deterministic")

    def test_no_wall_clock_anywhere(self):
        """No artifact carries a generated timestamp."""
        for name in ("spectral_reading.md", "summary.json",
                     "reconciliation.tsv", "diagnostics.tsv",
                     "spectral_reading_report.md"):
            with open(os.path.join(BUILD_DIR, name), encoding="utf-8") as f:
                self.assertNotRegex(f.read(), r"20\d\d-\d\d-\d\d",
                                    f"{name} carries a timestamp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
