#!/usr/bin/env python3
"""Regression tests for the POS-tagging module (Chainlink #37).

Run from the workspace root:

    python work/test_pos_tagging.py
    python -m unittest discover -s work -p 'test_*.py'

What is pinned here:
  * the tagger produces sensible tags for common English words;
  * the three-tier system (lexicon, suffix, context) works as specified;
  * the v3 stream and reference corpora are loaded and tagged;
  * the output artifacts exist and are well-formed.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import pos_tagging as pt  # noqa: E402

OUT = ROOT / "analysis" / "pos_tagging"


class LexiconTest(unittest.TestCase):
    """The curated lexicon parses correctly."""

    def test_lexicon_nonempty(self):
        self.assertGreater(len(pt.LEXICON), 800)

    def test_lexicon_has_function_words(self):
        for w in ("the", "a", "and", "of", "to", "in", "is", "was"):
            self.assertIn(w, pt.LEXICON)

    def test_lexicon_has_common_content_words(self):
        for w in ("time", "run", "good", "quickly", "dog"):
            self.assertIn(w, pt.LEXICON)

    def test_brown_verbs_loaded(self):
        self.assertGreater(len(pt.BROWN_VERBS), 100)


class TaggerTest(unittest.TestCase):
    """The three-tier tagger produces sensible tags."""

    def test_simple_sentence(self):
        tagged = pt.tag_tokens(["the", "dog", "runs", "quickly"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[0], "DT")   # the
        self.assertEqual(tags[1], "NN")   # dog
        self.assertEqual(tags[2], "VB")   # runs (context: after NN -> VB)
        self.assertEqual(tags[3], "RB")   # quickly

    def test_modal_verb(self):
        tagged = pt.tag_tokens(["she", "will", "go"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[0], "PRP")  # she
        self.assertEqual(tags[1], "MD")   # will
        self.assertEqual(tags[2], "VB")   # go

    def test_suffix_ing_is_vbg_or_nn(self):
        # "running" should get VBG from suffix (after "is" -> VBG via context)
        tagged = pt.tag_tokens(["he", "is", "running"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[2], "VBG")

    def test_suffix_ed_is_vbd(self):
        tagged = pt.tag_tokens(["she", "walked"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[1], "VBD")

    def test_suffix_ly_is_rb(self):
        tagged = pt.tag_tokens(["she", "ran", "quickly"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[2], "RB")

    def test_capitalized_unknown_is_nnp(self):
        tagged = pt.tag_tokens(["ToKAGE", "is", "here"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[0], "NNP")

    def test_context_disambiguation_det_nn(self):
        # After a determiner, a verb-looking word becomes a noun
        tagged = pt.tag_tokens(["the", "run"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[1], "NN")

    def test_context_disambiguation_md_vb(self):
        # After a modal, a noun-looking word becomes a verb
        tagged = pt.tag_tokens(["will", "run"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[1], "VB")

    def test_unknown_word_defaults_to_nn(self):
        tagged = pt.tag_tokens(["xyzabc"])
        tags = [t for _w, t in tagged]
        self.assertEqual(tags[0], "NN")

    def test_empty_input(self):
        self.assertEqual(pt.tag_tokens([]), [])


class CorpusTest(unittest.TestCase):
    """The v3 stream and reference corpora load correctly."""

    def test_v3_stream_exists(self):
        self.assertTrue(pt.V3_STREAM.exists())

    def test_v3_tokens_nonempty(self):
        tokens = pt.load_v3_tokens()
        self.assertGreater(len(tokens), 10000)

    def test_reference_corpora_exist(self):
        corpora = pt.load_reference_tokens()
        self.assertGreater(len(corpora), 5)

    def test_reference_tokens_nonempty(self):
        corpora = pt.load_reference_tokens()
        for name, tokens in corpora.items():
            self.assertGreater(len(tokens), 30, name)


class MetricsTest(unittest.TestCase):
    """The metrics functions produce sensible output."""

    def test_noun_verb_balance(self):
        tagged = pt.tag_tokens(["the", "dog", "runs", "quickly", "in", "the", "park"])
        bal = pt.noun_verb_balance(tagged)
        self.assertEqual(bal["total_tokens"], 7)
        self.assertGreater(bal["noun_count"], 0)
        self.assertGreater(bal["verb_count"], 0)

    def test_pos_distribution(self):
        tagged = pt.tag_tokens(["the", "dog", "runs"])
        dist = pt.pos_distribution(tagged)
        self.assertEqual(dist["DT"], 1)
        self.assertEqual(dist["NN"], 1)
        self.assertEqual(dist["VB"], 1)

    def test_chi_square(self):
        result = pt.chi_square_2x2(100, 50, 200, 100)
        self.assertIn("chi2", result)
        self.assertIn("p_value", result)
        self.assertGreaterEqual(result["p_value"], 0)
        self.assertLessEqual(result["p_value"], 1)


@unittest.skipUnless(OUT.exists(), "run work/pos_tagging.py first")
class ArtifactTest(unittest.TestCase):
    """The output artifacts exist and are well-formed."""

    def test_report_exists(self):
        self.assertTrue((OUT / "pos_comparison_report.md").exists())

    def test_summary_json_exists(self):
        self.assertTrue((OUT / "summary.json").exists())

    def test_summary_json_valid(self):
        with (OUT / "summary.json").open() as fh:
            s = json.load(fh)
        self.assertIn("v3_tokens", s)
        self.assertIn("ref_tokens", s)
        self.assertIn("noun_pct_delta", s)
        self.assertIn("p_value", s)

    def test_pos_distribution_tsv_exists(self):
        self.assertTrue((OUT / "pos_distribution.tsv").exists())

    def test_noun_verb_balance_tsv_exists(self):
        self.assertTrue((OUT / "noun_verb_balance.tsv").exists())

    def test_tagged_v3_sample_exists(self):
        self.assertTrue((OUT / "tagged_v3_sample.tsv").exists())

    def test_tagged_ref_sample_exists(self):
        self.assertTrue((OUT / "tagged_ref_sample.tsv").exists())

    def test_report_mentions_key_terms(self):
        text = (OUT / "pos_comparison_report.md").read_text()
        self.assertIn("noun", text)
        self.assertIn("verb", text)
        self.assertIn("drift", text.lower())
        self.assertIn("v3", text)
        self.assertIn("reference", text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
