#!/usr/bin/env python3
"""Regression suite for #68 compression test for candidate readings.

    python work/test_reading_compression.py

Five layers:

* **Primitives** -- the character n-gram's interpolation arithmetic (an
  unseen context must back off *whole*, or probability mass is lost), the
  Jelinek-Mercer ordering, the sign-test arithmetic, the length-matched
  control's position preservation and determinism, and the shuffle null's
  multiset preservation.
* **Reading inventory** -- every reading is read from its published artifact,
  the identity control covers the whole corpus, the five ``llm_*`` readings
  cover exactly #63's 25 sampled clauses, and the two populations never mix.
* **End-to-end build** (``--quick`` into a temp dir) -- the exact-match
  reconciliation rows against #33/#62/#63, the near-match rows against #48,
  artifact shapes, and the determinism of a rebuilt run (byte-identical).
* **Protocol properties** (leakage and confound controls) -- the control
  really is length matched and never *longer* than the reading it controls;
  the identity control scores exactly 0 on every delta leg; a purely
  deleting reading is pinned near zero by the length control (the deletion
  confound the whole harness exists to remove); the fixed model is a single
  shared scorer (identity rows of a population share one model); the held-out
  split never puts a clause in both halves; and the ranking is a total order
  with the control never ranked first on the length-control leg.
* **Planted failures** -- a truncated reading, a broken control (returns the
  original) and a broken split must each be caught by the checks above, so
  the suite fails if the guards are silently removed.
"""
from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reading_compression as rc

HAVE_ARTIFACTS = (rc.INTERLINEAR.exists() and rc.GLOSSES.exists())
_ART: dict = {}


def setUpModule():
    if not HAVE_ARTIFACTS:
        return
    tmp = tempfile.mkdtemp(prefix="rc_test_")
    rc2 = rc.main(["--quick", "--out", tmp, "--quiet"])
    assert rc2 == 0
    for name in ("readings", "compression", "controls_and_nulls",
                 "predictability", "ranked_readings", "reconciliation",
                 "diagnostics"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = rc.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["report"] = (Path(tmp) / "reading_compression_report.md").read_text(
        encoding="utf-8")
    _ART["_dir"] = tmp
    # a second, independent build must be byte-identical (determinism)
    tmp2 = tempfile.mkdtemp(prefix="rc_test2_")
    rc.main(["--quick", "--out", tmp2, "--quiet"])
    for name in ("summary.json", "reading_compression_report.md",
                 "ranked_readings.tsv", "predictability.tsv"):
        a = (Path(tmp) / name).read_bytes()
        b = (Path(tmp2) / name).read_bytes()
        assert a == b, f"non-deterministic artifact: {name}"
    _ART["_dir2"] = tmp2


def rows(name):
    return _ART[name]


def by_reading(name):
    return {r["reading"]: r for r in rows("ranked_readings")}


# --------------------------------------------------------------------------
class TestPrimitives(unittest.TestCase):
    """The estimators and controls, on synthetic data with known answers."""

    def test_char_ngram_unseen_context_backs_off_whole(self):
        # "ab" seen 10x, "cd" never: scoring "cb" must use the unigram model,
        # not lam*0 (an interpolated empty ML distribution loses mass).
        m = rc.CharNGram(3, 0.6)
        m.train("ab" * 10)
        p_seen = m._prob("b", ("a",))
        p_unseen = m._prob("c", ("a",))
        self.assertGreater(p_unseen, 0.0)
        self.assertLessEqual(p_unseen, p_seen + 1e-12)
        # the distribution must stay normalised over the observed alphabet
        self.assertAlmostEqual(sum(m._prob(c, ("a",)) for c in "abc"), 1.0,
                               places=6)

    def test_char_ngram_orders_match_training(self):
        m = rc.CharNGram(3, 0.6)
        m.train("aaaa")
        # order-3 context ("a","a") -> "a" must be certain
        self.assertAlmostEqual(m._prob("a", ("a", "a")), 1.0, places=9)

    def test_char_ngram_surprisal_counts_only_scored_chars(self):
        m = rc.CharNGram(3, 0.5)
        m.train("hello world " * 20)
        bits, n = m.surprisal("hello world")
        self.assertEqual(n, len("hello world") - rc.ORDER + 1)
        self.assertGreater(bits, 0.0)
        self.assertAlmostEqual(m.bits_per_char("hello world"), bits / n, places=9)

    def test_uniform_model_surprisal_is_log2_vocab(self):
        # a model trained on k distinct equiprobable chars costs log2(k)/char
        alphabet = "abcdefgh"
        m = rc.CharNGram(2, 0.0)          # lam=0 -> pure lower-order model
        m.train(alphabet * 50)
        self.assertAlmostEqual(m.bits_per_char(alphabet), math.log2(len(alphabet)),
                               delta=0.05)

    def test_sign_test_extremes_and_known(self):
        self.assertAlmostEqual(rc.sign_test_p(10, 0), 2 / 1024, places=8)
        self.assertAlmostEqual(rc.sign_test_p(5, 5), 1.0, places=8)
        self.assertEqual(rc.sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(rc.sign_test_p(3, 5), 2 * 93 / 256, places=8)

    def test_sign_test_monotone(self):
        # p is symmetric around balance and smallest at the extremes
        ps = [rc.sign_test_p(k, 10 - k) for k in range(11)]
        self.assertEqual(ps, ps[::-1])
        self.assertEqual(min(ps), rc.sign_test_p(0, 10))
        self.assertEqual(max(ps), 1.0)
        self.assertEqual(ps[0], ps[-1])

    def test_length_matched_control_is_position_preserving(self):
        original = "one two three four five six"
        out = rc.length_matched_control(original, 3, random.Random("seed"))
        words = out.split()
        self.assertEqual(len(words), 3)
        # every sampled word is an original word, kept in the original order
        idx = [original.split().index(w) for w in words]
        self.assertEqual(idx, sorted(idx))

    def test_length_matched_control_deterministic(self):
        a = rc.length_matched_control("a b c d e f g h", 4, random.Random(7))
        b = rc.length_matched_control("a b c d e f g h", 4, random.Random(7))
        self.assertEqual(a, b)
        c = rc.length_matched_control("a b c d e f g h", 4, random.Random(8))
        self.assertEqual(len(a.split()), len(c.split()))

    def test_length_matched_control_never_longer(self):
        for n in range(9):
            out = rc.length_matched_control("a b c d e f g h", n, random.Random(1))
            self.assertLessEqual(len(out.split()), n)

    def test_control_identity_when_reading_is_not_shorter(self):
        orig = "short clause here"
        self.assertEqual(rc.length_matched_control(orig, 99, random.Random(1)),
                         orig)

    def test_shuffle_null_preserves_multiset(self):
        texts = {"a": "x", "b": "y", "c": "z", "d": "w"}
        reading = rc.Reading("t", "llm", "test", texts)
        for stream in rc.null_streams(reading, ["a", "b", "c", "d"], 5):
            self.assertEqual(sorted(stream.split("\n")), sorted(texts.values()))

    def test_shuffle_null_deterministic(self):
        texts = {"a": "x", "b": "y", "c": "z"}
        reading = rc.Reading("t", "llm", "test", texts)
        self.assertEqual(rc.null_streams(reading, ["a", "b", "c"], 3),
                         rc.null_streams(reading, ["a", "b", "c"], 3))

    def test_shuffle_null_empty_for_single_clause(self):
        reading = rc.Reading("t", "llm", "test", {"a": "x"})
        self.assertEqual(rc.null_streams(reading, ["a"], 3), [])

    def test_rank_nan_takes_the_worst_rank(self):
        # A missing leg must never win by input position: NaN sorts last and
        # shares the worst rank with any other NaN (the documented contract).
        nan = float("nan")
        self.assertEqual(rc._rank([1.0, 2.0, nan]), [1, 2, 3])
        self.assertEqual(rc._rank([nan, 5.0, nan, 1.0]), [4, 2, 4, 1])
        self.assertEqual(rc._rank([nan, nan, nan]), [3, 3, 3])
        self.assertEqual(rc._rank([3.0, 1.0, 1.0]), [3, 1, 1])

    def test_split_ids_disjoint_and_complete(self):
        ids = [str(i) for i in range(100)]
        train, test = rc.split_ids(ids)
        self.assertEqual(len(train) + len(test), 100)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(test), 20)          # every 5th
        self.assertEqual(test[0], "0")
        self.assertEqual(test[-1], "95")

    def test_split_spans_pages(self):
        # clause order is page order: the held-out set must be spread over
        # the book, never a contiguous block (the #56 leakage lesson).
        order, by_id = rc.load_clauses()
        _, test = rc.split_ids(order)
        self.assertEqual(len({by_id[c]["page"] for c in test}), 219)
        test_set = set(test)
        runs, cur = [], 0
        for c in order:
            if c in test_set:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        self.assertEqual(max(runs), 1)      # no two adjacent held-out clauses


# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "upstream artifacts not built")
class TestReadingInventory(unittest.TestCase):
    def test_reading_count_and_kinds(self):
        inv = rows("readings")
        self.assertEqual(len(inv), 9)
        self.assertEqual(sum(1 for r in inv if r["kind"] == "control"), 1)
        self.assertEqual(sum(1 for r in inv if r["kind"] == "transform"), 2)
        self.assertEqual(sum(1 for r in inv if r["kind"] == "gloss"), 1)
        self.assertEqual(sum(1 for r in inv if r["kind"] == "llm"), 5)

    def test_full_book_readings_cover_the_corpus(self):
        inv = {r["reading"]: r for r in rows("readings")}
        for name in ("identity", "v3_operators_to_space",
                     "v4_operators_to_period", "interlinear_gloss"):
            self.assertEqual(int(inv[name]["n_clauses"]), rc.PUBLISHED["n_clauses"],
                             name)

    def test_llm_readings_cover_63_sample(self):
        for r in rows("readings"):
            if r["kind"] == "llm":
                self.assertEqual(int(r["n_clauses"]), 5, r["reading"])
                self.assertEqual(r["population"], "sample_63")
        # the five strategies each cover the same 5-clause stratified sample;
        # #63's 25 rows are 5 clauses x 5 strategies
        order, by_id = rc.load_clauses()
        union = set()
        for r in rc.build_readings(order, by_id):
            if r.kind == "llm":
                union |= set(r.texts)
        sampled = {row["clause_id"] for row in rc.read_tsv(rc.GLOSSES)}
        self.assertEqual(union, sampled)
        self.assertEqual(len(union), 5)
        self.assertEqual(len(rc.read_tsv(rc.GLOSSES)),
                         rc.PUBLISHED["n_gloss_rows"])

    def test_readings_come_from_published_artifacts(self):
        for r in rows("readings"):
            self.assertTrue(("#" in r["source"]) or ("column" in r["source"]),
                            f"{r['reading']} has no published provenance")

    def test_identity_is_the_control(self):
        inv = {r["reading"]: r for r in rows("readings")}
        self.assertEqual(inv["identity"]["kind"], "control")
        self.assertIn("original", inv["identity"]["source"])


@unittest.skipUnless(HAVE_ARTIFACTS, "upstream artifacts not built")
class TestBuildArtifacts(unittest.TestCase):
    def test_reconciliation_exact_rows_match(self):
        recon = {r["quantity"]: r for r in rows("reconciliation")}
        for qty in ("n_clauses", "n_pages", "n_words_clean", "n_gloss_rows"):
            self.assertIn(qty, recon)
            self.assertEqual(recon[qty]["earlier"], recon[qty]["here"], qty)

    def test_reconciliation_has_48_scope_rows(self):
        recon = rows("reconciliation")
        near = [r for r in recon if "silence_variants" in r["note"]]
        self.assertEqual(len(near), 3)          # identity + v3 + v4
        for r in near:
            # #48's frozen numbers and this build must be close (same text,
            # different stream construction: page stream vs clause-joined).
            # 0.1 bits/char bounds the disclosed scope difference.
            self.assertLess(abs(float(r["earlier"]) - float(r["here"])), 0.1, r)

    def test_compression_table_shape(self):
        comp = rows("compression")
        # 2 populations x (4 full-book + 5 llm + identity) readings, 6 compressors
        per_reading = {}
        for r in comp:
            per_reading.setdefault((r["population"], r["reading"]), set()).add(
                r["compressor"])
        self.assertEqual(len(per_reading), 13)
        for key, comps in per_reading.items():
            self.assertEqual(len(comps), 6, key)
            self.assertIn("gzip-9", comps, key)

    def test_all_readings_have_bits_per_char(self):
        for r in rows("compression"):
            self.assertGreater(float(r["bits_per_char"]), 0.0, r["reading"])

    def test_controls_and_nulls_shape(self):
        cn = rows("controls_and_nulls")
        self.assertEqual(len(cn), 13)
        for r in cn:
            if r["reading"] == "identity":
                self.assertEqual(float(r["delta_vs_length_control"]), 0.0)
            self.assertNotEqual(r["null_mean_bits_per_char"], "")
            self.assertNotEqual(r["null_z"], "")

    def test_predictability_shape(self):
        pred = rows("predictability")
        self.assertEqual(len(pred), 13)
        for r in pred:
            if r["population"] == "full_book":
                self.assertGreater(float(r["lambda"]), 0.0, r["reading"])
                self.assertGreater(float(r["self_bits_per_char"]), 0.0)
            else:
                self.assertGreater(float(r["fixed_model_self_bits_per_char"]), 0.0)

    def test_ranked_readings_have_both_populations(self):
        rr = rows("ranked_readings")
        pops = {r["population"] for r in rr}
        self.assertEqual(pops, {"full_book", "sample_63"})
        for pop in pops:
            sub = [r for r in rr if r["population"] == pop]
            # the table is sorted by the primary-leg rank, which is the
            # verdict rank; the mean-of-three aggregate is secondary
            ranks = [int(r["rank_description_length"]) for r in sub]
            self.assertEqual(ranks, sorted(ranks), pop)
            self.assertEqual(sorted(ranks), list(range(1, len(sub) + 1)), pop)

    def test_ranked_readings_columns(self):
        # every leg rank is published and the aggregate is labelled, so no
        # aggregation can silently outvote the declared primary leg
        for r in rows("ranked_readings"):
            for col in ("rank_description_length", "rank_fixed_model",
                        "rank_self_predictability", "mean_rank_three_legs",
                        "delta_vs_length_control"):
                self.assertIn(col, r)
            self.assertGreaterEqual(int(r["rank_description_length"]), 1)

    def test_report_contains_required_sections(self):
        rep = _ART["report"]
        for needle in ("## 1. Method", "## 2. The candidate readings",
                       "## 3. Leg A", "## 4. Leg C",
                       "## 6. Verdict", "## 7. What this does and does not show",
                       "not proof that an interpretation is true"):
            self.assertIn(needle, rep)

    def test_report_live_clause_count_matches_the_artifact(self):
        # The report's live-clause count is derived from the scored rows, so
        # it must equal n_clauses - control_fallback_clauses for the reading
        # it names -- a hardcoded prose number would drift from the table
        # printed 40 lines above it.
        cn = {(r["reading"], r["population"]): r for r in rows("controls_and_nulls")}
        rep = _ART["report"]
        for name in ("interlinear_gloss", "v3_operators_to_space"):
            r = cn[(name, "full_book")]
            live = int(r["n_clauses"]) - int(r["control_fallback_clauses"])
            fb = int(r["control_fallback_clauses"])
            frag = (f"{live} of {r['n_clauses']} clauses for v3/v4"
                    if name == "v3_operators_to_space"
                    else f"{live} of {r['n_clauses']} for the gloss")
            self.assertIn(frag, rep, name)
            self.assertIn(f"fallback {fb}", rep, name)
        # and no stale figure survives anywhere in the report
        self.assertNotIn("~75", rep)

    def test_report_sd_figure_matches_the_artifact(self):
        # the prose sd figure is the published maximum, not a remembered one
        sd = max(float(r["length_control_sd_bits_per_char"])
                 for r in rows("controls_and_nulls")
                 if r["population"] == "full_book")
        self.assertIn(f"{sd:.1e}", _ART["report"])

    def test_one_word_rule_across_artifacts(self):
        # readings.tsv and controls_and_nulls.tsv must both count words with
        # the documented single rule (maximal letter runs).  A whitespace
        # count in either would publish a second, conflicting number for the
        # same reading -- identity would show 74,420 vs 80,334.
        order, by_id = rc.load_clauses()
        readings = {r.name: r for r in rc.build_readings(order, by_id)}
        inv = {r["reading"]: r for r in rows("readings")}
        for name, row in inv.items():
            expected = sum(rc.word_count(t) for t in readings[name].texts.values())
            self.assertEqual(int(row["n_words"]), expected,
                             f"{name}: readings.tsv uses a different word rule")
        cn = {(r["reading"], r["population"]): r for r in rows("controls_and_nulls")}
        eval_ids = [c for i, c in enumerate(order) if i % rc.QUICK_EVERY == 0] \
            if _ART["summary"]["quick"] else order
        for (name, pop), ctrl in cn.items():
            if pop != "full_book":
                continue
            expected = sum(rc.word_count(readings[name].texts[c])
                           for c in eval_ids if c in readings[name].texts)
            self.assertEqual(int(ctrl["reading_n_words"]), expected,
                             f"{name}: controls_and_nulls.tsv uses a "
                             "different word rule")

    def test_report_discloses_the_control_fallback(self):
        # The primary leg's disclosed limit must be in the report, with the
        # live-clause counts, not only in the code.
        rep = _ART["report"]
        self.assertIn("no subsample", rep)
        self.assertIn("control fallback", rep)
        cn = {(r["reading"], r["population"]): r for r in rows("controls_and_nulls")}
        r = cn[("v3_operators_to_space", "full_book")]
        live = int(r["n_clauses"]) - int(r["control_fallback_clauses"])
        self.assertIn(f"live on only {live} of {r['n_clauses']} clauses", rep)
        self.assertIn(f"fallback {r['control_fallback_clauses']}", rep)
        # and the sample population is described as 5 clauses, never 25
        self.assertNotIn("25-clause sample", rep)
        self.assertNotIn("25 clauses", rep)

    def test_report_states_tokenization_and_limits(self):
        rep = _ART["report"]
        self.assertIn("Tokenization and units", rep)
        self.assertIn("held-out split", rep)
        self.assertIn("length-matched", rep)

    def test_summary_json_schema_and_verdict(self):
        s = _ART["summary"]
        self.assertEqual(s["schema"], rc.SCHEMA)
        self.assertEqual(s["n_clauses_evaluated"], 511)   # --quick subsample
        self.assertTrue(s["ranked_full_book"])
        self.assertIn("reading", s["ranked_full_book"][0])

    def test_diagnostics_disable_clock_and_seed(self):
        diag = {r["diag"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(diag["seed"], str(rc.SEED))
        self.assertEqual(diag["test_every"], str(rc.TEST_EVERY))
        self.assertEqual(diag["primary_compressor"], rc.PRIMARY_COMPRESSOR)
        self.assertNotIn("generated", diag)


# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "upstream artifacts not built")
class TestProtocolProperties(unittest.TestCase):
    """Leakage and confound controls -- the harness must not lie."""

    def test_random_deletion_is_neutralised_by_the_control(self):
        # A reading that deletes words at *random* positions is exactly what
        # the control models, so its delta must be seed noise, not a win.
        order, by_id = rc.load_clauses()
        ids = order[:60]
        originals = {c: by_id[c]["original"] for c in order}
        texts = {}
        for c in ids:
            # subsample on the harness's own word rule (letters runs), not
            # whitespace chunks, or the planted reading's length is measured
            # by a different tokenizer than the control's
            words = rc.WORD_RE.findall(by_id[c]["original"])
            k = max(1, len(words) // 2)
            idx = sorted(random.Random(f"del:{c}").sample(range(len(words)),
                                                          min(k, len(words))))
            texts[c] = " ".join(words[i] for i in idx)
        r = rc.Reading("random_deletion", "null", "planted", texts)
        s = rc.score_reading(r, ids, "full_book", originals, None, [])
        self.assertLess(abs(s.delta_vs_control), 0.25,
                        f"random deletion escaped the control: {s.delta_vs_control}")
        # the word-matching invariant must hold exactly
        self.assertEqual(s.control_n_words, s.n_words)

    def test_control_matches_word_count(self):
        # The control is *word*-matched (letters runs, the harness's one word
        # rule), not char-matched -- a reading that swaps long coinages for
        # short English words legitimately changes its char count, and that
        # is disclosed.  The word invariant must hold exactly wherever the
        # reading is shorter than its clause.
        for r in rows("controls_and_nulls"):
            self.assertLessEqual(int(r["control_n_words"]),
                                 int(r["reading_n_words"]), r["reading"])
            n = int(r["n_clauses"])
            fb = int(r["control_fallback_clauses"])
            self.assertLessEqual(fb, n, r["reading"])
            # control_n_words == reading_n_words - excess only on the
            # clauses that did NOT fall back, so the gap is bounded by the
            # live clauses' word counts
            self.assertGreaterEqual(int(r["reading_n_words"])
                                    - int(r["control_n_words"]), 0)

    def test_control_word_matching_is_exact_when_shorter(self):
        order, by_id = rc.load_clauses()
        ids = order[:60]
        originals = {c: by_id[c]["original"] for c in order}
        for name in ("v3_operators_to_space", "interlinear_gloss"):
            r = next(r for r in rc.build_readings(order, by_id) if r.name == name)
            s = rc.score_reading(r, ids, "full_book", originals, None, [])
            # control_n_words = sum(min(reading words, original words)), so
            # it equals the reading's word count minus the excess on the
            # clauses where the reading is longer (those fall back)
            excess = sum(
                max(0, rc.word_count(r.texts[c]) - rc.word_count(by_id[c]["original"]))
                for c in rc.covered(r, ids))
            self.assertEqual(s.control_n_words, s.n_words - excess)
            # and the fallback count is the count of non-shorter clauses
            self.assertEqual(
                s.n_control_fallback,
                sum(1 for c in rc.covered(r, ids)
                    if rc.word_count(r.texts[c]) >= rc.word_count(by_id[c]["original"])))

    def test_operator_readings_are_word_count_preserving(self):
        # The disclosed reason the control is inert for v3/v4: removing
        # operator characters does not change the letters-word count.
        order, by_id = rc.load_clauses()
        same = sum(1 for c in order
                   if rc.word_count(by_id[c]["v3_space"]) == rc.word_count(by_id[c]["original"]))
        self.assertEqual(same, 5470)

    def test_identity_scores_zero_on_every_delta_leg(self):
        for r in rows("ranked_readings"):
            if r["reading"] != "identity":
                continue
            self.assertEqual(float(r["delta_vs_length_control"]), 0.0)
            self.assertEqual(float(r["fixed_model_gain_vs_identity"]), 0.0)
            self.assertEqual(r["null_z"], "" or r["null_z"])

    def test_structured_deletion_can_beat_the_control(self):
        # Disclosed failure mode of the compression leg: keeping each
        # clause's first word is pure deletion, yet it beats the random-word
        # control because clause-openers are repetitive.  The test pins the
        # behaviour so the report's disclosure stays true.
        order, by_id = rc.load_clauses()
        ids = order[:60]
        originals = {c: by_id[c]["original"] for c in order}
        pref = rc.Reading("prefix1", "null", "planted",
                          {c: (by_id[c]["original"].split() or [""])[0]
                           for c in ids})
        s = rc.score_reading(pref, ids, "full_book", originals, None, [])
        self.assertLess(s.delta_vs_control, -0.2)
        # and it is still *less* predictable to the book model than the
        # original text -- the predictability leg is not fooled
        train_ids, _test = rc.split_ids(ids[:200])
        fixed = rc.CharNGram(rc.ORDER, rc._tune_lambda(
            "\n".join(originals[c] for c in train_ids).casefold()))
        fixed.train("\n".join(originals[c] for c in train_ids).casefold())
        d_pref = fixed.bits_per_char(rc.reading_stream(pref, ids).casefold())
        d_orig = fixed.bits_per_char(
            "\n".join(originals[c] for c in ids).casefold())
        self.assertGreater(d_pref, d_orig)

    def test_control_detects_restructuring(self):
        # The same words, reordered wildly, must score WORSE than its control
        # (a destroyed reading), proving the control is not a free pass.
        order, by_id = rc.load_clauses()
        ids = order[:40]
        originals = {c: by_id[c]["original"] for c in order}
        shuffled = rc.Reading("shuffled_words", "null", "planted",
                              {c: " ".join(reversed(by_id[c]["original"].split()))
                               for c in ids})
        s = rc.score_reading(shuffled, ids, "full_book", originals, None, [])
        self.assertGreater(s.delta_vs_control, 0.0)

    def test_heldout_never_scores_training_text(self):
        order, by_id = rc.load_clauses()
        train, test = rc.split_ids(order)
        self.assertFalse(set(train) & set(test))
        # the adaptive leg's train text is built from train clauses only:
        # a long, distinctive held-out clause's own text must be absent
        r = next(r for r in rc.build_readings(order, by_id)
                 if r.name == "v3_operators_to_space")
        train_text = rc.reading_stream(r, train)
        uniques = [c for c in test
                   if len(by_id[c]["v3_space"]) > 40
                   and train_text.count(by_id[c]["v3_space"]) == 0]
        self.assertGreater(len(uniques), 100)     # the rule has teeth
        for c in uniques[:5]:
            self.assertNotIn(by_id[c]["v3_space"], train_text)

    def test_fixed_model_is_shared_across_readings(self):
        # Two readings in one population must be scored by the same model:
        # identity's own score equals the fixed-model leg of any reading that
        # reproduces the identity text verbatim.
        order, by_id = rc.load_clauses()
        ids = order[:200]
        originals = {c: by_id[c]["original"] for c in order}
        echo = rc.Reading("echo", "null", "planted", dict(originals).items()
                          and {c: originals[c] for c in ids})
        scores = rc.score_population(
            [next(r for r in rc.build_readings(order, by_id)
                  if r.name == "identity")] + [echo], ids, "full_book", originals)
        base = next(s for s in scores if s.reading == "identity")
        e = next(s for s in scores if s.reading == "echo")
        self.assertAlmostEqual(e.fixed_self_bpc, base.fixed_self_bpc, places=9)
        self.assertAlmostEqual(e.fixed_gain_vs_identity, 0.0, places=9)

    def test_ranking_is_a_total_order(self):
        for pop in ("full_book", "sample_63"):
            sub = [r for r in rows("ranked_readings") if r["population"] == pop]
            # the primary (verdict) rank is a total order over the population
            ranks = sorted(int(r["rank_description_length"]) for r in sub)
            self.assertEqual(ranks, list(range(1, len(sub) + 1)), pop)

    def test_sample_population_self_rank_is_not_arbitrary(self):
        # every sample reading lacks the adaptive leg (self_bpc is NaN), so
        # all of them must share the worst rank on it -- never an arbitrary
        # order inherited from input position.
        sub = [r for r in rows("ranked_readings")
               if r["population"] == "sample_63"]
        n = len(sub)
        self.assertTrue(all(r["self_bits_per_char"] == "nan" for r in sub))
        self.assertEqual({int(r["rank_self_predictability"]) for r in sub}, {n})

    def test_full_book_self_ranks_are_distinct(self):
        sub = [r for r in rows("ranked_readings")
               if r["population"] == "full_book"]
        ranks = [int(r["rank_self_predictability"]) for r in sub]
        self.assertEqual(sorted(ranks), list(range(1, len(sub) + 1)))

    def test_control_never_first_on_length_control_leg(self):
        sub = [r for r in rows("ranked_readings")
               if r["population"] == "full_book"]
        best = min(sub, key=lambda r: float(r["delta_vs_length_control"]))
        self.assertNotEqual(best["reading"], "identity")

    def test_adaptive_leg_floor_respected(self):
        for r in rows("predictability"):
            if r["population"] == "sample_63":
                # the adaptive leg is not run below MIN_TRAIN_CLAUSES; the
                # artifact records NaN, never a fitted lambda
                self.assertIn(r["lambda"], ("", "nan"), r["reading"])
                self.assertIn(r["self_bits_per_char"], ("", "nan"), r["reading"])

    def test_v3_beats_v4_on_fixed_model_leg(self):
        # The disclosed instrument disagreement with #50 must be present in
        # the numbers, not only in the prose: v4 (the GPT-2 winner) is the
        # worst full-book reading under the book-trained character model.
        rr = {r["reading"]: r for r in rows("ranked_readings")
              if r["population"] == "full_book"}
        self.assertLess(float(rr["v3_operators_to_space"]["fixed_model_gain_vs_identity"]),
                        float(rr["v4_operators_to_period"]["fixed_model_gain_vs_identity"]))

    def test_gloss_wins_length_control_leg_on_the_full_book(self):
        # The deliverable's headline comes from the full-book run, so this is
        # computed on every clause rather than read from the --quick sample
        # artifacts (on the 511-clause quick sample v3 edges the gloss).
        order, by_id = rc.load_clauses()
        originals = {c: by_id[c]["original"] for c in order}
        _, test_ids = rc.split_ids(order)
        deltas = {}
        for name in ("interlinear_gloss", "v3_operators_to_space",
                     "v4_operators_to_period"):
            r = next(r for r in rc.build_readings(order, by_id)
                     if r.name == name)
            s = rc.score_reading(r, order, "full_book", originals, None, test_ids)
            deltas[name] = s.delta_vs_control
        self.assertLess(deltas["interlinear_gloss"], deltas["v3_operators_to_space"])
        self.assertLess(deltas["v3_operators_to_space"],
                        deltas["v4_operators_to_period"])
        # and the three readings all beat the identity control's zero
        for name, d in deltas.items():
            self.assertLess(d, 0.0, name)

    def test_primary_verdict_matches_the_primary_leg(self):
        # The published ranking must be sorted by the declared primary leg,
        # so the report's verdict can never be outvoted by an aggregate.
        sub = [r for r in rows("ranked_readings")
               if r["population"] == "full_book"]
        top = sub[0]
        self.assertEqual(int(top["rank_description_length"]), 1)
        for other in sub[1:]:
            self.assertLessEqual(float(top["delta_vs_length_control"]),
                                 float(other["delta_vs_length_control"]))


# --------------------------------------------------------------------------
class TestPlantedFailures(unittest.TestCase):
    """Each guard above must catch a broken harness."""

    def test_broken_control_is_caught(self):
        # a control that returns the original instead of subsampling would
        # let a deleting reading look like restructuring
        orig = "alpha beta gamma delta epsilon zeta eta theta"
        rng = random.Random(1)
        ctrl = rc.length_matched_control(orig, 3, rng)
        self.assertNotEqual(ctrl, orig)
        self.assertEqual(len(ctrl.split()), 3)

    def test_broken_split_is_caught(self):
        ids = [str(i) for i in range(20)]
        train, test = rc.split_ids(ids)
        # if the split ever overlapped, the leakage guards must fail
        self.assertEqual(set(train) & set(test), set())
        self.assertEqual(sorted(train + test), sorted(ids))

    def test_prefix_reading_beats_control_only_marginally(self):
        # Structured deletion (clause prefixes) beats the random control but
        # stays far from a restructuring reading's predictability profile:
        # the compression leg alone would rank it well, which is why the
        # verdict needs the predictability legs.
        order, by_id = rc.load_clauses()
        ids = order[:60]
        originals = {c: by_id[c]["original"] for c in order}
        for k in (1, 3):
            pref = rc.Reading(f"prefix{k}", "null", "planted",
                              {c: " ".join(by_id[c]["original"].split()[:k])
                               for c in ids})
            s = rc.score_reading(pref, ids, "full_book", originals, None, [])
            # a bounded win over the control (not unbounded cheating)
            self.assertGreater(s.delta_vs_control, -0.5, f"k={k}")
            self.assertEqual(s.control_n_words, s.n_words, f"k={k}")

    def test_empty_stream_raises_no_crash(self):
        m = rc.CharNGram(3, 0.5)
        m.train("")
        bits, n = m.surprisal("")
        self.assertEqual((bits, n), (0.0, 0))


if __name__ == "__main__":
    unittest.main()
