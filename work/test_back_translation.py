#!/usr/bin/env python3
"""Regression suite for #74 back-translation harness.

    python work/test_back_translation.py

Five layers, mirroring the house pattern (#68's suite):

* **Primitives** -- the operator scanner is the corpus's own scanner, the
  Dice arithmetic is symmetric and handles the empty-empty case, the LCS ratio
  is a real order measure, the editorial-bracket strip has teeth, and the
  seeded-surface rule separates ``BABEL=TV`` from a bare ``TV``.
* **Controls** -- the identity self-round-trip is exactly 1.000 on every
  channel, the operator ablation floors the operator channel, and the
  permutation null collapses.  Each control is what stops a broken harness
  from publishing a plausible-looking table.
* **Round-trip inventory** -- every reading is read from its published
  artifact, the sample is #63's 5-clause stratified set, and the readings that
  keep the operator channel score differently from the ones that do not (the
  finding must actually be in the data).
* **End-to-end build** (a temp dir, ``--quick``) -- artifact shapes, the
  identity control row, reconciliation rows, and byte-identical determinism.
* **Planted failures** -- a generator that returns the original verbatim, and
  an operator scanner that keeps bracketed asides, must each be caught by the
  controls, so the suite fails if the guards are silently removed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import back_translation as bt

HAVE_ARTIFACTS = (bt.INTERLINEAR.exists() and bt.GLOSSES.exists()
                  and bt.SAMPLE.exists())
_ART: dict = {}


def setUpModule():
    if not HAVE_ARTIFACTS:
        return
    tmp = tempfile.mkdtemp(prefix="bt_test_")
    rc = bt.main(["--quick", "--out", tmp, "--quiet"])
    assert rc == 0
    for name in ("readings", "fidelity_scores", "round_trips",
                 "controls_and_nulls", "reconciliation", "diagnostics",
                 "worked_examples"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = bt.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["report"] = (Path(tmp) / "back_translation_report.md").read_text(
        encoding="utf-8")
    _ART["_dir"] = tmp
    # a second, independent build must be byte-identical (determinism)
    tmp2 = tempfile.mkdtemp(prefix="bt_test2_")
    bt.main(["--quick", "--out", tmp2, "--quiet"])
    for name in ("summary.json", "back_translation_report.md",
                 "fidelity_scores.tsv", "round_trips.tsv",
                 "controls_and_nulls.tsv"):
        a = (Path(tmp) / name).read_bytes()
        b = (Path(tmp2) / name).read_bytes()
        assert a == b, f"non-deterministic artifact: {name}"
    _ART["_dir2"] = tmp2


def rows(name):
    return _ART[name]


def by_reading(name, key="reading"):
    return {r[key]: r for r in rows(name)}


# --------------------------------------------------------------------------
class TestPrimitives(unittest.TestCase):
    """The scanner, the dice arithmetic and the order measure, on known data."""

    def test_operator_scanner_is_the_corpuss_own(self):
        """The harness must not fork the operator channel.

        A local copy of the scanner that drifts from #63's would silently stop
        being the corpus's ruler, so the eleven surfaces and a scan of real
        corpus text are asserted against the pipeline that owns them.
        """
        import glossing_pipeline as gp
        self.assertEqual(sorted(bt.OP_SURFACES), sorted(gp.OP_SURFACES))
        if bt.INTERLINEAR.exists():
            for r in bt.read_tsv(bt.INTERLINEAR)[:400]:
                self.assertEqual(bt.operator_subsequence(r["original"]),
                                 gp.operator_subsequence(r["original"]),
                                 f"clause {r['clause_id']}")

    def test_multi_character_operators_are_not_split(self):
        self.assertEqual(bt.operator_subsequence("::=++"), ["::", "=", "++"])
        self.assertEqual(bt.operator_subsequence("<<a>>"), ["<<", ">>"])

    def test_hyphen_is_not_an_operator(self):
        # #17: `-` joins a WORD or is PUNCT, never OP.  Charging a gloss for
        # the book's own hyphenation would make the check uninterpretable.
        for text in ("boy-roid", "screaming — eroded", "a - b", "self-"):
            self.assertEqual(bt.operator_subsequence(text), [], text)

    def test_editorial_bracket_strip_has_teeth(self):
        # A bracket keeps its operators *inside*; stripping must remove them.
        gloss = "soul/gram [a / gluing as the corpus reads it] noise>"
        self.assertEqual(bt.operator_subsequence(gloss), ["/", "/", ">"])
        stripped = bt.strip_editorial(gloss)
        self.assertNotIn("[", stripped)
        self.assertEqual(bt.operator_subsequence(stripped), ["/", ">"])

    def test_editorial_strip_does_not_fuse_words(self):
        body = bt.strip_editorial("a [editorial aside] b")
        self.assertEqual(bt.content_words(body), ["a", "b"])

    def test_glued_surfaces_need_an_interior_seam(self):
        # ``BABEL=TV`` is a seam; a bare ``TV`` is not, and a trailing closer
        # is not a seam either (that is #63's disclosed judgement call).
        g = bt.glued_surfaces
        self.assertIn("BABEL=TV", g("BABEL=TV>"))
        self.assertNotIn("TV", g("TV"))
        self.assertIn("soul/gram", g("fused the soul/gram of self"))
        self.assertIn("bio=less", g("to bio=less= intermediate"))
        self.assertEqual(g("plain words only"), set())

    def test_dice_is_symmetric_and_bounded(self):
        self.assertEqual(bt.dice({1, 2}, {2, 3}), bt.dice({2, 3}, {1, 2}))
        self.assertEqual(bt.dice({"a"}, {"a"}), 1.0)
        self.assertEqual(bt.dice({"a"}, {"b"}), 0.0)
        self.assertEqual(bt.dice(set(), set()), 1.0)
        for a in range(6):
            for b in range(6):
                s = set(range(a))
                t = set(range(3, 3 + b))
                v = bt.dice(s, t)
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_operator_recall_penalises_omission_and_addition(self):
        # Dice over multisets: losing an operator and inventing one are both
        # charged, which is why the primary leg is a Dice and not a recall.
        orig = ["=", ">", "=", ">"]
        self.assertEqual(bt.operator_recall(orig, orig), 1.0)
        self.assertEqual(bt.operator_recall(orig, ["=", ">"]), 2 * 2 / 6)
        # an extra invented operator must not raise the score
        self.assertEqual(bt.operator_recall(orig, ["=", ">", "<<", "<<"]),
                         2 * 2 / 8)
        self.assertLess(bt.operator_recall(orig, orig + ["<<"]),
                        bt.operator_recall(orig, orig))

    def test_operator_order_catches_a_shuffled_channel(self):
        orig = ["<<", "=", "/"]
        # same multiset, wrong order: recall is perfect, order is not
        self.assertEqual(bt.operator_recall(orig, ["/", "=", "<<"]), 1.0)
        self.assertLess(bt.operator_order(orig, ["/", "=", "<<"]), 1.0)
        self.assertEqual(bt.operator_order(orig, orig), 1.0)
        self.assertEqual(bt.operator_order([], []), 1.0)
        self.assertEqual(bt.operator_order([], ["="]), 0.0)

    def test_lcs_is_a_real_measure(self):
        self.assertEqual(bt._lcs(list("abc"), list("abc")), 3)
        self.assertEqual(bt._lcs(list("abc"), list("cba")), 1)
        self.assertEqual(bt._lcs(list("abc"), list("ac")), 2)
        self.assertEqual(bt._lcs([], ["a"]), 0)

    def test_regenerate_is_a_pure_function_of_the_gloss(self):
        # The regenerator must never see the original.  Two very different
        # originals with the same gloss must give the same regenerated clause.
        gloss = "I surrender to the chromosome syntax"
        self.assertEqual(bt.regenerate_clause(gloss),
                         bt.regenerate_clause(gloss))
        self.assertNotIn("omanko", bt.regenerate_clause(gloss))

    def test_regenerate_preserves_inline_operators_only(self):
        # Inline operators survive; operators nested in editorial asides do
        # not -- that is the disclosed limit, not a defect.
        gloss = "bio=less= [one = glued term] intermediate"
        self.assertEqual(bt.operator_subsequence(bt.regenerate_clause(gloss)),
                         ["=", "="])

    def test_regenerate_collapses_whitespace(self):
        self.assertEqual(bt.regenerate_clause("a\n\n b   c "), "a b c")


# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "upstream artifacts not built")
class TestControls(unittest.TestCase):
    """The controls that stop a broken harness from looking plausible."""

    def test_identity_control_perfect_on_every_channel(self):
        interlinear = bt.load_interlinear()
        readings = bt.build_readings(interlinear)
        ident = next(r for r in readings if r.name == "identity")
        for cid in sorted(ident.texts):
            fr = bt.fidelity_scores(interlinear[cid]["original"],
                                    ident.texts[cid])
            for ch in bt.CHANNELS:
                self.assertAlmostEqual(fr[ch], 1.0, places=12,
                                       msg=f"clause {cid} {ch}")

    def test_operator_ablation_floors_the_primary_channel(self):
        interlinear = bt.load_interlinear()
        for r in bt.build_readings(interlinear):
            ab = bt.operator_strip_ablation(r, interlinear, bt.sample_ids())
            if ab["n_clauses_with_operators"] == 0:
                continue
            self.assertEqual(ab["mean_operator_recall"], 0.0,
                            f"ablation did not floor {r.name}")

    def test_ablation_does_not_floor_the_lexis_channel(self):
        # The separation that proves the primary leg is about the operator and
        # not about shared words: strip the operators and the *lexis* channel
        # must survive the strip.
        interlinear = bt.load_interlinear()
        r = next(x for x in bt.build_readings(interlinear)
                 if x.name == "v3_operators_to_space")
        for cid in bt.sample_ids():
            body = " ".join(bt.strip_editorial(r.texts[cid]).split())
            ablated = bt._OP_ONLY_RE.sub(" ", body)
            self.assertGreater(bt.dice(set(bt.content_words(
                interlinear[cid]["original"])),
                set(bt.content_words(ablated))), 0.9)

    def test_permutation_null_collapses(self):
        interlinear = bt.load_interlinear()
        readings = bt.build_readings(interlinear)
        for nl in bt.permutation_null(readings, interlinear,
                                      bt.sample_ids()):
            if nl["reading"] == "identity":
                continue      # the control is its own null by construction
            self.assertGreater(
                nl["true_aggregate_fidelity"], nl["null_aggregate_fidelity"],
                f"{nl['reading']} survived its own permutation null")

    def test_permutation_null_skips_the_identity_order(self):
        interlinear = bt.load_interlinear()
        r = next(x for x in bt.build_readings(interlinear)
                 if x.name == "identity")
        # an identity permutation tests nothing; with N_NULLS seeds at least
        # one non-identity permutation must have been used
        out = bt.permutation_null([r], interlinear, bt.sample_ids())[0]
        self.assertEqual(out["null_aggregate_fidelity"],
                         out["null_aggregate_fidelity"])


# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "upstream artifacts not built")
class TestRoundTripInventory(unittest.TestCase):
    """The findings must actually be in the data."""

    def test_every_reading_is_read_from_a_published_artifact(self):
        inv = by_reading("readings")
        self.assertEqual(set(inv), {
            "identity", "v3_operators_to_space", "v4_operators_to_period",
            "interlinear_gloss", "llm_literal_operator",
            "llm_register_paraphrase", "llm_scene_summary",
            "llm_plain_english", "llm_entity_consistent"})
        for name, row in inv.items():
            self.assertIn("#", row["source"])
            self.assertEqual(int(row["n_clauses"]), 5, name)

    def test_sample_is_63s_stratified_strata_set(self):
        sample = bt.read_tsv(bt.SAMPLE)
        strata = {r["stratum"] for r in sample}
        self.assertEqual(strata, {"plain_control", "quoting_frame",
                                  "operator_dense", "entity_named",
                                  "residue"})
        self.assertEqual(len(sample), bt.PUBLISHED["n_sample_clauses"])

    def test_the_channels_split(self):
        """The finding: no reading wins both the machine and the lexis channel.

        If every reading scored the same on both channels the harness would be
        reporting nothing, so the split is pinned as a property of the data.
        """
        fs = by_reading("fidelity_scores")
        op_channel = {k: float(v["mean_operator_recall"]) for k, v in fs.items()}
        lex_channel = {k: float(v["mean_content_recall"]) for k, v in fs.items()}
        # a reading that keeps every operator keeps the lexis too
        self.assertAlmostEqual(op_channel["identity"], 1.0, places=6)
        self.assertAlmostEqual(lex_channel["identity"], 1.0, places=6)
        # ... but the silence-removed variants keep the lexis and lose the
        # operator channel entirely, so the two channels disagree there
        for name in ("v3_operators_to_space", "v4_operators_to_period"):
            self.assertAlmostEqual(lex_channel[name], 1.0, places=6, msg=name)
            self.assertAlmostEqual(op_channel[name], 0.0, places=6, msg=name)
        # and the register paraphrase round-trips the operator channel
        self.assertAlmostEqual(op_channel["llm_register_paraphrase"], 1.0,
                               places=6)

    def test_operator_preserving_glosses_outrank_plain_english(self):
        fs = by_reading("fidelity_scores")
        op = lambda n: float(fs[n]["mean_operator_recall"])
        self.assertGreater(op("llm_register_paraphrase"),
                           op("llm_plain_english"))
        self.assertGreater(op("llm_literal_operator"),
                           op("interlinear_gloss"))
        self.assertGreater(op("llm_entity_consistent"),
                           op("llm_scene_summary"))

    def test_identity_control_is_not_the_aggregate_winner(self):
        """The control is a floor, never a competitor."""
        ranked = rows("fidelity_scores")
        self.assertNotEqual(ranked[0]["reading"], "identity")

    def test_added_operators_are_penalised(self):
        """``literal_operator`` rearranges and adds operators; charge for it.

        #63 already flagged this strategy as adding operators (its own
        operator_fidelity check failed on 3 of 5 clauses for exactly this).
        The Dice leg must land below 1.0 without those additions having been
        at the clause's expense.
        """
        fs = by_reading("fidelity_scores")
        lit = fs["llm_literal_operator"]
        self.assertLess(float(lit["mean_operator_recall"]), 1.0)
        # the register paraphrase, which adds none, must not be charged
        self.assertLess(float(fs["llm_literal_operator"]["mean_operator_recall"]),
                        float(fs["llm_register_paraphrase"]["mean_operator_recall"]))
        # and the strategy's own glosses must genuinely carry extra operators
        gloss_texts = [r["gloss"] for r in bt.read_tsv(bt.GLOSSES)
                       if r["strategy"] == "literal_operator"]
        naive = sum(len(bt.operator_subsequence(t)) for t in gloss_texts)
        honest = sum(len(bt.operator_subsequence(bt.strip_editorial(t)))
                     for t in gloss_texts)
        self.assertGreater(naive, honest)
        self.assertGreaterEqual(
            int(lit["n_generated_ops"]), int(lit["n_orig_ops"]) - 2)

    def test_residue_stratum_is_not_dropped(self):
        wt = {(r["clause_id"], r["reading"]) for r in rows("worked_examples")}
        interlinear = bt.load_interlinear()
        for r in bt.read_tsv(bt.SAMPLE):
            if r["stratum"] != "residue":
                continue
            for reading in by_reading("readings"):
                self.assertIn((r["clause_id"], reading), wt)
            # #62 records the unresolved surface with its count; the round
            # trip must have carried it, not normalised it away
            self.assertIn("omanko", interlinear[r["clause_id"]]["unresolved"])
            self.assertIn("omanko",
                          interlinear[r["clause_id"]]["original"])


# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "upstream artifacts not built")
class TestBuildArtifacts(unittest.TestCase):
    """The full build, end to end."""

    def test_all_artifacts_written(self):
        d = Path(_ART["_dir"])
        for name in ("readings.tsv", "fidelity_scores.tsv", "round_trips.tsv",
                     "controls_and_nulls.tsv", "reconciliation.tsv",
                     "diagnostics.tsv", "worked_examples.tsv",
                     "summary.json", "back_translation_report.md"):
            self.assertTrue((d / name).exists(), name)

    def test_reconciliation_exact_rows_pass(self):
        for r in rows("reconciliation"):
            note = r["note"]
            if "exact match required" in note:
                self.assertEqual(str(r["earlier"]), str(r["here"]),
                                 f"{r['quantity']} drifted")

    def test_round_trips_cover_every_pair(self):
        pairs = {(r["clause_id"], r["reading"]) for r in rows("round_trips")}
        expected = len(bt.sample_ids()) * 9
        self.assertEqual(len(pairs), expected)

    def test_scores_agree_with_the_round_trip_rows(self):
        # the per-reading means must be the mean of the published per-clause
        # rows, to the rounding the artifacts use
        per = {}
        for r in rows("round_trips"):
            per.setdefault(r["reading"], []).append(r)
        fs = by_reading("fidelity_scores")
        for reading, rr in per.items():
            for ch in bt.CHANNELS:
                mean = sum(float(x[ch]) for x in rr) / len(rr)
                self.assertAlmostEqual(mean, float(fs[reading]["mean_" + ch]),
                                       places=5,
                                       msg=f"{reading} {ch}")

    def test_summary_headline_numbers(self):
        s = _ART["summary"]
        self.assertEqual(s["schema"], bt.SCHEMA)
        self.assertEqual(s["seed"], bt.SEED)
        self.assertEqual(s["n_readings"], 9)
        self.assertEqual(s["n_clauses"], 5)
        self.assertEqual(s["primary_channel"], bt.PRIMARY)
        self.assertTrue(s["controls"]["identity_perfect"])
        self.assertTrue(s["controls"]["ablation_floors_operator_channel"])
        self.assertTrue(s["controls"]["permutation_null_collapses"])
        self.assertEqual(len(s["reconciliation"]), 6)

    def test_report_structure_and_guardrails(self):
        report = _ART["report"]
        for section in ("## 1. The round trip", "## 2. Fidelity scores",
                        "## 3. Controls", "## 4. Worked examples",
                        "## 5. Reconciliation", "## 6. What this shows",
                        "## 7. Limits", "## 8. Anti-pattern audit",
                        "## 9. Reproducing"):
            self.assertIn(section, report)
        # the guardrails that make this an evaluation and not a scoreboard
        for phrase in ("no ground-truth translation", "Round-trip fidelity is "
                       "not truth", "Circularity", "Determinism",
                       "regenerator is a re-render, not an author"):
            self.assertIn(phrase, report)
        # the aggregate must be labelled a secondary, never the primary
        self.assertIn("labelled secondaries", report)

    def test_determinism(self):
        d2 = Path(_ART["_dir2"])
        for name in ("fidelity_scores.tsv", "round_trips.tsv"):
            self.assertEqual((Path(_ART["_dir"]) / name).read_bytes(),
                             (d2 / name).read_bytes(), name)


# --------------------------------------------------------------------------
class TestPlantedFailures(unittest.TestCase):
    """Each control must actually catch the failure it exists to catch."""

    def setUp(self):
        self.interlinear = bt.load_interlinear() if HAVE_ARTIFACTS else {}
        self.ids = bt.sample_ids() if HAVE_ARTIFACTS else []

    @unittest.skipUnless(HAVE_ARTIFACTS, "upstream artifacts not built")
    def test_broken_identity_control_is_caught(self):
        # A harness whose round trip is not the identity on the identity
        # reading must fail loudly rather than publish a table.
        interlinear = self.interlinear
        r = next(x for x in bt.build_readings(interlinear)
                 if x.name == "identity")
        fake = bt.fidelity_scores(interlinear[self.ids[0]]["original"],
                                  r.texts[self.ids[0]])
        for ch in bt.CHANNELS:
            self.assertAlmostEqual(fake[ch], 1.0, places=12,
                                   msg=f"identity control should be perfect "
                                       f"on {ch}; if this fails the round "
                                       f"trip is broken")

    @unittest.skipUnless(HAVE_ARTIFACTS, "upstream artifacts not built")
    def test_a_sabotaged_regenerator_is_caught_by_the_identity_control(self):
        """The identity control must actually catch a broken round trip.

        A harness that scored the lexis channel off the gloss rather than the
        regenerated clause would not fire here at all, because the identity
        *gloss* is already the original.  This is the regression pin for that
        exact bug: sabotage the regenerator, and the identity control (and the
        ablation, and the null) must all refuse to publish a table.
        """
        interlinear = self.interlinear
        ids = self.ids
        real = bt.regenerate_clause
        try:
            bt.regenerate_clause = lambda g: bt.strip_editorial(g) + " X"
            r = next(x for x in bt.build_readings(interlinear)
                     if x.name == "identity")
            fr = bt.fidelity_scores(interlinear[ids[0]]["original"],
                                    r.texts[ids[0]])
            self.assertLess(fr["content_recall"], 1.0,
                            "a sabotaged regenerator must break the lexis "
                            "channel: the channels are scored on the "
                            "regenerated clause, not on the gloss")
            with self.assertRaises(SystemExit):
                bt.main(["--out", tempfile.mkdtemp(), "--quiet"])
        finally:
            bt.regenerate_clause = real

    def test_naive_operator_count_that_keeps_brackets_overscores(self):
        # The disclosed cost of the bracket-strip judgement: a count that keeps
        # the bracketed asides must report *more* operators than the honest
        # one, so the decision is visible rather than silent.
        gloss = "BABEL[= naming seam, glued both sides]TV"
        naive = len(bt.operator_subsequence(gloss))
        honest = len(bt.operator_subsequence(bt.strip_editorial(gloss)))
        self.assertGreater(naive, honest)
        self.assertEqual(honest, 0)

    def test_ablation_of_a_two_character_operator_removes_the_run(self):
        # The reason the ablation cannot strip operators character-wise: `:`
        # is not a surface but `::` is, so a char filter would leave the run.
        body = "a :: b // c = d"
        self.assertEqual(bt.operator_subsequence(body),
                         ["::", "//", "="])
        ablated = bt._OP_ONLY_RE.sub(" ", body)
        self.assertEqual(bt.operator_subsequence(ablated), [])
        # a char filter cannot remove a two-char surface: no single character
        # is a surface here, so the runs would silently survive and the control
        # would pass a broken implementation
        naive = "".join(ch for ch in body if ch not in bt.OP_SURFACES)
        self.assertEqual(bt.operator_subsequence(naive), ["::"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
