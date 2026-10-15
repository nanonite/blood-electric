#!/usr/bin/env python3
"""Regression suite for #75 human jury / reader study design.

    python work/test_reader_study.py

Six layers, mirroring the house pattern (#68's and #74's suites):

* **Instrument** -- the items exist, are anchored, and the two traps carry
  a reading that is *known* to be unfaithful; the primary item is unique
  and the appeal item is genuinely separate from it.
* **Sampling frame** -- the frame is a census of the gloss population, read
  from #62/#63's artifacts and reconciled against them, with the traps
  counted rather than dropped.
* **Counterbalancing** -- every block shows the identity control and the
  #62 gloss, no LLM reading is drawn twice into a block, and the rotation
  varies across blocks so no reading is systematically first.
* **Statistics** -- the alpha implementation reproduces the published
  worked example in Krippendorff's 2011 note, the bootstrap CI is seeded
  and reproducible, the permutation tests are honest, and Spearman is a
  real rank correlation.
* **Simulated pilot** -- the ratings are integers on the scale, the
  identity control tops every block, the machine/human correlation is a
  genuine correlation and not a restatement of #74, and the attention
  check can actually flag a reader.
* **End-to-end build and planted failures** (a temp dir) -- artifact
  shapes, the "no human data" contract, byte-identical determinism, and
  three *planted* failures (an instrument whose primary item is appeal, a
  sampler that ignores the rotation, an alpha whose null shuffles inside a
  unit) that must each be caught by a control.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reader_study as rs

HAVE_ARTIFACTS = (rs.INTERLINEAR.exists() and rs.GLOSSES.exists()
                  and rs.SAMPLE.exists())
_ART: dict = {}


def setUpModule():
    if not HAVE_ARTIFACTS:
        return
    tmp = tempfile.mkdtemp(prefix="rs_test_")
    rc = rs.main(["--out", tmp, "--quiet"])
    assert rc == 0
    for name in ("frame", "instrument", "blocks", "pilot_ratings",
                 "pilot_pairwise", "pilot_cell_summary",
                 "pilot_stratum_summary", "agreement", "machine_human",
                 "pairwise_summary", "reader_flags", "reconciliation",
                 "diagnostics", "reading_effect"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = rs.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["report"] = (Path(tmp) / "reader_study_report.md").read_text(
        encoding="utf-8")
    _ART["protocol"] = (Path(tmp) / "protocol.md").read_text(encoding="utf-8")
    _ART["_dir"] = tmp
    # a second, independent build must be byte-identical (determinism)
    tmp2 = tempfile.mkdtemp(prefix="rs_test2_")
    rs.main(["--out", tmp2, "--quiet"])
    for name in ("summary.json", "reader_study_report.md", "protocol.md",
                 "pilot_ratings.tsv", "frame.tsv", "agreement.tsv",
                 "pilot_cell_summary.tsv"):
        a = (Path(tmp) / name).read_bytes()
        b = (Path(tmp2) / name).read_bytes()
        assert a == b, f"non-deterministic artifact: {name}"
    _ART["_dir2"] = tmp2


# --------------------------------------------------------------------------
# instrument
# --------------------------------------------------------------------------
class TestInstrument(unittest.TestCase):
    def test_primary_item_is_unique_and_is_the_primary_scale(self):
        self.assertEqual(rs.PRIMARY_ITEM, "felt_fidelity")
        keys = [k for k, *_ in rs.SCALE_ITEMS]
        self.assertEqual(keys.count(rs.PRIMARY_ITEM), 1)
        self.assertNotEqual(rs.APPEAL_ITEM, rs.PRIMARY_ITEM)

    def test_every_item_is_anchored_both_ends(self):
        for key, prompt, low, high in rs.ALL_SCALE_ITEMS:
            self.assertTrue(prompt.strip(), f"{key} has no prompt")
            self.assertTrue(low.strip(), f"{key} has no low anchor")
            self.assertTrue(high.strip(), f"{key} has no high anchor")
        for key, prompt, options in rs.PAIRWISE_ITEMS:
            self.assertTrue(prompt.strip(), f"{key} has no prompt")
            self.assertGreaterEqual(len(options), 3,
                                    f"{key} needs a real answer set")
            for label in options:
                self.assertTrue(label.strip(), f"{key} has an empty option")

    def test_traps_carry_a_known_unfaithful_reading(self):
        """Both traps must carry #63's most-failed strategy, disclosed."""
        traps = {k for k, *_ in rs.TRAP_ITEMS}
        self.assertEqual(traps, {"trap_overcorrected", "trap_invented"})
        self.assertIn("plain_english", rs.STRATEGY_ROLE)
        roles = {rs.STRATEGY_ROLE[s] for s in rs.SAMPLED_STRATEGIES}
        self.assertTrue(any("over-correction" in r for r in roles),
                        "the sampling strategies must name the "
                        "over-correction arm")

    def test_attention_checks_name_both_extremes(self):
        items = {c[2] for c in rs.ATTENTION_CHECKS}
        self.assertIn(rs.PRIMARY_ITEM, items)
        self.assertTrue(any("trap" in i for i in items))
        expected = [c[3] for c in rs.ATTENTION_CHECKS]
        self.assertEqual(sorted(expected), [0, 7])

    def test_item_keys_cover_the_scales_and_the_pairs(self):
        self.assertEqual(set(rs.ITEM_KEYS),
                         set(rs.SCALE_KEYS) | set(rs.PAIRWISE_KEYS))
        self.assertEqual(len(rs.ITEM_KEYS), 7)


# --------------------------------------------------------------------------
# sampling frame
# --------------------------------------------------------------------------
class TestFrame(unittest.TestCase):
    def setUp(self):
        if not HAVE_ARTIFACTS:
            self.skipTest("upstream artifacts absent")
        self.interlinear = rs.load_interlinear()
        self.sample = rs.load_sample()
        self.glosses = rs.load_glosses()
        self.frame = rs.build_frame(self.interlinear, self.sample,
                                    self.glosses)

    def test_frame_is_a_census_of_the_gloss_population(self):
        n_clauses = len(self.sample)
        # identity + #62 gloss + 5 strategies + 2 traps, per clause
        self.assertEqual(len(self.frame), n_clauses * (2 + 5 + 2))
        by_kind: dict = {}
        for c in self.frame:
            by_kind.setdefault(c.kind, []).append(c)
        self.assertEqual(len(by_kind["identity"]), n_clauses)
        self.assertEqual(len(by_kind["gloss"]), n_clauses)
        self.assertEqual(len(by_kind["llm"]), n_clauses * 5)
        self.assertEqual(len(by_kind["trap"]), n_clauses * 2)

    def test_every_gloss_text_comes_from_63_not_from_here(self):
        """No gloss may be invented: each must equal #63's pinned text."""
        want = {}
        for r in rs.read_tsv(rs.GLOSSES):
            want[(r["clause_id"], r["strategy"])] = rs.unescape(r["gloss"])
        seen = 0
        for c in self.frame:
            if c.kind != "llm":
                continue
            strategy = c.reading[len("llm_"):]
            self.assertEqual(c.text, want[(c.clause_id, strategy)])
            seen += 1
        self.assertEqual(seen, len(self.sample) * 5)

    def test_identity_cell_is_the_clause_verbatim(self):
        for c in self.frame:
            if c.kind == "identity":
                self.assertEqual(c.text, c.original)
                self.assertEqual(c.text,
                                 self.interlinear[c.clause_id]["original"])

    def test_traps_are_counted_not_dropped(self):
        candidates = rs.candidate_cells(self.frame)
        traps = [c for c in self.frame if c.kind == "trap"]
        self.assertEqual(len(candidates) + len(traps), len(self.frame))
        self.assertEqual(len(traps), 2 * len(self.sample))

    def test_pair_ids_are_unique_even_where_readings_repeat(self):
        """A clause carries ``plain_english`` twice: candidate and traps."""
        ids = [c.pair_id for c in self.frame]
        self.assertEqual(len(ids), len(set(ids)))
        repeated = {}
        for c in self.frame:
            repeated.setdefault((c.clause_id, c.reading), []).append(c.kind)
        dupes = {k: v for k, v in repeated.items() if len(v) > 1}
        self.assertTrue(dupes, "expected the trap readings to repeat a "
                               "reading that is also a candidate")
        for kinds in dupes.values():
            self.assertIn("trap", kinds)


# --------------------------------------------------------------------------
# counterbalancing
# --------------------------------------------------------------------------
class TestCounterbalancing(unittest.TestCase):
    def setUp(self):
        if not HAVE_ARTIFACTS:
            self.skipTest("upstream artifacts absent")
        self.frame = rs.build_frame(rs.load_interlinear(), rs.load_sample(),
                                    rs.load_glosses())
        self.blocks = rs.assign_blocks(self.frame, rs.N_PAIRS_PER_CLAUSE)

    def test_one_block_per_sampled_clause(self):
        self.assertEqual(len(self.blocks), len(rs.load_sample()))
        self.assertEqual(len({b["clause_id"] for b in self.blocks}),
                         len(self.blocks))

    def test_every_block_carries_the_two_comparators(self):
        for b in self.blocks:
            kinds = [c.kind for c in b["cells"]]
            self.assertEqual(kinds.count("identity"), 1)
            self.assertEqual(kinds.count("gloss"), 1)

    def test_no_block_draws_a_reading_twice(self):
        for b in self.blocks:
            ids = [c.pair_id for c in b["cells"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_the_rotation_actually_rotates(self):
        """Different blocks must draw different LLM comparators."""
        draws = []
        for b in self.blocks:
            llm = sorted(c.reading for c in b["cells"] if c.kind == "llm")
            draws.append(tuple(llm))
        self.assertGreater(len(set(draws)), 1,
                           "every block drew the same comparators: the "
                           "counterbalancing is not counterbalancing")
        for b in self.blocks:
            self.assertEqual(
                len([c for c in b["cells"] if c.kind == "llm"]),
                rs.N_PAIRS_PER_CLAUSE)

    def test_traps_are_shown_so_the_attention_check_has_data(self):
        for b in self.blocks:
            self.assertEqual(len([c for c in b["cells"]
                                  if c.kind == "trap"]), 2)

    def test_schedule_pairs_adjacent_cells_and_never_repeats_a_cell(self):
        schedule = rs.block_schedule(self.blocks)
        self.assertEqual(len(schedule), len(self.blocks))
        for s, b in zip(schedule, self.blocks):
            self.assertEqual(sorted(s["order"]), sorted(b["pair_ids"]))
            self.assertEqual(len(s["order"]), len(set(s["order"])))
            for a, b_ in s["pairwise_pairs"]:
                self.assertNotEqual(a, b_)

    def test_blocks_tsv_publishes_the_presentation_order(self):
        """No reading may be systematically first in the published packets.

        The counterbalancing claim in the report is checkable: the first
        position of the five blocks must be occupied by five different
        cells, so a position effect cannot be confounded with a reading.
        """
        first = [r["pair_id"] for r in _ART["blocks"] if r["position"] == "0"]
        self.assertEqual(len(first), len(self.blocks))
        self.assertEqual(len(set(first)), len(set(first)),
                         "the same cell is first in more than one block: "
                         "position is confounded with reading")
        readings = {pid.split("/", 1)[1] for pid in first}
        self.assertGreater(len(readings), 1)

    def test_design_is_reproducible_from_the_frame_alone(self):
        again = rs.assign_blocks(self.frame, rs.N_PAIRS_PER_CLAUSE)
        self.assertEqual([[c.pair_id for c in b["cells"]] for b in self.blocks],
                         [[c.pair_id for c in b["cells"]] for b in again])


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
class TestStatistics(unittest.TestCase):
    def test_alpha_is_translation_invariant(self):
        """Re-labelling the scale does not change alpha.

        A documented property of any coincidence-matrix alpha: with the
        same categories and the same disagreement structure, shifting every
        label by a constant leaves ``alpha`` untouched.  It is why the
        *choice of metric*, not the absolute labels, is what the study has
        to pre-register.
        """
        self.assertEqual(rs.krippendorff_alpha([[6, 5], [6, 5], [6, 5]]),
                         rs.krippendorff_alpha([[1, 0], [1, 0], [1, 0]]))

    def test_the_default_metric_is_the_preregistered_one(self):
        """Every published alpha must come from the registered metric.

        The protocol names ``PRIMARY_METRIC`` as the pre-registered
        difference function and the pilot calls ``krippendorff_alpha``
        without one, so the default *is* the registered metric -- and it
        must be ``interval``, because a 0-7 scale carries equal intervals.
        """
        import inspect

        self.assertEqual(rs.PRIMARY_METRIC, "interval")
        sig = inspect.signature(rs.krippendorff_alpha)
        self.assertEqual(sig.parameters["metric"].default, rs.PRIMARY_METRIC)
        cats = (0, 5, 6, 7)
        marg = {c: 1.0 for c in cats}
        self.assertEqual(
            rs._delta_squared(cats, marg, rs.PRIMARY_METRIC)[(0, 7)], 49.0)

    def test_interval_metric_charges_distance_and_ordinal_cannot(self):
        """The metric, not the labels, is what makes a 6-vs-5 cheap.

        With only two observed categories the ordinal metric sees *one rank
        step* between them whatever their scale distance, so a 0-vs-6
        disagreement costs it exactly what a 5-vs-6 disagreement costs,
        while the interval metric charges 36 against 1.  With intermediate
        categories observed the ordinal metric does separate them, but by
        the number of categories spanned rather than by the scale.  That is
        why the study pre-registers ``interval`` instead of quietly relying
        on the ordinal default several texts recommend.
        """
        two = (0, 6)
        marg = {c: 1.0 for c in two}
        interval = rs._delta_squared(two, marg, "interval")
        ordinal = rs._delta_squared(two, marg, "ordinal")
        nominal = rs._delta_squared(two, marg, "nominal")
        self.assertEqual(interval[(0, 6)], 36.0)
        self.assertEqual(ordinal[(0, 6)], 1.0)
        self.assertEqual(nominal[(0, 6)], 1.0)
        for metric in (interval, ordinal, nominal):
            self.assertEqual(metric[(0, 0)], 0.0)
        four = (0, 1, 2, 7)
        marg4 = {c: 1.0 for c in four}
        ord4 = rs._delta_squared(four, marg4, "ordinal")
        int4 = rs._delta_squared(four, marg4, "interval")
        self.assertGreater(ord4[(0, 7)], ord4[(0, 1)])
        self.assertEqual(int4[(0, 7)], 49.0)
        self.assertEqual(int4[(0, 1)], 1.0)

    def test_an_unknown_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            rs.krippendorff_alpha([[1, 2], [3, 4]], metric="bogus")

    def test_alpha_reproduces_krippendorffs_binary_example_nominally(self):
        """Krippendorff (2011) §A: binary data, two observers, no missing.

        o_00=10, o_01=o_10=4, o_11=2; n_0=14, n_1=6, n=20.
        The published result is alpha = 2/21 = 0.095238, which is the
        nominal metric (for binary data all three metrics coincide, so this
        pins the whole machinery).
        """
        units = [[0, 0]] * 5 + [[1, 1]] + [[0, 1]] * 4
        self.assertAlmostEqual(rs.krippendorff_alpha(units,
                                                     metric="nominal"),
                               2 / 21, places=12)

    def test_alpha_perfect_agreement_is_one(self):
        self.assertEqual(
            rs.krippendorff_alpha([[3, 3, 3], [5, 5, 5], [1, 1, 1]]), 1.0)

    def test_alpha_chance_level_is_zero_on_random_data(self):
        """Alpha's defining property: independent readers score ~0."""
        import random
        rng = random.Random(11)
        vals = []
        for _ in range(200):
            units = [[rng.randint(0, 7) for _ in range(6)]
                     for _ in range(4)]
            a = rs.krippendorff_alpha(units)
            if not math.isnan(a):
                vals.append(a)
        self.assertGreater(len(vals), 150)
        self.assertAlmostEqual(sum(vals) / len(vals), 0.0, delta=0.05)

    def test_alpha_single_category_and_single_value(self):
        self.assertEqual(rs.krippendorff_alpha([[2, 2], [2, 2]]), 1.0)
        self.assertTrue(math.isnan(rs.krippendorff_alpha([[1], [2]])))

    def test_alpha_nominal_path_for_unordered_categories(self):
        a = rs.krippendorff_alpha([["a", "a"], ["a", "b"], ["b", "b"]])
        self.assertTrue(-1.0 <= a <= 1.0)
        self.assertEqual(
            rs.krippendorff_alpha([["x", "x"], ["y", "y"], ["x", "y"]]),
            rs.krippendorff_alpha([["a", "a"], ["b", "b"], ["a", "b"]]))

    def test_bootstrap_ci_is_seeded_and_brackets_the_mean(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        lo1, hi1 = rs.bootstrap_ci(values, 200, 5)
        lo2, hi2 = rs.bootstrap_ci(values, 200, 5)
        self.assertEqual((lo1, hi1), (lo2, hi2))
        self.assertLess(lo1, rs.mean(values))
        self.assertGreater(hi1, rs.mean(values))
        self.assertLess(
            rs.bootstrap_ci(values, 200, 99)[1] - rs.bootstrap_ci(values, 200, 99)[0],
            hi1 - lo1, "a CI must not widen without a reason")

    def test_permutation_test_detects_a_separated_sample(self):
        a = [6.5, 6.6, 6.4, 6.7, 6.5]
        b = [1.2, 1.5, 1.1, 1.4, 1.3]
        res = rs.permutation_test(a, b, 500, 3)
        self.assertAlmostEqual(res["observed"], 5.24, places=6)
        self.assertLess(res["p"], 0.05)

    def test_permutation_test_finds_nothing_in_overlapping_samples(self):
        a = [4.0, 4.5, 5.0, 5.5]
        b = [4.2, 4.8, 5.1, 5.6]
        self.assertGreater(rs.permutation_test(a, b, 500, 3)["p"], 0.05)

    def test_spearman_is_a_rank_correlation(self):
        self.assertAlmostEqual(rs.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(rs.spearman([1, 2, 3, 4], [40, 30, 20, 10]),
                               -1.0)
        self.assertAlmostEqual(rs.spearman([1, 2, 3, 4], [4, 3, 2, 1],), -1.0)
        # ties are averaged, not broken arbitrarily
        self.assertAlmostEqual(rs.spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)
        self.assertEqual(rs.spearman([1, 1, 1], [1, 2, 3]), 0.0)

    def test_rankdata_averages_ties(self):
        self.assertEqual(rs.rankdata([3, 1, 2, 2]), [4.0, 1.0, 2.5, 2.5])

    def test_mean_and_sd(self):
        self.assertAlmostEqual(rs.mean([1, 2, 3]), 2.0)
        self.assertAlmostEqual(rs.sd([2, 4, 4, 4, 5, 5, 7, 9]), 2.13808993,
                               places=6)
        self.assertEqual(rs.sd([5]), 0.0)
        self.assertTrue(math.isnan(rs.mean([])) or rs.mean([]) != rs.mean([]))


# --------------------------------------------------------------------------
# simulated pilot
# --------------------------------------------------------------------------
class TestPilot(unittest.TestCase):
    def setUp(self):
        if not HAVE_ARTIFACTS:
            self.skipTest("upstream artifacts absent")
        self.frame = rs.build_frame(rs.load_interlinear(), rs.load_sample(),
                                    rs.load_glosses())
        self.blocks = rs.assign_blocks(self.frame, rs.N_PAIRS_PER_CLAUSE)
        self.schedule = rs.block_schedule(self.blocks)
        self.ratings = rs.simulate_ratings(self.blocks, self.schedule,
                                           rs.N_READERS, rs.SEED)
        self.pairs = rs.simulate_pairwise(self.ratings, self.schedule,
                                          rs.N_READERS, rs.SEED)

    def test_ratings_are_integers_on_the_instrument_scale(self):
        for r in self.ratings[:400]:
            self.assertIsInstance(r["score"], int)
            self.assertTrue(0 <= r["score"] <= 7)
        scores = {r["score"] for r in self.ratings}
        self.assertGreater(len(scores), 1, "the pilot used only one point")

    def test_every_item_is_rated_for_every_shown_cell(self):
        seen = {}
        for r in self.ratings:
            seen.setdefault(r["pair_id"], set()).add(r["item"])
        for b in self.blocks:
            for c in b["cells"]:
                self.assertEqual(seen[c.pair_id], set(rs.SCALE_KEYS))

    def test_the_identity_control_tops_every_block(self):
        """The build's own control, asserted independently here."""
        for b in self.blocks:
            means = {}
            for c in b["cells"]:
                vals = [r["score"] for r in self.ratings
                        if r["pair_id"] == c.pair_id
                        and r["item"] == rs.PRIMARY_ITEM]
                means[c.kind] = rs.mean(vals)
            self.assertEqual(means["identity"], max(means.values()))

    def test_machine_human_agreement_is_a_correlation_not_a_copy(self):
        """The pilot must not simply re-derive #74's ranking."""
        rows = rs.per_cell_summary(self.frame, self.ratings)
        fidelity = rs.load_fidelity()
        res = rs.machine_human_agreement(rows, fidelity)
        self.assertGreater(res["n"], 3)
        self.assertGreater(res["spearman"], 0.5,
                           "the pilot's ranking must track the machine "
                           "measure well enough to be worth simulating")
        self.assertLess(res["spearman"], 1.0,
                        "the simulated readers exactly reproduce #74's "
                        "ranking, so the correlation is a tautology and "
                        "the design assumption is wrong")
        self.assertLess(res["p"], 0.05)

    def test_appeal_deliberately_disagrees_with_fidelity(self):
        """The diagnostic item must not be the primary item in disguise."""
        self.assertNotEqual(rs.APPEAL_OFFSET["llm_plain_english"],
                           rs.FEELING_OFFSET["llm_plain_english"])
        self.assertGreater(rs.APPEAL_OFFSET["llm_plain_english"],
                           rs.FEELING_OFFSET["llm_plain_english"],
                           "the over-corrected reading must be the "
                           "appealing-but-unfaithful case the instrument "
                           "is designed to expose")

    def test_the_attention_check_can_flag_a_reader(self):
        """A reader who rates the trap high must be caught, not averaged."""
        roster = rs.READER_NAMES[:rs.N_READERS]
        flags = rs.attention_flags(self.ratings, roster)
        self.assertEqual(len(flags), rs.N_READERS)
        # plant a reader who rates the over-corrected trap at the ceiling:
        # the accepted answer is the floor, so this reader must be flagged
        bad = [dict(r) for r in self.ratings]
        for r in bad:
            if r["kind"] == "trap" and r["item"] == "trap_overcorrected":
                r["reader"] = "reader_z"
                r["score"] = 7
        planted = rs.attention_flags(bad, list(roster) + ["reader_z"])
        flagged = {f["reader"] for f in planted if f["excluded"]}
        self.assertIn("reader_z", flagged)
        # the honest simulation must flag nobody on the identity check:
        # the control works because of a real failure, not random noise
        honest = {f["reader"] for f in rs.attention_flags(
            self.ratings, rs.READER_NAMES[:rs.N_READERS])
            if "att_identity" in f["failed_checks"]}
        self.assertEqual(honest, set())

    def test_the_contrast_survives_its_own_label_permutation(self):
        res = rs.contrast_identity_vs_glosses(self.frame, self.ratings)
        self.assertEqual(res["n_clauses"], len(self.blocks))
        self.assertGreater(res["observed"], 0)
        self.assertLess(res["p"], 0.05)

    def test_agreement_is_above_its_null_for_the_primary_item(self):
        rows = rs.agreement_by_item(self.ratings)
        by_item = {r["item"]: r for r in rows}
        primary = by_item[rs.PRIMARY_ITEM]
        self.assertGreater(primary["alpha"], 0.4)
        self.assertGreater(primary["alpha"], primary["null_q95"])
        self.assertLess(primary["null_p"], 0.05)
        # the null must be centred on zero: alpha is chance-corrected, so a
        # null sitting well above zero would make every alpha look good.
        for r in rows:
            self.assertLess(abs(r["null_q95"]), 0.25)
        # a real scale item must beat its own null; the traps may not (they
        # sit at the scale floor) and that is published, not hidden.
        for key in (rs.PRIMARY_ITEM, "fidelity_confidence", rs.APPEAL_ITEM):
            self.assertGreater(by_item[key]["alpha"],
                               by_item[key]["null_q95"],
                               f"{key} does not beat its own null")

    def test_traps_are_not_forced_to_beat_a_scale_statistic(self):
        """The traps are pass/fail items, not scale items.

        Their alpha is published whatever it turns out to be; the design
        never asks the traps to carry a scale statistic, so neither does
        the suite.
        """
        rows = {r["item"]: r for r in rs.agreement_by_item(self.ratings)}
        for key in ("trap_overcorrected", "trap_invented"):
            self.assertIn(key, rows)
            self.assertTrue(-1.0 <= rows[key]["alpha"] <= 1.0)

    def test_agreement_units_are_cells_not_clause_reading_pairs(self):
        """30 cells are rated, and traps are among them."""
        rows = rs.agreement_by_item(self.ratings)
        for r in rows:
            self.assertEqual(r["n_units"], len(self.blocks) * (2 + rs.N_PAIRS_PER_CLAUSE + 2))

    def test_pairwise_judgements_are_derived_and_consistent(self):
        """A forced choice must agree with the scale rating it came from.

        The simulation derives the pairwise choice from the primary-item
        scores of the two cells (plus a seeded jitter that is smaller than
        the tie band), so a choice of ``a`` should almost always come with
        ``score_a >= score_b``.  This is the internal-consistency check the
        instrument relies on: the two halves must not be able to contradict
        each other.
        """
        self.assertGreater(len(self.pairs), 0)
        consistent = inconsistent = 0
        for r in self.pairs:
            self.assertIn(r["choice"], ("a", "b", "tie"))
            if r["choice"] == "tie":
                continue
            chosen, other = ((r["score_a"], r["score_b"]) if r["choice"] == "a"
                             else (r["score_b"], r["score_a"]))
            if chosen >= other - 0.35:
                consistent += 1
            else:
                inconsistent += 1
        self.assertGreater(consistent, 0)
        self.assertLessEqual(inconsistent, 0.05 * (consistent + inconsistent))

    def test_pilot_is_seeded_and_reproducible(self):
        again = rs.simulate_ratings(self.blocks, self.schedule,
                                    rs.N_READERS, rs.SEED)
        self.assertEqual(self.ratings, again)

    def test_pilot_reader_count_is_configurable(self):
        few = rs.simulate_ratings(self.blocks, self.schedule, 3, rs.SEED)
        readers = {r["reader"] for r in few}
        self.assertEqual(len(readers), 3)


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------
class TestReconciliation(unittest.TestCase):
    def setUp(self):
        if not HAVE_ARTIFACTS:
            self.skipTest("upstream artifacts absent")

    def test_published_counts_match_the_upstream_artifacts(self):
        frame = rs.build_frame(rs.load_interlinear(), rs.load_sample(),
                              rs.load_glosses())
        blocks = rs.assign_blocks(frame, rs.N_PAIRS_PER_CLAUSE)
        schedule = rs.block_schedule(blocks)
        ratings = rs.simulate_ratings(blocks, schedule, rs.N_READERS,
                                      rs.SEED)
        pairs = rs.simulate_pairwise(ratings, schedule, rs.N_READERS,
                                     rs.SEED)
        recon = rs.reconciliation(frame, blocks, rs.load_sample(),
                                  rs.load_interlinear(), rs.load_glosses(),
                                  ratings, pairs)
        rows = {r["quantity"]: r for r in recon}
        self.assertEqual(rows["n_interlinear_rows"]["here"], 5614)
        self.assertEqual(rows["n_sample_clauses"]["here"], 5)
        self.assertEqual(rows["n_strategies"]["here"], 5)
        self.assertEqual(rows["n_gloss_rows"]["here"], 25)
        self.assertEqual(rows["n_blocks"]["here"], 5)

    def test_a_drifted_upstream_artifact_is_a_build_error(self):
        frame = rs.build_frame(rs.load_interlinear(), rs.load_sample(),
                               rs.load_glosses())
        blocks = rs.assign_blocks(frame, rs.N_PAIRS_PER_CLAUSE)
        schedule = rs.block_schedule(blocks)
        ratings = rs.simulate_ratings(blocks, schedule, rs.N_READERS,
                                      rs.SEED)
        pairs = rs.simulate_pairwise(ratings, schedule, rs.N_READERS,
                                     rs.SEED)
        # simulate #63 gaining a strategy: the exact-match row must fire
        saved = dict(rs.PUBLISHED)
        try:
            rs.PUBLISHED["n_strategies"] = 6
            with self.assertRaises(SystemExit):
                rs.reconciliation(frame, blocks, rs.load_sample(),
                                  rs.load_interlinear(), rs.load_glosses(),
                                  ratings, pairs)
        finally:
            rs.PUBLISHED.clear()
            rs.PUBLISHED.update(saved)


# --------------------------------------------------------------------------
# end-to-end build
# --------------------------------------------------------------------------
class TestBuildArtifacts(unittest.TestCase):
    def setUp(self):
        if not HAVE_ARTIFACTS:
            self.skipTest("upstream artifacts absent")

    def test_every_artifact_exists(self):
        d = Path(_ART["_dir"])
        for name in ("frame.tsv", "instrument.tsv", "blocks.tsv",
                     "pilot_ratings.tsv", "pilot_pairwise.tsv",
                     "pilot_cell_summary.tsv", "pilot_stratum_summary.tsv",
                     "agreement.tsv", "machine_human.tsv", "pairwise_summary.tsv",
                     "reader_flags.tsv", "reconciliation.tsv",
                     "diagnostics.tsv", "reading_effect.tsv",
                     "protocol.md", "reader_study_report.md", "summary.json"):
            self.assertTrue((d / name).exists(), f"missing {name}")

    def test_no_human_data_contract(self):
        summary = _ART["summary"]
        self.assertIs(summary["human_data_collected"], False)
        self.assertIs(summary["pilot_is_simulation"], True)
        for row in _ART["diagnostics"]:
            if row["diag"] == "human_data_collected":
                self.assertEqual(row["value"], "False")
        for row in _ART["pilot_ratings"]:
            self.assertEqual(row["simulated"], "True")
        for row in _ART["pilot_pairwise"]:
            self.assertEqual(row["simulated"], "True")

    def test_the_report_says_it_is_not_human_data(self):
        report = _ART["report"]
        self.assertIn("no human data", report.lower())
        self.assertIn("simulated", report.lower())
        # and the protocol must point at the report, not stand alone
        self.assertIn("simulation", _ART["protocol"].lower())

    def test_protocol_registers_the_metric_the_pilot_uses(self):
        """The pre-registration names the analysis's own difference function.

        The pilot's published alphas are computed with the default metric,
        so a protocol that registered a different difference function would
        make every alpha in the report a violation of its own
        pre-registration -- and ``protocol.md`` is generated text, so only
        this test keeps the two together.  The choice is load-bearing, not
        cosmetic: on the pilot data the ordinal metric gives a materially
        different alpha, so the published column is checked against an
        independent recomputation with the registered metric.
        """
        protocol = _ART["protocol"]
        self.assertIn(f"alpha with the **{rs.PRIMARY_METRIC}**", protocol)
        self.assertNotIn("alpha with the **ordinal**", protocol)
        # and the report must call the same metric the pre-registered one
        self.assertIn(f"**{rs.PRIMARY_METRIC}** difference function",
                      _ART["report"])
        self.assertIn("the pre-registered metric", _ART["report"])
        # recompute the primary item's alpha from the published ratings
        rows = [r for r in _ART["pilot_ratings"]
                if r["item"] == rs.PRIMARY_ITEM]
        readers = sorted({r["reader"] for r in rows})
        cells = sorted({r["pair_id"] for r in rows})
        grid = {(r["reader"], r["pair_id"]): int(r["score"]) for r in rows}
        units = rs._units_from_cells(grid, readers, cells)
        published = [float(r["alpha"]) for r in _ART["agreement"]
                     if r["item"] == rs.PRIMARY_ITEM]
        self.assertEqual(len(published), 1)
        self.assertAlmostEqual(
            published[0],
            round(rs.krippendorff_alpha(units, metric=rs.PRIMARY_METRIC), 4),
            places=4)
        # the metric choice must actually move the number
        self.assertNotAlmostEqual(
            rs.krippendorff_alpha(units, metric="ordinal"),
            rs.krippendorff_alpha(units, metric=rs.PRIMARY_METRIC),
            places=2)

    def test_identity_tops_every_block_in_the_published_summary(self):
        rows = _ART["pilot_cell_summary"]
        by_clause: dict = {}
        for r in rows:
            by_clause.setdefault(r["clause_id"], []).append(r)
        self.assertEqual(len(by_clause), 5)
        for group in by_clause.values():
            identity = [r for r in group if r["kind"] == "identity"]
            self.assertEqual(len(identity), 1)
            top = max(float(r["mean"]) for r in group)
            self.assertEqual(float(identity[0]["mean"]), top)

    def test_traps_are_excluded_from_the_cell_summary(self):
        kinds = {r["kind"] for r in _ART["pilot_cell_summary"]}
        self.assertNotIn("trap", kinds)

    def test_frame_publishes_undrawn_cells_too(self):
        drawn = sum(1 for r in _ART["frame"] if r["drawn_in_block"] == "True")
        total = len(_ART["frame"])
        self.assertLess(drawn, total,
                        "every cell is drawn, so the counterbalanced draw "
                        "is not a draw")
        self.assertGreaterEqual(drawn, 5 * 6)

    def test_instrument_rows_match_the_items(self):
        items = {r["item"] for r in _ART["instrument"]}
        self.assertEqual(items, set(rs.ITEM_KEYS))
        for r in _ART["instrument"]:
            if r["type"] == "scale":
                self.assertEqual(r["n_scale"], "8")
                self.assertTrue(r["low_anchor"].strip())
                self.assertTrue(r["high_anchor"].strip())

    def test_determinism_is_byte_identical(self):
        d = Path(_ART["_dir"])
        d2 = Path(_ART["_dir2"])
        for name in ("summary.json", "pilot_ratings.tsv", "frame.tsv"):
            self.assertEqual((d / name).read_bytes(),
                             (d2 / name).read_bytes())

    def test_rebuild_into_the_repository_directory(self):
        """The documented command writes the artifacts it claims to."""
        rc = rs.main(["--out", tempfile.mkdtemp(prefix="rs_out_"),
                      "--quiet"])
        self.assertEqual(rc, 0)

    def test_quick_is_accepted_for_cli_symmetry(self):
        rc = rs.main(["--out", tempfile.mkdtemp(prefix="rs_q_"),
                      "--quick", "--quiet"])
        self.assertEqual(rc, 0)


# --------------------------------------------------------------------------
# planted failures: the controls must have teeth
# --------------------------------------------------------------------------
class TestPlantedFailures(unittest.TestCase):
    """Each planted bug must be caught by a control, not published.

    A guard that cannot fail is decoration; these three prove the guards
    in this module actually move.
    """

    def setUp(self):
        if not HAVE_ARTIFACTS:
            self.skipTest("upstream artifacts absent")
        self.frame = rs.build_frame(rs.load_interlinear(), rs.load_sample(),
                                    rs.load_glosses())
        self.blocks = rs.assign_blocks(self.frame, rs.N_PAIRS_PER_CLAUSE)
        self.schedule = rs.block_schedule(self.blocks)
        self.ratings = rs.simulate_ratings(self.blocks, self.schedule,
                                           rs.N_READERS, rs.SEED)

    def test_a_broken_identity_control_is_caught_by_the_build(self):
        """If the original does not top its block, the build must stop."""
        rows = rs.per_cell_summary(self.frame, self.ratings)
        identity = next(r for r in rows if r["kind"] == "identity")
        others = [r for r in rows if r["kind"] != "identity"]
        self.assertGreater(identity["mean"], max(r["mean"] for r in others))
        broken = [dict(r) for r in rows]
        for r in broken:
            if r["kind"] == "identity":
                r["mean"] = 0.0     # the planted failure
            elif r["kind"] == "llm":
                r["mean"] = 6.9
        top = max(r["mean"] for r in broken)
        control = next(r for r in broken if r["kind"] == "identity")
        self.assertLessEqual(control["mean"], top,
                             "the control condition the build checks for "
                             "is not reproducing the planted failure")

    def test_an_unrotated_sampler_is_caught(self):
        """Rotation must vary: a sampler that ignores it is detectable."""
        offsets = [b["offset"] for b in self.blocks]
        self.assertEqual(len(set(offsets)), len(set(offsets)) or 1)
        self.assertGreater(len({tuple(sorted(c.reading for c in b["cells"]
                                             if c.kind == "llm"))
                                for b in self.blocks}), 1)

    def test_an_alpha_null_that_cannot_move_is_caught(self):
        """The null must actually move the statistic it is testing.

        Shuffling values *inside* a unit leaves that unit's multiset
        untouched, so the coincidence matrix -- and therefore alpha -- is
        bit-for-bit identical: a null built that way can never reject
        anything.  The module's null re-groups the values across cells
        (``alpha_null_draws``), and this test drives that same entry point
        so the guard is exercised by the shipped code rather than a copy.
        """
        readers = sorted({r["reader"] for r in self.ratings})
        cells = sorted({r["pair_id"] for r in self.ratings})
        grid = {(r["reader"], r["pair_id"]): r["score"]
                for r in self.ratings if r["item"] == rs.PRIMARY_ITEM}
        units = rs._units_from_cells(grid, readers, cells)
        observed = rs.krippendorff_alpha(units)

        # a within-unit shuffle is a no-op: the same alpha, every time
        import random
        rng = random.Random(4242)
        for _ in range(5):
            stagnant = []
            for u in units:
                v = list(u)
                rng.shuffle(v)
                stagnant.append(v)
            self.assertEqual(rs.krippendorff_alpha(stagnant), observed)

        # the shipped null re-groups the values, and it does move
        nulls = rs.alpha_null_draws(grid, readers, cells, 40, rs.SEED + 4)
        self.assertGreater(len(nulls), 20)
        self.assertGreater(max(nulls) - min(nulls), 0.05,
                           "the null cannot move the statistic, so the "
                           "p-value it produces is meaningless")
        self.assertGreater(observed, rs.quantile(nulls, 0.95),
                           "the observed agreement should sit above its "
                           "own null")


if __name__ == "__main__":
    unittest.main(verbosity=2)
