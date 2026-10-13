#!/usr/bin/env python3
"""Regression suite for #52 operator-word mutual information.

Runs the builder once into a temp directory, then checks the invariants that
matter: the operator screen (#17/#10 counts), the term resolution on the unit
layer (free vs in-unit operators, page boundaries), the domain seed matching
(#29's literal), the MI/G^2/BH formulas, the matrix marginals, the issue's own
two verdicts, and the artifacts.

    python work/test_operator_word_mi.py
"""
import collections
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import operator_word_mi as owm  # noqa: E402

_ARTIFACTS = {}
N_SHUFFLE_TEST = 40


def setUpModule():
    tmp = Path(tempfile.mkdtemp(prefix="opwordmi_test_"))
    owm.build(tmp, N_SHUFFLE_TEST, owm.SEED, log=lambda *a, **k: None)
    for name in ("mi_matrix", "operator_profile", "operator_seed_terms",
                 "mi_summary", "diagnostics", "reconciliation"):
        _ARTIFACTS[name] = owm.read_tsv(tmp / f"{name}.tsv")
    with (tmp / "summary.json").open(encoding="utf-8") as fh:
        _ARTIFACTS["summary"] = json.load(fh)
    _ARTIFACTS["_dir"] = tmp
    _ARTIFACTS["report"] = (tmp / "operator_domain_mi_report.md").read_text(
        encoding="utf-8")


def rows(name):
    return _ARTIFACTS[name]


def diag():
    return {r["diag"]: r["value"] for r in rows("diagnostics")}


def matrix(side, op, domain):
    for r in rows("mi_matrix"):
        if r["side"] == side and r["operator"] == op and r["domain"] == domain:
            return r
    raise KeyError((side, op, domain))


def profile(side, op):
    for r in rows("operator_profile"):
        if r["side"] == side and r["operator"] == op:
            return r
    raise KeyError((side, op))


def seeds():
    return owm.load_domain_seeds()


# ---------------------------------------------------------------------------
class TestScreening(unittest.TestCase):
    def test_operator_stream_matches_17(self):
        d = diag()
        self.assertEqual(d["op_tokens_stream"], "11000")      # #17
        self.assertEqual(d["op_tokens_markup_removed"], "732")
        self.assertEqual(d["op_occurrences"], "10268")

    def test_partition_of_occurrences(self):
        d = diag()
        self.assertEqual(int(d["op_in_unit"]) + int(d["op_free"]),
                         int(d["op_occurrences"]))
        # #17's seam count, minus the one inside a markup span
        self.assertEqual(d["op_in_unit"], "2815")
        self.assertEqual(d["op_free"], "7453")

    def test_no_free_operator_inside_a_unit_span(self):
        # geometric containment reproduces #17's own `unit` column
        self.assertEqual(diag()["op_free_inside_unit_span"], "0")

    def test_header_operators_are_carried(self):
        self.assertEqual(diag()["header_region_ops"], "51")

    def test_operator_vocabulary_is_10s(self):
        ops = {r["operator"] for r in rows("mi_matrix")}
        self.assertEqual(ops, {"::", "//", "=", "/", ">", "<<", ">>", "<",
                               "++", "|", "+"})


# ---------------------------------------------------------------------------
class TestTermResolution(unittest.TestCase):
    """Pure geometry: a page's units + operators -> the terms on each side."""

    def test_free_operator_takes_unit_surfaces(self):
        units = [(0, 5, "alpha"), (6, 11, "beta"), (12, 17, "gamma")]
        ops = [(5, 6, "::")]
        self.assertEqual(owm.resolve_page(units, ops),
                         [("::", False, "alpha", "beta")])

    def test_free_operator_skips_intervening_operators(self):
        # `word << // word`: the `//` neighbour is the next unit, not `<<`
        units = [(0, 5, "alpha"), (20, 25, "beta")]
        ops = [(6, 8, "<<"), (9, 11, "//")]
        out = {o: (iu, lt, rt) for o, iu, lt, rt in
               owm.resolve_page(units, ops)}
        self.assertEqual(out["//"], (False, "alpha", "beta"))

    def test_in_unit_operator_takes_operand_substrings(self):
        units = [(0, 10, "suck=blood")]
        ops = [(4, 5, "=")]
        self.assertEqual(owm.resolve_page(units, ops),
                         [("=", True, "suck", "blood")])

    def test_in_unit_hyphen_operand(self):
        units = [(0, 8, "boy-roid")]
        ops = [(3, 4, "-")]
        self.assertEqual(owm.resolve_page(units, ops),
                         [("-", True, "boy", "roid")])

    def test_multi_seam_unit_both_operators(self):
        units = [(0, 13, "gene=TV/spasm")]
        ops = [(4, 5, "="), (7, 8, "/")]
        out = {o: (iu, lt, rt) for o, iu, lt, rt in
               owm.resolve_page(units, ops)}
        self.assertEqual(out["="], (True, "gene", "TV/spasm"))
        self.assertEqual(out["/"], (True, "gene=TV", "spasm"))

    def test_page_boundary_is_none(self):
        # an operator after the last unit has no right term
        units = [(0, 5, "alpha")]
        self.assertEqual(owm.resolve_page(units, [(5, 6, "::")]),
                         [("::", False, "alpha", None)])
        # an operator before the first unit has no left term
        units = [(10, 15, "beta")]
        self.assertEqual(owm.resolve_page(units, [(0, 1, "::")]),
                         [("::", False, None, "beta")])
        # an operator inside a unit is in-unit, with operand substrings
        units = [(0, 5, "alpha")]
        self.assertEqual(owm.resolve_page(units, [(1, 2, "::")]),
                         [("::", True, "a", "pha")])

    def test_no_units_at_all(self):
        self.assertEqual(owm.resolve_page([], [(0, 1, "=")]),
                         [("=", False, None, None)])


# ---------------------------------------------------------------------------
class TestDomains(unittest.TestCase):
    def test_seeds_are_29s(self):
        s = seeds()
        self.assertEqual(set(s), set(owm.DOMAINS))
        self.assertIn("blood", s["body"])
        self.assertIn("tv", s["machine"])
        # #29's comment says "blood is body + violence" but the literal
        # violence set has no `blood`; the literal is what this build uses.
        self.assertNotIn("blood", s["violence"])
        self.assertEqual(len(s), 6)

    def test_compound_carries_its_parts_domain(self):
        s = seeds()
        self.assertEqual(owm.term_domains("drug-embryo", s),
                         {"body", "genetics"})
        self.assertEqual(owm.term_domains("body-omotya", s), {"body"})

    def test_single_seed_word(self):
        s = seeds()
        self.assertEqual(owm.term_domains("blood", s), {"body"})
        self.assertEqual(owm.term_domains("TV", s), {"machine"})
        self.assertEqual(owm.term_domains("embryo", s), {"body", "genetics"})
        self.assertEqual(owm.term_domains("clone", s),
                         {"machine", "genetics"})

    def test_no_seed_match(self):
        s = seeds()
        self.assertEqual(owm.term_domains("soul/gram", s), set())
        self.assertEqual(owm.term_domains("cold-blooded", s), set())
        self.assertEqual(owm.term_domains("zzzz", s), set())

    def test_strict_variant_is_whole_term_only(self):
        s = seeds()
        self.assertEqual(owm.term_domains("drug-embryo", s, strict=True),
                         set())
        self.assertEqual(owm.term_domains("blood", s, strict=True),
                         {"body"})

    def test_labels(self):
        s = seeds()
        self.assertEqual(owm.domain_label(None, s), "boundary")
        self.assertEqual(owm.domain_label("zzzz", s), "none")
        self.assertEqual(owm.domain_label("embryo", s), "body+genetics")


# ---------------------------------------------------------------------------
class TestStatistics(unittest.TestCase):
    def test_total_mi_of_independent_pairs_is_zero(self):
        pairs = [(chr(97 + i % 2), chr(65 + (i // 2) % 4)) for i in range(200)]
        self.assertAlmostEqual(owm.total_mi(pairs), 0.0, places=9)

    def test_total_mi_of_deterministic_pairs_is_entropy(self):
        pairs = [("op1", "a"), ("op1", "a"), ("op1", "a"),
                 ("op2", "b"), ("op2", "b")]
        # H(O) = -(3/5 log 3/5 + 2/5 log 2/5)
        h = -(0.6 * math.log2(0.6) + 0.4 * math.log2(0.4))
        self.assertAlmostEqual(owm.total_mi(pairs), h, places=9)

    def test_pmi_cell_formula_and_zero_cell(self):
        self.assertIsNone(owm.pmi_cell(0, 10, 10, 100))
        self.assertIsNone(owm.pmi_cell(5, 0, 10, 100))
        self.assertAlmostEqual(
            owm.pmi_cell(10, 50, 50, 100),
            math.log2(10 * 100 / (50 * 50)), places=9)

    def test_benjamini_hochberg(self):
        q = owm.benjamini_hochberg([0.01, 0.02, 0.03])
        self.assertEqual([round(x, 4) for x in q], [0.03, 0.03, 0.03])
        self.assertEqual(owm.benjamini_hochberg([]), [])

    def test_benjamini_hochberg_known_example(self):
        # the step-down min: q[2] is 0.084 (0.042*10/5), not 0.039*10/3
        q = owm.benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042,
                                    0.06, 0.074, 0.205, 0.212, 0.216])
        self.assertAlmostEqual(q[0], 0.01, places=6)
        self.assertAlmostEqual(q[2], 0.084, places=6)
        self.assertAlmostEqual(q[-1], 0.216, places=6)

    def test_g2_binary_at_chance_is_zero(self):
        self.assertAlmostEqual(owm.g2_binary(10, 100, 0.1), 0.0, places=9)

    def test_g2_binary_over_representation_is_large(self):
        g = owm.g2_binary(90, 100, 0.1)
        self.assertGreater(g, 100.0)

    def test_null_stats(self):
        mean, _sd, z, emp = owm.null_stats(5.0, [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(mean, 2.5, places=9)
        self.assertGreater(z, 0)
        self.assertEqual(emp, 0.2)          # 1 of 5 nulls >= 5.0

    def test_incidence_mi_of_single_domain_is_zero(self):
        occ = [{"op": "=", "left": "a", "right": "b"}]
        s = {"body": {"b"}}
        self.assertAlmostEqual(
            owm.incidence_mi(occ, "right", s), 0.0, places=9)

    def test_mi_contribution_sums_to_total(self):
        occ = [{"op": "=" if i % 2 else "::",
                "left": "x",
                "right": "blood" if i % 3 else "zzz"}
               for i in range(300)]
        s = {"body": {"blood"}}
        contrib = {r["operator"]: r["mi_contribution_bits"]
                   for r in owm.mi_contribution_rows(occ, "right", s)}
        self.assertAlmostEqual(sum(contrib.values()),
                               owm.total_mi(owm.side_pairs(occ, "right", s)[1]),
                               places=3)


# ---------------------------------------------------------------------------
class TestMatrix(unittest.TestCase):
    def test_grid_is_complete_and_unique(self):
        keys = [(r["side"], r["operator"], r["domain"])
                for r in rows("mi_matrix")]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), 2 * 11 * 6)

    def test_marginals_reconcile(self):
        by_side = collections.defaultdict(list)
        for r in rows("mi_matrix"):
            by_side[r["side"]].append(r)
        for rs in by_side.values():
            n_inc = {r["n_incidences"] for r in rs}
            self.assertEqual(len(n_inc), 1)
            n_inc = int(n_inc.pop())
            self.assertEqual(sum(int(r["f_cell"]) for r in rs), n_inc)
            for op in {r["operator"] for r in rs}:
                op_rows = [r for r in rs if r["operator"] == op]
                self.assertEqual(sum(int(r["f_cell"]) for r in op_rows),
                                 int(op_rows[0]["f_op_incidences"]))
            for d in owm.DOMAINS:
                d_rows = [r for r in rs if r["domain"] == d]
                self.assertEqual(sum(int(r["f_cell"]) for r in d_rows),
                                 int(d_rows[0]["f_domain_incidences"]))

    def test_expected_and_lift_follow_from_counts(self):
        for r in rows("mi_matrix"):
            f = int(r["f_cell"])
            fo, fd, n = (int(r["f_op_incidences"]),
                         int(r["f_domain_incidences"]),
                         int(r["n_incidences"]))
            self.assertAlmostEqual(float(r["expected"]),
                                   round(fo * fd / n, 3), places=2)
            if f > 0:
                self.assertAlmostEqual(
                    float(r["lift"]), round(f * n / (fo * fd), 4), places=3)
                self.assertAlmostEqual(float(r["pmi_bits"]),
                                       round(math.log2(f * n / (fo * fd)), 4),
                                       places=3)
            else:
                self.assertEqual(r["pmi_bits"], "")

    def test_known_cells(self):
        # verified against #12's own inventories and the seed vocabulary
        self.assertEqual(matrix("right", "=", "body")["f_cell"], "322")
        self.assertEqual(matrix("right", "=", "machine")["f_cell"], "274")
        self.assertEqual(matrix("right", "::", "machine")["f_cell"], "150")
        self.assertEqual(matrix("right", "::", "body")["f_cell"], "68")
        self.assertEqual(matrix("right", "<<", "body")["f_cell"], "42")
        self.assertEqual(matrix("left", "=", "genetics")["f_cell"], "308")

    def test_p_and_q_are_probabilities(self):
        for r in rows("mi_matrix"):
            if r["p_value"] != "":
                self.assertGreaterEqual(float(r["p_value"]), 0.0)
                self.assertLessEqual(float(r["p_value"]), 1.0)
                self.assertLessEqual(float(r["q_bh"]), 1.0)
                self.assertLessEqual(float(r["p_value"]), float(r["q_bh"]))


# ---------------------------------------------------------------------------
class TestVerdicts(unittest.TestCase):
    """The issue's own two hypotheses, plus the block-delimiter one."""

    def test_double_colon_avoids_body_terms(self):
        r = matrix("right", "::", "body")
        self.assertLess(float(r["lift"]), 1.0)
        self.assertLess(float(r["q_bh"]), 0.05)
        self.assertEqual(owm.q_lift_verdict(r), "avoids")

    def test_equals_binds_machine_and_body(self):
        m = matrix("right", "=", "machine")
        self.assertGreater(float(m["lift"]), 1.0)
        self.assertLess(float(m["q_bh"]), 0.05)
        self.assertEqual(owm.q_lift_verdict(m), "over-selects")
        b = matrix("right", "=", "body")
        self.assertGreater(float(b["lift"]), 1.0)
        self.assertLess(float(b["q_bh"]), 0.05)

    def test_equals_left_side_is_genetics(self):
        r = matrix("left", "=", "genetics")
        self.assertGreater(float(r["lift"]), 1.0)
        self.assertLess(float(r["q_bh"]), 0.05)

    def test_block_open_prefers_body(self):
        r = matrix("right", "<<", "body")
        self.assertGreater(float(r["lift"]), 1.0)
        self.assertLess(float(r["q_bh"]), 0.05)

    def test_verdict_helper(self):
        self.assertEqual(
            owm.q_lift_verdict({"q_bh": 0.049, "lift": 2.0}), "over-selects")
        self.assertEqual(
            owm.q_lift_verdict({"q_bh": 0.049, "lift": 0.5}), "avoids")
        self.assertEqual(
            owm.q_lift_verdict({"q_bh": 0.5, "lift": 2.0}), "no preference")
        self.assertEqual(owm.q_lift_verdict({"q_bh": "", "lift": ""}),
                         "no preference")


# ---------------------------------------------------------------------------
class TestProfile(unittest.TestCase):
    def test_operator_counts_sum_to_occurrences(self):
        for side in owm.SIDES:
            n = sum(int(r["n_op"]) for r in rows("operator_profile")
                    if r["side"] == side)
            self.assertEqual(n, int(diag()["op_occurrences"]))

    def test_shares_partition_the_stream(self):
        d = diag()
        n = int(d["op_occurrences"])
        for side in owm.SIDES:
            total = (int(d[f"domain_bearing_{side}"])
                     + int(d[f"none_{side}"]) + int(d[f"boundary_{side}"]))
            self.assertEqual(total, n, side)

    def test_profile_shares_are_valid(self):
        for r in rows("operator_profile"):
            for k in ("p_none", "p_boundary", "p_multi_domain"):
                self.assertGreaterEqual(float(r[k]), 0.0)
                self.assertLessEqual(float(r[k]), 1.0)
            self.assertIn(r["verdict"], ("over-selects", "avoids",
                                         "no preference"))

    def test_mi_contribution_sums_to_total(self):
        for r in rows("mi_summary"):
            if r["side"] == "right" and r["measure"] == "I_term":
                total = float(r["observed_bits"])
        got = sum(float(r["mi_contribution_bits"])
                  for r in rows("operator_profile") if r["side"] == "right")
        self.assertAlmostEqual(got, total, places=1)

    def test_top_domain_consistent_with_matrix(self):
        for r in rows("operator_profile"):
            if r["side"] != "right" or not r["top_domain"]:
                continue
            cell = matrix("right", r["operator"], r["top_domain"])
            self.assertEqual(int(cell["f_cell"]), int(r["top_f"]))


# ---------------------------------------------------------------------------
class TestTotals(unittest.TestCase):
    def test_mi_far_above_the_null(self):
        for r in rows("mi_summary"):
            if r["measure"].startswith("I_"):
                self.assertGreater(float(r["observed_bits"]),
                                   float(r["null_mean"]))
                self.assertGreater(float(r["z"]), 0.0)
                self.assertLessEqual(float(r["p_empirical"]), 1.0)

    def test_entropy_rows_present(self):
        got = {(r["side"], r["measure"]) for r in rows("mi_summary")}
        for side in owm.SIDES:
            self.assertIn((side, "H_T"), got)
            self.assertIn((side, "H_T_given_O"), got)

    def test_conditional_entropy_below_entropy(self):
        vals = {(r["side"], r["measure"]): float(r["observed_bits"])
                for r in rows("mi_summary")}
        for side in owm.SIDES:
            self.assertLess(vals[(side, "H_T_given_O")],
                            vals[(side, "H_T")])
            # I(O;T) == H(T) - H(T|O)
            self.assertAlmostEqual(
                vals[(side, "H_T")] - vals[(side, "H_T_given_O")],
                vals[(side, "I_term")], places=1)

    def test_i_term_exceeds_i_label(self):
        vals = {(r["side"], r["measure"]): float(r["observed_bits"])
                for r in rows("mi_summary")}
        for side in owm.SIDES:
            self.assertGreater(vals[(side, "I_term")], vals[(side, "I_label")])

    def test_scale_against_11(self):
        vals = {(r["side"], r["measure"]): float(r["observed_bits"])
                for r in rows("mi_summary")}
        self.assertGreater(vals[("right", "I_term")], owm.OP_LAG1_MI_11)


# ---------------------------------------------------------------------------
class TestSeedTerms(unittest.TestCase):
    def test_only_seed_terms_are_listed(self):
        s = seeds()
        for r in rows("operator_seed_terms"):
            self.assertTrue(r["domains"])
            for d in r["domains"].split("|"):
                self.assertIn(d, owm.DOMAINS)
            self.assertTrue(owm.term_domains(r["term"], s))

    def test_known_operand_pairs(self):
        idx = {(r["side"], r["operator"], r["term"]): r
               for r in rows("operator_seed_terms")}
        self.assertEqual(idx[("right", "=", "blood")]["f_cell"], "124")
        self.assertEqual(idx[("right", "=", "TV")]["f_cell"], "164")
        self.assertGreater(float(idx[("right", "=", "blood")]["pmi_bits"]),
                           float(idx[("right", "::", "blood")]["pmi_bits"]))

    def test_counts_within_marginals(self):
        for r in rows("operator_seed_terms"):
            self.assertLessEqual(int(r["f_cell"]), int(r["f_op"]))
            self.assertLessEqual(int(r["f_cell"]), int(r["f_term"]))
            self.assertLessEqual(int(r["f_term"]), int(r["n"]))


# ---------------------------------------------------------------------------
class TestSensitivity(unittest.TestCase):
    def test_seam_split_adds_part_only_hits(self):
        summary = _ARTIFACTS["summary"]
        right = summary["strict"]["right"]
        self.assertGreater(right["n_dom_only_from_parts"], 0)
        self.assertLess(right["n_inc"], 1753)
        # the domain total barely moves between matching rules
        default = summary["mi"]["right"]["I_label"][0]
        self.assertAlmostEqual(default, right["I_label"], delta=0.01)

    def test_halves_are_stable(self):
        halves = _ARTIFACTS["summary"]["halves"]
        for side in owm.SIDES:
            a, b = halves[side]["first"][1], halves[side]["second"][1]
            self.assertGreater(min(a, b) / max(a, b), 0.85)


# ---------------------------------------------------------------------------
class TestArtifacts(unittest.TestCase):
    def test_report_headline(self):
        r = _ARTIFACTS["report"]
        self.assertIn("# Operator-word mutual information (#52)", r)
        self.assertIn("## 3. The MI matrix", r)
        self.assertIn("`=` before tech terms?", r)
        self.assertIn("Generated by `work/operator_word_mi.py`", r)

    def test_report_matrix_has_every_operator(self):
        r = _ARTIFACTS["report"]
        for op in ("::", "//", "=", "/", ">", "<<", ">>", "<", "++", "+"):
            self.assertIn(f"| `{op}` |", r)

    def test_summary_counts_match_diagnostics(self):
        s = _ARTIFACTS["summary"]
        d = diag()
        self.assertEqual(s["schema"], owm.SCHEMA)
        self.assertEqual(s["counts"]["op_occurrences"],
                         int(d["op_occurrences"]))

    def test_top_cells_ranked(self):
        cells = _ARTIFACTS["summary"]["top_right_cells"]
        vals = [float(c["pmi_bits"]) for c in cells]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_tsv_read_with_split(self):
        for name in ("mi_matrix", "operator_profile", "operator_seed_terms",
                     "mi_summary", "diagnostics", "reconciliation"):
            path = _ARTIFACTS["_dir"] / f"{name}.tsv"
            with path.open(encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                for line in fh:
                    if not line.strip():
                        continue
                    self.assertEqual(len(line.rstrip("\n").split("\t")),
                                     len(header), name)

    def test_reconciliation_ids(self):
        got = {r["quantity"] for r in rows("reconciliation")}
        self.assertIn("operator tokens (#17)", got)
        self.assertIn("in-unit operators", got)
        self.assertIn("domain seeds (#29)", got)


# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_build(self):
        import tempfile
        a = Path(tempfile.mkdtemp(prefix="opwordmi_a_"))
        b = Path(tempfile.mkdtemp(prefix="opwordmi_b_"))
        owm.build(a, 10, 7, log=lambda *x, **k: None)
        owm.build(b, 10, 7, log=lambda *x, **k: None)
        self.assertEqual(
            (a / "mi_summary.tsv").read_text(encoding="utf-8"),
            (b / "mi_summary.tsv").read_text(encoding="utf-8"))
        self.assertEqual(
            (a / "mi_matrix.tsv").read_text(encoding="utf-8"),
            (b / "mi_matrix.tsv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
