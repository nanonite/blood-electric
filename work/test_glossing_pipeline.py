#!/usr/bin/env python3
"""Regression suite for the LLM glossing pipeline (#63).

Run the whole work/ suite with:
    python -m unittest discover -s work -p 'test_*.py'

The suite is two-layered, like the workspace's other test suites:

  * **synthetic tests** on hand-built clauses, which pin the operator
    scanner, the four checks and their failure paths.  These are the tests
    that matter most here, because three of the four checks were found
    wrong *by this suite's own reasoning* before any model was run and a
    check that cannot fail is worthless.  Each one therefore has a planted
    failure that must be caught, and the tests that pin the bugs found
    during development are kept:

      - the operator scanner used to count the intra-word hyphen in
        `porno-nerve` as an operator, so a faithful gloss was reported as
        having invented a `-`;
      - `plain_english` and `scene_summary` used to be failed for
        *dropping* operators, when dropping them is what their own
        instruction asks for (the contract was scored the wrong way round);
      - `entity_fidelity` used to fire on `English`, `Put`, `While` and
        `ND's`, and to call `BABEL=TV` dropped when a gloss had split it
        across the naming seam;
      - the per-gloss columns of `glosses.tsv` used to count a check that
        did not apply as a pass, so the published row read "3 checks passed
        of 2 applicable" and named `residue_discipline` for a clause with
        no residue at all -- the denominator defect already fixed in
        `summary.json`, contradicted one table over;
      - the sample's tie-break is on the clause id as text (`1003` before
        `101`), not on #33's reading order.  It is disclosed as such rather
        than changed, because re-drawing the sample would move every prompt
        and would need a fresh live `--capture`.
  * **corpus tests** (guarded by `skipUnless`), which assert the scanner
    agrees with #17's token stream over every clause in the book, that
    the pinned build is byte-identical on rebuild, and that every number
    the report publishes re-derives from the artifacts.

`build()` is exercised into temporary directories only, never into
`analysis/glossing/`.
"""
import collections
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))
import glossing_pipeline as gp


HAVE_ARTIFACTS = all(p.exists() for p in (
    gp.CLAUSES, gp.TOKENS, gp.ENTITY, gp.GLOSS_TOKENS, gp.INTERLINEAR /
    "interlinear.tsv"))


def clause(cid="1", page=4, original="", region="body", gloss=""):
    """A #33/#62 clause row, the shape `pick_sample` and `check_gloss` read."""
    return {"clause_id": cid, "page": str(page), "region": region,
            "opener": "-", "start": "0", "end": "100",
            "original": original, "gloss": gloss}


def ents(*names):
    return [{"entity": n, "role": "persona", "n_occurrences": "3",
             "does": "", "done_to": ""} for n in names]


# ---------------------------------------------------------------------------
# the operator scanner: the check everything else rests on
# ---------------------------------------------------------------------------

class TestOperatorScanner(unittest.TestCase):
    """`operator_subsequence` must agree with #17, not with intuition."""

    def test_multi_character_operators_are_not_split(self):
        self.assertEqual(gp.operator_subsequence("a::b//c"), ["::", "//"])
        self.assertEqual(gp.operator_subsequence("x<<y>>z"), ["<<", ">>"])
        self.assertEqual(gp.operator_subsequence("p++q"), ["++"])

    def test_intra_word_hyphen_is_not_an_operator(self):
        """The bug that made every operator failure uninterpretable.

        `porno-nerve` and `boy-roid` carry a hyphen that #17 records as a
        part of the WORD token.  Counting it would fail a gloss for
        faithfully reproducing the book's own hyphenation.
        """
        for text in ("porno-nerve", "boy-roid", "vital-icon",
                     "acid-humanix", "cadavermechanism"):
            self.assertEqual(gp.operator_subsequence(text), [], text)

    def test_a_real_operator_beside_a_hyphen_is_still_found(self):
        """The hyphen rule must not swallow the neighbouring operator."""
        self.assertEqual(gp.operator_subsequence("suck=blood"), ["="])
        self.assertEqual(gp.operator_subsequence("vital-icon+our"), ["+"])

    def test_typographic_dash_is_not_an_operator(self):
        for text in ("screaming — eroded", "boy–roid", "a – b"):
            self.assertEqual(gp.operator_subsequence(text), [], text)

    def test_free_standing_ascii_dash_is_not_an_operator(self):
        """#17 classes a whitespace-bounded `-` as PUNCT, not OP."""
        self.assertEqual(gp.operator_subsequence("it - then"), [])

    def test_the_eleven_operator_surfaces_are_recognised(self):
        self.assertEqual(sorted(gp.OP_SURFACES),
                         sorted(["<<", ">>", "::", "//", "++", "=", "+",
                                 "<", ">", "/", "|"]))

    def test_order_is_preserved_not_counted(self):
        self.assertEqual(gp.operator_subsequence("::=+"), ["::", "=", "+"])


# ---------------------------------------------------------------------------
# the four checks, and the failures they must catch
# ---------------------------------------------------------------------------

class TestOperatorFidelity(unittest.TestCase):
    def test_exact_sequence_passes(self):
        ok, _msg, _got = gp.check_operator_fidelity(
            ["::", "="], "a::b=c", "literal_operator")
        self.assertTrue(ok)

    def test_dropped_operator_fails(self):
        ok, msg, _got = gp.check_operator_fidelity(
            ["::", "="], "a b c", "literal_operator")
        self.assertFalse(ok)
        self.assertIn("dropped 2", msg)

    def test_added_operator_fails(self):
        ok, msg, _got = gp.check_operator_fidelity(
            ["="], "a=b=c", "literal_operator")
        self.assertFalse(ok)
        self.assertIn("added 1", msg)

    def test_repeated_surface_is_reported_as_added_not_reordered(self):
        """The detail must describe the failure accurately.

        A gloss that repeats `soul/gram` inside its own annotation has
        *added* operators; reporting that as "reordered" -- which is what
        a set-membership diff yields -- makes the operator table
        unreadable.
        """
        ok, msg, _got = gp.check_operator_fidelity(
            ["/"], "the soul/gram [soul/gram: linker] of self",
            "literal_operator")
        self.assertFalse(ok)
        self.assertIn("added", msg)
        self.assertNotIn("reordered", msg)

    def test_reordering_is_reported_as_reordering(self):
        ok, msg, _got = gp.check_operator_fidelity(
            ["=", "::"], "a::b=c", "literal_operator")
        self.assertFalse(ok)
        self.assertIn("reordered", msg)

    def test_drop_strategies_must_have_an_empty_channel(self):
        """`plain_english` is told to verbalise the operators.

        Scoring it against the clause's own sequence failed a compliant
        gloss for *dropping* `=`, which is the bug this pins.
        """
        ok, _m, _g = gp.check_operator_fidelity(["="], "no operator here",
                                                "plain_english")
        self.assertTrue(ok, "dropping the operator is compliance here")
        ok2, msg2, _g2 = gp.check_operator_fidelity(
            ["="], "a=b", "plain_english")
        self.assertFalse(ok2, "keeping it violates the contract")
        self.assertIn("verbalise", msg2)

    def test_scene_summary_is_also_a_drop_strategy(self):
        ok, _m, _g = gp.check_operator_fidelity(["<<", ">>"], "the dog barked",
                                                "scene_summary")
        self.assertTrue(ok)


class TestEntityFidelity(unittest.TestCase):
    def test_name_held_passes(self):
        ok, _msg = gp.check_entity_fidelity(ents("NDRO"), "the NDRO device")
        self.assertTrue(ok)

    def test_dropped_name_fails(self):
        ok, msg = gp.check_entity_fidelity(ents("NDRO"), "the device sings")
        self.assertFalse(ok)
        self.assertIn("dropped NDRO", msg)

    def test_invented_name_fails(self):
        ok, msg = gp.check_entity_fidelity(
            ents("NDRO"), "NDRO meets Zebulon here")
        self.assertFalse(ok)
        self.assertIn("invented Zebulon", msg)

    def test_ordinary_capitalisation_is_not_an_invention(self):
        """The false positives that made the check cry wolf.

        `English` is licensed by the `plain_english` strategy's own
        instruction; `Put` and `While` are sentence starters; `ND's` is a
        possessive.  Each fired at least once in a real capture.  The
        glossary name is present in each case so that the *only* possible
        failure is the invented-name test.
        """
        for text, vocab in (
                ("Here the English seam opens on NDRO", ["english"]),
                ("Put out the NDRO channel", []),
                ("While NDRO burns", [])):
            ok, msg = gp.check_entity_fidelity(
                ents("NDRO"), text, allowed_vocab=vocab)
            self.assertTrue(ok, "%s -> %s" % (text, msg))

    def test_strategy_instruction_words_are_licensed_vocabulary(self):
        """`English` in a `plain_english` gloss comes from the strategy's own
        instruction; without the exemption the check would punish
        compliance."""
        vocab = gp.strategy_vocabulary("plain_english")
        self.assertIn("English", vocab)
        ok, msg = gp.check_entity_fidelity(
            ents("NDRO"), "the English of the NDRO device",
            allowed_vocab=vocab)
        self.assertTrue(ok, msg)

    def test_single_capital_initial_is_not_a_name(self):
        ok, _msg = gp.check_entity_fidelity(
            ents("NDRO"), "an I/O operation on NDRO")
        self.assertTrue(ok)

    def test_an_abbreviation_is_not_carried_and_not_invented(self):
        """`ND` is not `NDRO`, and no length rule separates that from `AD`
        for `ADAM` in a corpus holding AD, ADAM and ANTI-ADAM.

        So an abbreviation is scored as a *dropped* name -- the true and
        checkable fact -- and, being a prefix of a known name, is not
        additionally reported as invented.  Reporting both would charge one
        behaviour twice.
        """
        ok, msg = gp.check_entity_fidelity(ents("NDRO"), "ND resists the burn")
        self.assertFalse(ok)
        self.assertIn("dropped NDRO", msg)
        self.assertNotIn("invented", msg)

    def test_a_compound_of_licensed_words_is_not_an_invention(self):
        """`TV-persona` is the glossary entity plus a #30 role word.

        Reading it as an invented character would be the check inventing a
        character of its own.  A compound with an unlicensed part is still
        caught, so the exemption is not a blanket pass for hyphenated
        tokens.
        """
        ents4 = ents("TV", "BABEL")
        ok, msg = gp.check_entity_fidelity(
            ents4, "the screen-channeled TV-persona of BABEL")
        self.assertTrue(ok, msg)
        ok2, msg2 = gp.check_entity_fidelity(
            ents4, "the Zebulon-TV of BABEL")
        self.assertFalse(ok2, msg2)
        self.assertIn("invented Zebulon-TV", msg2)

    def test_role_vocabulary_is_read_from_the_glossary(self):
        """A new #30 role licenses itself; no code change needed."""
        words = gp.role_words()
        self.assertIsInstance(words, frozenset)
        self.assertTrue(words)
        if HAVE_ARTIFACTS:
            roles = {r for row in gp.load_glossary().values()
                     for r in (row.get("role") or "").split() if len(r) >= 3}
            self.assertTrue(roles.issubset(words),
                            "a #30 role is not licensed vocabulary")

    def test_a_genuinely_new_name_is_still_an_invention(self):
        ok, msg = gp.check_entity_fidelity(ents("NDRO"), "NDRO meets Zebulon")
        self.assertFalse(ok)
        self.assertIn("invented Zebulon", msg)

    def test_operator_bearing_entity_is_matched_across_its_seam(self):
        """`BABEL=TV` split across a naming seam is still `BABEL=TV`."""
        ok = gp.entity_present(
            "BABEL=TV", "Put out BABEL[persona]=[naming seam]TV[screen]")
        self.assertTrue(ok)

    def test_operator_bearing_entity_is_not_satisfied_by_a_hyphen(self):
        """`BABEL-TV` is a different #30 entity."""
        self.assertFalse(gp.entity_present("BABEL=TV", "the BABEL-TV unit"))

    def test_operator_bearing_entity_is_not_satisfied_by_bare_parts(self):
        """Two unrelated words are not the joined construct."""
        self.assertFalse(gp.entity_present("BABEL=TV", "BABEL and TV"))

    def test_verbatim_entity_name_passes(self):
        self.assertTrue(gp.entity_present("BABEL=TV", "the BABEL=TV node"))


class TestResidueDiscipline(unittest.TestCase):
    def test_no_residue_passes(self):
        ok, _msg = gp.check_residue_discipline([], "anything at all")
        self.assertTrue(ok)

    def test_kept_marked_surface_passes(self):
        ok, _msg = gp.check_residue_discipline(
            ["omanko"], "her omanko [unresolved] hydromachine")
        self.assertTrue(ok)

    def test_smoothed_away_surface_fails(self):
        """The L9.8 failure this check exists to catch.

        A gloss that turns an unexplained surface into an ordinary word
        asserts a reading the workspace has no evidence for.
        """
        ok, msg = gp.check_residue_discipline(
            ["omanko"], "her wonderful flowering machine")
        self.assertFalse(ok)
        self.assertIn("omanko", msg)

    def test_compound_containing_the_surface_counts_as_kept(self):
        ok, _msg = gp.check_residue_discipline(
            ["omanko"], "the omanko-machine")
        self.assertTrue(ok)


class TestFramePreservation(unittest.TestCase):
    def test_frame_held_passes(self):
        ok, _msg = gp.check_frame_preservation(["<<", ">>"], "<<the label>>x")
        self.assertTrue(ok)

    def test_dropped_frame_fails(self):
        ok, msg = gp.check_frame_preservation(["<<", ">>"], "the label x")
        self.assertFalse(ok)
        self.assertIn("1<<", msg)

    def test_clause_without_a_frame_passes(self):
        ok, _msg = gp.check_frame_preservation(["::"], "a::b")
        self.assertTrue(ok)

    def test_drop_strategies_are_not_held_to_a_character_match(self):
        """`scene_summary` is told to drop the frame; holding it to one
        would fail a compliant gloss."""
        self.assertFalse(gp.want_frame(["<<"], "scene_summary"))
        self.assertFalse(gp.want_frame(["<<"], "plain_english"))
        self.assertTrue(gp.want_frame(["<<"], "literal_operator"))


class TestCheckGloss(unittest.TestCase):
    def test_verdict_is_fail_if_any_applicable_check_fails(self):
        c = clause(original="a::b")
        verdict, rows = gp.check_gloss(c, ["::"], [], [], "literal_operator",
                                       "a b")
        self.assertEqual(verdict, "fail")
        self.assertEqual(len(rows), gp.CHECKS_PER_GLOSS)

    def test_verdict_is_pass_only_when_applicable_checks_pass(self):
        c = clause(original="a::b")
        verdict, _rows = gp.check_gloss(c, ["::"], [], [], "literal_operator",
                                        "a::b")
        self.assertEqual(verdict, "pass")

    def test_a_non_applicable_check_cannot_fail_the_verdict(self):
        """The `scene_summary` frame case: not applicable, so not fatal."""
        c = clause(original="<<x")
        verdict, rows = gp.check_gloss(c, ["<<"], [], [], "scene_summary",
                                       "the dog barks")
        fr = next(r for r in rows if r["check"] == "frame_preservation")
        self.assertFalse(fr["applies"])
        self.assertEqual(verdict, "pass")

    def test_every_check_row_carries_a_contract(self):
        """A verdict a reader cannot interpret is a verdict they ignore."""
        c = clause(original="a::b")
        _verdict, rows = gp.check_gloss(c, ["::"], ents("NDRO"), ["omanko"],
                                        "literal_operator", "NDRO a::b omanko")
        for r in rows:
            self.assertTrue(r["contract"], r["check"])
            self.assertTrue(r["detail"] or r["pass"], r["check"])

    def test_checks_never_call_the_backend(self):
        """Circularity guard: the model writes, the checks judge."""
        c = clause(original="a::b")
        with mock.patch.object(gp, "run_live",
                               side_effect=AssertionError("check called "
                                                           "the model")):
            gp.check_gloss(c, ["::"], [], [], "literal_operator", "a::b")

    def test_an_unknown_strategy_is_a_clear_error(self):
        """A typo'd strategy id must not surface as a bare KeyError three
        frames later, inside whichever check happened to run first."""
        with self.assertRaises(gp.GlossingError) as ctx:
            gp.check_gloss(clause(original="a::b"), ["::"], [], [],
                           "literal_operator_", "a::b")
        self.assertIn("literal_operator_", str(ctx.exception))
        self.assertIn("plain_english", str(ctx.exception))

    def test_a_sample_clause_with_no_33_row_is_refused(self):
        """The sample must come from #33; an id with no clause row is a
        sample drawn from something else, and is not glossed from offsets."""
        sample = {"7": "plain_control"}
        with self.assertRaises(gp.GlossingError) as ctx:
            gp.resolve_glosses(sample, {}, gp.load_glossary(), {}, {},
                               gp.STRATEGY_IDS, clauses={})
        self.assertIn("7", str(ctx.exception))


# ---------------------------------------------------------------------------
# strategies and prompts
# ---------------------------------------------------------------------------

class TestStrategies(unittest.TestCase):
    def test_the_five_strategies_the_task_names_are_present(self):
        self.assertEqual(
            sorted(gp.STRATEGY_IDS),
            sorted(["literal_operator", "register_paraphrase",
                    "scene_summary", "plain_english", "entity_consistent"]))

    def test_every_strategy_has_a_non_empty_instruction_and_contract(self):
        for s in gp.STRATEGIES:
            self.assertTrue(s["instruction"].strip(), s["id"])
            self.assertTrue(s["contract"].strip(), s["id"])
            self.assertIn(s["id"], gp.STRATEGY_OWES)

    def test_the_owing_table_covers_every_strategy(self):
        self.assertEqual(sorted(gp.STRATEGY_OWES), sorted(gp.STRATEGY_IDS))

    def test_drop_strategies_are_declared_consistently(self):
        """The operator contract, the owing table and the frame rule must
        not disagree -- that disagreement is what the two earlier bugs were.
        """
        for sid in gp.OPERATOR_DROP:
            self.assertIn("instruction", gp.STRATEGY_OWES[sid]["operators"],
                          sid)
            # a drop strategy also does not owe the frame's characters
            self.assertIn("instruction", gp.STRATEGY_OWES[sid]["frame"], sid)
        for sid in set(gp.STRATEGY_IDS) - gp.OPERATOR_DROP:
            self.assertEqual(gp.STRATEGY_OWES[sid]["operators"],
                             "verbatim, in order", sid)
            self.assertEqual(gp.STRATEGY_OWES[sid]["frame"], "kept", sid)


class TestPromptText(unittest.TestCase):
    def clause_with_ops(self):
        return clause(original="the soul/gram of self crushes to NDRO::x")

    def test_prompt_carries_the_clause_verbatim(self):
        c = self.clause_with_ops()
        text = gp.prompt_text(c, gp.operator_constraints(["/", "::"]),
                              ents("NDRO"), [], [], "literal_operator")
        self.assertIn(c["original"], text)

    def test_prompt_shows_only_the_operators_the_clause_contains(self):
        """If the OPERATORS block listed every operator, a "preserved"
        channel would prove nothing -- the model could be copying a list
        rather than the clause.  Checked against the OPERATORS block
        specifically, since the instruction text names the operator
        alphabet as a rule ("copy these characters exactly"), which is a
        different thing: it tells the model the alphabet, not the clause's
        contents.
        """
        c = self.clause_with_ops()
        text = gp.prompt_text(c, gp.operator_constraints(["/", "::"]),
                              [], [], [], "literal_operator")
        block = text.split("OPERATORS IN THIS CLAUSE")[1].split("\n\n")[0]
        self.assertIn("/", block)
        self.assertIn("::", block)
        for absent in ("++", "|", ">>", "<<"):
            self.assertNotIn(absent, block,
                             "%s is not in this clause" % absent)

    def test_prompt_carries_the_entity_role_and_verbs(self):
        c = self.clause_with_ops()
        e = [{"entity": "NDRO", "role": "persona", "n_occurrences": "23",
              "does": "is(3)", "done_to": "black(3)"}]
        text = gp.prompt_text(c, {}, e, [], [], "entity_consistent")
        self.assertIn("NDRO", text)
        self.assertIn("persona", text)
        self.assertIn("black(3)", text)

    def test_prompt_lists_unresolved_surfaces_and_ocr_artefacts(self):
        c = clause(original="her omanko href machine")
        text = gp.prompt_text(c, {}, [], ["href"], ["omanko"],
                              "literal_operator")
        self.assertIn("omanko", text)
        self.assertIn("href", text)

    def test_prompt_is_deterministic(self):
        c = self.clause_with_ops()
        ops = gp.operator_constraints(["/", "::"])
        a = gp.prompt_text(c, ops, ents("NDRO"), [], [], "literal_operator")
        b = gp.prompt_text(c, ops, ents("NDRO"), [], [], "literal_operator")
        self.assertEqual(a, b)
        self.assertEqual(gp.prompt_sha256(a), gp.prompt_sha256(b))

    def test_prompt_differs_by_strategy(self):
        c = self.clause_with_ops()
        ops = gp.operator_constraints(["/"])
        hashes = {gp.prompt_sha256(gp.prompt_text(c, ops, [], [], [], s))
                  for s in gp.STRATEGY_IDS}
        self.assertEqual(len(hashes), len(gp.STRATEGY_IDS))

    def test_the_standing_block_is_identical_across_strategies(self):
        c = self.clause_with_ops()
        ops = gp.operator_constraints(["/"])
        tails = {gp.prompt_text(c, ops, [], [], [], s).split("\n\n")[-1]
                 for s in gp.STRATEGY_IDS}
        self.assertEqual(len(tails), 1,
                         "the honesty clause must not vary by strategy")

    def test_the_standing_block_forbids_inventing_names_and_resolving(self):
        c = self.clause_with_ops()
        text = gp.prompt_text(c, {}, [], [], [], "plain_english")
        self.assertIn("Never invent a name", text)
        self.assertIn("Do not resolve what is unresolved", text)

    def test_an_oversized_prompt_is_an_error_not_a_truncation(self):
        c = clause(original="x" * gp.MAX_PROMPT_CHARS)
        with self.assertRaises(gp.GlossingError):
            gp.prompt_text(c, {}, [], [], [], "literal_operator")


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------

class TestAgreement(unittest.TestCase):
    def test_identical_glosses_agree_completely(self):
        g = {"a": "the dog barks", "b": "the dog barks", "c": "the dog barks"}
        self.assertEqual(gp.agreement("1", g)["mean_jaccard"], 1.0)

    def test_disjoint_glosses_agree_not_at_all(self):
        g = {"a": "alpha beta", "b": "gamma delta", "c": "epsilon zeta"}
        self.assertEqual(gp.agreement("1", g)["mean_jaccard"], 0.0)

    def test_operators_are_excluded_from_agreement(self):
        """Two strategies that both emit `::` agree trivially, and that is
        not the agreement the committee cares about."""
        g = {"a": "the dog::barks", "b": "the dog::barks",
             "c": "the dog::barks"}
        self.assertEqual(gp.agreement("1", g)["mean_jaccard"], 1.0)
        self.assertNotIn("::", gp.gloss_tokens("the dog::barks"))

    def test_reports_the_least_agreeing_pair(self):
        g = {"a": "alpha beta gamma", "b": "alpha beta gamma",
             "c": "delta epsilon zeta"}
        row = gp.agreement("1", g)
        self.assertEqual(row["least_agreeing_pair"], "a|c")

    def test_agreement_is_symmetric_in_its_arguments(self):
        g = {"a": "alpha beta", "b": "gamma delta"}
        self.assertEqual(gp.agreement("1", g)["mean_jaccard"],
                         gp.agreement("1", dict(reversed(list(
                             g.items()))))["mean_jaccard"])


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

class TestSampling(unittest.TestCase):
    def test_residue_rank_prefers_an_unread_word_over_markup(self):
        """`omanko` (a true_coinage with no reading) must outrank `href`
        (markup that is not book text); only the first can fail
        `residue_discipline` for the reason that check exists."""
        self.assertLess(gp.residue_rank([("omanko", "unresolved")]),
                        gp.residue_rank([("href", "junk_19")]))

    def test_clause_stratum_is_assigned_by_the_published_rule(self):
        self.assertEqual(gp.clause_stratum(["::"], set(), [("x", "u")]),
                         "residue")
        self.assertEqual(gp.clause_stratum(["<<", ">>"], set(), []),
                         "quoting_frame")
        self.assertEqual(gp.clause_stratum(["=", "::", "/", "+"], set(), []),
                         "operator_dense")
        self.assertEqual(gp.clause_stratum(["::"], {"adam"}, []),
                         "entity_named")
        self.assertEqual(gp.clause_stratum(["::"], set(), []),
                         "plain_control")

    def test_sample_is_deterministic(self):
        clauses = {"1": clause("1", original="a::b c"),
                   "2": clause("2", original="x<<y>>z"),
                   "3": clause("3", original="adam::c")}
        opmap = {"1": ["::"], "2": ["<<", ">>"], "3": ["::"]}
        glossary = {"adam": {"entity": "ADAM", "role": "persona",
                             "n_occurrences": "3", "does_verbs": "",
                             "done_to_verbs": ""}}
        a = gp.pick_sample(clauses, opmap, glossary, {})
        b = gp.pick_sample(clauses, opmap, glossary, {})
        self.assertEqual(a, b)
        self.assertEqual(a["2"], "quoting_frame")
        self.assertEqual(a["3"], "entity_named")
        self.assertEqual(a["1"], "plain_control")

    def test_a_clause_with_no_operators_is_never_sampled(self):
        clauses = {"1": clause("1", original="plain prose only")}
        self.assertEqual(gp.pick_sample(clauses, {"1": []}, {}, {}), {})

    def test_a_clause_with_no_text_is_never_sampled(self):
        clauses = {"1": clause("1", original="")}
        self.assertEqual(gp.pick_sample(clauses, {"1": ["::"]}, {}, {}), {})

    def test_every_sample_stratum_is_a_declared_stratum(self):
        clauses = {"1": clause("1", original="a::b")}
        for st in gp.pick_sample(clauses, {"1": ["::"]}, {}, {}).values():
            self.assertIn(st, gp.STRATA)

    def test_the_tie_break_is_on_the_clause_id_as_text(self):
        """The sampling rule is published, and this is what it actually is.

        The tie-break compares clause ids as text, so `100` is drawn before
        `99` -- it is *not* #33's reading order, and the report says so
        rather than claiming the numerically earliest clause.  Pinned here so
        that changing the tie-break (which would move every prompt, and so
        would need a fresh `--capture`) is a deliberate, visible edit.
        """
        clauses = {"99": clause("99", original="a::b c d"),
                   "100": clause("100", original="a::b c d")}
        sample = gp.pick_sample(clauses, {"99": ["::"], "100": ["::"]}, {}, {})
        self.assertEqual(sample, {"100": "plain_control"})


# ---------------------------------------------------------------------------
# pins
# ---------------------------------------------------------------------------

class TestPins(unittest.TestCase):
    def test_pin_key_is_stable(self):
        self.assertEqual(gp.pin_key("2789", "literal_operator"),
                         "2789|literal_operator")

    def test_every_pin_carries_a_prompt_hash_and_a_backend(self):
        """A pin without the hash of the prompt it answers is not checkable,
        and a build cannot detect that the constraint assembly moved."""
        for key, pin in gp.PINS.items():
            self.assertIn("prompt_sha256", pin, key)
            self.assertIn("backend", pin, key)
            self.assertIn("captured", pin, key)
            self.assertIn("response", pin, key)
            self.assertEqual(len(pin["prompt_sha256"]), 64, key)
            self.assertTrue(pin["response"].strip(), key)

    def test_pins_cover_every_sample_and_strategy(self):
        if not HAVE_ARTIFACTS:
            self.skipTest("frozen artifacts absent")
        sample = gp.pick_sample(gp.load_clauses(),
                                gp.load_operator_token_map(),
                                gp.load_glossary(), gp.load_clause_residue())
        for cid in sample:
            for sid in gp.STRATEGY_IDS:
                self.assertIn(gp.pin_key(cid, sid), gp.PINS)

    def test_a_missing_pin_is_a_hard_error(self):
        """A five-strategy table with a hole in it measures the hole, so the
        build must refuse rather than gloss whatever happens to be pinned."""
        if not HAVE_ARTIFACTS:
            self.skipTest("frozen artifacts absent")
        clauses = gp.load_clauses()
        sample = gp.pick_sample(clauses, gp.load_operator_token_map(),
                                gp.load_glossary(), gp.load_clause_residue())
        with mock.patch.dict(gp.PINS, {}, clear=True):
            with self.assertRaises(gp.GlossingError):
                gp.resolve_glosses(
                    sample, gp.load_operator_token_map(), gp.load_glossary(),
                    gp.load_clause_residue(),
                    gp.load_clause_ocr(gp.load_ocr_class()),
                    gp.STRATEGY_IDS, clauses=clauses)

    def test_a_pin_recorded_against_another_prompt_is_refused(self):
        """Stale-pin guard: if the constraints change, the recorded response
        no longer answers the prompt being asked."""
        if not HAVE_ARTIFACTS:
            self.skipTest("frozen artifacts absent")
        clauses = gp.load_clauses()
        opmap = gp.load_operator_token_map()
        glossary = gp.load_glossary()
        residue = gp.load_clause_residue()
        ocr = gp.load_clause_ocr(gp.load_ocr_class())
        sample = gp.pick_sample(clauses, opmap, glossary, residue)
        cid = sorted(sample, key=int)[0]
        stale = {gp.pin_key(cid, gp.STRATEGY_IDS[0]): {
            "prompt_sha256": "0" * 64, "backend": "x", "captured": "y",
            "response": "z"}}
        with mock.patch.dict(gp.PINS, stale, clear=True):
            with self.assertRaises(gp.GlossingError) as ctx:
                gp.resolve_glosses(sample, opmap, glossary, residue, ocr,
                                   [gp.STRATEGY_IDS[0]], clauses=clauses)
        self.assertIn("stale", str(ctx.exception))


class TestBackendIsolation(unittest.TestCase):
    def test_the_backend_is_a_subprocess_without_a_shell(self):
        """Clause text must never reach a shell."""
        cmd = ["opencode", "run", "--model", "m", "p"]
        with mock.patch.object(subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ok\n",
                                         stderr="")
            self.assertEqual(gp.run_live("p", "m"), "ok")
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        self.assertEqual(run.call_args.args[0], cmd)

    def test_a_missing_cli_is_a_clear_error(self):
        with mock.patch.object(subprocess, "run",
                               side_effect=FileNotFoundError("nope")):
            with self.assertRaises(gp.GlossingError) as ctx:
                gp.run_live("p", "m")
        self.assertIn("--live needs it", str(ctx.exception))

    def test_a_nonzero_exit_is_an_error(self):
        with mock.patch.object(subprocess, "run",
                               return_value=mock.Mock(returncode=3,
                                                     stdout="", stderr="bad")):
            with self.assertRaises(gp.GlossingError):
                gp.run_live("p", "m")

    def test_a_timeout_is_an_error(self):
        with mock.patch.object(subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("c", 5)):
            with self.assertRaises(gp.GlossingError) as ctx:
                gp.run_live("p", "m", timeout=5)
        self.assertIn("timed out", str(ctx.exception))


# ---------------------------------------------------------------------------
# corpus tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts absent")
class TestAgainstTheCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clauses = gp.load_clauses()
        cls.opmap = gp.load_operator_token_map()

    def test_the_scanner_agrees_with_17_on_every_clause(self):
        """The load-bearing property: `operator_subsequence` must reproduce
        #17's operator stream, or every operator verdict is measuring the
        scanner rather than the model."""
        mismatch = []
        for cid, row in self.clauses.items():
            if not row["original"]:
                continue
            if gp.operator_subsequence(row["original"]) != \
                    self.opmap.get(cid, []):
                mismatch.append(cid)
        self.assertEqual(mismatch, [],
                         "%d clauses disagree with #17, e.g. %s"
                         % (len(mismatch), mismatch[:5]))

    def test_operator_counts_match_17s_totals(self):
        """Cross-check on the totals, not just the per-clause agreement.

        #17's stream carries 11,000 operator tokens over the whole book;
        5,227 of them fall inside a #33 clause, and the rest are the
        published remainder outside the segmentation (#62 reconciles the
        same two numbers).  Both are stated so a drift in either is caught.
        """
        in_clauses = sum(len(v) for v in self.opmap.values())
        self.assertEqual(in_clauses, 5227,
                         "#17's in-clause operator count moved; the "
                         "sampler's strata are defined against it")
        with gp.TOKENS.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            kind_ix = header.index("kind")
            total = sum(1 for line in fh
                        if line.split("\t")[kind_ix] == "OP")
        self.assertEqual(total, 11000,
                         "#17's total operator token count moved")
        self.assertLess(in_clauses, total)

    def test_clauses_carry_their_text_from_62(self):
        """Every #33 clause is joined to #62's text except the one empty
        header clause, which is legitimately empty and is listed here so a
        *new* empty clause is a change rather than a silent drop."""
        missing = [cid for cid, r in self.clauses.items()
                   if not r["original"]]
        self.assertEqual(missing, ["5612"],
                         "#33 rows with no #62 text: %s (a new empty clause "
                         "means the join moved)" % missing)

    def test_the_sample_has_one_clause_per_declared_stratum(self):
        sample = gp.pick_sample(self.clauses, self.opmap, gp.load_glossary(),
                                gp.load_clause_residue())
        self.assertEqual(len(sample), len(gp.STRATA))
        self.assertEqual(sorted(sample.values()), sorted(gp.STRATA))

    def test_the_residue_stratum_carries_a_surface(self):
        """Otherwise `residue_discipline` is vacuous on this sample."""
        sample = gp.pick_sample(self.clauses, self.opmap, gp.load_glossary(),
                                gp.load_clause_residue())
        residue = gp.load_clause_residue()
        rid = [c for c, s in sample.items() if s == "residue"]
        self.assertTrue(rid)
        self.assertTrue(gp.residue_surfaces(residue[rid[0]]))

    def test_the_operator_dense_stratum_actually_has_four_operators(self):
        sample = gp.pick_sample(self.clauses, self.opmap, gp.load_glossary(),
                                gp.load_clause_residue())
        dense = [c for c, s in sample.items() if s == "operator_dense"]
        self.assertTrue(dense)
        self.assertGreaterEqual(len(self.opmap[dense[0]]), 4)


@unittest.skipUnless(HAVE_ARTIFACTS, "frozen artifacts absent")
class TestBuild(unittest.TestCase):
    def test_the_pinned_build_is_byte_identical_on_rebuild(self):
        """Determinism is the reason the pins exist at all."""
        a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        gp.build(out=a, quiet=True)
        gp.build(out=b, quiet=True)
        for name in ("glosses.tsv", "checks.tsv", "agreement.tsv",
                     "sample.tsv", "prompts.tsv", "diagnostics.tsv"):
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(),
                             name)

    def test_the_summary_reproduces_identically_apart_from_the_stamp(self):
        a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        sa = gp.build(out=a, quiet=True)
        sb = gp.build(out=b, quiet=True)
        sa.pop("generated"), sb.pop("generated")
        self.assertEqual(sa, sb)

    def test_every_published_number_re_derives_from_the_artifacts(self):
        out = Path(tempfile.mkdtemp())
        summary = gp.build(out=out, quiet=True)
        glosses = gp.cs.read_tsv(out / "glosses.tsv")
        checks = gp.cs.read_tsv(out / "checks.tsv")
        agree = gp.cs.read_tsv(out / "agreement.tsv")
        self.assertEqual(summary["glosses"], len(glosses))
        self.assertEqual(summary["checks_total"], len(checks))
        self.assertEqual(summary["verdicts"]["pass"],
                         sum(1 for r in glosses if r["verdict"] == "pass"))
        self.assertEqual(summary["verdicts"]["fail"],
                         sum(1 for r in glosses if r["verdict"] == "fail"))
        # The check aggregates are counted over the rows the check applies
        # to, so they re-derive from `checks.tsv` with that filter on.
        applicable = [r for r in checks if r["applies"] == "True"]
        self.assertEqual(summary["checks_applicable"], len(applicable))
        self.assertEqual(summary["checks_passed"],
                         sum(1 for r in applicable if r["pass"] == "True"))
        self.assertLessEqual(summary["checks_passed"],
                             sum(1 for r in checks if r["pass"] == "True"))
        self.assertEqual(summary["checks_total"], len(checks))
        self.assertEqual(summary["clauses_sampled"], len(agree))
        self.assertEqual(summary["checks_per_gloss"], gp.CHECKS_PER_GLOSS)
        self.assertEqual(summary["mean_clause_agreement"],
                         round(sum(float(a["mean_jaccard"]) for a in agree)
                               / len(agree), 4))

    def test_a_check_is_never_counted_as_passing_a_gloss_it_did_not_judge(self):
        """The denominators: `residue_discipline` applies to the glosses whose
        clause carries an unresolved surface, not to all of them.

        An earlier revision counted every published row, so the summary read
        `residue_discipline: 1.0` over 25 glosses when only 5 carried a
        residue surface and the check judged 5 -- a vacuous 100%, which is the
        residue-dropping anti-pattern in aggregate form, and it disagreed with
        the report's own table.  Both artifacts must now say the same thing.
        """
        out = Path(tempfile.mkdtemp())
        summary = gp.build(out=out, quiet=True)
        checks = gp.cs.read_tsv(out / "checks.tsv")
        report = (out / "glossing_report.md").read_text(encoding="utf-8")
        for check, counts in summary["check_counts"].items():
            rows = [r for r in checks if r["check"] == check]
            ap = [r for r in rows if r["applies"] == "True"]
            self.assertEqual(counts["rows"], len(rows), check)
            self.assertEqual(counts["applicable"], len(ap), check)
            self.assertEqual(counts["passed"],
                             sum(1 for r in ap if r["pass"] == "True"),
                             check)
            if ap:
                self.assertEqual(
                    summary["check_pass_rate"][check],
                    round(counts["passed"] / len(ap), 4), check)
                # the report's table publishes the same denominator
                self.assertIn("| `%s` | %d | %d | %d |" % (
                    check, counts["rows"], counts["applicable"],
                    counts["passed"]), report)
        # and the demo case: residue runs on the residue stratum only
        residue = summary["check_counts"]["residue_discipline"]
        self.assertLess(residue["applicable"], summary["glosses"])
        self.assertEqual(residue["applicable"], residue["passed"])

    def test_the_per_gloss_counts_never_exceed_the_applicable_rows(self):
        """The same denominator rule, per gloss instead of per summary.

        `n_passed` and `checks_passed` used to be counted over every row the
        gloss published, so 20 of the 25 rows read "3 checks passed of 2
        applicable" and listed `residue_discipline` as passed for clauses
        carrying no residue at all -- the aggregate fix above, contradicted
        one table over.  A check that does not apply cannot pass, and a row
        that says otherwise is the same anti-pattern in miniature.
        """
        out = Path(tempfile.mkdtemp())
        gp.build(out=out, quiet=True)
        checks = gp.cs.read_tsv(out / "checks.tsv")
        applicable = collections.defaultdict(list)
        for c in checks:
            if c["applies"] == "True":
                applicable[(c["clause_id"], c["strategy"])].append(c["check"])
        for g in gp.cs.read_tsv(out / "glosses.tsv"):
            key = (g["clause_id"], g["strategy"])
            self.assertLessEqual(int(g["n_passed"]), int(g["n_applicable"]),
                                 key)
            self.assertLessEqual(int(g["n_applicable"]), int(g["n_checks"]),
                                 key)
            listed = [n for n in g["checks_passed"].split(",") if n]
            self.assertEqual(listed, [n for n in applicable[key]
                                      if n in listed], key)
            # a pass is exactly "broke no rule it was held to", so the two
            # published columns must agree with each other
            self.assertEqual(int(g["n_passed"]) == int(g["n_applicable"]),
                             g["verdict"] == "pass", key)
        self.assertTrue(any(int(g["n_passed"]) < int(g["n_applicable"])
                            for g in gp.cs.read_tsv(out / "glosses.tsv")),
                        "no gloss in this sample exercises a check that does "
                        "not apply, so this test would pass vacuously")

    def test_every_gloss_carries_its_strategy_backend_and_prompt_hash(self):
        """Provenance: a gloss with no model recorded cannot be audited."""
        out = Path(tempfile.mkdtemp())
        gp.build(out=out, quiet=True)
        for r in gp.cs.read_tsv(out / "glosses.tsv"):
            self.assertIn(r["strategy"], gp.STRATEGY_IDS)
            self.assertTrue(r["backend"], r["clause_id"])
            self.assertEqual(len(r["prompt_sha256"]), 64)
            self.assertEqual(len(r["response_sha256"]), 64)
            self.assertTrue(r["captured"], r["clause_id"])

    def test_no_gloss_is_dropped_from_the_published_table(self):
        """Residue-dropping in the pipeline itself would be self-defeating."""
        out = Path(tempfile.mkdtemp())
        summary = gp.build(out=out, quiet=True)
        glosses = gp.cs.read_tsv(out / "glosses.tsv")
        self.assertEqual(len(glosses),
                         summary["clauses_sampled"] * summary["strategies"])
        for r in glosses:
            self.assertTrue(r["gloss"].strip(),
                            "%s/%s has an empty gloss" % (r["clause_id"],
                                                          r["strategy"]))

    def test_the_verdict_table_and_the_gloss_table_agree(self):
        out = Path(tempfile.mkdtemp())
        summary = gp.build(out=out, quiet=True)
        glosses = gp.cs.read_tsv(out / "glosses.tsv")
        for sid, d in summary["per_strategy_verdict"].items():
            self.assertEqual(
                sum(1 for r in glosses if r["strategy"] == sid
                    and r["verdict"] == "pass"), d["pass"], sid)
            self.assertEqual(
                sum(1 for r in glosses if r["strategy"] == sid
                    and r["verdict"] == "fail"), d["fail"], sid)

    def test_the_report_states_the_capture_and_the_sample_size(self):
        out = Path(tempfile.mkdtemp())
        summary = gp.build(out=out, quiet=True)
        text = (out / "glossing_report.md").read_text(encoding="utf-8")
        self.assertIn(gp.CAPTURE, text)
        self.assertIn("not a gloss of the book", text)
        self.assertIn(str(summary["clauses_sampled"]), text)
        for sid in gp.STRATEGY_IDS:
            self.assertIn(sid, text)

    def test_the_report_says_the_checks_judge_and_the_model_writes(self):
        """The circularity guard has to be stated, not just implemented."""
        out = Path(tempfile.mkdtemp())
        gp.build(out=out, quiet=True)
        text = (out / "glossing_report.md").read_text(encoding="utf-8")
        self.assertIn("Circularity", text)

    def test_the_prompts_are_published_for_every_gloss(self):
        out = Path(tempfile.mkdtemp())
        gp.build(out=out, quiet=True)
        prompts = gp.cs.read_tsv(out / "prompts.tsv")
        glosses = gp.cs.read_tsv(out / "glosses.tsv")
        self.assertEqual(len(prompts), len(glosses))
        keys = {(p["clause_id"], p["strategy"]) for p in prompts}
        self.assertEqual(keys, {(g["clause_id"], g["strategy"])
                                for g in glosses})

    def test_published_prompt_hashes_match_the_pins(self):
        """The build must be answering the prompt the pin recorded."""
        out = Path(tempfile.mkdtemp())
        gp.build(out=out, quiet=True)
        for p in gp.cs.read_tsv(out / "prompts.tsv"):
            pin = gp.PINS[gp.pin_key(p["clause_id"], p["strategy"])]
            self.assertEqual(p["prompt_sha256"], pin["prompt_sha256"],
                             p["clause_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)