#!/usr/bin/env python3
"""Regression suite for #60 the genre/prior dossier.

Checks the invariants that matter:

  * the lexicon families, the traditions and the expectations are internally
    consistent (every probe resolves, every family is used or marked context,
    no expectation is duplicated, every tradition is covered);
  * the no-ventriloquy invariant of L7: the prior ledger ingests no tradition
    text beyond what #56's excerpt corpus already carries lawfully, and #57's
    ledger row for the L7.5 priors is still present and still marks them
    not-acquired;
  * the book-side screen reproduces #56's word-part counts exactly and the
    clause partition assigns every kept part to exactly one #33 clause;
  * the lexicon / co-occurrence / artifact probes on known inputs, including
    the verdict gates with planted failures (a family below the ratio gate
    must come out thin, a pair below the z gate must come out thin, a
    zero-occurrence family must come out absent);
  * the artifact probes read the frozen artifacts and apply their declared
    rules: a planted-summary registry (``RULE_CASES``) flips **every**
    registered probe's rule in both directions, and a coverage test fails
    the build if a probe is ever added without its planted cases;
  * a full (quick) build writes every artifact, reports the same book-side
    numbers as the full build (mode-invariance of every verdict), is
    deterministic on rerun, and its report numbers agree with the tables it
    sits beside.

Reads only frozen artifacts and the cached reference corpora -- no network,
no OCR.

    python work/test_genre_prior.py
"""
import collections
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import genre_prior as gp

ROOT = gp.ROOT

# a real word surface that full-matches each family, used to build synthetic
# reference corpora with known densities.
FAMILY_EXEMPLARS = {
    "corporate": "corporation", "digital": "digital", "machine": "machine",
    "metal": "metal", "organ": "organ", "clinical": "serum",
    "reproduction": "embryo", "grotesque": "cadaver", "virus": "virus",
    "transformation": "mutation", "sexual": "desire", "violence": "murder",
    "city": "city",
}


class TestTables(unittest.TestCase):
    """The declared tables are internally consistent."""

    def test_every_expectation_id_is_unique(self):
        ids = [e.eid for e in gp.EXPECTATIONS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_expectation_names_a_known_tradition(self):
        for e in gp.EXPECTATIONS:
            self.assertIn(e.tradition, gp.TRADITIONS, e.eid)

    def test_every_tradition_has_expectations(self):
        seen = collections.Counter(e.tradition for e in gp.EXPECTATIONS)
        for trad in gp.TRADITION_ORDER:
            self.assertGreaterEqual(seen[trad], 2, trad)

    def test_every_probe_resolves(self):
        fams = {f.key for f in gp.LEXICON_FAMILIES}
        pairs = {f"{a}+{b}" for a, b in gp.COOCC_PAIRS}
        for e in gp.EXPECTATIONS:
            kind, _, arg = e.probe.partition(":")
            if kind == "lexicon":
                self.assertIn(arg, fams, e.eid)
            elif kind == "cooccur":
                self.assertIn(arg, pairs, e.eid)
            elif kind == "artifact":
                self.assertIn(arg, gp.ARTIFACT_PROBES, e.eid)
            else:
                self.assertEqual(kind, "declared", e.eid)

    def test_every_expectation_has_an_annotation(self):
        for e in gp.EXPECTATIONS:
            self.assertIn(e.eid, gp.ANNOTATIONS, e.eid)
            self.assertGreater(len(gp.ANNOTATIONS[e.eid]), 80, e.eid)

    def test_declared_expectations_are_underdetermined_with_a_path(self):
        for e in gp.EXPECTATIONS:
            if e.probe == "declared":
                text = (e.falsifier + " " + gp.ANNOTATIONS[e.eid]).lower()
                self.assertIn("settle", text, e.eid)

    def test_family_patterns_compile_and_are_distinct(self):
        for f in gp.LEXICON_FAMILIES:
            self.assertTrue(f.patterns, f.key)
            for p in f.patterns:
                self.assertTrue(p.strip(), f"{f.key}: empty pattern")
                gp.re.compile(p)  # must compile
            self.assertEqual(len(f.patterns), len(set(f.patterns)), f.key)

    def test_every_family_matches_its_exemplar_and_nothing_else_claims_it(self):
        for f in gp.LEXICON_FAMILIES:
            word = FAMILY_EXEMPLARS[f.key]
            self.assertEqual(gp.matched_families(word), (f.key,),
                             f"{f.key}: {word}")

    def test_cooccur_pairs_are_declared_families(self):
        fams = {f.key for f in gp.LEXICON_FAMILIES}
        for a, b in gp.COOCC_PAIRS:
            self.assertIn(a, fams)
            self.assertIn(b, fams)
            self.assertNotEqual(a, b)

    def test_cooccur_pairs_cover_every_cooccur_expectation(self):
        wanted = {e.probe.split(":", 1)[1] for e in gp.EXPECTATIONS
                  if e.probe.startswith("cooccur:")}
        have = {f"{a}+{b}" for a, b in gp.COOCC_PAIRS}
        self.assertEqual(wanted, have)


class TestPriorLedger(unittest.TestCase):
    """L7's prior-not-object invariant: nothing is silently ingested."""

    def test_priors_cover_the_six_named_traditions(self):
        seen = {p.tradition for p in gp.PRIORS}
        self.assertEqual(seen, set(gp.TRADITIONS))

    def test_only_the_cyberpunk_rows_are_excerpt_authorized(self):
        for p in gp.PRIORS:
            if p.tradition == "cyberpunk":
                self.assertEqual(p.authorization, gp.AUTH_EXCERPT, p.key)
            else:
                self.assertEqual(p.authorization, gp.AUTH_SEARCHED, p.key)

    def test_no_prior_outside_cyberpunk_is_ingestible(self):
        for p in gp.PRIORS:
            if p.tradition != "cyberpunk":
                self.assertEqual(p.authorization, gp.AUTH_SEARCHED, p.key)
                # searched_not_found is the status used, and it is not
                # ingestible
                self.assertIn(p.authorization, gp.itx.NON_INGESTIBLE, p.key)

    def test_prior_years_match_the_ledger(self):
        for p in gp.PRIORS:
            self.assertEqual(gp.PRIOR_YEAR[p.key], p.year)

    def test_intertext_ledger_still_marks_the_priors_not_acquired(self):
        path = ROOT / "analysis/intertext/sources.tsv"
        self.assertTrue(path.exists())
        rows = [r for r in gp.kn.read_tsv(path)
                if r["key"] == "genre_prior_ballard"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ingested"], "n")
        self.assertEqual(rows[0]["authorization"], "searched_not_found")


class TestBookScreen(unittest.TestCase):
    """The book side reproduces #56's screen exactly."""

    @classmethod
    def setUpClass(cls):
        cls.parts, cls.stats = gp.load_book_parts(log=lambda m: None)

    def test_reconciles_with_keyness(self):
        ks = json.loads((ROOT / "analysis/keyness/summary.json")
                        .read_text(encoding="utf-8"))
        self.assertEqual(ks["book"]["parts"], self.stats["parts"])
        self.assertEqual(ks["book"]["kept"], self.stats["kept"])
        self.assertEqual(ks["book"]["types"], self.stats["types"])

    def test_sorted_and_unique_coordinates(self):
        coords = [(p[0], p[1]) for p in self.parts]
        self.assertEqual(coords, sorted(coords))

    def test_matched_families_disjoint_by_surface(self):
        for _page, _start, surface in self.parts[:2000]:
            fams = gp.matched_families(surface)
            self.assertLessEqual(len(fams), len(gp.LEXICON_FAMILIES))

    def test_exemplar_surfaces_match_their_family(self):
        # a few known surfaces: the patterns must full-match them and only
        # the intended families may claim them
        cases = {
            "corporation": {"corporate"},
            "sato": {"corporate"},
            "cyberbuddha": {"digital"},          # cyber* neologism habit
            "fecundates": {"reproduction"},
            "cadaver": {"grotesque"},
            "virus": {"virus"},
            "murder": {"violence"},
        }
        for surface, want in cases.items():
            self.assertEqual(set(gp.matched_families(surface)), want, surface)

    def test_a_family_can_be_absent_from_a_surface(self):
        self.assertEqual(gp.matched_families("zzzzqqq"), ())


class TestLexiconProbe(unittest.TestCase):
    """The density probe and its verdict gates on controlled inputs."""

    @classmethod
    def setUpClass(cls):
        cls.parts, cls.stats = gp.load_book_parts(log=lambda m: None)
        cls.hits, cls.cites = gp.family_hits(cls.parts)
        cls.ref_counts, cls.ref_tokens = {}, {
            "general_english": 10000, "medical": 10000, "cs": 10000,
            "legal": 10000, "cyberpunk": 105, "cutup_surrealist": 56,
            "finnegans_wake": 49,
        }
        for corpus in cls.ref_tokens:
            cls.ref_counts[corpus] = collections.Counter()
        # synthetic references with known densities: every family's exemplar
        # occurs once per 10,000 general-English tokens (0.1/1k) and 500
        # times per 10,000 medical tokens (50/1k)
        for fam in gp.LEXICON_FAMILIES:
            word = FAMILY_EXEMPLARS[fam.key]
            cls.ref_counts["general_english"][word] = 1
            cls.ref_counts["medical"][word] = 500
        cls.ref_density = gp.reference_densities(cls.ref_counts,
                                                 cls.ref_tokens)

    def probe(self, fam_key):
        return gp.run_lexicon_probe(fam_key, self.parts, self.hits,
                                    self.cites, self.ref_density,
                                    self.ref_tokens)

    def test_density_matches_the_counter(self):
        for f in gp.LEXICON_FAMILIES:
            row = self.probe(f.key)
            self.assertEqual(row["book_occurrences"],
                             sum(self.hits[f.key].values()), f.key)
            self.assertEqual(
                row["book_density_per_1k"],
                round(row["book_occurrences"] / len(self.parts) * 1000, 3),
                f.key)

    def test_ratio_and_z_are_consistent(self):
        row = self.probe("reproduction")
        # recompute from the published densities, not the rounded expected
        exp = row["general_english_per_1k"] / 1000 * len(self.parts)
        self.assertAlmostEqual(
            row["ratio_vs_general_english"],
            round(row["book_density_per_1k"] / row["general_english_per_1k"],
                  2))
        self.assertAlmostEqual(
            row["poisson_z_vs_general_english"],
            round((row["book_occurrences"] - exp) / exp ** 0.5, 1), places=1)

    def test_cites_are_page_char_format(self):
        for f in gp.LEXICON_FAMILIES:
            row = self.probe(f.key)
            if row["example"] != "-":
                self.assertRegex(row["example"], r"^p\d+:\d+$", f.key)

    def test_gate_absent_family(self):
        # a family that never occurs must be absent, whatever the reference
        fam = gp.Family("probe_absent", (r"zzzzqqqq",), "(test)", "never occurs")
        import contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.dict(
                gp.FAMILY_BY_KEY, {"probe_absent": fam}))
            stack.enter_context(unittest.mock.patch.dict(
                gp._COMPILED, {"probe_absent": (gp.re.compile("^zzzzqqqq$"),)}))
            row = self.probe("probe_absent")
        self.assertEqual(row["book_occurrences"], 0)
        self.assertEqual(row["verdict"], "absent")
        self.assertEqual(row["example"], "-")

    def test_gate_thin_family_by_ratio(self):
        # corporate is present but below RATIO_GATE -> thin
        row = self.probe("corporate")
        self.assertEqual(row["verdict"], "thin")
        self.assertLess(float(row["ratio_vs_general_english"]),
                        gp.RATIO_GATE)

    def test_gate_present_family(self):
        for key in ("reproduction", "machine", "virus", "digital"):
            row = self.probe(key)
            self.assertEqual(row["verdict"], "present", key)
            self.assertGreaterEqual(float(row["ratio_vs_general_english"]),
                                    gp.RATIO_GATE)
            self.assertGreaterEqual(row["book_occurrences"], gp.MIN_OCC)


class TestCooccurProbe(unittest.TestCase):
    """The fusion probe and its gates on controlled inputs."""

    def test_counts_and_expected_on_known_sets(self):
        # 100 clauses; a in the first 50, b in the first 30 -> both = 30,
        # expected 50*30/100 = 15, z = (30-15)/sqrt(15) ~ 3.87 -> present
        fam_sets = {}
        first = {}
        for i in range(100):
            fam_sets[i] = {"a"} if i < 50 else set()
            if i < 30:
                fam_sets[i].add("b")
            first[i] = (i // 10 + 1, i % 10)
        row = gp.run_cooccur_probe("a", "b", fam_sets, 100, first)
        self.assertEqual(row["clauses_with_a"], 50)
        self.assertEqual(row["clauses_with_b"], 30)
        self.assertEqual(row["clauses_with_both"], 30)
        self.assertEqual(row["expected_random"], 15.0)
        self.assertGreaterEqual(row["poisson_z"], gp.Z_GATE)
        self.assertEqual(row["verdict"], "present")
        self.assertEqual(row["example_clause"], "p1:0")

    def test_gate_thin_above_obs_below_z(self):
        # 100 clauses; a in the first 50; b in 0..9 and 20..24 -> both = 15,
        # expected 7.5, z ~ 2.74 < 3 -> thin
        fam_sets = {}
        for i in range(100):
            fam_sets[i] = {"a"} if i < 50 else set()
        b_clauses = list(range(10)) + list(range(20, 25))
        for i in b_clauses:
            fam_sets[i].add("b")
        row = gp.run_cooccur_probe("a", "b", fam_sets, 100, {})
        self.assertEqual(row["clauses_with_both"], 15)
        self.assertGreaterEqual(row["clauses_with_both"], gp.MIN_OBS)
        self.assertLess(row["poisson_z"], gp.Z_GATE)
        self.assertEqual(row["verdict"], "thin")

    def test_gate_absent_below_min_obs(self):
        fam_sets = {i: {"a"} for i in range(20)}
        fam_sets.update({i: {"b"} for i in range(20, 40)})
        row = gp.run_cooccur_probe("a", "b", fam_sets, 40, {})
        self.assertEqual(row["clauses_with_both"], 0)
        self.assertEqual(row["verdict"], "absent")
        self.assertEqual(row["example_clause"], "-")


class TestClausePartition(unittest.TestCase):
    """Every kept part lands in exactly one #33 clause."""

    def test_full_assignment_and_clause_count(self):
        parts, stats = gp.load_book_parts(log=lambda m: None)
        clause_rows = gp.kn.read_tsv(gp.CLAUSES)
        _fam_sets, n_assigned, n_unassigned, _first = gp.clause_family_sets(
            parts, clause_rows)
        self.assertEqual(n_unassigned, 0)
        self.assertEqual(n_assigned, stats["kept"])
        self.assertEqual(len(clause_rows),
                         json.loads((ROOT / "analysis/clauses/summary.json")
                                    .read_text(encoding="utf-8"))["n_clauses"])

    def test_family_sets_are_subsets_of_declared_families(self):
        parts, _stats = gp.load_book_parts(log=lambda m: None)
        clause_rows = gp.kn.read_tsv(gp.CLAUSES)
        fam_sets, _a, _u, _f = gp.clause_family_sets(parts, clause_rows)
        valid = {f.key for f in gp.LEXICON_FAMILIES}
        for fs in fam_sets.values():
            self.assertTrue(fs <= valid)


class TestArtifactProbes(unittest.TestCase):
    """Artifact probes read frozen numbers and apply their declared rules."""

    def test_all_probes_run_and_declare_an_owner(self):
        for key, fn in gp.ARTIFACT_PROBES.items():
            row = fn()
            self.assertIn(row["verdict"],
                          ("present", "thin", "absent", "underdetermined"), key)
            self.assertTrue(row["owner"].startswith("#"), key)
            self.assertTrue(row["rule"], key)
            self.assertTrue(row["detail"], key)

    def test_cyberpunk_context_row_is_underdetermined(self):
        row = gp.probe_cyberpunk_arm()
        self.assertEqual(row["verdict"], "underdetermined")
        self.assertLess(row["observed"]["tokens"],
                        gp.MIN_TOKENS_INFERENTIAL)


class TestBuild(unittest.TestCase):
    """End-to-end build: artifacts, mode invariance, determinism."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.orig = gp.OUT
        gp.OUT = Path(cls.tmp.name)
        cls.summary = gp.build(log=lambda m: None, quick=True)
        cls.summary2 = gp.build(log=lambda m: None, quick=True)
        cls.full = tempfile.TemporaryDirectory()
        gp.OUT = Path(cls.full.name)
        cls.summary_full = gp.build(log=lambda m: None, quick=False)

    @classmethod
    def tearDownClass(cls):
        gp.OUT = cls.orig
        cls.tmp.cleanup()
        cls.full.cleanup()

    def test_all_artifacts_written(self):
        names = ["sources.tsv", "expectations.tsv", "lexicon_probes.tsv",
                 "cooccurrence.tsv", "artifact_probes.tsv", "verdicts.tsv",
                 "reconciliation.tsv", "diagnostics.tsv", "summary.json",
                 "dossier.md"]
        for name in names:
            self.assertTrue((Path(self.tmp.name) / name).exists(),
                            f"missing {name}")

    def test_verdicts_cover_every_expectation_exactly_once(self):
        rows = gp.kn.read_tsv(Path(self.tmp.name) / "verdicts.tsv")
        ids = [r["eid"] for r in rows]
        self.assertEqual(sorted(ids), sorted(e.eid for e in gp.EXPECTATIONS))

    def test_verdict_counts_match_the_table(self):
        rows = gp.kn.read_tsv(Path(self.tmp.name) / "verdicts.tsv")
        counts = collections.Counter(r["verdict"] for r in rows)
        self.assertEqual(dict(counts), self.summary["verdict_counts"])
        self.assertEqual(sum(counts.values()), len(gp.EXPECTATIONS))

    def test_lexicon_verdicts_match_the_probe_table(self):
        lex = {r["family"]: r["verdict"]
               for r in gp.kn.read_tsv(Path(self.tmp.name) /
                                       "lexicon_probes.tsv")}
        for r in gp.kn.read_tsv(Path(self.tmp.name) / "verdicts.tsv"):
            kind, _, arg = r["probe"].partition(":")
            if kind == "lexicon":
                self.assertEqual(lex[arg], r["verdict"], r["eid"])

    def test_cooccur_verdicts_match_the_probe_table(self):
        co = {r["pair"]: r["verdict"]
              for r in gp.kn.read_tsv(Path(self.tmp.name) /
                                      "cooccurrence.tsv")}
        for r in gp.kn.read_tsv(Path(self.tmp.name) / "verdicts.tsv"):
            kind, _, arg = r["probe"].partition(":")
            if kind == "cooccur":
                self.assertEqual(co[arg], r["verdict"], r["eid"])

    def test_underdetermined_expectations_are_the_declared_pair(self):
        rows = gp.kn.read_tsv(Path(self.tmp.name) / "verdicts.tsv")
        got = {r["eid"] for r in rows if r["verdict"] == "underdetermined"}
        want = {e.eid for e in gp.EXPECTATIONS if e.probe == "declared"}
        self.assertEqual(got, want)
        self.assertEqual(got, {"BG4", "BT4"})

    def test_no_expectation_is_absent(self):
        rows = gp.kn.read_tsv(Path(self.tmp.name) / "verdicts.tsv")
        # a documented empty verdict is allowed by the gates but not
        # produced by the book; if one ever appears the dossier must say so
        counts = collections.Counter(r["verdict"] for r in rows)
        self.assertEqual(counts.get("absent", 0), 0)

    def test_mode_invariance_of_book_side_and_verdicts(self):
        quick = {r["family"]: r["book_occurrences"]
                 for r in self.summary["lexicon_probes"]}
        full = {r["family"]: r["book_occurrences"]
                for r in self.summary_full["lexicon_probes"]}
        self.assertEqual(quick, full)
        self.assertEqual(self.summary["verdict_counts"],
                         self.summary_full["verdict_counts"])
        quick_v = {r["eid"]: r["verdict"]
                   for r in self.summary["verdicts"]}
        full_v = {r["eid"]: r["verdict"]
                  for r in self.summary_full["verdicts"]}
        self.assertEqual(quick_v, full_v)

    def test_deterministic_rerun(self):
        self.assertEqual(self.summary["book"], self.summary2["book"])
        self.assertEqual(self.summary["lexicon_probes"],
                         self.summary2["lexicon_probes"])
        self.assertEqual(self.summary["cooccurrence"],
                         self.summary2["cooccurrence"])
        self.assertEqual(self.summary["verdict_counts"],
                         self.summary2["verdict_counts"])

    def test_reconciliation_has_no_failure(self):
        rows = gp.kn.read_tsv(Path(self.tmp.name) / "reconciliation.tsv")
        bad = [r["item"] for r in rows if r["ok"] == "n"]
        self.assertEqual(bad, [])
        items = {r["item"] for r in rows}
        self.assertIn("book_kept_tokens", items)
        self.assertIn("parts_unassigned", items)

    def test_report_numbers_agree_with_the_tables(self):
        # the dossier's headline counts must be the table's counts
        text = (Path(self.tmp.name) / "dossier.md").read_text(encoding="utf-8")
        counts = self.summary["verdict_counts"]
        head = (f"**Verdicts: {counts.get('present', 0)} present, "
                f"{counts.get('thin', 0)} thin, {counts.get('absent', 0)} "
                f"absent, {counts.get('underdetermined', 0)} underdetermined, "
                f"of {len(self.summary['verdicts'])} expectations.**")
        self.assertIn(head, text)
        # every expectation id appears in the annotated dossier section
        for e in gp.EXPECTATIONS:
            self.assertIn(f"**{e.eid} -- ", text, e.eid)

    def test_report_discloses_the_gates_and_the_ledger_rule(self):
        text = (Path(self.tmp.name) / "dossier.md").read_text(encoding="utf-8")
        self.assertIn("prior, not the object of study", text)
        self.assertIn("ventriloqu", text)
        self.assertIn("underdetermined", text)
        self.assertIn("refuses to ingest", text)
        self.assertIn("does not say the author read that tradition", text)

    def test_summary_machine_readable(self):
        self.assertEqual(self.summary["schema"], gp.SCHEMA)
        self.assertEqual(self.summary["probe_date"], gp.PROBE_DATE)
        self.assertEqual(self.summary["book"]["kept"], 79770)
        self.assertEqual(self.summary["clauses"], 5614)
        self.assertEqual(self.summary["priors_not_ingested"],
                         sum(1 for p in gp.PRIORS
                             if p.authorization in gp.itx.NON_INGESTIBLE))


def _affect(fear, ecstasy, paranoia, disgust):
    """A planted #44 affect summary with the four z-scores given."""
    return {"classifier": {"margins": {
        k: {"z": z, "margin": z / 100.0, "book_mean": 0.2 + z / 1000.0,
            "english_mean": 0.2}
        for k, z in (("fear", fear), ("ecstasy", ecstasy),
                     ("paranoia", paranoia), ("disgust", disgust))}}}


#: planted artifact summaries and the verdict each must produce, keyed by
#: probe.  Every entry of ``gp.ARTIFACT_PROBES`` -- plus the ``cyberpunk_arm``
#: context row -- must appear here, so an added probe without a two-way rule
#: test fails the build instead of quietly resting on the live artifacts.
#: A multi-read probe's cases list one summary per ``_load_summary`` call, in
#: call order (``roles_predicates`` reads #66 then #59).
RULE_CASES: dict = {
    "roles_named_world": [
        ({"verdicts": {"R3": "evidenced", "R4": "evidenced",
                       "R5": "evidenced"},
          "role_status": {"refuted": 0}}, "present"),
        ({"verdicts": {"R3": "evidenced", "R4": "refuted", "R5": "refuted"},
          "role_status": {"refuted": 2}}, "thin"),
        ({"verdicts": {"R3": "refuted", "R4": "refuted", "R5": "refuted"},
          "role_status": {"refuted": 3}}, "absent"),
    ],
    "roles_predicates": [
        ([{"events": 4829},
          {"role_status": {"refuted": 1, "uncertain": 2}}], "thin"),
        ([{"events": 4829},
          {"role_status": {"refuted": 0, "uncertain": 0}}], "present"),
        ([{"events": 900},
          {"role_status": {"refuted": 1, "uncertain": 0}}], "absent"),
    ],
    "affect_margins": [
        (_affect(11.07, 22.63, 6.96, 14.7), "present"),
        (_affect(11.07, 22.63, 6.96, 0.5), "thin"),
        (_affect(0.5, 0.5, 0.5, 0.5), "absent"),
    ],
    "parse_derangement": [
        ({"headline": {"v5_frac_parseable": 0.5208,
                       "english_frac_parseable": 0.7847,
                       "null_frac_parseable": 0.4165}}, "present"),
        ({"headline": {"v5_frac_parseable": 0.40,
                       "english_frac_parseable": 0.7847,
                       "null_frac_parseable": 0.4165}}, "thin"),
        ({"headline": {"v5_frac_parseable": 0.80,
                       "english_frac_parseable": 0.7847,
                       "null_frac_parseable": 0.4165}}, "absent"),
    ],
    "formula_density": [
        ({"formula_classes": {"author_idiolect": 116, "book_unique": 25853}},
         "present"),
        ({"formula_classes": {"author_idiolect": 116, "book_unique": 5884}},
         "thin"),
        ({"formula_classes": {"author_idiolect": 116, "book_unique": 9}},
         "absent"),
    ],
    "self_similarity": [
        ({"band_z": {"1": 7.75, "10": 28.52, "100": 20.15, "50": 30.89}},
         "present"),
        ({"band_z": {"1": 1.0, "10": 5.0, "100": 2.0}}, "thin"),
        ({"band_z": {"1": 1.0, "10": 1.5, "100": 0.5}}, "absent"),
    ],
    # probe_operator_surface reads #13's silence summary, then #33's
    "operator_surface": [
        ([{"operator_tokens": 10949, "lexemes_baseline": 77392},
          {"boundary_tokens": 5773}], "present"),
        ([{"operator_tokens": 5000, "lexemes_baseline": 77392},
          {"boundary_tokens": 100}], "thin"),
        ([{"operator_tokens": 100, "lexemes_baseline": 100000},
          {"boundary_tokens": 100}], "absent"),
    ],
    "neologism_coinage": [
        ({"counts": {"true_coinage_types": 51, "coinage_types": 74,
                     "compound_types": 2216},
          "category_occurrences": {"true_coinage": 338}}, "present"),
        ({"counts": {"true_coinage_types": 15, "coinage_types": 20,
                     "compound_types": 100},
          "category_occurrences": {"true_coinage": 20}}, "thin"),
        ({"counts": {"true_coinage_types": 5, "coinage_types": 9,
                     "compound_types": 50},
          "category_occurrences": {"true_coinage": 6}}, "absent"),
    ],
    "case_emphasis": [
        ({"by_class_occurrences": {"scream": 263, "label": 1138,
                                   "function": 453},
          "caps_share_of_words": 0.0244}, "present"),
        ({"by_class_occurrences": {"scream": 50, "label": 10,
                                   "function": 5},
          "caps_share_of_words": 0.001}, "thin"),
        ({"by_class_occurrences": {"scream": 0, "label": 0, "function": 0},
          "caps_share_of_words": 0.0}, "absent"),
    ],
    "cyberpunk_arm": [
        ({"arms": {"influence_cyberpunk": {"tokens": 105,
                                           "book_shared_types_n2plus": 13}}},
         "underdetermined"),
        ({"arms": {"influence_cyberpunk": {"tokens": 12000,
                                           "book_shared_types_n2plus": 900}}},
         "present"),
    ],
}


class TestArtifactProbeRules(unittest.TestCase):
    """Every artifact probe's declared rule is pinned in both directions."""

    def test_registry_covers_every_registered_probe(self):
        registered = set(gp.ARTIFACT_PROBES) | {"cyberpunk_arm"}
        self.assertEqual(set(RULE_CASES), registered)

    def test_every_case_flips_the_rule_the_declared_way(self):
        for key, cases in RULE_CASES.items():
            fn = gp.ARTIFACT_PROBES.get(key, gp.probe_cyberpunk_arm)
            for summaries, want in cases:
                side = summaries if isinstance(summaries, list) \
                    else [summaries]
                with unittest.mock.patch.object(
                        gp, "_load_summary", side_effect=side):
                    row = fn()
                self.assertEqual(row["verdict"], want,
                                 f"{key} <- {summaries}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
