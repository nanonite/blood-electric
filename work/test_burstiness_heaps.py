#!/usr/bin/env python3
"""Regression suite for #53 burstiness and Heaps' law.

    python work/test_burstiness_heaps.py

Six layers, mirroring the house pattern:

* **Primitives** -- the growth curve, the Heaps estimator, the Fano factor,
  the Goh-Barabasi burstiness, the relocation null and the verdict rule, on
  synthetic data where the answer is known.
* **Entity location** -- #30's own phrase rule re-derived: every glossary row
  must locate exactly, and a locator that ignored adjacency, capitalization or
  case must fail (planted).
* **Reconciliation** -- every beta must reproduce #24's ``fits.tsv`` row for
  row, both stream sizes must reproduce the screens they claim, and the
  entity mentions must sum to #30's glossary.
* **Estimator choice** -- the jackknife mean reproduces the point estimate
  (the property the Method claims) while a page bootstrap biases beta down
  (the failure mode it is chosen to avoid).
* **End-to-end build** (a temp dir, ``--quick``) -- artifact shapes, every
  reconciliation row ``ok``, the report's headline numbers equal the
  artifacts', and a second build is byte-identical.
* **Nulls have teeth** -- a clustered position set is caught, an even one is
  not, and the draws are seeded (order-independent, reproducible).
"""
from __future__ import annotations

import csv
import itertools
import json
import random
import re
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import burstiness_heaps as bh  # noqa: E402

HAVE_ARTIFACTS = bh.GLOSSARY.exists()
_ART: dict = {}


def setUpModule():
    if not HAVE_ARTIFACTS:
        return
    # the book-only path (no reference cache needed), the house test path
    tmp = tempfile.mkdtemp(prefix="burstiness_test_")
    rc = bh.main(["--quick", "--out", tmp, "--no-download", "--quiet"])
    assert rc == 0, "build reported a reconciliation mismatch"
    for name in ("heaps_fits", "growth_curve", "segment_growth",
                 "entity_growth", "entity_burstiness", "controls_burstiness",
                 "reconciliation", "diagnostics"):
        path = Path(tmp) / f"{name}.tsv"
        _ART[name] = bh.read_tsv(path) if path.exists() else []
    with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
        _ART["summary"] = json.load(fh)
    _ART["report"] = (Path(tmp) / "burstiness_report.md").read_text(
        encoding="utf-8")
    _ART["burst_rows"] = _ART["entity_burstiness"]
    _ART["_dir"] = tmp
    # a second, independent build must be byte-identical (determinism)
    tmp2 = tempfile.mkdtemp(prefix="burstiness_test2_")
    bh.main(["--quick", "--out", tmp2, "--no-download", "--quiet"])
    for name in ("summary.json", "burstiness_report.md", "heaps_fits.tsv",
                 "entity_burstiness.tsv", "controls_burstiness.tsv",
                 "reconciliation.tsv"):
        a = (Path(tmp) / name).read_bytes()
        b = (Path(tmp2) / name).read_bytes()
        assert a == b, f"non-deterministic artifact: {name}"
    _ART["_dir2"] = tmp2


def rows(name):
    return _ART[name]


def fits_by_corpus():
    return {r["corpus"]: r for r in rows("heaps_fits")}


# --------------------------------------------------------------------------
class TestPrimitives(unittest.TestCase):
    """The estimators on data where the answer is known."""

    def test_growth_curve_is_exact_and_monotone(self):
        toks = list("abcab") + ["z"] * 3 + list("qwr")
        curve = bh.growth_curve(toks, n_points=4)
        self.assertEqual(curve[-1], (len(toks), len(set(toks))))
        vs = [v for _n, v in curve]
        self.assertEqual(vs, sorted(vs))
        ns = [n for n, _v in curve]
        self.assertEqual(ns, sorted(ns))
        # every checkpoint's V is the exact cumulative type count
        for n, v in curve:
            self.assertEqual(len(set(toks[:n])), v)

    def test_growth_curve_small_input(self):
        self.assertEqual(bh.growth_curve(["x"]), [(1, 1)])

    def test_heaps_beta_orders_novel_against_closed_streams(self):
        # every type unique: V = N exactly -> beta 1, log K 0
        novel = [f"w{i}" for i in range(400)]
        fit = bh.heaps_fit_r2(novel)
        self.assertAlmostEqual(fit["heaps_beta"], 1.0, places=6)
        self.assertAlmostEqual(fit["heaps_logk"], 0.0, places=6)
        # a closed vocabulary stops growing: the ramp is a boundary effect
        closed = [f"t{i % 50}" for i in range(4000)]
        fit2 = bh.heaps_fit_r2(closed)
        self.assertLess(fit2["heaps_beta"], 0.1)

    def test_heaps_fit_is_deterministic(self):
        toks = [f"t{i % 700}" for i in range(7000)]
        self.assertEqual(bh.heaps_fit_r2(toks), bh.heaps_fit_r2(toks))

    def test_heaps_r2_is_a_fit_quality(self):
        toks = [f"t{i % 300}" for i in range(6000)]
        r2 = bh.heaps_fit_r2(toks)["heaps_r2"]
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)

    def test_v_at_marks_exact_counts(self):
        toks = [f"t{i % 100}" for i in range(60_000)]
        marks = bh.v_at(toks)
        for m in bh.V_MARKS:
            self.assertEqual(marks[m], 100)
            self.assertEqual(marks[m], len(set(toks[:m])))

    def test_fano_poisson_like_uniform_sprinkle(self):
        # a Bernoulli sprinkle at its own rate: F ~ 1
        rng = random.Random(53)
        n_total = 100_000
        pos = sorted(rng.choices(range(n_total), k=1000))
        f = bh.fano_factor(pos, 1000, n_total)
        self.assertAlmostEqual(f, 1.0, delta=0.25)

    def test_fano_detects_even_and_clustered(self):
        n_total = 100_000
        # perfectly even: exactly one hit per window -> F == 0
        even = list(range(0, 100_000, 1000))
        self.assertAlmostEqual(bh.fano_factor(even, 1000, n_total), 0.0)
        # all hits inside 10 adjacent windows -> F >> 1
        clustered = [i for i in range(10_000)]
        f = bh.fano_factor(clustered, 1000, n_total)
        self.assertGreater(f, 8.0)

    def test_fano_needs_windows(self):
        pos = list(range(50))
        self.assertIsNone(bh.fano_factor(pos, 10_000, 60_000))  # 6 windows
        self.assertIsNotNone(bh.fano_factor(pos, 1000, 60_000))

    def test_burstiness_directions(self):
        # equal gaps: sd 0 -> B = -1 (clockwork)
        self.assertAlmostEqual(bh.burstiness([5, 5, 5, 5, 5]), -1.0)
        # memoryless gaps (exponential): CV 1 -> B ~ 0
        rng = random.Random(7)
        expo = [rng.expovariate(0.1) for _ in range(4000)]
        self.assertLess(abs(bh.burstiness(expo)), 0.15)
        # two-regime gaps: a dense short-gap mass with a rare long jump -- B
        # strongly positive (a bursty arrival process)
        mixed = [1] * 990 + [1000] * 10
        self.assertGreater(bh.burstiness(mixed), 0.5)

    def test_burstiness_needs_gaps(self):
        self.assertIsNone(bh.burstiness([7]))
        self.assertIsNone(bh.burstiness([]))

    def test_null_fanos_seeded_and_reproducible(self):
        a = bh.null_fanos(500, 1000, 100_000, seed=53, draws=50)
        b = bh.null_fanos(500, 1000, 100_000, seed=53, draws=50)
        self.assertEqual(a, b)                      # same seed -> same draws
        c = bh.null_fanos(500, 1000, 100_000, seed=54, draws=50)
        self.assertNotEqual(a, c)                   # different seed -> different
        self.assertAlmostEqual(statistics.median(a), 1.0, delta=0.15)

    def test_entity_seed_is_order_independent(self):
        # the null for a term must not depend on which row it sits in
        self.assertEqual(bh.entity_seed("ADAM"), bh.entity_seed("ADAM"))
        self.assertNotEqual(bh.entity_seed("ADAM"), bh.entity_seed("cadaver"))
        self.assertEqual(bh.null_fanos(200, 1000, 100_000, bh.entity_seed("x"),
                                       5),
                         bh.null_fanos(200, 1000, 100_000, bh.entity_seed("x"),
                                       5))

    def test_burst_p_value_add_one_corrected(self):
        nulls = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(bh.burst_p_value(0.5, nulls), 1.0)
        self.assertAlmostEqual(bh.burst_p_value(5.0, nulls), 1 / 5)
        self.assertAlmostEqual(bh.burst_p_value(2.5, nulls), 3 / 5)
        self.assertIsNone(bh.burst_p_value(None, nulls))
        self.assertIsNone(bh.burst_p_value(1.0, []))

    def test_verdict_rule_requires_both_legs(self):
        # HIV's real case: p tiny but B negative -> not bursty
        self.assertEqual(bh.verdict(3.29, 0.005, -0.262), "neutral")
        self.assertEqual(bh.verdict(3.29, 0.005, 0.3), "bursty")
        # p not small enough: neutral unless under-dispersed
        self.assertEqual(bh.verdict(1.2, 0.3, 0.3), "neutral")
        # under-dispersed is `even` even at p = 1
        self.assertEqual(bh.verdict(0.8, 1.0, -0.5), "even")
        self.assertEqual(bh.verdict(None, None, None), "untested")

    def test_gap_stats(self):
        stats = bh.gap_stats([10, 20, 25])
        self.assertEqual(stats["mean_gap"], 7.5)
        self.assertEqual(stats["max_gap"], 10)
        self.assertAlmostEqual(stats["gap_cv"], statistics.pstdev([10, 5])
                               / 7.5)

    def test_segment_table_conserves_novelty(self):
        toks = [f"t{i % 900}" for i in range(9000)]
        segs = bh.segment_table(toks, k=5)
        self.assertEqual(sum(s["n"] for s in segs), len(toks))
        self.assertEqual(sum(s["new_types"] for s in segs), len(set(toks)))
        self.assertEqual(segs[-1]["cum_v"], len(set(toks)))
        for i, s in enumerate(segs, 1):
            self.assertEqual(s["segment"], i)

    def test_groups_for_jackknife_page_and_block(self):
        tagged = [(1, "a"), (1, "b"), (2, "c"), (2, "d")]
        self.assertEqual(bh.groups_for_jackknife(tagged),
                         [["a", "b"], ["c", "d"]])
        blocks = bh.groups_for_jackknife(list(enumerate("abcdef")), block=2)
        self.assertEqual(blocks, [["a", "b"], ["c", "d"], ["e", "f"]])


class TestEstimatorChoice(unittest.TestCase):
    """Why the jackknife and not a bootstrap -- on synthetic page corpora."""

    def test_jackknife_mean_reproduces_point_estimate(self):
        # a corpus of 50-token pseudo-pages, each introducing some new types
        pages = []
        for p in range(40):
            pages.extend([f"p{p}w{i}" for i in range(20)])
            pages.extend([f"core{i}" for i in range(30)])
        groups = [pages[i:i + 50] for i in range(0, len(pages), 50)]
        point = bh.heaps_fit_r2(pages)["heaps_beta"]
        refits = bh.jackknife_beta(groups)
        self.assertAlmostEqual(statistics.mean(refits), point, delta=0.02)
        self.assertLess(statistics.pstdev(refits), 0.05)

    def test_page_bootstrap_biases_beta_down(self):
        # the failure mode the Method names: duplicating a page contributes no
        # new types, so a page-resampled corpus grows slower
        rng = random.Random(53)
        pages = [[f"p{p}w{i}" for i in range(20)]
                 + [f"core{i}" for i in range(30)] for p in range(40)]
        stream = [t for page in pages for t in page]
        point = bh.heaps_fit_r2(stream)["heaps_beta"]
        boots = []
        for _ in range(60):
            resampled = [t for p in rng.choices(pages, k=len(pages)) for t in p]
            boots.append(bh.heaps_fit_r2(resampled)["heaps_beta"])
        self.assertLess(statistics.mean(boots), point - 0.02)


# --------------------------------------------------------------------------
class TestEntityLocation(unittest.TestCase):
    """#30's phrase rule, re-derived -- and its teeth."""

    @staticmethod
    def fake_stream():
        # body built so that adjacency is checkable by offset; every token
        # carries its own start (duplicate surfaces must not share one)
        toks = ["the", "Sato", "Corporation", "kills", "Sato", "//",
                "Corporation", "cadaver", "city", "Cadaver", "City",
                "Super", "Cherry", "666"]
        body = " ".join(toks)
        starts, cursor = [], 0
        for t in toks:
            starts.append(cursor)
            cursor += len(t) + 1
        return {
            "surface": toks,
            "page": [1] * len(toks),
            "start": starts,
            "uid": list(range(len(toks))),
            "bodies": {1: body},
            "unit_meta": {i: (1, starts[i], starts[i] + len(toks[i]),
                              toks[i]) for i in range(len(toks))},
            "n": len(toks),
        }

    def test_single_surface_matches_exactly(self):
        s = self.fake_stream()
        self.assertEqual(bh.locate("Sato", s), [1, 4])

    def test_phrase_requires_capitalized_words(self):
        s = self.fake_stream()
        self.assertEqual(bh.locate("cadaver city", s), [])
        self.assertEqual(bh.locate("Cadaver City", s), [9])

    def test_phrase_requires_whitespace_adjacency(self):
        # the second `Sato Corporation` is split by `//` in the body
        s = self.fake_stream()
        self.assertEqual(bh.locate("Sato Corporation", s), [1])

    def test_numeric_designator_attaches_to_phrase(self):
        s = self.fake_stream()
        self.assertEqual(bh.locate("Super Cherry 666", s), [11])

    @unittest.skipUnless(HAVE_ARTIFACTS, "#30 glossary missing")
    def test_every_glossary_entity_locates_exactly(self):
        stream = bh.screened_stream()
        self.assertEqual(stream["n"], 74_077)  # the #27/#30 screen
        glossary = bh.load_glossary()
        positions, deltas = bh.entity_positions(glossary, stream)
        self.assertEqual(deltas, [])
        for r in glossary:
            pos = positions[r["entity"]]
            self.assertEqual(len(pos), int(r["n_occurrences"]), r["entity"])
            self.assertEqual(pos, sorted(set(pos)), r["entity"])
        self.assertEqual(sum(len(p) for p in positions.values()), 2_452)

    @unittest.skipUnless(HAVE_ARTIFACTS, "#30 glossary missing")
    def test_a_case_insensitive_locator_would_fail(self):
        # planted: dropping the exact-case rule over-locates, so the
        # reconciliation row is what keeps the case rule honest
        stream = bh.screened_stream()
        glossary = bh.load_glossary()
        positions, _ = bh.entity_positions(glossary, stream)
        exact = sum(len(p) for p in positions.values())
        folded = {}
        for i, u in enumerate(stream["surface"]):
            folded.setdefault(u.lower(), []).append(i)
        over = 0
        for r in glossary:
            for part in r["surface_variants"].split("|"):
                if part and len(part.rpartition(":")[0].split(" ")) == 1:
                    over += len(folded.get(part.rpartition(":")[0].lower(),
                                           []))
        self.assertGreater(over, exact)

    @unittest.skipUnless(HAVE_ARTIFACTS, "#30 glossary missing")
    def test_pages_of_located_positions_match_glossary(self):
        stream = bh.screened_stream()
        glossary = bh.load_glossary()
        positions, _ = bh.entity_positions(glossary, stream)
        for r in glossary:
            pos = positions[r["entity"]]
            pages = {stream["page"][i] for i in pos}
            self.assertEqual(len(pages), int(r["n_pages"]), r["entity"])
            self.assertEqual(min(pages), int(r["first_page"]), r["entity"])
            self.assertEqual(max(pages), int(r["last_page"]), r["entity"])


class TestReconciliation(unittest.TestCase):
    """Every number this build publishes must reproduce its upstream."""

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_betas_reproduce_24_fits(self):
        if not bh.FREQ_FITS.exists():
            self.skipTest("#24 fits.tsv not built")
        up = {r["corpus"]: r for r in bh.read_tsv(bh.FREQ_FITS)}
        for corpus, row in fits_by_corpus().items():
            if corpus == "en_mean":
                expected = float(up["en_mean"]["heaps_beta"])
            elif corpus in up:
                expected = float(up[corpus]["heaps_beta"])
            else:
                continue
            self.assertAlmostEqual(float(row["heaps_beta"]), expected,
                                   places=9, msg=corpus)

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_stream_sizes_reproduce_both_screens(self):
        diag = {r["item"]: r["value"] for r in rows("diagnostics")}
        self.assertEqual(diag["stream_growth_n"], "74318")   # #24 screen
        self.assertEqual(diag["stream_entity_n"], "74077")    # #30 screen
        self.assertEqual(diag["screen_delta"], "241")

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_every_reconciliation_row_is_ok(self):
        bad = [r for r in rows("reconciliation") if r["status"] == "MISMATCH"]
        self.assertEqual(bad, [])
        checks = [r["check"] for r in rows("reconciliation")]
        self.assertTrue(any("fits.tsv" in c for c in checks))
        self.assertTrue(any("located exactly" in c for c in checks))

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_jackknife_ranges_bracket_the_point_estimate(self):
        # the property the Method claims, on the real book layers
        for corpus, row in fits_by_corpus().items():
            if not row["beta_loo_min"] or corpus == "en_mean":
                continue
            lo, hi = float(row["beta_loo_min"]), float(row["beta_loo_max"])
            beta = float(row["heaps_beta"])
            self.assertLessEqual(lo, beta, corpus)
            self.assertGreaterEqual(hi, beta, corpus)


# --------------------------------------------------------------------------
class TestBuild(unittest.TestCase):
    """The end-to-end build's artifact shapes and internal consistency."""

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_artifact_headers(self):
        for name, header in (
            ("heaps_fits", ["corpus", "kind", "N", "V", "heaps_beta",
                            "heaps_logk", "heaps_r2", "beta_loo_sd",
                            "beta_loo_min", "beta_loo_max",
                            "jackknife_groups"] + [f"V_{m // 1000}k"
                                                   for m in bh.V_MARKS]),
            ("growth_curve", ["corpus", "N", "V"]),
            ("segment_growth", ["corpus", "segment", "n", "v", "new_types",
                                "cum_v", "beta_seg"]),
            ("entity_growth", ["corpus", "N", "V"]),
            ("entity_burstiness", ["entity", "role", "n", "n_pages",
                                   "first_page", "last_page", "top_page",
                                   "max_page_count", "top_page_share",
                                   "mean_gap", "max_gap", "gap_cv",
                                   "burstiness_b"]
             + [f"fano_{w}" for w in bh.FANO_WINDOWS]
             + ["null_fano_median", "null_fano_q95", "p_bursty", "verdict"]),
            ("controls_burstiness", ["kind", "term", "n", "n_pages",
                                     "mean_gap", "max_gap", "gap_cv",
                                     "burstiness_b"]
             + [f"fano_{w}" for w in bh.FANO_WINDOWS]
             + ["null_fano_median", "p_bursty", "verdict"]),
            ("reconciliation", ["check", "expected", "actual", "status",
                                "note"]),
            ("diagnostics", ["item", "value", "note"]),
        ):
            with open(Path(_ART["_dir"]) / f"{name}.tsv", encoding="utf-8") as fh:
                self.assertEqual(fh.readline().rstrip("\n").split("\t"), header,
                                 name)

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_tsv_safety(self):
        for name in ("heaps_fits", "growth_curve", "segment_growth",
                     "entity_growth", "entity_burstiness",
                     "controls_burstiness", "reconciliation", "diagnostics"):
            with open(Path(_ART["_dir"]) / f"{name}.tsv",
                      encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                fields = set(reader.fieldnames)
                for row in reader:
                    self.assertEqual({k for k in row if k is not None},
                                     fields, name)
                    self.assertNotIn(None, row.values(), name)

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_summary_schema_and_consistency(self):
        s = _ART["summary"]
        self.assertEqual(s["schema"], "burstiness-heaps/1")
        self.assertEqual(s["screens"]["growth_stream_n"], 74_318)
        self.assertEqual(s["screens"]["entity_stream_n"], 74_077)
        self.assertEqual(s["screens"]["screen_delta"], 241)
        self.assertTrue(s["heaps"]["book_units_in_english_range"])
        self.assertLess(s["heaps"]["entity_layer_beta"],
                        s["heaps"]["book_units_beta"])
        b = s["burstiness"]
        self.assertEqual(b["bursty"] + b["even"] + b["neutral"],
                         b["entities_tested"])
        self.assertEqual(b["entities_tested"], len(rows("entity_burstiness")))
        # the control rows are the artifact's rows (as parsed strings)
        self.assertEqual(len(b["controls"]), len(rows("controls_burstiness")))
        self.assertEqual([c["term"] for c in b["controls"]],
                         [r["term"] for r in rows("controls_burstiness")])
        self.assertEqual(s["reconciliation_mismatches"], [])

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_report_headlines_equal_the_artifacts(self):
        report = _ART["report"]
        s = _ART["summary"]
        m = re.search(r"Heaps' beta \*\*([0-9.]+)\*\*", report)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1),
                         f"{s['heaps']['book_units_beta']:.3f}")
        m = re.search(r"\*\*(\d+) are bursty\*\*", report)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), s["burstiness"]["bursty"])
        m = re.search(r"entity median is F\(1000\) ([0-9]+\.[0-9]+)", report)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1),
                         f"{s['burstiness']['entity_median_fano_1000']:.2f}")
        # the cast-growth thirds must sum to the glossary size
        thirds = s["growth"]["entity_first_attestation_by_stream_third"]
        self.assertEqual(sum(thirds), 98)
        for section in ("## Verdict", "## 1. Heaps' law", "## 2. The entity",
                        "## 3. Burstiness", "## 4. Controls", "## 5. Limits"):
            self.assertIn(section, report)

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_burstiness_table_is_sorted_and_verdicts_consistent(self):
        fanos = [float(r["fano_1000"]) for r in rows("entity_burstiness")]
        self.assertEqual(fanos, sorted(fanos, reverse=True))
        nulls_used = _ART["summary"]["burstiness"]["nulls"]
        p_floor = 1 / (nulls_used + 1)
        for r in rows("entity_burstiness"):
            self.assertGreaterEqual(int(r["n"]), bh.MIN_N)
            self.assertIn(r["verdict"], ("bursty", "even", "neutral", "untested"))
            p = float(r["p_bursty"])
            self.assertGreaterEqual(p, p_floor - 1e-12)
            self.assertLessEqual(p, 1.0)
            if r["verdict"] == "bursty":
                self.assertGreater(float(r["burstiness_b"]), bh.B_LEVEL)
                self.assertLessEqual(p, bh.P_LEVEL + 1e-12)
            if r["verdict"] == "even":
                self.assertLess(float(r["fano_1000"]), 0.9)

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_null_fano_is_the_bernoulli_anchor(self):
        # the relocation null must sit on 1.0, not on the entity's value
        for r in rows("entity_burstiness")[:5]:
            self.assertAlmostEqual(float(r["null_fano_median"]), 1.0,
                                   delta=0.15, msg=r["entity"])

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_growth_curve_per_corpus_is_monotone(self):
        by_corpus = {}
        for r in rows("growth_curve"):
            by_corpus.setdefault(r["corpus"], []).append(
                (int(r["N"]), float(r["V"])))
        self.assertIn("book_units", by_corpus)
        self.assertIn("en_mean", by_corpus)
        self.assertIn("entity_layer", by_corpus)
        for corpus, pts in by_corpus.items():
            vs = [v for _n, v in pts]
            ns = [n for n, _v in pts]
            self.assertEqual(vs, sorted(vs), corpus)
            self.assertEqual(ns, sorted(ns), corpus)
        # the book's curve must end at #24's published V
        self.assertEqual(by_corpus["book_units"][-1], (74_318, 5_868))
        # the English mean curve is an average, so it can be fractional
        self.assertTrue(all(v == int(v) for _n, v in
                            by_corpus["book_units"]))
        self.assertTrue(any(v != int(v) for _n, v in by_corpus["en_mean"]))

    @unittest.skipUnless(HAVE_ARTIFACTS, "artifacts missing")
    def test_book_only_path_runs_without_references(self):
        tmp = tempfile.mkdtemp(prefix="burstiness_noref_")
        rc = bh.main(["--quick", "--out", tmp, "--no-reference", "--quiet"])
        self.assertEqual(rc, 0)
        with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
            s = json.load(fh)
        self.assertIsNone(s["heaps"]["english_mean_beta"])
        self.assertIsNone(s["heaps"]["book_units_in_english_range"])
        report = (Path(tmp) / "burstiness_report.md").read_text(encoding="utf-8")
        self.assertIn("no English comparison loaded", report)
        self.assertEqual(s["reconciliation_mismatches"], [])
        # the entity legs do not depend on the English comparison
        self.assertGreater(s["burstiness"]["entities_tested"], 0)


class TestNullsHaveTeeth(unittest.TestCase):
    """The relocation null must separate clustered from even use."""

    def test_clustered_positions_are_caught(self):
        n_total = 60_000
        clustered = [i for i in range(0, 30_000, 3)]        # one dense half
        nulls = bh.null_fanos(len(clustered), 1000, n_total,
                              bh.entity_seed("test"), 200)
        p = bh.burst_p_value(bh.fano_factor(clustered, 1000, n_total), nulls)
        self.assertLessEqual(p, 0.05)

    def test_even_positions_are_not_caught(self):
        n_total = 60_000
        even = list(range(0, n_total, n_total // 300))
        nulls = bh.null_fanos(len(even), 1000, n_total,
                              bh.entity_seed("test2"), 200)
        p = bh.burst_p_value(bh.fano_factor(even, 1000, n_total), nulls)
        self.assertGreater(p, 0.5)

    def test_synthetic_burst_profile_splits(self):
        # three regimes, one process each: clockwork, memoryless, clustered
        n_total = 60_000
        clock = list(range(0, n_total, 200))                 # B -> -1
        rng = random.Random(3)
        memoryless = sorted(rng.choices(range(n_total), k=300))  # B -> ~0
        clustered = []
        for start in (0, 20_000, 40_000):
            clustered.extend(range(start, start + 2_000, 10))
        clock_b = bh.burstiness([b - a for a, b in
                                 itertools.pairwise(clock)])
        mem_b = bh.burstiness([b - a for a, b in
                               itertools.pairwise(memoryless)])
        clu_b = bh.burstiness([b - a for a, b in
                               itertools.pairwise(clustered)])
        self.assertLess(clock_b, mem_b)
        self.assertLess(mem_b, clu_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
