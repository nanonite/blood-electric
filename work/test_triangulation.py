#!/usr/bin/env python3
"""Regression suite for #73 triangulation / convergence report.

Builds the artifacts into a temp directory from the frozen #29 / #39 / #40 /
#41 / #51 tables and checks the invariants that matter:

  * every method is a partition of the same 5,614 clauses and reconciles with
    its source analysis (391 nodes / 6 communities, k=5/6/22 topics, 1,362
    BERTopic noise clauses, 12 embedding clusters summing to 5,614);
  * the re-derived embedding partition matches #40's published catalogue
    (label overlap, sizes, and per-motif containment);
  * the agreement table covers all ten method pairs with ARI/NMI in range and
    permutation p-values inside their bounds;
  * the convergence matrix covers every motif and every #29 domain, with
    purities and Jaccards in [0, 1] and a verdict from the fixed vocabulary;
  * the motif counts reconcile with #40's own inventory;
  * the periodicity rows are complete and no motif series is called periodic
    (the #51 negative result must reproduce, or the build has changed);
  * two builds produce byte-identical TSVs (determinism);
  * the statistics themselves behave on synthetic inputs (ARI/NMI, ACF,
    periodogram peak, flatness, k-means, verdict thresholds).

    python work/test_triangulation.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import triangulation as tri

PERMUTATIONS = 60
SHUFFLES = 25
_A = {}


def setUpModule():
    tmp = Path(tempfile.mkdtemp(prefix="triangulation_test_"))
    tri.build(tmp, permutations=PERMUTATIONS, shuffles=SHUFFLES, quiet=True)
    _A["dir"] = tmp
    _A["summary"] = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
    _A["matrix"] = tri.read_tsv(tmp / "convergence_matrix.tsv")
    _A["domains"] = tri.read_tsv(tmp / "domain_convergence.tsv")
    _A["inventory"] = tri.read_tsv(tmp / "motif_inventory.tsv")
    _A["cells"] = tri.read_tsv(tmp / "motif_method_cells.tsv")
    _A["agree"] = tri.read_tsv(tmp / "method_agreement.tsv")
    _A["clause_agree"] = tri.read_tsv(tmp / "clause_agreement.tsv")
    _A["validation"] = tri.read_tsv(tmp / "cluster_validation.tsv")
    _A["motif_recon"] = tri.read_tsv(tmp / "motif_count_reconciliation.tsv")
    _A["per"] = tri.read_tsv(tmp / "periodicity_motifs.tsv")
    _A["recon"] = tri.read_tsv(tmp / "reconciliation.tsv")
    _A["diag"] = {r["metric"]: r["value"]
                  for r in tri.read_tsv(tmp / "diagnostics.tsv")}
    _A["report"] = (tmp / "triangulation_report.md").read_text(encoding="utf-8")
    _A["matrix_md"] = (tmp / "convergence_matrix.md").read_text(encoding="utf-8")


def recon(quantity):
    for row in _A["recon"]:
        if row["quantity"] == quantity:
            return row
    raise AssertionError(f"no reconciliation row {quantity!r}")


def cell(motif, method):
    for row in _A["cells"]:
        if row["motif"] == motif and row["method"] == method:
            return row
    raise AssertionError(f"no cell {motif}/{method}")


class TestSchema(unittest.TestCase):
    def test_schema_version(self):
        self.assertEqual(_A["summary"]["schema"], tri.SCHEMA)

    def test_seed_recorded(self):
        self.assertEqual(_A["summary"]["seed"], tri.DEFAULT_SEED)
        self.assertEqual(_A["summary"]["permutations"], PERMUTATIONS)
        self.assertEqual(_A["summary"]["shuffles_per_mode"], SHUFFLES)

    def test_no_wallclock_in_artifacts(self):
        for name in ("triangulation_report.md", "convergence_matrix.md"):
            text = (_A["dir"] / name).read_text(encoding="utf-8")
            self.assertNotIn("20", name.split(".")[0])   # no date in the file name
            self.assertNotIn("T00:00", text)
            self.assertNotIn("T22:", text)


class TestReconciliation(unittest.TestCase):
    def test_clause_and_page_counts(self):
        self.assertEqual(recon("clauses")["here"], "5614")
        self.assertEqual(recon("clauses")["earlier"], "5614")
        self.assertEqual(recon("pages carrying clauses")["here"], "225")
        self.assertEqual(_A["summary"]["n_clauses"], 5614)

    def test_network_partition(self):
        self.assertEqual(recon("network nodes")["here"], "391")
        self.assertEqual(recon("network communities")["here"], "6")
        assigned = int(recon("clauses with a network community")["here"])
        self.assertLess(assigned, 5614)
        self.assertGreater(assigned, 0.9 * 5614)

    def test_topic_partitions(self):
        self.assertEqual(recon("LDA topics")["here"], "5")
        self.assertEqual(recon("NMF topics")["here"], "6")
        self.assertEqual(recon("BERTopic-style topics")["here"], "22")
        self.assertEqual(recon("BERTopic noise clauses")["here"], "1362")

    def test_embedding_partition_is_a_full_partition(self):
        self.assertEqual(recon("embedding clusters (re-derived)")["here"], "12")
        self.assertEqual(recon("k-means clause sizes sum")["here"], "5614")

    def test_claim_counts_are_declared(self):
        self.assertIn("motif / domain rows in the matrix",
                      {row["quantity"] for row in _A["recon"]})
        self.assertEqual(recon("external corpora consulted")["here"], "0")


class TestMethodAgreement(unittest.TestCase):
    def test_all_ten_pairs_present(self):
        seen = {(r["method_a"], r["method_b"]) for r in _A["agree"]}
        self.assertEqual(len(seen), len(tri.PAIRS))
        self.assertEqual(seen, set(tri.PAIRS))

    def test_statistics_in_range(self):
        for row in _A["agree"]:
            self.assertGreaterEqual(float(row["adjusted_rand"]), -1.0)
            self.assertLessEqual(float(row["adjusted_rand"]), 1.0)
            self.assertGreaterEqual(float(row["nmi"]), 0.0)
            self.assertLessEqual(float(row["nmi"]), 1.0)
            self.assertGreater(int(row["n_shared_clauses"]), 0)
            self.assertLessEqual(int(row["n_shared_clauses"]), 5614)

    def test_permutation_p_in_bounds(self):
        floor = 1.0 / (PERMUTATIONS + 1)
        for row in _A["agree"]:
            for key in ("p_permutation_ari", "p_permutation_nmi"):
                p = float(row[key])
                # the TSV rounds to 5 decimals, hence the 1e-4 tolerance
                self.assertGreaterEqual(p, floor - 1e-4)
                self.assertLessEqual(p, 1.0)

    def test_every_pair_above_chance(self):
        # the book really is structured, so no pair may look like its null
        for row in _A["agree"]:
            self.assertEqual(
                row["above_chance"], "1",
                f"{row['method_a']}|{row['method_b']} is not above its null")
            self.assertGreater(float(row["adjusted_rand"]),
                               float(row["ari_null_mean"]))

    def test_clause_level_coverage_disclosed(self):
        by_method = {r["method"]: r for r in _A["clause_agree"]}
        self.assertEqual(len(by_method), len(tri.METHOD_KEYS))
        for method, row in by_method.items():
            assigned = int(row["n_assigned"])
            self.assertEqual(assigned + int(row["unassigned_clauses"]),
                             int(row["n_clauses"]))
            if method == "bertopic":
                self.assertEqual(int(row["unassigned_clauses"]), 1362)
            else:
                self.assertGreater(float(row["coverage"]), 0.9)


class TestEmbeddingValidation(unittest.TestCase):
    def test_every_derived_cluster_matched(self):
        self.assertEqual(len(_A["validation"]), tri.KMEANS_K)
        for row in _A["validation"]:
            self.assertNotEqual(row["matched_published_cluster"], "")
            self.assertGreater(float(row["label_jaccard"]), 0.0)

    def test_sizes_close_to_published(self):
        for row in _A["validation"]:
            ratio = float(row["size_ratio"])
            self.assertGreater(ratio, 0.8, f"cluster {row['derived_cluster']} far from #40")
            self.assertLess(ratio, 1.2, f"cluster {row['derived_cluster']} far from #40")

    def test_motif_containment_reproduces_40(self):
        checked = 0
        for row in _A["motif_recon"]:
            if int(row["n_clauses_40"]) == 0:
                continue
            checked += 1
            self.assertEqual(row["same_top_cluster_as_40"], "1",
                             f"motif {row['motif']} landed in a different #40 cluster")
            self.assertLess(
                abs(float(row["purity_kmeans_73"])
                    - float(row["frac_in_top_cluster_40"])), 0.25)
        self.assertGreaterEqual(checked, 15)


class TestConvergenceMatrix(unittest.TestCase):
    def test_row_counts(self):
        motifs = [r for r in _A["matrix"] if r["kind"] == "motif"]
        domains = [r for r in _A["matrix"] if r["kind"] == "domain"]
        self.assertEqual(len(motifs), len(tri.MOTIFS))
        self.assertEqual(len(domains), 6)
        self.assertEqual(len(_A["domains"]), 6)
        self.assertEqual(len(_A["inventory"]), len(_A["matrix"]))

    def test_every_row_has_every_method_cell(self):
        for row in _A["matrix"]:
            for key in tri.METHOD_KEYS:
                purity = row[f"pur_{key}"]
                if int(row["n_clauses"]) == 0:
                    self.assertEqual(purity, "")       # zero-occurrence control
                    continue
                self.assertGreaterEqual(float(purity), 0.0)
                self.assertLessEqual(float(purity), 1.0)
                self.assertNotEqual(row[f"grp_{key}"], "")
            for a, b in tri.PAIRS:
                value = row[f"jac_{a}_{b}"]
                if int(row["n_clauses"]) == 0:
                    continue
                self.assertGreaterEqual(float(value), 0.0)
                self.assertLessEqual(float(value), 1.0)

    def test_verdict_vocabulary(self):
        for row in _A["matrix"]:
            self.assertIn(row["verdict"],
                          ("convergent", "partial", "divergent", "untestable"))
            self.assertIn(row["per_verdict"],
                          ("periodic", "periodic(full-band)", "drift-only",
                           "not-periodic", "no occurrences", "degenerate",
                           "untested"))

    def test_cell_count(self):
        self.assertEqual(len(_A["cells"]), len(_A["matrix"]) * len(tri.METHOD_KEYS))
        for row in _A["cells"]:
            self.assertIn(row["source_issue"], ("#29", "#40", "#41"))
            if row["n_in_group"]:
                self.assertLessEqual(int(row["n_in_group"]),
                                     int(row["n_clauses"]))

    def test_verdict_counts_reconcile_with_the_row_split(self):
        counts = _A["summary"]["verdict_counts"]
        for kind in ("motif", "domain", "all"):
            self.assertEqual(sum(counts[kind].values()),
                             len([r for r in _A["matrix"]
                                  if kind == "all" or r["kind"] == kind]))
        for verdict in tri.VERDICTS:
            self.assertEqual(counts["all"][verdict],
                             counts["motif"][verdict] + counts["domain"][verdict])
        self.assertEqual(sum(counts["motif"].values()), len(_A["matrix"]) - 6)
        self.assertEqual(sum(counts["domain"].values()), 6)
        # #29's named domains are the documented case: all six diverge
        self.assertEqual(counts["domain"]["divergent"], 6)
        self.assertEqual(counts["domain"]["convergent"], 0)

    def test_headline_counts_are_motif_only(self):
        # the §1 sentence must quote the motif counts, never the all-rows
        # counts.  The two differ in `divergent`, because #29's six domain
        # rows all diverge -- so that is the number to pin.
        counts = _A["summary"]["verdict_counts"]
        n_motifs = len(_A["matrix"]) - 6
        m = counts["motif"]
        a = counts["all"]
        d = counts["domain"]
        self.assertNotEqual(m["divergent"], a["divergent"])

        def headline(c, n):
            return (f"Of {n} motif rows, {c['convergent']} are **convergent** "
                    f"(mean pairwise Jaccard >= {tri.JACCARD_STRONG:.2f}), "
                    f"{c['partial']} **partial**, {c['divergent']} "
                    f"**divergent**")

        self.assertIn(headline(m, n_motifs) + f" and {m['untestable']} untestable",
                      _A["report"])
        self.assertNotIn(headline(a, n_motifs), _A["report"])
        self.assertIn(f"These counts are the {n_motifs} motif rows only.",
                      _A["report"])
        # the domain split is stated too, so neither count is quoted alone
        self.assertIn(
            f"{d['convergent']} convergent, {d['partial']} partial, "
            f"{d['divergent']} divergent, {d['untestable']} untestable, for "
            f"{len(_A['matrix'])} matrix rows in total",
            _A["report"])

    def test_verdict_thresholds_are_respected(self):
        for row in _A["matrix"]:
            mean_j = row["mean_pair_jaccard"]
            n = int(row["n_clauses"])
            if row["verdict"] == "convergent":
                self.assertGreaterEqual(float(mean_j), tri.JACCARD_STRONG)
            elif row["verdict"] == "divergent":
                self.assertLess(float(mean_j), tri.JACCARD_WEAK)
            elif row["verdict"] == "untestable":
                self.assertLess(n, tri.MIN_CLAUSES)

    def test_known_motifs_land_in_the_expected_buckets(self):
        # the report's headline claim, pinned: #41's five poles are the
        # motifs the matrix calls convergent
        converged = {r["motif"] for r in _A["matrix"]
                     if r["kind"] == "motif" and r["verdict"] == "convergent"}
        for motif in ("ice nebula", "artificial sun", "drug embryo"):
            self.assertIn(motif, converged)
        # and the zero-occurrence control is untestable, never converged
        control = next(r for r in _A["matrix"] if r["motif"] == "necropolis")
        self.assertEqual(control["verdict"], "untestable")
        self.assertEqual(control["n_clauses"], "0")

    def test_matrix_markdown_has_the_promised_tables(self):
        text = _A["matrix_md"]
        self.assertIn("## Purity and verdict", text)
        self.assertIn("## Dominant bucket per method", text)
        self.assertIn("## Pairwise Jaccard of the dominant buckets", text)
        for key in tri.METHOD_KEYS:
            self.assertIn("p:" + key, text)


class TestPeriodicity(unittest.TestCase):
    def test_one_row_per_matrix_row(self):
        self.assertEqual(len(_A["per"]), len(_A["matrix"]))
        self.assertEqual({r["motif"] for r in _A["per"]},
                         {r["motif"] for r in _A["matrix"]})

    def test_bounds(self):
        floor = 1.0 / (SHUFFLES + 1)
        for row in _A["per"]:
            for key in ("p_global", "p_block", "p_detrend", "p_block_short",
                        "p_detrend_short", "q_global", "q_block", "q_detrend",
                        "q_global_short", "q_block_short", "q_detrend_short"):
                if row[key] == "":
                    continue
                p = float(row[key])
                # TSVs round to 5 decimals, hence the 1e-4 tolerance
                self.assertGreaterEqual(p, floor - 1e-4)
                self.assertLessEqual(p, 1.0)
            if row["flatness"] != "":
                self.assertGreater(float(row["flatness"]), 0.0)
                self.assertLess(float(row["flatness"]), 1.0)
            if row["gap_cv"] != "":
                self.assertGreaterEqual(float(row["gap_cv"]), 0.0)

    def test_no_motif_is_periodic(self):
        # #51's negative result must reproduce per motif; if this fires, the
        # verdict logic or the series changed and the report has to be rewritten
        periodic = [r["motif"] for r in _A["per"]
                    if r["verdict"].startswith("periodic")]
        self.assertEqual(periodic, [])

    def test_zero_occurrence_control(self):
        row = next(r for r in _A["per"] if r["motif"] == "necropolis")
        self.assertEqual(row["verdict"], "no occurrences")
        self.assertEqual(row["n_pages_present"], "0")

    def test_drift_only_rows_have_a_long_period_peak(self):
        for row in _A["per"]:
            if row["verdict"] == "drift-only":
                self.assertGreaterEqual(float(row["peak_period_pages"]), 2.0)

    def test_book_context_is_cited_not_recomputed(self):
        rows = tri.read_tsv(_A["dir"] / "periodicity_book_context.tsv")
        series = {r["series"] for r in rows}
        for name in ("tok_op", "tok_op_screened", "clause_len"):
            self.assertIn(name, series)
        screened = next(r for r in rows if r["series"] == "tok_op_screened")
        self.assertEqual(screened["verdict_51"], "no cycle (white)")


class TestReport(unittest.TestCase):
    def test_sections_present(self):
        for heading in ("## 1. Verdict", "## 2. What is compared",
                        "## 4. Clause-level method agreement",
                        "## 6. The convergence matrix", "### 6.1 Agreement",
                        "### 6.2 Divergence", "## 7. Periodicity",
                        "## 8. Domains", "## 10. Negative results"):
            self.assertIn(heading, _A["report"])

    def test_all_four_methods_named(self):
        for name in ("network community", "topic model", "embedding cluster",
                     "periodicity"):
            self.assertIn(name, _A["report"])

    def test_source_analyses_cited(self):
        for issue in ("#29", "#40", "#41", "#51"):
            self.assertIn(issue, _A["report"])

    def test_scope_is_stated(self):
        self.assertIn("Only analyses *of this book*", _A["report"])
        self.assertIn("No reference corpus", _A["report"])

    def test_no_unrendered_none_leaks_into_the_report(self):
        # an order-sensitive summary lookup once rendered "net|lda None" here
        for text in (_A["report"], _A["matrix_md"]):
            self.assertNotIn("None", text)

    def test_crosscheck_cites_the_real_clause_level_figure(self):
        net_lda = next(float(r["adjusted_rand"]) for r in _A["agree"]
                       if {r["method_a"], r["method_b"]} == {"net", "lda"})
        self.assertIn(f"(net|lda {net_lda} and its neighbours)", _A["report"])

    def test_pair_ari_is_order_insensitive(self):
        summary = _A["summary"]
        forward = tri.pair_ari(summary, "net", "lda")
        backward = tri.pair_ari(summary, "lda", "net")
        self.assertIsNotNone(forward)
        self.assertEqual(forward, backward)
        with self.assertRaises(KeyError):
            tri.pair_ari(summary, "net", "not-a-method")

    def test_no_ground_truth_claim(self):
        self.assertIn("no method here is a ground truth", _A["report"])
        for banned in ("ground truth translation", "the correct translation",
                       "true meaning of"):
            self.assertNotIn(banned, _A["report"])

    def test_agreement_and_divergence_both_reported(self):
        self.assertIn("### 6.1 Agreement", _A["report"])
        self.assertIn("### 6.2 Divergence (leads)", _A["report"])


class TestDeterminism(unittest.TestCase):
    def test_two_builds_identical(self):
        tmp = Path(tempfile.mkdtemp(prefix="triangulation_test2_"))
        tri.build(tmp, permutations=PERMUTATIONS, shuffles=SHUFFLES, quiet=True)
        for name in sorted(p.name for p in _A["dir"].glob("*.tsv")):
            a = (_A["dir"] / name).read_text(encoding="utf-8")
            b = (tmp / name).read_text(encoding="utf-8")
            self.assertEqual(a, b, f"{name} differs between builds")
        self.assertEqual((_A["dir"] / "summary.json").read_text(encoding="utf-8"),
                         (tmp / "summary.json").read_text(encoding="utf-8"))


class TestStatistics(unittest.TestCase):
    """The estimators on synthetic inputs where the answer is known."""

    def test_ari_identical_partitions(self):
        a = tri.encode([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(tri.adjusted_rand(a, a, 3, 3), 1.0, places=9)

    def test_ari_permutation_invariant(self):
        a = tri.encode([0, 0, 1, 1, 2, 2])
        b = tri.encode([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(tri.adjusted_rand(a, b, 3, 3), 1.0, places=9)

    def test_ari_and_nmi_bounds_on_random_partitions(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.integers(0, 5, 200)
            b = rng.integers(0, 6, 200)
            ari = tri.adjusted_rand(a, b, 5, 6)
            nmi = tri.normalised_mi(a, b, 5, 6)
            self.assertGreaterEqual(ari, -1.0)
            self.assertLessEqual(ari, 1.0)
            self.assertGreaterEqual(nmi, 0.0)
            self.assertLessEqual(nmi, 1.0)

    def test_autocorrelation_recovers_a_period(self):
        t = np.arange(512)
        x = np.sin(2 * math.pi * t / 32.0)
        acf = tri.autocorrelation(x, 64)
        # the biased estimator loses (n - lag)/n, so the peak is near 1
        self.assertGreater(acf[32], 0.9)
        self.assertGreater(acf[32], acf[31])
        # half a period later the sine is in anti-phase, so r(16) ~ -1
        self.assertLess(acf[16], -0.9)

    def test_periodogram_peak_period(self):
        t = np.arange(256)
        x = np.sin(2 * math.pi * t / 16.0)
        freqs, power = tri.periodogram(x)
        idx = tri.band_slice(freqs, power, None)
        rel, period = tri.peak_rel(power, freqs, idx)
        self.assertAlmostEqual(period, 16.0, delta=0.6)
        self.assertGreater(rel, 10.0)

    def test_short_band_excludes_the_drift_peak(self):
        drift = np.linspace(0, 5, 225) ** 2          # slow rate change only
        freqs, power = tri.periodogram(drift)
        full = tri.peak_rel(power, freqs, tri.band_slice(freqs, power, None))
        short = tri.peak_rel(power, freqs,
                             tri.band_slice(freqs, power, 1.0 / tri.SHORT_PERIOD))
        self.assertGreater(full[1], tri.SHORT_PERIOD)     # drift sits at the end
        self.assertLessEqual(short[1], tri.SHORT_PERIOD + 1e-6)

    def test_local_line_statistic_ignores_the_envelope(self):
        # a heterogeneous (not cyclic) page-rate series must not pass the
        # local line test, while a planted line must pass it easily
        rng = np.random.default_rng(11)
        t = np.arange(225)
        envelope = np.exp(-t / 40.0) + 0.1
        heterogeneous = (rng.random(225) < envelope / envelope.max()).astype(float)
        planted = (np.sin(2 * math.pi * t / 13.0) > 0).astype(float)
        for series, lo, hi in ((heterogeneous, 0.0, 40.0), (planted, 100.0, None)):
            freqs, power = tri.periodogram(series)
            idx = tri.band_slice(freqs, power, 1.0 / tri.SHORT_PERIOD)
            line = tri.peak_rel_local(power, idx)
            self.assertIsNotNone(line)
            self.assertGreaterEqual(line, lo)
            if hi is not None:
                self.assertLess(line, hi)

    def test_local_line_statistic_is_blank_safe(self):
        freqs, power = tri.periodogram(np.ones(64))
        idx = tri.band_slice(freqs, power, 1.0 / tri.SHORT_PERIOD)
        self.assertIsNone(tri.peak_rel_local(power, idx))

    def test_spectral_flatness_of_white_noise_near_the_anchor(self):
        rng = np.random.default_rng(1)
        flat = [tri.spectral_flatness(tri.periodogram(rng.normal(size=4096))[1])
                for _ in range(8)]
        for value in flat:
            self.assertGreater(value, 0.35)
            self.assertLess(value, 0.85)
        self.assertGreater(min(flat), tri.WHITE_FLATNESS * 0.7)

    def test_series_stats_flags_a_planted_short_cycle(self):
        t = np.arange(225)
        # a cycle whose period is not a multiple of the 5-page block size, so
        # the block shuffle is a valid null for it
        cyclic = (np.sin(2 * math.pi * t / 13.0) > 0).astype(float)
        stats = tri.series_stats(cyclic, shuffles=60)
        self.assertFalse(stats["degenerate"])
        self.assertEqual(tri.periodicity_verdict(stats), "periodic")
        # gated on the global and detrended nulls, not the block null: block
        # shuffling creates runs and inflates its own null (see series_stats)
        self.assertLess(stats["p_global_short"], 0.05)
        self.assertLess(stats["p_detrend_short"], 0.05)
        self.assertGreaterEqual(stats["p_block_short"], 1.0)
        self.assertAlmostEqual(stats["peak_period_short"], 225 / 17.0,
                               delta=1.5)

    def test_block_shuffle_inflates_its_own_null(self):
        # a perfectly periodic series must NOT be beaten by the block null;
        # if it were, the block shuffle would be a valid gate
        t = np.arange(225)
        cyclic = (np.sin(2 * math.pi * t / 13.0) > 0).astype(float)
        stats = tri.series_stats(cyclic, shuffles=60)
        self.assertEqual(stats["p_block_short"], 1.0)
        self.assertLess(stats["p_global_short"], 0.05)

    def test_series_stats_on_noise_is_not_periodic(self):
        rng = np.random.default_rng(2)
        noisy = (rng.random(225) < 0.3).astype(float)
        stats = tri.series_stats(noisy, shuffles=40)
        self.assertNotEqual(tri.periodicity_verdict(stats), "periodic")

    def test_series_stats_degenerate_input(self):
        self.assertTrue(tri.series_stats(np.zeros(50))["degenerate"])
        self.assertEqual(tri.periodicity_verdict(tri.series_stats(np.zeros(50))),
                         "degenerate")

    def test_shuffle_modes_preserve_length_and_rate(self):
        rng = np.random.default_rng(3)
        x = (rng.random(60) < 0.25).astype(float)
        for mode in ("global", "block", "detrend"):
            y = tri.shuffle_series(x, mode, rng)
            self.assertEqual(y.size, x.size)
        self.assertAlmostEqual(float(np.sort(tri.shuffle_series(
            x, "global", rng)).sum()), float(x.sum()), places=6)

    def test_renewal_stats(self):
        ren = tri.renewal_stats([0, 4, 8, 12])
        self.assertEqual(ren["n_intervals"], 3)
        self.assertAlmostEqual(ren["gap_mean"], 4.0)
        self.assertAlmostEqual(ren["gap_cv"], 0.0, places=9)
        self.assertIsNone(tri.renewal_stats([1]))

    def test_kmeans_separates_blobs(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(200, 4)) + 6.0
        b = rng.normal(size=(200, 4)) - 6.0
        X = np.vstack([a, b]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        labels, _inertia, sizes = tri.kmeans(X, k=2, seed=0, n_init=3,
                                            max_iter=50)
        self.assertEqual(sum(sizes), 400)
        self.assertEqual(sorted(sizes), [200, 200])
        self.assertEqual(len({int(labels[i]) for i in range(200)}), 1)
        self.assertEqual(len({int(labels[i]) for i in range(200, 400)}), 1)
        self.assertNotEqual(int(labels[0]), int(labels[200]))

    def test_verdict_thresholds(self):
        self.assertEqual(tri.verdict_for(100, 0.6)[0], "convergent")
        self.assertEqual(tri.verdict_for(100, 0.4)[0], "partial")
        self.assertEqual(tri.verdict_for(100, 0.1)[0], "divergent")
        self.assertEqual(tri.verdict_for(2, 0.9)[0], "untestable")
        self.assertEqual(tri.verdict_for(100, None)[0], "untestable")

    def test_motif_specs_default_to_their_own_name(self):
        spec = tri.M("adam", "test")
        self.assertEqual(tri.spec_phrases(spec), ((), ("adam",)))
        hits = tri.motif_hits(spec, ["the boy-roid clause", "adam and anti-adam"])
        self.assertEqual(hits, [1])        # substring rule: anti-adam counts too
        self.assertEqual(tri.match_rule(spec), "any of [adam]")
        # a bare motif name can never degenerate into "every clause"
        self.assertEqual(
            tri.motif_hits(tri.M("zzz-neologism", "t"), ["zzz-neologism here"]),
            [0])

    def test_short_cycle_lead_is_disclosed(self):
        for row in _A["per"]:
            if row["motif"] == "necropolis":
                self.assertEqual(row["short_cycle_lead"], "")
                continue
            self.assertIn(row["short_cycle_lead"], ("0", "1"))
            if row["short_cycle_lead"] == "1":
                self.assertLess(float(row["q_global_short"]), 0.05)
                self.assertLess(float(row["q_detrend_short"]), 0.05)

    def test_bh_qvalues_are_monotone_and_bounded(self):
        q = tri.bh_qvalues([0.001, 0.02, 0.5, 1.0])
        self.assertLessEqual(q[0], q[1])
        self.assertLess(q[1], q[2])
        # BH step-up: q_(i) = p_(i) * n / i, made monotone from the top
        for got, want in zip(q, [0.004, 0.04, 2 / 3, 1.0]):
            self.assertAlmostEqual(got, want, places=9)
        for value in q:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(tri.bh_qvalues([None, None]), [1.0, 1.0])

    def test_jaccard_and_dominant_group(self):
        labels_a = [0, 0, 0, 1, 1, 1]
        labels_b = [1, 1, 1, 1, 1, 1]
        members = [0, 1, 2, 3, 4, 5]
        ga, na, pa = tri.dominant_group(members, labels_a)
        gb, nb, pb = tri.dominant_group(members, labels_b)
        self.assertEqual((ga, na, pa), (0, 3, 0.5))     # tie -> lowest id
        self.assertEqual((gb, nb, pb), (1, 6, 1.0))
        self.assertAlmostEqual(
            tri.jaccard(members, labels_a, labels_b, ga, gb), 3 / 6)
        self.assertIsNone(tri.dominant_group([], labels_a)[0])
        self.assertEqual(tri.dominant_group([], labels_a)[1], 0)

    def test_domain_seeds_are_read_from_the_29_builder(self):
        seeds = tri.load_domain_seeds()
        self.assertEqual(sorted(seeds), ["body", "corporate", "genetics",
                                          "machine", "sex", "violence"])
        self.assertIn("embryo", seeds["body"])
        self.assertIn("boy-roid", seeds["genetics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)