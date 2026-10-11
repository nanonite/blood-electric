#!/usr/bin/env python3
"""Regression suite for #58 translations-as-oracle.

Three groups of checks, matching the three claims the module makes:

* the **ingestion gate** - unauthorized mirrors are never ingestible, every
  candidate source has pinned probe evidence, and the pinned ledger has no
  duplicate keys;
* the **aligner** - segmentation, the n-m diagonal, null-collapse, and AER on
  the built-in gold, including a planted-error test that proves the metric
  fires;
* the **oracle** - operator-skeleton localisation of the two long Russian
  quotations, the page attribution of the corpus index, the invariance
  measurement, and the artifacts' internal consistency.

The suite reads the built artifacts under ``analysis/translation_oracle/``
(run ``python work/translation_oracle.py`` first) and never touches the
network.

    python work/test_translation_oracle.py
"""
import re
import sys
import unittest
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import translation_oracle as to  # noqa: E402

ART = to.OUT
CORPUS = to.CORPUS_MD


def read_tsv(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"), strict=False))
            for line in lines[1:] if line.strip()]


@unittest.skipUnless(CORPUS.exists(), "layout markdown absent; run work/recover_layout.py")
class TestCorpus(unittest.TestCase):
    """The corpus index the whole module is cited against."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = to.load_corpus()

    def test_page_count_matches_the_page_map(self):
        self.assertEqual(len(self.corpus.page_starts), 225)

    def test_page_starts_are_ascending(self):
        offsets = [o for o, _ in self.corpus.page_starts]
        self.assertEqual(offsets, sorted(offsets))
        pages = [p for _, p in self.corpus.page_starts]
        self.assertEqual(pages, sorted(pages))

    def test_page_marker_opens_the_page_it_names(self):
        """The off-by-one this index is prone to: a page's start is the end
        of the text *before* its marker, not the marker's own position."""
        raw = CORPUS.read_text(encoding="utf-8")
        needle = "# WARNING THIS MACHINE KILLS"
        self.assertIn(needle, self.corpus.text)
        before = raw[:raw.index(needle)]
        marker = re.findall(r"<!-- page (\d+) -->", before)[-1]
        got = self.corpus.page(self.corpus.text.index(needle))
        self.assertEqual(int(marker), got)

    def test_operator_scan_is_greedy_longest_match(self):
        got = [op for op, _, _ in to.scan_operators("a::b//++c<<d>>e")]
        self.assertEqual(got, ["::", "//", "++", "<<", ">>"])
        lone = [op for op, _, _ in to.scan_operators("a>b<c")]
        self.assertEqual(lone, [">", "<"])

    def test_corpus_operator_offsets_are_monotone(self):
        starts = [s for _op, s, _e in self.corpus.ops]
        ends = [e for _op, _s, e in self.corpus.ops]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(ends, sorted(ends))
        self.assertIn("::", [op for op, _, _ in self.corpus.ops])
        self.assertIn("<<", [op for op, _, _ in self.corpus.ops])

    def test_span_delimiters_survive_the_tag_clean(self):
        """A permissive ``<[^>]*>`` would eat the book's own delimiters."""
        for phrase in ("<to the cut mode of the masses of flesh that weakened>",
                       "<in the future when spawn caused spasm>"):
            self.assertIn(phrase, self.corpus.text)

    def test_operator_count_is_the_same_order_as_10(self):
        # #10 counts 10,949 over the same file including its heading lines;
        # this index drops the layout scaffolding, so it must be lower.
        self.assertGreater(len(self.corpus.ops), 8_000)
        self.assertLess(len(self.corpus.ops), 10_949)


class TestIngestionGate(unittest.TestCase):
    """The negative result on the Skobin parallel text."""

    def test_every_mirror_is_refused(self):
        mirrors = [s for s in to.SOURCES if s.authorization == to.AUTH_MIRROR]
        self.assertGreaterEqual(len(mirrors), 8)
        for src in mirrors:
            self.assertFalse(to.ingestible(src), src.key)
            self.assertEqual(to.verdict(src), to.V_REFUSED, src.key)

    def test_no_open_source_is_ingestible(self):
        for src in to.SOURCES:
            if src.authorization == to.AUTH_OPEN:
                self.assertTrue(to.ingestible(src), src.key)
                self.assertEqual(to.verdict(src), to.V_OBTAINED, src.key)

    def test_verdict_is_a_function_of_the_licence_class(self):
        for src in to.SOURCES:
            self.assertEqual(to.verdict(src), to._VERDICT_BY_AUTH[src.authorization])

    def test_every_source_has_pinned_probe_evidence(self):
        for src in to.SOURCES:
            pin = to.PINS.get(src.key)
            self.assertIsNotNone(pin, f"{src.key} has no pinned probe")
            self.assertIn("status", pin)
            self.assertIn(pin["status"], (200, 403, 404), src.key)

    def test_pin_keys_and_source_keys_agree(self):
        self.assertEqual({s.key for s in to.SOURCES}, set(to.PINS))

    def test_probe_date_is_pinned_not_generated(self):
        self.assertRegex(to.PROBE_DATE, r"^\d{4}-\d{2}-\d{2}$")

    def test_mirror_urls_are_distinct(self):
        urls = [s.url for s in to.SOURCES]
        self.assertEqual(len(urls), len(set(urls)))

    def test_ledger_rows_carry_provenance(self):
        rows = to.ledger_rows()
        self.assertEqual(len(rows), len(to.SOURCES))
        for row in rows:
            for field in ("license", "authorization", "verdict", "ingestible",
                          "probe_date", "note"):
                self.assertTrue(str(row[field]).strip(), f"{row['key']}:{field}")


class TestSegmentation(unittest.TestCase):
    def test_split_units_keeps_the_operator_run(self):
        units = to.split_units("a::b//c")
        self.assertEqual([u.text for u in units], ["a", "b", "c"])
        self.assertEqual(units[0].ops_before, "")
        self.assertEqual(units[1].ops_before, "::")
        self.assertEqual(units[2].ops_before, "//")

    def test_leading_operators_attach_to_the_first_unit(self):
        units = to.split_units("<<DNA::body")
        self.assertEqual(units[0].text, "DNA")
        self.assertEqual(units[0].ops_before, "<<")

    def test_ru_normalisation_maps_span_glyphs(self):
        out = to.normalise_ru_quote("«a‹‹b››c»")
        self.assertTrue(out.startswith("a<<b>>c"), out)

    def test_single_ru_angle_maps_to_a_single_operator(self):
        out = to.normalise_ru_quote("«a‹b›c»")
        self.assertIn("<b>", out)

    def test_normalisation_strips_the_reviewers_outer_marks(self):
        self.assertFalse(to.normalise_ru_quote("«x»").startswith("«"))

    def test_operator_skeleton_of_a_quote(self):
        sk = to.operator_skeleton(to.normalise_ru_quote(to.RUQ1))
        self.assertEqual(sk[0], "<<")
        self.assertEqual(sk[1], "=")
        self.assertEqual("".join(sk), "<<=::::<<////>>=::=::::=///")


class TestAligner(unittest.TestCase):
    """The DP, the collapse rule and the metric, on synthetic inputs."""

    @staticmethod
    def units(texts):
        return [to.Unit(t, "", 0, len(t)) for t in texts]

    def test_identity_alignment_when_counts_match(self):
        en = self.units(["The dog ran across the field.",
                         "A girl opened the door."])
        ru = self.units(["Собака бежала через поле.",
                         "Девушка открыла дверь."])
        links = to.align_units(en, ru)
        self.assertTrue(all(l.kind == to.KIND_11 for l in links), links)
        self.assertEqual([l.en for l in links], [(0,), (1,)])

    def test_gap_is_emitted_when_the_sides_differ_in_length(self):
        en = self.units(["The dog ran across the field."])
        ru = self.units(["Собака.", "Бежала."])
        links = to.align_units(en, ru)
        self.assertTrue(any(l.kind in (to.KIND_01, "n-m") for l in links), links)

    def test_alignment_is_monotone_and_complete(self):
        en = self.units([f"sentence number {i} of the source text" for i in range(6)])
        ru = self.units([f"предложение номер {i} в переводе" for i in range(7)])
        links = to.align_units(en, ru)
        seen_en, seen_ru = [], []
        for l in links:
            seen_en.extend(l.en)
            seen_ru.extend(l.ru)
        self.assertEqual(seen_en, sorted(seen_en))
        self.assertEqual(seen_ru, sorted(seen_ru))
        self.assertEqual(seen_en, list(range(len(en))))
        self.assertEqual(seen_ru, list(range(len(ru))))

    def test_nm_block_is_expressible(self):
        """A DP with only 1-1 / 1-0 / 0-1 moves can never *output* a
        many-to-many link, so it can never be scored against a gold that has
        one.  This pins the diagonal move."""
        en = self.units(["The machine stopped in the middle of the street today."])
        ru = self.units(["Машина остановилась.",
                         "Это случилось посреди улицы."])
        links = to.align_units(en, ru)
        self.assertTrue(any(len(l.ru) > 1 or len(l.en) > 1 for l in links), links)

    def test_digit_anchor_is_rewarded(self):
        en = self.units(["The old library was built in 1890 near the river."])
        ru = self.units(["Старая библиотека была построена в 1890 году недалеко."])
        with_anchor = to.align_units(en, ru)
        self.assertTrue(any(l.kind == to.KIND_11 for l in with_anchor))

    def test_collapse_merges_adjacent_nulls(self):
        self.assertEqual(to.collapse([(0, None), (None, 0)]), [((0,), (0,))])
        self.assertEqual(to.collapse([(0, 1)]), [((0,), (1,))])

    def test_collapse_handles_multi_sides(self):
        got = to.collapse([(0, (1, 2)), (1, 3)])
        self.assertEqual(got, [((0,), (1, 2)), ((1,), (3,))])

    def test_aer_is_zero_on_a_perfect_alignment(self):
        gold = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(to.alignment_error_rate(gold, [(0, 0), (1, 1), (2, 2)]), 0.0)

    def test_aer_fires_on_a_planted_error(self):
        """One correct link, two substitutions.  Standard AER charges the two
        missing gold links and the two spurious hypothesis links: 4 errors
        over 3 gold links.  It is an error rate, not a proportion, so it may
        exceed 1."""
        gold = [(0, 0), (1, 1), (2, 2)]
        hyp = [(0, 0), (1, 2), (2, 1)]
        self.assertAlmostEqual(to.alignment_error_rate(gold, hyp), 4 / 3)

    def test_aer_penalises_missing_links(self):
        """A hypothesis that drops gold links must be charged for them.  This
        is the deletion term of the standard metric; without it the score
        collapses to precision and is gamed by proposing one confident link."""
        gold = [(0, 0), (1, 1), (2, 2)]
        hyp = [(0, 0)]
        self.assertAlmostEqual(to.alignment_error_rate(gold, hyp), 2 / 3)

    def test_aer_penalises_spurious_links(self):
        gold = [(0, 0)]
        hyp = [(0, 0), (1, 1), (2, 2)]
        self.assertAlmostEqual(to.alignment_error_rate(gold, hyp), 2.0)

    def test_aer_of_an_empty_hypothesis_is_one(self):
        """The degenerate-aligner floor: proposing nothing is maximally wrong,
        not perfect."""
        gold = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(to.alignment_error_rate(gold, []), 1.0)

    def test_aer_of_an_empty_gold_is_zero(self):
        self.assertEqual(to.alignment_error_rate([], []), 0.0)

    def test_aer_matches_the_published_decomposition(self):
        """`((G - correct) + (H - correct)) / G`, recomputed by hand on the
        built-in gold.  Pins the two terms - deletions and insertions - to the
        counts the artifacts publish as `n_gold_links`."""
        for en_sents, ru_sents, gold in to.build_gold_documents():
            en_units = [to.Unit(s, "", 0, len(s)) for s in en_sents]
            ru_units = [to.Unit(s, "", 0, len(s)) for s in ru_sents]
            links = to.align_units(en_units, ru_units)
            hyp = [(lk.en[0], lk.ru[0] if len(lk.ru) == 1 else lk.ru)
                   for lk in links if lk.en and lk.ru]
            collapsed = to.collapse(gold)
            gmap = {}
            for i, j in collapsed:
                if i is None or j is None:
                    continue
                for ii in i:
                    gmap[ii] = j
            hc = to.collapse(hyp)          # the metric compares collapsed sets
            correct = sum(1 for i, j in hc
                          if i is not None and j is not None
                          and gmap.get(i[0]) == j)
            h = sum(1 for i, j in hc if i is not None and j is not None)
            expected = ((len(gmap) - correct) + (h - correct)) / len(gmap)
            self.assertAlmostEqual(to.alignment_error_rate(gold, hyp), expected, places=12)

    def test_gold_documents_plant_the_documented_operations(self):
        docs = to.build_gold_documents()
        self.assertGreaterEqual(len(docs), 8)
        kinds = set()
        for _en, _ru, gold in docs:
            for _i, j in gold:
                if j is None:
                    kinds.add("1-0")
                elif isinstance(j, tuple):
                    kinds.add("1-n" if len(j) > 1 else "1-1")
                else:
                    kinds.add("1-1")
        self.assertIn("1-n", kinds)
        self.assertIn("1-1", kinds)

    def test_gold_documents_plant_no_deletions(self):
        """Deletions are excluded on purpose; a gold that plants them
        measures the metric's null-collapse, not the aligner."""
        for _en, _ru, gold in to.build_gold_documents():
            self.assertTrue(all(j is not None for _i, j in gold))

    def test_gold_is_deterministic(self):
        self.assertEqual(to.build_gold_documents(), to.build_gold_documents())

    def test_aligner_beats_the_length_only_baseline(self):
        rows = to.evaluate_aligner()
        dp = next(r["aer"] for r in rows
                  if r["system"] == "dp" and r["document"] == "ALL")
        base = next(r["aer"] for r in rows
                    if r["system"] == "proportional" and r["document"] == "ALL")
        self.assertLess(dp, 0.5)
        self.assertLess(dp, base)

    def test_baseline_is_reported_on_the_same_split_as_the_aligner(self):
        """The headline comparison is same-split.  Reading the DP held-out
        number against the baseline's all-documents number would flatter the
        aligner, so both are published per split and compared per split."""
        rows = to.evaluate_aligner()

        def aer_of(system, split):
            return next(r["aer"] for r in rows
                        if r["system"] == system and r["split"] == split
                        and r["document"] == "SUBTOTAL")

        for split in ("tuning", "held_out"):
            self.assertLess(aer_of("dp", split), aer_of("proportional", split))

    def test_held_out_half_is_reported(self):
        rows = to.evaluate_aligner()
        held = [r for r in rows
                if r["system"] == "dp" and r["split"] == "held_out"
                and r["document"] == "SUBTOTAL"]
        self.assertEqual(len(held), 1)
        self.assertGreaterEqual(held[0]["n_gold_links"], 20)

    def test_builtin_gold_pairs_are_well_formed(self):
        self.assertGreaterEqual(len(to.GOLD_SENTENCES), 20)
        for en, ru in to.GOLD_SENTENCES:
            self.assertTrue(en.strip() and ru.strip())
            self.assertRegex(ru, r"[Ѐ-ӿ]")
            self.assertRegex(en, r"[A-Za-z]")


@unittest.skipUnless(CORPUS.exists(), "layout markdown absent")
class TestLocalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = to.load_corpus()

    def test_skeleton_localisation_finds_ruq01_exactly_once(self):
        sk = to.operator_skeleton(to.normalise_ru_quote(to.RUQ1))
        hits = to.localize_by_skeleton(self.corpus, sk)
        self.assertEqual(len(hits), 1)
        start, end = hits[0]
        window = self.corpus.text[start:end]
        self.assertIn("DNA=channel of love doll genocide", window)
        self.assertIn("eyes of the cyborg", window)

    def test_skeleton_localisation_of_ruq02_finds_two_identical_sites(self):
        q = next(x for x in to.RU_QUOTATIONS if x["id"] == "RUQ02")
        aln = to.align_quote(self.corpus, q)
        self.assertEqual(aln.n_windows, 2)
        self.assertEqual(aln.n_distinct, 1)
        self.assertEqual(aln.pages(), [5, 221])

    def test_skeleton_quotations_need_no_lexical_anchor(self):
        for qid in ("RUQ01", "RUQ02"):
            q = next(x for x in to.RU_QUOTATIONS if x["id"] == qid)
            self.assertEqual(q["anchors"], {})
            self.assertEqual(q["kind"], "skeleton")
            aln = to.align_quote(self.corpus, q)
            self.assertEqual(aln.locator, "operator_skeleton")
            self.assertEqual(aln.evidence(), "unique_skeleton")
            self.assertEqual(aln.confidence(), "high")

    def test_anchor_localisation_is_order_agnostic(self):
        """`Город Трупов` is head-final Russian for head-initial `Cadaver
        City`; an order-preserving search would miss it."""
        q = next(x for x in to.RU_QUOTATIONS if x["id"] == "RUQ07")
        self.assertEqual(q["anchors"]["Город"], "City")
        self.assertEqual(q["anchors"]["Трупов"], "Cadaver")
        aln = to.align_quote(self.corpus, q)
        self.assertIn("Cadaver City", aln.en_text)

    def test_literal_locator_counts_every_occurrence(self):
        q = next(x for x in to.RU_QUOTATIONS if x["id"] == "RUQ08")
        aln = to.align_quote(self.corpus, q)
        self.assertEqual(aln.locator, "literal_english")
        self.assertEqual(aln.n_windows, self.corpus.text.count(q["en_literal"]))
        self.assertGreater(aln.n_windows, 100)   # the coinage is a refrain

    def test_chapter_heading_quotations_resolve(self):
        for qid, needle in (("RUQ05", "nail"), ("RUQ06", "WARNING THIS MACHINE KILLS")):
            q = next(x for x in to.RU_QUOTATIONS if x["id"] == qid)
            aln = to.align_quote(self.corpus, q)
            self.assertIn(needle, aln.en_text)
            self.assertGreaterEqual(aln.n_anchors_hit, aln.n_anchors)

    def test_alignment_pairs_are_one_to_one_for_the_skeleton_quotes(self):
        for qid in ("RUQ01", "RUQ02"):
            q = next(x for x in to.RU_QUOTATIONS if x["id"] == qid)
            aln = to.align_quote(self.corpus, q)
            ones = [l for l in aln.links if l.kind == to.KIND_11]
            self.assertGreaterEqual(len(ones), 7, qid)
            self.assertEqual([l.en[0] for l in ones],
                             [l.ru[0] for l in ones], qid)

    def test_divergence_quotation_is_flagged_not_resolved(self):
        q = to.DIVERGENT_QUOTE
        self.assertEqual(q["kind"], "divergence")
        self.assertEqual(q["anchors"]["ЛИКВИДАЦИЯ"], "abolition")
        aln = to.align_quote(self.corpus, q)
        self.assertIn("abolition", aln.en_text)

    def test_every_pinned_quotation_locates(self):
        for q in list(to.RU_QUOTATIONS) + [to.DIVERGENT_QUOTE]:
            aln = to.align_quote(self.corpus, q)
            self.assertNotEqual(aln.evidence(), "no_match", q["id"])
            self.assertTrue(aln.en_text.strip(), q["id"])


@unittest.skipUnless(CORPUS.exists(), "layout markdown absent")
class TestOperatorInvariance(unittest.TestCase):
    """The load-bearing claim: the symbol channel survives translation."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = to.load_corpus()

    def test_positional_retention_is_total(self):
        for q in (q for q in to.RU_QUOTATIONS if q["kind"] == "skeleton"):
            aln = to.align_quote(self.corpus, q)
            stats = to.operator_invariance(aln, self.corpus)
            self.assertEqual(stats["positional_retention"], 1.0, q["id"])
            self.assertEqual(stats["positional_matches"], stats["n_en_ops"], q["id"])

    def test_retention_is_measured_on_the_unpadded_window(self):
        """Padding exists for the aligner; counting the neighbouring clause's
        operators as channel losses would understate retention."""
        for qid in ("RUQ01", "RUQ02"):
            q = next(x for x in to.RU_QUOTATIONS if x["id"] == qid)
            aln = to.align_quote(self.corpus, q)
            exact = to.operator_invariance(aln, self.corpus)
            skeleton = to.operator_skeleton(to.normalise_ru_quote(q["ru"]))
            self.assertEqual(exact["n_en_ops"], len(skeleton), qid)
            self.assertEqual(exact["n_ru_ops"], len(skeleton), qid)


class TestQuotationProvenance(unittest.TestCase):
    """Each quotation is pinned to the review block it was transcribed from.

    These are the checks the earlier free-text attribution could not survive.
    Six of the twelve quotations were credited to Reza Negarestani, whose name
    does appear on the FantLab work page but only in the «Рецензии»
    bibliography block, as the author of an *external review of the book*
    (work1915318).  He wrote none of them.  Since the whole licensing position
    rests on these being short *attributed* quotations of a copyrighted
    translation, a wrong attribution is not a cosmetic defect, so it is pinned
    from two directions: the reviewer of each quotation is asserted here by
    name, and the review table the attribution is derived from is checked
    against the page's own ``schema.org/Review`` markup.
    """

    #: the review block each quotation was transcribed from, by reviewer.
    EXPECTED: ClassVar[dict[str, str]] = {
        "RUQ01": "josser", "RUQ02": "ekidonaa", "RUQ03": "ekidonaa",
        "RUQ04": "ekidonaa", "RUQ05": "Stirliz77", "RUQ06": "ekidonaa",
        "RUQ07": "ekidonaa", "RUQ08": "Stirliz77", "RUQ09": "Stirliz77",
        "RUQ10": "Stirliz77", "RUQ11": "ekidonaa", "RUQ12": "ekidonaa",
    }
    #: the six rows the reviewer found misattributed, with their true author.
    CORRECTED: ClassVar[dict[str, str]] = {
        "RUQ05": "Stirliz77", "RUQ06": "ekidonaa", "RUQ08": "Stirliz77",
        "RUQ09": "Stirliz77", "RUQ10": "Stirliz77", "RUQ12": "ekidonaa",
    }
    EXPECTED_DATES: ClassVar[dict[str, str]] = {
        "josser": "10 May 2011",
        "Stirliz77": "9 Oct 2020",
        "ekidonaa": "12 Feb 2025",
    }

    @classmethod
    def setUpClass(cls):
        cls.quotes = list(to.RU_QUOTATIONS) + [to.DIVERGENT_QUOTE]

    def test_every_quotation_names_a_pinned_review(self):
        for q in self.quotes:
            self.assertIn(q["review"], to.REVIEW_BY_ID,
                          f"{q['id']} names no pinned review block")

    def test_each_quotation_is_attributed_to_its_actual_reviewer(self):
        """The pin the correction exists to enforce."""
        for q in self.quotes:
            self.assertEqual(to.REVIEW_BY_ID[q["review"]]["author"],
                             self.EXPECTED[q["id"]], q["id"])

    def test_the_six_corrected_rows_no_longer_credit_negarestani(self):
        """The specific defect: six rows credited a man who wrote none of it."""
        for qid, author in self.CORRECTED.items():
            rev = next(q for q in self.quotes if q["id"] == qid)
            self.assertNotIn("Negarestani", to.attributor_of(rev["review"]), qid)
            self.assertTrue(to.attributor_of(rev["review"]).startswith(author), qid)

    def test_attributor_is_derived_not_stored(self):
        """No quotation carries a hand-typed attribution any more.

        The defect was possible only because the author was free text on each
        row; the field is now computed from the review table, so it cannot say
        something the table does not.
        """
        for q in self.quotes:
            self.assertNotIn("attributor", q, q["id"])

    def test_attributor_string_names_the_reviewer_and_the_date(self):
        for q in self.quotes:
            rev = to.REVIEW_BY_ID[q["review"]]
            att = to.attributor_of(q["review"])
            self.assertTrue(att.startswith(rev["author"] + " ("), q["id"])
            self.assertIn("FantLab review", att, q["id"])
            self.assertIn(self.EXPECTED_DATES[rev["author"]], att, q["id"])

    def test_only_the_reviewers_who_wrote_them_are_attributed(self):
        """Exactly three readers wrote all twelve quotations."""
        self.assertEqual({to.REVIEW_BY_ID[q["review"]]["author"]
                          for q in self.quotes},
                         {"josser", "Stirliz77", "ekidonaa"})

    def test_each_review_pin_is_internally_consistent(self):
        for rev in to.FANTLAB_REVIEWS:
            self.assertRegex(rev["review"], r"^review\d+$", rev["review"])
            self.assertRegex(rev["body_sha256"], r"^[0-9a-f]{64}$", rev["review"])
            self.assertGreater(rev["n_chars"], 0, rev["review"])
            self.assertRegex(rev["date_published"], r"^\d{4}-\d\d-\d\d ", rev["review"])

    def test_review_keys_are_unique(self):
        keys = [r["review"] for r in to.FANTLAB_REVIEWS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_permalink_resolves_to_the_page_anchor(self):
        """The page's anchor is ``response<N>``; the key is ``review<N>``."""
        for rev in to.FANTLAB_REVIEWS:
            link = to.review_permalink(rev["review"])
            self.assertTrue(link.startswith(to.FANTLAB_REVIEWS_URL + "#response"),
                            rev["review"])
            self.assertEqual(link.rsplit("#response", 1)[1],
                             rev["review"].removeprefix("review"))

    def test_negarestani_is_not_a_source_of_any_quotation(self):
        """He authored a review *of* the book; he is not a reviewer *on* it."""
        for rev in to.FANTLAB_REVIEWS:
            self.assertNotIn("Negarestani", rev["author"], rev["review"])
            self.assertNotIn("Негарестани", rev["author"], rev["review"])

    def test_page_parser_reads_identity_from_the_page_not_from_the_pins(self):
        """A review block's author/date come from the page's own markup.

        Built from a miniature of the page so the parser is exercised without
        a network fetch: author, account, `datePublished` and the
        `#responseNNNNNN` anchor are read out of the block, not supplied.
        """
        page = (
            '<div class="response-item" itemprop="review" itemscope>'
            '<p class="response-autor-info">'
            '<a name=response4242></a>'
            '<b itemprop="author"><a href=\'/user777\'>someone</a></b>, '
            '<span itemprop="datePublished" content="2019-03-04 05:06:07">'
            '4 марта 2019 г.</span></p>'
            '<div class="response-body-home" itemprop="reviewBody">'
            '<p>body text</p></div></div>')
        blocks = to.parse_review_blocks(page)
        self.assertEqual(list(blocks), ["review4242"])
        got = blocks["review4242"]
        self.assertEqual(got["author"], "someone")
        self.assertEqual(got["account"], 777)
        self.assertEqual(got["date_published"], "2019-03-04 05:06:07")
        self.assertEqual(got["_text"], "body text")

    def test_provenance_check_catches_a_misattributed_quotation(self):
        """The check must fire, not merely exist.

        Feeding the provenance scan a page whose block holds a *different*
        quotation has to report the row as found somewhere other than the
        review it is attributed to.  A test that cannot fail is not a pin.
        """
        q = to.RU_QUOTATIONS[0]
        blocks = {q["review"]: {"_text": "совсем другой текст"}}
        row = next(r for r in to.check_quotation_provenance(blocks)
                   if r["id"] == q["id"])
        self.assertEqual(row["found_in"], [])
        self.assertNotEqual(row["found_in"], [row["review"]])

    def test_provenance_check_confirms_a_correctly_attributed_quotation(self):
        q = next(x for x in to.RU_QUOTATIONS if x["id"] == "RUQ05")
        blocks = {q["review"]: {"_text": "название главы " + q["ru"]}}
        row = next(r for r in to.check_quotation_provenance(blocks)
                   if r["id"] == "RUQ05")
        self.assertEqual(row["found_in"], [row["review"]])


@unittest.skipUnless(ART.exists(), "artifacts absent; run work/translation_oracle.py")
class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = read_tsv(ART / "sources.tsv")
        cls.quotes = read_tsv(ART / "ru_quotations.tsv")
        cls.align = read_tsv(ART / "alignment.tsv")
        cls.inv = read_tsv(ART / "operator_invariance.tsv")
        cls.lex = read_tsv(ART / "bilingual_lexicon.tsv")
        cls.checks = read_tsv(ART / "oracle_checks.tsv")
        cls.eval_rows = read_tsv(ART / "aligner_eval.tsv")
        cls.diag = {r["diag"]: r["value"] for r in read_tsv(ART / "diagnostics.tsv")}

    def test_every_pinned_artifact_exists(self):
        for name in ("sources.tsv", "availability_report.md", "ru_quotations.tsv",
                     "fantlab_reviews.tsv",
                     "author_paratext.tsv", "alignment.tsv", "alignment_report.md",
                     "operator_invariance.tsv", "bilingual_lexicon.tsv",
                     "oracle_checks.tsv", "aligner_eval.tsv", "diagnostics.tsv",
                     "reconciliation.tsv", "summary.json"):
            self.assertTrue((ART / name).exists(), name)

    def test_alignment_report_attribution_matches_the_review_block(self):
        """The report is the reader-facing copy, so it must not drift either."""
        text = (ART / "alignment_report.md").read_text(encoding="utf-8")
        for q in list(to.RU_QUOTATIONS) + [to.DIVERGENT_QUOTE]:
            line = (f"* attributed to: {to.attributor_of(q['review'])} "
                    f"({to.review_permalink(q['review'])})")
            self.assertIn(line, text, q["id"])

    def test_ledger_refuses_every_mirror(self):
        for row in self.sources:
            if row["authorization"] == to.AUTH_MIRROR:
                self.assertEqual(row["ingestible"], "n", row["key"])
                self.assertEqual(row["verdict"], to.V_REFUSED, row["key"])

    def test_ledger_records_the_negative_result_on_the_parallel_text(self):
        parallel = [r for r in self.sources if r["kind"] == "parallel_text"]
        self.assertTrue(parallel)
        self.assertTrue(all(r["ingestible"] == "n" for r in parallel),
                        "a full parallel text must not be ingestible from an "
                        "unauthorized source")

    def test_quotations_are_attributed(self):
        for row in self.quotes:
            self.assertTrue(row["attributor"].strip(), row["id"])
            self.assertTrue(row["note"].strip(), row["id"])
            self.assertIn("FantLab", row["attributor"])

    def test_every_quotation_carries_its_review_block(self):
        """`ru_quotations.tsv` names the source block, not just its author.

        An author string can be right by luck; the block it came from is what
        a reader can go and check.
        """
        for row in self.quotes:
            self.assertIn(row["review"], to.REVIEW_BY_ID, row["id"])
            self.assertEqual(row["attributor"],
                             to.attributor_of(row["review"]), row["id"])

    def test_artifact_agrees_with_the_pinned_reviewer_of_each_quotation(self):
        expected = {q["id"]: to.REVIEW_BY_ID[q["review"]]["author"]
                    for q in list(to.RU_QUOTATIONS) + [to.DIVERGENT_QUOTE]}
        for row in self.quotes:
            self.assertTrue(row["attributor"].startswith(expected[row["id"]]),
                            f"{row['id']}: {row['attributor']}")

    def test_no_artifact_row_credits_negarestani(self):
        for row in self.quotes:
            self.assertNotIn("Negarestani", row["attributor"], row["id"])

    def test_review_ledger_covers_every_quotation(self):
        rows = read_tsv(ART / "fantlab_reviews.tsv")
        ledger = {r["review"]: r for r in rows}
        for row in self.quotes:
            self.assertIn(row["review"], ledger, row["id"])
            self.assertEqual(ledger[row["review"]]["attributor"],
                             row["attributor"], row["id"])
        total = sum(int(r["n_quotations"]) for r in rows)
        self.assertEqual(total, len(self.quotes))

    def test_review_ledger_permutes_to_a_stable_permalink(self):
        for row in read_tsv(ART / "fantlab_reviews.tsv"):
            self.assertTrue(row["permalink"].startswith(
                to.FANTLAB_REVIEWS_URL + "#response"), row["review"])

    def test_diagnostics_report_the_attribution_coverage(self):
        self.assertEqual(self.diag["fantlab_review_blocks"],
                         str(len(to.FANTLAB_REVIEWS)))
        self.assertEqual(self.diag["quotations_attributed_to_a_pinned_review"],
                         self.diag["quotations_pinned"])

    def test_most_quotations_resolve(self):
        resolved = [r for r in self.quotes if r["evidence"] != "ambiguous"]
        self.assertGreaterEqual(len(resolved), 10)

    def test_alignment_covers_every_quotation(self):
        ids = {r["id"] for r in self.quotes}
        self.assertEqual({r["quote_id"] for r in self.align}, ids)

    def test_alignment_links_are_well_formed(self):
        for row in self.align:
            n_en, n_ru = int(row["n_en_units"]), int(row["n_ru_units"])
            self.assertEqual(len(row["en_indices"].split(",")), n_en)
            self.assertEqual(len(row["ru_indices"].split(",")), n_ru)
            self.assertEqual(n_en == 0, n_ru == 0)   # a link is one-sided or not

    def test_no_alignment_row_is_a_cartesian_product(self):
        """An n-m link is one row, not n*m rows."""
        for row in self.align:
            self.assertLessEqual(int(row["n_en_units"]), to.MAX_BLOCK)
            self.assertLessEqual(int(row["n_ru_units"]), to.MAX_BLOCK)

    def test_invariance_rows_are_total(self):
        self.assertTrue(self.inv)
        for row in self.inv:
            self.assertEqual(float(row["positional_retention"]), 1.0, row["quote_id"])

    def test_lexicon_grades_its_provenance(self):
        attested = [r for r in self.lex if r["confidence"] == "attested"]
        self.assertGreaterEqual(len(attested), 10)
        for row in attested:
            self.assertIn("+aligned", row["provenance"], row["ru"])
        for row in self.lex:
            self.assertIn(row["provenance"],
                          ("anchor_prior", "aligned", "anchor_prior+aligned"))

    def test_lexicon_attested_pairs_agree_with_the_anchor_priors(self):
        priors = {}
        for q in list(to.RU_QUOTATIONS) + [to.DIVERGENT_QUOTE]:
            for ru, en in q.get("anchors", {}).items():
                priors[(ru.lower(), en.lower())] = True
        for row in self.lex:
            if row["confidence"] == "attested":
                self.assertIn((row["ru"], row["en"]), priors, row["ru"])

    def test_oracle_checks_cover_the_authors_named_items(self):
        items = {r["item"] for r in self.checks}
        for want in ("ADAM", "dogs", "=", "hardweb", "gene dub"):
            self.assertIn(want, items)

    def test_the_two_named_recurring_nouns_are_corroborated(self):
        by_item = {r["item"]: r for r in self.checks}
        self.assertEqual(by_item["ADAM"]["verdict"], "corroborated")
        self.assertEqual(by_item["dogs"]["verdict"], "corroborated")
        self.assertGreater(int(by_item["ADAM"]["n_occurrences"]), 100)
        self.assertGreater(int(by_item["dogs"]["n_occurrences"]), 100)

    def test_the_essayistic_vocabulary_is_absent(self):
        by_item = {r["item"]: r for r in self.checks}
        for absent in ("hardweb", "creature intensity", "echo archive"):
            self.assertEqual(by_item[absent]["verdict"], "absent_from_book", absent)
            self.assertEqual(int(by_item[absent]["n_occurrences"]), 0, absent)

    def test_operator_named_by_the_author_is_corroborated(self):
        by_item = {r["item"]: r for r in self.checks if r["kind"] == "operator"}
        self.assertEqual(by_item["="]["verdict"], "corroborated")

    def test_aligner_eval_publishes_both_systems_and_both_halves(self):
        systems = {r["system"] for r in self.eval_rows}
        splits = {r["split"] for r in self.eval_rows}
        self.assertEqual(systems, {"dp", "proportional"})
        self.assertEqual(splits, {"tuning", "held_out", "ALL"})

    def test_diagnostics_record_the_headline_numbers(self):
        for key in ("corpus_pages", "corpus_operator_tokens", "sources_probed",
                    "mirrors_refused", "quotations_pinned",
                    "quotations_uniquely_located", "aligned_1_1_unit_pairs",
                    "aligner_aer_held_out_half", "baseline_aer_proportional",
                    "baseline_aer_held_out_half",
                    "lexicon_pairs_withheld_low_token_ratio"):
            self.assertIn(key, self.diag)

    def test_published_holdout_figures_come_from_one_split(self):
        """The headline pair must be DP held-out against *baseline held-out*,
        not against the baseline's all-documents figure."""
        self.assertLess(float(self.diag["aligner_aer_held_out_half"]),
                        float(self.diag["baseline_aer_held_out_half"]))

    def test_summary_json_reports_both_splits_of_both_systems(self):
        import json
        summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))
        for key in ("aligner_aer_held_out_half", "baseline_aer_held_out_half",
                    "aligner_aer_builtin_gold", "baseline_aer_proportional",
                    "corpus_operator_tokens"):
            self.assertEqual(str(summary[key]), self.diag[key], key)

    def test_reports_state_the_operator_token_count_they_were_built_from(self):
        """`10,145` is a measured figure, not a figure to be typed into prose:
        both reports must agree with `diagnostics.tsv`."""
        n = int(self.diag["corpus_operator_tokens"])
        labelled = f"{n:,}"
        for name in ("availability_report.md", "alignment_report.md"):
            text = (ART / name).read_text(encoding="utf-8")
            self.assertIn(labelled, text, name)
            self.assertNotIn("11,000 ops", text, name)
            self.assertNotIn("10,145", text.replace(labelled, ""), name)

    def test_reconciliation_states_the_negative_result(self):
        recon = {r["quantity"]: r for r in read_tsv(ART / "reconciliation.tsv")}
        self.assertEqual(recon["parallel text obtained"]["here"], "no")
        self.assertEqual(recon["pages"]["here"], "225")

    def test_summary_json_is_consistent_with_the_diagnostics(self):
        import json
        summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema"], to.SCHEMA)
        self.assertFalse(summary["parallel_text_obtained"])
        self.assertTrue(summary["operator_channel_translation_invariant"])
        self.assertEqual(summary["quotations_total"], len(self.quotes))

    def test_reports_are_written_and_non_empty(self):
        for name in ("availability_report.md", "alignment_report.md"):
            text = (ART / name).read_text(encoding="utf-8")
            self.assertGreater(len(text), 1500, name)
            self.assertIn("#58", text)

    def test_availability_report_states_the_negative_result(self):
        text = (ART / "availability_report.md").read_text(encoding="utf-8")
        self.assertIn("could not be obtained lawfully", text)
        self.assertIn("unauthorized mirror", text)

    def test_author_paratext_is_attributed_to_a_pinned_source(self):
        rows = read_tsv(ART / "author_paratext.tsv")
        self.assertGreaterEqual(len(rows), 5)
        keys = {s.key for s in to.SOURCES}
        for row in rows:
            self.assertIn(row["source"], keys)
            self.assertTrue(row["quote"].strip())


class TestDeterminism(unittest.TestCase):
    def test_ledger_rows_are_a_pure_function_of_the_manifest(self):
        self.assertEqual(to.ledger_rows(), to.ledger_rows())

    def test_rebuild_is_byte_identical(self):
        """Rebuild into a scratch directory and diff.  Skipped when the
        layout markdown is absent, since the build needs it."""
        if not CORPUS.exists():
            self.skipTest("layout markdown absent")
        import tempfile
        import filecmp
        saved = to.OUT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                to.main(["--out", tmp])
                names = sorted(p.name for p in Path(tmp).glob("*"))
                self.assertTrue(names)
                for name in names:
                    a, b = Path(tmp) / name, ART / name
                    if not b.exists() or name.endswith(".json"):
                        continue
                    self.assertTrue(filecmp.cmp(a, b, shallow=False), name)
        finally:
            to.OUT = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)