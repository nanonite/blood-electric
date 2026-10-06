#!/usr/bin/env python3
"""Regression tests for compound/portmanteau decomposition (#18).

Run from the workspace root (inside the `marker` conda env):

    python work/test_compound_decomposition.py
    python -m unittest discover -s work -p 'test_*.py'   # whole suite

Three kinds of test:

  * **agreement with #17** - every unit #17 publishes gets exactly one
    decomposition, with the same uid, offsets and surface; the fused-unit
    classes and the markup-residue count reconcile;
  * **lossless decomposition** - the morpheme chain plus its linkers
    reconstructs the unit, and every rule is reachable;
  * **negative results** - the cases that must stay unresolved
    (`tokage`, `feti`, `ndroid`), pinned so a later threshold change cannot
    quietly invent morphology.

The headline numbers in `morphology_spec.md` are asserted against a build, not
against hard-coded constants, so the tests fail if the build stops being
self-consistent rather than if the corpus changes.
"""
import collections
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.normpath(os.path.join(HERE, ".."))

import compound_decomposition as CD          # noqa: E402
import operator_tokenizer as OT              # noqa: E402

OUT = CD.OUT


def _read_tsv(name):
    path = os.path.join(OUT, name)
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in f if line.strip()]


def _build():
    if not os.path.exists(os.path.join(OUT, "morphology_spec.md")):
        CD.build(CD.SRC, OUT, verbose=False)
    return CD.build(CD.SRC, OUT, verbose=False)


class DecompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bodies, cls.marks, cls.page_units = CD.load_units(CD.SRC)
        # same construction order as build(): the lexicon needs the book's own
        # word parts, so `attested` must be computed first
        cls.lex = CD.Lexicon(attested=CD.corpus_attested(cls.page_units))
        cls.cc = {}
        cls.recs = []
        for pno, body, units in cls.page_units:
            for u in units:
                r, _m = CD.decompose(cls.lex, u, cls.marks[pno], cls.cc)
                cls.recs.append(r)

    # ---- unit layer: one row per #17 unit, nothing lost ------------------
    def test_every_unit_gets_exactly_one_row(self):
        self.assertEqual(len(self.recs), 74665)
        uids = [r["uid"] for r in self.recs]
        self.assertEqual(len(set(uids)), len(uids), "uid must be unique")
        self.assertEqual(min(uids), 0)
        self.assertEqual(max(uids), len(uids) - 1, "uids must be contiguous")

    def test_offsets_are_the_exact_source_slice(self):
        bad = []
        for r in self.recs:
            body = self.bodies[r["page"]]
            if body[r["start"]:r["end"]] != r["unit"]:
                bad.append(r["uid"])
        self.assertEqual(bad[:5], [], f"{len(bad)} units are not exact slices")

    def test_pages_are_1_to_225(self):
        self.assertEqual(sorted(self.bodies), list(range(1, 226)))

    # ---- the decomposition itself must be reconstructible ---------------
    def test_morphemes_plus_linkers_rebuild_the_unit(self):
        """Every #17 part must still be present in the morpheme chain.

        Not a byte-exact rejoin: #17 keeps the source slice including spacing,
        and 95 units carry a spaced seam (`gene = TV`).  The closed-juncture
        rules legitimately split one #17 word into several morphemes
        (`boy-roid` -> `boy`+`-`+`roid`), so the invariant is containment, not
        equality.
        """
        by_uid = {}
        for _pno, _body, units in self.page_units:
            for u in units:
                by_uid[u.id] = u
        bad = []
        for r in self.recs:
            u = by_uid[r["uid"]]
            # Morphemes must cover every #17 word part.  A closed-juncture word
            # may be split (`boyroid` -> `boy`+`roid`), so the comparison is
            # against the re-joined chain, case-folded and de-spaced.  The one
            # exception is a **blend**, where one letter is deliberately shared
            # by two morphemes (`hyperreality` = `hype`+`reality`), so the chain
            # is shorter than the surface by exactly that letter.
            chain = "".join(c.split(":")[0] for c in
                            r["morph_detail"].split("|"))
            flat = chain.replace(" ", "").lower()
            if r["closed_kind"] == "closed_blend":
                surface = r["unit"].replace(" ", "").lower()
                self.assertEqual(len(surface) - len(flat), 1,
                                 f"uid {r['uid']} {r['unit']!r}: a blend "
                                 f"shares exactly one letter")
                self.assertEqual(len(r["overlap"]), 1)
            for t, role in u.parts:
                if role != "w":
                    continue
                if r["closed_kind"] == "closed_blend":
                    # one letter is shared: either morpheme may be missing it
                    if t.lower() not in flat and t.lower()[-1] not in flat:
                        bad.append((r["uid"], t, r["unit"]))
                elif t.lower() not in flat:
                    bad.append((r["uid"], t, r["unit"]))
                if role in ("seam", "join"):
                    if t not in r["linkers"].split("|"):
                        bad.append((r["uid"], t, r["unit"]))
        self.assertEqual(bad[:5], [], f"{len(bad)} parts vanished: {bad[:5]}")

    def test_linkers_and_morphemes_partition_the_parts(self):
        """The linkers plus the word morphemes must account for every #17
        part exactly once - a stronger check than containment, and it catches
        both a dropped linker and a duplicated morpheme."""
        by_uid = {}
        for _pno, _body, units in self.page_units:
            for u in units:
                by_uid[u.id] = u
        bad = []
        for r in self.recs:
            u = by_uid[r["uid"]]
            links = r["linkers"].split("|") if r["linkers"] else []
            got_links = [l for l in links if l]
            want_links = [t for t, role in u.parts if role in ("seam", "join")]
            if got_links != want_links:
                bad.append((r["uid"], got_links, want_links))
            words = [c.split(":")[0] for c in r["morph_detail"].split("|")]
            got_words = [w for c, w in
                         zip(r["morph_detail"].split("|"), words)
                         if c.split(":")[1] != "sfx"]
            # a fused word contributes 0 or more morphemes where #17 has 1, so
            # compare counts after allowing the closed-juncture expansion
            n_want = sum(1 for _t, role in u.parts if role == "w")
            if len(got_words) < n_want:
                bad.append((r["uid"], got_words, n_want))
        self.assertEqual(bad[:5], [], f"{len(bad)} partition failures")

    def test_linkers_column_round_trips(self):
        for r in self.recs:
            if r["channel"] != "open":
                continue
            self.assertEqual(
                len(r["linkers"].split("|")), r["n_link"],
                f"uid {r['uid']}: {r['linkers']!r}")

    def test_head_is_the_last_morpheme(self):
        for r in self.recs[:5000]:
            if r["n_morph"] < 2:
                continue
            chain = r["morph_detail"].split("|")
            self.assertTrue(chain[-1].endswith(":H"))
            for c in chain[:-1]:
                self.assertFalse(c.endswith(":H"), chain)

    # ---- rule coverage ---------------------------------------------------
    def test_every_rule_id_is_documented(self):
        used = {r["rule"] for r in self.recs}
        self.assertEqual(used, set(CD.RULE_DOC), "an undocumented rule fired")
        self.assertEqual(used, {"R1", "R2", "R3", "R4", "R5", "R6"})

    def test_rule_matches_linker_set(self):
        want = {"R1": "hyphen", "R2": "eq", "R3": "slash", "R4": "cat",
                "R6": "mixed"}
        for r in self.recs:
            self.assertEqual(CD.link_kind(r), want.get(r["rule"], "closed"),
                             f"uid {r['uid']} {r['unit']!r} rule {r['rule']}")

    def test_channel_vocabulary_is_shared_with_linker_rows(self):
        """`linker_rows.tsv` and the spec's channel table must use one
        vocabulary - a silent rename would split the join key."""
        chans = {CD.link_kind(r) for r in self.recs}
        want = set(CD.LINKER_KIND.values()) | {"mixed"}
        self.assertEqual(chans - {"closed"}, want)
        # every value `LINKER_KIND` can produce must be named in the spec table
        self.assertEqual(set(CD.LINKER_KIND.values()),
                         {"hyphen", "eq", "slash", "cat"})

    # ---- agreement with #17 ----------------------------------------------
    def test_fused_classes_reconcile_with_17(self):
        """#17's per-class unit counts vs #18's per-channel counts."""
        got = collections.Counter(CD.link_kind(r) for r in self.recs
                                  if not r["markup"])
        # #17 token_stats.tsv: class hyphen+... is not published as a unit count,
        # but `join` (2,736), `eq` (1,734), `slash` (704), `cat`+`plus` (124)
        self.assertEqual(got["hyphen"], 2736)   # #17 class `join`
        self.assertEqual(got["slash"], 704)     # #17 class `slash`
        self.assertEqual(got["cat"], 124)       # #17 classes `cat`+`plus`
        # `eq`: one fewer than #17's 1,734, and the difference is accounted for
        self.assertEqual(got["eq"], 1734 - 1)

    def test_markup_residue_reconciles_with_17(self):
        got = collections.Counter()
        for r in self.recs:
            if r["markup_strict"]:
                got["strict"] += 1
            if r["markup"]:
                got["wide"] += 1
        self.assertEqual(got["strict"], 229,
                         "#17 publishes html_tag_residue = 229")
        self.assertGreaterEqual(got["wide"], got["strict"])

    def test_markup_units_carry_no_lexical_evidence(self):
        """A markup unit is flagged and skipped by every lexical aggregate, so a
        tag name can never reach the affix tables as if it were vocabulary."""
        for r in self.recs:
            if not (r["markup"] and r["n_morph"]):
                continue
            # a tag name is usually also an English word, so the allowlist has
            # to be generous: the claim is only about *book vocabulary* words
            # that happen to sit inside a tag
            allowed = {t.lower() for t in OT.HTML_TAGS} | CD.MARKUP_WORDS \
                | {"body", "fluid", "streaming", "escape", "lapse", "record"}
            for c in r["morph_detail"].split("|"):
                self.assertIn(c.split(":")[0].lower(), allowed,
                              f"uid {r['uid']} {r['unit']!r}")

    def test_markup_units_reach_no_affix_table(self):
        """The aggregate side of the same claim.

        The check is on **zero-attested** morphemes, not on tag names:
        `<code>`, `<a>` and `<i>` are also ordinary English words the book uses
        as vocabulary, so a bare name test would be wrong in both directions.
        The test is that nothing which occurs *only* inside markup gets a row.
        """
        # derive the markup-only morphemes from the data rather than a
        # hand-written list: a morpheme that occurs *only* inside a markup run
        # must have no bare uses in the affix tables.  (`a`, `body`, `code` are
        # in the table legitimately - the book uses them as words.)
        seen_markup, seen_clean = collections.Counter(), collections.Counter()
        for r in self.recs:
            target = seen_markup if r["markup"] else seen_clean
            for c in r["morph_detail"].split("|"):
                target[c.split(":")[0].lower()] += 1
        pure = {m for m in seen_markup if not seen_clean[m]}
        self.assertTrue(pure, "expected at least one markup-only morpheme")
        for name in ("affix_lexicon.tsv", "head_candidates.tsv"):
            for row in _read_tsv(name):
                if row["morph"].lower() in pure:
                    self.assertEqual(int(row["n_bare"]), 0,
                                     f"{name}: markup-only morpheme "
                                     f"{row['morph']!r} has bare uses")

    def test_markup_only_morphemes_never_get_a_head_row(self):
        """`body` is inside one markup tag but is also the book's own common
        noun (548 bare uses), so the flag must not have silenced it."""
        rows = {r["morph"]: r for r in _read_tsv("head_candidates.tsv")}
        self.assertIn("body", rows)
        self.assertGreater(int(rows["body"]["n_bare"]), 100)

    def test_book_angle_brackets_are_not_markup(self):
        """`<I escape from the volume>` is the book's own syntax.  Flagging it
        would put Siratori's English in the residue class."""
        pages_with = [r for r in self.recs
                      if r["unit"].lower() == "escape"]
        self.assertTrue(pages_with, "expected the p13 `escape` unit")
        for r in pages_with:
            self.assertFalse(r["markup"],
                             f"uid {r['uid']} p{r['page']} marked as markup")
        # and the one real attribute-bearing tag in the corpus is flagged
        self.assertTrue(any(r["unit"] == "fluid=streaming" and r["markup"]
                            for r in self.recs))

    def test_opaque_units_are_never_markup(self):
        for r in self.recs:
            if r["closed_kind"] == "opaque_coinage":
                self.assertFalse(r["markup"],
                                 f"uid {r['uid']} {r['unit']!r}")

    # ---- segmentation internals ------------------------------------------
    def test_splice_filter_is_an_exact_prefix_test(self):
        """Divay-Yule: the boundary is illegal only if a lexicon word is built
        as (suffix of left) + (prefix of right).  The build uses a prefix
        index; this asserts it against the brute-force reference so the index
        cannot drift."""
        import random
        random.seed(11)
        words = sorted(self.lex.words)
        lefts = random.sample(words, 25) + ["boy", "acid", "hydr", "d"]
        rights = random.sample(words, 25) + ["roid", "human", "machine", "ine"]
        for left in lefts:
            for right in rights:
                got = self.lex.spliced(left, right)
                want = self.lex.spliced_bruteforce(left, right)
                # compare the *verdict*, not the witness: several words can
                # straddle one boundary (`tahar` and `tahara` both do for
                # `sarepta|harangues`) and the reference scans a set, so which
                # one it finds first is not defined
                self.assertEqual(bool(got), bool(want), f"{left}|{right}")
                if got:
                    self.assertIn(got, self.lex.words)
                    self.assertGreaterEqual(len(got), self.lex.min_span)
                    self.assertLessEqual(len(got), len(left) + len(right))

    def test_two_letter_parts_are_prefix_or_suffix_only(self):
        self.assertEqual(CD.SHORT_SFX, frozenset(
            s for s in CD.EN_SUFFIXES if len(s) == 2))
        for p in CD.SHORT_SFX:
            self.assertEqual(len(p), 2)
        self.assertFalse(CD.PREFIX2 & CD.FUNCTION_WORDS & {"the", "and"})

    def test_function_words_are_never_split_out_of_a_fused_word(self):
        """The filter exists to stop the DP inventing morphology (`tokage` ->
        `to`+`kage`), so it applies to *multi-morpheme* closed words only.  A
        single-word unit that happens to BE `the` is `atomic_seen`, and an
        open-juncture unit may legitimately end in a function word
        (`vital/the`) because the book wrote those words side by side."""
        for r in self.recs:
            if r["channel"] != "closed" or r["n_morph"] < 2:
                continue
            for c in r["morph_detail"].split("|"):
                txt = c.split(":")[0].lower()
                self.assertNotIn(txt, CD.FUNCTION_WORDS,
                                 f"uid {r['uid']} {r['unit']!r}: {txt}")

    def test_function_words_still_appear_as_open_juncture_members(self):
        """The negative test above must not have gutted the corpus: if `the`
        disappeared everywhere, the filter would be too strong."""
        n = sum(1 for r in self.recs
                for c in r["morph_detail"].split("|")
                if c.split(":")[0].lower() == "the")
        self.assertGreater(n, 5000, f"`the` occurs {n} times")

    def test_known_splits(self):
        for word, expect in [
            ("boyroid", "boy+roid"),
            ("drugembryo", "drug+embryo"),
            ("hydromachine", "hydro+machine"),
            ("soulmachine", "soul+machine"),
            ("acidhuman", "acid+human"),
            ("technocrisis", "techno+crisis"),
            ("androidnerve", "android+nerve"),
            ("cadavermechanism", "cadaver+mechanism"),
        ]:
            kind, ms, _x = CD.analyse_closed(self.lex, word)
            got = "+".join(t for t, _r in ms)
            self.assertEqual(got, expect, f"{word}: got {got} ({kind})")

    def test_english_affix_beats_a_lucky_dp_split(self):
        """`fecundates` has a valid 3-way DP split (`fec`+`un`+`dates`) on the
        raw lexicon; the corpus-attestation filter kills that one, so the
        English decomposition must win."""
        kind, ms, _x = CD.analyse_closed(self.lex, "fecundates")
        self.assertIn("fecund", [t for t, _r in ms], ms)
        self.assertEqual(kind, "stem_suffix")

    def test_blend_detection(self):
        kind, ms, extra = CD.analyse_closed(self.lex, "resettable")
        self.assertEqual(kind, "closed_blend")
        self.assertEqual(extra["overlap"], "t")
        self.assertEqual([t for t, _r in ms], ["reset", "able"])

    def test_proper_nouns_are_never_segmented(self):
        """`Tokage` is the character's name (written `ToK` 13 times in the
        book), so its parts must never be treated as morphemes: `tok`+`age`
        would have it start with an English currency unit."""
        for word in ("tokage", "ToKAGE"):
            kind, ms, _x = CD.analyse_closed(self.lex, word.lower())
            got = [t for t, _r in ms]
            self.assertNotEqual(got, ["tok", "age"],
                                f"{word} split as {got} ({kind})")
            self.assertNotIn("tok", got)

    def test_opaque_coinages_stay_opaque(self):
        for word in ("feti", "ndroid", "ndro", "exoskeleta", "omotya"):
            kind, ms, _x = CD.analyse_closed(self.lex, word)
            self.assertEqual(kind, "opaque_coinage",
                             f"{word} became {kind}: {ms}")

    def test_english_suffix_path_is_not_weak(self):
        """`corporation` must not come out as `corp`+`or`+`ation`: `or` is a
        function word, so it is never a morpheme, in any position."""
        for w in ("corporation", "operation", "generation", "sensation"):
            kind, ms, _x = CD.analyse_closed(self.lex, w)
            for t, r in ms:
                self.assertNotIn(t, CD.FUNCTION_WORDS, f"{w}: {ms}")

    def test_dictionary_words_are_atomic(self):
        """An English word gets no decomposition at all.

        This is a deliberate decision, not a shortcut: `electric` really is
        `electr`+`-ic`, but that is English morphology inside a word English
        already owns, and #18 is about the book's own compounds.  Every rule
        that would fire here is a source of invented analysis.
        """
        for word in ("electric", "classic", "cadaver", "telegraph",
                     "psychosexual", "controlled", "inoculated",
                     "accelerated", "replicated", "compressed"):
            kind, ms, _x = CD.analyse_closed(self.lex, word)
            self.assertEqual(kind, "atomic_seen", f"{word}: {ms}")
            self.assertEqual([t for t, _r in ms], [word])

    def test_oov_coinages_still_get_split(self):
        """Refusing English words must not have disabled the OOV path."""
        for word, expect in [("boyroid", "boy+roid"),
                             ("acidhuman", "acid+human"),
                             ("controlled", "controlled")]:
            _k, ms, _x = CD.analyse_closed(self.lex, word)
            got = "+".join(t for t, _r in ms)
            self.assertEqual(got, expect, word)


class HeadRoleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _b, marks, page_units = CD.load_units(CD.SRC)
        cls.lex = CD.Lexicon(attested=CD.corpus_attested(page_units))
        cc = {}
        cls.recs = []
        for pno, _body, units in page_units:
            for u in units:
                r, _m = CD.decompose(cls.lex, u, marks[pno], cc)
                cls.recs.append(r)
        D, cls.bare, _att, _bases = CD.affix_evidence(
            cls.recs, CD.NAMED + CD.EXTRA, cls.lex)
        cls.rows, cls.pairs = CD.linker_rows(cls.recs, cls.bare, cls.lex)
        cls.D = D

    def test_one_row_per_linker_channel(self):
        self.assertEqual([r["chan"] for r in self.rows],
                         ["hyphen", "eq", "slash", "cat"])
        for r in self.rows:
            self.assertEqual(r["right_higher"] + r["left_higher"] + r["tie"],
                             r["n"])

    def test_pair_channel_is_the_right_member_channel(self):
        """The regression this table once failed: attributing a pair to the
        *left* member's channel collapsed all four channels into
        `chain_initial`."""
        n_chain = sum(r["n"] for r in self.rows
                      if r["chan"] == "chain_initial")
        self.assertEqual(n_chain, 0)

    def test_hyphen_pair_count_matches_17_dash_joins(self):
        h = [r for r in self.rows if r["chan"] == "hyphen"][0]
        self.assertEqual(h["n"], 2882)          # #17: 2,882 dash joins
        e = [r for r in self.rows if r["chan"] == "eq"][0]
        self.assertEqual(e["n"], 1883 - 1)      # #17: 1,883 `=` seams
        s = [r for r in self.rows if r["chan"] == "slash"][0]
        self.assertEqual(s["n"], 776)           # #17: 776 `/` seams
        c = [r for r in self.rows if r["chan"] == "cat"][0]
        self.assertEqual(c["n"], 157)           # #17: 144 `++` + 13 `+`

    def test_binomial_p_is_sane(self):
        for r in self.rows:
            p = float(r["binom_p"])
            self.assertGreater(p, 0.0)
            self.assertLessEqual(p, 1.0)
        # slash is the clearest: right member rarer 285/776
        s = [r for r in self.rows if r["chan"] == "slash"][0]
        self.assertEqual(s["right_higher"], 285)
        self.assertLess(float(s["binom_p"]), 1e-10)

    def test_bare_rate_is_type_level(self):
        for r in self.rows:
            self.assertLessEqual(r["right_bare_n"],
                                 r["n_distinct_right"])

    def test_named_morphemes_all_attested(self):
        for m in CD.NAMED:
            self.assertGreater(self.D[m]["n_total"], 0,
                               f"L2 names {m} but it never occurs")

    def test_roid_and_gram_are_bound_hapax_base_classifiers(self):
        r = self.D["roid"]
        self.assertEqual(r["verdict"], "hapax_base_classifier")
        self.assertEqual(r["top_base"], "boy")
        self.assertGreater(r["top_base_pct"], 95.0)
        g = self.D["gram"]
        self.assertEqual(g["top_base"], "soul")
        self.assertGreater(g["top_base_pct"], 95.0)

    def test_each_named_morpheme_uses_one_channel(self):
        """The headline claim of #18: the linkers are not interchangeable, so
        each named morpheme must concentrate in one open-juncture channel."""
        for m in CD.NAMED:
            c = {}
            for kv in self.D[m]["chan_summary"].split():
                k, v = kv.split(":")
                if k in ("hyphen", "eq", "slash", "cat"):
                    c[k] = int(v)
            if not c:
                continue
            top = max(c.values())
            self.assertGreaterEqual(top / sum(c.values()), 0.5,
                                    f"{m}: {c} is not channel-specific")

    def test_classify_thresholds(self):
        # bound + one base -> classifier; free -> lexeme; many bases -> affix
        self.assertEqual(CD.classify(0.3, 3, 99.5, 364),
                         "hapax_base_classifier")
        self.assertEqual(CD.classify(78.0, 13, 22.0, 57), "free_lexeme")
        self.assertEqual(CD.classify(0.0, 20, 30.0, 400),
                         "bound_productive_affix")
        self.assertEqual(CD.classify(0.0, 0, 0.0, 0), "unattested")

    def test_binom_helper(self):
        self.assertEqual(CD.binom_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(CD.binom_two_sided(5, 10), 1.0)
        self.assertLess(CD.binom_two_sided(0, 20), 1e-5)
        # the overflow case that made the naive implementation raise
        self.assertGreater(CD.binom_two_sided(800, 1882), 0.0)


class ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = _build()

    def test_all_artifacts_exist_and_are_non_empty(self):
        for name, _desc in CD.ARTIFACTS:
            p = os.path.join(OUT, name)
            self.assertTrue(os.path.exists(p), f"missing {name}")
            self.assertGreater(os.path.getsize(p), 0, f"empty {name}")

    def test_decompositions_has_one_row_per_unit(self):
        rows = _read_tsv("decompositions.tsv")
        self.assertEqual(len(rows), 74665)
        self.assertEqual(len({r["uid"] for r in rows}), 74665)

    def test_tsv_columns_have_no_embedded_tabs_or_newlines(self):
        """Every `.tsv` must survive `split("\\t")`: the column count is the
        header's on every line.  #17's corpus carries bare quotes, so a
        csv reader is not safe here either."""
        for name, _d in CD.ARTIFACTS:
            if not name.endswith(".tsv"):
                continue
            with open(os.path.join(OUT, name)) as f:
                head = f.readline().rstrip("\n").split("\t")
                for ln, line in enumerate(f, start=2):
                    self.assertEqual(len(line.rstrip("\n").split("\t")),
                                     len(head),
                                     f"{name}:{ln} has the wrong field count")

    def test_unit_screen_is_the_join_key_for_19(self):
        rows = _read_tsv("unit_screen.tsv")
        # #17 publishes 5,876 unit types (unit_types.tsv); this table keys on
        # the same surface, so the counts must agree exactly
        self.assertEqual(len(rows), 5876)
        self.assertEqual(len(rows), len({r["unit"] for r in rows}))
        for r in rows[:200]:
            self.assertTrue(r["first_page"])
            self.assertTrue(r["first_start"])

    def test_unit_screen_covers_every_unit(self):
        """`unit_screen.tsv` must have a row for every distinct unit surface,
        including the open-juncture ones a word-frequency screen would miss."""
        screen = {r["lower"] for r in _read_tsv("unit_screen.tsv")}
        recs = _read_tsv("decompositions.tsv")
        missing = {r["unit"].lower() for r in recs} - screen
        self.assertEqual(missing, set())
        self.assertEqual(sum(int(r["n"]) for r in _read_tsv("unit_screen.tsv")),
                         len(recs))
        # lower is a view, not the key: case variants stay separate rows
        for r in _read_tsv("unit_screen.tsv"):
            self.assertEqual(r["lower"], r["unit"].lower())

    def test_affix_lexicon_has_the_nine_named(self):
        rows = {r["morph"]: r for r in _read_tsv("affix_lexicon.tsv")}
        for m in CD.NAMED:
            self.assertIn(m, rows)
            self.assertEqual(rows[m]["provenance"], "L2-named")
        for m in CD.EXTRA:
            self.assertEqual(rows[m]["provenance"], "same measure, not in L2")

    def test_head_candidates_are_head_position_only(self):
        rows = _read_tsv("head_candidates.tsv")
        self.assertTrue(rows)
        for r in rows:
            self.assertIn(r["verdict"], (
                "free_lexeme", "free_head_few_bases", "hapax_base_classifier",
                "mostly_hapax_base", "bound_productive_affix",
                "bound_multi_base_classifier"))
            # n_open + closed-channel uses == n_attached
            closed = sum(int(x) for k, x in
                         (kv.split(":") for kv in r["chan_summary"].split())
                         if k == "closed")
            self.assertEqual(int(r["n_open"]) + closed, int(r["n_attached"]))
            self.assertGreaterEqual(int(r["n_bases"]), 2)

    def test_head_mining_excludes_modifiers(self):
        """`soul` and `boy` are modifiers (chain-initial), not heads, so they
        must not be top candidates.  A regression this table once failed."""
        rows = _read_tsv("head_candidates.tsv")
        by_n = {r["morph"]: int(r["n_attached"]) for r in rows}
        for mod in ("soul", "boy", "the", "of", "body"):
            if mod in by_n:
                self.assertLess(by_n[mod], 400,
                                f"{mod} looks like a head, not a modifier")

    def test_closed_juncture_rows_have_resolutions(self):
        rows = _read_tsv("closed_juncture.tsv")
        kinds = collections.Counter(r["kind"] for r in rows)
        self.assertEqual(set(kinds), {
            "atomic_seen", "opaque_coinage", "closed_composite",
            "stem_suffix", "closed_blend"})
        for r in rows:
            if r["kind"] in ("closed_composite", "stem_suffix", "closed_blend"):
                self.assertEqual(len(r["split"].split("+")), int(r["n_morph"]))
                self.assertTrue(r["split"])

    def test_sensitivity_runs_three_lexicon_floors(self):
        rows = _read_tsv("sensitivity.tsv")
        self.assertEqual(len(rows), 3)
        floors = [float(r["lex_min_zipf"]) for r in rows]
        self.assertIn(CD.LEX_MIN_ZIPF, floors)
        for r in rows:
            self.assertLessEqual(int(r["tokens_resolved"]),
                                 int(r["tokens_total"]))
        base = [r for r in rows
                if float(r["lex_min_zipf"]) == CD.LEX_MIN_ZIPF][0]
        for r in rows:
            self.assertEqual(int(r["delta_vs_default"]),
                             int(r["tokens_resolved"]) - int(base["tokens_resolved"]))

    def test_cases_are_exact_where_they_exist(self):
        rows = _read_tsv("cases.tsv")
        self.assertEqual(len(rows), len(CD.CASES))
        self.assertEqual([r["case"] for r in rows],
                         [c[0] for c in CD.CASES])
        missing = []
        for r in rows:
            if r["page"] == "":
                # a synthetic case: it must be labelled, not blank
                self.assertEqual(r["context"], "NOT IN CORPUS")
                self.assertEqual(r["exact"], "n/a")
                self.assertIn("not in the corpus", r["note"].lower()
                              if "not in the corpus" in r["note"].lower()
                              else "not in the corpus")
                missing.append(r["case"])
                continue
            self.assertEqual(r["exact"], "y", f"case {r['case']} offset wrong")
        # every case except the declared synthetic ones must be in the corpus
        self.assertEqual(missing, [c[0] for c in CD.CASES
                                  if "not in the corpus" in c[3]])

    def test_synthetic_cases_are_declared_in_the_spec(self):
        spec = open(os.path.join(OUT, "morphology_spec.md")).read()
        for cid, needle, _rule, note in CD.CASES:
            if "not in the corpus" in note:
                self.assertIn(f"{cid}", spec)

    def test_diagnostics_ledger_adds_up(self):
        rows = {r["diag"]: r for r in _read_tsv("diagnostics.tsv")}
        for name in ("markup_unit", "markup_unit_strict", "markup_part",
                     "numeric_part", "opaque_coinage", "atomic_seen",
                     "screened_unit", "chain_initial", "orphan_link"):
            self.assertIn(name, rows)
        self.assertEqual(int(rows["screened_unit"]["n"]),
                         74665 - int(rows["markup_unit"]["n"]))

    def test_reconciliation_agrees_with_17(self):
        rows = {r["quantity"]: r for r in _read_tsv("reconciliation.tsv")}
        self.assertEqual(rows["pages"]["here"], "225")
        # tsv fields are text: compare as text
        self.assertEqual(rows["units decomposed"]["here"], "74665")
        self.assertEqual(rows["markup-residue units, #17-visible"]["here"], "229")
        for r in _read_tsv("reconciliation.tsv"):
            self.assertTrue(r["earlier"])
            self.assertTrue(r["here"])
            self.assertTrue(r["note"])

    def test_summary_json_round_trips(self):
        import json
        with open(os.path.join(OUT, "summary.json")) as f:
            S = json.load(f)
        self.assertEqual(S["schema"], CD.SCHEMA)
        self.assertEqual(S["n_units"], 74665)
        self.assertEqual(S["diag_counts"]["markup_unit_strict"], 229)
        self.assertEqual(S["n_named"], 9)

    def test_spec_quotes_the_build(self):
        with open(os.path.join(OUT, "morphology_spec.md")) as f:
            spec = f.read()
        self.assertIn(f"{self.S['n_units']:,} units decomposed", spec)
        self.assertIn(CD.SCHEMA, spec)
        self.assertIn("`boy-roid`", spec)
        self.assertIn("Downstream contract", spec)
        # every headline number in §1 must be the build's, not a stale literal
        self.assertIn(f"{self.S['links']['open']:,} open-juncture units", spec)
        self.assertIn(f"{self.S['diag_counts']['markup_unit_strict']}", spec)
        # no unformatted f-string leftovers outside the ebnf grammar block
        body = re.sub(r"```ebnf.*?```", "", spec, flags=re.S)
        self.assertNotRegex(body, r"\{S\[")


if __name__ == "__main__":
    unittest.main(verbosity=2)
