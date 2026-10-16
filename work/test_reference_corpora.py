#!/usr/bin/env python3
"""Regression suite for #56 reference corpora.

Checks the invariants that matter: the manifest structure and provenance (and
its ``hash_role`` semantics: enforced pins vs observed checksums with drift
reporting), the fair-use excerpt files (embedded constants), the corpus
sizes, the hash verification of the pinned #24 references, the Gutenberg
strip rule, the download retry logic, and the #80 guarantees -- ``--no-download``
is cache-only for *every* source (general English and legal included), and
arXiv summaries are entity-decoded before tokenization.  #78 adds the
per-record rights ledger for the PubMed corpus: PMID parsing, rights-class
derivation, Europe PMC batching, ledger coverage of the snapshot, and the
build wiring that re-verifies a stale ledger in download mode but warns
(network-free) under ``--no-download``.  #79 adds the modern legal-abstract
register (DOAJ): the historical label on the Gutenberg legal texts, the record
parsing and stub gate, the per-record licence ledger resolved from the
journals' DOAJ records, and the same build wiring.  The live PubMed / arXiv /
DOAJ corpora are read from the cache built by ``python work/reference_corpora.py``;
the suite does not re-query them.

    python work/test_reference_corpora.py
"""
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import frequency_zipf as fz
import reference_corpora as rc

ROOT = rc.ROOT
CORPORA = rc.OUT
MANIFEST = CORPORA / "manifest.tsv"


def read_tsv(path):
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh if line.strip()]


@contextmanager
def patched(**attrs):
    """Temporarily replace module attributes on ``rc``."""
    orig = {k: getattr(rc, k) for k in attrs}
    for k, v in attrs.items():
        setattr(rc, k, v)
    try:
        yield
    finally:
        for k, v in orig.items():
            setattr(rc, k, v)


GUT_HEADER = "*** START OF THE PROJECT GUTENBERG EBOOK FAKE ***\n"
GUT_FOOTER = "*** END OF THE PROJECT GUTENBERG EBOOK FAKE ***\n"


def gutenberg_body(n_words=3000, word="word"):
    """A synthetic Gutenberg download: header + body + footer."""
    return (GUT_HEADER + " ".join([word] * n_words) + "\n"
            + GUT_FOOTER).encode("utf-8")


def stripped_body(n_words=3000, word="word"):
    """The same body after the strip rule (what a cached corpus holds)."""
    return (" ".join([word] * n_words) + "\n").encode("utf-8")


def no_network(*_a, **_k):
    raise AssertionError("cache-only mode must never touch the network")


def fake_fetch(url, path, log=print, **_k):
    """A stand-in download that serves a synthetic Gutenberg body."""
    data = gutenberg_body()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return data


class TestExcerpts(unittest.TestCase):
    """The fair-use excerpt files are embedded constants, written verbatim."""

    def test_excerpt_files_match_constants(self):
        for rel, text in rc.EXCERPTS.items():
            path = CORPORA / rel
            self.assertTrue(path.exists(), f"missing excerpt {rel}")
            self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_excerpt_files_have_attribution(self):
        for text in rc.EXCERPTS.values():
            self.assertIn("#", text.splitlines()[0])
            self.assertIn("--", text)  # at least one attribution line

    def test_excerpt_corpora_present(self):
        for corpus in ("cyberpunk", "cutup_surrealist", "finnegans_wake"):
            files = list((CORPORA / corpus).glob("*.txt"))
            self.assertTrue(files, f"no excerpt files for {corpus}")


class TestManifest(unittest.TestCase):
    """The manifest records provenance for every corpus file."""

    def test_manifest_exists(self):
        self.assertTrue(MANIFEST.exists(), "manifest.tsv missing")

    def test_manifest_columns(self):
        rows = read_tsv(MANIFEST)
        self.assertTrue(rows)
        for col in ("corpus", "file", "source", "license", "sha256",
                    "bytes", "tokens", "types"):
            self.assertIn(col, rows[0], f"manifest missing column {col}")

    def test_manifest_corpora(self):
        rows = read_tsv(MANIFEST)
        corpora = {r["corpus"] for r in rows}
        for c in ("general_english", "cyberpunk", "cutup_surrealist",
                  "finnegans_wake", "medical", "cs", "legal"):
            self.assertIn(c, corpora, f"manifest missing corpus {c}")

    def test_manifest_hashes_match_files(self):
        rows = read_tsv(MANIFEST)
        for r in rows:
            p = CORPORA / r["file"]
            self.assertTrue(p.exists(), f"manifest file missing: {r['file']}")
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(got, r["sha256"],
                             f"hash mismatch for {r['file']}")

    def test_general_english_size(self):
        rows = read_tsv(MANIFEST)
        ge = [r for r in rows if r["corpus"] == "general_english"]
        self.assertEqual(len(ge), 15, "general_english should have 15 files")
        total = sum(int(r["tokens"]) for r in ge)
        self.assertGreater(total, 1_000_000,
                           "general_english should exceed 1M tokens")

    def test_legal_size(self):
        rows = read_tsv(MANIFEST)
        legal = [r for r in rows if r["corpus"] == "legal"]
        self.assertEqual(len(legal), 5, "legal should have 5 files")

    def test_medical_and_cs_present(self):
        rows = read_tsv(MANIFEST)
        med = [r for r in rows if r["corpus"] == "medical"]
        cs = [r for r in rows if r["corpus"] == "cs"]
        self.assertEqual(len(med), 1)
        self.assertEqual(len(cs), 1)
        self.assertGreater(int(med[0]["tokens"]), 100_000)
        self.assertGreater(int(cs[0]["tokens"]), 50_000)


class TestPinnedReferences(unittest.TestCase):
    """The eight pinned #24 references are copied with their hashes intact."""

    def test_pinned_hashes(self):
        for name, _author, _gid, pinned in rc.PINNED_GENERAL:
            p = CORPORA / "general_english" / f"{name}.txt"
            self.assertTrue(p.exists(), f"missing pinned {name}")
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(got, pinned, f"pinned hash mismatch for {name}")

    def _world(self, tmp):
        """A tmp OUT + FREQ_CACHE; the pinned source holds *data*."""
        out, freq = Path(tmp) / "out", Path(tmp) / "freq"
        freq.mkdir(parents=True)
        (out / "general_english").mkdir(parents=True)
        return out, freq

    def test_pinned_hash_mismatch_aborts(self):
        """#80: enforced_pin means enforced -- a mismatch is never absorbed."""
        with tempfile.TemporaryDirectory() as tmp:
            out, freq = self._world(tmp)
            (freq / "pride_and_prejudice.txt").write_bytes(b"tampered bytes")
            with (patched(OUT=out, FREQ_CACHE=freq),
                  self.assertRaises(ValueError)):
                rc.build_general_english(log=lambda m: None)

    def test_pinned_missing_cache_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, freq = self._world(tmp)
            with (patched(OUT=out, FREQ_CACHE=freq),
                  self.assertRaises(FileNotFoundError)):
                rc.build_general_english(log=lambda m: None)


class TestNoDownloadCacheOnly(unittest.TestCase):
    """#80: --no-download is cache-only for every source, network or not."""

    def _world(self, tmp, extra=("cached_novel", "missing_novel"),
               legal=("cached_law", "missing_law")):
        """A minimal pinned + extra-general + legal world under *tmp*."""
        out, freq = Path(tmp) / "out", Path(tmp) / "freq"
        freq.mkdir(parents=True)
        pinned_data = stripped_body(50, "pinned")
        (freq / "pinned.txt").write_bytes(pinned_data)
        (out / "general_english").mkdir(parents=True)
        (out / "legal").mkdir(parents=True)
        (out / "general_english" / "cached_novel.txt").write_bytes(
            stripped_body())
        (out / "legal" / "cached_law.txt").write_bytes(stripped_body())
        pinned = [("pinned", "An Author", 1,
                   hashlib.sha256(pinned_data).hexdigest())]
        extras = [(name, "B Author", i) for i, name in enumerate(extra, 2)]
        laws = [(name, "C Author", i) for i, name in enumerate(legal, 2)]
        return out, freq, pinned, extras, laws

    def test_general_english_skips_missing_instead_of_fetching(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            out, freq, pinned, extras, _laws = self._world(tmp)
            with patched(OUT=out, FREQ_CACHE=freq, PINNED_GENERAL=pinned,
                         EXTRA_GENERAL=extras, download=no_network):
                rows = rc.build_general_english(logs.append,
                                                allow_download=False)
            names = {r["file"] for r in rows}
            self.assertIn("general_english/pinned.txt", names)
            self.assertIn("general_english/cached_novel.txt", names)
            self.assertNotIn("general_english/missing_novel.txt", names)
            self.assertTrue(any("missing_novel" in m and "WARNING" in m
                                for m in logs),
                            f"the skip must be reported: {logs}")

    def test_general_english_download_mode_still_fetches(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, freq, pinned, extras, _laws = self._world(tmp)
            with patched(OUT=out, FREQ_CACHE=freq, PINNED_GENERAL=pinned,
                         EXTRA_GENERAL=extras, download=fake_fetch):
                rows = rc.build_general_english(log=lambda m: None)
            names = {r["file"] for r in rows}
            self.assertIn("general_english/missing_novel.txt", names,
                          "download mode must still fetch a missing novel")

    def test_legal_skips_missing_instead_of_fetching(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            out, freq, pinned, _extras, laws = self._world(tmp)
            with patched(OUT=out, FREQ_CACHE=freq, PINNED_GENERAL=pinned,
                         LEGAL=laws, download=no_network):
                rows = rc.build_legal(logs.append, allow_download=False)
            names = {r["file"] for r in rows}
            self.assertIn("legal/cached_law.txt", names)
            self.assertNotIn("legal/missing_law.txt", names)
            self.assertTrue(any("missing_law" in m and "WARNING" in m
                                for m in logs),
                            f"the skip must be reported: {logs}")

    def test_legal_download_mode_still_fetches(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, freq, pinned, _extras, laws = self._world(tmp)
            with patched(OUT=out, FREQ_CACHE=freq, PINNED_GENERAL=pinned,
                         LEGAL=laws, download=fake_fetch):
                rows = rc.build_legal(log=lambda m: None)
            self.assertIn("legal/missing_law.txt", {r["file"] for r in rows})

    def test_end_to_end_no_download_is_network_free_and_idempotent(self):
        """The whole cache-only build: no network, stable manifest, roles."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            out, freq, pinned, extras, laws = self._world(tmp)
            (out / "medical").mkdir(parents=True)
            (out / "medical" / "pubmed_abstracts.txt").write_bytes(
                stripped_body())
            with patched(OUT=out, FREQ_CACHE=freq, PINNED_GENERAL=pinned,
                         EXTRA_GENERAL=extras, LEGAL=laws,
                         LIVE_FILES={"medical": "medical/pubmed_abstracts.txt"},
                         LIVE_SOURCES={"medical": ("fake eutils",
                                                   "open access")},
                         MIN_TOKENS={"medical": 100, "cs": 100},
                         download=no_network):
                rows = rc.ensure_corpora(allow_download=False, log=logs.append)
                manifest = out / "manifest.tsv"
                first = manifest.read_bytes()
                self.assertFalse((out / rc.DRIFT_FILE).exists(),
                                 "a clean build reports no drift")
                rows2 = rc.ensure_corpora(allow_download=False,
                                          log=lambda m: None)
                self.assertEqual(manifest.read_bytes(), first,
                                 "a second cache-only rebuild must be "
                                 "byte-identical")
            files = {r["file"] for r in rows}
            self.assertEqual(files, {
                "general_english/pinned.txt",
                "general_english/cached_novel.txt",
                "legal/cached_law.txt", "medical/pubmed_abstracts.txt",
                *rc.EXCERPTS.keys()})
            for row in rows:
                self.assertTrue((out / row["file"]).exists(),
                                f"manifest names a missing file {row['file']}")
            roles = {r["file"]: r["hash_role"] for r in rows}
            self.assertEqual(roles["general_english/pinned.txt"],
                             "enforced_pin")
            for f, role in roles.items():
                if f != "general_english/pinned.txt":
                    self.assertEqual(role, "observed_checksum", f)
            self.assertEqual(len(rows), len(rows2))

    def test_no_download_reports_a_stale_legal_abstracts_ledger(self):
        """#79: cache-only still checks the rights position, network-free."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            out, freq, pinned, extras, laws = self._world(tmp)
            (out / "medical").mkdir(parents=True)
            (out / "medical" / "pubmed_abstracts.txt").write_bytes(
                stripped_body())
            la = rc.parse_doaj_records(
                [_doaj_page(_record(doi="10.1/a"))])
            (out / "legal_abstracts").mkdir(parents=True)
            (out / "legal_abstracts" / "legal_abstracts.txt").write_bytes(
                rc.doaj_corpus_bytes(la))
            # a ledger that does not cover the corpus's DOI
            rc.write_doaj_rights(out / rc.DOAJ_RIGHTS_FILE, [{
                "doi": "10.9/z", "journal": "J", "publisher": "P",
                "year": "2025", "language": "EN", "issn": "1",
                "license": "CC BY", "license_class": "open_access",
                "verified_via": "DOAJ journal record",
                "verified_at": "2026-10-16"}])
            with patched(OUT=out, FREQ_CACHE=freq, PINNED_GENERAL=pinned,
                         EXTRA_GENERAL=extras, LEGAL=laws,
                         MIN_TOKENS={"medical": 100, "cs": 100,
                                     "legal_abstracts": 50},
                         download=no_network):
                rc.ensure_corpora(allow_download=False, log=logs.append)
        self.assertTrue(any(rc.DOAJ_RIGHTS_FILE in m and "WARNING" in m
                            for m in logs), logs)

    def test_main_wires_no_download_flag(self):
        orig_ensure, orig_out = rc.ensure_corpora, rc.OUT
        seen = {}

        def _fake_ensure(allow_download=True, log=print):
            seen["allow_download"] = allow_download
            return []

        rc.ensure_corpora = _fake_ensure
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc.main(["--no-download", "--out", tmp])
                self.assertFalse(seen["allow_download"])
                rc.main(["--out", tmp])
                self.assertTrue(seen["allow_download"])
        finally:
            rc.ensure_corpora, rc.OUT = orig_ensure, orig_out


class TestManifestPinsAndDrift(unittest.TestCase):
    """#80: the manifest says what each sha256 means, and drift is reported."""

    def test_manifest_carries_hash_role_column(self):
        rows = read_tsv(MANIFEST)
        for col in ("sha256", "hash_role"):
            self.assertIn(col, rows[0])
        pinned = {f"general_english/{n}.txt"
                  for n, *_ in rc.PINNED_GENERAL}
        for r in rows:
            want = "enforced_pin" if r["file"] in pinned else "observed_checksum"
            self.assertEqual(r["hash_role"], want, r["file"])

    def test_drift_is_reported_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            mpath = Path(tmp) / "manifest.tsv"
            rc.write_tsv(mpath, ["file", "sha256", "bytes"],
                         [{"file": "a.txt", "sha256": "a" * 64, "bytes": "10"}])
            prior = rc.read_manifest(mpath)
            new = [{"corpus": "x", "file": "a.txt", "sha256": "b" * 64,
                    "bytes": "12"}]
            logs = []
            drift = rc.manifest_drift(new, prior, logs.append)
            self.assertEqual(len(drift), 1)
            self.assertEqual(drift[0]["expected_sha256"], "a" * 64)
            self.assertEqual(drift[0]["observed_sha256"], "b" * 64)
            self.assertTrue(any("DRIFT" in m for m in logs),
                            f"drift must be logged: {logs}")
            rc.write_drift_report(mpath, drift, logs.append)
            report = Path(tmp) / rc.DRIFT_FILE
            self.assertTrue(report.exists(), "drift must be persisted")
            drows = read_tsv(report)
            self.assertEqual(drows[0]["expected_sha256"], "a" * 64)
            self.assertEqual(drows[0]["observed_sha256"], "b" * 64)
            self.assertEqual(drows[0]["expected_bytes"], "10")
            self.assertEqual(drows[0]["observed_bytes"], "12")

    def test_no_drift_when_pins_hold_and_stale_report_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            mpath = Path(tmp) / "manifest.tsv"
            row = [{"file": "a.txt", "sha256": "a" * 64, "bytes": "10"}]
            rc.write_tsv(mpath, ["file", "sha256", "bytes"], row)
            prior = rc.read_manifest(mpath)
            logs = []
            same = [{"corpus": "x", "file": "a.txt", "sha256": "a" * 64,
                     "bytes": "10"}]
            self.assertEqual(rc.manifest_drift(same, prior, logs.append), [])
            self.assertFalse(any("DRIFT" in m for m in logs))
            report = Path(tmp) / rc.DRIFT_FILE
            report.write_text("stale\n", encoding="utf-8")
            rc.write_drift_report(mpath, [], logs.append)
            self.assertFalse(report.exists(),
                             "a clean rebuild clears the stale drift report")

    def test_new_file_is_not_drift(self):
        # A fresh clone has no prior manifest: first-build observations are
        # not drift, and a file the prior manifest never saw is not either.
        self.assertEqual(rc.read_manifest(Path("/nonexistent/manifest.tsv")), {})
        new = [{"corpus": "x", "file": "fresh.txt", "sha256": "c" * 64,
                "bytes": "3"}]
        self.assertEqual(rc.manifest_drift(new, {}, lambda m: None), [])
        self.assertEqual(
            rc.manifest_drift(new, {"other.txt": {"sha256": "d" * 64}},
                              lambda m: None), [])


class TestGutenbergStrip(unittest.TestCase):
    """The Gutenberg header / footer strip rule matches #24."""

    def test_strip_removes_header(self):
        text = ("*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
                "body text here\n"
                "*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\n")
        stripped = rc.strip_gutenberg(text)
        self.assertIn("body text here", stripped)
        self.assertNotIn("START OF", stripped)
        self.assertNotIn("END OF", stripped)


ATOM = ("<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'>"
        "<entry><summary>{}</summary></entry>"
        "<entry><summary>{}</summary></entry></feed>")


class TestParseArxiv(unittest.TestCase):
    """Only a real Atom feed yields abstracts; anything else yields nothing."""

    def test_parses_summaries(self):
        raw = ATOM.format("first abstract text", "second abstract text")
        clean = rc.parse_arxiv(raw.encode("utf-8")).decode("utf-8")
        self.assertIn("first abstract text", clean)
        self.assertIn("second abstract text", clean)
        self.assertNotIn("<summary>", clean)

    def test_error_page_yields_nothing(self):
        # An arXiv error page must not be mistaken for a corpus.
        for body in (b"", b"<html><body>503 Service Unavailable</body></html>",
                     b"not xml at all"):
            self.assertEqual(rc.parse_arxiv(body), b"")


class TestParseArxivEntities(unittest.TestCase):
    """#80: Atom summaries are entity-decoded before they become corpus text.

    arXiv escapes abstract text (``&amp;``, ``&#39;``, ``&lt;``, ...).  Left
    raw, each escape becomes a fake word token (``amp``, ``rsquo``, ``quot``)
    in the #24 tokenizer and inflates the counts, so the corpus would carry
    markup residue as vocabulary.
    """

    def test_named_and_numeric_entities_are_decoded(self):
        summary = ("Deep &amp; reinforcement learning &lt;3 &#39;quotes&#39; "
                   "don&#x2019;t &quot;overfit&quot; &apos;models&apos; "
                   "GAN&amp;style O&#39;Brien &mdash; end")
        raw = ATOM.format(summary, "plain second abstract")
        clean = rc.parse_arxiv(raw.encode("utf-8")).decode("utf-8")
        self.assertIn("Deep & reinforcement learning <3 'quotes' "
                      "don\u2019t \"overfit\" 'models' "
                      "GAN&style O'Brien \u2014 end", clean)

    def test_no_entity_tokens_leak_into_tokenization(self):
        raw = ATOM.format("R&amp;D isn&#39;t trivial &mdash; 5 &lt; 6 &gt; 1",
                          "plain second abstract")
        clean = rc.parse_arxiv(raw.encode("utf-8")).decode("utf-8")
        tokens = {t.lower() for t in fz.EN_WORD_RE.findall(clean)}
        for fake in ("amp", "rsquo", "quot", "lt", "gt", "apos", "mdash",
                     "39", "2019"):
            self.assertNotIn(fake, tokens,
                             f"entity residue leaked a token: {fake!r}")
        self.assertIn("isn't", tokens)
        self.assertIn("trivial", tokens)

    def test_decode_does_not_re_enter_the_markup(self):
        # A decoded &lt; must not become a tag the summary regex matches:
        # extraction happens first, decoding after, so the parse yields
        # exactly the two summaries even when their text contains tags.
        raw = ATOM.format("a &lt;summary&gt; b", "c &amp; d")
        clean = rc.parse_arxiv(raw.encode("utf-8")).decode("utf-8")
        self.assertEqual(clean, "a <summary> b\n\nc & d")

    def test_plain_text_is_left_unchanged(self):
        # html.unescape is a no-op on already-decoded text: rebuilds that
        # re-parse stay stable, and a lone & is not an entity.
        plain = "a & b 'c' \u2014 d"
        raw = ATOM.format(plain, "second")
        clean = rc.parse_arxiv(raw.encode("utf-8")).decode("utf-8")
        self.assertIn(plain, clean)


class TestCorpusGate(unittest.TestCase):
    """write_corpus refuses a below-gate payload and leaves data intact."""

    def _good(self, n_words=3000):
        return (" ".join(["word"] * n_words)).encode("utf-8")

    def test_write_corpus_writes_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.txt"
            n = rc.write_corpus(path, self._good(), 1_000, log=lambda m: None)
            self.assertEqual(n, 3_000)
            self.assertEqual(path.read_bytes(), self._good())

    def test_write_corpus_rejects_small_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.txt"
            path.write_bytes(self._good())
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                rc.write_corpus(path, b"too small", 1_000, log=lambda m: None)
            self.assertEqual(path.read_bytes(), before,
                             "a rejected write must not modify the corpus")

    def test_write_corpus_rejects_empty_payload(self):
        # The exact historical failure: an empty parse overwriting good data.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.txt"
            path.write_bytes(self._good())
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                rc.write_corpus(path, b"", 1_000, log=lambda m: None)
            self.assertEqual(path.read_bytes(), before)

    def test_write_corpus_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.txt"
            rc.write_corpus(path, self._good(), 1_000, log=lambda m: None)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["c.txt"])

    def test_corpus_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.txt"
            self.assertFalse(rc.corpus_ok(path, 1_000))
            path.write_bytes(b"tiny")
            self.assertFalse(rc.corpus_ok(path, 1_000))
            path.write_bytes(self._good())
            self.assertTrue(rc.corpus_ok(path, 1_000))
            self.assertFalse(rc.corpus_ok(path, 10_000))


class TestCsIdempotency(unittest.TestCase):
    """build_cs must not destroy an intact cached corpus.

    Regression: the first implementation re-parsed the cached file with the
    `<summary>` regex on every run.  The cached file is *already* cleaned, so
    the parse yielded zero abstracts and the run overwrote a good corpus with
    an empty one (observed: a 103k-token corpus replaced by 0 bytes, leaving
    the manifest recording an empty file).
    """

    def test_rerun_preserves_clean_cached_corpus(self):
        good = b"\n\n".join(
            " ".join(["token"] * 60).encode("utf-8") for _ in range(2_000))
        self.assertGreater(rc.corpus_tokens(good), rc.MIN_TOKENS["cs"])
        orig_out, orig_sleep, orig_download = rc.OUT, rc.time.sleep, rc.download
        rc.time.sleep = lambda s: None  # never reached; fail fast if it is
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc.OUT = Path(tmp)
                dst = Path(tmp) / "cs" / "arxiv_abstracts.txt"
                dst.parent.mkdir(parents=True)
                dst.write_bytes(good)

                def _no_network(*a, **k):  # pragma: no cover - must not run
                    raise AssertionError("cached corpus should skip the fetch")

                rc.download = _no_network
                try:
                    rc.build_cs(log=lambda m: None)
                    self.assertEqual(dst.read_bytes(), good,
                                     "re-running destroyed the cached corpus")
                    row = rc.live_row("cs", good)
                    self.assertEqual(row["sha256"], hashlib.sha256(
                        good).hexdigest())
                finally:
                    rc.download = orig_download
        finally:
            rc.OUT, rc.time.sleep = orig_out, orig_sleep

    def test_bad_fetch_raises_and_preserves_prior_file(self):
        """All retries erroring must raise and leave the prior file intact.

        The prior file here is a *below-gate* cache (the state the historical
        bug left behind: a truncated or empty cs corpus).  A failed refetch
        must be reported, not absorbed.
        """
        prior = b"\n\n".join(
            " ".join(["stale"] * 5).encode("utf-8") for _ in range(20))
        orig_out, orig_sleep, orig_download = rc.OUT, rc.time.sleep, rc.download
        rc.time.sleep = lambda s: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc.OUT = Path(tmp)
                dst = Path(tmp) / "cs" / "arxiv_abstracts.txt"
                dst.parent.mkdir(parents=True)
                dst.write_bytes(prior)

                def _error_page(url, path, log=print, **k):
                    body = b"<html>503 Service Unavailable</html>"
                    path.write_bytes(body)
                    return body

                rc.download = _error_page
                try:
                    with self.assertRaises(ValueError):
                        rc.build_cs(log=lambda m: None)
                    self.assertEqual(dst.read_bytes(), prior,
                                     "a failed refetch modified the corpus")
                finally:
                    rc.download = orig_download
        finally:
            rc.OUT, rc.time.sleep = orig_out, orig_sleep

    def test_fetch_recovers_and_writes_clean(self):
        """A successful fetch extracts abstracts and passes the gate."""
        orig_out, orig_sleep, orig_download = rc.OUT, rc.time.sleep, rc.download
        rc.time.sleep = lambda s: None
        # Lower the gate so the synthetic feed can be small and readable
        # (the two fake abstracts below tokenise to 8 words).
        orig_gate = rc.MIN_TOKENS["cs"]
        rc.MIN_TOKENS["cs"] = 5
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc.OUT = Path(tmp)
                dst = Path(tmp) / "cs" / "arxiv_abstracts.txt"
                atom = ATOM.format("alpha beta gamma delta",
                                   "epsilon zeta eta theta").encode("utf-8")

                def _atom_fetch(url, path, log=print, **k):
                    path.write_bytes(atom)
                    return atom

                rc.download = _atom_fetch
                try:
                    rc.build_cs(log=lambda m: None)
                    text = dst.read_text(encoding="utf-8")
                    self.assertNotIn("<summary>", text)
                    self.assertIn("alpha beta gamma delta", text)
                    self.assertIn("epsilon zeta eta theta", text)
                    self.assertEqual(
                        sorted(p.name for p in dst.parent.iterdir()),
                        ["arxiv_abstracts.txt"],
                        "the staging file must be cleaned up")
                finally:
                    rc.download = orig_download
        finally:
            rc.OUT, rc.time.sleep = orig_out, orig_sleep
            rc.MIN_TOKENS["cs"] = orig_gate


class TestLiveRowMetadata(unittest.TestCase):
    """Manifest provenance is identical in download and --no-download mode."""

    def test_live_row_matches_cached_manifest_row(self):
        rows = read_tsv(MANIFEST)
        for corpus, rel in rc.LIVE_FILES.items():
            row = next(r for r in rows if r["corpus"] == corpus)
            data = (CORPORA / rel).read_bytes()
            fresh = rc.live_row(corpus, data)
            self.assertEqual(fresh["sha256"], row["sha256"])
            self.assertEqual(fresh["source"], row["source"],
                             f"{corpus} source string drifts between modes")
            self.assertEqual(fresh["license"], row["license"])

    def test_live_corpora_clear_their_gates(self):
        for corpus, rel in rc.LIVE_FILES.items():
            self.assertTrue(rc.corpus_ok(CORPORA / rel, rc.MIN_TOKENS[corpus]),
                            f"{corpus} cache fails its own token gate")


class TestMedicalRights(unittest.TestCase):
    """#78: PubMed rights are per-record facts -- verified, never assumed open.

    A PubMed abstract is the publisher's copyright unless a licence is
    recorded for that record, so the corpus ships a per-record ledger
    (``medical/rights.tsv``) built from Europe PMC.  These tests pin the
    parsing, the class derivation, the batching, and the build wiring; the
    tree-level checks below pin what the *built* ledger actually claims.
    """

    BODY = (b"1. Some J. 2026 Jan;1(1):1. doi: 10.1/x.\n\n"
            b"An abstract.\n\nPMID: 100\n\n\n"
            b"2. Other J. 2026 Feb;2(2):2.\n\nAnother abstract.\n\n"
            b"PMID: 200 [Indexed for MEDLINE]\n")

    @staticmethod
    def _epmc(pmids, licenses):
        """A canned Europe PMC core response for (pmid, license) pairs."""
        body = {"resultList": {"result": [
            {"pmid": p, "license": lic, "isOpenAccess": "Y" if lic else "N",
             "inEPMC": "Y", "inPMC": "Y", "hasPDF": "Y"}
             for p, lic in zip(pmids, licenses)]}}
        return json.dumps(body).encode("utf-8")

    @staticmethod
    def _row(pmid, lic="", epmc="N"):
        return {"pmid": str(pmid), "license": lic, "is_open_access": "Y" if lic else "N",
                "in_epmc": epmc, "in_pmc": epmc, "has_pdf": "N",
                "rights_class": rc.rights_class(lic, epmc),
                "verified_via": "Europe PMC core", "verified_at": "2026-10-16"}

    def test_parse_pmids_reads_every_record(self):
        self.assertEqual(rc.parse_pmids(self.BODY), ["100", "200"])

    def test_parse_pmids_dedups_preserving_order(self):
        self.assertEqual(rc.parse_pmids(b"PMID: 3\nPMID: 1\nPMID: 3\n"),
                         ["3", "1"])

    def test_rights_class_derivation(self):
        self.assertEqual(rc.rights_class("cc by", "N"), "open_access")
        self.assertEqual(rc.rights_class("cc by-nc-nd", "Y"), "open_access")
        self.assertEqual(rc.rights_class("", "Y"), "free_full_text")
        self.assertEqual(rc.rights_class("", "N"), "publisher_copyright")

    def test_fetch_rights_verifies_every_record(self):
        def fetch(url, path, log=print, **_k):
            return self._epmc(["10", "20"], ["cc by", "cc by-nc-nd"])
        with tempfile.TemporaryDirectory() as tmp:
            rows = rc.fetch_rights(["10", "20"], Path(tmp),
                                   log=lambda m: None, fetch=fetch)
        by = {r["pmid"]: r for r in rows}
        self.assertEqual(len(rows), 2)
        self.assertEqual(by["10"]["license"], "cc by")
        self.assertEqual(by["10"]["rights_class"], "open_access")
        self.assertEqual(by["20"]["license"], "cc by-nc-nd")
        self.assertEqual(by["20"]["is_open_access"], "Y")
        self.assertTrue(by["10"]["verified_at"])

    def test_fetch_rights_records_missing_records_as_unverified(self):
        def fetch(url, path, log=print, **_k):
            return self._epmc(["10"], ["cc by"])
        with tempfile.TemporaryDirectory() as tmp:
            rows = rc.fetch_rights(["10", "99"], Path(tmp),
                                   log=lambda m: None, fetch=fetch)
        by = {r["pmid"]: r for r in rows}
        self.assertEqual(by["10"]["rights_class"], "open_access")
        # An id Europe PMC does not return is never upgraded into a right.
        self.assertEqual(by["99"]["rights_class"], "unverified")
        self.assertEqual(by["99"]["license"], "")
        self.assertEqual(by["99"]["is_open_access"], "")
        self.assertIn("no record returned", by["99"]["verified_via"])

    def test_fetch_rights_survives_a_non_json_response(self):
        def fetch(url, path, log=print, **_k):
            return b"<html>502 Bad Gateway</html>"
        with tempfile.TemporaryDirectory() as tmp:
            rows = rc.fetch_rights(["10"], Path(tmp),
                                   log=lambda m: None, fetch=fetch)
        self.assertEqual([r["rights_class"] for r in rows], ["unverified"])

    def test_fetch_rights_batches_large_id_sets(self):
        calls = []

        def fetch(url, path, log=print, **_k):
            calls.append(url)
            return self._epmc([], [])
        pmids = [str(100000 + i) for i in range(450)]
        with tempfile.TemporaryDirectory() as tmp:
            rows = rc.fetch_rights(pmids, Path(tmp),
                                   log=lambda m: None, fetch=fetch)
        self.assertEqual(len(calls), 3, "200 ids per request; 450 -> 3 batches")
        self.assertEqual(len(rows), 450)
        self.assertIn("src%3Amed", calls[0])
        self.assertIn(f"pageSize={rc.RIGHTS_CHUNK}", calls[0])

    def test_write_rights_roundtrip_is_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rights.tsv"
            rc.write_rights(path, [self._row(2, "cc by"), self._row(1)])
            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0],
                             "\t".join(rc.RIGHTS_HEADER))
            self.assertEqual(list(rc.read_rights(path)), ["1", "2"])

    def test_rights_ledger_covers_exact_pmid_sets_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rights.tsv"
            rc.write_rights(path, [self._row(1), self._row(2)])
            self.assertTrue(rc.rights_ledger_covers(path, ["1", "2"]))
            self.assertFalse(rc.rights_ledger_covers(path, ["1"]),
                             "a changed corpus invalidates the ledger")
            self.assertFalse(rc.rights_ledger_covers(path, ["1", "3"]),
                             "an uncovered record is not a covering ledger")
            self.assertFalse(rc.rights_ledger_covers(path, []),
                             "an empty corpus has no rights to verify")
            self.assertFalse(rc.rights_ledger_covers(Path(tmp) / "absent.tsv",
                                                     ["1"]))

    # -- build wiring ------------------------------------------------------
    def _medical_world(self, tmp, pmid_set=("11", "22"), ledger_set=None):
        """A cached corpus under *tmp*, with a ledger over ``ledger_set``.

        The ledger defaults to the corpus's own PMIDs (a covering ledger);
        the stale-ledger tests pass a deliberately shorter ``ledger_set``.
        """
        out = Path(tmp) / "out"
        (out / "medical").mkdir(parents=True)
        body = "".join(f"PMID: {p}\n" for p in pmid_set)
        body += " ".join(["word"] * 3000)
        (out / "medical" / "pubmed_abstracts.txt").write_text(body)
        rc.write_rights(out / rc.RIGHTS_FILE,
                        [self._row(p) for p in (ledger_set or pmid_set)])
        return out

    def test_cached_corpus_with_a_covering_ledger_needs_no_network(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            out = self._medical_world(tmp)
            with patched(OUT=out, MIN_TOKENS={"medical": 100, "cs": 100},
                         download=no_network):
                rows = rc.build_medical(logs.append)
            self.assertEqual(len(rows), 1)
            self.assertTrue(any("covers" in m for m in logs), logs)

    def test_stale_ledger_warns_instead_of_fetching_in_cache_only_mode(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            out = self._medical_world(tmp, ledger_set=("11",))
            with patched(OUT=out, MIN_TOKENS={"medical": 100, "cs": 100},
                         download=no_network):
                rc.build_medical(logs.append, allow_download=False)
            self.assertTrue(any("unverified" in m and "WARNING" in m
                                for m in logs), logs)
            # nothing was fetched: the stale ledger is left exactly as it was
            self.assertTrue(rc.rights_ledger_covers(out / rc.RIGHTS_FILE,
                                                    ["11"]))
            self.assertFalse(rc.rights_ledger_covers(out / rc.RIGHTS_FILE,
                                                     ["11", "22"]))

    def test_stale_ledger_is_reverified_in_download_mode(self):
        logs = []

        def fetch(url, path, log=print, **_k):
            return self._epmc([], [])
        with tempfile.TemporaryDirectory() as tmp:
            out = self._medical_world(tmp, ledger_set=("11",))
            with patched(OUT=out, MIN_TOKENS={"medical": 100, "cs": 100},
                         download=fetch):
                rc.build_medical(logs.append)
            self.assertTrue(
                rc.rights_ledger_covers(out / rc.RIGHTS_FILE, ["11", "22"]),
                "download mode must re-verify rather than keep a stale ledger")
            self.assertTrue(any("records verified" in m for m in logs), logs)


class TestMedicalRightsInTree(unittest.TestCase):
    """The built corpus's ledger is a real, complete per-record verification."""

    def test_tree_ledger_covers_the_snapshot(self):
        corpus = CORPORA / rc.LIVE_FILES["medical"]
        self.assertTrue(corpus.exists(), "build the medical corpus first")
        pmids = rc.parse_pmids(corpus.read_bytes())
        ledger = rc.read_rights(CORPORA / rc.RIGHTS_FILE)
        self.assertTrue(pmids)
        self.assertEqual([p for p in pmids if p not in ledger], [],
                         "every record in the snapshot has a rights row")
        self.assertEqual(len(ledger), len(pmids),
                         "the ledger covers the snapshot exactly")
        for row in ledger.values():
            self.assertTrue(row["verified_at"], "a rights claim carries a date")
            self.assertTrue(row["verified_via"])

    def test_tree_ledger_contradicts_blanket_open_access(self):
        # The manifest used to label the whole corpus "open access (PubMed
        # abstracts)"; the verified distribution must keep contradicting that.
        ledger = rc.read_rights(rc.OUT / rc.RIGHTS_FILE)
        self.assertTrue(ledger)
        counts = {}
        for row in ledger.values():
            counts[row["rights_class"]] = counts.get(row["rights_class"], 0) + 1
        self.assertLess(counts.get("open_access", 0), len(ledger))
        self.assertGreater(counts.get("publisher_copyright", 0), 0)

    def test_manifest_no_longer_labels_medical_open_access(self):
        rows = read_tsv(MANIFEST)
        medical = [r for r in rows if r["corpus"] == "medical"]
        self.assertEqual(len(medical), 1)
        lic = medical[0]["license"]
        self.assertNotIn("open access", lic.lower())
        self.assertIn("rights", lic.lower())
        self.assertIn(rc.RIGHTS_FILE, lic)


class TestDownloadRetry(unittest.TestCase):
    """The download helper retries transient failures with backoff."""

    def test_download_writes_file(self):
        # A small, stable public-domain text (Gutenberg #1, ~1 KB).
        url = "https://www.gutenberg.org/cache/epub/1/pg1.txt"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decl.txt"
            data = rc.download(url, path, log=lambda m: None, retries=2)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), data)
            self.assertGreater(len(data), 100)

    def test_download_retries_then_raises(self):
        # An unreachable host must raise after the retries are exhausted.
        url = "http://127.0.0.1:9/nope.txt"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.txt"
            with self.assertRaises(RuntimeError):
                rc.download(url, path, log=lambda m: None, retries=2)


def _doaj_page(*records):
    """A DOAJ article-search page for synthetic record dicts."""
    return json.dumps({"results": [
        {"id": r.pop("_id", "rec") or "rec", "bibjson": r} for r in records
    ]}).encode("utf-8")


def _abstract(n_words=60, word="legal"):
    """A register-shaped abstract (well over the stub gate)."""
    return " ".join([word] * n_words)


def _record(doi="10.1/x", journal="Journal of Law", year="2025",
            language="EN", issns=("1234-5678",), abstract=None):
    return {
        "_id": doi,
        "identifier": ([{"id": doi, "type": "doi"}] if doi else []),
        "title": "A legal article",
        "year": year,
        "abstract": abstract if abstract is not None else _abstract(),
        "journal": {"title": journal, "publisher": "Test University Press",
                    "language": [language], "issns": list(issns)},
    }


def _doaj_journals(*journals):
    """A DOAJ journal-search page for (issn, licence) pairs."""
    results = []
    for issn, lic in journals:
        results.append({"bibjson": {
            "title": f"Journal {issn}", "pissn": issn, "eissn": None,
            "publisher": {"country": "XX", "name": "Test Press"},
            "license": ([{"type": lic, "url": "https://example.org"}] if lic
                        else [])}})
    return json.dumps({"results": results}).encode("utf-8")


class TestLegalRegisterLabel(unittest.TestCase):
    """#79: the Gutenberg legal register is historical, not modern legal writing.

    The five texts are 1215-1850 public-domain full texts.  The register is
    kept as its own inferential reference, but every label must say what it
    is, so no claim in the workspace can cite it as modern legal writing or
    as legal abstracts.
    """

    def test_register_label_names_the_historical_scope(self):
        label = rc.LEGAL_REGISTER_LABEL
        self.assertIn("historical", label)
        self.assertIn("not legal abstracts", label)
        self.assertIn("not modern legal writing", label)

    def test_docstring_labels_legal_historical(self):
        head = rc.__doc__.split("Every corpus is written")[0]
        self.assertIn("**historical** legal/political *full texts*", head)
        self.assertIn("**not** legal abstracts", head)
        # the docstring wraps mid-phrase, so match up to the line break
        self.assertIn("**not** modern legal", head)
        # and the modern register is documented as its own corpus
        self.assertIn("legal_abstracts   modern law-journal abstracts", head)

    def test_manifest_legal_rows_are_public_domain(self):
        rows = read_tsv(MANIFEST)
        legal = [r for r in rows if r["corpus"] == "legal"]
        self.assertEqual(len(legal), len(rc.LEGAL))
        for r in legal:
            self.assertEqual(r["license"], "public domain", r["file"])
            self.assertIn("gutenberg.org", r["source"], r["file"])

    def test_expected_historical_texts_present(self):
        rows = read_tsv(MANIFEST)
        files = {r["file"] for r in rows if r["corpus"] == "legal"}
        self.assertEqual(files,
                         {f"legal/{name}.txt" for name, *_ in rc.LEGAL})


class TestDoajRecords(unittest.TestCase):
    """#79: DOAJ article pages become register records, stubs skipped."""

    def test_parses_records_with_provenance(self):
        page = _doaj_page(_record())
        records = rc.parse_doaj_records([page])
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["doi"], "10.1/x")
        self.assertEqual(r["journal"], "Journal of Law")
        self.assertEqual(r["year"], "2025")
        self.assertEqual(r["language"], "EN")
        self.assertEqual(r["issns"], ["1234-5678"])

    def test_stub_abstracts_are_skipped(self):
        # DOAJ editorial stubs and one-word "abstracts" are not register text.
        stubs = [_record(doi=f"10.1/stub{i}",
                         abstract=t) for i, t in enumerate(
                             ["Comment.", "Introduction.",
                              "Abstract not available.", ""])]
        page = _doaj_page(*stubs)
        self.assertEqual(rc.parse_doaj_records([page]), [])

    def test_short_but_real_abstracts_clear_the_gate(self):
        rec = _record(doi="10.1/short", abstract=_abstract(
            rc.MIN_ABSTRACT_WORDS))
        records = rc.parse_doaj_records([_doaj_page(rec)])
        self.assertEqual(len(records), 1)

    def test_missing_doi_falls_back_to_the_record_id(self):
        rec = _record(doi=None)
        records = rc.parse_doaj_records([_doaj_page(rec)])
        self.assertEqual(records[0]["doi"], "doaj:rec")

    def test_markup_is_stripped_and_entities_decoded(self):
        rec = _record(doi="10.1/markup",
                      abstract="<p>The law &amp; the state &lt;br&gt; "
                               + _abstract(40) + "</p>")
        records = rc.parse_doaj_records([_doaj_page(rec)])
        text = records[0]["abstract"]
        self.assertNotIn("<p>", text)
        self.assertNotIn("&amp;", text)
        self.assertIn("The law & the state", text)

    def test_non_json_page_is_ignored(self):
        self.assertEqual(rc.parse_doaj_records([b"<html>503</html>"]), [])

    def test_issn_less_record_header_round_trips(self):
        # A record with no ISSN must still be readable back from the corpus
        # file, so the ledger can always be bound to the corpus bytes.
        records = rc.parse_doaj_records(
            [_doaj_page(_record(doi="10.1/noissn", issns=()))])
        data = rc.doaj_corpus_bytes(records)
        self.assertEqual(rc.parse_dois(data), ["10.1/noissn"])
        back = rc.parse_records_from_corpus(data)
        self.assertEqual(back[0]["doi"], "10.1/noissn")
        self.assertEqual(back[0]["issns"], [])

    def test_corpus_header_round_trips(self):
        # The header carries DOI + ISSNs so a cached corpus can re-resolve
        # its licences from its own bytes.
        records = rc.parse_doaj_records(
            [_doaj_page(_record(), _record(doi="10.2/y", issns=("9", "8")))])
        data = rc.doaj_corpus_bytes(records)
        self.assertEqual(rc.parse_dois(data), ["10.1/x", "10.2/y"])
        back = rc.parse_records_from_corpus(data)
        self.assertEqual([r["doi"] for r in back], ["10.1/x", "10.2/y"])
        self.assertEqual(back[0]["journal"], "Journal of Law")
        self.assertEqual(back[0]["year"], "2025")
        self.assertEqual(back[0]["language"], "EN")
        self.assertEqual(back[1]["issns"], ["9", "8"])
        # the abstract text survives beside the header
        self.assertIn(_abstract(), data.decode("utf-8"))


class TestDoajRights(unittest.TestCase):
    """#79: the licence is a per-journal fact, resolved and never assumed."""

    def test_rights_class_derivation(self):
        self.assertEqual(rc.doaj_rights_class("CC BY"), "open_access")
        self.assertEqual(rc.doaj_rights_class("CC BY-NC-ND"), "open_access")
        self.assertEqual(rc.doaj_rights_class(""), "free_to_read")

    def test_resolves_licences_by_issn(self):
        records = rc.parse_doaj_records([_doaj_page(_record())])
        rows = rc.resolve_doaj_licences(
            records, log=lambda m: None,
            fetch=lambda url, path, log=print, **k: _doaj_journals(
                ("1234-5678", "CC BY")))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["license"], "CC BY")
        self.assertEqual(rows[0]["license_class"], "open_access")
        self.assertEqual(rows[0]["publisher"], "Test Press")
        self.assertEqual(rows[0]["verified_via"], "DOAJ journal record")
        self.assertTrue(rows[0]["verified_at"])

    def test_missing_journal_record_is_unverified(self):
        records = rc.parse_doaj_records([_doaj_page(_record())])
        rows = rc.resolve_doaj_licences(
            records, log=lambda m: None,
            fetch=lambda url, path, log=print, **k: _doaj_journals())
        self.assertEqual(rows[0]["license_class"], "unverified")
        self.assertEqual(rows[0]["license"], "")
        self.assertIn("no record returned", rows[0]["verified_via"])

    def test_journal_without_a_licence_is_free_to_read_not_open(self):
        records = rc.parse_doaj_records([_doaj_page(_record())])
        rows = rc.resolve_doaj_licences(
            records, log=lambda m: None,
            fetch=lambda url, path, log=print, **k: _doaj_journals(
                ("1234-5678", "")))
        self.assertEqual(rows[0]["license_class"], "free_to_read")
        self.assertEqual(rows[0]["license"], "")

    def test_record_without_issn_is_unverified_by_stated_reason(self):
        records = rc.parse_doaj_records(
            [_doaj_page(_record(issns=()))])
        rows = rc.resolve_doaj_licences(
            records, log=lambda m: None,
            fetch=lambda url, path, log=print, **k: _doaj_journals())
        self.assertEqual(rows[0]["license_class"], "unverified")
        self.assertIn("no ISSN", rows[0]["verified_via"])
        self.assertTrue(rows[0]["verified_at"])

    def test_survives_a_non_json_response(self):
        records = rc.parse_doaj_records([_doaj_page(_record())])
        rows = rc.resolve_doaj_licences(
            records, log=lambda m: None,
            fetch=lambda url, path, log=print, **k: b"<html>502</html>")
        self.assertEqual([r["license_class"] for r in rows], ["unverified"])

    def test_batches_issn_lookups(self):
        calls = []
        records = rc.parse_doaj_records(
            [_doaj_page(*[_record(doi=f"10.1/{i}", issns=(f"{i:04d}-0001",))
                          for i in range(60)])])

        def fetch(url, path, log=print, **k):
            calls.append(url)
            return _doaj_journals()

        rows = rc.resolve_doaj_licences(records, log=lambda m: None,
                                        fetch=fetch)
        self.assertEqual(len(rows), 60)
        self.assertGreaterEqual(len(calls), 2,
                                "60 ISSNs must batch into several requests")
        self.assertIn(f"pageSize={rc.DOAJ_RIGHTS_CHUNK}", calls[0])

    def test_ledger_roundtrip_is_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rights.tsv"
            rc.write_doaj_rights(path, [
                {"doi": "10.9/z", "journal": "J", "publisher": "P",
                 "year": "2025", "language": "EN", "issn": "1",
                 "license": "CC BY", "license_class": "open_access",
                 "verified_via": "DOAJ journal record",
                 "verified_at": "2026-10-16"},
                {"doi": "10.1/a", "journal": "K", "publisher": "Q",
                 "year": "2024", "language": "EN", "issn": "2",
                 "license": "", "license_class": "unverified",
                 "verified_via": "DOAJ journal record (no record returned)",
                 "verified_at": "2026-10-16"},
            ])
            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0],
                             "\t".join(rc.DOAJ_RIGHTS_HEADER))
            back = rc.read_doaj_rights(path)
            self.assertEqual([r["doi"] for r in back], ["10.1/a", "10.9/z"])

    def test_ledger_covers_exact_doi_sets_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rights.tsv"
            rc.write_doaj_rights(path, [
                {"doi": d, "journal": "J", "publisher": "P", "year": "2025",
                 "language": "EN", "issn": "1", "license": "CC BY",
                 "license_class": "open_access",
                 "verified_via": "DOAJ journal record",
                 "verified_at": "2026-10-16"} for d in ("10.1/a", "10.2/b")])
            self.assertTrue(rc.doaj_rights_ledger_covers(path,
                                                         ["10.1/a", "10.2/b"]))
            self.assertFalse(rc.doaj_rights_ledger_covers(path, ["10.1/a"]),
                             "a changed corpus invalidates the ledger")
            self.assertFalse(rc.doaj_rights_ledger_covers(
                path, ["10.1/a", "10.3/c"]),
                "an uncovered record is not a covering ledger")
            self.assertFalse(rc.doaj_rights_ledger_covers(path, []))
            self.assertFalse(rc.doaj_rights_ledger_covers(
                Path(tmp) / "absent.tsv", ["10.1/a"]))

    # -- build wiring ------------------------------------------------------
    def _legal_world(self, tmp, ledger_set=None):
        """A cached corpus under *tmp*, with a ledger over ``ledger_set``."""
        out = Path(tmp) / "out"
        (out / "legal_abstracts").mkdir(parents=True)
        records = rc.parse_doaj_records(
            [_doaj_page(_record(doi="10.1/a"), _record(doi="10.2/b"))])
        (out / "legal_abstracts" / "legal_abstracts.txt").write_bytes(
            rc.doaj_corpus_bytes(records))
        rows = [{"doi": r["doi"], "journal": r["journal"],
                 "publisher": "Test Press", "year": r["year"],
                 "language": r["language"], "issn": "/".join(r["issns"]),
                 "license": "", "license_class": "unverified",
                 "verified_via": "DOAJ journal record (no record returned)",
                 "verified_at": "2026-10-16"} for r in records]
        rc.write_doaj_rights(out / rc.DOAJ_RIGHTS_FILE,
                             [r for r in rows if r["doi"] in
                              (ledger_set or [r["doi"] for r in rows])])
        return out

    def test_cached_corpus_with_a_covering_ledger_needs_no_network(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            out = self._legal_world(tmp)
            with patched(OUT=out, MIN_TOKENS={"legal_abstracts": 100},
                         download=no_network):
                rows = rc.build_legal_abstracts(logs.append,
                                                allow_download=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["license"],
                             rc.LIVE_SOURCES["legal_abstracts"][1])
            self.assertTrue(any("covers" in m for m in logs), logs)

    def test_stale_ledger_warns_instead_of_fetching_in_cache_only_mode(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            out = self._legal_world(tmp, ledger_set=("10.1/a",))
            with patched(OUT=out, MIN_TOKENS={"legal_abstracts": 100},
                         download=no_network):
                rc.build_legal_abstracts(logs.append, allow_download=False)
            self.assertTrue(any("unverified" in m and "WARNING" in m
                                for m in logs), logs)
            self.assertTrue(rc.doaj_rights_ledger_covers(
                out / rc.DOAJ_RIGHTS_FILE, ["10.1/a"]))

    def test_stale_ledger_is_resolved_in_download_mode(self):
        logs = []

        def fetch(url, path, log=print, **k):
            return _doaj_journals(("1234-5678", "CC BY"))

        with tempfile.TemporaryDirectory() as tmp:
            out = self._legal_world(tmp, ledger_set=("10.1/a",))
            with patched(OUT=out, MIN_TOKENS={"legal_abstracts": 100},
                         download=fetch):
                rc.build_legal_abstracts(logs.append)
            self.assertTrue(rc.doaj_rights_ledger_covers(
                out / rc.DOAJ_RIGHTS_FILE, ["10.1/a", "10.2/b"]),
                "download mode must re-resolve rather than keep a stale ledger")
            ledger = rc.read_doaj_rights(out / rc.DOAJ_RIGHTS_FILE)
            self.assertEqual({r["license"] for r in ledger}, {"CC BY"})
            self.assertTrue(any("records verified" in m for m in logs), logs)


class TestLegalAbstractsInTree(unittest.TestCase):
    """The built corpus ships a real, complete per-record licence ledger."""

    def test_manifest_row_and_license_cell(self):
        rows = read_tsv(MANIFEST)
        la = [r for r in rows if r["corpus"] == "legal_abstracts"]
        self.assertEqual(len(la), 1)
        self.assertEqual(la[0]["file"],
                         rc.LIVE_FILES["legal_abstracts"])
        self.assertIn("doaj.org", la[0]["source"])
        lic = la[0]["license"]
        self.assertIn("rights", lic.lower())
        self.assertIn(rc.DOAJ_RIGHTS_FILE, lic)

    def test_corpus_clears_its_gate(self):
        p = CORPORA / rc.LIVE_FILES["legal_abstracts"]
        self.assertTrue(p.exists(), "build the legal_abstracts corpus first")
        self.assertTrue(rc.corpus_ok(p, rc.MIN_TOKENS["legal_abstracts"]))

    def test_tree_ledger_covers_the_snapshot(self):
        data = (CORPORA / rc.LIVE_FILES["legal_abstracts"]).read_bytes()
        dois = rc.parse_dois(data)
        ledger = rc.read_doaj_rights(CORPORA / rc.DOAJ_RIGHTS_FILE)
        self.assertTrue(dois)
        self.assertEqual({r["doi"] for r in ledger}, set(dois),
                         "the ledger covers the snapshot exactly")
        self.assertEqual(len(ledger), len(set(dois)),
                         "one row per DOI")
        for row in ledger:
            self.assertTrue(row["verified_at"], "a rights claim carries a date")
            self.assertTrue(row["verified_via"])
            self.assertIn(row["license_class"],
                          ("open_access", "free_to_read", "unverified"))

    def test_tree_ledger_names_real_licences(self):
        ledger = rc.read_doaj_rights(CORPORA / rc.DOAJ_RIGHTS_FILE)
        self.assertTrue(ledger)
        licences = {r["license"] for r in ledger if r["license"]}
        self.assertTrue(all(l.startswith("CC ") for l in licences),
                        f"DOAJ law-journal licences are CC: {licences}")
        classes = {r["license_class"] for r in ledger}
        self.assertIn("open_access", classes)


if __name__ == "__main__":
    unittest.main()
