#!/usr/bin/env python3
"""Regression suite for #56 reference corpora.

Checks the invariants that matter: the manifest structure and provenance (and
its ``hash_role`` semantics: enforced pins vs observed checksums with drift
reporting), the fair-use excerpt files (embedded constants), the corpus
sizes, the hash verification of the pinned #24 references, the Gutenberg
strip rule, the download retry logic, and the #80 guarantees -- ``--no-download``
is cache-only for *every* source (general English and legal included), and
arXiv summaries are entity-decoded before tokenization.  The live PubMed /
arXiv corpora are read from the cache built by ``python work/reference_corpora.py``;
the suite does not re-query them.

    python work/test_reference_corpora.py
"""
import hashlib
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


if __name__ == "__main__":
    unittest.main()
