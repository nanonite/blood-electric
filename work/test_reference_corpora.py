#!/usr/bin/env python3
"""Regression suite for #56 reference corpora.

Checks the invariants that matter: the manifest structure and provenance, the
fair-use excerpt files (embedded constants), the corpus sizes, the hash
verification of the pinned #24 references, the Gutenberg strip rule, and the
download retry logic.  The live PubMed / arXiv corpora are read from the
cache built by ``python work/reference_corpora.py``; the suite does not
re-query them.

    python work/test_reference_corpora.py
"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reference_corpora as rc

ROOT = rc.ROOT
CORPORA = rc.OUT
MANIFEST = CORPORA / "manifest.tsv"


def read_tsv(path):
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh if line.strip()]


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
