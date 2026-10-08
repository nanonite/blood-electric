#!/usr/bin/env python3
"""#56 Keyness harness -> analysis/keyness/

Compares the vocabulary of *Blood Electric* against each reference corpus
assembled by ``work/reference_corpora.py`` and ranks the words that are
diagnostic of the book.

**Book side.**  The #17 *word-part* layer (``part_index.tsv``, role ``w``) -
the individual words inside the maximal units, so ``gene=TV`` contributes
``gene`` and ``TV``.  This is the layer comparable to ordinary English words;
the unit layer fuses compounds and the token layer adds operators, so neither
is comparable to a reference corpus.  Parts are markup-screened with #19's
ledger plus #27's tag-run detector (a part is dropped when its unit uid is a
#19 markup residue, when its ``(page, start)`` falls inside a detected tag
span, or when its lower-cased surface is a #19 junk/markup type).  Nothing
else is repaired.

**Reference side.**  Each corpus is tokenised with the #24 English word regex
(``[A-Za-z]+(?:['’][A-Za-z]+)*``) and lower-cased.  Excerpt files (cyberpunk,
cut-up/Surrealist, Finnegans Wake) have their ``#`` header and ``--``
attribution lines stripped so only the quoted register text is counted.

**Statistics.**  For every type in the union of the book and a reference
corpus, the log-likelihood keyness statistic G² (Dunning 1993; Rayson &
Garside 2000) with its chi-square p-value, a Benjamini-Hochberg q-value, and
the log-ratio effect size (0.5-smoothed).  G² is the ranking measure; the
sign is positive when the word is over-represented in the book and negative
when it is under-represented.  A word is *key* when it is over-represented
(positive G²) and *anti-key* when it is under-represented (negative G²).

**Multiple testing.**  The union runs to tens of thousands of types per
corpus, so the significance gate is the FDR-controlled **q < 0.05**, not the
raw p < 0.05: uncorrected, that many tests admit hundreds of false positives
by construction.  Both p and q are written per word; p is retained because
the keyness literature (Dunning; Rayson & Garside) reports uncorrected p,
and the pair shows how much the correction moves.

Reproduce::

    conda activate marker
    python work/reference_corpora.py     # first: assemble the corpora
    python work/keyness.py
    python work/test_keyness.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
PART_INDEX = ROOT / "analysis" / "tokenizer" / "part_index.tsv"
MARKUP = ROOT / "analysis" / "neologism" / "markup_residue.tsv"
SCREENING = ROOT / "analysis" / "neologism" / "screening.tsv"
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
CORPORA = ROOT / "analysis" / "reference_corpora"
OUT = ROOT / "analysis" / "keyness"

SCHEMA = "keyness/1"

CORPUS_DIRS = [
    "general_english", "cyberpunk", "cutup_surrealist", "finnegans_wake",
    "medical", "cs", "legal",
]
EXCERPT_CORPORA = {"cyberpunk", "cutup_surrealist", "finnegans_wake"}

# A word is "key" at BH-adjusted q < Q_THRESHOLD (false-discovery-rate
# controlled: with ~48k types in the union an uncorrected p < 0.05 gate would
# admit hundreds of false positives) and at least this many book hits.
# P_THRESHOLD is still reported per word; q is the gate.
Q_THRESHOLD = 0.05
P_THRESHOLD = 0.05
MIN_BOOK_FREQ = 5
TOP_N = 50
# Below this a reference corpus is too small to carry a keyness test; its
# tables are reported but flagged as non-inferential (see the report caveat).
MIN_TOKENS_INFERENTIAL = 10_000

EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")

# ---------------------------------------------------------------------------
# io helpers (quote-unsafe TSVs: split on tab, never a csv reader -- #17)
# ---------------------------------------------------------------------------
def cell(value) -> str:
    s = "" if value is None else str(value)
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(cell(r.get(h, "")) for h in header) + "\n")


def read_tsv(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# markup screen (#19 ledger + #27 tag-run detector)
# ---------------------------------------------------------------------------
_PAGE_RE = re.compile(r"<!-- page (\d+) -->")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAGS = {
    "a", "p", "pre", "br", "li", "hr", "th", "td", "tr", "table", "tbody",
    "thead", "body", "html", "head", "span", "div", "img", "meta", "link",
    "b", "i", "u", "em", "strong", "font", "small", "big", "center",
    "script", "style", "title", "form", "input", "button", "ol", "ul", "dl",
    "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6", "nav", "main", "footer",
    "header", "section", "article", "blockquote", "code", "q", "sup", "sub",
}
STRICT_TAGS = {"a", "th", "p", "pre", "li", "br", "hr", "td", "tr", "table",
               "img", "meta", "link", "input", "script", "style"}
_TAG_RUN_RE = re.compile(
    r"</?([a-zA-Z][a-zA-Z0-9]*)"
    r"((?:\s+[a-zA-Z-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))*)\s*/?>")


def find_markup_spans(body: str):
    """(start, end) spans of real HTML tag runs in a page body (#27)."""
    spans = []
    for m in _TAG_RUN_RE.finditer(body):
        name = m.group(1).lower()
        if name not in HTML_TAGS:
            continue
        closing = body[m.start():m.start() + 2] == "</"
        if closing or m.group(2).strip() or name in STRICT_TAGS:
            spans.append((m.start(), m.end()))
    return spans


def load_markup():
    """(spans, uids, types) - the #27 exclusion sets."""
    spans = collections.defaultdict(list)
    uids = set()
    for r in read_tsv(MARKUP):
        uids.add(int(r["uid"]))
        spans[int(r["page"])].append((int(r["start"]), int(r["end"])))
    text = LAYOUT.read_text(encoding="utf-8", errors="ignore")
    parts = _PAGE_RE.split(text)
    bodies = {int(parts[k]): parts[k + 1] for k in range(1, len(parts) - 1, 2)}
    for page, body in bodies.items():
        spans[page].extend(find_markup_spans(body))
    types = set()
    if SCREENING.exists():
        for r in read_tsv(SCREENING):
            if r["category"] in ("junk", "markup_residue"):
                types.add(r["unit"].lower())
    return spans, uids, types


# ---------------------------------------------------------------------------
# book side: screened word parts
# ---------------------------------------------------------------------------
def load_book_words(log=print):
    """Counter of the book's markup-screened word parts, lower-cased."""
    spans, markup_uids, markup_types = load_markup()
    counts = collections.Counter()
    n_parts = n_dropped = 0
    for r in read_tsv(PART_INDEX):
        if r["role"] != "w":
            continue
        n_parts += 1
        uid = int(r["uid"])
        page = int(r["page"])
        start = int(r["start"])
        surface = r["text"]
        if (uid in markup_uids or surface.lower() in markup_types
                or any(a <= start < b for a, b in spans.get(page, ()))):
            n_dropped += 1
            continue
        counts[surface.lower()] += 1
    log(f"[keyness] book word parts: {n_parts:,} total, {n_dropped:,} "
        f"markup-screened, {sum(counts.values()):,} kept, "
        f"{len(counts):,} types")
    return counts, {"parts": n_parts, "dropped": n_dropped,
                    "kept": sum(counts.values()), "types": len(counts)}


# ---------------------------------------------------------------------------
# reference side
# ---------------------------------------------------------------------------
def load_reference_words(corpus: str, log=print):
    """Counter of a reference corpus's words, lower-cased."""
    cdir = CORPORA / corpus
    files = corpus_files(corpus)
    if not files:
        raise FileNotFoundError(f"no corpus files in {cdir}")
    counts = collections.Counter()
    n_files = 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        if corpus in EXCERPT_CORPORA:
            text = "\n".join(
                ln for ln in text.splitlines()
                if not ln.lstrip().startswith(("#", "--")))
        for tok in EN_WORD_RE.findall(text):
            counts[tok.lower()] += 1
        n_files += 1
    log(f"[keyness] {corpus}: {n_files} files, {sum(counts.values()):,} tokens, "
        f"{len(counts):,} types")
    return counts


# ---------------------------------------------------------------------------
# keyness statistics
# ---------------------------------------------------------------------------
def log_likelihood(o11: int, o12: int, n1: int, n2: int):
    """Signed log-likelihood G^2 (Dunning 1993; Rayson & Garside 2000).

    Positive = over-represented in the book, negative = under-represented.
    """
    if o11 == 0 and o12 == 0:
        return 0.0
    n = n1 + n2
    e1 = n1 * (o11 + o12) / n
    e2 = n2 * (o11 + o12) / n
    g2 = 0.0
    if o11 > 0:
        g2 += o11 * math.log(o11 / e1)
    if o12 > 0:
        g2 += o12 * math.log(o12 / e2)
    g2 *= 2.0
    return g2 if o11 >= e1 else -g2


def log_ratio(o11: int, o12: int, n1: int, n2: int) -> float:
    """0.5-smoothed log-ratio effect size (log fold change, book vs ref)."""
    return math.log(((o11 + 0.5) / n1) / ((o12 + 0.5) / n2))


def benjamini_hochberg(p_values):
    """BH (1995) step-up adjusted p-values (q-values).

    The union of book and reference types runs to tens of thousands, so an
    uncorrected p < 0.05 gate admits hundreds of false positives by
    construction.  q is the smallest FDR at which the word is called
    significant, which is the honest gate for a ranked keyness table.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    q = [1.0] * n
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, p_values[i] * n / (rank + 1))
        q[i] = min(1.0, running)
    return q


def keyness_rows(book: collections.Counter, ref: collections.Counter):
    """One row per type in the union, ranked by signed G^2 descending."""
    n1 = sum(book.values())
    n2 = sum(ref.values())
    rows = []
    for word in set(book) | set(ref):
        o11 = book.get(word, 0)
        o12 = ref.get(word, 0)
        g2 = log_likelihood(o11, o12, n1, n2)
        rows.append({
            "word": word,
            "book_freq": o11,
            "ref_freq": o12,
            "book_rel": 1e6 * o11 / n1,
            "ref_rel": 1e6 * o12 / n2,
            "log_likelihood": g2,
            "p_value": float(chi2.sf(abs(g2), 1)),
            "log_ratio": log_ratio(o11, o12, n1, n2),
        })
    for row, q in zip(rows, benjamini_hochberg([r["p_value"] for r in rows])):
        row["q_value"] = q
    rows.sort(key=lambda r: (-r["log_likelihood"], -r["book_freq"], r["word"]))
    return rows, n1, n2


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def top_overlap(list_a, list_b, n: int = TOP_N):
    """Jaccard overlap of the top-*n* key words of two corpora."""
    a = {r["word"] for r in list_a[:n]}
    b = {r["word"] for r in list_b[:n]}
    union = a | b
    if not union:
        return 0.0, 0, len(a & b)
    return len(a & b) / len(union), len(a), len(a & b)


def reference_cores(per_corpus, key_lists):
    """(core, inferential, ge_only) - which key findings survive a re-reference?

    ``core`` is the set of words in the top-``TOP_N`` key list of *every*
    corpus that is large enough to produce one; ``ge_only`` is the residue of
    general-English keyness that no technical register reproduces.
    """
    inferential = [c for c in CORPUS_DIRS
                   if per_corpus[c]["tokens"] >= MIN_TOKENS_INFERENTIAL]
    keys = {c: {r["word"] for r in key_lists[c]} for c in CORPUS_DIRS}
    base = key_lists["general_english"]
    core = {r["word"] for r in base[:TOP_N]}
    for c in inferential:
        if c == "general_english":
            continue
        core &= {r["word"] for r in key_lists[c][:TOP_N]}
    technical = ("medical", "cs", "legal")
    ge_only = sorted(keys["general_english"]
                    - set().union(*(keys[c] for c in technical)))
    return sorted(core), inferential, ge_only


def cross_corpus_section(per_corpus, key_lists, cores, log=print):
    """Which key findings survive a change of reference corpus?

    A keyness table is only as good as its reference.  The same word can be
    diagnostic against general English and unremarkable against a technical
    register, so the overlap between per-corpus key lists is what separates a
    stable finding about the book from a reference artifact.

    Only the corpora above ``MIN_TOKENS_INFERENTIAL`` can carry a top-N key
    list at all -- an excerpt corpus of ~50 tokens admits almost nothing
    through the q-gate, so folding it into the intersection would manufacture
    an empty "nothing is stable" result out of corpus size alone.  Those rows
    are still printed, flagged, and excluded from the core.
    """
    core, inferential, ge_only = cores
    base = key_lists["general_english"]
    lines = []
    lines.append("## Cross-corpus stability of the key words")
    lines.append("")
    lines.append(
        f"Top-{TOP_N} key words per corpus, compared with the "
        f"`general_english` list. High overlap = the word is diagnostic of "
        f"the book regardless of which reference is used; low overlap = the "
        f"word is largely an artifact of that particular reference.")
    lines.append("")
    lines.append("| corpus | tokens | inferential | top-N keys "
                 "| shared with general_english | Jaccard |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for corpus in CORPUS_DIRS:
        top = key_lists[corpus]
        j, size, shared = top_overlap(top, base)
        if corpus in inferential:
            lines.append(f"| {corpus} | {per_corpus[corpus]['tokens']:,} | yes "
                         f"| {size} | {shared} | {j:.3f} |")
        else:
            note = f"{len(top)} key" + ("" if len(top) == 1 else "s")
            lines.append(f"| {corpus} | {per_corpus[corpus]['tokens']:,} | no "
                         f"({note}) | {size} | - | - |")
    lines.append("")
    if core:
        n_core = f"{len(core)} word" + ("" if len(core) == 1 else "s")
        lines.append("**Reference-independent core.** "
                     f"{n_core} are in the top {TOP_N} of every "
                     "inferential corpus (" + ", ".join(f"`{w}`" for w in core)
                     + "). These are diagnostic of the book against general "
                     "English prose *and* against the medical, CS and legal "
                     "registers, so they are findings about the book and not "
                     "about a reference.")
    else:
        lines.append(f"No word is in the top {TOP_N} of every inferential "
                     "corpus; the keyness profile is reference-dependent "
                     "throughout. That is the expected result for a book whose "
                     "lexis is genre-crossed, and it is why no single "
                     "reference is treated as decisive here.")
    lines.append("")
    if ge_only:
        lines.append(
            f"**Reference-dependent residue.** {len(ge_only)} of the "
            f"general-English key words are key against *no* technical "
            "register (" + ", ".join(f"`{w}`" for w in ge_only) + "). They are "
            "over-used relative to public-domain prose but not against modern "
            "medical, CS or legal writing, so their keyness is at least partly "
            "an artifact of how old the general-English reference is; the "
            "corpora that decide them are `medical` / `cs`, not "
            "`general_english`.")
        lines.append("")
    log(f"[keyness] reference-independent core: {len(core)} words; "
        f"general-English-only residue: {len(ge_only)}")
    return lines


def write_report(path: Path, book_stats, per_corpus, key_lists, antikey_lists,
                 cores, log=print):
    core, _inferential, ge_only = cores
    core_words = set(core)
    # Words in the reference-independent core that the general-English
    # reference never uses: their keyness cannot then be an artifact of a
    # frequency difference.
    zero_ref = [r["word"] for r in key_lists["general_english"]
                if r["word"] in core_words and r["ref_freq"] == 0]
    L = []
    a = L.append
    a("# Keyness report - Blood Electric vs reference corpora")
    a("")
    a("Book side: the #17 word-part layer, markup-screened with #19's ledger + "
      "#27's tag-run detector. Reference side: each corpus tokenised with the "
      "#24 English word regex. Statistic: signed log-likelihood G\u00b2 (Dunning "
      "1993; Rayson & Garside 2000), chi-square p-value, Benjamini-Hochberg "
      "q-value, 0.5-smoothed log-ratio effect size. Positive G\u00b2 = "
      "over-represented in the book.")
    a("")
    a("**Multiple testing.** The union of book and reference types runs to tens "
      f"of thousands of tests per corpus, so the gate below is the "
      f"BH-adjusted **q < {Q_THRESHOLD}**, not the raw p < {P_THRESHOLD}: at "
      "that many tests an uncorrected p-gate admits hundreds of false "
      "positives by construction. `q` is the smallest false-discovery rate at "
      "which the word is called significant. Both p and q are written per word.")
    a("")
    a("**Headline.** "
      f"{len(core)} words are in the top {TOP_N} of *every* inferential "
      "reference, so they are diagnostic of the book rather than of a "
      "reference (full list in the cross-corpus section). "
      f"{len(zero_ref)} of them never appear in the general-English reference "
      f"at all. {len(ge_only)} of the {len(key_lists['general_english'])} "
      "general-English key words survive no technical register and are "
      "reported as reference residue. The excerpt corpora carry no keyness "
      "test of their own (they are too small -- see the caveat), so no claim "
      "here rests on them.")
    a("")
    a("## Sizes")
    a("")
    a("| corpus | files | tokens | types | key words | anti-key words |")
    a("|---|---:|---:|---:|---:|---:|")
    a(f"| **book (word parts)** | - | {book_stats['kept']:,} | "
      f"{book_stats['types']:,} | - | - |")
    for corpus, st in per_corpus.items():
        a(f"| {corpus} | {st['files']} | {st['tokens']:,} | {st['types']:,} "
          f"| {len(key_lists[corpus]):,} | {len(antikey_lists[corpus]):,} |")
    a("")
    a(f"Book word parts: {book_stats['parts']:,} total, "
      f"{book_stats['dropped']:,} markup-screened, {book_stats['kept']:,} kept.")
    a("")
    # The copyright-cleared excerpt corpora are orders of magnitude smaller
    # than the others.  A chi-square on a 50-token reference is formally
    # computable and evidentially worthless: a single occurrence of a word
    # that the book happens not to use is "significantly anti-key".  Say so
    # in the report rather than letting the reader assume the tables are
    # comparable across rows.
    small = {c: st["tokens"] for c, st in per_corpus.items()
             if st["tokens"] < MIN_TOKENS_INFERENTIAL}
    if small:
        a(f"**Corpus-size caveat.** Every table below is produced by the same "
          f"code, but they are *not* equally evidential. A corpus under "
          f"{MIN_TOKENS_INFERENTIAL:,} tokens "
          f"({', '.join(f'`{c}` ({n:,})' for c, n in sorted(small.items()))}) "
          "cannot support a keyness test: these are short fair-use excerpt "
          "corpora, included because the named authors are in copyright, and "
          "a single occurrence of a word the book happens not to use comes out "
          "as a large 'significant' anti-key. Read those rows as *which words "
          "the excerpts contain*, not as a measured register difference, and "
          "weight the conclusions from the large corpora "
          f"(`general_english`, `medical`, `cs`, `legal`).")
        a("")
    a("## Key words (over-represented in the book)")
    a("")
    a("Top "
      f"{TOP_N} by G² at q < {Q_THRESHOLD} and book_freq >= {MIN_BOOK_FREQ}. "
      "Full tables in `keyness_<corpus>.tsv`.")
    a("")
    for corpus in CORPUS_DIRS:
        a(f"### vs {corpus}")
        a("")
        a("| word | book_freq | ref_freq | book_rel | ref_rel | G² | p | q | log_ratio |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in key_lists[corpus][:TOP_N]:
            a(f"| {r['word']} | {r['book_freq']} | {r['ref_freq']} | "
              f"{fmt(r['book_rel'])} | {fmt(r['ref_rel'])} | "
              f"{fmt(r['log_likelihood'])} | {r['p_value']:.2e} | "
              f"{r['q_value']:.2e} | {fmt(r['log_ratio'])} |")
        a("")
    a("## Anti-key words (under-represented in the book)")
    a("")
    a("Top "
      f"{TOP_N} by |G²| at q < {Q_THRESHOLD}. These are the register words "
      "the book avoids.")
    a("")
    for corpus in CORPUS_DIRS:
        a(f"### vs {corpus}")
        a("")
        a("| word | book_freq | ref_freq | book_rel | ref_rel | G² | p | q | log_ratio |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in antikey_lists[corpus][:TOP_N]:
            a(f"| {r['word']} | {r['book_freq']} | {r['ref_freq']} | "
              f"{fmt(r['book_rel'])} | {fmt(r['ref_rel'])} | "
              f"{fmt(r['log_likelihood'])} | {r['p_value']:.2e} | "
              f"{r['q_value']:.2e} | {fmt(r['log_ratio'])} |")
        a("")
    L.extend(cross_corpus_section(per_corpus, key_lists, cores, log))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def corpus_files(corpus: str):
    """The counted ``.txt`` files of a corpus (staging ``_`` files excluded)."""
    return sorted(f for f in (CORPORA / corpus).glob("*.txt")
                  if not f.name.startswith("_"))


def run(log=print):
    OUT.mkdir(parents=True, exist_ok=True)
    book, book_stats = load_book_words(log)
    per_corpus = {}
    key_lists, antikey_lists = {}, {}
    for corpus in CORPUS_DIRS:
        ref = load_reference_words(corpus, log)
        rows, _n1, n2 = keyness_rows(book, ref)
        per_corpus[corpus] = {
            "files": len(corpus_files(corpus)),
            "tokens": n2, "types": len(ref),
        }
        write_tsv(OUT / f"keyness_{corpus}.tsv",
                  ["word", "book_freq", "ref_freq", "book_rel", "ref_rel",
                   "log_likelihood", "p_value", "q_value", "log_ratio"], rows)
        # FDR-controlled gate; see the module docstring and the report note.
        keyed = [r for r in rows if r["log_likelihood"] > 0
                 and r["q_value"] < Q_THRESHOLD
                 and r["book_freq"] >= MIN_BOOK_FREQ]
        antikey = [r for r in rows if r["log_likelihood"] < 0
                   and r["q_value"] < Q_THRESHOLD]
        antikey.sort(key=lambda r: (r["log_likelihood"], -r["ref_freq"], r["word"]))
        key_lists[corpus] = keyed
        antikey_lists[corpus] = antikey
        log(f"[keyness] {corpus}: {len(keyed)} key, {len(antikey)} anti-key "
            f"(q < {Q_THRESHOLD}; of {len(rows):,} tested types)")
    cores = reference_cores(per_corpus, key_lists)
    write_report(OUT / "keyness_report.md", book_stats, per_corpus,
                 key_lists, antikey_lists, cores, log)
    (OUT / "summary.json").write_text(json.dumps({
        "schema": SCHEMA,
        "book": book_stats,
        "corpora": per_corpus,
        "q_threshold": Q_THRESHOLD,
        "p_threshold": P_THRESHOLD,
        "min_book_freq": MIN_BOOK_FREQ,
    }, indent=2), encoding="utf-8")
    log(f"[keyness] report -> {OUT/'keyness_report.md'}")
    return 0


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    OUT = Path(args.out)
    return run()


if __name__ == "__main__":
    sys.exit(main())
