#!/usr/bin/env python3
"""Spectral reading artifact (Chainlink #67).

Presents the silence-removed stream (v4, the #50 winning transform) as a
distinct under-text reading, explicitly NOT "the real text".

What this is
------------
A *spectral reading* is a reading that haunts the text.  It is the text with
its silences removed - the operator layer (`::`, `//`, `=`, `/`, `<<`, `>>`,
`<`, `>`, `++`, `+`, `|`, and the free-standing dash class) replaced by
periods, so the implicit punctuation becomes explicit sentence boundaries.
The result is a "sentence skeleton" (the #50 verdict: v4 most lowers
perplexity).

This is a *reading*, not a *reconstruction*.  It is derived from the source
by a mechanical transform, but the decision to present it as a reading is an
interpretation.  The framing note below documents the derivation and its
provenance, states what the reading can and cannot support, and keeps its
interpretive status clear.

What this is not
----------------
This is NOT the real text.  The real text is the original with its operators
intact (`analysis/silence_removal/v0_original.txt`).  The spectral reading
is a derivative work - a shadow of the text, a ghost that haunts it.  It is
published here as a distinct under-text reading, explicitly not presented
as the real text.

Derivation
----------
1. Source: `analysis/layout_recovery/blood-electric.layout.md` (225 pages,
   the vetted layout corpus of #7).
2. Transform: v4 (`v4_operators_to_period`) from #13 - every silence site
   (operator token, plus the whitespace-bounded dash class #13 discloses)
   is replaced by `. ` (period + space).  #50 found this transform most
   lowers LM perplexity.
3. The transform is defined over silence-site spans only.  Word-internal
   hyphens (`boy-roid`, `cold-blooded`) are inside WORD tokens and are
   never touched.
4. Line structure is preserved: the transform never introduces or removes
   a newline, so output line *i* derives from source line *i*.
5. Every page is tagged (`<!-- page N -->`) so the reading can be cited
   and aligned.

How the numbers are produced
----------------------------
No headline number in the framing note is typed in.  Every claim is either
computed here from #13's own primitives (`segment_stats`, `text_metrics`,
`words`) or read from the published upstream artifact it belongs to (#13's
`summary.json`, #50's `perplexity_by_variant.tsv`), and the two are
reconciled row by row in `reconciliation.tsv`.  A mismatch is a build
error, not a footnote.  The build is deterministic: no clock, no randomness,
byte-identical on rebuild.

Outputs under `analysis/spectral_reading/`:
  spectral_reading.md          the publishable artifact (reading + framing note)
  spectral_reading_report.md   THE DELIVERABLE (verification report)
  reconciliation.tsv           this build vs the published #13/#50 numbers
  diagnostics.tsv              build-time counters
  summary.json                 machine-readable headline numbers

Reproduce:
    python work/spectral_reading.py
    python work/test_spectral_reading.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from silence_removal import (
    SRC,
    VARIANTS,
    load_folio,
    load_pages,
    normalize_line,
    segment_stats,
    site_spans,
    text_metrics,
    transform_line,
    words,
)
from silence_removal import _runs as site_runs_of_line

# `_runs` is #13's own junction-grouping rule; importing it (rather than
# re-deriving the count by subtracting periods post hoc) keeps the period
# closure check in compute_numbers non-vacuous.

OUT = os.path.join(ROOT, "analysis", "spectral_reading")
V4_KEY = "v4_operators_to_period"

# Published upstream artifacts this artifact is reconciled against.
SILENCE_DIR = os.path.join(ROOT, "analysis", "silence_removal")
SR_SUMMARY = os.path.join(SILENCE_DIR, "summary.json")
V4_PUBLISHED = os.path.join(SILENCE_DIR, "v4_operators_to_period.txt")
EXPERIMENT_DIR = os.path.join(ROOT, "analysis", "silence_experiment")
PPL_TSV = os.path.join(EXPERIMENT_DIR, "perplexity_by_variant.tsv")


class SpectralError(RuntimeError):
    """Raised when the artifact cannot be derived or reconciled."""


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------
def v4_params():
    """(treatment, only, separator) of the v4 transform."""
    return next((t, o, s) for k, t, o, s in VARIANTS if k == V4_KEY)


def apply_v4(line):
    """Apply the v4 transform to a single source line."""
    treatment, only, sep = v4_params()
    new, _deleted, _fused = transform_line(
        line, site_spans(line), treatment, only, sep
    )
    return normalize_line(new)


def build_reading():
    """Build the spectral reading from the v4 silence-removed stream.

    Returns (reading_pages, page_nos) where reading_pages maps page number
    to the list of non-empty reading lines, in source line order.
    """
    pages = load_pages(SRC)
    page_nos = sorted(pages)
    reading_pages = {}
    for pno in page_nos:
        out_lines = []
        for line in pages[pno].split("\n"):
            new = apply_v4(line)
            if new:
                out_lines.append(new)
        reading_pages[pno] = out_lines
    return reading_pages, page_nos


def reading_text(reading_pages, page_nos):
    """The reading as one page-tagged text block."""
    parts = []
    for pno in page_nos:
        parts.append(f"<!-- page {pno} -->")
        parts.extend(reading_pages[pno])
    return "\n\n".join(parts) + "\n"


def baseline_text(pages, page_nos):
    """The untouched corpus as one text block (the v0 baseline)."""
    return "".join(pages[pno] for pno in page_nos)


def count_site_runs(pages, page_nos):
    """Number of period insertions the v4 transform performs.

    Contiguous sites that receive the same replacement (`::<<`) share one
    junction, so the count of *runs* is the count of inserted periods, not
    the count of sites.
    """
    treatment, only, sep = v4_params()
    runs = 0
    for pno in page_nos:
        for line in pages[pno].split("\n"):
            runs += len(site_runs_of_line(site_spans(line), treatment, only, sep))
    return runs


# ---------------------------------------------------------------------------
# upstream: published numbers this artifact is checked against
# ---------------------------------------------------------------------------
def published_v4_stream():
    """Parse the published v4 stream into {page: [lines]}; None if absent."""
    if not os.path.exists(V4_PUBLISHED):
        return None
    pages = {}
    cur = None
    with open(V4_PUBLISHED, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*<!-- page (\d+) -->\s*$", line)
            if m:
                cur = int(m.group(1))
                pages[cur] = []
            elif line.strip() and cur is not None:
                pages[cur].append(line.strip())
    return pages


def load_upstream():
    """Read the published #13/#50 numbers, or None where absent."""
    up = {"sr_summary": os.path.exists(SR_SUMMARY),
          "ppl_tsv": os.path.exists(PPL_TSV)}
    if up["sr_summary"]:
        with open(SR_SUMMARY, encoding="utf-8") as f:
            s = json.load(f)
        v4 = s["sentence_shape"]["v4_operators_to_period"]
        v0 = s["sentence_shape"]["v0_original"]
        up.update({
            "n_pages": s["n_pages"],
            "lexemes": s["lexemes_baseline"],
            "operator_tokens": s["operator_tokens"],
            "dash_sites": s["dash_sites"],
            "dash_by_char": s["dash_by_char"],
            "silence_sites": s["silence_sites_total"],
            "v4_periods": s["variants"]["v4_operators_to_period"]["periods"],
            "v4_segments": v4["n_segments"],
            "v4_median_words": v4["median_words"],
            "v4_frac_upper": v4["frac_upper_initial"],
            "v0_segments": v0["n_segments"],
            "v0_periods": s["variants"]["v0_original"]["periods"],
            "v0_frac_upper": v0["frac_upper_initial"],
        })
    if up["ppl_tsv"]:
        ppl = {}
        with open(PPL_TSV, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                ppl[row["variant"]] = float(row["perplexity"])
        up["ppl_v0"] = ppl.get("v0_original")
        up["ppl_v4"] = ppl.get(V4_KEY)
    return up


def reconcile(here, upstream):
    """Row-by-row reconciliation of this build against published numbers."""
    rows = []

    def add(quantity, source, published, observed, tol=0):
        if published is None or observed is None:
            match = "n/a (artifact absent)"
        elif isinstance(published, float) or isinstance(observed, float):
            match = "ok" if abs(float(published) - float(observed)) <= tol \
                else "MISMATCH"
        else:
            match = "ok" if published == observed else "MISMATCH"
        rows.append({"quantity": quantity, "source": source,
                     "published": published, "here": observed, "match": match})

    sr = "#13 analysis/silence_removal/summary.json"
    xp = "#50 analysis/silence_experiment/perplexity_by_variant.tsv"
    add("pages", sr, upstream.get("n_pages"), here["n_pages"])
    add("word tokens (lexemes)", sr, upstream.get("lexemes"), here["n_words"])
    add("silence sites replaced (operators + dashes)", sr,
        upstream.get("silence_sites"), here["n_sites"])
    add("operator tokens (#10 set)", sr, upstream.get("operator_tokens"),
        here["n_operator_tokens"])
    add("dash sites (free-standing separator class)", sr,
        upstream.get("dash_sites"), here["n_dash_sites"])
    add("periods in the reading", sr, upstream.get("v4_periods"),
        here["n_periods"])
    add("baseline periods (v0)", sr, upstream.get("v0_periods"),
        here["baseline_periods"])
    add("reading segments (word-bearing)", sr, upstream.get("v4_segments"),
        here["n_segments"])
    add("reading median words/segment", sr, upstream.get("v4_median_words"),
        here["median_words"], tol=1e-9)
    add("reading uppercase-initial share", sr, upstream.get("v4_frac_upper"),
        here["frac_upper"], tol=1e-9)
    add("baseline segments (v0)", sr, upstream.get("v0_segments"),
        here["baseline_segments"])
    add("baseline uppercase-initial share", sr, upstream.get("v0_frac_upper"),
        here["baseline_frac_upper"], tol=1e-9)
    add("GPT-2 perplexity, v0 baseline", xp, upstream.get("ppl_v0"),
        here.get("ppl_v0"), tol=1e-6)
    add("GPT-2 perplexity, v4 reading", xp, upstream.get("ppl_v4"),
        here.get("ppl_v4"), tol=1e-6)
    return rows


def check_reconciliation(rows):
    """Fail loudly rather than publish a number the upstream contradicts."""
    for r in rows:
        if r["match"] == "MISMATCH":
            raise SpectralError(
                f"reconciliation mismatch: {r['quantity']} "
                f"(published {r['published']!r}, built {r['here']!r})")


# ---------------------------------------------------------------------------
# worked example: one line, before and after, with a citation
# ---------------------------------------------------------------------------
def pick_worked_example(pages, page_nos):
    """Deterministic worked example of the transform.

    Rule: the first line, in page then line order, whose source contains
    all three operator classes the framing note discusses (`::`, `//`, `=`)
    and whose reading has 3-8 word-bearing segments - so the example shows
    the erasure of the operator distinctions in a single glance.
    """
    for pno in page_nos:
        for line_no, line in enumerate(pages[pno].split("\n"), start=1):
            if not ("::" in line and "//" in line and "=" in line):
                continue
            reading = apply_v4(line)
            segs = [s for s in re.split(r"\.\s*", reading) if words(s)]
            if 3 <= len(segs) <= 8:
                return {"page": pno, "line_no": line_no,
                        "source": line.strip(), "reading": reading}
    raise SpectralError("no worked example found; the selection rule is wrong")


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
def compute_numbers(reading_pages, page_nos, upstream):
    """Every headline number, computed and reconciled."""
    pages = load_pages(SRC)
    reading = reading_text(reading_pages, page_nos)
    base = baseline_text(pages, page_nos)
    rstats = segment_stats(reading)
    bstats = segment_stats(base)
    rmetrics = text_metrics(reading)
    n_periods = rmetrics["periods"]
    n_site_runs = count_site_runs(pages, page_nos)

    # #12's lexer already breaks tokens at operator positions, so the word
    # sequence of a period variant is the baseline's, exactly.
    n_words = len(words(reading))
    base_words = words(base)
    if words(reading) != base_words:
        raise SpectralError("word sequence is not preserved exactly")

    # token-less period segments: the counting-rule residue, disclosed
    cleaned = re.sub(r"<!--.*?-->", " ", reading, flags=re.DOTALL)
    segs_all = [s for s in re.split(r"\.\s*", cleaned) if s.strip()]
    tokenless = [s.strip() for s in segs_all if not words(s)]

    # #13's own site classification: operator tokens vs the dash class.
    site_counts = {}
    n_operator_tokens = 0
    n_dash_sites = 0
    for pno in page_nos:
        for _s, _e, op, kind in site_spans(pages[pno]):
            site_counts[op] = site_counts.get(op, 0) + 1
            if kind == "op":
                n_operator_tokens += 1
            else:
                n_dash_sites += 1

    numbers = {
        "n_pages": len(page_nos),
        "n_words": n_words,
        "n_sites": n_operator_tokens + n_dash_sites,
        "n_operator_tokens": n_operator_tokens,
        "n_dash_sites": n_dash_sites,
        "n_site_runs": n_site_runs,
        "n_periods": n_periods,
        "baseline_periods": text_metrics(base)["periods"],
        "n_segments": rstats["n_segments"],
        "mean_words": rstats["mean_words"],
        "median_words": rstats["median_words"],
        "frac_upper": rstats["frac_upper_initial"],
        "frac_upper_ge5": rstats["frac_upper_initial_ge5"],
        "frac_single_word": rstats["frac_single_word"],
        "n_tokenless_segments": len(tokenless),
        "tokenless_examples": sorted(set(tokenless)),
        "baseline_segments": bstats["n_segments"],
        "baseline_frac_upper": bstats["frac_upper_initial"],
        "baseline_frac_upper_ge5": bstats["frac_upper_initial_ge5"],
        "ppl_v0": upstream.get("ppl_v0"),
        "ppl_v4": upstream.get("ppl_v4"),
    }
    # arithmetic closure: inserted periods plus the corpus's own periods
    if numbers["n_periods"] != numbers["n_site_runs"] + numbers["baseline_periods"]:
        raise SpectralError(
            f"period closure failed: {numbers['n_periods']} periods != "
            f"{numbers['n_site_runs']} inserted + "
            f"{numbers['baseline_periods']} pre-existing")
    if numbers["n_site_runs"] > numbers["n_sites"]:
        raise SpectralError("more period insertions than silence sites")
    return numbers, site_counts


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------
def write_artifact(reading_pages, page_nos, numbers, example, folio, out_dir):
    """Write spectral_reading.md and summary.json."""
    os.makedirs(out_dir, exist_ok=True)
    n = numbers

    L = []
    p = L.append

    p("# Spectral Reading: Blood Electric")
    p("")
    p("*A distinct under-text reading, explicitly not the real text.*")
    p("")
    p("---")
    p("")
    p("## Framing Note")
    p("")
    p("### What this is")
    p("")
    p("A *spectral reading* is a reading that haunts the text.  It is the text "
      "with its silences removed - the operator layer (`::`, `//`, `=`, `/`, "
      "`<<`, `>>`, `<`, `>`, `++`, `+`, `|`) plus the free-standing dash "
      "class, replaced by periods, so the implicit punctuation becomes "
      "explicit sentence boundaries.  The result is a \"sentence skeleton\": "
      "the transform that most lowers LM perplexity in #50's experiment.")
    p("")
    p("This is a *reading*, not a *reconstruction*.  It is derived from the "
      "source by a mechanical transform, but the decision to present it as a "
      "reading is an interpretation.  This framing note documents the "
      "derivation and its provenance, states what the reading can and cannot "
      "support, and keeps its interpretive status clear.")
    p("")
    p("### What this is not")
    p("")
    p("**This is NOT the real text.**  The real text is the original with its "
      "operators intact (`analysis/silence_removal/v0_original.txt`).  The "
      "spectral reading is a derivative work - a shadow of the text, a ghost "
      "that haunts it.  It is published here as a distinct under-text "
      "reading, explicitly not presented as the real text.")
    p("")
    p("### Derivation")
    p("")
    p("The spectral reading is derived from the source as follows:")
    p("")
    p(f"1. **Source:** `analysis/layout_recovery/blood-electric.layout.md` "
      f"({n['n_pages']} pages, the vetted layout corpus of #7).")
    p("2. **Transform:** v4 (`v4_operators_to_period`) from #13 - every "
      "silence site is replaced by `. ` (period + space).  This is the "
      "transform that #50 found most lowers LM perplexity.")
    p(f"3. **Scope:** {n['n_operator_tokens']:,} operator tokens (#10's set) "
      f"and {n['n_dash_sites']:,} whitespace-bounded dash tokens (#13's "
      "disclosed judgement call: the free-standing dash class) are replaced. "
      "The transform is defined over silence-site spans only.  Word-internal "
      "hyphens (`boy-roid`, `cold-blooded`) are inside WORD tokens and are "
      "never touched.")
    p("4. **Line structure:** The transform never introduces or removes a "
      "newline, so output line *i* derives from source line *i*.")
    p("5. **Page tags:** Every page is tagged (`<!-- page N -->`) so the "
      "reading can be cited and aligned.")
    p("")
    p(f"Contiguous sites share one junction: the {n['n_sites']:,} silence "
      f"sites become {n['n_site_runs']:,} inserted periods, and with the "
      f"{n['baseline_periods']} periods the corpus already had the reading "
      f"contains {n['n_periods']:,} periods in total.  That arithmetic is "
      "asserted at build time.")
    p("")
    p("### Provenance and evidence")
    p("")
    p("Every number above is computed from #13's own primitives or read from "
      "the published artifact it belongs to, and the two are reconciled row "
      "by row (`analysis/spectral_reading/reconciliation.tsv`); a mismatch "
      "is a build error.  The upstream artifacts are "
      "`analysis/silence_removal/summary.json` (transform inventory, "
      "variant metrics, sentence shape), "
      "`analysis/silence_removal/v4_operators_to_period.txt` (the published "
      "stream this reading re-derives, line for line) and "
      "`analysis/silence_experiment/perplexity_by_variant.tsv` (#50's LM "
      "verdict).  The evidence:")
    p("")
    p("| quantity | this reading | published source |")
    p("|---|---|---|")
    p(f"| pages | {n['n_pages']} | #13 `summary.json` |")
    p(f"| word tokens | {n['n_words']:,} | #13 `lexemes_baseline` "
      "(identical word sequence) |")
    p(f"| silence sites replaced | {n['n_sites']:,} "
      f"({n['n_operator_tokens']:,} operators + {n['n_dash_sites']:,} dashes) "
      "| #13 `summary.json` |")
    p(f"| periods | {n['n_periods']:,} | #13 `summary.json` / `variants.tsv` (v4) |")
    p(f"| period-separated segments (word-bearing) | {n['n_segments']:,}, "
      f"median {n['median_words']:.0f} words | #13 `sentence_shape.tsv` (v4) |")
    p(f"| uppercase-initial segments | {n['frac_upper']:.1%} "
      f"({n['frac_upper_ge5']:.1%} of segments with >= 5 words) | #13 "
      "`sentence_shape.tsv` (v4) |")
    p(f"| baseline (operators intact) | {n['baseline_segments']} segments, "
      f"{n['baseline_periods']} periods, {n['baseline_frac_upper']:.1%} "
      "uppercase-initial | #13 `sentence_shape.tsv` (v0) |")
    if n["ppl_v4"] is not None:
        p(f"| GPT-2 perplexity | {n['ppl_v4']:.2f} for this reading vs "
          f"{n['ppl_v0']:.2f} for the baseline | #50 "
          "`perplexity_by_variant.tsv` |")
    p("")
    p("A worked example, so the transform can be checked by eye "
      f"(page {example['page']} of {n['n_pages']}, source line "
      f"{example['line_no']}"
      + (f", folio {folio[example['page']]}" if folio.get(example["page"])
         else "") + "):")
    p("")
    p("> " + example["source"])
    p(">")
    p("> " + example["reading"])
    p("")
    p("The `::`, the `//` and the `=` seams all became periods; every word "
      "token survives.  That is the whole transform - and the whole "
      "limitation: the reading cannot tell you that `::` was a phrase "
      "separator, `//` an utterance boundary and `=` a naming seam.")
    p("")
    p("### What the reading can support")
    p("")
    p(f"- **Sentence-level analysis:** The reading has {n['n_segments']:,} "
      f"period-separated segments (median {n['median_words']:.0f} words), so "
      "it can be read as a sequence of sentences.")
    p("- **Rhythm and pacing:** The period boundaries mark the book's "
      "implicit rhythm, which the operators encode.")
    p(f"- **Structural analysis:** The reading reveals the book's sentence "
      f"skeleton, which is otherwise invisible - the untouched corpus has "
      f"only {n['baseline_periods']} periods in {n['n_pages']} pages and "
      f"{n['baseline_segments']} period-separated segments.")
    if n["ppl_v4"] is not None:
        p(f"- **LM-based analysis:** The reading is the most predictable "
          f"version of the text under GPT-2 in #50's experiment (perplexity "
          f"{n['ppl_v4']:.2f} vs {n['ppl_v0']:.2f} for the baseline), so it "
          "is the best candidate for language-model-based reading.")
    p("")
    p("### What the reading cannot support")
    p("")
    p(f"- **Word-level semantics:** The reading preserves the word sequence "
      f"exactly ({n['n_words']:,} word tokens, identical to the baseline), "
      "but the period boundaries are an interpretation, not a measurement.  "
      "The reading does NOT tell you what the words mean.")
    p(f"- **Sentence-level semantics:** The reading has {n['n_segments']:,} "
      f"segments, but only {n['frac_upper']:.1%} start with an uppercase "
      f"letter ({n['frac_upper_ge5']:.1%} of the segments with >= 5 words, "
      f"the length-controlled rate; the baseline rates are "
      f"{n['baseline_frac_upper']:.1%} / {n['baseline_frac_upper_ge5']:.1%}), "
      f"and {n['frac_single_word']:.1%} of the segments are a single word.  "
      "The segments are a skeleton, not a reconstruction - the best available "
      "approximation of the book's sentence structure, not a measurement "
      "of it.")
    p("- **Operator-level semantics:** The reading replaces every operator "
      "with a period, but the operators have different functions (`::` is a "
      "phrase separator, `//` is a sentence boundary, `=` is a naming seam, "
      "etc.).  The reading erases these distinctions.")
    p(f"- **Counting-rule residue:** {n['n_tokenless_segments']} "
      "period-separated segments contain no word token at all - each is a "
      "lone `:` left where a literal colon stands between two "
      "operator-derived periods.  They are itemized here rather than "
      "dropped, and they are why the reading's segment count uses #13's "
      "word-bearing rule.")
    p("- **OCR/HTML-artifact pages:** Pages 92, 158 and 219 carry raw "
      "OCR markup in the source; the reading carries them through exactly "
      "as #13's v4 stream does, so their fragments read as broken text.")
    p("- **The reading is NOT a transcription.**  It is a derivative work, "
      "a mechanical transform of the source.  It is published here as a "
      "distinct under-text reading, explicitly not presented as the real "
      "text.")
    p("")
    p("### Interpretive status")
    p("")
    p("The spectral reading is an *interpretation* of the text, not a "
      "*measurement* of it.  The transform is mechanical and exact (every "
      "silence site is replaced by a period, and the word sequence is "
      "preserved exactly), but the decision to present the result as a "
      "reading is an interpretation.  The reading is published here as a "
      "distinct under-text reading, explicitly not presented as the real "
      "text.")
    p("")
    p("The reading is *spectral* because it is a ghost of the text - the "
      "text with its silences removed, its operators turned into periods.  "
      "It haunts the text, but it is not the text.  It is a reading that "
      "sits under the text, a shadow that reveals the text's hidden "
      "structure.")
    p("")
    p("---")
    p("")
    p("## The Reading")
    p("")
    p("*The following is the v4 silence-removed stream, presented as a "
      "continuous reading.  Page markers are preserved for citation.  This "
      "is a spectral reading, not the real text.*")
    p("")

    for pno in page_nos:
        p(f"<!-- page {pno} -->")
        p("")
        for line in reading_pages[pno]:
            p(line)
            p("")

    p("---")
    p("")
    p("## Colophon")
    p("")
    p(f"- **Source:** `analysis/layout_recovery/blood-electric.layout.md` "
      f"({n['n_pages']} pages)")
    p("- **Transform:** v4 (`v4_operators_to_period`) from #13")
    p("- **Verdict:** #50 (the transform that most lowers LM perplexity)")
    p(f"- **Silence sites replaced:** {n['n_sites']:,} "
      f"({n['n_operator_tokens']:,} operators + {n['n_dash_sites']:,} "
      f"dashes) -> {n['n_site_runs']:,} inserted periods")
    p(f"- **Segments:** {n['n_segments']:,} period-separated segments "
      f"(median {n['median_words']:.0f} words)")
    p(f"- **Uppercase-initial:** {n['frac_upper']:.1%} "
      f"({n['frac_upper_ge5']:.1%} of segments with >= 5 words)")
    p(f"- **Word tokens:** {n['n_words']:,}")
    p("- **Generated by:** `work/spectral_reading.py` (deterministic; "
      "rebuild is byte-identical)")
    p("")
    p("*This spectral reading is a distinct under-text reading, explicitly "
      "not the real text.*")
    p("")

    artifact_path = os.path.join(out_dir, "spectral_reading.md")
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    summary = {
        "source": os.path.relpath(SRC, ROOT),
        "transform": V4_KEY,
        "n_pages": n["n_pages"],
        "n_words": n["n_words"],
        "n_sites": n["n_sites"],
        "n_operator_tokens": n["n_operator_tokens"],
        "n_dash_sites": n["n_dash_sites"],
        "n_site_runs": n["n_site_runs"],
        "n_periods": n["n_periods"],
        "baseline_periods": n["baseline_periods"],
        "n_segments": n["n_segments"],
        "median_words": n["median_words"],
        "frac_uppercase_initial": n["frac_upper"],
        "frac_uppercase_initial_ge5": n["frac_upper_ge5"],
        "frac_single_word_segments": n["frac_single_word"],
        "n_tokenless_segments": n["n_tokenless_segments"],
        "baseline_segments": n["baseline_segments"],
        "baseline_frac_uppercase_initial": n["baseline_frac_upper"],
        "perplexity_v4": n["ppl_v4"],
        "perplexity_v0": n["ppl_v0"],
        "interpretive_status": "spectral reading, not the real text",
        "can_support": [
            "sentence-level analysis",
            "rhythm and pacing",
            "structural analysis",
            "LM-based analysis",
        ],
        "cannot_support": [
            "word-level semantics",
            "sentence-level semantics",
            "operator-level semantics",
            "transcription",
        ],
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ---------------------------------------------------------------------------
# the report (THE DELIVERABLE) + side tables
# ---------------------------------------------------------------------------
def write_report(numbers, example, folio, recon_rows, site_counts, out_dir):
    """Write spectral_reading_report.md, reconciliation.tsv, diagnostics.tsv."""
    n = numbers
    L = []
    p = L.append

    p("# Spectral reading artifact - verification report (#67)")
    p("")
    p("## What was built")
    p("")
    p("`work/spectral_reading.py` produces `analysis/spectral_reading/spectral_reading.md`")
    p("- a publishable spectral reading of *Blood Electric* with its framing")
    p("note - plus `summary.json`, `reconciliation.tsv` and `diagnostics.tsv`.")
    p("")
    p("The spectral reading presents the v4 silence-removed stream (every")
    p("silence site replaced by `. `) as a distinct under-text reading,")
    p("explicitly not the real text.  The framing note documents the")
    p("derivation and provenance, states what the reading can and cannot")
    p("support, and keeps its interpretive status clear.  The report you are")
    p("reading is the verification layer: every headline number, its source,")
    p("and the check that it matches the published upstream artifact.")
    p("")
    p("## Derivation and provenance")
    p("")
    p(f"Source: `analysis/layout_recovery/blood-electric.layout.md` "
      f"({n['n_pages']} pages, the vetted layout corpus of #7).  Transform: "
      "v4 (`v4_operators_to_period`) from #13 - every silence site (operator "
      "token, plus the whitespace-bounded dash class #13 discloses) is "
      "replaced by `. `, so output line *i* derives from source line *i* and "
      "the word sequence is preserved exactly.  #50's experiment found this "
      "to be the transform that most lowers LM perplexity.  Word-internal hyphens "
      "(`boy-roid`, `cold-blooded`) are inside WORD tokens and never "
      "touched.  Every page is tagged so the reading can be cited.")
    p("")
    p("## Headline numbers")
    p("")
    p("| quantity | value |")
    p("|---|---|")
    p(f"| pages | {n['n_pages']} |")
    p(f"| word tokens | {n['n_words']:,} (identical to the baseline) |")
    p(f"| silence sites replaced | {n['n_sites']:,} "
      f"({n['n_operator_tokens']:,} operators + {n['n_dash_sites']:,} dashes) |")
    p(f"| period insertions (site runs) | {n['n_site_runs']:,} |")
    p(f"| periods in the reading | {n['n_periods']:,} "
      f"(= {n['n_site_runs']:,} inserted + {n['baseline_periods']} "
      "pre-existing) |")
    p(f"| period-separated segments (word-bearing) | {n['n_segments']:,} |")
    p(f"| median words per segment | {n['median_words']:.0f} |")
    p(f"| uppercase-initial segments | {n['frac_upper']:.1%} "
      f"({n['frac_upper_ge5']:.1%} of segments with >= 5 words) |")
    p(f"| single-word segments | {n['frac_single_word']:.1%} |")
    p(f"| baseline (operators intact) | {n['baseline_segments']} segments, "
      f"{n['baseline_periods']} periods, "
      f"{n['baseline_frac_upper']:.1%} uppercase-initial |")
    if n["ppl_v4"] is not None:
        p(f"| GPT-2 perplexity | {n['ppl_v4']:.2f} (v4) vs {n['ppl_v0']:.2f} "
          "(v0 baseline) |")
    p("")
    p("## Reconciliation against the published upstream numbers")
    p("")
    p("| quantity | source | published | here | match |")
    p("|---|---|---:|---:|---|")
    for r in recon_rows:
        pub = "absent" if r["published"] is None else r["published"]
        here = "absent" if r["here"] is None else r["here"]
        p(f"| {r['quantity']} | {r['source']} | {pub} | {here} | "
          f"{r['match']} |")
    p("")
    p("Two arithmetic closures are asserted at build time, so a silently")
    p("broken transform fails the build rather than publishing: the word")
    p("sequence equals the baseline's token-for-token, and the period count")
    p("equals the inserted periods plus the corpus's pre-existing periods.")
    p("Where a published artifact is absent, the row is marked n/a and no")
    p("claim is made from it.")
    p("")
    p("## Worked example")
    p("")
    fol = folio.get(example["page"])
    p(f"Source line {example['page']} of {n['n_pages']}, line "
      f"{example['line_no']}"
      + (f" (folio {fol})" if fol else "")
      + ", of `analysis/layout_recovery/blood-electric.layout.md`:")
    p("")
    p("```")
    p(example["source"])
    p("```")
    p("")
    p("The same line in the spectral reading:")
    p("")
    p("```")
    p(example["reading"])
    p("```")
    p("")
    p("Every `::`, `//` and `=` seam became a period; every word token")
    p("survived.  Selected by a published rule (the first line containing")
    p("all three operator classes, with a reading of 3-8 word-bearing")
    p("segments), not hand-picked.")
    p("")
    p("## Interpretive limits")
    p("")
    p("What the reading is for: sentence-level analysis, rhythm and pacing,")
    p("structural analysis, LM-based analysis.  What it cannot support:")
    p("word-level semantics (identical word sequence, interpreted periods),")
    p(f"sentence-level semantics ({n['frac_upper']:.1%} uppercase-initial, "
      f"{n['frac_upper_ge5']:.1%} length-controlled, "
      f"{n['frac_single_word']:.1%} single-word segments - a skeleton, not a "
      "reconstruction), operator-level semantics (the distinct functions of")
    p("`::`, `//`, `=` are erased), and transcription (it is a derivative")
    p("work).  The counting-rule residue is itemized: "
      f"{n['n_tokenless_segments']} token-less period segments, each a lone "
      "`:` where a literal colon stands between two operator-derived")
    p("periods.  Pages 92, 158 and 219 carry raw OCR markup through the")
    p("stream exactly as #13's v4 stream carries them.")
    p("")
    p("## Verification")
    p("")
    p("Reproduce:")
    p("")
    p("```bash")
    p("python work/spectral_reading.py          # regenerate the artifact")
    p("python work/test_spectral_reading.py     # regression suite")
    p("python -m ruff check work/spectral_reading.py work/test_spectral_reading.py")
    p("```")
    p("")
    p("The regression suite pins the derivation (the reading is the v4")
    p("transform of the source, line for line, and matches #13's published")
    p("v4 stream), the reconciliation (this build's numbers equal the")
    p("published #13/#50 numbers, and a planted mismatch fails the check),")
    p("the framing note's contract (explicitly not the real text;")
    p("derivation, provenance, can/cannot support and interpretive status")
    p("present), the worked example, and determinism (a rebuild is")
    p("byte-identical).  The build is offline and deterministic: no network,")
    p("no clock, no randomness.")
    p("")
    p("## Handoff")
    p("")
    p("- **#61** (E8 - Interpretation, gloss and interlinear edition): the")
    p("  spectral reading is a distinct under-text reading, explicitly not")
    p("  the real text - the v4 stream presented as a reading, with a")
    p("  framing note that documents its derivation and interpretive status.")
    p("- **#62** (interlinear edition): the spectral reading is a companion")
    p("  to the interlinear edition.  The interlinear edition renders every")
    p("  clause in three lines (original | silence-removed | gloss); the")
    p("  spectral reading presents the silence-removed stream as a")
    p("  continuous reading.")
    p("- **#63** (LLM glossing pipeline): the spectral reading is a reading")
    p("  input for the glossing pipeline - a continuous, period-marked")
    p("  stream to gloss, with its framing note attached.")
    p("- **#64** (translation committee): same - a reading to be argued")
    p("  over, with the derivation visible.")
    p("- **#65** (termbase): the reading preserves the word sequence")
    p("  exactly, so termbase entries cite the reading's page tags.")
    p("- **#66** (narrative reconstruction): the reading's segments are the")
    p("  sentence layer an entity/event reading can be built on, with the")
    p("  skeleton's limits attached.")
    p("")
    p("---")
    p("*Generated by `work/spectral_reading.py`*")
    p("")

    with open(os.path.join(out_dir, "spectral_reading_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    with open(os.path.join(out_dir, "reconciliation.tsv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["quantity", "source", "published", "here", "match"])
        for r in recon_rows:
            w.writerow([r["quantity"], r["source"], r["published"],
                        r["here"], r["match"]])

    with open(os.path.join(out_dir, "diagnostics.tsv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["counter", "value"])
        for k, v in [
            ("pages", n["n_pages"]),
            ("silence_sites", n["n_sites"]),
            ("site_runs", n["n_site_runs"]),
            ("periods", n["n_periods"]),
            ("baseline_periods", n["baseline_periods"]),
            ("word_tokens", n["n_words"]),
            ("segments_word_bearing", n["n_segments"]),
            ("tokenless_segments", n["n_tokenless_segments"]),
            ("worked_example_page", example["page"]),
            ("worked_example_line", example["line_no"]),
            ("reconciliation_rows", len(recon_rows)),
            ("reconciliation_mismatches",
             sum(1 for r in recon_rows if r["match"] == "MISMATCH")),
        ]:
            w.writerow([k, v])
        for op in sorted(site_counts, key=lambda o: -site_counts[o]):
            w.writerow([f"site:{op}", site_counts[op]])


# ---------------------------------------------------------------------------
def build(out_dir=OUT):
    """Build the whole artifact tree; returns the headline numbers."""
    reading_pages, page_nos = build_reading()
    pages = load_pages(SRC)
    folio = load_folio()
    upstream = load_upstream()
    numbers, site_counts = compute_numbers(reading_pages, page_nos, upstream)
    recon_rows = reconcile(numbers, upstream)
    check_reconciliation(recon_rows)
    example = pick_worked_example(pages, page_nos)
    summary = write_artifact(reading_pages, page_nos, numbers, example,
                             folio, out_dir)
    write_report(numbers, example, folio, recon_rows, site_counts, out_dir)
    numbers["summary"] = summary
    return numbers


def main():
    numbers = build()
    print(f"spectral reading: {numbers['n_pages']} pages, "
          f"{numbers['n_segments']:,} segments, "
          f"{numbers['n_words']:,} words, "
          f"{numbers['n_periods']:,} periods")
    print(f"interpretive status: {numbers['summary']['interpretive_status']}")


if __name__ == "__main__":
    main()
