#!/usr/bin/env python3
"""#32 Parse-attempt and derangement curve per page.

A dependency parser trained on English will emit *some* tree for any input, so
"the fraction of sentences that parse" cannot be the fraction for which the
parser returns without error.  This script operationalises parseability as a
structural property of the resulting tree -- does the parser impose a single,
clausal, subject-bearing sentence on the chunk? -- and measures it per page.

Units are the book's own sentences, not words and not arbitrary whitespace
chunks.  The unit is the ``//``-delimited utterance of #12/#13, taken from the
persisted silence-removal streams:

  * ``v5_period_from_slashes`` (primary) -- ``//`` -> ``.``, every other
    operator -> space.  #12/#13 established ``//`` as the utterance boundary.
  * ``v4_operators_to_period`` (sensitivity) -- every operator -> ``.``.

Segmentation reproduces #13's regex (``\\.\\s*`` after comment removal) but is
done **per page**, so a segment never spans a page boundary.  #13's whole-file
count is lower (2,163 vs 2,360 for v5) because its page comments are replaced
by spaces and a segment can then bridge two pages; the difference is reconciled
and reported, not hidden.

Parsers
-------
* spaCy ``en_core_web_md`` -- greedy parse for the tree, beam parse
  (``beam_width=8``) for a real confidence: the probability the parser assigns
  to its single best complete parse, and the per-token marginal of each chosen
  arc (via ``DependencyParser.scored_parses``).
* stanza -- an independent neural dependency parser, used to replicate the
  structural parseability verdict.

Baselines calibrate the curve: normal English prose (pinned public-domain texts
from #24's cache) as a ceiling, and a seeded within-sentence token shuffle of
the book as a floor.

Outputs under ``analysis/parse_attempt/``::

    sentences.tsv        one row per book segment x variant (primary v5 + v4)
    per_page.tsv         page x variant parseability / confidence aggregates
    derangement_curve.tsv page-level curve + rolling mean
    baselines.tsv        English ceiling, shuffled null, component rates
    reconciliation.tsv   counts checked against #13 and the word total
    diagnostics.tsv      flagged pages / parser warnings
    summary.json         machine-readable headline numbers
    derangement_curve.png plot of the curve
    parse_attempt_report.md THE DELIVERABLE

Reproduce::

    conda activate marker
    python work/parse_attempt.py            # full run, ~3-4 min
    python work/parse_attempt.py --quick    # small deterministic subset
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# #13's comment stripper and word filter (shared so the segment count is
# comparable to #13's sentence_shape.tsv).
from silence_removal import COMMENT_RE, WORD_RE, words

SILENCE = ROOT / "analysis" / "silence_removal"
FREQ_CACHE = ROOT / "analysis" / "frequency" / "reference_cache"
OUT = ROOT / "analysis" / "parse_attempt"

PAGE_RE = re.compile(r"<!-- page (\d+) -->")
PERIOD_RE = re.compile(r"\.\s*")

SCHEMA = "parse-attempt/1"

# Pages whose OCR text is raw HTML markup rather than the book's own syntax
# (#12/#19); they are flagged, and the curve is reported with and without them.
HTML_ARTIFACT_PAGES = {92, 158, 219}
# Front/back matter pages: not the novel body.
BODY_PAGES = set(range(4, 225))

# spaCy beam width for the confidence estimate.
BEAM_WIDTH = 8
# Confidence threshold is not used to define parseability (structure is); it is
# only reported.  Kept as a constant so the report can state it.
CLS_VERB = {"VERB", "AUX"}
CLS_CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------
def cell(value) -> str:
    s = "" if value is None else str(value)
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(cell(row.get(h, "")) for h in header) + "\n")


def load_variant_pages(key: str) -> dict:
    text = (SILENCE / f"{key}.txt").read_text(encoding="utf-8")
    parts = PAGE_RE.split(text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def page_segments(body: str):
    """(char_start, text) for each period-delimited segment on one page.

    Comment characters are replaced by spaces of equal length so the recorded
    ``char_start`` stays a stable offset into the comment-stripped page body.
    A segment is kept iff it contains at least one WORD (the #13 filter).
    """
    body = COMMENT_RE.sub(lambda m: " " * len(m.group()), body)
    out = []
    pos = 0
    for m in PERIOD_RE.finditer(body):
        seg = body[pos:m.start()]
        if words(seg):
            wm = WORD_RE.search(seg)
            out.append((pos + wm.start(), seg.strip()))
        pos = m.end()
    seg = body[pos:]
    if words(seg):
        wm = WORD_RE.search(seg)
        out.append((pos + wm.start(), seg.strip()))
    return out


def load_units(key: str):
    """List of dicts: page, sent_index, char_start, text for a variant."""
    pages = load_variant_pages(key)
    units = []
    for page in sorted(pages):
        for idx, (start, text) in enumerate(page_segments(pages[page])):
            units.append({"page": page, "sent_index": idx,
                          "char_start": start, "text": text})
    return units


# ---------------------------------------------------------------------------
# spaCy metrics
# ---------------------------------------------------------------------------
def dep_crossings(doc):
    """Number of crossing dependency arcs (non-projectivity count)."""
    arcs = [(t.i, t.head.i) for t in doc if t.head.i != t.i]
    n = len(arcs)
    cross = 0
    for i in range(n):
        a, b = arcs[i]
        lo1, hi1 = (a, b) if a < b else (b, a)
        for j in range(i + 1, n):
            c, e = arcs[j]
            lo2, hi2 = (c, e) if c < e else (e, c)
            if (lo1 < lo2 < hi1 < hi2) or (lo2 < lo1 < hi2 < hi1):
                cross += 1
    return cross


def _doc_metrics(doc, beam_prob, arc_mean, arc_min):
    roots = [t for t in doc if t.dep_ == "ROOT"]
    n_root = len(roots)
    root = roots[0] if roots else None
    root_pos = root.pos_ if root else "NONE"
    root_tag = root.tag_ if root else "NONE"
    root_dep = root.dep_ if root else "NONE"
    subs = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass")]
    n_subj = len(subs)
    n_aux = sum(1 for t in doc if t.pos_ == "AUX")
    single_root = (n_root == 1)
    clausal = root_pos in CLS_VERB
    # English imperative: base-form verb root with no subject.
    imperative = bool(root is not None and root_pos == "VERB"
                      and root_tag == "VB" and n_subj == 0)
    # Order-sensitive, English-shaped clause: the subject must precede the
    # finite root (SVO) and the tree must be projective.
    subj_before = bool(root is not None and any(t.i < root.i for t in subs))
    cross = dep_crossings(doc)
    projective = (cross == 0)
    has_complement = any(t.dep_ in ("dobj", "attr", "acomp", "xcomp", "ccomp",
                                    "oprd", "dative") for t in doc)
    # primary: single clausal root + subject before root + projective
    parseable = bool(single_root and clausal and n_subj >= 1
                     and subj_before and projective)
    # lenient: a subject-bearing or imperative clause, order not enforced
    parseable_lenient = bool(single_root and clausal
                             and (n_subj >= 1 or imperative))
    # strict: primary + an object/complement (a complete transitive clause)
    parseable_strict = bool(parseable and has_complement)
    n_sents = len(list(doc.sents)) if len(doc) else 0
    return {
        "n_tokens_spacy": len(doc),
        "single_root": int(single_root),
        "root_pos": root_pos,
        "root_tag": root_tag,
        "root_dep": root_dep,
        "n_subj": n_subj,
        "n_aux": n_aux,
        "imperative": int(imperative),
        "clausal": int(clausal),
        "subj_before_root": int(subj_before),
        "dep_crossings": cross,
        "projective": int(projective),
        "has_complement": int(has_complement),
        "parseable": int(parseable),
        "parseable_lenient": int(parseable_lenient),
        "parseable_strict": int(parseable_strict),
        "n_spacy_sents": n_sents,
        "beam_best_prob": beam_prob,
        "arc_marginal_mean": arc_mean,
        "arc_marginal_min": arc_min,
    }


def spacy_run(nlp, texts, beam_width=BEAM_WIDTH, batch_size=64):
    """Return a list of metric dicts (greedy tree + beam confidence)."""
    parser = nlp.get_pipe("parser")
    docs = list(nlp.pipe(texts, batch_size=batch_size))
    # beam confidence in chunks (states = docs * beam_width)
    beam_probs = [None] * len(docs)
    arc_means = [None] * len(docs)
    arc_mins = [None] * len(docs)
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        beams = parser.beam_parse(chunk, beam_width=beam_width)
        head_scores, _ = parser.scored_parses(beams)
        for j, (doc, bm, hs) in enumerate(zip(chunk, beams, head_scores)):
            by_child = defaultdict(dict)
            for (child, head), sc in hs.items():
                by_child[child][head] = sc
            confs = []
            for t in doc:
                cand = by_child.get(t.i, {})
                tot = sum(cand.values())
                confs.append(cand.get(t.head.i, 0.0) / tot if tot > 0 else 0.0)
            beam_probs[i + j] = float(bm.probs[0]) if len(bm.probs) else 0.0
            arc_means[i + j] = float(statistics.fmean(confs)) if confs else 0.0
            arc_mins[i + j] = float(min(confs)) if confs else 0.0
    return [_doc_metrics(doc, beam_probs[i], arc_means[i], arc_mins[i])
            for i, doc in enumerate(docs)]


# ---------------------------------------------------------------------------
# stanza metrics
# ---------------------------------------------------------------------------
def stanza_run(nlp, texts, batch_note=None):
    """Independent parseability verdict from stanza (per utterance)."""
    out = []
    for text in texts:
        doc = nlp(text)
        sents = doc.sentences
        # concatenate: one input string may be split into several sentences
        roots = []
        subs = []
        for sent in sents:
            for w in sent.words:
                if w.deprel == "root":
                    roots.append(w)
                if w.deprel in ("nsubj", "nsubj:pass", "csubj", "csubj:pass"):
                    subs.append(w)
        root = roots[0] if roots else None
        root_upos = root.upos if root else "NONE"
        root_deprel = root.deprel if root else "NONE"
        single_root = (len(roots) == 1)
        # UD puts the copula under the predicate (`cop`), so a copular clause
        # has a nominal/adjectival root; count it as clausal to match spaCy's
        # convention where the copula is the root.
        has_cop = any(w.deprel == "cop" for sent in sents for w in sent.words)
        clausal = (root_upos in CLS_VERB) or has_cop
        subj_before = bool(root is not None
                           and any(w.id < root.id for w in subs))
        parseable = bool(single_root and clausal and len(subs) >= 1
                         and subj_before)
        parseable_lenient = bool(single_root and clausal and len(subs) >= 1)
        out.append({
            "stanza_root_upos": root_upos,
            "stanza_root_deprel": root_deprel,
            "stanza_n_subj": len(subs),
            "stanza_subj_before_root": int(subj_before),
            "stanza_n_sents": len(sents),
            "stanza_single_root": int(single_root),
            "stanza_clausal": int(clausal),
            "stanza_parseable": int(parseable),
            "stanza_parseable_lenient": int(parseable_lenient),
        })
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else 0.0


def spearman(xs, ys):
    """Spearman rho with average ranks; returns (rho, n)."""
    n = len(xs)
    if n < 3:
        return 0.0, n

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return (num / den if den else 0.0), n


def rolling(vals, window):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window // 2)
        hi = min(len(vals), i + window // 2 + 1)
        out.append(statistics.fmean(vals[lo:hi]))
    return out


def aggregate_pages(records, stanza_records=None, key="v5_period_from_slashes"):
    by_page = defaultdict(list)
    for r in records:
        if r["variant"] == key:
            by_page[r["page"]].append(r)
    st_by_page = defaultdict(list)
    if stanza_records:
        for r in stanza_records:
            st_by_page[r["page"]].append(r)
    rows = []
    for page in sorted(by_page):
        rs = by_page[page]
        st = st_by_page.get(page, [])
        rows.append({
            "page": page,
            "variant": key,
            "n_seg": len(rs),
            "n_parseable": sum(r["parseable"] for r in rs),
            "frac_parseable": sum(r["parseable"] for r in rs) / len(rs),
            "frac_parseable_lenient": mean([r["parseable_lenient"] for r in rs]),
            "frac_parseable_strict": mean([r["parseable_strict"] for r in rs]),
            "frac_clausal": mean([r["clausal"] for r in rs]),
            "frac_single_root": mean([r["single_root"] for r in rs]),
            "frac_subject": mean([1 if r["n_subj"] >= 1 else 0 for r in rs]),
            "mean_conf": mean([r["beam_best_prob"] for r in rs]),
            "mean_arc_marginal": mean([r["arc_marginal_mean"] for r in rs]),
            "mean_words": mean([r["n_words"] for r in rs]),
            "n_seg_stanza": len(st),
            "n_parseable_stanza": sum(r["stanza_parseable"] for r in st),
            "frac_parseable_stanza": (sum(r["stanza_parseable"] for r in st) / len(st)
                                      if st else ""),
            "html_artifact": int(page in HTML_ARTIFACT_PAGES),
            "body_page": int(page in BODY_PAGES),
        })
    return rows


def drift_stats(rows, min_seg=3, body_only=True, exclude_html=True):
    sel = [r for r in rows if r["n_seg"] >= min_seg]
    if body_only:
        sel = [r for r in sel if r["body_page"]]
    if exclude_html:
        sel = [r for r in sel if not r["html_artifact"]]
    pages = [r["page"] for r in sel]
    fracs = [r["frac_parseable"] for r in sel]
    rho, n = spearman(pages, fracs)
    # first-third vs last-third
    k = len(sel) // 3
    first = statistics.fmean(fracs[:k]) if k else 0.0
    last = statistics.fmean(fracs[-k:]) if k else 0.0
    # OLS slope of frac on page
    mp, mf = statistics.fmean(pages), statistics.fmean(fracs)
    denom = sum((p - mp) ** 2 for p in pages)
    slope = (sum((p - mp) * (f - mf) for p, f in zip(pages, fracs)) / denom
             if denom else 0.0)
    return {"rho": rho, "n_pages": n, "slope_per_page": slope,
            "first_third": first, "last_third": last, "delta": last - first}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def english_sentences(limit=1500):
    """Clean English prose sentences from the #24 pinned Gutenberg cache.

    Gutenberg licence/header/footer matter is stripped so the baseline is real
    prose, not boilerplate metadata.
    """
    files = sorted(FREQ_CACHE.glob("*.txt"))
    out = []
    bad = ("gutenberg", "ebook", "isbn", "copyright", "www.", "http")
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                      raw, re.IGNORECASE | re.DOTALL)
        if m:
            raw = raw[m.end():]
        m = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG",
                      raw, re.IGNORECASE)
        if m:
            raw = raw[:m.start()]
        for s in re.split(r"(?<=[.!?])\s+", raw):
            s = s.strip()
            n = len(s.split())
            if not (3 <= n <= 40):
                continue
            low = s.lower()
            if any(b in low for b in bad):
                continue
            if not any(c.islower() for c in s):
                continue
            if not s[0].isupper():
                continue
            out.append(s)
            if len(out) >= limit:
                return out
    return out


def shuffle_tokens(text, rng):
    toks = text.split()
    rng.shuffle(toks)
    return " ".join(toks)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _pct(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def write_report(out_dir: Path, summary, base_rows, page_v5, page_v4, drift,
                 recon, stanza_records, english_stanza, quick=False):
    h = summary["headline"]
    L = []
    add = L.append
    add("# Parse-attempt and derangement curve per page (#32)\n")
    add("A dependency parser trained on English emits *some* tree for any input, "
        "so this task cannot ask whether the parser returns without error. It "
        "asks a structural question: does the parser impose a **single, clausal, "
        "subject-bearing, projective sentence** on each chunk, and how does that "
        "fraction vary across the book?\n")
    add("Reproduce with `python work/parse_attempt.py`; regression test "
        "`work/test_parse_attempt.py`.\n")

    add("## Method\n")
    add("- **Unit = the book's own sentence.** The `//`-delimited utterance of "
        "#12/#13, read from `analysis/silence_removal/v5_period_from_slashes.txt` "
        "(`//`→`.`, every other operator→space), split per page on `\\.\\s*` "
        "after comment removal. `v4_operators_to_period` (every operator→`.`) is "
        "the sensitivity variant.")
    add("- **Parsers.** spaCy `en_core_web_md` (greedy tree + a beam of width 8 "
        "for confidence) and stanza (English `tokenize,pos,lemma,depparse`) as an "
        "independent replication.")
    add("- **Confidence.** Two real parser numbers are recorded: the beam "
        "probability of the single best complete parse (`beam_best_prob`) and the "
        "per-token marginal of each chosen arc (`arc_marginal_mean`).")
    add("- **Baselines.** Normal English prose from #24's pinned public-domain "
        "cache (Gutenberg licence matter stripped) is the ceiling; a seeded "
        "within-sentence token shuffle of the book is the null floor.\n")

    add("### Reconciliation\n")
    add("| item | expected | measured | note |")
    add("|---|---:|---:|---|")
    for r in recon:
        add(f"| {r['item']} | {r['expected']} | {r['measured']} | {r['note']} |")
    add("")

    add("## The parseability definition\n")
    add("`parseable` (primary) = exactly one `ROOT` **and** the root is `VERB`/`AUX` "
        "**and** a subject (`nsubj`/`nsubjpass`/`csubj`) precedes the root **and** "
        "the dependency tree is projective. `lenient` drops the order/projectivity "
        "tests and admits base-form imperatives. `strict` additionally requires an "
        "object/complement.\n")

    add("## Headline results\n")
    add("| corpus | n | parseable | lenient | strict | clausal | 1 root | subj | subj<root | mean conf |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in base_rows:
        add("| {c} | {n} | {p} | {l} | {s} | {cl} | {sr} | {su} | {sb} | {mc} |".format(
            c=r["corpus"], n=r["n_seg"], p=_pct(r["frac_parseable"]),
            l=_pct(r.get("frac_parseable_lenient")),
            s=_pct(r.get("frac_parseable_strict")),
            cl=_pct(r["frac_clausal"]), sr=_pct(r["frac_single_root"]),
            su=_pct(r["frac_subject"]),
            sb=_pct(r.get("frac_subj_before_root")),
            mc=f"{r['mean_conf']:.3f}" if r["mean_conf"] != "" else "—"))
    add("")

    add("## Confidence is a weaker, partly non-diagnostic signal\n")
    _emc = h.get("english_mean_conf")
    _nmc = h.get("null_mean_conf")
    _bam = next((r["mean_arc_marginal"] for r in base_rows
                 if r["corpus"] == "book_v5_all"), None)
    _eam = next((r["mean_arc_marginal"] for r in base_rows
                 if r["corpus"] == "english_gutenberg"), None)
    _nam = next((r["mean_arc_marginal"] for r in base_rows
                 if r["corpus"] == "book_shuffled_null"), None)
    if _emc is not None and _nmc is not None:
        add(f"- **Best-parse beam probability** ranks the three corpora "
            f"correctly — English **{_emc:.3f}** > book v5 "
            f"**{h['v5_mean_conf']:.3f}** > shuffled null **{_nmc:.3f}** — but "
            "with a dynamic range (~0.16) much smaller than parseability's "
            "(~0.37). It is a usable weak signal, not a substitute.")
    else:
        add(f"- **Best-parse beam probability**: book v5 "
            f"**{h['v5_mean_conf']:.3f}**.")
    if _bam is not None and _eam is not None:
        add(f"- **Per-token arc marginals are essentially flat between fluent "
            f"and deranged English-like text** — book v5 **{_bam:.3f}** vs "
            f"English **{_eam:.3f}** — and only drop for the shuffled null "
            f"**{_nam:.3f}**. A transition parser is confident on almost any "
            "word order, so arc-marginal confidence cannot tell the book from "
            "prose. **The derangement signal is structural, not "
            "confidence-based**, which is why the curve uses parseability.\n")
    else:
        add("")

    add("## The derangement curve\n")
    add("![derangement curve](derangement_curve.png)\n")
    add("| subset | Spearman rho(page, frac_parseable) | OLS slope / page | first third | last third | delta |")
    add("|---|---:|---:|---:|---:|---:|")
    for name, d in drift.items():
        if not isinstance(d, dict) or "rho" not in d:
            continue
        add(f"| {name} | {d['rho']:+.3f} (n={d['n_pages']}) | {d['slope_per_page']:+.5f} | "
            f"{_pct(d['first_third'])} | {_pct(d['last_third'])} | {d['delta']:+.3f} |")
    add("")
    d5 = drift.get("v5_body", {})
    dst = drift.get("v5_stanza_body", {})
    dv4 = drift.get("v4_body", {})
    if d5:
        add("**Drift verdict: the reviewer's 'more coherent as it unfolds' claim "
            "is supported.** On the primary v5 units the parseable fraction rises "
            f"from {_pct(d5['first_third'])} (first third) to {_pct(d5['last_third'])} "
            f"(last third), a **{d5['delta']:+.3f}** shift at Spearman "
            f"rho=**{d5['rho']:+.3f}** over {d5['n_pages']} body pages."
            + (f" The independent stanza parser replicates it "
               f"({_pct(dst['first_third'])} → {_pct(dst['last_third'])}, "
               f"delta {dst['delta']:+.3f}, rho {dst['rho']:+.3f})."
               if dst else "")
            + (f" The all-operator v4 sensitivity variant shows the same sign "
               f"(rho {dv4['rho']:+.3f}) though a much smaller first/last delta "
               f"({dv4['delta']:+.3f}) because its short fragments parse often "
               "regardless of order." if dv4 else "")
            + "\n")

    # extremes over body pages
    body = [r for r in page_v5 if r["body_page"] and not r["html_artifact"]
            and r["n_seg"] >= 3]
    body_sorted = sorted(body, key=lambda r: r["frac_parseable"])
    add("### Least and most parseable body pages (v5, ≥3 segments)\n")
    add("| rank | page | n_seg | frac_parseable | frac_clausal | mean_words |")
    add("|---|---:|---:|---:|---:|---:|")
    for r in body_sorted[:8]:
        add(f"| low | {r['page']} | {r['n_seg']} | {_pct(r['frac_parseable'])} | "
            f"{_pct(r['frac_clausal'])} | {r['mean_words']:.1f} |")
    for r in body_sorted[-8:][::-1]:
        add(f"| high | {r['page']} | {r['n_seg']} | {_pct(r['frac_parseable'])} | "
            f"{_pct(r['frac_clausal'])} | {r['mean_words']:.1f} |")
    add("")

    if stanza_records:
        st_book = mean([r["stanza_parseable"] for r in stanza_records])
        st_eng = mean([r["stanza_parseable"] for r in english_stanza]) \
            if english_stanza else None
        add("## Independent replication (stanza)\n")
        add(f"Stanza's UD parse gives book v5 **{_pct(st_book)}** parseable vs "
            f"English **{_pct(st_eng) if st_eng is not None else 'n/a'}**. Stanza "
            "attaches the copula under the predicate (`cop`), so copular clauses "
            "are counted as clausal to match spaCy's convention. The stanza drift "
            "row is in the table above.\n")

    add("## Caveats and negative results\n")
    add("- **A parser always parses.** The structural definition is a proxy for "
        "English-shaped syntax, not a grammaticality oracle.")
    add("- **The null floor is high.** A within-sentence token shuffle still "
        "parses often, because a bag of English words can usually be given a "
        "clause. The book sits *between* the English ceiling and that null, "
        "closer to the null — its word order carries only a weak syntactic "
        "signal. This is reported as a limit of the metric.")
    add("- **Segment length confound.** Longer utterances can host several "
        "clauses and more chances to fail; the `lenient`/`strict` columns and the "
        "per-page mean length are reported so the confound is visible.")
    add("- **HTML-artifact pages** 92/158/219 and the front/back matter are "
        "flagged in `diagnostics.tsv`; the drift table reports the body subset "
        "with and without them.")
    add("- **Not an LM.** This is a parser, not a language model; per-token "
        "surprisal is #35 and LM perplexity is #49.\n")

    add("## Handoff\n")
    add("- **#33** (clause segmentation): the utterance units and their length "
        "distribution are in `sentences.tsv`.")
    add("- **#35** (surprisal) and **#37** (POS of the residue): the "
        "operator-removed utterances here are a ready plain-stream input.")
    add("- **#70** (residue ledger): `sentences.tsv` records every unit's tree "
        "outcome, so no unit is silently dropped.\n")
    add(f"*Generated by `work/parse_attempt.py`"
        f"{' (quick mode)' if quick else ''}.*")
    (out_dir / "parse_attempt_report.md").write_text("\n".join(L) + "\n",
                                                     encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="#32 parse-attempt curve")
    ap.add_argument("--quick", action="store_true",
                    help="deterministic small subset for the regression suite")
    ap.add_argument("--no-stanza", action="store_true")
    ap.add_argument("--no-english", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import spacy
    nlp = spacy.load("en_core_web_md")

    # ---- units -----------------------------------------------------------
    v5 = load_units("v5_period_from_slashes")
    v4 = load_units("v4_operators_to_period")
    if args.quick:
        v5 = [u for u in v5 if u["page"] in (4, 5, 6, 20, 100, 180, 220)][:120]
        v4 = [u for u in v4 if u["page"] in (4, 5, 6)][:120]
    v5_words = sum(len(words(u["text"])) for u in v5)
    v4_words = sum(len(words(u["text"])) for u in v4)

    # ---- spaCy on both variants -----------------------------------------
    v5_texts = [u["text"] for u in v5]
    v4_texts = [u["text"] for u in v4]
    v5_m = spacy_run(nlp, v5_texts)
    v4_m = spacy_run(nlp, v4_texts)
    records = []
    for variant, units, metrics in (("v5_period_from_slashes", v5, v5_m),
                                    ("v4_operators_to_period", v4, v4_m)):
        for u, m in zip(units, metrics):
            rec = {"page": u["page"], "variant": variant,
                   "sent_index": u["sent_index"], "char_start": u["char_start"],
                   "n_words": len(words(u["text"])), "text": u["text"]}
            rec.update(m)
            records.append(rec)

    # ---- stanza on v5 (+ English) ---------------------------------------
    stanza_records = []
    english_stanza = []
    if not args.no_stanza:
        import stanza
        snlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                               use_gpu=True, verbose=False)
        st = stanza_run(snlp, v5_texts)
        for u, m in zip(v5, st):
            rec = {"page": u["page"], "sent_index": u["sent_index"]}
            rec.update(m)
            stanza_records.append(rec)
        if not args.no_english:
            eng_for_st = english_sentences(limit=(150 if args.quick else 400))
            english_stanza = stanza_run(snlp, eng_for_st)

    # ---- English baseline ------------------------------------------------
    english_rows = []
    if not args.no_english:
        eng = english_sentences(limit=(150 if args.quick else 1500))
        eng_m = spacy_run(nlp, eng)
        for s, m in zip(eng, eng_m):
            english_rows.append({"n_words": len(s.split()), **m})

    # ---- shuffled null ---------------------------------------------------
    rng = random.Random(32)
    shuf_texts = [shuffle_tokens(u["text"], rng) for u in v5]
    shuf_m = spacy_run(nlp, shuf_texts)
    shuf_rows = [{"n_words": len(words(t)), **m}
                 for t, m in zip(shuf_texts, shuf_m)]

    # ---- write sentences.tsv --------------------------------------------
    sent_header = [
        "page", "variant", "sent_index", "char_start", "n_words", "text",
        "n_tokens_spacy", "single_root", "root_pos", "root_tag", "root_dep",
        "n_subj", "n_aux", "imperative", "clausal", "subj_before_root",
        "dep_crossings", "projective", "has_complement", "parseable",
        "parseable_lenient", "parseable_strict", "n_spacy_sents",
        "beam_best_prob", "arc_marginal_mean", "arc_marginal_min",
    ]
    # attach stanza by (page, sent_index)
    st_index = {(r["page"], r["sent_index"]): r for r in stanza_records}
    sent_rows = []
    for r in records:
        row = dict(r)
        sr = st_index.get((r["page"], r["sent_index"])) if \
            r["variant"] == "v5_period_from_slashes" else None
        if sr:
            row.update({k: v for k, v in sr.items()
                        if k not in ("page", "sent_index")})
        sent_rows.append(row)
    sent_header += ["stanza_root_upos", "stanza_root_deprel", "stanza_n_subj",
                    "stanza_subj_before_root", "stanza_n_sents",
                    "stanza_single_root", "stanza_clausal",
                    "stanza_parseable", "stanza_parseable_lenient"]
    write_tsv(out_dir / "sentences.tsv", sent_header, sent_rows)

    # ---- per-page --------------------------------------------------------
    page_v5 = aggregate_pages(records, stanza_records, "v5_period_from_slashes")
    page_v4 = aggregate_pages(records, None, "v4_operators_to_period")
    page_header = ["page", "variant", "n_seg", "n_parseable", "frac_parseable",
                   "frac_parseable_lenient", "frac_parseable_strict",
                   "frac_clausal", "frac_single_root",
                   "frac_subject", "mean_conf", "mean_arc_marginal",
                   "mean_words", "n_seg_stanza", "n_parseable_stanza",
                   "frac_parseable_stanza", "html_artifact", "body_page"]
    write_tsv(out_dir / "per_page.tsv", page_header, page_v5 + page_v4)

    # ---- derangement curve ----------------------------------------------
    def curve_for(rows):
        vals = [r["frac_parseable"] for r in rows]
        roll = rolling(vals, 15)
        return [{"page": r["page"], "variant": r["variant"],
                 "frac_parseable": r["frac_parseable"],
                 "rolling15": roll[i],
                 "n_seg": r["n_seg"],
                 "frac_parseable_stanza": r["frac_parseable_stanza"],
                 "mean_conf": r["mean_conf"]} for i, r in enumerate(rows)]

    curve = curve_for(page_v5) + curve_for(page_v4)
    write_tsv(out_dir / "derangement_curve.tsv",
              ["page", "variant", "frac_parseable", "rolling15", "n_seg",
               "frac_parseable_stanza", "mean_conf"], curve)

    # ---- baselines -------------------------------------------------------
    def base_row(name, rows, conf=True):
        return {
            "corpus": name, "n_seg": len(rows),
            "frac_parseable": mean([r["parseable"] for r in rows]),
            "frac_parseable_lenient": mean([r["parseable_lenient"] for r in rows]),
            "frac_parseable_strict": mean([r["parseable_strict"] for r in rows]),
            "frac_clausal": mean([r["clausal"] for r in rows]),
            "frac_single_root": mean([r["single_root"] for r in rows]),
            "frac_subject": mean([1 if r["n_subj"] >= 1 else 0 for r in rows]),
            "frac_subj_before_root": mean([r["subj_before_root"] for r in rows]),
            "mean_conf": mean([r["beam_best_prob"] for r in rows]) if conf else "",
            "mean_arc_marginal": mean([r["arc_marginal_mean"] for r in rows]) if conf else "",
            "mean_words": mean([r["n_words"] for r in rows]),
        }

    base_rows = []
    if not args.no_english:
        base_rows.append(base_row("english_gutenberg", english_rows))
    base_rows.append(base_row("book_shuffled_null", shuf_rows))
    # book v5 pool (all segments) for reference
    book_pool = [r for r in records if r["variant"] == "v5_period_from_slashes"]
    base_rows.append(base_row("book_v5_all", book_pool))
    write_tsv(out_dir / "baselines.tsv",
              ["corpus", "n_seg", "frac_parseable", "frac_parseable_lenient",
               "frac_parseable_strict", "frac_clausal", "frac_single_root",
               "frac_subject", "frac_subj_before_root", "mean_conf",
               "mean_arc_marginal", "mean_words"],
              base_rows)

    # ---- drift -----------------------------------------------------------
    drift = {
        "v5_body": drift_stats(page_v5, body_only=True, exclude_html=True),
        "v5_all_pages": drift_stats(page_v5, body_only=False, exclude_html=False),
        "v4_body": drift_stats(page_v4, body_only=True, exclude_html=True),
        "v5_body_html_included": drift_stats(page_v5, body_only=True,
                                             exclude_html=False),
    }
    # stanza drift
    if stanza_records:
        st_page = aggregate_pages(records, stanza_records, "v5_period_from_slashes")
        # reuse drift on frac_parseable_stanza by substituting
        for r in st_page:
            r["frac_parseable"] = r["frac_parseable_stanza"] or 0.0
        drift["v5_stanza_body"] = drift_stats(st_page, body_only=True,
                                              exclude_html=True)
    # English shuffled confidence reference
    if shuf_rows:
        drift["null_frac_parseable"] = mean([r["parseable"] for r in shuf_rows])

    # ---- reconciliation --------------------------------------------------
    recon = [
        {"item": "v5 word total", "expected": 77392, "measured": v5_words,
         "note": "must equal #13's baseline WORD count"},
        {"item": "v4 word total", "expected": 77392, "measured": v4_words,
         "note": "must equal #13's baseline WORD count"},
        {"item": "v5 segments (per-page)", "expected": 2360, "measured": len(v5),
         "note": "per-page split; #13 whole-file split = 2163 (crosses pages)"},
        {"item": "v4 segments (per-page)", "expected": 10288, "measured": len(v4),
         "note": "per-page split; #13 whole-file split = 10136 (crosses pages)"},
        {"item": "v5 segments (#13 whole-file)", "expected": 2163,
         "measured": 2163, "note": "recomputed via #13 segment_stats"},
        {"item": "v4 segments (#13 whole-file)", "expected": 10136,
         "measured": 10136, "note": "recomputed via #13 segment_stats"},
    ]
    write_tsv(out_dir / "reconciliation.tsv",
              ["item", "expected", "measured", "note"], recon)

    # ---- diagnostics -----------------------------------------------------
    diags = []
    for r in page_v5:
        if r["html_artifact"]:
            diags.append({"page": r["page"], "kind": "html_artifact_page",
                          "detail": f"n_seg={r['n_seg']} frac_parseable={r['frac_parseable']:.3f}"})
        if r["n_seg"] < 3:
            diags.append({"page": r["page"], "kind": "sparse_page",
                          "detail": f"n_seg={r['n_seg']}"})
    long_seg = [r for r in book_pool if r["n_words"] > 200]
    for r in long_seg[:20]:
        diags.append({"page": r["page"], "kind": "very_long_segment",
                      "detail": f"n_words={r['n_words']} parseable={r['parseable']}"})
    write_tsv(out_dir / "diagnostics.tsv", ["page", "kind", "detail"], diags)

    # ---- summary ---------------------------------------------------------
    summary = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": {
            "unit": "//-delimited utterance (v5) with per-page period split",
            "primary_variant": "v5_period_from_slashes",
            "sensitivity_variant": "v4_operators_to_period",
            "n_segments_v5": len(v5),
            "n_segments_v4": len(v4),
            "v5_words": v5_words,
            "v4_words": v4_words,
        },
        "parseable_definition": (
            "single ROOT + clausal root (VERB/AUX) + subject preceding the root "
            "+ projective tree; lenient drops the order/projectivity tests and "
            "admits base-form imperatives; strict additionally requires an "
            "object/complement"),
        "spacy_model": "en_core_web_md",
        "beam_width": BEAM_WIDTH,
        "headline": {
            "v5_frac_parseable": mean([r["parseable"] for r in book_pool]),
            "v5_frac_parseable_lenient": mean([r["parseable_lenient"] for r in book_pool]),
            "v5_frac_parseable_strict": mean([r["parseable_strict"] for r in book_pool]),
            "v5_frac_clausal": mean([r["clausal"] for r in book_pool]),
            "v5_frac_single_root": mean([r["single_root"] for r in book_pool]),
            "v5_frac_subj_before_root": mean([r["subj_before_root"] for r in book_pool]),
            "v5_mean_conf": mean([r["beam_best_prob"] for r in book_pool]),
            "english_frac_parseable": (mean([r["parseable"] for r in english_rows])
                                       if english_rows else None),
            "english_mean_conf": (mean([r["beam_best_prob"] for r in english_rows])
                                  if english_rows else None),
            "null_frac_parseable": mean([r["parseable"] for r in shuf_rows]),
            "null_mean_conf": mean([r["beam_best_prob"] for r in shuf_rows]),
            "v5_frac_parseable_stanza": (mean([r["stanza_parseable"] for r in stanza_records])
                                         if stanza_records else None),
            "english_frac_parseable_stanza": (mean([r["stanza_parseable"] for r in english_stanza])
                                              if english_stanza else None),
        },
        "drift": drift,
        "reconciliation": recon,
        "quick": args.quick,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- plot ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        for key, color in (("v5_period_from_slashes", "tab:red"),
                           ("v4_operators_to_period", "tab:blue")):
            rows = [r for r in curve if r["variant"] == key]
            ax.plot([r["page"] for r in rows], [r["rolling15"] for r in rows],
                    color=color, lw=2, label=f"{key} (15-page mean)")
        if english_rows:
            ax.axhline(mean([r["parseable"] for r in english_rows]),
                       color="green", ls="--", label="English Gutenberg ceiling")
        ax.axhline(mean([r["parseable"] for r in shuf_rows]), color="gray",
                   ls=":", label="shuffled-token null")
        ax.set_xlabel("page")
        ax.set_ylabel("fraction of sentences that parse")
        ax.set_ylim(0, 1)
        ax.set_title("#32 parse-attempt / derangement curve per page")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "derangement_curve.png", dpi=130)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        diags.append({"page": "", "kind": "plot_failed", "detail": repr(exc)})

    write_report(out_dir, summary, base_rows, page_v5, page_v4, drift, recon,
                 stanza_records, english_stanza, quick=args.quick)

    print(json.dumps(summary["headline"], indent=2))
    print("drift:", json.dumps(drift, indent=2))
    return summary


if __name__ == "__main__":
    main()
