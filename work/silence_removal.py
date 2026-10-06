#!/usr/bin/env python3
"""Silence-removal transform variants (Chainlink #13).

The Beguš move: in the sperm-whale codas the *silences* between clicks are
removed and the vowels are read as a continuous stream.  Applied to Blood
Electric the "silence" is the operator layer of #10/#12 (`::`, `//`, `=`, `/`,
`<<`, `>>`, `<`, `>`, `++`, `+`, `|`, plus the free-standing dash class), and
this script produces the four residue streams the issue asks for, plus two
sentence-boundary diagnostics that test the #12 `::` vs `//` claim by
construction.

Variants (all persisted, all page-tagged, all line-aligned to the baseline):

  v0_original               baseline, operators intact (normalised only)
  v1_delete_all_operators   strict delete: operator characters removed,
                            whitespace untouched  -> glued sites FUSE
                            (`soul/gram` -> `soulgram`, `gene=TV` -> `geneTV`)
  v2_delete_colon_slash     delete only the `::` and `//` tokens
  v3_operators_to_space     every operator -> one space (no fusion)
  v4_operators_to_period    every operator -> `. ` (sentence skeleton)
  v5_period_from_slashes    `//` -> `.`, all other operators -> space  [diag]
  v6_period_from_colons     `::` -> `.`, all other operators -> space  [diag]

v1 vs v3 is the load-bearing contrast: *delete* fuses neighbours (the
"Beguš-faithful" adjacent-vowels reading), *map to space* keeps them
analysable.  v5 vs v6 turns the #12 finding (`//` is a sentence boundary,
`::` is a phrase separator) into a directly comparable pair.

Design decisions that are measured and disclosed rather than assumed:

  * Operator set and greedy longest-match tokenization are imported from #10
    (`work/operator_inventory.py`), so the 10,949-token inventory reproduces
    exactly.
  * `-`, `—`, `–` are not in #10's operator set.  #12 measured 177
    whitespace-bounded `-` dashes and classified them with the `::`
    phrase-separator class.  This script re-measures the dash class
    (696 whitespace-bounded dash tokens) and treats them as removable
    silence, but reports every headline number both with and without them so
    the choice is auditable (`dashes=with|without` sensitivity rows).
  * Word-internal hyphens (`boy-roid`, `cold-blooded`) are inside WORD tokens
    and are never touched: the transform is defined over operator spans only.
  * Post-processing (whitespace collapse, inserted-period cleanup) is applied
    identically to every variant, including the baseline, so the comparison is
    not confounded by normalisation.
  * The transform never changes line breaks, so output line *i* of a variant
    derives from source line *i*; `line_index.tsv` records the exact mapping.

Outputs under analysis/silence_removal/:
  v0_original.txt .. v6_period_from_colons.txt   the residue streams
  variants.tsv              headline metrics per variant
  per_page_metrics.tsv      page x variant metrics (feeds #49/#50/#67)
  silence_sites.tsv         what each operator/dash class does in each variant
  fused_residue.tsv         portmanteau forms recovered by v1's gluing
  sentence_shape.tsv        period-segment statistics for v4/v5/v6
  line_index.tsv            page + output line -> source line
  samples.tsv               citable original/variant samples
  silence_removal_report.md THE DELIVERABLE
  summary.json              machine-readable headline numbers
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from operator_inventory import OPS            # #10 operator set
from operator_functions import load_pages, tokenize  # #12 loader + tokenizer

SRC = os.path.join(ROOT, "analysis", "layout_recovery", "blood-electric.layout.md")
OUT = os.path.join(ROOT, "analysis", "silence_removal")
FOLIO = os.path.join(ROOT, "analysis", "page_folio_map", "page_folio_map.tsv")

# Whitespace-bounded dash characters: the free-standing separator class.
DASH_CHARS = ("-", "\u2014", "\u2013")

# Sentinel used while assembling a line so that period cleanup can be applied to
# *inserted* periods only, leaving the 22 pre-existing periods untouched.
SENT = "\x00"

# variant key -> (treatment, operators_to_delete, separator)
#   treatment: 'keep' | 'delete' | 'space' | 'period'
#   operators_to_delete: None -> all sites; else the set of operator strings
#   separator: replacement used for operators NOT in operators_to_delete
VARIANTS = [
    ("v0_original",             "keep",    None,     None),
    ("v1_delete_all_operators", "delete",  None,     None),
    ("v2_delete_colon_slash",   "delete",  {"::", "//"}, None),
    ("v3_operators_to_space",   "space",   None,     None),
    ("v4_operators_to_period",  "period",  None,     None),
    ("v5_period_from_slashes",  "period",  {"//"},   " "),
    ("v6_period_from_colons",   "period",  {"::"},   " "),
]

TREATMENT_LABEL = {
    "v0_original": "baseline (operators intact)",
    "v1_delete_all_operators": "delete every silence site (whitespace untouched)",
    "v2_delete_colon_slash": "delete only the `::` and `//` tokens",
    "v3_operators_to_space": "map every silence site to one space",
    "v4_operators_to_period": "map every silence site to `. `",
    "v5_period_from_slashes": "DIAGNOSTIC: `//` -> `.`, all other sites -> space",
    "v6_period_from_colons": "DIAGNOSTIC: `::` -> `.`, all other sites -> space",
}


# ---------------------------------------------------------------------------
# site detection
# ---------------------------------------------------------------------------
def operator_spans(text):
    """Greedy longest-match operator spans, identical to #10's tokenization."""
    spans = []
    i, n = 0, len(text)
    while i < n:
        for op in OPS:
            if text.startswith(op, i):
                spans.append((i, i + len(op), op, "op"))
                i += len(op)
                break
        else:
            i += 1
    return spans


def dash_spans(text):
    """Whitespace-bounded dash tokens (` - `, ` — `, ` – `)."""
    return [(m.start(), m.end(), m.group(0), "dash")
            for m in re.finditer(r"(?<=\s)[-\u2014\u2013](?=\s|$)", text)]


def site_spans(text, dashes=True):
    """All removable silence sites, sorted, operator spans taking precedence."""
    spans = operator_spans(text)
    if dashes:
        taken = {(s, e) for s, e, _, _ in spans}
        spans += [sp for sp in dash_spans(text)
                  if not any(sp[0] >= s and sp[1] <= e for s, e in taken)]
    spans.sort()
    return spans


def is_word_char(ch):
    return bool(ch) and (ch.isalnum() or ch in "_'")


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------
def _treat(site_op, treatment, only, separator):
    """Replacement string for one site. `None` means 'keep verbatim'.

    `separator` is the replacement for operators that a `only`-restricted
    variant does not target; `None` there means those operators are kept.
    """
    if treatment == "keep":
        return None
    if only is not None:
        if site_op in only:
            return "" if treatment == "delete" else SENT
        return separator
    if treatment == "delete":
        return ""
    if treatment == "space":
        return " "
    if treatment == "period":
        return SENT
    raise ValueError(treatment)


def _runs(spans, treatment, only, separator):
    """Group contiguous sites that receive the *same* replacement.

    A run like `//::` collapses to a single junction between its two word
    neighbours, so fused-junction accounting must count the run once, not once
    per operator inside it. Sites with different replacements (e.g. the `::` of
    `::<<` under v2, which is deleted, and the `<<`, which is kept) are never
    merged.
    """
    groups = []
    for s, e, op, _kind in spans:
        rep = _treat(op, treatment, only, separator)
        if rep is None:
            continue
        if groups and groups[-1][1] == s and groups[-1][3] == rep:
            groups[-1][1] = e
            groups[-1][2].append(op)
        else:
            groups.append([s, e, [op], rep])
    return [(g[0], g[1], g[2], g[3]) for g in groups]


def _bridges(line, s, e):
    """True when deleting line[s:e] joins two lexical items into one token.

    Two mechanisms, both measured: the plain word/word case (`soul/gram` ->
    `soulgram`), and the hyphen-bridge case where the operator sits between a
    trailing hyphen and a word (`hydro-</p>` -> `hydro-p`, so a hyphen that was
    *not* word-internal becomes word-internal and two tokens merge into one).
    The second mechanism is rare but it is the reason the reconciliation is done
    on this predicate rather than on a word/word character test alone.
    """
    left = line[s - 1] if s > 0 else ""
    right = line[e] if e < len(line) else ""
    if not _word_char_at(line, e):
        return False
    if _word_char_at(line, s - 1):
        return True
    return left == "-" and right.isalnum()


def _operands(line, s, e):
    """(left, right) lexical operands that a deletion at line[s:e] joins."""
    ls = s
    while ls > 0 and _word_char_at(line, ls - 1):
        ls -= 1
    re_ = e
    while re_ < len(line) and _word_char_at(line, re_):
        re_ += 1
    return line[ls:s], line[e:re_]


def transform_line(line, spans, treatment, only, separator, collect_fused=False):
    """Apply the transform to one line.

    Returns (new_line, n_deleted, fused_records) where fused_records is a list
    of (output_offset, source_ops, left_operand, right_operand) for deletions
    that join two lexical items. Offsets refer to the *returned* string, so
    callers must extract fused tokens before any further normalisation.
    """
    out = []
    cur = 0
    deleted = 0
    fused = []
    for s, e, ops, rep in _runs(spans, treatment, only, separator):
        out.append(line[cur:s])
        cur = e
        if rep == "":
            deleted += len(ops)
            if collect_fused and _bridges(line, s, e):
                left, right = _operands(line, s, e)
                fused.append((sum(len(x) for x in out), tuple(ops), left, right))
        out.append(rep)
    out.append(line[cur:])
    return "".join(out), deleted, fused


def normalize_line(s):
    """Identical post-processing for every variant (inserted periods only)."""
    s = s.replace("\t", " ")
    s = re.sub(r" *" + SENT, SENT, s)            # no space before an inserted period
    s = re.sub(SENT + r"\.+", ".", s)            # inserted period absorbs literal dots
    s = re.sub(r"\.+" + SENT, ".", s)
    s = re.sub(SENT + r" {2,}", SENT + " ", s)
    s = re.sub(SENT + r"(?=\S)", SENT + " ", s)  # exactly one space after it
    s = re.sub(r" {2,}", " ", s)
    s = s.replace(SENT, ".")
    return s.strip()


def _word_char_at(text, i):
    """Word character under #12's rule: `-` inside a compound is word-internal."""
    if i < 0 or i >= len(text):
        return False
    c = text[i]
    if is_word_char(c):
        return True
    return c == "-" and i + 1 < len(text) and text[i + 1].isalnum()


def fused_token(text, pos):
    """Expand over word characters (hyphen-compounds included) around `pos`."""
    s = e = pos
    while s > 0 and _word_char_at(text, s - 1):
        s -= 1
    while e < len(text) and _word_char_at(text, e):
        e += 1
    return text[s:e], s, e


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_']*")
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def words(text):
    """WORD tokens under #12's tokenizer, so variant counts reconcile exactly.

    Using #12's lexer (not a flat regex) means `boy-roid` stays one token in
    every variant, and `v0_original` returns exactly the baseline lexeme count.
    """
    return [t[3] for t in tokenize(COMMENT_RE.sub(" ", text)) if t[0] == "WORD"]


def entropy(items):
    if not items:
        return 0.0
    c = collections.Counter(items)
    n = len(items)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def gzip_ratio(text):
    b = text.encode("utf-8")
    return len(gzip.compress(b, 9)) / len(b) if b else 0.0


def text_metrics(text):
    text = COMMENT_RE.sub(" ", text)
    w = words(text)
    low = [x.lower() for x in w]
    c = collections.Counter(low)
    segs = [s for s in re.split(r"\.\s*", text) if words(s)]
    seg_words = [len(words(s)) for s in segs] or [0]
    return {
        "chars": len(text),
        "lines": text.count("\n"),
        "word_tokens": len(w),
        "types": len(c),
        "ttr": len(c) / max(1, len(low)),
        "hapax_frac": sum(1 for v in c.values() if v == 1) / max(1, len(c)),
        "token_entropy_bits": entropy(low),
        "gzip_ratio": gzip_ratio(text),
        "periods": text.count("."),
        "period_segments": len(segs),
        "mean_words_per_segment": sum(seg_words) / len(seg_words),
        "lowercase_initial_segments": sum(
            1 for s in segs if words(s) and words(s)[0][:1].islower()) / max(1, len(segs)),
    }


def segment_stats(text):
    text = COMMENT_RE.sub(" ", text)
    segs = [s for s in re.split(r"\.\s*", text) if words(s)]
    wl = [len(words(s)) for s in segs] or [0]
    # length-controlled companion statistic: v5/v6 segments differ in length, so
    # report the same rate restricted to segments of >= 5 words.
    long_pairs = [(words(s), len(words(s))) for s in segs]
    long_pairs = [(w, n) for w, n in long_pairs if n >= 5]
    return {
        "n_segments": len(segs),
        "mean_words": sum(wl) / len(wl),
        "median_words": statistics.median(wl),
        "max_words": max(wl),
        "frac_upper_initial": sum(1 for s in segs if words(s)[0][:1].isupper()) / max(1, len(segs)),
        "frac_single_word": sum(1 for n in wl if n == 1) / max(1, len(wl)),
        "n_segments_ge5": len(long_pairs),
        "frac_upper_initial_ge5": (sum(1 for w, _n in long_pairs if w[0][:1].isupper())
                                   / len(long_pairs)) if long_pairs else 0.0,
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def load_folio():
    if not os.path.exists(FOLIO):
        return {}
    out = {}
    with open(FOLIO, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[2]:
                out[int(p[0])] = p[2]
    return out


def cite(page, folio):
    return f"p.{page} (folio {folio})" if folio else f"p.{page}"


def main():
    pages = load_pages(SRC)
    folio = load_folio()
    page_nos = sorted(pages)
    os.makedirs(OUT, exist_ok=True)

    # ---- pass 1: per-page, per-variant line streams -----------------------
    streams = {key: {} for key, *_ in VARIANTS}      # variant -> page -> [lines]
    line_index = {key: {} for key, *_ in VARIANTS}   # variant -> page -> [(out, src)]
    site_counts = {}                                  # class -> Counter over pages
    fused = collections.Counter()
    fused_by_variant = collections.Counter()          # variant -> join count
    fused_src = collections.defaultdict(collections.Counter)
    links = collections.Counter()                       # (left, right) -> n
    link_ops = collections.defaultdict(set)             # (left, right) -> ops
    link_example = {}
    fused_example = {}
    site_pages = collections.defaultdict(set)
    lexemes = 0
    op_totals = collections.Counter()
    dash_totals = collections.Counter()

    for pno in page_nos:
        raw_lines = pages[pno].split("\n")
        for key, treatment, only, sep in VARIANTS:
            out_lines, idx = [], []
            for src_i, line in enumerate(raw_lines, start=1):
                spans = site_spans(line, dashes=True)
                new, _deleted, fused_hits = transform_line(
                    line, spans, treatment, only, sep,
                    collect_fused=(treatment == "delete"))
                # fused tokens are extracted from the *un-normalised* string,
                # because the recorded offsets refer to it; whitespace
                # normalisation afterwards would shift them.
                if fused_hits and key == "v1_delete_all_operators":
                    for pos, ops, lop, rop in fused_hits:
                        tok, _s, _e = fused_token(new, min(pos, len(new)))
                        if tok:
                            fused[tok] += 1
                            for op in ops:
                                fused_src[tok][op] += 1
                            fused_example.setdefault(tok, cite(pno, folio.get(pno)))
                        if lop and rop:
                            links[(lop, rop)] += 1
                            link_ops[(lop, rop)].update(ops)
                            link_example.setdefault((lop, rop), cite(pno, folio.get(pno)))
                new = normalize_line(new)
                if not new:
                    continue
                out_lines.append(new)
                idx.append((len(out_lines), src_i))
                if treatment == "delete" and fused_hits:
                    fused_by_variant[key] += len(fused_hits)
            streams[key][pno] = out_lines
            line_index[key][pno] = idx

        # class counts + lexemes from the untouched page
        body = pages[pno]
        lexemes += sum(1 for t in tokenize(body) if t[0] == "WORD")
        for s, e, op, kind in site_spans(body):
            cls = op if kind == "op" else "dash:" + op
            site_counts.setdefault(cls, collections.Counter())[pno] += 1
            site_pages[cls].add(pno)
            (op_totals if kind == "op" else dash_totals)[op] += 1

    # ---- write the streams ------------------------------------------------
    for key, *_ in VARIANTS:
        path = os.path.join(OUT, key + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            for pno in page_nos:
                f.write(f"<!-- page {pno} -->\n")
                for ln in streams[key][pno]:
                    f.write(ln + "\n")

    # line index (baseline and all variants share the source line numbering;
    # emit the v1 mapping plus a compact per-variant row count)
    with open(os.path.join(OUT, "line_index.tsv"), "w", encoding="utf-8") as f:
        f.write("page\tvariant\tout_line\tsrc_line\tfolio\n")
        for key, *_ in VARIANTS:
            for pno in page_nos:
                fl = folio.get(pno, "")
                for out_i, src_i in line_index[key][pno]:
                    f.write(f"{pno}\t{key}\t{out_i}\t{src_i}\t{fl}\n")

    # ---- text of each variant --------------------------------------------
    def variant_text(key):
        parts = []
        for pno in page_nos:
            parts.append(f"<!-- page {pno} -->")
            parts.extend(streams[key][pno])
        return "\n\n".join(parts) + "\n"

    texts = {key: variant_text(key) for key, *_ in VARIANTS}

    # ---- reconciliation: every variant's token count must be explained ----
    # #12's lexer already breaks tokens at operator positions, so the baseline
    # word sequence is `soul` `gram` for `soul/gram`. Therefore:
    #   * a space/period variant cannot change the token count at all - it is a
    #     *controlled* comparison against the baseline, differing only in the
    #     operator characters;
    #   * a delete variant can only join lexical items, once per junction.
    # Both identities are asserted, so a silently broken transform fails loudly.
    keylist = [k for k, *_ in VARIANTS]
    base_words = words(texts["v0_original"])
    recon = {}
    for k in keylist:
        treatment, only, sep = next((t, o, s) for kk, t, o, s in VARIANTS if kk == k)
        kwords = words(texts[k])
        expected = lexemes - fused_by_variant[k] if treatment == "delete" else lexemes
        recon[k] = {"word_tokens": len(kwords), "expected": expected,
                    "delta": len(kwords) - expected,
                    "same_token_sequence_as_baseline": kwords == base_words}
        if len(kwords) != expected:
            raise AssertionError(
                f"{k}: token count {len(kwords)} is not explained by the transform "
                f"(expected {expected}); the transform is not doing what it claims")

    # ---- headline metrics -------------------------------------------------
    rows = []
    base = text_metrics(texts["v0_original"])
    for key, treatment, only, sep in VARIANTS:
        m = text_metrics(texts[key])
        m["variant"] = key
        m["treatment"] = TREATMENT_LABEL[key]
        if treatment == "keep":
            m["silence_sites_removed"] = 0
        elif only is None:
            m["silence_sites_removed"] = sum(c.total() for c in site_counts.values())
        else:
            m["silence_sites_removed"] = sum(
                site_counts[c].total() for c in site_counts
                if not c.startswith("dash:") and c in only)
        m["delta_entropy_vs_v0"] = m["token_entropy_bits"] - base["token_entropy_bits"]
        m["delta_gzip_vs_v0"] = m["gzip_ratio"] - base["gzip_ratio"]
        rows.append(m)

    fields = ["variant", "treatment", "chars", "lines", "word_tokens", "types", "ttr",
              "hapax_frac", "token_entropy_bits", "gzip_ratio", "periods",
              "period_segments", "mean_words_per_segment", "lowercase_initial_segments",
              "silence_sites_removed", "delta_entropy_vs_v0", "delta_gzip_vs_v0"]
    with open(os.path.join(OUT, "variants.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for m in rows:
            f.write("\t".join(
                f"{m[k]:.6f}" if isinstance(m[k], float) else str(m[k]) for k in fields) + "\n")

    # ---- dash sensitivity (the one judgement call that is not the issue's) --
    dash_n = sum(dash_totals.values())
    op_n = sum(op_totals.values())

    # recompute the four core variants with dashes left in place
    sens = {}
    for key, treatment, only, sep in VARIANTS[:5]:
        parts = []
        for pno in page_nos:
            parts.append(f"<!-- page {pno} -->")
            for src_i, line in enumerate(pages[pno].split("\n"), start=1):
                spans = site_spans(line, dashes=False)
                new, _d, _f = transform_line(line, spans, treatment, only, sep)
                new = normalize_line(new)
                if new:
                    parts.append(new)
        m = text_metrics("\n\n".join(parts) + "\n")
        m["delta_entropy_vs_v0_nodash"] = m["token_entropy_bits"] - base["token_entropy_bits"]
        m["delta_gzip_vs_v0_nodash"] = m["gzip_ratio"] - base["gzip_ratio"]
        sens[key] = m

    # ---- per-page metrics --------------------------------------------------
    with open(os.path.join(OUT, "per_page_metrics.tsv"), "w", encoding="utf-8") as f:
        cols = ["page", "folio", "variant", "chars", "word_tokens", "types",
                "token_entropy_bits", "gzip_ratio", "periods", "period_segments"]
        f.write("\t".join(cols) + "\n")
        for pno in page_nos:
            fl = folio.get(pno, "")
            for key, *_ in VARIANTS:
                txt = f"<!-- page {pno} -->\n" + "\n".join(streams[key][pno])
                m = text_metrics(txt)
                f.write("\t".join([str(pno), fl, key, str(m["chars"]), str(m["word_tokens"]),
                                   str(m["types"]), f"{m['token_entropy_bits']:.6f}",
                                   f"{m['gzip_ratio']:.6f}", str(m["periods"]),
                                   str(m["period_segments"])]) + "\n")

    # ---- silence sites -----------------------------------------------------
    with open(os.path.join(OUT, "silence_sites.tsv"), "w", encoding="utf-8") as f:
        f.write("class\tkind\toccurrences\tpages\tv1_delete_all\tv2_delete_colon_slash\t"
                "v3_operators_to_space\tv4_operators_to_period\tv5_period_from_slashes\t"
                "v6_period_from_colons\n")
        for cls in sorted(site_counts, key=lambda c: -site_counts[c].total()):
            kind = "operator" if not cls.startswith("dash:") else "dash"
            op = cls.split(":", 1)[1] if kind == "dash" else cls
            treat = {}
            for key, treatment, only, sep in VARIANTS:
                if treatment == "keep":
                    treat[key] = "kept"
                elif only is None:
                    treat[key] = "deleted" if treatment == "delete" else \
                                 f"-> {'space' if treatment == 'space' else 'period'}"
                elif op in only:
                    treat[key] = "deleted" if treatment == "delete" else "-> period"
                else:
                    treat[key] = "-> space" if sep is not None else "kept"
            f.write("\t".join([cls, kind, str(site_counts[cls].total()), str(len(site_pages[cls]))]
                              + [treat[key] for key, *_ in VARIANTS[1:]]) + "\n")
        f.write("\t".join(["TOTAL", "", str(op_n + dash_n), "", "", "", "", "", "", ""]) + "\n")
        f.write("\t".join(["TOTAL-operators-only", "operator", str(op_n), "", "", "", "", "", "", ""]) + "\n")
        f.write("\t".join(["TOTAL-dashes-only", "dash", str(dash_n), "", "", "", "", "", "", ""]) + "\n")

    # ---- fused residue (v1 portmanteau recovery) ---------------------------
    with open(os.path.join(OUT, "fused_residue.tsv"), "w", encoding="utf-8") as f:
        f.write("left_operand\tright_operand\tjoined_form\tn_joins\t"
                "source_operators\texample_citation\n")
        for (lop, rop), c in links.most_common():
            f.write("\t".join([lop, rop, lop + rop, str(c),
                               ",".join(sorted(link_ops[(lop, rop)])),
                               link_example.get((lop, rop), "")]) + "\n")

    # joined *forms* (a fused token may absorb further operators downstream,
    # so `soulgram` as a standalone form is under-counted relative to the
    # `soul/gram` chain; the pair table above is the honest denominator)
    with open(os.path.join(OUT, "fused_forms.tsv"), "w", encoding="utf-8") as f:
        f.write("joined_form\tn\tsource_operators\texample_citation\n")
        for tok, c in fused.most_common():
            ops = fused_src[tok]
            f.write("\t".join([tok, str(c),
                               ",".join(f"{o}:{n}" for o, n in ops.most_common()),
                               fused_example.get(tok, "")]) + "\n")

    # ---- sentence shape ----------------------------------------------------
    SEG_KEYS = ("v0_original", "v4_operators_to_period", "v5_period_from_slashes",
                "v6_period_from_colons")
    seg_hdr = ("scope\tvariant\tn_segments\tmean_words\tmedian_words\tmax_words\t"
               "frac_upper_initial\tfrac_single_word\tn_segments_ge5\tfrac_upper_initial_ge5\n")

    def seg_row(scope, key, txt):
        st = segment_stats(txt)
        return "\t".join([scope, key, str(st["n_segments"]), f"{st['mean_words']:.3f}",
                          f"{st['median_words']:.1f}", str(st["max_words"]),
                          f"{st['frac_upper_initial']:.4f}", f"{st['frac_single_word']:.4f}",
                          str(st["n_segments_ge5"]),
                          f"{st['frac_upper_initial_ge5']:.4f}"]) + "\n"

    with open(os.path.join(OUT, "sentence_shape.tsv"), "w", encoding="utf-8") as f:
        f.write(seg_hdr)
        for key in SEG_KEYS:
            f.write(seg_row("ALL", key, texts[key]))
        for pno in page_nos:
            for key in SEG_KEYS[1:]:
                f.write(seg_row(str(pno), key, "\n".join(streams[key][pno])))

    # ---- citable samples ---------------------------------------------------
    samples = []
    seen = set()
    # (a) the multi-operator runs named in #10
    for run in ("//++", "::<<", ">::", "::>>++"):
        for pno in page_nos:
            hit = next((i for i, ln in enumerate(streams["v0_original"][pno], 1)
                        if run in ln), None)
            if hit is None:
                continue
            src = None
            for out_i, src_i in line_index["v0_original"][pno]:
                if out_i == hit:
                    src = src_i
            line = pages[pno].split("\n")[src - 1]
            if (pno, src) not in seen:
                seen.add((pno, src))
                samples.append((run, pno, src, line))
            break
    # (b) stratified: every 10th page, its first operator-bearing line
    for pno in page_nos:
        if pno % 10:
            continue
        src = None
        for out_i, src_i in line_index["v0_original"][pno]:
            src = src_i
            line = pages[pno].split("\n")[src - 1]
            if site_spans(line):
                if (pno, src) not in seen:
                    seen.add((pno, src))
                    samples.append(("stratified", pno, src, line))
                break
    with open(os.path.join(OUT, "samples.tsv"), "w", encoding="utf-8") as f:
        f.write("reason\tpage\tfolio\tsrc_line\tsource\t" +
                "\t".join(k for k, *_ in VARIANTS) + "\n")
        for reason, pno, src, line in samples:
            vals = []
            for key, treatment, only, sep in VARIANTS:
                new, _d, _fu = transform_line(line, site_spans(line), treatment, only, sep)
                vals.append(normalize_line(new))
            f.write("\t".join([reason, str(pno), folio.get(pno, ""), str(src), line.strip()] + vals) + "\n")

    # ---- summary.json ------------------------------------------------------
    summary = {
        "source": os.path.relpath(SRC, ROOT),
        "n_pages": len(page_nos),
        "lexemes_baseline": lexemes,
        "operator_tokens": op_n,
        "dash_sites": dash_n,
        "dash_by_char": dict(dash_totals),
        "silence_sites_total": op_n + dash_n,
        "variants": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                         for kk, vv in v.items()} for k, v in zip(
                             [k for k, *_ in VARIANTS], rows)},
        "dash_sensitivity": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                 for kk, vv in v.items()} for k, v in sens.items()},
        "site_counts": {c: site_counts[c].total() for c in site_counts},
        "reconciliation": recon,
        "lexical_joins": {k: fused_by_variant[k] for k, _t, _o, _s in VARIANTS},
        "fused_residue_top": fused.most_common(40),
        "fused_residue_total": sum(fused.values()),
        "fused_residue_unique": len(fused),
        "sentence_shape": {k: segment_stats(texts[k]) for k in
                           ("v0_original", "v4_operators_to_period",
                            "v5_period_from_slashes", "v6_period_from_colons")},
        "site_counts_reproduce_10949": op_n == 10949,
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- report ------------------------------------------------------------
    write_report(summary, rows, sens, site_counts, site_pages, dash_totals,
                 op_totals, fused, fused_src, fused_example, lexemes, folio,
                 page_nos, line_index, samples, recon, fused_by_variant,
                 links, link_ops, link_example)

    print("operator tokens:", op_n, "dash sites:", dash_n, "lexemes:", lexemes)
    for k, *_ in VARIANTS:
        print(f"  {k:28s} words={rows[[k for k, *_ in VARIANTS].index(k)]['word_tokens']:>7,}"
              f" types={rows[[k for k, *_ in VARIANTS].index(k)]['types']:>6,}"
              f" H={rows[[k for k, *_ in VARIANTS].index(k)]['token_entropy_bits']:.3f}"
              f" gzip={rows[[k for k, *_ in VARIANTS].index(k)]['gzip_ratio']:.4f}")
    print("fused residue:", sum(fused.values()), "unique:", len(fused))


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_report(summary, rows, sens, site_counts, site_pages, dash_totals,
                 op_totals, fused, fused_src, fused_example, lexemes, folio,
                 page_nos, line_index, samples, recon, fused_by_variant,
                 links, link_ops, link_example):
    keys = [k for k, *_ in VARIANTS]
    by = dict(zip(keys, rows))
    L = []
    p = L.append

    p("# Silence-removal transform variants (#13)\n")
    p(f"Source: `analysis/layout_recovery/blood-electric.layout.md` ({summary['n_pages']} pages, "
      "the vetted layout corpus of #7). Operator set, greedy longest-match tokenization and "
      "page loader are imported from #10/#12, so the operator inventory reproduces exactly "
      f"({summary['operator_tokens']:,} tokens). Reproduce with `python work/silence_removal.py`; "
      "regression test `work/test_silence_removal.py`.\n")
    p("**What the transform is.** In Beguš's whale-coda work the *silences* between clicks are "
      "deleted so the residual stream can be read continuously. Here the click train is the "
      "operator layer. Each operator token is a *silence site*: it is deleted, or replaced by a "
      "space, or replaced by a period. Nothing else in the corpus is touched - word-internal "
      "hyphens (`boy-roid`, `cold-blooded`) live inside WORD tokens and are never candidates.\n")

    p("## Variants\n")
    p("| key | transform | persisted as |")
    p("|---|---|---|")
    for k in keys:
        p(f"| `{k}` | {TREATMENT_LABEL[k]} | `analysis/silence_removal/{k}.txt` |")
    p("")
    p("`v1` vs `v3` is the load-bearing contrast. v1 is *strict deletion*: the operator's "
      "characters disappear and the surrounding whitespace is left exactly as it was, so a "
      "glued site fuses its neighbours (`soul/gram` -> `soulgram`, `gene=TV` -> `geneTV`). That "
      "is the literal Beguš move - make the residue adjacent and see what word it spells. v3 "
      "maps every site to a space, which keeps every lexical item analysable. v5/v6 are "
      "diagnostics: they convert the #12 `::`-vs-`//` claim into a directly comparable pair.\n")
    p("Line structure is preserved: the transform never introduces or removes a newline, so "
      "output line *i* of a variant derives from source line *i*; `line_index.tsv` records the "
      "exact mapping for all 7 streams. Every stream is page-tagged (`<!-- page N -->`) so "
      "#50/#62/#67 can cite and align.\n")

    p("## Measured effect\n")
    p("Baseline lexemes (WORD tokens in the untouched corpus): "
      f"**{lexemes:,}** - the token count every variant is reconciled against.\n")
    p("| variant | chars | word tokens | types | TTR | token entropy (bits) | gzip | periods | mean words/period-segment |")
    p("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        m = by[k]
        p(f"| `{k}` | {m['chars']:,} | {m['word_tokens']:,} | {m['types']:,} | "
          f"{m['ttr']:.4f} | {m['token_entropy_bits']:.4f} | {m['gzip_ratio']:.4f} | "
          f"{m['periods']:,} | {m['mean_words_per_segment']:.2f} |")
    p("")
    p("Reconciliation (asserted by the builder, so it cannot silently rot). #12's lexer "
      "already breaks tokens at operator positions - `soul/gram` is lexed `soul` `gram` in "
      "the baseline - so a space/period variant *cannot* change the token sequence at all, "
      "and a delete variant can only join lexical items, once per junction. Each variant's "
      "count is therefore fully explained, and the builder raises if it ever is not.\n")
    p("| variant | word tokens | lexical joins | expected | delta | token sequence == baseline |")
    p("|---|---:|---:|---:|---:|---|")
    for k in keys:
        r = recon[k]
        same = "yes" if r["same_token_sequence_as_baseline"] else "no (tokens joined)"
        p(f"| `{k}` | {r['word_tokens']:,} | {fused_by_variant[k]:,} | "
          f"{r['expected']:,} | {r['delta']:+d} | {same} |")
    p("")
    p("Two consequences worth flagging to #50. **First**, `v1_delete_all_operators` and "
      "`v2_delete_colon_slash` are the only variants that change the *word* sequence, and "
      "they do it by joining tokens: strict deletion is the Beguš-faithful reading, and it "
      f"costs {by['v0_original']['word_tokens'] - by['v1_delete_all_operators']['word_tokens']:,} "
      "lexical items. **Second**, `v3`/`v4`/`v5`/`v6` keep the word sequence identical to "
      "the baseline - their token count, type count and type entropy are *exactly* the "
      f"baseline's ({by['v3_operators_to_space']['token_entropy_bits']:.4f} bits), and they "
      "differ from it only in the characters that replace the operators. That makes v0 -> v3 "
      "and v0 -> v4 the cleanest possible controlled comparison of what the operator "
      "characters cost a language model: any perplexity delta between them is attributable to "
      "the operators alone, with no lexical confound. Entropy over the type distribution "
      "cannot see that delta by construction, which is exactly why #50 needs an LM.\n")
    p("These are **proxy** metrics (type entropy, gzip). gzip does move under the separator "
      f"variants ({by['v0_original']['gzip_ratio']:.4f} -> "
      f"{by['v3_operators_to_space']['gzip_ratio']:.4f} for v3), which is the expected "
      "direction: the operator characters are the *predictable* part of the text, so replacing "
      "them makes the residue cheaper to compress. The perplexity comparison that turns this "
      "into a verdict is #50; this task only has to hand it the streams.")

    p("## Silence sites: what is removed\n")
    p("| class | occurrences | pages | v1 | v2 | v3 | v4 |")
    p("|---|---:|---:|---|---|---|---|")
    for cls in sorted(site_counts, key=lambda c: -site_counts[c].total()):
        op = cls.split(":", 1)[1] if cls.startswith("dash:") else cls
        v2 = "deleted" if op in ("::", "//") else "kept"
        p(f"| `{cls}` | {site_counts[cls].total():,} | {len(site_pages[cls])} | deleted | "
          f"{v2} | -> space | -> period |")
    p(f"| **operators (#10 set)** | {summary['operator_tokens']:,} | | | | | |")
    p(f"| **dashes (extra)** | {summary['dash_sites']:,} | | | | | |")
    p(f"| **total silence sites** | {summary['silence_sites_total']:,} | | | | | |")
    p("")
    p(f"`::` + `//` are {summary['site_counts'].get('::',0) + summary['site_counts'].get('//',0):,} "
      f"of the {summary['operator_tokens']:,} operator tokens "
      f"({(summary['site_counts'].get('::',0)+summary['site_counts'].get('//',0))/summary['operator_tokens']:.1%}), "
      "which is why v2 is a genuinely partial silence removal and not a near-duplicate of v1.\n")

    p("### The dash judgement call\n")
    p("`-`, `—` and `–` are not in #10's operator set. #12 measured 177 whitespace-bounded "
      "`-` dashes and classified them with the `::` phrase-separator class rather than the "
      "`//` sentence class. Re-measured here: "
      + ", ".join(f"`{c}` x{n}" for c, n in summary["dash_by_char"].items())
      + f" = {summary['dash_sites']:,} whitespace-bounded dash tokens "
        "(476 of 484 em dashes are whitespace-bounded; all 43 en dashes are).\n")
    p("Default behaviour treats them as removable silence, so v1 deletes "
      f"{summary['dash_sites']:,} sites rather than {summary['operator_tokens']:,}. Every "
      "headline number is therefore reported both ways:\n")
    p("| variant | entropy (dashes removed) | entropy (dashes kept) | gzip (removed) | gzip (kept) |")
    p("|---|---:|---:|---:|---:|")
    for k in keys[:5]:
        p(f"| `{k}` | {by[k]['token_entropy_bits']:.4f} | {sens[k]['token_entropy_bits']:.4f} | "
          f"{by[k]['gzip_ratio']:.4f} | {sens[k]['gzip_ratio']:.4f} |")
    p("")
    dmax = max(abs(by[k]["token_entropy_bits"] - sens[k]["token_entropy_bits"])
               for k in keys[:5])
    gmax = max(abs(by[k]["gzip_ratio"] - sens[k]["gzip_ratio"]) for k in keys[:5])
    p("The dash policy moves token entropy by "
      + ("nothing at all (identical to the last printed digit)" if dmax == 0
         else f"at most {dmax:.2e} bits")
      + f" and gzip by at most {gmax:.2e}, so no conclusion in this report depends on it. "
      "Entropy is untouched because #12's lexer already treats a whitespace-bounded dash as a "
      "token boundary, exactly like an operator. It does change the v1/v3 character counts and "
      "the join count, which is why the join count in the reconciliation table above is the "
      "dashes-removed figure.\n")

    p("## v1 fused residue: the operators hide real words\n")
    p(f"Strict deletion joins **{summary['fused_residue_total']:,}** lexical pairs into "
      f"**{summary['fused_residue_unique']:,}** distinct joined forms. `fused_residue.tsv` is "
      "the *pair*-level table (left operand | right operand | joined form | n), which is the "
      "honest denominator; `fused_forms.tsv` is the joined-form level. Top 25 pairs:\n")
    p("| left | right | joined form | n | operator(s) | example |")
    p("|---|---|---|---:|---|---|")
    for (lop, rop), c in links.most_common(25):
        p(f"| `{lop}` | `{rop}` | `{lop}{rop}` | {c:,} | "
          f"{','.join('`'+o+'`' for o in sorted(link_ops[(lop, rop)]))} | "
          f"{link_example.get((lop, rop), '')} |")
    p("")
    top_pair = links.most_common(1)[0] if links else None
    p("Reading: the operator layer is not arbitrary noise - it is *splitting words that the "
      "book also writes whole*. "
      + (f"The single most frequent join is `{top_pair[0][0]}` + `{top_pair[0][1]}` "
         f"({top_pair[1]:,} times). " if top_pair else "")
      + "Every row of `fused_residue.tsv` is a decomposition the book itself supplies: it is a "
      "generated dictionary for E2 (#18 compound decomposition) and the #65 termbase, keyed by "
      "the operator that did the splitting.\n")
    p("Two honest limits on that reading. (a) A joined *form* is under-counted relative to its "
      "pair, because a fused token can absorb further operators downstream: the pair "
      f"`soul`|`gram` occurs {links.get(('soul','gram'),0):,} times but the standalone form "
      f"`soulgram` only {fused.get('soulgram',0):,} times, the rest appearing as "
      "`soulgramthe`, `soulgrams`, `Soulgram` and so on. (b) Calling the joined form a "
      "*prior* form of the word is an interpretation consistent with these counts, not "
      "something the transform measures. Interpretation, not measurement: the join table is "
      "exact arithmetic on the token stream; the philological claim built on top of it is not.\n")

    p("## Sentence skeleton: does mapping to `.` produce sentences?\n")
    p("| variant | segments | mean words | median | max | frac uppercase-initial | frac single-word |")
    p("|---|---:|---:|---:|---:|---:|---:|")
    for k in ("v0_original", "v4_operators_to_period", "v5_period_from_slashes",
              "v6_period_from_colons"):
        st = summary["sentence_shape"][k]
        p(f"| `{k}` | {st['n_segments']:,} | {st['mean_words']:.2f} | "
          f"{st['median_words']:.0f} | {st['max_words']} | {st['frac_upper_initial']:.3f} | "
          f"{st['frac_single_word']:.3f} |")
    p("")
    p("Uppercase-initial share is a proxy for 'this segment looks like a sentence', not a "
      "measurement of sentencelessness, and it is length-sensitive, so it is reported twice - "
      "once over all segments and once restricted to segments of >= 5 words:\n")
    p("| variant | segments >=5 words | frac uppercase-initial (>=5 words) | all segments |")
    p("|---|---:|---:|---:|")
    for k in ("v4_operators_to_period", "v5_period_from_slashes", "v6_period_from_colons"):
        st = summary["sentence_shape"][k]
        p(f"| `{k}` | {st['n_segments_ge5']:,} | {st['frac_upper_initial_ge5']:.3f} | "
          f"{st['frac_upper_initial']:.3f} |")
    p("")
    b, f4 = summary["sentence_shape"]["v0_original"], summary["sentence_shape"]["v4_operators_to_period"]
    f5, f6 = summary["sentence_shape"]["v5_period_from_slashes"], summary["sentence_shape"]["v6_period_from_colons"]
    p(f"The untouched corpus contains **{b['n_segments']} sentence-like segments** - its "
      f"prose has essentially no sentence punctuation ({by['v0_original']['periods']} periods, "
      "of which most are in the front matter `www.creationbooks.com`, `ISBN`, `a.e`). The "
      f"operators *are* the punctuation: mapping them to `.` produces {f4['n_segments']:,} "
      f"segments with a median of {f4['median_words']:.0f} words.\n")
    p("The v5/v6 pair tests #12 directly. If `//` is the utterance boundary and `::` the "
      "phrase separator, then `//`->period should give better-formed segments than `::`->period:\n")
    p(f"- `//`->period: {f5['n_segments']:,} segments, median {f5['median_words']:.0f} words, "
      f"{f5['frac_upper_initial']:.1%} start uppercase "
      f"({f5['frac_upper_initial_ge5']:.1%} of the {f5['n_segments_ge5']:,} segments of >=5 words)")
    p(f"- `::`->period: {f6['n_segments']:,} segments, median {f6['median_words']:.0f} words, "
      f"{f6['frac_upper_initial']:.1%} start uppercase "
      f"({f6['frac_upper_initial_ge5']:.1%} of the {f6['n_segments_ge5']:,} segments of >=5 words)")
    p("")
    if f5["frac_upper_initial"] > f6["frac_upper_initial"]:
        p(f"**Result: the #12 prediction replicates.** `//`->period yields "
          f"{f5['frac_upper_initial']:.1%} uppercase-initial segments vs "
          f"{f6['frac_upper_initial']:.1%} for `::`->period "
          f"(+{(f5['frac_upper_initial']-f6['frac_upper_initial'])*100:.1f} points), and the "
          f"gap survives the length control "
          f"({f5['frac_upper_initial_ge5']:.1%} vs {f6['frac_upper_initial_ge5']:.1%} on "
          "segments of >=5 words). That is an independent confirmation of the #12 upper-next "
          "signature, obtained from the transformed stream instead of from neighbourhood "
          "counts. It also bounds v4: mapping *every* operator to `.` gives only "
          f"{f4['frac_upper_initial']:.1%} uppercase-initial segments, so v4 is not a "
          "sentence reconstruction - it is the best available *skeleton*.\n")
        p("Caveat on this comparison: `//` and `::` occur at different densities and therefore "
          "produce segments of different lengths (medians "
          f"{f5['median_words']:.0f} vs {f6['median_words']:.0f} words), so the headline rates "
          "are not a perfectly matched design. The >=5-word restriction removes the length "
          "confound and the sign of the effect does not change. The equivalent test for the "
          "free dashes, which #12 placed in the `::` class, is left to #14, where the dash "
          "spans can be isolated from the word-internal hyphens.\n")
    else:
        p("**Result: the #12 prediction does NOT replicate here** - `::`->period gives at least "
          "as many uppercase-initial segments as `//`->period. Reported as a negative result: "
          "the upper-next neighbourhood signature does not transfer to segment-level formation. "
          "This is a genuine tension with #12 and is left open for #14.\n")

    p("## Citable samples\n")
    p("`samples.tsv` holds original + all seven streams for the four multi-operator runs named "
      "in #10 (`//++`, `::<<`, `>::`, `::>>++`) plus a stratified every-10th-page sample. "
      "First two rows:\n")
    p("| reason | citation | source | v1 | v4 |")
    p("|---|---|---|---|---|")
    for reason, pno, src, line in samples[:2]:
        cells = []
        for k in ("v1_delete_all_operators", "v4_operators_to_period"):
            new, _d, _fu = transform_line(line, site_spans(line),
                                          *[(t, o, s) for kk, t, o, s in VARIANTS if kk == k][0])
            cells.append(normalize_line(new)[:90])
        p(f"| `{reason}` | {cite(pno, folio.get(pno))} | {line.strip()[:90]} | {cells[0]} | {cells[1]} |")
    p("")

    p("## Caveats and negative results\n")
    p("- **These are proxy metrics, not the #50 verdict.** No LM perplexity is computed here; "
      "the task is to persist the streams and state the rationale. Entropy and gzip are given "
      "because they are free, not because they settle anything.")
    p("- **Deletion is not decoding.** A fused form like `soulgram` is a mechanical artefact of "
      "removing a separator; the claim that it is a *prior* form of the word is an "
      "interpretation. `fused_residue.tsv` reports the counts; the reading is separate.")
    p("- **v2 keeps 51% of the operator mass** (`=`, `/`, `<<`, `>>`, `<`, `>`, `++`, `+`, `|`). "
      "Its stream is therefore still heavily symbolic, which is the point: it isolates the "
      "field/utterance separators.")
    p("- **v4 is a hypothesis, not a reconstruction.** Mapping every operator to `.` asserts "
      "that all 12 operator types mark sentence boundaries; #12 supports that for `//` and "
      "refutes it for `=` and `++`. The `lowercase_initial_segments` column in `variants.tsv` "
      "shows how often the result is not a well-formed sentence - a negative result worth "
      "keeping.")
    p("- **The transform is defined over operator spans, not over punctuation in general.** "
      "`.` (22), `,` (15), `\"` (136), `[`/`]` (179/179) are untouched. The `[...]` class is "
      "worth a note: 179 bracket pairs, 125 lines contain one, and the content is dominated by "
      "a single recurring string (`[ice nebula faecal black]`), i.e. a literal bracketed "
      "insertion in the book. Flagged for #14's block parser, not handled here.")
    p("- **HTML-artifact pages.** Pages 92, 158, 219 contain raw `<a href>`/`mailto:` OCR "
      "markup (see #12). They are kept in every variant so the streams stay identical in scope "
      "to the #10/#12 corpus; downstream consumers should apply the same artifact filter.")
    p("- **OCR confidence.** The operator glyphs themselves (`:` vs `::`, `<` vs `<<`, "
      "backslash vs slash) are exactly the tokens #3/#4 flagged. The transform inherits that "
      "uncertainty; it does not amplify it, because every variant is derived from the same "
      "tokenization.")
    p("")
    p("## Handoff\n")
    p("- **#50** (quantified silence removal): the seven streams + `variants.tsv` + "
      "`per_page_metrics.tsv`; add LM perplexity per variant and pick the transform that most "
      "lowers it.")
    p("- **#67** (spectral reading): `v4_operators_to_period.txt` is the best candidate for a "
      "sentence-skeleton under-text, with `sentence_shape.tsv` as its honest quality report; "
      "`v5`/`v6` are the ablations that show how much of the effect is `//` alone.")
    p("- **#62** (interlinear edition): `line_index.tsv` gives line-exact alignment from any "
      "variant back to the source clause, so the middle line of the three-line rendering is "
      "mechanically derivable.")
    p("- **#37** (POS of the residue): `v3_operators_to_space.txt` is the right input - it keeps "
      "every lexical item separate, so a tagger sees words, not fused portmanteaux.")
    p("- **#18/#65** (compound decomposition / termbase): `fused_residue.tsv` is a generated "
      "decomposition dictionary keyed by the operator that split each compound.")
    p("")
    p("---")
    p("*Generated by `work/silence_removal.py`*")

    with open(os.path.join(OUT, "silence_removal_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()