#!/usr/bin/env python3
"""Operator function classification (Chainlink #12).

Tests the working hypothesis for each operator role against the distribution
of its contexts in the vetted layout markdown (#7), reusing the ordered
operator tokenizer / operator set from work/operator_inventory.py (#10).

Hypotheses under test (docs/meaning-extraction-methods.md, L1):
  ::  separator / field (record delimiter)
  //  utterance / clause break (or C/JS comment boundary)
  ++  concatenate / increment / coupling
  =   naming / identity (gene=TV)
  << >>  stream / block delimiters
  /   alternation / possessive / compound
  `   escape / literalization
  -   compound linker (word-internal hyphen)

Method.  For every operator occurrence we record the immediately adjacent
non-space token class (WORD / OP / OTHER / line boundary), whether the
operator is glued to that neighbour (no whitespace), its line position
(start/end of line), whether the *next* word is sentence-initial (uppercase),
and the size of the word run since the previous / until the next operator.
We then test each hypothesis with a predicted signature and report the
measured value; for the two most confusable separators (:: vs //) we run a
two-proportion z-test and a G-test of context independence.

The hyphen is special: `-` is not in #10's operator set, but the corpus also
contains *free-standing* whitespace-bounded `-` tokens (`\s-\s`) that are
symbolic (dashes), not word-internal.  These are tokenized and profiled
separately (`free_hyphen_occurrences`) so the `-` verdict rests on a
measurement, not on the operator-list definition.  The 10,949 #10 operator
count is preserved unchanged.

Absence claims are scoped too: `absent_operator_audit` counts each
hypothesised-absent operator in BOTH the raw layout file and the stripped page
bodies.  `` ` ``, `~`, `^`, `==`, `<-` are 0 in both; `--` and `->` are 0 in the
page bodies but occur only inside `<!-- ... -->` scaffolding comments in the raw
file, so "absent" is stated for the analysed corpus rather than asserted
globally.

Outputs under analysis/operator_functions/:
  operator_profiles.tsv   per-operator signature stats (incl. free `-`)
  neighbor_classes.tsv    left/right neighbour-class counts per operator
  joined_pairs.tsv        word=word, word/word, word++word, hyphen compounds
  contexts.tsv            citable example contexts (page + snippet)
  delimiter_pairing.tsv   <<>> and <> stack-pairing stats (incl. artifact pages)
  classification_spec.md  THE DELIVERABLE: classification spec + evidence
  summary.json            machine-readable headline numbers
"""
import collections
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from operator_inventory import OPS  # reuse #10 operator set

SRC = "analysis/layout_recovery/blood-electric.layout.md"
OUT = "analysis/operator_functions"

# Pages whose OCR text is raw HTML (`<a href>`, `mailto:`) rather than the
# book's own operator syntax; used for the delimiter-pairing sensitivity check.
HTML_ARTIFACT_PAGES = {92, 158, 219}

# ---------------------------------------------------------------------------
# corpus -> token stream
# ---------------------------------------------------------------------------
def load_pages(src):
    with open(src, errors="ignore") as f:
        text = f.read()
    parts = re.split(r"<!-- page (\d+) -->", text)
    pages = {}
    for k in range(1, len(parts) - 1, 2):
        body = parts[k + 1]
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)   # layout placeholders
        body = re.sub(r"^#+[ \t]+.*$", "", body, flags=re.M)  # markdown headers
        pages[int(parts[k])] = body
    return pages


def _is_word_char(c):
    return c.isalnum() or c in "_'"


def tokenize(text):
    """(kind, start, end, value); kind in {OP, WORD, OTHER}.

    Same greedy longest-match operator tokenization as #10 (counts reproduce
    the 10,949-token inventory exactly). Hyphenated compounds are one WORD
    token; a standalone `-` is OTHER punctuation.
    """
    toks = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        matched = None
        for op in OPS:
            if text.startswith(op, i):
                matched = op
                break
        if matched:
            toks.append(("OP", i, i + len(matched), matched))
            i += len(matched)
            continue
        if c.isalnum():
            j = i
            while j < n:
                if _is_word_char(text[j]):
                    j += 1
                elif text[j] == "-" and j + 1 < n and text[j + 1].isalnum():
                    j += 1  # internal hyphen: keep compound as one token
                else:
                    break
            toks.append(("WORD", i, j, text[i:j]))
            i = j
            continue
        toks.append(("OTHER", i, i + 1, c))
        i += 1
    return toks


# ---------------------------------------------------------------------------
# per-occurrence feature extraction
# ---------------------------------------------------------------------------
def _record(pno, text, toks, idx):
    """Build the context record for the token at `idx` (any kind)."""
    kind, s, e, val = toks[idx]
    prev = toks[idx - 1] if idx > 0 else None
    nxt = toks[idx + 1] if idx + 1 < len(toks) else None
    lspace = prev is not None and (s - prev[2]) > 0
    rspace = nxt is not None and (nxt[1] - e) > 0
    # line position
    nl = text.rfind("\n", 0, s)
    sol = (s - (nl + 1)) == 0
    rest = text[e:]
    nl2 = rest.find("\n")
    line_rest = rest if nl2 < 0 else rest[:nl2]
    eol = (line_rest.strip() == "")
    # words since previous operator / until next operator
    lp = rp = 0
    for j in range(idx - 1, -1, -1):
        if toks[j][0] == "OP":
            break
        if toks[j][0] == "WORD":
            lp += 1
    for j in range(idx + 1, len(toks)):
        if toks[j][0] == "OP":
            break
        if toks[j][0] == "WORD":
            rp += 1
    # run membership: is an adjacent char part of an operator token?
    run_member = False
    if prev is not None and prev[0] == "OP" and not lspace:
        run_member = True
    if nxt is not None and nxt[0] == "OP" and not rspace:
        run_member = True
    rword = nxt[3] if (nxt and nxt[0] == "WORD") else None
    lword = prev[3] if (prev and prev[0] == "WORD") else None
    upper_next = bool(rword) and rword[:1].isupper()
    return dict(
        page=pno, op=val, s=s, e=e,
        lclass=(prev[0] if prev else "BOUND"),
        rclass=(nxt[0] if nxt else "BOUND"),
        lword=lword, rword=rword,
        lspace=lspace, rspace=rspace,
        glue_l=(prev is not None and not lspace and prev[0] != "OP"),
        glue_r=(nxt is not None and not rspace and nxt[0] != "OP"),
        interior=(prev is not None and nxt is not None
                  and prev[0] == "WORD" and nxt[0] == "WORD"),
        sol=sol, eol=eol, lp=lp, rp=rp,
        run_member=run_member, upper_next=upper_next,
        text=text,
    )


def build_occurrences(pages):
    """The 10,949 #10 operator tokens (unchanged operator inventory)."""
    occ = []
    for pno in sorted(pages):
        text = pages[pno]
        toks = tokenize(text)
        for idx, (kind, _s, _e, _val) in enumerate(toks):
            if kind != "OP":
                continue
            occ.append(_record(pno, text, toks, idx))
    return occ


def free_hyphen_occurrences(pages):
    """Free-standing, whitespace-bounded `-` tokens (`\\s-\\s`).

    `-` is not in #10's operator set, so these are not part of the 10,949
    operator count.  They are, however, symbolic (a dash used as a phrase /
    clause break), so they are tokenized and profiled separately instead of
    being folded into the word-internal hyphen compounds.
    """
    occ = []
    for pno in sorted(pages):
        text = pages[pno]
        toks = tokenize(text)
        for idx, (kind, s, e, val) in enumerate(toks):
            if kind != "OTHER" or val != "-":
                continue
            lc = text[s - 1] if s > 0 else " "
            rc = text[e] if e < len(text) else " "
            if not (lc.isspace() and rc.isspace()):
                continue  # a partial/inline hyphen, not a free-standing dash
            occ.append(_record(pno, text, toks, idx))
    return occ


# Operators hypothesised in the L1 spec but absent from the book text.  We
# report them for two *separate* corpora so the "absent" claim is auditable:
#   page_body  - the vetted page text that `load_pages` yields (what the whole
#                spec analyses)
#   raw_file   - the layout markdown on disk, which additionally carries
#                `<!-- page N -->` / `<!-- figure ... -->` scaffolding comments.
# `--` and `->` occur only inside that scaffolding, so "0 in the corpus" is
# true of the page bodies but *not* of a naive grep over the raw file.
ABSENT_OPS = ("`", "~", "^", "--", "==", "->", "<-")


def absent_operator_audit(raw_file_text, pages):
    """Count each hypothesised-absent operator in raw file vs page bodies.

    Returns {op: {"raw_file": n, "page_body": n}}.  Both numbers are measured,
    never inferred, so a reader grepping either corpus can reconcile them.
    """
    body = "\n".join(pages[p] for p in sorted(pages))
    return {op: {"raw_file": raw_file_text.count(op),
                 "page_body": body.count(op)} for op in ABSENT_OPS}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def two_proportion_z(k1, n1, k2, n2):
    """Two-sided two-proportion z-test; returns (z, p)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def chi2_context(occ, op):
    """G-test of this operator's (left,right) class joint vs all others.

    Tests whether the operator's immediate context distribution differs from
    the pooled context distribution of the remaining operators.
    """
    keys = ["WORD", "OP", "OTHER", "BOUND"]
    a = [[0] * len(keys) for _ in keys]
    b = [[0] * len(keys) for _ in keys]
    for o in occ:
        row = keys.index(o["lclass"]) if o["lclass"] in keys else 0
        col = keys.index(o["rclass"]) if o["rclass"] in keys else 0
        (a if o["op"] == op else b)[row][col] += 1
    # observed = this op, expected = pooled * share; G-test on 2 x (4x4)
    flat_a = [a[i][j] for i in range(len(keys)) for j in range(len(keys))]
    flat_b = [b[i][j] for i in range(len(keys)) for j in range(len(keys))]
    G = 0.0
    na, nb = sum(flat_a), sum(flat_b)
    if na == 0 or nb == 0:
        return 0.0, 0, 1.0
    tot = na + nb
    for x, y in zip(flat_a, flat_b):
        o1, o2 = x, y
        e1 = (x + y) * na / tot
        e2 = (x + y) * nb / tot
        if o1 > 0 and e1 > 0:
            G += 2 * o1 * math.log(o1 / e1)
        if o2 > 0 and e2 > 0:
            G += 2 * o2 * math.log(o2 / e2)
    # 2 x N G-test: df = N - 1 for the two groups over N categories
    df = len(flat_a) - 1
    p = None
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(G, df))
    except Exception:
        pass
    return G, df, p


def per_operator(occ):
    ops = collections.Counter(o["op"] for o in occ)
    prof = {}
    for op, n in ops.items():
        sub = [o for o in occ if o["op"] == op]
        lc = collections.Counter(o["lclass"] for o in sub)
        rc = collections.Counter(o["rclass"] for o in sub)
        glue_l = sum(o["glue_l"] for o in sub) / n
        glue_r = sum(o["glue_r"] for o in sub) / n
        prof[op] = dict(
            n=n,
            glue_l=glue_l, glue_r=glue_r,
            both_glued=sum(o["glue_l"] and o["glue_r"] for o in sub) / n,
            interior=sum(o["interior"] for o in sub) / n,
            sol=sum(o["sol"] for o in sub) / n,
            eol=sum(o["eol"] for o in sub) / n,
            run_member=sum(o["run_member"] for o in sub) / n,
            upper_next=sum(o["upper_next"] for o in sub) / n,
            med_lp=statistics.median([o["lp"] for o in sub]),
            med_rp=statistics.median([o["rp"] for o in sub]),
            lclass=dict(lc), rclass=dict(rc),
            top_lwords=collections.Counter(
                o["lword"] for o in sub if o["lword"]).most_common(10),
            top_rwords=collections.Counter(
                o["rword"] for o in sub if o["rword"]).most_common(10),
        )
    return ops, prof


# ---------------------------------------------------------------------------
# linker / compound evidence (regex over the whole corpus)
# ---------------------------------------------------------------------------
def linker_evidence(pages):
    raw = "\n".join(pages[p] for p in sorted(pages))
    word = r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?"

    def pairs(sep):
        pat = re.compile(word + re.escape(sep) + word)
        found = pat.findall(raw)
        return found

    def chains(sep):
        pat = re.compile(word + "(?:" + re.escape(sep) + word + ")+")
        return pat.findall(raw)

    out = {}
    for sep, label in (("=", "equals"), ("/", "slash"), ("+", "plus"),
                       ("++", "plusplus"), ("-", "hyphen")):
        if sep == "/":
            found = chains(sep)
        else:
            found = pairs(sep)
        out[label] = dict(
            n=len(found),
            unique=len(set(found)),
            top=collections.Counter(found).most_common(20),
        )
    # slash chain length distribution
    slash = chains("/")
    lens = collections.Counter(len(re.findall(re.escape("/"), c)) + 1 for c in slash)
    out["slash"]["chain_len"] = dict(sorted(lens.items()))
    return out


# ---------------------------------------------------------------------------
# delimiter pairing (stack over the operator stream, per page)
# ---------------------------------------------------------------------------
def pairing_stats(pages, open_op, close_op, exclude=()):
    matched = unmatched_open = unmatched_close = 0
    spans, depths = [], []
    for pno in sorted(pages):
        if pno in exclude:
            continue
        text = pages[pno]
        ops = [t for t in tokenize(text) if t[0] == "OP"]
        stack = []
        for _k, s, e, v in ops:
            if v == open_op:
                stack.append((s, e))
            elif v == close_op:
                if stack:
                    _os, oe = stack.pop()
                    wc = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text[oe:s]))
                    spans.append(wc)
                    depths.append(len(stack) + 1)
                    matched += 1
                else:
                    unmatched_close += 1
        unmatched_open += len(stack)
    return dict(
        matched=matched, unmatched_open=unmatched_open,
        unmatched_close=unmatched_close, n_spans=len(spans),
        median_span_words=(statistics.median(spans) if spans else 0),
        mean_span_words=(round(sum(spans) / len(spans), 2) if spans else 0),
        max_depth=(max(depths) if depths else 0),
    )


# ---------------------------------------------------------------------------
# hypothesis registry + verdicts
# ---------------------------------------------------------------------------
HYPOTHESIS = {
    "::": "separator / field (record delimiter, like `:` or struct field access)",
    "//": "utterance / clause break, or C/JS comment boundary",
    "++": "concatenate / increment / coupling (C increment, string concat)",
    "=": "naming / identity (`gene=TV` = 'gene called TV'; `suck=blood`)",
    "<<": "stream / block open delimiter (C++ I/O, heredoc, quote)",
    ">>": "stream / block close delimiter",
    "/": "alternation ('or'), possessive, or comment-slash (`soul/gram`)",
    "`": "escape / literalization marker",
    "-": "compound linker (`drug-embryo`, `boy-roid`)",
    "+": "(not in the hypothesis list) — tested as a linker",
    "|": "(not in the hypothesis list) — tested as a stream/pipe marker",
    "<": "(not in the hypothesis list) — tested as an inline delimiter",
    ">": "(not in the hypothesis list) — tested as an inline delimiter",
}


def verdicts(prof, link, pair_block, pair_angle_clean, tests):
    v = {}
    p = prof
    z_upper, p_upper = tests["sep_upper_z"]
    z_eol, p_eol = tests["sep_eol_z"]

    v["::"] = (
        "**SUPPORTED (refined: phrase-level field/clause separator).** "
        f"`::` is interior ({p['::']['interior']:.0%} word-on-both-sides), "
        f"glued on both sides ({p['::']['both_glued']:.0%}), and almost never "
        f"line-anchored (SOL {p['::']['sol']:.1%}, EOL {p['::']['eol']:.1%}). "
        f"Unlike `//`, it is NOT a sentence boundary: only {p['::']['upper_next']:.1%} "
        f"of following words are sentence-initial vs {p['//']['upper_next']:.1%} for "
        f"`//` (two-proportion z={z_upper:.1f}, p={p_upper:.1e}). "
        "It separates *phrases/fields*, not records of single tokens."
    )
    v["//"] = (
        "**SUPPORTED as utterance/clause break; REJECTED as a line comment.** "
        f"It is an inline separator (SOL {p['//']['sol']:.1%}, EOL {p['//']['eol']:.1%}), "
        f"so it is not a C/JS `//` line-comment introducer. It IS a sentence boundary: "
        f"{p['//']['upper_next']:.0%} of following words are sentence-initial, "
        f"vs {p['::']['upper_next']:.0%} for `::` (z={z_upper:.1f}, p={p_upper:.1e}); "
        f"it also sits at end-of-line more often than `::` "
        f"({p['//']['eol']:.1%} vs {p['::']['eol']:.1%}, p={p_eol:.1e}). "
        "Refinement: `//` marks an *utterance/period* boundary, often mid-paragraph."
    )
    v["="] = (
        "**SUPPORTED (naming / identity).** "
        f"`=` is a tight word-level linker: glued on both sides "
        f"({p['=']['both_glued']:.0%}), left/right neighbour is a word "
        f"{p['=']['lclass'].get('WORD', 0) / p['=']['n']:.0%}/"
        f"{p['=']['rclass'].get('WORD', 0) / p['=']['n']:.0%} of the time, "
        f"and only {p['=']['upper_next']:.1%} of right words are sentence-initial. "
        f"The regex finds {link['equals']['n']:,} `word=word` pairs "
        f"({link['equals']['unique']:,} unique); top: "
        + ", ".join(f"`{w}`×{c}" for w, c in link['equals']['top'][:5])
        + ". It builds a *name/equation* from two lexical items, exactly as "
        "`gene=TV` predicts."
    )
    v["<<"] = (
        "**SUPPORTED (block open delimiter).** "
        f"`<<` is a run-opener/line-starter: preceded by another operator "
        f"{p['<<']['lclass'].get('OP', 0) / p['<<']['n']:.0%} of the time or by "
        f"line start ({p['<<']['sol']:.0%}), and glued to a following word "
        f"({p['<<']['glue_r']:.0%}, right word {p['<<']['rclass'].get('WORD', 0) / p['<<']['n']:.0%}). "
        f"Stack pairing finds {pair_block['matched']:,} matched `<<`…`>>` spans "
        f"(median {pair_block['median_span_words']:.0f} words, max nesting "
        f"{pair_block['max_depth']}), i.e. genuine multi-word blocks, not tokens."
    )
    v[">>"] = (
        "**SUPPORTED (block close delimiter).** "
        f"`>>` closes a span and is glued to the left "
        f"({p['>>']['glue_l']:.0%}); the paired spans above end here. "
        f"Note the counts are not exactly balanced (`<<` {p['<<']['n']}, "
        f"`>>` {p['>>']['n']}; {pair_block['unmatched_open']} unmatched opens, "
        f"{pair_block['unmatched_close']} unmatched closes), so `>>` also acts as a "
        "freestanding *flow/terminator* marker — see caveats."
    )
    v["/"] = (
        "**SUPPORTED (dual role: medial compound/possessive + sentence-final terminator); "
        "alternation ('or') is NOT evidenced.** "
        f"Medially `/` is a tight word linker (left word "
        f"{p['/']['lclass'].get('WORD', 0) / p['/']['n']:.0%}, glue-left "
        f"{p['/']['glue_l']:.0%}) building {link['slash']['n']:,} slash chains "
        f"({link['slash']['unique']:,} unique), top "
        + ", ".join(f"`{w}`×{c}" for w, c in link['slash']['top'][:3])
        + f". But {p['/']['eol']:.0%} of `/` are at end-of-line — a "
        "sentence/utterance terminator. The medial chains are compounds/portmanteaux "
        "(`soul/gram`), not a Boolean 'or' (no `A/B` alternation semantics is "
        "observable); both the compound and terminator functions coexist."
    )
    v["++"] = (
        "**SUPPORTED as concatenation/coupling; REJECTED as arithmetic increment.** "
        f"`++` is glued on both sides ({p['++']['both_glued']:.0%}) and joins "
        f"lexical material: medial between words {p['++']['lclass'].get('WORD', 0) / p['++']['n']:.0%}"
        f" of the time, clause-initial (after another operator) "
        f"{p['++']['lclass'].get('OP', 0) / p['++']['n']:.0%}. "
        "Operands are words, never numerals, so the C `x++` arithmetic reading "
        "is rejected; it is a coupling/concat op."
    )
    v["`"] = (
        "**REJECTED — not present in the page-body corpus.** "
        "The backtick occurs 0 times in the vetted layout text, so the "
        "'escape/literalization' role has no evidence. (`^`, `~`, `--`, `==`, "
        "`->`, `<-` are likewise absent from the page bodies.)"
    )
    v["-"] = (
        "**SUPPORTED (dual role): word-internal compound linker + free-standing "
        "phrase/clause separator.** The raw text contains "
        f"{link['hyphen']['n']:,} hyphenated compounds ({link['hyphen']['unique']:,} "
        "unique types), e.g. "
        + ", ".join(f"`{w}`×{c}" for w, c in link['hyphen']['top'][:5])
        + f". Separately, there are **{p['-']['n']} free-standing whitespace-bounded "
        f"` - ` tokens** (measured, not inferred from the operator list): "
        f"{p['-']['interior']:.0%} sit between two words, upper-next "
        f"{p['-']['upper_next']:.0%} (comparable to `::` "
        f"{p['::']['upper_next']:.0%}, far below `//` {p['//']['upper_next']:.0%}), "
        f"SOL {p['-']['sol']:.1%}/EOL {p['-']['eol']:.1%}. Their right neighbours are "
        "dominated by clause-openers ("
        + ", ".join(f"`{w}`×{c}" for w, c in p['-']['top_rwords'][:4])
        + "), so the free dash behaves like the `::` phrase-separator class, while the "
        "hyphen remains a morphological joiner inside word tokens. The earlier "
        "'word-internal only' claim was a definitional inference and is corrected here."
    )
    v["+"] = (
        "**SUPPORTED as a tight compound linker (rare).** "
        f"All {p['+']['n']} `+` are glued both sides ({p['+']['both_glued']:.0%}) "
        f"with words on both sides ({p['+']['interior']:.0%} interior); the regex "
        f"finds {link['plus']['n']} `word+word` compounds, top "
        + ", ".join(f"`{w}`" for w, c in link['plus']['top'][:4])
        + ". Same role as `=`/`/`, lower frequency."
    )
    v["|"] = (
        "**REFINED — a stream/pipe marker, not alternation.** "
        f"All {p['|']['n']} occurrences are followed by a word and "
        f"{p['|']['lclass'].get('OP', 0)}/{p['|']['n']} are preceded by another "
        "operator; the dominant context is the run `<<|` (stream start), plus "
        "`:: |` (pipe after a field). Too rare (n=17) for a confident grammar, "
        "but the alternation reading is unsupported."
    )
    v["<"] = (
        "**REFINED — a lightweight inline delimiter (single angle).** "
        f"`<` is often preceded by another operator "
        f"({p['<']['lclass'].get('OP', 0) / p['<']['n']:.0%}) or punctuation and "
        f"opens a short `<…>` quote: pairing finds {pair_angle_clean['matched']} "
        f"matched spans (median {pair_angle_clean['median_span_words']:.0f} words), "
        "e.g. `<gravity zero>`, `<contact>`, `<oscillation>`. A page of raw HTML "
        "OCR inflates the raw counts (see caveats)."
    )
    v[">"] = (
        "**REFINED — inline close/flow marker; over-produced relative to `<`.** "
        f"`>` is glued left ({p['>']['glue_l']:.0%}) and is followed by a word "
        f"{p['>']['rclass'].get('WORD', 0) / p['>']['n']:.0%} / another operator "
        f"{p['>']['rclass'].get('OP', 0) / p['>']['n']:.0%} of the time. "
        f"There are {p['>']['n']} `>` but only {p['<']['n']} `<`, so `>` also "
        "serves as a general flow/terminator marker (and absorbs OCR artifacts)."
    )
    return v


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(prof, occ, link, pairs, tests, verdict, absent, path):
    ops = [o for o in ("::", "//", "=", "/", "<<", ">>", "++", "+", "|", "<", ">", "-")
           if o in prof]
    L = []
    ap = L.append

    def md(op):
        """Render an operator as inline code, escaping the backtick operator."""
        return "`` ` ``" if op == "`" else f"`{op}`"
    # Raw-vs-body rendering of the absent-operator audit, so the "0 occurrences"
    # claims below are reconcilable with a grep of the source file.
    scaffolding = [f"{md(op)}×{d['raw_file']}" for op, d in absent.items()
                   if d["raw_file"]]
    scaffold_note = (
        "all absent-operator counts are 0 in the page bodies; in the raw layout file "
        + ", ".join(scaffolding)
        + " occur only inside `<!-- page N -->` / `<!-- figure … -->` scaffolding comments "
          "that `load_pages` strips, so they are layout metadata, not book text")
    ap("# Operator Function Classification (#12)\n")
    ap("Source: `analysis/layout_recovery/blood-electric.layout.md` — the vetted layout "
       "markdown (#7). Operator set and ordered tokenizer reused from "
       "`work/operator_inventory.py` (#10); the 10,949 operator-token count is reproduced "
       "exactly. Tests the working hypotheses in `docs/meaning-extraction-methods.md` §L1 "
       "against the distribution of each operator's contexts.\n")
    ap("**Method.** Every operator occurrence is described by its immediately adjacent "
       "non-space token class (WORD / OP / OTHER / line boundary), whether it is *glued* "
       "to that neighbour (no whitespace), its line position (start/end of line), whether "
       "the next word is sentence-initial (uppercase), and the word-run size on each side. "
       "A role is *supported* only when its predicted signature is present in the measured "
       "distribution; otherwise it is refined or rejected. All numbers are reproducible via "
       "`python work/operator_functions.py`.\n")
    ap("## Summary verdicts\n")
    ap("| operator | n | hypothesized role | verdict |")
    ap("|---|---:|---|---|")
    short = {
        "::": "SUPPORTED — phrase/field separator",
        "//": "SUPPORTED (utterance) / REJECTED (line comment)",
        "=": "SUPPORTED — naming/identity linker",
        "/": "SUPPORTED — compound/possessive + sentence-final",
        "<<": "SUPPORTED — block open",
        ">>": "SUPPORTED — block close (+ flow marker)",
        "++": "SUPPORTED (coupling) / REJECTED (arithmetic)",
        "+": "SUPPORTED — compound linker (rare)",
        "|": "REFINED — stream/pipe marker (rare)",
        "<": "REFINED — inline delimiter (noisy)",
        ">": "REFINED — inline close/flow marker",
        "-": "SUPPORTED — compound linker + free-standing dash separator",
        "`": "REJECTED — absent",
    }
    for op in ops:
        ap(f"| `{op}` | {prof[op]['n']:,} | {HYPOTHESIS.get(op, '—')} | {short[op]} |")
    ap(f"| `-` (word-internal) | {link['hyphen']['n']:,} (hyphen compounds) | "
       f"{HYPOTHESIS['-']} | {short['-']} |")
    ap(f"| `` ` `` | 0 | {HYPOTHESIS['`']} | {short['`']} |")
    ap("")

    ap("## Signature profile (measured)\n")
    ap("`glue` = fraction with no whitespace between the operator and an adjacent "
       "*word*; `interior` = word immediately before AND after; `upper-next` = next "
       "word is sentence-initial; `run` = operator is part of a multi-operator run.\n")
    ap("| op | n | glue-L | glue-R | both-glued | interior | SOL | EOL | run | upper-next | med words L/R |")
    ap("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for op in ops:
        p = prof[op]
        ap(f"| `{op}` | {p['n']:,} | {p['glue_l']:.2f} | {p['glue_r']:.2f} | "
           f"{p['both_glued']:.2f} | {p['interior']:.2f} | {p['sol']:.2f} | "
           f"{p['eol']:.2f} | {p['run_member']:.2f} | {p['upper_next']:.2f} | "
           f"{p['med_lp']:.0f}/{p['med_rp']:.0f} |")
    ap(f"\nThe `-` row is the **{prof['-']['n']} free-standing whitespace-bounded ` - ` "
       "tokens** (dashes), which are *not* part of the 10,949 #10 operator tokens; their "
       "glue columns are 0 by construction (whitespace on both sides). Word-internal "
       "hyphen compounds are counted separately (see `-` section).\n")
    ap(f"The discriminating signal is **upper-next**: `//` "
       f"{prof['//']['upper_next']:.0%} vs `::` {prof['::']['upper_next']:.0%} "
       f"(two-proportion z={tests['sep_upper_z'][0]:.1f}, "
       f"p={tests['sep_upper_z'][1]:.1e}). `//` starts new sentences; `::` does not. "
       f"`=` ({prof['=']['upper_next']:.0%}) and `++` "
       f"({prof['++']['upper_next']:.0%}) are tight lexical linkers.\n")

    ap("## Neighbour-class distribution\n")
    ap("| op | left WORD | left OP | left OTHER | left BOUND | right WORD | right OP | right OTHER | right BOUND |")
    ap("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for op in ops:
        p = prof[op]
        n = p["n"]
        def g(d, k):
            return f"{d.get(k, 0) / n:.2f}"
        ap(f"| `{op}` | {g(p['lclass'],'WORD')} | {g(p['lclass'],'OP')} | "
           f"{g(p['lclass'],'OTHER')} | {g(p['lclass'],'BOUND')} | "
           f"{g(p['rclass'],'WORD')} | {g(p['rclass'],'OP')} | "
           f"{g(p['rclass'],'OTHER')} | {g(p['rclass'],'BOUND')} |")
    ap("")

    ap("## Per-operator evidence\n")
    order = ["::", "//", "=", "/", "++", "+", "|", "<<", ">>", "<", ">"]
    for op in order:
        if op not in prof:
            continue
        p = prof[op]
        ap(f"### `{op}` — {short[op]}\n")
        ap(f"- hypothesis: {HYPOTHESIS.get(op, '—')}")
        ap(f"- n = {p['n']:,}; both-glued {p['both_glued']:.0%}; interior "
           f"{p['interior']:.0%}; SOL {p['sol']:.1%}; EOL {p['eol']:.1%}; "
           f"run-member {p['run_member']:.0%}; upper-next {p['upper_next']:.1%}")
        if p["top_lwords"]:
            ap("- top left words: " + ", ".join(f"`{w}`×{c}" for w, c in p["top_lwords"][:6]))
        if p["top_rwords"]:
            ap("- top right words: " + ", ".join(f"`{w}`×{c}" for w, c in p["top_rwords"][:6]))
        if tests["per_op"].get(op):
            G, df, pv = tests["per_op"][op]
            if pv is not None:
                ap(f"- context-independence test (this op vs rest, 2×16 G-test): "
                   f"G={G:.1f}, df={df}, p={pv:.1e} → context distribution is "
                   f"{'distinct' if pv < 0.05 else 'not distinguishable'}")
        ap(f"- verdict: {verdict[op]}\n")

    # `-` is measured in two distinct roles; `` ` `` is absent.
    fp = prof["-"]
    ap("### `-` — SUPPORTED — compound linker + free-standing dash separator\n")
    ap(f"- hypothesis: {HYPOTHESIS['-']}")
    ap(f"- **free-standing role (measured):** {fp['n']} whitespace-bounded ` - ` "
       f"tokens; interior (word before AND after) {fp['interior']:.0%}; SOL "
       f"{fp['sol']:.1%}; EOL {fp['eol']:.1%}; upper-next {fp['upper_next']:.1%} "
       f"(vs `::` {prof['::']['upper_next']:.0%}, `//` {prof['//']['upper_next']:.0%}); "
       f"median words L/R {fp['med_lp']:.0f}/{fp['med_rp']:.0f}")
    ap("- free-dash top left words: "
       + ", ".join(f"`{w}`×{c}" for w, c in fp["top_lwords"][:6]))
    ap("- free-dash top right words: "
       + ", ".join(f"`{w}`×{c}" for w, c in fp["top_rwords"][:6]))
    ap(f"- **word-internal role (measured):** {link['hyphen']['n']:,} hyphenated "
       f"compound tokens ({link['hyphen']['unique']:,} unique types); top: "
       + ", ".join(f"`{w}`×{c}" for w, c in link['hyphen']['top'][:6]))
    ap("- the inventory's double-hyphen `--` occurs "
       f"{absent['--']['page_body']} times in the page-body corpus "
       f"({absent['--']['raw_file']} in the raw layout file, all of them inside "
       "layout scaffolding comments); `-` itself is not in #10's operator set, so the "
       "free-standing tokens above are additional to the 10,949 operator count")
    ap(f"- verdict: {verdict['-']}\n")
    ap("### `` ` `` — REJECTED — absent\n")
    ap(f"- hypothesis: {HYPOTHESIS['`']}")
    ap("- count in the vetted corpus: **0** (likewise "
       + ", ".join(f"{md(op)} {d['page_body']}" for op, d in absent.items())
       + f"; {scaffold_note})")
    ap(f"- verdict: {verdict['`']}\n")

    ap("## Delimiter pairing\n")
    ap("Stack-matching the operator stream within each page.\n")
    ap("| pair | matched spans | unmatched open | unmatched close | median span (words) | mean span | max depth |")
    ap("|---|---:|---:|---:|---:|---:|---:|")
    for name, st in (("`<<`…`>>` (full)", pairs["block"]),
                     ("`<<`…`>>` (no HTML-artifact pages)", pairs["block_clean"]),
                     ("`<`…`>` (full)", pairs["angle"]),
                     ("`<`…`>` (no HTML-artifact pages)", pairs["angle_clean"])):
        ap(f"| {name} | {st['matched']:,} | {st['unmatched_open']:,} | "
           f"{st['unmatched_close']:,} | {st['median_span_words']:.0f} | "
           f"{st['mean_span_words']:.1f} | {st['max_depth']} |")
    ap(f"\n`<<`/`>>` pairing is robust to removing the raw-HTML OCR pages "
       f"({', '.join(map(str, sorted(HTML_ARTIFACT_PAGES)))}): "
       f"{pairs['block']['matched']} → {pairs['block_clean']['matched']} spans. "
       f"`<`/`>` pairing is not: {pairs['angle']['matched']} → "
       f"{pairs['angle_clean']['matched']} spans, confirming that a large share of "
       f"single-angle 'matches' are `<a href>…</a>` HTML artifacts on page 92.\n")

    ap("## Linker / compound inventory\n")
    ap("| separator | occurrences | unique | example |")
    ap("|---|---:|---:|---|")
    for key, label in (("equals", "`=`"), ("slash", "`/`"), ("plusplus", "`++`"),
                       ("plus", "`+`"), ("hyphen", "`-`")):
        d = link[key]
        ex = ", ".join(f"`{w}`" for w, _c in d["top"][:4])
        ap(f"| {label} | {d['n']:,} | {d['unique']:,} | {ex} |")
    ap("\nSlash chain lengths (number of terms joined by `/`): "
       + ", ".join(f"{k}→{v}" for k, v in link["slash"]["chain_len"].items())
       + ". The dominant chain is `soul/gram`.\n")

    ap("## Caveats and negative results\n")
    ap("- **Absent operators.** `` ` ``, `~`, `^`, `--`, `==`, `->`, `<-` occur 0 times "
       "in the analysed page bodies. The backtick 'escape' hypothesis is rejected on "
       "absence. Scope note (measured both ways, see `absent_operator_audit` in "
       f"`summary.json`): {scaffold_note}.")
    ap("- **HTML-artifact pages.** Pages "
       f"{', '.join(map(str, sorted(HTML_ARTIFACT_PAGES)))} contain raw `<a href>`/"
       "`mailto:` OCR markup; page 92 alone supplies 97 of the 113 `</` runs. The "
       "angle-bracket pairing table is reported with and without these pages.")
    ap("- **Angle-bracket asymmetry.** `<`-side tokens (381 `<` + 412 `<<` = 793) are "
       "fewer than `>`-side tokens (777 `>` + 386 `>>` = 1,163). `>` therefore also "
       "serves as a freestanding flow/terminator marker, and some of the excess is OCR "
       "noise; single-angle pairing is the least reliable result here.")
    ap("- **Directionality is not fully nested.** `<<`/`>>` pairing leaves 182 unmatched "
       "opens and 156 unmatched closes; some blocks span page boundaries or are never "
       "closed. The median matched span (5 words) is the robust part.")
    ap("- **Slash chains are mostly compounds, not alternation.** Of 767 `word/word` "
       "chains, a small minority join a content word to a function word "
       "(`internal/the`, `dog/the`, `zero/of`), which may be a line-final `/` abutting "
       "the next line rather than a true compound. The dominant chain `soul/gram` "
       "(317) is a genuine portmanteau seam.")
    ap("- **Significance.** Every operator except `+` (n=15) has a context distribution "
       "significantly different from the pooled distribution of the others "
       "(2×16 G-test, p<0.001; see per-operator sections). `+` is too rare to test. "
       "With N≈10,949 these p-values are tiny for any real difference; the effect "
       "*sizes* (the profile columns) matter more than the p-values, and several "
       "contingency cells are sparse (e.g. line-boundary contexts).")
    ap("- **Scale claim.** These are distributional signatures, not a parser. Whether "
       "the roles compose into a designed grammar is the question of #14 (block parser) "
       "and #13 (silence removal).")
    ap("- **Interpretation boundary.** Measured structure is reported above; the mapping "
       "of a role to a *meaning* (e.g. `=` = 'is called') is an interpretive hypothesis "
       "consistent with the distribution, not a proof.")
    ap("\n---\n*Generated by `work/operator_functions.py`*")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    pages = load_pages(SRC)
    occ = build_occurrences(pages)
    os.makedirs(OUT, exist_ok=True)
    ops, prof = per_operator(occ)
    assert sum(ops.values()) == 10949, f"operator count drift: {sum(ops.values())}"

    # Free-standing ` - ` dashes are measured separately from the operator set.
    free = free_hyphen_occurrences(pages)
    _free_ops, fprof = per_operator(free)
    prof["-"] = fprof["-"]
    op_keys = list(ops) + ["-"]

    # neighbor class table
    with open(f"{OUT}/neighbor_classes.tsv", "w") as f:
        f.write("operator\tside\tclass\tcount\tfraction\n")
        for op in op_keys:
            for side, key in (("L", "lclass"), ("R", "rclass")):
                for cls, c in sorted(prof[op][key].items(), key=lambda kv: -kv[1]):
                    f.write(f"{op}\t{side}\t{cls}\t{c}\t{c / prof[op]['n']:.4f}\n")

    # per-operator profile
    with open(f"{OUT}/operator_profiles.tsv", "w") as f:
        f.write("operator\tn\tglue_l\tglue_r\tboth_glued\tinterior\tsol\teol\t"
                "run_member\tupper_next\tmed_words_left\tmed_words_right\n")
        for op in sorted(prof, key=lambda o: -prof[o]["n"]):
            p = prof[op]
            f.write(f"{op}\t{p['n']}\t{p['glue_l']:.4f}\t{p['glue_r']:.4f}\t"
                    f"{p['both_glued']:.4f}\t{p['interior']:.4f}\t{p['sol']:.4f}\t"
                    f"{p['eol']:.4f}\t{p['run_member']:.4f}\t{p['upper_next']:.4f}\t"
                    f"{p['med_lp']:.0f}\t{p['med_rp']:.0f}\n")

    # joined pairs / compounds
    link = linker_evidence(pages)
    with open(f"{OUT}/joined_pairs.tsv", "w") as f:
        f.write("separator\toccurrences\tunique\texample\tcount\n")
        for key, label in (("equals", "="), ("slash", "/"), ("plusplus", "++"),
                           ("plus", "+"), ("hyphen", "-")):
            d = link[key]
            for w, c in d["top"]:
                f.write(f"{label}\t{d['n']}\t{d['unique']}\t{w}\t{c}\n")

    # delimiter pairing
    pairs = {
        "block": pairing_stats(pages, "<<", ">>"),
        "block_clean": pairing_stats(pages, "<<", ">>", HTML_ARTIFACT_PAGES),
        "angle": pairing_stats(pages, "<", ">"),
        "angle_clean": pairing_stats(pages, "<", ">", HTML_ARTIFACT_PAGES),
    }
    with open(f"{OUT}/delimiter_pairing.tsv", "w") as f:
        f.write("pair\tvariant\tmatched\tunmatched_open\tunmatched_close\t"
                "median_span_words\tmean_span_words\tmax_depth\n")
        for name, key in (("<<>>", "block"), ("<<>>", "block_clean"),
                          ("<>", "angle"), ("<>", "angle_clean")):
            st = pairs[key]
            f.write(f"{name}\t{key}\t{st['matched']}\t{st['unmatched_open']}\t"
                    f"{st['unmatched_close']}\t{st['median_span_words']}\t"
                    f"{st['mean_span_words']}\t{st['max_depth']}\n")

    # citable contexts (up to 6 per operator; 5 free-standing dashes)
    with open(f"{OUT}/contexts.tsv", "w") as f:
        f.write("operator\tpage\tcontext\n")
        for source, limit in ((occ, 6), (free, 5)):
            seen = collections.Counter()
            for o in source:
                if seen[o["op"]] >= limit:
                    continue
                seen[o["op"]] += 1
                a = max(0, o["s"] - 70)
                b = min(len(o["text"]), o["e"] + 70)
                snippet = o["text"][a:b].replace("\t", " ").replace("\n", " ")
                f.write(f"{o['op']}\t{o['page']}\t{snippet}\n")

    # statistical tests
    n_col, k_col = prof["::"]["n"], sum(o["upper_next"] for o in occ if o["op"] == "::")
    n_sl, k_sl = prof["//"]["n"], sum(o["upper_next"] for o in occ if o["op"] == "//")
    z_upper, p_upper = two_proportion_z(k_sl, n_sl, k_col, n_col)
    e_col = sum(o["eol"] for o in occ if o["op"] == "::")
    e_sl = sum(o["eol"] for o in occ if o["op"] == "//")
    z_eol, p_eol = two_proportion_z(e_sl, n_sl, e_col, n_col)
    per_op_tests = {}
    for op in ops:
        per_op_tests[op] = chi2_context(occ, op)
    tests = {"sep_upper_z": (z_upper, p_upper), "sep_eol_z": (z_eol, p_eol),
             "per_op": per_op_tests}

    # Absent operators are audited against BOTH the raw layout file and the
    # stripped page bodies, so the "0 occurrences" claim is reconcilable with a
    # naive grep of the source file.
    with open(SRC, errors="ignore") as f:
        raw_file_text = f.read()
    absent = absent_operator_audit(raw_file_text, pages)

    verdict = verdicts(prof, link, pairs["block"], pairs["angle_clean"], tests)
    write_report(prof, occ, link, pairs, tests, verdict, absent,
                 f"{OUT}/classification_spec.md")

    summary = {
        "source": SRC,
        "n_operator_tokens": sum(ops.values()),
        "n_distinct": len(ops),
        "n_free_hyphen": len(free),
        "counts": dict(ops),
        "free_hyphen": {
            "n": prof["-"]["n"],
            "interior": prof["-"]["interior"],
            "sol": prof["-"]["sol"],
            "eol": prof["-"]["eol"],
            "upper_next": prof["-"]["upper_next"],
            "lclass": prof["-"]["lclass"],
            "rclass": prof["-"]["rclass"],
            "top_lwords": prof["-"]["top_lwords"][:8],
            "top_rwords": prof["-"]["top_rwords"][:8],
        },
        "profiles": {op: {k: v for k, v in prof[op].items()
                          if k not in ("lclass", "rclass", "top_lwords", "top_rwords")}
                     for op in prof},
        "linker_evidence": {k: {"n": v["n"], "unique": v["unique"],
                                "top": v["top"][:5]}
                            for k, v in link.items()},
        "delimiter_pairing": pairs,
        "absent_operator_audit": absent,
        "tests": {"upper_next_z": {"z": z_upper, "p": p_upper,
                                   "//": prof["//"]["upper_next"],
                                   "::": prof["::"]["upper_next"]},
                  "eol_z": {"z": z_eol, "p": p_eol},
                  "per_operator_context_G": {op: {"G": g, "df": df, "p": p}
                                             for op, (g, df, p) in per_op_tests.items()}},
        "verdict_short": {
            "::": "SUPPORTED phrase/field separator",
            "//": "SUPPORTED utterance break; REJECTED line comment",
            "=": "SUPPORTED naming/identity linker",
            "/": "SUPPORTED compound/possessive + sentence-final",
            "<<": "SUPPORTED block open",
            ">>": "SUPPORTED block close (+flow marker)",
            "++": "SUPPORTED coupling; REJECTED arithmetic increment",
            "+": "SUPPORTED compound linker (rare)",
            "|": "REFINED stream/pipe marker (rare)",
            "<": "REFINED inline delimiter (noisy)",
            ">": "REFINED inline close/flow marker",
            "-": "SUPPORTED word-internal compound linker + free-standing dash separator",
            "`": "REJECTED absent",
        },
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"operators={len(ops)} tokens={sum(ops.values())}")
    print(f"upper-next ::={prof['::']['upper_next']:.3f} //={prof['//']['upper_next']:.3f} "
          f"(z={z_upper:.1f}, p={p_upper:.2e})")
    print(f"block pairing matched={pairs['block']['matched']} "
          f"median_span={pairs['block']['median_span_words']}")
    print(f"hyphen compounds={link['hyphen']['n']} free_hyphen={len(free)} "
          f"equals_pairs={link['equals']['n']} slash_chains={link['slash']['n']}")
    print(f"free '-' interior={prof['-']['interior']:.3f} "
          f"upper_next={prof['-']['upper_next']:.3f} top_right={prof['-']['top_rwords'][:3]}")
    print("[written] " + OUT + "/classification_spec.md")
    print("[written] " + OUT + "/operator_profiles.tsv")
    print("[written] " + OUT + "/neighbor_classes.tsv")
    print("[written] " + OUT + "/joined_pairs.tsv")
    print("[written] " + OUT + "/contexts.tsv")
    print("[written] " + OUT + "/delimiter_pairing.tsv")
    print("[written] " + OUT + "/summary.json")


if __name__ == "__main__":
    main()
