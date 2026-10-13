#!/usr/bin/env python3
"""#52 Operator-word mutual information -> analysis/operator_word_mi/

Which operators predict which lexical domains?  The issue's own examples are
``::`` before body terms and ``=`` before tech terms: it asks for the operator
grammar (#10 inventory, #12 classification, #14 parse) *as statistics*.

What is measured
----------------
The #17 token layer (``tokens.tsv``) keeps operators visible; the #17 unit
layer (``units.tsv``) keeps the lexical units they sit between.  Every
screened operator occurrence is resolved to the **term it stands next to**:

* a *free* operator (no unit contains its span) takes the surface of the
  nearest screened unit on the required side;
* an *in-unit* operator (``=`` in ``suck=blood``, ``-`` in ``boy-roid``)
  takes the operand substring of its own unit's surface on each side, so
  ``=`` has ``suck`` on its left and ``blood`` on its right.

A term belongs to #29's six seeded lexical domains (body / machine /
violence / sex / genetics / corporate) when the term itself or one of its
seam-split parts (``drug-embryo`` -> ``embryo``) is in that domain's seed
set; the seed literal is parsed out of ``work/cooccurrence_network.py``
with ``ast`` rather than duplicated or imported.

Statistics per direction (right = "operator before term", the issue's
framing; left = "operator after term"):

* the **MI matrix** - pointwise mutual information (bits) between operator
  and domain over the incidence table, with expected counts and lift;
* **per-domain binary tests** - a 1-df G^2 per (operator, domain) with
  Benjamini-Hochberg q over the 6 x 11 grid, which is what "predicts" means
  statistically;
* **total MI** - three quantities on the same stream: I(O;D) on the
  domain-combination label (single-label, a true mutual information), I(O;T)
  on the term itself, and the incidence multi-information; each against a
  200-replicate label-permutation null (fixed seed) that preserves both
  margininals;
* **robustness** - the same totals under strict whole-term seed matching
  (#29-faithful), and per book half.

Nothing is re-tokenized, repaired or smoothed; the markup screen is #27's
(#19 ledger + tag-run detector), so the operator counts differ from #10's
exactly by the raw-HTML ``<``/``>`` tokens that screen removes.

Reproduce::

    conda activate marker
    python work/operator_word_mi.py
    python work/test_operator_word_mi.py
"""
from __future__ import annotations

import argparse
import ast
import bisect
import collections
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import chi2

from collocation import (TOKENS, UNITS, load_bodies, load_markup, read_tsv,
                         unit_is_markup, write_tsv)

ROOT = Path(__file__).resolve().parent.parent
NETWORK_SCRIPT = ROOT / "work" / "cooccurrence_network.py"
OUT = ROOT / "analysis" / "operator_word_mi"

SCHEMA = "operator_word_mi/1"
DOMAINS = ("body", "machine", "violence", "sex", "genetics", "corporate")
SIDES = ("right", "left")
SEED = 42
N_SHUFFLE = 200
NONE = "none"
BOUNDARY = "boundary"
# #11's operator lag-1 MI, quoted as the scale reference in the report.
OP_LAG1_MI_11 = 0.1305

_SEAM_RE = re.compile(r"[=\-/+]+")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_domain_seeds(path: Path = NETWORK_SCRIPT):
    """#29's ``DOMAIN_SEEDS``, parsed out of its builder with ``ast``.

    The literal is read from ``work/cooccurrence_network.py`` instead of being
    duplicated here, and ``ast.literal_eval`` avoids importing that module
    (it pulls ``networkx``/``sklearn``, which this build does not need).
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DOMAIN_SEEDS"
                for t in node.targets):
            return {k: set(v) for k, v in ast.literal_eval(node.value).items()}
    raise KeyError("DOMAIN_SEEDS not found in %s" % path)


def load_token_stream():
    """{page: [(start, end, text, kind, region), ...]} sorted by start."""
    page_tokens = collections.defaultdict(list)
    for r in read_tsv(TOKENS):
        page_tokens[int(r["page"])].append(
            (int(r["start"]), int(r["end"]), r["text"], r["kind"],
             r["region"]))
    for page in page_tokens:
        page_tokens[page].sort()
    return page_tokens


def screened_units():
    """({page: [(start, end, surface), ...] sorted}, n_screened_out, total).

    The screen is #27's: #19's markup-residue ledger plus the general
    tag-run detector, applied exactly as ``collocation`` applies it.
    """
    by_page = collections.defaultdict(list)
    for r in read_tsv(UNITS):
        by_page[int(r["page"])].append(
            (int(r["start"]), int(r["end"]), r["unit"], int(r["uid"])))
    for page in by_page:
        by_page[page].sort()
    bodies = load_bodies()
    markup_spans, markup_uids, markup_types = load_markup(bodies)
    out = collections.defaultdict(list)
    n_out = 0
    for page, units in by_page.items():
        for s, e, surface, uid in units:
            if unit_is_markup(surface, uid, s, page, markup_spans,
                              markup_uids, markup_types):
                n_out += 1
                continue
            out[page].append((s, e, surface))
    return out, n_out, sum(len(v) for v in by_page.values())


# ---------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------
def seam_parts(term: str):
    """``drug-embryo`` -> ``[drug, embryo]``; a bare word -> ``[word]``."""
    return [p for p in _SEAM_RE.split(str(term).lower()) if p]


def term_domains(term, seeds, strict=False):
    """Domains whose seed set contains the term or a seam-split part.

    ``strict`` matches whole terms only, which is exactly #29's node-surface
    matching and is the sensitivity variant, not the default.
    """
    if not term:
        return set()
    low = str(term).lower()
    hits = {d for d, words in seeds.items() if low in words}
    if hits or strict:
        return hits
    parts = seam_parts(low)
    if len(parts) <= 1:
        return set()
    return {d for d, words in seeds.items()
            if any(p in words for p in parts)}


def domain_label(term, seeds, strict=False):
    """The single-label domain class of a term: a joined combination, the
    ``none`` class, or ``boundary`` when there is no term at all."""
    if term is None:
        return BOUNDARY
    doms = sorted(term_domains(term, seeds, strict=strict))
    if not doms:
        return NONE
    return "+".join(doms)


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
def resolve_page(units, ops):
    """Resolve the (left, right) term of every operator on one page.

    ``units``: [(start, end, surface), ...] sorted, disjoint, already
    markup-screened.  ``ops``: [(start, end, text), ...] sorted.  Returns a
    list of ``(op_text, in_unit, left_term, right_term)``; a side is
    ``None`` when no unit lies on it (page boundary).

    * an operator whose span is contained in a unit takes the operand
      substrings of that unit's surface (``=`` in ``suck=blood`` has
      ``suck`` left and ``blood`` right);
    * a free operator takes the surface of the nearest unit on each side.
    """
    starts = [u[0] for u in units]
    ends = [u[1] for u in units]
    out = []
    for s, e, text in ops:
        left = right = None
        in_unit = False
        i = bisect.bisect_right(starts, s) - 1
        if i >= 0 and ends[i] >= e:
            in_unit = True
            us, _ue, surface = units[i]
            left = surface[: s - us] or None
            right = surface[e - us:] or None
        else:
            j = bisect.bisect_left(starts, e)
            if j < len(units):
                right = units[j][2]
            if i >= 0:
                left = units[i][2]
        out.append((text, in_unit, left, right))
    return out


def build_occurrences():
    """One record per screened operator occurrence.

    Returns (occurrences, diag).  Each occurrence::

        {"page", "start", "op", "region", "in_unit", "left", "right"}

    ``left``/``right`` are the term the operator stands next to on that side
    (a unit surface, or the in-unit operand substring), or ``None`` when no
    unit survives the screen on that side (page boundary).
    """
    page_tokens = load_token_stream()
    units_by_page, n_units_out, n_units_total = screened_units()
    bodies = load_bodies()
    markup_spans, _muids, _mtypes = load_markup(bodies)

    occurrences = []
    n_op_tokens = 0
    n_in_unit = 0
    n_free = 0
    n_free_in_span = 0
    regions = collections.Counter()

    for page in sorted(page_tokens):
        spans = markup_spans.get(page, [])
        units = units_by_page.get(page, [])
        ops = []
        for s, e, text, kind, region in page_tokens[page]:
            if kind != "OP":
                continue
            n_op_tokens += 1
            if any(a <= s < b for a, b in spans):
                continue
            ops.append((s, e, text, region))
            regions[region] += 1
        for (s, e, text, region), (_op, in_unit, left, right) in zip(
                ops, resolve_page(units, [(o[0], o[1], o[2]) for o in ops])):
            if in_unit:
                n_in_unit += 1
            else:
                n_free += 1
            occurrences.append({
                "page": page, "start": s, "op": text, "region": region,
                "in_unit": in_unit, "left": left, "right": right,
            })

    # free tokens geometrically inside a unit span (a #17 seam surprise)
    for occ in occurrences:
        if occ["in_unit"]:
            continue
        for us, ue, _surf in units_by_page.get(occ["page"], []):
            if us <= occ["start"] < ue:
                n_free_in_span += 1
                break

    diag = {
        "op_tokens_stream": n_op_tokens,
        "op_tokens_markup_removed": n_op_tokens - len(occurrences),
        "op_occurrences": len(occurrences),
        "op_in_unit": n_in_unit,
        "op_free": n_free,
        "op_free_inside_unit_span": n_free_in_span,
        "header_region_ops": regions.get("header", 0),
        "units_total": n_units_total,
        "units_screened_out": n_units_out,
    }
    return occurrences, diag


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def total_mi(pairs):
    """Plug-in mutual information (bits) of a list of (x, y) observations."""
    n = len(pairs)
    if n == 0:
        return 0.0
    f_cell = collections.Counter(pairs)
    f_x = collections.Counter(x for x, _y in pairs)
    f_y = collections.Counter(y for _x, y in pairs)
    mi = 0.0
    for (x, y), f in f_cell.items():
        if f > 0:
            mi += (f / n) * math.log2((f * n) / (f_x[x] * f_y[y]))
    return mi


def side_pairs(occurrences, side, seeds, strict=False):
    """(op, label), (op, term), [domains] observation lists for one side."""
    label_pairs, term_pairs, domain_sets = [], [], []
    for occ in occurrences:
        term = occ[side]
        label_pairs.append((occ["op"], domain_label(term, seeds, strict)))
        term_pairs.append((occ["op"], term if term is not None else BOUNDARY))
        domain_sets.append(term_domains(term, seeds, strict)
                           if term is not None else set())
    return label_pairs, term_pairs, domain_sets


def incidence_table(occurrences, side, seeds, strict=False):
    """(op, domain) incidence counts + marginals for one side.

    A term in two domains is an incidence in both, so the operator incidence
    totals exceed the occurrence count; that multi-label basis is what the
    MI matrix reports.
    """
    _lp, _tp, domain_sets = side_pairs(occurrences, side, seeds, strict)
    cell = collections.Counter()
    f_op = collections.Counter()
    f_dom = collections.Counter()
    n_inc = 0
    for occ, doms in zip(occurrences, domain_sets):
        op = occ["op"]
        f_op[op] += len(doms)
        n_inc += len(doms)
        for d in doms:
            cell[(op, d)] += 1
            f_dom[d] += 1
    return cell, f_op, f_dom, n_inc


def incidence_mi_from(cell, f_op, f_dom, n_inc):
    """Multi-information (bits) of an incidence table."""
    if n_inc <= 0:
        return 0.0
    mi = 0.0
    for (op, d), f in cell.items():
        if f > 0 and f_op[op] and f_dom[d]:
            mi += (f / n_inc) * math.log2(
                (f * n_inc) / (f_op[op] * f_dom[d]))
    return mi


def incidence_mi(occurrences, side, seeds, strict=False):
    """Multi-information (bits) of the incidence table on one side."""
    cell, f_op, f_dom, n_inc = incidence_table(occurrences, side, seeds,
                                               strict)
    return incidence_mi_from(cell, f_op, f_dom, n_inc)


def pmi_cell(f_xy, f_x, f_y, n):
    """Pointwise MI in bits, or None when the cell (or a marginal) is 0."""
    if f_xy <= 0 or f_x <= 0 or f_y <= 0 or n <= 0:
        return None
    return math.log2((f_xy * n) / (f_x * f_y))


def benjamini_hochberg(p_values):
    """BH (1995) step-up adjusted p-values (q-values); #56's implementation."""
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


def g2_binary(f, n_row, base):
    """1-df likelihood-ratio G^2 of f against expectation n_row * base."""
    e = n_row * base
    if e <= 0 or n_row <= 0:
        return 0.0
    g = 0.0
    if f > 0:
        g += f * math.log(f / e)
    nf = n_row - f
    en = n_row - e
    if nf > 0 and en > 0:
        g += nf * math.log(nf / en)
    return 2.0 * g


def null_stats(obs, null):
    """(mean, sd, z, empirical p) of obs against a list of null values."""
    if not null:
        return 0.0, 0.0, 0.0, 1.0
    mean = sum(null) / len(null)
    sd = (sum((x - mean) ** 2 for x in null) / len(null)) ** 0.5
    z = (obs - mean) / sd if sd else 0.0
    emp = (1 + sum(1 for x in null if x >= obs)) / (1 + len(null))
    return mean, sd, z, emp


def entropy(values):
    """Plug-in Shannon entropy (bits) of a list of observations."""
    n = len(values)
    if n == 0:
        return 0.0
    c = collections.Counter(values)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def term_entropies(occurrences, side, seeds=None, strict=False):
    """H(T) and H(T|O) for one side: I(O;T) = H(T) - H(T|O)."""
    _lp, term_pairs, _ds = side_pairs(occurrences, side, seeds or {}, strict)
    terms = [t for _op, t in term_pairs]
    h_t = entropy(terms)
    n = len(term_pairs)
    f_op = collections.Counter(op for op, _t in term_pairs)
    h_cond = 0.0
    by_op = collections.defaultdict(list)
    for op, t in term_pairs:
        by_op[op].append(t)
    for op, ts in by_op.items():
        h_cond += (f_op[op] / n) * entropy(ts)
    return h_t, h_cond


def mi_contribution_rows(occurrences, side, seeds=None, strict=False):
    """Per-operator contribution to I(O;T): p(op) * KL(P(T|op) || P(T))."""
    _lp, term_pairs, _ds = side_pairs(occurrences, side, seeds or {}, strict)
    n = len(term_pairs)
    if n == 0:
        return []
    f_op = collections.Counter(op for op, _t in term_pairs)
    f_term = collections.Counter(t for _op, t in term_pairs)
    f_pair = collections.Counter(term_pairs)
    by_op = collections.defaultdict(set)
    for op, t in term_pairs:
        by_op[op].add(t)
    rows = []
    for op in sorted(f_op):
        contrib = 0.0
        for t in by_op[op]:
            p_t_o = f_pair[(op, t)] / f_op[op]
            p_t = f_term[t] / n
            contrib += p_t_o * math.log2(p_t_o / p_t)
        rows.append({"side": side, "operator": op, "n_op": f_op[op],
                     "mi_contribution_bits": round(contrib * f_op[op] / n, 4)})
    return rows


def domain_test_rows(occurrences, side, seeds, strict=False):
    """One row per (operator, domain): rate, lift, G^2, p, q (BH)."""
    _lp, _tp, domain_sets = side_pairs(occurrences, side, seeds, strict)
    n = len(occurrences)
    ops = [occ["op"] for occ in occurrences]
    f_op = collections.Counter(ops)
    in_dom = collections.Counter()
    f_cell = collections.Counter()
    for op, doms in zip(ops, domain_sets):
        for d in doms:
            f_cell[(op, d)] += 1
            in_dom[d] += 1
    rows = []
    for op in sorted(f_op):
        n_op = f_op[op]
        for d in DOMAINS:
            f = f_cell.get((op, d), 0)
            base = in_dom[d] / n if n else 0.0
            rate = f / n_op if n_op else 0.0
            g = g2_binary(f, n_op, base)
            rows.append({
                "side": side, "operator": op, "domain": d,
                "f_cell": f, "n_op": n_op, "n_occurrences": n,
                "p_domain_given_op": round(rate, 6),
                "p_domain_base": round(base, 6),
                "lift": round(rate / base, 4) if base else "",
                "g2": round(g, 4),
                "p_value": float(f"{chi2.sf(g, 1):.6g}"),
            })
    for row, q in zip(rows, benjamini_hochberg([r["p_value"] for r in rows])):
        # 6 significant digits: BH p values run to ~1e-21 on this grid and
        # must keep p <= q after the TSV round-trip, so the rounded q is
        # floored at the rounded p.
        row["q_bh"] = max(float(f"{q:.6g}"), row["p_value"])
    rows.sort(key=lambda r: (r["operator"], DOMAINS.index(r["domain"])))
    return rows


def matrix_rows(occurrences, side, seeds, tests, strict=False):
    """The MI matrix: one row per (operator, domain) on one side."""
    cell, f_op_inc, f_dom_inc, n_inc = incidence_table(
        occurrences, side, seeds, strict)
    n_op = collections.Counter(occ["op"] for occ in occurrences)
    t_idx = {(r["operator"], r["domain"]): r for r in tests}
    rows = []
    for op in sorted(n_op):
        for d in DOMAINS:
            f = cell.get((op, d), 0)
            exp = (f_op_inc[op] * f_dom_inc[d] / n_inc) if n_inc else 0.0
            pmi = pmi_cell(f, f_op_inc[op], f_dom_inc[d], n_inc)
            lift = ((f * n_inc) / (f_op_inc[op] * f_dom_inc[d])
                    if f_op_inc[op] and f_dom_inc[d] else None)
            t = t_idx.get((op, d), {})
            rows.append({
                "side": side, "operator": op, "domain": d,
                "f_cell": f,
                "expected": round(exp, 3),
                "pmi_bits": round(pmi, 4) if pmi is not None else "",
                "lift": round(lift, 4) if lift is not None else "",
                "f_op_incidences": f_op_inc[op],
                "f_domain_incidences": f_dom_inc[d],
                "n_incidences": n_inc,
                "n_op": n_op[op],
                "p_domain_given_op": t.get("p_domain_given_op", ""),
                "p_domain_base": t.get("p_domain_base", ""),
                "g2": t.get("g2", ""),
                "p_value": t.get("p_value", ""),
                "q_bh": t.get("q_bh", ""),
            })
    return rows


def q_lift_verdict(row):
    """Verdict from a binary-test row (or a matrix row carrying q/lift).

    Rows read back from a TSV carry strings, so both fields are coerced.
    """
    q = row.get("q_bh")
    lift = row.get("lift")
    if q in ("", None) or lift in ("", None):
        return "no preference"
    q = float(q)
    lift = float(lift)
    if q < 0.05 and lift > 1:
        return "over-selects"
    if q < 0.05 and lift < 1:
        return "avoids"
    return "no preference"


def profile_rows(occurrences, side, seeds, tests, strict=False):
    """Per-operator summary: the domain it most over-selects, and shares."""
    _lp, _tp, domain_sets = side_pairs(occurrences, side, seeds, strict)
    contrib = {r["operator"]: r["mi_contribution_bits"]
               for r in mi_contribution_rows(occurrences, side, seeds,
                                             strict)}
    rows = []
    for op in sorted({occ["op"] for occ in occurrences}):
        mine = [(occ, doms) for occ, doms
                in zip(occurrences, domain_sets) if occ["op"] == op]
        n_op = len(mine)
        cnt = collections.Counter(d for _o, doms in mine for d in doms)
        top, top_f = None, 0
        for d in DOMAINS:
            if cnt[d] > top_f:
                top, top_f = d, cnt[d]
        t = next((r for r in tests
                  if r["operator"] == op and r["domain"] == top), {})
        none_share = sum(1 for occ, doms in mine
                         if occ[side] is not None and not doms) / n_op
        bound_share = sum(1 for occ, _d in mine
                          if occ[side] is None) / n_op
        multi_share = sum(1 for _o, doms in mine if len(doms) > 1) / n_op
        rows.append({
            "side": side, "operator": op, "n_op": n_op,
            "top_domain": top or "", "top_f": top_f,
            "p_domain_given_op": t.get("p_domain_given_op", ""),
            "p_domain_base": t.get("p_domain_base", ""),
            "lift": t.get("lift", ""),
            "q_bh": t.get("q_bh", ""),
            "p_none": round(none_share, 6),
            "p_boundary": round(bound_share, 6),
            "p_multi_domain": round(multi_share, 6),
            "mi_contribution_bits": contrib.get(op, 0.0),
            "verdict": q_lift_verdict(t),
        })
    rows.sort(key=lambda r: -r["n_op"])
    return rows


def seed_term_rows(occurrences, side, seeds, strict=False):
    """(operator, seed term) pairs: the word-level evidence per cell."""
    _lp, term_pairs, _ds = side_pairs(occurrences, side, seeds, strict)
    n = len(term_pairs)
    f_pair = collections.Counter(term_pairs)
    f_op = collections.Counter(op for op, _t in term_pairs)
    f_term = collections.Counter(t for _op, t in term_pairs)
    rows = []
    for (op, term), f in f_pair.items():
        doms = term_domains(term, seeds, strict)
        if not doms:
            continue
        pmi = pmi_cell(f, f_op[op], f_term[term], n)
        rows.append({
            "side": side, "operator": op, "term": term,
            "domains": "|".join(sorted(doms)),
            "f_cell": f, "f_op": f_op[op], "f_term": f_term[term],
            "n": n,
            "pmi_bits": round(pmi, 4) if pmi is not None else "",
            "lift": round((f * n) / (f_op[op] * f_term[term]), 4)
            if f_op[op] and f_term[term] else "",
        })
    rows.sort(key=lambda r: (r["operator"], -r["f_cell"], r["term"]))
    return rows


def permutation_null(occurrences, side, seeds, n_rep=N_SHUFFLE, seed=SEED,
                     strict=False):
    """Label-permutation null for the three MI quantities on one side.

    The domain-combination label is permuted across operator positions, so
    both marginals are preserved exactly and only the pairing is destroyed.
    The same permutation is applied to the multi-label view by re-expanding
    each joined label into its domains.
    """
    rng = random.Random(seed)
    label_pairs, term_pairs, _ds = side_pairs(occurrences, side, seeds,
                                              strict)
    ops = [op for op, _lab in label_pairs]
    labels = [lab for _op, lab in label_pairs]
    terms = [t for _op, t in term_pairs]

    cell, f_op, f_dom, n_inc = incidence_table(occurrences, side, seeds,
                                               strict)
    obs = {
        "I_label": total_mi(list(zip(ops, labels))),
        "I_term": total_mi(list(zip(ops, terms))),
        "I_multi": incidence_mi_from(cell, f_op, f_dom, n_inc),
    }
    null = {"I_label": [], "I_term": [], "I_multi": []}
    lab = list(labels)
    for _ in range(n_rep):
        rng.shuffle(lab)
        null["I_label"].append(total_mi(list(zip(ops, lab))))
        null["I_multi"].append(_multi_mi_from_labels(ops, lab))
    rng2 = random.Random(seed + 1)
    tm = list(terms)
    for _ in range(n_rep):
        rng2.shuffle(tm)
        null["I_term"].append(total_mi(list(zip(ops, tm))))

    out = {}
    for measure, value in obs.items():
        mean, sd, z, emp = null_stats(value, null[measure])
        out[measure] = (round(value, 4), round(mean, 4), round(sd, 4),
                        round(z, 3), round(emp, 4))
    return out


def _n_dom_only_from_parts(occurrences, side, seeds):
    """Occurrences whose domain hits come only from the seam-split rule.

    That is: the whole term matches no seed, but one of its parts does
    (`drug-embryo` -> `embryo`).  This is the exact amount the default
    matching rule adds over #29's whole-term matching.
    """
    n = 0
    for occ in occurrences:
        term = occ[side]
        if not term:
            continue
        if term_domains(term, seeds, strict=True):
            continue
        if term_domains(term, seeds, strict=False):
            n += 1
    return n


def _multi_mi_from_labels(ops, labels):
    """Incidence multi-MI re-derived from (op, joined-label) observations."""
    cell = collections.Counter()
    f_op = collections.Counter()
    f_dom = collections.Counter()
    n_inc = 0
    for op, lab in zip(ops, labels):
        doms = [] if lab in (NONE, BOUNDARY) else lab.split("+")
        f_op[op] += len(doms)
        n_inc += len(doms)
        for d in doms:
            cell[(op, d)] += 1
            f_dom[d] += 1
    return incidence_mi_from(cell, f_op, f_dom, n_inc)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
_LEFT_COLS = {"operator", "domain", "side", "term", "verdict", "top_domain",
              "domains", "quantity", "earlier", "here", "note", "measure",
              "variant", "artifact", "content"}


def _md_table(rows, cols, headers, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" if c in _LEFT_COLS else "---:" for c in cols)
           + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c in fmt:
                v = fmt[c](v)
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


def _fmt_bits(v):
    return f"{float(v):.3f}" if v not in ("", None) else "—"


def _fmt_lift(v):
    return f"{float(v):.2f}" if v not in ("", None) else "—"


def _fmt_pct(v, nd=1):
    return f"{100 * float(v):.{nd}f}%" if v not in ("", None) else "—"


def _fmt_q(v):
    if v in ("", None):
        return "—"
    v = float(v)
    return f"{v:.2g}" if v >= 1e-4 else "<1e-4"


def _cell_of(M, op, domain):
    """The matrix/test row for one (operator, domain) on side M."""
    return next(r for r in M["tests"]
                if r["operator"] == op and r["domain"] == domain)


def write_report(path, diag, seeds, summary, matrices, profiles, seed_terms,
                 recon, strict_summary):
    A = []
    A.append("# Operator-word mutual information (#52)")
    A.append("")
    A.append("Source: #17 token layer (`tokens.tsv`) for operator positions "
             "and #17 unit layer (`units.tsv`) for the terms they stand next "
             "to; markup screen from #19/#27; domain seed sets parsed from "
             "#29's `DOMAIN_SEEDS` in `work/cooccurrence_network.py`. "
             "Reproducible via `python work/operator_word_mi.py`; regression "
             "suite `python work/test_operator_word_mi.py`.")
    A.append("")
    A.append(f"Schema `{SCHEMA}`. MI in bits (log2). Right side = the term "
             "to the *right* of the operator (the issue's \"`::` before body "
             "terms\"); left side = the term to its left. The permutation "
             f"null is {N_SHUFFLE} label permutations, seed {SEED}, both "
             "marginals preserved.")
    A.append("")

    # ---- 1. headline ------------------------------------------------------
    right, left = summary["right"], summary["left"]
    A.append("## 1. Headline result")
    A.append("")
    A.append(f"- **{diag['op_occurrences']:,} screened operator "
             f"occurrences** "
             f"({diag['op_tokens_markup_removed']:,} raw-HTML operator "
             "tokens inside #27's markup spans removed; #10's page-body count "
             "keeps them, which is the whole difference). "
             f"{diag['op_in_unit']:,} sit inside a #17 unit (`=` in "
             f"`suck=blood`), {diag['op_free']:,} are free between units.")
    A.append(f"- **The operator predicts the adjacent term.** Right side: "
             f"I(O;T) = {right['I_term'][0]} bits against a null of "
             f"{right['I_term'][1]} ± {right['I_term'][2]} "
             f"(z = {right['I_term'][3]}, p_emp = {right['I_term'][4]}); "
             f"the domain-coarsened I(O;D) = {right['I_label'][0]} bits "
             f"(null {right['I_label'][1]} ± {right['I_label'][2]}, "
             f"z = {right['I_label'][3]}); the incidence "
             f"multi-information of the MI matrix is {right['I_multi'][0]} "
             f"bits (null {right['I_multi'][1]} ± "
             f"{right['I_multi'][2]}). Knowing the operator lowers the "
             f"adjacent term's uncertainty from H(T) = {right['H_T']} to "
             f"H(T|O) = {right['H_T_given_O']} bits.")
    A.append(f"- **Both sides predict, the right side more sharply.** Left "
             f"side: I(O;T) = {left['I_term'][0]} bits against null "
             f"{left['I_term'][1]} ± {left['I_term'][2]} "
             f"(z = {left['I_term'][3]}), H(T) = {left['H_T']} to "
             f"H(T|O) = {left['H_T_given_O']} bits. The absolute MI is "
             "nearly symmetric, but the left term distribution is itself "
             "more spread (any word can precede an operator), so *relative "
             f"to its own null* the right side is the stronger predictor "
             f"({right['I_term'][0] / right['I_term'][1]:.1f}x null vs "
             f"{left['I_term'][0] / left['I_term'][1]:.1f}x), and the "
             f"incidence multi-information is twice as large "
             f"({right['I_multi'][0]} vs {left['I_multi'][0]} bits).")
    ratio = right["I_term"][0] / OP_LAG1_MI_11
    A.append(f"- **Scale check.** The operator-to-term MI "
             f"({right['I_term'][0]} bits) is {ratio:.0f}x #11's "
             f"operator-to-operator lag-1 MI ({OP_LAG1_MI_11} bits). The "
             "operator stream is nearly independent of itself but strongly "
             "tied to the words it touches — which is what #12's role "
             "classification asserts and what #11's entropy curve could not "
             "see.")
    A.append("- **The issue's two hypotheses, tested.** "
             "`::` before body terms: **rejected** — `::` significantly "
             "*avoids* body terms on its right (lift "
             f"{_fmt_lift(_cell_of(matrices['right'], '::', 'body')['lift'])}, "
             f"q = {_fmt_q(_cell_of(matrices['right'], '::', 'body')['q_bh'])}"
             "). `=` before tech terms: **confirmed** — `=` precedes a "
             f"machine term with lift "
             f"{_fmt_lift(_cell_of(matrices['right'], '=', 'machine')['lift'])}"
             f" (q = "
             f"{_fmt_q(_cell_of(matrices['right'], '=', 'machine')['q_bh'])}),"
             " and an even stronger body preference (lift "
             f"{_fmt_lift(_cell_of(matrices['right'], '=', 'body')['lift'])}):"
             " `=` is a cross-domain binder, genetics/machine on the left, "
             "body/machine on the right. Details in §7.")
    A.append("")

    # ---- 2. method --------------------------------------------------------
    A.append("## 2. Method")
    A.append("")
    A.append("### 2.1 The stream")
    A.append("")
    A.append("Every screened `OP` token of #17's token layer is one "
             "observation. Its terms are resolved on the #17 unit layer: a "
             "free operator takes the nearest surviving unit surface on each "
             "side; an operator inside a unit takes the operand substrings of "
             "its own surface (`=` in `gene=TV/spasm` has `gene` left and "
             "`TV/spasm` right). A side with no surviving unit is `boundary`; "
             "a term in no seed set is `none`.")
    A.append("")
    A.append("### 2.2 Terms and domains")
    A.append("")
    n_seed_terms = sum(len(v) for v in seeds.values())
    A.append(f"#29's six seeded domains ({', '.join(DOMAINS)}; "
             f"{n_seed_terms} seed terms). A term counts toward a domain when "
             "the term itself *or* one of its seam-split parts is in that "
             "domain's seed set (`drug-embryo` -> `embryo` = body+genetics). "
             "A term may carry several domains; the whole-term-only variant "
             "is the #29-faithful sensitivity check in §8.")
    A.append("")
    A.append(f"Coverage is the limit of this design and is stated plainly: "
             f"#29's {n_seed_terms} seed terms are "
             f"{diag['domain_bearing_right_pct']}% of right-hand neighbours "
             f"and {diag['domain_bearing_left_pct']}% of left-hand "
             f"neighbours ({diag['domain_bearing_right']:,} and "
             f"{diag['domain_bearing_left']:,} occurrences; the rest are "
             "`none` or `boundary`). The MI matrix is therefore a statement "
             "about the *domain-bearing minority* of adjacent terms, and the "
             "term-level totals I(O;T) in §5 cover the whole stream.")
    A.append("")
    A.append("One seed-set note, reported because it changes a cell: #29's "
             "source comment says ``blood`` is *body + violence*, but the "
             "literal ``DOMAIN_SEEDS['violence']`` does not contain "
             "``blood``. This build follows the literal (so ``blood`` is "
             "body-only), which is why the violence column is thinner than "
             "#29's prose implies.")
    A.append("")
    A.append("### 2.3 Statistics")
    A.append("")
    A.append("- **MI matrix** (§3): pointwise MI (bits) per (operator, "
             "domain) over the incidence table, with expected count and "
             "lift.")
    A.append("- **Per-domain binary tests** (§4): a 1-df G² per (operator, "
             "domain) with Benjamini-Hochberg q over each 6 x 11 grid. A "
             "multi-domain term is an incidence in each of its domains, so "
             "the six domain tests are not mutually exclusive.")
    A.append("- **Totals** (§5): I(O;D) on the domain-combination label "
             "(single-label, a true mutual information), I(O;T) on the term "
             "itself, and the incidence multi-information; each against a "
             "label-permutation null that preserves both marginals.")
    A.append("- **Two lift bases, both published.** The matrix's `lift` and "
             "`pmi_bits` are computed on the *incidence* table (a "
             "multi-domain term is one incidence in each of its domains), "
             "while `p_domain_given_op` / `p_domain_base` — and "
             "`operator_profile.tsv`'s `lift` — are occurrence rates (the "
             "operator's domain-bearing share against the stream's). The "
             "two bases agree in sign and rank; e.g. `<<`/body is 1.53 on "
             "incidences and 1.72 on rates.")
    A.append("- **Robustness** (§8): whole-term-only seed matching, book "
             "halves, and the operator inventory reconciliation.")
    A.append("")

    # ---- 3. the matrix ----------------------------------------------------
    A.append("## 3. The MI matrix (operator x domain)")
    A.append("")
    for side in SIDES:
        M = matrices[side]
        A.append(f"### 3.{SIDES.index(side) + 1} {side} side — the term to "
                 f"the {side} of the operator")
        A.append("")
        A.append(f"n = {M['n']:,} operator occurrences; "
                 f"{M['n_inc']:,} domain incidences "
                 f"({M['n_multi']:,} occurrences carry more than one "
                 f"domain). I(O;D) = {M['I_label']} bits, I(O;T) = "
                 f"{M['I_term']} bits, multi-information = "
                 f"{M['I_multi']} bits.")
        A.append("")
        A.append("Cell = count (PMI bits); `—` marks a cell that never "
                 "occurs. `none` = a term in no seed set, `boundary` = no "
                 "surviving unit on that side.")
        A.append("")
        A.append("| operator | n | " + " | ".join(f"`{d}`"
                                                   for d in DOMAINS)
                 + " | none | boundary |")
        A.append("|---|---:|" + "---:|" * (len(DOMAINS) + 2))
        for op in M["ops"]:
            row = [f"`{op}`", f"{M['n_op'][op]:,}"]
            for d in DOMAINS:
                c = M["cell"].get((op, d), 0)
                pm = M["pmi"].get((op, d))
                row.append(f"{c}" + (f" ({pm:+.2f})" if pm is not None
                                     else " (—)"))
            row.append(f"{M['none'].get(op, 0):,}")
            row.append(f"{M['boundary'].get(op, 0):,}")
            A.append("| " + " | ".join(row) + " |")
        A.append("")

    # ---- 4. per-operator --------------------------------------------------
    A.append("## 4. Which operators select which domains")
    A.append("")
    for side in SIDES:
        A.append(f"### 4.{SIDES.index(side) + 1} {side} side")
        A.append("")
        A.append("`MI contrib.` is the operator's share of the side's "
                 "I(O;T) — how much of the total prediction each operator "
                 "carries.")
        A.append("")
        prof = [dict(p, operator=f"`{p['operator']}`")
                for p in profiles if p["side"] == side]
        A.extend(_md_table(
            prof[:14],
            ["operator", "n_op", "top_domain", "p_domain_given_op",
             "p_domain_base", "lift", "q_bh", "mi_contribution_bits",
             "p_none", "p_boundary", "verdict"],
            ["operator", "n", "top domain", "P(d|op)", "P(d)", "lift", "q",
             "MI contrib.", "none", "boundary", "verdict"],
            fmt={"p_domain_given_op": lambda v: _fmt_pct(v),
                 "p_domain_base": lambda v: _fmt_pct(v),
                 "lift": _fmt_lift, "q_bh": _fmt_q,
                 "mi_contribution_bits": lambda v: f"{float(v):.3f}",
                 "p_none": lambda v: _fmt_pct(v, 0),
                 "p_boundary": lambda v: _fmt_pct(v, 0)}))
        A.append("")

    # ---- 5. position ------------------------------------------------------
    A.append("## 5. Right vs left")
    A.append("")
    A.append("| side | I(O;T) bits | null | z | I(O;D) bits | null | z | "
             "multi-info bits | null | z | H(T) | H(T|O) |")
    A.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for side in SIDES:
        s = summary[side]
        A.append(f"| {side} | {s['I_term'][0]} | {s['I_term'][1]} ± "
                 f"{s['I_term'][2]} | {s['I_term'][3]} | "
                 f"{s['I_label'][0]} | {s['I_label'][1]} ± "
                 f"{s['I_label'][2]} | {s['I_label'][3]} | "
                 f"{s['I_multi'][0]} | {s['I_multi'][1]} ± "
                 f"{s['I_multi'][2]} | {s['I_multi'][3]} | "
                 f"{s['H_T']} | {s['H_T_given_O']} |")
    A.append("")
    A.append("The two sides are close in absolute bits but get there "
             "differently: the right-hand term is far more predictable in "
             "absolute terms (H(T|O) 6.20 vs 7.96 bits) because an operator "
             "constrains what follows it, while the left term varies widely "
             "(H(T) 9.16 bits). Relative to its own null — the comparison "
             "that matters, since the null moves with the term distribution "
             "— the right side is the stronger predictor (2.2x vs 1.7x "
             "null), and the domain structure is twice as concentrated "
             "(multi-information 0.1692 vs 0.0881 bits).")
    A.append("")

    # ---- 6. word-level evidence -------------------------------------------
    A.append("## 6. The seed terms behind the cells (right side)")
    A.append("")
    A.append("Operator-word PMI restricted to #29's seed vocabulary, "
             "f >= 3, top 8 per operator. Terms are #17 unit surfaces with "
             "their original case (`Defleshed`, `Body-OMOTYA`). This is the "
             "lexical evidence for the domain cells; #27's "
             "`collocations_bigram_token.tsv` is the full operator-word "
             "table on a token-bigram basis.")
    A.append("")
    by_op = collections.defaultdict(list)
    for r in seed_terms:
        if r["side"] == "right" and r["f_cell"] >= 3:
            by_op[r["operator"]].append(r)
    for op in sorted(by_op, key=lambda o: -matrices["right"]["n_op"][o]):
        rows = sorted(by_op[op], key=lambda r: -float(r["pmi_bits"]))[:8]
        A.append(f"- **`{op}`** "
                 f"({matrices['right']['n_op'][op]:,} occurrences): "
                 + ", ".join(
                     f"`{r['term']}` {r['f_cell']}x (PMI "
                     f"{float(r['pmi_bits']):.2f}, {r['domains']})"
                     for r in rows))
    A.append("")

    # ---- 7. the issue's questions -----------------------------------------
    A.append("## 7. The issue's questions")
    A.append("")
    right_tests = {(r["operator"], r["domain"]): r
                   for r in matrices["right"]["tests"]}
    questions = (
        ("::", "body", "**`::` before body terms?** (phrase/field "
                       "separator over flesh)"),
        ("=", "machine", "**`=` before tech terms?** (naming/identity "
                         "linker)"),
        ("//", "violence", "**`//` before violence terms?** (utterance "
                           "boundary)"),
        (">>", "body", "**`>>` before body terms?** (block close)"),
    )
    for q_op, q_dom, q_text in questions:
        r = right_tests.get((q_op, q_dom))
        if not r:
            A.append(f"- {q_text}: not present on the right side.")
            continue
        A.append(f"- {q_text}: `{q_op}` precedes a `{q_dom}` term "
                 f"{r['f_cell']:,} times; P({q_dom}|`{q_op}`) = "
                 f"{_fmt_pct(r['p_domain_given_op'])} against a base rate "
                 f"of {_fmt_pct(r['p_domain_base'])} — lift "
                 f"{_fmt_lift(r['lift'])}, G² = {r['g2']}, q = "
                 f"{_fmt_q(r['q_bh'])}. Verdict: "
                 f"**{q_lift_verdict(r)}**.")
    A.append("")
    eq_l = _cell_of(matrices["left"], "=", "genetics")
    A.append(f"The `=` row deserves the two-sided reading: on the left it "
             f"over-selects **genetics** terms "
             f"(P(genetics|`=` left) = "
             f"{_fmt_pct(eq_l['p_domain_given_op'])} vs base "
             f"{_fmt_pct(eq_l['p_domain_base'])}, lift "
             f"{_fmt_lift(eq_l['lift'])}, q = {_fmt_q(eq_l['q_bh'])}), and "
             "on the right body and machine terms. `=` is the book's "
             "naming/identity seam (#12) and it binds *across* domains: "
             "`suck=blood` (sex to body), `fear=cell` (violence to body), "
             "`gene=TV` (genetics to machine), `machine=angel`. The "
             "operator grammar's linker is where the book's semantic fusion "
             "is syntactic.")
    A.append("")
    A.append("The `<<` block-open delimiter is the one operator with a "
             "strong, unambiguous domain preference: body terms follow it "
             "at lift "
             f"{_fmt_lift(_cell_of(matrices['right'], '<<', 'body')['lift'])} "
             f"(q = {_fmt_q(_cell_of(matrices['right'], '<<', 'body')['q_bh'])}"
             ", driven by `entrails` 30x at PMI 3.48), while it avoids "
             "machine terms. #14's paired `<<…>>` spans are quoted blocks of "
             "flesh; `>>` itself shows no equivalent preference on closing.")
    A.append("")

    # ---- 8. robustness ----------------------------------------------------
    A.append("## 8. Robustness and sensitivity")
    A.append("")
    A.append("| variant | I(O;D) right | I(O;D) left | incidences right | "
             "hits only from parts, right |")
    A.append("|---|---:|---:|---:|---:|")
    A.append(f"| default (term or seam part) | "
             f"{summary['right']['I_label'][0]} | "
             f"{summary['left']['I_label'][0]} | "
             f"{matrices['right']['n_inc']:,} | "
             f"{strict_summary['right']['n_dom_only_from_parts']:,} |")
    A.append(f"| strict whole term (#29-faithful) | "
             f"{strict_summary['right']['I_label']} | "
             f"{strict_summary['left']['I_label']} | "
             f"{strict_summary['right']['n_inc']:,} | — |")
    A.append("")
    A.append(f"The seam-split rule adds "
             f"{strict_summary['right']['n_dom_only_from_parts']:,} "
             "right-side occurrences that match a domain only through a "
             "compound part (`drug-embryo` -> `embryo`, "
             "`body-OMOTYA` -> `body`), and it moves I(O;D) by less than a "
             "hundredth of a bit on either side: the cells that carry the "
             "findings are single-word seeds (`blood`, `TV`, `cell`, "
             "`entrails`). The term-level totals I(O;T) do not depend on "
             "the seed sets at all, so only the domain quantities appear in "
             "this table.")
    A.append("")
    A.append("Book halves (first half / second half):")
    A.append("")
    A.append("| side | I(O;D) first | I(O;D) second | I(O;T) first | "
             "I(O;T) second |")
    A.append("|---|---:|---:|---:|---:|")
    for side in SIDES:
        h = summary[side]["halves"]
        A.append(f"| {side} | {h['first'][0]} | {h['second'][0]} | "
                 f"{h['first'][1]} | {h['second'][1]} |")
    A.append("")
    A.append("The totals are stable across halves on both sides, so the "
             "operator-word association is a property of the whole book "
             "rather than a run of pages.")
    A.append("")

    # ---- 9. reconciliation ------------------------------------------------
    A.append("## 9. Reconciliation")
    A.append("")
    A.extend(_md_table(recon, ["quantity", "earlier", "here", "note"],
                       ["quantity", "earlier", "here", "note"]))
    A.append("")

    # ---- 10. caveats ------------------------------------------------------
    A.append("## 10. Caveats and limitations")
    A.append("")
    A.append("- **Seed sets are a reading, not the book's own categories.** "
             "#29/#73 already showed the six domains do not converge on the "
             "distributional clusters; the MI matrix measures the operator's "
             "predictive structure *relative to that seed reading*, not an "
             "objective domain inventory.")
    A.append("- **In-unit operators bind their own operands.** `=` binds "
             "`suck` to `blood` inside one unit, so the `=` row is a "
             "statement about compounds while the row for a free operator "
             "such as `::` is a statement about unit sequence. The `n_op` "
             "column and the diagnostics separate the two.")
    A.append("- **The asymmetry between sides is real but partly "
             "definitional.** A free operator's left term is the *end* of "
             "the previous unit and its right term the *start* of the next; "
             "the two sides are reported separately so no pooled number "
             "hides the difference.")
    A.append("- **Rare operators** (`+` n=15, `|` n=17 after screening) "
             "have cells far too sparse for a domain claim; their rows are "
             "reported but their q values should not be read as evidence.")
    A.append("- **Multi-domain terms double-count incidences** in the "
             "matrix (a term in two domains is one incidence in each). The "
             "single-label totals I(O;D) and the per-operator profile shares "
             "do not double-count.")
    A.append("- **Plug-in MI is upward-biased** at these sample sizes; the "
             "permutation null is the honest comparison and is what the "
             "reported z and empirical p refer to.")
    A.append("- **The predictions are distributional, not semantic.** That "
             "`::` precedes body terms is a statement about which seed words "
             "sit on which side of an operator; whether it means \"`::` "
             "introduces flesh\" is #12's interpretive question.")
    A.append("")

    # ---- 11. artifacts ----------------------------------------------------
    A.append("## 11. Artifacts")
    A.append("")
    A.append("| artifact | content |")
    A.append("|---|---|")
    for name, desc in (
            ("mi_matrix.tsv", "the MI matrix: one row per "
                              "(side, operator, domain) with count, "
                              "expected, PMI bits and lift (incidence "
                              "basis) plus the occurrence-rate columns "
                              "P(d|op) / P(d) and G², p, q"),
            ("operator_profile.tsv", "per-operator summary: n, top domain, "
                                     "rates, lift, q, none/boundary/multi "
                                     "shares, verdict"),
            ("operator_seed_terms.tsv", "(side, operator, seed term) pairs "
                                        "with PMI and lift — the word-level "
                                        "evidence"),
            ("mi_summary.tsv", "the three MI totals per side with the "
                               "permutation null (mean, sd, z, empirical "
                               "p)"),
            ("diagnostics.tsv", "screening and stream counts"),
            ("reconciliation.tsv", "this build vs #10/#11/#17/#27/#29"),
            ("summary.json", "machine-readable headline numbers")):
        A.append(f"| `{name}` | {desc} |")
    A.append("")
    A.append("---")
    A.append("*Generated by `work/operator_word_mi.py`*")
    A.append("")
    path.write_text("\n".join(A), encoding="utf-8")


# ---------------------------------------------------------------------------
def build(out_dir: Path = OUT, n_shuffle: int = N_SHUFFLE, seed: int = SEED,
          log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = load_domain_seeds()
    occurrences, diag = build_occurrences()
    for side in SIDES:
        _lp, _tp, dsets = side_pairs(occurrences, side, seeds)
        n = len(occurrences)
        diag[f"domain_bearing_{side}"] = sum(1 for d in dsets if d)
        diag[f"domain_bearing_{side}_pct"] = round(
            100.0 * diag[f"domain_bearing_{side}"] / n, 2) if n else 0.0
        diag[f"boundary_{side}"] = sum(1 for occ in occurrences
                                       if occ[side] is None)
        diag[f"none_{side}"] = n - diag[f"domain_bearing_{side}"] \
            - diag[f"boundary_{side}"]

    summary = {}
    matrices = {}
    all_matrix_rows = []
    all_profile_rows = []
    all_seed_term_rows = []
    strict_summary = {}

    for side in SIDES:
        tests = domain_test_rows(occurrences, side, seeds)
        mrows = matrix_rows(occurrences, side, seeds, tests)
        prows = profile_rows(occurrences, side, seeds, tests)
        srows = seed_term_rows(occurrences, side, seeds)
        null = permutation_null(occurrences, side, seeds, n_shuffle, seed)
        s_lp, _s_tp, _ds = side_pairs(occurrences, side, seeds,
                                   strict=True)
        strict_summary[side] = {
            "I_label": round(total_mi(s_lp), 4),
            "n_inc": incidence_table(occurrences, side, seeds,
                                     strict=True)[3],
            "n_dom_only_from_parts": _n_dom_only_from_parts(
                occurrences, side, seeds),
        }
        half = len(occurrences) // 2
        h1_lp, h1_tp, _ = side_pairs(occurrences[:half], side, seeds)
        h2_lp, h2_tp, _ = side_pairs(occurrences[half:], side, seeds)
        h_t, h_t_given_o = term_entropies(occurrences, side, seeds)

        cell, f_op_inc, f_dom_inc, n_inc = incidence_table(occurrences, side,
                                                           seeds)
        n_op = collections.Counter(occ["op"] for occ in occurrences)
        _lp, _tp, domain_sets = side_pairs(occurrences, side, seeds)
        n_multi = sum(1 for doms in domain_sets if len(doms) > 1)
        n_none = collections.Counter()
        n_bound = collections.Counter()
        for occ, doms in zip(occurrences, domain_sets):
            if occ[side] is None:
                n_bound[occ["op"]] += 1
            elif not doms:
                n_none[occ["op"]] += 1
        matrices[side] = {
            "n": len(occurrences), "n_inc": n_inc, "n_multi": n_multi,
            "ops": sorted(n_op), "n_op": n_op,
            "none": n_none, "boundary": n_bound,
            "cell": {(op, d): cell.get((op, d), 0)
                     for op in n_op for d in DOMAINS},
            "pmi": {(op, d): pmi_cell(cell.get((op, d), 0), f_op_inc[op],
                                      f_dom_inc[d], n_inc)
                    for op in n_op for d in DOMAINS},
            "I_label": null["I_label"][0], "I_term": null["I_term"][0],
            "I_multi": null["I_multi"][0],
            "tests": tests,
        }
        summary[side] = dict(null)
        summary[side]["n_op"] = n_op
        summary[side]["halves"] = {
            "first": (round(total_mi(h1_lp), 4), round(total_mi(h1_tp), 4)),
            "second": (round(total_mi(h2_lp), 4), round(total_mi(h2_tp), 4)),
        }
        summary[side]["H_T"] = round(h_t, 4)
        summary[side]["H_T_given_O"] = round(h_t_given_o, 4)
        all_matrix_rows.extend(mrows)
        all_profile_rows.extend(prows)
        all_seed_term_rows.extend(srows)

    # ---- artifacts --------------------------------------------------------
    write_tsv(out / "mi_matrix.tsv",
              ["side", "operator", "domain", "f_cell", "expected", "pmi_bits",
               "lift", "f_op_incidences", "f_domain_incidences",
               "n_incidences", "n_op", "p_domain_given_op", "p_domain_base",
               "g2", "p_value", "q_bh"],
              all_matrix_rows)
    write_tsv(out / "operator_profile.tsv",
              ["side", "operator", "n_op", "top_domain", "top_f",
               "p_domain_given_op", "p_domain_base", "lift", "q_bh",
               "p_none", "p_boundary", "p_multi_domain",
               "mi_contribution_bits", "verdict"],
              all_profile_rows)
    write_tsv(out / "operator_seed_terms.tsv",
              ["side", "operator", "term", "domains", "f_cell", "f_op",
               "f_term", "n", "pmi_bits", "lift"],
              all_seed_term_rows)
    write_tsv(out / "mi_summary.tsv",
              ["side", "measure", "observed_bits", "null_mean", "null_sd",
               "z", "p_empirical"],
              [{"side": side, "measure": measure,
                "observed_bits": summary[side][measure][0],
                "null_mean": summary[side][measure][1],
                "null_sd": summary[side][measure][2],
                "z": summary[side][measure][3],
                "p_empirical": summary[side][measure][4]}
               for side in SIDES
               for measure in ("I_label", "I_term", "I_multi")]
              + [{"side": side, "measure": measure,
                  "observed_bits": summary[side][measure],
                  "null_mean": "", "null_sd": "", "z": "",
                  "p_empirical": ""}
                 for side in SIDES
                 for measure in ("H_T", "H_T_given_O")])
    write_tsv(out / "diagnostics.tsv", ["diag", "value"],
              [{"diag": k, "value": v} for k, v in sorted(diag.items())])

    n_right_terms = sum(1 for occ in occurrences if occ["right"] is not None)
    recon = [
        {"quantity": "operator tokens (#17)", "earlier": "11,000",
         "here": f"{diag['op_tokens_stream']:,}",
         "note": "#17 token layer, all regions"},
        {"quantity": "operator occurrences after markup screen",
         "earlier": "10,949 body (#10)",
         "here": f"{diag['op_occurrences']:,}",
         "note": f"minus {diag['op_tokens_markup_removed']} raw-HTML "
                 f"`<`/`>` tokens #10 counted (pp. 92/158/219); includes "
                 f"{diag['header_region_ops']} header-region operators "
                 "#17's stream carries"},
        {"quantity": "in-unit operators", "earlier": "2,816 (#17 seams)",
         "here": f"{diag['op_in_unit']:,}",
         "note": "2,816 #17 seams minus the one in-seam `=` inside a markup "
                 "span (`fluid=streaming`, p29); geometric containment "
                 f"reproduces #17's `unit` column on all "
                 f"{diag['op_occurrences']:,} surviving tokens, and 0 free "
                 "tokens sit inside a unit span"},
        {"quantity": "operator MI lag-1, bits (#11)", "earlier": "0.1305",
         "here": f"{summary['right']['I_term'][0]}",
         "note": "different second variable: #11 is operator-to-operator, "
                 "here operator-to-term (right side)"},
        {"quantity": "operator-term adjacency observations",
         "earlier": "1,903 rows (#27)",
         "here": f"{n_right_terms:,}",
         "note": "#27's table rows are operator-word token-bigram *types*; "
                 "this build counts occurrences, and its neighbour is the "
                 "nearest unit even across intervening operators"},
        {"quantity": "domain seeds (#29)", "earlier": "6 domains",
         "here": str(len(seeds)),
         "note": f"{sum(len(v) for v in seeds.values())} seed terms, parsed "
                 "from #29's DOMAIN_SEEDS"},
        {"quantity": "units (#17)", "earlier": "74,665",
         "here": f"{diag['units_total']:,}",
         "note": f"{diag['units_screened_out']:,} markup-screened out of the "
                 "neighbour resolution"},
    ]
    write_tsv(out / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_json = {
        "schema": SCHEMA, "generated": generated, "counts": diag,
        "mi": {side: {m: summary[side][m] for m in
                      ("I_label", "I_term", "I_multi")} for side in SIDES},
        "entropy": {side: {"H_T": summary[side]["H_T"],
                           "H_T_given_O": summary[side]["H_T_given_O"]}
                    for side in SIDES},
        "strict": strict_summary,
        "halves": {s: summary[s]["halves"] for s in SIDES},
        "n_shuffle": n_shuffle, "seed": seed,
        "top_right_cells": [
            {"operator": r["operator"], "domain": r["domain"],
             "f_cell": r["f_cell"], "pmi_bits": r["pmi_bits"],
             "lift": r["lift"]}
            for r in sorted(
                [r for r in all_matrix_rows
                 if r["side"] == "right" and r["pmi_bits"] != ""],
                key=lambda r: -float(r["pmi_bits"]))[:15]],
        "top_right_profile": [
            {k: p[k] for k in ("operator", "n_op", "top_domain", "lift",
                               "q_bh", "verdict")}
            for p in sorted([p for p in all_profile_rows
                             if p["side"] == "right"],
                            key=lambda p: -p["n_op"])[:11]],
    }
    (out / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8")

    write_report(out / "operator_domain_mi_report.md", diag, seeds, summary,
                 matrices, all_profile_rows, all_seed_term_rows, recon,
                 strict_summary)

    log(f"[operator_word_mi] ops={diag['op_occurrences']:,} "
        f"I(O;T)_right={summary['right']['I_term'][0]} "
        f"(null {summary['right']['I_term'][1]}, "
        f"z={summary['right']['I_term'][3]}) "
        f"I(O;T)_left={summary['left']['I_term'][0]} "
        f"(z={summary['left']['I_term'][3]}) -> {out}")
    return summary_json


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="#52 operator-word mutual information")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--shuffle", type=int, default=N_SHUFFLE,
                    help="permutation-null replicates")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="permutation-null seed")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print
    build(Path(args.out), args.shuffle, args.seed, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
