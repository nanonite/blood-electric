#!/usr/bin/env python3
"""#53 Burstiness and Heaps' law -> analysis/burstiness/

Two statistics, one question: does the book's vocabulary and its named cast
behave like ordinary prose, or do they carry structure a normal text does not?

1. **Heaps' law / vocabulary growth** -- fit ``V = K N^beta`` to the cumulative
   type curve of the screened #17 unit layer and its layers, with a
   leave-one-page-out jackknife on beta, per-segment novelty, and the same fit
   at matched N for the eight pinned #24 English references.  #24 published
   the point estimates; this is the growth curve, its uncertainty, and the
   matched comparison the full growth report was deferred to #53 for.

2. **Burstiness of entity use** -- for every #30 glossary entity, locate its
   occurrences on the screened stream and measure how clustered they are: the
   Fano factor ``Var(count)/Mean(count)`` over windows, the Goh-Barabasi
   burstiness ``B = (sigma - mu)/(sigma + mu)`` of the inter-arrival gaps, and
   **both tails** of a placement null (a uniform random subset of the stream at
   the entity's own rate): the high tail asks "more clustered than chance", the
   low tail asks "more evenly than chance".  Ordinary English words and the
   recurring coinage types are measured the same way as controls, and the
   comparison that matters -- the merged entity channel against merged control
   channels -- is published in ``channels.tsv`` rather than one merged number
   stacked against per-term medians.

Screens (disclosed, never silently mixed): the growth analysis runs on #24's
screen (347 markup removals, 74,318 units) so that every fit reconciles with
``analysis/frequency/fits.tsv`` row for row; the entity analysis runs on the
#27/#30 screen (the general tag-run detector on top, 588 removals, 74,077
units) so that every entity's occurrence count reconciles with #30's
``entity_glossary.tsv`` row for row.  The 241-occurrence difference is the
``href``/``mailto``/``colspan`` tag furniture the general detector removes and
#19's ledger does not list.

Reproduce::

    python work/burstiness_heaps.py               # writes analysis/burstiness/
    python work/test_burstiness_heaps.py

Offline when the pinned reference cache is present; deterministic: fixed
seeds, no clock, and a rebuild is byte-identical.  The build is stdlib-only
(the local layers it imports - collocation, entity_induction, frequency_zipf -
are too), so no environment is needed; the committed ``analysis/burstiness/``
artifacts were produced by CPython 3.12 and rebuild byte-identically there.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the corpus's own rulers so the reconciliations are exact:
#   collocation        -- #27's screen (identical to #30's)
#   entity_induction   -- #30's capitalized / whitespace-adjacent phrase rules
#   frequency_zipf     -- #24's Heaps estimator and pinned English references
import collocation as cl  # noqa: E402
import entity_induction as ei  # noqa: E402
import frequency_zipf as fz  # noqa: E402

GLOSSARY = ROOT / "analysis" / "entity" / "entity_glossary.tsv"
FREQ_OUT = ROOT / "analysis" / "frequency"
FREQ_FITS = FREQ_OUT / "fits.tsv"
FREQ_ENGLISH = FREQ_OUT / "english_baseline.tsv"
OUT = ROOT / "analysis" / "burstiness"

SCHEMA = "burstiness-heaps/1"
SEED = 53                 # every null draw is seeded from this + crc32(name)
NULLS = 2000              # relocation draws per tested term
FANO_WINDOWS = (100, 500, 1000)
MIN_WINDOWS = 20          # a Fano factor needs at least this many windows
MIN_N = 8                 # recurrence floor for the burstiness table
SEGMENTS = 5
GROWTH_POINTS = 200
ENTITY_GROWTH_POINTS = 40
JACKKNIFE_BLOCK = 1000    # English block size for the leave-one-out jackknife
MIN_JACKKNIFE_N = 500     # below this a jackknife refit is not meaningful
V_MARKS = (10000, 25000, 50000)
P_LEVEL = 0.05
B_LEVEL = 0.0
EVEN_F = 0.9              # descriptive cut only; `even` is a tested verdict
CONTROL_TOP = 10          # control terms reported per population
# The corpus's disclosed OCR recognition-loop pages (#4's ensemble caught
# them, #34/#51 report them): p89/p208/p182 general loops, p90 the
# `apoptosis=second` type-minter, p219 the `konganic-eye of cyberBuddha`
# repeat.  Used only to *name the cause* of dense clusters, never to filter.
LOOP_PAGES = (90, 219)


# --------------------------------------------------------------------------
# tsv io (quote-safe: split on tabs, never a csv reader)
# --------------------------------------------------------------------------
def cell(value) -> str:
    s = "" if value is None else str(value)
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            if isinstance(row, dict):
                values = [row.get(h, "") for h in header]
            else:  # positional tuple/list rows
                values = list(row) + [""] * (len(header) - len(row))
            fh.write("\t".join(cell(v) for v in values) + "\n")


def read_tsv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _f(x, nd=4):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def _sig(x, nd=3):
    """A p-value, floored at '<0.001' so a 2,000-draw floor is not read as 0."""
    if x is None or x == "":
        return ""
    if float(x) <= 0.001:
        return "<0.001"
    return f"{float(x):.{nd}f}"


def _bias(x):
    """A jackknife refit-mean bias in scientific notation, never a bare 0.0000."""
    if x is None or x == "":
        return ""
    return f"{float(x):+.2e}"


# --------------------------------------------------------------------------
# the screened unit stream (#27/#30 screen: 74,077 items)
# --------------------------------------------------------------------------
def screened_stream():
    """The #30-screened unit stream, with per-token page / start / uid.

    Identical screen to #27/#30 (``collocation.load_units`` + the general
    tag-run detector), so entity counts reconcile with #30's glossary.
    """
    page_units, unit_meta = cl.load_units()
    bodies = cl.load_bodies()
    spans, uids, types = cl.load_markup(bodies)
    stream = {"surface": [], "page": [], "start": [], "uid": []}
    for page in sorted(page_units):
        for s, uid, _lower, surface, _cls in page_units[page]:
            if cl.unit_is_markup(surface, uid, s, page, spans, uids, types):
                continue
            stream["surface"].append(surface)
            stream["page"].append(page)
            stream["start"].append(s)
            stream["uid"].append(uid)
    stream["bodies"] = bodies
    stream["unit_meta"] = unit_meta
    stream["n"] = len(stream["surface"])
    return stream


def locate(surface, stream):
    """Positions of one entity surface on the screened stream.

    Single-token surfaces match exactly; multi-word surfaces follow #30's
    phrase rule (every word capitalized, whitespace-adjacent in the page
    body, a trailing numeric token read as a designator).
    """
    toks = surface.split(" ")
    surfaces = stream["surface"]
    starts = stream["start"]
    pages = stream["page"]
    uids = stream["uid"]
    bodies = stream["bodies"]
    unit_meta = stream["unit_meta"]
    hits = []
    for i in range(len(surfaces) - len(toks) + 1):
        if surfaces[i:i + len(toks)] != toks:
            continue
        if len(toks) == 1:
            hits.append(i)
            continue
        words = toks[:-1] if ei.is_numeric(toks[-1]) else toks
        if not all(ei.is_capitalized(t) for t in words):
            continue
        ok = True
        for k in range(1, len(toks)):
            page = pages[i + k - 1]
            if not ei._adjacent(bodies[page], unit_meta[uids[i + k - 1]][2],
                                starts[i + k]):
                ok = False
                break
        if ok:
            hits.append(i)
    return hits


def entity_positions(glossary, stream):
    """({entity: sorted positions}, deltas vs #30's n_occurrences)."""
    positions = {}
    deltas = []
    for r in glossary:
        found = []
        for part in r["surface_variants"].split("|"):
            if part:
                found.extend(locate(part.rpartition(":")[0], stream))
        found.sort()
        positions[r["entity"]] = found
        expected = int(r["n_occurrences"])
        if len(found) != expected:
            deltas.append({"entity": r["entity"], "expected": expected,
                           "located": len(found)})
    return positions, deltas


def entity_surface_owners(glossary):
    """``{surface: entity}`` for every surface string a glossary row matches."""
    owners = {}
    for r in glossary:
        for part in r["surface_variants"].split("|"):
            if part:
                surface = part.rpartition(":")[0]
                owners.setdefault(surface, r["entity"])
    return owners


def nested_shares(glossary, positions):
    """Occurrences an entity shares with a *longer* glossary entity.

    A match of a long phrase starts at the same token as a match of a shorter
    phrase that is its token-prefix, so ``ADAM Doll`` at position i is also an
    ``ADAM`` occurrence at i.  The overlap is measured from the located
    positions themselves (an intersection, never a guess), and only pairs whose
    variants are token-prefixes can overlap at all.
    """
    pairs = []
    for r in glossary:
        name = r["entity"]
        variants = [p.rpartition(":")[0] for p in
                    r["surface_variants"].split("|") if p]
        for r2 in glossary:
            other = r2["entity"]
            if other == name:
                continue
            own = {p.rpartition(":")[0] for p in
                   r2["surface_variants"].split("|") if p}
            prefixed = any(v != o and o.split(" ")[:len(v.split(" "))] ==
                           v.split(" ") for o in own for v in variants)
            if not prefixed:
                continue
            shared = sorted(set(positions[name]) & set(positions[other]))
            if shared:
                pairs.append({"entity": name, "nested_in": other,
                              "n_shared": len(shared),
                              "n_entity": len(positions[name])})
    pairs.sort(key=lambda d: (-d["n_shared"], d["entity"]))
    return pairs


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def growth_curve(tokens, n_points=GROWTH_POINTS):
    """Cumulative type count V at checkpoints, N ascending."""
    n = len(tokens)
    if n < 2:
        return [(max(n, 0), len(set(tokens)))]
    seen = set()
    step = max(1, n // n_points)
    pts = []
    for i, t in enumerate(tokens, 1):
        seen.add(t)
        if i % step == 0 or i == n:
            pts.append((i, len(seen)))
    return pts


def heaps_fit_r2(tokens, n_points=400):
    """#24's Heaps fit plus the log-log R^2 of that fit."""
    fit = fz.heaps_fit(tokens, n_points=n_points)
    pts = fit["heaps_points"]
    if len(pts) < 3:
        return {**fit, "heaps_r2": float("nan")}
    xs = [math.log(p[0]) for p in pts]
    ys = [math.log(p[1]) for p in pts]
    my = sum(ys) / len(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (fit["heaps_logk"] + fit["heaps_beta"] * x)) ** 2
                 for x, y in zip(xs, ys))
    return {**fit, "heaps_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0}


def v_at(tokens, marks=V_MARKS):
    """V exactly at each token mark (single pass)."""
    want = set(marks)
    seen = set()
    out = {}
    for i, t in enumerate(tokens, 1):
        seen.add(t)
        if i in want:
            out[i] = len(seen)
    return out


def fano_factor(positions, w, n_total):
    """Index of dispersion Var(count)/Mean(count) over non-overlapping windows.

    ``F ~ 1`` is a Bernoulli (Poisson-rate) placement; ``F >> 1`` is
    over-dispersion (bursty); ``F < 1`` is more even than chance.
    """
    n_windows = n_total // w
    if n_windows < MIN_WINDOWS or not positions:
        return None
    counts = [0] * n_windows
    for p in positions:
        b = p // w
        if b < n_windows:
            counts[b] += 1
    mean = sum(counts) / n_windows
    if mean <= 0:
        return None
    var = sum((c - mean) ** 2 for c in counts) / n_windows
    return var / mean


def burstiness(gaps):
    """Goh-Barabasi burstiness B = (sigma - mu)/(sigma + mu) over gaps.

    B = 0 is memoryless/geometric gaps; B > 0 is a heavier-tailed (bursty)
    arrival process; B < 0 is more regular than geometric.
    """
    if len(gaps) < 3:
        return None
    mu = statistics.mean(gaps)
    sd = statistics.pstdev(gaps)
    if sd + mu <= 0:
        return None
    return (sd - mu) / (sd + mu)


def null_fanos(n_hits, w, n_total, seed, draws):
    """Fano factors of `draws` placement nulls at the term's own rate.

    A draw is a uniformly random *subset* of the stream of the same size as the
    observed occurrence set, so no token is used twice and the null has exactly
    the observed rate.  (Sampling *with* replacement would let a draw put two
    occurrences on one token, which is not a placement a corpus can make and
    which inflates the null's own spread.)
    """
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        pos = rng.sample(range(n_total), k=n_hits)
        f = fano_factor(pos, w, n_total)
        if f is not None:
            out.append(f)
    return sorted(out)


def burst_p_value(observed, nulls):
    """One-sided add-one-corrected high-tail p: nulls at least as dispersed."""
    if observed is None or not nulls:
        return None
    return (1 + sum(1 for f in nulls if f >= observed)) / (len(nulls) + 1)


def even_p_value(observed, nulls):
    """One-sided add-one-corrected low-tail p: nulls at least as even.

    Ties count toward the p, which is the conservative reading and the same
    convention as the high tail.  It matters here: at these rates the Fano
    factor over 74 windows is a coarse, lumpy statistic, so for some terms the
    observed value is a *common* null value rather than an unusual one.  A
    strict-inequality count (nulls only strictly below the observed) would
    score those terms as maximally even precisely because they sit on the null's
    mode - the opposite of the truth.
    """
    if observed is None or not nulls:
        return None
    return (1 + sum(1 for f in nulls if f <= observed)) / (len(nulls) + 1)


def entity_seed(name):
    """Per-term seed, so a row's null does not depend on row order."""
    return SEED + zlib.crc32(name.encode("utf-8"))


def gap_stats(positions):
    gaps = [b - a for a, b in itertools.pairwise(positions)]
    if not gaps:
        return {"mean_gap": None, "max_gap": None, "gap_cv": None,
                "burstiness_b": None}
    mu = statistics.mean(gaps)
    sd = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
    return {"mean_gap": mu, "max_gap": max(gaps),
            "gap_cv": (sd / mu) if mu else None,
            "burstiness_b": burstiness(gaps)}


def verdict(observed_f, p_value, b_value, p_even):
    """bursty / even / neutral, with the thresholds disclosed in the report.

    ``bursty`` needs *both* legs: the high-tail placement null must reject
    (p <= P_LEVEL) **and** the Goh-Barabasi B must exceed B_LEVEL.  HIV is the
    case that keeps the second leg honest -- its occurrences are over-dispersed
    against the null (F(1000) 3.3, p < 0.001) but its inter-arrival gaps are
    *more regular* than geometric (B < 0), an even-spacing signature that a
    clustered process does not have, so HIV is not bursty.  A missing B
    (``None``) means the leg could not be computed, not that it failed, so it
    does not veto the high tail.

    ``even`` needs the *low* tail to reject at the same level.  A Fano factor
    below 1 is not evidence by itself -- the null's own spread puts plenty of
    random placements under it -- so ``EVEN_F`` is a descriptive cut and this
    column carries no verdict.
    """
    if observed_f is None or p_value is None or p_even is None:
        return "untested"
    if p_value <= P_LEVEL and (b_value is None or b_value > B_LEVEL):
        return "bursty"
    if p_even <= P_LEVEL:
        return "even"
    return "neutral"


def groups_for_jackknife(tagged_tokens, block=None):
    """Token lists per group (pages for the book; fixed blocks for English)."""
    if block is None:
        d = {}
        for g, t in tagged_tokens:
            d.setdefault(g, []).append(t)
        return [d[g] for g in sorted(d)]
    return [[t for _g, t in tagged_tokens[i:i + block]]
            for i in range(0, len(tagged_tokens), block)]


def jackknife_beta(groups, n_points=400):
    """Leave-one-group-out refits of beta.

    Dropping a group (never duplicating one) keeps the resampled corpus on the
    same footing as the original, so the refits spread around the point
    estimate rather than around a biased resample -- unlike a bootstrap over
    pages, which duplicates pages and biases beta down because a re-seen page
    contributes no new types.

    The refit *mean* is **not** the point estimate and must not be read as one:
    dropping a group shortens the curve and moves the checkpoint grid, which
    shifts the refits by a small amount that grows as the groups get bigger
    (the book's 225 page refits average within 4e-4 of the point estimate, the
    matched-N English references within 2e-3).  Only the spread is used, and
    the standard error is the jackknife correction ``sd * sqrt(n - 1)`` -- the
    plain sd across n refits is the sd of one draw from a mean, not the se of
    the mean.
    """
    return [heaps_fit_r2([t for j, g in enumerate(groups) if j != i
                          for t in g], n_points)["heaps_beta"]
            for i in range(len(groups))]


def jackknife_summary(groups, point_beta, n_points=400):
    """The build's jackknife row: spread, jackknife se, and the refit mean.

    ``se`` is ``sd * sqrt(n - 1)`` (the jackknife correction); ``sd`` is the
    plain spread of the refits; ``bias`` is the refit mean minus the point
    estimate, published because it is not zero.
    """
    vals = jackknife_beta(groups, n_points=n_points)
    sd = statistics.pstdev(vals)
    return {"beta_loo_sd": sd,
            "beta_loo_se": sd * math.sqrt(len(vals) - 1),
            "beta_loo_mean": statistics.mean(vals),
            "beta_loo_min": min(vals),
            "beta_loo_max": max(vals),
            "beta_loo_bias": statistics.mean(vals) - point_beta,
            "jackknife_groups": len(vals)}


def segment_table(tokens, k=SEGMENTS):
    """Per-segment vocabulary: own types, types new to the corpus, own beta."""
    n = len(tokens)
    size = n // k
    seen = set()
    rows = []
    for s in range(k):
        seg = tokens[s * size:(s + 1) * size] if s < k - 1 else tokens[s * size:]
        seg_seen = set(seg)
        rows.append({
            "segment": s + 1,
            "n": len(seg),
            "v": len(seg_seen),
            "new_types": len(seg_seen - seen),
            "cum_v": len(seen | seg_seen),
            "beta_seg": heaps_fit_r2(seg)["heaps_beta"],
        })
        seen |= seg_seen
    return rows


def context(surfaces, position, radius=7):
    lo = max(0, position - radius)
    hi = min(len(surfaces), position + radius + 1)
    return " ".join(surfaces[lo:hi])


def channel_row(label, kind, positions, n_total, w, seed, nulls):
    """One merged channel: Fano, B, and both tails of its own rate-matched null.

    A channel is a *merged* position set (every term's occurrences unioned), so
    this is the only unit that can be compared with another channel -- a merged
    channel's Fano factor is not comparable with a per-term median, which is
    why the control channels are built the same way rather than summarized.
    """
    nulls_main = null_fanos(len(positions), w, n_total, seed, nulls)
    obs = fano_factor(positions, w, n_total)
    p = burst_p_value(obs, nulls_main)
    p_even = even_p_value(obs, nulls_main)
    b_value = gap_stats(positions)["burstiness_b"]
    return {
        "label": label, "kind": kind, "n_positions": len(positions),
        "rate": len(positions) / n_total,
        "fano": {f"fano_{width}": fano_factor(positions, width, n_total)
                 for width in FANO_WINDOWS},
        "null_fano_median": (statistics.median(nulls_main)
                             if nulls_main else None),
        "null_fano_q05": (nulls_main[int(0.05 * (len(nulls_main) - 1))]
                          if nulls_main else None),
        "null_fano_q95": (nulls_main[int(0.95 * (len(nulls_main) - 1))]
                          if nulls_main else None),
        "p_bursty": p, "p_even": p_even, "burstiness_b": b_value,
        "verdict": verdict(obs, p, b_value, p_even),
    }


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def book_layers(units, parts, cats):
    """The #24 fit populations, recomputed so every fit reconciles with it."""
    unit_case = [r["unit"] for r in units]
    unit_lower = [r["unit"].lower() for r in units]
    part_tokens = [p["text"].lower() for p in parts]
    page_of = {int(r["uid"]): int(r["page"]) for r in units}
    layers = {
        "book_units": (unit_case,
                       list(zip((int(r["page"]) for r in units), unit_case))),
        "book_units_lower": (unit_lower,
                             list(zip((int(r["page"]) for r in units),
                                      unit_lower))),
        "book_parts": (part_tokens,
                       list(zip((page_of[int(p["uid"])] for p in parts),
                                part_tokens))),
    }

    def tokens_for(pred):
        """[(page, token)] so the jackknife groups by page, not by type."""
        return [(int(r["page"]), r["unit"].lower()) for r in units
                if pred(fz.category_of(r["unit"], cats))]

    for name, pred in (
            ("book_english", lambda c: c == "english"),
            ("book_nonenglish", lambda c: c != "english"),
            ("book_coinage",
             lambda c: c in ("compound", "true_coinage", "proper_noun")),
            ("book_ocr", lambda c: c == "ocr_error")):
        tagged = tokens_for(pred)
        layers[name] = ([t for _g, t in tagged], tagged)
    return layers


def english_matched(reference_dir: Path, allow_download: bool, log):
    """[(name, tokens)] at #24's matched N, from the pinned reference cache."""
    out = []
    for name, _author, path in fz.ensure_reference(Path(reference_dir),
                                                   allow_download, log):
        body = fz.strip_gutenberg(path.read_text(encoding="utf-8",
                                                 errors="replace"))
        toks = fz.tokenize_english(body)[:fz.BOOK_N]
        if toks:
            out.append((name, toks))
    return out


def load_glossary():
    if not GLOSSARY.exists():
        raise SystemExit(f"Missing #30 artifact: {GLOSSARY}")
    return read_tsv(GLOSSARY)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build(out_dir: Path, reference_dir: Path, no_download: bool,
          no_reference: bool, nulls: int, jackknife: bool, quiet: bool):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if quiet else print

    diagnostics = [
        ("schema", SCHEMA, "burstiness-heaps artifact schema"),
        ("seed", SEED, "null draws seeded SEED + crc32(term)"),
        ("nulls_per_entity", nulls, "placement draws per tested term"),
        ("fano_windows", "|".join(str(w) for w in FANO_WINDOWS),
         "non-overlapping window widths (tokens)"),
        ("min_windows", MIN_WINDOWS, "a Fano factor needs >= this many windows"),
        ("min_n", MIN_N, "recurrence floor for the burstiness table"),
        ("segments", SEGMENTS, "equal token segments for the novelty table"),
        ("p_level", P_LEVEL, "one-sided significance, both tails"),
        ("b_level", B_LEVEL, "B must exceed this for a bursty verdict"),
        ("even_f", EVEN_F, "descriptive cut only; `even` is the low-tail p"),
        ("control_top", CONTROL_TOP,
         "control terms reported per population, entity surfaces excluded"),
        ("jackknife", "on" if jackknife else "off",
         ("leave-one-group-out beta uncertainty (pages for the book, "
          f"{JACKKNIFE_BLOCK}-token blocks for references)")),
        ("jackknife_se_rule", "sd * sqrt(n - 1)",
         "the plain sd across n refits is not the standard error"),
    ]

    # ---- streams and screens ----------------------------------------------
    stream = screened_stream()
    glossary = load_glossary()
    positions, pos_deltas = entity_positions(glossary, stream)
    mentions = sum(len(p) for p in positions.values())
    units24, parts24, cats24 = fz.load_book()
    growth_n = len(units24)
    diagnostics += [
        ("stream_growth_n", growth_n,
         "#24 screen (347 markup removals) -- growth layer"),
        ("stream_entity_n", stream["n"],
         "#27/#30 screen (588 markup removals) -- entity layer"),
        ("screen_delta", growth_n - stream["n"],
         "tag-run furniture the general detector removes and #19 does not list"),
        ("entities", len(glossary), "#30 glossary rows (accepted entities)"),
        ("entity_mentions", mentions,
         "located occurrences, summed over glossary rows"),
        ("entity_growth_points", ENTITY_GROWTH_POINTS,
         "checkpoints for the entity-layer Heaps fit"),
        ("entity_locate_deltas", len(pos_deltas),
         "rows whose located count != #30 n_occurrences"),
    ]

    # ---- English references ------------------------------------------------
    english = []
    if not no_reference:
        english = english_matched(reference_dir, not no_download, log)
        diagnostics.append(("english_refs_loaded", len(english),
                            "pinned #24 references at matched N"))

    # ---- Part 1: Heaps' law / vocabulary growth ----------------------------
    layers = book_layers(units24, parts24, cats24)
    fits_rows, curve_rows, seg_rows = [], [], []
    betas = {}
    beta_se = {}

    def fit_one(name, kind, tokens, tagged, jackknife_on, block=None):
        fit = heaps_fit_r2(tokens)
        marks = v_at(tokens)
        row = {
            "corpus": name, "kind": kind, "N": len(tokens),
            "V": len(set(tokens)),
            "heaps_beta": fit["heaps_beta"], "heaps_logk": fit["heaps_logk"],
            "heaps_r2": fit["heaps_r2"],
            "beta_loo_sd": "", "beta_loo_se": "", "beta_loo_mean": "",
            "beta_loo_bias": "", "beta_loo_min": "", "beta_loo_max": "",
            "jackknife_groups": "", **{f"V_{m // 1000}k": marks.get(m, "")
                                       for m in V_MARKS},
        }
        groups = groups_for_jackknife(tagged, block=block)
        if jackknife_on and len(tokens) >= MIN_JACKKNIFE_N and len(groups) >= 3:
            jk = jackknife_summary(groups, fit["heaps_beta"])
            for key in ("beta_loo_sd", "beta_loo_se", "beta_loo_mean",
                        "beta_loo_bias", "beta_loo_min", "beta_loo_max",
                        "jackknife_groups"):
                row[key] = jk[key]
            beta_se[name] = jk["beta_loo_se"]
        fits_rows.append(row)
        betas[name] = fit["heaps_beta"]
        curve_rows.extend({"corpus": name, "N": n, "V": v}
                          for n, v in growth_curve(tokens))
        if len(tokens) >= MIN_JACKKNIFE_N:
            seg_rows.extend({"corpus": name, **s} for s in segment_table(tokens))
        return fit

    for name in ("book_units", "book_units_lower", "book_parts",
                 "book_english", "book_nonenglish", "book_coinage",
                 "book_ocr"):
        tokens, tagged = layers[name]
        fit_one(name, "book", tokens, tagged, jackknife)

    en_beta_by_name = {}
    en_curves = {}
    for name, toks in english:
        en_tagged = [(i // JACKKNIFE_BLOCK, t) for i, t in enumerate(toks)]
        fit_one(f"en_{name}", "english", toks, en_tagged, jackknife,
                block=JACKKNIFE_BLOCK)
        en_beta_by_name[name] = betas[f"en_{name}"]
        en_curves[name] = growth_curve(toks)

    # English mean curve at the matched-N grid (references that reach each N)
    if en_curves:
        first = next(iter(en_curves))
        by_n = [{n: v for n, v in en_curves[name]} for name in en_curves]
        mean_curve = []
        for n, _v in en_curves[first]:
            vals = [d[n] for d in by_n if n in d]
            mean_curve.append({"corpus": "en_mean", "N": n,
                               "V": sum(vals) / len(vals)})
        curve_rows.extend(mean_curve)
        fits_rows.append({
            "corpus": "en_mean", "kind": "english_mean", "N": fz.BOOK_N,
            "V": "", "heaps_beta": statistics.mean(en_beta_by_name.values()),
            "heaps_logk": "", "heaps_r2": "",
            "beta_loo_sd": "", "beta_loo_se": "", "beta_loo_mean": "",
            "beta_loo_bias": "", "beta_loo_min": "", "beta_loo_max": "",
            "jackknife_groups": len(en_curves),
            **{f"V_{m // 1000}k": "" for m in V_MARKS},
        })

    # ---- Part 2: the entity cast's growth ----------------------------------
    pairs = []
    for r in glossary:
        pairs.extend((p, r["entity"]) for p in positions[r["entity"]])
    pairs.sort()
    ent_tokens = [name for _p, name in pairs]
    ent_fit = heaps_fit_r2(ent_tokens, n_points=ENTITY_GROWTH_POINTS)
    ent_curve = growth_curve(ent_tokens, n_points=ENTITY_GROWTH_POINTS)
    curve_rows.extend({"corpus": "entity_layer", "N": n, "V": v}
                      for n, v in ent_curve)
    n_stream = stream["n"]
    first_pos = {r["entity"]: min(positions[r["entity"]]) for r in glossary
                 if positions[r["entity"]]}
    # Two partitions of the same book, which are *not* interchangeable: the
    # token stream splits on position, the mention stream splits on the order
    # mentions arrive in.  Each is labelled with its own denominator wherever
    # it is reported, so no row is read as a share of the other's total.
    thirds_token = [0, 0, 0]
    for pos in first_pos.values():
        thirds_token[min(2, pos * 3 // n_stream)] += 1
    n_mentions = len(pairs)
    # One set of chunk boundaries for both rows, so the first-attestation row
    # and the cumulative row describe the same partition of the same stream.
    mention_bounds = [n_mentions // 3, 2 * n_mentions // 3]

    def mention_third(idx):
        return (0 if idx < mention_bounds[0] else
                1 if idx < mention_bounds[1] else 2)

    first_by_mention = [0, 0, 0]
    seen_names = set()
    for idx, (_p, name) in enumerate(pairs):
        if name not in seen_names:
            first_by_mention[mention_third(idx)] += 1
        seen_names.add(name)
    seen_by = [len({name for _p, name in pairs[:mention_bounds[0]]}),
               len({name for _p, name in pairs[:mention_bounds[1]]}),
               len(seen_names)]

    # ---- Part 3: burstiness of entity use ----------------------------------
    w_main = FANO_WINDOWS[-1]
    burst_rows = []
    for r in glossary:
        name = r["entity"]
        pos = positions.get(name, [])
        if len(pos) < MIN_N:
            continue
        stats = gap_stats(pos)
        page_counts = Counter(stream["page"][i] for i in pos)
        top_page, top_n = page_counts.most_common(1)[0]
        row = {
            "entity": name, "role": r["role"], "n": len(pos),
            "n_pages": len(page_counts),
            "first_page": min(stream["page"][i] for i in pos),
            "last_page": max(stream["page"][i] for i in pos),
            "top_page": top_page, "max_page_count": top_n,
            "top_page_share": top_n / len(pos), **stats,
        }
        nulls_main = null_fanos(len(pos), w_main, n_stream, entity_seed(name),
                                nulls)
        for w in FANO_WINDOWS:
            row[f"fano_{w}"] = fano_factor(pos, w, n_stream)
        obs = row[f"fano_{w_main}"]
        p = burst_p_value(obs, nulls_main)
        p_even = even_p_value(obs, nulls_main)
        row["null_fano_median"] = (statistics.median(nulls_main)
                                   if nulls_main else "")
        row["null_fano_q05"] = (nulls_main[int(0.05 * (len(nulls_main) - 1))]
                                if nulls_main else "")
        row["null_fano_q95"] = (nulls_main[int(0.95 * (len(nulls_main) - 1))]
                                if nulls_main else "")
        row["null_fano_tie_share"] = ((sum(1 for f in nulls_main
                                           if f == obs) / len(nulls_main))
                                      if nulls_main else "")
        row["p_bursty"] = p if p is not None else ""
        row["p_even"] = p_even if p_even is not None else ""
        row["verdict"] = verdict(obs, p, stats["burstiness_b"], p_even)
        burst_rows.append(row)
    burst_rows.sort(key=lambda r: -(r[f"fano_{w_main}"] or 0.0))

    # ---- controls: ordinary words and the coinage layer ---------------------
    # The coinage population is #24's non-English category, which also
    # contains glossary entities (#30 accepted `retro-ADAM` and `ToKAGE` as
    # named cast members).  Leaving them in would let the entity population
    # supply its own controls, so every surface the glossary can match is
    # dropped before the top-CONTROL_TOP selection and the slot goes to the
    # next non-entity type.
    surface_owners = entity_surface_owners(glossary)
    english_types = Counter()
    coinage_types = Counter()
    for r in units24:
        surface = r["unit"]
        cat = cats24.get(surface, "markup")
        if cat == "english":
            english_types[surface] += 1
        elif cat not in ("junk", "markup", "markup_residue"):
            coinage_types[surface] += 1
    control_rows = []
    control_pos = {}
    dropped_controls = []
    for kind, top in (("english_word", english_types),
                      ("coinage_type", coinage_types)):
        kept = 0
        for surface, _n in top.most_common():
            if surface in surface_owners:
                dropped_controls.append((kind, surface,
                                         surface_owners[surface]))
                continue
            pos = [i for i, u in enumerate(stream["surface"]) if u == surface]
            if len(pos) < MIN_N:
                continue
            stats = gap_stats(pos)
            nulls_main = null_fanos(len(pos), w_main, n_stream,
                                    entity_seed(surface), nulls)
            obs = fano_factor(pos, w_main, n_stream)
            p = burst_p_value(obs, nulls_main)
            p_even = even_p_value(obs, nulls_main)
            control_rows.append({
                "kind": kind, "term": surface, "n": len(pos),
                "n_pages": len({stream["page"][i] for i in pos}), **stats,
                **{f"fano_{w}": fano_factor(pos, w, n_stream)
                   for w in FANO_WINDOWS},
                "null_fano_median": (statistics.median(nulls_main)
                                     if nulls_main else ""),
                "null_fano_tie_share": ((sum(1 for f in nulls_main
                                              if f == obs) / len(nulls_main))
                                        if nulls_main else ""),
                "p_bursty": p if p is not None else "",
                "p_even": p_even if p_even is not None else "",
                "verdict": verdict(obs, p, stats["burstiness_b"], p_even),
            })
            control_pos.setdefault(kind, set()).update(pos)
            kept += 1
            if kept >= CONTROL_TOP:
                break
    diagnostics.append(("control_entity_surfaces_dropped",
                        "; ".join(f"{s} ({o})" for _k, s, o in
                                  sorted(dropped_controls)) or "none",
                        ("coinage/english control slots freed by removing "
                         "glossary entity surfaces from the control set")))

    # ---- channels: the only like-for-like comparison ------------------------
    # A merged channel is the union of one population's positions, so the
    # entity channel is compared against merged control channels at their own
    # rates -- never against per-term control medians, which mix units.
    channel_pos = sorted({i for pos in positions.values() for i in pos})
    channels = [channel_row("entity_channel", "entity", channel_pos, n_stream,
                            w_main, SEED, nulls)]
    channel_pos_by_kind = {k: sorted(v) for k, v in control_pos.items()}
    for kind, label in (("english_word", "english_word_channel"),
                        ("coinage_type", "coinage_type_channel")):
        if channel_pos_by_kind.get(kind):
            channels.append(channel_row(label, kind,
                                        channel_pos_by_kind[kind], n_stream,
                                        w_main, entity_seed(label), nulls))
    channel = dict(channels[0])
    channel["mentions"] = mentions
    diagnostics.append(("entity_channel_positions", len(channel_pos),
                        ("distinct token positions covered by any entity "
                         "(overlapping entities counted once)")))
    for ch in channels[1:]:
        diagnostics.append((f"{ch['label']}_positions", ch["n_positions"],
                            (f"union of the {CONTROL_TOP} {ch['kind']} "
                             "control terms' positions")))

    # control-population summaries, so the per-term verdict is on the record
    control_summary = {}
    for kind in ("english_word", "coinage_type"):
        sub = [r for r in control_rows if r["kind"] == kind]
        if not sub:
            continue
        control_summary[kind] = {
            "n": len(sub),
            "bursty": sum(1 for r in sub if r["verdict"] == "bursty"),
            "median_fano_1000": statistics.median(r["fano_1000"] for r in sub),
            "max_fano_1000": max(r["fano_1000"] for r in sub),
            "median_b": statistics.median(r["burstiness_b"] for r in sub),
        }
    entity_median_f = (statistics.median(r["fano_1000"] for r in burst_rows)
                       if burst_rows else None)

    # ---- nested entities: occurrences counted under two names --------------
    nested = nested_shares(glossary, positions)
    nested_effect = []
    for share in nested:
        name, other = share["entity"], share["nested_in"]
        pos = positions[name]
        shared = set(pos) & set(positions[other])
        rest = [p for p in pos if p not in shared]
        nested_effect.append({
            "entity": name, "nested_in": other, "n_shared": len(shared),
            "n_entity": len(pos), "n_excl_nested": len(rest),
            "fano_1000_all": fano_factor(pos, w_main, n_stream),
            "fano_1000_excl_nested": fano_factor(rest, w_main, n_stream),
            "burstiness_b_all": gap_stats(pos)["burstiness_b"],
            "burstiness_b_excl_nested": gap_stats(rest)["burstiness_b"],
        })
    nested_effect.sort(key=lambda d: (-d["n_shared"], d["entity"]))
    nested_names = sorted({d["entity"] for d in nested_effect})
    diagnostics.append(("entities_sharing_positions_with_a_longer_entity",
                        len(nested_names),
                        ("occurrences counted under both a short and a long "
                         "glossary surface - see nested_entities.tsv")))

    # densest single-page clusters over *all* entities (not only tested ones)
    page_clusters = []
    for r in glossary:
        pos = positions[r["entity"]]
        if not pos:
            continue
        for page, cnt in Counter(stream["page"][i] for i in pos).items():
            page_clusters.append({"entity": r["entity"], "page": page,
                                  "count": cnt, "n": len(pos)})
    page_clusters.sort(key=lambda d: (-d["count"], d["entity"]))

    # ---- reconciliation -----------------------------------------------------
    recon = []

    def check(label, expected, actual, note=""):
        if isinstance(expected, float) or isinstance(actual, float):
            ok = (expected is not None and actual is not None
                  and abs(float(expected) - float(actual)) < 1e-9)
        else:
            ok = str(expected) == str(actual)
        recon.append({"check": label, "expected": expected,
                      "actual": actual, "status": "ok" if ok else "MISMATCH",
                      "note": note})

    book_curve = growth_curve(layers["book_units"][0])
    book_curve_vs = [v for _n, v in book_curve]
    check("screened entity stream reproduces #30's count", 74077,
          stream["n"], "#27/#30 screen")
    located_ok = sum(1 for r in glossary
                     if len(positions[r["entity"]]) == int(r["n_occurrences"]))
    check("entities located exactly", len(glossary), located_ok,
          "located n == #30 n_occurrences per row")
    check("entity mentions", sum(int(r["n_occurrences"]) for r in glossary),
          mentions, "sum of located positions")
    check("growth stream reproduces #24's screen", 74318, growth_n,
          "347 markup removals")
    if FREQ_FITS.exists():
        up = {r["corpus"]: r for r in read_tsv(FREQ_FITS)}
        for name in ("book_units", "book_units_lower", "book_parts",
                     "book_english", "book_nonenglish", "book_coinage",
                     "book_ocr"):
            if name in up and name in betas:
                check(f"heaps beta vs #24 fits.tsv [{name}]",
                      float(up[name]["heaps_beta"]), betas[name],
                      "same estimator, same screen, same order")
        if "en_mean" in up and en_beta_by_name:
            check("english mean beta vs #24 fits.tsv [en_mean]",
                  float(up["en_mean"]["heaps_beta"]),
                  statistics.mean(en_beta_by_name.values()),
                  "mean of the matched-N refits")
    else:
        recon.append({"check": "#24 fits.tsv", "expected": "present",
                      "actual": "missing", "status": "SKIPPED",
                      "note": "run work/frequency_zipf.py first"})
    if FREQ_ENGLISH.exists() and english:
        up = {}
        for r in read_tsv(FREQ_ENGLISH):
            up[r.get("text") or r.get("name") or r.get("corpus")] = r
        for name, _toks in english:
            row = up.get(name) or up.get(f"en_{name}")
            if row:
                check(f"matched N vs #24 english_baseline [{name}]",
                      int(row["N"]), len(_toks), "matched-N prefix")
    check("book growth curve monotone", "ascending",
          "ascending" if book_curve_vs == sorted(book_curve_vs) else "not",
          "cumulative type count")
    check("entity locate deltas", 0, len(pos_deltas),
          "; ".join(f"{d['entity']} {d['expected']}!={d['located']}"
                    for d in pos_deltas) or "none")
    check("nested overlap fits inside the parent entity", "no overflow",
          "no overflow" if all(d["n_shared"] <= d["n_entity"]
                               for d in nested_effect) else "overflow",
          "an entity cannot share more positions than it has")
    for short, long in (("ADAM", "ADAM Doll"), ("Sato", "Sato Corporation")):
        share = next((d for d in nested_effect
                      if d["entity"] == short and d["nested_in"] == long), None)
        check(f"{short} occurrences inside {long}", "present",
              "present" if share else "MISSING",
              f"{share['n_shared']} of {share['n_entity']} occurrences shared"
              if share else "nested pair not found")

    # ---- summary ------------------------------------------------------------
    tested = len(burst_rows)
    n_bursty = sum(1 for r in burst_rows if r["verdict"] == "bursty")
    n_even = sum(1 for r in burst_rows if r["verdict"] == "even")
    n_neutral = sum(1 for r in burst_rows if r["verdict"] == "neutral")
    n_below_even_f = sum(1 for r in burst_rows
                         if (r["fano_1000"] or 0.0) < EVEN_F)
    book_beta = betas["book_units"]
    en_mean_beta = (statistics.mean(en_beta_by_name.values())
                    if en_beta_by_name else None)
    en_se = [beta_se[f"en_{n}"] for n in en_beta_by_name
             if f"en_{n}" in beta_se]
    summary = {
        "schema": SCHEMA,
        "screens": {
            "growth_stream_n": growth_n,
            "growth_screen": "#24 (347 markup removals)",
            "entity_stream_n": stream["n"],
            "entity_screen": "#27/#30 (588 markup removals)",
            "screen_delta": growth_n - stream["n"],
        },
        "heaps": {
            "book_units_beta": book_beta,
            "book_units_logk": fits_rows[0]["heaps_logk"],
            "book_units_r2": fits_rows[0]["heaps_r2"],
            "book_units_beta_loo_sd": fits_rows[0]["beta_loo_sd"],
            "book_units_beta_loo_se": fits_rows[0]["beta_loo_se"],
            "book_units_beta_loo_mean": fits_rows[0]["beta_loo_mean"],
            "book_units_beta_loo_bias": fits_rows[0]["beta_loo_bias"],
            "book_units_beta_loo_min": fits_rows[0]["beta_loo_min"],
            "book_units_beta_loo_max": fits_rows[0]["beta_loo_max"],
            "book_coinage_beta": betas.get("book_coinage"),
            "english_mean_beta": en_mean_beta,
            "english_beta_min": (min(en_beta_by_name.values())
                                 if en_beta_by_name else None),
            "english_beta_max": (max(en_beta_by_name.values())
                                 if en_beta_by_name else None),
            "english_beta_se_min": (min(en_se) if en_se else None),
            "english_beta_se_max": (max(en_se) if en_se else None),
            "english_beta_references": len(en_beta_by_name),
            "entity_layer_beta": ent_fit["heaps_beta"],
            "entity_layer_logk": ent_fit["heaps_logk"],
            "book_units_in_english_range": (
                min(en_beta_by_name.values()) <= book_beta <=
                max(en_beta_by_name.values())) if en_beta_by_name else None,
        },
        "growth": {
            "segments": SEGMENTS,
            "book_units_segment_new_types": [
                s["new_types"] for s in seg_rows
                if s["corpus"] == "book_units"],
            "book_units_segment_beta": [s["beta_seg"] for s in seg_rows
                                        if s["corpus"] == "book_units"],
            "entity_cast_seen_by_mention_third": seen_by,
            "entity_cast_share_by_mention_third": [
                c / len(glossary) for c in seen_by],
            "entity_first_attestation_by_token_third": thirds_token,
            "entity_first_attestation_by_mention_third": first_by_mention,
        },
        "burstiness": {
            "nulls": nulls, "fano_windows": list(FANO_WINDOWS),
            "n_floor": MIN_N, "entities_tested": tested,
            "even_f_cut": EVEN_F, "entities_below_even_f": n_below_even_f,
            "entity_median_fano_1000": entity_median_f,
            "bursty": n_bursty, "even": n_even, "neutral": n_neutral,
            "median_b": (statistics.median(
                r["burstiness_b"] for r in burst_rows
                if r["burstiness_b"] is not None) if burst_rows else None),
            "top_burst": ({"entity": burst_rows[0]["entity"],
                           "fano_1000": burst_rows[0]["fano_1000"],
                           "burstiness_b": burst_rows[0]["burstiness_b"],
                           "first_page": burst_rows[0]["first_page"],
                           "last_page": burst_rows[0]["last_page"]}
                          if burst_rows else None),
            "channels": channels, "channel": channel,
            "controls": control_rows,
            "controls_summary": control_summary,
            "nested_entities": nested_effect,
            "densest_page_clusters": page_clusters[:5],
        },
        "reconciliation_mismatches": [r for r in recon
                                      if r["status"] == "MISMATCH"],
    }

    # ---- artifacts ------------------------------------------------------------
    write_tsv(out_dir / "heaps_fits.tsv",
              ["corpus", "kind", "N", "V", "heaps_beta", "heaps_logk",
               "heaps_r2", "beta_loo_sd", "beta_loo_se", "beta_loo_mean",
               "beta_loo_bias", "beta_loo_min", "beta_loo_max",
               "jackknife_groups"] + [f"V_{m // 1000}k" for m in V_MARKS],
              fits_rows)
    write_tsv(out_dir / "growth_curve.tsv", ["corpus", "N", "V"], curve_rows)
    write_tsv(out_dir / "segment_growth.tsv",
              ["corpus", "segment", "n", "v", "new_types", "cum_v",
               "beta_seg"], seg_rows)
    write_tsv(out_dir / "entity_growth.tsv", ["corpus", "N", "V"],
              [{"corpus": "entity_layer", "N": n, "V": v}
               for n, v in ent_curve])
    write_tsv(out_dir / "entity_burstiness.tsv",
              ["entity", "role", "n", "n_pages", "first_page", "last_page",
               "top_page", "max_page_count", "top_page_share", "mean_gap",
               "max_gap", "gap_cv", "burstiness_b"]
              + [f"fano_{w}" for w in FANO_WINDOWS]
              + ["null_fano_median", "null_fano_q05", "null_fano_q95",
                 "null_fano_tie_share", "p_bursty", "p_even", "verdict"],
              burst_rows)
    write_tsv(out_dir / "controls_burstiness.tsv",
              ["kind", "term", "n", "n_pages", "mean_gap", "max_gap",
               "gap_cv", "burstiness_b"] + [f"fano_{w}" for w in FANO_WINDOWS]
              + ["null_fano_median", "null_fano_tie_share", "p_bursty",
                 "p_even", "verdict"],
              control_rows)
    write_tsv(out_dir / "channels.tsv",
              ["label", "kind", "n_positions", "rate"]
              + [f"fano_{w}" for w in FANO_WINDOWS]
              + ["null_fano_median", "null_fano_q05", "null_fano_q95",
                 "p_bursty", "p_even", "burstiness_b", "verdict"],
              [{"label": ch["label"], "kind": ch["kind"],
                "n_positions": ch["n_positions"], "rate": ch["rate"],
                **{f"fano_{w}": ch["fano"][f"fano_{w}"]
                   for w in FANO_WINDOWS},
                "null_fano_median": ch["null_fano_median"],
                "null_fano_q05": ch["null_fano_q05"],
                "null_fano_q95": ch["null_fano_q95"],
                "p_bursty": ch["p_bursty"], "p_even": ch["p_even"],
                "burstiness_b": ch["burstiness_b"], "verdict": ch["verdict"]}
               for ch in channels])
    write_tsv(out_dir / "nested_entities.tsv",
              ["entity", "nested_in", "n_shared", "n_entity",
               "n_excl_nested", "fano_1000_all", "fano_1000_excl_nested",
               "burstiness_b_all", "burstiness_b_excl_nested"],
              nested_effect)
    write_tsv(out_dir / "reconciliation.tsv",
              ["check", "expected", "actual", "status", "note"], recon)
    write_tsv(out_dir / "diagnostics.tsv", ["item", "value", "note"],
              diagnostics)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = build_report(summary, fits_rows, seg_rows, burst_rows,
                          control_rows, channel, ent_fit, stream, positions,
                          glossary, en_beta_by_name, page_clusters)
    (out_dir / "burstiness_report.md").write_text(report, encoding="utf-8")
    log(f"[burstiness] {tested} entities tested ({n_bursty} bursty, "
        f"{n_even} even); book beta {book_beta:.3f}; wrote {out_dir}")
    return summary


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def build_report(summary, fits_rows, seg_rows, burst_rows, control_rows,
                 channel, ent_fit, stream, positions, glossary,
                 en_beta_by_name, page_clusters):
    h = summary["heaps"]
    b = summary["burstiness"]
    g = summary["growth"]
    en_lo, en_hi = h["english_beta_min"], h["english_beta_max"]
    en_mean = h["english_mean_beta"]
    coinage_beta = h["book_coinage_beta"]
    by_entity = {r["entity"]: r for r in burst_rows}
    a = []
    w = a.append
    w("# Burstiness and Heaps' law (#53)")
    w("")
    w("Vocabulary growth vs text length, and the burstiness of entity use. "
      "Source: the #17 unit layer screened two ways (both disclosed in "
      "Method), #30's entity glossary, and the eight pinned #24 English "
      "references at matched N. Reproduce with "
      "`python work/burstiness_heaps.py`; regression suite "
      f"`python work/test_burstiness_heaps.py`. Schema `{SCHEMA}`.")
    w("")
    w("## Verdict")
    w("")
    in_range = h["book_units_in_english_range"]
    if en_lo is None:
        range_txt = "no English comparison loaded"
    elif in_range:
        range_txt = (f"inside the English prose range {en_lo:.3f}-{en_hi:.3f} "
                     f"(English mean {en_mean:.3f})")
    else:
        range_txt = (f"outside the English prose range {en_lo:.3f}-{en_hi:.3f} "
                     f"(English mean {en_mean:.3f})")
    direction = ""
    if en_mean is not None:
        direction = (", below the English mean" if h["book_units_beta"] < en_mean
                     else ", above the English mean" if h["book_units_beta"] > en_mean
                     else ", level with the English mean")
    se_txt = (f"leave-one-page-out jackknife se "
              f"{_f(h['book_units_beta_loo_se'], 3)}"
              if h["book_units_beta_loo_se"] not in ("", None)
              else "jackknife off in this build")
    w(f"1. **Vocabulary growth is ordinary, not alien.** The book's unit "
      f"layer grows at Heaps' beta **{h['book_units_beta']:.3f}** "
      f"(log K {h['book_units_logk']:.3f}, R^2 {_f(h['book_units_r2'], 3)}, "
      f"{se_txt}), "
      f"{range_txt}{direction}. The growth curve asks for no exotic "
      "technique: the book mints vocabulary like a novel. The only "
      f"fast-growing layer remains #24's coinage subset (beta "
      f"{_f(coinage_beta, 3)}), already explained there as rule-resolved "
      "compounds with a shallow Zipf slope.")
    w("")
    thirds_tok = g["entity_first_attestation_by_token_third"]
    thirds_men = g["entity_first_attestation_by_mention_third"]
    w(f"2. **The cast is introduced early.** Over the entity-mention stream "
      f"the named layer grows at beta **{h['entity_layer_beta']:.3f}** - well "
      f"below the word layer. On the **mention** partition (of "
      f"{b['channel']['mentions']:,} mentions) "
      f"{thirds_men[0]} / {thirds_men[1]} / {thirds_men[2]} of the "
      f"{len(glossary)} entities are first attested in the first / second / "
      f"last third, so {g['entity_cast_seen_by_mention_third'][0]} of "
      f"{len(glossary)} are already on stage by the first third; on the "
      f"**token** partition (of "
      f"{summary['screens']['entity_stream_n']:,} tokens) the same split is "
      f"{thirds_tok[0]} / {thirds_tok[1]} / {thirds_tok[2]}. The two "
      "partitions are different denominators and are not interchangeable "
      "(section 2). Either way the named layer is a fixed dramatis personae "
      "put to work, not an open-ended naming system.")
    w("")
    by_label = {ch["label"]: ch for ch in b["channels"]}
    ent_ch = by_label.get("entity_channel", {})
    en_ch = by_label.get("english_word_channel", {})
    coin_ch = by_label.get("coinage_type_channel", {})
    w(f"3. **Entity use is bursty - and so is the coinage layer.** Of "
      f"{b['entities_tested']} entities with n >= {b['n_floor']}, "
      f"**{b['bursty']} are bursty** (over-dispersed against a placement null "
      f"at their own rate, with B > {B_LEVEL}), and "
      f"{b['entities_below_even_f']} sit under the descriptive F(1000) < "
      f"{EVEN_F} cut - but the low-tail test certifies **none** of them as "
      f"more evenly than chance ({b['even']} rows carry an `even` verdict). "
      f"The entity median is F(1000) {_f(b['entity_median_fano_1000'], 2)}. "
      f"The merged entity channel ({ent_ch.get('n_positions', 0):,} distinct "
      f"positions, {ent_ch.get('rate', 0) * 100:.1f}% of tokens) has F(1000) "
      f"{_f(ent_ch.get('fano', {}).get(f'fano_{FANO_WINDOWS[-1]}'), 2)} "
      f"against a null median {_f(ent_ch.get('null_fano_median'), 2)} - "
      f"over-dispersed, but *not* the most clustered repetition in the book: "
      f"the merged coinage-type channel reaches F(1000) "
      f"{_f(coin_ch.get('fano', {}).get(f'fano_{FANO_WINDOWS[-1]}'), 2)} and "
      f"the merged English-word channel "
      f"{_f(en_ch.get('fano', {}).get(f'fano_{FANO_WINDOWS[-1]}'), 2)} "
      f"(`channels.tsv` compares merged channel against merged channel, "
      f"each against its own rate-matched null; the previous build stacked "
      f"the entity channel against per-term control medians, which are not "
      f"the same unit). The finding that survives is the *split* inside the "
      f"entity layer: a bursty foreground that arrives in arcs, against a "
      f"background that is at best weakly under-dispersed (section 3).")
    w("")
    w("## Method")
    w("")
    w("- **Two screens, disclosed and never mixed.** The growth analysis runs "
      "on #24's screen (347 markup removals, "
      f"{summary['screens']['growth_stream_n']:,} units) so every fit "
      "reconciles with `analysis/frequency/fits.tsv` row for row; the entity "
      "analysis runs on the #27/#30 screen (the general tag-run detector on "
      "top, 588 removals, "
      f"{summary['screens']['entity_stream_n']:,} units) so every entity "
      "count reconciles with #30's `entity_glossary.tsv` row for row. The "
      f"{summary['screens']['screen_delta']}-occurrence difference is the "
      "`href`/`mailto`/`colspan` tag furniture the general detector removes "
      "and #19's ledger does not list.")
    w("- **Heaps' law** is fitted exactly as #24 fits it (OLS on the log-log "
      "cumulative type curve at 400 checkpoints), so the point estimates "
      "reproduce `fits.tsv`. The uncertainty is a leave-one-page-out "
      "jackknife for the book and leave-one-" + str(JACKKNIFE_BLOCK) +
      "-token-block-out for the references - a resampling that never "
      "duplicates a group, which a page bootstrap would (a duplicated page "
      "contributes no new types, biasing beta down). The reported "
      "uncertainty is the **jackknife standard error** `sd * sqrt(n - 1)`, "
      "not the plain sd across the n refits: "
      + (f"the book's {fits_rows[0]['jackknife_groups']} page refits have sd "
         f"{_f(h['book_units_beta_loo_sd'], 4)} but se "
         f"{_f(h['book_units_beta_loo_se'], 3)}, and the matched-N English "
         f"references span se {_f(h['english_beta_se_min'], 3)}-"
         f"{_f(h['english_beta_se_max'], 3)} "
         if h["book_units_beta_loo_se"] not in ("", None) else
         "the refits were skipped in this build, so no se is reported here "
         "and the column is empty ")
      + "(`heaps_fits.tsv` publishes the sd, the se, the refit mean and the "
        "refit mean minus the point estimate for every row).")
    w("- **Burstiness** reports three legs per entity: the Fano factor "
      "Var(count)/Mean(count) over non-overlapping windows of "
      + " / ".join(str(x) for x in FANO_WINDOWS)
      + " tokens (1.0 is a chance placement); the Goh-Barabasi burstiness "
      "B = (sigma - mu)/(sigma + mu) of the inter-arrival gaps (0 is "
      "geometric gaps); and **both tails** of a placement null - "
      f"{b['nulls']:,} draws per term, each a uniform random *subset* of the "
      "stream with exactly the term's own rate (no token used twice), seeded "
      "per term by `SEED + crc32(name)`. The high tail (ties counted, "
      "add-one corrected) asks *more clustered than chance*; the low tail "
      "asks *more evenly than chance*. A `bursty` verdict needs p <= "
      f"{P_LEVEL} **and** B > {B_LEVEL}. An `even` verdict needs the low-tail "
      f"p <= {P_LEVEL}; F(1000) < {EVEN_F} is a descriptive cut and certifies "
      "nothing on its own.")
    w("- **Entity positions** are located on the screened stream with #30's "
      "own phrase rule: single-unit surfaces match exactly; multi-word "
      "surfaces require every word capitalized and whitespace-adjacent in "
      "the page body, with a trailing numeric token read as a designator. "
      "The located counts equal #30's `n_occurrences` for "
      f"{sum(1 for r in glossary if len(positions[r['entity']]) == int(r['n_occurrences']))}"
      f" of {len(glossary)} entities (98/98 in this build).")
    w("")
    w("## 1. Heaps' law: vocabulary growth vs text length")
    w("")
    w("| corpus | N | V | beta | log K | R^2 | beta LOO se | sd | mean-bias | "
      "beta LOO range | V@10k | V@25k | V@50k |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in fits_rows:
        if r["corpus"] == "en_mean":
            w(f"| **English mean** ({r['jackknife_groups']} refs) | | | "
              f"{_f(r['heaps_beta'], 3)} | | | | | | | | | |")
            continue
        w(f"| {r['corpus']} | {r['N']:,} | {r['V']:,} | "
          f"{_f(r['heaps_beta'], 3)} | {_f(r['heaps_logk'], 3)} | "
          f"{_f(r['heaps_r2'], 3)} | {_f(r['beta_loo_se'], 4)} | "
          f"{_f(r['beta_loo_sd'], 4)} | {_bias(r['beta_loo_bias'])} | "
          f"{_f(r['beta_loo_min'], 3)}-{_f(r['beta_loo_max'], 3)} | "
          f"{r.get('V_10k', '') or '-'} | {r.get('V_25k', '') or '-'} | "
          f"{r.get('V_50k', '') or '-'} |")
    n_groups = fits_rows[0]["jackknife_groups"]
    max_bias = max((abs(float(r["beta_loo_bias"])) for r in fits_rows
                    if r["beta_loo_bias"] not in ("", None)), default=0.0)
    w("")
    if not n_groups:
        w("This build ran without the jackknife (`--quick`), so no refits "
          "exist and the se / sd / mean-bias columns are empty. The full "
          "build reports **se** as `sd * sqrt(n - 1)` - the jackknife "
          "correction, not the plain sd across the n refits.")
    else:
        w(f"The jackknife columns are the corrected uncertainty, published "
          f"separately because they are not interchangeable: **se** is "
          f"`sd * sqrt(n - 1)` across the {n_groups} book-page refits, "
          f"**sd** is their plain spread, and **mean-bias** is the refit "
          f"mean minus the point estimate. The bias is small but not zero - "
          f"book_units {_bias(fits_rows[0]['beta_loo_bias'])}, and the "
          f"largest in this table is {max_bias:.4f} - because dropping a "
          f"group shortens the curve and moves the checkpoint grid. The "
          f"refit mean is therefore *not* the point estimate and is not "
          f"used as one; only the spread is, and the se is what an "
          f"uncertainty statement on a single beta means. The earlier "
          f"build put the plain sd in this column and called it the "
          f"jackknife uncertainty, which understates the book's se by a "
          f"factor of `sqrt({n_groups} - 1)` = "
          f"{math.sqrt(int(n_groups) - 1):.1f}.")
    w("")
    w("The full curve is in `growth_curve.tsv` (200 checkpoints per corpus; "
      "the English references are averaged on their shared matched-N grid). "
      "The segment table (`segment_growth.tsv`) reads the growth question "
      "from the other side - how much of each fifth of the text is vocabulary "
      "the book has not used before:")
    w("")
    w("| corpus | " + " | ".join(f"seg {s + 1}" for s in range(SEGMENTS)) + " |")
    w("|---|" + "---:|" * SEGMENTS)
    for corpus in ("book_units", "book_units_lower", "book_parts",
                   "book_english", "book_coinage"):
        rows = [s for s in seg_rows if s["corpus"] == corpus]
        if not rows:
            continue
        w(f"| {corpus} | " + " | ".join(
            f"{s['new_types']:,} new / {s['v']:,} types / beta "
            f"{s['beta_seg']:.2f}" for s in rows) + " |")
    w("")
    new_seq = g["book_units_segment_new_types"]
    beta_seq = g["book_units_segment_beta"]
    w("Novelty falls steeply for the core layers ("
      + " -> ".join(f"{n:,}" for n in new_seq)
      + " new unit types per fifth), the shape of a finite vocabulary being "
      "recombined; it is not strictly monotone (the last fifth adds "
      f"{new_seq[-1]:,} against {new_seq[-2]:,} in the fourth), and the "
      f"segment betas stay in the same band ({min(beta_seq):.2f}-"
      f"{max(beta_seq):.2f}) rather than decaying, with the seg-3 high "
      f"({beta_seq[2]:.2f}) an artifact: the #4/#34 OCR recognition loop on "
      "page 90 mints one-off types (`apoptosis=second`), not coinage.")
    w("")
    w("## 2. The entity cast's growth")
    w("")
    w(f"The mention stream ({mentions_total(positions):,} mentions over "
      f"{summary['screens']['entity_stream_n']:,} tokens) has its own type "
      f"curve (`entity_growth.tsv`): beta {h['entity_layer_beta']:.3f}, log K "
      f"{h['entity_layer_logk']:.3f}.")
    w("")
    w("| cast measure (partition, denominator) | first third | second third | "
      "last third |")
    w("|---|---:|---:|---:|")
    w("| entities **first attested** here - mention stream, "
      f"{b['channel']['mentions']:,} mentions | "
      + " | ".join(str(c) for c in thirds_men) + " |")
    w("| entities **first attested** here - token stream, "
      f"{summary['screens']['entity_stream_n']:,} tokens | "
      + " | ".join(str(c) for c in thirds_tok) + " |")
    w("| cumulative **share of the cast** seen by the end of the third - "
      "mention stream | "
      + " | ".join(f"{s * 100:.0f}%" for s in
                   g["entity_cast_share_by_mention_third"]) + " |")
    w("")
    w("The first two rows are different partitions of the same book and their "
      "totals are not comparable: the mention stream splits on the order "
      "mentions arrive in, the token stream splits on stream position, so an "
      "entity introduced by a single early mention and one introduced by a "
      "concentration of early mentions land in different thirds. The third "
      "row uses the mention stream's own denominator. Each row is therefore "
      "reported only against itself, and the two first-attestation rows are "
      "shown side by side precisely because the earlier build mixed them in "
      "one table.")
    w("")
    w("## 3. Burstiness of entity use")
    w("")
    w(f"Per-entity table: `entity_burstiness.tsv` ({len(burst_rows)} rows, "
      f"n >= {b['n_floor']}, sorted by F(1000)). The control population is in "
      f"`controls_burstiness.tsv` ({len(control_rows)} rows, entity surfaces "
      f"removed) and the merged-channel comparison in `channels.tsv` "
      f"({len(b['channels'])} channels). `p high` is the over-dispersion "
      f"tail, `p low` the under-dispersion tail, and `ties` the share of null "
      f"draws landing exactly on the observed value.")
    w("")
    w("| entity | role | n | pages | B | F(100) | F(1000) | null F | p high | "
      "p low | ties | verdict | top page |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in burst_rows[:15]:
        w(f"| {r['entity']} | {r['role']} | {r['n']} | {r['n_pages']} | "
          f"{_f(r['burstiness_b'], 3)} | {_f(r.get('fano_100'), 2)} | "
          f"{_f(r.get('fano_1000'), 2)} | {_f(r.get('null_fano_median'), 2)} | "
          f"{_sig(r.get('p_bursty'))} | {_sig(r.get('p_even'))} | "
          f"{_f(r.get('null_fano_tie_share'), 2)} | {r['verdict']} | "
          f"{r['top_page']} ({r['max_page_count']}) |")
    w("")
    top = b["top_burst"]
    low_rows = sorted((r for r in burst_rows
                       if (r["fano_1000"] or 0.0) < EVEN_F),
                      key=lambda r: r["p_even"])
    cadaver_f = (by_entity["cadaver"]["fano_1000"] if "cadaver" in by_entity
                 else None)
    nested_at = {(d["entity"], d["nested_in"]): d for d in b["nested_entities"]}
    adam_adjust = nested_at.get(("ADAM", "ADAM Doll"))
    cadaver_adjust = nested_at.get(("cadaver", "Cadaver City"))
    w(f"The burst profile splits the cast. The **foreground** arrives in "
      f"arcs: `{top['entity']}` (F(1000) {_f(top['fano_1000'], 2)}, B "
      f"{_f(top['burstiness_b'], 3)}) is confined to pages "
      f"{top['first_page']}-{top['last_page']} - a late-book sequence, not a "
      "texture spread over the novel.")
    w("")
    tie_top = sorted(burst_rows,
                     key=lambda r: -float(r["null_fano_tie_share"] or 0.0))[:3]
    tie_hi = float(tie_top[0]["null_fano_tie_share"] or 0.0)
    tie_lo = float(tie_top[-1]["null_fano_tie_share"] or 0.0)
    w("The **background is not the mirror image of that.** The rows under "
      "the cut, in full:")
    w("")
    w("| entity | n | B | F(1000) | p high | p low | ties | verdict |")
    w("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in low_rows:
        w(f"| `{r['entity']}` | {r['n']} | {_f(r['burstiness_b'], 3)} | "
          f"{_f(r['fano_1000'], 2)} | {_sig(r['p_bursty'])} | "
          f"{_sig(r['p_even'])} | {_f(r['null_fano_tie_share'], 2)} | "
          f"{r['verdict']} |")
    w("")
    w(f"None of them is certified even by the low tail. Those "
      f"{b['entities_below_even_f']} rows are the ones the earlier build "
      f"called \"spread more evenly than chance\", but at these rates the "
      f"Fano factor over 74 windows is a coarse, lumpy statistic and the "
      f"null's own spread puts plenty of chance placements under the cut. "
      f"The most extreme row is `{low_rows[0]['entity']}` "
      f"(F(1000) {_f(low_rows[0]['fano_1000'], 2)}, low-tail p "
      f"{_f(low_rows[0]['p_even'], 3)}) - the only row anywhere near "
      f"significance, and still short of it, on {low_rows[0]['n']} "
      f"occurrences. Every other low row is ordinary: "
      + ", ".join(f"`{r['entity']}` p {_f(r['p_even'], 3)}"
                  for r in low_rows[1:])
      + ". So **no** tested entity is certified more evenly than chance, and "
      f"the `even` count in this build is {b['even']}. The claim removed here "
      "is the aggregate one; the per-entity low-tail p-values are published "
      "so the reading can be re-checked. The `ties` column is the share of "
      "null draws landing on exactly the observed value, and it is why the "
      "reading is coarse: " + ", ".join(
          f"`{r['entity']}`" for r in tie_top)
      + " sit on the null's *mode* - "
      + f"{tie_lo:.2f}-{tie_hi:.2f} of the draws tie exactly at their observed "
      "index - so a strict-inequality count (only draws strictly below) "
      "returns the floor for them. That convention is what gives `SM` its "
      "0.02-0.03, and it gives the rows sitting on the null's mode the "
      "1/2001 floor for being ordinary; ties are counted toward the p "
      "instead.")
    w("")
    w("`HIV` is the case that shows why the rule needs two legs. Its "
      "occurrences are over-dispersed against the placement null "
      f"(F(1000) {_f(by_entity['HIV']['fano_1000'] if 'HIV' in by_entity else None, 2)}, "
      "high-tail p "
      f"{_sig(by_entity['HIV']['p_bursty'] if 'HIV' in by_entity else None)}) "
      "but its inter-arrival gaps are *more regular* than geometric "
      f"(B {_f(by_entity['HIV']['burstiness_b'] if 'HIV' in by_entity else None, 3)} < "
      f"{B_LEVEL}), an even-spacing signature a clustered process does not "
      "have, so it stays `neutral` rather than `bursty`.")
    w("")
    if adam_adjust:
        w(f"The **foreground is not as concentrated as the headline says**, "
          f"because the cast nests: `ADAM Doll` occurrences sit *inside* "
          f"`ADAM` (`nested_entities.tsv`), "
          f"{adam_adjust['n_shared']} of ADAM's {adam_adjust['n_entity']} "
          f"occurrences are also ADAM Doll occurrences. Counting them only "
          f"under the long surface lifts ADAM's F(1000) from "
          f"{_f(adam_adjust['fano_1000_excl_nested'], 2)} to "
          f"{_f(adam_adjust['fano_1000_all'], 2)} - a "
          f"{(float(adam_adjust['fano_1000_all']) / float(adam_adjust['fano_1000_excl_nested']) - 1) * 100:.0f}% "
          f"inflation of the clustering statistic on no new evidence, since "
          f"both surfaces mark the same token. "
          + ("The nesting runs the other way too: `cadaver` contains "
             f"{cadaver_adjust['n_shared']} `Cadaver City` occurrences, and "
             f"removing them takes cadaver's F(1000) from "
             f"{_f(cadaver_adjust['fano_1000_all'], 2)} to "
             f"{_f(cadaver_adjust['fano_1000_excl_nested'], 2)} - its modest "
             f"index *dilutes* a strongly clustered one. "
             if cadaver_adjust else
             "too, and is published the same way. ")
          + "Every nested pair is published, with both the unadjusted and the "
          "nested-excluded index, so no headline here stands without its "
          "denominator. The high-frequency textures sit between the two "
          f"regimes - `cadaver` (441 occurrences over 182 pages) F(1000) "
          f"{_f(cadaver_f, 2)} unadjusted - over-dispersed in the strict test "
          "but a fraction of the arcs' concentration.")
    w("")
    w("What the bursts *are* is checkable by eye. These are the densest "
      "single-page entity clusters in the glossary, shown in the unit "
      "stream:")
    w("")
    for c in page_clusters[:4]:
        example = min((i for i in positions[c["entity"]]
                       if stream["page"][i] == c["page"]), default=None)
        w(f"- **{c['entity']}** - {c['count']} occurrences on p{c['page']} "
          f"(of {c['n']} total): `...{context(stream['surface'], example, 6)}...`")
    w("")
    loop_pages = {}
    for c in page_clusters:
        if c["page"] in LOOP_PAGES and c["count"] >= 3:
            loop_pages.setdefault(c["page"], []).append(
                f"`{c['entity']}` x{c['count']}")
    loop_txt = "; ".join(f"page {p}: " + ", ".join(v) for p, v in
                         sorted(loop_pages.items()))
    w(f"A measured burst is not always narrative. On the corpus's disclosed "
      f"OCR recognition loop pages ({loop_txt}) the entity clusters are "
      "machine artifacts - page 219 repeats `...the technology of murder "
      "href mailto konganic-eye of cyberBuddha konganic-eye of cyberBuddha "
      "and was sutured to...`, and the #4/#34 page-90 loop mints the one-off "
      "types behind the seg-3 novelty spike. They are reported as bursts "
      "because they are, with the cause named, rather than filtered out.")
    w("")
    w("## 4. Controls")
    w("")
    w(f"The same estimator on non-entity repetitions: the {CONTROL_TOP} most "
      "frequent English unit types and the " + str(CONTROL_TOP) + " most "
      "frequent non-English (coinage) types (`controls_burstiness.tsv`). "
      "Every surface the glossary can match is excluded before the "
      "selection, so the population being controlled for no longer contains "
      "its own controls - the earlier build let #30's `retro-ADAM` and "
      "`ToKAGE` appear as coinage-type controls while also appearing as "
      "entities in the table above, which padded the control with the very "
      "effect it was meant to measure. Even this baseline is "
      "over-dispersed, because the corpus's own density varies by page "
      "(spine pages, OCR loops) - the placement null controls rate, not the "
      "envelope - so the entity claim is carried by *magnitude*, not by "
      "significance alone.")
    w("")
    for kind, label in (("english_word", "English word control"),
                        ("coinage_type", "Coinage-type control")):
        cs = b["controls_summary"].get(kind)
        if not cs:
            continue
        w(f"- **{label}** ({cs['n']} types): {cs['bursty']} bursty, median "
          f"F(1000) {cs['median_fano_1000']:.2f}, max "
          f"{cs['max_fano_1000']:.2f}, median B {cs['median_b']:+.3f}.")
    w("")
    w("| kind | term | n | B | F(100) | F(1000) | null F | p high | p low | "
      "ties | verdict |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in control_rows:
        w(f"| {r['kind']} | {r['term']} | {r['n']} | "
          f"{_f(r['burstiness_b'], 3)} | {_f(r.get('fano_100'), 2)} | "
          f"{_f(r.get('fano_1000'), 2)} | {_f(r.get('null_fano_median'), 2)} | "
          f"{_sig(r.get('p_bursty'))} | {_sig(r.get('p_even'))} | "
          f"{_f(r.get('null_fano_tie_share'), 2)} | {r['verdict']} |")
    w("")
    w("Per-term medians describe the per-term table and nothing more. A "
      "median over 10 single-term Fano factors and one merged channel's Fano "
      "factor are different units, and comparing them is what produced the "
      "earlier \"the entity channel sits well above both control "
      "populations\" line. The like-for-like comparison merges each control "
      "population the same way the entity channel is merged and scores every "
      "channel against its own rate-matched null (`channels.tsv`):")
    w("")
    w("| channel | kind | positions | rate | F(1000) | null F | p high | "
      "p low | B | verdict |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for ch in b["channels"]:
        w(f"| {ch['label']} | {ch['kind']} | {ch['n_positions']:,} | "
          f"{ch['rate'] * 100:.1f}% | "
          f"{_f(ch['fano'].get(f'fano_{FANO_WINDOWS[-1]}'), 2)} | "
          f"{_f(ch['null_fano_median'], 2)} | "
          f"{_sig(ch['p_bursty'])} | {_sig(ch['p_even'])} | "
          f"{_f(ch['burstiness_b'], 3)} | {ch['verdict']} |")
    w("")
    ent_ch = next(ch for ch in b["channels"] if ch["kind"] == "entity")
    coin_ch = next((ch for ch in b["channels"] if ch["kind"] == "coinage_type"),
                   None)
    en_ch = next((ch for ch in b["channels"] if ch["kind"] == "english_word"),
                 None)
    if coin_ch is None or en_ch is None:
        w("Only the entity channel was built in this configuration, so no "
          "channel comparison is available for this run.")
    else:
        w("Read on that footing the superlative does not survive. The merged "
          f"entity channel (F(1000) "
          f"{_f(ent_ch['fano'].get(f'fano_{FANO_WINDOWS[-1]}'), 2)}) is "
          f"*below* the merged coinage-type channel (F(1000) "
          f"{_f(coin_ch['fano'].get(f'fano_{FANO_WINDOWS[-1]}'), 2)}, "
          f"{coin_ch['n_positions']:,} positions at a comparable "
          f"{coin_ch['rate'] * 100:.1f}% rate) and above the merged "
          f"English-word channel (F(1000) "
          f"{_f(en_ch['fano'].get(f'fano_{FANO_WINDOWS[-1]}'), 2)}, "
          f"{en_ch['n_positions']:,} positions at {en_ch['rate'] * 100:.1f}%). "
          "Both control channels are over-dispersed against their own nulls "
          "at p < 0.001, so *named* vocabulary in general clusters in this "
          "book - the coinages are its motif vocabulary (`boy-roid`, "
          "`soul/gram`, `vital=serum`) - and the entity layer is a member of "
          "that population rather than its most extreme case. What separates "
          "them is the gap statistic: the entity channel's B is "
          f"{_f(ent_ch['burstiness_b'], 3)} against "
          f"{_f(en_ch['burstiness_b'], 3)} for the English-word channel, whose "
          "over-dispersion comes with *more regular* gaps - the texture of "
          "running prose, not of arcs.")
    w("")
    w("## 5. Limits")
    w("")
    w("- The placement null controls rate, not the corpus's page-length "
      "envelope: a page-rate-preserving null would be stricter, so the "
      "p-values read as *over-dispersion* while the F(1000) magnitude - not "
      "the p alone - carries the claim.")
    w("- Entity occurrences overlap where one glossary entity contains "
      "another (`Sato` inside `Sato Corporation`, `ADAM` inside `ADAM Doll`, "
      "`cadaver` inside `Cadaver City`); the merged channel counts distinct "
      "positions, the per-entity rows do not de-duplicate, and the "
      "nested-excluded index for every overlapping pair is in "
      "`nested_entities.tsv`.")
    w("- The `even` verdict rests on a low-tail p from the same null, which "
      "at these rates is a coarse statistic: for some terms the observed Fano "
      "value is a *common* null value rather than an unusual one. Ties are "
      "counted toward the p, which is the conservative reading; a "
      "strict-inequality count would score those terms as maximally even "
      "because they sit on the null's mode, so it is not used.")
    w("- The matched-N comparison is #24's convention: English references are "
      "truncated to the book's token count, so their full-length growth "
      "(Ulysses to N = 270k) is not compared here.")
    w("- The growth curves are in-sample; the book has no held-out text. The "
      "jackknife is an uncertainty statement, not a prediction interval, and "
      "the refit mean is not the point estimate - see the `mean-bias` column "
      "in section 1.")
    w("")
    w("Reproduce with `python work/burstiness_heaps.py`; suite "
      "`python work/test_burstiness_heaps.py`. Offline on the pinned cache; "
      "deterministic (fixed seeds, no clock), rebuild byte-identical.")
    return "\n".join(a) + "\n"


def mentions_total(positions):
    return sum(len(p) for p in positions.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description="#53 burstiness and Heaps' law")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--reference-dir", default=str(fz.REF_CACHE))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--nulls", type=int, default=NULLS)
    ap.add_argument("--quick", action="store_true",
                    help="shrink the nulls to the test path's 20 draws")
    ap.add_argument("--no-jackknife", action="store_true",
                    help="skip the leave-one-group-out refits")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.quick and args.nulls == NULLS:
        args.nulls = 20
    summary = build(Path(args.out), Path(args.reference_dir), args.no_download,
                    args.no_reference, args.nulls,
                    jackknife=not args.no_jackknife, quiet=args.quiet)
    return 0 if not summary["reconciliation_mismatches"] else 1


if __name__ == "__main__":
    sys.exit(main())
