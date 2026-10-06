#!/usr/bin/env python3
"""#19 Neologism detection -> analysis/neologism/neologisms.tsv

Screen the *unit* layer, never words.  The unit layer is #17's
``analysis/tokenizer/units.tsv`` joined to #18's
``analysis/morphology/decompositions.tsv`` on ``uid`` (and the join is
cross-checked on page+start+end).  A word-frequency list cannot see the
2,046 operator-linked unit types (``gene=TV``, ``soul/gram``,
``suck=blood`` ...), so screening words would silently drop them; this
script screens units.

Pipeline
--------
1. Join units + decompositions on uid (assert byte-exact agreement).
2. Flag markup residue at the *occurrence* level:
   - #17/#18 base set (231 occurrences; tokenizer's strict set is 229),
   - plus an extended set this issue finds: opening tag names carrying an
     attribute (``<a href=...>``, ``<th colspan=...>``) that the strict
     pattern misses.
3. Aggregate the surviving occurrences into case-preserved unit *types*,
   recording counts and first attestation.
4. OOV screen: wordfreq ``en`` wordlist + Zipf frequency.
5. Classify: true coinage / compound of English parts / OCR error, plus
   numeric, proper-noun and markup flags.
6. OCR evidence: #6's confirmed corrections + structural dropped-linker
   drift (a closed fused form that a far more frequent open form of the
   same letters dominates) + edit-distance substitution suspects.
7. Write artifacts under ``analysis/neologism/``.

Reproduce::

    conda activate marker
    python work/neologism_detection.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import wordfreq

try:  # optional; only used for the substitution-suspect evidence table
    from rapidfuzz import distance as rf_distance
    from rapidfuzz import process as rf_process
except ImportError:  # pragma: no cover
    rf_distance = None
    rf_process = None

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
DECOMP = ROOT / "analysis" / "morphology" / "decompositions.tsv"
TOK_DIAG = ROOT / "analysis" / "tokenizer" / "diagnostics.tsv"
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
CORRECTIONS = ROOT / "analysis" / "lexical_correction" / "corrections.tsv"
GAZETTEER = ROOT / "analysis" / "lexical_correction" / "gazetteer.tsv"
OUT = ROOT / "analysis" / "neologism"

SCHEMA = "neologism-detection/1"

# #18's English lexicon floor (wordfreq zipf >= 1.5).  Below this a form is
# recorded as `rare` and is *not* asserted to be a coinage.
ZIPF_FLOOR = 1.5

# HTML tag names that can survive as fake 1-4 character "words" when a tag is
# not stripped.  Only flagged when the occurrence sits inside a tag-shaped run
# (see `_extended_markup`).
TAG_NAMES = {
    "a", "p", "pre", "br", "li", "hr", "th", "td", "tr", "table", "tbody",
    "thead", "body", "html", "head", "span", "div", "img", "meta", "link",
    "b", "i", "u", "em", "strong", "font", "small", "big", "center",
    "script", "style", "title", "form", "input", "button", "ol", "ul", "dl",
    "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6", "nav", "main", "footer",
    "header", "section", "article", "blockquote", "code", "q", "sup", "sub",
}

CATEGORIES = [
    "true_coinage", "compound", "ocr_error", "numeric",
    "proper_noun", "english", "rare_english", "junk", "markup_residue",
]

_ATTR_RE = re.compile(r"^\s+[a-zA-Z][a-zA-Z0-9-]*=")
_PAGE_RE = re.compile(r"<!-- page (\d+) -->")
_DELINK_RE = re.compile(r"[-—=/+]")


def cell(value) -> str:
    """TSV-safe text: tabs/newlines escaped, never a csv reader."""
    s = "" if value is None else str(value)
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(cell(row.get(h, "")) for h in header) + "\n")


def read_tsv(path: Path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_pages() -> dict:
    """Page number -> page body, from the vetted layout markdown (#7)."""
    text = LAYOUT.read_text(encoding="utf-8")
    parts = _PAGE_RE.split(text)
    pages = {}
    for i in range(1, len(parts), 2):
        pages[int(parts[i])] = parts[i + 1]
    return pages


def load_wordlist():
    words = set(wordfreq.iter_wordlist("en"))
    return words


def zipf(word: str) -> float:
    return float(wordfreq.zipf_frequency(word.lower(), "en"))


def delink(word: str) -> str:
    """Remove every junction linker, lower-cased: boy-roid -> boyroid."""
    return _DELINK_RE.sub("", word).lower()


# --------------------------------------------------------------------------
# join + markup screen
# --------------------------------------------------------------------------
def join_units(units, decomps):
    """Join on uid and assert the two layers describe the same span."""
    by_uid = {r["uid"]: r for r in units}
    if len(by_uid) != len(units):
        raise SystemExit("duplicate uid in units.tsv")
    joined = []
    for d in decomps:
        u = by_uid.get(d["uid"])
        if u is None:
            raise SystemExit(f"decomposition uid {d['uid']} missing from units.tsv")
        for key in ("unit", "page", "start", "end"):
            if u[key] != d[key]:
                raise SystemExit(
                    f"uid {d['uid']} disagrees on {key}: {u[key]!r} != {d[key]!r}"
                )
        joined.append((u, d))
    if len(joined) != len(units):
        raise SystemExit("units.tsv and decompositions.tsv are not 1:1")
    return joined


def extended_markup(joined, pages, wordlist, base_markup=frozenset()):
    """Occurrences that are an HTML tag *name* inside a tag-shaped run.

    Strict/base residue is #17/#18's own flag.  This catches the opening tag
    names those patterns miss: ``<a href=...>`` and ``<th colspan=...>``.
    A leading ``<`` alone is not enough for a real English word (the book
    writes its own ``<a defleshed memory>`` blocks), so a word that is a
    normal English entry needs ``>``, ``/>`` or an ``attribute=`` after it.
    A non-word tag name such as ``th`` is flagged on the leading ``<`` alone.
    """
    found = set()
    for u, d in joined:
        surface = d["unit"]
        if surface not in TAG_NAMES or surface != surface.lower():
            continue
        page = pages.get(int(d["page"]))
        if not page:
            continue
        s, e = int(d["start"]), int(d["end"])
        if s <= 0 or page[s - 1] != "<":
            continue
        tail = page[e:e + 30]
        tag_shaped = tail.startswith((">", "/>")) or _ATTR_RE.match(tail)
        non_word = surface not in wordlist or zipf(surface) < ZIPF_FLOOR
        if tag_shaped or non_word:
            found.add(d["uid"])

    # Type-level cleanup: a tag name whose *only* non-`<`-preceded uses are
    # already base markup (e.g. `th`, 94 markup + one mangled `<<th ...`)
    # is markup through and through.  This cannot fire for a real word such
    # as `a`, which has hundreds of ordinary article uses.
    by_tag = defaultdict(list)
    for u, d in joined:
        if d["unit"] in TAG_NAMES and d["unit"] == d["unit"].lower():
            page = pages.get(int(d["page"]))
            s = int(d["start"])
            preceded = bool(page) and s > 0 and page[s - 1] == "<"
            by_tag[d["unit"]].append((d["uid"], preceded,
                                      d["uid"] in found or d["uid"] in base_markup))
    for tag, items in by_tag.items():
        if all(pre or base for _, pre, base in items):
            for uid, pre, _ in items:
                if pre:
                    found.add(uid)
    return found


def ocr_seeds():
    """#6's vetted corrections, keyed by lower-cased source token."""
    seeds = {}
    for row in read_tsv(CORRECTIONS):
        seeds[row["token"].lower()] = {
            "correct": row["corrected"],
            "distance": row["edit_distance"],
            "reason": row["reason"],
            "count": row["count"],
        }
    return seeds


def load_gazetteer():
    g = {}
    for row in read_tsv(GAZETTEER):
        g[row["token"].lower()] = row["category"]
    return g


# --------------------------------------------------------------------------
# OCR structural detection
# --------------------------------------------------------------------------
def dropped_linker_evidence(kept_occ, closed_n_by_lower, skip=frozenset()):
    """For a closed fused form, is there a far more frequent open form?

    ``coldblooded`` is the drift of ``cold-blooded``: same letters, the
    linker lost.  Only fires when the open form clearly dominates, so a real
    closed coinage (``hydromachine``, 62x, vs ``Hydro-machine`` 2x) is not
    called an error.  Core dictionary words are skipped unless #6 already
    confirmed them, since English hyphenation variants are not OCR drift.
    """
    open_map = defaultdict(lambda: [0, None])
    for occ in kept_occ:
        if occ["rule"] == "R5":
            continue
        key = delink(occ["unit"])
        open_map[key][0] += 1
        if open_map[key][1] is None or len(occ["unit"]) < len(open_map[key][1]):
            open_map[key][1] = occ["unit"]
    evidence = {}
    for lower, closed_n in closed_n_by_lower.items():
        if lower in skip:
            continue
        key = delink(lower)
        if key not in open_map:
            continue
        open_n, open_unit = open_map[key]
        if open_n > closed_n and open_n >= 2 * closed_n:
            # Strong = #6's own standard (open form >=20 uses and >=5x the
            # drift form).  Weak = dominated but below that bar; kept as a
            # suspect flag rather than called an error.
            strength = "strong" if (open_n >= 20 and open_n >= 5 * closed_n) else "weak"
            evidence[lower] = {
                "rule": "dropped_linker",
                "correct": open_unit,
                "open_n": open_n,
                "closed_n": closed_n,
                "strength": strength,
            }
    return evidence


def substitution_evidence(closed_types, wordlist):
    """Edit-distance-1/2 neighbours of rare OOV closed forms (suspects only)."""
    if rf_process is None:
        return {}
    pool = [w for w in wordfreq.top_n_list("en", 80000) if len(w) >= 4]
    out = {}
    for lower, info in closed_types.items():
        if lower in wordlist or any(c.isdigit() for c in lower) or not lower.isalpha():
            continue
        if len(lower) < 4 or info["n"] > 3:
            continue
        maxd = 1 if len(lower) < 8 else 2
        cands = rf_process.extract(
            lower, pool, scorer=rf_distance.Levenshtein.distance, limit=6
        )
        good = [(c, int(dist)) for c, dist, _ in cands
                if dist <= maxd and zipf(c) >= 4.5]
        if not good:
            continue
        good.sort(key=lambda x: (-zipf(x[0]), x[1]))
        best, dist = good[0]
        runner = zipf(good[1][0]) if len(good) > 1 else 0.0
        out[lower] = {
            "correct": best,
            "distance": dist,
            "zipf": round(zipf(best), 2),
            "margin": round(zipf(best) - runner, 2),
        }
    return out


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def classify(lower, en_status, rule, closed_kind, proper, gaz):
    """Return (category, is_neologism, is_candidate).

    Categories: true_coinage | compound | ocr_error | numeric |
    proper_noun | english | rare_english | junk | markup_residue.
    """
    if gaz == "junk":
        return "junk", False, False
    if en_status == "dict":
        return "english", False, False
    if en_status == "rare":
        # In the wordlist but marginal (zipf < 1.5).  If #6's gazetteer
        # already names it a book coinage/entity (e.g. `feti`), honour that;
        # otherwise it is not asserted a coinage.  It may still be a book
        # compound, which the rule layer shows.
        if gaz in ("neologism", "entity"):
            return ("proper_noun" if proper else "true_coinage"), True, True
        if rule != "R5" or closed_kind in ("closed_composite", "closed_blend", "stem_suffix"):
            return "compound", True, True
        return "rare_english", False, False
    # ---- OOV from here ----
    if rule != "R5":
        return "compound", True, True
    if closed_kind in ("closed_composite", "closed_blend", "stem_suffix"):
        return "compound", True, True
    if closed_kind == "opaque_coinage":
        if proper:
            return "proper_noun", True, True
        return "true_coinage", True, True
    # atomic_seen but OOV should not happen; be explicit rather than silent
    return "true_coinage", True, True


def main(argv=None):
    ap = argparse.ArgumentParser(description="#19 neologism detection")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)

    units = read_tsv(UNITS)
    decomps = read_tsv(DECOMP)
    joined = join_units(units, decomps)
    pages = load_pages()
    wordlist = load_wordlist()
    gaz = load_gazetteer()
    seeds = ocr_seeds()

    tok_diag = {r["diag"]: r for r in read_tsv(TOK_DIAG)}
    base_strict = int(tok_diag["html_tag_residue"]["n"]) if "html_tag_residue" in tok_diag else None

    # ---- markup screen (occurrence level) ----
    base_markup = {d["uid"] for _, d in joined if d["markup"] == "y"}
    ext_markup = extended_markup(joined, pages, wordlist, base_markup)
    all_markup = base_markup | ext_markup

    kept = [(u, d) for u, d in joined if d["uid"] not in all_markup]
    dropped = [(u, d) for u, d in joined if d["uid"] in all_markup]

    # ---- aggregate to case-preserved unit types ----
    occ_by_type = defaultdict(list)
    for u, d in kept:
        occ_by_type[d["unit"]].append((u, d))

    closed_n_by_lower = defaultdict(int)
    for surface, occs in occ_by_type.items():
        if occs[0][1]["rule"] == "R5":
            closed_n_by_lower[surface.lower()] += len(occs)

    # A core English word with a hyphenated variant in the book is a style
    # variant, not OCR drift; skip it unless #6 already confirmed a correction.
    core_dict_skip = {
        lower for lower in closed_n_by_lower
        if lower in wordlist and zipf(lower) >= ZIPF_FLOOR and lower not in seeds
    }
    dropped_ev = dropped_linker_evidence(
        [d for _, d in kept], closed_n_by_lower, skip=core_dict_skip
    )

    closed_types = {}
    for surface, occs in occ_by_type.items():
        if occs[0][1]["rule"] == "R5":
            closed_types[surface.lower()] = {"n": len(occs), "surface": surface}
    sub_ev = substitution_evidence(closed_types, wordlist)

    # ---- per-type screen ----
    # Case-preserved rows are the unit types; proper-noun evidence is pooled
    # across case variants so `siratori` is recognised by its `Siratori` uses.
    lower_cap = defaultdict(float)
    for surface, occs in occ_by_type.items():
        cf = sum(1 for u, _ in occs if surface[:1].isupper()) / len(occs)
        lower_cap[surface.lower()] = max(lower_cap[surface.lower()], cf)

    rows = []
    for surface, occs in occ_by_type.items():
        occs = sorted(occs, key=lambda ud: (int(ud[1]["page"]), int(ud[1]["start"]), int(ud[0]["uid"])))
        first_u, first_d = occs[0]
        rules = {d["rule"] for _, d in occs}
        kinds = {d["closed_kind"] for _, d in occs if d["closed_kind"]}
        rule = sorted(rules)[0]
        closed_kind = sorted(kinds)[0] if kinds else ""
        lower = surface.lower()
        in_wl = lower in wordlist
        z = zipf(lower)
        en_status = "dict" if (in_wl and z >= ZIPF_FLOOR) else ("rare" if in_wl else "oov")

        # decomposition parts + English part accounting
        detail = first_d["morph_detail"]
        toks = [t for t in detail.split("|") if t]
        names = [t.split(":")[0] for t in toks]
        n_parts = len(names)
        if n_parts <= 1:
            parts_english = "na"
            n_eng = n_oov = 0
        else:
            n_eng = sum(1 for nm in names if nm.lower() in wordlist and zipf(nm) >= ZIPF_FLOOR)
            n_oov = n_parts - n_eng
            parts_english = "y" if n_oov == 0 else "n"

        # numeric / markup
        has_digit = any(d["has_digit"] == "y" for _, d in occs)
        is_numeric = has_digit or bool(re.fullmatch(r"[0-9]+", surface))
        ext_only = all(d["uid"] in ext_markup for _, d in occs)

        # proper-noun / entity signal
        cap_frac = lower_cap[lower]
        gaz_cat = gaz.get(lower, "")
        proper = cap_frac >= 0.5 or gaz_cat == "entity"

        # OCR: a #6-confirmed seed or a *strong* dropped-linker is called an
        # error; a weak dropped-linker or an edit-distance neighbour is only a
        # suspect flag on top of the lexical category.
        ocr = None
        if lower in seeds:
            ocr = {"rule": "seed_correction", "correct": seeds[lower]["correct"],
                   "detail": seeds[lower]["reason"], "strength": "confirmed"}
        elif lower in dropped_ev and dropped_ev[lower]["strength"] == "strong":
            ev = dropped_ev[lower]
            ocr = {"rule": "dropped_linker", "correct": ev["correct"],
                   "detail": f"open form {ev['correct']} x{ev['open_n']} dominates "
                             f"closed x{ev['closed_n']}",
                   "strength": "strong"}
        sub = sub_ev.get(lower)
        if sub is None and lower in dropped_ev and dropped_ev[lower]["strength"] == "weak":
            ev = dropped_ev[lower]
            sub = {"correct": ev["correct"], "distance": 0,
                   "detail": f"open form x{ev['open_n']} dominates closed x{ev['closed_n']}"}

        # category precedence
        if ext_only:
            category, is_neol, is_cand = "markup_residue", False, False
        elif is_numeric:
            category, is_neol, is_cand = "numeric", False, False
        elif ocr is not None:
            category, is_neol, is_cand = "ocr_error", False, True
        else:
            category, is_neol, is_cand = classify(lower, en_status, rule, closed_kind, proper, gaz_cat)

        rows.append({
            "unit": surface,
            "lower": lower,
            "n": len(occs),
            "n_pages": len({d["page"] for _, d in occs}),
            "first_page": first_d["page"],
            "first_start": first_d["start"],
            "first_uid": first_u["uid"],
            "cls": first_d["cls"],
            "rule": rule,
            "channel": first_d["channel"],
            "n_morph": first_d["n_morph"],
            "morph": first_d["morph"],
            "head": first_d["head"],
            "closed_kind": closed_kind,
            "compound_kind": (
                "open_chain" if rule != "R5" else
                "closed_composite" if closed_kind == "closed_composite" else
                "closed_blend" if closed_kind == "closed_blend" else
                "english_suffixation" if closed_kind == "stem_suffix" else ""
            ),
            "zipf_en": round(z, 2),
            "in_wordlist": "y" if in_wl else "n",
            "en_status": en_status,
            "parts": "+".join(names),
            "n_parts": n_parts,
            "n_english_parts": n_eng,
            "n_oov_parts": n_oov,
            "parts_english": parts_english,
            "category": category,
            "is_neologism": "y" if is_neol else "n",
            "is_candidate": "y" if is_cand else "n",
            "proper_noun": "y" if proper else "n",
            "cap_frac": round(cap_frac, 2),
            "gazetteer": gaz_cat,
            "numeric": "y" if is_numeric else "n",
            "ocr_rule": ocr["rule"] if ocr else "",
            "ocr_correct": ocr["correct"] if ocr else "",
            "ocr_strength": ocr["strength"] if ocr else "",
            "ocr_suspect": sub["correct"] if sub else "",
            "ocr_suspect_distance": sub["distance"] if sub else "",
            "markup_base": "y" if any(d["uid"] in base_markup for _, d in occs) else "n",
            "markup_extended": "y" if any(d["uid"] in ext_markup for _, d in occs) else "n",
        })

    rows.sort(key=lambda r: (r["category"], -r["n"], r["unit"]))

    # ---- outputs ----
    header = list(rows[0].keys())
    write_tsv(out_dir / "screening.tsv", header, rows)

    neol_header = header
    neologisms = [r for r in rows if r["is_candidate"] == "y"]
    write_tsv(out_dir / "neologisms.tsv", neol_header, neologisms)

    # Case-folded aggregate of the candidates: the unit layer keeps case
    # (#22 owns case), so `boy-roid`/`Boy-roid` are separate rows above; this
    # view makes the counts comparable to #18/first-pass word counts.
    folded = {}
    for r in neologisms:
        key = r["lower"]
        a = folded.setdefault(key, {
            "lower": key, "n": 0, "n_surface_types": 0, "n_pages": 0,
            "first_page": r["first_page"], "first_start": r["first_start"],
            "categories": set(), "rules": set(), "morph": r["morph"],
            "parts_english": r["parts_english"], "surfaces": set(),
        })
        a["n"] += int(r["n"])
        a["n_surface_types"] += 1
        a["n_pages"] = max(a["n_pages"], int(r["n_pages"]))
        if (int(r["first_page"]), int(r["first_start"])) < (int(a["first_page"]), int(a["first_start"])):
            a["first_page"], a["first_start"] = r["first_page"], r["first_start"]
        a["categories"].add(r["category"])
        a["rules"].add(r["rule"])
        a["surfaces"].add(r["unit"])
    folded_rows = []
    for key, a in sorted(folded.items(), key=lambda kv: -kv[1]["n"]):
        folded_rows.append({
            "lower": key, "n": a["n"], "n_surface_types": a["n_surface_types"],
            "n_pages": a["n_pages"], "first_page": a["first_page"],
            "first_start": a["first_start"],
            "category": "+".join(sorted(a["categories"])),
            "rules": "+".join(sorted(a["rules"])), "morph": a["morph"],
            "parts_english": a["parts_english"],
            "surfaces": "|".join(sorted(a["surfaces"])),
        })
    write_tsv(out_dir / "neologisms_lower.tsv",
              ["lower", "n", "n_surface_types", "n_pages", "first_page",
               "first_start", "category", "rules", "morph", "parts_english", "surfaces"],
              folded_rows)

    ocr_rows = [r for r in rows if r["category"] == "ocr_error" or r["ocr_suspect"]]
    write_tsv(out_dir / "ocr_evidence.tsv", header, ocr_rows)

    num_rows = [r for r in rows if r["category"] == "numeric"]
    write_tsv(out_dir / "numeric_channel.tsv", header, num_rows)

    # markup ledger (occurrence level)
    markup_rows = []
    for u, d in sorted(dropped, key=lambda ud: (int(ud[1]["page"]), int(ud[1]["start"]))):
        markup_rows.append({
            "uid": d["uid"], "page": d["page"], "start": d["start"], "end": d["end"],
            "unit": d["unit"], "base": "y" if d["uid"] in base_markup else "n",
            "extended": "y" if d["uid"] in ext_markup else "n",
            "context": (pages.get(int(d["page"]), "")[max(0, int(d["start"]) - 12):int(d["end"]) + 12]),
        })
    write_tsv(out_dir / "markup_residue.tsv",
              ["uid", "page", "start", "end", "unit", "base", "extended", "context"],
              markup_rows)

    cat_counts = Counter(r["category"] for r in rows)
    neol_occ = sum(r["n"] for r in neologisms)
    kept_types = {d["unit"] for _, d in kept}
    dropped_types = {d["unit"] for _, d in dropped}
    markup_only_types = sorted(dropped_types - kept_types)
    seam_types = sum(1 for r in rows if r["rule"] in ("R2", "R3", "R4", "R6"))
    open_types = sum(1 for r in rows if r["rule"] != "R5")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    diag = {
        "schema": SCHEMA,
        "generated": generated,
        "units_joined": len(joined),
        "screened_occurrences": len(kept),
        "screened_types": len(rows),
        "markup_base_occurrences": len(base_markup),
        "markup_extended_occurrences": len(ext_markup),
        "markup_extended_new_occurrences": len(ext_markup - base_markup),
        "markup_total_occurrences": len(all_markup),
        "tokenizer_html_tag_residue": base_strict,
        "markup_only_types_dropped": len(markup_only_types),
        "neologism_candidate_types": len(neologisms),
        "neologism_candidate_types_case_folded": len(folded_rows),
        "neologism_candidate_occurrences": neol_occ,
        "coinage_types": cat_counts.get("true_coinage", 0) + cat_counts.get("proper_noun", 0),
        "true_coinage_types": cat_counts.get("true_coinage", 0),
        "proper_noun_types": cat_counts.get("proper_noun", 0),
        "compound_types": cat_counts.get("compound", 0),
        "ocr_error_types": cat_counts.get("ocr_error", 0),
        "numeric_types": cat_counts.get("numeric", 0),
        "english_types": cat_counts.get("english", 0),
        "rare_english_types": cat_counts.get("rare_english", 0),
        "junk_types": cat_counts.get("junk", 0),
        "markup_residue_types": cat_counts.get("markup_residue", 0),
        "seam_unit_types_excl_markup": seam_types,
        "open_juncture_types_excl_markup": open_types,
    }
    write_tsv(out_dir / "diagnostics.tsv", ["diag", "value"],
              [{"diag": k, "value": v} for k, v in diag.items()])

    # reconciliation with #17/#18/#6
    recon = [
        {"quantity": "pages", "earlier": "225 (#7/#14/#17/#18)", "here": "225", "note": "same layout markdown"},
        {"quantity": "unit occurrences", "earlier": "74,665 (#17/#18)", "here": str(len(joined)), "note": "1:1 join on uid"},
        {"quantity": "markup-residue occurrences", "earlier": f"{base_strict} (#17 strict)", "here": str(len(all_markup)),
         "note": f"#18 widened base {len(base_markup)}; #19 adds "
                 f"{diag['markup_extended_new_occurrences']} new opening-tag occurrences (a, th) beyond the base"},
        {"quantity": "screened occurrences", "earlier": "74,434 (#18 diagnostics)", "here": str(len(kept)),
         "note": f"outside every markup run; #19 removes "
                 f"{diag['markup_extended_new_occurrences']} opening-tag occurrences more than #18"},
        {"quantity": "seam unit types", "earlier": "1,293 (#17 summary)", "here": str(seam_types),
         "note": "minus `fluid=streaming`, a markup unit"},
        {"quantity": "open-juncture unit types", "earlier": "2,047 (#18)", "here": str(open_types),
         "note": "minus `fluid=streaming`, a markup unit"},
        {"quantity": "#6 OCR corrections", "earlier": "15 (#6)", "here": str(cat_counts.get("ocr_error", 0)),
         "note": "one (#6 `retroadam`) is subsumed: the unit layer already writes `retro-ADAM`"},
    ]
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    summary = {
        "schema": SCHEMA,
        "generated": generated,
        "counts": diag,
        "markup_only_types": markup_only_types,
        "category_types": dict(cat_counts),
        "category_occurrences": {c: sum(r["n"] for r in rows if r["category"] == c) for c in cat_counts},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if not args.quiet:
        print(f"[neologism] screened {len(rows)} unit types / {len(kept)} occurrences")
        for c in CATEGORIES:
            if cat_counts.get(c):
                print(f"  {c:16s} types={cat_counts[c]:5d} occ={summary['category_occurrences'][c]}")
        print(f"[neologism] candidates: {len(neologisms)} types / {neol_occ} occurrences -> {out_dir/'neologisms.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
