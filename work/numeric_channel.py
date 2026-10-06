#!/usr/bin/env python3
"""#21 Numeric and symbol channel catalogue -> analysis/numeric_channel/

Catalogue in-text numbers (666, XX, zero, v1.0) as a separate channel that
may encode counts, versions, or pseudo-code.  Screens the *unit* layer
(#17's ``analysis/tokenizer/units.tsv``), never words, and joins on ``uid``.

The #19 handoff (``analysis/neologism/numeric_channel.tsv``) caught the 18
digit-bearing unit types.  This issue widens the net to the forms a digit
screen cannot see:

- Roman-numeral markers (XX, XXX, XXXXXX, X, ii) — no digits at all;
- spelled-out number words (zero, one, two, four, nine, eleven);
- paratext (ISBN, copyright year) and markup residue (``th colspan=2``),
  which are numbers in the file but not in-text numbers.

Pipeline
--------
1. Load units (#17) and the vetted layout markdown (#7).
2. Classify every unit occurrence into the numeric/symbol channel:
   digit-bearing / roman-marker / spelled-number, with paratext and
   markup-residue subclasses.
3. Aggregate to case-preserved types with counts and first attestation.
4. Sub-inventories: numbered entities (Creature N, Super Cherry N,
   Necrodrome N, NDRO N), measurements (N seconds, N milligram), levels
   (Level N), dimension markers (N-D), the zero channel, roman markers.
5. Loop evidence: numeric contexts that repeat across distant pages.
6. Negative results: version strings, ordinals, pure-symbol units, dates.
7. Write artifacts under ``analysis/numeric_channel/``.

Reproduce::

    conda activate marker
    python work/numeric_channel.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
NEO_NUMERIC = ROOT / "analysis" / "neologism" / "numeric_channel.tsv"
OUT = ROOT / "analysis" / "numeric_channel"

SCHEMA = "numeric-channel/1"

PAGE_RE = re.compile(r"<!-- page (\d+) -->")

# Spelled-out number words that are not ordinary English function words.
# `second` is excluded (190 uses, almost all the ordinary English word);
# `first` is excluded (3 uses, ordinary English).
SPELLED_NUMBERS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion",
}

# Roman-numeral-letter-only strings that are ordinary English words or
# fragments and must not be counted as roman markers.
ROMAN_FALSE_POSITIVES = {
    "mimic", "mix", "vivid", "live", "evil", "civil", "limit", "mild",
    "lid", "did", "dim", "dig", "dip", "lip", "lit", "lil", "ill", "vim",
    "mil", "mid", "civic", "clinic", "valid", "lucid", "liquid", "livid",
    "li", "vi", "ii",  # fragments: "li licking", "vi vi tal", "game ii"
}

# Single-letter units that are the pronoun "I" and its OCR variant "l"
# (98 occurrences: "l gather", "l witness", "l record" ...), not roman markers.
# "m" is markup residue (inside an HTML tag).  Only X and V are genuine
# single-letter roman markers.
SINGLE_ROMAN_EXCLUDE = {"i", "l", "m"}

# Entity names that take a numeric identifier.
ENTITY_PREFIXES = ("creature", "super cherry", "necrodrome", "ndro")

# Measurement units that follow a number.
MEASURE_UNITS = {"seconds", "second", "milligram", "milligrams", "mg"}


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


def load_units():
    """#17 unit layer: one dict per occurrence, in uid order."""
    with UNITS.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in fh]


def load_pages():
    """Page number -> page body, from the vetted layout markdown (#7)."""
    text = LAYOUT.read_text(encoding="utf-8")
    parts = PAGE_RE.split(text)
    pages = {}
    for i in range(1, len(parts), 2):
        pages[int(parts[i])] = parts[i + 1]
    return pages


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def classify_occurrence(u, pages):
    """Classify one unit occurrence into the numeric/symbol channel.

    Returns (in_channel, channel, kind, value, note) where:
    - channel: digit_bearing | roman_marker | spelled_number | paratext |
      markup_residue | None
    - kind: pure_digit | digit_word_fusion | isbn | year | roman |
      spelled | entity_id | measurement_time | measurement_dose |
      level | dimension | count | variable | None
    - value: normalized numeric value where applicable, else ""
    """
    s = u["unit"]
    low = s.lower()
    page = pages.get(int(u["page"]), "")
    start, end = int(u["start"]), int(u["end"])

    # --- digit-bearing -------------------------------------------------
    if re.search(r"[0-9]", s):
        # markup residue: a digit inside an HTML tag attribute (th colspan="2").
        # The book's own "<<" block operator also precedes digits (p21 "<<1 record"),
        # so a "<" only counts as a tag when it is not part of "<<".
        if start > 0 and page[start - 1] == "<" and not (start >= 2 and page[start - 2] == "<"):
            return (True, "markup_residue", "markup", "", "inside HTML tag")
        # check the raw text before the unit for a tag-attribute pattern
        prefix = page[max(0, start - 30):start]
        if re.search(r"colspan\s*=?\s*\"?\s*$", prefix, re.I):
            return (True, "markup_residue", "markup", "", "th colspan attribute")
        if re.match(r"^\d+-\d+-\d+-\d+$", s):
            return (True, "paratext", "isbn", s, "ISBN")
        if re.match(r"^(19|20)\d{2}$", s):
            return (True, "paratext", "year", int(s), "copyright year")
        if re.match(r"^\d+-D$", s):
            return (True, "digit_bearing", "dimension", s, "N-D dimension marker")
        m = re.match(r"^(\d+)mg$", s)
        if m:
            return (True, "digit_bearing", "measurement_dose", int(m.group(1)), f"{s} (dosage)")
        if re.match(r"^\d+$", s):
            return _classify_pure_digit(u, page, start, end)
        return (True, "digit_bearing", "digit_word_fusion", s, "digit+letter fusion")

    # --- roman-numeral markers ----------------------------------------
    if len(s) >= 1 and re.fullmatch(r"[IVXLCDM]+", s, re.I):
        if low in ROMAN_FALSE_POSITIVES:
            return (False, None, None, "", "")
        if len(s) == 1:
            if low in SINGLE_ROMAN_EXCLUDE:
                return (False, None, None, "", "")
            return (True, "roman_marker", "variable", s, "single-letter variable/placeholder")
        return (True, "roman_marker", "roman", s, "roman-numeral marker")

    # --- spelled-out numbers -------------------------------------------
    if low in SPELLED_NUMBERS:
        return _classify_spelled(u, page, start, end, low)

    return (False, None, None, "", "")


def _classify_pure_digit(u, page, start, end):
    """Sub-classify an all-digit unit from its context."""
    s = u["unit"]
    n = int(s)
    # look back up to 20 chars for an entity prefix (case-insensitive:
    # "Creature 13", "Super Cherry 666", "Necrodrome 9", "NDRO 4").
    # rstrip() because the prefix ends with a space ("... Creature 13").
    prefix = page[max(0, start - 20):start].lower()
    for ent in ENTITY_PREFIXES:
        if prefix.rstrip().endswith(ent):
            return (True, "digit_bearing", "entity_id", n, f"{ent} {n}")
    if re.search(r"level\s*$", prefix, re.I):
        return (True, "digit_bearing", "level", n, f"Level {n}")
    # look ahead for a measurement unit
    tail = page[end:end + 20].lower()
    m = re.match(r"\s*(seconds?|milligrams?|mg)\b", tail)
    if m:
        unit = m.group(1)
        if unit.startswith("mg") or "milli" in unit:
            return (True, "digit_bearing", "measurement_dose", n, f"{n} {unit}")
        return (True, "digit_bearing", "measurement_time", n, f"{n} {unit}")
    return (True, "digit_bearing", "count", n, "count/amount")


def _classify_spelled(u, page, start, end, low):
    """Sub-classify a spelled-out number from its context."""
    s = u["unit"]
    prefix = page[max(0, start - 20):start].lower()
    tail = page[end:end + 20].lower()
    if low == "zero":
        if re.search(r"level\s*$", prefix, re.I):
            return (True, "spelled_number", "level", 0, "Level zero")
        if re.search(r"gravity\s*$", prefix):
            return (True, "spelled_number", "null_point", 0, "gravity zero")
        if re.match(r"\s*gravity\b", tail):
            return (True, "spelled_number", "null_point", 0, "zero gravity")
        return (True, "spelled_number", "null_point", 0, "zero (null/origin)")
    if low in ("one", "two", "four", "nine", "eleven"):
        # entity identifier? "eleven of crucified memory", "Nine vital=zones"
        if re.match(r"\s*(of|vital|the)\b", tail):
            return (True, "spelled_number", "entity_id", low, f"{low} (numbered entity)")
        return (True, "spelled_number", "count", low, f"{low} (count)")
    return (True, "spelled_number", "count", low, f"{low} (count)")


# --------------------------------------------------------------------------
# context helper
# --------------------------------------------------------------------------
def build_contexts(units):
    """page -> units sorted by start, for ±k context windows."""
    by_page = defaultdict(list)
    for u in units:
        by_page[int(u["page"])].append(u)
    for p in by_page:
        by_page[p].sort(key=lambda u: int(u["start"]))
    return by_page


def context(by_page, page, start, k=4):
    us = by_page[page]
    idx = next((i for i, u in enumerate(us) if int(u["start"]) == start), None)
    if idx is None:
        return ""
    lo, hi = max(0, idx - k), min(len(us), idx + k + 1)
    return " ".join(u["unit"] for u in us[lo:hi])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description="#21 numeric and symbol channel catalogue")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="dry run")
    args = ap.parse_args(argv)
    OUT = Path(args.out)

    units = load_units()
    pages = load_pages()
    by_page = build_contexts(units)

    # --- classify every occurrence --------------------------------------
    occ = []  # (unit_dict, in_channel, channel, kind, value, note)
    for u in units:
        in_ch, channel, kind, value, note = classify_occurrence(u, pages)
        occ.append((u, in_ch, channel, kind, value, note))

    channel_occ = [o for o in occ if o[1]]
    non_channel = [o for o in occ if not o[1]]

    # --- aggregate to types ---------------------------------------------
    types = {}
    for u, in_ch, channel, kind, value, note in channel_occ:
        key = u["unit"]
        if key not in types:
            types[key] = {
                "unit": key,
                "lower": key.lower(),
                "n": 0,
                "pages": set(),
                "first_page": int(u["page"]),
                "first_start": int(u["start"]),
                "first_uid": int(u["uid"]),
                "channel": channel,
                "kind": kind,
                "value": value,
                "note": note,
                "context": context(by_page, int(u["page"]), int(u["start"])),
            }
        t = types[key]
        t["n"] += 1
        t["pages"].add(int(u["page"]))
        # keep the earliest attestation
        if (int(u["page"]), int(u["start"])) < (t["first_page"], t["first_start"]):
            t["first_page"] = int(u["page"])
            t["first_start"] = int(u["start"])
            t["first_uid"] = int(u["uid"])
            t["context"] = context(by_page, int(u["page"]), int(u["start"]))

    # --- sub-inventories -------------------------------------------------
    # numbered entities: digit units preceded by an entity name
    entity_rows = []
    for u, in_ch, channel, kind, value, note in channel_occ:
        if kind == "entity_id":
            entity_rows.append({
                "entity": note.rsplit(" ", 1)[0] if " " in note else note,
                "number": u["unit"],
                "unit": u["unit"],
                "page": u["page"],
                "start": u["start"],
                "context": context(by_page, int(u["page"]), int(u["start"]), 2),
            })

    # measurements
    measure_rows = []
    for u, in_ch, channel, kind, value, note in channel_occ:
        if kind in ("measurement_time", "measurement_dose"):
            measure_rows.append({
                "unit": u["unit"],
                "kind": kind,
                "value": value,
                "note": note,
                "page": u["page"],
                "start": u["start"],
                "context": context(by_page, int(u["page"]), int(u["start"]), 2),
            })

    # levels
    level_rows = []
    for u, in_ch, channel, kind, value, note in channel_occ:
        if kind == "level":
            level_rows.append({
                "unit": u["unit"],
                "value": value,
                "note": note,
                "page": u["page"],
                "start": u["start"],
                "context": context(by_page, int(u["page"]), int(u["start"]), 2),
            })

    # zero channel: all units whose surface contains `zero` (case-insens)
    zero_rows = []
    for u in units:
        if "zero" in u["unit"].lower():
            zero_rows.append({
                "unit": u["unit"],
                "page": u["page"],
                "start": u["start"],
                "uid": u["uid"],
                "context": context(by_page, int(u["page"]), int(u["start"]), 2),
            })

    # roman markers
    roman_rows = []
    for u, in_ch, channel, kind, value, note in channel_occ:
        if channel == "roman_marker":
            roman_rows.append({
                "unit": u["unit"],
                "kind": kind,
                "note": note,
                "page": u["page"],
                "start": u["start"],
                "context": context(by_page, int(u["page"]), int(u["start"]), 2),
            })

    # --- loop evidence ----------------------------------------------------
    # numeric-channel contexts that repeat across distant pages
    ctx_map = defaultdict(list)
    for u, in_ch, channel, kind, value, note in channel_occ:
        c = context(by_page, int(u["page"]), int(u["start"]), 5)
        ctx_map[c].append((int(u["page"]), u["unit"]))
    loop_rows = []
    for c, locs in sorted(ctx_map.items(), key=lambda x: -len(x[1])):
        if len(locs) > 1:
            page_list = sorted(set(p for p, _ in locs))
            loop_rows.append({
                "context": c,
                "n": len(locs),
                "pages": ",".join(str(p) for p in page_list),
            })

    # --- negative results -------------------------------------------------
    all_text = "\n".join(pages[p] for p in sorted(pages))
    negatives = {
        "version_strings": len(re.findall(r"\bv?\d+\.\d+\b", all_text)),
        "ordinals": len(re.findall(r"\b\d+(?:st|nd|rd|th)\b", all_text)),
        "dates": len(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", all_text)),
        "times": len(re.findall(r"\b\d{1,2}:\d{2}\b", all_text)),
        "pure_symbol_units": sum(
            1 for u in units if not re.search(r"[A-Za-z0-9]", u["unit"])
        ),
    }

    # --- reconciliation with #19 ------------------------------------------
    neo_types = set()
    if NEO_NUMERIC.exists():
        with NEO_NUMERIC.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                row = dict(zip(header, line.rstrip("\n").split("\t")))
                neo_types.add(row["unit"])
    # all digit-bearing types (digit_bearing + paratext + markup_residue)
    digit_types_here = {
        k for k, v in types.items()
        if v["channel"] in ("digit_bearing", "paratext", "markup_residue")
    }

    # --- write artifacts ---------------------------------------------------
    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)

        # main inventory
        inv_header = [
            "unit", "lower", "n", "n_pages", "first_page", "first_start",
            "first_uid", "channel", "kind", "value", "note", "context",
        ]
        inv_rows = []
        for k in sorted(types, key=lambda k: (-types[k]["n"], k)):
            t = types[k]
            inv_rows.append({
                "unit": t["unit"], "lower": t["lower"], "n": t["n"],
                "n_pages": len(t["pages"]), "first_page": t["first_page"],
                "first_start": t["first_start"], "first_uid": t["first_uid"],
                "channel": t["channel"], "kind": t["kind"],
                "value": t["value"], "note": t["note"],
                "context": t["context"],
            })
        write_tsv(OUT / "numeric_channel.tsv", inv_header, inv_rows)

        # sub-inventories
        def _ent_key(r):
            try:
                return (r["entity"], 0, -int(r["number"]))
            except ValueError:
                return (r["entity"], 1, r["number"])
        write_tsv(OUT / "entity_numbers.tsv",
                  ["entity", "number", "unit", "page", "start", "context"],
                  sorted(entity_rows, key=_ent_key))
        def _meas_key(r):
            try:
                return (r["kind"], 0, -int(r["value"]))
            except (ValueError, TypeError):
                return (r["kind"], 1, str(r["value"]))
        write_tsv(OUT / "measurements.tsv",
                  ["unit", "kind", "value", "note", "page", "start", "context"],
                  sorted(measure_rows, key=_meas_key))
        write_tsv(OUT / "levels.tsv",
                  ["unit", "value", "note", "page", "start", "context"],
                  sorted(level_rows, key=lambda r: (str(r["value"]), r["page"])))
        write_tsv(OUT / "zero_channel.tsv",
                  ["unit", "page", "start", "uid", "context"],
                  sorted(zero_rows, key=lambda r: (int(r["page"]), int(r["start"]))))
        write_tsv(OUT / "roman_markers.tsv",
                  ["unit", "kind", "note", "page", "start", "context"],
                  sorted(roman_rows, key=lambda r: (r["unit"], int(r["page"]))))
        write_tsv(OUT / "loop_evidence.tsv",
                  ["context", "n", "pages"],
                  sorted(loop_rows, key=lambda r: -r["n"]))

        # diagnostics
        chan_counts = Counter(o[2] for o in channel_occ)
        kind_counts = Counter(o[3] for o in channel_occ)
        diag_rows = [
            {"metric": "unit_occurrences_total", "value": len(units)},
            {"metric": "channel_occurrences", "value": len(channel_occ)},
            {"metric": "non_channel_occurrences", "value": len(non_channel)},
            {"metric": "channel_types", "value": len(types)},
            {"metric": "zero_channel_occurrences", "value": len(zero_rows)},
            {"metric": "roman_marker_occurrences", "value": len(roman_rows)},
            {"metric": "entity_number_occurrences", "value": len(entity_rows)},
            {"metric": "measurement_occurrences", "value": len(measure_rows)},
            {"metric": "level_occurrences", "value": len(level_rows)},
            {"metric": "loop_contexts_repeated", "value": len(loop_rows)},
        ]
        for ch, n in sorted(chan_counts.items()):
            diag_rows.append({"metric": f"channel:{ch}", "value": n})
        for kd, n in sorted(kind_counts.items()):
            diag_rows.append({"metric": f"kind:{kd}", "value": n})
        for name, n in sorted(negatives.items()):
            diag_rows.append({"metric": f"negative:{name}", "value": n})
        write_tsv(OUT / "diagnostics.tsv", ["metric", "value"], diag_rows)

        # reconciliation
        recon_rows = [
            {"quantity": "unit occurrences", "here": len(units),
             "earlier": "74,665 (#17)", "note": "same unit layer"},
            {"quantity": "digit-bearing types", "here": len(digit_types_here),
             "earlier": f"{len(neo_types)} (#19 handoff)",
             "note": "exact match" if digit_types_here == neo_types else "MISMATCH"},
            {"quantity": "digit-bearing occurrences", "here": sum(
                t["n"] for t in types.values()
                if t["channel"] in ("digit_bearing", "paratext", "markup_residue")),
             "earlier": "73 (#19 handoff)", "note": ""},
            {"quantity": "channel types (all)", "here": len(types),
             "earlier": "—", "note": "widened net: roman + spelled + paratext + markup"},
        ]
        write_tsv(OUT / "reconciliation.tsv",
                  ["quantity", "here", "earlier", "note"], recon_rows)

        # summary.json
        summary = {
            "schema": SCHEMA,
            "unit_occurrences": len(units),
            "channel_occurrences": len(channel_occ),
            "channel_types": len(types),
            "by_channel": dict(chan_counts),
            "by_kind": dict(kind_counts),
            "zero_channel_occurrences": len(zero_rows),
            "roman_marker_occurrences": len(roman_rows),
            "entity_number_occurrences": len(entity_rows),
            "measurement_occurrences": len(measure_rows),
            "level_occurrences": len(level_rows),
            "loop_contexts_repeated": len(loop_rows),
            "negatives": negatives,
            "digit_types_match_19": digit_types_here == neo_types,
        }
        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --- console summary ---------------------------------------------------
    if not args.quiet:
        print(f"units: {len(units)}  channel occurrences: {len(channel_occ)}  "
              f"channel types: {len(types)}")
        print("by channel:", dict(chan_counts))
        print("by kind:", dict(kind_counts))
        print("negatives:", negatives)
        print(f"digit types match #19: {digit_types_here == neo_types}")
        if not args.no_write:
            print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
