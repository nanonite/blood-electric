#!/usr/bin/env python3
"""#22 Case and emphasis inventory -> analysis/case_emphasis/

Extract every ALL-CAPS token and decide, with evidence, whether it is a
**header**, a **scream**, or a **label**.  Report the inventory against the
section headers (#7).

Where the task text says "tokens", this screens the *token* layer of #17
(``analysis/tokenizer/tokens.tsv``), because case is a property of the token,
not of the maximal unit: ``retro-ADAM`` is one mixed-case unit whose ``ADAM``
is *not* an all-caps token, while a standalone ``ADAM`` is.  Every occurrence
still carries its ``unit`` (uid) so it joins the #17 unit layer and, through
it, #18/#19/#21.

Decision rule (each signal is measured, the label override is curated)
---------------------------------------------------------------------
1. ``header``  - the token lives in a layout heading region (#7).  The title
   is upper-cased *because it is a heading*, so even its function words
   (``OF``, ``THIS``) are structural, not emphasis.
2. ``function``- the single letters ``I`` (pronoun) and ``A`` (article).  They
   are upper-case only because they are one character; they are excluded from
   the emphasis decision and reported separately.
3. ``label``   - a designator: a single-letter variable (``G``, ``K``, ``X``),
   a curated acronym (``TV``, ``DNA``, ``HDD``), or a curated proper name /
   invented entity (``ADAM``, ``ToK``-adjacent ``NDRO``, ``BABEL``).  A
   curated type is a label even when an English lowercase form also exists
   (``ANDROID``, ``OKAMA``), because its all-caps use is referential.
4. ``scream``  - an ordinary English word upper-cased for emphasis or as a
   console command.  Test: the exact lowercase form occurs in the book, or
   ``wordfreq`` Zipf(en) >= 2.5.  Subtypes: ``command`` (imperative/console
   verb) and ``emphasis`` (everything else).
5. otherwise ``label`` - an OOV all-caps form with no English word behind it.

The curated label inventory is listed in ``LABEL_REASON`` with a reason, so a
reviewer can audit every override.  Everything else follows the measured rule.

Reproduce::

    conda activate marker
    python work/case_emphasis.py
    python work/test_case_emphasis.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENS = ROOT / "analysis" / "tokenizer" / "tokens.tsv"
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
SECTIONS = ROOT / "analysis" / "layout_recovery" / "sections.tsv"
MARKUP = ROOT / "analysis" / "neologism" / "markup_residue.tsv"
OUT = ROOT / "analysis" / "case_emphasis"

SCHEMA = "case-emphasis/1"
ZIPF_FLOOR = 2.5          # ordinary-English threshold (see #18/#19)

PAGE_RE = re.compile(r"<!-- page (\d+) -->")

# --- rule inputs -----------------------------------------------------------
# Single letters that are English function words, not emphasis.
FUNCTION_LETTERS = {"I", "A"}

# Console/imperative all-caps verbs: a scream sub-register.
COMMANDS = {
    "LOAD", "PLAY", "SAVE", "REC", "RELOAD", "UPLOAD", "DOWNLOAD",
    "OUTPUT", "DELETE", "STOP", "PUSH", "PUT",
}

# Curated label inventory.  Reason is one of acronym | proper_name |
# invented_designator | measure_prefix.  Drawn from the publisher synopsis /
# jacket entities, the #7 section titles, and #12's `word=word` naming units
# (every name here occurs as a `=`, `/` or hyphen entity in the corpus).
LABEL_REASON = {
    # acronyms / abbreviations / initialisms
    "TV": "acronym", "DNA": "acronym", "HIV": "acronym", "HDD": "acronym",
    "VTR": "acronym", "SM": "acronym", "XX": "acronym", "XXX": "acronym",
    "XXXXXX": "acronym", "ND": "acronym", "CD-ROM": "acronym",
    "ISBN": "acronym", "AAA": "acronym", "BIOS": "acronym", "UV": "acronym",
    "PSION": "acronym", "VTV": "acronym", "KK": "acronym", "4-D": "acronym",
    "3-D": "acronym", "A-Z": "acronym", "ADAM-ROM": "acronym",
    "BABEL-TV": "acronym", "BABEL-HDD": "acronym", "RAPTURE-HDD": "acronym",
    "SM-PLAY": "acronym", "UP-LOAD": "acronym", "STAGE-X": "acronym",
    # proper names / entities
    "ADAM": "proper_name", "ANTI-ADAM": "proper_name", "BABEL": "proper_name",
    "SADO": "proper_name", "SODO": "proper_name", "NDRO": "proper_name",
    "NDROID": "proper_name", "ANDROID": "proper_name", "EVOL": "proper_name",
    "NIHIL": "proper_name", "SODOM": "proper_name", "OMOTYA": "proper_name",
    "OKAMA": "proper_name", "GAIA": "proper_name", "SATO": "proper_name",
    "NEMESIS": "proper_name", "MORBUS": "proper_name", "VAMP": "proper_name",
    "CODA": "proper_name",
    # invented designators
    "SCANIMAL": "invented_designator", "DIGITANIMAL": "invented_designator",
    "GODNAM": "invented_designator", "ROKUDENASI": "invented_designator",
    "GARAKUTA": "invented_designator", "DUSTNIRVERNA": "invented_designator",
    "XTERMINATOR": "invented_designator", "BIOCRASH": "invented_designator",
    "BIOSYNDROME": "invented_designator", "DOWNMIX": "invented_designator",
    "HYDROMANIA": "invented_designator", "M\u00d8NKEY": "invented_designator",
    "KE-MO-NO": "invented_designator", "SLASH-RAPE": "invented_designator",
    "GUT-SUCK": "invented_designator", "GRAVITY-FREE": "invented_designator",
    "ALIEN-COM": "invented_designator", "CODA-CODA": "invented_designator",
    "LEVELDOWN": "invented_designator", "GAMEOVER": "invented_designator",
    "HYPE": "invented_designator",
    # measure / unit prefixes used as designators
    "GIGA": "measure_prefix", "TERA": "measure_prefix", "MAX": "measure_prefix",
    "REV": "measure_prefix",
}


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
def cell(value) -> str:
    """TSV-safe text: tabs/newlines escaped; never read with csv."""
    s = "" if value is None else str(value)
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
            .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(cell(row.get(h, "")) for h in header) + "\n")


def read_tsv(path: Path):
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh]


def load_pages():
    """Page number -> page body, from the vetted layout markdown (#7)."""
    text = LAYOUT.read_text(encoding="utf-8")
    parts = PAGE_RE.split(text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def load_sections():
    return read_tsv(SECTIONS)


def load_markup_uids():
    """uids of #19's HTML-tag-residue units (must not enter as caps words)."""
    if not MARKUP.exists():
        return set()
    return {r["uid"] for r in read_tsv(MARKUP) if r.get("uid")}


# ---------------------------------------------------------------------------
# wordfreq (optional: falls back to an in-book lowercase test)
# ---------------------------------------------------------------------------
try:
    import wordfreq as _wf

    def zipf(word: str) -> float:
        try:
            return float(_wf.zipf_frequency(word.lower(), "en"))
        except Exception:
            return 0.0
except Exception:                                    # pragma: no cover
    def zipf(word: str) -> float:
        return 0.0


def is_caps(text: str) -> bool:
    return bool(text) and text == text.upper() and any(c.isalpha() for c in text)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def classify(text, region, lower_count, z, label_reason):
    """Return (cls, subtype, reason) for one all-caps token occurrence."""
    if region == "header":
        return "header", "title_word", "region=header"
    if text in FUNCTION_LETTERS:
        return "function", "function_letter", "pronoun/article, single letter"
    if len(text) == 1:
        return "label", "single_letter", "single-letter variable/designator"
    if label_reason:
        return "label", label_reason, f"curated {label_reason}"
    if lower_count > 0:
        sub = "command" if text in COMMANDS else "emphasis"
        return "scream", sub, f"lowercase form occurs ({lower_count})"
    if z >= ZIPF_FLOOR:
        sub = "command" if text in COMMANDS else "emphasis"
        return "scream", sub, f"english word, zipf={z:.2f}"
    return "label", "oov_designator", f"oov all-caps, zipf={z:.2f}"


def context(pages, page, start, end, k=45):
    body = pages.get(page, "")
    left = body[max(0, start - k):start].replace("\n", "\\n")
    right = body[end:end + k].replace("\n", "\\n")
    return left, right


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description="#22 case and emphasis inventory")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="dry run")
    args = ap.parse_args(argv)
    OUT = Path(args.out)

    tokens = read_tsv(TOKENS)
    units = {r["uid"]: r for r in read_tsv(UNITS)}
    pages = load_pages()
    sections = load_sections()
    markup_uids = load_markup_uids()

    word_tokens = [t for t in tokens if t["kind"] == "WORD"]
    lower_counts = collections.Counter(
        t["text"] for t in word_tokens if t["text"].islower())
    zcache: dict[str, float] = {}

    def zipf_cached(text):
        if text not in zcache:
            zcache[text] = zipf(text)
        return zcache[text]

    # --- classify every all-caps occurrence --------------------------------
    occ = []
    for t in tokens:
        if t["kind"] != "WORD" or not is_caps(t["text"]):
            continue
        text = t["text"]
        uid = t.get("unit", "")
        u = units.get(uid, {})
        lc = lower_counts.get(text.lower(), 0)
        z = zipf_cached(text)
        cls, sub, reason = classify(text, t["region"], lc, z,
                                    LABEL_REASON.get(text, ""))
        in_naming = "y" if u.get("sig", "") else ""
        in_fused = "y" if (u.get("sig") or "join" in u.get("class", "")) else ""
        left, right = context(pages, int(t["page"]), int(t["start"]),
                              int(t["end"]))
        occ.append({
            "tid": t["tid"], "page": t["page"], "start": t["start"],
            "end": t["end"], "text": text, "region": t["region"],
            "unit_uid": uid, "unit": u.get("unit", ""),
            "unit_class": u.get("class", ""), "sig": u.get("sig", ""),
            "n_words": u.get("n_words", ""), "container": t.get("container", ""),
            "ast_path": t.get("ast_path", ""),
            "in_naming_unit": in_naming, "in_fused_unit": in_fused,
            "class": cls, "subtype": sub, "reason": reason,
            "lower_exact_n": lc, "zipf_en": round(z, 2),
            "markup_residue": "y" if uid in markup_uids else "",
            "left_ctx": left, "right_ctx": right,
        })

    # --- aggregate to types -------------------------------------------------
    types: dict[str, dict] = {}
    for o in occ:
        k = o["text"]
        a = types.setdefault(k, {
            "type": k, "n": 0, "pages": set(), "header_n": 0, "para_n": 0,
            "classes": collections.Counter(), "subtypes": collections.Counter(),
            "naming_n": 0, "fused_n": 0, "markup_n": 0,
            "lower_exact_n": o["lower_exact_n"], "zipf_en": o["zipf_en"],
            "first_page": int(o["page"]), "first_start": int(o["start"]),
            "first_tid": o["tid"], "left_ctx": o["left_ctx"],
            "right_ctx": o["right_ctx"], "reason": o["reason"],
        })
        a["n"] += 1
        a["pages"].add(int(o["page"]))
        a["classes"][o["class"]] += 1
        a["subtypes"][o["subtype"]] += 1
        if o["region"] == "header":
            a["header_n"] += 1
        else:
            a["para_n"] += 1
        if o["in_naming_unit"]:
            a["naming_n"] += 1
        if o["in_fused_unit"]:
            a["fused_n"] += 1
        if o["markup_residue"]:
            a["markup_n"] += 1
        if (int(o["page"]), int(o["start"])) < (a["first_page"], a["first_start"]):
            a["first_page"], a["first_start"] = int(o["page"]), int(o["start"])
            a["first_tid"] = o["tid"]
            a["left_ctx"], a["right_ctx"] = o["left_ctx"], o["right_ctx"]
            a["reason"] = o["reason"]

    for a in types.values():
        a["class"] = a["classes"].most_common(1)[0][0]
        a["subtype"] = a["subtypes"].most_common(1)[0][0]

    # --- section-header report ---------------------------------------------
    WORD_RE = re.compile(r"[A-Za-z0-9\u00d8][A-Za-z0-9\u00d8'\-]*")
    sec_rows = []
    for s in sections:
        title = s["title"]
        caps = [w for w in WORD_RE.findall(title) if is_caps(w)]
        words = [w for w in WORD_RE.findall(title) if any(c.isalpha() for c in w)]
        sec_rows.append({
            "index": s["index"], "page": s["page"], "level": s["level"],
            "title": title, "n_words": len(words), "caps_tokens": "|".join(caps),
            "n_caps": len(caps),
            "all_caps_title": "y" if words and all(is_caps(w) for w in words) else "n",
        })

    # --- per-page report ----------------------------------------------------
    page_word = collections.Counter()
    for t in word_tokens:
        page_word[int(t["page"])] += 1
    page_caps = collections.Counter()
    page_hdr = collections.Counter()
    for o in occ:
        p = int(o["page"])
        page_caps[p] += 1
        if o["region"] == "header":
            page_hdr[p] += 1
    page_rows = []
    for p in sorted(page_word):
        n = page_word[p]
        c = page_caps.get(p, 0)
        page_rows.append({
            "page": p, "word_tokens": n, "caps_tokens": c,
            "caps_rate": f"{c / n:.4f}" if n else "0",
            "header_caps": page_hdr.get(p, 0),
        })

    # --- all-caps *units* (the #17 join view) ------------------------------
    caps_units = []
    for u in units.values():
        s = u["unit"]
        if is_caps(s) and len(WORD_RE.findall(s)) == 1:
            caps_units.append(s)
    caps_unit_counter = collections.Counter(caps_units)

    # --- class / subtype tallies -------------------------------------------
    class_occ = collections.Counter(o["class"] for o in occ)
    subtype_occ = collections.Counter(o["subtype"] for o in occ)
    class_types = collections.Counter(a["class"] for a in types.values())
    naming_occ = sum(1 for o in occ if o["in_naming_unit"])
    markup_occ = sum(1 for o in occ if o["markup_residue"])
    header_occ = class_occ.get("header", 0)

    summary = {
        "schema": SCHEMA,
        "word_tokens": len(word_tokens),
        "caps_occurrences": len(occ),
        "caps_types": len(types),
        "caps_share_of_words": round(len(occ) / len(word_tokens), 4),
        "by_class_occurrences": dict(class_occ),
        "by_class_types": dict(class_types),
        "by_subtype_occurrences": dict(subtype_occ),
        "header_caps_occurrences": header_occ,
        "body_caps_occurrences": len(occ) - header_occ,
        "header_caps_types": sum(1 for a in types.values() if a["header_n"]),
        "naming_unit_caps_occurrences": naming_occ,
        "markup_residue_caps_occurrences": markup_occ,
        "all_caps_unit_types": len(caps_unit_counter),
        "all_caps_unit_occurrences": sum(caps_unit_counter.values()),
        "sections": len(sections),
        "sections_all_caps": sum(1 for r in sec_rows if r["all_caps_title"] == "y"),
    }

    # --- write artifacts ----------------------------------------------------
    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)

        write_tsv(OUT / "caps_occurrences.tsv", [
            "tid", "page", "start", "end", "text", "region", "unit_uid",
            "unit", "unit_class", "sig", "n_words", "container", "ast_path",
            "in_naming_unit", "in_fused_unit", "class", "subtype", "reason",
            "lower_exact_n", "zipf_en", "markup_residue", "left_ctx", "right_ctx",
        ], occ)

        type_rows = []
        for a in types.values():
            type_rows.append({
                "type": a["type"], "n": a["n"], "n_pages": len(a["pages"]),
                "class": a["class"], "subtype": a["subtype"],
                "classes": "|".join(f"{k}:{v}" for k, v in
                                    sorted(a["classes"].items())),
                "header_n": a["header_n"], "para_n": a["para_n"],
                "lower_exact_n": a["lower_exact_n"], "zipf_en": a["zipf_en"],
                "label_reason": LABEL_REASON.get(a["type"], ""),
                "naming_unit_n": a["naming_n"], "fused_unit_n": a["fused_n"],
                "markup_residue_n": a["markup_n"],
                "first_page": a["first_page"], "first_start": a["first_start"],
                "first_tid": a["first_tid"], "reason": a["reason"],
                "left_ctx": a["left_ctx"], "right_ctx": a["right_ctx"],
            })
        type_rows.sort(key=lambda r: (-r["n"], r["type"]))
        write_tsv(OUT / "caps_types.tsv", [
            "type", "n", "n_pages", "class", "subtype", "classes", "header_n",
            "para_n", "lower_exact_n", "zipf_en", "label_reason",
            "naming_unit_n", "fused_unit_n", "markup_residue_n", "first_page",
            "first_start", "first_tid", "reason", "left_ctx", "right_ctx",
        ], type_rows)

        write_tsv(OUT / "section_headers.tsv", [
            "index", "page", "level", "title", "n_words", "caps_tokens",
            "n_caps", "all_caps_title"], sec_rows)

        write_tsv(OUT / "per_page.tsv", [
            "page", "word_tokens", "caps_tokens", "caps_rate", "header_caps"],
            page_rows)

        write_tsv(OUT / "all_caps_units.tsv", ["unit", "n"],
                  [{"unit": k, "n": v}
                   for k, v in sorted(caps_unit_counter.items(),
                                      key=lambda x: (-x[1], x[0]))])

        diag_rows = [
            {"metric": "word_tokens", "value": len(word_tokens)},
            {"metric": "caps_occurrences", "value": len(occ)},
            {"metric": "caps_types", "value": len(types)},
            {"metric": "header_caps_occurrences", "value": header_occ},
            {"metric": "body_caps_occurrences", "value": len(occ) - header_occ},
            {"metric": "naming_unit_caps_occurrences", "value": naming_occ},
            {"metric": "markup_residue_caps_occurrences", "value": markup_occ},
            {"metric": "sections", "value": len(sections)},
            {"metric": "sections_all_caps", "value": summary["sections_all_caps"]},
            {"metric": "all_caps_unit_types", "value": len(caps_unit_counter)},
            {"metric": "all_caps_unit_occurrences",
             "value": sum(caps_unit_counter.values())},
        ]
        for k, v in sorted(class_occ.items()):
            diag_rows.append({"metric": f"class:{k}", "value": v})
        for k, v in sorted(subtype_occ.items()):
            diag_rows.append({"metric": f"subtype:{k}", "value": v})
        write_tsv(OUT / "diagnostics.tsv", ["metric", "value"], diag_rows)

        recon_rows = [
            {"quantity": "word tokens", "here": len(word_tokens),
             "earlier": "77,485 (#17)", "note": "same token layer"},
            {"quantity": "all-caps tokens", "here": len(occ),
             "earlier": "—", "note": "not previously inventoried"},
            {"quantity": "markup-residue all-caps", "here": markup_occ,
             "earlier": "229 tag units (#17)", "note": "0 overlap: tag names are lowercase"},
            {"quantity": "section headers", "here": len(sections),
             "earlier": "56 (#7)", "note": "same sections.tsv"},
        ]
        write_tsv(OUT / "reconciliation.tsv",
                  ["quantity", "here", "earlier", "note"], recon_rows)

        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    if not args.quiet:
        print(f"word tokens: {len(word_tokens):,}  all-caps: {len(occ):,} "
              f"({summary['caps_share_of_words']*100:.2f}%)  types: {len(types)}")
        print("by class (occ):", dict(class_occ))
        print("by subtype (occ):", dict(subtype_occ))
        print(f"header caps: {header_occ}  body caps: {len(occ)-header_occ}")
        print("top types:", [(a['type'], a['n'], a['class'])
                             for a in sorted(types.values(),
                                             key=lambda a: -a['n'])[:12]])
        if not args.no_write:
            print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
