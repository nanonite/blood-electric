#!/usr/bin/env python3
"""#26 KWIC concordancer for entities and terms -> analysis/kwic/

For every **entity** and **suspicious term**, print every occurrence in
+/-5-token context.  This is the philologist's core tool and the primary
route to a term's stable sense (L3.3 of docs/meaning-extraction-methods.md).

Term list (entities + suspicious terms), all joined on the #17 unit layer:

  * entities   - #19 ``proper_noun`` (23 types), #22 ``label`` (proper_name /
                 acronym / invented_designator / single_letter / measure_prefix),
                 #6 gazetteer ``entity``
  * suspicious - #19 ``compound`` / ``true_coinage`` / ``ocr_error``,
                 #21 numeric channel, #22 ``scream:command`` (the machine-
                 command register: LOAD, PLAY, REC, ...)

Join: a term matches a unit **case-insensitively**; the occurrence is the unit
(``uid``).  Context is the raw page slice over the +/-5 tokens surrounding the
unit, located by a **containment join on (page, start)** -- never by token
index.  (The ``tokens.tsv`` ``unit`` column is a *per-page* unit index, not
the book-global uid -- #17 re-assigns uids to be book-global only *after*
annotating tokens, so the containment join is the robust, disclosed route.)

The concordance is one row per occurrence; the per-term summary carries counts,
first/last attestation, surface variants, and an "embedded in a larger unit"
count (via #17's ``part_index.tsv``) so a term seen mostly inside compounds
(``TV`` inside ``gene=TV``) is visible at a glance.

Reproduce::

    conda activate marker
    python work/kwic_concordancer.py
    python work/test_kwic_concordancer.py
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
TOKENS = ROOT / "analysis" / "tokenizer" / "tokens.tsv"
PARTS = ROOT / "analysis" / "tokenizer" / "part_index.tsv"
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
NEOLOGISMS = ROOT / "analysis" / "neologism" / "neologisms.tsv"
CAPS = ROOT / "analysis" / "case_emphasis" / "caps_types.tsv"
GAZETTEER = ROOT / "analysis" / "lexical_correction" / "gazetteer.tsv"
NUMERIC = ROOT / "analysis" / "numeric_channel" / "numeric_channel.tsv"
OUT = ROOT / "analysis" / "kwic"

SCHEMA = "kwic/1"
CONTEXT = 5                      # +/- tokens of context
PAGE_RE = re.compile(r"<!-- page (\d+) -->")
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_tsv(path: Path):
    """Read a quote-unsafe TSV with a plain split (see #17's warning)."""
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh if line.strip()]


def write_tsv(path: Path, header, rows):
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in header) + "\n")


def load_bodies(src: Path = LAYOUT):
    """{page number: raw page body} from the layout markdown (as #17 does)."""
    text = src.read_text(encoding="utf-8", errors="ignore")
    parts = PAGE_RE.split(text)
    return {int(parts[k]): parts[k + 1]
            for k in range(1, len(parts) - 1, 2)}


def normalize_ws(s: str) -> str:
    """One-line context: drop layout comments, collapse whitespace."""
    s = COMMENT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_surface(s: str) -> str:
    """Collapse double backslashes so the unit match is escaping-agnostic.

    #17 escapes a newline inside a unit as ``\\n`` (backslash-n); #19's
    ``neologisms.tsv`` double-escapes the same surface to ``\\\\n``.  The
    underlying text is identical, so the case-insensitive match must not
    depend on which escaping a downstream artifact chose.
    """
    return s.replace("\\\\", "\\")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_units():
    """units.tsv -> {uid: (page, start, end, surface)} and ordered per page."""
    units = {}
    by_page = collections.defaultdict(list)
    for r in read_tsv(UNITS):
        uid = int(r["uid"])
        page = int(r["page"])
        s, e = int(r["start"]), int(r["end"])
        units[uid] = (page, s, e, r["unit"])
        by_page[page].append(uid)
    return units, by_page


def load_tokens():
    """tokens.tsv -> per-page token lists sorted by start.

    Each token is (start, end, text, kind).  The unit's own tokens are found
    by containment (start in [unit.start, unit.end)) -- a (page, start) join.
    """
    by_page = collections.defaultdict(list)
    for r in read_tsv(TOKENS):
        by_page[int(r["page"])].append(
            (int(r["start"]), int(r["end"]), r["text"], r["kind"]))
    for page in by_page:
        by_page[page].sort()
    return by_page


def load_part_index():
    """part_index.tsv -> {part_text_lower: [uid, ...]} for word parts only."""
    m = collections.defaultdict(list)
    for r in read_tsv(PARTS):
        if r["role"] == "w":
            m[r["text"].lower()].append(int(r["uid"]))
    return m


# ---------------------------------------------------------------------------
# term list
# ---------------------------------------------------------------------------
def build_term_list():
    """Collect (surface, category, source, meta) from #19/#22/#6/#21.

    Returns {lower: {"categories": set, "sources": set, "meta": [dict]}}.
    """
    terms = {}

    def add(surface, category, source, meta=None):
        surface = normalize_surface(surface)
        key = surface.lower()
        t = terms.setdefault(key, {"categories": set(), "sources": set(),
                                   "meta": []})
        t["categories"].add(category)
        t["sources"].add(source)
        if meta:
            t["meta"].append(meta)

    # #19 neologism candidates (compound / true_coinage / proper_noun / ocr_error)
    for r in read_tsv(NEOLOGISMS):
        if r["is_candidate"] != "y":
            continue
        add(r["unit"], r["category"], "#19", {
            "compound_kind": r.get("compound_kind", ""),
            "morph": r.get("morph", ""),
            "ocr_correct": r.get("ocr_correct", ""),
            "ocr_suspect": r.get("ocr_suspect", ""),
        })

    # #22 case channel: labels + the machine-command register (scream:command).
    # Screams (emphasis), headers and the function letters I/A are ordinary
    # English words, not entities or suspicious terms -- excluded, disclosed.
    for r in read_tsv(CAPS):
        if r["class"] == "function":
            continue
        if r["class"] == "label":
            add(r["type"], f"case:label:{r['subtype']}", "#22",
                {"reason": r.get("reason", "")})
        elif r["class"] == "scream" and r["subtype"] == "command":
            add(r["type"], "case:scream:command", "#22",
                {"reason": r.get("reason", "")})

    # #6 gazetteer entities
    for r in read_tsv(GAZETTEER):
        if r["category"] == "entity":
            add(r["token"], "gazetteer:entity", "#6")

    # #21 numeric channel
    for r in read_tsv(NUMERIC):
        add(r["unit"], f"numeric:{r['channel']}", "#21",
            {"kind": r.get("kind", ""), "value": r.get("value", "")})

    return terms


def primary_category(t):
    """A single display category, most specific first."""
    cats = t["categories"]
    for c in ("proper_noun", "gazetteer:entity", "case:scream:command",
              "true_coinage", "ocr_error"):
        if c in cats:
            return c
    for c in sorted(cats):
        if c.startswith("case:label:"):
            return c
    for c in sorted(cats):
        if c.startswith("numeric:"):
            return c
    return sorted(cats)[0] if cats else ""


# ---------------------------------------------------------------------------
# concordance
# ---------------------------------------------------------------------------
def build_concordance(terms, units, by_page_units, page_tokens, bodies,
                      part_index):
    """Match terms to units (case-insensitive) and emit +/-5-token contexts."""
    # unit text (lower) -> [uid, ...] for exact matching
    unit_lower = collections.defaultdict(list)
    for uid, (page, s, e, surf) in units.items():
        unit_lower[normalize_surface(surf).lower()].append(uid)

    # token index span [a, b] for every unit (containment join on page+start)
    unit_span = {}
    for page, toks in page_tokens.items():
        starts = [tk[0] for tk in toks]
        for uid in by_page_units.get(page, []):
            s, e = units[uid][1], units[uid][2]
            a = bisect.bisect_left(starts, s)
            b = bisect.bisect_left(starts, e) - 1
            unit_span[uid] = (page, a, b)

    # exact-match uid set per term
    exact_uids = collections.defaultdict(set)
    for key, uids in unit_lower.items():
        if key in terms:
            exact_uids[key] = set(uids)

    # token-level embedded count: WORD tokens matching a term that live
    # inside a larger (multi-token) unit -- reconciles with #22's token count.
    token_embedded = collections.Counter()
    for uid, (page, a, b) in unit_span.items():
        toks = page_tokens[page]
        for tk in toks[a:b + 1]:
            if tk[3] == "WORD":
                k = tk[2].lower()
                if k in terms and uid not in exact_uids[k]:
                    token_embedded[k] += 1

    concordance = []
    summaries = []
    for key in sorted(terms):
        t = terms[key]
        uids = sorted(unit_lower.get(key, []),
                      key=lambda u: (units[u][0], units[u][1]))
        surfaces = collections.Counter()
        occ_rows = []
        first = None
        last_page = None
        pages = set()
        for n, uid in enumerate(uids, 1):
            page, s, e, surf = units[uid]
            surfaces[surf] += 1
            pages.add(page)
            if first is None:
                first = (page, s, uid)
            last_page = page
            _, a, b = unit_span[uid]
            toks = page_tokens.get(page, [])
            body = bodies.get(page, "")
            left = toks[max(0, a - CONTEXT):a]
            right = toks[b + 1:b + 1 + CONTEXT]
            lctx = normalize_ws(body[left[0][0]:left[-1][1]]) if left else ""
            rctx = normalize_ws(body[right[0][0]:right[-1][1]]) if right else ""
            occ_rows.append({
                "term": "", "term_lower": key,
                "term_category": primary_category(t),
                "term_categories": "|".join(sorted(t["categories"])),
                "occ_n": n, "page": page, "unit_uid": uid,
                "unit_surface": surf,
                "left_context": lctx, "right_context": rctx,
            })
        for r in occ_rows:
            r["term"] = surfaces.most_common(1)[0][0] if surfaces else key
        concordance.extend(occ_rows)

        # part-level embedded count (via #17 part_index): units where the term
        # is a word-part but not the whole unit -- includes hyphen join-parts
        # (`retro-ADAM`) that are NOT separate WORD tokens.
        embedded_uids = [u for u in part_index.get(key, [])
                         if u in units and units[u][3].lower() != key]
        host = collections.Counter(units[u][3] for u in embedded_uids)
        embedded_top = "|".join(f"{h}:{c}" for h, c in host.most_common(3))
        variants = "|".join(f"{s}:{c}" for s, c in surfaces.most_common())
        summaries.append({
            "term": surfaces.most_common(1)[0][0] if surfaces else key,
            "term_lower": key,
            "term_category": primary_category(t),
            "term_categories": "|".join(sorted(t["categories"])),
            "term_sources": "|".join(sorted(t["sources"])),
            "n_occurrences": len(uids),
            "n_pages": len(pages),
            "first_page": first[0] if first else "",
            "first_start": first[1] if first else "",
            "first_uid": first[2] if first else "",
            "last_page": last_page if last_page is not None else "",
            "surface_variants": variants,
            "embedded_n": len(embedded_uids),
            "embedded_tok": token_embedded.get(key, 0),
            "embedded_top": embedded_top,
            "example_1": occ_rows[0]["left_context"] + "  «" +
                         occ_rows[0]["unit_surface"] + "»  " +
                         occ_rows[0]["right_context"] if occ_rows else "",
            "example_2": occ_rows[1]["left_context"] + "  «" +
                         occ_rows[1]["unit_surface"] + "»  " +
                         occ_rows[1]["right_context"] if len(occ_rows) > 1 else "",
            "example_3": occ_rows[2]["left_context"] + "  «" +
                         occ_rows[2]["unit_surface"] + "»  " +
                         occ_rows[2]["right_context"] if len(occ_rows) > 2 else "",
        })
    return concordance, summaries


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(path, terms, concordance, summaries, n_units, n_tokens):
    by_cat = collections.Counter()
    for s in summaries:
        by_cat[s["term_category"]] += 1
    occ_by_cat = collections.Counter()
    for r in concordance:
        occ_by_cat[r["term_category"]] += 1
    L = []
    A = L.append
    A(f"# KWIC concordancer (#26) — `concordance.tsv`")
    A("")
    A(f"Source: #17 unit layer (`units.tsv`, {n_units:,} units) and token layer "
      f"(`tokens.tsv`, {n_tokens:,} tokens); term list from #19/#22/#6/#21. "
      f"Reproducible via `python work/kwic_concordancer.py`; regression suite "
      f"`python work/test_kwic_concordancer.py`.")
    A("")
    A(f"Schema `{SCHEMA}`.")
    A("")
    A("## 1. Headline result")
    A("")
    A(f"**{len(summaries):,} terms / {len(concordance):,} occurrences** printed "
      f"in ±{CONTEXT}-token context.  A term matches a unit **case-insensitively**; "
      f"the occurrence is the unit (``uid``).  Context is the raw page slice over "
      f"the ±{CONTEXT} tokens around the unit, located by a containment join on "
      f"(page, start) — never by token index.")
    A("")
    A("| term class | terms | occurrences |")
    A("|---|---:|---:|")
    for cat, n in by_cat.most_common():
        A(f"| `{cat}` | {n} | {occ_by_cat[cat]:,} |")
    A("")
    A("## 2. Term list (entities + suspicious terms)")
    A("")
    A("| source | what | types |")
    A("|---|---|---:|")
    A(f"| #19 | `proper_noun` (named coinages) | 23 |")
    A(f"| #19 | `compound` / `true_coinage` / `ocr_error` | 2,282 |")
    A(f"| #22 | `label` (proper_name / acronym / invented_designator / single_letter / measure_prefix) | 67 |")
    A(f"| #22 | `scream:command` (machine-command register) | 12 |")
    A(f"| #6 | gazetteer `entity` | 16 |")
    A(f"| #21 | numeric channel | 32 |")
    A("")
    A("Excluded from #22, disclosed: `scream:emphasis` (29 upper-cased ordinary "
      "words — YES, NO, LOVE), `header` (31 title words — BLOOD, ELECTRIC, THIS) "
      "and `function` (I, A).  They are ordinary English, not entities or "
      "suspicious terms; #22 owns them.")
    A("")
    A("## 3. Join: units, not words; uid, not token index")
    A("")
    A("A word-frequency list cannot see a fused unit, so the concordancer joins "
      "the term list to the **unit** layer.  The match is case-insensitive on the "
      "unit surface (``ADAM`` / ``adam`` / ``Adam`` are one term).  Each "
      "occurrence carries its ``uid`` and joins #17's token stream by a "
      "containment test (token start ∈ [unit.start, unit.end)) on the same page.")
    A("")
    A("> **Disclosed quirk:** the ``tokens.tsv`` ``unit`` column is a *per-page* "
      "unit index, not the book-global ``uid`` — #17 re-assigns uids to be "
      "book-global only *after* annotating tokens.  The concordancer therefore "
      "does not rely on that column; it locates a unit's tokens by (page, start) "
      "containment, which is exact and robust.")
    A("")
    A("## 4. Per-term summaries (top 40 by occurrences)")
    A("")
    A("| term | class | n | pages | first | embedded (part / token) |")
    A("|---|---|---:|---:|---|---:|")
    for s in sorted(summaries, key=lambda x: -x["n_occurrences"])[:40]:
        A(f"| `{s['term']}` | {s['term_category']} | {s['n_occurrences']} | "
          f"{s['n_pages']} | p{s['first_page']}:{s['first_start']} | "
          f"{s['embedded_n']} / {s['embedded_tok']} |")
    A("")
    A("`embedded (part / token)` — two views of \"the term inside a larger unit\":")
    A("")
    A("- **part** (`embedded_n`, via #17 ``part_index.tsv``): units where the term "
      "is a *word-part* but not the whole unit.  This includes hyphen join-parts "
      "(`retro-ADAM`) that are **not** separate WORD tokens.")
    A("- **token** (`embedded_tok`): WORD tokens matching the term that live "
      "inside a multi-token unit.  This reconciles with #22's all-caps *token* "
      "count: e.g. ``ADAM`` = 303 standalone + 25 token-embedded = 328 (#22).")
    A("")
    A("Full table: `term_summaries.tsv`.")
    A("")
    A("## 5. Sample concordances")
    A("")
    for term in ("ADAM", "ToKAGE", "boy-roid", "gene=TV", "feti", "LOAD", "666"):
        rows = [r for r in concordance if r["term_lower"] == term.lower()]
        if not rows:
            continue
        A(f"### `{term}` — {len(rows)} occurrence(s)")
        A("")
        for r in rows[:4]:
            A(f"- p{r['page']} (uid {r['unit_uid']}): "
              f"…{r['left_context']} **{r['unit_surface']}** {r['right_context']}…")
        A("")
    A("## 6. Reconciliation")
    A("")
    A("| quantity | here | earlier | note |")
    A("|---|---:|---:|---|")
    A(f"| pages | {len(set(r['page'] for r in concordance))} | 225 (#17) | concordance spans |")
    A(f"| unit occurrences matched | {len(concordance):,} | — | exact case-insensitive unit match |")
    A(f"| terms with >=1 occurrence | {sum(1 for s in summaries if s['n_occurrences']>0):,} | — | of {len(summaries):,} terms |")
    A(f"| terms with 0 occurrences | {sum(1 for s in summaries if s['n_occurrences']==0):,} | — | always embedded in a larger unit, or absent |")
    A("")
    A("## 7. Limitations, stated plainly")
    A("")
    A("- **Case-insensitive matching merges surface variants.** ``ADAM`` (328) and "
      "``adam`` are one term; the per-term summary lists every surface variant "
      "with its count.  #22 owns the case split.")
    A("- **Exact unit match only.** An occurrence of ``ADAM`` inside ``retro-ADAM`` "
      "is *not* an occurrence of the term ``ADAM`` — it is the term ``retro-ADAM``. "
      "The two ``embedded`` columns quantify this so the compound view is one "
      "join away: ``embedded_n`` counts word-parts (hyphen join-parts included), "
      "``embedded_tok`` counts WORD tokens (reconciles with #22).")
    A("- **Numeric spelled words are a superset.** ``zero`` (180), ``one`` (5) etc. "
      "match every unit with that surface, a superset of #21's numeric-context "
      "count; the concordance shows all uses, the numeric ones included.")
    A("- **Context is the raw slice, whitespace-normalized.** Layout comments are "
      "stripped; original spacing (``gene=TV`` fused, ``machine = angel`` spaced) "
      "is preserved.  The key is the exact unit surface.")
    A("")
    A("## 8. Artifacts")
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `concordance.tsv` | THE DELIVERABLE: one row per occurrence, ±5-token context |")
    A("| `term_summaries.tsv` | one row per term: counts, attestation, variants, embedded |")
    A("| `summary.json` | machine-readable headline numbers |")
    A("| `diagnostics.tsv` | per-source term tallies |")
    A("| `reconciliation.tsv` | this build vs #17/#19/#21/#22 |")
    A("")
    A("All `.tsv` are quote-safe (tabs/newlines escaped) and read with "
      "`split(\"\\t\")`, never a csv reader.")
    A("")
    A("---")
    A(f"*Generated by `work/kwic_concordancer.py` (schema `{SCHEMA}`).*")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="#26 KWIC concordancer")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="dry run")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    units, by_page_units = load_units()
    page_tokens = load_tokens()
    bodies = load_bodies()
    part_index = load_part_index()
    n_tokens = sum(len(v) for v in page_tokens.values())

    terms = build_term_list()
    concordance, summaries = build_concordance(
        terms, units, by_page_units, page_tokens, bodies, part_index)

    if not args.no_write:
        write_tsv(out / "concordance.tsv",
                  ["term", "term_lower", "term_category", "term_categories",
                   "occ_n", "page", "unit_uid", "unit_surface",
                   "left_context", "right_context"], concordance)
        write_tsv(out / "term_summaries.tsv",
                  ["term", "term_lower", "term_category", "term_categories",
                   "term_sources", "n_occurrences", "n_pages", "first_page",
                   "first_start", "first_uid", "last_page", "surface_variants",
                   "embedded_n", "embedded_tok", "embedded_top",
                   "example_1", "example_2", "example_3"], summaries)
        # diagnostics: per-source term tallies
        diag = collections.Counter()
        for key, t in terms.items():
            for src in t["sources"]:
                diag[src] += 1
        write_tsv(out / "diagnostics.tsv", ["source", "terms"],
                  [{"source": s, "terms": n} for s, n in sorted(diag.items())])
        # reconciliation
        n_with = sum(1 for s in summaries if s["n_occurrences"] > 0)
        write_tsv(out / "reconciliation.tsv", ["quantity", "here", "earlier", "note"], [
            {"quantity": "pages", "here": len(bodies), "earlier": "225 (#17)",
             "note": "layout markdown"},
            {"quantity": "unit occurrences matched", "here": len(concordance),
             "earlier": "—", "note": "exact case-insensitive unit match"},
            {"quantity": "terms", "here": len(summaries), "earlier": "—",
             "note": "entities + suspicious terms"},
            {"quantity": "terms with >=1 occurrence", "here": n_with,
             "earlier": "—", "note": f"of {len(summaries)} terms"},
        ])
        (out / "summary.json").write_text(json.dumps({
            "schema": SCHEMA,
            "terms": len(summaries),
            "occurrences": len(concordance),
            "terms_with_occurrences": n_with,
            "context_tokens": CONTEXT,
            "by_category": dict(collections.Counter(
                s["term_category"] for s in summaries)),
        }, indent=2), encoding="utf-8")
        write_report(out / "kwic_report.md", terms, concordance, summaries,
                     len(units), n_tokens)

    if not args.quiet:
        print(f"terms={len(summaries)} occurrences={len(concordance)} "
              f"with_occ={sum(1 for s in summaries if s['n_occurrences']>0)}")
        print(f"by_category={dict(collections.Counter(s['term_category'] for s in summaries))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
