#!/usr/bin/env python3
"""#30 Entity and proper-noun induction (dramatis personae) -> analysis/entity/

Builds the book's cast of named entities from three measured signals:

  * **capitalization** - a unit type's share of upper-case occurrences
    (``cap_frac``), and membership in #22's ALL-CAPS label channel;
  * **recurrence** - occurrence count ``n`` and page spread ``n_pages``
    over the #17 unit layer;
  * **collocation** - window partners (PMI / logDice, +/-5 units) and the
    entity-verb direction split (what an entity *does* vs what is *done to*
    it), recomputed here on the same screened sequence so every induced
    entity carries fresh evidence.

The unit layer is screened exactly as #27 screens it (the 347 #19
markup-residue uids plus the general tag-run detector), so the counts
reconcile with #27's frozen ``entity_inventory.tsv`` / ``entity_collocations.tsv``
entity-for-entity.  Joins are on ``uid`` / ``(page, start)``, never on token
index.

Induction is seed-driven and open-ended.  Seeds are the curated proper names
(#19 ``proper_noun``, #22 ``label``, #6 gazetteer ``entity``); on top of them
the script accepts (a) capitalized non-English types with ``n >= 3``,
(b) recurrent adjacent capitalized-unit phrases (``Sato Corporation``,
``Placenta World``, ``Cadaver City``...), (c) numbered phrases
(``Super Cherry 666``, ``Creature 13``), and (d) names attested by a #7
section title (``<<ADAM>>``, ``NDRO``, ``Lecien``...).

**False-positive control** (the reviewer's concern): a non-seed type is
rejected when its surface is an English dictionary word (``zipf >= 2.5``)
regardless of how it is cased - this filters pronouns (``I``), common nouns
used as names (``Doll``, ``Corporation``, ``God``, ``Heaven``), and
title-case common nouns (``Project``, ``Reverse``).  #22 ``function`` class
types and single letters are skipped outright.  Phrase constituents must be
content words (no function words, no operator characters, no single letters),
so ``The ADAM``, ``An ADAM Doll``, ``ADAM Doll/the`` and ``Cadaver City I``
are all rejected.  Every accept/reject decision is recorded with its reason
in ``entity_candidates.tsv``.

Roles (``persona``, ``antagonist``, ``organization``, ``place``, ``creature``,
``machine-system``, ``concept``, ``author``, ``name``) are assigned by an
ordered rule set; each rule cites its evidence, so every glossary row carries
a ``role_evidence`` string.  The rule order is disclosed in the report.

Reproduce::

    conda activate marker
    python work/entity_induction.py
    python work/test_entity_induction.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import wordfreq

# Reuse #27's screening + association code so the counts reconcile exactly.
import collocation as cl

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
NEOLOGISMS = ROOT / "analysis" / "neologism" / "neologisms.tsv"
CAPS = ROOT / "analysis" / "case_emphasis" / "caps_types.tsv"
GAZETTEER = ROOT / "analysis" / "lexical_correction" / "gazetteer.tsv"
SECTIONS = ROOT / "analysis" / "layout_recovery" / "sections.tsv"
INV27 = ROOT / "analysis" / "collocation" / "entity_inventory.tsv"
NUM21 = ROOT / "analysis" / "numeric_channel" / "entity_numbers.tsv"
VERB_LEXICON = ROOT / "analysis" / "collocation" / "verb_lexicon.tsv"
DIAG27 = ROOT / "analysis" / "collocation" / "diagnostics.tsv"
OUT = ROOT / "analysis" / "entity"

SCHEMA = "entity/1"
ZIPF_FLOOR = 2.5          # ordinary-English threshold (#18/#19 convention)
MIN_N = 3                # recurrence floor for non-seed unit candidates
WINDOW = 5               # entity co-occurrence window (+/- units), as #27
CONTEXT = 45             # example-context radius in characters

# English function words: never phrase constituents, never entities.
FUNCTION_WORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "was", "are", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "he", "she", "they",
    "we", "you", "i", "his", "her", "their", "our", "your", "my", "not",
    "no", "but", "if", "then", "than", "so", "such", "all", "any", "each",
    "every", "both", "few", "more", "most", "other", "some", "only",
    "own", "same", "too", "very", "can", "will", "just", "should", "now",
    "also", "even", "still", "yet", "here", "there", "when", "where",
    "what", "which", "who", "how", "why", "above", "below", "under",
    "over", "between", "among", "through", "across", "along", "around",
    "behind", "beyond", "inside", "outside", "within", "without", "toward",
    "towards", "upon", "onto", "into", "out", "off", "up", "down", "far",
    "near", "today", "tomorrow", "yesterday", "ago", "however", "therefore",
    "thus", "hence", "moreover", "furthermore", "nevertheless",
    "nonetheless", "otherwise", "meanwhile", "afterwards", "before",
    "after", "during", "since", "until", "while", "because", "although",
    "though", "unless", "whether", "once", "again", "further", "much",
    "many", "little", "less", "least", "lot", "lots", "plenty", "deal",
    "bit", "part", "piece", "rest", "another", "either", "neither",
    "none", "nobody", "somebody", "anybody", "everybody", "nothing",
    "something", "anything", "everything", "someone", "anyone", "everyone",
}
# Operator characters: a constituent containing one is a compound, not a
# name phrase (``ADAM Doll/the``, ``ADAM Doll++to``).
OPERATOR_CHARS = set("=/+<>*#&|~^%$@!?;:()[]{}\"'`\\")

# Role-rule evidence sets -------------------------------------------------
ORG_HEADS = {"corporation", "company", "corp", "syndicate", "incorporated",
             "inc", "conglomerate", "industries", "enterprises"}
PLACE_HEADS = {"city", "world", "drome", "zone", "town", "land", "planet",
               "necrodrome", "garden", "kingdom", "empire", "domain"}
CREATURE_HEADS = {"doll", "creature", "dog", "robot", "android", "clone",
                  "succubus", "beast", "mutant", "drone"}
CREATURE_MARKERS = ("creature", "cherry", "succubus", "dog", "doll", "beast")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_seeds():
    """Curated proper-name seeds: #19 proper_noun, #22 label, #6 entity.

    Returns {lower: {"categories": set, "sources": set, "surfaces": set}}.
    """
    seeds = {}

    def add(surface, category, source):
        key = surface.lower()
        s = seeds.setdefault(key, {"categories": set(), "sources": set(),
                                   "surfaces": set()})
        s["categories"].add(category)
        s["sources"].add(source)
        s["surfaces"].add(surface)

    for r in cl.read_tsv(NEOLOGISMS):
        if r["category"] == "proper_noun":
            add(r["unit"], "proper_noun", "#19")
    for r in cl.read_tsv(CAPS):
        if r["class"] == "label":
            add(r["type"], f"label:{r['subtype']}", "#22")
    for r in cl.read_tsv(GAZETTEER):
        if r["category"] == "entity":
            add(r["token"], "entity", "#6")
    return seeds


def load_verb_lexicon():
    """{lower: source} from #27's frozen verb lexicon."""
    return {r["verb"].lower(): r["source"] for r in cl.read_tsv(VERB_LEXICON)}


def load_numbered():
    """#21 entity_numbers: {entity_lower: {number: [row, ...]}}.

    Each #21 row is one occurrence; the count per (entity, number) is the
    number of rows.
    """
    numbered = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in cl.read_tsv(NUM21):
        numbered[r["entity"].lower()][r["number"]].append(r)
    return numbered


def load_section_keys():
    """#7 section titles, normalised to candidate keys (set of lower strings)."""
    keys = set()
    for r in cl.read_tsv(SECTIONS):
        t = r["title"].strip()
        if t.startswith("<<") and t.endswith(">>"):
            t = t[2:-2].strip()
        keys.add(t.lower())
    return keys


def load_caps_types():
    """#22 caps_types: {lower: row} for lower-exact counts and class."""
    return {r["type"].lower(): r for r in cl.read_tsv(CAPS)}


def load_acronyms(seeds):
    """#22 label:acronym surfaces (lower) - the technical designator channel."""
    return {k for k, v in seeds.items()
            if any(c.startswith("label:acronym") for c in v["categories"])}


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def build_unit_signals(page_units, excluded_uids):
    """Per unit type (case-folded): n, pages, cap_frac, zipf, surfaces."""
    sig = {}
    for page in sorted(page_units):
        for s, uid, lower, surface, cls in page_units[page]:
            if uid in excluded_uids:
                continue
            e = sig.setdefault(lower, {
                "n": 0, "pages": set(), "first_page": page, "last_page": page,
                "cap_n": 0, "surfaces": collections.Counter(),
                "first_uid": uid, "first_start": s, "cls": cls,
            })
            e["n"] += 1
            e["pages"].add(page)
            e["last_page"] = page
            if surface[:1].isupper():
                e["cap_n"] += 1
            e["surfaces"][surface] += 1
    for e in sig.values():
        e["n_pages"] = len(e["pages"])
        e["cap_frac"] = round(e["cap_n"] / e["n"], 4) if e["n"] else 0.0
        surface = e["surfaces"].most_common(1)[0][0]
        alpha = re.sub(r"[^a-z]", "", surface.lower())
        e["zipf_en"] = round(wordfreq.zipf_frequency(alpha, "en"), 2) \
            if alpha else 0.0
        e["surface"] = surface
    return sig


def is_capitalized(surface: str) -> bool:
    return bool(surface) and surface[:1].isupper() and any(
        c.isalpha() for c in surface)


def is_numeric(surface: str) -> bool:
    return bool(surface) and any(c.isdigit() for c in surface) and not any(
        c.isalpha() for c in surface)


def _adjacent(body, prev_end, next_start):
    """True when only whitespace separates the two offsets."""
    if next_start < prev_end:
        return False
    gap = body[prev_end:next_start]
    return gap.strip() == "" and len(gap) <= 3


def build_phrase_signals(page_units, excluded_uids, bodies, unit_meta):
    """Adjacent capitalized-unit runs (+ optional trailing number) per page.

    Returns {phrase_key: {"n", "pages", "first_page", "last_page",
    "surfaces", "designator", "units"}}.
    A run is a maximal set of consecutive capitalized units joined by
    whitespace only; every contiguous subsequence of length 2..4 is a phrase
    candidate, and a whitespace-adjacent numeric unit is a designator.
    """
    phrases = {}

    def emit(page, units_seq, designator):
        key = " ".join(units_seq).lower()
        p = phrases.setdefault(key, {
            "n": 0, "pages": set(), "first_page": 0, "last_page": 0,
            "surfaces": collections.Counter(), "designator": "",
            "units": units_seq,
        })
        p["n"] += 1
        p["pages"].add(page)
        if not p["first_page"] or page < p["first_page"]:
            p["first_page"] = page
        p["last_page"] = page
        p["surfaces"][" ".join(units_seq)] += 1
        if designator:
            p["designator"] = designator

    for page in sorted(page_units):
        seq = []
        for s, uid, lower, surface, cls in page_units[page]:
            if uid in excluded_uids:
                continue
            seq.append((s, unit_meta[uid][2], uid, surface))
        body = bodies.get(page, "")
        n = len(seq)
        i = 0
        while i < n:
            if not is_capitalized(seq[i][3]):
                i += 1
                continue
            j = i + 1
            while j < n and is_capitalized(seq[j][3]) \
                    and _adjacent(body, seq[j - 1][1], seq[j][0]):
                j += 1
            run = seq[i:j]
            for a in range(len(run)):
                for b in range(a + 2, min(a + 5, len(run) + 1)):
                    sub = run[a:b]
                    emit(page, [u[3] for u in sub], "")
                    after = i + b
                    if after < n and is_numeric(seq[after][3]) \
                            and _adjacent(body, sub[-1][1], seq[after][0]):
                        emit(page, [u[3] for u in sub] + [seq[after][3]],
                             seq[after][3])
            i = j if j > i else i + 1
    for p in phrases.values():
        p["n_pages"] = len(p["pages"])
        p["surface"] = p["surfaces"].most_common(1)[0][0]
    return phrases


# ---------------------------------------------------------------------------
# collocation evidence (recomputed on the same screened sequence as #27)
# ---------------------------------------------------------------------------
def collocation_evidence(page_units, excluded_uids, targets, verbs):
    """Window partners for each target entity.  Same window/association as
    #27, so seed entities reconcile with the frozen tables."""
    seqs = []
    for page in sorted(page_units):
        seq = [(s, uid, lower) for s, uid, lower, surface, cls
               in page_units[page] if uid not in excluded_uids]
        seqs.append((page, seq))
    n_units = sum(len(s) for _, s in seqs)

    partner_freq = collections.Counter()
    entity_freq = collections.Counter()
    for _, seq in seqs:
        partner_freq.update(u[2] for u in seq)
        entity_freq.update(u[2] for u in seq if u[2] in targets)

    pairs = collections.Counter()
    left = collections.Counter()
    right = collections.Counter()
    example = {}
    for page, seq in seqs:
        for i, u in enumerate(seq):
            e = u[2]
            if e not in targets:
                continue
            for j in range(max(0, i - WINDOW), min(len(seq), i + WINDOW + 1)):
                if j == i:
                    continue
                p = seq[j][2]
                key = (e, p)
                pairs[key] += 1
                if j < i:
                    left[key] += 1
                else:
                    right[key] += 1
                if key not in example:
                    example[key] = (page, u[0], seq[j][0])

    out = {}
    for e in targets:
        rows = []
        for (e2, p), n_pair in pairs.items():
            if e2 != e:
                continue
            a = cl.association(n_pair, entity_freq[e], partner_freq[p],
                               n_units)
            rows.append({
                "partner": p, "n_pair": n_pair,
                "n_right": right[(e2, p)], "n_left": left[(e2, p)],
                "is_verb": p in verbs,
                "verb_source": verbs.get(p, ""),
                "logdice": a["logdice"], "pmi": a["pmi"],
                "example": example.get((e2, p)),
            })
        rows.sort(key=lambda r: (-r["logdice"], -r["n_pair"], r["partner"]))
        out[e] = {"n_entity": entity_freq[e], "rows": rows}
    return out, n_units


# ---------------------------------------------------------------------------
# role assignment
# ---------------------------------------------------------------------------
def assign_role(key, surface, kind, sig, numbered_designators, accepted_keys,
                seeds, caps_types, acronyms, ev, compound_evidence,
                org_constituent):
    """Ordered rules; first match wins.  Returns (role, evidence)."""
    lower = key
    n = sig["n"] if sig else 0
    srcs = seeds.get(key, {}).get("sources", set())
    cats = seeds.get(key, {}).get("categories", set())
    curated = bool(srcs)
    ev_rows = ev.get(key, {}).get("rows", []) if ev else []
    does = sorted([r for r in ev_rows if r["is_verb"] and r["n_right"] > 0],
                  key=lambda r: (-r["n_right"], r["partner"]))
    done = sorted([r for r in ev_rows if r["is_verb"] and r["n_left"] > 0],
                  key=lambda r: (-r["n_left"], r["partner"]))
    verb_ev = bool(does or done)
    comp = compound_evidence.get(key, {})

    def vstr(rows, k):
        return ", ".join(f"`{r['partner']}` ({r[k]})" for r in rows[:4])

    def top_partners():
        return vstr([r for r in ev_rows if not r["is_verb"]], "n_pair")

    # 1 author
    if lower == "siratori":
        return ("author",
                f"#19 proper_noun; author meta-reference (n={n})")
    if lower == "kenji siratori" or lower == "kenji":
        return ("author",
                f"author name (n={n}); part of 'Kenji Siratori'")
    # 2 organization: phrase head
    if kind == "phrase":
        head = lower.split()[-1]
        if head in ORG_HEADS:
            return ("organization",
                    f"phrase head `{head}`; n={n}; sources "
                    f"{'|'.join(sorted(srcs)) or '#30'}")
    # 2b organization: single-unit surface variant of an organization phrase
    if org_constituent:
        return ("organization",
                f"surface variant of organization phrase; n={n}; sources "
                f"{'|'.join(sorted(srcs))}")
    # 3 place: head noun
    head = lower.split()[-1]
    if head in PLACE_HEADS:
        return ("place",
                f"place head `{head}`; n={n}; top partners {top_partners()}")
    # 4 creature: numbered + creature marker
    if numbered_designators and any(m in lower for m in CREATURE_MARKERS):
        nums = ", ".join(sorted(numbered_designators))
        return ("creature",
                f"#21 numeric designator {nums}; n={n}")
    # 5 antagonist: ANTI- prefix of an accepted entity
    if lower.startswith("anti-") and lower[5:] in accepted_keys:
        return ("antagonist",
                f"ANTI- prefix of accepted entity `{lower[5:]}`; n={n}")
    # 6 persona: phrase built on a curated proper-name constituent
    if kind == "phrase" and curated:
        return ("persona",
                f"phrase on curated name `{surface}`; n={n}; sources "
                f"{'|'.join(sorted(srcs))}")
    # 7 creature: creature-head phrase
    if kind == "phrase" and head in CREATURE_HEADS:
        return ("creature",
                f"creature head `{head}`; n={n}; sources "
                f"{'|'.join(sorted(srcs)) or '#30'}")
    # 8 machine-system: acronym or binding-seam technical unit
    is_acronym = (surface == surface.upper()
                  and any(c.isalpha() for c in surface)
                  and len(surface) <= 6)
    seam_acronym = False
    if any(op in surface for op in ("=", "/")):
        for part in re.split(r"[=/]", surface):
            if (part and part.isupper() and any(c.isalpha() for c in part)
                    and (part.lower() in acronyms or len(part) <= 4)):
                seam_acronym = True
                break
    if (is_acronym and not curated) or seam_acronym:
        return ("machine-system",
                f"acronym/binding seam `{surface}`; n={n}; sources "
                f"{'|'.join(sorted(srcs)) or '#30'}")
    # 9 persona: curated proper name with verb evidence / recurrence /
    # compound-head recurrence
    if curated and (verb_ev or n >= 3 or sum(comp.values()) >= 5):
        ev_bits = []
        if does:
            ev_bits.append("does " + vstr(does, "n_right"))
        if done:
            ev_bits.append("done-to " + vstr(done, "n_left"))
        if not ev_bits:
            ev_bits.append(f"n={n}")
        if comp:
            ev_bits.append("compound heads "
                           + ", ".join(f"`{k}` ({v})"
                                       for k, v in sorted(comp.items())[:3]))
        return ("persona",
                f"curated proper name ({'|'.join(sorted(cats))}); "
                + "; ".join(ev_bits))
    # 10 concept: recurrent non-curated non-English name
    if n >= MIN_N:
        return ("concept",
                f"recurrent non-curated name (n={n}, cap_frac "
                f"{sig['cap_frac'] if sig else 0}, zipf "
                f"{sig['zipf_en'] if sig else 0})")
    # 11 name: attested proper name (seed or section-titled) with little
    # recurrence
    extra = ""
    if comp:
        extra = ("; compound heads "
                 + ", ".join(f"`{k}` ({v})"
                             for k, v in sorted(comp.items())[:3]))
    return ("name",
            f"attested name (n={n}, sources "
            f"{'|'.join(sorted(srcs)) or '#30'}){extra}")


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------
def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    page_units, unit_meta = cl.load_units()
    bodies = cl.load_bodies()
    markup_spans, markup_uids, markup_types = cl.load_markup(bodies)
    seeds = load_seeds()
    verbs = load_verb_lexicon()
    numbered = load_numbered()
    section_keys = load_section_keys()
    caps_types = load_caps_types()
    acronyms = load_acronyms(seeds)

    # screened unit occurrences (identical to #27)
    excluded_uids = set()
    for page, units in page_units.items():
        for s, uid, lower, surface, cls in units:
            if cl.unit_is_markup(surface, uid, s, page, markup_spans,
                                 markup_uids, markup_types):
                excluded_uids.add(uid)

    unit_sig = build_unit_signals(page_units, excluded_uids)
    phrase_sig = build_phrase_signals(page_units, excluded_uids, bodies,
                                      unit_meta)

    # ---- induction: single-unit candidates --------------------------------
    candidates = []          # dicts for entity_candidates.tsv
    accepted = {}            # key -> entry dict

    for key in sorted(unit_sig):
        sig = unit_sig[key]
        surface = sig["surface"]
        srcs = seeds.get(key, {}).get("sources", set())
        in_sections = key in section_keys
        is_eng = sig["zipf_en"] >= ZIPF_FLOOR
        cap = caps_types.get(key, {})
        # #22 function class (I / A) and single letters are never entities
        if cap.get("class") == "function":
            candidates.append({
                "candidate": surface, "kind": "unit", "n": sig["n"],
                "n_pages": sig["n_pages"], "first_page": sig["first_page"],
                "last_page": sig["last_page"], "cap_frac": sig["cap_frac"],
                "zipf_en": sig["zipf_en"], "is_english":
                    "y" if is_eng else "n", "seed_sources":
                    "|".join(sorted(srcs)), "in_sections":
                    "y" if in_sections else "n", "accepted": "n",
                "reject_reason": "#22 function class (pronoun/article)",
                "role": ""})
            continue
        if len(surface) == 1:
            candidates.append({
                "candidate": surface, "kind": "unit", "n": sig["n"],
                "n_pages": sig["n_pages"], "first_page": sig["first_page"],
                "last_page": sig["last_page"], "cap_frac": sig["cap_frac"],
                "zipf_en": sig["zipf_en"], "is_english":
                    "y" if is_eng else "n", "seed_sources":
                    "|".join(sorted(srcs)), "in_sections":
                    "y" if in_sections else "n", "accepted": "n",
                "reject_reason": "single letter", "role": ""})
            continue
        cand = {
            "candidate": surface, "kind": "unit", "n": sig["n"],
            "n_pages": sig["n_pages"], "first_page": sig["first_page"],
            "last_page": sig["last_page"], "cap_frac": sig["cap_frac"],
            "zipf_en": sig["zipf_en"], "is_english": "y" if is_eng else "n",
            "seed_sources": "|".join(sorted(srcs)),
            "in_sections": "y" if in_sections else "n",
        }
        # false-positive control: English dictionary words that are not
        # curated seeds are rejected regardless of casing (filters I, Doll,
        # Corporation, God, Heaven, Saturn, Venus, ...)
        if not srcs and is_eng:
            candidates.append({**cand, "accepted": "n",
                               "reject_reason":
                                   f"English dictionary word (zipf "
                                   f"{sig['zipf_en']}), not a curated seed",
                               "role": ""})
            continue
        if not srcs and sig["n"] < MIN_N:
            candidates.append({**cand, "accepted": "n",
                               "reject_reason":
                                   f"below recurrence floor (n={sig['n']} "
                                   f"< {MIN_N})", "role": ""})
            continue
        if not srcs and sig["cap_frac"] < 0.5:
            candidates.append({**cand, "accepted": "n",
                               "reject_reason":
                                   f"cap_frac {sig['cap_frac']} < 0.5 and "
                                   f"no seed", "role": ""})
            continue
        candidates.append({**cand, "accepted": "y", "reject_reason": "",
                           "role": ""})
        accepted[key] = {"kind": "unit", "sig": sig, "surface": surface,
                         "srcs": srcs,
                         "cats": seeds.get(key, {}).get("categories", set())}

    # ---- induction: phrase candidates -------------------------------------
    phrase_rows = []
    for key in sorted(phrase_sig):
        p = phrase_sig[key]
        words = key.split()
        designator = p["designator"]
        # constituent filters: no function words, no operator chars, no
        # single letters
        bad = [w for w in words
               if w in FUNCTION_WORDS or len(w) == 1
               or any(c in OPERATOR_CHARS for c in w)]
        has_seed = any(w in seeds for w in words)
        has_non_eng = any(
            wordfreq.zipf_frequency(w, "en") < ZIPF_FLOOR for w in words)
        in_sections = key in section_keys
        row = {
            "phrase": p["surface"], "n": p["n"], "n_pages": p["n_pages"],
            "first_page": p["first_page"], "last_page": p["last_page"],
            "surfaces": "|".join(f"{s}:{c}" for s, c in
                                 p["surfaces"].most_common()),
            "designator": designator,
            "section_title": "y" if in_sections else "n",
        }
        if bad:
            phrase_rows.append({**row, "accepted": "n",
                                "reject_reason":
                                    f"constituent filter: {bad}"})
            continue
        ok = (p["n"] >= 2 and (has_seed or has_non_eng or in_sections)) \
            or (in_sections and (has_seed or has_non_eng)) \
            or (designator and p["n"] >= 2)
        if not ok:
            phrase_rows.append({**row, "accepted": "n",
                                "reject_reason":
                                    "no seed/non-English/section/"
                                    "designator evidence or n<2"})
            continue
        phrase_rows.append({**row, "accepted": "y", "reject_reason": ""})
        accepted[key] = {"kind": "phrase", "sig": {
            "n": p["n"], "n_pages": p["n_pages"], "first_page": p["first_page"],
            "last_page": p["last_page"], "cap_frac": 1.0, "zipf_en": 0.0,
            "surface": p["surface"], "surfaces": p["surfaces"],
        }, "surface": p["surface"], "srcs": set(), "cats": set(),
            "designator": designator, "units": p["units"]}

    # ---- merge n=1 numbered phrases into an accepted base ------------------
    for row in phrase_rows:
        if row["accepted"] != "y" or not row["designator"] or row["n"] >= 2:
            continue
        base_units = row["phrase"].lower().rsplit(" ", 1)[0]
        target = None
        if base_units in accepted and accepted[base_units]["kind"] == "unit":
            target = accepted[base_units]["surface"]
        else:
            for other in phrase_rows:
                if other is row or other["accepted"] != "y" \
                        or not other["designator"]:
                    continue
                if other["phrase"].lower().rsplit(" ", 1)[0] == base_units:
                    target = other["phrase"]
                    break
        if target:
            row["accepted"] = "n"
            row["reject_reason"] = (f"n=1 numbered variant of accepted "
                                    f"`{target}`")

    # ---- organization constituents (single-unit surface variants) ---------
    org_constituent = set()
    for key, e in accepted.items():
        if e["kind"] == "phrase":
            head = key.split()[-1]
            if head in ORG_HEADS:
                first = key.split()[0]
                if first in accepted and accepted[first]["kind"] == "unit":
                    org_constituent.add(first)

    # ---- compound-head evidence -------------------------------------------
    # For each accepted entity, count accepted compound units that contain it
    # as a word-boundary substring (e.g. OMOTYA -> body-OMOTYA).
    accepted_unit_keys = [k for k, e in accepted.items() if e["kind"] == "unit"]
    compound_evidence = {}
    for key in accepted:
        comp = {}
        for uk in accepted_unit_keys:
            if uk == key:
                continue
            if re.search(r"(^|[^a-z])" + re.escape(key) + r"($|[^a-z])",
                         uk):
                comp[uk] = unit_sig[uk]["n"]
        if comp:
            compound_evidence[key] = comp

    # ---- collocation evidence for accepted entities ------------------------
    targets = set(accepted)
    ev, n_units = collocation_evidence(page_units, excluded_uids, targets,
                                       verbs)

    # ---- roles --------------------------------------------------------------
    accepted_keys = set(accepted)
    for key in sorted(accepted):
        e = accepted[key]
        sig = e["sig"]
        numbered_ds = sorted(numbered.get(key, {}).keys())
        role, evidence = assign_role(
            key, e["surface"], e["kind"], sig, numbered_ds, accepted_keys,
            seeds, caps_types, acronyms, ev, compound_evidence.get(key, {}),
            key in org_constituent)
        e["role"] = role
        e["role_evidence"] = evidence
        for c in candidates:
            if c["candidate"] == e["surface"] and c["accepted"] == "y" \
                    and c["kind"] == "unit":
                c["role"] = role

    # ---- glossary ------------------------------------------------------------
    glossary = []
    for key in sorted(accepted):
        e = accepted[key]
        sig = e["sig"]
        evd = ev.get(key, {"n_entity": sig["n"], "rows": []})
        rows = evd["rows"]
        top_partners = [r for r in rows if not r["is_verb"]][:5]
        does = sorted([r for r in rows if r["is_verb"] and r["n_right"] > 0],
                      key=lambda r: (-r["n_right"], r["partner"]))
        done = sorted([r for r in rows if r["is_verb"] and r["n_left"] > 0],
                      key=lambda r: (-r["n_left"], r["partner"]))
        # example context: first occurrence of the entity
        page, start = sig["first_page"], sig.get("first_start", 0)
        body = bodies.get(page, "")
        lo = max(0, start - CONTEXT)
        hi = min(len(body), start + CONTEXT)
        example = cl.normalize_ws(body[lo:hi])
        variants = "|".join(
            f"{s}:{c}" for s, c in
            sorted(sig["surfaces"].items(), key=lambda kv: (-kv[1], kv[0])))
        glossary.append({
            "entity": e["surface"],
            "role": e["role"],
            "n_occurrences": evd["n_entity"] if e["kind"] == "unit"
            else sig["n"],
            "n_pages": sig["n_pages"],
            "first_page": sig["first_page"],
            "last_page": sig["last_page"],
            "surface_variants": variants,
            "sources": "|".join(sorted(e["srcs"])) or "#30",
            "categories": "|".join(sorted(e["cats"])),
            "numeric_designator": e.get("designator", ""),
            "section_title": "y" if key in section_keys else "",
            "top_partners": "|".join(
                f"{r['partner']}({r['n_pair']})" for r in top_partners),
            "does_verbs": "|".join(
                f"{r['partner']}({r['n_right']})" for r in does[:5]),
            "done_to_verbs": "|".join(
                f"{r['partner']}({r['n_left']})" for r in done[:5]),
            "example_context": example,
            "role_evidence": e["role_evidence"],
        })
    role_order = sorted({g["role"] for g in glossary})
    glossary.sort(key=lambda g: (role_order.index(g["role"]),
                                 -g["n_occurrences"], g["entity"].lower()))

    # ---- verb evidence table ------------------------------------------------
    verb_rows = []
    for key in sorted(accepted):
        e = accepted[key]
        for r in ev.get(key, {"rows": []})["rows"]:
            if not r["is_verb"]:
                continue
            direction = "does" if r["n_right"] > 0 else "done_to"
            n = r["n_right"] if direction == "does" else r["n_left"]
            if n == 0:
                continue
            page, s, _ = r["example"] if r["example"] else (0, 0, 0)
            body = bodies.get(page, "")
            lo = max(0, s - CONTEXT)
            hi = min(len(body), s + CONTEXT)
            verb_rows.append({
                "entity": e["surface"], "direction": direction,
                "verb": r["partner"], "n": n, "logdice": r["logdice"],
                "verb_source": r["verb_source"],
                "example_context": cl.normalize_ws(body[lo:hi]),
            })
    verb_rows.sort(key=lambda r: (r["entity"].lower(),
                                  0 if r["direction"] == "does" else 1,
                                  -r["n"], r["verb"]))

    # ---- numbered table -------------------------------------------------------
    num_rows = []
    for key in sorted(numbered):
        for number in sorted(numbered[key]):
            rows = numbered[key][number]
            r = rows[0]
            num_rows.append({
                "entity": r["entity"], "number": number,
                "n": len(rows), "page": r["page"], "start": r["start"],
                "context": r["context"],
            })
    num_rows.sort(key=lambda r: (r["entity"], -r["n"], r["number"]))

    # ---- diagnostics -----------------------------------------------------------
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    role_counts = collections.Counter(g["role"] for g in glossary)
    diag = {
        "schema": SCHEMA,
        "generated": generated,
        "unit_items_total": len(unit_meta),
        "markup_units_removed": len(excluded_uids),
        "unit_items_screened": len(unit_meta) - len(excluded_uids),
        "seed_types": len(seeds),
        "unit_candidate_types": len(unit_sig),
        "phrase_candidate_types": len(phrase_sig),
        "accepted_entities": len(glossary),
        "rejected_candidates": sum(1 for c in candidates if c["accepted"] == "n"),
        "roles": "|".join(f"{r}:{role_counts[r]}" for r in role_order),
    }

    # ---- reconciliation ---------------------------------------------------------
    recon = []
    inv27 = {r["lower"]: r for r in cl.read_tsv(INV27)}
    recon.append({"check": "#27 entity inventory size",
                  "expected": len(inv27), "actual": len(inv27),
                  "status": "ok"})
    d27 = {r["diag"]: r["value"] for r in cl.read_tsv(DIAG27)}
    recon.append({"check": "screened unit items vs #27",
                  "expected": d27.get("unit_items", ""),
                  "actual": diag["unit_items_screened"],
                  "status": "ok" if str(diag["unit_items_screened"])
                  == d27.get("unit_items", "") else "MISMATCH"})
    n_match = n_mismatch = 0
    for key in accepted:
        if key in inv27:
            if int(inv27[key]["n_occurrences"]) == \
                    ev.get(key, {}).get("n_entity", -1):
                n_match += 1
            else:
                n_mismatch += 1
    recon.append({"check": "accepted seed entities n vs #27",
                  "expected": "all match",
                  "actual": f"{n_match} match, {n_mismatch} mismatch",
                  "status": "ok" if n_mismatch == 0 else "MISMATCH"})
    adam = next((g for g in glossary if g["entity"] == "ADAM"), None)
    recon.append({"check": "ADAM unit occurrences vs #27 (303)",
                  "expected": 303,
                  "actual": adam["n_occurrences"] if adam else "",
                  "status": "ok" if adam and adam["n_occurrences"] == 303
                  else "MISMATCH"})
    caps22 = {r["type"].lower(): r for r in cl.read_tsv(CAPS)}
    adam_caps = int(caps22.get("adam", {}).get("n", 0) or 0)
    recon.append({"check": "ADAM ALL-CAPS tokens vs #22 (328)",
                  "expected": 328, "actual": adam_caps,
                  "status": "ok" if adam_caps == 328 else "MISMATCH"})
    recon.append({"check": "#19 markup residue uids screened",
                  "expected": 347, "actual": len(markup_uids),
                  "status": "ok" if len(markup_uids) == 347 else "MISMATCH"})

    # ---- write artifacts --------------------------------------------------------
    cl.write_tsv(out / "entity_glossary.tsv",
                 ["entity", "role", "n_occurrences", "n_pages", "first_page",
                  "last_page", "surface_variants", "sources", "categories",
                  "numeric_designator", "section_title", "top_partners",
                  "does_verbs", "done_to_verbs", "example_context",
                  "role_evidence"], glossary)
    cl.write_tsv(out / "entity_candidates.tsv",
                 ["candidate", "kind", "n", "n_pages", "first_page",
                  "last_page", "cap_frac", "zipf_en", "is_english",
                  "seed_sources", "in_sections", "accepted", "reject_reason",
                  "role"], candidates)
    cl.write_tsv(out / "entity_phrases.tsv",
                 ["phrase", "n", "n_pages", "first_page", "last_page",
                  "surfaces", "designator", "section_title", "accepted",
                  "reject_reason"], phrase_rows)
    cl.write_tsv(out / "entity_verb_evidence.tsv",
                 ["entity", "direction", "verb", "n", "logdice",
                  "verb_source", "example_context"], verb_rows)
    cl.write_tsv(out / "entity_numbered.tsv",
                 ["entity", "number", "n", "page", "start", "context"],
                 num_rows)
    cl.write_tsv(out / "diagnostics.tsv", ["diag", "value"],
                 [{"diag": k, "value": v} for k, v in sorted(diag.items())])
    cl.write_tsv(out / "reconciliation.tsv",
                 ["check", "expected", "actual", "status"], recon)
    summary = {
        "schema": SCHEMA,
        "generated": generated,
        "accepted_entities": len(glossary),
        "roles": dict(role_counts),
        "rejected_candidates": diag["rejected_candidates"],
        "unit_items_screened": diag["unit_items_screened"],
        "seed_types": len(seeds),
        "reconciliation": {r["check"]: r["status"] for r in recon},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    write_report(out / "entity_report.md", glossary, candidates, phrase_rows,
                 num_rows, diag, recon)
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(path, glossary, candidates, phrase_rows, num_rows, diag,
                 recon):
    L = []
    A = L.append
    role_counts = collections.Counter(g["role"] for g in glossary)
    A("# Entity and proper-noun induction (#30) — dramatis personae")
    A("")
    A("Source: #17 unit layer (`units.tsv`), markup-screened with #19's "
      "ledger and #27's tag-run detector; seeds from #19 `proper_noun`, #22 "
      "`label`, #6 gazetteer `entity`; collocation recomputed on the same "
      "screened sequence (±5-unit window, PMI/logDice) as #27; numeric "
      "designators from #21; section titles from #7.  Reproducible via "
      "`python work/entity_induction.py`; regression suite "
      "`python work/test_entity_induction.py`.")
    A("")
    A(f"Schema `{SCHEMA}`.  Signals: **capitalization** (`cap_frac`), "
      f"**recurrence** (`n`, `n_pages`), **collocation** (window partners, "
      f"entity-verb direction).  Recurrence floor `n >= {MIN_N}` for "
      f"non-seed units; English floor `zipf >= {ZIPF_FLOOR}`.")
    A("")
    A("## 1. Headline result")
    A("")
    A(f"- **{len(glossary)} induced entities** "
      f"({', '.join(f'{r} {role_counts[r]}' for r in sorted(role_counts))}).")
    A(f"- **{diag['unit_items_screened']:,} screened unit items** "
      f"({diag['markup_units_removed']} markup removals), "
      f"{diag['seed_types']} curated seed types, "
      f"{diag['rejected_candidates']} rejected candidates "
      f"(disclosed in `entity_candidates.tsv`).")
    A("- The cast is dominated by **ADAM** (303) and its clones "
      "**retro-ADAM** (165) / **ANTI-ADAM** (7), the machine-system "
      "**ToK/ToKAGE** (127+12), the corporation **Sato Corporation** (25), "
      "the places **Cadaver City** and **Placenta World**, and the creature "
      "**Super Cherry 666** (4 numbered + section title).")
    A("")
    A("## 2. Method")
    A("")
    A("1. **Screen** the #17 unit layer exactly as #27 does (347 #19 "
      "markup-residue uids + the general tag-run detector).")
    A("2. **Single-unit induction**: a type is accepted if it is a curated "
      "seed (#19 `proper_noun` / #22 `label` / #6 `entity`), or a "
      "capitalized non-English type with `n >= 3`.  **False-positive "
      "control**: a non-seed type whose surface is an English dictionary "
      "word (`zipf >= 2.5`) is rejected regardless of casing — this "
      "filters pronouns (`I`), common nouns used as names (`Doll`, "
      "`Corporation`, `God`, `Heaven`), and title-case common nouns "
      "(`Project`, `Reverse`).  #22 `function` class types and single "
      "letters are skipped outright.")
    A("3. **Phrase induction**: maximal runs of whitespace-adjacent "
      "capitalized units; every contiguous subsequence of length 2-4 is a "
      "candidate, plus a trailing numeric designator.  Constituents must be "
      "content words (no function words, no operator characters, no single "
      "letters), so `The ADAM`, `An ADAM Doll`, `ADAM Doll/the` and "
      "`Cadaver City I` are rejected.  A phrase is accepted with `n >= 2` "
      "and a seed/non-English/section-title signal, or a numeric "
      "designator with `n >= 2`.")
    A("4. **Numbered variants** with `n = 1` merge into their accepted base "
      "(`NDRO 4` → `NDRO`; `Super Cherry 17/9` → `Super Cherry 666`).")
    A("5. **Collocation evidence**: window partners (±5 units) and the "
      "entity-verb direction split, recomputed on the screened sequence.")
    A("6. **Role assignment**: ordered rules (below), each citing its "
      "evidence.")
    A("")
    A("## 3. Dramatis personae (entity glossary)")
    A("")
    A("One row per induced entity.  `does` = high-confidence verb to the "
      "entity's right; `done-to` = verb to its left.  Full table in "
      "`entity_glossary.tsv`.")
    A("")
    A("| role | entity | n | pages | does | done-to | designator |")
    A("|---|---|---:|---:|---|---|---|")
    for g in glossary:
        A(f"| {g['role']} | **{g['entity']}** | {g['n_occurrences']} | "
          f"{g['n_pages']} | {g['does_verbs'] or '—'} | "
          f"{g['done_to_verbs'] or '—'} | {g['numeric_designator'] or '—'} |")
    A("")
    A("## 4. Worked profiles (top evidence)")
    A("")
    for g in glossary:
        if g["role"] not in ("persona", "antagonist", "creature",
                             "organization", "place"):
            continue
        A(f"**{g['entity']}** ({g['role']}, n={g['n_occurrences']}) — "
          f"{g['role_evidence']}.")
        A(f"  top partners: {g['top_partners'] or '—'}; "
          f"e.g. “{g['example_context']}”")
        A("")
    A("## 5. Numbered entities (#21)")
    A("")
    A("| entity | number | n | example |")
    A("|---|---|---:|---|")
    for r in num_rows:
        A(f"| `{r['entity']}` | {r['number']} | {r['n']} | "
          f"{r['context'][:70]} |")
    A("")
    A("## 6. Rejected candidates (negative results, disclosed)")
    A("")
    A("High-frequency types rejected as English dictionary words (the "
      "false-positive control):")
    A("")
    eng = [c for c in candidates
           if c["accepted"] == "n" and "English dictionary word"
           in c.get("reject_reason", "")]
    eng.sort(key=lambda r: -int(r["n"]))
    for c in eng[:20]:
        A(f"- `{c['candidate']}` (n={c['n']}, zipf {c['zipf_en']}): "
          f"{c['reject_reason']}")
    A("")
    A("Section titles that name no accepted entity (ordinary English or "
      "one-off titles):")
    A("")
    seen = set()
    for c in candidates:
        if c["accepted"] == "n" and c["in_sections"] == "y":
            seen.add(c["candidate"])
    A(", ".join(f"`{s}`" for s in sorted(seen)) or "(none)")
    A("")
    A("Recurring characters that are **not** proper nouns and therefore out "
      "of scope for this induction: `dog` (the book's recurring dog, always "
      "lowercase), `Love Doll` (n=1).  They are disclosed here rather than "
      "forced into the cast.")
    A("")
    A("## 7. Reconciliation")
    A("")
    A("| check | expected | actual | status |")
    A("|---|---|---|---|")
    for r in recon:
        A(f"| {r['check']} | {r['expected']} | {r['actual']} | {r['status']} |")
    A("")
    A("## 8. Reproduce")
    A("")
    A("```bash")
    A("conda activate marker")
    A("python work/entity_induction.py          # writes analysis/entity/")
    A("python work/test_entity_induction.py")
    A("```")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="#30 entity / proper-noun induction (dramatis personae)")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    summary = build(Path(args.out))
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
