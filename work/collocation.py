#!/usr/bin/env python3
"""#27 Collocation and association (PMI / logDice) -> analysis/collocation/

Ranked collocation tables over the book's two tokenizable layers:

  * the **unit** layer (#17 ``units.tsv``) - maximal lexical units, so
    ``gene=TV`` is one item.  Bigrams/trigrams here are *lexical* collocations.
  * the **token** layer (#17 ``tokens.tsv``) - WORD + OP + PUNCT, so the
    operator is visible.  Bigrams/trigrams here are *operator-word* pairs
    (``::`` before what, ``=`` binding which words).

Both layers are screened: the 347 markup-residue occurrences that #19 removed
(HTML tag names such as ``th``/``p``/``href``) are dropped from the unit stream,
and the 349 tokens that live inside those spans are dropped from the token
stream.  Nothing else is repaired.

Association measures (all standard; see ``collocation_report.md``):
PMI, PPMI, logDice (14 + log2(...)), Dice, t-score.  ``N`` is the number of
items in the sequence; PMI's absolute value depends on that choice but its
*ranking* does not.

The entity-verb question ("what does ToKAGE do? what is done to it?") is
answered by ``entity_collocations.tsv`` (every partner of every entity in a
+/-5-unit window, with a direction split) and the focused
``entity_verb_pairs.tsv``.  A partner is tagged as a verb candidate from a
frozen English verb lexicon (Brown corpus surface forms, parsed with stdlib;
the Brown directory is auto-detected and the frozen `verb_lexicon.tsv` is the
fallback when it is absent) plus a transparent book-internal morphological
signal.  The lexicon is a *candidate* filter, not a parser; coinage verbs it
misses are still visible in the unfiltered entity table.  That limit is stated
in the report.

Reproduce::

    conda activate marker
    python work/collocation.py
    python work/test_collocation.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import wordfreq

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "analysis" / "tokenizer" / "units.tsv"
TOKENS = ROOT / "analysis" / "tokenizer" / "tokens.tsv"
MARKUP = ROOT / "analysis" / "neologism" / "markup_residue.tsv"
NEOLOGISMS = ROOT / "analysis" / "neologism" / "neologisms.tsv"
CAPS = ROOT / "analysis" / "case_emphasis" / "caps_types.tsv"
GAZETTEER = ROOT / "analysis" / "lexical_correction" / "gazetteer.tsv"
LAYOUT = ROOT / "analysis" / "layout_recovery" / "blood-electric.layout.md"
OUT = ROOT / "analysis" / "collocation"
VERB_LEXICON = OUT / "verb_lexicon.tsv"

SCHEMA = "collocation/1"
MIN_BIGRAM = 2                  # main bigram table floor
MIN_TRIGRAM = 2                 # main trigram table floor
WINDOW = 5                      # entity co-occurrence window (+/- units)
LOG_DICE_BASE = 14.0            # Sketch Engine logDice offset

# Brown corpus POS tags that are verbs / auxiliaries (Brown tag set).
BROWN_VERB_TAGS = {
    "vb", "vbd", "vbg", "vbn", "vbz",                 # lexical verbs
    "be", "bez", "bed", "bedz", "bem", "ber", "ben", "beg",  # be
    "hv", "hvd", "hvg", "hvn", "hvz",                 # have
    "do", "dod", "doz", "don",                        # do
    "md",                                             # modals
}
DEFAULT_BROWN_DIR = Path.home() / "nltk_data" / "corpora" / "brown"

_PAGE_RE = re.compile(r"<!-- page (\d+) -->")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# English verb-forming suffixes; used only for the book-internal ``-s`` signal.
VERB_FORMING = ("ate", "ite", "ise", "ize", "ify", "fy", "en")

# HTML tag names that can survive the OCR as fake units.  From #19's screen.
HTML_TAGS = {
    "a", "p", "pre", "br", "li", "hr", "th", "td", "tr", "table", "tbody",
    "thead", "body", "html", "head", "span", "div", "img", "meta", "link",
    "b", "i", "u", "em", "strong", "font", "small", "big", "center",
    "script", "style", "title", "form", "input", "button", "ol", "ul", "dl",
    "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6", "nav", "main", "footer",
    "header", "section", "article", "blockquote", "code", "q", "sup", "sub",
}
# Tag names that are book-furniture even without an attribute or a closing
# slash.  A bare English word such as ``body`` is only markup when it is a
# closing tag or carries an attribute (below), so book text is not eaten.
STRICT_TAGS = {"a", "th", "p", "pre", "li", "br", "hr", "td", "tr", "table",
               "img", "meta", "link", "input", "script", "style"}
_TAG_RUN_RE = re.compile(
    r"</?([a-zA-Z][a-zA-Z0-9]*)"
    r"((?:\s+[a-zA-Z-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))*)\s*/?>")


def find_markup_spans(body: str):
    """(start, end) spans of real HTML tag runs in a page body.

    The book writes its own ``<...>`` blocks, so a bare ``<word>`` is not
    enough: a run counts only when the tag name is a known HTML tag *and* it
    is a closing tag, carries an attribute, or is in STRICT_TAGS.  This catches
    attribute names (#19's screen missed ``colspan``, ``href``, ``mailto``).
    """
    spans = []
    for m in _TAG_RUN_RE.finditer(body):
        name = m.group(1).lower()
        if name not in HTML_TAGS:
            continue
        closing = body[m.start():m.start() + 2] == "</"
        if closing or m.group(2).strip() or name in STRICT_TAGS:
            spans.append((m.start(), m.end()))
    return spans



# ---------------------------------------------------------------------------
# io helpers (quote-unsafe TSVs: split on tab, never a csv reader -- #17)
# ---------------------------------------------------------------------------
def read_tsv(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t")))
                for line in fh if line.strip()]


def cell(value) -> str:
    s = "" if value is None else str(value)
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(cell(r.get(h, "")) for h in header) + "\n")


def load_bodies(src: Path = LAYOUT):
    text = src.read_text(encoding="utf-8", errors="ignore")
    parts = _PAGE_RE.split(text)
    return {int(parts[k]): parts[k + 1] for k in range(1, len(parts) - 1, 2)}


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", _COMMENT_RE.sub(" ", s)).strip()


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_units():
    """Return (page_units, unit_meta).

    page_units: {page: [(start, uid, lower, surface, cls), ...] sorted}
    unit_meta:  {uid: (page, start, end, surface)}
    """
    page_units = collections.defaultdict(list)
    unit_meta = {}
    for r in read_tsv(UNITS):
        uid = int(r["uid"])
        page = int(r["page"])
        s, e = int(r["start"]), int(r["end"])
        surface = r["unit"]
        page_units[page].append((s, uid, surface.lower(), surface, r["class"]))
        unit_meta[uid] = (page, s, e, surface)
    for page in page_units:
        page_units[page].sort()
    return page_units, unit_meta


def load_tokens():
    """{page: [(start, end, text, kind), ...] sorted}."""
    page_tokens = collections.defaultdict(list)
    for r in read_tsv(TOKENS):
        page_tokens[int(r["page"])].append(
            (int(r["start"]), int(r["end"]), r["text"], r["kind"]))
    for page in page_tokens:
        page_tokens[page].sort()
    return page_tokens


def load_markup(bodies):
    """Markup exclusion: (spans, uids, types).

    spans  - {page: [(start, end), ...]} from the general tag-run detector
             unioned with #19's markup-residue ledger;
    uids   - #19 markup-residue unit uids (kept explicit for reconciliation);
    types  - #19 ``junk`` unit types (``href``, ``mailto``), which are tag
             furniture but are not in the residue ledger.
    """
    spans = collections.defaultdict(list)
    uids = set()
    for r in read_tsv(MARKUP):
        uids.add(int(r["uid"]))
        spans[int(r["page"])].append((int(r["start"]), int(r["end"])))
    for page, body in bodies.items():
        spans[page].extend(find_markup_spans(body))
    types = set()
    screening = ROOT / "analysis" / "neologism" / "screening.tsv"
    if screening.exists():
        for r in read_tsv(screening):
            if r["category"] in ("junk", "markup_residue"):
                types.add(r["unit"].lower())
    return spans, uids, types


def unit_is_markup(surface, uid, start, page, spans, markup_uids, markup_types):
    if uid in markup_uids or surface.lower() in markup_types:
        return True
    return any(a <= start < b for a, b in spans.get(page, ()))


def load_entities():
    """Entity inventory: #19 proper_noun, #22 label, #6 gazetteer entity.

    Same definition as #26's concordancer so the two joins agree.
    Returns {lower: {"categories": set, "sources": set}}.
    """
    ents = {}

    def add(surface, category, source):
        key = surface.lower()
        e = ents.setdefault(key, {"categories": set(), "sources": set(),
                                  "surface": surface})
        e["categories"].add(category)
        e["sources"].add(source)

    for r in read_tsv(NEOLOGISMS):
        if r["category"] == "proper_noun":
            add(r["unit"], "proper_noun", "#19")
    for r in read_tsv(CAPS):
        if r["class"] == "label":
            add(r["type"], f"label:{r['subtype']}", "#22")
    for r in read_tsv(GAZETTEER):
        if r["category"] == "entity":
            add(r["token"], "entity", "#6")
    return ents


# ---------------------------------------------------------------------------
# verb lexicon
# ---------------------------------------------------------------------------
def build_verb_lexicon(brown_dir: Path = DEFAULT_BROWN_DIR):
    """Brown corpus verb surface forms with counts, parsed with stdlib only.

    The Brown files are ``word/tag`` tokens; a token is a verb if its tag is in
    BROWN_VERB_TAGS.  No nltk import is needed to read them.
    """
    counts = collections.Counter()
    vocab = collections.Counter()
    files = sorted(glob.glob(str(Path(brown_dir) / "c*")))
    for fn in files:
        try:
            text = Path(fn).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for tok in text.split():
            if "/" not in tok:
                continue
            word, tag = tok.rsplit("/", 1)
            w = word.lower()
            vocab[w] += 1
            if tag in BROWN_VERB_TAGS:
                counts[w] += 1
    return counts, vocab, len(files)


def load_verb_lexicon():
    """{lower: {"n": int, "source": str}} from the frozen artifact."""
    lex = {}
    for r in read_tsv(VERB_LEXICON):
        lex[r["verb"].lower()] = {"n": int(r["n"]), "source": r["source"]}
    return lex


def add_book_verb_candidates(lex, gaz, book_types):
    """Augment the English verb lexicon with book-internal candidates.

    Transparent, and disclosed as a candidate signal rather than a parse.
    Restricted to the book's own unit types, so the lexicon stays bounded and
    a word-frequency list is never the source of a verb claim:

      * ``book``  - a #6 gazetteer neologism/entity with a verb suffix
        (catches coinages such as ``fecundates``, ``defleshed``);
      * ``morph`` - ``-ed`` / ``-ing`` whose stem is an English word (zipf >=
        1.5); these suffixes are verb-diagnostic;
      * ``morph_s`` - ``-s`` / ``-es`` whose stem is a common English word
        (zipf >= 3.0); ambiguous with plural nouns, flagged separately.
    """
    for low in book_types:
        if low in lex or len(low) < 4 or not low.isalpha():
            continue
        if low.endswith("ed"):
            stem, suffix = low[:-2], "ed"
        elif low.endswith("ing"):
            stem, suffix = low[:-3], "ing"
        elif low.endswith("s"):
            stem, suffix = low[:-1], "s"
        else:
            continue
        if len(stem) < 3:
            continue
        if gaz.get(low) in ("neologism", "entity"):
            # A book coinage with a verb suffix.  ``-ed``/``-ing`` are
            # verb-diagnostic; ``-s`` is only a verb when the stem ends in a
            # verb-forming suffix (``fecundate`` -> ``fecundates``), so noun
            # coinages such as ``apoptosis``/``technocrisis`` are not swept in.
            # Both the strip-``s`` and strip-``es`` candidates are tried.
            if suffix in ("ed", "ing"):
                lex[low] = {"n": 0, "source": "book"}
            else:
                cands = {stem}
                if low.endswith("es"):
                    cands.add(low[:-2])
                if any(c.endswith(VERB_FORMING) for c in cands):
                    lex[low] = {"n": 0, "source": "book"}
        elif suffix in ("ed", "ing") and wordfreq.zipf_frequency(stem, "en") >= 1.5:
            lex[low] = {"n": 0, "source": "morph"}
        elif suffix == "s" and wordfreq.zipf_frequency(stem, "en") >= 3.0:
            lex[low] = {"n": 0, "source": "morph_s"}
    return lex


# ---------------------------------------------------------------------------
# association measures
# ---------------------------------------------------------------------------
def association(f_xy, f_x, f_y, n):
    """PMI / PPMI / logDice / Dice / t-score for a pair."""
    if f_xy <= 0 or f_x <= 0 or f_y <= 0 or n <= 0:
        pmi = 0.0
    else:
        pmi = math.log2((f_xy * n) / (f_x * f_y))
    ppmi = max(pmi, 0.0)
    denom = f_x + f_y
    logdice = LOG_DICE_BASE + math.log2(2 * f_xy / denom) if denom else 0.0
    dice = 2 * f_xy / denom if denom else 0.0
    expected = f_x * f_y / n if n else 0.0
    tscore = (f_xy - expected) / math.sqrt(f_xy) if f_xy > 0 else 0.0
    return {"pmi": round(pmi, 4), "ppmi": round(ppmi, 4),
            "logdice": round(logdice, 4), "dice": round(dice, 5),
            "tscore": round(tscore, 4)}


def verb_confidence(source: str) -> str:
    """High for Brown surface forms and verb-diagnostic morphology; low for
    the ``-s``/``-es`` signal, which is ambiguous with plural nouns."""
    if source in ("brown", "morph", "book"):
        return "high"
    if source == "morph_s":
        return "low"
    return ""


def trigram_association(f_xyz, f_x, f_y, f_z, n):
    """3-way PMI and a logDice extension for a trigram."""
    if f_xyz <= 0 or f_x <= 0 or f_y <= 0 or f_z <= 0 or n <= 0:
        pmi3 = 0.0
    else:
        pmi3 = math.log2((f_xyz * n * n) / (f_x * f_y * f_z))
    denom = f_x + f_y + f_z
    logdice3 = LOG_DICE_BASE + math.log2(3 * f_xyz / denom) if denom else 0.0
    return {"pmi3": round(pmi3, 4), "logdice3": round(logdice3, 4)}


# ---------------------------------------------------------------------------
# n-gram tables
# ---------------------------------------------------------------------------
def count_ngrams(seqs, n):
    c = collections.Counter()
    for seq in seqs:
        for i in range(len(seq) - n + 1):
            c[tuple(seq[i:i + n])] += 1
    return c


def tag_item(x, entities, verbs):
    """Feature columns for a sequence item (case-folded)."""
    return {
        "entity": "|".join(sorted(entities[x]["categories"]))
        if x in entities else "",
        "verb": verbs[x]["source"] if x in verbs else "",
    }


def build_bigram_rows(seqs, n_items, entities, verbs, min_count):
    uni = collections.Counter()
    for seq in seqs:
        uni.update(seq)
    bi = count_ngrams(seqs, 2)
    rows = []
    for (x, y), f_xy in bi.items():
        if f_xy < min_count:
            continue
        f_x, f_y = uni[x], uni[y]
        a = association(f_xy, f_x, f_y, n_items)
        tx, ty = tag_item(x, entities, verbs), tag_item(y, entities, verbs)
        rows.append({
            "x": x, "y": y, "f_x": f_x, "f_y": f_y, "f_xy": f_xy,
            "n": n_items, **a,
            "x_entity": tx["entity"], "y_entity": ty["entity"],
            "x_verb": tx["verb"], "y_verb": ty["verb"],
        })
    rows.sort(key=lambda r: (-r["logdice"], -r["f_xy"], r["x"], r["y"]))
    return rows, bi, uni


def build_trigram_rows(seqs, n_items, entities, verbs, min_count):
    uni = collections.Counter()
    for seq in seqs:
        uni.update(seq)
    tri = count_ngrams(seqs, 3)
    bi = count_ngrams(seqs, 2)
    rows = []
    for (x, y, z), f_xyz in tri.items():
        if f_xyz < min_count:
            continue
        a = trigram_association(f_xyz, uni[x], uni[y], uni[z], n_items)
        rows.append({
            "x": x, "y": y, "z": z,
            "f_x": uni[x], "f_y": uni[y], "f_z": uni[z], "f_xyz": f_xyz,
            "f_xy": bi[(x, y)], "f_yz": bi[(y, z)],
            "n": n_items, **a,
            "x_entity": entities.get(x, {}).get("surface", ""),
            "y_entity": entities.get(y, {}).get("surface", ""),
            "z_entity": entities.get(z, {}).get("surface", ""),
            "x_verb": verbs.get(x, {}).get("source", ""),
            "y_verb": verbs.get(y, {}).get("source", ""),
            "z_verb": verbs.get(z, {}).get("source", ""),
        })
    rows.sort(key=lambda r: (-r["logdice3"], -r["f_xyz"], r["x"], r["y"], r["z"]))
    return rows, tri


def build_token_bigram_rows(page_tokens, markup_spans, entities, verbs,
                            min_count):
    """Token bigrams with WORD/OP/PUNCT kind columns (operator-word pairs)."""
    seqs = []
    for page, toks in page_tokens.items():
        spans = markup_spans.get(page, [])
        seq = []
        for s, e, text, kind in toks:
            if any(a <= s < b for a, b in spans):
                continue
            seq.append((text.lower(), kind))
        seqs.append(seq)
    n_items = sum(len(s) for s in seqs)
    uni = collections.Counter()
    for seq in seqs:
        uni.update(seq)
    bi = collections.Counter()
    for seq in seqs:
        for i in range(len(seq) - 1):
            bi[(seq[i], seq[i + 1])] += 1
    rows = []
    for (x, y), f_xy in bi.items():
        if f_xy < min_count:
            continue
        f_x, f_y = uni[x], uni[y]
        a = association(f_xy, f_x, f_y, n_items)
        rows.append({
            "x": x[0], "y": y[0], "x_kind": x[1], "y_kind": y[1],
            "f_x": f_x, "f_y": f_y, "f_xy": f_xy, "n": n_items, **a,
            "x_entity": entities.get(x[0], {}).get("surface", ""),
            "y_entity": entities.get(y[0], {}).get("surface", ""),
            "x_verb": verbs.get(x[0], {}).get("source", ""),
            "y_verb": verbs.get(y[0], {}).get("source", ""),
            "has_op": "y" if (x[1] == "OP" or y[1] == "OP") else "n",
        })
    rows.sort(key=lambda r: (-r["logdice"], -r["f_xy"], r["x"], r["y"]))
    return rows, bi, uni, n_items


def build_token_trigram_rows(page_tokens, markup_spans, min_count):
    seqs = []
    for page, toks in page_tokens.items():
        spans = markup_spans.get(page, [])
        seq = [(text.lower(), kind) for s, e, text, kind in toks
               if not any(a <= s < b for a, b in spans)]
        seqs.append(seq)
    n_items = sum(len(s) for s in seqs)
    uni = collections.Counter()
    for seq in seqs:
        uni.update(seq)
    tri = collections.Counter()
    for seq in seqs:
        for i in range(len(seq) - 2):
            tri[(seq[i], seq[i + 1], seq[i + 2])] += 1
    rows = []
    for (x, y, z), f_xyz in tri.items():
        if f_xyz < min_count:
            continue
        a = trigram_association(f_xyz, uni[x], uni[y], uni[z], n_items)
        kinds = (x[1], y[1], z[1])
        rows.append({
            "x": x[0], "y": y[0], "z": z[0],
            "x_kind": x[1], "y_kind": y[1], "z_kind": z[1],
            "f_xyz": f_xyz, "n": n_items, **a,
            "has_op": "y" if "OP" in kinds else "n",
        })
    rows.sort(key=lambda r: (-r["logdice3"], -r["f_xyz"], r["x"], r["y"], r["z"]))
    return rows, tri


# ---------------------------------------------------------------------------
# entity collocations
# ---------------------------------------------------------------------------
def build_entity_tables(page_units, excluded_uids, entities, verbs, bodies,
                        window=WINDOW):
    """Per-entity window partners + focused entity-verb pairs."""
    seqs = []
    for page in sorted(page_units):
        seq = [u for u in page_units[page] if u[1] not in excluded_uids]
        seqs.append((page, seq))

    n_units = sum(len(s) for _, s in seqs)
    partner_freq = collections.Counter()
    entity_freq = collections.Counter()
    for _, seq in seqs:
        partner_freq.update(u[2] for u in seq)
        entity_freq.update(u[2] for u in seq if u[2] in entities)

    pairs = collections.Counter()          # (e, p) -> window co-occurrences
    left = collections.Counter()
    right = collections.Counter()
    adj_left = collections.Counter()
    adj_right = collections.Counter()
    min_dist = {}
    example = {}

    for page, seq in seqs:
        for i, u in enumerate(seq):
            e = u[2]
            if e not in entities:
                continue
            lo = max(0, i - window)
            hi = min(len(seq), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                p = seq[j][2]
                key = (e, p)
                pairs[key] += 1
                if j < i:
                    left[key] += 1
                else:
                    right[key] += 1
                if j == i - 1:
                    adj_left[key] += 1
                if j == i + 1:
                    adj_right[key] += 1
                d = abs(j - i)
                if key not in min_dist or d < min_dist[key]:
                    min_dist[key] = d
                if key not in example:
                    s, e2 = u[0], seq[j][0]
                    lo_c = max(0, min(s, e2) - 45)
                    hi_c = max(s, e2) + 45
                    example[key] = normalize_ws(
                        bodies.get(page, "")[lo_c:hi_c])

    coll_rows = []
    verb_rows = []
    for (e, p), n_pair in pairs.items():
        f_e, f_p = entity_freq[e], partner_freq[p]
        a = association(n_pair, f_e, f_p, n_units)
        cats = "|".join(sorted(entities[e]["categories"]))
        srcs = "|".join(sorted(entities[e]["sources"]))
        base = {
            "entity": entities[e]["surface"], "entity_lower": e,
            "entity_categories": cats, "entity_sources": srcs,
            "partner": p, "n_pair": n_pair,
            "n_entity": f_e, "n_partner": f_p,
            "n_left": left[(e, p)], "n_right": right[(e, p)],
            "n_adjacent_left": adj_left[(e, p)],
            "n_adjacent_right": adj_right[(e, p)],
            "min_distance": min_dist.get((e, p), ""),
            "partner_is_verb": "y" if p in verbs else "n",
            "verb_source": verbs.get(p, {}).get("source", ""),
            "verb_confidence": verb_confidence(verbs.get(p, {}).get("source", "")),
            "partner_is_entity": "y" if p in entities else "n",
            "example_context": example.get((e, p), ""),
            **a,
        }
        coll_rows.append(base)
        if p in verbs:
            verb_rows.append(base)

    coll_rows.sort(key=lambda r: (r["entity_lower"], -r["logdice"], r["partner"]))
    verb_rows.sort(key=lambda r: (r["entity_lower"], -r["logdice"], r["partner"]))
    return coll_rows, verb_rows, n_units, entity_freq, partner_freq


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def top_table(rows, cols, headers, n=20, key=None):
    key = key or (lambda r: (-r["logdice"], -r["f_xy"]))
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in sorted(rows, key=key)[:n]:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return out


def write_report(path, big, tri, tok_bi, tok_tri, coll, verb_pairs,
                 entities, verbs, diag):
    L = []
    A = L.append
    A("# Collocation and association (#27) — PMI / logDice")
    A("")
    A("Source: #17 unit layer (`units.tsv`) and token layer (`tokens.tsv`), "
      "markup-screened with #19's ledger; entity list from #19/#22/#6. "
      "Reproducible via `python work/collocation.py`; regression suite "
      "`python work/test_collocation.py`.")
    A("")
    A(f"Schema `{SCHEMA}`.  Association: PMI, PPMI, logDice "
      f"(`14 + log2(2f/(fx+fy))`), Dice, t-score.  Window for entity "
      f"co-occurrence: ±{WINDOW} units.  Main bigram floor `f>= {MIN_BIGRAM}`, "
      f"trigram floor `f>= {MIN_TRIGRAM}`.")
    A("")
    A("## 1. Headline result")
    A("")
    A(f"- **Unit layer** (lexical): {diag['screened_items']:,} items after "
      f"#19's {diag['markup_n19_residue']} markup-residue removals, then "
      f"{diag['markup_extra_removed']} more tag-furniture occurrences "
      f"(`href`, `mailto`, `colspan`) the general tag-run detector catches -> "
      f"{diag['unit_items']:,} collocation items; "
      f"{diag['unit_bigram_types_all']:,} bigram types / "
      f"{diag['unit_trigram_types_all']:,} trigram types.")
    A(f"- **Token layer** (operator-aware): {diag['token_items']:,} tokens "
      f"after removing {diag['markup_tokens_removed']} markup tokens; "
      f"{diag['token_bigram_types_all']:,} bigram types / "
      f"{diag['token_trigram_types_all']:,} trigram types.")
    A(f"- **Entities**: {diag['entities']} types from #19 `proper_noun`, "
      f"#22 `label`, #6 gazetteer `entity`; "
      f"{diag['entity_occurrences']:,} occurrences, "
      f"{diag['entity_partner_pairs']:,} (entity, partner) window pairs.")
    A(f"- **Verb candidates**: {diag['verbs']:,} forms "
      f"({diag['verbs_brown']:,} Brown, {diag['verbs_book']:,} verb-diagnostic "
      f"book-internal, {diag['verbs_morph_s']:,} ambiguous `-s`/`-es`).")
    A("")
    A("The tables answer two questions the issue names directly:")
    A("")
    A("- **operator-word**: which lexical material an operator selects for — "
      "the token-layer table carries an `x_kind`/`y_kind` column "
      "(`WORD`/`OP`/`PUNCT`) and a `has_op` flag.")
    A("- **entity-verb**: what an entity does (verb to its right) and what is "
      "done to it (verb to its left) — `entity_collocations.tsv` and "
      "`entity_verb_pairs.tsv`.")
    A("")
    A("## 2. Top unit bigrams by logDice")
    A("")
    A("Adjacent maximal units, case-folded.  logDice rewards pairs that are "
      "both frequent and tightly bound; PMI alone would surface hapax pairs.  "
      "Restricted here to `f_xy >= 5` so the `Dice = 1.0` two-occurrence "
      "saturation does not dominate (the full `f_xy >= 2` table is the "
      "artifact).")
    A("")
    L.extend(top_table([r for r in big if r["f_xy"] >= 5],
                       ["x", "y", "f_xy", "f_x", "f_y", "pmi", "logdice",
                        "dice", "tscore", "x_entity", "y_verb"],
                       ["x", "y", "f_xy", "f_x", "f_y", "PMI", "logDice",
                        "Dice", "t", "x_ent", "y_verb"], n=25))
    A("")
    A("## 3. Top unit bigrams by raw frequency (the loop)")
    A("")
    L.extend(top_table(big, ["x", "y", "f_xy", "logdice", "pmi"],
                       ["x", "y", "f_xy", "logDice", "PMI"], n=25,
                       key=lambda r: (-r["f_xy"], r["x"], r["y"])))
    A("")
    A("## 4. Top unit trigrams")
    A("")
    L.extend(top_table(tri, ["x", "y", "z", "f_xyz", "pmi3", "logdice3"],
                       ["x", "y", "z", "f_xyz", "PMI3", "logDice3"], n=20,
                       key=lambda r: (-r["f_xyz"], r["x"], r["y"], r["z"])))
    A("")
    A("## 5. Operator-word bigrams (token layer)")
    A("")
    A(f"{diag['token_bigram_types_all']:,} token bigram types; "
      f"{diag['token_bigrams_with_op']:,} contain at least one operator.  "
      f"Below: the most frequent operator-containing bigrams.")
    A("")
    op_rows = [r for r in tok_bi if r["has_op"] == "y"]
    L.extend(top_table(op_rows, ["x", "y", "x_kind", "y_kind", "f_xy",
                                 "f_x", "f_y", "pmi", "logdice"],
                       ["x", "y", "x_kind", "y_kind", "f_xy", "f_x", "f_y",
                        "PMI", "logDice"], n=25,
                       key=lambda r: (-r["f_xy"], r["x"], r["y"])))
    A("")
    A("### 5.1 Which words a seam binds")
    A("")
    A("The seam operators (`=`, `/`, `++`, `+`) sit between two words inside a "
      "unit; the token layer exposes each direction.  Top right-partners of "
      "each seam (word after the operator) by frequency:")
    A("")
    A("| seam | top right partners (f) |")
    A("|---|---|")
    for op in ("=", "/", "++", "+"):
        rs = [r for r in tok_bi if r["x"] == op and r["x_kind"] == "OP"
              and r["y_kind"] == "WORD"]
        rs.sort(key=lambda r: -r["f_xy"])
        if rs:
            A(f"| `{op}` | " + ", ".join(
                f"`{r['y']}` ({r['f_xy']})" for r in rs[:8]) + " |")
    A("")
    A("### 5.2 Segment operators before lexical material")
    A("")
    A("| operator | top following WORDs (f) |")
    A("|---|---|")
    for op in ("::", "//", ">", "<", ">>", "<<"):
        rs = [r for r in tok_bi if r["x"] == op and r["x_kind"] == "OP"
              and r["y_kind"] == "WORD"]
        rs.sort(key=lambda r: -r["f_xy"])
        if rs:
            A(f"| `{op}` | " + ", ".join(
                f"`{r['y']}` ({r['f_xy']})" for r in rs[:8]) + " |")
    A("")
    A("### 5.3 Operator trigrams")
    A("")
    A(f"{diag['token_trigram_types_all']:,} token trigram types; the most "
      f"frequent that contain an operator:")
    A("")
    op_tri = [r for r in tok_tri if r["has_op"] == "y"]
    L.extend(top_table(op_tri, ["x", "y", "z", "x_kind", "y_kind", "z_kind",
                                "f_xyz", "pmi3", "logdice3"],
                       ["x", "y", "z", "kx", "ky", "kz", "f_xyz", "PMI3",
                        "logDice3"], n=15,
                       key=lambda r: (-r["f_xyz"], r["x"], r["y"], r["z"])))
    A("")
    A("## 6. Entity-verb pairs")
    A("")
    A(f"For each entity, partners within ±{WINDOW} units are scored; a partner "
      "is a verb candidate from the frozen lexicon.  `n_right` is the count of "
      "windows where the verb follows the entity (**entity does verb**); "
      "`n_left` where it precedes (**verb is done to entity / verb-entity**).")
    A("")
    A("| entity | verb | source | conf | n_pair | left | right | adj_R | "
      "adj_L | logDice | PMI |")
    A("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    ranked = [r for r in verb_pairs if r["n_pair"] >= 2]
    ranked.sort(key=lambda r: (r["verb_confidence"] != "high",
                               -r["logdice"], -r["n_pair"]))
    for r in ranked[:35]:
        A(f"| `{r['entity']}` | `{r['partner']}` | {r['verb_source']} | "
          f"{r['verb_confidence']} | {r['n_pair']} | {r['n_left']} | "
          f"{r['n_right']} | {r['n_adjacent_right']} | {r['n_adjacent_left']} | "
          f"{r['logdice']} | {r['pmi']} |")
    A("")
    A("`conf = low` is the `-s`/`-es` morphological signal, which is ambiguous "
      "with plural nouns (`SATO victims`); `conf = high` is a Brown surface "
      "form or verb-diagnostic morphology.  Rows with `n_pair = 1` are in the "
      "artifact, not this table.")
    A("")
    A("### 6.1 Worked profiles")
    A("")
    A("`does` = a high-confidence verb candidate to the entity's right "
      "(entity → verb); `done to` = a high-confidence verb to its left "
      "(verb → entity).")
    A("")
    for ent in ("ToKAGE", "ADAM", "cadaver", "NDRO", "BABEL", "SATO"):
        key = ent.lower()
        if key not in entities:
            continue
        vp = [r for r in verb_pairs if r["entity_lower"] == key
              and r["verb_confidence"] == "high"]
        does = sorted(vp, key=lambda r: (-r["n_right"], -r["logdice"]))
        done = sorted(vp, key=lambda r: (-r["n_left"], -r["logdice"]))
        ap = [r for r in coll if r["entity_lower"] == key]
        ap.sort(key=lambda r: -r["n_pair"])
        A(f"**`{entities[key]['surface']}`** "
          f"({diag['entity_freq'].get(key, 0)} occurrences)")
        A("")
        A("- does (verb to the right): " + (", ".join(
            f"`{r['partner']}` ×{r['n_right']}" for r in does[:6]
            if r["n_right"] > 0) or "—"))
        A("- done to (verb to the left): " + (", ".join(
            f"`{r['partner']}` ×{r['n_left']}" for r in done[:6]
            if r["n_left"] > 0) or "—"))
        A("- top window partners (any POS): " + ", ".join(
            f"`{r['partner']}` ({r['n_pair']})" for r in ap[:10]))
        A("")
    A("## 7. Verb lexicon (candidate filter, not a parser)")
    A("")
    A(f"{diag['verbs']:,} forms: {diag['verbs_brown']:,} English surface forms "
      f"read from the Brown corpus POS tags "
      f"(`{diag['brown_files']}` files, stdlib parse, frozen in "
      f"`verb_lexicon.tsv`), plus {diag['verbs_book']:,} verb-diagnostic "
      f"book-internal candidates (`morph`/`book`) and "
      f"{diag['verbs_morph_s']:,} ambiguous `-s`/`-es` candidates "
      f"(`morph_s`, flagged `conf=low`).")
    A("")
    A("**Limit, stated plainly:** a Brown-surface lexicon cannot contain a "
      "coinage verb.  Forms such as `fecundates` and `defleshed` are only "
      "caught by the `book` signal (a #6 gazetteer neologism with a verb "
      "suffix); genuinely novel verbs without a gazetteer entry are *not* "
      "flagged.  They are still visible in the unfiltered "
      "`entity_collocations.tsv`, which lists every partner.  This is a "
      "candidate filter, not a parse.")
    A("")
    A("## 8. Method and limits")
    A("")
    A("- **Two layers, never token indices.** The unit stream is the lexical "
      "view (`gene=TV` is one item); the token stream is the operator view. "
      "Both are built from `(page, start)` order, so a page boundary can "
      "never join two items.")
    A("- **PMI absolute value is normalization-dependent.** `N` is the item "
      "count of the sequence; only rankings are interpreted.  logDice and "
      "Dice are count-based and normalization-free.")
    A("- **PMI favours rare pairs.** The main table is sorted by logDice; the "
      "raw-frequency table is given beside it.  `f_xy` is printed on every "
      "row so the reader can filter.")
    A("- **Markup residue removed, not repaired.** #19's 347 occurrences "
      "(tag names `th`, `p`, `pre`, ...) are dropped from both streams, plus "
      f"{diag['markup_extra_removed']} tag-furniture occurrences #19 left in "
      "(`href`, `mailto`, `colspan`), found by a general HTML tag-run "
      f"detector; nothing else is.  {diag['markup_tokens_removed']} tokens "
      "inside those spans are dropped from the token stream.")
    A("- **Window association is approximate.** The entity window counts "
      f"(entity, partner) pairs within ±{WINDOW} units and scores them with "
      "the same formulas at `N = units`; it is a co-occurrence screen, not a "
      "dependency relation.")
    A("- **Verb detection is a candidate filter**, disclosed in §7; it is not "
      "a POS tagger and no parse-attempt is made here (#27 is distributional).")
    A("")
    A("## 9. Artifacts")
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `collocations_bigram.tsv` | THE DELIVERABLE: unit bigrams, ranked |")
    A("| `collocations_trigram.tsv` | unit trigrams, ranked |")
    A("| `collocations_bigram_token.tsv` | token bigrams incl. operators |")
    A("| `collocations_trigram_token.tsv` | token trigrams incl. operators |")
    A("| `entity_collocations.tsv` | every entity × window partner |")
    A("| `entity_verb_pairs.tsv` | entity × verb-candidate partners |")
    A("| `verb_lexicon.tsv` | frozen verb-candidate lexicon + provenance |")
    A("| `entity_inventory.tsv` | the entity list and its sources |")
    A("| `summary.json` | machine-readable headline numbers |")
    A("| `diagnostics.tsv` | build ledger |")
    A("| `reconciliation.tsv` | this build vs #17/#19 |")
    A("")
    A("All `.tsv` are read with `split(\"\\t\")`, never a csv reader.")
    A("")
    A("---")
    A(f"*Generated by `work/collocation.py` (schema `{SCHEMA}`).*")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="#27 collocation / association")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--min-bigram", type=int, default=MIN_BIGRAM)
    ap.add_argument("--min-trigram", type=int, default=MIN_TRIGRAM)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--brown-dir", default=str(DEFAULT_BROWN_DIR))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    page_units, unit_meta = load_units()
    page_tokens = load_tokens()
    bodies = load_bodies()
    markup_spans, markup_uids, markup_types = load_markup(bodies)
    entities = load_entities()
    gaz = {r["token"].lower(): r["category"] for r in read_tsv(GAZETTEER)}

    # excluded unit occurrences: #19 ledger + junk types + the tag-run detector
    excluded_uids = set()
    for page, units in page_units.items():
        for s, uid, lower, surface, cls in units:
            if unit_is_markup(surface, uid, s, page, markup_spans,
                              markup_uids, markup_types):
                excluded_uids.add(uid)

    # ---- verb lexicon ----
    lexicon_path = out / "verb_lexicon.tsv"
    brown_dir = Path(args.brown_dir)
    if brown_dir.is_dir():
        # 0.1 s for the whole Brown corpus; rebuild so the current candidate
        # rules always apply rather than trusting a stale frozen artifact.
        counts, _brown_vocab, n_files = build_verb_lexicon(brown_dir)
        verbs = {w: {"n": c, "source": "brown"} for w, c in counts.items()}
    else:
        verbs = load_verb_lexicon()
        n_files = 0
    # ---- unit sequences (needed to bound the book-internal verb signal) ----
    unit_seqs = []
    for page in sorted(page_units):
        seq = [u[2] for u in page_units[page] if u[1] not in excluded_uids]
        unit_seqs.append(seq)
    unit_items = sum(len(s) for s in unit_seqs)
    book_types = set()
    for seq in unit_seqs:
        book_types.update(seq)
    verbs = add_book_verb_candidates(verbs, gaz, book_types)
    # persist the augmented view for inspection (frozen artifact + book rows)
    write_tsv(lexicon_path, ["verb", "n", "source"],
              [{"verb": w, "n": v["n"], "source": v["source"]}
               for w, v in sorted(verbs.items())])

    big, bi_all, _uni_all = build_bigram_rows(
        unit_seqs, unit_items, entities, verbs, args.min_bigram)
    tri, tri_all = build_trigram_rows(
        unit_seqs, unit_items, entities, verbs, args.min_trigram)

    # ---- token sequences ----
    tok_bi, tok_bi_all, _tok_uni, token_items = build_token_bigram_rows(
        page_tokens, markup_spans, entities, verbs, args.min_bigram)
    tok_tri, tok_tri_all = build_token_trigram_rows(
        page_tokens, markup_spans, args.min_trigram)

    # ---- entity tables ----
    coll, verb_pairs, _n_units, entity_freq, _partner_freq = build_entity_tables(
        page_units, excluded_uids, entities, verbs, bodies, args.window)

    # ---- diagnostics ----
    n_markup_tokens = 0
    for page, toks in page_tokens.items():
        spans = markup_spans.get(page, [])
        n_markup_tokens += sum(
            1 for s, e, t, k in toks if any(a <= s < b for a, b in spans))
    verbs_brown = sum(1 for v in verbs.values() if v["source"] == "brown")
    verbs_book = sum(1 for v in verbs.values()
                     if v["source"] in ("book", "morph"))
    verbs_morph_s = sum(1 for v in verbs.values() if v["source"] == "morph_s")
    entity_occ = sum(entity_freq.values())
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    diag = {
        "schema": SCHEMA,
        "generated": generated,
        "unit_items": unit_items,
        "unit_bigram_types_all": len(bi_all),
        "unit_trigram_types_all": len(tri_all),
        "unit_bigram_positions": sum(bi_all.values()),
        "unit_trigram_positions": sum(tri_all.values()),
        "unit_bigrams_ge_min": len(big),
        "unit_trigrams_ge_min": len(tri),
        "token_items": token_items,
        "token_bigram_types_all": len(tok_bi_all),
        "token_trigram_types_all": len(tok_tri_all),
        "token_bigram_positions": sum(tok_bi_all.values()),
        "token_trigram_positions": sum(tok_tri_all.values()),
        "token_bigrams_ge_min": len(tok_bi),
        "token_bigrams_with_op": sum(1 for r in tok_bi if r["has_op"] == "y"),
        "markup_units_removed": len(excluded_uids),
        "markup_n19_residue": len(markup_uids),
        "markup_extra_removed": len(excluded_uids) - len(markup_uids),
        "screened_items": len(unit_meta) - len(markup_uids),
        "markup_tokens_removed": n_markup_tokens,
        "entities": len(entities),
        "entity_occurrences": entity_occ,
        "entity_partner_pairs": len(coll),
        "entity_verb_pairs": len(verb_pairs),
        "verbs": len(verbs),
        "verbs_brown": verbs_brown,
        "verbs_book": verbs_book,
        "verbs_morph_s": verbs_morph_s,
        "brown_files": n_files,
        "window": args.window,
        "min_bigram": args.min_bigram,
        "min_trigram": args.min_trigram,
    }

    # ---- outputs ----
    write_tsv(out / "collocations_bigram.tsv",
              ["x", "y", "f_x", "f_y", "f_xy", "n", "pmi", "ppmi", "logdice",
               "dice", "tscore", "x_entity", "y_entity", "x_verb", "y_verb"],
              big)
    write_tsv(out / "collocations_trigram.tsv",
              ["x", "y", "z", "f_x", "f_y", "f_z", "f_xyz", "f_xy", "f_yz",
               "n", "pmi3", "logdice3", "x_entity", "y_entity", "z_entity",
               "x_verb", "y_verb", "z_verb"], tri)
    write_tsv(out / "collocations_bigram_token.tsv",
              ["x", "y", "x_kind", "y_kind", "f_x", "f_y", "f_xy", "n", "pmi",
               "ppmi", "logdice", "dice", "tscore", "has_op", "x_entity",
               "y_entity", "x_verb", "y_verb"], tok_bi)
    write_tsv(out / "collocations_trigram_token.tsv",
              ["x", "y", "z", "x_kind", "y_kind", "z_kind", "f_xyz", "n",
               "pmi3", "logdice3", "has_op"], tok_tri)
    coll_cols = ["entity", "entity_lower", "entity_categories", "entity_sources",
                 "partner", "n_pair", "n_entity", "n_partner", "n_left",
                 "n_right", "n_adjacent_left", "n_adjacent_right",
                 "min_distance", "partner_is_verb", "verb_source",
                 "verb_confidence", "partner_is_entity", "pmi", "ppmi",
                 "logdice", "dice", "tscore", "example_context"]
    write_tsv(out / "entity_collocations.tsv", coll_cols, coll)
    write_tsv(out / "entity_verb_pairs.tsv", coll_cols, verb_pairs)
    write_tsv(out / "entity_inventory.tsv",
              ["entity", "lower", "categories", "sources", "n_occurrences"],
              [{"entity": e["surface"], "lower": k,
                "categories": "|".join(sorted(e["categories"])),
                "sources": "|".join(sorted(e["sources"])),
                "n_occurrences": entity_freq.get(k, 0)}
               for k, e in sorted(entities.items())])
    write_tsv(out / "diagnostics.tsv", ["diag", "value"],
              [{"diag": k, "value": v} for k, v in diag.items()])

    recon = [
        {"quantity": "pages", "earlier": "225 (#7/#17)", "here": "225",
         "note": "same layout markdown"},
        {"quantity": "units", "earlier": "74,665 (#17)", "here": unit_items,
         "note": f"minus {len(excluded_uids)} markup occurrences "
                 f"(#19 ledger + junk types + tag-run detector)"},
        {"quantity": "screened occurrences", "earlier": "74,318 (#19/#24)",
         "here": diag["screened_items"],
         "note": "74,665 minus #19's 347 markup-residue occurrences"},
        {"quantity": "collocation unit items", "earlier": "74,318 (#19/#24)",
         "here": unit_items,
         "note": f"also minus {diag['markup_extra_removed']} tag-furniture "
                 f"occurrences #19 left in as `junk`/compound "
                 f"(`href`, `mailto`, `colspan`)"},
        {"quantity": "tokens", "earlier": "90,220 (#17)", "here": token_items,
         "note": f"minus {n_markup_tokens} tokens inside markup spans"},
        {"quantity": "operator-word bigrams", "earlier": "#52 (open)",
         "here": diag["token_bigrams_with_op"],
         "note": "included here per #27's description; #52 owns the MI matrix"},
    ]
    write_tsv(out / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    summary = {
        "schema": SCHEMA, "generated": generated, "counts": diag,
        "top_unit_bigram_by_freq": [
            {"x": r["x"], "y": r["y"], "f_xy": r["f_xy"], "logdice": r["logdice"]}
            for r in sorted(big, key=lambda r: -r["f_xy"])[:10]],
        "top_unit_bigram_by_logdice": [
            {"x": r["x"], "y": r["y"], "f_xy": r["f_xy"], "logdice": r["logdice"]}
            for r in big[:10]],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    diag_for_report = dict(diag)
    diag_for_report["entity_freq"] = entity_freq
    write_report(out / "collocation_report.md", big, tri, tok_bi, tok_tri,
                 coll, verb_pairs, entities, verbs, diag_for_report)

    if not args.quiet:
        print(f"[collocation] unit bigrams={len(big)} trigrams={len(tri)} "
              f"token bigrams={len(tok_bi)} with_op={diag['token_bigrams_with_op']}")
        print(f"[collocation] entities={len(entities)} "
              f"entity_pairs={len(coll)} entity_verb_pairs={len(verb_pairs)} "
              f"verbs={len(verbs)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
