#!/usr/bin/env python3
"""Multi-engine ensemble OCR with glyph voting - Chainlink #4.

L0 step 2 of `docs/meaning-extraction-methods.md`: re-OCR the same pages with
several independent engines, align the outputs, take a character-level vote, and
produce a *disagreement atlas* that localizes the genuinely ambiguous glyphs
(`::` vs `:`, `<<` vs `<`, backtick vs apostrophe, `l`/`I`/`1`, ...).

The reference engine is the corpus engine, marker/surya (#3 already dumped its
per-character stream and confidences).  Three independent engines are added:

    tesseract 5.3   (LSTM, CLI)          -- different training data, different
                                            layout analysis, different errors
    docTR 1.0       (db_resnet50 + crnn) -- different detector and recogniser
    easyocr 1.7     (CRAFT + CRNN)       -- third architecture

Why not the method doc's literal list (Tesseract/PaddleOCR/docTR)?  PaddleOCR
was not installable in this environment; EasyOCR replaces it as the third
independent engine.  The requirement is ">= 2 other engines"; three are run.

Subcommands
-----------
  ocr    run the requested engines over a page range, appending one JSON record
         per (engine, page) to a resumable cache under analysis/ensemble_ocr/
  build  align the cached engines against marker, vote, and write the consensus
         transcript + disagreement atlas + report

Heavy engines are run in the background, e.g.:

  source ~/anaconda3/etc/profile.d/conda.sh && conda activate marker
  nohup python work/ensemble_ocr.py ocr --engines tesseract,doctr,easyocr \
        > analysis/ensemble_ocr/ocr_run.log 2>&1 &
  python work/ensemble_ocr.py build

The pure functions (`normalize_ws`, `align_pair`, `build_columns`, `vote`,
`categorize`) are unit-tested in work/test_ensemble_ocr.py without needing a GPU.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import difflib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "analysis", "ensemble_ocr")
ENGINE_DIR = os.path.join(OUT, "engines")
PDF = os.path.join(HERE, "input", "blood-electric.pdf")
# #3's per-block line table is the marker reference; join on (page, block),
# never on a token index.
LINES_TSV = os.path.join(ROOT, "analysis", "ocr_confidence", "lines.tsv")

SCHEMA = "ensemble-ocr/1"
ALL_ENGINES = ["marker", "tesseract", "doctr", "easyocr"]
DEFAULT_ENGINES = ["tesseract", "doctr", "easyocr"]
DPI = 300

# ---------------------------------------------------------------------------
# glyph classes - fixed before looking at the atlas, so the categories are not
# tuned to the result
# ---------------------------------------------------------------------------
OPERATOR_CHARS = set(":;<>/=\\`'|!+-*&^%~@#$[]{}()")
WHITESPACE = set(" \t\r\n\f\v")
# marker's surviving HTML markup (#17/#19: mailto anchors, <th> page furniture,
# </p>, ...).  The book's own `<...>` words must NOT match, so only real tag
# names are listed.  Columns inside a match are flagged, not treated as glyph
# ambiguity.
_HTML_TAG = re.compile(
    r"</?(?:p|b|i|u|em|strong|br|hr|h[1-6]|sub|sup|span|div|img|content-ref|math"
    r"|table|thead|tbody|tr|td|th|ul|ol|li|figure|figcaption|a)\b[^<>]{0,200}>"
    r"|<a\s+href[^<>]*"          # fragmented mailto anchor: `<a href="mailto:` (no `>`)
    r"|</a>",
    re.IGNORECASE)


def markup_spans(text):
    """(start, end) spans of real HTML tags in `text` (the book's `<words>` are
    excluded by the tag-name whitelist)."""
    return [(m.start(), m.end()) for m in _HTML_TAG.finditer(text)]
# classic OCR homoglyph families
DIGIT_LETTER = [
    set("1lI|i"), set("0Oo"), set("5S"), set("8B"), set("2Z"), set("6G"),
    set("9gq"), set("4A"), set("7T"),
]
# the specific collisions the task names
FOCUS_GLYPHS = [
    ("::", ":"), ("<<", "<"), (">>", ">"), ("//", "/"),
    ("`", "'"), ("`", '"'), ("l", "I"), ("l", "1"), ("I", "1"),
    ("0", "O"), ("o", "O"), ("-", "="), ("=", "-"),
]


# ==========================================================================
# whitespace normalisation with an offset map
# ==========================================================================
def normalize_ws(text):
    """Collapse every whitespace run to a single space.

    Returns (norm, norm_to_orig) where norm_to_orig[k] is the index in the
    original string of the character that produced norm[k].  A trailing
    whitespace run is dropped so the alignment never ends on a space.
    """
    out = []
    idx = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] in WHITESPACE:
            j = i
            while j < n and text[j] in WHITESPACE:
                j += 1
            if out:  # never start with whitespace, never emit two in a row
                out.append(" ")
                idx.append(i)
            i = j
        else:
            out.append(text[i])
            idx.append(i)
            i += 1
    # strip a single trailing space
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


# ==========================================================================
# pairwise character alignment against a reference
# ==========================================================================
def align_pair(ref, eng):
    """Align `eng` to `ref` at the character level.

    Returns (mapping, insertions):
      mapping[i]   = index into eng aligned to ref[i], or None (deletion)
      insertions[b]= list of indices into eng inserted at boundary b
                     (between ref[b-1] and ref[b]; b in 0..len(ref))
    Uses difflib's longest-matching-block opcodes; a `replace` block of unequal
    length is paired positionally and the surplus becomes deletion/insertion.
    """
    mapping = [None] * len(ref)
    insertions = collections.defaultdict(list)
    if not ref and not eng:
        return mapping, insertions
    sm = difflib.SequenceMatcher(None, ref, eng, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag == "replace":
            li, lj = i2 - i1, j2 - j1
            m = min(li, lj)
            for k in range(m):
                mapping[i1 + k] = j1 + k
            if lj > li:                     # surplus in engine -> insertion
                insertions[i2].extend(range(j1 + li, j2))
            # surplus in ref stays None -> deletion
        elif tag == "insert":
            insertions[i1].extend(range(j1, j2))
        # 'delete' leaves the mapping as None
    return mapping, insertions


def align_multi(seqs):
    """Progressive multiple alignment of a short group of equal-length strings.

    `seqs` is a list of strings (one per engine) that all belong to the same
    insertion slot.  Returns a list of columns; each column is a list, one entry
    per engine, of a character or None.  The first non-empty string is the
    pivot; every other string is aligned to it and its surplus becomes extra
    columns.
    """
    n = len(seqs)
    piv = next((i for i, s in enumerate(seqs) if s), None)
    if piv is None:
        return []
    pivot = seqs[piv]
    cols = [[None] * n for _ in pivot]
    for k, ch in enumerate(pivot):
        cols[k][piv] = ch
    for i, s in enumerate(seqs):
        if i == piv or not s:
            continue
        mapping, ins = align_pair(pivot, s)
        for k, j in enumerate(mapping):
            if j is not None:
                cols[k][i] = s[j]
        off = 0
        for b in sorted(ins):
            chars = [s[j] for j in ins[b]]
            newcols = [[None] * n for _ in chars]
            for c, ch in enumerate(chars):
                newcols[c][i] = ch
            pos = b + off
            cols[pos:pos] = newcols
            off += len(chars)
    return cols


def build_columns(ref, engine_texts):
    """Build the aligned column matrix for one page.

    `ref` is the marker reference text; `engine_texts` is a dict
    {engine_name: text} for the *other* engines.  Returns a list of columns;
    each column is a dict with `ref_index` (int or None for an inserted column)
    and `values` = {engine_name: char or None}, marker included.
    """
    engines = ["marker"] + [e for e in engine_texts]
    per = {"marker": ([i for i in range(len(ref))], {})}
    for name, text in engine_texts.items():
        per[name] = align_pair(ref, text)

    cols = []
    for b in range(len(ref) + 1):
        # insertion columns for this boundary
        ins_strings = []
        for name in engines:
            _m, ins = per[name]
            if name == "marker":
                ins_strings.append("")
            else:
                text = engine_texts[name]
                ins_strings.append("".join(text[j] for j in ins.get(b, [])))
        if any(ins_strings):
            for col in align_multi(ins_strings):
                vals = {name: col[i] for i, name in enumerate(engines)}
                cols.append({"ref_index": None, "boundary": b, "values": vals})
        if b < len(ref):
            vals = {}
            for name in engines:
                m, _ins = per[name]
                j = m[b]
                if name == "marker":
                    vals[name] = ref[b]
                else:
                    vals[name] = engine_texts[name][j] if j is not None else None
            cols.append({"ref_index": b, "boundary": b, "values": vals})
    return cols


def vote(values, engines):
    """Character-level vote over one column.

    `values` = {engine: char or None}.  None is a first-class vote meaning
    "this engine saw no character here" (a deletion/insertion signal).  Returns
    (consensus_char_or_None, n_agree, verdict) where verdict is one of
    'unanimous', 'majority' (a strict majority exists), 'ambiguous'.
    """
    vals = [values.get(e) for e in engines]
    counts = collections.Counter(vals)
    top, topn = counts.most_common(1)[0]
    threshold = len(engines) // 2 + 1
    if len(counts) == 1:
        return top, len(engines), "unanimous"
    if topn >= threshold:
        return top, topn, "majority"
    return None, topn, "ambiguous"


def categorize(vals, ref_char):
    """Classify a non-unanimous column by the kind of glyph collision.

    `vals` is the list of per-engine values (char or None).
    """
    present = [v for v in vals if v is not None]
    uniq = set(present)
    has_none = any(v is None for v in vals)
    if uniq and all(c in WHITESPACE for c in uniq):
        return "whitespace"
    if len(uniq) <= 1:
        if has_none:
            return "deletion" if ref_char is not None else "insertion"
        return "case" if len(uniq) == 0 else "other"
    if all(c in OPERATOR_CHARS for c in uniq):
        return "operator"
    if any(c in WHITESPACE for c in uniq):
        return "whitespace"
    if all(c.isalpha() for c in uniq) and len({c.lower() for c in uniq}) == 1:
        return "case"
    for fam in DIGIT_LETTER:
        if uniq <= fam:
            return "digit_letter"
    if all(c.isalpha() for c in uniq):
        return "letter"
    return "other"


def pair_label(vals):
    """Stable label for a confusion set, e.g. '∅|:|::' -> sorted '::|:'.

    Returns 'char|char|...' over the distinct values, with '∅' for None.
    """
    labels = sorted({("∅" if v is None else v) for v in vals})
    return "|".join(labels)


# ==========================================================================
# engine runners
# ==========================================================================
def render_page(pdf, page_id, dpi=DPI):
    """Return a PIL RGB image of the 0-based PDF page."""
    bmp = pdf[page_id].render(scale=dpi / 72.0)
    return bmp.to_pil().convert("RGB")


def run_tesseract(img, psm=3, lang="eng"):
    """Tesseract via CLI, TSV output -> (text, words).

    words = [(word, x0, y0, x1, y1, conf, line_key)] in image pixels.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", str(psm), "-l", lang, "tsv"],
        input=buf.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr.decode(errors='replace')[:400]}")
    words = []
    lines = collections.OrderedDict()
    for ln in proc.stdout.decode("utf-8", errors="replace").splitlines():
        parts = ln.split("\t")
        if len(parts) < 12 or parts[0] != "5":
            continue
        _lvl, _pg, block, par, line, _w = parts[:6]
        left, top, width, height = (int(float(parts[6])), int(float(parts[7])),
                                    int(float(parts[8])), int(float(parts[9])))
        conf = float(parts[10])
        text = parts[11]
        if not text:
            continue
        key = (block, par, line)
        words.append((text, left, top, left + width, top + height, conf, key))
        lines.setdefault(key, []).append(text)
    text = "\n".join(" ".join(v) for v in lines.values())
    return text, words


def load_doctr_model():
    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True)
    try:
        if hasattr(model, "cuda"):
            model = model.cuda()
    except Exception:
        pass
    return model


def run_doctr(model, img, page_id):
    """docTR -> (text, words).  Boxes normalised by docTR, converted to pixels."""
    import numpy as np
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    # the predictor takes a list of HxWx3 numpy arrays directly
    res = model([arr])
    words = []
    line_texts = []
    for page in res.export()["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                toks = []
                for wd in line["words"]:
                    g = wd["geometry"]
                    x0 = g[0][0] * w
                    y0 = g[0][1] * h
                    x1 = g[1][0] * w
                    y1 = g[1][1] * h
                    words.append((wd["value"], x0, y0, x1, y1, wd.get("confidence", -1.0), None))
                    toks.append(wd["value"])
                line_texts.append(" ".join(toks))
    return "\n".join(line_texts), words


def load_easyocr_model():
    import easyocr
    return easyocr.Reader(["en"], gpu=True, verbose=False)


def run_easyocr(reader, img):
    """EasyOCR -> (text, words).  Four-point boxes reduced to axis-aligned."""
    import numpy as np
    arr = np.asarray(img)
    res = reader.readtext(arr, detail=1, paragraph=False)
    words = []
    toks = []
    for box, text, conf in res:
        if not text:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        words.append((text, min(xs), min(ys), max(xs), max(ys), float(conf), None))
        toks.append(text)
    return " ".join(toks), words


def load_marker_reference():
    """Per page: the marker block lines from #3, in block order."""
    pages = collections.defaultdict(list)
    with open(LINES_TSV, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            pid = int(parts[col["page_id"]])
            pages[pid].append({
                "block_id": parts[col["block_id"]],
                "block_index": int(parts[col["block_index"]]),
                "line_index": int(parts[col["line_index"]]),
                "bbox": parts[col["bbox"]],
                "text": parts[col["text"]].replace("\\n", "\n"),
            })
    for pid in pages:
        pages[pid].sort(key=lambda r: (r["block_index"], r["line_index"]))
    return pages


# ==========================================================================
# ocr subcommand
# ==========================================================================
def engine_cache_path(engine):
    return os.path.join(ENGINE_DIR, f"{engine}.jsonl")


def read_engine_cache(engine):
    path = engine_cache_path(engine)
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[int(rec["page_id"])] = rec
    return done


def append_engine_cache(engine, rec):
    os.makedirs(ENGINE_DIR, exist_ok=True)
    with open(engine_cache_path(engine), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def cmd_ocr(args):
    import pypdfium2 as pdfium
    engines = [e for e in args.engines.split(",") if e]
    unknown = [e for e in engines if e not in ALL_ENGINES or e == "marker"]
    if unknown:
        print(f"[ocr] unknown/unrunnable engines: {unknown}", file=sys.stderr)
        return 2
    pdf = pdfium.PdfDocument(args.pdf)
    n = len(pdf)
    if args.page_range:
        lo, hi = (int(x) for x in args.page_range.split("-"))
        pages = list(range(lo, hi + 1))
    else:
        pages = list(range(n))

    models = {}
    for eng in engines:
        done = read_engine_cache(eng)
        if args.force:
            done = {}
        todo = [p for p in pages if p not in done]
        print(f"[ocr] {eng}: {len(done)} cached, {len(todo)} to run", flush=True)
        if not todo:
            continue
        if eng == "doctr":
            t = time.time()
            models[eng] = load_doctr_model()
            print(f"[ocr] {eng} model loaded in {time.time() - t:.1f}s", flush=True)
        elif eng == "easyocr":
            t = time.time()
            models[eng] = load_easyocr_model()
            print(f"[ocr] {eng} model loaded in {time.time() - t:.1f}s", flush=True)
        for k, pid in enumerate(todo, 1):
            t = time.time()
            try:
                img = render_page(pdf, pid, args.dpi)
                if eng == "tesseract":
                    text, words = run_tesseract(img, psm=args.psm)
                elif eng == "doctr":
                    text, words = run_doctr(models[eng], img, pid)
                elif eng == "easyocr":
                    text, words = run_easyocr(models[eng], img)
                rec = {
                    "page_id": pid, "engine": eng, "text": text,
                    "words": [[str(w[0]), round(float(w[1]), 1), round(float(w[2]), 1),
                               round(float(w[3]), 1), round(float(w[4]), 1),
                               round(float(w[5]), 4) if w[5] is not None else None]
                              for w in words],
                    "n_words": len(words), "elapsed_s": round(time.time() - t, 2),
                    "dpi": args.dpi, "error": None,
                }
            except Exception as exc:  # keep the run resumable
                rec = {"page_id": pid, "engine": eng, "text": "", "words": [],
                       "n_words": 0, "elapsed_s": round(time.time() - t, 2),
                       "dpi": args.dpi, "error": repr(exc)[:500]}
                print(f"[ocr] {eng} page {pid} ERROR {exc!r}", file=sys.stderr, flush=True)
            append_engine_cache(eng, rec)
            if k % 10 == 0 or k == len(todo):
                print(f"[ocr] {eng} {k}/{len(todo)} (page {pid}, {rec['elapsed_s']}s)",
                      flush=True)
    return 0


# ==========================================================================
# build subcommand
# ==========================================================================
def words_to_spans(text, words):
    """Map each engine word to (start, end, bbox) offsets in its own text.

    `text` is assumed to be the words joined by single spaces / newlines, in
    the order they appear in `words`.
    """
    spans = []
    pos = 0
    for w in words:
        wt = w[0]
        if not wt:
            continue
        # the word should appear at the next position; tolerate drift
        found = text.find(wt, pos)
        if found < 0:
            found = pos
        spans.append((found, found + len(wt), (w[1], w[2], w[3], w[4]), w[5]))
        pos = found + len(wt)
    return spans


def locate(spans, idx):
    """Return (bbox, conf, word) for the word span containing text index idx."""
    if idx is None:
        return None, None, None
    for start, end, bbox, conf in spans:
        if start <= idx < end:
            return bbox, conf, None
    # nearest preceding
    best = None
    for start, end, bbox, conf in spans:
        if start <= idx:
            best = (bbox, conf)
    if best:
        return best[0], best[1], None
    return None, None, None


def build_page(page_id, ref_lines, engine_pages, engines, words_by_engine, dpi):
    """Align one page and return (columns, consensus_text, events).

    events: one dict per non-unanimous column with page-localization.
    """
    ref_text = "\n".join(r["text"] for r in ref_lines)
    ref_norm, ref_map = normalize_ws(ref_text)
    # map normalised ref index -> line info, and line -> start offset
    line_of = []
    line_starts = {}
    off = 0
    for i, r in enumerate(ref_lines):
        line_starts[(r["block_index"], r["line_index"])] = off
        line_of.extend([r] * len(r["text"]))
        off += len(r["text"])
        if i < len(ref_lines) - 1:
            line_of.append(r)
            off += 1
    fallback = ref_lines[-1] if ref_lines else {"block_id": "", "line_index": 0}
    while len(line_of) < len(ref_text) + 1:
        line_of.append(fallback)

    engine_norm = {}
    engine_norm_map = {}
    engine_texts = {}
    for eng in engines:
        if eng == "marker":
            continue
        rec = engine_pages.get(eng)
        if not rec or not rec.get("text"):
            continue  # missing or errored page: do not vote it as a deletion
        t, m = normalize_ws(rec["text"])
        engine_norm[eng] = t
        engine_norm_map[eng] = m
        engine_texts[eng] = t

    mark_spans = markup_spans(ref_text)
    active = ["marker"] + list(engine_texts)
    maps = {eng: align_pair(ref_norm, engine_texts[eng]) for eng in engine_texts}
    cols = build_columns(ref_norm, engine_texts)
    consensus_parts = []
    events = []
    for ci, col in enumerate(cols):
        vals = col["values"]
        ref_idx = col["ref_index"]
        ref_char = ref_norm[ref_idx] if ref_idx is not None else None
        consensus, n_agree, verdict = vote(vals, active)
        if verdict == "ambiguous":
            # no strict majority: keep marker (None for an inserted column)
            consensus = vals.get("marker")
        consensus_parts.append(consensus if consensus is not None else "")
        if verdict == "unanimous":
            continue
        vals_list = [vals.get(e) for e in engines]
        # ---- localisation: marker original offset -> line + char-in-line ----
        orig = ref_map[ref_idx] if ref_idx is not None else (
            ref_map[col["boundary"]] if col["boundary"] < len(ref_map) else len(ref_text))
        li = line_of[orig] if 0 <= orig < len(line_of) else line_of[-1]
        char_in_line = orig - sum(len(r["text"]) + 1 for r in ref_lines
                                  if r["block_index"] < li["block_index"] or
                                  (r["block_index"] == li["block_index"] and
                                   r["line_index"] < li["line_index"]))
        # ---- bbox from the first engine that has the character --------------
        bbox = None
        bbox_engine = None
        for eng in active:
            if eng == "marker":
                continue
            m = engine_norm_map[eng]
            j = None
            # recover engine index from the aligned char
            if ref_idx is not None:
                mp, _ins = maps[eng]
                j = mp[ref_idx]
            spans = words_by_engine.get(eng, {}).get(page_id, [])
            if j is not None and j < len(m):
                bb, _conf, _ = locate(spans, m[j])
                if bb:
                    bbox = bb
                    bbox_engine = eng
                    break
        anchor = ref_idx if ref_idx is not None else min(col["boundary"], len(ref_norm))
        ctx_lo = max(0, anchor - 30)
        ctx_hi = min(len(ref_norm), anchor + 31)
        in_markup = any(s <= orig < e for s, e in mark_spans)
        events.append({
            "page_id": page_id,
            "ref_offset": ref_idx if ref_idx is not None else "",
            "boundary": col["boundary"],
            "block_id": li["block_id"],
            "line_index": li["line_index"],
            "char_in_line": char_in_line,
            "marker_char": ref_char if ref_char is not None else "",
            "marker_prev": (ref_norm[ref_idx - 1]
                            if ref_idx is not None and ref_idx > 0 else ""),
            "marker_next": (ref_norm[ref_idx + 1]
                            if ref_idx is not None and ref_idx + 1 < len(ref_norm) else ""),
            "tesseract": vals.get("tesseract") or "",
            "doctr": vals.get("doctr") or "",
            "easyocr": vals.get("easyocr") or "",
            "consensus_char": consensus if consensus is not None else "",
            "n_agree": n_agree,
            "verdict": verdict,
            "category": "markup" if in_markup else categorize(vals_list, ref_char),
            "in_markup": 1 if in_markup else 0,
            "pair": pair_label(vals_list),
            "bbox_engine": bbox_engine or "",
            "bbox": ",".join(str(round(v, 1)) for v in bbox) if bbox else "",
            "context": ref_norm[ctx_lo:ctx_hi],
        })
    return cols, "".join(consensus_parts), events, ref_text, ref_norm


def engine_versions():
    """Best-effort provenance: the exact engine builds used."""
    v = {}
    try:
        out = subprocess.run(["tesseract", "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, check=False).stdout
        v["tesseract"] = out.splitlines()[0].strip()
    except Exception as exc:                                    # pragma: no cover
        v["tesseract"] = f"unknown ({exc!r})"
    for name, dist in [("marker", "marker-pdf"), ("surya", "surya-ocr"),
                       ("doctr", "python-doctr"), ("easyocr", "easyocr")]:
        try:
            import importlib.metadata as md
            v[name] = md.version(dist)
        except Exception:
            try:
                mod = __import__(name if name != "marker" else "marker")
                v[name] = getattr(mod, "__version__", "unknown")
            except Exception:
                v[name] = "not installed"
    return v


def cmd_build(args):
    os.makedirs(OUT, exist_ok=True)
    ref_pages = load_marker_reference()
    engine_pages = {}
    engine_words = {}
    engines = ["marker"]
    for eng in ALL_ENGINES:
        if eng == "marker":
            continue
        cache = read_engine_cache(eng)
        if cache:
            engine_pages[eng] = cache
            engines.append(eng)
    # marker is always present (the reference); require >=2 other engines
    if len(engines) - 1 < 2:
        print(f"[build] need >=2 non-marker engines, found {engines[1:]}", file=sys.stderr)
        return 2

    for eng in engines[1:]:
        words_by_engine = []
        for pid, rec in engine_pages[eng].items():
            words_by_engine.append((pid, words_to_spans(rec["text"], rec["words"])))
        engine_words[eng] = dict(words_by_engine)

    pages = sorted(ref_pages)
    if args.page_range:
        lo, hi = (int(x) for x in args.page_range.split("-"))
        pages = [p for p in pages if lo <= p <= hi]
    all_events = []
    consensus_rows = []
    page_rows = []
    page_length_rows = []
    total_cols = 0
    total_changed = 0
    engine_agree = collections.Counter()
    engine_disagree = collections.Counter()
    engine_present = collections.Counter()

    for pid in pages:
        ref_lines = ref_pages[pid]
        ep = {eng: engine_pages[eng].get(pid) for eng in engines[1:]}
        cols, consensus, events, ref_text, ref_norm = build_page(
            pid, ref_lines, ep, engines, engine_words, args.dpi)
        total_cols += len(cols)
        for col in cols:
            if col["ref_index"] is None:
                continue  # agreement is measured on marker's own columns
            vals = col["values"]
            m = vals.get("marker")
            for eng in engines[1:]:
                engine_present[eng] += 1
                if vals.get(eng) == m:
                    engine_agree[eng] += 1
                else:
                    engine_disagree[eng] += 1
        all_events.extend(events)
        changed = sum(1 for e in events
                      if e["consensus_char"] != e["marker_char"])
        total_changed += changed
        consensus_rows.append((pid, consensus, ref_text))
        eng_lens = {}
        for eng in engines[1:]:
            rec = ep.get(eng)
            eng_lens[eng] = (len(normalize_ws(rec["text"])[0])
                             if rec and rec.get("text") else "")
        lens = [v for v in eng_lens.values() if isinstance(v, int)]
        mean_len = statistics.fmean(lens) if lens else 0
        words = ref_norm.split()
        wc = collections.Counter(tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3)))
        top_gram, top_rep = wc.most_common(1)[0] if wc else ((), 0)
        page_length_rows.append({
            "page_id": pid, "marker_len": len(ref_norm),
            "tesseract_len": eng_lens.get("tesseract", ""),
            "doctr_len": eng_lens.get("doctr", ""),
            "easyocr_len": eng_lens.get("easyocr", ""),
            "mean_engine_len": round(mean_len, 1),
            "marker_over_engines": round(len(ref_norm) / mean_len, 3) if mean_len else "",
            "max_word4gram_repeat": top_rep,
            "top_word4gram": " ".join(top_gram),
        })
        page_rows.append({
            "page_id": pid, "n_cols": len(cols), "n_events": len(events),
            "n_markup": sum(1 for e in events if e["in_markup"]),
            "n_changed": changed,
            "pct_disagree": round(100.0 * len(events) / max(1, len(cols)), 3),
            "pct_disagree_ex_markup": round(
                100.0 * sum(1 for e in events if not e["in_markup"]) / max(1, len(cols)), 3),
            "ref_chars": len(ref_norm),
        })

    # ---------------- write consensus transcript ----------------
    with open(os.path.join(OUT, "consensus.tsv"), "w", encoding="utf-8") as fh:
        fh.write("page_id\tconsensus_text\n")
        for pid, consensus, _ref in consensus_rows:
            fh.write(f"{pid}\t{consensus.replace(chr(9), ' ')}\n")
    with open(os.path.join(OUT, "consensus.md"), "w", encoding="utf-8") as fh:
        fh.write("# Blood Electric - ensemble consensus transcript\n\n")
        fh.write("Character-level majority vote over marker/surya + "
                 + ", ".join(engines[1:])
                 + ". Each section is tagged with an HTML comment giving the "
                   "0-based PDF page index (the #7/#8 citation convention).\n\n")
        for pid, consensus, _ref in consensus_rows:
            fh.write(f"<!-- page {pid} -->\n\n{consensus}\n\n")
    with open(os.path.join(OUT, "marker_reference.tsv"), "w", encoding="utf-8") as fh:
        fh.write("page_id\tmarker_text\n")
        for pid, _c, ref in consensus_rows:
            escaped = ref.replace("\t", " ").replace("\n", "\\n")
            fh.write(f"{pid}\t{escaped}\n")

    # ---------------- write atlas ----------------
    ev_cols = ["page_id", "ref_offset", "boundary", "block_id", "line_index",
               "char_in_line", "marker_char", "marker_prev", "marker_next",
               "tesseract", "doctr", "easyocr",
               "consensus_char", "n_agree", "verdict", "category", "in_markup",
               "pair", "bbox_engine", "bbox", "context"]
    with open(os.path.join(OUT, "disagreement_atlas.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\t".join(ev_cols) + "\n")
        for e in all_events:
            row = []
            for c in ev_cols:
                v = e[c]
                if c in ("tesseract", "doctr", "easyocr", "marker_char", "consensus_char"):
                    v = {"\t": "\\t", "\n": "\\n", "\r": "\\r", "": "∅"}.get(v, v)
                row.append(str(v).replace("\t", " ").replace("\n", "\\n"))
            fh.write("\t".join(row) + "\n")

    # ---------------- summaries ----------------
    def write_tsv(name, header, rows):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for r in rows:
                fh.write("\t".join(str(x) for x in r) + "\n")

    pair_counts = collections.Counter(e["pair"] for e in all_events)
    pair_counts_x = collections.Counter(e["pair"] for e in all_events if not e["in_markup"])
    cat_counts = collections.Counter(e["category"] for e in all_events)
    verdict_counts = collections.Counter(e["verdict"] for e in all_events)
    write_tsv("confusion_pairs.tsv", ["pair", "n", "pct"],
              [(p, n, round(100.0 * n / max(1, len(all_events)), 3))
               for p, n in pair_counts.most_common()])
    write_tsv("category_counts.tsv", ["category", "n", "pct"],
              [(c, n, round(100.0 * n / max(1, len(all_events)), 3))
               for c, n in cat_counts.most_common()])
    write_tsv("verdict_counts.tsv", ["verdict", "n"],
              [(v, n) for v, n in verdict_counts.most_common()])
    write_tsv("page_disagreement.tsv",
              ["page_id", "n_cols", "n_events", "n_markup", "n_changed",
               "pct_disagree", "pct_disagree_ex_markup", "ref_chars"],
              [[r[c] for c in ["page_id", "n_cols", "n_events", "n_markup",
                               "n_changed", "pct_disagree",
                               "pct_disagree_ex_markup", "ref_chars"]]
               for r in page_rows])
    write_tsv("engine_agreement.tsv", ["engine", "present", "agree_with_marker",
                                       "disagree_with_marker", "pct_agree"],
              [[eng, engine_present[eng], engine_agree[eng], engine_disagree[eng],
                round(100.0 * engine_agree[eng] / max(1, engine_present[eng]), 3)]
               for eng in engines[1:]])
    write_tsv("page_length.tsv",
              ["page_id", "marker_len", "tesseract_len", "doctr_len", "easyocr_len",
               "mean_engine_len", "marker_over_engines", "max_word4gram_repeat",
               "top_word4gram"],
              [[r[c] for c in ["page_id", "marker_len", "tesseract_len", "doctr_len",
                               "easyocr_len", "mean_engine_len", "marker_over_engines",
                               "max_word4gram_repeat", "top_word4gram"]]
               for r in page_length_rows])
    gross = [r for r in page_length_rows
             if isinstance(r["marker_over_engines"], float)
             and (r["marker_over_engines"] >= 1.3 or r["marker_over_engines"] <= 0.7)]
    repetitions = [r for r in page_length_rows if r["max_word4gram_repeat"] >= 10]

    # focused glyph table
    glyph_events = [e for e in all_events if not e["in_markup"]]
    focus_rows = []
    for a, b in FOCUS_GLYPHS:
        n = sum(1 for e in glyph_events if {a, b} <= set(e["pair"].split("|")))
        # count where the pair is exactly these two (plus at most ∅)
        exact = sum(1 for e in glyph_events
                    if set(e["pair"].split("|")) <= {a, b, "∅"}
                    and a in e["pair"] and b in e["pair"])
        focus_rows.append([a, b, n, exact])
    write_tsv("focus_glyphs.tsv", ["glyph_a", "glyph_b", "co_occurring", "exact_pair"],
              focus_rows)

    # doubled-operator losses: `::` read as `:` etc. show up as the deletion of
    # one copy of the glyph, so they are counted by marker-neighbour, not by a
    # single-column substitution.
    doubled = []
    for ch in ["::", "<<", ">>", "//", "++", "--"]:
        c = ch[0]
        evs = [e for e in glyph_events
               if e["marker_char"] == c and (e["marker_prev"] == c or e["marker_next"] == c)]
        dropped = sum(1 for e in evs if "∅" in e["pair"].split("|"))
        doubled.append([ch, c, len(evs), dropped])
    write_tsv("doubled_operators.tsv",
              ["operator", "glyph", "events_touching_run", "events_dropping_a_copy"],
              doubled)

    # changes only
    changed_rows = [e for e in all_events if e["consensus_char"] != e["marker_char"]]
    write_tsv("consensus_changes.tsv", ev_cols,
              [[e[c] for c in ev_cols] for e in changed_rows])

    # ---------------- summary.json ----------------
    summary = {
        "schema": SCHEMA,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "engines": engines,
        "reference_engine": "marker",
        "dpi": args.dpi,
        "pages": len(pages),
        "total_columns": total_cols,
        "total_events": len(all_events),
        "total_markup_events": sum(1 for e in all_events if e["in_markup"]),
        "total_events_ex_markup": sum(1 for e in all_events if not e["in_markup"]),
        "total_changed": total_changed,
        "pct_disagree": round(100.0 * len(all_events) / max(1, total_cols), 4),
        "pct_disagree_ex_markup": round(
            100.0 * sum(1 for e in all_events if not e["in_markup"]) / max(1, total_cols), 4),
        "pct_changed": round(100.0 * total_changed / max(1, total_cols), 4),
        "verdict_counts": dict(verdict_counts),
        "category_counts": dict(cat_counts),
        "engine_agreement": {eng: {
            "present": engine_present[eng],
            "agree": engine_agree[eng],
            "disagree": engine_disagree[eng],
            "pct_agree": round(100.0 * engine_agree[eng] / max(1, engine_present[eng]), 4),
        } for eng in engines[1:]},
        "top_confusion_pairs": [[p, n] for p, n in pair_counts.most_common(25)],
        "top_confusion_pairs_ex_markup": [[p, n] for p, n in pair_counts_x.most_common(25)],
        "gross_divergence_pages": [
            [r["page_id"], r["marker_len"], r["mean_engine_len"],
             r["marker_over_engines"]]
            for r in sorted(gross, key=lambda r: -abs(r["marker_over_engines"] - 1))],
        "repetition_pages": [
            [r["page_id"], r["max_word4gram_repeat"], r["top_word4gram"]]
            for r in sorted(repetitions, key=lambda r: -r["max_word4gram_repeat"])],
        "engine_cache": {eng: len(engine_pages[eng]) for eng in engines[1:]},
        "engine_versions": engine_versions(),
        "config": {"psm": args.psm, "dpi": args.dpi},
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    write_report(summary, all_events, page_rows, focus_rows, engine_words, doubled)
    print(json.dumps(summary, indent=2)[:2000])
    return 0


def write_report(summary, events, page_rows, focus_rows, engine_words, doubled):
    L = []
    p = L.append
    p("# Blood Electric - ensemble OCR consensus & disagreement atlas (L0 / Chainlink #4)")
    p("")
    p(f"_Generated {summary['generated_utc']} by `work/ensemble_ocr.py`._")
    p("")
    p("## 0. Method")
    p("")
    p("Four OCR engines read every page of the source scan at "
      f"{summary['dpi']} dpi: **marker/surya** (the corpus engine, taken from #3's "
      "per-character dump) plus three independent engines - **tesseract 5.3**, "
      "**docTR 1.0** (db_resnet50 + crnn_vgg16_bn) and **easyocr 1.7** (CRAFT + CRNN). "
      "PaddleOCR from the method doc is not installable here; easyocr replaces it as "
      "the third independent engine, so the task's \">= 2 other engines\" is met with three.")
    p("")
    vers = summary.get("engine_versions", {})
    if vers:
        p("Engine builds: " + "; ".join(f"`{k}` {v}" for k, v in vers.items()) + ".")
        p("")
    p("Each engine's text is whitespace-collapsed (runs -> one space) with an offset "
      "map, then aligned to the marker reference at the **character level** with "
      "`difflib` longest-matching-block opcodes. Surplus characters on the engine side "
      "become insertion columns; surplus on the marker side become deletions. Every "
      "column is then voted over the four engines, where **absence is a vote** "
      "(so `::` vs `:` is a 3-of-4 deletion, and an engine-only character is an "
      "insertion). The consensus is the strict-majority character (threshold "
      "`n//2 + 1`); a column with no majority is marked ambiguous and keeps marker.")
    p("")
    p("## 1. Coverage & headline")
    p("")
    p(f"- pages: **{summary['pages']}**; engines: {', '.join(summary['engines'])}")
    p(f"- aligned columns: **{summary['total_columns']:,}**")
    p(f"- non-unanimous columns (the atlas): **{summary['total_events']:,}** "
      f"({summary['pct_disagree']}%)")
    p(f"- of those, **{summary['total_markup_events']:,}** are marker's surviving "
      f"HTML-markup residue (`<a href=\"mailto:...\">`, `<th>`, `</p>`, ...) that the "
      f"image engines cannot see; excluding them, the glyph-disagreement rate is "
      f"**{summary['total_events_ex_markup']:,}** columns "
      f"({summary['pct_disagree_ex_markup']}%)")
    p(f"- columns where the vote overrides marker: **{summary['total_changed']:,}** "
      f"({summary['pct_changed']}%)")
    p("")
    p("The markup events are kept in the atlas (flagged `in_markup=1`, category "
      "`markup`) because they are the ensemble's own detection of marker's markup "
      "residue, and the vote deletes most of them from the consensus; read the glyph "
      "tables below with them excluded.")
    p("")
    p("| verdict | n |")
    p("|---|---|")
    for v, n in sorted(summary["verdict_counts"].items(), key=lambda kv: -kv[1]):
        p(f"| {v} | {n:,} |")
    p("")
    p("## 2. Per-engine agreement with marker (character level)")
    p("")
    p("| engine | present cols | agree | disagree | % agree |")
    p("|---|---|---|---|---|")
    for eng, s in summary["engine_agreement"].items():
        p(f"| {eng} | {s['present']:,} | {s['agree']:,} | {s['disagree']:,} | {s['pct_agree']}% |")
    p("")
    p("## 3. What kind of disagreement")
    p("")
    p("| category | n | % |")
    p("|---|---|---|")
    for c, n in sorted(summary["category_counts"].items(), key=lambda kv: -kv[1]):
        p(f"| {c} | {n:,} | {round(100.0*n/max(1,summary['total_events']),2)}% |")
    p("")
    p("## 4. Top confusion sets (markup excluded)")
    p("")
    p("`∅` means an engine saw no character at that column. Marker's HTML-markup "
      "residue is excluded here so it does not swamp the glyph families.")
    p("")
    p("| chars in disagreement | n |")
    p("|---|---|")
    for pair, n in summary.get("top_confusion_pairs_ex_markup",
                               summary["top_confusion_pairs"])[:25]:
        p(f"| `{pair}` | {n:,} |")
    p("")
    p("## 5. The named glyph collisions")
    p("")
    p("| glyph A | glyph B | co-occurring | exact pair |")
    p("|---|---|---|---|")
    for a, b, n, exact in focus_rows:
        p(f"| `{a}` | `{b}` | {n:,} | {exact:,} |")
    p("")
    p("The task names `::` vs `:` and `<<` vs `<`. A doubled operator read as a "
      "single one does **not** appear as a one-column substitution: one copy still "
      "aligns, and the other becomes a deletion (`:|∅`). The table below counts those "
      "by asking whether a marker glyph sits next to an identical glyph (a run), and "
      "how many of those events drop a copy.")
    p("")
    p("| operator | glyph | events touching a run | events dropping a copy |")
    p("|---|---|---|---|")
    for op, g, nrun, dropped in doubled:
        p(f"| `{op}` | `{g}` | {nrun:,} | {dropped:,} |")
    p("")
    p("### 5.1 Examples of the named collisions")
    p("")
    glyph_events = [e for e in events if not e["in_markup"]]
    for a, b in FOCUS_GLYPHS:
        ex = [e for e in glyph_events
              if {a, b} <= set(e["pair"].split("|"))][:6]
        if not ex:
            continue
        p(f"**`{a}` / `{b}`** - "
          f"{sum(1 for e in glyph_events if {a, b} <= set(e['pair'].split('|'))):,} events")
        p("")
        p("| page | marker | tesseract | docTR | easyocr | verdict | context |")
        p("|---|---|---|---|---|---|---|")
        for e in ex:
            ctx = e["context"].replace("|", "\\|")
            if len(ctx) > 60:
                ctx = "…" + ctx[-60:]
            p(f"| {e['page_id']} | `{e['marker_char'] or '∅'}` | `{e['tesseract'] or '∅'}` | "
              f"`{e['doctr'] or '∅'}` | `{e['easyocr'] or '∅'}` | {e['verdict']} | {ctx} |")
        p("")
    worst = sorted(page_rows, key=lambda r: -r["pct_disagree_ex_markup"])[:15]
    p("## 6. Worst pages for agreement")
    p("")
    p("Ranked by disagreement **excluding markup**. A page whose disagreement was "
      "almost entirely markup (e.g. page 91's mailto anchors) drops down the list; "
      "`markup` shows how many of its events were marker's HTML residue.")
    p("")
    p("| page | columns | events | markup | changed | % disagree | % ex-markup |")
    p("|---|---|---|---|---|---|---|")
    for r in worst:
        p(f"| {r['page_id']} | {r['n_cols']:,} | {r['n_events']:,} | {r['n_markup']:,} | "
          f"{r['n_changed']:,} | {r['pct_disagree']}% | {r['pct_disagree_ex_markup']}% |")
    p("")
    gross = summary.get("gross_divergence_pages", [])
    p("## 6b. Gross length divergences (marker hallucination / engine failure)")
    p("")
    p("The character vote cannot fix a page where marker's *amount* of text is wrong "
      "(a recognition loop) or where an engine under-detected most of a page. Those "
      "show up as a large ratio of marker length to the mean engine length "
      "(`marker_over_engines`), so they are flagged separately.")
    p("")
    if gross:
        p("| page | marker chars | mean engine chars | marker/engines |")
        p("|---|---|---|---|")
        for pid, mlen, elen, ratio in gross[:20]:
            p(f"| {pid} | {mlen:,} | {elen:,} | {ratio} |")
        p("")
        p("Reading the outliers: **89, 208, 182** are recognition loops (below); "
          "**91** and **218** are marker's mailto-anchor markup (218 duplicates the "
          "anchor's visible text); **224** is the final near-blank page where marker "
          "read one stray mark (`o`) and the engines read different ones.")
        p("")
    else:
        p("No page diverges by more than 30% from the mean engine length.")
        p("")
    reps = summary.get("repetition_pages", [])
    if reps:
        p("Marker's recognition loops are the direct cause of most of the length "
          "divergence. Counting repeated word 4-grams in marker's own text "
          "(`max_word4gram_repeat`), these pages loop a phrase 10+ times:")
        p("")
        p("| page | max repeat | repeated phrase |")
        p("|---|---|---|")
        for pid, n, phrase in reps[:20]:
            p(f"| {pid} | {n} | `{phrase}` |")
        p("")
        p("Page 89 is the clearest case: marker repeats *\"of the second of\"* ~145 "
          "times (~2,000 surplus characters) while all three image engines read the "
          "real ~2,200-character page, so the vote deletes the loop from the "
          "consensus. This is the single most consequential correction the ensemble "
          "makes - and it means downstream statistics (#19/#24) inherited marker "
          "loops on these pages.")
        p("")
    p("## 7. Files")
    p("")
    p("| file | what it is |")
    p("|---|---|")
    p("| `consensus.tsv` / `consensus.md` | the voted transcript, one row/section per page |")
    p("| `marker_reference.tsv` | the marker reference text each page was aligned to |")
    p("| `disagreement_atlas.tsv` | one row per non-unanimous column, with page/block/line/char citation, all engine values, verdict, category and bbox |")
    p("| `consensus_changes.tsv` | the subset where the vote overrides marker (the actual corrections) |")
    p("| `confusion_pairs.tsv`, `category_counts.tsv`, `verdict_counts.tsv` | aggregate views |")
    p("| `engine_agreement.tsv`, `page_disagreement.tsv`, `page_length.tsv`, `focus_glyphs.tsv`, `doubled_operators.tsv` | per-engine / per-page / length-divergence / focused-glyph / doubled-operator tables |")
    p("| `engines/<engine>.jsonl` | the raw per-page OCR cache (text + word boxes) |")
    p("| `summary.json` | all headline numbers, machine-readable |")
    p("")
    p("## 8. Caveats")
    p("")
    p("1. **Absence is a vote but absence can be misalignment.** A column where the "
      "three independent engines are all `None` and marker has a character is read as "
      "a deletion; where they all agree on a character marker lacks, as an insertion. "
      "Three independent engines agreeing is strong evidence, but difflib can still "
      "place a boundary wrongly, so every event is citable (page + block + line + "
      "char offset + context) rather than auto-applied.")
    p("2. **Consensus is not a corrected corpus.** It is a vote; #6 remains the "
      "gazetteer-protected correction step. The consensus keeps marker wherever there "
      "is no strict majority.")
    p("3. **Engine quality differs.** Tesseract and docTR handle the operator glyphs "
      "very differently from marker; easyocr is weakest on `//` and `::`. Agreement "
      "between two weak engines is not automatically truth - the atlas exposes the "
      "disagreement so a human (or #6) can adjudicate.")
    p("4. **marker's own char bboxes are block-level in #3's dump**, so localisation "
      "here uses the word boxes of the *other* engines (tesseract preferred); a blank "
      "bbox means no independent engine had a box for that column (typically a "
      "deletion, where the other engines had no character to box).")
    p("5. **The consensus is whitespace-normalised.** Alignment collapses every "
      "whitespace run to one space, so `consensus.tsv` keeps word boundaries but not "
      "paragraph/line layout. Whitespace disagreements (a space or newline present in "
      "some engines and not others) are still recorded in the atlas as `whitespace`.")
    p("")
    with open(os.path.join(OUT, "ensemble_ocr_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("ocr")
    o.add_argument("--engines", default=",".join(DEFAULT_ENGINES))
    o.add_argument("--pdf", default=PDF)
    o.add_argument("--dpi", type=int, default=DPI)
    o.add_argument("--psm", type=int, default=3)
    o.add_argument("--page-range", default=None)
    o.add_argument("--force", action="store_true")
    o.set_defaults(func=cmd_ocr)
    b = sub.add_parser("build")
    b.add_argument("--dpi", type=int, default=DPI)
    b.add_argument("--psm", type=int, default=3)
    b.add_argument("--page-range", default=None)
    b.set_defaults(func=cmd_build)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
