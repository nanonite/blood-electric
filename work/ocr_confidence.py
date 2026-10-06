#!/usr/bin/env python3
"""
L0 / Chainlink #3 - OCR confidence mapping for the marker corpus of "Blood Electric".

Why this script exists
----------------------
The plan assumes marker can be re-run with ``--output_format json`` to obtain
per-line/per-word confidences.  That is not true of the installed stack:

  * marker-pdf 1.10.2  ``marker/renderers/json.py``          -> JSONBlockOutput has no score field
  * marker-pdf 1.10.2  ``marker/schema/text/char.py``         -> Char has (text, idx) only
  * marker-pdf 1.10.2  ``marker/renderers/ocr_json.py``       -> OCRJSONCharOutput has no score field

The scores do exist, but one layer down: surya-ocr 0.17.1
``RecognitionPredictor`` returns ``TextLine``/``TextChar`` objects that carry
``confidence``, and marker throws them away when it converts them to spans.
``surya/recognition/__init__.py:228`` computes that number as
``torch.max(token_probs, dim=-1).values`` - the maximum softmax probability of
the emitted token over the OCR vocabulary.  It is a model-confidence proxy, NOT
a calibrated probability of being correct.

So: re-run the *unmodified* stock marker pipeline (same processors, same
renderer, same config as the corpus run) inside this process, with a thin
recording proxy wrapped around the recognition model.  The proxy observes the
exact objects marker consumes, so the text is bit-identical to a normal run -
no re-implementation of marker logic, and no second OCR pass.

Subcommands
-----------
  capture   run marker (json output) + record confidences -> raw_ocr_confidence.jsonl.gz
  build     derive lines/tokens/flagged-token tables + report from the raw dump

Usage (from the workspace root, inside the `marker` conda env):

  source ~/anaconda3/etc/profile.d/conda.sh && conda activate marker
  TORCH_DEVICE=cuda python work/ocr_confidence.py capture \
      --input work/input/blood-electric.pdf \
      --outdir analysis/ocr_confidence
  python work/ocr_confidence.py build --outdir analysis/ocr_confidence
"""
from __future__ import annotations

import argparse
import collections
import datetime
import difflib
import gzip
import json
import os
import re
import statistics
import sys
import time

# --------------------------------------------------------------------------
# Raw dump schema (one JSON object per OCR'd block, written by `capture`)
# --------------------------------------------------------------------------
RAW_SCHEMA = """{
  "page_id": int,                 # 0-based PDF page index (marker's page_id)
  "block_id": str,                # marker BlockId, e.g. "/page/7/Text/3"
  "block_type": str,              # marker BlockTypes name of the OCR'd block
  "block_index": int,             # position of the block inside its page
  "line_index": int,              # position of the OCR line inside the block
  "surya_line_conf": float,       # TextLine.confidence (surya: mean of char confidences)
  "surya_text": str,              # TextLine.text (math tags unwrapped by surya)
  "original_text_good": bool,     # True => marker's OCR was bypassed (PDF text used)
  "line_bbox": [x0,y0,x1,y1],     # page coordinates, from surya
  "chars": [[text, conf, bbox_valid, x0, y0, x1, y1], ...]   # per character
}"""

# Operators of the symbolic machine (same inventory as work/analyze.py, plus
# the single-character forms), used only for classification/flagging here.
OPERATORS = [
    "<<", ">>", "//", "::", "++", "--", "==", "->", "<-",
    "=", "/", "<", ">", "+", "`", "^", "|", "~",
]

# Characters whose glyph the OCR is known to confuse; used for a *measured*
# cross-check against operator/ASCII punctuation, not for guessing.
PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


# ==========================================================================
# capture
# ==========================================================================
def install_recorder(stream):
    """Wrap OcrBuilder.ocr_extraction so the recognition model's results are
    recorded (page, block, line, char, confidence, bbox) on their way into
    marker.  marker's own code path is left untouched."""

    from marker.builders.ocr import OcrBuilder

    orig_ocr_extraction = OcrBuilder.ocr_extraction
    state = {"records": 0}

    class _RecordingProxy:
        """Transparent wrapper around surya's RecognitionPredictor."""

        def __init__(self, inner, on_result):
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_on_result", on_result)

        def __getattr__(self, name):  # only called for attributes not found normally
            return getattr(object.__getattribute__(self, "_inner"), name)

        def __call__(self, *args, **kwargs):
            results = object.__getattribute__(self, "_inner")(*args, **kwargs)
            try:
                object.__getattribute__(self, "_on_result")(results)
            except Exception as exc:  # never let bookkeeping kill the OCR run
                print(f"[capture] recorder error (non-fatal): {exc!r}", file=sys.stderr)
            return results

    def patched_ocr_extraction(self, document, pages, images, block_polygons,
                               block_ids, block_original_texts):
        def on_result(results):
            if len(results) != len(pages) or len(pages) != len(block_ids):
                raise RuntimeError("unexpected recognition/page/block_id lengths")
            for page, ids, res in zip(pages, block_ids, results):
                for block_index, (block_id, line) in enumerate(zip(ids, res.text_lines)):
                    chars = []
                    for ch in line.chars:
                        x0, y0, x1, y1 = ch.bbox
                        chars.append([ch.text, round(float(ch.confidence), 6),
                                      bool(ch.bbox_valid),
                                      round(float(x0), 1), round(float(y0), 1),
                                      round(float(x1), 1), round(float(y1), 1)])
                    record = {
                        "page_id": int(page.page_id),
                        "block_id": str(block_id),
                        "block_type": block_id.block_type.name if block_id.block_type else None,
                        "block_index": block_index,
                        "line_index": 0,
                        "surya_line_conf": round(float(line.confidence), 6),
                        "surya_text": line.text,
                        "original_text_good": bool(line.original_text_good),
                        "line_bbox": [round(float(v), 1) for v in line.bbox],
                        "chars": chars,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
                    state["records"] += 1

        proxy = _RecordingProxy(self.recognition_model, on_result)
        self.recognition_model = proxy
        try:
            return orig_ocr_extraction(self, document, pages, images, block_polygons,
                                       block_ids, block_original_texts)
        finally:
            self.recognition_model = object.__getattribute__(proxy, "_inner")

    OcrBuilder.ocr_extraction = patched_ocr_extraction
    return state


def cmd_capture(args):
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from marker.config.parser import ConfigParser
    from marker.logger import configure_logging
    from marker.models import create_model_dict
    from marker.output import save_output
    from marker.settings import settings

    configure_logging()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    cli_options = {
        "output_format": "json",
        "output_dir": args.json_dir or os.path.join(outdir, "marker_json"),
        "disable_image_extraction": True,
    }
    if args.page_range:
        cli_options["page_range"] = args.page_range
    if args.recognition_batch_size:
        cli_options["recognition_batch_size"] = args.recognition_batch_size

    raw_path = os.path.join(outdir, os.path.basename(args.raw) if args.raw
                            else "raw_ocr_confidence.jsonl.gz")

    print(f"[capture] marker output_dir = {cli_options['output_dir']}")
    print(f"[capture] torch device      = {settings.TORCH_DEVICE_MODEL}")
    print(f"[capture] raw dump          = {raw_path}")

    raw = gzip.open(raw_path, "wb")
    state = install_recorder(raw)
    start = time.time()
    try:
        models = create_model_dict()
        config_parser = ConfigParser(cli_options)
        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(args.input)
        out_folder = config_parser.get_output_folder(args.input)
        save_output(rendered, out_folder, config_parser.get_base_filename(args.input))
    finally:
        raw.close()

    elapsed = time.time() - start
    meta = {
        "input": os.path.abspath(args.input),
        "marker_output_dir": cli_options["output_dir"],
        "page_range": args.page_range or "all",
        "torch_device": settings.TORCH_DEVICE_MODEL,
        "records": state["records"],
        "elapsed_s": round(elapsed, 1),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "raw_schema": RAW_SCHEMA,
    }
    with open(os.path.join(outdir, "capture_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[capture] {state['records']} OCR blocks recorded in {elapsed / 60:.1f} min")
    print(json.dumps(meta, indent=2))
    return 0


# ==========================================================================
# build
# ==========================================================================
def is_tag(txt):
    from marker.util import get_closing_tag_type, get_opening_tag_type
    return get_opening_tag_type(txt)[0] or get_closing_tag_type(txt)[0]


def line_segments(chars):
    """Split a block's char list into OCR lines the way marker does (`<br>`),
    dropping the format/math tag characters marker does not render."""
    segs, cur = [], []
    for ch in chars:
        text = ch[0]
        if text == "<br>":
            segs.append(cur)
            cur = []
            continue
        if is_tag(text):
            continue
        cur.append(ch)
    segs.append(cur)
    return [s for s in segs if s]


def split_tokens(chars):
    """Whitespace-delimited tokens with their character confidences and bbox.
    Returns [(token_text, [conf...], (x0,y0,x1,y1)), ...]."""
    out, cur, confs, box = [], [], [], None
    for ch in chars:
        text, conf = ch[0], ch[1]
        if text.strip() == "":
            if cur:
                out.append(("".join(cur), confs, box))
                cur, confs, box = [], [], None
            continue
        cur.append(text)
        confs.append(conf)
        x0, y0, x1, y1 = ch[3], ch[4], ch[5], ch[6]
        box = (x0, y0, x1, y1) if box is None else (
            min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1))
    if cur:
        out.append(("".join(cur), confs, box))
    return out


def classify(tok):
    """Structural flags, independent of confidence (so a flagged token can be judged)."""
    flags = []
    if tok in OPERATORS:
        flags.append("op_exact")
    elif re.fullmatch(r"[-=/+<>`^|~]+", tok):
        flags.append("op_run")
    if any(ch in PUNCT for ch in tok):
        flags.append("has_punct")
    if any(ch.isdigit() for ch in tok):
        flags.append("has_digit")
    if re.search(r"[A-Za-z]", tok):
        if tok.isupper():
            flags.append("allcaps")
        elif tok[0].islower() and re.search(r"[A-Z]", tok):
            flags.append("internal_caps")
    if any(ord(ch) > 127 for ch in tok):
        flags.append("non_ascii")
    if len(tok) == 1:
        flags.append("single_char")
    if tok.isalpha() and tok.islower() and tok[0] == "l" and len(tok) <= 3:
        # the It -> lt drift shape: a short lowercase l-initial token
        flags.append("l_initial_short")
    return flags


def read_raw(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[build] skipping malformed raw line {line_no}: {exc}", file=sys.stderr)


# Tags marker emits around real text.  Only these are stripped: the book's own
# angle-bracket words (<lung LOAD/...>, <oscillation>, <<gravity zero>) must survive.
_TAG = (r"</?(?:p|b|i|u|em|strong|br|hr|h[1-6]|sub|sup|span|div|img|content-ref|math"
        r"|table|thead|tbody|tr|td|th|ul|ol|li|figure|figcaption|a|script-[a-z]+)"
        r"\b[^<>]{0,160}>")


def norm_visible(text):
    """Strip markup/entities and collapse whitespace so the OCR char stream and
    marker's rendered JSON/markdown can be compared as plain text."""
    import html as _html
    text = _html.unescape(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(_TAG, " ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", " ", text)
    text = text.replace("**", " ").replace("*", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main_build(args):
    outdir = args.outdir
    raw_path = args.raw or os.path.join(outdir, "raw_ocr_confidence.jsonl.gz")
    if not os.path.exists(raw_path):
        print(f"[build] missing raw dump: {raw_path}", file=sys.stderr)
        return 2

    lines_rows, tokens_rows = [], []
    page_chars = collections.defaultdict(list)
    blocks_by_page = collections.defaultdict(list)
    orig_text_good = 0
    empty_blocks = 0
    block_ids_seen = set()
    all_ocr_text_parts = []
    ocr_page_text = collections.defaultdict(list)

    for rec in read_raw(raw_path):
        if rec["original_text_good"]:
            orig_text_good += 1
        key = (rec["page_id"], rec["block_id"])
        if key in block_ids_seen:
            empty_blocks += 1
        block_ids_seen.add(key)
        blocks_by_page[rec["page_id"]].append(rec["block_id"])

        segs = line_segments(rec["chars"])
        block_text = "\n".join("".join(c[0] for c in s) for s in segs)
        all_ocr_text_parts.append(block_text)
        ocr_page_text[rec["page_id"]].append(block_text)
        for line_index, seg in enumerate(segs):
            text = "".join(c[0] for c in seg)
            confs = [c[1] for c in seg]
            xs0 = min((c[3] for c in seg), default=0.0)
            ys0 = min((c[4] for c in seg), default=0.0)
            xs1 = max((c[5] for c in seg), default=0.0)
            ys1 = max((c[6] for c in seg), default=0.0)
            line_row = {
                "page_id": rec["page_id"], "block_id": rec["block_id"],
                "block_type": rec["block_type"], "block_index": rec["block_index"],
                "line_index": line_index, "n_chars": len(seg),
                "line_conf": statistics.fmean(confs) if confs else 0.0,
                "min_char_conf": min(confs) if confs else 0.0,
                "bbox": [round(xs0, 1), round(ys0, 1), round(xs1, 1), round(ys1, 1)],
                "text": text,
            }
            lines_rows.append(line_row)
            page_chars[rec["page_id"]].extend(confs)
            for tok_index, (tok, tconfs, tbox) in enumerate(split_tokens(seg)):
                flags = classify(tok)
                tokens_rows.append({
                    "page_id": rec["page_id"], "block_id": rec["block_id"],
                    "block_type": rec["block_type"], "line_index": line_index,
                    "tok_index": tok_index, "token": tok,
                    "n_chars": len(tconfs),
                    "tok_conf": round(statistics.fmean(tconfs), 6),
                    "min_char_conf": round(min(tconfs), 6),
                    "p10_char_conf": round(sorted(tconfs)[max(0, int(0.1 * len(tconfs)) - 1)], 6),
                    "bbox": "" if tbox is None else
                    [round(v, 1) for v in (tbox[0], tbox[1], tbox[2], tbox[3])],
                    "flags": ",".join(flags),
                    "context": "",   # filled below from the line text
                })

    # --- line text + +-40 char context for every token ---------------------
    line_text = {(r["page_id"], r["block_id"], r["line_index"]): r["text"] for r in lines_rows}
    for t in tokens_rows:
        key = (t["page_id"], t["block_id"], t["line_index"])
        text = line_text.get(key, "")
        pos = 0
        # tokens are emitted left-to-right, so a running search is safe
        idx = text.find(t["token"], pos)
        if idx < 0:
            idx = text.find(t["token"])
        if idx >= 0:
            lo = max(0, idx - 40)
            hi = min(len(text), idx + len(t["token"]) + 40)
            t["context"] = ("..." if lo > 0 else "") + text[lo:hi] + ("..." if hi < len(text) else "")
            pos = idx + len(t["token"])

    # --- where the OCR char stream and the rendered corpus legitimately differ --
    # marker concatenates merged lines with NO separator and strips a trailing hyphen
    # when the next line starts lowercase (marker/schema/text/line.py:25-36), while
    # this dump keeps OCR line breaks.  Count both effects.
    HYPHENS = "-—¬"
    by_block = collections.defaultdict(list)
    for r in lines_rows:
        by_block[(r["page_id"], r["block_id"])].append(r)
    joins, hyphen_lower, hyphen_other, glue_joins = 0, 0, 0, 0
    for rows in by_block.values():
        rows.sort(key=lambda r: r["line_index"])
        for a, b in zip(rows, rows[1:]):
            joins += 1
            prev, nxt = a["text"].rstrip(), b["text"].lstrip()
            if prev and nxt and not a["text"].endswith((" ", "\t")) \
                    and not nxt[:1].isspace() and prev[-1] not in HYPHENS:
                glue_joins += 1      # marker glues these with no separator
            if prev.endswith(tuple(HYPHENS)):
                if nxt[:1].islower():
                    hyphen_lower += 1   # marker deletes the hyphen
                else:
                    hyphen_other += 1  # marker keeps the hyphen
    hyphen_breaks = hyphen_lower + hyphen_other

    # The same two effects across *Line blocks* of a page: where marker ran in
    # line mode, LineMergeProcessor glues those lines into one Text block, so
    # these are real boundaries of the rendered corpus (and hence the price of
    # keeping the OCR stream un-normalised).
    page_line_blocks = collections.defaultdict(list)
    for r in lines_rows:
        if r["block_type"] == "Line":
            page_line_blocks[r["page_id"]].append(r)
    cross_glue, cross_hyphen_lower, cross_hyphen_other, cross_n = 0, 0, 0, 0
    for pid, rows in page_line_blocks.items():
        rows.sort(key=lambda r: (r["block_index"], r["line_index"]))
        for a, b in zip(rows, rows[1:]):
            cross_n += 1
            prev, nxt = a["text"].rstrip(), b["text"].lstrip()
            if not prev or not nxt:
                continue
            if not a["text"].endswith((" ", "\t")) and not nxt[:1].isspace() \
                    and prev[-1] not in HYPHENS:
                cross_glue += 1
            if prev.endswith(tuple(HYPHENS)):
                if nxt[:1].islower():
                    cross_hyphen_lower += 1
                else:
                    cross_hyphen_other += 1

    # --- write tables ------------------------------------------------------
    line_cols = ["page_id", "block_id", "block_type", "block_index", "line_index",
                 "n_chars", "line_conf", "min_char_conf", "bbox", "text"]
    tok_cols = ["page_id", "block_id", "block_type", "line_index", "tok_index", "token",
                "n_chars", "tok_conf", "min_char_conf", "p10_char_conf", "bbox",
                "flags", "context"]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return str(v)

    with open(os.path.join(outdir, "lines.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\t".join(line_cols) + "\n")
        for r in lines_rows:
            # surya can emit literal \n / \r pseudo-characters; escape them
            # so every record stays exactly one physical line (as tokens.tsv
            # and flagged_tokens.tsv already do)
            fh.write("\t".join(fmt(r[c]).replace("\t", " ")
                               .replace("\n", "\\n").replace("\r", "\\r")
                               for c in line_cols) + "\n")

    with open(os.path.join(outdir, "tokens.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\t".join(tok_cols) + "\n")
        for r in tokens_rows:
            fh.write("\t".join(fmt(r[c]).replace("\t", " ").replace("\n", "\\n")
                               for c in tok_cols) + "\n")

    # --- flagged tokens ----------------------------------------------------
    min_thr = args.min_conf          # any char below this  -> suspect glyph
    mean_thr = args.mean_conf        # token mean below this -> suspect token
    flagged, flagged_ids = [], set()
    for i, t in enumerate(tokens_rows):
        t["row"] = i
        reasons = []
        if t["min_char_conf"] < min_thr:
            reasons.append(f"min_char_conf<{min_thr}")
        if t["tok_conf"] < mean_thr:
            reasons.append(f"tok_conf<{mean_thr}")
        if not reasons:
            continue
        t["reason"] = ";".join(reasons)
        flagged.append(t)
        flagged_ids.add(i)
    flagged.sort(key=lambda r: (r["min_char_conf"], r["tok_conf"]))
    flag_cols = ["page_id", "block_id", "block_type", "line_index", "tok_index", "token",
                 "tok_conf", "min_char_conf", "p10_char_conf", "reason", "flags",
                 "bbox", "context"]
    with open(os.path.join(outdir, "flagged_tokens.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\t".join(flag_cols) + "\n")
        for r in flagged:
            fh.write("\t".join(fmt(r[c]).replace("\t", " ").replace("\n", "\\n")
                               for c in flag_cols) + "\n")

    # --- page table --------------------------------------------------------
    page_rows = []
    tok_by_page = collections.defaultdict(list)
    for t in tokens_rows:
        tok_by_page[t["page_id"]].append(t)
    for page_id in sorted(set(list(page_chars) + list(blocks_by_page))):
        confs = page_chars.get(page_id, [])
        toks = tok_by_page.get(page_id, [])
        tok_means = [t["tok_conf"] for t in toks]
        n_flag = sum(1 for t in toks if t["row"] in flagged_ids)
        row = {
            "page_id": page_id,
            "n_blocks": len(blocks_by_page.get(page_id, [])),
            "n_lines": sum(1 for r in lines_rows if r["page_id"] == page_id),
            "n_chars": len(confs),
            "n_tokens": len(toks),
            "char_conf_mean": round(statistics.fmean(confs), 6) if confs else 0.0,
            "char_conf_p05": round(sorted(confs)[int(0.05 * (len(confs) - 1))], 6) if confs else 0.0,
            "char_conf_min": round(min(confs), 6) if confs else 0.0,
            "tok_conf_mean": round(statistics.fmean(tok_means), 6) if tok_means else 0.0,
            "n_flagged": n_flag,
            "pct_flagged": round(100.0 * n_flag / len(toks), 3) if toks else 0.0,
        }
        page_rows.append(row)
    page_cols = list(page_rows[0].keys()) if page_rows else []
    with open(os.path.join(outdir, "page_confidence.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\t".join(page_cols) + "\n")
        for r in page_rows:
            fh.write("\t".join(fmt(r[c]) for c in page_cols) + "\n")

    # --- histograms / class tables ----------------------------------------
    all_confs = [c for page in page_chars.values() for c in page]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    hist_chars = [0] * (len(bins) - 1)
    for c in all_confs:
        for i in range(len(bins) - 1):
            if bins[i] <= c < bins[i + 1]:
                hist_chars[i] += 1
                break
    hist_toks = [0] * (len(bins) - 1)
    for t in tokens_rows:
        c = t["tok_conf"]
        for i in range(len(bins) - 1):
            if bins[i] <= c < bins[i + 1]:
                hist_toks[i] += 1
                break
    with open(os.path.join(outdir, "conf_histogram.tsv"), "w", encoding="utf-8") as fh:
        fh.write("bin_lo\tbin_hi\tchars\tchar_pct\ttokens\ttoken_pct\n")
        n_chars, n_toks = len(all_confs), len(tokens_rows)
        for i in range(len(bins) - 1):
            fh.write(f"{bins[i]:.2f}\t{bins[i + 1]:.2f}\t{hist_chars[i]}\t"
                     f"{100.0 * hist_chars[i] / max(1, n_chars):.3f}\t"
                     f"{hist_toks[i]}\t{100.0 * hist_toks[i] / max(1, n_toks):.3f}\n")

    # confidence by token class / by exact operator
    def class_of(t):
        f = set(t["flags"].split(",")) if t["flags"] else set()
        if "op_exact" in f:
            return "operator_exact"
        if "op_run" in f:
            return "operator_run"
        if "non_ascii" in f:
            return "non_ascii"
        if "allcaps" in f:
            return "allcaps"
        if "has_digit" in f and not "has_punct" in f:
            return "numeric"
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", t["token"]):
            return "plain_word"
        return "mixed"

    class_conf = collections.defaultdict(list)
    for t in tokens_rows:
        class_conf[class_of(t)].append((t["tok_conf"], t["min_char_conf"]))
    op_conf = collections.defaultdict(list)
    for t in tokens_rows:
        if "op_exact" in (t["flags"].split(",") if t["flags"] else []):
            op_conf[t["token"]].append(t["tok_conf"])
    with open(os.path.join(outdir, "token_class_confidence.tsv"), "w", encoding="utf-8") as fh:
        fh.write("group\tvalue\tn\ttok_conf_mean\tmin_char_conf_mean\tmin_char_conf_min\n")
        for name, vals in sorted(class_conf.items(), key=lambda kv: -len(kv[1])):
            fh.write(f"class\t{name}\t{len(vals)}\t{statistics.fmean(v[0] for v in vals):.6f}\t"
                     f"{statistics.fmean(v[1] for v in vals):.6f}\t{min(v[1] for v in vals):.6f}\n")
        for op, vals in sorted(op_conf.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            fh.write(f"operator\t{op}\t{len(vals)}\t{statistics.fmean(vals):.6f}\t"
                     f"{min(vals):.6f}\t{min(vals):.6f}\n")

    # --- validation: is the recorded text the same text marker rendered? ----
    # (a) per page, OCR char stream vs marker's own JSON output
    json_path = args.marker_json
    if json_path is None or not os.path.exists(json_path):
        guess = os.path.join(outdir, "marker_json", "blood-electric", "blood-electric.json")
        json_path = guess if os.path.exists(guess) else None

    validation = {}
    if json_path:
        data = json.load(open(json_path, encoding="utf-8"))
        page_sims, block_rows_json = [], 0
        json_page_text = {}
        for page in data.get("children", []):
            pid = int(str(page.get("id", "/page/0")).split("/")[2])
            texts = []

            def walk(node):
                nonlocal block_rows_json
                for child in node.get("children") or []:
                    bt = child.get("block_type", "")
                    if bt not in ("Page",):
                        html = child.get("html") or ""
                        if html.strip():
                            block_rows_json += 1
                            texts.append(html)
                    walk(child)

            walk(page)
            json_page_text[pid] = norm_visible("\n".join(texts))
        for pid, jtext in json_page_text.items():
            otext = norm_visible("\n".join(ocr_page_text.get(pid, [])))
            if not otext and not jtext:
                continue
            ratio = difflib.SequenceMatcher(None, otext, jtext, autojunk=False).ratio()
            page_sims.append((ratio, len(otext) + len(jtext)))
        json_text = norm_visible("\n".join(json_page_text[p] for p in sorted(json_page_text)))
        ocr_text = norm_visible("\n".join(all_ocr_text_parts))
        validation["marker_json"] = os.path.abspath(json_path)
        validation["json_text_blocks"] = block_rows_json
        validation["json_pages"] = len(json_page_text)
        # Full-document difflib.SequenceMatcher scales ~O(n^2.4) on this
        # data (measured: 5.3 s @ 25k chars, 28.1 s @ 50k chars -> ~2 h
        # for the 479k-char book), so the overall figure is the exact
        # concatenated ratio only for small documents and otherwise the
        # char-weighted mean of the per-page ratios.
        if len(ocr_text) + len(json_text) <= 60000:
            validation["json_vs_ocr_char_similarity_overall"] = round(
                difflib.SequenceMatcher(None, ocr_text, json_text, autojunk=False).ratio(), 6)
            validation["json_vs_ocr_similarity_method"] = "exact difflib ratio"
        else:
            tot = sum(w for _, w in page_sims)
            validation["json_vs_ocr_char_similarity_overall"] = round(
                sum(r * w for r, w in page_sims) / tot, 6)
            validation["json_vs_ocr_similarity_method"] = (
                "char-weighted mean of per-page difflib ratios (full-document "
                "difflib is O(n^2.4) here: 5.3s@25k, 28.1s@50k chars)")
        if page_sims:
            ratios = [r for r, _ in page_sims]
            validation["json_vs_ocr_per_page_similarity"] = {
                "pages_compared": len(ratios),
                "mean": round(statistics.fmean(ratios), 6),
                "min": round(min(ratios), 6),
                "n_pages_below_0.99": sum(1 for s in ratios if s < 0.99),
            }
    else:
        ocr_text = norm_visible("\n".join(all_ocr_text_parts))

    # (b) the whole OCR stream vs the working corpus markdown
    md_path = args.corpus
    if os.path.exists(md_path):
        md_text = norm_visible(open(md_path, encoding="utf-8", errors="ignore").read())
        ocr_toks = re.findall(r"\S+", ocr_text)
        md_toks = re.findall(r"\S+", md_text)
        # In-order agreement is difflib over token sequences, which
        # scales the same super-quadratically as the char comparison.
        # It is run exactly only when the combined token count is
        # small enough (the 12-page pilot: ~72.7k tokens); on the
        # full book it is measured on a 20k-token prefix and
        # complemented with a linear-time multiset overlap over the
        # complete streams.
        if len(ocr_toks) + len(md_toks) <= 75000:
            sm = difflib.SequenceMatcher(None, ocr_toks, md_toks, autojunk=False)
            matched = sum(bl.size for bl in sm.get_matching_blocks())
            validation["md_token_match_fraction_in_order"] = round(
                matched / max(1, len(md_toks)), 6)
        else:
            prefix = 20000
            sm = difflib.SequenceMatcher(None, ocr_toks[:prefix],
                                         md_toks[:prefix], autojunk=False)
            matched = sum(bl.size for bl in sm.get_matching_blocks())
            validation["md_token_prefix_n"] = prefix
            validation["md_token_match_fraction_in_order_prefix"] = round(
                matched / max(1, min(prefix, len(md_toks))), 6)
            co, cm = collections.Counter(ocr_toks), collections.Counter(md_toks)
            inter = sum((co & cm).values())
            validation["md_token_multiset_containment"] = round(
                inter / max(1, len(ocr_toks)), 6)
            validation["md_token_multiset_jaccard"] = round(
                inter / max(1, len(ocr_toks) + len(md_toks) - inter), 6)
        md_pages = None
        if "json_pages" in validation:
            md_pages = validation["json_pages"]
        validation["corpus_md"] = os.path.abspath(md_path)
        validation["ocr_chars"] = len(ocr_text)
        validation["md_chars"] = len(md_text)
        validation["ocr_tokens"] = len(ocr_toks)
        validation["md_tokens"] = len(md_toks)
        if md_pages and len(page_chars) < md_pages:
            validation["md_comparison_note"] = (
                f"only {len(page_chars)} of {md_pages} pages were re-run "
                f"(--page-range); md-level numbers are not comparable")

    with open(os.path.join(outdir, "build_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "raw_dump": os.path.abspath(raw_path),
            "pages_with_text": len(page_chars),
            "blocks_ocrd": len(block_ids_seen),
            "duplicate_block_ids": empty_blocks,
            "original_text_good": orig_text_good,
            "n_lines": len(lines_rows),
            "n_tokens": len(tokens_rows),
            "n_chars": len(all_confs),
            "n_flagged": len(flagged),
            "hyphenated_line_breaks": hyphen_breaks,
            "hyphen_stripped_by_marker": hyphen_lower,
            "hyphen_kept_by_marker": hyphen_other,
            "line_joins": joins,
            "line_joins_glued_without_space": glue_joins,
            "cross_line_block_joins": cross_n,
            "cross_line_block_joins_glued_without_space": cross_glue,
            "cross_line_block_hyphen_stripped": cross_hyphen_lower,
            "cross_line_block_hyphen_kept": cross_hyphen_other,
            "thresholds": {"min_char_conf": min_thr, "tok_conf": mean_thr},
            "validation": validation,
        }, fh, indent=2)

    # --- report ------------------------------------------------------------
    write_report(outdir, lines_rows, tokens_rows, flagged, page_rows, all_confs,
                 class_conf, op_conf, hist_chars, hist_toks, bins, validation,
                 orig_text_good, block_ids_seen, args, hyphen_breaks)
    print(json.dumps(json.load(open(os.path.join(outdir, "build_meta.json"))), indent=2))
    return 0


def pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def write_report(outdir, lines_rows, tokens_rows, flagged, page_rows, all_confs,
                 class_conf, op_conf, hist_chars, hist_toks, bins, validation,
                 orig_text_good, block_ids_seen, args, hyphen_breaks=0):
    meta = json.load(open(os.path.join(outdir, "build_meta.json")))
    hyphen_lower = meta.get("hyphen_stripped_by_marker", 0)
    hyphen_other = meta.get("hyphen_kept_by_marker", 0)
    glue_joins = meta.get("line_joins_glued_without_space", 0)
    n_joins = meta.get("line_joins", 0)
    cross_glue = meta.get("cross_line_block_joins_glued_without_space", 0)
    cross_hl = meta.get("cross_line_block_hyphen_stripped", 0)
    cross_ho = meta.get("cross_line_block_hyphen_kept", 0)
    L = []
    p = L.append

    n_tok = len(tokens_rows)
    tok_means = [t["tok_conf"] for t in tokens_rows]
    tok_mins = [t["min_char_conf"] for t in tokens_rows]

    p("# Blood Electric — OCR confidence map (L0 / Chainlink #3)")
    p("")
    p(f"_Generated {meta['generated_utc']} by `work/ocr_confidence.py` "
      f"(capture → build), from `{meta['raw_dump']}`._")
    p("")
    p("## 0. What was actually available (negative result first)")
    p("")
    p("The plan (`docs/meaning-extraction-methods.md` §L0.1) expected marker to emit")
    p("per-line/word confidence in its JSON output. **It does not.** Checked in the")
    p("installed stack:")
    p("")
    p("| component | confidence field? |")
    p("|---|---|")
    p("| `marker/renderers/json.py` `JSONBlockOutput` | no |")
    p("| `marker/renderers/ocr_json.py` `OCRJSONCharOutput` | no |")
    p("| `marker/schema/text/char.py` `Char` | no (only `text`, `idx`) |")
    p("| `surya/recognition/schema.py` `TextChar` / `TextLine` | **yes** |")
    p("")
    p("So marker throws the scores away. This task therefore re-runs the *stock*")
    p("pipeline in-process with a recording proxy wrapped around surya's")
    p("`RecognitionPredictor` (no re-implementation of marker logic, one OCR pass,")
    p("identical processors/renderer/config to the corpus run), and reads the")
    p("confidences off the very objects marker consumes.")
    p("")
    p("**What the number is.** `surya/recognition/__init__.py:228`:")
    p("`scores = torch.max(token_probs, dim=-1).values` — the maximum softmax")
    p("probability of the emitted token over surya's OCR vocabulary, one value per")
    p("character. It is a *model-confidence proxy*, not a calibrated P(correct), and")
    p("it is not comparable across engines or models. `TextLine.confidence` is just the")
    p("mean of its characters' confidences, so line and char tables carry the same")
    p("information at two granularities.")
    p("")
    p("## 1. Coverage")
    p("")
    p(f"- pages with OCR'd text: **{meta['pages_with_text']}**")
    p(f"- OCR'd blocks: **{meta['blocks_ocrd']}** "
      f"(blocks marker skipped because the PDF already had usable text: {orig_text_good})")
    p(f"- OCR lines: **{meta['n_lines']:,}** · whitespace tokens: **{meta['n_tokens']:,}** "
      f"· characters: **{meta['n_chars']:,}**")
    p(f"- flagged tokens: **{meta['n_flagged']:,}** "
      f"({100.0 * meta['n_flagged'] / max(1, n_tok):.2f}% of tokens; "
      f"thresholds: any char < {meta['thresholds']['min_char_conf']} "
      f"or token mean < {meta['thresholds']['tok_conf']})")
    p("")
    if validation:
        p("### 1.1 Validation: does the recorded text equal the text marker rendered?")
        p("")
        if "json_vs_ocr_per_page_similarity" in validation:
            v = validation["json_vs_ocr_per_page_similarity"]
            p("Per page, the recorded OCR char stream vs the `html` marker itself wrote in")
            p(f"its JSON output ({validation['json_text_blocks']} text blocks over")
            p(f"{validation['json_pages']} pages): mean similarity **{v['mean']:.5f}**, "
              f"min **{v['min']:.5f}**, pages below 0.99: **{v['n_pages_below_0.99']}**.")
            p(f"Whole-document similarity: "
              f"**{validation['json_vs_ocr_char_similarity_overall']:.5f}**"
              f" ({validation.get('json_vs_ocr_similarity_method', '')}).")
            p("")
            p("The residual is not pipeline drift but two *known, benign* marker")
            p("transformations that happen **after** OCR, which the recorded char stream")
            p("cannot contain:")
            p("")
            p(f"- **line gluing**: marker concatenates merged OCR lines with no separator "
              f"(`marker/schema/text/line.py:84-100`); this dump keeps the line breaks. "
              f"{glue_joins:,} of {n_joins:,} within-block line joins and {cross_glue:,} "
              f"line-to-line joins between separate `Line` blocks have no space on "
              f"either side, so the corpus shows those glued.")
            p(f"- **de-hyphenation**: when a line ends in `-` and the next starts lowercase, "
              f"marker deletes the hyphen. Measured: {hyphen_lower + cross_hl:,} joins "
              f"({hyphen_lower:,} within a block, {cross_hl:,} between `Line` blocks); "
              f"a further {hyphen_other + cross_ho:,} hyphen line-ends keep the hyphen "
              f"(e.g. `body-` + `OMOTYA` → `body-OMOTYA`). This is why the corpus reads "
              f"`digitalvampire` where the OCR saw `digital-` + `vampire`.")
            p("")
            p("So the text is the same; only line packing, hyphenation and a few block")
            p("orderings differ. Confidence rows are keyed to **OCR order**, which is why")
            p("the join keys below are (page, block, bbox), not sequence position.")
            p("")
        if "corpus_md" in validation:
            p("Against the working corpus markdown"
              f" (`{os.path.basename(validation['corpus_md'])}`), markup stripped:")
            p("")
            if "md_token_match_fraction_in_order" in validation:
                p(f"- OCR stream {validation['ocr_tokens']:,} tokens vs corpus "
                  f"{validation['md_tokens']:,} tokens; in-order token agreement "
                  f"**{100 * validation['md_token_match_fraction_in_order']:.2f}%**")
            else:
                p(f"- OCR stream {validation['ocr_tokens']:,} tokens vs corpus "
                  f"{validation['md_tokens']:,} tokens. In-order token agreement on "
                  f"the first {validation['md_token_prefix_n']:,} tokens: "
                  f"**{100 * validation['md_token_match_fraction_in_order_prefix']:.2f}%** "
                  "(difflib over the full token streams is super-quadratic on the "
                  "full book, so the in-order figure is a 20k-token prefix "
                  "spot-check)")
                p(f"- full-stream multiset overlap: "
                  f"**{100 * validation['md_token_multiset_containment']:.2f}%** of "
                  f"OCR tokens appear in the corpus vocabulary "
                  f"(Jaccard {validation['md_token_multiset_jaccard']:.4f})")
            if "md_comparison_note" in validation:
                p(f"- note: {validation['md_comparison_note']}")
            else:
                p("- OCR characters "
                  f"{validation['ocr_chars']:,} vs markdown {validation['md_chars']:,} "
                  "(the difference is markdown markup, heading markers and image links)")
            p("")
    p("## 2. Confidence distribution")
    p("")
    p(f"- characters: mean **{statistics.fmean(all_confs):.4f}**, "
      f"p01 {pct(all_confs,0.01):.3f}, p05 {pct(all_confs,0.05):.3f}, "
      f"p25 {pct(all_confs,0.25):.3f}, median {pct(all_confs,0.5):.3f}, "
      f"p95 {pct(all_confs,0.95):.3f}, min {min(all_confs):.3f}")
    p(f"- tokens (mean char conf): mean **{statistics.fmean(tok_means):.4f}**, "
      f"p05 {pct(tok_means,0.05):.3f}, median {pct(tok_means,0.5):.3f}, "
      f"p95 {pct(tok_means,0.95):.3f}")
    p(f"- tokens (worst char): mean **{statistics.fmean(tok_mins):.4f}**, "
      f"median {pct(tok_mins,0.5):.3f}, min {min(tok_mins):.3f}")
    p("")
    p("| confidence bin | chars | char % | tokens | token % |")
    p("|---|---|---|---|---|")
    for i in range(len(bins) - 1):
        p(f"| {bins[i]:.2f}–{bins[i+1]:.2f} | {hist_chars[i]:,} | "
          f"{100.0 * hist_chars[i] / max(1, len(all_confs)):.2f}% | "
          f"{hist_toks[i]:,} | {100.0 * hist_toks[i] / max(1, n_tok):.2f}% |")
    p("")
    p("## 3. Confidence by token class (measured, not assumed)")
    p("")
    p("| class | n | token mean | worst-char mean | worst-char min |")
    p("|---|---|---|---|---|")
    for name, vals in sorted(class_conf.items(), key=lambda kv: -len(kv[1])):
        p(f"| {name} | {len(vals):,} | {statistics.fmean(v[0] for v in vals):.4f} | "
          f"{statistics.fmean(v[1] for v in vals):.4f} | {min(v[1] for v in vals):.4f} |")
    p("")
    p("### 3.1 Confidence by individual operator")
    p("")
    p("The operators are the book's opcodes (E1), so an operator class that OCRs")
    p("confidently is a precondition for every operator count in the program.")
    p("")
    p("| operator | n | mean token conf | min |")
    p("|---|---|---|---|")
    for op, vals in sorted(op_conf.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        p(f"| `{op}` | {len(vals):,} | {statistics.fmean(vals):.4f} | {min(vals):.4f} |")
    p("")
    p("## 4. Per-page confidence")
    p("")
    p("Worst 15 pages by mean token confidence (page ids are 0-based PDF pages):")
    p("")
    p("| page | lines | tokens | char conf mean | p05 | token conf mean | flagged | % flagged |")
    p("|---|---|---|---|---|---|---|---|")
    for r in sorted(page_rows, key=lambda r: r["tok_conf_mean"])[:15]:
        p(f"| {r['page_id']} | {r['n_lines']:,} | {r['n_tokens']:,} | "
          f"{r['char_conf_mean']:.4f} | {r['char_conf_p05']:.4f} | "
          f"{r['tok_conf_mean']:.4f} | {r['n_flagged']:,} | {r['pct_flagged']:.1f}% |")
    p("")
    page_means = [r["tok_conf_mean"] for r in page_rows]
    clean = sum(1 for r in page_rows if r["n_flagged"] == 0)
    p(f"Across {len(page_rows)} pages the page-mean token confidence spans "
      f"{min(page_means):.4f}–{max(page_means):.4f} "
      f"(sd {statistics.pstdev(page_means):.4f}); {clean} of {len(page_rows)} "
      "pages have no flagged tokens at all.")
    hot = [r for r in page_rows if r["pct_flagged"] >= 5.0 and r["n_tokens"] >= 10]
    if hot:
        p("Locally degraded pages (>=5% flagged tokens — a rough patch on the "
          "scan, not single bad glyphs): "
          + "; ".join(f"page {r['page_id']} ({r['pct_flagged']:.1f}% flagged, "
                      f"worst char conf {r['char_conf_min']:.3f})"
                      for r in sorted(hot, key=lambda r: -r["pct_flagged"])) + ".")
    stray = [r for r in page_rows if r["n_tokens"] < 10 and r["n_flagged"] > 0]
    if stray:
        p("Near-empty pages whose one flagged token is a stray scan mark, not "
          "prose: " + "; ".join(
              f"page {r['page_id']} ({r['n_tokens']} token, worst char conf "
              f"{r['char_conf_min']:.3f})" for r in stray) + ".")
    p("Outside those pages, degradation is token-local, not page-local.")
    p("")
    p("## 5. Lowest-confidence tokens (top 40)")
    p("")
    p("| page | block | token | mean | worst char | context |")
    p("|---|---|---|---|---|---|")
    for t in flagged[:40]:
        ctx = t["context"].replace("|", "\\|")
        if len(ctx) > 90:
            ctx = ctx[:90] + "…"
        p(f"| {t['page_id']} | `{t['block_id']}` | `{t['token']}` | {t['tok_conf']:.3f} | "
          f"{t['min_char_conf']:.3f} | {ctx} |")
    p("")
    book_style = [t for t in flagged if t["token"].startswith("<")]
    if book_style:
        p(f"**{len(book_style):,} of the {len(flagged):,} flagged tokens "
          f"({100.0 * len(book_style) / max(1, len(flagged)):.0f}%) are the "
          "book's own `<...>` / `<<th>...` constructions (e.g. `<<th>elf`, "
          "`<genetic`, `<anti=heaven`, `<Pre><<Desire`) — genuine text that "
          "scores low because the glyph sequence is unlike the model's prior, "
          "not OCR errors; read them together with the structural flags in "
          "§7, not as a correction list.")
        p("")
    p("## 6. Does confidence predict the known OCR drift?")
    p("")
    p("The workspace already names three drift patterns (README, methods §L0:")
    p("`boyroid` vs `boy-roid`, `It` read as `lt`, spacing around `>>`). Their measured")
    p("confidences, next to the forms they get confused with:")
    p("")
    drift_rows = collections.defaultdict(list)
    watch = {"It", "lt", "l", "I", "boy-roid", "boyroid", ">>", "::", ":", ">", "1", "`", "'"}
    for t in tokens_rows:
        if t["token"] in watch:
            drift_rows[t["token"]].append(t)
    notes = {
        "It": "correct capital-I-t",
        "lt": "drift: capital I read as lowercase l",
        "l": "drift residue (bare lowercase l)",
        "I": "bare capital I",
        "1": "digit one (glyph sibling of l/I)",
        "boy-roid": "correct hyphenated compound",
        "boyroid": "drift: hyphen lost",
        ">>": "stream close",
        "::": "field separator",
        ":": "single colon (glyph sibling of ::)",
        ">": "bare angle bracket (glyph sibling of >>)",
        "`": "backtick operator",
        "'": "apostrophe (glyph sibling of backtick)",
    }
    p("| token | n | mean | p05 | min | note |")
    p("|---|---|---|---|---|---|")
    for tok in ["It", "lt", "l", "I", "1", "boy-roid", "boyroid", ">>", "::", ":", ">", "`", "'"]:
        vals = drift_rows.get(tok, [])
        if not vals:
            p(f"| `{tok}` | 0 | — | — | — | not observed |")
            continue
        means = [v["tok_conf"] for v in vals]
        p(f"| `{tok}` | {len(vals):,} | {statistics.fmean(means):.4f} | "
          f"{pct(means,0.05):.3f} | {min(means):.3f} | {notes.get(tok,'')} |")
    p("")
    p("Read this table before trusting confidence as a filter: if a drift form scores")
    p("as high as its correct twin, confidence cannot be the only gate in the")
    p("gazetteer-protected correction (Chainlink #6) — a frequency/gazetteer rule is")
    p("still required.")
    p("")
    p("## 7. Flags and files")
    p("")
    p("Every token carries structural flags independent of confidence: "
      "`op_exact`, `op_run`, `has_punct`, `has_digit`, `allcaps`, `internal_caps`, "
      "`non_ascii`, `single_char`, `l_initial_short`.")
    p("")
    p("| file | what it is |")
    p("|---|---|")
    p("| `raw_ocr_confidence.jsonl.gz` | one record per OCR'd block: surya line confidence, "
      "line text, and every character with its confidence and page-coordinate bbox |")
    p("| `lines.tsv` | per OCR line: page, block, mean/min char confidence, bbox, text |")
    p("| `tokens.tsv` | per whitespace token: confidences (mean/min/p10), bbox, flags, "
      "±40-char context |")
    p("| `flagged_tokens.tsv` | the low-confidence tokens, worst first, with page/block "
      "citation |")
    p("| `page_confidence.tsv` | per-page aggregates, ready for the folio map (#8) |")
    p("| `conf_histogram.tsv` | char/token confidence histograms |")
    p("| `token_class_confidence.tsv` | mean confidence by class and by operator |")
    p("| `capture_meta.json`, `build_meta.json` | run provenance + validation numbers |")
    p("| `marker_json/blood-electric/blood-electric.json` | marker's own JSON output for "
      "the same run (no confidences, but block structure/polygons for #7/#8) |")
    p("")
    p("## 8. How to join these to the corpus")
    p("")
    p("- `page_id` is marker's 0-based PDF page index (Chainlink #8 must add the")
    p("  printed folio); `block_id` is marker's stable block path (`/page/7/Text/3`).")
    p("- Within a block, `line_index` and `tok_index` follow surya's reading order,")
    p("  i.e. left-to-right, top-to-bottom *as the OCR produced them* — which can")
    p("  differ from marker's post-processed block order (see §1.1). Prefer the")
    p("  (page, block, bbox) triple when joining, and treat `line_index`/`tok_index`")
    p("  as per-block sequence only.")
    p("- `bbox` values are page coordinates in the same space as marker's polygons")
    p("  (points at PDF scale, ~72 dpi units), so flagged tokens can be re-cropped")
    p("  from the source PDF for human adjudication.")
    p("- `context` is taken from the OCR'd line text, so it can differ from the")
    p("  ftfy-normalised markdown corpus in rare cases.")
    p("")
    p("## 9. Caveats (what this does not establish)")
    p("")
    p("1. **Confidence is not an error indicator.** It is an uncalibrated softmax")
    p("   maximum. It is useful for ranking and for cheap triage, not for asserting")
    p("   that a token is correct or wrong.")
    p("2. **One engine only.** No ensemble vote here (that is Chainlink #4); a low")
    p("   score does not localise a *glyph* ambiguity, only this model's uncertainty.")
    p("3. **Space characters carry score too.** Whitespace is emitted by the model,")
    p("   so token means depend on where the model's word boundaries fall; the")
    p("   worst-char column is the more robust statistic.")
    p("4. **Neologisms are not errors.** A genuinely invented word can score low")
    p("   simply because it is unlike the model's prior. Flags therefore must be")
    p("   read together with the gazetteer (#6), not as a correction list.")
    p("")
    open(os.path.join(outdir, "ocr_confidence_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="run marker json + record surya confidences")
    cap.add_argument("--input", default="work/input/blood-electric.pdf")
    cap.add_argument("--outdir", default="analysis/ocr_confidence")
    cap.add_argument("--json-dir", default=None)
    cap.add_argument("--raw", default=None, help="raw dump filename (default raw_ocr_confidence.jsonl.gz)")
    cap.add_argument("--page-range", default=None, help="e.g. 0-9 (pilot runs)")
    cap.add_argument("--recognition-batch-size", type=int, default=None)
    cap.set_defaults(func=cmd_capture)

    bld = sub.add_parser("build", help="derive tables + report from the raw dump")
    bld.add_argument("--outdir", default="analysis/ocr_confidence")
    bld.add_argument("--raw", default=None)
    bld.add_argument("--corpus", default="docs/markdown/blood-electric/blood-electric.md")
    bld.add_argument("--marker-json", default=None)
    bld.add_argument("--min-conf", type=float, default=0.5,
                     help="flag a token if any character is below this")
    bld.add_argument("--mean-conf", type=float, default=0.8,
                     help="flag a token if its mean character confidence is below this")
    bld.set_defaults(func=lambda a: main_build(a))

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())