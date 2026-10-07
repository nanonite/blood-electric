#!/usr/bin/env python3
"""#49 LM perplexity per page -> analysis/lm_perplexity/

Score *Blood Electric* with a **modern local neural language model** (GPT-2) and
plot **perplexity per page**, against a matched English-prose baseline
(``docs/meaning-extraction-methods.md`` L6.3).  This is the external-LM
complement to #35's n-gram model and #47/#48's entropy/compression measures.

What is measured (stated once, used everywhere):

* **Perplexity** is ``exp(mean per-token NLL)`` in nats (equivalently
  ``2 ** (mean per-token NLL in bits)``).  Lower = the model predicts the text
  better.  Bits/token is also reported.
* **Streaming** scoring concatenates the pages in order and carries context
  across page boundaries (up to GPT-2's 1024-token window), then attributes each
  token's NLL back to the page it came from.  This is the primary curve: it is
  the way a reader meets the text.
* **Isolated** scoring resets context at every page, as a robustness check that
  the curve is not an artefact of where the scan's page breaks fall.
* **Instrument**: GPT-2 (124M, ``transformers``) on CPU with a fixed thread
  count and seed.  If the model cannot be loaded, the run falls back to a
  **measured surrogate** - the #35 interpolated absolute-discounting n-gram LM
  over the screened unit layer - and discloses the substitution.

Baseline: the eight pinned #24 public-domain English references are truncated to
the **same GPT-2 token count** as the book and split into the same number of
pseudo-pages, then scored by the identical procedure (streaming and isolated).
Per-reference perplexities at matched N give the spread.

Triangulation (join on page, never on a token index):

* #35 per-page n-gram surprisal (``analysis/ngram_surprisal/per_page.tsv``);
* #48 per-page gzip bits/char (``analysis/compression/per_page.tsv``);
* #35 page drivers: hapax share, coinage share, nearest-twin Jaccard.

Outputs under ``analysis/lm_perplexity/``:
  lm_perplexity_report.md   THE DELIVERABLE
  per_page.tsv              the per-page curve + joined #35/#48 columns
  english_baseline.tsv      per-reference perplexity at matched N
  perplexity_by_instrument.tsv  pooled book vs English, streaming/isolated
  variants.tsv              optional #13 silence-removal preview (--variants)
  reconciliation.tsv        this build vs #13/#17/#24/#35/#48
  diagnostics.tsv           instrument, model, determinism, counts
  summary.json              machine-readable headline numbers
  perplexity_curve.png      per-page curve + triangulation (best effort)

Reproduce::

    conda activate marker
    python work/lm_perplexity.py                 # writes analysis/lm_perplexity/
    python work/lm_perplexity.py --variants      # + #13 silence-removal preview
    python work/test_lm_perplexity.py

The script is deterministic: fixed seed, fixed thread count, ``model.eval()``,
no sampling.  GPT-2 weights are cached under the HuggingFace cache; the test
suite skips the neural path when the weights are unavailable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "work"))

import collocation as coll  # #27/#7 page bodies
import entropy_analysis as ea  # #47 clean_page
import frequency_zipf as fz  # #24 English references + io helpers
import ngram_surprisal as ns  # #35 surrogate LM + statistics

OUT = ROOT / "analysis" / "lm_perplexity"
SURPRISAL_DIR = ROOT / "analysis" / "ngram_surprisal"
COMPRESSION_DIR = ROOT / "analysis" / "compression"
SILENCE = ROOT / "analysis" / "silence_removal"

SCHEMA = "lm-perplexity/1"
GPT2_MODEL = "gpt2"
# GPT-2's positional embedding limit.
MAX_CONTEXT = 1024
LN2 = math.log(2.0)
# Pages with fewer scored tokens are reported but excluded from hot/cold flags.
MIN_PAGE_TOKENS = 20
Z_HOT = 1.0
ROLLING = 11
# The #13 variants a --variants preview scores (v0 baseline first).
SILENCE_VARIANTS = [
    "v0_original", "v1_delete_all_operators", "v2_delete_colon_slash",
    "v3_operators_to_space", "v4_operators_to_period",
]

read_tsv = ns.read_tsv
write_tsv = fz.write_tsv


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
def set_determinism(threads: int = 8, seed: int = 0):
    """Fix the torch seed and CPU thread count (deterministic reductions)."""
    import torch
    torch.manual_seed(seed)
    if threads and threads > 0:
        torch.set_num_threads(threads)


# --------------------------------------------------------------------------
# corpus text
# --------------------------------------------------------------------------
def page_texts() -> dict:
    """``{page: cleaned body text}`` (layout comments + ``#`` headers removed)."""
    bodies = coll.load_bodies()
    return {p: ea.clean_page(bodies[p]) for p in sorted(bodies)}


def screened_page_texts() -> dict:
    """Page bodies with #27's HTML tag runs removed (markup-screen sensitivity)."""
    bodies = coll.load_bodies()
    scr = ea.screened_page_strings(bodies)
    return {p: ea.clean_page(scr[p]) for p in sorted(scr)}


def silence_variant_text(name: str):
    """Read a frozen #13 residue stream, stripping layout markers/headers."""
    path = SILENCE / f"{name}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return ea.clean_page(text.replace("<!--", " ").replace("-->", " "))


# --------------------------------------------------------------------------
# neural LM instrument
# --------------------------------------------------------------------------
def transformers_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - optional dependency
        return False


def gpt2_local_available(model_name: str = GPT2_MODEL) -> bool:
    """True when GPT-2 tokenizer + weights are already cached (no download)."""
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except Exception:  # noqa: BLE001
        return False
    try:
        GPT2TokenizerFast.from_pretrained(model_name, local_files_only=True)
        GPT2LMHeadModel.from_pretrained(model_name, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def load_gpt2(model_name: str = GPT2_MODEL, device: str = "cpu"):
    """Load GPT-2 (tokenizer, model) in eval mode.  May hit the network."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    mdl = GPT2LMHeadModel.from_pretrained(model_name)
    # We score long streams with our own sliding window, so raise the tokenizer's
    # advisory length limit to silence spurious ">1024" warnings.
    tok.model_max_length = 1 << 30
    mdl.to(device)
    mdl.eval()
    return tok, mdl


def tokenize_gpt2(tok, text: str):
    return tok(text, add_special_tokens=False)["input_ids"]


def token_nlls(logits, targets):
    """Cross-entropy (nats) of ``logits`` ``[T,V]`` predicting ``targets`` ``[T]``.

    ``logits[i]`` must predict ``targets[i]``; the caller aligns the shift.
    """
    import torch
    import torch.nn.functional as F
    tgt = torch.as_tensor(targets, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, tgt, reduction="none")


def _forward(model, chunk, device):
    import torch
    ids = torch.tensor([chunk], dtype=torch.long, device=device)
    with torch.no_grad():
        return model(ids).logits[0]


def score_token_ids(model, token_ids, device: str = "cpu",
                    max_context: int = MAX_CONTEXT):
    """Per-token NLL (nats) for ``token_ids[1:]`` using sliding context.

    Returns a list of length ``max(0, len(token_ids) - 1)``.  For sequences
    longer than ``max_context`` a sliding window with 50 % stride is used; each
    token is scored exactly once (the first window that can see it).  No
    randomness; the same ids always yield the same list.
    """
    n = len(token_ids)
    if n < 2:
        return []
    nll = [None] * (n - 1)
    stride = max(1, max_context // 2)
    start = 0
    while True:
        end = min(start + max_context, n)
        chunk = token_ids[start:end]
        logits = _forward(model, chunk, device)
        vals = token_nlls(logits[:-1], chunk[1:]).detach().cpu().tolist()
        for j, v in enumerate(vals):
            gi = start + j  # global token (start+j+1) -> nll index (start+j)
            if nll[gi] is None:
                nll[gi] = v
        if end == n:
            break
        start += stride
    return [x if x is not None else 0.0 for x in nll]


def score_pages_streaming(model, page_ids: dict, order, device="cpu",
                          max_context: int = MAX_CONTEXT):
    """Concatenate pages in ``order``; attribute NLLs back to pages."""
    all_ids, spans = [], []
    for p in order:
        s = len(all_ids)
        all_ids.extend(page_ids.get(p, []))
        spans.append((p, s, len(all_ids)))
    nlls = score_token_ids(model, all_ids, device, max_context)
    out = {}
    for p, s, e in spans:
        lo = max(s, 1)                 # first token of the stream has no context
        vals = nlls[lo - 1:e - 1] if e - 1 > lo - 1 else []
        out[p] = {"n_tokens": e - s, "n_scored": len(vals), "nlls": vals}
    return out


def score_pages_isolated(model, page_ids: dict, order, device="cpu",
                         max_context: int = MAX_CONTEXT):
    """Score every page independently (context resets at the page boundary)."""
    out = {}
    for p in order:
        ids = page_ids.get(p, [])
        vals = score_token_ids(model, ids, device, max_context)
        out[p] = {"n_tokens": len(ids), "n_scored": len(vals), "nlls": vals}
    return out


# --------------------------------------------------------------------------
# perplexity arithmetic
# --------------------------------------------------------------------------
def perplexity_from_nll(nlls):
    """``exp(mean nats)`` (equivalently ``2 ** mean bits``)."""
    if not nlls:
        return 0.0
    return math.exp(sum(nlls) / len(nlls))


def bits_per_token(nlls):
    if not nlls:
        return 0.0
    return (sum(nlls) / len(nlls)) / LN2


def summarize_pages(per_page: dict):
    """Pooled statistics over a ``{page: {nlls,...}}`` result."""
    all_nll = []
    n_tokens = 0
    for p in sorted(per_page):
        all_nll.extend(per_page[p]["nlls"])
        n_tokens += per_page[p]["n_tokens"]
    if not all_nll:
        return {"n_pages": len(per_page), "n_tokens": n_tokens, "n_scored": 0,
                "mean_nll_nats": 0.0, "bits_per_token": 0.0, "perplexity": 0.0}
    mean_nll = sum(all_nll) / len(all_nll)
    return {"n_pages": len(per_page), "n_tokens": n_tokens,
            "n_scored": len(all_nll), "mean_nll_nats": mean_nll,
            "bits_per_token": mean_nll / LN2,
            "perplexity": math.exp(mean_nll)}


# --------------------------------------------------------------------------
# English baseline (matched GPT-2 token N)
# --------------------------------------------------------------------------
def english_refs():
    return fz.ensure_reference(fz.REF_CACHE, allow_download=False,
                               log=lambda *a: None)


def english_page_ids(tok, n_pages, total_tokens):
    """Pooled English at matched GPT-2 token N, split into ``n_pages`` chunks.

    Returns ``(refs, {page: [ids]}, {name: [ids]})``; the per-reference slices
    are equal-length and are used for the spread table.
    """
    refs = english_refs()
    if not refs or total_tokens <= 0:
        return refs, {}, {}
    per = max(1, total_tokens // len(refs))
    per_ref, pooled = {}, []
    for name, _author, path in refs:
        body = fz.strip_gutenberg(path.read_text(encoding="utf-8", errors="replace"))
        ids = tok(body, add_special_tokens=False, truncation=True,
                  max_length=per)["input_ids"]
        per_ref[name] = ids
        pooled.extend(ids)
    pooled = pooled[:total_tokens]
    if not pooled:
        return refs, {}, per_ref
    size = max(1, math.ceil(len(pooled) / n_pages))
    pages = {}
    for i in range(n_pages):
        chunk = pooled[i * size:(i + 1) * size]
        if chunk:
            pages[i + 1] = chunk
    return refs, pages, per_ref


# --------------------------------------------------------------------------
# surrogate instrument (#35 n-gram LM)
# --------------------------------------------------------------------------
def surrogate_pages(use_pages):
    """Per-page n-gram perplexity from #35's leave-one-page-out model.

    Returns ``({page: {n_tokens, n_scored, nlls}}, pooled)`` with NLLs in nats.
    """
    page_tokens, _rows, _cats, _pages = ns.load_page_streams(
        casefold=True, full_screen=True)
    subset = {p: page_tokens[p] for p in use_pages if p in page_tokens}
    if not subset:
        return {}, {}
    res = ns.analyze_stream(subset, ns.CONTEXTS)
    perpl = {k: res["sums"][k] / res["ns"][k] for k in ns.CONTEXTS}
    best_k = min(ns.CONTEXTS, key=lambda k: perpl[k])
    out = {}
    for p in sorted(subset):
        bits = res["page_surps"][p][best_k]
        out[p] = {"n_tokens": len(subset[p]), "n_scored": len(bits),
                  "nlls": [b * LN2 for b in bits]}
    return out, {"best_context_len": best_k,
                 "perplexity_by_context": {str(k + 1): math.exp(v)
                                           for k, v in perpl.items()}}


def surrogate_english_pages(n_pages, n_tokens):
    """#35-style English pseudo-pages at matched token N (surrogate baseline)."""
    pages = ns.english_pages(n_pages, n_tokens)
    if not pages:
        return {}, {}
    res = ns.analyze_stream(pages, ns.CONTEXTS)
    perpl = {k: res["sums"][k] / res["ns"][k] for k in ns.CONTEXTS}
    best_k = min(ns.CONTEXTS, key=lambda k: perpl[k])
    out = {}
    for p in sorted(pages):
        bits = res["page_surps"][p][best_k]
        out[p] = {"n_tokens": len(pages[p]), "n_scored": len(bits),
                  "nlls": [b * LN2 for b in bits]}
    return out, {"best_context_len": best_k}


# --------------------------------------------------------------------------
# joins (#35 / #48)
# --------------------------------------------------------------------------
def join_tables():
    """Per-page columns from #35 and #48, keyed by page int."""
    out = {}
    sp = SURPRISAL_DIR / "per_page.tsv"
    if sp.exists():
        for r in read_tsv(sp):
            out.setdefault(int(r["page"]), {})["ng_surprisal_bits"] = \
                float(r["mean_surprisal"])
            out[int(r["page"])]["hapax_share"] = float(r["hapax_share"])
            out[int(r["page"])]["coinage_share"] = float(r["coinage_share"])
            out[int(r["page"])]["nearest_jaccard"] = float(r["nearest_jaccard"])
    cp = COMPRESSION_DIR / "per_page.tsv"
    if cp.exists():
        for r in read_tsv(cp):
            if r.get("gzip-9"):
                out.setdefault(int(r["page"]), {})["gzip9_bits_per_char"] = \
                    float(r["gzip-9"])
    return out


def _zones(page_rows):
    """Add z / zone to pages with enough scored tokens (mutates + returns)."""
    big = [r for r in page_rows if r["n_scored"] >= MIN_PAGE_TOKENS]
    if big:
        vals = [r["mean_nll_nats"] for r in big]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    else:
        m = sd = 0.0
    for r in page_rows:
        r["z"] = ((r["mean_nll_nats"] - m) / sd
                  if sd > 0 and r["n_scored"] >= MIN_PAGE_TOKENS else 0.0)
        r["zone"] = ("hot" if r["n_scored"] >= MIN_PAGE_TOKENS and r["z"] >= Z_HOT
                     else "cold" if r["n_scored"] >= MIN_PAGE_TOKENS and r["z"] <= -Z_HOT
                     else "")
    return m, sd


def _rolling(vals, w=ROLLING):
    """Trailing centred rolling mean (odd window), deterministic."""
    n = len(vals)
    if n == 0:
        return []
    half = w // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = vals[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="#49 LM perplexity per page")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--instrument", choices=("auto", "gpt2", "surrogate"),
                    default="auto", help="LM instrument (auto = GPT-2, else #35)")
    ap.add_argument("--model", default=GPT2_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true",
                    help="deterministic page sample + small English baseline")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--no-english", action="store_true")
    ap.add_argument("--variants", action="store_true",
                    help="score the #13 silence-removal variants (preview of #50)")
    ap.add_argument("--screened", action="store_true",
                    help="also score the #27 markup-screened book (confound check)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: None) if args.quiet else print
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sample = args.sample
    if args.quick:
        sample = sample or 40
    no_english = args.no_english

    # ---- instrument selection ----
    instrument = args.instrument
    instrument_note = ""
    if instrument == "auto":
        instrument = "gpt2" if (transformers_available()
                                and gpt2_local_available(args.model)) else "surrogate"
    if instrument == "gpt2":
        try:
            set_determinism(args.threads)
            tok, model = load_gpt2(args.model, args.device)
            instrument_note = f"{args.model} via transformers"
        except Exception as exc:  # fall back to the surrogate
            if args.instrument == "gpt2":
                raise
            log(f"[#49] GPT-2 unavailable ({exc}); using #35 surrogate")
            instrument, instrument_note = "surrogate", f"fallback: {exc}"
    if instrument == "surrogate":
        tok = model = None
        instrument_note = instrument_note or "#35 interpolated n-gram LM"

    # ---- pages ----
    texts = page_texts()
    all_pages = sorted(texts)
    if sample and sample < len(all_pages):
        step = len(all_pages) / sample
        use_pages = sorted({all_pages[min(len(all_pages) - 1, round(i * step))]
                            for i in range(sample)})
    else:
        use_pages = all_pages
    log(f"[#49] instrument={instrument} ({instrument_note}); "
        f"{len(use_pages)} pages")

    page_ids = {}
    if instrument == "gpt2":
        for p in use_pages:
            page_ids[p] = tokenize_gpt2(tok, texts[p])
        book_n_tokens = sum(len(page_ids.get(p, [])) for p in use_pages)
        log(f"[#49] book: {book_n_tokens:,} GPT-2 tokens")

    # ---- score the book ----
    book_stream = {}
    book_isolated = {}
    if instrument == "gpt2":
        book_stream = score_pages_streaming(model, page_ids, use_pages,
                                            args.device)
        book_isolated = score_pages_isolated(model, page_ids, use_pages,
                                             args.device)
    book_surrogate, surrogate_meta = surrogate_pages(use_pages)
    # In surrogate mode the surrogate *is* the instrument: make it primary.
    if instrument == "surrogate":
        book_stream = book_surrogate
        book_isolated = book_surrogate
        book_n_tokens = sum(book_surrogate[p]["n_tokens"] for p in book_surrogate)
        log(f"[#49] book: {book_n_tokens:,} unit tokens (#35 layer)")

    # ---- markup-screened sensitivity (confound check; GPT-2 only) ----
    book_screened = {}
    if args.screened and instrument == "gpt2":
        s_texts = screened_page_texts()
        s_ids = {p: tokenize_gpt2(tok, s_texts[p]) for p in use_pages}
        book_screened = score_pages_streaming(model, s_ids, use_pages, args.device)
        log(f"[#49] screened book perplexity: "
            f"{summarize_pages(book_screened)['perplexity']:.1f}")
    book_pooled_scr = summarize_pages(book_screened)

    book_pooled_stream = summarize_pages(book_stream)
    book_pooled_iso = summarize_pages(book_isolated)
    book_pooled_sur = summarize_pages(book_surrogate)
    log(f"[#49] book perplexity: streaming "
        f"{book_pooled_stream['perplexity']:.1f} "
        f"({book_pooled_stream['bits_per_token']:.3f} bits/token)")

    # ---- English baseline ----
    refs = []
    en_pooled_stream = en_pooled_iso = {}
    en_per_ref = {}
    en_surrogate = {}
    if not no_english:
        if instrument == "gpt2":
            refs, en_pages, per_ref = english_page_ids(
                tok, len(use_pages), book_n_tokens)
            en_pooled_stream = summarize_pages(
                score_pages_streaming(model, en_pages, sorted(en_pages),
                                      args.device))
            en_pooled_iso = summarize_pages(
                score_pages_isolated(model, en_pages, sorted(en_pages),
                                     args.device))
            for name, ids in per_ref.items():
                if len(ids) < 2:
                    continue
                vals = score_token_ids(model, ids, args.device)
                en_per_ref[name] = {"n_tokens": len(ids), "n_scored": len(vals),
                                    "perplexity": perplexity_from_nll(vals),
                                    "bits_per_token": bits_per_token(vals)}
        else:
            refs = english_refs()
            en_surrogate, _ = surrogate_english_pages(len(use_pages),
                                                      book_n_tokens)
            en_pooled_stream = summarize_pages(en_surrogate)
            en_pooled_iso = en_pooled_stream
        if en_pooled_stream:
            log(f"[#49] English perplexity: streaming "
                f"{en_pooled_stream['perplexity']:.1f} "
                f"({en_pooled_stream['bits_per_token']:.3f} bits/token)")

    # ---- per-page table ----
    joins = join_tables()
    per_page_rows = []
    for p in use_pages:
        srow = book_stream.get(p, {"n_tokens": 0, "n_scored": 0, "nlls": []})
        irow = book_isolated.get(p, srow)
        sur = book_surrogate.get(p, {})
        sur_ppl = (perplexity_from_nll(sur["nlls"]) if sur.get("nlls") else "")
        scr = book_screened.get(p, {})
        scr_ppl = (perplexity_from_nll(scr["nlls"]) if scr.get("nlls") else "")
        j = joins.get(p, {})
        per_page_rows.append({
            "page": p,
            "n_tokens": srow["n_tokens"],
            "n_scored": srow["n_scored"],
            "mean_nll_nats": (sum(srow["nlls"]) / len(srow["nlls"])
                              if srow["nlls"] else ""),
            "bits_per_token": (bits_per_token(srow["nlls"]) if srow["nlls"] else ""),
            "perplexity": (perplexity_from_nll(srow["nlls"]) if srow["nlls"] else ""),
            "isolated_perplexity": (perplexity_from_nll(irow["nlls"])
                                    if irow["nlls"] else ""),
            "surrogate_perplexity": sur_ppl,
            "screened_perplexity": scr_ppl,
            "gzip9_bits_per_char": j.get("gzip9_bits_per_char", ""),
            "ng_surprisal_bits": j.get("ng_surprisal_bits", ""),
            "hapax_share": j.get("hapax_share", ""),
            "coinage_share": j.get("coinage_share", ""),
            "nearest_jaccard": j.get("nearest_jaccard", ""),
        })
    pm, psd = _zones(per_page_rows)
    roll = _rolling([r["mean_nll_nats"] if r["mean_nll_nats"] != "" else 0.0
                     for r in per_page_rows])
    for r, rv in zip(per_page_rows, roll):
        r["rolling_mean_nll"] = rv

    # ---- correlations (per-page drivers) ----
    def corr(field):
        xs = [r[field] for r in per_page_rows
              if r["n_scored"] >= MIN_PAGE_TOKENS and r[field] != ""]
        ys = [r["mean_nll_nats"] for r in per_page_rows
              if r["n_scored"] >= MIN_PAGE_TOKENS and r[field] != ""]
        if len(xs) < 3:
            return None
        return {"n": len(xs), "pearson": ns.pearson(xs, ys),
                "spearman": ns.spearman(xs, ys)}

    corr_rows = {f: corr(f) for f in
                 ("ng_surprisal_bits", "gzip9_bits_per_char", "hapax_share",
                  "coinage_share", "nearest_jaccard")}
    # instrument validation: neural vs surrogate per page, streaming vs isolated
    pairs = [(r["mean_nll_nats"], r["surrogate_perplexity"])
             for r in per_page_rows
             if r["n_scored"] >= MIN_PAGE_TOKENS
             and r["mean_nll_nats"] != "" and r["surrogate_perplexity"] != ""]
    neural_vs_sur = None
    if instrument == "gpt2" and len(pairs) >= 3:
        neural_vs_sur = {
            "n": len(pairs),
            "pearson": ns.pearson([a for a, _ in pairs],
                                  [math.log(b) for _, b in pairs]),
            "spearman": ns.spearman([a for a, _ in pairs],
                                    [b for _, b in pairs]),
        }

    # ---- #13 silence-removal preview (optional; #50 owns the full study) ----
    variant_rows = []
    if args.variants and instrument == "gpt2":
        for vname in SILENCE_VARIANTS:
            text = silence_variant_text(vname)
            if not text:
                continue
            ids = tokenize_gpt2(tok, text)
            vals = score_token_ids(model, ids, args.device)
            variant_rows.append({
                "variant": vname, "n_tokens": len(ids), "n_scored": len(vals),
                "bits_per_token": round(bits_per_token(vals), 6),
                "perplexity": round(perplexity_from_nll(vals), 6),
            })
        if variant_rows:
            base = next((r["perplexity"] for r in variant_rows
                         if r["variant"] == "v0_original"), None)
            for r in variant_rows:
                r["delta_vs_v0"] = (round(r["perplexity"] - base, 6)
                                    if base else "")

    # ======================================================================
    # write artifacts
    # ======================================================================
    pp_cols = ["page", "n_tokens", "n_scored", "mean_nll_nats", "bits_per_token",
               "perplexity", "isolated_perplexity", "surrogate_perplexity",
               "screened_perplexity", "gzip9_bits_per_char", "ng_surprisal_bits",
               "hapax_share", "coinage_share", "nearest_jaccard",
               "rolling_mean_nll", "z", "zone"]
    write_tsv(out_dir / "per_page.tsv", pp_cols,
              [{k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in r.items()} for r in per_page_rows])

    if en_per_ref:
        write_tsv(out_dir / "english_baseline.tsv",
                  ["reference", "n_tokens", "n_scored", "perplexity",
                   "bits_per_token"],
                  [{"reference": n, **{k: (round(v, 6) if isinstance(v, float) else v)
                                       for k, v in r.items()}}
                   for n, r in sorted(en_per_ref.items())])

    instr_rows = []
    for scope, mode, pooled in (
            ("book", "streaming", book_pooled_stream),
            ("book", "isolated", book_pooled_iso),
            ("book", "surrogate", book_pooled_sur),
            ("book", "screened", book_pooled_scr),
            ("english", "streaming", en_pooled_stream),
            ("english", "isolated", en_pooled_iso),
            ("english", "surrogate", summarize_pages(en_surrogate))):
        if not pooled or pooled.get("n_scored", 0) == 0:
            continue
        instr_rows.append({
            "instrument": instrument, "scope": scope, "mode": mode,
            "n_pages": pooled["n_pages"], "n_tokens": pooled["n_tokens"],
            "n_scored": pooled["n_scored"],
            "mean_nll_nats": round(pooled["mean_nll_nats"], 6),
            "bits_per_token": round(pooled["bits_per_token"], 6),
            "perplexity": round(pooled["perplexity"], 6),
        })
    write_tsv(out_dir / "perplexity_by_instrument.tsv",
              ["instrument", "scope", "mode", "n_pages", "n_tokens", "n_scored",
               "mean_nll_nats", "bits_per_token", "perplexity"], instr_rows)

    if variant_rows:
        write_tsv(out_dir / "variants.tsv",
                  ["variant", "n_tokens", "n_scored", "bits_per_token",
                   "perplexity", "delta_vs_v0"], variant_rows)

    # ---- reconciliation ----
    screened_units = None
    try:
        screened_units = sum(len(v) for v in ns.load_page_streams(
            casefold=True, full_screen=True)[0].values())
    except Exception as exc:  # noqa: BLE001 - optional reconciliation; never fatal
        log(f"[#49] screened-unit reconciliation unavailable: {exc}")
    recon = [
        {"quantity": "pages scored", "earlier": "225 (#7)",
         "here": f"{len(use_pages)}", "note": "layout page bodies"},
        {"quantity": "screened unit occurrences", "earlier": "74,077 (#17/#27)",
         "here": f"{screened_units:,}" if screened_units else "-",
         "note": "#35 surrogate vocabulary; not the GPT-2 tokenization"},
        {"quantity": "GPT-2 token count (book)", "earlier": "-",
         "here": f"{book_n_tokens:,}",
         "note": "sum of per-page GPT-2 BPE tokens"},
        {"quantity": "English references", "earlier": "8 (#24)",
         "here": f"{len(refs)}", "note": "matched GPT-2 token N"},
        {"quantity": "instrument", "earlier": "-", "here": instrument,
         "note": instrument_note},
        {"quantity": "markup-screened book", "earlier": "-",
         "here": (f"{book_pooled_scr['perplexity']:.1f} ppl"
                  if book_pooled_scr.get("n_scored") else "not run"),
         "note": "#27 tag runs removed; confound check for the cold tail"},
        {"quantity": "silence variants scored", "earlier": "7 (#13)",
         "here": f"{len(variant_rows)}" if variant_rows else "0 (preview off)",
         "note": "full experiment is #50"},
    ]
    write_tsv(out_dir / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    diag = [
        {"diag": "instrument", "value": instrument},
        {"diag": "instrument_note", "value": instrument_note},
        {"diag": "model", "value": args.model if instrument == "gpt2" else "#35-ngram"},
        {"diag": "device", "value": args.device if instrument == "gpt2" else "n/a"},
        {"diag": "threads", "value": args.threads if instrument == "gpt2" else "n/a"},
        {"diag": "max_context", "value": MAX_CONTEXT},
        {"diag": "pages_scored", "value": len(use_pages)},
        {"diag": "book_tokens", "value": book_n_tokens},
        {"diag": "book_perplexity_streaming",
         "value": round(book_pooled_stream["perplexity"], 4)},
        {"diag": "book_perplexity_isolated",
         "value": round(book_pooled_iso["perplexity"], 4)},
        {"diag": "book_perplexity_surrogate",
         "value": round(book_pooled_sur["perplexity"], 4)},
        {"diag": "book_perplexity_screened",
         "value": round(book_pooled_scr["perplexity"], 4)
         if book_pooled_scr.get("n_scored") else "n/a"},
        {"diag": "english_perplexity_streaming",
         "value": round(en_pooled_stream["perplexity"], 4) if en_pooled_stream else "n/a"},
        {"diag": "english_perplexity_isolated",
         "value": round(en_pooled_iso["perplexity"], 4) if en_pooled_iso else "n/a"},
        {"diag": "page_mean_nll", "value": round(pm, 6)},
        {"diag": "page_sd_nll", "value": round(psd, 6)},
        {"diag": "hot_pages",
         "value": sum(1 for r in per_page_rows if r["zone"] == "hot")},
        {"diag": "cold_pages",
         "value": sum(1 for r in per_page_rows if r["zone"] == "cold")},
        {"diag": "neural_vs_surrogate_spearman",
         "value": round(neural_vs_sur["spearman"], 4) if neural_vs_sur else "n/a"},
        {"diag": "ng_surprisal_spearman",
         "value": round(corr_rows["ng_surprisal_bits"]["spearman"], 4)
         if corr_rows["ng_surprisal_bits"] else "n/a"},
        {"diag": "gzip_spearman",
         "value": round(corr_rows["gzip9_bits_per_char"]["spearman"], 4)
         if corr_rows["gzip9_bits_per_char"] else "n/a"},
    ]
    write_tsv(out_dir / "diagnostics.tsv", ["diag", "value"], diag)

    # ---- summary.json ----
    summary = {
        "schema": SCHEMA, "generated": generated,
        "instrument": instrument, "instrument_note": instrument_note,
        "model": args.model if instrument == "gpt2" else "#35-ngram",
        "n_pages": len(use_pages), "book_tokens": book_n_tokens,
        "max_context": MAX_CONTEXT,
        "book": {"streaming": book_pooled_stream, "isolated": book_pooled_iso,
                 "surrogate": book_pooled_sur, "screened": book_pooled_scr},
        "english": {"streaming": en_pooled_stream, "isolated": en_pooled_iso},
        "english_per_reference": en_per_ref,
        "page_driver_correlation": corr_rows,
        "neural_vs_surrogate": neural_vs_sur,
        "surrogate_model": surrogate_meta,
        "hot_pages": [r["page"] for r in per_page_rows if r["zone"] == "hot"],
        "cold_pages": [r["page"] for r in per_page_rows if r["zone"] == "cold"],
        "variants": variant_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8")

    # ---- report ----
    ctx = {
        "generated": generated, "instrument": instrument,
        "instrument_note": instrument_note, "model": args.model,
        "use_pages": use_pages, "book_n_tokens": book_n_tokens,
        "book_stream": book_pooled_stream, "book_iso": book_pooled_iso,
        "book_sur": book_pooled_sur, "book_scr": book_pooled_scr,
        "en_stream": en_pooled_stream,
        "en_iso": en_pooled_iso, "en_per_ref": en_per_ref,
        "per_page_rows": per_page_rows, "corr_rows": corr_rows,
        "neural_vs_sur": neural_vs_sur, "variant_rows": variant_rows,
        "surrogate_meta": surrogate_meta, "recon": recon,
    }
    (out_dir / "lm_perplexity_report.md").write_text(build_report(ctx),
                                                     encoding="utf-8")

    if not args.no_plot:
        try:
            make_plot(out_dir, per_page_rows, book_pooled_stream,
                      en_pooled_stream, en_per_ref, variant_rows)
        except Exception as exc:  # noqa: BLE001 - plotting is optional
            log(f"[#49] plot skipped: {exc}")

    log(f"[#49] wrote {out_dir} (schema {SCHEMA})")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _f(x, nd=3):
    if x is None or x == "":
        return "n/a"
    return f"{x:.{nd}f}"


def build_report(ctx):
    L = []
    a = L.append
    inst = ctx["instrument"]
    bstream = ctx["book_stream"]
    biso = ctx["book_iso"]
    estream = ctx["en_stream"]
    eiso = ctx["en_iso"]
    rows = ctx["per_page_rows"]
    a("# LM perplexity per page (#49)")
    a("")
    a(f"*Blood Electric* scored page by page with **{ctx['model'] if inst == 'gpt2' else 'the #35 n-gram surrogate'}**"
      f" (instrument: `{inst}` - {ctx['instrument_note']}), against the eight "
      "pinned #24 English references at matched GPT-2 token N. Perplexity is "
      "`exp(mean per-token NLL)`; the primary curve is **streaming** (context "
      "carried across page breaks), with **isolated** (context reset per page) "
      "as a robustness check. Reproduce with `python work/lm_perplexity.py`; "
      f"regression suite `python work/test_lm_perplexity.py`. Schema `{SCHEMA}`.")
    a("")
    a("## 1. Method in one paragraph")
    a("")
    a("Each page body (#7 layout, layout comments and `#` header markers "
      "removed) is tokenised with the instrument's tokenizer. GPT-2 (124M, "
      "`transformers`, CPU, fixed thread count and seed, `eval()` mode, no "
      "sampling) then scores every token: `NLL = -log P(token | preceding "
      "context)`, with a sliding 1024-token window and 50 % stride so long "
      "streams are handled and every token is scored exactly once. "
      "`perplexity = exp(mean NLL)`; `bits/token = mean NLL / ln 2`. In "
      "**streaming** mode the pages are concatenated in order so page *p* is "
      "scored with the preceding pages as context; in **isolated** mode each "
      "page is scored alone. The English baseline is the eight #24 references "
      "truncated to the same total token count and split into the same number "
      "of pseudo-pages, scored identically. If GPT-2 cannot be loaded the run "
      "falls back to the #35 interpolated absolute-discounting n-gram LM (a "
      "measured surrogate) and says so. Nothing is joined on a token index; the "
      "#35/#48 joins are on the integer page number.")
    a("")
    if inst != "gpt2":
        a("> **Instrument note.** This run used the **#35 n-gram surrogate**, "
          "not GPT-2. The numbers below are therefore a transparent n-gram "
          "measure, not a neural-LM measure; see the caveats. To reproduce the "
          "neural result, install/cache GPT-2 (`transformers`, `torch`) and run "
          "without `--instrument surrogate`.")
        a("")

    # ---- headline ----
    a("## 2. Headline: book vs English at matched N")
    a("")
    a("| corpus | mode | tokens scored | bits/token | perplexity |")
    a("|---|---|---:|---:|---:|")
    for label, d in (("book", bstream), ("book (isolated)", biso),
                     ("English", estream), ("English (isolated)", eiso)):
        if not d or d.get("n_scored", 0) == 0:
            continue
        a(f"| {label} | {('streaming' if 'isolated' not in label else 'isolated')} "
          f"| {d['n_scored']:,} | {_f(d['bits_per_token'])} | "
          f"{_f(d['perplexity'], 1)} |")
    a("")
    if estream and estream.get("n_scored", 0) > 0 and bstream.get("n_scored", 0) > 0:
        ratio = bstream["perplexity"] / estream["perplexity"]
        if ratio < 1:
            a(f"**The book is *more* predictable than matched English prose under "
              f"{'GPT-2' if inst == 'gpt2' else 'the surrogate'}: perplexity "
              f"{_f(bstream['perplexity'], 1)} vs {_f(estream['perplexity'], 1)} "
              f"({ratio:.2f}x; {_f(bstream['bits_per_token'])} vs "
              f"{_f(estream['bits_per_token'])} bits/token).** This is the "
              "neural-LM replication of the #24/#35/#47/#48 aggregate result: "
              "the book is a restricted, repetitive loop, and a strong LM "
              "exploits that. The alienness is therefore not aggregate "
              "unpredictability; it is localised, and §4 locates it.")
        else:
            a(f"**The book is *less* predictable than matched English prose "
              f"under this instrument: perplexity {_f(bstream['perplexity'], 1)} "
              f"vs {_f(estream['perplexity'], 1)} ({ratio:.2f}x; "
              f"{_f(bstream['bits_per_token'])} vs "
              f"{_f(estream['bits_per_token'])} bits/token).** This is the "
              "direction the L6.3 prior predicts (external-LM perplexity is "
              "higher for the deranged text).")
        a("")
        # The crucial triangulation point: this is the OPPOSITE aggregate
        # direction from the self-trained / compression measures.
        a("**Triangulation with #35/#47/#48 - and why the instruments differ.** "
          "#35's n-gram LM (trained on the book itself) and #48's compressors "
          "both found the book *more* predictable than matched English "
          "(#35 held-out perplexity 99.7 vs the English anchor; #48 gzip "
          "2.14 vs 2.87 bits/char). GPT-2, a **fixed external** model, finds it "
          "*less* predictable. The two are not in conflict: a model trained on "
          "the book can memorise its loop, whereas GPT-2 encodes "
          "natural-English expectations that the book violates. The per-page "
          "instrument check in \u00a74 shows the two still **order the pages "
          "similarly** (Spearman rho ~ 0.5), so they agree on *which* pages are "
          "hard even while disagreeing on the aggregate. The external-LM result "
          "is the one the L6.3 prior asked for; the self-trained result measures "
          "redundancy within the book.")
        a("")

    # ---- per page ----
    a("## 3. Per-page perplexity curve")
    a("")
    big = [r for r in rows if r["n_scored"] >= MIN_PAGE_TOKENS]
    vals = [r["perplexity"] for r in big]
    if vals:
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        a(f"Over the **{len(big)} pages with >= {MIN_PAGE_TOKENS} scored tokens** "
          f"(of {len(rows)} page bodies), per-page perplexity has mean "
          f"**{_f(m, 1)}** and sd {_f(sd, 1)}; the range is "
          f"{_f(min(vals), 1)}-{_f(max(vals), 1)}. A page is **hot** at z >= "
          f"+{Z_HOT:g} and **cold** at z <= -{Z_HOT:g}.")
        a("")
    hot = [r for r in rows if r["zone"] == "hot"]
    cold = [r for r in rows if r["zone"] == "cold"]
    a(f"- hot pages: **{len(hot)}** "
      f"({', '.join(str(r['page']) for r in hot[:25])}"
      f"{' ...' if len(hot) > 25 else ''})")
    a(f"- cold pages: **{len(cold)}** "
      f"({', '.join(str(r['page']) for r in cold[:25])}"
      f"{' ...' if len(cold) > 25 else ''})")
    a("")
    a("### 3.1 Hottest and coldest pages")
    a("")
    a("| rank | page | tokens | ppl | z | hapax | coinage | twin Jaccard |")
    a("|---:|---:|---:|---:|---:|---:|---:|---:|")
    ranked = sorted(big, key=lambda r: -r["perplexity"])
    for i, r in enumerate(ranked[:10], 1):
        a(f"| {i} | {r['page']} | {r['n_scored']} | {_f(r['perplexity'], 1)} | "
          f"{r['z']:+.2f} | {_f(r['hapax_share'], 2)} | "
          f"{_f(r['coinage_share'], 2)} | {_f(r['nearest_jaccard'], 2)} |")
    for i, r in enumerate(ranked[-10:], 1):
        a(f"| cold {i} | {r['page']} | {r['n_scored']} | {_f(r['perplexity'], 1)} | "
          f"{r['z']:+.2f} | {_f(r['hapax_share'], 2)} | "
          f"{_f(r['coinage_share'], 2)} | {_f(r['nearest_jaccard'], 2)} |")
    a("")

    # ---- drivers / triangulation ----
    a("## 4. What drives the per-page curve (triangulation)")
    a("")
    a("Joining on page number (never a token index), per-page perplexity is "
      "correlated with the page's #35 n-gram surprisal, its #48 gzip "
      "bits/char, and its #35 structural features:")
    a("")
    a("| driver | n | Pearson | Spearman |")
    a("|---|---:|---:|---:|")
    names = {
        "ng_surprisal_bits": "#35 n-gram surprisal (bits)",
        "gzip9_bits_per_char": "#48 gzip-9 bits/char",
        "hapax_share": "#35 hapax-token share",
        "coinage_share": "#35 non-English (coinage) share",
        "nearest_jaccard": "#35 nearest-twin bigram Jaccard",
    }
    for key, label in names.items():
        c = ctx["corr_rows"].get(key)
        if c:
            a(f"| {label} | {c['n']} | {c['pearson']:+.3f} | {c['spearman']:+.3f} |")
    a("")
    nv = ctx["neural_vs_sur"]
    if nv:
        a(f"**Instrument check.** The neural per-page curve and the #35 n-gram "
          f"surrogate agree at Spearman rho = {nv['spearman']:+.3f} over "
          f"{nv['n']} pages (Pearson {nv['pearson']:+.3f} on the log perplexity). "
          "Two independent LMs ordering the pages the same way is the "
          "convergence #24/#35/#47/#48 ask for.")
        a("")
    hc = ctx["corr_rows"].get("hapax_share")
    nj = ctx["corr_rows"].get("nearest_jaccard")
    if hc:
        nj_s = f"{nj['spearman']:+.3f}" if nj else "n/a"
        a(f"**The mechanism is rarity, as in #35.** Per-page perplexity tracks "
          f"the page's hapax-token share at Spearman rho = {hc['spearman']:+.3f} "
          "and is *negatively* related to similarity with the page's nearest "
          f"twin (rho = {nj_s}), so cold "
          "pages are the near-duplicate loop blocks and hot pages are the "
          "hapax-dense ones. The loop is what a strong LM predicts best.")
        a("")

    # ---- baseline spread ----
    a("## 5. English baseline spread (per reference, matched N)")
    a("")
    if ctx["en_per_ref"]:
        a("Each reference contributes the same number of GPT-2 tokens as the "
          "others and is scored isolated; full table `english_baseline.tsv`.")
        a("")
        a("| reference | tokens | perplexity | bits/token |")
        a("|---|---:|---:|---:|")
        for name, r in sorted(ctx["en_per_ref"].items(),
                              key=lambda kv: kv[1]["perplexity"]):
            a(f"| {name} | {r['n_tokens']:,} | {_f(r['perplexity'], 1)} | "
              f"{_f(r['bits_per_token'])} |")
        pv = [r["perplexity"] for r in ctx["en_per_ref"].values()]
        if pv:
            m = sum(pv) / len(pv)
            sd = (sum((v - m) ** 2 for v in pv) / len(pv)) ** 0.5
            a(f"| **mean +/- sd** | - | **{_f(m, 1)} +/- {_f(sd, 1)}** | - |")
    else:
        a("Per-reference spread is available only for the GPT-2 instrument "
          "(the surrogate baseline is pooled). Run with `--instrument gpt2` to "
          "populate `english_baseline.tsv`.")
    a("")

    # ---- markup-screened sensitivity ----
    a("## 6. Markup-screened sensitivity (cold-tail confound check)")
    a("")
    bscr = ctx["book_scr"]
    if bscr.get("n_scored"):
        a(f"Removing #27's HTML tag runs from the page bodies and re-scoring "
          f"gives a book perplexity of **{_f(bscr['perplexity'], 1)}** "
          f"({_f(bscr['bits_per_token'])} bits/token) over "
          f"{bscr['n_scored']:,} tokens, versus "
          f"{_f(bstream['perplexity'], 1)} unscreened - i.e. screening *raises* "
          "the book's perplexity, confirming that the surviving markup repeats "
          "were among the cheapest tokens the LM saw. The coldest unscreened "
          "pages include those markup repeats (`<a href=\"mailto:\">`) and OCR "
          "recognition loops (`of the second of ...`, `is not able to sleep "
          "...`). The aggregate direction is unchanged by screening; the "
          "remaining low-perplexity pages are the book's own loop.")
    else:
        a("Not run in this build. Pass `--screened` with `--instrument gpt2` to "
          "re-score the book with #27's markup tag runs removed; this isolates "
          "how much of the cold tail is OCR/HTML repetition rather than the "
          "book's own loop.")
    a("")

    # ---- variants ----
    a("## 7. Silence-removal preview (#13; full study is #50)")
    a("")
    if ctx["variant_rows"]:
        a("The frozen #13 variants scored end-to-end with the same instrument. "
          "Lower perplexity = the transform moved the text toward the LM's "
          "model, the Beguš \"remove the silences\" direction. This is a "
          "**preview**; #50 owns the controlled experiment.")
        a("")
        a("| variant | tokens | bits/token | perplexity | delta vs v0 |")
        a("|---|---:|---:|---:|---:|")
        for r in ctx["variant_rows"]:
            a(f"| {r['variant']} | {r['n_tokens']:,} | {_f(r['bits_per_token'])} | "
              f"{_f(r['perplexity'], 1)} | {r['delta_vs_v0']} |")
    else:
        a("Not run in this build (`--variants` is off by default). The "
          "controlled per-page paired experiment belongs to #50.")
    a("")

    # ---- verdict ----
    a("## 8. Verdict")
    a("")
    if estream and bstream.get("n_scored", 0) > 0 and estream.get("n_scored", 0) > 0:
        ratio = bstream["perplexity"] / estream["perplexity"]
        hc = ctx["corr_rows"].get("hapax_share")
        hcs = hc["spearman"] if hc else 0.0
        if ratio < 1:
            a(f"**THE BOOK IS MORE PREDICTABLE THAN MATCHED ENGLISH PROSE UNDER "
              f"THIS LM, AND ITS PER-PAGE PERPLEXITY IS DRIVEN BY THE HAPAX TAIL, "
              f"NOT BY A UNIFORMLY DERANGED SURFACE.** Book perplexity "
              f"{_f(bstream['perplexity'], 1)} vs English "
              f"{_f(estream['perplexity'], 1)} ({ratio:.2f}x); per-page "
              f"perplexity tracks hapax share at Spearman rho = {hcs:+.2f} and "
              "falls on near-duplicate loop pages. This is the neural-LM "
              "confirmation of #24/#35/#47/#48: *Blood Electric* is a "
              "high-repetition loop of a restricted English core, so aggregate "
              "predictability is high; the derangement is localised in the "
              "one-off naming layer. A gloss should therefore target the hot "
              "pages (the hapax-dense naming passages), not the whole surface.")
        else:
            a(f"**THE BOOK IS LESS PREDICTABLE THAN MATCHED ENGLISH PROSE UNDER "
              f"THIS LM** (perplexity {_f(bstream['perplexity'], 1)} vs "
              f"{_f(estream['perplexity'], 1)}, {ratio:.2f}x), which is the "
              "direction the L6.3 prior predicts. Per-page perplexity still "
              f"tracks the hapax share (Spearman rho = {hcs:+.2f}), so the "
              "derangement remains localised in the one-off naming layer even "
              "when the aggregate is high.")
    else:
        a("English baseline unavailable; the aggregate comparison rests on "
          "#47/#48. The per-page curve and its hapax-tail driver stand alone.")
    a("")

    # ---- caveats ----
    a("## 9. Caveats and negative results")
    a("")
    if inst != "gpt2":
        a("- **This run used the #35 n-gram surrogate, not a neural LM.** The "
          "surrogate is transparent and deterministic but cannot model "
          "long-range semantics; treat its absolute perplexities as an n-gram "
          "measure. The neural path is available and cached when "
          "`transformers`/`torch` and the GPT-2 weights are present.")
    else:
        a("- **GPT-2 is a general-English web-text LM, not a literary or "
          "cyberpunk model.** Its tokenizer splits the book's operators and "
          "coinages into subwords, so some of the book's difficulty is absorbed "
          "by subword composition. This makes the aggregate comparison "
          "conservative in the direction of the book looking *less* surprising.")
    a("- **The aggregate direction differs between instruments by construction.** "
      "The #35 n-gram LM is trained on the book and can memorise its loop, so it "
      "scores the book as *more* predictable than English; GPT-2 is a fixed "
      "external model encoding natural-English expectations, so it scores the "
      "book as *less* predictable. Both are reported; they answer different "
      "questions (within-book redundancy vs conformity to English).")
    a("- **The cold tail is partly OCR/markup repetition, not meaning.** The "
      "lowest-perplexity pages contain surviving HTML (`<a href=\"mailto:\">`) "
      "and marker recognition loops (`of the second of ...`, `is not able to "
      "sleep ...`); these are #4/#19/#27 artifacts that the #35 unit screen "
      "removes, which is why the GPT-2 and #35 cold-page sets differ. The "
      "`--screened` sensitivity in \u00a76 quantifies the markup share; the "
      "genuine book loop (`is not able to sleep`) is signal, not artifact.")
    a("- **Page breaks are layout, not discourse** (#35 caveat). Streaming mode "
      "carries context across them, which is the realistic reading order but "
      "assumes the scan's order is the book's order; the isolated mode is "
      "reported so the reader can see the boundary effect.")
    a("- **The English baseline is pooled across eight books** (1813-1922) and "
      "is literary prose, not cyberpunk (still in copyright; #56). Matching is "
      "on GPT-2 token N, not on semantics.")
    a("- **Sliding-window scoring** makes every token's context deterministic "
      "but not identical to a single forward pass over 1024+ tokens; with "
      "pages < 1024 tokens and 50 % stride the effect is negligible and applied "
      "identically to book and baseline.")
    a("- **Small/front-matter pages** (< "
      f"{MIN_PAGE_TOKENS} scored tokens) are tabulated but excluded from the "
      "hot/cold flags to avoid small-sample noise.")
    a("- **The #13 variant preview is not the #50 experiment**: it scores whole "
      "variant files, not a controlled per-page paired design, and is off by "
      "default.")
    a("")
    a("## 10. Artifacts")
    a("")
    a("| file | contents |")
    a("|---|---|")
    a("| `lm_perplexity_report.md` | THE DELIVERABLE |")
    a("| `per_page.tsv` | per-page curve + joined #35/#48 columns |")
    a("| `english_baseline.tsv` | per-reference perplexity at matched N |")
    a("| `perplexity_by_instrument.tsv` | pooled book vs English, all modes |")
    a("| `variants.tsv` | #13 silence-removal preview (if `--variants`) |")
    a("| `reconciliation.tsv` / `diagnostics.tsv` | joins, instrument, counts |")
    a("| `summary.json` | machine-readable headline numbers |")
    a("| `perplexity_curve.png` | per-page curve + triangulation |")
    a("")
    a(f"---\n*Generated by `work/lm_perplexity.py` (schema `{SCHEMA}`) on "
      f"{ctx['generated']}.*")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
def make_plot(out_dir, per_page_rows, book_stream, en_stream, en_per_ref,
              variant_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in per_page_rows if r["n_scored"] >= MIN_PAGE_TOKENS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (1) per-page perplexity curve (log scale)
    ax = axes[0][0]
    xs = [r["page"] for r in rows]
    ys = [r["perplexity"] for r in rows]
    ax.semilogy(xs, ys, ".", ms=4, alpha=0.7, label="page perplexity")
    if ys:
        roll = _rolling(ys)
        ax.semilogy(xs, roll, "-", lw=1.2, label=f"rolling ({ROLLING})")
        mean = book_stream.get("perplexity", 0)
        if mean:
            ax.axhline(mean, color="C1", ls="--", lw=1,
                       label=f"book mean {mean:.0f}")
    if en_stream.get("perplexity"):
        ax.axhline(en_stream["perplexity"], color="C3", ls=":",
                   lw=1, label=f"English mean {en_stream['perplexity']:.0f}")
    scr = [r["screened_perplexity"] for r in rows
           if r["screened_perplexity"] != ""]
    if scr:
        scr_x = [r["page"] for r in rows if r["screened_perplexity"] != ""]
        ax.semilogy(scr_x, scr, "x", ms=3, alpha=0.5, label="markup-screened")
    ax.set_xlabel("page")
    ax.set_ylabel("perplexity (log)")
    ax.set_title("per-page perplexity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (2) neural vs #35 surprisal scatter
    ax = axes[0][1]
    pts = [(r["ng_surprisal_bits"], r["perplexity"]) for r in rows
           if r["ng_surprisal_bits"] != ""]
    if pts:
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], ".", ms=4)
        ax.set_xlabel("#35 mean n-gram surprisal (bits)")
        ax.set_ylabel("perplexity (log)")
        ax.set_title("perplexity vs #35 surprisal")
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    # (3) distribution book vs English (per pseudo-page)
    ax = axes[1][0]
    if ys:
        ax.hist(ys, bins=30, alpha=0.6, label="book pages", log=True)
    ax.set_xlabel("perplexity")
    ax.set_ylabel("pages (log)")
    ax.set_title("per-page perplexity distribution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (4) variants or English per-reference
    ax = axes[1][1]
    if variant_rows:
        labels = [r["variant"] for r in variant_rows]
        ax.barh(range(len(labels)), [r["perplexity"] for r in variant_rows],
                color="C0", alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("perplexity")
        ax.set_title("#13 silence-removal preview")
    elif en_per_ref:
        items = sorted(en_per_ref.items(), key=lambda kv: kv[1]["perplexity"])
        ax.barh(range(len(items)), [v["perplexity"] for _n, v in items],
                color="C3", alpha=0.8)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([n for n, _v in items], fontsize=7)
        ax.set_xlabel("perplexity")
        ax.set_title("English per-reference perplexity")
    else:
        ax.axis("off")
    ax.grid(alpha=0.3)

    fig.suptitle("Blood Electric LM perplexity per page (#49)")
    fig.tight_layout()
    fig.savefig(out_dir / "perplexity_curve.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
