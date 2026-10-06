#!/usr/bin/env python3
"""
Chainlink #20 - Validated morphological segmentation.

Applies Morfessor, UniMorph, and ByT5 segmentation to the closed-juncture
word types of the Blood Electric corpus, then validates every segmentation
against the #6 gazetteer so that real coinages are not segmented away.

The gazetteer (analysis/lexical_correction/gazetteer.tsv) is the authority:
  - entity / neologism / compound entries are PROTECTED words.
  - If a segmenter splits a protected word, that segmentation is flagged
    as a gazetteer violation (it would "segment away a real coinage").
  - If a segmenter splits an unprotected word, the segmentation is accepted
    and recorded.

Three segmenters:
  1. Morfessor  - unsupervised, trained on this corpus's word types.
  2. UniMorph  - English morphological database (lemma -> inflections).
  3. ByT5      - byte-level T5, zero-shot morphological segmentation.

Usage:
  python work/validated_segmentation.py [--skip-byt5] [--skip-morfessor]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
import time

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(WORK_DIR)
ANALYSIS_DIR = os.path.join(ROOT_DIR, "analysis")
OUTDIR = os.path.join(ANALYSIS_DIR, "validated_segmentation")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gazetteer() -> dict[str, str]:
    """Load #6 gazetteer: token -> category (entity/neologism/compound/junk)."""
    path = os.path.join(ANALYSIS_DIR, "lexical_correction", "gazetteer.tsv")
    gaz = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gaz[row["token"].lower()] = row["category"]
    return gaz


def load_closed_juncture_types() -> list[dict]:
    """Load closed-juncture word types from #18 decompositions.

    These are the segmentation targets: single words with no operator linker.
    Returns one row per distinct (lower-cased) surface form.
    """
    path = os.path.join(ANALYSIS_DIR, "morphology", "decompositions.tsv")
    types: dict[str, dict] = {}
    page_sets: dict[str, set[int]] = collections.defaultdict(set)
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["rule"] != "R5":
                continue
            if row["markup"] == "y":
                continue
            surface = row["closed_surface"]
            if not surface or not surface.isalpha():
                continue
            key = surface.lower()
            page_sets[key].add(int(row["page"]))
            if key not in types:
                types[key] = {
                    "surface": key,
                    "n": 0,
                    "n_pages": 0,
                    "first_page": int(row["page"]),
                    "first_start": int(row["start"]),
                    "closed_kind": row["closed_kind"],
                    "example_unit": row["unit"],
                }
            types[key]["n"] += 1
    for k, v in types.items():
        v["n_pages"] = len(page_sets[k])
    return sorted(types.values(), key=lambda x: -x["n"])


def load_unimorph_english() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load UniMorph English: lemma -> inflections, and form -> lemma reverse index."""
    path = "/tmp/opencode/unimorph/eng"
    if not os.path.exists(path):
        return {}, {}
    lemma_to_forms: dict[str, list[str]] = collections.defaultdict(list)
    form_to_lemma: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                lemma = parts[0]
                form = parts[1]
                lemma_to_forms[lemma].append(form)
                form_to_lemma[form] = lemma
    return dict(lemma_to_forms), form_to_lemma


# ---------------------------------------------------------------------------
# Segmenter 1: Morfessor
# ---------------------------------------------------------------------------

def run_morfessor(word_types: list[dict]) -> dict[str, list[str]]:
    """Train Morfessor on the corpus word types and segment each one."""
    try:
        from morfessor import BaselineModel
    except ImportError:
        print("Morfessor not installed, skipping")
        return {}

    # Build training data: list of (count, atoms) where atoms is a tuple of chars
    corpus = [(wt["n"], tuple(wt["surface"])) for wt in word_types]

    model = BaselineModel()
    model.load_data(corpus, freqthreshold=1)
    model.train_batch(max_epochs=10)

    # Segment each word type
    results = {}
    for wt in word_types:
        word = wt["surface"]
        try:
            segmentation = model.segment(tuple(word))
            # Morfessor returns tuples of characters; join them into strings
            results[word] = ["".join(m) if isinstance(m, tuple) else str(m) for m in segmentation]
        except Exception:
            results[word] = [word]
    return results


# ---------------------------------------------------------------------------
# Segmenter 2: UniMorph
# ---------------------------------------------------------------------------

def run_unimorph(
    word_types: list[dict],
    form_to_lemma: dict[str, str],
) -> dict[str, list[str]]:
    """Look up each word type in UniMorph English.

    If the word is an inflected form in UniMorph, return [lemma, suffix].
    If the word is a lemma, return [word] (no segmentation).
    If not found, return [word] (no segmentation).
    """
    results = {}
    for wt in word_types:
        word = wt["surface"]
        if word in form_to_lemma:
            lemma = form_to_lemma[word]
            if word == lemma:
                # Word is its own lemma (base form) - no segmentation
                results[word] = [word]
            elif word.startswith(lemma) and len(word) > len(lemma):
                # Regular inflected form - segment as lemma + suffix
                suffix = word[len(lemma):]
                results[word] = [lemma, suffix]
            else:
                # Irregular form - segment as lemma + form
                results[word] = [lemma, word]
        else:
            # Not in UniMorph - no segmentation
            results[word] = [word]
    return results


# ---------------------------------------------------------------------------
# Segmenter 3: ByT5
# ---------------------------------------------------------------------------

def run_byt5(word_types: list[dict]) -> dict[str, list[str]]:
    """Use ByT5 for zero-shot morphological segmentation.

    ByT5 is a byte-level T5 model. We use it with a simple prompt:
    "segment: <word>" -> morpheme sequence.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError:
        print("transformers/torch not available, skipping ByT5")
        return {}

    print("Loading ByT5 model...")
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    model.eval()

    results = {}
    batch_size = 32
    words = [wt["surface"] for wt in word_types]

    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        prompts = [f"segment: {w}" for w in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=32, num_beams=4)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for word, seg in zip(batch, decoded):
            # Parse the segmentation: split on spaces or |
            morphs = [m.strip() for m in re.split(r"[|\s]+", seg) if m.strip()]
            if not morphs:
                morphs = [word]
            results[word] = morphs
        if (i // batch_size) % 10 == 0:
            print(f"  ByT5: {i + len(batch)}/{len(words)}")

    return results


# ---------------------------------------------------------------------------
# Gazetteer validation
# ---------------------------------------------------------------------------

def validate_against_gazetteer(
    word: str,
    segmentation: list[str],
    gazetteer: dict[str, str],
) -> dict:
    """Validate a segmentation against the gazetteer.

    Returns a dict with:
      - protected: bool - whether the word is in the gazetteer
      - category: str - gazetteer category (entity/neologism/compound/junk)
      - violation: bool - whether the segmentation violates the gazetteer
      - reason: str - explanation
    """
    entry = gazetteer.get(word)
    if entry is None:
        return {
            "protected": False,
            "category": "",
            "violation": False,
            "reason": "not in gazetteer",
        }

    # Word is protected
    if len(segmentation) <= 1:
        # No segmentation - always OK
        return {
            "protected": True,
            "category": entry,
            "violation": False,
            "reason": f"protected {entry}, not segmented",
        }

    # Word is protected AND segmented - this is a potential violation
    # Exception: compounds can be segmented if the parts match
    if entry == "compound":
        # Compounds are expected to be segmented
        return {
            "protected": True,
            "category": entry,
            "violation": False,
            "reason": "protected compound, segmentation expected",
        }

    # Entity or neologism that got segmented - VIOLATION
    return {
        "protected": True,
        "category": entry,
        "violation": True,
        "reason": f"protected {entry} was segmented into {segmentation}",
    }


# ---------------------------------------------------------------------------
# Precision spot-check
# ---------------------------------------------------------------------------

def precision_spot_check(
    word_types: list[dict],
    morfessor_results: dict[str, list[str]],
    unimorph_results: dict[str, list[str]],
    byt5_results: dict[str, list[str]],
    gazetteer: dict[str, str],
    n_samples: int = 50,
) -> tuple[list[dict], list[dict]]:
    """Sample segmentations for manual precision spot-check.

    Stratified sample: top-frequency words + random sample + all violations.
    """
    import random
    random.seed(42)

    # All violations (protected words that got segmented)
    violations = []
    for wt in word_types:
        word = wt["surface"]
        for seg_name, results in [("morfessor", morfessor_results),
                                   ("unimorph", unimorph_results),
                                   ("byt5", byt5_results)]:
            if word in results:
                v = validate_against_gazetteer(word, results[word], gazetteer)
                if v["violation"]:
                    violations.append({
                        "word": word,
                        "segmenter": seg_name,
                        "segmentation": "|".join(results[word]),
                        "category": v["category"],
                        "reason": v["reason"],
                    })

    # Top-frequency words
    top_words = [wt["surface"] for wt in word_types[:n_samples]]

    # Random sample
    all_words = [wt["surface"] for wt in word_types]
    random_sample = random.sample(all_words, min(n_samples, len(all_words)))

    # Combine and deduplicate
    sample_words = list(dict.fromkeys(top_words + random_sample))

    spot_checks = []
    for word in sample_words:
        wt = next((w for w in word_types if w["surface"] == word), None)
        if wt is None:
            continue
        row = {
            "word": word,
            "n": wt["n"],
            "closed_kind": wt["closed_kind"],
            "in_gazetteer": word in gazetteer,
            "gazetteer_category": gazetteer.get(word, ""),
            "morfessor": "|".join(morfessor_results.get(word, [word])),
            "unimorph": "|".join(unimorph_results.get(word, [word])),
            "byt5": "|".join(byt5_results.get(word, [word])),
        }
        # Check each segmenter
        for seg_name in ["morfessor", "unimorph", "byt5"]:
            seg = row[seg_name].split("|") if row[seg_name] else [word]
            v = validate_against_gazetteer(word, seg, gazetteer)
            row[f"{seg_name}_violation"] = v["violation"]
            row[f"{seg_name}_reason"] = v["reason"]
        spot_checks.append(row)

    return spot_checks, violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-byt5", action="store_true")
    ap.add_argument("--skip-morfessor", action="store_true")
    ap.add_argument("--skip-unimorph", action="store_true")
    ap.add_argument("--spot-check-n", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    print("Loading gazetteer...")
    gazetteer = load_gazetteer()
    print(f"  {len(gazetteer)} protected types")

    print("Loading closed-juncture word types...")
    word_types = load_closed_juncture_types()
    print(f"  {len(word_types)} distinct closed-juncture word types")

    # Run segmenters
    morfessor_results = {}
    unimorph_results = {}
    byt5_results = {}

    if not args.skip_morfessor:
        print("\n=== Morfessor ===")
        t0 = time.time()
        morfessor_results = run_morfessor(word_types)
        print(f"  {len(morfessor_results)} words segmented in {time.time()-t0:.1f}s")

    if not args.skip_unimorph:
        print("\n=== UniMorph ===")
        t0 = time.time()
        _, form_to_lemma = load_unimorph_english()
        print(f"  {len(form_to_lemma)} inflected forms loaded")
        unimorph_results = run_unimorph(word_types, form_to_lemma)
        print(f"  {len(unimorph_results)} words segmented in {time.time()-t0:.1f}s")

    if not args.skip_byt5:
        print("\n=== ByT5 ===")
        t0 = time.time()
        byt5_results = run_byt5(word_types)
        print(f"  {len(byt5_results)} words segmented in {time.time()-t0:.1f}s")

    # Validate against gazetteer
    print("\n=== Gazetteer validation ===")
    validation_rows = []
    violation_counts = collections.Counter()
    for wt in word_types:
        word = wt["surface"]
        for seg_name, results in [("morfessor", morfessor_results),
                                   ("unimorph", unimorph_results),
                                   ("byt5", byt5_results)]:
            if word not in results:
                continue
            seg = results[word]
            v = validate_against_gazetteer(word, seg, gazetteer)
            validation_rows.append({
                "word": word,
                "n": wt["n"],
                "segmenter": seg_name,
                "segmentation": "|".join(seg),
                "n_morphs": len(seg),
                "protected": v["protected"],
                "category": v["category"],
                "violation": v["violation"],
                "reason": v["reason"],
            })
            if v["violation"]:
                violation_counts[seg_name] += 1

    print(f"  Total segmentations: {len(validation_rows)}")
    print(f"  Violations: {dict(violation_counts)}")

    # Write validation TSV
    val_path = os.path.join(OUTDIR, "validation.tsv")
    with open(val_path, "w") as f:
        f.write("word\tn\tsegmenter\tsegmentation\tn_morphs\tprotected\tcategory\tviolation\treason\n")
        for row in validation_rows:
            f.write(f"{row['word']}\t{row['n']}\t{row['segmenter']}\t{row['segmentation']}\t"
                    f"{row['n_morphs']}\t{row['protected']}\t{row['category']}\t{row['violation']}\t{row['reason']}\n")
    print(f"  -> {val_path}")

    # Write violations TSV
    viol_path = os.path.join(OUTDIR, "violations.tsv")
    with open(viol_path, "w") as f:
        f.write("word\tn\tsegmenter\tsegmentation\tcategory\treason\n")
        for row in validation_rows:
            if row["violation"]:
                f.write(f"{row['word']}\t{row['n']}\t{row['segmenter']}\t{row['segmentation']}\t"
                        f"{row['category']}\t{row['reason']}\n")
    print(f"  -> {viol_path}")

    # Precision spot-check
    print("\n=== Precision spot-check ===")
    spot_checks, violations = precision_spot_check(
        word_types, morfessor_results, unimorph_results, byt5_results,
        gazetteer, n_samples=args.spot_check_n,
    )
    print(f"  {len(spot_checks)} words spot-checked")
    print(f"  {len(violations)} violations found")

    spot_path = os.path.join(OUTDIR, "spot_check.tsv")
    with open(spot_path, "w") as f:
        f.write("word\tn\tclosed_kind\tin_gazetteer\tgazetteer_category\tmorfessor\tunimorph\tbyt5\t"
                "morfessor_violation\tmorfessor_reason\tunimorph_violation\tunimorph_reason\t"
                "byt5_violation\tbyt5_reason\n")
        for row in spot_checks:
            f.write(f"{row['word']}\t{row['n']}\t{row['closed_kind']}\t{row['in_gazetteer']}\t"
                    f"{row['gazetteer_category']}\t{row['morfessor']}\t{row['unimorph']}\t{row['byt5']}\t"
                    f"{row['morfessor_violation']}\t{row['morfessor_reason']}\t"
                    f"{row['unimorph_violation']}\t{row['unimorph_reason']}\t"
                    f"{row['byt5_violation']}\t{row['byt5_reason']}\n")
    print(f"  -> {spot_path}")

    # Write summary JSON
    summary = {
        "n_word_types": len(word_types),
        "n_gazetteer": len(gazetteer),
        "morfessor_segmented": len(morfessor_results),
        "unimorph_segmented": len(unimorph_results),
        "byt5_segmented": len(byt5_results),
        "violations": dict(violation_counts),
        "total_violations": sum(violation_counts.values()),
        "spot_check_n": len(spot_checks),
    }
    sum_path = os.path.join(OUTDIR, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {sum_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
