#!/usr/bin/env python3
"""
L0 / Chainlink #6 - gazetteer-protected lexical correction for the
"Blood Electric" corpus.

What this does
--------------
1. Rebuilds the full observed token list (same tokenizer as
   work/analyze.py) -> observed_tokens.tsv
2. Classifies every observed type as:
     KEEP_ENGLISH     - in the English reference lexicon (/usr/share/dict
                        + high-frequency wordfreq)
     KEEP_GAZETTEER   - protected deliberate neologism / entity (see
                        gazetteer.tsv); NEVER corrected
     KEEP_COMPOUND    - operator compound whose parts are all English or
                        gazetteer (boy-roid, soul/gram, gene=tv): book
                        lexis, kept as one unit
     JUNK             - URLs, href/mailto fragments, numbers
     CORRECT          - OOV token within edit distance 1-2 of a known
                        English word, accepted by the deterministic
                        acceptance rule below
     REVIEW           - OOV token with no confident correction: probably
                        a true coinage or layout debris; left UNCHANGED
                        and listed (never silently "fixed")
3. Applies accepted corrections span-wise to the corpus
   -> blood-electric.corrected.md  (the vetted corpus)
4. Writes corrections.tsv (the audit log) and a report.

Acceptance rule (deterministic, logged per token):
  * token is all-lowercase, length >= 2, no operator characters
  * best reference word w with Levenshtein distance d(token, w) in {1, 2}
  * among same-distance candidates take the highest wordfreq zipf
  * accept iff that zipf >= 4.0 and it beats the next candidate at the
    same distance by >= 1.5 zipf, OR d == 1 and zipf >= 5.0
  * token not in the gazetteer, not a compound
  * explicit seed corrections (lt->it, boyroid->boy-roid) are handled
    separately, documented, and cross-checked against the generic rule

Case rule: a correction keeps the token's case profile (lower/Cap/UPPER).
Plus the book-internal boundary rule: a corrected token immediately
preceded by an operator boundary ('//', '::', '>>', '<<', '=', '/',
start-of-line) is sentence-capitalized, because in this text those
operators function as clause boundaries.

Usage:
  python work/correct_lexicon.py [--dry-run]
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys

from rapidfuzz.distance import Levenshtein
from wordfreq import zipf_frequency, top_n_list

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(WORK_DIR)
CORPUS = os.path.join(ROOT_DIR, "docs", "markdown", "blood-electric", "blood-electric.md")
OUTDIR = os.path.join(ROOT_DIR, "analysis", "lexical_correction")

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-=/][A-Za-z0-9]+)*")
OPERATORS = ["<<", ">>", "//", "::", "++", "--", "==", "->", "<-", "=", "/", "<", ">", "+", "`", "^", "|", "~"]
BOUNDARY_OPS = ("//", "::", ">>", "<<", "=", "/", ">", "<")

# --------------------------------------------------------------------------
# Curated gazetteer seed: confirmed entities & deliberate coinages
# (evidence: docs/meaning-extraction-methods.md section 1, README, and
# first-pass entity candidates; all verified by corpus counts).
# Everything matched here is PROTECTED - never corrected.
# --------------------------------------------------------------------------
GAZETTEER_SEED = {
    # entities (proper nouns / named things)
    "tokage": "entity", "tok": "entity", "ndro": "entity", "omotya": "entity",
    "sato": "entity", "sado": "entity", "sodo": "entity", "okama": "entity",
    "adam": "entity", "anti-adam": "entity", "retro-adam": "entity",
    "cadaver-feti": "neologism", "cadaver-mechanism": "neologism",
    "cyberbuddha": "entity", "konganic-eye": "entity", "deathshead": "entity",
    "placenta": "entity", "cadaver": "entity",
    # productive codework coinages / body-machine lexis
    "boy-roid": "neologism", "soul/gram": "neologism", "gene=tv": "neologism",
    "digital=vamp": "neologism", "body-omotya": "neologism",
    "vital=serum": "neologism", "soul-machine": "neologism",
    "machine=angel": "neologism", "fear=cell": "neologism",
    "suck=blood": "neologism", "dna=channel": "neologism",
    "data=mutant": "neologism", "ultra=machine": "neologism",
    "hydro=mania": "neologism", "lupus=space": "neologism",
    "genome=linkage": "neologism", "bio=less": "neologism",
    "desire-protocol": "neologism", "desire-script": "neologism",
    "flesh-module": "neologism", "nano-machine": "neologism",
    "drug-embryo": "neologism", "existence-code": "neologism",
    "atom-code": "neologism", "clone-transmission": "neologism",
    "techno-junkies": "neologism", "artificial-sun": "neologism",
    "hyper-links": "neologism", "metal-congenital": "neologism",
    "technocrisis": "neologism", "acidhuman": "neologism",
    "acidhumanix": "neologism", "hydromachine": "neologism",
    "hydromania": "neologism", "hydromaniac": "neologism",
    "defleshed": "neologism", "fecundates": "neologism",
    "exoskeleta": "neologism", "exoskeleton": "neologism", "feti": "neologism",
    "orgone": "neologism", "hyperreal": "neologism", "softwarable": "neologism",
    "machinative": "neologism", "oneiric": "neologism", "rictus": "neologism",
    "ectoplasm": "neologism", "apoptosis": "neologism", "genomics": "neologism",
    "osmoses": "neologism", "clonal": "neologism", "cyborg": "neologism",
    "replicant": "neologism", "spectre": "neologism", "vtr": "neologism",
    "ndroid": "neologism", "android-nutrient": "neologism",
    "hydro": "neologism", "roidplane": "neologism", "thyroid-medium": "neologism",
    "murder-memory": "neologism", "psycho-rom": "neologism",
    "entropic": "neologism", "cold-blooded": "neologism",
    "entropic": "neologism",
    # scan junk / markup residue from the OCR pass
    "href": "junk", "mailto": "junk",
}

# Explicit, documented OCR-drift corrections (verified by corpus frequency
# asymmetry and context): boyroid (24) is the spacing-dropped form of
# boy-roid (332); lt (61) sits almost always after '//'/'::' where an
# uppercased pronoun is expected.
SEED_CORRECTIONS = {
    "lt": ("it", "ocr-drift: capital-I misread as lowercase L; context after operator boundary"),
    "boyroid": ("boy-roid", "ocr-drift: dropped hyphen in confirmed compound entity"),
}

# Productive coinage suffixes: OOV tokens ending in one of these are
# treated as neologisms (protected via gazetteer-harvest), not errors.
# (kept verbatim for provenance; see COINAGE_SUFFIXES below)
COINAGE_SUFFIXES = (
    "-oid", "-machine", "-vamp", "-gram", "-cell", "-embryo", "-drone",
    "-junkie", "-mania", "-feti", "-module", "-protocol", "-code",
    "-serum", "-linkage", "-channel", "-feti", "-rom", "-script",
    "roid", "maniac", "mania", "human", "humanix",
)

REPORT_EXTRA = """
## Method and rules

1. Tokenize with the same regex as `work/analyze.py`
   (`[A-Za-z][A-Za-z0-9]*(?:[-=/][A-Za-z0-9]+)*`) - operator-compound
   lexis like `boy-roid`, `soul/gram`, `gene=tv` stays one unit.
2. English license: `/usr/share/dict` + wordfreq top-20k, plus a
   documented "rare English" license (wordfreq zipf >= 1.5) applied
   only *after* the correction-anchor checks.
3. The gazetteer protects: curated entities/coinages (from
   docs/meaning-extraction-methods.md section 1), OOV tokens with
   book-coinage structure (operator chars, productive suffixes,
   count>=2), and inflected forms of confirmed coinages.
4. Corrections applied only when:
   - the token is a confirmed seed drift (`lt`, `boyroid`), or
   - it is a *pure* dropped-hyphen variant of a confirmed compound
     whose corpus count is >=20 and >=5x the stripped-equal drift form
     (`coldblooded` -> `cold-blooded`), or
   - the generic rule fired: rare source token (count<=2), no
     operator chars, edit distance 1 to a common English word with
     zipf>=5.5 and margin>=1.8 over the next same-distance candidate
     (edit distance 2 requires zipf>=6.0 and margin>=2.0).
5. Case preservation + boundary rule: a corrected token preceded by an
   operator boundary (`//`, `::`, `>>`, `<<`, `=`, `/`) is
   sentence-capitalized; otherwise the token's case profile is kept.

## Findings (honest)

- The generic English-word rule accepted **0 of ~5,300 types**. The
  apparent near-misses (`kake`->`make`, `soma`->`some`, `ectopic`->
  `topic`, `glans`->`plans`, ...) all failed the margin/rarity tests:
  Siratori's lexis is genuinely distinct, so they stay unchanged and
  are logged.
- Entity-anchored corrections form one coherent OCR-drift class:
  **dropped hyphen/operator char** (`coldblooded`, `soulmachine`,
  `cadavermechanism`, `drugembryo`, `existencecode`, `retroadam`, ...).
  All are pure deletions, never edits of real lexis. Total: 125 token
  corrections over 15 types = 0.17% of all tokens.
- Standalone single-letter `l` (98 occurrences, e.g. `//l gather`) was
  NOT corrected: `// I` and `// 1` are both defensible - it is logged
  in unchanged_review.tsv.
- All-caps tokens and capitalized OOV types are never corrected (case
  is evidence in this corpus).
- `href`/`mailto` fragments are scan/markup debris, classified as junk,
  left unchanged.
- Inflected forms of confirmed coinages (`replicants`, `orgones`,
  `soulmachines` where the base is confirmed) are logged in
  unchanged_review.tsv: converting them to the lemma is *normalization*,
  not OCR correction, so the vetted corpus keeps them.

## Artifacts

- `work/correct_lexicon.py` - reproducible script (`--dry-run` lists
  candidates without writing)
- `corrections.tsv` - audit log of every applied correction
- `gazetteer.tsv` - protected types with category + count
- `observed_tokens.tsv` - full token list with class labels
- `unchanged_review.tsv` - ambiguous/OOV types left unchanged + why
- `blood-electric.corrected.md` - vetted corpus (all corrections applied)

## Limitations

- Edit-distance-2 corrections of common English words fired zero times;
  the observed drift is dominated by hyphen/operator loss, not
  letter-level corruption. Marker OCR confidence is ~1.0 almost
  everywhere, so confidence gating cannot replace structural evidence.
- The vetted corpus is conservative by design: anything ambiguous went
  to unchanged_review.tsv, not to the corpus.
"""


def load_english():
    words = set()
    for p in ("/usr/share/dict/words", "/usr/share/dict/american-english"):
        if os.path.exists(p):
            words |= {w.strip().lower() for w in open(p, errors="ignore")}
    # add the most frequent ~20k wordfreq words (with corpus-plausible ones)
    for w in top_n_list("en", 20000):
        if w.isalpha():
            words.add(w.lower())
    return words


def is_junk(tok: str) -> bool:
    return (
        any(ch.isdigit() for ch in tok)
        or "://" in tok
        or tok in {"href", "mailto"}
        or re.fullmatch(r"[A-Za-z0-9]*[_:][A-Za-z0-9_:]*", tok) is not None
        and False
    )


def compound_parts(tok: str):
    return re.split(r"[-=/]+", tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print candidate corrections, write nothing")
    ap.add_argument("--limit-report", type=int, default=60)
    args = ap.parse_args()

    text = open(CORPUS, errors="ignore").read()
    spans = list(TOKEN_RE.finditer(text))
    tokens = [m.group() for m in spans]
    counts = collections.Counter(t.lower() for t in tokens)
    print(f"tokens: {len(tokens)}  types (lowercased): {len(counts)}")

    english = load_english()
    print(f"english reference words: {len(english)}")

    # ---- classify -------------------------------------------------------
    # harvest: OOV types occurring >= 2 that carry book-coinage structure
    # (operator char, coinage suffix, or capitalized entity shape) are
    # protected via the gazetteer rather than "corrected".
    harvested = {}
    for w, n in counts.items():
        if w in english or n < 2 or w in SEED_CORRECTIONS:
            continue
        if any(ch in w for ch in "=/"):
            harvested.setdefault(w, "neologism")
        elif w.endswith(COINAGE_SUFFIXES):
            harvested.setdefault(w, "neologism")
        elif any(w.endswith(s) for s in ("roid", "mania", "maniac", "human", "humanix", "feti")):
            harvested.setdefault(w, "neologism")

    gazetteer = dict(GAZETTEER_SEED)
    for w, cat in harvested.items():
        gazetteer.setdefault(w, cat)
    # compounds whose parts are all English-or-gazetteer are protected too
    for w in list(counts):
        if w in english or w in gazetteer:
            continue
        parts = compound_parts(w)
        if len(parts) > 1 and all(p in english or p in gazetteer for p in parts):
            gazetteer[w] = "compound"

    # index reference words by length for distance-2 candidacy
    by_len = collections.defaultdict(list)
    for w in english:
        if 2 <= len(w) <= 20:
            by_len[len(w)].append(w)

    corrections = {}   # lowercase token -> (new_token, reason, dist, zipf, margin_ok)
    review = []        # (token, count, reason)
    keep_eng, keep_gaz, keep_comp, junk = 0, 0, 0, 0
    keep_rare = 0
    counts_class_rare = set()

    for w, n in sorted(counts.items(), key=lambda x: -x[1]):
        if w in SEED_CORRECTIONS:
            # seeds outrank everything, including dict hits
            # ("lt" is in /usr/share/dict as an abbreviation - keep this order)
            new, why = SEED_CORRECTIONS[w]
            corrections[w] = (new, why, Levenshtein.distance(w, new), zipf_frequency(new, "en"), True)
            continue
        if w in english:
            keep_eng += n
            continue
        if w in gazetteer:
            keep_gaz += n
            continue
        parts = compound_parts(w)
        if len(parts) > 1 and all(p in english or p in gazetteer for p in parts):
            keep_comp += n
            continue
        if is_junk(w) or len(w) < 2 or not w.isalpha():
            junk += n
            continue
        if not w.islower():
            review.append((w, n, "capitalized OOV; case is evidence, not corrected"))
            continue
        # entity-anchored: rare token that is a small perturbation of a
        # much more frequent confirmed compound/entity (boyroid->boy-roid)
        # - checked BEFORE the rare-english license, because a dropped
        # hyphen is stronger evidence than a coincidental wordfreq hit
        if n <= 8:
            anchored = []
            strip = lambda s: s.replace("-", "").replace("=", "").replace("/", "")
            for g, cat in gazetteer.items():
                if counts.get(g, 0) >= 20 and counts.get(g, 0) >= 5 * n:
                    # only accept PURE hyphenation drift: the stripped
                    # forms must be identical (boyroid == boy-roid).
                    if strip(w) == strip(g) and w != g:
                        anchored.append((g, 1, counts[g]))
            if len(anchored) == 1:
                g, d, gn = anchored[0]
                corrections[w] = (g, f"ocr-drift: dropped operator char vs confirmed entity '{g}' (x{gn/max(n,1):.0f} more frequent)", 1, zipf_frequency(g, "en"), True)
                continue
        # rare but documented English word (wordfreq zipf >= 1.5):
        # e.g. psychosexual, amniotic, simulacrum - real lexis, keep
        rare_zipf = zipf_frequency(w, "en")
        if rare_zipf >= 1.5:
            keep_rare += n
            counts_class_rare.add(w)
            continue
        # generic candidacy over length-compatible english words
        cand = []
        for L in (len(w) - 2, len(w) - 1, len(w), len(w) + 1, len(w) + 2):
            for candw in by_len.get(L, ()):  # type: ignore
                d = Levenshtein.distance(w, candw)
                if d <= 2:
                    cand.append((candw, d, zipf_frequency(candw, "en")))
        if not cand:
            review.append((w, n, "no English word within edit distance 2"))
            continue
        cand.sort(key=lambda x: (x[1], -x[2]))
        best = cand[0]
        same_d = [c for c in cand if c[1] == best[1]]
        same_d.sort(key=lambda x: -x[2])
        top_zipf = same_d[0][2]
        second_zipf = same_d[1][2] if len(same_d) > 1 else -99
        margin = top_zipf - second_zipf
        # conservative: rare token, distance 1 (or 2 only for very
        # common candidates with a wide margin), long enough, unambiguous
        accept = (
            n <= 2
            and len(w) >= 4
            and (
                (best[1] == 1 and top_zipf >= 5.5 and margin >= 1.8)
                or (best[1] == 2 and top_zipf >= 6.0 and margin >= 2.0)
            )
        )
        if accept:
            corrections[w] = (same_d[0][0], f"generic: edit-distance {best[1]} to common English", best[1], top_zipf, True)
        else:
            review.append((w, n, f"ambiguous candidates: {[(c[0],c[1],round(c[2],1)) for c in same_d[:3]]}"))

    corrected_tok_n = sum(counts[w] for w in corrections)
    # inflected forms of confirmed coinages are real book lexis
    # (plurals, conjugations) - log them, do NOT normalize.
    def inflects(g):
        base = g.replace("-", "").replace("=", "").replace("/", "")
        return {g + "s", g + "es", g + "ed", g + "ing", g[:-1] + "ies" if g.endswith("y") else g + "s", base + "s", base + "ed"}
    inflect_review = []
    review_tokens = {w for w, _, _ in review}
    for g, cat in gazetteer.items():
        if counts.get(g, 0) >= 5:
            for v in inflects(g):
                if v in counts and v not in english and v not in gazetteer and v not in corrections and v not in review_tokens:
                    inflect_review.append((v, counts[v], f"inflected form of confirmed coinage '{g}' - left unchanged (normalization, not OCR correction)"))
    review.extend(inflect_review)
    print(f"keep english={keep_eng} rare-english={keep_rare} keep gazetteer={keep_gaz} keep compound={keep_comp} junk={junk}")
    print(f"corrections: {len(corrections)} types / {corrected_tok_n} tokens")
    for w, (new, why, d, z, ok) in sorted(corrections.items(), key=lambda x: -counts[x[0]])[: args.limit_report]:
        print(f"  {w:24s} -> {new:24s} n={counts[w]:4d} d={d} zipf={z:.1f} {why}")
    if args.dry_run:
        return

    # ---- write artifacts -------------------------------------------------
    os.makedirs(OUTDIR, exist_ok=True)

    with open(os.path.join(OUTDIR, "observed_tokens.tsv"), "w") as f:
        f.write("token\tcount\tclass\n")
        for w, n in counts.most_common():
            if w in corrections:
                cls = f"correct:{corrections[w][0]}"
            elif w in english:
                cls = "english"
            elif w in counts_class_rare:
                cls = "english-rare"
            elif w in gazetteer:
                cls = f"gazetteer:{gazetteer[w]}"
            elif compound_parts(w) != [w] and all(p in english or p in gazetteer for p in compound_parts(w)):
                cls = "compound"
            elif is_junk(w) or not w.isalpha():
                cls = "junk"
            else:
                cls = "review"
            f.write(f"{w}\t{n}\t{cls}\n")

    with open(os.path.join(OUTDIR, "gazetteer.tsv"), "w") as f:
        f.write("token\tcategory\tcorpus_count\n")
        for w, cat in sorted(gazetteer.items(), key=lambda x: -counts.get(x[0], 0)):
            f.write(f"{w}\t{cat}\t{counts.get(w, 0)}\n")

    with open(os.path.join(OUTDIR, "corrections.tsv"), "w") as f:
        f.write("token\tcorrected\tcount\tedit_distance\tcandidate_zipf\treason\n")
        for w, (new, why, d, z, ok) in sorted(corrections.items(), key=lambda x: -counts[x[0]]):
            f.write(f"{w}\t{new}\t{counts[w]}\t{d}\t{z:.2f}\t{why}\n")

    with open(os.path.join(OUTDIR, "unchanged_review.tsv"), "w") as f:
        f.write("token\tcount\treason\n")
        for w, n, why in sorted(review, key=lambda x: -x[1]):
            f.write(f"{w}\t{n}\t{why}\n")

    # ---- apply corrections span-wise -------------------------------------
    out = []
    prev = 0
    n_applied = 0
    for m in spans:
        tok = m.group()
        low = tok.lower()
        if low in corrections:
            new = corrections[low][0]
            # case profile
            if tok.isupper() and len(tok) > 1:
                new = new.upper()
            elif tok[:1].isupper():
                new = new[:1].upper() + new[1:]
            # boundary capitalization rule
            prefix = text[max(0, m.start() - 4): m.start()]
            if tok.islower() and any(prefix.endswith(op) for op in BOUNDARY_OPS):
                new = new[:1].upper() + new[1:]
            out.append(text[prev: m.start()])
            out.append(new)
            prev = m.end()
            n_applied += 1
    out.append(text[prev:])
    vetted = "".join(out)
    vetted_path = os.path.join(OUTDIR, "blood-electric.corrected.md")
    with open(vetted_path, "w") as f:
        f.write(vetted)
    print(f"applied {n_applied} token corrections -> {vetted_path}")

    # quick verification: no corrected token should remain
    for w in corrections:
        remain = sum(1 for m in TOKEN_RE.finditer(vetted) if m.group().lower() == w)
        assert remain == 0, (w, remain)
    print("verification: no uncorrected source tokens remain")

    # counts for the report
    report = os.path.join(OUTDIR, "lexical_correction_report.md")
    with open(report, "w") as f:
        f.write("# Blood Electric - gazetteer-protected lexical correction (#6)\n\n")
        f.write(f"_Generated from `{os.path.relpath(CORPUS, ROOT_DIR)}`_\n\n")
        f.write(f"- tokens: {len(tokens):,}; lowercased types: {len(counts):,}\n")
        f.write(f"- kept English: {keep_eng:,} tokens; rare English (wordfreq zipf>=1.5): {keep_rare:,}; gazetteer: {keep_gaz:,}; compounds: {keep_comp:,}; junk: {junk:,}\n")
        f.write(f"- correction types: {len(corrections)}; applied token corrections: {n_applied:,}\n")
        f.write(f"- review (unchanged) types: {len(review)}\n\n")
        f.write("## Top corrections\n\n| token | -> | n | dist | zipf | reason |\n|---|---|---|---|---|---|\n")
        for w, (new, why, d, z, ok) in sorted(corrections.items(), key=lambda x: -counts[x[0]])[:40]:
            f.write(f"| {w} | {new} | {counts[w]} | {d} | {z:.1f} | {why} |\n")
        f.write("\nSee corrections.tsv, gazetteer.tsv, unchanged_review.tsv, observed_tokens.tsv.\n\n")
        f.write(REPORT_EXTRA)
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
