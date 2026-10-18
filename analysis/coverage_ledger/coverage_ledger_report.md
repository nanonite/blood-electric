# Coverage accounting / residue ledger (#70)

Does the analysis account for every token and every operator, or does it quietly drop the awkward residue?  One ledger is built over the whole corpus at four levels, and each level is forced to close.

**Verdict: it accounts for everything, and the residue is itemized.** Every one of the 480,962 page-body characters, all 90,220 tokens, all 11,000 operator tokens and all 74,665 unit occurrences land in a named bucket, so the coverage metric is 1.000000 at every level and the build fails if it ever drops.  The awkward part is therefore not hidden residue but *named* residue: 11 classes across the levels -- 623 characters of layout scaffolding and unclassifiable punctuation, 937 operator tokens that bind nothing, and 1,266 unit-occurrence entries (one class deliberately overlaps another: the 202 `href`/`mailto` occurrences are #19 `junk` *and* #27 tag-run furniture).  Two classes are itemized by no artifact until this ledger: the 13 occurrences of real book text that #27's tag-run screen deletes, and the 2 unit surfaces that #17 and #19 store with a different number of backslashes.

## 1. The coverage metric

| level | total | covered | unexplained | coverage |
|---|---:|---:|---:|---:|
| L1 characters | 480,962 | 480,962 | 0 | 1.000000 |
| L2 tokens | 90,220 | 90,220 | 0 | 1.000000 |
| L3 operator tokens | 11,000 | 11,000 | 0 | 1.000000 |
| L4 unit occurrences | 74,665 | 74,665 | 0 | 1.000000 |

Coverage counts a bucket as covered when the ledger can *name* it, including the residue classes an artifact owns: naming a removal is what makes it auditable.  Only the unnamed and the unclaimed count against coverage, and on this corpus there are none.  The metric is a closure test, not a score -- a level that silently lost 0.1% of its items would still read 0.999000, which is why the tests plant residue and require the detector to fire (`work/test_coverage_ledger.py`).

## 2. What the buckets are

### L1 characters -- 480,962 page-body characters

| bucket | chars | share | status |
|---|---:|---:|---|
| `gap_whitespace` | 71,021 | 14.766% | accounted |
| `layout_comment` | 468 | 0.097% | accounted |
| `token_op` | 17,269 | 3.591% | accounted |
| `token_punct_bang` | 1 | 0.000% | accounted |
| `token_punct_bracket` | 367 | 0.076% | accounted |
| `token_punct_colon` | 324 | 0.067% | accounted |
| `token_punct_comma` | 15 | 0.003% | accounted |
| `token_punct_dash` | 715 | 0.149% | accounted |
| `token_punct_other` | 155 | 0.032% | accounted |
| `token_punct_period` | 22 | 0.005% | accounted |
| `token_punct_quote` | 136 | 0.028% | accounted |
| `token_word` | 390,469 | 81.185% | accounted |

The page body is reconstructed from #17's artifacts alone (`"".join(gap + text)` + `page_tails.tsv`), and both directions are asserted before the ledger is built: the reconstruction equals #7's layout markdown on all 225 pages, and #17's `tokens.tsv` equals what `operator_tokenizer.tokenize_page` emits.  A ledger over a drifting stream would be worthless.

The only non-token, non-whitespace characters in the book are the 12 figure-block scaffolding comments (468 chars on pages 1, 224, 225).  #47's char total (480,309) is reconcilable rather than merely different: it replaces each comment with one space (468 chars -> 12), strips the markdown heading markers (197 chars) and case-folds.

### L2/L3 tokens and operator roles

| level | bucket | n | share |
|---|---|---:|---:|
| L2 tokens | `token_OP` | 11,000 | 12.192% |
| L2 tokens | `token_PUNCT` | 1,735 | 1.923% |
| L2 tokens | `token_WORD` | 77,485 | 85.885% |
| L3 operator tokens | `seam` | 2,816 | 25.600% |
| L3 operator tokens | `seam_orphan_left` | 253 | 2.300% |
| L3 operator tokens | `seam_orphan_right` | 218 | 1.982% |
| L3 operator tokens | `seam_orphan_side` | 4 | 0.036% |
| L3 operator tokens | `segment` | 7,247 | 65.882% |
| L3 operator tokens | `slash_boundary` | 3 | 0.027% |
| L3 operator tokens | `slash_free` | 459 | 4.173% |

The token partition reproduces #14's 77,485 Lex nodes and #17's 11,000 operators (10,949 body + 51 header).  Every one of the 11,000 operator tokens carries exactly one #17 role, so the operator level closes by construction: 2,816 seams and 7,247 segment operators that do work, and 937 that bind nothing.

### L4 unit occurrences

| bucket | occurrences | share |
|---|---:|---:|
| `compound` | 6,736 | 9.022% |
| `english` | 66,506 | 89.073% |
| `junk` | 202 | 0.271% |
| `numeric` | 73 | 0.098% |
| `ocr_error` | 127 | 0.170% |
| `proper_noun` | 287 | 0.384% |
| `rare_english` | 49 | 0.066% |
| `true_coinage` | 338 | 0.453% |
| `markup_ledger_occurrence` | 347 | 0.465% |

74,318 occurrences carry a #19 category and 347 sit in #19's per-occurrence markup ledger: `74,318 + 347 = 74,665`.  At the type level `5,868 + 8 = 5,876`.  Both directions close, and neither closes on a surface-only join -- that is the point of §3.

## 3. The residue ledger

| class | level | n | share | verdict | owner |
|---|---|---:|---:|---|---|
| layout scaffolding comments | character | 468 | 0.0973% | **disclosed** | #17 page_tails.tsv (tail spans); this ledger (all spans) |
| punctuation in no operator class | character | 155 | 0.0322% | **explained** | #17 PUNCT tokens (pclass `other`); #47 strips them (197 chars) before its char stream |
| operator tokens that bind nothing | operator token | 937 | 8.5180% | **explained** | #17 diagnostics.tsv (seam_orphan_* / slash_free / slash_boundary rows) |
| markup-residue unit occurrences | unit occurrence | 347 | 0.4647% | **explained** | #19 markup_residue.tsv (347 rows, one per occurrence) |
| markup-only unit types | unit type | 145 | 0.1942% | **explained** | #19 counts.markup_only_types_dropped (8) + this ledger per type |
| markup occurrences on a surviving surface | unit occurrence | 202 | 0.2705% | **explained** | #19 markup_residue.tsv (uid level) |
| tag furniture kept as English by the screen | unit occurrence | 202 | 0.2705% | **explained** | #19 screening.tsv category `junk`; #27 counts.markup_extra_removed (241) removes them again |
| attribute names and values inside tag runs | unit occurrence | 228 | 0.3054% | **explained** | #27 find_markup_spans (the general tag-run detector) |
| book text deleted by the tag-run screen | unit occurrence | 13 | 0.0174% | **disclosed** | no artifact; itemized by this ledger |
| artifact escape mismatch | unit occurrence | 2 | 0.0027% | **disclosed** | this ledger's normalisation; the mismatch is #19's writer |
| OCR-error units kept in the unit layer | unit occurrence | 127 | 0.1701% | **explained** | #6 corrections + #19 category `ocr_error` |

- **layout scaffolding comments** (character, 468 occurrences). 12 `<!-- figure/picture block (no text) -->` spans, 6 in page tails / 6 between tokens  
  Marker's figure-block placeholders. The tokenizer emits no tokens inside them, so they survive only as gap text; #17's spec discloses them qualitatively ('whitespace only except for the two figure-block comments') and never lists them. Six of the twelve sit between tokens.  
  _disclosed; pages: 1, 224, 225_
- **punctuation in no operator class** (character, 155 occurrences). '#' x141, '\\' x8, '©' x1, '•' x5  
  Counted, and none of it is deleted: #47's char total (480,309) is this ledger's raw total (480,962) less the 468 scaffolding chars, less the 197 marker chars, plus 12 replacement spaces.  
  _explained; pages: 1, 2, 3, 4, 9, 11, 12, 15 ... (54 pages)_
- **operator tokens that bind nothing** (operator token, 937 occurrences). seam_orphan_left 253, seam_orphan_right 218, seam_orphan_side 4, slash_free 459, slash_boundary 3  
  A seam operator with no word on one side, or a `/` that is not glued to the right. Kept as OP tokens with exact offsets, never repaired: the tokenizer deletes nothing.  
  _explained; no page list_
- **markup-residue unit occurrences** (unit occurrence, 347 occurrences). #19 markup ledger, uid level: a x201, th x95, p x29, pre x10, li x4, br x4, body x1, fluid=streaming x1, Pre x1, hr x1  
  Every occurrence is published with its uid, so the removal is per-occurrence and auditable; a surface-level join cannot reproduce it.  
  _explained; pages: 9, 11, 13, 17, 18, 19, 20, 23 ... (82 pages)_
- **markup-only unit types** (unit type, 145 occurrences). Pre x1, br x4, fluid=streaming x1, hr x1, li x4, p x29, pre x10, th x95  
  Surfaces whose every occurrence is markup, so the type leaves the 5,876-type vocabulary entirely.  
  _explained; pages: 9, 11, 13, 17, 18, 19, 20, 23 ... (80 pages)_
- **markup occurrences on a surviving surface** (unit occurrence, 202 occurrences). a x201 of 1126, body x1 of 542  
  The hazard this ledger exists for: #17's unit_types.tsv advertises `a` 1,126 times on 214 pages, but 201 of those are tag names. Only the uid join removes them.  
  _explained; pages: 4, 5, 6, 7, 8, 9, 10, 11 ... (218 pages)_
- **tag furniture kept as English by the screen** (unit occurrence, 202 occurrences). href x102, mailto x100  
  Two artifacts disagree about the same 202 occurrences by design: #19 publishes them, #27 screens them out.  
  _explained; pages: 92_
- **attribute names and values inside tag runs** (unit occurrence, 228 occurrences). '2' x13, 'colspan' x13, 'href' x102, 'mailto' x100  
  Invisible to #19's screen, which matches whole tags and not the runs they sit in. The 202 `href`/`mailto` occurrences overlap the `junk` row above (the same occurrences seen from #19's side); the 26 `colspan`/`2` occurrences are #27's alone.  
  _explained; pages: 47, 92, 158, 209, 219_
- **book text deleted by the tag-run screen** (unit occurrence, 13 occurrences). atrocity x1, cyberBuddha x3, genomics x1, konganic-eye x3, of x4, retro-ADAM x1  
  The only place in the workspace where a screen removes real text rather than markup: the OCR wrapped printed book text in `<a href="...">`, and #27's span rule deletes the anchor's href value with the tag. On p219 the deleted text is a duplicate of the visible anchor text that follows, and on p158 likewise; whether these 13 occurrences should be restored (span narrowing, not a wider screen) is #6/#19's call.  
  _disclosed; pages: 158, 219_
- **artifact escape mismatch** (unit occurrence, 2 occurrences). eruption=\njunk (p102, uid 34085), device++\nmass (p209, uid 69267)  
  Two unit surfaces cross a newline after a seam. #17 writes the surface escaped once; #19 reads it back with `csv.DictReader` (no unescaping) and writes it with a writer that escapes the backslash again, so `screening.tsv` stores two backslashes where `units.tsv` stores one. A surface-level join across the two artifacts loses exactly these 2 occurrences, which is the concrete reason #17 says to join on `uid`. This ledger normalises the escape and closes the level without it.  
  _disclosed; pages: 102, 209_
- **OCR-error units kept in the unit layer** (unit occurrence, 127 occurrences). 15 types, lt, boyroid, cadavermechanism, soulmachine, coldblooded, desireprotocol  
  Residue that is *kept* rather than dropped: #6's fixes are applied in the lexical-correction layer, not in the unit layer, so every downstream statistic still sees them.  
  _explained; pages: 4, 5, 11, 15, 16, 25, 30, 31 ... (14 pages)_

`explained` means a published artifact names the items; `disclosed` means the workspace counts them but itemizes them nowhere.  There is no `unexplained` row, and the build fails if one ever appears.

## 4. The one screen that deletes book text

#27 removes #19's 347 ledger occurrences and 241 more through its general tag-run detector.  The ledger decomposes those 241: 228 are tag furniture (`href` 102, `mailto` 100, `colspan` 13 and the `colspan` value `2` 13) and **13 are real book text** that the OCR wrapped in a fabricated anchor:

| page | offset | unit | context |
|---:|---:|---|---|
| 158 | 2382 | `atrocity` | `circuit of the mass of flesh-module where I was imprisoned::<<a href="atrocity` |
| 158 | 2391 | `genomics` | `f the mass of flesh-module where I was imprisoned::<<a href="atrocity genomics` |
| 158 | 2400 | `of` | `s of flesh-module where I was imprisoned::<<a href="atrocity genomics of retro` |
| 158 | 2403 | `retro-ADAM` | `f flesh-module where I was imprisoned::<<a href="atrocity genomics of retro-AD` |
| 219 | 304 | `konganic-eye` | ` inorganic substance murder++the technology of murder<<a href="mailto:konganic` |
| 219 | 317 | `of` | `bstance murder++the technology of murder<<a href="mailto:konganic-eye of cyber` |
| 219 | 320 | `cyberBuddha` | `ance murder++the technology of murder<<a href="mailto:konganic-eye of cyberBud` |
| 219 | 586 | `konganic-eye` | ` inorganic substance murder++the technology of murder<<a href="mailto:konganic` |
| 219 | 599 | `of` | `bstance murder++the technology of murder<<a href="mailto:konganic-eye of cyber` |
| 219 | 602 | `cyberBuddha` | `ance murder++the technology of murder<<a href="mailto:konganic-eye of cyberBud` |
| 219 | 868 | `konganic-eye` | ` inorganic substance murder++the technology of murder<<a href="mailto:konganic` |
| 219 | 881 | `of` | `bstance murder++the technology of murder<<a href="mailto:konganic-eye of cyber` |
| 219 | 884 | `cyberBuddha` | `ance murder++the technology of murder<<a href="mailto:konganic-eye of cyberBud` |

The printed page reads `<<a href="atrocity genomics of retro-ADAM">atrocity genomics of retro-ADAM` (p158) and `<<a href="mailto:konganic-eye of cyberBuddha">konganic-eye of cyberBuddha` (p219, three times): the anchor's href *value* is a copy of the visible text that follows, which is why deleting it is defensible.  But the deletion is never itemized, and the 74,318 vs 74,077 gap that #27 publishes as "241 tag-furniture occurrences" is 13 text occurrences larger than that description.  The fix is span narrowing (screen the tag, keep the href value) and it is #6/#19's call, not this ledger's.

## 5. Every downstream layer's split

| issue | layer | level total | kept | dropped | kept % | drop published in |
|---|---|---:|---:|---:|---:|---|
| #17 tokenizer | unit occurrence | 74,665 | 74,665 | 0 | 100.000% | the raw layer; nothing removed |
| #19 neologism screen | unit occurrence | 74,665 | 74,318 | 347 | 99.535% | #19 markup_residue.tsv, occurrence level (uid) |
| #24 frequency / Zipf | unit occurrence | 74,665 | 74,318 | 347 | 99.535% | #19 markup ledger |
| #28 static embeddings | unit occurrence | 74,665 | 74,318 | 347 | 99.535% | #19 markup ledger |
| #30 entity induction | unit occurrence | 74,665 | 74,077 | 588 | 99.213% | #19 ledger (347) + #27 tag runs (241) |
| #27 collocation | unit occurrence | 74,665 | 74,077 | 588 | 99.213% | #27 counts.markup_units_removed |
| #33 clauses | unit occurrence | 74,665 | 74,077 | 588 | 99.213% | #33 markup_units_removed |
| #35 n-gram surprisal | unit occurrence | 74,665 | 74,077 | 588 | 99.213% | #19 + #27 screen |
| #47 entropy | unit occurrence | 74,665 | 74,077 | 588 | 99.213% | #47 reconciliation.tsv |
| #56 language viability | unit occurrence | 74,665 | 74,077 | 588 | 99.213% | #19 + #27 screen |
| #56 keyness | word part | 80,363 | 79,770 | 593 | 99.262% | #56 book.dropped (markup-screened parts) |
| #27 collocation | token | 90,220 | 88,669 | 1,551 | 98.281% | #27 counts.markup_tokens_removed |
| #51 periodicity | token | 90,220 | 90,220 | 0 | 100.000% | the full token stream, nothing removed |
| #22 case / emphasis | word token | 77,485 | 77,485 | 0 | 100.000% | every word token, screened in place |

Each layer keeps a different share of the same 74,665 units, and each records its own drop.  The ledger's contribution is that the arithmetic is now checked in one place: `kept + dropped = level total` for all 14 rows, and a layer that changed its split without saying so fails this build.

## 6. Reconciliation

Every count this ledger re-derives, against the issue that published it.  A mismatch fails the build.

| quantity | published | here | owner | match |
|---|---:|---:|---|---|
| pages | 225 | 225 | #7 / #17 | ok |
| body characters | 480,962 | 480,962 | sum of #7 page-body lengths | ok |
| tokens | 90,220 | 90,220 | #14 / #17 | ok |
| word tokens | 77,485 | 77,485 | #14 Lex nodes / #22 | ok |
| operator tokens | 11,000 | 11,000 | #14 10,949 body + 51 header | ok |
| punctuation tokens | 1,735 | 1,735 | #17 | ok |
| operator role `seam` | 2,816 | 2,816 | #17 | ok |
| operator role `segment` | 7,247 | 7,247 | #17 | ok |
| unbound operator tokens | 937 | 937 | #17 unbound_ops | ok |
| unit occurrences | 74,665 | 74,665 | #17 units.total | ok |
| screened unit occurrences | 74,318 | 74,318 | #19 counts.screened_occurrences | ok |
| markup-ledger occurrences | 347 | 347 | #19 counts.markup_total_occurrences | ok |
| markup-only unit types | 8 | 8 | #19 counts.markup_only_types_dropped | ok |
| tag-run extra removals | 241 | 241 | #27 counts.markup_extra_removed | ok |
| book text inside tag runs | 13 | 13 | this ledger (unpublished elsewhere) | ok |
| scaffolding comment chars | 468 | 468 | #17 page_tails.tsv + gaps | ok |
| scaffolding comment spans | 12 | 12 | #17 spec, qualitative only | ok |
| #47 raw char total | 480,309 | 480,309 | #47 reconciliation.tsv (comments -> space, markers stripped) | ok |
| #47 heading-marker chars | 197 | 197 | #47 clean_page() | ok |

## 7. Method

Inputs are frozen artifacts only -- #7's layout markdown, #17's token/unit streams, #19's screen and markup ledger, #27's tag-run detector, and the published `summary.json` of every downstream layer. No OCR is run, nothing is downloaded and no reference corpus is consulted, so this ledger adds no new external dependency to the program (#78-#80 stay parked).  Character buckets come from one pass over each page body: layout comments win over tokens (the tokenizer emits nothing inside them), then token text by kind and #17's punctuation class, then the gap.  Operator roles are re-derived with #17's own `_is_seam` rule rather than read from its summary, so the published partition is tested, not assumed.

## 8. Limits

- **The ledger names; it does not repair.**  Nothing here corrects a corpus, changes a screen or re-opens a closed decision.  Where the verdict is `disclosed` the recommendation is an artifact, not a fix.
- **`explained` is not `correct`.**  #19 keeping `href`/`mailto` as `junk` and #27 removing them are both defensible; the ledger only makes the disagreement visible in one place.
- **Coverage is level-local.**  A level at 1.000000 says nothing about whether the *reading* built on it is right; that is #73's triangulation and #56's viability assessment.
- **The figure-only pages stay in.**  Pages 1, 224 and 225 carry 10, 2 and 2 tokens; every per-page statistic includes them, and this ledger does not decide whether it should.

## 9. Artifacts

| file | contents |
|---|---|
| `coverage_ledger_report.md` | THE DELIVERABLE |
| `coverage.tsv` | the ledger: every bucket at every level |
| `residue_ledger.tsv` | the awkward residue, itemized and verdicted |
| `per_page.tsv` | per-page accounting (where the residue lives) |
| `consumer_reconciliation.tsv` | every downstream layer's split |
| `reconciliation.tsv` | this ledger vs #7/#14/#17/#19/#27/#47 |
| `diagnostics.tsv` | build counters |
| `summary.json` | machine-readable headline numbers |
