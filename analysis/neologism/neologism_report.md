# Neologism detection (#19) — `neologisms.tsv`

Source: `analysis/tokenizer/units.tsv` (#17) joined to
`analysis/morphology/decompositions.tsv` (#18) on `uid`; English lexicon =
`wordfreq`'s `en` list (321,180 entries) with a Zipf floor of 1.5 (the same
lexicon #18 used). Reproducible via `python work/neologism_detection.py`;
regression suite `python work/test_neologism_detection.py` (32 tests).

Schema `neologism-detection/1`.

## 1. Headline result

Of the **5,868 screened unit types / 74,318 occurrences** (after markup
residue is removed), **2,305 types / 7,488 occurrences (2,260 case-folded
types) are not ordinary English words**. They split as the issue asks:

| class | types | occurrences | what it is |
|---|---:|---:|---|
| `compound` (of English parts) | 2,216 | 6,736 | `gene=TV`, `soul/gram`, `boy-roid`, `acidhuman` |
| `true_coinage` (opaque) | 51 | 338 | `exoskeleta`, `hydromania`, `rhythmus`, `feti` |
| `proper_noun` (opaque, named) | 23 | 287 | `ToKAGE`, `OMOTYA`, `NDRO`, `dustNirverna` |
| `ocr_error` | 15 | 127 | `boyroid`, `lt`, `coldblooded`, `cadaverfeti` |
| *(excluded)* `numeric` | 18 | 73 | `666`, `1-84068-060-1`, `4-D` — handed to #21 |
| *(excluded)* `rare_english` | 15 | 49 | in wordfreq below Zipf 1.5 (`exoskeletal`, `morbus`) |
| *(excluded)* `english` / `junk` | 3,528 / 2 | 66,506 / 202 | ordinary English; `mailto`/`href` scan debris |

`neologisms.tsv` is the 2,305-row candidate file (`is_candidate=y`); the full
5,868-row screen with every flag is `screening.tsv`.

## 2. Screen units, not words

A word-frequency list cannot see a fused unit. **2,046 unit types are
open-juncture** (`gene=TV`, `soul/gram`, `suck=blood`, `murder++the` …) and
**1,292 of them contain a binding seam** — #17's "1,293 unit types are opaque
to a word-frequency list", less `fluid=streaming`, which is markup. A standard
tokenizer (`wordfreq.simple_tokenize`, sklearn's `[A-Za-z]+`) keeps **0 %** of
these whole. Screening word parts instead of units would therefore discard the
entire compound layer.

The join is on `uid` and is asserted byte-exact on `page+start+end`; 74,665
units joined, no mismatch. Aggregation to types is case-preserved (the unit
layer's own contract), then a case-folded view is also written
(`neologisms_lower.tsv`) so counts are comparable to word-level reports.

## 3. Markup residue: tag names never enter as words

#17 flags **229** HTML-tag-residue occurrences (`</a>`×99, `<th>`×79,
`</p>`×28 …); #18 widens that to **231** with `<body fluid=streaming>`.
#19 removes those **and 116 more the strict patterns miss** — opening tag
names carrying an attribute, `<a href="mailto:…">` and `<th colspan="2">`
(mainly page 92's mailto block and pages 47/209's tables) — for a total of
**347 markup occurrences over 82 pages**. Eight types are markup through and
through (`p`, `pre`, `br`, `li`, `hr`, `Pre`, `fluid=streaming`, `th`) and are
dropped from the screen; `a` and `body` keep their hundreds of real English
occurrences. Full ledger: `markup_residue.tsv`.

The rule is deliberately conservative. A bare leading `<` is *not* enough for
a real English word, because the book writes its own angle-bracket blocks
(`<a defleshed memory//`, p34; `<A lapse of memory>`); a tag-shaped tail
(`>`, `/>`, `attribute=`) or a non-word tag name (`th`, Zipf 4.2 but only
ever inside tags) is required.

## 4. The three classes the issue asks for

### 4.1 Compounds of English parts (2,216 types)

Every unit whose decomposition is an open junction (R1–R6) or a closed
composite/blend, once the whole surface is out of lexicon. Breakdown by
`compound_kind`:

| kind | types | example |
|---|---:|---|
| `open_chain` | 2,040 | `gene=TV`, `soul/gram`, `boy-roid`, `murder++the` |
| `english_suffixation` | 86 | `defleshed` = `de`+`flesh`+`-ed` |
| `closed_composite` | 76 | `acidhuman` = `acid`+`human` |
| `closed_blend` | 14 | `machinative` = `machina`+`ive` (overlap `t`) |

2,085 of the candidates are built entirely from English parts
(`parts_english=y`); 141 embed an opaque part (`body-OMOTYA`, `cadaver-feti`).
The top of the frequency list is the book's naming system, exactly as #12/#18
predicted:

| unit | occ | rule | decomposition | first attestation |
|---|---:|---|---|---|
| `boy-roid` | 325 | R1 | `boy+roid` | p4 offset 387 |
| `soul/gram` | 313 | R3 | `soul+gram` | p4 offset 401 |
| `fecundates` | 289 | R5 | `fecund`+`-ates` | p7 offset 2086 |
| `cold-blooded` | 192 | R1 | `cold+blooded` | p6 offset 299 |
| `retro-ADAM` | 166 | R1 | `retro+ADAM` | p87 offset 1224 |
| `gene=TV` | 144 | R2 | `gene+TV` | p4 offset 1458 |
| `vital=serum` | 128 | R2 | `vital+serum` | p15 offset 693 |
| `suck=blood` | 120 | R2 | `suck+blood` | p4 offset 77 |

### 4.2 True coinages (51) and named coinages (23)

Opaque closed words no rule resolves. The 74 coinage types / 625 occurrences
are dominated by a handful of load-bearing terms:

| coinage | occ | kind | first attestation |
|---|---:|---|---|
| `ToKAGE` | 127 | proper_noun (`entity`) | p4 offset 537 |
| `feti` | 88 | true_coinage (`neologism`) | p18 offset 1684 |
| `hydromania` | 60 | proper_noun | p16 offset 575 |
| `hydromaniac` | 57 | true_coinage | p28 offset 931 |
| `oneiric` | 55 | true_coinage | p17 offset 1394 |
| `exoskeleta` | 54 | true_coinage | p4 offset 1908 |
| `NDRO` / `NDROID` | 23 / 20 | proper_noun | p37 / p13 |
| `OMOTYA` | 1 | proper_noun | p177 offset 1645 |
| `dustNirverna` | 15 | proper_noun | p14 |

`feti` and `oneiric` are in wordfreq only below Zipf 1.5; #6's gazetteer
already names them book coinages, and #19 honours that (`true_coinage`) rather
than calling them ordinary rare English.

### 4.3 OCR errors (15 types / 127 occurrences)

Two evidence routes, both auditable in `ocr_evidence.tsv`:

- **#6-confirmed** (14 types): `lt`→`it` (60×), `boyroid`→`boy-roid` (24×),
  `cadavermechanism`, `soulmachine`, `coldblooded`, `desireprotocol`,
  `technojunkies`, `drugembryo`, `metalcongenital`, `clonetransmission`,
  `desirescript`, `existencecode`, `fleshmodule`, `nanomachine`.
- **Structural dropped-linker** (new here): a closed fused form whose
  delinkered letters are a far more frequent open form. `cadaverfeti` (3×) →
  `cadaver-feti` (91×). The bar is #6's: the open form must have ≥20 uses and
  dominate ≥5×. A real closed coinage fails it — `hydromachine` (62×) beats
  `Hydro-machine` (2×), so it stays a compound.

A further **28 weak suspects** are flagged (`ocr_suspect`, no category change):
dropped-linkers below the bar (`murdermemory`→`murder-memory`, `bloodclot`→
`blood-clot`, `nonresettable`→`non-resettable`) and edit-distance neighbours
of rare OOV forms (`qirl`→`girl`, `ninternal`→`internal`, `nexternal`→
`external`, `rilled`→`killed`). These are reported, not asserted: some are
genuine rare English (`rilled`, `suppurated`) and some are probably drift.

## 5. First attestation and counts

Every row carries `n`, `n_pages`, and `first_page` / `first_start` /
`first_uid` — the earliest occurrence by `(page, offset)`. Examples: `boy-roid`
first at p4:387, `gene=TV` at p4:1458, `ToKAGE` at p4:537, `exoskeleta` at
p4:1908, `feti` at p18:1684, `lt` at p5:1109, `666` at p56:959. Counts are
occurrence-level after markup removal, so `boy-roid` = 325 (case-folded; 315
lowercase + 10 `Boy-roid`) rather than the first-pass 332, which counted the
raw token stream including markup.

## 6. Numeric channel (handoff to #21)

18 types / 73 occurrences hold a digit or are pure digits: `2` (13), `8` (12),
`13` (10), `1` (9), `9` (7), `666` (6), `4-D` (3), `1mg`, `2002`, `17`, `22`,
`3-D`, `4`, `6`, `9-drug`, `1-84068-060-1` (ISBN), `metal-congenital++1`,
`replicant=1mg`. They are catalogued in `numeric_channel.tsv` and excluded
from the lexical neologism count.

## 7. Reconciliation with #17 / #18 / #6

| quantity | earlier | here | note |
|---|---|---|---|
| pages | 225 | 225 | same layout markdown |
| unit occurrences | 74,665 (#17/#18) | 74,665 | 1:1 uid join, byte-exact spans |
| markup residue | 229 strict (#17) / 231 (#18) | 347 | +116 opening tag names (`a`, `th`) |
| screened occurrences | 74,434 (#18) | 74,318 | #18 removed 231; #19 removes 347 |
| seam unit types | 1,293 (#17) | 1,292 | less `fluid=streaming` (markup) |
| open-juncture types | 2,047 (#18) | 2,046 | less `fluid=streaming` |
| OCR corrections | 15 (#6) | 15 | #6's `retroadam` is subsumed: the unit layer already writes `retro-ADAM` |

## 8. Limitations and negative results, stated plainly

- **Case is preserved, not merged.** 345 lower-case groups have more than one
  surface type (`boy-roid`/`Boy-roid`, `ADAM`/`adam`). `screening.tsv` keeps
  them apart (#22 owns case); `neologisms_lower.tsv` gives the folded view.
- **Zipf 1.5 is a fitted floor.** It removes junk wordfreq entries (`idhu`,
  Zipf 1.24) but also real rare English (`exoskeletal`, `morbus`, `telescoped`)
  and some romaji loanwords that wordfreq lists in `en` (`okama`, `nemuri`,
  `mienai`, `omanko`, `garakuta`). These sit in `rare_english` or, when #6's
  gazetteer names them, in the coinage class; the report does not pretend the
  boundary is clean.
- **`english_suffixation` is not novel compounding.** The 86 types like
  `defleshed`, `fecundates`, `respires` are ordinary English stem+suffix forms
  wordfreq lacks; they are kept in `compound` for completeness but flagged via
  `compound_kind`.
- **OCR detection is conservative and admits residual uncertainty.** The
  generic edit-distance rule is noisy (Siratori's lexis is genuinely distinct,
  as #6 found), so substitution cases are suspects, not errors. Roughly a
  dozen opaque one-off forms (`nunleashed`, `sartificial`, `ofproliferates`,
  `vertabre`, `laple`, `emodule`) are *probably* drift but stay
  `true_coinage` with an `ocr_suspect` flag.
- **No gazetteer covers everything.** 74 coinage types is a lower bound on
  genuine coinage: some `compound` rows embed an opaque part (141 types), and
  some `rare_english` rows may be coinages wordfreq happens to list.

## 9. Artifacts

| file | contents |
|---|---|
| `neologisms.tsv` | THE DELIVERABLE: 2,305 candidate rows, one per case-preserved unit type |
| `screening.tsv` | all 5,868 screened types with every flag (the denominator) |
| `neologisms_lower.tsv` | case-folded candidate aggregate (2,260 types) |
| `ocr_evidence.tsv` | 15 confirmed errors + 28 suspects with the open-form evidence |
| `numeric_channel.tsv` | 18 numeric types, handoff to #21 |
| `markup_residue.tsv` | 347 tag-residue occurrences with page context |
| `diagnostics.tsv` | every reconciliation count |
| `reconciliation.tsv` | this build vs #17 / #18 / #6 |
| `summary.json` | machine-readable headline numbers |

Column reference for `neologisms.tsv` / `screening.tsv`: `unit`, `lower`, `n`,
`n_pages`, `first_page`, `first_start`, `first_uid`, `cls`, `rule`, `channel`,
`n_morph`, `morph`, `head`, `closed_kind`, `compound_kind`, `zipf_en`,
`in_wordlist`, `en_status`, `parts`, `n_parts`, `n_english_parts`,
`n_oov_parts`, `parts_english`, `category`, `is_neologism`, `is_candidate`,
`proper_noun`, `cap_frac`, `gazetteer`, `numeric`, `ocr_rule`, `ocr_correct`,
`ocr_strength`, `ocr_suspect`, `ocr_suspect_distance`, `markup_base`,
`markup_extended`.

All `.tsv` are quote-safe (tabs/newlines escaped) and read with
`split("\t")`, never a csv reader.

---

*Generated by `work/neologism_detection.py` (schema `neologism-detection/1`).*
