# Numeric and symbol channel catalogue (#21)

Source: `analysis/tokenizer/units.tsv` (#17 unit layer, 74,665 occurrences),
screened on `uid`, never words. Reproducible via
`python work/numeric_channel.py`; regression suite
`python work/test_numeric_channel.py` (32 tests).

Schema `numeric-channel/1`.

## 1. Headline result

**300 occurrences / 32 unit types** carry numeric or symbolic content —
0.40 % of the 74,665-unit layer. They split into five channels:

| channel | occ | types | what it is |
|---|---:|---:|---|
| `spelled_number` | 191 | 9 | `zero` (180), `eleven` (3), `one` (5), `two`, `four`, `Nine`, `Zero`, `ZERO` |
| `roman_marker` | 36 | 5 | `XX` (26), `XXX` (5), `XXXXXX` (1), `X` (2), `V` (2) |
| `digit_bearing` | 57 | 13 | `666`, `13`, `8`, `9`, `1`, `4-D`, `1mg`, `17`, `22`, `3-D`, `4`, `6`, `9-drug`, `metal-congenital++1`, `replicant=1mg` |
| `markup_residue` | 13 | 1 | `2` — `th colspan="2"` (p47 ×1, p209 ×12), HTML tag attribute |
| `paratext` | 3 | 2 | `1-84068-060-1` (ISBN), `2002` ×2 (copyright year) |

By kind: `null_point` 134 · `entity_id` 24 · `level` 48 · `count` 21 ·
`roman` 32 · `measurement_time` 10 · `measurement_dose` 4 · `dimension` 4 ·
`variable` 4 · `markup` 13 · `digit_word_fusion` 3 · `isbn` 1 · `year` 2.

The #19 handoff's 18 digit-bearing types / 73 occurrences are reproduced
**exactly** (reconciliation: `digit-bearing types 18 = 18, exact match`).

## 2. Method: screen units, not words

A word-frequency list cannot see a fused unit. The #19 handoff caught only
digit-bearing types; this issue widens the net to the forms a digit screen
cannot see — Roman-numeral markers (`XX` has no digits), spelled-out numbers
(`zero` has no digits), and the paratext/markup boundary. Every occurrence is
classified from its raw-text context (prefix/suffix around the unit span),
then aggregated to case-preserved types with first attestation.

## 3. The numbered-entity taxonomy (24 occurrences)

The book numbers its entities like a specimen list or bestiary:

| entity | numbers | evidence |
|---|---|---|
| `Creature` | 13 (×9), 9 (×3), 8 (×1) | p54:2154, p21:1891, p142:679 |
| `Super Cherry` | 666 (×4), 17, 9 | p56:959, p11:593, p33:2162 |
| `Necrodrome` | 13, 9 | p115:1188, p117:494 |
| `NDRO` | 4 | p183:98 |
| `eleven` / `Nine` | numbered entities | p13:735, p20:1362 |

`Creature 13` is the most frequent (×9, p54–p199). `Super Cherry 666` is the
book's signature entity — 666 is the Number of the Beast, and the book
deploys it as an apocalyptic techno-satanic marker: "Super Cherry 666",
"SUPER CHERRY 666 Robo-Succubus", "a sodom 666 virus", "Retro-sperm abortion
of 666". The numbers are arbitrary identifiers (like specimen numbers), not
counts — `Creature 13` is not the 13th creature, it is *named* 13.

## 4. The zero channel (206 occurrences)

`zero` is the book's null/origin/base-point marker — the single most frequent
numeric-channel item:

| form | count | example |
|---|---:|---|
| `zero` (standalone) | 180 | p5:796 "serum zero of the worldly desires" |
| `Level zero` | 48 (38+6+4 case) | p16:931 "Level zero angels evaporate" |
| `gravity zero` | 38 | p5:2011 "gravity zero zero=speed" |
| `zero gravity` | 15 | p15:708 "zero gravity to the" |
| `zero=` compounds | 13 | `zero=speed`, `zero=of=desire`, `zero=angel` |
| `zero/` compounds | 22 | `zero/of`, `zero/air`, `zero/nightmare` |
| zero-derived coinages | 5 | `zerotube`, `zeroised`, `zero-script`, `Zero-level`, `zero++her` |

Zero is used as (a) a physical null point ("gravity zero", "zero byte",
"zero growth", "zero seconds"), (b) a level/stage ("Level zero" ×48 — the
book's ground state), and (c) an entity name ("the animal zero", "dogs of
zero", "zero of the hydro-manias"). The zero-derived coinages show it is a
productive morpheme, not just a word.

## 5. Roman-numeral markers (36 occurrences)

`XX` ×26, `XXX` ×5, `XXXXXX` ×1, `X` ×2, `V` ×2 — a graded intensifier /
placeholder marker used as a modifier before nouns:

> "the **XX** brain universe" (p4:1926) · "the machinery torture **XX** genome"
> (p12:1583) · "the mutant **XX** sun" (p42:664) · "the **XX** milligram ice
> nebula" (p45:538) · "lupus **XX** angel mechanism" (p139:1375) · "the
> **XX** womb area machine" (p140:535) · "**XX** fractal world game" (p179:2)

`XX` reads as a placeholder for an unspeakable or classified name (Japanese
media convention: ダブルエックス), or an intensifier like "X-rated". The
grading X < XX < XXX < XXXXXX suggests scalable intensity. `X` ×2 is a
variable marker ("X of ToKAGE", "bunny-boy X"). `V` ×2 is ambiguous ("vision
V of an end" — possibly OCR of "vision 5", possibly a V marker).

## 6. Measurements and dimensions (18 occurrences)

- **Time**: "8 seconds" ×9, "8 second half-life" ×1 — pseudo-precise
  durations (p51–p206).
- **Dosage**: "1 milligram" ×2, "1mg" ×2 — drug dosages (p69, p113, p129, p161).
- **Dimension**: "4-D" ×3, "3-D" ×1 — spatial markers (p44, p168, p200, p203).
- **Counts**: "1 record", "1 am", "1 strike", "1 downloaded", "1 disappear",
  "1 digested" — first-person action counts (p21–p222).

## 7. Loop evidence: the numeric channel participates in the book's loop

The jacket copy calls the novel "a devastating loop of language". The numeric
channel participates directly — identical numeric contexts repeat across
distant pages:

| pages | repeated context |
|---|---|
| p4 = p220 | "the XX brain universe that is provoked and touched" |
| p5 = p221 | "the digital=vamp serum zero of the worldly desires machine" |
| p5 = p221 | "soul/gram of gravity zero zero=speed with the mass of flesh" |
| p6 = p222 | "gravity zero beasts machine nature=murder drone" |
| p6 = p222 | "the brain universe of XX erases the territory where degraded" |
| p7 = p223 | "we murder drone of gravity zero who caused crime wave" |
| p41 = p113 | "the region of gravity zero of acid-murder" |

The last ~20 pages (205–224) repeat the opening pages (1–20) with variations —
the numeric channel (zero, XX) is part of the loop's fingerprint.

## 8. Negative results, stated plainly

- **No version strings** (`v1.0`, `1.0`, `2.1`): 0 occurrences. The task
  description's example `v1.0` does not occur in this book.
- **No ordinals** (`1st`, `2nd`, `3rd`): 0 occurrences.
- **No dates** (`MM/DD/YYYY`): 0 occurrences.
- **No times** (`HH:MM`): 0 occurrences.
- **No pure-symbol units**: 0 — every unit contains at least one alphanumeric
  character. The "symbol channel" is entirely operator-bound (the operators
  `::`, `//`, `=`, `/` … are seams, not units); the numeric channel is the
  digit/number layer.
- **`second` ×190 is excluded** — almost all ordinary English ("the second
  of death"), not a numeric-channel item. `first` ×3 likewise.

## 9. Interpretation hypotheses (distinguished from measured structure)

**Measured**: the inventory above — 300 occurrences, 32 types, the entity
taxonomy, the zero channel, the roman markers, the loop repetitions.

**Hypothesis 1 — pseudo-code register.** The numbered entities, measurements,
and dimension markers mimic technical/scientific register (specimen numbers,
dosages, durations, spatial dimensions). This is consistent with #15's
finding that the book is "English punctuation roles encoded in C-family
symbols" — the numeric channel is the *data* layer of the pseudo-code, the
operators are the *syntax*.

**Hypothesis 2 — 666 as apocalyptic signature.** `Super Cherry 666` is the
book's signature entity; 666 (the Number of the Beast) is deployed as a
techno-satanic marker. This is interpretation, not measurement — but the
concentration of 666 in entity names (4 of 6 occurrences are "Super Cherry
666") is measured.

**Hypothesis 3 — zero as ground state.** `Level zero` ×48 and `gravity zero`
×38 suggest zero is the book's ground state / origin / null point — the
level from which everything departs and to which it returns. The loop
evidence (p4=p220, p5=p221, p6=p222, p7=p223) supports this: the book returns
to its zero point.

**Hypothesis 4 — XX as placeholder/intensifier.** The X-grading (X < XX < XXX
< XXXXXX) and the modifier position suggest a scalable placeholder for an
unspeakable name. This is interpretation; the distribution (XX dominant, 26
of 36) is measured.

## 10. Reconciliation with #17 / #19

| quantity | here | earlier | note |
|---|---|---|---|
| unit occurrences | 74,665 | 74,665 (#17) | same unit layer |
| digit-bearing types | 18 | 18 (#19) | exact match |
| digit-bearing occurrences | 73 | 73 (#19) | exact match |
| channel types (all) | 32 | — | widened net: roman + spelled + paratext + markup |

## 11. Limitations

- **Type-level classification is first-occurrence-wins.** A type like `9`
  (×7) is classified `entity_id` from its first occurrence ("Creature 9"),
  though later occurrences include "Level 9" and "9 interfaces". The
  sub-inventories (`levels.tsv`, `entity_numbers.tsv`) are occurrence-level
  and capture the distinction.
- **`V` ×2 is ambiguous** — possibly OCR of "5", possibly a V marker.
  Reported, not resolved.
- **`one` ×5 is ambiguous** — "one century", "One milligram", "one heat" are
  numeric uses, but `one` is also an ordinary English pronoun. The 5
  occurrences here are all numeric-determiner uses; the pronoun uses are
  excluded by the `SPELLED_NUMBERS` screen.
- **The zero channel's 206 count includes fused forms** (`zero=speed`,
  `zero/of`, `gravity=zero=system`). The standalone `zero` is 180; the 26
  fused forms are compound units where zero is a part.

## 12. Artifacts

| file | contents |
|---|---|
| `numeric_channel.tsv` | THE DELIVERABLE: 32 channel types with classification, first attestation, context |
| `entity_numbers.tsv` | 24 numbered-entity occurrences (Creature, Super Cherry, Necrodrome, NDRO) |
| `measurements.tsv` | 14 measurement occurrences (time, dosage) |
| `levels.tsv` | 48 level occurrences (Level zero/6/9) |
| `zero_channel.tsv` | 206 zero-bearing units with context |
| `roman_markers.tsv` | 36 roman-marker occurrences (XX, XXX, XXXXXX, X, V) |
| `loop_evidence.tsv` | 8 repeated numeric contexts (the book's loop) |
| `diagnostics.tsv` | every count |
| `reconciliation.tsv` | this build vs #17 / #19 |
| `summary.json` | machine-readable headline numbers |

All `.tsv` are quote-safe (tabs/newlines escaped) and read with
`split("\t")`, never a csv reader.

---

*Generated by `work/numeric_channel.py` (schema `numeric-channel/1`).*
