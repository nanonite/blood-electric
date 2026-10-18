# Case and emphasis inventory (#22)

Source: `analysis/tokenizer/tokens.tsv` (#17 token layer, 77,485 WORD tokens) and
`analysis/tokenizer/units.tsv` (the uid join). Section headers from
`analysis/layout_recovery/sections.tsv` (#7). Reproducible via
`python work/case_emphasis.py`; regression suite `python work/test_case_emphasis.py`.

Schema `case-emphasis/1`.

## 1. Headline result

**1,890 ALL-CAPS tokens (141 types) — 2.44 % of the 77,485 word tokens.** They
are not a scream channel: they are a **designator channel**. Four classes, by
occurrence:

| class | occurrences | share of caps | types | what it is |
|---|---:|---:|---:|---|
| `function` | 453 | 24.0 % | 2 | the single letters `I` (407) and `A` (46) |
| `label` | 1,138 | 60.2 % | 67 | names, acronyms, invented designators |
| `scream` | 263 | 13.9 % | 41 | upper-cased ordinary words (command/emphasis) |
| `header` | 36 | 1.9 % | 34 | words inside a layout heading (#7) |

`I`/`A` are upper-case only because they are one character — they are the
pronoun and the article, not emphasis — so the meaningful inventory is the
remaining **1,437 occurrences / 139 types**: **79.2 % labels, 18.3 % screams,
2.5 % headers.**

The label channel is dominated by a small core: the top 16 label types carry
**1,001 of 1,138 label occurrences (88 %)** — `ADAM` (328), `TV` (173),
`DNA` (138), `BABEL` (75), `HDD` (39), `VTR` (37), `SADO` (31), `XX` (27),
`SM` (24), `NDRO` (23), `SODO` (23), `ANDROID` (21), `NDROID` (21),
`EVOL` (15), `HIV` (13), `NIHIL` (13). These recur on 100+ pages each
(`ADAM` on 128 pages, `TV` 114, `DNA` 104): the all-caps form is how the book
marks an entity, not how it shouts.

**Verdict:** headers are a structural subset (36 occurrences, all in section
titles); **labels dominate**; a distinct **machine-command register** exists
(`LOAD` 140, `PLAY` 38, `REC` 7, `OUTPUT` 5, `DOWNLOAD` 4, `UPLOAD` 4,
`SAVE` 3, `RELOAD` 3, `DELETE`/`PUSH`/`PUT`/`STOP` 1 each = 208 occurrences);
genuine affective screams (`YES`, `NO`, `LOVE`, `HEAVEN`, `ANGEL`, `MOTHER`,
`REAL`, `RAPTURE`, `ORANGE`) are **hapax**. Of the 41 scream types, 26 are
hapax; of the 67 label types, 25 are hapax, but the recurring ones carry the
mass.

## 2. Method: screen tokens, join units

Case is a property of the **token**, not of the maximal unit: `retro-ADAM` is
one mixed-case unit whose `ADAM` is *not* an all-caps token, while a standalone
`ADAM` is. So the screen runs on `tokens.tsv`, and every occurrence carries its
`unit` (uid) so it joins the #17 unit layer (`units.tsv`) and, through it,
#18/#19/#21. A token is all-caps when it is a WORD, has a letter, and
`text == text.upper()`.

Decision rule (each signal is measured; the label override is curated and
auditable in `LABEL_REASON` in the script):

1. **`header`** — the token's layout region is a heading (#7). A title is
   upper-cased *because it is a title*, so even its function words (`OF`,
   `THIS`) are structural, not emphasis.
2. **`function`** — `I` and `A`: one-character function words, excluded.
3. **`label`** — a designator: a single-letter variable (`G`, `K`, `X`, `V`,
   `H`), a curated acronym, or a curated proper name / invented designator. A
   curated type stays a label even when an English lowercase form exists
   (`ANDROID` 47 lowercase, `OKAMA` 16), because its all-caps use is
   referential.
4. **`scream`** — an ordinary English word upper-cased for emphasis or as a
   console command. Test: the exact lowercase form occurs in the book, **or**
   `wordfreq` Zipf(en) ≥ 2.5. Subtype `command` (imperative/console verb) vs
   `emphasis`.
5. otherwise **`label`** — an OOV all-caps form with no English word behind it.

Two further signals are recorded per type: whether the occurrence sits inside a
seam-bearing naming unit (`gene=TV`, `SADO=machine`, `HYPE=ToKAGE`) and whether
it is part of a hyphen entity chain. **91 caps occurrences (4.8 %) are inside a
naming/fused unit** — independent evidence that these are names.

## 3. Report against section headers

`sections.tsv` has **56 headers**; **24 are fully upper-case titles**, and the
36 header caps occurrences land on **24 pages**. The other 32 headers are
lower-case or title-case (`abolition`, `Bias`, `Dead Bargain`, `Node`,
`<<AcidHUMAN>>`, `<< Process Of Evolution>>` …), i.e. the book uses all-caps
*titles* only sometimes.

| page | title | caps tokens |
|---:|---|---|
| 1 | `BLOOD ELECTRIC` | BLOOD, ELECTRIC |
| 3 | `WARNING THIS MACHINE KILLS` | WARNING, THIS, MACHINE, KILLS |
| 9 | `A-Z` | A-Z |
| 49 | `CODA` | CODA |
| 52 | `<<VIRUS OF SPECTRE>>` | VIRUS, OF, SPECTRE |
| 53 | `<<SPACE GEISHA>>` | SPACE, GEISHA |
| 54 | `<<GRAVITY-FREE>>` | GRAVITY-FREE |
| 58 | `<<PORNO>>` | PORNO |
| 60 | `<<SWASTIKA GASH>>` | SWASTIKA, GASH |
| 61 | `<<ALIEN-COM>>`, `<<CODA-CODA>>` | ALIEN-COM, CODA-CODA |
| 68 | `<<DUSTNIRVERNA>>` | DUSTNIRVERNA |
| 69 | `<<XTERMINATOR>>` | XTERMINATOR |
| 76 | `<<ZEN>>` | ZEN |
| 79 | `<<SUPER CHERRY 666>>` | SUPER, CHERRY |
| 84 | `COM` | COM |
| 88 | `<<ADAM>>` | ADAM |
| 107 | `<<GEISHA GASH>>` | GEISHA, GASH |
| 120 | `HYDROMANIA` | HYDROMANIA |
| 144 | `MØNKEY/gene-dub` | MØNKEY |
| 180 | `NDRO` | NDRO |
| 195 | `<<FATAL CODE>>` | FATAL, CODE |
| 203 | `OBJECT` | OBJECT |
| 215 | `<<SLASH-RAPE>>` | SLASH-RAPE |
| 217 | `<<GUT-SUCK>>` | GUT-SUCK |

The full per-header table is `section_headers.tsv`. Because a title word is a
structural signal, it is classified `header` and kept out of the label/scream
counts; a handful of types appear both in a title and in the body (`ADAM`,
`NDRO`, `CODA`, `BLOOD`, `ELECTRIC`, `VIRUS`, `SWASTIKA`) and carry both
classes in `caps_types.tsv` (`classes` column).

## 4. Labels

| subtype | occurrences | examples (count, first attestation) |
|---|---:|---|
| `proper_name` | 581 | `ADAM` 328 (p15), `BABEL` 75 (p9), `SADO` 31 (p12), `NDRO` 23 (p4), `SODO` 23 (p40), `ANDROID` 21 (p13), `NDROID` 21 (p110), `EVOL` 15 (p50), `NIHIL` 13 (p20) |
| `acronym` | 500 | `TV` 173 (p4), `DNA` 138 (p6), `HDD` 39 (p17), `VTR` 37 (p7), `XX` 27 (p4), `SM` 24 (p6), `HIV` 13 (p167), `CD-ROM` 5 (p29) |
| `invented_designator` | 33 | `SCANIMAL` 6 (p22), `HYPE` 10 (p126), `OMOTYA` 4 (p177), `GODNAM` 3 (p52), `LEVELDOWN` 3 (p30), `DUSTNIRVERNA` (p67), `XTERMINATOR` (p68) |
| `single_letter` | 18 | `K` 8 (p50), `G` 4 (p48), `X` 3 (p9), `V` 2 (p76), `H` 1 (p11) — variables in `channel [G]`, `nutrient [K]`, `self=[X]` |
| `measure_prefix` | 6 | `TERA` 3 (p152), `MAX` 1 (p167), `GIGA` 1 (p139), `REV` 1 (p90) |

The acronym/proper-name reading is checked against the corpus: `TV` occurs 14
times inside a naming unit (`gene=TV`), `DNA` 5 (`DNA=channel`), `BABEL` 4
(`BABEL-TV`, `BABEL=TV`), `HYPE` inside `HYPE=ToKAGE`, `SADO` inside
`SADO=machine`. `NDRO`, `NDROID`, `EVOL`, `NIHIL` are OOV and recur as bare
noun labels (`the quantum tragedy of NDRO`, p4).

## 5. Screams: command register vs emphasis

| subtype | occurrences | types | examples |
|---|---:|---:|---|
| `command` | 208 | 12 | `LOAD` 140, `PLAY` 38, `REC` 7, `OUTPUT` 5, `DOWNLOAD` 4, `UPLOAD` 4, `SAVE` 3, `RELOAD` 3, `DELETE`/`PUSH`/`PUT`/`STOP` 1 |
| `emphasis` | 55 | 29 | `LEVEL` 9, `DIGITAL` 6, `SEX` 6, `HEAD` 2, `HYPER` 2, `RAPTURE` 2, `REVERB` 2, `YES`/`NO`/`LOVE`/`HEAVEN`/`ANGEL`/`MOTHER`/`REAL`/`ORANGE` 1 |

`LOAD` (140 occurrences, 106 pages, first p4) is the single most frequent
non-label all-caps token; the console verbs cluster on the `//` utterance
breaks (`… distorted::LOAD//The genome state/…`, p4). The emphasis set is
mostly hapax and often the book's own affect vocabulary — `FEAR` is not
upper-cased, but `LOVE` (p52), `YES` (p60), `ANGEL` (p190) are one-off
uppercasings of ordinary words.

## 6. Edge cases, negative results, honesty

- **HTML-tag residue: 0 all-caps overlap.** #17 flagged 229 units born inside
  surviving source markup; every tag name is lower-case (`th`, `p`, `pre`,
  `a`), so none enters this inventory. The `A` (46) tokens are the article, not
  the `<a>` tag. `markup_residue_caps_occurrences = 0` (reconciliation row).
- **`I` / `A` are function words.** 453 of the 1,890 all-caps tokens are these
  two single letters. They are reported (`class=function`) but excluded from the
  label/scream decision.
- **Single-letter designators** (`K`, `G`, `X`, `V`, `H`, 18 occurrences) are
  labels, not screams: contexts show variables (`channel [G]`, p48/p52;
  `nutrient [K]`, p66/p75; `self=[X]`, p9).
- **Ambiguous label/scream boundary, disclosed not hidden.** `ANDROID`
  (21 caps / 47 lowercase) and `OKAMA` (4 / 16) have both forms; the all-caps
  form is kept as a label because it appears in entity chains
  (`ANDROID-beat-ANDROID-…`, `body/OKAMA`). `CODA` (5) is a section title and a
  recurring marker; classified label. `HYPE` (10, lower=0, Zipf 4.03) is put in
  `invented_designator` on the strength of `HYPE=ToKAGE`. These are decisions,
  and every signal (`lower_exact_n`, `zipf_en`, `naming_unit_n`,
  `label_reason`) is in `caps_types.tsv` so a reviewer can reverse any of them.
- **All-caps *units* (the #17 join view).** 1,512 occurrences / 134 types of
  single-word units are entirely upper-case (`ADAM` 303, `LOAD` 126, `DNA` 60 …);
  listed in `all_caps_units.tsv`. The 378-token gap to the 1,890 caps tokens is
  caps words that are parts of mixed-case units (`gene=TV`, `retro-ADAM`).

## 7. Reconciliation

| quantity | here | earlier | note |
|---|---:|---|---|
| word tokens | 77,485 | 77,485 (#17) | same token layer |
| all-caps tokens | 1,890 | — | not previously inventoried |
| markup-residue all-caps | 0 | 229 tag units (#17) | tag names are lower-case |
| section headers | 56 | 56 (#7) | same `sections.tsv` |

## 8. Files

| file | contents |
|---|---|
| `caps_occurrences.tsv` | one row per all-caps token: class, subtype, reason, unit uid, naming/fused flags, context |
| `caps_types.tsv` | one row per type: counts, class/subtype, header/para split, `lower_exact_n`, `zipf_en`, naming count, first attestation, context |
| `section_headers.tsv` | the 56 #7 headers with their caps tokens and `all_caps_title` flag |
| `per_page.tsv` | caps tokens / word tokens / rate per page |
| `all_caps_units.tsv` | all-caps single-word units (the #17 join view) |
| `diagnostics.tsv` | class/subtype tallies and screen totals |
| `reconciliation.tsv` | totals against #17/#7 |
| `summary.json` | machine-readable headline numbers |
