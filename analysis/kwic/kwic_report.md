# KWIC concordancer (#26) — `concordance.tsv`

Source: #17 unit layer (`units.tsv`, 74,665 units) and token layer (`tokens.tsv`, 90,220 tokens); term list from #19/#22/#6/#21. Reproducible via `python work/kwic_concordancer.py`; regression suite `python work/test_kwic_concordancer.py`.

Schema `kwic/1`.

## 1. Headline result

**2,341 terms / 9,329 occurrences** printed in ±5-token context.  A term matches a unit **case-insensitively**; the occurrence is the unit (``uid``).  Context is the raw page slice over the ±5 tokens around the unit, located by a containment join on (page, start) — never by token index.

| term class | terms | occurrences |
|---|---:|---:|
| `compound` | 2156 | 6,480 |
| `true_coinage` | 51 | 338 |
| `case:label:acronym` | 27 | 245 |
| `proper_noun` | 18 | 287 |
| `ocr_error` | 15 | 127 |
| `numeric:digit_bearing` | 13 | 53 |
| `gazetteer:entity` | 12 | 1,129 |
| `case:scream:command` | 12 | 248 |
| `case:label:proper_name` | 10 | 168 |
| `case:label:invented_designator` | 9 | 25 |
| `numeric:spelled_number` | 6 | 191 |
| `case:label:single_letter` | 5 | 17 |
| `case:label:measure_prefix` | 4 | 5 |
| `numeric:paratext` | 2 | 3 |
| `numeric:markup_residue` | 1 | 13 |

## 2. Term list (entities + suspicious terms)

| source | what | types |
|---|---|---:|
| #19 | `proper_noun` (named coinages) | 23 |
| #19 | `compound` / `true_coinage` / `ocr_error` | 2,282 |
| #22 | `label` (proper_name / acronym / invented_designator / single_letter / measure_prefix) | 67 |
| #22 | `scream:command` (machine-command register) | 12 |
| #6 | gazetteer `entity` | 16 |
| #21 | numeric channel | 32 |

Excluded from #22, disclosed: `scream:emphasis` (29 upper-cased ordinary words — YES, NO, LOVE), `header` (31 title words — BLOOD, ELECTRIC, THIS) and `function` (I, A).  They are ordinary English, not entities or suspicious terms; #22 owns them.

## 3. Join: units, not words; uid, not token index

A word-frequency list cannot see a fused unit, so the concordancer joins the term list to the **unit** layer.  The match is case-insensitive on the unit surface (``ADAM`` / ``adam`` / ``Adam`` are one term).  Each occurrence carries its ``uid`` and joins #17's token stream by a containment test (token start ∈ [unit.start, unit.end)) on the same page.

> **Disclosed quirk:** the ``tokens.tsv`` ``unit`` column is a *per-page* unit index, not the book-global ``uid`` — #17 re-assigns uids to be book-global only *after* annotating tokens.  The concordancer therefore does not rely on that column; it locates a unit's tokens by (page, start) containment, which is exact and robust.

## 4. Per-term summaries (top 40 by occurrences)

| term | class | n | pages | first | embedded (part / token) |
|---|---|---:|---:|---|---:|
| `cadaver` | gazetteer:entity | 441 | 182 | p4:563 | 192 / 14 |
| `boy-roid` | compound | 325 | 139 | p4:387 | 0 / 13 |
| `soul/gram` | compound | 313 | 136 | p4:401 | 0 / 0 |
| `ADAM` | gazetteer:entity | 303 | 123 | p15:912 | 214 / 25 |
| `fecundates` | compound | 289 | 140 | p7:2086 | 4 / 4 |
| `cold-blooded` | compound | 192 | 118 | p6:299 | 0 / 2 |
| `zero` | numeric:spelled_number | 180 | 121 | p5:796 | 23 / 21 |
| `retro-ADAM` | gazetteer:entity | 166 | 43 | p87:1224 | 0 / 3 |
| `defleshed` | compound | 162 | 117 | p6:674 | 2 / 2 |
| `gene=TV` | compound | 144 | 99 | p4:1458 | 0 / 0 |
| `LOAD` | case:scream:command | 130 | 102 | p4:1477 | 18 / 15 |
| `body-OMOTYA` | compound | 128 | 94 | p5:2425 | 0 / 3 |
| `vital=serum` | compound | 128 | 80 | p15:693 | 0 / 0 |
| `ToKAGE` | proper_noun | 127 | 84 | p4:537 | 9 / 8 |
| `respires` | compound | 124 | 93 | p6:2373 | 0 / 0 |
| `suck=blood` | compound | 120 | 85 | p4:77 | 0 / 0 |
| `soul-machine` | compound | 105 | 54 | p42:16 | 0 / 6 |
| `machine=angel` | compound | 92 | 63 | p9:1133 | 0 / 0 |
| `cadaver-feti` | compound | 91 | 60 | p11:1209 | 0 / 4 |
| `feti` | true_coinage | 88 | 38 | p18:1684 | 115 / 9 |
| `fear=cell` | compound | 87 | 75 | p4:909 | 0 / 0 |
| `flesh-module` | compound | 85 | 48 | p31:2277 | 0 / 0 |
| `technocrisis` | compound | 84 | 63 | p18:589 | 2 / 2 |
| `DNA=channel` | compound | 75 | 55 | p6:1254 | 0 / 0 |
| `Placenta` | gazetteer:entity | 71 | 50 | p5:711 | 3 / 3 |
| `respired` | compound | 69 | 59 | p4:891 | 3 / 3 |
| `cadaver-mechanism` | compound | 68 | 45 | p16:1137 | 0 / 0 |
| `digital=vamp` | compound | 65 | 53 | p5:777 | 0 / 0 |
| `hydromachine` | compound | 64 | 48 | p4:237 | 7 / 7 |
| `android` | case:label:proper_name | 63 | 48 | p13:467 | 15 / 6 |
| `DNA` | case:label:acronym | 60 | 49 | p11:1839 | 79 / 78 |
| `hydromania` | proper_noun | 60 | 52 | p11:1119 | 3 / 3 |
| `lt` | ocr_error | 60 | 50 | p5:1109 | 1 / 1 |
| `desire-protocol` | compound | 59 | 40 | p29:1667 | 0 / 1 |
| `existence-code` | compound | 58 | 34 | p30:89 | 0 / 0 |
| `hydromaniac` | true_coinage | 57 | 47 | p28:931 | 0 / 0 |
| `oneiric` | true_coinage | 55 | 41 | p17:1394 | 1 / 1 |
| `acidHUMAN` | compound | 54 | 44 | p16:202 | 4 / 2 |
| `exoskeleta` | true_coinage | 54 | 48 | p4:1908 | 5 / 5 |
| `drug-embryo` | compound | 53 | 30 | p4:952 | 0 / 0 |

`embedded (part / token)` — two views of "the term inside a larger unit":

- **part** (`embedded_n`, via #17 ``part_index.tsv``): units where the term is a *word-part* but not the whole unit.  This includes hyphen join-parts (`retro-ADAM`) that are **not** separate WORD tokens.
- **token** (`embedded_tok`): WORD tokens matching the term that live inside a multi-token unit.  This reconciles with #22's all-caps *token* count: e.g. ``ADAM`` = 303 standalone + 25 token-embedded = 328 (#22).

Full table: `term_summaries.tsv`.

## 5. Sample concordances

### `ADAM` — 303 occurrence(s)

- p15 (uid 3823): …the bondage= mode of **ADAM** to the masses of flesh…
- p16 (uid 4031): …brain rapes the mutation-gear of **ADAM** //<<our living body…
- p16 (uid 4081): …hydromania//The brain of **ADAM** murdered the artificial sun:…
- p16 (uid 4179): …recorded to control external of **ADAM** internal organ consciousness of the…

### `ToKAGE` — 127 occurrence(s)

- p4 (uid 125): …maso=traffic::of **ToKAGE** that crowds in the Cadaver…
- p5 (uid 415): …//++the brain of **ToKAGE** is respired//Of quantum…
- p6 (uid 852): …self//The planet that **ToKAGE** polarized to the soft fear…
- p9 (uid 1740): …the cold-blooded disease body of **ToKAGE** that functions rhythm>>::…

### `boy-roid` — 325 occurrence(s)

- p4 (uid 106): …the porno nerve gas of **boy-roid** //The soul/gram…
- p5 (uid 556): …excoriate the brain strip of **boy-roid** noise//<<to the…
- p5 (uid 657): …of the reproduction quality of **boy-roid** that was jointed to the…
- p6 (uid 994): …The logic of SM of **boy-roid** to the cerebral cortex of…

### `gene=TV` — 144 occurrence(s)

- p4 (uid 273): …//The emotional machine that **gene=TV** distorted::LOAD//The…
- p5 (uid 433): …murder the virtual image of **gene=TV** ::in disguise to the…
- p5 (uid 546): …>::the holograms that **gene=TV** was isolated respire and excoriate…
- p5 (uid 666): …to the self ruin-device of **gene=TV** and resuscitate the soul/…

### `feti` — 88 occurrence(s)

- p18 (uid 4982): …of the city-mimic//:: **feti** //accelerates++contaminate++…
- p104 (uid 34942): …the insanity software::medium **feti** ::language of the chromosome…
- p112 (uid 37426): …>>cadaver circuit::of **feti** //the drug nature of…
- p122 (uid 40332): …//the body-OMOTYA of okama **feti** ::<<Direct volume::…

### `LOAD` — 130 occurrence(s)

- p4 (uid 275): …gene=TV distorted:: **LOAD** //The genome state/…
- p6 (uid 749): …of the brain universe// **LOAD** to the internal organ consciousness…
- p7 (uid 1260): …a brain-cell::>>:: **LOAD** //Defleshed machine::I…
- p9 (uid 1839): …off her heart::<< **LOAD** of the self=X…

### `666` — 6 occurrence(s)

- p56 (uid 17945): …Creature 13 and Super Cherry **666** >>that were resolved and…
- p60 (uid 19313): …ultrasonic orgasm of Super Cherry **666** / ####…
- p79 (uid 26218): …## <<SUPER CHERRY **666** >> Robo-Succubus of SADO —…
- p119 (uid 39368): …deathshead rising of a sodom **666** virus/…

## 6. Reconciliation

| quantity | here | earlier | note |
|---|---:|---:|---|
| pages | 223 | 225 (#17) | concordance spans |
| unit occurrences matched | 9,329 | — | exact case-insensitive unit match |
| terms with >=1 occurrence | 2,340 | — | of 2,341 terms |
| terms with 0 occurrences | 1 | — | always embedded in a larger unit, or absent |

## 7. Limitations, stated plainly

- **Case-insensitive matching merges surface variants.** ``ADAM`` (328) and ``adam`` are one term; the per-term summary lists every surface variant with its count.  #22 owns the case split.
- **Exact unit match only.** An occurrence of ``ADAM`` inside ``retro-ADAM`` is *not* an occurrence of the term ``ADAM`` — it is the term ``retro-ADAM``. The two ``embedded`` columns quantify this so the compound view is one join away: ``embedded_n`` counts word-parts (hyphen join-parts included), ``embedded_tok`` counts WORD tokens (reconciles with #22).
- **Numeric spelled words are a superset.** ``zero`` (180), ``one`` (5) etc. match every unit with that surface, a superset of #21's numeric-context count; the concordance shows all uses, the numeric ones included.
- **Context is the raw slice, whitespace-normalized.** Layout comments are stripped; original spacing (``gene=TV`` fused, ``machine = angel`` spaced) is preserved.  The key is the exact unit surface.

## 8. Artifacts

| file | contents |
|---|---|
| `concordance.tsv` | THE DELIVERABLE: one row per occurrence, ±5-token context |
| `term_summaries.tsv` | one row per term: counts, attestation, variants, embedded |
| `summary.json` | machine-readable headline numbers |
| `diagnostics.tsv` | per-source term tallies |
| `reconciliation.tsv` | this build vs #17/#19/#21/#22 |

All `.tsv` are quote-safe (tabs/newlines escaped) and read with `split("\t")`, never a csv reader.

---
*Generated by `work/kwic_concordancer.py` (schema `kwic/1`).*
