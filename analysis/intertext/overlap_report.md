# Intertextual n-gram overlap report (#57)

**Question.**  Do *Blood Electric*'s phrase formulas also occur in (a) another work by the same author and (b) the cited influences?  Shared formulas are candidates for genre; formulas shared only with the same author are candidates for idiolect.

## Method

* **Book side:** the #17 word-part layer, markup-screened exactly as #56 and split at page boundaries — 79,770 tokens / 3,683 types over 225 pages.
* **Corpus side:** the same #24 word regex, split at file boundaries; influence excerpts have their `#`/`--` attribution headers stripped.  Each reference-corpus file is checked against #56's manifest sha256 before use, and the author arm reads only the author's own interview turns (interviewer turns are excluded).
* **Overlap:** exact match of n-grams for n ∈ [1, 2, 3, 4, 5, 6]; a book n-gram counts as *formula* when it recurs ≥ 2 times.
* **Null:** each arm's tokens are permuted (5 shuffles, seed 57) and the overlap recomputed, so every observed count is read against chance.

## Which arms carry which n-grams

| arm | kind | tokens | types | shared n-gram types (n≥2) |
|---|---|---:|---:|---:|
| siratori_other | same_author | 330 | 125 | 205 |
| author_paratext | author_meta | 167 | 95 | 75 |
| influence_cyberpunk | influence | 105 | 81 | 13 |
| influence_cutup_surrealist | influence | 56 | 46 | 7 |
| influence_finnegans_wake | influence | 49 | 42 | 2 |
| register_general_english | register | 1,609,978 | 46,678 | 5,617 |
| register_medical | register | 380,072 | 25,536 | 1,485 |
| register_legal | register | 350,406 | 13,378 | 2,413 |
| register_cs | register | 103,048 | 8,996 | 1,049 |

The influences are partial by construction: the largest excerpt arm is a few hundred tokens against the book's tens of thousands, so a *low* influence overlap bounds the genre claim from below and cannot refute it.  The register arms are full-size controls.

## Observed vs chance (shared instances)

| arm | n | observed | expected (shuffle) | observed/expected |
|---|---:|---:|---:|---:|
| siratori_other | 2 | 180 | 87.6 | 2.0548 |
| siratori_other | 3 | 97 | 7.2 | 13.4722 |
| siratori_other | 4 | 36 | 0.4 | 90.0 |
| siratori_other | 5 | 8 | 0.0 | inf |
| siratori_other | 6 | 3 | 0.0 | inf |
| author_paratext | 2 | 68 | 30.4 | 2.2368 |
| author_paratext | 3 | 13 | 2.6 | 5.0 |
| author_paratext | 4 | 2 | 0.0 | inf |
| author_paratext | 5 | 0 | 0.0 | 0.0 |
| author_paratext | 6 | 0 | 0.0 | 0.0 |
| influence_cyberpunk | 2 | 13 | 8.0 | 1.625 |
| influence_cyberpunk | 3 | 0 | 0.0 | 0.0 |
| influence_cyberpunk | 4 | 0 | 0.0 | 0.0 |
| influence_cyberpunk | 5 | 0 | 0.0 | 0.0 |
| influence_cyberpunk | 6 | 0 | 0.0 | 0.0 |
| influence_cutup_surrealist | 2 | 7 | 1.6 | 4.375 |
| influence_cutup_surrealist | 3 | 0 | 0.0 | 0.0 |
| influence_cutup_surrealist | 4 | 0 | 0.0 | 0.0 |
| influence_cutup_surrealist | 5 | 0 | 0.0 | 0.0 |
| influence_cutup_surrealist | 6 | 0 | 0.0 | 0.0 |
| influence_finnegans_wake | 2 | 2 | 2.2 | 0.9091 |
| influence_finnegans_wake | 3 | 0 | 0.0 | 0.0 |
| influence_finnegans_wake | 4 | 0 | 0.0 | 0.0 |
| influence_finnegans_wake | 5 | 0 | 0.0 | 0.0 |
| influence_finnegans_wake | 6 | 0 | 0.0 | 0.0 |
| register_general_english | 2 | 171,624 | 80,219.6 | 2.1394 |
| register_general_english | 3 | 10,400 | 1,147.6 | 9.0624 |
| register_general_english | 4 | 941 | 7.8 | 120.641 |
| register_general_english | 5 | 84 | 0.0 | inf |
| register_general_english | 6 | 0 | 0.0 | 0.0 |
| register_medical | 2 | 13,338 | 7,676.6 | 1.7375 |
| register_medical | 3 | 642 | 96.2 | 6.6736 |
| register_medical | 4 | 37 | 1.2 | 30.8333 |
| register_medical | 5 | 2 | 0.0 | inf |
| register_medical | 6 | 0 | 0.0 | 0.0 |
| register_legal | 2 | 46,884 | 24,997.2 | 1.8756 |
| register_legal | 3 | 3,295 | 579.0 | 5.6908 |
| register_legal | 4 | 319 | 6.8 | 46.9118 |
| register_legal | 5 | 23 | 0.4 | 57.5 |
| register_legal | 6 | 0 | 0.0 | 0.0 |
| register_cs | 2 | 5,480 | 3,109.0 | 1.7626 |
| register_cs | 3 | 224 | 29.2 | 7.6712 |
| register_cs | 4 | 3 | 0.0 | inf |
| register_cs | 5 | 0 | 0.0 | 0.0 |
| register_cs | 6 | 0 | 0.0 | 0.0 |

## Formula classes

Every book n-gram with book frequency ≥ 2 and n ≥ 2, classified by the strongest external source that contains it:

```
author_idiolect             116  
author_common                98  
genre_influence               1  
influence_common             14  
register_common            2902  ####
book_unique               22838  ###################################
```

* `author_idiolect` — occurs in the author's own other text (the *different* Siratori work and/or his interview answers) and in **no** general/technical register: a distinctive formula that travels across the author's works.
* `author_common` — occurs in the author's own text but also in ordinary/technical English; not distinctive.
* `genre_influence` — occurs in a cited-influence excerpt and in no register: a genre candidate.
* `influence_common` — occurs in an influence excerpt but also in a register; not distinctive.
* `register_common` — occurs only in a general/technical corpus; ordinary English or technical vocabulary, not genre.
* `book_unique` — occurs in no external source.

### Top idiolect formulas (author-only, not ordinary English)

| n | phrase | book freq | also in |
|---:|---|---:|---|
| 2 | `soul gram` | 326 | siratori_other |
| 2 | `blooded disease` | 188 | siratori_other |
| 3 | `cold blooded disease` | 188 | siratori_other |
| 3 | `mass of flesh` | 185 | siratori_other |
| 2 | `retro adam` | 173 | siratori_other |
| 2 | `of retro` | 147 | siratori_other |
| 3 | `of retro adam` | 144 | siratori_other |
| 3 | `the soul gram` | 134 | siratori_other |
| 2 | `dna channel` | 110 | siratori_other |
| 2 | `brain universe` | 106 | siratori_other |
| 2 | `disease animals` | 101 | siratori_other |
| 3 | `blooded disease animals` | 98 | siratori_other |
| 4 | `cold blooded disease animals` | 98 | siratori_other |
| 2 | `digital vamp` | 87 | siratori_other |
| 2 | `script of` | 82 | siratori_other |
| 3 | `the brain universe` | 73 | siratori_other |
| 3 | `for the grotesque` | 72 | siratori_other |
| 3 | `hunting for the` | 70 | siratori_other |
| 4 | `hunting for the grotesque` | 70 | siratori_other |
| 2 | `the hyperreal` | 56 | siratori_other |

### Top genre-influence formulas

| n | phrase | book freq | influences |
|---:|---|---:|---|
| 2 | `a virus` | 8 | influence_cutup_surrealist |

## Interpretation

1. **Idiolect is measurable and measured.**  116 book formulas (n≥2, 69 of them length ≥3) occur in the author's own other text and in no general/technical register: 104 of them in the *different* Siratori work and 12 in the interview paratext alone.  The signal is far above chance: for the same-author probe, n-gram instances hit 97 vs 7.2 expected under shuffling at n=3 (13.4722×), and 180 vs 87.6 at n=2 (2.0548×).  The stable, distinctive formula set (`soul gram`, `cold blooded disease`, `retro adam`, `brain universe`, `dna channel`, `data mutant` …) is the author's idiolect, as far as this corpus can demonstrate it.
2. **Genre is underpowered here, and that is reported as a bound.**  The cited influences are fair-use excerpts of a few dozen to a few hundred tokens; the only shared n-gram not also ordinary English is `a virus`, and no influence arm shares any trigram with the book.  `a virus` is the Burroughs line *"Language is a virus from outer space"* (*The Ticket That Exploded*) — one declared-influence phrase, quoted in the #56 excerpt, not evidence of broad genre transfer.  At this size genre cannot be separated from generic English; the `influence_common` and `register_common` classes show how much of the book's formula stock is simply ordinary English.
3. **The named-work comparison is not testable from lawful sources.**  None of *Headcode*, *Human_Worms*, *Smart-d*, *GENEDUB* or *Nonexistence* is servable; the ledger records each refusal/absence with its evidence.  No other text stands in for them.
4. **Provenance is checked, not asserted.**  The embedded Great Works excerpt is pinned by token count and sha256 (re-verified against the live page by `--probe`), every reference-corpus file is verified against #56's manifest sha256, and the one interviewer turn in #58's Q/A table is excluded from the author arm.  No comparison text is invented or silently substituted for another.

## Limitations

* The same-author probe is a *different* work (Great Works), not any of the five named publications; it can support an idiolect claim but cannot speak to those specific books.
* The influence arms are excerpt-length; their overlap is a lower bound on the true genre overlap.
* Matching is lexical (operators stripped as the word regex requires); it does not test operator-grammar transfer, which #12/#15 already treat as a separate (and non-genre) channel.
* Word-part tokenisation means fused compounds contribute their parts; a formula like `living body` is a part-level match, not a unit-level one.

## Reproduce

```bash
python work/intertext.py            # writes analysis/intertext/
python work/intertext.py --probe    # re-run the live source probes
python work/test_intertext.py
```

