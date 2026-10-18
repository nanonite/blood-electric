# Rosetta Mapping Against Real Notations (#15)

**Question.** Place the book's operator set beside C/C++, Perl, Lisp/S-expressions, HTML/XML, Markdown, regular expressions, molecular/genetic notation and ordinary English punctuation, and score which real syntax the book's code most resembles.

**Verdict (testable).** Of eight real notations, the book's operator *alphabet* scores highest against **C / C++** (total 0.75, role-symbol coverage 82.4%, exact symbol+role matches covering 64.9% of operator occurrences), then **Perl** (0.72) and **HTML / XML** (0.58); it scores lowest against **Regular expressions** (0.35). The resemblance is **symbolic, not functional**: the book reuses the C-family punctuation pool (`::` `//` `=` `/` `<<` `>>` `++` `|`) but assigns it record/naming/utterance semantics rather than arithmetic/comment semantics, so #12's measured roles *contradict* the notation's own meaning at exactly the points where the surface looks most code-like. This is falsifiable: re-measure an operator role (#12) or correct a notation signature and the ranking moves.

**The two halves disagree, and that is the finding.** On the *symbol* axis the book is code-like: C / C++ contains 100.0% of operator occurrences in its punctuation set, and exact symbol+role matches cover most of the mass. On the *role* axis it is natural-language-like: ordinary English punctuation — which shares **no** operator symbol with the book — still expresses the role of 79.3% of operator occurrences (sentence boundary, hyphen compound, quoting). The book is an English sentence machine wearing a C-family alphabet.

## Method

The book side is **measured**; the notation side is a **spec-based signature**. Each book operator carries the role #12 measured for it (`analysis/operator_functions/classification_spec.md`). Each reference notation is described by `(symbol, role, position)` entries taken from its specification (citations in `notation_signatures.tsv`). Every book operator occurrence is matched in four tiers — exact symbol+role (1.00), same-family role (0.85), role present with a different symbol (0.60), symbol collision with an unrelated role (0.40) — and the frequency-weighted sum is the notation's *role-symbol coverage*. Three further dimensions are combined: alphabet Jaccard (0.15), capability-feature agreement (0.15) and positional fidelity (0.15). All dimensions are reported separately in `scores.tsv` so the weighting can be changed.

The 10,949 #10 operator tokens are used for the headline score. The word-internal hyphen (2,834) and free-standing dash (177) that #12 measures separately are a sensitivity add-on, not part of the headline.

## Scorecard

| rank | notation | role-symbol coverage | exact (symbol+role) | alphabet contains | alphabet Jaccard | features | position | **total** |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | C / C++ | 82.4% | 64.9% | 100.0% | 0.42 | 75.0% | 83.4% | **0.75** |
| 2 | Perl | 81.7% | 61.3% | 96.5% | 0.37 | 60.0% | 82.8% | **0.72** |
| 3 | HTML / XML | 54.0% | 29.3% | 42.8% | 0.21 | 75.0% | 94.8% | **0.58** |
| 4 | English punctuation | 47.6% | 0.0% | 0.0% | 0.00 | 65.0% | 93.4% | **0.50** |
| 5 | Markdown | 49.0% | 0.2% | 7.3% | 0.09 | 45.0% | 79.0% | **0.47** |
| 6 | Molecular / genetic | 47.8% | 19.8% | 45.6% | 0.21 | 35.0% | 58.7% | **0.44** |
| 7 | Lisp / S-expressions | 22.6% | 0.0% | 43.0% | 0.21 | 50.0% | 90.1% | **0.37** |
| 8 | Regular expressions | 20.2% | 0.2% | 0.3% | 0.08 | 55.0% | 94.4% | **0.35** |

Weights: role 0.55, Jaccard 0.15, features 0.15, position 0.15. `alphabet contains` = share of operator occurrences whose symbol is somewhere in the notation's punctuation set; `symbol_recog` (the best explanation actually used that symbol) and `role_expressible` are in `scores.tsv`.

## The book side (measured, from #12)

| operator | n | measured role(s) | position | both-glued | interior | upper-next |
|---|---:|---|---|---:|---:|---:|
| `::` | 3,083 | field_sep (100%) | inline | 0.77 | 0.78 | 13.0% |
| `//` | 2,149 | clause_boundary (100%) | inline | 0.73 | 0.73 | 52.7% |
| `=` | 2,047 | naming (100%) | inline | 0.94 | 0.92 | 9.4% |
| `/` | 1,484 | compound_link (71%) + clause_boundary (29%) | any | 0.52 | 0.65 | 22.4% |
| `>` | 777 | inline_delim (70%) + clause_boundary (30%) | inline | 0.49 | 0.53 | 29.9% |
| `<<` | 412 | block_open (100%) | any | 0.18 | 0.23 | 18.2% |
| `>>` | 386 | block_close (70%) + clause_boundary (30%) | inline | 0.53 | 0.61 | 23.3% |
| `<` | 381 | inline_delim (100%) | any | 0.11 | 0.12 | 5.2% |
| `++` | 198 | coupling (100%) | inline | 0.67 | 0.73 | 5.6% |
| `\|` | 17 | pipe (100%) | any | 0.00 | 0.06 | 0.0% |
| `+` | 15 | compound_link (100%) | inline | 0.87 | 0.87 | 6.7% |

The discriminating facts: `::` is a *field/phrase* separator (13% sentence-initial), `//` is an *utterance boundary* (53% sentence-initial) and **not** a line comment, `=` is a tight naming linker (94% glued, 92% interior), `<<`/`>>` are a real paired block (230 matched spans, max depth 30, median 5 words) but only 56% balanced. These are the measured properties the notations are scored against.

## Per-operator Rosetta table

For each book operator, the notation whose signature best explains it (highest tier × frequency). This is the core mapping.

| operator | n | role | best match | tier | score | that notation's symbol |
|---|---:|---|---|---|---:|---|
| `::` | 3,083 | field_sep | C / C++ | exact | 1.00 | `::` (field_sep) |
| `//` | 2,149 | clause_boundary | Molecular / genetic | exact | 1.00 | `//` (clause_boundary) |
| `=` | 2,047 | naming | C / C++ | exact | 1.00 | `=` (naming) |
| `/` | 1,484 | compound_link + clause_boundary | English punctuation | role_only | 0.60 | `-` (compound_link) |
| `>` | 777 | inline_delim + clause_boundary | C / C++ | exact | 0.88 | `>` (inline_delim) |
| `<<` | 412 | block_open | C / C++ | exact | 1.00 | `<<` (block_open) |
| `>>` | 386 | block_close + clause_boundary | C / C++ | exact | 0.88 | `>>` (block_close) |
| `<` | 381 | inline_delim | C / C++ | exact | 1.00 | `<` (inline_delim) |
| `++` | 198 | coupling | Perl | role_only | 0.60 | `.` (coupling) |
| `\|` | 17 | pipe | C / C++ | exact | 1.00 | `\|` (pipe) |
| `+` | 15 | compound_link | English punctuation | role_only | 0.60 | `-` (compound_link) |

Reading the table: `::` is an exact C++/Perl scope/package separator; `=` an exact assignment/attribute linker; `<<`/`>>` exact C++ iostream stream operators; `|` an exact pipe/alternation in five notations. But `//` — the single most code-looking symbol — is *not* a comment anywhere in the book (#12); its closest real analogue is the GenBank/EMBL `//` record terminator (role-compatible, but line-anchored there, inline here). `++` is a symbol collision with C's increment (the book uses it to couple words, never numerals), and `<`/`>` collide with HTML tags and C++ templates. The mapping is therefore a **false-friend map** as much as a resemblance map.

## Positional diagnostics

Where the notation places a matched symbol (inline vs line-anchored) vs where the book places it. `//` is the clearest mismatch.

| operator | book | best notation | notation convention | fidelity |
|---|---|---|---|---:|
| `::` | inline | C / C++ | inline | 1.00 |
| `//` | inline | Molecular / genetic | line | 0.25 |
| `=` | inline | C / C++ | inline | 1.00 |
| `/` | any | English punctuation | inline | 0.75 |
| `>` | inline | C / C++ | inline | 0.85 |
| `<<` | any | C / C++ | inline | 0.75 |
| `>>` | inline | C / C++ | inline | 0.85 |
| `<` | any | C / C++ | inline | 0.75 |
| `++` | inline | Perl | inline | 1.00 |
| `\|` | any | C / C++ | inline | 0.75 |
| `+` | inline | English punctuation | inline | 1.00 |

## Structural profile (measured)

| pair | variant | matched | unmatched open | unmatched close | balance | median span (words) | max depth |
|---|---|---:|---:|---:|---:|---:|---:|
| `<<`…`>>` | block | 230 | 182 | 156 | 55.8% | 5 | 30 |
| `<<`…`>>` | block_clean | 229 | 175 | 155 | 56.7% | 5 | 30 |
| `<`…`>` | angle | 245 | 136 | 532 | 64.3% | 1 | 98 |
| `<`…`>` | angle_clean | 145 | 38 | 521 | 79.2% | 2 | 5 |

The book *does* build nested paired blocks (`<<`…`>>`, max depth 30, median 5 words), which rules out a purely linear notation (English punctuation, molecular FASTA) — but it closes only 230 of 412 `<<` opens (56%), far from the near-perfect balance of C braces, Lisp parens or HTML tags. That partial balance is itself the finding: a designed delimiter system with heavy residue (unclosed spans, OCR artifacts).

## Null model: is the winner better than chance?

Shuffling which book symbol carries which role (keeping the role multiset, 2000 permutations) gives a null maximum role-symbol coverage of 65.9% (sd 0.053, range 55.1%–80.4%). The observed winner scores 82.4% (z=3.1, p=0.0004998). The symbol→role pairing is therefore privileged, not an artifact of counting many symbols.

## Sensitivity

| variant | winning notation |
|---|---|
| headline (10,949 #10 operators) | C / C++ |
| + word-internal & free-dash hyphens | C / C++ |
| excluding raw-HTML OCR pages 92/158/219 | C / C++ |
| dropping `|` and `+` (n<20) | C / C++ |

Full variant rankings are in `summary.json` (`sensitivity.variant_rank`). The winner is stable across all four variants, so the result is not driven by the rare operators or by the OCR artifact pages.

## What this means (and what it does not)

- **Where the book is coming from.** Its punctuation alphabet is the C / C++ family's: scope/namespace `::`, assignment `=`, stream `<<`/`>>`, pipe `|`, increment-shaped `++`. That is a concrete, checkable claim about the book's typographic lineage.

- **Where it is not.** The measured *functions* diverge. `//` is not a comment; `++` is not arithmetic; `<<`/`>>` are not I/O but block quotes; `<`/`>` are not tags but lightweight quotes. The book is a **false friend** of code, which is exactly why it reads as 'fictional code'.

- **Molecular notation is a partial surprise.** The GenBank/EMBL `//` record terminator and FASTA `>` record header are role-compatible with the book's `//` boundary and `>` flow marker, but the book places them inline, not at line starts, so molecular notation ranks mid-pack — a real but positional mismatch.

- **Lisp/S-expressions and regex rank lowest** because the book has no parentheses and no quantifiers: its nesting is done with angle brackets and slashes, and it has no repetition operator.

- **The alphabet/role split.** C/C++ and Perl win on symbol+role coverage (the alphabet), while English punctuation — which shares **no** operator symbol with the book — reaches 47.6% role coverage, comparable to HTML/XML (54.1%), Markdown (49.0%) and molecular notation (47.8%), and ranks 4th overall. It can express the book's clause boundaries, compound links and quoting with `.`, `-`, quotes and parens. The book is best described as **English punctuation roles encoded in C-family symbols**, which is precisely the 'Meta-English / fictional code' description in the literature.

- **Limits.** No ground truth exists for 'what notation the author used'; this is a similarity ranking, not proof of influence. The notation signatures are spec-based judgments (cited, and reweightable); only the book side is measured. The 10,949-token base excludes the hyphen channels; the sensitivity table shows the winner does not depend on that choice.

---
*Generated by `work/rosetta_mapping.py`*
