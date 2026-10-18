# Operator-word mutual information (#52)

Source: #17 token layer (`tokens.tsv`) for operator positions and #17 unit layer (`units.tsv`) for the terms they stand next to; markup screen from #19/#27; domain seed sets parsed from #29's `DOMAIN_SEEDS` in `work/cooccurrence_network.py`. Reproducible via `python work/operator_word_mi.py`; regression suite `python work/test_operator_word_mi.py`.

Schema `operator_word_mi/1`. MI in bits (log2). Right side = the term to the *right* of the operator (the issue's "`::` before body terms"); left side = the term to its left. The permutation null is 200 label permutations, seed 42, both marginals preserved.

## 1. Headline result

- **10,268 screened operator occurrences** (732 raw-HTML operator tokens inside #27's markup spans removed; #10's page-body count keeps them, which is the whole difference). 2,815 sit inside a #17 unit (`=` in `suck=blood`), 7,453 are free between units.
- **The operator predicts the adjacent term.** Right side: I(O;T) = 1.1938 bits against a null of 0.5488 ± 0.0052 (z = 124.779, p_emp = 0.005); the domain-coarsened I(O;D) = 0.0776 bits (null 0.0089 ± 0.0009, z = 73.554); the incidence multi-information of the MI matrix is 0.1692 bits (null 0.0185 ± 0.0035). Knowing the operator lowers the adjacent term's uncertainty from H(T) = 7.3943 to H(T|O) = 6.2005 bits.
- **Both sides predict, the right side more sharply.** Left side: I(O;T) = 1.2013 bits against null 0.695 ± 0.0056 (z = 90.093), H(T) = 9.1597 to H(T|O) = 7.9584 bits. The absolute MI is nearly symmetric, but the left term distribution is itself more spread (any word can precede an operator), so *relative to its own null* the right side is the stronger predictor (2.2x null vs 1.7x), and the incidence multi-information is twice as large (0.1692 vs 0.0881 bits).
- **Scale check.** The operator-to-term MI (1.1938 bits) is 9x #11's operator-to-operator lag-1 MI (0.1305 bits). The operator stream is nearly independent of itself but strongly tied to the words it touches — which is what #12's role classification asserts and what #11's entropy curve could not see.
- **The issue's two hypotheses, tested.** `::` before body terms: **rejected** — `::` significantly *avoids* body terms on its right (lift 0.39, q = <1e-4). `=` before tech terms: **confirmed** — `=` precedes a machine term with lift 2.49 (q = <1e-4), and an even stronger body preference (lift 2.94): `=` is a cross-domain binder, genetics/machine on the left, body/machine on the right. Details in §7.

## 2. Method

### 2.1 The stream

Every screened `OP` token of #17's token layer is one observation. Its terms are resolved on the #17 unit layer: a free operator takes the nearest surviving unit surface on each side; an operator inside a unit takes the operand substrings of its own surface (`=` in `gene=TV/spasm` has `gene` left and `TV/spasm` right). A side with no surviving unit is `boundary`; a term in no seed set is `none`.

### 2.2 Terms and domains

#29's six seeded domains (body, machine, violence, sex, genetics, corporate; 117 seed terms). A term counts toward a domain when the term itself *or* one of its seam-split parts is in that domain's seed set (`drug-embryo` -> `embryo` = body+genetics). A term may carry several domains; the whole-term-only variant is the #29-faithful sensitivity check in §8.

Coverage is the limit of this design and is stated plainly: #29's 117 seed terms are 16.07% of right-hand neighbours and 29.98% of left-hand neighbours (1,650 and 3,078 occurrences; the rest are `none` or `boundary`). The MI matrix is therefore a statement about the *domain-bearing minority* of adjacent terms, and the term-level totals I(O;T) in §5 cover the whole stream.

One seed-set note, reported because it changes a cell: #29's source comment says ``blood`` is *body + violence*, but the literal ``DOMAIN_SEEDS['violence']`` does not contain ``blood``. This build follows the literal (so ``blood`` is body-only), which is why the violence column is thinner than #29's prose implies.

### 2.3 Statistics

- **MI matrix** (§3): pointwise MI (bits) per (operator, domain) over the incidence table, with expected count and lift.
- **Per-domain binary tests** (§4): a 1-df G² per (operator, domain) with Benjamini-Hochberg q over each 6 x 11 grid. A multi-domain term is an incidence in each of its domains, so the six domain tests are not mutually exclusive.
- **Totals** (§5): I(O;D) on the domain-combination label (single-label, a true mutual information), I(O;T) on the term itself, and the incidence multi-information; each against a label-permutation null that preserves both marginals.
- **Two lift bases, both published.** The matrix's `lift` and `pmi_bits` are computed on the *incidence* table (a multi-domain term is one incidence in each of its domains), while `p_domain_given_op` / `p_domain_base` — and `operator_profile.tsv`'s `lift` — are occurrence rates (the operator's domain-bearing share against the stream's). The two bases agree in sign and rank; e.g. `<<`/body is 1.53 on incidences and 1.72 on rates.
- **Robustness** (§8): whole-term-only seed matching, book halves, and the operator inventory reconciliation.

## 3. The MI matrix (operator x domain)

### 3.1 right side — the term to the right of the operator

n = 10,268 operator occurrences; 1,753 domain incidences (102 occurrences carry more than one domain). I(O;D) = 0.0776 bits, I(O;T) = 1.1938 bits, multi-information = 0.1692 bits.

Cell = count (PMI bits); `—` marks a cell that never occurs. `none` = a term in no seed set, `boundary` = no surviving unit on that side.

| operator | n | `body` | `machine` | `violence` | `sex` | `genetics` | `corporate` | none | boundary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `+` | 15 | 2 (+0.59) | 0 (—) | 1 (+1.13) | 1 (+2.16) | 0 (—) | 0 (—) | 12 | 0 |
| `++` | 198 | 2 (-2.31) | 12 (+0.26) | 3 (-0.19) | 2 (+0.25) | 11 (+1.16) | 0 (—) | 172 | 0 |
| `/` | 1,355 | 30 (-0.65) | 41 (-0.21) | 22 (+0.44) | 13 (+0.71) | 35 (+0.59) | 1 (+2.04) | 1,201 | 23 |
| `//` | 2,149 | 86 (-0.15) | 74 (-0.37) | 42 (+0.35) | 9 (-0.84) | 76 (+0.69) | 1 (+1.02) | 1,864 | 26 |
| `::` | 3,084 | 68 (-1.04) | 150 (+0.10) | 64 (+0.41) | 31 (+0.40) | 108 (+0.65) | 0 (—) | 2,696 | 5 |
| `<` | 148 | 4 (-0.41) | 2 (-1.41) | 3 (+0.72) | 3 (+1.75) | 4 (+0.61) | 0 (—) | 132 | 0 |
| `<<` | 432 | 42 (+0.61) | 9 (-1.62) | 17 (+0.84) | 7 (+0.59) | 8 (-0.76) | 0 (—) | 351 | 0 |
| `=` | 1,933 | 322 (+0.53) | 274 (+0.29) | 34 (-1.17) | 20 (-0.91) | 21 (-2.39) | 0 (—) | 1,273 | 0 |
| `>` | 531 | 10 (-0.73) | 11 (-0.60) | 10 (+0.81) | 8 (+1.52) | 11 (+0.43) | 0 (—) | 482 | 1 |
| `>>` | 406 | 15 (-0.05) | 11 (-0.51) | 4 (-0.42) | 3 (+0.19) | 13 (+0.76) | 1 (+3.64) | 359 | 5 |
| `|` | 17 | 0 (—) | 0 (—) | 0 (—) | 1 (+4.16) | 0 (—) | 0 (—) | 16 | 0 |

### 3.2 left side — the term to the left of the operator

n = 10,268 operator occurrences; 3,306 domain incidences (224 occurrences carry more than one domain). I(O;D) = 0.0722 bits, I(O;T) = 1.2013 bits, multi-information = 0.0881 bits.

Cell = count (PMI bits); `—` marks a cell that never occurs. `none` = a term in no seed set, `boundary` = no surviving unit on that side.

| operator | n | `body` | `machine` | `violence` | `sex` | `genetics` | `corporate` | none | boundary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `+` | 15 | 0 (—) | 0 (—) | 3 (+2.88) | 0 (—) | 0 (—) | 0 (—) | 12 | 0 |
| `++` | 198 | 16 (+0.41) | 16 (+0.02) | 12 (+0.68) | 0 (—) | 11 (-0.49) | 0 (—) | 149 | 0 |
| `/` | 1,355 | 72 (+0.28) | 93 (+0.26) | 33 (-0.17) | 15 (-0.46) | 57 (-0.43) | 2 (+2.28) | 1,105 | 0 |
| `//` | 2,149 | 185 (+0.31) | 188 (-0.06) | 79 (-0.23) | 31 (-0.74) | 199 (+0.05) | 1 (-0.05) | 1,541 | 0 |
| `::` | 3,084 | 279 (+0.44) | 287 (+0.09) | 101 (-0.34) | 27 (-1.40) | 245 (-0.11) | 2 (+0.49) | 2,220 | 3 |
| `<` | 148 | 3 (-0.87) | 12 (+0.75) | 1 (-1.77) | 1 (-0.92) | 8 (+0.18) | 0 (—) | 122 | 2 |
| `<<` | 432 | 24 (+0.13) | 29 (+0.02) | 23 (+0.76) | 3 (-1.34) | 21 (-0.42) | 0 (—) | 333 | 6 |
| `=` | 1,933 | 63 (-1.71) | 259 (-0.06) | 154 (+0.26) | 161 (+1.17) | 308 (+0.21) | 0 (—) | 1,005 | 0 |
| `>` | 531 | 47 (+0.40) | 38 (-0.30) | 20 (-0.15) | 9 (-0.46) | 49 (+0.09) | 0 (—) | 381 | 0 |
| `>>` | 406 | 32 (+0.40) | 23 (-0.47) | 22 (+0.54) | 4 (-1.08) | 30 (-0.06) | 0 (—) | 299 | 1 |
| `|` | 17 | 2 (+0.19) | 1 (-1.19) | 2 (+0.88) | 0 (—) | 3 (+0.41) | 0 (—) | 11 | 0 |

## 4. Which operators select which domains

### 4.1 right side

`MI contrib.` is the operator's share of the side's I(O;T) — how much of the total prediction each operator carries.

| operator | n | top domain | P(d|op) | P(d) | lift | q | MI contrib. | none | boundary | verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `::` | 3084 | machine | 4.9% | 5.7% | 0.86 | 0.14 | 0.200 | 87% | 0% | no preference |
| `//` | 2149 | body | 4.0% | 5.7% | 0.71 | 0.0028 | 0.185 | 87% | 1% | avoids |
| `=` | 1933 | body | 16.7% | 5.7% | 2.94 | <1e-4 | 0.344 | 66% | 0% | over-selects |
| `/` | 1355 | machine | 3.0% | 5.7% | 0.53 | <1e-4 | 0.157 | 89% | 2% | avoids |
| `>` | 531 | machine | 2.1% | 5.7% | 0.36 | 0.00029 | 0.070 | 91% | 0% | avoids |
| `<<` | 432 | body | 9.7% | 5.7% | 1.72 | 0.0043 | 0.078 | 81% | 0% | over-selects |
| `>>` | 406 | body | 3.7% | 5.7% | 0.65 | 0.2 | 0.055 | 88% | 1% | no preference |
| `++` | 198 | machine | 6.1% | 5.7% | 1.07 | 0.94 | 0.045 | 87% | 0% | no preference |
| `<` | 148 | body | 2.7% | 5.7% | 0.48 | 0.24 | 0.037 | 89% | 0% | no preference |
| `|` | 17 | sex | 5.9% | 1.0% | 6.16 | 0.36 | 0.013 | 94% | 0% | no preference |
| `+` | 15 | body | 13.3% | 5.7% | 2.36 | 0.49 | 0.009 | 80% | 0% | no preference |

### 4.2 left side

`MI contrib.` is the operator's share of the side's I(O;T) — how much of the total prediction each operator carries.

| operator | n | top domain | P(d|op) | P(d) | lift | q | MI contrib. | none | boundary | verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `::` | 3084 | machine | 9.3% | 9.2% | 1.01 | 0.91 | 0.175 | 72% | 0% | no preference |
| `//` | 2149 | genetics | 9.3% | 9.1% | 1.02 | 0.82 | 0.160 | 72% | 0% | no preference |
| `=` | 1933 | genetics | 15.9% | 9.1% | 1.76 | <1e-4 | 0.319 | 52% | 0% | over-selects |
| `/` | 1355 | machine | 6.9% | 9.2% | 0.74 | 0.008 | 0.175 | 82% | 0% | avoids |
| `>` | 531 | genetics | 9.2% | 9.1% | 1.02 | 0.92 | 0.083 | 72% | 0% | no preference |
| `<<` | 432 | machine | 6.7% | 9.2% | 0.73 | 0.16 | 0.086 | 77% | 1% | no preference |
| `>>` | 406 | body | 7.9% | 7.0% | 1.12 | 0.66 | 0.081 | 74% | 0% | no preference |
| `++` | 198 | body | 8.1% | 7.0% | 1.15 | 0.69 | 0.052 | 75% | 0% | no preference |
| `<` | 148 | machine | 8.1% | 9.2% | 0.88 | 0.74 | 0.052 | 82% | 1% | no preference |
| `|` | 17 | genetics | 17.6% | 9.1% | 1.95 | 0.43 | 0.008 | 65% | 0% | no preference |
| `+` | 15 | violence | 20.0% | 4.4% | 4.56 | 0.08 | 0.009 | 80% | 0% | no preference |

## 5. Right vs left

| side | I(O;T) bits | null | z | I(O;D) bits | null | z | multi-info bits | null | z | H(T) | H(T|O) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| right | 1.1938 | 0.5488 ± 0.0052 | 124.779 | 0.0776 | 0.0089 ± 0.0009 | 73.554 | 0.1692 | 0.0185 ± 0.0035 | 43.207 | 7.3943 | 6.2005 |
| left | 1.2013 | 0.695 ± 0.0056 | 90.093 | 0.0722 | 0.0097 ± 0.0011 | 56.422 | 0.0881 | 0.0102 ± 0.0021 | 37.112 | 9.1597 | 7.9584 |

The two sides are close in absolute bits but get there differently: the right-hand term is far more predictable in absolute terms (H(T|O) 6.20 vs 7.96 bits) because an operator constrains what follows it, while the left term varies widely (H(T) 9.16 bits). Relative to its own null — the comparison that matters, since the null moves with the term distribution — the right side is the stronger predictor (2.2x vs 1.7x null), and the domain structure is twice as concentrated (multi-information 0.1692 vs 0.0881 bits).

## 6. The seed terms behind the cells (right side)

Operator-word PMI restricted to #29's seed vocabulary, f >= 3, top 8 per operator. Terms are #17 unit surfaces with their original case (`Defleshed`, `Body-OMOTYA`). This is the lexical evidence for the domain cells; #27's `collocations_bigram_token.tsv` is the full operator-word table on a token-bigram basis.

- **`::`** (3,084 occurrences): `swastika-girl` 3x (PMI 1.74, sex), `clone-transmission` 4x (PMI 1.41, genetics|machine), `love` 4x (PMI 1.41, sex), `hunting` 7x (PMI 1.37, violence), `sexual` 3x (PMI 1.32, sex), `tomb-device` 3x (PMI 1.32, machine), `vital=serum` 3x (PMI 1.32, genetics), `cyber` 5x (PMI 1.25, machine)
- **`//`** (2,149 occurrences): `Defleshed` 7x (PMI 2.26, violence), `Masses` 6x (PMI 2.03, body), `Murderous` 4x (PMI 1.93, violence), `Output` 4x (PMI 1.93, machine), `Boy-roid` 8x (PMI 1.80, genetics), `Body-OMOTYA` 5x (PMI 1.77, body), `Clone` 4x (PMI 1.67, genetics|machine), `Murder` 4x (PMI 1.45, violence)
- **`=`** (1,933 occurrences): `boy` 3x (PMI 2.41, genetics), `less=masses` 3x (PMI 2.41, body), `system` 3x (PMI 2.41, machine), `cell` 91x (PMI 2.39, body), `blood` 124x (PMI 2.39, body), `TV` 164x (PMI 2.38, machine), `code` 13x (PMI 2.30, machine), `hologram` 12x (PMI 2.29, machine)
- **`/`** (1,355 occurrences): `assassin` 4x (PMI 1.22, violence), `clone` 4x (PMI 0.75, genetics|machine), `ADAM` 11x (PMI 0.45, genetics), `digital` 3x (PMI 0.42, machine), `brain` 4x (PMI 0.34, body), `LOAD` 8x (PMI 0.19, machine), `murder` 5x (PMI -0.00, violence), `body` 4x (PMI -0.25, body)
- **`>`** (531 occurrences): `ADAM` 5x (PMI 0.66, genetics)
- **`<<`** (432 occurrences): `entrails` 30x (PMI 3.48, body), `defleshed` 4x (PMI 2.25, violence), `murder` 6x (PMI 1.91, violence)
- **`>>`** (406 occurrences): `masses` 3x (PMI 1.66, body), `body` 3x (PMI 1.08, body), `LOAD` 4x (PMI 0.93, machine)
- **`++`** (198 occurrences): `boy-roid` 3x (PMI 2.58, genetics)

## 7. The issue's questions

- **`::` before body terms?** (phrase/field separator over flesh): `::` precedes a `body` term 68 times; P(body|`::`) = 2.2% against a base rate of 5.7% — lift 0.39, G² = 88.6877, q = <1e-4. Verdict: **avoids**.
- **`=` before tech terms?** (naming/identity linker): `=` precedes a `machine` term 274 times; P(machine|`=`) = 14.2% against a base rate of 5.7% — lift 2.49, G² = 187.5354, q = <1e-4. Verdict: **over-selects**.
- **`//` before violence terms?** (utterance boundary): `//` precedes a `violence` term 42 times; P(violence|`//`) = 2.0% against a base rate of 1.9% — lift 1.00, G² = 0.0005, q = 0.98. Verdict: **no preference**.
- **`>>` before body terms?** (block close): `>>` precedes a `body` term 15 times; P(body|`>>`) = 3.7% against a base rate of 5.7% — lift 0.65, G² = 3.3227, q = 0.2. Verdict: **no preference**.

The `=` row deserves the two-sided reading: on the left it over-selects **genetics** terms (P(genetics|`=` left) = 15.9% vs base 9.1%, lift 1.76, q = <1e-4), and on the right body and machine terms. `=` is the book's naming/identity seam (#12) and it binds *across* domains: `suck=blood` (sex to body), `fear=cell` (violence to body), `gene=TV` (genetics to machine), `machine=angel`. The operator grammar's linker is where the book's semantic fusion is syntactic.

The `<<` block-open delimiter is the one operator with a strong, unambiguous domain preference: body terms follow it at lift 1.72 (q = 0.0043, driven by `entrails` 30x at PMI 3.48), while it avoids machine terms. #14's paired `<<…>>` spans are quoted blocks of flesh; `>>` itself shows no equivalent preference on closing.

## 8. Robustness and sensitivity

| variant | I(O;D) right | I(O;D) left | incidences right | hits only from parts, right |
|---|---:|---:|---:|---:|
| default (term or seam part) | 0.0776 | 0.0722 | 1,753 | 324 |
| strict whole term (#29-faithful) | 0.0777 | 0.0744 | 1,352 | — |

The seam-split rule adds 324 right-side occurrences that match a domain only through a compound part (`drug-embryo` -> `embryo`, `body-OMOTYA` -> `body`), and it moves I(O;D) by less than a hundredth of a bit on either side: the cells that carry the findings are single-word seeds (`blood`, `TV`, `cell`, `entrails`). The term-level totals I(O;T) do not depend on the seed sets at all, so only the domain quantities appear in this table.

Book halves (first half / second half):

| side | I(O;D) first | I(O;D) second | I(O;T) first | I(O;T) second |
|---|---:|---:|---:|---:|
| right | 0.0838 | 0.0882 | 1.2728 | 1.3078 |
| left | 0.0753 | 0.0822 | 1.3799 | 1.3206 |

The totals are stable across halves on both sides, so the operator-word association is a property of the whole book rather than a run of pages.

## 9. Reconciliation

| quantity | earlier | here | note |
|---|---|---|---|
| operator tokens (#17) | 11,000 | 11,000 | #17 token layer, all regions |
| operator occurrences after markup screen | 10,949 body (#10) | 10,268 | minus 732 raw-HTML `<`/`>` tokens #10 counted (pp. 92/158/219); includes 51 header-region operators #17's stream carries |
| in-unit operators | 2,816 (#17 seams) | 2,815 | 2,816 #17 seams minus the one in-seam `=` inside a markup span (`fluid=streaming`, p29); geometric containment reproduces #17's `unit` column on all 10,268 surviving tokens, and 0 free tokens sit inside a unit span |
| operator MI lag-1, bits (#11) | 0.1305 | 1.1938 | different second variable: #11 is operator-to-operator, here operator-to-term (right side) |
| operator-term adjacency observations | 1,903 rows (#27) | 10,208 | #27's table rows are operator-word token-bigram *types*; this build counts occurrences, and its neighbour is the nearest unit even across intervening operators |
| domain seeds (#29) | 6 domains | 6 | 117 seed terms, parsed from #29's DOMAIN_SEEDS |
| units (#17) | 74,665 | 74,665 | 588 markup-screened out of the neighbour resolution |

## 10. Caveats and limitations

- **Seed sets are a reading, not the book's own categories.** #29/#73 already showed the six domains do not converge on the distributional clusters; the MI matrix measures the operator's predictive structure *relative to that seed reading*, not an objective domain inventory.
- **In-unit operators bind their own operands.** `=` binds `suck` to `blood` inside one unit, so the `=` row is a statement about compounds while the row for a free operator such as `::` is a statement about unit sequence. The `n_op` column and the diagnostics separate the two.
- **The asymmetry between sides is real but partly definitional.** A free operator's left term is the *end* of the previous unit and its right term the *start* of the next; the two sides are reported separately so no pooled number hides the difference.
- **Rare operators** (`+` n=15, `|` n=17 after screening) have cells far too sparse for a domain claim; their rows are reported but their q values should not be read as evidence.
- **Multi-domain terms double-count incidences** in the matrix (a term in two domains is one incidence in each). The single-label totals I(O;D) and the per-operator profile shares do not double-count.
- **Plug-in MI is upward-biased** at these sample sizes; the permutation null is the honest comparison and is what the reported z and empirical p refer to.
- **The predictions are distributional, not semantic.** That `::` precedes body terms is a statement about which seed words sit on which side of an operator; whether it means "`::` introduces flesh" is #12's interpretive question.

## 11. Artifacts

| artifact | content |
|---|---|
| `mi_matrix.tsv` | the MI matrix: one row per (side, operator, domain) with count, expected, PMI bits and lift (incidence basis) plus the occurrence-rate columns P(d|op) / P(d) and G², p, q |
| `operator_profile.tsv` | per-operator summary: n, top domain, rates, lift, q, none/boundary/multi shares, verdict |
| `operator_seed_terms.tsv` | (side, operator, seed term) pairs with PMI and lift — the word-level evidence |
| `mi_summary.tsv` | the three MI totals per side with the permutation null (mean, sd, z, empirical p) |
| `diagnostics.tsv` | screening and stream counts |
| `reconciliation.tsv` | this build vs #10/#11/#17/#27/#29 |
| `summary.json` | machine-readable headline numbers |

---
*Generated by `work/operator_word_mi.py`*
