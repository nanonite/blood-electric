# Within-text language viability assessment — *Blood Electric*

Chainlink #56 (revised scope: intrinsic evidence only).
Schema `language-viability/1` · generated 2026-10-10T04:34:21.027856+00:00 · seed 42 · 40 null draws per check · split: interleaved pages (even train, odd test)

**Verdict: VIABLE (BOUNDED) — 6 of 6 pre-registered checks pass.**

## 0. Question and scope

Do *Blood Electric*'s recurring lexical constructions and operator sequences support a viable, recoverable language — and where the evidence supports it, what does the text say? Per the operator's scope decision (2026-10-08), this assessment uses **only evidence from the text itself**: no external books, no reference corpora, no new datasets. The comparative keyness artifacts of the earlier #56 attempt (`analysis/keyness/`, `analysis/reference_corpora/`) are retained as historical output and are not consulted here; follow-ups #78–#80 are parked.

The working corpus is the frozen #17 operator-aware tokenizer output (`analysis/tokenizer/units.tsv`, `tokens.tsv`): **74,077 screened lexical units** over 225 pages (the #19 markup ledger + the #27 tag-run screen applied; nothing repaired), plus the #11 operator stream (**10,949 operator tokens**) and the #33 clause index (**5,614 clauses**). The grammar, entity/event and contextual-example analyses are reused, not redone: #14's AST supplies the `/` Link-vs-Slash split reconciled in §9, #30's glossary supplies each curated term's entity role and verb directions, #34's frames supply the frame-filler counts, and #26's KWIC concordancer supplies the quoted contexts in §5.1.

## 1. Method: falsifiable checks on held-out sections

A language is *viably recoverable* from a text if a learner restricted to one half of the text predicts the other half **significantly better than within-text nulls that preserve the text's own marginal statistics but destroy the structure under test**. Every check is pre-registered (thresholds fixed before running):

| check | claim under test | statistic | nulls | criterion |
|---|---|---|---|---|
| V0 | negative control: the harness must fail on a non-language | V1's machinery on the within-page-shuffled book | within-page shuffle | must NOT beat its own null |
| V1 | word order is learnable (lexical syntax) | held-out bigram CE, λ tuned on train-internal validation | within-page shuffle + type permutation | below both null minima, add-one p<0.05 |
| V2 | operator sequences are idiomatic (run syntax) | held-out operator bigram CE, Jelinek–Mercer λ=0.8 (#11's estimator) | within-run shuffle | below null minimum, add-one p<0.05 |
| V3 | constructions generalize to held-out clauses | test-clause coverage of train-mined frames (top-40) | shuffled-train frames | above null maximum, add-one p<0.05 |
| V4 | the lexicon transfers (reuse, not chance recurrence) | (a) held-out OOV rate; (b) Spearman rank correlation of shared-type frequencies | type permutation | (a) below null minimum; (b) above null maximum, add-one p<0.05 |
| V5 | term senses are stable across halves (meaning is recoverable) | median top-20 collocate Dice, same term across halves vs mismatched terms | mismatched-term pairs (within-text null) | same > mismatched, paired sign test p<0.01 |
| V6 | coinages are compositional from a stable part inventory | held-out part-sequence bigram CE within units | within-unit part shuffle | below null minimum, add-one p<0.05 |

p-values are add-one empirical: p = (1 + #{nulls at least as extreme}) / (B+1), B=40, seed 42. Verdict rule (pre-registered): ≥5 of 6 → VIABLE (BOUNDED); 3–4 → PARTIALLY VIABLE; ≤2 → NOT VIABLE. "Bounded" is earned by the bound diagnostics (§4), not assumed.

## 2. Results

| check | book | null mean | null min | null max | p | verdict |
|---|---|---|---|---|---|---|
| V0 | 9.7431 | 9.7477 | 9.7415 | 9.7545 | 0.1463 | CONTROL OK |
| V1 | 8.2619 | 10.0986 | 10.0910 | 10.1109 | 0.0244 | PASS |
| V2 | 2.6595 | 2.6927 | 2.6844 | 2.7015 | 0.0244 | PASS |
| V3 | 0.6766 | 0.6189 | 0.6115 | 0.6410 | 0.0244 | PASS |
| V4a | 0.0615 | 0.3448 | 0.2006 | 0.5142 | 0.0244 | PASS |
| V4b | 0.8340 | -0.0056 | -0.0382 | 0.0306 | 0.0244 | PASS |
| V5 | 0.5000 | 0.2863 | 0.0500 | 0.4500 | 8.4e-06 | PASS |
| V6 | 5.0014 | 6.1049 | 6.0408 | 6.2068 | 0.0244 | PASS |

Reading the table. For V1–V4 and V6 the *null* columns are 40 random draws of the stated within-text null, and the p-value is the add-one empirical p. **V0 is a control, not a scored check**: `CONTROL OK` means the harness correctly found no language in the shuffled book, which is the result the verdict needs — a FAIL there would invalidate the passes, not count against them. **V5 has no random draws**: its comparison set is the mismatched-term distribution over 40 pairs, so `n_nulls` for that row is the number of pairs and its p-value is an exact sign test.

Per-check reading:

* **V1 — lexical order (PASS).** A bigram model trained on the even pages reads the odd pages at **8.2619 bits/unit**, against 10.0986 for the within-page shuffle null and 20.4738 for the type-permutation null. The shuffle null is the informative one: it keeps the book's unigram frequencies and destroys only the ordering, so beating it is evidence about order. The type-permutation null is a *conservative* null — a random bijection of the vocabulary sends most held-out tokens to types the train half never saw, so its CE runs away to tens of bits and V1 beats it comfortably; it is reported because it is the check's second pre-registered criterion, not because it is hard to pass. The unigram baseline is 9.6744 bits/unit (order-0, `order_sweep.tsv`), so the bigram gain is 1.4125 bits — local syntax exists.
* **V2 — operator stream (PASS, thin margin).** Held-out operator bigram CE is **2.6595 bits/operator** vs 2.6927 for the within-run shuffle null. The margin is small (0.0332 bits) — this is the honest quantification of #11's "partially idiomatic": operator runs have a shallow local syntax, real but weak. Its own control is the shuffle, which preserves run composition and the operator marginals and destroys only the order within a run; V0 controls the *lexical* estimator, not this one (see §7).
* **V3 — construction transfer (PASS).** Frames mined on the train half (245 frames) cover **0.6766 of the held-out clauses** (top-40), against a shuffled-train null maximum of 0.6410. The book's recurring constructions — the `of the ⟨_⟩` NP-of-NP core, `I ⟨_⟩ the`, `the ⟨_⟩ that` — generalize to pages the miner never saw. The null is high (0.6189) because clauses share unigrams, so a skeleton can match by chance; the margin above it is the genuine construction signal.
* **V4 — lexicon reuse (PASS).** (a) Only **0.0615 of held-out tokens** are types unseen in the train half, against a type-permutation minimum of 0.2006 — the lexicon transfers, and the recurrence is not the frequency spectrum's doing. (b) Shared-type frequencies correlate at **ρ=0.8340** between halves (permutation maximum 0.0306): the book repeats itself with a stable weighting.
* **V5 — sense stability (PASS).** The top-20 content terms' collocate profiles correlate across halves: median same-term Dice **0.5000** vs 0.3000 for mismatched terms, with 34 of 40 paired comparisons positive (sign test p=8.4e-06). A term's company in the first half predicts its company in the second — senses are recoverable, not page-local accidents. Secondary statistics (`sense_stability.tsv`): top-10 Dice 0.6000 vs 0.2000; cosine over the top-30 union 0.9663 vs 0.2140. The comparison is *not* frequency-matched: a term is paired with the one 17 places further down the frequency ranking, which is typically a rarer word with a smaller profile (§7).
* **V6 — compositionality (PASS).** Held-out part-sequence bigram CE is **5.0014 bits/part** vs 6.0408 for the within-unit part-shuffle null (unigram baseline 8.8933). Coinages are built from a stable part inventory: `boy-roid`, `soul/gram`, `gene=TV` recombine parts the train half already attested, and the recombination is learnable.

## 3. Verdict

**VIABLE (BOUNDED) (6/6).** The text supports a viable, recoverable language at three levels: a **stable, reusable lexicon** (V4), **compositional morphology** (V6), and **local syntactic structure** — word order (V1), operator run syntax (V2), and recurring constructions (V3) — with **stable term senses** that make meaning extraction defensible (V5). This is a positive viability verdict with an explicit bound: everything that passes is *local* (short-context, construction-level); the bound diagnostics below show what does *not* generalize.

## 4. Where viability ends (bound diagnostics)

### D1 — order sweep: does a longer context help?

| n-gram order | λ (tuned on train-internal validation) | validation CE | held-out CE |
|---|---|---|---|
| 0 | 0.2 | 10.1384 | 9.6744 |
| 1 | 0.6 | 8.8431 | 8.2619 |
| 2 | 0.4 | 8.8376 | 8.2366 |
| 3 | 0.4 | 8.8856 | 8.2799 |

With λ tuned per order on a train-internal validation split, the best longer-context model is order-2: held-out CE is 8.2366 vs 8.2619 bits/unit for order-1, a modest improvement of 0.0253 bits/item (0.31%). Order-3 held-out CE is 8.2799 bits/unit. This difference has no separate uncertainty test, so treat it as a small short-context gain, not a robust long-range result. The book's recoverable structure remains local; the order sweep does not support a global grammar or metrical cycle. This triangulates #11 (shallow local run syntax), #47 (the operator channel gains little from context) and #51 (no latent periodicity).

### D2 — chronological stability

On the conservative contiguous split (train pp. 1-112, test pp. 113-225) the order-1 held-out CE is **8.3230 bits/unit** (λ=0.6), essentially identical to the interleaved split's 8.2619. There is no measurable drift: the language of the first half is the language of the second.

### D3 — sensitivity: without the artifact pages

Dropping the #4 OCR recognition-loop pages (89, 182, 208) and the #11/#51 HTML-artifact pages (92, 158, 219) — 219 pages remain — the order-1 held-out CE is **8.2932 bits/unit** (40 shuffle nulls, null min 10.1150, add-one p=0.0244), i.e. V1's result is unchanged when the corrupted pages are removed. Scope: this re-runs V1 only. The other five checks were not re-run without the artifact pages, so this is evidence that the *lexical* result is not an artifact of those six pages, not that every check survives their removal.

## 5. Meaning extraction

### 5.1 What the termbase supports

The curated termbase (`termbase_sample.tsv`, 49 terms) combines, per term: #19 screen category, #18 parts, #30 entity role and verb directions, book-wide collocates (±5 units, function words excluded), #34 frame-filler counts, up to two quoted #26 concordance contexts with page citations, and a gloss with a confidence. The confidence levels mean: **high** — ordinary English or directly attested entity (sense given by ordinary English plus consistent book usage); **medium** — compound/coinage whose sense is compositional (parts plus collocates) but not fully pinned; **low** — sparse attestation or a one-off construct. #26 indexed the compound and named constructs, so 21 of the 49 terms carry a quoted context and the ordinary-English terms do not — that is the concordancer's coverage, not a gap in the evidence for them. Three examples of what the evidence supports:

* **soul/gram** (compound, medium; n=313 over 136 pages): soul/gram: the self's record or measure (soul + gram, as in telegram); a recurring named construct Evidence: collocates self (45), internal (31), murder (27), cadaver (24), boy-roid (21), dog (18); frame filler: the ⟨_⟩ of x75; of the ⟨_⟩ x43; ⟨_⟩ of the x38. context (#26): p4: gas of boy-roid//The «soul/gram» of self crushes to the … p4: in the Cadaver City-city// «Soul/gram» of which liquefied internal body. 
* **tokage** (proper noun, high; n=127 over 84 pages): ToKAGE: a named persona (does: is/was/fecundates/murder; done-to: love/speed/like/crucified) Evidence: collocates body (11), brain (10), love (9), internal (8), cadaver (7), sun (7); frame filler: of ⟨_⟩ that x18; ⟨_⟩ of the x8; ⟨_⟩ to the x4. context (#26): p4: maso=traffic::of «ToKAGE» that crowds in the Cadaver … p5: //++the brain of «ToKAGE» is respired//Of quantum. entity role: persona; does: is(12)|was(8)|fecundates(5)|murder(3)|sun(3); done-to: love(7)|speed(6)|like(5)|crucified(4)|sun(4).
* **artificial-sun** (compound, high; n=29 over 16 pages): artificial-sun: the not-natural sun; the book's dominant content collocation Evidence: collocates line (4), self (4), internal (4), wild (3), level (3), scream (2); frame filler: of the ⟨_⟩ x18; ⟨_⟩ of the x2; the ⟨_⟩ of x2. context (#26): p4: //it joints to the «artificial-sun» territory: the radical restraint … p6: ++inoculation mouth//The «artificial-sun» is restrained to our inorganic. 

### 5.2 Sample glosses (page-cited)

Three passages are glossed token-by-token in `gloss_passages.md` (full tables) and `gloss_tokens.tsv` (one row per token, with the #17 uid, #19 category, #18 parts, #30 entity role, #34 frame fillers). Each passage is rendered three ways: the original, the #13 v3 operator-normalized form (operators → space), and the v4 form (operators → period). Headline readings:

* **Passage 1 — the opening block (p. 4)** (p. 4): 23 tokens, 21 glossed high, 2 medium, 0 low, 0 unresolved. The plain-English skeleton is fully recoverable; the residue is listed, not repaired.
* **Passage 2 — ToKAGE in Cadaver City (p. 4)** (p. 4): 16 tokens, 14 glossed high, 1 medium, 1 low, 0 unresolved. The plain-English skeleton is fully recoverable; the residue is listed, not repaired.
* **Passage 3 — the artificial-sun circuit (p. 6)** (p. 6): 32 tokens, 31 glossed high, 1 medium, 0 low, 0 unresolved. The plain-English skeleton is fully recoverable; the residue is listed, not repaired.

Passage 1 in brief — original (p. 4):

> I suck the porno nerve gas of boy-roid//The soul/gram of self crushes to the quantum tragedy of NDRO//

Constrained literal translation (confidence per token in `gloss_tokens.tsv`):

> «I suck the pornographic nerve-gas of boy-roid. The soul/gram of the self crushes to the quantum tragedy of NDRO.»

Every word in that rendering is either ordinary English (high confidence) or a named construct whose *referential status* is directly evidenced (`boy-roid` a recurring compound persona, `soul/gram` a recurring construct, `NDRO` a named persona). What is *not* claimed: what a boy-roid is, or what the quantum tragedy of NDRO is *about* — the text does not say, and the assessment does not ventriloquize.

## 6. Ambiguous cases and unexplained residue

Quantified in `residue_ledger.tsv` and `oov_types.tsv`:

* **Held-out residue proper:** 1883 types / 2273 tokens (6.2% of the test half) are never attested in the train half — the hard bound on recoverability. By #19 category: compound=997; english=848; true_coinage=17; rare_english=7; numeric=5; ocr_error=4; proper_noun=4; =1.
* **Ambiguous operator roles:** the `/` dual role (compound linker vs sentence-final separator), the `::` vs `//` boundary tension (#12, #50), and the `=` naming seam vs markup collision (`href=`) — each measured, each disclosed; none repaired.
* **Disclosed confound:** the `second` filler of `of the ⟨_⟩` is inflated by the page-89 OCR recognition loop (#34's confound, re-confirmed here).
* **Unexplained residue:** 17 of 17 held-out true-coinage types are attested once, unattested in the train half — no within-text evidence can fix their sense; they are listed in `oov_types.tsv` with their page.

## 7. Limitations

* **OCR is the floor.** The corpus is marker OCR of a scan; the #4 ensemble (tesseract 97.9% / docTR 97.3% / easyocr 93.9% character agreement) bounds the noise, and the D3 sensitivity run shows V1's result does not depend on the three recognition-loop pages — but per-token confidence in the glosses inherits OCR uncertainty.
* **Local structure only.** The lexical order sweep shows whether adding up to two prior items helps, and the operator check tests only bigrams; neither tests long-range narrative, global grammar, or a generative system. The D1 higher-order difference is small and has no separate uncertainty test.
* **Glosses are constrained, not translations.** The translation in §5.2 is a literal rendering of the recoverable skeleton; coinages are glossed compositionally (parts + collocates) and marked medium or low where the sense is not pinned. Where the text does not say, the assessment says so (unresolved tokens are listed, never repaired).
* **The nulls are within-text.** Type permutation and shuffles preserve the book's own statistics; they control for chance recurrence *of this text*, not for any external notion of language. That is the revised scope: intrinsic viability, deliberately.
* **Estimator choices are disclosed.** Jelinek–Mercer interpolation with λ tuned on a train-internal validation split (V1, D1); #11's λ=0.8 for the operator stream (V2); top-20 Dice for sense stability (V5). The V0 negative control shows the harness fails a non-language, so the passes are not vacuous.
* **V0 controls the lexical estimator, not the operator one.** The negative control runs V1's bigram machinery on a shuffled book. No equivalent control exists for V2: the operator stream is never word-order-shuffled, so V2's margin (0.0332 bits) rests on its within-run shuffle null alone. That null preserves run composition, so it cannot rule out an estimator artifact specific to operator runs. V2 is the thinnest passing check and should be read with that gap in mind.
* **V5's comparison is not frequency-matched.** Each term is paired with the term 17 places further down the frequency ranking, which is usually a rarer word with a smaller profile. Some of the same-vs-mismatched gap is therefore frequency, not sense. The check still bounds what it claims — term-specific company is recoverable across halves — but it over-states the size of the effect.
* **The type-permutation nulls are conservative.** A random vocabulary bijection maps most held-out tokens outside the train lexicon, so V1b and V4a are won by a wide margin; they are weak evidence, kept because they were pre-registered, not because they discriminate.

## 8. Reproduction

```bash
conda activate marker
python work/language_viability.py            # full run (~1 min)
python work/language_viability.py --nulls 10 # quick run
python work/test_language_viability.py       # regression suite
```

Deterministic: fixed seed 42, fixed tie-breaks. Every statistic and every table recomputes identically; the only field that varies between runs is the ``generated`` stamp above, which is provenance (as in #30) and no statistic is derived from. All statistics recompute from the frozen artifacts listed in §0.

## 9. Reconciliation with prior work

| artifact | frozen | recomputed | match |
|---|---|---|---|
| #10 operator inventory | 10,949 body operators | 10949 operator tokens in the #11 stream | True |
| #17 tokenizer (screened units) | 74,077 units (74,318 minus 241 tag-furniture) | 74077 screened units | True |
| #33 clause segmentation | 5,614 clauses | 5614 clauses | True |
| #11 operator stream | 90,220 tokens incl. 11,000 operators (body+header) | 10949 operators (body pages only; headers add 51) | True |
| #34 frames (book-wide, descriptive) | 675 frames (646 single / 29 two-slot) | 675 frames, mined on all 5614 clauses for the termbase's filler evidence (descriptive, not a held-out check) | True |
| #14 grammar: `/` Link vs Slash | 952 Link / 540 Slash nodes | 952 glued / 540 boundary-like slash tokens (#17) | True |
| #26 KWIC concordance coverage of the termbase | not a frozen count — coverage reported, not gated | 21 of 49 termbase terms carry a cited context | True |

The operator rows agree exactly: 10949 is #10's body count, and the #11 stream's remaining 51 header operators take it to the 11,000 of #17's full-book figure. The frame row is a full recount, not a reuse: 675 frames over all 5614 clauses, which reproduces #34's published catalogue frame-for-frame (descriptive evidence for the termbase, never a held-out check — V3's frames are mined on the train half alone). The #14 row compares two independent counts of the same phenomenon — #14's AST node classes and this module's recount of the #17 token stream — and they agree exactly, which is what licenses treating the `/` dual role as measured rather than as a reading. The #26 row reports coverage and is not gated.
