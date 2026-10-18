# Interlinear edition -- generator and first full-book rendering (#62)

Generated: 2026-10-10 14:56 UTC

Scope: all 225 pages, 5,614 clauses

## What was built

`work/interlinear_edition.py` renders every #33 clause in three
aligned lines -- original clause | operator-normalized (silence
removed) | plain-English gloss -- and writes the whole book to
`analysis/interlinear/interlinear.md`, with the per-clause TSV,
the per-token reading table, the readings lexicon, the residue
ledger and the reconciliation.  It is the full-book follow-up to
#56's three glossed passages and reuses #56's token rules, curated
termbase, operator readings and v3/v4 renderer unchanged.

## Method

* **Line 2 is measured, not invented.**  It is #13's v4 transform
  (every operator -> `. `) over the #17 token stream, with #56's
  `render_variant` semantics; v4 is the variant #50 scored closest
  to the underlying system (perplexity 84.1 vs 106.2 baseline).
  v3 (operators -> space) is carried in `interlinear.tsv`, so the
  middle line is overridable.  Two build-time invariants pin it: for
  every clause the token-assembled rendering equals the rendering
  recomputed by deleting the operator spans from the source
  substring, and every kept token's span reads back as its surface.
* **Line 3 substitutes only what is not ordinary English.**  The
  book is English lexis under a symbolic transform (#56's VIABLE
  verdict), so English/function words, numeric tokens, compounds
  of English parts and named constructs are kept verbatim, and the
  gloss line equals the v3-normalized line except at the
  non-English sites.  The overlap is a reported number below, not
  a hidden one.  Seam operators inside compounds (`soul/gram`,
  `gene=TV`) are dropped on line 3 exactly as #13's v3 drops them
  -- the parts stay separate words, never fused into #13's v1
  portmanteau (`soulgram`) -- and `gloss_tokens.tsv` records the
  containing #17 unit for every part, so the construct is
  recoverable from the table.
* **Readings are evidence-linked or unresolved.**  The rule order
  per token is published in the module docstring; the readings for
  the #19 `true_coinage` class are authored from the #26 KWIC
  contexts, #19's morphology and #30's roles, with #56's
  confidence scheme.  No LLM is used (that is #63's committee) and
  no external corpus is consulted or acquired.

## Headline numbers

| quantity | value |
|---|---|
| clauses rendered | 5,614 |
| pages | 225 |
| word tokens in clauses | 76,897 |
| silence tokens (operators/punctuation) | 5,752 |
| markup-excluded tokens (#19/#27 screen) | 1,657 |
| word tokens keeping an identity reading | 76,542 (99.5%) |
| curated plain readings (types) | 49 |
| unresolved word tokens | 4 (0.01% of words) |
| clauses with >= 1 unresolved surface | 4 (0.1%) |

Confidence over the 76,897 in-clause word tokens: high 71,980 / medium 4,771 / low 142
(unresolved 4 are itemized in residue.tsv).

Reading sources over the same tokens:

| source | tokens | share |
|---|---|---|
| #62 curated reading (PLAIN_READINGS) | 351 | 0.4% |
| identity -- ordinary English per the #19 screen | 71,664 | 86.7% |
| identity -- numeric channel (#21), kept verbatim | 60 | 0.1% |
| identity -- compound of English parts (#19), kept intact | 4,111 | 5.0% |
| identity -- named construct (#30); names are not translated | 307 | 0.4% |
| identity -- #19 `ocr_error` class, kept as printed | 128 | 0.2% |
| unresolved -- #19 `junk` markup residue | 1 | 0.0% |
| identity -- word-part inside a #17 unit | 272 | 0.3% |
| unresolved -- no evidence-linked reading (residue.tsv) | 3 | 0.0% |
| removed on the gloss line (operator/punctuation silence) | 5,752 | 7.0% |

## Samples

### p. 4 -- clause 37 (body region, opener `//`)

```text
1| it joints to the artificial-sun territory: the radical restraint line of exoskeleta >> the XX brain universe that is provoked and touched
2| it joints to the artificial-sun territory: the radical restraint line of exoskeleta. the XX brain universe that is provoked and touched
3| it joints to the artificial-sun territory: the radical restraint line of outer skeletons the XX brain universe that is provoked and touched
```

readings: high 22, medium 1, low 0, unresolved 0

### p. 149 -- clause 3769 (body region, opener `::`)

```text
1| oneiric locus of head is shut down
2| oneiric locus of head is shut down
3| dream-like locus of head is shut down
```

readings: high 7, medium 0, low 0, unresolved 0

### p. 223 -- clause 5610 (body region, opener `//`)

```text
1| The huge rape drone of the soul/gram who excavated inside the underground cervical  vertabre of the artificial arms of commute and avaled anthe planet produc the
2| The huge rape drone of the soul. gram who excavated inside the underground cervical  vertabre of the artificial arms of commute and avaled anthe planet produc the
3| The huge rape drone of the soul gram who excavated inside the underground cervical  vertebrae of the artificial arms of commute and avaled anthe planet produc the
```

readings: high 27, medium 1, low 0, unresolved 0

## Residue (listed, not repaired)

| class | item | n | pages | verdict |
|---|---|---|---|---|
| unresolved_unresolved | omanko | 2 | 108,129 | residue -- listed, not repaired (L9.8) |
| unresolved_junk_19 | href | 1 | 92 | residue -- listed, not repaired (L9.8) |
| unresolved_unresolved | laple | 1 | 222 | residue -- listed, not repaired (L9.8) |
| ocr_error_kept | lt | 61 | 5-223 (51 pages) | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | boyroid | 24 | 4-221 (22 pages) | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | cadavermechanism | 8 | 15,22,30,33,36,129,132,196 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | soulmachine | 8 | 50,54,67,69,73,190 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | coldblooded | 5 | 16,116,118,139,198 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | desireprotocol | 5 | 25,144,155,169,170 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | technojunkies | 5 | 33,160,166,169,173 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | cadaverfeti | 3 | 30,128,199 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | drugembryo | 2 | 11,113 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | metalcongenital | 2 | 37,151 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | clonetransmission | 1 | 51 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | desirescript | 1 | 128 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | existencecode | 1 | 170 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | fleshmodule | 1 | 128 | kept as printed -- never auto-corrected (L0.4) |
| ocr_error_kept | nanomachine | 1 | 31 | kept as printed -- never auto-corrected (L0.4) |

The #19 `true_coinage` types with no evidence-linked reading in
this workspace -- `omanko`, `laple` -- are left marked for #63's glossing
committee.

## Reconciliation with the frozen layers

| quantity | earlier | here | owner | match |
|---|---|---|---|---|
| pages rendered | 225 | 225 | #7 | true |
| clauses | 5614 | 5614 | #33 | true |
| word tokens inside clauses (#33 n_words_clean) | 76897 | 76897 | #33/#19 | true |
| token buckets close over the clause stream | 84306 | 84306 | #62 | true |
| clauses whose two normalization paths agree | 0 | 0 | #7/#17 | true |
| clauses whose token spans tile the page body | 0 | 0 | #7/#17 | true |
| #17 token stream | 90220 | 90220 | #17 | true |
| tokens in no #33 clause (the remainder) | 5914 | 5914 | #17/#33 | true |
| #14 marker events #33 counts inside clauses | 84447 | 84306 | #33/#14 | disclosed |
| operator tokens in the stream | 11000 | 11000 | #17 | true |
| `/` compound-link role (glued) | 952 | 952 | #14 | true |
| `/` sentence-final role | 540 | 540 | #14 | true |

## Anti-pattern audit (L9.8)

* **Hallucinated coherence**: no reading asserts a plot.  Every
  substitution is a single surface -> single plain reading with its
  KWIC context published in gloss_lexicon.tsv; unread surfaces are
  marked, never smoothed over.
* **Circularity**: no LLM translates or judges anything here; the
  readings come from the text's own counted evidence.  #63's
  committee and #75's jury are where models enter.
* **Over-correction**: #19's `ocr_error` class is kept as printed
  (15 types), compounds are kept intact, and coinages are never
  normalized into English spelling.
* **Single-method certainty**: every reading carries a confidence
  and a source; line 2 is measured (#13/#50), not asserted; the
  identity share above is stated so the reader knows how much of
  line 3 is the text itself.
* **Ignoring the residue**: tokens in no #33 clause (5,914) and the markup screen's 1,657 excluded tokens are counted in the reconciliation, and every unresolved surface is itemized.

## Determinism

No randomness and no hash-order dependence (all ties broken by a
fixed key).  A rebuild reproduces every artifact byte-identically
except this header's `Generated` stamp, which no statistic reads.

