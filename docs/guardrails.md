# Anti-pattern guardrails (#76)

Five failure modes this project can fall into, each paired with the
mechanism in the codebase that is supposed to stop it, and the check in
`work/guardrail_audit.py` that enforces the mechanism keeps working.
Run the audit with:

```bash
python work/guardrail_audit.py        # writes analysis/guardrails/
python work/test_guardrail_audit.py   # regression suite, 27 tests
```

The audit reads *published* artifacts of the modules it audits — it
never re-derives their numbers. A drift between what a module reports
and what this audit sees of it is a build error in the audit, not a
silent disagreement to paper over.

## 1. Hallucinated coherence

**The pattern.** Treating a reading as correct because it is
internally consistent, without checking whether anything independent
of the reading itself agrees with it. A single method dressed in
confident prose can look coherent while being wrong.

**The guard.** [#73 triangulation](../work/triangulation.py) compares
four independent methods (network community, topic model, embedding
cluster, periodicity) and reports a convergence verdict —
`convergent` / `partial` / `divergent` / `untestable` — per motif and
per domain, with disagreement disclosed as a lead, not suppressed.
Currently: 6 convergent, 19 partial, 18 divergent, 2 untestable across
45 rows — most of the book does *not* converge, and that is published,
not hidden.

**The check.** `check_hallucinated_coherence` fails if the published
verdict counts ever stop including divergent/partial rows alongside
convergent ones — i.e. if the report starts looking like it only ever
finds agreement.

## 2. LLM circularity

**The pattern.** The same model that produced a translation also
grades it, so "the model agrees with itself" gets reported as
validation.

**The guard.** [#63's glossing pipeline](../work/glossing_pipeline.py)
scores every gloss with four checks — `operator_fidelity`,
`entity_fidelity`, `residue_discipline`, `frame_preservation` — that
are pure functions over the frozen prompt/response/constraint data, no
model call. [#64's translation committee](../work/translation_committee.py)
re-judges #63's pinned readings the same way: the critic and comparer
recompute from frozen artifacts and cross-check against #63's own
published tables, so neither can report a verdict the model itself
chose.

**The check.** `check_llm_circularity` parses the source of each
verdict-computing function with `ast` (function-scoped, not a
whole-file grep, so an unrelated loader mentioning `opencode` elsewhere
in the same file cannot trip it) and fails if any of them contains a
token that would reach a model backend or call back into generation
(`opencode`, `subprocess`, `requests.`, `resolve_glosses(`, …).

## 3. Over-correction

**The pattern.** Forcing a confident, tidy reading onto data that is
genuinely ambiguous — "correcting" everything because leaving
something unresolved feels like a worse answer than a wrong one.

**The guard.** [#6's lexical corrector](../work/correct_lexicon.py)
only accepts a correction under a deterministic, disclosed rule
(edit distance 1–2, zipf margin over the runner-up); anything else is
left in a `REVIEW` class, unchanged and listed, never silently fixed.
Currently: 15 tokens corrected, 200 held back for review — the
corrector defers far more often than it acts.

**The check.** `check_over_correction` fails the moment the `REVIEW`
class (`analysis/lexical_correction/unchanged_review.tsv`) goes empty,
i.e. the corrector starts "fixing" every out-of-vocabulary token it
sees instead of holding back the ones it isn't sure about.

## 4. Single-method certainty

**The pattern.** Treating one method's output as a finding rather than
one estimator's opinion, especially when that method is the only one
ever run on the question.

**The guard.** Convergence claims in this workspace are required to
come from #73's triangulation, which runs five independent methods
(network community, LDA, NMF, BERTopic-style, KMeans) and reports
agreement only as *pairwise* overlap between them — never as a single
method's confidence score standing in for a finding.

**The check.** `check_single_method_certainty` fails if the published
method count ever drops below three.

## 5. Residue-dropping

**The pattern.** Reporting a clean, fully-explained result by quietly
not counting the part of the data that didn't fit — the awkward
residue gets dropped from the denominator instead of itemized.

**The guard.** [#70's coverage ledger](../work/coverage_ledger.py)
forces closure at four levels (characters, tokens, operator tokens,
unit occurrences): every occurrence must land in a named bucket, and
whatever doesn't is itemized in `residue_ledger.tsv` — one row per
class, with its count, pages, examples and a verdict of `explained` /
`disclosed` / `unexplained`. Currently 11 residue classes are itemized
(1,121 occurrences) and all four levels close at 0 unexplained.

**The check.** `check_residue_dropping` fails if the ledger ever
reports zero itemized residue rows (claiming nothing is awkward,
rather than having checked) or if any level reports nonzero
unexplained residue.

## CI-style wiring

`work/test_guardrail_audit.py` is the regression suite: synthetic
fixtures prove each check's failure path has teeth (an all-convergent
triangulation summary, a verdict function calling out to a model
backend, an empty REVIEW class, two methods instead of three, an
unexplained-residue ledger — each is constructed and asserted to
fail), and an artifact-backed suite runs the real audit against this
workspace's own published artifacts and pins today's result: all five
guardrails pass. It runs the same way every other suite in this
workspace does — `python work/test_guardrail_audit.py` or under
`python -m unittest discover -s work -p 'test_*.py'` — so it is picked
up by the same discovery a CI job would use; there is no separate CI
config in this repository to wire it into.
