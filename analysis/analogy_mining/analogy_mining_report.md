# Analogy and metaphor mining (#45)

## Result

Scored 139,128 four-entity pairings over 34 recurring entities from #30. The inventory is restricted to accepted single-unit entities occurring at least 8 times and present in all three #28 book-only models. Multiword glossary entities are excluded because averaging their component vectors would build the shared word into the result and produce trivial compositional offsets. The first 100 ranked candidates are in `equations.tsv`.

The ranking uses the median within-model percentile of cosine similarity between the two entity-pair offsets (`b - a` and `d - c`). A shared orientation is chosen by majority agreement across word2vec, fastText, and GloVe. The table also reports each model's offset cosine and the exact rank of `d` for `b - a + c` among the model's full vocabulary. Higher ranks and percentiles are descriptive; they are not significance tests.

## Finding

The top-100 offset matches do not produce a stable analogy: 0/100 predict `d` in the full-vocabulary top 10 in all three models, and 2/100 do so in at least two. The best candidate's worst model rank is 60 (best median rank 4). Thus the strongest offset similarities are not enough to establish a viable cross-model equation. The table is a ranked set of leads, with a negative result on stable completion.

## Leading candidates

Completion ranks are against each model's full vocabulary after excluding the three input terms; lower is better.

| Rank | Candidate equation | Offset percentile | word2vec rank | fastText rank | GloVe rank |
|---:|---|---:|---:|---:|---:|
| 1 | `BABEL=TV : sodom :: HIV : SODO` | 0.9994 | 1806 | 18 | 291 |
| 2 | `HIV : NDRO :: retro-ADAM : NDROID` | 0.9990 | 5094 | 123 | 525 |
| 3 | `deathshead : retro-ADAM :: NIHIL : HYPE` | 0.9990 | 1 | 55 | 2885 |
| 4 | `deathshead : retro-ADAM :: dustNirverna : HYPE` | 0.9986 | 22 | 30 | 1913 |
| 5 | `HIV : NDRO :: HYPE : NIHIL` | 0.9984 | 1 | 134 | 4541 |
| 6 | `DNA=channel : NIHIL :: retro-ADAM : XX` | 0.9983 | 3666 | 1784 | 68 |
| 7 | `BABEL=TV : SODO :: HIV : Sato` | 0.9982 | 3048 | 124 | 113 |
| 8 | `NDRO : SM :: NDROID : okama` | 0.9981 | 4065 | 104 | 55 |
| 9 | `HYPE : NDRO :: retro-ADAM : NDROID` | 0.9981 | 4566 | 75 | 300 |
| 10 | `dustNirverna : HYPE :: XX : retro-ADAM` | 0.9979 | 705 | 29 | 2177 |

## Reading the candidates

Each row includes the entity-induction role, occurrence/page counts, and one KWIC context for each term. These contexts attest the terms; they are not direct evidence that the text asserts the relation between them. Read them before interpreting an equation. Entity role and vector proximity do not by themselves establish a metaphor or a translation. The entity inventory contains acronyms and coined labels as well as named characters, so apparent letter-pattern relations can be artifacts of spelling or shared contexts.

All three models use only the book's screened unit stream; no external reference corpus is used. The models share a corpus and are not three independent textual sources. The results therefore form a ranked shortlist for close reading, not an automatic semantic glossary.

Reproduce with `python work/analogy_mining.py`.
