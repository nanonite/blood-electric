# Contrastive semantic axes (#43)

Find the dimensions that separate the #40 clusters and test the four axes the task names -- `organic<->machine`, `self<->other`, `body<->information`, `sacred<->profane` -- with cosine contrasts on antonym seed pairs (L5.5 of the method inventory).

**Headline.** All four named axes are real in the #39 clause-vector space: every one's seed pairs are coherent with a mean-of-poles-minus-mean-of-poles direction far beyond what randomized pole assignments produce (z = +2.73 to +3.19 against 200 random axes each), and every one separates the #40 cluster centroids beyond a random-direction null. The strongest separator is **organic<->machine** (centroid spread 0.1491); the weakest is **self<->other** (spread 0.058) -- the book has a thick first-person register and almost no lexicalised *other* (2 clauses carry `other`, 1 `another`), so that axis is carried by one pronoun-heavy cluster rather than by two opposed vocabularies. The largest *cluster-separating* dimensions overall are not the four named axes but the centroid gaps between the first-person cluster and every thematic cluster (§6).

## 1. Method

* **Vectors** are the #39 primary artifact (`analysis/clause_embeddings/clause_embeddings.npz`): SBERT `all-MiniLM-L6-v2`, mean pooling + L2 normalise, 384-d, `units` view (markup-screened #17 unit surfaces). 5,614 clauses, 225 pages. No encoder runs over the book.
* **Seed words are encoded with the same pinned #39 encoder** (`SentenceEncoder(ce.PRIMARY)`, CPU, `local_files_only`) so that a word contrast and a clause vector live in one space and the projection test is meaningful. This is the only encoder call in the module; #28's static embeddings are a different space and are never mixed in.
* **The partition is #40's, exactly**: KMeans k=12, `n_init=10`, seed 0 on the same matrix; the reproduction is cross-checked against #40's persisted catalogue (#40 catalogue reproduced exactly (sizes match, medoids match)).
* **Axis direction** = mean(+pole seed vectors) - mean(-pole seed vectors), L2-normalised. Four tests per axis:
  1. *Seed-pair coherence* -- cosine of each pair's difference vector to the axis (the task's antonym-pair contrasts).
  2. *Randomized-axis control* -- 200 fixed-seed random pole assignments over the same term pool each build a random axis, and the **named** pairs are scored on it; the z-score asks whether the named pairing is special, not merely nonzero.
  3. *Clause projection contrast* -- mean projection of clauses carrying a +pole term vs a -pole term (word-boundary match on the units text; clauses carrying both are disclosed, not dropped).
  4. *Cluster separation* -- sd of the 12 cluster-centroid projections along the axis vs a 200-direction random null.
* **Cluster-pair dimensions**: all 66 centroid-difference directions, each interpreted by the seed words it loads on most -- the separating dimensions whether or not they are one of the four named axes.

## 2. Reconciliation

| quantity | here | earlier | note |
|---|---:|---|---|
| clauses | 5,614 | 5,614 (#39) | every #33/#39 clause |
| pages | 225 | 225 (#39) | pages carrying clauses |
| screened units | 74,077 | 74,077 (#27) | markup-screened #17 units behind the vectors |
| kmeans | k=12, n_init=10, seed 0 | k=12, n_init=10, seed 0 (#40) | #40 catalogue reproduced exactly (sizes match, medoids match) |
| empty-text clauses | 36 | 36 (#39) | kept in the partition; all in cluster 9 (36) |

## 3. The four named axes

`mean_pair_cos` is the seed-pair coherence; `control_z` scores it against 200 randomized axes built from the same term pool; `contrast`/`d` are the +pole vs -pole clause projection contrast (effect size against the whole-book projection sd, `contrast_z` its Welch z); `cluster_spread` is the centroid spread with its random-direction z.

| axis | pairs | mean cos | control z | contrast | d | spread | spread z | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `organic<->machine` | 9 | 0.4667 | +2.91 | +0.1159 | +0.883 | 0.1491 | +17.43 | supported |
| `self<->other` | 6 | 0.4497 | +2.8 | +0.074 | +1.209 | 0.058 | +3.97 | supported |
| `body<->information` | 8 | 0.5185 | +2.73 | +0.1278 | +1.196 | 0.1035 | +9.94 | supported |
| `sacred<->profane` | 6 | 0.4438 | +3.19 | +0.139 | +1.596 | 0.081 | +6.26 | supported |

* **`organic<->machine`** -- 2,192 +pole clauses vs 1,443 -pole clauses (704 carry both); separates cluster 10 (cadaver · flesh · city) at the + end from cluster 9 (i · lt · l) at the - end.
* **`self<->other`** -- 1,492 +pole clauses vs 16 -pole clauses (10 carry both); separates cluster 1 (adam · doll · retro) at the + end from cluster 0 (sun · artificial · the) at the - end.
* **`body<->information`** -- 1,798 +pole clauses vs 655 -pole clauses (262 carry both); separates cluster 10 (cadaver · flesh · city) at the + end from cluster 9 (i · lt · l) at the - end.
* **`sacred<->profane`** -- 723 +pole clauses vs 1,295 -pole clauses (220 carry both); separates cluster 1 (adam · doll · retro) at the + end from cluster 5 (murder · assassin · entrails) at the - end.

## 4. Seed-pair contrasts

Every pair's difference vector against its axis, with the corpus clause counts of both terms. The pairs are the task's test: an axis survives only if its own pairs agree on the direction.

| axis | + | - | cos to axis | + clauses | - clauses |
|---|---|---:|---:|---:|---:|
| `organic<->machine` | flesh | machine | +0.5452 | 404 | 564 |
| `organic<->machine` | blood | digital | +0.6041 | 298 | 196 |
| `organic<->machine` | body | circuit | +0.404 | 716 | 219 |
| `organic<->machine` | gene | drone | +0.416 | 228 | 268 |
| `organic<->machine` | chromosome | android | +0.4877 | 220 | 73 |
| `organic<->machine` | embryo | cyborg | +0.4529 | 332 | 29 |
| `organic<->machine` | serum | replicant | +0.4294 | 239 | 100 |
| `organic<->machine` | organ | code | +0.4997 | 288 | 148 |
| `organic<->machine` | cell | data | +0.3617 | 205 | 89 |
| `self<->other` | self | other | +0.4687 | 380 | 2 |
| `self<->other` | i | another | +0.5362 | 389 | 1 |
| `self<->other` | my | they | +0.3204 | 54 | 3 |
| `self<->other` | we | you | +0.3995 | 24 | 4 |
| `self<->other` | our | alien | +0.5407 | 319 | 3 |
| `self<->other` | adam | mirror | +0.4327 | 502 | 3 |
| `body<->information` | body | data | +0.6195 | 716 | 89 |
| `body<->information` | flesh | code | +0.6913 | 404 | 148 |
| `body<->information` | organ | record | +0.5671 | 288 | 43 |
| `body<->information` | skin | signal | +0.5259 | 65 | 3 |
| `body<->information` | meat | noise | +0.4694 | 45 | 92 |
| `body<->information` | bone | memory | +0.4459 | 19 | 286 |
| `body<->information` | brain | language | +0.3456 | 501 | 23 |
| `body<->information` | lobe | information | +0.4829 | 92 | 7 |
| `sacred<->profane` | soul | murder | +0.5472 | 472 | 606 |
| `sacred<->profane` | angel | cadaver | +0.4908 | 214 | 625 |
| `sacred<->profane` | god | junk | +0.4418 | 33 | 75 |
| `sacred<->profane` | heaven | crime | +0.5474 | 22 | 92 |
| `sacred<->profane` | 666 | sin | +0.332 | 6 | 13 |
| `sacred<->profane` | cherry | hell | +0.3034 | 13 | 9 |

## 5. Cluster separation

Cluster-centroid projections on each axis (the #40 partition, reproduced). Sizes and labels are #40's.

| cluster | size | label | `organic<->machine` | `self<->other` | `body<->information` | `sacred<->profane` |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 365 | sun · artificial · the | 0.1535 | -0.1178 | 0.1922 | -0.0042 |
| 1 | 397 | adam · doll · retro | 0.2331 | 0.1007 | 0.2203 | 0.089 |
| 2 | 762 | brain · consciousness · organ | 0.2256 | -0.0331 | 0.243 | 0.0094 |
| 3 | 816 | second · machine · cyclops | -0.0366 | -0.0753 | 0.0581 | 0.012 |
| 4 | 440 | disease · cold · blooded | 0.3353 | -0.0673 | 0.3362 | -0.0574 |
| 5 | 423 | murder · assassin · entrails | 0.1259 | -0.0614 | 0.1546 | -0.2471 |
| 6 | 506 | gene · genome · tv | 0.3654 | -0.0492 | 0.1468 | -0.0082 |
| 7 | 339 | dog · dogs · a | 0.2724 | -0.0264 | 0.2609 | 0.0212 |
| 8 | 277 | embryo · drug · abortion | 0.3737 | -0.0396 | 0.2429 | -0.0598 |
| 9 | 434 | i · lt · l | -0.0734 | 0.0369 | 0.0088 | -0.0182 |
| 10 | 648 | cadaver · flesh · city | 0.4069 | -0.0053 | 0.3893 | -0.0901 |
| 11 | 207 | ice · faecal · black | 0.1372 | -0.109 | 0.1308 | 0.0417 |

### Pole exemplars

The clauses furthest along each axis -- the strongest supporting contrast for the pole readings (`extreme_clauses.tsv`).

**`organic<->machine`** -- + end:

* p95 `suck=blood chromosome-serum`
* p100 `The suck=blood chromosome`
* p35 `receptors tune the atrocity soul/gram of boy-roid to bio=less internal suck=blood`
* p130 `The chromosome=port clotted with faecal blood`
* p156 `the streaming-symbiotic vital=serums of her masses of flesh are broken down and stored in the human body genome`

- end:

* p16 `software`
* p43 `I walk to DIGITAL`
* p189 `the digital nightmare invades the quantum spiral of a cyber sun`
* p86 `the installation-machine`
* p116 `digital vampires of the artificial sun speed control`

**`self<->other`** -- + end:

* p88 `ADAM`
* p152 `ADAM`
* p26 `I`
* p172 `we excoriate the soul/gram mode of retro-ADAM that decayed to HYPE serum`
* p17 `ADAM=erases`

- end:

* p61 `ALIEN-COM`
* p72 `End clone fecundates interference`
* p131 `it is inoculated to nightmare HDD of acidHUMAN and the softwarable parasite of the genome=linkage where it clashes to se`
* p125 `acid=primal cyclops atom`
* p206 `emotional replicant committed suicide with the ice nebula faecal black mimic mode of the parasite visibility of the arti`

**`body<->information`** -- + end:

* p168 `body joint of she-meat which the drug embryo cut`
* p97 `the meat of brain boy-roid that restrains the cadaver-feti feared that body fluid zero`
* p215 `the body of an assassin burns up and cold-blooded disease animals are respired metal-congenital`
* p32 `the mass of flesh-module of the cold-blooded disease of self that is close to the eyeball medium of the Cadaver City sic`
* p154 `the internal organ that gradually rotted`

- end:

* p15 `Data sending`
* p86 `A DIGITAL channel`
* p189 `the digital frequency of BABEL-TV that clashes and splits it`
* p148 `8 seconds of entropy is emulated`
* p132 `l record the`

**`sacred<->profane`** -- + end:

* p79 `SUPER CHERRY 666`
* p11 `Love Doll Super Cherry 17`
* p50 `soulmachine of an angel`
* p77 `The planet of the angel mechanism of the ADAM Doll is EVOL`
* p120 `body-OMOTYA of the Super Cherry`

- end:

* p212 `It is murder`
* p155 `murder`
* p44 `murder`
* p98 `Reproduction crime`
* p92 `crime of the masses of flesh`

## 6. Cluster-pair dimensions

All 66 centroid-difference directions, each interpreted by the seed words it loads on most and scored against the four named axes (`cluster_pair_axes.tsv`, sorted by separation). The top twelve:

| pair | separation | clusters | top seed loadings | best named alignment |
|---|---:|---|---|---|
| 8-9 | 1.2075 | embryo · drug · abortion vs i · lt · l | i(-0.45), you(-0.42), my(-0.41), we(-0.40), record(-0.38) | `organic<->machine` +0.37 |
| 9-11 | 1.1793 | i · lt · l vs ice · faecal · black | i(+0.44), we(+0.42), our(+0.42), my(+0.39), language(+0.37) | `organic<->machine` -0.179 |
| 0-9 | 1.1727 | sun · artificial · the vs i · lt · l | our(-0.41), we(-0.41), i(-0.39), record(-0.39), noise(-0.38) | `organic<->machine` +0.194 |
| 4-9 | 1.1636 | disease · cold · blooded vs i · lt · l | i(-0.43), my(-0.39), we(-0.39), record(-0.38), our(-0.38) | `organic<->machine` +0.351 |
| 7-9 | 1.1309 | dog · dogs · a vs i · lt · l | i(-0.39), we(-0.37), our(-0.36), they(-0.36), my(-0.36) | `organic<->machine` +0.306 |
| 6-9 | 1.1237 | gene · genome · tv vs i · lt · l | i(-0.40), my(-0.39), our(-0.37), angel(-0.37), we(-0.37) | `organic<->machine` +0.391 |
| 1-9 | 1.1094 | adam · doll · retro vs i · lt · l | i(-0.42), my(-0.40), you(-0.39), we(-0.36), language(-0.35) | `organic<->machine` +0.276 |
| 9-10 | 1.066 | i · lt · l vs cadaver · flesh · city | i(+0.42), my(+0.39), our(+0.38), you(+0.38), we(+0.37) | `organic<->machine` -0.451 |
| 7-11 | 1.0554 | dog · dogs · a vs ice · faecal · black | brain(+0.22), soul(+0.14), body(+0.13), flesh(+0.12), organ(+0.12) | `organic<->machine` +0.128 |
| 2-9 | 1.0424 | brain · consciousness · organ vs i · lt · l | i(-0.41), my(-0.38), our(-0.37), we(-0.37), they(-0.35) | `organic<->machine` +0.287 |
| 8-11 | 1.0132 | embryo · drug · abortion vs ice · faecal · black | embryo(+0.39), gene(+0.18), brain(+0.16), chromosome(+0.12), machine(-0.11) | `organic<->machine` +0.233 |
| 5-9 | 1.0057 | murder · assassin · entrails vs i · lt · l | i(-0.41), we(-0.39), our(-0.38), my(-0.38), you(-0.36) | `sacred<->profane` -0.228 |

The pattern is consistent: the widest gaps in the space all oppose **cluster 9** (label `i · lt · l` -- the first-person `i`/`we`/`our` clauses, which also absorb the 36 markup-only empty-text clauses), which appears in 10 of the top 12 pairs, and their top loadings are the self/other pronouns. Read with the `self<->other` row in §3, that says the book's strongest semantic dimension is **first person vs everything else** -- the named self<->other axis is real but one-sided, because there is no second vocabulary to oppose to the first.

## 7. Controls and robustness

| axis | control | observed | null mean | null sd | z |
|---|---|---:|---:|---:|---:|
| `organic<->machine` | randomized_axis (named pairs vs random pole assignments) | 0.4667 | 0.0168 | 0.1545 | +2.91 |
| `organic<->machine` | random_direction (cluster-centroid spread) | 0.1491 | 0.03 | 0.0068 | +17.43 |
| `self<->other` | randomized_axis (named pairs vs random pole assignments) | 0.4497 | 0.0163 | 0.1547 | +2.8 |
| `self<->other` | random_direction (cluster-centroid spread) | 0.058 | 0.03 | 0.0071 | +3.97 |
| `body<->information` | randomized_axis (named pairs vs random pole assignments) | 0.5185 | -0.0164 | 0.196 | +2.73 |
| `body<->information` | random_direction (cluster-centroid spread) | 0.1035 | 0.0308 | 0.0073 | +9.94 |
| `sacred<->profane` | randomized_axis (named pairs vs random pole assignments) | 0.4438 | -0.019 | 0.1452 | +3.19 |
| `sacred<->profane` | random_direction (cluster-centroid spread) | 0.081 | 0.0311 | 0.008 | +6.26 |

* Both nulls are fixed-seed (seed 0, 200 draws each) and the whole build is deterministic: two runs produce byte-identical tables.
* The encoder is the #39 primary (SBERT); the seed words are encoded with it in the same call space as the clauses, so the contrast is measured, not assumed.
* Clause matching is word-boundary on the `units` text (#40's motif checks use substring; for single seed words the stricter convention is the honest one and is applied uniformly).

## 8. Negative results and limitations

* **No named axis failed its control** -- but the margins are modest (z between +2.73 and +3.19), so the axes are *directions*, not partitions: on #40's own continuum (silhouette 0.033 at k=12) they separate centroids, not individual clauses.
* **Thin poles**: `self<->other` (- pole 16 clauses). The axis direction is estimated from seeds, so the clause contrast rests on few clauses and is reported for completeness, not strength.
* **Seed words are the analyst's choice.** The pools are drawn from the book's own lexicon (every term's clause count is published), but a different seed set moves the exact direction; the randomized-axis control tests the pairing, not the choice of pool.
* **SBERT is an English model on a fractured Meta-English stream.** The axes are where the book's English lexis still carries ordinary semantics; whatever the operator channel (#11/#12) carries is invisible to this space by design, as it was to #40.
* The 36 empty-text clauses (#39) sit in one cluster (#40's own finding) and inherit whatever that cluster's projections are; they carry no pole evidence themselves.

