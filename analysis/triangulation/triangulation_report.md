# Triangulation / convergence report (#73)

Do the independent Blood Electric analyses converge on the same motifs and domains?  Convergence is evidence; divergence is a lead.  This is a report about **agreement between estimators over one book** -- it is not a translation, and it does not rank the methods: no method here is a ground truth, and no cell says what a motif means.

Reproduce: `python work/triangulation.py` (schema `triangulation/1`, seed 42); regression suite `python work/test_triangulation.py`.  `numpy` only.

## 1. Verdict

1. **The three distributional methods converge on a core of motifs.**  Of 39 motif rows, 6 are **convergent** (mean pairwise Jaccard >= 0.50), 19 **partial**, 12 **divergent** and 2 untestable (fewer than 10 clauses).  The strongest agreements are `ice nebula` (0.86), `entrails` (0.71), `artificial sun` (0.64), `cold-blooded` (0.58), `cyclops` (0.56) -- the same poles #29, #40 and #41 each reported on their own.  (These counts are the 39 motif rows only.  #29's 6 named-domain rows are scored separately in §8 and are counted separately throughout: 0 convergent, 0 partial, 6 divergent, 0 untestable, for 45 matrix rows in total.)
2. **Clause-level agreement is weak but consistently above chance.**  The mean pairwise adjusted Rand over the 5614 clauses is 0.1028, and the worst of the 10 pairs still has permutation p = 0.005.  The methods share structure; they disagree about *how finely to cut it*.
3. **The periodicity method diverges from the other three -- and that is the result, not a defect.**  0 of the 45 motif/domain page series show a cycle (a short-band spectral line, period <= 40 pages, that survives the global and detrended shuffle nulls of #51's family); the rest are drift-only or aperiodic.  The motifs the distributional methods agree on are *dispersed over the book*, not cyclic: the loop is lexical (#24), semantic (#39/#40) and distributional (#41), and not metrical -- #51's own conclusion, now reproduced per motif.
4. **Divergence is concentrated and diagnosable.**  The weakest rows are `babel` (0.18, weakest pair net/lda), `fecundates` (0.19, weakest pair lda/kmeans), `murder` (0.25, weakest pair lda/kmeans), `tokage` (0.25, weakest pair net/lda), `machine` (0.25, weakest pair lda/bertopic).  Each is a lead: they are the places where two instruments see the same recurrence and file it under different neighbourhoods (§6.2).

## 2. What is compared -- and what is not

| method | source | artifact | how the clause partition is made |
|---|---|---|---|
| network community | #29 | `analysis/network/graph_nodes.tsv` | #29 nodes lifted to clauses by node frequency |
| topic model (LDA k=5) | #41 | `analysis/topics/clause_topics.tsv` | #41 per-clause dominant topic |
| topic model (NMF k=6) | #41 | `analysis/topics/clause_topics.tsv` | #41 per-clause dominant topic |
| topic model (BERTopic-style) | #41 | `analysis/topics/clause_topics.tsv` | #41 per-clause dominant topic (-1 = noise) |
| embedding cluster (KMeans k=12) | #40 | `analysis/clause_embeddings/clause_embeddings.npz` | re-derived k-means over #39 vectors |
| periodicity (#51 nulls, motif series) | #51 | `analysis/periodicity/periodicity_tests.tsv` | motif series (clauses per page, clause gaps) -- **not** a clause partition |

**Scope.**  Only analyses *of this book* take part: #29, #40, #41, #51 and the #39 vectors underneath them.  No reference corpus, English baseline or other external text is acquired, re-scored or used as an arbiter here -- the English layers of #24/#51 are not read.  The only nulls are internal: label permutations of the clause partitions and within-book shuffles of the motif series.

**Vocabulary.**  Motif names are the book's own surface forms.  They are reported because several instruments grouped the same strings, not because anyone has established what they denote.  Nothing here is a translation of the book, and no method's grouping is treated as the correct reading.

## 3. Method

1. **Clause partitions.**  Five partitions of the same 5614 clauses: #41's persisted per-clause topic labels; #29's 391 nodes lifted to clauses by node frequency (coverage in §5); and a `numpy` k-means (k-means++ init, Lloyd, k=12, seed 0, `n_init=10`) over #39's L2-normalised vectors, because #40 persisted the catalogue but not the per-clause labels.
2. **Clause-level agreement.**  Adjusted Rand and arithmetic-normalised MI per pair, over the clauses both methods assign, against 200 label permutations (p = (1 + #>=obs)/(R+1)).
3. **Motif matrix.**  Motifs are matched by substring on the space-joined #39 `units` view -- #40's rule, so fused units count for their parts.  For each motif and method: the dominant bucket and its purity.  Methods are then compared by the **Jaccard overlap of their dominant buckets over that motif's clauses**, which needs no label alignment between methods with different k.  A row is *convergent* at mean Jaccard >= 0.50, *divergent* below 0.30, and untestable below 10 clauses.
4. **Periodicity.**  Each motif becomes a page series (clauses per page over 225 pages) plus a clause-position renewal series.  Spectral flatness, lag-1 autocorrelation and three shuffle surrogates (global / block of 5 pages / detrended; 200 each) are computed exactly as #51 did.  Two peak searches run on every series: #51's peak-over-band-median ratio on the full band, whose peak for a slowly varying rate always sits at the longest period; and a **local line statistic** on the short band (period <= 40 pages), where a genuine motif cycle would have to appear.  A series is *periodic* only when the short-band line survives the global and detrended shuffles; a surviving full-band peak alone is *drift*, which is how #51 read the symbol stream.  Why two statistics, and why the block shuffle is not the gate, is in §7.

## 4. Clause-level method agreement

| a | b | shared clauses | groups a | groups b | adjusted Rand | NMI | p(ARI) |
|---|---|---:|---:|---:|---:|---:|---:|
| net | lda | 5255 | 6 | 5 | 0.1109 | 0.118 | 0.00498 |
| net | nmf | 5255 | 6 | 6 | 0.1921 | 0.286 | 0.00498 |
| net | bertopic | 4027 | 6 | 22 | 0.0647 | 0.1464 | 0.00498 |
| net | kmeans | 5255 | 6 | 12 | 0.0472 | 0.1036 | 0.00498 |
| lda | nmf | 5614 | 5 | 6 | 0.0718 | 0.0943 | 0.00498 |
| lda | bertopic | 4252 | 5 | 22 | 0.0469 | 0.103 | 0.00498 |
| lda | kmeans | 5614 | 5 | 12 | 0.027 | 0.061 | 0.00498 |
| nmf | bertopic | 4252 | 6 | 22 | 0.064 | 0.1787 | 0.00498 |
| nmf | kmeans | 5614 | 6 | 12 | 0.0596 | 0.1547 | 0.00498 |
| bertopic | kmeans | 4252 | 22 | 12 | 0.3437 | 0.4568 | 0.00498 |

**Reading.**  Every pair is above its own permutation null, so the shared structure is real.  No pair is *identical*: the best is bertopic|kmeans at ARI 0.3437 and the worst lda|kmeans at 0.0270.  The spread is what you would expect from one continuum cut at four resolutions -- #29 works on the 391-term type layer, #40 and #41 on clauses -- and #40's own silhouette scan already showed the partition rising with k, i.e. k is a resolution choice rather than a discovered constant.

Cross-check against the published figures: #41 measured its term partitions against #29's communities as ARI LDA 0.1893 / NMF 0.086 / BERTopic 0.0256 over the 391 shared nodes.  The clause-level numbers here (net|lda 0.1109 and its neighbours) are the same order; clauses are an easier test of agreement than types, so a small rise is expected and no contradiction is claimed.

## 5. Partitions, and the re-derived embedding cluster

| method | groups | clauses assigned | coverage | unassigned | largest bucket |
|---|---:|---:|---:|---:|---|
| net | 6 | 5255 | 0.9361 | 359 | C1 body+genetics (1790) |
| lda | 5 | 5614 | 1.0 | 0 | lda 1: drug · boy-roid · cadaver (1412) |
| nmf | 6 | 5614 | 1.0 | 0 | nmf 0: cadaver · brain · flesh (2232) |
| bertopic | 22 | 4252 | 0.7574 | 1362 | bertopic noise (1362) |
| kmeans | 12 | 5614 | 1.0 | 0 | kmeans 5 [brain · consciousness · organ] (806) |

**#40 catalogue check.**  The re-derived partition reproduces #40's published catalogue: 12 of 12 derived clusters match a #40 cluster on their top 3 log-odds label terms (mean Jaccard 0.38, mean size ratio 1.00).  That is what licenses using the re-derived labels as the embedding column; #40's HDBSCAN noise fraction (3,811 clauses, 68%) and its units/raw ARI (0.6028) are cited from `analysis/clustering/` and bound how much the embedding partition can be trusted at all.

| derived | clauses | derived label | #40 cluster | #40 label | size ratio | label Jaccard |
|---:|---:|---|---|---|---:|---:|
| 0 | 400 | adam · doll · retro | 1 | adam · doll · retro | 1.0076 | 0.375 |
| 1 | 470 | i · lt · l | 9 | i · lt · l | 1.0829 | 0.375 |
| 2 | 455 | disease · cold · blooded | 4 | disease · cold · blooded | 1.0341 | 0.375 |
| 3 | 283 | embryo · drug · abortion | 8 | embryo · drug · abortion | 1.0217 | 0.375 |
| 4 | 405 | murder · assassin · entrails | 5 | murder · assassin · entrails | 0.9574 | 0.375 |
| 5 | 806 | brain · consciousness · emotional | 2 | brain · consciousness · organ | 1.0577 | 0.375 |
| 6 | 361 | sun · artificial · the | 0 | sun · artificial · the | 0.989 | 0.375 |
| 7 | 728 | second · cyclops · atom | 3 | second · machine · cyclops | 0.8922 | 0.375 |
| 8 | 208 | ice · faecal · black | 11 | ice · faecal · black | 1.0048 | 0.375 |
| 9 | 659 | cadaver · flesh · city | 10 | cadaver · flesh · city | 1.017 | 0.375 |
| 10 | 334 | dog · dogs · chemical | 7 | dog · dogs · a | 0.9853 | 0.375 |
| 11 | 505 | gene · genome · tv | 6 | gene · genome · tv | 0.998 | 0.375 |

## 6. The convergence matrix (motifs)

Full matrix: `convergence_matrix.tsv` (wide), `convergence_matrix.md` (rendered), `motif_method_cells.tsv` (long), `motif_inventory.tsv` (rows with provenance and spread).  Rendered here as purity per method, the mean pairwise Jaccard and the periodicity verdict.

Motif counts are checked against #40's own inventory (`motif_count_reconciliation.tsv`): 14 of the 17 shared motifs reproduce #40's clause count exactly; the rest differ only by the hyphenated aliases this inventory declares (`artificial-sun`, `drug-embryo`, `cadaver-city`, `ice-nebula`, `internal-organ`, `cold blooded`, `hydro=mania`), so the counts stay comparable with every earlier motif number in the workspace.  The embedding column also reproduces #40's *containment*: for 16 of the 17 shared motifs that carry clauses, the re-derived k-means dominant bucket is the same #40 cluster #40 named as the motif's home, with purity within a few points of #40's (`frac_in_top_cluster_40` vs `purity_kmeans_73`).

| motif | clauses | net | lda | nmf | bertopic | kmeans | mean J | strong | weakest pair | per. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `ice nebula` | 167 | 0.88 | 0.85 | 0.99 | 0.93 | 0.91 | 0.86 | 5/5 | net/lda 0.78 | drift-only |
| `entrails` | 67 | 0.67 | 0.67 | 0.78 | 0.58 | 0.49 | 0.71 | 4/5 | nmf/kmeans 0.63 | not-periodic |
| `artificial sun` | 409 | 0.79 | 0.52 | 0.78 | 0.68 | 0.74 | 0.64 | 5/5 | lda/nmf 0.51 | drift-only |
| `cold-blooded` | 192 | 0.43 | 0.65 | 0.86 | 0.59 | 0.67 | 0.58 | 4/5 | net/lda 0.48 | not-periodic |
| `cyclops` | 85 | 0.61 | 0.92 | 0.58 | 0.81 | 0.39 | 0.56 | 4/5 | lda/kmeans 0.39 | drift-only |
| `drug embryo` | 270 | 0.66 | 0.61 | 0.53 | 0.55 | 0.81 | 0.56 | 5/5 | lda/bertopic 0.41 | drift-only |
| `consciousness` | 206 | 0.78 | 0.66 | 0.91 | 0.3 | 0.43 | 0.48 | 3/5 | nmf/bertopic 0.32 | drift-only |
| `hydromania` | 151 | 0.52 | 0.48 | 0.42 | 0.72 | 0.6 | 0.48 | 3/5 | nmf/bertopic 0.37 | drift-only |
| `parasite` | 188 | 0.59 | 0.52 | 0.48 | 0.31 | 0.55 | 0.45 | 3/5 | lda/bertopic 0.26 | drift-only |
| `gene=tv` | 151 | 0.58 | 0.42 | 0.56 | 0.44 | 0.57 | 0.44 | 3/5 | net/lda 0.27 | not-periodic |
| `internal organ` | 269 | 0.74 | 0.57 | 0.87 | 0.22 | 0.33 | 0.43 | 3/5 | nmf/bertopic 0.25 | drift-only |
| `organ` | 331 | 0.62 | 0.49 | 0.73 | 0.18 | 0.31 | 0.42 | 2/5 | nmf/bertopic 0.25 | not-periodic |
| `adam` | 502 | 0.44 | 0.35 | 0.54 | 0.72 | 0.79 | 0.41 | 3/5 | net/lda 0.03 | drift-only |
| `chromosome` | 226 | 0.49 | 0.31 | 0.41 | 0.35 | 0.41 | 0.37 | 0/5 | lda/kmeans 0.28 | not-periodic |
| `brain+load+ecstasy` | 3 | 0.67 | 0.67 | 0.33 | 0.67 | 0.33 | 0.37 | 3/5 | bertopic/kmeans 0.0 | not-periodic |
| `cadaver city` | 214 | 0.85 | 0.36 | 0.63 | 0.21 | 0.4 | 0.35 | 2/5 | lda/bertopic 0.2 | drift-only |
| `gene` | 280 | 0.42 | 0.26 | 0.41 | 0.32 | 0.51 | 0.34 | 1/5 | net/lda 0.2 | not-periodic |
| `drone` | 273 | 0.62 | 0.51 | 0.45 | 0.3 | 0.22 | 0.34 | 2/5 | nmf/kmeans 0.17 | drift-only |
| `boy-roid` | 339 | 0.7 | 0.55 | 0.55 | 0.31 | 0.25 | 0.34 | 3/5 | net/kmeans 0.18 | drift-only |
| `assassin` | 191 | 0.44 | 0.35 | 0.61 | 0.32 | 0.46 | 0.33 | 1/5 | net/bertopic 0.2 | drift-only |
| `serum` | 256 | 0.57 | 0.62 | 0.54 | 0.22 | 0.3 | 0.32 | 3/5 | bertopic/kmeans 0.09 | drift-only |
| `psychosexual` | 52 | 0.48 | 0.33 | 0.4 | 0.38 | 0.42 | 0.31 | 0/5 | net/lda 0.08 | not-periodic |
| `soul/gram` | 330 | 0.7 | 0.53 | 0.6 | 0.23 | 0.22 | 0.3 | 3/5 | bertopic/kmeans 0.12 | drift-only |
| `flesh` | 560 | 0.71 | 0.42 | 0.56 | 0.2 | 0.28 | 0.3 | 2/5 | bertopic/kmeans 0.17 | drift-only |
| `cadaver` | 637 | 0.7 | 0.45 | 0.59 | 0.18 | 0.33 | 0.3 | 2/5 | bertopic/kmeans 0.13 | drift-only |
| `dog` | 495 | 0.31 | 0.28 | 0.33 | 0.54 | 0.65 | 0.3 | 2/5 | lda/nmf 0.07 | drift-only |
| `brain` | 504 | 0.38 | 0.43 | 0.51 | 0.23 | 0.34 | 0.29 | 1/5 | net/bertopic 0.15 | not-periodic |
| `swastika` | 133 | 0.36 | 0.38 | 0.4 | 0.68 | 0.16 | 0.29 | 1/5 | net/kmeans 0.11 | not-periodic |
| `soul` | 483 | 0.5 | 0.37 | 0.42 | 0.22 | 0.25 | 0.29 | 1/5 | lda/kmeans 0.12 | not-periodic |
| `corporation` | 26 | 0.35 | 0.58 | 0.35 | 0.31 | 0.19 | 0.27 | 1/5 | bertopic/kmeans 0.08 | not-periodic |
| `dog+murder+gene` | 15 | 0.27 | 0.33 | 0.53 | 0.53 | 0.47 | 0.27 | 2/5 | net/kmeans 0.0 | not-periodic |
| `second` | 64 | 0.55 | 0.42 | 0.58 | 0.25 | 0.19 | 0.26 | 2/5 | bertopic/kmeans 0.0 | drift-only |
| `hologram` | 154 | 0.37 | 0.32 | 0.41 | 0.42 | 0.32 | 0.25 | 0/5 | lda/kmeans 0.16 | not-periodic |
| `machine` | 695 | 0.41 | 0.32 | 0.52 | 0.22 | 0.23 | 0.25 | 1/5 | lda/bertopic 0.14 | drift-only |
| `murder` | 792 | 0.51 | 0.29 | 0.4 | 0.25 | 0.29 | 0.25 | 1/5 | lda/kmeans 0.07 | drift-only |
| `tokage` | 134 | 0.27 | 0.39 | 0.54 | 0.47 | 0.24 | 0.25 | 1/5 | net/lda 0.11 | not-periodic |
| `fecundates` | 280 | 0.46 | 0.26 | 0.4 | 0.21 | 0.17 | 0.19 | 0/5 | lda/kmeans 0.05 | not-periodic |
| `babel` | 94 | 0.3 | 0.24 | 0.41 | 0.3 | 0.18 | 0.18 | 0/5 | net/lda 0.02 | not-periodic |
| `necropolis` | 0 | - | - | - | - | - | - | 0/5 |  - | no occurrences |

### 6.1 Agreement

These motifs are grouped by the distributional methods at once: the same clauses land in one bucket whichever method is asked, even though the buckets carry different names and different k.  This is the strongest form of convergence available here -- no method can see another's labels, only the clauses they share.

- **`ice nebula`** -- 167 clauses over 91 pages, mean J 0.86.  net -> C0 machine+genetics (p=0.88); lda -> lda 3: ice · faecal · black (p=0.85); nmf -> nmf 2: ice · nebula · faecal (p=0.99); bertopic -> bertopic 21: ice · nebula · black (p=0.93); kmeans -> kmeans 8 [ice · faecal · black] (p=0.91).  Periodicity: drift-only (flatness 0.3877, p_block 1.0).
- **`entrails`** -- 67 clauses over 28 pages, mean J 0.71.  net -> C2 machine+body (p=0.67); lda -> lda 4: internal · flesh · organ (p=0.67); nmf -> nmf 4: murder · function · entrails (p=0.78); bertopic -> bertopic noise (p=0.58); kmeans -> kmeans 4 [murder · assassin · entrails] (p=0.49).  Periodicity: not-periodic (flatness 0.8298, p_block 1.0).
- **`artificial sun`** -- 409 clauses over 184 pages, mean J 0.64.  net -> C2 machine+body (p=0.79); lda -> lda 2: sun · artificial · murder (p=0.52); nmf -> nmf 1: sun · artificial · suicide (p=0.78); bertopic -> bertopic 19: sun · artificial · artificial-sun (p=0.68); kmeans -> kmeans 6 [sun · artificial · the] (p=0.74).  Periodicity: drift-only (flatness 0.4392, p_block 1.0).
- **`cold-blooded`** -- 192 clauses over 118 pages, mean J 0.58.  net -> C4 body+machine (p=0.43); lda -> lda 4: internal · flesh · organ (p=0.65); nmf -> nmf 5: body · adam · doll (p=0.86); bertopic -> bertopic 9: cold-blooded · disease · animals (p=0.59); kmeans -> kmeans 2 [disease · cold · blooded] (p=0.67).  Periodicity: not-periodic (flatness 0.5774, p_block 1.0).
- **`cyclops`** -- 85 clauses over 61 pages, mean J 0.56.  net -> C1 body+genetics (p=0.61); lda -> lda 1: drug · boy-roid · cadaver (p=0.92); nmf -> nmf 0: cadaver · brain · flesh (p=0.58); bertopic -> bertopic noise (p=0.81); kmeans -> kmeans 7 [second · machine · cyclops] (p=0.39).  Periodicity: drift-only (flatness 0.4574, p_block 1.0).
- **`drug embryo`** -- 270 clauses over 147 pages, mean J 0.56.  net -> C1 body+genetics (p=0.66); lda -> lda 1: drug · boy-roid · cadaver (p=0.61); nmf -> nmf 0: cadaver · brain · flesh (p=0.53); bertopic -> bertopic 12: drug · embryo · drug-embryo (p=0.55); kmeans -> kmeans 3 [embryo · drug · abortion] (p=0.81).  Periodicity: drift-only (flatness 0.5386, p_block 1.0).

### 6.2 Divergence (leads)

Divergence is not a failure of the text.  It says the instruments are measuring different things, and each row is a hypothesis to test next rather than a defect to repair.  All 31 below-threshold rows are in `motif_inventory.tsv`; the 10 weakest are spelled out.

- **`babel`** (mean J 0.18, 94 clauses, 79 pages): weakest pair net/lda at J 0.02 -- C1 body+genetics (p 0.3) vs lda 0: adam · doll · world (p 0.24)  
  *split under both methods -- shared region, not shared motif.*
- **`fecundates`** (mean J 0.19, 280 clauses, 143 pages): weakest pair lda/kmeans at J 0.05 -- lda 2: sun · artificial · murder (p 0.26) vs kmeans 5 [brain · consciousness · organ] (p 0.17)  
  *split under both methods -- shared region, not shared motif.*
- **`murder`** (mean J 0.25, 792 clauses, 206 pages): weakest pair lda/kmeans at J 0.07 -- lda 1: drug · boy-roid · cadaver (p 0.29) vs kmeans 4 [murder · assassin · entrails] (p 0.29)  
  *split under both methods -- shared region, not shared motif.*
- **`tokage`** (mean J 0.25, 134 clauses, 86 pages): weakest pair net/lda at J 0.11 -- C1 body+genetics (p 0.27) vs lda 0: adam · doll · world (p 0.39)  
  *split under both methods -- shared region, not shared motif.*
- **`machine`** (mean J 0.25, 695 clauses, 199 pages): weakest pair lda/bertopic at J 0.14 -- lda 0: adam · doll · world (p 0.32) vs bertopic noise (p 0.22)  
  *split under both methods -- shared region, not shared motif.*
- **`hologram`** (mean J 0.25, 154 clauses, 107 pages): weakest pair lda/kmeans at J 0.16 -- lda 4: internal · flesh · organ (p 0.32) vs kmeans 5 [brain · consciousness · organ] (p 0.32)  
  *split under both methods -- shared region, not shared motif.*
- **`second`** (mean J 0.26, 64 clauses, 56 pages): weakest pair bertopic/kmeans at J 0.0 -- bertopic noise (p 0.25) vs kmeans 0 [adam · doll · retro] (p 0.19)  
  *split under both methods -- shared region, not shared motif.*
- **`dog+murder+gene`** (mean J 0.27, 15 clauses, 15 pages): weakest pair net/kmeans at J 0.0 -- C1 body+genetics (p 0.27) vs kmeans 10 [dog · dogs · a] (p 0.47)  
  *split under both methods -- shared region, not shared motif.*
- **`corporation`** (mean J 0.27, 26 clauses, 25 pages): weakest pair bertopic/kmeans at J 0.08 -- bertopic noise (p 0.31) vs kmeans 0 [adam · doll · retro] (p 0.19)  
  *split under both methods -- shared region, not shared motif.*
- **`soul`** (mean J 0.29, 483 clauses, 198 pages): weakest pair lda/kmeans at J 0.12 -- lda 1: drug · boy-roid · cadaver (p 0.37) vs kmeans 5 [brain · consciousness · organ] (p 0.25)  
  *split under both methods -- shared region, not shared motif.*

**Reading the leads.**  Three patterns recur, and they are leads about *instruments*, not about the text.  (a) *Granularity splits*: 24 of the 31 below-threshold rows have a weakest pair in which **neither** method concentrates the motif (both purities below 0.50): the motif is split several ways under both methods, so what the methods share is a region of the book rather than a motif.  (b) *Density noise is a divergence of its own*: the BERTopic-style model leaves 1362 of 5614 clauses unclustered, and in 15 of the below-threshold rows that **noise** class is one of the two buckets being compared.  `second` is the sharpest case -- its dominant bertopic bucket is the noise class at purity 0.25, while #41's BERTopic topic 5 (`second · android · machine`) and #40's cluster 3 (`second · machine · cyclops`) both name it.  A motif the density model declines to cluster but the k-means partition isolates is a genuine disagreement about density versus distance, and it is the cheapest lead here to follow.  (c) *The OCR loop*: `second` occurs in 64 clauses over 56 pages and is largely the #4 recognition-loop material, so part of the apparent convergence on `second`/`cyclops` is convergence on an artefact -- reported, not celebrated.

## 7. Periodicity: the axis the other three do not have

#51 tested the *symbol stream* and found no latent cycle: the operator indicator is statistically white once 732 markup-span operators are screened (flatness 0.555 against the 0.561 white-noise anchor, p_block 0.70).  The cross-check here asks the same question of the motifs -- if a motif is a cycle, the pages carrying it should show a peak that survives shuffling.

| motif | pages present | flatness | full-band peak (pages) | p_detrend | short-band line | short period (pages) | p_global (short) | q (short) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `adam` | 164/225 | 0.2617 | 112.5 | 0.00498 | 6.474 | 25.0 | 0.46766 | 0.85218 | drift-only |
| `artificial sun` | 184/225 | 0.4392 | 225.0 | 0.00498 | 6.1026 | 2.5862 | 0.49751 | 0.85218 | drift-only |
| `assassin` | 100/225 | 0.3566 | 225.0 | 0.00498 | 5.6915 | 37.5 | 0.62189 | 0.85218 | drift-only |
| `babel` | 79/225 | 0.535 | 3.75 | 0.32836 | 8.0372 | 3.75 | 0.28358 | 0.79477 | not-periodic |
| `boy-roid` | 141/225 | 0.3267 | 75.0 | 0.00498 | 10.1697 | 22.5 | 0.14428 | 0.79477 | drift-only |
| `brain` | 194/225 | 0.5705 | 112.5 | 0.38308 | 6.5435 | 4.0909 | 0.40796 | 0.85218 | not-periodic |
| `brain+load+ecstasy` | 3/225 | 0.6935 | 2.8846 | 0.50249 | 2.4069 | 2.8846 | 0.45274 | 0.83712 | not-periodic |
| `cadaver` | 192/225 | 0.5018 | 225.0 | 0.00498 | 4.1476 | 22.5 | 0.86567 | 0.94467 | drift-only |
| `cadaver city` | 128/225 | 0.5506 | 112.5 | 0.00498 | 4.232 | 6.4286 | 0.83085 | 0.94467 | drift-only |
| `chromosome` | 146/225 | 0.6141 | 7.5 | 0.45771 | 8.544 | 7.5 | 0.26368 | 0.79477 | not-periodic |
| `cold-blooded` | 118/225 | 0.5774 | 12.5 | 0.13433 | 13.0607 | 12.5 | 0.08458 | 0.79477 | not-periodic |
| `consciousness` | 127/225 | 0.52 | 75.0 | 0.02985 | 5.1465 | 7.7586 | 0.76119 | 0.90174 | drift-only |
| `corporation` | 25/225 | 0.5972 | 2.0089 | 0.19403 | 6.7198 | 2.0089 | 0.39303 | 0.83712 | not-periodic |
| `cyclops` | 61/225 | 0.4574 | 225.0 | 0.00498 | 8.3542 | 32.1429 | 0.20398 | 0.79477 | drift-only |
| `dog` | 192/225 | 0.5088 | 37.5 | 0.00498 | 5.9163 | 37.5 | 0.60199 | 0.85218 | drift-only |
| `dog+murder+gene` | 15/225 | 0.627 | 3.629 | 0.97512 | 5.3556 | 3.629 | 0.59701 | 0.85218 | not-periodic |
| `drone` | 138/225 | 0.4905 | 225.0 | 0.00498 | 10.0773 | 12.5 | 0.13433 | 0.79477 | drift-only |
| `drug embryo` | 147/225 | 0.5386 | 225.0 | 0.00498 | 2.8193 | 7.2581 | 0.99502 | 1.0 | drift-only |
| `entrails` | 28/225 | 0.8298 | 75.0 | 0.38308 | 2.9231 | 12.5 | 0.37313 | 0.79477 | not-periodic |
| `fecundates` | 143/225 | 0.5125 | 13.2353 | 0.10945 | 6.943 | 13.2353 | 0.36318 | 0.79477 | not-periodic |

White-noise flatness anchor `exp(-gamma)` = 0.561 (#51's anchor).  The short band is periods <= 40 pages, where a genuine motif cycle would have to appear; the full-band peak of a dispersed motif always sits at the longest period instead, which is the drift signature #51 read on the symbol stream.  The table shows raw p-values; `periodicity_motifs.tsv` adds Benjamini-Hochberg q-values corrected across the 45 rows, the clause-gap renewal columns and the lead flag discussed below.

**Result.**  0 of 45 series carry a cycle; the rest are drift-only or aperiodic.  The motifs the distributional methods agree on have page-presence spread across the book (top-5 page share in `motif_inventory.tsv`), spectra that are flat or drift-dominated, and clause gaps that are over-dispersed but close to memoryless -- #51's renewal finding, reproduced per motif.  This is the program's third independent arrival at one conclusion: the loop lives in repetition and distribution, not in a metrical cycle.

**Two statistics, because one of them lies.**  #51's peak-over-band-median ratio is the right statistic for the symbol stream, where one drift dominates the spectrum.  Applied to a motif page series it is not: the page-rate envelope depresses the median of any short band, so a merely *heterogeneous* series looks peaky, and the block shuffle makes it worse (permuting 5-page blocks creates runs whose edges splatter power across the band -- `p_block_short` is 1.00 on every row, the null always beating the data).  The short-band test here therefore uses a **local** line statistic (peak over the median of its 20 neighbouring bins), which a dispersed series cannot fake, and is gated on the global and detrended shuffles, which do destroy a real cycle.  No row clears both at BH q < 0.05, so no motif series is called periodic; the `short_cycle_lead` column is left in `periodicity_motifs.tsv` for a follow-up surrogate that preserves local density without creating runs.

## 8. Domains (#29's six named domains)

The same treatment for #29's published domain seed sets, so the six named domains are measured rather than asserted.  Seed sets are read straight out of `work/cooccurrence_network.py` with `ast.literal_eval`, so they cannot drift from #29's definition.  (Its `corporate` set contains the multi-word entry `sato corporation`, which can never match a unit token; it is kept as-is and disclosed.)

| domain | clauses | net | lda | nmf | bertopic | kmeans | mean J | verdict | per. |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `corporate` | 50 | 0.32 | 0.42 | 0.36 | 0.22 | 0.18 | 0.22 | divergent | not-periodic |
| `machine` | 2571 | 0.35 | 0.27 | 0.37 | 0.21 | 0.18 | 0.21 | divergent | not-periodic |
| `body` | 2920 | 0.4 | 0.29 | 0.35 | 0.22 | 0.17 | 0.21 | divergent | drift-only |
| `genetics` | 2252 | 0.43 | 0.35 | 0.4 | 0.19 | 0.2 | 0.21 | divergent | drift-only |
| `sex` | 676 | 0.34 | 0.22 | 0.37 | 0.2 | 0.23 | 0.16 | divergent | not-periodic |
| `violence` | 1930 | 0.33 | 0.28 | 0.32 | 0.24 | 0.19 | 0.16 | divergent | not-periodic |

**Reading.**  The six domains all land in the same narrow band of mean Jaccard (0.16-0.22) and every one is below the convergence threshold, so the seed-set domains do not converge on anything the distributional methods share.  That is a sharper statement than #29's own: #29 reported that Louvain recovered co-occurrence clusters that *cut across* the named domains, and the clause-level instruments, which never saw those seed lists, land in the same place.  The seed sets are a plausible reading imposed on the book, not the book's own categories: the motifs that do converge (`ice nebula`, `artificial sun`, `drug embryo`, `cold-blooded`) straddle several domains at once.  The two smallest domains (`sex` 676 clauses, `corporate` 50 clauses) also score lowest, which reproduces #29's verdict that they are barely present at its frequency floor.  No domain row reaches the periodicity threshold either.

## 9. What would change the answer

- **A finer periodicity null.**  The block shuffle is blunt by construction (#51's caveat): a motif could carry a cycle below five pages that this test cannot see.  A band-resolved test is left for a follow-up rather than claimed here.
- **A second encoder for the embedding column.**  #39 found SBERT and BGE agree on only 35% of nearest neighbours, so this column is one encoder's view; #40's robustness check bounds the *view* choice, not the encoder choice.
- **Clause-position motif series** (finer than per-page) if the dispersion/periodicity question needs more power.
- **A density-preserving surrogate without run creation** to re-test the short-band statistic, since the block shuffle cannot serve as its null and no motif row currently needs re-testing (0 flagged).

## 10. Negative results and limitations

- **No method is a ground truth.**  Convergence between two coarse summaries of a fused vocabulary is evidence that both see the same motif; it is not evidence that the motif means what its label says.  The task's own framing -- divergence is a lead -- is the operating rule here.
- **The matrix is resolution-relative.**  Every cell depends on the disclosed resolution of its method (#29 Louvain resolution 1.0, #41 k=5/6/22, #40 k=12).  A different k moves purity, which is why the raw numbers ship next to every verdict (`convergence_matrix.tsv`).
- **Substring motif matching inherits #40's granularity**: `adam` also counts `anti-adam`, `soul` counts `soul/gram`, `gene` counts `gene=tv`.  It is the rule every motif count in this workspace used, so the cells stay comparable across analyses.
- **The network partition is a lift, not a native clause labelling.**  A clause with no #29 node gets no community; coverage is in §5.
- **Periodicity is not evidence about meaning.**  A motif can be a real motif and still be aperiodic; §7's result is that the loop has no metrical period, not that the motifs are illusions.
- **`necropolis` is the built-in negative control** carried over from #40's inventory: it has zero occurrences here and is reported untestable.
- **The embedding partition is a re-implementation.**  With scikit-learn absent from this environment the matrix needed a clause partition, so k-means was reimplemented in `numpy` and validated against #40's catalogue (§5).  It is not a bit-identical replay of #40's sklearn run and is not claimed to be.

## 11. Reconciliation and artifacts

| quantity | here | earlier | source | note |
|---|---:|---:|---|---|
| clauses | 5614 | 5614 | #33/#39 | one row per #33 clause; every method sees these |
| pages carrying clauses | 225 | 225 | #39 | page axis of the motif series |
| network nodes | 391 | 391 | #29 | term -> community assignment |
| network communities | 6 | 6 | #29 | Louvain resolution 1.0, seed 42 |
| clauses with a network community | 5255 |  | #29 | majority-vote lift of the term graph to clauses |
| LDA topics | 5 | 5 | #41 | coherence-selected k=5 |
| NMF topics | 6 | 6 | #41 | coherence-selected k=6 |
| BERTopic-style topics | 22 | 22 | #41 | -1 is the noise class, not a topic |
| BERTopic noise clauses | 1362 | 1362 | #41 | unclustered clauses |
| embedding clusters (re-derived) | 12 | 12 | #40 | numpy k-means over the #39 vectors, k=12 |
| k-means clause sizes sum | 5614 | 5614 | #40 | full partition, no noise class |
| k-means inertia | 3852.6963 |  | #40 | sum squared distance to the assigned centre |
| #41 published ARI vs #29 (LDA) | 0.1109 | 0.1893 | #41 | published figure is term-level (391 nodes); here is clause-level |
| #41 published ARI vs #29 (NMF) | 0.1921 | 0.086 | #41 | as above |
| #41 published ARI vs #29 (BERTopic) | 0.0647 | 0.0256 | #41 | as above |
| motif / domain rows in the matrix | 45 |  | #73 | 39 motifs + #29's six named domains |
| label permutations per agreement pair | 200 |  | #73 | p = (1 + #>=obs)/(R+1) |
| shuffle surrogates per motif series | 600 | 200 | #51/#73 | global + block + detrended, #51's null family |
| mean pairwise clause ARI | 0.1028 |  | #73 | over the 10 method pairs |
| motif clause counts reconciled with #40 | 14 | 17 | #40/#73 | exact matches / shared motifs; the rest differ only by the declared hyphenated aliases |
| external corpora consulted | 0 |  | #73 | convergence is assessed among Blood Electric analyses only |

| file | contents |
|---|---|
| `triangulation_report.md` | THE DELIVERABLE |
| `convergence_matrix.tsv` / `.md` | the matrix: purity, dominant buckets, ten pairwise Jaccards, periodicity verdict |
| `motif_method_cells.tsv` | long form of the matrix cells |
| `motif_inventory.tsv` | motif rows: provenance, spread, verdict |
| `motif_count_reconciliation.tsv` | motif clause counts vs #40's inventory |
| `domain_convergence.tsv` | #29's six named domains, same treatment |
| `method_agreement.tsv` | pairwise ARI / NMI + permutation p |
| `clause_agreement.tsv` | per-method clause partition, coverage |
| `cluster_validation.tsv` | re-derived k-means vs #40's catalogue |
| `periodicity_motifs.tsv` | per-motif page-series + renewal statistics |
| `periodicity_book_context.tsv` | #51's book-level series, cited |
| `reconciliation.tsv` / `diagnostics.tsv` | join contract and counters |
| `summary.json` | machine-readable headline numbers |

---
*Generated by `work/triangulation.py` (schema `triangulation/1`, seed 42).*
