# Periodicity and spectral analysis (#51)

Map the book's symbol stream to numeric series - a **delimiter indicator** (1 at an operator token, 0 else) and the **inter-delimiter distances** - then run autocorrelation and FFT for latent periodicity. This is the measurable form of the jacket-copy claim that *Blood Electric* is "a devastating loop of language" (`docs/meaning-extraction-methods.md` L6.5). Sources: #17 `tokens.tsv` (90,220 tokens; 11,000 operators), the lossless #17 character reconstruction, #33 clause lengths, and the eight pinned #24 English references. Reproduce with `python work/periodicity.py`; regression suite `python work/test_periodicity.py`. Schema `periodicity/1`.

## 1. Method in one paragraph

Each series is demeaned; **autocorrelation** `r(k)` is computed by FFT for all lags 1..L, and the **periodogram** is Hann-windowed. Periodicity is quantified by the peak power in a period band relative to the band median (`peak_rel`), by the **spectral flatness** (geometric/arithmetic mean of the power; the white-noise expectation is `exp(-gamma) = 0.561`), and by the interval statistics of the point process. Every claim is tested against three fixed-seed surrogates (200 shuffles each): a **global shuffle** (destroys all serial structure), a **block shuffle** (preserves local density up to the block scale), and a **detrended global shuffle** (removes slow drift first). The p-value is `(1 + #{null >= observed}) / (R+1)`. The distinction that matters: a peak that survives the global shuffle is *some* structure; a peak that survives the **block** shuffle is periodicity at a scale below the block; a peak that survives **detrending** is a cycle, not page drift.

## 2. Verdict

**NO LATENT FIXED PERIODICITY. The operator stream is close to white noise; its one strong spectral peak is a slow OCR/HTML density drift, and the inter-delimiter distances are close to geometric (a memoryless point process, slightly burstier). The "devastating loop of language" is not a metrical cycle in the symbol stream.** The delimiter indicator `tok_op` (N=90,220, rate 0.122) has spectral flatness 0.532 against the white-noise anchor 0.561. Its strongest periodogram peak sits at the *longest* period, 4511 tokens - the signature of a slowly varying rate, not a cycle.

That peak is an **OCR/HTML artifact**. 732 operator tokens lie inside #27 markup spans (`<a href=...>`, `<th colspan=...>`, `</a>`, `</p>`); 486 of them are on page 92 alone. Removing them (`tok_op_screened`) drops the flatness to 0.555 - the white-noise value - and the top peak to period 10 with **no significance** (p_global=0.333, p_block=0.701). The book's own operator channel is statistically white.

The inter-operator distances are close to **geometric** but over-dispersed: mean 8.18, CV 1.056 (a memoryless process has CV 0.937), observed variance / geometric variance 1.271, lag-1 interval correlation 0.0459. So the operators are ~27% burstier than a memoryless renewal process - the operator runs (`::<<` etc.) plus the page drift - but there is no periodic component. (The screened series' CV rises only because removing page 92's markup cluster leaves a single 1068-token hole; see §5.)
Dropping the three markup-heavy pages entirely (no holes) leaves the near-geometric, mildly over-dispersed property intact: CV 1.028, geometric variance ratio 1.196.

English punctuation is, if anything, *more regular* than the book's operators - CV 0.823 vs 1.056 and flatness 0.561 - so the operator channel is not unusually periodic.

The clause-length series (#33) also shows no periodicity (p_block=0.711, p_detrend=0.940).

## 3. Series and headline statistics

| series | corpus | N | rate / mean | flatness | peak_rel | peak period | interval CV | geo var ratio | interval lag1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tok_op` | book | 90,220 | 0.1219 | 0.532 | 77.0 | 4511 | 1.056 | 1.271 | 0.0459 |
| `tok_op_screened` | book | 90,220 | 0.1138 | 0.555 | 15.3 | 10 | 1.560 | 2.748 | -0.0036 |
| `tok_op_nomarkuppages` | book | 87,396 | 0.1171 | 0.567 | 14.0 | 7 | 1.028 | 1.196 | 0.0226 |
| `tok_punct` | book | 90,220 | 0.1412 | 0.518 | 135.1 | 4511 | 1.025 | 1.223 | 0.0676 |
| `char_op` | book | 480,267 | 0.0360 | 0.540 | 44.6 | 13722 | 1.578 | 2.584 | -0.1634 |
| `char_punct` | book | 480,267 | 0.0396 | 0.529 | 73.6 | 11435 | 1.490 | 2.313 | -0.1369 |
| `clause_len` | book | 5,614 | 13.8021 | 0.546 | 14.3 | 936 | - | - | - |
| `en_tok_punct` | english_pooled | 90,220 | 0.1726 | 0.561 | 19.9 | 2506 | 0.823 | 0.818 | 0.1757 |
| `en_char_punct` | english_pooled | 480,267 | 0.0364 | 0.560 | 28.4 | 11714 | 0.950 | 0.936 | 0.1886 |

_Flatness anchor for white noise: 0.561._

## 4. Null tests: is the spectral peak a cycle or drift?

| series | peak_rel | p(global shuffle) | p(block shuffle) | p(detrended shuffle) | harmonic ratio |
|---|---:|---:|---:|---:|---:|
| `tok_op` | 77.0 | 0.0050 | 0.0348 | 0.0050 | 23.857 |
| `tok_op_screened` | 15.3 | 0.3333 | 0.7015 | 0.4627 | 2.599 |
| `tok_op_nomarkuppages` | 14.0 | 0.6716 | 0.7214 | 0.6667 | 0.000 |
| `tok_punct` | 135.1 | 0.0050 | 0.0100 | 0.0050 | 46.244 |
| `char_op` | 44.6 | 0.0050 | 0.2338 | 0.0050 | 9.979 |
| `char_punct` | 73.6 | 0.0050 | 0.1692 | 0.0050 | 14.324 |
| `clause_len` | 14.3 | 0.0249 | 0.7114 | 0.9403 | 5.979 |
| `en_tok_punct` | 19.9 | 0.0249 | 0.6716 | 0.7960 | 7.836 |
| `en_char_punct` | 28.4 | 0.0100 | 0.5224 | 0.0547 | 4.717 |

Reading: `p(global) < 0.05` means serial structure exists; `p(block) > 0.05` means it does **not** survive reordering of local blocks, i.e. it is clustering/drift rather than a cycle; `p(detrended) > 0.05` means there is no periodicity once the slow trend is removed. The `harmonic ratio` is the power at `period/2` relative to the band median - a true pulse train would show a harmonic peak.

The **autocorrelation** tells the same story. For `tok_op`, `r(1)` = 0.018 exceeds the global-shuffle 95th percentile (0.006) but lies inside the block-shuffle envelope (p95 0.018) - short-range clustering, not a cycle. The same holds for the weak `r(11)` bump (0.041 vs block p95 0.041). At the page scale there is nothing: `r(401)` = 0.002 and `r(200)` = 0.005, inside the noise band, so the book has no page-level period. The full curve is in `autocorrelation.tsv` (with the shuffle envelopes) and `periodicity_autocorr.png`.

## 5. Inter-delimiter distances (renewal analysis)

| series | intervals | mean | variance | CV | geo var ratio | median | p90 | max | lag-1 corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gap_tok_op` | 10,999 | 8.183 | 74.74 | 1.056 | 1.271 | 6 | 18 | 451 | 0.0459 |
| `gap_char_op` | 17,268 | 27.730 | 1915.17 | 1.578 | 2.584 | 7 | 82 | 2123 | -0.1634 |
| `en_gap_punct` | 15,569 | 5.794 | 22.72 | 0.823 | 0.818 | 5 | 12 | 46 | 0.1757 |

`geo var ratio = observed variance / (mean*(mean-1))`, the variance of a geometric gap on {1,2,...} with the same mean. A ratio near 1 is the memoryless null; `< 1` is more regular than Poisson, `> 1` is burstier. (For a memoryless process CV = `sqrt((mean-1)/mean)`, slightly below 1.)

## 6. Top spectral peaks

### `tok_op`

| rank | period | freq | power | rel_power |
|---|---:|---:|---:|---:|
| 1 | 4511.0 | 0.000222 | 189480.95 | 77.0 |
| 2 | 2819.4 | 0.000355 | 121755.60 | 49.5 |
| 3 | 2148.1 | 0.000466 | 105483.57 | 42.9 |
| 4 | 3222.1 | 0.000310 | 100962.52 | 41.0 |
| 5 | 2438.4 | 0.000410 | 84652.12 | 34.4 |
| 6 | 11.0 | 0.091266 | 61732.65 | 25.1 |
| 7 | 10.4 | 0.096265 | 42467.37 | 17.3 |
| 8 | 11.0 | 0.090723 | 40992.89 | 16.7 |
| 9 | 11.0 | 0.090656 | 40088.38 | 16.3 |
| 10 | 11.0 | 0.091199 | 37539.64 | 15.3 |
| 11 | 11.0 | 0.090612 | 34844.80 | 14.2 |
| 12 | 1804.4 | 0.000554 | 34392.62 | 14.0 |

### `char_op`

| rank | period | freq | power | rel_power |
|---|---:|---:|---:|---:|
| 1 | 13721.9 | 0.000073 | 306226.65 | 44.6 |
| 2 | 11434.9 | 0.000087 | 285964.72 | 41.7 |
| 3 | 21.0 | 0.047642 | 265384.85 | 38.7 |
| 4 | 10.5 | 0.095141 | 230128.42 | 33.5 |
| 5 | 8893.8 | 0.000112 | 209196.88 | 30.5 |
| 6 | 10440.6 | 0.000096 | 200312.45 | 29.2 |
| 7 | 20.9 | 0.047773 | 195289.77 | 28.4 |
| 8 | 21.1 | 0.047494 | 182332.24 | 26.6 |
| 9 | 21.0 | 0.047536 | 169971.41 | 24.8 |
| 10 | 10.5 | 0.095132 | 168175.61 | 24.5 |
| 11 | 7504.2 | 0.000133 | 165062.61 | 24.0 |
| 12 | 10.5 | 0.095191 | 164836.18 | 24.0 |

### `tok_punct`

| rank | period | freq | power | rel_power |
|---|---:|---:|---:|---:|
| 1 | 4511.0 | 0.000222 | 370280.97 | 135.1 |
| 2 | 3007.3 | 0.000333 | 242517.18 | 88.5 |
| 3 | 2577.7 | 0.000388 | 188837.59 | 68.9 |
| 4 | 2148.1 | 0.000466 | 173464.36 | 63.3 |
| 5 | 11.0 | 0.090867 | 68249.20 | 24.9 |
| 6 | 1804.4 | 0.000554 | 65314.75 | 23.8 |
| 7 | 10.4 | 0.096265 | 46048.78 | 16.8 |
| 8 | 1455.2 | 0.000687 | 40263.21 | 14.7 |
| 9 | 7.5 | 0.133751 | 38860.14 | 14.2 |
| 10 | 835.4 | 0.001197 | 38690.51 | 14.1 |
| 11 | 5.1 | 0.197894 | 32767.82 | 12.0 |
| 12 | 9.4 | 0.106850 | 32693.34 | 11.9 |

### `clause_len`

| rank | period | freq | power | rel_power |
|---|---:|---:|---:|---:|
| 1 | 935.7 | 0.001069 | 5171557.88 | 14.3 |
| 2 | 21.6 | 0.046313 | 2998251.46 | 8.3 |
| 3 | 33.2 | 0.030103 | 2696736.88 | 7.5 |
| 4 | 6.7 | 0.149982 | 2674203.54 | 7.4 |
| 5 | 5.4 | 0.184361 | 2485596.34 | 6.9 |
| 6 | 19.4 | 0.051478 | 2352849.51 | 6.5 |
| 7 | 7.3 | 0.136623 | 2340607.03 | 6.5 |
| 8 | 5.5 | 0.183114 | 2326963.46 | 6.4 |
| 9 | 6.6 | 0.151763 | 2318032.37 | 6.4 |
| 10 | 10.1 | 0.098682 | 2268208.23 | 6.3 |
| 11 | 11.6 | 0.086035 | 2202952.21 | 6.1 |
| 12 | 7.1 | 0.140007 | 2198744.71 | 6.1 |

For the operator series the top peaks sit at the **longest** periods (thousands of tokens) - the signature of a slowly varying rate rather than a short repeating motif. A metrical loop would put the strongest peak at a short, stable period with harmonics. The clause-length series has no dominant peak at all (every `rel_power` is 6-14, i.e. a flat spectrum).

There is a weak secondary band at ~11 tokens in the raw `tok_op` (`rel_power` 14-25; band 9-13 `p_block`=0.030). It too is markup-driven: the same band is much weaker after screening (`rel_power` 14.2 at period 10.4), so it does not survive as a book-level rhythm.

_Note on `harmonic ratio`: it is meaningful for a narrow peak. For a low-frequency drift bump the `period/2` bin lies inside the same bump, so the ratio is also elevated (e.g. `tok_op` 23.9); it is not evidence of a pulse train. Only a narrow peak with a much smaller `period/2` lobe would be._

## 7. Fano factor (burstiness vs periodicity)

`F(T) = Var(count in window T)/Mean(count)`. `F = 1` is Poisson; growth with `T` indicates rate drift/clustering; oscillations would indicate periodicity. Book vs English token punctuation:

| window T | book tok_op | English tok_punct |
|---:|---:|---:|
| 1 | 0.878 | 0.827 |
| 2 | 0.896 | 0.850 |
| 4 | 0.910 | 0.885 |
| 8 | 0.921 | 0.864 |
| 16 | 0.993 | 0.941 |
| 32 | 1.167 | 1.088 |
| 64 | 1.591 | 1.301 |
| 128 | 2.391 | 1.725 |
| 256 | 4.135 | 2.435 |
| 512 | 7.053 | 3.245 |
| 1024 | 8.982 | 4.472 |
| 2048 | 16.434 | 5.916 |
| 4096 | 17.781 | 7.336 |

Both grow monotonically - no Fano oscillation - and the book grows faster, consistent with a more heterogeneous rate across the book than across the concatenated English references.

## 8. English baseline spread

Per-reference English punctuation statistics at matched N (`english_baseline.tsv`):

| reference | N | rate | flatness | peak_rel | peak period | interval CV | geo var ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| pride_and_prejudice | 90,220 | 0.1726 | 0.561 | 19.9 | 2506 | 0.823 | 0.818 |
| moby_dick | 90,220 | 0.1820 | 0.549 | 37.8 | 2654 | 0.765 | 0.715 |
| frankenstein | 86,128 | 0.1240 | 0.543 | 26.4 | 2778 | 0.719 | 0.591 |
| dracula | 90,220 | 0.1681 | 0.541 | 26.5 | 504 | 0.804 | 0.776 |
| sherlock_holmes | 90,220 | 0.1791 | 0.557 | 24.6 | 3759 | 0.931 | 1.056 |
| heart_of_darkness | 47,242 | 0.1730 | 0.551 | 29.4 | 4724 | 0.790 | 0.755 |
| dorian_gray | 90,220 | 0.1762 | 0.554 | 20.2 | 1327 | 0.800 | 0.778 |
| ulysses | 90,220 | 0.2055 | 0.557 | 19.3 | 2654 | 0.786 | 0.777 |

## 9. Reconciliation

| quantity | earlier | here | note |
|---|---|---|---|
| operator tokens (#17 OP) | 11,000 | 11,000 | 10,949 body + 51 header (#10) |
| body operator stream | 10,949 (#10/#11) | 10,949 | region=='para' OP tokens |
| operator stream identity | #11 operator_stream.tsv | match | operator-for-operator per page |
| token stream length | 90,220 (#17) | 90,220 | sorted by (page, start, tid) |
| reconstructed char length | - | 480,267 | gap+text over all tokens |
| operator chars | - | 17,269 | chars covered by OP spans |
| clauses (#33) | 5,614 | 5,614 | clause word-count series |
| markup OP tokens screened | - | 732 | #27 tag-run spans |
| white-noise flatness anchor | - | 0.5615 | exp(-Euler gamma) for exponential powers |

## 10. Caveats and negative results

- **The headline is a negative result for the periodicity claim.** The symbol stream is close to white noise (flatness 0.532 vs the 0.561 anchor; 0.555 after markup screening) and its intervals are close to geometric; there is no fixed cycle to find.
- **The dominant spectral power is slow OCR/HTML drift, not periodicity.** The raw `tok_op` peak (4511-token period, p_global=0.005) is carried by 732 operator tokens inside #27 markup spans (486 on page 92 alone). Removing them makes the spectrum white with no significant peak (p_global=0.333, p_block=0.701).
- **The screen-based `tok_op_screened` punches a hole.** Deleting page 92's dense markup cluster leaves one very long operator-free gap, which inflates the interval CV; the `tok_op_nomarkuppages` sensitivity (drop the three markup-heavy pages, no holes) keeps the near-geometric property and is the cleaner renewal estimate.
- **The character series is a dilation of the token series** - a multi-char operator such as `::` becomes two adjacent ones - so its short-lag autocorrelation is mechanically higher and its interval CV is inflated. The token series is the interpretable one.
- **English is literary prose, not cyberpunk** (still in copyright); the eight #24 references mix authors and dates, which inflates the English spread. The direction of every comparison is conservative toward the book.
- **Block shuffling is a blunt null**: it destroys all structure above the block scale by construction, so `p(block)` is a lower bound on the significance of any long-range cycle. A peak that fails the block test is long-range/clustered; a peak that passes it is local. Both are reported.
- **Absence of periodicity is not absence of loop.** The program's other instruments locate the loop in *repetition* (a restricted English core, #24; self-similar clauses, #39) and in *distribution* (near-uniform operator entropy, #47), not in a metrical cycle of the symbol stream. This result is consistent with #11's "shallow local run syntax".
- **No wall-clock quantity enters any compared artifact**; shuffles use a fixed seed (42) and numpy's PCG64 generator.

## 11. Artifacts

| file | contents |
|---|---|
| `periodicity_report.md` | THE DELIVERABLE |
| `series_specs.tsv` | every series: level, definition, N, rate |
| `autocorrelation.tsv` | ACF by lag for the primary series |
| `spectrum_peaks.tsv` | top spectral peaks with p-values |
| `spectrum.tsv` | decimated periodogram |
| `periodicity_tests.tsv` | one row per series: all statistics |
| `distance_histogram.tsv` | inter-delimiter distances vs geometric |
| `fano.tsv` | Fano factor F(T) per series |
| `english_baseline.tsv` | per-reference English statistics |
| `reconciliation.tsv` / `diagnostics.tsv` | join contract and counts |
| `summary.json` | machine-readable headline numbers |
| `periodicity_spectrum.png` / `periodicity_autocorr.png` | plots |

---
*Generated by `work/periodicity.py` (schema `periodicity/1`) on 2026-10-07T22:55:17Z.*
