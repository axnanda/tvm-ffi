**Part 1: balanced_uniform init (s=1, tail=0), H_o and full H at init.** zeros = count below 1e-8*max; split = log-gap (dominant, bulk) counts.
| setting | L | P | dom mean (r^2 eigs) | bulk mean (next r(d_in+d_out-2r)) | zeros / expected | split | ratio_paper | coherence | C1(H_o) | C2 (ratio=L, coh=L) | H_f/H_o | asym | sym(H): lam_max, lam_min, #neg, ratio_paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T (r=2,d_out=4) | 1 | 64 | 1.0000 | 1.0000 | 0 / 28 | (1,19) | 1.0000 | 1.0000 | no | yes, yes | 5.02e-16 | 0.0e+00 | 1.000, 1.000, 0, 1.000 |
| T (r=2,d_out=4) | 2 | 320 | 2.0000 | 1.0000 | 284 / 284 | (4,32) | 2.0000 | 2.0000 | yes | yes, yes | 1.432 | 5.0e-16 | 3.101, -1.227, 59, 2.290 |
| T (r=2,d_out=4) | 3 | 576 | 3.0000 | 1.0000 | 540 / 540 | (4,32) | 3.0000 | 3.0000 | yes | yes, yes | 1.607 | 5.4e-16 | 5.474, -1.304, 63, 3.372 |
| A (r=1,d_out=4) | 2 | 320 | 2.0000 | 1.0000 | 301 / 301 | (1,18) | 2.0000 | 2.0000 | yes | yes, yes | 1.818 | 5.9e-16 | 3.232, -1.380, 31, 2.379 |
| B (r=2,d_out=6) | 2 | 352 | 2.0000 | 1.0000 | 312 / 312 | (4,36) | 2.0000 | 2.0000 | yes | yes, yes | 1.511 | 5.8e-16 | 3.118, -1.312, 59, 2.272 |

**Part 2: non-uniform s_l (L=4, setting T).** cf = closed form sum_l prod_{m!=l} s_m^2 / max(prod_{m!=1}, prod_{m!=L}); ratio_paper uses the bulk MEAN; coherence_cf = dom / max_l prod_{m!=l} s_m^2.
| s | dom meas / cf | bulk_in cf | bulk_out cf | bulk mean meas | ratio_paper meas / cf(bulk mean) | dom/bulk_max meas / cf(max) | coherence meas / cf | L | split | eigs match cf |
|---|---|---|---|---|---|---|---|---|---|---|
| [1.0, 1.3, 0.8, 1.1] | 5.2096 / 5.2096 | 1.3087 | 1.0816 | 1.2803 | 4.0689 / 4.0689 | 3.9807 / 3.9807 | 2.5476 / 2.5476 | 4 | (4,32) | yes |

**Part 3: tail_value t (L=4, setting T).** closed form (post-hoc): dominant = L, bulk = sum_{k<L} t^{2k}, third cluster of (d_in-r)(d_out-r)=28 eigenvalues at L t^{2(L-1)}.
| t | dom [min,max] | bulk [min,max] | bulk cf | 3rd cluster mean (cf) | ratio_paper (cf) | ratio_split | split (dom,bulk) | rank | zeros | #>1e-3 max | #>1e-6 max | coherence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | [4.0000,4.0000] | [1.00000,1.00000] | 1.00000 | 8.01e-16 (0.0000) | 4.00000 (4.00000) | 4.0000 | (4,32) | 36 | 796 | 36 | 36 | 4.0000 |
| 0.2 | [4.0000,4.0000] | [1.04166,1.04166] | 1.04166 | 2.56e-04 (2.56e-04) | 3.84001 (3.84001) | 7.1985 | (4,60) | 64 | 768 | 36 | 64 | 4.0000 |

**Part 4: GD (lr=0.05) from balanced init, setting T; H_o and full H every 25 steps.** rates in 1/step (fit of log||H_f||_F); theory = ln(1 - lr*lambda_bulk).
| L | steps | loss init -> final | H_f/H_o init -> final | fit all: rate, R2, n | fit early: rate, R2 | fit late: rate, R2 | theory rate (bulk) | loss late rate (theory) | ratio_paper [min,max] final | coherence [min,max] final | balancedness max / final | ratio(sym H) final | asym max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 100 | 1.94e+00 -> 4.9e-01 | 1.607 -> 3.9e-01 | -0.01768, 0.715, 5 | -0.03714, 0.952 | -0.01768, 0.7150 | -0.02521 | -0.01220 (-0.05042) | [3.0000,3.3038] 3.3038 | [2.9704,3.0000] 2.9705 | 2.0e-02 / 2.0e-02 | 3.4588 | 5.4e-16 |

Post-hoc (part 4): ratio_paper predicted from the end-to-end singular values under exact balancedness vs measured:
| L | measured ratio range | predicted range | max abs diff | max rel err top-36 eigs | sigma_2 min |
|---|---|---|---|---|---|
| 3 | [3.0000,3.3038] | [3.0000,3.2791] | 0.0247 | 9.57e-03 | 0.116 |

**Part 4b (extra): GD reference numbers (lr=0.05, 1000 steps, checkpoints every 100) for T/A/B at L in {2,4,8}.**
| setting | L | loss final | H_f/H_o final | ratio_paper [min,max] final | coherence [min,max] final | balancedness max | fit rate (all), R2 | C5 (ratio, coh within 0.1% of L all along) |
|---|---|---|---|---|---|---|---|---|
| A | 2 | 3.9e-03 | 1.1e-01 | [2.0000,2.0136] 2.0055 | [1.9800,2.0000] 1.9918 | 8.0e-03 | -0.02844, 0.902 | no, no |

**Part 5: gaussian init (gain 1), setting T, lr = min(0.05, 1/lambda_max(H_init)), GD until loss<1e-6 (max 15000; state at 3000 also shown).** rank(H_o) = 36 for balanced init.
| L | seed | lr | stop (step) | loss @3000 / final | ratio_paper init / @3000 / final | ratio_gap init / final | log-gap dom count init / @3000 / final | ratio_split init / final | coherence init / @3000 / final | rank(H_o) init / final | balancedness mean init / final | top-eigvec layer energy final | H_f/H_o final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 0.0500 | max_steps (300) | 1.6e-05 / 1.6e-05 | 1.628 / 1.680 / 1.680 | 1.034 / 1.059 | 2 / 2 / 2 | 2.301 / 2.500 | 1.407 / 1.352 / 1.352 | 64 / 64 | 4.976 / 4.954 | [0.26, 0.74] | 2.2e-03 |
| 4 | 0 | 0.0353 | max_steps (300) | 2.9e-04 / 2.9e-04 | 2.695 / 2.724 / 2.724 | 1.012 / 1.064 | 3 / 14 / 14 | 4.526 / 4.745 | 2.683 / 2.451 / 2.451 | 64 / 64 | 5.639 / 5.625 | [0.15, 0.34, 0.31, 0.2] | 4.6e-03 |
