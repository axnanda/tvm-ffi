# Do spiking neural networks obey "Depth, Not Data" (arXiv 2602.00545)?

**Short answer: no.** The paper's spectral-bifurcation theory holds exactly for deep linear networks
(we reproduce every number), and approximately for tanh networks near a small balanced initialisation
(the paper's only nonlinear experiment), but leaky-integrate-and-fire spiking networks trained with
surrogate gradients break every one of its claims, for a structural reason: with a Heaviside forward pass
the Jacobian of the surrogate-gradient field is block upper-triangular in layer order, so the coherent
cross-layer summation that produces the paper's `ratio = L` cannot occur. What an SNN shows instead is a
"feature-mean" outlier cluster of size `d_out` created by the non-negativity of spike rates, i.e. exactly
the data-covariance mechanism the paper argues against, now generated inside the network.

Everything below was produced in this branch on a 4-CPU container without a GPU and without access to
arxiv.org (blocked by the environment's network policy); the paper's content was reconstructed from search
snippets, third-party summaries and our own derivation (see [docs/related_work.md](docs/related_work.md)
for sourced claims and [docs/theory_notes.md](docs/theory_notes.md) for derivations).

## 1. The claims under test

Deng, Liao, Ouyang, Pang and Yang, *Depth, Not Data: An Analysis of Hessian Spectral Bifurcation*
(arXiv 2602.00545, Jan 2026; accepted at ISIT 2026). Setting: deep linear network
`f(x) = W_L ... W_1 x`, MSE loss, whitened inputs (`E xx^T = I`), rank-`r` balanced targets, and a
balanced initialisation in which all `W_l` share one SVD with equal singular values (USI).

| id | claim | where the paper states it |
|---|---|---|
| C1 | The Hessian has a dominant cluster of `r^2` eigenvalues, a bulk cluster of `(d_0 + d_L - 2r) r` eigenvalues, and zeros elsewhere | Thm 3.5, Cor 3.6 |
| C2 | dominant / bulk = `Theta(L)`; exactly `L` under USI | abstract, Cor 3.6 |
| C3 | `H = H_o + H_f` (Gauss-Newton + functional); `H_f` decays exponentially under GD, so `H -> H_o` | Lemma A.8, Weyl step |
| C4 | The gap is caused by depth, not by data-covariance imbalance | title, abstract |
| C5 | GD from a balanced init keeps the structure (the ratio stays `L`) | Thm 3.5 for all `t` |
| C11 | Empirically, tanh networks with `L = 2..7` also show "Gap = L" | experiments section (v3) |

Our own derivation of C1/C2 (docs/theory_notes.md, section 1) gives the same counts and a closed form for
non-uniform singular values, `ratio = sum_l prod_{m != l} s_m^2 / max(prod_{m != 1} s_m^2, prod_{m != L} s_m^2)`,
which the experiments confirm to 1e-6.

Two predictions we add for nonlinear networks:

* **P1 (coherence bound).** For any model, `lambda_max(H_o) <= sum_l lambda_max(B_ll) <= L max_l lambda_max(B_ll)`
  (Cauchy-Schwarz on the per-sample Jacobians; `B_ll` is the layer-`l` diagonal block). The paper's
  `ratio = L` is precisely the saturation of this bound. We call `C = lambda_max(H_o) / max_l lambda_max(B_ll)`
  the *coherence*; it is `L` for the paper's setting and must be `< L` whenever layer Jacobians are data dependent.
* **P2 (feature-mean spike).** The readout block of any network with a linear readout is
  `I_{d_out} (x) Sigma_a`, `Sigma_a = a^T a / n` the uncentred second moment of the last hidden features.
  Spike rates are non-negative, so `Sigma_a = mu mu^T + Cov_a` has one dominant direction, giving `d_out`
  outliers (not `r^2`) that live in the readout layer. This is the Papyan / Sagun "class-mean" mechanism.

## 2. What "the Hessian" of an SNN is, and the block-triangular theorem

The true loss of an SNN is piecewise constant in the weights, so its Hessian is zero almost everywhere.
Three computable objects are used in practice; we compute all three (library: `snnland/hessian.py`):

| object | definition | properties |
|---|---|---|
| `H_o^sur` | `J^T J / n` with the surrogate Jacobian `J` (the Gauss-Newton / Fisher proxy used by A2SG and SpiKFAX) | symmetric PSD; has cross-layer blocks |
| `H = dg/dtheta` | Jacobian of the surrogate-gradient field `g` (double backprop through a *correctly implemented* straight-through spike, validated against finite differences to 1e-10) | **non-symmetric** (asymmetry 0.4-1.2), block upper-triangular |
| `sym(H)` | `(H + H^T)/2` | indefinite: 44-57 % negative eigenvalues at every depth |

**Theorem (verified numerically, experiment e7).** With a Heaviside forward pass, earlier-layer weights
reach later layers only through spike values, whose derivative is zero a.e. Hence `d g_l / d W_m = 0` for
`m < l`: `H` is block upper-triangular, its eigenvalues are the union of the per-layer diagonal blocks, and
the readout block is exactly `I_{d_out} (x) Sigma_a`. Numerically (64 configurations, `L = 2..6`, both
target settings, init and trained): max lower-block norm `0.0`, eigenvalue-union error `<= 7e-14`,
readout-block error `0.0`; ReLU and the smooth sigmoid-forward control have non-zero lower blocks (norms
2-50). Consequences:

* No symmetric cross-layer coupling exists in `H`, so the paper's mechanism cannot operate on it.
* The hidden-layer diagonal blocks contain only `sg''`-weighted residual terms (the paper's `H_f` type);
  they are indefinite and shrink with the residual (block norm 4.1 -> 0.25 at `L = 2` after training).
* At an interpolating point the spectrum of `H` is exactly `eig(Sigma_a)` repeated `d_out` times, plus zeros.
  Depth enters only through the rate statistics `Sigma_a`. After 400 Adam steps the top eigenvalue of `H`
  already equals the top eigenvalue of `Sigma_a` (e.g. 1.711 vs 1.711 at `L = 2`, 0.691 vs 0.691 at `L = 6`),
  see `figures/e7_update_field_structure.png`.

## 3. Methods

* **Library** `snnland/` (float64, CPU, 22 unit tests): functional deep linear / ReLU / tanh / smooth-SNN /
  LIF-SNN models (direct-current encoding over `T` steps, soft or hard reset, sigmoid or atan surrogate of
  slope `k`, rate or membrane readout, no biases); exactly whitened data (`X^T X / n = I`), rank-`r`
  targets; exact Hessians via `torch.func`; Gauss-Newton decomposition; cluster statistics
  (paper-index split at `r^2`, data-driven largest-log-gap split, zero count), layer coherence,
  top-eigenvector layer energy, feature second-moment statistics, spike diagnostics; GD / Adam training
  with Hessian logging. The straight-through spike uses a custom `autograd.Function` whose forward-mode
  derivative of the spike *value* is zero, so that double backprop returns `dg/dtheta` and not the Hessian
  of a smooth stand-in (the naive implementation returned a symmetric matrix that disagreed with finite
  differences by O(1)).
* **Protocol** ([experiments/PROTOCOL.md](experiments/PROTOCOL.md)): `n = 256`, `d_in = 16`, width 16,
  MSE; target settings A `(r = 1, d_out = 4)` and B `(r = 2, d_out = 6)` chosen so that `r^2 != d_out`;
  SNN defaults `T = 8`, `beta = 0.9`, `theta = 1`, `k = 4`, sigmoid surrogate, soft reset, rate readout,
  Kaiming init (mean spike rate 0.13-0.32 depending on depth); `L` = number of weight matrices.
* **Statistics** per Hessian: all eigenvalues of `H_o` and `sym(H)`; `ratio_paper` = mean of the top `r^2`
  over the mean of the next `r (d_in + d_out - 2r)`; `ratio_gap` = `eig[r^2] / eig[r^2 + 1]`;
  `ratio_split` and `split_n_dominant` from the largest log-gap; coherence `C`; `||H_f||_F / ||H_o||_F`;
  asymmetry `||H - H^T||_F / ||H||_F`; zero and negative counts; per-layer energy of the top eigenvectors;
  `||mu||^2 / lambda_1(Cov_a)` of the readout features; spike rates and surrogate-window occupancy.

## 4. Results

### 4.1 e1: the paper's setting is reproduced exactly (deep linear nets)

`figures/e1_linear_replication_*.png`, `results/e1_linear_replication/`.

* Balanced init, `L in {1..8}`, three target settings: top `r^2` eigenvalues `= L` to 1e-14, the next
  `r (d_in + d_out - 2r)` eigenvalues `= 1` to 1e-14, the rest `= 0`; `ratio_paper = coherence = L`
  exactly; top eigenvectors carry energy `1/L` in every layer. (C1, C2 hold; `L = 1` has no gap.)
* Non-uniform singular values `[1.0, 1.3, 0.8, 1.1]`: ratio 4.069 (closed form 4.069), coherence 2.548
  (closed form 2.548); `[1.0, 2.0, 1.0, 0.5]`: 4.545 (> L, because the two bulk sub-clusters split).
* Tail singular values `t > 0`: dominant unchanged, bulk `= sum_k t^{2k}`, a third cluster of
  `(d_in - r)(d_out - r)` eigenvalues at `L t^{2(L-1)}` appears (the Pith review's objection is real but mild).
* GD from balanced init (`L = 3, 5`, lr 0.05): `||H_f||_F` decays exponentially at the predicted rate
  `ln(1 - lr * lambda_bulk)` (fit -0.0511 / step vs theory -0.0510, R^2 = 1.000) down to 1e-14; the loss
  reaches 1e-30. With a target that is *not* aligned with the initialisation the ratio makes an excursion
  (3.00 -> 3.30 -> 3.02) because the end-to-end singular values temporarily split, exactly as the
  closed form predicts from the measured `sigma_t`; balancedness is conserved to 2e-2. (C3, C5 hold.)
* Gaussian (unbalanced) init: at `L = 8` the ratio is 5.1-7.5 at init and 5.0-6.5 after GD, and the
  coherence 3.5-4.0 (not 8); the clean `r^2` cluster still exists but C2 is only `Theta(L)` with a slope
  well below 1.

### 4.2 e8: the paper's own configuration, with tanh and SNN arms

Exactly the paper's dimensions (`d_0 = 10`, hidden 20, `d_L = 16`, `r = 4`, `L = 2..7`), balanced
rank-4 init with the target aligned to it (the paper's Initialization 2.1), `n = 512`, GD (linear / tanh /
relu, lr 0.1) or Adam (SNN, lr 5e-3), 1000 steps. `figures/e8_paper_faithful_vs_L.png`,
`results/e8_paper_faithful/summary.json`. Means over seeds; "init / end".

| arm | `ratio_paper` (slope vs L, R^2) | coherence (slope, R^2) | gap at index 16 | dominant count | zeros at `L = 3` (paper: 832) | `||H_f|| / ||H_o||` end | asym |
|---|---|---|---|---|---|---|---|
| linear | `= L` exactly / `= L` | `= L` / `= L` | `= L` / `= L` | 16 / 16 | 832 | 1e-16 | 0 |
| tanh (`s = 0.5`) | 1.95..6.90 (0.99, 1.00) / 2.4..4.4 (0.31, 0.65) | 1.96..6.88 (0.98, 1.00) / 1.2 (0.01) | 1.7..6.2 / 1.00 | 16 / 40 | 395 | 0.01 | 0 |
| relu | 2.7..10.1 (1.41, 0.99) / 2.5..18 (2.1, 0.49) | 1.45..4.5 (0.57, 0.95) / 1.3..5.5 (0.81, 0.94) | 1.0..1.4 / 1.0..2.8 | 16,4,4,1,1,2 / 32,32,32,4,4,2 | 43 | 0.03-0.33 | 0 |
| **SNN**, rank-4 balanced init (`s = 3`) | 2.6..5.1 (0.45, 0.93) / 1.8..5.1 (0.70, 0.97) | 1.2..2.5 (0.31, 0.70) / 1.5..3.7 (0.47, 0.97) | 1.0 / 1.0 | 16,4,4,4,4,7 / 56,4,4,4,20,4 | 1 | 0.80-0.94 | 0.65-1.10 |
| **SNN**, full-rank orthogonal init (`s = 3`) | 2.1..4.1 (0.37, 0.96) / 1.3..7.2 (1.2, 0.90) | 1.1..2.5 (0.25, 0.90) / 1.1..2.3 (0.20, 0.86) | 1.0-1.5 / 1.0 | 16,16,13,5,3,1 / 1,10,10,10,10,10 | 0 | 0.95-1.04 | 1.08-1.30 |

Reading: the paper's tanh result (C11) is reproduced *at initialisation*, where `tanh(z) ~ z` for the small
balanced weights, and disappears after training (coherence 1.2 at every depth, no gap at index 16).
The SNN never has a gap at index 16 (`eig[16]/eig[17] = 1.00-1.05`), its coherence is `~0.5 L`, its
functional Hessian never vanishes (`||H_f|| ~ ||H_o||` after training while the linear net reaches 1e-16),
and its surrogate Jacobian is as asymmetric as it is large. The fixed-index `ratio_paper` does grow with `L`
in the SNN (slope 0.5-0.7), but there is no cluster: it is the ratio of the top of a smoothly decaying
spectrum to its middle (bulk spread 4-15x), not a dominant/bulk ratio.

### 4.3 e2a: SNN depth sweep on whitened rank-r data (3 seeds, `L = 2..8`)

`figures/e2a_snn_depth_*.png`, `results/e2a_snn_depth/`. Protocol defaults, Kaiming init, Adam 800 steps.

| `L` | setting | coherence init -> 800 | gap at `r^2` | data-driven dominant count (`H_o` / `sym H`) | `ratio_split` | zero % of `H_o` | `||H_f||/||H_o||` 0 -> 200 -> 800 | asym |
|---|---|---|---|---|---|---|---|---|
| 2 | A (`r^2 = 1`, `d_out = 4`) | 1.10 -> 1.32 | 1.03 -> 1.22 | 4 / 4 -> 3 / 4 | 4.2 -> 2.1 | 0 | 1.32 -> 0.54 -> 0.67 | 0.48 |
| 4 | A | 1.99 -> 2.06 | 1.21 -> 2.78 | 3.7 / 2 -> 1.3 / 3 | 3.5 -> 3.6 | 2 -> 6 | 1.45 -> 0.83 -> 0.83 | 0.99 |
| 8 | A | 3.14 -> 4.26 | 2.47 -> 1.41 | 1.7 / 1.3 -> 2 / 1.3 | 3.4 -> 4.2 | 47 -> 52 | 1.71 -> 0.96 -> 0.89 | 1.18 |
| 2 | B (`r^2 = 4`, `d_out = 6`) | 1.14 -> 1.45 | 1.02 -> 1.01 | 6 / 6 -> 2.7 / 6 | 2.2 -> 2.1 | 0 | 1.39 -> 0.65 -> 0.81 | 0.48 |
| 4 | B | 1.92 -> 1.94 | 1.29 -> 1.09 | 4.3 / 5 -> 2 / 6 | 2.7 -> 3.3 | 2 -> 1 | 1.38 -> 0.92 -> 0.90 | 0.99 |
| 8 | B | 3.54 -> 4.15 | 1.19 -> 1.22 | 1.5 / 1 -> 4 / 2 | 2.9 -> 3.2 | 30 -> 31 | 1.77 -> 0.98 -> 0.96 | 1.14 |

Linear fits over `L = 2..8` (seed means): coherence slope 0.33 (A) / 0.40 (B) at init and 0.51 / 0.47 after
training, R^2 >= 0.94, intercept ~0.3-0.6, i.e. `C ~ 0.4-0.5 L` with no saturation up to `L = 8`;
`ratio_gap` slope -0.01 (A) / 0.03 (B) after training; `ratio_split` slope 0.12-0.32 with R^2 <= 0.84.
The only clean cluster is at `L = 2-3` and has exactly `d_out` members in every seed (4 in A, 6 in B; never
`r^2 = 1` or 4), with 85-89 % of the top eigenvector's energy in the readout layer and
`||mu||^2 / lambda_1(Cov_a) = 3.6-5.8` for the rate features (P2). Deeper nets lose the cluster because
the readout features' mean-to-variance ratio drops (1.8 at `L = 8`) and the zero space grows (47 % of
`H_o` at `L = 8`, from silent units). An "orthogonal balanced" SNN init (every singular value of every
`W_l` equal) does not help: coherence 1.05 / 2.14 / 2.12 at `L = 2 / 4 / 8`.

`||H_f|| / ||H_o||` does not decay: it falls from ~1.4 to ~0.6-0.9 in the first 200 steps and then rises
again (A, `L = 2`: 0.54 -> 0.67) while the loss falls 10x, because `||H_f|| / ||residual||` grows ~10x as
more membrane potentials enter the surrogate window. Replication with a fresh seed (11, setting B):
coherence 1.11 / 3.18 at `L = 2 / 6`, gap 1.02 / 1.32, dominant count 6 / 1, asymmetry 0.48 / 1.13,
lower-block norm 0.0 (`results/verify_headline.json`).

### 4.4 e2b: non-spiking controls (ReLU, smooth sigmoid-forward SNN, Gaussian-init linear)

Same protocol as e2a (`L = 2..8`, settings A and B, 2 seeds, Adam 800 steps; linear: GD).
`results/e2b_controls_depth/tables.md`, `figures/e2b_controls_depth_*.png`. Linear fits over `L` of the
`H_o` statistics (seed means; paper predicts slope 1):

| kind | coherence slope init / end (R^2) | `ratio_split` slope init / end | dominant count at `L = 2, 3, 4, 8` (init) | `||H_f||/||H_o||` after training | zero % of `H_o` at `L = 8` |
|---|---|---|---|---|---|
| ReLU, A / B | 0.43 / 0.24 (0.93 / 0.77); 0.48 / 0.15 (0.99 / 0.60) | -0.28 / -0.05; 0.07 / -0.18 (all R^2 < 0.7) | 4, 3, 3.5, 1; 6, 4, 1.5, 1 | 0.002-0.008 | 45-62 |
| smooth SNN (sigmoid forward, `k = 4`), A / B | 0.45 / 0.25 (0.95 / 0.87); 0.37 / 0.26 (0.97 / 0.95) | -0.09 / 0.15; 0.06 / 0.00 | 4, 3, 3.5, 1; 6, 4, 3.5, 1 | 0.000-0.025 | 52-64 |
| linear, Gaussian init, A / B | 0.40 / 0.50 (0.96 / 0.99); 0.38 / 0.38 (0.92 / 0.96) | **1.71 / 1.95**; **1.82 / 1.46** (R^2 >= 0.87) | 2.5, 3, 2, 1; 5.5, 3, 2, 2 | 0.001-0.006 | 80-97 (rank-deficient by construction) |
| LIF-SNN (e2a, for comparison), A / B | 0.33 / 0.51 (0.99 / 0.98); 0.40 / 0.47 (0.99 / 0.94) | -0.02 / 0.32; 0.11 / 0.12 | 4, 3.3, 3.7, 1.7; 6, 4.7, 4.3, 1.5 | **0.67-0.96** | 47-52 |

* Every nonlinear control behaves like the SNN on C1/C2: the dominant count is `d_out` at `L = 2`
  (4 in A, 6 in B, never `r^2`), decays to 1-2 by `L = 8`, and the data-driven dominant/bulk ratio is
  flat in `L` (2-5). Coherence grows with slope 0.4-0.5 at init in all of them and *drops* after
  training for ReLU and the smooth SNN (slope 0.15-0.26) while it *rises* for the LIF-SNN (0.47-0.51).
* The spiking-specific differences are (i) `H_f`: the true Hessians of ReLU and the smooth SNN become
  Gauss-Newton dominated after training (`||H_f||/||H_o|| <= 0.025`, as the paper's C3 says), whereas the
  LIF-SNN's surrogate `H_f` stays at 0.7-1.0; (ii) symmetry: asymmetry 1e-16 for the controls vs 0.5-1.2
  for the LIF-SNN.
* Gaussian-init *linear* nets are the only control with a growing gap: `ratio_split ~ 1.7-1.9 L`
  (13-15 at `L = 8`), but it is carried by 1-2 eigenvalues, not an `r^2` cluster, and the coherence
  is `~0.4-0.5 L`; i.e. without the balanced initialisation the paper's exact structure is already gone.
* Centring the readout features (setting B, seed 0) collapses the `d_out` cluster at `L = 2`: ReLU
  6 -> 2 dominant (readout block `lambda_max` 6.3 -> 2.0, top-eigenvector energy in `W_L` 0.88 -> 0.66),
  smooth SNN 6 -> 2 (`lambda_max(B_LL)` 1.87 -> 0.41, `ratio_split` 2.85 -> 1.14). At `L >= 4` the cluster
  is already absent and centring changes little. This confirms P2: the outliers of nonlinear nets on
  whitened data are a feature-mean (data-covariance) effect generated inside the network.

### 4.5 e7: structure of the update-field Jacobian

Summarised in section 2; `figures/e7_update_field_structure.png`, `figures/e7_scalars_vs_L.png`,
`results/e7_update_field_structure/summary.json` (64 rows).

### 4.6 e4: "depth, not data" under controlled covariance imbalance

Kinds SNN / ReLU / Gaussian-init linear, setting B, `L in {2, 4, 8}`, 2 seeds, 400 Adam steps (GD for
linear); input covariance with geometric spectrum of condition number `kappa in {1, 10, 100, 1000}` at
fixed trace, imbalanced target singular values `[1, 0.1]`, and an un-normalised condition (25 % of the
input coordinates scaled by 10, `kappa = 100`, trace 412). 108 runs; `results/e4_data_imbalance/tables.md`,
`figures/e4_data_imbalance_*.png`. Slopes of the `H_o` statistics after training:

| kind | metric | depth slope (per unit `L`, `kappa = 1`) | data slope (per decade of `kappa`, pooled) |
|---|---|---|---|
| SNN | `ratio_paper` / coherence / `ratio_split` | 0.89 / 0.37 / 0.24 | +0.16 / +0.07 / -0.22 |
| ReLU | `ratio_paper` / coherence / `ratio_split` | 0.21 / 0.12 / -0.19 | +0.52 / +0.14 / -0.01 |
| linear (Gaussian init) | `ratio_paper` / coherence / `ratio_split` | 0.42 / 0.38 / 1.26 | **+1.17 / +0.16 / +3.32** |

* For the SNN's surrogate Gauss-Newton matrix the *direction* of C4 holds: at fixed trace the gap
  statistics are flat in `kappa` (log-log exponents `<= 0.03`) and grow with depth (coherence
  1.25 -> 1.79 -> 3.44 for `L = 2, 4, 8`), and target imbalance changes nothing. The *mechanism* is not the
  paper's: there is no `r^2` cluster (data-driven dominant count 1.5-4), coherence is `~0.45 L`, both the
  top eigenvalue and the bulk edge scale like `kappa^0.1`, and the top eigenvectors migrate into the first
  layer as `kappa` grows (energy in `W_1` at `L = 2`: 0.09 -> 0.86), i.e. the dominant directions become
  input-covariance directions. The un-normalised condition collapses coherence to 1.0 with 96-100 % of
  the top-eigenvector energy in `W_1`: absolute input scale overrides depth entirely.
* For the SNN's actual update-field Jacobian `H`, C4 is moot: the largest eigenvalue equals
  `lambda_max(Sigma_a)` of the readout rates exactly at every `L` and `kappa` (`d_out = 6` copies), with no
  depth dependence; `||H_f||/||H_o||` stays 0.63-0.99.
* ReLU: depth is the more consistent driver, but `kappa` moves coherence at `L = 2` (1.13 -> 1.66) and
  `ratio_paper` at `L = 8` (5.6 -> 9.8) by comparable amounts.
* Gaussian-init linear nets *fail* C4: the gap grows with `kappa` as much as with depth
  (`ratio_split` +3.3 per decade vs +1.3 per layer), because without the balanced construction the first
  layer block of `H_o` is `(A^T A) (x) Sigma_x` and the input covariance enters the top of the spectrum
  directly. "Depth, not data" is a property of the balanced-init construction, not of deep linear nets
  in general.

### 4.7 e3 (SNN hyper-parameters), e5 (real data), e6 (training dynamics)

_Pending; see section 9._

## 5. Claim-by-claim verdict

| claim | linear, balanced init (paper) | linear, Gaussian init | tanh (small balanced init) | ReLU | smooth SNN (sigmoid forward) | LIF-SNN (surrogate gradient) |
|---|---|---|---|---|---|---|
| C1 `r^2` dominant + equal bulk + zeros | **holds** exactly (e1, e8) | partially: a 1-2 eigenvalue spike with a large gap, no `r^2` cluster, no equal bulk (e1, e2b) | holds at init, lost after training (e8) | fails: no gap at `r^2`; `d_out` cluster at `L <= 3` only (e2b, e8) | fails: same as ReLU (e2b) | **fails**: no gap at `r^2` (1.00-1.05); the only cluster has `d_out` members and lives in the readout (e2a, e8) |
| C2 ratio `= L` / `Theta(L)` | **holds** exactly | gap grows (`ratio_split ~ 1.8 L`) but coherence `~0.4-0.5 L` | slope 0.99 at init, ~0 after training | coherence slope 0.43-0.48 at init, 0.15-0.24 after training; `ratio_split` flat | coherence slope 0.37-0.45 / 0.25; `ratio_split` flat | **fails**: coherence `~0.4-0.5 L`, gap slope ~0; only the gapless fixed-index ratio grows |
| C3 `H_f -> 0` exponentially, `H -> H_o` | **holds** (rate matches theory) | holds | holds (`||H_f||/||H_o||` 0.01 after training) | holds (<= 0.008 after training) | holds (<= 0.025) | **fails**: 0.7-1.0 after training; `H` is not even symmetric |
| C4 depth, not data | holds by construction | **fails**: the gap grows with `kappa` as fast as with `L` (e4) | n/a | partially: depth dominates, but `kappa` moves coherence / `ratio_paper` by up to +75 % (e4) | not tested | **partially for `H_o`** (ratios flat in `kappa` at fixed trace, but eigenvectors migrate to `W_1` and absolute input scale collapses coherence to 1); **fails for `H`**, whose top eigenvalue is `lambda_max(Sigma_a)` at every `L` (e4, e7) |
| C5 ratio constant during GD | **holds** (excursion only for unaligned targets) | n/a | fails (coherence `L -> 1.2`) | fails (coherence falls during training) | fails | **fails** (no `L` to preserve; coherence drifts, e.g. 1.1 -> 1.3 at `L = 2`) |
| P1 coherence `< L` for nonlinear nets | `= L` | `< L` | `< L` after training | `< L` (0.3-0.5 L) | `< L` | **holds**: `0.4-0.5 L` |
| P2 `d_out` feature-mean outliers from the readout | absent (centred features) | absent | not tested | **holds**: 6 -> 2 outliers when features are centred (`L = 2`) | **holds**: 6 -> 2, readout `lambda_max` 1.87 -> 0.41 | **holds** for `L <= 3`, diluted by dead units for `L >= 4` |
| block-triangular `dg/dtheta` | symmetric | symmetric | symmetric | symmetric (lower blocks 2-50) | symmetric (lower blocks 1-7) | **exactly block upper-triangular** (e7) |

## 6. Why the SNN behaves differently

1. **The coherence bound is far from saturated.** The paper's `L` comes from `L` identical copies of the same
   `r^2` directions adding in phase. In an SNN each layer's Jacobian factor carries a data-dependent,
   sparse surrogate mask `diag(sg'(u))` (only 24-29 % of (sample, time, unit) triples are inside the
   window) and the features are spike rates, so the layer contributions are only partially aligned:
   `C ~ 0.4-0.5 L`.
2. **Non-negative rates create a mean direction.** `Sigma_a = mu mu^T + Cov_a` with
   `||mu||^2 / lambda_1(Cov_a) = 2-6`: the readout block alone produces `d_out` outliers (classic
   class-mean structure), and because the update-field Jacobian is block-triangular this readout block
   is *all* that survives at an interpolating point.
3. **The functional term does not vanish.** Surrogate gradients are not the gradient of any loss
   (Gygax and Zenke 2024), so there is no Lyapunov argument driving the residual-weighted `H_f` to zero,
   and empirically `||H_f|| ~ ||H_o||` after training.
4. **Depth kills units.** The zero space of `H_o` grows from 0 % at `L = 2` to 30-47 % at `L = 8`
   (silent units), so the "bulk" is not depth-constant, contrary to the paper's `m_t^{2(L-1)}` bulk.

## 7. Verification performed

* 22 unit tests (`python3 -m pytest tests -q`), including the exact `ratio = L` replication, `H = H_o + H_f`,
  finite-difference validation of the SNN `dg/dtheta` (relative error 1e-10) for soft/hard reset,
  sigmoid/atan surrogate, rate/membrane readout, centred features and CE loss.
* Every linear eigenvalue in e1 matched the closed forms to 1e-6 or better (tables in
  `results/e1_linear_replication/tables.md`, `posthoc.json`).
* `experiments/verify_headline.py` recomputed the e1 clusters from the raw eigenvalues, the e8 slopes
  from the summaries, and re-ran the SNN at a fresh seed (above).
* The e7 identities (zero lower blocks, eigenvalue union, `I (x) Sigma_a` readout block) hold to
  machine precision in all 64 configurations.

## 8. Limitations

Small models (width 16-20, `P <= 2.5k`) so that exact Hessians could be computed on CPU; `n = 256-512`
samples; a single neuron model (LIF with direct-current encoding) and MSE regression targets as in the
paper; Adam rather than GD for the SNN sweeps (GD stalls on spiking nets, see e6); 2-3 seeds; the paper's
text itself could not be fetched, so Theorem 3.5's exact hypotheses are reconstructed (our derivation
reproduces all reported numbers, including the paper's 16 / 72 / 832 counts at `L = 3`).

## 9. Reproduction and file map

```
pip install -r requirements.txt            # torch (CPU is enough), numpy, scipy, matplotlib, scikit-learn, pytest
python3 -m pytest tests -q                 # 22 tests, ~15 s
PYTHONPATH=. python3 experiments/e1_linear_replication.py      # ~30 min at 1 thread
PYTHONPATH=. python3 experiments/e2a_snn_depth.py              # ~70 min
PYTHONPATH=. python3 experiments/e7_update_field_structure.py  # ~25 min
PYTHONPATH=. python3 experiments/e8_paper_faithful.py          # ~2 h
PYTHONPATH=. python3 experiments/verify_headline.py
```

| path | content |
|---|---|
| `snnland/` | library (models, data, hessian, analysis, train) |
| `tests/test_snnland.py` | unit tests |
| `experiments/PROTOCOL.md` | shared experimental protocol |
| `experiments/e*.py` | experiment scripts (each saves `results/<exp>/summary.json`, raw `.npz`, `run.log`) |
| `figures/` | all figures referenced above |
| `docs/related_work.md` | sourced literature memo (paper claims, prior theory, SNN landscape literature) |
| `docs/theory_notes.md` | derivations: linear result, coherence bound, feature-mean spike, block-triangular Jacobian |

Sources used for the paper's content (arxiv.org itself was unreachable):
[arXiv abstract](https://arxiv.org/abs/2602.00545), [arXiv HTML v1](https://arxiv.org/html/2602.00545v1)
and [v3](https://arxiv.org/html/2602.00545v3) via search snippets,
[Pith automated review](https://pith.science/paper/2602.00545),
[ResearchGate record](https://www.researchgate.net/publication/400369101_Depth_Not_Data_An_Analysis_of_Hessian_Spectral_Bifurcation),
[awesomepapers summary](https://awesomepapers.io/machine-learning/papers/2602.00545).
SNN landscape literature: [Gygax and Zenke 2024](https://arxiv.org/abs/2404.14964),
[A2SG](https://arxiv.org/html/2606.11236v1), [SpiKFAX](https://arxiv.org/abs/2609.29379),
[QIF loss landscapes](https://arxiv.org/abs/2606.03935), [Singh et al. 2021 (H_o / H_f)](https://arxiv.org/abs/2106.16225),
[Papyan 2020](https://arxiv.org/pdf/2008.11865); full list in `docs/related_work.md`.
