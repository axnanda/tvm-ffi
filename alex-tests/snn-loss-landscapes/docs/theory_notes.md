# Theory notes (to be folded into the report)

## 1. Re-derivation of the paper's linear result (uniform balanced init)
Net f(x) = W_L ... W_1 x, W_l = U_{l+1} S U_l^T, S = diag(s,...,s (r times), 0,...). Whitened X (E xx^T = I).
GN Hessian H_o = (1/n) sum_i J_i^T J_i, J_{i,l} = dy_i/dvec(W_l) = (W_L..W_{l+1}) (.) (W_{l-1}..W_1 x_i).
Perturbation directions of layer l in basis u_a v_b^T (u from U_{l+1}, v from U_l):
 * a<=r, b<=r ("in-subspace", r^2 per layer): output change s^{L-1} u'_a (v_b . x~) identical across layers
   -> the L copies add coherently -> r^2 eigenvalues L s^{2(L-1)}.
 * layer 1, a<=r, b>r (input tail, r(d_in-r) dirs): s^{L-1} u'_a (v_b . x), E(v_b.x)^2 = 1 -> eigenvalue s^{2(L-1)}.
 * layer L, a>r, b<=r (output tail, r(d_out-r) dirs): eigenvalue s^{2(L-1)}.
 * everything else killed by a zero singular value of a neighbouring layer -> eigenvalue 0.
=> dominant r^2 @ L s^{2(L-1)}, bulk r(d_in+d_out-2r) @ s^{2(L-1)}, rest 0, ratio = L exactly.
Non-uniform s_l: dominant = sum_l prod_{m!=l} s_m^2, bulk = max(prod_{m!=1} s_m^2, prod_{m!=L} s_m^2),
ratio in [1, L]; -> Theta(L) when the s_l are within a constant factor ("initialization spread" condition).
Under gradient flow, W_{l+1}^T W_{l+1} - W_l W_l^T is conserved, so a balanced init stays balanced and the
ratio stays exactly L throughout training (C5). H_f = (1/n) sum_i sum_k (y_hat - y)_{ik} d^2 y_ik/dtheta^2 is
linear in the residual, which decays exponentially under GD -> H_f -> 0 exponentially (C3).

## 2. A general bound: ratio = L is the saturation of a coherence bound
For any model, H_o = (1/n) sum_i J_i^T J_i is PSD with blocks B_lm. For v = (v_1..v_L):
 v^T H_o v = (1/n) sum_i || sum_l J_{i,l} v_l ||^2 <= ( sum_l sqrt(lambda_max(B_ll)) ||v_l|| )^2
          <= ( sum_l lambda_max(B_ll) ) ||v||^2  (Cauchy-Schwarz)
=> lambda_max(H_o) <= sum_l lambda_max(B_ll) <= L max_l lambda_max(B_ll).
Define coherence C := lambda_max(H_o) / max_l lambda_max(B_ll) in [1, L].
In the paper's linear uniform case C = L exactly (and the bulk eigenvalue equals max_l lambda_max(B_ll)),
so "dominant/bulk = L" is precisely the statement that the layers' curvature adds perfectly coherently.
For nonlinear nets the per-sample Jacobian factors G_i^{(l)} = W_L D_i^{(L-1)} ... D_i^{(l)} and
activations a_i^{(l-1)} are data dependent, which decorrelates layers and must give C < L.
Prediction P1: C_SNN(L) < L, and the question is whether C grows linearly with slope < 1 or saturates.

## 3. Nonlinearity re-creates the "data" mechanism inside the network (feature-mean spike)
For a feed-forward nonlinear net, the last-layer block is B_LL = (1/n) sum_i I_{d_out} (x) a_i a_i^T
= I_{d_out} (x) Sigma_a, where Sigma_a is the (uncentred) second-moment matrix of the last hidden
features. ReLU features of whitened Gaussian inputs have a positive mean; SPIKE-RATE features are
non-negative with mean comparable to their std (rates in [0,1]). Hence Sigma_a = mu mu^T + Cov_a has a
single dominant direction mu with eigenvalue ||mu||^2 >> eigenvalues of Cov_a. That gives d_out (NOT r^2)
outlier eigenvalues of H_o coming essentially from the LAST layer alone, i.e. the classic
"class-mean / feature-mean" (Papyan-style, data-covariance) mechanism the paper argues against,
re-created by the nonlinearity even though the INPUT covariance is perfectly balanced.
Prediction P2: for the SNN (and ReLU), the number of dominant eigenvalues tracks d_out rather than r^2,
the top eigenvectors' energy concentrates in W_L, and the "ratio" is governed by ||mu||^2 / lambda(Cov_a)
(i.e., spike-rate statistics) rather than by L. Discriminating experiments:
  (a) choose r=1, d_out=4 (r^2=1 vs d_out=4) and r=2, d_out=6 (4 vs 6);
  (b) centre the hidden features fed to the readout (subtract batch mean of rates) -> the spike should
      collapse; (c) report per-layer energy of the top eigenvectors; (d) vary spike rate (threshold/gain).
Prediction P3: the SNN's dominant/bulk ratio depends strongly on surrogate slope k only through an
overall scale (H ~ k^2 roughly, A2SG) but the *structure* (counts, coherence) depends on T, theta and
beta through the spike statistics.

## 4. What "the Hessian" of an SNN even is
True loss is piecewise constant in theta (Heaviside), so the true Hessian is 0 a.e. The object that is
analysed in practice is the Jacobian of the surrogate-gradient field g(theta); because the forward uses
H(v) while the backward uses sg'(v), g is not the gradient of any scalar potential and dg/dtheta is not
symmetric in general. Its Gauss-Newton part H_o = J^T J / n (J = surrogate Jacobian) IS symmetric PSD and is
the natural analogue of the paper's H_o. The paper's decomposition therefore splits for SNNs into a
well-defined H_o plus an H_f that carries all the asymmetry. Measure asymmetry(H) and compare with the
smooth relaxation (sigmoid forward, 'smooth_snn') whose Hessian is exact and symmetric.

## 5. The SNN update-field Jacobian is block upper-triangular (verified numerically, experiment e7)
Let g(theta) be the surrogate gradient and H := dg/dtheta its Jacobian (what double-backprop through a
*correctly implemented* straight-through spike returns; the library validates it against finite differences).
With a Heaviside forward pass, the weights W_m of an EARLIER layer m < l reach g_l = delta^{(l)} (x) a^{(l-1)}
only through spike values s^{(m)}, whose derivative is zero almost everywhere.  Hence
    d g_l / d W_m = 0   for m < l   (a.e.),
i.e. H is block upper-triangular in layer order (numerically: lower blocks exactly 0, see e7).  Consequences:
 (i)  eig(H) = union over layers of eig(d g_l / d W_l)  (checked to 1e-14).  Cross-layer coherent summation,
      the mechanism behind the paper's ratio == L, cannot occur in H: there is no symmetric cross-layer coupling.
 (ii) The readout block is exactly  d g_L / d W_L = I_{d_out} (x) Sigma_a,  Sigma_a = a^T a / n the uncentred
      second moment of the readout features (rates).  Its spectrum is d_out copies of eig(Sigma_a).
 (iii) The hidden-layer diagonal blocks  d g_l / d W_l  contain only sg''-weighted residual terms (the paper's
      H_f type): they are indefinite and vanish when the residual vanishes.  The surrogate Gauss-Newton
      term J_sur^T J_sur does NOT appear in H except through the readout column, because the output y_hat is
      insensitive (a.e.) to hidden weights.
 (iv) Therefore at any interpolating point the spectrum of the SNN's update-field Jacobian is exactly
      {eig(Sigma_a)} x d_out  union  {0}: a dominant cluster of d_out eigenvalues ~ ||mu||^2 (feature-mean
      direction), a bulk given by the covariance spectrum of the rates, and zeros -- with no explicit
      dependence on L.  Depth enters only through how it changes the rate statistics Sigma_a.
 (v)  sym(H) = (H + H^T)/2 mixes the one-sided upper blocks into both triangles and produces spurious negative
      eigenvalues; H_o^sur = J_sur^T J_sur / n (used as a Fisher proxy by SpiKFAX / A2SG) is PSD and does show
      cross-layer coupling, but it is the curvature of a smooth stand-in model, not of the SNN's update field.
Relu and smooth_snn have symmetric Hessians with non-zero lower blocks (e7 contrast), so this is specific
to the Heaviside/surrogate mismatch.
