# Shared experiment protocol (read fully before writing an experiment script)

Project root: /home/user/tvm-ffi/alex-tests/snn-loss-landscapes  (run everything from here;
`import snnland as sl` works from this directory). Library API: see `python3 -c "import snnland; help(snnland)"`
and the signatures below. Tests: `python3 -m pytest tests -q` (22 pass). Theory notes: docs/theory_notes.md.

## Paper under test (arXiv 2602.00545, "Depth, Not Data: An Analysis of Hessian Spectral Bifurcation")
Deep LINEAR nets, MSE, whitened balanced data, rank-r balanced targets, balanced init (shared SVD, equal
singular value s per layer). Claims: C1 Hessian (and Gauss-Newton part H_o) = dominant cluster of r^2 eigenvalues
+ bulk cluster of r(d_in+d_out-2r) equal eigenvalues + zeros.  C2 dominant/bulk ratio == L (depth = number of
weight matrices) under uniform init, Theta(L) otherwise.  C3 H = H_o + H_f, H_f ~ residual decays exponentially
under GD so H -> H_o.  C4 the gap is caused by depth, not by data-covariance imbalance.  C5 GD preserves
balancedness so the ratio stays L during training.  Coherence bound (docs/theory_notes.md):
lambda_max(H_o) <= L * max_l lambda_max(H_o[l,l]); the paper's ratio == L saturates it.

## Fixed conventions
- float64, CPU. FIRST LINE of every script: `import snnland as sl; sl.set_threads(1)` (4 CPUs are shared by
  several experiments running at once). Deterministic seeds. No internet.
- Defaults unless the experiment varies them: n=256 samples, d_in=16, widths=16, loss="mse".
  Target settings: "A" = (r=1, d_out=4) and "B" = (r=2, d_out=6)  [chosen so r^2 != d_out].
  Data: X = sl.make_whitened_inputs(n, d_in, seed); Y, Wstar = sl.make_rank_r_targets(X, r, d_out, seed).
- SNN defaults: ModelSpec("snn", L, d_in, 16, d_out, T=8, beta=0.9, theta=1.0, k=4.0, surrogate="sigmoid",
  reset="soft", detach_reset=False, readout="rate"); init = spec.init_params(seed) (kaiming default).
  ALWAYS record sl.spike_diagnostics(spec, params, X) next to every SNN Hessian (mean rate, window fraction).
- Controls: ModelSpec("relu", ...) and ModelSpec("smooth_snn", ..., k=4.0) with default init;
  ModelSpec("linear", ...) with init="gaussian" (gain 1) and with init="balanced_uniform" (r=r, s=1).
- L = number of weight matrices (L-1 hidden spiking/relu layers + linear readout). Use L in {2,3,4,5,6,8}
  unless told otherwise.
- Training: sl.gd_train(spec, params, X, Y, lr, steps, optimizer="adam"|"gd", log_every=..., callbacks=[...]).
  For sweeps use optimizer="adam", lr=5e-3 (snn/relu/smooth_snn) or GD lr=0.1 (linear). Record the loss curve.
  Checkpoints: compute H_o (cheap, ~0.1 s) at many steps; the full surrogate Hessian H (expensive: ~6 s for
  P~850, ~40 s for P~1900) only at init, mid and end of training unless the experiment is about dynamics.
- For every Hessian evaluation record (as a dict): step, loss, P, eigenvalues of H_o (descending), eigenvalues
  of sym(H) (descending), sl.asymmetry(H), ||H_f||_F/||H_o||_F (H_f = H - H_o), sl.cluster_stats(eigs, r, d_in, d_out)
  for BOTH matrices, sl.layer_coherence(H_o, spec.layer_slices) (keep 'coherence', per-block lambda_max and
  top-eigvec energy), sl.top_eigvec_layer_energy(H_o, spec.layer_slices, k=max(r*r, d_out)),
  sl.feature_second_moment_stats(spec, params, X), spike diagnostics (snn), and wall time.
  Data-driven dominant count: cluster_stats[...]["largest_log_gap_index"]+1 (verify the key names in the dict).
- Save: results/<exp>/summary.json (list of per-run dicts with all scalar stats, JSON-serialisable: convert
  tensors/np to python floats/lists) and results/<exp>/raw/<run_name>.npz (eigenvalue arrays, loss curves).
  Figures: figures/<exp>_<what>.png (matplotlib, Agg backend, dpi 120, log-scale spectra, legends, titles,
  seeds as separate thin lines or mean +- std). Never overwrite another experiment's folder.
- Script location: experiments/<exp>.py. Run it as `nohup python3 experiments/<exp>.py > results/<exp>/run.log 2>&1 &`
  via a Bash call with run_in_background=true that waits for the process to finish (e.g.
  `python3 experiments/<exp>.py > results/<exp>/run.log 2>&1`), so you get a completion notification; do not
  busy-wait with chained sleeps. First run a REDUCED smoke version (small L, 1 seed, few steps) to catch bugs,
  then the full run. Keep the total wall time of your experiment under ~75 minutes at 1 thread; if the plan
  is too expensive, reduce seeds/checkpoints first (never silently drop whole conditions; log what was reduced).
- Final answer (structured): key numbers per condition (tables), file paths, an honest interpretation
  relative to each paper claim, anomalies/bugs encountered, and what was reduced.

## API signatures (from the library)
ModelSpec(kind, L, d_in, widths, d_out, bias=False, center_features=False, T=8, beta=0.9, theta=1.0, k=4.0,
          surrogate='sigmoid', reset='soft', detach_reset=False, readout='rate', dtype=float64)
  .init_params(seed=0, init=None, gain=None, r=None, s=1.0, tail_value=0.0)   # init in {kaiming/gaussian/balanced_uniform/...}
  .forward(params, X, feature_mean=None); .hidden_features(params, X); .layer_slices; .n_params; .params_to_vec/.vec_to_params
sl.make_whitened_inputs(n, d, seed); sl.make_covariance_imbalanced_inputs(n, d, spectrum, seed); sl.geometric_spectrum(d, ratio)
sl.make_rank_r_targets(X, r, d_out, seed, singular_values=None, noise_std=0.0); sl.load_digits_dataset(d=None, whiten=True, test_frac=0.2, seed=0)
sl.full_hessian(model, params, X, Y, loss='mse', method='auto', chunk_size=128)  # Jacobian of the (surrogate) gradient field
sl.gauss_newton(model, params, X, Y, loss='mse', return_blocks=False, return_J=False); sl.functional_part(H, H_o)
sl.symmetrize(H); sl.asymmetry(H); sl.eigh(H_sym); sl.eigvals(H); sl.top_k(H, k); sl.check_hessian_fd(...)
sl.loss_fn(model, params, X, Y, loss); sl.surrogate_gradient(model, params, X, Y, loss)
sl.cluster_stats(eigs, r, d_in, d_out, tol=1e-8); sl.layer_coherence(H, layer_slices); sl.top_eigvec_layer_energy(H_sym, layer_slices, k)
sl.feature_second_moment_stats(model, params, X); sl.spike_diagnostics(model, params, X); sl.spike_stats(model, params, X)
sl.gd_train(model, params, X, Y, lr, steps, loss='mse', log_every=0, callbacks=None, optimizer='gd'|'adam')
sl.hessian_summary_callback(model, X, Y, loss, r, d_in, d_out, tol=1e-8)
Inspect return dicts before relying on key names (print one example).
