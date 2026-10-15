"""Fast correctness tests for snnland (pytest; < 2 min on 2 CPU threads)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.func import jacrev

import snnland as sl

torch.set_num_threads(2)
# pytest discards warning filters installed at import time (collection runs under catch_warnings),
# so repeat snnland's filter for torch's internal, non-actionable jit.script deprecation here.
pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated:FutureWarning")

D_IN, H, D_OUT, R, N = 8, 8, 4, 2, 64


def _linear_setup(L: int, s: float = 1.0, tail_value: float = 0.0, seed: int = 0):
    X = sl.make_whitened_inputs(N, D_IN, seed=seed)
    Y, _ = sl.make_rank_r_targets(X, R, D_OUT, seed=seed + 1)
    model = sl.ModelSpec("linear", L, D_IN, H, D_OUT)
    params = model.init_params(seed=seed + 2, init="balanced_uniform", r=R, s=s, tail_value=tail_value)
    return X, Y, model, params


# 1. data ------------------------------------------------------------------- #
def test_whitened_inputs_exact():
    X = sl.make_whitened_inputs(N, D_IN, seed=3)
    assert X.dtype == torch.float64
    assert torch.allclose(X.T @ X / N, torch.eye(D_IN, dtype=torch.float64), atol=1e-10, rtol=0)


def test_covariance_imbalanced_inputs_exact():
    spec = sl.geometric_spectrum(D_IN, 0.5)
    X = sl.make_covariance_imbalanced_inputs(N, D_IN, spec, seed=3)
    assert torch.allclose(X.T @ X / N, torch.diag(torch.as_tensor(spec)), atol=1e-10, rtol=0)


def test_rank_r_targets():
    X = sl.make_whitened_inputs(N, D_IN, seed=0)
    Y, Wstar = sl.make_rank_r_targets(X, R, D_OUT, seed=0)
    assert torch.linalg.matrix_rank(Wstar) == R
    assert torch.allclose(Y, X @ Wstar.T)
    assert torch.allclose(torch.linalg.svdvals(Wstar)[:R], torch.ones(R, dtype=torch.float64))


def test_digits_dataset_loads():
    data = sl.load_digits_dataset(d=16, whiten=True, seed=0)
    assert data["X_train"].shape[1] == 16 and data["Y_train"].shape[1] == 10
    assert torch.allclose(data["Y_train"].sum(1), torch.ones(len(data["Y_train"]), dtype=torch.float64))


# 2. linear balanced-uniform spectrum ---------------------------------------- #
@pytest.mark.parametrize("L,s", [(2, 1.0), (3, 1.0), (5, 1.0), (3, 1.2)])
def test_linear_balanced_uniform_gauss_newton_spectrum(L: int, s: float):
    X, Y, model, params = _linear_setup(L, s=s)
    H = sl.full_hessian(model, params, X, Y)
    H_o = sl.gauss_newton(model, params, X, Y)
    H_f = sl.functional_part(H, H_o)
    assert float(torch.linalg.norm(H_f)) > 1e-3  # residual is nonzero at init
    assert torch.allclose(H_o, sl.linear_gauss_newton_analytic(model, params, X), atol=1e-12, rtol=0)

    eigs = sl.eigvals(H_o).numpy()
    n_dom, n_bulk = R * R, R * (D_IN + D_OUT - 2 * R)
    assert n_bulk == 16
    dom_expected, bulk_expected = L * s ** (2 * (L - 1)), s ** (2 * (L - 1))
    assert np.allclose(eigs[:n_dom], dom_expected, rtol=1e-8, atol=0), eigs[: n_dom + 2]
    assert np.allclose(eigs[n_dom : n_dom + n_bulk], bulk_expected, rtol=1e-8, atol=0), eigs[n_dom : n_dom + n_bulk + 2]
    assert np.all(np.abs(eigs[n_dom + n_bulk :]) < 1e-9), eigs[n_dom + n_bulk : n_dom + n_bulk + 3]

    stats = sl.cluster_stats(eigs, R, D_IN, D_OUT)
    assert abs(stats["ratio_paper"] - L) < 1e-8 * L
    assert abs(stats["ratio_gap"] - L) < 1e-8 * L
    assert stats["rank"] == n_dom + n_bulk
    assert stats["largest_log_gap_index"] == n_dom - 1
    assert stats["split_n_dominant"] == n_dom and stats["split_n_bulk"] == n_bulk
    coh = sl.layer_coherence(H_o, model.layer_slices)
    assert abs(coh["coherence"] - L) < 1e-8 * L
    energy = sl.top_eigvec_layer_energy(H_o, model.layer_slices, k=n_dom)
    assert torch.allclose(energy.sum(1), torch.ones(n_dom, dtype=torch.float64))
    assert torch.allclose(energy, torch.full_like(energy, 1.0 / L), atol=1e-8)  # coherent across layers


def test_linear_balanced_tail_value_approximate():
    L = 3
    X, Y, model, params = _linear_setup(L, tail_value=0.05)
    eigs = sl.eigvals(sl.gauss_newton(model, params, X, Y)).numpy()
    stats = sl.cluster_stats(eigs, R, D_IN, D_OUT)
    print("\ntail_value=0.05, L=3 top eigenvalues:", np.round(eigs[:24], 6).tolist())
    print("tail_value=0.05 cluster_stats:", {k: v for k, v in stats.items() if not isinstance(v, (list, type(None)))})
    assert abs(stats["dominant_mean"] - L) < 1e-6  # in-subspace directions untouched
    assert abs(stats["bulk_mean"] - 1.0) < 0.01  # bulk perturbed at O(tail^2)
    assert abs(stats["ratio_paper"] - L) < 0.02 * L
    assert stats["rank"] > R * R + R * (D_IN + D_OUT - 2 * R)  # tail directions are no longer exactly zero


def test_nonuniform_balanced_init_ratio_theta_L():
    L = 3
    X = sl.make_whitened_inputs(N, D_IN, seed=0)
    Y, _ = sl.make_rank_r_targets(X, R, D_OUT, seed=1)
    model = sl.ModelSpec("linear", L, D_IN, H, D_OUT)
    s_layers = [1.0, 1.5, 0.8]
    params = model.init_params(seed=2, init="balanced_uniform", r=R, s=s_layers)
    eigs = sl.eigvals(sl.gauss_newton(model, params, X, Y)).numpy()
    prod_all = np.prod(np.square(s_layers))
    dom = sum(prod_all / s_l**2 for s_l in s_layers)  # sum_l prod_{m != l} s_m^2
    in_tail = prod_all / s_layers[0] ** 2  # first-layer input-tail dirs: prod_{m != 1} s_m^2
    out_tail = prod_all / s_layers[-1] ** 2  # last-layer output-tail dirs: prod_{m != L} s_m^2
    expected_bulk = np.sort([in_tail] * (R * (D_IN - R)) + [out_tail] * (R * (D_OUT - R)))
    assert np.allclose(eigs[: R * R], dom, rtol=1e-8)
    assert np.allclose(np.sort(eigs[R * R : R * R + 16]), expected_bulk, rtol=1e-8)
    assert np.all(np.abs(eigs[R * R + 16 :]) < 1e-9)
    ratio = dom / expected_bulk.mean()
    assert 1.0 < ratio < L and abs(sl.cluster_stats(eigs, R, D_IN, D_OUT)["ratio_paper"] - ratio) < 1e-8


# 3. decomposition and symmetry --------------------------------------------- #
@pytest.mark.parametrize("kind", ["linear", "relu", "smooth_snn"])
def test_full_hessian_equals_gn_plus_functional_and_symmetric(kind: str):
    X = sl.make_whitened_inputs(32, D_IN, seed=0)
    Y, _ = sl.make_rank_r_targets(X, R, D_OUT, seed=1)
    model = sl.ModelSpec(kind, 3, D_IN, H, D_OUT, T=4)
    params = model.init_params(seed=0)
    H_full = sl.full_hessian(model, params, X, Y, method="func")
    H_o = sl.gauss_newton(model, params, X, Y)
    H_f = sl.functional_part(H_full, H_o)
    assert torch.allclose(H_o + H_f, H_full, atol=1e-13, rtol=0)
    assert sl.asymmetry(H_full) < 1e-10
    assert torch.allclose(H_full, sl.full_hessian(model, params, X, Y, method="hvp"), atol=1e-12, rtol=0)
    assert max(sl.check_hessian_fd(model, params, X, Y, H=H_full)) < 1e-6
    # zero residual => H == H_o exactly (H_f vanishes with the residual)
    Y0 = model.forward(params, X)
    assert torch.allclose(sl.full_hessian(model, params, X, Y0), sl.gauss_newton(model, params, X, Y0), atol=1e-12, rtol=0)


# 4. snn: finite differences of the surrogate gradient ----------------------- #
@pytest.mark.parametrize("kwargs", [dict(), dict(reset="hard", surrogate="atan", detach_reset=True), dict(center_features=True)])
def test_snn_full_hessian_matches_finite_differences(kwargs: dict):
    X = sl.make_whitened_inputs(32, D_IN, seed=0)
    Y, _ = sl.make_rank_r_targets(X, R, D_OUT, seed=1)
    model = sl.ModelSpec("snn", 3, D_IN, H, D_OUT, T=4, **kwargs)
    params = model.init_params(seed=0)
    eps = 1e-5
    diag = sl.spike_diagnostics(model, params, X)
    assert diag["min_abs_v"] > 10 * eps  # no neuron crosses threshold within the FD step
    H_full = sl.full_hessian(model, params, X, Y, method="func")
    errs = sl.check_hessian_fd(model, params, X, Y, H=H_full, n_dirs=3, eps=eps)
    print(f"\nsnn {kwargs} FD rel errors: {errs}; asymmetry = {sl.asymmetry(H_full):.4f}")
    assert max(errs) < 1e-5
    assert sl.asymmetry(H_full) > 1e-3  # surrogate gradient is not a gradient field
    with pytest.raises(ValueError):
        sl.full_hessian(model, params, X, Y, method="hvp")


def test_snn_jvp_loop_fallback_agrees():
    X = sl.make_whitened_inputs(16, 6, seed=0)
    Y, _ = sl.make_rank_r_targets(X, 1, 3, seed=1)
    model = sl.ModelSpec("snn", 3, 6, 6, 3, T=3)
    params = model.init_params(seed=0)
    H_func = sl.full_hessian(model, params, X, Y, method="func")
    H_loop = sl.full_hessian(model, params, X, Y, method="jvp_loop")
    assert torch.allclose(H_func, H_loop, atol=1e-13, rtol=0)
    assert max(sl.check_hessian_fd(model, params, X, Y, H=H_loop)) < 1e-5


def test_snn_gauss_newton_jacobian_matches_batched_jacrev():
    X = sl.make_whitened_inputs(16, D_IN, seed=0)
    Y, _ = sl.make_rank_r_targets(X, R, D_OUT, seed=1)
    for kwargs in (dict(), dict(center_features=True)):
        model = sl.ModelSpec("snn", 3, D_IN, H, D_OUT, T=4, **kwargs)
        params = model.init_params(seed=0)
        H_o, extra = sl.gauss_newton(model, params, X, Y, return_blocks=True, return_J=True)
        J = extra["J"]
        assert J.shape == (16, D_OUT, model.n_params)
        vec = model.params_to_vec(params)
        J_batched = jacrev(lambda v: model.forward(model.vec_to_params(v), X))(vec)
        assert torch.allclose(J, J_batched, atol=1e-12, rtol=0)
        Jf = J.reshape(-1, model.n_params)
        assert torch.allclose(H_o, Jf.T @ Jf / 16, atol=1e-12, rtol=0)
        assert [b.shape[0] for b in extra["blocks"]] == [sl_.stop - sl_.start for sl_ in model.layer_slices]
    # cross-entropy Gauss-Newton is symmetric PSD
    Yc = torch.nn.functional.one_hot(torch.arange(16) % D_OUT, D_OUT).double()
    H_ce = sl.gauss_newton(model, params, X, Yc, loss="ce")
    assert sl.asymmetry(H_ce) < 1e-12 and float(sl.eigvals(H_ce)[-1]) > -1e-12


# 5. snn forward ------------------------------------------------------------- #
def test_snn_spikes_binary_gradient_nonzero_diagnostics():
    X = sl.make_whitened_inputs(32, D_IN, seed=0)
    Y, _ = sl.make_rank_r_targets(X, R, D_OUT, seed=1)
    model = sl.ModelSpec("snn", 3, D_IN, H, D_OUT, T=8)
    params = model.init_params(seed=0)
    spikes, _ = model._spiking_stack(params, X)
    for s in spikes:
        assert set(torch.unique(s).tolist()) <= {0.0, 1.0}
    g = sl.surrogate_gradient(model, params, X, Y)
    assert float(torch.linalg.norm(g)) > 1e-6 and torch.isfinite(g).all()
    diag = sl.spike_diagnostics(model, params, X)
    print("\nsnn default-init spike diagnostics:", diag)
    assert 0.0 < diag["ever_spike_frac"] <= 1.0
    for key in ("mean_rate", "window_frac"):
        assert 0.0 < diag[key] < 1.0
    fs = sl.feature_second_moment_stats(model, params, X)
    assert fs["mu_sq_norm"] > 0 and len(fs["cov_top3"]) == 3 and 0 <= fs["frac_var_along_mu"] <= 1
    # centred readout: output of the centred model equals W_L (a - mean a)
    model_c = sl.ModelSpec("snn", 3, D_IN, H, D_OUT, T=8, center_features=True)
    a = model.hidden_features(params, X)
    assert torch.allclose(model_c.forward(params, X), (a - a.mean(0)) @ params["W3"].T)


def test_surrogates_are_cdfs_with_correct_slope():
    v = torch.linspace(-3, 3, 601, dtype=torch.float64).requires_grad_(True)
    for name in sl.models.SURROGATES:
        sg = sl.surrogate_fn(name, 4.0)
        y = sg(v)
        (dy,) = torch.autograd.grad(y.sum(), v)
        assert torch.all(dy >= -1e-12)
        assert abs(float(sg(torch.tensor(0.0, dtype=torch.float64))) - 0.5) < 1e-12
        assert abs(float(dy[300]) - {"sigmoid": 1.0, "atan": 2.0, "fast_sigmoid": 2.0, "triangle": 4.0}[name]) < 1e-6


# 6. training ----------------------------------------------------------------- #
def test_gd_train_linear_monotone_with_callbacks():
    X, Y, model, params = _linear_setup(3)
    cb = sl.hessian_summary_callback(model, X, Y, "mse", R, D_IN, D_OUT)
    hist = sl.gd_train(model, params, X, Y, lr=0.05, steps=40, log_every=20, callbacks=[cb])
    losses = np.array(hist["loss"])
    assert np.all(np.diff(losses) <= 1e-12)
    assert losses[-1] < 0.5 * losses[0]
    assert hist["logged_steps"] == [0, 20, 40]
    summary = hist["summaries"][-1]
    assert summary["H_f_norm"] < hist["summaries"][0]["H_f_norm"]  # residual term decays under GD
    coh = summary["coherence_H_o"]["coherence"]
    assert 1.0 - 1e-8 <= coh <= 3 + 1e-8  # provable bounds for a PSD block matrix with L = 3 blocks
    assert abs(hist["summaries"][0]["coherence_H_o"]["coherence"] - 3) < 1e-8  # exactly L at init
    hist_adam = sl.gd_train(model, params, X, Y, lr=0.01, steps=20, optimizer="adam")
    assert hist_adam["loss"][-1] < hist_adam["loss"][0]


def test_top_k_matches_dense():
    X, Y, model, params = _linear_setup(5)
    H_o = sl.gauss_newton(model, params, X, Y)
    big = torch.block_diag(H_o, H_o, H_o)  # P = 864 > 512 triggers the eigsh path
    evals, evecs = sl.top_k(big, 6)
    assert torch.allclose(evals, sl.eigvals(big)[:6], atol=1e-8)
    assert torch.allclose(big @ evecs, evecs * evals, atol=1e-8)
