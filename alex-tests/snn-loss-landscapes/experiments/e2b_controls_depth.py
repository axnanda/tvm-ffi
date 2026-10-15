import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""e2b_controls_depth: ANN controls with the SAME protocol as the SNN depth sweep (e2a_snn_depth).

Kinds: "relu" (kaiming init), "smooth_snn" (sigmoid(k v) forward, k=4, default kaiming init) and
"linear" (init="gaussian", gain 1).  Target settings A (r=1, d_out=4) / B (r=2, d_out=6),
L in {2,3,4,5,6,8}, seeds {0,1}.
Per run: full H + H_o at init; Adam lr=5e-3 for 800 steps (linear: GD, lr = min(0.1, 1/lambda_max(H_init)),
800 steps; lr 0.1 diverges for deep gaussian linear nets, see e1); H_o every 100 steps; full H at
steps 200 and 800.  Per checkpoint: eigenvalues of H_o and sym(H), cluster statistics, layer coherence,
top-eigenvector layer energies, readout-feature second moments, spike diagnostics (smooth_snn) /
activation sparsity (all kinds), ||H_f||/||H_o||, asymmetry, wall times.
Plus (seed 0, setting B, L in {2,4,8}) a center_features=True variant for relu and smooth_snn (P2 test).

Usage:  python3 experiments/e2b_controls_depth.py            # full run (results/e2b_controls_depth)
        python3 experiments/e2b_controls_depth.py --smoke    # reduced run (results/e2b_controls_depth_smoke)
        python3 experiments/e2b_controls_depth.py --figures  # only regenerate figures + analysis
By default the script RESUMES: runs whose (kind, setting, L, seed, centered) are already in
results/<exp>/summary.json (and whose raw npz exists) are skipped, new runs are appended to the same
summary.json; --fresh starts over.  --res-dir overrides the results directory (resume-logic test).
"""
import argparse
import json
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "e2b_controls_depth"
SNN_REF_DIR = os.path.join(ROOT, "results", "e2a_snn_depth")  # read-only reference (SNN sweep, same protocol)

N, D_IN, WIDTH = 256, 16, 16
SETTINGS = {"A": (1, 4), "B": (2, 6)}
LS = [2, 3, 4, 5, 6, 8]
KINDS = ["relu", "smooth_snn", "linear"]
LR_ADAM, LR_GD_CAP, STEPS, HO_EVERY = 5e-3, 0.1, 800, 100
FULL_H_STEPS = (0, 200, 800)
K_TOP = 16  # rows of the stored top-eigenvector layer-energy matrix (superset of max(r^2, d_out))
BUDGET_MIN = 66.0  # wall-clock budget for the full run (minutes); optional tiers are skipped beyond it
# per-run cost estimates (seconds, 1 thread, contended 4-CPU machine) from the smoke timing probes:
# full H at L=8: relu 1.6 s, linear 1.2 s, smooth_snn 61 s; eigh(1888) 1.5 s
EST_RUN_SEC = {"relu": {2: 4, 3: 6, 4: 10, 5: 15, 6: 20, 8: 35},
               "linear": {2: 4, 3: 6, 4: 10, 5: 15, 6: 20, 8: 35},
               "smooth_snn": {2: 20, 3: 45, 4: 80, 5: 125, 6: 175, 8: 290}}
KIND_COL = {"relu": "tab:green", "smooth_snn": "tab:purple", "linear": "tab:brown", "snn": "tab:red"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_py(o):
    """Recursively convert tensors / numpy / nan-safe scalars to JSON-serialisable python objects."""
    if isinstance(o, torch.Tensor):
        return to_py(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return [to_py(x) for x in o.tolist()] if o.ndim > 0 else to_py(o.item())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, slice):
        return [o.start, o.stop]
    return o


def make_spec(kind: str, L: int, d_out: int, centered: bool = False):
    if kind == "smooth_snn":
        return sl.ModelSpec("smooth_snn", L, D_IN, WIDTH, d_out, T=8, beta=0.9, theta=1.0, k=4.0, surrogate="sigmoid",
                            reset="soft", detach_reset=False, readout="rate", center_features=centered)
    if kind == "relu":
        return sl.ModelSpec("relu", L, D_IN, WIDTH, d_out, center_features=centered)
    if kind == "linear":
        if centered:
            raise ValueError("center_features is not defined for the linear kind")
        return sl.ModelSpec("linear", L, D_IN, WIDTH, d_out)
    raise ValueError(kind)


def make_init(spec, kind: str, seed: int):
    if kind == "linear":
        return spec.init_params(seed, init="gaussian", gain=1.0), "gaussian(gain=1)"
    return spec.init_params(seed), "kaiming(gain=sqrt2)"  # library default for relu / smooth_snn


def readout_features(spec, params, X):
    """Features actually multiplied by W_L (centred if spec.center_features)."""
    with torch.no_grad():
        a = spec.hidden_features(params, X)
        if spec.center_features:
            a = a - a.mean(0)
    return a


def feature_stats_from(a: torch.Tensor) -> Dict[str, float]:
    """Same statistics as sl.feature_second_moment_stats but for an explicit feature matrix."""
    n = a.shape[0]
    mu = a.mean(0)
    ac = a - mu
    cov = ac.T @ ac / n
    M = a.T @ a / n
    cov_eigs = sl.eigh(cov)[0]
    M_eigs = sl.eigh(M)[0]
    mu_sq = float(mu @ mu)
    tr_cov, tr_M = float(torch.trace(cov)), float(torch.trace(M))
    return {
        "mu_sq_norm": mu_sq,
        "cov_top3": [float(x) for x in cov_eigs[:3]],
        "second_moment_top3": [float(x) for x in M_eigs[:3]],
        "frac_var_along_mu": float(mu @ cov @ mu) / (mu_sq * tr_cov) if mu_sq > 0 and tr_cov > 0 else float("nan"),
        "frac_second_moment_along_mu": float(mu @ M @ mu) / (mu_sq * tr_M) if mu_sq > 0 and tr_M > 0 else float("nan"),
        "mean_frac_of_second_moment": mu_sq / tr_M if tr_M > 0 else float("nan"),
        "trace_cov": tr_cov,
        "trace_second_moment": tr_M,
    }


def centered_update_field_jacobian(spec, params, X, Y, chunk_size: int = 128) -> torch.Tensor:
    """True Jacobian ``H[i, j] = d g_i / d theta_j`` of the training gradient field of a
    ``center_features=True`` model (work-around, see run.log / report).

    The library subtracts the *detached* batch mean ``m(theta)`` of the readout features, so the
    training gradient is ``g(theta) = grad_theta L(theta, m)|_{m = m(theta)}`` (a stop-gradient
    field, not the gradient of any potential).  ``sl.full_hessian`` differentiates with ``m`` frozen
    and returns the symmetric ``grad^2_theta L(theta, m)``; the true Jacobian of ``g`` also contains
    ``grad^2_{theta m} L . dm/dtheta`` (non-zero for relu / smooth_snn, zero a.e. for snn) and is
    in general NOT symmetric.  Here ``m`` is composed into ``g`` without detaching, so forward-mode
    differentiation of ``g`` yields the full Jacobian (validated against finite differences)."""
    from torch.func import grad as fgrad, jvp as fjvp, vmap as fvmap
    X = X.to(spec.dtype)
    n = X.shape[0]
    WL = f"W{spec.L}"
    assert spec.center_features and not spec.bias

    def loss_explicit(vec, m):
        p = spec.vec_to_params(vec)
        a = spec.hidden_features(p, X)
        yhat = (a - m) @ p[WL].T
        return 0.5 * ((yhat - Y) ** 2).sum() / n

    g_theta = fgrad(loss_explicit, argnums=0)

    def g_field(vec):
        m = spec.hidden_features(spec.vec_to_params(vec), X).mean(0)  # NOT detached here
        return g_theta(vec, m)

    vec = spec.params_to_vec(params).detach()
    P = vec.numel()
    eye = torch.eye(P, dtype=vec.dtype)

    def column(d):
        return fjvp(g_field, (vec,), (d,))[1]

    cols = [fvmap(column)(eye[s: s + chunk_size]) for s in range(0, P, chunk_size)]
    return torch.cat(cols, 0).T


def activity_diagnostics(spec, params, X) -> Dict[str, object]:
    """Sparsity / activity of the hidden layers (all kinds).  relu: per-layer fraction of active
    (> 0) units; smooth_snn: per-layer mean sigmoid 'rate' and fraction > 0.5; linear: trivially dense.
    Also sparsity of the readout features a (fraction of exact zeros, fraction < 1e-3 * max)."""
    out: Dict[str, object] = {}
    with torch.no_grad():
        X = X.to(spec.dtype)
        if spec.kind == "relu":
            h = X
            active, means = [], []
            for l in range(1, spec.L):
                h = torch.relu(spec._affine(params, h, l))
                active.append(float((h > 0).to(torch.float64).mean()))
                means.append(float(h.mean()))
            out["per_layer_active_frac"] = active
            out["per_layer_mean"] = means
            out["mean_active_frac"] = float(np.mean(active)) if active else float("nan")
        elif spec.kind == "smooth_snn":
            spikes_all, _ = spec._spiking_stack(params, X)
            out["per_layer_active_frac"] = [float((s > 0.5).to(torch.float64).mean()) for s in spikes_all]
            out["per_layer_mean"] = [float(s.mean()) for s in spikes_all]
            out["mean_active_frac"] = float(np.mean(out["per_layer_active_frac"]))
        else:
            out["per_layer_active_frac"] = [1.0] * (spec.L - 1)
            out["per_layer_mean"] = []
            out["mean_active_frac"] = 1.0
        a = spec.hidden_features(params, X)
        amax = float(a.abs().max())
        out["readout_feat_exact_zero_frac"] = float((a == 0).to(torch.float64).mean())
        out["readout_feat_small_frac"] = float((a.abs() < 1e-3 * amax).to(torch.float64).mean()) if amax > 0 else 1.0
        out["readout_feat_mean"] = float(a.mean())
        out["readout_feat_std"] = float(a.std())
    return out


def spectrum_summary(H: torch.Tensor, spec, r: int, d_in: int, d_out: int, tag: str):
    """Everything the protocol asks for one matrix (one full eigh); returns (dict, eigs ndarray).
    Same formulas as sl.layer_coherence / sl.top_eigvec_layer_energy (cross-checked in validate_once)."""
    Hs = sl.symmetrize(H)
    eigs, evecs = sl.eigh(Hs)
    e = eigs.numpy()
    cs = sl.cluster_stats(e, r, d_in, d_out)
    n = len(e)
    slices = spec.layer_slices
    lam_max = float(eigs[0])
    top = evecs[:, 0]
    block_lams = [float(sl.eigh(Hs[s_, s_])[0][0]) for s_ in slices]
    energy = [float((top[s_] ** 2).sum()) for s_ in slices]
    fro = [float(torch.linalg.norm(Hs[s_, s_])) for s_ in slices]
    align = np.full((len(slices), len(slices)), np.nan)
    for i, si in enumerate(slices):
        for j, sj in enumerate(slices):
            den = fro[i] * fro[j]
            if den > 0:
                align[i, j] = float(torch.linalg.norm(Hs[si, sj])) ** 2 / den
    V = evecs[:, :K_TOP]
    topk = torch.stack([(V[s_] ** 2).sum(0) for s_ in slices], dim=1)
    topk = (topk / (V ** 2).sum(0).unsqueeze(1)).numpy()  # (K_TOP, L)
    out = {
        f"cluster_{tag}": cs,
        f"coh_{tag}": {
            "lambda_max": lam_max,
            "lambda_max_blocks": block_lams,
            "coherence": lam_max / max(block_lams) if max(block_lams) > 0 else float("nan"),
            "sum_block_lams_over_lambda_max": sum(block_lams) / lam_max if lam_max > 0 else float("nan"),
            "top_eigvec_layer_energy": energy,
            "block_alignment": align,
        },
        f"topk_energy_{tag}": topk,
        f"zero_frac_{tag}": cs["zero_count"] / n,
        f"neg_frac_{tag}": cs["negative_count"] / n,
        f"dominant_count_{tag}": cs["split_n_dominant"],
        f"trace_{tag}": float(torch.trace(Hs)),
        f"fro_{tag}": float(torch.linalg.norm(H)),
        f"lambda_min_{tag}": float(eigs[-1]),
        f"top20_{tag}": [float(x) for x in e[:20]],
    }
    return out, e


def make_callback(spec, X, Y, r, d_in, d_out, full_steps, raw: Dict[str, np.ndarray], verbose=True):
    def cb(step, params, g, loss_value):
        t0 = time.time()
        rec = {"step": int(step), "loss": float(loss_value), "grad_norm": float(torch.linalg.norm(g)),
               "P": int(spec.n_params)}
        if not math.isfinite(loss_value) or loss_value > 1e6:
            rec["diverged"] = True
            print(f"    step {step:4d} loss={loss_value:.3e} DIVERGED -> no Hessian", flush=True)
            return {"rec": rec}
        H_o = sl.gauss_newton(spec, params, X, Y)
        s_o, e_o = spectrum_summary(H_o, spec, r, d_in, d_out, "Ho")
        rec.update(s_o)
        rec["asym_Ho"] = sl.asymmetry(H_o)
        raw[f"eig_Ho_step{step}"] = e_o
        rec["features"] = sl.feature_second_moment_stats(spec, params, X)  # uncentred hidden features
        if spec.center_features:
            rec["features_centered"] = feature_stats_from(readout_features(spec, params, X))
        rec["activity"] = activity_diagnostics(spec, params, X)
        rec["spikes"] = sl.spike_diagnostics(spec, params, X) if spec.is_spiking else None
        rec["wall_Ho"] = time.time() - t0
        if step in full_steps:
            t1 = time.time()
            H = sl.full_hessian(spec, params, X, Y)
            H_f = sl.functional_part(H, H_o)
            s_h, e_h = spectrum_summary(H, spec, r, d_in, d_out, "H")
            rec.update(s_h)
            rec["asymmetry_H"] = sl.asymmetry(H)
            rec["Hf_norm"] = float(torch.linalg.norm(H_f))
            rec["Ho_norm"] = float(torch.linalg.norm(H_o))
            rec["H_norm"] = float(torch.linalg.norm(H))
            rec["Hf_over_Ho"] = rec["Hf_norm"] / rec["Ho_norm"] if rec["Ho_norm"] > 0 else float("nan")
            rec["asymmetry_Hf"] = sl.asymmetry(H_f)
            rec["Hf_sym_fro_over_Ho"] = float(torch.linalg.norm(sl.symmetrize(H_f))) / rec["Ho_norm"]
            raw[f"eig_H_step{step}"] = e_h
            rec["wall_H"] = time.time() - t1
            if spec.center_features:
                # library H above = frozen-mean Hessian; also record the true Jacobian of the update field
                t2 = time.time()
                Ht = centered_update_field_jacobian(spec, params, X, Y)
                s_t, e_t = spectrum_summary(Ht, spec, r, d_in, d_out, "Htrue")
                rec.update(s_t)
                rec["asymmetry_Htrue"] = sl.asymmetry(Ht)
                rec["Htrue_minus_Hlib_over_Hlib"] = float(torch.linalg.norm(Ht - H) / torch.linalg.norm(H))
                rec["Hf_true_over_Ho"] = float(torch.linalg.norm(Ht - H_o)) / rec["Ho_norm"]
                raw[f"eig_Htrue_step{step}"] = e_t
                rec["wall_Htrue"] = time.time() - t2
        if verbose:
            act = rec["spikes"]["mean_rate"] if rec["spikes"] else rec["activity"]["mean_active_frac"]
            msg = (f"    step {step:4d} loss={loss_value:.4e} lam_max(Ho)={rec['coh_Ho']['lambda_max']:.3e} "
                   f"coh={rec['coh_Ho']['coherence']:.2f} ndom={rec['dominant_count_Ho']} "
                   f"ratio_split={rec['cluster_Ho']['ratio_split']:.2f} act={act:.3f} "
                   f"wall={time.time() - t0:.1f}s")
            if step in full_steps:
                msg += f" | asym(H)={rec['asymmetry_H']:.1e} |Hf|/|Ho|={rec['Hf_over_Ho']:.3f} wall_H={rec['wall_H']:.1f}s"
            print(msg, flush=True)
        return {"rec": rec}
    return cb


def run_one(kind: str, setting: str, L: int, seed: int, centered: bool, steps: int, ho_every: int, full_steps,
            res_dir: str, verbose=True) -> Dict[str, object]:
    r, d_out = SETTINGS[setting]
    X = sl.make_whitened_inputs(N, D_IN, seed)
    Y, Wstar = sl.make_rank_r_targets(X, r, d_out, seed)
    # sanity: data matches (r, d_in, d_out) passed to cluster_stats
    assert X.shape == (N, D_IN) and Y.shape == (N, d_out)
    assert int(torch.linalg.matrix_rank(Y)) == r, "target rank != r"
    assert float((X.T @ X / N - torch.eye(D_IN, dtype=X.dtype)).abs().max()) < 1e-8, "inputs not whitened"
    spec = make_spec(kind, L, d_out, centered)
    params, init_desc = make_init(spec, kind, seed)
    if kind == "linear":
        optimizer = "gd"
        lam0 = float(sl.eigvals(sl.full_hessian(spec, params, X, Y))[0])
        lr = min(LR_GD_CAP, 1.0 / lam0)
        lr_desc = f"GD lr=min({LR_GD_CAP}, 1/lambda_max(H_init)={1.0 / lam0:.4f}) -> {lr:.4f}"
    else:
        optimizer, lr, lam0 = "adam", LR_ADAM, float("nan")
        lr_desc = f"Adam lr={lr}"
    name = f"{kind}_{setting}_L{L}_seed{seed}" + ("_centered" if centered else "")
    info: Dict[str, object] = {"kind": kind, "setting": setting, "r": r, "d_in": D_IN, "d_out": d_out, "L": L,
                               "seed": seed, "centered": centered, "init": init_desc, "P": spec.n_params,
                               "steps": steps, "optimizer": optimizer, "lr": lr, "lr_desc": lr_desc,
                               "lambda_max_H_init": lam0, "spec": repr(spec), "n": N, "width": WIDTH, "name": name}
    print(f"== run {name}: P={spec.n_params} r={r} d_out={d_out} {init_desc} {lr_desc}", flush=True)
    raw: Dict[str, np.ndarray] = {}
    cb = make_callback(spec, X, Y, r, D_IN, d_out, full_steps, raw, verbose=verbose)
    t0 = time.time()
    hist = sl.gd_train(spec, params, X, Y, lr=lr, steps=steps, loss="mse", log_every=ho_every,
                       callbacks=[cb], optimizer=optimizer)
    wall = time.time() - t0
    checkpoints = [s["rec"] for s in hist["summaries"]]
    loss_curve = np.asarray(hist["loss"], dtype=np.float64)
    gn_curve = np.asarray(hist["grad_norm"], dtype=np.float64)
    info.update({
        "wall_sec": wall, "logged_steps": hist["logged_steps"],
        "loss_init": float(loss_curve[0]), "loss_final": float(loss_curve[-1]),
        "loss_min": float(np.nanmin(loss_curve)),
        "diverged": bool(not np.all(np.isfinite(loss_curve)) or loss_curve[-1] > 1e6),
        "loss_curve_every10": [float(x) for x in loss_curve[::10]] + [float(loss_curve[-1])],
        "checkpoints": checkpoints,
    })
    raw["loss_curve"] = loss_curve
    raw["grad_norm_curve"] = gn_curve
    raw["logged_steps"] = np.asarray(hist["logged_steps"])
    raw["Wstar"] = Wstar.numpy()
    os.makedirs(os.path.join(res_dir, "raw"), exist_ok=True)
    np.savez(os.path.join(res_dir, "raw", f"{name}.npz"), **raw)
    print(f"   done {name}: loss {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f}, wall {wall:.0f}s", flush=True)
    return to_py(info)


# --------------------------------------------------------------------------- #
# smoke-test validations
# --------------------------------------------------------------------------- #
def validate_once(kind="relu", setting="A", L=3, seed=0, centered=False, print_dict=True):
    """Print one full stats dict and cross-check the local helpers against the library functions."""
    r, d_out = SETTINGS[setting]
    X = sl.make_whitened_inputs(N, D_IN, seed)
    Y, _ = sl.make_rank_r_targets(X, r, d_out, seed)
    spec = make_spec(kind, L, d_out, centered)
    p, _ = make_init(spec, kind, seed)
    H_o = sl.gauss_newton(spec, p, X, Y)
    H = sl.full_hessian(spec, p, X, Y)
    s_o, e_o = spectrum_summary(H_o, spec, r, D_IN, d_out, "Ho")
    s_h, e_h = spectrum_summary(H, spec, r, D_IN, d_out, "H")
    ref_cs = sl.cluster_stats(sl.eigvals(H_o), r, D_IN, d_out)
    ref_coh = sl.layer_coherence(H_o, spec.layer_slices)
    ref_topk = sl.top_eigvec_layer_energy(H_o, spec.layer_slices, k=max(r * r, d_out)).numpy()
    for k_ in ref_cs:
        a, b = ref_cs[k_], s_o["cluster_Ho"][k_]
        assert (a == b) or (isinstance(a, float) and isinstance(b, float) and (math.isnan(a) and math.isnan(b) or abs(a - b) <= 1e-9 * max(1, abs(a)))), (k_, a, b)
    assert abs(ref_coh["coherence"] - s_o["coh_Ho"]["coherence"]) < 1e-9
    assert np.allclose(ref_coh["lambda_max_blocks"], s_o["coh_Ho"]["lambda_max_blocks"])
    assert np.allclose(ref_coh["top_eigvec_layer_energy"], s_o["coh_Ho"]["top_eigvec_layer_energy"], atol=1e-8)
    assert np.allclose(ref_coh["block_alignment"].numpy(), s_o["coh_Ho"]["block_alignment"], equal_nan=True)
    assert np.allclose(ref_topk, s_o["topk_energy_Ho"][: ref_topk.shape[0]], atol=1e-8)
    cs = s_o["cluster_Ho"]
    assert cs["n_eigs"] == spec.n_params and cs["n_dominant_expected"] == r * r
    assert cs["n_bulk_expected"] == r * (D_IN + d_out - 2 * r)
    assert cs["largest_log_gap_index"] + 1 == cs["split_n_dominant"] == s_o["dominant_count_Ho"]
    assert abs(sum(e_o) - float(torch.trace(H_o))) < 1e-8 * abs(float(torch.trace(H_o)))
    assert e_o[-1] > -1e-10 * e_o[0], "H_o not PSD?"
    assert sl.asymmetry(H_o) < 1e-12
    asym = sl.asymmetry(H)
    assert asym < 1e-8, f"H of kind {kind} should be symmetric (asym={asym})"
    feats = sl.feature_second_moment_stats(spec, p, X)
    # last-layer block of H_o is I_{d_out} kron M (M = a^T a / n, a = readout features): lambda_max = lambda_max(M)
    if centered:
        fc = feature_stats_from(readout_features(spec, p, X))
        assert fc["mu_sq_norm"] < 1e-20
        assert abs(s_o["coh_Ho"]["lambda_max_blocks"][-1] - feats["cov_top3"][0]) < 1e-8 * feats["cov_top3"][0]
        assert abs(fc["second_moment_top3"][0] - feats["cov_top3"][0]) < 1e-8 * feats["cov_top3"][0]
    else:
        assert abs(s_o["coh_Ho"]["lambda_max_blocks"][-1] - feats["second_moment_top3"][0]) < 1e-8 * feats["second_moment_top3"][0]
    if kind == "linear":
        H_ref = sl.linear_gauss_newton_analytic(spec, p, X)
        assert torch.allclose(H_o, H_ref, atol=1e-10, rtol=1e-10), "H_o != analytic linear GN"
    fd = sl.check_hessian_fd(spec, p, X, Y, H=H, n_dirs=2, eps=1e-6)
    print(f"[validate {kind} centered={centered}] finite-difference rel. errors of library H:", fd)
    if centered:
        # library H = frozen-mean Hessian (symmetric) != Jacobian of the stop-gradient field g; work-around:
        Ht = centered_update_field_jacobian(spec, p, X, Y)
        g_lib = sl.surrogate_gradient(spec, p, X, Y)
        fd_t = sl.check_hessian_fd(spec, p, X, Y, H=Ht, n_dirs=2, eps=1e-6)
        print(f"[validate {kind} centered={centered}] finite-difference rel. errors of TRUE dg/dtheta:", fd_t,
              f"asym(Htrue)={sl.asymmetry(Ht):.3f} ||Htrue-Hlib||/||Hlib||={float(torch.linalg.norm(Ht - H) / torch.linalg.norm(H)):.3f}")
        assert max(fd_t) < 1e-4, fd_t
        assert max(fd) > 1e-3, "library H unexpectedly matches FD for a centred model (work-around not needed?)"
        # the training gradient used by gd_train equals J^T r / n with the frozen-mean per-sample Jacobian
        _, extra = sl.gauss_newton(spec, p, X, Y, return_J=True)
        resid = (spec.forward(p, X) - Y)
        g_gn = torch.einsum("iap,ia->p", extra["J"], resid) / N
        assert float(torch.linalg.norm(g_gn - g_lib) / torch.linalg.norm(g_lib)) < 1e-10
        print(f"[validate {kind} centered={centered}] training gradient == J_frozen^T r / n: OK (H_o is consistent with g)")
    else:
        assert max(fd) < 1e-4, fd
    print(f"[validate {kind} centered={centered}] all library cross-checks passed for", spec)
    rec = {"step": 0, "P": spec.n_params}
    rec.update(s_o)
    rec.update(s_h)
    rec["asymmetry_H"] = asym
    rec["Hf_over_Ho"] = float(torch.linalg.norm(H - H_o) / torch.linalg.norm(H_o))
    rec["features"] = feats
    rec["activity"] = activity_diagnostics(spec, p, X)
    rec["spikes"] = sl.spike_diagnostics(spec, p, X) if spec.is_spiking else None
    if print_dict:
        print(f"[validate] ONE FULL STATS DICT ({kind}, init, setting {setting}, L={L}, seed={seed}, centered={centered}):")
        print(json.dumps(to_py(rec), indent=1, default=str))
    print(f"[validate {kind}] top-12 eigs H_o:", np.round(e_o[:12], 4))
    print(f"[validate {kind}] top-12 eigs H  :", np.round(e_h[:12], 4), " min:", e_h[-1])


def time_hessians(kind, setting="B", L=8, seed=0):
    r, d_out = SETTINGS[setting]
    X = sl.make_whitened_inputs(N, D_IN, seed)
    Y, _ = sl.make_rank_r_targets(X, r, d_out, seed)
    spec = make_spec(kind, L, d_out)
    p, _ = make_init(spec, kind, seed)
    t0 = time.time(); H_o = sl.gauss_newton(spec, p, X, Y); t1 = time.time()
    H = sl.full_hessian(spec, p, X, Y); t2 = time.time()
    sl.eigh(sl.symmetrize(H)); t3 = time.time()
    print(f"[timing {kind} L={L} P={spec.n_params}] H_o {t1 - t0:.1f}s, full H {t2 - t1:.1f}s, eigh {t3 - t2:.1f}s", flush=True)


# --------------------------------------------------------------------------- #
# aggregation / figures
# --------------------------------------------------------------------------- #
def load_summary(res_dir):
    with open(os.path.join(res_dir, "summary.json")) as f:
        return json.load(f)


def load_snn_reference():
    """SNN kaiming runs of e2a_snn_depth (same protocol), tagged kind='snn'; [] if unavailable."""
    try:
        runs = load_summary(SNN_REF_DIR)
    except Exception as e:  # noqa: BLE001
        print("no SNN reference available:", e)
        return []
    out = []
    for r_ in runs:
        if r_.get("init") == "kaiming":
            r_ = dict(r_)
            r_["kind"] = "snn"
            r_["centered"] = False
            out.append(r_)
    return out


def ckpt(run, step):
    for c in run["checkpoints"]:
        if c["step"] == step:
            return c
    return None


def get(c, path):
    """Nested lookup 'a.b.c' (integer path elements index lists; -1 allowed) with None -> nan."""
    for p_ in path.split("."):
        if c is None:
            return np.nan
        if isinstance(c, dict):
            c = c.get(p_)
        elif isinstance(c, list):
            try:
                c = c[int(p_)]
            except (ValueError, IndexError):
                c = None
        else:
            c = None
    return np.nan if c is None else c


def sel(runs, kind, setting, L=None, centered=False):
    return [r_ for r_ in runs if r_["kind"] == kind and r_["setting"] == setting and r_.get("centered", False) == centered
            and (L is None or r_["L"] == L)]


def agg(runs, kind, setting, step, path, Ls=LS, centered=False):
    """mean, std, n over seeds per L of a nested scalar."""
    means, stds, ns, per = [], [], [], {}
    for L in Ls:
        vals = [get(ckpt(r_, step), path) for r_ in sel(runs, kind, setting, L, centered) if ckpt(r_, step) is not None]
        vals = np.asarray([v for v in vals if v is not None], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        per[L] = vals.tolist()
        means.append(vals.mean() if len(vals) else np.nan)
        stds.append(vals.std() if len(vals) > 1 else 0.0)
        ns.append(len(vals))
    return np.asarray(means), np.asarray(stds), ns, per


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x[m], y[m], 1)
    pred = a * x[m] + b
    ss_res = ((y[m] - pred) ** 2).sum()
    ss_tot = ((y[m] - y[m].mean()) ** 2).sum()
    return float(a), float(b), float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def make_figures(res_dir, fig_dir, exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_summary(res_dir)
    snn = load_snn_reference()
    allruns = runs + snn
    os.makedirs(fig_dir, exist_ok=True)
    main = [r_ for r_ in runs if not r_.get("centered", False)]
    kinds = [k for k in KINDS if any(r_["kind"] == k for r_ in main)]
    kinds_ref = kinds + (["snn"] if snn else [])
    Ls_present = sorted({r_["L"] for r_ in main})
    steps_full = sorted({c["step"] for r_ in main for c in r_["checkpoints"] if "cluster_H" in c})
    steps_all = sorted({c["step"] for r_ in main for c in r_["checkpoints"]})
    s_init, s_mid, s_end = steps_full[0], steps_full[len(steps_full) // 2], steps_full[-1]
    analysis: Dict[str, object] = {"Ls": Ls_present, "full_H_steps": steps_full, "Ho_steps": steps_all, "kinds": kinds,
                                   "snn_reference_runs": len(snn)}
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(Ls_present)))
    ckcol = {s_init: "tab:blue", s_mid: "tab:orange", s_end: "tab:red"}
    kname = {"relu": "ReLU (kaiming)", "smooth_snn": "smooth SNN (sigmoid k=4)", "linear": "linear (gaussian g=1)", "snn": "SNN (e2a ref)"}

    # (a) spectra per kind x setting ---------------------------------------------------
    for kind in kinds:
        for setting, (r, d_out) in SETTINGS.items():
            fig, axes = plt.subplots(2, len(Ls_present), figsize=(3.2 * len(Ls_present), 6.4), squeeze=False)
            for j, L in enumerate(Ls_present):
                for i, tag in enumerate(["Ho", "H"]):
                    ax = axes[i, j]
                    for r_ in sel(main, kind, setting, L):
                        z = np.load(os.path.join(res_dir, "raw", r_["name"] + ".npz"))
                        for st in (s_init, s_end):
                            key = f"eig_{tag}_step{st}"
                            if key not in z:
                                continue
                            e = z[key]
                            idx = np.arange(1, len(e) + 1)
                            pos = e > 0
                            ax.plot(idx[pos], e[pos], color=ckcol[st], lw=0.7, alpha=0.7,
                                    label=f"step {st}" if r_["seed"] == 0 else None)
                            if (~pos).any():
                                ax.plot(idx[~pos], np.abs(e[~pos]), color=ckcol[st], lw=0.7, alpha=0.5, ls=":",
                                        label=f"|neg| step {st}" if r_["seed"] == 0 else None)
                    ax.axvline(r * r, color="k", ls="--", lw=0.8, label=f"r^2={r*r}")
                    ax.axvline(d_out, color="gray", ls="-.", lw=0.8, label=f"d_out={d_out}")
                    ax.axvline(r * r + r * (D_IN + d_out - 2 * r), color="gray", ls=":", lw=0.8, label="r^2+bulk")
                    ax.set_xscale("log"); ax.set_yscale("log")
                    ax.set_title(f"{'H_o' if tag == 'Ho' else 'sym(H)'}  L={L}", fontsize=9)
                    ax.set_xlabel("index (1-based)")
                    if j == 0:
                        ax.set_ylabel("eigenvalue (|.| dotted for <0)")
                    if i == 0 and j == 0:
                        ax.legend(fontsize=6)
            fig.suptitle(f"{exp}: sorted spectra, {kname[kind]}, setting {setting} (r={r}, d_out={d_out}); thin lines = seeds", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"{exp}_spectra_{kind}_{setting}.png"), dpi=120)
            plt.close(fig)

    # (b) ratios + coherence vs L, per kind, Ho and H ------------------------------------
    fits = {}
    for kind in kinds:
        for tag in ["Ho", "H"]:
            metrics = [(f"cluster_{tag}.ratio_paper", "ratio_paper (mean r^2 / mean bulk)"),
                       (f"cluster_{tag}.ratio_gap", "ratio_gap (eig[r^2-1]/eig[r^2])"),
                       (f"cluster_{tag}.ratio_split", "ratio_split (data-driven clusters)"),
                       (f"coh_{tag}.coherence", "coherence lam_max/max_l lam_max(B_ll)")]
            fig, axes = plt.subplots(2, 4, figsize=(17, 8))
            for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
                for j, (path, label) in enumerate(metrics):
                    ax = axes[i, j]
                    for st in [s_init, s_mid, s_end]:
                        m, s_, ns, per = agg(main, kind, setting, st, path, Ls_present)
                        a, b, R2 = linfit(Ls_present, m)
                        fits[f"{kind}|{tag}|{setting}|{path}|step{st}"] = {"slope": a, "intercept": b, "R2": R2,
                                                                           "means": m.tolist(), "stds": s_.tolist(), "n": ns}
                        ax.errorbar(Ls_present, m, yerr=s_, marker="o", capsize=3, color=ckcol.get(st, "k"),
                                    label=f"step {st}: slope={a:.2f}, R2={R2:.2f}")
                        xx = np.array([min(Ls_present), max(Ls_present)])
                        ax.plot(xx, a * xx + b, color=ckcol.get(st, "k"), ls=":", lw=0.8)
                    xx = np.array([1, max(Ls_present) + 0.5])
                    ax.plot(xx, xx, "k--", lw=1, label="y = L (paper C2)")
                    ax.set_title(f"{setting} (r={r},d_out={d_out}): {label}", fontsize=8)
                    ax.set_xlabel("L (number of weight matrices)")
                    ax.legend(fontsize=6)
                    ax.grid(alpha=0.3)
            fig.suptitle(f"{exp}: {kname[kind]}: dominant/bulk ratios and coherence of {'H_o' if tag == 'Ho' else 'sym(H)'} vs depth (mean +- std over seeds)")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"{exp}_ratios_{tag}_{kind}.png"), dpi=120)
            plt.close(fig)
    analysis["linear_fits"] = fits

    # (b2) cross-kind comparison (H_o) incl. SNN reference --------------------------------
    comp_metrics = [("cluster_Ho.ratio_paper", "ratio_paper(H_o)"), ("cluster_Ho.ratio_split", "ratio_split(H_o)"),
                    ("coh_Ho.coherence", "coherence(H_o)"), ("dominant_count_Ho", "dominant count(H_o)")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    comp = {}
    for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
        for j, (path, label) in enumerate(comp_metrics):
            ax = axes[i, j]
            for kind in kinds_ref:
                for st, ls in [(s_init, "-"), (s_end, "--")]:
                    m, s_, ns, per = agg(allruns, kind, setting, st, path, Ls_present)
                    comp[f"{kind}|{setting}|{path}|step{st}"] = {"means": m.tolist(), "stds": s_.tolist(), "n": ns}
                    ax.errorbar(Ls_present, m, yerr=s_, marker="o", ms=4, ls=ls, capsize=2, color=KIND_COL[kind],
                                label=f"{kind} step {st}")
            if j < 3:
                xx = np.array([1, max(Ls_present) + 0.5]); ax.plot(xx, xx, "k:", lw=1, label="y = L")
            else:
                ax.axhline(r * r, color="k", ls="--", lw=1, label=f"r^2={r*r}"); ax.axhline(d_out, color="gray", ls="-.", lw=1, label=f"d_out={d_out}")
            ax.set_title(f"setting {setting} (r={r}, d_out={d_out}): {label}", fontsize=9)
            ax.set_xlabel("L"); ax.grid(alpha=0.3); ax.legend(fontsize=5, ncol=2)
    fig.suptitle(f"{exp}: kinds compared (solid = init, dashed = step {s_end}; mean +- std over seeds; snn = e2a reference)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_compare_kinds_Ho.png"), dpi=120)
    plt.close(fig)
    analysis["kind_comparison_Ho"] = comp

    # (c) dominant count vs L (rows = kinds, cols = settings) -------------------------------
    fig, axes = plt.subplots(len(kinds_ref), 2, figsize=(11, 3.4 * len(kinds_ref)), squeeze=False)
    counts = {}
    for ki, kind in enumerate(kinds_ref):
        for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
            ax = axes[ki, i]
            for tag, ls in [("Ho", "-"), ("H", "--")]:
                for st in [s_init, s_end]:
                    m, s_, ns, per = agg(allruns, kind, setting, st, f"dominant_count_{tag}", Ls_present)
                    counts[f"{kind}|{tag}|{setting}|step{st}"] = {"means": m.tolist(), "stds": s_.tolist(), "per_L": per}
                    ax.errorbar(np.array(Ls_present) + (0.05 if tag == "H" else -0.05), m, yerr=s_, marker="o", ls=ls,
                                capsize=3, color=ckcol[st], label=f"{'H_o' if tag=='Ho' else 'sym(H)'} step {st}")
            ax.axhline(r * r, color="k", ls="--", lw=1, label=f"r^2 = {r*r}")
            ax.axhline(d_out, color="gray", ls="-.", lw=1, label=f"d_out = {d_out}")
            ax.set_title(f"{kname[kind]}, setting {setting} (r={r}, d_out={d_out})", fontsize=9)
            ax.set_xlabel("L"); ax.set_ylabel("split_n_dominant (largest log gap + 1)")
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.suptitle(f"{exp}: data-driven number of dominant eigenvalues vs depth (mean +- std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_dominant_count.png"), dpi=120)
    plt.close(fig)
    analysis["dominant_counts"] = counts

    # (d) zero / negative fractions (rows: zero A, zero B, neg A, neg B; cols: kinds) ---------------
    fig, axes = plt.subplots(4, len(kinds), figsize=(5.5 * len(kinds), 14), squeeze=False)
    fr = {}
    for ki, kind in enumerate(kinds):
        for j, what in enumerate(["zero_frac", "neg_frac"]):
            for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
                ax = axes[2 * j + i, ki]
                for tag, ls in [("Ho", "-"), ("H", "--")]:
                    for st in [s_init, s_end]:
                        m, s_, ns, per = agg(main, kind, setting, st, f"{what}_{tag}", Ls_present)
                        fr[f"{kind}|{what}|{tag}|{setting}|step{st}"] = {"means": m.tolist(), "stds": s_.tolist()}
                        ax.errorbar(Ls_present, m, yerr=s_, marker="o", ls=ls, capsize=3, color=ckcol[st],
                                    label=f"{'H_o' if tag=='Ho' else 'sym(H)'} step {st}")
                if what == "zero_frac":
                    P_ = {L: D_IN * WIDTH + WIDTH * WIDTH * (L - 2) + d_out * WIDTH for L in Ls_present}
                    paper = [1 - (r * r + r * (D_IN + d_out - 2 * r)) / P_[L] for L in Ls_present]
                    ax.plot(Ls_present, paper, "k:", label="paper: 1 - (r^2 + r(d_in+d_out-2r))/P")
                ax.set_title(f"{kname[kind]}, setting {setting}: {'fraction |eig| < 1e-8 max' if what == 'zero_frac' else 'fraction eig < -1e-8 max'}", fontsize=8)
                ax.set_xlabel("L"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.suptitle(f"{exp}: zero-space and negative fractions vs depth")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_zero_neg_fractions.png"), dpi=120)
    plt.close(fig)
    analysis["fractions"] = fr

    # (e) ||H_f||/||H_o|| vs step (rows settings, cols kinds) -------------------------------------
    fig, axes = plt.subplots(2, len(kinds), figsize=(5.5 * len(kinds), 8), squeeze=False)
    c3 = {}
    for ki, kind in enumerate(kinds):
        for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
            ax = axes[i, ki]
            for L, col in zip(Ls_present, cols):
                ms, ss = [], []
                for st in steps_full:
                    m, s_, ns, per = agg(main, kind, setting, st, "Hf_over_Ho", [L])
                    ms.append(m[0]); ss.append(s_[0])
                c3[f"{kind}|{setting}|L{L}"] = {"steps": steps_full, "means": ms, "stds": ss}
                ax.errorbar(steps_full, ms, yerr=ss, marker="o", capsize=3, color=col, label=f"L={L}")
            ax.set_title(f"{kname[kind]}, setting {setting}: ||H_f||_F / ||H_o||_F", fontsize=9)
            ax.set_xlabel("training step"); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_yscale("log")
    fig.suptitle(f"{exp}: C3 test, functional part H_f = H - H_o during training (all three kinds have symmetric H)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_Hf_vs_step.png"), dpi=120)
    plt.close(fig)
    analysis["c3"] = c3

    # (f) top-eigenvector layer-energy heatmaps per kind x setting ---------------------------------
    energy_tab = {}
    for kind in kinds:
        for setting, (r, d_out) in SETTINGS.items():
            k_req = max(r * r, d_out)
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for i, st in enumerate([s_init, s_end]):
                for j, (what, title) in enumerate([("top1", "top-1 eigenvector of H_o"),
                                                   ("topk", f"mean over top-{k_req} eigenvectors of H_o")]):
                    M = np.full((max(Ls_present), len(Ls_present)), np.nan)
                    for c_, L in enumerate(Ls_present):
                        vals = []
                        for r_ in sel(main, kind, setting, L):
                            c = ckpt(r_, st)
                            if c is None or "topk_energy_Ho" not in c:
                                continue
                            E = np.asarray(c["topk_energy_Ho"])  # (K_TOP, L)
                            vals.append(E[0] if what == "top1" else E[:k_req].mean(0))
                        if vals:
                            M[:L, c_] = np.mean(vals, axis=0)
                    energy_tab[f"{kind}|{setting}|{what}|step{st}"] = M.tolist()
                    ax = axes[i, j]
                    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1, origin="lower")
                    ax.set_xticks(range(len(Ls_present))); ax.set_xticklabels(Ls_present)
                    ax.set_yticks(range(max(Ls_present))); ax.set_yticklabels([f"W{l+1}" for l in range(max(Ls_present))])
                    ax.set_xlabel("L"); ax.set_ylabel("layer")
                    for c_, L in enumerate(Ls_present):
                        for l in range(L):
                            if np.isfinite(M[l, c_]):
                                ax.text(c_, l, f"{M[l, c_]:.2f}", ha="center", va="center", fontsize=6,
                                        color="w" if M[l, c_] < 0.5 else "k")
                    ax.set_title(f"step {st}: {title}", fontsize=9)
                    fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"{exp}: {kname[kind]}: layer energy of top eigenvectors of H_o, setting {setting} (r={r}, d_out={d_out}); W{{L}} = readout")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"{exp}_eigvec_layer_energy_{kind}_{setting}.png"), dpi=120)
            plt.close(fig)
    analysis["eigvec_layer_energy"] = energy_tab

    # (g) feature mean vs covariance per kind ----------------------------------------------------
    feat = {}
    for kind in kinds:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
            ax = axes[0, i]
            for path, lab, mk in [("features.mu_sq_norm", "||mu||^2", "o"), ("features.cov_top3.0", "lambda_1(Cov_a)", "s"),
                                  ("features.second_moment_top3.0", "lambda_1(M = a^T a/n)", "^"),
                                  ("coh_Ho.lambda_max", "lambda_max(H_o)", "d"),
                                  ("coh_Ho.lambda_max_blocks.-1", "lambda_max(B_LL)", "x")]:
                for st in [s_init, s_end]:
                    m, s_, ns, per = agg(main, kind, setting, st, path, Ls_present)
                    feat[f"{kind}|{path}|{setting}|step{st}"] = {"means": m.tolist(), "stds": s_.tolist()}
                    ax.errorbar(Ls_present, m, yerr=s_, marker=mk, capsize=2, color=ckcol[st], ls="-" if st == s_init else "--",
                                label=f"{lab} step {st}", ms=4)
            ax.set_yscale("log"); ax.set_xlabel("L"); ax.grid(alpha=0.3); ax.legend(fontsize=5, ncol=2, loc="lower left")
            ax.set_title(f"{kname[kind]}, setting {setting}: readout-feature second moments vs L", fontsize=9)
            ax = axes[1, i]
            for st in [s_init, s_end]:
                m1, _, _, _ = agg(main, kind, setting, st, "features.mu_sq_norm", Ls_present)
                m2, _, _, _ = agg(main, kind, setting, st, "features.cov_top3.0", Ls_present)
                m3, s3, _, _ = agg(main, kind, setting, st, "cluster_Ho.ratio_split", Ls_present)
                m4, s4, _, _ = agg(main, kind, setting, st, "cluster_Ho.ratio_paper", Ls_present)
                ax.plot(Ls_present, m1 / m2, marker="o", color=ckcol[st], label=f"||mu||^2 / lambda_1(Cov_a) step {st}")
                ax.errorbar(Ls_present, m3, yerr=s3, marker="s", ls="--", color=ckcol[st], capsize=2, label=f"ratio_split(H_o) step {st}")
                ax.errorbar(Ls_present, m4, yerr=s4, marker="^", ls=":", color=ckcol[st], capsize=2, label=f"ratio_paper(H_o) step {st}")
                feat[f"{kind}|mu_over_cov|{setting}|step{st}"] = (m1 / m2).tolist()
            xx = np.array([1, max(Ls_present) + 0.5]); ax.plot(xx, xx, "k--", lw=1, label="y = L")
            ax.set_xlabel("L"); ax.grid(alpha=0.3); ax.legend(fontsize=6); ax.set_yscale("log")
            ax.set_title(f"{kname[kind]}, setting {setting}: feature-mean ratio vs measured Hessian ratios", fontsize=9)
        fig.suptitle(f"{exp}: P2 test, mean-spike of the readout features vs L ({kname[kind]})")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_feature_moments_{kind}.png"), dpi=120)
        plt.close(fig)
    analysis["features"] = feat

    # (h) loss curves + activity / spike rate (rows: loss A, loss B, activity; cols: kinds) -----------
    fig, axes = plt.subplots(3, len(kinds), figsize=(5.5 * len(kinds), 11), squeeze=False)
    for ki, kind in enumerate(kinds):
        for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
            ax = axes[i, ki]
            for L, col in zip(Ls_present, cols):
                for r_ in sel(main, kind, setting, L):
                    z = np.load(os.path.join(res_dir, "raw", r_["name"] + ".npz"))
                    ax.plot(z["loss_curve"], color=col, lw=0.7, alpha=0.8, label=f"L={L}" if r_["seed"] == 0 else None)
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE loss"); ax.legend(fontsize=6, ncol=2)
            opt = "GD lr=min(0.1,1/lam_max)" if kind == "linear" else "Adam lr=5e-3"
            ax.set_title(f"{kname[kind]}, setting {setting}: training loss ({opt})", fontsize=9); ax.grid(alpha=0.3)
        ax = axes[2, ki]
        path = "spikes.mean_rate" if kind == "smooth_snn" else "activity.mean_active_frac"
        for setting, ls in [("A", "-"), ("B", "--")]:
            for L, col in zip(Ls_present, cols):
                ms = []
                for st in steps_all:
                    m, s_, ns, per = agg(main, kind, setting, st, path, [L])
                    ms.append(m[0])
                ax.plot(steps_all, ms, marker="o", ms=3, ls=ls, color=col, label=f"L={L} {setting}")
        ax.set_xlabel("step"); ax.set_ylabel("mean rate (smooth_snn) / active fraction (relu)"); ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)
        ax.set_title(f"{kname[kind]}: hidden activity during training (seed mean)", fontsize=9)
    fig.suptitle(f"{exp}: loss curves and hidden-layer activity")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_loss_activity.png"), dpi=120)
    plt.close(fig)

    # (i) centering experiment (P2) ------------------------------------------------------------
    cent = [r_ for r_ in runs if r_.get("centered", False)]
    if cent:
        ckinds = [k for k in KINDS if any(r_["kind"] == k for r_ in cent)]
        Lc = sorted({r_["L"] for r_ in cent})
        setting_c = cent[0]["setting"]; r, d_out = SETTINGS[setting_c]
        fig, axes = plt.subplots(len(ckinds), len(Lc), figsize=(4.2 * len(Lc), 3.6 * len(ckinds)), squeeze=False)
        ctab = {}
        for ki, kind in enumerate(ckinds):
            for j, L in enumerate(Lc):
                ax = axes[ki, j]
                for centered, ls, lab in [(False, "-", "uncentred"), (True, "--", "centred")]:
                    for r_ in sel(runs, kind, setting_c, L, centered):
                        if r_["seed"] != 0:
                            continue
                        z = np.load(os.path.join(res_dir, "raw", r_["name"] + ".npz"))
                        for st in (s_init, s_end):
                            e = z[f"eig_Ho_step{st}"][:40]
                            idx = np.arange(1, len(e) + 1)
                            ax.plot(idx[e > 0], e[e > 0], color=ckcol[st], ls=ls, marker=".", ms=3, lw=0.9, label=f"{lab} step {st}")
                        for st in (s_init, s_end):
                            c = ckpt(r_, st)
                            ctab[f"{kind}|L{L}|{'centered' if centered else 'uncentered'}|step{st}"] = {
                                "dominant_count_Ho": c["dominant_count_Ho"], "ratio_split_Ho": c["cluster_Ho"]["ratio_split"],
                                "ratio_paper_Ho": c["cluster_Ho"]["ratio_paper"], "coherence_Ho": c["coh_Ho"]["coherence"],
                                "lambda_max_Ho": c["coh_Ho"]["lambda_max"], "lambda_max_BLL": c["coh_Ho"]["lambda_max_blocks"][-1],
                                "E_WL_top1": np.asarray(c["topk_energy_Ho"])[0, -1],
                                "E_WL_topk_mean": float(np.asarray(c["topk_energy_Ho"])[:max(r * r, d_out), -1].mean()),
                                "top8_Ho": c["top20_Ho"][:8], "loss": c["loss"],
                                "mu_sq_norm_uncentered_feats": c["features"]["mu_sq_norm"],
                                "cov_top1": c["features"]["cov_top3"][0],
                                "dominant_count_H": c.get("dominant_count_H"), "ratio_split_H": get(c, "cluster_H.ratio_split"),
                                "dominant_count_Htrue": c.get("dominant_count_Htrue"), "asymmetry_Htrue": c.get("asymmetry_Htrue"),
                                "Hf_true_over_Ho": c.get("Hf_true_over_Ho"), "Hf_over_Ho": c.get("Hf_over_Ho"),
                                "Htrue_minus_Hlib_over_Hlib": c.get("Htrue_minus_Hlib_over_Hlib"),
                                "mean_activity": c["spikes"]["mean_rate"] if c.get("spikes") else c["activity"]["mean_active_frac"],
                            }
                ax.axvline(r * r, color="k", ls="--", lw=0.8, label=f"r^2={r*r}"); ax.axvline(d_out, color="gray", ls="-.", lw=0.8, label=f"d_out={d_out}")
                ax.set_yscale("log"); ax.set_xlabel("index"); ax.set_title(f"{kname[kind]}, L={L}: top-40 eigs of H_o", fontsize=9)
                ax.grid(alpha=0.3); ax.legend(fontsize=5)
        fig.suptitle(f"{exp}: P2 test, centred readout features vs uncentred (setting {setting_c}, seed 0)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_centering_spectra.png"), dpi=120)
        plt.close(fig)
        # summary metrics
        mets = [("dominant_count_Ho", "dominant count(H_o)"), ("ratio_split_Ho", "ratio_split(H_o)"),
                ("coherence_Ho", "coherence(H_o)"), ("E_WL_top1", "E_WL(top-1 eigvec)")]
        fig, axes = plt.subplots(len(ckinds), len(mets), figsize=(4.2 * len(mets), 3.4 * len(ckinds)), squeeze=False)
        for ki, kind in enumerate(ckinds):
            for j, (mk, lab) in enumerate(mets):
                ax = axes[ki, j]
                for centered, ls, lab2 in [(False, "-", "uncentred"), (True, "--", "centred")]:
                    for st in (s_init, s_end):
                        ys = [ctab.get(f"{kind}|L{L}|{'centered' if centered else 'uncentered'}|step{st}", {}).get(mk, np.nan) for L in Lc]
                        ax.plot(Lc, ys, marker="o", ls=ls, color=ckcol[st], label=f"{lab2} step {st}")
                if mk == "dominant_count_Ho":
                    ax.axhline(r * r, color="k", ls="--", lw=1, label=f"r^2={r*r}"); ax.axhline(d_out, color="gray", ls="-.", lw=1, label=f"d_out={d_out}")
                if mk in ("ratio_split_Ho", "coherence_Ho"):
                    xx = np.array([1, max(Lc) + 0.5]); ax.plot(xx, xx, "k:", lw=1, label="y = L")
                ax.set_title(f"{kname[kind]}: {lab}", fontsize=9); ax.set_xlabel("L"); ax.grid(alpha=0.3); ax.legend(fontsize=5)
        fig.suptitle(f"{exp}: centring the readout features (setting {setting_c}, seed 0): does the dominant cluster collapse?")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_centering_metrics.png"), dpi=120)
        plt.close(fig)
        analysis["centering"] = ctab

    # tables ---------------------------------------------------------------------------
    lines = []
    for kind in kinds:
        for setting, (r, d_out) in SETTINGS.items():
            lines.append(f"\n### {kname[kind]}, setting {setting} (r={r}, r^2={r*r}, d_out={d_out}), mean +- std over seeds\n")
            lines.append("| L | n | step | loss | act | ratio_paper(Ho) | ratio_gap(Ho) | ratio_split(Ho) | coh(Ho) | ndom(Ho) | ndom(H) | ratio_split(H) | zero%(Ho) | zero%(H) | neg%(H) | |Hf|/|Ho| | E_WL(top1) | mu^2/l1(Cov) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for L in Ls_present:
                for st in [s_init, s_mid, s_end]:
                    def ms(path, fmt="{:.2f}"):
                        m, s_, ns, per = agg(main, kind, setting, st, path, [L])
                        return (fmt.format(m[0]) + ("+-" + fmt.format(s_[0]) if ns[0] > 1 else "")) if ns[0] else "-"
                    n_ = agg(main, kind, setting, st, "loss", [L])[2][0]
                    EWL = []
                    for r_ in sel(main, kind, setting, L):
                        if ckpt(r_, st) is not None and "topk_energy_Ho" in ckpt(r_, st):
                            EWL.append(np.asarray(ckpt(r_, st)["topk_energy_Ho"])[0, -1])
                    m1 = agg(main, kind, setting, st, "features.mu_sq_norm", [L])[0][0]
                    m2 = agg(main, kind, setting, st, "features.cov_top3.0", [L])[0][0]
                    act = "spikes.mean_rate" if kind == "smooth_snn" else "activity.mean_active_frac"
                    lines.append(f"| {L} | {n_} | {st} | {ms('loss','{:.2e}')} | {ms(act,'{:.3f}')} | {ms('cluster_Ho.ratio_paper')} | "
                                 f"{ms('cluster_Ho.ratio_gap')} | {ms('cluster_Ho.ratio_split')} | {ms('coh_Ho.coherence')} | "
                                 f"{ms('dominant_count_Ho','{:.1f}')} | {ms('dominant_count_H','{:.1f}')} | {ms('cluster_H.ratio_split')} | "
                                 f"{ms('zero_frac_Ho','{:.3f}')} | {ms('zero_frac_H','{:.3f}')} | {ms('neg_frac_H','{:.3f}')} | "
                                 f"{ms('Hf_over_Ho','{:.3f}')} | {np.mean(EWL) if EWL else float('nan'):.2f} | {m1/m2:.2f} |")
    lines.append("\n### Linear fits metric = slope*L + intercept over L (seed means), H_o\n")
    lines.append("| kind | setting | metric | step | slope | intercept | R^2 | values per L |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k_, v in fits.items():
        kind, tag, setting, path, st = k_.split("|")
        if tag != "Ho":
            continue
        lines.append(f"| {kind} | {setting} | {path.split('.')[-1]} | {st} | {v['slope']:.3f} | {v['intercept']:.3f} | {v['R2']:.3f} | "
                     + ", ".join(f"{x:.2f}" for x in v["means"]) + " |")
    if snn:
        lines.append("\n### SNN reference (e2a_snn_depth, kaiming) for comparison: H_o at init / end (seed means)\n")
        lines.append("| setting | L | n | ratio_split init/end | coh init/end | ndom(Ho) init/end | ratio_paper init/end |")
        lines.append("|---|---|---|---|---|---|---|")
        for setting in SETTINGS:
            for L in Ls_present:
                a1 = agg(snn, "snn", setting, s_init, "cluster_Ho.ratio_split", [L]); a2 = agg(snn, "snn", setting, s_end, "cluster_Ho.ratio_split", [L])
                b1 = agg(snn, "snn", setting, s_init, "coh_Ho.coherence", [L]); b2 = agg(snn, "snn", setting, s_end, "coh_Ho.coherence", [L])
                c1 = agg(snn, "snn", setting, s_init, "dominant_count_Ho", [L]); c2 = agg(snn, "snn", setting, s_end, "dominant_count_Ho", [L])
                d1 = agg(snn, "snn", setting, s_init, "cluster_Ho.ratio_paper", [L]); d2 = agg(snn, "snn", setting, s_end, "cluster_Ho.ratio_paper", [L])
                lines.append(f"| {setting} | {L} | {a1[2][0]} | {a1[0][0]:.2f}/{a2[0][0]:.2f} | {b1[0][0]:.2f}/{b2[0][0]:.2f} | {c1[0][0]:.1f}/{c2[0][0]:.1f} | {d1[0][0]:.2f}/{d2[0][0]:.2f} |")
    if cent:
        lines.append(f"\n### Centring experiment (setting {setting_c}, seed 0): H_o metrics uncentred -> centred\n")
        lines.append("| kind | L | step | ndom(Ho) unc/cen | ratio_split unc/cen | coh unc/cen | lam_max(Ho) unc/cen | lam_max(B_LL) unc/cen | E_WL(top1) unc/cen | ||mu||^2 / l1(Cov) (unc feats) | loss unc/cen |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for kind in ckinds:
            for L in Lc:
                for st in (s_init, s_end):
                    u = ctab.get(f"{kind}|L{L}|uncentered|step{st}"); c = ctab.get(f"{kind}|L{L}|centered|step{st}")
                    if u is None or c is None:
                        continue
                    lines.append(f"| {kind} | {L} | {st} | {u['dominant_count_Ho']}/{c['dominant_count_Ho']} | {u['ratio_split_Ho']:.2f}/{c['ratio_split_Ho']:.2f} | "
                                 f"{u['coherence_Ho']:.2f}/{c['coherence_Ho']:.2f} | {u['lambda_max_Ho']:.3f}/{c['lambda_max_Ho']:.3f} | "
                                 f"{u['lambda_max_BLL']:.3f}/{c['lambda_max_BLL']:.3f} | {u['E_WL_top1']:.2f}/{c['E_WL_top1']:.2f} | "
                                 f"{u['mu_sq_norm_uncentered_feats'] / u['cov_top1']:.2f} / {c['mu_sq_norm_uncentered_feats'] / c['cov_top1']:.2f} | {u['loss']:.2e}/{c['loss']:.2e} |")
    table_md = "\n".join(lines)
    analysis["tables_markdown"] = table_md
    with open(os.path.join(res_dir, "analysis.json"), "w") as f:
        json.dump(to_py(analysis), f, indent=1)
    print(table_md)
    return analysis


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_key(r_):
    """Identity of a run for the resume logic: (kind, setting, L, seed, variant=centered)."""
    return (str(r_["kind"]), str(r_["setting"]), int(r_["L"]), int(r_["seed"]), bool(r_.get("centered", False)))


def run_name(kind, setting, L, seed, centered):
    return f"{kind}_{setting}_L{L}_seed{seed}" + ("_centered" if centered else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures", action="store_true", help="only regenerate figures from saved results")
    ap.add_argument("--no-timing", action="store_true", help="smoke: skip the L=8 Hessian timing probes")
    ap.add_argument("--no-validate", action="store_true", help="smoke: skip the validate_once library cross-checks")
    ap.add_argument("--res-dir", default=None, help="override the results directory (used to test the resume logic)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing summary.json and start over (default: RESUME = skip runs already in it)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print which planned runs are missing from summary.json (resume check) and exit")
    args = ap.parse_args()
    exp = EXP + ("_smoke" if args.smoke else "")
    res_dir = args.res_dir or os.path.join(ROOT, "results", exp)
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(res_dir, exist_ok=True)
    if args.figures:
        make_figures(res_dir, fig_dir, exp)
        return
    T0 = time.time()
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, exp={exp}, res_dir={res_dir}", flush=True)
    if args.smoke:
        if not args.no_validate:
            validate_once("relu", "A", 3, 0, print_dict=True)
            validate_once("smooth_snn", "B", 3, 0, print_dict=True)
            validate_once("linear", "A", 3, 0, print_dict=False)
            validate_once("relu", "B", 3, 0, centered=True, print_dict=False)
            validate_once("smooth_snn", "B", 2, 0, centered=True, print_dict=False)
        if not args.no_timing:
            for kind in KINDS:
                time_hessians(kind, "B", 8)
        plan = [("relu", "A", 2, 0, False), ("smooth_snn", "A", 2, 0, False), ("linear", "A", 2, 0, False),
                ("relu", "B", 3, 0, False), ("smooth_snn", "B", 3, 0, False), ("linear", "B", 3, 0, False),
                ("relu", "A", 2, 1, False), ("linear", "B", 8, 0, False),
                ("relu", "B", 2, 0, True), ("smooth_snn", "B", 2, 0, True)]
        steps, ho_every, full_steps = 40, 20, (0, 20, 40)
        tiers = [(0, plan)]
    else:
        steps, ho_every, full_steps = STEPS, HO_EVERY, FULL_H_STEPS
        tier0 = [(k, s, L, 0, False) for k in KINDS for s in SETTINGS for L in LS]            # seed 0, all kinds
        tier0 += [(k, "B", L, 0, True) for k in ("relu", "smooth_snn") for L in (2, 4, 8)]      # centred variants
        tier0 += [(k, s, L, 1, False) for k in ("relu", "linear") for s in SETTINGS for L in LS]  # cheap seed 1
        tier1 = [("smooth_snn", s, L, 1, False) for s in SETTINGS for L in (2, 3, 4, 5)]
        tier2 = [("smooth_snn", s, L, 1, False) for s in SETTINGS for L in (6, 8)]
        tiers = [(0, tier0), (1, tier1), (2, tier2)]
    # ---- RESUME: load the runs that are already saved (summary.json + raw npz) ------------------
    sfile = os.path.join(res_dir, "summary.json")
    mfile = os.path.join(res_dir, "run_meta.json")
    summary: List[Dict[str, object]] = []
    done: Dict[tuple, str] = {}
    log: Dict[str, object] = {}
    if not args.fresh and os.path.exists(sfile):
        with open(sfile) as f:
            summary = json.load(f)
        for r_ in summary:
            done[run_key(r_)] = r_["name"]
        if os.path.exists(mfile):
            with open(mfile) as f:
                log = json.load(f)
        print(f"RESUME: {len(summary)} runs already in {sfile} (matched on kind/setting/L/seed/centered)", flush=True)
    log.setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    log["budget_min"] = BUDGET_MIN
    skipped: List[str] = log.setdefault("skipped", [])
    log["plan"] = [{"tier": t, "runs": [run_name(*x) for x in p]} for t, p in tiers]
    all_planned = [(t, x) for t, p in tiers for x in p]

    def is_done(key, name):
        """A run counts as done only if it is in summary.json AND its raw npz exists."""
        return key in done and os.path.exists(os.path.join(res_dir, "raw", f"{name}.npz"))

    missing = [run_name(*x) for t, x in all_planned if not is_done(x, run_name(*x))]
    stale = [run_name(*x) for t, x in all_planned if run_key(dict(kind=x[0], setting=x[1], L=x[2], seed=x[3], centered=x[4])) in done
             and not os.path.exists(os.path.join(res_dir, "raw", f"{run_name(*x)}.npz"))]
    this = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs_already_done": len(summary), "missing_runs": missing,
            "redo_missing_raw": stale, "done_this_session": []}
    log.setdefault("resumes", []).append(this)
    print(f"RESUME: {len(missing)} of {len(all_planned)} planned runs still to do: {missing}", flush=True)
    if stale:
        print(f"RESUME: {len(stale)} runs are in summary.json but their raw npz is missing -> redone: {stale}", flush=True)
    est_total = sum(EST_RUN_SEC[x[0]][x[2]] for t, x in all_planned if not is_done(x, run_name(*x))) / 60
    print(f"RESUME: estimated wall time for the missing runs: {est_total:.1f} min (budget {BUDGET_MIN} min)", flush=True)
    if args.dry_run:
        print("DRY RUN: exiting without running anything", flush=True)
        return
    for tier, plan in tiers:
        for (kind, setting, L, seed, centered) in plan:
            key = (kind, setting, L, seed, centered)
            name = run_name(kind, setting, L, seed, centered)
            if is_done(key, name):
                print(f"SKIP {name}: already in summary.json", flush=True)
                continue
            if key in done:  # summary entry without raw file -> drop and redo
                summary = [r_ for r_ in summary if run_key(r_) != key]
                del done[key]
            elapsed_min = (time.time() - T0) / 60
            est = EST_RUN_SEC[kind][L] / 60
            if tier > 0 and elapsed_min + est > BUDGET_MIN:
                msg = f"{name} (tier {tier}; elapsed {elapsed_min:.1f} min + est {est:.1f} > budget {BUDGET_MIN})"
                print("SKIP", msg, flush=True)
                skipped.append(msg)
                continue
            info = run_one(kind, setting, L, seed, centered, steps, ho_every, full_steps, res_dir)
            info["tier"] = tier
            assert run_key(info) == key, (run_key(info), key)
            summary.append(info)
            done[key] = name
            this["done_this_session"].append(name)
            this["elapsed_min"] = log["elapsed_min"] = (time.time() - T0) / 60
            with open(sfile + ".tmp", "w") as f:
                json.dump(summary, f)
            os.replace(sfile + ".tmp", sfile)  # atomic: an interrupted write never corrupts summary.json
            with open(mfile, "w") as f:
                json.dump(log, f, indent=1)
            print(f"[elapsed {log['elapsed_min']:.1f} min, {len(summary)} runs in summary, "
                  f"{len(this['done_this_session'])} done this session]", flush=True)
    this["finished"] = log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    this["elapsed_min"] = log["elapsed_min"] = (time.time() - T0) / 60
    log["n_runs"] = len(summary)
    with open(mfile, "w") as f:
        json.dump(log, f, indent=1)
    print(f"ALL RUNS DONE in {log['elapsed_min']:.1f} min this session ({len(this['done_this_session'])} new, "
          f"{len(summary)} total); skipped: {skipped}", flush=True)
    try:
        make_figures(res_dir, fig_dir, exp)
        print(f"FIGURES DONE; total {(time.time() - T0) / 60:.1f} min", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FIGURES FAILED ({e!r}); results are saved, rerun with --figures after fixing", flush=True)


if __name__ == "__main__":
    main()
