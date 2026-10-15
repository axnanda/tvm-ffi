import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""e6_training_dynamics: do the Hessian landscapes of SNNs EVOLVE the way the paper's linear nets do?

L = 4, setting B (r = 2, d_out = 6), n = 256, d_in = 16, widths 16, MSE, seeds {0, 1}.
Kinds: snn (protocol defaults), relu, smooth_snn, linear/gaussian, linear/balanced_uniform (r=2, s=1).
Optimiser: full-batch GD for 1500 steps at a fixed lr = clip(0.5 / lambda_max(sym H at init), 1e-3, 0.5)
(reported per run); plus snn and relu with Adam lr 5e-3 (same logging) because GD may stall for the SNN.
Logging: loss every step; H_o every 25 steps (61 checkpoints); full surrogate Hessian H every 125 steps
(13 checkpoints incl. 0 and 1500).  Per checkpoint: protocol stats (eigenvalues of H_o and sym(H), cluster
stats, coherence, top-eigenvector layer energies, feature second moments, spike diagnostics), top-5
eigenvalues, trace, ||H_f||_F, ||H_o||_F, ||H_f||/||H_o||, asymmetry(H), asymmetry(H_f), negative fraction,
spectral radius of the NON-symmetrised H (top-5 |eig|), balancedness ||W_{l+1}^T W_{l+1} - W_l W_l^T||_F
for each adjacent pair (raw and relative), wall times.

Usage:  python3 experiments/e6_training_dynamics.py --group snn|rest|all   # full run (results/e6_training_dynamics)
        python3 experiments/e6_training_dynamics.py --smoke                 # reduced run (results/e6_training_dynamics_smoke)
        python3 experiments/e6_training_dynamics.py --figures               # regenerate figures + analysis.json + summary.json
Runs are saved one JSON per run in results/<exp>/runs/<name>.json (+ raw/<name>.npz); existing runs are skipped
(resume).  --figures assembles summary.json from the per-run files.
"""
import argparse
import glob
import json
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "e6_training_dynamics"

N, D_IN, WIDTH, L = 256, 16, 16, 4
R, D_OUT = 2, 6  # setting B
SEEDS = [0, 1]
STEPS, HO_EVERY, FULL_EVERY = 1500, 25, 125
ADAM_LR = 5e-3
LR_CLIP = (1e-3, 0.5)
K_TOP = 16  # rows of the stored top-eigenvector layer-energy matrix
BUDGET_MIN = 80.0  # per-process wall budget; remaining runs are skipped (and logged) beyond it

# label -> (kind, init, optimizer).  Order = priority (seed 0 of everything runs before seed 1).
RUN_TYPES = {
    "snn": ("snn", None, "gd"),
    "linear_bal": ("linear", "balanced_uniform", "gd"),
    "linear_gauss": ("linear", "gaussian", "gd"),
    "relu": ("relu", None, "gd"),
    "smooth_snn": ("smooth_snn", None, "gd"),
    "snn_adam": ("snn", None, "adam"),
    "relu_adam": ("relu", None, "adam"),
}
GROUPS = {"snn": ["snn", "snn_adam"], "rest": ["linear_bal", "linear_gauss", "relu", "smooth_snn", "relu_adam"]}
GROUPS["all"] = list(RUN_TYPES)

STYLE = {  # colour, linestyle for the overlay figures
    "snn": ("#d62728", "-"), "snn_adam": ("#d62728", "--"),
    "relu": ("#1f77b4", "-"), "relu_adam": ("#1f77b4", "--"),
    "smooth_snn": ("#ff7f0e", "-"),
    "linear_bal": ("#2ca02c", "-"), "linear_gauss": ("#9467bd", "-"),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_py(o):
    """Recursively convert tensors / numpy / non-finite floats to JSON-serialisable python objects."""
    if isinstance(o, torch.Tensor):
        return to_py(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return [to_py(x) for x in o.tolist()] if o.ndim > 0 else to_py(o.item())
    if isinstance(o, (np.floating,)):
        return to_py(float(o))
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


def spectrum_summary(H: torch.Tensor, spec, tag: str):
    """Protocol stats for one matrix from ONE eigh of sym(H): cluster stats, layer coherence (same
    formulas as sl.layer_coherence, validated in --smoke), top-K_TOP eigenvector layer energies,
    zero / negative fractions, trace, Frobenius norm, top-5 eigenvalues.  Returns (dict, eigs ndarray)."""
    Hs = sl.symmetrize(H)
    eigs, evecs = sl.eigh(Hs)
    e = eigs.numpy()
    cs = sl.cluster_stats(e, R, D_IN, D_OUT)
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
        f"top5_{tag}": e[:5].tolist(),
        f"zero_frac_{tag}": cs["zero_count"] / n,
        f"neg_frac_{tag}": cs["negative_count"] / n,
        f"dominant_count_{tag}": cs["split_n_dominant"],
        f"trace_{tag}": float(torch.trace(Hs)),
        f"fro_{tag}": float(torch.linalg.norm(H)),
        f"lambda_min_{tag}": float(eigs[-1]),
    }
    return out, e


def balancedness(params, L: int):
    """||W_{l+1}^T W_{l+1} - W_l W_l^T||_F for l = 1..L-1 (raw) and relative to ||W_l W_l^T||_F."""
    raw, rel = [], []
    for l in range(1, L):
        Wl, Wn = params[f"W{l}"], params[f"W{l + 1}"]
        G = Wl @ Wl.T
        d = float(torch.linalg.norm(Wn.T @ Wn - G))
        raw.append(d)
        rel.append(d / max(float(torch.linalg.norm(G)), 1e-300))
    return raw, rel


def make_callback(spec, X, Y, full_every: int, total_steps: int, raw: Dict[str, np.ndarray], cache: Dict[str, object],
                  verbose: bool = True):
    """gd_train callback: H_o (+ stats) at every logged step, full H (+ stats) every `full_every` steps."""

    def cb(step, params, g, loss_value):
        t0 = time.time()
        rec: Dict[str, object] = {"step": int(step), "loss": float(loss_value),
                                  "grad_norm": float(torch.linalg.norm(g)), "P": int(spec.n_params)}
        full = (step % full_every == 0) or (step == total_steps)
        rec["full_H"] = bool(full)
        if not math.isfinite(loss_value) or not bool(torch.isfinite(g).all()):
            rec["diverged"] = True
            if verbose:
                print(f"    step {step:4d} loss={loss_value} -> DIVERGED, skipping Hessians", flush=True)
            return {"rec": rec}
        rec["diverged"] = False
        H_o = sl.gauss_newton(spec, params, X, Y)
        s_o, e_o = spectrum_summary(H_o, spec, "Ho")
        rec.update(s_o)
        rec["asym_Ho"] = sl.asymmetry(H_o)
        raw[f"eig_Ho_step{step}"] = e_o
        rec["features"] = sl.feature_second_moment_stats(spec, params, X)
        rec["spikes"] = sl.spike_diagnostics(spec, params, X) if spec.is_spiking else None
        rec["balancedness"], rec["balancedness_rel"] = balancedness(params, spec.L)
        rec["weight_norms"] = [float(torch.linalg.norm(params[f"W{l}"])) for l in range(1, spec.L + 1)]
        rec["wall_Ho"] = time.time() - t0
        if full:
            t1 = time.time()
            if step == 0 and "H0" in cache:
                H = cache.pop("H0")
            else:
                H = sl.full_hessian(spec, params, X, Y)
            H_f = sl.functional_part(H, H_o)
            s_h, e_h = spectrum_summary(H, spec, "H")
            rec.update(s_h)
            rec["asymmetry_H"] = sl.asymmetry(H)
            rec["asymmetry_Hf"] = sl.asymmetry(H_f)
            rec["Hf_norm"] = float(torch.linalg.norm(H_f))
            rec["Ho_norm"] = float(torch.linalg.norm(H_o))
            rec["H_norm"] = float(torch.linalg.norm(H))
            rec["Hf_over_Ho"] = rec["Hf_norm"] / rec["Ho_norm"] if rec["Ho_norm"] > 0 else float("nan")
            rec["Hf_sym_fro_over_Ho"] = float(torch.linalg.norm(sl.symmetrize(H_f))) / rec["Ho_norm"] if rec["Ho_norm"] > 0 else float("nan")
            # spectrum of the NON-symmetrised Jacobian (relevant for GD stability of a non-gradient field)
            ev = torch.linalg.eigvals(H)
            mods = torch.sort(ev.abs(), descending=True)[0]
            rec["eigH_nonsym_top5_abs"] = mods[:5].tolist()
            rec["eigH_nonsym_max_real"] = float(ev.real.max())
            rec["eigH_nonsym_min_real"] = float(ev.real.min())
            rec["eigH_nonsym_max_abs_imag"] = float(ev.imag.abs().max())
            raw[f"eig_symH_step{step}"] = e_h
            raw[f"eigH_nonsym_abs_step{step}"] = mods.numpy()
            rec["wall_H"] = time.time() - t1
        if verbose:
            msg = (f"    step {step:4d} loss={loss_value:.4e} lam_max(Ho)={rec['coh_Ho']['lambda_max']:.3e} "
                   f"coh={rec['coh_Ho']['coherence']:.2f} ndom={rec['dominant_count_Ho']} "
                   f"ratio_paper={rec['cluster_Ho']['ratio_paper']:.2f} ratio_split={rec['cluster_Ho']['ratio_split']:.2f} "
                   f"bal={np.mean(rec['balancedness']):.3e}")
            if rec["spikes"] is not None:
                msg += f" rate={rec['spikes']['mean_rate']:.3f}"
            msg += f" wall={time.time() - t0:.1f}s"
            if full:
                msg += (f" | lam_max(symH)={rec['coh_H']['lambda_max']:.3e} asym(H)={rec['asymmetry_H']:.3f} "
                        f"|Hf|/|Ho|={rec['Hf_over_Ho']:.3e} negfrac={rec['neg_frac_H']:.3f} wall_H={rec['wall_H']:.1f}s")
            print(msg, flush=True)
        return {"rec": rec}

    return cb


def make_data(seed: int):
    X = sl.make_whitened_inputs(N, D_IN, seed)
    Y, Wstar = sl.make_rank_r_targets(X, R, D_OUT, seed)
    # sanity: the data matches (r, d_in, d_out) passed to cluster_stats
    assert tuple(X.shape) == (N, D_IN), X.shape
    assert tuple(Y.shape) == (N, D_OUT), Y.shape
    rank_Y = int(torch.linalg.matrix_rank(Y))
    assert rank_Y == R, f"rank(Y) = {rank_Y} != r = {R}"
    assert torch.allclose(X.T @ X / N, torch.eye(D_IN, dtype=X.dtype), atol=1e-8), "inputs not whitened"
    return X, Y, Wstar


def run_one(label: str, seed: int, steps: int, ho_every: int, full_every: int, res_dir: str,
            verbose: bool = True, validate: bool = False) -> Dict[str, object]:
    kind, init, opt = RUN_TYPES[label]
    name = f"{label}_s{seed}"
    t_start = time.time()
    X, Y, Wstar = make_data(seed)
    spec = sl.ModelSpec(kind, L, D_IN, WIDTH, D_OUT)  # snn defaults: T=8 beta=0.9 theta=1 k=4 sigmoid soft rate
    if init == "balanced_uniform":
        params = spec.init_params(seed, init="balanced_uniform", r=R, s=1.0)
    elif init == "gaussian":
        params = spec.init_params(seed, init="gaussian")  # gain 1
    else:
        params = spec.init_params(seed)  # kaiming default for relu / snn kinds
    print(f"[{name}] {spec} P={spec.n_params} init={init or 'default'} opt={opt}", flush=True)
    # learning rate from the init Hessian (cached for the step-0 checkpoint)
    t0 = time.time()
    H0 = sl.full_hessian(spec, params, X, Y)
    lam0_symH = float(sl.eigvals(H0)[0])
    lam0_Ho = float(sl.eigvals(sl.gauss_newton(spec, params, X, Y))[0])
    if opt == "gd":
        lr_raw = 0.5 / lam0_symH
        lr = float(min(max(lr_raw, LR_CLIP[0]), LR_CLIP[1]))
    else:
        lr_raw, lr = ADAM_LR, ADAM_LR
    print(f"[{name}] lambda_max(sym H0)={lam0_symH:.4f} lambda_max(H_o0)={lam0_Ho:.4f} -> lr={lr:.5f} "
          f"(raw 0.5/lam={lr_raw:.5f}, clipped={lr != lr_raw}) 2/lr={2 / lr:.3f} [H0 in {time.time() - t0:.1f}s]", flush=True)
    raw: Dict[str, np.ndarray] = {}
    cache = {"H0": H0}
    if validate:  # cross-check my spectrum_summary against the library functions on H_o
        H_o = sl.gauss_newton(spec, params, X, Y)
        mine, _ = spectrum_summary(H_o, spec, "Ho")
        ref = sl.layer_coherence(H_o, spec.layer_slices)
        ref_cs = sl.cluster_stats(sl.eigvals(H_o), R, D_IN, D_OUT)
        ref_topk = sl.top_eigvec_layer_energy(H_o, spec.layer_slices, k=max(R * R, D_OUT)).numpy()
        d_coh = abs(mine["coh_Ho"]["coherence"] - ref["coherence"])
        d_cs = max(abs(float(mine["cluster_Ho"][k]) - float(ref_cs[k])) for k in ("dominant_mean", "bulk_mean", "ratio_paper")
                   if ref_cs[k] == ref_cs[k])
        d_topk = float(np.abs(mine["topk_energy_Ho"][: max(R * R, D_OUT)] - ref_topk).max())
        print(f"[{name}] VALIDATE coherence diff={d_coh:.2e} cluster diff={d_cs:.2e} topk-energy diff={d_topk:.2e}", flush=True)
        assert d_coh < 1e-8 and d_cs < 1e-8 and d_topk < 1e-6
    cb = make_callback(spec, X, Y, full_every, steps, raw, cache, verbose=verbose)
    hist = sl.gd_train(spec, params, X, Y, lr, steps, optimizer=opt, log_every=ho_every, callbacks=[cb])
    ckpts = [s["rec"] for s in hist["summaries"]]
    loss_curve = np.asarray(hist["loss"], dtype=np.float64)
    gn_curve = np.asarray(hist["grad_norm"], dtype=np.float64)
    raw["loss"] = loss_curve
    raw["grad_norm"] = gn_curve
    raw["logged_steps"] = np.asarray(hist["logged_steps"])
    os.makedirs(os.path.join(res_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(res_dir, "runs"), exist_ok=True)
    np.savez(os.path.join(res_dir, "raw", f"{name}.npz"), **raw)
    fin = loss_curve[np.isfinite(loss_curve)]
    run = {
        "name": name, "label": label, "kind": kind, "init": init or ("kaiming" if kind != "linear" else "gaussian"),
        "optimizer": opt, "seed": seed, "L": L, "r": R, "d_in": D_IN, "d_out": D_OUT, "n": N, "width": WIDTH,
        "P": int(spec.n_params), "steps": steps, "ho_every": ho_every, "full_every": full_every,
        "lr": lr, "lr_raw": lr_raw, "lr_clipped": bool(lr != lr_raw), "two_over_lr": 2 / lr,
        "lambda_max_symH_init": lam0_symH, "lambda_max_Ho_init": lam0_Ho,
        "loss_init": float(loss_curve[0]), "loss_final": float(loss_curve[-1]), "loss_min": float(fin.min()) if fin.size else None,
        "loss_max": float(fin.max()) if fin.size else None, "diverged": bool(not np.isfinite(loss_curve[-1])),
        "n_loss_increases": int(np.sum(np.diff(loss_curve) > 0)) if np.all(np.isfinite(loss_curve)) else None,
        "loss_curve": loss_curve.tolist(), "grad_norm_curve": gn_curve.tolist(),
        "checkpoints": ckpts, "wall_sec": time.time() - t_start,
    }
    with open(os.path.join(res_dir, "runs", f"{name}.json"), "w") as f:
        json.dump(to_py(run), f)
    print(f"[{name}] done: loss {run['loss_init']:.4e} -> {run['loss_final']:.4e} in {run['wall_sec'] / 60:.1f} min "
          f"(loss increases: {run['n_loss_increases']})", flush=True)
    return run


# --------------------------------------------------------------------------- #
# analysis + figures
# --------------------------------------------------------------------------- #
def load_runs(res_dir: str) -> List[Dict[str, object]]:
    runs = []
    for fn in sorted(glob.glob(os.path.join(res_dir, "runs", "*.json"))):
        with open(fn) as f:
            runs.append(json.load(f))
    order = {k: i for i, k in enumerate(RUN_TYPES)}
    runs.sort(key=lambda r_: (order.get(r_["label"], 99), r_["seed"]))
    return runs


def get(d, path, default=float("nan")):
    cur = d
    for p in path:
        if cur is None:
            return default
        if isinstance(p, int):
            if not isinstance(cur, list) or p >= len(cur):
                return default
            cur = cur[p]
        else:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
    if cur is None:
        return default
    if isinstance(cur, list):
        return cur
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def series(run, path, full_only=False, reducer=None):
    xs, ys = [], []
    for c in run["checkpoints"]:
        if c.get("diverged"):
            continue
        if full_only and not c.get("full_H"):
            continue
        v = get(c, path)
        if reducer is not None and isinstance(v, list):
            v = reducer(v)
        if isinstance(v, list):
            continue
        xs.append(c["step"])
        ys.append(v)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def exp_fit(x, y, floor=1e-12, x_min=None):
    """Fit log y = a - rate x over the decaying portion: from the argmax of y to the end, keeping only
    y > floor (numerical floor, e.g. the 1e-31 loss plateau of an interpolating linear net) and, with
    x_min, only x >= x_min (a 'late' fit that ignores the initial transient).  Also returns the plateau
    value (mean of the last 4 kept points) so a non-decaying ratio is quantified."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y) & (y > floor)
    if x_min is not None:
        ok &= x >= x_min
    x, y = x[ok], y[ok]
    if len(y) < 4:
        return {"rate": None, "R2": None, "n_points": int(len(y)), "x_start": None, "halving_steps": None}
    i0 = int(np.argmax(y))
    x, y = x[i0:], y[i0:]
    if len(y) < 4:
        return {"rate": None, "R2": None, "n_points": int(len(y)), "x_start": float(x[0]) if len(x) else None, "halving_steps": None}
    ly = np.log(y)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rate = -float(coef[1])
    return {"rate": rate, "R2": r2, "n_points": int(len(y)), "x_start": float(x[0]), "x_end": float(x[-1]), "intercept": float(coef[0]),
            "halving_steps": math.log(2) / rate if rate > 0 else None, "plateau_last4": float(np.mean(y[-4:])),
            "y_start": float(y[0]), "y_end": float(y[-1])}


# (key, title, extractor(run) -> (x, y), log_y)
def _bal_mean(v):
    return float(np.mean(v))


QUANTITIES = [
    ("loss", "loss (every step)", lambda r_: (np.arange(len(r_["loss_curve"])), np.asarray([np.nan if v is None else v for v in r_["loss_curve"]], float)), True),
    ("lam_max_symH", "lambda_max(sym H)  [dotted: 2/lr edge of stability, GD runs]", lambda r_: series(r_, ("coh_H", "lambda_max"), full_only=True), True),
    ("lr_x_lam_max_symH", "lr * lambda_max(sym H)  (GD runs; grey: 2 = edge of stability)", lambda r_: (lambda xy: (xy[0], r_["lr"] * xy[1]))(series(r_, ("coh_H", "lambda_max"), full_only=True)) if r_["optimizer"] == "gd" else (np.array([]), np.array([])), False),
    ("lam_max_Ho", "lambda_max(H_o)", lambda r_: series(r_, ("coh_Ho", "lambda_max")), True),
    ("Hf_over_Ho", "||H_f||_F / ||H_o||_F  (C3; dotted: exp. fit, seed 0)", lambda r_: series(r_, ("Hf_over_Ho",), full_only=True), True),
    ("ratio_paper_Ho", "ratio_paper(H_o) = mean top r^2 / mean next r(d_in+d_out-2r)  (C5; grey: L)", lambda r_: series(r_, ("cluster_Ho", "ratio_paper")), True),
    ("ratio_split_Ho", "ratio_split(H_o) data-driven  (C5; grey: L)", lambda r_: series(r_, ("cluster_Ho", "ratio_split")), True),
    ("ratio_paper_H", "ratio_paper(sym H)  (C5; grey: L)", lambda r_: series(r_, ("cluster_H", "ratio_paper"), full_only=True), True),
    ("ratio_split_H", "ratio_split(sym H)  (C5; grey: L)", lambda r_: series(r_, ("cluster_H", "ratio_split"), full_only=True), True),
    ("coherence_Ho", "coherence(H_o) = lam_max / max_l lam_max(B_ll)  (P1; grey: L)", lambda r_: series(r_, ("coh_Ho", "coherence")), False),
    ("coherence_H", "coherence(sym H)  (grey: L)", lambda r_: series(r_, ("coh_H", "coherence"), full_only=True), False),
    ("split_n_dominant_Ho", "split_n_dominant(H_o)  (grey: r^2 = 4, d_out = 6)", lambda r_: series(r_, ("cluster_Ho", "split_n_dominant")), False),
    ("split_n_dominant_H", "split_n_dominant(sym H)  (grey: r^2 = 4, d_out = 6)", lambda r_: series(r_, ("cluster_H", "split_n_dominant"), full_only=True), False),
    ("asymmetry_H", "asymmetry(H) = ||H - H^T|| / ||H||  (snn only non-zero)", lambda r_: series(r_, ("asymmetry_H",), full_only=True), False),
    ("neg_frac_H", "negative eigenvalue fraction of sym H", lambda r_: series(r_, ("neg_frac_H",), full_only=True), False),
    ("balancedness", "mean_l ||W_{l+1}^T W_{l+1} - W_l W_l^T||_F", lambda r_: series(r_, ("balancedness",), reducer=_bal_mean), True),
    ("spike_rate", "mean spike rate (spiking kinds)", lambda r_: series(r_, ("spikes", "mean_rate")), False),
    ("spectral_radius_H", "max |eig(H)| of the NON-symmetrised H  (dotted: 2/lr, GD runs)", lambda r_: series(r_, ("eigH_nonsym_top5_abs", 0), full_only=True), True),
    ("grad_norm", "||g|| (every step)", lambda r_: (np.arange(len(r_["grad_norm_curve"])), np.asarray([np.nan if v is None else v for v in r_["grad_norm_curve"]], float)), True),
]


def plot_quantity(ax, key, title, extractor, logy, runs, fits, clip=True):
    labels_done = set()
    for run in runs:
        x, y = extractor(run)
        if len(x) == 0 or not np.any(np.isfinite(y)):
            continue
        col, ls = STYLE[run["label"]]
        lw = 1.6 if run["seed"] == 0 else 1.0
        alpha = 1.0 if run["seed"] == 0 else 0.55
        lab = run["label"] if run["label"] not in labels_done else None
        labels_done.add(run["label"])
        if logy:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, color=col, ls=ls, lw=lw, alpha=alpha, label=lab)
        if key in ("lam_max_symH", "spectral_radius_H") and run["optimizer"] == "gd" and run["seed"] == 0:
            ax.axhline(2 / run["lr"], color=col, ls=":", lw=0.9, alpha=0.8)
        if key == "Hf_over_Ho" and run["seed"] == 0:
            ft = fits.get(run["name"])
            if ft and ft.get("rate") is not None:
                xx = np.linspace(ft["x_start"], ft.get("x_end", x.max()), 50)
                ax.plot(xx, np.exp(ft["intercept"] - ft["rate"] * xx), color="k", ls=":", lw=0.8, alpha=0.7)
    if key.startswith("ratio") or key.startswith("coherence"):
        ax.axhline(L, color="grey", ls="--", lw=0.8)
    if key == "lr_x_lam_max_symH":
        ax.axhline(2.0, color="grey", ls="--", lw=0.8)
    if key.startswith("split_n_dominant"):
        ax.axhline(R * R, color="grey", ls="--", lw=0.8)
        ax.axhline(D_OUT, color="grey", ls=":", lw=0.8)
    if logy:
        ax.set_yscale("log")
    if clip and key == "loss":
        ax.set_ylim(bottom=1e-8)
    if clip and key == "Hf_over_Ho":
        ax.set_ylim(1e-7, 5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("step", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


def make_figures(res_dir: str, fig_dir: str, exp: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(res_dir)
    if not runs:
        print("no runs found")
        return
    os.makedirs(fig_dir, exist_ok=True)
    # --- exponential fits of ||H_f||/||H_o|| and of the loss over the decaying portion
    fits, loss_fits, late_fits, late_loss_fits = {}, {}, {}, {}
    for run in runs:
        x, y = series(run, ("Hf_over_Ho",), full_only=True)
        fits[run["name"]] = exp_fit(x, y, floor=1e-12)
        late_fits[run["name"]] = exp_fit(x, y, floor=1e-12, x_min=250)
        lc = np.asarray([np.nan if v is None else v for v in run["loss_curve"]], float)
        loss_fits[run["name"]] = exp_fit(np.arange(len(lc)), lc, floor=1e-26)
        late_loss_fits[run["name"]] = exp_fit(np.arange(len(lc)), lc, floor=1e-26, x_min=250)
    # --- overview grid
    nq = len(QUANTITIES)
    ncol = 4
    nrow = int(math.ceil(nq / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow))
    axes = axes.ravel()
    for ax, (key, title, extr, logy) in zip(axes, QUANTITIES):
        plot_quantity(ax, key, title, extr, logy, runs, fits)
    for ax in axes[nq:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:nq]:
        h, l_ = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l_):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle(f"{exp}: L={L}, setting B (r={R}, d_out={D_OUT}), GD at lr=0.5/lam_max(sym H_0) (clipped) / Adam 5e-3; "
                 f"thick: seed 0, thin: seed 1; dashed: Adam", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(os.path.join(fig_dir, f"{exp}_overview.png"), dpi=120)
    plt.close(fig)
    # --- one figure per quantity (loss and ||H_f||/||H_o||: full range + readable zoom)
    for key, title, extr, logy in QUANTITIES:
        if key in ("loss", "Hf_over_Ho"):
            fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
            plot_quantity(axs[0], key, title + " [full range]", extr, logy, runs, fits, clip=False)
            plot_quantity(axs[1], key, title + " [zoom]", extr, logy, runs, fits, clip=True)
            axs[0].legend(fontsize=8)
            fig.suptitle(f"{exp}: {title}", fontsize=9)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            plot_quantity(ax, key, title, extr, logy, runs, fits)
            ax.legend(fontsize=8)
            ax.set_title(f"{exp}: {title}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_{key}.png"), dpi=120)
        plt.close(fig)
    # --- spectra at init / mid / end for seed 0 (H_o and sym H)
    fig, axes = plt.subplots(2, len(RUN_TYPES), figsize=(3.2 * len(RUN_TYPES), 6.0), sharey="row")
    for j, label in enumerate(RUN_TYPES):
        rr = [r_ for r_ in runs if r_["label"] == label and r_["seed"] == 0]
        if not rr:
            axes[0, j].axis("off"); axes[1, j].axis("off"); continue
        run = rr[0]
        z = np.load(os.path.join(res_dir, "raw", run["name"] + ".npz"))
        full_steps = [c["step"] for c in run["checkpoints"] if c.get("full_H") and not c.get("diverged")]
        sel = sorted({full_steps[0], full_steps[len(full_steps) // 2], full_steps[-1]})
        for i, (mat, pref) in enumerate((("H_o", "eig_Ho_step"), ("sym H", "eig_symH_step"))):
            ax = axes[i, j]
            for s_, alpha in zip(sel, (0.35, 0.65, 1.0)):
                e = z[f"{pref}{s_}"]
                idx = np.arange(1, len(e) + 1)
                pos = e > 0
                ax.plot(idx[pos], e[pos], ".", ms=2.5, alpha=alpha, color=STYLE[label][0], label=f"step {s_}")
                if (e < 0).any():
                    ax.plot(idx[~pos], -e[~pos], "x", ms=2.5, alpha=alpha, color="k", label=f"-(neg) step {s_}" if s_ == sel[-1] else None)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.axvline(R * R + 0.5, color="grey", ls="--", lw=0.7); ax.axvline(R * R + R * (D_IN + D_OUT - 2 * R) + 0.5, color="grey", ls=":", lw=0.7)
            ax.set_title(f"{label} s0: {mat}", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
            ax.legend(fontsize=6)
            ax.set_xlabel("index", fontsize=7)
    fig.suptitle(f"{exp}: spectra at init / mid / end (seed 0); grey: r^2 and r^2 + bulk boundaries; x = |negative|", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(fig_dir, f"{exp}_spectra.png"), dpi=120)
    plt.close(fig)

    # --- analysis tables
    def at(run, path, which, full_only=False, reducer=None):
        x, y = series(run, path, full_only=full_only, reducer=reducer)
        if len(y) == 0:
            return float("nan")
        if which == "init":
            return float(y[0])
        if which == "end":
            return float(y[-1])
        if which == "mid":
            return float(y[len(y) // 2])
        if which == "max":
            return float(np.nanmax(y))
        if which == "min":
            return float(np.nanmin(y))
        raise ValueError(which)

    per_run = []
    for run in runs:
        x_l, y_l = series(run, ("coh_H", "lambda_max"), full_only=True)
        row = {
            "name": run["name"], "label": run["label"], "seed": run["seed"], "optimizer": run["optimizer"], "lr": run["lr"],
            "lr_clipped": run["lr_clipped"], "two_over_lr": run["two_over_lr"],
            "lambda_max_symH_init": run["lambda_max_symH_init"], "lambda_max_Ho_init": run["lambda_max_Ho_init"],
            "lambda_max_symH_end": at(run, ("coh_H", "lambda_max"), "end", True),
            "lambda_max_symH_max": at(run, ("coh_H", "lambda_max"), "max", True),
            "lr_x_lam_max_symH_max": float(run["lr"] * np.nanmax(y_l)) if len(y_l) else float("nan"),
            "spectral_radius_H_max": at(run, ("eigH_nonsym_top5_abs", 0), "max", True),
            "lr_x_spectral_radius_max": float(run["lr"] * at(run, ("eigH_nonsym_top5_abs", 0), "max", True)),
            "loss_init": run["loss_init"], "loss_final": run["loss_final"], "loss_min": run["loss_min"],
            "n_loss_increases": run["n_loss_increases"], "diverged": run["diverged"],
            "Hf_over_Ho_init": at(run, ("Hf_over_Ho",), "init", True), "Hf_over_Ho_mid": at(run, ("Hf_over_Ho",), "mid", True),
            "Hf_over_Ho_end": at(run, ("Hf_over_Ho",), "end", True),
            "Hf_fit": fits[run["name"]], "loss_fit": loss_fits[run["name"]],
            "Hf_late_fit": late_fits[run["name"]], "loss_late_fit": late_loss_fits[run["name"]],
            "Hf_over_Ho_plateau_last4": fits[run["name"]].get("plateau_last4"),
            "ratio_paper_Ho_init": at(run, ("cluster_Ho", "ratio_paper"), "init"), "ratio_paper_Ho_end": at(run, ("cluster_Ho", "ratio_paper"), "end"),
            "ratio_split_Ho_init": at(run, ("cluster_Ho", "ratio_split"), "init"), "ratio_split_Ho_end": at(run, ("cluster_Ho", "ratio_split"), "end"),
            "ratio_paper_H_init": at(run, ("cluster_H", "ratio_paper"), "init", True), "ratio_paper_H_end": at(run, ("cluster_H", "ratio_paper"), "end", True),
            "ratio_split_H_init": at(run, ("cluster_H", "ratio_split"), "init", True), "ratio_split_H_end": at(run, ("cluster_H", "ratio_split"), "end", True),
            "coherence_Ho_init": at(run, ("coh_Ho", "coherence"), "init"), "coherence_Ho_end": at(run, ("coh_Ho", "coherence"), "end"),
            "coherence_Ho_max": at(run, ("coh_Ho", "coherence"), "max"),
            "coherence_H_init": at(run, ("coh_H", "coherence"), "init", True), "coherence_H_end": at(run, ("coh_H", "coherence"), "end", True),
            "n_dom_Ho_init": at(run, ("cluster_Ho", "split_n_dominant"), "init"), "n_dom_Ho_end": at(run, ("cluster_Ho", "split_n_dominant"), "end"),
            "n_dom_H_init": at(run, ("cluster_H", "split_n_dominant"), "init", True), "n_dom_H_end": at(run, ("cluster_H", "split_n_dominant"), "end", True),
            "asymmetry_H_init": at(run, ("asymmetry_H",), "init", True), "asymmetry_H_end": at(run, ("asymmetry_H",), "end", True),
            "asymmetry_H_max": at(run, ("asymmetry_H",), "max", True), "asymmetry_H_min": at(run, ("asymmetry_H",), "min", True),
            "neg_frac_H_init": at(run, ("neg_frac_H",), "init", True), "neg_frac_H_end": at(run, ("neg_frac_H",), "end", True),
            "balancedness_init": at(run, ("balancedness",), "init", reducer=_bal_mean), "balancedness_end": at(run, ("balancedness",), "end", reducer=_bal_mean),
            "balancedness_rel_init": at(run, ("balancedness_rel",), "init", reducer=_bal_mean), "balancedness_rel_end": at(run, ("balancedness_rel",), "end", reducer=_bal_mean),
            "spike_rate_init": at(run, ("spikes", "mean_rate"), "init"), "spike_rate_end": at(run, ("spikes", "mean_rate"), "end"),
            "window_frac_init": at(run, ("spikes", "window_frac"), "init"), "window_frac_end": at(run, ("spikes", "window_frac"), "end"),
            "mean_frac_second_moment_init": at(run, ("features", "mean_frac_of_second_moment"), "init"),
            "mean_frac_second_moment_end": at(run, ("features", "mean_frac_of_second_moment"), "end"),
            "topvec_energy_last_layer_Ho_init": at(run, ("coh_Ho", "top_eigvec_layer_energy", L - 1), "init"),
            "topvec_energy_last_layer_Ho_end": at(run, ("coh_Ho", "top_eigvec_layer_energy", L - 1), "end"),
            "wall_min": run["wall_sec"] / 60,
        }
        per_run.append(row)
    # mean +- std over seeds per label
    per_label = {}
    scalar_keys = [k for k, v in per_run[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for label in RUN_TYPES:
        rows = [r_ for r_ in per_run if r_["label"] == label]
        if not rows:
            continue
        agg = {"n_seeds": len(rows), "seeds": [r_["seed"] for r_ in rows]}
        for k in scalar_keys:
            vals = np.asarray([r_[k] for r_ in rows], float)
            vals = vals[np.isfinite(vals)]
            agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()) if len(vals) > 1 else 0.0, "n": int(len(vals))} if len(vals) else None
        for fk in ("Hf_fit", "loss_fit", "Hf_late_fit", "loss_late_fit"):
            rates = np.asarray([r_[fk]["rate"] for r_ in rows if r_[fk]["rate"] is not None], float)
            r2s = np.asarray([r_[fk]["R2"] for r_ in rows if r_[fk]["R2"] is not None], float)
            agg[fk] = {"rate_mean": float(rates.mean()) if len(rates) else None, "rate_std": float(rates.std()) if len(rates) > 1 else 0.0,
                       "R2_mean": float(r2s.mean()) if len(r2s) else None, "n": int(len(rates)),
                       "per_seed": [r_[fk] for r_ in rows]}
        per_label[label] = agg
    analysis = {"per_run": per_run, "per_label": per_label, "config": {"L": L, "r": R, "d_in": D_IN, "d_out": D_OUT, "n": N, "width": WIDTH,
                "steps": STEPS, "ho_every": HO_EVERY, "full_every": FULL_EVERY, "adam_lr": ADAM_LR, "lr_rule": "clip(0.5/lambda_max(sym H_0), 1e-3, 0.5)"}}
    with open(os.path.join(res_dir, "analysis.json"), "w") as f:
        json.dump(to_py(analysis), f, indent=1)
    with open(os.path.join(res_dir, "summary.json"), "w") as f:
        json.dump(to_py(runs), f)
    # --- markdown tables
    lines = []

    def fmt(a, k, f="{:.3g}"):
        v = a.get(k)
        if v is None:
            return "-"
        return (f.format(v["mean"]) + (f" ± {f.format(v['std'])}" if v["n"] > 1 else ""))

    lines.append("| cond | lr | lr·λmax(symH) max | loss 0 → end | #loss↑ | ‖Hf‖/‖Ho‖ 0 → mid → end | Hf exp rate ×1e-3/step (R²) [all / step≥250] | loss exp rate ×1e-3 (R²) [all / step≥250] |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def fit_s(ff):
        return f"{1e3 * ff['rate_mean']:.2f}±{1e3 * ff['rate_std']:.2f} ({ff['R2_mean']:.2f})" if ff["rate_mean"] is not None else "-"

    for label, a in per_label.items():
        lines.append(f"| {label} | {fmt(a, 'lr', '{:.4f}')} | {fmt(a, 'lr_x_lam_max_symH_max', '{:.2f}')} | {fmt(a, 'loss_init', '{:.2e}')} → {fmt(a, 'loss_final', '{:.2e}')} | "
                     f"{fmt(a, 'n_loss_increases', '{:.0f}')} | {fmt(a, 'Hf_over_Ho_init', '{:.2e}')} → {fmt(a, 'Hf_over_Ho_mid', '{:.2e}')} → {fmt(a, 'Hf_over_Ho_end', '{:.2e}')} | "
                     f"{fit_s(a['Hf_fit'])} / {fit_s(a['Hf_late_fit'])} | {fit_s(a['loss_fit'])} / {fit_s(a['loss_late_fit'])} |")
    lines.append("")
    lines.append("| cond | ratio_paper(Ho) 0 → end | ratio_split(Ho) 0 → end | ratio_paper(symH) 0 → end | n_dom(Ho) 0 → end | coherence(Ho) 0 → end (max) | asym(H) 0 → end | neg frac(symH) 0 → end | balancedness 0 → end | rate 0 → end |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, a in per_label.items():
        lines.append(f"| {label} | {fmt(a, 'ratio_paper_Ho_init', '{:.2f}')} → {fmt(a, 'ratio_paper_Ho_end', '{:.2f}')} | "
                     f"{fmt(a, 'ratio_split_Ho_init', '{:.2f}')} → {fmt(a, 'ratio_split_Ho_end', '{:.2f}')} | "
                     f"{fmt(a, 'ratio_paper_H_init', '{:.2f}')} → {fmt(a, 'ratio_paper_H_end', '{:.2f}')} | "
                     f"{fmt(a, 'n_dom_Ho_init', '{:.1f}')} → {fmt(a, 'n_dom_Ho_end', '{:.1f}')} | "
                     f"{fmt(a, 'coherence_Ho_init', '{:.2f}')} → {fmt(a, 'coherence_Ho_end', '{:.2f}')} ({fmt(a, 'coherence_Ho_max', '{:.2f}')}) | "
                     f"{fmt(a, 'asymmetry_H_init', '{:.3f}')} → {fmt(a, 'asymmetry_H_end', '{:.3f}')} | "
                     f"{fmt(a, 'neg_frac_H_init', '{:.3f}')} → {fmt(a, 'neg_frac_H_end', '{:.3f}')} | "
                     f"{fmt(a, 'balancedness_init', '{:.2e}')} → {fmt(a, 'balancedness_end', '{:.2e}')} | "
                     f"{fmt(a, 'spike_rate_init', '{:.3f}')} → {fmt(a, 'spike_rate_end', '{:.3f}')} |")
    md = "\n".join(lines)
    with open(os.path.join(res_dir, "tables.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"figures written to {fig_dir}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--group", default="all", choices=list(GROUPS))
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (default: protocol seeds)")
    ap.add_argument("--print-one", action="store_true", help="print one full checkpoint dict (smoke default)")
    args = ap.parse_args()
    exp = EXP + ("_smoke" if args.smoke else "")
    res_dir = os.path.join(ROOT, "results", exp)
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(os.path.join(res_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(res_dir, "raw"), exist_ok=True)
    if args.figures:
        make_figures(res_dir, fig_dir, exp)
        return
    if args.smoke:
        steps, ho_every, full_every = 60, 20, 30
        seeds = [0]
    else:
        steps, ho_every, full_every = STEPS, HO_EVERY, FULL_EVERY
        seeds = SEEDS
    if args.seeds is not None:
        seeds = [int(s_) for s_ in args.seeds.split(",")]
    labels = GROUPS[args.group]
    t_all = time.time()
    log = {"exp": exp, "group": args.group, "steps": steps, "ho_every": ho_every, "full_every": full_every, "seeds": seeds,
           "runs": [], "skipped_budget": [], "skipped_existing": []}
    print(f"== {exp} group={args.group} steps={steps} ho_every={ho_every} full_every={full_every} seeds={seeds} labels={labels}", flush=True)
    first = True
    for seed in seeds:
        for label in labels:
            name = f"{label}_s{seed}"
            if os.path.exists(os.path.join(res_dir, "runs", f"{name}.json")):
                print(f"[{name}] exists, skipping (resume)", flush=True)
                log["skipped_existing"].append(name)
                continue
            elapsed = (time.time() - t_all) / 60
            if elapsed > BUDGET_MIN:
                print(f"[{name}] SKIPPED: wall budget exceeded ({elapsed:.1f} min)", flush=True)
                log["skipped_budget"].append(name)
                continue
            run = run_one(label, seed, steps, ho_every, full_every, res_dir, validate=args.smoke and first)
            if (args.smoke or args.print_one) and first:
                c = [c_ for c_ in run["checkpoints"] if c_.get("full_H")][0]
                print("---- one full checkpoint dict (step 0) ----")
                print(json.dumps(to_py(c), indent=1, default=str)[:6000])
                print("---- keys:", sorted(c.keys()))
            first = False
            log["runs"].append({"name": name, "wall_min": run["wall_sec"] / 60, "lr": run["lr"], "loss_final": run["loss_final"]})
            with open(os.path.join(res_dir, f"run_meta_{args.group}.json"), "w") as f:
                json.dump(to_py(log), f, indent=1)
    log["total_wall_min"] = (time.time() - t_all) / 60
    with open(os.path.join(res_dir, f"run_meta_{args.group}.json"), "w") as f:
        json.dump(to_py(log), f, indent=1)
    print(f"== group {args.group} finished in {log['total_wall_min']:.1f} min; skipped_budget={log['skipped_budget']}", flush=True)
    if args.smoke:
        make_figures(res_dir, fig_dir, exp)


if __name__ == "__main__":
    main()
