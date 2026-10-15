import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""e4_data_imbalance: "depth, not data" test (claim C4) for SNN / ReLU / linear nets.

Setting B (r=2, d_out=6), L in {2,4,8}, seeds {0,1}, n=256, d_in=16, width 16, MSE.
Kinds: "snn" (protocol defaults, kaiming init), "relu" (kaiming), "linear" (init="gaussian", gain 1).
Data conditions (same Haar frame / same Wstar for a given seed across conditions):
  k1, k10, k100, k1000 : X^T X / n = diag(geometric spectrum) with condition number kappa (mean eigenvalue 1,
                         i.e. the same trace as whitened); sl.geometric_spectrum(d, ratio) = [1, ratio, ..,
                         ratio^(d-1)] / mean, so ratio = kappa^(1/(d_in-1)) gives condition number exactly kappa;
                         targets Y = X Wstar^T with balanced Wstar (singular values [1, 1]).
  sv                   : whitened X, imbalanced target singular values [1.0, 0.1].
  scale25              : whitened X, then a random 25% of the input coordinates (4 of 16) scaled by 10
                         (condition number 100 but trace 412 instead of 16 -- NOT trace-normalised, as specified).
Per run: H_o at steps 0,100,..,400; full H (Jacobian of the surrogate-gradient field) at steps 0 and 400;
Adam lr 5e-3 (snn, relu) / GD lr = min(0.1, 1/lambda_max(H_init)) (linear) for 400 steps.  All protocol stats
per checkpoint; for snn also max Re eig of the raw non-symmetric H and the readout-block top eigenvalue.

Usage:  PYTHONPATH=. python3 experiments/e4_data_imbalance.py            # full run  -> results/e4_data_imbalance
        PYTHONPATH=. python3 experiments/e4_data_imbalance.py --smoke    # reduced   -> results/e4_data_imbalance_smoke
        PYTHONPATH=. python3 experiments/e4_data_imbalance.py --figures  # only regenerate figures/tables
"""
import argparse
import json
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "e4_data_imbalance"

N, D_IN, WIDTH = 256, 16, 16
R, D_OUT = 2, 6  # setting B
LS = [2, 4, 8]
SEEDS = [0, 1]
KINDS = ["snn", "relu", "linear"]
CONDS = ["k1", "k10", "k100", "k1000", "sv", "scale25"]
KAPPA = {"k1": 1, "k10": 10, "k100": 100, "k1000": 1000}
COND_LABEL = {"k1": "κ=1", "k10": "κ=10", "k100": "κ=100", "k1000": "κ=1000", "sv": "sv=[1,.1]", "scale25": "25%×10"}
SCALE25_FRAC, SCALE25_FACTOR = 0.25, 10.0
STEPS, HO_EVERY = 400, 100
FULL_H_STEPS = (0, 400)
LR_ADAM, LR_GD_CAP = 5e-3, 0.1
K_TOP = 16  # rows of the stored top-eigenvector layer-energy matrix (superset of max(r^2, d_out) = 6)
BUDGET_MIN = 58.0  # wall-clock budget for the run phase (minutes); optional tiers are skipped beyond it
# per-run cost estimates (seconds, 1 thread, contended machine), calibrated from the smoke timing probe
EST_RUN_SEC = {"snn": {2: 14, 3: 30, 4: 55, 8: 200}, "relu": {2: 4, 3: 6, 4: 8, 8: 30}, "linear": {2: 4, 3: 6, 4: 8, 8: 30}}
KIND_COL = {"snn": "tab:red", "relu": "tab:green", "linear": "tab:brown"}
L_COL = {2: "tab:blue", 3: "tab:cyan", 4: "tab:orange", 8: "tab:green"}
COND_COL = {"k1": "#1f77b4", "k10": "#6baed6", "k100": "#fd8d3c", "k1000": "#d62728", "sv": "#9467bd", "scale25": "#2ca02c"}
LAYER_CMAP = "viridis"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_py(o):
    """Recursively convert tensors / numpy / nan to JSON-serialisable python objects."""
    if isinstance(o, torch.Tensor):
        return to_py(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return [to_py(x) for x in o.tolist()] if o.ndim > 0 else to_py(o.item())
    if isinstance(o, np.floating):
        return to_py(float(o))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
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


def fnum(x) -> float:
    return float("nan") if x is None else float(x)


def make_data(cond: str, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Inputs / targets for one data condition; returns (X, Y, Wstar, info) and checks the realised statistics."""
    info: Dict[str, object] = {"cond": cond, "seed": seed, "label": COND_LABEL[cond]}
    sv = None
    requested_spectrum = None
    if cond in KAPPA:
        kappa = KAPPA[cond]
        if kappa == 1:
            X = sl.make_whitened_inputs(N, D_IN, seed)
            ratio = 1.0
            requested_spectrum = np.ones(D_IN)
        else:
            ratio = kappa ** (1.0 / (D_IN - 1))  # geometric_spectrum: [1, ratio, .., ratio^(d-1)] / mean
            requested_spectrum = sl.geometric_spectrum(D_IN, ratio)
            X = sl.make_covariance_imbalanced_inputs(N, D_IN, requested_spectrum, seed)
        info["kappa_requested"] = kappa
        info["geometric_ratio"] = ratio
    elif cond == "sv":
        X = sl.make_whitened_inputs(N, D_IN, seed)
        sv = [1.0, 0.1]
        info["kappa_requested"] = 1
        info["target_sv_requested"] = sv
    elif cond == "scale25":
        X = sl.make_whitened_inputs(N, D_IN, seed).clone()
        gen = torch.Generator().manual_seed(10_000 + seed)
        n_scaled = int(round(SCALE25_FRAC * D_IN))
        idx = torch.randperm(D_IN, generator=gen)[:n_scaled]
        X[:, idx] = X[:, idx] * SCALE25_FACTOR
        info["scaled_coords"] = sorted(int(i) for i in idx)
        info["kappa_requested"] = SCALE25_FACTOR ** 2
    else:
        raise ValueError(cond)
    Y, Wstar = sl.make_rank_r_targets(X, R, D_OUT, seed, singular_values=sv)
    # ---- realised statistics + sanity checks (r, d_in, d_out used below are the ones passed to cluster_stats)
    assert X.shape == (N, D_IN) and Y.shape == (N, D_OUT) and Wstar.shape == (D_OUT, D_IN)
    assert torch.allclose(Y, X @ Wstar.T, atol=1e-12), "make_rank_r_targets did not use the given X"
    assert int(torch.linalg.matrix_rank(Y)) == R, "target rank != r"
    Sigma = X.T @ X / N
    ev = torch.linalg.eigvalsh(Sigma).flip(0).numpy()  # descending
    offdiag = float((Sigma - torch.diag(torch.diagonal(Sigma))).abs().max())
    info.update({
        "cov_eigs": [float(x) for x in ev],
        "cov_trace": float(ev.sum()),
        "kappa_realised": float(ev[0] / ev[-1]),
        "cov_max_offdiag": offdiag,
        "target_sv": [float(x) for x in torch.linalg.svdvals(Wstar)],
        "cross_cov_sv": [float(x) for x in torch.linalg.svdvals(X.T @ Y / N)],  # Sigma Wstar^T singular values
        "Y_sv": [float(x) for x in torch.linalg.svdvals(Y / math.sqrt(N))],
    })
    assert offdiag < 1e-8, f"covariance not diagonal (max offdiag {offdiag})"
    if requested_spectrum is not None:
        assert np.allclose(np.sort(ev), np.sort(requested_spectrum), atol=1e-8), "realised spectrum != requested"
        assert abs(info["kappa_realised"] - KAPPA[cond]) < 1e-6 * KAPPA[cond], info["kappa_realised"]
    if cond == "sv":
        assert np.allclose(sorted(info["target_sv"][:R], reverse=True), [1.0, 0.1])
    return X, Y, Wstar, info


def make_spec(kind: str, L: int):
    if kind == "snn":
        return sl.ModelSpec("snn", L, D_IN, WIDTH, D_OUT, T=8, beta=0.9, theta=1.0, k=4.0, surrogate="sigmoid",
                            reset="soft", detach_reset=False, readout="rate")
    if kind == "relu":
        return sl.ModelSpec("relu", L, D_IN, WIDTH, D_OUT)
    if kind == "linear":
        return sl.ModelSpec("linear", L, D_IN, WIDTH, D_OUT)
    raise ValueError(kind)


def make_init(spec, kind: str, seed: int):
    if kind == "linear":
        return spec.init_params(seed, init="gaussian", gain=1.0), "gaussian(gain=1)"
    return spec.init_params(seed), "kaiming(gain=sqrt2)"


def spectrum_summary(H: torch.Tensor, spec, tag: str) -> Tuple[Dict[str, object], np.ndarray]:
    """All protocol statistics for one matrix from a single eigh of sym(H) (same formulas as
    sl.cluster_stats / sl.layer_coherence / sl.top_eigvec_layer_energy; cross-checked in validate_once)."""
    Hs = sl.symmetrize(H)
    eigs, evecs = sl.eigh(Hs)
    e = eigs.numpy()
    n = len(e)
    cs = sl.cluster_stats(e, R, D_IN, D_OUT)
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
    k1 = cs["split_n_dominant"]
    n_dom = R * R
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
        f"dominant_count_{tag}": k1,
        f"lambda1_{tag}": float(e[0]),
        f"bulk_edge_paper_{tag}": float(e[n_dom]) if n > n_dom else float("nan"),  # (r^2+1)-th eigenvalue
        f"bulk_edge_split_{tag}": float(e[k1]) if (k1 is not None and n > k1) else float("nan"),
        f"readout_block_lambda_max_{tag}": block_lams[-1],
        f"first_block_lambda_max_{tag}": block_lams[0],
        f"zero_frac_{tag}": cs["zero_count"] / n,
        f"neg_frac_{tag}": cs["negative_count"] / n,
        f"trace_{tag}": float(torch.trace(Hs)),
        f"fro_{tag}": float(torch.linalg.norm(H)),
        f"lambda_min_{tag}": float(eigs[-1]),
        f"top20_{tag}": [float(x) for x in e[:20]],
    }
    return out, e


def lower_block_rel_fro(H: torch.Tensor, slices: List[slice]) -> float:
    """||strictly lower layer-blocks of H||_F / ||H||_F (theory note 5 predicts 0 for snn)."""
    tot = float(torch.linalg.norm(H))
    if tot == 0:
        return 0.0
    acc = 0.0
    for i, si in enumerate(slices):
        for j, sj in enumerate(slices):
            if i > j:
                acc += float(torch.linalg.norm(H[si, sj])) ** 2
    return math.sqrt(acc) / tot


def raw_eig_stats(H: torch.Tensor, slices: List[slice], tol: float = 1e-10, force_full: bool = False):
    """Eigenvalues of the RAW (non-symmetric) H.  If the strictly-lower layer blocks vanish (block
    upper-triangular, theory note 5) the spectrum is the union of the diagonal-block spectra, which is
    much cheaper than a dense eigvals of a 1888x1888 matrix; otherwise (or with force_full) dense eigvals."""
    low = lower_block_rel_fro(H, slices)
    if low < tol and not force_full:
        ev = torch.cat([torch.linalg.eigvals(H[s_, s_]) for s_ in slices])
        method = "block_union"
    else:
        ev = torch.linalg.eigvals(H)
        method = "dense_eigvals"
    re, im = ev.real.numpy(), ev.imag.numpy()
    order = np.argsort(-re)
    re, im = re[order], im[order]
    scale = float(np.abs(re).max()) if len(re) else 0.0
    stats = {
        "method": method, "lower_block_rel_fro": low,
        "max_re": float(re[0]), "min_re": float(re[-1]), "max_abs_im": float(np.abs(im).max()),
        "n_neg_re": int((re < -1e-8 * scale).sum()), "n_complex": int((np.abs(im) > 1e-8 * scale).sum()),
        "top10_re": [float(x) for x in re[:10]],
        "readout_block_eigs_top3": [float(x) for x in np.sort(torch.linalg.eigvals(H[slices[-1], slices[-1]]).real.numpy())[::-1][:3]],
    }
    return stats, re, im


def make_callback(spec, X, Y, full_steps, raw: Dict[str, np.ndarray], H_cache: Dict[int, torch.Tensor], verbose=True):
    def cb(step, params, g, loss_value):
        t0 = time.time()
        rec: Dict[str, object] = {"step": int(step), "loss": float(loss_value), "grad_norm": float(torch.linalg.norm(g)),
                                  "P": int(spec.n_params)}
        if not math.isfinite(loss_value) or loss_value > 1e8:
            rec["diverged"] = True
            print(f"    step {step:4d} loss={loss_value:.3e} DIVERGED -> no Hessian", flush=True)
            return {"rec": rec}
        H_o = sl.gauss_newton(spec, params, X, Y)
        s_o, e_o = spectrum_summary(H_o, spec, "Ho")
        rec.update(s_o)
        rec["asym_Ho"] = sl.asymmetry(H_o)
        raw[f"eig_Ho_step{step}"] = e_o
        raw[f"topk_energy_Ho_step{step}"] = s_o["topk_energy_Ho"]
        rec["features"] = sl.feature_second_moment_stats(spec, params, X)
        rec["spikes"] = sl.spike_diagnostics(spec, params, X) if spec.is_spiking else None
        rec["wall_Ho"] = time.time() - t0
        if step in full_steps:
            t1 = time.time()
            H = H_cache.pop(step, None)
            if H is None:
                H = sl.full_hessian(spec, params, X, Y)
            H_f = sl.functional_part(H, H_o)
            s_h, e_h = spectrum_summary(H, spec, "H")
            rec.update(s_h)
            rec["asymmetry_H"] = sl.asymmetry(H)
            rec["Hf_norm"] = float(torch.linalg.norm(H_f))
            rec["Ho_norm"] = float(torch.linalg.norm(H_o))
            rec["H_norm"] = float(torch.linalg.norm(H))
            rec["Hf_over_Ho"] = rec["Hf_norm"] / rec["Ho_norm"] if rec["Ho_norm"] > 0 else float("nan")
            rec["asymmetry_Hf"] = sl.asymmetry(H_f)
            rec["lower_block_rel_fro_H"] = lower_block_rel_fro(H, spec.layer_slices)
            raw[f"eig_H_step{step}"] = e_h
            raw[f"topk_energy_H_step{step}"] = s_h["topk_energy_H"]
            if spec.kind == "snn":
                t2 = time.time()
                rs, re, im = raw_eig_stats(H, spec.layer_slices)
                rs["wall"] = time.time() - t2
                rec["rawH"] = rs
                raw[f"eig_Hraw_re_step{step}"] = re
                raw[f"eig_Hraw_im_step{step}"] = im
            rec["wall_H"] = time.time() - t1
        if verbose:
            act = rec["spikes"]["mean_rate"] if rec["spikes"] else float("nan")
            msg = (f"    step {step:4d} loss={loss_value:.4e} lam1(Ho)={rec['lambda1_Ho']:.3e} coh={rec['coh_Ho']['coherence']:.2f} "
                   f"ndom={rec['dominant_count_Ho']} r_split={rec['cluster_Ho']['ratio_split']:.2f} "
                   f"r_paper={rec['cluster_Ho']['ratio_paper']:.2f} rate={act:.3f} wall={time.time() - t0:.1f}s")
            if step in full_steps:
                msg += (f" | asym(H)={rec['asymmetry_H']:.2e} |Hf|/|Ho|={rec['Hf_over_Ho']:.3f} "
                        f"lam1(symH)={rec['lambda1_H']:.3e} wall_H={rec['wall_H']:.1f}s")
                if spec.kind == "snn":
                    msg += f" maxRe(H)={rec['rawH']['max_re']:.3e} ({rec['rawH']['method']}, {rec['rawH']['wall']:.1f}s)"
            print(msg, flush=True)
        return {"rec": rec}
    return cb


def run_one(kind: str, cond: str, L: int, seed: int, steps: int, ho_every: int, full_steps, res_dir: str,
            verbose=True) -> Dict[str, object]:
    X, Y, Wstar, data_info = make_data(cond, seed)
    spec = make_spec(kind, L)
    params, init_desc = make_init(spec, kind, seed)
    H_cache: Dict[int, torch.Tensor] = {}
    if kind == "linear":
        optimizer = "gd"
        H0 = sl.full_hessian(spec, params, X, Y)
        lam0 = float(sl.eigvals(H0)[0])
        lr = min(LR_GD_CAP, 1.0 / lam0)
        lr_desc = f"GD lr=min({LR_GD_CAP}, 1/lambda_max(H_init)={1.0 / lam0:.4g}) -> {lr:.4g}"
        H_cache[0] = H0
    else:
        optimizer, lr, lam0 = "adam", LR_ADAM, float("nan")
        lr_desc = f"Adam lr={lr}"
    name = f"{kind}_{cond}_L{L}_seed{seed}"
    info: Dict[str, object] = {"name": name, "kind": kind, "cond": cond, "cond_label": COND_LABEL[cond],
                               "kappa": data_info["kappa_realised"], "r": R, "d_in": D_IN, "d_out": D_OUT, "L": L,
                               "seed": seed, "init": init_desc, "P": spec.n_params, "steps": steps, "optimizer": optimizer,
                               "lr": lr, "lr_desc": lr_desc, "lambda_max_H_init": lam0, "spec": repr(spec), "n": N,
                               "width": WIDTH, "data": data_info}
    print(f"== run {name}: P={spec.n_params} kappa={data_info['kappa_realised']:.4g} trace={data_info['cov_trace']:.4g} "
          f"target_sv={np.round(data_info['target_sv'][:R], 3).tolist()} {init_desc} {lr_desc}", flush=True)
    raw: Dict[str, np.ndarray] = {}
    cb = make_callback(spec, X, Y, full_steps, raw, H_cache, verbose=verbose)
    t0 = time.time()
    hist = sl.gd_train(spec, params, X, Y, lr=lr, steps=steps, loss="mse", log_every=ho_every, callbacks=[cb],
                       optimizer=optimizer)
    wall = time.time() - t0
    checkpoints = [s["rec"] for s in hist["summaries"]]
    loss_curve = np.asarray(hist["loss"], dtype=np.float64)
    gn_curve = np.asarray(hist["grad_norm"], dtype=np.float64)
    info.update({
        "wall_sec": wall, "logged_steps": hist["logged_steps"],
        "loss_init": float(loss_curve[0]), "loss_final": float(loss_curve[-1]), "loss_min": float(np.nanmin(loss_curve)),
        "diverged": bool(not np.all(np.isfinite(loss_curve)) or loss_curve[-1] > 1e8),
        "loss_curve_every10": [float(x) for x in loss_curve[::10]] + [float(loss_curve[-1])],
        "checkpoints": checkpoints,
    })
    raw["loss_curve"] = loss_curve
    raw["grad_norm_curve"] = gn_curve
    raw["logged_steps"] = np.asarray(hist["logged_steps"])
    raw["Wstar"] = Wstar.numpy()
    raw["cov_eigs"] = np.asarray(data_info["cov_eigs"])
    os.makedirs(os.path.join(res_dir, "raw"), exist_ok=True)
    np.savez(os.path.join(res_dir, "raw", f"{name}.npz"), **raw)
    print(f"   done {name}: loss {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f}, wall {wall:.0f}s", flush=True)
    return to_py(info)


# --------------------------------------------------------------------------- #
# smoke-test validations
# --------------------------------------------------------------------------- #
def _trunc(o, maxlen=8):
    """Shorten long lists for printing."""
    if isinstance(o, dict):
        return {k: _trunc(v, maxlen) for k, v in o.items()}
    if isinstance(o, list):
        if len(o) > maxlen:
            return [_trunc(v, maxlen) for v in o[:maxlen]] + [f"... ({len(o)} items)"]
        return [_trunc(v, maxlen) for v in o]
    return o


def validate_once(kind: str, cond: str, L: int, seed: int = 0, print_dict: bool = True):
    """Print one full stats dict and cross-check the local helpers against the library functions."""
    X, Y, Wstar, dinfo = make_data(cond, seed)
    print(f"[validate {kind} {cond} L={L}] data: kappa_realised={dinfo['kappa_realised']:.6g} trace={dinfo['cov_trace']:.6g} "
          f"cov_eigs[:3]={np.round(dinfo['cov_eigs'][:3], 4).tolist()} cov_eigs[-3:]={np.round(dinfo['cov_eigs'][-3:], 4).tolist()} "
          f"target_sv={np.round(dinfo['target_sv'], 4).tolist()} cross_cov_sv={np.round(dinfo['cross_cov_sv'], 4).tolist()}")
    spec = make_spec(kind, L)
    p, _ = make_init(spec, kind, seed)
    raw: Dict[str, np.ndarray] = {}
    cb = make_callback(spec, X, Y, (0,), raw, {}, verbose=False)
    g = sl.surrogate_gradient(spec, p, X, Y)
    rec = cb(0, p, g, float(sl.loss_fn(spec, p, X, Y)))["rec"]
    if print_dict:
        print(f"[validate {kind} {cond} L={L}] FULL STATS DICT (lists truncated):")
        print(json.dumps(_trunc(to_py(rec)), indent=1))
    H_o = sl.gauss_newton(spec, p, X, Y)
    H = sl.full_hessian(spec, p, X, Y)
    # library cross-checks of the one-eigh summary
    ref_cs = sl.cluster_stats(sl.eigvals(H_o), R, D_IN, D_OUT)
    ref_coh = sl.layer_coherence(H_o, spec.layer_slices)
    ref_topk = sl.top_eigvec_layer_energy(H_o, spec.layer_slices, k=max(R * R, D_OUT)).numpy()
    for k_ in ref_cs:
        a, b = ref_cs[k_], rec["cluster_Ho"][k_]
        ok = (a == b) or (isinstance(a, float) and isinstance(b, float) and
                          ((math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1, abs(a))))
        assert ok, (k_, a, b)
    assert abs(ref_coh["coherence"] - rec["coh_Ho"]["coherence"]) < 1e-9
    assert np.allclose(ref_coh["lambda_max_blocks"], rec["coh_Ho"]["lambda_max_blocks"])
    assert np.allclose(ref_coh["top_eigvec_layer_energy"], rec["coh_Ho"]["top_eigvec_layer_energy"], atol=1e-8)
    assert np.allclose(ref_coh["block_alignment"].numpy(), rec["coh_Ho"]["block_alignment"], equal_nan=True)
    assert np.allclose(ref_topk, np.asarray(rec["topk_energy_Ho"])[: ref_topk.shape[0]], atol=1e-8)
    cs = rec["cluster_Ho"]
    assert cs["n_eigs"] == spec.n_params and cs["n_dominant_expected"] == R * R == 4
    assert cs["n_bulk_expected"] == R * (D_IN + D_OUT - 2 * R) == 36
    assert cs["largest_log_gap_index"] + 1 == cs["split_n_dominant"] == rec["dominant_count_Ho"]
    e_o = raw["eig_Ho_step0"]
    assert abs(sum(e_o) - float(torch.trace(H_o))) < 1e-8 * abs(float(torch.trace(H_o)))
    assert e_o[-1] > -1e-10 * e_o[0], "H_o not PSD?"
    assert sl.asymmetry(H_o) < 1e-12
    assert abs(rec["bulk_edge_paper_Ho"] - e_o[R * R]) == 0 and abs(rec["lambda1_Ho"] - e_o[0]) == 0
    # readout block of H_o is I_{d_out} kron M (M = a^T a / n): lambda_max == second_moment_top3[0]
    feats = rec["features"]
    assert abs(rec["coh_Ho"]["lambda_max_blocks"][-1] - feats["second_moment_top3"][0]) < 1e-8 * feats["second_moment_top3"][0]
    assert abs(rec["readout_block_lambda_max_Ho"] - rec["coh_Ho"]["lambda_max_blocks"][-1]) == 0
    # finite differences of the library H (eps below the distance to threshold for snn)
    eps = 1e-6
    if kind == "snn":
        eps = min(1e-6, 0.1 * rec["spikes"]["min_abs_v"])
    fd = sl.check_hessian_fd(spec, p, X, Y, H=H, n_dirs=2, eps=eps)
    print(f"[validate {kind} {cond} L={L}] finite-difference rel. errors of H (eps={eps:.1e}): {fd}")
    assert max(fd) < 1e-4, fd
    if kind == "linear":
        H_ref = sl.linear_gauss_newton_analytic(spec, p, X)  # uses Sigma = X^T X / n -> checks the imbalanced covariance path
        assert torch.allclose(H_o, H_ref, atol=1e-10, rtol=1e-10), "H_o != analytic linear GN with imbalanced Sigma"
        print(f"[validate {kind} {cond} L={L}] H_o == analytic (A^T A) kron (B Sigma B^T) with the imbalanced Sigma: OK")
        assert sl.asymmetry(H) < 1e-10
    if kind == "relu":
        assert sl.asymmetry(H) < 1e-10
    if kind == "snn":
        print(f"[validate {kind} {cond} L={L}] asym(H)={sl.asymmetry(H):.3f} lower_block_rel_fro={rec['lower_block_rel_fro_H']:.2e} "
              f"rawH={ {k: v for k, v in rec['rawH'].items() if k != 'top10_re'} }")
        # cross-check block-union raw eigenvalues against the dense non-symmetric eigvals
        rs_full, re_full, im_full = raw_eig_stats(H, spec.layer_slices, force_full=True)
        re_blk = raw["eig_Hraw_re_step0"]
        scale = max(abs(re_full).max(), 1e-300)
        print(f"[validate {kind} {cond} L={L}] dense eigvals: max_re={rs_full['max_re']:.6e} (block-union {rec['rawH']['max_re']:.6e}), "
              f"wall dense={rs_full.get('wall', float('nan'))} max|re diff|/scale={np.abs(np.sort(re_full) - np.sort(re_blk)).max() / scale:.2e}")
        assert np.abs(np.sort(re_full) - np.sort(re_blk)).max() < 1e-6 * scale
        # readout block of the raw H equals I_{d_out} kron M (theory note 5 (ii))
        with torch.no_grad():
            a = spec.hidden_features(p, X)
        M = a.T @ a / N
        sL = spec.layer_slices[-1]
        blk = H[sL, sL]
        ref_blk = torch.kron(torch.eye(D_OUT, dtype=M.dtype), M)
        err = float(torch.linalg.norm(blk - ref_blk) / torch.linalg.norm(ref_blk))
        print(f"[validate {kind} {cond} L={L}] ||H[L,L] - I kron Sigma_a|| / ||.|| = {err:.2e}; "
              f"readout-block top eig raw H = {rec['rawH']['readout_block_eigs_top3'][0]:.6g} vs lambda_max(M) = {feats['second_moment_top3'][0]:.6g}")
        assert err < 1e-8
    print(f"[validate {kind} {cond} L={L}] all cross-checks passed for {spec}", flush=True)


def timing_probe(kind="snn", L=8, cond="k100"):
    X, Y, _, _ = make_data(cond, 0)
    spec = make_spec(kind, L)
    p, _ = make_init(spec, kind, 0)
    t = time.time(); H_o = sl.gauss_newton(spec, p, X, Y); t_ho = time.time() - t
    t = time.time(); H = sl.full_hessian(spec, p, X, Y); t_h = time.time() - t
    t = time.time(); spectrum_summary(H, spec, "H"); t_e = time.time() - t
    t = time.time(); rs, _, _ = raw_eig_stats(H, spec.layer_slices); t_r = time.time() - t
    t = time.time(); sl.gd_train(spec, p, X, Y, lr=LR_ADAM, steps=20, optimizer="adam"); t_tr = (time.time() - t) / 21 * STEPS
    print(f"[timing {kind} L={L} P={spec.n_params}] H_o {t_ho:.1f}s, full H {t_h:.1f}s, eigh+summary {t_e:.1f}s, "
          f"raw eig ({rs['method']}) {t_r:.1f}s, {STEPS} train steps est {t_tr:.0f}s -> est run "
          f"{t_tr + 2 * (t_h + t_e + t_r) + 5 * (t_ho + t_e):.0f}s (EST_RUN_SEC={EST_RUN_SEC[kind][L]})", flush=True)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
METRICS = ["ratio_paper", "ratio_split", "split_n_dominant", "coherence", "lambda1", "bulk_edge_paper", "bulk_edge_split",
           "split_bulk_mean", "bulk_mean", "readout_block_lmax", "first_block_lmax", "w1_energy_topr2", "wL_energy_topr2",
           "w1_energy_top1", "wL_energy_top1", "max_re_raw", "Hf_over_Ho", "asymmetry_H", "mu_sq_norm", "cov_top1", "mean_rate",
           "window_frac", "loss", "neg_frac", "zero_frac"]


def rec_val(rec: Optional[dict], mat: str, metric: str) -> float:
    if rec is None or rec.get("diverged"):
        return float("nan")
    cs = rec.get(f"cluster_{mat}")
    if cs is None:
        return float("nan")
    if metric in cs:
        return fnum(cs[metric])
    if metric == "coherence":
        return fnum(rec[f"coh_{mat}"]["coherence"])
    if metric == "lambda1":
        return fnum(rec[f"lambda1_{mat}"])
    if metric in ("bulk_edge_paper", "bulk_edge_split"):
        return fnum(rec[f"{metric}_{mat}"])
    if metric == "readout_block_lmax":
        return fnum(rec[f"readout_block_lambda_max_{mat}"])
    if metric == "first_block_lmax":
        return fnum(rec[f"first_block_lambda_max_{mat}"])
    if metric in ("neg_frac", "zero_frac"):
        return fnum(rec[f"{metric}_{mat}"])
    topk = np.asarray(rec[f"topk_energy_{mat}"], dtype=float)
    if metric == "w1_energy_topr2":
        return float(topk[: R * R, 0].mean())
    if metric == "wL_energy_topr2":
        return float(topk[: R * R, -1].mean())
    if metric == "w1_energy_top1":
        return float(topk[0, 0])
    if metric == "wL_energy_top1":
        return float(topk[0, -1])
    if metric == "max_re_raw":
        return fnum(rec["rawH"]["max_re"]) if "rawH" in rec else float("nan")
    if metric == "Hf_over_Ho":
        return fnum(rec.get("Hf_over_Ho"))
    if metric == "asymmetry_H":
        return fnum(rec.get("asymmetry_H"))
    if metric == "mu_sq_norm":
        return fnum(rec["features"]["mu_sq_norm"])
    if metric == "cov_top1":
        return fnum(rec["features"]["cov_top3"][0])
    if metric == "mean_rate":
        return fnum(rec["spikes"]["mean_rate"]) if rec.get("spikes") else float("nan")
    if metric == "window_frac":
        return fnum(rec["spikes"]["window_frac"]) if rec.get("spikes") else float("nan")
    if metric == "loss":
        return fnum(rec["loss"])
    raise KeyError(metric)


def stage_rec(run: dict, stage: str) -> Optional[dict]:
    cps = run["checkpoints"]
    if not cps:
        return None
    if stage == "init":
        return cps[0] if cps[0]["step"] == 0 else None
    if stage == "end":
        return cps[-1] if cps[-1]["step"] == run["steps"] else None
    raise ValueError(stage)


def aggregate(summary: List[dict]):
    """agg[kind][cond][L][stage][mat][metric] = {"mean", "std", "n", "values"}; plus per-run data info."""
    kinds = sorted({r["kind"] for r in summary}, key=KINDS.index)
    conds = [c for c in CONDS if any(r["cond"] == c for r in summary)]
    Ls = sorted({r["L"] for r in summary})
    agg: Dict = {}
    for kind in kinds:
        agg[kind] = {}
        for cond in conds:
            agg[kind][cond] = {}
            for L in Ls:
                runs = [r for r in summary if r["kind"] == kind and r["cond"] == cond and r["L"] == L]
                agg[kind][cond][L] = {"n_runs": len(runs), "seeds": [r["seed"] for r in runs],
                                     "loss_final": [r["loss_final"] for r in runs], "lr": [r["lr"] for r in runs]}
                for stage in ("init", "end"):
                    agg[kind][cond][L][stage] = {}
                    for mat in ("Ho", "H"):
                        d = {}
                        for m in METRICS:
                            vals = np.array([rec_val(stage_rec(r, stage), mat, m) for r in runs], dtype=float)
                            good = vals[np.isfinite(vals)]
                            d[m] = {"mean": float(good.mean()) if good.size else float("nan"),
                                    "std": float(good.std()) if good.size > 1 else 0.0 if good.size else float("nan"),
                                    "n": int(good.size), "values": [float(v) for v in vals]}
                        agg[kind][cond][L][stage][mat] = d
    return agg, kinds, conds, Ls


def fit_slope(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or len(np.unique(x[m])) < 2:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def slopes(agg, kinds, conds, Ls) -> Dict:
    """Depth slopes (metric vs L at fixed condition) and data slopes (metric vs log10 kappa at fixed L)."""
    out: Dict = {}
    kconds = [c for c in conds if c in KAPPA]
    for kind in kinds:
        out[kind] = {}
        for mat in ("Ho", "H"):
            out[kind][mat] = {}
            for stage in ("init", "end"):
                o = {"depth": {}, "depth_loglog": {}, "data": {}, "data_loglog": {}}
                for m in ("ratio_paper", "ratio_split", "coherence", "lambda1", "bulk_edge_split", "bulk_edge_paper", "w1_energy_topr2", "split_n_dominant"):
                    o["depth"][m] = {}; o["depth_loglog"][m] = {}; o["data"][m] = {}; o["data_loglog"][m] = {}
                    for cond in conds:
                        xs, ys = [], []
                        for L in Ls:
                            for v in agg[kind][cond][L][stage][mat][m]["values"]:
                                xs.append(L); ys.append(v)
                        o["depth"][m][cond] = fit_slope(xs, ys)
                        o["depth_loglog"][m][cond] = fit_slope(np.log(xs), np.log(np.maximum(ys, 1e-300)))
                    pooled_x, pooled_y = [], []
                    for L in Ls:
                        xs, ys = [], []
                        for cond in kconds:
                            for v in agg[kind][cond][L][stage][mat][m]["values"]:
                                xs.append(math.log10(KAPPA[cond])); ys.append(v)
                        o["data"][m][f"L{L}"] = fit_slope(xs, ys)
                        o["data_loglog"][m][f"L{L}"] = fit_slope(np.array(xs) * math.log(10.0), np.log(np.maximum(ys, 1e-300)))
                        pooled_x += xs; pooled_y += ys
                    o["data"][m]["pooled"] = fit_slope(pooled_x, pooled_y)
                    o["data_loglog"][m]["pooled"] = fit_slope(np.array(pooled_x) * math.log(10.0), np.log(np.maximum(pooled_y, 1e-300)))
                out[kind][mat][stage] = o
    return out


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    ax = abs(x)
    if ax != 0 and (ax >= 1e4 or ax < 1e-2):
        return f"{x:.{nd}e}"
    return f"{x:.{nd}f}"


def mean_std(agg_entry, nd=2):
    m, s, n = agg_entry["mean"], agg_entry["std"], agg_entry["n"]
    if n == 0:
        return "nan"
    return f"{fmt(m, nd)}±{fmt(s, nd)}" if n > 1 else fmt(m, nd)


def make_tables(agg, kinds, conds, Ls, slp, summary) -> str:
    lines: List[str] = []
    L_focus = 4 if 4 in Ls else Ls[len(Ls) // 2]
    # data conditions realised
    lines.append("### Data conditions (realised, seed 0)\n")
    lines.append("| cond | kappa(X^T X/n) | trace | top-3 cov eigs | bottom-3 | target sv | cross-cov sv (Sigma Wstar^T) |")
    lines.append("|---|---|---|---|---|---|---|")
    for cond in conds:
        r0 = next(r for r in summary if r["cond"] == cond and r["seed"] == 0)
        d = r0["data"]
        lines.append(f"| {COND_LABEL[cond]} | {fmt(d['kappa_realised'])} | {fmt(d['cov_trace'])} | {[fmt(v) for v in d['cov_eigs'][:3]]} | "
                     f"{[fmt(v) for v in d['cov_eigs'][-3:]]} | {[fmt(v) for v in d['target_sv'][:R]]} | {[fmt(v) for v in d['cross_cov_sv'][:R]]} |")
    for kind in kinds:
        for mat in ("Ho", "H"):
            matname = "H_o" if mat == "Ho" else "sym(H)"
            # (c) two-way tables L x cond: ratio_split and coherence (end of training and init)
            for stage in ("init", "end"):
                for m in ("ratio_split", "coherence", "ratio_paper", "split_n_dominant"):
                    lines.append(f"\n### {kind}: {m} of {matname} at {stage} (rows L, cols condition; mean±std over seeds)\n")
                    lines.append("| L | " + " | ".join(COND_LABEL[c] for c in conds) + " |")
                    lines.append("|---|" + "---|" * len(conds))
                    for L in Ls:
                        lines.append(f"| {L} | " + " | ".join(mean_std(agg[kind][c][L][stage][mat][m]) for c in conds) + " |")
        # (a)/(b) at L = focus: per condition summary of H_o and sym(H)
        for stage in ("init", "end"):
            lines.append(f"\n### {kind}: L={L_focus}, {stage}: per-condition summary (H_o | sym(H))\n")
            cols = ["ratio_paper", "ratio_split", "split_n_dominant", "coherence", "lambda1", "bulk_edge_paper", "bulk_edge_split", "w1_energy_topr2", "wL_energy_topr2"]
            lines.append("| cond | " + " | ".join(cols) + " | loss | " + ("mean_rate | maxRe(rawH) | " if kind == "snn" else "") + "|Hf|/|Ho| |")
            lines.append("|---|" + "---|" * (len(cols) + 1 + (2 if kind == "snn" else 0) + 1))
            for cond in conds:
                e = agg[kind][cond][L_focus][stage]
                cells = [f"{mean_std(e['Ho'][m])} \\| {mean_std(e['H'][m])}" for m in cols]
                extra = ""
                if kind == "snn":
                    extra = f"{mean_std(e['Ho']['mean_rate'], 3)} | {mean_std(e['H']['max_re_raw'])} | "
                lines.append(f"| {COND_LABEL[cond]} | " + " | ".join(cells) + f" | {mean_std(e['Ho']['loss'], 3)} | {extra}{mean_std(e['H']['Hf_over_Ho'])} |")
        # slopes
        lines.append(f"\n### {kind}: slopes (H_o): depth slope d(metric)/dL at fixed condition; data slope d(metric)/dlog10(kappa) at fixed L\n")
        lines.append("| stage | metric | " + " | ".join(f"depth@{COND_LABEL[c]}" for c in conds) + " | " + " | ".join(f"data@L{L}" for L in Ls) + " | data pooled |")
        lines.append("|---|---|" + "---|" * (len(conds) + len(Ls) + 1))
        for stage in ("init", "end"):
            for m in ("ratio_paper", "ratio_split", "coherence"):
                s = slp[kind]["Ho"][stage]
                lines.append(f"| {stage} | {m} | " + " | ".join(fmt(s["depth"][m][c]) for c in conds) + " | "
                             + " | ".join(fmt(s["data"][m][f"L{L}"]) for L in Ls) + f" | {fmt(s['data'][m]['pooled'])} |")
        lines.append(f"\n### {kind}: log-log exponents (H_o): d log(metric)/d log L at fixed condition; d log(metric)/d log kappa at fixed L (paper: ratio ~ L^1, kappa^0)\n")
        lines.append("| stage | metric | " + " | ".join(f"depth@{COND_LABEL[c]}" for c in conds) + " | " + " | ".join(f"data@L{L}" for L in Ls) + " | data pooled |")
        lines.append("|---|---|" + "---|" * (len(conds) + len(Ls) + 1))
        for stage in ("init", "end"):
            for m in ("ratio_paper", "ratio_split", "coherence", "lambda1", "bulk_edge_split"):
                s = slp[kind]["Ho"][stage]
                lines.append(f"| {stage} | {m} | " + " | ".join(fmt(s["depth_loglog"][m][c]) for c in conds) + " | "
                             + " | ".join(fmt(s["data_loglog"][m][f"L{L}"]) for L in Ls) + f" | {fmt(s['data_loglog'][m]['pooled'])} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(res_dir: str, fig_prefix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(res_dir, "summary.json")) as f:
        summary = json.load(f)
    agg, kinds, conds, Ls = aggregate(summary)
    slp = slopes(agg, kinds, conds, Ls)
    L_focus = 4 if 4 in Ls else Ls[len(Ls) // 2]
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    figs: List[str] = []
    cond_x = {c: i for i, c in enumerate(conds)}

    def savefig(fig, what):
        path = os.path.join(fig_dir, f"{fig_prefix}_{what}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figs.append(path)

    def set_cond_axis(ax):
        ax.set_xticks([cond_x[c] for c in conds])
        ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=35, fontsize=8, ha="right")

    def series_vs_cond(kind, L, stage, mat, m):
        means = np.array([agg[kind][c][L][stage][mat][m]["mean"] for c in conds])
        stds = np.array([agg[kind][c][L][stage][mat][m]["std"] for c in conds])
        vals = [agg[kind][c][L][stage][mat][m]["values"] for c in conds]
        return np.array([cond_x[c] for c in conds]), means, stds, vals

    def series_vs_L(kind, cond, stage, mat, m):
        means = np.array([agg[kind][cond][L][stage][mat][m]["mean"] for L in Ls])
        stds = np.array([agg[kind][cond][L][stage][mat][m]["std"] for L in Ls])
        vals = [agg[kind][cond][L][stage][mat][m]["values"] for L in Ls]
        return np.array(Ls, float), means, stds, vals

    def draw(ax, x, means, stds, vals, color, ls, label, marker="o"):
        n_seed = max(len(v) for v in vals) if vals else 0
        for s in range(n_seed):
            ys = np.array([v[s] if s < len(v) else np.nan for v in vals])
            ax.plot(x, ys, color=color, ls=ls, lw=0.6, alpha=0.4)
        ax.errorbar(x, means, yerr=stds, color=color, ls=ls, marker=marker, ms=4, lw=1.6, capsize=2, label=label)

    main_metrics = [("ratio_paper", "ratio_paper = mean(top r²)/mean(next r(d_in+d_out-2r))"),
                    ("ratio_split", "ratio_split (data-driven log-gap split)"),
                    ("split_n_dominant", "split_n_dominant (count above largest log gap)"),
                    ("coherence", "coherence = λmax(H_o)/max_l λmax(block l)")]

    for kind in kinds:
        # (a1) metrics vs condition at L = L_focus ------------------------------------------------
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
        for i, stage in enumerate(("init", "end")):
            for j, (m, title) in enumerate(main_metrics):
                ax = axes[i, j]
                draw(ax, *series_vs_cond(kind, L_focus, stage, "Ho", m), color=KIND_COL[kind], ls="-", label="H_o")
                draw(ax, *series_vs_cond(kind, L_focus, stage, "H", m), color="k", ls="--", label="sym(H)", marker="s")
                if m in ("ratio_paper", "ratio_split", "coherence"):
                    ax.axhline(L_focus, color="gray", ls=":", lw=1, label=f"L={L_focus} (paper)")
                if m == "split_n_dominant":
                    ax.axhline(R * R, color="gray", ls=":", lw=1, label=f"r²={R * R}")
                    ax.axhline(D_OUT, color="gray", ls="-.", lw=1, label=f"d_out={D_OUT}")
                set_cond_axis(ax)
                ax.set_title(f"{kind} L={L_focus} {stage}: {m}", fontsize=9)
                ax.set_ylabel(title, fontsize=7)
                ax.grid(alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(f"{EXP}: {kind}, L={L_focus}: cluster ratios / count / coherence vs data condition (mean±std over seeds, thin = seeds)")
        fig.tight_layout()
        savefig(fig, f"metrics_vs_cond_L{L_focus}_{kind}")

        # (a2) metrics vs L at kappa=1 vs 100 (and 1000, sv, scale25 thin) ---------------------------
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
        for i, stage in enumerate(("init", "end")):
            for j, (m, title) in enumerate(main_metrics):
                ax = axes[i, j]
                for cond in conds:
                    if cond in ("k1", "k100"):
                        draw(ax, *series_vs_L(kind, cond, stage, "Ho", m), color=COND_COL[cond], ls="-", label=f"H_o {COND_LABEL[cond]}")
                        x, me, sd, _ = series_vs_L(kind, cond, stage, "H", m)
                        ax.plot(x, me, color=COND_COL[cond], ls="--", marker="s", ms=3, lw=1, label=f"sym(H) {COND_LABEL[cond]}")
                    else:
                        x, me, sd, _ = series_vs_L(kind, cond, stage, "Ho", m)
                        ax.plot(x, me, color=COND_COL[cond], ls=":", marker="^", ms=3, lw=1, alpha=0.8, label=f"H_o {COND_LABEL[cond]}")
                if m in ("ratio_paper", "ratio_split", "coherence"):
                    ax.plot(Ls, Ls, color="gray", ls=":", lw=1, label="= L (paper)")
                if m == "split_n_dominant":
                    ax.axhline(R * R, color="gray", ls=":", lw=1); ax.axhline(D_OUT, color="gray", ls="-.", lw=1)
                ax.set_xlabel("L"); ax.set_xticks(Ls)
                ax.set_title(f"{kind} {stage}: {m} vs L", fontsize=9)
                ax.set_ylabel(title, fontsize=7)
                ax.grid(alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"{EXP}: {kind}: metrics vs depth L at κ=1 vs κ=100 (solid: H_o; dashed: sym(H); dotted: other conditions)")
        fig.tight_layout()
        savefig(fig, f"metrics_vs_L_{kind}")

        # (b) top eigenvalue and bulk edge vs condition -----------------------------------------------
        fig, axes = plt.subplots(2, len(Ls), figsize=(5.2 * len(Ls), 7.5), squeeze=False)
        for i, stage in enumerate(("init", "end")):
            for j, L in enumerate(Ls):
                ax = axes[i, j]
                for m, col, ls, lab in (("lambda1", "tab:red", "-", "λ1(H_o)"),
                                        ("bulk_edge_paper", "tab:blue", "-", f"λ_{R * R + 1}(H_o) (paper bulk edge)"),
                                        ("bulk_edge_split", "tab:green", "-", "λ_{k+1}(H_o) (split bulk edge)"),
                                        ("split_bulk_mean", "tab:olive", "--", "split bulk mean (H_o)"),
                                        ("bulk_mean", "tab:purple", "--", "paper bulk mean (H_o)"),
                                        ("readout_block_lmax", "tab:brown", ":", "λmax(readout block H_o)"),
                                        ("first_block_lmax", "tab:pink", ":", "λmax(W1 block H_o)")):
                    x, me, sd, vals = series_vs_cond(kind, L, stage, "Ho", m)
                    draw(ax, x, me, sd, vals, color=col, ls=ls, label=lab)
                x, me, sd, vals = series_vs_cond(kind, L, stage, "H", "lambda1")
                ax.plot(x, me, color="k", ls="--", marker="s", ms=3, lw=1, label="λ1(sym H)")
                if kind == "snn":
                    x, me, sd, vals = series_vs_cond(kind, L, stage, "H", "max_re_raw")
                    ax.plot(x, me, color="gray", ls="-.", marker="x", ms=4, lw=1, label="max Re eig(raw H)")
                ax.set_yscale("log"); set_cond_axis(ax); ax.grid(alpha=0.3, which="both")
                ax.set_title(f"{kind} L={L} {stage}", fontsize=9)
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.suptitle(f"{EXP}: {kind}: top eigenvalue and bulk edges vs data condition")
        fig.tight_layout()
        savefig(fig, f"top_bulk_vs_cond_{kind}")

        # (d1) layer energy of the top-k eigenvectors of H_o: grid L x cond (stacked bars, seed mean) ---
        k_show = max(R * R, D_OUT)
        for stage in ("init", "end"):
            fig, axes = plt.subplots(len(Ls), len(conds), figsize=(2.6 * len(conds), 2.3 * len(Ls)), squeeze=False, sharey=True)
            for i, L in enumerate(Ls):
                cmap = plt.get_cmap(LAYER_CMAP, L)
                for j, cond in enumerate(conds):
                    ax = axes[i, j]
                    runs = [r for r in summary if r["kind"] == kind and r["cond"] == cond and r["L"] == L]
                    mats = []
                    for r in runs:
                        rec = stage_rec(r, stage)
                        if rec is not None and not rec.get("diverged"):
                            mats.append(np.asarray(rec["topk_energy_Ho"], float)[:k_show])
                    if not mats:
                        ax.set_axis_off(); continue
                    E = np.mean(mats, axis=0)  # (k_show, L)
                    bottom = np.zeros(k_show)
                    for l in range(L):
                        ax.bar(np.arange(1, k_show + 1), E[:, l], bottom=bottom, color=cmap(l), width=0.8,
                               label=f"W{l + 1}" if l in (0, L - 1) else None, edgecolor="none")
                        bottom += E[:, l]
                    ax.axvline(R * R + 0.5, color="k", ls=":", lw=0.8)
                    ax.set_ylim(0, 1); ax.set_xticks(range(1, k_show + 1)); ax.tick_params(labelsize=7)
                    ax.set_title(f"L={L} {COND_LABEL[cond]}", fontsize=8)
                    if j == 0:
                        ax.set_ylabel("energy fraction", fontsize=7)
                        ax.legend(fontsize=6, loc="lower left")
                    if i == len(Ls) - 1:
                        ax.set_xlabel("eigenvector index", fontsize=7)
            fig.suptitle(f"{EXP}: {kind}, {stage}: per-layer energy of the top-{k_show} eigenvectors of H_o (dark=W1 ... light=W_L; dotted line = r²)")
            fig.tight_layout()
            savefig(fig, f"layer_energy_{kind}_{stage}")

        # (d2) W1 / W_L energy of the top-r^2 eigenvectors vs condition -----------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for i, stage in enumerate(("init", "end")):
            for j, (m, lab) in enumerate((("w1_energy_topr2", "mean W1 energy of top-r² eigvecs"), ("wL_energy_topr2", "mean W_L (readout) energy of top-r² eigvecs"))):
                ax = axes[i, j]
                for L in Ls:
                    draw(ax, *series_vs_cond(kind, L, stage, "Ho", m), color=L_COL.get(L, "k"), ls="-", label=f"H_o L={L}")
                    x, me, sd, _ = series_vs_cond(kind, L, stage, "H", m)
                    ax.plot(x, me, color=L_COL.get(L, "k"), ls="--", marker="s", ms=3, lw=1, label=f"sym(H) L={L}")
                for L in Ls:
                    ax.axhline(1.0 / L, color=L_COL.get(L, "k"), ls=":", lw=0.7)
                set_cond_axis(ax); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
                ax.set_title(f"{kind} {stage}: {lab}", fontsize=9)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7, ncol=2)
        fig.suptitle(f"{EXP}: {kind}: does covariance imbalance move the top eigenvectors into W1? (dotted: 1/L = uniform)")
        fig.tight_layout()
        savefig(fig, f"w1_energy_vs_cond_{kind}")

        # spectra (seed 0, from raw npz): rows L, cols init/end; H_o eigenvalues per condition -----------
        fig, axes = plt.subplots(len(Ls), 2, figsize=(12, 3.4 * len(Ls)), squeeze=False)
        for i, L in enumerate(Ls):
            for j, stage in enumerate(("init", "end")):
                ax = axes[i, j]
                for cond in conds:
                    runs = [r for r in summary if r["kind"] == kind and r["cond"] == cond and r["L"] == L and r["seed"] == 0]
                    if not runs:
                        continue
                    r = runs[0]
                    step = 0 if stage == "init" else r["steps"]
                    path = os.path.join(res_dir, "raw", f"{r['name']}.npz")
                    if not os.path.exists(path):
                        continue
                    z = np.load(path)
                    key = f"eig_Ho_step{step}"
                    if key not in z:
                        continue
                    e = z[key]
                    pos = e > 0
                    ax.plot(np.arange(1, len(e) + 1)[pos], e[pos], color=COND_COL[cond], lw=1.2, label=f"H_o {COND_LABEL[cond]}")
                    keyh = f"eig_H_step{step}"
                    if keyh in z and cond in ("k1", "k100"):
                        eh = z[keyh]; posh = eh > 0
                        ax.plot(np.arange(1, len(eh) + 1)[posh], eh[posh], color=COND_COL[cond], lw=0.8, ls="--", label=f"sym(H) {COND_LABEL[cond]}")
                ax.axvline(R * R, color="k", ls=":", lw=0.8); ax.axvline(R * R + R * (D_IN + D_OUT - 2 * R), color="k", ls="-.", lw=0.8)
                ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
                ax.set_title(f"{kind} L={L} {stage} (seed 0; dotted r², dash-dot r²+bulk)", fontsize=9)
                ax.set_xlabel("index"); ax.set_ylabel("eigenvalue")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"{EXP}: {kind}: positive eigenvalues of H_o (and sym(H) for κ=1, κ=100) by condition")
        fig.tight_layout()
        savefig(fig, f"spectra_{kind}")

    # snn raw-H diagnostics ------------------------------------------------------------------------------
    if "snn" in kinds:
        fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
        for i, stage in enumerate(("init", "end")):
            ax = axes[i, 0]
            for L in Ls:
                draw(ax, *series_vs_cond("snn", L, stage, "H", "max_re_raw"), color=L_COL.get(L, "k"), ls="-", label=f"max Re eig(raw H) L={L}")
                x, me, sd, _ = series_vs_cond("snn", L, stage, "H", "lambda1")
                ax.plot(x, me, color=L_COL.get(L, "k"), ls="--", marker="s", ms=3, lw=1, label=f"λ1(sym H) L={L}")
                x, me, sd, _ = series_vs_cond("snn", L, stage, "Ho", "lambda1")
                ax.plot(x, me, color=L_COL.get(L, "k"), ls=":", marker="^", ms=3, lw=1, label=f"λ1(H_o) L={L}")
            ax.set_yscale("log"); set_cond_axis(ax); ax.grid(alpha=0.3, which="both"); ax.set_title(f"snn {stage}: top eigenvalues", fontsize=9)
            if i == 0:
                ax.legend(fontsize=6, ncol=3)
            ax = axes[i, 1]
            for L in Ls:
                draw(ax, *series_vs_cond("snn", L, stage, "Ho", "readout_block_lmax"), color=L_COL.get(L, "k"), ls="-", label=f"λmax(readout block H_o) L={L}")
                x, me, sd, _ = series_vs_cond("snn", L, stage, "Ho", "mu_sq_norm")
                ax.plot(x, me, color=L_COL.get(L, "k"), ls="--", marker="s", ms=3, lw=1, label=f"||mu||² (rates) L={L}")
                x, me, sd, _ = series_vs_cond("snn", L, stage, "Ho", "cov_top1")
                ax.plot(x, me, color=L_COL.get(L, "k"), ls=":", marker="^", ms=3, lw=1, label=f"λ1(Cov_a) L={L}")
            ax.set_yscale("log"); set_cond_axis(ax); ax.grid(alpha=0.3, which="both"); ax.set_title(f"snn {stage}: readout-feature second moments", fontsize=9)
            if i == 0:
                ax.legend(fontsize=6, ncol=2)
            ax = axes[i, 2]
            for L in Ls:
                draw(ax, *series_vs_cond("snn", L, stage, "Ho", "mean_rate"), color=L_COL.get(L, "k"), ls="-", label=f"mean rate L={L}")
                x, me, sd, _ = series_vs_cond("snn", L, stage, "Ho", "window_frac")
                ax.plot(x, me, color=L_COL.get(L, "k"), ls="--", marker="s", ms=3, lw=1, label=f"window frac L={L}")
                x, me, sd, _ = series_vs_cond("snn", L, stage, "H", "asymmetry_H")
                ax.plot(x, me, color=L_COL.get(L, "k"), ls=":", marker="^", ms=3, lw=1, label=f"asym(H) L={L}")
            set_cond_axis(ax); ax.grid(alpha=0.3); ax.set_title(f"snn {stage}: spike diagnostics / asymmetry", fontsize=9)
            if i == 0:
                ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"{EXP}: snn raw (non-symmetric) H, readout block and spike statistics vs data condition")
        fig.tight_layout()
        savefig(fig, "snn_rawH")

    # cross-kind summary --------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, m in enumerate(("ratio_split", "coherence")):
        ax = axes[i, 0]
        for kind in kinds:
            draw(ax, *series_vs_cond(kind, L_focus, "end", "Ho", m), color=KIND_COL[kind], ls="-", label=f"{kind} end")
            x, me, sd, _ = series_vs_cond(kind, L_focus, "init", "Ho", m)
            ax.plot(x, me, color=KIND_COL[kind], ls="--", marker="s", ms=3, lw=1, label=f"{kind} init")
        ax.axhline(L_focus, color="gray", ls=":", lw=1)
        set_cond_axis(ax); ax.grid(alpha=0.3); ax.set_title(f"{m} of H_o vs condition, L={L_focus}", fontsize=9); ax.legend(fontsize=7, ncol=2)
        for j, cond in enumerate(("k1", "k100")):
            ax = axes[i, 1 + j]
            if cond not in conds:
                ax.set_axis_off(); continue
            for kind in kinds:
                draw(ax, *series_vs_L(kind, cond, "end", "Ho", m), color=KIND_COL[kind], ls="-", label=f"{kind} end")
                x, me, sd, _ = series_vs_L(kind, cond, "init", "Ho", m)
                ax.plot(x, me, color=KIND_COL[kind], ls="--", marker="s", ms=3, lw=1, label=f"{kind} init")
            ax.plot(Ls, Ls, color="gray", ls=":", lw=1, label="= L")
            ax.set_xticks(Ls); ax.set_xlabel("L"); ax.grid(alpha=0.3); ax.set_title(f"{m} of H_o vs L at {COND_LABEL[cond]}", fontsize=9)
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{EXP}: depth vs data imbalance across kinds (H_o; solid end of training, dashed init)")
    fig.tight_layout()
    savefig(fig, "summary_kinds")

    # loss curves ---------------------------------------------------------------------------------------
    fig, axes = plt.subplots(len(kinds), len(Ls), figsize=(5 * len(Ls), 3.3 * len(kinds)), squeeze=False)
    for i, kind in enumerate(kinds):
        for j, L in enumerate(Ls):
            ax = axes[i, j]
            for cond in conds:
                runs = [r for r in summary if r["kind"] == kind and r["cond"] == cond and r["L"] == L]
                for r in runs:
                    lc = np.asarray(r["loss_curve_every10"], float)
                    xs = np.arange(len(lc)) * 10
                    xs[-1] = r["steps"]
                    ax.plot(xs, lc, color=COND_COL[cond], lw=0.9, alpha=0.9 if r["seed"] == 0 else 0.5,
                            label=COND_LABEL[cond] if r["seed"] == 0 else None)
            ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.set_title(f"{kind} L={L}", fontsize=9)
            ax.set_xlabel("step"); ax.set_ylabel("MSE loss")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"{EXP}: training loss curves by data condition (seed 0 bold, seed 1 faint)")
    fig.tight_layout()
    savefig(fig, "loss_curves")

    # analysis + tables ---------------------------------------------------------------------------------
    tables = make_tables(agg, kinds, conds, Ls, slp, summary)
    with open(os.path.join(res_dir, "tables.md"), "w") as f:
        f.write(tables)
    with open(os.path.join(res_dir, "analysis.json"), "w") as f:
        json.dump(to_py({"agg": agg, "slopes": slp, "kinds": kinds, "conds": conds, "Ls": Ls, "L_focus": L_focus, "figures": figs}), f, indent=1)
    print(f"FIGURES: {len(figs)} written; tables -> {os.path.join(res_dir, 'tables.md')}", flush=True)
    return figs


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures", action="store_true", help="only regenerate figures/tables from saved results")
    ap.add_argument("--no-timing", action="store_true", help="smoke: skip the L=8 timing probe")
    ap.add_argument("--budget", type=float, default=None, help="override BUDGET_MIN (minutes)")
    args = ap.parse_args()
    budget = args.budget if args.budget is not None else BUDGET_MIN
    exp = EXP + ("_smoke" if args.smoke else "")
    res_dir = os.path.join(ROOT, "results", exp)
    os.makedirs(res_dir, exist_ok=True)
    if args.figures:
        make_figures(res_dir, exp)
        return
    T0 = time.time()
    if args.smoke:
        steps, ho_every, full_steps = 40, 20, (0, 40)
        Ls_run, seeds, conds = [2, 3], [0], CONDS
        validate_once("snn", "k100", 3)
        validate_once("linear", "k1000", 3, print_dict=False)
        validate_once("relu", "scale25", 2, print_dict=False)
        validate_once("snn", "sv", 2, print_dict=False)
        if not args.no_timing:
            timing_probe("snn", 8, "k100")
        plan = [(k, c, L, s) for s in seeds for c in conds for k in KINDS for L in Ls_run]
        tiers = [(0, plan)]
    else:
        steps, ho_every, full_steps = STEPS, HO_EVERY, FULL_H_STEPS
        tier0 = [(k, c, L, 0) for c in CONDS for k in KINDS for L in LS]                 # seed 0, everything
        tier0 += [(k, c, L, 1) for c in CONDS for k in ("relu", "linear") for L in LS]   # cheap seed 1
        tier1 = [("snn", c, L, 1) for c in CONDS for L in (2, 4)]                          # snn seed 1, L=2,4
        tier2 = [("snn", c, 8, 1) for c in CONDS]                                          # snn seed 1, L=8
        tiers = [(0, tier0), (1, tier1), (2, tier2)]
    summary: List[dict] = []
    skipped: List[str] = []
    prior_min = 0.0  # wall time already spent before a resume (container restarts): counts against the budget
    done_names = set()
    if not args.smoke and os.path.exists(os.path.join(res_dir, "summary.json")):
        with open(os.path.join(res_dir, "summary.json")) as f:
            summary = json.load(f)
        done_names = {r["name"] for r in summary}
        if os.path.exists(os.path.join(res_dir, "run_meta.json")):
            with open(os.path.join(res_dir, "run_meta.json")) as f:
                prior_min = float(json.load(f).get("elapsed_min", 0.0))
        print(f"RESUME: {len(done_names)} runs already in summary.json, prior elapsed {prior_min:.1f} min", flush=True)
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "budget_min": budget, "steps": steps, "ho_every": ho_every,
            "full_steps": list(full_steps), "skipped": skipped, "prior_elapsed_min": prior_min, "resumed_runs": len(done_names),
            "plan": [{"tier": t, "runs": [f"{k}_{c}_L{L}_seed{s}" for (k, c, L, s) in p]} for t, p in tiers]}
    for tier, plan in tiers:
        for (kind, cond, L, seed) in plan:
            if f"{kind}_{cond}_L{L}_seed{seed}" in done_names:
                continue
            elapsed_min = prior_min + (time.time() - T0) / 60
            est = EST_RUN_SEC[kind].get(L, 60) / 60
            if tier > 0 and elapsed_min + est > budget:
                msg = f"{kind}_{cond}_L{L}_seed{seed} (tier {tier}; elapsed {elapsed_min:.1f} min + est {est:.1f} > budget {budget})"
                print("SKIP " + msg, flush=True)
                skipped.append(msg)
                continue
            info = run_one(kind, cond, L, seed, steps, ho_every, full_steps, res_dir)
            info["tier"] = tier
            summary.append(info)
            meta["elapsed_min"] = prior_min + (time.time() - T0) / 60
            with open(os.path.join(res_dir, "summary.json"), "w") as f:
                json.dump(summary, f)
            with open(os.path.join(res_dir, "run_meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            print(f"[elapsed {meta['elapsed_min']:.1f} min, {len(summary)} runs done]", flush=True)
    meta["elapsed_min"] = prior_min + (time.time() - T0) / 60
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(res_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"ALL RUNS DONE in {meta['elapsed_min']:.1f} min; skipped: {skipped}", flush=True)
    try:
        make_figures(res_dir, exp)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FIGURES FAILED ({e!r}); results are saved, rerun with --figures after fixing", flush=True)
    print(f"TOTAL {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
