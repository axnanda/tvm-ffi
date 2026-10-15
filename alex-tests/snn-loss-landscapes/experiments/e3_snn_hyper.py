import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""e3_snn_hyper: SNN-specific factors (one-factor-at-a-time around the protocol base config).

Base: kind="snn", L=4, setting B (r=2, d_out=6), kaiming init, T=8, beta=0.9, theta=1, k=4, sigmoid
surrogate, soft reset, detach_reset=False, rate readout, no feature centering; Adam lr 5e-3, 400 steps.
Sweeps (seeds {0,1}, evaluated at init AND after training, full H + H_o at both):
  T {1,2,4,8,16}; k {1,2,4,8,25}; beta {0,0.5,0.9,0.99}; detach_reset {F,T}; reset {soft,hard};
  surrogate {sigmoid,atan}; readout {rate,membrane}; init gain multiplier {0.5,1,2}; center_features {F,T}.
Depth interaction (seed 0): k in {1,4,25} x T in {1,4,16} x L in {2,4,6}  (L=4 rows of the k=4 line come
  from the T sweep; the other L=4 cells are run explicitly so every (k,T) line has L in {2,4,6}).
Per Hessian evaluation: everything in PROTOCOL.md plus (full H only) the real parts of eig(H) of the raw
non-symmetric H, max |imag|, max Frobenius norm of the strictly lower-left layer blocks (theory notes
section 5: exactly 0 for snn), the readout block's top eigenvalue (and its distance to I kron Sigma_a).

Usage:  python3 experiments/e3_snn_hyper.py            # full run    -> results/e3_snn_hyper
        python3 experiments/e3_snn_hyper.py --smoke    # reduced run -> results/e3_snn_hyper_smoke
        python3 experiments/e3_snn_hyper.py --figures  [--smoke]   # only figures + analysis + report
"""
import argparse
import json
import math
import time
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "e3_snn_hyper"

N, D_IN, WIDTH = 256, 16, 16
R, D_OUT = 2, 6  # setting B
LR, STEPS = 5e-3, 400
K_TOP = 16
BASE = dict(L=4, T=8, k=4.0, beta=0.9, detach_reset=False, reset="soft", surrogate="sigmoid",
            readout="rate", gain_mult=1.0, center_features=False)
SWEEPS = {
    "T": [1, 2, 4, 8, 16],
    "k": [1.0, 2.0, 4.0, 8.0, 25.0],
    "beta": [0.0, 0.5, 0.9, 0.99],
    "detach_reset": [False, True],
    "reset": ["soft", "hard"],
    "surrogate": ["sigmoid", "atan"],
    "readout": ["rate", "membrane"],
    "gain_mult": [0.5, 1.0, 2.0],
    "center_features": [False, True],
}
DEPTH_K, DEPTH_T, DEPTH_L = [1.0, 4.0, 25.0], [1, 4, 16], [2, 4, 6]
SEEDS = [0, 1]
BUDGET_MIN = 60.0  # wall-clock guard (phases 2-4 only; phase 1 = seed 0 of every base condition always runs)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_py(o):
    if isinstance(o, torch.Tensor):
        return to_py(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return [to_py(x) for x in o.tolist()] if o.ndim > 0 else to_py(o.item())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return to_py(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, slice):
        return [o.start, o.stop]
    return o


def tag_of(hp) -> str:
    diffs = [k for k in BASE if hp[k] != BASE[k] and k != "L"]
    if not diffs:
        return "base"
    short = {"T": "T", "k": "k", "beta": "beta", "detach_reset": "detach", "reset": "reset", "surrogate": "sur",
             "readout": "read", "gain_mult": "gain", "center_features": "center"}
    if set(diffs) <= {"k", "T"} and (hp["L"] != 4 or len(diffs) == 2):
        return f"depth_k{hp['k']:g}_T{hp['T']}"
    return "_".join(f"{short[d]}{hp[d]:g}" if isinstance(hp[d], float) else f"{short[d]}{hp[d]}" for d in diffs)


def make_spec(hp):
    return sl.ModelSpec("snn", hp["L"], D_IN, WIDTH, D_OUT, center_features=hp["center_features"], T=hp["T"],
                        beta=hp["beta"], theta=1.0, k=hp["k"], surrogate=hp["surrogate"], reset=hp["reset"],
                        detach_reset=hp["detach_reset"], readout=hp["readout"])


def spectrum_summary(H: torch.Tensor, spec, tag: str):
    """Cluster stats / coherence / layer energies of sym(H) from one eigh (cross-checked vs library in --smoke)."""
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
    topk = (topk / (V ** 2).sum(0).unsqueeze(1)).numpy()
    kk = max(R * R, D_OUT)
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
        f"topk_energy_lastlayer_mean_{tag}": float(topk[:kk, -1].mean()),
        f"zero_frac_{tag}": cs["zero_count"] / n,
        f"neg_frac_{tag}": cs["negative_count"] / n,
        f"dominant_count_{tag}": cs["split_n_dominant"],
        f"trace_{tag}": float(torch.trace(Hs)),
        f"fro_{tag}": float(torch.linalg.norm(H)),
        f"lambda_min_{tag}": float(eigs[-1]),
    }
    return out, e


def block_norms(H: torch.Tensor, slices) -> np.ndarray:
    L = len(slices)
    B = np.zeros((L, L))
    for i, si in enumerate(slices):
        for j, sj in enumerate(slices):
            B[i, j] = float(torch.linalg.norm(H[si, sj]))
    return B


def evaluate(spec, params, X, Y, step: int, loss_value: float, raw: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Full H + H_o evaluation with everything the protocol (and this experiment) asks for."""
    t0 = time.time()
    rec: Dict[str, object] = {"step": int(step), "loss": float(loss_value), "P": int(spec.n_params)}
    H_o = sl.gauss_newton(spec, params, X, Y)
    s_o, e_o = spectrum_summary(H_o, spec, "Ho")
    rec.update(s_o)
    rec["asym_Ho"] = sl.asymmetry(H_o)
    raw[f"eig_Ho_step{step}"] = e_o
    rec["wall_Ho"] = time.time() - t0
    t1 = time.time()
    H = sl.full_hessian(spec, params, X, Y)
    rec["wall_H_build"] = time.time() - t1
    H_f = sl.functional_part(H, H_o)
    s_h, e_h = spectrum_summary(H, spec, "H")
    rec.update(s_h)
    raw[f"eig_symH_step{step}"] = e_h
    rec["asymmetry_H"] = sl.asymmetry(H)
    rec["asymmetry_Hf"] = sl.asymmetry(H_f)
    rec["H_norm"] = float(torch.linalg.norm(H))
    rec["Ho_norm"] = float(torch.linalg.norm(H_o))
    rec["Hf_norm"] = float(torch.linalg.norm(H_f))
    rec["Hf_over_Ho"] = rec["Hf_norm"] / rec["Ho_norm"] if rec["Ho_norm"] > 0 else float("nan")
    # --- extras: raw (non-symmetric) spectrum, block structure, readout block ------------------
    ev = np.linalg.eigvals(H.numpy())
    re = np.sort(ev.real)[::-1]
    raw[f"eigH_raw_re_step{step}"] = re
    raw[f"eigH_raw_im_step{step}"] = ev.imag
    rec["max_re_eigH"] = float(re[0])
    rec["min_re_eigH"] = float(re[-1])
    rec["max_abs_imag_eigH"] = float(np.abs(ev.imag).max())
    rec["max_abs_imag_over_max_re"] = float(np.abs(ev.imag).max() / abs(re[0])) if re[0] != 0 else float("nan")
    rec["cluster_Hre"] = sl.cluster_stats(re, R, D_IN, D_OUT)
    rec["dominant_count_Hre"] = rec["cluster_Hre"]["split_n_dominant"]
    rec["neg_frac_Hre"] = rec["cluster_Hre"]["negative_count"] / len(re)
    rec["zero_frac_Hre"] = rec["cluster_Hre"]["zero_count"] / len(re)
    slices = spec.layer_slices
    L = spec.L
    B = block_norms(H, slices)
    rec["block_norms_H"] = B
    rec["max_lower_block_norm"] = float(max([B[i, j] for i in range(L) for j in range(L) if j < i] + [0.0]))
    rec["max_upper_block_norm"] = float(max([B[i, j] for i in range(L) for j in range(L) if j > i] + [0.0]))
    rec["max_lower_over_H_norm"] = rec["max_lower_block_norm"] / rec["H_norm"] if rec["H_norm"] > 0 else float("nan")
    # eigenvalues of the diagonal blocks of the raw H (union == eig(H) iff block triangular)
    diag_re = []
    hidden_block_max_re, hidden_block_min_re = [], []
    for s_ in slices:
        evb = np.linalg.eigvals(H[s_, s_].numpy())
        diag_re.append(np.sort(evb.real)[::-1])
    rec["eig_union_err"] = float(np.abs(np.sort(np.concatenate(diag_re))[::-1] - re).max())
    for d_ in diag_re[:-1]:
        hidden_block_max_re.append(float(d_[0]))
        hidden_block_min_re.append(float(d_[-1]))
    rec["hidden_block_max_re"] = hidden_block_max_re
    rec["hidden_block_min_re"] = hidden_block_min_re
    rec["max_hidden_block_re"] = float(max(hidden_block_max_re)) if hidden_block_max_re else float("nan")
    Hro = H[slices[-1], slices[-1]]
    ro_eigs = sl.eigh(sl.symmetrize(Hro))[0].numpy()
    rec["readout_block_top"] = float(ro_eigs[0])
    rec["readout_block_top_dout"] = [float(x) for x in ro_eigs[:D_OUT]]
    rec["readout_block_asym"] = sl.asymmetry(Hro)
    with torch.no_grad():
        a = spec.hidden_features(params, X)
        F = a - a.mean(0) if spec.center_features else a
        Sig = (F.T @ F / N).numpy()
    kron = np.kron(np.eye(D_OUT), Sig)
    rec["readout_block_vs_I_kron_Sigma_rel_err"] = float(np.linalg.norm(Hro.numpy() - kron) / np.linalg.norm(kron)) if np.linalg.norm(kron) > 0 else float("nan")
    rec["max_re_eigH_is_readout"] = bool(abs(rec["max_re_eigH"] - rec["readout_block_top"]) <= 1e-8 * max(1.0, abs(rec["max_re_eigH"])))
    # --- features / spikes --------------------------------------------------------------------
    rec["features"] = sl.feature_second_moment_stats(spec, params, X)
    rec["spikes"] = sl.spike_diagnostics(spec, params, X)
    rec["wall_H"] = time.time() - t1
    rec["wall"] = time.time() - t0
    return rec


def run_one(hp: Dict[str, object], seed: int, steps: int, res_dir: str) -> Dict[str, object]:
    X = sl.make_whitened_inputs(N, D_IN, seed)
    Y, Wstar = sl.make_rank_r_targets(X, R, D_OUT, seed)
    assert X.shape == (N, D_IN) and Y.shape == (N, D_OUT)
    assert int(torch.linalg.matrix_rank(Y)) == R, "target rank != r"
    assert float((X.T @ X / N - torch.eye(D_IN, dtype=X.dtype)).abs().max()) < 1e-8, "inputs not whitened"
    spec = make_spec(hp)
    gain = math.sqrt(2.0) * hp["gain_mult"]
    params = spec.init_params(seed, gain=gain)
    tag = tag_of(hp)
    name = f"{tag}_L{hp['L']}_seed{seed}"
    info: Dict[str, object] = {"name": name, "tag": tag, "hp": dict(hp), "seed": seed, "L": hp["L"], "r": R,
                               "d_in": D_IN, "d_out": D_OUT, "n": N, "width": WIDTH, "P": spec.n_params,
                               "steps": steps, "lr": LR, "gain": gain, "spec": repr(spec)}
    print(f"== run {name}: P={spec.n_params} {spec}", flush=True)
    raw: Dict[str, np.ndarray] = {}
    t0 = time.time()
    loss0 = float(sl.loss_fn(spec, params, X, Y))
    rec0 = evaluate(spec, params, X, Y, 0, loss0, raw)
    hist = sl.gd_train(spec, params, X, Y, lr=LR, steps=steps, loss="mse", optimizer="adam")
    t_train = time.time() - t0 - rec0["wall"]
    rec1 = evaluate(spec, hist["params"], X, Y, steps, float(hist["loss"][-1]), raw)
    wall = time.time() - t0
    loss_curve = np.asarray(hist["loss"], dtype=np.float64)
    info.update({"wall_sec": wall, "wall_train": t_train, "loss_init": float(loss_curve[0]),
                 "loss_final": float(loss_curve[-1]), "loss_min": float(loss_curve.min()),
                 "loss_curve_every10": [float(x) for x in loss_curve[::10]] + [float(loss_curve[-1])],
                 "init": rec0, "end": rec1})
    raw["loss_curve"] = loss_curve
    raw["grad_norm_curve"] = np.asarray(hist["grad_norm"], dtype=np.float64)
    raw["Wstar"] = Wstar.numpy()
    os.makedirs(os.path.join(res_dir, "raw"), exist_ok=True)
    np.savez(os.path.join(res_dir, "raw", f"{name}.npz"), **raw)
    for lab, rc in (("init", rec0), ("end", rec1)):
        print(f"   {lab:4s} loss={rc['loss']:.4e} lam(Ho)={rc['coh_Ho']['lambda_max']:.3e} lam(symH)={rc['coh_H']['lambda_max']:.3e} "
              f"maxRe(H)={rc['max_re_eigH']:.3e} ro_top={rc['readout_block_top']:.3e} coh={rc['coh_Ho']['coherence']:.2f} "
              f"ndom(Ho/H/Hre)={rc['dominant_count_Ho']}/{rc['dominant_count_H']}/{rc['dominant_count_Hre']} "
              f"rsplit={rc['cluster_Ho']['ratio_split']:.2f} rpaper={rc['cluster_Ho']['ratio_paper']:.2f} "
              f"asym={rc['asymmetry_H']:.3f} Hf/Ho={rc['Hf_over_Ho']:.3f} lower={rc['max_lower_block_norm']:.1e} "
              f"imag={rc['max_abs_imag_eigH']:.1e} rate={rc['spikes']['mean_rate']:.3f} wall={rc['wall']:.1f}s", flush=True)
    print(f"   done {name}: loss {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f}, train {t_train:.0f}s, total {wall:.0f}s", flush=True)
    return to_py(info)


# --------------------------------------------------------------------------- #
# smoke validation
# --------------------------------------------------------------------------- #
def validate_once(hp, seed=0):
    X = sl.make_whitened_inputs(N, D_IN, seed)
    Y, _ = sl.make_rank_r_targets(X, R, D_OUT, seed)
    spec = make_spec(hp)
    p = spec.init_params(seed, gain=math.sqrt(2.0) * hp["gain_mult"])
    p_default = spec.init_params(seed)
    assert all(torch.equal(p[k], p_default[k]) for k in p), "gain_mult=1 must reproduce the default kaiming init"
    H_o = sl.gauss_newton(spec, p, X, Y)
    s_o, e_o = spectrum_summary(H_o, spec, "Ho")
    ref_cs = sl.cluster_stats(sl.eigvals(H_o), R, D_IN, D_OUT)
    ref_coh = sl.layer_coherence(H_o, spec.layer_slices)
    ref_topk = sl.top_eigvec_layer_energy(H_o, spec.layer_slices, k=max(R * R, D_OUT)).numpy()
    for k_ in ref_cs:
        a, b = ref_cs[k_], s_o["cluster_Ho"][k_]
        same = (a == b) or (isinstance(a, float) and isinstance(b, float) and ((math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1, abs(a))))
        assert same, (k_, a, b)
    assert abs(ref_coh["coherence"] - s_o["coh_Ho"]["coherence"]) < 1e-9
    assert np.allclose(ref_coh["lambda_max_blocks"], s_o["coh_Ho"]["lambda_max_blocks"])
    assert np.allclose(ref_coh["top_eigvec_layer_energy"], s_o["coh_Ho"]["top_eigvec_layer_energy"], atol=1e-8)
    assert np.allclose(ref_topk, s_o["topk_energy_Ho"][: ref_topk.shape[0]], atol=1e-8)
    cs = s_o["cluster_Ho"]
    assert cs["n_eigs"] == spec.n_params and cs["n_dominant_expected"] == R * R
    assert cs["n_bulk_expected"] == R * (D_IN + D_OUT - 2 * R)
    assert cs["largest_log_gap_index"] + 1 == cs["split_n_dominant"]
    assert e_o[-1] > -1e-10 * e_o[0], "H_o not PSD?"
    raw = {}
    rec = evaluate(spec, p, X, Y, 0, float(sl.loss_fn(spec, p, X, Y)), raw)
    feats = rec["features"]
    assert abs(rec["coh_Ho"]["lambda_max_blocks"][-1] - feats["second_moment_top3"][0]) < 1e-8 * feats["second_moment_top3"][0]
    assert abs(rec["trace_H"] - float(np.sum(raw["eigH_raw_re_step0"]))) < 1e-6 * max(1.0, abs(rec["trace_H"]))
    assert abs(rec["trace_Ho"] - float(np.sum(e_o))) < 1e-8 * max(1.0, abs(rec["trace_Ho"]))
    print("[validate] max lower block norm =", rec["max_lower_block_norm"], " eig-union err =", rec["eig_union_err"],
          " readout vs I kron Sigma rel err =", rec["readout_block_vs_I_kron_Sigma_rel_err"])
    H = sl.full_hessian(spec, p, X, Y)
    fd = sl.check_hessian_fd(spec, p, X, Y, H=H, n_dirs=2, eps=1e-6)
    print("[validate] finite-difference rel. errors of H:", fd)
    print("[validate] ONE FULL STATS DICT (init, %s):" % spec)
    print(json.dumps(to_py(rec), indent=1, default=str))
    print("[validate] top-10 eigs H_o:", np.round(e_o[:10], 4))
    print("[validate] top-10 eigs sym(H):", np.round(raw["eig_symH_step0"][:10], 4), " min:", raw["eig_symH_step0"][-1])
    print("[validate] top-10 Re eig(H):", np.round(raw["eigH_raw_re_step0"][:10], 4), " min:", raw["eigH_raw_re_step0"][-1])
    print("[validate] all library cross-checks passed", flush=True)


# --------------------------------------------------------------------------- #
# run plan
# --------------------------------------------------------------------------- #
def plan(smoke: bool):
    """Returns ordered phases of (hp, seed) with a flag whether the budget guard applies."""
    def hp_with(**kw):
        hp = dict(BASE); hp.update(kw); return hp
    if smoke:
        conds = [hp_with(), hp_with(T=16), hp_with(k=25.0), hp_with(center_features=True), hp_with(readout="membrane"),
                 hp_with(gain_mult=0.5), hp_with(detach_reset=True), hp_with(reset="hard"), hp_with(surrogate="atan"),
                 hp_with(beta=0.0), hp_with(T=1)]
        depth = [hp_with(k=25.0, T=16, L=2), hp_with(k=1.0, T=4, L=6)]
        return [("p1", [(c, 0) for c in conds], False), ("p3", [(c, 0) for c in depth], True)]
    base_conds = [hp_with()]
    for f, vals in SWEEPS.items():
        for v in vals:
            if v != BASE[f]:
                base_conds.append(hp_with(**{f: v}))
    core = [c for c in base_conds if tag_of(c) == "base" or tag_of(c).startswith("k") or tag_of(c).startswith("T")]
    rest = [c for c in base_conds if c not in core]
    depth = []
    for k in DEPTH_K:
        for T in DEPTH_T:
            for L in DEPTH_L:
                if L == 4 and k == BASE["k"]:
                    continue  # from the T sweep
                depth.append(hp_with(k=k, T=T, L=L))
    depth.sort(key=lambda c: (c["L"] * max(c["T"], 2)))  # cheap first
    return [("p1_seed0_base", [(c, 0) for c in base_conds], False),
            ("p2_seed1_kT", [(c, 1) for c in core], True),
            ("p3_depth_seed0", [(c, 0) for c in depth], True),
            ("p4_seed1_rest", [(c, 1) for c in rest], True)]


def estimate_cost(hp, walls: Dict[str, float]) -> float:
    """Rough per-run wall estimate (s) from measured walls of the same tag or a scaling of the base run."""
    tag = tag_of(hp)
    if tag in walls:
        return walls[tag]
    base = walls.get("base", 60.0)
    Lf = {2: 0.3, 3: 0.55, 4: 1.0, 5: 1.7, 6: 2.6}.get(hp["L"], 3.0)
    return base * Lf * max(hp["T"], 2) / 8.0


# --------------------------------------------------------------------------- #
# aggregation helpers
# --------------------------------------------------------------------------- #
def get(c, path):
    for p_ in path.split("."):
        if c is None:
            return np.nan
        if isinstance(c, dict):
            c = c.get(p_)
        elif isinstance(c, (list, tuple)):
            try:
                c = c[int(p_)]
            except (ValueError, IndexError):
                c = None
        else:
            c = None
    if isinstance(c, bool):
        return float(c)
    return np.nan if c is None else c


def hp_matches(hp, free):
    return all(hp[k] == BASE[k] for k in BASE if k not in free)


def sweep_runs(runs, factor):
    return [r_ for r_ in runs if hp_matches(r_["hp"], {factor})]


def agg_vals(rs, when, path):
    vals = np.asarray([get(r_[when], path) for r_ in rs], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return (vals.mean() if len(vals) else np.nan, vals.std() if len(vals) > 1 else 0.0, len(vals), vals.tolist())


def by_value(runs, factor, when, path):
    rs = sweep_runs(runs, factor)
    vals = sorted({r_["hp"][factor] for r_ in rs}, key=lambda v: (str(type(v)), v))
    out = {}
    for v in vals:
        out[v] = agg_vals([r_ for r_ in rs if r_["hp"][factor] == v], when, path)
    return out


def loglog_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(y) & (y > 0) & (x > 0)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(x[m]), np.log(y[m])
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss_res = ((ly - pred) ** 2).sum(); ss_tot = ((ly - ly.mean()) ** 2).sum()
    return float(a), float(b), float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


METRICS = [  # (path, short label)
    ("coh_Ho.lambda_max", "lam_max(H_o)"),
    ("coh_H.lambda_max", "lam_max(sym H)"),
    ("max_re_eigH", "max Re eig(H)"),
    ("readout_block_top", "readout block top"),
    ("max_hidden_block_re", "max hidden-block Re"),
    ("trace_Ho", "trace(H_o)"),
    ("trace_H", "trace(H)"),
    ("cluster_Ho.ratio_paper", "ratio_paper(H_o)"),
    ("cluster_Ho.ratio_split", "ratio_split(H_o)"),
    ("cluster_H.ratio_split", "ratio_split(sym H)"),
    ("cluster_Hre.ratio_split", "ratio_split(Re H)"),
    ("coh_Ho.coherence", "coherence(H_o)"),
    ("coh_H.coherence", "coherence(sym H)"),
    ("dominant_count_Ho", "n_dom(H_o)"),
    ("dominant_count_H", "n_dom(sym H)"),
    ("dominant_count_Hre", "n_dom(Re H)"),
    ("zero_frac_Ho", "zero frac(H_o)"),
    ("zero_frac_H", "zero frac(sym H)"),
    ("neg_frac_H", "neg frac(sym H)"),
    ("neg_frac_Hre", "neg frac(Re H)"),
    ("asymmetry_H", "asymmetry(H)"),
    ("Hf_over_Ho", "|H_f|/|H_o|"),
    ("max_lower_block_norm", "max lower block"),
    ("max_abs_imag_eigH", "max |Im eig H|"),
    ("spikes.mean_rate", "mean rate"),
    ("spikes.window_frac", "window frac"),
    ("features.mu_sq_norm", "||mu||^2"),
    ("features.mean_frac_of_second_moment", "||mu||^2/tr M"),
    ("features.cov_top3.0", "lam1(Cov_a)"),
    ("topk_energy_lastlayer_mean_Ho", "top-6 evec energy in W_L (H_o)"),
    ("loss", "loss"),
]
TABLE_COLS = [("coh_Ho.lambda_max", "lam(Ho)"), ("coh_H.lambda_max", "lam(symH)"), ("max_re_eigH", "maxRe(H)"),
              ("readout_block_top", "ro_top"), ("trace_Ho", "tr(Ho)"), ("cluster_Ho.ratio_paper", "r_paper"),
              ("cluster_Ho.ratio_split", "r_split"), ("coh_Ho.coherence", "coh"), ("dominant_count_Ho", "ndom(Ho)"),
              ("dominant_count_Hre", "ndom(ReH)"), ("zero_frac_Ho", "zero(Ho)"), ("neg_frac_H", "neg(symH)"),
              ("asymmetry_H", "asym"), ("Hf_over_Ho", "Hf/Ho"), ("spikes.mean_rate", "rate"),
              ("features.mean_frac_of_second_moment", "mu2/trM"), ("loss", "loss")]


def fmt(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.2e}"
    return f"{v:.3g}"


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    for r_ in rows:
        s += "| " + " | ".join(r_) + " |\n"
    return s


# --------------------------------------------------------------------------- #
# figures + analysis
# --------------------------------------------------------------------------- #
def make_figures(res_dir, fig_dir, exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(res_dir, "summary.json")) as f:
        runs = json.load(f)
    os.makedirs(fig_dir, exist_ok=True)
    figures: List[str] = []
    analysis: Dict[str, object] = {"n_runs": len(runs)}
    tables: Dict[str, str] = {}
    whens = [("init", "tab:blue"), ("end", "tab:red")]

    # ---- 0. base-config table (init and end) -------------------------------------------------
    base_rs = [r_ for r_ in runs if hp_matches(r_["hp"], set())]
    rows = []
    for when, _ in whens:
        rows.append([when, str(len(base_rs))] + [fmt(agg_vals(base_rs, when, p)[0]) for p, _ in TABLE_COLS])
    tables["base"] = md_table(["step", "n_seeds"] + [lab for _, lab in TABLE_COLS], rows)

    # ---- 1. numeric factor panels -------------------------------------------------------------
    numeric = [("k", "log"), ("T", "log"), ("beta", "linear"), ("gain_mult", "log")]
    panel_metrics = [
        [("coh_Ho.lambda_max", "lam_max(H_o)"), ("coh_H.lambda_max", "lam_max(sym H)"), ("max_re_eigH", "max Re eig(H)"),
         ("readout_block_top", "readout block top"), ("max_hidden_block_re", "max hidden block Re")],
        [("trace_Ho", "trace(H_o)"), ("trace_H", "trace(H)")],
        [("cluster_Ho.ratio_paper", "ratio_paper(H_o)"), ("cluster_Ho.ratio_split", "ratio_split(H_o)"),
         ("cluster_Hre.ratio_split", "ratio_split(Re H)"), ("coh_Ho.coherence", "coherence(H_o)"), ("coh_H.coherence", "coherence(sym H)")],
        [("dominant_count_Ho", "n_dom(H_o)"), ("dominant_count_H", "n_dom(sym H)"), ("dominant_count_Hre", "n_dom(Re H)")],
        [("zero_frac_Ho", "zero frac(H_o)"), ("zero_frac_H", "zero frac(sym H)"), ("neg_frac_H", "neg frac(sym H)"), ("neg_frac_Hre", "neg frac(Re H)")],
        [("asymmetry_H", "asymmetry(H)"), ("Hf_over_Ho", "|H_f|/|H_o|"), ("max_abs_imag_over_max_re", "max|Im|/maxRe")],
        [("spikes.mean_rate", "mean rate"), ("spikes.window_frac", "window frac"), ("features.mean_frac_of_second_moment", "||mu||^2/tr M")],
        [("features.mu_sq_norm", "||mu||^2"), ("features.cov_top3.0", "lam1(Cov_a)"), ("topk_energy_lastlayer_mean_Ho", "top-6 evec energy in W_L")],
    ]
    panel_scale = ["log", "log", "linear", "linear", "linear", "linear", "linear", "log"]
    markers = ["o", "s", "^", "v", "D", "x"]
    factor_tables = {}
    for factor, xscale in numeric:
        rs = sweep_runs(runs, factor)
        if not rs:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
        for ax, mets, ysc in zip(axes.ravel(), panel_metrics, panel_scale):
            for mi, (path, lab) in enumerate(mets):
                for when, col in whens:
                    bv = by_value(runs, factor, when, path)
                    xs = np.array(list(bv.keys()), float)
                    m = np.array([bv[v][0] for v in bv]); s_ = np.array([bv[v][1] for v in bv])
                    if ysc == "log":
                        ok = m > 0
                        if ok.any():
                            ax.errorbar(xs[ok], m[ok], yerr=np.minimum(s_[ok], 0.99 * m[ok]), marker=markers[mi % 6], ms=4, capsize=2,
                                        color=col, alpha=0.5 + 0.5 * (mi == 0), ls=["-", "--", ":", "-.", (0, (5, 1)), "-"][mi % 6], lw=1,
                                        label=f"{lab} ({when})")
                        if (~ok).any():
                            ax.plot(xs[~ok], np.abs(m[~ok]), marker=markers[mi % 6], ms=4, color=col, ls="none", mfc="none", label=f"|{lab}|<0 ({when})")
                    else:
                        ax.errorbar(xs, m, yerr=s_, marker=markers[mi % 6], ms=4, capsize=2, color=col, alpha=0.5 + 0.5 * (mi == 0),
                                    ls=["-", "--", ":", "-.", (0, (5, 1)), "-"][mi % 6], lw=1, label=f"{lab} ({when})")
            if xscale == "log":
                ax.set_xscale("log")
            if ysc == "log":
                ax.set_yscale("log")
            ax.set_xlabel(factor + (" (x sqrt2)" if factor == "gain_mult" else ""))
            ax.grid(alpha=0.3); ax.legend(fontsize=6)
            if xscale == "linear" and factor == "beta":
                ax.set_xticks([0, 0.5, 0.9, 0.99])
        fig.suptitle(f"{exp}: one-factor sweep of {factor} (L=4, setting B r=2 d_out=6; mean +- std over seeds; blue=init, red=after {STEPS} Adam steps)", fontsize=10)
        fig.tight_layout()
        fn = os.path.join(fig_dir, f"{exp}_sweep_{factor}.png"); fig.savefig(fn, dpi=120); plt.close(fig); figures.append(fn)
        # table (end of training, plus init lam_max)
        vals = sorted({r_["hp"][factor] for r_ in rs})
        rows = []
        for v in vals:
            rr = [r_ for r_ in rs if r_["hp"][factor] == v]
            rows.append([f"{v:g}", str(len(rr)), fmt(agg_vals(rr, "init", "coh_Ho.lambda_max")[0])] + [fmt(agg_vals(rr, "end", p)[0]) for p, _ in TABLE_COLS])
        factor_tables[factor] = md_table([factor, "n", "lam(Ho)@init"] + [lab + "@end" for _, lab in TABLE_COLS], rows)
        tables[factor] = factor_tables[factor]

    # ---- 2. k scaling fits -----------------------------------------------------------------------
    fits = {}
    rs_k = sweep_runs(runs, "k")
    if rs_k:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        scale_paths = [("coh_Ho.lambda_max", "lam_max(H_o)"), ("coh_H.lambda_max", "lam_max(sym H)"), ("max_re_eigH", "max Re eig(H)"),
                       ("readout_block_top", "readout block top"), ("max_hidden_block_re", "max hidden block Re"),
                       ("trace_Ho", "trace(H_o)"), ("trace_H", "trace(H)"), ("H_norm", "||H||_F"), ("Ho_norm", "||H_o||_F")]
        inv_paths = [("cluster_Ho.ratio_paper", "ratio_paper(H_o)"), ("cluster_Ho.ratio_split", "ratio_split(H_o)"),
                     ("coh_Ho.coherence", "coherence(H_o)"), ("dominant_count_Ho", "n_dom(H_o)"), ("zero_frac_Ho", "zero frac(H_o)"),
                     ("neg_frac_H", "neg frac(sym H)"), ("asymmetry_H", "asymmetry(H)"), ("Hf_over_Ho", "|H_f|/|H_o|"),
                     ("spikes.mean_rate", "mean rate"), ("spikes.window_frac", "window frac")]
        for when, col in whens:
            for mi, (path, lab) in enumerate(scale_paths):
                xs = np.array([r_["hp"]["k"] for r_ in rs_k], float)
                ys = np.array([get(r_[when], path) for r_ in rs_k], float)
                a, b, R2 = loglog_fit(xs, np.abs(ys))
                fits[f"{path}|{when}"] = {"exponent": a, "intercept": b, "R2": R2, "n_points": int(np.isfinite(ys).sum()),
                                          "any_negative": bool((ys < 0).any())}
                ax = axes[0] if mi < 5 else axes[1]
                ax.plot(xs, np.abs(ys), marker=markers[mi % 6], ls="none", color=col, alpha=0.6, ms=4)
                kk = np.array([xs.min(), xs.max()])
                if np.isfinite(a):
                    ax.plot(kk, np.exp(b) * kk ** a, color=col, ls=["-", "--", ":", "-.", (0, (5, 1))][mi % 5], lw=1,
                            label=f"{lab} {when}: k^{a:.2f} (R2={R2:.2f})")
            for mi, (path, lab) in enumerate(inv_paths):
                xs = np.array([r_["hp"]["k"] for r_ in rs_k], float)
                ys = np.array([get(r_[when], path) for r_ in rs_k], float)
                a, b, R2 = loglog_fit(xs, np.abs(ys))
                fits[f"{path}|{when}"] = {"exponent": a, "intercept": b, "R2": R2, "n_points": int(np.isfinite(ys).sum())}
        for ax in axes[:2]:
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("k (surrogate slope)"); ax.grid(alpha=0.3); ax.legend(fontsize=5.5)
            kk = np.array([1, 25.0]); ax.plot(kk, kk ** 2 * 0.1, "k:", lw=0.8, label="k^2 ref")
        axes[0].set_title("scale: top eigenvalues vs k (|.|, log-log fits)", fontsize=9)
        axes[1].set_title("scale: traces / norms vs k", fontsize=9)
        ax = axes[2]
        for mi, (path, lab) in enumerate(inv_paths[:8]):
            for when, col in whens:
                bv = by_value(runs, "k", when, path)
                xs = np.array(list(bv.keys()), float); m = np.array([bv[v][0] for v in bv])
                ref = m[np.argmin(np.abs(xs - 4.0))]
                ax.plot(xs, m / ref if ref not in (0, np.nan) and np.isfinite(ref) and ref != 0 else m, marker=markers[mi % 6], ms=4, color=col,
                        ls=["-", "--", ":", "-.", (0, (5, 1)), "-"][mi % 6], lw=1, label=f"{lab} ({when})")
        ax.set_xscale("log"); ax.set_xlabel("k"); ax.set_ylabel("value / value at k=4"); ax.axhline(1, color="k", lw=0.8)
        ax.set_title("structure metrics vs k (normalised to k=4)", fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=5.5)
        fig.suptitle(f"{exp}: test of P3 (scale ~ k^2 vs k-invariant structure); per-seed points, fits over all seeds", fontsize=10)
        fig.tight_layout()
        fn = os.path.join(fig_dir, f"{exp}_k_scaling.png"); fig.savefig(fn, dpi=120); plt.close(fig); figures.append(fn)
        rows = []
        for path, lab in scale_paths + inv_paths:
            rows.append([lab] + [f"{fits[f'{path}|{w}']['exponent']:.2f} (R2 {fits[f'{path}|{w}']['R2']:.2f})" if np.isfinite(fits[f'{path}|{w}']['exponent']) else "nan" for w, _ in whens])
        tables["k_exponents"] = md_table(["metric", "exponent@init", "exponent@end"], rows)
    analysis["k_loglog_fits"] = fits

    # ---- 3. categorical factors -------------------------------------------------------------------
    cat = [("detach_reset", True), ("reset", "hard"), ("surrogate", "atan"), ("readout", "membrane"), ("center_features", True)]
    cat_metrics = [("coh_Ho.lambda_max", "lam_max(H_o)", "log"), ("max_re_eigH", "max Re eig(H)", "log"), ("cluster_Ho.ratio_split", "ratio_split(H_o)", "linear"),
                   ("coh_Ho.coherence", "coherence(H_o)", "linear"), ("dominant_count_Ho", "n_dom(H_o)", "linear"), ("dominant_count_Hre", "n_dom(Re H)", "linear"),
                   ("asymmetry_H", "asymmetry(H)", "linear"), ("Hf_over_Ho", "|H_f|/|H_o|", "linear"), ("neg_frac_H", "neg frac(sym H)", "linear"),
                   ("zero_frac_Ho", "zero frac(H_o)", "linear"), ("features.mean_frac_of_second_moment", "||mu||^2/tr M", "linear"), ("spikes.mean_rate", "mean rate", "linear")]
    conds = [("base", base_rs)] + [(f"{f}={v}", [r_ for r_ in runs if hp_matches(r_["hp"], {f}) and r_["hp"][f] == v]) for f, v in cat]
    conds = [(n_, rs) for n_, rs in conds if rs]
    if conds:
        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        xs = np.arange(len(conds))
        for ax, (path, lab, ysc) in zip(axes.ravel(), cat_metrics):
            for wi, (when, col) in enumerate(whens):
                m = np.array([agg_vals(rs, when, path)[0] for _, rs in conds]); s_ = np.array([agg_vals(rs, when, path)[1] for _, rs in conds])
                ax.bar(xs + (wi - 0.5) * 0.38, m, 0.38, yerr=s_, color=col, alpha=0.7, capsize=2, label=when)
            ax.set_xticks(xs); ax.set_xticklabels([n_ for n_, _ in conds], rotation=20, fontsize=7)
            if ysc == "log":
                ax.set_yscale("log")
            ax.set_title(lab, fontsize=9); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)
        fig.suptitle(f"{exp}: categorical factors (L=4, setting B; mean +- std over seeds; init vs end)", fontsize=10)
        fig.tight_layout()
        fn = os.path.join(fig_dir, f"{exp}_categorical.png"); fig.savefig(fn, dpi=120); plt.close(fig); figures.append(fn)
        rows = []
        for n_, rs in conds:
            rows.append([n_, str(len(rs)), fmt(agg_vals(rs, "init", "coh_Ho.lambda_max")[0]), fmt(agg_vals(rs, "init", "asymmetry_H")[0])]
                        + [fmt(agg_vals(rs, "end", p)[0]) for p, _ in TABLE_COLS])
        tables["categorical"] = md_table(["condition", "n", "lam(Ho)@init", "asym@init"] + [lab + "@end" for _, lab in TABLE_COLS], rows)

    # ---- 4. depth interaction ----------------------------------------------------------------------
    drs = [r_ for r_ in runs if hp_matches(r_["hp"], {"k", "T", "L"}) and r_["hp"]["k"] in DEPTH_K and r_["hp"]["T"] in DEPTH_T and r_["seed"] == 0]
    depth_tab = {}
    if drs:
        dmetrics = [("coh_Ho.coherence", "coherence(H_o)", "linear"), ("cluster_Ho.ratio_split", "ratio_split(H_o)", "linear"),
                    ("cluster_Ho.ratio_paper", "ratio_paper(H_o)", "linear"), ("coh_Ho.lambda_max", "lam_max(H_o)", "log"),
                    ("max_re_eigH", "max Re eig(H)", "log"), ("dominant_count_Ho", "n_dom(H_o)", "linear"),
                    ("asymmetry_H", "asymmetry(H)", "linear"), ("spikes.mean_rate", "mean rate", "linear")]
        kcol = {1.0: "tab:green", 4.0: "tab:blue", 25.0: "tab:purple"}; tls = {1: ":", 4: "--", 16: "-"}
        fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
        for ax, (path, lab, ysc) in zip(axes.ravel(), dmetrics):
            for when, mk in (("init", "o"), ("end", "s")):
                for k in DEPTH_K:
                    for T in DEPTH_T:
                        pts = sorted([(r_["hp"]["L"], get(r_[when], path)) for r_ in drs if r_["hp"]["k"] == k and r_["hp"]["T"] == T])
                        if not pts:
                            continue
                        Ls_ = [p_[0] for p_ in pts]; ys = [p_[1] for p_ in pts]
                        ax.plot(Ls_, ys, marker=mk, ms=4, color=kcol[k], ls=tls[T], lw=1, alpha=0.9 if when == "end" else 0.45,
                                label=f"k={k:g} T={T} ({when})")
                        depth_tab[f"k{k:g}_T{T}_{when}_{path}"] = dict(zip([str(l_) for l_ in Ls_], ys))
            if "coherence" in path or "ratio" in path:
                ax.plot([2, 6], [2, 6], "k--", lw=0.8, label="= L (paper C2)")
            if ysc == "log":
                ax.set_yscale("log")
            ax.set_xlabel("L"); ax.set_xticks(DEPTH_L); ax.set_title(lab, fontsize=9); ax.grid(alpha=0.3)
            if path == "coh_Ho.coherence":
                ax.legend(fontsize=5, ncol=2)
        fig.suptitle(f"{exp}: depth interaction (seed 0): colour = k (green 1, blue 4, purple 25), linestyle = T (dotted 1, dashed 4, solid 16); faint=init, solid=end", fontsize=9)
        fig.tight_layout()
        fn = os.path.join(fig_dir, f"{exp}_depth_interaction.png"); fig.savefig(fn, dpi=120); plt.close(fig); figures.append(fn)
        rows = []
        for k in DEPTH_K:
            for T in DEPTH_T:
                row = [f"{k:g}", str(T)]
                for path in ["coh_Ho.coherence", "cluster_Ho.ratio_split", "coh_Ho.lambda_max", "dominant_count_Ho"]:
                    for L in DEPTH_L:
                        rr = [r_ for r_ in drs if r_["hp"]["k"] == k and r_["hp"]["T"] == T and r_["hp"]["L"] == L]
                        row.append(fmt(get(rr[0]["end"], path)) if rr else "-")
                rows.append(row)
        hdr = ["k", "T"] + [f"{lab}@L{L}" for lab in ["coh", "r_split", "lam(Ho)", "ndom"] for L in DEPTH_L]
        tables["depth_interaction_end"] = md_table(hdr, rows)
        # slopes of coherence vs L per (k, T)
        slopes = {}
        for k in DEPTH_K:
            for T in DEPTH_T:
                for when in ("init", "end"):
                    pts = sorted([(r_["hp"]["L"], get(r_[when], "coh_Ho.coherence")) for r_ in drs if r_["hp"]["k"] == k and r_["hp"]["T"] == T])
                    if len(pts) >= 2:
                        a, b = np.polyfit([p_[0] for p_ in pts], [p_[1] for p_ in pts], 1)
                        slopes[f"k{k:g}_T{T}_{when}"] = {"slope": float(a), "intercept": float(b), "points": pts}
        analysis["coherence_vs_L_slopes"] = slopes
    analysis["depth_table_values"] = depth_tab

    # ---- 5. spectra of selected conditions ---------------------------------------------------------
    sel = [("base", {}), ("k=1", {"k": 1.0}), ("k=25", {"k": 25.0}), ("T=1", {"T": 1}), ("T=16", {"T": 16}),
           ("gain x0.5", {"gain_mult": 0.5}), ("gain x2", {"gain_mult": 2.0}), ("center", {"center_features": True}), ("membrane", {"readout": "membrane"})]
    sel_runs = []
    for lab, kw in sel:
        hp = dict(BASE); hp.update(kw)
        rr = [r_ for r_ in runs if r_["hp"] == hp and r_["seed"] == 0]
        if rr:
            sel_runs.append((lab, rr[0]))
    if sel_runs:
        fig, axes = plt.subplots(3, len(sel_runs), figsize=(2.6 * len(sel_runs), 8.5), squeeze=False)
        for j, (lab, r_) in enumerate(sel_runs):
            z = np.load(os.path.join(res_dir, "raw", r_["name"] + ".npz"))
            for i, (key, ttl) in enumerate([("eig_Ho", "H_o"), ("eig_symH", "sym(H)"), ("eigH_raw_re", "Re eig(H)")]):
                ax = axes[i, j]
                for st, col in ((0, "tab:blue"), (r_["steps"], "tab:red")):
                    e = z[f"{key}_step{st}"]
                    idx = np.arange(1, len(e) + 1); pos = e > 0
                    ax.plot(idx[pos], e[pos], color=col, lw=0.8, label=f"step {st}")
                    if (~pos).any():
                        ax.plot(idx[~pos], np.abs(e[~pos]), color=col, lw=0.8, ls=":", label=f"|neg| step {st}")
                ax.axvline(R * R, color="k", ls="--", lw=0.7); ax.axvline(D_OUT, color="gray", ls="-.", lw=0.7)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(f"{ttl}: {lab}", fontsize=8)
                if i == 2:
                    ax.set_xlabel("index")
                if j == 0:
                    ax.set_ylabel("eigenvalue (|.| dotted if <0)")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.suptitle(f"{exp}: sorted spectra (seed 0; dashed r^2={R*R}, dash-dot d_out={D_OUT})", fontsize=10)
        fig.tight_layout()
        fn = os.path.join(fig_dir, f"{exp}_spectra.png"); fig.savefig(fn, dpi=120); plt.close(fig); figures.append(fn)

    # ---- 6. loss curves per factor -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, factor in zip(axes.ravel(), SWEEPS):
        rs = sweep_runs(runs, factor)
        vals = sorted({r_["hp"][factor] for r_ in rs})
        cols = plt.cm.viridis(np.linspace(0, 0.9, max(len(vals), 2)))
        for v, c in zip(vals, cols):
            for r_ in rs:
                if r_["hp"][factor] == v:
                    y = np.array(r_["loss_curve_every10"]); x = np.arange(len(y)) * 10; x[-1] = r_["steps"]
                    ax.plot(x, y, color=c, lw=0.8, label=f"{factor}={v}" if r_["seed"] == 0 else None)
        ax.set_yscale("log"); ax.set_title(factor, fontsize=9); ax.set_xlabel("Adam step"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    fig.suptitle(f"{exp}: training loss (MSE) per factor (thin lines = seeds)", fontsize=10)
    fig.tight_layout()
    fn = os.path.join(fig_dir, f"{exp}_losses.png"); fig.savefig(fn, dpi=120); plt.close(fig); figures.append(fn)

    # ---- 7. block-structure check across all runs ---------------------------------------------------
    chk = {"max_lower_block_norm_overall": float(max(get(r_[w], "max_lower_block_norm") for r_ in runs for w in ("init", "end"))),
           "max_eig_union_err_overall": float(max(get(r_[w], "eig_union_err") for r_ in runs for w in ("init", "end"))),
           "max_readout_block_vs_I_kron_Sigma_rel_err": float(max(get(r_[w], "readout_block_vs_I_kron_Sigma_rel_err") for r_ in runs for w in ("init", "end"))),
           "max_abs_imag_over_max_re_overall": float(max(get(r_[w], "max_abs_imag_over_max_re") for r_ in runs for w in ("init", "end"))),
           "frac_evals_where_maxRe_is_readout_block": float(np.mean([get(r_[w], "max_re_eigH_is_readout") for r_ in runs for w in ("init", "end")])),
           "max_asym_Ho": float(max(get(r_[w], "asym_Ho") for r_ in runs for w in ("init", "end")))}
    analysis["structure_checks"] = chk
    analysis["tables_markdown"] = tables
    analysis["figures"] = figures
    with open(os.path.join(res_dir, "analysis.json"), "w") as f:
        json.dump(to_py(analysis), f, indent=1)
    for k_, t_ in tables.items():
        print(f"\n### {k_}\n{t_}")
    print("structure checks:", json.dumps(chk, indent=1))
    return analysis


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    exp = EXP + ("_smoke" if args.smoke else "")
    res_dir = os.path.join(ROOT, "results", exp)
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(res_dir, exist_ok=True)
    if args.figures:
        make_figures(res_dir, fig_dir, exp)
        return
    steps = 20 if args.smoke else STEPS
    T0 = time.time()
    if args.smoke:
        validate_once(dict(BASE))
    phases = plan(args.smoke)
    runs: List[Dict[str, object]] = []
    walls: Dict[str, float] = {}
    reductions: List[str] = []
    n_total = sum(len(p_[1]) for p_ in phases)
    print(f"planned runs: {n_total}; phases: {[(p_[0], len(p_[1])) for p_ in phases]}", flush=True)
    for pname, items, guarded in phases:
        for hp, seed in items:
            elapsed = (time.time() - T0) / 60.0
            est = estimate_cost(hp, walls) / 60.0
            if guarded and elapsed + est > BUDGET_MIN:
                msg = f"SKIPPED {tag_of(hp)}_L{hp['L']}_seed{seed} (phase {pname}): elapsed {elapsed:.1f} min + est {est:.1f} min > budget {BUDGET_MIN} min"
                print(msg, flush=True); reductions.append(msg)
                continue
            try:
                info = run_one(hp, seed, steps, res_dir)
            except Exception as ex:  # keep going; report
                msg = f"FAILED {tag_of(hp)}_L{hp['L']}_seed{seed}: {type(ex).__name__}: {ex}"
                print(msg, flush=True); reductions.append(msg)
                continue
            info["phase"] = pname
            runs.append(info)
            walls[tag_of(hp) + ("" if hp["L"] == 4 else f"_L{hp['L']}")] = info["wall_sec"]
            with open(os.path.join(res_dir, "summary.json"), "w") as f:
                json.dump(runs, f, indent=1)
            with open(os.path.join(res_dir, "run_meta.json"), "w") as f:
                json.dump({"exp": exp, "steps": steps, "lr": LR, "base": BASE, "sweeps": SWEEPS, "depth_grid": {"k": DEPTH_K, "T": DEPTH_T, "L": DEPTH_L},
                           "seeds": SEEDS, "n_runs_done": len(runs), "n_runs_planned": n_total, "reductions": reductions,
                           "elapsed_min": (time.time() - T0) / 60.0, "budget_min": BUDGET_MIN}, f, indent=1)
            print(f"   [{len(runs)}/{n_total}] elapsed {(time.time() - T0) / 60.0:.1f} min", flush=True)
    print(f"RUNS DONE: {len(runs)}/{n_total}, elapsed {(time.time() - T0) / 60.0:.1f} min; reductions: {reductions}", flush=True)
    make_figures(res_dir, fig_dir, exp)
    with open(os.path.join(res_dir, "run_meta.json"), "r") as f:
        meta = json.load(f)
    meta["wall_minutes_total"] = (time.time() - T0) / 60.0
    with open(os.path.join(res_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"ALL DONE; total {(time.time() - T0) / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
