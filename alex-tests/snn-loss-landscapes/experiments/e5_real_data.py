import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""e5_real_data: a REAL classification task (sklearn digits, PCA-whitened to d_in=16, C=10 classes)
instead of synthetic whitened regression.

Data: sl.load_digits_dataset(d=16, whiten=True, test_frac=0.2, seed=0) -> 1437 train / 360 test.
Training is on the FULL training split (1437 samples; cheap: <= 0.15 s/step for the deepest SNN);
all Hessians are evaluated on a FIXED random subset of n=256 training samples (same subset for every run).
Kinds: "snn" (protocol defaults, kaiming), "relu" (kaiming), "linear" (gaussian gain 1).  L in {2,3,4,6}.
Seeds {0,1} (model init; data split / Hessian subset are fixed).  Losses: "mse" on one-hot targets
(primary, comparable to the paper) and "ce" (secondary; always at L=4, other L as the budget allows).
Adam lr 5e-3, 1000 full-batch steps (ALL kinds, incl. linear, as the task prescribes); H_o every 200 steps,
full H + H_o at init and end.  cluster_stats uses r=10 (rank of the one-hot target map: r^2=100,
bulk r(d_in+d_out-2r)=60); besides the paper's fixed-index statistics we report the data-driven
dominant count (largest log-gap), gap-separated counts and the gaps at indices 10 (=C) and 100 (=r^2).

Usage:  python3 experiments/e5_real_data.py            # full run (results/e5_real_data)
        python3 experiments/e5_real_data.py --smoke    # reduced run (results/e5_real_data_smoke)
        python3 experiments/e5_real_data.py --figures  # only regenerate figures + tables
The script RESUMES by default (runs present in summary.json with an existing raw npz are skipped);
--fresh starts over.
"""
import argparse
import json
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from snnland.hessian import per_output_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "e5_real_data"

N_HESS, D_IN, WIDTH, D_OUT, R = 256, 16, 16, 10, 10
LS = [2, 3, 4, 6]
KINDS = ["relu", "linear", "snn"]
SEEDS = [0, 1]
LR, STEPS, HO_EVERY = 5e-3, 1000, 200
FULL_H_STEPS = (0, 1000)
K_TOP = 100  # top eigenvectors whose layer energies are stored (npz); summary keeps the first 20
BUDGET_MIN = 62.0  # wall-clock budget (min) after which optional tier-2 runs are skipped
# per-run cost estimates (s, 1 thread, contended machine; from the timing probe): train 1000 steps on 1437
# samples + 2 full H + 6 H_o (+ eighs).  snn full H: ~3 s (L2), 8 s (L3), 17 s (L4), 55 s (L6).
EST_RUN_SEC = {"snn": {2: 60, 3: 100, 4: 150, 6: 300}, "relu": {2: 8, 3: 10, 4: 12, 6: 18},
               "linear": {2: 8, 3: 10, 4: 12, 6: 18}}
KIND_COL = {"relu": "tab:green", "linear": "tab:brown", "snn": "tab:red"}
LOSS_LS = {"mse": "-", "ce": "--"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_py(o):
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
    return spec.init_params(seed), "kaiming(default)"


def load_data(subset_seed: int = 0) -> Dict[str, torch.Tensor]:
    D = sl.load_digits_dataset(d=D_IN, whiten=True, test_frac=0.2, seed=0)
    n_tr = D["X_train"].shape[0]
    g = torch.Generator().manual_seed(subset_seed)
    idx = torch.randperm(n_tr, generator=g)[:N_HESS]
    D["idx"] = idx
    D["X_sub"], D["Y_sub"], D["y_sub"] = D["X_train"][idx], D["Y_train"][idx], D["y_train"][idx]
    return D


def data_sanity(D: Dict[str, torch.Tensor]) -> Dict[str, object]:
    Xs, Ys, ys = D["X_sub"], D["Y_sub"], D["y_sub"]
    assert Xs.shape == (N_HESS, D_IN) and Ys.shape == (N_HESS, D_OUT), (Xs.shape, Ys.shape)
    assert int(torch.linalg.matrix_rank(Ys)) == R, "rank of one-hot target subset != r=10"
    assert int(torch.unique(ys).numel()) == D_OUT, "subset misses a class"
    assert torch.equal(Ys.argmax(1), ys)
    n_tr = D["X_train"].shape[0]
    cov_full = D["X_train"].T @ D["X_train"] / n_tr
    cov_sub = Xs.T @ Xs / N_HESS
    ef = torch.linalg.eigvalsh(cov_full)
    es = torch.linalg.eigvalsh(cov_sub)
    # class-mean structure of the INPUTS (between-class share of the input second moment)
    mu_c = torch.stack([Xs[ys == c].mean(0) for c in range(D_OUT)])
    n_c = torch.bincount(ys, minlength=D_OUT).to(Xs.dtype)
    B = (mu_c.T * (n_c / N_HESS)) @ mu_c
    info = {
        "n_train_full": int(n_tr), "n_test": int(D["X_test"].shape[0]), "n_hess": N_HESS,
        "train_cov_eig_min": float(ef.min()), "train_cov_eig_max": float(ef.max()),
        "subset_cov_eig_min": float(es.min()), "subset_cov_eig_max": float(es.max()),
        "subset_class_counts": torch.bincount(ys, minlength=D_OUT).tolist(),
        "subset_input_between_class_trace_frac": float(torch.trace(B) / torch.trace(cov_sub)),
        "subset_input_between_class_eigs": [float(x) for x in torch.linalg.eigvalsh(B).flip(0)[:D_OUT]],
        "subset_idx_first10": D["idx"][:10].tolist(),
    }
    return info


FLOOR_REL = 1e-2  # gaps whose lower eigenvalue is < FLOOR_REL * lambda_max are rank/tail boundaries, not outlier structure
# (the paper's bulk is lambda_max / L >= lambda_max / 6, so any cluster relevant to C1-C5 lies far above 1% of lambda_max;
#  without the floor the largest log-gap of a smoothly decaying spectrum is always in its tail, e.g. linear L=2: gap 2.2x at index 158)


def _largest_log_gap_pos(e: np.ndarray, start: int, stop: int, floor: float = 0.0) -> Optional[int]:
    """Same as snnland.analysis._largest_log_gap but on an array that already excludes numerical zeros,
    considering only gaps whose lower eigenvalue e_{i+1} >= floor."""
    best, best_gap = None, -np.inf
    for i in range(start, min(stop, len(e) - 1)):
        if e[i + 1] < floor:
            break
        gap = np.log(e[i] / e[i + 1])
        if gap > best_gap:
            best, best_gap = i, gap
    return best


def gap_metrics(e_all: np.ndarray, window: int = 200, tol: float = 1e-8) -> Dict[str, object]:
    """Gap statistics of a descending spectrum, computed on the NON-ZERO part ``e = e_all[e_all > tol * e_all[0]]``.

    Work-around: ``sl.cluster_stats`` searches its data-driven split among the first ``4 r^2 = 400``
    eigenvalues *including* numerically-zero ones, so whenever rank(H_o) < 400 (always true for the
    linear kind, true for relu/snn at small L) it returns the rank boundary (split_n_dominant == rank,
    ratio_split ~ 1e15).  ``ndom_pos`` / ``ratio_split_pos`` replicate the library's two-gap procedure on
    the strictly positive eigenvalues (zero tolerance ``tol * lambda_max`` as in ``cluster_stats``)."""
    e = e_all[e_all > tol * max(float(e_all[0]), 0.0)] if len(e_all) and e_all[0] > 0 else e_all[:0]
    n = len(e)
    m = max(min(window, n - 1), 0)
    n_dom, n_bulk = R * R, R * (D_IN + D_OUT - 2 * R)
    floor = FLOOR_REL * float(e[0]) if n else 0.0
    idx = _largest_log_gap_pos(e, 0, min(4 * n_dom, n), floor)
    idx_nf = _largest_log_gap_pos(e, 0, min(4 * n_dom, n))
    if idx is None:
        pos = {"ndom_pos": None, "nbulk_pos": None, "ratio_split_pos": float("nan"), "ndom_pos_gap_ratio": float("nan"),
               "ndom_pos_lam": float("nan")}
    else:
        k1 = idx + 1
        idx2 = _largest_log_gap_pos(e, k1, min(k1 + 4 * n_bulk, n), floor)
        k2 = idx2 + 1 if idx2 is not None else int((e >= floor).sum())
        bulk = e[k1:k2]
        pos = {"ndom_pos": int(k1), "nbulk_pos": int(k2 - k1),
               "ratio_split_pos": float(e[:k1].mean() / bulk.mean()) if bulk.size else float("nan"),
               "ndom_pos_gap_ratio": float(e[idx] / e[idx + 1]), "ndom_pos_lam": float(e[idx])}
    pos["ndom_pos_nofloor"] = int(idx_nf + 1) if idx_nf is not None else None
    pos["n_above_floor"] = int((e >= floor).sum())
    if n < 2:
        return {"n_pos": int(n), **pos, "largest_gap_count_200": None, "largest_gap_ratio_200": None, "top3_gaps_200": [],
                "n_sep_gap2": 0, "n_sep_gap1p5": 0, "n_sep_gap1p2": 0}
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where((e[:m] > 0) & (e[1:m + 1] > 0), e[:m] / e[1:m + 1], np.nan)
    logr = np.log(ratios)
    order = np.argsort(-np.nan_to_num(logr, nan=-np.inf))[:3]
    top3 = [(int(i + 1), float(ratios[i])) for i in order if np.isfinite(ratios[i])]

    def n_sep(th):
        idx = np.where((ratios >= th) & (e[1:m + 1] >= floor))[0]
        return int(idx.max() + 1) if idx.size else 0

    def safe_ratio(i, j):
        return float(e[i] / e[j]) if n > j and e[j] > 0 else float("nan")

    def mean_ratio(a, b, c, d):
        num = e[a:b]
        den = e[c:d]
        return float(num.mean() / den.mean()) if num.size and den.size and den.mean() > 0 else float("nan")

    return {
        "n_pos": int(n), **pos,
        "largest_gap_count_200": top3[0][0] if top3 else None,
        "largest_gap_ratio_200": top3[0][1] if top3 else None,
        "top3_gaps_200": top3,  # [(count, ratio)]
        "n_sep_gap2": n_sep(2.0), "n_sep_gap1p5": n_sep(1.5), "n_sep_gap1p2": n_sep(1.2),
        "gap_at_1": safe_ratio(0, 1), "gap_at_9": safe_ratio(8, 9), "gap_at_10": safe_ratio(9, 10),
        "gap_at_90": safe_ratio(89, 90), "gap_at_100": safe_ratio(99, 100), "gap_at_160": safe_ratio(159, 160),
        "ratio_C10": mean_ratio(0, 10, 10, 100),           # top-10 vs the next 90
        "ratio_C10_vs_bulk60": mean_ratio(0, 10, 100, 160),  # top-10 vs the paper's bulk slot
        "ratio_r2_vs_bulk60": mean_ratio(0, 100, 100, 160),  # == cluster_stats ratio_paper for r=10
        "ratio_10_100_vs_bulk60": mean_ratio(10, 100, 100, 160),
        "lam_1": float(e[0]), "lam_10": float(e[9]) if n > 9 else float("nan"),
        "lam_100": float(e[99]) if n > 99 else float("nan"), "lam_160": float(e[159]) if n > 159 else float("nan"),
    }


def class_feature_stats(a: torch.Tensor, y: torch.Tensor) -> Dict[str, object]:
    """Class-structured second-moment decomposition of the readout features a (n, h):
    M = a^T a / n = B + W with B = sum_c (n_c/n) mu_c mu_c^T (class means, rank <= C) and W the within-class
    covariance; plus the overall-mean part mu mu^T.  The readout block of H_o (mse) is I_{d_out} (x) M."""
    n, h = a.shape
    mu = a.mean(0)
    M = a.T @ a / n
    mu_c = torch.stack([a[y == c].mean(0) for c in range(D_OUT)])
    n_c = torch.bincount(y, minlength=D_OUT).to(a.dtype)
    B = (mu_c.T * (n_c / n)) @ mu_c
    W = M - B
    M_e = sl.eigh(M)[0]
    B_e = sl.eigh(B)[0]
    W_e = sl.eigh(W)[0]
    trM = float(torch.trace(M))
    w_top = float(W_e[0])
    return {
        "M_eigs": [float(x) for x in M_e],
        "B_eigs_top10": [float(x) for x in B_e[:D_OUT]],
        "W_eigs_top3": [float(x) for x in W_e[:3]],
        "trace_M": trM, "trace_B": float(torch.trace(B)), "trace_W": float(torch.trace(W)),
        "mean_frac": float(mu @ mu) / trM if trM > 0 else float("nan"),
        "class_mean_frac": float(torch.trace(B)) / trM if trM > 0 else float("nan"),
        "n_M_eigs_above_W_top": int((M_e > w_top).sum()) if w_top > 0 else None,
        "n_B_eigs_above_W_top": int((B_e > w_top).sum()) if w_top > 0 else None,
        "M_eig1_over_eig2": float(M_e[0] / M_e[1]) if M_e[1] > 0 else float("nan"),
        "M_eig10_over_eig11": float(M_e[9] / M_e[10]) if h > 10 and M_e[10] > 0 else float("nan"),
        "feat_mean": float(a.mean()), "feat_std": float(a.std()),
        "feat_zero_frac": float((a == 0).to(torch.float64).mean()),
    }


def evaluate(spec, params, X, Y, y, loss: str) -> Dict[str, float]:
    with torch.no_grad():
        out = spec.forward(params, X)
        return {"loss": float(per_output_loss(out, Y, loss)),
                "acc": float((out.argmax(1) == y).to(torch.float64).mean()),
                "mse_loss": float(per_output_loss(out, Y, "mse"))}


def activity_diagnostics(spec, params, X) -> Dict[str, object]:
    out: Dict[str, object] = {}
    with torch.no_grad():
        if spec.kind == "relu":
            h = X
            active = []
            for l in range(1, spec.L):
                h = torch.relu(spec._affine(params, h, l))
                active.append(float((h > 0).to(torch.float64).mean()))
            out["per_layer_active_frac"] = active
            out["mean_active_frac"] = float(np.mean(active))
        else:
            out["per_layer_active_frac"] = [1.0] * (spec.L - 1)
            out["mean_active_frac"] = 1.0
    return out


def spectrum_summary(H: torch.Tensor, spec, tag: str):
    """One eigh of sym(H): cluster stats (r=10), gap metrics, coherence, top-k layer energies."""
    Hs = sl.symmetrize(H)
    eigs, evecs = sl.eigh(Hs)
    e = eigs.numpy()
    n = len(e)
    cs = sl.cluster_stats(e, R, D_IN, D_OUT)
    gaps = gap_metrics(e)
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
        f"gaps_{tag}": gaps,
        f"coh_{tag}": {
            "lambda_max": lam_max, "lambda_max_blocks": block_lams,
            "coherence": lam_max / max(block_lams) if max(block_lams) > 0 else float("nan"),
            "sum_block_lams_over_lambda_max": sum(block_lams) / lam_max if lam_max > 0 else float("nan"),
            "top_eigvec_layer_energy": energy, "block_alignment": align,
        },
        f"topk_energy_{tag}": topk[:20],
        f"last_layer_energy_top10_mean_{tag}": float(topk[:10, -1].mean()),
        f"last_layer_energy_10_100_mean_{tag}": float(topk[10:100, -1].mean()) if topk.shape[0] >= 100 else None,
        f"dominant_count_{tag}": gaps["ndom_pos"],  # data-driven count on the non-zero spectrum (see gap_metrics)
        f"dominant_count_lib_{tag}": cs["split_n_dominant"],  # library value (rank boundary when rank < 400)
        f"zero_frac_{tag}": cs["zero_count"] / n, f"neg_frac_{tag}": cs["negative_count"] / n,
        f"trace_{tag}": float(torch.trace(Hs)), f"fro_{tag}": float(torch.linalg.norm(H)),
        f"lambda_min_{tag}": float(eigs[-1]), f"top200_{tag}": [float(x) for x in e[:200]],
    }
    return out, e, topk


def validate_once(H_o: torch.Tensor, spec, s_o: Dict[str, object]):
    """Cross-check the hand-rolled summaries against the library helpers on one matrix."""
    lc = sl.layer_coherence(H_o, spec.layer_slices)
    assert abs(lc["coherence"] - s_o["coh_Ho"]["coherence"]) < 1e-8, (lc["coherence"], s_o["coh_Ho"]["coherence"])
    assert np.allclose(lc["lambda_max_blocks"], s_o["coh_Ho"]["lambda_max_blocks"], rtol=1e-8, atol=1e-10)
    te = sl.top_eigvec_layer_energy(H_o, spec.layer_slices, k=max(R * R, D_OUT)).numpy()
    assert np.allclose(np.sort(te[:20], axis=1), np.sort(np.asarray(s_o["topk_energy_Ho"]), axis=1), atol=1e-6), "topk energy mismatch"
    assert abs(np.asarray(s_o["topk_energy_Ho"]).sum(1) - 1).max() < 1e-8
    print("   validate_once: coherence / block lambda / top-k energies agree with sl.layer_coherence & sl.top_eigvec_layer_energy", flush=True)


# --------------------------------------------------------------------------- #
# per-checkpoint callback
# --------------------------------------------------------------------------- #
def make_callback(spec, D, loss: str, full_steps, raw: Dict[str, np.ndarray], validate: bool = False, verbose=True):
    Xs, Ys, ys = D["X_sub"], D["Y_sub"], D["y_sub"]
    state = {"validated": not validate}

    def cb(step, params, g, loss_value):
        t0 = time.time()
        rec: Dict[str, object] = {"step": int(step), "loss_train_full": float(loss_value),
                                  "grad_norm": float(torch.linalg.norm(g)), "P": int(spec.n_params)}
        rec["eval"] = {"train_full": evaluate(spec, params, D["X_train"], D["Y_train"], D["y_train"], loss),
                       "subset": evaluate(spec, params, Xs, Ys, ys, loss),
                       "test": evaluate(spec, params, D["X_test"], D["Y_test"], D["y_test"], loss)}
        rec["loss_hess_subset"] = rec["eval"]["subset"]["loss"]
        if not math.isfinite(loss_value) or loss_value > 1e6:
            rec["diverged"] = True
            print(f"    step {step:4d} loss={loss_value:.3e} DIVERGED -> no Hessian", flush=True)
            return {"rec": rec}
        H_o = sl.gauss_newton(spec, params, Xs, Ys, loss)
        s_o, e_o, topk_o = spectrum_summary(H_o, spec, "Ho")
        if not state["validated"]:
            validate_once(H_o, spec, s_o)
            state["validated"] = True
        rec.update(s_o)
        rec["asym_Ho"] = sl.asymmetry(H_o)
        raw[f"eig_Ho_step{step}"] = e_o
        raw[f"topk_energy_Ho_step{step}"] = topk_o
        rec["features"] = sl.feature_second_moment_stats(spec, params, Xs)
        with torch.no_grad():
            a = spec.hidden_features(params, Xs)
        rec["class_features"] = class_feature_stats(a, ys)
        raw[f"M_eigs_step{step}"] = np.asarray(rec["class_features"]["M_eigs"])
        rec["activity"] = activity_diagnostics(spec, params, Xs)
        rec["spikes"] = sl.spike_diagnostics(spec, params, Xs) if spec.is_spiking else None
        rec["wall_Ho"] = time.time() - t0
        if step in full_steps:
            t1 = time.time()
            H = sl.full_hessian(spec, params, Xs, Ys, loss)
            H_f = sl.functional_part(H, H_o)
            s_h, e_h, topk_h = spectrum_summary(H, spec, "H")
            rec.update(s_h)
            rec["asymmetry_H"] = sl.asymmetry(H)
            rec["Hf_norm"] = float(torch.linalg.norm(H_f))
            rec["Ho_norm"] = float(torch.linalg.norm(H_o))
            rec["H_norm"] = float(torch.linalg.norm(H))
            rec["Hf_over_Ho"] = rec["Hf_norm"] / rec["Ho_norm"] if rec["Ho_norm"] > 0 else float("nan")
            rec["asymmetry_Hf"] = sl.asymmetry(H_f)
            Hf_sym_e = sl.eigvals(H_f).numpy()
            rec["Hf_sym_fro_over_Ho"] = float(torch.linalg.norm(sl.symmetrize(H_f))) / rec["Ho_norm"]
            rec["Hf_sym_lambda_max"] = float(Hf_sym_e[0])
            rec["Hf_sym_lambda_min"] = float(Hf_sym_e[-1])
            rec["Hf_sym_lambda_absmax_over_Ho_lambda_max"] = float(np.abs(Hf_sym_e).max() / e_o[0]) if e_o[0] > 0 else None
            # lower-triangular block mass of H (e7: exactly 0 for snn)
            lower = 0.0
            for i, si in enumerate(spec.layer_slices):
                for j, sj in enumerate(spec.layer_slices):
                    if j < i:
                        lower += float(torch.linalg.norm(H[si, sj])) ** 2
            rec["H_lower_block_frac"] = math.sqrt(lower) / rec["H_norm"] if rec["H_norm"] > 0 else None
            raw[f"eig_H_step{step}"] = e_h
            raw[f"eig_Hf_sym_step{step}"] = Hf_sym_e
            raw[f"topk_energy_H_step{step}"] = topk_h
            rec["wall_H"] = time.time() - t1
        if verbose:
            act = rec["spikes"]["mean_rate"] if rec["spikes"] else rec["activity"]["mean_active_frac"]
            ev = rec["eval"]
            msg = (f"    step {step:4d} loss={loss_value:.4e} acc(tr/te)={ev['train_full']['acc']:.3f}/{ev['test']['acc']:.3f} "
                   f"lam_max(Ho)={rec['coh_Ho']['lambda_max']:.3e} coh={rec['coh_Ho']['coherence']:.2f} "
                   f"ndom={rec['dominant_count_Ho']} nsep2={rec['gaps_Ho']['n_sep_gap2']} gap10={rec['gaps_Ho']['gap_at_10']:.2f} "
                   f"gap100={rec['gaps_Ho']['gap_at_100']:.2f} ratio_split={rec['gaps_Ho']['ratio_split_pos']:.2f} "
                   f"ratioC10={rec['gaps_Ho']['ratio_C10']:.2f} act={act:.3f} wall={time.time() - t0:.1f}s")
            if step in full_steps:
                msg += f" | asym(H)={rec['asymmetry_H']:.1e} |Hf|/|Ho|={rec['Hf_over_Ho']:.3f} wall_H={rec['wall_H']:.1f}s"
            print(msg, flush=True)
        return {"rec": rec}

    return cb


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def run_one(kind: str, L: int, seed: int, loss: str, D, steps: int, ho_every: int, full_steps, res_dir: str,
            validate: bool = False, print_full_rec: bool = False) -> Dict[str, object]:
    spec = make_spec(kind, L)
    params, init_desc = make_init(spec, kind, seed)
    name = f"{kind}_{loss}_L{L}_seed{seed}"
    info: Dict[str, object] = {"kind": kind, "loss": loss, "r": R, "d_in": D_IN, "d_out": D_OUT, "L": L,
                               "seed": seed, "init": init_desc, "P": spec.n_params, "steps": steps,
                               "optimizer": "adam", "lr": LR, "spec": repr(spec), "n_hess": N_HESS,
                               "n_train": int(D["X_train"].shape[0]), "width": WIDTH, "name": name,
                               "ho_every": ho_every, "full_steps": list(full_steps)}
    print(f"== run {name}: P={spec.n_params} r={R} d_in={D_IN} d_out={D_OUT} {init_desc} Adam lr={LR} steps={steps}",
          flush=True)
    t0 = time.time()
    raw: Dict[str, np.ndarray] = {}
    cb = make_callback(spec, D, loss, full_steps, raw, validate=validate)
    hist = sl.gd_train(spec, params, D["X_train"], D["Y_train"], LR, steps, loss=loss, log_every=ho_every,
                       callbacks=[cb], optimizer="adam")
    recs = [s["rec"] for s in hist["summaries"]]
    if print_full_rec:
        print("---- FULL STATS DICT (first checkpoint) ----", flush=True)
        print(json.dumps(to_py(recs[0]), indent=1), flush=True)
        print("---- END FULL STATS DICT ----", flush=True)
    info["checkpoints"] = recs
    info["loss_init"] = hist["loss"][0]
    info["loss_final"] = hist["loss"][-1]
    info["final_eval"] = recs[-1]["eval"]
    info["init_eval"] = recs[0]["eval"]
    info["wall"] = time.time() - t0
    raw["loss_curve"] = np.asarray(hist["loss"])
    raw["grad_norm_curve"] = np.asarray(hist["grad_norm"])
    raw["logged_steps"] = np.asarray(hist["logged_steps"])
    raw["layer_slices"] = np.asarray([[s_.start, s_.stop] for s_ in spec.layer_slices])
    raw["subset_idx"] = D["idx"].numpy()
    for k_ in ("train_full", "subset", "test"):
        raw[f"acc_{k_}"] = np.asarray([r_["eval"][k_]["acc"] for r_ in recs])
        raw[f"loss_{k_}"] = np.asarray([r_["eval"][k_]["loss"] for r_ in recs])
    np.savez(os.path.join(res_dir, "raw", f"{name}.npz"), **raw)
    fe = info["final_eval"]
    print(f"   done {name}: loss {info['loss_init']:.4f} -> {info['loss_final']:.4f}, acc train {fe['train_full']['acc']:.3f} "
          f"test {fe['test']['acc']:.3f}, wall {info['wall']:.0f}s", flush=True)
    return info


# --------------------------------------------------------------------------- #
# main sweep
# --------------------------------------------------------------------------- #
def plan(smoke: bool):
    """Tiered run list: (tier, kind, L, seed, loss).  Tier 0 always runs; tiers >= 1 are budget-guarded."""
    runs = []
    if smoke:
        runs += [(0, "relu", 2, 0, "mse"), (0, "linear", 2, 0, "mse"), (0, "snn", 2, 0, "mse"), (0, "snn", 4, 0, "mse"),
                 (1, "relu", 2, 0, "ce"), (1, "snn", 2, 0, "ce")]
        return runs
    for seed in SEEDS:
        for kind in KINDS:
            for L in LS:
                runs.append((0, kind, L, seed, "mse"))
    for seed in SEEDS:  # CE at L=4 for all kinds, and all L for the cheap controls
        for kind in KINDS:
            runs.append((1, kind, 4, seed, "ce"))
        for kind in ["relu", "linear"]:
            for L in [x for x in LS if x != 4]:
                runs.append((1, kind, L, seed, "ce"))
    for seed in SEEDS:  # tier 2: SNN CE at the other depths
        for L in [2, 3, 6]:
            runs.append((2, "snn", L, seed, "ce"))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--res-dir", default=None)
    args = ap.parse_args()
    exp = EXP + ("_smoke" if args.smoke else "")
    res_dir = args.res_dir or os.path.join(ROOT, "results", exp)
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(os.path.join(res_dir, "raw"), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    summ_path = os.path.join(res_dir, "summary.json")
    if args.figures:
        make_figures(res_dir, fig_dir, exp)
        return
    steps, ho_every, full_steps = (200, 100, (0, 200)) if args.smoke else (STEPS, HO_EVERY, FULL_H_STEPS)
    torch.manual_seed(0)
    D = load_data()
    dinfo = data_sanity(D)
    print("data:", json.dumps(dinfo), flush=True)
    rows: List[Dict[str, object]] = []
    if os.path.exists(summ_path) and not args.fresh:
        rows = json.load(open(summ_path))
        rows = [r_ for r_ in rows if os.path.exists(os.path.join(res_dir, "raw", f"{r_['name']}.npz"))]
        print(f"resuming: {len(rows)} runs already done", flush=True)
    done = {r_["name"] for r_ in rows}
    meta = {"exp": exp, "data": dinfo, "steps": steps, "ho_every": ho_every, "full_steps": list(full_steps),
            "lr": LR, "optimizer": "adam", "r": R, "d_in": D_IN, "d_out": D_OUT, "n_hess": N_HESS,
            "train_on": "full training split (1437)", "skipped": [], "budget_min": BUDGET_MIN}
    t_start = time.time()
    first = True
    for tier, kind, L, seed, loss in plan(args.smoke):
        name = f"{kind}_{loss}_L{L}_seed{seed}"
        if name in done:
            continue
        elapsed = (time.time() - t_start) / 60
        est = EST_RUN_SEC[kind][L] / 60
        if tier >= 1 and not args.smoke and elapsed + est > BUDGET_MIN:
            print(f"SKIPPED (budget: elapsed {elapsed:.1f} min + est {est:.1f} > {BUDGET_MIN}): {name}", flush=True)
            meta["skipped"].append(name)
            continue
        info = run_one(kind, L, seed, loss, D, steps, ho_every, full_steps, res_dir,
                       validate=first, print_full_rec=first and args.smoke)
        first = False
        info["tier"] = tier
        rows.append(info)
        json.dump(to_py(rows), open(summ_path, "w"))
        json.dump(to_py(meta), open(os.path.join(res_dir, "meta.json"), "w"), indent=1)
        print(f"   [elapsed {(time.time() - t_start) / 60:.1f} min, {len(rows)} runs saved]", flush=True)
    meta["total_wall_min"] = (time.time() - t_start) / 60
    json.dump(to_py(meta), open(os.path.join(res_dir, "meta.json"), "w"), indent=1)
    print(f"ALL RUNS DONE in {meta['total_wall_min']:.1f} min; skipped: {meta['skipped']}", flush=True)
    make_figures(res_dir, fig_dir, exp)


# --------------------------------------------------------------------------- #
# figures + tables
# --------------------------------------------------------------------------- #
def _ck(row, step):
    for c in row["checkpoints"]:
        if c["step"] == step:
            return c
    return None


def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _legend(ax, **kw):
    h, l_ = ax.get_legend_handles_labels()
    if h:
        ax.legend(**kw)


def _mean_std(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return (float(v.mean()), float(v.std()), len(v)) if v.size else (float("nan"), float("nan"), 0)


def make_figures(res_dir: str, fig_dir: str, exp: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(os.path.join(res_dir, "summary.json")))
    end_step = max(c["step"] for r_ in rows for c in r_["checkpoints"])
    kinds = [k for k in KINDS if any(r_["kind"] == k for r_ in rows)]
    Ls = sorted({r_["L"] for r_ in rows})
    losses = [l_ for l_ in ["mse", "ce"] if any(r_["loss"] == l_ for r_ in rows)]

    def raw_of(row):
        return np.load(os.path.join(res_dir, "raw", f"{row['name']}.npz"))

    def sel(kind=None, L=None, loss=None, seed=None):
        return [r_ for r_ in rows if (kind is None or r_["kind"] == kind) and (L is None or r_["L"] == L)
                and (loss is None or r_["loss"] == loss) and (seed is None or r_["seed"] == seed)]

    # ---- F1: spectra per kind and L, init vs end, H_o and sym(H), per loss ----
    for loss in losses:
        for mat in ["Ho", "H"]:
            Ls_here = sorted({r_["L"] for r_ in sel(loss=loss)})
            fig, axes = plt.subplots(len(kinds), len(Ls_here), figsize=(3.6 * len(Ls_here), 2.9 * len(kinds)), squeeze=False)
            for i, kind in enumerate(kinds):
                for j, L in enumerate(Ls_here):
                    ax = axes[i, j]
                    for row in sel(kind, L, loss):
                        rw = raw_of(row)
                        for step, ls, lab in [(0, "--", "init"), (end_step, "-", f"step {end_step}")]:
                            key = f"eig_{mat}_step{step}"
                            if key not in rw:
                                continue
                            e = rw[key]
                            pos = e > 0
                            idx = np.arange(1, len(e) + 1)
                            ax.plot(idx[pos], e[pos], ls, color=KIND_COL[kind], lw=0.9, alpha=0.9 if row["seed"] == 0 else 0.5,
                                    label=f"{lab} s{row['seed']}")
                            neg = e < 0
                            if neg.any() and mat == "H":
                                ax.plot(idx[neg], -e[neg], ls, color="gray", lw=0.6, alpha=0.6, label="|neg| s%d" % row["seed"] if step == end_step else None)
                    for x_, c_ in [(10, "k"), (100, "b"), (160, "c")]:
                        ax.axvline(x_, color=c_, lw=0.7, ls=":")
                    ax.set_xscale("log"); ax.set_yscale("log")
                    c0 = [_ck(r_, end_step) for r_ in sel(kind, L, loss)]
                    nd = [c[f"dominant_count_{mat}"] for c in c0 if c and c.get(f"dominant_count_{mat}") is not None]
                    ax.set_title(f"{kind} L={L} ({loss}) ndom_end={nd}", fontsize=9)
                    if i == len(kinds) - 1:
                        ax.set_xlabel("eigenvalue index (dotted: 10, 100, 160)")
                    if j == 0:
                        ax.set_ylabel(f"eig {'H_o' if mat == 'Ho' else 'sym(H)'}")
                    _legend(ax, fontsize=6, loc="lower left")
            fig.suptitle(f"{exp}: spectra of {'H_o' if mat == 'Ho' else 'sym(H)'} on digits (n=256, d_in=16, C=10), loss={loss}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"{exp}_spectra_{mat}_{loss}.png"), dpi=120)
            plt.close(fig)

    # ---- F2: outlier counts vs L (init / end), H_o, per loss ----
    count_keys = [("dominant_count_Ho", "data-driven dominant (largest log-gap)"),
                  ("gaps_Ho.n_sep_gap2", "# separated by a gap >= 2x"),
                  ("gaps_Ho.n_sep_gap1p5", "# separated by a gap >= 1.5x"),
                  ("gaps_Ho.largest_gap_count_200", "largest gap in top-200 (count)"),
                  ("gaps_Ho.ndom_pos_gap_ratio", "gap ratio e_k/e_k+1 at the data-driven split")]

    def getk(c, key):
        return _get(c, *key.split("."))

    for loss in losses:
        fig, axes = plt.subplots(2, len(count_keys), figsize=(3.8 * len(count_keys), 6.2), squeeze=False)
        for a_, (step, lab) in enumerate([(0, "init"), (end_step, f"end (step {end_step})")]):
            for b_, (key, title) in enumerate(count_keys):
                ax = axes[a_, b_]
                for kind in kinds:
                    Ls_k = sorted({r_["L"] for r_ in sel(kind, loss=loss)})
                    ms = [_mean_std([getk(_ck(r_, step), key) for r_ in sel(kind, L, loss)]) for L in Ls_k]
                    ax.errorbar(Ls_k, [m[0] for m in ms], yerr=[m[1] for m in ms], color=KIND_COL[kind], marker="o", capsize=3, label=kind)
                    for r_ in sel(kind, loss=loss):
                        v = getk(_ck(r_, step), key)
                        if v is not None:
                            ax.plot([r_["L"]], [v], ".", color=KIND_COL[kind], alpha=0.5, ms=4)
                if "ratio" in key:
                    ax.set_yscale("log"); ax.axhline(2, color="gray", ls=":", lw=0.8, label="2x")
                else:
                    ax.axhline(10, color="k", ls=":", lw=0.8, label="C=10")
                    ax.axhline(100, color="b", ls=":", lw=0.8, label="r^2=100")
                    ax.set_yscale("symlog", linthresh=10)
                ax.set_xlabel("L"); ax.set_title(f"{title}\n{lab}, H_o, {loss}", fontsize=9)
                _legend(ax, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_outlier_count_vs_L_{loss}.png"), dpi=120)
        plt.close(fig)

    # ---- F3: ratios and coherence vs L ----
    ratio_keys = [("gaps_Ho.ratio_split_pos", "ratio_split (data-driven dom/bulk, nonzero)"),
                  ("gaps_Ho.ratio_C10", "mean(top10)/mean(10..100)"),
                  ("cluster_Ho.ratio_paper", "paper r=10: mean(top100)/mean(100..160)"),
                  ("gaps_Ho.gap_at_10", "gap e10/e11"), ("gaps_Ho.gap_at_100", "gap e100/e101"),
                  ("coh_Ho.coherence", "coherence lam_max/max_l lam_max(B_ll)")]
    for loss in losses:
        fig, axes = plt.subplots(2, len(ratio_keys), figsize=(3.4 * len(ratio_keys), 6.2), squeeze=False)
        for a_, (step, lab) in enumerate([(0, "init"), (end_step, f"end")]):
            for b_, (key, title) in enumerate(ratio_keys):
                ax = axes[a_, b_]
                for kind in kinds:
                    Ls_k = sorted({r_["L"] for r_ in sel(kind, loss=loss)})
                    ms = [_mean_std([getk(_ck(r_, step), key) for r_ in sel(kind, L, loss)]) for L in Ls_k]
                    ax.errorbar(Ls_k, [m[0] for m in ms], yerr=[m[1] for m in ms], color=KIND_COL[kind], marker="o", capsize=3, label=kind)
                if "coherence" in key or "ratio" in key:
                    ax.plot(Ls, Ls, "k:", lw=0.8, label="= L")
                ax.set_yscale("log")
                ax.set_xlabel("L"); ax.set_title(f"{title}\n{lab}, H_o, {loss}", fontsize=8)
                _legend(ax, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_ratio_coherence_vs_L_{loss}.png"), dpi=120)
        plt.close(fig)

    # ---- F4: top-eigenvector layer energy (H_o, end and init), mse, seed 0 ----
    for loss in losses:
        for step, lab in [(0, "init"), (end_step, "end")]:
            Ls_here = sorted({r_["L"] for r_ in sel(loss=loss)})
            fig, axes = plt.subplots(len(kinds), len(Ls_here), figsize=(3.6 * len(Ls_here), 2.6 * len(kinds)), squeeze=False)
            for i, kind in enumerate(kinds):
                for j, L in enumerate(Ls_here):
                    ax = axes[i, j]
                    rs = sel(kind, L, loss, seed=0) or sel(kind, L, loss)
                    if not rs:
                        ax.axis("off"); continue
                    rw = raw_of(rs[0])
                    key = f"topk_energy_Ho_step{step}"
                    if key not in rw:
                        ax.axis("off"); continue
                    E = rw[key][:30]  # (k, L)
                    bottom = np.zeros(E.shape[0])
                    cmap = plt.get_cmap("viridis")
                    for l_ in range(E.shape[1]):
                        ax.bar(np.arange(1, E.shape[0] + 1), E[:, l_], bottom=bottom, color=cmap(l_ / max(E.shape[1] - 1, 1)),
                               width=0.9, label=f"W{l_ + 1}" + (" (readout)" if l_ == E.shape[1] - 1 else ""))
                        bottom += E[:, l_]
                    ax.axvline(10.5, color="k", ls=":", lw=0.8)
                    ax.set_ylim(0, 1); ax.set_title(f"{kind} L={L} ({loss}, {lab})", fontsize=9)
                    if j == 0:
                        ax.set_ylabel("layer energy of eigvec k")
                    if i == len(kinds) - 1:
                        ax.set_xlabel("eigenvector index k (H_o)")
                    _legend(ax, fontsize=5, ncol=2, loc="lower right")
            fig.suptitle(f"{exp}: per-layer energy of the top-30 eigenvectors of H_o ({loss}, {lab}, seed 0)", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"{exp}_top_eigvec_layer_energy_{loss}_{lab}.png"), dpi=120)
            plt.close(fig)

    # ---- F5: readout-feature statistics ----
    feat_keys = [("class_features.mean_frac", "||mu||^2 / tr M (overall-mean share)"),
                 ("class_features.class_mean_frac", "tr B / tr M (class-mean share)"),
                 ("class_features.n_M_eigs_above_W_top", "# eig(M) above lam_max(W_within)"),
                 ("class_features.M_eig1_over_eig2", "eig1(M)/eig2(M)"),
                 ("class_features.M_eig10_over_eig11", "eig10(M)/eig11(M)")]
    for loss in losses:
        fig, axes = plt.subplots(2, len(feat_keys) + 1, figsize=(3.3 * (len(feat_keys) + 1), 6.2), squeeze=False)
        for a_, (step, lab) in enumerate([(0, "init"), (end_step, "end")]):
            for b_, (key, title) in enumerate(feat_keys):
                ax = axes[a_, b_]
                for kind in kinds:
                    Ls_k = sorted({r_["L"] for r_ in sel(kind, loss=loss)})
                    ms = [_mean_std([getk(_ck(r_, step), key) for r_ in sel(kind, L, loss)]) for L in Ls_k]
                    ax.errorbar(Ls_k, [m[0] for m in ms], yerr=[m[1] for m in ms], color=KIND_COL[kind], marker="o", capsize=3, label=kind)
                ax.set_xlabel("L"); ax.set_title(f"{title}\n{lab}, {loss}", fontsize=8); _legend(ax, fontsize=7)
                if "over" in key:
                    ax.set_yscale("log")
            ax = axes[a_, len(feat_keys)]
            for kind in kinds:
                for row in sel(kind, 4, loss):
                    rw = raw_of(row)
                    key = f"M_eigs_step{step}"
                    if key in rw:
                        m_ = rw[key]
                        ax.plot(np.arange(1, len(m_) + 1), np.maximum(m_, 1e-16), "o-", color=KIND_COL[kind], ms=3, lw=0.8,
                                alpha=0.9 if row["seed"] == 0 else 0.5, label=f"{kind} s{row['seed']}")
                    cf = _ck(row, step)["class_features"]
                    ax.plot(np.arange(1, 11), np.maximum(cf["B_eigs_top10"], 1e-16), "x:", color=KIND_COL[kind], ms=3, lw=0.6,
                            alpha=0.9 if row["seed"] == 0 else 0.5)
            ax.axvline(10, color="k", ls=":", lw=0.8)
            ax.set_yscale("log"); ax.set_title(f"eig(M) (o) and eig(B) (x), L=4, {lab}", fontsize=8); ax.set_xlabel("index"); _legend(ax, fontsize=6)
        fig.suptitle(f"{exp}: readout-feature second moment M = a^T a/n = B(class means) + W(within), {loss}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_feature_stats_{loss}.png"), dpi=120)
        plt.close(fig)

    # ---- F6: asymmetry and H_f / H_o ----
    asym_keys = [("asymmetry_H", "asymmetry ||H-H^T||/||H||"), ("Hf_over_Ho", "||H_f||_F / ||H_o||_F"),
                 ("Hf_sym_lambda_absmax_over_Ho_lambda_max", "max|eig sym(H_f)| / lam_max(H_o)"),
                 ("H_lower_block_frac", "lower-block mass of H")]
    fig, axes = plt.subplots(len(losses), len(asym_keys), figsize=(3.6 * len(asym_keys), 3.1 * len(losses)), squeeze=False)
    for a_, loss in enumerate(losses):
        for b_, (key, title) in enumerate(asym_keys):
            ax = axes[a_, b_]
            for kind in kinds:
                for step, ls, lab in [(0, "--", "init"), (end_step, "-", "end")]:
                    Ls_k = sorted({r_["L"] for r_ in sel(kind, loss=loss)})
                    ms = [_mean_std([_ck(r_, step).get(key) for r_ in sel(kind, L, loss)]) for L in Ls_k]
                    ax.errorbar(Ls_k, [m[0] for m in ms], yerr=[m[1] for m in ms], color=KIND_COL[kind], ls=ls, marker="o", ms=4,
                                capsize=3, label=f"{kind} {lab}")
            ax.set_yscale("log"); ax.set_xlabel("L"); ax.set_title(f"{title}, {loss}", fontsize=8); _legend(ax, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_asym_Hf.png"), dpi=120)
    plt.close(fig)

    # ---- F7: CE vs MSE at L=4 ----
    if "ce" in losses:
        fig, axes = plt.subplots(2, len(kinds), figsize=(4.2 * len(kinds), 6.2), squeeze=False)
        for j, kind in enumerate(kinds):
            for a_, mat in enumerate(["Ho", "H"]):
                ax = axes[a_, j]
                for loss in losses:
                    for row in sel(kind, 4, loss, seed=0):
                        rw = raw_of(row)
                        for step, ls in [(0, ":"), (end_step, "-")]:
                            e = rw[f"eig_{mat}_step{step}"]
                            pos = e > 0
                            idx = np.arange(1, len(e) + 1)
                            ax.plot(idx[pos], e[pos] / e[0], ls, color="tab:blue" if loss == "mse" else "tab:orange", lw=1.0,
                                    label=f"{loss} {'init' if step == 0 else 'end'}")
                for x_ in (10, 100):
                    ax.axvline(x_, color="k", ls=":", lw=0.7)
                ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{kind} L=4, {'H_o' if mat == 'Ho' else 'sym(H)'} / lam_max", fontsize=9)
                _legend(ax, fontsize=7); ax.set_xlabel("index")
        fig.suptitle(f"{exp}: CE vs MSE spectra at L=4 (seed 0, normalised by lambda_max)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_ce_vs_mse_L4.png"), dpi=120)
        plt.close(fig)

    # ---- F8: training curves / accuracy ----
    fig, axes = plt.subplots(len(losses), 3, figsize=(12, 3.2 * len(losses)), squeeze=False)
    for a_, loss in enumerate(losses):
        for row in sel(loss=loss):
            rw = raw_of(row)
            lw = 0.9 if row["seed"] == 0 else 0.5
            alpha = {2: 0.35, 3: 0.55, 4: 0.75, 6: 1.0}.get(row["L"], 0.8)
            axes[a_, 0].plot(rw["loss_curve"], color=KIND_COL[row["kind"]], lw=lw, alpha=alpha, label=f"{row['kind']} L{row['L']}" if row["seed"] == 0 else None)
            axes[a_, 1].plot(rw["logged_steps"], rw["acc_train_full"], "o-", color=KIND_COL[row["kind"]], lw=lw, alpha=alpha, ms=3)
            axes[a_, 2].plot(rw["logged_steps"], rw["acc_test"], "o-", color=KIND_COL[row["kind"]], lw=lw, alpha=alpha, ms=3)
        axes[a_, 0].set_yscale("log"); axes[a_, 0].set_title(f"train loss ({loss}, full split)"); _legend(axes[a_, 0], fontsize=6, ncol=3)
        axes[a_, 1].set_title("train accuracy (full split)"); axes[a_, 2].set_title("test accuracy")
        for ax in axes[a_]:
            ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_training.png"), dpi=120)
    plt.close(fig)

    # ---- F9: H_o dynamics every 200 steps ----
    dyn_keys = [("coh_Ho.lambda_max", "lam_max(H_o)"), ("dominant_count_Ho", "data-driven dominant count"),
                ("gaps_Ho.ratio_split_pos", "ratio_split (nonzero)"), ("gaps_Ho.gap_at_10", "gap e10/e11"),
                ("gaps_Ho.gap_at_100", "gap e100/e101"), ("coh_Ho.coherence", "coherence")]
    for loss in losses:
        fig, axes = plt.subplots(len(kinds), len(dyn_keys), figsize=(3.2 * len(dyn_keys), 2.7 * len(kinds)), squeeze=False)
        for i, kind in enumerate(kinds):
            for b_, (key, title) in enumerate(dyn_keys):
                ax = axes[i, b_]
                for row in sel(kind, loss=loss):
                    steps_ = [c["step"] for c in row["checkpoints"]]
                    vals = [getk(c, key) for c in row["checkpoints"]]
                    vals = [v if v is not None else np.nan for v in vals]
                    alpha = {2: 0.35, 3: 0.55, 4: 0.75, 6: 1.0}.get(row["L"], 0.8)
                    ax.plot(steps_, vals, "o-", color=KIND_COL[kind], alpha=alpha, lw=0.9 if row["seed"] == 0 else 0.5, ms=3,
                            label=f"L{row['L']}" if row["seed"] == 0 else None)
                if "count" in key:
                    ax.set_yscale("symlog", linthresh=10); ax.axhline(10, color="k", ls=":", lw=0.7); ax.axhline(100, color="b", ls=":", lw=0.7)
                else:
                    ax.set_yscale("log")
                ax.set_title(f"{kind}: {title} ({loss})", fontsize=8); _legend(ax, fontsize=6)
                if i == len(kinds) - 1:
                    ax.set_xlabel("step")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_Ho_dynamics_{loss}.png"), dpi=120)
        plt.close(fig)

    print_tables(rows, end_step)


def print_tables(rows, end_step):
    def fmt(ms, nd=2):
        m, s, n = ms
        if n == 0:
            return "nan"
        return f"{m:.{nd}f}" + (f"±{s:.{nd}f}" if n > 1 else "")

    for loss in ["mse", "ce"]:
        rs = [r_ for r_ in rows if r_["loss"] == loss]
        if not rs:
            continue
        print(f"\n### {loss.upper()}: per (kind, L), mean±std over seeds; H_o on the fixed 256-subset; end = step {end_step}")
        print("| kind | L | P | n_seeds | rank(Ho) end | train acc | test acc | loss_end(full) | ndom init | ndom end | nsep2 init | nsep2 end | gap10 init | gap10 end | gap100 init | gap100 end | ratio_split_pos init | ratio_split_pos end | ratioC10 end | ratio_paper(r=10) end | coh init | coh end | asym(H) end | Hf/Ho init | Hf/Ho end | mean_frac end | class_mean_frac end | #eigM>W end |")
        print("|" + "---|" * 28)
        for kind in KINDS:
            for L in LS:
                g = [r_ for r_ in rs if r_["kind"] == kind and r_["L"] == L]
                if not g:
                    continue
                c0 = [_ck(r_, 0) for r_ in g]
                c1 = [_ck(r_, end_step) for r_ in g]
                vals = [kind, L, g[0]["P"], len(g), fmt(_mean_std([c["gaps_Ho"]["n_pos"] for c in c1]), 0),
                        fmt(_mean_std([r_["final_eval"]["train_full"]["acc"] for r_ in g]), 3),
                        fmt(_mean_std([r_["final_eval"]["test"]["acc"] for r_ in g]), 3),
                        fmt(_mean_std([r_["loss_final"] for r_ in g]), 4),
                        fmt(_mean_std([c["dominant_count_Ho"] for c in c0]), 1), fmt(_mean_std([c["dominant_count_Ho"] for c in c1]), 1),
                        fmt(_mean_std([c["gaps_Ho"]["n_sep_gap2"] for c in c0]), 1), fmt(_mean_std([c["gaps_Ho"]["n_sep_gap2"] for c in c1]), 1),
                        fmt(_mean_std([c["gaps_Ho"]["gap_at_10"] for c in c0])), fmt(_mean_std([c["gaps_Ho"]["gap_at_10"] for c in c1])),
                        fmt(_mean_std([c["gaps_Ho"]["gap_at_100"] for c in c0])), fmt(_mean_std([c["gaps_Ho"]["gap_at_100"] for c in c1])),
                        fmt(_mean_std([c["gaps_Ho"]["ratio_split_pos"] for c in c0])), fmt(_mean_std([c["gaps_Ho"]["ratio_split_pos"] for c in c1])),
                        fmt(_mean_std([c["gaps_Ho"]["ratio_C10"] for c in c1])), fmt(_mean_std([c["cluster_Ho"]["ratio_paper"] for c in c1])),
                        fmt(_mean_std([c["coh_Ho"]["coherence"] for c in c0])), fmt(_mean_std([c["coh_Ho"]["coherence"] for c in c1])),
                        fmt(_mean_std([c.get("asymmetry_H") for c in c1]), 3),
                        fmt(_mean_std([c.get("Hf_over_Ho") for c in c0]), 3), fmt(_mean_std([c.get("Hf_over_Ho") for c in c1]), 3),
                        fmt(_mean_std([c["class_features"]["mean_frac"] for c in c1])), fmt(_mean_std([c["class_features"]["class_mean_frac"] for c in c1])),
                        fmt(_mean_std([c["class_features"]["n_M_eigs_above_W_top"] for c in c1]), 1)]
                print("| " + " | ".join(str(v) for v in vals) + " |")
        print(f"\n### {loss.upper()}: sym(H) at end: ndom, nsep2, gap10, gap100, neg_frac, last-layer energy of top-10 eigvecs (H_o end)")
        print("| kind | L | ndom_H end | nsep2_H end | gap10_H end | gap100_H end | neg_frac_H end | lam_min_H end | E_last(top10,Ho) end | E_last(10..100,Ho) end | E_last(top10,Ho) init | spikes rate end | window end |")
        print("|" + "---|" * 13)
        for kind in KINDS:
            for L in LS:
                g = [r_ for r_ in rows if r_["loss"] == loss and r_["kind"] == kind and r_["L"] == L]
                if not g:
                    continue
                c0 = [_ck(r_, 0) for r_ in g]
                c1 = [_ck(r_, end_step) for r_ in g]
                sp = [c["spikes"] for c in c1 if c.get("spikes")]
                vals = [kind, L, fmt(_mean_std([c.get("dominant_count_H") for c in c1]), 1), fmt(_mean_std([_get(c, "gaps_H", "n_sep_gap2") for c in c1]), 1),
                        fmt(_mean_std([_get(c, "gaps_H", "gap_at_10") for c in c1])), fmt(_mean_std([_get(c, "gaps_H", "gap_at_100") for c in c1])),
                        fmt(_mean_std([c.get("neg_frac_H") for c in c1]), 3), fmt(_mean_std([c.get("lambda_min_H") for c in c1]), 3),
                        fmt(_mean_std([c["last_layer_energy_top10_mean_Ho"] for c in c1])), fmt(_mean_std([c["last_layer_energy_10_100_mean_Ho"] for c in c1])),
                        fmt(_mean_std([c["last_layer_energy_top10_mean_Ho"] for c in c0])),
                        fmt(_mean_std([s["mean_rate"] for s in sp]), 3) if sp else "-", fmt(_mean_std([s["window_frac"] for s in sp]), 3) if sp else "-"]
                print("| " + " | ".join(str(v) for v in vals) + " |")


if __name__ == "__main__":
    main()
