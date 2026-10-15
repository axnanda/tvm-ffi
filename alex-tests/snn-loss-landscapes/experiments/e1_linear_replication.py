import snnland as sl; sl.set_threads(1)  # noqa: E702  (protocol: first line, 1 thread)
"""e1_linear_replication: replicate arXiv 2602.00545 ("Depth, Not Data") on deep LINEAR nets.

Reference experiment for the SNN comparisons.  Parts:
 (1) balanced_uniform init (s=1, tail 0): H_o and full H at init, L in {1,2,3,4,5,6,8}, settings
     T=(r=2,d_out=4) [+ A=(1,4), B=(2,6) at L in {2,4,8}] -> C1 counts, C2 ratio==L, coherence==L.
 (2) non-uniform per-layer s_l at L=4 -> ratio vs closed form sum_l prod_{m!=l} s_m^2 / max(...).
 (3) tail_value in {0, 0.05, 0.2} at L=4 -> degradation of the clean structure.
 (4) GD (lr=0.05) from balanced init, L=3 and L=5, 2000 steps, H_o AND full H every 25 steps
     -> C3 (exponential decay of ||H_f||_F), C5 (ratio, coherence, balancedness over training).
 (4b, extra) GD reference numbers for settings T, A, B at L in {2,4,8} (1000 steps, every 50).
 (5) gaussian init (gain 1), L in {2,4,6,8}, 2 seeds, GD to convergence (loss<1e-6 or 3000 steps; the
     state at step 3000 is recorded and non-converged runs are continued up to 15000 steps).
Usage:  python3 experiments/e1_linear_replication.py            (full run)
        E1_SMOKE=1 python3 experiments/e1_linear_replication.py  (reduced smoke test)
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/home/user/tvm-ffi/alex-tests/snn-loss-landscapes"
EXP = "e1_linear_replication"
SMOKE = os.environ.get("E1_SMOKE", "0") == "1"
RES = f"{ROOT}/results/{EXP}" + ("/smoke" if SMOKE else "")
RAW = f"{RES}/raw"
FIG = f"{RES}/figures" if SMOKE else f"{ROOT}/figures"
os.makedirs(RAW, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

N, D_IN, WIDTH = 256, 16, 16
DATA_SEED = 0
INIT_SEED = 0
SETTINGS = {"T": (2, 4), "A": (1, 4), "B": (2, 6)}  # name -> (r, d_out)
TOL = 1e-8

# ----------------------------------------------------------------------------- configuration
if SMOKE:
    CFG = dict(
        p1_L_T=[1, 2, 3], p1_L_AB=[2],
        p2_s_lists=[[1.0, 1.3, 0.8, 1.1]], p3_tails=[0.0, 0.2],
        p4_L=[3], p4_steps=100, p4_every=25,
        p4b_runs=[("A", 2)], p4b_steps=100, p4b_every=50,
        p5_L=[2, 4], p5_seeds=[0], p5_max_steps=300, p5_ho_every=100, p5_full_every=200,
    )
else:
    CFG = dict(
        p1_L_T=[1, 2, 3, 4, 5, 6, 8], p1_L_AB=[2, 4, 8],
        p2_s_lists=[[1.0, 1.3, 0.8, 1.1], [1.0, 2.0, 1.0, 0.5]], p3_tails=[0.0, 0.05, 0.2],
        p4_L=[3, 5], p4_steps=2000, p4_every=25,
        p4b_runs=[(s, L) for s in ("T", "A", "B") for L in (2, 4, 8)], p4b_steps=1000, p4b_every=100,
        p5_L=[2, 4, 6, 8], p5_seeds=[0, 1], p5_max_steps=15000, p5_ho_every=250, p5_full_every=1000,
    )
GD_LR = 0.05
GAUSS_LR_CAP = 0.05
GAUSS_LR_FAC = 1.0  # gaussian lr = min(cap, FAC / lambda_max(H_init)) (probe: stable for all L; lr=0.05 diverges at L>=6)
CONV_LOSS = 1e-6
SPEC_STEPS_P5 = 3000  # the task's step budget; runs that have not converged by then continue to p5_max_steps (logged)

# palette (dataviz reference palette; categorical in fixed order, sequential blue ramp for L)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.5,
                     "legend.frameon": False, "figure.dpi": 120, "lines.linewidth": 1.5})


def seq_color(i, n):
    idx = int(round(i * (len(SEQ) - 1) / max(n - 1, 1)))
    return SEQ[idx]


T_START = time.time()


def log(*a):
    print(f"[{time.time() - T_START:8.1f}s]", *a, flush=True)


def to_py(o):
    """Recursively convert tensors / numpy / slices to JSON-serialisable python objects."""
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
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    return o


# ----------------------------------------------------------------------------- data / models
X_ALL = sl.make_whitened_inputs(N, D_IN, seed=DATA_SEED)
DATA = {}
for name, (r, d_out) in SETTINGS.items():
    Y, Wstar = sl.make_rank_r_targets(X_ALL, r, d_out, seed=DATA_SEED)
    sv = torch.linalg.svdvals(Wstar)
    assert Y.shape == (N, d_out), (name, Y.shape)
    assert int((sv > 1e-10).sum()) == r, (name, sv)
    DATA[name] = (Y, Wstar)
    log(f"data {name}: r={r} d_out={d_out} Y{tuple(Y.shape)} rank(Wstar)={int((sv > 1e-10).sum())} "
        f"sv={[round(float(x), 6) for x in sv]} ||X^T X/n - I||={float(torch.linalg.norm(X_ALL.T @ X_ALL / N - torch.eye(D_IN))):.2e}")


def make_spec(L, d_out):
    return sl.ModelSpec("linear", L, D_IN, WIDTH, d_out)


def balancedness(spec, params):
    Ws = [params[f"W{l}"] for l in range(1, spec.L + 1)]
    return [float(torch.linalg.norm(Ws[l + 1].T @ Ws[l + 1] - Ws[l] @ Ws[l].T)) for l in range(spec.L - 1)]


def layer_svals(spec, params, k):
    return [[float(x) for x in torch.linalg.svdvals(params[f"W{l}"])[:k]] for l in range(1, spec.L + 1)]


def expected_counts(spec, r, d_in, d_out):
    n_dom, n_bulk = r * r, r * (d_in + d_out - 2 * r)
    return n_dom, n_bulk, spec.n_params - n_dom - n_bulk


def hess_record(spec, params, X, Y, r, d_in, d_out, step=0, loss=None, full=True):
    """One Hessian evaluation: protocol dict + raw arrays."""
    t0 = time.time()
    if loss is None:
        loss = float(sl.loss_fn(spec, params, X, Y))
    H_o = sl.gauss_newton(spec, params, X, Y)
    eig_Ho = sl.eigvals(H_o)
    cs_Ho = sl.cluster_stats(eig_Ho, r, d_in, d_out, tol=TOL)
    coh = sl.layer_coherence(H_o, spec.layer_slices)
    k = max(r * r, d_out)
    energy_k = sl.top_eigvec_layer_energy(H_o, spec.layer_slices, k)
    fsm = sl.feature_second_moment_stats(spec, params, X)
    n_dom, n_bulk, n_zero = expected_counts(spec, r, d_in, d_out)
    rec = {
        "step": int(step), "loss": float(loss), "P": int(spec.n_params), "L": int(spec.L),
        "r": int(r), "d_in": int(d_in), "d_out": int(d_out),
        "expected_n_dominant": n_dom, "expected_n_bulk": n_bulk, "expected_n_zero": n_zero,
        "Ho_lambda_max": float(eig_Ho[0]), "Ho_norm": float(torch.linalg.norm(H_o)),
        "cluster_Ho": cs_Ho,
        "coherence": float(coh["coherence"]), "lambda_max_blocks": coh["lambda_max_blocks"],
        "top_eigvec_layer_energy": coh["top_eigvec_layer_energy"],
        "block_alignment": to_py(coh["block_alignment"]),
        "topk_layer_energy": to_py(energy_k), "topk_k": k,
        "feature_second_moment": to_py(fsm),
        "balancedness": balancedness(spec, params),
        "layer_svals": layer_svals(spec, params, max(r + 2, 4)),
        "residual_norm": float(torch.linalg.norm(spec.forward(params, X) - Y)),
        "full_H": bool(full),
    }
    arrays = {"eig_Ho": eig_Ho.numpy()}
    if full:
        H = sl.full_hessian(spec, params, X, Y)
        H_f = sl.functional_part(H, H_o)
        eig_H = sl.eigvals(H)
        Hf_n, Ho_n = float(torch.linalg.norm(H_f)), float(torch.linalg.norm(H_o))
        rec.update({
            "asymmetry": float(sl.asymmetry(H)), "H_norm": float(torch.linalg.norm(H)),
            "Hf_norm": Hf_n, "Hf_over_Ho": Hf_n / Ho_n if Ho_n > 0 else float("nan"),
            "H_lambda_max": float(eig_H[0]), "H_lambda_min": float(eig_H[-1]),
            "cluster_H": sl.cluster_stats(eig_H, r, d_in, d_out, tol=TOL),
            "coherence_H": float(sl.layer_coherence(H, spec.layer_slices)["coherence"]),
        })
        arrays["eig_H"] = eig_H.numpy()
    rec["wall_time"] = time.time() - t0
    return rec, arrays


def c1_check(rec, which="cluster_Ho"):
    """C1: r^2 dominant (all equal), r(d_in+d_out-2r) bulk (all equal), rest zero."""
    cs = rec[which]
    L = rec["L"]
    out = {
        "dominant_equal": abs(cs["dominant_max"] - cs["dominant_min"]) <= 1e-6 * max(abs(cs["dominant_max"]), 1e-300),
        "bulk_equal": abs(cs["bulk_max"] - cs["bulk_min"]) <= 1e-6 * max(abs(cs["bulk_max"]), 1e-300),
        "zero_count_ok": cs["zero_count"] == rec["expected_n_zero"],
        "split_counts_ok": (cs["split_n_dominant"] == rec["expected_n_dominant"]
                            and cs["split_n_bulk"] == rec["expected_n_bulk"]),
        "dominant_eq_L": abs(cs["dominant_mean"] - L) <= 1e-6 * L,
        "bulk_eq_1": abs(cs["bulk_mean"] - 1.0) <= 1e-6,
    }
    out["C1_pass"] = bool(out["dominant_equal"] and out["bulk_equal"] and out["zero_count_ok"] and out["split_counts_ok"])
    return out


def fit_exp(steps, values, floor):
    """Least-squares fit of log(values) = rate * step + b on values > floor."""
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    m = np.isfinite(values) & (values > floor)
    if m.sum() < 3:
        return {"rate": float("nan"), "r2": float("nan"), "n_points": int(m.sum()), "step_range": None}
    x, y = steps[m], np.log(values[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ coef
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"rate": float(coef[0]), "intercept": float(coef[1]), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n_points": int(m.sum()), "step_range": [float(x.min()), float(x.max())]}


def gd_with_checkpoints(spec, params, X, Y, r, d_in, d_out, lr, max_steps, ho_every, full_every,
                        stop_loss=None, tag=""):
    """GD via sl.gd_train in chunks of ho_every steps; Hessian record at every chunk boundary."""
    recs, eig_Ho_list, eig_H_list, eig_H_steps = [], [], [], []
    loss_curve, grad_curve = [], []
    step = 0
    full = True
    rec, arr = hess_record(spec, params, X, Y, r, d_in, d_out, step=0, full=full)
    recs.append(rec); eig_Ho_list.append(arr["eig_Ho"]); eig_H_list.append(arr["eig_H"]); eig_H_steps.append(0)
    log(f"  {tag} step 0 loss={rec['loss']:.3e} ratio={rec['cluster_Ho']['ratio_paper']:.4f} coh={rec['coherence']:.4f} "
        f"Hf/Ho={rec['Hf_over_Ho']:.3e} lam_max(H)={rec['H_lambda_max']:.3f} ({rec['wall_time']:.1f}s)")
    stopped_reason = "max_steps"
    while step < max_steps:
        chunk = min(ho_every, max_steps - step)
        h = sl.gd_train(spec, params, X, Y, lr=lr, steps=chunk, optimizer="gd")
        params = h["params"]
        # h["loss"] has chunk+1 entries (loss before each update, plus the final); drop the duplicate first
        loss_curve.extend(h["loss"] if step == 0 else h["loss"][1:])
        grad_curve.extend(h["grad_norm"] if step == 0 else h["grad_norm"][1:])
        step += chunk
        cur_loss = h["loss"][-1]
        if not np.isfinite(cur_loss):
            stopped_reason = "diverged"
            log(f"  {tag} DIVERGED at step {step}")
            break
        converged = stop_loss is not None and cur_loss < stop_loss
        full = (full_every is not None and step % full_every == 0) or step >= max_steps or converged
        rec, arr = hess_record(spec, params, X, Y, r, d_in, d_out, step=step, loss=cur_loss, full=full)
        recs.append(rec); eig_Ho_list.append(arr["eig_Ho"])
        if full:
            eig_H_list.append(arr["eig_H"]); eig_H_steps.append(step)
        if step % (ho_every * 4) == 0 or full:
            msg = f"  {tag} step {step} loss={cur_loss:.3e} ratio={rec['cluster_Ho']['ratio_paper']:.4f} coh={rec['coherence']:.4f}"
            if full:
                msg += f" Hf/Ho={rec['Hf_over_Ho']:.3e} Hf={rec['Hf_norm']:.3e} lam_max(H)={rec['H_lambda_max']:.3f}"
            msg += f" bal={['%.2e' % b for b in rec['balancedness']]} ({rec['wall_time']:.1f}s)"
            log(msg)
        if converged:
            stopped_reason = f"loss<{stop_loss:g}"
            break
    return {
        "params": params, "records": recs, "loss_curve": np.array(loss_curve), "grad_curve": np.array(grad_curve),
        "eig_Ho": np.array(eig_Ho_list), "eig_H": np.array(eig_H_list), "eig_H_steps": np.array(eig_H_steps),
        "final_step": step, "stopped_reason": stopped_reason, "lr": lr,
    }


SUMMARY = []


def add(run_name, part, rec, extra=None, arrays=None):
    d = {"run": run_name, "part": part, **rec}
    if extra:
        d.update(extra)
    SUMMARY.append(to_py(d))
    if arrays is not None:
        np.savez(f"{RAW}/{run_name}.npz", **{k: np.asarray(v) for k, v in arrays.items()})


def dump_summary():
    with open(f"{RES}/summary.json", "w") as f:
        json.dump(SUMMARY, f, indent=1)


# ============================================================================= PART 1
log("=" * 30, "PART 1: balanced_uniform init (s=1, tail 0), H_o and full H at init")
P1 = {}
first_printed = False
for name, Ls in [("T", CFG["p1_L_T"]), ("A", CFG["p1_L_AB"]), ("B", CFG["p1_L_AB"])]:
    r, d_out = SETTINGS[name]
    Y, _ = DATA[name]
    for L in Ls:
        spec = make_spec(L, d_out)
        params = spec.init_params(seed=INIT_SEED, init="balanced_uniform", r=r, s=1.0, tail_value=0.0)
        rec, arr = hess_record(spec, params, X_ALL, Y, r, D_IN, d_out, step=0, full=True)
        c1o, c1h = c1_check(rec, "cluster_Ho"), c1_check(rec, "cluster_H")
        cs = rec["cluster_Ho"]
        extra = {"setting": name, "init": "balanced_uniform", "s": 1.0, "tail_value": 0.0, "seed": INIT_SEED,
                 "C1_Ho": c1o, "C1_H": c1h,
                 "C2_ratio_eq_L": abs(cs["ratio_paper"] - L) <= 1e-6 * L,
                 "C2_coherence_eq_L": abs(rec["coherence"] - L) <= 1e-6 * L}
        run = f"p1_{name}_L{L}"
        add(run, "1", rec, extra, arr)
        P1[(name, L)] = SUMMARY[-1]
        if not first_printed:
            log("FULL STATS DICT (sanity check):\n" + json.dumps(SUMMARY[-1], indent=1, default=str))
            first_printed = True
            # sanity assertions on the first record
            assert rec["P"] == 256 * (L - 1) + d_out * 16, rec["P"]
            assert cs["n_eigs"] == rec["P"]
            assert cs["n_dominant_expected"] == r * r and cs["n_bulk_expected"] == r * (D_IN + d_out - 2 * r)
            assert rec["topk_k"] == max(r * r, d_out)
        log(f"{run}: P={rec['P']} dom={cs['dominant_mean']:.6f} (min {cs['dominant_min']:.6f}) bulk={cs['bulk_mean']:.6f} "
            f"(min {cs['bulk_min']:.6f}, max {cs['bulk_max']:.6f}) zeros={cs['zero_count']}/{rec['expected_n_zero']} "
            f"rank={cs['rank']} split=({cs['split_n_dominant']},{cs['split_n_bulk']}) ratio={cs['ratio_paper']:.6f} "
            f"ratio_gap={cs['ratio_gap']:.4g} coh={rec['coherence']:.6f} Hf/Ho={rec['Hf_over_Ho']:.4f} "
            f"asym={rec['asymmetry']:.2e} C1(Ho)={c1o['C1_pass']} C1(H)={c1h['C1_pass']} "
            f"H: lam_max={rec['H_lambda_max']:.4f} lam_min={rec['H_lambda_min']:.4f} neg={rec['cluster_H']['negative_count']} "
            f"ratio_H={rec['cluster_H']['ratio_paper']:.4f} coh_H={rec['coherence_H']:.4f} ({rec['wall_time']:.1f}s)")
dump_summary()

# ============================================================================= PART 2
log("=" * 30, "PART 2: non-uniform per-layer s_l at L=4 (setting T)")
P2 = []
r, d_out = SETTINGS["T"]
Y, _ = DATA["T"]
for s_list in CFG["p2_s_lists"]:
    L = len(s_list)
    spec = make_spec(L, d_out)
    params = spec.init_params(seed=INIT_SEED, init="balanced_uniform", r=r, s=s_list, tail_value=0.0)
    rec, arr = hess_record(spec, params, X_ALL, Y, r, D_IN, d_out, step=0, full=True)
    s2 = np.array(s_list) ** 2
    prod_except = [float(np.prod(np.delete(s2, l))) for l in range(L)]
    dom_cf = float(sum(prod_except))
    bulk_in, bulk_out = prod_except[0], prod_except[-1]
    n_in, n_out = r * (D_IN - r), r * (d_out - r)
    bulk_mean_cf = (n_in * bulk_in + n_out * bulk_out) / (n_in + n_out)
    closed = {
        "s": s_list, "dominant_cf": dom_cf, "bulk_in_cf": bulk_in, "bulk_out_cf": bulk_out,
        "ratio_cf_max": dom_cf / max(bulk_in, bulk_out),          # the closed form asked for
        "ratio_cf_bulkmean": dom_cf / bulk_mean_cf,                # what ratio_paper (dominant_mean/bulk_mean) measures
        "coherence_cf": dom_cf / max(prod_except),                 # lambda_max / max_l lambda_max(B_ll)
        "lambda_max_blocks_cf": prod_except,
    }
    cs = rec["cluster_Ho"]
    bulk_sorted = np.sort(arr["eig_Ho"][r * r: r * r + n_in + n_out])
    bulk_cf_sorted = np.sort(np.array([bulk_in] * n_in + [bulk_out] * n_out))
    closed["bulk_values_match_cf"] = bool(np.allclose(bulk_sorted, bulk_cf_sorted, rtol=1e-6, atol=1e-9))
    closed["dominant_match_cf"] = bool(np.allclose(arr["eig_Ho"][: r * r], dom_cf, rtol=1e-6))
    closed["ratio_measured_over_bulkmax"] = cs["dominant_mean"] / cs["bulk_max"]
    run = f"p2_T_L{L}_s" + "-".join(f"{x:g}" for x in s_list)
    add(run, "2", rec, {"setting": "T", "init": "balanced_uniform", "tail_value": 0.0, "seed": INIT_SEED, "closed_form": closed}, arr)
    P2.append(SUMMARY[-1])
    log(f"{run}: dom_meas={cs['dominant_mean']:.6f} dom_cf={dom_cf:.6f} | bulk_mean_meas={cs['bulk_mean']:.6f} cf={bulk_mean_cf:.6f} "
        f"bulk_max_meas={cs['bulk_max']:.6f} cf={max(bulk_in, bulk_out):.6f} | ratio_paper={cs['ratio_paper']:.6f} "
        f"ratio_cf_bulkmean={closed['ratio_cf_bulkmean']:.6f} | dom/bulk_max={closed['ratio_measured_over_bulkmax']:.6f} "
        f"ratio_cf_max={closed['ratio_cf_max']:.6f} | coh={rec['coherence']:.6f} coh_cf={closed['coherence_cf']:.6f} | L={L} "
        f"blocks={[round(b, 4) for b in rec['lambda_max_blocks']]} cf={[round(b, 4) for b in prod_except]} "
        f"zeros={cs['zero_count']}/{rec['expected_n_zero']} split=({cs['split_n_dominant']},{cs['split_n_bulk']}) "
        f"bulk_match={closed['bulk_values_match_cf']} dom_match={closed['dominant_match_cf']}")
dump_summary()

# ============================================================================= PART 3
log("=" * 30, "PART 3: tail_value in", CFG["p3_tails"], "at L=4 (setting T)")
P3 = []
for tail in CFG["p3_tails"]:
    L = 4
    spec = make_spec(L, d_out)
    params = spec.init_params(seed=INIT_SEED, init="balanced_uniform", r=r, s=1.0, tail_value=tail)
    rec, arr = hess_record(spec, params, X_ALL, Y, r, D_IN, d_out, step=0, full=True)
    cs = rec["cluster_Ho"]
    e = arr["eig_Ho"]
    extra = {"setting": "T", "init": "balanced_uniform", "s": 1.0, "tail_value": tail, "seed": INIT_SEED,
             "n_above_1e-3": int((e > 1e-3 * e[0]).sum()), "n_above_1e-6": int((e > 1e-6 * e[0]).sum()),
             "eig_r2": float(e[r * r - 1]), "eig_r2_plus1": float(e[r * r]),
             "eig_bulk_end": float(e[r * r + cs["n_bulk_expected"] - 1]), "eig_after_bulk": float(e[r * r + cs["n_bulk_expected"]]),
             "C1_Ho": c1_check(rec, "cluster_Ho")}
    run = f"p3_T_L4_tail{tail:g}"
    add(run, "3", rec, extra, arr)
    P3.append(SUMMARY[-1])
    log(f"{run}: ratio_paper={cs['ratio_paper']:.5f} ratio_gap={cs['ratio_gap']:.4g} ratio_split={cs['ratio_split']:.5f} "
        f"split=({cs['split_n_dominant']},{cs['split_n_bulk']}) gap_idx={cs['largest_log_gap_index']} "
        f"zeros={cs['zero_count']} rank={cs['rank']} n>1e-3={extra['n_above_1e-3']} n>1e-6={extra['n_above_1e-6']} "
        f"dom[min,max]=[{cs['dominant_min']:.5f},{cs['dominant_max']:.5f}] bulk[min,max]=[{cs['bulk_min']:.5f},{cs['bulk_max']:.5f}] "
        f"next={extra['eig_after_bulk']:.3e} coh={rec['coherence']:.5f} Hf/Ho={rec['Hf_over_Ho']:.3f}")
dump_summary()

# ============================================================================= PART 4
log("=" * 30, "PART 4: GD from balanced init, setting T, lr", GD_LR, "steps", CFG["p4_steps"], "H_o+H every", CFG["p4_every"])
P4 = {}
for L in CFG["p4_L"]:
    spec = make_spec(L, d_out)
    params = spec.init_params(seed=INIT_SEED, init="balanced_uniform", r=r, s=1.0, tail_value=0.0)
    out = gd_with_checkpoints(spec, params, X_ALL, Y, r, D_IN, d_out, lr=GD_LR, max_steps=CFG["p4_steps"],
                              ho_every=CFG["p4_every"], full_every=CFG["p4_every"], tag=f"p4 L={L}")
    recs = out["records"]
    steps = np.array([q["step"] for q in recs])
    Hf = np.array([q["Hf_norm"] for q in recs])
    Ho_n = np.array([q["Ho_norm"] for q in recs])
    ratio = np.array([q["cluster_Ho"]["ratio_paper"] for q in recs])
    coh = np.array([q["coherence"] for q in recs])
    bal = np.array([q["balancedness"] for q in recs])
    lossv = np.array([q["loss"] for q in recs])
    floor = 1e-8
    fit_all = fit_exp(steps, Hf, floor)
    m = Hf > floor
    n_ok = int(m.sum())
    late_idx = np.where(m)[0][n_ok // 2:] if n_ok >= 6 else np.where(m)[0]
    fit_late = fit_exp(steps[late_idx], Hf[late_idx], floor)
    early_idx = np.where(m)[0][: max(3, n_ok // 2)]
    fit_early = fit_exp(steps[early_idx], Hf[early_idx], floor)
    fit_loss = fit_exp(steps, lossv, 1e-16)
    fit_loss_late = fit_exp(steps[late_idx], lossv[late_idx], 1e-16)
    bulk_end = recs[-1]["cluster_Ho"]["bulk_mean"]
    dom_end = recs[-1]["cluster_Ho"]["dominant_mean"]
    theory = {"rate_Hf_bulk": math.log(1 - GD_LR * bulk_end) if GD_LR * bulk_end < 1 else float("nan"),
              "rate_Hf_dominant": math.log(abs(1 - GD_LR * dom_end)),
              "rate_loss_bulk": 2 * math.log(1 - GD_LR * bulk_end) if GD_LR * bulk_end < 1 else float("nan"),
              "bulk_end": bulk_end, "dominant_end": dom_end}
    summary4 = {
        "setting": "T", "init": "balanced_uniform", "seed": INIT_SEED, "lr": GD_LR, "steps": out["final_step"],
        "stopped_reason": out["stopped_reason"], "n_checkpoints": len(recs), "checkpoint_every": CFG["p4_every"],
        "C3_fit_logHf_all": fit_all, "C3_fit_logHf_early": fit_early, "C3_fit_logHf_late": fit_late,
        "C3_fit_logloss_all": fit_loss, "C3_fit_logloss_late": fit_loss_late, "C3_theory_rates": theory,
        "Hf_floor": floor, "Hf_norm_init": float(Hf[0]), "Hf_norm_final": float(Hf[-1]),
        "Hf_over_Ho_init": float(Hf[0] / Ho_n[0]), "Hf_over_Ho_final": float(Hf[-1] / Ho_n[-1]),
        "loss_init": float(lossv[0]), "loss_final": float(lossv[-1]),
        "ratio_paper_min": float(ratio.min()), "ratio_paper_max": float(ratio.max()), "ratio_paper_final": float(ratio[-1]),
        "coherence_min": float(coh.min()), "coherence_max": float(coh.max()), "coherence_final": float(coh[-1]),
        "balancedness_init": bal[0].tolist(), "balancedness_max_over_training": bal.max(0).tolist(),
        "balancedness_final": bal[-1].tolist(),
        "C5_ratio_stays_L": bool(np.all(np.abs(ratio - L) <= 1e-3 * L)),
        "C5_coherence_stays_L": bool(np.all(np.abs(coh - L) <= 1e-3 * L)),
        "ratio_H_final": recs[-1]["cluster_H"]["ratio_paper"], "coherence_H_final": recs[-1]["coherence_H"],
        "asymmetry_max": float(max(q["asymmetry"] for q in recs)),
        "trajectory": {"step": steps.tolist(), "loss": lossv.tolist(), "Hf_norm": Hf.tolist(), "Ho_norm": Ho_n.tolist(),
                       "Hf_over_Ho": (Hf / Ho_n).tolist(), "ratio_paper": ratio.tolist(), "coherence": coh.tolist(),
                       "ratio_H": [q["cluster_H"]["ratio_paper"] for q in recs],
                       "zero_count_Ho": [q["cluster_Ho"]["zero_count"] for q in recs],
                       "split_n_dominant_Ho": [q["cluster_Ho"]["split_n_dominant"] for q in recs],
                       "balancedness": bal.tolist(), "H_lambda_max": [q["H_lambda_max"] for q in recs],
                       "H_lambda_min": [q["H_lambda_min"] for q in recs],
                       "layer_svals": [q["layer_svals"] for q in recs]},
        "records": recs,
    }
    run = f"p4_T_L{L}_gd"
    add(run, "4", {**recs[-1], "step": out["final_step"]}, summary4,
        {"loss_curve": out["loss_curve"], "grad_curve": out["grad_curve"], "eig_Ho": out["eig_Ho"], "eig_H": out["eig_H"],
         "eig_H_steps": out["eig_H_steps"], "steps": steps, "Hf_norm": Hf, "ratio_paper": ratio, "coherence": coh,
         "balancedness": bal})
    P4[L] = summary4
    log(f"{run}: loss {lossv[0]:.3e} -> {lossv[-1]:.3e}; ||H_f|| {Hf[0]:.3e} -> {Hf[-1]:.3e}; "
        f"fit all: rate={fit_all['rate']:.5f} R2={fit_all['r2']:.4f} (n={fit_all['n_points']}); early: rate={fit_early['rate']:.5f} R2={fit_early['r2']:.4f}; "
        f"late: rate={fit_late['rate']:.5f} R2={fit_late['r2']:.4f}; theory bulk rate={theory['rate_Hf_bulk']:.5f}; "
        f"loss fit late rate={fit_loss_late['rate']:.5f} (theory {theory['rate_loss_bulk']:.5f}); "
        f"ratio in [{ratio.min():.5f},{ratio.max():.5f}] coh in [{coh.min():.5f},{coh.max():.5f}] "
        f"bal max={np.max(bal):.2e} C5={summary4['C5_ratio_stays_L']}/{summary4['C5_coherence_stays_L']}")
dump_summary()

# ============================================================================= PART 4b (extra)
log("=" * 30, "PART 4b (extra): GD reference numbers for T/A/B at L in {2,4,8}, steps", CFG["p4b_steps"])
P4B = {}
for name, L in CFG["p4b_runs"]:
    r_, d_out_ = SETTINGS[name]
    Y_, _ = DATA[name]
    spec = make_spec(L, d_out_)
    params = spec.init_params(seed=INIT_SEED, init="balanced_uniform", r=r_, s=1.0, tail_value=0.0)
    out = gd_with_checkpoints(spec, params, X_ALL, Y_, r_, D_IN, d_out_, lr=GD_LR, max_steps=CFG["p4b_steps"],
                              ho_every=CFG["p4b_every"], full_every=CFG["p4b_every"], tag=f"p4b {name} L={L}")
    recs = out["records"]
    steps = np.array([q["step"] for q in recs])
    Hf = np.array([q["Hf_norm"] for q in recs]); Ho_n = np.array([q["Ho_norm"] for q in recs])
    ratio = np.array([q["cluster_Ho"]["ratio_paper"] for q in recs]); coh = np.array([q["coherence"] for q in recs])
    bal = np.array([q["balancedness"] for q in recs]); lossv = np.array([q["loss"] for q in recs])
    fit_all = fit_exp(steps, Hf, 1e-8)
    extra = {"setting": name, "init": "balanced_uniform", "seed": INIT_SEED, "lr": GD_LR, "steps": out["final_step"],
             "stopped_reason": out["stopped_reason"], "checkpoint_every": CFG["p4b_every"],
             "C3_fit_logHf_all": fit_all, "loss_init": float(lossv[0]), "loss_final": float(lossv[-1]),
             "Hf_over_Ho_init": float(Hf[0] / Ho_n[0]), "Hf_over_Ho_final": float(Hf[-1] / Ho_n[-1]),
             "ratio_paper_min": float(ratio.min()), "ratio_paper_max": float(ratio.max()), "ratio_paper_final": float(ratio[-1]),
             "coherence_min": float(coh.min()), "coherence_max": float(coh.max()), "coherence_final": float(coh[-1]),
             "balancedness_max_over_training": bal.max(0).tolist(), "balancedness_final": bal[-1].tolist(),
             "C5_ratio_stays_L": bool(np.all(np.abs(ratio - L) <= 1e-3 * L)),
             "C5_coherence_stays_L": bool(np.all(np.abs(coh - L) <= 1e-3 * L)),
             "ratio_H_final": recs[-1]["cluster_H"]["ratio_paper"],
             "trajectory": {"step": steps.tolist(), "loss": lossv.tolist(), "Hf_norm": Hf.tolist(), "Hf_over_Ho": (Hf / Ho_n).tolist(),
                            "ratio_paper": ratio.tolist(), "coherence": coh.tolist(), "balancedness": bal.tolist(),
                            "ratio_H": [q["cluster_H"]["ratio_paper"] for q in recs]},
             "record_init": recs[0]}
    run = f"p4b_{name}_L{L}_gd"
    add(run, "4b", {**recs[-1], "step": out["final_step"]}, extra,
        {"loss_curve": out["loss_curve"], "grad_curve": out["grad_curve"], "eig_Ho": out["eig_Ho"], "eig_H": out["eig_H"],
         "eig_H_steps": out["eig_H_steps"], "steps": steps, "Hf_norm": Hf, "ratio_paper": ratio, "coherence": coh,
         "balancedness": bal})
    P4B[(name, L)] = extra
    log(f"{run}: loss {lossv[0]:.3e} -> {lossv[-1]:.3e}; Hf/Ho {Hf[0] / Ho_n[0]:.3f} -> {Hf[-1] / Ho_n[-1]:.3e}; "
        f"ratio in [{ratio.min():.5f},{ratio.max():.5f}] coh in [{coh.min():.5f},{coh.max():.5f}] bal max={np.max(bal):.2e} "
        f"fit rate={fit_all['rate']:.5f} R2={fit_all['r2']:.4f}")
dump_summary()

# ============================================================================= PART 5
log("=" * 30, "PART 5: gaussian init (gain 1), L in", CFG["p5_L"], "seeds", CFG["p5_seeds"], "GD to loss<1e-6 or", CFG["p5_max_steps"])
P5 = {}
for L in CFG["p5_L"]:
    for seed in CFG["p5_seeds"]:
        spec = make_spec(L, d_out)
        params = spec.init_params(seed=seed, init="gaussian", gain=1.0)
        lam0 = float(sl.eigvals(sl.full_hessian(spec, params, X_ALL, Y))[0])
        lr = min(GAUSS_LR_CAP, GAUSS_LR_FAC / lam0)
        log(f"p5 L={L} seed={seed}: lambda_max(H_init)={lam0:.3f} -> lr={lr:.5f}")
        out = gd_with_checkpoints(spec, params, X_ALL, Y, r, D_IN, d_out, lr=lr, max_steps=CFG["p5_max_steps"],
                                  ho_every=CFG["p5_ho_every"], full_every=CFG["p5_full_every"], stop_loss=CONV_LOSS,
                                  tag=f"p5 L={L} s={seed}")
        if out["stopped_reason"] == "diverged":  # retry once with a 4x smaller lr (logged)
            log(f"p5 L={L} seed={seed}: diverged at lr={lr:.5f}; retrying with lr/4")
            params = spec.init_params(seed=seed, init="gaussian", gain=1.0)
            lr = lr / 4
            out = gd_with_checkpoints(spec, params, X_ALL, Y, r, D_IN, d_out, lr=lr, max_steps=CFG["p5_max_steps"],
                                      ho_every=CFG["p5_ho_every"], full_every=CFG["p5_full_every"], stop_loss=CONV_LOSS,
                                      tag=f"p5 L={L} s={seed} retry")
        recs = out["records"]
        r0, r1 = recs[0], recs[-1]

        def pick(q):
            cs = q["cluster_Ho"]
            return {"ratio_paper": cs["ratio_paper"], "ratio_gap": cs["ratio_gap"], "ratio_split": cs["ratio_split"],
                    "split_n_dominant": cs["split_n_dominant"], "split_n_bulk": cs["split_n_bulk"],
                    "largest_log_gap_index": cs["largest_log_gap_index"], "coherence": q["coherence"],
                    "rank": cs["rank"], "zero_count": cs["zero_count"], "dominant_mean": cs["dominant_mean"],
                    "bulk_mean": cs["bulk_mean"], "lambda_max": q["Ho_lambda_max"],
                    "top_eigvec_layer_energy": q["top_eigvec_layer_energy"], "balancedness": q["balancedness"],
                    "Hf_over_Ho": q.get("Hf_over_Ho"), "ratio_H": q["cluster_H"]["ratio_paper"] if "cluster_H" in q else None,
                    "H_lambda_min": q.get("H_lambda_min"), "loss": q["loss"], "step": q["step"]}

        steps = np.array([q["step"] for q in recs])
        ratio = np.array([q["cluster_Ho"]["ratio_paper"] for q in recs]); coh = np.array([q["coherence"] for q in recs])
        r3000 = recs[min(int(np.searchsorted(steps, min(SPEC_STEPS_P5, out["final_step"]))), len(recs) - 1)]
        extra = {"setting": "T", "init": "gaussian", "gain": 1.0, "seed": seed, "lr": lr, "lambda_max_H_init": lam0,
                 "steps": out["final_step"], "stopped_reason": out["stopped_reason"], "converged": out["stopped_reason"].startswith("loss<"),
                 "converged_within_3000": bool(out["stopped_reason"].startswith("loss<") and out["final_step"] <= SPEC_STEPS_P5),
                 "loss_init": r0["loss"], "loss_final": r1["loss"], "loss_at_3000": r3000["loss"],
                 "at_init": pick(r0), "at_end": pick(r1), "at_3000": pick(r3000),
                 "trajectory": {"step": steps.tolist(), "loss": [q["loss"] for q in recs], "ratio_paper": ratio.tolist(),
                                "coherence": coh.tolist(), "split_n_dominant": [q["cluster_Ho"]["split_n_dominant"] for q in recs],
                                "rank": [q["cluster_Ho"]["rank"] for q in recs],
                                "balancedness": [q["balancedness"] for q in recs]}}
        run = f"p5_T_L{L}_gauss_seed{seed}"
        add(run, "5", {**r1}, extra,
            {"loss_curve": out["loss_curve"], "grad_curve": out["grad_curve"], "eig_Ho": out["eig_Ho"], "eig_H": out["eig_H"],
             "eig_H_steps": out["eig_H_steps"], "steps": steps, "ratio_paper": ratio, "coherence": coh})
        P5[(L, seed)] = extra
        a, b = extra["at_init"], extra["at_end"]
        log(f"{run}: {out['stopped_reason']} at step {out['final_step']} loss {r0['loss']:.3e} -> {r1['loss']:.3e} (at 3000: {r3000['loss']:.3e}) lr={lr:.4g} | "
            f"init: ratio={a['ratio_paper']:.3f} gap={a['ratio_gap']:.3f} split={a['split_n_dominant']} ratio_split={a['ratio_split']:.3f} coh={a['coherence']:.3f} rank={a['rank']} | "
            f"end: ratio={b['ratio_paper']:.3f} gap={b['ratio_gap']:.3f} split={b['split_n_dominant']} ratio_split={b['ratio_split']:.3f} coh={b['coherence']:.3f} rank={b['rank']} "
            f"bal_end={np.mean(b['balancedness']):.3f} Hf/Ho_end={b['Hf_over_Ho']:.2e} topvec_energy_end={[round(x, 2) for x in b['top_eigvec_layer_energy']]}")
dump_summary()

# ============================================================================= FIGURES
log("=" * 30, "FIGURES")


def fig_spectra():
    rows = [("T", CFG["p1_L_T"]), ("A", CFG["p1_L_AB"]), ("B", CFG["p1_L_AB"])]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.2 * len(rows)), squeeze=False)
    for i, (name, Ls) in enumerate(rows):
        r_, d_out_ = SETTINGS[name]
        n_dom, n_bulk = r_ * r_, r_ * (D_IN + d_out_ - 2 * r_)
        for j, key in enumerate(["eig_Ho", "eig_H"]):
            ax = axes[i, j]
            for li, L in enumerate(Ls):
                d = np.load(f"{RAW}/p1_{name}_L{L}.npz")[key]
                nmax = min(len(d), 100)
                idx = np.arange(1, nmax + 1)
                thr = 1e-12 * np.abs(d).max()
                pos = d[:nmax] > thr
                col = seq_color(li, len(Ls))
                ax.plot(idx[pos], d[:nmax][pos], "-", color=col, lw=1.4, label=f"L={L}")
                neg = d[:nmax] < -thr
                if neg.any():
                    ax.plot(idx[neg], -d[:nmax][neg], ":", color=col, lw=1.2)
            ax.axvline(n_dom + 0.5, color=MUTED, lw=0.8, ls="--")
            ax.axvline(n_dom + n_bulk + 0.5, color=MUTED, lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 30)
            ax.set_xlabel("eigenvalue index (descending)")
            ax.set_ylabel("eigenvalue")
            lab = "H_o = J^T J / n" if key == "eig_Ho" else "sym(H), dotted = |negative|"
            ax.set_title(f"setting {name} (r={r_}, d_out={d_out_}): {lab}\ndashed: index r^2={n_dom} and r^2 + r(d_in+d_out-2r)={n_dom + n_bulk}", fontsize=8)
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("e1: balanced-uniform init (s=1, tail 0), spectra at init; zero eigenvalues not shown (log axis)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{EXP}_spectra.png", dpi=120)
    plt.close(fig)


def fig_ratio_coherence():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    Lmax = max(CFG["p1_L_T"]) + 0.5
    for ax, key, ttl in [(axes[0], "ratio", "dominant/bulk ratio of H_o (ratio_paper) vs depth"),
                         (axes[1], "coh", "layer coherence lambda_max(H_o)/max_l lambda_max(B_ll) vs depth")]:
        ax.plot([0.5, Lmax], [0.5, Lmax], color=MUTED, lw=1, ls="--", label="y = L (paper)")
        for ci, (name, Ls) in enumerate([("T", CFG["p1_L_T"]), ("A", CFG["p1_L_AB"]), ("B", CFG["p1_L_AB"])]):
            v = [P1[(name, L)]["cluster_Ho"]["ratio_paper"] if key == "ratio" else P1[(name, L)]["coherence"] for L in Ls]
            r_, d_out_ = SETTINGS[name]
            ax.plot(Ls, v, "o-", color=CAT[ci], ms=6, lw=1.2, mfc="none", label=f"balanced s=1, {name} (r={r_}, d_out={d_out_})")
        # non-uniform s and tails (L=4)
        for k2, s4 in enumerate(P2):
            v = s4["cluster_Ho"]["ratio_paper"] if key == "ratio" else s4["coherence"]
            ax.plot([4], [v], marker="s", color=CAT[3 + k2], ms=7, ls="none", label=f"non-uniform s={s4['closed_form']['s']}")
            cf = s4["closed_form"]["ratio_cf_bulkmean"] if key == "ratio" else s4["closed_form"]["coherence_cf"]
            ax.plot([4], [cf], marker="x", color=CAT[3 + k2], ms=9, ls="none", label=f"  closed form ({cf:.3f})")
        for k3, t3 in enumerate(P3):
            if t3["tail_value"] == 0:
                continue
            v = t3["cluster_Ho"]["ratio_paper"] if key == "ratio" else t3["coherence"]
            ax.plot([4], [v], marker="^", color=CAT[5 + k3], ms=7, ls="none", label=f"tail_value={t3['tail_value']:g}")
        # Gaussian init at init / end
        for L in CFG["p5_L"]:
            vi = [P5[(L, s)]["at_init"]["ratio_paper" if key == "ratio" else "coherence"] for s in CFG["p5_seeds"]]
            ve = [P5[(L, s)]["at_end"]["ratio_paper" if key == "ratio" else "coherence"] for s in CFG["p5_seeds"]]
            ax.plot([L] * len(vi), vi, marker="d", color=CAT[7], ms=5, ls="none", mfc="none", label="gaussian init, at init" if L == CFG["p5_L"][0] else None)
            ax.plot([L] * len(ve), ve, marker="d", color=CAT[7], ms=5, ls="none", label="gaussian init, after GD" if L == CFG["p5_L"][0] else None)
        ax.set_xlabel("depth L (number of weight matrices)")
        ax.set_ylabel("ratio_paper" if key == "ratio" else "coherence")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks(sorted(set(CFG["p1_L_T"]) | set(CFG["p5_L"])))
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("e1: C2 (ratio == L) and the coherence bound (coherence <= L) in deep linear nets", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{EXP}_ratio_coherence.png", dpi=120)
    plt.close(fig)


def fig_hf_decay():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for li, L in enumerate(CFG["p4_L"]):
        s4 = P4[L]
        tr = s4["trajectory"]
        st, Hf = np.array(tr["step"]), np.array(tr["Hf_norm"])
        col = CAT[li]
        ax = axes[0, 0]
        ax.plot(st, Hf, "o-", color=col, ms=3, lw=1.2, label=f"L={L}: ||H_f||_F")
        f = s4["C3_fit_logHf_all"]
        if np.isfinite(f["rate"]):
            xs = np.array(f["step_range"])
            ax.plot(xs, np.exp(f["intercept"] + f["rate"] * xs), "--", color=col, lw=1,
                    label=f"  fit all: rate={f['rate']:.4f}/step, R2={f['r2']:.3f}")
        fl = s4["C3_fit_logHf_late"]
        if np.isfinite(fl["rate"]):
            xs = np.array(fl["step_range"])
            ax.plot(xs, np.exp(fl["intercept"] + fl["rate"] * xs), ":", color=col, lw=1.5,
                    label=f"  fit late: rate={fl['rate']:.4f} (theory ln(1-lr*bulk)={s4['C3_theory_rates']['rate_Hf_bulk']:.4f})")
        ax.axhline(s4["Hf_floor"], color=MUTED, lw=0.7, ls="--")
        ax = axes[0, 1]
        ax.plot(st, tr["loss"], "-", color=col, lw=1.4, label=f"L={L}: loss")
        ax.plot(st, np.array(tr["Hf_over_Ho"]), "--", color=col, lw=1.2, label=f"L={L}: ||H_f||/||H_o||")
        ax = axes[1, 0]
        ax.plot(st, tr["ratio_paper"], "-", color=col, lw=1.6, label=f"L={L}: ratio_paper(H_o)")
        ax.plot(st, tr["coherence"], "--", color=col, lw=1.2, label=f"L={L}: coherence(H_o)")
        ax.plot(st, tr["ratio_H"], ":", color=col, lw=1.2, label=f"L={L}: ratio_paper(sym H)")
        ax.axhline(L, color=MUTED, lw=0.7, ls="--")
        ax = axes[1, 1]
        bal = np.array(tr["balancedness"])
        for l in range(bal.shape[1]):
            ax.plot(st, np.maximum(bal[:, l], 1e-18), "-", color=col, lw=1.0, alpha=0.5 + 0.5 * l / max(bal.shape[1] - 1, 1),
                    label=f"L={L}: layer pair ({l + 1},{l + 2})")
    axes[0, 0].set_yscale("log"); axes[0, 0].set_title("C3: ||H_f||_F = ||H - H_o||_F during GD (dashed line: fit floor)", fontsize=9)
    axes[0, 0].set_xlabel("GD step"); axes[0, 0].set_ylabel("||H_f||_F"); axes[0, 0].legend(fontsize=6.5)
    axes[0, 1].set_yscale("log"); axes[0, 1].set_title("loss and ||H_f||_F / ||H_o||_F", fontsize=9)
    axes[0, 1].set_xlabel("GD step"); axes[0, 1].legend(fontsize=7)
    axes[1, 0].set_title("C5: dominant/bulk ratio and coherence during GD (dashed = L)", fontsize=9)
    axes[1, 0].set_xlabel("GD step"); axes[1, 0].legend(fontsize=6.5)
    axes[1, 1].set_yscale("log"); axes[1, 1].set_title("balancedness ||W_{l+1}^T W_{l+1} - W_l W_l^T||_F per layer pair", fontsize=9)
    axes[1, 1].set_xlabel("GD step"); axes[1, 1].legend(fontsize=6.5)
    fig.suptitle(f"e1: GD (lr={GD_LR}) from balanced-uniform init, setting T (r=2, d_out=4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{EXP}_Hf_decay.png", dpi=120)
    plt.close(fig)


def fig_gaussian():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    Ls = CFG["p5_L"]
    for ax, key, ttl in [(axes[0], "ratio_paper", "ratio_paper = mean(top r^2) / mean(next r(d_in+d_out-2r))"),
                         (axes[1], "coherence", "coherence lambda_max(H_o)/max_l lambda_max(B_ll)"),
                         (axes[2], "split_n_dominant", "log-gap dominant count (r^2 = 4 expected)")]:
        if key != "split_n_dominant":
            ax.plot([1.5, max(Ls) + 0.5], [1.5, max(Ls) + 0.5], color=MUTED, lw=1, ls="--", label="y = L (paper, balanced)")
        else:
            ax.axhline(r * r, color=MUTED, lw=1, ls="--", label="r^2")
        for si, seed in enumerate(CFG["p5_seeds"]):
            vi = [P5[(L, seed)]["at_init"][key] for L in Ls]
            v3 = [P5[(L, seed)]["at_3000"][key] for L in Ls]
            ve = [P5[(L, seed)]["at_end"][key] for L in Ls]
            lw = 1.4 if si == 0 else 0.8
            ax.plot(Ls, vi, "o--", color=CAT[0], ms=5, lw=lw, mfc="none", label="at init (seeds: thick/thin)" if si == 0 else None)
            ax.plot(Ls, v3, "^:", color=CAT[2], ms=5, lw=lw, mfc="none", label="after 3000 GD steps" if si == 0 else None)
            ax.plot(Ls, ve, "o-", color=CAT[1], ms=5, lw=lw, label="after GD to loss<1e-6 (or 15000 steps)" if si == 0 else None)
        if key == "ratio_paper":
            vb = [P1[("T", L)]["cluster_Ho"]["ratio_paper"] for L in Ls if ("T", L) in P1]
            ax.plot([L for L in Ls if ("T", L) in P1], vb, "s", color=CAT[6], ms=7, mfc="none", label="balanced init (part 1)")
        ax.set_xlabel("depth L"); ax.set_title(ttl, fontsize=8.5); ax.set_xticks(Ls); ax.legend(fontsize=6.5)
    fig.suptitle("e1: gaussian init (gain 1), setting T (r=2, d_out=4): does the r^2 / ratio ~ L structure appear?", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{EXP}_gaussian_ratio.png", dpi=120)
    plt.close(fig)


def fig_perturbations():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    for k2, s4 in enumerate(P2):
        d = np.load(f"{RAW}/{s4['run']}.npz")["eig_Ho"][:60]
        idx = np.arange(1, len(d) + 1); pos = d > 1e-12 * d[0]
        ax.plot(idx[pos], d[pos], "o-", color=CAT[k2], ms=3, lw=1, label=f"s={s4['closed_form']['s']}")
    d = np.load(f"{RAW}/p1_T_L4.npz")["eig_Ho"][:60] if ("T", 4) in P1 else None
    if d is not None:
        idx = np.arange(1, len(d) + 1); pos = d > 1e-12 * d[0]
        ax.plot(idx[pos], d[pos], "-", color=MUTED, lw=1, label="uniform s=1")
    ax.set_yscale("log"); ax.set_title("part 2: non-uniform s_l, H_o spectrum\n(L=4, setting T)", fontsize=9)
    ax.set_xlabel("index"); ax.set_ylabel("eigenvalue"); ax.legend(fontsize=7)
    ax = axes[1]
    for k3, t3 in enumerate(P3):
        d = np.load(f"{RAW}/{t3['run']}.npz")["eig_Ho"]
        idx = np.arange(1, len(d) + 1); pos = d > 1e-12 * d[0]
        ax.plot(idx[pos], d[pos], "-", color=CAT[k3], lw=1.2, label=f"tail_value={t3['tail_value']:g} (split: {t3['cluster_Ho']['split_n_dominant']} dom, {t3['cluster_Ho']['split_n_bulk']} bulk)")
    ax.set_yscale("log"); ax.set_xscale("log"); ax.set_title("part 3: tail_value, H_o spectrum\n(L=4, setting T)", fontsize=9)
    ax.set_xlabel("index (log)"); ax.set_ylabel("eigenvalue"); ax.legend(fontsize=7)
    ax = axes[2]
    for li, L in enumerate(CFG["p5_L"]):
        for seed in CFG["p5_seeds"]:
            z = np.load(f"{RAW}/p5_T_L{L}_gauss_seed{seed}.npz")
            e0, e1 = z["eig_Ho"][0][:80], z["eig_Ho"][-1][:80]
            idx = np.arange(1, 81)
            ax.plot(idx[e0 > 1e-14], e0[e0 > 1e-14], "--", color=seq_color(li, len(CFG["p5_L"])), lw=0.9, label=f"L={L} init" if seed == CFG["p5_seeds"][0] else None)
            ax.plot(idx[e1 > 1e-14], e1[e1 > 1e-14], "-", color=seq_color(li, len(CFG["p5_L"])), lw=1.2, label=f"L={L} after GD" if seed == CFG["p5_seeds"][0] else None)
    ax.axvline(r * r + 0.5, color=MUTED, lw=0.8, ls="--"); ax.axvline(r * r + r * (D_IN + d_out - 2 * r) + 0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_title("part 5: gaussian init, H_o spectrum\nat init (dashed) and after GD (solid)", fontsize=9)
    ax.set_xlabel("index"); ax.set_ylabel("eigenvalue"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{EXP}_perturbations.png", dpi=120)
    plt.close(fig)


for fn in (fig_spectra, fig_ratio_coherence, fig_hf_decay, fig_gaussian, fig_perturbations):
    try:
        fn()
        log(f"figure {fn.__name__} done")
    except Exception as ex:  # a figure bug must not lose the results
        import traceback
        log(f"FIGURE {fn.__name__} FAILED: {ex!r}\n{traceback.format_exc()}")

dump_summary()
log(f"DONE. summary: {RES}/summary.json ({len(SUMMARY)} runs), raw: {RAW}, figures: {FIG}. total wall {time.time() - T_START:.1f}s")
