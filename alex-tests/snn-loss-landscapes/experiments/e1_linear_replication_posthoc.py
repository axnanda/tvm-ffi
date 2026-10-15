import snnland as sl; sl.set_threads(1)  # noqa: E702
"""Post-hoc analysis for e1_linear_replication (run after the main script).

(a) Part 4 (GD from balanced init): under EXACT balancedness with shared SVD bases the H_o
    eigenvalues are a closed-form function of the end-to-end singular values sigma_a
    (lambda_{ab} = sum_l sigma_a^{c (L-l)/(L-1)} sigma_b^{c (l-1)/(L-1)}, c = 2(L-1)/L, for the
    r^2 dominant directions; bulk = sigma_a^c).  We re-run the deterministic GD trajectory,
    compute sigma_a at every checkpoint and compare the predicted dominant/bulk eigenvalues and
    ratio_paper with the measured ones (stored in raw/p4_T_L{L}_gd.npz).  This tests whether the
    transient deviation of ratio_paper from L is explained by sigma_1 != sigma_2 (unequal
    end-to-end singular values during the rotation towards the target subspace) rather than by
    loss of balancedness.
(b) Part 3 (tail_value t): closed form dominant = L, bulk = sum_{k=0}^{L-1} t^{2k},
    'zero' cluster -> (d_in-r)(d_out-r) eigenvalues at L t^{2(L-1)}; compared with the data.
Writes results/e1_linear_replication/posthoc.json and figures/e1_linear_replication_ratio_excursion.png.
Usage: E1_SMOKE=1 for the smoke results.
"""
import json
import os

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
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#898781"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.linewidth": 0.5, "grid.color": "#e1e0d9",
                     "legend.frameon": False, "figure.dpi": 120})

summary = json.load(open(f"{RES}/summary.json"))
by_run = {d["run"]: d for d in summary}
N, D_IN = 256, 16
X = sl.make_whitened_inputs(N, D_IN, seed=0)
out = {}


def predicted_eigs(sig, L, r, d_in, d_out):
    """Closed-form H_o eigenvalues under exact balancedness with shared SVD bases."""
    c = 2.0 * (L - 1) / L
    dom = []
    for a in range(r):
        for b in range(r):
            dom.append(sum(sig[a] ** (c * (L - l) / (L - 1)) * sig[b] ** (c * (l - 1) / (L - 1)) for l in range(1, L + 1)))
    bulk = []
    for a in range(r):
        bulk += [sig[a] ** c] * (d_in - r)   # layer-1 input-tail directions
        bulk += [sig[a] ** c] * (d_out - r)  # layer-L output-tail directions
    return np.sort(np.array(dom))[::-1], np.sort(np.array(bulk))[::-1]


# ------------------------------------------------------------------ (a) part 4 ratio excursion
p4 = {}
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for li, run in enumerate([k for k in by_run if k.startswith("p4_T_L") and k.endswith("_gd")]):
    d = by_run[run]
    L, r, d_in, d_out, lr = d["L"], d["r"], d["d_in"], d["d_out"], d["lr"]
    z = np.load(f"{RAW}/{run}.npz")
    steps = z["steps"]
    Y, _ = sl.make_rank_r_targets(X, r, d_out, seed=0)
    spec = sl.ModelSpec("linear", L, d_in, 16, d_out)
    params = spec.init_params(seed=d["seed"], init="balanced_uniform", r=r, s=1.0, tail_value=0.0)
    rows = []
    step = 0
    for i, st in enumerate(steps):
        if st > step:
            h = sl.gd_train(spec, params, X, Y, lr=lr, steps=int(st - step), optimizer="gd")
            params = h["params"]
            step = int(st)
        with torch.no_grad():
            E = torch.eye(d_in, dtype=torch.float64)
            for l in range(1, L + 1):
                E = params[f"W{l}"] @ E
            sig = torch.linalg.svdvals(E).numpy()
        pred_dom, pred_bulk = predicted_eigs(sig[:r], L, r, d_in, d_out)
        n_dom, n_bulk = r * r, r * (d_in + d_out - 2 * r)
        # the clusters can overlap during the transient (sigma_1 != sigma_2), so compare the GLOBALLY sorted lists,
        # exactly as cluster_stats does for the measured spectrum
        pred_all = np.sort(np.concatenate([pred_dom, pred_bulk]))[::-1]
        meas = z["eig_Ho"][i]
        meas_all = meas[: n_dom + n_bulk]
        pd_, pb_ = pred_all[:n_dom], pred_all[n_dom:n_dom + n_bulk]
        md_, mb_ = meas_all[:n_dom], meas_all[n_dom:n_dom + n_bulk]
        rows.append({
            "step": int(st), "sigma": sig[: r + 1].tolist(), "loss": float(z["loss_curve"][int(st)]),
            "ratio_measured": float(md_.mean() / mb_.mean()),
            "ratio_predicted_balanced": float(pd_.mean() / pb_.mean()),
            "top_r2_measured": md_.tolist(), "top_r2_predicted": pd_.tolist(),
            "in_subspace_eigs_predicted(unsorted, a,b)": pred_dom.tolist(),
            "bulk_measured_minmax": [float(mb_.min()), float(mb_.max())],
            "bulk_predicted_minmax": [float(pb_.min()), float(pb_.max())],
            "max_rel_err_dominant": float(np.max(np.abs(md_ - pd_) / pd_)),
            "max_rel_err_bulk": float(np.max(np.abs(mb_ - pb_) / pb_)),
            "max_rel_err_top36": float(np.max(np.abs(meas_all - pred_all) / pred_all)),
            "balancedness": z["balancedness"][i].tolist(),
        })
    p4[run] = rows
    st = np.array([q["step"] for q in rows])
    ax = axes[0]
    ax.plot(st, [q["ratio_measured"] for q in rows], "-", color=CAT[li], lw=1.6, label=f"L={L}: measured ratio_paper(H_o)")
    ax.plot(st, [q["ratio_predicted_balanced"] for q in rows], "--", color=CAT[li], lw=1.2, label=f"L={L}: predicted from sigma_a (exact balancedness)")
    ax.axhline(L, color=MUTED, lw=0.7, ls="--")
    ax = axes[1]
    sg = np.array([q["sigma"] for q in rows])
    for a in range(r):
        ax.plot(st, sg[:, a], "-" if a == 0 else "--", color=CAT[li], lw=1.3, label=f"L={L}: sigma_{a + 1}")
    ax = axes[2]
    ax.plot(st, [q["max_rel_err_dominant"] for q in rows], "-", color=CAT[li], lw=1.3, label=f"L={L}: dominant cluster")
    ax.plot(st, [q["max_rel_err_bulk"] for q in rows], "--", color=CAT[li], lw=1.3, label=f"L={L}: bulk cluster")
    ax.plot(st, np.max(np.array([q["balancedness"] for q in rows]), axis=1), ":", color=CAT[li], lw=1.3, label=f"L={L}: max balancedness defect")
    print(f"{run}: ratio measured range [{min(q['ratio_measured'] for q in rows):.4f}, {max(q['ratio_measured'] for q in rows):.4f}], "
          f"predicted range [{min(q['ratio_predicted_balanced'] for q in rows):.4f}, {max(q['ratio_predicted_balanced'] for q in rows):.4f}], "
          f"max |meas-pred| ratio = {max(abs(q['ratio_measured'] - q['ratio_predicted_balanced']) for q in rows):.4f}, "
          f"max rel err dominant = {max(q['max_rel_err_dominant'] for q in rows):.2e}, bulk = {max(q['max_rel_err_bulk'] for q in rows):.2e}, "
          f"sigma range [{sg[:, :r].min():.3f}, {sg[:, :r].max():.3f}]")
axes[0].set_title("ratio_paper during GD: measured vs closed form\nfrom end-to-end singular values (dashed = L)", fontsize=9)
axes[0].set_xlabel("GD step"); axes[0].legend(fontsize=6.5)
axes[1].set_title("end-to-end singular values sigma_1, sigma_2 (target: 1, 1)", fontsize=9)
axes[1].set_xlabel("GD step"); axes[1].legend(fontsize=6.5)
axes[2].set_yscale("log"); axes[2].set_title("max relative error of the closed form (eigenvalue-wise)\nand balancedness defect", fontsize=9)
axes[2].set_xlabel("GD step"); axes[2].legend(fontsize=6.5)
fig.suptitle("e1 post-hoc: the transient ratio != L during GD is explained by sigma_1 != sigma_2, not by loss of balancedness", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{FIG}/{EXP}_ratio_excursion.png", dpi=120)
plt.close(fig)
out["part4_ratio_excursion"] = p4

# ------------------------------------------------------------------ (b) part 3 tail closed form
p3 = {}
for run in [k for k in by_run if k.startswith("p3_")]:
    d = by_run[run]
    L, r, d_in, d_out, t = d["L"], d["r"], d["d_in"], d["d_out"], d["tail_value"]
    e = np.load(f"{RAW}/{run}.npz")["eig_Ho"]
    n_dom, n_bulk = r * r, r * (d_in + d_out - 2 * r)
    n_tail = (d_in - r) * (d_out - r)
    bulk_cf = sum(t ** (2 * k) for k in range(L))
    tail_cf = L * t ** (2 * (L - 1))
    p3[run] = {
        "tail_value": t, "dominant_cf": L, "bulk_cf": bulk_cf, "third_cluster_cf": tail_cf, "n_third_cluster": n_tail,
        "ratio_cf": L / bulk_cf,
        "dominant_measured_mean": float(e[:n_dom].mean()), "bulk_measured_mean": float(e[n_dom:n_dom + n_bulk].mean()),
        "third_cluster_measured_mean": float(e[n_dom + n_bulk:n_dom + n_bulk + n_tail].mean()),
        "third_cluster_measured_minmax": [float(e[n_dom + n_bulk:n_dom + n_bulk + n_tail].min()), float(e[n_dom + n_bulk:n_dom + n_bulk + n_tail].max())],
        "ratio_measured": d["cluster_Ho"]["ratio_paper"],
        "n_nonzero_measured": int((e > 1e-12 * e[0]).sum()), "n_nonzero_cf": n_dom + n_bulk + (n_tail if t > 0 else 0),
    }
    q = p3[run]
    print(f"{run}: dominant {q['dominant_measured_mean']:.6f} (cf {L}), bulk {q['bulk_measured_mean']:.6f} (cf {bulk_cf:.6f}), "
          f"third cluster {q['third_cluster_measured_mean']:.3e} [{q['third_cluster_measured_minmax'][0]:.3e},{q['third_cluster_measured_minmax'][1]:.3e}] "
          f"(cf {tail_cf:.3e} x{n_tail}), ratio {q['ratio_measured']:.6f} (cf {q['ratio_cf']:.6f}), nonzero {q['n_nonzero_measured']} (cf {q['n_nonzero_cf']})")
out["part3_tail_closed_form"] = p3

json.dump(out, open(f"{RES}/posthoc.json", "w"), indent=1)
print("wrote", f"{RES}/posthoc.json", "and", f"{FIG}/{EXP}_ratio_excursion.png")
