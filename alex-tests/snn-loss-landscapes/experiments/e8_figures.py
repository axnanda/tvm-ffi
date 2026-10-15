"""Figures for e8 (paper-faithful configuration, L = 2..7)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
s = json.load(open("results/e8_paper_faithful/summary.json"))
arms = [("linear", 0.0, "linear (paper)"), ("tanh", 0.0, "tanh (paper's nonlinear test)"), ("relu", 0.0, "relu"), ("snn", 0.0, "LIF-SNN, rank-4 balanced init"), ("snn", 3.0, "LIF-SNN, full-rank orthogonal init")]
cols = ["k", "tab:green", "tab:orange", "tab:red", "tab:purple"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
keys = [("Ho_ratio_paper", "ratio_paper = mean(top r^2=16) / mean(next 72)"), ("Ho_ratio_gap", "gap at index r^2: eig[16]/eig[17]"), ("coherence", "coherence lambda_max(H_o)/max_l lambda_max(B_ll)"), ("Ho_split_n_dominant", "data-driven dominant count (paper: 16)")]
for row, (ci, title) in enumerate([(0, "init"), (-1, "after training (1000 steps)")]):
    for col, (key, label) in enumerate(keys):
        ax = axes[row, col]
        for (kind, tail, name), c in zip(arms, cols):
            rs = [r for r in s if r["kind"] == kind and r["init"]["tail_value"] == tail]
            Ls = sorted(set(r["L"] for r in rs))
            m = [np.mean([r["checkpoints"][ci][key] for r in rs if r["L"] == L]) for L in Ls]
            ax.plot(Ls, m, "o-", c=c, label=name, alpha=0.85)
        if key in ("Ho_ratio_paper", "Ho_ratio_gap", "coherence"):
            ax.plot([2, 7], [2, 7], "--", c="gray", lw=1, label="y = L")
        if key == "Ho_split_n_dominant":
            ax.axhline(16, ls="--", c="gray", lw=1, label="r^2 = 16")
        ax.set_title(f"{label} [{title}]", fontsize=9); ax.set_xlabel("L"); ax.grid(alpha=0.3)
        if row == 0 and col == 0: ax.legend(fontsize=7)
plt.suptitle("e8: paper configuration d_0=10, hidden 20, d_L=16, r=4, aligned balanced targets; means over seeds")
plt.tight_layout(); plt.savefig("figures/e8_paper_faithful_vs_L.png", dpi=120)
# spectra at L=5 init
fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
for ax, (kind, tail, name) in zip(axes, arms):
    for L, c in [(2, "tab:blue"), (5, "tab:red")]:
        for step in [0, 1000]:
            try:
                d = np.load(f"results/e8_paper_faithful/raw/{kind}_L{L}_tail{tail}_s0_step{step}.npz")
            except FileNotFoundError:
                continue
            e = d["eig_Ho"]; e = e[e > 1e-12 * e.max()]
            ax.plot(np.arange(1, len(e) + 1), e / e.max(), "-" if step == 0 else ":", c=c, label=f"L={L} step {step}")
    ax.axvline(16, c="gray", ls="--", lw=1); ax.axvline(88, c="gray", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(name, fontsize=9); ax.set_xlabel("index"); ax.legend(fontsize=7)
axes[0].set_ylabel("eig(H_o) / max")
plt.suptitle("e8 spectra of H_o (normalised); dashed = index 16 (r^2), dotted = index 88 (r^2 + bulk)")
plt.tight_layout(); plt.savefig("figures/e8_spectra.png", dpi=120); print("figures saved")
