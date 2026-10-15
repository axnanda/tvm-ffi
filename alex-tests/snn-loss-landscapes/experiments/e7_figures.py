"""Figures for e7 (block structure of the SNN update-field Jacobian)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = json.load(open("results/e7_update_field_structure/summary.json"))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
# (a) block norm heatmaps, L=4, setting B, seed 0, init
for i, kind in enumerate(["snn", "relu"]):
    r = [x for x in rows if x["kind"] == kind and x["L"] == 4 and x["setting"] == "B" and x["seed"] == 0 and x["step"] == 0][0]
    B = np.array(r["block_norms"]); B = B / B.max()
    ax = axes[i]; im = ax.imshow(B, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"{kind}: ||d g_l / d W_m||_F (normalised), L=4, init")
    ax.set_xlabel("m (w.r.t. layer m weights)"); ax.set_ylabel("l (gradient of layer l)")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels([f"W{m+1}" for m in range(4)]); ax.set_yticklabels([f"g{l+1}" for l in range(4)])
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{B[a,b]:.2f}", ha="center", va="center", color="w" if B[a,b] < 0.6 else "k", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
# (c) top eigenvalue of H vs top eigenvalue of Sigma_a, by L, snn, after training
ax = axes[2]
for setting, mk in [("A", "o"), ("B", "s")]:
    for step, col in [(0, "tab:gray"), (400, "tab:red")]:
        rs = [x for x in rows if x["kind"] == "snn" and x["setting"] == setting and x["step"] == step]
        ax.scatter([x["Sigma_a_top"][0] for x in rs], [x["eigH_real_top"][0] for x in rs], marker=mk, c=col, label=f"snn {setting} step {step}", alpha=0.8)
lim = [0, 2.6]; ax.plot(lim, lim, "k--", lw=1, label="y = x"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("lambda_max(Sigma_a) of readout rates"); ax.set_ylabel("max Re eig(dg/dtheta)")
ax.set_title("SNN: top update-field eigenvalue vs readout feature moment"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/e7_update_field_structure.png", dpi=120)
# (d) asymmetry, union error and coherence vs L
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
Ls = sorted(set(x["L"] for x in rows if x["kind"] == "snn"))
for ax, key, title in zip(axes, ["asym", "coherence_Ho", "ratio_Sigma_a"], ["asymmetry ||H-H^T||/||H|| (snn)", "coherence of H_o^sur (snn)", "lambda_1/lambda_2 of Sigma_a (snn)"]):
    for step, col in [(0, "tab:gray"), (400, "tab:red")]:
        vals = [np.mean([x[key] for x in rows if x["kind"] == "snn" and x["L"] == L and x["step"] == step]) for L in Ls]
        ax.plot(Ls, vals, "o-", c=col, label=f"step {step}")
    if key == "coherence_Ho": ax.plot(Ls, Ls, "k--", lw=1, label="y = L")
    ax.set_xlabel("L"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/e7_scalars_vs_L.png", dpi=120); print("saved")
