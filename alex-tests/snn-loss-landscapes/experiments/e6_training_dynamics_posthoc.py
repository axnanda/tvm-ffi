import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""e6 post-hoc: block structure of H_f = H - H_o at the END of the e6 GD runs (seed 0).

Re-trains snn / linear_bal / relu / smooth_snn (seed 0, same lr as the e6 run, deterministic -> same
trajectory, no callbacks) and decomposes ||H_f||_F^2 into block-lower, block-diagonal and block-upper parts
(layer blocks), at init and at step 1500.  Theory note 5: for the snn the lower blocks of H vanish, so H_f
keeps -lower(H_o) and ||H_f||/||H_o|| cannot go to 0.  Output: results/e6_training_dynamics/posthoc_blocks.json
"""
import json, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e6_training_dynamics")
N, D_IN, WIDTH, L, R, D_OUT, STEPS = 256, 16, 16, 4, 2, 6, 1500
runs = json.load(open(os.path.join(OUT, "summary.json")))
lr_of = {r["name"]: r["lr"] for r in runs}


def block_parts(M, slices):
    low = diag = up = 0.0
    for i, si in enumerate(slices):
        for j, sj in enumerate(slices):
            v = float(torch.linalg.norm(M[si, sj])) ** 2
            if i == j: diag += v
            elif i > j: low += v
            else: up += v
    tot = float(torch.linalg.norm(M)) ** 2
    return {"lower_frac": low / tot, "diag_frac": diag / tot, "upper_frac": up / tot, "fro": tot ** 0.5}


def analyse(spec, params, X, Y, tag):
    t0 = time.time()
    H = sl.full_hessian(spec, params, X, Y); H_o = sl.gauss_newton(spec, params, X, Y); H_f = H - H_o
    sl_ = spec.layer_slices
    out = {"tag": tag, "loss": float(sl.loss_fn(spec, params, X, Y)), "Hf_over_Ho": float(torch.linalg.norm(H_f) / torch.linalg.norm(H_o)),
           "asymmetry_H": sl.asymmetry(H), "H": block_parts(H, sl_), "H_o": block_parts(H_o, sl_), "H_f": block_parts(H_f, sl_)}
    # norm of H_f restricted to each block triangle, relative to ||H_o||
    Ho_n = float(torch.linalg.norm(H_o))
    for part in ("lower", "diag", "upper"):
        out[f"Hf_{part}_over_Ho"] = (out["H_f"][f"{part}_frac"] ** 0.5) * out["Hf_over_Ho"]
    out["Ho_lower_over_Ho"] = out["H_o"]["lower_frac"] ** 0.5
    # is H_f's lower triangle exactly -H_o's lower triangle (i.e. H lower blocks == 0)?
    low_H = sum(float(torch.linalg.norm(H[si, sj])) ** 2 for i, si in enumerate(sl_) for j, sj in enumerate(sl_) if i > j) ** 0.5
    out["H_lower_fro_over_H_fro"] = low_H / float(torch.linalg.norm(H))
    # readout block of H vs I kron Sigma_a (theory note 5 ii)
    a = spec.hidden_features(params, X).detach(); Sig = a.T @ a / N
    blk = H[sl_[-1], sl_[-1]]
    out["readout_block_rel_err_vs_I_kron_Sigma"] = float(torch.linalg.norm(blk - torch.kron(torch.eye(D_OUT, dtype=blk.dtype), Sig)) / torch.linalg.norm(blk))
    out["wall"] = time.time() - t0
    print(json.dumps(out), flush=True)
    return out


res = []
for label, kind, init in [("snn", "snn", None), ("linear_bal", "linear", "balanced_uniform"), ("relu", "relu", None), ("smooth_snn", "smooth_snn", None)]:
    X = sl.make_whitened_inputs(N, D_IN, 0); Y, _ = sl.make_rank_r_targets(X, R, D_OUT, 0)
    spec = sl.ModelSpec(kind, L, D_IN, WIDTH, D_OUT)
    p = spec.init_params(0, init="balanced_uniform", r=R, s=1.0) if init == "balanced_uniform" else spec.init_params(0)
    lr = lr_of[f"{label}_s0"]
    res.append(analyse(spec, p, X, Y, f"{label}_s0_step0"))
    hist = sl.gd_train(spec, p, X, Y, lr, STEPS, optimizer="gd")
    e6_final = [r for r in runs if r["name"] == f"{label}_s0"][0]["loss_final"]
    print(f"[{label}] retrained: final loss {hist['loss'][-1]:.6e} vs e6 run {e6_final:.6e} (same trajectory: {abs(hist['loss'][-1] - e6_final) < 1e-9 * max(1, abs(e6_final))})", flush=True)
    r_end = analyse(spec, hist["params"], X, Y, f"{label}_s0_step{STEPS}"); r_end["e6_loss_final"] = e6_final; res.append(r_end)
json.dump(res, open(os.path.join(OUT, "posthoc_blocks.json"), "w"), indent=1)
print("POSTHOC_DONE")
