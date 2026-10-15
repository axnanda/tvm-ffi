"""e7: structure of the Jacobian of the surrogate-gradient field dg/dtheta for SNNs.

Hypothesis (docs/theory_notes.md section 5): with a Heaviside forward pass, earlier-layer weights
reach later layers only through spikes (zero derivative a.e.), so dg/dtheta is block
upper-triangular in layer order, its eigenvalues are the union of the per-layer diagonal blocks,
and the readout block is exactly I_{d_out} (x) Sigma_a (uncentred second moment of the readout
features).  Consequently no cross-layer coherent summation (the paper's depth mechanism) can
occur in the update-field Jacobian.  Contrast with relu / smooth_snn, whose Hessians are symmetric.
"""
import json, time, sys
import numpy as np, torch
import snnland as sl
sl.set_threads(1)

OUT = "results/e7_update_field_structure"
SETTINGS = {"A": (1, 4), "B": (2, 6)}
n, d_in, h, T = 256, 16, 16, 8
rows = []

def block_norms(H, slices):
    L = len(slices)
    B = np.zeros((L, L))
    for i, si in enumerate(slices):
        for j, sj in enumerate(slices):
            B[i, j] = H[si, sj].norm().item()
    return B

def analyse(kind, spec, params, X, Y, tag, extra):
    t0 = time.time()
    H = sl.full_hessian(spec, params, X, Y)
    Ho = sl.gauss_newton(spec, params, X, Y)
    slices = spec.layer_slices; L = spec.L
    B = block_norms(H, slices)
    lower = max([B[i, j] for i in range(L) for j in range(L) if j < i] + [0.0])
    upper = max([B[i, j] for i in range(L) for j in range(L) if j > i] + [0.0])
    ev = np.linalg.eigvals(H.numpy())
    diag_eigs = np.concatenate([np.linalg.eigvals(H[s, s].numpy()) for s in slices])
    a = np.sort(ev.real)[::-1]; b = np.sort(diag_eigs.real)[::-1]
    feats = spec.hidden_features(params, X)
    Sa = (feats.T @ feats / n).numpy(); ev_Sa = np.sort(np.linalg.eigvalsh(Sa))[::-1]
    mu = feats.mean(0).numpy(); cov = np.cov(feats.numpy().T, bias=True); ev_cov = np.sort(np.linalg.eigvalsh(cov))[::-1]
    Hro = H[slices[-1], slices[-1]].numpy()
    ro_eigs = np.sort(np.linalg.eigvalsh((Hro + Hro.T) / 2))[::-1]
    # readout block == I kron Sigma_a ?
    kron = np.kron(np.eye(spec.d_out), Sa)
    ro_err = np.linalg.norm(Hro - kron) / np.linalg.norm(kron)
    hidden_ranges = [(float(np.linalg.eigvals(H[s, s].numpy()).real.min()), float(np.linalg.eigvals(H[s, s].numpy()).real.max())) for s in slices[:-1]]
    hidden_norms = [float(H[s, s].norm()) for s in slices[:-1]]
    resid = (spec.forward(params, X) - Y).norm().item() / np.sqrt(n)
    loss = sl.loss_fn(spec, params, X, Y).item()
    Hs = sl.symmetrize(H); es = np.sort(np.linalg.eigvalsh(Hs.numpy()))[::-1]
    eo = np.sort(np.linalg.eigvalsh(Ho.numpy()))[::-1]
    coh = sl.layer_coherence(Ho, slices)
    row = dict(kind=kind, tag=tag, L=L, P=int(H.shape[0]), loss=loss, resid_rms=resid,
               asym=sl.asymmetry(H), max_lower_block_norm=lower, max_upper_block_norm=upper,
               block_norms=B.tolist(), eig_union_err=float(np.abs(a - b).max()),
               max_imag=float(np.abs(ev.imag).max()),
               readout_block_vs_I_kron_Sigma_rel_err=float(ro_err),
               readout_block_top=ro_eigs[: 2 * spec.d_out].tolist(), Sigma_a_top=ev_Sa[:6].tolist(),
               cov_a_top=ev_cov[:6].tolist(), mu_sq=float(mu @ mu),
               ratio_Sigma_a=float(ev_Sa[0] / ev_Sa[1]) if ev_Sa[1] > 0 else None,
               hidden_block_eig_ranges=hidden_ranges, hidden_block_norms=hidden_norms,
               eigH_real_top=a[:12].tolist(), eigH_real_bottom=a[-6:].tolist(),
               eig_symH_top=es[:12].tolist(), eig_symH_bottom=es[-6:].tolist(),
               eig_Ho_top=eo[:12].tolist(), coherence_Ho=coh["coherence"],
               n_neg_symH=int((es < -1e-10).sum()), n_neg_H=int((a < -1e-10).sum()),
               wall=time.time() - t0, **extra)
    if kind == "snn":
        row["spikes"] = {k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in sl.spike_diagnostics(spec, params, X).items()}
    np.savez(f"{OUT}/raw/{kind}_{tag}.npz", eigH=ev, eig_symH=es, eig_Ho=eo, Sigma_a=ev_Sa, B=B)
    return row

smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
Ls = [2, 3, 4, 6] if not smoke else [2, 3]
seeds = [0, 1] if not smoke else [0]
steps = 400 if not smoke else 20
for setting, (r, d_out) in SETTINGS.items():
    for seed in seeds:
        X = sl.make_whitened_inputs(n, d_in, seed); Y, _ = sl.make_rank_r_targets(X, r, d_out, seed)
        for L in Ls:
            for kind in (["snn", "relu", "smooth_snn"] if L in (2, 4) else ["snn"]):
                spec = sl.ModelSpec(kind, L, d_in, h, d_out, T=T)
                params = spec.init_params(seed)
                extra = dict(setting=setting, r=r, d_out=d_out, seed=seed)
                rows.append(analyse(kind, spec, params, X, Y, f"{setting}_L{L}_s{seed}_init", dict(step=0, **extra)))
                print(json.dumps({k: rows[-1][k] for k in ["kind", "tag", "asym", "max_lower_block_norm", "eig_union_err", "readout_block_vs_I_kron_Sigma_rel_err", "coherence_Ho", "wall"]}), flush=True)
                hist = sl.gd_train(spec, params, X, Y, lr=5e-3, steps=steps, optimizer="adam")
                p2 = hist["params"]
                rows.append(analyse(kind, spec, p2, X, Y, f"{setting}_L{L}_s{seed}_step{steps}", dict(step=steps, loss_curve=[float(v) for v in hist["loss"][::10]], **extra)))
                print(json.dumps({k: rows[-1][k] for k in ["kind", "tag", "loss", "asym", "max_lower_block_norm", "eig_union_err", "readout_block_vs_I_kron_Sigma_rel_err", "coherence_Ho", "wall"]}), flush=True)
                with open(f"{OUT}/summary{'_smoke' if smoke else ''}.json", "w") as f:
                    json.dump(rows, f, indent=1)
print("DONE")
