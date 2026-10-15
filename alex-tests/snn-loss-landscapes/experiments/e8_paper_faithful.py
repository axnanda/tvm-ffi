"""e8: the paper's own experimental configuration (d_0=10, hidden 20, d_L=16, r=4; L = 2..7),
with the target ALIGNED with the balanced initialisation (Initialization 2.1 of the paper:
Wstar = (W_L ... W_1)/s^L has unit singular values on the init's r-dimensional subspace), for
linear (paper's theorem), tanh (paper's only nonlinear experiment, L=2..7 'Gap = L'), relu, and
LIF-SNN arms.  Expected for linear: r^2=16 dominant, (d_0+d_L-2r)r=72 bulk, 832 zeros at L=3,
ratio == L at init and throughout GD (aligned target keeps balancedness/alignment).
"""
import json, sys, time
import numpy as np, torch
import snnland as sl
sl.set_threads(1)

OUT = "results/e8_paper_faithful"
d0, hdim, dL, r, n = 10, 20, 16, 4, 512
smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
Ls = [2, 3, 4, 5, 6, 7] if not smoke else [2, 3]
seeds = [0, 1] if not smoke else [0]
steps = 1000 if not smoke else 30
rows = []

def aligned_target(spec, params, X, s):
    W = None
    for l in range(1, spec.L + 1):
        Wl = params[f"W{l}"]
        W = Wl if W is None else Wl @ W
    Wstar = W / (s ** spec.L)
    sv = torch.linalg.svdvals(Wstar)
    return X @ Wstar.T, sv

def stats(spec, params, X, Y, step, full):
    t0 = time.time()
    Ho = sl.gauss_newton(spec, params, X, Y)
    eo = sl.eigvals(Ho)
    cs = sl.cluster_stats(eo, r, d0, dL)
    coh = sl.layer_coherence(Ho, spec.layer_slices)
    row = dict(step=step, loss=sl.loss_fn(spec, params, X, Y).item(),
               Ho_top=eo[:20].tolist(), Ho_ratio_paper=cs["ratio_paper"], Ho_ratio_gap=cs["ratio_gap"],
               Ho_ratio_split=cs["ratio_split"], Ho_split_n_dominant=cs["split_n_dominant"],
               Ho_zero_count=cs["zero_count"], Ho_rank=cs["rank"], Ho_dominant_mean=cs["dominant_mean"],
               Ho_bulk_mean=cs["bulk_mean"], Ho_bulk_min=cs["bulk_min"], Ho_bulk_max=cs["bulk_max"],
               coherence=coh["coherence"], lambda_max_blocks=[float(v) for v in coh["lambda_max_blocks"]])
    if full:
        H = sl.full_hessian(spec, params, X, Y)
        Hs = sl.symmetrize(H); es = sl.eigvals(Hs); css = sl.cluster_stats(es, r, d0, dL)
        Hf = sl.functional_part(H, Ho)
        row.update(asym=sl.asymmetry(H), Hf_over_Ho=(Hf.norm() / Ho.norm()).item(),
                   symH_top=es[:20].tolist(), symH_ratio_paper=css["ratio_paper"], symH_ratio_split=css["ratio_split"],
                   symH_split_n_dominant=css["split_n_dominant"], symH_negative=css["negative_count"],
                   symH_zero_count=css["zero_count"])
        if spec.kind == "snn":
            ev = np.linalg.eigvals(H.numpy()); a = np.sort(ev.real)[::-1]
            sls = spec.layer_slices
            lower = max(H[si, sj].norm().item() for i, si in enumerate(sls) for j, sj in enumerate(sls) if j < i)
            row.update(eigH_real_top=a[:20].tolist(), eigH_real_min=float(a[-1]), max_lower_block_norm=lower,
                       max_imag=float(np.abs(ev.imag).max()))
        np.savez(f"{OUT}/raw/{spec.kind}_{row.get('tag','')}", eig_Ho=eo.numpy(), eig_symH=es.numpy())
    if spec.kind == "snn":
        sd = sl.spike_diagnostics(spec, params, X)
        row["mean_rate"] = float(sd["mean_rate"]); row["window_frac"] = float(sd["window_frac"])
        row["per_layer_mean_rate"] = [float(v) for v in sd["per_layer_mean_rate"]]
    row["wall"] = time.time() - t0
    return row

ARMS = [  # (kind, init kwargs, optimizer, lr)
    ("linear", dict(s=0.5, tail_value=0.0), "gd", 0.1),
    ("tanh", dict(s=0.5, tail_value=0.0), "gd", 0.1),
    ("relu", dict(s=0.5, tail_value=0.0), "gd", 0.1),
    ("snn", dict(s=3.0, tail_value=0.0), "adam", 5e-3),
    ("snn", dict(s=3.0, tail_value=3.0), "adam", 5e-3),
]
for seed in seeds:
    X = sl.make_whitened_inputs(n, d0, seed)
    for L in Ls:
        for kind, ik, opt, lr in ARMS:
            if seed > 0 and (kind == "relu" or (kind == "snn" and ik["tail_value"] > 0)):
                continue  # budget: second seed only for linear / tanh / snn(tail 0)
            spec = sl.ModelSpec(kind, L, d0, hdim, dL, T=8)
            params = spec.init_params(seed, init="balanced_uniform", r=r, **ik)
            Y, sv = aligned_target(spec, params, X, ik["s"])
            tag = f"{kind}_tail{ik['tail_value']}_L{L}_s{seed}"
            run = dict(kind=kind, L=L, seed=seed, init=ik, P=spec.n_params, target_sv=sv[:6].tolist(), tag=tag, opt=opt, lr=lr, checkpoints=[])
            full_steps = {0, steps}  # budget: full surrogate Hessian only at init and end
            every = max(1, steps // 20)
            def cb(step, p, loss):
                pass
            # manual loop with checkpoints via gd_train in chunks
            p = params
            for s0 in range(0, steps + 1, every):
                row = stats(spec, p, X, Y, s0, s0 in full_steps); row["tag"] = f"{tag}_step{s0}"
                run["checkpoints"].append(row)
                if s0 == steps: break
                hist = sl.gd_train(spec, p, X, Y, lr=lr, steps=every, optimizer=opt)
                p = hist["params"]
            # balancedness at end
            bal = []
            for l in range(1, L):
                A = p[f"W{l+1}"].T @ p[f"W{l+1}"]; B = p[f"W{l}"] @ p[f"W{l}"].T
                bal.append((A - B).norm().item() / max(B.norm().item(), 1e-12))
            run["final_balancedness_rel"] = bal
            rows.append(run)
            c0, c1 = run["checkpoints"][0], run["checkpoints"][-1]
            print(json.dumps(dict(tag=tag, P=spec.n_params, loss0=c0["loss"], loss1=c1["loss"], ratio0=c0["Ho_ratio_paper"], ratio1=c1["Ho_ratio_paper"],
                                  coh0=c0["coherence"], coh1=c1["coherence"], nd0=c0["Ho_split_n_dominant"], nd1=c1["Ho_split_n_dominant"],
                                  zeros0=c0["Ho_zero_count"], rate=c1.get("mean_rate"), asym1=c1.get("asym"), wall=sum(c["wall"] for c in run["checkpoints"]))), flush=True)
            with open(f"{OUT}/summary{'_smoke' if smoke else ''}.json", "w") as f:
                json.dump(rows, f, indent=1)
print("DONE")
