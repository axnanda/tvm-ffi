"""Independent re-computation of headline numbers from raw/summary files + a fresh-seed SNN check.
Replaces the per-experiment verifier agents.  Output: results/verify_headline.json"""
import json, glob, numpy as np, time
import snnland as sl
sl.set_threads(1)
out = {}
# --- e1: exactness at init from raw eigenvalues
e1 = json.load(open("results/e1_linear_replication/summary.json"))
chk = []
for r in e1:
    if r.get("part") in ("p1", "part1", 1) or str(r.get("run", "")).startswith("p1"):
        L, rr, din, dout = r["L"], r["r"], r["d_in"], r["d_out"]
        f = glob.glob(f"results/e1_linear_replication/raw/{r['run']}*.npz")
        if not f: continue
        e = np.sort(np.load(f[0])["eig_Ho"])[::-1]
        nb = rr * (din + dout - 2 * rr)
        chk.append(dict(run=r["run"], L=L, top_r2=float(e[:rr*rr].mean()), top_r2_spread=float(e[:rr*rr].max()-e[:rr*rr].min()),
                        bulk_mean=float(e[rr*rr:rr*rr+nb].mean()), bulk_spread=float(e[rr*rr:rr*rr+nb].max()-e[rr*rr:rr*rr+nb].min()),
                        rest_max=float(np.abs(e[rr*rr+nb:]).max()) if len(e) > rr*rr+nb else 0.0))
out["e1_part1_recomputed"] = chk
# --- e2a: recompute slopes of coherence and ratio_gap vs L from summary
e2 = json.load(open("results/e2a_snn_depth/summary.json"))
print("e2a keys sample:", list(e2[0].keys())[:60] if isinstance(e2, list) else type(e2))
out["e2a_rows"] = len(e2) if isinstance(e2, list) else None
# --- e8: linear exactness + tanh/snn ratios
e8 = json.load(open("results/e8_paper_faithful/summary.json"))
lin = [(r["L"], r["checkpoints"][0]["Ho_ratio_paper"], r["checkpoints"][-1]["Ho_ratio_paper"], r["checkpoints"][0]["Ho_zero_count"], r["P"]) for r in e8 if r["kind"] == "linear"]
out["e8_linear"] = [dict(L=L, ratio_init=a, ratio_end=b, zeros=z, expected_zeros=P - 16 - 72) for L, a, b, z, P in lin]
def slope(xs, ys):
    xs, ys = np.array(xs, float), np.array(ys, float); A = np.vstack([xs, np.ones_like(xs)]).T
    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]; pred = m*xs+c; r2 = 1 - ((ys-pred)**2).sum()/max(((ys-ys.mean())**2).sum(), 1e-30)
    return float(m), float(c), float(r2)
for kind, tail in [("tanh", 0.0), ("snn", 0.0), ("snn", 3.0), ("relu", 0.0)]:
    rs = [r for r in e8 if r["kind"] == kind and r["init"]["tail_value"] == tail]
    for ci, nm in [(0, "init"), (-1, "end")]:
        xs = [r["L"] for r in rs]
        out[f"e8_{kind}_tail{tail}_{nm}_slopes"] = {k: slope(xs, [r["checkpoints"][ci][k] for r in rs]) for k in ["Ho_ratio_paper", "coherence", "Ho_ratio_gap"]}
# --- fresh-seed SNN check (seed 11, setting B, L=2 and L=6): coherence, gap at r^2, dominant count, asymmetry, lower-block norm
fresh = []
for L in [2, 6]:
    t0 = time.time()
    X = sl.make_whitened_inputs(256, 16, 11); Y, _ = sl.make_rank_r_targets(X, 2, 6, 11)
    m = sl.ModelSpec("snn", L, 16, 16, 6, T=8); p = m.init_params(11)
    Ho = sl.gauss_newton(m, p, X, Y); H = sl.full_hessian(m, p, X, Y)
    eo = sl.eigvals(Ho); cs = sl.cluster_stats(eo, 2, 16, 6); coh = sl.layer_coherence(Ho, m.layer_slices)
    sls = m.layer_slices
    lower = max(H[si, sj].norm().item() for i, si in enumerate(sls) for j, sj in enumerate(sls) if j < i)
    fresh.append(dict(L=L, coherence=coh["coherence"], ratio_gap=cs["ratio_gap"], ratio_paper=cs["ratio_paper"], split_n_dominant=cs["split_n_dominant"],
                      asym=sl.asymmetry(H), max_lower_block=lower, Hf_over_Ho=(sl.functional_part(H, Ho).norm()/Ho.norm()).item(), wall=time.time()-t0))
    print("fresh", fresh[-1], flush=True)
out["fresh_seed11_snn_B"] = fresh
json.dump(out, open("results/verify_headline.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k.startswith("e8_") and "slopes" in k}, indent=0))
print("e1 recomputed (first 5):", chk[:5])
