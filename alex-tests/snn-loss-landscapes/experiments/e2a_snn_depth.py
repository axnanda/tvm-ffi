import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""e2a_snn_depth: the core depth test for spiking nets (LIF + sigmoid surrogate, rate readout).

For each target setting A (r=1, d_out=4) / B (r=2, d_out=6), depth L in {2,3,4,5,6,8} and seed:
  * kaiming init (protocol default); full surrogate Hessian H at steps {0, 200, 800};
    Gauss-Newton H_o every 100 steps of Adam (lr 5e-3, 800 steps, full batch, MSE);
  * per checkpoint: eigenvalues of H_o (and sym(H)), cluster statistics, layer coherence,
    top-eigenvector layer energies, time-averaged-rate feature second moments, spike diagnostics.
Plus (seed 0, L in {2,4,8}) an "orthogonal balanced" init: balanced_uniform with r = d_out and
tail_value = s so that EVERY singular value of EVERY layer equals s (full rank), with s chosen on a
grid so the mean spike rate at init lies in [0.2, 0.4].

Usage:  python3 experiments/e2a_snn_depth.py            # full run (results/e2a_snn_depth)
        python3 experiments/e2a_snn_depth.py --smoke    # reduced run (results/e2a_snn_depth_smoke)
        python3 experiments/e2a_snn_depth.py --figures  # only regenerate figures + analysis
"""
import argparse
import json
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "e2a_snn_depth"

N, D_IN, WIDTH = 256, 16, 16
SETTINGS = {"A": (1, 4), "B": (2, 6)}
LS = [2, 3, 4, 5, 6, 8]
LR, STEPS, HO_EVERY = 5e-3, 800, 100
FULL_H_STEPS = (0, 200, 800)
K_TOP = 16  # rows of the stored top-eigenvector layer-energy matrix (superset of max(r^2, d_out))
BUDGET_MIN = 72.0  # wall-clock budget for the full run (minutes); optional tiers are skipped beyond it
EST_RUN_SEC = {2: 25, 3: 45, 4: 80, 5: 125, 6: 175, 8: 300}  # measured-ish per-run cost (1 thread)
GAIN_GRID = [0.5, 0.7, 0.85, 1.0, 1.2, 1.41, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_py(o):
    """Recursively convert tensors / numpy / nan-safe scalars to JSON-serialisable python objects."""
    if isinstance(o, torch.Tensor):
        return to_py(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return [to_py(x) for x in o.tolist()] if o.ndim > 0 else to_py(o.item())
    if isinstance(o, (np.floating,)):
        return float(o)
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


def spectrum_summary(H: torch.Tensor, spec, r: int, d_in: int, d_out: int, tag: str) -> Dict[str, object]:
    """Everything the protocol asks for one matrix (one full eigh); returns (dict, eigs ndarray)."""
    Hs = sl.symmetrize(H)
    eigs, evecs = sl.eigh(Hs)
    e = eigs.numpy()
    cs = sl.cluster_stats(e, r, d_in, d_out)
    n = len(e)
    scale = float(np.max(np.abs(e)))
    # layer coherence (same formulas as sl.layer_coherence, reusing the eigh above; validated in --smoke)
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
        f"coh_{tag}": {
            "lambda_max": lam_max,
            "lambda_max_blocks": block_lams,
            "coherence": lam_max / max(block_lams) if max(block_lams) > 0 else float("nan"),
            "sum_block_lams_over_lambda_max": sum(block_lams) / lam_max if lam_max > 0 else float("nan"),
            "top_eigvec_layer_energy": energy,
            "block_alignment": align,
        },
        f"topk_energy_{tag}": topk,
        f"zero_frac_{tag}": cs["zero_count"] / n,
        f"neg_frac_{tag}": cs["negative_count"] / n,
        f"dominant_count_{tag}": cs["split_n_dominant"],
        f"trace_{tag}": float(torch.trace(Hs)),
        f"fro_{tag}": float(torch.linalg.norm(H)),
        f"lambda_min_{tag}": float(eigs[-1]),
    }
    return out, e


def make_callback(spec, X, Y, r, d_in, d_out, full_steps, raw: Dict[str, np.ndarray], verbose=True):
    def cb(step, params, g, loss_value):
        t0 = time.time()
        H_o = sl.gauss_newton(spec, params, X, Y)
        rec = {"step": int(step), "loss": float(loss_value), "grad_norm": float(torch.linalg.norm(g)),
               "P": int(spec.n_params)}
        s_o, e_o = spectrum_summary(H_o, spec, r, d_in, d_out, "Ho")
        rec.update(s_o)
        rec["asym_Ho"] = sl.asymmetry(H_o)
        raw[f"eig_Ho_step{step}"] = e_o
        rec["features"] = sl.feature_second_moment_stats(spec, params, X)
        rec["spikes"] = sl.spike_diagnostics(spec, params, X)
        rec["wall_Ho"] = time.time() - t0
        if step in full_steps:
            t1 = time.time()
            H = sl.full_hessian(spec, params, X, Y)
            H_f = sl.functional_part(H, H_o)
            s_h, e_h = spectrum_summary(H, spec, r, d_in, d_out, "H")
            rec.update(s_h)
            rec["asymmetry_H"] = sl.asymmetry(H)
            rec["Hf_norm"] = float(torch.linalg.norm(H_f))
            rec["Ho_norm"] = float(torch.linalg.norm(H_o))
            rec["H_norm"] = float(torch.linalg.norm(H))
            rec["Hf_over_Ho"] = rec["Hf_norm"] / rec["Ho_norm"] if rec["Ho_norm"] > 0 else float("nan")
            rec["asymmetry_Hf"] = sl.asymmetry(H_f)
            rec["Hf_sym_fro_over_Ho"] = float(torch.linalg.norm(sl.symmetrize(H_f))) / rec["Ho_norm"]
            raw[f"eig_H_step{step}"] = e_h
            rec["wall_H"] = time.time() - t1
        if verbose:
            msg = (f"    step {step:4d} loss={loss_value:.4e} lam_max(Ho)={rec['coh_Ho']['lambda_max']:.3e} "
                   f"coh={rec['coh_Ho']['coherence']:.2f} ndom={rec['dominant_count_Ho']} "
                   f"ratio_split={rec['cluster_Ho']['ratio_split']:.2f} rate={rec['spikes']['mean_rate']:.3f} "
                   f"wall={time.time() - t0:.1f}s")
            if step in full_steps:
                msg += f" | asym(H)={rec['asymmetry_H']:.3f} |Hf|/|Ho|={rec['Hf_over_Ho']:.3f} wall_H={rec['wall_H']:.1f}s"
            print(msg, flush=True)
        return {"rec": rec}
    return cb


def choose_gain(spec, X, lo=0.2, hi=0.4, target=0.3):
    """Pick s on GAIN_GRID so the init mean spike rate is in [lo, hi] (closest to target)."""
    rows = []
    for s in GAIN_GRID:
        p = spec.init_params(0, init="balanced_uniform", r=spec.d_out, s=s, tail_value=s)
        rate = sl.spike_diagnostics(spec, p, X)["mean_rate"]
        rows.append((s, rate))
    inside = [(s, rt) for s, rt in rows if lo <= rt <= hi]
    pool = inside if inside else rows
    s_best, rate_best = min(pool, key=lambda sr: abs(sr[1] - target))
    return s_best, rate_best, rows


def run_one(setting: str, L: int, seed: int, init: str, steps: int, ho_every: int, full_steps,
            res_dir: str, verbose=True) -> Dict[str, object]:
    r, d_out = SETTINGS[setting]
    X = sl.make_whitened_inputs(N, D_IN, seed)
    Y, Wstar = sl.make_rank_r_targets(X, r, d_out, seed)
    # sanity: data matches (r, d_in, d_out) passed to cluster_stats
    assert X.shape == (N, D_IN) and Y.shape == (N, d_out)
    assert int(torch.linalg.matrix_rank(Y)) == r, "target rank != r"
    assert float((X.T @ X / N - torch.eye(D_IN, dtype=X.dtype)).abs().max()) < 1e-8, "inputs not whitened"
    spec = sl.ModelSpec("snn", L, D_IN, WIDTH, d_out, T=8, beta=0.9, theta=1.0, k=4.0, surrogate="sigmoid",
                        reset="soft", detach_reset=False, readout="rate")
    info: Dict[str, object] = {"setting": setting, "r": r, "d_in": D_IN, "d_out": d_out, "L": L, "seed": seed,
                               "init": init, "P": spec.n_params, "steps": steps, "lr": LR,
                               "spec": repr(spec), "n": N, "width": WIDTH}
    if init == "kaiming":
        params = spec.init_params(seed)
        info["gain"] = math.sqrt(2.0)
    elif init == "orth_balanced":
        s_best, rate_best, rows = choose_gain(spec, X)
        params = spec.init_params(seed, init="balanced_uniform", r=d_out, s=s_best, tail_value=s_best)
        sv = {k: torch.linalg.svdvals(v).numpy() for k, v in params.items()}
        assert all(np.allclose(v, s_best) for v in sv.values()), "not all singular values equal"
        info["gain"] = s_best
        info["gain_sweep"] = [{"s": s, "mean_rate": rt} for s, rt in rows]
        info["init_mean_rate"] = rate_best
        info["init_desc"] = (f"balanced_uniform(r=d_out={d_out}, s={s_best}, tail_value={s_best}): all "
                             f"singular values of every W_l equal {s_best} (orthogonal up to scale)")
    else:
        raise ValueError(init)
    name = f"{setting}_L{L}_seed{seed}_{init}"
    print(f"== run {name}: P={spec.n_params} r={r} d_out={d_out} gain={info['gain']:.3f}", flush=True)
    raw: Dict[str, np.ndarray] = {}
    cb = make_callback(spec, X, Y, r, D_IN, d_out, full_steps, raw, verbose=verbose)
    t0 = time.time()
    hist = sl.gd_train(spec, params, X, Y, lr=LR, steps=steps, loss="mse", log_every=ho_every,
                       callbacks=[cb], optimizer="adam")
    wall = time.time() - t0
    checkpoints = [s["rec"] for s in hist["summaries"]]
    loss_curve = np.asarray(hist["loss"], dtype=np.float64)
    gn_curve = np.asarray(hist["grad_norm"], dtype=np.float64)
    info.update({
        "name": name, "wall_sec": wall, "logged_steps": hist["logged_steps"],
        "loss_init": float(loss_curve[0]), "loss_final": float(loss_curve[-1]),
        "loss_min": float(loss_curve.min()),
        "loss_curve_every10": [float(x) for x in loss_curve[::10]] + [float(loss_curve[-1])],
        "checkpoints": checkpoints,
    })
    raw["loss_curve"] = loss_curve
    raw["grad_norm_curve"] = gn_curve
    raw["logged_steps"] = np.asarray(hist["logged_steps"])
    raw["Wstar"] = Wstar.numpy()
    os.makedirs(os.path.join(res_dir, "raw"), exist_ok=True)
    np.savez(os.path.join(res_dir, "raw", f"{name}.npz"), **raw)
    print(f"   done {name}: loss {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f}, wall {wall:.0f}s", flush=True)
    return to_py(info)


# --------------------------------------------------------------------------- #
# smoke-test validations
# --------------------------------------------------------------------------- #
def validate_once(setting="A", L=3, seed=0):
    """Print one full stats dict and cross-check the local helpers against the library functions."""
    r, d_out = SETTINGS[setting]
    X = sl.make_whitened_inputs(N, D_IN, seed)
    Y, _ = sl.make_rank_r_targets(X, r, d_out, seed)
    spec = sl.ModelSpec("snn", L, D_IN, WIDTH, d_out)
    p = spec.init_params(seed)
    H_o = sl.gauss_newton(spec, p, X, Y)
    H = sl.full_hessian(spec, p, X, Y)
    s_o, e_o = spectrum_summary(H_o, spec, r, D_IN, d_out, "Ho")
    s_h, e_h = spectrum_summary(H, spec, r, D_IN, d_out, "H")
    # cross-check against the library
    ref_cs = sl.cluster_stats(sl.eigvals(H_o), r, D_IN, d_out)
    ref_coh = sl.layer_coherence(H_o, spec.layer_slices)
    ref_topk = sl.top_eigvec_layer_energy(H_o, spec.layer_slices, k=max(r * r, d_out)).numpy()
    for k_ in ref_cs:
        a, b = ref_cs[k_], s_o["cluster_Ho"][k_]
        assert (a == b) or (isinstance(a, float) and isinstance(b, float) and (math.isnan(a) and math.isnan(b) or abs(a - b) <= 1e-9 * max(1, abs(a)))), (k_, a, b)
    assert abs(ref_coh["coherence"] - s_o["coh_Ho"]["coherence"]) < 1e-9
    assert np.allclose(ref_coh["lambda_max_blocks"], s_o["coh_Ho"]["lambda_max_blocks"])
    assert np.allclose(ref_coh["top_eigvec_layer_energy"], s_o["coh_Ho"]["top_eigvec_layer_energy"], atol=1e-8)
    assert np.allclose(ref_coh["block_alignment"].numpy(), s_o["coh_Ho"]["block_alignment"], equal_nan=True)
    assert np.allclose(ref_topk, s_o["topk_energy_Ho"][: ref_topk.shape[0]], atol=1e-8)
    cs = s_o["cluster_Ho"]
    assert cs["n_eigs"] == spec.n_params and cs["n_dominant_expected"] == r * r
    assert cs["n_bulk_expected"] == r * (D_IN + d_out - 2 * r)
    assert abs(sum(e_o) - float(torch.trace(H_o))) < 1e-8 * abs(float(torch.trace(H_o)))
    assert e_o[-1] > -1e-10 * e_o[0], "H_o not PSD?"
    assert sl.asymmetry(H_o) < 1e-12
    feats = sl.feature_second_moment_stats(spec, p, X)
    # last-layer block of H_o is I_{d_out} kron M (M = a^T a / n): its top eigenvalue is lambda_max(M)
    assert abs(s_o["coh_Ho"]["lambda_max_blocks"][-1] - feats["second_moment_top3"][0]) < 1e-8 * feats["second_moment_top3"][0]
    fd = sl.check_hessian_fd(spec, p, X, Y, H=H, n_dirs=2, eps=1e-6)
    print("[validate] finite-difference rel. errors of H:", fd)
    print("[validate] all library cross-checks passed for", spec)
    rec = {"step": 0, "P": spec.n_params}
    rec.update(s_o)
    rec.update(s_h)
    rec["asymmetry_H"] = sl.asymmetry(H)
    rec["Hf_over_Ho"] = float(torch.linalg.norm(H - H_o) / torch.linalg.norm(H_o))
    rec["features"] = feats
    rec["spikes"] = sl.spike_diagnostics(spec, p, X)
    print("[validate] ONE FULL STATS DICT (init, setting %s, L=%d, seed=%d):" % (setting, L, seed))
    print(json.dumps(to_py(rec), indent=1, default=str))
    print("[validate] top-12 eigs H_o:", np.round(e_o[:12], 4))
    print("[validate] top-12 eigs sym(H):", np.round(e_h[:12], 4), " min:", e_h[-1])


# --------------------------------------------------------------------------- #
# aggregation / figures
# --------------------------------------------------------------------------- #
def load_summary(res_dir):
    with open(os.path.join(res_dir, "summary.json")) as f:
        return json.load(f)


def ckpt(run, step):
    for c in run["checkpoints"]:
        if c["step"] == step:
            return c
    return None


def get(c, path):
    """Nested lookup 'a.b.c' with None -> nan."""
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
    return np.nan if c is None else c


def agg(runs, setting, init, step, path, Ls=LS):
    """mean, std, n over seeds per L of a nested scalar."""
    means, stds, ns, per = [], [], [], {}
    for L in Ls:
        vals = [get(ckpt(r_, step), path) for r_ in runs
                if r_["setting"] == setting and r_["init"] == init and r_["L"] == L and ckpt(r_, step) is not None]
        vals = np.asarray([v for v in vals if v is not None], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        per[L] = vals.tolist()
        means.append(vals.mean() if len(vals) else np.nan)
        stds.append(vals.std() if len(vals) > 1 else 0.0)
        ns.append(len(vals))
    return np.asarray(means), np.asarray(stds), ns, per


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x[m], y[m], 1)
    pred = a * x[m] + b
    ss_res = ((y[m] - pred) ** 2).sum()
    ss_tot = ((y[m] - y[m].mean()) ** 2).sum()
    return float(a), float(b), float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def make_figures(res_dir, fig_dir, exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_summary(res_dir)
    os.makedirs(fig_dir, exist_ok=True)
    kai = [r_ for r_ in runs if r_["init"] == "kaiming"]
    Ls_present = sorted({r_["L"] for r_ in kai})
    steps_full = sorted({c["step"] for r_ in kai for c in r_["checkpoints"] if "cluster_H" in c})
    steps_all = sorted({c["step"] for r_ in kai for c in r_["checkpoints"]})
    s_init, s_mid, s_end = steps_full[0], steps_full[len(steps_full) // 2], steps_full[-1]
    analysis: Dict[str, object] = {"Ls": Ls_present, "full_H_steps": steps_full, "Ho_steps": steps_all}
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(Ls_present)))
    ckcol = {s_init: "tab:blue", s_mid: "tab:orange", s_end: "tab:red"}

    # (a) spectra -----------------------------------------------------------------
    for setting, (r, d_out) in SETTINGS.items():
        fig, axes = plt.subplots(2, len(Ls_present), figsize=(3.2 * len(Ls_present), 6.4), squeeze=False)
        for j, L in enumerate(Ls_present):
            for i, tag in enumerate(["Ho", "H"]):
                ax = axes[i, j]
                emax = 0.0
                for r_ in kai:
                    if r_["setting"] != setting or r_["L"] != L:
                        continue
                    z = np.load(os.path.join(res_dir, "raw", r_["name"] + ".npz"))
                    for st in (s_init, s_end):
                        key = f"eig_{tag}_step{st}"
                        if key not in z:
                            continue
                        e = z[key]
                        emax = max(emax, float(np.abs(e).max()))
                        idx = np.arange(1, len(e) + 1)
                        pos = e > 0
                        ax.plot(idx[pos], e[pos], color=ckcol[st], lw=0.7, alpha=0.7,
                                label=f"step {st}" if r_["seed"] == 0 else None)
                        if (~pos).any():
                            ax.plot(idx[~pos], np.abs(e[~pos]), color=ckcol[st], lw=0.7, alpha=0.5, ls=":",
                                    label=f"|neg| step {st}" if r_["seed"] == 0 else None)
                ax.axvline(r * r, color="k", ls="--", lw=0.8, label=f"r^2={r*r}")
                ax.axvline(d_out, color="gray", ls="-.", lw=0.8, label=f"d_out={d_out}")
                ax.set_xscale("log"); ax.set_yscale("log")
                if emax > 0:  # the null space of H_o is numerically ~1e-20..1e-170: clip to 10 decades
                    ax.set_ylim(1e-10 * emax, 3 * emax)
                ax.set_title(f"{'H_o' if tag == 'Ho' else 'sym(H)'}  L={L}", fontsize=9)
                ax.set_xlabel("index (1-based)")
                if j == 0:
                    ax.set_ylabel("eigenvalue (|.| dotted for <0)")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.suptitle(f"{exp}: sorted spectra, setting {setting} (r={r}, d_out={d_out}); SNN kaiming init; thin lines = seeds", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_spectra_{setting}.png"), dpi=120)
        plt.close(fig)

    # (b) ratios + coherence vs L ----------------------------------------------------
    fits = {}
    for tag in ["Ho", "H"]:
        metrics = [(f"cluster_{tag}.ratio_paper", "ratio_paper (mean r^2 / mean bulk)"),
                   (f"cluster_{tag}.ratio_gap", "ratio_gap (eig[r^2-1]/eig[r^2])"),
                   (f"cluster_{tag}.ratio_split", "ratio_split (data-driven clusters)"),
                   (f"coh_{tag}.coherence", "coherence lam_max/max_l lam_max(B_ll)")]
        fig, axes = plt.subplots(2, 4, figsize=(17, 8))
        for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
            for j, (path, label) in enumerate(metrics):
                ax = axes[i, j]
                steps_here = steps_full if tag == "H" else [s_init, s_mid, s_end]
                for st in steps_here:
                    m, s_, ns, per = agg(kai, setting, "kaiming", st, path, Ls_present)
                    a, b, R2 = linfit(Ls_present, m)
                    fits[f"{tag}|{setting}|{path}|step{st}"] = {"slope": a, "intercept": b, "R2": R2,
                                                                 "means": m.tolist(), "stds": s_.tolist(), "n": ns}
                    ax.errorbar(Ls_present, m, yerr=s_, marker="o", capsize=3, color=ckcol.get(st, "k"),
                                label=f"step {st}: slope={a:.2f}, R2={R2:.2f}")
                    xx = np.array([min(Ls_present), max(Ls_present)])
                    ax.plot(xx, a * xx + b, color=ckcol.get(st, "k"), ls=":", lw=0.8)
                xx = np.array([1, max(Ls_present) + 0.5])
                ax.plot(xx, xx, "k--", lw=1, label="y = L (paper C2)")
                ax.set_title(f"{setting} (r={r},d_out={d_out}): {label}", fontsize=8)
                ax.set_xlabel("L (number of weight matrices)")
                ax.legend(fontsize=6)
                ax.grid(alpha=0.3)
        fig.suptitle(f"{exp}: dominant/bulk ratios and coherence of {'H_o' if tag == 'Ho' else 'sym(H)'} vs depth (mean +- std over seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_ratios_{tag}.png"), dpi=120)
        plt.close(fig)
    analysis["linear_fits"] = fits

    # (c) dominant count vs L ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    counts = {}
    for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
        ax = axes[i]
        for tag, ls in [("Ho", "-"), ("H", "--")]:
            for st in [s_init, s_end]:
                m, s_, ns, per = agg(kai, setting, "kaiming", st, f"dominant_count_{tag}", Ls_present)
                counts[f"{tag}|{setting}|step{st}"] = {"means": m.tolist(), "stds": s_.tolist(), "per_L": per}
                ax.errorbar(np.array(Ls_present) + (0.05 if tag == "H" else -0.05), m, yerr=s_, marker="o", ls=ls,
                            capsize=3, color=ckcol[st], label=f"{'H_o' if tag=='Ho' else 'sym(H)'} step {st}")
        ax.axhline(r * r, color="k", ls="--", lw=1, label=f"r^2 = {r*r}")
        ax.axhline(d_out, color="gray", ls="-.", lw=1, label=f"d_out = {d_out}")
        ax.set_title(f"setting {setting} (r={r}, d_out={d_out}): data-driven dominant count")
        ax.set_xlabel("L"); ax.set_ylabel("split_n_dominant (largest log gap + 1)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"{exp}: number of dominant eigenvalues vs depth (mean +- std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_dominant_count.png"), dpi=120)
    plt.close(fig)
    analysis["dominant_counts"] = counts

    # (d) zero / negative fractions ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fr = {}
    for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
        for j, what in enumerate(["zero_frac", "neg_frac"]):
            ax = axes[j, i]
            for tag, ls in [("Ho", "-"), ("H", "--")]:
                for st in [s_init, s_end]:
                    m, s_, ns, per = agg(kai, setting, "kaiming", st, f"{what}_{tag}", Ls_present)
                    fr[f"{what}|{tag}|{setting}|step{st}"] = {"means": m.tolist(), "stds": s_.tolist()}
                    ax.errorbar(Ls_present, m, yerr=s_, marker="o", ls=ls, capsize=3, color=ckcol[st],
                                label=f"{'H_o' if tag=='Ho' else 'sym(H)'} step {st}")
            if what == "zero_frac":
                P_ = {L: D_IN * WIDTH + WIDTH * WIDTH * (L - 2) + d_out * WIDTH for L in Ls_present}
                paper = [1 - (r * r + r * (D_IN + d_out - 2 * r)) / P_[L] for L in Ls_present]
                ax.plot(Ls_present, paper, "k:", label="paper: 1 - (r^2 + r(d_in+d_out-2r))/P")
            ax.set_title(f"setting {setting}: {'fraction |eig| < 1e-8 max' if what == 'zero_frac' else 'fraction eig < -1e-8 max'}")
            ax.set_xlabel("L"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"{exp}: zero-space and negative fractions vs depth")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_zero_neg_fractions.png"), dpi=120)
    plt.close(fig)
    analysis["fractions"] = fr

    # (e) asymmetry and ||H_f||/||H_o|| vs step --------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    c3 = {}
    for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
        for j, path in enumerate(["asymmetry_H", "Hf_over_Ho"]):
            ax = axes[j, i]
            for L, col in zip(Ls_present, cols):
                ms, ss = [], []
                for st in steps_full:
                    m, s_, ns, per = agg(kai, setting, "kaiming", st, path, [L])
                    ms.append(m[0]); ss.append(s_[0])
                c3[f"{path}|{setting}|L{L}"] = {"steps": steps_full, "means": ms, "stds": ss}
                ax.errorbar(steps_full, ms, yerr=ss, marker="o", capsize=3, color=col, label=f"L={L}")
            ax.set_title(f"setting {setting}: {'asymmetry ||H-H^T||/||H||' if path == 'asymmetry_H' else '||H_f||_F / ||H_o||_F'}")
            ax.set_xlabel("training step"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
            if path == "Hf_over_Ho":
                ax.set_yscale("log")
    fig.suptitle(f"{exp}: C3 test, asymmetry of H and functional part during Adam training")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_asym_Hf_vs_step.png"), dpi=120)
    plt.close(fig)
    analysis["c3"] = c3

    # (f) top-eigenvector layer-energy heatmaps -------------------------------------
    energy_tab = {}
    for setting, (r, d_out) in SETTINGS.items():
        k_req = max(r * r, d_out)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for i, st in enumerate([s_init, s_end]):
            for j, (what, title) in enumerate([("top1", "top-1 eigenvector of H_o"),
                                               ("topk", f"mean over top-{k_req} eigenvectors of H_o")]):
                M = np.full((max(Ls_present), len(Ls_present)), np.nan)
                for c_, L in enumerate(Ls_present):
                    vals = []
                    for r_ in kai:
                        if r_["setting"] != setting or r_["L"] != L:
                            continue
                        c = ckpt(r_, st)
                        if c is None:
                            continue
                        E = np.asarray(c["topk_energy_Ho"])  # (K_TOP, L)
                        vals.append(E[0] if what == "top1" else E[:k_req].mean(0))
                    if vals:
                        M[:L, c_] = np.mean(vals, axis=0)
                energy_tab[f"{setting}|{what}|step{st}"] = M.tolist()
                ax = axes[i, j]
                im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1, origin="lower")
                ax.set_xticks(range(len(Ls_present))); ax.set_xticklabels(Ls_present)
                ax.set_yticks(range(max(Ls_present))); ax.set_yticklabels([f"W{l+1}" for l in range(max(Ls_present))])
                ax.set_xlabel("L"); ax.set_ylabel("layer")
                for c_, L in enumerate(Ls_present):
                    for l in range(L):
                        if np.isfinite(M[l, c_]):
                            ax.text(c_, l, f"{M[l, c_]:.2f}", ha="center", va="center", fontsize=6,
                                    color="w" if M[l, c_] < 0.5 else "k")
                ax.set_title(f"step {st}: {title}", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{exp}: layer energy of top eigenvectors of H_o, setting {setting} (r={r}, d_out={d_out}); W{{L}} = readout")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_eigvec_layer_energy_{setting}.png"), dpi=120)
        plt.close(fig)
    analysis["eigvec_layer_energy"] = energy_tab

    # (g) feature mean vs covariance ------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    feat = {}
    for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
        ax = axes[0, i]
        for path, lab, mk in [("features.mu_sq_norm", "||mu||^2", "o"), ("features.cov_top3.0", "lambda_1(Cov_a)", "s"),
                              ("features.second_moment_top3.0", "lambda_1(M = a^T a/n)", "^"),
                              ("coh_Ho.lambda_max", "lambda_max(H_o)", "d"),
                              ("coh_Ho.lambda_max_blocks.-1", "lambda_max(B_LL)", "x")]:
            for st in [s_init, s_end]:
                if path.endswith(".-1"):
                    m, s_, ns, per = agg(kai, setting, "kaiming", st, "coh_Ho.lambda_max_blocks", Ls_present)
                    vals = []
                    for L in Ls_present:
                        v = [get(ckpt(r_, st), "coh_Ho.lambda_max_blocks")[-1] for r_ in kai
                             if r_["setting"] == setting and r_["L"] == L and ckpt(r_, st) is not None]
                        vals.append((np.mean(v), np.std(v)))
                    m = np.array([v[0] for v in vals]); s_ = np.array([v[1] for v in vals])
                else:
                    m, s_, ns, per = agg(kai, setting, "kaiming", st, path, Ls_present)
                feat[f"{path}|{setting}|step{st}"] = {"means": m.tolist(), "stds": s_.tolist()}
                ax.errorbar(Ls_present, m, yerr=s_, marker=mk, capsize=2, color=ckcol[st], ls="-" if st == s_init else "--",
                            label=f"{lab} step {st}", ms=4)
        ax.set_yscale("log"); ax.set_xlabel("L"); ax.grid(alpha=0.3); ax.legend(fontsize=5, ncol=2, loc="lower left")
        ax.set_title(f"setting {setting}: readout-feature (rate) second moments vs L", fontsize=9)
        ax = axes[1, i]
        for st in [s_init, s_end]:
            m1, _, _, _ = agg(kai, setting, "kaiming", st, "features.mu_sq_norm", Ls_present)
            m2, _, _, _ = agg(kai, setting, "kaiming", st, "features.cov_top3.0", Ls_present)
            m3, s3, _, _ = agg(kai, setting, "kaiming", st, "cluster_Ho.ratio_split", Ls_present)
            m4, s4, _, _ = agg(kai, setting, "kaiming", st, "cluster_Ho.ratio_paper", Ls_present)
            ax.plot(Ls_present, m1 / m2, marker="o", color=ckcol[st], label=f"||mu||^2 / lambda_1(Cov_a) step {st}")
            ax.errorbar(Ls_present, m3, yerr=s3, marker="s", ls="--", color=ckcol[st], capsize=2, label=f"ratio_split(H_o) step {st}")
            ax.errorbar(Ls_present, m4, yerr=s4, marker="^", ls=":", color=ckcol[st], capsize=2, label=f"ratio_paper(H_o) step {st}")
            feat[f"mu_over_cov|{setting}|step{st}"] = (m1 / m2).tolist()
        xx = np.array([1, max(Ls_present) + 0.5]); ax.plot(xx, xx, "k--", lw=1, label="y = L")
        ax.set_xlabel("L"); ax.grid(alpha=0.3); ax.legend(fontsize=6); ax.set_yscale("log")
        ax.set_title(f"setting {setting}: feature-mean ratio vs measured Hessian ratios")
    fig.suptitle(f"{exp}: P2 test, mean-spike of the rate features vs L")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_feature_moments.png"), dpi=120)
    plt.close(fig)
    analysis["features"] = feat

    # (h) loss curves + spike rates --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
        ax = axes[0, i]
        for L, col in zip(Ls_present, cols):
            for r_ in kai:
                if r_["setting"] == setting and r_["L"] == L:
                    z = np.load(os.path.join(res_dir, "raw", r_["name"] + ".npz"))
                    ax.plot(z["loss_curve"], color=col, lw=0.7, alpha=0.8, label=f"L={L}" if r_["seed"] == 0 else None)
            for r_ in runs:
                if r_["init"] == "orth_balanced" and r_["setting"] == setting and r_["L"] == L:
                    z = np.load(os.path.join(res_dir, "raw", r_["name"] + ".npz"))
                    ax.plot(z["loss_curve"], color=col, lw=1.2, ls="--", label=f"L={L} orth-balanced")
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE loss"); ax.legend(fontsize=6, ncol=2)
        ax.set_title(f"setting {setting}: training loss (Adam lr=5e-3; dashed = orth-balanced init)"); ax.grid(alpha=0.3)
        ax = axes[1, i]
        for L, col in zip(Ls_present, cols):
            ms = []
            for st in steps_all:
                m, s_, ns, per = agg(kai, setting, "kaiming", st, "spikes.mean_rate", [L])
                ms.append(m[0])
            ax.plot(steps_all, ms, marker="o", color=col, label=f"L={L}")
        ax.set_xlabel("step"); ax.set_ylabel("mean spike rate (all LIF layers)"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
        ax.set_title(f"setting {setting}: spike rate during training (kaiming, seed mean)")
    fig.suptitle(f"{exp}: loss curves and spike rates")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_loss_spikes.png"), dpi=120)
    plt.close(fig)

    # (i) orthogonal-balanced init comparison ----------------------------------------
    orth = [r_ for r_ in runs if r_["init"] == "orth_balanced"]
    if orth:
        Lo = sorted({r_["L"] for r_ in orth})
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        comp = {}
        for i, (setting, (r, d_out)) in enumerate(SETTINGS.items()):
            for j, (path, lab) in enumerate([("cluster_Ho.ratio_split", "ratio_split(H_o)"),
                                             ("coh_Ho.coherence", "coherence(H_o)"),
                                             ("dominant_count_Ho", "dominant count(H_o)")]):
                ax = axes[i, j]
                for st in [s_init, s_end]:
                    mk, _, _, _ = agg(kai, setting, "kaiming", st, path, Lo)
                    mo, _, _, _ = agg(orth, setting, "orth_balanced", st, path, Lo)
                    comp[f"{path}|{setting}|step{st}"] = {"kaiming": mk.tolist(), "orth_balanced": mo.tolist(), "Ls": Lo}
                    ax.plot(Lo, mk, marker="o", color=ckcol[st], label=f"kaiming step {st}")
                    ax.plot(Lo, mo, marker="s", ls="--", color=ckcol[st], label=f"orth-balanced step {st}")
                if j < 2:
                    xx = np.array([1, max(Lo) + 0.5]); ax.plot(xx, xx, "k--", lw=1, label="y = L")
                else:
                    ax.axhline(r * r, color="k", ls="--", lw=1, label=f"r^2={r*r}"); ax.axhline(d_out, color="gray", ls="-.", lw=1, label=f"d_out={d_out}")
                ax.set_xlabel("L"); ax.set_title(f"setting {setting}: {lab}", fontsize=9); ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.suptitle(f"{exp}: kaiming vs orthogonal-balanced (all singular values equal) init, seed 0")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_orth_balanced.png"), dpi=120)
        plt.close(fig)
        analysis["orth_balanced"] = comp
        analysis["orth_balanced_gains"] = {r_["name"]: {"gain": r_["gain"], "init_mean_rate": r_["init_mean_rate"],
                                                        "final_mean_rate": r_["checkpoints"][-1]["spikes"]["mean_rate"],
                                                        "loss_init": r_["loss_init"], "loss_final": r_["loss_final"]}
                                           for r_ in orth}

    # tables ---------------------------------------------------------------------------
    lines = []
    for setting, (r, d_out) in SETTINGS.items():
        lines.append(f"\n### Setting {setting} (r={r}, r^2={r*r}, d_out={d_out}), SNN kaiming init, mean +- std over seeds\n")
        lines.append("| L | n | step | loss | rate | ratio_paper(Ho) | ratio_gap(Ho) | ratio_split(Ho) | coh(Ho) | ndom(Ho) | ndom(H) | ratio_split(H) | zero%(Ho) | zero%(H) | neg%(H) | asym(H) | |Hf|/|Ho| | E_WL(top1) | mu^2/l1(Cov) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for L in Ls_present:
            for st in [s_init, s_mid, s_end]:
                def ms(path, fmt="{:.2f}"):
                    m, s_, ns, per = agg(kai, setting, "kaiming", st, path, [L])
                    return (fmt.format(m[0]) + ("+-" + fmt.format(s_[0]) if ns[0] > 1 else "")) if ns[0] else "-"
                n_ = agg(kai, setting, "kaiming", st, "loss", [L])[2][0]
                EWL = []
                for r_ in kai:
                    if r_["setting"] == setting and r_["L"] == L and ckpt(r_, st) is not None:
                        EWL.append(np.asarray(ckpt(r_, st)["topk_energy_Ho"])[0, -1])
                m1 = agg(kai, setting, "kaiming", st, "features.mu_sq_norm", [L])[0][0]
                m2 = agg(kai, setting, "kaiming", st, "features.cov_top3.0", [L])[0][0]
                lines.append(f"| {L} | {n_} | {st} | {ms('loss','{:.3f}')} | {ms('spikes.mean_rate','{:.3f}')} | {ms('cluster_Ho.ratio_paper')} | "
                             f"{ms('cluster_Ho.ratio_gap')} | {ms('cluster_Ho.ratio_split')} | {ms('coh_Ho.coherence')} | "
                             f"{ms('dominant_count_Ho','{:.1f}')} | {ms('dominant_count_H','{:.1f}')} | {ms('cluster_H.ratio_split')} | "
                             f"{ms('zero_frac_Ho','{:.3f}')} | {ms('zero_frac_H','{:.3f}')} | {ms('neg_frac_H','{:.3f}')} | "
                             f"{ms('asymmetry_H','{:.3f}')} | {ms('Hf_over_Ho','{:.3f}')} | {np.mean(EWL):.2f} | {m1/m2:.2f} |")
    lines.append("\n### Linear fits metric = slope*L + intercept over L (seed means), H_o\n")
    lines.append("| setting | metric | step | slope | intercept | R^2 | values per L |")
    lines.append("|---|---|---|---|---|---|---|")
    for k_, v in fits.items():
        tag, setting, path, st = k_.split("|")
        if tag != "Ho":
            continue
        lines.append(f"| {setting} | {path.split('.')[-1]} | {st} | {v['slope']:.3f} | {v['intercept']:.3f} | {v['R2']:.3f} | "
                     + ", ".join(f"{x:.2f}" for x in v["means"]) + " |")
    if orth:
        lines.append("\n### Orthogonal-balanced init (seed 0)\n")
        lines.append("| run | gain s | init rate | final rate | loss init->final | ratio_split(Ho) init/end | coh(Ho) init/end | ndom(Ho) init/end |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r_ in orth:
            ci, ce = ckpt(r_, s_init), ckpt(r_, s_end)
            lines.append(f"| {r_['name']} | {r_['gain']:.2f} | {r_['init_mean_rate']:.3f} | {ce['spikes']['mean_rate']:.3f} | "
                         f"{r_['loss_init']:.3f}->{r_['loss_final']:.3f} | {ci['cluster_Ho']['ratio_split']:.2f}/{ce['cluster_Ho']['ratio_split']:.2f} | "
                         f"{ci['coh_Ho']['coherence']:.2f}/{ce['coh_Ho']['coherence']:.2f} | {ci['dominant_count_Ho']}/{ce['dominant_count_Ho']} |")
    table_md = "\n".join(lines)
    analysis["tables_markdown"] = table_md
    with open(os.path.join(res_dir, "analysis.json"), "w") as f:
        json.dump(to_py(analysis), f, indent=1)
    print(table_md)
    return analysis


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures", action="store_true", help="only regenerate figures from saved results")
    args = ap.parse_args()
    exp = EXP + ("_smoke" if args.smoke else "")
    res_dir = os.path.join(ROOT, "results", exp)
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(res_dir, exist_ok=True)
    if args.figures:
        make_figures(res_dir, fig_dir, exp)
        return
    T0 = time.time()
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, exp={exp}", flush=True)
    if args.smoke:
        validate_once("A", 3, 0)
        plan = [("A", 2, 0, "kaiming"), ("A", 3, 0, "kaiming"), ("B", 2, 0, "kaiming"), ("B", 3, 0, "kaiming"),
                ("A", 2, 1, "kaiming"), ("A", 2, 0, "orth_balanced"), ("B", 3, 0, "orth_balanced")]
        steps, ho_every, full_steps = 40, 20, (0, 20, 40)
        tiers = [(0, plan)]
    else:
        steps, ho_every, full_steps = STEPS, HO_EVERY, FULL_H_STEPS
        tier1 = [(s, L, seed, "kaiming") for seed in (0, 1) for s in SETTINGS for L in LS]
        tier1 += [(s, L, 0, "orth_balanced") for s in SETTINGS for L in (2, 4, 8)]
        tier2 = [(s, L, 2, "kaiming") for s in SETTINGS for L in (2, 3, 4, 5)]
        tier3 = [(s, L, 2, "kaiming") for s in SETTINGS for L in (6, 8)]
        tiers = [(0, tier1), (1, tier2), (2, tier3)]
    summary: List[Dict[str, object]] = []
    skipped: List[str] = []
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "budget_min": BUDGET_MIN, "skipped": skipped}
    for tier, plan in tiers:
        for (setting, L, seed, init) in plan:
            elapsed_min = (time.time() - T0) / 60
            est = EST_RUN_SEC[L] / 60
            if tier > 0 and elapsed_min + est > BUDGET_MIN:
                msg = f"{setting}_L{L}_seed{seed}_{init} (tier {tier}; elapsed {elapsed_min:.1f} min + est {est:.1f} > budget {BUDGET_MIN})"
                print("SKIP", msg, flush=True)
                skipped.append(msg)
                continue
            info = run_one(setting, L, seed, init, steps, ho_every, full_steps, res_dir)
            info["tier"] = tier
            summary.append(info)
            log["elapsed_min"] = (time.time() - T0) / 60
            with open(os.path.join(res_dir, "summary.json"), "w") as f:
                json.dump(summary, f)
            with open(os.path.join(res_dir, "run_meta.json"), "w") as f:
                json.dump(log, f, indent=1)
            print(f"[elapsed {log['elapsed_min']:.1f} min, {len(summary)} runs done]", flush=True)
    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log["elapsed_min"] = (time.time() - T0) / 60
    with open(os.path.join(res_dir, "run_meta.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"ALL RUNS DONE in {log['elapsed_min']:.1f} min; skipped: {skipped}", flush=True)
    make_figures(res_dir, fig_dir, exp)
    print(f"FIGURES DONE; total {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
