import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""Compact tables for e2b_controls_depth (init / end values per kind, setting, L; seed mean +- std).
Usage: python3 experiments/e2b_controls_depth_tables.py [results_dir]   (default results/e2b_controls_depth)
Writes <results_dir>/tables.md and prints it."""
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2b_controls_depth import SETTINGS, LS, KINDS, agg, sel, ckpt, get, load_snn_reference, linfit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "e2b_controls_depth")
runs = json.load(open(os.path.join(res_dir, "summary.json")))
snn = load_snn_reference()
main = [r for r in runs if not r.get("centered", False)]
kinds = [k for k in KINDS if any(r["kind"] == k for r in main)]
Ls = sorted({r["L"] for r in main})
steps_full = sorted({c["step"] for r in main for c in r["checkpoints"] if "cluster_H" in c})
s0, s1 = steps_full[0], steps_full[-1]
lines = []


def f(m, s, n, fmt="{:.2f}"):
    if n == 0 or not np.isfinite(m):
        return "-"
    return fmt.format(m) + (("+-" + fmt.format(s)) if n > 1 else "")


def cell(rs, kind, setting, L, path, fmt="{:.2f}"):
    a0 = agg(rs, kind, setting, s0, path, [L]); a1 = agg(rs, kind, setting, s1, path, [L])
    return f"{f(a0[0][0], a0[1][0], a0[2][0], fmt)} / {f(a1[0][0], a1[1][0], a1[2][0], fmt)}"


def ewl(rs, kind, setting, L, st):
    v = [np.asarray(ckpt(r, st)["topk_energy_Ho"])[0, -1] for r in sel(rs, kind, setting, L) if ckpt(r, st) is not None and "topk_energy_Ho" in ckpt(r, st)]
    return np.mean(v) if v else np.nan


allk = kinds + (["snn"] if snn else [])
allr = runs + snn
for kind in allk:
    for setting, (r, d_out) in SETTINGS.items():
        n_bulk = r * (16 + d_out - 2 * r)
        lines.append(f"\n**{kind}, setting {setting} (r={r}, r^2={r*r}, d_out={d_out}, paper bulk count {n_bulk}); values init / step {s1}; mean+-std over seeds**\n")
        lines.append("| L | n | loss end | act | ratio_paper(Ho) | ratio_split(Ho) | coh(Ho) | ndom(Ho) | ndom(symH) | ratio_split(symH) | zero(Ho) | zero(H) | neg(H) | Hf/Ho | E_WL(top1) | mu^2/l1Cov |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for L in Ls:
            n = agg(allr, kind, setting, s0, "loss", [L])[2][0]
            if n == 0:
                continue
            act = "spikes.mean_rate" if kind in ("smooth_snn", "snn") else "activity.mean_active_frac"
            le = agg(allr, kind, setting, s1, "loss", [L])
            m1 = agg(allr, kind, setting, s0, "features.mu_sq_norm", [L])[0][0]; m2 = agg(allr, kind, setting, s0, "features.cov_top3.0", [L])[0][0]
            m3 = agg(allr, kind, setting, s1, "features.mu_sq_norm", [L])[0][0]; m4 = agg(allr, kind, setting, s1, "features.cov_top3.0", [L])[0][0]
            lines.append(f"| {L} | {n} | {f(le[0][0], le[1][0], le[2][0], '{:.1e}')} | {cell(allr, kind, setting, L, act)} | {cell(allr, kind, setting, L, 'cluster_Ho.ratio_paper')} | "
                         f"{cell(allr, kind, setting, L, 'cluster_Ho.ratio_split')} | {cell(allr, kind, setting, L, 'coh_Ho.coherence')} | {cell(allr, kind, setting, L, 'dominant_count_Ho', '{:.1f}')} | "
                         f"{cell(allr, kind, setting, L, 'dominant_count_H', '{:.1f}')} | {cell(allr, kind, setting, L, 'cluster_H.ratio_split')} | {cell(allr, kind, setting, L, 'zero_frac_Ho', '{:.3f}')} | "
                         f"{cell(allr, kind, setting, L, 'zero_frac_H', '{:.3f}')} | {cell(allr, kind, setting, L, 'neg_frac_H', '{:.3f}')} | {cell(allr, kind, setting, L, 'Hf_over_Ho', '{:.3f}')} | "
                         f"{ewl(allr, kind, setting, L, s0):.2f} / {ewl(allr, kind, setting, L, s1):.2f} | {m1/m2:.2f} / {m3/m4:.2f} |")

lines.append(f"\n**Linear fits y = slope*L + b over L (seed means), H_o; paper C2 predicts slope 1, intercept 0**\n")
lines.append("| kind | setting | metric | slope init | R2 init | slope end | R2 end | values init | values end |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for kind in allk:
    for setting in SETTINGS:
        for path, lab in [("cluster_Ho.ratio_paper", "ratio_paper"), ("cluster_Ho.ratio_split", "ratio_split"), ("coh_Ho.coherence", "coherence")]:
            m0 = agg(allr, kind, setting, s0, path, Ls)[0]; m1 = agg(allr, kind, setting, s1, path, Ls)[0]
            a0, b0, R0 = linfit(Ls, m0); a1, b1, R1 = linfit(Ls, m1)
            lines.append(f"| {kind} | {setting} | {lab} | {a0:.3f} | {R0:.2f} | {a1:.3f} | {R1:.2f} | {', '.join(f'{x:.2f}' for x in m0)} | {', '.join(f'{x:.2f}' for x in m1)} |")

cent = [r for r in runs if r.get("centered", False)]
if cent:
    lines.append("\n**Centring experiment (setting B, seed 0): uncentred -> centred readout features; H_o unless noted**\n")
    lines.append("| kind | L | step | ndom(Ho) | ratio_split(Ho) | coh(Ho) | lam_max(Ho) | lam_max(B_LL) | E_WL(top1) | mu^2/l1Cov(unc feats) | ndom(symH_lib) | ndom(symH_true) | asym(H_true) | Hf_true/Ho | loss |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for kind in ("relu", "smooth_snn"):
        for L in sorted({r["L"] for r in cent if r["kind"] == kind}):
            for st in (s0, s1):
                row = []
                for centered in (False, True):
                    rr = [r for r in sel(runs, kind, "B", L, centered) if r["seed"] == 0]
                    if not rr:
                        row.append(None); continue
                    c = ckpt(rr[0], st); row.append(c)
                u, c = row
                if u is None or c is None:
                    continue
                def pair(path, fmt="{:.2f}"):
                    a, b = get(u, path), get(c, path)
                    return f"{fmt.format(a) if a is not None and np.isfinite(a) else '-'} -> {fmt.format(b) if b is not None and np.isfinite(b) else '-'}"
                mu_u = u["features"]["mu_sq_norm"] / u["features"]["cov_top3"][0]; mu_c = c["features"]["mu_sq_norm"] / c["features"]["cov_top3"][0]
                lines.append(f"| {kind} | {L} | {st} | {pair('dominant_count_Ho', '{:.0f}')} | {pair('cluster_Ho.ratio_split')} | {pair('coh_Ho.coherence')} | {pair('coh_Ho.lambda_max', '{:.3f}')} | "
                             f"{pair('coh_Ho.lambda_max_blocks.-1', '{:.3f}')} | {np.asarray(u['topk_energy_Ho'])[0, -1]:.2f} -> {np.asarray(c['topk_energy_Ho'])[0, -1]:.2f} | {mu_u:.2f} -> {mu_c:.2f} | "
                             f"{pair('dominant_count_H', '{:.0f}')} | {get(c, 'dominant_count_Htrue')} | {get(c, 'asymmetry_Htrue'):.3f} | {get(c, 'Hf_true_over_Ho'):.3f} | {u['loss']:.1e} -> {c['loss']:.1e} |")

# e2a comparison numbers in one line per kind
lines.append("\n**Kind comparison, H_o at init / end, seed means over L (ratio_split, coherence, ndom)**\n")
lines.append("| kind | setting | ratio_split init | ratio_split end | coh init | coh end | ndom init | ndom end |")
lines.append("|---|---|---|---|---|---|---|---|")
for kind in allk:
    for setting in SETTINGS:
        vals = []
        for path in ["cluster_Ho.ratio_split", "coh_Ho.coherence", "dominant_count_Ho"]:
            for st in (s0, s1):
                vals.append(", ".join(f"{x:.1f}" for x in agg(allr, kind, setting, st, path, Ls)[0]))
        lines.append(f"| {kind} | {setting} | " + " | ".join(vals) + " |")

md = "\n".join(lines)
open(os.path.join(res_dir, "tables.md"), "w").write(md)
print(md)
