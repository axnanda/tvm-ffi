"""Print compact markdown tables from results/e1_linear_replication/summary.json (+ posthoc.json)."""
import json
import os
import sys

import numpy as np

ROOT = "/home/user/tvm-ffi/alex-tests/snn-loss-landscapes"
SMOKE = os.environ.get("E1_SMOKE", "0") == "1"
RES = f"{ROOT}/results/e1_linear_replication" + ("/smoke" if SMOKE else "")
S = json.load(open(f"{RES}/summary.json"))
PH = json.load(open(f"{RES}/posthoc.json")) if os.path.exists(f"{RES}/posthoc.json") else {}
by = {d["run"]: d for d in S}


def f(x, n=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{n}f}"


out = []
out.append("**Part 1: balanced_uniform init (s=1, tail=0), H_o and full H at init.** zeros = count below 1e-8*max; split = log-gap (dominant, bulk) counts.")
out.append("| setting | L | P | dom mean (r^2 eigs) | bulk mean (next r(d_in+d_out-2r)) | zeros / expected | split | ratio_paper | coherence | C1(H_o) | C2 (ratio=L, coh=L) | H_f/H_o | asym | sym(H): lam_max, lam_min, #neg, ratio_paper |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in S:
    if d["part"] != "1":
        continue
    c, ch = d["cluster_Ho"], d["cluster_H"]
    out.append(f"| {d['setting']} (r={d['r']},d_out={d['d_out']}) | {d['L']} | {d['P']} | {f(c['dominant_mean'])} | {f(c['bulk_mean'])} | {c['zero_count']} / {d['expected_n_zero']} | ({c['split_n_dominant']},{c['split_n_bulk']}) | {f(c['ratio_paper'])} | {f(d['coherence'])} | {f(d['C1_Ho']['C1_pass'])} | {f(d['C2_ratio_eq_L'])}, {f(d['C2_coherence_eq_L'])} | {f(d['Hf_over_Ho'],3)} | {d['asymmetry']:.1e} | {f(ch['dominant_max'],3)}, {f(ch['most_negative'],3)}, {ch['negative_count']}, {f(ch['ratio_paper'],3)} |")

out.append("")
out.append("**Part 2: non-uniform s_l (L=4, setting T).** cf = closed form sum_l prod_{m!=l} s_m^2 / max(prod_{m!=1}, prod_{m!=L}); ratio_paper uses the bulk MEAN; coherence_cf = dom / max_l prod_{m!=l} s_m^2.")
out.append("| s | dom meas / cf | bulk_in cf | bulk_out cf | bulk mean meas | ratio_paper meas / cf(bulk mean) | dom/bulk_max meas / cf(max) | coherence meas / cf | L | split | eigs match cf |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for d in S:
    if d["part"] != "2":
        continue
    c, cf = d["cluster_Ho"], d["closed_form"]
    out.append(f"| {cf['s']} | {f(c['dominant_mean'])} / {f(cf['dominant_cf'])} | {f(cf['bulk_in_cf'])} | {f(cf['bulk_out_cf'])} | {f(c['bulk_mean'])} | {f(c['ratio_paper'])} / {f(cf['ratio_cf_bulkmean'])} | {f(cf['ratio_measured_over_bulkmax'])} / {f(cf['ratio_cf_max'])} | {f(d['coherence'])} / {f(cf['coherence_cf'])} | {d['L']} | ({c['split_n_dominant']},{c['split_n_bulk']}) | {f(cf['bulk_values_match_cf'] and cf['dominant_match_cf'])} |")

out.append("")
out.append("**Part 3: tail_value t (L=4, setting T).** closed form (post-hoc): dominant = L, bulk = sum_{k<L} t^{2k}, third cluster of (d_in-r)(d_out-r)=28 eigenvalues at L t^{2(L-1)}.")
out.append("| t | dom [min,max] | bulk [min,max] | bulk cf | 3rd cluster mean (cf) | ratio_paper (cf) | ratio_split | split (dom,bulk) | rank | zeros | #>1e-3 max | #>1e-6 max | coherence |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in S:
    if d["part"] != "3":
        continue
    c = d["cluster_Ho"]
    q = PH.get("part3_tail_closed_form", {}).get(d["run"], {})
    out.append(f"| {d['tail_value']} | [{f(c['dominant_min'])},{f(c['dominant_max'])}] | [{f(c['bulk_min'],5)},{f(c['bulk_max'],5)}] | {f(q.get('bulk_cf'),5)} | {f(q.get('third_cluster_measured_mean'))} ({f(q.get('third_cluster_cf'))}) | {f(c['ratio_paper'],5)} ({f(q.get('ratio_cf'),5)}) | {f(c['ratio_split'])} | ({c['split_n_dominant']},{c['split_n_bulk']}) | {c['rank']} | {c['zero_count']} | {d['n_above_1e-3']} | {d['n_above_1e-6']} | {f(d['coherence'])} |")

out.append("")
out.append("**Part 4: GD (lr=0.05) from balanced init, setting T; H_o and full H every 25 steps.** rates in 1/step (fit of log||H_f||_F); theory = ln(1 - lr*lambda_bulk).")
out.append("| L | steps | loss init -> final | H_f/H_o init -> final | fit all: rate, R2, n | fit early: rate, R2 | fit late: rate, R2 | theory rate (bulk) | loss late rate (theory) | ratio_paper [min,max] final | coherence [min,max] final | balancedness max / final | ratio(sym H) final | asym max |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in S:
    if d["part"] != "4":
        continue
    fa, fe, fl, fl2, th = d["C3_fit_logHf_all"], d["C3_fit_logHf_early"], d["C3_fit_logHf_late"], d["C3_fit_logloss_late"], d["C3_theory_rates"]
    out.append(f"| {d['L']} | {d['steps']} | {d['loss_init']:.2e} -> {d['loss_final']:.1e} | {f(d['Hf_over_Ho_init'],3)} -> {d['Hf_over_Ho_final']:.1e} | {f(fa['rate'],5)}, {f(fa['r2'],3)}, {fa['n_points']} | {f(fe['rate'],5)}, {f(fe['r2'],3)} | {f(fl['rate'],5)}, {f(fl['r2'],4)} | {f(th['rate_Hf_bulk'],5)} | {f(fl2['rate'],5)} ({f(th['rate_loss_bulk'],5)}) | [{f(d['ratio_paper_min'])},{f(d['ratio_paper_max'])}] {f(d['ratio_paper_final'])} | [{f(d['coherence_min'])},{f(d['coherence_max'])}] {f(d['coherence_final'])} | {max(d['balancedness_max_over_training']):.1e} / {max(d['balancedness_final']):.1e} | {f(d['ratio_H_final'])} | {d['asymmetry_max']:.1e} |")
if "part4_ratio_excursion" in PH:
    out.append("")
    out.append("Post-hoc (part 4): ratio_paper predicted from the end-to-end singular values under exact balancedness vs measured:")
    out.append("| L | measured ratio range | predicted range | max abs diff | max rel err top-36 eigs | sigma_2 min |")
    out.append("|---|---|---|---|---|---|")
    for run, rows in PH["part4_ratio_excursion"].items():
        L = by[run]["L"]
        mr = [q["ratio_measured"] for q in rows]; pr = [q["ratio_predicted_balanced"] for q in rows]
        out.append(f"| {L} | [{min(mr):.4f},{max(mr):.4f}] | [{min(pr):.4f},{max(pr):.4f}] | {max(abs(a-b) for a,b in zip(mr,pr)):.4f} | {max(q['max_rel_err_top36'] for q in rows):.2e} | {min(q['sigma'][1] for q in rows):.3f} |")

out.append("")
out.append("**Part 4b (extra): GD reference numbers (lr=0.05, 1000 steps, checkpoints every 100) for T/A/B at L in {2,4,8}.**")
out.append("| setting | L | loss final | H_f/H_o final | ratio_paper [min,max] final | coherence [min,max] final | balancedness max | fit rate (all), R2 | C5 (ratio, coh within 0.1% of L all along) |")
out.append("|---|---|---|---|---|---|---|---|---|")
for d in S:
    if d["part"] != "4b":
        continue
    fa = d["C3_fit_logHf_all"]
    out.append(f"| {d['setting']} | {d['L']} | {d['loss_final']:.1e} | {d['Hf_over_Ho_final']:.1e} | [{f(d['ratio_paper_min'])},{f(d['ratio_paper_max'])}] {f(d['ratio_paper_final'])} | [{f(d['coherence_min'])},{f(d['coherence_max'])}] {f(d['coherence_final'])} | {max(d['balancedness_max_over_training']):.1e} | {f(fa['rate'],5)}, {f(fa['r2'],3)} | {f(d['C5_ratio_stays_L'])}, {f(d['C5_coherence_stays_L'])} |")

out.append("")
out.append("**Part 5: gaussian init (gain 1), setting T, lr = min(0.05, 1/lambda_max(H_init)), GD until loss<1e-6 (max 15000; state at 3000 also shown).** rank(H_o) = 36 for balanced init.")
out.append("| L | seed | lr | stop (step) | loss @3000 / final | ratio_paper init / @3000 / final | ratio_gap init / final | log-gap dom count init / @3000 / final | ratio_split init / final | coherence init / @3000 / final | rank(H_o) init / final | balancedness mean init / final | top-eigvec layer energy final | H_f/H_o final |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in S:
    if d["part"] != "5":
        continue
    a, b, c3 = d["at_init"], d["at_end"], d["at_3000"]
    out.append(f"| {d['L']} | {d['seed']} | {d['lr']:.4f} | {d['stopped_reason']} ({d['steps']}) | {d['loss_at_3000']:.1e} / {d['loss_final']:.1e} | {f(a['ratio_paper'],3)} / {f(c3['ratio_paper'],3)} / {f(b['ratio_paper'],3)} | {f(a['ratio_gap'],3)} / {f(b['ratio_gap'],3)} | {a['split_n_dominant']} / {c3['split_n_dominant']} / {b['split_n_dominant']} | {f(a['ratio_split'],3)} / {f(b['ratio_split'],3)} | {f(a['coherence'],3)} / {f(c3['coherence'],3)} / {f(b['coherence'],3)} | {a['rank']} / {b['rank']} | {np.mean(a['balancedness']):.3f} / {np.mean(b['balancedness']):.3f} | {[round(x,2) for x in b['top_eigvec_layer_energy']]} | {b['Hf_over_Ho']:.1e} |")

txt = "\n".join(out)
print(txt)
open(f"{RES}/tables.md", "w").write(txt + "\n")
