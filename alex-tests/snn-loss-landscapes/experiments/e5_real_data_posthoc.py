import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import snnland as sl; sl.set_threads(1)  # noqa: E401,E702
"""e5_real_data_posthoc: checks on the saved e5 data (no new training).

1. e7 prediction on real data: for the SNN the raw update-field Jacobian H is block upper-triangular, so
   eig(H) = {eig(Sigma_a)} x d_out  U  eig(hidden diagonal blocks).  Compare the top real eigenvalues of raw H
   with d_out=10 copies of eig(M), M = a^T a / n the readout-feature second moment (stored as M_eigs).
2. Where do the top-10 / 10..100 eigenvectors of H_o live (layer energy), and how does the H_o spectrum of the
   SNN relate to the readout block I_10 (x) M alone (10 copies of eig(M)) -- i.e. is the dominant structure the
   feature-mean / class-mean (data) mechanism re-created inside the network (theory notes sec. 3 / P2)?
3. Prints compact tables; writes results/e5_real_data/posthoc.json.
"""
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "e5_real_data")
rows = json.load(open(os.path.join(RES, "summary.json")))
end_step = max(c["step"] for r in rows for c in r["checkpoints"])
out = []
print(f"### SNN raw eig(H) vs 10 x eig(M) (readout block), and H_o top eigenvalues vs eig(M)  (end = step {end_step})")
print("| run | step | lam1(M) | raw H re top1..3 | #raw eig within 5% of lam1(M) (expect 10) | #raw within 5% of lam2(M) (expect 10) | max|imag| | H_o lam1..3 | H_o lam1/lam1(M) | H_o lam10/lam11 | lam1(M)/lam2(M) | E_last top10 | E_last 10..100 |")
print("|" + "---|" * 13)
for r in rows:
    if r["kind"] != "snn":
        continue
    rw = np.load(os.path.join(RES, "raw", f"{r['name']}.npz"))
    for step in (0, end_step):
        c = [c_ for c_ in r["checkpoints"] if c_["step"] == step][0]
        M = np.asarray(c["class_features"]["M_eigs"])
        key = f"eig_H_raw_re_step{step}"
        if key not in rw:
            continue
        re = rw[key]
        im = rw[f"eig_H_raw_im_step{step}"]
        n1 = int((np.abs(re - M[0]) < 0.05 * M[0]).sum())
        n2 = int((np.abs(re - M[1]) < 0.05 * M[1]).sum())
        eo = rw[f"eig_Ho_step{step}"]
        E = rw[f"topk_energy_Ho_step{step}"]
        rec = {"name": r["name"], "step": step, "lam1_M": float(M[0]), "lam2_M": float(M[1]), "rawH_top3": [float(x) for x in re[:3]],
               "n_raw_near_lam1_M": n1, "n_raw_near_lam2_M": n2, "max_abs_imag": float(np.abs(im).max()),
               "Ho_top3": [float(x) for x in eo[:3]], "Ho_lam1_over_lam1_M": float(eo[0] / M[0]), "Ho_gap10": float(eo[9] / eo[10]),
               "lam1_M_over_lam2_M": float(M[0] / M[1]), "E_last_top10": float(E[:10, -1].mean()), "E_last_10_100": float(E[10:100, -1].mean())}
        out.append(rec)
        print(f"| {r['name']} | {step} | {M[0]:.3f} | {re[0]:.3f}, {re[1]:.3f}, {re[2]:.3f} | {n1} | {n2} | {rec['max_abs_imag']:.1e} | "
              f"{eo[0]:.3f}, {eo[1]:.3f}, {eo[2]:.3f} | {rec['Ho_lam1_over_lam1_M']:.2f} | {rec['Ho_gap10']:.3f} | {rec['lam1_M_over_lam2_M']:.2f} | "
              f"{rec['E_last_top10']:.2f} | {rec['E_last_10_100']:.2f} |")

print("\n### all kinds: where the top eigenvectors of H_o live at the end (mean over seeds): last-layer energy of eigvecs 1-10 / 11-100 / 101-200; mean over hidden layers")
print("| loss | kind | L | E_last(1-10) | E_last(11-100) | E_first(1-10) | E_first(11-100) | lam1/lam1(M) | lam10/lam11 | lam100/lam101 |")
print("|" + "---|" * 10)
for loss in ("mse", "ce"):
    for kind in ("relu", "linear", "snn"):
        for L in (2, 3, 4, 6):
            g = [r for r in rows if r["kind"] == kind and r["L"] == L and r["loss"] == loss]
            if not g:
                continue
            vals = []
            for r in g:
                rw = np.load(os.path.join(RES, "raw", f"{r['name']}.npz"))
                E = rw[f"topk_energy_Ho_step{end_step}"]
                eo = rw[f"eig_Ho_step{end_step}"]
                c = [c_ for c_ in r["checkpoints"] if c_["step"] == end_step][0]
                M = np.asarray(c["class_features"]["M_eigs"])
                vals.append([E[:10, -1].mean(), E[10:100, -1].mean(), E[:10, 0].mean(), E[10:100, 0].mean(), eo[0] / M[0], eo[9] / eo[10], eo[99] / eo[100]])
            v = np.mean(vals, 0)
            print(f"| {loss} | {kind} | {L} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
json.dump(out, open(os.path.join(RES, "posthoc.json"), "w"), indent=1)
