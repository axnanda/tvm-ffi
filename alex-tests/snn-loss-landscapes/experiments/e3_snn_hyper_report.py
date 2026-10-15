"""Build results/e3_snn_hyper/report.json from summary.json / analysis.json / run_meta.json.

Usage: PYTHONPATH=. python3 experiments/e3_snn_hyper_report.py            (after e3_snn_hyper.py --figures)
The claim verdicts / anomalies below are filled in by hand from the analysis output.
"""
import json, os, glob, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "e3_snn_hyper"
RES = os.path.join(ROOT, "results", EXP)

runs = json.load(open(os.path.join(RES, "summary.json")))
analysis = json.load(open(os.path.join(RES, "analysis.json")))
meta = json.load(open(os.path.join(RES, "run_meta.json")))
verdicts = json.load(open(os.path.join(RES, "verdicts.json")))  # hand-written: claim_verdicts, anomalies, notes

report = {
    "exp": EXP,
    "script": "experiments/e3_snn_hyper.py (figures/tables regenerated with --figures; report via experiments/e3_snn_hyper_report.py)",
    "result_files": [os.path.relpath(os.path.join(RES, f), ROOT) for f in ["summary.json", "analysis.json", "run_meta.json", "run.log", "verdicts.json", "report.json"]]
                    + [f"results/{EXP}/raw/<run_name>.npz ({len(glob.glob(os.path.join(RES, 'raw', '*.npz')))} files: eig_Ho_step*, eig_symH_step*, eigH_raw_re_step*, eigH_raw_im_step*, loss_curve, grad_norm_curve, Wstar; run_name = <tag>_L<L>_seed<seed> as in summary.json)"],
    "figures": sorted([os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "figures", EXP + "_*.png")) if "_smoke_" not in p]),
    "n_runs": len(runs),
    "n_runs_planned": meta.get("n_runs_planned"),
    "conditions": {"base": meta["base"], "sweeps": meta["sweeps"], "depth_grid": meta["depth_grid"], "seeds": meta["seeds"],
                   "steps": meta["steps"], "lr": meta["lr"], "n": 256, "d_in": 16, "width": 16, "setting": "B (r=2, d_out=6)"},
    "key_tables_markdown": analysis["tables_markdown"],
    "k_loglog_fits": {k: {kk: vv for kk, vv in v.items() if kk in ("exponent", "R2")} for k, v in analysis["k_loglog_fits"].items()},
    "block_loglog_fits": analysis.get("block_loglog_fits", {}),
    "coherence_vs_L_slopes": {k: {"slope": v["slope"], "intercept": v["intercept"]} for k, v in analysis.get("coherence_vs_L_slopes", {}).items()},
    "structure_checks": analysis["structure_checks"],
    "claim_verdicts": verdicts["claim_verdicts"],
    "anomalies": verdicts["anomalies"],
    "reductions": meta.get("reductions", []) + verdicts.get("reductions_extra", []),
    "wall_minutes": meta.get("wall_minutes_total", meta.get("elapsed_min")),
    "notes": verdicts.get("notes", []),
}
with open(os.path.join(RES, "report.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
