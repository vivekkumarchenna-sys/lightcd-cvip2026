"""Collect the runs into results/r2/summary.json: mean, sample standard
deviation and seed count for every model, on the LEVIR-CD test set."""
import os, json, glob, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R2 = os.path.join(ROOT, "results", "r2")
KEYS = ("precision", "recall", "f1", "iou", "oa", "kappa")
runs = {}
for d in glob.glob(os.path.join(R2, "*_s*")):
    tj = os.path.join(d, "test_levir_test.json")
    if not os.path.exists(tj):
        continue
    name, seed = os.path.basename(d).rsplit("_s", 1)
    runs.setdefault(name, {})[seed] = json.load(open(tj))
out = {}
for m, sd in sorted(runs.items()):
    row = {"seeds": sorted(sd), "n": len(sd)}
    for k in KEYS:
        v = [100 * sd[s][k] for s in sd]
        row[k] = st.mean(v); row[k + "_std"] = st.stdev(v) if len(v) > 1 else 0.0
    h = os.path.join(R2, f"{m}_s{sorted(sd)[0]}", "summary.json")
    row["best_epoch_first_seed"] = json.load(open(h))["best_val"]["epoch"] if os.path.exists(h) else None
    out[m] = row
json.dump(out, open(os.path.join(R2, "summary.json"), "w"), indent=2)
for m, r in out.items():
    print(f"{m:16s} n={r['n']} F1 {r['f1']:.2f}+-{r['f1_std']:.2f} IoU {r['iou']:.2f} P {r['precision']:.2f} R {r['recall']:.2f} k {r['kappa']:.2f}")
