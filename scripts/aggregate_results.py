"""Aggregate per-run results (val summary + test metrics) with efficiency stats
into a single CSV + Markdown table for the paper."""
import os, json, glob, csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    eff = load_json(os.path.join(RES, "efficiency.json")) or {"rows": []}
    eff_by_model = {r["model"]: r for r in eff.get("rows", [])}

    rows = []
    for d in sorted(glob.glob(os.path.join(RES, "*"))):
        if not os.path.isdir(d):
            continue
        run = os.path.basename(d)
        if run.startswith("_"):
            continue
        summ = load_json(os.path.join(d, "summary.json"))
        if not summ:
            continue
        model = summ.get("model", run)
        # prefer in-domain test metrics if present
        test = load_json(os.path.join(d, f"test_{summ.get('dataset','levir')}_test.json"))
        e = eff_by_model.get(model, {})
        row = {
            "run": run,
            "model": model,
            "dataset": summ.get("dataset"),
            "params_M": e.get("params_M", round(summ.get("n_params", 0) / 1e6, 3)),
            "gmacs": e.get("gmacs"),
            "latency_ms": e.get("latency_ms_bs1"),
            "fps": e.get("throughput_imgs_s"),
            "val_f1": round(summ.get("best_val_f1", 0), 4),
        }
        if test:
            row.update({
                "test_f1": round(test["f1"], 4),
                "test_iou": round(test["iou"], 4),
                "test_prec": round(test["precision"], 4),
                "test_rec": round(test["recall"], 4),
                "test_oa": round(test["oa"], 4),
                "test_kappa": round(test["kappa"], 4),
            })
        rows.append(row)

    if not rows:
        print("no runs found yet")
        return

    keys = ["model", "dataset", "params_M", "gmacs", "latency_ms", "fps",
            "val_f1", "test_f1", "test_iou", "test_prec", "test_rec", "test_oa", "test_kappa"]
    csv_path = os.path.join(RES, "main_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # markdown
    md = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in sorted(rows, key=lambda x: x.get("test_f1") or x.get("val_f1") or 0):
        md.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    md_str = "\n".join(md)
    with open(os.path.join(RES, "main_table.md"), "w") as f:
        f.write(md_str + "\n")
    print(md_str)
    print("\nwrote", csv_path)


if __name__ == "__main__":
    main()
