"""Cross-dataset generalization: evaluate LEVIR-trained models on other datasets'
test splits (OSCD, SYSU). Also re-runs in-domain LEVIR for a complete row.
Skips a dataset whose test npy is absent."""
import os, sys, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MODELS = ["lightcd_large", "lightcd", "fc_siam_diff"]
DATASETS = ["levir", "egy", "sysu", "oscd"]


def has_test(ds):
    return os.path.exists(os.path.join(DATA_ROOT, ds, "test_A.npy"))


def main():
    for model in MODELS:
        ckpt = os.path.join(ROOT, "models", f"{model}_levir.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] no ckpt for {model}"); continue
        for ds in DATASETS:
            if not has_test(ds):
                print(f"[skip] no test npy for {ds}"); continue
            cmd = [PY, "-u", "code/eval.py", "--model", model, "--ckpt", ckpt,
                   "--dataset", ds, "--split", "test", "--workers", "0",
                   "--tag", f"{model}_levir"]
            print("[cross]", model, "->", ds, flush=True)
            subprocess.run(cmd, cwd=ROOT)
    print("CROSS_EVAL_DONE")


if __name__ == "__main__":
    main()
