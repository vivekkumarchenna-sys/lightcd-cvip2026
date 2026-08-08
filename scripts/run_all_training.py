"""Orchestrate training + in-domain test evaluation for all models.

Runs each model in a fresh subprocess (clean GPU memory). Resumable: skips a
model whose test-metrics json already exists. Logs to logs/<run>.log.
"""
import os, sys, json, subprocess, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
LOGS = os.path.join(ROOT, "logs")
os.makedirs(LOGS, exist_ok=True)

# (model, epochs, lr, batch_size, pretrained)
# ordered by priority: core comparison first, ablations last
CONFIG = [
    ("lightcd",        40, 3e-4, 16, True),   # our ultra-light model
    ("lightcd_large",  40, 3e-4, 16, True),   # our higher-accuracy variant (MobileNetV3-Large)
    ("lightcd_hr",     40, 3e-4, 16, True),   # negative variant (output_stride=16)
    ("fc_siam_diff",   35, 1e-3, 16, True),   # key classic baseline
    ("resnet18_siam",  30, 3e-4, 16, True),   # heavier backbone baseline
    ("fc_ef",          35, 1e-3, 16, True),
    ("fc_siam_conc",   35, 1e-3, 16, True),
    # ablations of ours
    ("lightcd_diff",   28, 3e-4, 16, True),
    ("lightcd_concat", 28, 3e-4, 16, True),
    ("lightcd_noca",   28, 3e-4, 16, True),
    ("lightcd_nosa",   28, 3e-4, 16, True),
]

DATASET = "levir"
WORKERS = 0  # 0 = no spawned workers -> memory-safe on this 16GB box; GPU-normalize keeps it fast


def already_done(run, dataset):
    tj = os.path.join(ROOT, "results", run, f"test_{dataset}_test.json")
    return os.path.exists(tj)


def run(cmd, logpath):
    with open(logpath, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== {' '.join(cmd)} @ {time.strftime('%H:%M:%S')} ===\n")
        lf.flush()
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
        return p.returncode


def main():
    for model, epochs, lr, bs, pre in CONFIG:
        run_name = f"{model}_{DATASET}"
        logpath = os.path.join(LOGS, f"{run_name}.log")
        if already_done(run_name, DATASET):
            print(f"[skip] {run_name} already has test metrics", flush=True)
            continue
        print(f"[train] {run_name} epochs={epochs} lr={lr} bs={bs}", flush=True)
        t0 = time.time()
        train_cmd = [PY, "-u", "code/train.py", "--model", model, "--dataset", DATASET,
                     "--epochs", str(epochs), "--lr", str(lr), "--batch_size", str(bs),
                     "--workers", str(WORKERS), "--run", run_name]
        if not pre:
            train_cmd.append("--no_pretrained")
        rc = run(train_cmd, logpath)
        if rc != 0:
            print(f"[ERROR] train {run_name} rc={rc}; see {logpath}", flush=True)
            continue
        ckpt = os.path.join(ROOT, "models", f"{run_name}.pt")
        eval_cmd = [PY, "-u", "code/eval.py", "--model", model, "--ckpt", ckpt,
                    "--dataset", DATASET, "--split", "test", "--workers", str(WORKERS),
                    "--tag", run_name]
        rc = run(eval_cmd, logpath)
        dt = (time.time() - t0) / 60
        print(f"[done] {run_name} in {dt:.1f} min (eval rc={rc})", flush=True)
    print("ALL_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
