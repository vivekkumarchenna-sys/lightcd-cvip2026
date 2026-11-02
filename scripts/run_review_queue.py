"""Runs (CVIP 2026): every model on LEVIR-CD for the same 40 epochs
over several seeds. Resumable and parallel: re-reads scripts/review_jobs.txt and
scripts/review_slots.txt, so jobs and the slot count can change while it runs.

Job line: <model> <seed> <lr>
Results:  results/r2/<model>_s<seed>/   Checkpoints: $LIGHTCD_MODELS, default models/.
"""
import os, sys, time, subprocess, threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
JOBS = os.path.join(ROOT, "scripts", "review_jobs.txt")
SLOTS = os.path.join(ROOT, "scripts", "review_slots.txt")
LOGS = os.path.join(ROOT, "logs", "r2")
MODELS = os.environ.get("LIGHTCD_MODELS", os.path.join(ROOT, "models"))
EPOCHS, BS, MAX_TRIES = 40, 16, 10
os.makedirs(LOGS, exist_ok=True)
lock = threading.Lock()
running, tries, active = set(), {}, []


def done(m, s):
    return os.path.exists(os.path.join(ROOT, "results", "r2", f"{m}_s{s}", "test_levir_test.json"))


def jobs():
    out = []
    for line in open(JOBS, encoding="utf-8"):
        p = line.split()
        if len(p) >= 3 and not line.startswith("#"):
            out.append((p[0], p[1], p[2]))
    return out


def slots():
    try:
        return int(open(SLOTS).read().strip())
    except Exception:
        return 3


def work(m, s, lr):
    run = f"r2/{m}_s{s}"
    log = os.path.join(LOGS, f"{m}_s{s}.log")
    ckpt = os.path.join(MODELS, f"{run}.pt")
    train = [PY, "-u", "code/train.py", "--model", m, "--dataset", "levir", "--epochs", str(EPOCHS),
             "--lr", lr, "--batch_size", str(BS), "--workers", "0", "--seed", s, "--run", run,
             "--models_dir", MODELS]
    ev = [PY, "-u", "code/eval.py", "--model", m, "--ckpt", ckpt, "--dataset", "levir",
          "--split", "test", "--workers", "0", "--tag", run]
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        rc = subprocess.run(train, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        if rc == 0:
            rc = subprocess.run(ev, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    print(f"[{time.strftime('%H:%M')}] {m} s{s} rc={rc} {(time.time()-t0)/60:.1f} min", flush=True)
    with lock:
        running.discard((m, s))


def main():
    print("QUEUE_START", flush=True)
    while True:
        with lock:
            active[:] = [t for t in active if t.is_alive()]
            pending = [j for j in jobs() if not done(j[0], j[1]) and (j[0], j[1]) not in running
                       and tries.get(j[:2], 0) < MAX_TRIES]
            free = slots() - len(active)
            for m, s, lr in pending[:max(free, 0)]:
                running.add((m, s)); tries[(m, s)] = tries.get((m, s), 0) + 1
                t = threading.Thread(target=work, args=(m, s, lr)); t.start(); active.append(t)
            idle = not active and not pending
        if idle:
            break
        time.sleep(20)
    print("QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
