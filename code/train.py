"""Training harness for change-detection models (mixed precision, 8GB-friendly).

Selects the best checkpoint by validation F1. Logs per-epoch metrics to
results/<run>/ and saves the best weights to models/<run>.pt.
"""
import os, sys, json, time, argparse, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ChangeDetectionDataset, normalize_gpu
from models import build_model
from losses import BCEDiceLoss
from metrics import CDMetrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def loaders(dataset, batch_size, workers, size=256):
    tr = ChangeDetectionDataset(dataset, "train", train_aug=True, size=size)
    va = ChangeDetectionDataset(dataset, "val", train_aug=False, size=size)
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(va, batch_size=batch_size, shuffle=False, **common),
    )


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    met = CDMetrics()
    tot_loss, n = 0.0, 0
    for batch in loader:
        a = normalize_gpu(batch["A"].to(device, non_blocking=True))
        b = normalize_gpu(batch["B"].to(device, non_blocking=True))
        m = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(a, b)
            if loss_fn is not None:
                tot_loss += float(loss_fn(logits, m)); n += 1
        met.update(logits.float(), m)
    res = met.compute()
    if loss_fn is not None:
        res["loss"] = tot_loss / max(n, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="levir")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", default=None)
    ap.add_argument("--no_pretrained", action="store_true")
    args = ap.parse_args()

    run = args.run or f"{args.model}_{args.dataset}"
    outdir = os.path.join(ROOT, "results", run)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = loaders(args.dataset, args.batch_size, args.workers)
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loss_fn = BCEDiceLoss(pos_weight=args.pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = []
    best_f1 = -1.0
    best_metrics = None
    ckpt_path = os.path.join(ROOT, "models", f"{run}.pt")
    print(f"[{run}] params={n_params/1e6:.3f}M device={device} epochs={args.epochs} bs={args.batch_size}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss, iters = 0.0, 0
        for batch in train_loader:
            a = normalize_gpu(batch["A"].to(device, non_blocking=True))
            b = normalize_gpu(batch["B"].to(device, non_blocking=True))
            m = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(a, b)
                loss = loss_fn(logits, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss); iters += 1
        sched.step()
        val = evaluate(model, val_loader, device, loss_fn)
        dt = time.time() - t0
        rec = {"epoch": epoch, "train_loss": run_loss / max(iters, 1),
               "val_f1": val["f1"], "val_iou": val["iou"], "val_prec": val["precision"],
               "val_rec": val["recall"], "val_loss": val.get("loss"), "sec": round(dt, 1)}
        history.append(rec)
        print(f"  ep{epoch:02d} loss={rec['train_loss']:.4f} valF1={val['f1']:.4f} "
              f"valIoU={val['iou']:.4f} P={val['precision']:.3f} R={val['recall']:.3f} {dt:.0f}s")
        if val["f1"] > best_f1:
            best_f1 = val["f1"]; best_metrics = {**val, "epoch": epoch}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val": best_metrics, "n_params": n_params}, ckpt_path)
        with open(os.path.join(outdir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    summary = {"run": run, "model": args.model, "dataset": args.dataset,
               "n_params": n_params, "best_val_f1": best_f1, "best_val": best_metrics,
               "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{run}] BEST val F1={best_f1:.4f} (epoch {best_metrics['epoch']}) -> {ckpt_path}")
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
