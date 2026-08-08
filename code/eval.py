"""Evaluate a trained checkpoint on a dataset split (in-domain or cross-dataset)."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import torch
from torch.utils.data import DataLoader
from data import ChangeDetectionDataset, normalize_gpu
from models import build_model
from metrics import CDMetrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@torch.no_grad()
def run_eval(model_name, ckpt, dataset, split="test", batch_size=16, workers=4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, pretrained=False).to(device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd["model"])
    model.eval()
    ds = ChangeDetectionDataset(dataset, split, train_aug=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    met = CDMetrics()
    for batch in loader:
        a = normalize_gpu(batch["A"].to(device)); b = normalize_gpu(batch["B"].to(device)); m = batch["mask"].to(device)
        with torch.amp.autocast("cuda"):
            logits = model(a, b)
        met.update(logits.float(), m)
    return met.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="levir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    res = run_eval(args.model, args.ckpt, args.dataset, args.split, args.batch_size, args.workers)
    print(json.dumps(res, indent=2))
    run = args.tag or f"{args.model}_{args.dataset}"
    outdir = os.path.join(ROOT, "results", run)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"test_{args.dataset}_{args.split}.json"), "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "split": args.split, **res}, f, indent=2)
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
