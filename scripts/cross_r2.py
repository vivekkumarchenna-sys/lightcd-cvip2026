"""Cross-dataset protocol (seed-42 checkpoints).

1. Zero-shot, native ground sampling distance: the rerun of the published table.
2. Zero-shot, EGY-BCD resampled to LEVIR's 0.5 m: images are halved before the
   network and logits restored to 256x256, so the labels stay untouched.
3. Few-shot fine-tuning: k labelled pairs from the target's own non-test split,
   for k in {10, 50} (OSCD also 122, its whole train split), three random draws,
   200 iterations at batch 8, AdamW 1e-4, the same loss and augmentation as
   training. The target test set is never used for fine-tuning.

All inputs are the 3-band 8-bit images of each dataset. Output: results/r2/cross.json
"""
import os, sys, json, argparse, random
os.environ.setdefault("HF_HOME", r"C:\Users\vivek\cvip_data\hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from data import ChangeDetectionDataset, build_transforms, normalize_gpu, REPOS
from models import build_model
from losses import BCEDiceLoss
from metrics import CDMetrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(os.environ.get("LIGHTCD_MODELS", os.path.join(ROOT, "models")), "r2")
OUT = os.path.join(ROOT, "results", "r2", "cross.json")
DEV = "cuda"
SUPPORT = {"egy": "val", "sysu": "val", "oscd": "train"}


class Support(Dataset):
    def __init__(self, ds, idx):
        self.d, self.idx, self.tf = ds, idx, build_transforms(train=True)

    def __len__(self):
        return 10 ** 6

    def __getitem__(self, i):
        ex = self.d[int(self.idx[i % len(self.idx)])]
        a = np.asarray(ex["imageA"].convert("RGB")); b = np.asarray(ex["imageB"].convert("RGB"))
        m = (np.asarray(ex["label"].convert("L")) > 127).astype(np.uint8)
        o = self.tf(image=a, imageB=b, mask=m)
        return {"A": o["image"], "B": o["imageB"], "mask": o["mask"].unsqueeze(0).float()}


def load_model(name):
    m = build_model(name, pretrained=False).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODELS, f"{name}_s42.pt"), map_location=DEV)["model"])
    return m


@torch.no_grad()
def test(model, dataset, scale=1.0):
    model.eval()
    met = CDMetrics()
    dl = DataLoader(ChangeDetectionDataset(dataset, "test", train_aug=False), batch_size=16, num_workers=0)
    for b in dl:
        a = normalize_gpu(b["A"].to(DEV)); bb = normalize_gpu(b["B"].to(DEV)); y = b["mask"].to(DEV)
        if scale != 1.0:
            a = F.interpolate(a, scale_factor=scale, mode="bilinear", antialias=True, align_corners=False)
            bb = F.interpolate(bb, scale_factor=scale, mode="bilinear", antialias=True, align_corners=False)
        with torch.amp.autocast("cuda"):
            lo = model(a, bb)
        if lo.shape[-2:] != y.shape[-2:]:
            lo = F.interpolate(lo.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        met.update(lo.float(), y)
    return met.compute()


def finetune(model, ds, idx, iters=200, bs=8, lr=1e-4):
    dl = iter(DataLoader(Support(ds, idx), batch_size=bs, shuffle=True, num_workers=0, drop_last=True))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = BCEDiceLoss().to(DEV)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for _ in range(iters):
        b = next(dl)
        a = normalize_gpu(b["A"].to(DEV)); bb = normalize_gpu(b["B"].to(DEV)); y = b["mask"].to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = loss_fn(model(a, bb), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()


def main():
    from datasets import load_dataset
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    hf = {k: load_dataset(REPOS[k], verification_mode="no_checks")[s] for k, s in SUPPORT.items()}
    for name in ("lightcd_large", "lightcd"):
        res.setdefault(name, {})
        base = load_model(name)
        for ds in ("egy", "sysu", "oscd"):
            res[name].setdefault(ds, {})
            if "zero" not in res[name][ds]:
                res[name][ds]["zero"] = test(base, ds)
            if ds == "egy" and "zero_gsd05" not in res[name][ds]:
                res[name][ds]["zero_gsd05"] = test(base, ds, scale=0.5)
            n = len(hf[ds])
            for k in ((10, 50, 122) if ds == "oscd" else (10, 50)):
                key = f"k{k}"
                draws = res[name][ds].setdefault(key, [])
                for draw in range(len(draws), 3 if k < n else 1):
                    torch.manual_seed(draw); np.random.seed(draw); random.seed(draw)
                    idx = np.random.RandomState(draw).permutation(n)[:k]
                    m = load_model(name)
                    finetune(m, hf[ds], idx)
                    r = test(m, ds)
                    draws.append(r)
                    print(name, ds, key, draw, f"F1 {r['f1']:.4f}", flush=True)
                    json.dump(res, open(OUT, "w"), indent=2)
                    del m; torch.cuda.empty_cache()
            json.dump(res, open(OUT, "w"), indent=2)
            print(name, ds, "zero", round(res[name][ds]["zero"]["f1"], 4), flush=True)
    print("CROSS_DONE")


if __name__ == "__main__":
    main()
