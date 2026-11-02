"""Does the TDA spatial attention concentrate on changed pixels?

Over the LEVIR-CD test set: mean spatial attention inside and outside the true
change mask, per pyramid level, and the AUROC of attention as a change detector.
Attention is bilinearly resized to 256x256 and measured at full precision without
autocast; AUROC uses a fine histogram in logit space, so saturated gates still rank.
The mean of the channel gate is recorded per level.
Output: results/r2/attention_quant.json
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import ChangeDetectionDataset, normalize_gpu
from models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(os.environ.get("LIGHTCD_MODELS", os.path.join(ROOT, "models")), "r2")
BINS, LO, HI = 400000, -40.0, 40.0


def auroc(hpos, hneg):
    hpos, hneg = hpos / hpos.sum(), hneg / hneg.sum()
    tpr = np.cumsum(hpos[::-1]); fpr = np.cumsum(hneg[::-1])
    return float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))


@torch.no_grad()
def run(name):
    dev = "cuda"
    m = build_model(name, pretrained=False).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(MODELS, f"{name}_s42.pt"), map_location=dev)["model"])
    logits, chan = {}, {}
    hooks = []
    for i, f in enumerate(m.fusions):
        hooks.append(f.sa.conv.register_forward_hook(lambda _m, _i, o, i=i: logits.__setitem__(i, o)))
        hooks.append(f.ca.register_forward_hook(lambda _m, _i, o, i=i: chan.__setitem__(i, o)))
    n = len(m.fusions)
    hp = [torch.zeros(BINS, device=dev, dtype=torch.float64) for _ in range(n)]
    hn = [torch.zeros(BINS, device=dev, dtype=torch.float64) for _ in range(n)]
    sp = np.zeros(n); sn = np.zeros(n); sc = np.zeros(n); cp = cn = nc = 0
    dl = DataLoader(ChangeDetectionDataset("levir", "test", train_aug=False), batch_size=16, num_workers=0)
    for b in dl:
        a = normalize_gpu(b["A"].to(dev)); bb = normalize_gpu(b["B"].to(dev)); y = b["mask"].to(dev) > 0.5
        m(a, bb)
        cp += int(y.sum()); cn += int((~y).sum()); nc += y.shape[0]
        for i in range(n):
            u = F.interpolate(torch.sigmoid(logits[i].double()), size=y.shape[-2:], mode="bilinear", align_corners=False)
            sp[i] += float(u[y].sum()); sn[i] += float(u[~y].sum())
            sc[i] += float(chan[i].double().flatten(1).mean(1).sum())
            lg = torch.log(u.clamp_min(1e-300)) - torch.log1p(-u.clamp_max(1 - 1e-16))
            idx = (((lg - LO) / (HI - LO)) * BINS).long().clamp(0, BINS - 1)
            hp[i] += torch.bincount(idx[y], minlength=BINS).double()
            hn[i] += torch.bincount(idx[~y], minlength=BINS).double()
    for h in hooks:
        h.remove()
    return {f"level{i+1}": {"mean_change": sp[i] / cp, "mean_nochange": sn[i] / cn, "channel_mean": sc[i] / nc,
                            "auroc": auroc(hp[i].cpu().numpy(), hn[i].cpu().numpy())} for i in range(n)}


if __name__ == "__main__":
    out = {m: run(m) for m in ("lightcd_large", "lightcd")}
    os.makedirs(os.path.join(ROOT, "results", "r2"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "r2", "attention_quant.json"), "w"), indent=2)
    for k, v in out.items():
        for l, d in v.items():
            print(k, l, {a: round(b, 4) for a, b in d.items()})
    print("ATT_DONE")
