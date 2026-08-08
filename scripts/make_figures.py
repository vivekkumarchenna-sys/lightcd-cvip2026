"""Generate all paper figures:
  1) qualitative change maps  (A | B | GT | baseline | ours)
  2) error map for ours       (TP/FP/FN colour overlay)
  3) accuracy-vs-efficiency scatter (params & GMACs vs F1)
  4) validation F1 training curves
  5) TDA spatial-attention visualisation (explainability)

Robust to missing checkpoints: each figure is skipped with a message if its
inputs are absent.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "levir")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_split(split="test"):
    A = np.load(os.path.join(DATA, f"{split}_A.npy"), mmap_mode="r")
    B = np.load(os.path.join(DATA, f"{split}_B.npy"), mmap_mode="r")
    M = np.load(os.path.join(DATA, f"{split}_mask.npy"), mmap_mode="r")
    return A, B, M


def to_tensor(img_uint8):
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def load_model(name):
    ckpt = os.path.join(ROOT, "models", f"{name}_levir.pt")
    if not os.path.exists(ckpt):
        return None
    m = build_model(name, pretrained=False).to(DEV).eval()
    m.load_state_dict(torch.load(ckpt, map_location=DEV)["model"])
    return m


@torch.no_grad()
def predict(model, A, B, idx):
    a = to_tensor(np.ascontiguousarray(A[idx])).to(DEV)
    b = to_tensor(np.ascontiguousarray(B[idx])).to(DEV)
    logit = model(a, b)
    return (torch.sigmoid(logit)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def pick_indices(M, n=6, seed=0):
    fr = np.array([M[i].mean() for i in range(0, len(M), max(1, len(M) // 400))])
    # choose diverse samples with meaningful change
    idxs_all = list(range(0, len(M), max(1, len(M) // 400)))
    order = np.argsort(-fr)
    chosen = [idxs_all[o] for o in order[:n]]
    return chosen


def fig_qualitative(baseline="fc_siam_diff", ours="lightcd_large", n=6):
    A, B, M = load_split("test")
    mb = load_model(baseline); mo = load_model(ours)
    if mo is None:
        print("[qual] ours ckpt missing, skip"); return
    idxs = pick_indices(M, n)
    cols = ["Time 1 (A)", "Time 2 (B)", "Ground truth", "FC-Siam-diff", "LightCD-L (ours)"]
    fig, axes = plt.subplots(n, 5, figsize=(13, 2.5 * n))
    for r, idx in enumerate(idxs):
        imgs = [np.asarray(A[idx]), np.asarray(B[idx]), np.asarray(M[idx]) * 255]
        pb = predict(mb, A, B, idx) * 255 if mb is not None else np.zeros((256, 256))
        po = predict(mo, A, B, idx) * 255
        imgs += [pb, po]
        for c in range(5):
            ax = axes[r, c] if n > 1 else axes[c]
            im = imgs[c]
            ax.imshow(im, cmap=None if c < 2 else "gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
    plt.tight_layout()
    p = os.path.join(FIG, "qualitative.png")
    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print("[qual] wrote", p)


def fig_error_map(ours="lightcd_large", n=4):
    A, B, M = load_split("test")
    mo = load_model(ours)
    if mo is None:
        print("[errmap] skip"); return
    idxs = pick_indices(M, n)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.6 * n))
    titles = ["Time 1", "Time 2", "GT", "Ours: TP(green) FP(red) FN(blue)"]
    for r, idx in enumerate(idxs):
        gt = np.asarray(M[idx]).astype(bool)
        pr = predict(mo, A, B, idx).astype(bool)
        err = np.zeros((256, 256, 3), np.uint8)
        err[gt & pr] = [0, 200, 0]
        err[~gt & pr] = [220, 0, 0]
        err[gt & ~pr] = [0, 0, 220]
        panels = [np.asarray(A[idx]), np.asarray(B[idx]), (gt * 255).astype(np.uint8), err]
        for c in range(4):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(panels[c], cmap="gray" if c == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
    plt.tight_layout()
    p = os.path.join(FIG, "error_map.png")
    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print("[errmap] wrote", p)


def fig_efficiency():
    eff = os.path.join(RES, "efficiency.json")
    tbl = os.path.join(RES, "main_table.csv")
    if not (os.path.exists(eff) and os.path.exists(tbl)):
        print("[eff] missing inputs, skip"); return
    import csv
    e = {r["model"]: r for r in json.load(open(eff))["rows"]}
    rows = list(csv.DictReader(open(tbl)))
    xs, ys, gs, labels = [], [], [], []
    for r in rows:
        m = r["model"]
        f1 = r.get("test_f1") or r.get("val_f1")
        if not f1 or m not in e:
            continue
        xs.append(e[m]["params_M"]); gs.append(e[m].get("gmacs")); ys.append(float(f1)); labels.append(m)
    if not xs:
        print("[eff] no data"); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for x, y, l in zip(xs, ys, labels):
        c = "crimson" if "lightcd" == l else "steelblue"
        ax[0].scatter(x, y, c=c, s=70)
        ax[0].annotate(l, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("Parameters (M)"); ax[0].set_ylabel("Test F1"); ax[0].set_title("Accuracy vs. Params")
    ax[0].grid(alpha=0.3)
    for g, y, l in zip(gs, ys, labels):
        if g is None:
            continue
        c = "crimson" if "lightcd" == l else "steelblue"
        ax[1].scatter(g, y, c=c, s=70)
        ax[1].annotate(l, (g, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[1].set_xlabel("GMACs"); ax[1].set_ylabel("Test F1"); ax[1].set_title("Accuracy vs. Compute")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(FIG, "efficiency_scatter.png")
    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print("[eff] wrote", p)


def fig_curves(models=("lightcd_large", "lightcd", "fc_siam_diff", "resnet18_siam")):
    plt.figure(figsize=(7, 5))
    plotted = 0
    for m in models:
        hp = os.path.join(RES, f"{m}_levir", "history.json")
        if not os.path.exists(hp):
            continue
        h = json.load(open(hp))
        ep = [r["epoch"] for r in h]; f1 = [r["val_f1"] for r in h]
        plt.plot(ep, f1, marker="o", ms=3, label=m)
        plotted += 1
    if not plotted:
        print("[curves] none"); return
    plt.xlabel("Epoch"); plt.ylabel("Validation F1"); plt.title("Validation F1 during training")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(FIG, "training_curves.png")
    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print("[curves] wrote", p)


def fig_attention(ours="lightcd_large", n=4):
    A, B, M = load_split("test")
    mo = load_model(ours)
    if mo is None:
        print("[attn] skip"); return
    acts = {}
    hooks = []
    for i, fu in enumerate(mo.fusions):
        if hasattr(fu, "sa"):
            def mk(i):
                def hook(mod, inp, out):
                    acts[i] = out.detach().cpu().numpy()[0, 0]
                return hook
            hooks.append(fu.sa.register_forward_hook(mk(i)))
    idxs = pick_indices(M, n)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.6 * n))
    titles = ["Time 1", "Time 2", "GT", "TDA spatial attention"]
    import numpy as _np
    for r, idx in enumerate(idxs):
        acts.clear()
        _ = predict(mo, A, B, idx)
        # take the finest-scale attention available, resize to 256
        if acts:
            key = min(acts.keys())
            att = acts[key]
            att = (att - att.min()) / (float(np.ptp(att)) + 1e-6)
            from PIL import Image
            att = _np.asarray(Image.fromarray((att * 255).astype(_np.uint8)).resize((256, 256))) / 255.0
        else:
            att = _np.zeros((256, 256))
        panels = [np.asarray(A[idx]), np.asarray(B[idx]), np.asarray(M[idx]) * 255]
        for c in range(3):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(panels[c], cmap="gray" if c == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        axa = axes[r, 3] if n > 1 else axes[3]
        axa.imshow(np.asarray(B[idx])); axa.imshow(att, cmap="jet", alpha=0.5)
        axa.set_xticks([]); axa.set_yticks([])
        if r == 0:
            axa.set_title(titles[3], fontsize=10)
    for h in hooks:
        h.remove()
    plt.tight_layout()
    p = os.path.join(FIG, "tda_attention.png")
    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print("[attn] wrote", p)


if __name__ == "__main__":
    fig_qualitative()
    fig_error_map()
    fig_efficiency()
    fig_curves()
    fig_attention()
    print("FIGURES_DONE")
