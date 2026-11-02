"""Figures, regenerated from results/r2/ and the seed-42 checkpoints.
Plots are vector PDF with embedded fonts; image panels are RGB PNG with no alpha
channel. Sizes match the 122 mm LNCS text width, so lettering prints at 7 pt or more;
markers, line styles and lightness carry the meaning in grayscale.

Test images are chosen by change density alone (the 50th, 75th and 90th percentiles of
the test pairs that contain change), never by how well a model did on them.
"""
import os, sys, io, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np, torch
import torch.nn.functional as F
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from PIL import Image
from models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "paper", "figures")
R2 = os.path.join(ROOT, "results", "r2")
DATA = os.path.join(ROOT, "data", "levir")
MODELS = os.path.join(os.environ.get("LIGHTCD_MODELS", os.path.join(ROOT, "models")), "r2")
DEV = "cuda"
W = 4.8
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
STYLE = {"fc_ef": ("FC-EF", "v", "#8a8a8a"), "fc_siam_diff": ("FC-Siam-diff", "<", "#8a8a8a"),
         "fc_siam_conc": ("FC-Siam-conc", ">", "#8a8a8a"), "resnet18_siam": ("ResNet18-Siam", "s", "#5a5a5a"),
         "bit": ("BIT", "D", "#2b6ca3"), "tinycd": ("TinyCD", "^", "#2b6ca3"), "a2net": ("A2Net", "P", "#2b6ca3"),
         "lightcd": ("LightCD", "o", "#b0171f"), "lightcd_large": ("LightCD-L", "*", "#b0171f")}


def save_pdf(fig, n):
    fig.savefig(os.path.join(FIG, n), bbox_inches="tight", pad_inches=0.02); plt.close(fig); print(" ", n)


def save_png(fig, n, dpi=300):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white"); plt.close(fig)
    b.seek(0); Image.open(b).convert("RGB").save(os.path.join(FIG, n), dpi=(dpi, dpi), optimize=True); print(" ", n)


def fig_efficiency():
    S = json.load(open(os.path.join(R2, "summary.json")))
    E = {r["model"]: r for r in json.load(open(os.path.join(R2, "efficiency.json")))["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15), sharey=True)
    for ax, key, xl, ticks in ((axes[0], "gmacs", "GMACs (log scale)", [0.5, 1, 2, 5, 10]),
                               (axes[1], "cpu4_lat_ms", "CPU latency, 4 threads (ms, log scale)", [20, 30, 50, 100])):
        for m, (lab, mk, c) in STYLE.items():
            if m not in S or m not in E:
                continue
            ax.errorbar(E[m][key], S[m]["f1"], yerr=S[m]["f1_std"], fmt=mk, color=c, mec="black", mew=0.4,
                        ms=8 if mk == "*" else 5.5, elinewidth=0.7, capsize=1.5, label=lab, zorder=3)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.grid(alpha=0.3, lw=0.4, which="both")
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test F1 (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.12), handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); save_pdf(fig, "efficiency-scatter.pdf")


def fig_curves():
    runs = [("lightcd_large", "LightCD-L", "-", "#b0171f"), ("lightcd", "LightCD", "--", "#b0171f"),
            ("a2net", "A2Net", "-", "#2b6ca3"), ("tinycd", "TinyCD", "--", "#2b6ca3"),
            ("resnet18_siam", "ResNet18-Siam", "-.", "#555555"), ("fc_siam_diff", "FC-Siam-diff", ":", "#000000")]
    fig, ax = plt.subplots(figsize=(W, 1.75))
    for m, lab, ls, c in runs:
        p = os.path.join(R2, f"{m}_s42", "history.json")
        if os.path.exists(p):
            h = json.load(open(p))
            ax.plot([r["epoch"] + 1 for r in h], [100 * r["val_f1"] for r in h], ls, color=c, lw=1.0, label=lab)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation F1 (%)"); ax.set_xlim(1, 40); ax.set_ylim(60, 95)
    ax.grid(alpha=0.3, lw=0.4); ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.27), columnspacing=0.9, handlelength=1.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save_pdf(fig, "training-curves.pdf")


def load(name):
    m = build_model(name, pretrained=False).to(DEV).eval()
    m.load_state_dict(torch.load(os.path.join(MODELS, f"{name}_s42.pt"), map_location=DEV)["model"])
    return m


def tensor(x):
    x = (np.ascontiguousarray(x).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEV)


@torch.no_grad()
def pred(m, a, b):
    return (torch.sigmoid(m(tensor(a), tensor(b)))[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def picks(M):
    fr = np.asarray(M).mean((1, 2))
    idx = np.where(fr > 0)[0]
    order = idx[np.argsort(fr[idx])]
    return [int(order[int(q * (len(order) - 1))]) for q in (0.5, 0.75, 0.9)]


def error_rgb(p, g):
    # lightness differs for each outcome, so errors stay legible in grayscale
    out = np.full(p.shape + (3,), 30, np.uint8)                    # true negative: near black
    out[(p == 1) & (g == 1)] = (235, 235, 235)                     # true positive: near white
    out[(p == 1) & (g == 0)] = (214, 60, 40)                       # false positive: red
    out[(p == 0) & (g == 1)] = (90, 190, 240)                      # false negative: light blue
    return out


def fig_qualitative(A, B, M):
    mo, mb = load("lightcd_large"), load("fc_siam_diff")
    idx = picks(M)
    cols = ["$t_1$", "$t_2$", "Ground truth", "FC-Siam-diff", "LightCD-L", "Errors, LightCD-L"]
    fig, axes = plt.subplots(3, 6, figsize=(W, 2.45))
    for r, i in enumerate(idx):
        a, b, g = np.asarray(A[i]), np.asarray(B[i]), np.asarray(M[i])
        po, pb = pred(mo, a, b), pred(mb, a, b)
        for c, im in enumerate([a, b, g * 255, pb * 255, po * 255, error_rgb(po, g)]):
            axes[r, c].imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255, interpolation="nearest")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=7)
    fig.tight_layout(pad=0.1, w_pad=0.15, h_pad=0.15); save_png(fig, "qualitative.png")


@torch.no_grad()
def fig_attention(A, B, M):
    m = load("lightcd")
    maps = {}
    for i, f in enumerate(m.fusions):
        f.sa.register_forward_hook(lambda _m, _i, o, i=i: maps.__setitem__(i, o))
    idx = picks(M)[1:]
    cols = ["$t_2$", "Ground truth", "Level 2", "Level 4", "Level 5"]
    fig, axes = plt.subplots(2, 5, figsize=(W, 1.95))
    im = None
    for r, i in enumerate(idx):
        a, b, g = np.asarray(A[i]), np.asarray(B[i]), np.asarray(M[i])
        m(tensor(a), tensor(b))
        axes[r, 0].imshow(b); axes[r, 1].imshow(g * 255, cmap="gray", vmin=0, vmax=255)
        for c, lvl in ((2, 1), (3, 3), (4, 4)):
            u = F.interpolate(maps[lvl], size=(256, 256), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            im = axes[r, c].imshow(u, cmap="gray", vmin=0, vmax=1)
        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=7)
    fig.tight_layout(pad=0.1, w_pad=0.15, h_pad=0.15, rect=(0, 0, 0.92, 1))
    cax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Attention $a_s$", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    save_png(fig, "tda-attention.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    A = np.load(os.path.join(DATA, "test_A.npy"), mmap_mode="r"); B = np.load(os.path.join(DATA, "test_B.npy"), mmap_mode="r")
    M = np.load(os.path.join(DATA, "test_mask.npy"), mmap_mode="r")
    what = sys.argv[1:] or ["eff", "curves", "qual", "attn"]
    if "eff" in what: fig_efficiency()
    if "curves" in what: fig_curves()
    if "qual" in what: fig_qualitative(A, B, M)
    if "attn" in what: fig_attention(A, B, M)
    print("FIGS_DONE")
