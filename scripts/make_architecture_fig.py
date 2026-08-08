"""Journal-quality LightCD architecture diagram (figures/architecture.png).
(a) overall pipeline; (b) the TDA module. Generously sized boxes (no text
overflow) and orthogonal, non-overlapping connectors (clean flow).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

C_IMG = "#cfe4ff"; C_ENC = "#ffe0b3"; C_FEAT = "#fff0d6"
C_TDA = "#ffc9cf"; C_DEC = "#c9e8cc"; C_OUT = "#e5ccf0"; C_ATT = "#ded2f5"; C_OP = "#bfe6f5"
EDGE = "#2b2b2b"


def box(ax, cx, cy, w, h, text, color, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=1.4, edgecolor=EDGE, facecolor=color, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.35)
    return dict(cx=cx, cy=cy, l=cx-w/2, r=cx+w/2, t=cy+h/2, b=cy-h/2)


def arr(ax, p0, p1, color=EDGE, lw=1.7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                 lw=lw, color=color, zorder=2, shrinkA=0, shrinkB=0,
                 joinstyle="miter", capstyle="butt"))


def poly(ax, pts, color=EDGE, lw=1.7):
    """orthogonal polyline with an arrow head on the last segment."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, lw=lw, zorder=2, solid_capstyle="round")
    arr(ax, pts[-2], pts[-1], color=color, lw=lw)


fig, (axa, axb) = plt.subplots(2, 1, figsize=(11.5, 8.2),
                               gridspec_kw={"height_ratios": [0.92, 1.08]})

# ================= (a) pipeline =================
axa.set_xlim(0, 32); axa.set_ylim(0, 10); axa.axis("off")
axa.text(16, 9.3, "(a) LightCD pipeline", ha="center", fontsize=12.5, fontweight="bold")

imgA = box(axa, 2.4, 6.7, 3.3, 1.5, "Image $t_1$", C_IMG, 11)
imgB = box(axa, 2.4, 3.1, 3.3, 1.5, "Image $t_2$", C_IMG, 11)
enc  = box(axa, 8.6, 4.9, 4.6, 3.3, "Siamese\nMobileNetV3\nencoder\n(shared weights)", C_ENC, 10.5, True)
feat = box(axa, 14.8, 4.9, 3.5, 2.5, "Multi-scale\nfeatures\n$f_A^{(l)},\\, f_B^{(l)}$", C_FEAT, 10.5)
tda  = box(axa, 20.2, 4.9, 3.4, 2.5, "TDA\nfusion", C_TDA, 12, True)
dec  = box(axa, 25.2, 4.9, 3.2, 2.5, "U-Net\ndecoder", C_DEC, 11, True)
out  = box(axa, 29.6, 4.9, 3.0, 2.2, "Change\nmap $\\hat{y}$", C_OUT, 10.5, True)

arr(axa, (imgA["r"], imgA["cy"]), (enc["l"], enc["cy"]+0.85))
arr(axa, (imgB["r"], imgB["cy"]), (enc["l"], enc["cy"]-0.85))
for A, B in [(enc, feat), (feat, tda), (tda, dec), (dec, out)]:
    arr(axa, (A["r"], 4.9), (B["l"], 4.9))

# ================= (b) TDA module =================
axb.set_xlim(0, 32); axb.set_ylim(0, 12); axb.axis("off")
axb.text(16, 11.3, "(b) Temporal-Difference Attention (TDA) module, per scale $l$",
         ha="center", fontsize=12.5, fontweight="bold")

src  = box(axb, 3.4, 6.0, 4.4, 1.7, "Bi-temporal\nfeatures\n$f_A^{(l)},\\, f_B^{(l)}$", C_FEAT, 10.5)
diff = box(axb, 10.2, 6.0, 4.2, 1.5, "$d = |f_A^{(l)} - f_B^{(l)}|$", "#ffb3ba", 11, True)
fuse = box(axb, 17.6, 9.2, 5.6, 1.8, "Concat $[f_A;\\, f_B;\\, d]$\n$+\\; 1{\\times}1$ conv $\\to u$", C_ENC, 10)
attn = box(axb, 17.6, 3.2, 6.4, 2.6, "", C_ATT, 9.5)
axb.text(17.6, 3.95, "Difference-guided attention", ha="center", va="center",
         fontsize=10, fontweight="bold", zorder=5)
axb.text(17.6, 3.15, r"channel:  $a_c=\sigma(\mathrm{SE}(d))$", ha="center", va="center",
         fontsize=9.5, zorder=5)
axb.text(17.6, 2.55, r"spatial:  $a_s=\sigma(\mathrm{conv}(d))$", ha="center", va="center",
         fontsize=9.5, zorder=5)
gate = box(axb, 24.6, 6.0, 1.9, 1.6, "$\\otimes$", C_OP, 16, True)
fused = box(axb, 29.0, 6.0, 3.4, 1.8, "Gated\nfeature $\\tilde{u}^{(l)}$", C_TDA, 10.5, True)

# src -> diff (straight); src -> fuse (top arc, labelled f_A,f_B)
arr(axb, (src["r"], 6.0), (diff["l"], 6.0))
poly(axb, [(src["cx"], src["t"]), (src["cx"], 10.6), (fuse["cx"]-1.0, 10.6), (fuse["cx"]-1.0, fuse["t"])])
axb.text(9.5, 10.85, "$f_A, f_B$", fontsize=9, color="#444")
# diff -> fuse (up) ; diff -> attn (down)
poly(axb, [(diff["cx"], diff["t"]), (diff["cx"], fuse["cy"]), (fuse["l"], fuse["cy"])])
poly(axb, [(diff["cx"], diff["b"]), (diff["cx"], attn["cy"]), (attn["l"], attn["cy"])])
axb.text(diff["cx"]+0.2, 7.2, "$d$", fontsize=9.5, color="#444")
# fuse -> gate (down) ; attn -> gate (up)
poly(axb, [(fuse["r"], fuse["cy"]), (gate["cx"], fuse["cy"]), (gate["cx"], gate["t"])])
poly(axb, [(attn["r"], attn["cy"]), (gate["cx"], attn["cy"]), (gate["cx"], gate["b"])])
axb.text(21.0, 9.5, "$u$", fontsize=9.5, color="#444")
axb.text(21.0, 3.5, "$a_c, a_s$", fontsize=9, color="#444")
# gate -> fused
arr(axb, (gate["r"], 6.0), (fused["l"], 6.0))

plt.tight_layout()
p = os.path.join(FIG, "architecture.png")
plt.savefig(p, dpi=175, bbox_inches="tight", facecolor="white"); plt.close()
print("wrote", p)
