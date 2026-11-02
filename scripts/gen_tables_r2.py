"""Tables and number macros for the paper, generated from results/r2/ so no
figure is typed by hand. Run scripts/aggregate_r2.py first.

Published F1 values are read from the original papers (see references/pdfs/) and are
the only numbers here that are not from our own runs.
"""
import os, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R2 = os.path.join(ROOT, "results", "r2")
OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)


def jl(n):
    p = os.path.join(R2, n)
    return json.load(open(p)) if os.path.exists(p) else None


S = jl("summary.json") or {}
E = {r["model"]: r for r in (jl("efficiency.json") or {"rows": []})["rows"]}
X = jl("cross.json") or {}
MAIN = [("fc_ef", "FC-EF~\\cite{daudt2018fcsiam}", "None", ""),
        ("fc_siam_diff", "FC-Siam-diff~\\cite{daudt2018fcsiam}", "None", ""),
        ("fc_siam_conc", "FC-Siam-conc~\\cite{daudt2018fcsiam}", "None", ""),
        ("resnet18_siam", "ResNet18-Siam", "ImageNet", ""),
        ("bit", "BIT~\\cite{chen2022bit}", "ImageNet", "89.31"),
        ("tinycd", "TinyCD~\\cite{codegoni2023tinycd}", "ImageNet", "91.05"),
        ("a2net", "A2Net~\\cite{li2023a2net}", "ImageNet", "91.73"),
        ("lightcd", "\\textbf{LightCD}", "ImageNet", ""),
        ("lightcd_large", "\\textbf{LightCD-L}", "ImageNet", "")]
ABL = [("lightcd_diff", "Difference only, $[\\mathbf{d}]$"),
       ("lightcd_concat", "Concatenation, $[\\mathbf{f}_A;\\mathbf{f}_B]$"),
       ("lightcd_nogate", "With difference, no gates, $[\\mathbf{f}_A;\\mathbf{f}_B;\\mathbf{d}]$"),
       ("lightcd_noca", "Spatial gate only"),
       ("lightcd_nosa", "Channel gate only"),
       ("lightcd", "\\textbf{Both gates (TDA)}")]


def ms(m, k, d=1):
    r = S.get(m)
    if not r:
        return "--"
    return f"{r[k]:.{d}f}$\\pm${r[k+'_std']:.{d}f}" if r["n"] > 1 else f"{r[k]:.{d}f}"


def mean_only(m, k, d=1):
    return f"{S[m][k]:.{d}f}" if m in S else "--"


def table_main():
    rows = []
    best = max((S[m]["f1"] for m, *_ in MAIN if m in S), default=None)
    for m, lab, pre, pub in MAIN:
        cells = [lab, pre, str(S[m]["n"]) if m in S else "--", mean_only(m, "precision"), mean_only(m, "recall"), ms(m, "f1"), ms(m, "iou"),
                 mean_only(m, "oa", 2), mean_only(m, "kappa")]
        if m in S and S[m]["f1"] == best:
            cells[5] = "\\textbf{" + cells[5] + "}"
        rows.append(" & ".join(cells + [pub]) + " \\\\")
    rows.append("\\midrule\nChangeFormer~\\cite{bandara2022changeformer} & -- & -- & \\multicolumn{6}{c}{not trained here} & 90.40 \\\\")
    n = sorted({S[m]["n"] for m, *_ in MAIN if m in S})
    seeds = f"{n[0]}" if len(n) == 1 else f"{n[0]} to {n[-1]}"
    tex = ("\\begin{table}[!htbp]\n\\centering\n"
           f"\\caption{{Accuracy on the LEVIR-CD test set (means, with s.d. of F1 and IoU where two seeds ran). All models except ChangeFormer trained for 40 epochs "
           "under one protocol. Weights are the encoder initialisation of our runs. The last column is the F1 published by the original "
           "authors under their own protocols.}\n"
           "\\label{tab:main}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{llcccccccc}\n\\toprule\n"
           "Method & Weights & Seeds & P (\\%) & R (\\%) & F1 (\\%) & IoU (\\%) & OA (\\%) & $\\kappa$ (\\%) & Publ.\\ F1 \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(OUT, "main-table.tex"), "w", newline="\n").write(tex)


def table_eff():
    rows = []
    for m, lab, *_ in MAIN:
        e = E.get(m)
        if not e:
            rows.append(lab + " & -- & -- & -- & -- & -- & -- \\\\"); continue
        rows.append(" & ".join([lab, f"{e['params_M']:.2f}", f"{e['gmacs']:.2f}", f"{e['gpu_lat_ms']:.1f}",
                                f"{e['gpu_fps_bs16']:.0f}", f"{e['cpu1_lat_ms']:.0f}", f"{e['cpu4_lat_ms']:.0f}"]) + " \\\\")
    tex = ("\\begin{table}[!htbp]\n\\centering\n"
           "\\caption{Efficiency at $256{\\times}256$ with 32-bit floats. GPU: RTX~5060 laptop. CPU: Core i7-14650HX limited to 1 or "
           "4 threads. Latency is the median single-image time; throughput uses batch size 16.}\n"
           "\\label{tab:eff}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
           " & & & \\multicolumn{2}{c}{GPU} & \\multicolumn{2}{c}{CPU latency (ms)} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
           "Method & Params (M) & GMACs & Latency (ms) & Images/s & 1 thread & 4 threads \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(OUT, "eff-table.tex"), "w", newline="\n").write(tex)


def table_abl():
    rows = []
    for m, lab in ABL:
        e = E.get(m, {})
        f = ms(m, "f1")
        rows.append(" & ".join([lab, f"{e.get('params_M', 0):.2f}" if e else "--",
                                f"{e.get('gmacs', 0):.3f}" if e else "--", f, ms(m, "iou"),
                                str(S[m]["n"]) if m in S else "--"]) + " \\\\")
    tex = ("\\begin{table}[!htbp]\n\\centering\n"
           "\\caption{Ablation of the fusion module on LEVIR-CD (test), all on MobileNetV3-Small and 40 epochs, mean $\\pm$ s.d. over seeds.}\n"
           "\\label{tab:ablation}\n\\footnotesize\n\\begin{tabular}{lccccc}\n\\toprule\n"
           "Fusion variant & Params (M) & GMACs & F1 (\\%) & IoU (\\%) & Seeds \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(OUT, "ablation-table.tex"), "w", newline="\n").write(tex)


def fs(d, key):
    """F1 % of one draw or the mean of the draws."""
    v = d.get(key)
    if v is None:
        return "--"
    if isinstance(v, list):
        a = [100 * x["f1"] for x in v]
        if not a:
            return "--"
        m = sum(a) / len(a)
        sd = (sum((x - m) ** 2 for x in a) / (len(a) - 1)) ** 0.5 if len(a) > 1 else 0.0
        return f"{m:.1f}$\\pm${sd:.1f}" if len(a) > 1 else f"{m:.1f}"
    return f"{100 * v['f1']:.1f}"


def table_cross():
    if not X:
        return
    names = {"egy": "EGY-BCD", "sysu": "SYSU-CD", "oscd": "OSCD"}
    rows = []
    for ds, lab in names.items():
        a, b = X.get("lightcd_large", {}).get(ds, {}), X.get("lightcd", {}).get(ds, {})
        rows.append(" & ".join([lab, fs(a, "zero"), fs(a, "k10"), fs(a, "k50"), fs(b, "zero"), fs(b, "k10"), fs(b, "k50")]) + " \\\\")
    tex = ("\\begin{table}[!htbp]\n\\centering\n"
           "\\caption{Transfer from LEVIR-CD: F1 (\\%) on each target test set with no target labels (0) and after fine-tuning on $k$ labelled "
           "target pairs from a non-test split, mean $\\pm$ s.d. over three random draws.}\n"
           "\\label{tab:cross}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
           " & \\multicolumn{3}{c}{LightCD-L} & \\multicolumn{3}{c}{LightCD} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
           "Target & $k{=}0$ & $k{=}10$ & $k{=}50$ & $k{=}0$ & $k{=}10$ & $k{=}50$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(OUT, "cross-table.tex"), "w", newline="\n").write(tex)


def macros():
    L = []
    def cmd(n, v): L.append(f"\\newcommand{{\\{n}}}{{{v}}}")
    for tag, m in (("A", "lightcd"), ("B", "lightcd_large"), ("C", "tinycd"), ("D", "a2net"), ("E", "bit"), ("F", "resnet18_siam"), ("G", "fc_siam_conc")):
        if m in S:
            cmd(f"Fone{tag}", f"{S[m]['f1']:.1f}"); cmd(f"Fsd{tag}", f"{S[m]['f1_std']:.1f}")
        if m in E:
            for k, c in (("params_M", "Par"), ("gmacs", "Mac"), ("gpu_lat_ms", "Glat"), ("gpu_fps_bs16", "Gfps"),
                         ("cpu1_lat_ms", "Cone"), ("cpu4_lat_ms", "Cfour")):
                d = {"params_M": 2, "gmacs": 2, "gpu_lat_ms": 1}.get(k, 0)
                cmd(f"{c}{tag}", f"{E[m][k]:.{d}f}")
    if all(m in E for m in ("lightcd", "lightcd_concat", "lightcd_nogate", "lightcd_diff")):
        p = {m: E[m]["params_M"] for m in ("lightcd", "lightcd_concat", "lightcd_nogate", "lightcd_diff")}
        cmd("ConcatParams", f"{p['lightcd_concat']:.2f}")
        cmd("TDAvsConcat", f"{p['lightcd'] - p['lightcd_concat']:.2f}")
        cmd("DPath", f"{p['lightcd_nogate'] - p['lightcd_concat']:.3f}")
        cmd("GatePath", f"{p['lightcd'] - p['lightcd_nogate']:.3f}")
        cmd("TDAvsDiff", f"{p['lightcd'] - p['lightcd_diff']:.2f}")
    for tag, m in (("Diff", "lightcd_diff"), ("Concat", "lightcd_concat"), ("Nogate", "lightcd_nogate"),
                   ("Noca", "lightcd_noca"), ("Nosa", "lightcd_nosa")):
        if m in S:
            cmd(f"Fone{tag}", f"{S[m]['f1']:.1f}"); cmd(f"Fsd{tag}", f"{S[m]['f1_std']:.1f}")

    def mean(d, k):
        v = d[k]
        return sum(100 * x["f1"] for x in v) / len(v) if isinstance(v, list) else 100 * v["f1"]
    for tag, m in (("L", "lightcd_large"), ("S", "lightcd")):
        for dtag, ds in (("Egy", "egy"), ("Sysu", "sysu"), ("Oscd", "oscd")):
            d = X.get(m, {}).get(ds, {})
            for ktag, k in (("Zero", "zero"), ("Half", "zero_gsd05"), ("Ten", "k10"), ("Fifty", "k50"), ("All", "k122")):
                if k in d:
                    cmd(f"X{tag}{dtag}{ktag}", f"{mean(d, k):.1f}")
    A = jl("attention_quant.json") or {}
    for tag, m in (("L", "lightcd_large"), ("S", "lightcd")):
        for i, w in enumerate(("One", "Two", "Three", "Four", "Five"), 1):
            r = A.get(m, {}).get(f"level{i}")
            if r:
                cmd(f"Auc{tag}{w}", f"{r['auroc']:.2f}")
    open(os.path.join(OUT, "macros.tex"), "w", newline="\n").write("\n".join(L) + "\n")


if __name__ == "__main__":
    table_main(); table_eff(); table_abl(); table_cross(); macros()
    print("TABLES_DONE", len(S), "models", len(E), "bench rows")
