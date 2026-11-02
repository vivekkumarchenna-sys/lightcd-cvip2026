"""Efficiency benchmark, run on an otherwise idle machine.

GPU: batch-1 latency and batch-16 throughput, 32-bit float, no autocast.
CPU: batch-1 latency, 32-bit float, with 1 and with 4 threads.
Latency is the median over repeated forward passes on random 256x256 inputs after
warm-up; throughput is images per second at batch 16. Each figure is the median of three
full passes over the models. Parameters are counted directly. MACs come from ptflops,
which counts each pass through the shared encoder.
Output: results/r2/efficiency.json
"""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.dirname(__file__))
import torch
from models import build_model
from benchmark import count_flops

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PASSES = 3
MODELS = ["fc_ef", "fc_siam_diff", "fc_siam_conc", "resnet18_siam", "bit", "a2net", "tinycd",
          "lightcd", "lightcd_large", "lightcd_concat", "lightcd_diff", "lightcd_nogate",
          "lightcd_noca", "lightcd_nosa"]


@torch.no_grad()
def timed(model, device, bs, n, warm):
    a = torch.randn(bs, 3, 256, 256, device=device); b = torch.randn(bs, 3, 256, 256, device=device)
    for _ in range(warm):
        model(a, b)
    out = []
    for _ in range(n):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(a, b)
        if device == "cuda":
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def measure(name):
    m = build_model(name, pretrained=False).eval()
    torch.backends.cudnn.benchmark = True
    g = m.to("cuda")
    lat = timed(g, "cuda", 1, 200, 30); t16 = timed(g, "cuda", 16, 100, 20)
    c = m.to("cpu")
    cpu = {}
    for th in (1, 4):
        torch.set_num_threads(th)
        cpu[th] = timed(c, "cpu", 1, 40, 5)
    torch.set_num_threads(4)
    del m, g, c; torch.cuda.empty_cache()
    return {"gpu_lat_ms": lat * 1e3, "gpu_fps_bs16": 16 / t16, "cpu1_lat_ms": cpu[1] * 1e3, "cpu4_lat_ms": cpu[4] * 1e3}


def main():
    names = []
    for name in sys.argv[1:] or MODELS:
        try:
            build_model(name, pretrained=False)
        except ImportError:
            print("skipping", name, "(port not included)"); continue
        names.append(name)
    runs = {n: [] for n in names}
    for p in range(PASSES):
        for n in names:
            runs[n].append(measure(n)); print(p + 1, n, runs[n][-1], flush=True)
    rows = []
    for n in names:
        m = build_model(n, pretrained=False).eval()
        params = sum(p.numel() for p in m.parameters())
        macs = count_flops(m.to("cpu"), "cpu")
        r = {"model": n, "params_M": round(params / 1e6, 6), "gmacs": round(macs / 1e9, 3)}
        for k in runs[n][0]:
            r[k] = round(statistics.median(x[k] for x in runs[n]), 2)
            r[k + "_passes"] = [round(x[k], 2) for x in runs[n]]
        rows.append(r)
    json.dump({"gpu": torch.cuda.get_device_name(0), "cpu": "Intel Core i7-14650HX", "res": 256, "passes": PASSES, "rows": rows},
              open(os.path.join(ROOT, "results", "r2", "efficiency.json"), "w"), indent=2)
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
