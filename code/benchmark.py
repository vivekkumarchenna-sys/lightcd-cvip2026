"""Efficiency benchmark: parameters, FLOPs (GMACs), and real GPU latency /
throughput on the local device (RTX 5060). Honest, reproducible numbers that
back the paper's efficiency claims.

FLOPs are counted over conv/linear/bn ops via ptflops; the cheap elementwise
attention/abs/interpolate ops are not counted (they are negligible vs conv).
Latency is wall-clock with CUDA synchronize, warmup, and multiple repeats.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import torch
from models import build_model, MODEL_NAMES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def count_flops(model, device, res=256):
    try:
        from ptflops import get_model_complexity_info

        def ctor(input_res):
            return {
                "a": torch.randn(1, 3, res, res, device=device),
                "b": torch.randn(1, 3, res, res, device=device),
            }

        macs, params = get_model_complexity_info(
            model, (3, res, res), input_constructor=ctor,
            as_strings=False, print_per_layer_stat=False, verbose=False,
        )
        return macs  # multiply-accumulates
    except Exception as e:
        print("  ptflops failed:", e)
        return None


@torch.no_grad()
def measure_latency(model, device, res=256, bs=1, warmup=20, iters=100):
    model.eval()
    a = torch.randn(bs, 3, res, res, device=device)
    b = torch.randn(bs, 3, res, res, device=device)
    for _ in range(warmup):
        model(a, b)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(a, b)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return dt  # seconds per forward (for batch of bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    rows = []
    for name in args.models:
        model = build_model(name, pretrained=False).to(device)
        params = sum(p.numel() for p in model.parameters())
        macs = count_flops(model, device, args.res)
        lat1 = measure_latency(model, device, args.res, bs=1)
        try:
            lat16 = measure_latency(model, device, args.res, bs=16)
            thru = 16 / lat16
        except RuntimeError:
            thru = 1 / lat1
        row = {
            "model": name,
            "params_M": round(params / 1e6, 3),
            "gmacs": round(macs / 1e9, 3) if macs else None,
            "latency_ms_bs1": round(lat1 * 1000, 3),
            "throughput_imgs_s": round(thru, 1),
        }
        rows.append(row)
        print(f"{name:16s} params={row['params_M']:.3f}M GMACs={row['gmacs']} "
              f"lat={row['latency_ms_bs1']}ms fps={row['throughput_imgs_s']}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    out = {"device": dev_name, "res": args.res, "rows": rows}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "efficiency.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
