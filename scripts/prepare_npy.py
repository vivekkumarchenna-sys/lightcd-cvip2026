"""One-time: decode a HF change-detection dataset into compact uint8 .npy arrays
for fast, Windows-friendly, decode-free training via memory-mapping.

Usage: python scripts/prepare_npy.py --dataset levir
"""
import os, sys, argparse, time
os.environ.setdefault("HF_HOME", r"C:\Users\vivek\cvip_data\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from datasets import load_dataset

REPOS = {
    "levir": "ericyu/LEVIRCD_Cropped_256",
    "sysu": "ericyu/SYSU_CD",
    "oscd": "ericyu/OSCD_Cropped_256",
    "egy": "ericyu/EGY_BCD",
}
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def prep(dataset_key, size=256, only_splits=None):
    outdir = os.path.join(DATA_ROOT, dataset_key)
    os.makedirs(outdir, exist_ok=True)
    hf = load_dataset(REPOS[dataset_key], verification_mode="no_checks")
    wanted = only_splits if only_splits else ["train", "val", "test"]
    split_map = {}
    for want in wanted:
        s = want
        if s not in hf:
            if s == "val" and "validation" in hf:
                s = "validation"
            else:
                continue
        split_map[want] = s
    for want, s in split_map.items():
        d = hf[s]
        n = len(d)
        A = np.zeros((n, size, size, 3), np.uint8)
        B = np.zeros((n, size, size, 3), np.uint8)
        M = np.zeros((n, size, size), np.uint8)
        t0 = time.time()
        for i in range(n):
            ex = d[i]
            a = np.asarray(ex["imageA"].convert("RGB"))
            b = np.asarray(ex["imageB"].convert("RGB"))
            m = np.asarray(ex["label"].convert("L"))
            if a.shape[:2] != (size, size):
                from PIL import Image
                a = np.asarray(ex["imageA"].convert("RGB").resize((size, size)))
                b = np.asarray(ex["imageB"].convert("RGB").resize((size, size)))
                m = np.asarray(ex["label"].convert("L").resize((size, size)))
            A[i] = a; B[i] = b; M[i] = (m > 127).astype(np.uint8)
            if (i + 1) % 1000 == 0:
                print(f"  {want}: {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
        np.save(os.path.join(outdir, f"{want}_A.npy"), A)
        np.save(os.path.join(outdir, f"{want}_B.npy"), B)
        np.save(os.path.join(outdir, f"{want}_mask.npy"), M)
        print(f"[{dataset_key}] {want}: saved {n} samples, pos_frac={M.mean():.4f}", flush=True)
    print(f"PREP_DONE_{dataset_key}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="levir")
    ap.add_argument("--splits", default=None, help="comma list, e.g. test")
    args = ap.parse_args()
    only = args.splits.split(",") if args.splits else None
    prep(args.dataset, only_splits=only)
