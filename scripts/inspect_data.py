"""Sanity-check LEVIR-CD: mask value range, change-pixel fraction (class balance),
and dump a couple of preview PNGs to figures/."""
import os
os.environ.setdefault("HF_HOME", r"C:\Users\vivek\cvip_data\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from datasets import load_dataset
from PIL import Image

ds = load_dataset("ericyu/LEVIRCD_Cropped_256", verification_mode="no_checks")

for split in ["train", "val", "test"]:
    d = ds[split]
    n = len(d)
    # sample up to 400 masks for class-balance estimate
    idxs = np.linspace(0, n - 1, min(400, n)).astype(int)
    pos = 0
    tot = 0
    uniq = set()
    for i in idxs:
        m = np.array(d[int(i)]["label"])
        if len(uniq) < 10:
            uniq |= set(np.unique(m).tolist())
        pos += (m > 127).sum()
        tot += m.size
    print(f"{split}: n={n} change_pixel_frac={pos/tot:.4f} mask_unique(sample)={sorted(uniq)[:10]}")

# save one preview triptych
os.makedirs("figures", exist_ok=True)
ex = ds["train"][0]
A = np.array(ex["imageA"]); B = np.array(ex["imageB"]); M = np.array(ex["label"])
Mrgb = np.stack([M, M, M], -1)
strip = np.concatenate([A, B, Mrgb], axis=1)
Image.fromarray(strip).save("figures/_sample_triptych.png")
print("saved figures/_sample_triptych.png ; shapes A", A.shape, "B", B.shape, "M", M.shape)
print("DONE_INSPECT")
