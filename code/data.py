"""LEVIR-CD (and compatible) bi-temporal change-detection data pipeline.

Fast path: reads pre-decoded uint8 .npy arrays (memory-mapped) produced by
scripts/prepare_npy.py -> instant random access, no JPEG decode, Windows-safe.
Fallback: decodes on the fly from the HuggingFace parquet dataset.

Identical geometric augmentation is applied to A, B and mask; mild photometric
jitter is applied identically to A and B to preserve change semantics. Images
are normalized with ImageNet statistics (encoders are ImageNet-pretrained).
"""
import os
os.environ.setdefault("HF_HOME", r"C:\Users\vivek\cvip_data\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MEAN_T = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def normalize_gpu(x_uint8):
    """Normalize a uint8 image batch (B,3,H,W) with ImageNet stats on its device.
    Kept out of the CPU DataLoader workers to reduce their CPU/memory cost."""
    x = x_uint8.float().div_(255.0)
    return (x - _MEAN_T.to(x.device)) / _STD_T.to(x.device)

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
REPOS = {
    "levir": "ericyu/LEVIRCD_Cropped_256",
    "sysu": "ericyu/SYSU_CD",
    "oscd": "ericyu/OSCD_Cropped_256",
    "egy": "ericyu/EGY_BCD",
}


def build_transforms(train: bool):
    # NOTE: no Normalize here -- images stay uint8; normalization happens on GPU
    # (see normalize_gpu) to keep CPU workers light. ToTensorV2 returns uint8 CHW.
    add_targets = {"imageB": "image", "mask": "mask"}
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                ToTensorV2(),
            ],
            additional_targets=add_targets,
        )
    return A.Compose([ToTensorV2()], additional_targets=add_targets)


def _npy_paths(dataset_key, split):
    d = os.path.join(DATA_ROOT, dataset_key)
    return (
        os.path.join(d, f"{split}_A.npy"),
        os.path.join(d, f"{split}_B.npy"),
        os.path.join(d, f"{split}_mask.npy"),
    )


class ChangeDetectionDataset(Dataset):
    def __init__(self, dataset_key="levir", split="train", train_aug=None, size=256):
        self.size = size
        if train_aug is None:
            train_aug = split == "train"
        self.tf = build_transforms(train=train_aug)
        pa, pb, pm = _npy_paths(dataset_key, split)
        if os.path.exists(pa):
            self.mode = "npy"
            # store paths only; open memmaps lazily per-process so the Dataset
            # stays picklable for DataLoader workers on Windows (spawn).
            self._paths = (pa, pb, pm)
            self.A = self.B = self.M = None
            self.n = int(np.load(pa, mmap_mode="r").shape[0])
        else:
            # HF fallback (slower)
            from datasets import load_dataset
            self.mode = "hf"
            hf = load_dataset(REPOS[dataset_key], verification_mode="no_checks")
            if split not in hf and split == "val" and "validation" in hf:
                split = "validation"
            self.data = hf[split]
            self.n = len(self.data)

    def __len__(self):
        return self.n

    def _ensure_open(self):
        if self.A is None:
            pa, pb, pm = self._paths
            self.A = np.load(pa, mmap_mode="r")
            self.B = np.load(pb, mmap_mode="r")
            self.M = np.load(pm, mmap_mode="r")

    def __getitem__(self, idx):
        idx = int(idx)
        if self.mode == "npy":
            self._ensure_open()
            a = np.ascontiguousarray(self.A[idx])
            b = np.ascontiguousarray(self.B[idx])
            m = np.ascontiguousarray(self.M[idx])
        else:
            ex = self.data[idx]
            a = np.asarray(ex["imageA"].convert("RGB"))
            b = np.asarray(ex["imageB"].convert("RGB"))
            m = (np.asarray(ex["label"].convert("L")) > 127).astype(np.uint8)
        out = self.tf(image=a, imageB=b, mask=m)
        # A,B kept as uint8 (normalized on GPU); mask as float (1,H,W)
        return {
            "A": out["image"],
            "B": out["imageB"],
            "mask": out["mask"].unsqueeze(0).float(),
        }


def make_loaders(dataset_key="levir", batch_size=16, num_workers=4, size=256):
    tr = ChangeDetectionDataset(dataset_key, "train", True, size)
    va = ChangeDetectionDataset(dataset_key, "val", False, size)
    te = ChangeDetectionDataset(dataset_key, "test", False, size)
    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(va, batch_size=batch_size, shuffle=False, **common),
        DataLoader(te, batch_size=batch_size, shuffle=False, **common),
    )


if __name__ == "__main__":
    tl, vl, te = make_loaders(batch_size=8, num_workers=0)
    b = next(iter(tl))
    print("mode", tl.dataset.mode, "A", tuple(b["A"].shape), "pos_frac", float(b["mask"].mean()))
    print("sizes", len(tl.dataset), len(vl.dataset), len(te.dataset))
    print("DATA_OK")
