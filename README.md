# LightCD: Efficient Temporal-Difference Attention for Urban Change Detection

Code for a lightweight bi-temporal change-detection (CD) network for urban
growth monitoring, designed to train and run on a single consumer 8 GB GPU
(developed on an NVIDIA RTX 5060 with PyTorch 2.11 / CUDA 12.8, AMP).

## Method
LightCD = weight-shared **MobileNetV3** Siamese encoder + a **Temporal-Difference
Attention (TDA)** fusion module + a compact U-Net decoder. TDA fuses
`[f_A ; f_B ; |f_A - f_B|]` and gates it with channel (SE) and spatial attention
computed *from the temporal difference*. Two operating points:
- **LightCD** (MobileNetV3-Small): ~2.35M params, ~0.48 GMACs, ~1400 img/s.
- **LightCD-L** (MobileNetV3-Large): ~7.31M params, ~1.38 GMACs, higher accuracy.

## Main results (LEVIR-CD test)
| Model | Params (M) | GMACs | F1 (%) | IoU (%) |
|---|---|---|---|---|
| LightCD-L (ours) | 7.31 | 1.38 | **89.2** | **80.5** |
| ResNet18-Siam | 14.74 | 8.55 | 88.0 | 78.6 |
| FC-Siam-conc | 2.14 | 4.78 | 86.7 | 76.5 |
| LightCD (ours) | 2.35 | 0.48 | 86.0 | 75.4 |
| FC-Siam-diff | 1.94 | 4.18 | 83.9 | 72.2 |
| FC-EF | 1.94 | 3.10 | 80.4 | 67.2 |

LightCD-L beats the ResNet-18 baseline with 2x fewer params and 6x fewer FLOPs;
TDA adds +2.8 F1 over plain fusion (ablation). Zero-shot cross-dataset transfer
(EGY-BCD / SYSU / OSCD) largely fails, a candid negative result.

## Repository layout
```
code/          model, data, losses, metrics, train/eval/benchmark
  models/          fc.py (FC-EF/Siam baselines), siamese.py (LightCD + TDA), factory
  data.py          fast memmapped .npy loaders + GPU-side normalization
  train.py         AMP training, best-val checkpointing
  eval.py          test / cross-dataset evaluation
  benchmark.py     params, GMACs (ptflops), latency/throughput
scripts/       prepare data, run experiments, aggregate results, make figures
figures/       result figures (efficiency, curves, qualitative, attention)
results/       metrics json/csv, efficiency.json, aggregated tables
```

## Reproduce
1. Install:
   ```bash
   pip install torch torchvision timm datasets albumentations opencv-python \
               scikit-learn matplotlib ptflops einops pandas pillow pyarrow
   ```
2. Prepare data (downloads from HuggingFace, decodes to fast .npy in `data/`):
   ```bash
   python scripts/prepare_npy.py --dataset levir
   python scripts/prepare_npy.py --dataset egy  --splits test
   python scripts/prepare_npy.py --dataset sysu --splits test
   python scripts/prepare_npy.py --dataset oscd
   ```
3. Efficiency benchmark: `python code/benchmark.py`
4. Train all models + ablations: `python scripts/run_all_training.py`
   (or a single model: `python code/train.py --model lightcd_large --epochs 40 --batch_size 16 --workers 0`)
5. Cross-dataset eval + aggregate + figures:
   ```bash
   python scripts/run_cross_eval.py
   python scripts/aggregate_results.py
   python scripts/make_figures.py
   ```

## Models (`code/models/__init__.py`)
`fc_ef`, `fc_siam_diff`, `fc_siam_conc`, `resnet18_siam`, `lightcd`,
`lightcd_large`, `lightcd_hr`, and ablations `lightcd_diff`, `lightcd_concat`,
`lightcd_noca`, `lightcd_nosa`.

## Datasets (decoded to `data/` via scripts/prepare_npy.py)
- **LEVIR-CD** (`ericyu/LEVIRCD_Cropped_256`) — primary benchmark
- **EGY-BCD** (`ericyu/EGY_BCD`) — cross-dataset (building change)
- **SYSU-CD** (`ericyu/SYSU_CD`) — cross-dataset (broader urban change)
- **OSCD** (`ericyu/OSCD_Cropped_256`) — cross-dataset (Sentinel-2)

## Notes (Windows / small-RAM machines)
Training uses `--workers 0` by default: each spawned DataLoader worker imports
PyTorch (~2.5 GB commit), so many workers can exhaust the system commit limit on
16 GB machines. Image normalization is done on the GPU to keep the single-process
data path fast (~1 min/epoch on the RTX 5060).

## License
MIT (code). Datasets retain their original licenses.
