# LightCD: Efficient Temporal-Difference Attention for Urban Change Detection

Code for a lightweight bi-temporal change-detection (CD) network for urban
growth monitoring, designed to train and run on a single consumer 8 GB GPU
(developed on an NVIDIA RTX 5060 with PyTorch 2.11 / CUDA 12.8, AMP).

## Method
LightCD is a weight-shared MobileNetV3 Siamese encoder, a Temporal-Difference Attention
(TDA) fusion module and a compact U-Net decoder. TDA fuses `[f_A ; f_B ; |f_A - f_B|]` and gates
it with channel and spatial attention read from the temporal difference. LightCD uses
MobileNetV3-Small and LightCD-L uses MobileNetV3-Large.

## Results (LEVIR-CD test)
Every model trained for 40 epochs at batch size 16 under one protocol. Latency is the median single-image time at 256x256 with
32-bit floats; throughput uses batch size 16 on an RTX 5060 laptop GPU.

| Model | Seeds | Params (M) | GMACs | CPU 1 thread (ms) | GPU images/s | F1 (%) | IoU (%) |
|---|---|---|---|---|---|---|---|
| A2Net (our port) | 1 | 3.78 | 2.98 | 78 | 512 | 89.6 | 81.2 |
| LightCD-L (ours) | 2 | 7.31 | 1.38 | 47 | 771 | 89.3 | 80.6 |
| ResNet18-Siam | 1 | 14.74 | 8.55 | 221 | 428 | 88.2 | 78.9 |
| FC-Siam-conc | 1 | 2.14 | 4.78 | 115 | 466 | 87.3 | 77.4 |
| BIT (our port) | 1 | 3.04 | 8.72 | 252 | 274 | 86.6 | 76.3 |
| LightCD (ours) | 2 | 2.35 | 0.48 | 25 | 1288 | 86.1 | 75.6 |
| TinyCD (our port) | 1 | 0.29 | 1.49 | 126 | 244 | 84.9 | 73.8 |
| FC-Siam-diff | 1 | 1.94 | 4.18 | 105 | 515 | 84.9 | 73.7 |
| FC-EF | 1 | 1.94 | 3.10 | 71 | 731 | 81.7 | 69.1 |

The ablation found no measurable F1 gain from TDA over plain concatenation
(`results/r2/summary.json`), and transfer to EGY-BCD, SYSU-CD and OSCD fails without target
labels. TinyCD, A2Net and BIT reach higher F1 under their authors' longer budgets. `results/`
outside `r2/` holds earlier runs.

## Repository layout
```
code/          models, data, losses, metrics, train/eval/benchmark
  models/          fc.py (FC baselines), siamese.py (LightCD, TDA), tinycd.py, bit.py, factory
  train.py         AMP training, best-val checkpointing, per-epoch resume
  eval.py          test evaluation
  benchmark_r2.py  params, GMACs (ptflops), GPU and CPU latency, throughput
scripts/       review runs, cross-dataset study, attention measure, tables, figures
figures/       figures of the paper
results/r2/    every result behind the paper
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
3. Train every model and seed (resumable; set `LIGHTCD_MODELS` to keep checkpoints elsewhere):
   ```bash
   python scripts/run_review_queue.py
   ```
4. Benchmark, transfer, attention, tables and figures:
   ```bash
   python code/benchmark_r2.py
   python scripts/cross_r2.py
   python scripts/attention_quant.py
   python scripts/aggregate_r2.py
   python scripts/gen_tables_r2.py
   python scripts/make_figures_r2.py
   ```

## Models (`code/models/__init__.py`)
`fc_ef`, `fc_siam_diff`, `fc_siam_conc`, `resnet18_siam`, `tinycd`, `bit`, `a2net`, `lightcd`,
`lightcd_large`, `lightcd_hr`, and ablations `lightcd_diff`, `lightcd_concat`, `lightcd_nogate`,
`lightcd_noca`, `lightcd_nosa`.

`a2net` needs `code/models/a2net.py`, which is not included, because its source repository
(github.com/guanyuezhen/A2Net) states no licence. Porting it from that repository restores the
model; `scripts/run_review_queue.py` and `code/benchmark_r2.py` expect the class `A2Net`.

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
MIT for our code. `code/models/tinycd.py` and `code/models/bit.py` are ported from their authors'
repositories, which release the code for non-commercial and research purposes only, so those two
files keep that restriction. Datasets retain their original licences.
