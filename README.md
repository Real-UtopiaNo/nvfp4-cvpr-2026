# Experiment

## Reproducing the Swin 500K seed-scatter figures

Both figures plot the test AUPRC of every seed (dots), the mean (bar) and a t-based 95% CI per recipe; the dashed line is the Baseline mean.

| Figure | Recipes |
|--------|---------|
| `results/lgg/slice-random/single/swin/swin_500k_test_auprc_seeds_bf16_int4.png` | 0, 1, 8, 9, 10 |
| `results/lgg/slice-random/single/swin/swin_500k_test_auprc_seeds_bf16_all_layers.png` | 0, 1, 8, 11, 12 |

Recipes used here besides 0 (Baseline) and 1 (NVFP4 Full):

| ID | Recipe | Trained? | Description |
|----|--------|----------|-------------|
| 8 | NVFP4 PTQ (RTN) | No | Baseline weights, W4A4 NVFP4 on the layers recipe 1 quantizes (all transformer Linears) |
| 9 | INT4 PTQ (ch/tok) | No | Baseline weights, simulated symmetric INT4 W4A4 on the same layers, per-output-channel weight / per-token activation scales |
| 10 | INT4 PTQ (g16) | No | As 9 with one FP32 scale per 16 elements along K |
| 11 | NVFP4 PTQ (all layers) | No | As 8, plus the patch embedding, decoder ConvTranspose2d layers and classifier |
| 12 | NVFP4 Full (all layers) | Yes, from scratch | Recipe 1 extended to the patch embedding, decoder and classifier |

### Requirements

- Blackwell GPU (SM 10.x / 12.x) with Transformer Engine NVFP4 support; recipes 1, 8, 11 and 12 need it (INT4 PTQ is simulated in PyTorch).
- Run all commands from the repository root (inside the `nvfp4` container in our setup). Replace `/data/lgg-mri-segmentation/kaggle_3m` with your dataset path.
- Keep `--split slice-random --batch-size 16 --amp-dtype bf16 --epochs 100` identical across training and plotting: they enter the config hash that links checkpoints, metrics and PTQ results. All runs use split seed 2026 and model seeds 2026–2035.

### Step 1: train Baseline (0) and NVFP4 Full (1)

```bash
python main.py train --arch swin --sizes matched_500k \
  --dataset lgg --split slice-random \
  --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
  --recipes 0 1 --epochs 100 --batch-size 16 --amp-dtype bf16 \
  --data-path /data/lgg-mri-segmentation/kaggle_3m
```

Each run writes its best checkpoint (`recipe_<id>_matched_500k_100ep_tversky_<hash>.pt`) and test metrics (`metrics/test_metrics_r<id>_*.json`) to `results/lgg/slice-random/single/swin/seed_<seed>/`. Already trained recipes are skipped on rerun.

### Step 2: INT4 PTQ figure (recipes 0, 1, 8, 9, 10)

```bash
python scripts/plot_seed_scatter.py --arch swin --model-size matched_500k \
  --recipes 0 1 8 9 10 --amp-dtype bf16 --tag int4 \
  --data-path /data/lgg-mri-segmentation/kaggle_3m
```

The PTQ recipes are not trained: on first use the script loads each seed's Baseline checkpoint, quantizes it, evaluates it on the test set and caches `test_metrics_r{8,9,10}_*.json` next to the trained recipes. Add `--force-ptq` to re-evaluate. Paired per-seed differences and Wilcoxon p-values are printed to the console.

### Step 3: train NVFP4 Full on all layers (12)

```bash
python main.py train --arch swin --sizes matched_500k \
  --dataset lgg --split slice-random \
  --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
  --recipes 12 --epochs 100 --batch-size 16 --amp-dtype bf16 \
  --data-path /data/lgg-mri-segmentation/kaggle_3m
```

The model is small, so several seeds can share one GPU, e.g. one process per seed:

```bash
for s in 2026 2027 2028; do
  python main.py train --arch swin --sizes matched_500k --dataset lgg --split slice-random \
    --seeds $s --recipes 12 --epochs 100 --batch-size 16 --amp-dtype bf16 \
    --data-path /data/lgg-mri-segmentation/kaggle_3m > results/swin500k_r12_seed$s.log 2>&1 &
done
wait
```

### Step 4: all-layers figure (recipes 0, 1, 8, 11, 12)

```bash
python scripts/plot_seed_scatter.py --arch swin --model-size matched_500k \
  --recipes 0 1 8 11 12 --amp-dtype bf16 --tag all_layers \
  --data-path /data/lgg-mri-segmentation/kaggle_3m
```

Recipe 11 is evaluated and cached from the Baseline checkpoints the same way as recipes 8–10.

For reference, on one RTX 5060 Ti (16 GB) a 100-epoch Swin 500K run takes about 21 min for recipe 1 and 35 min for recipe 12; each PTQ evaluation takes well under a minute per seed.

# Not All NVFP4 QAT Recipes Are Equal

**How Architecture and Scale Shape Model Quality for Anomaly Segmentation**

*Zijian Du, Oleg Rybakov — NVIDIA*

**CVPR 2026 Workshop on Visual Anomaly and Novelty Detection (VAND)**

---

## Overview

This repository contains the code for our controlled study of NVFP4 quantization-aware training (QAT) across three vision architectures (CNN, ViT, Swin Transformer), three matched parameter scales (530K–13.7M), and eight QAT recipes on a recall-critical brain tumor segmentation task.

**Key findings:**
- Architecture choice determines FP4 quantization robustness — Swin is the most robust
- Advanced QAT recipes prevent two distinct failure modes: softmax attention discretization (small transformers) and gradient noise amplification (large CNNs)
- 4M parameters is the practical sweet spot; Swin is robust to recipe choice at this scale

## Note on NVFP4 QAT Runtime

NVFP4 quantization-aware training requires a compatible NVFP4 training environment. This public release provides the experiment orchestration, model definitions, evaluation metrics, and analysis scripts used for the paper. Vendor/runtime-specific quantization integration is intentionally omitted from the repository.

The provided code supports baseline training and can be adapted to other public quantization frameworks. Non-baseline NVFP4 QAT recipes are retained as experiment metadata and will report a clear runtime-compatibility message in this release.

## Repository Structure

```
nvfp4-cvpr-2026/
├── main.py                  # Unified experiment runner (train / evaluate / analyze)
├── common/
│   ├── loss.py              # Loss functions, AUPRC, evaluation metrics
│   ├── utils.py             # Training loop, data loading, inference
│   └── data_utils.py        # Dataset utilities
├── cnn/
│   └── cnn_qat.py           # Scalable encoder-decoder CNN with NVFP4 quantization
├── swin/
│   └── swin_qat.py          # Swin Transformer with NVFP4 quantization
├── vit/
│   └── vit_qat.py           # Vision Transformer with NVFP4 quantization
├── scripts/
│   ├── analyze_multi_seed.py           # Multi-seed statistical analysis
│   ├── generate_cv_figure.py           # Figure 7: CV robustness
│   └── plot_normalized_recipe_impact.py # Figure 6: QAT sensitivity
├── requirements.txt
└── README.md
```

## Dataset

We use the [LGG Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) dataset from Kaggle.

1. Download from Kaggle and extract to `lgg-mri-segmentation/kaggle_3m/`
2. The dataset contains 3,929 paired MRI slices and binary tumor masks from 110 patients

## Usage

### Training

Train all architectures at 4M scale with patient-level holdout split:

```bash
python main.py train \
  --arch all --sizes matched_4m \
  --dataset lgg --split patient-holdout \
  --seeds 2026 --recipes all --epochs 100
```

Train Swin and CNN with 5-fold patient-level cross-validation:

```bash
python main.py train \
  --arch swin cnn --sizes matched_4m \
  --dataset lgg --split patient-kfold --folds 5 \
  --seeds 2026 --recipes 0 5 --epochs 100
```

### Evaluation

```bash
python main.py evaluate --arch all --sizes all --dataset lgg
```

### Analysis

```bash
python main.py analyze --output-dir results/
```

### Key Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `mode` | `train`, `evaluate`, `analyze` | — | Operation mode |
| `--arch` | `cnn`, `vit_adamw`, `vit_adamax`, `swin`, `all` | `all` | Architecture(s) |
| `--sizes` | `matched_500k`, `matched_4m`, `matched_15m`, `all` | `all` | Model scale(s) |
| `--split` | `patient-holdout`, `patient-kfold` | `patient-holdout` | Data split strategy |
| `--folds` | int | `5` | Number of CV folds |
| `--seeds` | list of ints | `[2026]` | Random seeds |
| `--recipes` | list of ints or `all` | `all` | QAT recipe IDs |
| `--epochs` | int | `100` | Training epochs |
| `--gpu` | int | `0` | GPU device index |

### QAT Recipes

| ID | Recipe | 2D | RHT | SR | Description |
|----|--------|----|-----|-----|-------------|
| 0 | Baseline | — | — | — | FP16/BF16 (no quantization) |
| 1 | NVFP4 Full | | | | Base quantization, all passes |
| 2 | Fwd-Only | | | | Forward pass quantized only |
| 3 | Chain Rule | | | | Reuses forward quantized values in backward |
| 4 | 2D+RHT | ✓ | ✓ | | 2D block-scaling + RHT |
| 5 | 2D+RHT+SR | ✓ | ✓ | ✓ | All three advanced techniques |
| 6 | SR Only | | | ✓ | Stochastic rounding on gradients |
| 7 | Fwd+RHT | | ✓ | | Forward-only with RHT |

## Hardware

Experiments were run on two NVIDIA RTX PRO 6000 (Blackwell) GPUs with 96 GB memory each.

## Citation

```bibtex
@inproceedings{du2026nvfp4,
  title={Not All {NVFP4} {QAT} Recipes Are Equal: How Architecture and Scale Shape Model Quality for Anomaly Segmentation},
  author={Du, Zijian and Rybakov, Oleg},
  booktitle={CVPR 2026 Workshop on Visual Anomaly and Novelty Detection (VAND)},
  year={2026}
}
```

## License

Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
