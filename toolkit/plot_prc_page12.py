"""Reproduce the PDF p12-style figure: Swin PRC + prediction distribution.

Plots the two trained seed-2026 runs (Baseline vs NVFP4 Full) on the held-out
test set, styled after slide p12 of S81547.pdf.

Run from the repo root:  python plot_prc_page12.py
"""
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main

RUNS = [
    ("1. Baseline", "checkpoints/500k_baseline/lgg/patient-holdout/single/swin/seed_2026",
     "recipe_0_matched_500k_100ep_tversky_4cec18fe.pt", 0, "black"),
    # ("1. Baseline", "checkpoints/500k_baseline/lgg/patient-kfold/fold_0/swin/seed_2026",
    #  "recipe_0_matched_500k_100ep_tversky_4cec18fe.pt", 0, "black"),
    # ("0. Baseline_4m", "checkpoints/4m_baseline/lgg/patient-holdout/single/swin/seed_2026",
    # "recipe_0_matched_4m_100ep_tversky_4989cc7e.pt", 0, "#b8aead"),
    
    ("8. NVFP4 Full", "checkpoints/500k_NVFP4Full/lgg/patient-holdout/single/swin/seed_2026",
     "recipe_1_matched_500k_100ep_tversky_4cec18fe.pt", 1, "#c0392b"),
    ("7. Fwd-Only", "checkpoints/500k_FwdOnly/lgg/patient-holdout/single/swin/seed_2026",
     "recipe_2_matched_500k_100ep_tversky_4cec18fe.pt", 2, "#921b0e")
]

OUT = "results/prc_page12_repro.png"


def load_result(run_dir, ckpt_name, recipe_id, loader, device):
    ckpt = torch.load(Path(run_dir) / ckpt_name, map_location="cpu", weights_only=False)
    model = main.create_swin_qat("matched_500k", recipe_id=recipe_id).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    res = main.run_validation_inference(model, loader, device)
    del model
    torch.cuda.empty_cache()
    return res


def main_fn():
    device = torch.device("cuda:0")
    data = main.load_lgg_dataset_patient_aware("lgg-mri-segmentation/kaggle_3m")
    train, val, test = main.patient_holdout_split(data, seed=2026)[0]
    _, _, loader = main.create_dataloaders(train, val, test, batch_size=16)

    results = {}
    for name, run_dir, ckpt_name, rid, color in RUNS:
        res = load_result(run_dir, ckpt_name, rid, loader, device)
        results[name] = (res, color)
        print(f"{name}: test AUPRC = {res['auprc']:.4f}")

    base_auprc = results["1. Baseline"][0]["auprc"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel A: PRC ----
    ax = axes[0]
    for name, (res, color) in results.items():
        delta = res["auprc"] - base_auprc
        label = f"{name} ({res['auprc']:.3f})" if delta == 0 else f"{name} ({delta:+.3f})"
        ax.plot(res["pr_recall"], res["pr_precision"], color=color, lw=1.8, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Swin - PRC  (test set, seed 2026, 500K)")
    ax.grid(alpha=0.25)
    ax.legend(title="Recipe (vs Baseline)", loc="lower left", fontsize=10)

    # ---- Panel B: prediction distribution ----
    ax = axes[1]
    rng = np.random.default_rng(0)
    bins = np.linspace(0, 1, 201)
    for name, (res, color) in results.items():
        p = np.asarray(res["all_probs"], dtype=np.float32).ravel()
        if p.size > 3_000_000:
            p = rng.choice(p, 3_000_000, replace=False)
        filled = name.startswith("1.")
        ax.hist(p, bins=bins, density=True, histtype="stepfilled" if filled else "step",
                alpha=0.35 if filled else 1.0, color=color, lw=1.5,
                label=name.split(". ", 1)[1])
    ax.set_yscale("log")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density (log)")
    ax.set_title("Swin - Prediction Distribution")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main_fn()
