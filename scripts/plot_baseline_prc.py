#!/usr/bin/env python3
"""
Recipe PRC + Prediction Distribution
====================================
Re-evaluates trained Swin checkpoints on the val and test sets and draws the two
panels from the GTC S81547 "QAT Recipes" slide: the pixel-level precision-recall
curve and the log-density histogram of predicted probabilities.

Per-recipe figures go to <seed dir>/figures/; figures overlaying all requested
recipes go to <seed dir>/ itself.

Usage:
    python scripts/plot_baseline_prc.py --amp-dtype bf16 --recipes 0 1
"""
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as runner  # noqa: E402
from utils import (  # noqa: E402
    RECIPE_NAMES, RECIPE_COLORS,
    create_dataloaders, run_validation_inference,
    compute_config_hash, get_training_config, get_checkpoint_path,
)

# Baseline is drawn in black, matching the slide.
COLORS = {**RECIPE_COLORS, 0: "black"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="slice-random",
                   choices=["patient-holdout", "slice-random"])
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--model-size", default="matched_500k")
    p.add_argument("--recipes", nargs="+", type=int, default=[0, 1])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"],
                   help="Precision the checkpoints were trained with; selects them by config hash")
    p.add_argument("--data-path", default=str(ROOT / "lgg-mri-segmentation" / "kaggle_3m"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--ckpt-dir", default=None,
                   help="Defaults to <results-dir>/lgg/<split>/single/swin/seed_<seed>")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def find_checkpoint(ckpt_dir: Path, recipe_id: int, args) -> Path:
    cfg = runner.ARCH_CONFIGS["swin"]
    config = get_training_config(
        model_size=args.model_size, num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"],
        loss_type=runner.LOSS_TYPE, seed=args.seed, amp_dtype=args.amp_dtype,
    )
    path = get_checkpoint_path(ckpt_dir, recipe_id, args.model_size, args.epochs,
                               runner.LOSS_TYPE, compute_config_hash(config))
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {ckpt_dir} (check --amp-dtype/--batch-size/--epochs)")
    return path


def recipe_label(recipe_id: int, res) -> str:
    return f"{recipe_id + 1}. {RECIPE_NAMES.get(recipe_id, recipe_id)} ({res['auprc']:.3f})"


def plot_prc(results, out_path: Path, label: str):
    """``results``: list of (recipe_id, inference result) drawn on one axis."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for recipe_id, res in results:
        # Exact curve over all unique scores, consistent with average_precision_score.
        precision, recall, _ = precision_recall_curve(res["all_targets"] > 0.5, res["all_probs"])
        ax.plot(recall, precision, color=COLORS.get(recipe_id), lw=2.5 if recipe_id == 0 else 2,
                label=recipe_label(recipe_id, res))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title(f"Swin — PRC ({label})", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(title="Recipe", loc="lower left", prop={"weight": "bold"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pred_dist(results, out_path: Path, label: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for recipe_id, res in results:
        name = RECIPE_NAMES.get(recipe_id, recipe_id)
        if recipe_id == 0:
            ax.hist(res["all_probs"], bins=100, range=(0, 1), density=True,
                    histtype="stepfilled", facecolor="0.6", edgecolor="black", lw=1.2,
                    alpha=0.8, label=name)
        else:
            ax.hist(res["all_probs"], bins=100, range=(0, 1), density=True,
                    histtype="step", color=COLORS.get(recipe_id), lw=1.8, label=name)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted Probability", fontweight="bold")
    ax.set_ylabel("Density (log)", fontweight="bold")
    ax.set_title(f"Swin — Prediction Distribution ({label})", fontweight="bold")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        Path(args.results_dir) / "lgg" / args.split / "single" / "swin" / f"seed_{args.seed}")
    ckpt_paths = {rid: find_checkpoint(ckpt_dir, rid, args) for rid in args.recipes}

    df = runner.load_lgg_dataset_patient_aware(args.data_path)
    (train_df, val_df, test_df), = runner.build_splits(df, args.split, args.split_seed)
    _, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, batch_size=args.batch_size)
    loaders = (("val", val_loader), ("test", test_loader))

    fig_dir = ckpt_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    size_tag = args.model_size.replace("matched_", "")
    amp_tag = f"_{args.amp_dtype}" if args.amp_dtype != "fp16" else ""

    results = {name: [] for name, _ in loaders}
    for rid, ckpt_path in ckpt_paths.items():
        rname = RECIPE_NAMES.get(rid, str(rid))
        slug = rname.lower().replace(" ", "_")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = runner.create_swin_qat(model_size=args.model_size, recipe_id=rid).to(device)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        print(f"Recipe {rid} ({rname}): {ckpt_path.name}  "
              f"best epoch {ckpt.get('best_epoch', '?')} / {ckpt.get('actual_epochs', '?')}")

        for name, loader in loaders:
            res = run_validation_inference(model, loader, device, amp_dtype=args.amp_dtype)
            results[name].append((rid, res))
            plot_prc([(rid, res)], fig_dir / f"swin_{size_tag}_{slug}_prc_{name}{amp_tag}.png", name)
            plot_pred_dist([(rid, res)], fig_dir / f"swin_{size_tag}_{slug}_pred_dist_{name}{amp_tag}.png", name)
            print(f"  {name:<4}: AUPRC={res['auprc']:.4f}  AUC={res['auc']:.4f}  "
                  f"Recall={res['recall']:.4f}  F2={res['f2_score']:.4f}")
        del model
        torch.cuda.empty_cache()

    print(f"Per-recipe figures: {fig_dir}")
    if len(args.recipes) > 1:
        for name, res_list in results.items():
            plot_prc(res_list, ckpt_dir / f"swin_{size_tag}_prc_{name}{amp_tag}.png", name)
            plot_pred_dist(res_list, ckpt_dir / f"swin_{size_tag}_pred_dist_{name}{amp_tag}.png", name)
        print(f"Combined figures: {ckpt_dir}")


if __name__ == "__main__":
    main()
