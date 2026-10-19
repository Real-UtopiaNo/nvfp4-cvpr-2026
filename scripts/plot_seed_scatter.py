#!/usr/bin/env python3
"""
Per-Seed Test AUPRC Scatter
===========================
Reproduces the GTC S81547 "Scaling behavior" panel (page 14) for one
architecture/size: one column per recipe with every seed as a jittered dot, the
mean as a thick bar and a t-based 95% CI as the error bar. The Y axis is the
actual test-set AUPRC instead of % of Baseline; the dashed line is the Baseline
mean.

Recipe 8 is W4A4 NVFP4 PTQ with round-to-nearest: each seed's Baseline weights
are loaded into the recipe-1 model, so exactly the layers recipe 1 trains with
NVFP4 are quantized, through the same TE inference path.

Recipes 9 and 10 are simulated W4A4 INT4 PTQ (symmetric RTN, FP32 scales) on the
same layers (the ones recipe 1 builds as te.Linear) of the Baseline model:
  9  per-output-channel weight scales, per-token activation scales
  10 one scale per 16 elements along K (NVFP4's block geometry)

Recipe 11 is recipe 8 on the layers of recipe 12 (Swin): the Baseline weights are
also NVFP4-quantized in the patch embedding, decoder and classifier.

Missing PTQ results are evaluated once (needs the TE NVFP4 runtime, i.e. the
container) and cached as test_metrics_r{8,9,10}_*.json next to the trained
recipes' test metrics.

Usage:
    python scripts/plot_seed_scatter.py --amp-dtype bf16 --recipes 0 1 8 9 10 \
        --data-path /data/lgg-mri-segmentation/kaggle_3m
"""
import re
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))

from utils import RECIPE_NAMES, RECIPE_COLORS  # noqa: E402

COLORS = {**RECIPE_COLORS, 0: "#2c3e50"}
ARCH_TITLES = {"swin": "Swin", "cnn": "CNN", "vit_adamw": "ViT", "vit_adamax": "ViT"}

# PTQ recipe -> (recipe whose trained weights are quantized, recipe whose quantized layers are used,
#                INT4 group size: None = NVFP4 via TE, 0 = per channel/token, n = per n elements)
PTQ_RECIPES = {8: (0, 1, None), 9: (0, 1, 0), 10: (0, 1, 16), 11: (0, 12, None)}
PTQ_FORMATS = {None: "nvfp4_te", 0: "int4_per_channel_token", 16: "int4_group16"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="slice-random",
                   choices=["patient-holdout", "slice-random"])
    p.add_argument("--arch", default="swin")
    p.add_argument("--model-size", default="matched_500k")
    p.add_argument("--recipes", nargs="+", type=int, default=[0, 1, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2036)))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--amp-dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    # Only used when PTQ results have to be evaluated.
    p.add_argument("--data-path", default=str(ROOT / "lgg-mri-segmentation" / "kaggle_3m"))
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=16,
                   help="PTQ eval batch size; keep equal to training so per-batch activation scaling matches")
    p.add_argument("--force-ptq", action="store_true", help="Re-evaluate cached PTQ results")
    p.add_argument("--tag", default="", help="Suffix for the output figure name")
    p.add_argument("--allow-missing", action="store_true",
                   help="Plot trained recipes with only the seeds that have results")
    return p.parse_args()


def find_test_metrics(metrics_dir: Path, recipe_id: int, args):
    """Returns (path, metrics) of the single test result matching size/epochs/amp."""
    pattern = f"test_metrics_r{recipe_id}_*_{args.model_size}_{args.epochs}ep_*.json"
    matches = [(p, json.loads(p.read_text())) for p in metrics_dir.glob(pattern)]
    matches = [(p, m) for p, m in matches if m.get("amp_dtype", "fp16") == args.amp_dtype]
    if len(matches) > 1:
        raise RuntimeError(f"Expected 1 {args.amp_dtype} result for {pattern} in {metrics_dir}, found {len(matches)}")
    return matches[0] if matches else (None, None)


class PTQEvaluator:
    """Evaluates trained weights on the test set with another recipe's quantized layers."""

    def __init__(self, args):
        import torch
        import main as runner
        from utils import create_dataloaders, run_validation_inference

        self.torch, self.args, self.run_inference = torch, args, run_validation_inference
        self.runner = runner
        self.factory = runner._get_model_factory(args.arch)
        self.device = torch.device("cuda")
        self._layer_names = {}
        df = runner.load_lgg_dataset_patient_aware(args.data_path)
        (train_df, val_df, test_df), = runner.build_splits(df, args.split, args.split_seed)
        _, _, self.test_loader = create_dataloaders(train_df, val_df, test_df, batch_size=args.batch_size)

    def quantized_layer_names(self, quant_recipe: int) -> list:
        """Names of the Linear layers the quant recipe builds as te.Linear."""
        if quant_recipe not in self._layer_names:
            model = self.factory(model_size=self.args.model_size, recipe_id=quant_recipe)
            names = [n for n, m in model.named_modules() if type(m).__module__.startswith("transformer_engine")]
            if not names:
                raise RuntimeError(f"{self.args.arch} recipe {quant_recipe} has no te.Linear layers to quantize")
            self._layer_names[quant_recipe] = names
            del model
        return self._layer_names[quant_recipe]

    def __call__(self, ckpt_path: Path, recipe_id: int) -> dict:
        src_recipe, quant_recipe, group_size = PTQ_RECIPES[recipe_id]
        if not self.runner.can_run_recipe(self.args.arch, quant_recipe):
            raise RuntimeError(f"Recipe {quant_recipe} needs the TE NVFP4 runtime (run inside the container)")
        state = self.torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model_recipe = quant_recipe if group_size is None else src_recipe
        model = self.factory(model_size=self.args.model_size, recipe_id=model_recipe).to(self.device)
        missing, _ = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("_extra_state")]
        if missing:
            raise RuntimeError(f"{ckpt_path.name} is missing weights: {missing}")
        if group_size is not None:
            from int4_ptq import apply_int4_ptq
            apply_int4_ptq(model, self.quantized_layer_names(quant_recipe), group_size or None)
        res = self.run_inference(model, self.test_loader, self.device, amp_dtype=self.args.amp_dtype)
        del model
        self.torch.cuda.empty_cache()
        return res


def run_ptq(seed_dir: Path, recipe_id: int, seed: int, args, evaluator: PTQEvaluator) -> dict:
    src_recipe, quant_recipe, group_size = PTQ_RECIPES[recipe_id]
    src_path, _ = find_test_metrics(seed_dir / "metrics", src_recipe, args)
    if src_path is None:
        raise RuntimeError(f"No recipe {src_recipe} test metrics in {seed_dir} to locate its checkpoint")
    config_hash = src_path.stem.rsplit("_", 1)[-1]
    ckpt_path, = seed_dir.glob(f"recipe_{src_recipe}_{args.model_size}_{args.epochs}ep_*_{config_hash}.pt")

    res = evaluator(ckpt_path, recipe_id)
    recipe_name = RECIPE_NAMES.get(recipe_id, f"R{recipe_id}")
    metrics = {
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "source_recipe_id": src_recipe,
        "quantized_layers_recipe_id": quant_recipe,
        "ptq_format": PTQ_FORMATS[group_size],
        "source_checkpoint": ckpt_path.name,
        "arch": args.arch,
        "model_size": args.model_size,
        "seed": seed,
        "fold": "single",
        "dataset": "lgg",
        "split": args.split,
        "amp_dtype": args.amp_dtype,
        "eval_set": "test",
        "auprc": res.get("auprc", 0),
        "dice": res.get("dice", 0),
        "recall": res.get("recall", 0),
        "f2": res.get("f2_score", 0),
        "precision": res.get("precision", 0),
        "auc": res.get("auc", 0),
        "iou": res.get("iou", 0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    slug = re.sub(r"[^A-Za-z0-9]", "", recipe_name)
    out = seed_dir / "metrics" / f"test_metrics_r{recipe_id}_{slug}_{args.model_size}_{args.epochs}ep_{config_hash}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"  seed {seed}: {recipe_name} AUPRC={metrics['auprc']:.4f} (from {ckpt_path.name}) -> {out.name}")
    return metrics


def load_test_auprc(arch_dir: Path, recipe_id: int, seed: int, args, get_evaluator) -> float:
    seed_dir = arch_dir / f"seed_{seed}"
    path, metrics = find_test_metrics(seed_dir / "metrics", recipe_id, args)
    if recipe_id in PTQ_RECIPES and (metrics is None or args.force_ptq):
        if path is not None:
            path.unlink()
        metrics = run_ptq(seed_dir, recipe_id, seed, args, get_evaluator())
    if metrics is None:
        if args.allow_missing:
            return np.nan
        raise RuntimeError(f"No {args.amp_dtype} recipe {recipe_id} test result in {seed_dir / 'metrics'}")
    return metrics["auprc"]


def ci95(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))


def main():
    args = parse_args()
    arch_dir = Path(args.results_dir) / "lgg" / args.split / "single" / args.arch

    evaluator = None

    def get_evaluator():
        nonlocal evaluator
        if evaluator is None:
            evaluator = PTQEvaluator(args)
        return evaluator

    auprc = {rid: np.array([load_test_auprc(arch_dir, rid, s, args, get_evaluator) for s in args.seeds])
             for rid in args.recipes}

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(args.recipes), 5))
    if 0 in auprc:
        ax.axhline(np.nanmean(auprc[0]), color="#3498db", ls="--", lw=2.5, zorder=1)

    for x, rid in enumerate(args.recipes):
        vals, color = auprc[rid][~np.isnan(auprc[rid])], COLORS.get(rid)
        if len(vals) == 0:
            raise RuntimeError(f"No {args.amp_dtype} recipe {rid} test results for the requested seeds")
        mean, half = vals.mean(), ci95(vals)
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(vals)), vals, s=36, color=color,
                   alpha=0.7, edgecolor="white", lw=0.6, zorder=2)
        if len(vals) > 1:
            ax.errorbar(x, mean, yerr=half, color=color, capsize=7, capthick=2, lw=2, zorder=3)
        ax.hlines(mean, x - 0.25, x + 0.25, color=color, lw=3.5, zorder=4)
        n_note = f" (n={len(vals)})" if len(vals) < len(args.seeds) else ""
        ax.text(x + 0.3, mean, f"{mean:.3f}{n_note}", va="center", fontsize=9, fontweight="bold", color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=5)
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"Recipe {rid} ({RECIPE_NAMES.get(rid, rid)}): n={len(vals)}, mean={mean:.4f} ± {half:.4f} (95% CI), "
              f"std={std:.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

    for i, ra in enumerate(args.recipes):
        for rb in args.recipes[i + 1:]:
            both = ~np.isnan(auprc[ra]) & ~np.isnan(auprc[rb])
            diff = auprc[rb][both] - auprc[ra][both]
            if len(diff) < 2:
                p_str = "n/a"
            else:
                p_str = f"{stats.wilcoxon(auprc[rb][both], auprc[ra][both]).pvalue:.3f}" if np.any(diff) else "1.000"
            print(f"{RECIPE_NAMES.get(rb, rb)} - {RECIPE_NAMES.get(ra, ra)} (paired by seed): "
                  f"mean diff={diff.mean():+.4f}, higher in {int((diff > 0).sum())}/{len(diff)}, "
                  f"Wilcoxon p={p_str}")

    size_tag = args.model_size.replace("matched_", "").upper()
    ax.set_title(f"{ARCH_TITLES.get(args.arch, args.arch)} | {size_tag}", fontweight="bold")
    ax.set_xticks(range(len(args.recipes)))
    ax.set_xticklabels([RECIPE_NAMES.get(r, str(r)) for r in args.recipes],
                       rotation=45, ha="right", fontweight="bold")
    ax.set_xlim(-0.6, len(args.recipes) - 0.15)
    ax.set_ylabel("Test AUPRC", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.02, f"{len(args.seeds)} random seeds, mean ± 95% CI",
            transform=ax.transAxes, fontsize=8, color="0.3")
    fig.tight_layout()

    amp_tag = f"_{args.amp_dtype}" if args.amp_dtype != "fp16" else ""
    tag = f"_{args.tag}" if args.tag else ""
    out_path = arch_dir / f"{args.arch}_{size_tag.lower()}_test_auprc_seeds{amp_tag}{tag}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
