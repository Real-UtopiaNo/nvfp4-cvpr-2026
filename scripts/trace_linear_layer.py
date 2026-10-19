#!/usr/bin/env python3
"""
Linear-Layer Forward Trace (Baseline vs NVFP4 QAT vs NVFP4 PTQ)
===============================================================
Runs one test slice through three Swin models and records one Linear layer's
real input/output, then rebuilds every arithmetic step of that layer's forward
pass and checks the rebuild against the layer's actual output:

  recipe 0  Baseline weights, nn.Linear under BF16 autocast
            X -> BF16(X), W -> BF16(W), BF16 GEMM with FP32 accumulation, BF16 out
  recipe 1  NVFP4 Full (QAT) weights, te.Linear with NVFP4BlockScaling
  recipe 8  Baseline weights in the recipe-1 model (W4A4 NVFP4 PTQ, RTN)
            per operand: global amax -> S_enc = 448*6/amax, 1x16 block amax,
            block scale s = E4M3(block_amax / 6 * S_enc), q = E2M1(x * S_enc / s),
            FP4 GEMM reads q and s: Y = sum_b s_x s_w (q_x . q_w) / (S_x S_w), FP32 accum,
            BF16 out; x_hat = q * s / S_enc is plotted as the equivalent dequantized view

The FP4 codes, E4M3 block scales and global amax are read from Transformer
Engine's own NVFP4Tensor, so the intermediate values are TE's, not a re-derivation.
GEMMs are re-accumulated in FP64 to keep TF32 out of the reference.

Outputs (in <seed dir>/layer_trace_<layer>/):
  flow.png                         computation steps of both pipelines
  r{0,1,8}_values.png              heatmaps of every step's full matrix
  r{1,8}_diff_vs_r0.png            element-wise % difference to the recipe-0 matrix
  r{1,8}_tile16.png                first 16x16 tile of each step with the numbers printed
  values/*.csv, diffs/*.csv        every matrix in full (rows x cols as in the figures)
  summary.csv                      per-step ranges and difference statistics
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "scripts"))

E2M1_LUT = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]
FP4_MAX, E4M3_MAX, BLOCK = 6.0, 448.0, 16
DIFF_KEYS = ["X_in", "X_op", "W_in", "W_op", "b", "Y_acc", "Y_bias", "Y_out"]
# Which models are traced: recipe -> (model recipe, checkpoint recipe)
TRACED = {0: (0, 0), 1: (1, 1), 8: (1, 0)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="swin")
    p.add_argument("--model-size", default="matched_500k")
    p.add_argument("--layer", default="blocks.0.attn.qkv")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sample-index", type=int, default=None,
                   help="Test-set index; default = slice with the largest tumor area")
    p.add_argument("--split", default="slice-random", choices=["patient-holdout", "slice-random"])
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16"])
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--data-path", default=str(ROOT / "lgg-mri-segmentation" / "kaggle_3m"))
    return p.parse_args()


# ───────────────────────────── capture ─────────────────────────────

def capture(args):
    import torch
    import main as runner
    from utils import create_dataloaders
    from plot_seed_scatter import find_test_metrics

    seed_dir = Path(args.results_dir) / "lgg" / args.split / "single" / args.arch / f"seed_{args.seed}"
    ckpts = {}
    for rid in (0, 1):
        path, _ = find_test_metrics(seed_dir / "metrics", rid, args)
        config_hash = path.stem.rsplit("_", 1)[-1]
        ckpts[rid], = seed_dir.glob(f"recipe_{rid}_{args.model_size}_{args.epochs}ep_*_{config_hash}.pt")

    df = runner.load_lgg_dataset_patient_aware(args.data_path)
    (train_df, val_df, test_df), = runner.build_splits(df, args.split, args.split_seed)
    _, _, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size=1)
    ds = test_loader.dataset
    idx = args.sample_index
    if idx is None:
        idx = int(np.argmax([float((ds[i][1] > 0).sum()) for i in range(len(ds))]))
    image, mask = ds[idx]
    sample = {"index": idx, "image_path": test_df.iloc[idx, 1], "tumor_pixels": int((mask > 0).sum())}

    factory = runner._get_model_factory(args.arch)
    device = torch.device("cuda")
    captured = {}
    for rid, (model_recipe, ckpt_recipe) in TRACED.items():
        model = factory(model_size=args.model_size, recipe_id=model_recipe).to(device).eval()
        state = torch.load(ckpts[ckpt_recipe], map_location="cpu", weights_only=False)["state_dict"]
        missing, _ = model.load_state_dict(state, strict=False)
        assert not [k for k in missing if not k.endswith("_extra_state")], missing
        layer = model.get_submodule(args.layer)
        rec = {}
        hook = layer.register_forward_hook(
            lambda m, inp, out: rec.update(x=inp[0].detach().clone(), y=out.detach().clone()))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(image[None].to(device))
        hook.remove()
        captured[rid] = {
            "x": rec["x"].reshape(-1, rec["x"].shape[-1]),
            "y": rec["y"].reshape(-1, rec["y"].shape[-1]),
            "w": layer.weight.detach().float().clone(),
            "b": layer.bias.detach().float().clone(),
            "x_dtype": str(rec["x"].dtype).replace("torch.", "").upper().replace("BFLOAT16", "BF16")
                                          .replace("FLOAT32", "FP32").replace("FLOAT16", "FP16"),
            "layer_type": type(layer).__module__ + "." + type(layer).__name__,
            "ckpt": ckpts[ckpt_recipe].name,
        }
        del model
    return captured, sample, seed_dir


# ───────────────────────────── step rebuild ─────────────────────────────

def nvfp4_steps(t, name):
    """TE NVFP4 1x16 RTN quantization of a 2D FP32 tensor, every intermediate exposed."""
    import torch
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    q = NVFP4Quantizer(rowwise=True, columnwise=False, with_rht=False, stochastic_rounding=False,
                       with_2d_quantization=False).quantize(t.contiguous())
    rows, cols = t.shape
    amax = q._amax_rowwise.float().item()
    s_enc = E4M3_MAX * FP4_MAX / amax
    scale = q._rowwise_scale_inv[:rows, :cols // BLOCK].view(torch.float8_e4m3fn).float()
    data = q._rowwise_data
    codes = torch.stack((data & 0xF, data >> 4), -1).reshape(rows, cols).long()
    fp4 = torch.tensor(E2M1_LUT, device=t.device)[codes]
    scale_full = scale.repeat_interleave(BLOCK, 1)
    deq = fp4 * scale_full / s_enc
    block_amax = t.float().view(rows, -1, BLOCK).abs().amax(-1)

    assert torch.equal((block_amax / FP4_MAX * s_enc).to(torch.float8_e4m3fn).float(), scale)
    ref = q.dequantize().float()  # TE dequantizes into the input dtype
    rtol = 2.0 ** -8 if t.dtype in (torch.bfloat16, torch.float16) else 1e-6
    assert bool(((deq - ref).abs() <= rtol * deq.abs() + 1e-6).all())
    scalars = {f"{name}_amax": amax, f"{name}_S_enc": s_enc}
    steps = {
        f"{name}_blk_amax": block_amax,
        f"{name}_scale": scale,
        f"{name}_scaled": t.float() * s_enc / scale_full,
        f"{name}_fp4": fp4,
        f"{name}_op": deq,
    }
    return steps, scalars


def gemm_steps(x_op, w_op, b_op):
    import torch
    acc = x_op.double() @ w_op.double().t()
    biased = acc + b_op.double()
    return {"Y_acc": acc, "Y_bias": biased, "Y_out": biased.to(torch.bfloat16).double()}


def rebuild(captured):
    import torch
    out, scalars, checks = {}, {}, {}
    for rid, c in captured.items():
        x, w, b = c["x"].float(), c["w"], c["b"]
        if rid == 0:
            s = {"X_in": x, "X_op": x.to(torch.bfloat16).float(), "W_in": w, "W_op": w.to(torch.bfloat16).float()}
        else:
            xs, xsc = nvfp4_steps(c["x"], "X")
            ws, wsc = nvfp4_steps(w, "W")
            s = {"X_in": x, **xs, "W_in": w, **ws}
            scalars[rid] = {**xsc, **wsc}
        s["b"] = b.to(torch.bfloat16).float()[None, :]
        s.update(gemm_steps(s["X_op"], s["W_op"], s["b"]))
        actual = c["y"].double()
        diff = (s["Y_out"] - actual).abs()
        bf16_ulp = 2.0 ** (torch.floor(torch.log2(actual.abs().clamp_min(1e-30))) - 7)
        checks[rid] = {
            "identical_frac": float((diff == 0).double().mean()),
            "within_1ulp_frac": float((diff <= bf16_ulp + 1e-12).double().mean()),
            "max_abs_diff": float(diff.max()),
            "rel_fro_pct": float(diff.norm() / actual.norm() * 100),
        }
        out[rid] = {k: v.double().cpu().numpy() for k, v in s.items()}
    return out, scalars, checks


# ───────────────────────────── metadata ─────────────────────────────

def step_specs(rid, x_dtype="FP32"):
    """(key, row, label, colormap kind) in plotting order."""
    if rid == 0:
        noop = " (no-op)" if x_dtype == "BF16" else ""
        return [
            ("X_in", "A", f"A0  X layer input ({x_dtype})", "signed"),
            ("X_op", "A", f"A1  X_op = BF16(X){noop}  [GEMM operand]", "signed"),
            ("W_in", "W", "W0  W master weight (FP32)", "signed"),
            ("W_op", "W", "W1  W_op = BF16(W)  [GEMM operand]", "signed"),
            ("b", "Y", "G0  b = BF16(bias)", "signed"),
            ("Y_acc", "Y", "G1  Y_acc = X_op @ W_op^T  (FP32 accum)", "signed"),
            ("Y_bias", "Y", "G2  Y_acc + b", "signed"),
            ("Y_out", "Y", "G3  Y = BF16(Y_acc + b)  [layer output]", "signed"),
        ]
    specs = []
    for op, row, src in (("X", "A", f"layer input ({x_dtype})"), ("W", "W", "master weight (FP32)")):
        tag = "A" if op == "X" else "W"
        specs += [
            (f"{op}_in", row, f"{tag}0  {op} {src}", "signed"),
            (f"{op}_blk_amax", row, f"{tag}1  block amax |{op}|  (1x16 blocks)", "positive"),
            (f"{op}_scale", row, f"{tag}2  s = E4M3(blk_amax / 6 * S_enc)  [read by GEMM]", "positive"),
            (f"{op}_scaled", row, f"{tag}3  {op} * S_enc / s  (before rounding)", "signed"),
            (f"{op}_fp4", row, f"{tag}4  q = E2M1 RTN  {{0, ±0.5 ... ±6}}  [read by GEMM]", "signed"),
            (f"{op}_op", row, f"{tag}5  {op}q = q*s/S_enc  [equiv. view, not in TE]", "signed"),
        ]
    specs += [
        ("b", "Y", "G0  b = BF16(bias)", "signed"),
        ("Y_acc", "Y", "G1  Y_acc  (FP4 GEMM on q, s; FP32 accum)", "signed"),
        ("Y_bias", "Y", "G2  Y_acc + b", "signed"),
        ("Y_out", "Y", "G3  Y = BF16(Y_acc + b)  [layer output]", "signed"),
    ]
    return specs


RECIPE_TITLES = {0: "Recipe 0 Baseline (BF16)", 1: "Recipe 1 NVFP4 Full (QAT weights)",
                 8: "Recipe 8 NVFP4 PTQ RTN (Baseline weights)"}


def pct_diff(v, ref):
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (v - ref) / np.abs(ref) * 100
    pct[(ref == 0) & (v == 0)] = 0.0
    return pct


def diff_stats(v, ref):
    pct = pct_diff(v, ref)
    finite = np.abs(pct[np.isfinite(pct)])
    return {
        "rel_fro_pct": float(np.linalg.norm(v - ref) / np.linalg.norm(ref) * 100),
        "median_abs_pct": float(np.median(finite)),
        "p95_abs_pct": float(np.percentile(finite, 95)),
        "max_abs_diff": float(np.abs(v - ref).max()),
        "identical_frac": float((v == ref).mean()),
    }


# ───────────────────────────── plots ─────────────────────────────

def heat(ax, arr, title, kind, vmax=None):
    if kind == "positive":
        im = ax.imshow(arr, aspect="auto", cmap="viridis", interpolation="nearest")
    else:
        vmax = vmax or (np.nanmax(np.abs(arr)) or 1.0)
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)


def layout(specs):
    rows = {r: [s for s in specs if s[1] == r] for r in ("A", "W", "Y")}
    return rows, max(len(v) for v in rows.values())


def plot_values(rid, steps, scalars, args, sample, out, x_dtype):
    rows, ncols = layout(step_specs(rid, x_dtype))
    fig, axes = plt.subplots(3, ncols, figsize=(3.6 * ncols, 12.5), squeeze=False)
    for r, row in enumerate(("A", "W", "Y")):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(rows[row]):
                ax.axis("off")
                continue
            key, _, label, kind = rows[row][c]
            arr = steps[key]
            extra = f"\n{arr.shape[0]}x{arr.shape[1]}  min {arr.min():.4g}  max {arr.max():.4g}"
            if key.endswith("_blk_amax") and rid in scalars:
                op = key[0]
                extra += (f"\nglobal amax {scalars[rid][f'{op}_amax']:.5g}  ->  "
                          f"S_enc = 448*6/amax = {scalars[rid][f'{op}_S_enc']:.5g}")
            heat(ax, arr, label + extra, kind)
    fig.suptitle(f"{RECIPE_TITLES[rid]} | Swin {args.model_size} {args.layer} | seed {args.seed} | "
                 f"test slice #{sample['index']} ({Path(sample['image_path']).name})\n"
                 f"rows: A = activation path, W = weight path, Y = GEMM / output. "
                 f"X: {steps['X_in'].shape[0]} tokens x {steps['X_in'].shape[1]} input features, "
                 f"W: {steps['W_in'].shape[0]} output x {steps['W_in'].shape[1]} input features",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"r{rid}_values.png", dpi=150)
    plt.close(fig)


def plot_diffs(rid, steps, ref, stats, args, out, x_dtype):
    specs = [s for s in step_specs(rid, x_dtype) if s[0] in DIFF_KEYS]
    rows, ncols = layout(specs)
    fig, axes = plt.subplots(3, ncols, figsize=(4.2 * ncols, 12.5), squeeze=False)
    for r, row in enumerate(("A", "W", "Y")):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(rows[row]):
                ax.axis("off")
                continue
            key, _, label, _ = rows[row][c]
            pct = pct_diff(steps[key], ref[key])
            st = stats[(rid, key)]
            heat(ax, np.clip(pct, -100, 100), f"{label}\nvs recipe 0: rel. Frobenius {st['rel_fro_pct']:.2f}%  "
                 f"median |%| {st['median_abs_pct']:.2f}  identical {st['identical_frac'] * 100:.1f}%",
                 "signed", vmax=100)
    fig.suptitle(f"{RECIPE_TITLES[rid]} minus Recipe 0, element-wise (v - v_r0) / |v_r0| x 100%  "
                 f"(color clipped to ±100%; gray = recipe-0 value is 0)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"r{rid}_diff_vs_r0.png", dpi=150)
    plt.close(fig)


def plot_tile(rid, all_steps, args, out, n=16):
    s, r0 = all_steps[rid], all_steps[0]
    rows = [
        [(r0, "X_in", "r0 A0 X (input)"), (r0, "X_op", "r0 A1 BF16(X)"), (s, "X_in", f"r{rid} A0 X (input)"),
         (s, "X_scaled", f"r{rid} A3 X*S_enc/s"), (s, "X_fp4", f"r{rid} A4 q (E2M1)"),
         (s, "X_op", f"r{rid} A5 Xq"), ("pct", "X_op", f"r{rid} Xq vs r0 BF16(X), %")],
        [(r0, "W_in", "r0 W0 W (FP32)"), (r0, "W_op", "r0 W1 BF16(W)"), (s, "W_in", f"r{rid} W0 W (FP32)"),
         (s, "W_scaled", f"r{rid} W3 W*S_enc/s"), (s, "W_fp4", f"r{rid} W4 q (E2M1)"),
         (s, "W_op", f"r{rid} W5 Wq"), ("pct", "W_op", f"r{rid} Wq vs r0 BF16(W), %")],
        [(r0, "Y_acc", "r0 G1 Y_acc"), (r0, "Y_out", "r0 G3 Y (BF16)"), (s, "Y_acc", f"r{rid} G1 Y_acc"),
         (s, "Y_bias", f"r{rid} G2 Y_acc+b"), (s, "Y_out", f"r{rid} G3 Y (BF16)"),
         ("pct", "Y_acc", f"r{rid} Y_acc vs r0, %"), ("pct", "Y_out", f"r{rid} Y vs r0 Y, %")],
    ]
    fig, axes = plt.subplots(3, 7, figsize=(7 * 5.0, 3 * 5.2))
    for r, row in enumerate(rows):
        for c, (src, key, title) in enumerate(row):
            ax = axes[r, c]
            if src == "pct":
                arr = pct_diff(s[key], r0[key])[:n, :n]
                shown = np.clip(arr, -100, 100)
                vmax = 100
            else:
                arr = src[key][:n, :n]
                shown, vmax = arr, (np.abs(arr).max() or 1.0)
            ax.imshow(shown, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    txt = "nan" if not np.isfinite(v) else (f"{v:.0f}" if src == "pct" else f"{v:.3g}")
                    dark = np.isfinite(shown[i, j]) and abs(shown[i, j]) > 0.65 * vmax
                    ax.text(j, i, txt, ha="center", va="center", fontsize=4.6, color="white" if dark else "black")
            ax.set_title(title, fontsize=9, loc="left")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.tick_params(labelsize=5)
            if key in ("X_scaled", "W_scaled") and src is not r0:
                sc = s[key[0] + "_scale"][:n, 0]
                ax.set_yticks(range(n), [f"s={v:.3g}" for v in sc], fontsize=5)
    fig.suptitle(f"{RECIPE_TITLES[rid]} vs Recipe 0 | first {n}x{n} tile (rows 0-{n - 1}, columns 0-{n - 1} = "
                 f"exactly one NVFP4 block per row) | X/W rows: tokens / output channels, Y columns: output features",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"r{rid}_tile16.png", dpi=200)
    plt.close(fig)


def plot_flow(out, args):
    fig, ax = plt.subplots(figsize=(18, 7.5))
    ax.axis("off")

    def box(x, y, text, color, w=0.205, h=0.13):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=color, ec="0.3", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2))

    ax.text(0.0, 0.97, "Recipe 0 (Baseline): nn.Linear under torch.autocast(bfloat16)", fontsize=11, fontweight="bold")
    box(0.00, 0.72, "A0  X  layer input\n(FP32 or BF16)", "#d6eaf8")
    box(0.26, 0.72, "A1  X_op = BF16(X)", "#d6eaf8")
    box(0.00, 0.54, "W0  W  (FP32 master)", "#fdebd0")
    box(0.26, 0.54, "W1  W_op = BF16(W)", "#fdebd0")
    box(0.52, 0.63, "G1  Y_acc = X_op @ W_op^T\nBF16 GEMM, FP32 accumulate", "#d5f5e3")
    box(0.78, 0.63, "G2/G3  Y = BF16(Y_acc + BF16(b))\nlayer output", "#d5f5e3")
    for y in (0.785, 0.605):
        arrow(0.205, y, 0.26, y)
    arrow(0.465, 0.785, 0.52, 0.72)
    arrow(0.465, 0.605, 0.52, 0.66)
    arrow(0.725, 0.695, 0.78, 0.695)

    ax.text(0.0, 0.44, "Recipe 1 / 8 (NVFP4): te.Linear under NVFP4BlockScaling(1x16 blocks, RTN, no RHT, no SR)",
            fontsize=11, fontweight="bold")
    steps = ["0  X (FP32)", "1  global amax -> S_enc = 448*6/amax\nblock amax over 1x16",
             "2  s = E4M3(blk_amax/6*S_enc)", "3  X*S_enc/s\n(in [-6, 6])",
             "4  q = RTN to E2M1\n{0,±0.5,±1,±1.5,±2,±3,±4,±6}", "5  Xq = q*s/S_enc\n(equivalent view)"]
    widths, gap = [0.08, 0.175, 0.15, 0.09, 0.17, 0.1], 0.016
    xs = np.concatenate(([0.0], np.cumsum(np.array(widths[:-1]) + gap)))
    for op, color, y in (("A", "#d6eaf8", 0.22), ("W", "#fdebd0", 0.02)):
        for i, (x, w, t) in enumerate(zip(xs, widths, steps)):
            box(x, y, op + t.replace("X", "X" if op == "A" else "W"), color, w=w, h=0.15)
            if i:
                arrow(xs[i - 1] + widths[i - 1], y + 0.075, x, y + 0.075)
    gx = xs[-1] + widths[-1] + 0.03
    box(gx, 0.09, "G1  FP4 GEMM reads q, s:\nY_acc = sum_b s_x*s_w*(q_x.q_w)\n* 1/(S_x*S_w), FP32 accum\n"
        "G2/G3  Y = BF16(Y_acc + b)", "#d5f5e3",
        w=1.0 - gx, h=0.22)
    arrow(gx - 0.03, 0.295, gx, 0.25)
    arrow(gx - 0.03, 0.095, gx, 0.14)
    ax.text(0.0, -0.06, "Recipe 1 uses its own QAT-trained weights; recipe 8 loads the recipe-0 (Baseline) weights into "
            "the same te.Linear layers, so its inputs match recipe 0 only up to the first quantized layer. "
            "Xq/Wq are an equivalent view: the GEMM consumes q and s directly.",
            fontsize=9, color="0.25")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.08, 1.0)
    fig.savefig(out / "flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────── main ─────────────────────────────

def main():
    args = parse_args()
    captured, sample, seed_dir = capture(args)
    steps, scalars, checks = rebuild(captured)
    out = seed_dir / f"layer_trace_{args.layer}"
    (out / "values").mkdir(parents=True, exist_ok=True)
    (out / "diffs").mkdir(exist_ok=True)

    print(f"Sample: test #{sample['index']} {sample['image_path']} (tumor pixels {sample['tumor_pixels']})")
    for rid, c in captured.items():
        print(f"recipe {rid}: {c['layer_type']} from {c['ckpt']} | X {tuple(c['x'].shape)} {c['x'].dtype} "
              f"| Y {tuple(c['y'].shape)} {c['y'].dtype}")
        print(f"  rebuilt output vs actual layer output: identical {checks[rid]['identical_frac'] * 100:.2f}%, "
              f"within 1 BF16 ulp {checks[rid]['within_1ulp_frac'] * 100:.2f}%, "
              f"max |diff| {checks[rid]['max_abs_diff']:.3g}, rel {checks[rid]['rel_fro_pct']:.4f}%")
        if rid in scalars:
            print("  " + ", ".join(f"{k}={v:.6g}" for k, v in scalars[rid].items()))

    rows, stats = [], {}
    for rid in TRACED:
        for key, _, label, _ in step_specs(rid, captured[rid]["x_dtype"]):
            arr = steps[rid][key]
            np.savetxt(out / "values" / f"r{rid}_{key}.csv", arr, delimiter=",", fmt="%.9g")
            row = {"recipe": rid, "step": key, "label": label, "shape": "x".join(map(str, arr.shape)),
                   "min": arr.min(), "max": arr.max(), "mean_abs": np.abs(arr).mean()}
            if rid != 0 and key in DIFF_KEYS:
                st = diff_stats(arr, steps[0][key])
                stats[(rid, key)] = st
                row.update(st)
                np.savetxt(out / "diffs" / f"r{rid}_{key}_pct_vs_r0.csv", pct_diff(arr, steps[0][key]),
                           delimiter=",", fmt="%.6g")
            rows.append(row)
    import pandas as pd
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[summary.recipe != 0].dropna(subset=["rel_fro_pct"])[
            ["recipe", "step", "rel_fro_pct", "median_abs_pct", "p95_abs_pct", "identical_frac"]].to_string(index=False))

    plot_flow(out, args)
    for rid in TRACED:
        plot_values(rid, steps[rid], scalars, args, sample, out, captured[rid]["x_dtype"])
    for rid in (1, 8):
        plot_diffs(rid, steps[rid], steps[0], stats, args, out, captured[rid]["x_dtype"])
        plot_tile(rid, steps, args, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
