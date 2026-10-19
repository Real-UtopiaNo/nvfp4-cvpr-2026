#!/usr/bin/env python3
"""
Transformer Engine NVFP4 smoke test
===================================
1. Layer check: the repo's ``quantized_linear`` path (te.Linear under the NVFP4
   Full recipe) runs forward + backward, under both an FP16 and a BF16 outer
   autocast, and is really quantized: errors against an FP32 nn.Linear reference
   should be a few percent, far above FP16/BF16 rounding (~1e-3).
2. Model check: Swin matched_500k with recipe 1 loads the recipe-0 weights and
   runs a few training steps the same way ``train_single_recipe`` does, with
   FP16 (+GradScaler) and BF16 (no loss scaling) autocast.

Usage (inside the NGC container):
    python scripts/te_nvfp4_smoke.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "vit", "swin", "common"):
    sys.path.insert(0, str(ROOT / sub))

import transformer_engine  # noqa: E402
from vit_qat import (  # noqa: E402
    NVFPQuantizer, make_linear, quantized_linear, available_qat_recipes, TE_RECIPE_KWARGS,
)
from swin_qat import create_swin_qat  # noqa: E402
from loss import tversky_loss  # noqa: E402

# Swin matched_500k Linear shapes (in, out); M = batch 16 x 256 tokens
SHAPES = [(64, 192), (64, 64), (192, 64)]
M = 16 * 256


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def layer_case(in_f, out_f, quantizer, amp_dtype):
    torch.manual_seed(0)
    ref = nn.Linear(in_f, out_f).cuda()
    lin = make_linear(in_f, out_f, quantizer=quantizer).cuda()
    lin.load_state_dict(ref.state_dict(), strict=False)

    x = torch.randn(M, in_f, device="cuda")
    g = torch.randn(M, out_f, device="cuda")

    x_ref = x.clone().requires_grad_(True)
    ref(x_ref).backward(g)

    x_q = x.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=amp_dtype):
        y_q = quantized_linear(x_q, lin, quantizer)
    y_q.float().backward(g)

    with torch.no_grad():
        y_ref = ref(x)
    return {
        "y": rel_err(y_q, y_ref),
        "dx": rel_err(x_q.grad, x_ref.grad),
        "dw": rel_err(lin.weight.grad, ref.weight.grad),
        "db": rel_err(lin.bias.grad, ref.bias.grad),
    }


def model_check(amp_dtype, steps=5, batch=16):
    torch.manual_seed(2026)
    base = create_swin_qat("matched_500k", recipe_id=0)
    model = create_swin_qat("matched_500k", recipe_id=1).cuda()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    real_missing = [k for k in missing if not k.endswith("_extra_state")]
    assert not real_missing and not unexpected, (real_missing, unexpected)
    n_te = sum(type(m).__module__.startswith("transformer_engine") for m in model.modules())
    print(f"  Swin 500K recipe 1: {n_te} te.Linear layers, baseline weights loaded")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    x = torch.randn(batch, 3, 256, 256, device="cuda")
    y = (torch.rand(batch, 1, 256, 256, device="cuda") > 0.97).float()

    model.train()
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), y) + tversky_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        print(f"  step {step}: loss={loss.item():.4f} grad_norm={gn.item():.4f} "
              f"logits={logits.dtype} {(time.time() - t0) * 1000:.0f} ms")
        assert torch.isfinite(loss), "non-finite loss"

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
        out = model(x[:9])
    assert out.shape == (9, 1, 256, 256)
    print("  eval forward with odd batch (9): OK")


def main():
    cc = torch.cuda.get_device_capability()
    print(f"torch {torch.__version__} | TE {transformer_engine.__version__} | "
          f"{torch.cuda.get_device_name()} (sm_{cc[0]}{cc[1]})")
    print(f"Runnable QAT recipes here: {sorted(available_qat_recipes())}")
    print(f"Recipe 1 = NVFP4BlockScaling({TE_RECIPE_KWARGS[1]})\n")

    quantizer = NVFPQuantizer(1)
    ok = True
    print("Layer check (outer autocast -> quantized_linear):")
    for amp_dtype in (torch.float16, torch.bfloat16):
        for in_f, out_f in SHAPES:
            try:
                e = layer_case(in_f, out_f, quantizer, amp_dtype)
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"  [FAIL] {str(amp_dtype):<15} {in_f:>4}->{out_f:<4} {type(exc).__name__}: {exc}")
                continue
            quantized = min(e["y"], e["dx"], e["dw"]) > 1e-2
            ok &= quantized
            tag = "OK  " if quantized else "WARN"
            print(f"  [{tag}] {str(amp_dtype):<15} {in_f:>4}->{out_f:<4} "
                  f"rel_err y={e['y']:.4f} dx={e['dx']:.4f} dw={e['dw']:.4f} db={e['db']:.4f}")

    for amp_dtype in (torch.float16, torch.bfloat16):
        print(f"\nModel check ({amp_dtype}):")
        try:
            model_check(amp_dtype)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  [FAIL] {type(exc).__name__}: {exc}")

    print("\nNVFP4 Full path works." if ok else
          "\nSomething is off: see failures above, or errors too small to be NVFP4 (quantization not applied).")


if __name__ == "__main__":
    main()
