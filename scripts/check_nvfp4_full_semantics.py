#!/usr/bin/env python3
"""
Does te.Linear under our recipe 1 implement the paper's NVFP4 Full exactly?
===========================================================================
Reference emulation of the paper's data flow (Sec. 3.3-3.4, Fig. 2, Table 2):

    FPROP  Y  = Q_row(X)   . Q_row(W)^T          blocks along K
    DGRAD  dX = Q_row(G)   . Q_col(W)            blocks along N, W re-quantized
    WGRAD  dW = Q_col(G)^T . Q_col(X)            blocks along M, X re-quantized

Q = NVFP4: E2M1 values, round-to-nearest-even, 1x16 blocks with an E4M3 block
scale, FP32 per-tensor scale (no 2D, no RHT, no SR). TE's results are compared
against this and against neighbouring recipes; NVFP4 Full should be the only
near match (~1%, from rounding-boundary flips; the others differ by >10%).

Usage (inside the NGC container):
    python scripts/check_nvfp4_full_semantics.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for sub in ("vit", "common"):
    sys.path.insert(0, str(ROOT / sub))

from vit_qat import NVFPQuantizer, make_linear, quantized_linear  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid, nearest with ties to even mantissa, saturating at 6."""
    grid, mid = E2M1.to(v.device), MIDPOINTS.to(v.device)
    a = v.abs().clamp(max=6.0)
    lo = torch.bucketize(a, mid, right=False)
    hi = torch.bucketize(a, mid, right=True)
    idx = torch.where(lo % 2 == 0, lo, hi)  # grid indices 0,2,4,6 have mantissa 0
    return torch.sign(v) * grid[idx]


def nvfp4(x: torch.Tensor, block: int = 16, fp8_scale: bool = True) -> torch.Tensor:
    """Fake-quantize along the last dim (blocks of 16) and dequantize to FP32."""
    x = x.float()
    s_enc = (6.0 * 448.0) / x.abs().amax().clamp_min(1e-30)
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    s_dec = xb.abs().amax(-1, keepdim=True) / 6.0 * s_enc
    if fp8_scale:
        s_dec = s_dec.to(torch.float8_e4m3fn).float()
    safe = torch.where(s_dec == 0, torch.ones_like(s_dec), s_dec)
    q = round_e2m1(xb * (s_enc / safe))
    return (q * safe / s_enc).reshape(x.shape)


def q_row(t):  # blocks along the last dim
    return nvfp4(t)


def q_col(t):  # blocks along dim 0
    return nvfp4(t.t().contiguous()).t()


def q_row_noscale(t):
    return nvfp4(t, fp8_scale=False)


def q_col_noscale(t):
    return nvfp4(t.t().contiguous(), fp8_scale=False).t()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def check(M, K, N, seed=0):
    torch.manual_seed(seed)
    quantizer = NVFPQuantizer(1)
    lin = make_linear(K, N, quantizer=quantizer).cuda()
    x = torch.randn(M, K, device="cuda").bfloat16()
    g = torch.randn(M, N, device="cuda").bfloat16()

    xq = x.clone().requires_grad_(True)
    y_te = quantized_linear(xq, lin, quantizer)
    y_te.backward(g)
    dx_te, dw_te = xq.grad, lin.weight.grad

    # TE quantizes the FP32 master weight directly, not a BF16 copy of it.
    W, b, X, G = lin.weight.detach().float(), lin.bias.detach().float(), x.float(), g.float()
    Wf = W
    cands = {
        "NVFP4 Full (paper)": (
            q_row(X) @ q_row(W).t() + b,
            q_row(G) @ q_col(W),
            q_col(G).t() @ q_col(X),
        ),
        "Full, no FP8 block scale": (
            q_row_noscale(X) @ q_row_noscale(W).t() + b,
            q_row_noscale(G) @ q_col_noscale(W),
            q_col_noscale(G).t() @ q_col_noscale(X),
        ),
        "Full, bwd reuses fwd Q(W),Q(X)": (
            q_row(X) @ q_row(W).t() + b,
            q_row(G) @ q_row(W),
            q_col(G).t() @ q_row(X),
        ),
        "Chain Rule (G in BF16)": (
            q_row(X) @ q_row(W).t() + b,
            G @ q_row(W),
            G.t() @ q_row(X),
        ),
        "Fwd-Only": (
            q_row(X) @ q_row(W).t() + b,
            G @ Wf,
            G.t() @ X,
        ),
        "BF16 (no quant)": (X @ Wf.t() + b, G @ Wf, G.t() @ X),
    }
    print(f"\nM={M} K={K} N={N}   rel. difference of TE vs each candidate")
    print(f"  {'candidate':<32} {'Y':>8} {'dX':>8} {'dW':>8}")
    for name, (y, dx, dw) in cands.items():
        print(f"  {name:<32} {rel(y_te, y):8.4f} {rel(dx_te, dx):8.4f} {rel(dw_te, dw):8.4f}")


def main():
    for M, K, N in [(4096, 64, 192), (4096, 64, 64), (4096, 192, 64)]:
        check(M, K, N)


if __name__ == "__main__":
    main()
