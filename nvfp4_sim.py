"""BF16-simulated NVFP4 quantization runtime (research implementation).

Implements the NVFP4 numerics used by the paper's QAT recipes:

    value = q(E2M1) * s(E4M3, per-16 block) * g(FP32, per tensor)

- q(E2M1)   : 4-bit float, values {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}
- s(E4M3)   : FP8 scale per block of 16 elements along a chosen axis
- g(FP32)   : tensor-wide scale, amax-based: amax / (E4M3_MAX * E2M1_MAX)

Quantization is simulated: tensors are quantized then immediately dequantized
back to the input dtype, and all arithmetic runs on standard BF16/FP32
operators. This reproduces the *quantization error semantics* of the paper's
recipes (precision experiments, slides pp. 12-14) without requiring native FP4
hardware kernels. It does NOT reproduce FP4 speedups (slides p. 24).

Design decisions (explicit assumptions where the paper's omitted runtime is
underspecified):
- Global scale is computed from the current tensor's amax (no history/EMA).
- Block scales are rounded through a real E4M3 cast (torch.float8_e4m3fn),
  saturating at 448; they are additionally clamped to the smallest E4M3
  subnormal (2^-9) to avoid divide-by-zero on near-empty blocks.
- E2M1 rounding is round-to-nearest with ties resolved by torch.bucketize
  on grid midpoints (ties go up). Saturating at |6|.
- Blocks containing fewer than 16 elements (tail) are zero-padded first;
  the pad does not affect amax and is discarded afterwards.
- Quantization can be applied along any axis; the backward GEMMs of the
  "NVFP4 Full" recipe require quantizing operands along their reduction axis
  (Q(W^T) != Q(W)^T for 1-D block scaling), which this module supports via
  the `axis` argument.

This module has no GPU-specific requirements and runs on CPU or CUDA.
"""
from __future__ import annotations

import torch

E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0 ** -9
BLOCK = 16


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round a float tensor to the nearest E2M1-representable value (saturating).

    Ties at grid midpoints round up (bucketize default). Values with |x| >= 5
    saturate at +-6.
    """
    grid = torch.tensor(E2M1_VALUES, device=x.device, dtype=x.dtype)
    mids = (grid[1:] + grid[:-1]) / 2.0  # [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    idx = torch.bucketize(x.abs().clamp(max=E2M1_MAX), mids)
    return x.sign() * grid[idx]


def _cast_e4m3(s: torch.Tensor) -> torch.Tensor:
    """Round scale factors through a real E4M3 cast, then clamp to the valid range."""
    s = s.clamp(max=E4M3_MAX)
    s = s.to(torch.float8_e4m3fn).float()
    return s.clamp(min=E4M3_MIN_SUBNORMAL)


def nvfp4_qdq(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Simulated NVFP4 quantize-dequantize.

    Quantizes `x` along `axis` in blocks of 16 elements using per-block E4M3
    scales and a per-tensor FP32 scale, then dequantizes back to `x.dtype`.
    The returned tensor has the same shape and dtype as the input.
    """
    if not x.is_floating_point():
        raise TypeError(f"nvfp4_qdq expects a float tensor, got {x.dtype}")

    in_dtype = x.dtype
    axis = axis % x.dim()

    # Work in FP32 with the quantization axis moved to the end.
    xf = x.detach().float().movedim(axis, -1).contiguous()
    lead_shape = xf.shape[:-1]
    k = xf.shape[-1]

    pad = (-k) % BLOCK
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))

    blocks = xf.reshape(-1, (k + pad) // BLOCK, BLOCK)          # (L, nblk, 16)

    # Per-tensor FP32 global scale (amax-based).
    amax = blocks.abs().amax()
    global_scale = (amax / (E4M3_MAX * E2M1_MAX)).clamp(min=torch.finfo(torch.float32).tiny)

    # Per-block E4M3 scale.
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)         # (L, nblk, 1)
    block_scale = _cast_e4m3(block_amax / (E2M1_MAX * global_scale))

    # Quantize to the E2M1 grid and dequantize.
    scaled = blocks / (block_scale * global_scale)
    q = round_to_e2m1(scaled)
    deq = q * block_scale * global_scale

    deq = deq.reshape(*lead_shape, k + pad)
    if pad:
        deq = deq[..., :k]
    return deq.movedim(-1, axis).to(in_dtype)


# ─────────────────────────── self test ───────────────────────────

def _self_test() -> None:
    torch.manual_seed(0)
    print("nvfp4_sim self test")

    # 1) zero tensor stays zero, no NaN/Inf
    z = torch.zeros(4, 32)
    out = nvfp4_qdq(z)
    assert torch.equal(out, z), "zero tensor must stay zero"
    assert torch.isfinite(out).all()

    # 2) quantized values live on the E2M1 grid (relative to their scales)
    x = torch.randn(8, 64) * 3
    out = nvfp4_qdq(x)
    grid = torch.tensor(E2M1_VALUES)
    amax = x.abs().amax()
    g = amax / (E4M3_MAX * E2M1_MAX)
    blocks_x = x.view(8, 4, 16)
    bs = _cast_e4m3(blocks_x.abs().amax(-1, keepdim=True) / (E2M1_MAX * g))
    ratio = (out.view(8, 4, 16) / (bs * g)).abs().reshape(-1)
    uniq = torch.unique(ratio.round(decimals=6))
    bad = [u for u in uniq.tolist() if u > 1e-6 and min(abs(u - v) for v in E2M1_VALUES) > 1e-4]
    assert not bad, f"values off grid: {bad[:5]}"
    print(f"  [ok] grid check, {len(uniq)} distinct ratios on grid")

    # 3) quantization actually perturbs the tensor (MSE > 0) but stays bounded
    mse = (out - x).pow(2).mean().item()
    rel = ((out - x).norm() / x.norm()).item()
    assert mse > 0, "quantization had no effect"
    assert rel < 0.5, f"relative error too large: {rel}"
    print(f"  [ok] MSE = {mse:.4e}, relative error = {rel:.4f}")

    # 4) tail blocks (length not divisible by 16) work
    xt = torch.randn(3, 100)
    out_t = nvfp4_qdq(xt)
    assert out_t.shape == xt.shape and torch.isfinite(out_t).all()
    print("  [ok] tail padding (K=100)")

    # 5) axis argument: quantizing along dim 0 differs from dim 1 result
    w = torch.randn(32, 48)
    o0 = nvfp4_qdq(w, axis=0)
    o1 = nvfp4_qdq(w, axis=-1)
    assert not torch.allclose(o0, o1), "axis must change the quantization"
    o0t = nvfp4_qdq(w.t().contiguous(), axis=-1).t()
    assert torch.equal(o0, o0t), "axis=0 must equal transposed axis=-1"
    print("  [ok] axis semantics (Q(W^T) path consistent)")

    # 6) dtype preserved
    xb = torch.randn(4, 32, dtype=torch.bfloat16)
    assert nvfp4_qdq(xb).dtype == torch.bfloat16
    print("  [ok] dtype preserved (bf16)")

    # 7) extreme dynamic range: no NaN/Inf (block-scale clamp active)
    xe = torch.zeros(1, 64)
    xe[0, 0] = 1e6
    out_e = nvfp4_qdq(xe)
    assert torch.isfinite(out_e).all()
    print("  [ok] extreme amax edge case")

    print("all self tests passed")


if __name__ == "__main__":
    _self_test()
