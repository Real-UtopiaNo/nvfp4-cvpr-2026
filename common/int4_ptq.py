"""
INT4 W4A4 post-training quantization (simulated).

Symmetric round-to-nearest INT4 in [-7, 7] with FP32 scales, applied along the
GEMM reduction dim K. ``group_size=None`` gives one scale per row (weights: per
output channel, activations: per token), the standard INT4 GEMM layout whose
scales factor out of the K loop. ``group_size=16`` gives one scale per 16
elements, the same block geometry as NVFP4.

Quantize -> dequantize -> FP32 GEMM matches an integer GEMM with FP32 rescaling,
since the integer products accumulate exactly.
"""
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

INT4_QMAX = 7


def int4_fake_quant(t: torch.Tensor, group_size: Optional[int] = None) -> torch.Tensor:
    """Symmetric RTN INT4 along the last dim; FP32 result on the INT4 grid."""
    shape = t.shape
    t = t.float()
    if group_size:
        if shape[-1] % group_size:
            raise ValueError(f"Last dim {shape[-1]} is not divisible by group size {group_size}")
        t = t.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    scale = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / INT4_QMAX
    return ((t / scale).round().clamp(-INT4_QMAX, INT4_QMAX) * scale).reshape(shape)


class Int4FakeQuantLinear(nn.Module):
    """W4A4 INT4 Linear: weights quantized once, activations quantized per forward call."""

    def __init__(self, linear: nn.Linear, group_size: Optional[int] = None):
        super().__init__()
        self.group_size = group_size
        self.register_buffer("weight_q", int4_fake_quant(linear.weight.detach(), group_size))
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        with torch.autocast("cuda", enabled=False):
            y = F.linear(int4_fake_quant(x, self.group_size), self.weight_q,
                         None if self.bias is None else self.bias.float())
        return y.to(out_dtype)


def apply_int4_ptq(model: nn.Module, layer_names: Iterable[str], group_size: Optional[int] = None) -> int:
    """Replace the named ``nn.Linear`` submodules with ``Int4FakeQuantLinear``; returns the count."""
    count = 0
    for name in layer_names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        linear = getattr(parent, child)
        if type(linear) is not nn.Linear:
            raise TypeError(f"{name} is {type(linear).__name__}, expected nn.Linear")
        setattr(parent, child, Int4FakeQuantLinear(linear, group_size))
        count += 1
    return count
