"""GemLite-backed int4 Linear for packed AutoGPTQ-v1 weights.

GemLite (https://github.com/mobiusml/gemlite) ships production-tuned Triton
int4 GEMM kernels that, on Blackwell, run within ~15% of bf16 cuBLAS while
keeping weights in int4 (≈1/3 VRAM). Unlike the HQQ-based ``create_gemlite``
helper, this packs OneCompression's *exact* GPTQ scale / zero / int weights
directly, so the dequantized result matches the GPTQ checkpoint to fp16
rounding (~5e-4 relative error) instead of re-quantizing.

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import torch
from torch import nn

from .quant_utils import unpack_int_weights, unpack_zeros


def gemlite_available() -> bool:
    try:
        import gemlite  # noqa: F401
        return True
    except Exception:
        return False


class GemLiteInt4Linear(nn.Module):
    """Drop-in int4 Linear that dispatches to a GemLite Triton GEMM.

    Built from the packed buffers of a 4-bit / groupsize=32 GPTQ layer
    (AutoGPTQ-v1 format). Input/output run in fp16 internally (GemLite
    requirement); the wrapper casts to/from the caller's dtype.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: torch.Tensor | None = None,
        groupsize: int = 32,
        v1: bool = True,
        device: torch.device | str = "cuda:0",
    ):
        super().__init__()
        from gemlite.core import DType, GemLiteLinearTriton

        self.in_features = in_features
        self.out_features = out_features
        dev = torch.device(device)
        self._device = dev

        # (out, in) uint8 int weights, (out, num_groups) scales / zeros.
        weight_int = unpack_int_weights(
            qweight.to(dev), 4, (out_features, in_features)
        ).to(torch.uint8)
        zeros = unpack_zeros(qzeros.to(dev), 4, out_features)
        if v1:
            zeros = (zeros + 1) & 0xF
        scales_t = scales.to(device=dev, dtype=torch.float16).t().contiguous()
        zeros_t = zeros.to(dtype=torch.float16).t().contiguous()

        gl = GemLiteLinearTriton(
            W_nbits=4,
            group_size=groupsize,
            in_features=in_features,
            out_features=out_features,
            input_dtype=DType.FP16,
            output_dtype=DType.FP16,
        )
        gl.pack(weight_int, scales_t, zeros_t, bias=None)
        self._gl = gl.to(dev)

        if bias is not None:
            self.register_buffer("bias", bias.to(device=dev, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        out = self._gl(x.to(torch.float16))
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.to(orig_dtype)

    @torch.no_grad()
    def warmup(self, m_values=(64, 128, 256, 1024, 4096)) -> None:
        dev = self._device
        for m in m_values:
            if m <= 0:
                continue
            x = torch.zeros((int(m), self.in_features), dtype=torch.float16, device=dev)
            _ = self.forward(x)
