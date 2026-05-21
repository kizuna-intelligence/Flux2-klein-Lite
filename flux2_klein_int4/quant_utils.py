"""Minimal AutoGPTQ-v1 4-bit unpack + dequant helpers.

Vendored from OneCompression's onecomp.quantizer.gptq.gptq_layer. Only the
4-bit code paths are kept — this runtime ships int4 checkpoints exclusively.
Used for the eager-dequant fallback when a layer cannot run on the fused
Triton kernel (e.g. groupsize != 32 or actorder=True).

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import torch


def _unpack_rows_int4(packed: torch.Tensor, num_rows: int) -> torch.Tensor:
    packed_rows, cols = packed.shape
    pack_factor = 8  # 32 // 4
    unpacked = torch.zeros(
        packed_rows, pack_factor, cols, dtype=torch.int32, device=packed.device
    )
    for i in range(pack_factor):
        unpacked[:, i, :] = (packed >> (i * 4)) & 0x0F
    return unpacked.reshape(packed_rows * pack_factor, cols)[:num_rows]


def unpack_int_weights(
    packed: torch.Tensor, wbits: int, original_shape: tuple[int, int]
) -> torch.Tensor:
    if wbits != 4:
        raise ValueError(f"only wbits=4 supported, got {wbits}")
    in_features = original_shape[1]
    unpacked = _unpack_rows_int4(packed, in_features)
    return unpacked.t().contiguous()


def unpack_zeros(packed_zeros: torch.Tensor, wbits: int, out_features: int) -> torch.Tensor:
    if wbits != 4:
        raise ValueError(f"only wbits=4 supported, got {wbits}")
    return _unpack_rows_int4(packed_zeros.t().contiguous(), out_features).t().contiguous()


def dequant_gptq_to_fp(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    v1: bool = True,
) -> torch.Tensor:
    """Dequantize an AutoGPTQ-v1 packed 4-bit Linear back to (out, in) ``dtype``."""
    weight_int = unpack_int_weights(qweight, 4, (out_features, in_features))
    zeros = unpack_zeros(qzeros, 4, out_features)
    if v1:
        zeros = (zeros + 1) & 0x0F
    scale_expanded = scales[g_idx, :].T
    zero_expanded = zeros[g_idx, :].T
    weight = scale_expanded.float() * (weight_int.float() - zero_expanded.float())
    return weight.to(dtype)
