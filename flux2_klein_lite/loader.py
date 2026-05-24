"""Fast int4 loader for packed FLUX.2 transformer checkpoints.

Thin adapter over :func:`onecomp_runtime.diffusion.load_int4_model`. The only
FLUX-specific bit is the model class: a packed checkpoint produced by
OneCompression's ``Flux2DiTAdapter.save_quantized_model`` rebuilds a
``Flux2Transformer2DModel`` whose quantized ``nn.Linear`` modules are swapped for
fused / gemlite / eager int4 modules.

Usage::

    from flux2_klein_lite import load_int4_transformer
    model = load_int4_transformer("model.safetensors", device="cuda:0")

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

import torch

from onecomp_runtime.diffusion import load_int4_model


def load_int4_transformer(
    checkpoint_path: str,
    device: str = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    backend: str | None = "auto",
    use_fused: bool = True,
    warmup: bool = True,
    warmup_m_values: tuple[int, ...] = (64, 128, 256, 1024, 4096),
):
    """Rebuild a packed-int4 ``Flux2Transformer2DModel`` for fast inference.

    Args:
        checkpoint_path: path to the packed ``model.safetensors``.
        device: target device, e.g. ``"cuda:0"``.
        dtype: compute dtype for the non-quantized tensors (bf16 recommended).
        backend: int4 GEMM backend — ``"auto"`` (GemLite if available, else
            fused Triton), ``"gemlite"``, ``"fused"``, or ``"eager"``.
        use_fused: legacy switch; ``False`` forces the eager-dequant fallback.
        warmup: trigger Triton JIT for the M-buckets inference will hit.
        warmup_m_values: token counts to pre-compile (batch * seq).

    Returns:
        an ``eval``-mode ``Flux2Transformer2DModel`` on ``device``.
    """
    from diffusers import Flux2Transformer2DModel

    return load_int4_model(
        checkpoint_path,
        lambda cfg: Flux2Transformer2DModel.from_config(cfg),
        device=device,
        dtype=dtype,
        backend=backend,
        use_fused=use_fused,
        warmup=warmup,
        warmup_m_values=warmup_m_values,
        label="flux2_klein_lite",
    )
