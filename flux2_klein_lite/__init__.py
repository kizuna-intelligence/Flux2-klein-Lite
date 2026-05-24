"""flux2_klein_lite — fast int4 inference runtime for FLUX.2-klein.

Loads OneCompression-packed int4 FLUX.2 transformer checkpoints and runs each
Linear with an int4 GEMM kernel (no per-call weight unpacking). Three backends:
``gemlite`` (preferred, GemLite Triton kernels), ``fused`` (bundled dequant+GEMM
Triton kernel), and ``eager`` (one-time bf16 dequant baseline).

The int4 kernels and loader live in the shared ``onecomp-runtime`` package; this
repo is a thin FLUX.2 adapter over them.

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

from onecomp_runtime.layers import (
    FusedInt4Linear,
    GemLiteInt4Linear,
    fused_int4_gemm,
    gemlite_available,
)

from .loader import load_int4_transformer

__all__ = [
    "load_int4_transformer",
    "FusedInt4Linear",
    "fused_int4_gemm",
    "GemLiteInt4Linear",
    "gemlite_available",
]
