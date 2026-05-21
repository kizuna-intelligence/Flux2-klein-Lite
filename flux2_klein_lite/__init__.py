"""flux2_klein_lite — fast int4 inference runtime for FLUX.2-klein.

Loads OneCompression-packed int4 FLUX.2 transformer checkpoints and runs each
Linear with an int4 GEMM kernel (no per-call weight unpacking). Three backends:
``gemlite`` (preferred, GemLite Triton kernels), ``fused`` (bundled dequant+GEMM
Triton kernel), and ``eager`` (one-time bf16 dequant baseline).

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

from .fused_int4_linear import FusedInt4Linear, fused_int4_gemm
from .gemlite_int4_linear import GemLiteInt4Linear, gemlite_available
from .loader import load_int4_transformer

__all__ = [
    "load_int4_transformer",
    "FusedInt4Linear",
    "fused_int4_gemm",
    "GemLiteInt4Linear",
    "gemlite_available",
]
