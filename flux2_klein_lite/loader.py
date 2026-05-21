"""Fast int4 loader for packed FLUX.2 transformer checkpoints.

Reads a checkpoint produced by OneCompression's
``Flux2DiTAdapter.save_quantized_model`` (safetensors with ``config_json`` /
``quant_layers_json`` metadata) and rebuilds a ``Flux2Transformer2DModel`` whose
quantized ``nn.Linear`` modules are replaced by :class:`FusedInt4Linear` — a
Triton kernel that fuses int4 dequant + GEMM, so weights are never unpacked to
bf16 in VRAM and no per-call Python dequant runs.

Layers that don't fit the fused kernel (groupsize != 32, actorder, or odd
shapes) fall back to an eager-dequant ``nn.Linear`` built once at load time.

Usage::

    from flux2_klein_lite import load_int4_transformer
    model = load_int4_transformer("model.safetensors", device="cuda:0")

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .fused_int4_linear import FusedInt4Linear
from .gemlite_int4_linear import GemLiteInt4Linear, gemlite_available
from .quant_utils import dequant_gptq_to_fp

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[str(dtype).lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype: {dtype!r}")


def _can_use_fused(entry: dict[str, Any]) -> bool:
    return (
        not bool(entry.get("actorder", False))
        and int(entry["wbits"]) == 4
        and int(entry["groupsize"]) == 32
        and int(entry["in_features"]) % 32 == 0
        and int(entry["out_features"]) % 8 == 0
    )


def _build_gemlite(entry: dict, st: dict, device: torch.device) -> GemLiteInt4Linear:
    return GemLiteInt4Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"],
        scales=st["scales"],
        qzeros=st["qzeros"],
        bias=(st["bias"] if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
        device=device,
    )


def _build_fused(entry: dict, st: dict, device: torch.device) -> FusedInt4Linear:
    return FusedInt4Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"].to(device),
        scales=st["scales"].to(device=device, dtype=torch.float16),
        qzeros=st["qzeros"].to(device),
        bias=(st["bias"].to(device) if "bias" in st else None),
        groupsize=32,
    )


def _build_eager(entry: dict, st: dict, dtype: torch.dtype,
                 device: torch.device) -> torch.nn.Linear:
    in_f, out_f = int(entry["in_features"]), int(entry["out_features"])
    g_idx = st.get("g_idx")
    if g_idx is None:
        gs = int(entry["groupsize"])
        g_idx = (torch.arange(in_f) // gs).to(torch.long)
    weight = dequant_gptq_to_fp(
        qweight=st["qweight"], scales=st["scales"], qzeros=st["qzeros"],
        g_idx=g_idx, in_features=in_f, out_features=out_f, dtype=dtype,
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
    ).to(device)
    has_bias = "bias" in st
    lin = torch.nn.Linear(in_f, out_f, bias=has_bias, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if has_bias:
            lin.bias.copy_(st["bias"].to(dtype=dtype, device=device))
    return lin


def _resolve_backend(backend: str | None, use_fused: bool) -> str:
    """Pick the int4 GEMM backend.

    ``backend`` takes precedence; ``use_fused=False`` (legacy) forces eager.
    ``"auto"`` (default) prefers GemLite when importable — it runs within ~15%
    of bf16 at FLUX's large-M shapes — then the fused Triton kernel, else eager.
    """
    if not use_fused:
        return "eager"
    b = (backend or "auto").lower()
    if b == "auto":
        return "gemlite" if gemlite_available() else "fused"
    if b == "gemlite" and not gemlite_available():
        raise RuntimeError(
            "backend='gemlite' requested but the 'gemlite' package is not "
            "importable; pip install gemlite or use backend='fused'"
        )
    if b not in ("gemlite", "fused", "eager"):
        raise ValueError(f"unknown backend: {backend!r}")
    return b


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

    backend = _resolve_backend(backend, use_fused)
    resolved = _resolve_dtype(dtype)
    dev = torch.device(device)
    path = Path(checkpoint_path)

    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        cfg_json = md.get("config_json")
        quant_layers_json = md.get("quant_layers_json")
        if cfg_json is None or quant_layers_json is None:
            raise ValueError(
                f"{path} is missing 'config_json' / 'quant_layers_json' "
                "metadata; not a packed OneCompression FLUX.2 checkpoint"
            )
        cfg = {k: v for k, v in json.loads(cfg_json).items() if not k.startswith("_")}
        quant_layers = json.loads(quant_layers_json)
        ckpt_fmt = md.get("checkpoint_format", "gptq")
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    # Build the bare model on the meta device, then materialise non-quant
    # tensors directly on the target device — never allocate the full bf16
    # weight set in VRAM.
    with torch.device("meta"):
        model = Flux2Transformer2DModel.from_config(cfg)

    modules = dict(model.named_modules())
    gemlite_n = fused_n = eager_n = 0
    quant_keys: set[str] = set()
    quant_names: list[str] = []
    for entry in quant_layers:
        entry["checkpoint_format"] = ckpt_fmt
        name = entry["name"]
        quant_names.append(name)
        parent_name, _, child = name.rpartition(".")
        parent = modules.get(parent_name) if parent_name else model
        if parent is None:
            raise KeyError(f"quant layer parent not found: {parent_name!r}")
        st = {}
        for s in ("qweight", "scales", "qzeros", "g_idx", "bias"):
            k = f"{name}.{s}"
            if k in tensors:
                st[s] = tensors[k]
                quant_keys.add(k)
        quant_keys.add(f"{name}.weight")
        if backend == "gemlite" and _can_use_fused(entry):
            layer = _build_gemlite(entry, st, dev)
            gemlite_n += 1
        elif backend in ("gemlite", "fused") and _can_use_fused(entry):
            layer = _build_fused(entry, st, dev)
            fused_n += 1
        else:
            layer = _build_eager(entry, st, resolved, dev)
            eager_n += 1
        setattr(parent, child, layer)

    # Materialise the remaining (non-quant) tensors on-device.
    non_quant = {
        k: v.to(device=dev, dtype=resolved if v.dtype.is_floating_point else v.dtype)
        for k, v in tensors.items()
        if k not in quant_keys
    }
    missing, unexpected = model.load_state_dict(non_quant, strict=False, assign=True)
    # Swapped quant modules own their own buffers (fused: qweight/scales/qzeros;
    # gemlite: _gl.*); none come from the checkpoint, so drop any missing key
    # that lives under a quantized layer's subtree.
    quant_prefixes = tuple(f"{n}." for n in quant_names)
    real_missing = [
        m for m in missing
        if m not in quant_keys and not m.startswith(quant_prefixes)
    ]
    if real_missing:
        raise RuntimeError(f"missing keys: {real_missing[:8]} ...")
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:8]} ...")

    # Any leftover meta params (e.g. buffers) → materialise empty on device.
    for n, p in model.named_parameters():
        if p.is_meta:
            raise RuntimeError(f"parameter left on meta device: {n}")

    model.eval()
    print(f"[flux2_klein_lite] loaded {len(quant_layers)} int4 layers "
          f"(gemlite={gemlite_n}, fused={fused_n}, eager={eager_n}) on {dev}")

    if warmup and dev.type == "cuda":
        t0 = time.perf_counter()
        seen: dict[tuple, Any] = {}
        for m in model.modules():
            if isinstance(m, (FusedInt4Linear, GemLiteInt4Linear)):
                seen.setdefault((m.in_features, m.out_features, m.bias is not None), m)
        if seen:
            for layer in seen.values():
                layer.warmup(m_values=warmup_m_values)
            torch.cuda.synchronize(dev)
            print(f"[flux2_klein_lite] warmup {(time.perf_counter()-t0)*1000:.0f} ms "
                  f"({len(seen)} unique signatures)")
    return model
