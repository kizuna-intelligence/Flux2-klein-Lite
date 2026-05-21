"""Load a packed FLUX.2-klein int4 checkpoint and benchmark forward latency.

Reports VRAM peak and per-forward latency for the fused-int4 runtime, and
(with --compare) the naive eager-dequant baseline for the same checkpoint.

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python example/bench.py /path/to/model.safetensors --compare

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from flux2_klein_int4 import load_int4_transformer


def _mb(x: int) -> float:
    return x / (1024 * 1024)


def _make_inputs(model, grid: int, txt_seq: int, dev: torch.device):
    cfg = dict(model.config)
    img_seq = grid * grid
    img_ids = torch.zeros((img_seq, 4), device=dev)
    img_ids[:, 1] = torch.arange(grid, device=dev).repeat_interleave(grid)[:img_seq]
    img_ids[:, 2] = torch.arange(grid, device=dev).repeat(grid)[:img_seq]
    txt_ids = torch.zeros((txt_seq, 4), device=dev)
    return dict(
        hidden_states=torch.randn(1, img_seq, int(cfg["in_channels"]), device=dev, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(1, txt_seq, int(cfg["joint_attention_dim"]), device=dev, dtype=torch.bfloat16),
        timestep=torch.rand(1, device=dev, dtype=torch.bfloat16),
        img_ids=img_ids,
        txt_ids=txt_ids,
    )


@torch.no_grad()
def _bench(model, fwd, dev, iters: int) -> float:
    for _ in range(3):  # warm
        model(**fwd, return_dict=False)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(**fwd, return_dict=False)
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / iters * 1000.0


def _run(checkpoint, dev, grid, txt_seq, iters, backend, tag):
    torch.cuda.reset_peak_memory_stats(dev)
    model = load_int4_transformer(checkpoint, device=str(dev), dtype="bfloat16",
                                  backend=backend)
    after_load = _mb(torch.cuda.max_memory_allocated(dev))
    fwd = _make_inputs(model, grid, txt_seq, dev)
    ms = _bench(model, fwd, dev, iters)
    peak_a = _mb(torch.cuda.max_memory_allocated(dev))
    peak_r = _mb(torch.cuda.max_memory_reserved(dev))
    print(f"  [{tag}] load_alloc={after_load:.0f}MB  peak_alloc={peak_a:.0f}MB  "
          f"peak_res={peak_r:.0f}MB  latency={ms:.2f}ms/forward")
    del model
    torch.cuda.empty_cache()
    return ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=int, default=32, help="image side (grid*grid tokens)")
    ap.add_argument("--text-seq-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--backend", default="gemlite",
                    choices=["auto", "gemlite", "fused", "eager"],
                    help="primary int4 GEMM backend to benchmark")
    ap.add_argument("--compare", action="store_true",
                    help="also run the fused and eager backends for comparison")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.init()
    print(f"FLUX.2-klein int4 bench  grid={args.grid} ({args.grid**2} img tok) "
          f"txt={args.text_seq_len} iters={args.iters}")
    primary = _run(args.checkpoint, dev, args.grid, args.text_seq_len, args.iters,
                   args.backend, args.backend)
    if args.compare:
        results = {args.backend: primary}
        for b in ("gemlite", "fused", "eager"):
            if b == args.backend:
                continue
            results[b] = _run(args.checkpoint, dev, args.grid, args.text_seq_len,
                              args.iters, b, b)
        base = results.get("eager")
        if base:
            print()
            for b, ms in results.items():
                print(f"  speedup vs eager [{b}]: {base / ms:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
