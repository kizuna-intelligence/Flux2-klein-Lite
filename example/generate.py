"""Generate images with the int4 FLUX.2-klein transformer.

Loads the packed-int4 ``Flux2Transformer2DModel`` with this runtime and plugs it
into the diffusers ``Flux2KleinPipeline``. The text encoder and VAE default to
the upstream bf16/fp16 weights, so the only extra dependency is ``diffusers``.

Optionally, pass ``--te <dir>`` to also load an int4-quantized Qwen3 text encoder
(produced by OneCompression). That needs ``onecompression`` importable and drops
the whole-pipeline VRAM to ~3.3 GB.

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python example/generate.py \\
        --dit /path/to/model.safetensors \\
        --outdir ./outputs

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from flux2_klein_int4 import load_int4_transformer

DEFAULT_PROMPTS = [
    ("cyberpunk_market",
     "A bustling cyberpunk night market in the rain, neon signs reflecting on wet "
     "asphalt, a lone figure in a translucent raincoat, steam rising from food "
     "stalls, cinematic, ultra-detailed, volumetric light"),
    ("dragon_library",
     "An ancient dragon curled around a towering spiral library, sunbeams through "
     "stained glass, floating candles, dust motes, oil painting style, warm golden "
     "hour lighting"),
    ("astronaut_jellyfish",
     "An astronaut floating among giant bioluminescent jellyfish in a deep-space "
     "nebula, reflections on the helmet visor, vivid teal and magenta, dreamlike, "
     "photorealistic render"),
    ("kyoto_autumn",
     "A traditional Kyoto temple courtyard in peak autumn, vivid red and orange "
     "maple leaves, a stone lantern, koi pond reflections, soft morning mist, "
     "serene, photographic"),
]


def _load_text_encoder(te_dir: str):
    """Load an int4 Qwen3 text encoder via OneCompression (optional path)."""
    from onecomp.quantized_model_loader import QuantizedModelLoader

    te, _tok = QuantizedModelLoader.load_quantized_model(
        te_dir, torch_dtype=torch.float16, device_map="cpu")
    # The pipeline only consumes intermediate hidden states, not vocab logits;
    # dropping lm_head saves the wasteful projection and avoids a dtype clash.
    te.lm_head = torch.nn.Identity()
    return te


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="black-forest-labs/FLUX.2-klein-4B",
                    help="HF repo (or local dir) for tokenizer / VAE / default text encoder")
    ap.add_argument("--dit", required=True, help="packed int4 DiT safetensors")
    ap.add_argument("--te", default=None,
                    help="optional int4 Qwen3 text-encoder dir (needs onecompression)")
    ap.add_argument("--backend", default="auto", choices=["auto", "gemlite", "fused", "eager"])
    ap.add_argument("--outdir", default="./outputs")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--offload", action="store_true",
                    help="enable model CPU offload (lowest VRAM)")
    ap.add_argument("--prompt", action="append", default=None,
                    help="custom prompt (repeatable); overrides the built-in set")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    from diffusers import Flux2KleinPipeline

    # Run the whole pipeline in fp16: GemLite int4 requires fp16 I/O, and mixing
    # bf16 components triggers Half/BFloat16 mismatches at norm/conv boundaries.
    print(f"[generate] loading int4 DiT (backend={args.backend}) ...", flush=True)
    dit = load_int4_transformer(args.dit, device="cuda:0", dtype="float16",
                                backend=args.backend, warmup=False)

    extra = {}
    if args.te:
        print(f"[generate] loading int4 text encoder from {args.te} ...", flush=True)
        extra["text_encoder"] = _load_text_encoder(args.te)

    print("[generate] assembling Flux2KleinPipeline ...", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(
        args.repo, transformer=dit, torch_dtype=torch.float16, **extra)

    if args.offload:
        pipe.enable_model_cpu_offload(device="cuda:0")
    else:
        pipe.to("cuda:0")

    prompts = (
        [(f"prompt_{i:02d}", p) for i, p in enumerate(args.prompt)]
        if args.prompt else DEFAULT_PROMPTS
    )

    print(f"[generate] generating {len(prompts)} image(s) at {args.size}px, "
          f"{args.steps} steps ...", flush=True)
    for i, (name, prompt) in enumerate(prompts):
        t0 = time.perf_counter()
        img = pipe(
            prompt=prompt, height=args.size, width=args.size,
            guidance_scale=args.guidance, num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed + i),
        ).images[0]
        path = os.path.join(args.outdir, f"{i:02d}_{name}.png")
        img.save(path)
        print(f"  saved {path}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    peak = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
    print(f"[generate] done. peak VRAM {peak:.2f} GB. images in {os.path.abspath(args.outdir)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
