"""Generate images with the int4 FLUX.2-klein transformer.

Loads the packed-int4 ``Flux2Transformer2DModel`` with this runtime and plugs it
into the diffusers ``Flux2KleinPipeline``. The text encoder and VAE default to
the upstream bf16/fp16 weights, so the only extra dependency is ``diffusers``.

With no ``--dit``, the packed int4 DiT is auto-downloaded from ``--weights-repo``
(default ``kizuna-intelligence/FLUX.2-klein-4B-int4``). Add ``--int4-te`` to also
pull the int4 Qwen3 text encoder from the same repo (needs ``onecompression``
importable); combined with ``--offload`` the whole-pipeline VRAM drops to ~3.3 GB.

Run (everything auto-downloaded)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python example/generate.py --int4-te --offload --outdir ./outputs

Copyright 2025-2026 Kizuna Intelligence.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from flux2_klein_lite import load_int4_transformer

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
    ap.add_argument("--weights-repo", default="kizuna-intelligence/FLUX.2-klein-4B-int4",
                    help="HF repo holding the packed int4 weights (transformer/, text_encoder/)")
    ap.add_argument("--dit", default=None,
                    help="packed int4 DiT safetensors (local). If omitted, downloaded from --weights-repo")
    ap.add_argument("--te", default=None,
                    help="local int4 Qwen3 text-encoder dir (needs onecompression)")
    ap.add_argument("--int4-te", action="store_true",
                    help="use the int4 text encoder from --weights-repo (auto-download)")
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

    dit_path = args.dit
    if dit_path is None:
        from huggingface_hub import hf_hub_download
        print(f"[generate] downloading int4 DiT from {args.weights_repo} ...", flush=True)
        dit_path = hf_hub_download(args.weights_repo, "transformer/model.safetensors")

    te_dir = args.te
    if te_dir is None and args.int4_te:
        from huggingface_hub import snapshot_download
        print(f"[generate] downloading int4 text encoder from {args.weights_repo} ...", flush=True)
        snap = snapshot_download(args.weights_repo, allow_patterns="text_encoder/*")
        te_dir = os.path.join(snap, "text_encoder")

    # Run the whole pipeline in fp16: GemLite int4 requires fp16 I/O, and mixing
    # bf16 components triggers Half/BFloat16 mismatches at norm/conv boundaries.
    print(f"[generate] loading int4 DiT (backend={args.backend}) ...", flush=True)
    dit = load_int4_transformer(dit_path, device="cuda:0", dtype="float16",
                                backend=args.backend, warmup=False)

    extra = {}
    if te_dir:
        print(f"[generate] loading int4 text encoder from {te_dir} ...", flush=True)
        extra["text_encoder"] = _load_text_encoder(te_dir)

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
