# Flux2-klein-Lite

*[日本語版 README はこちら / Japanese README](README.ja.md)*

A lightweight runtime that runs the **FLUX.2-klein-4B transformer entirely in 4-bit**.

It loads a checkpoint that OneCompression has GPTQ-quantized and packed, and runs each
Linear with an **int4 GEMM kernel**. Weights are never expanded to bf16 in VRAM, and there
is no per-forward Python dequant.

Three backends are available:

- **`gemlite`** (default / recommended): the production-tuned Triton int4 kernels from
  [GemLite](https://github.com/mobiusml/gemlite). It packs the GPTQ scale / zero / int
  weights *as-is*, so the error is at the level of fp16 rounding (relative error ~5e-4).
- **`fused`**: the bundled in-house fused Triton kernel (dequant + GEMM fused into a single
  kernel). A fallback for environments without GemLite.
- **`eager`**: a plain `nn.Linear` dequantized to bf16 once at load time. VRAM goes back to
  bf16 levels; useful only as a speed baseline.

---

## Install

```bash
pip install -e .            # runtime only
pip install -e ".[gemlite]" # with the GemLite backend (recommended)
```

## Usage

```python
from flux2_klein_lite import load_int4_transformer

# backend: "auto" (default: GemLite if available, else fused) /
# "gemlite" / "fused" / "eager"
model = load_int4_transformer(
    "model.safetensors", device="cuda:0", dtype="bfloat16", backend="auto",
)
out = model(hidden_states=..., encoder_hidden_states=..., timestep=...,
            img_ids=..., txt_ids=..., return_dict=False)[0]
```

Benchmark (backend comparison):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python example/bench.py /path/to/model.safetensors --backend gemlite --compare
```

## Image generation

`example/generate.py` plugs the int4 DiT into the diffusers `Flux2KleinPipeline` and
generates images. The text encoder / VAE default to the upstream weights, so the only extra
dependency is `diffusers`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python example/generate.py --dit /path/to/model.safetensors --outdir ./outputs
```

Pass `--te <dir>` with a directory holding an int4-quantized Qwen3 text encoder (produced by
OneCompression) to run the text encoder in int4 as well. Combined with `--offload`, the
whole-pipeline peak VRAM drops to about 3.3 GB (requires `onecompression`). Repeat `--prompt`
to generate arbitrary prompts.

```bash
python example/generate.py --dit model.safetensors --te ./flux2_te_int4 \
  --offload --prompt "a calico cat astronaut on the moon, photorealistic"
```

---

## How it works

- **GemLite backend** (`flux2_klein_lite/gemlite_int4_linear.py`): unpacks the AutoGPTQ-v1
  pack (qweight int32 / scales fp16 / qzeros int32 v1-offset) and re-packs it directly into
  GemLite. Unlike a re-quant through HQQ (~2% error), this preserves the GPTQ values. I/O is
  fp16 (a GemLite requirement); outputs are cast back to the caller's dtype.
- **Fused kernel** (`flux2_klein_lite/fused_int4_linear.py`): reads the same pack format
  directly, pads K, and feeds Triton's `tl.dot`. Launch configs are cached per M bucket.
- **Loader** (`flux2_klein_lite/loader.py`): builds the model on the meta device, swaps
  quantized layers for the selected backend, then materializes only the non-quantized tensors
  on-device. The full bf16 weight set is never expanded into VRAM, so the load-time peak is low.
- **Fallback**: any layer that can't ride the int4 kernel (groupsize≠32, actorder, etc.)
  becomes an `nn.Linear` eager-dequantized once at load time (for FLUX.2-klein all 109 layers
  are kernel-eligible).

## Measurements (RTX PRO 4000 Blackwell, FLUX.2-klein-4B, txt=128, all 109 Linears int4)

### Disk / VRAM

| | bf16 (eager expand) | fused int4 | **gemlite int4** |
| --- | ---: | ---: | ---: |
| safetensors | 7.7 GB | 2.1 GB | **2.1 GB** |
| VRAM after load (alloc) | 9.77 GB | 4.78 GB | **2.73 GB** |
| forward peak VRAM (peak_res) | ~9.95 GB | 5.01 GB | **3.35 GB** |

### Forward latency (per backend, mean of 20 iters)

| image tokens | gemlite int4 | fused int4 | eager (bf16 expand) |
| --- | ---: | ---: | ---: |
| grid 8 (64 tok) | 47.1 ms | – | 38.3 ms |
| grid 16 (256 tok) | 78.3 ms | – | 60.2 ms |
| grid 24 (576 tok) | 130.6 ms | – | 102.1 ms |
| grid 32 (1024 tok) | 211.5 ms | 241.5 ms | 152.5 ms |

**Important note**: FLUX image generation is **compute-bound** at large M (hundreds to
thousands of tokens). Weight-only int4 keeps the weights in int4 and does dequant + GEMM, so
it cannot beat bf16 cuBLAS (which already has the weights expanded in VRAM). In this regime
GemLite is about 0.72–0.81x and the in-house fused kernel about 0.63x. **int4's win here is
~1/3 the VRAM** (2.73 GB vs 9.77 GB), not speed — the opposite of the small-M, memory-bound
TTS DiT case. Of the three backends GemLite is both the fastest and the smallest in VRAM, so
prefer GemLite when you want int4; if you have VRAM to spare and want maximum speed, use bf16.

**Warmup**: on the first load GemLite runs a Triton autotune that takes 60–90 seconds (11
signatures). This is a one-time cost and is not part of the inference itself.

## License

MIT (Copyright 2025-2026 Kizuna Intelligence).
`flux2_klein_lite/quant_utils.py` is vendored from OneCompression and is
Copyright 2025-2026 Fujitsu Ltd. (also MIT).

The **model weights follow the upstream FLUX.2-klein license** (Black Forest Labs).
