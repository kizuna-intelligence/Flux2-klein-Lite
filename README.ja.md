# Flux2-klein-Lite

*[English README (standard)](README.md)*

FLUX.2-klein-4B の transformer を **4-bit 量子化したまま推論する軽量ランタイム**です。

OneCompression で GPTQ 量子化・パックしたチェックポイントを読み込み、各 Linear を
**int4 GEMM カーネル**で動かします。重みを bf16 に展開して VRAM に置くことも、forward
ごとに Python で dequant することもありません。

3 つのバックエンドを選べます:

- **`gemlite`**（既定 / 推奨）: [GemLite](https://github.com/mobiusml/gemlite) の
  本番チューニング済み Triton int4 カーネル。GPTQ の scale / zero / int 重みを
  *そのまま* パックするので、誤差は fp16 丸め相当（相対誤差 ~5e-4）。
- **`fused`**: 同梱の自前融合 Triton カーネル（dequant + GEMM を 1 カーネルに融合）。
  GemLite が入っていない環境向けフォールバック。
- **`eager`**: ロード時に一度だけ bf16 へ dequant した素の `nn.Linear`。VRAM は
  bf16 相当に戻るが、速度比較のベースライン用。

---

## インストール

```bash
pip install -e .            # ランタイム本体
pip install -e ".[gemlite]" # GemLite バックエンド込み（推奨）
```

## 使い方

```python
from flux2_klein_lite import load_int4_transformer

# backend は "auto"（既定: GemLite があれば GemLite、無ければ fused）/
# "gemlite" / "fused" / "eager"
model = load_int4_transformer(
    "model.safetensors", device="cuda:0", dtype="bfloat16", backend="auto",
)
out = model(hidden_states=..., encoder_hidden_states=..., timestep=...,
            img_ids=..., txt_ids=..., return_dict=False)[0]
```

ベンチ（バックエンド比較）:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python example/bench.py /path/to/model.safetensors --backend gemlite --compare
```

## 画像生成

`example/generate.py` は int4 DiT を diffusers の `Flux2KleinPipeline` に差し込んで
画像を生成します。テキストエンコーダ / VAE は既定で上流の重みを使うので、追加依存は
`diffusers` だけです。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python example/generate.py --dit /path/to/model.safetensors --outdir ./outputs
```

オプション `--te <dir>` に OneCompression で int4 量子化した Qwen3 テキストエンコーダの
ディレクトリを渡すと、テキストエンコーダも int4 になり、`--offload` 併用で
パイプライン全体のピーク VRAM が約 3.3 GB まで下がります（要 `onecompression`）。
`--prompt` を複数回指定すれば任意のプロンプトで生成できます。

```bash
python example/generate.py --dit model.safetensors --te ./flux2_te_int4 \
  --offload --prompt "a calico cat astronaut on the moon, photorealistic"
```

---

## 仕組み

- **GemLite バックエンド** (`flux2_klein_lite/gemlite_int4_linear.py`): AutoGPTQ-v1
  パック (qweight int32 / scales fp16 / qzeros int32 v1-offset) を unpack して
  GemLite に直接 pack。HQQ 経由の再量子化（誤差 ~2%）と違い GPTQ 値を保存する。
  入出力は fp16（GemLite 要件）、呼び出し側 dtype へキャストして返す。
- **融合カーネル** (`flux2_klein_lite/fused_int4_linear.py`): 同じパック形式を直接
  読み、K をパディングして Triton の `tl.dot` に流す。M バケットごとに起動設定を
  キャッシュ。
- **ローダ** (`flux2_klein_lite/loader.py`): メタデバイス上にモデルを構築 → 量子化層を
  選択バックエンドに差し替え → 非量子化テンソルだけを実機に materialise。bf16 重み一式を
  VRAM に展開しないので、ロード時のピークが低い。
- **フォールバック**: groupsize≠32 / actorder などで int4 カーネルに乗らない層は、
  ロード時に一度だけ eager dequant した `nn.Linear` になる（FLUX.2-klein では全 109 層が
  カーネル対象）。

## 計測（実機 RTX PRO 4000 Blackwell, FLUX.2-klein-4B, txt=128, 全 109 Linear int4）

### ディスク / VRAM

| | bf16 (eager 展開) | fused int4 | **gemlite int4** |
| --- | ---: | ---: | ---: |
| safetensors | 7.7 GB | 2.1 GB | **2.1 GB** |
| ロード後 VRAM (alloc) | 9.77 GB | 4.78 GB | **2.73 GB** |
| forward ピーク VRAM (peak_res) | ~9.95 GB | 5.01 GB | **3.35 GB** |

### forward レイテンシ（バックエンド別, 20 iter 平均）

| 画像トークン | gemlite int4 | fused int4 | eager(bf16展開) |
| --- | ---: | ---: | ---: |
| grid 8 (64 tok) | 47.1 ms | – | 38.3 ms |
| grid 16 (256 tok) | 78.3 ms | – | 60.2 ms |
| grid 24 (576 tok) | 130.6 ms | – | 102.1 ms |
| grid 32 (1024 tok) | 211.5 ms | 241.5 ms | 152.5 ms |

**注意（重要）**: FLUX の画像生成は大 M（数百〜千トークン）で **compute-bound**。
weight-only int4 は重みを int4 に保ったまま dequant + GEMM するため、bf16 cuBLAS
（重みを VRAM に展開済み）より速くはならない。この領域では GemLite で約 0.72〜0.81x、
自前融合カーネルで約 0.63x。**int4 の利点は VRAM 約 1/3**（2.73GB vs 9.77GB）であって、
速度ではない。小 M で memory-bound な TTS DiT とは逆の傾向。3 バックエンドの中では
GemLite が最速かつ最小 VRAM なので、int4 を使うなら GemLite を推奨。VRAM に余裕があり
速度最優先なら bf16。

**warmup**: GemLite は初回ロード時に Triton autotune が走り 60〜90 秒かかる（11 シグネチャ）。
これは 1 回限りのコストで、推論本体には乗らない。

## ライセンス

MIT（Copyright 2025-2026 Kizuna Intelligence）。
`flux2_klein_lite/quant_utils.py` は OneCompression からの vendored で
Copyright 2025-2026 Fujitsu Ltd.（同じく MIT）。

**モデル重みは上流 FLUX.2-klein のライセンス**（Black Forest Labs）に従います。
