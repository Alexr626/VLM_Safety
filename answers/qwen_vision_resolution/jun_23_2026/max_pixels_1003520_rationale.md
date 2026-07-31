# Why `MAX_PIXELS=1003520` for Qwen2.5-VL CHAIR/AMBER runs?

**Date:** 2026-06-23  
**Context:** Qwen Exp2 rotation-strength sweep on pinned CHAIR/AMBER subsets.

## The number

```
1003520 = 1280 × 28 × 28
```

## What it means in Qwen2.5-VL

Qwen2-VL / 2.5-VL resize images so height and width are multiples of **28** (patch size 14 × merge factor 2). Each **visual token** fed to the LLM corresponds to a **28×28 pixel** region after resize.

`max_pixels` is therefore not an arbitrary byte budget — it is a **pixel-area cap** on the resized image. Rough token count:

```
visual_tokens ≈ (resized_height × resized_width) / (28 × 28)
```

Setting `max_pixels = 1280 × 28 × 28` caps an image at roughly **1280 visual tokens** (before any further sequence packing). This is the upper end of the range Qwen's own HuggingFace model card recommends for balancing quality vs cost:

```python
min_pixels = 256 * 28 * 28   # ~256 tokens minimum
max_pixels = 1280 * 28 * 28  # ~1280 tokens maximum
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)
```

(Source: [Qwen2.5-VL-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct))

The processor's **native default** is much higher: `longest_edge ≈ 12 845 056` pixels (~16 384 tokens), which is why Policy A ("native resolution") is dynamic and can be very expensive on large photos.

## Why we needed a cap on this machine

On CHAIR/AMBER, some images are multi-megapixel (e.g. a 4898×3265 ≈ 16 MP AMBER photo). At native/default resize:

| Setting | Patches (pre-merge) | ViT attention cost | Outcome on A6000 |
|---------|---------------------|--------------------|------------------|
| Native (`longest_edge` ≈ 12.8M) | ~64 896 | O(patches²) → **~251 GiB** allocation | CUDA OOM in ViT `scaled_dot_product_attention` |
| `max_pixels = 1003520` | ~4 988 | Manageable | Runs; ~1 247 merged visual tokens |

The OOM happens in the **vision encoder**, not the LLM decode loop, and is independent of `max_new_tokens`.

## Why 1280 specifically (not e.g. 512 or 2048)

1. **Official recommended "high detail but bounded" setting** — Qwen documents 256–1280 tokens as the practical tuning band; 1280 is the top of that band, not an ad-hoc number.
2. **Empirically sufficient here** — validated on the actual OOMing 16 MP images: both score successfully at 1003520 with 0 OOM.
3. **VRAM headroom on shared GPU 0** — leaves room to run alongside LLaVA (~14 GB) on a 48 GB A6000; higher caps (e.g. `2048×28×28`) increase per-image memory and slow the sweep.
4. **Not the same as LLaVA's fixed 336²** — LLaVA always uses 576 tokens; Qwen is inherently dynamic. Capping at 1280 tokens is a deliberate, documented compromise for this hardware, and should be logged in `RESEARCH_LOG.md` when results are reported.

## Alternatives

| `max_pixels` | Approx max tokens | Tradeoff |
|--------------|-------------------|----------|
| `512 × 28 × 28` = 401 408 | ~512 | Faster, less detail; may hurt fine-grained CHAIR/AMBER items |
| `1280 × 28 × 28` = 1 003 520 | ~1280 | **Current choice** — Qwen-recommended upper practical bound |
| `2048 × 28 × 28` = 1 605 632 | ~2048 | More detail; higher VRAM and runtime |
| Native (unset) | up to ~16 384 | Policy A; OOMs on large CHAIR/AMBER images on 48 GB |

## Implementation note

In transformers 4.50.1, resizing is driven by `image_processor.size["longest_edge"]`, not the `max_pixels` attribute alone. `Qwen2VLWrapper.load()` now sets both so the cap is actually applied.
