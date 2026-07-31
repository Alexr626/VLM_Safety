# VTI textual direction extraction: is the same image used for positive vs negative?

**Date:** 2026-06-28

## Question

For textual VTI steering vectors built from `data/vti/demos.jsonl`, is the **exact same unaltered image** fed to the VLM when extracting activations for the positive (truthful) vs negative (hallucinated) forward pass? Has any visual perturbation been implemented?

## Short answer

**Yes** — for the **textual** arm you have been running, the same unaltered COCO image is used for both passes; only the caption text differs (`value` vs `h_value`). **No** — this repo has **not** implemented the vision-encoder VTI arm or any visual perturbation during experiments.

## What `demos.jsonl` contains

Each line is one COCO train2014 image plus paired captions:

| Field | Role |
|-------|------|
| `image` | Single COCO filename (e.g. `COCO_train2014_000000103108.jpg`) |
| `value` | Truthful / clean caption |
| `h_value` | Hallucinated caption (objects or details inserted or altered in text) |
| `question` | Prompt prefix (typically `"Describe this image in detail."`) |

There is **no** second image field and **no** image perturbation metadata in the file.

## What our pipeline does (`evaluation/interventions/vti/directions.py`)

`get_hiddenstates()` loads the demo image **once** per demo, then runs two `wrapper.forward_vl(image, text)` calls:

1. **Hallucinated:** `question + h_value`
2. **Clean:** `question + value`

The `image` object is identical across both forwards. Last-token decoder hidden states are stacked per layer; PCA is fit on `clean − hallucinated` (`obtain_textual_vti`).

Steering at eval time (`vti_textual_*` interventions) hooks **decoder** layers only. No vision-tower hooks exist.

## What has *not* been implemented

Per `IMPLEMENTATION.md` §Visual encoder hooks:

- No `obtain_visual_vti`, no `get_visual_hiddenstates`
- No `vti_visual_*` intervention in the registry
- No `alpha_image` / vision-arm steering during eval

All experiments to date are **textual β** (decoder) only.

## Reference VTI repo nuance (for when you add the vision arm)

The vendored `VTI/` code separates the two arms:

| Arm | Negative condition | Positive condition | Image in forward |
|-----|-------------------|--------------------|------------------|
| **Textual** (`get_hiddenstates`) | `h_value` caption | `value` caption | **Same** unaltered image (`image_tensor[example_id][-1]` for both styles) |
| **Visual** (`get_visual_hiddenstates`) | **Random patch-masked** image(s) | Unaltered image | **Different** tensors (masking via `mask_patches`, not caption change) |

So even in the authors' code, caption pairs in `demos.jsonl` drive **textual** directions. The **visual** directions contrast corrupted vs clean **pixels**, not clean vs hallucinated captions on the same pixels.

## Corrections to common assumptions

| Assumption | Verdict |
|------------|---------|
| "Only textual steering has been tested in this repo" | **Correct** |
| "Negative demos alter caption text only" | **Correct** for textual extraction |
| "Same image for both textual activation passes" | **Correct** |
| "We have implemented visual perturbation for steering vectors" | **Incorrect** — not implemented here; reference visual arm would use patch masking, not `h_value` |
