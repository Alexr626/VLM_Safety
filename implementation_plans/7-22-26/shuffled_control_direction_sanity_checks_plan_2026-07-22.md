# Shuffled-Control Direction — Sanity-Check Plan

Date: 2026-07-22
Author: analyst-side (Claude)
Status: to be implemented and run by Cursor before any comparison experiment.
Control for: deployed direction `demosv2_9a44f4af_all_nd200_s42_r2_prefix` (demos_v2 `all` @ nd200).

## Purpose

Construct the shuffled-control direction for LLaVA-1.5-7B and Qwen2.5-VL-7B and verify it
differs from the deployed `all`/nd200 direction only by the image swap. Produce a verification
report per model. Run nothing downstream: no cosine, no magnitude comparison as an experimental
result, no behavioral runs. This plan exists to catch construction errors before the comparison
experiment is built on top of them.

This is not a hypothesis test — it is a construct-and-verify pass.

## Background the implementer needs

The deployed direction is built by `compute_or_load_textual_directions_v2(...)` in
`src/directions/directions_v2.py` (confirm the module path against IMPLEMENTATION.md) with
`dimension="all", num_demos=200, rank=2, seed=42`. Its construction:

- Demo selection: `select_prefix_demos(rows_by_id, order_obj["ids"], "all", 200)` walks the
  seed-42 master order in `data/vti/demos_v2_order_s42.json` and keeps the first 200 ids valid
  for `all` (`_row_has_dimension` requires non-empty `h_values["all"]` and `value`). The exact
  200 ids are recorded as `ids_used` in the deployed direction's `metadata.json`.
- Each selected demo carries `image = row["image"]`, `value` (truthful caption),
  `h_value = row["h_values"]["all"]` (hallucinated caption).
- Activations: `ensure_variant_activation(...)` forwards
  `wrapper.forward_vl(load_demo_image(demo), _demo_prompt(demo, caption))` and takes the
  last-token hidden states across the embedding row plus every decoder layer. It caches by
  `(demo id, "vl_v2_"+variant)` under
  `experiment_artifacts/vti/{model_short}/textual_v2/_act_cache`. The image is not part of the
  cache key.
- Direction: per-demo diff `value − h_value` of the flattened last-token stacks, rank-2 PCA,
  direction = PC1 + mean (`obtain_textual_vti_v2_from_stacks`), shape `(num_layers, hidden_dim)`.

Two consequences shape this plan:

1. The extractor has no parameter to pair a demo's captions with another demo's image. The image
   is bound to the demo by id. Building the shuffled control requires new code: a per-demo
   image-path override applied before the forward pass.
2. Because the activation cache key omits the image, a shuffled run pointed at the existing
   `textual_v2/_act_cache` would return original-image activations and silently reproduce the
   deployed direction. The shuffled run must use a separate cache namespace so every pair is
   recomputed with its swapped image.

## Net-new code (does not exist today; flagged explicitly)

- Per-demo image-path override: build the same `flat_demos` list as the deployed run (same ids,
  same order, same captions), then set each demo's `image` field to the source demo's image path
  per the derangement map, leaving `id` and both captions unchanged.
- Separate activation-cache namespace for the shuffled run: construct `ActivationCache` pointed at
  `experiment_artifacts/vti/{model_short}/shuffled_control/_act_cache` (empty at start), not the
  `textual_v2/_act_cache`.
- Derangement generation, validation, and serialization.
- Shuffled-control output directory and naming under `shuffled_control/` (do not use
  `textual_v2_slug`).
- Verification-report emission.

Reuse existing functions where possible: `load_or_build_master_order`, `select_prefix_demos`,
`ensure_variant_activation`, `obtain_textual_vti_v2_from_stacks`. The only substituted input is
the per-demo image path; the only substituted output locations are the cache namespace and the
direction directory.

## Dataset construction

Shuffled-control variant of the deployed nd200 `all` demo set: the same 200 demo ids (from
`ids_used` in the deployed `metadata.json`), same order, same truthful/hallucinated caption pairs.
Each demo's image path is overridden to another demo's image via a single derangement over the 200
ids — no demo keeps its own image; seed 1234; map written to disk. Size 200. Demo selection is
model-independent, so one derangement map applies to both models. Purpose: preserve the caption
contrast, destroy the image-caption binding.

## Models (cells)

Two cells, same construction and checks in each.

| Cell | Model | model_short | HF id (confirm against `src/model.py` `create_wrapper`) |
|---|---|---|---|
| LLaVA | LLaVA-1.5-7B | `llava-1.5-7b-hf` | `llava-hf/llava-1.5-7b-hf` |
| Qwen2.5 | Qwen2.5-VL-7B | model_short the deployed Qwen2.5 direction was written under | `Qwen/Qwen2.5-VL-7B-Instruct` |

Use the same wrapper-loading path the deployed directions used. Do not change model, tokenizer, or
the numpy/transformers/tokenizers pins.

## Procedure

1. Read `ids_used` and `direction_layer_norms` from the deployed direction's `metadata.json` at
   `experiment_artifacts/vti/{model_short}/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/`.
2. Generate the derangement over the 200 ids (seed 1234); validate no fixed point; write the map
   plus seed to `shuffled_control_image_derangement_nd200_s1234.json`. This map is
   model-independent — generate it once and reuse for both cells.
3. Build the shuffled `flat_demos` (same ids, order, and captions as deployed; image path
   overridden per the map).
4. For each model: extract the shuffled-control direction into the fresh cache namespace under
   `shuffled_control/_act_cache`; write the direction to `shuffled_control/all_nd200/`.
5. Emit the verification report for that model.

## Checks and report contents

Report primitives only; no cosine.

Gating checks (any failure means the shuffled direction is not a valid control — stop and fix
before it is trusted):

- Derangement validity — number of demos paired with their own image. Expect 0.
- Captions unchanged — number of demos whose forwarded (truthful, hallucinated) caption pair
  equals their own pair from the demos file. Expect 200 of 200.
- Same demos / order / token — number of positional mismatches between the shuffled run's used-id
  list and the deployed `ids_used`. Expect 0. Direction array shape. Expect
  `(num_layers, hidden_dim)`.
- No cache reuse — number of forward passes actually executed against the empty shuffled cache
  namespace. Expect 400 (200 demos × 2 caption variants: `value` and `all`). A count below 400
  means activations were loaded from an existing cache and the swap did not take effect.

Non-gating check (does not decide validity; catches a broken extraction before plotting):

- Well-formedness — per-layer magnitude (L2 norm) of the shuffled-control direction, printed
  alongside the deployed direction's `direction_layer_norms`, per layer index. Expect finite
  values in the same rough range. This is a well-formedness readout, not the magnitude comparison
  that belongs to the next plan.

## Artifacts

- `experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json` — demo-id →
  source-image-demo-id map and the seed. Model-independent; written once.
- `experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/directions.npz` — the
  shuffled-control direction.
- `experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/metadata.json` — records the
  deployed direction this is the control for (`demosv2_9a44f4af_all_nd200_s42_r2_prefix`), the
  derangement file name and seed, ids used, and per-layer magnitudes, for lineage.
- `shuffled_control_sanity_report_{model_short}.md` — the check primitives above.

Logging is local only, under `experiment_artifacts/vti/{model_short}/shuffled_control/`. No W&B.

## What confirms the control is valid

Fixed-point count 0, caption match 200 of 200, positional id mismatches 0, 400 forward passes
executed per model, finite per-layer magnitudes in the same rough range as the deployed direction,
and direction shape `(num_layers, hidden_dim)`. All must hold in both cells. Any deviation stops
progression to the comparison plan until it is fixed.

## Open questions

- Qwen2.5 `model_short`: fill from the deployed Qwen2.5 direction's directory; the plan assumes the
  same slug `demosv2_9a44f4af_all_nd200_s42_r2_prefix` under that model's `textual_v2/`.
- HF ids above are the standard public ids; confirm against `src/model.py` `create_wrapper` before
  loading.
