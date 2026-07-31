# Visual VTI (LLaVA) — feasibility review of implementation plans

**Date:** 2026-06-28  
**Plans reviewed:** `plan_visual_vti_reimplementation.md`, `vti_visual_discrepancies_and_smoke_plan.md`

## Verdict

**Build (vision arm on LLaVA-1.5): feasible** on lambdab2 with the existing textual-VTI patterns (`steer.py`, hook context managers, `experiment_artifacts/` caching). Expect ~1–2 days for core plumbing plus a reference-parity test script.

**Smoke plan: partially blocked** — the gate readout (c-AUC, c-curve, intervention-agnostic driver) does **not** exist yet and is non-trivial net-new work on top of the vision arm. Stage 0 textual gate validation is feasible once that machinery exists.

---

## What aligns well with the codebase

| Plan claim | Code reality |
|------------|--------------|
| LLaVA vision path `wrapper.model.vision_tower` | Correct for `LLaVAWrapper` (`llava-hf/llava-1.5-7b-hf`). Uses HF `processor` + `model.generate(**inputs)`; ViT runs inside `LlavaForConditionalGeneration` forward. |
| 24 layers, dim 1024, 576 patch tokens in LLM | Correct (`num_image_tokens` = 576; ViT internal seq = 577 with CLS). |
| Reuse `steer()` from `evaluation/interventions/vti/steer.py` | Correct — geometry-agnostic on `(batch, seq, hidden)`. |
| ViT runs once per image on prefill | Correct — hooks on `vision_tower` fire during `generate_vl` / `forward_with_logits`. |
| `compute_yes_prob` can see steered vision | **Conditionally true** — `forward_with_logits` → `wrapper.model(**_prepare_vl(...))` includes vision tower; hooks work if `compute_yes_prob` is called **inside** `visual_hook_ctx`. Not automatic; smoke driver must wire this. |
| Demo images `data/vti/demos.jsonl` + `data/coco/train2014/` | Present; visual arm uses images only (captions irrelevant). |
| AMBER-25 sample IDs | Pinned in `evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/amber/...json` (`ordered` list, 25 ids). |
| CHAIR-5 sample IDs | Pinned in `evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/chair/...json` (5 ids; seed 5678 draw from CHAIR-500). |
| h− / h+ flip accounting | Reusable from `evaluation/vti_rotation_strength/rotation_strength.py` (`flip_fp_to_tn` = h−, `flip_tn_to_fp` = h+). |

---

## Misplaced or underspecified assumptions (resolve before build)

### 1. PCA layout — **most important**

The build plan describes `top_pc` as PCA **per (layer, token)** with demos as the batch dimension.

The **reference** `obtain_visual_vti` does something different:

1. Stack all ViT `hidden_states` layers (including embedding row) → tensor `(n_layers_in_stack, 577, 1024)`.
2. For each demo, `Δ = masked − clean`, then **`reshape(577, -1)`** — concatenating **all layers** into one vector per token (`577 × (25·1024)`).
3. `fit_data` shape `(577, num_demos, n_layers·1024)` — PCA **per token** across demos on the **layer-concatenated** vector.
4. Reconstruct direction, `view(n_layers, n_tokens, 1024)`, caller slices `[1:]`.

So `legacy_pc_plus_mean` must reproduce the **per-token, all-layers-concat** layout, not per-(layer,token) PCA.

`top_pc` as specified is an **intentional alternate** (paper wording), not a tweak of the legacy path. The acceptance test “numerical parity with reference” applies only when **all** of these match: `legacy_pc_plus_mean`, `mask_fill=mean`, `mask_ratio=0.99`, `num_trials=50`, `data[:70]`, and the concat PCA layout.

**Also:** reference includes the **embedding row** in the concat, then drops it via `[1:]` on the final direction. The plan drops embedding earlier. For legacy parity, embedding must participate in the concat step even if steering skips layer 0.

### 2. Demo subsampling differs from textual arm

| Arm | Demo selection |
|-----|----------------|
| Textual (current) | `random.sample(demos, 70)` with `seed=42` |
| Visual (plan) | `data[:70]` file order (matches reference `get_demos`) |

Not wrong for visual parity, but textual and visual directions are fit on **different 70-image sets** unless aligned deliberately.

### 3. `IMPLEMENTATION.md` has a stale visual-demo sentence

§Visual encoder hooks item 6 says vision PCA contrasts “clean vs hallucinated caption conditions on the same image.” That is **textual-arm logic**. Visual arm = **masked vs clean image** (plan and reference are correct).

### 4. ShareGPT4V `_encode_image` path is not LLaVA-1.5

`IMPLEMENTATION.md` documents ShareGPT4V’s separate `_vision_tower` / `_encode_image` path. **`LLaVAWrapper` does not use that** — it uses standard HF `processor` + `model(**inputs)`. Vision hooks attach to `wrapper.model.vision_tower`; extraction calls `vision_tower(pixel_values, output_hidden_states=True)` directly (as reference does), not a custom encode helper.

### 5. Eval harness gaps

- `run_eval.py` / `eval_runner` only thread `--beta` into `get_intervention(..., beta=...)`. Vision needs **`--alpha`** (or a unified coefficient with arm-specific naming) and result-dir encoding (e.g. `vti_visual_additive_layer__a0.4` vs textual `__b{beta}`).
- No `vti_visual_*` interventions exist yet (expected).

### 6. Gate machinery does not exist

Smoke plan Stage 0–2 assumes:

- **c-AUC** from per-item `p_yes` under active hooks
- **c-curve** (threshold sweep on baseline `p_yes`)
- Intervention-agnostic gate driver

None of this is implemented. `compute_yes_prob` exists; the aggregation/plotting layer does not. Budget this as part of the smoke deliverable (or descope Stage 0 until gate lands).

### 7. Reference parity test vs “no runtime import”

Build plan says never import vendored `VTI/` at runtime, but acceptance check #3 requires numerical parity against `obtain_visual_vti`. **Feasible** as an isolated test script (`tests/test_visual_vti_reference_parity.py` or `diagnostic_experiments/...`) that imports reference code only in tests — not in production `get_intervention` path.

### 8. `mm_vision_select_layer` (-2) vs steering all 24 blocks

LLaVA inference projects features from **layer −2** and **drops CLS** before the LLM. Visual steering still hooks all 24 encoder blocks (matching reference). Effect propagates through later layers into what gets projected — intentional, but worth noting when interpreting “which representation” is steered vs what the LLM sees.

---

## Feasibility by component

| Component | Feasibility | Notes |
|-----------|-------------|-------|
| `VisionDispatch` + `verify_layout` | High | Straightforward for LLaVA CLIP path. |
| `visual_perturb.py` | High | Port `mask_patches` on `(C,H,W)` CLIP tensors; verify processor output shape `(1,C,H,W)`. |
| `capture_visual_hiddenstates` | High | Direct `vision_tower` forward; running-mean over trials. |
| `obtain_visual_vti` + cache | Medium | **Clarify PCA layout** (above); two reconstruction branches. |
| `visual_hook_ctx` + 4 interventions | High | Mirror `vti_hook_ctx`; per-position `steer()`. |
| `run_eval` α wiring | Medium | Small harness change. |
| Reference parity test | Medium | Needs vendored import in test env + aligned preprocessing. |
| Perturbed PNG dump | Medium | De-normalize CLIP mean/std correctly. |
| Smoke gate (c-AUC, c-curve) | Medium–High effort | Net-new; not in repo today. |
| Full Stage 1 grid (12 cells × 2 benchmarks × gen) | High runtime | Manageable at n=25+5; direction extraction (70×50 ViT forwards) is the heavy one-time cost per config. |

---

## Clarifying questions (for Romanus before implementation)

1. **PCA layout:** Confirm `top_pc` = per-(layer,token) PCA (plan as written) and `legacy_pc_plus_mean` = reference concat-across-layers-per-token layout. Should legacy parity also include embedding in the concat before `[1:]` slice?

2. **Demo alignment:** Keep visual `data[:70]` for reference fidelity, or switch to `seed=42` sample of 70 to match textual directions?

3. **Gate scope:** Build c-AUC + c-curve + smoke driver in this PR, or implement vision arm first and gate in a follow-up (deferring Stage 0)?

4. **Alpha in eval tree:** Prefer `__a{alpha}` suffix for visual cells, or encode α only in `intervention_config` / `metric_summary.json`?

5. **Smoke driver:** Dedicated `diagnostic_experiments/vti_visual_smoke/` script (recommended — many configs, gate readout, subset ids) vs extending `run_eval.py`?

6. **Stage 1 scope:** Run all 12 visual cells × 2 benchmarks on first pass, or start with paper cell (`additive_layer`) + code cell (`uniform_rotation_mlp`) before full grid?

7. **Reference parity tolerance:** What rtol/atol is acceptable given `transformers==4.50.1` vs authors' older stack?
