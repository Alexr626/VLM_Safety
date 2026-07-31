# VTI Re-Implementation — Phase 1: Textual Arm

**For:** Cursor (implementation agent), against the VLM hallucination-mitigation repo.
**Scope of this file:** the **text-decoder** intervention only, for **all four target models**, with **three explicitly distinct steering variants** and one diagnostic. The vision-encoder arm is **Phase 2** (separate plan) — see "Phase 2 on the horizon" below.
**Source material:** authors' VTI reference (`VTI/` subdir — read-only reference, out of pipeline scope), `IMPLEMENTATION.md`, `src/model.py`.

**Identifier convention:** `code font` items confirmed present in `model.py` / `IMPLEMENTATION.md` are real. Items tagged **[NEW]** must be created. Do not invent module paths; if something needed isn't confirmed, flag it, don't guess.

---

## 0. Why textual-only, and why first

The text-decoder intervention maps onto interfaces that already exist and are confirmed in `model.py`: `forward_vl` returns `outputs.hidden_states` (the language model's per-layer hidden states), and `num_layers` / `hidden_dim` resolve correctly for every target model (including the nested `text_config` path used by the Qwen2 family, `model.py` lines 1456–1467). Decoder-layer hooks are analogous to the existing `patch_hook_ctx` discipline in `src/mediation.py`. **No new vision-encoder accessors are required for Phase 1.**

This is also a legitimate research result on its own: textual-only VTI is a real ablation, and comparing the three steering *geometries* on the decoder is the sharpest form of the rotation-vs-displacement question (which is the project's actual direction). The visual arm (Phase 2) is net-new infrastructure across four architecturally distinct encoders; it is deliberately not bundled here.

---

## 1. Hypothesis / success criteria

VTI's textual intervention adds a steering signal to decoder-layer outputs to reduce hallucination. We implement three steering geometries and compare them.

**Confirms:** on LLaVA-1.5-7B, at least one textual variant improves POPE accuracy/F1 over the `no_intervention` baseline (confirmed baselines below) without collapsing informativeness.

**Baselines already on disk** (`IMPLEMENTATION.md` §"POPE no-intervention baselines (2026-06-16)", 200 samples × 3 splits):

| Model (HF id) | `model_short` | Avg Acc | Avg F1 |
|---------------|---------------|--------:|-------:|
| `llava-hf/llava-1.5-7b-hf` | `llava-1.5-7b-hf` | 84.7% | 85.2% |
| `Qwen/Qwen-VL-Chat` | `qwen-vl-chat` | 85.7% | 85.1% |
| `Qwen/Qwen2-VL-7B-Instruct` | `qwen2-vl-7b-instruct` | 88.0% | 87.4% |

(No baseline yet for Qwen2.5-VL — run `no_intervention` for it as part of this phase; see §7.)

**Falsifies / null:** no variant beats baseline, **or** a variant "improves" hallucination metrics only by suppressing object mentions (degenerate). Guard against the latter with the informativeness check (§7) and by inspecting raw generations, not just metrics.

**Important scope caveat:** Phase 1 cannot reproduce the *paper's* headline numbers, because the paper's reported results use **both** sites (visual + textual). Phase 1 validates the textual mechanism and the three-variant comparison; full paper reproduction is gated on Phase 2.

---

## 2. The three steering variants — implement all three, switchable

These are three genuinely different operations, discovered by comparing the paper's described method against the reference code (`VTI/vti_utils/llm_layers.py`, `VTILayer.forward`). Name them **exactly** as below. A single config field selects the variant.

Let `x` be a decoder-layer output, shape `(batch, seq, hidden)`; `d` the per-layer steering direction (unit-normalized inside the op); `alpha` the strength.

### Variant `additive` — *what the paper describes*
Plain displacement. Norm not preserved.
```
x_out = x + alpha * normalize(d)
```

### Variant `uniform_rotation` — *what the reference code actually runs*
Spherical renormalization with a **constant** strength gate (`lambda_sim = 1.0`). The hardcoded `0.1` coefficient and the renorm-onto-sphere are the method.
```
norm   = ||x||                              # per-token, keepdim
d_unit = normalize(d)
y      = alpha * 1.0 * d_unit               # lambda_sim == 1.0
x_out  = normalize( normalize(x) + 0.1 * y ) * norm
```
Output keeps the input's per-token L2 norm: the steer **rotates** the activation on a fixed-radius sphere rather than translating it.

### Variant `gated_rotation` — *what the reference code gestures at but disables*
Identical to `uniform_rotation` except `lambda_sim` is **activation-dependent**, reviving the commented-out branch in the reference. During single-token decode (`x.size(1) < 2`):
```
lambda_sim = 1.0 + max(0, cosine_similarity(x, -d_unit))      # per token, clamped at 0
```
For multi-token chunks (`x.size(1) >= 2`, i.e. prefill) `lambda_sim = 1.0`, matching the reference's branch structure. Interpretation (derived from the expression, **to be verified empirically** by the diagnostic in §6, not trusted): tokens whose activation points *away* from `d` (toward `-d`) get amplified steering; already-aligned tokens get baseline steering. The cosine sign and the `1.0 +` offset are easy to transcribe wrong — confirm against `VTI/vti_utils/llm_layers.py` line by line.

### Reference op (adapt; verify shapes against real hidden states)
```python
import torch
import torch.nn.functional as F

def steer(x, direction, alpha, variant, eps_coeff=0.1):
    # x: (batch, seq, hidden); direction: (hidden,) for the hooked layer
    d = F.normalize(direction.float(), dim=-1)
    if variant == "additive":
        return (x.float() + alpha * d).to(x.dtype)

    norm = x.float().norm(dim=-1, keepdim=True)
    if variant == "gated_rotation" and x.size(1) < 2:
        lam = 1.0 + torch.clamp(
            F.cosine_similarity(x.float(), -d[None, None, :], dim=-1), min=0.0
        ).unsqueeze(-1)
    else:                       # uniform_rotation, or gated during prefill
        lam = 1.0
    y = alpha * lam * d
    x_out = F.normalize(F.normalize(x.float(), dim=-1) + eps_coeff * y, dim=-1) * norm
    return x_out.to(x.dtype)
```

**Caching interaction (load-bearing).** Whether the `x.size(1) < 2` branch ever fires at eval time depends on KV caching. In `model.py`, `Qwen2VLWrapper._generate_from_inputs` uses `use_cache=True` (line 1604) — so during decode each step feeds a single token, `x.size(1) == 1`, and the gated branch **fires every decode step**. The reference's MMHal runner used `use_cache=False` (whole sequence re-fed; branch rarely fires). Confirm each wrapper's `generate_*` cache setting and record it; if any wrapper generates with `use_cache=False`, the gated variant will behave differently there. **Do not change wrapper cache defaults** to suit VTI without flagging it — it affects baseline comparability.

---

## 3. Direction computation (textual)

Port the textual-direction core from `VTI/vti_utils/utils.py`, rewritten to go through the wrapper instead of LLaVA globals.

**Port verbatim:** `pca.py` (the batched PCA/SVD `nn.Module`) — no changes.

**Rewrite to use the wrapper:**
- `obtain_textual_vti(...)` and its helper `get_hiddenstates(...)`. The reference calls the model directly with LLaVA-specific signatures (`model(input_ids, images=..., output_hidden_states=True)`). Replace with `wrapper.forward_vl(image, text, output_attentions=False)`, which returns `(hidden_states, attentions, input_ids)` for **all four** models (`model.py`). Pull `hidden_states[layer][:, -1, :]` per layer (last prefill token) — consistent with `get_last_token_activations` in `extraction.py` (`IMPLEMENTATION.md` lines 203–206).

**Paired demo structure.** Each demo provides a *clean* caption (`value`) and a *hallucinated* caption (`h_value`). The textual direction at each layer is computed (per the reference) from the difference of last-token hidden states between the clean and hallucinated forward, reduced by PCA (`rank=1`). The paired captions come from the demo file (§4).

**Per-layer direction shape.** The result is one direction vector per decoder layer, each of size `hidden_dim`. The reference slices `vti_text[1:]` — dropping index 0 (the embedding-layer row of the `hidden_states` tuple, which has `num_layers + 1` entries). **[CHECK]** confirm the hidden_states tuple length is `num_layers + 1` for each wrapper and that index 0 is the embedding output, so the slice aligns directions to transformer layers `0..num_layers-1`.

**Cache directions to disk.** Extraction is deterministic given seed and moderately expensive. Save per-model under `experiment_artifacts_dir("vti", model_short)` (`src/paths.py`) as `.npz`, keyed by `(model_short, num_demos, rank, seed)`. Recompute only on cache miss.

---

## 4. Demo data

Ships with the reference: `VTI/experiments/data/hallucination_vti_demos.jsonl` (COCO entries; each has `value` = clean caption, `h_value` = hallucinated caption, `image` = `COCO_train2014_*.jpg`, `question` = "Describe this image in detail.").

- **[CHECK] train2014 vs val2014.** Demos reference **train2014**, but `IMPLEMENTATION.md` (on-disk layout) documents `data/coco/val2014/` for POPE/CHAIR. Confirm train2014 images are present, or add a download step. **Do not silently substitute val2014.**
- Copy the JSONL to `data/vti/demos.jsonl`; add a path helper rather than a hard-coded relative path.
- Defaults: `num_demos=70`, `rank=1`, fixed `seed`. (`num_trials` / `mask_ratio` are visual-arm-only — Phase 2.)
- Demo images are fed through each wrapper's own processor via `forward_vl` — no resolution handling needed here (textual side reads decoder hidden states regardless of vision-token count).

---

## 5. The intervention class

Subclass `InterventionBase` per `IMPLEMENTATION.md` §"Adding new components".

```python
# evaluation/interventions/vti/intervention.py   [NEW]
class VTITextualIntervention(InterventionBase):
    def __init__(self, model_id,
                 variant="uniform_rotation",      # additive | uniform_rotation | gated_rotation
                 alpha_text=0.9,
                 num_demos=70, rank=1, seed=42,
                 eps_coeff=0.1,
                 demos_path=..., direction_cache=...,
                 log_lambda_sim=False):           # diagnostic toggle, see §6
        ...
    # Lifecycle:
    #   1. on first use: compute or load cached per-layer textual directions
    #   2. register forward hooks on DECODER layers only (resolve the decoder
    #      layer ModuleList the same way mediation.py's dispatch does)
    #   3. generate(wrapper, image, question, max_new_tokens, caption=None)
    #         -> wrapper.generate_vl with hooks active
    #   4. remove hooks after generation (context manager, so a crash can't
    #      leave a poisoned model)
```

Design points:
- **Hooks, not module surgery.** The reference mutates structure (`layer.mlp = nn.Sequential(...)`). Use forward hooks on decoder layer outputs instead — reversible, composes with `generate_vl`, matches `mediation.py`. Preserve the `steer()` math exactly (§2).
- **Decoder layer list resolution.** `src/mediation.py` already resolves per-family decoder layers via `get_dispatch(wrapper)` / `FamilyDispatch.get_layer(wrapper, i)` (families `llava`, `llama_raw`, `qwen_vl_chat`; `IMPLEMENTATION.md` lines 258–271). Reuse this to attach hooks. **[CHECK]** confirm a dispatch family resolves for `Qwen2VLWrapper` (covers Qwen2-VL **and** Qwen2.5-VL, which share the wrapper) and for `QwenVLWrapper`. If `qwen2` has no dispatch entry, adding one is in-scope for Phase 1 (it's decoder-side only).
- **Hook teardown discipline** mirrors `patch_hook_ctx`. Never leave hooks attached across samples or into a baseline run sharing the wrapper.
- **`intervention_config`** (written into `metric_summary.json`) must capture `variant, alpha_text, num_demos, rank, seed, eps_coeff`.

Register in `evaluation/interventions/__init__.py`: add to `ALL_INTERVENTIONS` and handle in `get_intervention`. Suggested registry names so variants are independently runnable/comparable:

```python
ALL_INTERVENTIONS = [
    "no_intervention",
    "vti_textual_additive",
    "vti_textual_uniform_rotation",
    "vti_textual_gated_rotation",
]
```
`get_intervention` maps each to `VTITextualIntervention(variant=...)`.

---

## 6. Diagnostic — per-token `lambda_sim` under the gate

**Question:** when `gated_rotation` is active, which tokens actually get amplified, and by how much? This empirically tests the §2 interpretation (the cosine sign is a transcription risk; measure, don't assume).

New diagnostic experiment per `IMPLEMENTATION.md` §"Adding new components → Diagnostic experiment":

```
diagnostic_experiments/vti_lambda_sim/
    run_scripts/run_vti_lambda_sim.sh
    compute_lambda_sim.py
    plot_lambda_sim.py
```

- When `log_lambda_sim=True`, the `gated_rotation` hook records, per generated token and per hooked layer: the decoded token string, `lambda_sim` value, and `cosine_similarity(x, -d_unit)` (pre-clamp). Buffer in memory; flush per sample.
- `compute_lambda_sim.py --model <id> --benchmark pope --limit N`: runs generation under `gated_rotation` with logging on; writes per-token records to `diagnostic_results_dir("vti_lambda_sim", model_short)`.
- `plot_lambda_sim.py --model <id>`: distribution of `lambda_sim` across tokens; mean `lambda_sim` by layer; and the top-amplified token types (do content-bearing object tokens get steered harder than function words?). Write to `diagnostic_plots_dir(...)`.
- Use `src.paths` helpers throughout; optionally also write to `experiment_artifacts_dir("vti_lambda_sim", model_short)`.

This is decoder-side only, so it lives fully in Phase 1. **Caching note:** the gate only fires per-token when generation uses `use_cache=True` (§2). The Qwen2 wrapper does; confirm the others. If a wrapper generates with `use_cache=False`, `lambda_sim` will be logged at prefill granularity instead of per-token, which is a different (and less interesting) measurement — note it in the output if so.

---

## 7. Validation gates (do not skip)

1. **No-op check:** `alpha_text=0` (any variant) must reproduce `no_intervention` numbers exactly. If not, hooks alter activations when they shouldn't.
2. **Sphere check (rotation variants):** after `uniform_rotation` / `gated_rotation`, assert `||x_out|| ≈ ||x||` per token on a random tensor unit test. `additive` should *fail* this (sanity that the variants differ).
3. **Variant-distinctness check:** on a fixed sample, the three variants must produce measurably different generations (or different logits). If `additive` and `uniform_rotation` give identical output, the variant switch isn't wired.
4. **Shape assertions** in `steer()` and in `obtain_textual_vti` — per-layer direction size == `hidden_dim`; number of directions == number of hooked decoder layers.
5. **Informativeness:** mean generation length per variant must not collapse vs baseline. A large drop = degenerate-suppression false positive, not a real hallucination win.
6. **Qwen2.5-VL baseline:** no `no_intervention` baseline exists yet for `Qwen/Qwen2.5-VL-7B-Instruct`. Run it (200 × 3 POPE splits) before comparing any VTI variant on that model, so the comparison has a reference. Record vision-token count (Policy A, dynamic) in the `RESEARCH_LOG.md` entry.

---

## 8. Run commands

Per-model, comparing baseline against all three textual variants on POPE (extend with `amber` once POPE looks sane):

```bash
for split in random popular adversarial; do
  python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --benchmarks pope \
    --interventions no_intervention \
                    vti_textual_additive \
                    vti_textual_uniform_rotation \
                    vti_textual_gated_rotation \
    --pope_split $split --limit 200 \
    --output_dir evaluation/results --skip_if_exists
done
```

Target models for Phase 1 (all textual-arm-ready today):
`llava-hf/llava-1.5-7b-hf`, `Qwen/Qwen-VL-Chat`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`.

`lambda_sim` diagnostic (gated variant only), LLaVA first:
```bash
MODEL=llava-hf/llava-1.5-7b-hf \
  bash diagnostic_experiments/vti_lambda_sim/run_scripts/run_vti_lambda_sim.sh
```

---

## 9. Logging

- Local: existing `evaluation/results/{model_short}/{benchmark}/{intervention}/` schema (`responses.json`, `metric_summary.json`).
- W&B: optional; public benchmarks, so permitted. Defer unless run volume grows (current two-file handoff stands).
- **Append to `RESEARCH_LOG.md`** after each model: exact commands, commit, baseline-vs-three-variant POPE acc/F1 (with `by_category` per split), Qwen2.5-VL baseline numbers + vision-token count, and 3–5 raw generation samples per variant for the informativeness check.

---

## Phase 2 on the horizon (context, not scope)

Phase 2 adds the **vision-encoder arm** — VTI's visual intervention — for the same four models. It is **not** part of this plan and requires net-new infrastructure the repo has never had: the vision-encoder hidden states are not exposed by any wrapper today (confirmed: `model.py` public surface and `extraction.py`'s documented API are entirely decoder-side / last-token). Phase 2 will add, per model, **[NEW]** accessors for standalone vision-tower forward with `output_hidden_states=True`, the encoder block `ModuleList` for hook registration, patch-token count, and vision hidden dim — plus handling for the Qwen family's **dynamic/variable vision-token count** (Policy A), which the authors' fixed-576-grid PCA and patch-masking never contemplated, and Qwen2.5-VL's window-attention encoder specifically.

Relevant forward-looking note for Phase 1 implementers: keep the `steer()` op and the variant switch **model-agnostic and site-agnostic** (it takes `x` and `direction`, nothing wrapper-specific), so Phase 2 can reuse the identical op on vision-encoder layer outputs without modification. The only Phase-2 additions should be direction *extraction* (visual) and hook *registration site* (vision tower) — not new steering math.
