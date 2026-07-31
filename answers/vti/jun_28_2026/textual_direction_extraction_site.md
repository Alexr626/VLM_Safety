# Textual VTI: activation extraction site vs inference hook site

**Date:** 2026-06-28

## Question

For offline textual steering-vector calculation, are activations taken from the **layer** (post-residual stream) site or the **MLP output** (pre-residual) site?

## Short answer

**Extraction is always at the layer / post-residual site** — the per-layer entries in HuggingFace `outputs.hidden_states` from `wrapper.forward_vl`, last prefill token.

**Inference steering** is a separate knob (`hook_site`): `mlp` (default, matches reference code) or `layer` (paper-style residual stream). The **same cached direction vectors** are used for both hook sites; only where `steer()` is applied changes.

There is therefore an **extraction–application mismatch** for all `*_mlp` interventions: directions fit on post-residual states, steered on MLP sub-block outputs. This mirrors the authors' reference (`hidden_states` for PCA, `VTILayer` after MLP for eval). `*_layer` interventions align extraction and application sites.

## Code path (extraction)

`evaluation/interventions/vti/directions.py` → `get_hiddenstates()`:

```python
hidden_states, _, _ = wrapper.forward_vl(image, text)
# hidden_states[L][0, -1, :] for each L in 0..num_layers
```

`forward_vl` calls the full VLM with `output_hidden_states=True` and returns `outputs.hidden_states`. In Transformers decoder models, tuple index `0` is embeddings; index `L≥1` is the **full decoder block output after both residual adds** (attention + MLP). This is **not** the isolated MLP submodule output.

PCA is fit on `value − h_value` last-token vectors per layer; cache stores `full[1:]` → one direction per transformer layer `0..L-1`.

## Code path (inference)

`evaluation/interventions/vti/hooks.py` → `vti_hook_ctx(hook_site=...)`:

| `hook_site` | Module hooked | Tensor semantics |
|-------------|---------------|------------------|
| `mlp` | `dispatch.get_mlp(wrapper, i)` | MLP sub-block output, **before** residual add |
| `layer` | `dispatch.get_layer(wrapper, i)` | Full decoder layer output, **after** residuals |

`VTITextualIntervention` default: `hook_site="mlp"` (reference-faithful).

## What the Phase 1 plan specified

`implementation_plans/vti_phase1_textual.md` **§3 (Direction computation)** explicitly says:

- Rewrite `get_hiddenstates` to use `wrapper.forward_vl(image, text)`.
- Pull `hidden_states[layer][:, -1, :]` per layer (last prefill token).

It does **not** offer a choice of MLP vs layer for extraction. It describes the `hidden_states` tuple from a full forward — i.e. **post-residual layer outputs**.

**§5** originally described hooks on "decoder layer outputs"; the implemented cross-product (`hook_site` × `variant`) was added later (see `answers/concepts/jun_19_2026/vti_rotation_strength_motivation.md`). The plan's registry sketch (`vti_textual_additive`, etc.) predates the `_mlp` / `_layer` suffixes.

**Not specified in the plan:** that extraction site and default steering site would differ. That follows from porting reference extraction (`output_hidden_states`) while also exposing both hook sites for the paper-vs-code comparison grid.

## Reference VTI (authors' code)

| Stage | Site |
|-------|------|
| `get_hiddenstates` / `obtain_textual_vti` | `model(..., output_hidden_states=True).hidden_states` — post-residual |
| `add_vti_layers` at eval | `layer.mlp = Sequential(original_mlp, VTILayer(...))` — **MLP output** |

Same extraction/application split as our `*_mlp` path.

## Implications for experiments

- **`vti_textual_*_mlp`**: directions = layer-stream PCA; steer at MLP (weaker effective intervention; reference default).
- **`vti_textual_*_layer`**: directions = layer-stream PCA; steer at layer (stronger; needs separate β calibration — see rotation-strength sweeps).
- **Future variant to consider:** extract directions from MLP-hook captures (would require a separate extraction pass with forward hooks, not `output_hidden_states` alone) so `*_mlp` extraction and application match.

## Preview: visual arm (not implemented)

Reference visual VTI (`get_visual_hiddenstates`, `obtain_visual_vti`, `add_vti_layers` on `vision_model`):

- **Extraction:** ViT `output_hidden_states` — full per-block outputs, all patch tokens.
- **Steering (reference):** `VTILayer` chained after each ViT **MLP** (same pattern as textual: layer-stack states for PCA, MLP site for intervention).

Our planned vision port (`IMPLEMENTATION.md` §Visual encoder hooks) should treat **extraction site** and **hook site** as explicit, independent design choices — especially for Qwen2-VL's dynamic patch counts.
