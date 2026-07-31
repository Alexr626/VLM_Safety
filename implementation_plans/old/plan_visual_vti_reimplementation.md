# Re-implementation plan — VTI visual steering arm (LLaVA-1.5)

**For:** Cursor implementation agent (Remote-SSH, lambdab2, repo open).
**Author:** research/analysis agent, via Romanus.
**Companion doc:** `vti_visual_discrepancies_and_smoke_plan.md` — the code-vs-manuscript
discrepancy ledger and the smoke-test plan that exercises the non-default configurations.
Read it after this file; this file is the build spec, that file is the test spec.

**Scope:** LLaVA-1.5 only. Net-new vision-encoder arm of VTI (the `alpha_image` side),
currently not implemented. Per-patch directions (faithful to reference), ONE extraction
site, FOUR application variants. Every experimental variable below is an exposed,
recorded parameter; **defaults are paper-faithful**, and the authors' code-faithful
configuration must be exactly reproducible by flags. Qwen2-VL/2.5 deferred (dynamic
patch counts; reference has no Qwen2 path).

**Reference source (vendored, read-only, do not import):**
`VTI/vti_utils/icv_utils.py` (`get_demos`, `mask_patches`, `get_visual_hiddenstates`,
`obtain_visual_vti`) and `VTI/vti_utils/llm_layers.py` (`VTILayer`, `add_vti_layers`).
Port semantics to our repo's interfaces; never call their code at runtime.

---

## Design decisions already made (do not re-litigate; rationale in companion doc)

- **Per-patch directions**, not pooled: direction tensor is `(n_layers, n_tokens, 1024)`.
- **One extraction site:** ViT **post-residual layer outputs** (HF `hidden_states`) — this
  is what both manuscript and code use. Four application modes reuse this one direction.
- **Four intervention variants** (mirror the textual arm's grid):
  `vti_visual_additive_mlp`, `vti_visual_additive_layer`,
  `vti_visual_uniform_rotation_mlp`, `vti_visual_uniform_rotation_layer`.
- **Sign conventions are per-arm and both are correct** (no discrepancy): visual =
  `masked − clean`; textual = `clean − hallucinated`. Preserve each arm's sign exactly;
  do NOT unify them.
- **Embedding layer: never steered, and final directions cover encoder blocks 0..23
  only** — but the capture retains the embedding row, because the legacy parity branch
  requires it to participate in the PCA before the reference's `[1:]` slice (step 4).
  `top_pc` drops it before PCA (equivalent, since per-(layer,token) PCAs are independent
  per layer). Assert direction-for-block-i applies at block i — off-by-one here is the
  classic bug.
- **CLS token included** in extraction and steering (all 577 positions), matching the code.
  Expose `include_cls: bool = True` so it can be toggled after smoke review. (The
  reference's `# no CLS token` comment is wrong about its own code — it keeps CLS.)
- **No gating** (`lambda_sim`): excluded entirely, not a variant.
- **Decode branch dead:** the ViT always sees all tokens at once; implement only the
  reference's `x.size(1) >= 2` path.

---

## Parameters (all recorded in `intervention_config` and direction-cache metadata)

| Param | Default (paper-faithful) | Alternates exposed | Source of default |
|---|---|---|---|
| `perturb_type` | `patch_mask` | `gaussian_noise` | manuscript: "randomly mask a few patches"; gaussian from manuscript diagnostics |
| `mask_ratio` | **0.99** | any float | manuscript **appendix** (explicit: "the mask ratio to compute visual direction is set to 0.99") = code; the main text's "a few patches" is an internal manuscript inconsistency, not a paper-vs-code discrepancy (companion doc N1) |
| `mask_fill` | **`zero`** | `mean` (code) | manuscript unspecified; plain masking inferred as blanking |
| `noise_sigma` | 0.1 (only if `perturb_type=gaussian_noise`) | any float | manuscript diagnostics don't pin σ; expose |
| `num_trials` | **50** | any int | manuscript appendix ("average across 50 random masks") = code; no discrepancy |
| `num_demos` | **70** | any int | README run command (manuscript says only "a few examples N") |
| `direction_recon` | **`top_pc`** | `legacy_pc_plus_mean` (code) | manuscript: "first principle direction"; code adds PCA mean back (companion doc §B) |
| `include_cls` | `True` | `False` | code behavior; manuscript silent |
| `alpha` | run-time grid `{0.2, 0.4, 0.9}` | any | manuscript appendix task-specific alphas |
| `eps_coeff` | 0.1 (rotation variants only) | — | hardcoded in reference `VTILayer`; already in our `steer.py` |
| `patch_size` | 14 | — | CLIP ViT-L/14 |

`alpha` is the **vision** coefficient, threaded separately from textual `beta`
(coefficient-naming constraint). This plan is vision-only; no textual arm composed.

---

## Compute routing

- Everything in this plan runs on **lambdab2** (build + the companion smoke). Tag
  `target=lambdab2`. One A6000, single GPU, no CPU offload, no sharding (hooks). Pick a
  free device via `nvidia-smi` → `CUDA_VISIBLE_DEVICES`.
- No RunAI submissions from this plan; scaling is gated on the smoke.

---

## Verified LLaVA vision anatomy (ground truth from IMPLEMENTATION.md)

- `wrapper.model.vision_tower` → `CLIPVisionModel`; per-layer:
  `wrapper.model.vision_tower.vision_model.encoder.layers[i]` (submodules `self_attn`,
  `mlp`, `layer_norm1/2`). 24 encoder layers, hidden dim 1024.
- ViT block structure: `x → x + attn(ln1(x)) → x + mlp(ln2(x))`. So **`mlp` site** = MLP
  sub-block output (pre-residual-add); **`layer` site** = full block output
  (post-residual). Direct analogs of the textual arm's two sites.
- Layer outputs carry **577** positions (CLS at index 0 + 576 patches).
- HF `hidden_states` for the ViT: index 0 = patch-embedding output, 1..24 = encoder block
  outputs (verify on our `transformers==4.50.1`; the `[1:]` slice must drop exactly the
  embedding row).
- The ViT runs **once per image during prefill**; vision hooks fire on that forward only.
  Steering there is what reaches generation — intended.
- **Last-block caveat:** LLaVA projects features from `hidden_states[-2]` (layer −2), so
  steering the final ViT block is a **no-op for generation** (its output is
  `hidden_states[-1]`, which the LLM never sees). We steer all 24 blocks anyway for
  reference fidelity — effectively blocks 1..23 reach the model. Note this when
  interpreting per-layer ablations later (a flat last-layer effect is expected, not a bug).
- Reference registers on `model.model.vision_tower.vision_tower.vision_model` (their
  LLaVA-fork double path). Ours is `wrapper.model.vision_tower`. Port to ours.

**Demo data:** the reference demo file (`hallucination_vti_demos.jsonl` equivalent —
our `data/vti/demos.jsonl`) over `data/coco/train2014/`. The visual arm uses **images
only** (caption fields are textual-arm only). `num_demos=70` default. **Demo selection:
use the SAME seed-42 sample of 70 demos as the textual arm** (not the reference's
`data[:num_demos]` file order), so textual and visual directions are fit on the same
image set; record the selected ids in the direction-cache metadata. This does not affect
the reference-parity test, which feeds both implementations identical inputs. (Cursor:
also confirm which demo-selection helper the reference *driver* actually imports — if a
`random.sample` variant exists alongside `icv_utils.py`'s `data[:n]`, file-order
"fidelity" was never well-defined in the reference anyway; note the finding.)

---

## Build

### 1. `VisionDispatch` (new; sibling to `FamilyDispatch` in `src/mediation.py`)

```python
class VisionDispatch:
    def get_vit_layer(wrapper, i) -> nn.Module    # encoder.layers[i]
    def get_vit_mlp(wrapper, i)   -> nn.Module    # encoder.layers[i].mlp
    def n_vit_layers(wrapper)     -> int          # 24
    def vit_hidden_dim(wrapper)   -> int          # 1024
    def n_tokens(wrapper)         -> int          # 577 incl CLS
    def cls_index(wrapper)        -> int          # 0
```

`verify_layout(wrapper, VisionDispatch)`: assert paths resolve; run a dummy image and
assert each captured layer output has 577 positions and dim 1024. Fail loudly. Run this
before anything else.

### 2. Perturbation module → `evaluation/interventions/vti/visual_perturb.py`

- `mask_patches(image_tensor, indices, patch_size=14, fill="zero"|"mean")`: port the
  reference's patch-grid arithmetic exactly (`patches_per_row = W//14`, row/col from
  flat index, block assignment). `fill="mean"` = reference behavior (per-image spatial
  mean); `fill="zero"` = zeros (default).
- `perturb(image_tensor, perturb_type, mask_ratio, mask_fill, noise_sigma, rng)`:
  - `patch_mask`: `randperm(total_patches)[: int(mask_ratio * total_patches)]` →
    `mask_patches(...)` (mirrors `get_demos`).
  - `gaussian_noise`: `image + noise_sigma * randn_like(image)` (whole image; manuscript-
    diagnostics style).
- Deterministic given a seed; record the seed.
- **Perturbed-image dump (for human review — required):** whenever directions are
  extracted, save a small sample of the perturbed images to disk as PNGs so Romanus can
  see what the perturbation actually does at the configured ratio/fill: the first **3
  demo images × 5 trials each**, plus each clean original, under the direction artifact
  directory (`experiment_artifacts/vti/{model_short}/visual/{config_id}/perturbed_examples/`),
  named `{demo_idx}_trial{t}_{perturb_type}_r{mask_ratio}_{fill}.png` and
  `{demo_idx}_clean.png`. **De-normalize before saving:** the perturbation operates on
  the CLIP-preprocessed tensor (mean/std normalized), so invert the CLIP normalization
  back to pixel space, clamp to [0,1], then save — otherwise the PNGs will look like
  garbage regardless of the perturbation. Note this means mean-fill blocks will render
  as the image's average color and zero-fill blocks will render as the (de-normalized)
  CLIP-mean color, not pure black — name the files by the fill actually applied so this
  isn't confusing on review. Keep the dump to this handful per config, never all 70×50.

### 3. Activation capture → extend `src/extraction.py` or a vision sibling

`capture_visual_hiddenstates(wrapper, dispatch, image_tensor) -> (25, 577, 1024)`:
one ViT forward with `output_hidden_states=True`, stack ALL rows **including the
embedding output at index 0** (the legacy branch needs it inside the PCA; `top_pc`
drops it at direction time), keep all 577 tokens, detach to CPU. (Do NOT reuse the
decoder `_make_capture_hook` — wrong axis: it grabs last-token only.)

Per demo image:
- clean: one capture of the unperturbed image.
- corrupted: `num_trials` captures of independently perturbed copies, **averaged
  elementwise** over trials (reference `average_tuples`) → one `(24, 577, 1024)`.

Memory note: 50 trials × (24·577·1024) fp16 ≈ manageable if you accumulate a running
mean on GPU or CPU rather than stacking all trials; do that (the reference stacks — we
don't need to reproduce its memory profile, only its math).

### 4. Direction computation → `evaluation/interventions/vti/visual_directions.py`

```python
obtain_visual_vti(
    wrapper, dispatch, demo_images,
    perturb_type, mask_ratio, mask_fill, noise_sigma,
    num_trials, num_demos, direction_recon, include_cls, seed,
) -> torch.Tensor  # (24, n_tokens, 1024); n_tokens = 577 or 576 per include_cls
```

Per demo `i`: `Δᵢ = corruptedᵢ − cleanᵢ` (VISUAL sign: masked minus clean, matching
manuscript Eq. 1 and code). Then, **per (layer, token) position** — the token axis is
the PCA batch dimension, exactly as the reference fits per-token:

- `direction_recon="top_pc"` (default, manuscript): drop the embedding row first
  (encoder blocks only — per-(layer,token) PCAs are independent per layer, so this is
  equivalent to slicing after). For each (layer, token), stack the 70 demo diffs
  `(num_demos, 1024)`, take the top principal component (rank-1 PCA), sign-align it to
  the mean diff (dot with mean diff ≥ 0) so the PC points the masked−clean way,
  unit-normalize per position at application time (our `steer()` normalizes `d`;
  storing unnormalized is fine, record which). This is a **clean per-(layer,token) PCA
  — an intentional paper-faithful branch, NOT a tweak of the legacy path** (see below).
- `direction_recon="legacy_pc_plus_mean"` (code): reproduce the reference's tensor
  plumbing **verbatim, scramble included** (layout flag: `legacy_repo_reshape`).
  Critical detail (ledger #16): the reference's `Δ.reshape(n_tokens, -1)` acts on a
  **layer-major** `(25, 577, 1024)` tensor, so it does NOT produce per-token
  layer-concat vectors — it scrambles the batch axis (each PCA row is a 25-token strip
  of a single layer, some straddling layer boundaries). Do NOT "fix" this with a
  `permute(1,0,2)` — parity requires the scramble. The **embedding row participates**
  in the PCA (all 25 rows in), and the `[1:]` layer slice is applied to the FINAL
  direction after `.view(n_layers, n_tokens, -1)`, matching the reference caller.
  Reconstruction: `(pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(1)`.
  Note: the mean term survives the reshape (it commutes — a correct per-(layer,token)
  mean diff); the PC term is fit on garbled rows. This mean-inclusive object is what
  the authors actually steer with; parity with the vendored function is this branch's
  acceptance test.

Cache to `experiment_artifacts/vti/{model_short}/visual/` with FULL metadata (every
param above + seed + demo file hash). One artifact per config; the four intervention
variants all load the same artifact.

### 5. Steering hooks + four interventions

- **Reuse `steer()` from `evaluation/interventions/vti/steer.py`** — it already
  implements both geometries (`additive`: `x + α·d̂`; `uniform_rotation`:
  `normalize(normalize(x) + 0.1·α·d̂)·‖x‖`, the reference `VTILayer` math). Do not write
  a second copy.
- **Per-patch application:** at ViT layer L, the hook applies direction row
  `D[L, t, :]` at token position `t` — position-indexed, `(batch, 577, 1024)` against
  `(577, 1024)`. Assert runtime `seq_len == D.shape[1]`; if `include_cls=False`
  (576-row direction), skip position 0 and apply rows to positions `[1:]` — assert that
  alignment explicitly.
- **Two hook sites:** `vit_mlp` (forward hook on `encoder.layers[i].mlp`, steering its
  output pre-residual — the reference's site) and `vit_layer` (forward hook on
  `encoder.layers[i]`, steering the block output post-residual — the site the direction
  was measured at). Implement as a vision hook context parallel to `vti_hook_ctx`.
- **Register in `ALL_INTERVENTIONS`:** `vti_visual_additive_mlp`,
  `vti_visual_additive_layer`, `vti_visual_uniform_rotation_mlp`,
  `vti_visual_uniform_rotation_layer`. Each subclasses `InterventionBase`, wraps
  `generate_vl`, records every parameter in `intervention_config`.
- Steering applies at all 24 encoder blocks (embedding layer never steered).
- **Harness wiring:** thread `--alpha` through `run_eval.py` → `get_intervention(...)`
  for the `vti_visual_*` family (parallel to `--beta` for textual), and encode it in the
  result-dir suffix as `__a{alpha}` (e.g. `vti_visual_additive_layer__a0.4`), mirroring
  the textual `__b{beta}` convention. Reserve `__a{alpha}__b{beta}` for future combined
  VTI cells. Alpha is also recorded in `intervention_config` / `metric_summary.json`.

### 6. Gate dependency

Confirm `compute_yes_prob` (`src/mediation.py`) returns the **steered** probability when
called inside the vision hook context (the ViT-side hooks must be active during that
forward). If it bypasses the intervention wrapper, add a thin path. This is required by
the smoke plan's gate readout (companion doc); flag if non-trivial.

### 7. Acceptance checks (before handing to the smoke)

1. `verify_layout` passes (577/1024 assertions).
2. Direction cache round-trips with metadata; shapes `(24, 577, 1024)` (or 576).
3. **Reference parity (two-part test; vendored import allowed in the test script ONLY,
   e.g. `tests/test_visual_vti_reference_parity.py`, never in the production path):**
   (i) *Math parity* — feed identical pre-captured activation diffs into our
   `legacy_pc_plus_mean` branch and the vendored `obtain_visual_vti`; require rtol≈1e-5
   in float32 with PCA sign alignment handled (SVD sign is ambiguous). This isolates the
   direction computation from the model stack. (ii) *End-to-end parity* — same images
   through both capture paths; require per-(layer,token) **cosine similarity > 0.999**
   (robust to transformers-version / fp16 drift; fails loudly on layout errors). If (i)
   passes and (ii) fails, the bug is in capture/preprocessing, not the math.
4. Hook smoke: one image, `vti_visual_additive_layer` at alpha=0.4 — assert the steered
   ViT layer outputs differ from clean (cosine < 1) at every hooked layer, no NaNs, and
   generation runs end-to-end.
5. `remove`/context-exit restores the original modules (mirror `remove_vti_layers`
   semantics — no lingering wrapped MLPs between runs).
6. Perturbed-example PNGs saved and visually sane: masked blocks render as flat color
   patches covering ~`mask_ratio` of the image (at 0.99, nearly the whole image with a
   few surviving patches), clean originals look normal — i.e. the de-normalization is
   correct and the patch-grid arithmetic masks where it should.

Everything beyond build + acceptance (which configs to test, on what samples, with what
readout) lives in the companion smoke plan.
