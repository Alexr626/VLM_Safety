# Geometric Comparison: Deployed vs Shuffled-Control Direction

Date: 2026-07-22
Author: analyst-side (Claude)
Status: to be implemented and run by Cursor.
Depends on: `implementation_plans/7-22-26/shuffled_control_direction_sanity_checks_plan_2026-07-22.md`
(all gating checks PASS in both cells, RESEARCH_LOG 2026-07-22).
Stage: 1 of 2 (geometric only). The behavioral arm is a separate plan, written only if these
results motivate it.

## Exploratory question

Does the deployed `all`/nd200 steering direction encode image-caption binding, or is it
reconstructible from caption contrast alone?

Measured by how closely the shuffled-control direction matches the deployed direction, per layer,
within each model. The shuffled-control direction was built from the same 200 demos with the same
caption pairs and each demo's image replaced by another demo's image (derangement, seed 1234,
fixed points 0), so the caption contrast is preserved and the image-caption binding is destroyed.

This is an exploratory comparison, not a hypothesis test. No behavioral runs, no steering, no
model loading.

## Cost and placement

CPU-only. Loads two `.npz` direction files per model and computes per-layer dot products and
norms. No GPU, no forward passes, no model weights. Runtime: seconds.

Analysis outputs go under `diagnostic_experiments/perception_diag/control/geometric_comparison/`.
`experiment_artifacts/` is for raw experiment outputs only (activation arrays, norms, steering
vectors) and is read-only for this plan.

## Inputs (all verified on disk as of 2026-07-22)

Per model, two directions:

| Model | model_short | Deployed direction | Shuffled-control direction | Shape |
|---|---|---|---|---|
| LLaVA-1.5-7B | `llava-1.5-7b-hf` | `experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/` | `experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control/all_nd200/` | (32, 4096) |
| Qwen2.5-VL-7B | `qwen2.5-vl-7b-instruct` | `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/` | `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/all_nd200/` | (28, 3584) |

Both directories carry `directions.npz`, `metadata.json`, `components.npz` and share the npz
schema, so the existing loader in `evaluation/interventions/vti/directions_v2.py` (confirm the
loader name — `load_textual_v2_directions(cache_dir)` as of the last read) reads both without
modification. If it does not, load the npz directly rather than adding a new loader.

## Cells

Two, within-model only. Deployed vs shuffled-control for LLaVA; deployed vs shuffled-control for
Qwen2.5. Cross-model comparison is not meaningful here and is not to be plotted or reported.

## Preconditions (assert before computing; fail loud)

1. Shape and layer-index match. For each model, both directions load as the same
   `(num_layers, hidden_dim)` and are decoder-layer-aligned with the same convention (layer index
   *l* refers to the same decoder layer in both). Expect (32, 4096) for LLaVA and (28, 3584) for
   Qwen2.5. A mismatch means the two directions are not comparable per layer and everything
   downstream is meaningless.
2. Mean-dominance of the shuffled-control direction (sign-trustworthiness precondition). Each
   direction is PC1 + mean. The mean term has a definite sign; PC1's sign is set by `svd_flip` and
   is data-dependent, so at any layer where PC1 is a large fraction of the direction, the sign of
   the cosine could be a decomposition artifact rather than a fact about the representation. A1
   established the deployed direction is mean-dominated (mean over PC1 by 14–310x). This
   precondition checks the same for the shuffled-control direction: report per layer the norm of
   the PC1 component and the norm of the full direction, for both directions, and mark layers
   where the PC1 norm is a large fraction of the direction norm. Where mean-dominance holds, the
   cosine sign is trustworthy; where it does not, that layer's sign is read with caution and
   annotated as such. This does not stop the run — it scopes which signs can be interpreted.

   Source of the PC1 component: `components.npz` in each direction directory, or
   `metadata.json` if the per-layer PC1 norms are recorded there. Confirm which before writing the
   code; do not recompute the SVD.

## Metrics

Primitives except where noted; lineage written out.

1. **Per-layer cosine similarity** between the deployed and shuffled-control direction:
   `cosine_l = dot(deployed_l, shuffled_l) / (norm(deployed_l) * norm(shuffled_l))`.
   A composite of one dot product and two norms. Reported as the headline because the question is
   directional alignment, which is norm-invariant, and the interpretation below is written in it.
   Range −1 to 1. y-axis of the cosine plot.
2. **Per-layer magnitude**, as two primitive curves on one axis: `norm(deployed_l)` and
   `norm(shuffled_l)`, both L2, plotted overlaid against layer index. No ratio — the ratio hides
   which of the two moved. Both are already recorded as `direction_layer_norms` in each
   `metadata.json`; recomputing from the arrays as a cross-check is fine and cheap.
3. **PC1-fraction diagnostic table** (precondition 2 above): per layer, PC1 norm and direction
   norm for each of the two directions. Table only; not plotted.

## Procedure

For each model independently:

1. Load both directions; assert precondition 1.
2. Compute the PC1-fraction table; record which layers are mean-dominated (precondition 2).
3. Compute per-layer cosine and per-layer norms.
4. Write the CSV, then the two plots.

## Artifacts

Under `diagnostic_experiments/perception_diag/control/geometric_comparison/`. Local logging only;
no W&B.

- `cosine_deployed_vs_shuffled_control_by_layer_{model_short}.png` — x: decoder layer index;
  y: cosine similarity, fixed range −1 to 1 with a horizontal line at 0. Title states model and
  that it compares the deployed `all`/nd200 direction against the shuffled-image control. Layers
  not mean-dominated (per precondition 2) marked visibly, with the marking explained in the
  caption.
- `magnitude_deployed_vs_shuffled_control_by_layer_{model_short}.png` — x: decoder layer index;
  y: L2 norm; two labelled curves, "deployed direction" and "shuffled-image control direction".
- `geometric_comparison_{model_short}.csv` — columns: `layer`, `cosine`, `deployed_norm`,
  `shuffled_norm`, `deployed_pc1_norm`, `shuffled_pc1_norm`.
- `geometric_comparison_summary.md` — the two preconditions' outcomes per model, and the per-layer
  cosine table. Facts only; interpretation belongs in RESEARCH_LOG with Alex.

Plots must be legible without this plan: axis labels in plain English, no internal slugs in axis
labels or titles.

## What the result reveals

Pre-registered reading, to be applied per model:

- **Cosine high across layers** (near 1), magnitudes comparable: the deployed direction is
  reproducible from caption contrast with the images scrambled. The steering vector is
  substantially a caption-text-contrast direction that happens to be extracted in a visual
  context. This would materially reframe the method.
- **Cosine low**, especially across Qwen2.5's behaviorally active late windows (roughly layers
  15–27, where the windowed sweep showed the yes-push): image-caption binding contributes to what
  the direction encodes.
- **Layer-dependent profile** (e.g. high early, low late): itself informative, and directly
  relevant to which windows the behavioral arm should target.

Two limits on what any of these license, to be carried into the write-up:

- Each demo's difference is (truthful caption | image *i*) − (hallucinated caption | image *i*) —
  the same image on both sides. A purely additive image contribution therefore cancels within a
  demo before averaging. This control tests whether the image-caption *interaction* matters to the
  aggregate direction, not whether the model uses the image at all.
- The direction is a mean over 200 demos. If per-demo image effects exist but roughly cancel
  across demos, the aggregate direction is already text-contrast-dominated and the cosine is high
  even though per-demo binding is real. So a high cosine licenses "the deployed aggregate
  direction is reconstructible from caption contrast alone" — a claim about the vector actually
  steered with, which is the right target — not "the model ignores the image."

## Open questions

- Loader name and PC1-component location: confirm `load_textual_v2_directions` and whether
  per-layer PC1 norms live in `components.npz` or `metadata.json` before writing code. Both were
  read at an earlier date and have not been re-verified since the MCP bridge went down.
- Threshold for "PC1 is a large fraction" in precondition 2 is left to the implementer to report
  rather than binarize: emit the fraction per layer and let the plot annotate the top of the
  distribution. No fixed cutoff is imposed here.
