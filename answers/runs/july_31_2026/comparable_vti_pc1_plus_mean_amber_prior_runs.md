# Comparable prior AMBER runs with VTI PC1+mean (vs 2026-07-30 meandiff grid)

## Recent experiment (reference)

`evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/run_manifest_llava-1.5-7b-hf.json` (Qwen twin exists):

| Knob | Value |
|------|--------|
| Benchmark | AMBER discriminative, `data/amber/pinned_amber_disc_450.json` (n=450) |
| Intervention | `vti_textual_additive_mlp` |
| Dimension | `all` |
| Direction | demos_850 disjoint partitions, `steer_reconstruction=raw_mean_difference` |
| Slugs | `demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition` |
| Betas | 0.2, 0.5, 0.9 |
| Layer sets | `all`, `5-14`, late (`20-29` LLaVA) |
| Diff polarity / token policy | `value_minus_h_value`; last token of image+caption forward |

Cells: `evaluation/results/2026-07-30/{llava-1.5-7b-hf,qwen2.5-vl-7b-instruct}/amber/*__meandiff__layers_*`

## Verdict

**No.** There is no prior AMBER evaluation that matches that grid with the only change being VTI-style `PC1 + mean` instead of raw mean-difference.

Matching demos850 PCA directions (`live_pc1_plus_mean`, same partition / nd / dimension) **exist on disk** for LLaVA and Qwen (`…/textual_v2/demos850_ba05bd96_all_nd*_s42_r2_partition`), but they were **not** wired into any AMBER eval under `evaluation/results/` (only a hash snapshot file mentions `r2_partition`).

## Closest prior PC1+mean AMBER work

### 1. 2026-07-13 demos_v2 qualitative grid (closest eval cells)

Path: `evaluation/results/2026-07-13/{llava-1.5-7b-hf,qwen2.5-vl-7b-instruct}/amber/`

- `steer_reconstruction`: **`live_pc1_plus_mean`** (global flatten PCA; steering = reshape(PC1 + mean); caption diffs `value − h_value`)
- Same: `vti_textual_additive_mlp`, dimension `all`, betas `{0.2,0.5,0.9}`, nd `{50,100,200,500}`, seed 42
- Differs: AMBER **n=25** (`data/vti/qual_subset_chair5_amber25.json`), not amber-450; demos **`demos_v2.jsonl`** + **`shuffled_prefix`**, not demos_850 + `disjoint_partition`; **no** `--layer_set` (all layers only); also ran other interventions/dimensions

Example cell config logged: `…/vti_textual_additive_mlp__b0.5__dall__nd500/metric_summary.json` → `steer_reconstruction=live_pc1_plus_mean`, `n_total=25`.

### 2. LLaVA AMBER-100 windowed steering dumps (layer windows overlap)

Path: `data/amber/dumps/llava-1.5-7b-hf/amber100_windowed_steering/`

- Direction slug: `demosv2_9a44f4af_all_nd200_s42_r2_prefix` → metadata **`live_pc1_plus_mean`**
- Overlapping cells: `additive_mlp_{0.2,0.5,0.9}_layers_{5_14,20_29,all}` (plus extra windows)
- Differs: **n=100** (`augmented_amber100.jsonl` leading-clause conditions), **nd=200 only**, demos_v2 prefix not demos850 partition; dump/augment pipeline, not the 07-30 `run_eval` amber-450 grid
- No matching Qwen `amber100_windowed_steering` tree under `data/amber/dumps/`

### 3. 2026-06-22 Exp1 AMBER-450 (author-demo VTI)

Path: `evaluation/results/2026-06-22/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b{0.1…0.9}/`

- Same subset size (**n=450**), additive MLP, betas include 0.2/0.5/0.9
- Direction: legacy author demos `textual_directions_nd70_rank1_seed42.npz` — codebase documents steering as reshape(**PC1 + mean**)
- Differs: **nd=70 only**, author `demos.jsonl` (not demos_v2/850), **no** layer windows; `intervention_config` does not record `steer_reconstruction`

## Not comparable as “same config, different vector”

| Artifact | Why not a drop-in PC1+mean twin of 07-30 |
|----------|------------------------------------------|
| demos850 `*_r2_partition` directions | Extracted; **no AMBER eval cells** |
| 07-13 demos_v2 grid | PC1+mean yes; amber-25; no layer sets; different demo pool/selection |
| amber100 windowed dumps | PC1+mean + some layer/β overlap; amber-100 + prompts + nd200 only |
| 06-22 Exp1 | amber-450 + PC1+mean (author); nd70; no layer grid |
