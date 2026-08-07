# POPE experimental configuration: 2026-06-19 vs 2026-07-30

Date: 2026-08-06  
Question: What differed between the LLaVA POPE runs that improved under textual additive steering on 2026-06-19 and the 2026-07-30 meandiff grid that did not move POPE similarly?

This file is a **configuration inventory** (facts from plans, logged answers, `RESEARCH_LOG.md`, `IMPLEMENTATION.md`, and on-disk `metric_summary.json` / direction `metadata.json`). It does not assign a cause.

---

## Sources consulted

| Kind | Path |
|------|------|
| 06-19 run record | `RESEARCH_LOG.md` § 2026-06-19 |
| 06-19 tables | `evaluation/results/2026-06-19/meeting_summary_2026-06-19.md` |
| 06-19 answers | `answers/concepts/jun_19_2026/vti_rotation_strength_motivation.md`, `…/pope_metrics_yes_ratio_and_steering_probes.md` |
| Phase-1 method plan (pre-dates the dated-plan layout) | `implementation_plans/old/vti_phase1_textual.md` |
| 07-30 plan | `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md` |
| 07-30 run record | `RESEARCH_LOG.md` § 2026-07-30 |
| Prior comparable-PC1 note | `answers/runs/july_31_2026/comparable_vti_pc1_plus_mean_amber_prior_runs.md` |
| Code / layout facts | `IMPLEMENTATION.md` (author demos, demos_850, meandiff, textual PCA) |
| On-disk cells | `evaluation/results/2026-06-19/…`, `evaluation/results/2026-07-30/…` |
| Direction caches | `experiment_artifacts/vti/llava-1.5-7b-hf/textual_directions_nd70_rank1_seed42.npz`; `…/textual_v2/demos850_ba05bd96_all_nd*_s42_meandiff_partition/` |

There is **no** `implementation_plans/6-19-26/…` file. The 06-19 POPE β-grid is the Phase-1 textual VTI reproduction driven by `evaluation/run_scripts/run_vti_pope_beta_grid.sh`, with method defined in `vti_phase1_textual.md` and the run logged in `RESEARCH_LOG.md`.

---

## What was held constant (LLaVA POPE, additive MLP)

For the cells that are closest apples-to-apples (`vti_textual_additive_mlp`, LLaVA-1.5-7B, POPE):

| Knob | Both runs |
|------|-----------|
| Model | `llava-hf/llava-1.5-7b-hf` |
| Intervention registry name | `vti_textual_additive_mlp` |
| Geometry | `variant=additive` |
| Hook site | `hook_site=mlp` |
| Strength param | `beta` (06-18 cells logged `alpha_text`; same coefficient after rename) |
| `eps_coeff` | 0.1 (unused by additive; present in config) |
| Decode | greedy (`do_sample=False`) |
| POPE N | 200 per split via `--limit 200` (pinned first-N of `combined.json`; same pin as `data/pope/pinned_eval_ids.json`) |
| Diff polarity | `value − h_value` (legacy PCA path: `pos−neg` with clean=value, hall=h_value; meandiff metadata: `value_minus_h_value`) |
| Token policy | last token of `forward_vl(image, question + ' ' + caption)` |
| Seed for demo sampling / partition | 42 |
| Hardware (06-19 / 07-30 LLaVA) | lambdab2 A6000, conda env |

**Baseline identity check.** LLaVA `no_intervention` POPE accuracies match exactly across the two run dates:

| Split | 2026-06-19 | 2026-07-30 |
|-------|------------|------------|
| random | 0.890 | 0.890 |
| popular | 0.865 | 0.865 |
| adversarial | 0.785 | 0.785 |
| split-average | 0.847 | 0.847 |

So the eval item set and unscored baseline behavior are the same; the divergence is in the steered cells.

---

## Most-similar cells (comparison frame)

Restrict to the intersection of the two grids, not the full day’s work:

| Matched knob | Value |
|--------------|-------|
| Model | LLaVA-1.5-7B |
| Intervention | `vti_textual_additive_mlp` only |
| Layers | all decoder layers (`06-19` default; `07-30` `layers_all`) |
| β | `{0.2, 0.5, 0.9}` (present on both; `06-19` also ran other βs — ignored here) |
| Benchmark | POPE 200/split, same pin / identical baselines |

Dropped as non-differences under this frame: extra βs on `06-19`, extra interventions/sites on `06-19`, layer windows other than `all` on `07-30`, extra models, CHAIR/AMBER/rotation-strength arms.

---

## Irreducible differences (most-similar cells only)

| Knob | **2026-06-19 matched cells** | **2026-07-30 matched cells** |
|------|------------------------------|------------------------------|
| Demo file | Author `data/vti/demos.jsonl` (100 rows) | `data/vti/demos_850.jsonl` (hash `ba05bd960cad0c18`) |
| Caption recipe | Free-form `value` / `h_value` (GPT co-object injection; ~605-char `value`) | demos_v2.1 **4-sentence** pairs; dimension `all` = existence+attribute+counting+relation stacked (~253-char `value`) |
| Demo count / selection | `num_demos=70`, `random.sample` seed 42 from the 100 | `nd ∈ {50,100,200,500}`, `disjoint_partition` blocks (no `nd=70` cell; blocks are non-overlapping) |
| Direction math | Global flatten PCA → reshape(**PC1 + mean**); cache `textual_directions_nd70_rank1_seed42.npz` | **Raw mean-difference** only (`steer_reconstruction=raw_mean_difference`); slug `demos850_ba05bd96_all_nd{N}_s42_meandiff_partition` |

That is the full differing set under the matched frame. Demo file and caption recipe are two faces of one demo-pipeline change; demo count/selection and direction math are separate knobs.

Legacy PCA reconstruction (for reference):

```97:114:evaluation/interventions/vti/directions.py
    Fits PCA (rank=1) on flattened ``clean - hallucinated`` last-token states
    across demos, then reshapes to ``(num_layers + 1, hidden_dim)``.
    ...
    direction = (
        pca.components_.sum(dim=1, keepdim=True) + pca.mean_
    ).mean(0).view(...)
```

Split-averaged LLaVA accuracy on the overlapping βs (`layers_all` for `07-30`):

| β | 06-19 (nd70, PC1+mean) | 07-30 nd50 | nd100 | nd200 | nd500 | baseline |
|---|------------------------|------------|-------|-------|-------|----------|
| 0.2 | 0.837 | 0.840 | 0.843 | 0.842 | 0.847 | 0.847 |
| 0.5 | 0.855 | 0.843 | 0.828 | 0.828 | 0.833 | 0.847 |
| 0.9 | 0.880 | 0.828 | 0.828 | 0.823 | 0.832 | 0.847 |

---

## Logged `intervention_config` snapshots (LLaVA, adversarial, β=0.9)

**2026-06-19** (`…/pope_adversarial/vti_textual_additive_mlp__b0.9/metric_summary.json`):

```json
{
  "variant": "additive",
  "hook_site": "mlp",
  "beta": 0.9,
  "num_demos": 70,
  "rank": 1,
  "seed": 42,
  "eps_coeff": 0.1,
  "log_lambda_sim": false
}
```

Accuracy that cell: **0.84** (baseline 0.785). Split-averaged LLaVA `additive_mlp` at β=0.9: **88.0 / 87.8** vs baseline **84.7 / 85.3** (`meeting_summary` / `RESEARCH_LOG`).

**2026-07-30** (`…/pope_adversarial/vti_textual_additive_mlp__b0.9__dall__nd500__meandiff__layers_all/metric_summary.json`):

```json
{
  "variant": "additive",
  "hook_site": "mlp",
  "beta": 0.9,
  "num_demos": 500,
  "rank": 1,
  "seed": 42,
  "eps_coeff": 0.1,
  "log_lambda_sim": false,
  "directions_dir": "…/demos850_ba05bd96_all_nd500_s42_meandiff_partition",
  "direction_slug": "demos850_ba05bd96_all_nd500_s42_meandiff_partition",
  "steer_reconstruction": "raw_mean_difference",
  "demos_content_hash_sha256_16": "ba05bd960cad0c18",
  "dimension": "all",
  "selection_policy": "disjoint_partition",
  "diff_polarity": "value_minus_h_value",
  "token_policy": "last_token_of_full_sequence: …",
  "layer_indices": null,
  "layer_set_label": "all"
}
```

Accuracy that cell: **0.765** (baseline 0.785). Split-averaged for the same arm: **0.832** vs baseline **0.847**.

Other 07-30 `layers_all` β=0.9 nd arms (split-avg accuracy): nd50 0.828, nd100 0.828, nd200 0.823, nd500 0.832. Windowed arms (`5_14`, `20_29`) stay closer to baseline (~0.84).

---

## Related prior note (PC1+mean twin of 07-30)

`answers/runs/july_31_2026/comparable_vti_pc1_plus_mean_amber_prior_runs.md` records that demos850 **PCA** directions (`*_r2_partition`, `live_pc1_plus_mean`) already existed on disk for the same partition blocks, but were **not** run through the 07-30 POPE/AMBER/CHAIR eval grid. The closest author-demo PC1+mean POPE cells remain the 06-19 (and 06-18) trees. Separating “demo pool” from “PCA vs meandiff” therefore still requires a new cell set (or the 08-05 AMBER expanded grid, which does cross meandiff vs `pc1_plus_mean` on AMBER-1500, not POPE).

---

## Pointers for isolating factors (inventory only)

If the goal is to attribute the POPE gap to one knob, the on-disk pieces that already exist vs still missing are:

| Comparison | Status on disk |
|------------|----------------|
| Same POPE pin, same baseline | Yes (06-19 and 07-30) |
| Author demos + PC1+mean + additive_mlp + all layers + β∈{0.2,0.5,0.9} | Yes inside 06-19 tree |
| demos850 + meandiff + additive_mlp + all layers + β∈{0.2,0.5,0.9} | Yes inside 07-30 tree |
| demos850 + **PC1+mean** (`*_r2_partition`) + POPE eval | Directions exist; **POPE eval cells not present** under 07-30 |
| Author demos + **meandiff** + POPE | Not present as a logged cell |
| demos_v2/850 captions + PC1+mean + POPE at nd≈70 | Not present as a matched cell |

No causal ranking is implied by that table.
