# LLaVA AMBER discriminative — consolidated results (sectional views)

Generated: 2026-07-31T08:59:38
Model: `llava-1.5-7b-hf` · Run date: `2026-07-30` · Subset: pinned AMBER-450
Cells: **37** / 37 · Intervention: `vti_textual_additive_mlp` / baseline `no_intervention`

Δ / pp = steered accuracy minus baseline accuracy, in percentage points.
Baseline accuracy used for Δ: **76.7%** (345/450, n_unparsed=0).

## Where the files are

```
evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/{iv_dir}/
  metric_summary.json
  responses.json
```

This file: `evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/llava_amber_consolidated_results.md`

## 1. Baseline (`no_intervention`)

| metric | value | counts |
|---|---|---|
| accuracy | 76.7% | 345 / 450 |
| precision | 69.3% | |
| recall | 71.3% | |
| F1 | 70.3% | |
| yes_ratio (over all items) | 39.8% | |
| accuracy on gold-no | 80.1% | n_gold_no=276 |
| accuracy on gold-yes | 71.3% | n_gold_yes=174 |
| n_unparsed | 0 | |

### Baseline by question type

| qtype | accuracy | gold-no acc | gold-yes acc | n_total | n_unparsed |
|---|---|---|---|---|---|
| attribute | 76.0% | 67.6% | 82.9% | 150 | 0 |
| existence | 82.7% | 82.7% | 0.0% | 150 | 0 |
| relation | 71.3% | 87.9% | 60.9% | 150 | 0 |

## 2. View by layer set

Each subsection fixes the steered layers; rows are direction sample size (`nd`), columns are `beta`.
Cell = accuracy (Δ pp vs baseline).

### 2.1 all layers (0–31)

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | 75.1% (-1.6 pp) | 75.6% (-1.1 pp) | 71.3% (-5.3 pp) |
| 100 | 76.9% (+0.2 pp) | 76.9% (+0.2 pp) | 76.0% (-0.7 pp) |
| 200 | 77.1% (+0.4 pp) | 77.6% (+0.9 pp) | 73.8% (-2.9 pp) |
| 500 | 77.1% (+0.4 pp) | 77.1% (+0.4 pp) | 74.2% (-2.4 pp) |

Gold-label split (accuracy on gold-no / gold-yes):

| nd \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| 50 | 75.7% / 74.1% | 69.6% / 85.1% | 64.1% / 82.8% |
| 100 | 76.1% / 78.2% | 71.0% / 86.2% | 64.9% / 93.7% |
| 200 | 76.1% / 78.7% | 70.3% / 89.1% | 60.5% / 94.8% |
| 500 | 76.4% / 78.2% | 70.3% / 87.9% | 64.1% / 90.2% |

yes_ratio (over all items):

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | 43.6% | 51.6% | 54.0% |
| 100 | 44.9% | 51.1% | 57.8% |
| 200 | 45.1% | 52.7% | 60.9% |
| 500 | 44.7% | 52.2% | 56.9% |

### 2.2 early–middle window (layers 5–14)

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | 75.8% (-0.9 pp) | 76.4% (-0.2 pp) | 76.2% (-0.4 pp) |
| 100 | 76.9% (+0.2 pp) | 77.6% (+0.9 pp) | 77.3% (+0.7 pp) |
| 200 | 77.1% (+0.4 pp) | 78.0% (+1.3 pp) | 78.2% (+1.6 pp) |
| 500 | 76.9% (+0.2 pp) | 77.6% (+0.9 pp) | 77.6% (+0.9 pp) |

Gold-label split (accuracy on gold-no / gold-yes):

| nd \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| 50 | 77.5% / 73.0% | 75.4% / 78.2% | 73.6% / 80.5% |
| 100 | 77.5% / 75.9% | 75.4% / 81.0% | 71.0% / 87.4% |
| 200 | 77.5% / 76.4% | 75.7% / 81.6% | 73.2% / 86.2% |
| 500 | 77.5% / 75.9% | 75.0% / 81.6% | 71.7% / 86.8% |

yes_ratio (over all items):

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | 42.0% | 45.3% | 47.3% |
| 100 | 43.1% | 46.4% | 51.6% |
| 200 | 43.3% | 46.4% | 49.8% |
| 500 | 43.1% | 46.9% | 50.9% |

### 2.3 late window (layers 20–29)

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | 76.9% (+0.2 pp) | 76.4% (-0.2 pp) | 76.4% (-0.2 pp) |
| 100 | 76.9% (+0.2 pp) | 76.7% (+0.0 pp) | 76.7% (+0.0 pp) |
| 200 | 76.4% (-0.2 pp) | 76.2% (-0.4 pp) | 76.4% (-0.2 pp) |
| 500 | 76.7% (+0.0 pp) | 76.4% (-0.2 pp) | 76.7% (+0.0 pp) |

Gold-label split (accuracy on gold-no / gold-yes):

| nd \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| 50 | 80.1% / 71.8% | 79.0% / 72.4% | 77.9% / 74.1% |
| 100 | 80.1% / 71.8% | 79.7% / 71.8% | 79.7% / 71.8% |
| 200 | 79.3% / 71.8% | 79.0% / 71.8% | 77.9% / 74.1% |
| 500 | 79.7% / 71.8% | 79.0% / 72.4% | 78.3% / 74.1% |

yes_ratio (over all items):

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | 40.0% | 40.9% | 42.2% |
| 100 | 40.0% | 40.2% | 40.2% |
| 200 | 40.4% | 40.7% | 42.2% |
| 500 | 40.2% | 40.9% | 42.0% |

## 3. View by direction sample size (`nd`)

Each subsection fixes how many demos built the mean-difference direction; rows are layer set, columns are `beta`.
Cell = accuracy (Δ pp vs baseline).

### 3.1 nd = 50

| layer_set \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| all layers (0–31) | 75.1% (-1.6 pp) | 75.6% (-1.1 pp) | 71.3% (-5.3 pp) |
| early–middle window (layers 5–14) | 75.8% (-0.9 pp) | 76.4% (-0.2 pp) | 76.2% (-0.4 pp) |
| late window (layers 20–29) | 76.9% (+0.2 pp) | 76.4% (-0.2 pp) | 76.4% (-0.2 pp) |

Gold-label split (gold-no / gold-yes):

| layer_set \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| all | 75.7% / 74.1% | 69.6% / 85.1% | 64.1% / 82.8% |
| 5-14 | 77.5% / 73.0% | 75.4% / 78.2% | 73.6% / 80.5% |
| 20-29 | 80.1% / 71.8% | 79.0% / 72.4% | 77.9% / 74.1% |

### 3.2 nd = 100

| layer_set \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| all layers (0–31) | 76.9% (+0.2 pp) | 76.9% (+0.2 pp) | 76.0% (-0.7 pp) |
| early–middle window (layers 5–14) | 76.9% (+0.2 pp) | 77.6% (+0.9 pp) | 77.3% (+0.7 pp) |
| late window (layers 20–29) | 76.9% (+0.2 pp) | 76.7% (+0.0 pp) | 76.7% (+0.0 pp) |

Gold-label split (gold-no / gold-yes):

| layer_set \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| all | 76.1% / 78.2% | 71.0% / 86.2% | 64.9% / 93.7% |
| 5-14 | 77.5% / 75.9% | 75.4% / 81.0% | 71.0% / 87.4% |
| 20-29 | 80.1% / 71.8% | 79.7% / 71.8% | 79.7% / 71.8% |

### 3.3 nd = 200

| layer_set \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| all layers (0–31) | 77.1% (+0.4 pp) | 77.6% (+0.9 pp) | 73.8% (-2.9 pp) |
| early–middle window (layers 5–14) | 77.1% (+0.4 pp) | 78.0% (+1.3 pp) | 78.2% (+1.6 pp) |
| late window (layers 20–29) | 76.4% (-0.2 pp) | 76.2% (-0.4 pp) | 76.4% (-0.2 pp) |

Gold-label split (gold-no / gold-yes):

| layer_set \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| all | 76.1% / 78.7% | 70.3% / 89.1% | 60.5% / 94.8% |
| 5-14 | 77.5% / 76.4% | 75.7% / 81.6% | 73.2% / 86.2% |
| 20-29 | 79.3% / 71.8% | 79.0% / 71.8% | 77.9% / 74.1% |

### 3.4 nd = 500

| layer_set \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| all layers (0–31) | 77.1% (+0.4 pp) | 77.1% (+0.4 pp) | 74.2% (-2.4 pp) |
| early–middle window (layers 5–14) | 76.9% (+0.2 pp) | 77.6% (+0.9 pp) | 77.6% (+0.9 pp) |
| late window (layers 20–29) | 76.7% (+0.0 pp) | 76.4% (-0.2 pp) | 76.7% (+0.0 pp) |

Gold-label split (gold-no / gold-yes):

| layer_set \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| all | 76.4% / 78.2% | 70.3% / 87.9% | 64.1% / 90.2% |
| 5-14 | 77.5% / 75.9% | 75.0% / 81.6% | 71.7% / 86.8% |
| 20-29 | 79.7% / 71.8% | 79.0% / 72.4% | 78.3% / 74.1% |

## 4. View by beta

Each subsection fixes steering strength; rows are layer set, columns are `nd`.
Cell = accuracy (Δ pp vs baseline).

### 4.1 beta = 0.2

| layer_set \ nd | 50 | 100 | 200 | 500 |
|---|---|---|---|---|
| all layers (0–31) | 75.1% (-1.6 pp) | 76.9% (+0.2 pp) | 77.1% (+0.4 pp) | 77.1% (+0.4 pp) |
| early–middle window (layers 5–14) | 75.8% (-0.9 pp) | 76.9% (+0.2 pp) | 77.1% (+0.4 pp) | 76.9% (+0.2 pp) |
| late window (layers 20–29) | 76.9% (+0.2 pp) | 76.9% (+0.2 pp) | 76.4% (-0.2 pp) | 76.7% (+0.0 pp) |

### 4.2 beta = 0.5

| layer_set \ nd | 50 | 100 | 200 | 500 |
|---|---|---|---|---|
| all layers (0–31) | 75.6% (-1.1 pp) | 76.9% (+0.2 pp) | 77.6% (+0.9 pp) | 77.1% (+0.4 pp) |
| early–middle window (layers 5–14) | 76.4% (-0.2 pp) | 77.6% (+0.9 pp) | 78.0% (+1.3 pp) | 77.6% (+0.9 pp) |
| late window (layers 20–29) | 76.4% (-0.2 pp) | 76.7% (+0.0 pp) | 76.2% (-0.4 pp) | 76.4% (-0.2 pp) |

### 4.3 beta = 0.9

| layer_set \ nd | 50 | 100 | 200 | 500 |
|---|---|---|---|---|
| all layers (0–31) | 71.3% (-5.3 pp) | 76.0% (-0.7 pp) | 73.8% (-2.9 pp) | 74.2% (-2.4 pp) |
| early–middle window (layers 5–14) | 76.2% (-0.4 pp) | 77.3% (+0.7 pp) | 78.2% (+1.6 pp) | 77.6% (+0.9 pp) |
| late window (layers 20–29) | 76.4% (-0.2 pp) | 76.7% (+0.0 pp) | 76.4% (-0.2 pp) | 76.7% (+0.0 pp) |

## 5. Δ accuracy only (pp vs baseline)

### all layers (0–31)

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | -1.6 | -1.1 | -5.3 |
| 100 | +0.2 | +0.2 | -0.7 |
| 200 | +0.4 | +0.9 | -2.9 |
| 500 | +0.4 | +0.4 | -2.4 |

### early–middle window (layers 5–14)

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | -0.9 | -0.2 | -0.4 |
| 100 | +0.2 | +0.9 | +0.7 |
| 200 | +0.4 | +1.3 | +1.6 |
| 500 | +0.2 | +0.9 | +0.9 |

### late window (layers 20–29)

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | +0.2 | -0.2 | -0.2 |
| 100 | +0.2 | +0.0 | +0.0 |
| 200 | -0.2 | -0.4 | -0.2 |
| 500 | +0.0 | -0.2 | +0.0 |

## 6. File index

| layer_set | nd | beta | metric_summary.json |
|---|---|---|---|
| baseline | — | — | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/no_intervention/metric_summary.json` |
| 20-29 | 50 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd50__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 50 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd50__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 50 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd50__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 100 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd100__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 100 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd100__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 100 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd100__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 200 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd200__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 200 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd200__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 200 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd200__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 500 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd500__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 500 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd500__meandiff__layers_20_29/metric_summary.json` |
| 20-29 | 500 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd500__meandiff__layers_20_29/metric_summary.json` |
| 5-14 | 50 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd50__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 50 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd50__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 50 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd50__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 100 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd100__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 100 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd100__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 100 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd100__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 200 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd200__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 200 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd200__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 200 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd200__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 500 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd500__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 500 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd500__meandiff__layers_5_14/metric_summary.json` |
| 5-14 | 500 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd500__meandiff__layers_5_14/metric_summary.json` |
| all | 50 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd50__meandiff__layers_all/metric_summary.json` |
| all | 50 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd50__meandiff__layers_all/metric_summary.json` |
| all | 50 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd50__meandiff__layers_all/metric_summary.json` |
| all | 100 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd100__meandiff__layers_all/metric_summary.json` |
| all | 100 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd100__meandiff__layers_all/metric_summary.json` |
| all | 100 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd100__meandiff__layers_all/metric_summary.json` |
| all | 200 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd200__meandiff__layers_all/metric_summary.json` |
| all | 200 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd200__meandiff__layers_all/metric_summary.json` |
| all | 200 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd200__meandiff__layers_all/metric_summary.json` |
| all | 500 | 0.2 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.2__dall__nd500__meandiff__layers_all/metric_summary.json` |
| all | 500 | 0.5 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.5__dall__nd500__meandiff__layers_all/metric_summary.json` |
| all | 500 | 0.9 | `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/vti_textual_additive_mlp__b0.9__dall__nd500__meandiff__layers_all/metric_summary.json` |

---
Numbers copied from each cell `metric_summary.json`. No interpretation.
