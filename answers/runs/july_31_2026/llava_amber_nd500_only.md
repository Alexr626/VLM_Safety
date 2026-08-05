# LLaVA AMBER — nd = 500 only

Generated: 2026-07-31T09:10:51
Model: `llava-1.5-7b-hf` · Benchmark: AMBER discriminative · Run date: `2026-07-30`
Filter: **direction sample size nd = 500 (all layer sets × all betas)**
Steered cells in this view: **9**

Source cells: `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/*/metric_summary.json`

## Baseline (`no_intervention`) — shared reference

| metric | value | counts |
|---|---|---|
| accuracy | 76.7% | 345 / 450 |
| gold-no accuracy | 80.1% | n_gold_no=276 |
| gold-yes accuracy | 71.3% | n_gold_yes=174 |
| yes_ratio | 39.8% | |
| n_unparsed | 0 | |

Δ pp below = steered accuracy − this baseline accuracy.

## Steered results with nd = 500

Rows = layer set; columns = beta.

### Accuracy (Δ pp vs baseline)

| layer_set \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| all layers (0–31) | 77.1% (+0.4 pp) | 77.1% (+0.4 pp) | 74.2% (-2.4 pp) |
| early–middle window (layers 5–14) | 76.9% (+0.2 pp) | 77.6% (+0.9 pp) | 77.6% (+0.9 pp) |
| late window (layers 20–29) | 76.7% (+0.0 pp) | 76.4% (-0.2 pp) | 76.7% (+0.0 pp) |

### Gold-label split (gold-no / gold-yes accuracy)

| layer_set \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| all layers (0–31) | 76.4% / 78.2% | 70.3% / 87.9% | 64.1% / 90.2% |
| early–middle window (layers 5–14) | 77.5% / 75.9% | 75.0% / 81.6% | 71.7% / 86.8% |
| late window (layers 20–29) | 79.7% / 71.8% | 79.0% / 72.4% | 78.3% / 74.1% |

### yes_ratio (over all items)

| layer_set \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| all layers (0–31) | 44.7% | 52.2% | 56.9% |
| early–middle window (layers 5–14) | 43.1% | 46.9% | 50.9% |
| late window (layers 20–29) | 40.2% | 40.9% | 42.0% |

### Per-cell detail

| layer_set | nd | beta | accuracy | Δ pp | gold-no | gold-yes | yes_ratio | n_correct/n_total | n_unparsed |
|---|---|---|---|---|---|---|---|---|---|
| 20-29 | 500 | 0.2 | 76.7% | +0.0 | 79.7% | 71.8% | 40.2% | 345/450 | 0 |
| 20-29 | 500 | 0.5 | 76.4% | -0.2 | 79.0% | 72.4% | 40.9% | 344/450 | 0 |
| 20-29 | 500 | 0.9 | 76.7% | +0.0 | 78.3% | 74.1% | 42.0% | 345/450 | 0 |
| 5-14 | 500 | 0.2 | 76.9% | +0.2 | 77.5% | 75.9% | 43.1% | 346/450 | 0 |
| 5-14 | 500 | 0.5 | 77.6% | +0.9 | 75.0% | 81.6% | 46.9% | 349/450 | 0 |
| 5-14 | 500 | 0.9 | 77.6% | +0.9 | 71.7% | 86.8% | 50.9% | 349/450 | 0 |
| all | 500 | 0.2 | 77.1% | +0.4 | 76.4% | 78.2% | 44.7% | 347/450 | 0 |
| all | 500 | 0.5 | 77.1% | +0.4 | 70.3% | 87.9% | 52.2% | 347/450 | 0 |
| all | 500 | 0.9 | 74.2% | -2.4 | 64.1% | 90.2% | 56.9% | 334/450 | 0 |

---
Numbers from each cell `metric_summary.json`. No interpretation.
