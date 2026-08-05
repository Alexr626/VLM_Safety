# LLaVA AMBER — all layers only

Generated: 2026-07-31T09:10:51
Model: `llava-1.5-7b-hf` · Benchmark: AMBER discriminative · Run date: `2026-07-30`
Filter: **layer_set = all (layers 0–31); all nd × all betas**
Steered cells in this view: **12**

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

## Steered results with layer_set = all

Rows = nd; columns = beta.

### Accuracy (Δ pp vs baseline)

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | 75.1% (-1.6 pp) | 75.6% (-1.1 pp) | 71.3% (-5.3 pp) |
| 100 | 76.9% (+0.2 pp) | 76.9% (+0.2 pp) | 76.0% (-0.7 pp) |
| 200 | 77.1% (+0.4 pp) | 77.6% (+0.9 pp) | 73.8% (-2.9 pp) |
| 500 | 77.1% (+0.4 pp) | 77.1% (+0.4 pp) | 74.2% (-2.4 pp) |

### Gold-label split (gold-no / gold-yes accuracy)

| nd \ beta | 0.2 (no / yes) | 0.5 (no / yes) | 0.9 (no / yes) |
|---|---|---|---|
| 50 | 75.7% / 74.1% | 69.6% / 85.1% | 64.1% / 82.8% |
| 100 | 76.1% / 78.2% | 71.0% / 86.2% | 64.9% / 93.7% |
| 200 | 76.1% / 78.7% | 70.3% / 89.1% | 60.5% / 94.8% |
| 500 | 76.4% / 78.2% | 70.3% / 87.9% | 64.1% / 90.2% |

### yes_ratio (over all items)

| nd \ beta | 0.2 | 0.5 | 0.9 |
|---|---|---|---|
| 50 | 43.6% | 51.6% | 54.0% |
| 100 | 44.9% | 51.1% | 57.8% |
| 200 | 45.1% | 52.7% | 60.9% |
| 500 | 44.7% | 52.2% | 56.9% |

### Per-cell detail

| layer_set | nd | beta | accuracy | Δ pp | gold-no | gold-yes | yes_ratio | n_correct/n_total | n_unparsed |
|---|---|---|---|---|---|---|---|---|---|
| all | 50 | 0.2 | 75.1% | -1.6 | 75.7% | 74.1% | 43.6% | 338/450 | 0 |
| all | 50 | 0.5 | 75.6% | -1.1 | 69.6% | 85.1% | 51.6% | 340/450 | 0 |
| all | 50 | 0.9 | 71.3% | -5.3 | 64.1% | 82.8% | 54.0% | 321/450 | 0 |
| all | 100 | 0.2 | 76.9% | +0.2 | 76.1% | 78.2% | 44.9% | 346/450 | 0 |
| all | 100 | 0.5 | 76.9% | +0.2 | 71.0% | 86.2% | 51.1% | 346/450 | 0 |
| all | 100 | 0.9 | 76.0% | -0.7 | 64.9% | 93.7% | 57.8% | 342/450 | 0 |
| all | 200 | 0.2 | 77.1% | +0.4 | 76.1% | 78.7% | 45.1% | 347/450 | 0 |
| all | 200 | 0.5 | 77.6% | +0.9 | 70.3% | 89.1% | 52.7% | 349/450 | 0 |
| all | 200 | 0.9 | 73.8% | -2.9 | 60.5% | 94.8% | 60.9% | 332/450 | 0 |
| all | 500 | 0.2 | 77.1% | +0.4 | 76.4% | 78.2% | 44.7% | 347/450 | 0 |
| all | 500 | 0.5 | 77.1% | +0.4 | 70.3% | 87.9% | 52.2% | 347/450 | 0 |
| all | 500 | 0.9 | 74.2% | -2.4 | 64.1% | 90.2% | 56.9% | 334/450 | 0 |

---
Numbers from each cell `metric_summary.json`. No interpretation.
