# LLaVA AMBER — nd = 500 ∩ all layers

Generated: 2026-07-31T09:10:51
Model: `llava-1.5-7b-hf` · Benchmark: AMBER discriminative · Run date: `2026-07-30`
Filter: **nd = 500 and layer_set = all (beta sweep only)**
Steered cells in this view: **3**

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

## Steered results with nd = 500 and layer_set = all

Only beta varies in this view.

| beta | accuracy | Δ pp | gold-no | gold-yes | yes_ratio | precision | recall | F1 | n_correct/n_total | n_unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 77.1% | +0.4 | 76.4% | 78.2% | 44.7% | 67.7% | 78.2% | 72.5% | 347/450 | 0 |
| 0.5 | 77.1% | +0.4 | 70.3% | 87.9% | 52.2% | 65.1% | 87.9% | 74.8% | 347/450 | 0 |
| 0.9 | 74.2% | -2.4 | 64.1% | 90.2% | 56.9% | 61.3% | 90.2% | 73.0% | 334/450 | 0 |

### By question type (existence / attribute / relation)

| beta | qtype | accuracy | gold-no | gold-yes | n_total | n_unparsed |
|---|---|---|---|---|---|---|
| 0.2 | attribute | 76.7% | 64.7% | 86.6% | 150 | 0 |
| 0.2 | existence | 81.3% | 81.3% | 0.0% | 150 | 0 |
| 0.2 | relation | 73.3% | 77.6% | 70.7% | 150 | 0 |
| 0.5 | attribute | 78.0% | 58.8% | 93.9% | 150 | 0 |
| 0.5 | existence | 77.3% | 77.3% | 0.0% | 150 | 0 |
| 0.5 | relation | 76.0% | 65.5% | 82.6% | 150 | 0 |
| 0.9 | attribute | 72.0% | 51.5% | 89.0% | 150 | 0 |
| 0.9 | existence | 78.7% | 78.7% | 0.0% | 150 | 0 |
| 0.9 | relation | 72.0% | 41.4% | 91.3% | 150 | 0 |

### Per-cell detail

| layer_set | nd | beta | accuracy | Δ pp | gold-no | gold-yes | yes_ratio | n_correct/n_total | n_unparsed |
|---|---|---|---|---|---|---|---|---|---|
| all | 500 | 0.2 | 77.1% | +0.4 | 76.4% | 78.2% | 44.7% | 347/450 | 0 |
| all | 500 | 0.5 | 77.1% | +0.4 | 70.3% | 87.9% | 52.2% | 347/450 | 0 |
| all | 500 | 0.9 | 74.2% | -2.4 | 64.1% | 90.2% | 56.9% | 334/450 | 0 |

---
Numbers from each cell `metric_summary.json`. No interpretation.
