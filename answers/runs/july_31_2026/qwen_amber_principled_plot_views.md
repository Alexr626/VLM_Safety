# Qwen AMBER principled plot views (accuracy)

Date: 2026-07-31

Deleted prior gold-label plots under `qwen_amber_results/plots/`.

## View: fix steering-vector sample size (`nd`)

`qwen_amber_results/steering_vector_sample_size/{50,100,200,500}/accuracy_by_layer_window_and_beta.png`

Grouped bars: x = layer window (early–middle 5–14, late 15–24, all 0–27); within each group four bars = baseline + beta 0.2 / 0.5 / 0.9; y = overall accuracy (%).

## View: fix layer window

`qwen_amber_results/layer_windows/{early,late,all}/accuracy_vs_beta_by_steering_vector_sample_size.png`

Line plot: x = beta; one line per nd ∈ {50,100,200,500}; dashed baseline; y = overall accuracy (%).

(Inside a fixed layer-window folder, lines are per `nd`, not per layer window.)

Metric: overall accuracy from each cell `metric_summary.json`. No interpretation.
