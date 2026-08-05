# Cross-model AMBER plot mirrors

Date: 2026-07-31

Applied each model's plot families to both LLaVA and Qwen under matching subdirectory structures.

## Accuracy views (originally Qwen)

`{llava,qwen}_amber_results/plots/steering_vector_sample_size/{50,100,200,500}/accuracy_by_layer_window_and_beta.png`

`{llava,qwen}_amber_results/plots/layer_windows/{early,late,all}/accuracy_vs_beta_by_steering_vector_sample_size.png`

Plus stitched early|late over all:
`plots/layer_windows/accuracy_vs_beta_by_steering_vector_sample_size_early_late_all.png`

## Gold-label yes/no views (originally LLaVA)

`{llava,qwen}_amber_results/plots/accuracy_yes_vs_no_comparisons/`
- `gold_label_accuracy_by_layer_window_nd500_beta0.9.png`
- `gold_label_accuracy_vs_beta_nd500_layers_{all,5_14,late}.png`

Late-window filenames: LLaVA `layers_20_29`, Qwen `layers_15_24`.
