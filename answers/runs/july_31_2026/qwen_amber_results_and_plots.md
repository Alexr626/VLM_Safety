# Qwen AMBER results markdown + gold-label plots

Date: 2026-07-31

Mirrored the LLaVA AMBER analysis layout for Qwen under:

`evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/`

## Structure

```
qwen_amber_consolidated_results.json
qwen_amber_results/
  qwen_amber_consolidated_results.md
  qwen_amber_nd500_only.md
  qwen_amber_layers_all_only.md
  qwen_amber_nd500_layers_all.md
  plots/
    gold_label_accuracy_by_layer_window_nd500_beta0.9.png
    gold_label_accuracy_vs_beta_nd500_layers_all.png
    gold_label_accuracy_vs_beta_nd500_layers_5_14.png
    gold_label_accuracy_vs_beta_nd500_layers_15_24.png
```

Source: `evaluation/results/2026-07-30/qwen2.5-vl-7b-instruct/amber/*/metric_summary.json` (37/37 cells).

Layer labels: all (0–27), early–middle (5–14), late (15–24).

Baseline (for cross-check): accuracy 76.0% (342/450), gold-no 96.4%, gold-yes 43.7%, n_unparsed=21.

No interpretation.
