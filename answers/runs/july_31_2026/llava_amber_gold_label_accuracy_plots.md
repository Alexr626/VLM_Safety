# LLaVA AMBER gold-label accuracy plots (nd=500)

Created four plots under:

`evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/llava_amber_results/plots/`

Source cells: `evaluation/results/2026-07-30/llava-1.5-7b-hf/amber/*/metric_summary.json`
(`neg_item_accuracy` = gold-no; `pos_item_accuracy` = gold-yes; n_gold_no=276, n_gold_yes=174).

## Files

1. `gold_label_accuracy_by_layer_window_nd500_beta0.9.png` — gold-no and gold-yes accuracy for baseline, all layers (0–31), layers 5–14, and layers 20–29 at nd=500, beta=0.9.

2. `gold_label_accuracy_vs_beta_nd500_layers_all.png` — gold-no / gold-yes vs beta ∈ {0.2, 0.5, 0.9} for all layers, nd=500 (baseline dashed).

3. `gold_label_accuracy_vs_beta_nd500_layers_5_14.png` — same for early–middle window (layers 5–14).

4. `gold_label_accuracy_vs_beta_nd500_layers_20_29.png` — same for late window (layers 20–29).

## Values used at nd=500, beta=0.9 (for cross-check)

| condition | gold-no | gold-yes |
|---|---|---|
| baseline | 80.1% | 71.3% |
| all layers | 64.1% | 90.2% |
| layers 5–14 | 71.7% | 86.8% |
| layers 20–29 | 78.3% | 74.1% |

No interpretation.
