# AMBER plot qualitative HTML galleries

Created six response galleries (N=50 each, independent seeds) via
`helper_scripts/build_amber_plot_qualitative_html.py`.

## Outputs

### LLaVA (`llava_amber_results/qualitative/`)

| HTML | Source plot | Configs | Sampling |
|---|---|---|---|
| `gold_label_accuracy_by_layer_window_nd500_beta0.9.html` | gold-label by window nd500 β0.9 | baseline + all / 5–14 / 20–29 | stratified gold-yes/no (~19/31) |
| `accuracy_vs_beta_by_steering_vector_sample_size_layers_all.html` | layer_windows/all vs β × nd | baseline + 4 nd × 3 β | uniform |
| `accuracy_by_layer_window_and_beta_nd500.html` | nd500 window × β | baseline + 3 windows × 3 β | uniform |

### Qwen (`qwen_amber_results/qualitative/`)

| HTML | Source plot | Configs | Sampling |
|---|---|---|---|
| `accuracy_vs_beta_by_steering_vector_sample_size_layers_early.html` | early (5–14) vs β × nd | baseline + 4 nd × 3 β | uniform |
| `accuracy_vs_beta_by_steering_vector_sample_size_layers_late.html` | late (15–24) vs β × nd | baseline + 4 nd × 3 β | uniform |
| `accuracy_by_layer_window_and_beta_nd500.html` | nd500 window × β | baseline + 3 windows × 3 β | uniform |

Layout: image left; question + gold; baseline block; steered responses in a labeled table (rows × cols) or flat list. Parser matches eval scoring.
