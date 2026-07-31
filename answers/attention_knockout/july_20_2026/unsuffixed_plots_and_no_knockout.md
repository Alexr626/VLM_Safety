# Unsuffixed plots vs no-knockout baselines (2026-07-20)

## Were the unsuffixed Qwen plots duplicates of all-downstream?

Yes. `analyze_knockout.py` wrote each cell’s curve to
`{cell}__{metric}.png` **and** overwrote the bare `{metric}.png` with the
**last cell analyzed**. Analysis order is last-token then all-downstream, so
the bare files matched all-downstream exactly. Those bare KO copies are no
longer written.

Use cell-prefixed names for knockout curves, e.g.
`qwen25_block_last_token_reading__flip_recovery_rate_by_layer_window.png`.

## No-knockout plots (new)

Under `{model}/results/plots/`:

- `flip_recovery_rate_no_knockout.png` — recovery without KO (~0 on the flip set by definition)
- `mean_yes_probability_recovery_no_knockout.png` — recovery without KO (identically 0)
- `yes_probability_by_prompt_condition_no_knockout.png` — mean `p_yes` for neutral / misleading / fillers on the flip set (the useful no-KO readout)

Layer-window x-axis on the flat recovery plots is only for visual alignment with the KO curves; no layers are blocked.
