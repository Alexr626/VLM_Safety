# POPE-30 plot cleanup + joint 2×2 — 2026-07-21

## Layout

- `plots/llava-1.5-7b-hf/` — LLaVA-only PNGs (9 files, all with baseline bars)
- `plots/qwen2.5-vl-7b-instruct/` — Qwen-only PNGs (same)
- Removed dashed-line-only set and nested `with_baseline_bars/` folders

## Joint figure

`plots/pope30_mean_p_yes_raw_by_layer_window_llava_vs_qwen_additive_vs_rotation_mlp.png`

- Rows: LLaVA-1.5 / Qwen2.5-VL  
- Columns: additive @ mlp / rotation @ mlp  
- Within each pane: neutral + leading-toward-no stacked; baseline bars included
