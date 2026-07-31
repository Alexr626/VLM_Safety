# POPE-30 MLP 2×2 condensed JSON summaries — 2026-07-21

Two further-condensed result files matching the joint 2×2 plot axes (LLaVA/Qwen × additive/rotation @ mlp):

1. `diagnostic_experiments/perception_diag/windowed_steering_summary/pope30_mlp_2x2_mean_p_yes_raw_summary.json`  
   — mean `score_p_yes_raw` + bootstrap 95% CI + Δ vs baseline (plot data)

2. `diagnostic_experiments/perception_diag/windowed_steering_summary/pope30_mlp_2x2_accuracy_and_flips_summary.json`  
   — accuracy among parseable + yes→no flip counts on the same axes

Builder: `build_pope30_mlp_2x2_summary.py`. Excludes `rotation_layer`. Much smaller than the full per-item `windowed_steering_consolidated_results.json`.
