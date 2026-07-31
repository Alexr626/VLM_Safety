# LLaVA POPE-30 merged flips (gold yes vs no)

**Date:** 2026-07-22

Same layout as AMBER attribute/relation flips merged plots, for LLaVA POPE-30 only:

- Top: gold=yes (neutral + assertive toward no)
- Bottom: gold=no (neutral + assertive toward yes)
- Columns: additive @ mlp | rotation @ mlp

**File:**
`diagnostic_experiments/perception_diag/windowed_steering_summary/plots/merged_plots/pope-30-runs/pope30_flips_from_baseline_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`

```
python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py --joint_pope_llava
```
