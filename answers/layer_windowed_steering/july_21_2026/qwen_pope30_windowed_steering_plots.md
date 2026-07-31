# Qwen POPE-30 windowed steering plots — 2026-07-21

Same plot set as LLaVA, for Qwen2.5-VL POPE-30 (54 cells).

**Dir:** `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/qwen2.5-vl-7b-instruct/`  
(+ `with_baseline_bars/` sibling)

**Command:** `python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py --model qwen2.5-vl-7b-instruct`

## Baseline denominator flag (plot 3)

Unlike LLaVA (27/30 under both conditions), Qwen baseline parsed-yes counts **differ**:

| condition | parsed yes | mean P(yes) raw |
|-----------|------------|-----------------|
| neutral | 24/30 | 0.78 |
| assertive_toward_no | 21/30 | 0.44 |

Three items (`pope_*_00000`) are baseline-yes under neutral but baseline-no under the leading clause. Flip-count bars are therefore not on the same denominator across the two rows.
