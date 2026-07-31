# POPE-30 windowed steering plots (LLaVA) — 2026-07-21

Executed `implementation_plans/pope30_windowed_steering_plots_plan_2026-07-21.md`.

## Outputs

Script: `diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py`

Plots under `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/`:

1. `pope30_parsed_yes_rate_by_layer_window.png`
2. `pope30_mean_p_yes_raw_by_layer_window.png`
3. `pope30_flips_from_baseline_yes_by_layer_window.png`

## Open question 2 (baseline parse coverage)

Does **not** differ materially between conditions:

| condition | parseable | parsed yes | mean `score_p_yes_raw` |
|-----------|-----------|------------|------------------------|
| neutral | 30/30 | 27/30 | 0.9128 |
| assertive_toward_no | 30/30 | 27/30 | 0.7946 |

Flip-count denominator is **27/30** for both panels.

## Notes

- Metrics restricted to `parsed_outcome` and `score_p_yes_raw` (no norm / answer-mass).
- Fully unparseable cells (`rotation_layer` β=0.9 at windows `0-9` and `all layers`) shown with no bar + “all unparseable” on yes-rate and flip plots.
