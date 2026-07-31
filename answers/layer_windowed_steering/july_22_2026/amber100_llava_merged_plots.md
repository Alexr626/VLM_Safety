# LLaVA AMBER-100 merged plots

**Date:** 2026-07-22

## Confirmed decisions

1. Metrics: accuracy, mean first-token prob, flips
2. Nested conditions: neutral + assertive in each gold×method cell
3. Mean-P PNG: top = P(yes), bottom = P(no)
4. Output: `…/plots/merged_plots/amber-100-runs/`
5. Methods: additive @ mlp + rotation @ mlp only

## How to regenerate

```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --joint_amber_llava
```

Optional: `--amber_capabilities attribute` or `relation`.

## Layout

| | additive @ mlp | rotation @ mlp |
|---|---|---|
| gold=yes | neutral + assertive | neutral + assertive |
| gold=no | neutral + assertive | neutral + assertive |

Six PNGs: attribute × {accuracy, mean_p, flips} and relation × same.
