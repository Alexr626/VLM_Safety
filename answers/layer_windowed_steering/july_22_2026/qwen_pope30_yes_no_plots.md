# Qwen POPE-30-yes/no windowed-steering plots

Generated 2026-07-22 to match LLaVA layout.

## Paths (9 PNGs each)

- `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/qwen2.5-vl-7b-instruct/pope30_yes/`
- `…/plots/qwen2.5-vl-7b-instruct/pope30_no/`

## Metrics

Same as LLaVA: accuracy, mean P(yes) for yes-set / mean P(no) for no-set, flips; methods rotation_mlp, rotation_layer (β=0.9 omitted), additive_mlp; grey hatched in-grid baseline.

## Baseline headline (n=30)

- yes neutral: acc=0.733, mean_p_yes=0.7131
- yes assertive_toward_no: acc=0.536 (28 parseable), mean_p_yes=0.3212
- no neutral: acc=1.000, mean_p_no=0.9419
- no assertive_toward_yes: acc=0.967, mean_p_no=0.8454
