# LLaVA AMBER-1500 plots + contingency (2026-08-05)

Date: 2026-08-06

## Design differences vs 2026-07-30 (from plan)

- Subset: pinned AMBER-1500 (not 450); baseline accuracy 1135/1500.
- Direction sample size: **nd = 500 only** (no 50/100/200).
- Two steering conditions from the same 500 demos: **mean-difference** (`meandiff`) and **VTI PC1+mean** (`pc1_plus_mean`).
- Same betas {0.2, 0.5, 0.9} and LLaVA layer windows {all, 5–14, 20–29}.
- 19/19 LLaVA cells complete.

## Artifacts

Under `evaluation/results/2026-08-05/_analysis_steering_vector_validation_continuation/llava_amber_results/`:

```
mean_difference/          # baseline vs meandiff
vti_pc1_plus_mean/        # baseline vs VTI PC1+mean
  plots/steering_vector_sample_size/500/
  plots/layer_windows/{early,late,all}/ + stitched
  plots/accuracy_yes_vs_no_comparisons/ (+ stitched gold-label)
  contingency_tables/baseline_vs_steered_correctness_contingency_nd500.{md,json}
```

Layer-window accuracy-vs-beta plots have a single line (nd=500 only), matching the expanded grid.
