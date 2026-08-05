# Qwen AMBER on lambdab2 — status

Date: 2026-07-31 ~13:05 local

## AMBER: finished

- `metric_summary.json`: **37 / 37** (1 baseline + 36 steered; all layer sets × nd × beta).
- Missing: none.
- Last AMBER cell finished: **2026-07-31 12:58:03** —
  `vti_textual_additive_mlp__b0.9__dall__nd100__meandiff__layers_15_24`

Results dir: `evaluation/results/2026-07-30/qwen2.5-vl-7b-instruct/amber/`

## What the driver is doing now

Still on lambdab2: `run_steering_visual_reasoning_validation.sh Qwen/Qwen2.5-VL-7B-Instruct` (pid 2053146, since Jul 30).

Current child (since 12:58): CHAIR baseline —
`run_eval.py … --benchmarks chair --interventions no_intervention … pinned_chair_500`
CHAIR cells with `metric_summary.json` so far: **0** (dir created; run in progress).

Log: `logs/steering_visual_reasoning_validation_qwen25_2026-07-30.log`
