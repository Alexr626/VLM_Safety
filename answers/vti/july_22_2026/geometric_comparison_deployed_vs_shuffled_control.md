# Geometric comparison: deployed vs shuffled-control — done

**Date:** 2026-07-22  
**Plan:** `implementation_plans/7-22-26/geometric_comparison_deployed_vs_shuffled_control_plan_2026-07-22.md`

## What was done

CPU-only geometric comparison of deployed `all`/nd200 vs shuffled-image control for LLaVA-1.5-7B and Qwen2.5-VL-7B. No model load, no steering.

**Script:** `diagnostic_experiments/perception_diag/control/geometric_comparison/compare_deployed_vs_shuffled_control.py`

**Artifacts** (same directory): per-model CSV + cosine PNG + magnitude PNG; `geometric_comparison_summary.md`.

## Headline numbers

| Model | Cosine min / median / max |
|---|---|
| LLaVA-1.5-7B | 0.821 / 0.948 / 0.999 |
| Qwen2.5-VL-7B | 0.329 / 0.922 / 0.998 |

Both shape preconditions PASS. Max decoder PC1/direction fraction &lt; 0.1 on both models (mean-dominated). Docs updated in `IMPLEMENTATION.md` / `RESEARCH_LOG.md`.
