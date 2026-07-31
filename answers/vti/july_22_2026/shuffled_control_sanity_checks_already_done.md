# Shuffled-control direction sanity checks — status

**Date:** 2026-07-22  
**Plan:** `implementation_plans/7-22-26/shuffled_control_direction_sanity_checks_plan_2026-07-22.md`

## Verdict

Already implemented and run. Both cells (LLaVA-1.5-7B, Qwen2.5-VL-7B) pass all gating checks. No re-run needed unless you want `--force_recompute`.

## Code

- `evaluation/interventions/vti/shuffled_control.py`
- `evaluation/run_scripts/extract_shuffled_control_directions.py`

## Artifacts

- Derangement: `experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json` (seed 1234, 200 ids, 0 fixed points)
- Per model under `experiment_artifacts/vti/{model_short}/shuffled_control/`:
  - `all_nd200/{directions.npz,metadata.json,components.npz}`
  - `_act_cache/` (400 files each)
  - `shuffled_control_sanity_report_{model_short}.md`

## Gating (both models)

| Check | Result |
|---|---|
| Fixed points | 0 |
| Captions unchanged | 200/200 |
| Positional id mismatches | 0 |
| Forwards executed | 400 |
| Shape | LLaVA `[32, 4096]`; Qwen `[28, 3584]` |

Documented in `RESEARCH_LOG.md` (2026-07-22 entry) and `IMPLEMENTATION.md` (§ Shuffled-control direction).
