## VTI Mean-Difference Implementation Patterns

This note records the source excerpts and API patterns needed to implement raw mean-difference textual VTI directions for `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md`.

Key findings:

- `evaluation/interventions/vti/directions_meandiff.py` does not exist. The only match for `directions_meandiff` is the implementation plan.
- The existing `demos_850` PCA partition path lives in `evaluation/interventions/vti/directions_partition.py` and writes standard `directions.npz` plus `metadata.json` through `save_textual_v2_directions`.
- Existing activation cache stacks are loaded through `ActivationCache`, `variant_suffix("value") -> "vl_v2_value"`, and dimension/all suffixes like `vl_v2_all`.
- The PCA path constructs diffs as `value - h_value`, documented by `DIFF_POLARITY = "value_minus_h_value"`, then slices off the embedding row with `full[1:]` before saving decoder-layer directions.
- `eval_runner.run_evaluation` defaults `chair_max_new_tokens` to 256, but `evaluation/run_eval.py`, `readme.md`, and `run_exp1_repro_grid.sh` still use or document 64.
- `print_comparison_table` reads result directories under `{results_root}/{run_date}/{model_short}`, discovers `metric_summary.json`, renders POPE split columns first, then AMBER/CHAIR/Hallusion/MMHal columns, and prints legends for AMBER/CHAIR/MMHal.

Relevant files:

- `evaluation/interventions/vti/directions_partition.py`
- `evaluation/interventions/vti/directions_v2.py`
- `evaluation/interventions/vti/intervention.py`
- `evaluation/interventions/__init__.py`
- `evaluation/runners/eval_runner.py`
- `evaluation/run_eval.py`
- `evaluation/chair_amber_diagnostics/step0_chair_token_cap.py`
- `src/paths.py`
- `evaluation/vti_rotation_strength/rotation_strength.py`
- `evaluation/run_scripts/extract_demos850_partition_directions.py`
- `evaluation/chair_amber_diagnostics/run_scripts/run_exp1_repro_grid.sh`
- `readme.md`
