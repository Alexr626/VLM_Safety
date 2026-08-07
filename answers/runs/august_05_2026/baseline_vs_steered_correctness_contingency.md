# Baseline × steered correctness contingency tables

Date: 2026-08-05

Paired 2×2 tables: rows = baseline correct/incorrect; columns = steered correct/incorrect.

Created under (nd fixed at 500; one table per beta × layer window):

- `.../llava_amber_results/contingency_tables/`
- `.../qwen_amber_results/contingency_tables/`
- `.../llava_pope_results/contingency_tables/`
- `.../qwen_pope_results/contingency_tables/`

Each has `baseline_vs_steered_correctness_contingency_nd500.md` and `.json`.

POPE files include per-split tables (random / popular / adversarial) plus a pooled 600-item section.

Correctness: `_normalize_yes_no(response) == ground_truth`; unparsed = incorrect.
