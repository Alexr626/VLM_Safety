# POPE-30 Windowed Steering: Plotting Task (LLaVA-1.5)

Date: 2026-07-21
Status: approved by Alex in chat; ready for implementation
Scope: plots only — no new runs, no new data files. Three plot files from existing manifests.

## Inputs

- Steered cells: `data/pope/dumps/llava-1.5-7b-hf/pope30_windowed_steering/{cell_dir}/manifest.jsonl` — 63 cells (3 configs x 3 betas x 7 windows), verified present on disk 2026-07-21.
- Baseline: `data/pope/dumps/llava-1.5-7b-hf/pope30_existence_yes_baseline/baseline/manifest.jsonl` (cell dir verified on disk), FILTERED to the two conditions used in the steered runs: `neutral` and `assertive_toward_no`. This is a filter of the existing baseline dump; build nothing new.
- All 30 POPE items are gold=yes; every item has exactly the two conditions above.

## Metric decisions (from chat, binding)

- Use ONLY: `parsed_outcome` (from the actual generated response) and `score_p_yes_raw` / `score_p_no_raw` (unconditional first-token probabilities at the last prefill position).
- Do NOT plot or report `score_p_yes_norm` or `score_answer_mass` anywhere. (Rationale recorded: the conditional P(yes | yes-or-no) is meaningless when almost no first-token mass is on the answer tokens, as in degenerate cells.)

## Shared layout (all three files)

- Grouped bar charts. Facet rows = condition: "neutral question" (top), "leading clause toward no" (bottom). Facet columns = config: rotation @ mlp, rotation @ layer, additive @ mlp.
- x-axis = steering layer window, ordered by window start: 0-9, 5-14, 10-19, 15-24, 20-29, 22-31, then "all layers" rightmost with a small visual gap or divider (it is not part of the sliding sequence).
- Within each window group: three bars side by side, one per beta (0.2, 0.5, 0.9). One consistent beta-to-color mapping across all three files, with a single legend.
- Baseline: horizontal dashed line per condition panel (baseline has no window or beta dependence). Label it "no-intervention baseline" in the legend.
- Titles, axis labels, annotations in plain English; every file legible standalone by a reader with no context. n=30 items per cell-condition; state this in a caption or subtitle.
- Matplotlib only; PNG output.

## Plot files (self-describing names)

Output directory: `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/`
Script: `diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py`

1. `pope30_parsed_yes_rate_by_layer_window.png`
   - y = fraction of PARSEABLE responses whose parsed answer is "yes" (on this all-gold-yes set this equals accuracy among parseable).
   - Per-bar annotation of the unparseable count when nonzero (e.g. "u=7"), so a collapsed cell reads as collapsed rather than as a low yes-rate.
   - A cell with zero parseable responses gets no bar and a full-height annotation "all unparseable".
   - Title: "LLaVA-1.5, POPE-30 (all gold=yes): parsed yes-rate by steering layer window".

2. `pope30_mean_p_yes_raw_by_layer_window.png`
   - y = mean unconditional P(yes) at the first answer token (`score_p_yes_raw`) across ALL 30 items — the score exists regardless of parseability, so no items are excluded here.
   - Thin error bars: bootstrap 95% CI over items (1000 resamples).
   - Title: "LLaVA-1.5, POPE-30: mean unconditional P(yes) at first answer token, by steering layer window".

3. `pope30_flips_from_baseline_yes_by_layer_window.png`
   - y = integer count of items flipped from baseline "yes" to steered "no".
   - Lineage (per chat): pair each steered record with the baseline record for the same `item_id` AND same `condition_id`; count pairs where baseline `parsed_outcome` == "yes" and steered `parsed_outcome` == "no".
   - Pairs where either member is unparseable are EXCLUDED from the count and reported as a per-bar annotation when nonzero (e.g. "x=5 excl").
   - Print the baseline yes-count for each condition in the panel subtitle (the denominator for the count, e.g. "baseline: 28/30 parsed yes under neutral").
   - Integer y-axis ticks.
   - Title: "LLaVA-1.5, POPE-30: items flipped from baseline yes to no, by steering layer window".

## Not in scope

- No ROC / AUC on POPE-30 (single gold class; no false-positive rate is definable). ROC plots are staged behind LLaVA AMBER-100 completion and will be specified separately.
- No Qwen plots (Stage B not started).
- No aggregate composite metrics beyond the flip count defined above.

## Open questions

1. If the per-bar annotations get visually crowded (63 cells x 2 conditions), acceptable fallback: annotate only nonzero values, smaller font — do not drop the information.
2. If baseline parse coverage differs materially between conditions, flag it in chat before Alex reads plot 3 (it changes the flip-count denominator interpretation).
