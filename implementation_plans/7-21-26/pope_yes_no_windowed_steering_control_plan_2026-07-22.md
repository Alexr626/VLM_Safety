# Windowed Steering on POPE-30-yes (expanded) and POPE-30-no (control)

Date: 2026-07-22
Status: approved by Alex in chat; ready for implementation
Depends on: `pope_yes_no_30_dataset_rebuild_plan_2026-07-22.md` — pins built, asserts passed, verification table reported. Do not launch before that.
Reuses: all windowed-steering machinery from `layer_windowed_steering_yes_no_probability_plan_v2_2026-07-21.md` (layer_indices hooks, --windowed_grid, --conditions gold_conditional). No new steering code.

## Question and pre-registered predictions

The completed all-gold-yes POPE runs cannot distinguish two hypotheses about the late-window rotation effect on Qwen2.5-VL (p_yes 0.436 -> 0.86 under the toward-no prefix at 18-27 beta 0.9; 10/10 yes at all-layers):

- H-bias (yes-bias knob): steering shifts the yes/no operating point regardless of image content. Predicts on gold=no items: accuracy WORSENS under the toward-yes prefix and worsens under neutral too, most sharply in Qwen late windows (15-24, 18-27, all) at beta 0.9.
- H-grounding (counter-sycophancy / grounding restoration): steering restores image-grounded answers. Predicts on gold=no items: "no" answers survive the toward-yes prefix; neutral accuracy is not degraded.

Secondary pre-registered check: the Qwen 5-14 sign reversal (p_yes pushed DOWN to 0.271 at beta 0.9 under the toward-no prefix). If it replicates as a no-push, it should IMPROVE gold=no accuracy in the 5-14 window.

Tertiary: LLaVA's small additive-beta-0.9 early-band effect (0-9 and all-layers, 3/9 unique flips toward no) — does it appear on gold=no items as an accuracy gain (no-push helps) and does 0-9 continue to track all-layers.

All prior numbers are n=10 unique items, one seed; this run is n=30 unique per set.

## Datasets and conditions

- POPE-30-yes: `data/pope/augmented_pope30_yes.jsonl` (pin `pinned_pope_existence_yes_30.json`, rebuilt). Gold-conditional conditions: neutral + `assertive_toward_no` per item.
- POPE-30-no: `data/pope/augmented_pope30_no.jsonl` (pin `pinned_pope_existence_no_30.json`). Gold-conditional conditions: neutral + `assertive_toward_yes` per item.
- 60 records per cell per dataset (30 items x 2 conditions). max_new_tokens=128, greedy decoding, Qwen max_pixels default 1003520, direction demos_v2 `all`@nd200 (`--num_demos 200` default).

## Cells

Per model per dataset: the windowed grid (3 configs x beta {0.2, 0.5, 0.9} x 7 LLaVA / 6 Qwen windows) PLUS one no-intervention `baseline` cell — 64 LLaVA / 55 Qwen cells. Baselines must be run this time: 20 of the yes items and all 30 no items have no baseline anywhere. If `--windowed_grid` does not currently include the `baseline` cell spec, add it to the generated grid (flagged as the only code touch in this plan).

Records: LLaVA (64+64) x 60 = 7,680; Qwen (55+55) x 60 = 6,600; total 14,280.

## Run structure and order (time-critical ordering per Alex)

Run tags: `pope30_no_windowed_steering` and `pope30_yes_windowed_steering`, under `data/pope/dumps/{model_short}/`. Old dump trees untouched.

1. LLaVA, POPE-30-no (the discriminating evidence; ~3,840 records, roughly 4-6.5 h)
2. LLaVA, POPE-30-yes
3. Qwen2.5-VL, POPE-30-no
4. Qwen2.5-VL, POPE-30-yes

Single free A6000, unsharded, nohup + logs, resumable as before. If two GPUs are free, Qwen may start its no-set in parallel with LLaVA's yes-set — Alex has approved that specific overlap; nothing else runs in parallel.

## Reproducibility check (reported, not gated)

The 10 overlapping yes items (old pin's unique inputs = new pin items 1-10) re-run under identical settings in every steered cell and baseline. After the yes-set completes per model: compare `response` and all `score_*` fields against the old `pope30_windowed_steering` / `pope30_existence_yes_baseline` manifests for matching (cell_id, item, condition); report the count of divergent records and list any divergent cell_ids. Divergence is an environment-drift signal for Alex, not a stop condition.

## Metrics

Same primitives as before per record: raw response, `parsed_outcome`, `score_p_yes_raw`, `score_p_no_raw`, logit sums/margin, degeneracy/truncation/status. Binding from chat: `p_yes_norm` and `answer_mass` are not reported or plotted anywhere.

For the gold=no set, define explicitly in all summaries: accuracy = fraction of parseable responses parsed "no"; the flip direction of concern is baseline-"no" -> steered-"yes" (`flips_no_to_yes`), the signature of H-bias. For the yes set, as before: accuracy = fraction parsed "yes"; `flips_yes_to_no`.

## Artifacts

- Standard dump trees (manifest.jsonl, acts/, norms/, metadata) per cell; acts capture stays on.
- Consolidated JSON: extend `build_windowed_steering_summary.py` to include the two new run tags with a `dataset` field (`pope30_yes` / `pope30_no`); same schema otherwise; `_completeness` as before.
- Condensed summaries matching the existing 2x2 format, one pair per dataset: `pope30_yes_mlp_2x2_accuracy_and_flips_summary.json`, `pope30_no_mlp_2x2_accuracy_and_flips_summary.json`, and the corresponding mean-p_yes files — same axes (model x additive/rotation @ mlp), with the gold=no accuracy/flip definitions above, and n=30 stated. rotation_layer excluded from the 2x2 as before but present in the dumps and consolidated JSON.

## What confirms or falsifies

Read per (model, config, beta, window), gold=no set: neutral accuracy and toward-yes-prefix accuracy vs baseline. H-bias confirmed if late-window/all-layers rotation on Qwen at beta 0.9 drops gold=no accuracy substantially in both conditions while raising gold=yes accuracy; H-grounding confirmed if gold=no accuracy holds or rises under the prefix. 5-14 improving gold=no accuracy = sign-reversal replication. Mixed outcomes get interpreted in chat before any further plan.

## Open questions

1. If runtime estimates are badly off after the first run tag, report actuals before continuing the sequence.
2. Whether the old 2x2 summary files should be regenerated with /3-corrected counts for side-by-side reading with the new n=30 files — cheap, but only on Alex's word.
