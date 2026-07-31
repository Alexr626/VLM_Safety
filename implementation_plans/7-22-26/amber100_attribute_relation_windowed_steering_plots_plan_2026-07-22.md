# AMBER-100 Attribute/Relation Windowed Steering Plots (LLaVA-1.5)

Date: 2026-07-22
Status: approved by Alex in chat; ready for implementation
Delivery note: the analyst's MCP write access is down; Alex saves this file to `implementation_plans/` under this exact filename before handing to Cursor.
Scope: plots only. LLaVA-1.5 only. No runs, no new data files — every subset below is a filter of existing manifests.

## Question (descriptive)

How do the leading-prefix effect and the windowed-steering effects look on AMBER-100's attribute and relation strata? Alex's prior expectation (from an earlier experiment) is that LLaVA is substantially more prefix-susceptible on attribute/relation reasoning than on existence items (where POPE-30-yes/no showed near-inertness). The baseline bars give the prefix-susceptibility read directly; the steered bars give the window x beta structure. Descriptive — no gating hypothesis.

## Verified inputs (checked 2026-07-22 against uploaded IMPLEMENTATION.md, RESEARCH_LOG.md, and a sample cell manifest)

- Steered dumps: `data/amber/dumps/llava-1.5-7b-hf/amber100_windowed_steering/` — 63 steered cells. Each cell manifest has 200 records = 100 items x 2 gold-conditional conditions. Verified in the sample cell `additive_mlp_0.2_layers_0_9`: gold=no items carry `neutral` + `assertive_toward_yes`; gold=yes items carry `neutral` + `assertive_toward_no`; strata are 20 items per (qtype x gold) with qtype in {existence, attribute, relation}; all 200 records status=ok; manifests carry `qtype`, `gold`, `condition_id`, and the standard `score_*` fields.
- Baseline is NOT in-grid for this run tag (unlike `pope30_{yes,no}_windowed_steering`): it lives in the separate tree `data/amber/dumps/llava-1.5-7b-hf/amber100_baseline/baseline/manifest.jsonl` (500 records = 100 items x 5 conditions; JSONL-normalized per the 2026-07-21 log entries). Same `max_new_tokens=128` as the windowed run — comparable.
- Plot machinery: `diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py` (`--model`, `--datasets`), writing per-model x dataset dirs under `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/{model}/{dataset}/`; grey hatched no-intervention baseline bars; per-method PNGs (rotation_mlp | rotation_layer | additive_mlp) for accuracy / flips / mean first-token probability; gold-aware metric definitions; rotation @ layer beta=0.9 omitted from plots by convention (present in dumps).

## Subsets (each a filter of existing manifests; named per the no-silent-construction rule)

| dataset key | filter | n items | conditions present |
|---|---|---|---|
| `amber100_attribute_yes` | qtype==attribute, gold==yes | 20 | neutral + assertive_toward_no |
| `amber100_attribute_no`  | qtype==attribute, gold==no  | 20 | neutral + assertive_toward_yes |
| `amber100_relation_yes`  | qtype==relation,  gold==yes | 20 | neutral + assertive_toward_no |
| `amber100_relation_no`   | qtype==relation,  gold==no  | 20 | neutral + assertive_toward_yes |

The existence stratum (20 gold=no) is deliberately excluded per Alex's request; it remains available for a later existence-focused read.

## Implementation

Extend `plot_pope30_windowed_steering.py` so `--datasets` accepts the four subset keys above (keep the script name; renaming is deferred — references exist in the log). Three differences from the pope datasets, all to be handled explicitly:

1. Dump source: read `data/amber/dumps/{model}/amber100_windowed_steering/{cell}/manifest.jsonl` and filter rows by the manifest's `qtype` and `gold` fields to the subset definition.
2. Baseline source: load from the separate `amber100_baseline/baseline/manifest.jsonl`, filtered per item to the same two gold-conditional conditions (`neutral` + the gold-opposing assertive). Flip pairing joins (item_id, condition_id) across the two trees. If the LLaVA baseline manifest still has pretty-printed residue from the 2026-07-21 JSONDecodeError incident, normalize it first (as was done for Qwen with `--normalize_manifests`) and record the rewrite in the log.
3. Captions/subtitles state n=20 per subset (half the POPE-30 n; window-to-window differences correspondingly noisier).

## Plots (same conventions as the existing pope30 plots, per subset)

For each subset key, the standard three metric families, one PNG per method, filenames following the script's existing `{dataset}_{metric}_by_layer_window_{method}.png` scheme:

- Accuracy by layer window: gold=yes subsets — fraction of parseable responses parsed "yes"; gold=no subsets — fraction parsed "no". Unparseable counts annotated per bar as in the pope plots.
- Flips from baseline by layer window: gold=yes — baseline-yes -> steered-no; gold=no — baseline-no -> steered-yes. Pairs with an unparseable member excluded and annotated; baseline correct-count per condition in the panel subtitle (the denominator).
- Mean first-token probability by layer window: gold=yes — mean `score_p_yes_raw`; gold=no — mean `score_p_no_raw` (filenames use `mean_p_no_raw` for the no-subsets, mirroring the pope30_no convention).

Layout per PNG unchanged: windows 0-9 ... 22-31 + all on x, betas side by side within each window group, condition rows (neutral / leading), grey hatched baseline bars, rotation @ layer beta=0.9 omitted.

Binding metric exclusions from chat: `p_yes_norm` and `answer_mass` appear nowhere.

Output: 4 subset dirs x 3 metrics x 3 methods = 36 PNGs under `plots/llava-1.5-7b-hf/{subset_key}/`.

## Invocation

```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --model llava-1.5-7b-hf \
  --datasets amber100_attribute_yes amber100_attribute_no amber100_relation_yes amber100_relation_no
```

## Open questions

1. Qwen: the same invocation with `--model qwen2.5-vl-7b-instruct` applies once/if the Qwen `amber100_windowed_steering` dumps are complete — not scheduled in this plan.
2. After eyeballing, a cross-stratum comparison figure (attribute vs relation vs the POPE existence numbers, baselines only) may be worth a small follow-up task — flagged, not included.
3. If any of the four subsets shows a stratum count other than exactly 20 items x 2 conditions per cell during filtering, stop and report rather than plot.
