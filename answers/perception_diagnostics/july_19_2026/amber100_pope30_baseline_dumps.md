# AMBER-100 + POPE-30 baseline dumps — decisions applied

Date: 2026-07-19

## Decisions (from Romanus)

1. **AMBER-100:** keep existing AMBER-25 IDs; fill to **20 per stratum** (5 strata: existence×no, attribute×yes/no, relation×yes/no).
2. **Steering:** baseline only for now.
3. **POPE:** **POPE-30** — 10 gold=yes from each of random / popular / adversarial.
4. **`max_new_tokens`:** 128 for both LLaVA and Qwen2.5.
5. **Separate runs** — do not concatenate into existing `amber25_*` dump trees.

## Artifacts

| Kind | Path |
|------|------|
| AMBER-100 pin | `data/amber/pinned_amber_disc_100.json` |
| POPE-30 pin | `data/pope/pinned_pope_existence_yes_30.json` |
| Augment JSONL | `augment/outputs/augmented_amber100.jsonl`, `augmented_pope30.jsonl` |
| Driver | `run_scripts/run_amber100_pope30_baseline_dumps.sh` |
| Dump tags | `amber100_baseline`, `pope30_existence_yes_baseline` under each model’s `dumps/` |

## Dump completion (2026-07-19)

| Model | amber100 | pope30 |
|-------|----------|--------|
| LLaVA-1.5-7B | 500 rows / 500 acts | 150 / 150 |
| Qwen2.5-VL-7B | 500 / 500 | 150 / 150 |

Log: `diagnostic_experiments/perception_diag/logs/amber100_pope30_baseline_dumps_20260719_145123.log`
