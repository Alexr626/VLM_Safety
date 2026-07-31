# Why Qwen2.5-VL AMBER baselines differ between Exp1 and Exp2

**Date:** 2026-06-28  
**Question:** Same pinned AMBER-450 subset — why does Exp1 baseline show acc=64.2% / yes_ratio=14.4% / n_unparsed=84 while Exp2 baseline shows acc=76.0% / yes_ratio=17.8% / n_unparsed=21?

## Short answer

The **subset IDs are identical** (`data/amber/pinned_amber_disc_450.json`, n=450), but the two “baselines” are **not the same inference run** and were produced under **different generation configs**. For Qwen2.5-VL specifically, Exp1’s baseline run failed on 65 samples (empty responses from caught exceptions, almost certainly vision-encoder OOM at native resolution), which alone explains most of the metric gap. Where both runs produced non-empty text, predictions agree on **366/366** parseable samples.

## Data sources in the diagnostic summary

| Experiment | Baseline source | Path |
|------------|-----------------|------|
| Exp1 | `run_eval.py` → `no_intervention` cell | `evaluation/results/2026-06-22/qwen2.5-vl-7b-instruct/amber/no_intervention/metric_summary.json` |
| Exp2 | Regenerated inline at start of rotation-strength sweep | `evaluation/results/2026-06-22/qwen2.5-vl-7b-instruct/amber_rotation_strength/sweep_uniform_rotation_layer_n450.json` → `metrics_by_beta.baseline` |

Exp2 does **not** load Exp1’s `responses.json`; it calls `_gen()` fresh for each sample in `rotation_strength_chair_amber.py`.

## Config differences (Qwen2.5-VL AMBER)

| Knob | Exp1 (`run_exp1_repro_grid.sh` → `run_eval.py`) | Exp2 (`run_exp2_rotation_strength.sh` → `rotation_strength_chair_amber.py`) |
|------|--------------------------------------------------|-------------------------------------------------------------------------------|
| `max_new_tokens` | **256** (default; only CHAIR gets `--chair_max_new_tokens=64`) | **64** |
| `max_pixels` | **Unset** (native / `longest_edge` ≈ 12.8M) | **1 003 520** (`MAX_PIXELS=1003520`) |
| OOM handling | Exception → `response=""` in `eval_runner._run_one` | Sample skipped on OOM; sweep JSON records `n_failed_oom` |

Recorded in Exp2 sweep JSON: `"max_new_tokens": 64`, `"max_pixels": 1003520`, `"n_failed_oom": 0`.

## Empirical comparison (Qwen2.5-VL, same 450 IDs)

| Metric | Exp1 baseline | Exp2 baseline |
|--------|---------------|---------------|
| accuracy | 0.642 | 0.760 |
| yes_ratio | 0.144 | 0.178 |
| n_unparsed | 84 | 21 |
| neg_item_accuracy | 0.826 | 0.964 |

**Response-level facts:**

- Byte-identical responses: **326 / 450**
- Exp1 completely empty responses: **65 / 450** (Exp2: **0**)
- Prediction differs (any): **63 / 450**
- On samples where **both** parse to yes/no: **366 / 366 agree**

**Counterfactual:** If Exp1 empty responses are excluded (score only n=385 non-empty), accuracy rises to **~75.1%** with n_unparsed=19 — essentially aligned with Exp2’s 76.0% / n_unparsed=21.

Empty Exp1 responses cluster in contiguous index blocks (e.g. indices 434–449), consistent with OOM / run instability rather than random scorer noise.

## Why LLaVA does not show this problem

LLaVA Exp1 vs Exp2 AMBER baselines match to four decimals (acc=76.67%, yes_ratio=39.78%, n_unparsed=0). LLaVA uses fixed 336² resolution and does not hit vision-encoder OOM on these images; Exp2 also ran with `max_pixels=None`. **444/450** baseline strings are byte-identical despite Exp1 using max_new_tokens=256 and Exp2 using 64 (AMBER answers are short).

## Conclusion

Your intuition is correct that **the same examples** were used, but incorrect that **the same baseline generations** were reused or that **generation settings were frozen** across experiments. The diagnostic summary labels both “baseline,” but they are independent Qwen runs with mismatched vision/decode policy. The large accuracy gap is primarily **Exp1 generation failures** (65 empty → unparsed), not a contradiction in the pinned subset or scorer.

## If aligned baselines are desired

1. Align `max_new_tokens` for AMBER (64 vs 256).
2. Align Qwen `max_pixels` (cap both or cap neither).
3. Have Exp2 load Exp1 `no_intervention/responses.json` instead of regenerating, **or** re-run Exp1 Qwen AMBER baseline with `MAX_PIXELS=1003520`.
4. Document in reports that cross-experiment “baseline” rows are comparable only when generation config matches.
