# Mean-difference textual steering validation — sanity checks that gate the grid
design_spec: designs/07_30_26/steering_vector_visual_reasoning_validation.md

Date: 2026-07-30
Status: run this first. The main plan
(`implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md`)
waits on the results of checks 6 and 7, which can change what the design measures.

---

## Purpose

The design launches 370 benchmark invocations over 1550 items each. Nine assumptions about the
data, the parser, the extraction, and the intervention plumbing sit under it. Every one is
cheap to check and expensive to discover afterwards. Two of them (checks 6 and 7) have on-disk
evidence that does not support what the spec assumes, so their results go back to Alex before
the grid is launched.

Checks 1, 2, 3, 8, 9 are CPU-only and read files already on disk. Checks 4, 5, 10 need a GPU
and ≤ 30 items. Check 6 needs a GPU and 20 images × 3 caps × 2 models. Check 7 is a read of
files already on disk plus a re-read after the fresh baselines land.

Run order: 1, 2, 8, 9 (pure reads) → 3 (needs the extraction from the main plan step 1) → 5, 4,
10 (GPU smoke) → 6, 7 (the two that report back to Alex).

---

## Check 1 — every activation the mean-difference needs is on disk and loadable (gate)

**What it verifies.** That the one primitive this design requires can be built from cache with
zero forward passes, as the spec's Primitives section states.

**Procedure.** CPU only, no model load.

```python
from evaluation.interventions.vti.directions_partition import build_or_load_partition
from evaluation.interventions.vti.directions_v2 import variant_suffix, _stack_from_act_dict, act_cache_dir
from src.extraction import ActivationCache
from src.paths import vti_demos_850_path, vti_demos_850_partition_path
```

For each `model_short` in `("llava-1.5-7b-hf", "qwen2.5-vl-7b-instruct")` and each block size in
`(50, 100, 200, 500)`, for every id in `partition["blocks"][str(n)]`:

- `ActivationCache(str(act_cache_dir(model_short))).exists(demo_id, suffix=variant_suffix("value"))`
- the same for `suffix=variant_suffix("all")`
- load both, build the stack with `_stack_from_act_dict`, assert shape is `(33, 4096)` for
  `llava-1.5-7b-hf` and `(29, 3584)` for `qwen2.5-vl-7b-instruct` (`num_layers + 1` rows, row 0
  the embedding row), and `np.isfinite(...).all()`.

Use `ActivationCache.load` only. Never call `.save`. Never construct a wrapper.

**Expected.** 850 ids × 2 variants × 2 models = 3400 files present, all finite, all correctly
shaped. Both `_act_cache` directories currently hold 5100 files each (850 demos × 6 variants:
`value` plus the five entries of `DIMENSIONS`, `directions_v2.py:33`).

**What invalidates the design.** A single missing or non-finite file means the mean-difference
set for that block cannot be built from cache, the spec's "zero forward passes" claim is false,
and producing the missing activations is a forward-pass job that is not in this plan. Stop and
report which ids and variants are missing.

**Output.** `experiment_artifacts/vti/meandiff_activation_cache_coverage_2026-07-30.json` with,
per model and block size, `n_ids`, `n_present_value`, `n_present_all`, `missing_ids`, and the
observed stack shape.

---

## Check 2 — the four partition blocks are disjoint and hash-matched (gate)

**What it verifies.** That nd50 / nd100 / nd200 / nd500 are four disjoint demo blocks over the
same 850-row pool, which is what the spec's Primitives section asserts ("The four blocks are
**disjoint**, not nested").

**Procedure.** Call the existing checker, unchanged:

```python
from evaluation.interventions.vti.directions_partition import (
    build_or_load_partition, check_partition_integrity)
part = build_or_load_partition(vti_demos_850_path(), vti_demos_850_partition_path(), seed=42)
check_partition_integrity(vti_demos_850_path(), part)   # raises on any failure
```

It asserts 850 unique ids, `content_hash_sha256_16` agreement between
`data/vti/demos_850_partition_s42.json` and `data/vti/demos_850.jsonl`, per-block length and
uniqueness, pairwise disjointness, and that the union is the full id set
(`directions_partition.py:175-208`).

**Expected.** No exception. The recorded hash is `ba05bd960cad0c18`
(`experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demos850_ba05bd96_all_nd200_s42_r2_partition/metadata.json`),
`selection_policy: disjoint_partition`, `seed: 42`.

**What invalidates the design.** A hash mismatch means the demos file moved under the partition
and the existing `_r2_partition` direction sets are no longer reproducible from it; overlapping
blocks mean the four sample sizes are not the independent samples the spec describes. Either
one stops the extraction.

---

## Check 3 — the extracted mean-difference direction is usable at every layer (gate)

**Runs after** the extraction step of the main plan (step 1), on its outputs. This is a
descriptive check on the eight new artifacts, not a comparison against anything.

**What it verifies.** That `steer` can consume the direction at every steered layer.
`steer.py:43` runs `F.normalize(direction.float(), dim=-1)` before applying it, so a layer whose
mean-difference vector is the zero vector, or non-finite, is silently unsteered or poisons the
activation with NaN.

**Procedure.** For each of the eight new `directions.npz`:

- `load_textual_v2_directions(cache_dir)` returns `(num_layers, hidden_dim)` float32; assert the
  shape is `(32, 4096)` for LLaVA and `(28, 3584)` for Qwen, matching the existing
  `_r2_partition` sets already on disk.
- `np.isfinite(directions).all()`.
- Per-layer L2 norm > 0 for every one of the 32 / 28 decoder rows. Record all norms.
- `metadata.json` records `n_pairs` equal to the block size, `ids_used` equal to
  `partition["blocks"][str(n)]` as an ordered list, `diff_polarity == "value_minus_h_value"`,
  and a construction field that is **not** `live_pc1_plus_mean`.
- No `components.npz` exists in the new directories.

**Expected.** All eight pass. As a scale reference already on disk, the existing
`_r2_partition` metadata records `mean_diff_layer_norms_mean_over_demos[0] == 0.0` for the
embedding row — the embedding row is dropped by the `[1:]` slice and never reaches the hook, so
a zero there is expected and is not a failure; a zero in any decoder row is.

**What invalidates the design.** A zero-norm or non-finite decoder row means that layer
contributes nothing (or a NaN) under `additive` steering, and the all-layers arm is not the
all-layers arm.

**Output.**
`experiment_artifacts/vti/meandiff_direction_shapes_and_layer_norms_2026-07-30.json`.

---

## Check 4 — steering actually reaches the model, at exactly the requested layers (gate)

**What it verifies.** That the new `--direction_dir` and `--layer_range` plumbing does what it
says, and that hooks fire. A steered run that is byte-identical to baseline is a null by
construction, and this project's known trap is that `device_map` sharding silently breaks
hook-based interventions.

**Procedure.** lambdab2, `CUDA_VISIBLE_DEVICES=0`, LLaVA only, POPE random, `--limit 10`, a
throwaway `--run_date 2026-07-30_smoke` and `--output_dir evaluation/results` (a smoke date dir,
never a grid date dir).

Four invocations at `--beta 0.9` (largest beta in the grid, so any effect is visible), using the
nd200 mean-difference direction set:

| Arm | `--layer_range` | Hooks expected |
|---|---|---|
| baseline | — (`no_intervention`) | 0 |
| all layers | `all` | 32 |
| early window | `5-14` | 10 |
| late window | `20-29` | 10 |

Instrument `vti_hook_ctx` for this check only by asserting `len(handles)` after registration
(`hooks.py:134-136`); print it, do not leave the print in.

Assertions:

- Hook count equals the window size in every arm.
- For each steered arm, at least one of the 10 responses differs from the baseline response for
  the same item id.
- `metric_summary.json` → `intervention_config` carries `layer_indices`, `layer_set`,
  `direction_slug`, `variant == "additive"`, `hook_site == "mlp"`, `beta == 0.9`.
- The three steered result directories are distinct paths and do not collide:
  `vti_textual_additive_mlp__b0.9__demos850_ba05bd96_all_nd200_s42_meandiff_partition__Lall`,
  `…__L5-14`, `…__L20-29`.
- `vti_hook_ctx` raises on an out-of-range index: assert `--layer_range 20-31` is accepted for
  LLaVA (31 < 32) and `--layer_range 20-31` raises for Qwen (`hooks.py:92-96`).

**What invalidates the design.** Identical responses across every item in a steered arm means
the direction is not being applied and every steered cell in the grid is a null for a plumbing
reason. A hook count that does not match the window means the layer-set factor is not the
factor the spec names.

**Output.** `evaluation/results/2026-07-30_smoke/_diagnostics/steering_plumbing_smoke_llava.json`.

---

## Check 5 — the model loads on one device, unsharded (gate)

**What it verifies.** That `device_map="auto"` has not split the decoder across devices.
Hook-based steering breaks silently under sharding.

**Procedure.** Immediately after `create_wrapper(model_id).load()` in the runner, before any
generation:

```python
devs = {str(p.device) for p in wrapper.model.parameters()}
assert len(devs) == 1 and next(iter(devs)).startswith("cuda"), devs
```

Run for both models. On lambdab2, launch with `CUDA_VISIBLE_DEVICES=0` — GPU 0 is the only free
device (48673 MiB free; GPUs 1, 2, 3 are each holding ~48 GB). On RunAI the pod exposes the
single allocated GPU as device 0.

This assertion is not smoke-only: it goes into `run_evaluation` as a permanent guard, raising
rather than warning.

**What invalidates the design.** More than one device, or any parameter on `cpu` / `meta`.
`Qwen2VLWrapper.load` sets a `max_memory` budget with CPU offload when total VRAM is under
20 GiB (`src/model.py:1431-1438`); neither target hits that branch, and if it ever does, the run
must stop rather than proceed offloaded.

---

## Check 6 — is 256 new tokens actually enough for CHAIR, on both models (gate; reports to Alex)

**What the spec assumes.** "`CHAIR_CAP = 256` is already frozen in
`evaluation/chair_amber_diagnostics/make_diagnostic_summary.py:37`, set by the Step 0 sweep
whose output is `evaluation/results/2026-06-22/_diagnostics/step0_chair_token_cap.json`."

**What is actually in that file** (factual read, verified 2026-07-30). The sweep ran
`--caps 64 512` — 256 was never measured. It ran LLaVA-1.5-7B only — Qwen2.5-VL was never
measured. n was 20 images. The two measured caps are far apart:

| cap | chair_s | chair_i | avg objects mentioned | avg caption chars | coverage recall |
|---|---|---|---|---|---|
| 64 | 0.300 | 0.1667 | 2.40 | 266.2 | 0.5714 |
| 512 | 0.650 | 0.2400 | 3.75 | 465.1 | 0.8143 |

`CHAIR_CAP = 256` in `make_diagnostic_summary.py` is a module constant that no measurement
supports and that nothing else in the tree reads. Separately, the 2026-06-22 CHAIR baselines on
disk were generated at cap **64**, not 256 — `run_exp1_repro_grid.sh:43` sets `CHAIR_CAP=64`, and
the LLaVA baseline's `avg_caption_len_chars` of 269.806 matches the cap-64 probe's 266.2. They
are therefore not reusable as B1/B2 for a design that requires 256.

**Procedure.** lambdab2, `CUDA_VISIBLE_DEVICES=0`, one model at a time.

```
CUDA_VISIBLE_DEVICES=0 python evaluation/chair_amber_diagnostics/step0_chair_token_cap.py \
  --model llava-hf/llava-1.5-7b-hf --num_images 20 --caps 128 256 512 \
  --seed 1234 --prompt "Please Describe this image in detail." --run_date 2026-07-30
```

and the same with `--model Qwen/Qwen2.5-VL-7B-Instruct`.

`--seed 1234` and the verbatim prompt reproduce the same 20 images the 2026-06-22 probe used
(`_draw_images` samples with `random.Random(seed)` over `data/chair/combined.json`), so the
cap-512 column is directly comparable to the existing file.

**Required code edit (new).** `step0_chair_token_cap.py:140` writes
`evaluation/results/{run_date}/_diagnostics/step0_chair_token_cap.json` — keyed on run date
only, not on model. Running it for two models on one date silently overwrites the first. Change
the filename to `step0_chair_token_cap_{model_short}.json`. Nothing in the tree reads the old
name programmatically (verified by grep), and the 2026-06-22 file keeps its name and stays
reachable.

**What passing looks like.** `chair_s`, `chair_i`, and `avg_objects_mentioned` are materially
unchanged between 256 and 512 for both models, and `avg_caption_len_chars` at 256 is well below
the 256-token ceiling — i.e. captions are terminating on EOS, not on the cap.

**What invalidates the design.** If any of those still move from 256 to 512 on either model,
captions are being truncated at 256 and `chair_i` / `chair_s` move through their denominators
rather than through hallucination behaviour. The CHAIR rows are the ones Alex names as doing the
most work in separating explanations A, B, and C, so a moving cap makes those rows unreadable.
**Report the three-cap table to Alex and stop.** Do not pick a different cap; the cap is a
condition and the spec names 256.

**Output.** `evaluation/results/2026-07-30/_diagnostics/step0_chair_token_cap_llava-1.5-7b-hf.json`
and `…_qwen2.5-vl-7b-instruct.json`, plus a three-row-per-model table in the report.

---

## Check 7 — what fraction of AMBER answers the yes/no parser can read, per model (gate; reports to Alex)

**What the design assumes.** That the parsed answer per item is the primitive under accuracy,
precision, recall, F1, and h+/h-, over the pinned 450.

**What is on disk** (factual read of
`evaluation/results/2026-06-22/{model}/amber/no_intervention/`, verified 2026-07-30):

| model | n_total | n_unparsed | accuracy_overall | yes_ratio | gold-no accuracy | gold-yes accuracy |
|---|---|---|---|---|---|---|
| llava-1.5-7b-hf | 450 | 0 | 0.7667 | 0.3978 | 0.8007 (n=276) | 0.7126 (n=174) |
| qwen2.5-vl-7b-instruct | 450 | 84 | 0.6422 | 0.1444 | 0.8261 (n=276) | 0.3506 (n=174) |

The 84 Qwen unparsed items split 44 gold-no / 40 gold-yes and 36 attribute / 25 existence /
23 relation; two are empty strings and the rest are descriptive prose that never says "yes" or
"no" (for example `"It is ambiguous whether the mountain is short or not. It can be both short
and tall."`).

**Procedure.** Recompute the same numbers from the freshly run B1 and B2 per-item dumps once
they land (they are the matched baselines for h+/h- anyway), using
`evaluation.classifiers.metrics._normalize_yes_no` directly on `responses.json`. Report
`n_total`, `n_unparsed`, `tp`, `fp`, `tn`, `fn`, `n_gold_no`, `n_gold_yes`, gold-no accuracy,
gold-yes accuracy, and yes rate, per model.

**What invalidates the design.** At an ~19% unparsed rate on Qwen: `tp + fp + tn + fn` does not
sum to 450, `accuracy_overall` is a ratio over a base that steering itself can move,
gold-yes accuracy of 0.3506 against gold-no accuracy of 0.8261 is exactly the imbalance the
spec's own AMBER assumption says would make the aggregate accuracy row unreadable as
discrimination, and `_flip_counts` (`rotation_strength.py:141-145`) skips any item whose
baseline or steered answer is unparsed — so h+ and h- for Qwen are computed over at most ~81% of
the set, with the excluded fraction free to change with beta.

**Report to Alex and stop before the grid.** Two facts to put in front of him and not resolve:
the design's prediction table gives a Qwen AMBER baseline of 74 where the pinned 450 on disk
gives 64.2, and a LLaVA baseline of 70 where disk gives 76.7. Whether the parse rate is
acceptable, and whether the table's baseline column should be read against these numbers, is his
call.

**Output.**
`evaluation/visual_reasoning_validation/results/2026-07-30/amber450_parse_rate_and_gold_split_baseline.json`
and a markdown table in the report.

---

## Check 8 — the POPE 600 that gets scored is the pinned 600 (gate)

**What it verifies.** That `--limit 200` per split selects exactly the ids in
`data/pope/pinned_eval_ids.json`.

**Why it is done this way.** `load_pope` takes `data_dir`, `split`, `limit` and **no**
`subset_ids` parameter (`src/dataset.py:146-150`), while `_benchmark_kwargs`
(`eval_runner.py:70-72`) forwards `subset_ids` for any benchmark present in the subset map. So
passing a subset file that contains a `pope` key raises `TypeError`. `pinned_eval_ids.json`
stores its ids under `splits`, not under `pope`, so it is not a valid `--subset_ids_file`
either. The deterministic first-200 per split is the mechanism, and its `selection` field says
so: `"deterministic first-N of data/pope/combined.json filtered by category/task==split; no
RNG."`

**Procedure.** The guard already written into `run_vti_pope_beta_grid.sh:57-70`, run as a
standalone step:

```python
import json
from src.dataset import combined_json_path
data = json.load(open(combined_json_path('pope')))
pin = json.load(open('data/pope/pinned_eval_ids.json'))
for split, ids in pin['splits'].items():
    ents = [e for e in data if e.get('category') == split or e.get('task') == split]
    assert [e['id'] for e in ents[:pin['limit_per_split']]] == ids, split
```

**Expected.** All three splits match, 200 ids each, 600 total.

**What invalidates the design.** A drift means the 600 items being scored are not the 600 the
spec names, and POPE cells are not comparable to the 2026-06-18 baselines or to each other
across arms.

---

## Check 9 — the AMBER and CHAIR pinned subsets still resolve (gate)

**What it verifies.** That the two subset files load to exactly the item counts and gold
composition the spec's Cells and Sample size sections state.

**Procedure.**

```python
from evaluation.benchmarks import load_amber_eval, load_chair_eval
import json
amber_ids = set(json.load(open('data/amber/pinned_amber_disc_450.json'))['amber'])
chair_ids = set(json.load(open('data/chair/pinned_chair_500.json'))['chair'])
a = load_amber_eval(task='discriminative', subset_ids=amber_ids)
c = load_chair_eval(subset_ids=chair_ids, prompt_override='Please Describe this image in detail.')
```

Assertions:

- `len(a) == 450`; the gold label counts are `{"no": 276, "yes": 174}`, matching `_meta.gold_counts`
  in the pinned file; the three question-type strata are 150 / 150 / 150
  (existence / attribute / relation, per `_meta.strata`).
- Every AMBER sample carries a non-empty `ground_truth` — the gold labels come from
  `data/amber/data/annotations.json` at load time (`src/dataset.py:283-289`), not from
  `combined.json`, so a missing annotation file yields 450 samples with empty gold and a scorer
  that silently skips all of them (`metrics.py:164-165`).
- `len(c) == 500`, 500 distinct `metadata.raw.coco_id`, none `None` — `score_chair_records`
  drops any record with a missing `coco_id` into `n_missing_image_id` and never scores it
  (`metrics.py:322-324`).
- Every sample's `question` on CHAIR is the verbatim prompt string.
- Every image loads: no `sample.image is None`.

**What invalidates the design.** Any count other than 450 / 500, any empty AMBER gold, or any
missing `coco_id` means the n per cell in the spec is not the n being scored.

---

## Check 10 — measured throughput before committing the grid (gate on launch, not on design)

**What it verifies.** The 370-invocation budget.

**Procedure.** Run one complete arm end to end on the target hardware —
LLaVA, `--layer_range all`, nd200, `--beta 0.5` — across all five benchmark invocations
(AMBER 450, POPE random / popular / adversarial at 200 each, CHAIR 500 at
`--chair_max_new_tokens 256`). Record wall clock per invocation, the samples/s the runner prints
every 50 items (`eval_runner.py:149-152`), and peak GPU memory from `nvidia-smi`.

**Reference points on disk.** POPE at 200 items ran at 0.60–0.75 samples/s on an A6000
(`evaluation/results/_logs/*.log`). CHAIR at a 256-token cap has never been timed in this repo.

**What this gates.** If the extrapolated total exceeds the RunAI allocation available, the
launch decision is Alex's — do not silently drop arms, betas, or sample sizes to fit.

**Output.** `evaluation/visual_reasoning_validation/results/2026-07-30/throughput_calibration_one_arm.json`.

---

## Environment prerequisites for the RunAI leg (verify, do not assume)

Not sanity checks on the design; failures here block the run rather than the reading.

- **Qwen2.5-VL-7B-Instruct weights on NFS.** `helper_scripts/runai/setup_vlm.sh:49` provisions
  `llava-hf/llava-1.5-7b-hf` only. Verify `Qwen/Qwen2.5-VL-7B-Instruct` under
  `HF_HOME=/home/datalake/romanus/hf_cache`; if absent, `snapshot_download` it in-pod before the
  Qwen job.
- **AMBER images and annotations on NFS.** `.gitignore:126-128` excludes `data/amber/*` except
  the pinned JSONs, so `data/amber/images/` and `data/amber/data/annotations.json` are not in
  `vti_repo.tar.gz`. Run `python data_scripts/download_amber.py` in-pod, or stage them by SFTP.
  Check 9 fails loudly if this is missed.
- **COCO val2014** is provisioned by `setup_vlm.sh` via `data_scripts/download_chair.py`; POPE
  and CHAIR both resolve against it.
- **The eight new `_meandiff_partition` direction directories** are under
  `experiment_artifacts/`, which `.gitignore:69` excludes only at `**/_act_cache/`, so they are
  tracked. Commit them before `git archive -o vti_repo.tar.gz HEAD`, or the RunAI job has no
  directions to load.
- **Results come back by SFTP.** `.gitignore:107` excludes `evaluation/results`, so nothing
  written in the pod arrives via git.

---

## Artifacts this plan produces

| Path | Contents |
|---|---|
| `experiment_artifacts/vti/meandiff_activation_cache_coverage_2026-07-30.json` | Check 1 per-model, per-block file presence and stack shapes |
| `experiment_artifacts/vti/meandiff_direction_shapes_and_layer_norms_2026-07-30.json` | Check 3 shapes, finiteness, per-layer L2 norms for all eight new sets |
| `evaluation/results/2026-07-30_smoke/_diagnostics/steering_plumbing_smoke_llava.json` | Check 4 hook counts, response-difference counts, recorded intervention config |
| `evaluation/results/2026-07-30/_diagnostics/step0_chair_token_cap_llava-1.5-7b-hf.json` | Check 6 caps 128 / 256 / 512, LLaVA |
| `evaluation/results/2026-07-30/_diagnostics/step0_chair_token_cap_qwen2.5-vl-7b-instruct.json` | Check 6 caps 128 / 256 / 512, Qwen |
| `evaluation/visual_reasoning_validation/results/2026-07-30/amber450_parse_rate_and_gold_split_baseline.json` | Check 7 counts per model |
| `evaluation/visual_reasoning_validation/results/2026-07-30/throughput_calibration_one_arm.json` | Check 10 wall clock and peak memory |
| `evaluation/visual_reasoning_validation/results/2026-07-30/sanity_checks_report.md` | All ten checks, pass / fail, with the numbers |

---

## What confirms the design can be run as written

All ten checks pass, and in particular:

- Checks 1, 2, 3: the mean-difference primitive is buildable from cache with zero forward
  passes, over four disjoint blocks, and is non-degenerate at every steered layer.
- Checks 4, 5: steering demonstrably reaches the model at exactly the requested layers, on one
  unsharded device, with the layer set and direction slug recorded in every result summary.
- Check 6: CHAIR metrics are flat between 256 and 512 tokens on both models, so the CHAIR rows
  of the prediction table measure hallucination and not caption truncation.
- Check 7: the parse rate on AMBER is high enough on both models that accuracy, the four
  confusion counts, and the h+/h- join cover the intended 450 items.
- Checks 8, 9: the scored item sets are the pinned 450 / 600 / 500.

Checks 6 and 7 are the two that can come back negative on the current evidence. Both go to Alex
with the numbers and no recommendation.

---

## Open questions

None that block these checks. The questions raised by the results of checks 6 and 7 are stated
in the main plan's Open questions and are Alex's to answer.
