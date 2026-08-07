# Expanded AMBER discriminative set under baseline, mean-difference, and VTI PCA steering
extraction_spec: extractions/08_05_26/steering_vector_validation_continuation.md

Date: 2026-08-05
Kind: extraction. One spec, one plan.
Status: ready to implement. No sanity check gates this plan — see that section for why.

---

## Question

What primitives does this produce, and for what later use?

Generated answers, per item, for both models on a newly drawn 1500-item AMBER discriminative
set, under no intervention and under textual steering by two direction sets built from the same
500 demos — the raw mean difference already used in the 2026-07-30 run, and the VTI PCA
(PC1 + mean) direction — across three steering coefficients and three layer windows per model.

Nothing is contrasted here. The per-configuration accuracy files and the per-item answer file
this run writes exist so that the matched-pair, dependent-sample comparison the extraction spec
describes can be computed later, once a design spec under `designs/` says what the comparison
is. The spec's own scope note sets those terms; this plan does not restate or extend them.

---

## Extraction spec reference

`extractions/08_05_26/steering_vector_validation_continuation.md`

- Field 1 fixes: direction sample size 500 only; AMBER only (POPE and CHAIR skipped); the
  1500-item expanded AMBER set with per-question-type counts of 500 each and per-question-type
  yes-ratios matching the whole benchmark; three direction conditions (none, mean-difference,
  VTI); both models; betas 0.2 / 0.5 / 0.9; the same three layer windows each model used on
  2026-07-30.
- Field 2 fixes: the 1500 is a strict superset of `data/amber/pinned_amber_disc_450.json`; it is
  disjoint from the 500 demos; every configuration is generated fresh over all 1500 items and no
  2026-07-30 response is carried forward; the independence unit is an image with a unique
  prompt.
- Field 3 fixes what must be computable afterwards: accuracy / precision / recall and related
  quantities per configuration, the same broken down by question type and by ground-truth label,
  and a per-item correctness record under a stable item id over the identical 1500 items in every
  configuration.
- The spec's `## Code gaps` section is empty. Four gaps found by reading the code during
  planning are written into **Code changes** below as implementation steps. None of them changes
  a cell, a metric, an item set, or a condition.

Grid settings inherited from the 2026-07-30 run and unchanged here, because field 1 says
"exactly as in the 7-30 run": greedy decoding (`do_sample=False` in every wrapper
`generate_*`), `--max_new_tokens 256`, `--amber_task discriminative`, Qwen
`--max_pixels 1003520`, intervention `vti_textual_additive_mlp`, direction dimension `all`.

---

## Cells

38 evaluation invocations: 19 per model, each over the identical 1500 items.

Layer indices are absolute decoder indices. LLaVA-1.5-7B has 32 decoder layers, Qwen2.5-VL-7B
has 28; both windows exclude each model's last layer and the all-layers arm includes it. These
are the same windows the 2026-07-30 cells used, read from the run-slug directory names under
`evaluation/results/2026-07-30/*/amber/` (`layers_all`, `layers_5_14`, `layers_20_29` for LLaVA;
`layers_all`, `layers_5_14`, `layers_15_24` for Qwen).

| Model (HF id) | Steering condition | `--layer_set` | Beta | Invocations |
|---|---|---|---|---|
| `llava-hf/llava-1.5-7b-hf` | none (`no_intervention`) | — | — | 1 |
| `llava-hf/llava-1.5-7b-hf` | mean difference, 500 demos | `all` (0–31), `5-14`, `20-29` | 0.2, 0.5, 0.9 | 9 |
| `llava-hf/llava-1.5-7b-hf` | VTI PCA (PC1 + mean), 500 demos | `all` (0–31), `5-14`, `20-29` | 0.2, 0.5, 0.9 | 9 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | none (`no_intervention`) | — | — | 1 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | mean difference, 500 demos | `all` (0–27), `5-14`, `15-24` | 0.2, 0.5, 0.9 | 9 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | VTI PCA (PC1 + mean), 500 demos | `all` (0–27), `5-14`, `15-24` | 0.2, 0.5, 0.9 | 9 |

38 invocations × 1500 items = **57,000 generations**. Zero forward passes for direction
extraction: both direction sets already exist on disk (see Data).

Result directory per cell, under a new run date so nothing on disk is touched:

```
evaluation/results/2026-08-05/{llava-1.5-7b-hf|qwen2.5-vl-7b-instruct}/amber/
    no_intervention/
    vti_textual_additive_mlp__b{beta}__dall__nd500__meandiff__layers_{all|5_14|20_29|15_24}/
    vti_textual_additive_mlp__b{beta}__dall__nd500__pc1_plus_mean__layers_{all|5_14|20_29|15_24}/
        responses.json
        metric_summary.json
```

The `meandiff` / `pc1_plus_mean` component is what keeps the two steering conditions from
writing to the same directory; producing it requires the code change described under Code
changes, gap 1. `no_intervention.config` is `{}`, so the baseline takes the bare-`{iv}` branch
at `eval_runner.py:331`.

---

## Data

### The expanded AMBER set — new file, built by a new script

**Path:** `data/amber/pinned_amber_disc_1500.json`, shaped like the existing pins
(`{"amber": [ids...], "_meta": {...}}`) so it passes straight to
`run_eval.py --subset_ids_file`. `run_evaluation` reads only `spec["amber"]`
(`eval_runner.py:210-218`); `_meta` is ignored by the runner and is provenance.

**Source pool.** `data/amber/combined.json` filtered to `task == "discriminative"` (14,216
items), joined to `data/amber/data/annotations.json` by raw question id for gold `truth` and
question type via `_amber_discriminative_qtype` (`src/dataset.py:226-250`). Every item resolves
an annotation — 0 misses. Pool composition read during planning, and it matches the table in the
spec exactly:

| question type | n | yes | no | distinct images |
|---|---|---|---|---|
| existence | 4924 | 0 | 4924 | 1004 |
| attribute | 7628 | 3814 | 3814 | 1004 |
| relation | 1664 | 975 | 689 | 1004 |

1004 distinct images overall; all 1004 image files are present under `data/amber/images/`
(0 missing).

**Target composition**, taken verbatim from the spec's agent-derived per-cell counts:

| question type | gold yes | gold no | total |
|---|---|---|---|
| existence | 0 | 500 | 500 |
| attribute | 250 | 250 | 500 |
| relation | 293 | 207 | 500 |
| **total** | **543** | **957** | **1500** |

**The pinned 450 is kept whole.** Its composition, read from disk: existence 150 no; attribute
82 yes / 68 no; relation 92 yes / 58 no. The 1050 added items are therefore 350 existence-no,
168 attribute-yes, 182 attribute-no, 201 relation-yes, 149 relation-no.

**Duplicate prompts.** The discriminative pool contains 10 (image, question text) pairs that
appear under two distinct item ids — e.g. `amber_disc_08383` and `amber_disc_08385` are both
"Is there a person in this image?" on `AMBER_156.jpg`. The spec's independence unit is an image
with a unique prompt, so the builder keeps at most one item per (image, question text): within
each such group it keeps the member already in the pinned 450 if there is one, otherwise the
lowest-sorted id. No duplicate group has more than one member in the 450, so this never
conflicts with the superset requirement. 14,206 items remain eligible.

**Draw, deterministic, seed 1234** (the seed the existing AMBER pin was drawn with):

1. Seed the selection with all 450 pinned ids and initialise a per-image usage counter from
   them.
2. Take the five (question type, gold) buckets in increasing order of
   `pool_size / items_still_needed`, i.e. scarcest first: relation-no, relation-yes,
   existence-no, attribute-no, attribute-yes.
3. Within a bucket, shuffle its eligible non-pinned candidates once with the seeded RNG, then
   fill in passes over increasing image-usage level: pass 0 takes only candidates whose image is
   so far unused, pass 1 candidates whose image has been used once, and so on, until the
   bucket's target is met. Increment the image counter on every pick.

Prototyped during planning against the real files; the result is exact and near-optimal on image
distinctness: all five bucket targets hit exactly, 1500 unique ids, **1004 distinct images** —
every image in the benchmark — with 515 images used once, 482 twice and 7 three times. The seven
triples are inherited from the pinned 450, which already used those images three times; 2 is the
floor for 1500 items over 1004 images, so no item outside the 450 pushes an image past it.
Overall 543 yes / 957 no, 36.2 percent yes.

**Item order at evaluation time is not the pin's order.** `load_amber` filters
`combined.json`'s own entry order by the id set (`src/dataset.py:269-275`), so every
configuration scores the identical 1500 items in the identical order regardless of how the pin
file is sorted. This is what satisfies the spec's "same items in the same order in every
configuration"; it needs no code.

### Steering directions — both already on disk, zero forward passes

Both sets were built from the same 500-demo block of `data/vti/demos_850_partition_s42.json`
(seed 42, `selection_policy: disjoint_partition`), over demos file `data/vti/demos_850.jsonl`
with content hash `ba05bd960cad0c18`. Their `ids_used` lists are element-for-element identical,
verified during planning for both models — so the demo sample of 500 is genuinely held constant
across the two steering conditions and only the reconstruction differs.

| Steering condition | Directory (per model, under `experiment_artifacts/vti/{model_short}/textual_v2/`) | `steer_reconstruction` in metadata |
|---|---|---|
| mean difference | `demos850_ba05bd96_all_nd500_s42_meandiff_partition` | `raw_mean_difference` |
| VTI PCA | `demos850_ba05bd96_all_nd500_s42_r2_partition` | `live_pc1_plus_mean` |

Shapes verified: `(32, 4096)` float32 for LLaVA and `(28, 3584)` float32 for Qwen in all four
`directions.npz`, which is `(wrapper.num_layers, wrapper.hidden_dim)` and passes the assertion in
`VTITextualIntervention.ensure_directions` (`intervention.py:149-160`).

Resolved facts that make the VTI PCA set usable through the same path as the mean-difference set,
with no new extraction code:

- `directions.npz` in the `_r2_partition` directory already stores the applied steering vector,
  `PC1 + mean` reshaped and sliced to the decoder layers — it is `direction_flat` from
  `_live_pca_fit` (`directions_v2.py:252-270`), not the raw PCA components, which are stored
  separately in `components.npz` and are not read by the steering path.
- Both direction families are loaded by the same `load_textual_v2_directions(cache_dir)`
  (`directions_v2.py:358`), which reads `directions.npz` + `metadata.json` and ignores
  `components.npz`. `IMPLEMENTATION.md:683-685` records this: the PCA `*_r2_partition`
  directories are loadable through `--directions_dir` in the same on-disk format; the 2026-07-30
  grid simply used meandiff slugs.
- `steer` L2-normalises each layer's direction slice before applying it (`steer.py:43`), so the
  two conditions apply perturbations of identical magnitude at identical beta and differ only in
  per-layer orientation. Beta remains the only magnitude knob.

Selection note: the demos_v2 sets (`demosv2_9a44f4af_all_nd500_s42_r2_prefix`) also carry
`steer_reconstruction: live_pc1_plus_mean`, but they are drawn from a different pool under
`shuffled_prefix` selection and would not hold the 500 demos constant against the
mean-difference arm. They are not used.

**Read-only.** The driver passes neither `--demos_path` nor `--vector_dimension`, so
`ensure_directions` takes the `directions_dir` branch and never calls
`compute_or_load_textual_directions_v2`; nothing is written under `experiment_artifacts/` and
`textual_v2/_act_cache/` is not touched. Verification records this rather than assuming it.

### Additivity, traced

- The new pin is a new path. Nothing in `src/`, `evaluation/`, `helper_scripts/` or
  `data_scripts/` globs or iterates `data/*/pinned_*`; every reference is an explicit path
  string. `pinned_amber_disc_450.json` is neither read for modification nor rewritten, and the
  builder refuses to write any path other than `data/amber/pinned_amber_disc_1500.json`.
- Results go under a new run date, `2026-08-05`. The 2026-07-30 tree is untouched.
- The result-directory naming change (Code changes, gap 1) maps `raw_mean_difference` to the
  literal `meandiff`, which is exactly what the current code emits, so every directory already on
  disk keeps the name the code would generate for it. Every existing `__meandiff__` cell lives
  under `evaluation/results/2026-07-30/`; a repo-wide search found none elsewhere.
- Direction directories, the demos file, the partition file and the activation cache are read
  only.

---

## Metrics

All named in plain English, all stated with the counts they are formed from, per the reporting
convention in `CLAUDE.md`. Positive class is `yes` throughout, matching the repo's AMBER scorer
(`evaluation/classifiers/metrics.py:208-209`, `positive_class: "yes"` / `metric_convention: "pope"`). Answers are parsed
by `_normalize_yes_no` (`metrics.py:10-27`). `p_yes_norm` and `answer_mass` appear nowhere.

### Per-item primitive

The generated response string per item, retained under the sample id in each cell's
`responses.json` (`eval_runner.py:143-154`), together with `ground_truth`, and
`metadata.category` (existence / attribute / relation) and `metadata.image_path` which the AMBER
loader already attaches (`src/dataset.py:294-305`). No new fields are needed on the record.

### Per configuration

For each of the 38 cells, from `metric_summary.json` and recomputed from `responses.json`:

- **tp, fp, tn, fn** — the four parsed yes/no outcome counts.
- **n_total, n_unparsed**, and `n_unparsed` split into **n_empty_response** (the generated string
  is empty) and **n_nonempty_unparsed** (a non-empty answer the parser could not resolve).
  The split matters because `eval_runner.py:137-142` catches a per-sample exception and stores
  `response = ""`, which is otherwise indistinguishable from a genuinely unparseable answer, and
  an item lost that way in one configuration but not another is exactly what a later matched-pair
  count would trip over.
- **accuracy** = (tp + tn) / n_total, unparsed answers counted incorrect, matching the scorer.
- **precision** = tp / (tp + fp); **recall** = tp / (tp + fn);
  **F1** = 2 · precision · recall / (precision + recall).
- **yes rate over parsed answers** = (tp + fp) / (n_total − n_unparsed), reported alongside the
  scorer's own **`yes_ratio` over all items** = (tp + fp) / n_total (`metrics.py:132`) under
  distinct column names, since the two differ whenever anything is unparsed.
- **accuracy on gold-no items** with **n_gold_no**, and **accuracy on gold-yes items** with
  **n_gold_yes** — emitted by the scorer as `neg_item_accuracy` / `pos_item_accuracy` /
  `n_neg_total` / `n_pos_total`.

### Per configuration and question type

The same list again for each of existence, attribute and relation. The scorer already computes
this: `score_amber_discriminative_records` returns `by_qtype`, keyed on the record's
`metadata.category`, each entry carrying accuracy, precision, recall, F1, `yes_ratio`,
`neg_item_accuracy`, `pos_item_accuracy` and the four counts
(`metrics.py:186-223`). The 2026-07-30 summaries on disk already have the three keys.

### Per item, per configuration

Parsed answer, correctness, and the empty/unparsed distinction for every one of the 1500 items in
every one of the 38 cells, under the stable AMBER item id. This is the record that makes a
discordant-pair count possible later without rereading 38 response files.

### What this plan deliberately does not compute

No count of flips against a baseline, no hallucinations-induced / hallucinations-removed table,
no confidence interval, no comparison against the 2026-07-30 cells. Each of those contrasts two
arms, which is the line the spec's scope note draws and the reason the metric files are not to be
read as evidence yet. `evaluation/steering_visual_reasoning_validation/build_result_tables.py`
computes all of them and is **not** run against this tree. The artifacts below are shaped so the
design spec that comes next can compute them from files rather than from GPU time.

---

## Artifacts

### The pinned item set

```
data/amber/pinned_amber_disc_1500.json
```

`_meta` records: `benchmark`, `task`, `seed`, `n`, `superset_of` (the 450 pin path and its
sha256), `dedupe_policy` ("at most one item per (image, question text); pinned member preferred,
else lowest id"), `strata` (per question type: available, drawn), `gold_counts_by_question_type`,
`gold_counts`, `n_distinct_images`, `items_per_image_histogram`, `source`, `drawn_at`.

### Evaluation cells

38 directories as laid out in Cells, each with `responses.json` (~950 KB) and
`metric_summary.json`. Plus one run manifest per model process, written before its first
invocation:

```
evaluation/results/2026-08-05/_analysis_steering_vector_validation_continuation/run_manifest_{model_short}.json
```

recording git commit, model id, `max_pixels`, `max_new_tokens`, the subset file and its sha256,
the beta and layer-set grids, both direction directories with the sha256 of each `directions.npz`
and `metadata.json` and each one's `steer_reconstruction` and `n_pairs`, and the launch
timestamp — because the baseline cell's `intervention_config` is `{}` and otherwise carries no
record of the settings it ran under.

### Tables

All under
`evaluation/results/2026-08-05/_analysis_steering_vector_validation_continuation/`, written by
the new script described in Code changes. Every one must run correctly against a partial grid,
skipping any cell without a `metric_summary.json`.

- `per_configuration_counts_and_accuracy.csv` — one row per cell. Columns:
  `model, steer_reconstruction, layer_window, beta, direction_sample_size, direction_slug,
  n_total, n_unparsed, n_empty_response, n_nonempty_unparsed, tp, fp, tn, fn, n_correct,
  accuracy, precision, recall, f1, yes_rate_over_parsed_answers, yes_ratio_over_all_items,
  accuracy_gold_no, n_gold_no, accuracy_gold_yes, n_gold_yes`. `steer_reconstruction` takes
  `none`, `raw_mean_difference` or `live_pc1_plus_mean` — the metadata's own vocabulary, not a
  new one; `layer_window` takes `none`, `all`, `5-14`, `20-29`, `15-24`.
- `per_configuration_counts_and_accuracy_by_question_type.csv` — the same columns plus
  `question_type`, three rows per cell.
- `per_item_answers_and_correctness.csv` — 57,000 rows, one per (cell, item). Columns:
  `model, steer_reconstruction, layer_window, beta, direction_sample_size, item_id,
  question_type, ground_truth, image_file, response_is_empty, parsed_answer, is_correct`.
  `parsed_answer` takes `yes`, `no` or `unparsed`.
- `coverage.json` — which of the 38 cells are complete and which are absent, each identified by
  `model`, `steer_reconstruction`, `layer_window` and `beta` spelled out; plus `n_complete` and
  `n_expected: 38`.
- `result_tables.json` — the same content machine-readable, `schema_version: 1`.

No plots. The spec asks for none, and a figure of these numbers is a reading of them.

### Verification reports

```
evaluation/results/2026-08-05/_analysis_steering_vector_validation_continuation/amber_1500_pin_verification.json
evaluation/results/2026-08-05/_analysis_steering_vector_validation_continuation/run_verification.json
```

Contents specified under Verification below.

**Naming rule for everything written to disk.** No filename, directory component, JSON key, CSV
column, log line or console banner carries a code for a phase, block, stage or cell. Each is
identified by the values that define it — model, steering condition, layer window, beta — spelled
out.

---

## Sanity checks that must pass first

**None.** Every premise this run rests on was settled by reading the repo during planning and is
recorded above as a resolved fact with its source: the annotation join covers all 14,216
discriminative items with no misses; all 1004 image files exist on disk; the ten duplicate
(image, prompt) pairs are enumerated and none has two members in the pinned 450, so the
independence unit and the superset requirement do not conflict; the target composition is exactly
achievable and was prototyped against the real files; both direction sets load through the same
loader at the shapes the intervention asserts, and were built from element-for-element identical
demo id lists; `steer` normalises each layer slice so the two conditions differ only in
orientation; and both models ran this identical pairing, on this card, with these generation
settings over AMBER on 2026-07-30 without a single out-of-memory line in either log. What remains
is environment and throughput, which is below and is not a sanity check.

The one behavioural fact worth carrying into the run rather than checking: on 2026-07-30 the Qwen
AMBER baseline over the 450 left 21 of 450 answers unparsed with `--max_pixels 1003520` enforced,
and LLaVA left 0. That is a measured property of the model, reported next to every accuracy
figure as `n_unparsed`, not a premise that can fail.

---

## Launch prerequisites

Environment and scheduling facts. Not sanity checks.

### Target: lambdab2, GPU 0, both models concurrent on the one card

Verified 2026-08-05: `nvidia-smi` reports GPU 0 = NVIDIA RTX A6000, 49140 MiB total, **48673 MiB
free, zero compute processes**; GPUs 1, 2 and 3 are each held at ~48.3 GB by other users. GPU 0
is the target, which is Alex's routing instruction for this run and overrides the standing note
in `IMPLEMENTATION.md` against scheduling long sweeps on lambdab2, exactly as the 2026-07-30 run
did.

- **Two concurrent processes, one per model**, both with `CUDA_VISIBLE_DEVICES=0`, staggered by
  five minutes so the two model loads do not coincide.
- **No `device_map` sharding, no CPU offload.** With one device visible there is nothing to shard
  across, but confirm it: `wrapper.load()` prints the model device (`src/model.py:131-144`) and it
  must be a single `cuda:0` for both processes with no "Model is on CPU" warning. Mixed-device
  tensors corrupt the forward hooks the steering depends on.
- **Launch under `nohup` or tmux.** A 2026-06-22 launch was lost to a dropped SSH tunnel. The
  runner is resume-safe (`--skip_if_exists` plus `responses.checkpoint.json`), but a lost session
  costs the in-flight cell.

### GPU memory

The strongest available evidence is the run itself: on 2026-07-30 these two models, on this card,
at these generation settings, with `--max_pixels 1003520` on Qwen, completed all 37 AMBER
configurations each while running concurrently, with zero out-of-memory or CUDA-error lines in
either driver log and zero babysitter restarts. RESEARCH_LOG 2026-07-31 records Qwen alone at
~16.6 GiB during that grid; RESEARCH_LOG 2026-07-29 records the pair at ~30,792 MiB resident on
GPU 0. Against 48,673 MiB free that leaves roughly 17.5 GiB of headroom.

Item count does not move the peak: generation is batch 1 and `eval_runner.py:147` calls
`cleanup_gpu()` after every sample. Image resolution does not either — LLaVA's tower is fixed at
336×336, and Qwen's input is bounded by `max_pixels 1003520` with `image_processor.size` pinned
in `load()` (`IMPLEMENTATION.md:216`), which is what prevents the ViT-attention blow-up on the
large AMBER photographs. The expanded set draws from all 1004 AMBER images, including one at
24 megapixels, and that cap is what makes them safe.

### Host memory and disk

`load_amber` decodes every sample's image eagerly (`Image.open(path).convert("RGB")`,
`src/dataset.py:129-132`) and holds it for the invocation. AMBER images average 2.8 megapixels,
so 1500 samples is roughly **12.7 GB resident per process**, about 25 GB for the two — against
479 GB available. Disk: 38 × ~950 KB of responses plus the tables is under 50 MB, against 743 GB
free on `/`.

### Wall clock

Basis: the measured final per-cell rates over the 450-item AMBER cells of the 2026-07-30 run,
both models concurrent on this card — LLaVA median 0.67 samples/s (min 0.64, max 0.93), Qwen
median 0.29 samples/s (min 0.22, max 0.58). Model load is ~1.5 min per invocation.

| | per invocation (1500 items) | × 19 invocations |
|---|---|---|
| LLaVA, sharing the card | ~37 min + load | **~12 h** |
| Qwen, sharing the card | ~86 min + load | ~28 h |

Qwen's tail accelerates once the LLaVA process exits at ~12 h and frees the card: at the observed
un-contended rate the remaining ten or eleven cells take roughly 8 h rather than 16. **Expected
combined wall clock is 20–22 hours**, i.e. one overnight plus a morning. The run goes as far as it
goes; nothing here is scheduled against a deadline.

---

## Execution order

Sequence only; changes no cell, metric, item set or condition.

Both processes run the same sequence and differ only in speed:

1. **The AMBER baseline** — `no_intervention`, one invocation. First, so the reference cell for
   every later comparison exists before anything else.
2. **The mean-difference block** — layer windows in the order all, then 5–14, then the late
   window; betas ascending 0.2 → 0.5 → 0.9 within each window. Nine invocations. This block runs
   first of the two steering conditions because it is the continuation of the arm already run at
   450 items, so the earliest complete cells are the ones that pair with existing work.
3. **The VTI PCA block** — same window and beta order. Nine invocations.

A cell counts as complete only when its `metric_summary.json` exists; `_run_one` writes it last
(`eval_runner.py:154-163`) and leaves `responses.checkpoint.json` behind on a mid-cell stop, so
a stop inside a block never produces a half-filled row and `--skip_if_exists` resumes cleanly.

---

## Code changes

Four gaps, found by reading the code. The spec's `## Code gaps` section is empty; these are
implementation steps, and none changes a cell, metric, item set or condition.

### Gap 1 — the result-directory suffix hardcodes `meandiff`

`evaluation/runners/eval_runner.py:316-321`:

```python
elif beta is not None and "beta" in cfg:
    if directions_dir is not None:
        iv_dir = (
            f"{iv_name}__b{beta}__d{cfg['dimension']}__nd{cfg['num_demos']}"
            f"__meandiff__layers_{cfg['layer_set_label']}"
        )
```

The literal `meandiff` is written regardless of which direction set was loaded, so the VTI PCA
cell at a given beta, sample size and layer window resolves to the same directory as the
mean-difference cell. With `--skip_if_exists` the second condition would be silently skipped and
its 1500 generations never run; without it, the first condition's responses would be overwritten.
This is the single change that makes the grid expressible at all.

Fix: derive the component from the direction metadata the intervention already exposes.
`intervention.config` carries `steer_reconstruction` from `metadata.json`
(`intervention.py:113-129`). Map it through an explicit table —
`{"raw_mean_difference": "meandiff", "live_pc1_plus_mean": "pc1_plus_mean"}` — and raise a
`ValueError` naming the directory and the unmapped value on anything else, rather than falling
back to a name that could collide. `raw_mean_difference` maps to the literal the code emits
today, so every directory under `evaluation/results/2026-07-30/` keeps the name the code would
generate for it.

Update `IMPLEMENTATION.md:1365-1366`, which currently documents the suffix as unconditionally
`__meandiff__`, in the same change.

### Gap 2 — no builder for a superset-constrained AMBER draw

`evaluation/chair_amber_diagnostics/draw_subsets.py:90-133` draws a flat 150 per question type
by simple random sample with no gold-label targets, no superset constraint and no image-spread
control, and writes to the 450 path with `--force`. It cannot produce this set and must not be
modified — it is the record of how the 450 was drawn.

**New — `data_scripts/draw_amber_discriminative_1500.py`** (CPU, no model). Implements the draw
under Data exactly:

```bash
python data_scripts/draw_amber_discriminative_1500.py \
    --seed 1234 \
    --pinned_450 data/amber/pinned_amber_disc_450.json \
    --out data/amber/pinned_amber_disc_1500.json
```

Reuses `combined_json_path`, `benchmark_data_dir`, `_load_amber_annotations` and
`_amber_discriminative_qtype` from `src.dataset`, exactly as `draw_subsets.py` does. Targets are
module constants named for what they are (`EXISTENCE_GOLD_NO`, `ATTRIBUTE_GOLD_YES`,
`ATTRIBUTE_GOLD_NO`, `RELATION_GOLD_YES`, `RELATION_GOLD_NO` = 500, 250, 250, 293, 207). Refuses
to overwrite an existing output without `--force` and refuses to write any path other than the
one above. Prints the realised composition, image histogram and superset check.

### Gap 3 — the 2026-07-30 driver hardcodes the mean-difference slug and the 450 pin

`evaluation/run_scripts/run_steering_visual_reasoning_validation.sh:69` pins
`data/amber/pinned_amber_disc_450.json` and lines 84-87 build the directions directory as
`…_nd${nd}_s42_meandiff_partition` with no way to name a second condition. Leave it in place — it
is the record of the 2026-07-30 run.

**New — `evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh`**, one model per
invocation so the two processes are independent. Structure follows the existing driver
(conda activation with `SKIP_CONDA_ACTIVATE` escape, `MODEL_SHORT` via
`src.model._normalize_model_name`, `--max_pixels` added only for Qwen2 model ids, run manifest
written before the first invocation, `print_comparison_table` after each completed block).

Env knobs and defaults: `RUN_DATE=2026-08-05`, `OUTPUT_DIR=evaluation/results`,
`BETAS="0.2 0.5 0.9"`, `LAYER_SETS` defaulting to `all 5-14 20-29` for LLaVA and
`all 5-14 15-24` for Qwen, `STEER_RECONSTRUCTIONS="raw_mean_difference live_pc1_plus_mean"`,
`MAX_NEW_TOKENS=256`, `MAX_PIXELS=1003520`, `CUDA_VISIBLE_DEVICES=0`,
`AMBER_SUBSET=data/amber/pinned_amber_disc_1500.json`. The directions directory for a condition
is `experiment_artifacts/vti/${MODEL_SHORT}/textual_v2/demos850_ba05bd96_all_nd500_s42_${stem}_partition`
with `stem` = `meandiff` for `raw_mean_difference` and `r2` for `live_pc1_plus_mean`; the script
exits non-zero if `directions.npz` is missing before running any cell.

The two invocation forms — `$MAXPX` is `--max_pixels 1003520` for Qwen and empty for LLaVA:

```bash
# baseline, once per model
python evaluation/run_eval.py --model "$MODEL" --benchmarks amber --amber_task discriminative \
  --interventions no_intervention \
  --subset_ids_file data/amber/pinned_amber_disc_1500.json \
  --max_new_tokens 256 --run_date 2026-08-05 --output_dir "$OUTPUT_DIR" --skip_if_exists $MAXPX

# one steered cell
python evaluation/run_eval.py --model "$MODEL" --benchmarks amber --amber_task discriminative \
  --interventions vti_textual_additive_mlp \
  --beta "$BETA" --layer_set "$LAYER_SET" --directions_dir "$DIR" \
  --subset_ids_file data/amber/pinned_amber_disc_1500.json \
  --max_new_tokens 256 --run_date 2026-08-05 --output_dir "$OUTPUT_DIR" --skip_if_exists $MAXPX
```

Console banners name the block in words, for example
`=== AMBER 1500 | mean-difference direction | late window (layers 20-29) | beta=0.5 ===`.

Launch:

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup bash \
  evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh llava-hf/llava-1.5-7b-hf \
  > logs/amber_expanded_steering_direction_grid_llava_2026-08-05.log 2>&1 &

sleep 300

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MAX_PIXELS=1003520 nohup bash \
  evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh Qwen/Qwen2.5-VL-7B-Instruct \
  > logs/amber_expanded_steering_direction_grid_qwen25_2026-08-05.log 2>&1 &
```

### Gap 4 — the existing table builder cannot see the new cells, and computes contrasts

`evaluation/steering_visual_reasoning_validation/build_result_tables.py:34-37` matches cell
directories with a regex that hardcodes `meandiff`, so every VTI PCA cell would be silently
dropped as unparseable; it also has `EXPECTED_CELL_COUNT = 370` and computes baseline-joined flip
counts and Wilson intervals, which are contrasts this plan does not produce. Leave it untouched
so the 2026-07-30 tree still builds.

**New — `evaluation/steering_vector_validation_continuation/build_per_configuration_and_per_item_tables.py`**:

```bash
python evaluation/steering_vector_validation_continuation/build_per_configuration_and_per_item_tables.py \
    --run_date 2026-08-05 --output_dir evaluation/results
```

Walks `evaluation/results/{run_date}/{model_short}/amber/{cell_dir}/`, skips any cell without a
`metric_summary.json`, and parses the cell directory name with a regex whose reconstruction
component is a captured alternation (`meandiff|pc1_plus_mean`) rather than a literal. For each
complete cell it recomputes the counts, the two unparsed sub-counts, the rates and the
per-gold-label accuracies from `responses.json` using `_normalize_yes_no` imported from
`evaluation.classifiers.metrics`, and cross-checks the recomputed per-gold-label figures and the
per-question-type breakdown against `neg_item_accuracy` / `pos_item_accuracy` / `n_neg_total` /
`n_pos_total` / `by_qtype` in the cell's own `metric_summary.json` — a mismatch is a hard error.
It reads `intervention_config.steer_reconstruction` and `intervention_config.direction_slug` from
the summary rather than inferring the condition from the directory name. Writes the five files
listed under Artifacts. It computes no quantity that joins two cells.

---

## Verification

Run and recorded; distinct from sanity checks, which gate an experiment's premises.

**Of the pinned set**, written to `amber_1500_pin_verification.json` by a new
`helper_scripts/verify_amber_discriminative_1500_pin.py`, following the existing `verify_*`
scripts. All gating:

- `load_amber(task="discriminative", subset_ids=<the 1500>)` returns exactly 1500 samples, and
  every sample has a non-null `image_pil`.
- The 450 pinned ids are all present in the 1500.
- Per (question type, gold) counts are exactly 500/0, 250/250 and 293/207; overall 543 yes /
  957 no.
- 1004 distinct images; no (image, question text) pair appears twice; items-per-image histogram
  is 515 / 482 / 7 at 1, 2 and 3, and every image used three times is one the 450 already used
  three times.
- sha256 of the pin file, recorded and echoed into both run manifests.

**Of the run**, written to `run_verification.json` after the grid finishes (or at any stopping
point):

- 38 cell directories each with `responses.json` and `metric_summary.json`, each with
  `n_total == 1500`.
- The set of sample ids is identical across all present `responses.json` files.
- Each steered cell's `intervention_config` records the expected `directions_dir`,
  `direction_slug`, `steer_reconstruction`, `layer_indices`, `layer_set_label`, `beta` and
  `num_demos == 500`.
- sha256 of all four `directions.npz` and `metadata.json` files taken before the first launch and
  again at the end: unchanged.
- `experiment_artifacts/vti/{model_short}/textual_v2/_act_cache/` file count and mtime digest
  before and after: unchanged, confirming zero forward passes into the cache.
- Count of `[error] sample` lines per cell, grepped from the two driver logs, recorded per cell
  and reconciled against `n_empty_response`.

---

## What confirms or falsifies

There is no hypothesis here and nothing to falsify. The extraction is complete and usable when:

- all 38 cells exist with 1500 scored items each, over an id set identical in every cell;
- the per-configuration, per-question-type and per-item tables build with no cross-check failure
  against the cells' own `metric_summary.json`;
- the pin verification passes every gating item above;
- the direction files and the activation cache are byte-identical before and after.

The extraction has failed in a way that must stop the run if the id sets differ between cells, if
any cell's `n_total` is not 1500, if a direction file's hash changes, or if the recomputed
per-gold-label numbers disagree with the scorer's. The count of items answered differently
between any two configurations is not a completion criterion — it falls where it falls, per the
spec, and it is not counted in this plan.

---

## Open questions

1. This plan writes the per-configuration and per-item files and deliberately stops short of any
   cell-to-cell contrast: no flip counts against the baseline, no confidence intervals, no
   per-item comparison against the 2026-07-30 cells on the 450 shared items, and no plots. That
   reading of the scope note may be tighter than intended, since field 3 asks for those to be
   *computable* rather than computed. Confirm that the design spec written after this data lands
   is where they come from, or say which of them should be produced here.
2. Within each model the mean-difference block runs before the VTI PCA block, so under a partial
   run the arm that pairs with the existing 450-item work completes first. If the VTI PCA arm is
   the one you would rather have complete at the twelve-hour mark, it is a reordering of the
   `STEER_RECONSTRUCTIONS` environment variable and changes nothing else.
3. The 2026-07-30 Qwen AMBER baseline left 21 of 450 answers unparsed with `--max_pixels
   1003520` enforced, against 0 for LLaVA; at that rate roughly 70 of the 1500 items will be
   unparsed in each Qwen cell. This plan reports it as counts and splits it into empty-response
   and non-empty-unparseable, which is all it can do without changing what is run. Whether that
   loss rate is acceptable for the matched-pair comparison the data is meant to support is a
   design question, and it is worth settling before the design spec fixes the comparison rather
   than after.
