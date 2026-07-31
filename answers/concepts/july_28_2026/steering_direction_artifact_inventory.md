# Inventory: what steering-direction artifacts exist, and what n=500 shuffled control would cost

Date: 2026-07-28
Question: "Remind me what exists under `experiment_artifacts/` — I believe we have activations
for randomly shuffled image examples but only at n=200. Correct me. What is needed to extract
the same at n=500?"

Factual read of files and source. No interpretation, no ranking of candidate experiments.
Verified by direct inspection on 2026-07-28; file counts from `ls`, code claims cited to
`file:line`.

**Filing note:** this belongs under `answers/vti/july_28_2026/`, but writes to `answers/vti/**`
were denied by the permission layer on 2026-07-28 despite `settings.json` allowing
`Write(answers/**)`. Parked here; move it when that is resolved.

---

## 1. Your recollection is correct

The shuffled-image control exists at **n=200 only**, on **the two primary models only**.

```
experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control/
    all_nd200/{directions.npz, components.npz, metadata.json}
    _act_cache/                         400 files, 132 MB
    shuffled_control_sanity_report_llava-1.5-7b-hf.md
experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/
    all_nd200/{directions.npz, components.npz, metadata.json}
    _act_cache/                          400 files,  83 MB
    shuffled_control_sanity_report_qwen2.5-vl-7b-instruct.md
experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json
```

Act cache composition, confirmed by suffix count: 200 distinct demo ids × 2 variants
(`vl_v2_all`, `vl_v2_value`). Nothing for `qwen2-vl-7b-instruct` or `qwen-vl-chat`. The
derangement file is shared across both models — one seed-1234 derangement over the 200 ids,
`n_fixed_points: 0`.

The sanity reports cover derangement validity, caption-unchanged, id/order match, direction
shape, forward count, and per-layer norms. The extraction script's docstring states explicitly
that no cosine or behavioural comparison was performed
(`evaluation/run_scripts/extract_shuffled_control_directions.py:8`).

## 2. Two things that are not in your recollection and change the picture

### 2.1 The truthful directions already span n

Both primary models already have **20 direction sets each** — 5 dimensions × 4 sample sizes:

```
experiment_artifacts/vti/{llava-1.5-7b-hf, qwen2.5-vl-7b-instruct}/textual_v2/
    demosv2_9a44f4af_{all,existence,attribute,relation,counting}_nd{50,100,200,500}_s42_r2_prefix/
```

`qwen2-vl-7b-instruct` has only `all_nd200` and `all_nd500`. So n=50 / 100 / 200 / 500 for the
truthful direction is already on disk and needs no new GPU work.

### 2.2 But they are nested prefixes, not independent samples

`select_prefix_demos` (`evaluation/interventions/vti/directions_v2.py:119-149`) walks a fixed
master order (seed 42) and takes the **first** `num_demos` ids valid for the dimension. So

$$\text{ids}(nd50) \subset \text{ids}(nd100) \subset \text{ids}(nd200) \subset \text{ids}(nd500)$$

Consequence, stated mechanically: any cosine between two of these directions is a
**same-sample** comparison in the sense of
`answers/concepts/july_28_2026/noise_floor_and_disattenuation.md` §5 — the smaller set's demos
are entirely contained in the larger set's, so the two estimation errors are positively
correlated rather than independent. The formula $\sqrt{r_A r_B}$ derived for independent errors
does not apply to these pairs, and cross-fitting is not possible between them because there is
no disjoint fold.

Two independent estimates cannot be obtained by picking two different `nd` values from what is
on disk. That is a property of the selection policy, not of the data.

## 3. The activation cache is the important asset

`ActivationCache._path` (`src/extraction.py:65-66`) keys on `sample_{sample_id}_{suffix}.npz` —
per-demo, per-variant. **The key does not contain n.** `ensure_variant_activation`
(`directions_v2.py:187-212`) returns the cached stack on hit and only runs a forward pass on
miss.

Confirmed cache contents, `textual_v2/_act_cache`, per primary model:

| | count |
|---|---|
| distinct demo ids | 500 |
| variants per id | 6 (`value`, `all`, `existence`, `attribute`, `relation`, `counting`) |
| total files | 3000 |
| size | 622 MB (Qwen2.5-VL) / 985 MB (LLaVA) |

So the per-demo last-token activation stack for **all 500 demos and all 6 caption variants is
already on disk** for both primary models. Any direction over any subset of those 500 demos —
any split, any subsample, any n, any number of resamples — is a numpy operation over cached
files feeding `obtain_textual_vti_v2_from_stacks(h_stacks, v_stacks, rank=RANK)`. Zero forward
passes, no GPU.

This is a factual statement about what the cache permits, not a recommendation about what to run.

## 4. What n=500 shuffled control actually costs

**It is not a flag.** `extract_shuffled_control_directions.py` exposes `--model`,
`--demos_path`, `--deployed_slug`, `--derangement_seed`, `--max_pixels`, `--force_recompute`.
There is no `--n`. Four constants in `evaluation/interventions/vti/shuffled_control.py` fix
n=200:

| Line | Constant / function | Fixed value |
|---|---|---|
| 49 | `DEPLOYED_SLUG` | `demosv2_9a44f4af_all_nd200_s42_r2_prefix` |
| 51 | `DERANGEMENT_FILENAME` | `shuffled_control_image_derangement_nd200_s1234.json` |
| 53 | `NUM_DEMOS` | `200` |
| 66-67 | `shuffled_direction_dir()` | `.../shuffled_control/all_nd200` |

`--deployed_slug` could be pointed at the nd500 direction, but line 191-192 then raises:

```python
if len(ids_used) != NUM_DEMOS:
    raise RuntimeError(f"Expected {NUM_DEMOS} ids_used, got {len(ids_used)}")
```

So n=500 requires parameterising `NUM_DEMOS`, the output directory, and the derangement
filename. Small change, but a change.

### 4.1 The cache hazard — read this before running anything

`shuffled_act_cache_dir()` (`shuffled_control.py:62-63`) returns **one directory per model with
no n in the path**. The module docstring (lines 5-8) explains why this matters: activation cache
keys omit the image, so a cache entry's *contents* depend on which derangement was in force when
it was written, while its *key* does not record that.

A derangement over 500 ids will not agree with the existing derangement over the 200-id subset,
so all 400 existing entries become stale — same keys, wrong images. The code already defends
against this in the only two ways available:

- lines 207-211: refuse to run if the shuffled cache is non-empty
- lines 212-214: `--force_recompute` **deletes every `sample_*.npz`** in that directory

Both behaviours are correct. The consequence is that running n=500 through the current code path
with `--force_recompute` destroys the n=200 activations — 400 files and 400 forward passes per
model — and the n=200 direction becomes unreproducible without redoing them. Parameterising the
cache directory by n (or by derangement id) before any n=500 run avoids this entirely.

### 4.2 Forward-pass cost

Because a new derangement invalidates every existing entry, the count is the full set, not the
increment:

| | forwards per model |
|---|---|
| n=500 shuffled control, new derangement | 500 ids × 2 variants = **1000** |
| naive "just the extra 300" assumption | 600 — **wrong**, would silently reuse stale stacks |

For reference the existing n=200 run was 400 forwards per model
(`checks["expected_forwards"] = NUM_DEMOS * 2`, line 341).

Note the asymmetry with §3: the truthful direction needs **no** new forwards at any n, because
its cache is keyed by demo id and the image never changes. The shuffled control needs a full
re-extraction for every new derangement, because the image is exactly what changes and the key
does not record it.

## 5. Workflow state

- `designs/` is **empty**. No design spec exists yet, so `/plan` has nothing to expand and the
  pre-write hook will block any write into `implementation_plans/`.
- `implementation_plans/` contains `7-21-26/`, `7-22-26/`, `old/` — all predating the gate.
- `templates/design_template.md` is the file to copy to `designs/<exp_id>_design.md`.

Per `WORKFLOW.md`, the loop is: design spec → `/examine-design <exp_id>` → `/plan <exp_id>` →
Cursor implements → `analysis/<run_id>_reading.md` → `/examine-results <run_id>`.

The design template requires: the question (one sentence), at least two competing explanations
stated as mechanisms, a prediction table with one row per condition and one column per
explanation, cells (≤3), item sets with paths for anything already on disk, a primary
measurement that is a primitive, n per cell with justification, a written prediction, an
abandonment criterion, and assumptions to check first.

## Related

- `answers/concepts/july_28_2026/noise_floor_and_disattenuation.md` — §5 on correlated errors and
  cross-fitting; §6 on what a control must be matched on.
- `answers/concepts/july_28_2026/cosine_vs_sample_size_mean_difference_directions.md` — §9 on
  what comparison separates reliability from alignment.
