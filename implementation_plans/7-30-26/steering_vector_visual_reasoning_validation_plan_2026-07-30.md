# Mean-difference textual steering: does it improve visual reasoning on AMBER, POPE, and CHAIR
design_spec: designs/07_30_26/steering_vector_visual_reasoning_validation.md

Date: 2026-07-30
Status: ready to implement. Gated on
`implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_sanity_checks_plan_2026-07-30.md`,
whose checks 6 and 7 report to Alex before the grid is launched.

---

## Question

Does textual steering with a steering vector computed solely as the mean difference between
paired truthful and hallucinated captions — no PCA, no rotation — improve the visual reasoning
capability of LLaVA-1.5-7B and Qwen2.5-VL-7B, as measured on two discriminative benchmarks
(AMBER discriminative 450, POPE 600) and one generative benchmark (CHAIR 500)?

The design distinguishes three explanations. A: steering makes the model answer more truthfully,
so discriminative accuracy rises and CHAIR is unchanged. B: steering causally improves visual
reasoning, so discriminative accuracy rises and the CHAIR score falls. C: steering does nothing
on either.

## Design spec reference

`designs/07_30_26/steering_vector_visual_reasoning_validation.md`.

Everything below expands that file. The factor grid, the layer windows, the beta grid, the item
sets, the direction sample sizes, the `additive` / `mlp` fixing, the `all` category slice, the
one required primitive, the six code gaps, and the primary measurement lineage are all taken
from it verbatim. Nothing here adds a cell, a condition, a metric, a model, a benchmark, an item
set, a seed, or a hyperparameter.

The spec declares one primitive under **Primitives this design requires** — raw mean-difference
textual directions — and states that the plan produces it under this spec with no separate
extraction spec. That is why this plan carries a single `design_spec:` declaration and includes
the extraction as step 1.

---

## Cells

Full factorial, as the spec's Cells section states. Cell shorthand S1–S6, B1, B2 is the spec's.

| Cell | Model | Model string | Decoder layers | Layer set | `--layer_range` |
|---|---|---|---|---|---|
| S1 | LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | 32 (0–31) | all | `all` |
| S2 | LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | 32 (0–31) | early-middle | `5-14` |
| S3 | LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | 32 (0–31) | mid-late | `20-29` |
| S4 | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | 28 (0–27) | all | `all` |
| S5 | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | 28 (0–27) | early-middle | `5-14` |
| S6 | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | 28 (0–27) | mid-late | `15-24` |
| B1 | LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | — | none | — |
| B2 | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | — | none | — |

Layer counts verified: `directions.npz` in the existing partition sets records `num_layers=32,
hidden_dim=4096` for `llava-1.5-7b-hf` and `num_layers=28, hidden_dim=3584` for
`qwen2.5-vl-7b-instruct`. Both windows exclude the last layer (31 and 27 respectively); the
all-layers arm includes it, as the spec permits. Indices are absolute decoder-layer indices and
are passed to `vti_hook_ctx(..., layer_indices=...)` (`hooks.py:62-98`), which validates the
range and raises on anything outside `0..num_layers-1`.

Held fixed across every steered arm, from the spec:

- Intervention: `vti_textual_additive_mlp` — selected by naming it in `--interventions`, never
  by relying on `VTITextualIntervention`'s `variant="uniform_rotation"` default
  (`intervention.py:38`). The registry key is built from `STEER_VARIANTS × HOOK_SITES`
  (`evaluation/interventions/__init__.py:29-33`).
- Direction construction: raw mean difference, no PCA.
- Direction category slice: `all`.
- Decoding: greedy (`do_sample=False`, `use_cache=True`, hard-coded in every wrapper's
  `generate_vl`), so no seed is a factor.

Crossed factors: direction sample size ∈ {50, 100, 200, 500}; beta ∈ {0.2, 0.5, 0.9}.

Run accounting, matching the spec:

- Steered arms: 2 models × 3 layer sets × 4 sample sizes × 3 betas = **72**
- Benchmark invocations per arm: 5 (AMBER ×1, POPE ×3 splits, CHAIR ×1) — POPE is three
  invocations because `run_eval.py:33` takes one `--pope_split` and `eval_runner.py:284` keys
  the output `pope_{split}`
- Steered invocations: **360**; baseline invocations: 2 × 5 = **10**; **total 370**

---

## Data

### Item sets — all four already exist on disk; none is constructed

| Set | Path | n | How it is selected |
|---|---|---|---|
| AMBER discriminative 450 | `data/amber/pinned_amber_disc_450.json` (key `amber`) | 450 | `--subset_ids_file` + `--amber_task discriminative`; `load_amber(subset_ids=…)` filters before image load (`src/dataset.py:270-273`) |
| POPE 600 | `data/pope/pinned_eval_ids.json` (key `splits`) | 200 × 3 | `--pope_split {random,popular,adversarial} --limit 200`; the pinned file records the selection as the deterministic first-200 per split, verified by sanity check 8 |
| CHAIR 500 | `data/chair/pinned_chair_500.json` (key `chair`) | 500 | `--subset_ids_file`; `load_chair(subset_ids=…)` filters before image load (`src/dataset.py:325-329`) |
| Demo blocks for the directions | `data/vti/demos_850_partition_s42.json` over `data/vti/demos_850.jsonl` | 50 / 100 / 200 / 500 | `blocks[str(n)]`; `selection_policy: disjoint_partition`, seed 42, content hash `ba05bd960cad0c18` |

POPE cannot use `--subset_ids_file`: `load_pope` has no `subset_ids` parameter
(`src/dataset.py:146-150`) while `_benchmark_kwargs` would forward one for any benchmark in the
subset map (`eval_runner.py:70-72`), raising `TypeError`. `--limit 200` is the mechanism, and
sanity check 8 is what makes it safe.

AMBER gold labels are joined at load time from `data/amber/data/annotations.json`
(`src/dataset.py:283-289`), not from `combined.json`; without that file every AMBER item loads
with empty gold and the scorer skips all 450 silently (`metrics.py:164-165`). Sanity check 9
catches it.

### The one primitive this plan produces: raw mean-difference textual directions

The spec's Primitives section. Zero forward passes — everything it reads is already cached.

**What is computed.** For each (model, nd ∈ {50, 100, 200, 500}), over the ids in
`partition["blocks"][str(nd)]`, at the `all` category slice:

```
D = (1/n) * Σ_i ( stack_value_i − stack_all_i )        # (num_layers+1, hidden_dim)
directions = D[1:]                                     # (num_layers, hidden_dim) float32
```

No centering, no SVD, no component selection. Diff polarity is `value_minus_h_value` (clean −
hallucinated), inherited from `DIFF_POLARITY` (`directions_v2.py:41`), so no sign-alignment step
is needed. Row 0 of `D` is the embedding row and is dropped by the `[1:]` slice, matching what
the hook consumer expects (`directions_v2.py:432`).

**Inputs, all on disk.** `experiment_artifacts/vti/{model_short}/textual_v2/_act_cache/`, 5100
`.npz` files per model (850 demos × 6 variants: `value` plus the five entries of `DIMENSIONS`),
verified present for both models. File naming is
`sample_{demo_id}_{suffix}.npz` with `suffix = variant_suffix(variant) = f"vl_v2_{variant}"`
(`ActivationCache._path`, `src/extraction.py:65-66`; `directions_v2.py:188-190`). Read with
`ActivationCache.load` only — never `.save`, never a wrapper, never a forward pass. This is the
same read-only pattern `perlayer_pca.load_stack_readonly` uses
(`evaluation/interventions/vti/perlayer_pca.py:114-122`).

**Where it is written.** Eight new sibling directories, one per (model, nd):

```
experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition/
experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition/
```

Each holds `directions.npz` and `metadata.json`. No `components.npz` — there are no components.
The slug is exactly the one the spec names.

**Additivity trace.** Additive means reachable, not merely present, so the chain:

- `partition_slug` (`directions_partition.py:141-159`) is not modified, so all 40 existing
  `demos850_ba05bd96_*_r2_partition` directories keep their names and stay reachable.
- The new slug builder lives in a new module and shares no constant with the existing ones;
  `textual_v2_slug`, `partition_slug`, `perlayer_partition_cache_dir` and
  `shuffled_demos850_direction_dir` are untouched.
- `data/vti/demos_850.jsonl` and `data/vti/demos_850_partition_s42.json` are opened read-only,
  so `content_hash_sha256_16` stays `ba05bd960cad0c18` and every existing `metadata.json` that
  records it still validates.
- `_act_cache` is read with `ActivationCache.load` and never written, so no `.npz` is created,
  replaced, or touched. `ActivationCache.__init__` calls `mkdir(exist_ok=True)` on a directory
  that already exists; nothing else.
- Evaluation output goes to a fresh `--run_date` directory under `evaluation/results/`, and the
  new `iv_dir` suffixes appear only when the new flags are passed, so the 2026-06-18,
  2026-06-22, 2026-07-02 and 2026-07-13 result trees are untouched.
- The only edit to an existing output path is `step0_chair_token_cap.py`'s filename, which gains
  `_{model_short}`; the 2026-06-22 file keeps its name and no code reads it.

Verification of this trace is a step, not an assertion: hash every file under the 40 existing
`_r2_partition` directories before and after, and diff the manifests.

**How the direction is consumed, and what that means for magnitude.** `steer` L2-normalises the
direction per layer before applying it (`steer.py:43`), so the per-layer magnitude profile of
the raw mean difference is discarded at application time; only its per-layer orientation reaches
the model, and beta is the sole magnitude knob. Under `variant="additive"` the operation is
`x + beta * d̂` (`steer.py:44-45`). This is identical to how the existing PCA directions are
consumed.

---

## Code to write

Every item is new or an explicit edit; nothing below exists today. Items 1a, 1b, 2, 3, 4, 5
close the spec's Code gaps in order.

### 1a. New module — `evaluation/interventions/vti/mean_diff_partition.py`

No mean-difference extractor exists anywhere in the tree.
`obtain_textual_vti_v2_from_stacks` (`directions_v2.py:277-323`) forms the per-demo diffs and
immediately fits PCA (`_live_pca_fit`, line 302), returning PC1 + mean.

CPU-only. Imports and reuses, without editing them:

```python
from .directions_partition import (
    PARTITION_SEED, PARTITION_SIZES, SELECTION_POLICY,
    build_or_load_partition, check_partition_integrity, select_block_demos,
)
from .directions_v2 import (
    DIFF_POLARITY, DIMENSIONS, TOKEN_POLICY, _git_commit, _stack_from_act_dict,
    act_cache_dir, demos_content_hash, load_demos_v2_rows,
    load_textual_v2_directions, save_textual_v2_directions, variant_suffix,
)
from .perlayer_pca import EXPECTED_DECODER_SHAPES   # {"llava-1.5-7b-hf": (32,4096), "qwen2.5-vl-7b-instruct": (28,3584)}
```

Public surface:

```python
MEANDIFF_SLUG_SUFFIX = "meandiff_partition"
STEER_RECONSTRUCTION = "mean_of_paired_diffs"       # deliberately not "live_pc1_plus_mean"
CONSTRUCTION = "raw_mean_difference_no_pca"

def meandiff_partition_slug(demos_hash: str, dimension: str, num_demos: int,
                            seed: int = 42) -> str:
    """demos850_{hash[:8]}_{dimension}_nd{n}_s{seed}_meandiff_partition"""

def meandiff_partition_cache_dir(model_short: str, slug: str) -> Path:
    """experiment_artifacts/vti/{model_short}/textual_v2/{slug}"""

def obtain_mean_diff_from_cache(
    cache: ActivationCache,
    demo_ids: Sequence[str],
    dimension: str,
    expected_rows: int,
    hidden_dim: int,
) -> Tuple[np.ndarray, dict]:
    """Streaming mean of (value − dimension) stacks.

    Accumulates in float64, one demo at a time, so nd500 never holds more than two
    (rows, hidden) arrays at once. Returns the (rows, hidden) float64 mean plus a diag
    dict carrying n_pairs, per_layer_norm_of_mean_diff (length rows), and
    mean_over_demos_of_per_layer_diff_norm (length rows).
    """

def compute_or_load_meandiff_partition_directions(
    model_short: str,
    *,
    dimension: str = "all",
    num_demos: int,
    seed: int = PARTITION_SEED,
    demos_path: Optional[Path] = None,
    partition_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, dict]:
    """(num_layers, hidden_dim) float32 + metadata. Loads from disk if present."""
```

Behaviour of `compute_or_load_meandiff_partition_directions`:

1. Resolve `demos_path` / `partition_path` from `vti_demos_850_path()` /
   `vti_demos_850_partition_path()`.
2. `build_or_load_partition(...)` then `check_partition_integrity(...)` — raise on failure.
3. `block_ids = part["blocks"][str(num_demos)]`; `select_block_demos(rows_by_id, block_ids,
   dimension)` to confirm every id has both a `value` and an `h_values[dimension]` caption.
4. `cache = ActivationCache(str(act_cache_dir(model_short)))`; for each id call
   `cache.load(demo_id, suffix=variant_suffix(v))` for `v in ("value", dimension)`. A miss
   raises `FileNotFoundError` — do not fall back to a forward pass.
5. Reduce with `obtain_mean_diff_from_cache`; slice `[1:]`; cast float32.
6. Assert the shape equals `EXPECTED_DECODER_SHAPES[model_short]`. No wrapper is constructed.
7. `save_textual_v2_directions(directions, meta, cache_dir)` with `components=None` and
   `pca_mean=None`, so only `directions.npz` and `metadata.json` are written.

`metadata.json` fields: `demos_path`, `demos_file`, `content_hash_sha256_16`, `partition_file`,
`partition_content_hash`, `block_size`, `dimension`, `num_demos_requested`, `n_pairs`,
`ids_used` (the ordered block ids), `skipped_ids` (empty), `question`, `token_policy`,
`diff_polarity`, `construction` = `CONSTRUCTION`, `steer_reconstruction` =
`STEER_RECONSTRUCTION`, `selection_policy`, `seed`, `direction_layer_norms`,
`mean_diff_layer_norms_mean_over_demos`, `model_short`, `slug`, `act_cache_dir`,
`forward_passes` = 0, `date`, `git_commit`. No `rank`, no `steer_component`, no
`explained_variance_ratio`, no `pc1_*`, no `pc2_*` — those fields do not exist for this object
and their absence is what distinguishes it from a `_r2_partition` set on inspection.

### 1b. New run script — `evaluation/run_scripts/extract_demos850_meandiff_directions.py`

Modelled on `extract_demos850_partition_directions.py` but with no model load and no
`--max_pixels` (there are no forward passes, so the Qwen visual budget is irrelevant to this
step; the cached activations it reads were produced at `max_pixels=1003520`, recorded in the
existing `_r2_partition` metadata, and are not regenerated).

```
python evaluation/run_scripts/extract_demos850_meandiff_directions.py \
  --model llava-hf/llava-1.5-7b-hf \
  --dimension all --num_demos 50 100 200 500 \
  --demos_path data/vti/demos_850.jsonl \
  --partition_path data/vti/demos_850_partition_s42.json \
  --seed 42
```

Arguments: `--model` (required, mapped through `_normalize_model_name`), `--dimension`
(default `all`, choices `DIMENSIONS`), `--num_demos` (nargs +, default `PARTITION_SIZES`),
`--demos_path`, `--partition_path`, `--seed`, `--force_recompute`.

Takes the same `fcntl.flock` lock on
`experiment_artifacts/vti/{model_short}/textual_v2/.extract_demos850_lock_{model_short}` that
`extract_demos850_partition_directions.py:89-102` takes, so it cannot interleave with an
extractor that does write to the cache.

Writes a run manifest to
`experiment_artifacts/vti/{model_short}/textual_v2/demos850_meandiff_run_meta_{model_short}_2026-07-30.json`
recording model, hashes, block sizes, output paths, per-set sha256 of `directions.npz`,
`forward_passes: 0`, and wall clock.

Run once per model. Both runs are CPU-only and take minutes.

### 1c. New verification script — `helper_scripts/verify_meandiff_extraction.py`

Modelled on `helper_scripts/verify_demos850_partition_extraction.py`. Two jobs:

- **Additivity.** sha256 every file under the 40 existing
  `experiment_artifacts/vti/*/textual_v2/demos850_ba05bd96_*_r2_partition/` directories, plus
  every file under both `_act_cache` directories (name, size, mtime, and sha256 of a fixed
  sample of 50 files per model), snapshotted **before** the extraction and compared **after**.
  Any difference fails.
- **New artifacts.** Sanity check 3 of the gating plan: shapes, finiteness, per-layer L2 norms,
  `n_pairs == block size`, `ids_used == partition["blocks"][str(n)]`, absence of
  `components.npz`.

Output: `experiment_artifacts/vti/meandiff_extraction_verification_2026-07-30.md` plus the two
JSON manifests named in the gating plan.

### 2. Edit — layer window, end to end

`vti_hook_ctx` already accepts `layer_indices` (`hooks.py:62`, validated at 88-98).
`VTITextualIntervention` has no such parameter and forwards none (`intervention.py:151-160`);
`run_eval.py` exposes no flag. This blocks S2, S3, S5, S6.

**`evaluation/interventions/vti/intervention.py`** — `VTITextualIntervention.__init__` gains four
keyword-only parameters, all defaulting to `None` so existing callers are unchanged:

```python
layer_indices: Optional[Sequence[int]] = None,
layer_set_name: Optional[str] = None,
direction_slug: Optional[str] = None,
direction_construction: Optional[str] = None,
```

`generate` passes `layer_indices=self._layer_indices` into the `vti_hook_ctx(...)` call at
`intervention.py:151`. `config` gains the four values under the keys `layer_indices`,
`layer_set`, `direction_slug`, `direction_construction`, each included only when not `None`, so
the config dicts of existing runs are byte-comparable.

**`evaluation/run_eval.py`** — two new arguments:

```python
p.add_argument("--layer_range", default=None,
               help="Decoder layers to steer: 'all', or an inclusive absolute "
                    "range 'START-END' (e.g. '5-14'). Default None = all layers, "
                    "the existing behaviour.")
p.add_argument("--direction_dir", default=None,
               help="Directory holding directions.npz + metadata.json for a "
                    "pre-extracted textual direction set. Bypasses PCA extraction; "
                    "the array is preloaded and passed to the intervention.")
```

`--layer_range` parsing lives in `run_eval.py` as a module-level helper
`parse_layer_range(spec: str, num_layers: int) -> Tuple[Optional[List[int]], str]`, returning
`(None, "all")` for `"all"` and `(list(range(a, b+1)), spec)` for `"a-b"`, raising `ValueError`
on anything else or on `a > b`. `num_layers` is not known until the wrapper loads, so the parse
of the range string happens in `run_evaluation` after `create_wrapper(...).load()`; out-of-range
indices are rejected by `vti_hook_ctx` itself (`hooks.py:92-96`), and the plan does not
duplicate that check.

**`evaluation/runners/eval_runner.py`** — `run_evaluation` gains `layer_range: Optional[str] =
None` and `direction_dir: Optional[str] = None`, both threaded from `run_eval.py:82-104`.

### 3. Edit — make a pre-extracted direction set reachable from the intervention

`VTITextualIntervention` resolves directions through `compute_or_load_textual_directions_v2`
(`intervention.py:118-126`), which is PCA-only with `SELECTION_POLICY = "shuffled_prefix"`
(`directions_v2.py:36`). Neither `intervention.py` nor `run_eval.py` imports
`directions_partition` or `mean_diff_partition`, and `run_eval.py` has no flag naming a
direction directory. This blocks S1–S6.

`__init__` already accepts a preloaded `_directions` array (`intervention.py:49`), so this is a
pass-through, not a rewrite. In `run_evaluation`, after the wrapper loads and the single-device
assertion passes:

```python
if direction_dir is not None:
    from evaluation.interventions.vti.directions_v2 import load_textual_v2_directions
    dir_path = Path(direction_dir)
    directions, dmeta = load_textual_v2_directions(dir_path)
    if directions.shape != (wrapper.num_layers, wrapper.hidden_dim):
        raise RuntimeError(
            f"Direction shape {directions.shape} != "
            f"({wrapper.num_layers}, {wrapper.hidden_dim}) for {dir_path}")
    iv_kwargs["_directions"] = directions
    iv_kwargs["direction_slug"] = dir_path.name
    iv_kwargs["direction_construction"] = dmeta.get("construction") or dmeta.get(
        "steer_reconstruction")
```

Guard: inject `_directions`, `direction_slug`, `direction_construction`, `layer_indices` and
`layer_set_name` **only** for intervention names starting with `vti_textual_`. `no_intervention`
swallows all kwargs (`evaluation/interventions/__init__.py:44`), but the visual factory drops
only `beta` (line 22) and would raise on the rest. Since `iv_runs` is built before the wrapper
loads (`eval_runner.py:262-263`), move that construction to after the wrapper load so the shape
check can run first.

`load_textual_v2_directions` works unchanged on the new directories: it reads `num_layers` and
the `layer_{i}` keys from `directions.npz` and `metadata.json`, and never touches
`components.npz`.

### 4. Edit — result directory naming, so the 72 arms do not collide

Today `iv_dir` is `f"{iv_name}__b{beta}"`, with `__d{dim}__nd{n}` appended only when
`uses_demos_v2` is true (`eval_runner.py:294-305`). With preloaded directions and no
`--vector_dimension`, `cfg` carries no `dimension` key, so S1, S2 and S3 at the same beta would
all write to `vti_textual_additive_mlp__b0.5` and overwrite each other. Extend the composition:

```python
elif beta is not None and "beta" in cfg:
    iv_dir = f"{iv_name}__b{beta}"
    dim = cfg.get("dimension") or vector_dimension
    nd = cfg.get("num_demos") if vector_dimension or demos_path else None
    if dim is not None:
        iv_dir = f"{iv_dir}__d{dim}"
    if nd is not None and (vector_dimension is not None or demos_path):
        iv_dir = f"{iv_dir}__nd{nd}"
    if cfg.get("direction_slug"):
        iv_dir = f"{iv_dir}__{cfg['direction_slug']}"
    if cfg.get("layer_set"):
        iv_dir = f"{iv_dir}__L{cfg['layer_set']}"
```

The two new suffixes appear only when the new flags are used, so every existing result directory
name is unchanged. Resulting names, self-describing without this plan in hand:

```
vti_textual_additive_mlp__b0.5__demos850_ba05bd96_all_nd200_s42_meandiff_partition__Lall
vti_textual_additive_mlp__b0.5__demos850_ba05bd96_all_nd200_s42_meandiff_partition__L5-14
vti_textual_additive_mlp__b0.9__demos850_ba05bd96_all_nd50_s42_meandiff_partition__L20-29
```

Baseline cells stay at the bare `no_intervention`.

### 5. Edit — two defaults that contradict the spec

(a) `run_eval.py:50` defaults `--chair_max_new_tokens` to 64. This design requires 256. Pass it
explicitly on every CHAIR invocation; do not change the default (other scripts depend on it, and
a silent default change would make old and new runs indistinguishable by config). The same trap
exists at `rotation_strength.py:69` and `rotation_strength_chair_amber.py:81`; neither is used
here.

(b) `VTITextualIntervention` defaults `variant="uniform_rotation"` (`intervention.py:38`). Never
rely on the default: `--interventions vti_textual_additive_mlp` selects `variant="additive"`,
`hook_site="mlp"` through the registry.

Add a hard guard in `run_evaluation`: if `"chair" in benchmarks` and `chair_max_new_tokens != 256`
and `direction_dir is not None`, raise. Cheap, and it makes a forgotten flag a crash rather than
a quietly wrong CHAIR number.

### 6. Edit — POPE scoring emits the gold-label split the spec now requires

`score_pope_records` (`metrics.py:38-113`) tracks `tp`, `fp`, `fn`, `tn` internally but returns
no `tn` and no per-gold-label accuracy. AMBER already returns what the spec asks for via
`_amber_disc_finalize` (`metrics.py:124-142`): `yes_ratio`, `neg_item_accuracy`,
`pos_item_accuracy`, `n_neg_total`, `n_pos_total`.

Add to the returned dict of `score_pope_records`, without changing or removing any existing key:

```python
"tp": tp, "fp": fp, "fn": fn, "tn": tn,
"neg_item_accuracy": tn / n_neg if n_neg else 0.0,
"pos_item_accuracy": tp / n_pos if n_pos else 0.0,
"n_neg_total": n_neg,
"n_pos_total": n_pos,
```

where `n_neg` / `n_pos` are accumulated in the existing loop from `gt`, exactly as the AMBER
scorer does (`metrics.py:172-176`). The same four counts go into each `by_category` entry.
`print_comparison_table._fmt_pope` reads only `accuracy_overall` and `f1_overall`
(`eval_runner.py:330-331`) and is unaffected. Existing `metric_summary.json` files on disk are
not rewritten; they simply lack the new keys, which is why the analysis script below recomputes
everything from per-item records regardless.

Mirror the four raw counts into the AMBER return as well — `score_amber_discriminative_records`
returns the derived rates but not `tp`/`fp`/`tn`/`fn` themselves (`metrics.py:206-223`), and the
spec requires the four counts.

### 7. New — reusable flip accounting and the per-item join

`evaluation/classifiers/metrics.py` defines no flip accounting. It exists only inline at
`rotation_strength.py:132-161`, computed against a baseline generated in the same process.
Reporting h+/h- from `run_eval.py` output needs a per-item join against the retained B1/B2
dumps; the records it needs are written (`eval_runner.py:154`) but the join does not exist.

New module `evaluation/visual_reasoning_validation/flip_counts.py`:

```python
def flip_counts_vs_baseline(steered_records: list[dict],
                            baseline_records: list[dict]) -> dict:
    """Join two responses.json record lists on `id` and count decision flips.

    Semantics are identical to rotation_strength._flip_counts (lines 132-161):
    an item is skipped when either side is unparsed or when the two parsed
    answers agree; gold comes from the record's `ground_truth`, parsed with
    evaluation.classifiers.metrics._normalize_yes_no.

    Returns:
      hallucinations_induced   int  # gold no, baseline no -> steered yes  (tn -> fp)
      hallucinations_removed   int  # gold no, baseline yes -> steered no  (fp -> tn)
      n_decision_flips         int
      n_items_joined           int
      n_excluded_unparsed_either int
      n_excluded_missing_gold  int
    """
```

The spec's Primary measurement names h+ = tn→fp and h- = fp→tn and nothing else, so those are
the two counts reported. The remaining four counts `_flip_counts` returns are not computed here.
The three exclusion counts are the base h+ and h- are measured over, and are reported alongside
per the project's reporting convention.

### 8. New — cell summarisation and plots

`evaluation/visual_reasoning_validation/summarize_cells.py`

```
python evaluation/visual_reasoning_validation/summarize_cells.py \
  --run_date 2026-07-31 --output_dir evaluation/results
```

Walks `evaluation/results/{run_date}/{model_short}/{bench_key}/{iv_dir}/responses.json`, parses
each `iv_dir` back into (intervention, beta, direction slug, nd, layer set), recomputes every
metric from per-item records rather than trusting `metric_summary.json`, joins each steered
discriminative cell against the same model's `no_intervention` cell for the same `bench_key`, and
writes three tidy CSVs plus one JSON per cell. Fails loudly if a steered cell has no matching
baseline, or if the two record sets do not cover the same id set.

`evaluation/visual_reasoning_validation/make_plots.py` reads those CSVs and writes the plots
listed under Artifacts. Every plot title and axis label is plain English; no cell codes appear in
any figure.

### 9. New — the driver

`evaluation/run_scripts/run_meandiff_visual_reasoning_validation.sh`, env knobs
`MODELS`, `LAYER_RANGES`, `NDS`, `BETAS`, `RUN_DATE`, `OUTPUT_DIR`, `CUDA_VISIBLE_DEVICES`.

Loop order, chosen so a complete comparable slice lands first: baselines → layer set
(`all`, then `5-14`, then the model's late window) → nd (`200`, `500`, `100`, `50`) → beta
(`0.5`, `0.2`, `0.9`) → benchmark (`amber`, `pope_random`, `pope_popular`, `pope_adversarial`,
`chair`). Every invocation carries `--skip_if_exists` so the job resumes.

Exact invocations, one arm:

```bash
DIR=experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demos850_ba05bd96_all_nd200_s42_meandiff_partition

# AMBER discriminative 450
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --benchmarks amber --amber_task discriminative \
  --interventions vti_textual_additive_mlp \
  --subset_ids_file data/amber/pinned_amber_disc_450.json \
  --direction_dir "$DIR" --layer_range all --beta 0.5 \
  --max_new_tokens 256 \
  --output_dir evaluation/results --run_date "$RUN_DATE" --skip_if_exists

# POPE 200 per split, three invocations
for SPLIT in random popular adversarial; do
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --benchmarks pope --pope_split "$SPLIT" --limit 200 \
  --interventions vti_textual_additive_mlp \
  --direction_dir "$DIR" --layer_range all --beta 0.5 \
  --max_new_tokens 256 \
  --output_dir evaluation/results --run_date "$RUN_DATE" --skip_if_exists
done

# CHAIR 500
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --benchmarks chair \
  --interventions vti_textual_additive_mlp \
  --subset_ids_file data/chair/pinned_chair_500.json \
  --chair_prompt "Please Describe this image in detail." \
  --chair_max_new_tokens 256 \
  --direction_dir "$DIR" --layer_range all --beta 0.5 \
  --output_dir evaluation/results --run_date "$RUN_DATE" --skip_if_exists
```

Baseline cells B1 / B2 are the same five invocations with `--interventions no_intervention` and
without `--beta`, `--direction_dir`, `--layer_range`.

Qwen arms use `--model Qwen/Qwen2.5-VL-7B-Instruct` and `--layer_range 15-24` for the late
window. Do **not** pass `--max_pixels`: the previous Qwen evaluation runs on these subsets did
not, and baseline and steered arms must share generation settings. The `max_pixels=1003520`
recorded in the direction metadata applies to how the cached activations were produced, not to
inference.

### 10. Edit — `step0_chair_token_cap.py` output filename

Covered in the gating plan, check 6. `evaluation/results/{run_date}/_diagnostics/step0_chair_token_cap.json`
becomes `…/step0_chair_token_cap_{model_short}.json` so two models on one date do not overwrite
each other.

---

## Metrics

All named in plain English. Primitives are marked as such. Composite lineage is the spec's.

### Retained per item, on every cell

Already written by `eval_runner._sample_to_meta` + `rec["response"]` (`eval_runner.py:43-51,
143-145`): sample id, question, benchmark, task, gold answer, category (AMBER question type /
POPE split), and the raw model response string. Nothing new needs retaining; the per-item file is
`responses.json` in every cell directory.

Derived per item at analysis time, and stored in the per-cell JSON:

- Discriminative: the **parsed answer** — `yes`, `no`, or unparsed — from
  `_normalize_yes_no(response)` (`metrics.py:10-27`). This is the primitive under every rate
  below.
- CHAIR: objects mentioned and objects hallucinated per caption, from `parse_caption_objects` and
  `gt_objects_for_image` (`evaluation/classifiers/chair_objects.py:141, 182`) against the
  Rohrbach synonym list vendored at `evaluation/classifiers/chair_synonyms.txt`; caption length
  in characters; whether the caption is empty.

### Per discriminative cell (AMBER 450, and each POPE split at 200)

Counts, all primitives:

- `tp`, `fp`, `tn`, `fn`, `n_unparsed`, `n_total`, `n_gold_no`, `n_gold_yes`

Rates, each reported next to the counts it is built from:

- accuracy = `(tp + tn) / n_total`
- precision = `tp / (tp + fp)`, recall = `tp / (tp + fn)`, F1 = harmonic mean — `_prf1`,
  `metrics.py:30-35`, positive class = yes on both benchmarks
- yes rate = `(tp + fp) / n_total`
- accuracy on the gold-no subset = `tn / n_gold_no`; accuracy on the gold-yes subset =
  `tp / n_gold_yes`

Against the matched baseline (B1 for LLaVA cells, B2 for Qwen cells, same benchmark, same items):

- **hallucinations induced (h+)** = number of items flipping gold-no correct → gold-no wrong,
  i.e. tn → fp. A count, not a rate.
- **hallucinations removed (h-)** = number of items flipping fp → tn. A count, not a rate.
- reported with `n_items_joined`, `n_excluded_unparsed_either`, `n_excluded_missing_gold`, and
  `n_decision_flips`.

Why the gold-label split is reported, from the spec: AMBER 450 is not balanced —
`_meta.gold_counts` is `{"no": 276, "yes": 174}`, 61.3% gold-no. Accuracy on an unbalanced
yes/no set moves both when the model discriminates better and when its answer rate shifts toward
the majority label, and the aggregate number does not say which happened. Retaining these per
item is what makes the aggregate attributable by reanalysis rather than by rerunning the
benchmark. Alex's stated assumption, to be checked against these numbers rather than assumed: the
model should perform comparably on the gold-no and gold-yes subsets.

`p_yes_norm` and `answer_mass` are banned by `CLAUDE.md` and appear nowhere in this plan.

### Per generative cell (CHAIR 500)

Counts, all primitives:

- sum over captions of hallucinated object mentions
- sum over captions of object mentions
- number of captions containing at least one hallucinated object
- `n_total`, `n_nonempty`, `n_empty`, `n_missing_image_id`
- sum of caption lengths in characters

CHAIR score, the spec's composite, lower is better and not an accuracy:

- `chair_i` = (hallucinated object mentions summed over captions) / (object mentions summed over
  captions)
- `chair_s` = (captions containing ≥1 hallucinated object) / (non-empty captions)

Both defined at `metrics.py:338-343` and both computed over non-empty captions only, because
`chair_i`'s denominator goes to 0/0 on empty captions and a collapsed run would otherwise look
hallucination-free.

Reported alongside, because `score_chair_records` documents that CHAIR confounds with caption
length in both directions and must never be read in isolation: `avg_objects_mentioned`
(= object mentions / non-empty captions — the numerator is `chair_i`'s denominator),
`avg_caption_len_chars` over all scoreable captions including empties, and `empty_fraction`.
`avg_caption_len_chars` is also the direct observable for the truncation assumption the spec
names, so it carries the CHAIR-cap check into the main run.

---

## Artifacts

### Directions (step 1)

| Path | Contents |
|---|---|
| `experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition/directions.npz` | `layer_0`…`layer_31` (4096,) float32, `num_layers=32`, `hidden_dim=4096` |
| `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition/directions.npz` | `layer_0`…`layer_27` (3584,) float32, `num_layers=28`, `hidden_dim=3584` |
| the same eight directories, `metadata.json` | fields listed under Code to write 1a |
| `experiment_artifacts/vti/{model_short}/textual_v2/demos850_meandiff_run_meta_{model_short}_2026-07-30.json` | run manifest, sha256 per set, `forward_passes: 0` |
| `experiment_artifacts/vti/meandiff_extraction_verification_2026-07-30.md` | additivity report: existing `_r2_partition` and `_act_cache` byte-identical before and after |

### Benchmark runs (step 4)

`evaluation/results/{run_date}/{model_short}/{amber|chair|pope_random|pope_popular|pope_adversarial}/{iv_dir}/`
with `responses.json` and `metric_summary.json` per cell. 370 cell directories.

### Analysis (step 5)

Under `evaluation/visual_reasoning_validation/results/{run_date}/`:

| File | Contents |
|---|---|
| `amber450_cell_metrics_all_arms.csv` | one row per (model, layer set, direction sample size, beta) with the four counts, accuracy, precision, recall, F1, yes rate, gold-no accuracy, gold-yes accuracy, `n_unparsed`, plus the baseline row per model |
| `pope600_cell_metrics_all_arms_by_split.csv` | the same, one row per (model, split, layer set, sample size, beta) |
| `chair500_cell_metrics_all_arms.csv` | one row per arm with `chair_i`, `chair_s`, hallucinated-mention and mention sums, hallucinating-caption count, non-empty count, empty count, average objects mentioned, average caption length |
| `amber450_hallucinations_induced_and_removed_vs_baseline.csv` | h+, h-, `n_decision_flips`, and the three exclusion counts per arm |
| `pope600_hallucinations_induced_and_removed_vs_baseline.csv` | the same, per split |
| `cell_manifest.json` | every cell directory, its parsed factor values, the direction slug and its sha256, git commit, generation settings, and the run date — so any row traces back to bytes |

Plots, self-describing names, plain-English titles and axes:

- `amber450_accuracy_by_beta_and_layer_set_llava-1.5-7b-hf.png` and the Qwen counterpart —
  x: steering coefficient beta; y: accuracy on AMBER discriminative 450; one line per layer set;
  one panel per direction sample size; horizontal reference line at the no-intervention baseline
- `amber450_accuracy_by_gold_label_and_beta_{model_short}.png` — two lines, accuracy on the 276
  gold-no items and on the 174 gold-yes items
- `amber450_yes_rate_by_beta_and_layer_set_{model_short}.png`
- `pope600_accuracy_by_split_beta_and_layer_set_{model_short}.png` — one panel per POPE split
- `pope600_accuracy_by_gold_label_and_beta_{model_short}.png`
- `chair500_hallucination_rates_by_beta_and_layer_set_{model_short}.png` — CHAIR-i and CHAIR-s,
  y-axis labelled "CHAIR hallucination rate (lower is better)"
- `chair500_caption_length_and_objects_mentioned_by_beta_{model_short}.png`
- `amber450_hallucinations_induced_and_removed_by_beta_and_layer_set_{model_short}.png`
- `pope600_hallucinations_induced_and_removed_by_beta_and_layer_set_{model_short}.png`
- `discriminative_and_generative_change_from_baseline_by_direction_sample_size_{model_short}.png`
  — x: direction sample size (50 / 100 / 200 / 500); y: change from the no-intervention baseline;
  one series for AMBER accuracy and one for CHAIR-i, which is the comparison the spec names as
  doing the most work

---

## Sanity checks that must pass first

The full set, with procedures and invalidation criteria, is
`implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_sanity_checks_plan_2026-07-30.md`.
Nothing in this plan runs until they pass. In brief:

1. Every activation the mean difference needs is on disk, loadable, correctly shaped, finite —
   3400 files across the two models.
2. The four partition blocks are disjoint and hash-matched to `ba05bd960cad0c18`.
3. Each of the eight new direction sets has the right shape, is finite, and has a non-zero L2
   norm at every decoder layer — a zero-norm layer is silently unsteered because `steer.py:43`
   normalises before applying.
4. Steering demonstrably reaches the model at exactly the requested layers: hook count equals
   window size, steered responses differ from baseline, the three layer-set result directories
   are distinct, and `intervention_config` records the layer set and the direction slug.
5. The model loads on one CUDA device, unsharded — a permanent guard in `run_evaluation`, not a
   smoke-only assertion.
6. **CHAIR at a 256-token cap is measured, not assumed.** The Step 0 sweep that the spec cites
   ran caps 64 and 512 on LLaVA only, n=20, and the metrics move a lot between them
   (chair_s 0.300 → 0.650, average objects mentioned 2.40 → 3.75, coverage recall 0.5714 →
   0.8143). 256 was never measured and Qwen was never measured. Re-run at caps 128 / 256 / 512
   on both models. If the numbers still move from 256 to 512, captions are truncated and the
   CHAIR rows are unreadable — **report and stop.**
7. **AMBER parse rate per model.** The 2026-06-22 no-intervention run on the same pinned 450
   shows LLaVA 0/450 unparsed and Qwen2.5-VL 84/450 (18.7%), with Qwen gold-yes accuracy 0.3506
   against gold-no accuracy 0.8261. `_flip_counts` skips any item unparsed on either side, so
   Qwen h+/h- would be computed over at most ~81% of the set. **Report and stop.**
8. The POPE 600 that `--limit 200` selects is the pinned 600.
9. The AMBER and CHAIR pinned subsets load to exactly 450 and 500, with gold counts
   `{"no": 276, "yes": 174}` and 500 distinct COCO ids.
10. Throughput measured on one complete arm before the remaining 71 are launched.

---

## Order of operations

1. Sanity checks 1, 2, 8, 9 (CPU, minutes).
2. Snapshot hashes of the 40 existing `_r2_partition` directories and both `_act_cache`
   directories.
3. Extract the eight mean-difference direction sets (CPU, both models). Re-snapshot and verify
   additivity. Run sanity check 3.
4. Land code items 2, 3, 4, 5, 6, 10. Run sanity checks 5 and 4 as a 10-item smoke on lambdab2
   GPU 0, under `--run_date 2026-07-30_smoke`.
5. Run sanity check 6 (both models, lambdab2 GPU 0) and produce sanity check 7 from the existing
   2026-06-22 dumps. **Report both to Alex. Stop here for his decision.**
6. On his go-ahead: commit, `git archive -o vti_repo.tar.gz HEAD`, stage to NFS, verify the RunAI
   prerequisites listed in the gating plan, submit two jobs — one per model, since the `nlm-mh`
   project quota is typically 2 GPUs.
7. Within each job: baselines first (B1 or B2, five invocations), then sanity check 10 on the
   first steered arm, then the remaining 35 arms in the driver's order.
8. Pull `evaluation/results/{run_date}/` back by SFTP (`.gitignore:107` excludes it from git).
9. Run `summarize_cells.py` then `make_plots.py`.

---

## Cost

**Extraction.** Zero forward passes. Reads 2 × (50 + 100 + 200 + 500) = 1700 cached `.npz` per
model, 3400 total, out of the 5100 present per model. Peak RSS under 1 GB with streaming float64
accumulation. Wall clock: minutes on CPU. Disk added: 8 × ~0.5 MB ≈ 4 MB. No GPU needed, so this
step does not contend for the one free device.

**Benchmark grid.** 370 invocations, 1550 items per arm.

- Per-invocation model load ≈ 45 s → ≈ 4.6 h aggregate across 370 launches, about 5% overhead.
  Accepted rather than restructured, because one process per (arm, benchmark) is what
  `--skip_if_exists` resume depends on.
- Measured reference: POPE at 200 items ran at 0.60–0.75 samples/s on an A6000
  (`evaluation/results/_logs/*.log`). CHAIR at a 256-token cap has never been timed here, which
  is why sanity check 10 gates the launch. Order-of-magnitude estimate on H100: AMBER ≈ 10 min,
  POPE ≈ 15 min for three splits, CHAIR ≈ 20–40 min → ≈ 50–70 min per arm → ≈ 30–45 h per model
  → **≈ 60–90 GPU-hours total**, ≈ 1.5–2 days wall clock at a 2-GPU quota.
- This is a multi-day full-benchmark sweep, so per `IMPLEMENTATION.md` routing it goes to RunAI
  (H100 80 GB, `h100-pool`, `--gpu-devices-request 1`), not lambdab2. Only the smoke checks and
  the CHAIR cap probe run locally on GPU 0.

**Memory.** A 7B model in fp16/bf16 is ≈ 15–17 GB of weights; KV cache for these sequence
lengths plus vision-encoder intermediates keeps peak under ≈ 25 GB. Hooks add one `(hidden_dim,)`
tensor per steered layer — 128 KB at most. Free on lambdab2 right now: GPU 0 has 48673 MiB free;
GPUs 1, 2, 3 are each holding ~48 GB and are unusable. RunAI H100 has 80 GB. Fits unsharded on
either, which matters because `device_map="auto"` sharding silently breaks hook-based
interventions — hence the single-device guard and `CUDA_VISIBLE_DEVICES` pinned to one device.

**Disk.** ≈ 1–2 MB of `responses.json` per invocation × 370 ≈ 0.6–1 GB, plus checkpoints during
the run. 742 GB free on `/`.

---

## What confirms or falsifies

Read against the spec's prediction table. The table's baseline column is Alex's reference, not a
measurement; the measured baselines are B1 and B2 in the same run.

| Condition | Explanation A predicts | Explanation B predicts | Explanation C predicts |
|---|---|---|---|
| LLaVA on AMBER discriminative (accuracy) | 70 → 77 | 70 → 77 | 70 → 70 |
| Qwen on AMBER discriminative (accuracy) | 74 → 78 | 74 → 78 | 74 → 74 |
| LLaVA on CHAIR (CHAIR score) | 20 → 20 | 20 → 16 | 20 → 20 |
| Qwen on CHAIR (CHAIR score) | 17 → 17 | 17 → 15 | 17 → 17 |

The spec names rows 3 and 4 — CHAIR — as doing most of the work separating A, B and C, with row 3
(LLaVA on CHAIR) the most telling, and names the comparison between the relative change from
baseline in row 1 and the relative change in row 3 as what reveals whether the vector is
improving visual reasoning or changing behaviour some other way. The last plot in the Artifacts
list is that comparison.

Abandonment criterion, in Alex's words as clarified 2026-07-30: if steering moves neither AMBER
nor POPE off their baselines, the truthfulness hypothesis is abandoned. CHAIR being unchanged is
a prediction of Explanation A, not a trigger for abandonment. POPE and AMBER discriminative test
the same ability, so a statement about AMBER covers both.

A null for a plumbing reason is not a null for an experimental reason. Sanity checks 3, 4 and 5
are what separate the two, and every steered cell's `metric_summary.json` carries
`intervention_config` with the layer set, the direction slug and the direction construction, so
the separation is checkable afterwards from the artifacts alone.

---

## Open questions

1. **Which arm the prediction table refers to.** The table has one predicted number per (model,
   benchmark), but the design produces 36 steered arms per model. Whether the table is read
   against one nominated cell, against the best-performing arm, or against every arm as a
   surface is a reading decision. The plan reports every arm and nominates none.

2. **The measured baselines differ from the table's baseline column.** On the identical pinned
   AMBER 450, the 2026-06-22 no-intervention runs give LLaVA 76.7 accuracy where the table says
   70, and Qwen 64.2 where the table says 74 (Qwen with 84/450 unparsed). Whether the table's
   baseline column should be re-anchored to the freshly measured B1/B2 numbers before the
   prediction is read is Alex's call.

3. **The Qwen AMBER parse rate.** 18.7% of Qwen answers on the pinned 450 are descriptive prose
   the yes/no parser cannot read, and two are empty. This shrinks the denominator of accuracy,
   breaks `tp + fp + tn + fn = 450`, and removes those items from the h+/h- join. Whether that
   is acceptable, or whether Qwen's discriminative rows should be read differently, is a design
   decision the plan does not make.

4. **The confidence-interval bound Alex named.** The spec's Sample size section says a
   confidence-interval bound on the baseline "is performed at analysis time, not stated here",
   and declines to state a detectable effect size. Neither the interval type nor the level is
   specified, so the plan retains everything needed to form one — the four counts and n for every
   cell — and computes none.

5. **The category slice.** The plan uses `all`, which the spec fixes as the agent's choice with
   an explicit override line. If a per-category slice (`existence`, `attribute`, `counting`,
   `relation`) was intended, the extraction step multiplies by five and the plan changes.
