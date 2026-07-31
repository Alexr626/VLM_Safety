# Shuffled-Control Direction Implementation Summary

Date: 2026-07-29

## Extractor Code

- Main module: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/evaluation/interventions/vti/shuffled_control.py`
- Runner: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/evaluation/run_scripts/extract_shuffled_control_directions.py`
- Shared demos_v2 extractor helpers: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/evaluation/interventions/vti/directions_v2.py`
- Geometric comparison: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/diagnostic_experiments/perception_diag/control/geometric_comparison/compare_deployed_vs_shuffled_control.py`

Key functions/signatures in `shuffled_control.py`:

- `derangement_path() -> Path`
- `shuffled_act_cache_dir(model_short: str) -> Path`
- `shuffled_direction_dir(model_short: str) -> Path`
- `sanity_report_path(model_short: str) -> Path`
- `deployed_direction_dir(model_short: str, slug: str = DEPLOYED_SLUG) -> Path`
- `generate_derangement(ids: Sequence[str], seed: int = DERANGEMENT_SEED) -> Dict[str, str]`
- `validate_derangement(ids: Sequence[str], mapping: Dict[str, str]) -> int`
- `load_or_write_derangement(ids: Sequence[str], path: Optional[Path] = None, seed: int = DERANGEMENT_SEED, force: bool = False) -> Tuple[Dict[str, str], Path]`
- `build_shuffled_flat_demos(flat_demos: Sequence[dict], rows_by_id: Dict[str, dict], derangement: Dict[str, str]) -> List[dict]`
- `extract_shuffled_control_direction(wrapper, model_short: str, *, derangement: Dict[str, str], derangement_file: Path, demos_path: Optional[Path] = None, order_path: Path = DEFAULT_ORDER_PATH, deployed_slug: str = DEPLOYED_SLUG, force_recompute: bool = False) -> Tuple[np.ndarray, dict, dict]`
- `write_sanity_report(model_short: str, checks: dict, meta: dict, path: Optional[Path] = None) -> Path`

## Derangement

The deployed direction slug is hardcoded as `demosv2_9a44f4af_all_nd200_s42_r2_prefix`. The control uses `DERANGEMENT_SEED = 1234`, `DIMENSION = "all"`, `NUM_DEMOS = 200`, `RANK = 2`, and `ORDER_SEED = 42`.

`load_or_write_derangement()` loads the deployed direction metadata, takes its `ids_used`, and writes or reuses `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json`. The JSON contains `seed`, `n_ids`, `n_fixed_points`, `deployed_slug_control_for`, `dimension`, `num_demos`, `created`, `map`, and `note`.

The map keys are the retained demo ids. The map values are source-image demo ids. `generate_derangement()` shuffles the same id list with `random.Random(seed)` until every key maps to a different id. `validate_derangement()` checks that keys and values are exactly the deployed `ids_used` set and returns the fixed-point count.

## Relation To Demos And Caches

The shuffled-control extractor is tied to `data/vti/demos_v2.jsonl` by default through `vti_demos_v2_path()`. It loads the deployed metadata from `experiment_artifacts/vti/{model_short}/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/`, uses that metadata's `ids_used`, then rebuilds `select_prefix_demos(..., "all", 200)` from `demos_v2_order_s42.json`.

It requires the rebuilt `used_ids` to exactly equal the deployed `ids_used` in order and length. It then keeps each demo id, question, `value`, and `h_value` fixed, but replaces `image` with the image from the deranged source demo id and records `image_source_demo_id` in the temporary flat demo records.

It intentionally does not use `textual_v2/_act_cache`. It writes activations under `experiment_artifacts/vti/{model_short}/shuffled_control/_act_cache/` because `ActivationCache` keys use demo id plus variant suffix and do not include the image path. Reusing the normal demos_v2 cache would silently reuse original-image activations.

For each shuffled demo, it calls `ensure_variant_activation()` for variants `"all"` and `"value"` with suffixes `vl_v2_all` and `vl_v2_value`, then calls `obtain_textual_vti_v2_from_stacks()` from `directions_v2.py`. The extraction math is the same live textual path as demos_v2: last-token VL activations, `value_minus_h_value`, global flat PCA, and live `PC1 + mean` reconstruction; the saved decoder directions are `full[1:]`.

## Outputs And Metadata

The extractor writes:

- Derangement file: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json`
- Direction directory: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/`
- Activation cache: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/experiment_artifacts/vti/{model_short}/shuffled_control/_act_cache/`
- Sanity report: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/experiment_artifacts/vti/{model_short}/shuffled_control/shuffled_control_sanity_report_{model_short}.md`

The direction directory uses `save_textual_v2_directions()`, so it writes `directions.npz`, `metadata.json`, and `components.npz`. The visible repository search only surfaced `metadata.json`, but the code path writes all three when extraction runs.

Metadata fields include `control_for_deployed_slug`, `control_type`, `derangement_file`, `derangement_seed`, `derangement_fixed_points`, `demos_path`, `demos_file`, `content_hash_sha256_16`, `dimension`, `num_demos_requested`, `n_pairs`, `ids_used`, `skipped_ids`, `question`, `token_policy`, `diff_polarity`, `sign_convention`, `selection_policy`, `seed`, `rank`, `steer_component`, `steer_reconstruction`, PCA variance and per-layer norm fields, `model_short`, `act_cache_dir`, `forwards_executed`, `date`, and `git_commit`.

Existing reports exist for `llava-1.5-7b-hf` and `qwen2.5-vl-7b-instruct`; both report `derangement_fixed_points = 0`, `Captions unchanged = 200 of 200`, `positional id mismatches = 0`, `forwards_executed = 400`, and finite shuffled norms.

## Runner Sanity Checks

The runner `extract_shuffled_control_directions.py` checks that deployed metadata exists, reads deployed `ids_used`, writes/loads the derangement, loads the wrapper, calls `extract_shuffled_control_direction()`, writes the sanity report, and prints direction shape, forward count, and report path.

Inside `extract_shuffled_control_direction()`, checks include:

- Deployed `directions.npz` exists.
- Deployed `ids_used` length equals hardcoded `NUM_DEMOS` (`200`).
- Derangement keys and values match `ids_used`; fixed points counted.
- If shuffled directions already exist and `force_recompute` is false, extraction is skipped.
- If shuffled act cache contains `sample_*.npz` and `force_recompute` is false, extraction aborts to avoid partial cache reuse.
- Rebuilt prefix `used_ids` from `demos_v2_order_s42.json` must exactly equal deployed metadata `ids_used`.
- Captions must match source rows for `value` and `h_values["all"]`.
- Direction shape must equal `(wrapper.num_layers, wrapper.hidden_dim)`.
- Sanity report gates fixed points, unchanged captions, same id order, direction shape, and no-cache-reuse forward count (`NUM_DEMOS * 2 = 400`).

The runner itself has a `--deployed_slug` argument and `--derangement_seed`, but `shuffled_control.py` still hardcodes several payload fields and extraction constants. The runner's Qwen `--max_pixels` help says to use `1003520` to match deployed demos_v2 extraction, but unlike the demos_850 partition runner, it does not enforce that value.

## Geometric Comparison Loading

The comparison script loads deployed directions from `experiment_artifacts/vti/{model_short}/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/` and shuffled directions from `experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/`, using `load_textual_v2_directions(cache_dir)`.

It is CPU-only and limited to `MODEL_SPECS` for `llava-1.5-7b-hf` with expected shape `(32, 4096)` and `qwen2.5-vl-7b-instruct` with expected shape `(28, 3584)`. It validates deployed and shuffled shapes against these hardcoded expected shapes and against each other.

It computes per-layer decoder cosine and L2 norms. It loads PC1 norms from `components.npz` key `pc0`, dropping embedding row `[1:]`; if `components.npz` is absent, it falls back to `metadata.json` field `pc1_layer_norms` sliced `[1:]`. It cross-checks recomputed direction norms against `metadata["direction_layer_norms"][1:]` when present.

It writes CSVs, plots, and summary under `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/diagnostic_experiments/perception_diag/control/geometric_comparison/`:

- `geometric_comparison_{model_short}.csv`
- `cosine_deployed_vs_shuffled_control_by_layer_{model_short}.png`
- `magnitude_deployed_vs_shuffled_control_by_layer_{model_short}.png`
- `geometric_comparison_summary.md`

The CSV columns are `layer`, `cosine`, `deployed_norm`, `shuffled_norm`, `deployed_pc1_norm`, and `shuffled_pc1_norm`.

## Hardcoding Relevant To Demos_850 Partition Generalization

`shuffled_control.py` is specific to the deployed demos_v2 all/nd200 direction:

- `DEPLOYED_SLUG = "demosv2_9a44f4af_all_nd200_s42_r2_prefix"`
- `DERANGEMENT_FILENAME = "shuffled_control_image_derangement_nd200_s1234.json"`
- `DIMENSION = "all"`
- `NUM_DEMOS = 200`
- `RANK = 2`
- `ORDER_SEED = 42`
- `shuffled_direction_dir()` returns `shuffled_control/all_nd200`
- `extract_shuffled_control_direction()` requires `len(ids_used) == 200`
- It rebuilds `select_prefix_demos(..., "all", 200)` from `DEFAULT_ORDER_PATH`, not a demos_850 partition file.
- Metadata writes `dimension`, `num_demos_requested`, `seed`, `selection_policy`, and `derangement_seed` from module constants, not fully from runtime arguments.
- `expected_forwards` is `NUM_DEMOS * 2`.

`extract_shuffled_control_directions.py` exposes `--demos_path`, `--deployed_slug`, and `--derangement_seed`, but because the module constants remain fixed, changing only CLI args is not enough to support demos_850 block sizes or dimensions.

`compare_deployed_vs_shuffled_control.py` is also hardcoded:

- `DEPLOYED_SLUG = "demosv2_9a44f4af_all_nd200_s42_r2_prefix"`
- Shuffled directory is always `shuffled_control/all_nd200`
- Summary title says demos_v2 `all` @ nd200
- `MODEL_SPECS` and expected shapes are fixed to two models
- `--models` choices are restricted to those model shorts

The existing demos_850 partition implementation lives separately in `evaluation/interventions/vti/directions_partition.py` and already supports `PARTITION_SIZES = (50, 100, 200, 500)`, `SELECTION_POLICY = "disjoint_partition"`, slugs like `demos850_{hash8}_{dimension}_nd{num_demos}_s{seed}_r{rank}_partition`, `data/vti/demos_850.jsonl`, and `data/vti/demos_850_partition_s42.json`. A shuffled-control generalization for demos_850 would need to use block ids from that partition rather than demos_v2 prefix ids, generate/store derangements per block or per exact `ids_used`, and include block size/dimension/hash in output paths and derangement filenames to avoid collisions.
