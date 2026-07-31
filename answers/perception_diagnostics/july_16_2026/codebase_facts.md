# Perception Diagnostics Codebase Facts

Date: 2026-07-16

## VTI demos_v2 directions

- `evaluation/interventions/vti/directions_v2.py` writes `directions.npz` with keys `layer_0 ... layer_{num_layers-1}`, plus `num_layers` and `hidden_dim`; these are decoder-layer directions only, because the full `(num_layers+1, hidden_dim)` direction drops embedding row via `full[1:]`.
- `components.npz` is optional and contains `pca_mean_flat`, `n_layers_plus`, `hidden_dim`, `rank`, and `pc{i}` arrays reshaped to `(num_layers+1, hidden_dim)`.
- `metadata.json` includes demos path/file/hash, dimension, requested and actual pairs, ids/skips, token policy, diff polarity `value_minus_h_value`, sign convention, selection policy `shuffled_prefix`, seed, rank, steer component `0`, reconstruction `live_pc1_plus_mean`, EVR/layer norms, model short name, slug, date, and git commit.
- Cache slug: `demosv2_{demos_hash[:8]}_{dimension}_nd{num_demos}_s{seed}_r{rank}_{policy}`, where policy is usually `prefix`.
- PC1 load path: `compute_or_load_textual_directions_v2(...)` returns cached decoder directions from `directions.npz` or computes PCA as `PC1 + mean`, slices `[1:]`, and returns `(num_layers, hidden_dim)`.
- Mean-diff delta can be recomputed from cached activation stacks as mean over `(value_stack - h_stack)`; only its per-layer norms are stored in metadata as `mean_diff_layer_norms_mean_over_demos`, not the vector itself.
- Qwen2.5 extraction CLI is `CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demosv2_directions.py --model Qwen/Qwen2.5-VL-7B-Instruct --dimensions all --num_demos 50 100 200 500 --rank 2 [--max_pixels 1003520]`.

## Steering and hooks

- `steer(x, direction, alpha, variant, eps_coeff=0.1, *, log_lambda_sim=False, log_records=None, layer_idx=None, token_strings=None) -> torch.Tensor`.
- Variants are `additive`, `uniform_rotation`, `gated_rotation`; hook sites are `mlp`, `layer`.
- Additive returns `x + alpha * normalize(direction)`.
- Rotation variants preserve original activation norm: `normalize(normalize(x) + eps_coeff * alpha * lam * d) * norm`; `uniform_rotation` uses `lam = 1`, while `gated_rotation` uses `lam = 1 + clamp(cos(x, -d), min=0)` on single-token decode steps and can log per-token lambda diagnostics.
- `vti_hook_ctx(wrapper, directions, *, variant, alpha, hook_site, eps_coeff=0.1, log_lambda_sim=False, log_records=None, decode_input_ids=None, steer_prefill=True, skip_first_token=False)` registers one forward hook per decoder layer and removes them on exit.
- Hooks are ordinary PyTorch forward hooks, so nested capture hooks can compose mechanically. Ordering matters: a later capture hook on the same module sees any output replacement made by earlier hooks; an earlier capture hook sees pre-replacement output.

## Activation extraction and mediation

- `ActivationCache(cache_dir: str)` stores `{cache_dir}/sample_{sample_id}_{suffix}.npz`; keys are `layer_{l}` arrays. Methods: `exists(sample_id, suffix="vl")`, `save(sample_id, activations, suffix="vl")`, `load(sample_id, suffix="vl")`, `load_or_none(sample_id, suffix="vl")`.
- `get_last_token_activations(hidden_states) -> Dict[int, np.ndarray]` returns every hidden-state row as `hidden_states[l][0, -1, :]`, including embedding row.
- Wrapper `forward_vl` / `forward_text` methods request `output_hidden_states=True`, `use_cache=False`, and return `(hidden_states, attentions|None, input_ids)`. For Qwen2/2.5 this is in `src/model.py` around lines 1587-1611.
- All-layer last-prefill hidden states are obtained by `hidden, _, input_ids = wrapper.forward_vl(image, text)` then `get_last_token_activations(hidden)`; this returns keys `0..num_layers` where key 0 is embedding output.
- `FamilyDispatch.get_layer(wrapper, idx)` is abstract in `src/mediation.py`; Qwen2-VL resolves to `wrapper.model.model.layers[idx]`, LLaVA to `wrapper.model.language_model.model.layers[idx]`, Qwen-VL-Chat to `wrapper.model.transformer.h[idx]`.

## Scoring target / logit scoring

- `ScoringTarget` is in `src/mediation.py`.
- `ScoringTarget.yes_no(wrapper, yes_variants=None, no_variants=None) -> Tuple[ScoringTarget, ScoringTarget]`; defaults are `["Yes", "yes", " YES", " yes"]` and `["No", "no", " NO", " no"]`.
- `ScoringTarget.from_variants(wrapper, variants: List[str]) -> ScoringTarget` resolves deduped first-subtoken ids via `resolve_token_ids`.
- Existing first-token logit scoring is `forward_with_logits(wrapper, image, text) -> Tuple[torch.Tensor, int]`, which returns `logits[0, -1, :]`, then `target_token_probs(wrapper, image, text, target)` sums softmax probability over target token ids. This currently implements LLaVA, ShareGPT4V/raw Llama, and Qwen-VL-Chat in `_prepare_inputs_for_logits`; Qwen2-VL falls through to `NotImplementedError`.

## Yes/no normalization

- `_normalize_yes_no(text: str) -> Optional[str]` is in `evaluation/classifiers/metrics.py`.
- Returns `"yes"`, `"no"`, or `None`.
- `None` represents empty, ambiguous, or unparseable text. In POPE/AMBER scoring, unparseable predictions count as incorrect for accuracy; if gold is yes they also count as false negatives, while gold-no unparseables are not false positives.

## Existing AMBER Qwen2.5 result trees

Under `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/evaluation/results/2026-06-22/qwen2.5-vl-7b-instruct/amber/`:

- `vti_textual_uniform_rotation_mlp__b0.4/responses.json` (`n_total=450`; highest available Qwen uniform-rotation MLP beta on this date)
- `vti_textual_uniform_rotation_mlp__b0.2/responses.json` (`n_total=450`)
- `vti_textual_uniform_rotation_mlp__b0.1/responses.json` (`n_total=450`)
- `vti_textual_additive_layer__b0.4/responses.json` (`n_total=450`)
- `vti_textual_additive_mlp__b0.4/responses.json` (`n_total=450`)
- `no_intervention/responses.json` (`n_total=450`)

No Qwen `uniform_rotation_layer` AMBER responses were found under `2026-06-22`.

## Eval token defaults

- CLI defaults are in `evaluation/run_eval.py`: `--max_new_tokens` default `256`; `--chair_max_new_tokens` default `64`.
- Runner defaults are in `evaluation/runners/eval_runner.py`: `run_evaluation(..., max_new_tokens=256, chair_max_new_tokens=64, ...)`.
- Per-benchmark cap is set in `evaluation/runners/eval_runner.py`: CHAIR uses `chair_max_new_tokens`; all other benchmarks use `max_new_tokens`.
- To make global 512, change `run_eval.py` defaults and `run_evaluation` defaults; CHAIR remains 64 unless `chair_max_new_tokens` is also changed to 512.

## AMBER subset and fields

- `data/amber/pinned_amber_disc_450.json` is a dict with key `"amber"` mapped to 450 sample ids like `amber_disc_11238`.
- `evaluation/run_eval.py --subset_ids_file` passes to `run_evaluation`; `evaluation/runners/eval_runner.py` loads either `{benchmark: [ids]}` or flat `[ids]` and passes `subset_ids` into the loader.
- `src/dataset.py load_amber(..., subset_ids=None)` filters by `row["id"] in subset_ids` before image loading; subset ids override `limit`.
- AMBER `combined.json` samples use `text` for the question, while raw AMBER stores `query`; result records store `question` from the eval sample and `metadata.raw.query`.
- Gold labels and qtypes are joined from `data/amber/data/annotations.json` by numeric raw `id`, reading `truth` and `type`; qtypes are mapped to `existence`, `attribute`, or `relation`.

## Diagnostic experiment layout pattern

- Existing layouts are `diagnostic_experiments/{experiment}/run_*.py` or `compute_*.py`, optional `plot_*.py`/`analyze_*.py`, and `run_scripts/run_{experiment}.sh`.
- Scripts usually set `_PROJECT_ROOT`, insert it into `sys.path`, use `src.paths.diagnostic_results_dir(EXPERIMENT, model_short)` for outputs, and save JSON via `src.extraction.save_json`.
- Examples to mirror: `diagnostic_experiments/causal_mediation/run_mediation.py`, `diagnostic_experiments/modality_shift/compute_modality_shift.py`, `diagnostic_experiments/vti_lambda_sim/compute_lambda_sim.py`, and shell wrappers under each `run_scripts/`.
