# VLM Hallucination Control — Agent Reference

## Project

Tests which internal signals can *control* (not just detect) object
hallucination in VLMs. A gated minimum-norm correction of the hidden state
(LiSeCo in text models; BRACS in VLMs) is held fixed while the signal it acts
on varies along a ladder: from BRACS's mean pre-softmax image attention, through
head-selective and signed attention scores, to free linear probes on the
RMSNorm-normalized state. Read-outs: detection AUROC at naming steps (Stage 1),
and CHAIR_i / recall / step-level object changes under correction (Stage 2), plus
a certificate test of whether a calibrated probe's bound holds on outputs.

Models: `llava-hf/llava-1.5-7b-hf` (primary), `Qwen/Qwen2.5-VL-7B-Instruct`.
Benchmarks: CHAIR on COCO val2014, POPE, AMBER.

**Status: skeleton.** Nothing in the experimental design is implemented yet:
no decoding loop with reruns/forks, cone correction, probes, signal ladder,
BRACS/VGD reimplementation, or image pools. Do not add these without a plan
from the team.

## Layout

```
src/                 model.py, model_internals.py, hooks.py, dataset.py,
                     extraction.py, paths.py, liseco.py
evaluation/          benchmarks/, classifiers/, interventions/, runners/, run_eval.py
data_scripts/        download_chair.py (COCO), download_pope.py, download_amber.py
experiments/         one dir per experiment; outputs/{model_short}/results/
artifacts/           large intermediates (gitignored)
scripts/smoke_test.py, tests/
llm-control/         LiSeCo authors' repo, cloned by each teammate (gitignored)
data/                all datasets (gitignored)
```

## Core modules

### `src/model.py`
- `create_wrapper(model_id, pad_to_square=False, max_pixels=None, torch_dtype=..., device_map="auto")`.
  Wrapper-specific kwargs are dropped for the other family.
- `VLMWrapperBase`: `load()`, `prepare_vl(image, text)`, `prepare_text(text)`
  (processor inputs on device), `forward_vl/forward_text` →
  `(hidden_states, attentions, input_ids)`, `generate_vl/generate_text/generate_caption`
  (greedy), `decode`, `get_image_token_span(input_ids)` → `[start, end)`,
  `get_text_token_positions`, `num_layers`, `hidden_dim`, `model_name`, `device`.
- `LLaVAWrapper`: 32 layers, d=4096, 576 image tokens, image token id 32000,
  prompt `"USER: <image>\n{text} ASSISTANT:"`, BOS at position 0. fp16.
  `pad_to_square=True` = original LLaVA preprocessing (pad with CLIP mean colour)
  instead of HF resize + centre crop; the proposal fixes this on.
- `Qwen2VLWrapper` (Qwen2.5-VL-7B-Instruct): 28 layers, d=3584, chat template,
  dynamic image-token count, M-RoPE, **no BOS**, q_proj has a bias. bf16 where
  supported, else fp16 (T4). `max_pixels` caps vision tokens.
- `expand2square(image, color)`, `_normalize_model_name(model_id)` →
  `llava-1.5-7b-hf`, `qwen2.5-vl-7b-instruct`.

### `src/model_internals.py`
`get_internals(wrapper)` → `ModelInternals` with `layers`, `layer(i)`, `attn(i)`,
`mlp(i)`, `input_layernorm(i)`, `post_attention_layernorm(i)`, `final_norm`,
`rotary_emb`, `lm_head`, `rmsnorm_params(i)` → `(gain, eps)`, `qk_proj(i)`,
`num_heads`, `num_kv_heads`, `head_dim`, `image_token_span(ids)`, `bos_position(ids)`.
Resolves submodule paths across transformers versions.

**Layer indexing:** `i` is a 0-based decoder-layer index. HF
`hidden_states[0]` is the embedding output; `hidden_states[i + 1]` is the
output of `layer(i)`. State which convention you use in every result file.

### `src/hooks.py`
- `capture_residual(mi, layers, site="output"|"input"|"normed", positions="last"|"all"|[idx])`
  → yields `{layer: [tensor per forward call]}` (CPU float32).
- `edit_residual(mi, layer, edit_fn, site="output"|"input")`; `edit_fn(hidden, call_idx)`
  returns the edited `(batch, seq, d)` tensor or `None`.
- Inside `generate()` with KV cache, call 0 is the prefill (full prompt) and each
  later call is one decoded token (seq_len 1). `"normed"` is
  `g * x / RMS(x)` from the layer's input RMSNorm (what attention reads).

### `src/liseco.py`
`load_liseco_module("control_wrapper")` puts `llm-control/src` on `sys.path` and
imports it. Raises a clear error if the clone is missing. Do not import
`train_probe` / `main` (they parse argv at import). Reference behaviour of
`LiSeCoWrapper`: replaces a decoder layer in the `ModuleList`, edits only the
last position of the **raw** layer output, batch size 1, assumes the layer
returns a tuple.

### `src/dataset.py`
`load_pope(split=..., limit=..., load_images=True)`, `load_amber(task=..., subset_ids=...)`,
`load_chair(subset_ids=..., prompt_override=...)`, `load_benchmark(name)`,
`load_image_for_sample(sample)`, `ALL_BENCHMARKS = ["pope", "amber", "chair"]`.
`CAPTION_PROMPT = "Please describe this image in detail."`,
`CAPTION_MAX_NEW_TOKENS = 512` (OPERA protocol). Sample dict:
`{id, image_path, image_pil, text, label, label_idx, benchmark, task, category, raw}`;
CHAIR samples carry `raw["coco_id"]`. Pass `load_images=False` for large sets.

### `src/extraction.py`
`cleanup_gpu()`, `get_last_token_activations(hidden_states)`,
`ActivationCache(dir)` (`sample_{id}_{suffix}.npz`, keys `layer_{l}`),
`load_activation_matrix`, `save_json/load_json/save_pickle/save_npz`.

### `evaluation/`
- `classifiers/metrics.py`: `score_pope_records`, `score_amber_records`
  (discriminative, yes-positive, `neg_item_accuracy`, `yes_ratio`),
  `score_chair_records` → `chair_i`, `chair_s`, `recall`, `avg_objects_mentioned`,
  `avg_caption_len_chars`, `n_empty` (empty captions excluded from CHAIR).
- `classifiers/chair_objects.py`: vendored Rohrbach synonym list, n-gram parser
  (no nltk / pattern.en), COCO GT = `instances_val2014` ∪ objects parsed from
  `captions_val2014` (canonical CHAIR).
- `interventions/`: `InterventionBase.generate(wrapper, image, question, max_new_tokens)`;
  register new ones in `interventions/__init__.py:_REGISTRY`.
- `runners/eval_runner.py`: per-sample resume, checkpoint every 10 samples,
  `metric_summary.json`, `print_comparison_table`.

## Conventions
- Scripts take `--model <hf-id>` and write under `model_short`.
- Use `src/paths.py` helpers, not hardcoded paths.
- Greedy decoding; CHAIR length frozen at 512 tokens across all conditions.
- COCO image pools for probe training / calibration / evaluation must be
  disjoint draws from val2014 with POPE's 500 images excluded (proposal Table 3).
- Large arrays go to `artifacts/`; commit only small JSON summaries and plots.
- Keep `tests/` CPU-only; run `pytest -q tests/` before committing.
