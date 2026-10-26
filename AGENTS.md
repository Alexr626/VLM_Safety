# AGENTS.md: guide for coding agents

This file is for AI coding agents working in this repo. Humans should start with
[`README.md`](README.md). When you answer questions about the code, cite file
paths and function names from the sections below, and read the source to confirm
details before stating them.

## 1. What this project is

Research code for **"Which hallucination signals can control a vision-language
model?"** The fixed intervention is a gated, minimum-norm correction of a
decoder hidden state (LiSeCo in text LMs; BRACS in VLMs). The signal the
correction acts on is varied along a ladder:

- L0: BRACS's mean pre-softmax image attention
- head-selective and signed attention scores
- L5: free linear probes on the RMSNorm-normalized residual state

Read-outs:
- Stage 1: detection AUROC at object-naming steps.
- Stage 2: CHAIR_i, object recall, and step-level object changes under
  correction.
- A certificate test of whether a calibrated probe's bound holds on the outputs.

- Models: `llava-hf/llava-1.5-7b-hf` (primary) and `Qwen/Qwen2.5-VL-7B-Instruct`.
- Benchmarks: CHAIR on COCO val2014, POPE, and AMBER.

### Current status: skeleton

Only infrastructure exists. **None of the experimental design is implemented:**
- the decoding loop that reruns and forks generation
- the cone projection / correction
- the probes (L5, L5′, L5x, L5u), difference-of-means, and Platt scaling
- the signal ladder
- the BRACS and VGD reimplementations
- the COCO image-pool splits

Don't invent these or add placeholder versions of them unless the user asks
for that specific piece. If a question assumes one of them exists, say that it
doesn't yet.

## 2. Commands

```bash
# Environment
conda env create -f environment.yml && conda activate vlm_hallucination_control
pip install -r requirements.txt          # Kaggle/Colab alternative (torch preinstalled)
git clone https://github.com/chengemily1/llm-control.git   # LiSeCo code, gitignored

# Data (into data/, gitignored; contains absolute paths, so rebuild per machine)
bash data_scripts/download_all_benchmarks.sh

# Tests: CPU only, no downloads; tiny random-weight models are built in-process
pytest -q tests/

# GPU smoke test (real weights)
python scripts/smoke_test.py --model llava-hf/llava-1.5-7b-hf --pad_to_square
python scripts/smoke_test.py --model Qwen/Qwen2.5-VL-7B-Instruct

# Baseline evaluation
python evaluation/run_eval.py --model llava-hf/llava-1.5-7b-hf \
    --benchmarks chair --limit 50 --pad_to_square
```

Scripts are run from the repo root. Each entry point adds the root to
`sys.path`, so `src.*` and `evaluation.*` import as packages.

## 3. Repository map

```
src/
  model.py            VLMWrapperBase, LLaVAWrapper, Qwen2VLWrapper, create_wrapper
  model_internals.py  get_internals(wrapper) -> ModelInternals (layers, norms, q/k, rotary, lm_head, spans)
  hooks.py            capture_residual / edit_residual context managers
  liseco.py           import shim for llm-control/src
  dataset.py          POPE / AMBER / CHAIR loaders, CAPTION_PROMPT, CAPTION_MAX_NEW_TOKENS
  extraction.py       ActivationCache, last-token activations, JSON/NPZ/pickle I/O, cleanup_gpu
  paths.py            every path in the project (data, experiments, artifacts, results, llm-control)
evaluation/
  benchmarks/__init__.py      EvalSample + load_{pope,amber,chair}_eval
  classifiers/metrics.py      score_{pope,amber,chair}_records, compute_metric_records
  classifiers/chair_objects.py  CHAIR object parser + COCO ground truth
  classifiers/chair_synonyms.txt  vendored Rohrbach et al. synonym list
  interventions/              InterventionBase, NoIntervention, _REGISTRY
  runners/eval_runner.py      run_evaluation, print_comparison_table
  run_eval.py                 CLI
data_scripts/                 download_{chair,pope,amber}.py, gdrive_utils.py, download_all_benchmarks.sh
experiments/                  experiment code goes here (only a README now)
scripts/smoke_test.py         GPU end-to-end check
tests/                        CPU tests (see section 7)
llm-control/                  LiSeCo authors' repo, cloned locally, gitignored
data/ artifacts/ evaluation/results/   gitignored outputs
```

Data flow for an evaluation run:

1. `run_eval.py` calls `run_evaluation`.
2. The runner loads `EvalSample`s (images load lazily) and builds the
   wrapper with `create_wrapper`.
3. For each benchmark and intervention, it calls
   `intervention.generate(wrapper, image, question, max_new_tokens)` and
   writes the records.
4. `compute_metric_records` scores them and writes `metric_summary.json`.

## 4. Module reference

### `src/model.py`
- `create_wrapper(model_id, **kwargs)` picks `LLaVAWrapper` or `Qwen2VLWrapper`.
  It silently drops the other family's kwargs (`pad_to_square` is LLaVA-only,
  `max_pixels` is Qwen-only) and raises `ValueError` for unsupported models.
  Call `.load()` before use.
- `VLMWrapperBase` gives the shared API:
  - `prepare_vl(image, text)` and `prepare_text(text)` return processor
    inputs already on the device.
  - `forward_vl` and `forward_text` return `(hidden_states, attentions, input_ids)`.
  - `generate_vl`, `generate_text`, and `generate_caption` use greedy decoding.
    The default is 512 new tokens, and they return the decoded string of new
    tokens only.
  - Other members: `decode`, `get_image_token_span(input_ids)` (returns
    `(start, end)`, end exclusive), `get_text_token_positions`, `num_layers`,
    `hidden_dim`, `model_name`, `device`, `cleanup()`.
- `LLaVAWrapper(model_id, pad_to_square=False)`:
  - 32 layers, d=4096, 576 image tokens, image token id 32000.
  - Prompt `"USER: <image>\n{text} ASSISTANT:"`, with BOS at position 0.
  - fp16.
  - `pad_to_square=True` uses `expand2square` with the CLIP mean colour
    (original LLaVA preprocessing) instead of HF's resize and centre crop.
    The project protocol uses it.
- `Qwen2VLWrapper(model_id, max_pixels=None)` (Qwen2.5-VL-7B-Instruct):
  - 28 layers, d=3584.
  - Uses the chat template, which inserts Qwen's default system prompt.
  - Dynamic image-token count, M-RoPE, **no BOS token**, and `q_proj` has a
    bias.
  - bf16 when the GPU supports it, else fp16 (T4).
  - `max_pixels` caps the vision tokens. On GPUs under 20 GiB a memory cap
    is set, so part of the model is offloaded to the CPU.
- `_normalize_model_name(model_id)` gives `model_short`: `llava-1.5-7b-hf` or
  `qwen2.5-vl-7b-instruct`. Use it as the directory key for all outputs.

### `src/model_internals.py`
`get_internals(wrapper)` returns a `ModelInternals` object with these members:
- `layers`, `num_layers`, `layer(i)`, `attn(i)`, `mlp(i)`
- `input_layernorm(i)`, `post_attention_layernorm(i)`, `final_norm`,
  `rotary_emb`, `lm_head`
- `rmsnorm_params(i)` returns `(gain, eps)` for layer i's input RMSNorm.
- `qk_proj(i)` returns `(q_proj, k_proj)`.
- `text_config`, `num_heads`, `num_kv_heads`, `head_dim`
- `image_token_span(ids)`, `bos_position(ids)` (`None` for Qwen)

It finds submodules by trying several attribute paths, because transformers
moved them after 4.50. Add a path to `_DECODER_PATHS` or `_LM_HEAD_PATHS`
rather than hardcoding one elsewhere.

### `src/hooks.py`
- `capture_residual(mi, layers, site="output", positions="last")` is a
  context manager that yields `{layer: [tensor per forward call]}`, as CPU
  float32. `positions` can be:
  - `"last"`, giving `(batch, d)`
  - `"all"`, giving `(batch, seq, d)`
  - a list of indices, giving `(batch, k, d)`
- `edit_residual(mi, layer, edit_fn, site="output")` calls
  `edit_fn(hidden, call_idx)`. `hidden` is the full `(batch, seq, d)` tensor
  in model dtype. Return the edited tensor, or `None` to leave it unchanged.
- Sites, for 0-based decoder layer `i`:
  - `"input"`: the residual stream entering layer i, equal to
    `hidden_states[i]`.
  - `"normed"`: `g * x / RMS(x)` from layer i's input RMSNorm, which is
    what attention reads. Capture only.
  - `"output"`: the residual stream leaving layer i, equal to
    `hidden_states[i + 1]`.
- Inside `generate()` with the KV cache, call 0 is the prefill (the whole
  prompt). Each later call is one decoded token, with seq_len 1.
- Hooks don't do any correction math; the experiment provides the edit.

### `src/liseco.py`
- `load_liseco_module("control_wrapper")` adds `llm-control/src` to
  `sys.path` and imports it. LiSeCo uses bare imports such as
  `from data_util import ...`. If the clone is missing, it raises an error
  telling the user to clone it.
- **Never import `train_probe` or `main`**: they parse `sys.argv` at import
  time.
- How the reference `LiSeCoWrapper` behaves: it replaces a decoder layer in
  the `ModuleList`, edits only the last position of the *raw* (un-normed)
  layer output, assumes batch size 1, and expects the layer to return a
  tuple. Treat it as a reference implementation, not a dependency of the
  pipeline.

### `src/dataset.py`
- Loaders:
  - `load_pope(split, limit, load_images=True)`
  - `load_amber(task, subset_ids, load_images=True)`
  - `load_chair(subset_ids, prompt_override, load_images=True)`
  - `load_benchmark(name)`, `load_image_for_sample(sample)`
- Constants: `ALL_BENCHMARKS = ["pope", "amber", "chair"]`,
  `CAPTION_PROMPT = "Please describe this image in detail."`, and
  `CAPTION_MAX_NEW_TOKENS = 512` (OPERA protocol).
- Sample dict:
  `{id, image_path, image_pil, text, label, label_idx, benchmark, task, category, raw}`.
  CHAIR samples carry `raw["coco_id"]`.
- Pass `load_images=False` for large sets; there are about 40k COCO images.
- Each loader reads `data/{benchmark}/combined.json`, which the download
  scripts build.

### `src/extraction.py`
- `ActivationCache(dir)` stores `sample_{id}_{suffix}.npz` files with keys
  `layer_{l}`. Methods: `save`, `load`, `load_or_none`, `exists`.
- `get_last_token_activations(hidden_states)` and `load_activation_matrix`.
- `cleanup_gpu()`.
- `save_json` and `load_json` (numpy-safe), `save_pickle`, `save_npz`.

### `src/paths.py`
Use these helpers instead of hardcoded paths:
- `project_root`, `data_root`, `coco_val2014_dir`, `coco_annotations_dir`
- `pope_data_dir`, `amber_data_dir`, `chair_data_dir`
- `experiment_results_dir(exp, model_short)`, which gives
  `experiments/{exp}/outputs/{model_short}/results/`
- `experiment_plots_dir`
- `artifacts_dir(exp, model_short)`, which gives
  `artifacts/{exp}/{model_short}/`
- `evaluation_results_dir(run_date, model_short, benchmark_key, intervention)`
- `llm_control_root`

### `evaluation/`
- `EvalSample(id, question, image, benchmark, task, ground_truth, metadata)`
  has a `.load_image()` method, so images load lazily. The loaders are
  `load_{pope,amber,chair}_eval`.
- `classifiers/metrics.py`:
  - POPE and AMBER-discriminative are yes/no scoring, with "yes" as the
    positive class: accuracy, precision, recall, F1, `yes_ratio`, and for
    AMBER `neg_item_accuracy`.
  - `score_chair_records` returns `chair_i`, `chair_s`, `recall`,
    `avg_objects_mentioned`, `avg_caption_len_chars`, and `n_empty`. Empty
    captions are excluded from CHAIR.
  - `compute_metric_records(records, benchmark)` dispatches to the right
    scorer.
- `classifiers/chair_objects.py`:
  - An n-gram synonym parser with no nltk or pattern.en dependency.
  - `load_gt_objects()` builds the ground truth as COCO
    `instances_val2014` labels unioned with objects parsed from
    `captions_val2014`. This matches the canonical CHAIR, so scores differ
    from instance-only implementations.
- `interventions/`: an intervention subclasses `InterventionBase`, which has
  `name`, `config`, and
  `generate(wrapper, image, question, max_new_tokens) -> str`. Register it in
  `_REGISTRY` in `interventions/__init__.py`. Only `no_intervention` exists.
- `runners/eval_runner.py`:
  - Writes to
    `evaluation/results/{run_date}/{model_short}/{benchmark_key}/{intervention}/`,
    where `benchmark_key` is `pope_{split}` for POPE.
  - Each directory gets `responses.json`, `metric_summary.json`, and a
    `responses.checkpoint.json` that is saved every 10 samples and deleted
    when the run completes.
  - Resume is per sample. `--skip_if_exists` skips finished runs.
  - CHAIR gets its own frozen token budget (`--chair_max_new_tokens`,
    default 512); POPE and AMBER use `--max_new_tokens` (default 128).
  - `--subset_ids_file` pins exact sample ids.

## 5. How to do common tasks

**Capture hidden states during captioning**
```python
from src.model import create_wrapper
from src.model_internals import get_internals
from src.hooks import capture_residual
from src.dataset import CAPTION_PROMPT, CAPTION_MAX_NEW_TOKENS

w = create_wrapper("llava-hf/llava-1.5-7b-hf", pad_to_square=True).load()
mi = get_internals(w)
with capture_residual(mi, layers=[15, 20], site="normed") as store:
    caption = w.generate_vl(image, CAPTION_PROMPT, max_new_tokens=CAPTION_MAX_NEW_TOKENS)
# store[15][t]: (1, d) state at the last position of forward call t, the call that
# predicts generated token t. Call 0 is the prefill.
```

**Edit the residual stream during generation**: write an `edit_fn(hidden, call_idx)`
that changes `hidden[:, -1, :]`, and wrap the generate call in
`with edit_residual(mi, layer, edit_fn):`. Return `None` on calls you don't
want to change, for example `call_idx == 0` to leave the prefill alone.

**Add an intervention to the evaluation runner**
1. Add `evaluation/interventions/my_method.py` with an `InterventionBase`
   subclass. Inside `generate`, open the hooks and call `wrapper.generate_vl`.
2. Register it as `"my_method": lambda model_id, **kw: MyMethod(...)` in
   `_REGISTRY`.
3. Run `python evaluation/run_eval.py --model ... --interventions no_intervention my_method`.

**Add an experiment**: create `experiments/{name}/` with entry points that take
`--model <hf-id>`. Write small JSON and plots to `experiment_results_dir(name, model_short)`
and large arrays to `artifacts_dir(name, model_short)`. See
[`experiments/README.md`](experiments/README.md).

**Support a new transformers version**: if `get_internals` can't find the
decoder or lm_head, add the new attribute path to `_DECODER_PATHS` or
`_LM_HEAD_PATHS`. Then run `pytest tests/test_hooks_tiny.py`.

## 6. Conventions and pitfalls

- **Layer indexing.** `layer(i)` and hook layers are 0-based decoder indices.
  HF `hidden_states[0]` is the embedding output, and `hidden_states[i + 1]`
  is the output of `layer(i)`. HF applies the final norm to the *last*
  `hidden_states` entry, so it differs from the raw output of the last
  layer. Every result file should say which convention it uses.
- **Prompt positions differ by model.** LLaVA has BOS at 0, then the image
  tokens. Qwen has no BOS, and its chat template adds a system prompt before
  the image. Always get spans from `image_token_span` and `bos_position`
  instead of hardcoding offsets.
- **Decoding protocol.** Use greedy decoding everywhere. Keep the CHAIR caption
  length frozen at 512 tokens across all conditions, because caption length
  changes CHAIR.
- **Square padding.** LLaVA runs in the project protocol should pass
  `pad_to_square=True`; it defaults to off.
- **Precision.** LLaVA runs in fp16. Qwen runs in bf16, or fp16 on T4s, where
  outputs need checking.
- **Image pools.** The COCO pools for probe training, calibration, and
  evaluation must be disjoint draws from val2014, excluding POPE's 500
  images (proposal Table 3).
- **Dependency pins.** Keep `transformers==4.50.1` unless the user asks to
  upgrade; the hooks and internals were verified against it.
- **What goes in git.** Commit only small JSON summaries and plots. `data/`,
  `artifacts/`, `evaluation/results/`, `llm-control/`, and checkpoints are
  gitignored.
- **Scope.** This repo is only about hallucination control. Don't
  reintroduce code from the parent project (VLM safety, refusal
  classification, or other model families).

## 7. Tests

`pytest -q tests/` runs on CPU without network access or downloaded data:
- `test_imports.py`: the `src` and `evaluation` modules import,
  `create_wrapper` dispatch works, and `expand2square` pads correctly.
- `test_chair_objects.py`: synonym parsing and object extraction.
- `test_metrics.py`: CHAIR, recall, and POPE scoring against stubbed ground
  truth.
- `test_hooks_tiny.py`: builds tiny random LLaVA and Qwen2.5-VL models and
  checks three things:
  - the `input` and `output` sites match `output_hidden_states`, and
    `normed` matches the RMSNorm output;
  - edits land where intended;
  - one `generate()` call makes one prefill hook call plus one per later
    token.

Run these after changing `src/` or `evaluation/`. `scripts/smoke_test.py` is
the GPU check against real weights.
