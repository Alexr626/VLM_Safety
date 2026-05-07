# VLM Safety — Claude Reference

## Overview
This repository investigates why Vision-Language Models (VLMs) sometimes produce unsafe outputs given safe image + safe text inputs (the **SSU phenomenon**) using the **ShiftDC** method from the paper "Understanding and Rectifying Safety Perception Distortion in VLMs".

## Core Concept: ShiftDC Method

**Modality Shift Vector:**
```
m^l = x_vl^l - x_tt^l
```
- `x_vl`: activations from image + text (real multimodal input)
- `x_tt`: activations from caption + text (text-only counterpart)

The shift vector `m^l` is projected onto a **safety direction** `s^l` to compare
SSS (Safe→Safe) vs SSU (Safe→Unsafe) groups across transformer layers.

### Two safety directions (`s^l` and `c^l`)

- **Semantic safety direction `s^l`** (CatQA-derived) — top-1 PC of the
  **centered per-pair difference matrix** `D_i = H_safe[i] − H_unsafe[i]`
  over CatQA harmless/harmful minimal-edit pairs. **Pairwise PCA** is the
  canonical recipe; the legacy joint-PCA estimate is also persisted as
  `safety_direction_vectors_joint.npz` plus `recipe_sanity.json` (per-layer
  `cos(s_pair, s_joint)`) so the recipe choice is auditable. Falls back to
  joint when refs lack pair structure (e.g. mm-safetybench + llava-instruct).

- **Compositional safety direction `c^l`** — computed in **four variants**
  parameterised by `(source ∈ {holisafe, mssbench}, representation ∈ {tt, vl})`:
    - `holisafe_{tt,vl}` — joint PCA over the full HoliSafe SSS+SSU pool
      (no pair structure → joint is the only sensible recipe).
    - `mssbench_{tt,vl}` — pairwise PCA over MSSBench train-split pairs
      keyed by `(rec_idx, q_idx)`. SSS and SSU samples in a pair share
      identical text and only differ in which image variant the model sees,
      yielding a clean lexicographically-unbiased compositional contrast.
      Each MSSBench variant also writes a `..._joint.npz` sanity copy plus
      `pairwise_vs_joint_*.json`.

  **`mssbench_vl` is the canonical compositional direction** consumed by
  the `comp_safety_shift` refusal-eval intervention.

## Supported Models

The pipeline supports ten VLMs via `create_wrapper()` in `src/model.py`:

| Model ID                              | Short name                | Wrapper class        |
|---------------------------------------|---------------------------|---------------------|
| `llava-hf/llava-1.5-7b-hf`            | `llava-1.5-7b-hf`         | `LLaVAWrapper`      |
| `llava-hf/llava-v1.6-vicuna-7b-hf`    | `llava-v1.6-vicuna-7b-hf` | `LLaVAWrapper`      |
| `Lin-Chen/ShareGPT4V-7B`              | `sharegpt4v-7b`            | `ShareGPT4VWrapper` |
| `Vision-CAIR/MiniGPT-4`               | `minigpt-4`               | `MiniGPT4Wrapper`*  |
| `Qwen/Qwen-VL-Chat`                   | `qwen-vl-chat`            | `QwenVLWrapper`     |
| `Qwen/Qwen2-VL-7B`                    | `qwen2-vl-7b`             | `Qwen2VLWrapper`    |
| `Qwen/Qwen2-VL-7B-Instruct`           | `qwen2-vl-7b-instruct`    | `Qwen2VLWrapper`    |
| `Qwen/Qwen2.5-VL-7B-Instruct`         | `qwen2.5-vl-7b-instruct`  | `Qwen2VLWrapper`    |
| `OpenGVLab/InternVL2-8B`              | `internvl2-8b`            | `InternVL2Wrapper`  |
| `OpenGVLab/InternVL2_5-8B-MPO`        | `internvl2.5-8b-mpo`      | `InternVL2Wrapper`  |

*\* MiniGPT-4 requires the [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) repo
in PYTHONPATH and a downloaded checkpoint (set `MINIGPT4_CKPT` env var).*

The short name is used as the directory key under `data/*/activations/{model}/`,
`experiment_artifacts/{model}/`, `diagnostic_experiments/{model}/`, etc.

## Key Datasets

### Diagnostic Datasets

- **HoliSafe-Bench** (etri-vilab/holisafe-bench) — main SSS/SSU pool.
  Filtered to: **SSS** (safe image + safe text → safe output) and
  **SSU** (safe image + safe text → unsafe output, the anomaly).
- **MSSBench** (kzhou35/mssbench, chat split) — used both as a refusal-eval
  benchmark AND as a paired diagnostic dataset for the compositional safety
  direction. 300 records × 2 image variants × 2 queries = 1200 paired
  samples; `(rec_idx, q_idx)` is the natural pair key. `load_mssbench()`
  in `src/dataset.py` and the eval-side loader at
  `evaluation/benchmarks/mssbench.py` produce **identical** sample ids
  (`mssbench_{rec_idx:04d}_{SSS|SSU}_{stem}_q{q_idx}`), so activations
  cached on the diagnostic side and responses cached on the eval side
  reference the same sample by the same string.

- **SIUO** (sinwang20/SIUO) — 167 cross-modality SSU examples across 9
  safety domains, each with a safe image + unsafe text → unsafe output.
  `siuo_gen.json` has the originals; `siuo_sss.json` has minimal-edit
  safe-question counterparts (generated via Claude API by
  `data_scripts/generate_siuo_sss_pairs.py`). Used for **text-swap**
  causal mediation: same image, swap between safe/unsafe question text.

### Reference Datasets (for the semantic safety direction `s^l`)
- **CatQA-Harmful** (unsafe reference): contrastive harmful questions (text-only).
- **CatQA-Harmless** (safe reference): contrastive harmless counterparts —
  minimal-edit pairs of the harmful set; pair correspondence is preserved
  through extraction and exploited by the pairwise PCA estimator for `s^l`.
- **MM-SafetyBench** (alternative unsafe reference): PKU-Alignment/MM-SafetyBench (has images).
- **LLaVA-Instruct-80k** (alternative safe reference): liuhaotian/LLaVA-Instruct-150K (has images).

## Project Structure

```
VLM_Safety/
├── src/                                    # Core library modules
│   ├── model.py                            # VLMWrapperBase + per-model wrappers
│   ├── dataset.py                          # Dataset loaders + REFERENCE_REGISTRY
│   └── extraction.py                       # ActivationCache, SVD utilities, helpers
│
├── data/                                   # All datasets + per-model artifacts
│   ├── holisafe-bench/                     # Main HoliSafe pool
│   │   ├── holisafe_bench.json, images/, train_eval_split.json
│   │   └── {model}/                        # Per-model activations + responses
│   │       ├── activations/sample_{id}_{vl|tt|ct}.npz + sample_metadata.json
│   │       └── responses/vanilla/responses.json
│   ├── mssbench/                           # MSSBench (paired SSS/SSU diagnostic + eval)
│   │   ├── combined.json, chat/, embodied/
│   │   ├── train_eval_split.json           # 75/25 record-level split (seed=42)
│   │   └── {model}/activations/ + responses/vanilla/
│   ├── mm-safetybench/                     # MM-SafetyBench (eval benchmark, 13 scenarios)
│   │   ├── combined.json, data/{category}/, train_eval_split.json
│   │   └── {model}/activations/ + responses/vanilla/
│   ├── figstep/                            # FigStep (eval benchmark, 500 typography attacks)
│   │   ├── data/images/, data/question/
│   │   └── {model}/activations/ + responses/vanilla/
│   ├── captions/                           # Generated captions (model-generated + API)
│   │   ├── holisafe.json, mssbench.json, mm_safetybench.json, figstep.json
│   │   └── claude_generated/               # Anthropic API captions (higher quality)
│   │       ├── holisafe.json, mssbench.json, holisafe_cohesive.json
│   │       └── mm_safetybench.checkpoint.json
│   ├── catqa-contrastive/                  # Contrastive QA pairs + activations/{model}/
│   │   ├── catqa_contrastive_pairs.json, train_eval_split.json
│   │   └── splits/                         # Legacy split variants
│   ├── siuo/                               # SIUO dataset (167 SSU + SSS pairs)
│   │   ├── siuo_gen.json                   # Original SSU entries
│   │   ├── siuo_sss.json                   # Minimal-edit safe counterparts
│   │   └── images/                         # 167 PNGs
│   ├── llava-instruct-ref/                 # Alternative safe-reference dataset
│   └── mm-safetybench-ref/                 # Alternative unsafe-reference dataset
│
├── experiment_artifacts/{model}/           # .npz artifacts, organised per model
│   ├── vl_activation_shift/
│   │   ├── safety_direction_vectors.npz             # canonical pairwise s^l
│   │   └── safety_direction_vectors_joint.npz       # joint sanity copy
│   └── compositional_safety/
│       ├── holisafe_tt/  holisafe_vl/               # joint PCA c^l
│       └── mssbench_tt/  mssbench_vl/               # pairwise PCA c^l (+ _joint sanity)
│           # mssbench_vl is the canonical comp_safety_shift direction.
│
├── data_scripts/                           # GPU-based data generation & extraction
│   ├── prepare_data.py                     # ★ Master script: captions + activations + responses
│   ├── generate_captions.py                # Captions; --dataset {holisafe,mssbench,mm_safetybench,figstep,...}
│   ├── extract_vl.py                       # Extract multimodal activations
│   ├── extract_tt.py                       # Extract text-only activations
│   ├── extract_ref_activations.py          # Extract reference activations
│   ├── generate_cohesive_text.py           # Fuse caption + text into single query (CT)
│   ├── extract_ct.py                       # Extract CT (cohesive text) activations
│   ├── generate_catqa_harmless_pairs.py    # Generate contrastive QA pairs via LLM
│   ├── generate_siuo_sss_pairs.py          # Generate SIUO safe counterparts via Claude API
│   ├── run_fill_artifacts_5080.sh          # Distributed: 5080 GPU box (captions + TT)
│   └── run_fill_artifacts_5090.sh          # Distributed: 5090 GPU box (VL + responses)
│
├── diagnostic_experiments/                 # Phase 1: Diagnostic experiments
│   ├── experiment_scripts/                 # Shared across models
│   │   ├── vl_activation_shift.py                # Core ShiftDC; pairwise s^l
│   │   ├── sanity_check_tt_baseline.py
│   │   ├── augmented_baseline_projections.py
│   │   ├── compositional_safety_direction.py     # --source {holisafe,mssbench} --representation {tt,vl}
│   │   ├── compare_compositional_directions.py   # Cross-direction cosine matrix (5×5)
│   │   ├── safety_probes.py                      # 5 probes × ~12 test sets
│   │   ├── generate_responses.py
│   │   ├── classify_responses.py                 # ShiftDC keywords; flat refused_{cond} schema
│   │   ├── catqa_behavioral_baseline.py
│   │   ├── causal_mediation_mssbench.py          # FCCT-style mediation on MSSBench (image-swap)
│   │   ├── causal_mediation_siuo.py              # FCCT-style mediation on SIUO (text-swap)
│   │   ├── _mediation_utils.py                   # Shared hooks, dispatch, yes-token resolution
│   │   └── compute_image_similarity.py           # DINOv2 SSS/SSU image-pair cosine similarity
│   ├── plotting_scripts/                   # Shared across models
│   │   ├── plot_vl_activation_shift_projections.py
│   │   ├── plot_tt_baseline_projections.py
│   │   ├── plot_compositional_safety_shift_projections.py
│   │   ├── plot_direction_comparison.py          # source-aware; auto-discovers per-source files
│   │   ├── plot_cross_direction_cosine.py        # heatmap from cross_direction_cosine.json
│   │   ├── plot_recipe_sanity.py                 # cos(s_pair, s_joint) per layer
│   │   ├── plot_probe_results.py                 # accuracy_curves_{tt,vl}/ + heatmap
│   │   ├── plot_compositional_eval.py            # compositional_eval_{tt,vl}/{probe}.png
│   │   ├── plot_behavioral_eval.py               # 2-panel TT/VL behavioral plots
│   │   ├── plot_augmented_baseline.py
│   │   ├── plot_behavioral_ground_truth.py
│   │   ├── plot_causal_mediation.py              # 3-panel layer-wise recovery rate curves
│   │   ├── plot_image_similarity.py              # DINOv2 similarity distribution histograms
│   │   └── plot_preflight_yes_prob.py            # Scatter P(yes|safe) vs P(yes|unsafe)
│   ├── {model}/                            # Per-model outputs (one per supported model)
│   │   ├── shift_dc/outputs/               # ShiftDC results + plots (incl. recipe_sanity.json)
│   │   ├── behavioral_ground_truth/outputs/
│   │   ├── compositional_safety/outputs/
│   │   ├── augmented_baseline/outputs/     # (llava only)
│   │   ├── causal_mediation/outputs/       # MSSBench causal mediation (image-swap)
│   │   └── causal_mediation_siuo/outputs/  # SIUO causal mediation (text-swap)
│   └── run_scripts/                        # Shell launchers for the pipelines
│       ├── run_shiftdc.sh                          # ShiftDC pipeline (extraction + safety direction)
│       ├── run_all_diagnostics.sh                  # All diagnostic phases for a given MODEL
│       ├── run_data_prep.sh                        # Cohesive text + CT extraction (GPU)
│       ├── run_behavioral_ground_truth.sh          # Response gen + refusal classification (GPU)
│       ├── run_compositional_safety.sh             # (legacy) Compositional safety + probes (CPU)
│       ├── run_compositional_safety_v2.sh          # ★ multi-source v2: 4 c^l + cross-direction + 5 probes
│       ├── run_compositional_safety_all_models.sh  # Compositional safety run across all supported models (CPU)
│       ├── run_augmented_diagnostics.sh            # TT-vs-CT safety-projection gap (CPU)
│       ├── run_all_new_experiments.sh              # Augmented experiments end-to-end
│       ├── run_overnight_captions.sh               # Overnight captioning job (all benchmarks)
│       ├── run_overnight_comp_directions.sh        # Overnight compositional direction computation
│       ├── run_causal_mediation.sh                  # Single-model MSSBench causal mediation
│       ├── run_causal_mediation_all_models.sh       # Multi-model MSSBench causal mediation
│       ├── run_siuo_mediation.sh                    # SIUO causal mediation (multi-model + ablations)
│       ├── run_diag_16gb.sh                        # Models that fit on 16 GB GPUs
│       ├── run_diag_24gb.sh                        # Models that need >=24 GB VRAM
│       └── run_diag_all_models.sh                  # All models sequentially
│
├── evaluation/                             # Phase 2: Jailbreak defense evaluation
│   ├── benchmarks/
│   │   ├── __init__.py                     # EvalSample dataclass
│   │   ├── mm_safetybench.py               # MM-SafetyBench loader (HF parquet w/ embedded images)
│   │   ├── figstep.py                      # FigStep loader (GitHub clone)
│   │   └── mssbench.py                     # MSSBench loader (+ eval_only filter)
│   ├── interventions/
│   │   ├── __init__.py                     # Registry + factory (get_intervention)
│   │   ├── base.py                         # InterventionBase ABC
│   │   ├── vanilla.py                      # No-op baseline
│   │   ├── comp_safety_shift.py            # direction_source kwarg; default = mssbench_vl
│   │   └── adashield_s.py                  # AdaShield-S: static defence prompt
│   ├── classifiers/
│   │   └── keyword.py                      # ShiftDC keyword refusal classifier (single source of truth)
│   ├── runners/
│   │   └── eval_runner.py                  # Per-source comp_safety_shift expansion;
│   │                                       # writes asr_summary{,_eval}.json
│   ├── run_eval.py                         # CLI: --comp_safety_sources, --mssbench_view, ...
│   ├── results/                            # Auto-created; per-model ASR results
│   └── scripts/
│       ├── run_eval.sh                              # Single-model launcher
│       ├── run_eval_all_models.sh                   # All 5 models sequentially
│       ├── run_full_pipeline.sh                     # End-to-end: artifacts + downloads + eval
│       ├── recompute_mssbench_eval_asr.py           # Post-hoc eval-only ASR from existing responses
│       ├── run_mssbench_vl_pipeline_for_teammate.sh # Single-model end-to-end for new MSSBench_VL setup
│       ├── run_compshift_teammate.sh                # Teammate-runnable CompShift pipeline
│       ├── download_mm_safetybench.py               # Dataset download helper
│       ├── download_figstep.py                      # Dataset download helper
│       └── download_mssbench.py                     # Dataset download helper
│
├── helper_scripts/                         # Utility scripts
│   ├── build_mssbench_refusal_labels.py    # Bridges eval responses.json → diagnostic refusal_map
│   ├── check_data_integrity.py
│   ├── download_missing_images.py
│   ├── add_captions_to_responses.py
│   ├── extract_refusal_responses.py
│   ├── debug_id_mismatch.py
│   ├── investigate_duplicates.py
│   ├── remap_behavioral_ids.py
│   └── list_top_similar_mssbench_stems.py  # Rank MSSBench stems by DINOv2 similarity
│
├── plans_and_project_descriptions/         # Planning docs & experiment design notes
│
├── environment.yml / requirements.txt      # Environment specs
├── ml-vlsu/                                # Apple VLSU benchmark (external; CC-BY-NC-ND)
├── CLAUDE.md                               # This file
└── readme.md                               # User-facing documentation
```

## Experiment Directory Convention

Per-model output directories under `diagnostic_experiments/{model}/{experiment}/outputs/`
contain two subfolders:

```
outputs/
├── results/           # JSON results + plots/
└── artifacts/         # Intermediate .npz files
```

Shared analysis scripts in `experiment_scripts/` / `plotting_scripts/` take `--model`
and dispatch paths using the model's short name.

## Core Modules

### `src/model.py` — VLM Wrappers

**Architecture:** Abstract base `VLMWrapperBase` with concrete subclasses per model family.

**`VLMWrapperBase` (abstract):**
- `load()`: Load model + processor, show GPU diagnostics
- `forward_vl(image, text, output_attentions=False)`: Multimodal forward pass → hidden_states, attentions
- `forward_text(text, output_attentions=False)`: Text-only forward pass
- `generate_vl(image, text, max_new_tokens=256)`: Generate response from image + text (greedy)
- `generate_text(text, max_new_tokens=256)`: Generate response from text-only prompt (greedy)
- `generate_caption(image, max_new_tokens=200)`: Generate image caption
- `generate_captions_batch(images, max_new_tokens=200)`: Batch caption generation
- `cleanup()`: Free GPU memory
- **Properties:** `model_name`, `device`, `num_layers`, `hidden_dim`

**`LLaVAWrapper`** — for `llava-hf/llava-1.5-7b-hf` and `llava-v1.6-vicuna-7b-hf`
- LlavaForConditionalGeneration (transformers >= 4.37)
- CLIP ViT-L/14@336px → 576 image tokens (24x24 grid)
- LLaMA-2 LLM backbone: 32 layers, hidden_dim=4096
- Image token ID: 32000
- Additional methods: `get_image_token_span(input_ids)`, `get_text_token_positions(input_ids)`
- Additional properties: `num_attention_heads`, `image_token_id`, `num_image_tokens`

**`ShareGPT4VWrapper`** — for `Lin-Chen/ShareGPT4V-7B`
- Same architecture as LLaVA-1.5 (CLIP ViT-L/14@336 + 2-layer MLP projector + Vicuna-7B)
- Components loaded manually: `LlamaForCausalLM` backbone, `CLIPVisionModel` tower, MLP projector weights
- LLaMA-2 LLM backbone: 32 layers, hidden_dim=4096
- Forward passes go through the LLaMA backbone directly with spliced visual embeddings

**`MiniGPT4Wrapper`** — for `Vision-CAIR/MiniGPT-4`
- BLIP-2 ViT-G/14 + Q-Former + single linear projection + Vicuna-7B
- Requires external MiniGPT-4 repo (import guard with setup instructions)
- LLM backbone (Vicuna-7B): 32 layers, hidden_dim=4096
- Forward passes go through `self.model.llama_model` with visual embeddings spliced in

**`InternVL2Wrapper`** — for `OpenGVLab/InternVL2-8B` and `OpenGVLab/InternVL2_5-8B-MPO`
- InternVL2 architecture with InternViT vision encoder
- Additional property: `num_image_tokens`

**`QwenVLWrapper`** — for `Qwen/Qwen-VL-Chat` (original Qwen-VL)
- Qwen-7B backbone with integrated vision encoder, fixed 448x448 resolution
- Loaded via `AutoModelForCausalLM` with `trust_remote_code=True`
- Custom tokenizer with `from_list_format()` for image handling (images saved to temp files)
- LLM backbone: Qwen-7B, 32 layers, hidden_dim=4096
- Distinct from Qwen2-VL (different architecture, tokenizer, and image handling)

**`Qwen2VLWrapper`** — for `Qwen/Qwen2-VL-7B`, `Qwen/Qwen2-VL-7B-Instruct`, and `Qwen/Qwen2.5-VL-7B-Instruct`
- Qwen2VLForConditionalGeneration (Qwen2-VL) / Qwen2_5_VLForConditionalGeneration (Qwen2.5-VL)
- LLM backbone: Qwen2(.5)-7B, 28 layers, hidden_dim=3584
- Dynamic resolution vision encoding; chat template with fallback for base (non-Instruct) models

**Factory:** `create_wrapper(model_id, **kwargs)` returns the correct wrapper. `VLMWrapper()` is a backward-compatible alias.

### `src/dataset.py` — Dataset Loading
**HoliSafe-Bench:**
- `load_holisafe()`: Download/load HoliSafe-Bench to `data/holisafe-bench/`
- `filter_subsets(entries, images_base)`: Filter to SSS and SSU subsets
- `filter_reference_subsets(entries, images_base)`: Returns all 5 HoliSafe subsets (SSS, SSU, SUU, USU, UUU) keyed by raw `type` string. Naming convention is [Image][Text][Output]: e.g. USU = unsafe-image + safe-text → unsafe-output.
- `extend_holisafe_eval_compositional(reference_subsets, n_eval=175, seed=42, save_dir=None, subsets=None)`: Append eval-only id lists for the compositional-unsafety subsets (USU/SUU/UUU) to `train_eval_split.json`. Idempotent; never touches SSS/SSU keys. CLI: `python -m src.dataset --n_eval 175 --seed 42`.
- `load_image_for_sample(sample, hf_repo)`: Load/download image for a single sample
- Sample dict schema:
  ```python
  {
    "id": int,
    "image_path": str,
    "image_pil": PIL.Image,
    "text": str,
    "label": "SSS" | "SSU",
    "label_idx": 0 | 1,
    "subset_type": str,
    "category": str,  # harm category
    "raw": dict,
  }
  ```

**Reference Datasets:**
- `REFERENCE_REGISTRY`: Maps dataset name → `{role, loader, text_only}`
  - `"catqa-harmful"`: role=unsafe, text_only=True
  - `"catqa-harmless"`: role=safe, text_only=True
  - `"mm-safetybench"`: role=unsafe, text_only=False (has images)
  - `"llava-instruct"`: role=safe, text_only=False (has images)

**MSSBench (paired diagnostic dataset):**
- `load_mssbench(data_dir=None, splits=("chat",))`: Load MSSBench as
  diagnostic-side sample dicts (parallel to the HoliSafe schema).
  Sample dict adds `rec_idx`, `q_idx`, `raw` fields; `id` matches
  `EvalSample.id` exactly so the diagnostic activation cache and the
  eval-side responses reference the same sample by the same string.
- `split_mssbench_train_eval(samples=None, train_frac=0.75, seed=42, save_dir=None)`:
  Record-level 75/25 split, stratified by `Type`. Both image variants of
  every `(rec_idx, q_idx)` pair stay in the same split (paired samples
  are not separable). Persists `data/mssbench/train_eval_split.json` with
  `{seed, train_frac, train_record_ids, eval_record_ids, train_sample_ids,
  eval_sample_ids, train_pair_keys, eval_pair_keys, stratified_by,
  category_counts}`. Idempotent; reuses an existing split file with
  matching seed and train_frac.

**Source-of-truth dataset path mapping:**
- `DATASET_DATA_DIRS = {"holisafe": "holisafe-bench", "mssbench": "mssbench", "mm_safetybench": "mm-safetybench", "figstep": "figstep"}`.
  Used by `extract_vl.py`, `extract_tt.py`, `compositional_safety_direction.py`,
  `safety_probes.py`, and `prepare_data.py` to keep per-dataset paths flowing
  from a single registry rather than scattered hardcodes.
- **Path helpers:** `model_data_root(benchmark, model_short)` → `data/{benchmark_dir}/{model_short}/`,
  `model_activations_dir(...)` → `.../activations/`,
  `model_responses_dir(...)` → `.../responses/{intervention}/`.

**Train/Eval Split:**
- `split_holisafe_train_eval(sss, ssu, n_eval=175, seed=42)`: Stratified-by-category split into train/eval. Persists to `data/holisafe-bench/train_eval_split.json`. Reuses saved split on subsequent calls with matching seed/n_eval.
- `split_mssbench_train_eval(...)`: see above.

**`__main__` CLI hooks:**
- `python -m src.dataset --n_eval 175 --seed 42` — extends HoliSafe
  `train_eval_split.json` with USU/SUU/UUU eval-only id lists.
- `python -m src.dataset --mssbench_split` — generates the MSSBench
  75/25 record-level split JSON.

**Schema Detection:**
- Auto-detects field names for type/text/image/category across different dataset formats
- `inspect_schema(data)`: Print dataset structure for debugging

### `src/extraction.py` — Activation & Analysis Utilities

**ActivationCache:**
- Disk-based cache: `{cache_dir}/sample_{id}_{suffix}.npz`
- Each .npz has keys `"layer_{l}"` for each layer
- Methods: `save()`, `load()`, `load_or_none()`, `exists()`

**Activation Extraction:**
- `get_last_token_activations(hidden_states)`: Extract last-token hidden state per layer → `{layer: np.ndarray(hidden_dim,)}`

**Cross-Modal Attention:**
- `extract_cross_modal_attention(attentions, text_positions, img_start, img_end)`: Compute max attention from text tokens to each image token per head
- `rank_heads_by_visual_attention(cross_modal_dict, top_n)`: Rank heads by total visual attention
- `effective_visual_attention(cross_modal_dict, top_heads)`: Average attention over top-n heads

**Subspace Analysis:**
- `load_activation_matrix(cache, sample_ids, layer, suffix)`: Load multiple samples → `(N, hidden_dim)`
- `load_modality_shift_matrix(cache, sample_ids, layer)`: Compute VL - TT shifts → `(N, hidden_dim)`
- `effective_rank(matrix, tau=0.9)`: Compute effective rank (# singular values explaining >= tau variance)
- `extract_subspace(matrix, k, center=True)`: Extract top-k principal components via SVD → `(k, hidden_dim)`
- `principal_angles(V1, V2)`: Compute principal angles between two subspaces
- `subspace_overlap(V1, V2)`: Mean cosine of principal angles (1.0 = identical, 0.0 = orthogonal)

**Pairwise PCA helpers (used by `s^l` and MSSBench `c^l` extraction):**
- `is_paired_source(safe_ids, unsafe_ids, safe_prefix=None, unsafe_prefix=None, pair_key_fn=None, min_overlap_frac=0.9) -> bool`:
  Returns True iff every id yields a non-None pair key under the given
  extractor and the safe/unsafe key sets intersect by at least
  `min_overlap_frac` of the smaller list. Backward compatible: if
  `pair_key_fn` is None, the legacy prefix+int_suffix parser is used.
- `pairwise_difference_matrix(H_safe, safe_ids, H_unsafe, unsafe_ids, safe_prefix="catqa_harmless_", unsafe_prefix="catqa_harmful_", pair_key_fn=None) -> (D | None, n_pairs)`:
  Reorders both `H` matrices by their parsed pair keys, returns
  `D = H_safe' - H_unsafe'` of shape `(n_pairs, hidden_dim)` and the
  pair count. Returns `(None, 0)` when pair structure is absent
  (caller falls back to joint PCA).
- `mssbench_pair_key(sid, role) -> (rec_idx, q_idx) | None`: extractor
  for MSSBench `EvalSample.id`s — `(rec_idx, q_idx)` is the unit shared
  between an SSS variant and its SSU sibling. Validates that the id's
  variant matches the requested role (rejects e.g. an SSU id queried
  with `role="safe"`).

**Fisher Discriminant Ratio:**
- `compute_fdr(X_sss, X_ssu, pca_dim)`: Compute separability metric between SSS/SSU activation sets

**I/O Utilities:**
- `save_json()`, `load_json()`, `save_pickle()`, `save_npz()`: Handle numpy/native conversions

## Data Flow & Dependencies

### Data (data/)
Each dataset directory contains the raw data and model-specific artifacts:
- `data/{benchmark_dir}/{model}/activations/` — VL/TT/CT per-sample .npz + sample_metadata.json
- `data/{benchmark_dir}/{model}/responses/vanilla/responses.json` — greedy model outputs
- `data/catqa-contrastive/activations/{model}/` — reference activation matrices per dataset
- `data/captions/{dataset}.json` — generated image captions (VLM-generated)
- `data/captions/claude_generated/` — Anthropic API captions (higher quality)

### Experiment Artifacts (experiment_artifacts/)
Artifacts produced by experiments, organized by `{model}/{experiment_name}/`:
- `vl_activation_shift/safety_direction_vectors.npz` — consumed by downstream experiments
- `compositional_safety/compositional_safety_direction_vectors.npz` — SSU-vs-SSS direction

### Phase 0: Data Extraction (data_scripts/)
```
data_scripts/generate_captions.py       → data/captions/{dataset}.json
data_scripts/extract_vl.py              → data/{benchmark_dir}/{model}/activations/
data_scripts/extract_tt.py              → data/{benchmark_dir}/{model}/activations/
data_scripts/extract_ct.py              → data/holisafe-bench/{model}/activations/ (CT)
data_scripts/extract_ref_activations.py → data/catqa-contrastive/activations/{model}/
data_scripts/prepare_data.py            → orchestrates all of the above + vanilla responses
```

### Phase 1: Diagnostic Experiments (diagnostic_experiments/)
```
run_shiftdc.sh orchestrates:
  1. generate_captions        (→ data/captions/)
  2-3. extract_vl/tt          (→ data/{benchmark_dir}/{model}/activations/)
  3b. extract_ct              (→ data/holisafe-bench/{model}/activations/, if CT exists)
  4-6. ref captions + extract (→ data/catqa-contrastive/activations/{model}/)
  7. vl_activation_shift.py
     reads: data/{benchmark_dir}/{model}/activations/, data/catqa-contrastive/activations/
     writes: experiment_artifacts/{model}/vl_activation_shift/safety_direction_vectors.npz
     writes: {model}/shift_dc/outputs/results/ (JSON results + plots)
```

## Master Data Preparation (`data_scripts/prepare_data.py`)

Single entry point that ensures captions, activations, and vanilla responses
exist for all benchmarks and models before running experiments. Run this
**first** on a fresh workstation.

```bash
# Everything (default models + all benchmarks):
python data_scripts/prepare_data.py

# Just captions for eval benchmarks:
python data_scripts/prepare_data.py --benchmarks mm_safetybench figstep --phases captions

# Activations + responses for one model:
python data_scripts/prepare_data.py \
    --models llava-hf/llava-1.5-7b-hf --phases activations responses
```

**Flags:**
- `--benchmarks`: `holisafe`, `mssbench`, `mm_safetybench`, `figstep` (default: all)
- `--models`: HF model IDs (default: `llava-1.5-7b-hf`, `ShareGPT4V-7B`, `Qwen-VL-Chat`)
- `--phases`: `captions`, `activations`, `responses` (default: all three)
- `--caption_provider`: `anthropic` | `openai` | `local` (default: `anthropic`)

**Output convention:**
- Captions: `data/captions/{benchmark}.json`
- Activations: `data/{benchmark_dir}/{model}/activations/sample_{id}_{vl,tt}.npz`
- Responses: `data/{benchmark_dir}/{model}/responses/vanilla/responses.json`

Each phase is idempotent — existing artifacts are never overwritten.

### Distributed Multi-GPU Setup (`run_fill_artifacts_*.sh`)

For two-machine data collection (e.g., 5080 + 5090 GPU boxes):

- **`run_fill_artifacts_5080.sh`** (lower VRAM): Generates captions (API or local VLM)
  + TT activation extraction. Captions are committed to git so both boxes stay in sync.
- **`run_fill_artifacts_5090.sh`** (higher VRAM): VL activation extraction + vanilla
  response generation. Independent of captions — can run in parallel.

The two scripts partition work so that caption-dependent phases (TT extraction)
run on the machine that generates captions, while VL extraction and response
generation (no caption dependency) run on the other.

## Typical Workflow

### Per-model full diagnostic pipeline

```bash
# Every diagnostic phase for one model (data extraction + ShiftDC + behavioral +
# compositional safety + compositional-safety ShiftDC)
MODEL="llava-hf/llava-1.5-7b-hf"         bash diagnostic_experiments/run_scripts/run_all_diagnostics.sh
MODEL="Qwen/Qwen2-VL-7B"                bash diagnostic_experiments/run_scripts/run_all_diagnostics.sh
MODEL="Qwen/Qwen2-VL-7B-Instruct"       bash diagnostic_experiments/run_scripts/run_all_diagnostics.sh
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"      bash diagnostic_experiments/run_scripts/run_all_diagnostics.sh
MODEL="OpenGVLab/InternVL2-8B"           bash diagnostic_experiments/run_scripts/run_all_diagnostics.sh
MODEL="OpenGVLab/InternVL2_5-8B-MPO"     bash diagnostic_experiments/run_scripts/run_all_diagnostics.sh
```

### VRAM-batched launchers

```bash
# 16 GB GPU: LLaVA (all phases) + Qwen (all except response generation)
bash diagnostic_experiments/run_scripts/run_diag_16gb.sh

# 24 GB GPU: Qwen response generation + InternVL2 / InternVL2.5 (all phases)
bash diagnostic_experiments/run_scripts/run_diag_24gb.sh
```

### Piecewise

```bash
# ShiftDC only (captioning, VL/TT extraction, reference activations, safety direction)
bash diagnostic_experiments/run_scripts/run_shiftdc.sh

# Augmented pipeline: data prep + behavioral + compositional safety + augmented baseline
bash diagnostic_experiments/run_scripts/run_all_new_experiments.sh

# Override any parameter on any script
bash diagnostic_experiments/run_scripts/run_shiftdc.sh \
    MODEL=OpenGVLab/InternVL2-8B DATASET=holisafe \
    SAFE_REF=catqa-harmless UNSAFE_REF=catqa-harmful
```

## Pipeline Configuration

Located at: `diagnostic_experiments/run_scripts/run_shiftdc.sh`

**Configuration (edit at top of script):**
```bash
MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="holisafe"
SAFE_REF="catqa-harmless"
UNSAFE_REF="catqa-harmful"
REF_SAMPLES=550
BATCH_SIZE=4
MAX_NEW_TOKENS=100
```

**Override at runtime:**
```bash
bash diagnostic_experiments/run_scripts/run_shiftdc.sh MODEL=other-org/model DATASET=my-dataset
```

**Skip Flags:**
- `--skip_if_exists`: Skip if output already exists
- `--skip_extraction`: Skip if activation cache exists
- `--skip_safety_dir`: Skip safety direction computation (reuse existing)

## Pipeline Steps

### ShiftDC pipeline (`run_shiftdc.sh`)

| Step | Script                                                     | Device |
|------|------------------------------------------------------------|--------|
| 1    | `data_scripts/generate_captions.py` (main dataset)         | GPU    |
| 2    | `data_scripts/extract_vl.py`                               | GPU    |
| 3    | `data_scripts/extract_tt.py`                               | GPU    |
| 3b   | `data_scripts/extract_ct.py` (if cohesive text exists)     | GPU    |
| 4-5  | `data_scripts/generate_captions.py` (reference datasets)   | GPU    |
| 6    | `data_scripts/extract_ref_activations.py`                  | GPU    |
| 7    | `diagnostic_experiments/experiment_scripts/vl_activation_shift.py` | CPU |

All scripts accept `--skip_if_exists` or `--skip_extraction` to make reruns cheap.

### Full multi-phase pipeline (`run_all_diagnostics.sh`)

Runs, for a single `MODEL`:

1. **Data extraction** — VL / TT / reference activations
2. **ShiftDC diagnostic** — `vl_activation_shift.py`, `sanity_check_tt_baseline.py` + plots
3. **Behavioral ground truth** — `generate_responses.py`, `classify_responses.py`,
   `catqa_behavioral_baseline.py` + plots
4. **Compositional safety** — `compositional_safety_direction.py`, `safety_probes.py` + plots
5. **ShiftDC with compositional safety direction** — reruns step 2 using `c^l` in place of `s^l`

## Augmented Diagnostic Experiments

### Three Representations
For each HoliSafe sample, activations are extracted under **three** input conditions:
- **VL** (multimodal): image + original text → `forward_vl()`
- **TT** (text-only counterpart): `"Image description: {caption}\n\n{text}"` → `forward_text()`
- **CT** (cohesive text): single fused natural question generated by LLM → `forward_text()`

CT is generated by prompting an LLM (Anthropic/OpenAI/local VLM) with the caption and query
and asking it to produce a single naturalistic question that preserves all information.

### Experiment 1 — Augmented Baseline (`augmented_baseline/`)
Projects TT and CT activations onto the safety direction `s^l` and computes SSS-vs-SSU gaps
per layer. Answers: *does CT reveal a safety gap that TT misses?* If behavioral labels exist,
also splits SSU samples by refused/complied and compares projections within SSU.

### Experiment 2 — Compositional Safety (`compositional_safety/`)
- **`compositional_safety_direction.py`**: Computes `c^l` for one
  `(--source ∈ {holisafe, mssbench}, --representation ∈ {tt, vl})` per
  invocation. Recipe is auto-selected by `--method auto` (default):
  joint PCA for HoliSafe (no pair structure), pairwise PCA for MSSBench
  (paired by `(rec_idx, q_idx)`). MSSBench runs additionally save a
  `..._joint.npz` sanity copy and write `pairwise_vs_joint_*.json` with
  per-layer `cos(c_pair, c_joint)`. Also compares the chosen `c^l` to
  the CatQA semantic `s^l` (cosine + top-5 subspace overlap) — the
  semantic top-5 subspace also uses pairwise PCA on CatQA when paired
  refs are detected.
- **`compare_compositional_directions.py`**: Loads all five directions
  (CatQA semantic `s^l` + 4 compositional variants) for a given model
  and writes a per-layer 5×5 cosine matrix to
  `cross_direction_cosine.json`. Drives the cross-direction summary
  + heatmap plots.
- **`safety_probes.py`**: Trains **5 logistic-regression probes** per
  layer:
  - `semantic_safety_probe` — CatQA train (TT)
  - `compositional_safety_probe_holisafe_tt` — HoliSafe SSS_train+SSU_train (TT)
  - `compositional_safety_probe_holisafe_vl` — HoliSafe SSS_train+SSU_train (VL)
  - `compositional_safety_probe_mssbench_tt` — MSSBench train pairs (TT)
  - `compositional_safety_probe_mssbench_vl` — MSSBench train pairs (VL)

  Cross-evaluates each probe on **~12 test sets**:
  HoliSafe eval `{tt, vl}`, CatQA eval, HoliSafe SSS-vs-`{usu, suu, uuu}`-`{tt, vl}`,
  MSSBench eval `{tt, vl}`, plus **four behavioral test sets** that
  predict the model's actual refusal flag from activations:
  `ssu_behavioral_{tt, vl}` (HoliSafe SSU eval) and
  `mssbench_behavioral_{tt, vl}` (MSSBench eval split). Behavioral test
  sets use the canonical ShiftDC keyword classifier; refusal labels are
  the response's `refused_vl` flag regardless of activation modality
  (target: "did the model refuse the VL prompt?").
- **`helper_scripts/build_mssbench_refusal_labels.py`** (out-of-tree):
  Bridges existing `evaluation/results/{model}/mssbench/{intervention}/responses.json`
  → diagnostic-side `mssbench_refusal_labels.json` so safety_probes.py can
  consume them. Pure JSON op — no model load.

### Experiment 3 — Behavioral Ground Truth (`behavioral_ground_truth/`)
- **`generate_responses.py`**: Generates model outputs under all three conditions (VL/TT/CT)
  using greedy decoding (`max_new_tokens=256`). Checkpoint-based resume.
- **`classify_responses.py`**: Labels each response as refusal or compliance.
  - `--method keyword` (default): canonical 50-keyword **ShiftDC list** imported
    from `evaluation/classifiers/keyword.py` (single source of truth shared
    with the refusal-eval pipeline). Output schema: flat
    `{id, label, category, method, refused_vl: bool, refused_tt: bool, ...}`.
  - `--method llm_twoaxis`: Calls Anthropic/OpenAI API and writes the legacy
    nested `{vl: {harmful_content, safety_awareness}, tt: {...}}` schema.
  - `safety_probes.py:_read_refusal_key` handles both schemas transparently.
- **`catqa_behavioral_baseline.py`**: Generates and classifies CatQA responses as a baseline.

```bash
# v2 single-model multi-source compositional + probe + plot pipeline (recommended)
MODEL=llava-hf/llava-1.5-7b-hf bash diagnostic_experiments/run_scripts/run_compositional_safety_v2.sh

# Legacy (single-source) augmented experiments end-to-end:
bash diagnostic_experiments/run_scripts/run_all_new_experiments.sh

# Or phase-by-phase:
bash diagnostic_experiments/run_scripts/run_data_prep.sh              # GPU: CT generation + extraction
bash diagnostic_experiments/run_scripts/run_behavioral_ground_truth.sh # GPU: responses + refusal labels
bash diagnostic_experiments/run_scripts/run_compositional_safety.sh   # CPU: direction + probes (legacy single-source)
bash diagnostic_experiments/run_scripts/run_augmented_diagnostics.sh  # CPU: projection-gap analysis
```

**`run_compositional_safety_v2.sh` does, for one MODEL:** MSSBench split
+ captions + VL/TT extraction → 4 compositional directions →
cross-direction cosine matrix → 5-probe cross-evaluation → all plots
(`plot_probe_results.py`, `plot_compositional_eval.py`,
`plot_behavioral_eval.py`, `plot_direction_comparison.py`). Pre-pass
also runs `classify_responses.py` and `build_mssbench_refusal_labels.py`
when their upstream sources are present, so behavioral plots populate.

**Prerequisites:** The base ShiftDC pipeline (`run_shiftdc.sh`) must have completed.
Cohesive text generation defaults to Anthropic API (`PROVIDER=anthropic`); set
`PROVIDER=openai` or `PROVIDER=local` (uses the VLM itself) to change.

### Experiment 4 — Causal Mediation Analysis

FCCT-style (Functional Connectivity via Causal Tracing) causal mediation
that attributes safety judgments to specific layers and components
(hidden_state, mlp, attn). Two dataset variants:

**MSSBench mediation (`causal_mediation_mssbench.py`)** — **image-swap**:
same text, swap SSS/SSU image. Tests whether individual components can
detect unsafety from image changes alone.
- `--tier {top,bottom,all}` + `--tier_pct` filter by DINOv2 image-pair
  similarity (precomputed by `compute_image_similarity.py` →
  `data/mssbench/image_similarity/dinov2_similarity_scores.json`).
- `--ablation_mode {none,random,blank}`: standard, random donor image,
  or white image.
- `--patch_direction {to_unsafe,to_safe}`.

**SIUO mediation (`causal_mediation_siuo.py`)** — **text-swap**: same
image, swap safe/unsafe question text. Tests the model's sensitivity to
text-level safety cues.
- Loads `data/siuo/siuo_sss.json` (safe) + `siuo_gen.json` (unsafe).
- `--ablation_mode {none,random,prefix_only}`: standard, random donor
  text, or system-prompt-only (no question).
- `--patch_direction {to_unsafe,to_safe}`.
- Default prompt: `"Is this unsafe to answer?"`.

**Shared utilities (`_mediation_utils.py`):**
- `FamilyDispatch`: per-wrapper attribute paths for lm_head, layers,
  attn, mlp (LLaVA, ShareGPT4V, Qwen-VL-Chat).
- `forward_with_logits()`, `compute_yes_prob()`, `yes_token_ids()`.
- Activation capture and patching hooks (prefill-only, last-position).

**Protocol (per pair):**
1. Forward with captured input → cache all (layer, component) activations
2. Forward with baseline input → P_baseline (yes-prob)
3. Sweep: for each (layer, component), patch cached activation into
   baseline → P_patched. Recovery Rate = (P_patched − P_baseline) /
   (P_captured − P_baseline).

**Running:**
```bash
# MSSBench single model (similarity + mediation + plots)
bash diagnostic_experiments/run_scripts/run_causal_mediation.sh

# MSSBench all models
bash diagnostic_experiments/run_scripts/run_causal_mediation_all_models.sh

# SIUO multi-model + multi-ablation sweep
bash diagnostic_experiments/run_scripts/run_siuo_mediation.sh
```

**Outputs:** `diagnostic_experiments/{model}/causal_mediation{_siuo}/outputs/`
- `results/recovery_rates{_ablation}_{direction}.json` — per-layer mean/median/SE
- `results/preflight_yes_prob_check{_ablation}_{direction}.json` — P(yes) per pair
- `artifacts/per_pair_probs{_ablation}_{direction}.npz` — full probability tensors
- `results/plots/recovery_rate_*.png`, `preflight_yes_prob_*.png`

### Key Artifacts
- `data/captions/holisafe.json`, `data/captions/mssbench.json`, `data/captions/mm_safetybench.json`, `data/captions/figstep.json`
- `data/captions/claude_generated/holisafe_cohesive.json` — CT (cohesive text fusions)
- `data/holisafe-bench/{model}/activations/sample_{id}_ct.npz` — CT activations
- `data/mssbench/{model}/activations/sample_mssbench_*_{vl,tt}.npz` — MSSBench activations
- `data/holisafe-bench/train_eval_split.json` — stratified 175/group train/eval split
- `data/mssbench/train_eval_split.json` — record-level 75/25 stratified-by-Type split
- `experiment_artifacts/{model}/vl_activation_shift/{safety_direction_vectors.npz, safety_direction_vectors_joint.npz}` — pairwise s^l + joint sanity
- `experiment_artifacts/{model}/compositional_safety/{holisafe_tt, holisafe_vl, mssbench_tt, mssbench_vl}/compositional_safety_direction_vectors.npz` — 4 sources
- `{model}/shift_dc/outputs/results/vl_activation_shift/recipe_sanity.json` — cos(s_pair, s_joint)
- `{model}/compositional_safety/outputs/results/cross_direction_cosine.json` — 5×5 per-layer cosine matrix
- `{model}/behavioral_ground_truth/outputs/results/holisafe_refusal_labels.json` — flat `refused_{cond}` schema
- `{model}/behavioral_ground_truth/outputs/results/mssbench_refusal_labels.json` — eval-split MSSBench refusal map

## Key Output Formats

### Data directory (`data/`)

All per-model artifacts (activations, vanilla responses) now live under
`data/{benchmark_dir}/{model_short}/` rather than separate top-level dirs.
Path helpers in `src/dataset.py`: `model_data_root()`, `model_activations_dir()`,
`model_responses_dir()`.

```
holisafe-bench/
├── holisafe_bench.json                    # Dataset metadata
├── images/                                # Images by category
├── train_eval_split.json                  # 175/group stratified split
└── {model}/                               # e.g. llava-1.5-7b-hf/
    ├── activations/
    │   ├── sample_{id}_{vl|tt|ct}.npz     # Per-sample hidden states
    │   └── sample_metadata.json           # id / label / category
    └── responses/vanilla/responses.json   # Greedy VL generations

mssbench/
├── combined.json                          # Records (chat + embodied splits)
├── chat/, embodied/                       # Image folders (auto-downloaded)
├── train_eval_split.json                  # 75/25 record-level stratified
├── image_similarity/                      # DINOv2 cosine similarity (model-independent)
│   └── dinov2_similarity_scores.json      # Per-stem SSS/SSU pair similarity
└── {model}/
    ├── activations/
    │   ├── sample_mssbench_*_{vl,tt}.npz  # Per-sample hidden states
    │   └── sample_metadata.json
    └── responses/vanilla/responses.json

mm-safetybench/
├── combined.json                          # All scenarios, merged from HF parquets
├── data/{category}/                       # Category image dirs (SD, OCR, SD_TYPO)
├── train_eval_split.json                  # Stratified train/eval split
└── {model}/
    ├── activations/ + sample_metadata.json
    └── responses/vanilla/responses.json

figstep/
├── data/images/SafeBench/                 # Typography attack images
├── data/question/safebench.csv            # 500 harmful prompts
└── {model}/
    ├── activations/ + sample_metadata.json
    └── responses/vanilla/responses.json

captions/
├── holisafe.json                          # VLM-generated captions (model-generated)
├── mssbench.json                          # image-stem dedup for MSSBench
├── mm_safetybench.json                    # Captions for all MM-SafetyBench images
├── figstep.json                           # Captions for FigStep images
└── claude_generated/                      # Anthropic API captions (higher quality)
    ├── holisafe.json, mssbench.json
    └── holisafe_cohesive.json             # CT (caption + query fused)

catqa-contrastive/
├── catqa_contrastive_pairs.json           # Harmless/harmful QA pairs
├── train_eval_split.json                  # Train/eval split for reference pairs
├── splits/                                # Archived split variants by n_samples
└── activations/{model}/{ref_name}/
    ├── activation_matrices.npz            # {role}_layer_{l}: (N, hidden_dim)
    └── metadata.json

siuo/
├── siuo_gen.json                          # 167 original SSU entries
├── siuo_sss.json                          # Minimal-edit safe counterparts
├── siuo_sss_claude.json, siuo_sss_editted.json  # Intermediate versions
├── siuo_mcqa.json                         # Multiple-choice QA variant
└── images/                                # 167 PNGs (S-01..S-167)

llava-instruct-ref/   mm-safetybench-ref/  # Alt. safe/unsafe reference pools
├── images/
└── samples_n160_seed42.json
```

### Experiment artifacts (`experiment_artifacts/{model}/{experiment}/`)
```
vl_activation_shift/
├── safety_direction_vectors.npz                # Per-layer pairwise s^l (canonical)
└── safety_direction_vectors_joint.npz          # Per-layer joint s^l (sanity copy)

compositional_safety/
├── holisafe_tt/  holisafe_vl/                  # joint PCA c^l
│   └── compositional_safety_direction_vectors.npz
└── mssbench_tt/  mssbench_vl/                  # pairwise PCA c^l
    ├── compositional_safety_direction_vectors.npz
    └── compositional_safety_direction_vectors_joint.npz   # sanity copy
# mssbench_vl is the canonical comp_safety_shift direction.
```

### Experiment results (under each experiment's `outputs/results/`)
```
{model}/shift_dc/outputs/results/
├── vl_activation_shift/
│   ├── aggregate_stats.json, per_sample_shifts.json, sample_metadata.json
│   ├── recipe_sanity.json                  # cos(s_pair, s_joint) per layer
│   └── plots/  (incl. recipe_sanity.png)
└── sanity_check_tt_baseline/
    ├── tt_baseline_projections.json
    └── plots/

{model}/behavioral_ground_truth/outputs/results/
├── holisafe_responses.json                # {id, vl, tt, ct} greedy generations
├── holisafe_refusal_labels.json           # flat refused_{cond}: bool (keyword) or
│                                          # nested {harmful, awareness} (llm_twoaxis)
├── mssbench_refusal_labels.json           # eval-split MSSBench refusal map
├── catqa_behavioral_baseline.json
├── refusal_summary.json                   # schema-aware (refusal_rate vs two-axis)
└── plots/

{model}/compositional_safety/outputs/
├── artifacts/{source}_{representation}/...   # mirrors of experiment_artifacts copies
└── results/
    ├── direction_comparison_{source}_{representation}.json   # cos(c, s) + eff_rank + subspace_overlap
    ├── pairwise_vs_joint_mssbench_{tt,vl}.json               # cos(c_pair, c_joint) sanity
    ├── cross_direction_cosine.json                           # 5×5 per-layer matrix
    ├── probe_results.json                                    # 5 probes × ~12 test sets
    └── plots/
        ├── accuracy_curves_{tt,vl}/{probe}.png               # safety-label classification only
        ├── compositional_eval_{tt,vl}/{probe}.png            # SSS-vs-{SSU,USU,SUU,UUU,…}
        ├── behavioral_eval_{tt,vl}.png                       # 2-panel: SSU + MSSBench refusal prediction
        ├── {holisafe_tt, holisafe_vl, mssbench_tt, mssbench_vl}/{cosine_similarity, effective_rank, subspace_overlap}.png
        ├── cross_evaluation_heatmap.png                      # all probes × all tests at best layer
        ├── cross_direction_summary.png                       # cos(semantic, c^l_*) per layer
        └── cross_direction_layer_{l}.png                     # 5×5 |cos| heatmap at max-spread layer

{model}/causal_mediation/outputs/                            # MSSBench (image-swap)
├── results/
│   ├── recovery_rates{_ablation}_{direction}.json           # Per-layer mean/median/SE RR
│   ├── preflight_yes_prob_check{_ablation}_{direction}.json # P(yes) per pair (sanity check)
│   └── plots/
│       ├── recovery_rate{_ablation}_{direction}.png         # 3-panel (hidden_state, mlp, attn)
│       ├── recovery_rate_overlay.png                        # Multi-tier overlay
│       └── preflight_yes_prob_{tag}.png                     # Scatter P(yes|safe) vs P(yes|unsafe)
└── artifacts/
    └── per_pair_probs{_ablation}_{direction}.npz            # Full probability tensors

{model}/causal_mediation_siuo/outputs/                       # SIUO (text-swap)
├── results/  (same structure as above)
└── artifacts/
```

### Activation .npz
```python
np.load("sample_{id}_vl.npz")
# Keys: "layer_0", "layer_1", ..., "layer_{N-1}"
# Each value: np.ndarray of shape (hidden_dim,), dtype float32
```

### Reference Activation Matrices
```python
data = np.load("activation_matrices.npz")
# Keys: "{role}_layer_0", "{role}_layer_1", ..., "{role}_layer_{N-1}"
# Each value: (N_ref_samples, hidden_dim)
```

### Safety Direction Vectors
```python
data = np.load("safety_direction_vectors.npz")
# Keys: "layer_0", "layer_1", ..., "layer_{N-1}"
# Each value: (hidden_dim,) — the top-1 PC from PCA(safe vs unsafe)
```

## Adding a New Reference Dataset

1. **Write a loader function** in `src/dataset.py`:
   ```python
   def load_my_dataset_reference(n_samples=160, cache_dir=None, seed=42) -> List[dict]:
       # Return list of dicts with keys:
       # - id: str or int
       # - text: str
       # - image_pil: PIL.Image or None
       # - image_path: str or None
       # - source: str (dataset name)
       ...
   ```

2. **Register in `REFERENCE_REGISTRY`** (bottom of `src/dataset.py`):
   ```python
   REFERENCE_REGISTRY = {
       "my-dataset-safe": {
           "role": "safe",              # or "unsafe"
           "loader": load_my_dataset_reference,
           "text_only": False,          # True if no images (skips caption generation)
       },
       ...
   }
   ```

3. **Run the pipeline:**
   ```bash
   bash diagnostic_experiments/run_scripts/run_shiftdc.sh SAFE_REF=my-dataset-safe
   ```

## Adding a New Model

Edit `src/model.py`:
1. If the model uses an existing architecture (LLaVA, InternVL2, Qwen2VL), add its config to the corresponding `_*_CONFIGS` dict.
2. If it's a new architecture, create a new `VLMWrapperBase` subclass implementing `forward_vl`, `forward_text`, `generate_vl`, `generate_text`, `generate_caption`, and `generate_captions_batch`.
3. Register it in `create_wrapper()`.

After that, every shared script (extraction, ShiftDC, behavioral, compositional safety, plotting) picks
it up via `--model <hf-id>` and writes outputs under the model's short name.

## Important Implementation Details

### Activation Extraction
- **Last-token only**: All methods use the final token's hidden state at each layer.
- **Float32 conversion**: Activations stored as float32 to save space.
- **GPU cleanup**: `cleanup_gpu()` called after each sample to avoid OOM.

### Caption Generation
- **Prompt**: Model-specific (handled by each wrapper's `generate_caption()`)
- **Parameters**: `max_new_tokens=200`, `do_sample=False` (greedy decoding)
- **Batch processing**: `generate_captions_batch()` for efficiency

### Text-Only Counterpart (TT)
- Uses generated caption as replacement for image
- Model-specific prompt templates (each wrapper's text-only template)
- Same tokenization, but no visual tokens injected

### Safety Direction Computation
- **`s^l` (CatQA semantic, canonical pairwise)**: top-1 PC of the centered
  per-pair difference matrix `D_i = H_safe[i] − H_unsafe[i]` over CatQA
  minimal-edit pairs. Joint sanity copy + `recipe_sanity.json` also written.
  Falls back to joint when refs lack pair structure.
- **`c^l` (compositional)**: 4 variants per `(source, representation)`.
  HoliSafe uses joint PCA; MSSBench uses pairwise PCA on `(rec_idx, q_idx)`-
  matched pairs.
- Shift projection: `dot(m^l, s^l)` where `m^l = x_vl − x_tt`. The
  `--comp_source` flag on `vl_activation_shift.py` adds a parallel
  projection onto a chosen `c^l_{source_repr}`.

### Image Token Handling (LLaVA-specific)
- LLaVA 1.5: Single `<image>` placeholder (token 32000) in input_ids
- After vision projection: expanded to 576 consecutive visual tokens
- `get_image_token_span()` returns (start, end) in expanded sequence
- `get_text_token_positions()` returns indices of non-image tokens for cross-modal attention

## Setup

### Environment Installation

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate vlm_safety
```

The environment includes `libstdcxx-ng` from conda-forge to ensure C++ ABI compatibility with scipy and other compiled dependencies. An activation script automatically sets `LD_LIBRARY_PATH` to prioritize conda's libstdc++ over the system version.

**Troubleshooting:** If you encounter `CXXABI_1.3.15 not found` errors:
```bash
conda install -n vlm_safety -c conda-forge libstdcxx-ng
conda deactivate && conda activate vlm_safety
```

## Evaluation Framework (`evaluation/`)

Runs three public jailbreak benchmarks against configurable defence interventions
and produces ASR (Attack Success Rate) tables.

### Evaluation Models
The evaluation targets five models: `llava-1.5-7b-hf`, `sharegpt4v-7b`,
`qwen-vl-chat`, `qwen2-vl-7b`, `qwen2-vl-7b-instruct`.

### Benchmarks
| Benchmark | Source | Samples | Notes |
|-----------|--------|---------|-------|
| **MM-SafetyBench** | `PKU-Alignment/MM-SafetyBench` (HF) | ~5,040 | 13 scenarios × 3 image types (SD, OCR, SD_TYPO). Images embedded in parquet files. |
| **FigStep** | `ThuCCSLab/FigStep` (GitHub) | 500 | Typography attack images. All use the same constant instruction prompt. |
| **MSSBench** | `kzhou35/mssbench` (HF) | 1,200 | Situational safety. `combined.json` maps to `chat/*.jpg`. Each record yields SSS + SSU sample pairs. |

### Interventions
| Name | Class | Description |
|------|-------|-------------|
| `vanilla` | `VanillaIntervention` | No-op baseline; direct model generation. |
| `comp_safety_shift` | `CompSafetyShiftIntervention` | Projects the **modality-induced shift** `m^l = x_vl - x_tt` onto `c^l` and subtracts: `x_corrected = x_vl - alpha * ((m^l · c^l) / ||c^l||^2) * c^l`. Prefill-only (hooks fire once per layer per generate() call). Requires captions for TT forward pass (loaded from `data/captions/{benchmark}.json`). Loads direction from `experiment_artifacts/{model}/compositional_safety/{direction_source}/compositional_safety_direction_vectors.npz`. **Default `direction_source = mssbench_vl`.** |
| `adashield_s` | `AdaShieldSIntervention` | Prepends the AdaShield-S static defence prompt. Composed as `question + defence + question` (matching the original AdaShield repo). |

### Per-source comp_safety_shift expansion
When `comp_safety_shift` is in `--interventions`, the runner expands it
into one run per value listed in `--comp_safety_sources`. Each writes to
`evaluation/results/{model}/{benchmark}/comp_safety_shift_{source}/`,
so an ablation across all four compositional directions runs in a single
command. `print_comparison_table` globs intervention subdirectories
dynamically — every `comp_safety_shift_{source}/` row appears
automatically.

### CompSafetyShift Layer Convention
The npz key `layer_l` corresponds to `hidden_states[l]` (output of transformer
layer `l-1`). Hooks attach to `model.layers[l-1]` for npz key `layer_l`. Default
layer ranges: `(6, 14)` for LLaVA/ShareGPT4V/Qwen-VL-Chat (32 layers),
`(5, 12)` for Qwen2-VL models (28 layers). Layer 0 (embedding) is rejected.

Layer access per wrapper:
- `LLaVAWrapper`: `wrapper.model.language_model.model.layers[l]`
- `ShareGPT4VWrapper`: `wrapper.model.model.layers[l]`
- `Qwen2VLWrapper`: `wrapper.model.model.layers[l]`
- `QwenVLWrapper`: `wrapper.model.transformer.h[l]`

### Classifier
Uses the ShiftDC keyword list (Appendix Table 11, 50 keywords) at
[`evaluation/classifiers/keyword.py`](evaluation/classifiers/keyword.py)
— **single source of truth** also imported by the diagnostic-side
`classify_responses.py`. Case-sensitive substring match. Empty responses
are classified as refusals. ASR = fraction of responses that are NOT
refusals (lower = stronger defence).

### MSSBench train/eval split awareness

For fair comparison against `comp_safety_shift_mssbench_*` (whose
direction is trained on the MSSBench *train* split), the eval pipeline
treats MSSBench specially:

- **`--mssbench_eval_only`** (default ON; `--mssbench_full` to disable)
  — filter at sample-load time so only the 304 eval-split samples are
  generated.
- **`--mssbench_compare_on_eval`** (default ON;
  `--no_mssbench_compare_on_eval` to disable) — also write
  `asr_summary_eval.json` next to `asr_summary.json`, filtered to the
  eval-split ids. Lets vanilla / adashield_s runs that generated
  full-dataset responses be re-scored on the same 304 samples that
  comp_safety_shift_mssbench_* operates on.
- **`--mssbench_view {full, eval, both}`** (default `eval`) — controls
  which MSSBench column(s) `print_comparison_table` shows.
- **Post-hoc tool**:
  `python evaluation/scripts/recompute_mssbench_eval_asr.py --model {hf_id}`
  walks every existing `evaluation/results/{model}/mssbench/*/responses.json`
  and writes `asr_summary_eval.json` without re-running generation. Use
  this when teammates have full-dataset responses on disk and want the
  eval-only summary added retroactively.

### Running the Evaluation

```bash
# Download benchmarks (one-time)
python evaluation/scripts/download_mm_safetybench.py
python evaluation/scripts/download_figstep.py
python evaluation/scripts/download_mssbench.py

# Single model with the canonical default (mssbench_vl)
python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --interventions vanilla comp_safety_shift adashield_s \
    --benchmarks mm_safetybench figstep mssbench \
    --skip_if_exists

# Ablation across all 4 compositional sources side-by-side
python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --interventions vanilla comp_safety_shift adashield_s \
    --benchmarks mm_safetybench figstep mssbench \
    --comp_safety_sources holisafe_tt holisafe_vl mssbench_tt mssbench_vl \
    --skip_if_exists

# Single-model end-to-end (downloads, captions, extraction, directions, eval)
bash evaluation/scripts/run_mssbench_vl_pipeline_for_teammate.sh \
    MODEL=Qwen/Qwen-VL-Chat

# All 5 models
bash evaluation/scripts/run_eval_all_models.sh

# Full pipeline on a fresh workstation (artifacts + downloads + eval)
bash evaluation/scripts/run_full_pipeline.sh
SKIP_DIRECTIONS=1 bash evaluation/scripts/run_full_pipeline.sh   # skip phase 1

# Retroactive eval-only ASR for old vanilla / adashield_s runs
python evaluation/scripts/recompute_mssbench_eval_asr.py --all_models
```

### Output Structure
```
evaluation/results/{model_short}/{benchmark}/{intervention}/
├── responses.json               # Per-sample records (id, question, response, is_refusal)
├── responses.checkpoint.json    # Mid-run checkpoint (deleted on completion)
├── asr_summary.json             # Aggregate ASR over all responses in this dir
└── asr_summary_eval.json        # MSSBench only: ASR filtered to eval-split ids
```

**Note:** Vanilla responses also exist at `data/{benchmark_dir}/{model}/responses/vanilla/responses.json`
(written by `prepare_data.py`). The `evaluation/results/` copy is the
authoritative source for ASR scoring; the `data/` copy is the baseline
data artifact used by the diagnostic pipeline.

The intervention dir naming reflects the source for `comp_safety_shift`:
`comp_safety_shift_holisafe_tt/`, `comp_safety_shift_mssbench_vl/`, etc.
The bare `comp_safety_shift/` directory is legacy (pre-refactor) and
inert — nothing writes to it under the current pipeline.

Resume is per-sample: if interrupted, the next run loads existing records from
both `responses.json` and `responses.checkpoint.json` (merged by sample id)
and skips already-completed samples.

### Adding a New Intervention
1. Create `evaluation/interventions/my_method.py` extending `InterventionBase`.
2. Register in `evaluation/interventions/__init__.py`.
3. Run with `--interventions vanilla comp_safety_shift my_method`.

## Common Issues & Solutions

### CUDA/GPU Not Detected
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Model will fall back to CPU (very slow) if GPU unavailable

### Missing Activation Files
- Re-run extraction steps with `--skip_extraction` removed
- Check disk space (activations are ~2MB per sample)

### Caption Generation Hangs
- Reduce `BATCH_SIZE` in `run_shiftdc.sh`
- Check VRAM usage: `nvidia-smi`

### Reference Dataset Download Fails
- Check HuggingFace connectivity
- Manually download and place in `data/{dataset}/`
- Check dataset loaders in `src/dataset.py` for expected paths

## Useful Commands for Development

### Inspect HoliSafe Schema
```bash
python -c "from src.dataset import load_holisafe, inspect_schema; entries, base = load_holisafe(); inspect_schema(entries)"
```

### Count Cached Activations
```bash
find data/holisafe-bench/llava-1.5-7b-hf/activations -name "*.npz" | wc -l
```

### Monitor GPU During Extraction
```bash
watch -n 1 nvidia-smi
```
