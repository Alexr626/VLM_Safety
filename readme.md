# VLM Safety — Cross-Modal Safety Distortion Analysis

Investigates why Vision-Language Models (VLMs) sometimes produce unsafe outputs
given safe image + safe text inputs (SSU phenomenon), by analysing hidden-state
activations across transformer layers.

## Method: ShiftDC

For each sample, compute the **modality shift**:

```
m^l = x_vl^l - x_tt^l
```

- `x_vl`: activations from image + text (real multimodal input)
- `x_tt`: activations from caption + text (text-only counterpart)

Project `m^l` onto safety directions and compare SSS vs SSU groups across layers.

### Two safety directions

- **Semantic safety direction `s^l`** — top-1 PC of the centered per-pair
  difference matrix `D_i = H_safe[i] − H_unsafe[i]` over CatQA harmless/harmful
  minimal-edit pairs (**pairwise PCA**). Pairs cancel topic/style residual
  that joint PCA absorbs. The legacy joint estimate is also persisted as
  `safety_direction_vectors_joint.npz` plus `recipe_sanity.json` (per-layer
  `cos(s_pair, s_joint)`).
- **Compositional safety direction `c^l`** — computed in **four variants**
  parameterised by `(source ∈ {holisafe, mssbench}, representation ∈ {tt, vl})`.
  HoliSafe variants use joint PCA over the SSS+SSU pool; MSSBench variants
  use pairwise PCA over `(rec_idx, q_idx)`-matched paired samples (the same
  text query asked of both image variants). **`mssbench_vl`** is the
  canonical compositional direction consumed by the `comp_safety_shift`
  refusal-eval intervention.

## Supported Models

The pipeline supports ten VLMs via `create_wrapper()` in [src/model.py](src/model.py):

| Model ID                              | Short name                | Wrapper class        |
|---------------------------------------|---------------------------|---------------------|
| `llava-hf/llava-1.5-7b-hf`            | `llava-1.5-7b-hf`         | `LLaVAWrapper`      |
| `llava-hf/llava-v1.6-vicuna-7b-hf`    | `llava-v1.6-vicuna-7b-hf` | `LLaVAWrapper`      |
| `Lin-Chen/ShareGPT4V-7B`              | `sharegpt4v-7b`            | `ShareGPT4VWrapper` |
| `Vision-CAIR/MiniGPT-4`               | `minigpt-4`               | `MiniGPT4Wrapper`   |
| `Qwen/Qwen-VL-Chat`                   | `qwen-vl-chat`            | `QwenVLWrapper`     |
| `Qwen/Qwen2-VL-7B`                    | `qwen2-vl-7b`             | `Qwen2VLWrapper`    |
| `Qwen/Qwen2-VL-7B-Instruct`           | `qwen2-vl-7b-instruct`    | `Qwen2VLWrapper`    |
| `Qwen/Qwen2.5-VL-7B-Instruct`         | `qwen2.5-vl-7b-instruct`  | `Qwen2VLWrapper`    |
| `OpenGVLab/InternVL2-8B`              | `internvl2-8b`            | `InternVL2Wrapper`  |
| `OpenGVLab/InternVL2_5-8B-MPO`        | `internvl2.5-8b-mpo`      | `InternVL2Wrapper`  |

The short name is used as the directory key under `data/*/activations/{model}/`,
`experiment_artifacts/{model}/`, `diagnostic_experiments/{model}/`, etc.

## Project Structure

```
VLM_Safety/
├── src/                                    # Core library modules
│   ├── model.py                            # VLMWrapper: forward passes, caption & response generation
│   ├── dataset.py                          # Dataset loaders + REFERENCE_REGISTRY
│   └── extraction.py                       # ActivationCache, SVD utilities, helpers
│
├── data/                                   # All datasets + their activations
│   ├── holisafe-bench/                     # Main HoliSafe pool + activations/{model}/
│   ├── mssbench/                           # MSSBench (paired SSS/SSU diagnostic + eval)
│   │   ├── combined.json, chat/, embodied/
│   │   ├── train_eval_split.json           # 75/25 record-level split (seed=42)
│   │   └── activations/{model}/sample_mssbench_*_{vl,tt}.npz
│   ├── captions/                           # Generated captions (mssbench.json uses image-stem dedup)
│   ├── catqa-contrastive/                  # Contrastive QA pairs + activations/{model}/
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
│   └── generate_catqa_harmless_pairs.py    # Generate contrastive QA pairs via LLM
│
├── diagnostic_experiments/                 # Phase 1: Diagnostic experiments
│   ├── experiment_scripts/                 # Shared across models
│   │   ├── vl_activation_shift.py                # Core ShiftDC; pairwise s^l
│   │   ├── sanity_check_tt_baseline.py
│   │   ├── augmented_baseline_projections.py
│   │   ├── compositional_safety_direction.py     # --source {holisafe,mssbench} --representation {tt,vl}
│   │   ├── compare_compositional_directions.py   # 5×5 cross-direction cosine matrix
│   │   ├── safety_probes.py                      # 5 probes × ~12 test sets
│   │   ├── generate_responses.py
│   │   ├── classify_responses.py                 # ShiftDC keywords + flat refused_{cond} schema
│   │   └── catqa_behavioral_baseline.py
│   ├── plotting_scripts/                   # Shared across models
│   │   ├── plot_vl_activation_shift_projections.py
│   │   ├── plot_tt_baseline_projections.py
│   │   ├── plot_compositional_safety_shift_projections.py
│   │   ├── plot_direction_comparison.py          # source-aware
│   │   ├── plot_cross_direction_cosine.py        # 5×5 |cos| heatmap
│   │   ├── plot_recipe_sanity.py                 # cos(s_pair, s_joint) per layer
│   │   ├── plot_probe_results.py                 # accuracy_curves_{tt,vl}/ + heatmap
│   │   ├── plot_compositional_eval.py            # compositional_eval_{tt,vl}/{probe}.png
│   │   ├── plot_behavioral_eval.py               # behavioral_eval_{tt,vl}.png (2-panel)
│   │   ├── plot_augmented_baseline.py
│   │   └── plot_behavioral_ground_truth.py
│   ├── {model}/                            # Per-model outputs (one per supported model)
│   │   ├── shift_dc/outputs/               # ShiftDC + recipe_sanity.json
│   │   ├── behavioral_ground_truth/outputs/
│   │   ├── compositional_safety/outputs/
│   │   └── augmented_baseline/outputs/     # (llava only)
│   ├── run_scripts/
│   │   ├── run_shiftdc.sh                          # ShiftDC pipeline (extraction + safety direction)
│   │   ├── run_all_diagnostics.sh                  # All diagnostic phases for a given MODEL
│   │   ├── run_compositional_safety_v2.sh          # ★ multi-source v2: 4 c^l + cross + probes + plots
│   │   ├── run_compositional_safety.sh             # legacy single-source compositional pipeline
│   │   ├── run_data_prep.sh                        # Cohesive text + CT extraction (GPU)
│   │   ├── run_behavioral_ground_truth.sh          # Response gen + refusal classification (GPU)
│   │   ├── run_augmented_diagnostics.sh            # TT-vs-CT safety-projection gap (CPU)
│   │   ├── run_all_new_experiments.sh              # Augmented experiments end-to-end
│   │   ├── run_diag_16gb.sh                        # Models that fit on 16 GB GPUs
│   │   └── run_diag_24gb.sh                        # Models that need >=24 GB VRAM
│
├── subspace_analysis/                      # Phase 2: Subspace analysis (llava-1.5-7b-hf only)
│   ├── llava-1.5-7b-hf/
│   │   ├── effective_rank/                 # Exp A: Shift-space dimensionality
│   │   ├── safety_decomposition/           # Exp B: Multi-dim safety decomposition
│   │   ├── subspace_overlap/               # Exp C: Integration vs safety overlap
│   │   └── category_analysis/              # Exp D: Per-harm-category breakdown
│   └── run_followup.sh                     # Run all subspace experiments
│
├── evaluation/                             # Phase 2: Jailbreak defense evaluation
│   ├── benchmarks/                         # MM-SafetyBench, FigStep, MSSBench (+ eval_only)
│   ├── interventions/                      # comp_safety_shift takes direction_source kwarg
│   ├── classifiers/                        # ShiftDC keyword classifier (single source of truth)
│   ├── runners/                            # Per-source comp_safety_shift expansion;
│   │                                       # writes asr_summary{,_eval}.json
│   ├── run_eval.py                         # CLI: --comp_safety_sources, --mssbench_view, …
│   ├── results/                            # Per-model ASR results (auto-created)
│   └── scripts/
│       ├── run_eval.sh, run_eval_all_models.sh, run_full_pipeline.sh
│       ├── run_mssbench_vl_pipeline_for_teammate.sh   # End-to-end MSSBench-VL setup for one model
│       ├── recompute_mssbench_eval_asr.py             # Post-hoc eval-only ASR from existing responses
│       └── download_{mm_safetybench,figstep,mssbench}.py
│
├── helper_scripts/                         # Utility scripts
│   ├── build_mssbench_refusal_labels.py    # eval responses → diagnostic refusal_map
│   ├── check_data_integrity.py
│   ├── download_missing_images.py
│   ├── add_captions_to_responses.py
│   └── extract_refusal_responses.py
│
├── environment.yml / requirements.txt      # Environment specs
├── CLAUDE.md                               # Repository reference for Claude Code
└── readme.md                               # This file
```

Per-model output directories under `diagnostic_experiments/{model}/{experiment}/outputs/`
contain two subfolders:

```
outputs/
├── results/           # JSON results + plots/
└── artifacts/         # Intermediate .npz files
```

Shared analysis scripts in `experiment_scripts/` / `plotting_scripts/` take `--model`
and dispatch paths using the model's short name.

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
# Ensure libstdcxx-ng is installed
conda install -n vlm_safety -c conda-forge libstdcxx-ng

# The activation script should be auto-created, but if needed:
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Deactivate and reactivate the environment
conda deactivate && conda activate vlm_safety
```

## Quick Start

### Per-model full diagnostic pipeline

```bash
# Every diagnostic phase for one model (data extraction + ShiftDC + behavioral +
# compositional safety + compositional-safety ShiftDC)
MODEL="llava-hf/llava-1.5-7b-hf"         bash diagnostic_experiments/run_scripts/run_all_diagnostics.sh
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
# ShiftDC only (captioning, VL/TT extraction, reference activations, pairwise s^l)
bash diagnostic_experiments/run_scripts/run_shiftdc.sh

# Multi-source compositional pipeline (single MODEL):
#   MSSBench split + captions + extraction → 4 c^l directions →
#   cross-direction cosine matrix → 5-probe cross-evaluation → all plots.
MODEL=llava-hf/llava-1.5-7b-hf bash diagnostic_experiments/run_scripts/run_compositional_safety_v2.sh

# Legacy single-source augmented pipeline
bash diagnostic_experiments/run_scripts/run_all_new_experiments.sh

# Subspace analysis (llava only, CPU)
bash subspace_analysis/run_followup.sh

# Override any parameter on any script
bash diagnostic_experiments/run_scripts/run_shiftdc.sh \
    MODEL=OpenGVLab/InternVL2-8B DATASET=holisafe \
    SAFE_REF=catqa-harmless UNSAFE_REF=catqa-harmful
```

Edit the variables at the top of `run_shiftdc.sh` to change defaults:

| Variable        | Default                       | Description                        |
|-----------------|-------------------------------|------------------------------------|
| `MODEL`         | `llava-hf/llava-1.5-7b-hf`    | HuggingFace model ID               |
| `DATASET`       | `holisafe`                    | Main evaluation dataset            |
| `SAFE_REF`      | `catqa-harmless`              | Safe reference dataset             |
| `UNSAFE_REF`    | `catqa-harmful`               | Unsafe reference dataset           |
| `REF_SAMPLES`   | `550`                         | Reference samples per role         |
| `BATCH_SIZE`    | `4`                           | Caption generation batch size      |
| `MAX_NEW_TOKENS`| `100`                         | Max tokens for caption generation  |

## Pipeline Steps

### ShiftDC pipeline (`run_shiftdc.sh`)

| Step | Script                                                     | Device |
|------|------------------------------------------------------------|--------|
| 1    | `data_scripts/generate_captions.py` (main dataset)         | GPU    |
| 2    | `data_scripts/extract_vl.py`                               | GPU    |
| 3    | `data_scripts/extract_tt.py`                               | GPU    |
| 3b   | `data_scripts/extract_ct.py` (if cohesive text exists)     | GPU    |
| 4–5  | `data_scripts/generate_captions.py` (reference datasets)   | GPU    |
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

## Subspace Analysis (llava-1.5-7b-hf only)

Run after the diagnostic pipeline completes (CPU, experiments are independent):

```bash
# All experiments at once
bash subspace_analysis/run_followup.sh

# Or individually
python subspace_analysis/llava-1.5-7b-hf/effective_rank/experiment_scripts/experiment_a_effective_rank.py
python subspace_analysis/llava-1.5-7b-hf/safety_decomposition/experiment_scripts/experiment_b_safety_decomposition.py
python subspace_analysis/llava-1.5-7b-hf/subspace_overlap/experiment_scripts/experiment_c_subspace_overlap.py
python subspace_analysis/llava-1.5-7b-hf/category_analysis/experiment_scripts/experiment_d_category_analysis.py
```

## Augmented Diagnostic Experiments

Three experiments extend the ShiftDC analysis with a **cohesive text (CT)**
representation (caption + query fused into one natural question) and behavioral
ground truth.

| Experiment | Directory | Key Question |
|------------|-----------|--------------|
| **1. Augmented Baseline**    | `{model}/augmented_baseline/`    | Does CT reveal a safety-direction gap that TT misses? |
| **2. Compositional Safety**  | `{model}/compositional_safety/`  | Is the SSU-vs-SSS direction the same as the semantic-safety direction? Can linear probes separate them? |
| **3. Behavioral Ground Truth** | `{model}/behavioral_ground_truth/` | Under which input condition (VL/TT/CT) does the model actually refuse, and how does this correspond to activation-space signals? |

```bash
# All three experiments in order (GPU required for data prep + behavioral)
bash diagnostic_experiments/run_scripts/run_all_new_experiments.sh

# Or phase-by-phase:
bash diagnostic_experiments/run_scripts/run_data_prep.sh              # GPU: CT generation + extraction
bash diagnostic_experiments/run_scripts/run_behavioral_ground_truth.sh # GPU: responses + refusal labels
bash diagnostic_experiments/run_scripts/run_compositional_safety.sh   # CPU: direction + probes
bash diagnostic_experiments/run_scripts/run_augmented_diagnostics.sh  # CPU: projection-gap analysis
```

**Prerequisites:** The base ShiftDC pipeline (`run_shiftdc.sh`) must have completed.
Cohesive text generation defaults to Anthropic API (`PROVIDER=anthropic`); set
`PROVIDER=openai` or `PROVIDER=local` (uses the VLM itself) to change.

### New data artifacts
- `data/captions/holisafe_cohesive.json` — fused caption + query per sample
- `data/captions/mssbench.json` — image-stem-deduplicated MSSBench captions
- `data/holisafe-bench/activations/{model}/sample_{id}_ct.npz` — CT activations
- `data/holisafe-bench/train_eval_split.json` — HoliSafe stratified 175/group split
- `data/mssbench/train_eval_split.json` — MSSBench 75/25 record-level split
- `experiment_artifacts/{model}/compositional_safety/{holisafe,mssbench}_{tt,vl}/...` — 4 c^l variants

## Data Preparation

Before running any experiments or evaluations, prepare all data artifacts with
a single command:

```bash
# All benchmarks × default models (LLaVA, ShareGPT4V, Qwen-VL-Chat):
python data_scripts/prepare_data.py

# Just captions for eval benchmarks (CPU + Anthropic API):
python data_scripts/prepare_data.py --benchmarks mm_safetybench figstep --phases captions

# Specific models + phases:
python data_scripts/prepare_data.py \
    --models llava-hf/llava-1.5-7b-hf \
    --phases activations responses
```

This script is idempotent — existing artifacts are never regenerated.
See `python data_scripts/prepare_data.py --help` for all flags.

## Evaluation Framework

Evaluates inference-time defense interventions against three public jailbreak
benchmarks, producing ASR (Attack Success Rate) tables comparable to the
ShiftDC / ECSO / AdaShield literature.

### Benchmarks

| Benchmark | Source | Samples |
|-----------|--------|---------|
| **MM-SafetyBench** | `PKU-Alignment/MM-SafetyBench` (HF) | ~5,040 (13 scenarios × 3 image types) |
| **FigStep** | `ThuCCSLab/FigStep` (GitHub) | 500 |
| **MSSBench** | `kzhou35/mssbench` (HF) | 1,200 (600 SSS + 600 SSU); held-out eval split = 304 samples |

### Interventions

| Name | Description |
|------|-------------|
| `vanilla` | No-op baseline |
| `comp_safety_shift` | Projects the modality-induced shift `m^l = x_vl − x_tt` onto `c^l` and subtracts: `x_corrected = x_vl − α·dot(m,c)·c`. Requires captions (`data/captions/{benchmark}.json`) for the TT forward pass. Direction source configurable via `--comp_safety_sources`; **default `mssbench_vl`**. |
| `adashield_s` | Prepends static defence prompt (AdaShield-S, Wang et al. ECCV 2024) |

When `comp_safety_shift` is in `--interventions`, the runner expands it
into one run per source listed in `--comp_safety_sources`, writing each
to `evaluation/results/{model}/{benchmark}/comp_safety_shift_{source}/`.

### MSSBench train/eval split awareness

For fair comparison against `comp_safety_shift_mssbench_*` (whose
direction is trained on the MSSBench *train* split), MSSBench is treated
specially:

- `--mssbench_eval_only` (default ON; `--mssbench_full` to disable) —
  generation filter at sample-load time.
- `--mssbench_compare_on_eval` (default ON) — also writes
  `asr_summary_eval.json` filtered to the 304 eval-split ids alongside
  the canonical `asr_summary.json`. Lets vanilla / adashield_s runs that
  generated full-dataset responses be re-scored on the same 304 samples
  comp_safety_shift_mssbench_* operates on.
- `--mssbench_view {full, eval, both}` — controls the comparison-table
  column(s) (default `eval`).
- Post-hoc: `python evaluation/scripts/recompute_mssbench_eval_asr.py
  --model {hf_id}` adds `asr_summary_eval.json` to existing response
  directories without re-running generation.

### Quick Start

```bash
# 1. Download benchmarks
python evaluation/scripts/download_mm_safetybench.py
python evaluation/scripts/download_figstep.py
python evaluation/scripts/download_mssbench.py

# 2. Run evaluation (single model, canonical default = mssbench_vl)
python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --interventions vanilla comp_safety_shift adashield_s \
    --benchmarks mm_safetybench figstep mssbench \
    --skip_if_exists

# 2'. Ablation across all 4 compositional sources side-by-side
python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --interventions vanilla comp_safety_shift adashield_s \
    --benchmarks mm_safetybench figstep mssbench \
    --comp_safety_sources holisafe_tt holisafe_vl mssbench_tt mssbench_vl \
    --skip_if_exists

# 3. Single-model end-to-end (downloads + extraction + directions + eval)
bash evaluation/scripts/run_mssbench_vl_pipeline_for_teammate.sh \
    MODEL=Qwen/Qwen-VL-Chat

# 4. All 5 models
bash evaluation/scripts/run_eval_all_models.sh

# 5. Full pipeline on a fresh workstation
bash evaluation/scripts/run_full_pipeline.sh
SKIP_DIRECTIONS=1 bash evaluation/scripts/run_full_pipeline.sh

# Retroactive eval-only ASR for old vanilla / adashield_s responses
python evaluation/scripts/recompute_mssbench_eval_asr.py --all_models
```

Results are written to
`evaluation/results/{model}/{benchmark}/{intervention_or_source}/asr_summary{,_eval}.json`.
Per-sample resume is automatic — interrupted runs pick up from the last
completed sample.

## Adding a New Reference Dataset

1. Write a loader function in `src/dataset.py` returning `List[dict]` with keys:
   `id`, `text`, `image_pil`, `image_path`, `source`
2. Register it in `REFERENCE_REGISTRY`:

```python
"my-dataset-safe": {
    "role":      "safe",
    "loader":    my_loader_function,
    "text_only": False,   # True if no images (skips caption generation)
}
```

3. Run the pipeline with `SAFE_REF=my-dataset-safe`

## Adding a New Model

Edit `_MODEL_CONFIGS` in [src/model.py](src/model.py) with the model's class,
prompt templates, visual-token count, and caption prompt. After that, every
shared script (extraction, ShiftDC, behavioral, compositional safety, plotting) picks
it up via `--model <hf-id>` and writes outputs under the model's short name.

## Key Output Formats

### Data directory (`data/`)
```
holisafe-bench/
├── holisafe_bench.json                    # Dataset metadata
├── images/                                # Images by category
├── train_eval_split.json                  # 175/group stratified split
└── activations/{model}/
    ├── sample_{id}_{vl|tt|ct}.npz         # Per-sample hidden states
    └── sample_metadata.json               # id / label / category

captions/
├── {dataset}.json                         # {sample_id: caption_str}
└── holisafe_cohesive.json                 # CT (caption + query fused)

catqa-contrastive/
├── catqa_contrastive_pairs.json           # Harmless/harmful QA pairs
└── activations/{model}/{ref_name}/
    ├── activation_matrices.npz            # {role}_layer_{l}: (N, hidden_dim)
    └── metadata.json

llava-instruct-ref/   mm-safetybench-ref/  # Alt. safe/unsafe reference pools
├── images/
└── samples_n160_seed42.json
```

### Experiment artifacts (`experiment_artifacts/{model}/{experiment}/`)
```
vl_activation_shift/safety_direction_vectors.npz    # Per-layer s^l
compositional_safety/compositional_safety_direction_vectors.npz  # Per-layer c^l
effective_rank/singular_spectra.npz                 # (llava only)
safety_decomposition/{safety_subspace_bases,safety_subspace_spectra}.npz
subspace_overlap/{integration_spectra,integration_subspace_bases}.npz
```

### Experiment results (under each experiment's `outputs/results/`)
```
{model}/shift_dc/outputs/results/
├── vl_activation_shift/
│   ├── aggregate_stats.json, per_sample_shifts.json, sample_metadata.json
│   └── plots/
└── sanity_check_tt_baseline/
    ├── tt_baseline_projections.json
    └── plots/

{model}/behavioral_ground_truth/outputs/results/
├── holisafe_responses.json                # {id: {vl, tt, ct}} greedy generations
├── holisafe_refusal_labels.json           # refusal / compliance per condition
├── catqa_behavioral_baseline.json
├── refusal_summary.json
└── plots/

{model}/compositional_safety/outputs/
├── artifacts/                              # Per-layer SSU-vs-SSS direction data
└── results/                                # Direction comparison + probe results
```
