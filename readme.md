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

Project `m^l` onto a **safety direction** `s^l` (derived via PCA from safe/unsafe
reference datasets) and compare SSS vs SSU groups across layers.

## Project Structure

```
VLM_Safety/
├── src/                                    # Core library modules
│   ├── model.py                            # VLMWrapper: forward passes, caption generation
│   ├── dataset.py                          # Dataset loaders + REFERENCE_REGISTRY
│   └── extraction.py                       # ActivationCache, SVD utilities, helpers
│
├── data/                                   # All datasets + their activations
│   ├── holisafe-bench/                     # Main evaluation dataset + activations/{model}/
│   ├── captions/                           # Generated image captions
│   └── catqa-contrastive/                  # Contrastive QA pairs + activations/{model}/
│
├── experiment_artifacts/                   # Artifacts produced by experiments
│   └── llava-1.5-7b-hf/
│       ├── vl_activation_shift/            # safety_direction_vectors.npz
│       ├── effective_rank/                 # singular_spectra.npz
│       ├── safety_decomposition/           # subspace bases + spectra
│       └── subspace_overlap/               # integration bases + spectra
│
├── data_scripts/                           # GPU-based data generation & extraction
│   ├── generate_captions.py                # Step 1: Generate image captions
│   ├── extract_vl.py                       # Step 2: Extract multimodal activations
│   ├── extract_tt.py                       # Step 3: Extract text-only activations
│   ├── extract_ref_activations.py          # Step 4: Extract reference activations
│   ├── generate_cohesive_text.py           # Fuse caption + text into single query (CT)
│   └── extract_ct.py                       # Extract CT (cohesive text) activations
│
├── diagnostic_experiments/                 # Phase 1: Diagnostic experiments
│   └── llava-1.5-7b-hf/
│       ├── shift_dc/                       # Base ShiftDC diagnostic
│       │   ├── experiment_scripts/
│       │   ├── plotting_scripts/
│       │   └── run_shiftdc.sh              # Full pipeline orchestrator
│       ├── behavioral_ground_truth/        # Exp 3: Model response refusal labels
│       ├── augmented_baseline/             # Exp 1: TT vs CT projection gaps
│       ├── combinatorial_safety/           # Exp 2: SSU-vs-SSS direction + probes
│       ├── run_data_prep.sh                # Cohesive text + CT extraction (GPU)
│       ├── run_behavioral_ground_truth.sh  # Response gen + refusal classification (GPU)
│       ├── run_augmented_diagnostics.sh    # Analysis + plots (CPU)
│       └── run_all_new_experiments.sh      # All of the above, in order
│
├── subspace_analysis/                      # Phase 2: Subspace analysis
│   ├── llava-1.5-7b-hf/
│   │   ├── effective_rank/                 # Experiment A: Shift space dimensionality
│   │   ├── safety_decomposition/           # Experiment B: Multi-dim safety decomposition
│   │   ├── subspace_overlap/               # Experiment C: Integration vs safety overlap
│   │   └── category_analysis/              # Experiment D: Per-category breakdown
│   └── run_followup.sh                     # Run all subspace experiments
│
├── helper_scripts/                         # Utility scripts
├── plans_and_project_descriptions/         # Planning docs
└── .gitignore
```

Each experiment directory follows a consistent structure:
```
<experiment_group>/<model_name>/<experiment>/
├── experiment_scripts/     # Python scripts
├── plotting_scripts/       # Visualization scripts
├── outputs/
│   ├── results/            # JSON results + plots/
│   └── artifacts/          # Intermediate .npz files
└── run_<group>.sh
```

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

```bash
# 1. Run the base diagnostic pipeline (skips already-completed steps)
bash diagnostic_experiments/llava-1.5-7b-hf/shift_dc/run_shiftdc.sh

# 2. Run subspace analysis experiments
bash subspace_analysis/run_followup.sh

# 3. Run augmented diagnostic experiments (Experiments 1-3)
bash diagnostic_experiments/llava-1.5-7b-hf/run_all_new_experiments.sh

# Override any parameter
bash diagnostic_experiments/llava-1.5-7b-hf/shift_dc/run_shiftdc.sh \
    DATASET=holisafe SAFE_REF=catqa-harmless UNSAFE_REF=catqa-harmful
```

Edit the variables at the top of `run_shiftdc.sh` to change defaults:

| Variable        | Default                       | Description                        |
|-----------------|-------------------------------|------------------------------------|
| `MODEL`         | `llava-hf/llava-1.5-7b-hf`   | HuggingFace model ID               |
| `DATASET`       | `holisafe`                    | Main evaluation dataset            |
| `SAFE_REF`      | `catqa-harmless`              | Safe reference dataset             |
| `UNSAFE_REF`    | `catqa-harmful`               | Unsafe reference dataset           |
| `BATCH_SIZE`    | `4`                           | Caption generation batch size      |
| `MAX_NEW_TOKENS`| `100`                         | Max tokens for caption generation  |

## Pipeline Steps

| Step | Script | Device | Skippable |
|------|--------|--------|-----------|
| 1. Generate captions (main dataset) | `data_scripts/generate_captions.py` | GPU | `--skip_if_exists` |
| 2. Extract VL activations | `data_scripts/extract_vl.py` | GPU | `--skip_extraction` |
| 3. Extract TT activations | `data_scripts/extract_tt.py` | GPU | `--skip_extraction` |
| 4-5. Generate captions (reference datasets) | `data_scripts/generate_captions.py` | GPU | `--skip_if_exists` |
| 6. Extract reference activations | `data_scripts/extract_ref_activations.py` | GPU | `--skip_if_exists` |
| 7. ShiftDC analysis | `.../shift_dc/experiment_scripts/vl_activation_shift.py` | CPU | `--skip_safety_dir` |

## Subspace Analysis

Run after the diagnostic pipeline completes (CPU only, can run in parallel):

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

Three new diagnostic experiments extend the ShiftDC analysis with a
**cohesive text (CT)** representation (caption + query fused into one natural
question) and behavioral ground truth.

| Experiment | Directory | Key Question |
|------------|-----------|--------------|
| **1. Augmented Baseline** | `augmented_baseline/` | Does CT reveal a safety-direction gap that TT misses? |
| **2. Combinatorial Safety** | `combinatorial_safety/` | Is the SSU-vs-SSS direction the same as the content-safety direction? Can linear probes separate them? |
| **3. Behavioral Ground Truth** | `behavioral_ground_truth/` | Under which input condition (VL/TT/CT) does the model actually refuse, and how does this correspond to activation-space signals? |

```bash
# Run all three experiments in order (GPU required for phases 0 and 1)
bash diagnostic_experiments/llava-1.5-7b-hf/run_all_new_experiments.sh

# Or phase-by-phase:
bash diagnostic_experiments/llava-1.5-7b-hf/run_data_prep.sh              # GPU: CT gen + extraction
bash diagnostic_experiments/llava-1.5-7b-hf/run_behavioral_ground_truth.sh  # GPU: responses + refusal labels
bash diagnostic_experiments/llava-1.5-7b-hf/run_augmented_diagnostics.sh  # CPU: analysis + plots
```

**Prerequisites:** The base ShiftDC pipeline (`run_shiftdc.sh`) must have completed.
Cohesive text generation defaults to Anthropic API (`PROVIDER=anthropic`); set
`PROVIDER=openai` or `PROVIDER=local` (uses the VLM) to change.

### New data artifacts
- `data/captions/holisafe_cohesive.json` — fused caption + query per sample
- `data/holisafe-bench/activations/{model}/sample_{id}_ct.npz` — CT activations
- `data/holisafe-bench/train_eval_split.json` — stratified train/eval split (175/group)
- `experiment_artifacts/{model}/combinatorial_safety/combinatorial_direction_vectors.npz`

## Intervention Experiments (Just for testing)

After establishing the diagnostic picture, we test whether steering the hidden
state at inference time can reduce the SSU attack success rate.

Two intervention strategies are implemented in `intervention/spherical_shiftdc.py`:

| Method | Formula | Norm preserved? |
|--------|---------|----------------|
| **Original ShiftDC** (Zou et al. 2025) | `x̂ = x_vl − proj_{s^l}(m^l)` | ✗ (norm changes ≈ 2%) |
| **Spherical ShiftDC** (proposed) | Slerp from `x_vl` toward ShiftDC target by `t` | ✓ (exact) |

The motivation for Spherical ShiftDC: LLaVA's LLaMA backbone uses **RMSNorm**,
which normalises each layer's output; norm-preserving interventions are less
likely to degrade generation quality.

### Running interventions

```bash
cd intervention/llava-1.5-7b-hf/experiment_scripts

# Baseline (no intervention)
python run_intervention.py --method none

# Original ShiftDC (Zou et al.)
python run_intervention.py --method original --layer 31

# Spherical ShiftDC (proposed, t=1.0 = full rotation)
python run_intervention.py --method spherical --layer 31 --t 1.0

# Spherical with gate (scale rotation by cosine alignment)
python run_intervention.py --method spherical --layer 31 --t 1.0 --gate_by_alignment
```

Results are written to `intervention/llava-1.5-7b-hf/outputs/results/`.
The script supports **checkpoint-based resume**: if interrupted, re-run the same
command and it will pick up from the last checkpoint (saved every 20 samples).

### Plotting

```bash
cd intervention/llava-1.5-7b-hf/plotting_scripts
python plot_intervention_results.py
```

Generates bar charts comparing ASR, helpfulness cost, and norm preservation
across all methods found in the results directory.

### Current results (keyword-based refusal detection, layer 31, eval split)

| Method | SSU ASR ↓ | SSS False-Refusal ↓ | Norm ratio |
|--------|-----------|---------------------|------------|
| None (baseline) | 97.2% | 0.15% | — |
| Original ShiftDC | 97.2% | 0.15% | 1.007 ± 0.009 |
| Spherical ShiftDC (t=1.0) | 97.2% | 0.15% | **1.000 ± 0.000** |

> **Note:** ASR is measured by keyword-based refusal detection (fast heuristic).
> The Spherical method perfectly preserves norm (ratio ≈ 1.0000000),
> while Original ShiftDC introduces ~0.7% norm inflation on SSU samples.

### Intervention directory structure

```
intervention/
├── spherical_shiftdc.py                        # Core math: ShiftDC + Spherical ShiftDC
└── llava-1.5-7b-hf/
    ├── experiment_scripts/
    │   └── run_intervention.py                 # Main inference + intervention script
    ├── plotting_scripts/
    │   └── plot_intervention_results.py        # Comparison plots
    └── outputs/results/
        ├── none_summary.json                   # Baseline aggregate stats
        ├── original_summary.json               # Original ShiftDC aggregate stats
        ├── spherical_t1.00_summary.json        # Spherical ShiftDC aggregate stats
        └── plots/                              # Bar charts (ASR, helpfulness, norm)
```

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

## Key Output Formats

### Data directory (`data/`)
```
holisafe-bench/
├── holisafe_bench.json                    # Dataset metadata
├── images/                                # Images by category
└── activations/{model}/
    ├── sample_{id}_{vl|tt}.npz            # Per-sample hidden states
    └── sample_metadata.json               # id / label / category

captions/{dataset}.json                    # {sample_id: caption_str}

catqa-contrastive/
├── catqa_contrastive_pairs.json           # Harmless/harmful QA pairs
└── activations/{model}/{ref_name}/
    ├── activation_matrices.npz            # {role}_layer_{l}: (N, hidden_dim)
    └── metadata.json
```

### Experiment artifacts (`experiment_artifacts/{model}/{experiment}/`)
```
vl_activation_shift/safety_direction_vectors.npz    # Per-layer safety direction s^l
effective_rank/singular_spectra.npz
safety_decomposition/{safety_subspace_bases,safety_subspace_spectra}.npz
subspace_overlap/{integration_spectra,integration_subspace_bases}.npz
```

### Experiment results (under each experiment's `outputs/results/`)
```
shift_dc/outputs/results/
├── vl_activation_shift/
│   ├── aggregate_stats.json, per_sample_shifts.json, sample_metadata.json
│   └── plots/
└── sanity_check_tt_baseline/
    ├── tt_baseline_projections.json
    └── plots/
```
