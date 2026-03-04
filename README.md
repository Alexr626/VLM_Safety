# VLM Safety Diagnostics

Diagnostic tests probing **why Vision-Language Models fail to detect emergent harm** -- cases where individually safe images and safe text combine to produce unsafe outputs. Built on top of the [HoliSafe-Bench](https://huggingface.co/datasets/etri-vilab/holisafe-bench) dataset and targeting LLaVA-family models.

The project has two phases:
1. **Three original diagnostic methods** adapted from recent VLM safety papers (see `diagnostic_methods_implementation_guide.md`)
2. **Five follow-up subspace analysis experiments** that investigate *why* SSU examples experience larger modality-induced activation shifts than SSS examples (see `followup_experiments_implementation_guide.md`)

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU (tested with CUDA 12.8)

```bash
git clone https://github.com/Alexr626/VLM_Safety.git
cd VLM_Safety

# Option A: Conda (recommended)
conda env create -f environment.yaml
conda activate vlm_safety

# Option B: pip (install PyTorch with CUDA first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Datasets (HoliSafe-Bench, MM-SafetyBench references, LLaVA-Instruct references) are downloaded automatically on first run and cached under `data/`.

## Project Structure

```
VLM_Safety/
├── src/                                        # Shared library
│   ├── model.py                                #   VLMWrapper — load model, forward passes, captioning
│   ├── dataset.py                              #   HoliSafe-Bench + reference dataset loaders
│   └── extraction.py                           #   ActivationCache, attention extraction, FDR, I/O utils
│
├── diagnostic_exploratory_tests/               # All experiments
│   ├── jailbound_boundary_probing/
│   │   └── method1_boundary_probing.py         #   Linear probing of safety boundaries
│   ├── shiftdc/
│   │   └── method2_activation_shift.py         #   Modality-induced activation shift analysis
│   ├── RAS_activation/
│   │   └── method3_attention_fdr.py            #   Cross-modal attention + Fisher Discriminant Ratio
│   ├── followup_subspace_analysis/             #   Follow-up SVD-based experiments
│   │   ├── sanity_check_tt_baseline.py
│   │   ├── experiment_a_effective_rank.py
│   │   ├── experiment_b_safety_decomposition.py
│   │   ├── experiment_c_subspace_overlap.py
│   │   └── experiment_d_category_analysis.py
│   └── outputs/                                #   All experiment outputs (see below)
│
├── data/                                       #   Auto-downloaded datasets (gitignored)
├── environment.yaml
├── requirements.txt
├── diagnostic_methods_implementation_guide.md   #   Original 3 methods: design + methodology
└── followup_experiments_implementation_guide.md #   Follow-up experiments: design + methodology
```

## Output Directory Layout

All experiments write to a shared output tree organized by model. Activations are computed once and reused across all experiments.

```
diagnostic_exploratory_tests/outputs/llava-1.5-7b-hf/
│
├── activations/                           # Shared activation cache (per-sample .npz files)
│   ├── sample_{id}_vl.npz                 #   Vision-language (image + text) hidden states
│   └── sample_{id}_tt.npz                 #   Text-only (caption + text) hidden states
│
├── method1_boundary_probing/              # Method 1 results
│   ├── per_layer_results.json             #   Accuracy, AUC, bias metrics per layer
│   ├── classifiers.pkl                    #   Trained logistic regression classifiers
│   ├── per_layer_weights.npz              #   Classifier weight vectors
│   ├── per_layer_distances.npz            #   Signed distances to boundary
│   ├── per_layer_probabilities.npz        #   P(unsafe) per sample
│   └── sample_metadata.json               #   Sample IDs and labels
│
├── method2_activation_shift/              # Method 2 results
│   ├── aggregate_stats.json               #   Per-layer SSS/SSU shift stats + p-values
│   ├── per_sample_shifts.json             #   Per-sample cosine and projection values
│   ├── safety_direction_vectors.npz       #   Safety direction s^l per layer
│   ├── reference_activation_matrices.npz  #   Safe/unsafe reference activations for SVD
│   ├── holisafe_captions.json             #   Generated image captions
│   ├── ref_safe_captions.json             #   Safe reference captions
│   ├── ref_unsafe_captions.json           #   Unsafe reference captions
│   ├── reference_metadata.json            #   Reference sample details
│   ├── sample_metadata.json               #   SSS/SSU sample IDs and labels
│   └── plots/                             #   Visualization outputs
│       ├── cosine_similarity.png
│       ├── projection_magnitude.png
│       └── p_values.png
│
├── method3_attention_fdr/                 # Method 3 results
│   ├── attention_aggregate.json           #   Per-layer FDR and attention stats
│   ├── per_sample_attention.json          #   Per-sample attention scores
│   └── sample_metadata.json
│
├── sanity_check_tt_baseline/              # Sanity check results
│   ├── tt_baseline_projections.json       #   Raw TT projections SSS vs SSU
│   ├── tt_baseline_centered_projections.json  # Centered projections + fraction on safe side
│   └── plots/
│       ├── raw_projections.png
│       ├── projection_gap.png
│       ├── centered_projections.png
│       ├── fraction_safe_side.png
│       └── p_values.png
│
├── experiment_a_effective_rank/           # Experiment A results
│   ├── effective_rank_results.json        #   Effective rank at multiple τ thresholds
│   ├── singular_spectra.npz              #   Normalized singular value spectra per layer
│   └── plots/
│       ├── effective_rank_tau09.png
│       ├── rank_gap.png
│       ├── multi_threshold.png
│       └── cumulative_variance_spectra.png
│
├── experiment_b_safety_decomposition/     # Experiment B results
│   ├── per_component_projections.json     #   SSS/SSU projections onto each SVD component
│   ├── dominant_vs_method2_cosine.json    #   Cosine similarity: SVD component 0 vs s^l
│   ├── safety_subspace_bases.npz          #   Top-10 SVD basis vectors per layer
│   ├── safety_subspace_spectra.npz        #   Singular value spectra
│   └── plots/
│       ├── dominant_vs_method2_cosine.png
│       ├── ssu_sss_projection_heatmap.png
│       ├── variance_explained.png
│       └── p_values_per_component.png
│
├── experiment_c_subspace_overlap/         # Experiment C results
│   ├── subspace_overlap_results.json      #   SSS/SSU overlap with safety subspace
│   ├── sensitivity_analysis.json          #   Overlap at varying subspace dimensionality k
│   ├── integration_subspace_bases.npz     #   SSS/SSU integration subspace bases
│   ├── integration_spectra.npz            #   Integration subspace spectra
│   └── plots/
│       ├── integration_safety_overlap.png
│       ├── sss_ssu_integration_overlap.png
│       ├── overlap_gap.png
│       └── sensitivity_dimensionality.png
│
├── experiment_d_category_analysis/        # Experiment D results
│   ├── per_category_projections.json      #   Per-category projections onto safety components
│   ├── category_subspace_overlap.json     #   Per-category overlap + effective rank
│   ├── anova_across_categories.json       #   ANOVA: do categories differ on safety components?
│   ├── category_sample_counts.json        #   Sample counts per harm category
│   └── plots/
│       ├── per_category_projections.png
│       ├── category_overlap_safety.png
│       ├── anova_heatmap.png
│       └── per_category_effective_rank.png
│
└── plotting_scripts/                      # All visualization scripts (run from repo root)
    ├── plot_ShiftDC_results.py
    ├── plot_tt_baseline_projections.py
    ├── plot_effective_rank.py
    ├── plot_safety_decomposition.py
    ├── plot_subspace_overlap.py
    └── plot_category_analysis.py
```

## Running the Experiments

All scripts are run from the repository root. Output paths resolve automatically.

### Phase 1: Original Diagnostic Methods

These require a GPU and run model forward passes.

```bash
# Method 1: Safety Boundary Probing (JailBound-style)
python diagnostic_exploratory_tests/jailbound_boundary_probing/method1_boundary_probing.py \
    --model llava-hf/llava-1.5-7b-hf

# Method 2: Activation Shift Analysis (ShiftDC-style)
python diagnostic_exploratory_tests/shiftdc/method2_activation_shift.py \
    --model llava-hf/llava-1.5-7b-hf

# Method 3: Cross-Modal Attention + FDR (RAS-style)
python diagnostic_exploratory_tests/RAS_activation/method3_attention_fdr.py \
    --model llava-hf/llava-1.5-7b-hf
```

**Useful flags:**

| Flag | Applies to | Description |
|------|-----------|-------------|
| `--limit N` | All | Process only N samples per class (for quick testing) |
| `--skip_extraction` | All | Skip forward passes, reuse cached activations |
| `--skip_captions` | Method 2 | Skip caption generation if caption files already exist |
| `--skip_fdr_formulations` | Method 3 | Skip Part 3C (FDR under modified prompts) |
| `--inspect` | All | Print dataset schema and exit (no GPU needed) |

**Run order:** Methods can run independently, but running **Method 2 first** is recommended -- it generates the VL/TT activations and safety direction vectors that all other experiments depend on. Methods 1 and 3 can then reuse cached activations via `--skip_extraction`.

### Phase 2: Follow-Up Subspace Analysis Experiments

These are CPU-only (no GPU needed) and operate on the cached activations and artifacts produced by Method 2. **Method 2 must be run first.**

```bash
# Sanity Check: Do SSU text-only activations already sit more "unsafe"?
# Dependencies: activations/, method2 safety_direction_vectors.npz + sample_metadata.json
python diagnostic_exploratory_tests/followup_subspace_analysis/sanity_check_tt_baseline.py

# Experiment A: Effective rank of SSS vs SSU modality shift spaces
# Dependencies: activations/, method2 sample_metadata.json
python diagnostic_exploratory_tests/followup_subspace_analysis/experiment_a_effective_rank.py

# Experiment B: Multi-dimensional safety subspace decomposition (SVD)
# Dependencies: activations/, method2 reference_activation_matrices.npz + sample_metadata.json
python diagnostic_exploratory_tests/followup_subspace_analysis/experiment_b_safety_decomposition.py

# Experiment C: Subspace overlap between integration and safety subspaces
# Dependencies: activations/, method2 artifacts, experiment B safety_subspace_bases.npz
python diagnostic_exploratory_tests/followup_subspace_analysis/experiment_c_subspace_overlap.py

# Experiment D: Per-category analysis across harm types
# Dependencies: activations/, method2 artifacts, experiment B safety_subspace_bases.npz
python diagnostic_exploratory_tests/followup_subspace_analysis/experiment_d_category_analysis.py
```

**Run order:** The follow-up experiments have a dependency chain:

```
Method 2 ──► Sanity Check (independent)
         ├─► Experiment A  (independent)
         ├─► Experiment B  (independent)
         │       │
         │       ├──► Experiment C (needs B's safety_subspace_bases.npz)
         │       └──► Experiment D (needs B's safety_subspace_bases.npz)
```

### Generating Plots

Plotting scripts read from each experiment's output directory and save PNGs to a `plots/` subdirectory within that same experiment directory.

```bash
# Method 2 (ShiftDC) plots → method2_activation_shift/plots/
python diagnostic_exploratory_tests/outputs/llava-1.5-7b-hf/plotting_scripts/plot_ShiftDC_results.py

# Sanity check plots → sanity_check_tt_baseline/plots/
python diagnostic_exploratory_tests/outputs/llava-1.5-7b-hf/plotting_scripts/plot_tt_baseline_projections.py

# Experiment A plots → experiment_a_effective_rank/plots/
python diagnostic_exploratory_tests/outputs/llava-1.5-7b-hf/plotting_scripts/plot_effective_rank.py

# Experiment B plots → experiment_b_safety_decomposition/plots/
python diagnostic_exploratory_tests/outputs/llava-1.5-7b-hf/plotting_scripts/plot_safety_decomposition.py

# Experiment C plots → experiment_c_subspace_overlap/plots/
python diagnostic_exploratory_tests/outputs/llava-1.5-7b-hf/plotting_scripts/plot_subspace_overlap.py

# Experiment D plots → experiment_d_category_analysis/plots/
python diagnostic_exploratory_tests/outputs/llava-1.5-7b-hf/plotting_scripts/plot_category_analysis.py
```

## Key Concepts

- **SSS (Safe-Safe-Safe):** Examples where a safe image + safe text produces a safe model output
- **SSU (Safe-Safe-Unsafe):** Examples where a safe image + safe text produces an *unsafe* model output -- the failure mode under investigation
- **Safety direction** `s^l`: Per-layer unit vector pointing from mean unsafe to mean safe reference activations (computed in Method 2)
- **Modality shift:** The difference between vision-language activations (image + text) and text-only activations (caption + text) -- measures how image integration changes internal representations
- **Safety-critical layers:** Layers 6--14 in LLaVA-1.5-7B, where SSU modality shifts disproportionately project onto the safety direction
- **Harm categories:** hate, violence, self_harm, illegal_activity, specialized_advice (from HoliSafe-Bench)

## `src/` API Overview

**`VLMWrapper`** (`src/model.py`) -- Load and run LLaVA-family models:
- `forward_vl(image, text)` -- multimodal forward pass returning hidden states + attentions
- `forward_text(text)` -- text-only forward pass
- `generate_caption(image)` -- generate an image description

**Dataset loaders** (`src/dataset.py`):
- `load_holisafe()` -- download/load HoliSafe-Bench, returns entries + image base path
- `filter_subsets(entries, images_base)` -- extract SSS and SSU sample lists
- `load_mmsafetybench_reference()` / `load_llava_instruct_reference()` -- reference datasets for Method 2

**`ActivationCache`** (`src/extraction.py`) -- Disk-backed per-sample activation storage:
- `save(sample_id, acts, suffix)` / `load_or_none(sample_id, suffix)`
- Suffixes: `"vl"` (vision-language), `"tt"` (text-only), `"ref"` (reference samples)

## VS Code

Launch configurations are included in `.vscode/launch.json` for all three original diagnostic methods.
