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

The shift vector `m^l` is projected onto a **safety direction** `s^l` (derived via PCA from safe/unsafe reference datasets) to compare SSS (Safe→Safe) vs SSU (Safe→Unsafe) groups across transformer layers.

## Key Datasets

### Main Evaluation Dataset
- **HoliSafe-Bench** (etri-vilab/holisafe-bench)
  - Filtered to two subsets:
    - **SSS**: Safe image + Safe text → Safe output
    - **SSU**: Safe image + Safe text → Unsafe output (the anomaly)

### Reference Datasets (for safety direction computation)
- **CatQA-Harmful** (unsafe reference): contrastive harmful questions
- **CatQA-Harmless** (safe reference): contrastive harmless counterparts
- **MM-SafetyBench** (alternative unsafe reference): PKU-Alignment/MM-SafetyBench
- **LLaVA-Instruct-80k** (alternative safe reference): liuhaotian/LLaVA-Instruct-150K

## Project Structure

```
VLM_Safety/
├── src/                                    # Core library modules
│   ├── model.py                            # VLMWrapper: model loading, forward passes, caption generation
│   ├── dataset.py                          # Dataset loaders + REFERENCE_REGISTRY
│   └── extraction.py                       # ActivationCache, SVD utilities, helpers
│
├── data/                                   # All datasets + their activations
│   ├── holisafe-bench/                     # HoliSafe-Bench dataset (downloaded on first run)
│   │   └── activations/{model}/            # Per-sample VL/TT .npz + sample_metadata.json
│   ├── captions/                           # Generated image captions ({dataset}.json)
│   └── catqa-contrastive/                  # Contrastive QA pairs + reference activations
│       ├── catqa_contrastive_pairs.json
│       └── activations/{model}/            # catqa-harmful/ + catqa-harmless/ activation matrices
│
├── experiment_artifacts/                   # Artifacts produced by experiments, organized by model
│   └── llava-1.5-7b-hf/
│       ├── vl_activation_shift/            # safety_direction_vectors.npz
│       ├── effective_rank/                 # singular_spectra.npz
│       ├── safety_decomposition/           # safety_subspace_bases.npz, safety_subspace_spectra.npz
│       └── subspace_overlap/               # integration_spectra.npz, integration_subspace_bases.npz
│
├── data_scripts/                           # GPU-based data generation & extraction scripts
│   ├── generate_captions.py                # Generate image captions for text-only counterparts
│   ├── extract_vl.py                       # Extract multimodal (VL) activations
│   ├── extract_tt.py                       # Extract text-only (TT) activations using captions
│   ├── extract_ref_activations.py          # Extract reference dataset activations (safe & unsafe)
│   ├── generate_catqa_harmless_pairs.py    # Generate contrastive QA pairs via LLM
│   ├── generate_cohesive_text.py           # Fuse caption + query into cohesive text (CT)
│   └── extract_ct.py                       # Extract CT activations (cohesive text, text-only forward)
│
├── diagnostic_experiments/                 # Phase 1: Diagnostic experiments
│   └── llava-1.5-7b-hf/
│       ├── shift_dc/                       # ShiftDC modality shift analysis
│       │   ├── experiment_scripts/
│       │   │   ├── vl_activation_shift.py  # Compute safety direction, project shifts
│       │   │   └── sanity_check_tt_baseline.py
│       │   ├── plotting_scripts/
│       │   │   ├── plot_vl_activation_shift_projections.py
│       │   │   └── plot_tt_baseline_projections.py
│       │   ├── outputs/
│       │   │   ├── activations/            # Per-sample VL/TT .npz files
│       │   │   ├── artifacts/              # safety_direction_vectors.npz
│       │   │   └── results/                # aggregate_stats.json, plots/
│       │   └── run_shiftdc.sh              # Base ShiftDC pipeline orchestrator
│       ├── behavioral_ground_truth/        # Exp 3: Model refusal labels for VL/TT/CT
│       │   ├── experiment_scripts/{generate_responses,classify_responses}.py
│       │   ├── plotting_scripts/plot_behavioral_ground_truth.py
│       │   └── outputs/results/            # holisafe_responses.json, refusal_labels.json
│       ├── augmented_baseline/             # Exp 1: TT vs CT projection gaps
│       │   ├── experiment_scripts/augmented_baseline_projections.py
│       │   ├── plotting_scripts/plot_augmented_baseline.py
│       │   └── outputs/{results,artifacts}/
│       ├── combinatorial_safety/           # Exp 2: SSU-vs-SSS direction + probes
│       │   ├── experiment_scripts/{combinatorial_direction,safety_probes}.py
│       │   ├── plotting_scripts/{plot_direction_comparison,plot_probe_results}.py
│       │   └── outputs/{results,artifacts}/
│       ├── run_data_prep.sh                # Cohesive text + CT extraction (GPU)
│       ├── run_behavioral_ground_truth.sh  # Responses + refusal classification (GPU)
│       ├── run_augmented_diagnostics.sh    # Augmented analysis + plots (CPU)
│       └── run_all_new_experiments.sh      # All three augmented experiments, in order
│
├── subspace_analysis/                      # Phase 2: Subspace analysis experiments
│   ├── llava-1.5-7b-hf/
│   │   ├── effective_rank/                 # Experiment A: Effective rank of shift spaces
│   │   │   ├── experiment_scripts/
│   │   │   ├── plotting_scripts/
│   │   │   └── outputs/{results,artifacts}/
│   │   ├── safety_decomposition/           # Experiment B: Multi-dimensional safety decomposition
│   │   │   ├── experiment_scripts/
│   │   │   ├── plotting_scripts/
│   │   │   └── outputs/{results,artifacts}/
│   │   ├── subspace_overlap/               # Experiment C: Integration vs safety subspace overlap
│   │   │   ├── experiment_scripts/
│   │   │   ├── plotting_scripts/
│   │   │   └── outputs/{results,artifacts}/
│   │   └── category_analysis/              # Experiment D: Per-harm-category breakdown
│   │       ├── experiment_scripts/
│   │       ├── plotting_scripts/
│   │       └── outputs/{results}/
│   └── run_followup.sh                    # Run all subspace analysis experiments
│
├── helper_scripts/                         # Utility scripts
│   ├── check_data_integrity.py
│   └── download_missing_images.py
│
├── plans_and_project_descriptions/         # Planning docs & experiment design notes
│
├── CLAUDE.md                               # This file
└── readme.md                               # User-facing documentation
```

## Experiment Directory Convention

Each experiment group follows this structure:
```
<experiment_group>/<model_name>/<experiment_name>/
├── experiment_scripts/         # Python scripts that run the experiment
├── plotting_scripts/           # Python scripts that generate plots
├── outputs/
│   ├── results/                # JSON results + plots/ subdirectory
│   └── artifacts/              # Intermediate .npz files (bases, spectra, etc.)
└── run_<group>.sh              # Shell script to run all experiments in the group
```

## Core Modules

### `src/model.py` — VLMWrapper
Primary target: `llava-hf/llava-1.5-7b-hf` (LLaVA 1.5)
- LlavaForConditionalGeneration (transformers >= 4.37)
- CLIP ViT-L/14@336px → 576 image tokens (24x24 grid)
- LLaMA-2 LLM backbone: 32 layers, hidden_dim=4096
- Image token ID: 32000

**Key Methods:**
- `load()`: Load model + processor, show GPU diagnostics
- `forward_vl(image, text)`: Multimodal forward pass → hidden_states, attentions
- `forward_text(text)`: Text-only forward pass
- `generate_caption(image)`: Generate image caption
- `generate_captions_batch(images)`: Batch caption generation
- `generate_vl(image, text, max_new_tokens=256)`: Generate response from image + text (greedy)
- `generate_text(text, max_new_tokens=256)`: Generate response from text-only prompt (greedy)
- `get_image_token_span(input_ids)`: Find image token positions in expanded sequence
- `get_text_token_positions(input_ids)`: Get non-image (text) token positions

**Properties:**
- `num_layers`, `hidden_dim`, `num_attention_heads`, `image_token_id`, `num_image_tokens`, `model_name`, `device`

### `src/dataset.py` — Dataset Loading
**HoliSafe-Bench:**
- `load_holisafe()`: Download/load HoliSafe-Bench to `data/holisafe-bench/`
- `filter_subsets(entries, images_base)`: Filter to SSS and SSU subsets
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

**Train/Eval Split:**
- `split_holisafe_train_eval(sss, ssu, n_train=175, seed=42)`: Stratified-by-category split into train/eval. Persists to `data/holisafe-bench/train_eval_split.json`. Reuses saved split on subsequent calls with matching seed/n_train.

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
- `extract_subspace(matrix, k)`: Extract top-k principal components via SVD → `(k, hidden_dim)`
- `principal_angles(V1, V2)`: Compute principal angles between two subspaces
- `subspace_overlap(V1, V2)`: Mean cosine of principal angles (1.0 = identical, 0.0 = orthogonal)

**Fisher Discriminant Ratio:**
- `compute_fdr(X_sss, X_ssu, pca_dim)`: Compute separability metric between SSS/SSU activation sets

**I/O Utilities:**
- `save_json()`, `load_json()`, `save_pickle()`, `save_npz()`: Handle numpy/native conversions

## Data Flow & Dependencies

### Data (data/)
Each dataset directory contains the raw data and model-specific activations extracted from it:
- `data/holisafe-bench/activations/{model}/` — VL/TT per-sample .npz + sample_metadata.json
- `data/catqa-contrastive/activations/{model}/` — reference activation matrices per dataset
- `data/captions/{dataset}.json` — generated image captions

### Experiment Artifacts (experiment_artifacts/)
Artifacts produced by experiments, organized by `{model}/{experiment_name}/`:
- `vl_activation_shift/safety_direction_vectors.npz` — consumed by all subspace experiments
- `effective_rank/singular_spectra.npz`
- `safety_decomposition/safety_subspace_bases.npz`, `safety_subspace_spectra.npz`
- `subspace_overlap/integration_spectra.npz`, `integration_subspace_bases.npz`

### Phase 0: Data Extraction (data_scripts/)
```
data_scripts/generate_captions.py       → data/captions/{dataset}.json
data_scripts/extract_vl.py              → data/holisafe-bench/activations/{model}/
data_scripts/extract_tt.py              → data/holisafe-bench/activations/{model}/
data_scripts/extract_ref_activations.py → data/catqa-contrastive/activations/{model}/
```

### Phase 1: Diagnostic Experiments (diagnostic_experiments/)
```
run_shiftdc.sh orchestrates:
  1. generate_captions        (→ data/captions/)
  2-3. extract_vl/tt          (→ data/holisafe-bench/activations/{model}/)
  4-6. ref captions + extract (→ data/catqa-contrastive/activations/{model}/)
  7. vl_activation_shift.py
     reads: data/holisafe-bench/activations/, data/catqa-contrastive/activations/
     writes: experiment_artifacts/{model}/vl_activation_shift/safety_direction_vectors.npz
     writes: shift_dc/outputs/results/vl_activation_shift/ (JSON results + plots)
```

### Phase 2: Subspace Analysis (subspace_analysis/)
Each experiment reads:
- `data/holisafe-bench/activations/{model}/` — VL/TT activations + sample_metadata
- `data/catqa-contrastive/activations/{model}/` — reference matrices (experiments B, C, D)
- `experiment_artifacts/{model}/vl_activation_shift/safety_direction_vectors.npz`

Each saves results to its own `outputs/results/` and artifacts to `experiment_artifacts/{model}/{experiment_name}/`.

## Pipeline: run_shiftdc.sh

Located at: `diagnostic_experiments/llava-1.5-7b-hf/shift_dc/run_shiftdc.sh`

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
bash run_shiftdc.sh MODEL=other-org/model DATASET=my-dataset
```

**Skip Flags:**
- `--skip_if_exists`: Skip if output already exists
- `--skip_extraction`: Skip if activation cache exists
- `--skip_safety_dir`: Skip safety direction computation (reuse existing)

## Typical Workflow

### 1. Run Base Diagnostic Pipeline
```bash
bash diagnostic_experiments/llava-1.5-7b-hf/shift_dc/run_shiftdc.sh
```

### 2. Run Subspace Analysis (after pipeline completes)
```bash
bash subspace_analysis/run_followup.sh
```

### 3. Run Augmented Diagnostic Experiments
```bash
# All three new experiments (data prep + behavioral + analysis)
bash diagnostic_experiments/llava-1.5-7b-hf/run_all_new_experiments.sh

# Phase-by-phase:
bash diagnostic_experiments/llava-1.5-7b-hf/run_data_prep.sh                # GPU
bash diagnostic_experiments/llava-1.5-7b-hf/run_behavioral_ground_truth.sh  # GPU
bash diagnostic_experiments/llava-1.5-7b-hf/run_augmented_diagnostics.sh    # CPU
```

### 4. Run Individual Experiments or Plots
```bash
# Single experiment
python subspace_analysis/llava-1.5-7b-hf/effective_rank/experiment_scripts/experiment_a_effective_rank.py

# Single plot
python subspace_analysis/llava-1.5-7b-hf/effective_rank/plotting_scripts/plot_effective_rank.py
```

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

### Experiment 2 — Combinatorial Safety (`combinatorial_safety/`)
- **`combinatorial_direction.py`**: Applies the same CAST-style PCA procedure used in
  `vl_activation_shift.py` to SSU_train vs SSS_train TT activations → combinatorial
  direction `c^l`. Compares to CatQA-derived `s^l` via cosine similarity, subspace overlap
  (top-5 PCs), and effective rank.
- **`safety_probes.py`**: Trains two logistic-regression probes per layer:
  - Probe A: CatQA safe vs unsafe (content safety)
  - Probe B: SSS_train vs SSU_train (combinatorial safety)
  Cross-evaluates on four test sets: HoliSafe eval (TT, VL), CatQA full, and SSU behavioral
  (predicting refusal vs compliance on held-out SSU).

### Experiment 3 — Behavioral Ground Truth (`behavioral_ground_truth/`)
- **`generate_responses.py`**: Generates model outputs under all three conditions (VL/TT/CT)
  using greedy decoding (`max_new_tokens=256`). Checkpoint-based resume.
- **`classify_responses.py`**: Labels each response as refusal or compliance. Two methods:
  - `keyword` (default): matches refusal phrases ("I cannot", "I'm sorry", etc.)
  - `llm`: calls Anthropic/OpenAI API to classify ambiguous cases

### Key Artifacts
- `data/captions/holisafe_cohesive.json` — CT text per sample
- `data/holisafe-bench/activations/{model}/sample_{id}_ct.npz` — CT activations
- `data/holisafe-bench/train_eval_split.json` — stratified 175/group train/eval split
- `experiment_artifacts/{model}/combinatorial_safety/combinatorial_direction_vectors.npz`
- `behavioral_ground_truth/outputs/results/holisafe_refusal_labels.json` — refusal ground truth

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
   bash diagnostic_experiments/llava-1.5-7b-hf/shift_dc/run_shiftdc.sh SAFE_REF=my-dataset-safe
   ```

## Adding a New Experiment Group

1. Create the directory structure:
   ```
   new_experiment_group/
   └── llava-1.5-7b-hf/
       └── experiment_name/
           ├── experiment_scripts/
           ├── plotting_scripts/
           ├── outputs/
           │   ├── results/
           │   │   └── plots/
           │   └── artifacts/
           └── (optional: run_experiments.sh)
   ```

2. In experiment scripts, use this path convention:
   ```python
   _SCRIPT_DIR = Path(__file__).resolve().parent
   _EXPERIMENT_DIR = _SCRIPT_DIR.parent                      # experiment_name/
   _MODEL_NAME = _EXPERIMENT_DIR.parent.name                 # llava-1.5-7b-hf
   _PROJECT_ROOT = _EXPERIMENT_DIR.parent.parent.parent      # VLM_Safety/
   sys.path.insert(0, str(_PROJECT_ROOT))
   ```

3. To reference shared data and artifacts:
   ```python
   _DATA = _PROJECT_ROOT / "data"
   cache = ActivationCache(str(_DATA / "holisafe-bench" / "activations" / _MODEL_NAME))
   metadata = load_json(str(_DATA / "holisafe-bench" / "activations" / _MODEL_NAME / "sample_metadata.json"))
   safety_vecs = np.load(
       _PROJECT_ROOT / "experiment_artifacts" / _MODEL_NAME / "vl_activation_shift" / "safety_direction_vectors.npz")
   ```

4. To save experiment artifacts:
   ```python
   _EXPERIMENT_ARTIFACTS = _PROJECT_ROOT / "experiment_artifacts" / _MODEL_NAME / "my_experiment"
   _EXPERIMENT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
   save_npz(my_data, str(_EXPERIMENT_ARTIFACTS / "my_artifact.npz"))
   ```

## Key File Formats

### Captions JSON
```json
{
  "sample_id": "generated caption text",
  ...
}
```

### Activation .npz
```python
np.load("sample_{id}_vl.npz")
# Keys: "layer_0", "layer_1", ..., "layer_31"
# Each value: np.ndarray of shape (hidden_dim,), dtype float32
```

### Reference Activation Matrices
```python
data = np.load("activation_matrices.npz")
# Keys: "{role}_layer_0", "{role}_layer_1", ..., "{role}_layer_31"
# Each value: (N_ref_samples, hidden_dim)
```

### Safety Direction Vectors
```python
data = np.load("safety_direction_vectors.npz")
# Keys: "layer_0", "layer_1", ..., "layer_31"
# Each value: (hidden_dim,) — the top-1 PC from PCA(safe vs unsafe)
```

## Important Implementation Details

### Activation Extraction
- **Last-token only**: All methods use the final token's hidden state at each layer.
- **Float32 conversion**: Activations stored as float32 to save space.
- **GPU cleanup**: `cleanup_gpu()` called after each sample to avoid OOM.

### Caption Generation
- **Prompt**: `"USER: <image>\nDescribe this image in detail.\nASSISTANT:"`
- **Parameters**: `max_new_tokens=100`, `do_sample=False` (greedy decoding)
- **Batch processing**: `generate_captions_batch()` for efficiency

### Text-Only Counterpart (TT)
- Uses generated caption as replacement for image
- Prompt: `"USER: {caption}\n{original_text}\nASSISTANT:"`
- Same tokenization, but no visual tokens injected

### Safety Direction Computation
- PCA on reference dataset activations: `safe_ref` vs `unsafe_ref`
- Top-1 principal component = safety direction `s^l` per layer
- Shift projection: `dot(m^l, s^l)` where `m^l = x_vl - x_tt`

### Image Token Handling
- LLaVA 1.5: Single `<image>` placeholder (token 32000) in input_ids
- After vision projection: expanded to 576 consecutive visual tokens
- `get_image_token_span()` returns (start, end) in expanded sequence
- `get_text_token_positions()` returns indices of non-image tokens for cross-modal attention

## Model Support

### Currently Implemented
- `llava-hf/llava-1.5-7b-hf` (primary)
- `llava-hf/llava-v1.6-vicuna-7b-hf` (optional)

### Adding New Models
Edit `_MODEL_CONFIGS` in `src/model.py`:
```python
"new-org/new-model": {
    "model_class": "LlavaForConditionalGeneration",
    "num_image_tokens": 576,
    "prompt_template": "USER: <image>\n{text}\nASSISTANT:",
    "text_only_template": "USER: {text}\nASSISTANT:",
    "caption_prompt": "Describe this image in detail.",
}
```

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
find data/holisafe-bench/activations/llava-1.5-7b-hf -name "*.npz" | wc -l
```

### Monitor GPU During Extraction
```bash
watch -n 1 nvidia-smi
```
