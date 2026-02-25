# VLM Safety Diagnostics

Diagnostic tests probing **why Vision-Language Models fail to detect emergent harm** -- cases where individually safe images and safe text combine to produce unsafe outputs. Built on top of the [HoliSafe-Bench](https://huggingface.co/datasets/etri-vilab/holisafe-bench) dataset and targeting LLaVA-family models.

Each diagnostic method is adapted from a recent VLM safety paper. For implementation details and methodology, see `diagnostic_methods_implementation_guide.md`.

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
├── src/                                    # Shared library
│   ├── model.py                            #   VLMWrapper — load model, forward passes, captioning
│   ├── dataset.py                          #   HoliSafe-Bench + reference dataset loaders
│   └── extraction.py                       #   ActivationCache, attention extraction, FDR, I/O utils
│
├── diagnostic_exploratory_tests/           # Diagnostic methods
│   ├── jailbound_boundary_probing/
│   │   └── method1_boundary_probing.py     #   Linear probing of safety boundaries
│   ├── shiftdc/
│   │   └── method2_activation_shift.py     #   Modality-induced activation shift analysis
│   ├── RAS_activation/
│   │   └── method3_attention_fdr.py        #   Cross-modal attention + Fisher Discriminant Ratio
│   └── outputs/                            #   All experiment outputs (see below)
│
├── data/                                   #   Auto-downloaded datasets (gitignored)
├── environment.yaml
├── requirements.txt
└── diagnostic_methods_implementation_guide.md
```

### Output directory layout

All three methods write to a single shared output tree. Activations are computed once and reused across methods.

```
diagnostic_exploratory_tests/outputs/{model_name}/
├── activations/                    # Shared activation cache (per-sample .npz files)
├── method1_boundary_probing/       # Classifiers, weight vectors, boundary distances
├── method2_activation_shift/       # Safety direction vectors, shift stats, captions
├── method3_attention_fdr/          # Attention maps, FDR per layer
└── plots/                          # Visualization scripts + output figures
```

## Running the Diagnostic Tests

Each method is a standalone script. Run them from anywhere -- output paths resolve automatically.

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

### Useful flags

| Flag | Description |
|------|-------------|
| `--limit N` | Process only N samples per class (for quick testing) |
| `--skip_extraction` | Skip forward passes, reuse cached activations |
| `--skip_captions` | (Method 2) Skip caption generation if caption files exist |
| `--skip_fdr_formulations` | (Method 3) Skip Part 3C (FDR under modified prompts) |
| `--inspect` | Print dataset schema and exit (no GPU needed) |

### Run order

Methods can run independently, but running Method 2 first is recommended since it generates VL activations that Methods 1 and 3 can reuse via `--skip_extraction`.

### VS Code debugger

Launch configurations are included in `.vscode/launch.json` for all three methods.

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
