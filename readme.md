# VLM Safety — ShiftDC Diagnostic

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
VLM_Safety_new/
├── src/
│   ├── model.py                      # VLMWrapper: forward passes, caption generation
│   ├── dataset.py                    # Dataset loaders + REFERENCE_REGISTRY
│   └── extraction.py                 # ActivationCache, SVD utilities, helpers
├── extraction/
│   ├── generate_captions.py          # Step 1 — GPU: generate image captions
│   ├── extract_vl.py                 # Step 2 — GPU: multimodal activations
│   ├── extract_tt.py                 # Step 3 — GPU: text-only activations
│   └── extract_ref_activations.py    # Step 4 — GPU: reference dataset activations
├── methods/shift_dc/
│   └── method2_shiftdc_analysis.py   # Step 5 — CPU: safety direction + shift analysis
├── followup_analysis/
│   ├── experiment_a_effective_rank.py
│   ├── experiment_b_safety_decomposition.py
│   ├── experiment_c_subspace_overlap.py
│   ├── experiment_d_category_analysis.py
│   └── sanity_check_tt_baseline.py
├── outputs/
│   └── plotting_scripts/             # Visualisation (one script per experiment)
├── data/                             # Local dataset cache
├── run_shiftdc.sh                    # One-command pipeline
└── .gitignore
```

## Quick Start

```bash
# Run the full pipeline (skips already-completed steps)
bash run_shiftdc.sh

# Override any parameter
bash run_shiftdc.sh DATASET=holisafe SAFE_REF=llava-instruct UNSAFE_REF=mm-safetybench
```

Edit the variables at the top of `run_shiftdc.sh` to change defaults:

| Variable        | Default                       | Description                        |
|-----------------|-------------------------------|------------------------------------|
| `MODEL`         | `llava-hf/llava-1.5-7b-hf`   | HuggingFace model ID               |
| `DATASET`       | `holisafe`                    | Main evaluation dataset            |
| `SAFE_REF`      | `llava-instruct`              | Safe reference dataset             |
| `UNSAFE_REF`    | `mm-safetybench`              | Unsafe reference dataset           |
| `BATCH_SIZE`    | `4`                           | Caption generation batch size      |
| `MAX_NEW_TOKENS`| `100`                         | Max tokens for caption generation  |

## Pipeline Steps

| Step | Script | Device | Skippable |
|------|--------|--------|-----------|
| 1. Generate captions (main dataset) | `extraction/generate_captions.py` | GPU | `--skip_if_exists` |
| 2. Extract VL activations | `extraction/extract_vl.py` | GPU | `--skip_extraction` |
| 3. Extract TT activations | `extraction/extract_tt.py` | GPU | `--skip_extraction` |
| 4. Generate captions (reference datasets) | `extraction/generate_captions.py` | GPU | `--skip_if_exists` |
| 5. Extract reference activations | `extraction/extract_ref_activations.py` | GPU | `--skip_if_exists` |
| 6. ShiftDC analysis | `methods/shift_dc/method2_shiftdc_analysis.py` | CPU | `--skip_safety_dir` |

## Followup Analysis

Run after the pipeline completes (CPU only, can run in parallel):

```bash
cd followup_analysis/
python sanity_check_tt_baseline.py
python experiment_a_effective_rank.py
python experiment_b_safety_decomposition.py
python experiment_c_subspace_overlap.py
python experiment_d_category_analysis.py
```

## Visualisation

```bash
cd outputs/plotting_scripts/
python plot_ShiftDC_results.py
python plot_effective_rank.py
python plot_safety_decomposition.py
python plot_subspace_overlap.py
python plot_category_analysis.py
python plot_tt_baseline_projections.py

# Switch model
python plot_effective_rank.py --model other-org/other-model
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

## Outputs

```
outputs/{model_name}/
├── captions/                         # {dataset}.json
├── activations/{dataset}/            # sample_{id}_{vl|tt}.npz
├── reference/{dataset}/              # activation_matrices.npz + metadata.json
├── method2_shiftdc/                  # safety_direction_vectors.npz, aggregate_stats.json, ...
├── experiment_a_effective_rank/
├── experiment_b_safety_decomposition/
├── experiment_c_subspace_overlap/
├── experiment_d_category_analysis/
└── sanity_check_tt_baseline/
```
