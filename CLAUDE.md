# VLM Hallucination — Agent Reference

## Overview

Repository for VLM hallucination mitigation research using mechanistic interpretability: activation analysis, modality shifts (`m^l = x_vl - x_tt`), and FCCT-style causal mediation.

## Path Conventions

Use `src/paths.py` — do not hand-roll diagnostic paths.

| Purpose | Path |
|---------|------|
| Diagnostic results | `diagnostic_experiments/{experiment}/{model}/results/` |
| Diagnostic plots | `.../results/plots/` |
| Experiment artifacts | `experiment_artifacts/{experiment}/{model}/` |
| Per-model data | `data/{benchmark_dir}/{model}/activations/` |
| Responses | `data/{benchmark_dir}/{model}/responses/{intervention}/` |
| Eval results | `evaluation/results/{model}/{benchmark}/{intervention}/` |

## Benchmarks (`src/dataset.py`)

`BENCHMARK_REGISTRY` keys: `pope`, `amber`, `chair`, `hallusionbench`, `mmhal_bench`.

Uniform sample dict: `{id, image_path, image_pil, text, label, label_idx, benchmark, task, category, raw}`.

Download: `python data_scripts/download_{benchmark}.py`

## Core Modules

- `src/model.py` — VLM wrappers, `create_wrapper()`
- `src/extraction.py` — `ActivationCache`, `load_modality_shift_matrix()`, SVD helpers
- `src/mediation.py` — `FamilyDispatch`, `ScoringTarget`, capture/patch hooks
- `src/paths.py` — `diagnostic_results_dir()`, `experiment_artifacts_dir()`

## Diagnostic Experiments

### modality_shift/
Compares `||x_vl - x_tt||` per layer between label groups.
- `compute_modality_shift.py`, `plot_modality_shift.py`
- `run_scripts/run_modality_shift.sh`

### causal_mediation/
FCCT-style recovery rates on POPE yes/no targets.
- `run_mediation.py`, `plot_recovery_rates.py`
- `run_scripts/run_causal_mediation.sh`

## Evaluation (`evaluation/`)

- `no_intervention` replaces old `vanilla` baseline
- `eval_runner.py` writes `metric_summary.json`
- Classifiers in `evaluation/classifiers/metrics.py`

```bash
python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --benchmarks pope amber \
    --interventions no_intervention
```

## Data Pipeline

```bash
python data_scripts/prepare_data.py  # captions + VL/TT activations
python data_scripts/extract_vl.py --dataset pope --model ...
python data_scripts/extract_tt.py --dataset pope --model ...
```

Default response intervention dir: `no_intervention`.

## Adding a New Intervention

1. Subclass `InterventionBase` in `evaluation/interventions/`
2. Register in `evaluation/interventions/__init__.py`
3. Run with `--interventions your_method`

## Adding a New Diagnostic Experiment

1. Create `diagnostic_experiments/{name}/` with scripts + `run_scripts/`
2. Import paths from `src.paths`
3. Write artifacts to `experiment_artifacts/{name}/{model}/`
