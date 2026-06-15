# VLM Hallucination Mitigation Research

Mechanistic interpretability infrastructure for studying and mitigating hallucinations in Vision-Language Models (VLMs). Reuses model wrappers, activation extraction, and causal mediation hooks across five hallucination benchmarks.

## Supported Models

Ten VLMs via `create_wrapper()` in [src/model.py](src/model.py): LLaVA 1.5/1.6, ShareGPT4V, MiniGPT-4, Qwen-VL-Chat, Qwen2-VL, Qwen2.5-VL, InternVL2/2.5.

## Benchmarks

| Benchmark | Key | Task types |
|-----------|-----|------------|
| POPE | `pope` | Binary object-presence (yes/no) |
| AMBER | `amber` | Discriminative + generative |
| CHAIR | `chair` | Caption-level object hallucination |
| HallusionBench | `hallusionbench` | Mixed illusion types |
| MMHal-Bench | `mmhal_bench` | Reference-answer consistency |

## Project Structure

```
VLM_Safety/
├── src/
│   ├── model.py          # VLM wrappers
│   ├── dataset.py        # Benchmark loaders + BENCHMARK_REGISTRY
│   ├── extraction.py     # ActivationCache, SVD utilities
│   ├── mediation.py      # Causal mediation hooks
│   └── paths.py          # Central path helpers
├── data/
│   ├── {benchmark}/combined.json
│   ├── {benchmark}/{model}/activations/
│   ├── captions/{benchmark}.json
│   └── coco/val2014/     # Shared COCO images (POPE, CHAIR)
├── experiment_artifacts/{experiment}/{model}/
├── diagnostic_experiments/
│   ├── modality_shift/   # m^l = x_vl - x_tt analysis
│   └── causal_mediation/
└── evaluation/           # Benchmark × intervention runner
```

### Experiment-centric layout

- Results: `diagnostic_experiments/{experiment}/{model}/results/`
- Plots: `.../results/plots/`
- Artifacts: `experiment_artifacts/{experiment}/{model}/`
- Scripts co-located under each experiment directory.

## Quickstart

```bash
conda activate vlm_safety

# Download benchmarks
python data_scripts/download_pope.py
python data_scripts/download_amber.py
python data_scripts/download_chair.py
python data_scripts/download_hallusionbench.py
python data_scripts/download_mmhal_bench.py

# Data prep (captions + activations)
python data_scripts/prepare_data.py \
    --benchmarks pope \
    --models llava-hf/llava-1.5-7b-hf \
    --phases captions activations

# Diagnostic: modality shift
MODEL=llava-hf/llava-1.5-7b-hf bash \
    diagnostic_experiments/modality_shift/run_scripts/run_modality_shift.sh

# Evaluation (no intervention baseline)
python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --benchmarks pope \
    --interventions no_intervention \
    --limit 10
```

## Interventions

| Name | Description |
|------|-------------|
| `no_intervention` | Direct model generation (baseline) |

Additional hallucination mitigation methods can be added under `evaluation/interventions/`.

## Metrics

Evaluation writes `metric_summary.json` per run (replacing the old ASR/refusal scoring). Per-benchmark metrics include accuracy (POPE, HallusionBench), task-split accuracy (AMBER), and reference-match rate (MMHal-Bench). CHAIR uses post-hoc object-inventory scoring.
