# VLM Hallucination Mitigation Research

Mechanistic interpretability infrastructure for studying and mitigating hallucinations in Vision-Language Models (VLMs). The repo provides shared model wrappers, activation extraction (VL vs text-only), modality-shift analysis, FCCT-style causal mediation, and a benchmark × intervention evaluation harness across five public hallucination benchmarks.

**Scope:** hallucination evaluation (object presence, captions, illusion/consistency). Counting benchmarks (e.g. FSC-147-style MAE/RMSE) are not implemented yet.

## What is implemented today

| Area | Status |
|------|--------|
| Model wrappers (LLaVA, Qwen-VL, Qwen2-VL/2.5, InternVL, ShareGPT4V, MiniGPT-4) | ✓ |
| Benchmark loaders (POPE, AMBER, CHAIR, HallusionBench, MMHal-Bench) | ✓ |
| Scorers: POPE acc/F1, AMBER discriminative, CHAIR-s/i, MMHal judge, HallusionBench acc | ✓ |
| **VTI textual arm** (`vti_textual_*` interventions, β grid, rotation-strength sweeps on POPE) | ✓ |
| **VTI vision arm** (steering on the visual encoder / ViT; paper's α) | **Not implemented** — decoder hooks only today; see [Hooks & interventions](#hooks--interventions) |
| Diagnostic pipelines (modality shift, causal mediation, λ_sim) | ✓ |
| CHAIR + AMBER VTI diagnostics (generative + multi-dimension discriminative) | ✓ |

## Environment

**Machine:** lambdab2 — 4× NVIDIA RTX A6000 (48 GB, Ampere). Check GPU availability with `nvidia-smi` and pin a free device with `CUDA_VISIBLE_DEVICES`.

**Conda env:** `vlm_hallucination_mitigation` (from `environment.yml`). Do not use the removed `requirements.txt` / `pyproject.toml`; `environment.yml` is the source of truth.

```bash
conda env create -f environment.yml   # first time
conda activate vlm_hallucination_mitigation
```

**Caches** (recommended — keep large artifacts off `$HOME`):

```bash
export HF_HOME=/data/romanus/huggingface
export WANDB_DIR=/data/romanus/wandb
mkdir -p "$HF_HOME" "$WANDB_DIR"
```

Add those `export` lines to `~/.bashrc`. On Ubuntu 20.04, the env uses conda activate hooks to prepend `$CONDA_PREFIX/lib` for GLIBCXX compatibility.

**Notes:**
- `flash-attn` is intentionally **not** installed. Models fall back to SDPA/eager attention (preferred for interpretability — fused flash kernels do not expose attention weights).
- InternVL2 / InternVL2.5 may request flash attention in remote model code; load with SDPA/eager if you hit flash-attn import errors.
- MiniGPT-4 requires an external repo and checkpoint: set `MINIGPT4_CKPT=/path/to/pretrained_minigpt4_7b.pth`.

### Run policies (read before reproducing)

- **GPU contention:** lambdab2 is shared. Run `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` to a free card before long jobs.
- **Resolution (Policy A):** each model uses its native/default vision resolution; do not change it between baseline and intervention runs for the same model. Vision-token counts differ by family (e.g. LLaVA = 576 fixed; Qwen2-VL = dynamic).
- **Device for hooks / direction fitting:** VTI direction extraction and any activation-hook work must run with the **full model on one GPU** (no CPU offload). Mixed-device tensors corrupt hooks. Accuracy-only eval runs may use offload on Qwen-VL-Chat, but do not mix that setup with hook-based runs for the same model.
- **Long jobs:** use `nohup` or `tmux` — a dropped SSH session will kill foreground multi-hour sweeps.
- **Qwen2-VL / 2.5 on large images:** uncapped native resolution can OOM in ViT self-attention on high-res photos (CHAIR/AMBER). For constrained runs, pass `max_pixels` to `create_wrapper()` (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`, `max_pixels=1003520`). The CHAIR+AMBER Exp2 driver also accepts `MAX_PIXELS=1003520` for Qwen models.

## Supported models

Six VLM families via `create_wrapper()` in [`src/model.py`](src/model.py):

| Family | Example HuggingFace ID |
|--------|------------------------|
| LLaVA 1.5 / 1.6 | `llava-hf/llava-1.5-7b-hf` |
| ShareGPT4V | `Lin-Chen/ShareGPT4V-7B` |
| MiniGPT-4 | `Vision-CAIR/MiniGPT-4` (+ external ckpt) |
| Qwen-VL-Chat | `Qwen/Qwen-VL-Chat` |
| Qwen2-VL / Qwen2.5-VL | `Qwen/Qwen2-VL-7B-Instruct` |
| InternVL2 / 2.5 | `OpenGVLab/InternVL2-8B` |

Normalized short names (used in directory paths) come from `_normalize_model_name()` — e.g. `llava-hf/llava-1.5-7b-hf` → `llava-1.5-7b-hf`.

## Benchmarks

Registry keys in `BENCHMARK_REGISTRY` ([`src/dataset.py`](src/dataset.py)):

| Benchmark | Key | Task types | On-disk dir |
|-----------|-----|--------------|-------------|
| POPE | `pope` | Binary object-presence (yes/no) | `data/pope/` |
| AMBER | `amber` | Discriminative (yes/no, 3 qtypes) + generative (scorer placeholder) | `data/amber/` |
| CHAIR | `chair` | Caption-level object hallucination (CHAIR-s/i) | `data/chair/` |
| HallusionBench | `hallusionbench` | Mixed illusion types | `data/hallusionbench/` |
| MMHal-Bench | `mmhal_bench` | LLM-judge rating 0–6 (8 question types) | `data/mmhal-bench/` |

**Pinned subsets** (for reproducible diagnostics): `data/chair/pinned_chair_500.json`, `data/amber/pinned_amber_disc_450.json`, `data/pope/pinned_eval_ids.json`. Pass to eval via `--subset_ids_file PATH` (JSON `{benchmark: [ids]}` or flat list).

Each benchmark has a `combined.json` manifest. Uniform sample dict from loaders:

```python
{id, image_path, image_pil, text, label, label_idx, benchmark, task, category, raw}
```

### Download benchmarks

Run all downloads (CHAIR/COCO first, then POPE, AMBER, HallusionBench, MMHal-Bench):

```bash
bash data_scripts/download_all_benchmarks.sh
```

Or individually:

```bash
python data_scripts/download_chair.py          # COCO val2014 + CHAIR (run before POPE)
python data_scripts/download_chair.py --with-train2014   # + train2014 for VTI demos
python data_scripts/download_pope.py
python data_scripts/download_amber.py          # GitHub metadata + Google Drive images
python data_scripts/download_hallusionbench.py # GitHub JSON + Google Drive images
python data_scripts/download_mmhal_bench.py    # HuggingFace Shengcao1006/MMHal-Bench
```

Shared COCO images live under `data/coco/val2014/` (used by POPE and CHAIR). VTI textual direction extraction additionally requires `data/coco/train2014/` (~13 GiB; fetch with `--with-train2014` above).

### VTI textual demos

Paired clean/hallucinated caption demos for textual VTI direction extraction live at [`data/vti/demos.jsonl`](data/vti/demos.jsonl). This file is copied from the authors' [VTI](https://github.com/) reference repository (`hallucination_vti_demos.jsonl`); cloning the `VTI/` subdirectory in this repo is **not** required to run the pipeline.

Direction extraction feeds each demo as `question + caption` through `wrapper.forward_vl()` (not the caption-only `generate_caption()` path). POPE evaluation uses the benchmark question verbatim via `generate_vl(image, question)`.

## Project structure

```
vlm_hallucination_mitigation_summer_2026/
├── src/
│   ├── model.py          # VLM wrappers, create_wrapper()
│   ├── dataset.py        # BENCHMARK_REGISTRY, load_benchmark()
│   ├── extraction.py     # ActivationCache, SVD / modality-shift helpers
│   ├── mediation.py      # Causal mediation hooks, ScoringTarget
│   └── paths.py          # Central path helpers (use these — do not hand-roll)
├── data/
│   ├── {benchmark}/combined.json
│   ├── {benchmark}/{model}/activations/     # sample_{id}_{vl|tt}.npz
│   ├── {benchmark}/{model}/responses/{intervention}/
│   ├── captions/{benchmark}.json
│   ├── vti/demos.jsonl                      # VTI paired-caption demos (from authors)
│   └── coco/val2014/  (+ train2014/ for VTI)
├── tests/                # unit tests (e.g. test_VTI_text_steer.py)
├── helper_scripts/       # tracked one-off / ops scripts (see below)
├── data_scripts/         # download_*, extract_vl/tt, prepare_data, generate_captions
├── diagnostic_experiments/
│   ├── modality_shift/   # m^l = x_vl - x_tt analysis
│   ├── causal_mediation/ # FCCT-style recovery on POPE yes/no
│   └── vti_lambda_sim/   # gated_rotation lambda_sim diagnostic
├── evaluation/
│   ├── run_eval.py       # main benchmark × intervention runner
│   ├── chair_amber_diagnostics/  # CHAIR+AMBER VTI grid + rotation sweeps
│   ├── vti_rotation_strength/      # POPE rotation-strength experiment
│   ├── interventions/    # no_intervention + vti_textual_*
│   └── results/{run_date}/{model}/...
├── helper_scripts/       # qualitative review (review_responses, sample_responses)
└── experiment_artifacts/{experiment}/{model}/
```

### Path conventions

Use [`src/paths.py`](src/paths.py) for diagnostic and artifact paths:

| Purpose | Path |
|---------|------|
| Diagnostic results | `diagnostic_experiments/{experiment}/{model}/results/` |
| Diagnostic plots | `.../results/plots/` |
| Experiment artifacts | `experiment_artifacts/{experiment}/{model}/` |
| Per-model activations | `data/{benchmark_dir}/{model}/activations/` |
| Responses | `data/{benchmark_dir}/{model}/responses/{intervention}/` |
| Eval results | `evaluation/results/{run_date}/{model}/pope_{split}/{intervention}[__b{beta}]/` |
| VTI direction cache | `experiment_artifacts/vti/{model}/textual_directions_nd{N}_rank{r}_seed{s}.npz` |
| Rotation-strength results | `evaluation/vti_rotation_strength/results/{run_date}/{model}/sweep_{variant}_layer_{split}_n{N}.json` |
| CHAIR+AMBER diagnostic summary | `evaluation/results/{run_date}/_diagnostic_summary_chair_amber.md` |
| Qualitative sample galleries | `evaluation/results/{run_date}/_samples/{model}/{benchmark}/` |

## Quickstart

```bash
conda activate vlm_hallucination_mitigation

# 1. Benchmarks (if not already downloaded)
bash data_scripts/download_all_benchmarks.sh

# 2. Sanity-check loading
python -c "from src.dataset import load_benchmark; print(len(load_benchmark('pope', limit=5)))"

# 3. Data prep (captions + VL/TT activations)
python data_scripts/prepare_data.py \
    --benchmarks pope \
    --models llava-hf/llava-1.5-7b-hf \
    --phases captions activations

# 4. Diagnostic: modality shift
MODEL=llava-hf/llava-1.5-7b-hf bash \
    diagnostic_experiments/modality_shift/run_scripts/run_modality_shift.sh

# 5. Evaluation baseline (no intervention)
python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --benchmarks pope \
    --interventions no_intervention \
    --limit 10 \
    --run_date $(date +%Y-%m-%d)
```

**Compare interventions** (prints a table from existing results):

```bash
python -c "from evaluation.runners import print_comparison_table; \
  print_comparison_table('llava-1.5-7b-hf', 'evaluation/results', run_date='YYYY-MM-DD')"
```

Replace `YYYY-MM-DD` with the `--run_date` you used. Each cell also writes `metric_summary.json` with headline metrics.

## Data pipeline

| Phase | Script | Output |
|-------|--------|--------|
| Captions (for TT activations) | `generate_captions.py` or `prepare_data.py --phases captions` | `data/captions/{benchmark}.json` |
| VL activations | `extract_vl.py` | `.../activations/sample_{id}_vl.npz` |
| TT activations | `extract_tt.py` | `.../activations/sample_{id}_tt.npz` |
| Baseline responses | `prepare_data.py --phases responses` or `run_eval.py` | `.../responses/no_intervention/` |

Default response intervention directory name: `no_intervention` (replaces the old `vanilla` baseline).

Caption generation supports `--caption_provider anthropic|openai|local` (API keys via `.env` for cloud providers).

## Diagnostic experiments

### Modality shift (`diagnostic_experiments/modality_shift/`)

Compares per-layer modality shift `m^l = x_vl^l - x_tt^l` between label groups.

- `compute_modality_shift.py`, `plot_modality_shift.py`
- `run_scripts/run_modality_shift.sh`

### Causal mediation (`diagnostic_experiments/causal_mediation/`)

FCCT-style activation patching and recovery rates on POPE yes/no targets.

- `run_mediation.py`, `plot_recovery_rates.py`
- `run_scripts/run_causal_mediation.sh`

## Evaluation

Entry point: [`evaluation/run_eval.py`](evaluation/run_eval.py)

```bash
python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --benchmarks pope amber chair \
    --interventions no_intervention \
    --run_date $(date +%Y-%m-%d) \
    --limit 200 \
    --pope_split random \
    --amber_task discriminative \
    --chair_max_new_tokens 64 \
    --skip_if_exists
```

**Useful flags:**

| Flag | Purpose |
|------|---------|
| `--run_date YYYY-MM-DD` | Dated results subtree (default: today). Same-day reruns resume. |
| `--beta FLOAT` | Textual VTI coefficient β; result dir becomes `{iv}__b{beta}` |
| `--pope_split random\|popular\|adversarial` | POPE split (each → `pope_{split}/` dir) |
| `--amber_task discriminative\|generative` | AMBER task filter |
| `--chair_max_new_tokens N` | CHAIR caption length (default **64**; frozen across baselines/interventions) |
| `--chair_prompt STR` | Override CHAIR caption prompt (VTI uses capital-D *"Please Describe this image in detail."*) |
| `--subset_ids_file PATH` | Pin exact sample ids per benchmark |
| `--judge mock\|openai\|anthropic\|gemini` | MMHal judge (only used when `mmhal_bench` in `--benchmarks`) |
| `--skip_if_exists` | Skip cells that already have `responses.json` + `metric_summary.json` |

Results land under `evaluation/results/{run_date}/{model_short}/{benchmark}/{intervention}[__b{beta}]/`.

### Hooks & interventions

**Decoder-only today.** All registered interventions steer the **LLM decoder** (MLP or full layer output). There is no `vti_visual_*` intervention yet — the VTI paper's vision coefficient α (hooks on the ViT) is not implemented. Reference vision code exists under `VTI/` (vendored, not wired to `run_eval.py`).

**Vision encoder anatomy (for future work).** ViT modules differ by model family; hook targets are not unified yet:

| Model | Vision module (transformers 4.50) | ViT layers | ViT dim | Vision tokens (Policy A) |
|-------|-----------------------------------|------------|---------|--------------------------|
| LLaVA-1.5 | `model.vision_tower.vision_model.encoder.layers[i]` | 24 | 1024 | 576 fixed |
| Qwen2-VL / 2.5 | `model.visual.blocks[i]` | 32 | 1280 | dynamic (varies per image) |
| Qwen-VL-Chat | `model.transformer.visual` | (remote code) | — | 448×448 |
| InternVL2 | inside `model.extract_feature()` | (remote code) | — | 256 per tile |

Projector/merger maps ViT features into LLM embedding space (`multi_modal_projector` on LLaVA, merger inside Qwen2 `visual`, etc.). ShareGPT4V runs ViT on `wrapper._vision_tower` outside `wrapper.model` — a special case if adding vision hooks there.

| Name | Description |
|------|-------------|
| `no_intervention` | Direct model generation (baseline) |
| `vti_textual_additive_mlp` | VTI textual steering, **additive** geometry, **MLP** sub-block output (pre-residual) |
| `vti_textual_additive_layer` | additive geometry, **full residual-stream** (decoder-layer) output — the paper's *described* method |
| `vti_textual_uniform_rotation_mlp` | **norm-preserving rotation**, MLP site — the authors' *released-code* default |
| `vti_textual_uniform_rotation_layer` | rotation at the residual site (subject of the rotation-strength experiment) |
| `vti_textual_gated_rotation_{mlp,layer}` | cosine-gated rotation (numerically ≡ `uniform_rotation`; the gate is disabled in the reference) |

VTI textual interventions accept `--beta` (paper's β; default 0.9). See [VTI textual steering](#vti-textual-steering--implementation--reproduction).

**Add a new intervention:** subclass `InterventionBase` in `evaluation/interventions/`, register in `__init__.py`.

### Metrics

Each run writes `metric_summary.json`. Scorers: `evaluation/classifiers/metrics.py`.

| Benchmark | `metric` field | Headline fields | Direction |
|-----------|----------------|-----------------|-----------|
| POPE | `accuracy` | `accuracy_overall`, `f1_overall`, `yes_ratio` | higher acc/F1 better |
| AMBER (discriminative) | `amber_discriminative` | `accuracy_overall`, `neg_item_accuracy`, `yes_ratio`, `by_qtype` | higher acc / neg_item_acc better; watch `yes_ratio` for agreeableness confound |
| CHAIR | `chair` | `chair_s`, `chair_i`, `avg_objects_mentioned`, `avg_caption_len_chars`, `empty_fraction` | **lower** chair_s/chair_i better; read jointly with length/object coverage |
| MMHal-Bench | `mmhal_judge` | `avg_score` (0–6), `hallucination_rate`, `judge_name` | higher avg_score better |
| HallusionBench | `accuracy` | `accuracy_overall` | higher better |
| AMBER (generative) | `task_accuracy` | placeholder — not wired to official generative scoring |

## VTI textual steering — implementation & reproduction

Re-implementation of the **textual arm** of **VTI** (Visual & Textual Intervention; [arXiv:2410.15778](https://arxiv.org/abs/2410.15778)). The **vision arm** (α, hooks on the ViT) is **not** implemented in this pipeline — only the textual β arm below. Reference vision code exists under `VTI/` but is not registered in `run_eval.py`.

Code lives in [`evaluation/interventions/vti/`](evaluation/interventions/vti/):

| File | Role |
|------|------|
| `steer.py` | The three steering geometries (`additive`, `uniform_rotation`, `gated_rotation`) |
| `hooks.py` | Forward hooks that apply the direction; debug knobs `steer_prefill`, `skip_first_token` |
| `directions.py` | Direction extraction (rank-1 PCA on `clean − hallucinated` last-token states) + on-disk cache |
| `intervention.py` | `VTITextualIntervention`; registry name `vti_textual_{geometry}_{site}` |

**Design axes.** Geometry × hook site:
- **Geometry** — `additive`: `x + β·d̂`. `uniform_rotation`: norm-preserving renormalization `x ← ‖x‖ · normalize( normalize(x) + eps_coeff·β·d̂ )` (a small rotation). `gated_rotation`: same with a cosine gate that the reference leaves disabled (so it equals `uniform_rotation`).
- **Hook site** — `mlp`: the MLP sub-block output (before the residual add). `layer`: the full decoder-layer output (after the residual, i.e. the residual stream).
- The authors' **released code** does `uniform_rotation` at the **MLP** site; the **paper text** describes `additive` on the **residual stream**. Both are provided so they can be compared.

**Coefficients.** `--beta` is the textual coefficient β (intervention default `0.9`). `eps_coeff` (default `0.1`) is an additional rotation blend factor inherited from the reference; note that for the rotation geometries the *effective* coefficient is `eps_coeff·β ≈ 0.1·β`, so additive and rotation are **not** strength-matched at equal β.

**Directions** are computed automatically on first use (rank-1 PCA over `num_demos=70` paired COCO demos, `seed=42`) and cached at `experiment_artifacts/vti/{model}/textual_directions_nd70_rank1_seed42.npz`; subsequent runs load the cache.

### Prerequisites (data — `data/` is gitignored, so fetch it after cloning)

The POPE manifest (`data/pope/combined.json`), the pinned eval subset (`data/pope/pinned_eval_ids.json`), and the VTI paired-caption demos (`data/vti/demos.jsonl`) **are committed** (via `.gitignore` exceptions), so a fresh clone only needs the COCO images:

```bash
conda activate vlm_hallucination_mitigation

# COCO images: val2014 (POPE eval) + train2014 (VTI direction demos, ~13 GiB)
python data_scripts/download_chair.py --with-train2014
```

POPE is evaluated on the pinned subset `data/pope/pinned_eval_ids.json` (the grid driver verifies it before running). VTI textual directions are computed and cached automatically on the first run (rank-1 PCA over the demos), so no manual extraction step is needed.

### Experiment 1 — VTI POPE reproduction (β grid)

Sweeps β over `{0.1 … 1.0}` for the three grid interventions (`additive_mlp`, `additive_layer`, `uniform_rotation_mlp`) against a single `no_intervention` baseline, on POPE `random`/`popular`/`adversarial`, for all four target models (LLaVA-1.5-7B, Qwen-VL-Chat, Qwen2-VL-7B, Qwen2.5-VL-7B). β is the outermost loop, so a complete, paper-comparable slice lands after each β.

```bash
# One-command driver — 200 samples/split, one model in memory at a time (48 GB A6000)
CUDA_VISIBLE_DEVICES=0 LIMIT=200 \
  bash evaluation/run_scripts/run_vti_pope_beta_grid.sh
```

- Output: `evaluation/results/{run_date}/{model}/pope_{split}/{iv}__b{beta}/{responses.json,metric_summary.json}` (baseline under `.../no_intervention/`). Per-model comparison tables print after each β.
- Env knobs: `LIMIT` (samples/split), `BETAS`, `MODELS`, `IVS`, `RUN_DATE`, `OUTPUT_DIR`, `CUDA_VISIBLE_DEVICES`. Larger run: `LIMIT=3000 …` (the paper uses 3000/split).
- Single cell (direct):

```bash
python evaluation/run_eval.py --model llava-hf/llava-1.5-7b-hf \
  --benchmarks pope --pope_split random \
  --interventions vti_textual_additive_mlp vti_textual_uniform_rotation_mlp \
  --beta 0.4 --limit 200
```

### Experiment 2 — VTI rotation-strength sweep

Characterizes the residual-site (`layer`) `uniform_rotation` as β varies: POPE accuracy / precision / recall / F1, yes-ratio, mean response length, and a decision-flip decomposition (induced vs removed hallucinations), plus `decode_only` and `skip_position_0` mitigation probes at the strongest β. POPE `random` split.

```bash
CUDA_VISIBLE_DEVICES=0 NUM_SAMPLES=200 \
  bash evaluation/vti_rotation_strength/run_scripts/run_rotation_strength_sweep.sh
```

- Output: `evaluation/vti_rotation_strength/results/{run_date}/{model}/sweep_uniform_rotation_layer_random_n200.json` (`metrics_by_beta` + per-sample β trajectories in `per_sample`).
- Env knobs: `NUM_SAMPLES`, `POPE_SPLIT`, `BETAS`, `MODELS`, `VARIANTS`, `RUN_DATE`.
- Direct:

```bash
python evaluation/vti_rotation_strength/rotation_strength.py \
  --model llava-hf/llava-1.5-7b-hf --variant uniform_rotation \
  --num_samples 200 --pope_split random \
  --betas 0.6 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.1
```

### Run both, detached

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash evaluation/run_scripts/run_beta_grid_local.sh > /dev/null 2>&1 &
tail -f evaluation/results/_logs/beta_grid_local_*.log
```

`run_beta_grid_local.sh` runs Experiment 1 then Experiment 2 sequentially under one `RUN_DATE`. A RunAI / 3000-per-split variant is in `evaluation/run_scripts/run_beta_grid_runai.sh`.

### Experiment 3 — CHAIR + AMBER VTI diagnostics

Extends the POPE reproduction to a **generative** benchmark (CHAIR) and a **multi-dimension discriminative** benchmark (AMBER). Same textual VTI interventions; vision arm not included.

**Prerequisites** (once per clone):

```bash
# Pinned subsets (deterministic; already committed after first draw)
bash evaluation/chair_amber_diagnostics/run_scripts/run_prep_subsets.sh

# Optional: CHAIR token-cap provenance probe (fixed cap at 64)
bash evaluation/chair_amber_diagnostics/run_scripts/run_step0_chair_cap.sh
```

**Experiment 3a — reproduction grid** (β grid, both models):

```bash
CUDA_VISIBLE_DEVICES=0 RUN_DATE=$(date +%Y-%m-%d) \
  bash evaluation/chair_amber_diagnostics/run_scripts/run_exp1_repro_grid.sh
```

Uses `--subset_ids_file` for pinned CHAIR/AMBER ids, `--chair_max_new_tokens 64`, verbatim VTI CHAIR prompt. Output: `evaluation/results/{run_date}/{model}/{chair|amber}/{iv}__b{beta}/`.

**Experiment 3b — rotation-strength** (`uniform_rotation @ layer`, finer β grid):

```bash
CUDA_VISIBLE_DEVICES=0 RUN_DATE=$(date +%Y-%m-%d) \
  bash evaluation/chair_amber_diagnostics/run_scripts/run_exp2_rotation_strength.sh
```

Output: `evaluation/results/{run_date}/{model}/{chair|amber}_rotation_strength/sweep_uniform_rotation_layer_n{N}.json`.

**Report + qualitative samples** (after a run completes):

```bash
RUN_DATE=YYYY-MM-DD bash evaluation/chair_amber_diagnostics/run_scripts/run_report.sh
RUN_DATE=YYYY-MM-DD bash helper_scripts/run_sample_responses.sh
```

The report writes `evaluation/results/{run_date}/_diagnostic_summary_chair_amber.md`; sample galleries land under `evaluation/results/{run_date}/_samples/`. Use `--skip_if_exists` on the grid drivers to resume interrupted sweeps.

## Extending the repo

**New decoder intervention:** `evaluation/interventions/` + registry in `__init__.py`.

**New vision-encoder intervention (planned):** requires per-model ViT hook dispatch, per-patch activation capture, and `vti_visual_*` registry entries. Reference: `VTI/vti_utils/icv_utils.py` (`obtain_visual_vti`) and `VTI/vti_utils/llm_layers.py` (`add_vti_layers` on ViT MLP). Not wired to the eval harness yet.

**New diagnostic experiment:**

1. Create `diagnostic_experiments/{name}/` with scripts + `run_scripts/`
2. Import paths from `src.paths`
3. Write artifacts to `experiment_artifacts/{name}/{model}/` or `evaluation/results/` as appropriate

**New benchmark:** add loader + `_register()` in `src/dataset.py`, download script in `data_scripts/`, scorer in `evaluation/classifiers/metrics.py`, and wire eval loader in `evaluation/benchmarks/`.

## Helper scripts

`scripts/` is gitignored (local scratch). **`helper_scripts/` is tracked** — qualitative review and ops helpers.

| Path | Purpose |
|------|---------|
| `helper_scripts/review_responses.py` | Inspect a sample id next to its image + model response(s); `--html` builds a self-contained gallery |
| `helper_scripts/sample_responses.py` | Batch qualitative samples for a `run_date` → markdown/JSON + per-model HTML under `evaluation/results/{run_date}/_samples/` |
| `helper_scripts/run_sample_responses.sh` | Wrapper with default CHAIR+AMBER β slices |
| `helper_scripts/runai/setup_vlm.sh` | One-time RunAI setup: extract repo tarball on NFS, bootstrap micromamba, build env, download COCO val2014 + model weights |
| `helper_scripts/runai/run_vti.sh` | RunAI eval job: activate NFS env and run VTI POPE eval |
| `helper_scripts/runai/bootstrap_micromamba.py` | Download micromamba without curl/wget (used by `setup_vlm.sh`) |
| `helper_scripts/runai/run_bash_lf.py` | Strip Windows CRLF and run a shell script via bash (avoids line-ending failures in pods) |

Example — review one POPE sample against a results file:

```bash
python helper_scripts/review_responses.py pope_random_00166 \
  evaluation/results/YYYY-MM-DD/llava-1.5-7b-hf/pope_random/no_intervention/responses.json \
  --html
```

## RunAI (Bell Labs GPU cluster)

**Control plane:** submit jobs from **lambdab2** (`runai` CLI + kubeconfig). **Storage:** Bell Labs NFS (`gpustorage-1`, SFTP-only). **Compute:** GPU pods mount NFS at `/home/datalake`.

### Prerequisites (lambdab2)

```bash
cp /path/to/kubeconfig-mh-gpu ~/.kube/config
export KUBECONFIG=~/.kube/config   # add to ~/.bashrc
runai login
runai project set nlm-mh
```

### Deploy code to NFS

Build a tarball on Linux (LF line endings inside the archive):

```bash
git archive --format=tar.gz -o /tmp/vti_repo.tar.gz HEAD
# verify RunAI helpers are included:
tar -tzf /tmp/vti_repo.tar.gz | grep helper_scripts/runai
```

Upload `/tmp/vti_repo.tar.gz` to the NFS (e.g. `/airl-datalake/romanus/`) via WinSCP (binary mode). Do **not** upload helper scripts separately — they live inside the tarball.

### One-time setup job

Requires `--gpu-devices-request 1 --node-pools h100-pool` (0-GPU and `l40s-pool` hit container mount errors with the current image + NFS).

```bash
runai training submit setup-vlm -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; rm -rf "$REPO"; mkdir -p "$REPO"; tar -xzf "$BASE/vti_repo.tar.gz" -C "$REPO"; bash "$REPO/helper_scripts/runai/setup_vlm.sh"'
```

Persistent artifacts on NFS: `envs/vlm_hal/` (micromamba env), `hf_cache/` (model weights), `vlm_hallucination/` (extracted repo).

### VTI eval job

After setup completes:

```bash
runai training submit hal-vti -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; bash "$REPO/helper_scripts/runai/run_vti.sh"'
```

Monitor: `runai workload list -p nlm-mh` and `runai training logs <job-name> -p nlm-mh`.
