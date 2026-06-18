# VLM Hallucination Mitigation Research

Mechanistic interpretability infrastructure for studying and mitigating hallucinations in Vision-Language Models (VLMs). The repo provides shared model wrappers, activation extraction (VL vs text-only), modality-shift analysis, FCCT-style causal mediation, and a benchmark × intervention evaluation harness across five public hallucination benchmarks.

**Scope:** hallucination evaluation (object presence, captions, illusion/consistency). Counting benchmarks (e.g. FSC-147-style MAE/RMSE) are not implemented yet.

## Research workflow

This repo is used in a two-agent loop with an external research/analysis assistant. Shared state lives in two root files:

| File | Purpose |
|------|---------|
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | Ground-truth description of the codebase (modules, APIs, paths, environment). Updated whenever code changes. |
| `RESEARCH_LOG.md` | Append-only log of runs (commands, paths, headline metrics). Factual records only — no interpretation. |

Implementation agents and the analyst coordinate through these files; see [`.cursor/rules/research_workflow.mdc`](.cursor/rules/research_workflow.mdc) for the full contract.

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

## Supported models

Ten VLMs via `create_wrapper()` in [`src/model.py`](src/model.py):

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
| AMBER | `amber` | Discriminative + generative | `data/amber/` |
| CHAIR | `chair` | Caption-level object hallucination | `data/chair/` |
| HallusionBench | `hallusionbench` | Mixed illusion types | `data/hallusionbench/` |
| MMHal-Bench | `mmhal_bench` | Reference-answer consistency | `data/mmhal-bench/` |

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
├── experiment_artifacts/{experiment}/{model}/
└── evaluation/           # run_eval.py, interventions, metrics
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
| Eval results | `evaluation/results/{model}/{benchmark}/{intervention}/` |

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
    --limit 10
```

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
    --benchmarks pope amber \
    --interventions no_intervention
```

### Interventions

| Name | Description |
|------|-------------|
| `no_intervention` | Direct model generation (baseline) |

Add new methods under `evaluation/interventions/`:

1. Subclass `InterventionBase`
2. Register in `evaluation/interventions/__init__.py`
3. Run with `--interventions your_method`

### Metrics

Each run writes `metric_summary.json` under the eval output dir. Per-benchmark scorers live in `evaluation/classifiers/metrics.py`:

| Benchmark | Metric | Notes |
|-----------|--------|-------|
| POPE | `accuracy` | Yes/no normalization, split breakdown |
| AMBER | `task_accuracy` | Discriminative vs generative splits |
| HallusionBench | `accuracy` | Yes/no or substring match |
| MMHal-Bench | `reference_match` | Substring match to reference answer |
| CHAIR | `chair_pending` | Full CHAIR-s/i needs post-hoc COCO object inventory |

## Extending the repo

**New intervention:** `evaluation/interventions/` + registry in `__init__.py`.

**New diagnostic experiment:**

1. Create `diagnostic_experiments/{name}/` with scripts + `run_scripts/`
2. Import paths from `src.paths`
3. Write artifacts to `experiment_artifacts/{name}/{model}/`

**New benchmark:** add loader + `_register()` in `src/dataset.py`, download script in `data_scripts/`, and wire eval loader in `evaluation/benchmarks/`.

For full API signatures, hook behavior, and environment details, see [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## Helper scripts

`scripts/` is gitignored (local scratch for one-time fixes). **`helper_scripts/` is tracked** — ops and deployment helpers that teammates may need to reproduce cluster setup.

| Path | Purpose |
|------|---------|
| `helper_scripts/runai/setup_vlm.sh` | One-time RunAI setup: extract repo tarball on NFS, bootstrap micromamba, build env, download COCO val2014 + model weights |
| `helper_scripts/runai/run_vti.sh` | RunAI eval job: activate NFS env and run VTI POPE eval |
| `helper_scripts/runai/bootstrap_micromamba.py` | Download micromamba without curl/wget (used by `setup_vlm.sh`) |
| `helper_scripts/runai/run_bash_lf.py` | Strip Windows CRLF and run a shell script via bash (avoids line-ending failures in pods) |

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
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/dspy_image2:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; rm -rf "$REPO"; mkdir -p "$REPO"; tar -xzf "$BASE/vti_repo.tar.gz" -C "$REPO"; python3 "$REPO/helper_scripts/runai/run_bash_lf.py" "$REPO/helper_scripts/runai/setup_vlm.sh"'
```

Persistent artifacts on NFS: `envs/vlm_hal/` (micromamba env), `hf_cache/` (model weights), `vlm_hallucination/` (extracted repo).

### VTI eval job

After setup completes:

```bash
runai training submit hal-vti -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/dspy_image2:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; python3 "$REPO/helper_scripts/runai/run_bash_lf.py" "$REPO/helper_scripts/runai/run_vti.sh"'
```

Monitor: `runai workload list -p nlm-mh` and `runai training logs <job-name> -p nlm-mh`.
