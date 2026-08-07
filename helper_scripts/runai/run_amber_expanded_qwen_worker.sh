#!/usr/bin/env bash
# RunAI worker: one β slice of the AMBER-1500 expanded Qwen grid.
# Env: BETAS (single value), RUN_DATE, MAX_PIXELS, AMBER_SUBSET, LAYER_SETS,
#      STEER_RECONSTRUCTIONS. Called by the β-triple parent or alone.
set -eu
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

BASE=/home/datalake/romanus
REPO=$BASE/vlm_hallucination
ENV_PREFIX=$BASE/envs/vlm_hal
MM=$BASE/bin/micromamba
export MAMBA_ROOT_PREFIX=$BASE/mamba
export HF_HOME=$BASE/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export SKIP_CONDA_ACTIVATE=1

export RUN_DATE="${RUN_DATE:-2026-08-05}"
export OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export AMBER_SUBSET="${AMBER_SUBSET:-data/amber/pinned_amber_disc_1500.json}"
export LAYER_SETS="${LAYER_SETS:-all 5-14 15-24}"
export BETAS="${BETAS:?BETAS must be set (e.g. 0.2)}"
export STEER_RECONSTRUCTIONS="${STEER_RECONSTRUCTIONS:-raw_mean_difference live_pc1_plus_mean}"
export MODEL_SHORTS="${MODEL_SHORTS:-qwen2.5-vl-7b-instruct}"

JOB_NAME="${JOB_NAME:-amber-qwen-beta-${BETAS}}"
# shellcheck source=runai_job_logging.sh
source "$REPO/helper_scripts/runai/runai_job_logging.sh"
runai_job_logging_start

cd "$REPO"

if [[ "${SKIP_REMAP:-0}" != "1" && -f helper_scripts/runai/remap_lambdab2_paths.py ]]; then
  "$MM" run -p "$ENV_PREFIX" python helper_scripts/runai/remap_lambdab2_paths.py --apply || true
fi

if [[ "${SKIP_HF_PREFETCH:-0}" != "1" ]]; then
  echo "=== ensure HF weights present ==="
  "$MM" run -p "$ENV_PREFIX" python -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct'))"
fi

echo "=== AMBER expanded Qwen worker ==="
echo "  BETAS=$BETAS LAYER_SETS=$LAYER_SETS RUN_DATE=$RUN_DATE"
echo "  AMBER_SUBSET=$AMBER_SUBSET MAX_PIXELS=$MAX_PIXELS"
nvidia-smi || true

"$MM" run -p "$ENV_PREFIX" bash \
  evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh \
  Qwen/Qwen2.5-VL-7B-Instruct

echo "WORKER_OK beta=$BETAS"
echo "  durable_log=$RUNAI_JOB_LOG"
