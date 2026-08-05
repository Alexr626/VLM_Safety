#!/usr/bin/env bash
# RunAI: Qwen2.5-VL-7B steering validation grid (CHAIR and/or POPE).
# Default BENCHMARKS=chair pope. For the 3-GPU plan submit two jobs:
#   BENCHMARKS=chair  → hal-steer-qwen-chair
#   BENCHMARKS=pope   → hal-steer-qwen-pope
# AMBER continues on lambdab2; do not set BENCHMARKS=amber here unless intended.
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

export RUN_DATE="${RUN_DATE:-2026-07-30}"
export OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
export CHAIR_CAP="${CHAIR_CAP:-512}"
export BENCHMARKS="${BENCHMARKS:-chair pope}"
export LAYER_SETS="${LAYER_SETS:-all 5-14 15-24}"
export BETAS="${BETAS:-0.2 0.5 0.9}"
export NUM_DEMOS="${NUM_DEMOS:-50 500 200 100}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export SKIP_CONDA_ACTIVATE=1
export MODEL_SHORTS="${MODEL_SHORTS:-qwen2.5-vl-7b-instruct}"

JOB_NAME="${JOB_NAME:-hal-steer-qwen}"
# shellcheck source=runai_job_logging.sh
source "$REPO/helper_scripts/runai/runai_job_logging.sh"
runai_job_logging_start
if [[ "${SKIP_LAYOUT_VERIFY:-0}" != "1" ]]; then
  runai_verify_steering_layout
fi

cd "$REPO"

if [[ "${SKIP_REMAP:-0}" != "1" && -f helper_scripts/runai/remap_lambdab2_paths.py ]]; then
  "$MM" run -p "$ENV_PREFIX" python helper_scripts/runai/remap_lambdab2_paths.py --apply || true
fi

if [[ "${SKIP_HF_PREFETCH:-0}" != "1" ]]; then
  echo "=== ensure HF weights present ==="
  "$MM" run -p "$ENV_PREFIX" python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct'))"
fi

echo "=== RunAI Qwen steering validation ==="
echo "  REPO=$REPO RUN_DATE=$RUN_DATE CHAIR_CAP=$CHAIR_CAP"
echo "  BENCHMARKS=$BENCHMARKS LAYER_SETS=$LAYER_SETS"
nvidia-smi || true

"$MM" run -p "$ENV_PREFIX" bash \
  evaluation/run_scripts/run_steering_visual_reasoning_validation.sh \
  Qwen/Qwen2.5-VL-7B-Instruct

echo "GRID_OK"
echo "  durable_log=$RUNAI_JOB_LOG"
