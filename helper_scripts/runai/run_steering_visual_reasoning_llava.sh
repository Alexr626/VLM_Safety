#!/usr/bin/env bash
# RunAI: resume LLaVA steering visual-reasoning validation grid.
# Runs inside a pod with NFS at /home/datalake.
#
# Assumes:
#   - micromamba env at /home/datalake/romanus/envs/vlm_hal
#   - repo extracted at /home/datalake/romanus/vlm_hallucination
#   - meandiff direction dirs present under
#     experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demos850_*_meandiff_partition/
#   - CHAIR_CAP frozen to 512 (probe auto-bump from 2026-07-30)
#
# Default: CHAIR then POPE (AMBER already complete on lambdab2). Override with
# BENCHMARKS=... if you also synced AMBER cells and want skip_if_exists there.
#
# Logs to stdout (runai training logs) and $BASE/logs/runai/.
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
# libstdc++ for Pillow / conda libs (see IMPLEMENTATION.md 2026-07-17 note)
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export RUN_DATE="${RUN_DATE:-2026-07-30}"
export OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
export CHAIR_CAP="${CHAIR_CAP:-512}"
export BENCHMARKS="${BENCHMARKS:-chair pope}"
export LAYER_SETS="${LAYER_SETS:-all 5-14 20-29}"
export BETAS="${BETAS:-0.2 0.5 0.9}"
export NUM_DEMOS="${NUM_DEMOS:-50 500 200 100}"
# validation.sh would otherwise call conda activate (absent in RunAI pods)
export SKIP_CONDA_ACTIVATE=1
export MODEL_SHORTS="${MODEL_SHORTS:-llava-1.5-7b-hf}"

JOB_NAME="${JOB_NAME:-hal-steer-llava}"
# shellcheck source=runai_job_logging.sh
source "$REPO/helper_scripts/runai/runai_job_logging.sh"
runai_job_logging_start
if [[ "${SKIP_LAYOUT_VERIFY:-0}" != "1" ]]; then
  runai_verify_steering_layout
fi

cd "$REPO"

# Remap any lambdab2 absolute paths baked into data/artifacts (idempotent).
if [[ "${SKIP_REMAP:-0}" != "1" && -f helper_scripts/runai/remap_lambdab2_paths.py ]]; then
  "$MM" run -p "$ENV_PREFIX" python helper_scripts/runai/remap_lambdab2_paths.py --apply || true
fi

echo "=== RunAI LLaVA steering validation ==="
echo "  REPO=$REPO"
echo "  RUN_DATE=$RUN_DATE CHAIR_CAP=$CHAIR_CAP"
echo "  BENCHMARKS=$BENCHMARKS"
nvidia-smi || true

"$MM" run -p "$ENV_PREFIX" bash \
  evaluation/run_scripts/run_steering_visual_reasoning_validation.sh \
  llava-hf/llava-1.5-7b-hf

echo "GRID_OK"
echo "  durable_log=$RUNAI_JOB_LOG"
