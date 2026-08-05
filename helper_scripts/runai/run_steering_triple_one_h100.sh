#!/usr/bin/env bash
# One RunAI job / one H100: three concurrent steering processes.
#
#   1) LLaVA  — CHAIR then POPE (sequential inside that process)
#   2) Qwen   — CHAIR only
#   3) Qwen   — POPE only
#
# All share CUDA_VISIBLE_DEVICES=0. Parent verifies layout once, prefetches
# HF weights once, remaps paths once; children skip those steps.
#
# Stagger model load by LOAD_STAGGER_SEC (default 45) so three weight loads
# do not spike VRAM at the same instant; evals then overlap for the long run.
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
export MODEL_SHORTS="${MODEL_SHORTS:-llava-1.5-7b-hf qwen2.5-vl-7b-instruct}"
export RUN_DATE="${RUN_DATE:-2026-07-30}"
export CHAIR_CAP="${CHAIR_CAP:-512}"

LOAD_STAGGER_SEC="${LOAD_STAGGER_SEC:-45}"
JOB_NAME="${JOB_NAME:-hal-steer-triple}"

# shellcheck source=runai_job_logging.sh
source "$REPO/helper_scripts/runai/runai_job_logging.sh"
runai_job_logging_start
runai_verify_steering_layout

cd "$REPO"

echo "=== remap baked lambdab2 paths (once) ==="
if [[ -f helper_scripts/runai/remap_lambdab2_paths.py ]]; then
  "$MM" run -p "$ENV_PREFIX" python helper_scripts/runai/remap_lambdab2_paths.py --apply || true
fi

echo "=== prefetch HF weights (once) ==="
"$MM" run -p "$ENV_PREFIX" python -c "from huggingface_hub import snapshot_download; print(snapshot_download('llava-hf/llava-1.5-7b-hf')); print(snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct'))"

echo "=== GPU before workers ==="
nvidia-smi || true

LF="$REPO/helper_scripts/runai/run_bash_lf.py"
LLAVA="$REPO/helper_scripts/runai/run_steering_visual_reasoning_llava.sh"
QWEN="$REPO/helper_scripts/runai/run_steering_visual_reasoning_qwen.sh"

echo "=== start LLaVA CHAIR→POPE ==="
env SKIP_LAYOUT_VERIFY=1 SKIP_REMAP=1 SKIP_HF_PREFETCH=1 SKIP_CONDA_ACTIVATE=1 \
  CUDA_VISIBLE_DEVICES=0 RUN_DATE="$RUN_DATE" CHAIR_CAP="$CHAIR_CAP" \
  JOB_NAME=hal-steer-llava BENCHMARKS="chair pope" MODEL_SHORTS=llava-1.5-7b-hf \
  python3 "$LF" "$LLAVA" &
PID_LLAVA=$!
echo "  pid=$PID_LLAVA"

sleep "$LOAD_STAGGER_SEC"
echo "=== start Qwen CHAIR ==="
env SKIP_LAYOUT_VERIFY=1 SKIP_REMAP=1 SKIP_HF_PREFETCH=1 SKIP_CONDA_ACTIVATE=1 \
  CUDA_VISIBLE_DEVICES=0 RUN_DATE="$RUN_DATE" CHAIR_CAP="$CHAIR_CAP" \
  JOB_NAME=hal-steer-qwen-chair BENCHMARKS=chair MODEL_SHORTS=qwen2.5-vl-7b-instruct \
  python3 "$LF" "$QWEN" &
PID_QWEN_CHAIR=$!
echo "  pid=$PID_QWEN_CHAIR"

sleep "$LOAD_STAGGER_SEC"
echo "=== start Qwen POPE ==="
env SKIP_LAYOUT_VERIFY=1 SKIP_REMAP=1 SKIP_HF_PREFETCH=1 SKIP_CONDA_ACTIVATE=1 \
  CUDA_VISIBLE_DEVICES=0 RUN_DATE="$RUN_DATE" CHAIR_CAP="$CHAIR_CAP" \
  JOB_NAME=hal-steer-qwen-pope BENCHMARKS=pope MODEL_SHORTS=qwen2.5-vl-7b-instruct \
  python3 "$LF" "$QWEN" &
PID_QWEN_POPE=$!
echo "  pid=$PID_QWEN_POPE"

echo "=== all workers launched; waiting ==="
echo "  llava=$PID_LLAVA qwen_chair=$PID_QWEN_CHAIR qwen_pope=$PID_QWEN_POPE"
nvidia-smi || true

EC_LLAVA=0
EC_QWEN_CHAIR=0
EC_QWEN_POPE=0
wait "$PID_LLAVA" || EC_LLAVA=$?
wait "$PID_QWEN_CHAIR" || EC_QWEN_CHAIR=$?
wait "$PID_QWEN_POPE" || EC_QWEN_POPE=$?

echo "=== worker exit codes ==="
echo "  llava_chair_pope=$EC_LLAVA"
echo "  qwen_chair=$EC_QWEN_CHAIR"
echo "  qwen_pope=$EC_QWEN_POPE"
nvidia-smi || true

if [[ "$EC_LLAVA" -ne 0 || "$EC_QWEN_CHAIR" -ne 0 || "$EC_QWEN_POPE" -ne 0 ]]; then
  echo "TRIPLE_FAIL"
  echo "  durable_log=$RUNAI_JOB_LOG"
  echo "  child logs under $BASE/logs/runai/ (hal-steer-llava_*, hal-steer-qwen-chair_*, hal-steer-qwen-pope_*)"
  exit 1
fi

echo "TRIPLE_OK"
echo "  durable_log=$RUNAI_JOB_LOG"
echo "  child logs under $BASE/logs/runai/"
exit 0
