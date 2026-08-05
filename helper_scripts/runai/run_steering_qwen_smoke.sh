#!/usr/bin/env bash
# RunAI smoke: Qwen2.5-VL-7B + one steered POPE cell (limit 5).
# Prefetches HF weights if missing. Logs to NFS logs/runai/.
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
export MODEL_SHORTS="${MODEL_SHORTS:-qwen2.5-vl-7b-instruct}"

JOB_NAME="${JOB_NAME:-steer-qwen-smoke}"
# shellcheck source=runai_job_logging.sh
source "$REPO/helper_scripts/runai/runai_job_logging.sh"
runai_job_logging_start
runai_verify_steering_layout

LIMIT="${LIMIT:-5}"
SMOKE_TAG="${SMOKE_TAG:-runai_steering_smoke_qwen_2026-07-31}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DIR_REL="experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd50_s42_meandiff_partition"

cd "$REPO"

echo "=== RunAI Qwen steering smoke ==="
echo "  REPO=$REPO LIMIT=$LIMIT SMOKE_TAG=$SMOKE_TAG"
nvidia-smi || true

echo "=== ensure HF weights present ==="
"$MM" run -p "$ENV_PREFIX" python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"

"$MM" run -p "$ENV_PREFIX" python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'

echo "=== baseline: pope random limit=$LIMIT ==="
"$MM" run -p "$ENV_PREFIX" python evaluation/run_eval.py \
  --model "$MODEL" \
  --benchmarks pope --pope_split random \
  --interventions no_intervention \
  --limit "$LIMIT" \
  --max_pixels "$MAX_PIXELS" \
  --output_dir "evaluation/results/$SMOKE_TAG"

echo "=== steered: pope random nd50 layers=all beta=0.2 limit=$LIMIT ==="
"$MM" run -p "$ENV_PREFIX" python evaluation/run_eval.py \
  --model "$MODEL" \
  --benchmarks pope --pope_split random \
  --interventions vti_textual_additive_mlp \
  --beta 0.2 --layer_set all --directions_dir "$DIR_REL" \
  --limit "$LIMIT" \
  --max_pixels "$MAX_PIXELS" \
  --output_dir "evaluation/results/$SMOKE_TAG"

echo "=== smoke outputs ==="
find "evaluation/results/$SMOKE_TAG" -name metric_summary.json -print
echo "SMOKE_OK"
echo "  durable_log=$RUNAI_JOB_LOG"
