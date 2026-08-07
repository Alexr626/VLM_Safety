#!/usr/bin/env bash
# One RunAI job / one H100: three concurrent Qwen workers for remaining β=0.9 cells.
#
# Partitions by cell (layer window × reconstruction), not by sample shard:
#   worker meandiff-5-14  : BETAS=0.9 STEER=raw_mean_difference LAYER_SETS=5-14
#   worker meandiff-15-24 : BETAS=0.9 STEER=raw_mean_difference LAYER_SETS=15-24
#   worker pca-windows    : BETAS=0.9 STEER=live_pc1_plus_mean  LAYER_SETS="all 5-14 15-24"
#
# --skip_if_exists resumes any in-flight checkpoint (e.g. meandiff 5-14).
# Do NOT capture PIDs via $(...); redirect each worker to its own log and take $!.
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
export RUN_DATE="${RUN_DATE:-2026-08-05}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export AMBER_SUBSET="${AMBER_SUBSET:-data/amber/pinned_amber_disc_1500.json}"

LOAD_STAGGER_SEC="${LOAD_STAGGER_SEC:-45}"
JOB_NAME="${JOB_NAME:-amber-qwen-beta09-triple}"

# shellcheck source=runai_job_logging.sh
source "$REPO/helper_scripts/runai/runai_job_logging.sh"
runai_job_logging_start

VERIFY="$REPO/helper_scripts/runai/verify_amber_expanded_qwen_nfs_layout.sh"
python3 - "$VERIFY" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY
bash "$VERIFY"

cd "$REPO"

echo "=== remap baked lambdab2 paths (once) ==="
if [[ -f helper_scripts/runai/remap_lambdab2_paths.py ]]; then
  "$MM" run -p "$ENV_PREFIX" python helper_scripts/runai/remap_lambdab2_paths.py --apply || true
fi

echo "=== prefetch HF weights (once) ==="
"$MM" run -p "$ENV_PREFIX" python -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct'))"

echo "=== GPU before workers ==="
nvidia-smi || true

LF="$REPO/helper_scripts/runai/run_bash_lf.py"
WORKER="$REPO/helper_scripts/runai/run_amber_expanded_qwen_worker.sh"
LOG_DIR="$BASE/logs/runai"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# tag layers recon  — tag is filesystem-safe for log / JOB_NAME
start_worker() {
  local tag="$1"
  local layers="$2"
  local recon="$3"
  local wlog="$LOG_DIR/amber-qwen-b09-${tag}_${TS}.log"
  echo "=== start Qwen β=0.9 worker=${tag} layers=${layers} recon=${recon} (log=$wlog) ==="
  (
    unset RUNAI_JOB_LOG
    export SKIP_REMAP=1 SKIP_HF_PREFETCH=1 SKIP_CONDA_ACTIVATE=1
    export CUDA_VISIBLE_DEVICES=0
    export RUN_DATE MAX_PIXELS AMBER_SUBSET
    export BETAS=0.9
    export LAYER_SETS="$layers"
    export STEER_RECONSTRUCTIONS="$recon"
    export JOB_NAME="amber-qwen-b09-${tag}"
    exec python3 "$LF" "$WORKER" </dev/null >"$wlog" 2>&1
  ) &
  LAST_WORKER_PID=$!
  echo "  pid=$LAST_WORKER_PID"
}

start_worker meandiff-5-14 "5-14" raw_mean_difference
PID_A=$LAST_WORKER_PID

sleep "$LOAD_STAGGER_SEC"
start_worker meandiff-15-24 "15-24" raw_mean_difference
PID_B=$LAST_WORKER_PID

sleep "$LOAD_STAGGER_SEC"
start_worker pca-windows "all 5-14 15-24" live_pc1_plus_mean
PID_C=$LAST_WORKER_PID

echo "=== all β=0.9 cell workers launched; waiting ==="
echo "  meandiff-5-14=$PID_A meandiff-15-24=$PID_B pca-windows=$PID_C"
nvidia-smi || true

ECA=0; ECB=0; ECC=0
wait "$PID_A" || ECA=$?
wait "$PID_B" || ECB=$?
wait "$PID_C" || ECC=$?

echo "=== worker exit codes ==="
echo "  meandiff-5-14=$ECA meandiff-15-24=$ECB pca-windows=$ECC"
nvidia-smi || true

if [[ "$ECA" -ne 0 || "$ECB" -ne 0 || "$ECC" -ne 0 ]]; then
  echo "AMBER_QWEN_BETA09_TRIPLE_FAIL"
  echo "  durable_log=$RUNAI_JOB_LOG"
  echo "  worker_logs=$LOG_DIR/amber-qwen-b09-*_${TS}.log"
  exit 1
fi

echo "AMBER_QWEN_BETA09_TRIPLE_OK"
echo "  durable_log=$RUNAI_JOB_LOG"
echo "  worker_logs=$LOG_DIR/amber-qwen-b09-*_${TS}.log"
exit 0
