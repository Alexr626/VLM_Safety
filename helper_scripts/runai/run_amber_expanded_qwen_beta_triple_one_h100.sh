#!/usr/bin/env bash
# One RunAI job / one H100: three concurrent Qwen AMBER-1500 workers, one per β.
#
#   BETAS=0.2 | 0.5 | 0.9
#   Each runs both steer reconstructions and all three layer windows;
#   --skip_if_exists skips cells already present from the lambdab2 partial.
#
# Stagger model load by LOAD_STAGGER_SEC (default 45).
#
# Concurrency note: do NOT capture PIDs via $(launch ...). Command substitution waits
# until the child's stdout closes, which serializes workers that tee/log to stdout.
# Start each worker with its own redirected log, then take $! in this shell.
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
export LAYER_SETS="${LAYER_SETS:-all 5-14 15-24}"
export STEER_RECONSTRUCTIONS="${STEER_RECONSTRUCTIONS:-raw_mean_difference live_pc1_plus_mean}"

LOAD_STAGGER_SEC="${LOAD_STAGGER_SEC:-45}"
JOB_NAME="${JOB_NAME:-amber-qwen-beta-triple}"

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

# Launch one β-worker in the background. Redirect stdout/stderr to a per-β log so
# the parent is not blocked waiting for the child's stdout to close. Unset
# RUNAI_JOB_LOG inside a subshell so the worker opens its own durable log under
# JOB_NAME (parent's RUNAI_JOB_LOG stays intact).
start_worker() {
  local beta="$1"
  local wlog="$LOG_DIR/amber-qwen-beta-${beta}_${TS}.log"
  echo "=== start Qwen β=${beta} (log=$wlog) ==="
  (
    unset RUNAI_JOB_LOG
    export SKIP_REMAP=1 SKIP_HF_PREFETCH=1 SKIP_CONDA_ACTIVATE=1
    export CUDA_VISIBLE_DEVICES=0
    export RUN_DATE MAX_PIXELS AMBER_SUBSET LAYER_SETS STEER_RECONSTRUCTIONS
    export BETAS="$beta"
    export JOB_NAME="amber-qwen-beta-${beta}"
    exec python3 "$LF" "$WORKER" </dev/null >"$wlog" 2>&1
  ) &
  LAST_WORKER_PID=$!
  echo "  pid=$LAST_WORKER_PID"
}

start_worker 0.2
PID_02=$LAST_WORKER_PID

sleep "$LOAD_STAGGER_SEC"
start_worker 0.5
PID_05=$LAST_WORKER_PID

sleep "$LOAD_STAGGER_SEC"
start_worker 0.9
PID_09=$LAST_WORKER_PID

echo "=== all workers launched; waiting ==="
echo "  beta0.2=$PID_02 beta0.5=$PID_05 beta0.9=$PID_09"
nvidia-smi || true

EC02=0; EC05=0; EC09=0
wait "$PID_02" || EC02=$?
wait "$PID_05" || EC05=$?
wait "$PID_09" || EC09=$?

echo "=== worker exit codes ==="
echo "  beta0.2=$EC02 beta0.5=$EC05 beta0.9=$EC09"
nvidia-smi || true

if [[ "$EC02" -ne 0 || "$EC05" -ne 0 || "$EC09" -ne 0 ]]; then
  echo "AMBER_QWEN_TRIPLE_FAIL"
  echo "  durable_log=$RUNAI_JOB_LOG"
  echo "  worker_logs=$LOG_DIR/amber-qwen-beta-*_${TS}.log"
  exit 1
fi

echo "AMBER_QWEN_TRIPLE_OK"
echo "  durable_log=$RUNAI_JOB_LOG"
echo "  worker_logs=$LOG_DIR/amber-qwen-beta-*_${TS}.log"
exit 0
