#!/usr/bin/env bash
# RunAI sync: extract LLaVA+Qwen steering tarballs onto NFS, verify, log.
set -eu
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

BASE="${BASE:-/home/datalake/romanus}"
REPO="${REPO:-$BASE/vlm_hallucination}"
LOG_DIR="${LOG_DIR:-$BASE/logs/runai}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_NAME="${JOB_NAME:-sync-steering}"
mkdir -p "$LOG_DIR" "$REPO"
LOG="$LOG_DIR/${JOB_NAME}_${TS}.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== $JOB_NAME ==="
echo "  log_file=$LOG"
echo "  time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

TAR="$BASE/vti_repo_working_tree.tar.gz"
[[ -f "$TAR" ]] || TAR="$BASE/vti_repo.tar.gz"
echo "  repo_tar=$TAR"
test -f "$TAR"
test -f "$BASE/llava_meandiff_directions.tar.gz"
test -f "$BASE/qwen_meandiff_directions.tar.gz"

echo "=== extracting repo tarball ==="
tar -xzf "$TAR" -C "$REPO"
echo "=== extracting LLaVA meandiff directions ==="
tar -xzf "$BASE/llava_meandiff_directions.tar.gz" -C "$REPO"
echo "=== extracting Qwen meandiff directions ==="
tar -xzf "$BASE/qwen_meandiff_directions.tar.gz" -C "$REPO"
if [[ -f "$BASE/llava_partial_results_chair.tar.gz" ]]; then
  echo "=== extracting partial LLaVA CHAIR results ==="
  tar -xzf "$BASE/llava_partial_results_chair.tar.gz" -C "$REPO"
else
  echo "=== no llava_partial_results_chair.tar.gz (skip) ==="
fi

export BASE REPO
export MODEL_SHORTS="${MODEL_SHORTS:-llava-1.5-7b-hf qwen2.5-vl-7b-instruct}"
VERIFY="$REPO/helper_scripts/runai/verify_steering_nfs_layout.sh"
python3 - "$VERIFY" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY
bash "$VERIFY"

echo "SYNC_OK"
echo "  durable_log=$LOG"
