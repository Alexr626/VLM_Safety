#!/usr/bin/env bash
# RunAI sync: extract AMBER-Qwen expanded-grid tarballs onto NFS and verify.
set -eu
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

BASE="${BASE:-/home/datalake/romanus}"
REPO="${REPO:-$BASE/vlm_hallucination}"
LOG_DIR="${LOG_DIR:-$BASE/logs/runai}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_NAME="${JOB_NAME:-sync-amber-qwen}"
mkdir -p "$LOG_DIR" "$REPO"
LOG="$LOG_DIR/${JOB_NAME}_${TS}.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== $JOB_NAME ==="
echo "  log_file=$LOG"
echo "  time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

TAR="$BASE/vti_repo_amber_qwen.tar.gz"
test -f "$TAR"
test -f "$BASE/qwen_amber_directions.tar.gz"
test -f "$BASE/amber_images.tar.gz"

echo "=== extracting repo tarball ==="
tar -xzf "$TAR" -C "$REPO"
echo "=== extracting Qwen directions ==="
tar -xzf "$BASE/qwen_amber_directions.tar.gz" -C "$REPO"
echo "=== extracting AMBER images ==="
tar -xzf "$BASE/amber_images.tar.gz" -C "$REPO"
if [[ -f "$BASE/qwen_amber_partial_results.tar.gz" ]]; then
  echo "=== extracting Qwen partial results ==="
  tar -xzf "$BASE/qwen_amber_partial_results.tar.gz" -C "$REPO"
else
  echo "=== no partial results tarball (skip) ==="
fi

export BASE REPO RUN_DATE="${RUN_DATE:-2026-08-05}"
VERIFY="$REPO/helper_scripts/runai/verify_amber_expanded_qwen_nfs_layout.sh"
python3 - "$VERIFY" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY
bash "$VERIFY"

echo "SYNC_OK"
echo "  durable_log=$LOG"
