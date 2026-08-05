#!/usr/bin/env bash
# Shared preamble for RunAI steering jobs: tee stdout/stderr to a durable NFS log
# and optionally run layout verification. Source this; do not execute.
runai_job_logging_start() {
  : "${BASE:?BASE must be set}"
  : "${JOB_NAME:?JOB_NAME must be set}"
  local log_dir="${LOG_DIR:-$BASE/logs/runai}"
  local ts
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  RUNAI_JOB_LOG="${RUNAI_JOB_LOG:-$log_dir/${JOB_NAME}_${ts}.log}"
  export RUNAI_JOB_LOG
  # shellcheck disable=SC2094
  exec > >(tee -a "$RUNAI_JOB_LOG") 2>&1
  echo "LOG_FILE=$RUNAI_JOB_LOG"
  echo "JOB_NAME=$JOB_NAME"
  echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

runai_verify_steering_layout() {
  : "${REPO:?REPO must be set}"
  local verify="$REPO/helper_scripts/runai/verify_steering_nfs_layout.sh"
  if [[ ! -f "$verify" ]]; then
    echo "VERIFY_SYNC_FAIL missing $verify"
    return 1
  fi
  # Strip CRLF in-place copy via python, then bash — avoid nested run_bash_lf
  # (outer job is already running under run_bash_lf).
  python3 - "$verify" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY
  bash "$verify"
}
