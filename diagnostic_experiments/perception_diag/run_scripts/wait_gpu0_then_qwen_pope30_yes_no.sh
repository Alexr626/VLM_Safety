#!/usr/bin/env bash
# Wait for all current GPU-0 jobs to exit, then launch concurrent Qwen POPE-30-yes/no.
# Snapshot of blockers at watcher start (notebook + LLaVA dumps + LLaVA wrapper).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/diagnostic_experiments/perception_diag/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/qwen_after_gpu0_clear_${STAMP}.log"
POLL_SEC="${POLL_SEC:-120}"

# PIDs that were on GPU 0 / related wrappers when Alex asked to queue Qwen.
BLOCK_PIDS=(
  3849009   # jupyter-nbconvert parent
  3849013   # notebook python on GPU 0
  3907181   # LLaVA pope30_no run_dump
  3908945   # LLaVA pope30_yes run_dump
  3907167   # LLaVA concurrent wrapper
  3907176
  3908940
)

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

still_alive() {
  local alive=()
  for pid in "${BLOCK_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  # Also any LLaVA pope30 dump that might still be running under a new pid
  local dumps
  dumps="$(pgrep -f 'run_dump.py.*llava-hf/llava.*pope30_(yes|no)_windowed_steering' || true)"
  if [[ -n "$dumps" ]]; then
    alive+=($dumps)
  fi
  # jupyter nbconvert for the same notebook
  local nb
  nb="$(pgrep -f 'jupyter-nbconvert.*Romanus_Alexander_final_project' || true)"
  if [[ -n "$nb" ]]; then
    alive+=($nb)
  fi
  # LLaVA concurrent wrapper post-process
  local wrap
  wrap="$(pgrep -f 'run_llava_pope30_yes_no_concurrent.sh' || true)"
  if [[ -n "$wrap" ]]; then
    alive+=($wrap)
  fi
  # Deduplicate
  if ((${#alive[@]})); then
    printf '%s\n' "${alive[@]}" | sort -u | tr '\n' ' '
  fi
}

log "Watcher start: queue Qwen concurrent yes+no after GPU-0 blockers clear"
log "Initial blockers: $(still_alive)"

while true; do
  alive="$(still_alive)"
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ' || echo '?')"
  if [[ -z "${alive// }" ]]; then
    log "All tracked GPU-0 jobs gone (gpu0_used_mib=$used). Launching Qwen."
    break
  fi
  log "poll: still_alive=[$alive] gpu0_used_mib=$used"
  sleep "$POLL_SEC"
done

bash "$ROOT/diagnostic_experiments/perception_diag/run_scripts/run_qwen_pope30_yes_no_concurrent.sh" \
  >>"$LOG" 2>&1
ec=$?
log "Qwen concurrent finished exit=$ec"
exit "$ec"
