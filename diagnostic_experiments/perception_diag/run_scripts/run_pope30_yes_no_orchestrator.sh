#!/usr/bin/env bash
# Orchestrator: wait for GPU 0 to free, run concurrent LLaVA yes+no, then concurrent Qwen yes+no.
# Plan: implementation_plans/7-21-26/pope_yes_no_windowed_steering_control_plan_2026-07-22.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/diagnostic_experiments/perception_diag/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="$LOG_DIR/pope30_yes_no_orchestrator_${STAMP}.log"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
# MiB used below this threshold counts as "free enough" (driver overhead only).
FREE_USED_MIB="${FREE_USED_MIB:-1500}"
POLL_SEC="${POLL_SEC:-60}"

log() { echo "[$(date -Is)] $*" | tee -a "$ORCH_LOG"; }

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' '
}

# Also treat the known notebook nbconvert as the blocking process.
blocking_pids() {
  pgrep -f 'jupyter-nbconvert.*Romanus_Alexander_final_project' || true
  pgrep -f 'run_dump.py.*pope30_(yes|no)_windowed_steering' || true
}

log "Orchestrator start GPU=$GPU free_threshold_mib=$FREE_USED_MIB"
log "Waiting for GPU $GPU to free (used < ${FREE_USED_MIB} MiB) and blocking jobs to exit..."

while true; do
  used="$(gpu_used_mib || echo 99999)"
  blockers="$(blocking_pids | tr '\n' ' ')"
  log "poll: gpu${GPU}_used_mib=$used blockers='${blockers:-none}'"
  # Wait until our known notebook is gone AND memory is low.
  # Ignore other labmates' leftover allocations only if memory is already low.
  if [[ -z "${blockers}" ]] && [[ "$used" -lt "$FREE_USED_MIB" ]]; then
    log "GPU $GPU free (used=${used} MiB). Launching LLaVA concurrent pair."
    break
  fi
  # If notebook is gone but memory still high from something else, keep waiting
  # unless used dropped below a soft threshold of half the card (~24GB) and no
  # our processes remain — then proceed carefully. Prefer strict free.
  sleep "$POLL_SEC"
done

bash "$ROOT/diagnostic_experiments/perception_diag/run_scripts/run_llava_pope30_yes_no_concurrent.sh" \
  >>"$ORCH_LOG" 2>&1
llava_ec=$?
log "LLaVA concurrent pair finished (exit=$llava_ec). Launching Qwen concurrent pair."

bash "$ROOT/diagnostic_experiments/perception_diag/run_scripts/run_qwen_pope30_yes_no_concurrent.sh" \
  >>"$ORCH_LOG" 2>&1
qwen_ec=$?
log "Qwen concurrent pair finished (exit=$qwen_ec). Orchestrator complete."

# Append factual completion stub to RESEARCH_LOG (numbers live in dump trees / summary JSON).
python - <<'PY' >>"$ORCH_LOG" 2>&1
from pathlib import Path
from datetime import datetime
root = Path("/home/romanus/dev/vlm_hallucination_mitigation_summer_2026")
log = root / "RESEARCH_LOG.md"
stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
entry = f"""
## {stamp.split()[0]} — POPE-30-yes/no windowed steering control: dumps finished

**Orchestrator log:** see `diagnostic_experiments/perception_diag/logs/pope30_yes_no_orchestrator_*.log` (completion ~{stamp}).
**Dump roots:** `data/pope/dumps/{{llava-1.5-7b-hf,qwen2.5-vl-7b-instruct}}/{{pope30_yes_windowed_steering,pope30_no_windowed_steering}}/`
**Post-process:** reproducibility JSON under `windowed_steering_summary/pope30_yes_reproducibility_{{llava,qwen}}.json`; consolidated + `pope30_{{yes,no}}_mlp_2x2_*.json` rebuilt by drivers.
**Exit codes:** recorded in orchestrator log (LLaVA then Qwen concurrent pairs).
"""
log.write_text(log.read_text() + entry)
print("appended RESEARCH_LOG completion stub")
PY

exit $(( llava_ec != 0 || qwen_ec != 0 ? 1 : 0 ))
