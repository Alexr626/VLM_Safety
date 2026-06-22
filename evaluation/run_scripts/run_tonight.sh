#!/usr/bin/env bash
# Overnight driver: runs Deliverable A (VTI reproduction) FIRST so the meeting
# results are guaranteed, then Deliverable B (layer-rotation sweep). Sequential
# by design — only one model is ever loaded, so a single 48 GB A6000 won't OOM.
#
# Logs everything (timestamped) to evaluation/results/_logs/.
#
# Launch and detach (survives logout):
#   CUDA_VISIBLE_DEVICES=0 nohup bash evaluation/run_scripts/run_tonight.sh \
#     > /dev/null 2>&1 &
# then check progress with:
#   tail -f evaluation/results/_logs/run_tonight_*.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="evaluation/results/_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_tonight_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== run_tonight start $(date) on GPU $CUDA_VISIBLE_DEVICES ==="
  echo ""
  echo "########## PHASE A: VTI reproduction ##########"
  bash evaluation/run_scripts/run_vti_pope_repro.sh
  echo ""
  echo "########## PHASE B: VTI rotation-strength sweep ##########"
  bash evaluation/vti_rotation_strength/run_scripts/run_rotation_strength_sweep.sh
  echo ""
  echo "=== run_tonight done $(date) ==="
} 2>&1 | tee "$LOG"
