#!/usr/bin/env bash
# LOCAL driver (N=200/split): runs Deliverable A (VTI POPE beta-grid reproduction)
# then Deliverable B (VTI rotation-strength sweep), sequentially, sharing one
# RUN_DATE. Sized for a single 48 GB A6000 (one model loaded at a time).
#
# Launch + detach:
#   CUDA_VISIBLE_DEVICES=0 nohup bash evaluation/run_scripts/run_beta_grid_local.sh \
#     > /dev/null 2>&1 &
#   tail -f evaluation/results/_logs/beta_grid_local_*.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
LOG_DIR="evaluation/results/_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/beta_grid_local_$(date +%Y%m%d_%H%M%S).log"

# B defaults: uniform_rotation only (gated_rotation is numerically identical —
# see RESEARCH_LOG 2026-06-18). Override with VARIANTS="uniform_rotation gated_rotation".
export VARIANTS="${VARIANTS:-uniform_rotation}"

{
  echo "=== beta_grid_local start $(date)  date=$RUN_DATE  gpu=$CUDA_VISIBLE_DEVICES ==="
  echo ""
  echo "########## A: VTI POPE beta-grid reproduction (N=200) ##########"
  LIMIT=200 bash evaluation/run_scripts/run_vti_pope_beta_grid.sh
  echo ""
  echo "########## B: VTI rotation-strength sweep (n=200) ##########"
  NUM_SAMPLES=200 bash evaluation/vti_rotation_strength/run_scripts/run_rotation_strength_sweep.sh
  echo ""
  echo "=== beta_grid_local done $(date) ==="
} 2>&1 | tee "$LOG"
