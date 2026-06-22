#!/usr/bin/env bash
# RunAI driver (N=3000/split): runs Deliverable A (VTI POPE beta-grid
# reproduction) then Deliverable B (VTI rotation-strength sweep) at the paper's
# 3000-samples-per-split scale. Same logic as the local driver, only N changes.
#
# Intended to be the command your RunAI job spec executes. Device policy: VTI
# direction extraction + rotation steering MUST run with the full model on a
# single device (no offload/sharding) — request one whole GPU for the job.
#
#   CUDA_VISIBLE_DEVICES=0 bash evaluation/run_scripts/run_beta_grid_runai.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
LOG_DIR="evaluation/results/_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/beta_grid_runai_$(date +%Y%m%d_%H%M%S).log"

export VARIANTS="${VARIANTS:-uniform_rotation}"

{
  echo "=== beta_grid_runai start $(date)  date=$RUN_DATE  gpu=$CUDA_VISIBLE_DEVICES ==="
  echo ""
  echo "########## A: VTI POPE beta-grid reproduction (N=3000) ##########"
  LIMIT=3000 bash evaluation/run_scripts/run_vti_pope_beta_grid.sh
  echo ""
  echo "########## B: VTI rotation-strength sweep (n=3000) ##########"
  NUM_SAMPLES=3000 bash evaluation/vti_rotation_strength/run_scripts/run_rotation_strength_sweep.sh
  echo ""
  echo "=== beta_grid_runai done $(date) ==="
} 2>&1 | tee "$LOG"
