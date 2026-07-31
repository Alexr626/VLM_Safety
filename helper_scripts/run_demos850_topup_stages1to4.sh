#!/usr/bin/env bash
# Detached demos_850 top-up: stages 1 → 4 (resume-safe).
# Stage 0 must already have written stage0_candidates_topup_2026-07-28.jsonl.
# Do NOT run run_full.sh. Do NOT touch demos_v2.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

TOPUP="${TOPUP:-data/vti/v2/stage0_candidates_topup_2026-07-28.jsonl}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/demos850_topup_stages1to4_${STAMP}.log"

echo "=== demos_850 top-up stages 1-4 ===" | tee -a "$LOG"
echo "cwd=$(pwd) topup=$TOPUP log=$LOG" | tee -a "$LOG"
echo "started=$(date -Is)" | tee -a "$LOG"

if [[ ! -f "$TOPUP" ]]; then
  echo "ERROR: missing $TOPUP — run stage0 first; do not invent a new mine here." | tee -a "$LOG"
  exit 1
fi

echo "--- stage1 ---" | tee -a "$LOG"
python data_scripts/vti_demos_v2/stage1_verify_anchors.py --input "$TOPUP" 2>&1 | tee -a "$LOG"

echo "--- stage1b ---" | tee -a "$LOG"
python data_scripts/vti_demos_v2/stage1b_allocate.py 2>&1 | tee -a "$LOG"

echo "--- stage2 ---" | tee -a "$LOG"
python data_scripts/vti_demos_v2/stage2_write_truthful.py 2>&1 | tee -a "$LOG"

echo "--- stage3 ---" | tee -a "$LOG"
python data_scripts/vti_demos_v2/stage3_make_variants.py 2>&1 | tee -a "$LOG"

echo "--- stage4 ---" | tee -a "$LOG"
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py 2>&1 | tee -a "$LOG"

echo "finished=$(date -Is)" | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"
