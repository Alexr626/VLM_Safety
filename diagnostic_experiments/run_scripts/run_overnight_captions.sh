#!/bin/bash
# ============================================================
# Overnight: caption USU/SUU/UUU eval samples for the probe
# experiment. Pure CPU + Anthropic API. Safe to run alongside
# GPU work (run_overnight_comp_directions.sh).
# ============================================================
# Recommended invocation (writes log to disk, survives logout):
#
#   mkdir -p logs
#   nohup bash diagnostic_experiments/run_scripts/run_overnight_captions.sh \
#       > logs/captions_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $! > logs/captions.pid
#
# Then tail with: tail -f logs/captions_*.log
# Stop with:      kill $(cat logs/captions.pid)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda env if not already active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "vlm_safety" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate vlm_safety
fi

CAPTION_PROVIDER="${CAPTION_PROVIDER:-anthropic}"

echo "============================================================"
echo " Overnight Captioning  ($(date))"
echo "============================================================"
echo "  provider : $CAPTION_PROVIDER"
echo "  python   : $(which python)"
echo "  env      : ${CONDA_DEFAULT_ENV:-<none>}"
echo

# 1. Ensure the train_eval_split.json contains usu/suu/uuu eval id lists
#    (idempotent; near-instant if already populated).
echo "--- [1/2] Extending HoliSafe train_eval_split.json ---"
python -m src.dataset --n_eval 175 --seed 42

# 2. Caption USU/SUU/UUU eval samples. Resume-safe via the per-sample API
#    checkpoint at data/captions/holisafe.checkpoint.json. On completion the
#    new captions are merged into data/captions/holisafe.json (the existing
#    1,400 SSS+SSU captions are preserved by the merge-load in main()).
echo
echo "--- [2/2] Captioning USU/SUU/UUU eval samples ($CAPTION_PROVIDER) ---"
python data_scripts/generate_captions.py \
    --dataset holisafe \
    --holisafe_eval_only \
    --holisafe_subsets SUU USU UUU \
    --provider "$CAPTION_PROVIDER"

echo
echo "============================================================"
echo " Done.   $(date)"
echo "============================================================"
