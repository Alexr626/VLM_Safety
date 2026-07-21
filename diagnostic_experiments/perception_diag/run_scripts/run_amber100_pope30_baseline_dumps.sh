#!/usr/bin/env bash
# Baseline-only steered capture dumps for AMBER-100 and POPE-30 (existence×yes).
# Separate run tags from amber25 dumps. max_new_tokens=128 for both models.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source /data/romanus/miniconda3/etc/profile.d/conda.sh
conda activate vlm_hallucination_mitigation
export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

AUG_AMBER=data/amber/augmented_amber100.jsonl
AUG_POPE=data/pope/augmented_pope30.jsonl
test -f "$AUG_AMBER"
test -f "$AUG_POPE"

MODEL="${1:?usage: $0 llava|qwen25}"
case "$MODEL" in
  llava)
    HF_ID="llava-hf/llava-1.5-7b-hf"
    EXTRA=()
    ;;
  qwen25)
    HF_ID="Qwen/Qwen2.5-VL-7B-Instruct"
    EXTRA=(--max_pixels 1003520)
    ;;
  *)
    echo "MODEL must be llava or qwen25" >&2
    exit 1
    ;;
esac

run_one () {
  local aug="$1" tag="$2"
  echo "[dump] model=$MODEL tag=$tag gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python diagnostic_experiments/perception_diag/run_dump.py \
    --model "$HF_ID" \
    --augmented_jsonl "$aug" \
    --cells baseline \
    --run_tag "$tag" \
    --max_new_tokens 128 \
    --device_map cuda:0 \
    "${EXTRA[@]}"
}

run_one "$AUG_AMBER" "amber100_baseline"
run_one "$AUG_POPE" "pope30_existence_yes_baseline"
echo "Done for $MODEL"
