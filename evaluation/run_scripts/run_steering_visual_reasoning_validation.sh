#!/usr/bin/env bash
# Textual mean-difference steering validation grid (one model per process).
#
# Execution order (settled): benchmark outermost AMBER → CHAIR → POPE;
# within each benchmark: baseline, then layer sets (all → early-middle → late),
# within layer set: nd 50 → 500 → 200 → 100, betas 0.2 → 0.5 → 0.9.
#
# Env knobs: BENCHMARKS BETAS NUM_DEMOS LAYER_SETS RUN_DATE OUTPUT_DIR
#            CUDA_VISIBLE_DEVICES MAX_PIXELS CHAIR_CAP CHAIR_PROMPT
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <HF_MODEL_ID>" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

RUN_DATE="${RUN_DATE:-2026-07-30}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
CHAIR_PROMPT="${CHAIR_PROMPT:-Please Describe this image in detail.}"
CHAIR_CAP="${CHAIR_CAP:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"

CONDA_ENV="${CONDA_ENV:-vlm_hallucination_mitigation}"
# On lambdab2: activate conda. On RunAI: outer `micromamba run -p ... bash` already
# provides the env — set SKIP_CONDA_ACTIVATE=1 (see helper_scripts/runai/*).
if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
  :
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "[error] conda not found; set SKIP_CONDA_ACTIVATE=1 if already in env" >&2
  exit 2
fi

MODEL_SHORT="$(python - <<PY
from src.model import _normalize_model_name
print(_normalize_model_name("$MODEL"))
PY
)"

PIX_ARGS=()
if [[ "$MODEL" == *[Qq]wen2* || "$MODEL" == *[Qq]wen/Qwen2* ]]; then
  PIX_ARGS=(--max_pixels "$MAX_PIXELS")
fi

if [[ -n "${LAYER_SETS:-}" ]]; then
  read -r -a LAYER_SET_ARR <<< "$LAYER_SETS"
elif [[ "$MODEL_SHORT" == "llava-1.5-7b-hf" ]]; then
  LAYER_SET_ARR=(all 5-14 20-29)
else
  LAYER_SET_ARR=(all 5-14 15-24)
fi

read -r -a BENCH_ARR <<< "${BENCHMARKS:-amber chair pope}"
read -r -a BETA_ARR <<< "${BETAS:-0.2 0.5 0.9}"
read -r -a ND_ARR <<< "${NUM_DEMOS:-50 500 200 100}"

AMBER_SUBSET="data/amber/pinned_amber_disc_450.json"
CHAIR_SUBSET="data/chair/pinned_chair_500.json"
ANALYSIS_DIR="$OUTPUT_DIR/$RUN_DATE/_analysis_steering_visual_reasoning_validation"
mkdir -p "$ANALYSIS_DIR" logs

layer_block_name() {
  case "$1" in
    all) echo "all-layers block (layers all)" ;;
    5-14) echo "early-middle-window block (layers 5-14)" ;;
    20-29) echo "late-window block (layers 20-29)" ;;
    15-24) echo "late-window block (layers 15-24)" ;;
    *) echo "layer-set block (layers $1)" ;;
  esac
}

directions_dir_for() {
  local nd="$1"
  echo "experiment_artifacts/vti/${MODEL_SHORT}/textual_v2/demos850_ba05bd96_all_nd${nd}_s42_meandiff_partition"
}

write_run_manifest() {
  local manifest="$ANALYSIS_DIR/run_manifest_${MODEL_SHORT}.json"
  python - <<PY
import json, subprocess, hashlib
from datetime import datetime
from pathlib import Path
from src.paths import project_root, vti_demos_850_path, vti_demos_850_partition_path

def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(project_root()), text=True, stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    commit = None

dirs = {
    str(n): str(Path("experiment_artifacts/vti") / "$MODEL_SHORT" / "textual_v2"
               / f"demos850_ba05bd96_all_nd{n}_s42_meandiff_partition")
    for n in [50, 100, 200, 500]
}
pix = None
if "${PIX_ARGS[*]}":
    pix = int("$MAX_PIXELS")
payload = {
    "model": "$MODEL",
    "model_short": "$MODEL_SHORT",
    "run_date": "$RUN_DATE",
    "git_commit": commit,
    "max_pixels": pix,
    "chair_max_new_tokens": int("$CHAIR_CAP"),
    "chair_prompt": "$CHAIR_PROMPT",
    "max_new_tokens": int("$MAX_NEW_TOKENS"),
    "amber_subset": "$AMBER_SUBSET",
    "chair_subset": "$CHAIR_SUBSET",
    "amber_subset_hash_sha256_16": sha16("$AMBER_SUBSET"),
    "chair_subset_hash_sha256_16": sha16("$CHAIR_SUBSET"),
    "demos_850_hash_sha256_16": sha16(vti_demos_850_path()),
    "partition_hash_sha256_16": sha16(vti_demos_850_partition_path()),
    "betas": [float(x) for x in "${BETA_ARR[*]}".split()],
    "num_demos_order": [int(x) for x in "${ND_ARR[*]}".split()],
    "layer_sets": "${LAYER_SET_ARR[*]}".split(),
    "benchmarks": "${BENCH_ARR[*]}".split(),
    "direction_slugs": dirs,
    "intervention": "vti_textual_additive_mlp",
    "steer_reconstruction": "raw_mean_difference",
    "launched_at": datetime.now().isoformat(timespec="seconds"),
}
Path("$manifest").write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote $manifest")
PY
}

# Extra args after the intervention name (e.g. --beta ...) are run_eval flags.
run_eval_amber() {
  local iv="$1"; shift
  python evaluation/run_eval.py \
    --model "$MODEL" --benchmarks amber --amber_task discriminative \
    --interventions "$iv" "$@" \
    --subset_ids_file "$AMBER_SUBSET" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --run_date "$RUN_DATE" --output_dir "$OUTPUT_DIR" --skip_if_exists \
    "${PIX_ARGS[@]}"
}

run_eval_chair() {
  local iv="$1"; shift
  python evaluation/run_eval.py \
    --model "$MODEL" --benchmarks chair \
    --interventions "$iv" "$@" \
    --subset_ids_file "$CHAIR_SUBSET" \
    --chair_prompt "$CHAIR_PROMPT" \
    --chair_max_new_tokens "$CHAIR_CAP" \
    --run_date "$RUN_DATE" --output_dir "$OUTPUT_DIR" --skip_if_exists \
    "${PIX_ARGS[@]}"
}

run_eval_pope() {
  local iv="$1"; shift
  local split
  for split in random popular adversarial; do
    python evaluation/run_eval.py \
      --model "$MODEL" --benchmarks pope --pope_split "$split" \
      --limit 200 \
      --interventions "$iv" "$@" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --run_date "$RUN_DATE" --output_dir "$OUTPUT_DIR" --skip_if_exists \
      "${PIX_ARGS[@]}"
  done
}

run_cell() {
  local bench="$1"; shift
  case "$bench" in
    amber) run_eval_amber "$@" ;;
    chair) run_eval_chair "$@" ;;
    pope)  run_eval_pope "$@" ;;
    *) echo "Unknown benchmark: $bench" >&2; return 1 ;;
  esac
}

echo "=== steering visual reasoning validation ==="
echo "  model      : $MODEL ($MODEL_SHORT)"
echo "  run_date   : $RUN_DATE"
echo "  benchmarks : ${BENCH_ARR[*]}"
echo "  layer_sets : ${LAYER_SET_ARR[*]}"
echo "  num_demos  : ${ND_ARR[*]}"
echo "  betas      : ${BETA_ARR[*]}"
echo "  chair_cap  : $CHAIR_CAP"
echo "  gpu        : $CUDA_VISIBLE_DEVICES"
echo "  max_pixels : ${PIX_ARGS[*]:-n/a}"

write_run_manifest

for pin in "$AMBER_SUBSET" "$CHAIR_SUBSET" "data/chair/combined.json"; do
  if [[ ! -f "$pin" ]]; then
    echo "[error] missing required data file: $pin" >&2
    echo "[error] On RunAI, re-sync a pack that overlays gitignored chair data (pin + combined.json)." >&2
    exit 1
  fi
done

for BENCH in "${BENCH_ARR[@]}"; do
  if [[ "$BENCH" == "pope" ]]; then
    for jf in data/pope/output/coco/coco_pope_random.json \
              data/pope/output/coco/coco_pope_popular.json \
              data/pope/output/coco/coco_pope_adversarial.json \
              data/pope/combined.json; do
      if [[ ! -f "$jf" ]]; then
        echo "[error] missing POPE data file: $jf" >&2
        exit 1
      fi
    done
  fi
done

for BENCH in "${BENCH_ARR[@]}"; do
  echo ""
  echo "============================================================"
  echo "=== ${BENCH^^} | baseline (no_intervention) ==="
  echo "============================================================"
  run_cell "$BENCH" no_intervention \
    || echo "[warn] baseline failed: $BENCH (continuing)"

  for LAYER_SET in "${LAYER_SET_ARR[@]}"; do
    BLOCK_NAME="$(layer_block_name "$LAYER_SET")"
    for ND in "${ND_ARR[@]}"; do
      DIR="$(directions_dir_for "$ND")"
      if [[ ! -f "$DIR/directions.npz" ]]; then
        echo "[error] missing directions: $DIR" >&2
        exit 1
      fi
      for BETA in "${BETA_ARR[@]}"; do
        echo ""
        echo "=== ${BENCH^^} | ${BLOCK_NAME} | nd=${ND} | beta=${BETA} ==="
        run_cell "$BENCH" vti_textual_additive_mlp \
          --beta "$BETA" --layer_set "$LAYER_SET" --directions_dir "$DIR" \
          || echo "[warn] steered cell failed: $BENCH layers=$LAYER_SET nd=$ND beta=$BETA (continuing)"
      done
    done
    echo ""
    echo "=== comparison table after ${BENCH^^} | ${BLOCK_NAME} ==="
    python - <<PY || true
from evaluation.runners import print_comparison_table
print_comparison_table("$MODEL_SHORT", "$OUTPUT_DIR", run_date="$RUN_DATE")
PY
  done
done

echo ""
echo "=== grid process finished for $MODEL_SHORT ==="
