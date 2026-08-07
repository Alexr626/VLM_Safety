#!/usr/bin/env bash
# Expanded AMBER discriminative 1500-item grid under baseline, mean-difference,
# and VTI PCA (PC1 + mean) steering. One model per process.
#
# Execution order: AMBER baseline → mean-difference block → VTI PCA block.
# Within each steering block: layer windows all → 5-14 → late; betas 0.2 → 0.5 → 0.9.
#
# Env knobs: RUN_DATE OUTPUT_DIR BETAS LAYER_SETS STEER_RECONSTRUCTIONS
#            MAX_NEW_TOKENS MAX_PIXELS CUDA_VISIBLE_DEVICES AMBER_SUBSET
#            SKIP_CONDA_ACTIVATE
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

RUN_DATE="${RUN_DATE:-2026-08-05}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
AMBER_SUBSET="${AMBER_SUBSET:-data/amber/pinned_amber_disc_1500.json}"

CONDA_ENV="${CONDA_ENV:-vlm_hallucination_mitigation}"
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

read -r -a BETA_ARR <<< "${BETAS:-0.2 0.5 0.9}"
read -r -a RECON_ARR <<< "${STEER_RECONSTRUCTIONS:-raw_mean_difference live_pc1_plus_mean}"

ANALYSIS_DIR="$OUTPUT_DIR/$RUN_DATE/_analysis_steering_vector_validation_continuation"
mkdir -p "$ANALYSIS_DIR" logs

if [[ ! -f "$AMBER_SUBSET" ]]; then
  echo "[error] missing AMBER subset: $AMBER_SUBSET" >&2
  exit 1
fi

layer_block_name() {
  case "$1" in
    all) echo "all-layers window (layers all)" ;;
    5-14) echo "early-middle window (layers 5-14)" ;;
    20-29) echo "late window (layers 20-29)" ;;
    15-24) echo "late window (layers 15-24)" ;;
    *) echo "layer window (layers $1)" ;;
  esac
}

recon_stem() {
  case "$1" in
    raw_mean_difference) echo "meandiff" ;;
    live_pc1_plus_mean) echo "r2" ;;
    *)
      echo "[error] unknown steer_reconstruction: $1" >&2
      exit 1
      ;;
  esac
}

recon_banner() {
  case "$1" in
    raw_mean_difference) echo "mean-difference direction" ;;
    live_pc1_plus_mean) echo "VTI PCA (PC1 + mean) direction" ;;
    *) echo "$1" ;;
  esac
}

directions_dir_for() {
  local stem
  stem="$(recon_stem "$1")"
  echo "experiment_artifacts/vti/${MODEL_SHORT}/textual_v2/demos850_ba05bd96_all_nd500_s42_${stem}_partition"
}

write_run_manifest() {
  local manifest="$ANALYSIS_DIR/run_manifest_${MODEL_SHORT}.json"
  python - <<PY
import json, subprocess, hashlib
from datetime import datetime
from pathlib import Path
from src.paths import project_root

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(project_root()), text=True, stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    commit = None

recon_dirs = {}
for recon, stem in [
    ("raw_mean_difference", "meandiff"),
    ("live_pc1_plus_mean", "r2"),
]:
    d = Path("experiment_artifacts/vti") / "$MODEL_SHORT" / "textual_v2" / (
        f"demos850_ba05bd96_all_nd500_s42_{stem}_partition"
    )
    meta = json.loads((d / "metadata.json").read_text())
    recon_dirs[recon] = {
        "directions_dir": str(d),
        "directions_npz_sha256": sha256(d / "directions.npz"),
        "metadata_json_sha256": sha256(d / "metadata.json"),
        "steer_reconstruction": meta.get("steer_reconstruction"),
        "n_pairs": meta.get("n_pairs"),
        "slug": meta.get("slug"),
    }

pix = int("$MAX_PIXELS") if "${PIX_ARGS[*]}" else None
payload = {
    "model": "$MODEL",
    "model_short": "$MODEL_SHORT",
    "run_date": "$RUN_DATE",
    "git_commit": commit,
    "max_pixels": pix,
    "max_new_tokens": int("$MAX_NEW_TOKENS"),
    "amber_subset": "$AMBER_SUBSET",
    "amber_subset_sha256": sha256("$AMBER_SUBSET"),
    "betas": [float(x) for x in "${BETA_ARR[*]}".split()],
    "layer_sets": "${LAYER_SET_ARR[*]}".split(),
    "steer_reconstructions": "${RECON_ARR[*]}".split(),
    "direction_directories": recon_dirs,
    "intervention": "vti_textual_additive_mlp",
    "launched_at": datetime.now().isoformat(timespec="seconds"),
}
Path("$manifest").write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote $manifest")
PY
}

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

echo "=== AMBER expanded steering direction grid ==="
echo "  model      : $MODEL ($MODEL_SHORT)"
echo "  run_date   : $RUN_DATE"
echo "  subset     : $AMBER_SUBSET"
echo "  layer_sets : ${LAYER_SET_ARR[*]}"
echo "  betas      : ${BETA_ARR[*]}"
echo "  recons     : ${RECON_ARR[*]}"
echo "  gpu        : $CUDA_VISIBLE_DEVICES"
echo "  max_pixels : ${PIX_ARGS[*]:-n/a}"

# Fail closed if any required direction set is missing.
for RECON in "${RECON_ARR[@]}"; do
  DIR="$(directions_dir_for "$RECON")"
  if [[ ! -f "$DIR/directions.npz" ]]; then
    echo "[error] missing directions: $DIR" >&2
    exit 1
  fi
done

write_run_manifest

echo ""
echo "============================================================"
echo "=== AMBER 1500 | baseline (no_intervention) ==="
echo "============================================================"
run_eval_amber no_intervention \
  || echo "[warn] baseline failed (continuing)"

for RECON in "${RECON_ARR[@]}"; do
  DIR="$(directions_dir_for "$RECON")"
  BANNER="$(recon_banner "$RECON")"
  echo ""
  echo "============================================================"
  echo "=== AMBER 1500 | ${BANNER} block ==="
  echo "============================================================"
  for LAYER_SET in "${LAYER_SET_ARR[@]}"; do
    BLOCK_NAME="$(layer_block_name "$LAYER_SET")"
    for BETA in "${BETA_ARR[@]}"; do
      echo ""
      echo "=== AMBER 1500 | ${BANNER} | ${BLOCK_NAME} | beta=${BETA} ==="
      run_eval_amber vti_textual_additive_mlp \
        --beta "$BETA" --layer_set "$LAYER_SET" --directions_dir "$DIR" \
        || echo "[warn] steered cell failed: recon=$RECON layers=$LAYER_SET beta=$BETA (continuing)"
    done
    echo ""
    echo "=== comparison table after AMBER 1500 | ${BANNER} | ${BLOCK_NAME} ==="
    python - <<PY || true
from evaluation.runners import print_comparison_table
print_comparison_table("$MODEL_SHORT", "$OUTPUT_DIR", run_date="$RUN_DATE")
PY
  done
done

echo ""
echo "=== grid process finished for $MODEL_SHORT ==="
