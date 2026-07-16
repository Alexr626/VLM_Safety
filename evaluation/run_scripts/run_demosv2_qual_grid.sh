#!/usr/bin/env bash
# demos_v2.1 qualitative steering grid (CHAIR-5 + AMBER-25).
#
# Per-model pipeline:
#   1) Extract `all` directions (nd∈{50,100,200,500})
#   2) Smoke gate (1 CHAIR image, additive_mlp β=0.5, dim=all, nd=500)
#   3) Baselines → full `all` eval slice → remaining dimensions
#   4) Extract remaining dims after `all` directions land (before/during eval)
#   5) Render HTML galleries for completed dimension slices
#
# Launch two instances concurrently on the same free A6000:
#   CUDA_VISIBLE_DEVICES=0 nohup bash evaluation/run_scripts/run_demosv2_qual_grid.sh \
#     llava-hf/llava-1.5-7b-hf > logs/demosv2_qual_llava_2026-07-13.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=0 nohup bash evaluation/run_scripts/run_demosv2_qual_grid.sh \
#     Qwen/Qwen2.5-VL-7B-Instruct > logs/demosv2_qual_qwen_2026-07-13.log 2>&1 &
#
# Env knobs: RUN_DATE OUTPUT_DIR DEMOS_PATH SUBSET_IDS CUDA_VISIBLE_DEVICES
#            CHAIR_CAP CHAIR_PROMPT MAX_PIXELS BETAS NUM_DEMOS DIMENSIONS
#            SKIP_EXTRACT SKIP_SMOKE SKIP_EVAL SKIP_GALLERY
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

RUN_DATE="${RUN_DATE:-2026-07-13}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
DEMOS_PATH="${DEMOS_PATH:-data/vti/demos_v2.jsonl}"
SUBSET_IDS="${SUBSET_IDS:-data/vti/qual_subset_chair5_amber25.json}"
CHAIR_PROMPT="${CHAIR_PROMPT:-Please Describe this image in detail.}"
CHAIR_CAP="${CHAIR_CAP:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
RANK="${RANK:-2}"
SEED="${SEED:-42}"

# Qwen only; ignored for LLaVA.
MAX_PIXELS="${MAX_PIXELS:-1003520}"

read -r -a BETAS <<< "${BETAS:-0.5 0.2 0.9}"
read -r -a NUM_DEMOS <<< "${NUM_DEMOS:-50 100 200 500}"
read -r -a ALL_DIMS <<< "${DIMENSIONS:-all existence attribute counting relation}"
read -r -a IVS <<< "${IVS:-vti_textual_additive_layer vti_textual_additive_mlp vti_textual_uniform_rotation_layer vti_textual_uniform_rotation_mlp}"

CONDA_ENV="${CONDA_ENV:-vlm_hallucination_mitigation}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

PIX_ARGS=()
if [[ "$MODEL" == *[Qq]wen2* || "$MODEL" == *[Qq]wen/Qwen2* ]]; then
  PIX_ARGS=(--max_pixels "$MAX_PIXELS")
fi

MODEL_SHORT="$(python - <<PY
from src.model import _normalize_model_name
print(_normalize_model_name("$MODEL"))
PY
)"

echo "=== demos_v2 qual grid ==="
echo "  model     : $MODEL ($MODEL_SHORT)"
echo "  run_date  : $RUN_DATE"
echo "  demos     : $DEMOS_PATH"
echo "  subset    : $SUBSET_IDS"
echo "  betas     : ${BETAS[*]}"
echo "  num_demos : ${NUM_DEMOS[*]}"
echo "  dims      : ${ALL_DIMS[*]}"
echo "  chair_cap : $CHAIR_CAP"
echo "  gpu       : $CUDA_VISIBLE_DEVICES"
echo "  max_pixels: ${PIX_ARGS[*]:-n/a}"

run_extract() {
  local dims=("$@")
  [[ "${SKIP_EXTRACT:-0}" == "1" ]] && return 0
  python evaluation/run_scripts/extract_demosv2_directions.py \
    --model "$MODEL" \
    --demos_path "$DEMOS_PATH" \
    --dimensions "${dims[@]}" \
    --num_demos "${NUM_DEMOS[@]}" \
    --rank "$RANK" \
    --seed "$SEED" \
    "${PIX_ARGS[@]}"
}

run_smoke() {
  [[ "${SKIP_SMOKE:-0}" == "1" ]] && return 0
  # Isolated run_date so the smoke cell cannot poison the real grid dirs.
  # subset_ids takes precedence over --limit, so use a 1-id pin file.
  local smoke_date="${RUN_DATE}_smoke"
  local smoke_subset="${SMOKE_SUBSET_IDS:-data/vti/qual_subset_chair1_smoke.json}"
  echo "=== smoke gate (additive_mlp β=0.5 dim=all nd=500, 1 CHAIR) ==="
  python evaluation/run_eval.py \
    --model "$MODEL" \
    --benchmarks chair \
    --interventions no_intervention vti_textual_additive_mlp \
    --subset_ids_file "$smoke_subset" \
    --chair_prompt "$CHAIR_PROMPT" \
    --chair_max_new_tokens "$CHAIR_CAP" \
    --demos_path "$DEMOS_PATH" \
    --vector_dimension all \
    --num_demos 500 \
    --rank "$RANK" \
    --beta 0.5 \
    --output_dir "$OUTPUT_DIR" \
    --run_date "$smoke_date" \
    "${PIX_ARGS[@]}"
  local smoke_dir="$OUTPUT_DIR/$smoke_date/$MODEL_SHORT/chair/vti_textual_additive_mlp__b0.5__dall__nd500"
  local base_dir="$OUTPUT_DIR/$smoke_date/$MODEL_SHORT/chair/no_intervention"
  python - <<PY
import json, sys
from pathlib import Path
smoke = Path("$smoke_dir") / "responses.json"
base = Path("$base_dir") / "responses.json"
meta_json = Path("$smoke_dir") / "metric_summary.json"
if not smoke.exists() or not meta_json.exists():
    print("[smoke] FAIL: missing smoke outputs", smoke, meta_json)
    sys.exit(1)
summary = json.loads(meta_json.read_text())
cfg = summary.get("intervention_config") or {}
h = cfg.get("demos_content_hash_sha256_16")
if not h:
    print("[smoke] FAIL: metadata missing demos hash", cfg)
    sys.exit(1)
if "__dall__" not in smoke.parent.name or "__nd500" not in smoke.parent.name:
    print("[smoke] FAIL: result dir missing __d/__nd suffix:", smoke.parent.name)
    sys.exit(1)
resp = json.loads(smoke.read_text())
text = (resp[0].get("response") or "") if resp else ""
base_text = ""
if base.exists():
    br = json.loads(base.read_text())
    base_text = (br[0].get("response") or "") if br else ""
print(f"[smoke] ok hash={h} response_len={len(text)} dir={smoke.parent.name}")
print(f"[smoke] response preview: {text[:200]!r}")
if base_text:
    same = text == base_text
    print(f"[smoke] differs_from_baseline={not same} baseline_len={len(base_text)}")
    if same:
        print("[smoke] FAIL: additive_mlp response identical to baseline — check direction norms / hooks")
        sys.exit(1)
if not text.strip():
    print("[smoke] WARN: empty smoke response")
PY
}

run_baselines() {
  [[ "${SKIP_EVAL:-0}" == "1" ]] && return 0
  echo "=== baselines ==="
  python evaluation/run_eval.py \
    --model "$MODEL" \
    --benchmarks chair amber \
    --amber_task discriminative \
    --interventions no_intervention \
    --subset_ids_file "$SUBSET_IDS" \
    --chair_prompt "$CHAIR_PROMPT" \
    --chair_max_new_tokens "$CHAIR_CAP" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir "$OUTPUT_DIR" \
    --run_date "$RUN_DATE" \
    --skip_if_exists \
    "${PIX_ARGS[@]}"
}

run_dim_slice() {
  local dim="$1"
  [[ "${SKIP_EVAL:-0}" == "1" ]] && return 0
  for nd in "${NUM_DEMOS[@]}"; do
    for beta in "${BETAS[@]}"; do
      echo "=== eval dim=$dim nd=$nd beta=$beta ==="
      python evaluation/run_eval.py \
        --model "$MODEL" \
        --benchmarks chair amber \
        --amber_task discriminative \
        --interventions "${IVS[@]}" \
        --subset_ids_file "$SUBSET_IDS" \
        --chair_prompt "$CHAIR_PROMPT" \
        --chair_max_new_tokens "$CHAIR_CAP" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --demos_path "$DEMOS_PATH" \
        --vector_dimension "$dim" \
        --num_demos "$nd" \
        --rank "$RANK" \
        --beta "$beta" \
        --output_dir "$OUTPUT_DIR" \
        --run_date "$RUN_DATE" \
        --skip_if_exists \
        "${PIX_ARGS[@]}" \
        || echo "[warn] eval failed dim=$dim nd=$nd beta=$beta (continuing)"
    done
  done
}

render_gallery() {
  local dim="$1"
  [[ "${SKIP_GALLERY:-0}" == "1" ]] && return 0
  # Legacy one-HTML-per-dimension overview (still useful).
  python helper_scripts/render_demosv2_qual_review.py \
    --run_date "$RUN_DATE" \
    --model "$MODEL" \
    --dimension "$dim" \
    --output_dir "$OUTPUT_DIR" \
    || echo "[warn] legacy gallery render failed for dim=$dim"
  # Split galleries (preferred navigation): iv / dim / nd.
  python helper_scripts/render_demosv2_qual_split_review.py \
    --run_date "$RUN_DATE" \
    --model "$MODEL" \
    --dimensions "$dim" \
    --output_dir "$OUTPUT_DIR" \
    || echo "[warn] split gallery render failed for dim=$dim"
}

# ── Pipeline ─────────────────────────────────────────────────────────────────
run_extract all || { echo "[fatal] all-direction extraction failed"; exit 1; }
run_baselines || { echo "[warn] baselines failed (continuing to smoke)"; }
run_smoke || { echo "[fatal] smoke gate failed — aborting grid"; exit 1; }
run_dim_slice all
render_gallery all

# Remaining dimensions: extract then eval (all slice already delivering galleries).
OTHER_DIMS=()
for d in "${ALL_DIMS[@]}"; do
  [[ "$d" == "all" ]] && continue
  OTHER_DIMS+=("$d")
done
if ((${#OTHER_DIMS[@]})); then
  run_extract "${OTHER_DIMS[@]}" || { echo "[fatal] remaining-dim extraction failed"; exit 1; }
  for d in "${OTHER_DIMS[@]}"; do
    run_dim_slice "$d"
    render_gallery "$d"
  done
fi

echo "=== demos_v2 qual grid finished for $MODEL_SHORT ==="
