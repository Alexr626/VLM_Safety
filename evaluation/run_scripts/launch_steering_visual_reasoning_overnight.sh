#!/usr/bin/env bash
# Overnight orchestrator for steering visual reasoning validation.
#
# Sequence:
#   1) Snapshot pre-existing _r2_partition hashes (additivity gate)
#   2) CPU mean-difference extraction (0 forwards)
#   3) Verify extraction
#   4) CHAIR caption-length probe @ 256 and 512, both models, sequential on GPU 0
#   5) If any caption hits the 256 cap on either model → CHAIR_CAP=512
#      (never halt; never skip CHAIR; Alex 2026-07-30 overnight rule)
#   6) Launch both model grids (AMBER → CHAIR → POPE), staggered 5 minutes
#
# Usage:
#   nohup bash evaluation/run_scripts/launch_steering_visual_reasoning_overnight.sh \
#     > logs/steering_visual_reasoning_overnight_orchestrator_2026-07-30.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

RUN_DATE="${RUN_DATE:-2026-07-30}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
CHAIR_CAP_DEFAULT="${CHAIR_CAP_DEFAULT:-256}"
CHAIR_CAP_FALLBACK="${CHAIR_CAP_FALLBACK:-512}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
STAGGER_SEC="${STAGGER_SEC:-300}"
STATUS_DIR="$OUTPUT_DIR/$RUN_DATE/_analysis_steering_visual_reasoning_validation"
mkdir -p "$STATUS_DIR" logs

CONDA_ENV="${CONDA_ENV:-vlm_hallucination_mitigation}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

LLAVA="llava-hf/llava-1.5-7b-hf"
QWEN="Qwen/Qwen2.5-VL-7B-Instruct"
PRE_HASHES="$STATUS_DIR/pre_r2_partition_hashes_before_meandiff.json"
PROBE_DECISION="$STATUS_DIR/chair_cap_probe_decision.json"
ORCH_STATUS="$STATUS_DIR/overnight_orchestrator_status.json"

write_status() {
  local stage="$1"
  local detail="${2:-}"
  STAGE="$stage" DETAIL="$detail" ORCH_STATUS="$ORCH_STATUS" \
  CHAIR_CAP_DEFAULT="$CHAIR_CAP_DEFAULT" CHAIR_CAP_FALLBACK="$CHAIR_CAP_FALLBACK" \
  python - <<'PY'
import json, os
from datetime import datetime
from pathlib import Path
path = Path(os.environ["ORCH_STATUS"])
payload = {
    "stage": os.environ["STAGE"],
    "detail": os.environ.get("DETAIL", ""),
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "chair_cap_default": int(os.environ["CHAIR_CAP_DEFAULT"]),
    "chair_cap_fallback": int(os.environ["CHAIR_CAP_FALLBACK"]),
}
if path.exists():
    try:
        old = json.loads(path.read_text())
        hist = old.get("history", [])
        hist.append({"stage": old.get("stage"), "at": old.get("updated_at")})
        payload["history"] = hist
        for k in ("chair_cap", "llava_pid", "qwen_pid", "llava_log", "qwen_log"):
            if k in old:
                payload[k] = old[k]
    except Exception:
        payload["history"] = []
else:
    payload["history"] = []
path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"[status] {payload['stage']}")
PY
}

write_status "started" "overnight orchestrator launched"

# ── 1. Snapshot _r2_partition hashes before extraction ───────────────────────
write_status "snapshot_r2_hashes"
PRE_HASHES="$PRE_HASHES" python - <<'PY'
import hashlib, json
from pathlib import Path
from src.model import _normalize_model_name
from src.paths import vti_demos_850_path
from evaluation.interventions.vti.directions_partition import (
    PARTITION_SIZES, partition_cache_dir, partition_slug,
)
from evaluation.interventions.vti.directions_v2 import demos_content_hash
import os

demos_hash = demos_content_hash(vti_demos_850_path())
out = {}
for model in ("llava-hf/llava-1.5-7b-hf", "Qwen/Qwen2.5-VL-7B-Instruct"):
    ms = _normalize_model_name(model)
    for n in PARTITION_SIZES:
        slug = partition_slug(demos_hash, "all", n, seed=42, rank=2)
        p = partition_cache_dir(ms, slug) / "directions.npz"
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
path = Path(os.environ["PRE_HASHES"])
path.write_text(json.dumps(out, indent=2) + "\n")
print(f"wrote {path} ({len(out)} hashes)")
PY

# ── 2. Mean-difference extraction (CPU) ──────────────────────────────────────
write_status "meandiff_extraction"
python evaluation/run_scripts/extract_demos850_meandiff_directions.py \
  --models "$LLAVA" "$QWEN" \
  --dimension all --sizes 50 100 200 500 \
  || { write_status "failed_extraction"; exit 1; }

# ── 3. Verify extraction ─────────────────────────────────────────────────────
write_status "meandiff_verify"
python helper_scripts/verify_demos850_meandiff_extraction.py \
  --models "$LLAVA" "$QWEN" \
  --pre_r2_hashes "$PRE_HASHES" \
  || { write_status "failed_verify"; exit 1; }

# ── 4. CHAIR caption-length probe (sequential, exclusive GPU 0) ──────────────
write_status "chair_caption_length_probe"
python evaluation/chair_amber_diagnostics/step0_chair_token_cap.py \
  --model "$LLAVA" --num_images 20 --seed 1234 --caps 256 512 \
  --run_date "$RUN_DATE" \
  || { write_status "failed_probe_llava"; exit 1; }

python evaluation/chair_amber_diagnostics/step0_chair_token_cap.py \
  --model "$QWEN" --num_images 20 --seed 1234 --caps 256 512 \
  --max_pixels "$MAX_PIXELS" --run_date "$RUN_DATE" \
  || { write_status "failed_probe_qwen"; exit 1; }

# ── 5. Decide CHAIR cap: auto-bump to 512 if any 256 truncation ──────────────
CHAIR_CAP="$(
PROBE_DECISION="$PROBE_DECISION" RUN_DATE="$RUN_DATE" \
CHAIR_CAP_DEFAULT="$CHAIR_CAP_DEFAULT" CHAIR_CAP_FALLBACK="$CHAIR_CAP_FALLBACK" \
python - <<'PY'
import json, os
from pathlib import Path
from src.model import _normalize_model_name

run_date = os.environ["RUN_DATE"]
default_cap = int(os.environ["CHAIR_CAP_DEFAULT"])
fallback = int(os.environ["CHAIR_CAP_FALLBACK"])
models = [
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]
hits = {}
any_hit = False
for mid in models:
    ms = _normalize_model_name(mid)
    path = Path(f"evaluation/results/{run_date}/_diagnostics/step0_chair_token_cap_{ms}.json")
    data = json.loads(path.read_text())
    n = int((data.get("metrics_by_cap") or {}).get("256", {}).get("n_captions_at_token_cap") or 0)
    hits[ms] = n
    if n > 0:
        any_hit = True
chosen = fallback if any_hit else default_cap
decision = {
    "n_captions_at_token_cap_256_by_model": hits,
    "any_hit_at_256": any_hit,
    "chair_max_new_tokens_chosen": chosen,
    "policy": (
        "if any caption hits 256 on either model, use 512; "
        "never skip CHAIR; never halt"
    ),
}
Path(os.environ["PROBE_DECISION"]).write_text(json.dumps(decision, indent=2) + "\n")
print(chosen)
PY
)"

write_status "chair_cap_decided" "CHAIR_CAP=${CHAIR_CAP}"
echo "[orchestrator] CHAIR_CAP=${CHAIR_CAP}"

# ── 6. Launch both grid processes ────────────────────────────────────────────
write_status "launching_grids" "CHAIR_CAP=${CHAIR_CAP}"

LLAVA_LOG="logs/steering_visual_reasoning_validation_llava_${RUN_DATE}.log"
QWEN_LOG="logs/steering_visual_reasoning_validation_qwen25_${RUN_DATE}.log"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CHAIR_CAP="$CHAIR_CAP" RUN_DATE="$RUN_DATE" \
  nohup bash evaluation/run_scripts/run_steering_visual_reasoning_validation.sh "$LLAVA" \
  > "$LLAVA_LOG" 2>&1 &
LLAVA_PID=$!
echo "[orchestrator] LLaVA pid=$LLAVA_PID log=$LLAVA_LOG"

echo "[orchestrator] staggering ${STAGGER_SEC}s before Qwen launch ..."
sleep "$STAGGER_SEC"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MAX_PIXELS="$MAX_PIXELS" CHAIR_CAP="$CHAIR_CAP" RUN_DATE="$RUN_DATE" \
  nohup bash evaluation/run_scripts/run_steering_visual_reasoning_validation.sh "$QWEN" \
  > "$QWEN_LOG" 2>&1 &
QWEN_PID=$!
echo "[orchestrator] Qwen pid=$QWEN_PID log=$QWEN_LOG"

ORCH_STATUS="$ORCH_STATUS" CHAIR_CAP="$CHAIR_CAP" \
LLAVA_PID="$LLAVA_PID" QWEN_PID="$QWEN_PID" \
LLAVA_LOG="$LLAVA_LOG" QWEN_LOG="$QWEN_LOG" \
python - <<'PY'
import json, os
from datetime import datetime
from pathlib import Path
path = Path(os.environ["ORCH_STATUS"])
payload = json.loads(path.read_text()) if path.exists() else {}
payload.update({
    "stage": "grids_running",
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "chair_cap": int(os.environ["CHAIR_CAP"]),
    "llava_pid": int(os.environ["LLAVA_PID"]),
    "qwen_pid": int(os.environ["QWEN_PID"]),
    "llava_log": os.environ["LLAVA_LOG"],
    "qwen_log": os.environ["QWEN_LOG"],
})
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "[orchestrator] both grid processes launched; starting crash babysitter"
BABYSIT_LOG="logs/steering_visual_reasoning_babysitter_${RUN_DATE}.log"
nohup python evaluation/run_scripts/babysit_steering_visual_reasoning_grids.py \
  --run_date "$RUN_DATE" --poll_sec 60 \
  > "$BABYSIT_LOG" 2>&1 &
BABYSIT_PID=$!
echo "[orchestrator] babysitter pid=$BABYSIT_PID log=$BABYSIT_LOG"

write_status "grids_launched" "llava_pid=${LLAVA_PID} qwen_pid=${QWEN_PID} chair_cap=${CHAIR_CAP} babysitter_pid=${BABYSIT_PID}"
