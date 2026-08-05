#!/usr/bin/env bash
# Verify NFS layout for steering smoke / grid jobs.
# MODEL_SHORTS: space-separated model short names to require meandiff dirs for.
# Default: both LLaVA and Qwen.
set -eu

BASE="${BASE:-/home/datalake/romanus}"
REPO="${REPO:-$BASE/vlm_hallucination}"
RUN_DATE="${RUN_DATE:-2026-07-30}"
MODEL_SHORTS="${MODEL_SHORTS:-llava-1.5-7b-hf qwen2.5-vl-7b-instruct}"

echo "=== verify_steering_nfs_layout ==="
echo "  BASE=$BASE"
echo "  REPO=$REPO"
echo "  MODEL_SHORTS=$MODEL_SHORTS"
echo "  time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

fail=0
ok()   { echo "OK  $*"; }
bad()  { echo "FAIL $*"; fail=1; }
info() { echo "INFO $*"; }

[[ -d "$REPO" ]] && ok "repo dir" || bad "repo dir missing: $REPO"

for f in \
  evaluation/run_scripts/run_steering_visual_reasoning_validation.sh \
  evaluation/interventions/vti/directions_meandiff.py \
  helper_scripts/runai/run_steering_visual_reasoning_llava.sh \
  helper_scripts/runai/run_steering_visual_reasoning_qwen.sh \
  helper_scripts/runai/run_steering_triple_one_h100.sh \
  helper_scripts/runai/run_steering_llava_smoke.sh \
  helper_scripts/runai/run_steering_qwen_smoke.sh \
  helper_scripts/runai/run_bash_lf.py \
  helper_scripts/runai/runai_job_logging.sh
do
  if [[ -f "$REPO/$f" ]]; then ok "$f"; else bad "missing $f"; fi
done

if grep -q 'SKIP_CONDA_ACTIVATE' \
  "$REPO/evaluation/run_scripts/run_steering_visual_reasoning_validation.sh" 2>/dev/null; then
  ok "SKIP_CONDA_ACTIVATE in validation.sh"
else
  bad "SKIP_CONDA_ACTIVATE missing from validation.sh"
fi

shopt -s nullglob
for MODEL_SHORT in $MODEL_SHORTS; do
  echo "--- model $MODEL_SHORT ---"
  DIR_GLOB="$REPO/experiment_artifacts/vti/${MODEL_SHORT}/textual_v2/demos850_*_meandiff_partition"
  dirs=($DIR_GLOB)
  n=${#dirs[@]}
  info "meandiff_dirs=$n (expect >=4)"
  if [[ "$n" -ge 4 ]]; then ok "$MODEL_SHORT meandiff dir count"; else bad "$MODEL_SHORT meandiff dir count ($n < 4)"; fi
  for d in "${dirs[@]}"; do
    if [[ -f "$d/directions.npz" ]]; then
      ok "$(basename "$d")/directions.npz"
    else
      bad "missing directions.npz in $d"
    fi
  done
  chair_root="$REPO/evaluation/results/${RUN_DATE}/${MODEL_SHORT}/chair"
  if [[ -d "$chair_root" ]]; then
    c=$(find "$chair_root" -name metric_summary.json 2>/dev/null | wc -l | tr -d ' ')
    info "$MODEL_SHORT chair_summaries=$c"
    ok "$MODEL_SHORT chair results tree present"
  else
    info "$MODEL_SHORT chair results tree absent (ok if starting fresh)"
  fi
done

if [[ -x "$BASE/bin/micromamba" ]]; then ok "micromamba"; else bad "micromamba missing"; fi
if [[ -d "$BASE/envs/vlm_hal" ]]; then ok "envs/vlm_hal"; else bad "envs/vlm_hal missing"; fi

# Pins / manifests required by the steering validation driver (often gitignored).
for pin in \
  data/chair/pinned_chair_500.json \
  data/chair/combined.json \
  data/amber/pinned_amber_disc_450.json \
  data/pope/combined.json \
  data/pope/output/coco/coco_pope_random.json \
  data/pope/output/coco/coco_pope_popular.json \
  data/pope/output/coco/coco_pope_adversarial.json \
  data/vti/demos_850.jsonl \
  data/vti/demos_850_partition_s42.json
do
  if [[ -f "$REPO/$pin" ]]; then ok "$pin"; else bad "missing $pin"; fi
done

# Heavy assets not in the code tarball (must already exist on NFS from setup).
coco_val="$REPO/data/coco/val2014"
coco_ann="$REPO/data/coco/annotations/instances_val2014.json"
if [[ -d "$coco_val" ]]; then
  n=$(find "$coco_val" -maxdepth 1 -name '*.jpg' 2>/dev/null | head -5 | wc -l | tr -d ' ')
  info "coco val2014 jpg sample_count_first_find=$n"
  [[ "$n" -ge 1 ]] && ok "coco val2014 present" || bad "coco val2014 empty"
else
  bad "missing $coco_val (setup_vlm / download_chair on NFS)"
fi
if [[ -f "$coco_ann" ]]; then ok "instances_val2014.json"; else bad "missing $coco_ann (CHAIR metrics)"; fi

if [[ "$fail" -ne 0 ]]; then
  echo "VERIFY_SYNC_FAIL"
  exit 1
fi
echo "VERIFY_SYNC_OK"
exit 0
