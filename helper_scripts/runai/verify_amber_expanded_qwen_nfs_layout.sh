#!/usr/bin/env bash
# Verify NFS layout for the AMBER-1500 Qwen expanded grid (RunAI).
set -eu

BASE="${BASE:-/home/datalake/romanus}"
REPO="${REPO:-$BASE/vlm_hallucination}"
RUN_DATE="${RUN_DATE:-2026-08-05}"

echo "=== verify_amber_expanded_qwen_nfs_layout ==="
echo "  BASE=$BASE REPO=$REPO RUN_DATE=$RUN_DATE"
echo "  time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

fail=0
ok()   { echo "OK  $*"; }
bad()  { echo "FAIL $*"; fail=1; }

[[ -d "$REPO" ]] && ok "repo dir" || bad "repo dir missing: $REPO"

for f in \
  evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh \
  evaluation/runners/eval_runner.py \
  helper_scripts/runai/run_amber_expanded_qwen_worker.sh \
  helper_scripts/runai/run_amber_expanded_qwen_beta_triple_one_h100.sh \
  helper_scripts/runai/run_amber_expanded_qwen_smoke.sh \
  helper_scripts/runai/run_bash_lf.py \
  helper_scripts/runai/runai_job_logging.sh \
  data/amber/pinned_amber_disc_1500.json \
  data/amber/pinned_amber_disc_1500_smoke5.json
do
  if [[ -f "$REPO/$f" ]]; then ok "$f"; else bad "missing $f"; fi
done

# recon-stem naming must be present (not hard-coded meandiff-only).
if grep -q 'pc1_plus_mean' "$REPO/evaluation/runners/eval_runner.py"; then
  ok "eval_runner recon stem mapping"
else
  bad "eval_runner missing pc1_plus_mean mapping"
fi

IMG_DIR="$REPO/data/amber/images"
if [[ -d "$IMG_DIR" ]]; then
  n_img=$(find "$IMG_DIR" -type f | wc -l | tr -d ' ')
  info_n="$n_img"
  if [[ "$n_img" -ge 1000 ]]; then ok "amber images n=$info_n"; else bad "amber images too few n=$info_n"; fi
else
  bad "missing data/amber/images"
fi

SHORT=qwen2.5-vl-7b-instruct
for stem in meandiff r2; do
  d="$REPO/experiment_artifacts/vti/${SHORT}/textual_v2/demos850_ba05bd96_all_nd500_s42_${stem}_partition"
  if [[ -f "$d/directions.npz" && -f "$d/metadata.json" ]]; then
    ok "$(basename "$d")/directions.npz"
  else
    bad "missing directions for $stem at $d"
  fi
done

if [[ -x "$BASE/bin/micromamba" ]]; then ok "micromamba"; else bad "micromamba missing"; fi
if [[ -d "$BASE/envs/vlm_hal" ]]; then ok "envs/vlm_hal"; else bad "envs/vlm_hal missing"; fi

partial="$REPO/evaluation/results/${RUN_DATE}/${SHORT}/amber"
if [[ -d "$partial" ]]; then
  c=$(find "$partial" -name metric_summary.json 2>/dev/null | wc -l | tr -d ' ')
  echo "INFO qwen amber complete cells=$c (partial resume ok)"
  ok "partial results tree present"
else
  echo "INFO no partial results yet (fresh run)"
fi

if [[ "$fail" -ne 0 ]]; then
  echo "VERIFY_AMBER_QWEN_FAIL"
  exit 1
fi
echo "VERIFY_AMBER_QWEN_OK"
exit 0
