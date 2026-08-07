#!/usr/bin/env bash
# Pack AMBER-1500 Qwen expanded grid for RunAI (lambdab2 → WinSCP → NFS).
#
# Writes under /tmp/runai_amber_qwen_2026-08-06/:
#   vti_repo_amber_qwen.tar.gz
#   qwen_amber_directions.tar.gz          # meandiff + r2 nd500
#   qwen_amber_partial_results.tar.gz     # completed + checkpoint cells
#   amber_images.tar.gz
#   SUBMIT_AMBER_QWEN_RUNAI.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${OUT:-/tmp/runai_amber_qwen_2026-08-06}"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "=== packing working-tree code snapshot ==="
git archive --format=tar HEAD -o "$OUT/base.tar"
OVERLAY=(
  evaluation/runners/eval_runner.py
  evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh
  evaluation/steering_vector_validation_continuation
  data/amber/pinned_amber_disc_1500.json
  data/amber/pinned_amber_disc_1500_smoke5.json
  helper_scripts/runai/run_amber_expanded_qwen_worker.sh
  helper_scripts/runai/run_amber_expanded_qwen_beta_triple_one_h100.sh
  helper_scripts/runai/run_amber_expanded_qwen_smoke.sh
  helper_scripts/runai/verify_amber_expanded_qwen_nfs_layout.sh
  helper_scripts/runai/sync_amber_expanded_qwen.sh
  helper_scripts/runai/run_bash_lf.py
  helper_scripts/runai/runai_job_logging.sh
  helper_scripts/runai/remap_lambdab2_paths.py
  helper_scripts/runai/SUBMIT_AMBER_QWEN_RUNAI.md
)
OVERLAY_EXIST=()
for p in "${OVERLAY[@]}"; do
  [[ -e "$p" ]] && OVERLAY_EXIST+=("$p")
done
tar -rf "$OUT/base.tar" "${OVERLAY_EXIST[@]}"
gzip -c "$OUT/base.tar" > "$OUT/vti_repo_amber_qwen.tar.gz"
rm -f "$OUT/base.tar"
ls -lh "$OUT/vti_repo_amber_qwen.tar.gz"

echo "=== packing Qwen nd500 meandiff + r2 directions ==="
tar -czf "$OUT/qwen_amber_directions.tar.gz" \
  experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd500_s42_meandiff_partition \
  experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd500_s42_r2_partition
ls -lh "$OUT/qwen_amber_directions.tar.gz"

echo "=== packing AMBER images ==="
tar -czf "$OUT/amber_images.tar.gz" data/amber/images
ls -lh "$OUT/amber_images.tar.gz"

echo "=== packing Qwen partial results (2026-08-05) ==="
PARTIAL="evaluation/results/2026-08-05/qwen2.5-vl-7b-instruct"
if [[ -d "$PARTIAL" ]]; then
  tar -czf "$OUT/qwen_amber_partial_results.tar.gz" "$PARTIAL"
else
  echo "(no partial results)" > "$OUT/qwen_amber_partial_results.EMPTY.txt"
fi
ls -lh "$OUT"/qwen_amber_partial_results* 2>/dev/null || true

cp "$ROOT/helper_scripts/runai/SUBMIT_AMBER_QWEN_RUNAI.md" "$OUT/SUBMIT_AMBER_QWEN_RUNAI.md"
cp "$OUT/SUBMIT_AMBER_QWEN_RUNAI.md" "$OUT/SUBMIT.md"

ls -lh "$OUT"
echo "Wrote packages under $OUT"
