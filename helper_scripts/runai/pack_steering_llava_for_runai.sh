#!/usr/bin/env bash
# Build RunAI deploy packages on lambdab2 for LLaVA + Qwen steering resume.
#
# Produces under /tmp/runai_steering_2026-07-31/:
#   vti_repo_working_tree.tar.gz
#   llava_meandiff_directions.tar.gz
#   qwen_meandiff_directions.tar.gz
#   llava_partial_results_chair.tar.gz (if any)
#   SUBMIT.md / SUBMIT_STEERING_LLAVA.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${OUT:-/tmp/runai_steering_2026-07-31}"
rm -rf "$OUT"
mkdir -p "$OUT"

TAG="2026-07-30"

echo "=== packing working-tree code snapshot ==="
git archive --format=tar HEAD -o "$OUT/base.tar"
OVERLAY=(
  evaluation/interventions/vti/directions_meandiff.py
  evaluation/interventions/vti/intervention.py
  evaluation/interventions/__init__.py
  evaluation/runners/eval_runner.py
  evaluation/run_eval.py
  evaluation/chair_amber_diagnostics/step0_chair_token_cap.py
  evaluation/chair_amber_diagnostics/run_scripts/run_exp1_repro_grid.sh
  evaluation/run_scripts/run_steering_visual_reasoning_validation.sh
  evaluation/run_scripts/extract_demos850_meandiff_directions.py
  evaluation/run_scripts/launch_steering_visual_reasoning_overnight.sh
  evaluation/steering_visual_reasoning_validation
  helper_scripts/verify_demos850_meandiff_extraction.py
  data/chair/pinned_chair_500.json
  data/chair/combined.json
  helper_scripts/runai/run_steering_visual_reasoning_llava.sh
  helper_scripts/runai/run_steering_visual_reasoning_qwen.sh
  helper_scripts/runai/run_steering_triple_one_h100.sh
  helper_scripts/runai/run_steering_llava_smoke.sh
  helper_scripts/runai/run_steering_qwen_smoke.sh
  helper_scripts/runai/sync_steering_llava.sh
  helper_scripts/runai/verify_steering_nfs_layout.sh
  helper_scripts/runai/runai_job_logging.sh
  helper_scripts/runai/run_bash_lf.py
  helper_scripts/runai/SUBMIT_STEERING_LLAVA.md
  helper_scripts/runai/update_repo_from_tarball.sh
  readme.md
  IMPLEMENTATION.md
)
OVERLAY_EXIST=()
for p in "${OVERLAY[@]}"; do
  [[ -e "$p" ]] && OVERLAY_EXIST+=("$p")
done
tar -rf "$OUT/base.tar" "${OVERLAY_EXIST[@]}"
gzip -c "$OUT/base.tar" > "$OUT/vti_repo_working_tree.tar.gz"
rm -f "$OUT/base.tar"

pack_meandiff() {
  local short="$1" outname="$2"
  local dir_root="experiment_artifacts/vti/${short}/textual_v2"
  python - <<PY
from pathlib import Path
import tarfile
root = Path("$ROOT")
out = Path("$OUT") / "$outname"
base = root / "$dir_root"
short = "$short"
paths = sorted(base.glob("demos850_*_meandiff_partition"))
manifest = root / "experiment_artifacts/vti/demos850_meandiff_extraction_manifest_${TAG}.json"
with tarfile.open(out, "w:gz") as tf:
    for p in paths:
        tf.add(p, arcname=str(p.relative_to(root)))
    if manifest.is_file():
        tf.add(manifest, arcname=str(manifest.relative_to(root)))
print(f"{short}: {len(paths)} dirs -> {out} ({out.stat().st_size/1e6:.1f} MB)")
if len(paths) < 4:
    raise SystemExit(f"expected >=4 meandiff dirs for {short}, got {len(paths)}")
PY
}

echo "=== packing meandiff directions ==="
pack_meandiff "llava-1.5-7b-hf" "llava_meandiff_directions.tar.gz"
pack_meandiff "qwen2.5-vl-7b-instruct" "qwen_meandiff_directions.tar.gz"

echo "=== packing any completed LLaVA CHAIR cells ==="
CHAIR_ROOT="evaluation/results/${TAG}/llava-1.5-7b-hf/chair"
if [[ -d "$CHAIR_ROOT" ]]; then
  tar -czf "$OUT/llava_partial_results_chair.tar.gz" "$CHAIR_ROOT"
else
  echo "(no chair results yet)" > "$OUT/llava_partial_results_chair.EMPTY.txt"
fi

cp "$ROOT/helper_scripts/runai/SUBMIT_STEERING_LLAVA.md" "$OUT/SUBMIT_STEERING_LLAVA.md"
cp "$OUT/SUBMIT_STEERING_LLAVA.md" "$OUT/SUBMIT.md"

ls -lh "$OUT"
echo "Wrote packages under $OUT"
