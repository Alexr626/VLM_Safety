#!/usr/bin/env bash
# Download all hallucination benchmarks and build combined.json manifests.
#
# Usage (from anywhere):
#   bash data_scripts/download_all_benchmarks.sh
#
# Requires: conda env active (vlm_hallucination_mitigation), network access.
# Order matters: CHAIR first (fetches COCO val2014 used by POPE and CHAIR).
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/data_scripts"

echo "Project root: $ROOT"
echo "Python:       $(command -v "$PYTHON")"
echo ""

run_download() {
    local name="$1"
    local script="$2"
    echo "=== [$name] ==="
    "$PYTHON" "$SCRIPTS/$script"
    echo ""
}

t0=$SECONDS

# COCO val2014 + CHAIR manifest (must run before POPE)
run_download "CHAIR + COCO val2014" "download_chair.py"
run_download "POPE" "download_pope.py"
run_download "AMBER" "download_amber.py"
run_download "HallusionBench" "download_hallusionbench.py"
run_download "MMHal-Bench" "download_mmhal_bench.py"

elapsed=$((SECONDS - t0))
echo "All benchmark downloads finished in ${elapsed}s."
echo "Manifests:"
for bm in pope amber chair hallusionbench mmhal-bench; do
    path="$ROOT/data/$bm/combined.json"
    if [[ -f "$path" ]]; then
        echo "  ✓ $path"
    else
        echo "  ✗ $path (missing)"
    fi
done
