#!/usr/bin/env bash
# Full demos_v2 pipeline over all stage-0 candidates (keep stage0 + cooccurrence).
# Intended to be launched under nohup; maximizes finals via --n-final 300.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

LOG_DIR="$ROOT/data/vti/v2/logs"
mkdir -p "$LOG_DIR" "$ROOT/data/vti/v2/calls"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/full300_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== demos_v2 full-300 start $(date -Is) ==="
echo "log=$LOG"
wc -l data/vti/v2/stage0_candidates.jsonl

# Keep stage0 + cooccurrence; wipe downstream so resume does not skip old rejects.
rm -f data/vti/v2/stage1_*.jsonl \
      data/vti/v2/stage1_summary.json \
      data/vti/v2/stage2_*.jsonl \
      data/vti/v2/stage2_summary.json \
      data/vti/v2/stage3_*.jsonl \
      data/vti/v2/stage3_summary.json \
      data/vti/v2/stage4_*.jsonl \
      data/vti/v2/stage4_summary.json \
      data/vti/v2/stage5_summary.json \
      data/vti/v2/calls/stage{1,2,3,4}.jsonl \
      data/vti/demos_v2.jsonl \
      data/vti/demos_v2_*.jsonl

# No --limit: walk the full stage-0 pool. Stage 5 keeps every stage-4 pass (cap 300).
python data_scripts/vti_demos_v2/stage1_verify_anchors.py
python data_scripts/vti_demos_v2/stage2_write_truthful.py
python data_scripts/vti_demos_v2/stage3_make_variants.py
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 300

echo "=== stage summaries ==="
for s in 1 2 3 4 5; do
  f="data/vti/v2/stage${s}_summary.json"
  if [[ -f "$f" ]]; then
    echo "--- $f ---"
    python -c "import json; print(json.dumps(json.load(open('$f')), indent=2)[:2000])"
  fi
done

echo "=== demos_v2 full-300 done $(date -Is) ==="
wc -l data/vti/demos_v2.jsonl || true
