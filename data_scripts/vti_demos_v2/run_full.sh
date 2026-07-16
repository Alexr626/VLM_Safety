#!/usr/bin/env bash
# Fresh v2.1 downstream run. Keeps stage0/cooccurrence; N_CANDIDATES is its pool size.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate vlm_hallucination_mitigation
N_CANDIDATES="${N_CANDIDATES:-300}"
V2=data/vti/v2
rm -f "$V2"/stage{1,2,3,4}_*.jsonl "$V2"/stage{1,2,3,4}_summary.json \
      "$V2"/stage1b_allocation.jsonl "$V2"/stage1b_summary.json "$V2"/stage5_summary.json \
      "$V2"/calls/stage{1,2,3,4}.jsonl data/vti/demos_v2.jsonl data/vti/demos_v2_*.jsonl
python data_scripts/vti_demos_v2/stage1_verify_anchors.py
python data_scripts/vti_demos_v2/stage1b_allocate.py
python data_scripts/vti_demos_v2/stage2_write_truthful.py
python data_scripts/vti_demos_v2/stage3_make_variants.py
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final "$N_CANDIDATES"
