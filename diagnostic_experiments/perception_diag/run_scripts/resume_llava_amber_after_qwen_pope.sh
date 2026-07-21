#!/usr/bin/env bash
# Wait for Qwen POPE-30 windowed dump to finish, then resume LLaVA AMBER-100
# (resumable via DumpWriter.is_done). Written 2026-07-21 after pausing LLaVA
# so Qwen could finish POPE alone on GPU 0.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source /data/romanus/miniconda3/etc/profile.d/conda.sh
conda activate vlm_hallucination_mitigation
export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"

QWEN_POPE_DIR="$ROOT/data/pope/dumps/qwen2.5-vl-7b-instruct/pope30_windowed_steering"
EXPECTED_CELLS=54
EXPECTED_RECORDS=60
POLL_SEC=30
LOG_DIR="$ROOT/diagnostic_experiments/perception_diag/logs"
mkdir -p "$LOG_DIR"

qwen_pope_done () {
  python - "$QWEN_POPE_DIR" "$EXPECTED_CELLS" "$EXPECTED_RECORDS" <<'PY'
import sys
from pathlib import Path
run, need_cells, need_rec = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
if not run.is_dir():
    sys.exit(1)
complete = 0
for d in run.iterdir():
    if not d.is_dir():
        continue
    man = d / "manifest.jsonl"
    if not man.exists():
        continue
    n = sum(1 for line in open(man) if line.strip())
    if n >= need_rec:
        complete += 1
sys.exit(0 if complete >= need_cells else 1)
PY
}

echo "[$(date -Is)] watcher: waiting for Qwen POPE-30 ($EXPECTED_CELLS cells x $EXPECTED_RECORDS) at $QWEN_POPE_DIR" >&2
while ! qwen_pope_done; do
  python - "$QWEN_POPE_DIR" "$EXPECTED_RECORDS" <<'PY' || true
import sys
from pathlib import Path
run, need = Path(sys.argv[1]), int(sys.argv[2])
complete = partial = 0
part_name = ""
if run.is_dir():
    for d in run.iterdir():
        if not d.is_dir():
            continue
        man = d / "manifest.jsonl"
        n = sum(1 for line in open(man) if line.strip()) if man.exists() else 0
        if n >= need:
            complete += 1
        elif n > 0:
            partial += 1
            part_name = f"{d.name}:{n}"
print(f"qwen_pope complete={complete} partial={partial} {part_name}", flush=True)
PY
  sleep "$POLL_SEC"
done

echo "[$(date -Is)] watcher: Qwen POPE-30 complete — resuming LLaVA AMBER-100" >&2
CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/perception_diag/run_dump.py \
  --model llava-hf/llava-1.5-7b-hf \
  --augmented_jsonl data/amber/augmented_amber100.jsonl \
  --windowed_grid \
  --conditions gold_conditional \
  --run_tag amber100_windowed_steering \
  --max_new_tokens 128 \
  --device_map cuda:0

echo "[$(date -Is)] watcher: LLaVA AMBER-100 dump finished; rebuilding aggregator" >&2
python diagnostic_experiments/perception_diag/build_windowed_steering_summary.py \
  --models llava-1.5-7b-hf \
  || echo "[$(date -Is)] WARNING: aggregator failed after LLaVA AMBER resume" >&2
echo "[$(date -Is)] watcher: done" >&2
