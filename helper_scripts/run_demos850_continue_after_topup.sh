#!/usr/bin/env bash
# Wait for the live demos_850 top-up (stages 1–4), then finish the plan:
# optional round-2 mine if <295 admissible, assemble+partition, GPU extract, verify.
#
# Does NOT run run_full.sh. Does NOT write demos_v2.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation
export PYTHONUNBUFFERED=1

LOG_DIR=logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/demos850_overnight_continue_${STAMP}.log"
STATUS="$LOG_DIR/demos850_overnight_status.json"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -Is)] $*"; }
write_status() {
  python - "$STATUS" "$1" "$2" <<'PY'
import json, sys, datetime
path, state, detail = sys.argv[1], sys.argv[2], sys.argv[3]
Path = __import__("pathlib").Path
Path(path).write_text(json.dumps({
    "state": state,
    "detail": detail,
    "updated": datetime.datetime.now().isoformat(timespec="seconds"),
}, indent=2) + "\n")
PY
}

EXPECTED_DEMOS_V2_PREFIX=9a44f4afde0324b5
TOPUP1=data/vti/v2/stage0_candidates_topup_2026-07-28.jsonl
TOPUP2=data/vti/v2/stage0_candidates_topup_2026-07-28_round2.jsonl
SNAP=data/vti/v2/_summaries_snapshot_555_2026-07-28/pre_topup_checksums.txt
N_NEW=295
TOPUP_PID="${TOPUP_PID:-}"
SKIP_WAIT="${SKIP_WAIT:-0}"

log "=== demos850 overnight continuer ==="
log "cwd=$ROOT log=$LOG SKIP_WAIT=$SKIP_WAIT TOPUP_PID=${TOPUP_PID:-none}"
write_status waiting "waiting for top-up stages 1-4"

wait_for_topup() {
  if [[ "$SKIP_WAIT" == "1" ]]; then
    log "SKIP_WAIT=1 — not waiting for top-up processes"
    return 0
  fi
  if [[ -n "$TOPUP_PID" ]]; then
    log "waiting for TOPUP_PID=$TOPUP_PID"
    while kill -0 "$TOPUP_PID" 2>/dev/null; do
      sleep 60
    done
    log "TOPUP_PID=$TOPUP_PID exited"
  fi
  # Drain any remaining stage scripts from the top-up helper.
  local pat='data_scripts/vti_demos_v2/stage(1_verify_anchors|1b_allocate|2_write_truthful|3_make_variants|4_verify_faithfulness)|helper_scripts/run_demos850_topup_stages1to4'
  while pgrep -f "$pat" >/dev/null 2>&1; do
    log "stage pipeline still running; sleeping 60s"
    sleep 60
  done
  log "top-up stage processes clear"
}

count_admissible_new() {
  python - <<'PY'
from pathlib import Path
from data_scripts.vti_demos_v2.io_utils import read_jsonl
dims = ("existence", "attribute", "counting", "relation", "all")
base = {r["id"] for r in read_jsonl(Path("data/vti/demos_v2.jsonl"))}
n = 0
for r in read_jsonl(Path("data/vti/v2/stage4_verdicts.jsonl")):
    if r["id"] in base:
        continue
    if not (r.get("value") and str(r["value"]).strip()):
        continue
    hv = r.get("h_values") or {}
    if all(hv.get(d) and str(hv[d]).strip() for d in dims):
        n += 1
print(n)
PY
}

check_demos_v2_intact() {
  local h
  h=$(sha256sum data/vti/demos_v2.jsonl | awk '{print $1}')
  if [[ "${h:0:16}" != "$EXPECTED_DEMOS_V2_PREFIX" ]]; then
    log "FATAL: demos_v2.jsonl hash ${h:0:16} != $EXPECTED_DEMOS_V2_PREFIX"
    write_status failed "demos_v2.jsonl hash mismatch"
    exit 2
  fi
  log "demos_v2.jsonl hash ok (${h:0:16})"
}

check_stage_append_only() {
  # stage1b_allocation.jsonl is rewritten in place by stage1b_allocate.py
  # (existing ids preserved, file fully rewritten via write_jsonl) — byte-prefix
  # append-only does not apply. All other stage*.jsonl + demos_v2 are checked.
  python - "$SNAP" <<'PY'
import hashlib, sys
from pathlib import Path
from data_scripts.vti_demos_v2.io_utils import read_jsonl, load_ids

snap = Path(sys.argv[1])
bad = []
SKIP_PREFIX = {
    "data/vti/v2/stage1b_allocation.jsonl",  # rewritten by design on top-up
}
for line in snap.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    if len(parts) < 4:
        continue
    sha, nbytes, _lines, rel = parts[0], int(parts[1]), parts[2], parts[3]
    if not (rel.startswith("data/vti/v2/stage") and rel.endswith(".jsonl")) and rel != "data/vti/demos_v2.jsonl":
        continue
    if rel in SKIP_PREFIX:
        continue
    p = Path(rel)
    if not p.is_file():
        bad.append(f"missing {rel}")
        continue
    data = p.read_bytes()
    if len(data) < nbytes:
        bad.append(f"shrank {rel}: {len(data)} < {nbytes}")
        continue
    got = hashlib.sha256(data[:nbytes]).hexdigest()
    if got != sha:
        bad.append(f"prefix mismatch {rel}")

# stage1b id-preservation: every demos_v2 id must still have an allocation row
base_ids = {r["id"] for r in read_jsonl(Path("data/vti/demos_v2.jsonl"))}
alloc_ids = load_ids(Path("data/vti/v2/stage1b_allocation.jsonl"))
missing = sorted(base_ids - alloc_ids)
if missing:
    bad.append(f"stage1b missing {len(missing)} demos_v2 ids (e.g. {missing[:5]})")
if len(alloc_ids) < len(base_ids):
    bad.append(f"stage1b n_ids={len(alloc_ids)} < demos_v2={len(base_ids)}")

if bad:
    print("APPEND_ONLY_FAIL")
    for b in bad:
        print(b)
    raise SystemExit(1)
print("APPEND_ONLY_OK (stage1b checked by id-preservation, not byte-prefix)")
PY
}

pick_free_gpu() {
  python - <<'PY'
import subprocess
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
best = None
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    used = float(used)
    if used < 500:  # MiB — treat as free
        print(idx)
        raise SystemExit(0)
    if best is None or used < best[0]:
        best = (used, idx)
raise SystemExit(f"no free GPU (lowest used={best[0]} MiB on {best[1]})")
PY
}

run_round2_if_needed() {
  local n
  n=$(count_admissible_new)
  log "admissible new stage4 passes after round1: $n (need $N_NEW)"
  if (( n >= N_NEW )); then
    return 0
  fi
  log "shortfall — starting mining round 2 (n-candidates 300)"
  write_status round2 "admissible=$n; mining round2"
  python -u data_scripts/vti_demos_v2/stage0_mine_candidates.py \
    --n-candidates 300 \
    --emit-next-batch \
    --out "$TOPUP2" \
    --summary-out data/vti/v2/stage0_summary_topup_2026-07-28_round2.json
  python -u data_scripts/vti_demos_v2/stage1_verify_anchors.py --input "$TOPUP2"
  python -u data_scripts/vti_demos_v2/stage1b_allocate.py
  python -u data_scripts/vti_demos_v2/stage2_write_truthful.py
  python -u data_scripts/vti_demos_v2/stage3_make_variants.py
  python -u data_scripts/vti_demos_v2/stage4_verify_faithfulness.py
  check_demos_v2_intact
  check_stage_append_only
  n=$(count_admissible_new)
  log "admissible new stage4 passes after round2: $n"
  if (( n < N_NEW )); then
    log "FATAL: still only $n admissible (<$N_NEW). Stop — block sizes are field-2."
    write_status stopped_shortfall "admissible=$n need=$N_NEW after two rounds"
    exit 3
  fi
}

assemble_and_partition() {
  write_status assembling "building demos_850 + partition"
  local stage0_args=(--stage0-topup "$TOPUP1")
  if [[ -f "$TOPUP2" ]]; then
    stage0_args=(--stage0-topup "$TOPUP1" "$TOPUP2")
  fi
  python -u data_scripts/vti_demos_v2/assemble_demos_850.py \
    --n-new "$N_NEW" \
    --write-partition \
    "${stage0_args[@]}"
  check_demos_v2_intact
  # byte-identical prefix
  if ! diff -q <(head -555 data/vti/demos_850.jsonl) data/vti/demos_v2.jsonl >/dev/null; then
    # head -555 may break mid-line if pretty; demos are one-record-per-line so ok
    log "FATAL: demos_850 first 555 lines != demos_v2"
    write_status failed "demos_850 prefix mismatch"
    exit 4
  fi
  log "assemble+partition ok"
}

extract_and_verify() {
  local gpu
  gpu=$(pick_free_gpu)
  log "using CUDA_VISIBLE_DEVICES=$gpu"
  write_status extracting_llava "GPU=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u evaluation/run_scripts/extract_demos850_partition_directions.py \
    --model llava-hf/llava-1.5-7b-hf \
    --demos_path data/vti/demos_850.jsonl \
    --partition_path data/vti/demos_850_partition_s42.json \
    --dimensions all existence attribute counting relation \
    --num_demos 50 100 200 500 \
    --rank 2 --seed 42 \
    --check_cache_fidelity \
    > "logs/demos850_partition_llava_2026-07-28.log" 2>&1

  write_status verifying_llava "post-llava verify"
  python -u helper_scripts/verify_demos850_partition_extraction.py \
    --model llava-hf/llava-1.5-7b-hf

  # re-pick in case occupancy changed
  gpu=$(pick_free_gpu)
  log "using CUDA_VISIBLE_DEVICES=$gpu for Qwen"
  write_status extracting_qwen "GPU=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u evaluation/run_scripts/extract_demos850_partition_directions.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --demos_path data/vti/demos_850.jsonl \
    --partition_path data/vti/demos_850_partition_s42.json \
    --dimensions all existence attribute counting relation \
    --num_demos 50 100 200 500 \
    --rank 2 --seed 42 --max_pixels 1003520 \
    --check_cache_fidelity \
    > "logs/demos850_partition_qwen25_2026-07-28.log" 2>&1

  write_status verifying_qwen "post-qwen verify"
  python -u helper_scripts/verify_demos850_partition_extraction.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct
}

append_research_log() {
  python - <<'PY'
from pathlib import Path
import hashlib, datetime
root = Path(".")
demos = root / "data/vti/demos_850.jsonl"
h = hashlib.sha256(demos.read_bytes()).hexdigest()[:16] if demos.is_file() else "MISSING"
part = root / "data/vti/demos_850_partition_s42.json"
entry = f"""

## 2026-07-28/29 — demos_850 overnight continue (auto)

**Plan:** `implementation_plans/7-28-26/demos_850_disjoint_partition_activation_and_direction_extraction_plan_2026-07-28.md`
**Driver:** `helper_scripts/run_demos850_continue_after_topup.sh`
**Finished:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Pool:** `data/vti/demos_850.jsonl` content_hash_sha256_16=`{h}`
**Partition:** `{"present" if part.is_file() else "MISSING"}` (`data/vti/demos_850_partition_s42.json`)
**demos_v2.jsonl:** left at sha256[:16]=`9a44f4afde0324b5` (checked in driver)

**Extract logs:**
- `logs/demos850_partition_llava_2026-07-28.log`
- `logs/demos850_partition_qwen25_2026-07-28.log`
- continuer: see `logs/demos850_overnight_continue_*.log` and `logs/demos850_overnight_status.json`

**Verify reports:** under `experiment_artifacts/vti/{{llava-1.5-7b-hf,qwen2.5-vl-7b-instruct}}/textual_v2/demos850_partition_verification_report_*_2026-07-28.md`

No interpretation in this entry.
"""
path = root / "RESEARCH_LOG.md"
path.write_text(path.read_text() + entry)
print("appended RESEARCH_LOG.md")
PY
}

# ---- main ----
wait_for_topup
write_status post_topup_checks "top-up finished; verifying"
check_demos_v2_intact
check_stage_append_only
run_round2_if_needed
assemble_and_partition
extract_and_verify
append_research_log
write_status complete "assemble+extract+verify finished"
log "=== overnight continuer DONE ==="
log "status file: $STATUS"
