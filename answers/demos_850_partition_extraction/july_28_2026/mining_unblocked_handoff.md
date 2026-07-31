# demos_850 mining handoff — P1/P2 complete

Date: 2026-07-28

## Status

**P1 and P2 are complete. Mining is unblocked.**

| Gate | Status |
|------|--------|
| P1 `--summary-out` on `stage0_mine_candidates.py` | Done (default path unchanged when flag absent) |
| P2 snapshot + `pre_topup_checksums.txt` | Done under `data/vti/v2/_summaries_snapshot_555_2026-07-28/` |
| Check 0.1 (`demos_v2.jsonl` completeness) | Pass: 555 rows, 555 unique ids, all five `h_values`, question fixed |
| Check 0.4 cache fidelity (both models, GPU 0) | **Pass** — shared `_act_cache/` may be extended after assembly |

`demos_v2.jsonl` sha256 still starts with `9a44f4afde0324b5…`.

Cursor finished extract/assemble/partition code and check 0.4 while you mine. Remaining after your stages 1–4: assemble → partition → GPU direction extract → verify.

---

## Hard rules (read before any command)

1. **Never run** `data_scripts/vti_demos_v2/run_full.sh` — it `rm -f`s stage 1–5 artifacts and `demos_v2*.jsonl`.
2. **`data/vti/demos_v2.jsonl` is read-only.** Do not append to it. New pool is a separate `demos_850.jsonl` (assembled later by Cursor).
3. **Run stage 1 immediately after stage 0.** `--emit-next-batch` only sees new ids once stage 1 writes them; a second stage-0 run before stage 1 can re-emit the same candidates.
4. **Never mock against the real top-up file.** Mock writes real rows. If smoke-testing: `--provider mock --limit 5` with a scratch `--input`.
5. Stages 1b–4 take **no** `--input`; they skip already-processed ids and only touch new ones.
6. After stage 4: `sha256sum data/vti/demos_v2.jsonl` must still be `9a44f4af…`, and every `data/vti/v2/stage*.jsonl` must be append-only vs `pre_topup_checksums.txt`.
7. If fewer than 295 admissible stage-4 passes after **two** extra mining rounds: **stop** — do not change block sizes.

---

## Commands (run in order, conda env `vlm_hallucination_mitigation`)

Requires `ANTHROPIC_API_KEY` in repo-root `.env` (already present on this machine).

### Step 1 — mine 650 candidates

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
conda activate vlm_hallucination_mitigation

python data_scripts/vti_demos_v2/stage0_mine_candidates.py \
  --n-candidates 650 \
  --emit-next-batch \
  --out data/vti/v2/stage0_candidates_topup_2026-07-28.jsonl \
  --summary-out data/vti/v2/stage0_summary_topup_2026-07-28.json
```

Expected: 650 new rows; **must not** touch `data/vti/v2/stage0_summary.json`.

### Steps 2–5 — stages 1 → 4 (start stage 1 promptly)

```bash
python data_scripts/vti_demos_v2/stage1_verify_anchors.py \
  --input data/vti/v2/stage0_candidates_topup_2026-07-28.jsonl

python data_scripts/vti_demos_v2/stage1b_allocate.py
python data_scripts/vti_demos_v2/stage2_write_truthful.py
python data_scripts/vti_demos_v2/stage3_make_variants.py
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py
```

Default providers: stage1 Sonnet 5, stage2 Opus 4.8, stage3 Haiku 4.5, stage4 Sonnet 5.

Wall-clock estimate: ~2–5 hours for the 650-candidate batch (~2260 API calls).

### After stage 4 — verify before anything else

```bash
# demos_v2 must be unchanged
sha256sum data/vti/demos_v2.jsonl
# expect: 9a44f4afde0324b5...

# count new stage-4 passes (ids not in the original 555)
python - <<'PY'
import json
from pathlib import Path
base = {json.loads(l)["id"] for l in Path("data/vti/demos_v2.jsonl").read_text().splitlines() if l.strip()}
new = []
for l in Path("data/vti/v2/stage4_verdicts.jsonl").read_text().splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    if r["id"] not in base:
        new.append(r)
print(f"new_stage4_passes={len(new)}  need>=295")
PY
```

Append-only check against the snapshot (first N bytes must match):

```bash
SNAP=data/vti/v2/_summaries_snapshot_555_2026-07-28/pre_topup_checksums.txt
while read -r sha bytes lines path; do
  [[ "$sha" == \#* || -z "$sha" ]] && continue
  [[ "$path" != data/vti/v2/stage*.jsonl && "$path" != data/vti/demos_v2.jsonl ]] && continue
  cur=$(head -c "$bytes" "$path" | sha256sum | awk '{print $1}')
  # Note: pre_topup records full-file sha256; for append-only, compare prefix bytes:
  prefix_sha=$(python3 -c "import pathlib,hashlib; p=pathlib.Path('$path'); print(hashlib.sha256(p.read_bytes()[:$bytes]).hexdigest())")
  if [[ "$prefix_sha" != "$sha" ]]; then
    echo "FAIL append-only: $path"
  else
    echo "ok $path"
  fi
done < <(awk 'NF>=4 && $1 !~ /^#/ {print}' "$SNAP")
```

(Or manually: for each stage jsonl row in the checksum file, `python3 -c "… read_bytes()[:bytes] …"` must equal the recorded sha256.)

### If new stage-4 passes < 295 — one more round (max two total)

```bash
python data_scripts/vti_demos_v2/stage0_mine_candidates.py \
  --n-candidates 300 \
  --emit-next-batch \
  --out data/vti/v2/stage0_candidates_topup_2026-07-28_round2.jsonl \
  --summary-out data/vti/v2/stage0_summary_topup_2026-07-28_round2.json

python data_scripts/vti_demos_v2/stage1_verify_anchors.py \
  --input data/vti/v2/stage0_candidates_topup_2026-07-28_round2.jsonl
python data_scripts/vti_demos_v2/stage1b_allocate.py
python data_scripts/vti_demos_v2/stage2_write_truthful.py
python data_scripts/vti_demos_v2/stage3_make_variants.py
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py
```

Then re-count. If still < 295 after two rounds: stop and report the count — block sizes are a field-2 change only you make.

---

## What Cursor still needs from you (after mining)

1. Confirmation that stage 4 finished and `new_stage4_passes >= 295`.
2. Confirmation `demos_v2.jsonl` hash is still `9a44f4afde0324b5…` and stage files stayed append-only.
3. Which top-up stage0 file(s) to use for assembly rank order (default: `stage0_candidates_topup_2026-07-28.jsonl`; if round 2 ran, tell Cursor so rank order can include both).

Then Cursor runs `assemble_demos_850.py`, builds the partition, and starts GPU extraction (check 0.4 fidelity first).

---

## Snapshot location (do not delete)

`data/vti/v2/_summaries_snapshot_555_2026-07-28/`

Contains copied `stage*_summary.json` plus `pre_topup_checksums.txt` (stage jsonl + `demos_v2.jsonl` + `demos_v2_order_s42.json` + all 40 `demosv2_9a44f4af_*/directions.npz` hashes).
