# demos_v2.1 — next commands after diversity implementation

## Clarifications resolved (Task 0)

1. **Stage-0 gap (v2.0):** aggregate category extrema, multi-instance OK → **v2.1 uses singleton pairs + edge-to-edge** (`geometry.py`).
2. **Span replace (v2.0):** caption-global → **v2.1 sentence-scoped** `{sentence_idx, text}`.
3. **`run_full.sh`:** new sibling; `run_full300.sh` can remain as thin wrapper. Fresh v2.1 run required (no migration of v2.0 stage JSONLs).

## Frozen v2.0

`data/vti/v2_poc_2026-07-12/` — includes finals + stage artifacts. Do not delete before a new mine/wipe.

## Fresh v2.1 mine + mock dry-run (recommended before paid)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm_hallucination_mitigation

# Remine stage0 into v2.1 option-set schema (overwrites live stage0_*)
python data_scripts/vti_demos_v2/stage0_mine_candidates.py --n-candidates 300

# Wipe downstream only (keeps new stage0); then mock small-N
rm -f data/vti/v2/stage1_*.jsonl data/vti/v2/stage1_summary.json \
      data/vti/v2/stage1b_*.jsonl data/vti/v2/stage1b_summary.json \
      data/vti/v2/stage2_*.jsonl data/vti/v2/stage2_summary.json \
      data/vti/v2/stage3_*.jsonl data/vti/v2/stage3_summary.json \
      data/vti/v2/stage4_*.jsonl data/vti/v2/stage4_summary.json \
      data/vti/v2/stage5_summary.json \
      data/vti/v2/calls/stage{1,2,3,4}.jsonl \
      data/vti/demos_v2.jsonl data/vti/demos_v2_*.jsonl

python data_scripts/vti_demos_v2/stage1_verify_anchors.py --provider mock --limit 5
python data_scripts/vti_demos_v2/stage1b_allocate.py
python data_scripts/vti_demos_v2/stage2_write_truthful.py --provider mock --limit 5
python data_scripts/vti_demos_v2/stage3_make_variants.py --provider mock --limit 5
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --provider mock --limit 5
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 5
python data_scripts/vti_demos_v2/render_review.py --stage 1b
python data_scripts/vti_demos_v2/render_review.py --stage final --open
```

## Paid pilot (~20) then full pool

```bash
# after mock looks good — wipe stage1–5 again, keep stage0
# then omit --provider mock; use --limit 20 through stages 1–4
# full pool:
N_CANDIDATES=300 nohup bash data_scripts/vti_demos_v2/run_full.sh \
  > data/vti/v2/logs/full_v21_nohup.out 2>&1 &
```

## Tests

```bash
python -m pytest tests/test_demos_v2_validators.py tests/test_demos_v2_allocator.py -q
# 24 passed
```
