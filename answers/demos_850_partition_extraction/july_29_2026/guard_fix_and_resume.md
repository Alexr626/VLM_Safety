# Guard fix + resume

Date: 2026-07-29 ~05:10

## Fix

`check_stage_append_only` in `helper_scripts/run_demos850_continue_after_topup.sh`:
- Skip byte-prefix check for `stage1b_allocation.jsonl` (rewritten by design)
- Instead require every `demos_v2` id still present in stage1b
- Added `SKIP_WAIT=1` for resume after mining already finished

## Resume

Launched with `SKIP_WAIT=1`. Guard passed. Assembled `demos_850.jsonl` (hash `ba05bd960cad0c18`, 850 rows) + partition. LLaVA extract running on GPU 0 (`state: extracting_llava`).
