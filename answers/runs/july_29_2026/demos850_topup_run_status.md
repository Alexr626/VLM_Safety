# demos_850 top-up run status

Date asked: 2026-07-29  
Log: `logs/demos850_topup_nohup_20260728_211357.out`

## Top-up stages 1–4: completed

- Started `2026-07-28T21:13:58-04:00`, finished `2026-07-29T00:06:44-04:00`, ended with `=== done ===`.
- No Traceback / CUDA / OOM / Killed in the top-up logs.

| Stage | This-run pass | This-run reject | Notes |
|---|---:|---:|---|
| 1 | 487 | 145 | 632 pending / 650 input |
| 1b | — | — | Wrote 1335 allocations → `stage1b_allocation.jsonl` |
| 2 | 497 | 3 | |
| 3 | 487 | 10 | existence det+grammar=454, llm=33 |
| 4 | 357 | 130 | `n_pass_total` now 912 |

`demos_v2.jsonl` content hash still `9a44f4afde0324b5` (555 lines); intact per overnight check.

## Overnight continuer: stopped after post-topup check

`logs/demos850_overnight_continue_20260728_222436.log`:

1. Waited for TOPUP_PID=416721; it exited; stage processes clear.
2. `demos_v2.jsonl` hash ok.
3. `APPEND_ONLY_FAIL` / `prefix mismatch data/vti/v2/stage1b_allocation.jsonl`

That gate hashes the pre-topup byte prefix of each stage `*.jsonl` against
`data/vti/v2/_summaries_snapshot_555_2026-07-28/pre_topup_checksums.txt`.
Expected prefix for `stage1b_allocation.jsonl`: 8,772,894 bytes. Current file is
11,229,469 bytes (1335 lines). The first 8,772,894 bytes no longer match the
snapshot hash — so the file was not a pure append of new rows onto the old
bytes (rewrite / reorder / rewrite-in-place), and the continuer aborted before
assemble / partition / extraction.

Consequences as of check time:

- `data/vti/demos_850.jsonl` does not exist.
- `stage5_summary.json` still `n_final=555`.
- Status file left at `post_topup_checks` / "top-up finished; verifying".

Cache fidelity (earlier same evening): LLaVA and Qwen2.5 both `pass=True`.
