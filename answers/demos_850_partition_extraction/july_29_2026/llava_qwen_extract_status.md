# Did LLaVA/Qwen demos850 extracts run?

Date: 2026-07-29 ~05:07

## Answer

**No.** Neither LLaVA nor Qwen partition extraction ran. No `demos_850.jsonl`, no direction cells, act caches still at 3000 files/model.

## What did finish

Top-up stages 1–4 completed (~00:06). `demos_v2.jsonl` still `9a44f4afde0324b5`. Admissible new stage-4 passes: **357** (≥295).

## Why the overnight continuer stopped

At 00:07 it failed the append-only guard on `stage1b_allocation.jsonl` (prefix hash mismatch). Stage1b **rewrites** the allocation file via `write_jsonl` (existing rows preserved, new ids added), so a byte-prefix check is the wrong guard for that file. Other stage `*.jsonl` and `demos_v2.jsonl` passed append-only. Continuer exited; extract never started.

Status left at `"state": "post_topup_checks"`.
