# demos850 top-up run status

Date: 2026-07-28 ~21:42

## Status

**Alive, still in stage 1.** Helper + `stage1_verify_anchors.py` + `tee` all running (PIDs under the nohup job started 21:13:58).

| Stage | Top-up progress |
|-------|-----------------|
| 0 mine | done (650) |
| 1 verify | **207 / 650** (157 pass, 50 reject); 443 pending |
| 1b–4 | 0 top-up ids yet (queued after stage 1) |

Wall clock so far: ~28 min for ~189 new stage-1 items after the earlier ~18 (roughly ~7–9 s/item). Remaining stage 1 alone ≈ 1–1.5 h at that rate; stages 2–4 still ahead.

Logs look frozen at `--- stage1 ---` because Python is block-buffered under the pipe — progress is in the stage1 jsonl files, not the log text.
