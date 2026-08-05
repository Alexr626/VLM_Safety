# Concurrent grids after CHAIR probe; watcher auto-fix?

Date: 2026-07-30

## Concurrent models after the probe?

Yes. The overnight orchestrator still launches **both** model drivers on
`CUDA_VISIBLE_DEVICES=0` after the probe:

1. LLaVA grid immediately
2. sleep 300s
3. Qwen grid

Same card, concurrent processes — unchanged from the plan.

## Does the watcher fix mid-night runtime errors?

**No.** The watcher launched in this session is **monitor-only**: it polls
status/logs until grids are launched (or a probe/orchestrator failure stage),
then writes `watcher_launch_report.md`. It does **not**:

- watch cell-level OOM / CUDA / Python crashes after grids start
- patch code
- restart a dead driver

## What resume support *does* exist (manual or re-launch)

Each cell is invoked with `--skip_if_exists`. Completed cells with
`metric_summary.json` are skipped; mid-cell stops leave
`responses.checkpoint.json` and the runner can continue that cell on a fresh
invocation. Re-running
`run_steering_visual_reasoning_validation.sh <model>` with the same
`RUN_DATE` / `CHAIR_CAP` resumes from unfinished cells.

That resume path is **not** wired to an autonomous overnight fixer yet.

## Probe snapshot at time of question

LLaVA step0 finished: `n_captions_at_token_cap` = 0 at both 256 and 512;
metrics identical. Qwen probe was still running.
