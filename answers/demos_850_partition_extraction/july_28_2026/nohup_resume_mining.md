# Kill and resume mining under nohup

Date: 2026-07-28

## Answer

Yes. Kill the foreground stage-1 process and restart stages 1–4 under `nohup` (or tmux). Stage scripts skip already-processed ids, so the ~17 top-up ids already written are kept and not re-billed.

**Do not re-run stage 0** — `stage0_candidates_topup_2026-07-28.jsonl` (650 rows) already exists.

## Current state (at handoff)

- Stage 0: done (650 candidates)
- Stage 1: running in Cursor terminal; ~17/650 top-up ids done, ~633 pending
- PID (when checked): `403162` — confirm with `pgrep -af stage1_verify` before kill

## Commands

### 1. Stop the foreground stage 1

In the Cursor/SSH terminal where it is running: `Ctrl-C`

Or from another shell:

```bash
pgrep -af stage1_verify_anchors
kill <pid>          # SIGTERM first
# if still alive after a few seconds:
# kill -9 <pid>
```

### 2. Resume stages 1→4 detached (recommended: nohup + helper)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
chmod +x helper_scripts/run_demos850_topup_stages1to4.sh

nohup bash helper_scripts/run_demos850_topup_stages1to4.sh \
  > logs/demos850_topup_nohup_$(date +%Y%m%d_%H%M%S).out 2>&1 &

echo "pid=$!"
disown
```

You can close Cursor / SSH after that. The helper chains stage1 → 1b → 2 → 3 → 4, tees a stamped log under `logs/demos850_topup_stages1to4_*.log`, and stage1 will print something like `Stage 1: 633 pending / 650 input` (numbers depend on how far the kill landed).

### 3. Monitor later

```bash
pgrep -af 'stage[1-4]|run_demos850_topup'
tail -f logs/demos850_topup_stages1to4_*.log   # pick the newest
```

### Optional: tmux instead of nohup

```bash
tmux new -s demos850
cd ~/dev/vlm_hallucination_mitigation_summer_2026
conda activate vlm_hallucination_mitigation
bash helper_scripts/run_demos850_topup_stages1to4.sh
# Ctrl-b d to detach; reattach: tmux attach -t demos850
```

## After it finishes

Same checks as before: `sha256sum data/vti/demos_v2.jsonl` still `9a44f4af…`, and count new stage-4 passes ≥ 295. Then tell Cursor so assemble / extract can run.
