# nohup Exit 1 — wrong cd in helper

Date: 2026-07-28

## Cause

`helper_scripts/run_demos850_topup_stages1to4.sh` used `cd "$(dirname "$0")/../.."`, which landed in `/home/romanus/dev` instead of the repo root. Relative path to the top-up file failed → Exit 1. Stray log: `/home/romanus/dev/logs/demos850_topup_stages1to4_20260728_211300.log`.

## Fix

`ROOT="$(cd "$(dirname "$0")/.." && pwd)"` then `cd "$ROOT"`.

## Relaunch

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026

nohup bash helper_scripts/run_demos850_topup_stages1to4.sh \
  > logs/demos850_topup_nohup_$(date +%Y%m%d_%H%M%S).out 2>&1 &
echo "pid=$!"
disown

# confirm it stayed up
sleep 2
pgrep -af 'run_demos850_topup|stage1_verify'
ls -lt logs/demos850_topup_stages1to4_*.log | head -1
```
