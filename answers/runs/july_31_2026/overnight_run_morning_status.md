# Overnight steering validation — morning status (2026-07-31)

## SSH / Cursor reload

Cancelling the Cursor orchestrator agent on SSH reconnect did **not** stop the
nohup grid drivers or babysitter. Driver PIDs still alive with PPID 1 (or
babysitter), still writing cells.

## Probe / cap

Qwen hit 256-token cap on 6/20 probe captions → `CHAIR_CAP=512` for the grid.
LLaVA probe: 0 at 256.

## Live at check (~08:45 EDT)

- LLaVA: CHAIR steered cell `layers=all nd=50 beta=0.5` (~450/500)
- Qwen: AMBER steered cell `layers=5-14 nd=500 beta=0.9` (~250/450)
- GPU 0 ~30888 MiB, 100% util; both concurrent
- Babysitter: `watching`, **0 restarts**, no crash events

## Completed cells

57 with `metric_summary.json` (of 370 expected):
- LLaVA: 39 (AMBER essentially complete + CHAIR baseline + maybe early CHAIR)
- Qwen: 18 (AMBER partial)
- by bench: amber 55, chair 2; POPE not started

No Traceback / CUDA OOM killing either driver found in the grid logs at this check.
