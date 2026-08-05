# Overnight crash babysitter

Date: 2026-07-30

Script: `evaluation/run_scripts/babysit_steering_visual_reasoning_grids.py`

Detects true driver crashes (process gone before `grid process finished`
banner). Relaunches the same model driver with the same `RUN_DATE` /
`CHAIR_CAP` so `--skip_if_exists` resumes.

OOM-death while sibling still running: wait for sibling to finish, then
restart (concurrent re-OOM avoidance). Does not patch code. Max 8
restarts per model.
