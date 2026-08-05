# Why steer-qwen-smoke failed (Pending with Error pods)

Date: 2026-07-31

Two independent failures in the submit annotation:

1. **Backslash line break** inside `bash -c`:
   `python3 …/run_bash_lf.py \` newline `…/run_steering_qwen_smoke.sh`
   → first line runs `run_bash_lf.py` with **no argv** → IndexError;
   → second line is executed by bash as a path → "No such file or directory"
   if the script is missing, or would have been a confusing bare exec.

2. **`run_steering_qwen_smoke.sh` not on NFS** — prior sync was before that
   file existed in the pack. Need re-SCP of `/tmp/runai_steering_2026-07-31/`
   and a fresh sync that always extracts the repo tarball.

Pending with multiple Error pods = RunAI restarting a crashing job while
waiting for/holding a GPU slot. Deleted the broken `steer-qwen-smoke`.

Fix: single-line `python3 run_bash_lf.py script.sh` in SUBMIT; sync then smoke.
