# Built-in RunAI job logging + sync verify

Date: 2026-07-31

Preference: do not rely on separate verify probes for sync/smoke/full.

Implemented:
- `sync_steering_llava.sh` — extract tarballs, run layout verify, print
  `VERIFY_SYNC_OK` / `SYNC_OK`, tee to `$BASE/logs/runai/`.
- `verify_steering_nfs_layout.sh` — required scripts, SKIP_CONDA, 4 meandiff
  dirs + npz, micromamba env.
- `runai_job_logging.sh` — sourced by smoke/full; same tee + verify at start.
- Smoke/full end markers: `SMOKE_OK` / `GRID_OK`.

Always invoke helpers via `python3 …/run_bash_lf.py …/*.sh` (never `bash` on the `.py`).

Re-SCP updated `vti_repo_working_tree.tar.gz` before the next sync.
