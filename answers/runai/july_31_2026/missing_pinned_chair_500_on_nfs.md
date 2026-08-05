# CHAIR FileNotFoundError on RunAI triple job

Date: 2026-07-31

Cause: `data/chair/pinned_chair_500.json` is covered by gitignore `data/*`, so
`git archive` / sync never put it on NFS. Driver continued with `[warn]` and
still printed `GRID_OK` with zero CHAIR summaries.

Fix: gitignore exception `!data/chair/pinned_chair_500.json`; pack overlay;
verify requires the pin; validation.sh exits 1 if pin missing at start.
Re-SCP pack, sync, resubmit `hal-steer-triple`.
