# Auto cutover: β=0.5 done → β=0.9 cell-partitioned triple

Date: 2026-08-06  
Request: watch for β=0.5 finish, then run three concurrent Qwen instances on remaining β=0.9 cells.

## What was set up

1. **Launcher** `helper_scripts/runai/run_amber_expanded_qwen_beta09_cell_triple_one_h100.sh`  
   Three workers (concurrent `$!`, per-worker logs), all `BETAS=0.9`:
   - `meandiff-5-14` → `raw_mean_difference` + `LAYER_SETS=5-14` (resumes checkpoint)
   - `meandiff-15-24` → `raw_mean_difference` + `LAYER_SETS=15-24`
   - `pca-windows` → `live_pc1_plus_mean` + `LAYER_SETS="all 5-14 15-24"`

2. **Babysitter** `helper_scripts/runai/babysit_amber_qwen_cutover_beta09_triple.py`  
   Polls `amber-qwen-beta-triple` via `runai training exec` until all six β=0.5 steered
   cells have `responses.json`, then deletes that job and submits
   `amber-qwen-beta09-triple` with quote-safe NFS launcher patch (`echo b64 | base64 -d`).

## Status at arming (~13:59 ET)

β=0.5: 4/6 cells done; PCA `layers_5_14` in progress; PCA `layers_15_24` TODO.  
Babysitter started as a background process on lambdab2 (poll 45s).  
Log: `logs/babysit_amber_qwen_cutover_beta09_triple.log`.

## Operator checks

```bash
tail -f logs/babysit_amber_qwen_cutover_beta09_triple.log
python3 helper_scripts/runai/babysit_amber_qwen_cutover_beta09_triple.py --check_once
runai workload list -p nlm-mh | grep amber-qwen
```

Success markers: log line `CUTOVER_OK`, file `logs/amber_qwen_cutover_beta09_done.json`,
job `amber-qwen-beta09-triple` Running, pod log `LAUNCHER_PATCHED_BETA09` / three worker starts.

## Cutover result (2026-08-06 ~15:16 ET / 19:16Z)

Babysitter saw β=0.5 `ALL_DONE True` (6/6 cells), deleted `amber-qwen-beta-triple`, submitted
`amber-qwen-beta09-triple`. Confirmed `LAUNCHER_PATCHED_BETA09` and all three workers started
(`meandiff-5-14`, `meandiff-15-24`, `pca-windows`). Job Running with 1×H100.
