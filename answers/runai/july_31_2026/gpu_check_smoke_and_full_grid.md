# RunAI GPU check + LLaVA smoke / full grid commands

Date: 2026-07-31

## GPU / quota check (lambdab2)

```bash
export PATH="$HOME/.runai/bin:$PATH"
runai whoami
runai project set nlm-mh
runai workload list -p nlm-mh
runai workload list -p nlm-mh --status Running
runai project list --json   # deserved vs allocated under nlm-mh.status
```

`runai node list` / `nodepool list` are usually Forbidden for researchers.

Snapshot at write time: `nlm-mh` deserved **4** GPUs; status allocated **3**; one Running workload (`qwen3-embedding-4b2`, 1 GPU). Roughly **1** GPU of project headroom — enough for one training job if the cluster schedules it. Pending after submit ⇒ wait.

## What remains for LLaVA (local disk)

- AMBER: 37/37 done (stay on lambdab2; no need on RunAI)
- CHAIR: 3 cells done; rest + baseline remaining
- POPE (random/popular/adversarial): none yet

## Scripts added / fixed

- `helper_scripts/runai/run_steering_llava_smoke.sh` — 5-sample POPE baseline + one steered cell
- `helper_scripts/runai/SUBMIT_STEERING_LLAVA.md` — full command sheet
- `run_steering_visual_reasoning_validation.sh` — `SKIP_CONDA_ACTIVATE=1` for RunAI (conda absent in pods)
- Re-packed under `/tmp/runai_steering_llava_2026-07-30/` — **re-SCP** `vti_repo_working_tree.tar.gz` before sync

## Order

1. Re-upload updated tarball to NFS `/airl-datalake/romanus/`
2. `sync-steering-llava` (extract)
3. `steer-llava-smoke` → look for `SMOKE_OK`
4. `hal-steer-llava` → CHAIR+POPE with `--skip_if_exists`
