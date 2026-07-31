# Remap baked lambdab2 absolute paths on NFS

**Date:** 2026-07-17  
**Question:** Can we write a script to replace absolute file paths in the NFS repo copy so they use the correct parent directory?

## Answer

Yes. Added `helper_scripts/runai/remap_lambdab2_paths.py`.

It text-replaces the lambdab2 prefix

`/home/romanus/dev/vlm_hallucination_mitigation_summer_2026`

with the local `project_root()` (on RunAI: `/home/datalake/romanus/vlm_hallucination`) in `.json` / `.jsonl` under `data/`, augment outputs, and `experiment_artifacts/`.

Dry-run by default; `--apply` writes.

### On NFS (after tarball extract includes this script)

```bash
cd /home/datalake/romanus/vlm_hallucination
# dry-run
$MM run -p $ENV_PREFIX python helper_scripts/runai/remap_lambdab2_paths.py
# write
$MM run -p $ENV_PREFIX python helper_scripts/runai/remap_lambdab2_paths.py --apply
```

`run_perception_s3a_qwen2.sh` and `run_perception_s3_smoke.sh` now call `--apply` before `run_dump.py`, so a fresh extract + re-submit of S3a should clear the `AMBER_13.jpg` FileNotFoundError once the new helper is on NFS.

### Do not

- Run `--apply` on lambdab2 with `--new` set to the NFS path (would break local absolute paths).
- Rely on renaming the NFS repo folder (still wrong host prefix; breaks hardcoded `REPO=$BASE/vlm_hallucination`).

### Local dry-run check (2026-07-17)

With `--new /home/datalake/romanus/vlm_hallucination`: 16 files / 65812 replacements, including `augmented_amber25.jsonl` (25) and `data/amber/combined.json` (15220).
