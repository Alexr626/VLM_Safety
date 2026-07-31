# Will renaming NFS `vlm_hallucination` to match lambdab2 fix S3a?

**Date:** 2026-07-17  
**Context:** Qwen2 S3a RunAI smoke fails with `FileNotFoundError` on  
`/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/data/amber/images/AMBER_13.jpg`

## Short answer

**No — do not rename the NFS repo.** It would break RunAI helpers, and it would **not** fix this error.

## What actually failed

`run_dump.py` opens `item["image_path"]` from `augmented_amber25.jsonl`. That JSONL was built on lambdab2 and stores **absolute** paths:

```
/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/data/amber/images/AMBER_13.jpg
```

On the pod the same file lives under NFS, e.g.:

```
/home/datalake/romanus/vlm_hallucination/data/amber/images/AMBER_13.jpg
```

The mismatch is the full host prefix (`/home/romanus/dev/...` vs `/home/datalake/romanus/...`), not merely the final directory name.

## Would renaming NFS break other code?

**Yes.** Every RunAI helper hardcodes:

```bash
BASE=/home/datalake/romanus
REPO=$BASE/vlm_hallucination
```

Affected includes (non-exhaustive): `setup_vlm.sh`, `run_vti.sh`, `run_smoke_pope.sh`, `run_perception_s2_*.sh`, `run_perception_s3_smoke.sh`, `run_perception_s3a_qwen2.sh`, plus submit one-liners in `readme.md` / `HANDOFF.md`. Renaming without updating all of those (and existing WinSCP paths / prior result trees) breaks jobs.

## Fix (do this instead)

Rebuild (or rewrite) the aug JSONL **on NFS** so `image_path` resolves under `project_root()`:

1. On NFS, delete or move the bad  
   `vlm_hallucination/diagnostic_experiments/perception_diag/augment/outputs/augmented_amber25.jsonl`  
   (S2 only rebuilds when the file is **missing**.)
2. Re-run `build_augmented_jsonl.py` inside the pod (or let `run_perception_s3_smoke.sh` rebuild when missing).
3. Confirm one line’s `image_path` starts with `/home/datalake/romanus/vlm_hallucination/…` and the file exists.
4. Re-submit S3a.

Optional longer-term: teach `run_dump.py` to remap absolute `…/data/…` paths onto `project_root()/data/…` when the stored path is missing — that would make lambdab2-built JSONLs portable without a rebuild.
