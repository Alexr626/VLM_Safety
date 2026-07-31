# Prefer RunAI S2 staging job over WinSCP for AMBER/dirs

Yes — a single H100 job that **writes into the extracted repo as the pod user** is the right fix for SFTP permission denied.

## Added
`helper_scripts/runai/run_perception_s2_stage.sh`

Downloads AMBER (`download_amber.py`), ensures COCO val (+ train2014 if direction extract needed), HF weights for LLaVA/Qwen2.5/Qwen2, builds demos_v2 `all@nd200` caches, rebuilds augment JSONLs if missing, one-forward H100 smoke, prints `df`/inventory.

## Before you archive again
Commit (small, newly un-ignored):
- `helper_scripts/runai/run_perception_s2_stage.sh`
- `data/vti/demos_v2.jsonl`, `demos_v2_order_s42.json`, `qual_subset_chair5_amber25.json`
- `data/amber/pinned_amber_disc_450.json`
- `.gitignore` exceptions

Then rebuild `~/dev/vti_repo.tar.gz` and WinSCP **only the tarball** to `/airl-datalake/romanus/`.

## Submit S2
See script header / `readme.md` (`pd-s2-stage`). For a faster S3-only gate first:

```bash
# optional: LLaVA only
# STAGE_MODELS='llava-hf/llava-1.5-7b-hf'
```

(set inside the submit command before invoking the script if desired).

Then S3 (`pd-s3-smoke`) once S2 logs show `S2 staging COMPLETE`.
