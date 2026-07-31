# S2 dirs-only — extract demos_v2 directions

AMBER images, `demos_v2.jsonl`, and augment JSONLs already on NFS. Missing: textual_v2 direction caches for LLaVA / Qwen2.5 / Qwen2.

**New script:** `helper_scripts/runai/run_perception_s2_dirs.sh`  
Skips AMBER/augment; ensures COCO train2014 + HF weights; extracts `all@nd200` per model; tees to `logs/perception_s2_dirs.log`.

## Commit + tarball (required — script not on NFS yet)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
# commit helper + any pending S2 log-tee changes, then:
git archive --format=tar.gz -o ~/dev/vti_repo.tar.gz HEAD
```

WinSCP `~/dev/vti_repo.tar.gz` → `/airl-datalake/romanus/`.

## Submit

```bash
runai training delete pd-s2-dirs -p nlm-mh 2>/dev/null || true
runai training submit pd-s2-dirs -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; mkdir -p "$REPO"; tar -xzf "$BASE/vti_repo.tar.gz" -C "$REPO"; python3 "$REPO/helper_scripts/runai/run_bash_lf.py" "$REPO/helper_scripts/runai/run_perception_s2_dirs.sh"'
```

LLaVA-only (faster, enough for S3):
prepend `export STAGE_MODELS=llava-hf/llava-1.5-7b-hf;` before the `python3` call.

Success: log line `S2 dirs-only COMPLETE` and dirs under  
`experiment_artifacts/vti/{llava-1.5-7b-hf,qwen2.5-vl-7b-instruct,qwen2-vl-7b-instruct}/textual_v2/demosv2_*_all_nd200_s42_r2_prefix/`.
