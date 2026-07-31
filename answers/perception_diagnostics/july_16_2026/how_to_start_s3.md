# Starting S3 (RunAI smoke) — updated for S2 staging job

S3 = re-run **S0 config** (LLaVA, AMBER-25, cells B0/B2/B8) as one H100 job, then compare parsed outcomes to lambdab2 S0.

**Preferred path:** do **not** WinSCP AMBER images into `vlm_hallucination/` (permission denied on pod-owned dirs). Run **S2** on a GPU pod instead.

## 1. Commit + archive (lambdab2)

Include:
- `helper_scripts/runai/run_perception_s2_stage.sh` (+ S3 helper if not already)
- `diagnostic_experiments/perception_diag/` (dumps gitignored)
- Small data pins now un-ignored: `data/vti/demos_v2.jsonl`, `demos_v2_order_s42.json`, `qual_subset_chair5_amber25.json`, `data/amber/pinned_amber_disc_450.json`

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git archive --format=tar.gz -o ~/dev/vti_repo.tar.gz HEAD
tar -tzf ~/dev/vti_repo.tar.gz | grep -E 'run_perception_s2_stage|demos_v2.jsonl|run_dump'
```

WinSCP `~/dev/vti_repo.tar.gz` → `/airl-datalake/romanus/vti_repo.tar.gz` (binary).

## 2. Submit S2 staging

```bash
runai training submit pd-s2-stage -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; mkdir -p "$REPO"; tar -xzf "$BASE/vti_repo.tar.gz" -C "$REPO"; python3 "$REPO/helper_scripts/runai/run_bash_lf.py" "$REPO/helper_scripts/runai/run_perception_s2_stage.sh"'
```

Faster S3-only gate: set `STAGE_MODELS=llava-hf/llava-1.5-7b-hf` in that command before the python3 line.

Wait for logs: `S2 staging COMPLETE`.

## 3. Submit S3 smoke

```bash
runai training submit pd-s3-smoke -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; python3 "$REPO/helper_scripts/runai/run_bash_lf.py" "$REPO/helper_scripts/runai/run_perception_s3_smoke.sh"'
```

## 4. Pull + compare
WinSCP back: `…/dumps/s3_runai_smoke/` vs lambdab2 `…/dumps/s0_smoke/` (parsed outcomes).
