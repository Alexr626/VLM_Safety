# S3a / S3b launch (plan v2)

**Plan:** `implementation_plans/EXPERIMENT_PLAN_perception_diagnostics(1).md`  
S2 dirs-only completed (LLaVA + Qwen2.5 + Qwen2 nd200 caches on NFS).

## Config (both)
- Cells: `smoke` = B0–B9 + B11 (additive-layer 0.5 canary)
- AMBER-25, 5 conditions, `max_new_tokens=128`
- Manifest: `degeneracy_flag`, `truncated`, `status`

## S3b — lambdab2 (Qwen2.5-VL) — started
```bash
CUDA_VISIBLE_DEVICES=0 bash diagnostic_experiments/perception_diag/run_scripts/run_s3b_qwen25_amber25.sh
```
Out: `diagnostic_experiments/perception_diag/qwen2.5-vl-7b-instruct/dumps/s3b_qwen25_amber25/` (+ C1 JSON).

## S3a — RunAI (Qwen2-VL)

1. Commit new helpers + dump changes, then:
```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git archive --format=tar.gz -o ~/dev/vti_repo.tar.gz HEAD
tar -tzf ~/dev/vti_repo.tar.gz | grep run_perception_s3a_qwen2
```
2. WinSCP `~/dev/vti_repo.tar.gz` → `/airl-datalake/romanus/vti_repo.tar.gz`
3. Submit:
```bash
runai training submit pd-s3a-qwen2 -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; mkdir -p "$REPO"; tar -xzf "$BASE/vti_repo.tar.gz" -C "$REPO"; python3 "$REPO/helper_scripts/runai/run_bash_lf.py" "$REPO/helper_scripts/runai/run_perception_s3a_qwen2.sh"'
```
Log: `/airl-datalake/romanus/logs/perception_s3a_qwen2.log`  
Out: `…/qwen2-vl-7b-instruct/dumps/s3a_qwen2_amber25/`
