# Verify S2 staging completed + artifacts present

## 1. Job status + logs

```bash
runai workload list -p nlm-mh | grep pd-s2-stage
runai training standard describe pd-s2-stage -p nlm-mh
runai training standard logs pd-s2-stage -p nlm-mh --tail=200
```

**Pass signals in logs:**
- `SMOKE_OK`
- `=== S2 staging COMPLETE … ===`
- Status **Completed** (not Failed)

If status is Failed, the last ~100 log lines usually name the missing step.

## 2. NFS inventory (WinSCP or a tiny probe job)

Paths under `/airl-datalake/romanus/vlm_hallucination/` (pod: `/home/datalake/romanus/vlm_hallucination/`):

| Check | Path | Expect |
|-------|------|--------|
| AMBER images | `data/amber/images/*.jpg` | thousands of JPGs (~408MB tree); script required ≥100 |
| AMBER combined | `data/amber/combined.json` | present after `download_amber.py` |
| demos_v2 | `data/vti/demos_v2.jsonl` | present (~1.4MB) |
| Augment | `diagnostic_experiments/perception_diag/augment/outputs/augmented_amber25.jsonl` | present |
| LLaVA dirs | `experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demosv2_*_all_nd200_s42_r2_prefix/` | `directions.npz`, `components.npz`, `metadata.json` |
| Qwen2.5 dirs | `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demosv2_*_all_nd200_…/` | same (if default 3-model S2) |
| Qwen2 dirs | `experiment_artifacts/vti/qwen2-vl-7b-instruct/textual_v2/demosv2_*_all_nd200_…/` | same |
| COCO val | `data/coco/val2014/` | present (POPE) |
| COCO train | `data/coco/train2014/` | present **if** directions were extracted on-cluster |
| HF cache | `/airl-datalake/romanus/hf_cache/` | LLaVA (+ Qwen if staged) |

## 3. Optional one-GPU verify job

```bash
runai training submit pd-s2-check -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; R=/home/datalake/romanus/vlm_hallucination; echo AMBER=$(find $R/data/amber/images -name "*.jpg" 2>/dev/null|wc -l); echo AUG=$([ -f $R/diagnostic_experiments/perception_diag/augment/outputs/augmented_amber25.jsonl ] && echo ok || echo MISSING); ls -la $R/experiment_artifacts/vti/*/textual_v2/demosv2_*_all_nd200_* 2>/dev/null | head -40; ls $R/data/vti/demos_v2.jsonl'
```

Then: `runai training standard logs pd-s2-check -p nlm-mh --tail=80`

## Ready for S3?
Minimum for `pd-s3-smoke`: AMBER images + LLaVA nd200 dir cache + `augmented_amber25.jsonl` + LLaVA in `hf_cache`. If those exist and logs show `S2 staging COMPLETE`, submit S3.
