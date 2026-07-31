# Submit perception S2 staging as a RunAI job

From **lambdab2** (`runai login`; `runai project set nlm-mh`):

```bash
runai training submit pd-s2-stage -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; mkdir -p "$REPO"; tar -xzf "$BASE/vti_repo.tar.gz" -C "$REPO"; python3 "$REPO/helper_scripts/runai/run_bash_lf.py" "$REPO/helper_scripts/runai/run_perception_s2_stage.sh"'
```

Monitor:
```bash
runai workload list -p nlm-mh
runai training logs pd-s2-stage -p nlm-mh
```

Success line in logs: `S2 staging COMPLETE`.

Optional faster S3-only gate (LLaVA only) — insert before the python3 call:
`export STAGE_MODELS=llava-hf/llava-1.5-7b-hf;`
