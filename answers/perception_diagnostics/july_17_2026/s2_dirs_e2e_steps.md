# End-to-end: tarball → NFS → pd-s2-dirs

## 1. Commit (if not already)

On lambdab2, ensure `run_perception_s2_dirs.sh` is in HEAD:

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git status
git ls-files helper_scripts/runai/run_perception_s2_dirs.sh
```

If untracked/modified, commit it (and related helper/docs changes) before archiving.

## 2. Build tarball (not in /tmp)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git archive --format=tar.gz -o ~/dev/vti_repo.tar.gz HEAD
ls -lh ~/dev/vti_repo.tar.gz
tar -tzf ~/dev/vti_repo.tar.gz | grep run_perception_s2_dirs
```

File: `/home/romanus/dev/vti_repo.tar.gz`

## 3. Upload (WinSCP)

- Session: **lambdab2** → copy `/home/romanus/dev/vti_repo.tar.gz`
- Session: **NFS** (`air-datalake@gpustorage-1…`) → paste as `/airl-datalake/romanus/vti_repo.tar.gz` (binary mode)
- Do **not** put it only inside `vlm_hallucination/` — job reads `$BASE/vti_repo.tar.gz`

## 4. Submit dirs-only job (lambdab2)

```bash
runai login   # if token stale
runai project set nlm-mh

runai training delete pd-s2-dirs -p nlm-mh 2>/dev/null || true

runai training submit pd-s2-dirs -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; mkdir -p "$REPO"; tar -xzf "$BASE/vti_repo.tar.gz" -C "$REPO"; python3 "$REPO/helper_scripts/runai/run_bash_lf.py" "$REPO/helper_scripts/runai/run_perception_s2_dirs.sh"'
```

LLaVA-only (faster, enough for S3): insert  
`export STAGE_MODELS=llava-hf/llava-1.5-7b-hf;`  
immediately before the `python3` call in that `--command`.

## 5. Monitor

```bash
runai workload list -p nlm-mh | grep pd-s2-dirs
runai training standard logs pd-s2-dirs -p nlm-mh --tail=100
# if logs fail, WinSCP:
#   /airl-datalake/romanus/logs/perception_s2_dirs.log
```

Success markers: status **Completed**, log line `S2 dirs-only COMPLETE`, and dirs under  
`vlm_hallucination/experiment_artifacts/vti/*/textual_v2/demosv2_*_all_nd200_s42_r2_prefix/`.
