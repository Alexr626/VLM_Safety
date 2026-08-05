# RunAI: LLaVA + Qwen steering on **one H100** (lambdab2 submit)

Project: `nlm-mh`. Always: `--gpu-devices-request 1 --node-pools h100-pool`.

**Important:** keep each `python3 …/run_bash_lf.py …/script.sh` on **one line**
inside `bash -c` (no `\` line break). A break makes `run_bash_lf.py` run with no
argv (`IndexError`) and bash then tries to execute the script path as a command.

## Plan

**One** training job requests **one** GPU. Inside the pod, three processes share it:

| Process | Work |
|---------|------|
| LLaVA | CHAIR → POPE (sequential in that process) |
| Qwen | CHAIR only |
| Qwen | POPE only |

Markers: `VERIFY_SYNC_OK`, `SYNC_OK`, `SMOKE_OK`, `TRIPLE_OK` / `TRIPLE_FAIL`.
Logs: `/airl-datalake/romanus/logs/runai/`.

```bash
export PATH="$HOME/.runai/bin:$PATH"
IMG=blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1
NFS='path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite'
```

## 0. SCP to `/airl-datalake/romanus/`

From `/tmp/runai_steering_2026-07-31/`:

- `vti_repo_working_tree.tar.gz`
- `llava_meandiff_directions.tar.gz`
- `qwen_meandiff_directions.tar.gz`
- `llava_partial_results_chair.tar.gz` (if present)

## 1. Sync (puts Qwen smoke + triple launcher on NFS)

```bash
runai training delete sync-steering -p nlm-mh 2>/dev/null || true
runai training submit sync-steering -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; TAR=$BASE/vti_repo_working_tree.tar.gz; [[ -f $TAR ]] || TAR=$BASE/vti_repo.tar.gz; mkdir -p $REPO; tar -xzf $TAR -C $REPO; python3 $REPO/helper_scripts/runai/run_bash_lf.py $REPO/helper_scripts/runai/sync_steering_llava.sh'
```

Wait for `VERIFY_SYNC_OK` and `SYNC_OK`.

## 2. Qwen smoke

```bash
runai training delete steer-qwen-smoke -p nlm-mh 2>/dev/null || true
runai training submit steer-qwen-smoke -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; python3 /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_bash_lf.py /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_steering_qwen_smoke.sh'
```

Expect `SMOKE_OK`.

## 3. Triple grid on one H100

```bash
runai training delete hal-steer-triple -p nlm-mh 2>/dev/null || true
runai training submit hal-steer-triple -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; python3 /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_bash_lf.py /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_steering_triple_one_h100.sh'
```

Expect `TRIPLE_OK`.

## Recovery: missing chair data on NFS

`data/chair/pinned_chair_500.json` and `data/chair/combined.json` are under
`data/*` in `.gitignore`, so `git archive` omits them. The pack overlays both.
Sync verify must print `OK` for those paths (and for COCO val2014 /
`instances_val2014.json` from setup). Then resubmit `hal-steer-triple`.
