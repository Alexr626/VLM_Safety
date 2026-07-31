# S2 Failed — what the describe shows + how to recover

## What happened
`pd-s2-stage` is **Failed**. Describe shows **7 pods all Phase=Error** on `gpu-node005` (crash/restart loop ~21:48–22:24Z on 2026-07-16). First pod lived ~21 minutes, then rapid retries.

RunAI log streaming still fails (`could not locate desired workload`) — common once pods are gone. **Do not trust “no logs” as “nothing ran.”**

## Get a postmortem from NFS (no dependency on runai logs)

**A. WinSCP** look under `/airl-datalake/romanus/`:
- `vlm_hallucination/data/amber/` — zip / `images/` count
- `vlm_hallucination/data/coco/train2014.zip` size (partial download?)
- `vlm_hallucination/experiment_artifacts/vti/.../textual_v2/`
- `logs/` — empty until next S2 with log tee

**B. Submit inventory job** (after committing `run_perception_s2_check.sh` + new tarball, or paste inline):

```bash
runai training delete pd-s2-stage -p nlm-mh   # optional cleanup

runai training submit pd-s2-check -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; R=/home/datalake/romanus/vlm_hallucination; L=/home/datalake/romanus/logs; mkdir -p "$L"; { echo AMBER=$(find $R/data/amber/images -name "*.jpg" 2>/dev/null|wc -l); echo demos=$([ -f $R/data/vti/demos_v2.jsonl ] && echo ok || echo MISSING); echo aug=$([ -f $R/diagnostic_experiments/perception_diag/augment/outputs/augmented_amber25.jsonl ] && echo ok || echo MISSING); ls -la $R/data/amber 2>&1 | head; ls -la $R/experiment_artifacts/vti/*/textual_v2 2>&1 | head -40; ls -lh $R/data/coco/train2014.zip 2>&1; df -h /home/datalake; } | tee $L/perception_s2_check.txt'
```

Then WinSCP: `/airl-datalake/romanus/logs/perception_s2_check.txt`

## Next S2 run (logs on NFS)
`run_perception_s2_stage.sh` now tees to `/home/datalake/romanus/logs/perception_s2_stage.log`. Re-commit, rebuild `~/dev/vti_repo.tar.gz`, upload, delete old job, resubmit `pd-s2-stage`. After any failure, read that log file via WinSCP.

Also try once after `runai login`:
```bash
runai training standard logs pd-s2-stage -p nlm-mh --pod=pd-s2-stage-msnht --previous --tail=200
```
(last Error pod from describe).
