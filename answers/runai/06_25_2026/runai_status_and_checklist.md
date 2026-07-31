# RunAI status reminder and status-check checklist

**Date:** 2026-06-25  
**Context:** Romanus resuming RunAI end-to-end experiments after initial bring-up (2026-06-18).

---

## Where we left off (2026-06-18 session)

### Completed

| Step | Status | Evidence |
|------|--------|----------|
| `runai` CLI on lambdab2 | Working | `~/.runai/bin/runai`, project `nlm-mh` |
| NFS layout at `/airl-datalake/romanus/` | Working | Chun-Nam's directory; tarball + extracted repo |
| Pre-upload `bin/micromamba` to NFS | Done | probe-mm showed micromamba 2.8.1 |
| One-time setup job **`setup-vlm4`** | **Succeeded** | Logs ended with `=== SETUP COMPLETE ===` |
| micromamba env on NFS | Built | `envs/vlm_hal/` — torch 2.10.0+cu128, transformers 4.50.1 |
| COCO val2014 + annotations | Downloaded | POPE/CHAIR data on NFS |
| LLaVA-1.5-7B weights | Downloaded | `hf_cache/` on NFS |

### Not confirmed in logs / RESEARCH_LOG

| Step | Status |
|------|--------|
| VTI eval job **`hal-vti`** | Unknown — was the next step after setup; no run record found |
| End-to-end eval results on NFS | Check WinSCP / NFS for `evaluation/results/.../metric_summary.json` |

### Key lessons (do not repeat)

1. **Never use `python3` in job `--command`** — `dspy_image2:0.1` has no Python. Use `bash .../setup_vlm.sh` or `bash .../run_vti.sh`.
2. **Always** `--gpu-devices-request 1 --node-pools h100-pool`.
3. **Export PATH** in every `bash -c` command.
4. Code on NFS comes from **`vti_repo.tar.gz`** (built with `git archive` on lambdab2), not WinSCP of individual scripts.
5. **Pre-upload `bin/micromamba`** — container cannot bootstrap without Python.

---

## How to check where you stand (run on lambdab2)

### 1. Re-authenticate (token expires)

```bash
runai login          # open URL in browser (ThinkPad OK)
runai whoami
runai project set nlm-mh
```

As of 2026-06-25, `runai whoami` failed with **token expired** — login required before any cluster commands work.

### 2. List recent jobs

```bash
runai workload list -p nlm-mh
```

Look for:

- `setup-vlm4` → should be **Completed**
- `hal-vti` (or variants) → Completed / Failed / never submitted

### 3. Inspect a specific job

```bash
runai training describe setup-vlm4 -p nlm-mh
runai training logs setup-vlm4 -p nlm-mh | tail -30
runai training logs hal-vti -p nlm-mh | tail -50   # if submitted
```

### 4. Probe NFS from inside a pod (optional)

```bash
runai training submit probe-nfs -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/dspy_image2:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; ls -la /home/datalake/romanus/; ls -la /home/datalake/romanus/envs/vlm_hal/bin/python 2>&1; ls /home/datalake/romanus/vlm_hallucination/evaluation/results 2>&1'
```

### 5. Check NFS via WinSCP (no shell on storage)

Path: `/airl-datalake/romanus/`

Expected after successful setup:

```
vti_repo.tar.gz
bin/micromamba
envs/vlm_hal/
hf_cache/
mamba/
vlm_hallucination/
```

After a successful eval, also look under:

```
vlm_hallucination/evaluation/results/llava-1.5-7b-hf/pope/<intervention>/metric_summary.json
```

### 6. Check whether lambdab2 code is newer than NFS tarball

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git log -1 --oneline
# Compare to extract date in NFS vlm_hallucination/ or re-archive + re-upload if code changed
```

---

## Decision tree

```
runai login OK?
  NO → runai login, then continue
  YES → setup-vlm4 Completed?
          NO → re-run setup (bash command, new job name)
          YES → envs/vlm_hal exists on NFS?
                  NO → re-run setup (skip rm -rf if partial)
                  YES → code on NFS stale vs lambdab2?
                          YES → git archive → upload vti_repo.tar.gz
                          NO → ready for eval jobs
```

---

## Next actions for end-to-end test

1. `runai login`
2. Confirm `setup-vlm4` completed; probe NFS if unsure
3. If lambdab2 code changed since 2026-06-18: rebuild tarball, upload to NFS
4. Submit eval job (smallest first):

```bash
runai training submit hal-vti -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/dspy_image2:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; bash "$REPO/helper_scripts/runai/run_vti.sh"'
```

5. Monitor: `runai training logs hal-vti -p nlm-mh`
6. Pull results from NFS via WinSCP

For a **smaller** smoke test, add a dedicated RunAI helper script with `LIMIT=10` (default in `run_vti_pope_llava.sh` is 200).

---

## Reference docs in repo

- `helper_scripts/runai/HANDOFF.md` — full bring-up history (stale at "setup not succeeded"; reality: setup-vlm4 succeeded)
- `readme.md` — RunAI section with submit commands
- `evaluation/run_scripts/run_beta_grid_runai.sh` — larger beta-grid driver for RunAI (if extended)
