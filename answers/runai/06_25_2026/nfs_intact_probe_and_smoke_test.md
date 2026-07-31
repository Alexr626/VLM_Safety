# NFS “intact”, probe job checks, and smallest RunAI smoke test

**Date:** 2026-06-25

---

## What “NFS intact” means

For RunAI experiments, **intact** means the persistent artifacts from `setup-vlm4` are still present and usable — not that every file is unchanged, but that the **bench is ready** for eval jobs without re-running setup.

| Check | Intact if… |
|-------|------------|
| `bin/micromamba` | Exists, executable, prints a version (e.g. `2.8.1`) |
| `envs/vlm_hal/` | Directory exists; Python can import torch + transformers |
| `hf_cache/` | LLaVA-1.5-7B weights present (setup downloaded them) |
| `vti_repo.tar.gz` | Present (source for code re-extract) |
| `vlm_hallucination/` | Extracted repo tree (may be stale vs lambdab2 — OK for infra probe) |
| COCO val2014 | Under `vlm_hallucination/data/` (needed for POPE images) |

**Not required for “intact”:** fresh extracted code matching latest lambdab2 git, eval results from prior runs, or `train2014` (only needed for VTI direction *extraction*, not cached-direction eval).

**Broken / needs re-setup:** missing `envs/vlm_hal`, missing `micromamba`, missing `hf_cache` weights, or env import fails.

---

## How to check whether `probe-nfs` ran successfully

### 1. Job status

```bash
runai workload list -p nlm-mh
```

| Status | Meaning |
|--------|---------|
| **Completed** | Pod ran and exited 0 |
| **Failed** | Check logs |
| **Initializing** / **Running** | Still queued or running (GPU pool may be busy) |

As of 2026-06-25, `probe-nfs` was **Initializing** while `qwen3-30b3` held a GPU — probes wait like any other job.

### 2. Logs (after Completed)

```bash
runai training logs probe-nfs -p nlm-mh
```

Success looks like:
- `ls -la /home/datalake/romanus/` listing `bin/`, `envs/`, `hf_cache/`, `vti_repo.tar.gz`, `vlm_hallucination/`
- `micromamba --version` printing e.g. `2.8.1`
- No `ERROR` / traceback

If logs say `workload is not ready to stream: pod is not ready`, wait and retry.

### 3. Describe (optional)

```bash
runai training describe probe-nfs -p nlm-mh
```

Check **Phase: Completed** and pod phases not all `Error`.

---

## Smallest end-to-end eval: `run_smoke_pope.sh`

Added: `helper_scripts/runai/run_smoke_pope.sh`

- **5** POPE samples (`LIMIT=5`, overridable)
- **1** intervention: `no_intervention` (no VTI direction work)
- **1** split: `random`
- **1** model: `llava-hf/llava-1.5-7b-hf` (already on NFS from setup)
- Output: `evaluation/results/runai_smoke/llava-1.5-7b-hf/pope/no_intervention/`

Runtime: minutes, not hours.

### Deploy script to NFS

Script must be inside the tarball (do not WinSCP `.sh` to NFS root):

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git archive --format=tar.gz -o ~/vti_repo.tar.gz HEAD
# WinSCP ~/vti_repo.tar.gz → /airl-datalake/romanus/
```

Eval jobs use the **extracted** tree. Either:
- Re-run setup extract step only, or
- Submit smoke job with inline extract (command below includes `tar -xzf`).

### Submit smoke job

```bash
runai training submit smoke-pope -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/dspy_image2:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; tar -xzf "$BASE/vti_repo.tar.gz" -C "$REPO"; bash "$REPO/helper_scripts/runai/run_smoke_pope.sh"'
```

Note: `tar -xzf ... -C "$REPO"` overlays files into existing extract — refreshes helpers without wiping env on NFS.

### Monitor

```bash
runai training logs smoke-pope -p nlm-mh
```

Success: `=== Smoke test complete ===` and `metric_summary.json` printed in logs.

---

## View and extract results to lambdab2

### On NFS (WinSCP)

```
/airl-datalake/romanus/vlm_hallucination/evaluation/results/runai_smoke/llava-1.5-7b-hf/pope/no_intervention/
  metric_summary.json
  responses.json   (or similar per harness)
```

### Copy to lambdab2

**Via ThinkPad:** NFS → laptop → lambdab2 `/home/romanus/` or into repo `evaluation/results/runai_smoke/`.

**Direct (if SFTP from lambdab2 works):**

```bash
mkdir -p ~/dev/vlm_hallucination_mitigation_summer_2026/evaluation/results/runai_smoke
sftp air-datalake@gpustorage-1.cloud.bell-labs.com <<'EOF'
cd /airl-datalake/romanus/vlm_hallucination/evaluation/results/runai_smoke
get -r .
bye
EOF
```

### Compare to lambdab2 local run

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
conda activate vlm_hallucination_mitigation
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --benchmarks pope --pope_split random \
  --interventions no_intervention --limit 5 \
  --output_dir evaluation/results/runai_smoke_local
```

Compare `metric_summary.json` fields (accuracy, etc.) — should be similar on same pinned subset (not necessarily identical if sampling differs).

---

## Workflow rationale (Romanus)

- **lambdab2:** small-N raw outputs, new intervention/diagnostic iteration
- **RunAI:** large benchmark sweeps (many β, models, CHAIR/AMBER/POPE at scale)
- AMBER/CHAIR re-runs on Qwen2.5 + LLaVA suggest POPE gains may reflect agreeability/length rather than visual reasoning — motivates using RunAI for expensive confirmation runs, not lambdab2 multi-day jobs.
