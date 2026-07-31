# “Pod is not ready” vs GPU availability (nlm-mh)

**Date:** 2026-06-25

## Does “pod is not ready” mean no free GPUs?

**Not necessarily.** It is a **generic** RunAI/Kubernetes message: the workload exists, but the container is not yet running (or cannot start), so **logs are not streamed**.

Common causes:

| Cause | What you see |
|-------|----------------|
| **Queued for GPU** | Status `Initializing`, **GPU Alloc. 0.00**, no pod scheduled yet |
| **GPU allocated, container starting** | GPU Alloc. > 0, pod `Pending` / pulling image — logs may still say “not ready” briefly |
| **Container failed to start** | Status stays `Initializing` or `Failed`; `describe` shows `CreateContainerError`, mount errors, etc. |
| **Job running** | Status `Running` — logs should work |

### Your `probe-nfs` case (2026-06-25)

**Not a GPU queue issue.** The scheduler already assigned a GPU:

```
GPU Compute  Requested 1.00   Allocated 1.00
```

But the pod failed to start:

```
CreateContainerError: mount callback failed ... no users found
```

This is the **NFS mount / container setup** error (same family as early `setup-vlm` failures in HANDOFF.md), not “waiting for a free GPU.”

Check with:

```bash
runai training describe probe-nfs -p nlm-mh
```

Look at **Pods → Phase** and **Events** (not just `workload list` status).

---

## How to check GPU availability for `nlm-mh`

### 1. Project quota and utilization (best summary)

```bash
runai project list
```

Example output (2026-06-25):

```
 Name     GPU Quota   Allocated GPUs   GPU Allocation Ratio
 nlm-mh   2.00        2.00             100.00%
```

Interpretation:
- **GPU Quota** — max GPUs your project can use at once
- **Allocated GPUs** — currently reserved by running/initializing workloads
- **100%** — no spare quota; new jobs may **wait**, **preempt** (if preemptible), or fail depending on priority

### 2. Who is using GPUs

```bash
runai workload list -p nlm-mh
```

Focus on **GPU Alloc.** and **Status**:

| Workload | Status | GPU Alloc. | Notes |
|----------|--------|------------|-------|
| `qwen3-30b3` | Running | 1.00 | Non-preemptible inference |
| `probe-nfs` | Initializing | 1.00 | Holds quota but pod mount-failing |

Workloads with **0.00** GPU are not using project quota.

### 3. Per-job detail

```bash
runai training describe <job-name> -p nlm-mh
```

**Compute Resources → Allocated** shows whether a GPU was granted. **Events** show scheduling vs mount vs image errors.

### 4. Node pool / cluster capacity

```bash
runai nodepool list   # may require elevated permissions
runai node list       # may require elevated permissions
```

On lambdab2 these returned **insufficient permissions** for a normal researcher account. Use `project list` + `workload list` instead, or ask Chun-Nam / cluster admin for pool-level capacity.

---

## Practical guidance

1. **`pod is not ready` when fetching logs** → run `describe` first; do not assume GPU starvation.
2. **Quota full (2/2)** → wait for jobs to finish, cancel unused workloads, or use **preemptible** jobs (your probes are preemptible; `qwen3-30b3` is not).
3. **`mount callback failed ... no users found`** → infrastructure/NFS mount issue on that node; retry with new job name, same `h100-pool` + 1 GPU flags that worked for `setup-vlm4`. Escalate to manager if persistent.
4. For `probe-nfs` specifically: NFS was already validated by `probe-mm` and `setup-vlm4`; you can proceed to `smoke-pope` once a GPU slot frees up, or delete/retry the stuck probe.
