# How to check free GPUs on RunAI (`nlm-mh`)

Date: 2026-08-06  
Source: [Steering vector eval watcher](46a16eec-8ea2-4f73-9687-d19256b09419) clarification (deserved vs allocated).

## Commands

```bash
export PATH="$HOME/.runai/bin:$PATH"
runai whoami
runai project set nlm-mh
runai workload list -p nlm-mh
runai workload list -p nlm-mh --status Running
runai project list --json   # nlm-mh: resources.deserved vs status.allocated
```

`runai node list` / `nodepool list` are usually Forbidden for this account.

## What the numbers mean

| Field | Meaning |
|-------|---------|
| **deserved** | Project quota / fair-share ceiling (e.g. 4) — not “idle free GPUs” |
| **allocated** | GPUs currently charged to `nlm-mh` |
| **Running (visible)** | Workloads shown in `workload list` for the project |

**Headroom ≈ deserved − allocated**, not `4 − (rows in Running list)`.

## Why Running can look emptier than allocated

Allocated can be higher than the Running rows you see: other users’ jobs on the project, other workload types, filters, or stale accounting. The scheduler cares about **allocated vs deserved**. You generally cannot see other teams’ cluster-wide usage; you only see your project’s charge.

## How to interpret for this job

The AMBER Qwen β-triple needs **1 GPU** (three processes share one H100). If `deserved − allocated ≥ 1`, try submit. If the job sits **Pending**, quota/placement is the constraint — wait or free an allocated GPU — not proof that the cluster has only one GPU.

Practical test: submit; **Running** = you got one; **Pending** = wait.
