# Checking free H100 GPUs on RunAI

**Date:** 2026-07-15  
**Project:** `nlm-mh`  
**Preferred pool:** `h100-pool` (NVIDIA H100 80GB HBM3)  
**Submit host:** lambdab2 (`~/.runai/bin/runai`)

## 1. Auth + project context

```bash
export KUBECONFIG=~/.kube/config   # if not already in ~/.bashrc
runai login
runai project set nlm-mh
runai whoami
```

As of 2026-07-15 morning, the RunAI token on lambdab2 was **expired** — `runai login` is required before any list commands work.

## 2. Project GPU quota (your allocatable budget)

```bash
runai project list
```

Look at allocated vs. quota for `nlm-mh` (typically **2 GPUs**). If both quota slots are in use, new jobs wait even if the cluster has free H100s.

## 3. H100 node pool + nodes (cluster free capacity)

```bash
runai nodepool list
runai node list
```

Prefer JSON if the table columns are sparse:

```bash
runai nodepool list --json
runai node list --json
```

Filter for the H100 pool / free GPUs after listing. Jobs for this repo should use `--node-pools h100-pool --gpu-devices-request 1`.

## 4. What is already running in your project

```bash
runai workload list -p nlm-mh
runai workload list -p nlm-mh --status Running
runai workload list -p nlm-mh --status Pending
```

Pending often means either **project quota full** or **no free GPU in `h100-pool`**. Inspect a stuck job with:

```bash
runai training describe <job-name> -p nlm-mh
```

## 5. Practical decision rule

| Check | Meaning |
|-------|---------|
| `project list` shows free quota on `nlm-mh` | You can submit without waiting on *your* limit |
| `node` / `nodepool` shows free GPUs in `h100-pool` | Scheduler can place an H100 job |
| Both free | Safe to submit |
| Quota free, pool full | Job will Pending until an H100 frees up |
| Pool free, quota full | Stop/finish another `nlm-mh` job first |

## Note

There is no separate “pick this free GPU” step — RunAI schedules into `h100-pool`. Availability is inferred from **project quota** + **node/nodepool free GPUs** + **running/pending workloads**.
